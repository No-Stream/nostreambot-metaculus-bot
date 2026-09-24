"""Bounded process-run cache of complete reads, before caller verdict and presentation."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import count
from time import monotonic
from typing import Protocol

from metaculus_bot.constants import LOCAL_SOURCE_CACHE_MAX_BYTES
from metaculus_bot.research import document_cache, document_text, resolution_presentation
from metaculus_bot.research.fetch_ladder.digest import DigestPassages, bm25_digest
from metaculus_bot.research.fetch_ladder.policy import LadderPolicy
from metaculus_bot.research.fetch_ladder.throttle import matched_throttle_phrase
from metaculus_bot.research.fetch_ladder.verdict import HtmlVerdict, PageExtraction
from metaculus_bot.research.http_fetch import DatawrapperChartRef, pdf_parse_semaphore
from metaculus_bot.research.image_leads import ImageLead
from metaculus_bot.research.resolution_body_text import _truncate_with_marker
from metaculus_bot.research.resolution_fetch_result import FetchResult, FetchRoute
from metaculus_bot.research.source_documents import ParsedSource
from metaculus_bot.research.source_presentation import (
    digest_source,
    select_source_sections,
    source_inventory,
    source_text,
)
from metaculus_bot.research.wayback import WaybackSnapshot, snapshot_age_days, wayback_lead

_MAX_ENTRIES = 50


def _html_body_text(extraction: PageExtraction, chart_block: str) -> str:
    """The uncapped text a caller would present before its verdict and URL cap."""
    published = "" if extraction.chrome_metric_withheld else (extraction.text or "").strip()
    return "\n\n".join(part for part in (chart_block, published) if part)


def _throttled_result(
    *, url: str, http_status: int | None, content_type: str | None, route: FetchRoute, candidate: str, phrase: str
) -> FetchResult:
    return FetchResult(
        url=url,
        status="throttled",
        text="",
        http_status=http_status,
        content_type=content_type,
        route=route,
        throttle_phrase=phrase,
        throttle_chars=len(candidate.strip()),
    )


class ReadArtifact(Protocol):
    @property
    def url(self) -> str: ...

    def present(self, policy: LadderPolicy, *, query: str, route: FetchRoute, now: datetime) -> FetchResult | None: ...


@dataclass(frozen=True, slots=True)
class HtmlRead:
    url: str
    http_status: int | None
    content_type: str | None
    extraction: PageExtraction
    chart_block: str
    datawrapper_charts: tuple[DatawrapperChartRef, ...]
    unreadable_embeds: tuple[str, ...]
    links: tuple[str, ...]
    routing_body: bytes
    image_leads: tuple[ImageLead, ...] = ()

    def _prepare(self, policy: LadderPolicy, *, route: FetchRoute) -> tuple[HtmlVerdict, bool] | FetchResult | None:
        if policy.verdict.body_route(self.content_type or "", self.routing_body) != "html":
            return None
        candidate = _html_body_text(self.extraction, self.chart_block)
        phrase = matched_throttle_phrase(candidate)
        if phrase is not None:
            return _throttled_result(
                url=self.url,
                http_status=self.http_status,
                content_type=self.content_type,
                route=route,
                candidate=candidate,
                phrase=phrase,
            )
        read = policy.verdict.html(
            self.extraction, chart_block=self.chart_block, unreadable_embeds=list(self.unreadable_embeds)
        )
        floor = policy.thin_content_escalation_chars
        escalate = bool(
            floor is not None
            and not self.chart_block
            and (read.status != "success" or len(read.published_text.strip()) < floor)
        )
        return read, escalate

    def _result(
        self,
        policy: LadderPolicy,
        *,
        read: HtmlVerdict,
        escalate: bool,
        route: FetchRoute,
        text: str,
        passages: DigestPassages | None = None,
    ) -> FetchResult:
        if passages is None:
            passages_returned = None
            passages_grounded = None
            fallback_used = None
        else:
            passages_returned = passages.passages_returned
            passages_grounded = passages.passages_grounded
            fallback_used = passages.fallback_used
        if read.status == "success" and not text.strip():
            raise RuntimeError(f"successful HTML verdict rendered blank text for {self.url}")
        return FetchResult(
            url=self.url,
            status=read.status,
            text=text if read.status == "success" else "",
            http_status=self.http_status,
            content_type=self.content_type,
            datawrapper_charts=list(self.datawrapper_charts),
            unreadable_embeds=list(self.unreadable_embeds),
            status_reason=read.status_reason,
            route=route,
            chrome_metric_withheld=self.extraction.chrome_metric_withheld,
            precision_rescued=self.extraction.precision_rescued,
            links=list(self.links) if policy.collect_links else [],
            escalate_rendered=escalate,
            passages_returned=passages_returned,
            passages_grounded=passages_grounded,
            fallback_used=fallback_used,
            image_leads=self.image_leads if policy.collect_links else (),
        )

    def _ordinary_result(
        self, policy: LadderPolicy, *, read: HtmlVerdict, escalate: bool, route: FetchRoute
    ) -> FetchResult:
        text = ""
        if read.status == "success":
            text = resolution_presentation._page_text_with_leads(
                read.published_text,
                self.url,
                list(self.unreadable_embeds) if policy.disclose_unreadable_embeds else [],
                self.chart_block,
                cap=policy.per_url_max_chars,
            )
        return self._result(
            policy,
            read=read,
            escalate=escalate,
            route=route,
            text=text,
        )

    def present(self, policy: LadderPolicy, *, query: str, route: FetchRoute, now: datetime) -> FetchResult | None:
        del query, now
        prepared = self._prepare(policy, route=route)
        if prepared is None or isinstance(prepared, FetchResult):
            return prepared
        read, escalate = prepared
        return self._ordinary_result(policy, read=read, escalate=escalate, route=route)

    async def present_html(
        self,
        policy: LadderPolicy,
        *,
        query: str,
        route: FetchRoute,
        now: datetime,
        budget_seconds: float,
    ) -> FetchResult | None:
        del now
        started = monotonic()
        prepared = await asyncio.to_thread(self._prepare, policy, route=route)
        if prepared is None or isinstance(prepared, FetchResult):
            return prepared
        read, escalate = prepared
        ordinary = await asyncio.to_thread(self._ordinary_result, policy, read=read, escalate=escalate, route=route)
        if read.status != "success" or policy.per_url_max_chars is None:
            return ordinary
        if len(read.published_text.strip()) <= policy.per_url_max_chars:
            return ordinary
        digest_budget = budget_seconds - (monotonic() - started)
        if digest_budget <= 0.0:
            return ordinary
        digest = policy.digest or bm25_digest
        passages = await digest(
            read.published_text,
            query,
            budget_seconds=digest_budget,
        )
        if not passages.passages:
            return ordinary
        rendered = await asyncio.to_thread(
            document_text.render_flat_passages, passages.passages, query=query, max_chars=None
        )
        text = await asyncio.to_thread(
            resolution_presentation._page_text_with_leads,
            rendered,
            self.url,
            list(self.unreadable_embeds) if policy.disclose_unreadable_embeds else [],
            self.chart_block,
            cap=policy.per_url_max_chars,
        )
        return self._result(
            policy,
            read=read,
            escalate=escalate,
            route=route,
            text=text,
            passages=passages,
        )


@dataclass(frozen=True, slots=True)
class TextRead:
    url: str
    text: str
    http_status: int | None
    content_type: str | None
    lead: str = ""
    routing_body: bytes = b""

    def present(self, policy: LadderPolicy, *, query: str, route: FetchRoute, now: datetime) -> FetchResult | None:
        del query, now
        if policy.verdict.body_route(self.content_type or "", self.routing_body) != "text":
            return None
        phrase = matched_throttle_phrase(self.text)
        if phrase is not None:
            return _throttled_result(
                url=self.url,
                http_status=self.http_status,
                content_type=self.content_type,
                route=route,
                candidate=self.text,
                phrase=phrase,
            )
        if self.lead:
            body = resolution_presentation._lead_then_capped_body(
                self.lead, self.text, self.url, cap=policy.per_url_max_chars
            )
        elif policy.per_url_max_chars is None:
            body = self.text
        else:
            body = _truncate_with_marker(self.text, policy.per_url_max_chars, self.url)
        floor = policy.thin_content_escalation_chars
        return FetchResult(
            url=self.url,
            status="success",
            text=body,
            http_status=self.http_status,
            content_type=self.content_type,
            route=route,
            escalate_rendered=floor is not None and len(self.text) < floor,
        )


@dataclass(frozen=True, slots=True)
class PdfRead:
    url: str
    http_status: int
    content_type: str | None

    def present(self, policy: LadderPolicy, *, query: str, route: FetchRoute, now: datetime) -> FetchResult | None:
        del now
        if policy.verdict.body_route(self.content_type or "", b"%PDF-") != "document":
            return None
        pdf = document_cache.cached_document(self.url)
        if pdf is None:
            return None
        read = policy.verdict.document(pdf, query=query, max_chars=policy.per_url_max_chars, source_url=self.url)
        return FetchResult(
            url=self.url,
            status=read.status,
            text=read.text,
            http_status=self.http_status,
            content_type=self.content_type,
            status_reason=read.status_reason,
            route=route,
        )


@dataclass(frozen=True, slots=True)
class SourceRead:
    """A complete bounded local parse, presented according to the current caller policy."""

    url: str
    http_status: int
    content_type: str | None
    source: ParsedSource

    @property
    def retained_bytes(self) -> int:
        return self.source.retained_bytes

    def present(self, policy: LadderPolicy, *, query: str, route: FetchRoute, now: datetime) -> FetchResult:
        del now
        if policy.caller == "gap_fill_v2" and self.source.kind != "word":
            text = source_inventory(self.source)
            navigation_only = True
        elif policy.caller == "gap_fill_v2":
            text = source_text(select_source_sections(self.source))
            navigation_only = False
        else:
            if policy.per_url_max_chars is None:
                raise RuntimeError("resolution-source local reads require a presentation cap")
            digest = digest_source(
                self.source,
                query=query,
                source_url=self.url,
                max_chars=policy.per_url_max_chars,
            )
            text = digest.block if digest.passages else ""
            navigation_only = False
        if not text.strip():
            return FetchResult(
                url=self.url,
                status="no_resolving_content",
                text="",
                http_status=self.http_status,
                content_type=self.content_type,
                status_reason="no_matching_passage",
                route=route,
                local_kind=self.source.kind,
            )
        return FetchResult(
            url=self.url,
            status="success",
            text=text,
            http_status=self.http_status,
            content_type=self.content_type,
            route=route,
            local_kind=self.source.kind,
            navigation_only=navigation_only,
        )


@dataclass(frozen=True, slots=True)
class ImageSource:
    url: str
    body: bytes
    content_type: str | None

    @property
    def retained_bytes(self) -> int:
        return len(self.body)


@dataclass(frozen=True, slots=True)
class ImageRead:
    source: ImageSource
    http_status: int

    @property
    def url(self) -> str:
        return self.source.url

    @property
    def retained_bytes(self) -> int:
        return self.source.retained_bytes

    def present(self, policy: LadderPolicy, *, query: str, route: FetchRoute, now: datetime) -> FetchResult:
        del query, now
        if policy.caller != "gap_fill_v2":
            return FetchResult(
                url=self.url,
                status="unsupported_type",
                text="",
                http_status=self.http_status,
                content_type=self.source.content_type,
                status_reason="image_needs_reader",
                route=route,
            )
        return FetchResult(
            url=self.url,
            status="success",
            text=f"Raster image retained for local viewing: {self.url}",
            http_status=self.http_status,
            content_type=self.source.content_type,
            route=route,
            local_kind="image",
            navigation_only=True,
        )


@dataclass(frozen=True, slots=True)
class WaybackRead:
    """A dated archive capture plus the complete read made from its bytes."""

    url: str
    snapshot: WaybackSnapshot
    artifact: ReadArtifact
    live_status: str
    live_http_status: int | None
    live_content_type: str | None
    live_failure_class: str | None
    live_exc: str | None
    live_server: str | None

    def present(self, policy: LadderPolicy, *, query: str, route: FetchRoute, now: datetime) -> FetchResult | None:
        uncapped_policy = replace(policy, per_url_max_chars=None)
        snapshot_read = self.artifact.present(uncapped_policy, query=query, route=route, now=now)
        if snapshot_read is None:
            return None
        if snapshot_read.status != "success":
            return replace(snapshot_read, url=self.url, route="wayback")
        age_days = snapshot_age_days(self.snapshot, now)
        max_age_days = policy.wayback_max_age_days
        if age_days is None or (max_age_days is not None and age_days > max_age_days):
            return FetchResult(
                url=self.url,
                status="stale_data",
                text="",
                http_status=self.live_http_status,
                content_type=self.live_content_type,
                failure_class=self.live_failure_class,
                exc=self.live_exc,
                server=self.live_server,
                route="wayback",
            )
        lead = wayback_lead(self.snapshot, age_days, self.live_status)
        return replace(
            snapshot_read,
            url=self.url,
            text=resolution_presentation._lead_then_capped_body(
                lead, snapshot_read.text, self.url, cap=policy.per_url_max_chars
            ),
            route="wayback",
        )

    async def present_html(
        self,
        policy: LadderPolicy,
        *,
        query: str,
        route: FetchRoute,
        now: datetime,
        budget_seconds: float,
    ) -> FetchResult | None:
        if isinstance(self.artifact, HtmlRead):
            snapshot_read = await self.artifact.present_html(
                policy,
                query=query,
                route=route,
                now=now,
                budget_seconds=budget_seconds,
            )
        else:
            snapshot_read = await asyncio.to_thread(
                self.artifact.present,
                policy,
                query=query,
                route=route,
                now=now,
            )
        if snapshot_read is None:
            return None
        if snapshot_read.status != "success":
            return replace(snapshot_read, url=self.url, route="wayback")
        age_days = snapshot_age_days(self.snapshot, now)
        max_age_days = policy.wayback_max_age_days
        if age_days is None or (max_age_days is not None and age_days > max_age_days):
            return FetchResult(
                url=self.url,
                status="stale_data",
                text="",
                http_status=self.live_http_status,
                content_type=self.live_content_type,
                failure_class=self.live_failure_class,
                exc=self.live_exc,
                server=self.live_server,
                route="wayback",
            )
        lead = wayback_lead(self.snapshot, age_days, self.live_status)
        text = await asyncio.to_thread(
            resolution_presentation._lead_then_capped_body,
            lead,
            snapshot_read.text,
            self.url,
            cap=policy.per_url_max_chars,
        )
        return replace(snapshot_read, url=self.url, text=text, route="wayback")


@dataclass(slots=True)
class _Entry:
    artifact: ReadArtifact
    route: FetchRoute
    aliases: set[str]
    sequence: int


_CACHE: OrderedDict[str, _Entry] = OrderedDict()
_ENTRIES: OrderedDict[int, _Entry] = OrderedDict()
_SEQUENCES = count()


async def get(url: str, *, policy: LadderPolicy, query: str, now: datetime, budget_s: float) -> FetchResult | None:
    entry = _CACHE.get(url)
    if entry is None:
        return None
    if isinstance(entry.artifact, SourceRead):
        presentation = _present_source(
            entry.artifact, policy, query=query, route=entry.route, now=now, budget_s=budget_s
        )
    elif isinstance(entry.artifact, (HtmlRead, WaybackRead)):
        presentation = entry.artifact.present_html(
            policy,
            query=query,
            route=entry.route,
            now=now,
            budget_seconds=budget_s,
        )
    else:
        presentation = asyncio.to_thread(entry.artifact.present, policy, query=query, route=entry.route, now=now)
    result = await asyncio.wait_for(presentation, timeout=max(0.0, budget_s))
    if result is None:
        if _CACHE.get(url) is entry:
            _CACHE.pop(url)
            entry.aliases.discard(url)
            if not entry.aliases:
                _ENTRIES.pop(entry.sequence, None)
        return None
    if _CACHE.get(url) is entry:
        _CACHE.move_to_end(url)
        _ENTRIES.move_to_end(entry.sequence)
    return replace(result, cache_hit=True)


async def _present_source(
    artifact: SourceRead,
    policy: LadderPolicy,
    *,
    query: str,
    route: FetchRoute,
    now: datetime,
    budget_s: float,
) -> FetchResult:
    """Present cached source text under the parser gate without leaking a slot on cancellation."""
    started = monotonic()
    gate = pdf_parse_semaphore()
    await asyncio.wait_for(gate.acquire(), timeout=max(0.0, budget_s))
    remaining_s = budget_s - (monotonic() - started)
    if remaining_s <= 0.0:
        gate.release()
        raise TimeoutError
    worker = asyncio.create_task(asyncio.to_thread(artifact.present, policy, query=query, route=route, now=now))
    release_gate_here = True
    try:
        try:
            return await asyncio.wait_for(asyncio.shield(worker), timeout=remaining_s)
        except TimeoutError:
            release_gate_here = False
            worker.add_done_callback(lambda task: _release_source_gate(task, gate))
            raise
        except asyncio.CancelledError:
            release_gate_here = False
            worker.add_done_callback(lambda task: _release_source_gate(task, gate))
            raise
    finally:
        if release_gate_here:
            gate.release()


def _release_source_gate(worker: asyncio.Task[FetchResult], gate: asyncio.Semaphore) -> None:
    try:
        worker.exception()
    except asyncio.CancelledError:
        pass
    finally:
        gate.release()


def put(requested_url: str, artifact: ReadArtifact, *, route: FetchRoute) -> None:
    aliases = set(dict.fromkeys((requested_url, artifact.url)))
    entry = _Entry(artifact=artifact, route=route, aliases=aliases, sequence=next(_SEQUENCES))
    _ENTRIES[entry.sequence] = entry
    for key in aliases:
        previous = _CACHE.get(key)
        if previous is not None:
            previous.aliases.discard(key)
            if not previous.aliases:
                _ENTRIES.pop(previous.sequence, None)
        _CACHE[key] = entry
        _CACHE.move_to_end(key)
    _evict_to_bounds()


def _evict_to_bounds() -> None:
    while len(_ENTRIES) > _MAX_ENTRIES or _retained_bytes() > LOCAL_SOURCE_CACHE_MAX_BYTES:
        _, entry = _ENTRIES.popitem(last=False)
        for alias in entry.aliases:
            if _CACHE.get(alias) is entry:
                _CACHE.pop(alias)


def _retained_bytes() -> int:
    return sum(getattr(entry.artifact, "retained_bytes", 0) for entry in _ENTRIES.values())


def local_source_for(url: str) -> ParsedSource | None:
    entry = _CACHE.get(url)
    if entry is None or not isinstance(entry.artifact, SourceRead):
        return None
    _CACHE.move_to_end(url)
    _ENTRIES.move_to_end(entry.sequence)
    return entry.artifact.source


def image_source_for(url: str) -> ImageSource | None:
    entry = _CACHE.get(url)
    if entry is None or not isinstance(entry.artifact, ImageRead):
        return None
    _CACHE.move_to_end(url)
    _ENTRIES.move_to_end(entry.sequence)
    return entry.artifact.source


def with_lead(artifact: ReadArtifact, lead: str, *, url: str) -> ReadArtifact:
    if not isinstance(artifact, TextRead):
        raise TypeError("a provenance lead can only decorate a text read")
    return replace(artifact, url=url, lead=lead)


def clear() -> None:
    _CACHE.clear()
    _ENTRIES.clear()


def cacheable(artifact: ReadArtifact) -> bool:
    """Whether the successful artifact is reusable; throttle interstitials stay retryable."""
    if isinstance(artifact, HtmlRead):
        candidate = _html_body_text(artifact.extraction, artifact.chart_block)
        return matched_throttle_phrase(candidate) is None
    if isinstance(artifact, TextRead):
        return matched_throttle_phrase(artifact.text) is None
    if isinstance(artifact, WaybackRead):
        return cacheable(artifact.artifact)
    return True
