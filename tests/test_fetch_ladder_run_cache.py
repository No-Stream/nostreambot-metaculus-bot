"""Process-run cache behavior shared by both fetch-ladder callers."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from metaculus_bot.research.agentic import ladder_adapter
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.document_text import PdfText
from metaculus_bot.research.fetch_ladder import classify, guard, run_cache
from metaculus_bot.research.fetch_ladder.context import LadderContext
from metaculus_bot.research.fetch_ladder.ladder import fetch_url
from metaculus_bot.research.fetch_ladder.policy import GAP_FILL_FETCH_POLICY, RESOLUTION_SOURCE_POLICY, LadderPolicy
from metaculus_bot.research.fetch_ladder.verdict import BodyRoute, DocumentVerdict, HtmlVerdict, PageExtraction
from metaculus_bot.research.resolution_fetch_result import FetchResult, FetchRoute
from tests.resolution_source_fakes import FakeResponse, FakeSession

_URL = "https://example.com/report"
_FINAL_URL = "https://example.com/final/report"


@pytest.fixture(autouse=True)
def _public_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    async def public(_url: str) -> bool:
        return True

    monkeypatch.setattr(guard, "is_public_http_url", public)


@pytest.mark.asyncio
async def test_resolution_read_is_replayed_for_gap_fill_with_full_text_links_and_cache_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_text = "opening\n" + "x" * 8_200 + "\ntail value 917"

    def extraction(*_args: object, **_kwargs: object) -> PageExtraction:
        return PageExtraction(text=full_text)

    monkeypatch.setattr(classify, "_extract_page_text", extraction)
    body = b'<html><body><a href="/source">source</a><p>report</p></body></html>'
    session = FakeSession({_URL: FakeResponse(200, body=body)})

    first = await fetch_url(
        _URL,
        policy=RESOLUTION_SOURCE_POLICY,
        ctx=LadderContext(query="first ask", session=session, host_sems={}),
    )
    second = await agentic_tools.fetch(
        _URL,
        start_char=8_000,
        question_topic="second ask",
    )

    assert len(session.requested) == 1
    assert len(first.text) <= 6_000
    assert "tail value 917" not in first.text
    assert "tail value 917" in second.content_markdown
    assert second.links == ["https://example.com/source"]
    assert second.method == "cache"


@pytest.mark.asyncio
async def test_cache_hit_reapplies_the_current_verdict_and_presentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ChangedVerdict:
        def unread_route(self, _content_type: str) -> None:
            return None

        def body_route(self, _content_type: str, _body: bytes, _source_url: str | None = None) -> str:
            return "html"

        def html(self, extraction: PageExtraction, *, chart_block: str, unreadable_embeds: list[str]) -> HtmlVerdict:
            del extraction, chart_block, unreadable_embeds
            return HtmlVerdict(status="success", status_reason=None, published_text="rejudged for the new caller")

        def document(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("HTML replay must not enter the document verdict")

    original = "original " * 100
    monkeypatch.setattr(classify, "_extract_page_text", lambda *_args, **_kwargs: PageExtraction(text=original))
    session = FakeSession({_URL: FakeResponse(200, body=b"<html><p>original</p></html>")})
    await fetch_url(_URL, policy=RESOLUTION_SOURCE_POLICY, ctx=LadderContext(session=session, host_sems={}))

    changed_policy = replace(GAP_FILL_FETCH_POLICY, verdict=ChangedVerdict(), thin_content_escalation_chars=None)  # type: ignore[arg-type]
    replayed = await fetch_url(_URL, policy=changed_policy, ctx=LadderContext(session=session, host_sems={}))

    assert len(session.requested) == 1
    assert replayed.text == "rejudged for the new caller"
    assert replayed.cache_hit is True
    assert replayed.rung_attempts == []


@pytest.mark.asyncio
async def test_redirected_success_is_cached_under_the_requested_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_report = "final report " * 100
    monkeypatch.setattr(classify, "_extract_page_text", lambda *_args, **_kwargs: PageExtraction(text=final_report))
    session = FakeSession(
        {
            _URL: FakeResponse(302, headers={"Location": _FINAL_URL}),
            _FINAL_URL: FakeResponse(200, body=b"<html><p>final report</p></html>"),
        }
    )

    no_thin_escalation = replace(GAP_FILL_FETCH_POLICY, thin_content_escalation_chars=None)
    first = await fetch_url(_URL, policy=no_thin_escalation, ctx=LadderContext(session=session, host_sems={}))
    second = await fetch_url(_URL, policy=no_thin_escalation, ctx=LadderContext(session=session, host_sems={}))

    assert session.requested == [_URL, _FINAL_URL]
    assert first.url == second.url == _FINAL_URL
    assert ladder_adapter.as_plain_result(second, requested_url=_URL).method == "cache"


@pytest.mark.asyncio
async def test_incompatible_body_route_refetches_instead_of_reusing_wrong_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"<html><body><p>mislabeled report</p></body></html>"
    monkeypatch.setattr(
        classify,
        "_extract_page_text",
        lambda *_args, **_kwargs: PageExtraction(text="mislabeled report " * 40),
    )
    session = FakeSession({_URL: [FakeResponse(200, body=body, content_type="application/octet-stream")] * 2})

    first = await fetch_url(
        _URL,
        policy=replace(GAP_FILL_FETCH_POLICY, thin_content_escalation_chars=None),
        ctx=LadderContext(session=session, host_sems={}),
    )
    second = await fetch_url(
        _URL,
        policy=replace(RESOLUTION_SOURCE_POLICY, rungs_enabled=frozenset()),
        ctx=LadderContext(session=session, host_sems={}),
    )

    assert first.status == "success"
    assert second.status == "unsupported_type"
    assert session.requested == [_URL, _URL]


@pytest.mark.asyncio
async def test_cached_pdf_reuses_parse_and_applies_new_query_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = PdfText(2, 2, ("Alpha result is 17. " * 20, "Beta result is 29. " * 20), "", ())
    parse_threads: list[int] = []

    def parse_and_read(
        _body: bytes, *, max_seconds: float, pol: LadderPolicy, query: str, source_url: str
    ) -> tuple[PdfText, DocumentVerdict]:
        del max_seconds
        parse_threads.append(threading.get_ident())
        return pdf, pol.verdict.document(pdf, query=query, max_chars=pol.per_url_max_chars, source_url=source_url)

    class RecordingVerdict:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.thread_ids: list[int] = []

        def unread_route(self, content_type: str) -> BodyRoute | None:
            del content_type
            return None

        def body_route(self, content_type: str, body: bytes, _source_url: str | None = None) -> BodyRoute:
            del content_type, body
            return "document"

        def html(self, *_args: object, **_kwargs: object) -> HtmlVerdict:
            raise AssertionError("PDF replay must not enter the HTML verdict")

        def document(self, pdf_read: PdfText, *, query: str, max_chars: int | None, source_url: str) -> DocumentVerdict:
            del pdf_read, max_chars, source_url
            self.queries.append(query)
            self.thread_ids.append(threading.get_ident())
            return DocumentVerdict("success", None, f"selected for {query}")

    monkeypatch.setattr(classify, "_parse_and_read", parse_and_read)
    session = FakeSession({_URL: FakeResponse(200, body=b"%PDF-fake", content_type="application/pdf")})
    first = await fetch_url(
        _URL,
        policy=GAP_FILL_FETCH_POLICY,
        ctx=LadderContext(query="alpha", session=session, host_sems={}),
    )
    recording = RecordingVerdict()
    second = await fetch_url(
        _URL,
        policy=replace(RESOLUTION_SOURCE_POLICY, verdict=recording),  # type: ignore[arg-type]
        ctx=LadderContext(query="beta", session=session, host_sems={}),
    )

    assert first.status == second.status == "success"
    assert second.text == "selected for beta"
    assert len(parse_threads) == 1
    assert recording.queries == ["beta"]
    assert recording.thread_ids != [threading.get_ident()]
    assert session.requested == [_URL]


@pytest.mark.asyncio
async def test_error_and_throttle_interstitial_are_not_cached() -> None:
    throttle = b"Rate limit exceeded. Please try again later."
    session = FakeSession(
        {
            _URL: [
                FakeResponse(500),
                FakeResponse(200, body=throttle, content_type="text/plain"),
                FakeResponse(200, body=b"actual report", content_type="text/plain"),
            ]
        }
    )

    direct_only = replace(GAP_FILL_FETCH_POLICY, rungs_enabled=frozenset())
    failed = await fetch_url(_URL, policy=direct_only, ctx=LadderContext(session=session, host_sems={}))
    interstitial = await fetch_url(_URL, policy=direct_only, ctx=LadderContext(session=session, host_sems={}))
    succeeded = await fetch_url(_URL, policy=direct_only, ctx=LadderContext(session=session, host_sems={}))

    assert failed.status == "error"
    assert interstitial.status == "throttled"
    assert interstitial.text == ""
    assert succeeded.text == "actual report"
    assert session.requested == [_URL, _URL, _URL]


@pytest.mark.asyncio
async def test_known_api_keeps_priority_over_an_existing_cache_entry() -> None:
    session = FakeSession({_URL: FakeResponse(200, body=b"page answer", content_type="text/plain")})
    await fetch_url(_URL, policy=RESOLUTION_SOURCE_POLICY, ctx=LadderContext(session=session, host_sems={}))

    async def known_api(_url: str) -> FetchResult:
        return FetchResult(
            url=_URL,
            status="success",
            text="exact API answer",
            http_status=200,
            content_type="application/json",
        )

    result = await fetch_url(
        _URL,
        policy=replace(GAP_FILL_FETCH_POLICY, known_api=known_api),
        ctx=LadderContext(session=session, host_sems={}),
    )

    assert result.text == "exact API answer"
    assert len(session.requested) == 1


@pytest.mark.asyncio
async def test_cache_presentation_timeout_is_per_url_and_does_not_dial_or_drop_a_sibling() -> None:
    slow_url = "https://example.com/slow"
    fast_url = "https://example.com/fast"
    for url, text in ((slow_url, "slow artifact"), (fast_url, "fast artifact")):
        run_cache.put(
            url,
            run_cache.TextRead(url, text, 200, "text/plain", routing_body=b""),
            route="direct",
        )

    direct_only = replace(GAP_FILL_FETCH_POLICY, thin_content_escalation_chars=None)
    timed_out, succeeded = await asyncio.gather(
        fetch_url(
            slow_url,
            policy=direct_only,
            ctx=LadderContext(started=time.monotonic() - 100),
        ),
        fetch_url(fast_url, policy=direct_only, ctx=LadderContext()),
    )

    assert timed_out.status == "error"
    assert succeeded.status == "success"
    assert succeeded.text == "fast artifact"


@pytest.mark.asyncio
async def test_awaited_read_does_not_touch_an_evicted_or_replaced_entry() -> None:
    started = threading.Event()
    release = threading.Event()

    class SlowRead:
        url = _URL

        def present(self, policy: LadderPolicy, *, query: str, route: FetchRoute, now: datetime) -> FetchResult:
            del policy, query, route, now
            started.set()
            release.wait(timeout=2)
            return FetchResult(_URL, "success", "old", 200, "text/plain")

    run_cache.put(_URL, SlowRead(), route="direct")
    old_task = asyncio.create_task(
        run_cache.get(_URL, policy=GAP_FILL_FETCH_POLICY, query="", now=datetime.now(UTC), budget_s=2)
    )
    assert await asyncio.to_thread(started.wait, 1)
    for index in range(51):
        url = f"https://example.com/{index}"
        run_cache.put(url, run_cache.TextRead(url, "filler", 200, "text/plain"), route="direct")
    run_cache.put(_URL, run_cache.TextRead(_URL, "new", 200, "text/plain"), route="direct")
    release.set()

    old = await old_task
    current = await run_cache.get(_URL, policy=GAP_FILL_FETCH_POLICY, query="", now=datetime.now(UTC), budget_s=2)
    assert old is not None
    assert old.text == "old"
    assert current is not None
    assert current.text == "new"
