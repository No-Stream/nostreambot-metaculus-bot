"""The one classification path for a fetched body, whichever rung obtained it.

A response arrives here as bytes plus a content type and leaves as a :class:`FetchResult`:
the redirect and non-200 verdicts, the HTML extractor policy with its chrome and JS-wall
floors, the raw text and CSV branches, and the local document read. Sharing it is what makes
a rescued page indistinguishable downstream from a directly fetched one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import aiohttp
import trafilatura

from metaculus_bot.constants import (
    DOCUMENT_TEXT_MAX_PAGES,
    DOCUMENT_TEXT_MAX_SECONDS,
    DOCUMENT_TEXT_PDF_MAX_BYTES,
    RESOLUTION_SOURCE_CONTENT_SHARE_MIN,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
    RESOLUTION_SOURCE_META_REFRESH_MIN_BUDGET_S,
    RESOLUTION_SOURCE_PDF_MIN_BUDGET_S,
    RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S,
)
from metaculus_bot.research import document_cache
from metaculus_bot.research.document_text import PdfText, extract_pdf_text, is_pdf_body
from metaculus_bot.research.fetch_ladder import context, guard, run_cache
from metaculus_bot.research.fetch_ladder.policy import RESOLUTION_SOURCE_POLICY, LadderPolicy
from metaculus_bot.research.fetch_ladder.verdict import (
    _HTML_STRIPPED_TEXT_CONTENT_TYPES,
    BodyRoute,
    DocumentVerdict,
    PageExtraction,
    content_share,
    looks_like_page_chrome,
)
from metaculus_bot.research.http_fetch import (
    REDIRECT_STATUSES,
    decode_text_body,
    extract_datawrapper_charts,
    extract_page_links,
    meta_refresh_target,
    pdf_parse_semaphore,
    read_body_capped,
    rewrite_aria_tables,
    unreadable_data_embed_providers,
)
from metaculus_bot.research.image_leads import extract_image_leads
from metaculus_bot.research.resolution_body_text import strip_html_tags
from metaculus_bot.research.resolution_chart_data import render_inline_chart_data
from metaculus_bot.research.resolution_fetch_result import (
    _NON_OK_FETCH_STATUS,
    PDF_CONTENT_TYPES,
    FetchResult,
    FetchStatus,
    FetchStatusReason,
    LocalKind,
    http_failure_class,
    server_header_token,
    vacuous_body_status,
)
from metaculus_bot.research.source_documents import ParsedSource, SourceReadError, parse_source

logger = logging.getLogger(__name__)


def _extract_main_text(body: bytes | str, url: str, *, favor_precision: bool = False) -> str | None:
    """Trafilatura extraction. Callers wrap in ``await asyncio.to_thread(...)``.

    Takes bytes (the response body, letting trafilatura detect the encoding) or text (a body
    this module already decoded and rewrote — see :func:`_extract_page_text`). Returns None on
    an empty or failed extraction, so callers classify. Why both extractor settings exist and
    what each one alone loses: docs/architecture.md "The HTML extractor policy, and the one classification path".
    """
    try:
        out = trafilatura.extract(
            body,
            url=url,
            include_comments=False,
            include_tables=True,
            output_format="txt",
            favor_precision=favor_precision,
        )
    except (ValueError, TypeError, RuntimeError) as e:
        # Soft-fail: trafilatura raises on truly malformed input, and one page is not the run.
        logger.warning(f"trafilatura extraction failed for {url}: {e}")
        return None
    if not out or not out.strip():
        return None
    return out


async def _resolution_redirect_outcome(resp: Any, current_url: str, content_type: str) -> FetchResult | str:
    """Vet a 3xx hop: the next URL to follow, or a terminal error/blocked result."""
    status = resp.status
    location = resp.headers.get("Location")
    if not location:
        # Malformed redirect — no Location header.
        logger.info(f"resolution_source {urlparse(current_url).netloc}: {status} redirect with no Location header")
        return FetchResult(
            url=current_url,
            status="error",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    return await guard._vetted_hop_target(location, current_url, content_type=content_type, kind="redirect")


def _network_failure_class(exc: BaseException) -> str:
    """Bucket a transport exception for the fetch marker's ``failure_class``.

    Order is load-bearing and each token draws a line the field exists to draw; both receipts,
    including why ``malformed_response`` is not ``decode``, are in docs/architecture.md "The HTML extractor policy, and the one classification path". ``exc`` on the same
    marker line keeps the exact class name for anything this coarse vocabulary lumps together.
    """
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(
        exc, aiohttp.ClientConnectorCertificateError | aiohttp.ClientConnectorSSLError | aiohttp.ClientSSLError
    ):
        return "tls"
    if isinstance(exc, aiohttp.ClientConnectorDNSError):
        return "dns"
    if isinstance(exc, aiohttp.ClientPayloadError):
        return "decode"
    if isinstance(exc, aiohttp.ClientResponseError):
        return "malformed_response"
    return "connection"


def _resolution_status_outcome(
    status: int, current_url: str, content_type: str, *, server: str | None = None
) -> FetchResult | None:
    """Terminal result for a non-200 status, or None when the body should be read."""
    if status == 200:
        return None
    fetch_status = _NON_OK_FETCH_STATUS.get(status, "error")
    return FetchResult(
        url=current_url,
        status=fetch_status,
        text="",
        http_status=status,
        content_type=content_type or None,
        failure_class=http_failure_class(status),
        server=server_header_token(server),
    )


def _extract_page_text(
    html_text: str, body: bytes, url: str, undecodable_ratio: float, *, remaining_wall_s: float | None = None
) -> PageExtraction:
    """The publishable extraction of an HTML body: ARIA tables rewritten first, default
    recall as the primary extractor, precision as the fallback, both scored by line shape.

    All of it is CPU-bound sync work over a body up to the response cap, so it runs in one
    ``asyncio.to_thread`` hop rather than several. ``remaining_wall_s`` is the caller's remaining
    provider wall, under which the skippable precision pass declines; None means unbounded, which
    is what the direct-path tests drive it with. The calibration behind running both settings, the
    reason a skipped pass can only withhold, and why the rewrite is trusted only on a body that
    decoded with no replacement character at all: docs/architecture.md "The HTML extractor policy, and the one classification path".
    """
    started = time.monotonic()
    rewritten = rewrite_aria_tables(html_text) if undecodable_ratio == 0.0 else None
    source = body if rewritten is None else rewritten
    default = _extract_main_text(source, url)
    if (
        default is None
        or looks_like_page_chrome(default)
        or content_share(default) >= RESOLUTION_SOURCE_CONTENT_SHARE_MIN
    ):
        return PageExtraction(text=default)
    if remaining_wall_s is not None:
        wall_left_s = remaining_wall_s - (time.monotonic() - started)
        if wall_left_s < RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S:
            logger.info(
                "resolution_source: skipping the precision re-extraction for %s — %.1fs of wall budget left; "
                "withholding the default text",
                urlparse(url).netloc,
                wall_left_s,
            )
            return PageExtraction(text=default, chrome_metric_withheld=True)
    precision = _extract_main_text(source, url, favor_precision=True)
    if (
        precision is not None
        and not looks_like_page_chrome(precision)
        and content_share(precision) >= RESOLUTION_SOURCE_CONTENT_SHARE_MIN
    ):
        return PageExtraction(text=precision, precision_rescued=True)
    return PageExtraction(text=default, chrome_metric_withheld=True)


async def _meta_refresh_hop(
    html_text: str, current_url: str, ctx: context.LadderContext, *, from_status: FetchStatus, content_type: str
) -> FetchResult | str | None:
    """Follow a ``<meta http-equiv="refresh">`` stub, or None when there is nothing to follow.

    A hop rather than a terminal result on purpose: the target re-enters the same
    classification path (chrome floor, JS-wall floor, chart rung, PDF read) and consumes
    one of ``MAX_REDIRECTS``, so a refresh chain is bounded exactly like a 3xx chain and
    the meta-refresh check itself works on a body only a later hop could obtain.

    Only reached with no readable content, which is what keeps it off the pages that
    already worked: a real page that ALSO carries a refresh tag (some CMSs emit one for
    a canonical URL) is served as-is rather than re-fetched.
    """
    target = meta_refresh_target(html_text)
    if target is None:
        return None
    if (
        ctx.claim_rung_budget("meta_refresh", from_status, current_url, RESOLUTION_SOURCE_META_REFRESH_MIN_BUDGET_S)
        is None
    ):
        return None
    ctx.start_rung("meta_refresh", from_status, current_url)
    logger.info(
        f"resolution_source meta_refresh: {urlparse(current_url).netloc} -> {target} (direct read was {from_status})"
    )
    return await guard._vetted_hop_target(target, current_url, content_type=content_type, kind="meta_refresh")


@dataclass(frozen=True, slots=True)
class _HtmlClassification:
    """One classified HTML body, plus the decoded text the meta-refresh rung still needs.

    ``html_text`` rides along because the two callers want different things from the same
    decode: :func:`_resolution_html_outcome` looks for a refresh stub in it, while the
    rendered rung has already followed every hop a browser follows and only wants the verdict.
    Decoding twice would double the CPU on a body up to the 5 MiB response cap, or a rendered
    DOM up to ``RENDERED_DOM_MAX_CHARS`` (sized to it).
    """

    result: FetchResult
    html_text: str
    artifact: run_cache.HtmlRead


def capture_html_read(ctx: context.LadderContext, classified: _HtmlClassification) -> None:
    if classified.result.status == "success":
        ctx.capture_read(classified.result, classified.artifact)


def _routing_body(body: bytes) -> bytes:
    """Small body probe retaining every byte-level route fact the two verdicts inspect."""
    stripped = body.lstrip()
    if stripped.startswith(b"%PDF-"):
        return b"%PDF-"
    if stripped.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM")):
        return stripped[:8]
    if len(stripped) >= 12 and stripped.startswith(b"RIFF") and stripped[8:12] == b"WEBP":
        return stripped[:12]
    try:
        json.loads(body.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    else:
        return b"{}"
    if b"<html" in body.lower():
        return b"<html"
    return b""


async def _classify_html_body(
    body: bytes,
    current_url: str,
    content_type: str,
    *,
    http_status: int,
    query: str = "",
    remaining_wall_s: float | None = None,
    pol: LadderPolicy = RESOLUTION_SOURCE_POLICY,
) -> _HtmlClassification:
    """Trafilatura extraction plus the inline-chart rung, then the caller's own verdict on it.

    The ONE classification path for an HTML body, whichever rung obtained it and whichever
    caller asked: the direct fetch, a meta-refresh hop, or a headless-Chromium render. That is
    what makes a rescued page indistinguishable from a directly-fetched one downstream — same
    chart read, same ARIA rewrite, same extraction, same links. What the two callers judge
    differently is whether the extraction counts as content, and that is ``pol.verdict``
    (:mod:`verdict`); ``pol`` also decides the per-URL cap, the embed disclosure, link
    collection and the thin-content escalation.

    ``remaining_wall_s`` is the caller's remaining provider wall, handed to
    :func:`_extract_page_text` so its optional second pass can decline; None keeps it unbounded.
    What is load-bearing about the ORDER here — the raw-HTML embed scans, the chart read running
    on every page, and a chart block rescuing a withheld one: docs/architecture.md "The HTML extractor policy, and the one classification path".
    """
    started = time.monotonic()
    # Raw decoded HTML: trafilatura drops iframes and embed scripts at every setting.
    html_text, undecodable_ratio = decode_text_body(body, content_type)
    charts = extract_datawrapper_charts(html_text)
    unreadable_embeds = unreadable_data_embed_providers(html_text)
    extraction = await asyncio.to_thread(
        _extract_page_text, html_text, body, current_url, undecodable_ratio, remaining_wall_s=remaining_wall_s
    )
    # In a thread for the same reason the extraction is: sync CPU work over a capped body.
    chart_block = await asyncio.to_thread(render_inline_chart_data, html_text)
    # Against the DOCUMENT url, which after a client-side redirect is not the URL asked for.
    links = extract_page_links(html_text, current_url)
    image_leads = extract_image_leads(html_text, current_url)
    artifact = run_cache.HtmlRead(
        url=current_url,
        http_status=http_status,
        content_type=content_type or None,
        extraction=extraction,
        chart_block=chart_block,
        datawrapper_charts=tuple(charts),
        unreadable_embeds=tuple(unreadable_embeds),
        links=tuple(links),
        routing_body=_routing_body(body),
        image_leads=image_leads,
    )
    digest_budget = (
        float("inf") if remaining_wall_s is None else max(0.0, remaining_wall_s - (time.monotonic() - started))
    )
    result = await artifact.present_html(
        pol,
        query=query,
        route="direct",
        now=datetime.now(UTC),
        budget_seconds=digest_budget,
    )
    if result is None:
        raise RuntimeError("fresh HTML artifact was rejected by the policy that classified it")
    return _HtmlClassification(
        result=result,
        html_text=html_text,
        artifact=artifact,
    )


@dataclass(frozen=True, slots=True)
class _TextClassification:
    result: FetchResult
    artifact: run_cache.TextRead | None


def _raw_body_outcome(
    body: bytes, current_url: str, content_type: str, *, http_status: int, pol: LadderPolicy
) -> _TextClassification:
    """Classify a structured or plain-text body we already hold: the one copy of the rule.

    Structured bodies include JSON and XML; XML tags are retained because they carry the data's
    labels. Plain text and CSV keep the existing markup stripping rule.

    Reached from :func:`_classify_body`, so a body the impersonated retry read goes through the
    same charset-honouring decode, the same markup strip and the same vacuity refusal as a
    directly fetched one. ``pol.per_url_max_chars`` None leaves the body whole for a caller that
    windows it at presentation instead, and ``pol.thin_content_escalation_chars`` decides whether
    a short or unusable body earns the browser.
    """
    netloc = urlparse(current_url).netloc
    raw, undecodable_ratio = decode_text_body(body, content_type)
    # Text branches only: a JSON body's angle brackets are the data (see the doc).
    if any(ct in content_type for ct in _HTML_STRIPPED_TEXT_CONTENT_TYPES):
        raw = strip_html_tags(raw)
    floor = pol.thin_content_escalation_chars
    vacuous = vacuous_body_status(raw, undecodable_ratio, require_csv_rows=False)
    if vacuous is not None:
        # Reason line, not an outcome line: the marker already carries the status.
        logger.info(
            f"resolution_source {netloc}: 200 body carries no usable content "
            f"({vacuous}, {len(body)} bytes, undecodable={undecodable_ratio:.2f})"
        )
        return _TextClassification(
            result=FetchResult(
                url=current_url,
                status=vacuous,
                text="",
                http_status=http_status,
                content_type=content_type or None,
                # What we hold is replacement characters rather than the page, which a caller that
                # escalates says differently to the driver than a type it does not read at all.
                status_reason="undecodable_body" if vacuous == "unsupported_type" else None,
                escalate_rendered=floor is not None,
            ),
            artifact=None,
        )
    artifact = run_cache.TextRead(current_url, raw, http_status, content_type or None, routing_body=_routing_body(body))
    result = artifact.present(pol, query="", route="direct", now=datetime.now(UTC))
    if result is None:
        raise RuntimeError("fresh text artifact was rejected by the policy that classified it")
    return _TextClassification(result=result, artifact=artifact)


@dataclass(frozen=True, slots=True)
class _PendingDocument:
    """A PDF whose bytes we hold and whose parse has not started yet.

    Exists so the parse happens OUTSIDE the per-host politeness semaphore. That map is
    loop-wide, so a 20 s parse held inside it blocked every other concurrent question's
    fetch of any URL on that host — and this population is concentrated on a handful of
    government hosts, so same-host collisions across questions in one round are the
    expected case. Two questions queued behind one parse of a shared host exhaust their own
    ``RESOLUTION_SOURCE_WALL_TIMEOUT``, and the outer ``wait_for`` then discards every page
    they had already fetched.
    """

    url: str
    body: bytes
    http_status: int
    content_type: str
    from_status: FetchStatus


@dataclass(frozen=True, slots=True)
class _PendingSource:
    url: str
    body: bytes
    http_status: int
    content_type: str
    local_kind: LocalKind


def _parse_and_read(
    body: bytes, *, max_seconds: float, pol: LadderPolicy, query: str, source_url: str
) -> tuple[PdfText, DocumentVerdict]:
    """pypdf parse plus the caller's verdict on it: both CPU-bound, so ONE thread hop, never two.

    The fetcher's verdict runs a BM25 passage selection here, which is as CPU-bound as the parse
    and was once running inline on the event loop two lines below a call carefully threaded for
    exactly that reason (``select_passages`` measured at 96-235 ms per 400-page document,
    additive across the six concurrent questions). The gap-fill verdict instead joins the pages
    and holds the parse for its own later digest. Whichever runs, it runs in this hop.
    """
    pdf = extract_pdf_text(body, max_pages=DOCUMENT_TEXT_MAX_PAGES, max_seconds=max_seconds)
    read = pol.verdict.document(pdf, query=query, max_chars=pol.per_url_max_chars, source_url=source_url)
    return pdf, read


def _document_outcome(
    body: bytes,
    current_url: str,
    content_type: str,
    ctx: context.LadderContext,
    *,
    http_status: int,
    from_status: FetchStatus,
) -> FetchResult | _PendingDocument:
    """Hold a document body we already read, or refuse a body that is not one: the one copy of the rule.

    The bytes-level tail of the document branch: the ``%PDF-`` magic check, the
    :class:`_PendingDocument` construction and the ``pdf_local`` budget gate, split from the read
    so a body the impersonated retry read goes through the same rule as a directly fetched one.
    """
    netloc = urlparse(current_url).netloc
    if not is_pdf_body(body):
        # Declared a document and is not one: the label was never the thing we trusted.
        logger.info(f"resolution_source {netloc}: body is not a document we can read, ct={content_type!r}")
        return FetchResult(
            url=current_url,
            status="unsupported_type",
            text="",
            http_status=http_status,
            content_type=content_type or None,
        )
    pending = _PendingDocument(
        url=current_url,
        body=body,
        http_status=http_status,
        content_type=content_type,
        from_status=from_status,
    )
    # Before the response closes, so a question with no budget never queues for a slot.
    if ctx.claim_rung_budget("pdf_local", from_status, current_url, RESOLUTION_SOURCE_PDF_MIN_BUDGET_S) is None:
        return _document_not_parsed(pending, "budget_skipped")
    return pending


async def _finish_document(pending: _PendingDocument, ctx: context.LadderContext) -> FetchResult:
    """Parse a held PDF and serve what the caller's verdict makes of it, holding no gate.

    Runs after :func:`_fetch_one_hop` has left both the ``session.get`` context and the per-host
    gate, which is the whole point: holding a loop-wide ``Semaphore(1)`` for a host through a
    seconds-long parse stalls every other concurrent question's fetch of that host (see
    :class:`_PendingDocument`). It contends instead for :func:`http_fetch.pdf_parse_semaphore`,
    the 2-slot gate shared with the gap-fill v2 document ladder, on a bounded wait that degrades
    to the same leave-it-unread skip. Never raises: ``extract_pdf_text`` reports a mangled
    document through ``unreadable_reason``, and both verdicts are pure but for the gap-fill one's
    cache write.
    """
    netloc = urlparse(pending.url).netloc
    gate = pdf_parse_semaphore()
    budget_s = ctx.rung_budget_s()
    try:
        # Bounded, not a bare acquire: queueing past the wall costs every sibling page.
        await asyncio.wait_for(gate.acquire(), timeout=max(0.0, budget_s - RESOLUTION_SOURCE_PDF_MIN_BUDGET_S))
    except TimeoutError:
        logger.warning(
            "resolution_source: skipping the local PDF read for %s — no parse slot within %.1fs of wall budget",
            netloc,
            budget_s,
        )
        ctx.skip_rung("pdf_local", pending.from_status, pending.url, "parse_contention")
        return _document_not_parsed(pending, "parse_contention")
    try:
        # Re-read after the wait: `max_seconds` is wall-clock, and the queue spent some.
        budget_s = ctx.claim_rung_budget(
            "pdf_local", pending.from_status, pending.url, RESOLUTION_SOURCE_PDF_MIN_BUDGET_S, note=" after queueing"
        )
        if budget_s is None:
            return _document_not_parsed(pending, "budget_skipped")
        attempt = ctx.start_rung("pdf_local", pending.from_status, pending.url)
        pdf, read = await asyncio.to_thread(
            _parse_and_read,
            pending.body,
            max_seconds=min(DOCUMENT_TEXT_MAX_SECONDS, budget_s),
            pol=ctx.policy,
            query=ctx.query,
            source_url=pending.url,
        )
        # Inside the gate, so `wall_s` is the parse rather than the queue for a slot.
        attempt.wall_s = max(0.0, time.monotonic() - attempt.started_at)
    finally:
        gate.release()
    if read.status == "unreadable_document":
        logger.warning(
            f"resolution_source {netloc}: PDF carried no readable text ({read.status_reason}, "
            f"{pdf.page_count} pages, {pdf.pages_read} read)"
        )
    result = FetchResult(
        url=pending.url,
        status=read.status,
        text=read.text,
        http_status=pending.http_status,
        content_type=pending.content_type or None,
        status_reason=read.status_reason,
    )
    document_cache.cache_document(pending.url, pdf)
    if result.status == "success":
        ctx.capture_read(result, run_cache.PdfRead(pending.url, pending.http_status, pending.content_type or None))
    return result


def _local_kind(content_type: str, body: bytes) -> LocalKind:
    media_type = content_type.partition(";")[0].strip().lower()
    if "spreadsheet" in media_type or "excel" in media_type:
        return "workbook"
    if "wordprocessingml" in media_type or media_type in ("application/msword", "application/vnd.ms-word"):
        return "word"
    del body
    return "archive"


def _local_refusal(pending: _PendingSource) -> FetchResult:
    return FetchResult(
        url=pending.url,
        status="unsupported_type",
        text="",
        http_status=pending.http_status,
        content_type=pending.content_type or None,
        local_kind=pending.local_kind,
        local_read_refused=True,
    )


async def _finish_source(pending: _PendingSource, ctx: context.LadderContext) -> FetchResult:
    """Parse a bounded local source off the event loop under the shared parser gate."""
    gate = pdf_parse_semaphore()
    budget_s = ctx.rung_budget_s()
    if budget_s <= 0.0:
        return _local_refusal(pending)
    try:
        await asyncio.wait_for(gate.acquire(), timeout=budget_s)
    except TimeoutError:
        return _local_refusal(pending)
    parse_budget_s = ctx.rung_budget_s()
    if parse_budget_s <= 0.0:
        gate.release()
        return _local_refusal(pending)
    worker = asyncio.create_task(asyncio.to_thread(_parse_and_present_source, pending, ctx))
    release_gate_here = True
    try:
        try:
            artifact, result = await asyncio.wait_for(asyncio.shield(worker), timeout=parse_budget_s)
        except SourceReadError as exc:
            logger.info("local source read refused for %s: %s", pending.url, exc.reason)
            return _local_refusal(pending)
        except TimeoutError:
            release_gate_here = False
            worker.add_done_callback(lambda task: _release_parser_gate(task, gate, pending.url))
            return _local_refusal(pending)
        except asyncio.CancelledError:
            # `to_thread` keeps running after cancellation. Keep the parser slot occupied until
            # the worker really ends, while allowing the caller's wall timeout to return now.
            release_gate_here = False
            worker.add_done_callback(lambda task: _release_parser_gate(task, gate, pending.url))
            raise
    finally:
        if release_gate_here:
            gate.release()

    # The complete parse is policy neutral even when this caller's query matched nothing.
    ctx.capture_read(result, artifact)
    return result


def _parse_and_present_source(
    pending: _PendingSource, ctx: context.LadderContext
) -> tuple[run_cache.SourceRead, FetchResult]:
    source: ParsedSource = parse_source(pending.body, pending.content_type)
    artifact = run_cache.SourceRead(
        url=pending.url,
        http_status=pending.http_status,
        content_type=pending.content_type or None,
        source=source,
    )
    result = artifact.present(ctx.policy, query=ctx.query, route="direct", now=ctx.now)
    return artifact, result


def _release_parser_gate(
    worker: asyncio.Task[tuple[run_cache.SourceRead, FetchResult]], gate: asyncio.Semaphore, source_url: str
) -> None:
    """Release a parser slot after a cancelled caller's thread actually completes."""
    try:
        worker.exception()
    except asyncio.CancelledError:
        logger.warning("local source parser worker was unexpectedly cancelled for %s", source_url)
    finally:
        gate.release()


def _document_not_parsed(pending: _PendingDocument, reason: FetchStatusReason) -> FetchResult:
    """The result for a document we held and chose not to parse.

    ``unsupported_type`` rather than ``unreadable_document``: nothing read the bytes, so
    nothing established they carry no text, and only the latter is worth a paid document
    read later. ``reason`` says which rule declined — the same token the rung attempt's
    ``skipped_reason`` carries, repeated here because the two ride different markers
    (``RESOLUTION_SOURCE_ESCALATION`` versus ``RESOLUTION_SOURCE_FETCH``) and a reader of
    the per-fetch line should not have to join to learn we were holding a document.
    """
    return FetchResult(
        url=pending.url,
        status="unsupported_type",
        text="",
        http_status=pending.http_status,
        content_type=pending.content_type or None,
        status_reason=reason,
    )


async def _resolution_response_outcome(
    resp: Any, current_url: str, ctx: context.LadderContext
) -> FetchResult | _PendingDocument | _PendingSource | str:
    """Classify one response: a terminal FetchResult, a held document, or the next hop's URL.

    One read, then one routing decision, so a body's cap and its branch cannot disagree. The
    :class:`_PendingDocument` case is the document branch handing its parse back to the caller
    to run outside the host semaphore; every other branch is terminal or a hop.

    The read's byte cap turns on whether the server DECLARED a document: a declared one gets
    ``DOCUMENT_TEXT_PDF_MAX_BYTES`` because the receipt file is 6.7 MB and the page cap would
    refuse exactly the document the local read exists for, while an undeclared body keeps the
    5 MiB page cap, since it is far likelier to be an image or an archive than a document.
    """
    status = resp.status
    content_type = (resp.headers.get("Content-Type") or "").lower()

    if status in REDIRECT_STATUSES:
        return await _resolution_redirect_outcome(resp, current_url, content_type)

    # The `Server` header rides a non-200 so a 403 names the CDN that served it.
    server = resp.headers.get("Server")
    non_ok = _resolution_status_outcome(status, current_url, content_type, server=server)
    if non_ok is not None:
        return non_ok

    pol = ctx.policy
    unread = pol.verdict.unread_route(content_type)
    if unread is not None:
        return _unread_body_outcome(unread, current_url, content_type, http_status=status)
    declared_pdf = any(ct in content_type for ct in PDF_CONTENT_TYPES)
    netloc = urlparse(current_url).netloc
    body = await read_body_capped(
        resp,
        max_bytes=DOCUMENT_TEXT_PDF_MAX_BYTES if declared_pdf else RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
        label=f"{pol.caller} pdf {netloc}" if declared_pdf else f"{pol.caller} {netloc}",
    )
    if body is None:
        return _over_cap_outcome(current_url, content_type, http_status=status, declared_pdf=declared_pdf)
    return await _classify_body_or_hop(body, current_url, content_type, ctx, http_status=status)


async def _classify_body(
    body: bytes, current_url: str, content_type: str, ctx: context.LadderContext, *, http_status: int
) -> FetchResult | _PendingDocument | _PendingSource:
    """Route a 200 body this ladder holds to the branch the caller's verdict gives it.

    The ONE router, so a body the impersonated retry read takes the same branch a directly
    fetched one does. Which branch a body earns is ``policy.verdict.body_route``: the fetcher
    routes on the Content-Type header, with the ``%PDF-`` sniff inside the document branch; the
    gap-fill driver routes on the bytes first, because a mislabeled document is common on the
    hosts it reaches (docs/architecture.md "What a verdict decides"). Terminal by construction:
    the meta-refresh hop belongs to :func:`_classify_body_or_hop`, whose caller owns a redirect
    loop to follow it with.
    """
    route = ctx.policy.verdict.body_route(content_type, body)
    if route == "html":
        classified = await _classify_html_body(
            body,
            current_url,
            content_type,
            http_status=http_status,
            query=ctx.query,
            remaining_wall_s=ctx.rung_budget_s(),
            pol=ctx.policy,
        )
        capture_html_read(ctx, classified)
        return classified.result
    if route == "text":
        classified = _raw_body_outcome(body, current_url, content_type, http_status=http_status, pol=ctx.policy)
        if classified.artifact is not None:
            ctx.capture_read(classified.result, classified.artifact)
        return classified.result
    if route == "source":
        ctx.local_read_receipt.encountered = True
        return _PendingSource(
            url=current_url,
            body=body,
            http_status=http_status,
            content_type=content_type,
            local_kind=_local_kind(content_type, body),
        )
    if route == "image":
        if ctx.policy.caller != "gap_fill_v2":
            return FetchResult(
                url=current_url,
                status="unsupported_type",
                text="",
                http_status=http_status,
                content_type=content_type or None,
                status_reason="image_needs_reader",
                local_read_refused=True,
            )
        artifact = run_cache.ImageRead(
            source=run_cache.ImageSource(url=current_url, body=body, content_type=content_type or None),
            http_status=http_status,
        )
        result = artifact.present(ctx.policy, query=ctx.query, route="direct", now=ctx.now)
        ctx.capture_read(result, artifact)
        return result
    if route == "svg":
        return FetchResult(
            url=current_url,
            status="unsupported_type",
            text="",
            http_status=http_status,
            content_type=content_type or None,
            local_kind="image",
            local_read_refused=True,
        )
    if route == "invalid_image":
        return FetchResult(
            url=current_url,
            status="unsupported_type",
            text="",
            http_status=http_status,
            content_type=content_type or None,
            local_kind="image",
            local_read_refused=True,
        )
    if route == "unsupported":
        return _unread_body_outcome(route, current_url, content_type, http_status=http_status)
    return _document_outcome(
        body, current_url, content_type, ctx, http_status=http_status, from_status="unsupported_type"
    )


async def _classify_body_or_hop(
    body: bytes, current_url: str, content_type: str, ctx: context.LadderContext, *, http_status: int
) -> FetchResult | _PendingDocument | _PendingSource | str:
    """The router, plus the meta-refresh hop only the redirect loop may consume.

    Only once an HTML body carries no content anywhere does the hop run, which is what keeps it
    off the pages that already worked. It comes back as the next URL, so this returns
    ``FetchResult | _PendingDocument | str`` exactly as the redirect dispatcher does, and a
    refresh chain is bounded by the same ``MAX_REDIRECTS`` cap with the same per-hop SSRF
    re-guard. A rung with no redirect loop of its own calls :func:`_classify_body` instead.
    """
    if ctx.policy.verdict.body_route(content_type, body) != "html":
        return await _classify_body(body, current_url, content_type, ctx, http_status=http_status)
    classified = await _classify_html_body(
        body,
        current_url,
        content_type,
        http_status=http_status,
        query=ctx.query,
        remaining_wall_s=ctx.rung_budget_s(),
        pol=ctx.policy,
    )
    capture_html_read(ctx, classified)
    if classified.result.status in ("success", "throttled"):
        return classified.result
    hop = await _meta_refresh_hop(
        classified.html_text, current_url, ctx, from_status=classified.result.status, content_type=content_type
    )
    if hop is not None:
        return hop
    return classified.result


def _unread_body_outcome(route: BodyRoute, current_url: str, content_type: str, *, http_status: int) -> FetchResult:
    """The result for a body no local rung will turn into text: an image, or an unread type.

    Both are ``unsupported_type``; ``image_needs_reader`` is what separates the one shape a paid
    document read could still rescue from a content type this ladder simply does not read.
    """
    logger.info(f"resolution_source {urlparse(current_url).netloc}: not read as text, ct={content_type!r} ({route})")
    return FetchResult(
        url=current_url,
        status="unsupported_type",
        text="",
        http_status=http_status,
        content_type=content_type or None,
        status_reason="image_needs_reader" if route == "image" else None,
    )


def _over_cap_outcome(current_url: str, content_type: str, *, http_status: int, declared_pdf: bool) -> FetchResult:
    """The result for a body past its byte cap; which cap it was decides what a reader is told.

    A declared document past the document cap says so, because it was too big to read locally AND
    too big to be worth having a model retrieve, and a caller that would otherwise pay for those
    bytes keys on that reason.
    """
    return FetchResult(
        url=current_url,
        status="error",
        text="",
        http_status=http_status,
        content_type=content_type or None,
        status_reason="oversize_document" if declared_pdf else None,
    )
