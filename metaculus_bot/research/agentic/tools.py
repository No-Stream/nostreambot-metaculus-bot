"""The four tools the gap-fill v2 driver calls, and this caller's half of the fetch ladder.

The handlers are here (``search_news``, ``search_web``, ``fetch``, ``read_document``) and so is
``build_gap_fill_tools``, whose list order is the order the driver sees. What used to be here and is
not any more is the ladder itself: ``fetch`` and ``read_document``'s free acquisition both run the
SHARED ladder through ``_fetch_via_ladder``, which calls ``fetch_ladder.ladder.fetch_url`` under one
of the three gap-fill presets and maps what comes back through ``ladder_adapter``. The loop's
former duplicate rungs have been removed; the shared ladder is now the only fetch implementation.

What stays this caller's: the window presentation that serves ``start_char`` continuations, the
question-platform refusal that runs before anything is dialed, the throttle-phrase check on a body
the ladder read, the auto-escalation to ``read_document``, the paid reader and its robots pre-check,
and the two shared fetch markers emitted after each tool call. Support pieces live next door:
``tool_descriptions`` (driver-facing text and JSON schemas), ``tool_backends`` (the AskNews, Exa and
Gemini calls), ``fetch_outcomes`` (this ladder's result type and its refusals), ``local_document``
(what the free ladder holds, and the digest ``read_document`` serves), ``ladder_adapter`` (the two
vocabularies' one meeting point).

The seams the suite monkeypatches — ``_fetch_via_ladder``, ``_acquire_local_document``,
``_run_document_read_sync``, ``read_document``, ``_READ_DOCUMENT_TIMEOUT_S`` — are attributes of THIS
module and resolved here at call time. The ladder's own seams (``direct_fetch._fetch_direct``,
``rungs._rendered_rung``, ``rungs.render_page``, ``rungs.fetch_impersonated``) are patched there.
Detail: ``docs/agentic_gap_fill.md`` "The fetch ladder".
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from time import monotonic
from urllib.parse import urlparse

from metaculus_bot.constants import (
    ASKNEWS_BACKOFF_SECS,  # noqa: F401  # re-export: tests read the AskNews retry ladder's constants off this module
    ASKNEWS_CLIENT_ID_ENV,
    ASKNEWS_MAX_TRIES,  # noqa: F401  # re-export: see ASKNEWS_BACKOFF_SECS above
    ASKNEWS_SECRET_ENV,
    DOCUMENT_DIGEST_TOP_K,
    EXA_API_KEY_ENV,
    GOOGLE_API_KEY_ENV,
    RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS,
)
from metaculus_bot.research import document_cache, document_text, fetch_markers, source_presentation
from metaculus_bot.research.agentic import ladder_adapter, local_document
from metaculus_bot.research.agentic.dispatch import tool_content_body_budget
from metaculus_bot.research.agentic.fetch_outcomes import (
    DOCUMENT_NEEDED_METHOD,
    PlainFetchResult,
    _fetch_plain_url_block,
)
from metaculus_bot.research.agentic.image_tools import ImageViewState, view_acquired_image, view_image
from metaculus_bot.research.agentic.tool_backends import (
    _call_asknews_search,
    _call_exa_search,
    _format_asknews_results,
    _format_exa_results,
    _run_document_read_sync,
)
from metaculus_bot.research.agentic.tool_descriptions import (
    _FETCH_PARAMETERS,
    _READ_DOCUMENT_PARAMETERS,
    _SEARCH_NEWS_PARAMETERS,
    _SEARCH_WEB_PARAMETERS,
    _VIEW_IMAGE_PARAMETERS,
    FETCH_DESCRIPTION,
    READ_DOCUMENT_DESCRIPTION,
    SEARCH_NEWS_DESCRIPTION,
    SEARCH_WEB_DESCRIPTION,
    VIEW_IMAGE_DESCRIPTION,
)
from metaculus_bot.research.agentic.types import ToolOutcome, ToolSpec
from metaculus_bot.research.fetch_ladder import run_cache
from metaculus_bot.research.fetch_ladder.context import LadderContext, LocalReadReceipt, QuestionRungBudget
from metaculus_bot.research.fetch_ladder.digest import bm25_digest
from metaculus_bot.research.fetch_ladder.ladder import fetch_url
from metaculus_bot.research.fetch_ladder.policy import (
    GAP_FILL_DIRECT_POLICY,
    GAP_FILL_DOCUMENT_POLICY,
    GAP_FILL_FETCH_POLICY,
    LADDER_CALLER_GAP_FILL_V2,
    LadderPolicy,
)
from metaculus_bot.research.http_fetch import pdf_parse_semaphore
from metaculus_bot.research.image_leads import render_image_leads
from metaculus_bot.research.resolution_fetch_result import FetchResult
from metaculus_bot.research.robots_policy import ROBOTS_FETCH_TIMEOUT_S, google_extended_blocks_url, robots_host
from metaculus_bot.research.source_documents import ParsedSource

logger = logging.getLogger(__name__)

_FETCH_WINDOW_CHARS = 8000
_READ_DOCUMENT_TIMEOUT_S = 60.0
_LOCAL_DOCUMENT_BUDGET_S = GAP_FILL_DOCUMENT_POLICY.total_wall_s
_READ_DOCUMENT_TOTAL_BUDGET_S = 65.0
_FETCH_HOST_SEMAPHORES: dict[str, asyncio.Semaphore] = {}


def _slice_fetch_window(text: str, start_char: int, *, max_chars: int = _FETCH_WINDOW_CHARS) -> tuple[str, bool]:
    start = max(0, start_char)
    if start >= len(text):
        return "", False
    end = min(len(text), start + max_chars)
    if end >= len(text):
        return text[start:end], False
    while True:
        marker = f"\n[truncated at {end} of {len(text)} chars — call again with start_char={end}]"
        adjusted_end = min(len(text), start + max(0, max_chars - len(marker)))
        if adjusted_end == end:
            return text[start:end] + marker, True
        end = adjusted_end


def _format_fetch_error(message: str, *, status: str = "error", method: str = "plain") -> ToolOutcome:
    return ToolOutcome(content_markdown=message, method=method, status=status)


def _render_fetch_outcome(
    url: str,
    text: str,
    links: list[str],
    *,
    method: str,
    start_char: int,
    image_leads: str = "",
) -> ToolOutcome:
    del url
    prototype = ToolOutcome(content_markdown="", links=links, method=method)
    body_budget = tool_content_body_budget("fetch", prototype, _FETCH_WINDOW_CHARS)
    lead_block = f"\n\n{image_leads}" if image_leads and start_char == 0 else ""
    window, truncated = _slice_fetch_window(text, start_char, max_chars=max(0, body_budget - len(lead_block)))
    return ToolOutcome(
        content_markdown=window + lead_block,
        links=links,
        method=method,
        truncated=truncated,
    )


_NO_CONTENT_FETCH_MSG = (
    "No readable content: {url} returned HTTP 200 but produced no extractable text — "
    "neither the plain fetch nor the headless-browser render read anything (JavaScript "
    "wall, consent/anti-bot gate, or a genuinely empty page). Nothing from this URL was "
    "read; do NOT cite it as a fetched source. Try read_document(url, ask) for a targeted "
    "extraction, or find another source."
)


_THROTTLED_FETCH_MSG = (
    "Rate limited: {url} returned HTTP 200, but its body is a short interstitial carrying the "
    'throttle phrase "{phrase}", not the page. Nothing from this URL was read: do NOT cite it as '
    "a fetched source, and do NOT read it as evidence that the fact is unavailable — the page "
    "exists and we were refused for asking too fast. Do other work now (a different host, "
    "another gap) and call fetch on this URL again later in the run; a retry is a real request, "
    "not a replay of this one."
)


def _throttled_fetch_outcome(url: str, text: str, phrase: str, *, method: str, chars: int | None = None) -> ToolOutcome:
    """Outcome for a 200-OK body that is the host's rate-limit interstitial, not the page.

    Mirrors :func:`_empty_fetch_outcome` in both guards — a non-``"ok"`` status AND a method
    absent from ``provenance._METHOD_TO_TIER`` — so an interstitial can never be stamped
    ``fetched`` and supersede the briefing. Deliberately NOT cached, which is the half of
    this fix that q45191 turned on: the interstitial was cached under ``method="rendered"``
    and served straight back when the driver retried the same URL, so its retry could not
    have succeeded however many slots it spent.
    """
    marker_chars = len(text.strip()) if chars is None else chars
    logger.warning(f"AGENTIC_FETCH_THROTTLED: url={url} method={method} chars={marker_chars} phrase={phrase}")
    return ToolOutcome(
        content_markdown=_THROTTLED_FETCH_MSG.format(url=url, phrase=phrase),
        method="throttled",
        status="throttled",
    )


def _read_content_outcome(
    url: str,
    text: str,
    links: list[str],
    *,
    method: str,
    start_char: int,
    image_leads: str = "",
) -> ToolOutcome:
    """Render a successful body the shared ladder has already classified."""
    return _render_fetch_outcome(
        url,
        text,
        links,
        method=method,
        start_char=start_char,
        image_leads=image_leads,
    )


def _empty_fetch_outcome(url: str) -> ToolOutcome:
    """Outcome for a 200-OK page the ladder could not read (zero extractable text).

    Distinct ``status``/``method`` of ``"empty"`` — never ``"ok"``/``"plain"`` — so the
    loop's tier stamping (which grants "fetched" only on a ``status == "ok"``,
    fetched-class-method outcome; see ``loop._harvest_verification_tiers``) can never
    mark an unread page authoritative. Two deterministic guards, not one: the non-"ok"
    status AND the unmapped method. Deliberately NOT cached — caching the placeholder
    would let a later paginated fetch serve it back as ``method == "cache"`` (a
    fetched-tier method) and re-launder the tier.
    """
    return ToolOutcome(content_markdown=_NO_CONTENT_FETCH_MSG.format(url=url), method="empty", status="empty")


def _per_call_ctx(question_ctx: LadderContext | None, *, query: str) -> LadderContext:
    """A per-CALL context off the question's: its own wall origin, its own rung list, its own ask.

    ``started`` is the origin every rung bounds itself against and one tool call is one wall, so it
    is taken here rather than at intake. What stays the question's is ``shared``, the per-question
    rung budget the archive and paid-read caps count on, so a driver that spends both snapshots on
    one gap cannot spend two more on the next. A call with no question context (which is what the
    suite drives) gets a fresh budget, so it behaves exactly as one call always did.
    """
    base = LadderContext(host_sems=_FETCH_HOST_SEMAPHORES) if question_ctx is None else question_ctx
    return replace(base, query=query, rungs=[], read_captures=[], started=monotonic())


async def _fetch_via_ladder(
    url: str,
    *,
    query: str,
    pol: LadderPolicy,
    ctx: LadderContext | None,
    record: bool = True,
    local_read_receipt: LocalReadReceipt | None = None,
) -> PlainFetchResult:
    """One run of the shared fetch ladder for ``url``, as this ladder's own result.

    The question-platform refusal happens HERE rather than inside the ladder, because it is this
    caller's own policy (the resolution-source fetcher drops those URLs when it selects them) and
    because it must refuse before anything is dialed. Everything past it is the shared ladder:
    the direct fetch with its redirect vetting and local document read, then the rungs ``pol``
    enables. ``record`` is False for the robots.txt pre-check, which is not a fetch the driver made.
    """
    blocked = _fetch_plain_url_block(url)
    if blocked is not None:
        return blocked
    request_context = _per_call_ctx(ctx, query=query)
    if local_read_receipt is not None:
        request_context.local_read_receipt = local_read_receipt
    if request_context.policy.known_api is not None:
        pol = replace(pol, known_api=request_context.policy.known_api)
    result = await fetch_url(url, policy=pol, ctx=request_context)
    if record:
        _log_ladder_markers(result)
    return ladder_adapter.as_plain_result(result, requested_url=url)


def _log_ladder_markers(result: FetchResult) -> None:
    """The two shared fetch markers for one tool call, with ``question=None``.

    The loop has no question id in hand at a tool call, exactly as its three event markers do not,
    so a join to a question goes through the run id (docs/telemetry_markers.md).
    """
    logger.info(fetch_markers.fetch_marker_line(result, qid=None, caller=LADDER_CALLER_GAP_FILL_V2))
    for line in fetch_markers.escalation_marker_lines(result, qid=None, caller=LADDER_CALLER_GAP_FILL_V2):
        logger.info(line)


async def search_news(query: str) -> ToolOutcome:
    client_id = os.getenv(ASKNEWS_CLIENT_ID_ENV)
    secret = os.getenv(ASKNEWS_SECRET_ENV)
    if not client_id or not secret:
        return _format_fetch_error(
            f"AskNews credentials are not configured; set {ASKNEWS_CLIENT_ID_ENV} and {ASKNEWS_SECRET_ENV}.",
            method="news",
        )
    try:
        articles = await _call_asknews_search(query)
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # tool-handler soft-fail boundary: a dead provider becomes a tool result the driver can read, never a loop crash
        return _format_fetch_error(f"AskNews search failed: {type(exc).__name__}: {exc}", method="news")
    return ToolOutcome(content_markdown=_format_asknews_results(articles), method="news")


async def search_web(query: str, end_published_date: str | None = None) -> ToolOutcome:
    if not os.getenv(EXA_API_KEY_ENV):
        return _format_fetch_error(f"Exa API key is not configured; set {EXA_API_KEY_ENV}.", method="search")
    try:
        results = await _call_exa_search(query, end_published_date)
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # tool-handler soft-fail boundary: a dead provider becomes a tool result the driver can read, never a loop crash
        return _format_fetch_error(f"Exa search failed: {type(exc).__name__}: {exc}", method="search")
    return ToolOutcome(content_markdown=_format_exa_results(results), method="search")


def _generic_document_ask(question_topic: str) -> str:
    return f"Extract the main content relevant to: {question_topic}"


def _pdf_local_outcome(url: str, plain: PlainFetchResult, *, start_char: int) -> ToolOutcome:
    """Serve a locally extracted PDF, then hold it for the rest of the run.

    The text goes through the same window/cache path an HTML page does, so ``start_char``
    paginates a 220-page report exactly as it paginates a long article. The parse is also
    re-keyed under the URL the driver asked for: the extraction cached it under the final hop,
    and a later ``read_document`` on the original URL would otherwise refetch and reparse it.
    """
    pdf = document_cache.cached_document(plain.url)
    if pdf is not None:
        document_cache.cache_document(url, pdf)
    local_document.log_local_document_read(
        url,
        method=local_document.PDF_LOCAL_METHOD,
        chars=len(plain.text),
        pages=None if pdf is None else pdf.pages_read,
        passages=None,
    )
    return _read_content_outcome(url, plain.text, plain.links, method=plain.method, start_char=start_char)


def _blocked_outcome(blocked: PlainFetchResult) -> ToolOutcome:
    """The one ``blocked`` contract the driver reads, whichever tool or rung refused the URL."""
    return ToolOutcome(content_markdown=blocked.text, method=blocked.method, status="blocked")


def _held_from_result(url: str, result: PlainFetchResult) -> local_document.HeldDocument:
    """What one ladder rung's result leaves us holding for ``url``.

    A parse from the local PDF rung wins over the flat text, because its page offsets are what
    make a digest's ``[p.N]`` labels exact; a scan is still held (page structure, no text), so
    the caller knows the free route is exhausted rather than untried. Text we do hold is cached
    for the run, so a later paginated ``fetch`` of the same URL is free.
    """
    if result.method == local_document.OVERSIZE_DOCUMENT_METHOD:
        return local_document.HeldDocument(oversize=True)
    if result.local_read_refused or result.local_kind == "image":
        return local_document.HeldDocument(local_refusal=result)
    if result.status == "blocked" and _fetch_plain_url_block(result.url) is not None:
        # A 3xx onto a question platform, held so the paid reader (Google's address) declines the same hop.
        return local_document.HeldDocument(refused_landing=result)
    source = run_cache.local_source_for(url) or run_cache.local_source_for(result.url)
    if source is not None:
        return local_document.HeldDocument(source=source)
    pdf = document_cache.cached_document(result.url)
    if pdf is not None:
        held = local_document.held_pdf(pdf)
    elif result.status == "ok" and result.method != DOCUMENT_NEEDED_METHOD:
        held = local_document.HeldDocument(text=result.text.strip())
    else:
        # Reading the use-read_document placeholder as the page would digest our own instruction.
        return local_document.HeldDocument()
    return held


async def _run_local_document_ladder(
    url: str,
    *,
    ctx: LadderContext | None,
    local_read_receipt: LocalReadReceipt | None = None,
) -> local_document.HeldDocument:
    """The free rungs a document read gets: the shared ladder under ``GAP_FILL_DOCUMENT_POLICY``.

    Its 25 s wall is what every rung bounds itself against, and its rung set is ``fetch``'s minus
    the archive: this ladder sits immediately in front of the paid ``url_context`` read, and an
    archived copy is not what a document read was asked for. The impersonated retry matters more
    here than in ``fetch`` for the same reason — a 403 left standing was a paid read of a page the
    free retry fetches (a bls.gov PDF is one of the four measured rescues) — and the browser still
    runs on a page whose text is thin enough to look like a JavaScript shell, because digesting 100
    characters of navigation chrome would answer the ask out of furniture.
    """
    plain = await _fetch_via_ladder(
        url,
        query="",
        pol=GAP_FILL_DOCUMENT_POLICY,
        ctx=ctx,
        local_read_receipt=local_read_receipt,
    )
    return _held_from_result(url, plain)


async def _acquire_local_document(url: str, *, ctx: LadderContext | None = None) -> local_document.HeldDocument:
    """What the free ladder holds for ``url``: something already read this run, or a fresh try.

    Bounded by an outer ``_LOCAL_DOCUMENT_BUDGET_S`` wall matching the ladder policy, so a slow
    queue, host, or parser cannot spend the paid reader's budget as well as its own. Every rung
    also declines under its remaining-wall floor. A per-call receipt distinguishes the one timeout
    that is terminal: once local-source bytes are acquired, the parser timeout cannot fall through
    to a paid reader while its worker finishes in the background. An ordinary acquisition timeout
    keeps the established paid fallback.

    A local-source parsing thread may outlive its caller because Python cannot cancel it. That
    worker keeps the shared parse gate occupied until it really finishes, preventing abandoned
    archive or Office work from opening extra parser capacity.
    """
    cached_source = run_cache.local_source_for(url)
    if cached_source is not None:
        return local_document.HeldDocument(source=cached_source)
    cached_image = run_cache.image_source_for(url)
    if cached_image is not None:
        return local_document.HeldDocument(
            local_refusal=PlainFetchResult(
                status="error",
                method="local_navigation",
                text="This URL is a raster image; use view_image(url, crop) to inspect its pixels.",
                links=[],
                url=cached_image.url,
                content_type=cached_image.content_type,
                local_kind="image",
                navigation_only=True,
            )
        )
    cached_pdf = document_cache.cached_document(url)
    if cached_pdf is not None:
        return local_document.held_pdf(cached_pdf)
    local_read_receipt = LocalReadReceipt()
    try:
        return await asyncio.wait_for(
            _run_local_document_ladder(url, ctx=ctx, local_read_receipt=local_read_receipt),
            timeout=_LOCAL_DOCUMENT_BUDGET_S,
        )
    except TimeoutError:
        if local_read_receipt.encountered or run_cache.local_source_for(url) is not None:
            return local_document.HeldDocument(
                local_refusal=PlainFetchResult(
                    status="error",
                    method="local_selection",
                    text="Local source read timed out after its bytes were acquired; no model fallback was attempted.",
                    links=[],
                    url=url,
                    local_read_refused=True,
                )
            )
        logger.info(
            "agentic read_document local acquisition exceeded %.0fs, falling back to the reader: %s",
            _LOCAL_DOCUMENT_BUDGET_S,
            urlparse(url).netloc,
        )
        return local_document.HeldDocument()


async def _local_digest_outcome(
    url: str,
    ask: str,
    held: local_document.HeldDocument,
    *,
    policy: LadderPolicy,
    budget_seconds: float,
) -> ToolOutcome | None:
    """Answer the ask from text we hold, deterministically and for free — or None to pay instead.

    None means the one shape where a digest would answer the ask out of furniture: a sub-floor page
    with no parse behind it whose digest selected NO passage. All three conditions are needed, and
    the digest runs off the event loop for a measured reason; both receipts are in
    ``docs/agentic_gap_fill.md`` "Why the free digest can refuse to answer".
    """
    if held.pdf is not None:
        digest = await asyncio.to_thread(
            local_document.digest_held,
            held,
            ask=ask,
            top_k=DOCUMENT_DIGEST_TOP_K,
            max_chars=_FETCH_WINDOW_CHARS,
            source_url=url,
        )
        local_document.log_local_document_read(
            url,
            method=local_document.DIGEST_LOCAL_METHOD,
            chars=len(held.text),
            pages=held.pdf.pages_read,
            passages=digest.passages,
        )
        return ToolOutcome(content_markdown=digest.block, method=local_document.DIGEST_LOCAL_METHOD)

    minimum_content_chars = policy.thin_content_escalation_chars
    if minimum_content_chars is not None and len(held.text) < minimum_content_chars:
        matching_passages = await asyncio.to_thread(
            document_text.select_passages, held.text, ask, top_k=DOCUMENT_DIGEST_TOP_K
        )
        if not matching_passages:
            return None

    digest_fn = policy.digest or bm25_digest
    passages = await digest_fn(held.text, ask, budget_seconds=max(0.0, budget_seconds))
    rendered = await asyncio.to_thread(
        document_text.render_flat_passages,
        passages.passages,
        query=ask,
        max_chars=_FETCH_WINDOW_CHARS,
        source_url=url,
        source_chars=len(held.text),
    )
    local_document.log_local_document_read(
        url,
        method=local_document.DIGEST_LOCAL_METHOD,
        chars=len(held.text),
        pages=None,
        passages=passages.passages_grounded,
    )
    return ToolOutcome(content_markdown=rendered, method=local_document.DIGEST_LOCAL_METHOD)


def _local_source_for_result(url: str, plain: PlainFetchResult) -> ParsedSource:
    source = run_cache.local_source_for(url) or run_cache.local_source_for(plain.url)
    if source is None:
        raise RuntimeError(f"local source cache entry disappeared for {url}")
    return source


def _local_source_fetch_outcome(
    url: str,
    plain: PlainFetchResult,
    *,
    start_char: int,
    member: str | None,
    sheet: str | None,
) -> ToolOutcome:
    source = _local_source_for_result(url, plain)
    try:
        sections = source_presentation.select_source_sections(source, member=member, sheet=sheet)
        selected_member = next((entry for entry in source.members if entry.name == member), None)
        navigation_only = (
            (source.kind == "archive" and member is None)
            or (source.kind == "workbook" and sheet is None)
            or (selected_member is not None and selected_member.kind == "workbook" and sheet is None)
        )
        if navigation_only:
            return _render_fetch_outcome(
                url,
                source_presentation.source_inventory(source, member=member),
                plain.links,
                method=ladder_adapter.LOCAL_NAVIGATION_METHOD,
                start_char=start_char,
            )
        text = source_presentation.source_text(sections)
    except ValueError as error:
        return _format_fetch_error(str(error), method="local_selection")
    return _render_fetch_outcome(
        url,
        text,
        plain.links,
        method=ladder_adapter.LOCAL_SOURCE_METHOD,
        start_char=start_char,
    )


async def _local_source_digest_outcome(
    url: str,
    ask: str,
    source: ParsedSource,
    *,
    member: str | None,
    sheet: str | None,
    budget_seconds: float,
) -> ToolOutcome:
    deadline = monotonic() + max(0.0, budget_seconds)
    worker_gate = pdf_parse_semaphore()
    try:
        await asyncio.wait_for(worker_gate.acquire(), timeout=max(0.0, deadline - monotonic()))
    except TimeoutError:
        return _format_fetch_error(
            "Local source digest timed out; no model fallback was attempted.", method="local_selection"
        )

    worker = asyncio.create_task(
        asyncio.to_thread(
            source_presentation.digest_source,
            source,
            query=ask,
            source_url=url,
            max_chars=_FETCH_WINDOW_CHARS,
            member=member,
            sheet=sheet,
        )
    )
    worker.add_done_callback(lambda finished: _finish_local_digest_worker(finished, worker_gate))
    try:
        digest = await asyncio.wait_for(asyncio.shield(worker), timeout=max(0.0, deadline - monotonic()))
    except TimeoutError:
        return _format_fetch_error(
            "Local source digest timed out; no model fallback was attempted.", method="local_selection"
        )
    except ValueError as error:
        return _format_fetch_error(str(error), method="local_selection")
    return ToolOutcome(content_markdown=digest.block, method=local_document.DIGEST_LOCAL_METHOD)


def _finish_local_digest_worker(
    worker: asyncio.Task[document_text.DocumentDigest], worker_gate: asyncio.Semaphore
) -> None:
    worker_gate.release()
    if worker.cancelled():
        return
    worker.exception()


def _special_fetch_outcome(
    url: str,
    plain: PlainFetchResult,
    *,
    start_char: int,
    member: str | None,
    sheet: str | None,
) -> ToolOutcome | None:
    if plain.status == "blocked":
        return _blocked_outcome(plain)
    if plain.status == "throttled":
        if plain.throttle_phrase is None or plain.throttle_chars is None or plain.throttle_method is None:
            raise RuntimeError("throttled ladder result is missing marker metadata")
        return _throttled_fetch_outcome(
            url,
            "",
            plain.throttle_phrase,
            method=plain.throttle_method,
            chars=plain.throttle_chars,
        )
    if plain.method == local_document.PDF_LOCAL_METHOD:
        return _pdf_local_outcome(url, plain, start_char=start_char)
    if plain.local_read_refused:
        return ToolOutcome(content_markdown=plain.text, method=plain.method, status="error")
    if plain.local_kind in {"archive", "workbook", "word"}:
        return _local_source_fetch_outcome(
            url,
            plain,
            start_char=start_char,
            member=member,
            sheet=sheet,
        )
    if member is not None or sheet is not None:
        return _format_fetch_error(
            "member and sheet selectors require an archive or workbook", method="local_selection"
        )
    return None


async def fetch(
    url: str,
    start_char: int = 0,
    member: str | None = None,
    sheet: str | None = None,
    *,
    question_topic: str = "",
    ctx: LadderContext | None = None,
    image_state: ImageViewState | None = None,
) -> ToolOutcome:
    """Read ``url`` for the driver through the shared ladder, then return its requested window.

    The rungs — the impersonated retry on a host's 403, the browser on a page too thin to be the
    page, the archive on one our address never reached — all run inside ``fetch_url`` under
    ``GAP_FILL_FETCH_POLICY``, so what is left here is reading the outcome: refuse a blocked URL,
    paginate a locally read document, hand a document with no text layer to ``read_document``, and
    otherwise window and cache what was read. A page the browser could not rescue either comes back
    ``empty`` and NEVER as a plain success, because the loop grants the ``fetched`` tier on status
    alone (docs/agentic_gap_fill.md).
    """
    plain = await _fetch_via_ladder(url, query=question_topic, pol=GAP_FILL_FETCH_POLICY, ctx=ctx)
    if image_state is not None and plain.image_leads:
        await image_state.remember_leads(tuple(dict.fromkeys((url, plain.url))), plain.image_leads)
    if plain.local_kind == "image":
        return await view_acquired_image(url, plain, state=image_state or ImageViewState())
    special = _special_fetch_outcome(url, plain, start_char=start_char, member=member, sheet=sheet)
    if special is not None:
        return special
    if plain.method == DOCUMENT_NEEDED_METHOD:
        # `ladder_exhausted` says the free rungs just ran, so the reader does not re-request.
        return await read_document(plain.url, _generic_document_ask(question_topic), ladder_exhausted=True, ctx=ctx)
    if plain.status == "ok":
        return _read_content_outcome(
            url,
            plain.text,
            plain.links,
            method=plain.method,
            start_char=start_char,
            image_leads=render_image_leads(plain.image_leads),
        )
    if plain.status == "empty":
        return _empty_fetch_outcome(plain.url)
    return ToolOutcome(content_markdown=plain.text, method=plain.method, status="error")


_ROBOTS_DISALLOWED_MSG = (
    "Document read not attempted: {host}'s robots.txt disallows Google-Extended, the token "
    "Gemini's url_context reader identifies as, so that read is refused at the host and returns "
    "no content whatever it costs. Nothing from this URL was read; do NOT cite it as a fetched "
    "source, and do NOT read it as evidence the fact is unavailable. Retrying will not help — "
    "look for the same fact on another host."
)

_PAID_DOCUMENT_READ_CAP_MSG = "Document read not attempted: this question's paid document-read limit is exhausted."


async def _fetch_robots_txt(robots_url: str, *, ctx: LadderContext | None = None) -> str | None:
    """Read one robots.txt through the shared ladder's DIRECT fetch; None when we could not.

    ``GAP_FILL_DIRECT_POLICY`` is the point: one direct fetch, no escalation rung, and this
    caller's verdict, which has no content floor. A robots.txt body is 33 to 45 characters, so a
    floor would read every host as "no directives" and quietly open the paid rung on hosts that
    disallow it. Everything else is the shared path's: the SSRF preflight, the filtering resolver,
    the per-hop redirect vetting and the body cap. Bounded at ``ROBOTS_FETCH_TIMEOUT_S`` on top of
    the hop's own clamp, because an unbounded per-host acquire is not a sensible price for a
    pre-check whose only job is to avoid one paid call; a timeout reads as unreadable, which is the
    only direction this may fail in. The per-host cache is ``robots_policy``'s, shared with the
    Tier-1 reader, because a host's policy is a property of the host.
    """
    try:
        result = await asyncio.wait_for(
            _fetch_via_ladder(robots_url, query="", pol=GAP_FILL_DIRECT_POLICY, ctx=ctx, record=False),
            timeout=ROBOTS_FETCH_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # pre-check soft-fail boundary: a robots.txt we cannot read must degrade to paying, never to failing the read
        logger.debug("agentic robots.txt pre-check failed for %s: %s: %s", robots_url, type(exc).__name__, exc)
        return None
    if result.status == "ok" and result.method in ("plain", "cache"):
        return result.text
    return None


async def _url_context_robots_skip(url: str, *, ctx: LadderContext | None = None) -> bool:
    """True when this host tells ``Google-Extended`` to stay out of ``url``'s path.

    Only the paid ``url_context`` rung consults this: the free rungs dial from our own client
    under our own user agent, and this bot's reading of ``Content-Signal: use=reference`` is
    that reference use is permitted. Proven live 2026-09-03 — see ``robots_policy``, which owns
    the per-host cache this shares with the Tier-1 reader.
    """
    return await google_extended_blocks_url(url, fetch_text=lambda robots_url: _fetch_robots_txt(robots_url, ctx=ctx))


def _take_paid_document_read_attempt(url: str, ctx: LadderContext | None) -> bool:
    """Claim one paid read from a question context; standalone calls own an independent allowance."""
    if ctx is None or ctx.shared.take_url_context_attempt():
        return True
    logger.info(
        "agentic read_document: skipping the paid reader for %s — this question's %d paid read(s) are spent",
        urlparse(url).netloc,
        RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS,
    )
    return False


async def _free_route_outcome(
    url: str,
    ask: str,
    held: local_document.HeldDocument,
    *,
    policy: LadderPolicy,
    budget_seconds: float,
    member: str | None,
    sheet: str | None,
) -> ToolOutcome | None:
    """What the free ladder settles for ``read_document`` without a paid read; None gives the reader its turn.

    Three settled shapes. A URL that led onto a question platform is refused, because the paid
    reader would follow the same hop. An oversize body is an error rather than a reason to
    escalate. Text we hold is digested, and the size gate rides the same branch as the text it
    guards so the two can never disagree: a document we hold is served from the digest whatever
    its size, and the biggest are the clearest case (the nine archived documents past the gate
    carried 67% of the season's reader tokens and the largest of them returned nothing for the
    money). A None digest is the one shape that must not be served: sub-floor chrome that no
    passage matched, which the paid reader is the right rung for (see ``_local_digest_outcome``).
    """
    if held.refused_landing is not None:
        return _blocked_outcome(held.refused_landing)
    if held.local_refusal is not None:
        message = held.local_refusal.text
        if held.local_refusal.local_kind == "image":
            message = f"{message} Use view_image(url, crop) to inspect its pixels."
        return ToolOutcome(
            content_markdown=message,
            method=held.local_refusal.method,
            status="error",
        )
    if held.source is not None:
        return await _local_source_digest_outcome(
            url,
            ask,
            held.source,
            member=member,
            sheet=sheet,
            budget_seconds=budget_seconds,
        )
    if held.oversize:
        return _format_fetch_error(local_document.oversize_message(url), method=local_document.OVERSIZE_DOCUMENT_METHOD)
    if held.has_text or local_document.exceeds_url_context_size_gate(held.text):
        return await _local_digest_outcome(url, ask, held, policy=policy, budget_seconds=budget_seconds)
    return None


async def read_document(
    url: str,
    ask: str,
    member: str | None = None,
    sheet: str | None = None,
    *,
    ladder_exhausted: bool = False,
    ctx: LadderContext | None = None,
    image_state: ImageViewState | None = None,
) -> ToolOutcome:
    """Answer ``ask`` about ``url``: from the page's own text where we can get it, else Gemini.

    Acquisition-first. The free ladder runs before anything is spent (this run's cache, then
    the plain, impersonated-retry and rendered rungs ``fetch`` uses), and any text it holds is
    answered with a deterministic BM25 passage digest — ``method="digest_local"``. The paid
    ``url_context`` read happens only when the ladder holds nothing: a host that refuses us, a
    page with no text at all, or a PDF with no text layer. Measured 2026-09-03, that is two of 47
    archived fetch failures, against 191 reader calls over the 2026 summer season.

    A question-platform URL is refused before any rung runs, with the same ``blocked`` outcome
    ``fetch`` gives it, and so is a URL that 3xxes onto one (the free ladder's refusal of that hop
    comes back as ``HeldDocument.refused_landing``). Zero successful ``url_context`` retrievals withholds the ``fetched`` tier, and
    ``ladder_exhausted`` is internal and hidden from the driver-facing schema; both receipts are in
    ``docs/agentic_gap_fill.md`` "Why the paid reader's retrieval-count guard stays".
    """
    blocked = _fetch_plain_url_block(url)
    if blocked is not None:
        return _blocked_outcome(blocked)
    started = monotonic()
    held = local_document.HeldDocument() if ladder_exhausted else await _acquire_local_document(url, ctx=ctx)
    if held.local_refusal is not None and held.local_refusal.local_kind == "image":
        return await view_acquired_image(url, held.local_refusal, state=image_state or ImageViewState())
    settled = await _free_route_outcome(
        url,
        ask,
        held,
        policy=GAP_FILL_DOCUMENT_POLICY,
        budget_seconds=max(0.0, _READ_DOCUMENT_TOTAL_BUDGET_S - (monotonic() - started)),
        member=member,
        sheet=sheet,
    )
    if settled is not None:
        return settled
    if not os.getenv(GOOGLE_API_KEY_ENV):
        return _format_fetch_error(f"Google API key is not configured; set {GOOGLE_API_KEY_ENV}.", method="document")
    if await _url_context_robots_skip(url, ctx=ctx):
        # Its own status token, never tiered: nothing was read, and a retry cannot help.
        logger.info(f"AGENTIC_URLCONTEXT_ROBOTS_SKIP: url={url} host={robots_host(url)}")
        return _format_fetch_error(
            _ROBOTS_DISALLOWED_MSG.format(host=robots_host(url)),
            status="robots_disallowed",
            method="document",
        )
    if not _take_paid_document_read_attempt(url, ctx):
        return _format_fetch_error(_PAID_DOCUMENT_READ_CAP_MSG, method="document")
    try:
        # What the total budget has left (docs/agentic_gap_fill.md, the budget arithmetic).
        text, n_url_success, statuses = await asyncio.wait_for(
            asyncio.to_thread(_run_document_read_sync, url, ask),
            timeout=min(_READ_DOCUMENT_TIMEOUT_S, _READ_DOCUMENT_TOTAL_BUDGET_S - (monotonic() - started)),
        )
    except TimeoutError:
        return _format_fetch_error("Document read timed out.", method="document")
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # tool-handler soft-fail boundary: a dead reader becomes a tool result the driver can read, never a loop crash
        return _format_fetch_error(f"Document read failed: {type(exc).__name__}: {exc}", method="document")
    if n_url_success == 0:
        # Greppable and keyed on `statuses`, which is the only thing separating three zeroes.
        logger.warning(f"AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED: url={url} statuses={','.join(statuses) or 'none'}")
        return _format_fetch_error(
            f"Document read retrieved no URL content: Gemini's url_context tool fetched nothing from {url}, "
            "so any answer would be unsourced recall rather than a read of the document.",
            method="document",
        )
    return ToolOutcome(content_markdown=text, method="document")


def question_ladder_context(*, session: object | None = None) -> LadderContext:
    """The ONE fetch-ladder context a question's tool calls share.

    What it carries is the per-question half: a fresh :class:`QuestionRungBudget`, which is what
    caps this question at two archive snapshots and two paid reads however many URLs the driver
    picks, and this ladder's own per-host politeness map. Everything per call — the ask, the wall
    origin, the rung list — is derived off it in :func:`_per_call_ctx`. Its own function so the
    seam that builds it (``agentic_gap_fill.run_gap_fill_v2``) does not have to know the fields.
    """
    return LadderContext(shared=QuestionRungBudget(), host_sems=_FETCH_HOST_SEMAPHORES, session=session)


def build_gap_fill_tools(question_topic: str, *, ctx: LadderContext | None = None) -> list[ToolSpec]:
    """The four tools the driver sees, in the order it sees them.

    ``ctx`` is the question's fetch-ladder context (:func:`question_ladder_context`), captured into
    the two handlers that fetch. None means one fresh budget per call, which is what a direct call
    with no question in hand gets.
    """

    image_state = ImageViewState()

    async def _fetch_with_topic(
        url: str,
        start_char: int = 0,
        member: str | None = None,
        sheet: str | None = None,
    ) -> ToolOutcome:
        """``fetch`` with the question-scoped ladder and image state bound."""
        return await fetch(
            url,
            start_char,
            member,
            sheet,
            question_topic=question_topic,
            ctx=ctx,
            image_state=image_state,
        )

    async def _read_document_public(
        url: str,
        ask: str,
        member: str | None = None,
        sheet: str | None = None,
    ) -> ToolOutcome:
        """``read_document`` with (url, ask) only, so a hallucinated ``ladder_exhausted`` cannot pay.

        The loop binds handlers with ``**arguments`` straight off the model, so an advertised — or
        merely invented — ``ladder_exhausted: true`` would skip the free ladder. Resolves
        ``read_document`` as a module attribute at call time, so the suite's patches still land.
        """
        return await read_document(url, ask, member, sheet, ctx=ctx, image_state=image_state)

    async def _acquire_image(url: str) -> PlainFetchResult:
        return await _fetch_via_ladder(url, query=question_topic, pol=GAP_FILL_FETCH_POLICY, ctx=ctx)

    async def _view_image_public(url: str, crop: list[int] | None = None) -> ToolOutcome:
        return await view_image(url, crop, state=image_state, acquire=_acquire_image)

    tools = [
        ToolSpec(
            name="search_news",
            description=SEARCH_NEWS_DESCRIPTION,
            parameters=_SEARCH_NEWS_PARAMETERS,
            handler=search_news,
            timeout_s=90,
        ),
        ToolSpec(
            name="search_web",
            description=SEARCH_WEB_DESCRIPTION,
            parameters=_SEARCH_WEB_PARAMETERS,
            handler=search_web,
            timeout_s=20,
        ),
        ToolSpec(
            name="fetch",
            description=FETCH_DESCRIPTION,
            parameters=_FETCH_PARAMETERS,
            handler=_fetch_with_topic,
            # Above _READ_DOCUMENT_TIMEOUT_S, so the document escalation fits inside this budget.
            timeout_s=90,
        ),
        ToolSpec(
            name="read_document",
            description=READ_DOCUMENT_DESCRIPTION,
            parameters=_READ_DOCUMENT_PARAMETERS,
            handler=_read_document_public,
            # 70 is GAP_FILL_V2_CONCLUDE_THRESHOLD (docs/agentic_gap_fill.md, the budget arithmetic).
            timeout_s=70,
        ),
        ToolSpec(
            name="view_image",
            description=VIEW_IMAGE_DESCRIPTION,
            parameters=_VIEW_IMAGE_PARAMETERS,
            handler=_view_image_public,
            timeout_s=70,
        ),
    ]

    # Importing known_api.tools at module scope would complete the agentic package import through
    # agentic.types while agentic.__init__ is still importing this module.
    from metaculus_bot.research.known_api import backends, wiring  # noqa: PLC0415  # real circular import
    from metaculus_bot.research.known_api import tools as known_api_tools  # noqa: PLC0415  # real circular import

    if ctx is None:
        known_api_budget = backends.KalshiGetBudget()
        known_api_session = None
    else:
        known_api_budget = backends.KalshiGetBudget()
        known_api_session = ctx.session
        ctx.policy = replace(
            ctx.policy,
            known_api=wiring.build_known_api_fetcher(
                session=known_api_session,
                kalshi_detail_budget=known_api_budget,
            ),
        )
    return tools + known_api_tools.build_known_api_tools(
        session=known_api_session,
        kalshi_detail_budget=known_api_budget,
    )
