"""Provider calls behind the agentic search and document-read tools.

The outbound half of ``search_news`` / ``search_web`` / ``read_document``: the AskNews and
Exa clients with their retry ladders and concurrency caps, the Gemini url_context document
read, and the markdown formatting of whatever comes back (plus the small coercions that
absorb SDK payloads arriving as either mappings or attribute objects).

The tool handlers themselves stay in ``tools.py``: ``search_web`` resolves
``_call_exa_search`` and ``read_document`` resolves ``_run_document_read_sync`` as module
attributes of ``tools``, which the suite monkeypatches, so a handler defined here would
read its own global and silently bypass the patch.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Mapping
from typing import Any

from metaculus_bot.constants import (
    ASKNEWS_BACKOFF_SECS,
    ASKNEWS_CLIENT_ID_ENV,
    ASKNEWS_MAX_TRIES,
    ASKNEWS_SECRET_ENV,
    EXA_API_KEY_ENV,
    GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS,
    GAP_FILL_V2_READER_HTTP_ATTEMPTS,
    GAP_FILL_V2_READER_MODEL,
    GAP_FILL_V2_READER_THINKING_LEVEL,
    GOOGLE_API_KEY_ENV,
)
from metaculus_bot.research import providers as research_providers
from metaculus_bot.research.agentic.types import EvidencePassage
from metaculus_bot.research.gemini_client_config import gemini_retry_sleep_allowance_s
from metaculus_bot.research.url_context_reader import run_url_context_read

# Client-side HTTP ceilings sized against the tools' loop budgets so the underlying socket is
# torn down rather than left pinned to a wall deadline — a hung endpoint then frees its slot.
# Exa's sits strictly UNDER its tool's budget. The reader's is FIXED while the budget it runs
# under became variable, so the two no longer nest: see the paragraph below the constant.
_EXA_HTTP_TIMEOUT_S = 18.0  # under search_web's ToolSpec timeout_s in build_gap_fill_tools
_READ_DOCUMENT_HTTP_TIMEOUT_MS = 55_000  # fixed; read_document's own wait is 40-60 s (below)
# The retry ladder has to fit INSIDE that budget rather than beside it: read_document's
# outer ``asyncio.wait_for`` cancels the coroutine but not the ``to_thread`` worker, so a
# retried attempt that ran past the budget would leak a pooled thread for longer than
# today's 55s — a worse worst case, which this path will not take. So the budget stays put
# and the ATTEMPTS divide it, after the backoff sleeps are set aside:
# (55_000 - 2_000) // 2 = 26_500ms each, worst case 26.5 + <=2 + 26.5 = 55.0s. The cost is that
# ONE attempt now gets 26.5s instead of 55s; the retry is worth it because the failure it
# recovers (a 503 UNAVAILABLE) returns in milliseconds and leaves nearly the whole budget
# for the second try, and the reader's thinking level dropped a tier in the same change.
#
# That 55 s is NOT under read_document's wait any more. Since the free local-acquisition ladder
# landed ahead of the paid read, the coroutine waits
# ``min(_READ_DOCUMENT_TIMEOUT_S=60, _READ_DOCUMENT_TOTAL_BUDGET_S=65 - acquisition_elapsed)``,
# which is 55 s at 10 s of acquisition and 40 s at the 25 s acquisition cap. So past ~10 s of
# acquisition the worker can outlive the wait by up to 15 s and finish a billed call whose
# answer is thrown away. It cannot start a new billed request after the wait fires: the second
# attempt begins by 26.5 + 2 = 28.5 s, inside the 40 s floor. Deriving the per-attempt timeout
# from the variable wait instead would cut an attempt to 19 s on the 25 s-acquisition handover
# path, failing 20-26 s reads that succeed today, so the fixed arithmetic stays and the overrun
# is documented. ``tests/test_agentic_tools.py`` pins both halves of that claim.
_READ_DOCUMENT_HTTP_PER_ATTEMPT_TIMEOUT_MS = int(
    (_READ_DOCUMENT_HTTP_TIMEOUT_MS - 1000 * gemini_retry_sleep_allowance_s(GAP_FILL_V2_READER_HTTP_ATTEMPTS))
    // GAP_FILL_V2_READER_HTTP_ATTEMPTS
)
_EXA_RETRY_DELAYS_S = (1.0, 4.0)
_EXA_GLOBAL_SEMAPHORE = asyncio.Semaphore(4)

_RATE_LIMIT_RE = re.compile(r"\b(429|rate[\s-]?limit|too many requests|over limit|quota)\b", re.IGNORECASE)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except TypeError:
            return str(value)
    return str(value)


def _mapping_or_attrs_get(item: object, *names: str) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if name in item:
                return item[name]
        return None
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _format_asknews_results(articles: list[Any]) -> str:
    if not articles:
        return "No AskNews articles found."
    lines: list[str] = []
    for article in articles:
        title = _stringify(_mapping_or_attrs_get(article, "eng_title", "title")) or "Untitled article"
        date = _stringify(_mapping_or_attrs_get(article, "pub_date"))
        source = _stringify(_mapping_or_attrs_get(article, "source_id")) or "unknown"
        url = _stringify(_mapping_or_attrs_get(article, "article_url"))
        summary = _stringify(_mapping_or_attrs_get(article, "summary"))
        lines.append(f"### {title}")
        if date:
            lines.append(f"Date: {date}")
        lines.append(f"Source: {source}")
        if url:
            lines.append(f"URL: {url}")
        if summary:
            lines.append(f"Summary: {summary}")
        lines.append("")
    lines.pop()
    return "\n".join(lines)


def _asknews_evidence_passages(articles: list[Any]) -> list[EvidencePassage]:
    """Expose each provider summary only with the article URL carried by that result."""
    passages: list[EvidencePassage] = []
    for article in articles:
        source_url = _stringify(_mapping_or_attrs_get(article, "article_url"))
        summary = _stringify(_mapping_or_attrs_get(article, "summary"))
        if source_url and summary:
            passages.extend(
                EvidencePassage(
                    source_url=source_url,
                    text=paragraph,
                    retrieval_method="news",
                    content_origin="provider_summary",
                )
                for paragraph in summary.split("\n\n")
                if paragraph.strip() and len(paragraph) <= GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS
            )
    return passages


def _format_exa_results(results: list[Any]) -> str:
    if not results:
        return "No Exa results found."
    lines: list[str] = []
    for result in results:
        title = _stringify(_mapping_or_attrs_get(result, "title")) or "Untitled result"
        url = _stringify(_mapping_or_attrs_get(result, "url"))
        published = _stringify(_mapping_or_attrs_get(result, "published_date", "publishedDate"))
        highlights_raw = _mapping_or_attrs_get(result, "highlights")
        if isinstance(highlights_raw, str):
            highlights = [highlights_raw]
        elif isinstance(highlights_raw, list):
            highlights = [_stringify(item) for item in highlights_raw if _stringify(item)]
        else:
            highlights = []
        lines.append(f"### {title}")
        if url:
            lines.append(f"URL: {url}")
        if published:
            lines.append(f"Published: {published}")
        if highlights:
            lines.append("Highlights:")
            lines.extend(f"- {highlight}" for highlight in highlights)
        lines.append("")
    lines.pop()
    return "\n".join(lines)


def _exa_evidence_passages(results: list[Any]) -> list[EvidencePassage]:
    """Expose Exa highlights as provider summaries with their own structured result URL."""
    passages: list[EvidencePassage] = []
    for result in results:
        source_url = _stringify(_mapping_or_attrs_get(result, "url"))
        highlights_raw = _mapping_or_attrs_get(result, "highlights")
        if isinstance(highlights_raw, str):
            highlights = [highlights_raw]
        elif isinstance(highlights_raw, list):
            highlights = [_stringify(item) for item in highlights_raw if _stringify(item)]
        else:
            highlights = []
        if source_url:
            passages.extend(
                EvidencePassage(
                    source_url=source_url,
                    text=highlight,
                    retrieval_method="search",
                    content_origin="provider_summary",
                )
                for highlight in highlights
                if len(highlight) <= GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS
            )
    return passages


def _is_rate_limited_error(exc: BaseException) -> bool:
    """Generic 429/rate-limit/quota classifier (Exa and AskNews retry paths)."""
    return bool(_RATE_LIMIT_RE.search(str(exc)))


async def _asknews_search_with_retry(sdk: Any, query: str, tries: int, backoff: float) -> list[Any]:
    """Issue the AskNews search over an open SDK session, retrying only transient throttling.

    Split out of :func:`_call_asknews_search` so the retry ladder is not nested
    inside the semaphore + SDK context managers.
    """
    last_exc: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            await research_providers.asknews_rate_gate()
            response = await sdk.news.search_news(
                query=query,
                n_articles=6,
                return_type="both",
                strategy="news knowledge",
            )
            return list(response.as_dicts or [])
        except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
            last_exc = exc
            # Transient-only retry, matching the primary provider's ``_is_retryable``
            # (``providers.py``). The 403011 subscription-inactive error is PERMANENT —
            # off-season billing, not throttling — so it must NOT be exempted here: re-rolling
            # it burns the whole ``ASKNEWS_MAX_TRIES`` ladder of ``ASKNEWS_BACKOFF_SECS``
            # sleeps (a double-digit fraction of GAP_FILL_V2_WALL_DEADLINE, since the sleep
            # below grows as 3**attempt) on a call that can never succeed.
            if not _is_rate_limited_error(exc) and "concurrency limit" not in str(exc).lower():
                raise
            if attempt >= tries:
                raise
            await asyncio.sleep(backoff * (10 + 3**attempt))
    if last_exc is not None:
        raise last_exc
    return []


async def _call_asknews_search(query: str) -> list[Any]:
    from asknews_sdk import AsyncAskNewsSDK  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import

    client_id = os.getenv(ASKNEWS_CLIENT_ID_ENV)
    secret = os.getenv(ASKNEWS_SECRET_ENV)
    if not client_id or not secret:
        raise ValueError(f"Missing AskNews credentials: {ASKNEWS_CLIENT_ID_ENV} / {ASKNEWS_SECRET_ENV}")

    tries = max(1, int(ASKNEWS_MAX_TRIES))
    backoff = float(ASKNEWS_BACKOFF_SECS)
    semaphore = research_providers.get_asknews_semaphore()

    async with semaphore, AsyncAskNewsSDK(client_id=client_id, client_secret=secret, scopes={"news"}) as sdk:
        return await _asknews_search_with_retry(sdk, query, tries, backoff)


async def _run_exa_search(query: str, end_published_date: str | None) -> Any:
    import httpx  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import
    from exa_py import AsyncExa  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import

    api_key = os.getenv(EXA_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"Missing Exa API key: {EXA_API_KEY_ENV}")
    client = AsyncExa(api_key=api_key)
    # Inject a client-side-bounded httpx client (the SDK's default timeout is
    # 600s): a hung Exa endpoint gives up at _EXA_HTTP_TIMEOUT_S instead of
    # pinning the coroutine to the loop's wall deadline. The async client also
    # means no worker thread to leak — the old to_thread path could strand a
    # hung sync `requests` call in the shared ThreadPoolExecutor.
    client._client = httpx.AsyncClient(base_url=client.base_url, headers=client.headers, timeout=_EXA_HTTP_TIMEOUT_S)
    kwargs: dict[str, Any] = {
        "query": query,
        "type": "auto",
        "num_results": 8,
        "contents": {"highlights": True},
    }
    if end_published_date is not None:
        kwargs["end_published_date"] = end_published_date
    try:
        return await client.search(**kwargs)
    finally:
        await client._client.aclose()


async def _call_exa_search(query: str, end_published_date: str | None) -> list[Any]:
    async with _EXA_GLOBAL_SEMAPHORE:
        for attempt in range(len(_EXA_RETRY_DELAYS_S) + 1):
            try:
                response = await _run_exa_search(query, end_published_date)
                results = _mapping_or_attrs_get(response, "results")
                return list(results) if isinstance(results, list) else []
            except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
                if attempt >= len(_EXA_RETRY_DELAYS_S) or not _is_rate_limited_error(exc):
                    raise
                await asyncio.sleep(_EXA_RETRY_DELAYS_S[attempt])
    return []


def _run_document_read_sync(url: str, ask: str) -> tuple[str, int, list[str]]:
    """Read ``url`` via Gemini url_context. Returns ``(text, n_successful_retrievals, statuses)``.

    A thin binding of this tool's own settings onto the shared reader
    (``research/url_context_reader``), which the Tier-1 resolution-source ladder also calls.
    It stays HERE, with this name, because ``tools.read_document`` resolves
    ``_run_document_read_sync`` as a module attribute the suite monkeypatches — and because the
    timeout arithmetic above is this tool's, not the reader's.

    The retrieval count is returned, not discarded, because the text alone cannot tell a real
    document read from a fluent answer out of parametric memory, and ``read_document`` grants
    the highest verification tier the artifact renderer has.
    """
    api_key = os.getenv(GOOGLE_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"Missing Google API key: {GOOGLE_API_KEY_ENV}")
    return run_url_context_read(
        url,
        ask,
        api_key=api_key,
        role="read_document",
        model=GAP_FILL_V2_READER_MODEL,
        thinking_level=GAP_FILL_V2_READER_THINKING_LEVEL,
        timeout_ms=_READ_DOCUMENT_HTTP_PER_ATTEMPT_TIMEOUT_MS,
        attempts=GAP_FILL_V2_READER_HTTP_ATTEMPTS,
    )
