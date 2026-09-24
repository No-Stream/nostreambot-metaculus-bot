"""Gemini grounded search research provider.

Uses the `google-genai` SDK directly (NOT via OpenRouter) so we get real
first-party Google Search grounding rather than OpenRouter's Exa-backed web
plugin. This adds a genuinely distinct search index to the ensemble — the
Metaculus Fall 2025 writeup identified research breadth as the single
strongest predictor of winning bots.

Mirrors `_native_search_provider` in `research_providers.py` for consistency.
"""

import asyncio
import functools
import logging
import os
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from forecasting_tools.data_models.questions import MetaculusQuestion
from google import genai
from google.genai import types as genai_types

from metaculus_bot.constants import (
    GEMINI_SEARCH_DEFAULT_MODEL,
    GEMINI_SEARCH_HTTP_ATTEMPTS,
    GEMINI_SEARCH_HTTP_TIMEOUT_MS,
    GEMINI_SEARCH_LINK_RESOLVE_TIMEOUT_S,
    GEMINI_SEARCH_MODEL_ENV,
    GEMINI_SEARCH_THINKING_LEVEL,
    GEMINI_SEARCH_TIMEOUT,
    GOOGLE_API_KEY_ENV,
)
from metaculus_bot.prompts import web_research_prompt
from metaculus_bot.research.bracket_groups import (
    BRACKET_GROUP_RE,
    iter_group_items,
    join_group_items,
    rebuild_group,
)
from metaculus_bot.research.gemini_attribution import rewrite_unsupported_attributions
from metaculus_bot.research.gemini_client_config import build_gemini_http_options, gemini_thinking_config
from metaculus_bot.research.gemini_usage import log_gemini_usage
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.raw_log import record_raw_research
from metaculus_bot.research.search_redirects import (
    SEARCH_REDIRECT_HOST,
    SEARCH_REDIRECT_PATH_PREFIX,
    is_search_redirect,
    resolve_search_redirects,
)
from metaculus_bot.research.url_context_telemetry import (
    URL_RETRIEVAL_SUCCESS,
    extract_url_context_telemetry,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_gemini_client",
    "extract_url_context_telemetry",
    "gemini_search_provider",
    "invoke_gemini_grounded",
]

_MARKDOWN_LINK_RE = re.compile(r"\[(?P<label>(?:[^\[\]]|\[[^\[\]]*\])*)\]\((?P<url>https?://[^)\s]+)\)")
_RAW_SEARCH_REDIRECT_RE = re.compile(
    rf"https?://{re.escape(SEARCH_REDIRECT_HOST)}{re.escape(SEARCH_REDIRECT_PATH_PREFIX)}\S+"
)


@functools.lru_cache(maxsize=1)
def _cached_client_for_key(api_key: str) -> genai.Client:
    """Process-global cached genai.Client keyed on API key.

    SDK clients are designed to be long-lived; keeping one across a backtest
    lets TLS connections and HTTP/2 multiplexing be reused across the ~thousands
    of calls the Gemini provider + gap-fill make per round. Keyed on api_key so
    a rotated key (rare) produces a fresh client.

    The retry ladder rides on the CLIENT rather than the per-request options because the
    SDK builds its tenacity retryer once at construction from
    ``http_options.retry_options``; a bare client stops after one attempt (see
    ``research/gemini_client_config``).
    """
    return genai.Client(
        api_key=api_key,
        http_options=build_gemini_http_options(
            timeout_ms=GEMINI_SEARCH_HTTP_TIMEOUT_MS, attempts=GEMINI_SEARCH_HTTP_ATTEMPTS
        ),
    )


def build_gemini_client() -> genai.Client:
    """Return the cached google-genai Client for the operator's personal Gemini key.

    Reads GOOGLE_API_KEY (the operator's personal Google AI Studio key — in CI
    populated from ``secrets.GEMINI_API_KEY``). There is no Metaculus-donated
    Gemini key on the google-genai side; the donated path only exists for
    OpenRouter-routed Gemini models. Raises ValueError if the key is missing
    so misconfiguration is loud.
    """
    api_key = os.getenv(GOOGLE_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{GOOGLE_API_KEY_ENV} must be set to use the Gemini search provider")
    return _cached_client_for_key(api_key)


def _resolve_model(model_slug: str | None) -> str:
    return model_slug or os.getenv(GEMINI_SEARCH_MODEL_ENV, GEMINI_SEARCH_DEFAULT_MODEL)


_URL_CONTEXT_NONE_MARKER = "_url_context: none_"
_URL_CONTEXT_HEADER = "### URL Context Fetches"


def _format_url_context_marker(reported: bool, entries: list[tuple[str, str]]) -> str:
    """Build the greppable url_context telemetry block appended to persisted research.

    Only SUCCESSFUL fetches are listed inline (under ``### URL Context Fetches``) — those URLs were
    genuinely read, so they are real research context. Any other reported state (fired but fetched
    nothing, or every retrieval failed) collapses to the terse ``_url_context: none_`` marker, so a
    'did nothing useful' run never pushes failed/dead URLs at the forecaster. No url_context signal
    at all → empty string (no marker). Failed-fetch URLs are still captured in the INFO logs for
    auditing, just not in the forecaster-facing research blob.
    """
    successes = [
        (status, url)
        for status, url in entries
        if status == URL_RETRIEVAL_SUCCESS and url and not is_search_redirect(url)
    ]
    if successes:
        lines = ["", "", _URL_CONTEXT_HEADER]
        lines.extend(f"{status} — {url}" for status, url in successes)
        return "\n".join(lines)
    if reported:
        return f"\n\n{_URL_CONTEXT_NONE_MARKER}"
    return ""


# Gemini writes its OWN hierarchical citation indices — ``[2.4.1]``, ``[1.1.1, 1.1.2]``,
# ``[A: NASA, 1.1.2]`` — indexing a source list that does not exist on our side, alongside the
# resolvable ``[N]`` markers this formatter adds to verified search links.
# 173 of 323 archived sections carry them and 163 carry BOTH families, so half the corpus hands a
# forecaster a bracket field where some brackets resolve and some are decoration, with nothing to
# tell them apart (scratch/residual_2026-08-31/gemini_search_audit/cutB_pattern.md §3.1).
#
# A dotted run only counts as an index when it is DELIMITED the way a citation is — sitting at a
# group edge or against whitespace/``,``/``;``/``:`` — and when every component is at most two
# digits. Both bounds are measured, not guessed: across the 2,609 archived bracket groups that
# hold a dotted token, first components run 1..6 and the largest component anywhere is 39. The
# two-digit bound is therefore comfortably above real indices while excluding the content classes
# that would otherwise match — a year (``[2026.08]``), an IP octet (``[192.168.1.1]``) — and the
# delimiter rule excludes a quantity (``[3.8%]``, ``[$1.5]``, ``[1.5 million]``) and a version
# (``[v2.1.3]``). Zero of those 2,609 groups is anything but a citation index (validation:
# scratch/next_season_bundle_2026-09/item3_citation_strip/VALIDATION.md).
#
# What a bracket group IS, where its items split, and how a rewritten one is put back
# together all come from ``research/bracket_groups.py`` — the same grammar the attribution
# check reads the same text through immediately after this pass, so the two cannot come to
# disagree about one string. Only the index token itself is this pass's own.
_CITATION_INDEX_RE = re.compile(r"(?<![^\s,;:])\d{1,2}(?:\.\d{1,2})+(?=\s*(?:[,;:]|$))")


def _tidy_group_item(item: str) -> str:
    """Normalize one comma/semicolon item of a bracket group after index removal.

    Drops the separator a removed index left behind (``2.1.4: A`` -> ``A``,
    ``A: official 2.4.1`` -> ``A: official``) and collapses the whitespace it opened up.
    """
    return re.sub(r"\s+", " ", item).strip().strip(":").strip()


def _strip_model_citation_indices(text: str) -> str:
    """Remove Gemini's self-authored hierarchical citation indices from bracket groups.

    Runs after the link rewrite. Our markers are plain integers, so they survive this pass
    untouched; only dotted model-authored runs go.

    A group emptied of everything but punctuation is removed along with one preceding space, so
    ``"office [1.1.1, 1.1.2]. He"`` reads ``"office. He"`` rather than ``"office . He"``.
    Idempotent: a second pass finds no qualifying token.
    """

    def replace(match: re.Match[str]) -> str:
        inner = match.group("inner")
        stripped_inner = _CITATION_INDEX_RE.sub("", inner)
        if stripped_inner == inner:
            return match.group(0)
        # An item that is nothing but punctuation once its index is gone said only the
        # index; dropping it (rather than emptying it) is this pass's own filter, which is
        # why ``iter_group_items`` hands items over raw.
        kept: list[tuple[str, str]] = []
        for separator, item in iter_group_items(stripped_inner):
            tidied = _tidy_group_item(item)
            if any(char.isalnum() for char in tidied):
                kept.append((separator, tidied))
        if not kept:
            return ""
        return rebuild_group(match, join_group_items(kept))

    return BRACKET_GROUP_RE.sub(replace, text)


def _grounded_source_labels(sources: Sequence[tuple[int, str, str]]) -> list[str]:
    """Return the domains represented by the numbered verified source list."""
    return [domain for _number, domain, _url in sources]


def _render_sources_section(sources: Sequence[tuple[int, str, str]]) -> str:
    """Render the trailing ``### Sources`` block from numbered, resolved targets."""
    if not sources:
        return ""
    lines = ["", "", "### Sources"]
    lines.extend(f"[{number}] {domain} — {url}" for number, domain, url in sources)
    return "\n".join(lines)


def _check_attributions(text: str, sources: Sequence[tuple[int, str, str]], *, qid: int | None) -> str:
    """Mark the tier-tag attributions this response's own grounding record cannot back.

    Runs AFTER the citation-index strip, on the annotated body only (the ``### Sources``
    block is appended afterwards and never passes through), and only where we have
    renderable grounded labels to compare against — an empty label list is a measurement
    failure rather than a verdict, so it leaves every tag standing and records nothing.
    That absence is the signal: on a schema-v2 record, no ``unsupported_attributions``
    count means the check had no evidence base (or the record predates it), while a
    recorded 0 means it ran and found nothing.

    Deliberately NOT alertable and nothing keys on the count: 70% of the archived corpus's
    outlet-named tier tags are unsupported, so this is the model's habitual embellishment
    rather than a bot defect, and an absent outlet does not make the FACT wrong.
    """
    labels = _grounded_source_labels(sources)
    if not labels:
        return text
    checked = rewrite_unsupported_attributions(text, labels)
    # ``tier_tags`` rides alongside because the count this check exists to report is not
    # readable without it: the marker below is gated on ``unsupported``, so a response that
    # carried no tier tags at all and one whose every tag was backed both archive as
    # ``unsupported_attributions=0`` and log nothing. It counts OUTLET-NAMED tier items only;
    # tags naming a class of source rather than an outlet ("official", "peer-reviewed
    # journal") are rewritten too but counted apart as ``generic_tier_tags``, so the two
    # together are every checked tier item.
    record_provider_detail(
        qid,
        "gemini_search",
        {
            "counts": {
                "tier_tags": checked.tagged,
                "generic_tier_tags": checked.generic,
                "unsupported_attributions": checked.unsupported,
            }
        },
    )
    if checked.unsupported:
        # ``labels`` rides the line because the same count reads completely differently
        # against it: q38195 named 21 outlets over ONE verified domain. ``generic`` is
        # appended last (2026-09-24) so the fields before it keep their positions.
        logger.info(
            f"GEMINI_UNSUPPORTED_ATTRIBUTION: question={qid} tagged={checked.tagged} "
            f"unsupported={checked.unsupported} groups={checked.groups_rewritten} labels={len(labels)} "
            f"generic={checked.generic}"
        )
    return checked.text


def _rewrite_cited_links(
    text: str,
    matches: Sequence[re.Match[str]],
    resolved_redirects: dict[str, str],
    read_urls: set[str],
) -> tuple[str, list[tuple[int, str, str]], set[str], set[str]]:
    """Replace cited links and return text, numbered sources, verified URLs, and unverified URLs."""
    source_numbers: dict[str, int] = {}
    sources: list[tuple[int, str, str]] = []
    verified_urls: set[str] = set()
    unverified_urls: set[str] = set()
    replacements: list[tuple[int, int, str]] = []
    for match in matches:
        cited_url = match.group("url")
        if is_search_redirect(cited_url):
            target = resolved_redirects.get(cited_url)
        elif cited_url in read_urls:
            target = cited_url
        else:
            target = None
        if target is None:
            unverified_urls.add(cited_url)
            replacement = f"{match.group('label')} [unverified link]"
        else:
            verified_urls.add(cited_url)
            number = source_numbers.get(target)
            if number is None:
                number = len(sources) + 1
                source_numbers[target] = number
                hostname = urlsplit(target).hostname or target
                sources.append((number, hostname.removeprefix("www."), target))
            replacement = f"{match.group('label')} [{number}]"
        replacements.append((match.start(), match.end(), replacement))
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return _RAW_SEARCH_REDIRECT_RE.sub("[unverified link]", text), sources, verified_urls, unverified_urls


def _format_grounded_response(
    response: genai_types.GenerateContentResponse,
    resolved_redirects: dict[str, str] | None = None,
    *,
    qid: int | None = None,
    model: str | None = None,
) -> str:
    """Rewrite self-cited links, enforce the verification floor, and render resolved sources.

    Output format:
        <response text>

        ### Sources
        [1] <domain> — <resolved URL>
        [2] <domain> — <resolved URL>
        ...

    A response with no verified cited links is suppressed as ungrounded parametric output.
    ``qid`` and ``model`` make the suppression warning and self-citation marker greppable.
    """
    text = response.text or ""
    if not text:
        return ""

    candidates = response.candidates
    metadata = candidates[0].grounding_metadata if candidates else None
    _reported, _n_url_total, _n_url_success, url_entries = extract_url_context_telemetry(response)
    read_urls = {
        url for status, url in url_entries if status == URL_RETRIEVAL_SUCCESS and url and not is_search_redirect(url)
    }
    matches = list(_MARKDOWN_LINK_RE.finditer(text))
    cited_urls = [match.group("url") for match in matches]
    text, sources, verified_urls, unverified_urls = _rewrite_cited_links(
        text, matches, resolved_redirects or {}, read_urls
    )

    logger.info(
        f"GEMINI_SELF_CITATION: question={qid} model={model} links={len(cited_urls)} "
        f"unique={len(set(cited_urls))} resolved={len(verified_urls)} "
        f"unverified={len(unverified_urls)} sources={len(sources)}"
    )
    if not verified_urls:
        n_queries = len(metadata.web_search_queries or []) if metadata is not None else 0
        logger.warning(f"GEMINI_UNGROUNDED_SUPPRESSED: question={qid} model={model} queries={n_queries}")
        record_provider_detail(qid, "gemini_search", {"sources": {"grounding": "error(ungrounded_suppressed)"}})
        return ""
    annotated = _strip_model_citation_indices(text)
    return _check_attributions(annotated, sources, qid=qid) + _render_sources_section(sources)


async def _generate_grounded(
    client: genai.Client,
    model: str,
    prompt: str,
    config: genai_types.GenerateContentConfig,
    *,
    deadline: float,
    qid: int | None,
) -> genai_types.GenerateContentResponse:
    """One grounded call inside what is left of the shared wall, with its spend line and raw record."""
    remaining_s = deadline - asyncio.get_running_loop().time()
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(model=model, contents=prompt, config=config),
            timeout=remaining_s,
        )
    except TimeoutError:
        logger.warning(
            f"GeminiSearch: {model} timed out after {max(remaining_s, 0.0):.0f}s "
            f"(what remained of the {GEMINI_SEARCH_TIMEOUT}s wall)"
        )
        raise

    # Before any formatting branch, so the tokens are recorded on the suppressed-response
    # paths too: an ungrounded response we refuse to publish was billed exactly like a
    # useful one, and a spend line that only covers the responses we kept would understate
    # the bill by precisely the wasted calls.
    log_gemini_usage(
        response,
        role="grounded_search",
        model=model,
        question=str(qid) if qid is not None else None,
    )

    # Capture the raw SDK response (text + grounding metadata: the actual Google
    # queries and sources) before formatting drops most of it.
    record_raw_research(qid=qid, provider="gemini_search", payload=response)
    return response


async def _resolve_cited_redirects(response: genai_types.GenerateContentResponse, *, deadline: float) -> dict[str, str]:
    """Resolve cited Google redirect URLs inside the remaining provider wall."""
    remaining_s = deadline - asyncio.get_running_loop().time()
    if remaining_s <= 0:
        return {}
    cited_urls = [match.group("url") for match in _MARKDOWN_LINK_RE.finditer(response.text or "")]
    redirect_urls = [url for url in cited_urls if is_search_redirect(url)]
    return await resolve_search_redirects(
        redirect_urls, timeout_s=min(GEMINI_SEARCH_LINK_RESOLVE_TIMEOUT_S, remaining_s)
    )


async def invoke_gemini_grounded(
    prompt: str,
    *,
    model_slug: str | None = None,
    include_url_context: bool = True,
    qid: int | None = None,
) -> str:
    """Invoke Gemini with Google Search grounding and return formatted text.

    Used by the first-pass Gemini search provider (and the ablation harness);
    gap-fill uses OpenAI native search, not this google-genai grounded path.
    Enables the URL context tool alongside Google Search by default so the model
    can directly read specific URLs (e.g., resolution sources named in question
    fine print).

    Raises on SDK errors — callers decide whether to fail hard or soft.
    """
    client = build_gemini_client()
    model = _resolve_model(model_slug)

    tools: list[Any] = [{"google_search": {}}]
    if include_url_context:
        tools.append({"url_context": {}})

    # Thinking level is set explicitly rather than left at the model's default (HIGH on
    # gemini-3-flash-preview), which was most of this provider's token bill; see
    # GEMINI_SEARCH_THINKING_LEVEL. Still no max_tokens — capping output on a thinking
    # model truncates.
    config = genai_types.GenerateContentConfig(
        tools=tools,
        thinking_config=gemini_thinking_config(GEMINI_SEARCH_THINKING_LEVEL),
    )

    logger.info(f"GeminiSearch: calling {model} with grounding")
    deadline = asyncio.get_running_loop().time() + GEMINI_SEARCH_TIMEOUT
    response = await _generate_grounded(client, model, prompt, config, deadline=deadline, qid=qid)
    resolved_redirects = await _resolve_cited_redirects(response, deadline=deadline)
    formatted = _format_grounded_response(response, resolved_redirects, qid=qid, model=model)

    reported, n_url_total, n_url_success, url_entries = extract_url_context_telemetry(response)
    logger.info(
        f"GeminiSearch: got {len(formatted)} chars, {n_url_success}/{n_url_total} url_context fetches from {model}"
    )
    if url_entries:
        for status, url in url_entries:
            logger.info(f"GeminiSearch: url_context {status} — {url}")

    # Only annotate non-empty research; an empty result must stay empty so callers can soft-fail.
    if formatted:
        formatted += _format_url_context_marker(reported, url_entries)
    return formatted


def gemini_search_provider(
    model_slug: str | None = None,
    is_benchmarking: bool = False,
) -> ResearchCallable:
    """Research provider using Gemini with Google Search grounding.

    Mirrors the `_native_search_provider` contract (`MetaculusQuestion -> str`).
    """

    async def _fetch(question: MetaculusQuestion) -> str:
        prompt = web_research_prompt(
            question.question_text,
            # The MC ballot (None on other types): grounded search can only query candidate
            # names it has been shown (q44952 — zero retrieval on the eventual winner).
            options=getattr(question, "options", None),
            is_benchmarking=is_benchmarking,
            citation_style="search_links",
            allow_resolution_source_reading=True,
        )
        return await invoke_gemini_grounded(
            prompt, model_slug=model_slug, qid=getattr(question, "id_of_question", None)
        )

    return _fetch
