"""An LLM-extractive digest of a long web page, grounded literally against the page text.

A cited page over the per-URL character cap used to be read from the top, so its tail was
unreachable; the gap-fill loop's deterministic BM25 digest reached the tail but the operator does
not trust a lexical ranker as the primary mechanism (decision of 2026-09-09, recorded in
``scratch_docs_and_planning/fetch_ladder_unification_plan_2026-09-09.md``). So a cheap model reads
the page and returns the passages that bear on the caller's query, verbatim and ranked, and a
literal grounding check drops anything it did not copy. BM25 keeps two jobs: the free pre-filter
that cuts a very long page to a few thousand tokens before the model reads it, and the fallback
when the call fails, returns no grounded passage, or cannot fit inside the caller's remaining wall.
An empty, valid extractor answer is a no-match result and is never replaced with BM25 candidates.
The fallback is exactly the digest both callers shipped before, which is what makes this a
strictly-safer change on the fetch wall.

One entry point, :func:`digest_page`, takes the page text, the caller's query and the remaining
wall in seconds and returns a :class:`PageDigest`. Every call bills the ``page_digest_extractor``
role on the ``CREDIT_ROLE_SPEND`` ledger through the same builder every support role uses. The
three counters the digest carries ride the callers' ``RESOLUTION_SOURCE_FETCH`` marker as optional
tail fields (``passages_returned``, ``passages_grounded``, ``fallback_used``); this module emits no
marker of its own. Detail: docs/research.md "Page digest".
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import openai
from forecasting_tools import GeneralLlm
from pydantic import BaseModel, ValidationError

from metaculus_bot.constants import (
    DOCUMENT_DIGEST_TOP_K,
    DOCUMENT_DIGEST_WINDOW_CHARS,
    PAGE_DIGEST_EXTRACTOR_EFFORT,
    PAGE_DIGEST_EXTRACTOR_MODEL,
    PAGE_DIGEST_EXTRACTOR_TIMEOUT_S,
    PAGE_DIGEST_MIN_CALL_BUDGET_S,
    PAGE_DIGEST_PREFILTER_MAX_CHARS,
    PAGE_DIGEST_WALL_MARGIN_S,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.llm_retry import is_zero_output_failure
from metaculus_bot.research.document_text import Passage, select_passages
from metaculus_bot.research.source_evidence import SourceExcerpt

logger = logging.getLogger(__name__)

# The CREDIT_ROLE_SPEND row every extractor call books under (docs/operations.md "Per-role spend").
PAGE_DIGEST_ROLE = "page_digest_extractor"

# ``PageDigest.method`` values: which mechanism chose the passages a reader sees.
DIGEST_METHOD_LLM_EXTRACTIVE = "llm_extractive"
DIGEST_METHOD_BM25 = "bm25"

# Marks a real cut between pre-filter windows, so the model cannot read across our splice.
_WINDOW_SEPARATOR = "\n[...]\n"

# Deleted on both sides before the substring check: curly quotes retyped straight are still a copy (see provenance).
_QUOTE_CHARACTERS = "\"'\u2018\u2019\u201c\u201d`"
_QUOTE_GLYPHS = re.compile(f"[{_QUOTE_CHARACTERS}]")

DigestOutcome = Literal[
    "answered",
    "valid_empty",
    "ungrounded",
    "malformed",
    "api_error",
    "timeout",
    "empty_completion",
    "insufficient_budget",
    "empty_page",
]
DigestScope = Literal["full_page", "prefiltered_windows"]

# Each clause names a shape the grounding check drops unread (paraphrase, summary, splice); docs/research.md "Page digest".
PAGE_DIGEST_EXTRACTOR_PROMPT = """You are extracting passages from a web page for a forecaster who cannot read the whole page.

Query (what the forecaster needs from this page):
{query}

Return the passages of the page that bear on the query, most relevant first.
- Copy each passage VERBATIM from the page text below, character for character. Do not paraphrase, summarize, translate, correct, or join non-adjacent text: a passage that is not an exact copy is discarded unread.
- A passage is one contiguous span: a sentence, a few sentences, or a block of table rows. Keep every number with its label, date and unit.
- Return at most {max_passages} passages, and an empty list if nothing on the page bears on the query.

Page text:
{page_text}"""


class PageDigestPassages(BaseModel):
    """The extractor's structured answer: the verbatim passages, most relevant first."""

    passages: list[str]


@dataclass(frozen=True)
class PageDigest:
    """What a reader sees of a long page, plus the three counters the fetch marker carries.

    ``passages`` opens with the page's first window when selected passages exist, so a reader still
    sees what the page is, then the accepted passages in rank order. A valid empty answer carries
    no opening and no candidate excerpt. ``passages_returned`` is the
    raw count the model answered with and ``passages_grounded`` how many of those were literal
    substrings of the page, so the difference is the model's fabrication count; both are zero when
    no call was made or the call failed before answering. ``fallback_used`` is True whenever the
    passages after the opening came from BM25 rather than the model, and ``method`` names which.
    """

    passages: list[str]
    passages_returned: int
    passages_grounded: int
    fallback_used: bool
    method: str
    outcome: DigestOutcome
    scope: DigestScope
    evidence_passages: tuple[SourceExcerpt, ...]


async def digest_page(text: str, query: str, *, budget_seconds: float) -> PageDigest:
    """The passages of ``text`` that bear on ``query``, inside ``budget_seconds`` of the caller's wall.

    The BM25 ranking and the page normalisation run in one thread hop first, and the one paid call
    is then bounded by ``min(PAGE_DIGEST_EXTRACTOR_TIMEOUT_S, budget_seconds - elapsed -
    PAGE_DIGEST_WALL_MARGIN_S)`` and not attempted at all below ``PAGE_DIGEST_MIN_CALL_BUDGET_S`` or
    on an empty page, so a caller can hand over its remaining wall verbatim. The expected call
    failures and forecasting-tools' empty-completion ``RuntimeError`` degrade to the BM25 digest
    with ``fallback_used=True``, as does an answer with nothing grounded; a valid empty answer
    returns no passages and is not a fallback; an answer whose grounded
    passages all lie inside the opening returns the opening alone with ``fallback_used=False``;
    any other exception (the ``None``-content ``AssertionError`` included) propagates.
    """
    started = time.monotonic()
    opening = _opening_passage(text)
    ranked, page_norm = await asyncio.to_thread(_rank_and_normalize, text, query)
    call_budget = min(
        PAGE_DIGEST_EXTRACTOR_TIMEOUT_S, budget_seconds - (time.monotonic() - started) - PAGE_DIGEST_WALL_MARGIN_S
    )
    scope: DigestScope = "full_page" if len(text) <= PAGE_DIGEST_PREFILTER_MAX_CHARS else "prefiltered_windows"
    if not text.strip():
        logger.info("PAGE_DIGEST no call (empty page); no passages to serve")
        return PageDigest(
            passages=[],
            passages_returned=0,
            passages_grounded=0,
            fallback_used=False,
            method=DIGEST_METHOD_BM25,
            outcome="empty_page",
            scope="full_page",
            evidence_passages=(),
        )
    if call_budget < PAGE_DIGEST_MIN_CALL_BUDGET_S:
        logger.info(f"PAGE_DIGEST no call ({call_budget:.1f} s left, under the call floor); serving BM25 candidates")
        return _bm25_digest(ranked, opening=opening, outcome="insufficient_budget", scope=scope)

    prompt = PAGE_DIGEST_EXTRACTOR_PROMPT.format(
        query=query, max_passages=DOCUMENT_DIGEST_TOP_K, page_text=_prefiltered_text(text, ranked)
    )
    try:
        raw = await asyncio.wait_for(_build_extractor().invoke(prompt), timeout=call_budget)
        returned = PageDigestPassages.model_validate_json(raw).passages
    except (TimeoutError, openai.APIError, ValidationError) as exc:
        logger.warning(f"PAGE_DIGEST extractor call failed ({type(exc).__name__}); serving the BM25 digest")
        return _bm25_digest(ranked, opening=opening, outcome=_call_failure_outcome(exc), scope=scope)
    except RuntimeError as exc:
        if not is_zero_output_failure(exc):  # forecasting-tools raises a bare RuntimeError for an empty completion
            raise
        logger.warning("PAGE_DIGEST extractor returned an empty completion; serving the BM25 digest")
        return _bm25_digest(ranked, opening=opening, outcome="empty_completion", scope=scope)

    grounded, kept, evidence = _ground(returned, text, page_norm, opening)
    if not returned:
        return PageDigest(
            passages=[],
            passages_returned=0,
            passages_grounded=0,
            fallback_used=False,
            method=DIGEST_METHOD_LLM_EXTRACTIVE,
            outcome="valid_empty",
            scope=scope,
            evidence_passages=(),
        )
    if not grounded:
        logger.warning(f"PAGE_DIGEST nothing grounded ({len(returned)} returned); serving BM25 candidates")
        return _bm25_digest(ranked, opening=opening, outcome="ungrounded", scope=scope, passages_returned=len(returned))
    return PageDigest(
        passages=_with_opening(opening, kept[:DOCUMENT_DIGEST_TOP_K]),
        passages_returned=len(returned),
        passages_grounded=grounded,
        fallback_used=False,
        method=DIGEST_METHOD_LLM_EXTRACTIVE,
        outcome="answered",
        scope=scope,
        evidence_passages=tuple(
            SourceExcerpt(source_url="", text=passage, retrieval_method=DIGEST_METHOD_LLM_EXTRACTIVE)
            for passage in evidence[:DOCUMENT_DIGEST_TOP_K]
        ),
    )


def _call_failure_outcome(exc: TimeoutError | openai.APIError | ValidationError) -> DigestOutcome:
    """Each is an outcome a healthy deployment produces, and each has one remedy: the free BM25 digest."""
    match exc:
        case TimeoutError():  # asyncio.wait_for at the call budget, so the caller's wall never slips
            return "timeout"
        case openai.APIError():  # the root of every litellm provider and API error (llm_retry.llm_status_code)
            return "api_error"
        case ValidationError():  # the model answered off-schema, and a second paid attempt cannot fit inside the wall
            return "malformed"


def _build_extractor() -> GeneralLlm:
    return build_llm_with_openrouter_fallback(
        model=PAGE_DIGEST_EXTRACTOR_MODEL,
        role=PAGE_DIGEST_ROLE,
        reasoning={"effort": PAGE_DIGEST_EXTRACTOR_EFFORT},
        timeout=PAGE_DIGEST_EXTRACTOR_TIMEOUT_S,
        allowed_tries=1,  # the BM25 fallback is the retry; a second paid attempt cannot fit inside the wall
        response_format=PageDigestPassages,
        extra_body={"provider": {"require_parameters": True}},  # OpenRouter rejects, rather than drops, the schema
    )


def _rank_and_normalize(text: str, query: str) -> tuple[list[Passage], str]:
    """The CPU-bound work, done once off the event loop: the BM25 ranking both paths read, and the page as the grounding check reads it."""
    ranked = select_passages(text, query, top_k=PAGE_DIGEST_PREFILTER_MAX_CHARS // DOCUMENT_DIGEST_WINDOW_CHARS)
    return ranked, _normalize(text)


def _prefiltered_text(text: str, ranked: list[Passage]) -> str:
    """What the model reads: the whole page under the cap, else its best BM25 windows spliced in page order."""
    if len(text) <= PAGE_DIGEST_PREFILTER_MAX_CHARS:
        return text
    if not ranked:  # no query token anywhere on the page: the head is what a reader saw before
        return text[:PAGE_DIGEST_PREFILTER_MAX_CHARS]
    windows = sorted(ranked, key=lambda window: window.start)
    parts = [windows[0].text]
    for previous, window in pairwise(windows):
        between = text[previous.end : window.start]
        parts.append(between if not between.strip() else _WINDOW_SEPARATOR)
        parts.append(window.text)
    return "".join(parts)


def _ground(returned: list[str], page_text: str, page_norm: str, opening: str) -> tuple[int, list[str], list[str]]:
    """Ground passages literally and return exact source slices for evidence."""
    opening_norm = _normalize(opening)
    grounded = 0
    kept: list[str] = []
    evidence: list[str] = []
    seen: set[str] = set()
    for passage in returned:
        candidate = _normalize(passage)
        if not candidate or candidate not in page_norm:
            continue
        grounded += 1
        if candidate in seen:
            continue
        seen.add(candidate)
        source_passage = _source_slice_for_normalized_match(page_text, candidate)
        evidence.append(source_passage)
        if candidate not in opening_norm:
            kept.append(source_passage)
    return grounded, kept, evidence


def _bm25_digest(
    ranked: list[Passage],
    *,
    opening: str,
    outcome: DigestOutcome,
    scope: DigestScope,
    passages_returned: int = 0,
) -> PageDigest:
    """The deterministic candidates from the whole-page ranking, with the model outcome retained."""
    opening_norm = _normalize(opening)
    kept = [window.text for window in ranked[:DOCUMENT_DIGEST_TOP_K] if _normalize(window.text) not in opening_norm]
    return PageDigest(
        passages=_with_opening(opening, kept),
        passages_returned=passages_returned,
        passages_grounded=0,
        fallback_used=True,
        method=DIGEST_METHOD_BM25,
        outcome=outcome,
        scope=scope,
        evidence_passages=tuple(
            SourceExcerpt(source_url="", text=window.text, retrieval_method="bm25_candidate")
            for window in ranked[:DOCUMENT_DIGEST_TOP_K]
        ),
    )


def _source_slice_for_normalized_match(page_text: str, candidate: str) -> str:
    """The exact slice of ``page_text`` that ``_normalize`` turns into text containing ``candidate``.

    Mirrors ``_normalize`` one character at a time, remembering each kept character's source index.
    ``candidate`` is itself normalized, so it never starts or ends on a space or a quote glyph.
    """
    normalized_chars: list[str] = []
    source_indices: list[int] = []
    for index, char in enumerate(page_text):
        if char in _QUOTE_CHARACTERS:
            continue
        if char.isspace() and (not normalized_chars or normalized_chars[-1] == " "):
            continue
        normalized_chars.append(" " if char.isspace() else char)
        source_indices.append(index)
    offset = "".join(normalized_chars).find(candidate)
    if offset < 0:
        raise RuntimeError("grounded passage was not found in normalized source text")
    return page_text[source_indices[offset] : source_indices[offset + len(candidate) - 1] + 1]


def _opening_passage(text: str) -> str:
    """The page's first digest window, cut at its last whitespace so it never ends mid-token."""
    head = text[:DOCUMENT_DIGEST_WINDOW_CHARS]
    if len(text) > DOCUMENT_DIGEST_WINDOW_CHARS and (cut := re.search(r"\s\S*$", head)):
        head = head[: cut.start()]
    return head.strip()


def _with_opening(opening: str, passages: list[str]) -> list[str]:
    return [opening, *passages] if opening else list(passages)


def _normalize(text: str) -> str:
    return " ".join(_QUOTE_GLYPHS.sub("", text).split())
