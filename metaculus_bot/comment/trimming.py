"""Trim an assembled Metaculus comment, and its individual sections, to the platform's char limits.

Two entry points. ``trim_section`` shrinks one standalone section to its own budget
(``_section_budget`` maps the section name to it); the publish path no longer calls it. ``trim_comment``
shrinks the whole assembled comment through an ordered strategy chain. A structured comment with
rationale blocks goes through ``_trim_complete_comment``: optional research paragraphs first (every
disclosure block kept whole), then summary prose, then the middle of each rationale, and when even
that cannot fit, the explicitly incomplete explanation (``_incomplete_explanation``). Other shapes
shrink ``# RESEARCH`` alone (``_trim_research_section_first``), else keep the summary head and the
tail (``_trim_preserving_summary_and_tail``), else fall back to a plain header-preserving trim
(``_trim_with_notice``).

Two invariants every path holds. The output keeps its leading ``#``, which the publish validator
requires. And the bytes ``performance_analysis`` later parses out of the published comment survive:
each rationale's ``Model: openrouter/...`` attribution line and its trailing fenced json forecast
block, the per-model ``*Forecaster N*:`` bullets in the summary, the trailing ``STACKED=<bool>``
marker, and the provider-diagnostics ``lost=`` token. That is why the over-budget paths trim from
*within* each rationale block rather than keeping a header plus one long tail: a naive
header-and-tail trim ate Forecaster 1's ``Model:`` line in 29 of 29 measured July 2026 trims.

The section budgets, that measurement, and the stacker-combined misattribution this file exists to
prevent: docs/architecture.md "Comment trimming".
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import batched, pairwise
from typing import Final

from metaculus_bot.comment.markers import (
    BASE_MODEL_SUBBLOCK_SPLIT_RE,
    EXPLANATION_DETAIL_UNKNOWN_MARKER,
    STACKED_BASE_REASONING_HEADER,
)
from metaculus_bot.constants import (
    COMMENT_CHAR_LIMIT,
    FORECASTS_SECTION_CHAR_LIMIT,
    RESEARCH_SECTION_CHAR_LIMIT,
    SUMMARY_SECTION_CHAR_LIMIT,
)
from metaculus_bot.prompts import SUMMARIZER_SOFT_FAIL_BANNER
from metaculus_bot.research.agentic.artifact import PENDING_LEADS_END_MARKER, PENDING_LEADS_START_MARKER
from metaculus_bot.research.provider_diagnostics import PROVIDER_DIAGNOSTICS_HEADER

logger = logging.getLogger(__name__)

TRIM_NOTICE: Final[str] = "[... trimmed for length]"

# The h1 headers ForecastBot._create_comment emits; the section-aware trim splits on them.
_RESEARCH_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^# RESEARCH$", re.MULTILINE)
_FORECASTS_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^# FORECASTS$", re.MULTILINE)
_TRAILING_COMMENT_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?m)^[ \t]*(<!--\s*(?:STACKER_OUTCOME=[^>]+|STACKER_SKIP_REASON=[^>]+|STACKED=(?:true|false)"
    r"|TOOLS_USED=(?:true|false)|FORECASTERS_USED=\d+/\d+|EXPLANATION_DETAIL=unknown)\s*-->)[ \t]*$"
)
_FINAL_PREDICTION_LINE_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^\*Final Prediction\*:[ \t]*(.*)$")
_SUMMARY_METADATA_LINE_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^\*(?:Total Cost|Time Spent|LLMs|Bot Name)\*:")
_CORRECTION_DISCLOSURE_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?m)^#{3,8} ⚠ Driver-reported corrections[^\n]*$"
)
_RESEARCH_SUBSECTION_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^#{2,8} \S.*$")
_RESEARCH_PARAGRAPH_DELIMITER_RE: Final[re.Pattern[str]] = re.compile(r"\n[ \t]*\n")
_PROVIDER_DIAGNOSTICS_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    rf"(?m)^#{{2,8}} {re.escape(PROVIDER_DIAGNOSTICS_HEADER.split(' ', maxsplit=1)[1])}$"
)
_RESEARCH_COLLECTION_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^## Report \d+ Research[ \t]*$")
_PENDING_LEADS_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    rf"{re.escape(PENDING_LEADS_START_MARKER)}|{re.escape(PENDING_LEADS_END_MARKER)}"
)
_SUMMARIZER_SOFT_FAIL_BANNER_RE: Final[re.Pattern[str]] = re.compile(re.escape(SUMMARIZER_SOFT_FAIL_BANNER))

# The framework emits this right after the per-model bullets the residual parsers read.
_SUMMARY_END_MARKER: Final[str] = "### Research Summary"

_FORECAST_IN_REPORT_LINE: Final[str] = "*Final Prediction*: the full forecast remains in the published report."
_DETAIL_OMITTED_SENTENCE: Final[str] = (
    "Rationale detail was omitted for length; missing model attribution and reasoning are unknown."
)

# Fits the summary with margin, leaving most of the budget to the tail; see docs/architecture.md.
_COMMENT_HEAD_BUDGET: Final[int] = 10_000


@dataclass(frozen=True, slots=True)
class TrimConfig:
    notice: str = TRIM_NOTICE
    # Fallback for unknown section names: the most permissive budget, so a bare trim never over-trims.
    section_limit: int = FORECASTS_SECTION_CHAR_LIMIT
    summary_limit: int = SUMMARY_SECTION_CHAR_LIMIT
    research_limit: int = RESEARCH_SECTION_CHAR_LIMIT
    forecasts_limit: int = FORECASTS_SECTION_CHAR_LIMIT
    comment_limit: int = COMMENT_CHAR_LIMIT
    summary_end_marker: str = _SUMMARY_END_MARKER
    head_budget: int = _COMMENT_HEAD_BUDGET


@dataclass(frozen=True, slots=True)
class _ResearchSegment:
    start: int
    text: str
    required: bool


def _trim_with_notice(text: str, limit: int, notice: str, *, preserve_header: bool) -> tuple[str, bool]:
    if limit <= 0:
        return "", bool(text)
    if len(text) <= limit:
        return text, False
    if limit <= len(notice):
        return notice[:limit], True

    if preserve_header:
        # clean_indents output opens with a newline, which would leave an empty header and lose the "#".
        text = text.lstrip("\n")
        header, separator, remainder = text.partition("\n")
        if separator:
            available = limit - len(header) - len(notice) - 2
            if available > 0:
                tail = remainder[-available:]
                return f"{header}\n{notice}\n{tail}", True
            truncated_header = header[: max(0, limit - len(notice) - 1)]
            if truncated_header:
                return f"{truncated_header}\n{notice}", True
            # Header alone overflows; truncate it but keep the leading "#".
            return text[:limit], True
        # Single-line blob: truncate from the front so the leading character survives.
        if limit > len(notice) + 1:
            return f"{text[: limit - len(notice) - 1]}\n{notice}", True
        return text[:limit], True

    tail_available = limit - len(notice) - 1
    if tail_available <= 0:
        return notice[:limit], True
    tail = text[-tail_available:]
    return f"{notice}\n{tail}", True


# The report number is always 1 in prod, but the pattern stays robust to multi-report comments.
_RATIONALE_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^##\s+R\d+:\s+Forecaster\s+\d+\s+Reasoning[ \t]*$")

# Byte-stable because performance_analysis/comment_sections.py keys on it; see docs/architecture.md.
_MODEL_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^Model:[ \t]*[^\n]*$")

# Ends every rationale and carries the values the residual pipeline parses, so trims keep it.
_JSON_BLOCK_RE: Final[re.Pattern[str]] = re.compile(r"```json\b.*?```", re.DOTALL)


def _pin_summarizer_banner(text: str, limit: int, notice: str) -> tuple[str, bool]:
    """Header, degradation banner and trimmed tail, for a research section carrying
    the summarizer soft-fail banner.

    The plain header-preserving trim keeps header plus TAIL, which drops the banner,
    and a soft-fail is itself what pushes a bundle over the research budget, so the
    disclosure was most likely dropped precisely on the questions where it fired.
    The banner is located by search rather than assumed to lead the body, because
    the orchestrator prepends it to the AskNews provider's own text. Returns
    ``(text, False)`` when the section carries no banner or the budget cannot seat
    header, banner and notice, so the caller falls back to the plain trim. Why the
    pin at all: docs/architecture.md "Comment trimming".
    """
    stripped = text.lstrip("\n")
    header, separator, remainder = stripped.partition("\n")
    if not separator:
        return text, False
    banner_idx = remainder.find(SUMMARIZER_SOFT_FAIL_BANNER)
    if banner_idx < 0:
        return text, False

    after_banner = remainder[banner_idx + len(SUMMARIZER_SOFT_FAIL_BANNER) :]
    # header + \n + banner + \n + notice + \n + tail -> 3 joining newlines.
    tail_available = limit - len(header) - len(SUMMARIZER_SOFT_FAIL_BANNER) - len(notice) - 3
    if tail_available <= 0:
        return text, False
    return f"{header}\n{SUMMARIZER_SOFT_FAIL_BANNER}\n{notice}\n{after_banner[-tail_available:]}", True


def _section_budget(section_name: str, cfg: TrimConfig) -> tuple[int, bool]:
    """Map a comment section name to its ``(char_limit, block_aware)`` policy.

    The framework's unified comment is assembled from three sections whose
    ``trim_section`` names end in ``_summary`` / ``_research`` / ``_rationales``
    (see forecaster.py and comment.formatting). Each gets its own budget so the
    parser-critical FORECASTS (rationales) section is not starved by a uniform
    cap. The rationales section is additionally trimmed block-by-block
    (``block_aware=True``) so per-forecaster ``Model:`` attribution and JSON
    forecast blocks survive an overflow. Unknown names fall back to the generic
    per-section default.
    """
    if section_name.endswith("_rationales"):
        return cfg.forecasts_limit, True
    if section_name.endswith("_research"):
        return cfg.research_limit, False
    if section_name.endswith("_summary"):
        return cfg.summary_limit, False
    return cfg.section_limit, False


def trim_section(text: str, section_name: str, *, config: TrimConfig | None = None) -> str:
    cfg = config or TrimConfig()
    limit, block_aware = _section_budget(section_name, cfg)
    if block_aware:
        trimmed, did_trim = _trim_rationales_within_blocks(text, limit, cfg.notice)
    else:
        # The summarizer-degradation banner is pinned ahead of the tail; every other shape falls through.
        trimmed, did_trim = _pin_summarizer_banner(text, limit, cfg.notice) if len(text) > limit else (text, False)
        if not did_trim:
            trimmed, did_trim = _trim_with_notice(text, limit, cfg.notice, preserve_header=True)
    if did_trim:
        logger.warning(
            "Trimmed section '%s' from %s to %s characters",
            section_name,
            len(text),
            len(trimmed),
        )
    return trimmed


def _allocate_block_budgets(sizes: list[int], total: int) -> list[int]:
    """Water-fill ``total`` chars across blocks by size.

    Blocks that fit within the even share keep their full size; the freed budget
    is redistributed to the larger blocks. This keeps small rationales whole
    while trimming only the ones that actually overflow, and always sums exactly
    to ``total`` (any rounding remainder lands on the first still-unfilled
    block).
    """
    budgets = [0] * len(sizes)
    remaining_total = total
    remaining_idx = list(range(len(sizes)))
    while remaining_idx:
        share = remaining_total // len(remaining_idx)
        fits = [i for i in remaining_idx if sizes[i] <= share]
        if not fits:
            for i in remaining_idx:
                budgets[i] = share
            budgets[remaining_idx[0]] += remaining_total - share * len(remaining_idx)
            break
        for i in fits:
            budgets[i] = sizes[i]
            remaining_total -= sizes[i]
            remaining_idx.remove(i)
    return budgets


def _trim_block(block: str, budget: int, notice: str) -> str:
    """Trim one rationale block to ``budget`` chars, preserving attribution.

    Dispatches on block shape. A **stacker-combined** block (one that folds the
    stacker meta-analysis and every base model's reasoning under
    ``STACKED_BASE_REASONING_HEADER`` — the single-R1-block shape stacking
    produces) is trimmed per-sub-block by ``_trim_stacker_combined_block`` so no
    base model loses its ``Model:`` attribution or forecast block. Every other
    block (the non-stacked prod case: one forecaster per R1 block) is trimmed as
    a single body by ``_trim_single_body``. The return value never exceeds
    ``budget``.
    """
    if len(block) <= budget:
        return block
    if STACKED_BASE_REASONING_HEADER in block:
        return _trim_stacker_combined_block(block, budget, notice)
    return _trim_single_body(block, budget, notice)


def _trim_single_body(block: str, budget: int, notice: str) -> str:
    """Trim one single-forecaster body to ``budget`` chars, preserving attribution.

    Keeps (in order) the leading header line (the ``## R1: Forecaster N
    Reasoning`` header for a full block, or the ``Model:`` line for a base
    sub-block), the ``Model:`` line if it opens the body, a head of the
    reasoning prose, the trim notice, and the trailing fenced ```json forecast
    block. So every kept body retains the two things the residual pipeline
    parses — its model attribution and its forecast values — even when the
    middle prose is sacrificed. The return value never exceeds ``budget``.
    """
    if len(block) <= budget:
        return block

    header, separator, rest = block.partition("\n")
    if not separator:
        return block[:budget]

    head = header
    body = rest
    lead = rest.lstrip("\n")
    model_match = _MODEL_PREFIX_RE.match(lead)
    if model_match:
        head = f"{header}\n{model_match.group(0)}"
        body = lead[model_match.end() :]

    json_matches = list(_JSON_BLOCK_RE.finditer(body))
    json_tail = json_matches[-1].group(0) if json_matches else ""
    prose_before_json = body[: json_matches[-1].start()] if json_matches else body

    # head + \n + prose + \n + notice + \n + json_tail  -> 3 joining newlines.
    prose_budget = budget - len(head) - len(notice) - len(json_tail) - 3
    if prose_budget > 0:
        prose_head = prose_before_json[:prose_budget].rstrip("\n")
        parts = [head, prose_head, notice, json_tail]
        return "\n".join(p for p in parts if p)
    if json_tail and len(head) + len(notice) + len(json_tail) + 2 <= budget:
        return f"{head}\n{notice}\n{json_tail}"
    if len(head) + len(notice) + 1 <= budget:
        return f"{head}\n{notice}"
    return head[:budget]


def _trim_stacker_combined_block(block: str, budget: int, notice: str) -> str:
    """Trim a stacker-combined R1 body, keeping every base model's attribution.

    ``combine_stacker_and_base_reasoning`` folds the stacker's meta-analysis and all
    N base reasonings into one ``## R1: Forecaster 1 Reasoning`` block, separated by
    ``STACKED_BASE_REASONING_HEADER``, each part ending with its own fenced json
    forecast block. So this splits on the same delimiter and ``Model:`` regex the
    parser uses, water-fills ``budget`` across the parts with
    ``_allocate_block_budgets``, trims each from within via ``_trim_single_body``,
    and re-emits the delimiter verbatim; the return value never exceeds ``budget``.
    The misattribution a single-body trim causes instead: docs/architecture.md
    "Comment trimming".
    """
    stacker_portion, base_portion = block.split(STACKED_BASE_REASONING_HEADER, 1)
    matches = list(BASE_MODEL_SUBBLOCK_SPLIT_RE.finditer(base_portion))
    if not matches:
        # Delimiter but no base sub-blocks: nothing to attribute per base model.
        return _trim_single_body(block, budget, notice)

    stacker_unit = stacker_portion.rstrip()
    base_units = [
        base_portion[m.start() : (matches[i + 1].start() if i + 1 < len(matches) else len(base_portion))].rstrip()
        for i, m in enumerate(matches)
    ]
    units = [stacker_unit, *base_units]

    # The delimiter, its two newlines and the blank lines between base sub-blocks sit outside the budgets.
    fixed = len(STACKED_BASE_REASONING_HEADER) + 2 + 2 * (len(base_units) - 1)
    usable = budget - fixed
    if usable <= 0:
        # Unreachable in prod, but keep the block coherent rather than emit a headerless fragment.
        return _trim_single_body(block, budget, notice)

    budgets = _allocate_block_budgets([len(u) for u in units], usable)
    stacker_trimmed = _trim_single_body(stacker_unit, budgets[0], notice)
    base_trimmed = [_trim_single_body(unit, budgets[i + 1], notice) for i, unit in enumerate(base_units)]
    return f"{stacker_trimmed}\n{STACKED_BASE_REASONING_HEADER}\n" + "\n\n".join(base_trimmed)


def _trim_rationales_within_blocks(text: str, limit: int, notice: str) -> tuple[str, bool]:
    """Trim the FORECASTS rationales section, block by block.

    Splits the section into its ``## R1: Forecaster N Reasoning`` blocks and
    shrinks each over-budget block from *within* (see ``_trim_block``), so every
    forecaster keeps its ``Model:`` attribution line and JSON forecast block. A
    plain header+tail trim (``_trim_with_notice``) drops the head of the first
    block — including its ``Model:`` line — which the residual pipeline parses.

    Falls back to the plain header-preserving trim when the section has no
    recognizable rationale headers (or the budget is too small to seat them).
    The returned text never exceeds ``limit``.
    """
    if len(text) <= limit:
        return text, False

    headers = list(_RATIONALE_HEADER_RE.finditer(text))
    if not headers:
        return _trim_with_notice(text, limit, notice, preserve_header=True)

    preamble = text[: headers[0].start()]
    blocks = [
        text[h.start() : (headers[i + 1].start() if i + 1 < len(headers) else len(text))].rstrip()
        for i, h in enumerate(headers)
    ]

    separator = "\n\n"
    usable = limit - len(preamble) - len(separator) * (len(blocks) - 1)
    if usable <= 0:
        return _trim_with_notice(text, limit, notice, preserve_header=True)

    budgets = _allocate_block_budgets([len(b) for b in blocks], usable)
    trimmed_blocks = [_trim_block(block, budgets[i], notice) for i, block in enumerate(blocks)]
    return preamble + separator.join(trimmed_blocks), True


def _trim_preserving_summary_and_tail(text: str, cfg: TrimConfig) -> tuple[str, bool]:
    """Trim the middle, keep the summary head and the tail.

    The head runs to ``cfg.summary_end_marker`` ("### Research Summary"), which the
    framework emits right after the per-model ``*Forecaster N*:`` bullets the
    residual parsers read; the tail carries the trailing ``STACKED`` marker and as
    many R1 rationales as fit. On overflow, up to ``cfg.head_budget`` chars go to
    the head, the remaining budget to the tail, and the middle becomes the trim
    notice. Returns ``(text, False)`` when the marker is absent, so the caller falls
    back to the plain tail-only trim.
    """
    marker_idx = text.find(cfg.summary_end_marker)
    if marker_idx < 0:
        return text, False

    head_end = marker_idx + len(cfg.summary_end_marker)
    head = text[:head_end]
    notice = cfg.notice

    # Reserve head, newline, notice, newline before giving the rest to tail.
    tail_budget = cfg.comment_limit - len(head) - len(notice) - 2
    if len(head) > cfg.head_budget or tail_budget <= 0:
        # Pathologically large head, or no room for tail plus notice: let the caller fall back.
        return text, False

    tail = text[-tail_budget:]
    trimmed = f"{head}\n{notice}\n{tail}"
    return trimmed, True


def _split_structured_comment(text: str) -> tuple[str, str, str] | None:
    """``(head, research, forecasts)`` split at the ``# RESEARCH`` and ``# FORECASTS`` headers."""
    research_match = _RESEARCH_HEADER_RE.search(text)
    forecasts_match = _FORECASTS_HEADER_RE.search(text)
    if research_match is None or forecasts_match is None or research_match.start() >= forecasts_match.start():
        return None
    return (
        text[: research_match.start()],
        text[research_match.start() : forecasts_match.start()],
        text[forecasts_match.start() :],
    )


def _trim_research_section_first(text: str, cfg: TrimConfig) -> tuple[str, bool]:
    """Absorb the overflow by shrinking only the ``# RESEARCH`` section.

    For a structured comment without rationale headers. Model reasoning outranks research, so
    ``# SUMMARY`` and ``# FORECASTS`` stay whole. Returns ``(text, False)`` when the comment lacks
    the structure or shrinking research alone cannot cover the overflow;
    ``_trim_preserving_summary_and_tail`` handles those and still keeps the per-model bullets.
    """
    sections = _split_structured_comment(text)
    if sections is None:
        return text, False
    head, research_section, forecast_section = sections
    research_budget = cfg.comment_limit - len(head) - len(forecast_section)
    if research_budget <= 0:
        return text, False
    trimmed_research = _trim_research_section_to_budget(research_section, research_budget, cfg.notice)
    trimmed = f"{head}{trimmed_research}{forecast_section}"
    if len(trimmed) > cfg.comment_limit:
        return text, False
    return trimmed, True


def _count_paragraphs_that_fit(paragraph_sizes: Iterable[int], budget: int) -> int:
    """How many paragraphs, taken in order and joined by blank lines, fit in ``budget``."""
    count = 0
    used = 0
    for size in paragraph_sizes:
        used += size + (2 if count else 0)
        if used > budget:
            break
        count += 1
    return count


def _head_and_tail_counts(paragraphs: Sequence[str], budget: int) -> tuple[int, int]:
    """Leading paragraphs filling half of ``budget``, then trailing paragraphs filling the rest."""
    sizes = [len(paragraph) for paragraph in paragraphs]
    head_budget = budget // 2
    head_count = _count_paragraphs_that_fit(sizes, head_budget)
    tail_count = _count_paragraphs_that_fit(reversed(sizes[head_count:]), budget - head_budget)
    return head_count, tail_count


def _trim_research_section_to_budget(section: str, limit: int, notice: str) -> str:
    """Drop whole optional research paragraphs, keeping every disclosure block complete.

    Disclosures (summarizer banner, driver corrections, pending leads, provider diagnostics) are
    always kept, so the result can exceed ``limit`` when they alone do; callers check the total.
    """
    if len(section) <= limit:
        return section

    header, separator, _ = section.partition("\n")
    if not separator:
        trimmed, _ = _trim_with_notice(section, limit, notice, preserve_header=True)
        return trimmed

    segments = _research_segments(section, body_start=len(header) + 1)
    required = [segment for segment in segments if segment.required]
    optional = [segment for segment in segments if not segment.required]
    fixed_size = len("\n\n".join([header, *(segment.text for segment in required), notice]))
    optional_budget = limit - fixed_size - 4
    kept_optional: list[_ResearchSegment] = []
    if optional_budget > 0:
        head_count, tail_count = _head_and_tail_counts([segment.text for segment in optional], optional_budget)
        kept_optional = optional[:head_count] + optional[len(optional) - tail_count :]
    kept = sorted([*required, *kept_optional], key=lambda segment: segment.start)
    return "\n\n".join([header, *(segment.text for segment in kept), notice]) + "\n\n"


def _research_segments(section: str, body_start: int) -> list[_ResearchSegment]:
    """Split the research body into required disclosure blocks and optional paragraphs."""
    segments: list[_ResearchSegment] = []
    cursor = body_start
    for start, end in _research_disclosure_ranges(section, body_start):
        segments.extend(_research_paragraph_segments(section, cursor, start))
        disclosure = section[start:end].strip()
        if disclosure:
            segments.append(_ResearchSegment(start=start, text=disclosure, required=True))
        cursor = end
    segments.extend(_research_paragraph_segments(section, cursor, len(section)))
    return segments


def _research_paragraph_segments(section: str, start: int, end: int) -> list[_ResearchSegment]:
    boundaries = [start]
    for delimiter in _RESEARCH_PARAGRAPH_DELIMITER_RE.finditer(section, start, end):
        boundaries.extend([delimiter.start(), delimiter.end()])
    boundaries.append(end)
    return [
        _ResearchSegment(start=paragraph_start, text=section[paragraph_start:paragraph_end].strip(), required=False)
        for paragraph_start, paragraph_end in batched(boundaries, 2)
        if section[paragraph_start:paragraph_end].strip()
    ]


def _research_disclosure_ranges(section: str, body_start: int) -> list[tuple[int, int]]:
    """Spans of every disclosure block, found per research collection and merged where they overlap."""
    ranges = sorted(
        disclosure
        for collection_start, collection_end in _research_collection_ranges(section, body_start)
        for disclosure in _collection_disclosure_ranges(section, collection_start, collection_end)
    )
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _research_collection_ranges(section: str, body_start: int) -> list[tuple[int, int]]:
    """One span per ``## Report N Research`` collection, or the whole body when there is none.

    The framework renders collection content at heading level 3 or deeper (or escapes its ``#``s),
    so a level-2 report heading can only be the framework's own collection header.
    """
    header_starts = [header.start() for header in _RESEARCH_COLLECTION_HEADER_RE.finditer(section, body_start)]
    if not header_starts:
        return [(body_start, len(section))]
    return list(pairwise([*header_starts, len(section)]))


def _collection_disclosure_ranges(section: str, start: int, end: int) -> list[tuple[int, int]]:
    ranges = [(match.start(), match.end()) for match in _SUMMARIZER_SOFT_FAIL_BANNER_RE.finditer(section, start, end)]

    pending_markers = list(_PENDING_LEADS_MARKER_RE.finditer(section, start, end))
    if len(pending_markers) % 2:
        raise ValueError("Pending lead disclosure markers must appear as pairs in each research collection")
    for pending_start, pending_end in batched(pending_markers, 2):
        if pending_start.group() != PENDING_LEADS_START_MARKER or pending_end.group() != PENDING_LEADS_END_MARKER:
            raise ValueError("Pending lead disclosure markers must be properly paired in each research collection")
        ranges.append((pending_start.start(), pending_end.end()))

    for match in _CORRECTION_DISCLOSURE_HEADER_RE.finditer(section, start, end):
        next_header = _RESEARCH_SUBSECTION_HEADER_RE.search(section, match.end(), end)
        disclosure_end = end if next_header is None else next_header.start()
        pending_start = section.find(PENDING_LEADS_START_MARKER, match.end(), disclosure_end)
        if pending_start >= 0:
            disclosure_end = pending_start
        ranges.append((match.start(), disclosure_end))

    # The diagnostics block is appended last, so it runs to the end of its collection.
    ranges.extend((match.start(), end) for match in _PROVIDER_DIAGNOSTICS_HEADER_RE.finditer(section, start, end))
    return ranges


def _trim_summary_section_to_budget(section: str, limit: int, notice: str) -> str | None:
    """Trim only the prose after the summary bullets, keeping their parser boundary intact."""
    if len(section) <= limit:
        return section
    marker_index = section.find(_SUMMARY_END_MARKER)
    if marker_index < 0:
        return None

    marker_end = marker_index + len(_SUMMARY_END_MARKER)
    prefix = section[:marker_end].rstrip("\n")
    body = section[marker_end:].strip("\n")
    paragraph_budget = limit - len(prefix) - len(notice) - 8
    if paragraph_budget < 0:
        return None

    paragraphs = [paragraph.strip() for paragraph in _RESEARCH_PARAGRAPH_DELIMITER_RE.split(body) if paragraph.strip()]
    head_count, tail_count = _head_and_tail_counts(paragraphs, paragraph_budget)
    tail = paragraphs[len(paragraphs) - tail_count :] if tail_count else []
    trimmed = "\n\n".join([prefix, *paragraphs[:head_count], notice, *tail]) + "\n\n"
    return trimmed if len(trimmed) <= limit else None


def _rationale_content_preserved(original: str, trimmed: str) -> bool:
    """Every ``Model:`` line and json forecast block in ``original`` is still in ``trimmed``."""
    missing_models = Counter(_MODEL_PREFIX_RE.findall(original)) - Counter(_MODEL_PREFIX_RE.findall(trimmed))
    missing_json = Counter(_JSON_BLOCK_RE.findall(original)) - Counter(_JSON_BLOCK_RE.findall(trimmed))
    return not (missing_models or missing_json)


def _trim_rationales_to_fit(head: str, research_section: str, forecast_section: str, cfg: TrimConfig) -> str | None:
    """Trim within rationale blocks; ``None`` if that cannot fit without losing a model or forecast."""
    rationale_headers = list(_RATIONALE_HEADER_RE.finditer(forecast_section))
    marker_starts = [
        match.start()
        for match in _TRAILING_COMMENT_MARKER_RE.finditer(forecast_section)
        if match.start() > rationale_headers[-1].end()
    ]
    rationale_start = rationale_headers[0].start()
    rationale_end = marker_starts[0] if marker_starts else len(forecast_section)
    rationale_prefix = forecast_section[:rationale_start]
    rationale_text = forecast_section[rationale_start:rationale_end]
    trailing_markers = forecast_section[rationale_end:]
    rationale_budget = (
        cfg.comment_limit - len(head) - len(research_section) - len(rationale_prefix) - len(trailing_markers)
    )
    if rationale_budget <= 0:
        return None
    trimmed_rationales, _ = _trim_rationales_within_blocks(rationale_text, rationale_budget, cfg.notice)
    candidate = f"{head}{research_section}{rationale_prefix}{trimmed_rationales}{trailing_markers}"
    if len(candidate) > cfg.comment_limit or not _rationale_content_preserved(rationale_text, trimmed_rationales):
        return None
    return candidate


def _trim_complete_comment(text: str, cfg: TrimConfig) -> tuple[str, str] | None:
    """Trim a structured comment with rationale blocks; returns ``(trimmed, strategy)``.

    Research paragraphs go first, then the summary prose, then the middle of each rationale (keeping
    every model's attribution and forecast block). If even that cannot fit, the comment is replaced by
    the explicitly incomplete explanation. ``None`` when the comment lacks that structure.
    """
    sections = _split_structured_comment(text)
    if sections is None:
        return None
    head, research_section, forecast_section = sections
    if _RATIONALE_HEADER_RE.search(forecast_section) is None:
        return None

    research_section = _trim_research_section_to_budget(
        research_section, cfg.comment_limit - len(head) - len(forecast_section), cfg.notice
    )
    if len(head) + len(research_section) + len(forecast_section) > cfg.comment_limit:
        summary_budget = cfg.comment_limit - len(research_section) - len(forecast_section)
        head = _trim_summary_section_to_budget(head, summary_budget, cfg.notice) or head
    if len(head) + len(research_section) + len(forecast_section) <= cfg.comment_limit:
        return f"{head}{research_section}{forecast_section}", "complete_comment_allocator"

    trimmed = _trim_rationales_to_fit(head, research_section, forecast_section, cfg)
    if trimmed is not None:
        return trimmed, "complete_comment_allocator"
    return _incomplete_explanation(text, cfg), "incomplete_explanation"


def _final_prediction_block(text: str) -> str:
    """The ``*Final Prediction*`` line plus its continuation lines (multiple choice and numeric span several)."""
    prediction_match = _FINAL_PREDICTION_LINE_RE.search(text)
    if prediction_match is None:
        return _FORECAST_IN_REPORT_LINE
    next_metadata = _SUMMARY_METADATA_LINE_RE.search(text, prediction_match.end())
    prediction_end = len(text) if next_metadata is None else next_metadata.start()
    return text[prediction_match.start() : prediction_end].rstrip()


def _incomplete_explanation(text: str, cfg: TrimConfig) -> str:
    """A bounded explanation that marks omitted rationale detail as unknown.

    Tries progressively shorter renderings and returns the first that fits: full disclosures, then
    compacted corrections, then compacted pending leads, each first with the readable prediction and
    then with a pointer to the published report. Provider diagnostics run to the end of their
    collection, so many collections can overflow even the compact forms; the last rendering then
    drops the disclosures and keeps only the prediction pointer and the markers.
    """
    prediction_block = _final_prediction_block(text)
    markers = list(dict.fromkeys(match.group(1) for match in _TRAILING_COMMENT_MARKER_RE.finditer(text)))
    disclosures = _comment_research_disclosures(text)
    compact_corrections = _compact_research_disclosures(disclosures)
    compact_all = _compact_research_disclosures(disclosures, compact_pending_leads=True)

    def render(prediction: str, disclosure_blocks: list[str]) -> str:
        return "\n".join(
            [
                "# SUMMARY",
                prediction,
                _SUMMARY_END_MARKER,
                _DETAIL_OMITTED_SENTENCE,
                *disclosure_blocks,
                "# FORECASTS",
                cfg.notice,
                EXPLANATION_DETAIL_UNKNOWN_MARKER,
                *markers,
            ]
        )

    for candidate in (
        render(prediction_block, disclosures),
        render(prediction_block, compact_corrections),
        render(_FORECAST_IN_REPORT_LINE, compact_corrections),
        render(prediction_block, compact_all),
        render(_FORECAST_IN_REPORT_LINE, compact_all),
    ):
        if len(candidate) <= cfg.comment_limit:
            return candidate

    bare = render(_FORECAST_IN_REPORT_LINE, [])
    if len(bare) > cfg.comment_limit:
        raise ValueError(f"Minimal incomplete explanation is {len(bare)} chars, over the {cfg.comment_limit} limit")
    logger.warning("Research disclosures dropped from the incomplete explanation: they alone exceed the limit")
    return bare


def _comment_research_disclosures(text: str) -> list[str]:
    sections = _split_structured_comment(text)
    if sections is None:
        return []
    _, research_section, _ = sections
    body_start = research_section.index("\n") + 1
    return [
        research_section[start:end].strip() for start, end in _research_disclosure_ranges(research_section, body_start)
    ]


def _compact_research_disclosures(disclosures: list[str], *, compact_pending_leads: bool = False) -> list[str]:
    compact: list[str] = []
    for disclosure in disclosures:
        if _CORRECTION_DISCLOSURE_HEADER_RE.match(disclosure):
            compact.append("\n".join([*disclosure.splitlines()[:2], "[Correction details omitted for length.]"]))
        elif compact_pending_leads and disclosure.startswith(PENDING_LEADS_START_MARKER):
            compact.append(
                "\n".join(
                    [
                        PENDING_LEADS_START_MARKER,
                        "Pending leads:",
                        "[Pending lead details omitted for length.]",
                        PENDING_LEADS_END_MARKER,
                    ]
                )
            )
        else:
            compact.append(disclosure)
    return compact


def trim_comment(text: str, *, config: TrimConfig | None = None) -> str:
    cfg = config or TrimConfig()
    if len(text) <= cfg.comment_limit:
        return text

    # Every strategy in this chain preserves the leading "#", so the validator invariant holds.
    complete = _trim_complete_comment(text, cfg)
    if complete is not None:
        trimmed, strategy = complete
    else:
        trimmed, used = _trim_research_section_first(text, cfg)
        strategy = "research_first"
        if not used:
            trimmed, used = _trim_preserving_summary_and_tail(text, cfg)
            strategy = "summary_and_tail"
        if not used:
            trimmed, _ = _trim_with_notice(text, cfg.comment_limit, cfg.notice, preserve_header=True)
            strategy = "header_and_tail"

    logger.warning(
        "Trimmed Metaculus comment from %s to %s characters (strategy=%s)",
        len(text),
        len(trimmed),
        strategy,
    )
    return trimmed


__all__ = ["TRIM_NOTICE", "TrimConfig", "trim_comment", "trim_section"]
