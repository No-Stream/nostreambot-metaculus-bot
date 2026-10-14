"""Trim an assembled Metaculus comment, and its individual sections, to the platform's char limits.

Two entry points. ``trim_section`` shrinks one section of the framework's unified comment to its
own budget (``_section_budget`` maps the section name to it), and ``trim_comment`` shrinks the
whole assembled comment through an ordered strategy chain: shrink ``# RESEARCH`` alone
(``_trim_research_section_first``), else keep the summary head and the tail
(``_trim_preserving_summary_and_tail``), else fall back to a plain header-preserving trim
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
from collections.abc import Sequence
from dataclasses import dataclass
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
_CORRECTION_DISCLOSURE_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?m)^#{3,8} ⚠ Driver-reported corrections[^\n]*$"
)
_RESEARCH_SUBSECTION_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^#{2,8} \S.*$")
_RESEARCH_PARAGRAPH_DELIMITER_RE: Final[re.Pattern[str]] = re.compile(r"\n[ \t]*\n")
_PROVIDER_DIAGNOSTICS_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    rf"(?m)^#{{2,8}} {re.escape(PROVIDER_DIAGNOSTICS_HEADER.split(' ', maxsplit=1)[1])}$"
)
_RESEARCH_COLLECTION_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^#{2,8} Report \d+ Research[ \t]*$")
_PENDING_LEADS_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    rf"{re.escape(PENDING_LEADS_START_MARKER)}|{re.escape(PENDING_LEADS_END_MARKER)}"
)
_SUMMARIZER_SOFT_FAIL_BANNER_RE: Final[re.Pattern[str]] = re.compile(re.escape(SUMMARIZER_SOFT_FAIL_BANNER))

# The framework emits this right after the per-model bullets the residual parsers read.
_SUMMARY_END_MARKER: Final[str] = "### Research Summary"

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


def _trim_research_section_first(
    text: str,
    cfg: TrimConfig,
    research_collection_ranges: Sequence[tuple[int, int]] | None = None,
) -> tuple[str, bool]:
    """Absorb the overflow by shrinking only the ``# RESEARCH`` section.

    Model reasoning outranks research, so an overflowing comment shrinks the research
    middle first and keeps ``# SUMMARY`` (bullets and the summary-end marker) and
    ``# FORECASTS`` (rationales and the trailing markers) whole. Research is
    summary-style and front-loaded, so the *front* of its body is what survives.

    Returns ``(text, False)`` when the comment lacks the ``# RESEARCH`` /
    ``# FORECASTS`` structure, or when shrinking research alone cannot cover the
    overflow; ``_trim_preserving_summary_and_tail`` handles those correctly and still
    preserves the leading ``#`` and the per-model bullets.
    """
    research_match = _RESEARCH_HEADER_RE.search(text)
    forecasts_match = _FORECASTS_HEADER_RE.search(text)
    if research_match is None or forecasts_match is None or research_match.start() >= forecasts_match.start():
        return text, False

    head = text[: research_match.start()]  # everything up to "# RESEARCH"
    research_section = text[research_match.start() : forecasts_match.start()]
    tail = text[forecasts_match.start() :]  # "# FORECASTS" onward (incl. markers)
    research_budget = cfg.comment_limit - len(head) - len(tail)
    if research_budget <= 0:
        return text, False

    relative_collection_ranges = _relative_research_collection_ranges(
        research_collection_ranges,
        section_start=research_match.start(),
        body_start=research_match.end(),
        section_end=forecasts_match.start(),
    )
    trimmed_research = _trim_research_section_to_budget(
        research_section,
        research_budget,
        cfg.notice,
        relative_collection_ranges,
    )
    trimmed = f"{head}{trimmed_research}{tail}"
    if len(trimmed) > cfg.comment_limit:
        return text, False
    return trimmed, True


def _take_paragraphs(
    paragraphs: list[str],
    budget: int,
    *,
    reverse: bool = False,
    excluded: set[int] | None = None,
) -> tuple[list[tuple[int, str]], int]:
    selected: list[tuple[int, str]] = []
    used = 0
    excluded_indices = excluded or set()
    indices = range(len(paragraphs) - 1, -1, -1) if reverse else range(len(paragraphs))
    for index in indices:
        if index in excluded_indices:
            continue
        paragraph = paragraphs[index]
        added_size = len(paragraph) + (2 if selected else 0)
        if used + added_size > budget:
            break
        selected.append((index, paragraph))
        used += added_size
    if reverse:
        selected.reverse()
    return selected, used


def _trim_research_section_to_budget(
    section: str,
    limit: int,
    notice: str,
    research_collection_ranges: Sequence[tuple[int, int]] | None = None,
) -> str:
    """Trim optional research paragraphs while reserving complete publication disclosures."""
    if len(section) <= limit:
        return section

    header, separator, _ = section.partition("\n")
    if not separator:
        trimmed, _ = _trim_with_notice(section, limit, notice, preserve_header=True)
        return trimmed

    segments = _research_segments(section, len(header) + 1, research_collection_ranges)
    required_segments = [segment for segment in segments if segment.required]
    optional_segments = [segment for segment in segments if not segment.required]
    fixed_parts = [header, *(segment.text for segment in required_segments), notice]
    optional_budget = max(0, limit - len("\n\n".join(fixed_parts)) - 2)
    selected_optional = _select_optional_research_segments(optional_segments, optional_budget)
    kept = sorted([*required_segments, *selected_optional], key=lambda segment: segment.start)
    parts = [header, *(segment.text for segment in kept), notice]
    return "\n\n".join(parts) + "\n\n"


def _research_segments(
    section: str,
    body_start: int,
    research_collection_ranges: Sequence[tuple[int, int]] | None = None,
) -> list[_ResearchSegment]:
    segments: list[_ResearchSegment] = []
    cursor = body_start
    for start, end in _research_disclosure_ranges(section, body_start, research_collection_ranges):
        segments.extend(_research_paragraph_segments(section, cursor, start))
        disclosure = section[start:end].strip()
        if disclosure:
            segments.append(_ResearchSegment(start=start, text=disclosure, required=True))
        cursor = end
    segments.extend(_research_paragraph_segments(section, cursor, len(section)))
    return segments


def _select_optional_research_segments(segments: list[_ResearchSegment], budget: int) -> list[_ResearchSegment]:
    if not segments or budget <= 0:
        return []

    paragraph_budget = max(0, budget - 2)
    head_budget = paragraph_budget // 2
    paragraph_text = [segment.text for segment in segments]
    head, _ = _take_paragraphs(paragraph_text, head_budget)
    head_indices = {index for index, _ in head}
    tail, _ = _take_paragraphs(
        paragraph_text,
        paragraph_budget - head_budget,
        reverse=True,
        excluded=head_indices,
    )
    selected_indices = head_indices | {index for index, _ in tail}
    return [segments[index] for index in sorted(selected_indices)]


def _research_disclosure_ranges(
    section: str,
    body_start: int,
    research_collection_ranges: Sequence[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Find whole rendered disclosure blocks without crossing collection ownership."""
    if research_collection_ranges is None:
        research_collection_ranges = _infer_research_collection_ranges(section, body_start)

    ranges: list[tuple[int, int]] = []
    previous_end = body_start
    for collection_start, collection_end in research_collection_ranges:
        if collection_start < body_start or collection_end <= collection_start or collection_end > len(section):
            raise ValueError("Research collection ranges must be ordered spans inside the research body")
        if collection_start < previous_end:
            raise ValueError("Research collection ranges must not overlap")
        previous_end = collection_end
        ranges.extend(_collection_disclosure_ranges(section, collection_start, collection_end))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start < merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _infer_research_collection_ranges(section: str, body_start: int) -> list[tuple[int, int]]:
    """Use the framework's exact report heading when the caller has no composition context."""
    collection_headers = list(_RESEARCH_COLLECTION_HEADER_RE.finditer(section, body_start))
    if not collection_headers:
        return [(body_start, len(section))]
    return [
        (
            header.start(),
            collection_headers[index + 1].start() if index + 1 < len(collection_headers) else len(section),
        )
        for index, header in enumerate(collection_headers)
    ]


def _collection_disclosure_ranges(section: str, start: int, end: int) -> list[tuple[int, int]]:
    ranges = [(match.start(), match.end()) for match in _SUMMARIZER_SOFT_FAIL_BANNER_RE.finditer(section, start, end)]

    pending_markers = list(_PENDING_LEADS_MARKER_RE.finditer(section, start, end))
    if len(pending_markers) % 2:
        raise ValueError("Pending lead disclosure markers must appear as pairs in each research collection")
    for marker_index in range(0, len(pending_markers), 2):
        pending_start, pending_end = pending_markers[marker_index : marker_index + 2]
        if pending_start.group() != PENDING_LEADS_START_MARKER or pending_end.group() != PENDING_LEADS_END_MARKER:
            raise ValueError("Pending lead disclosure markers must be properly nested in each research collection")
        ranges.append((pending_start.start(), pending_end.end()))

    for match in _CORRECTION_DISCLOSURE_HEADER_RE.finditer(section, start, end):
        next_header = _RESEARCH_SUBSECTION_HEADER_RE.search(section, match.end(), end)
        disclosure_end = end if next_header is None else next_header.start()
        pending_start = section.find(PENDING_LEADS_START_MARKER, match.end(), disclosure_end)
        if pending_start >= 0:
            disclosure_end = pending_start
        ranges.append((match.start(), disclosure_end))

    ranges.extend((match.start(), end) for match in _PROVIDER_DIAGNOSTICS_HEADER_RE.finditer(section, start, end))
    return ranges


def _research_paragraph_segments(section: str, start: int, end: int) -> list[_ResearchSegment]:
    segments: list[_ResearchSegment] = []
    cursor = start
    for delimiter_match in _RESEARCH_PARAGRAPH_DELIMITER_RE.finditer(section, start, end):
        _append_research_paragraph_segment(segments, section, cursor, delimiter_match.start())
        cursor = delimiter_match.end()
    _append_research_paragraph_segment(segments, section, cursor, end)
    return segments


def _append_research_paragraph_segment(segments: list[_ResearchSegment], section: str, start: int, end: int) -> None:
    paragraph = section[start:end]
    left_trimmed = paragraph.lstrip()
    text = left_trimmed.rstrip()
    if text:
        leading_whitespace = len(paragraph) - len(left_trimmed)
        segments.append(_ResearchSegment(start=start + leading_whitespace, text=text, required=False))


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

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n[ \t]*\n", body) if paragraph.strip()]
    head_budget = paragraph_budget // 2
    head, _ = _take_paragraphs(paragraphs, head_budget)
    head_indices = {index for index, _ in head}
    tail, _ = _take_paragraphs(paragraphs, paragraph_budget - head_budget, reverse=True, excluded=head_indices)
    kept = [paragraph for _, paragraph in head + tail]
    parts = [prefix, *kept[: len(head)], notice, *kept[len(head) :]]
    trimmed = "\n\n".join(parts) + "\n\n"
    return trimmed if len(trimmed) <= limit else None


def _rationale_content_preserved(original: str, trimmed: str) -> bool:
    original_models = Counter(_MODEL_PREFIX_RE.findall(original))
    trimmed_models = Counter(_MODEL_PREFIX_RE.findall(trimmed))
    original_json = Counter(_JSON_BLOCK_RE.findall(original))
    trimmed_json = Counter(_JSON_BLOCK_RE.findall(trimmed))
    return not (original_models - trimmed_models or original_json - trimmed_json)


def _incomplete_explanation(
    text: str,
    cfg: TrimConfig,
    research_collection_ranges: Sequence[tuple[int, int]] | None = None,
) -> str:
    """Build a bounded explanation that marks omitted rationale detail as unknown."""
    prediction_match = _FINAL_PREDICTION_LINE_RE.search(text)
    if prediction_match is None:
        prediction_block = "*Final Prediction*: retained in the published forecast report."
    else:
        next_metadata = re.search(
            r"(?m)^\*(?:Total Cost|Time Spent|LLMs|Bot Name)\*:",
            text[prediction_match.end() :],
        )
        prediction_end = len(text) if next_metadata is None else prediction_match.end() + next_metadata.start()
        prediction_block = text[prediction_match.start() : prediction_end].rstrip()

    markers = list(dict.fromkeys(match.group(1) for match in _TRAILING_COMMENT_MARKER_RE.finditer(text)))
    disclosure_blocks = _comment_research_disclosures(text, research_collection_ranges)
    fixed_lines = [
        "# SUMMARY",
        prediction_block,
        "### Research Summary",
        "Rationale detail was omitted for length; missing model attribution and reasoning are unknown.",
        *disclosure_blocks,
        "# FORECASTS",
        cfg.notice,
        EXPLANATION_DETAIL_UNKNOWN_MARKER,
        *markers,
    ]
    kept_lines = [line for line in fixed_lines if line]
    rendered = "\n".join(kept_lines)
    if len(rendered) <= cfg.comment_limit:
        return rendered

    # Multiple-choice labels can be long. If the complete readable distribution cannot fit beside
    # the required markers, the published report remains the source of the full forecast.
    bounded_prediction = prediction_block
    compact_lines = [
        "# SUMMARY",
        bounded_prediction,
        "Rationale detail was omitted for length; missing model attribution and reasoning are unknown.",
        "# FORECASTS",
        EXPLANATION_DETAIL_UNKNOWN_MARKER,
        *markers,
    ]
    compact_disclosures = _compact_research_disclosures(disclosure_blocks)

    def render_compact(prediction: str) -> str:
        return "\n".join(
            [
                "# SUMMARY",
                prediction,
                "### Research Summary",
                compact_lines[2],
                *compact_disclosures,
                *compact_lines[3:],
            ]
        )

    compact = render_compact(bounded_prediction)
    if len(compact) <= cfg.comment_limit:
        return compact

    report_reference = "*Final Prediction*: the full forecast remains in the published report."
    compact = render_compact(report_reference)
    if len(compact) <= cfg.comment_limit:
        return compact

    compact_disclosures = _compact_research_disclosures(disclosure_blocks, compact_pending_leads=True)
    compact = render_compact(bounded_prediction)
    if len(compact) <= cfg.comment_limit:
        return compact

    compact = render_compact(report_reference)
    if len(compact) <= cfg.comment_limit:
        return compact

    # These fields are bounded by the small, fixed marker vocabulary and disclosure headers. Keep
    # the explicit incomplete status and all essential disclosures; do not cut a value or marker.
    return "\n".join(
        [
            "# SUMMARY",
            report_reference,
            *compact_disclosures,
            "# FORECASTS",
            EXPLANATION_DETAIL_UNKNOWN_MARKER,
            *markers,
        ]
    )


def _comment_research_disclosures(
    text: str,
    research_collection_ranges: Sequence[tuple[int, int]] | None = None,
) -> list[str]:
    research_match = _RESEARCH_HEADER_RE.search(text)
    forecasts_match = _FORECASTS_HEADER_RE.search(text)
    if research_match is None or forecasts_match is None or research_match.start() >= forecasts_match.start():
        return []
    section = text[research_match.start() : forecasts_match.start()]
    relative_ranges = _relative_research_collection_ranges(
        research_collection_ranges,
        section_start=research_match.start(),
        body_start=research_match.end(),
        section_end=forecasts_match.start(),
    )
    ranges = _research_disclosure_ranges(section, len("# RESEARCH\n"), relative_ranges)
    return [section[start:end].strip() for start, end in ranges]


def _compact_research_disclosures(disclosures: list[str], *, compact_pending_leads: bool = False) -> list[str]:
    compact: list[str] = []
    for disclosure in disclosures:
        if _CORRECTION_DISCLOSURE_HEADER_RE.match(disclosure):
            lines = disclosure.splitlines()
            correction_notice = "[Correction details omitted for length.]"
            compact.append("\n".join([*lines[:2], correction_notice]))
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


def _relative_research_collection_ranges(
    research_collection_ranges: Sequence[tuple[int, int]] | None,
    *,
    section_start: int,
    body_start: int,
    section_end: int,
) -> tuple[tuple[int, int], ...] | None:
    if research_collection_ranges is None:
        return None

    relative_ranges: list[tuple[int, int]] = []
    previous_end = body_start
    for start, end in research_collection_ranges:
        if start < body_start or end <= start or end > section_end or start < previous_end:
            raise ValueError("Research collection ranges must be ordered spans inside # RESEARCH")
        relative_ranges.append((start - section_start, end - section_start))
        previous_end = end
    return tuple(relative_ranges)


def _complete_comment_sections(
    text: str,
    research_collection_ranges: Sequence[tuple[int, int]] | None = None,
) -> tuple[str, str, str, list[re.Match[str]], tuple[tuple[int, int], ...] | None] | None:
    research_match = _RESEARCH_HEADER_RE.search(text)
    forecasts_match = _FORECASTS_HEADER_RE.search(text)
    if research_match is None or forecasts_match is None or research_match.start() >= forecasts_match.start():
        return None

    forecast_section = text[forecasts_match.start() :]
    rationale_headers = list(_RATIONALE_HEADER_RE.finditer(forecast_section))
    if not rationale_headers:
        return None

    research_section = text[research_match.start() : forecasts_match.start()]
    relative_ranges = _relative_research_collection_ranges(
        research_collection_ranges,
        section_start=research_match.start(),
        body_start=research_match.end(),
        section_end=forecasts_match.start(),
    )
    return text[: research_match.start()], research_section, forecast_section, rationale_headers, relative_ranges


def _allocate_research_and_summary(
    head: str,
    research_section: str,
    forecast_section: str,
    cfg: TrimConfig,
    *,
    research_collection_ranges: Sequence[tuple[int, int]] | None = None,
) -> tuple[str, str, str]:
    research_budget = cfg.comment_limit - len(head) - len(forecast_section)
    trimmed_research = _trim_research_section_to_budget(
        research_section,
        research_budget,
        cfg.notice,
        research_collection_ranges,
    )
    candidate = f"{head}{trimmed_research}{forecast_section}"
    if len(candidate) <= cfg.comment_limit:
        return head, trimmed_research, forecast_section

    summary_budget = cfg.comment_limit - len(trimmed_research) - len(forecast_section)
    trimmed_head = _trim_summary_section_to_budget(head, summary_budget, cfg.notice)
    if (
        trimmed_head is not None
        and len(trimmed_head) + len(trimmed_research) + len(forecast_section) <= cfg.comment_limit
    ):
        return trimmed_head, trimmed_research, forecast_section
    return head, trimmed_research, forecast_section


def _trim_rationales_to_fit(
    head: str,
    research_section: str,
    forecast_section: str,
    *,
    rationale_headers: list[re.Match[str]],
    cfg: TrimConfig,
) -> str | None:
    final_rationale_header = rationale_headers[-1]
    marker_matches = [
        match
        for match in _TRAILING_COMMENT_MARKER_RE.finditer(forecast_section)
        if match.start() > final_rationale_header.end()
    ]
    marker_start = marker_matches[0].start() if marker_matches else len(forecast_section)
    rationale_end = marker_start
    rationale_prefix = forecast_section[: rationale_headers[0].start()]
    rationale_text = forecast_section[rationale_headers[0].start() : rationale_end]
    trailing_markers = forecast_section[marker_start:]
    rationale_budget = (
        cfg.comment_limit - len(head) - len(research_section) - len(rationale_prefix) - len(trailing_markers)
    )
    if rationale_budget > 0:
        trimmed_rationales, _ = _trim_rationales_within_blocks(rationale_text, rationale_budget, cfg.notice)
        trimmed_forecast = f"{rationale_prefix}{trimmed_rationales}{trailing_markers}"
        candidate = f"{head}{research_section}{trimmed_forecast}"
        if len(candidate) <= cfg.comment_limit and _rationale_content_preserved(rationale_text, trimmed_rationales):
            return candidate
    return None


def _trim_complete_comment(
    text: str,
    cfg: TrimConfig,
    research_collection_ranges: Sequence[tuple[int, int]] | None = None,
) -> str | None:
    sections = _complete_comment_sections(text, research_collection_ranges)
    if sections is None:
        return None

    head, research_section, forecast_section, rationale_headers, relative_collection_ranges = sections
    head, research_section, forecast_section = _allocate_research_and_summary(
        head,
        research_section,
        forecast_section,
        cfg,
        research_collection_ranges=relative_collection_ranges,
    )
    if len(head) + len(research_section) + len(forecast_section) <= cfg.comment_limit:
        return f"{head}{research_section}{forecast_section}"

    trimmed = _trim_rationales_to_fit(
        head,
        research_section,
        forecast_section,
        rationale_headers=rationale_headers,
        cfg=cfg,
    )
    if trimmed is not None:
        return trimmed

    return _incomplete_explanation(text, cfg, research_collection_ranges)


def trim_comment(
    text: str,
    *,
    config: TrimConfig | None = None,
    research_collection_ranges: Sequence[tuple[int, int]] | None = None,
) -> str:
    cfg = config or TrimConfig()
    if len(text) <= cfg.comment_limit:
        return text

    # The complete composition is available here, after metadata and markers have been appended.
    # Keep fitting comments byte-identical; on real overflow shrink whole research paragraphs,
    # then trim rationale blocks while preserving each model and structured forecast.
    trimmed = _trim_complete_comment(text, cfg, research_collection_ranges)
    if trimmed is not None:
        strategy = "complete_comment_allocator"
    else:
        # Historical or unstructured shapes keep the established fallbacks.
        trimmed, used = _trim_research_section_first(text, cfg, research_collection_ranges)
        if used:
            strategy = "research_first"
        else:
            if (
                research_collection_ranges is not None
                and _RESEARCH_HEADER_RE.search(text) is not None
                and _FORECASTS_HEADER_RE.search(text) is not None
            ):
                trimmed = _incomplete_explanation(text, cfg, research_collection_ranges)
                strategy = "incomplete_explanation"
            else:
                trimmed, used = _trim_preserving_summary_and_tail(text, cfg)
                if used:
                    strategy = "summary_and_tail"
                else:
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
