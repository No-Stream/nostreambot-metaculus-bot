"""Targeted research pipeline for conditional stacking.

When base forecaster models disagree significantly, this module:
1. Extracts the crux of disagreement using a cheap analyzer model.
2. Runs a targeted web search via OpenAI native search to resolve it.

Also provides ``run_gap_fill_pass`` — an always-on second-pass that runs after
first-pass research. It identifies factual gaps in the first pass, drops the ones
the analyzer graded as not worth a search (``triage_gaps``), and resolves each
survivor via a parallel OpenAI native web search (OpenRouter, donated-key billed).
"""

import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from forecasting_tools import GeneralLlm, MetaculusQuestion
from pydantic import BaseModel, ConfigDict

from metaculus_bot.constants import (
    CRUX_SOFT_DEADLINE,
    GAP_FILL_ANALYZER_MODEL,
    GAP_FILL_ANALYZER_TIMEOUT,
    GAP_FILL_ANALYZER_WALL_TIMEOUT,
    GAP_FILL_MAX_GAPS,
    GAP_FILL_RESOLVER_MODEL,
    GAP_FILL_RESOLVER_REASONING_EFFORT,
    NATIVE_SEARCH_WALL_TIMEOUT,
)
from metaculus_bot.llm_retry import invoke_with_broad_retry, invoke_with_transient_retry
from metaculus_bot.prompts import (
    disagreement_crux_prompt,
    gap_fill_analyzer_prompt,
    gap_fill_search_prompt,
    targeted_search_prompt,
)
from metaculus_bot.research.providers import build_native_search_llm
from metaculus_bot.research.raw_log import record_raw_research
from metaculus_bot.structured_output_schema import extract_first_balanced_braces, extract_json_block

__all__ = [
    "DROP_ALREADY_ANSWERED",
    "DROP_NOT_ANSWERABLE",
    "DROP_OVER_CAP",
    "DROP_REASONS",
    "DROP_SAME_NEED",
    "DROP_SCHEMA",
    "GapTriage",
    "extract_disagreement_crux",
    "run_gap_fill_pass",
    "run_targeted_search",
    "triage_gaps",
]

logger: logging.Logger = logging.getLogger(__name__)

# Broad by design; the pass soft-fails to "" and CancelledError escapes. See docs/research.md "v1 implementation notes".
_GAP_FILL_SOFT_FAIL_EXCEPTIONS: tuple[type[BaseException], ...] = (Exception,)

# The pass-through allowlist _parse_gap_list copies; _grade validates the same three names. Reasons are marker fields.
GAP_GRADE_FIELDS: tuple[str, ...] = ("answerable_now", "already_in_first_pass", "same_need_as")
DROP_NOT_ANSWERABLE = "not_answerable"
DROP_ALREADY_ANSWERED = "in_first_pass"
DROP_SAME_NEED = "same_need"
DROP_SCHEMA = "schema"
DROP_OVER_CAP = "over_cap"
DROP_REASONS: tuple[str, ...] = (DROP_NOT_ANSWERABLE, DROP_ALREADY_ANSWERED, DROP_SAME_NEED, DROP_SCHEMA, DROP_OVER_CAP)


class GapCandidate(BaseModel):
    """Required analyzer fields, enforced by the provider before local triage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    gap: str
    why_matters: str
    search_query: str
    answerable_now: bool
    already_in_first_pass: bool
    same_need_as: int | None


class GapAnalysis(BaseModel):
    """An explicitly empty list is a successful analysis with no useful gaps."""

    model_config = ConfigDict(extra="forbid", strict=True)

    gaps: list[GapCandidate]


async def extract_disagreement_crux(
    analyzer_llm: GeneralLlm,
    question_text: str,
    base_prediction_texts: list[str],
) -> str:
    """Identify the core factual disagreement across base forecaster analyses.

    Args:
        analyzer_llm: A cheap, low-effort model used for extraction only (the
            DISAGREEMENT_ANALYZER_LLM slot in llm_configs.py).
        question_text: The full question text being forecasted.
        base_prediction_texts: Reasoning texts from base models (already stripped of model tags).

    Returns:
        A short string describing the factual crux of disagreement.
    """
    prompt = disagreement_crux_prompt(question_text, base_prediction_texts)
    logger.info(f"Extracting disagreement crux from {len(base_prediction_texts)} forecaster analyses")
    # A 30s-gated retry on an allowed_tries=1 analyzer. See docs/research.md "v1 implementation notes".
    crux = await invoke_with_broad_retry(
        lambda: analyzer_llm.invoke(prompt), wall_timeout=CRUX_SOFT_DEADLINE, label="disagreement_crux"
    )
    logger.info(f"Disagreement crux extracted: {len(crux)} chars")
    return crux


async def run_targeted_search(crux: str, question_text: str, *, is_benchmarking: bool = False) -> str:
    """Run a targeted web search to resolve a specific factual disagreement.

    Uses OpenAI native web search via OpenRouter (`build_native_search_llm`) to
    find current, authoritative information about the identified crux.

    Args:
        crux: The factual question(s) driving forecaster disagreement.
        question_text: The full question text being forecasted.
        is_benchmarking: If True, excludes prediction market data to avoid data leakage.

    Returns:
        Search results with inline citations addressing the crux.
    """
    llm = build_native_search_llm(role="targeted_search")
    prompt = targeted_search_prompt(crux, question_text, is_benchmarking=is_benchmarking)
    logger.info(
        f"Running targeted search via {llm.model} for crux: "
        f"{crux[:100]}..."  # HARNESS-SCAN-EXEMPT-subsampling: a log-line preview, not a data reduction
    )
    # The wall is the hard cap; litellm's per-request timeout is not. See docs/research.md "v1 implementation notes".
    result = await invoke_with_transient_retry(
        lambda: llm.invoke(prompt), wall_timeout=NATIVE_SEARCH_WALL_TIMEOUT, label="targeted_search"
    )
    logger.info(f"Targeted search complete: {len(result)} chars")
    return result


# ---------------------------------------------------------------------------
# Second-pass gap-fill
# ---------------------------------------------------------------------------


def _parse_gap_list(raw: str) -> list[dict[str, Any]]:
    """Extract the gap list from the analyzer's JSON output, one slot per listed item, ungraded and unclipped.

    Robust to light markdown wrapping (```json``` fences) and trailing commentary.
    Raises ValueError on an invalid envelope so callers can report degradation. Every list item
    keeps its slot (a non-dict item or one without gap text becomes an empty slot)
    so ``same_need_as`` positions stay the analyzer's own; the three grade fields
    (``GAP_GRADE_FIELDS``) pass through exactly as the analyzer typed them, and
    only when present, so ``triage_gaps`` can tell an omitted grade from a null
    one. Grading, the schema drops and the ``GAP_FILL_MAX_GAPS`` cap all happen there.
    """
    if not raw or not raw.strip():
        raise ValueError("Gap-fill analyzer returned an empty response")

    # Fenced first, then a balanced-brace scan for trailing prose. See docs/research.md "v1 implementation notes".
    fenced = extract_json_block(raw)
    stripped = fenced if fenced is not None else extract_first_balanced_braces(raw) or raw.strip()

    try:
        data: Any = json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            f"GapFill: could not parse analyzer JSON ({type(exc).__name__}): {exc}; "
            f"raw[:200]={raw[:200]!r}"  # HARNESS-SCAN-EXEMPT-subsampling: a log-line preview, not a data reduction
        )
        raise ValueError("Gap-fill analyzer returned invalid JSON") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Gap-fill analyzer output was not an object: {type(data).__name__}")

    gaps_raw = data.get("gaps")
    if not isinstance(gaps_raw, list):
        raise ValueError("Gap-fill analyzer output must contain a gaps list")

    gaps: list[dict[str, Any]] = []
    for item in gaps_raw:
        fields: dict[str, Any] = item if isinstance(item, dict) else {}
        gap_text = str(fields.get("gap", "")).strip()
        gap: dict[str, Any] = {
            "gap": gap_text,
            "search_query": str(fields.get("search_query", "") or gap_text).strip(),
            "why_matters": str(fields.get("why_matters", "")).strip(),
        }
        gap.update({field: fields[field] for field in GAP_GRADE_FIELDS if field in fields})
        gaps.append(gap)
    return gaps


@dataclass(frozen=True)
class GapTriage:
    """The analyzer's gap list split into the gaps the resolver searches and the ones it does not.

    ``kept`` is in analyzer order and capped at ``GAP_FILL_MAX_GAPS``; its index is the addendum's
    ``Gap N`` number and the index into the raw record's ``results``. Each ``dropped`` entry is the
    gap dict plus its 1-based analyzer ``position`` and a ``reason`` from ``DROP_REASONS``.
    """

    kept: list[dict[str, Any]]
    dropped: list[dict[str, Any]]

    @property
    def listed(self) -> int:
        return len(self.kept) + len(self.dropped)

    def dropped_for(self, reason: str) -> int:
        return sum(1 for gap in self.dropped if gap["reason"] == reason)


def _grade(gap: dict[str, Any], position: int) -> tuple[str | None, int | None]:
    """The drop reason a gap earns on its own grades (None when every grade passes) and its validated pointer.

    An empty slot, an absent or mistyped boolean grade, or a ``same_need_as`` the analyzer typed but
    that names no earlier position is schema drift and drops the gap, because a grade that defaulted
    to passing would spend exactly the money the grade exists to save; the pointer comes back None in
    that case. A MISSING ``same_need_as`` key reads as null instead. See docs/research.md "v1 triage".
    """
    answerable_now = gap.get("answerable_now")
    already_in_first_pass = gap.get("already_in_first_pass")
    same_need_as = gap.get("same_need_as")
    pointer_well_formed = same_need_as is None or (type(same_need_as) is int and 0 < same_need_as < position)
    if (
        not gap["gap"]
        or not isinstance(answerable_now, bool)
        or not isinstance(already_in_first_pass, bool)
        or not pointer_well_formed
    ):
        return DROP_SCHEMA, None
    if not answerable_now:
        return DROP_NOT_ANSWERABLE, same_need_as
    if already_in_first_pass:
        return DROP_ALREADY_ANSWERED, same_need_as
    return None, same_need_as


def triage_gaps(gaps: Sequence[dict[str, Any]], *, max_gaps: int) -> GapTriage:
    """Drop the gaps the analyzer graded as not worth a search, dedupe restatements, then cap the rest.

    Grades come first (not answerable now, already in the first pass), then ``same_need_as`` is
    followed to the NEED it names, the root of its pointer chain: a restatement of a need that a kept
    gap is searching, or that the first pass already answers, is dropped, while a restatement of a
    need nobody covers (its earlier phrasing was future-dated or ungraded) becomes the need's
    carrier and is kept, and every later restatement of that need is then dropped. The cap applies
    last, so a dropped gap never displaces a kept one. See docs/research.md "v1 triage".
    """
    kept: list[tuple[int, dict[str, Any]]] = []
    dropped: list[dict[str, Any]] = []
    need_of: dict[int, int] = {}
    covered_needs: set[int] = set()
    for position, gap in enumerate(gaps, start=1):
        reason, restates = _grade(gap, position)
        need = need_of[position] = position if restates is None else need_of[restates]
        if reason is None and need in covered_needs:
            reason = DROP_SAME_NEED
        if reason is None:
            kept.append((position, gap))
        else:
            dropped.append({**gap, "position": position, "reason": reason})
        if reason in (None, DROP_ALREADY_ANSWERED):
            covered_needs.add(need)
    for position, gap in kept[max_gaps:]:
        dropped.append({**gap, "position": position, "reason": DROP_OVER_CAP})
    return GapTriage(kept=[gap for _, gap in kept[:max_gaps]], dropped=dropped)


def _format_triage_marker(qid: int | None, triage: GapTriage) -> str:
    counts = " ".join(f"dropped_{reason}={triage.dropped_for(reason)}" for reason in DROP_REASONS)
    return f"GAP_FILL_V1_TRIAGE: question={qid} listed={triage.listed} kept={len(triage.kept)} {counts}"


async def _run_analyzer(
    question: MetaculusQuestion,
    first_pass_research: str,
    *,
    is_benchmarking: bool,
) -> list[dict[str, Any]]:
    """Call the analyzer LLM (no grounding) to identify and grade gaps.

    Runs gpt-6-sol at low effort via OpenRouter (with donated-key
    fallback; gpt-5.6-terra -> gpt-6-sol on the 2026-09-22 migration, Terra
    having no GPT-6 successor). The analyzer is non-grounded so it doesn't need
    Google's search index; the task is gap decomposition (not deep judgment)
    under a tight soft-fail wall cap, so low effort is the latency-safe tier.

    Strict structured output requires every grade; require_parameters prevents a
    provider from silently ignoring the schema. Local triage still validates the
    grades and positional pointers before any resolver spend.
    """
    from metaculus_bot.fallback_openrouter import (  # noqa: PLC0415  # late import: tests patch this at its source module
        build_llm_with_openrouter_fallback,
    )

    llm = build_llm_with_openrouter_fallback(
        model=GAP_FILL_ANALYZER_MODEL,
        role="gap_fill_analyzer",
        reasoning={"effort": "low"},
        # Reasoning models take the provider default. See docs/research.md "v1 implementation notes".
        temperature=None,
        timeout=GAP_FILL_ANALYZER_TIMEOUT,
        allowed_tries=1,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "gap_analysis", "strict": True, "schema": GapAnalysis.model_json_schema()},
        },
        extra_body={"provider": {"require_parameters": True}},
    )
    prompt = gap_fill_analyzer_prompt(
        question_text=question.question_text,
        resolution_criteria=question.resolution_criteria,
        fine_print=question.fine_print,
        first_pass_research=first_pass_research,
        is_benchmarking=is_benchmarking,
        max_gaps=GAP_FILL_MAX_GAPS,
        # A "no coverage of candidate X" gap needs the ballot (q44952). See docs/research.md "v1 implementation notes".
        options=getattr(question, "options", None),
    )
    logger.info(f"GapFill: calling analyzer {GAP_FILL_ANALYZER_MODEL} for gap identification")
    # The wall has headroom over the per-request timeout: 135s vs 120s. See docs/research.md "v1 implementation notes".
    raw_text = await invoke_with_transient_retry(
        lambda: llm.invoke(prompt), wall_timeout=GAP_FILL_ANALYZER_WALL_TIMEOUT, label="gap_fill_analyzer"
    )
    gaps = _parse_gap_list(raw_text)
    logger.info(f"GapFill: analyzer returned {len(gaps)} gap(s)")
    return gaps


async def _resolve_single_gap(
    gap: dict[str, Any],
    question: MetaculusQuestion,
    *,
    is_benchmarking: bool,
) -> str:
    """Run one OpenAI native web search (via OpenRouter) for a single gap.

    Migrated 2026-06-25 off direct-Google grounded Gemini (google-genai, personal
    GOOGLE_API_KEY) to native search on the Metaculus-donated key — this is the
    dominant cost-saving change since the resolver fans out up to GAP_FILL_MAX_GAPS
    calls per question. Runs GAP_FILL_RESOLVER_MODEL at GAP_FILL_RESOLVER_REASONING_EFFORT
    (the same low as the main native_search provider): the workers run in parallel, so
    latency is the slowest call, not the sum.

    Raises on SDK/OpenRouter errors — the caller uses ``asyncio.gather(..., return_exceptions=True)``
    so one failure doesn't kill the rest.
    """
    prompt = gap_fill_search_prompt(
        gap=gap["gap"],
        search_query=gap["search_query"],
        question_text=question.question_text,
        resolution_criteria=question.resolution_criteria,
        fine_print=question.fine_print,
        is_benchmarking=is_benchmarking,
    )
    llm = build_native_search_llm(
        GAP_FILL_RESOLVER_MODEL, reasoning_effort=GAP_FILL_RESOLVER_REASONING_EFFORT, role="gap_fill_resolver"
    )
    # The same shared wall as native_search; a hard cap either way. See docs/research.md "v1 implementation notes".
    return await invoke_with_transient_retry(
        lambda: llm.invoke(prompt), wall_timeout=NATIVE_SEARCH_WALL_TIMEOUT, label="gap_fill_resolver"
    )


async def run_gap_fill_pass(
    question: MetaculusQuestion,
    first_pass_research: str,
    *,
    is_benchmarking: bool = False,
    on_error: Callable[[BaseException], None] | None = None,
) -> str:
    """Identify, triage and resolve factual gaps in first-pass research.

    An analyzer call for the graded gap list, then ``triage_gaps``, then one parallel native web
    search per survivor. The models, the stage detail and the triage rules are in docs/research.md
    "v1: targeted gap-fill".

    Keeps successful research on upstream failure and reports one error per pass through
    ``on_error`` so the run exits nonzero after publishing. Valid empty analyses and
    deliberate triage drops are not failures. Cancellation of the pass still propagates.
    """
    qid = getattr(question, "id_of_question", None)
    try:
        gaps = await _run_analyzer(question, first_pass_research, is_benchmarking=is_benchmarking)
    except _GAP_FILL_SOFT_FAIL_EXCEPTIONS as exc:
        # A dead analyzer looks exactly like a question with no gaps. See docs/research.md "v1 implementation notes".
        logger.warning(f"GAP_FILL_ANALYZER_FAILED: question={qid} error={type(exc).__name__} detail={exc}")
        if on_error is not None:
            on_error(exc)
        # A scheduler checkpoint on the no-op path, for ASYNC910. See docs/research.md "v1 implementation notes".
        await asyncio.sleep(0)
        return ""

    triage = triage_gaps(gaps, max_gaps=GAP_FILL_MAX_GAPS)
    failure: BaseException | None = (
        ValueError(f"Gap-fill analyzer returned {triage.dropped_for(DROP_SCHEMA)} schema-invalid gap(s)")
        if triage.dropped_for(DROP_SCHEMA)
        else None
    )
    # Every slot failing the schema is v1 gone dark while the analyzer still bills: WARN, like a dead analyzer.
    wholesale_schema_drift = triage.listed > 0 and triage.dropped_for(DROP_SCHEMA) == triage.listed
    (logger.warning if wholesale_schema_drift else logger.info)(_format_triage_marker(qid, triage))
    for gap in triage.dropped:
        logger.info(
            f"GapFill: dropped gap #{gap['position']} reason={gap['reason']} "
            f"same_need_as={gap.get('same_need_as')}: {gap['gap']}"
        )

    search_tasks = [_resolve_single_gap(g, question, is_benchmarking=is_benchmarking) for g in triage.kept]
    # One SDK error must not take the whole addendum down. See docs/research.md "v1 implementation notes".
    results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Exceptions serialize to their str via the logger's encoder. See docs/research.md "v1 implementation notes".
    record_raw_research(
        qid=qid,
        provider="gap_fill",
        payload={"gaps": triage.kept, "results": results, "dropped": triage.dropped},
    )

    sections: list[str] = []
    for idx, (gap, res) in enumerate(zip(triage.kept, results, strict=True), start=1):
        if isinstance(res, BaseException):
            logger.warning(f"GapFill: gap #{idx} search failed ({type(res).__name__}): {res}")
            if failure is None:
                failure = res
            continue
        result_text = res
        if not result_text or not result_text.strip():
            continue
        why = gap.get("why_matters", "").strip()
        why_line = f"_Why it matters: {why}_\n\n" if why else ""
        sections.append(f"### Gap {idx}: {gap['gap']}\n\n{why_line}{result_text}")

    if failure is not None and on_error is not None:
        on_error(failure)
    if not sections:
        return ""

    logger.info(f"GapFill: produced addendum from {len(sections)}/{len(triage.kept)} gap resolutions")
    return "\n\n".join(sections)
