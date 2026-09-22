"""Replay one question's gap-fill v1 gaps through the resolver at several models and search context sizes.

Gap-fill v1 (``research/targeted.py``) answers each analyzer gap with one OpenAI native web
search built by ``build_native_search_llm``. That call is the bot's single largest spend line,
about $0.19 a call, most of it retrieved page text billed back as input tokens at the model's
rate (``scratch/cost_pass_2026-09-09/cost_anatomy.md``), and neither of its two price levers,
the model's input price and OpenAI's ``search_context_size``, had been measured. This probe
takes the gaps the archive recorded for ONE question, sends each through the production
resolver path (the same prompt builder and the same LLM builder, with the model and the
context size overridden) at every ``model:context_size`` cell of a grid, and writes the
answers side by side with OpenRouter's own per-call cost and token counts, so the operator
can read what a cheaper cell gives up.

It SPENDS the operator's personal OpenRouter key (gaps x cells calls, up to about $0.20
each), which is why it refuses to run without ``--i-accept-spend`` and prints its ceiling
first. The Metaculus-donated key is never used: the probe forces
``DONATED_OPENROUTER_KEY_ENABLED=false`` before any LLM is built, the switch a Mantic run
uses, so every cost figure it records is the whole charge rather than a BYOK platform fee.

    make probe_resolver QUESTION=44267 ARGS="--i-accept-spend"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forecasting_tools.helpers.metaculus_client import MetaculusClient

from metaculus_bot.constants import (
    DONATED_OPENROUTER_KEY_ENABLED_ENV,
    GAP_FILL_RESOLVER_MODEL,
    GAP_FILL_RESOLVER_REASONING_EFFORT,
    NATIVE_SEARCH_WALL_TIMEOUT,
)
from metaculus_bot.credit_telemetry import (
    NO_TOKENS,
    TokenCounts,
    drain_litellm_callbacks,
    install_role_spend_tracker,
    role_spend_rows,
)
from metaculus_bot.llm_retry import invoke_with_transient_retry
from metaculus_bot.prompts import gap_fill_search_prompt
from metaculus_bot.research.providers import build_native_search_llm

# A literal like gemini_verify.CANDIDATE_MODEL: the repo has not adopted this model, so no constant carries it.
CANDIDATE_MODEL = "openai/gpt-6-luna"
CURRENT_MODEL_ALIAS = "current"
CANDIDATE_MODEL_ALIAS = "luna"
MODEL_ALIASES: dict[str, str] = {CURRENT_MODEL_ALIAS: GAP_FILL_RESOLVER_MODEL, CANDIDATE_MODEL_ALIAS: CANDIDATE_MODEL}
SEARCH_CONTEXT_SIZES: tuple[str, ...] = ("low", "medium", "high")
PRODUCTION_CONTEXT_SIZE = "high"
REASONING_EFFORTS: tuple[str, ...] = ("low", "medium", "high")
PRODUCTION_REASONING_EFFORT = GAP_FILL_RESOLVER_REASONING_EFFORT
DEFAULT_GRID: tuple[str, ...] = (
    "current:high",
    "current:medium",
    "current:low",
    "luna:high",
    "luna:medium",
    "luna:low",
)

PER_CALL_CEILING_USD = 0.20  # the measured production call at search context high; cheaper cells come in under it
MAX_CONCURRENT_CALLS = 6  # close to production's fan-out width, so the recorded wall-clocks stay comparable

ARCHIVE_LATEST_DIR = Path("backtests/research_archive/latest")
ARCHIVE_RAW_DIR = Path("backtests/research_archive/raw")
OUTPUT_DIR = Path("scratch/probes")

_GAP_FILL_SECTION_HEADER = "## Targeted Gap-Fill (second pass)"
_GAP_HEADING = re.compile(r"^### Gap (\d+): ", re.MULTILINE)
_WHY_LINE = re.compile(r"\A\s*_Why it matters: (.*?)_\s*\n", re.DOTALL)
_TOP_LEVEL_HEADING = re.compile(r"^## ", re.MULTILINE)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridCell:
    """One ``model:context_size[:effort]`` cell of the grid; the effort defaults to production's."""

    model_alias: str
    model_slug: str
    search_context_size: str
    reasoning_effort: str = PRODUCTION_REASONING_EFFORT

    @property
    def model_label(self) -> str:
        """The model alias, suffixed with ``@effort`` when the effort is not production's."""
        if self.reasoning_effort == PRODUCTION_REASONING_EFFORT:
            return self.model_alias
        return f"{self.model_alias}@{self.reasoning_effort}"

    @property
    def label(self) -> str:
        if self.reasoning_effort == PRODUCTION_REASONING_EFFORT:
            return f"{self.model_alias}:{self.search_context_size}"
        return f"{self.model_alias}:{self.search_context_size}:{self.reasoning_effort}"


@dataclass(frozen=True)
class ArchivedGap:
    """One gap as the archive recorded it, with production's answer for comparison."""

    index: int
    gap: str
    search_query: str
    why_matters: str
    archived_answer: str


@dataclass(frozen=True)
class ArchivedQuestion:
    """The archive's latest ``artifact`` record for one question, reduced to what the probe replays."""

    question_id: int
    post_id: int
    run_id: str
    timestamp: str
    question_text: str
    gaps: tuple[ArchivedGap, ...]


@dataclass(frozen=True)
class QuestionWording:
    """What the resolver prompt reads off the question: title, resolution criteria, fine print."""

    question_text: str
    resolution_criteria: str | None
    fine_print: str | None


@dataclass
class CellResult:
    """One resolver call: the answer (or the error) plus OpenRouter's accounting for it.

    Mutable because the accounting arrives after the call, from the ledger join in ``run_grid``.
    """

    gap_index: int
    cell: GridCell
    role: str
    answer: str | None
    error: str | None
    wall_s: float
    calls: int = 0
    byok_calls: int = 0  # must stay 0: a BYOK call means the donated key served it after all
    cost_usd: float | None = None
    tokens: TokenCounts = NO_TOKENS

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class CellSummary:
    """One grid cell summed over its gaps."""

    cell: GridCell
    answered: int
    failed: int
    cost_usd: float | None
    tokens: TokenCounts
    mean_wall_s: float

    @property
    def mean_cost_usd(self) -> float | None:
        return None if self.cost_usd is None or self.answered == 0 else self.cost_usd / self.answered


@dataclass(frozen=True)
class ProbeRun:
    """One finished run: what was asked, of whom, and every answer with its accounting."""

    question: ArchivedQuestion
    wording: QuestionWording
    cells: tuple[GridCell, ...]
    results: tuple[CellResult, ...]
    run_at: datetime


# --- The archive ------------------------------------------------------------------------------


def parse_archived_gaps(research_text: str) -> list[ArchivedGap]:
    """Read the ``### Gap N:`` sections of the archived research text.

    Since 2026-09-09 the index is the survivor ordinal: ``run_gap_fill_pass`` numbers the gaps
    that passed triage (``docs/research.md`` "v1 triage"), and a triage-dropped gap's analyzer
    position lives only in the raw record's ``dropped`` list. Before that date it was the
    analyzer's own position over the full list. Either way a gap whose resolver failed leaves a
    hole. The suggested search query is not rendered; the
    caller fills it from the raw analyzer record when one exists, else from the gap text, which
    is the analyzer's own default when it omits one.
    """
    start = research_text.find(_GAP_FILL_SECTION_HEADER)
    if start == -1:
        return []
    body_start = start + len(_GAP_FILL_SECTION_HEADER)
    next_heading = _TOP_LEVEL_HEADING.search(research_text, body_start)
    section = research_text[body_start : next_heading.start() if next_heading else len(research_text)]

    parts = _GAP_HEADING.split(section)
    gaps: list[ArchivedGap] = []
    for index_text, rest in zip(parts[1::2], parts[2::2], strict=True):
        gap_text, _, remainder = rest.partition("\n")
        why_match = _WHY_LINE.match(remainder)
        why = why_match.group(1).strip() if why_match else ""
        answer = remainder[why_match.end() :] if why_match else remainder
        gaps.append(
            ArchivedGap(
                index=int(index_text),
                gap=gap_text.strip(),
                search_query=gap_text.strip(),
                why_matters=why,
                archived_answer=answer.strip(),
            )
        )
    return gaps


def raw_analyzer_gaps(raw_dir: Path, run_id: str, question_id: int) -> list[dict[str, str]] | None:
    """The gaps the resolver searched, from the run's raw research log, or None when the run has none.

    ``record_raw_research`` writes the ``gap_fill`` payload as ``{"gaps": [...], "results": [...]}``,
    plus ``"dropped": [...]`` since 2026-09-09 (the triage-dropped gaps with their analyzer position
    and reason; ``gaps`` is then the survivors, aligned with ``results`` and the rendered index), and
    ``scripts/download_raw_research.py`` archives one ``<run_id>.jsonl`` per run.
    """
    path = raw_dir / f"{run_id}.jsonl"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("qid") == question_id and record.get("provider") == "gap_fill":
            return list(record["payload"]["gaps"])
    return None


def load_archived_question(question_id: int, *, latest_dir: Path, raw_dir: Path) -> ArchivedQuestion:
    """Load the archive's latest record for ``question_id`` and its gap-fill v1 gaps.

    Only an ``artifact`` record is accepted: the ``comment_backfill`` and ``log_backfill`` classes
    carry a reconstructed research text whose gap section may be trimmed, and the archive rule is
    never to pool the three. When the run's raw analyzer record exists its gap list is
    authoritative (it carries the suggested search queries, and gaps whose resolver failed in
    production); the rendered section then only supplies production's answers.
    """
    record = json.loads((latest_dir / f"{question_id}.json").read_text(encoding="utf-8"))
    if record["source"] != "artifact":
        raise ValueError(
            f"question {question_id}: the latest archive record is {record['source']!r}, not 'artifact'; "
            "the probe replays only archive records written by the bot itself"
        )
    rendered = {gap.index: gap for gap in parse_archived_gaps(record["research_text"])}
    run_id = str(record["run_id"])
    raw_gaps = raw_analyzer_gaps(raw_dir, run_id, question_id)
    if raw_gaps is None:
        gaps = tuple(rendered[index] for index in sorted(rendered))
    else:
        gaps = tuple(
            ArchivedGap(
                index=index,
                gap=str(raw["gap"]).strip(),
                search_query=str(raw.get("search_query") or raw["gap"]).strip(),
                why_matters=str(raw.get("why_matters", "")).strip(),
                archived_answer=rendered[index].archived_answer if index in rendered else "",
            )
            for index, raw in enumerate(raw_gaps, start=1)
        )
    if not gaps:
        raise ValueError(f"question {question_id}: the archive record carries no gap-fill v1 gaps to replay")
    return ArchivedQuestion(
        question_id=question_id,
        post_id=int(record["post_id"]),
        run_id=run_id,
        timestamp=str(record["timestamp"]),
        question_text=str(record["question_text"]),
        gaps=gaps,
    )


def select_gaps(gaps: Sequence[ArchivedGap], selection: str | None) -> list[ArchivedGap]:
    """Narrow to the comma-separated gap indices in ``selection``; ``None`` keeps every gap."""
    if selection is None:
        return list(gaps)
    wanted = [int(token) for token in selection.split(",") if token.strip()]
    by_index = {gap.index: gap for gap in gaps}
    missing = [index for index in wanted if index not in by_index]
    if missing:
        raise ValueError(f"--gaps names gap(s) {missing} but the archive has {sorted(by_index)}")
    return [by_index[index] for index in wanted]


def fetch_question_wording(post_id: int) -> QuestionWording:
    """Read the title, resolution criteria and fine print off the Metaculus API (read-only, free).

    The archive record and the residual datasets carry the title only, and the resolver prompt
    reads the criteria, so this one read is what makes the replay faithful. Needs METACULUS_TOKEN.
    """
    question = MetaculusClient().get_question_by_post_id(post_id)
    return QuestionWording(
        question_text=question.question_text,
        resolution_criteria=question.resolution_criteria,
        fine_print=question.fine_print,
    )


# --- The grid ---------------------------------------------------------------------------------


def _split_cell_spec(spec: str) -> tuple[str, str, str]:
    """``model:size`` or ``model:size:effort`` from the right, so a slug carrying a colon still parses."""
    head, separator, last = spec.rpartition(":")
    if separator and last in REASONING_EFFORTS:
        model_token, size_separator, size = head.rpartition(":")
        if size_separator and size in SEARCH_CONTEXT_SIZES:
            return model_token, size, last
    if separator and last in SEARCH_CONTEXT_SIZES:
        return head, last, PRODUCTION_REASONING_EFFORT
    raise ValueError(
        f"grid cell {spec!r}: expected <model>:<{'|'.join(SEARCH_CONTEXT_SIZES)}>[:<{'|'.join(REASONING_EFFORTS)}>]"
    )


def parse_grid(specs: Sequence[str]) -> list[GridCell]:
    """Turn ``model:context_size[:effort]`` specs into cells; the model is an alias or a bare OpenRouter slug."""
    cells: list[GridCell] = []
    for spec in specs:
        model_token, size, effort = _split_cell_spec(spec)
        slug = MODEL_ALIASES.get(model_token, model_token)
        if "/" not in slug:
            aliases = ", ".join(MODEL_ALIASES)
            raise ValueError(
                f"grid cell {spec!r}: model must be an alias ({aliases}) or a vendor/model OpenRouter slug"
            )
        cell = GridCell(model_alias=model_token, model_slug=slug, search_context_size=size, reasoning_effort=effort)
        if cell in cells:
            raise ValueError(f"grid cell {spec!r} is listed twice")
        cells.append(cell)
    return cells


def estimate_ceiling_usd(n_gaps: int, n_cells: int) -> float:
    return n_gaps * n_cells * PER_CALL_CEILING_USD


def print_cost_estimate(question: ArchivedQuestion, gaps: Sequence[ArchivedGap], cells: Sequence[GridCell]) -> None:
    """State what this run spends, and on which assumptions, before anything is called."""
    n_calls = len(gaps) * len(cells)
    print("Estimated cost of this run")
    print(
        f"  Question {question.question_id} (post {question.post_id}), archive run {question.run_id} "
        f"at {question.timestamp}: {len(gaps)} gap(s) selected of {len(question.gaps)} archived"
    )
    print(f"  Grid, {len(cells)} cell(s):")
    for cell in cells:
        print(
            f"    {cell.label:<22} {cell.model_slug} at search_context_size={cell.search_context_size}, "
            f"reasoning effort {cell.reasoning_effort}"
        )
    print(f"  {len(gaps)} x {len(cells)} = {n_calls} resolver calls on the operator's PERSONAL OpenRouter key")
    print(
        f"  Ceiling: ${estimate_ceiling_usd(len(gaps), len(cells)):.2f} at ${PER_CALL_CEILING_USD:.2f} a call, the\n"
        "  measured production figure for the current model at search context high\n"
        "  (scratch/cost_pass_2026-09-09/cost_anatomy.md). Smaller contexts and the cheaper model come in\n"
        "  under it; OpenRouter's per-call usage, recorded after every call, is the real number."
    )
    print()


# --- The calls --------------------------------------------------------------------------------


def probe_role(gap_index: int, cell: GridCell) -> str:
    """The CREDIT_ROLE_SPEND role one call books under; unique per call so the ledger is per call."""
    return f"probe:gap{gap_index}:{cell.label}"


def _route_to_the_personal_key() -> None:
    """Force personal-key routing before any LLM is built, whatever ``.env`` says.

    The donated key is Metaculus's grant for its own tournaments, so an experiment bills the
    operator; the same switch makes OpenRouter's ``usage.cost`` the whole charge (no BYOK split).
    """
    os.environ[DONATED_OPENROUTER_KEY_ENABLED_ENV] = "false"


async def _resolve_cell(
    gap: ArchivedGap, cell: GridCell, wording: QuestionWording, semaphore: asyncio.Semaphore
) -> CellResult:
    """One resolver call through the production path, with the model and context size overridden."""
    prompt = gap_fill_search_prompt(
        gap=gap.gap,
        search_query=gap.search_query,
        question_text=wording.question_text,
        resolution_criteria=wording.resolution_criteria,
        fine_print=wording.fine_print,
    )
    role = probe_role(gap.index, cell)
    llm = build_native_search_llm(
        cell.model_slug,
        reasoning_effort=cell.reasoning_effort,
        search_context_size=cell.search_context_size,
        role=role,
    )
    async with semaphore:
        started = time.monotonic()
        # One failed cell must not lose the other cells' paid answers; the same shape as the pass.
        (outcome,) = await asyncio.gather(
            invoke_with_transient_retry(
                lambda: llm.invoke(prompt), wall_timeout=NATIVE_SEARCH_WALL_TIMEOUT, label=role
            ),
            return_exceptions=True,
        )
        wall_s = time.monotonic() - started
    if isinstance(outcome, BaseException):
        print(f"  {role}: FAILED after {wall_s:.1f}s ({type(outcome).__name__}: {outcome})")
        return CellResult(gap_index=gap.index, cell=cell, role=role, answer=None, error=repr(outcome), wall_s=wall_s)
    print(f"  {role}: {len(outcome)} chars in {wall_s:.1f}s")
    return CellResult(gap_index=gap.index, cell=cell, role=role, answer=outcome, error=None, wall_s=wall_s)


def _attach_ledger_usage(results: Sequence[CellResult]) -> None:
    """Join OpenRouter's per-call accounting onto each result by its unique role.

    ``charged_usd`` rather than ``usd``: the latter adds the echoed upstream cost to a non-BYOK
    call and so double counts exactly the personal-key calls this probe makes.
    """
    rows = {row.role: row for row in role_spend_rows()}
    for result in results:
        row = rows.get(result.role)
        if row is not None:
            result.calls, result.byok_calls = row.calls, row.byok_calls
            result.cost_usd, result.tokens = row.charged_usd, row.tokens


async def run_grid(
    gaps: Sequence[ArchivedGap], cells: Sequence[GridCell], wording: QuestionWording
) -> list[CellResult]:
    """Every gap at every cell, MAX_CONCURRENT_CALLS at a time, then the ledger join.

    The usage comes from the production ``CREDIT_ROLE_SPEND`` tracker (a litellm success
    callback), so the dollars here are the same dollars a run log books; the drain must run on
    the loop the completions ran on, which is why it sits inside this coroutine.
    """
    install_role_spend_tracker()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    results = await asyncio.gather(*(_resolve_cell(gap, cell, wording, semaphore) for gap in gaps for cell in cells))
    await drain_litellm_callbacks()
    _attach_ledger_usage(results)
    return list(results)


# --- The write-up -----------------------------------------------------------------------------


def summarize_cells(cells: Sequence[GridCell], results: Sequence[CellResult]) -> list[CellSummary]:
    """Per-cell totals over the gaps: answered/failed counts, dollars, tokens and mean wall-clock."""
    summaries: list[CellSummary] = []
    for cell in cells:
        own = [result for result in results if result.cell == cell]
        costed = [result.cost_usd for result in own if result.cost_usd is not None]
        tokens = NO_TOKENS
        for result in own:
            tokens = tokens + result.tokens
        summaries.append(
            CellSummary(
                cell=cell,
                answered=sum(result.succeeded for result in own),
                failed=sum(not result.succeeded for result in own),
                cost_usd=sum(costed) if costed else None,
                tokens=tokens,
                mean_wall_s=sum(result.wall_s for result in own) / len(own) if own else 0.0,
            )
        )
    return summaries


def cost_ratios(summaries: Sequence[CellSummary]) -> list[tuple[str, float | None]]:
    """The two families of ratio the grid is for: each size against high within one model and effort,
    and each model or effort against the production model at the same size. Fully costed cells only;
    ``None`` otherwise, including when the grid lacks the cell a ratio divides by."""
    cost = {summary.cell: summary.cost_usd for summary in summaries if summary.failed == 0}
    cells = [summary.cell for summary in summaries]
    ratios: list[tuple[str, float | None]] = []

    def ratio(numerator: GridCell, denominator: GridCell) -> float | None:
        top, bottom = cost.get(numerator), cost.get(denominator)
        return None if top is None or not bottom else top / bottom

    for cell in cells:
        if cell.search_context_size != PRODUCTION_CONTEXT_SIZE:
            at_high = GridCell(cell.model_alias, cell.model_slug, PRODUCTION_CONTEXT_SIZE, cell.reasoning_effort)
            ratios.append(
                (f"{cell.model_label} {cell.search_context_size}/{PRODUCTION_CONTEXT_SIZE}", ratio(cell, at_high))
            )
    for cell in cells:
        if cell.model_alias == CURRENT_MODEL_ALIAS and cell.reasoning_effort == PRODUCTION_REASONING_EFFORT:
            continue
        production = GridCell(CURRENT_MODEL_ALIAS, GAP_FILL_RESOLVER_MODEL, cell.search_context_size)
        ratios.append(
            (f"{cell.model_label}/{CURRENT_MODEL_ALIAS} at {cell.search_context_size}", ratio(cell, production))
        )
    return ratios


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def render_markdown(run: ProbeRun) -> str:
    """The side-by-side read: question, per-cell summary and ratios, then every gap with its answers."""
    question, wording, cells, results = run.question, run.wording, run.cells, run.results
    summaries = summarize_cells(cells, results)
    gaps_run = sorted({result.gap_index for result in results})
    lines = [
        f"# Gap-fill resolver probe: question {question.question_id} (post {question.post_id})",
        "",
        f"Run at {run.run_at.isoformat(timespec='seconds')} on the operator's personal OpenRouter key; gaps from "
        f"archive run {question.run_id} ({question.timestamp}); {len(gaps_run)} gap(s) x {len(cells)} cell(s) = "
        f"{len(results)} resolver calls; production is {CURRENT_MODEL_ALIAS}:{PRODUCTION_CONTEXT_SIZE} at reasoning "
        f"effort {PRODUCTION_REASONING_EFFORT}.",
        "",
        "## Question",
        "",
        f"**{wording.question_text}**",
        "",
        f"Resolution criteria: {wording.resolution_criteria or '(none)'}",
        "",
        f"Fine print: {wording.fine_print or '(none)'}",
        "",
        "## Summary per cell",
        "",
        "| cell | model | context | effort | answered | failed | total cost | mean cost/call | prompt tokens | completion | cached | reasoning | mean wall s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.cell.label} | {summary.cell.model_slug} | {summary.cell.search_context_size} | "
            f"{summary.cell.reasoning_effort} | "
            f"{summary.answered} | {summary.failed} | {_usd(summary.cost_usd)} | {_usd(summary.mean_cost_usd)} | "
            f"{summary.tokens.prompt} | {summary.tokens.completion} | {summary.tokens.cached} | "
            f"{summary.tokens.reasoning} | {summary.mean_wall_s:.1f} |"
        )
    lines += ["", "## Cost ratios (total over the same gaps; n/a when a cell has a failed or uncosted call)", ""]
    lines += [f"- {label}: {_ratio(value)}" for label, value in cost_ratios(summaries)]

    by_gap = {gap.index: gap for gap in question.gaps}
    for gap_index in gaps_run:
        gap = by_gap[gap_index]
        lines += [
            "",
            f"## Gap {gap.index}: {gap.gap}",
            "",
            f"_Why it matters: {gap.why_matters or '(not recorded)'}_",
            "",
            f"Suggested search query: {gap.search_query}",
            "",
            f"### Archived production answer (run {question.run_id})",
            "",
            gap.archived_answer or "(no archived answer: the resolver failed or returned nothing in production)",
        ]
        for result in (result for result in results if result.gap_index == gap_index):
            lines += [
                "",
                f"### {result.cell.label} ({result.cell.model_slug}, context {result.cell.search_context_size}, "
                f"effort {result.cell.reasoning_effort}): "
                f"{_usd(result.cost_usd)}, {result.tokens.prompt} prompt / {result.tokens.completion} completion tokens "
                f"({result.tokens.cached} cached, {result.tokens.reasoning} reasoning), {result.wall_s:.1f} s, "
                f"{result.calls} billed call(s)",
                "",
                f"ERROR: {result.error}" if result.error is not None else (result.answer or ""),
            ]
    lines.append("")
    return "\n".join(lines)


def build_payload(run: ProbeRun) -> dict[str, Any]:
    """Everything the run measured, JSON-ready, so the Markdown can be re-rendered or re-analysed later."""
    summaries = summarize_cells(run.cells, run.results)
    return {
        "run_at": run.run_at.isoformat(timespec="seconds"),
        "key": "personal",
        "production_cell": asdict(GridCell(CURRENT_MODEL_ALIAS, GAP_FILL_RESOLVER_MODEL, PRODUCTION_CONTEXT_SIZE)),
        "question": asdict(run.question),
        "wording": asdict(run.wording),
        "cells": [asdict(cell) for cell in run.cells],
        "results": [asdict(result) for result in run.results],
        "summary": [asdict(summary) | {"mean_cost_usd": summary.mean_cost_usd} for summary in summaries],
        "ratios": dict(cost_ratios(summaries)),
    }


def write_outputs(output_dir: Path, run: ProbeRun) -> Path:
    """Write ``<stem>.json`` and ``<stem>.md`` under ``output_dir``; returns the stem."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"gap_fill_resolver_probe_{run.question.question_id}_{run.run_at:%Y%m%dT%H%M%SZ}"
    stem.with_suffix(".json").write_text(json.dumps(build_payload(run), indent=2, ensure_ascii=False), encoding="utf-8")
    stem.with_suffix(".md").write_text(render_markdown(run), encoding="utf-8")
    return stem


# --- Entry point ------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one archived question's gap-fill v1 gaps through the resolver across models and "
        "search context sizes. SPENDS MONEY on the operator's personal OpenRouter key."
    )
    parser.add_argument("--question", type=int, required=True, help="Metaculus question id (not the post id).")
    parser.add_argument(
        "--grid",
        nargs="+",
        default=list(DEFAULT_GRID),
        metavar="MODEL:SIZE",
        help=f"Cells to run; a model is an alias ({', '.join(MODEL_ALIASES)}) or an OpenRouter slug "
        f"(default: {' '.join(DEFAULT_GRID)}).",
    )
    parser.add_argument("--gaps", default=None, help="Comma-separated archive gap indices to run (default: all).")
    parser.add_argument(
        "--i-accept-spend",
        action="store_true",
        help="Required. Confirms you accept gaps x cells live, billed resolver calls on your own OpenRouter key.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cells = parse_grid(args.grid)
    question = load_archived_question(args.question, latest_dir=ARCHIVE_LATEST_DIR, raw_dir=ARCHIVE_RAW_DIR)
    gaps = select_gaps(question.gaps, args.gaps)
    print_cost_estimate(question, gaps, cells)
    if not args.i_accept_spend:
        print(
            f"Refusing to run: this probe makes {len(gaps) * len(cells)} billed calls. "
            "Re-run with --i-accept-spend to proceed."
        )
        raise SystemExit(2)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _route_to_the_personal_key()
    wording = fetch_question_wording(question.post_id)
    run_at = datetime.now(UTC)
    print(f"Resolving {len(gaps)} gap(s) x {len(cells)} cell(s), {MAX_CONCURRENT_CALLS} calls at a time")
    results = asyncio.run(run_grid(gaps, cells, wording))

    stem = write_outputs(OUTPUT_DIR, ProbeRun(question, wording, tuple(cells), tuple(results), run_at))
    print()
    for summary in summarize_cells(cells, results):
        print(
            f"  {summary.cell.label:<16} answered={summary.answered} failed={summary.failed} "
            f"cost={_usd(summary.cost_usd)} prompt_tokens={summary.tokens.prompt} "
            f"completion_tokens={summary.tokens.completion} mean_wall_s={summary.mean_wall_s:.1f}"
        )
    total = sum(result.cost_usd or 0.0 for result in results)
    print(f"  total charged: ${total:.4f} over {sum(result.calls for result in results)} billed call(s)")
    print(f"Wrote {stem}.json and {stem}.md")


if __name__ == "__main__":
    main()
