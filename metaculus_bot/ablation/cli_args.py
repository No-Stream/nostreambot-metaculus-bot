"""Command-line surface and run plan for the ablation benchmark.

Everything that turns an argv list into a plan the orchestrator can execute:

* the ``--stages`` token list, the static force-cascade map, and the rate-limit dial;
* ``_build_parser`` — the argparse definition, whose help text is the operator-facing
  documentation for every knob;
* ``_StagePlan`` / ``_ArmStage`` — the parsed per-invocation plan and the aggregation-arm
  table ``ablation.cli`` iterates.

Split out of ``ablation.cli`` so the ~240-line parser and the stage vocabulary can be read
(and parser tests written) without the stage bodies in the way.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from metaculus_bot.ablation.prune import DEFAULT_BATCH_SIZE as PRUNE_DEFAULT_BATCH_SIZE
from metaculus_bot.ablation.qa_iterate import (
    DEFAULT_FORECASTABILITY_THRESHOLD as QA_ITERATE_DEFAULT_FORECASTABILITY_THRESHOLD,
)
from metaculus_bot.ablation.qa_iterate import (
    DEFAULT_LEAKAGE_THRESHOLD as QA_ITERATE_DEFAULT_LEAKAGE_THRESHOLD,
)
from metaculus_bot.ablation.qa_iterate import (
    DEFAULT_MAX_ITERATIONS as QA_ITERATE_DEFAULT_MAX_ITERATIONS,
)
from metaculus_bot.ablation.run_pdf import ARM_PDF_MIN2
from metaculus_bot.ablation.run_stacker import ARM_MEAN, ARM_MEDIAN, ARM_PDF, ARM_STACK, ARM_STACK_AUG

STAGES: list[str] = [
    "fetch",
    "research",
    "prune",
    "screen",
    "qa_iterate",
    "forecast",
    "stack",
    "stack_aug",
    "pdf",
    "median",
    "mean",
    "score",
]
DEFAULT_TOURNAMENTS: list[str] = ["spring-aib-2026"]
DEFAULT_RESOLVED_AFTER: str = "2026-01-01"
DEFAULT_CACHE_DIR: str = "backtests/ablation"

# Static cascade map: forcing a stage on the left invalidates the caches of
# every stage on the right (transitive closure already pre-computed). Without
# this, --force-stages forecast leaves stale stacker payloads on disk derived
# from the OLD forecaster outputs, and the next score run quietly compares
# fresh-vs-stale arms. See cli_audit_20260515.md (C1) for the operator footgun.
_FORCE_CASCADES: dict[str, set[str]] = {
    "fetch": set(),
    "research": {"prune", "screen", "qa_iterate", "forecast", "stack", "stack_aug", "pdf", "median", "mean"},
    "prune": {"screen", "qa_iterate", "forecast", "stack", "stack_aug", "pdf", "median", "mean"},
    "screen": {"qa_iterate"},
    "qa_iterate": set(),
    "forecast": {"stack", "stack_aug", "pdf", "median", "mean"},
    "stack": set(),
    "stack_aug": set(),
    "pdf": set(),
    "median": set(),
    "mean": set(),
    "score": set(),
}


def _expand_forced_stages(forced: set[str]) -> set[str]:
    """Apply the static cascade map to ``forced`` and return the expansion."""
    expanded = set(forced)
    for stage in forced:
        expanded.update(_FORCE_CASCADES.get(stage, set()))
    return expanded


# Rate-limit dial mapping. Each preset trades off wall-clock speed vs. tolerance
# for upstream-provider 429s. ``fast`` is the historical behavior; ``gentle`` is
# the new default and pairs lower per-forecaster parallelism with a longer retry
# budget so a single 429 wave doesn't shed forecasters from the lineup. ``slow``
# serializes per-forecaster runs entirely — useful on medium runs where wall-
# clock matters less than completing every cell. ``patient`` keeps ``slow``'s
# concurrency=1 but bumps the retry budget to 8 — "slow but persistent" for
# free-tier providers (qwen, minimax, gemma-4-26b) that frequently shed
# forecasters under tight retry budgets even though successive attempts often
# succeed.
_RATE_LIMIT_MODES: tuple[str, ...] = ("fast", "gentle", "slow", "patient")
_RATE_LIMIT_MODE_TO_KWARGS: dict[str, dict[str, int]] = {
    "fast": {"per_forecaster_concurrency": 4, "max_retries": 1},
    "gentle": {"per_forecaster_concurrency": 2, "max_retries": 3},
    "slow": {"per_forecaster_concurrency": 1, "max_retries": 5},
    "patient": {"per_forecaster_concurrency": 1, "max_retries": 8},
}


def _rate_limit_mode_kwargs(mode: str) -> dict[str, int]:
    """Return the (per_forecaster_concurrency, max_retries) kwargs for a mode."""
    return dict(_RATE_LIMIT_MODE_TO_KWARGS[mode])


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_strings(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_stages_arg(raw: str) -> list[str]:
    parsed = _parse_csv_strings(raw)
    invalid = [s for s in parsed if s not in STAGES]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid stage(s): {invalid}; valid stages: {STAGES}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ablation",
        description="Probabilistic-tools ablation benchmark — A/B test of PROBABILISTIC_TOOLS_ENABLED.",
    )
    parser.add_argument("--num-binary", type=int, default=0, help="Target binary question count.")
    parser.add_argument("--num-multiple-choice", type=int, default=0, help="Target MC question count.")
    parser.add_argument("--num-numeric", type=int, default=0, help="Target numeric question count.")
    parser.add_argument(
        "--qids",
        type=_parse_csv_ints,
        default=None,
        help=(
            "Comma-separated explicit qid list; bypasses fetching. When combined with "
            "--stages subsets that omit fetch, the working set is filtered to these qids "
            "after manifest hydration so downstream stages run only over the requested qids."
        ),
    )
    parser.add_argument(
        "--tournaments",
        type=_parse_csv_strings,
        default=DEFAULT_TOURNAMENTS,
        help="Comma-separated tournament slugs; default: spring-aib-2026.",
    )
    parser.add_argument(
        "--resolved-after",
        type=str,
        default=DEFAULT_RESOLVED_AFTER,
        help="ISO date YYYY-MM-DD; lower bound on actual_resolution_time.",
    )
    parser.add_argument(
        "--resolved-before",
        type=str,
        default=None,
        help="ISO date YYYY-MM-DD; optional upper bound on actual_resolution_time.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Disk cache root; default: {DEFAULT_CACHE_DIR}.",
    )
    parser.add_argument(
        "--stages",
        type=_parse_stages_arg,
        default=list(STAGES),
        help=f"Comma-separated subset of {STAGES}; default: all stages.",
    )
    parser.add_argument(
        "--qa-research",
        action="store_true",
        help="Halt after the screen stage; dump first 5 qids' research+verdict to a QA markdown file.",
    )
    parser.add_argument(
        "--force-stages",
        type=_parse_stages_arg,
        default=[],
        help=(
            "Comma-separated stages to re-run (bypass cache reads). Other stages still read cache. "
            "Forcing a stage AUTO-CASCADES to every downstream stage whose inputs change: "
            "research → prune,screen,qa_iterate,forecast,stack,stack_aug,pdf,median; "
            "prune → screen,qa_iterate,forecast,stack,stack_aug,pdf,median; "
            "screen → qa_iterate; forecast → stack,pdf. "
            "Without the cascade, downstream caches would silently serve stale outputs "
            "derived from the prior upstream artifact."
        ),
    )
    parser.add_argument(
        "--per-question-sleep",
        type=int,
        default=30,
        help=(
            "Seconds to sleep BETWEEN STAGES, AFTER each API-firing stage "
            "(research, prune, screen, qa_iterate, forecast, stack, pdf). "
            "Total pause for a full pipeline = 7 × value. Despite the name this is "
            "per-stage, not per-question: a 30-question run with --per-question-sleep=30 "
            "pauses ~210s total (7 stages × 30s), not 900s. Set to 0 to disable. "
            "Increase to back off OpenRouter rate limits. "
            "TODO: real per-question pacing would require restructuring run_forecasters_batch's "
            "asyncio.gather into a serial loop (or a per-question post-release sleep); "
            "documented but deliberately not shipped here."
        ),
    )
    parser.add_argument(
        "--gap-fill-max-gaps",
        type=int,
        default=3,
        help="Maximum number of gap-fill searches per question; default: 3 (no-op when --no-gap-fill is set).",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.5-flash",
        help=(
            "Gemini model for research grounded search. Default: gemini-2.5-flash "
            "(fully free at our scale per Google AI Studio rate limits). "
            "Production tournament uses gemini-3-flash-preview which requires Tier 1 billing. "
            "Canonical: this flag overrides any GEMINI_SEARCH_MODEL shell env var for the run."
        ),
    )
    gap_fill_group = parser.add_mutually_exclusive_group()
    gap_fill_group.add_argument(
        "--gap-fill",
        dest="gap_fill",
        action="store_true",
        help=(
            "Enable second-pass gap-fill grounded search. Off by default for the ablation; "
            "gap-fill amplifies leakage on resolved questions because the analyzer hunts for "
            "'factual gaps' which reliably surface resolution-revealing sentences."
        ),
    )
    gap_fill_group.add_argument(
        "--no-gap-fill",
        dest="gap_fill",
        action="store_false",
        help="Disable second-pass gap-fill (default). The benchmark deliberately uses single-pass Gemini.",
    )
    parser.set_defaults(gap_fill=False)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Global ceiling for OpenRouter parallelism; default: 4.",
    )
    # Default is ``patient`` (concurrency=1, max_retries=8) as of 2026-05-14
    # (Phase A.3 Package 3a). At 50q × 5 forecasters = 250 calls per arm,
    # ``gentle`` (concurrency=2, max_retries=3) was thrashing free-tier
    # per-minute throttles (qwen / minimax / gemma-4-26b) and bleeding
    # forecasters off the lineup — `patient`'s extra retry budget rides out
    # the 429 storms at the cost of wall-clock. Operators with a smoke
    # workload (≤4q) can opt back into ``gentle`` or ``fast``.
    parser.add_argument(
        "--rate-limit-mode",
        type=str,
        choices=list(_RATE_LIMIT_MODES),
        default="patient",
        help=(
            "Rate-limit dial. Maps to (per_forecaster_concurrency, max_retries) tuples: "
            "'fast' (4, 1) — historical behavior, lowest wall-clock; "
            "'gentle' (2, 3) — tolerates a single 429 wave per forecaster; "
            "'slow' (1, 5) — full serialization, for medium runs where wall-clock is secondary; "
            "'patient' (1, 8) — current default: slow but persistent, more retry "
            "budget to ride out free-tier 429 storms (qwen / minimax / gemma-4-26b) "
            "at 50q+ scale."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reproducible bootstrap CIs in scoring; default: 0.",
    )
    parser.add_argument(
        "--qa-iterate-mode",
        type=str,
        choices=["halt", "advisory", "skip"],
        default="halt",
        help=(
            "Mode for the qa_iterate stage. 'halt' (default) writes the QA summary then raises "
            "RuntimeError so the operator can review manual_rejects.json before forecast spend. "
            "'advisory' writes the summary but proceeds to forecast. 'skip' bypasses the stage entirely."
        ),
    )
    parser.add_argument(
        "--qa-iterate-max-iterations",
        type=int,
        default=QA_ITERATE_DEFAULT_MAX_ITERATIONS,
        help=f"Max iterations per qid in qa_iterate; default: {QA_ITERATE_DEFAULT_MAX_ITERATIONS}.",
    )
    parser.add_argument(
        "--qa-iterate-leakage-threshold",
        type=float,
        default=QA_ITERATE_DEFAULT_LEAKAGE_THRESHOLD,
        help=(f"Verifier leakage_risk threshold for accepting a qid; default: {QA_ITERATE_DEFAULT_LEAKAGE_THRESHOLD}."),
    )
    parser.add_argument(
        "--qa-iterate-forecastability-threshold",
        type=float,
        default=QA_ITERATE_DEFAULT_FORECASTABILITY_THRESHOLD,
        help=(
            "Verifier forecastability threshold below which a clean blob is rejected as "
            f"too thin to forecast from; default: {QA_ITERATE_DEFAULT_FORECASTABILITY_THRESHOLD}. "
            "At 50q+ the modal smoke-run forecastability sits near 0.18 — operators can "
            "tighten to 0.25 or relax to 0.15 after seeing the iteration distribution."
        ),
    )
    parser.add_argument(
        "--prune-batch-size",
        type=int,
        default=PRUNE_DEFAULT_BATCH_SIZE,
        help=(
            f"Redactor batch size; default: {PRUNE_DEFAULT_BATCH_SIZE}. Lower = smaller "
            "blast radius on flaky runs (each subprocess failure drops at most batch_size "
            "qids before per-qid recovery kicks in)."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help=(
            "Logging level for stage transitions and per-qid verdicts. Default INFO. "
            "Set DEBUG for subprocess invocations and raw API responses. "
            "Logs are tee'd to stderr and to <cache-dir>/logs/run_<timestamp>.log."
        ),
    )
    parser.add_argument(
        "--lineup",
        type=str,
        choices=["free", "prod"],
        default="free",
        help=(
            "Forecaster ensemble: 'free' for the 4-model free-tier ablation (default), "
            "'prod' for the 3-model paid ensemble (claude-opus-4.6, claude-opus-5.5, "
            "gpt-6-sol, all at medium reasoning effort). The 'prod' lineup also selects "
            "the opus-5.5 prod stacker under --plain-llm."
        ),
    )
    parser.add_argument(
        "--plain-llm",
        action="store_true",
        default=False,
        help=(
            "Construct stacker LLMs via plain GeneralLlm (no donated-key wrapper, no "
            "fallback wrapping). Intended for paid benchmark-mode runs where fail-fast "
            "is preferred over cost absorption."
        ),
    )
    parser.add_argument(
        "--no-stacker-fallback",
        action="store_true",
        default=False,
        help=(
            "Disable the stacker fallback chain: on primary stacker failure, propagate "
            "the error immediately instead of trying the fallback LLM or median fallback. "
            "Intended for paid benchmark-mode runs where we want to see failures rather "
            "than silently degrade."
        ),
    )
    return parser


@dataclass(frozen=True)
class _StagePlan:
    """Which stages this invocation runs, which are forced, and the inter-stage pause."""

    requested: set[str]
    forced: set[str]
    sleep_seconds: float

    def wants(self, stage: str) -> bool:
        return stage in self.requested

    def is_forced(self, stage: str) -> bool:
        return stage in self.forced


@dataclass(frozen=True)
class _ArmStage:
    """One aggregation arm's stage: how to announce it and where its payloads land."""

    name: str  # --stages token and log label
    arm: str  # ARM_* key handed to _stage_stack
    report_arm: str  # arm whose payload count the DONE line reports
    # Deterministic arms do zero API work: fixed ~1 min estimate, no inter-stage sleep.
    # None marks the LLM-backed arms, which get a question-count estimate and a pause.
    deterministic_note: str | None


_ARM_STAGES: tuple[_ArmStage, ...] = (
    _ArmStage("stack", ARM_STACK, ARM_STACK, None),
    _ArmStage("stack_aug", ARM_STACK_AUG, ARM_STACK_AUG, None),
    _ArmStage("pdf", ARM_PDF, ARM_PDF_MIN2, "deterministic structured-math"),
    _ArmStage("median", ARM_MEDIAN, ARM_MEDIAN, "deterministic aggregation"),
    _ArmStage("mean", ARM_MEAN, ARM_MEAN, "deterministic aggregation"),
)
