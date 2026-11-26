"""The call rows and what is computed from them: arm means, paired deltas, the Markdown summary.

Everything here is a pure function of the rows and the per-question summaries ``run.json`` keeps, which
is what lets ``--rescore`` rebuild a run's results without the archive, the questions or the network.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from metaculus_bot.ablation.scoring import bootstrap_mean_ci, sign_test
from metaculus_bot.scoring_common import baseline_to_peer_factor
from scripts.probes.bench_common.records import (
    STATUS_API_ERROR as STATUS_API_ERROR,
)
from scripts.probes.bench_common.records import (
    STATUS_BUILD_FAILED as STATUS_BUILD_FAILED,
)
from scripts.probes.bench_common.records import (
    STATUS_EXTRACTION_FAILED as STATUS_EXTRACTION_FAILED,
)
from scripts.probes.bench_common.records import (
    STATUS_SCORED,
)
from scripts.probes.bench_common.records import (
    STATUS_SKIPPED_SPEND_CAP as STATUS_SKIPPED_SPEND_CAP,
)
from scripts.probes.bench_common.records import (
    STATUS_UNIT_MISMATCH as STATUS_UNIT_MISMATCH,
)
from scripts.probes.section_strip_bench.bundle import FULL_ARM

BOOTSTRAP_DRAWS = 5000

PEER_SCALE = "all_peer_points"
PUBLISHED_COLUMN = "published"


@dataclass
class CallRow:
    """One (question, arm, replicate) call as written to ``calls.jsonl``."""

    question_id: int
    qtype: str
    n_options: int | None
    arm: str
    seed: int
    status: str
    score: float | None = None
    forecast: Any = None
    rung: str | None = None
    block_present: bool | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None
    elapsed_s: float = 0.0
    error: str | None = None
    rationale: str | None = None


def _scores_by_question_arm(rows: Sequence[CallRow]) -> dict[tuple[int, str], list[float]]:
    scores: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        if row.status == STATUS_SCORED and row.score is not None:
            scores.setdefault((row.question_id, row.arm), []).append(row.score)
    return scores


def _arm_summary(
    means: dict[tuple[int, str], float], summaries: Sequence[dict[str, Any]], arms: Sequence[str]
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Per arm (and the published reference), the mean and median question score by type.

    Never pooled across types: the three native scores sit on different scales (log base 2, log base K,
    halved natural log), so the only cross-type figure is the peer-point delta in ``_paired_deltas``.
    """
    types = sorted({s["qtype"] for s in summaries})
    type_of = {s["question_id"]: s["qtype"] for s in summaries}
    columns: dict[str, dict[int, float]] = {
        arm: {qid: value for (qid, row_arm), value in means.items() if row_arm == arm} for arm in arms
    }
    columns[PUBLISHED_COLUMN] = {
        s["question_id"]: s["published_score"] for s in summaries if s["published_score"] is not None
    }
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for label, by_qid in columns.items():
        out[label] = {}
        for qtype in types:
            values = [v for qid, v in by_qid.items() if type_of[qid] == qtype]
            if values:
                out[label][qtype] = {
                    "n": len(values),
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                }
    return out


def _replicate_spread(
    scores: dict[tuple[int, str], list[float]], arms: Sequence[str]
) -> dict[str, dict[str, float | int]]:
    """Per arm, the mean within-question standard deviation over questions with two or more scored replicates.

    Zero here means the replicates collapsed to identical forecasts, so the replicate axis bought nothing.
    """
    out: dict[str, dict[str, float | int]] = {}
    for arm in arms:
        spreads = [statistics.stdev(v) for (_, row_arm), v in scores.items() if row_arm == arm and len(v) >= 2]
        if spreads:
            out[arm] = {"n_questions": len(spreads), "mean_std": statistics.fmean(spreads)}
    return out


def _delta_stats(deltas: Sequence[float], *, seed: int) -> dict[str, float | int]:
    mean, low, high = bootstrap_mean_ci(list(deltas), n_bootstrap=BOOTSTRAP_DRAWS, seed=seed)
    return {
        "n": len(deltas),
        "mean_delta": mean,
        "ci95_low": low,
        "ci95_high": high,
        "median_delta": statistics.median(deltas),
        "n_full_better": sum(1 for d in deltas if d > 0),
        "n_arm_better": sum(1 for d in deltas if d < 0),
        "sign_test_p": sign_test(list(deltas)),
    }


def _paired_deltas(
    means: dict[tuple[int, str], float], summaries: Sequence[dict[str, Any]], arms: Sequence[str], *, seed: int
) -> dict[str, dict[str, dict[str, float | int]]]:
    """full-minus-arm per question, by type on the native score scale and pooled on spot-peer points."""
    types = sorted({s["qtype"] for s in summaries})
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for index, arm in enumerate(a for a in arms if a != FULL_ARM):
        native: dict[str, list[float]] = {qtype: [] for qtype in types}
        peer: list[float] = []
        for summary in summaries:
            qid = summary["question_id"]
            if (qid, FULL_ARM) not in means or (qid, arm) not in means:
                continue
            delta = means[(qid, FULL_ARM)] - means[(qid, arm)]
            native[summary["qtype"]].append(delta)
            peer.append(delta * baseline_to_peer_factor(summary["qtype"], n_options=summary["n_options"]))
        groups = {**native, PEER_SCALE: peer}
        out[arm] = {
            label: _delta_stats(deltas, seed=seed + 1000 * index + offset)
            for offset, (label, deltas) in enumerate(groups.items())
            if deltas
        }
    return out


def _per_question_table(
    means: dict[tuple[int, str], float], summaries: Sequence[dict[str, Any]], arms: Sequence[str]
) -> list[dict[str, Any]]:
    table = []
    for summary in summaries:
        qid = summary["question_id"]
        full = means.get((qid, FULL_ARM))
        deltas = {
            arm: (full - means[(qid, arm)]) if full is not None and (qid, arm) in means else None
            for arm in arms
            if arm != FULL_ARM
        }
        table.append({**summary, "arm_scores": {arm: means.get((qid, arm)) for arm in arms}, "deltas": deltas})
    return table


def _spend_summary(rows: Sequence[CallRow]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, float]] = {}
    for row in rows:
        arm = by_arm.setdefault(
            row.arm, {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
        )
        arm["usd"] += row.cost_usd or 0.0
        arm["prompt_tokens"] += row.prompt_tokens
        arm["completion_tokens"] += row.completion_tokens
        arm["reasoning_tokens"] += row.reasoning_tokens
        arm["cached_tokens"] += row.cached_tokens
    return {
        "forecaster_usd": sum(a["usd"] for a in by_arm.values()),
        "costed_calls": sum(1 for row in rows if row.cost_usd is not None),
        "by_arm": by_arm,
    }


def aggregate(
    rows: Sequence[CallRow], summaries: Sequence[dict[str, Any]], *, arms: Sequence[str], bootstrap_seed: int
) -> dict[str, Any]:
    """Everything ``results.json`` carries, computed from the call rows and the per-question summaries alone."""
    scores = _scores_by_question_arm(rows)
    means = {key: statistics.fmean(values) for key, values in scores.items()}
    return {
        "arm_summary": _arm_summary(means, summaries, arms),
        "replicate_spread": _replicate_spread(scores, arms),
        "paired_deltas": _paired_deltas(means, summaries, arms, seed=bootstrap_seed),
        "per_question": _per_question_table(means, summaries, arms),
        "call_status_by_arm": {arm: dict(Counter(row.status for row in rows if row.arm == arm)) for arm in arms},
        "extraction_rungs": dict(Counter(f"{row.arm}:{row.rung}" for row in rows if row.rung is not None)),
        "spend": _spend_summary(rows),
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": bootstrap_seed},
    }


def render_markdown(results: dict[str, Any], *, model: str, seeds: int) -> str:
    """The human summary: the paired deltas first, then the arm means, the call outcomes and the spend."""
    lines = [
        "# Section-strip bench",
        "",
        f"Model `{model}`, {seeds} replicate(s) per arm; scores are the platform log scores (binary and MC on their",
        "log-K baseline scale, numeric already halved). A positive delta means the FULL bundle scored higher than the arm.",
        f"The `{PUBLISHED_COLUMN}` row is the production ensemble's own published forecast, made with the question",
        "background the arms do not get; it is a reference level for the cheap model, not a fifth arm.",
        "",
        "## Paired deltas, full minus arm",
        "",
        "| arm | group | n | mean Δ | 95% CI | median Δ | full better / arm better | sign test p |",
        "|---|---|--:|--:|---|--:|--:|--:|",
    ]
    for arm, groups in results["paired_deltas"].items():
        for label, stats in groups.items():
            lines.append(
                f"| {arm} | {label} | {stats['n']} | {stats['mean_delta']:+.2f} | "
                f"[{stats['ci95_low']:+.2f}, {stats['ci95_high']:+.2f}] | {stats['median_delta']:+.2f} | "
                f"{stats['n_full_better']} / {stats['n_arm_better']} | {stats['sign_test_p']:.3f} |"
            )
    lines += ["", "## Mean question score by arm", "", "| arm | type | n | mean | median |", "|---|---|--:|--:|--:|"]
    for arm, by_type in results["arm_summary"].items():
        for qtype, stats in by_type.items():
            lines.append(f"| {arm} | {qtype} | {stats['n']} | {stats['mean']:.2f} | {stats['median']:.2f} |")
    lines += ["", "## Replicate spread (mean within-question std; zero means the replicates collapsed)", ""]
    lines += [
        f"- {arm}: {spread['mean_std']:.2f} over {spread['n_questions']} questions"
        for arm, spread in results["replicate_spread"].items()
    ] or ["- fewer than two scored replicates per question, so no spread to report"]
    lines += ["", "## Call outcomes by arm", ""]
    lines += [f"- {arm}: {statuses}" for arm, statuses in results["call_status_by_arm"].items()]
    spend = results["spend"]
    lines += ["", "## Spend", ""]
    lines.append(f"- forecaster charges: ${spend['forecaster_usd']:.4f} over {spend['costed_calls']} costed calls")
    if "parser_usd" in spend:
        lines.append(f"- parser salvage charges: ${spend['parser_usd']:.4f}")
    lines += ["", "Per-question scores and deltas: `results.json` (`per_question`); every call: `calls.jsonl`.", ""]
    return "\n".join(lines)
