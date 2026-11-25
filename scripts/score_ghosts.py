"""Score gap-fill v2 ghost forecasts against resolutions (paired vs published).

The agentic gap-fill loop (v2) logs an unpublished "ghost" forecast per question for
telemetry. Comparing those ghosts to the actually-published forecast on resolved
questions is the NAMED GATE for retiring v1 gap-fill: if the v2-driven ghost
consistently out-scores the published forecast, v2 is carrying its weight.

Marker sources: ``GHOST_FORECAST_JSON`` (full fidelity, which makes numeric ghosts scoreable)
is preferred over the legacy lossy ``GHOST_FORECAST`` summary for the same qid. ``GHOST_PRE_JSON``
(the turn-one dry run) feeds the pre-identity split: a concluding ghost byte-identical to its
dry run measures the driver's prior, not the loop's research, so the retirement gate reads the
loop-moved subset, never the pool. Two same-driver paired reads then score two ghost variants of
the same question in the same run against the resolution: pre versus post (delta = post minus
pre, what v2's own research did to the driver) and plain versus with-v1
(``GHOST_FORECAST_V1_JSON``, the ghost re-asked with gap-fill v1's section since 2026-09-09;
delta = v1's marginal value on the driver, the mirror read). A date ghost is counted by type but
unscoreable while the residual dataset excludes date questions.

All markers are harvested into ``backtests/telemetry_archive/`` by ``make sync_telemetry``;
resolved-question records come from a perf dataset JSON (``--perf-json``) or a live read-only
pull (``--tournament``, free). The n=0 path must run cleanly and report "waiting on resolutions".
Discrete-grid handling and the numeric approximation: docs/agentic_gap_fill.md "Scoring the ghosts".
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

from metaculus_bot.api_preflight import verify_metaculus_api_identity
from metaculus_bot.performance_analysis.collector import (
    build_performance_dataset,
    load_dataset,
    resolve_numeric_record_to_score_inputs,
)
from metaculus_bot.performance_analysis.ghosts import (
    ghost_binary_prob,
    ghost_cdf_on_published_grid,
    ghost_mc_probs,
    ghost_numeric_percentiles,
    latest_ghost_per_qid,
    normalize_json_ghost,
    records_by_post_id,
    select_ghost_per_qid,
)
from metaculus_bot.scoring_common import binary_log_score, mc_log_score, numeric_log_score
from scripts.telemetry.archive import load_marker_records

logger = logging.getLogger(__name__)


def _score_binary(ghost: dict, record: dict) -> dict | None:
    """Paired binary log-score row (ghost vs published), or None if not scoreable."""
    resolution = record.get("resolution_parsed")
    if not isinstance(resolution, bool):
        return None
    ghost_p = ghost_binary_prob(ghost)
    published_p = record.get("our_prob_yes")
    if ghost_p is None or published_p is None:
        return None
    ghost_ls = binary_log_score(ghost_p, resolution)
    published_ls = binary_log_score(float(published_p), resolution)
    return {
        "qid": ghost["qid"],
        "resolution": resolution,
        "ghost_prob": ghost_p,
        "published_prob": float(published_p),
        "ghost_log_score": ghost_ls,
        "published_log_score": published_ls,
        "delta": ghost_ls - published_ls,
    }


def _score_mc(ghost: dict, record: dict) -> dict | None:
    """Paired MC log-score row (ghost vs published), or None if not scoreable."""
    resolution = record.get("resolution_parsed")
    options = record.get("options") or []
    published_values = record.get("our_forecast_values")
    ghost_probs = ghost_mc_probs(ghost)
    if not isinstance(resolution, str) or resolution not in options:
        return None
    if not isinstance(ghost_probs, dict) or not all(opt in ghost_probs for opt in options):
        return None
    if not published_values or len(published_values) != len(options):
        return None
    correct_idx = options.index(resolution)
    ghost_ordered = [ghost_probs[opt] for opt in options]
    ghost_ls = mc_log_score(ghost_ordered, correct_idx)
    published_ls = mc_log_score(list(published_values), correct_idx)
    return {
        "qid": ghost["qid"],
        "resolution": resolution,
        "ghost_log_score": ghost_ls,
        "published_log_score": published_ls,
        "delta": ghost_ls - published_ls,
    }


def _paired_numeric_scores(
    ghost_cdf: list[float],
    published_cdf: list[float],
    score_inputs: tuple[float, float, float, float | None],
    *,
    open_lower: bool,
    open_upper: bool,
) -> dict:
    """Log-score the ghost and published CDFs against shared numeric inputs.

    Returns the scored fields WITHOUT ``qid`` (the caller stamps it), or a
    ``scoreable: False`` stub naming which of the two steps failed.
    """
    res_float, lower, upper, zero_point = score_inputs

    try:
        ghost_ls = numeric_log_score(
            ghost_cdf,
            res_float,
            lower,
            upper,
            open_lower_bound=open_lower,
            open_upper_bound=open_upper,
            zero_point=zero_point,
        )
        published_ls = numeric_log_score(
            published_cdf,
            res_float,
            lower,
            upper,
            open_lower_bound=open_lower,
            open_upper_bound=open_upper,
            zero_point=zero_point,
        )
    except (ValueError, ZeroDivisionError):
        return {"scoreable": False, "reason": "score_failed"}

    return {
        "scoreable": True,
        "reason": None,
        "resolution": res_float,
        "ghost_log_score": ghost_ls,
        "published_log_score": published_ls,
        "delta": ghost_ls - published_ls,
    }


def _score_numeric(ghost: dict, record: dict) -> dict:
    """Attempt a paired numeric log-score (ghost vs published).

    Builds the ghost's CDF from its declared percentiles with the production PCHIP builder on
    the SAME grid length as the published CDF, then scores both with the same Metaculus
    PMF-bucket log score on identical bounds and scaling, so the delta is a clean pair. Always
    returns a dict with ``scoreable``; when False, ``reason`` names the gap. A native-discrete
    grid is mirrored exactly; a continuous question prod integer-snapped is scored as the smooth
    distribution the ghost declared, a documented approximation mildly biased against the ghost
    on that minority (docs/agentic_gap_fill.md "Scoring the ghosts").
    """
    qid = ghost["qid"]
    percentiles = ghost_numeric_percentiles(ghost)
    if percentiles is None:
        reason = "legacy_median_only" if ghost["source"] == "legacy" else "no_declared_percentiles"
        return {"qid": qid, "scoreable": False, "reason": reason}

    published_cdf = record.get("our_forecast_values")
    if not published_cdf or len(published_cdf) < 2:
        return {"qid": qid, "scoreable": False, "reason": "no_published_cdf"}

    score_inputs = resolve_numeric_record_to_score_inputs(record)
    if score_inputs is None:
        return {"qid": qid, "scoreable": False, "reason": "no_score_inputs"}

    ghost_cdf = ghost_cdf_on_published_grid(percentiles, record)
    if ghost_cdf is None:
        return {"qid": qid, "scoreable": False, "reason": "cdf_build_failed"}

    outcome = _paired_numeric_scores(
        ghost_cdf,
        list(published_cdf),
        score_inputs,
        open_lower=bool(record.get("open_lower_bound", False)),
        open_upper=bool(record.get("open_upper_bound", False)),
    )
    return {"qid": qid, **outcome}


def _summarize_rows(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "mean_delta": mean(r["delta"] for r in rows) if rows else None,
        "rows": rows,
    }


def _pre_identity(ghost: dict, pre_by_qid: dict[int, dict]) -> bool | None:
    """Whether the concluding ghost is byte-identical to the pre-research dry run.

    ``True``/``False`` only when both halves are JSON payloads FROM THE SAME RUN
    (the two markers serialize through the same ``_summarize_ghost`` path, so dict
    equality IS the byte-identity the split needs). ``None`` when there is nothing
    to compare: no ``GHOST_PRE_JSON`` for this qid, a legacy summary-only ghost
    (the pre marker and the JSON ghost landed in the same merge, so a legacy ghost
    predates both), or a pre-ghost from a DIFFERENT run — a run can emit one
    marker without the other (a schema-invalid dry run suppresses GHOST_PRE_JSON;
    a deadline-hit run can bank a pre with no concluding ghost), and comparing
    across runs would file a cross-run pair into ``pre_identical``/``loop_moved``.
    """
    if ghost["source"] != "json":
        return None
    pre = pre_by_qid.get(ghost["qid"])
    if pre is None or pre.get("run_id") != ghost.get("run_id"):
        return None
    return pre["payload"] == ghost["payload"]


def _split_bucket(pre_identical: bool | None) -> str:
    if pre_identical is None:
        return "no_pre_marker"
    return "pre_identical" if pre_identical else "loop_moved"


def _json_ghosts_by_qid(records: list[dict] | None) -> dict[int, dict]:
    """Latest JSON-marker ghost per qid, normalized; a malformed record is dropped.

    All the JSON ghost markers (pre, concluding, v1) share the post-id space, since they carry
    the same log_prefix ref, so the same map serves every pairing.
    """
    by_qid: dict[int, dict] = {}
    for qid, record in latest_ghost_per_qid(records or []).items():
        normalized = normalize_json_ghost(record)
        if normalized is not None:
            by_qid[qid] = normalized
    return by_qid


def _ghost_and_published_log_scores(ghost: dict, record: dict) -> tuple[float, float] | None:
    """``(ghost, published)`` log scores for one ghost against its resolved record, or None when unscoreable."""
    qtype = ghost.get("qtype")
    if qtype == "binary":
        row = _score_binary(ghost, record)
    elif qtype == "multiple_choice":
        row = _score_mc(ghost, record)
    elif qtype == "numeric":
        numeric = _score_numeric(ghost, record)
        row = numeric if numeric["scoreable"] else None
    else:
        row = None
    return None if row is None else (row["ghost_log_score"], row["published_log_score"])


def _sign_test_p(n_pos: int, n_neg: int) -> float | None:
    """Two-sided exact binomial sign test on the non-tied pairs; None when there are none."""
    n = n_pos + n_neg
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(n_pos, n_neg) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _delta_stats(rows: list[dict]) -> dict:
    deltas = [row["delta"] for row in rows]
    n_toward = sum(delta > 0 for delta in deltas)
    n_away = sum(delta < 0 for delta in deltas)
    return {
        "n": len(rows),
        "mean_delta": mean(deltas) if deltas else None,
        "median_delta": median(deltas) if deltas else None,
        "n_toward": n_toward,
        "n_away": n_away,
        "sign_test_p": _sign_test_p(n_toward, n_away),
    }


def paired_same_run_read(left_by_qid: dict[int, dict], right_by_qid: dict[int, dict], records_by_post_id: dict) -> dict:
    """Score two ghost variants of the same driver on the same question in the same run.

    ``delta`` is the right variant's log score minus the left's, so positive means the right
    variant sat closer to the truth. A pair whose payloads are byte-identical has a delta of
    exactly 0 by construction and is reported apart from the ``moved`` pairs, the only ones that
    measure what the right variant added. ``unpaired`` counts why a right-hand ghost found no
    partner; a partner from another run is not a partner (a qid can be forecast in several runs).
    """
    rows: list[dict] = []
    unpaired: dict[str, int] = {}

    def _drop(reason: str) -> None:
        unpaired[reason] = unpaired.get(reason, 0) + 1

    for qid, right in right_by_qid.items():
        record = records_by_post_id.get(qid)
        left = left_by_qid.get(qid)
        if record is None:
            _drop("no resolved record")
            continue
        if left is None:
            _drop("no partner marker")
            continue
        if left.get("run_id") != right.get("run_id"):
            _drop("partner from a different run")
            continue
        if left.get("qtype") != right.get("qtype"):
            _drop("qtype mismatch")
            continue
        left_scores = _ghost_and_published_log_scores(left, record)
        right_scores = _ghost_and_published_log_scores(right, record)
        if left_scores is None or right_scores is None:
            _drop("one side unscoreable")
            continue
        rows.append(
            {
                "qid": qid,
                "qtype": right["qtype"],
                "identical": left["payload"] == right["payload"],
                "left_log_score": left_scores[0],
                "right_log_score": right_scores[0],
                "published_log_score": right_scores[1],
                "delta": right_scores[0] - left_scores[0],
            }
        )
    moved = [row for row in rows if not row["identical"]]
    return {
        "n_paired": len(rows),
        "n_identical": len(rows) - len(moved),
        "unpaired": unpaired,
        "moved": {
            **_delta_stats(moved),
            "by_type": {
                qtype: _delta_stats([row for row in moved if row["qtype"] == qtype])
                for qtype in sorted({row["qtype"] for row in moved})
            },
            "left_mean": mean(row["left_log_score"] for row in moved) if moved else None,
            "right_mean": mean(row["right_log_score"] for row in moved) if moved else None,
            "published_mean": mean(row["published_log_score"] for row in moved) if moved else None,
        },
        "rows": rows,
    }


def _count_sources(selected: dict[int, dict]) -> dict[str, int]:
    """How many of the ghosts to score came from each marker source."""
    source_counts: dict[str, int] = {"json": 0, "legacy": 0}
    for ghost in selected.values():
        source_counts[ghost["source"]] = source_counts.get(ghost["source"], 0) + 1
    return source_counts


def _qtype_label(ghost: dict) -> str:
    """The ghost's question type as a report key; a block that never parsed reads ``unknown``."""
    return str(ghost.get("qtype") or "unknown")


def _count_qtypes(selected: dict[int, dict]) -> dict[str, int]:
    """How many of the ghosts to score carry each question type, before the join.

    The inventory is what makes a date ghost visible at all: the residual dataset excludes
    date questions by decision (backlog task-26 "Mantic, awaiting live data"), so a date ghost never finds a
    record to join and would otherwise be indistinguishable from a question still waiting
    on its resolution.
    """
    counts: dict[str, int] = {}
    for ghost in selected.values():
        label = _qtype_label(ghost)
        counts[label] = counts.get(label, 0) + 1
    return counts


@dataclass
class _GhostScoreTally:
    """Per-type row buckets plus the pre-identity split, filled one ghost/record pair at a time.

    Every scored row is appended to BOTH its question-type bucket and its pre-identity
    bucket — the same dict object in each, so ``split_by_pre_identity`` partitions exactly
    the rows the per-type blocks report.
    """

    pre_by_qid: dict[int, dict]
    binary_rows: list[dict] = field(default_factory=list)
    mc_rows: list[dict] = field(default_factory=list)
    numeric_rows: list[dict] = field(default_factory=list)
    numeric_joined: int = 0
    numeric_unscoreable: dict[str, int] = field(default_factory=dict)
    joined_without_scorer: dict[str, int] = field(default_factory=dict)
    split_rows: dict[str, list[dict]] = field(
        default_factory=lambda: {"pre_identical": [], "loop_moved": [], "no_pre_marker": []}
    )

    def add(self, ghost: dict, record: dict) -> None:
        """Score one joined ghost/record pair into the buckets its question type feeds."""
        qtype = ghost.get("qtype")
        if qtype == "binary":
            self._add_row(self.binary_rows, ghost, _score_binary(ghost, record))
        elif qtype == "multiple_choice":
            self._add_row(self.mc_rows, ghost, _score_mc(ghost, record))
        elif qtype == "numeric":
            self._add_numeric(ghost, record)
        else:
            # No scorer for this type (a date ghost, or an unparsed block): counted so it never reads as unresolved.
            label = _qtype_label(ghost)
            self.joined_without_scorer[label] = self.joined_without_scorer.get(label, 0) + 1

    def _add_row(self, bucket: list[dict], ghost: dict, row: dict | None) -> None:
        if row is None:
            return
        bucket.append(row)
        row["pre_identical"] = _pre_identity(ghost, self.pre_by_qid)
        self.split_rows[_split_bucket(row["pre_identical"])].append(row)

    def _add_numeric(self, ghost: dict, record: dict) -> None:
        self.numeric_joined += 1
        outcome = _score_numeric(ghost, record)
        if outcome["scoreable"]:
            self._add_row(self.numeric_rows, ghost, outcome)
        else:
            reason = outcome["reason"]
            self.numeric_unscoreable[reason] = self.numeric_unscoreable.get(reason, 0) + 1

    @property
    def n_scored(self) -> int:
        return len(self.binary_rows) + len(self.mc_rows) + len(self.numeric_rows)


def join_and_score(
    json_ghosts: list[dict],
    legacy_ghosts: list[dict],
    records: list[dict],
    *,
    pre_ghosts: list[dict] | None = None,
    v1_ghosts: list[dict] | None = None,
) -> dict:
    """Join the latest ghost per qid to resolved records and compute paired log-score deltas.

    JSON-source ghosts are preferred over legacy summary-only ghosts for the same qid; a joined
    ghost of a type without a scorer is counted under ``joined_without_scorer``. ``pre_ghosts``
    and ``v1_ghosts`` feed the two same-driver paired reads (``pre_post``, ``v1_pairs``) beside
    the pre-identity split. Pure and in-memory, so the n=0 path is exercised in tests.

    The join key is ``post_id``, NOT ``question_id``: a ghost's qid comes from the seam's
    ``page_url`` ref (a Metaculus POST id), while the collector's ``question_id`` is the disjoint
    sub-question id, so keying on it would make every join silently miss. Group posts collapse
    their sub-questions to one record per post on both sides, a limitation of the post-level ref.
    """
    post_id_records = records_by_post_id(records)
    selected = select_ghost_per_qid(json_ghosts, legacy_ghosts)
    pre_by_qid = _json_ghosts_by_qid(pre_ghosts)
    post_by_qid = _json_ghosts_by_qid(json_ghosts)
    tally = _GhostScoreTally(pre_by_qid=pre_by_qid)

    n_joined = 0
    for qid, ghost in selected.items():
        record = post_id_records.get(qid)
        if record is None:
            continue
        n_joined += 1
        tally.add(ghost, record)

    return {
        # delta = post minus pre: what v2's own research did to the driver, the instrument for v2.
        "pre_post": paired_same_run_read(pre_by_qid, post_by_qid, post_id_records),
        # delta = with-v1 minus plain: v1's marginal value on the same driver, the mirror read for v1.
        "v1_pairs": paired_same_run_read(post_by_qid, _json_ghosts_by_qid(v1_ghosts), post_id_records),
        "n_ghosts": len(selected),
        "n_joined": n_joined,
        "n_scored": tally.n_scored,
        "source_counts": _count_sources(selected),
        "qtype_counts": _count_qtypes(selected),
        "joined_without_scorer": tally.joined_without_scorer,
        "binary": _summarize_rows(tally.binary_rows),
        "multiple_choice": _summarize_rows(tally.mc_rows),
        "numeric": {
            **_summarize_rows(tally.numeric_rows),
            "n_joined": tally.numeric_joined,
            "n_unscoreable": sum(tally.numeric_unscoreable.values()),
            "unscoreable_reasons": tally.numeric_unscoreable,
        },
        # The identical bucket measures the driver's PRIOR; only loop_moved says anything about v2's research.
        "split_by_pre_identity": {
            bucket: {"n": len(rows), "mean_delta": mean(r["delta"] for r in rows) if rows else None}
            for bucket, rows in tally.split_rows.items()
        },
    }


def _counts_inline(counts: dict[str, int]) -> str:
    return " ".join(f"{key}={count}" for key, count in sorted(counts.items()))


def _inventory_lines(summary: dict) -> list[str]:
    """The ghost inventory and the join, before any scoring: counts by source and by type."""
    source_counts = summary["source_counts"]
    qtype_counts = summary["qtype_counts"]
    lines = [
        f"Ghosts (latest per qid): {summary['n_ghosts']}",
        f"  by source: json={source_counts.get('json', 0)} legacy={source_counts.get('legacy', 0)}",
    ]
    if qtype_counts:
        lines.append("  by type: " + _counts_inline(qtype_counts))
    lines.append(f"Joined to resolved-question dataset: {summary['n_joined']}")
    if summary["joined_without_scorer"]:
        lines.append("  joined but no scorer for the type: " + _counts_inline(summary["joined_without_scorer"]))
    if qtype_counts.get("date"):
        lines.append(
            "  date ghosts cannot be scored yet: the residual dataset excludes date questions by decision "
            '(backlog task-26 "Mantic, awaiting live data")'
        )
    return lines


def _pre_identity_lines(summary: dict) -> list[str]:
    """The pooled ghost-versus-dry-run split; empty until something has been scored."""
    split = summary.get("split_by_pre_identity")
    if not split or not summary["n_scored"]:
        return []
    bucket_labels = {
        "pre_identical": "byte-identical to the pre-research dry run (measures the driver's prior)",
        "loop_moved": "loop moved the driver's forecast (the only bucket that measures v2's research)",
        "no_pre_marker": "no GHOST_PRE_JSON to compare (predates the marker, or legacy ghost)",
    }
    lines = ["Ghost vs pre-research dry run (pooled across types — do not quote the pooled delta):"]
    for bucket, label in bucket_labels.items():
        block = split[bucket]
        if block["n"]:
            lines.append(f"  {label}: n={block['n']} mean_delta={block['mean_delta']:+.4f}")
    return lines


def _paired_read_lines(block: dict, *, title: str, left: str, right: str, waiting: str) -> list[str]:
    """One same-driver paired read: the pair counts, the moved pairs' delta statistics, and the ladder."""
    lines = [title]
    unpaired = f"; unpaired: {_counts_inline(block['unpaired'])}" if block["unpaired"] else ""
    if block["n_paired"] == 0:
        lines.append(f"  paired: 0 ({waiting}){unpaired}")
        return lines
    moved = block["moved"]
    lines.append(
        f"  paired: {block['n_paired']}  byte-identical: {block['n_identical']}  moved: {moved['n']}{unpaired}"
    )
    if not moved["n"]:
        return lines
    sign_p = "n/a" if moved["sign_test_p"] is None else f"{moved['sign_test_p']:.3f}"
    lines.append(
        f"  moved: mean {moved['mean_delta']:+.4f} median {moved['median_delta']:+.4f} "
        f"toward {moved['n_toward']} / away {moved['n_away']}, sign test p={sign_p}"
    )
    for qtype, stats in moved["by_type"].items():
        lines.append(
            f"    {qtype}: n={stats['n']} mean {stats['mean_delta']:+.4f} toward {stats['n_toward']} / away {stats['n_away']}"
        )
    lines.append(
        f"  ladder on the moved pairs: {left} {moved['left_mean']:+.4f}, {right} {moved['right_mean']:+.4f}, "
        f"published {moved['published_mean']:+.4f}"
    )
    return lines


def _same_driver_read_lines(summary: dict) -> list[str]:
    return [
        *_paired_read_lines(
            summary["pre_post"],
            title=(
                "Pre-research dry run vs concluding ghost (same driver, same run; delta = post minus pre, "
                "positive = v2's research moved the driver toward the truth):"
            ),
            left="pre",
            right="post",
            waiting="no GHOST_PRE_JSON / GHOST_FORECAST_JSON pair on a resolved question yet",
        ),
        *_paired_read_lines(
            summary["v1_pairs"],
            title=(
                "Ghost with gap-fill v1 vs plain ghost (same driver, same run; delta = with-v1 minus plain, "
                "positive = v1's section moved the driver toward the truth):"
            ),
            left="plain",
            right="with v1",
            waiting="GHOST_FORECAST_V1 ships 2026-09-09; waiting on resolutions",
        ),
    ]


def render_report(summary: dict) -> str:
    """Human-readable summary. A positive mean delta = ghost out-scores published."""
    lines = ["=== Ghost-forecast scoring (gap-fill v2 ghost vs published) ===", *_inventory_lines(summary)]
    if summary["n_scored"] == 0:
        lines.append("Scored ghosts: n=0 — waiting on resolutions (v2 live in prod 2026-07-21; expected today).")
    else:
        lines.append(f"Scored ghosts: {summary['n_scored']}")
    for qtype in ("binary", "multiple_choice", "numeric"):
        block = summary[qtype]
        if block["n"]:
            delta = block["mean_delta"]
            verdict = "ghost better" if delta > 0 else "published better"
            lines.append(f"  {qtype}: n={block['n']} mean_delta={delta:+.4f} ({verdict})")
    numeric = summary["numeric"]
    if numeric["n_joined"]:
        lines.append(
            f"  numeric coverage: {numeric['n_joined']} joined, "
            f"{numeric['n']} scored, {numeric['n_unscoreable']} unscoreable"
        )
        for reason, count in sorted(numeric["unscoreable_reasons"].items()):
            lines.append(f"    - {reason}: {count}")
    lines.extend(_pre_identity_lines(summary))
    lines.extend(_same_driver_read_lines(summary))
    return "\n".join(lines)


def _load_records(perf_json: str | None, tournament: str | None) -> list[dict]:
    """Load resolved-question records from a perf JSON, else a live read-only pull, else []."""
    if perf_json:
        records = load_dataset(perf_json)
        logger.info(f"Loaded {len(records)} records from {perf_json}")
        return records
    if tournament:
        # Confirm the host is the real Metaculus before the token-sending pull.
        verify_metaculus_api_identity()
        logger.info(f"Building performance dataset from tournament '{tournament}' (read-only Metaculus pull)...")
        return build_performance_dataset(tournament=tournament)
    logger.warning("No --perf-json or --tournament given; reporting ghost inventory only (nothing to score against).")
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Score gap-fill v2 ghost forecasts vs published on resolutions.")
    parser.add_argument("--archive-dir", default="backtests/telemetry_archive", help="Telemetry archive dir")
    parser.add_argument("--perf-json", default=None, help="Pre-built performance-analysis dataset JSON")
    parser.add_argument("--tournament", default=None, help="Live read-only pull for this tournament slug (free)")
    parser.add_argument("--output", default=None, help="Optional path to dump the summary JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    json_ghosts = load_marker_records(Path(args.archive_dir), "ghost_forecast_json")
    legacy_ghosts = load_marker_records(Path(args.archive_dir), "ghost_forecast")
    pre_ghosts = load_marker_records(Path(args.archive_dir), "ghost_pre_json")
    v1_ghosts = load_marker_records(Path(args.archive_dir), "ghost_forecast_v1_json")
    logger.info(
        f"Loaded {len(json_ghosts)} ghost_forecast_json + {len(legacy_ghosts)} legacy ghost_forecast "
        f"+ {len(pre_ghosts)} ghost_pre_json + {len(v1_ghosts)} ghost_forecast_v1_json record(s) from {args.archive_dir}"
    )

    records = _load_records(args.perf_json, args.tournament)
    summary = join_and_score(json_ghosts, legacy_ghosts, records, pre_ghosts=pre_ghosts, v1_ghosts=v1_ghosts)
    print(render_report(summary))

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
        logger.info(f"Summary written to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
