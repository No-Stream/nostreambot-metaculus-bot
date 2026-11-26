"""Outcome-free forecast movement summaries for the cross-pollination bench."""

from __future__ import annotations

import itertools
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from forecasting_tools import BinaryQuestion

from metaculus_bot.scoring_common import baseline_to_peer_factor
from scripts.probes.cross_pollination_bench.aggregate import (
    FIRST_MEDIAN,
    AggregateScore,
    panel,
    score_forecast,
    variant_median,
)
from scripts.probes.cross_pollination_bench.manifest import Manifest, bench_resolution, rebuild_question
from scripts.probes.cross_pollination_bench.rows import STAGE_FIRST, UPDATE_VARIANTS, BenchCallRow


def forecast_distance(qtype: str, a: float | Sequence[float], b: float | Sequence[float]) -> float:
    if qtype == "binary":
        if isinstance(a, Sequence) or isinstance(b, Sequence):
            raise TypeError("binary forecasts must be scalar")
        return abs(float(a) - float(b)) * 100.0
    if not isinstance(a, Sequence) or not isinstance(b, Sequence):
        raise TypeError(f"{qtype} forecasts must be vectors")
    if qtype == "multiple_choice":
        if len(a) != len(b) or not a:
            raise ValueError("multiple-choice forecast vectors must have the same nonzero length")
        differences = [abs(float(left) - float(right)) for left, right in zip(a, b, strict=True)]
        return sum(differences) * 50.0
    if qtype in {"numeric", "discrete"}:
        if len(a) < 2 or len(b) < 2:
            raise ValueError(f"{qtype} CDFs must contain at least two points")

        def interpolate(values: Sequence[float], index: int, count: int) -> float:
            position = index * (len(values) - 1) / (count - 1)
            lower = math.floor(position)
            upper = min(math.ceil(position), len(values) - 1)
            fraction = position - lower
            return float(values[lower]) * (1 - fraction) + float(values[upper]) * fraction

        point_count = max(len(a), len(b))
        differences = [
            abs(interpolate(a, index, point_count) - interpolate(b, index, point_count)) for index in range(point_count)
        ]
        return statistics.fmean(differences) * 100.0
    raise ValueError(f"unknown question type {qtype!r}")


def _scores_by_question_draw(scores: Sequence[AggregateScore]) -> dict[tuple[int, int], dict[str, AggregateScore]]:
    grouped: dict[tuple[int, int], dict[str, AggregateScore]] = defaultdict(dict)
    for score in scores:
        grouped[(score.question_id, score.draw)][score.aggregate] = score
    return grouped


def _change_summary(changes: list[float]) -> dict[str, Any]:
    return {
        "typical_change": statistics.median(changes) if changes else None,
        "mean_change": statistics.fmean(changes) if changes else None,
        "share_changed": sum(change > 0 for change in changes) / len(changes) if changes else None,
        "n_question_draws": len(changes),
    }


def first_draw_noise(scores: Sequence[AggregateScore]) -> dict[str, Any]:
    by_question: dict[int, dict[int, AggregateScore]] = defaultdict(dict)
    for score in scores:
        if score.aggregate == FIRST_MEDIAN:
            by_question[score.question_id][score.draw] = score
    distances = [
        (draws[0].qtype, forecast_distance(draws[0].qtype, draws[0].forecast, draws[1].forecast))
        for draws in by_question.values()
        if 0 in draws and 1 in draws
    ]

    def summarize(items: list[tuple[str, float]]) -> dict[str, Any]:
        values = [value for _, value in items]
        return {
            "median": statistics.median(values) if values else None,
            "mean": statistics.fmean(values) if values else None,
            "n_questions": len(values),
        }

    return {
        **summarize(distances),
        "by_type": {
            qtype: summarize([item for item in distances if item[0] == qtype])
            for qtype in sorted({q for q, _ in distances})
        },
    }


def median_movement(scores: Sequence[AggregateScore]) -> dict[str, Any]:
    grouped = _scores_by_question_draw(scores)
    result: dict[str, Any] = {}
    for variant in UPDATE_VARIANTS:
        aggregate = variant_median(variant)
        changes: list[tuple[str, float]] = []
        for aggregates in grouped.values():
            if FIRST_MEDIAN in aggregates and aggregate in aggregates:
                first = aggregates[FIRST_MEDIAN]
                changes.append(
                    (first.qtype, forecast_distance(first.qtype, first.forecast, aggregates[aggregate].forecast))
                )
        result[variant] = {
            **_change_summary([change for _, change in changes]),
            "by_type": {
                qtype: _change_summary([change for kind, change in changes if kind == qtype])
                for qtype in sorted({kind for kind, _ in changes})
            },
        }
    result["first_draw_noise"] = first_draw_noise(scores)
    return result


def spread(qtype: str, forecasts: Sequence[float | Sequence[float]]) -> float:
    if len(forecasts) < 2:
        return 0.0
    return max(forecast_distance(qtype, left, right) for left, right in itertools.combinations(forecasts, 2))


def spread_change(rows: Sequence[BenchCallRow], manifest: Manifest) -> dict[str, Any]:
    entries = {entry.question_id: entry for entry in manifest.questions}
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for qid, draw in sorted({(row.question_id, row.draw) for row in rows}):
        qtype = entries[qid].qtype
        first = panel(rows, qid, draw, STAGE_FIRST)
        if first:
            values[qtype]["first"].append(spread(qtype, [member.forecast for member in first]))
        for variant in UPDATE_VARIANTS:
            members = panel(rows, qid, draw, variant)
            if members:
                values[qtype][variant].append(spread(qtype, [member.forecast for member in members]))
    return {
        qtype: {
            panel_name: {
                "mean_spread": statistics.fmean(spreads) if spreads else None,
                "n_question_draws": len(spreads),
            }
            for panel_name, spreads in panels.items()
        }
        for qtype, panels in values.items()
    }


def member_changes(rows: Sequence[BenchCallRow], manifest: Manifest) -> dict[str, Any]:
    entries = {entry.question_id: entry for entry in manifest.questions}
    by_key = {(row.question_id, row.draw, row.model, row.stage): row for row in rows}
    changes: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for (qid, draw, model, stage), update in by_key.items():
        if stage not in UPDATE_VARIANTS or update.status != "scored" or update.score is None or update.forecast is None:
            continue
        first = by_key.get((qid, draw, model, STAGE_FIRST))
        if first is None or first.status != "scored" or first.score is None or first.forecast is None:
            continue
        entry = entries[qid]
        factor = baseline_to_peer_factor(entry.qtype, n_options=entry.n_options)
        changes[stage][model].append(
            ((update.score - first.score) * factor, forecast_distance(entry.qtype, first.forecast, update.forecast))
        )
    result = {}
    for variant, models in changes.items():
        result[variant] = {}
        for model, pairs in models.items():
            score_deltas = [score_delta for score_delta, _ in pairs]
            distances = [distance for _, distance in pairs]
            n = len(pairs)
            toward = sum(delta > 0 for delta in score_deltas)
            away = sum(delta < 0 for delta in score_deltas)
            result[variant][model] = {
                "mean_own_score_change_peer_points": statistics.fmean(score_deltas),
                "mean_distance_moved": statistics.fmean(distances),
                "share_toward_outcome": toward / n,
                "share_away_from_outcome": away / n,
                "share_unchanged_score": 1 - (toward + away) / n,
                "n": n,
            }
    return result


def correction_counts(rows: Sequence[BenchCallRow]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.stage in UPDATE_VARIANTS:
            counts[row.stage][row.correction_kind or "unparsed"] += 1
    return {variant: dict(counts[variant]) for variant in UPDATE_VARIANTS}


def production_proxy(scores: Sequence[AggregateScore], manifest: Manifest) -> dict[str, Any]:
    entries = {entry.question_id: entry for entry in manifest.questions}
    grouped = _scores_by_question_draw(scores)
    distances: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    production_peer_scores: dict[str, list[float]] = defaultdict(list)
    for (qid, _), aggregates in grouped.items():
        if FIRST_MEDIAN not in aggregates:
            continue
        entry = entries[qid]
        question = rebuild_question(entry)
        production = entry.production_forecast
        if isinstance(question, BinaryQuestion):
            production_forecast: float | list[float] = float(production["prob_yes"])
        else:
            production_forecast = [float(value) for value in production["forecast_values"]]
        first = aggregates[FIRST_MEDIAN]
        distances[entry.qtype]["first_median"].append(
            forecast_distance(entry.qtype, first.forecast, production_forecast)
        )
        for variant in UPDATE_VARIANTS:
            aggregate = variant_median(variant)
            if aggregate in aggregates:
                distances[entry.qtype][variant].append(
                    forecast_distance(entry.qtype, aggregates[aggregate].forecast, production_forecast)
                )
        resolution = bench_resolution(entry, question)
        prod_score = score_forecast(question, resolution, production_forecast)
        production_peer_scores[entry.qtype].append(
            prod_score * baseline_to_peer_factor(entry.qtype, n_options=entry.n_options)
        )
    result = {}
    for qtype, by_aggregate in distances.items():
        result[qtype] = {
            f"{aggregate}_mean_distance": statistics.fmean(values) if values else None
            for aggregate, values in by_aggregate.items()
        }
        result[qtype]["production_mean_peer_points"] = (
            statistics.fmean(production_peer_scores[qtype]) if production_peer_scores[qtype] else None
        )
        first_scores = [
            score.peer_points for score in scores if score.qtype == qtype and score.aggregate == FIRST_MEDIAN
        ]
        result[qtype]["first_median_mean_peer_points"] = statistics.fmean(first_scores) if first_scores else None
    return result


def likelihood_ratio(differences_peer_points: Sequence[float]) -> float:
    return math.exp(sum(differences_peer_points) / 100.0)


def contrast_by_first_spread(
    paired: Sequence[dict[str, Any]], rows: Sequence[BenchCallRow], manifest: Manifest
) -> dict[str, Any]:
    entries = {entry.question_id: entry for entry in manifest.questions}
    by_question: dict[int, list[float]] = defaultdict(list)
    for qid, draw in {(row.question_id, row.draw) for row in rows}:
        first = panel(rows, qid, draw, STAGE_FIRST)
        if first:
            by_question[qid].append(spread(entries[qid].qtype, [member.forecast for member in first]))
    spreads = {qid: statistics.fmean(values) for qid, values in by_question.items()}
    type_medians = {
        qtype: statistics.median(values)
        for qtype in {entries[qid].qtype for qid in spreads}
        if (values := [value for qid, value in spreads.items() if entries[qid].qtype == qtype])
    }
    selected: dict[str, list[float]] = {"below_median_first_spread": [], "above_median_first_spread": []}
    for row in paired:
        qid = int(row["unit_id"])
        if qid not in spreads:
            continue
        bucket = (
            "below_median_first_spread"
            if spreads[qid] <= type_medians[entries[qid].qtype]
            else "above_median_first_spread"
        )
        selected[bucket].append(float(row["difference"]))
    return {
        key: {"mean_difference": statistics.fmean(values) if values else None, "n_question_draws": len(values)}
        for key, values in selected.items()
    }
