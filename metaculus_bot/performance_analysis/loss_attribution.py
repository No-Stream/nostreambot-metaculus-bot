"""Attribute losses by era using accounting and outcome diagnostics.

Intervals require ten observations and five resolution days. See the Loss attribution
section of docs/performance_analysis.md for interpretation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.optimize import isotonic_regression

from metaculus_bot.performance_analysis.analysis import PitReading, jeffreys_ci, per_model_cohort
from metaculus_bot.performance_analysis.cohorts import EXCLUSION_COHORTS, parse_exclude_qids
from metaculus_bot.performance_analysis.collector import load_dataset, resolve_numeric_record_to_score_inputs
from metaculus_bot.performance_analysis.era_gap import ClusterSums
from metaculus_bot.performance_analysis.ghosts import (
    ghost_binary_prob,
    ghost_cdf_on_published_grid,
    ghost_mc_probs,
    ghost_numeric_percentiles,
    load_concluding_ghosts,
)
from metaculus_bot.performance_analysis.loss_attribution_report import render_report
from metaculus_bot.performance_analysis.platform_scores import spot_peer_score
from metaculus_bot.performance_analysis.width_monitor import compute_pit_reading
from metaculus_bot.scoring_common import numeric_log_score, resolution_to_bucket_index, spot_peer_delta
from metaculus_bot.time_utils import parse_iso_utc

logger = logging.getLogger(__name__)


DEFAULT_SEED = 20261125
DEFAULT_DRAWS = 2000
MIN_N_FOR_INTERVAL = 10
MIN_CLUSTERS_FOR_INTERVAL = 5
# A bootstrap of a rare count is degenerate (no resampled winner gives an interval excluding zero), so the
# observed-versus-expected table also needs this many expected winners inside and outside the band.
MIN_EXPECTED_FOR_INTERVAL = 5
FLOOR_THRESHOLD = 0.0101
PROBABILITY_EDGES = (0.0, 0.02, 0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 0.98, 1.0)
BINARY_CONFIDENCE_EDGES = (0.5, 0.65, 0.85, 0.95, 1.0)
MC_CONFIDENCE_EDGES = (0.0, 0.35, 0.5, 0.65, 0.85, 1.0)
MC_OPTION_EDGES = (0.0, 0.02, 0.05, 0.15, 0.35, 0.65, 0.85, 1.0)
PIT_EDGES = (0.0, 0.025, 0.1, 0.9, 0.975, 1.0)
DISTANCE_EDGES = (0.0, 0.05, 0.15, 0.3, 1.0)


def _bin_label(edges: tuple[float, ...], index: int) -> str:
    return f"[{edges[index]:g}, {edges[index + 1]:g}{']' if index == len(edges) - 2 else ')'}"


def _interval_allowed(n: int, clusters: list[str] | set[str]) -> bool:
    return n >= MIN_N_FOR_INTERVAL and len(set(clusters)) >= MIN_CLUSTERS_FOR_INTERVAL


@dataclass(frozen=True, slots=True)
class CorpResult:
    """CORP decomposition of a mean proper score."""

    n: int
    mean_score: float
    mcb: float
    dsc: float
    unc: float


def corp_decomposition(probabilities: list[float], outcomes: list[bool], score: Literal["brier", "log"]) -> CorpResult:
    """Fit isotonic recalibration, pooling equal forecasts before calculating scores."""
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("Forecasts and outcomes must have equal nonzero length")
    if any(not math.isfinite(probability) or not 0 <= probability <= 1 for probability in probabilities):
        raise ValueError("Forecast probabilities must be finite and between 0 and 1")
    if score not in ("brier", "log"):
        raise ValueError(f"Unknown score: {score}")
    pairs = sorted(zip(probabilities, outcomes, strict=True))
    _unique, inverse, counts = np.unique([pair[0] for pair in pairs], return_inverse=True, return_counts=True)
    pooled = np.bincount(inverse, weights=[int(pair[1]) for pair in pairs]) / counts
    fitted = isotonic_regression(pooled, weights=counts).x[inverse]

    def loss(probability: float, outcome: bool) -> float:
        if score == "brier":
            return (probability - int(outcome)) ** 2
        realized = probability if outcome else 1 - probability
        return -math.log(realized) if realized > 0 else math.inf

    n = len(pairs)
    raw = sum(loss(probability, outcome) for probability, outcome in pairs) / n
    if not math.isfinite(raw):
        raise ValueError("A forecast assigns zero probability to the outcome")
    recalibrated = sum(loss(float(value), outcome) for value, (_, outcome) in zip(fitted, pairs, strict=True)) / n
    base_rate = sum(outcome for _, outcome in pairs) / n
    uncertainty = sum(loss(base_rate, outcome) for _, outcome in pairs) / n
    return CorpResult(n=n, mean_score=raw, mcb=raw - recalibrated, dsc=uncertainty - recalibrated, unc=uncertainty)


def winner_rank(probabilities: list[float], winner_index: int) -> float:
    """Return the winner's normalized average rank, including ties."""
    if len(probabilities) < 2 or not 0 <= winner_index < len(probabilities):
        raise ValueError("A ballot needs at least two options and a valid winner index")
    winner = probabilities[winner_index]
    higher = sum(probability > winner for probability in probabilities)
    tied = sum(probability == winner for probability in probabilities)
    return (higher + (tied - 1) / 2) / (len(probabilities) - 1)


@dataclass(frozen=True, slots=True)
class ScoreBin:
    label: str
    n: int
    sum_spot_peer: float
    mean_spot_peer: float
    negative_share: float
    fraction_negative: float
    interval: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class ProbabilityMassBin:
    label: str
    n_questions: int
    n_with_mass: int
    observed: int
    expected: float
    observed_minus_expected: float
    interval: tuple[float, float] | None


def probability_mass_calibration(
    records: list[dict], question_type: str, *, seed: int = DEFAULT_SEED, draws: int = DEFAULT_DRAWS
) -> tuple[ProbabilityMassBin, ...]:
    """Compare winner counts with forecast mass in outcome-probability bands."""
    if question_type not in ("binary", "multiple_choice"):
        raise ValueError(f"Unsupported question type: {question_type}")
    edges = PROBABILITY_EDGES if question_type == "binary" else MC_OPTION_EDGES
    observations_by_bin, question_count = _probability_mass_observations(records, question_type, edges)
    return tuple(
        _probability_mass_bin(index, edges, observations, question_count, seed=seed, draws=draws)
        for index, observations in enumerate(observations_by_bin)
    )


def _probability_mass_observations(
    records: list[dict], question_type: str, edges: tuple[float, ...]
) -> tuple[list[list[tuple[float, str, float, int]]], int]:
    observations_by_bin: list[list[tuple[float, str, float, int]]] = [[] for _ in range(len(edges) - 1)]
    question_count = 0
    for record in records:
        if record["type"] != question_type or _outcome_exclusion(record) is not None:
            continue
        if question_type == "binary":
            probabilities = [record["our_prob_yes"], 1 - record["our_prob_yes"]]
            winner_index = 0 if record["resolution_parsed"] else 1
        else:
            probabilities = record["our_forecast_values"]
            winner_index = record["options"].index(record["resolution_parsed"])
        cluster = _cluster(record)
        question_count += 1
        mass_by_bin = [0.0] * (len(edges) - 1)
        for probability in probabilities:
            mass_by_bin[_bin_index(float(probability), edges)] += float(probability)
        winner_bin = _bin_index(float(probabilities[winner_index]), edges)
        for index, mass in enumerate(mass_by_bin):
            observed = int(index == winner_bin)
            observations_by_bin[index].append((observed - mass, cluster, mass, observed))
    return observations_by_bin, question_count


def _probability_mass_bin(
    index: int,
    edges: tuple[float, ...],
    observations: list[tuple[float, str, float, int]],
    question_count: int,
    *,
    seed: int,
    draws: int,
) -> ProbabilityMassBin:
    deltas = [delta for delta, _, _, _ in observations]
    clusters = [cluster for _, cluster, _, _ in observations]
    masses = [mass for _, _, mass, _ in observations]
    active_clusters = [cluster for cluster, mass in zip(clusters, masses, strict=True) if mass > 0]
    active_count = sum(mass > 0 for mass in masses)
    expected = sum(masses)
    interval = None
    if (
        _interval_allowed(active_count, active_clusters)
        and expected >= MIN_EXPECTED_FOR_INTERVAL
        and active_count - expected >= MIN_EXPECTED_FOR_INTERVAL
    ):
        if draws <= 0:
            raise ValueError("draws must be positive")
        means = ClusterSums.from_values(deltas, clusters).resampled_means(np.random.default_rng(seed), draws)
        bounds = np.percentile(means * len(deltas), [2.5, 97.5])
        interval = float(bounds[0]), float(bounds[1])
    return ProbabilityMassBin(
        _bin_label(edges, index),
        question_count,
        active_count,
        sum(observed for _, _, _, observed in observations),
        expected,
        sum(deltas),
        interval,
    )


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    for index, (lower, upper) in enumerate(pairwise(edges)):
        if lower <= value < upper or (index == len(edges) - 2 and value == upper):
            return index
    raise ValueError(f"Value {value} outside bin edges {edges}")


def _cluster(record: dict) -> str:
    resolved = parse_iso_utc((record.get("metadata") or {}).get("actual_resolve_time"))
    if resolved is None:
        raise ValueError(f"Missing resolution timestamp for question {record.get('question_id')}")
    return resolved.date().isoformat()


def cluster_interval(
    values: list[float], clusters: list[str], *, seed: int = DEFAULT_SEED, draws: int = DEFAULT_DRAWS
) -> tuple[float, float] | None:
    """Return a resolution-day bootstrap interval when the sample is large enough."""
    if not _interval_allowed(len(values), clusters):
        return None
    if draws <= 0:
        raise ValueError("draws must be positive")
    means = ClusterSums.from_values(values, clusters).resampled_means(np.random.default_rng(seed), draws)
    bounds = np.percentile(means, [2.5, 97.5])
    return float(bounds[0]), float(bounds[1])


def _accounting_value(record: dict, question_type: str, *, confidence: bool) -> float | None:
    if question_type == "binary":
        probability = record["our_prob_yes"]
        if probability is None:
            return None
        if confidence:
            return max(probability, 1 - probability)
        outcome = record["resolution_parsed"]
        if not isinstance(outcome, bool):
            return None
        return probability if outcome else 1 - probability
    probabilities = record["our_forecast_values"]
    options = record["options"]
    if not probabilities or not options or len(probabilities) != len(options):
        return None
    if confidence:
        return max(probabilities)
    outcome = record["resolution_parsed"]
    return probabilities[options.index(outcome)] if outcome in options else None


def points_lost(
    records: list[dict],
    question_type: str,
    *,
    confidence: bool = False,
    p_yes: bool = False,
    seed: int = DEFAULT_SEED,
    draws: int = DEFAULT_DRAWS,
) -> tuple[ScoreBin, ...]:
    """Group scored binary or MC forecasts by realized mass or outcome-blind confidence."""
    outcome_edges = PROBABILITY_EDGES
    edges = (
        (BINARY_CONFIDENCE_EDGES if question_type == "binary" else MC_CONFIDENCE_EDGES) if confidence else outcome_edges
    )
    if question_type not in ("binary", "multiple_choice"):
        raise ValueError(f"Unsupported question type: {question_type}")
    bins: list[list[tuple[float, str]]] = [[] for _ in range(len(edges) - 1)]
    for record in records:
        if record["type"] != question_type:
            continue
        peer = spot_peer_score(record)
        if peer is None:
            continue
        value = record["our_prob_yes"] if p_yes else _accounting_value(record, question_type, confidence=confidence)
        if value is None:
            continue
        bins[_bin_index(float(value), edges)].append((peer, _cluster(record)))
    total_negative = sum(min(peer, 0.0) for group in bins for peer, _ in group)
    rows = []
    for index, group in enumerate(bins):
        if not group:
            continue
        values = [peer for peer, _ in group]
        clusters = [cluster for _, cluster in group]
        label = _bin_label(edges, index)
        rows.append(
            ScoreBin(
                label,
                len(group),
                sum(values),
                sum(values) / len(values),
                sum(min(value, 0.0) for value in values) / total_negative if total_negative else 0.0,
                sum(value < 0 for value in values) / len(values),
                cluster_interval(values, clusters, seed=seed, draws=draws),
            )
        )
    return tuple(rows)


def era_buckets(records: list[dict], era_field: str) -> dict[str, list[dict]]:
    """Keep null era labels and always include the fall configuration."""
    buckets: dict[str, list[dict]] = {"all": records}
    labels = sorted(
        {str(record.get(era_field)) if record.get(era_field) is not None else "(none)" for record in records}
    )
    for label in labels:
        buckets[label] = [
            record
            for record in records
            if (str(record.get(era_field)) if record.get(era_field) is not None else "(none)") == label
        ]
    if era_field != "triple_subera" or "triple_fall_config" not in buckets:
        buckets["triple_fall_config"] = [
            record for record in records if record.get("triple_subera") == "triple_fall_config"
        ]
    return buckets


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    label: str
    n: int
    mean_forecast: float
    observed: float
    jeffreys_interval: tuple[float, float] | None
    observed_minus_forecast: float


def reliability_table(
    pairs: list[tuple[float, bool]], edges: tuple[float, ...], *, clusters: list[str] | None = None
) -> tuple[ReliabilityBin, ...]:
    """Bin calibration observations and compute Jeffreys intervals."""
    if clusters is not None and len(clusters) != len(pairs):
        raise ValueError("Cluster count must match observation count")
    bins: list[list[tuple[float, bool, str]]] = [[] for _ in range(len(edges) - 1)]
    for (probability, outcome), cluster in zip(pairs, clusters or [""] * len(pairs), strict=True):
        bins[_bin_index(probability, edges)].append((probability, outcome, cluster))
    result = []
    for index, group in enumerate(bins):
        if not group:
            continue
        count = len(group)
        hits = sum(outcome for _, outcome, _ in group)
        mean_forecast = sum(probability for probability, _, _ in group) / count
        observed = hits / count
        _, lower, upper = jeffreys_ci(hits, count)
        interval = (lower, upper) if _interval_allowed(count, [cluster for _, _, cluster in group]) else None
        label = _bin_label(edges, index)
        result.append(ReliabilityBin(label, count, mean_forecast, observed, interval, observed - mean_forecast))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DiscriminationRow:
    name: str
    n: int
    incomplete: int
    zero_probability: int
    tied_top: int
    scored_n: int
    paired_n: int
    top_accuracy: float | None
    top_interval: tuple[float, float] | None
    mean_rank: float | None
    mean_winner_probability: float | None
    mean_counterfactual_spot: float | None
    mean_delta: float | None
    delta_interval: tuple[float, float] | None
    member_only_top: int
    published_only_top: int


def mc_discrimination(
    records: list[dict], *, seed: int = DEFAULT_SEED, draws: int = DEFAULT_DRAWS
) -> tuple[DiscriminationRow, ...]:
    """Compare complete member ballots with the published MC forecast."""
    ballots: dict[str, list[tuple[dict, list[float]]]] = {"published": []}
    incomplete: dict[str, int] = {}
    mc_records = [record for record in records if record["type"] == "multiple_choice"]
    for record in mc_records:
        options = record["options"]
        forecast = record["our_forecast_values"]
        if options and forecast and len(options) == len(forecast) and record["resolution_parsed"] in options:
            ballots["published"].append((record, forecast))
    for record, members in per_model_cohort(mc_records, cut="loss_attribution_mc"):
        options = record["options"]
        if not options or record["resolution_parsed"] not in options:
            continue
        for name, member in members.items():
            ballots.setdefault(name, [])
            if not isinstance(member, dict) or any(option not in member for option in options):
                incomplete[name] = incomplete.get(name, 0) + 1
                continue
            ballots[name].append((record, [float(member[option]) for option in options]))
    return tuple(
        _discrimination_row(name, items, incomplete_count=incomplete.get(name, 0), seed=seed, draws=draws)
        for name, items in ballots.items()
    )


@dataclass(frozen=True, slots=True)
class _BallotReading:
    top: bool
    tied_top: bool
    rank: float
    probability: float
    peer: float | None
    delta: float | None
    cluster: str
    member_only: bool
    published_only: bool
    zero_probability: bool
    comparable: bool


def _read_ballot(record: dict, forecast: list[float]) -> _BallotReading:
    options = record["options"]
    winner = options.index(record["resolution_parsed"])
    published = record["our_forecast_values"]
    member_top = forecast[winner] == max(forecast)
    published_top = bool(published and len(published) == len(options) and published[winner] == max(published))
    comparable = bool(published and len(published) == len(options))
    peer = spot_peer_score(record)
    zero = bool(comparable and (forecast[winner] <= 0 or published[winner] <= 0))
    delta = None
    if comparable and not zero and peer is not None:
        delta = spot_peer_delta(old_prob=published[winner], new_prob=forecast[winner], question_type="multiple_choice")
    return _BallotReading(
        forecast.index(max(forecast)) == winner,
        forecast.count(max(forecast)) > 1,
        winner_rank(forecast, winner),
        forecast[winner],
        peer,
        delta,
        _cluster(record),
        comparable and member_top and not published_top,
        comparable and published_top and not member_top,
        zero,
        comparable,
    )


def _discrimination_row(
    name: str, member_ballots: list[tuple[dict, list[float]]], *, incomplete_count: int, seed: int, draws: int
) -> DiscriminationRow:
    readings = [_read_ballot(record, forecast) for record, forecast in member_ballots]
    scored = [
        (reading.peer, reading.delta, reading.cluster)
        for reading in readings
        if reading.peer is not None and reading.delta is not None
    ]
    n = len(readings)
    hits = sum(reading.top for reading in readings)
    _, low, high = jeffreys_ci(hits, n) if n else (0.0, 0.0, 0.0)
    return DiscriminationRow(
        name,
        n,
        incomplete_count,
        sum(reading.zero_probability for reading in readings),
        sum(reading.tied_top for reading in readings),
        len(scored),
        sum(reading.comparable for reading in readings),
        hits / n if n else None,
        (low, high) if _interval_allowed(n, [reading.cluster for reading in readings]) else None,
        sum(reading.rank for reading in readings) / n if n else None,
        sum(reading.probability for reading in readings) / n if n else None,
        sum(peer + delta for peer, delta, _ in scored) / len(scored) if scored else None,
        sum(delta for _, delta, _ in scored) / len(scored) if scored else None,
        cluster_interval(
            [delta for _, delta, _ in scored], [cluster for _, _, cluster in scored], seed=seed, draws=draws
        ),
        sum(reading.member_only for reading in readings),
        sum(reading.published_only for reading in readings),
    )


@dataclass(frozen=True, slots=True)
class SliceRow:
    dimension: str
    value: str
    n: int
    mean_spot_peer: float
    interval: tuple[float, float] | None


def slice_ranking(records: list[dict], *, seed: int = DEFAULT_SEED, draws: int = DEFAULT_DRAWS) -> tuple[SliceRow, ...]:
    """Describe spot peer by question and pipeline attributes."""
    dimensions = (
        "type",
        "source_tournament",
        "option_count",
        "anchor_present",
        "gfv2_loop_ran",
        "was_stacked",
        "forecasters_used",
    )
    groups: dict[tuple[str, str], list[tuple[float, str]]] = {}
    for record in records:
        peer = spot_peer_score(record)
        if peer is None:
            continue
        for dimension in dimensions:
            if dimension == "option_count":
                if record["type"] != "multiple_choice":
                    continue
                value = str(len(record["options"]))
            elif dimension in ("anchor_present", "gfv2_loop_ran"):
                source = record.get("research_source_class")
                if source is None:
                    continue
                value = f"{source} / {record.get(dimension)}"
            else:
                value = str(record.get(dimension))
            groups.setdefault((dimension, value), []).append((peer, _cluster(record)))
    rows = []
    for (dimension, value), observations in groups.items():
        values = [peer for peer, _ in observations]
        clusters = [cluster for _, cluster in observations]
        rows.append(
            SliceRow(
                dimension,
                value,
                len(values),
                sum(values) / len(values),
                cluster_interval(values, clusters, seed=seed, draws=draws),
            )
        )
    return tuple(sorted(rows, key=lambda row: row.mean_spot_peer))


def pit_bin_label(reading: PitReading, record: dict) -> str:
    """Keep out-of-bounds PIT intervals separate from point-valued bins."""
    if reading.oob_side is not None:
        bound = "open_lower_bound" if reading.oob_side == "low" else "open_upper_bound"
        return f"{reading.oob_side}, {'open' if record[bound] else 'closed'}"
    if reading.point is None:
        raise ValueError("A PIT interval must have an out-of-bounds side")
    edges = PIT_EDGES
    index = _bin_index(reading.point, edges)
    return _bin_label(edges, index)


def pit_accounting(
    records: list[dict], *, seed: int = DEFAULT_SEED, draws: int = DEFAULT_DRAWS
) -> tuple[ScoreBin, ...]:
    """Describe numeric and discrete spot peer by PIT placement."""
    groups: dict[str, list[tuple[float, str]]] = {}
    for record in records:
        if record["type"] not in ("numeric", "discrete"):
            continue
        peer = spot_peer_score(record)
        reading = compute_pit_reading(record)
        if peer is None or reading is None:
            continue
        groups.setdefault(pit_bin_label(reading, record), []).append((peer, _cluster(record)))
    negative = sum(min(peer, 0.0) for group in groups.values() for peer, _ in group)
    rows = []
    for label, group in sorted(groups.items()):
        values = [peer for peer, _ in group]
        clusters = [cluster for _, cluster in group]
        rows.append(
            ScoreBin(
                label,
                len(values),
                sum(values),
                sum(values) / len(values),
                sum(min(value, 0.0) for value in values) / negative if negative else 0.0,
                sum(value < 0 for value in values) / len(values),
                cluster_interval(values, clusters, seed=seed, draws=draws),
            )
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class GhostComparison:
    status: str
    distance: float | None = None
    delta: float | None = None


def _bucket_mass(cdf: list[float], bucket: int) -> float:
    if bucket == 0:
        return cdf[0]
    if bucket == len(cdf):
        return 1.0 - cdf[-1]
    return cdf[bucket] - cdf[bucket - 1]


def _numeric_ghost_comparison(record: dict, ghost: dict) -> GhostComparison:
    percentiles = ghost_numeric_percentiles(ghost)
    inputs = resolve_numeric_record_to_score_inputs(record)
    published = record["our_forecast_values"]
    if percentiles is None or inputs is None or not published:
        return GhostComparison("missing_or_incomplete_vector")
    alternative = ghost_cdf_on_published_grid(percentiles, record)
    if alternative is None:
        return GhostComparison("cdf_build_failed")
    resolution, lower, upper, zero = inputs
    bucket = resolution_to_bucket_index(resolution, lower, upper, n_inbound=len(published) - 1, zero_point=zero)
    if _bucket_mass(published, bucket) <= 0 or _bucket_mass(alternative, bucket) <= 0:
        return GhostComparison("zero_probability_on_outcome")
    kwargs = {
        "open_lower_bound": record["open_lower_bound"],
        "open_upper_bound": record["open_upper_bound"],
        "zero_point": zero,
    }
    delta = numeric_log_score(alternative, resolution, lower, upper, **kwargs) - numeric_log_score(
        published, resolution, lower, upper, **kwargs
    )
    distance = sum(abs(first - second) for first, second in zip(published, alternative, strict=True)) / len(published)
    return GhostComparison("scored", distance, delta)


def ghost_comparison(record: dict, ghost: dict) -> GhostComparison:
    """Compare a ghost once and retain the reason when scoring is impossible."""
    question_type = record["type"]
    if question_type in ("numeric", "discrete") and ghost["qtype"] in ("numeric", "discrete"):
        return _numeric_ghost_comparison(record, ghost)
    if question_type != ghost["qtype"]:
        return GhostComparison("not_comparable_type")
    if question_type == "binary":
        alternative = ghost_binary_prob(ghost)
        published = record["our_prob_yes"]
        if alternative is None or published is None:
            return GhostComparison("missing_or_incomplete_vector")
        if not isinstance(record["resolution_parsed"], bool):
            return GhostComparison("invalid_outcome")
        distance = abs(published - alternative)
        old = published if record["resolution_parsed"] else 1 - published
        new = alternative if record["resolution_parsed"] else 1 - alternative
    elif question_type == "multiple_choice":
        alternatives = ghost_mc_probs(ghost)
        options = record["options"]
        published_vector = record["our_forecast_values"]
        if (
            not alternatives
            or not options
            or not published_vector
            or len(options) != len(published_vector)
            or any(option not in alternatives for option in options)
        ):
            return GhostComparison("missing_or_incomplete_vector")
        if record["resolution_parsed"] not in options:
            return GhostComparison("outcome_not_among_options")
        alternative_vector = [alternatives[option] for option in options]
        distance = (
            sum(abs(first - second) for first, second in zip(published_vector, alternative_vector, strict=True)) / 2
        )
        winner = options.index(record["resolution_parsed"])
        old, new = published_vector[winner], alternative_vector[winner]
    else:
        return GhostComparison("not_comparable_type")
    if old <= 0 or new <= 0:
        return GhostComparison("zero_probability_on_outcome")
    return GhostComparison("scored", distance, spot_peer_delta(old_prob=old, new_prob=new, question_type=question_type))


@dataclass(frozen=True, slots=True)
class FloorSummary:
    questions: int
    options: int
    hits: int
    expected_hits: float
    interval: tuple[float, float] | None


def floor_summary(records: list[dict]) -> FloorSummary:
    """Reproduce the floor calibration script's 1.01% option summary."""
    options: list[tuple[float, bool]] = []
    clusters: set[str] = set()
    questions = 0
    for record in records:
        if record["type"] != "multiple_choice":
            continue
        names, probabilities, outcome = record["options"], record["our_forecast_values"], record["resolution_parsed"]
        if not names or not probabilities or len(names) != len(probabilities) or outcome not in names:
            continue
        at_floor = [
            (probability, name == outcome)
            for name, probability in zip(names, probabilities, strict=True)
            if probability <= FLOOR_THRESHOLD
        ]
        if at_floor:
            questions += 1
            options.extend(at_floor)
            if record.get("metadata"):
                clusters.add(_cluster(record))
    hits = sum(outcome for _, outcome in options)
    interval = None
    if _interval_allowed(len(options), clusters):
        _, low, high = jeffreys_ci(hits, len(options))
        interval = (low, high)
    return FloorSummary(questions, len(options), hits, sum(probability for probability, _ in options), interval)


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    name: str
    brier: CorpResult
    log: CorpResult
    reliability: tuple[ReliabilityBin, ...]
    tied_top: int = 0


def calibration_reports(records: list[dict]) -> tuple[CalibrationReport, ...]:
    """Calculate binary, top-choice, and secondary option-level calibration."""
    binary: list[tuple[float, bool, str]] = []
    top: list[tuple[float, bool, str]] = []
    options: list[tuple[float, bool, str]] = []
    for record in records:
        if record["type"] == "binary":
            probability = record["our_prob_yes"]
            outcome = record["resolution_parsed"]
            if probability is not None and isinstance(outcome, bool):
                binary.append((probability, outcome, _cluster(record)))
        elif record["type"] == "multiple_choice":
            names, probabilities, outcome = (
                record["options"],
                record["our_forecast_values"],
                record["resolution_parsed"],
            )
            if not names or not probabilities or len(names) != len(probabilities) or outcome not in names:
                continue
            # A tie at the top scores the first tied option in option order, chosen before the outcome is read.
            top_index = probabilities.index(max(probabilities))
            top.append((probabilities[top_index], names[top_index] == outcome, _cluster(record)))
            options.extend(
                (probability, name == outcome, _cluster(record))
                for name, probability in zip(names, probabilities, strict=True)
            )
    specifications = (
        ("binary p_yes", binary, PROBABILITY_EDGES),
        ("MC top option", top, MC_CONFIDENCE_EDGES),
        (
            "MC secondary: options within a ballot are dependent",
            options,
            MC_OPTION_EDGES,
        ),
    )
    reports = []
    for name, pairs, edges in specifications:
        if not pairs:
            continue
        probabilities = [probability for probability, _, _ in pairs]
        outcomes = [outcome for _, outcome, _ in pairs]
        clusters = [cluster for _, _, cluster in pairs]
        observations = list(zip(probabilities, outcomes, strict=True))
        reports.append(
            CalibrationReport(
                name,
                corp_decomposition(probabilities, outcomes, "brier"),
                corp_decomposition(probabilities, outcomes, "log"),
                reliability_table(observations, edges, clusters=clusters),
                sum(
                    1
                    for record in records
                    if record["type"] == "multiple_choice"
                    and record["our_forecast_values"]
                    and record["options"]
                    and len(record["our_forecast_values"]) == len(record["options"])
                    and record["resolution_parsed"] in record["options"]
                    and record["our_forecast_values"].count(max(record["our_forecast_values"])) > 1
                )
                if name == "MC top option"
                else 0,
            )
        )
    return tuple(reports)


@dataclass(frozen=True, slots=True)
class GhostQuestion:
    post_id: int
    url: str
    distance: float
    spot_peer_change: float


@dataclass(frozen=True, slots=True)
class GhostBin:
    question_type: str
    label: str
    distance_lower: float | None
    n: int
    mean_delta: float
    interval: tuple[float, float] | None
    questions: tuple[GhostQuestion, ...]


@dataclass(frozen=True, slots=True)
class GhostCounts:
    question_type: str
    statuses: dict[str, int]
    excluded_spot_peer: float


@dataclass(frozen=True, slots=True)
class GhostReport:
    rows: tuple[GhostBin, ...]
    unjoinable: int
    unscoreable: int
    zero_probability: int
    excluded_spot_peer: float
    counts_by_type: tuple[GhostCounts, ...]


def disagreement_report(records: list[dict], ghosts: dict[int, dict], *, seed: int, draws: int) -> GhostReport:
    """Group joined ghosts by distance and count each exclusion reason."""
    groups: dict[tuple[str, int], list[tuple[float, str, GhostQuestion]]] = {}
    counts: dict[str, Counter[str]] = {}
    excluded_scores: dict[str, float] = {}
    for record in records:
        type_counts = counts.setdefault(record["type"], Counter())
        ghost = ghosts.get(record["post_id"])
        if ghost is None:
            type_counts["unjoinable"] += 1
            peer = spot_peer_score(record)
            if peer is not None:
                excluded_scores[record["type"]] = excluded_scores.get(record["type"], 0.0) + peer
            continue
        comparison = ghost_comparison(record, ghost)
        status = comparison.status
        if status == "scored" and spot_peer_score(record) is None:
            status = "no_spot_peer"
        type_counts[status] += 1
        if status != "scored":
            peer = spot_peer_score(record)
            if peer is not None:
                excluded_scores[record["type"]] = excluded_scores.get(record["type"], 0.0) + peer
            continue
        assert comparison.distance is not None
        assert comparison.delta is not None
        question = GhostQuestion(
            int(record["post_id"]),
            f"https://www.metaculus.com/questions/{record['post_id']}/",
            comparison.distance,
            comparison.delta,
        )
        observation = (comparison.delta, _cluster(record), question)
        groups.setdefault((record["type"], -1), []).append(observation)
        groups.setdefault((record["type"], _bin_index(comparison.distance, DISTANCE_EDGES)), []).append(observation)
    rows = []
    for (question_type, index), observations in sorted(groups.items()):
        values = [value for value, _, _ in observations]
        clusters = [cluster for _, cluster, _ in observations]
        questions = tuple(sorted((question for _, _, question in observations), key=lambda item: item.spot_peer_change))
        rows.append(
            GhostBin(
                question_type,
                "all distances" if index == -1 else _bin_label(DISTANCE_EDGES, index),
                None if index == -1 else DISTANCE_EDGES[index],
                len(values),
                sum(values) / len(values),
                cluster_interval(values, clusters, seed=seed, draws=draws),
                questions,
            )
        )
    by_type = tuple(
        GhostCounts(kind, dict(count), excluded_scores.get(kind, 0.0)) for kind, count in sorted(counts.items())
    )
    return GhostReport(
        tuple(rows),
        sum(c["unjoinable"] for c in counts.values()),
        sum(sum(v for k, v in c.items() if k not in ("unjoinable", "scored")) for c in counts.values()),
        sum(c["zero_probability_on_outcome"] for c in counts.values()),
        sum(excluded_scores.values()),
        by_type,
    )


@dataclass(frozen=True, slots=True)
class EraReport:
    era: str
    accounting: dict[str, tuple[ScoreBin, ...]]
    probability_mass: dict[str, tuple[ProbabilityMassBin, ...]]
    calibration: tuple[CalibrationReport, ...]
    floor: FloorSummary
    exclusions: dict[str, dict[str, int]]
    excluded_spot_peer: dict[str, float]
    discrimination: tuple[DiscriminationRow, ...]
    ghost: GhostReport | None
    slices: tuple[SliceRow, ...]
    worst: tuple[dict, ...]


def _ghost_discrimination(records: list[dict], ghosts: dict[int, dict], *, seed: int, draws: int) -> DiscriminationRow:
    ballots: list[tuple[dict, list[float]]] = []
    incomplete = 0
    for record in records:
        if record["type"] != "multiple_choice":
            continue
        ghost = ghosts.get(record["post_id"])
        if ghost is None or ghost["qtype"] != "multiple_choice":
            continue
        probabilities = ghost_mc_probs(ghost)
        options = record["options"]
        if not probabilities or not options or any(option not in probabilities for option in options):
            incomplete += 1
            continue
        if record["resolution_parsed"] in options:
            ballots.append((record, [probabilities[option] for option in options]))
    return _discrimination_row("ghost", ballots, incomplete_count=incomplete, seed=seed, draws=draws)


def _has_complete_member_vector(record: dict) -> bool:
    options = record.get("options") or []
    members = record.get("per_model_forecasts") or {}
    return (
        bool(options)
        and bool(members)
        and all(isinstance(member, dict) and all(option in member for option in options) for member in members.values())
    )


def _outcome_exclusion(record: dict, *, confidence: bool = False) -> str | None:
    kind = record["type"]
    if _accounting_value(record, kind, confidence=confidence) is not None:
        return None
    if kind == "binary" and record["our_prob_yes"] is not None:
        return "non-bool binary outcome"
    if kind == "multiple_choice":
        options = record["options"]
        probabilities = record["our_forecast_values"]
        if (
            options
            and probabilities
            and len(options) == len(probabilities)
            and record["resolution_parsed"] not in options
        ):
            return "outcome not among options"
    return "no usable forecast vector"


def _accounting_exclusion(record: dict, name: str) -> str | None:
    if spot_peer_score(record) is None:
        return "no spot peer"
    if record["type"] in ("numeric", "discrete"):
        return "no PIT reading" if compute_pit_reading(record) is None else None
    if name.endswith("by p_yes"):
        return "no usable forecast vector" if record["our_prob_yes"] is None else None
    return _outcome_exclusion(record, confidence=name.endswith("by our confidence"))


def _section_exclusions(
    records: list[dict], accounting: dict[str, tuple[ScoreBin, ...]]
) -> tuple[dict[str, dict[str, int]], dict[str, float]]:
    exclusions: dict[str, dict[str, int]] = {}
    excluded_spot_peer: dict[str, float] = {}
    for name in accounting:
        kinds = ("numeric", "discrete") if name.startswith("numeric and discrete:") else (name.split(":")[0],)
        counts: Counter[str] = Counter()
        excluded_score = 0.0
        for record in records:
            if record["type"] not in kinds:
                continue
            reason = _accounting_exclusion(record, name)
            if reason is not None:
                counts[reason] += 1
                peer = spot_peer_score(record)
                if peer is not None:
                    excluded_score += peer
        exclusions[name] = dict(counts)
        excluded_spot_peer[name] = excluded_score
    for kind in ("binary", "multiple_choice"):
        reasons = (_outcome_exclusion(record) for record in records if record["type"] == kind)
        exclusions[f"{kind}: outcome diagnostics"] = dict(Counter(reason for reason in reasons if reason is not None))
    exclusions["slices and worst decile"] = {"no spot peer": sum(spot_peer_score(record) is None for record in records)}
    exclusions["research slices"] = {
        "no research source class": sum(
            spot_peer_score(record) is not None and record.get("research_source_class") is None for record in records
        )
    }
    excluded_spot_peer["slices and worst decile"] = 0.0
    research_excluded_score = 0.0
    for record in records:
        if record.get("research_source_class") is None:
            peer = spot_peer_score(record)
            if peer is not None:
                research_excluded_score += peer
    excluded_spot_peer["research slices"] = research_excluded_score
    return exclusions, excluded_spot_peer


def build_era_report(
    era: str, records: list[dict], ghosts: dict[int, dict] | None, *, seed: int, draws: int
) -> EraReport:
    """Compute accounting and diagnostics for one era."""
    accounting = {}
    for question_type in ("binary", "multiple_choice"):
        for confidence in (False, True):
            label = f"{question_type}: {'by our confidence' if confidence else 'where the points went'}"
            accounting[label] = points_lost(records, question_type, confidence=confidence, seed=seed, draws=draws)
    accounting["binary: by p_yes"] = points_lost(records, "binary", p_yes=True, seed=seed, draws=draws)
    accounting["numeric and discrete: placement in the distribution"] = pit_accounting(records, seed=seed, draws=draws)
    probability_mass = {
        question_type: probability_mass_calibration(records, question_type, seed=seed, draws=draws)
        for question_type in ("binary", "multiple_choice")
    }
    scored: list[tuple[float, dict]] = []
    for record in records:
        score = spot_peer_score(record)
        if score is not None:
            scored.append((score, record))
    scored.sort(key=lambda item: item[0])
    worst = tuple(
        {
            "question_id": record["question_id"],
            "type": record["type"],
            "era": era,
            "spot_peer": peer,
            "title": record.get("title"),
            "url": f"https://www.metaculus.com/questions/{record['post_id']}/",
        }
        for peer, record in scored[: math.ceil(len(scored) / 10)]
    )
    exclusions, excluded_spot_peer = _section_exclusions(records, accounting)
    discrimination = mc_discrimination(records, seed=seed, draws=draws)
    if ghosts is not None:
        discrimination += (_ghost_discrimination(records, ghosts, seed=seed, draws=draws),)
    return EraReport(
        era,
        accounting,
        probability_mass,
        calibration_reports(records),
        floor_summary(records),
        exclusions,
        excluded_spot_peer,
        discrimination,
        None if ghosts is None else disagreement_report(records, ghosts, seed=seed, draws=draws),
        slice_ranking(records, seed=seed, draws=draws),
        worst,
    )


def main(argv: list[str] | None = None) -> None:
    """Write the report from cached data and an optional local telemetry archive."""
    parser = argparse.ArgumentParser(description="Offline loss attribution by era")
    parser.add_argument("--cached", required=True, help="Path to the cached performance dataset JSON")
    parser.add_argument("--exclude-qids", default="", help="Comma-separated question IDs or exclusion cohort names")
    parser.add_argument("--era-field", default="triple_subera", help="Dataset field used to group configuration eras")
    parser.add_argument("--telemetry-archive", help="Local telemetry archive directory for ghost comparisons")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Resolution-day bootstrap random seed")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS, help="Number of cluster bootstrap draws")
    parser.add_argument("--out", required=True, help="Output directory for Markdown and JSON reports")
    args = parser.parse_args(argv)
    if args.draws <= 0:
        raise ValueError("--draws must be positive")
    data = load_dataset(args.cached)
    exclusions = parse_exclude_qids(args.exclude_qids)
    before = Counter(record["type"] for record in data)
    mantic = []
    included = []
    for record in data:
        if record.get("platform") == "mantic" or "mantic" in str(record.get("source_tournament") or "").lower():
            mantic.append(record)
        elif str(record["question_id"]) not in exclusions:
            included.append(record)
    ghosts = None
    if args.telemetry_archive:
        archive = Path(args.telemetry_archive)
        if not archive.is_dir():
            raise ValueError(f"Telemetry archive directory does not exist: {archive}")
        ghosts = load_concluding_ghosts(archive)
    included_post_ids = {record["post_id"] for record in included}
    header = {
        "dataset": args.cached,
        "before": dict(before),
        "after": dict(Counter(record["type"] for record in included)),
        "mantic_excluded": len(mantic),
        "cohorts": {
            name: sum(str(record["question_id"]) in ids for record in data)
            for name, ids in EXCLUSION_COHORTS.items()
            if name in args.exclude_qids.split(",")
        },
        "excluded": len(data) - len(included) - len(mantic),
        "era_field": args.era_field,
        "seed": args.seed,
        "draws": args.draws,
        "min_n_for_interval": MIN_N_FOR_INTERVAL,
        "min_clusters_for_interval": MIN_CLUSTERS_FOR_INTERVAL,
        "floor_threshold": FLOOR_THRESHOLD,
        "mc_all": sum(record["type"] == "multiple_choice" for record in included),
        "mc_complete_member_vectors_before": sum(
            record["type"] == "multiple_choice" and _has_complete_member_vector(record) for record in data
        ),
        "mc_complete_member_vectors": sum(
            record["type"] == "multiple_choice" and _has_complete_member_vector(record) for record in included
        ),
        "missing_spot_peer": dict(Counter(record["type"] for record in included if spot_peer_score(record) is None)),
        "mc_fall": sum(
            record["type"] == "multiple_choice" and record.get("triple_subera") == "triple_fall_config"
            for record in included
        ),
        "unjoined_archived_ghosts": None
        if ghosts is None
        else dict(
            Counter(str(ghost["qtype"]) for post_id, ghost in ghosts.items() if post_id not in included_post_ids)
        ),
        "mc_joined_ghost": None
        if ghosts is None
        else sum(record["type"] == "multiple_choice" and record.get("post_id") in ghosts for record in included),
    }
    reports = tuple(
        build_era_report(era, records, ghosts, seed=args.seed, draws=args.draws)
        for era, records in era_buckets(included, args.era_field).items()
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"header": header, "eras": [asdict(report) for report in reports]}
    (output / "loss_attribution.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "loss_attribution.md").write_text(render_report(payload))
    logger.info("Wrote loss attribution reports to %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
