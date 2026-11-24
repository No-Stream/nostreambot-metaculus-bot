"""Replaying one ensemble member, or a subset of them, against an archived record.

``audit`` scores each member of a record on its own and ranks them. Two things a residual
round's per-model dimension needs on top, neither of which existed in the package:

* a SUBSET aggregate — the median or mean of an arbitrary set of members, recombined the way
  production recombines them and log-scored, which is what a leave-one-out or
  member-versus-ensemble counterfactual is made of. ``ensemble_analysis.EnsembleSimulator``
  answers the same question from live ``BenchmarkForBot`` objects; this reads the archived
  performance records, where the members survive only as declared values.
* CENSORING detection — whether a member's numeric log score is pinned at the grid's own
  constraint floor and so carries no information about how far off the forecast was.

Rebuilding one member's CDF on a record's own grid lives here too, since ``audit``'s
per-question ranking needs the identical build and used to carry its own copy.

A member's forecast arrives as the archive holds it, untyped: a probability for binary, an
option-to-probability map for multiple choice, a declared ``(percentile, value)`` curve for
numeric and discrete. The subset aggregate is close to but not byte-identical with what
production published. Both, with receipts: ``docs/performance_analysis.md`` "Replaying a
member subset".
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from forecasting_tools import NumericQuestion, Percentile, PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

from metaculus_bot.aggregation_strategies import (
    aggregate_binary_median,
    aggregate_multiple_choice_mean,
    aggregate_multiple_choice_median,
)
from metaculus_bot.numeric.config import OPEN_TAIL_MIN_MASS, PCHIP_CDF_POINTS, grid_step_constraints
from metaculus_bot.numeric.pchip_cdf import generate_pchip_cdf, safe_cdf_bounds
from metaculus_bot.numeric.pipeline import build_numeric_distribution
from metaculus_bot.numeric.utils import aggregate_binary_mean
from metaculus_bot.performance_analysis.collector import resolve_numeric_record_to_score_inputs
from metaculus_bot.performance_analysis.declared_value_recovery import _parse_block_in_body
from metaculus_bot.performance_analysis.parsing import declared_anchors, parse_per_model_reasoning_text
from metaculus_bot.scoring_common import (
    BOUNDARY_BASELINE,
    binary_log_score,
    mc_log_score,
    numeric_log_score,
    resolution_to_bucket_index,
)
from metaculus_bot.structured_output_schema import NumericStructured
from metaculus_bot.value_extraction import _tail_probabilities_from_block

NUMERIC_RECORD_TYPES: tuple[str, ...] = ("numeric", "discrete")

# Observed censored ratios cluster at 1.10-1.13, not on the bare minimum (docs).
CENSOR_RATIO_TOL: float = 1.5

AGGREGATION_METHODS: tuple[str, ...] = ("median", "mean")

__all__ = [
    "AGGREGATION_METHODS",
    "CENSOR_RATIO_TOL",
    "NUMERIC_RECORD_TYPES",
    "MemberScoringInputs",
    "censoring_ratio",
    "is_censored",
    "is_replayable",
    "mc_median_probs",
    "member_cdf",
    "member_scoring_inputs",
    "postprocess_replay_cdf",
    "record_cdf_size",
    "replay_solo_score",
    "replay_subset_score",
    "score_member_cdf",
    "theoretical_floor_score",
]


def record_cdf_size(record: dict) -> int:
    """The question's own CDF grid size: ``inbound_outcome_count + 1``, else the published
    CDF's length, else the standard 201. Discrete questions run coarser grids, and a
    per-member CDF rebuilt on the 201 default is scored against a 1/200 baseline instead of
    the question's own (up to 85 log pts on an 11-point grid, always inflating good
    coarse-grid scores)."""
    inbound = (record.get("scaling") or {}).get("inbound_outcome_count")
    if inbound:
        return int(inbound) + 1
    published = record.get("our_forecast_values") or []
    return len(published) if len(published) >= 3 else PCHIP_CDF_POINTS


@dataclass(frozen=True, slots=True)
class MemberScoringInputs:
    """Everything a member curve needs to be rebuilt and log-scored on one record."""

    resolution: float
    lower_bound: float
    upper_bound: float
    zero_point: float | None
    open_lower: bool
    open_upper: bool
    cdf_size: int
    min_step: float
    max_step: float


def member_scoring_inputs(record: dict, *, n_points: int | None = None) -> MemberScoringInputs | None:
    """Bundle the record's bounds, grid size and per-bin step constraints, or None.

    ``n_points`` overrides the record's own grid, which is how a round compares a rebuild on
    the question's real grid against one on the 201-point default.
    """
    score_inputs = resolve_numeric_record_to_score_inputs(record)
    if score_inputs is None:
        return None
    resolution, lower_bound, upper_bound, zero_point = score_inputs
    cdf_size = record_cdf_size(record) if n_points is None else n_points
    min_step, max_step = grid_step_constraints(cdf_size)
    return MemberScoringInputs(
        resolution=resolution,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        zero_point=zero_point,
        open_lower=bool(record.get("open_lower_bound", False)),
        open_upper=bool(record.get("open_upper_bound", False)),
        cdf_size=cdf_size,
        min_step=min_step,
        max_step=max_step,
    )


def member_reasoning(record: dict, model: str | None) -> str:
    """Recover this member's original fenced declaration from its comment."""
    if model is None:
        return ""
    return parse_per_model_reasoning_text(record.get("comment_text") or "").get(model, "")


def member_cdf(
    percentile_pairs: Sequence[Sequence[float]], inputs: MemberScoringInputs, *, reasoning: str = ""
) -> list[float]:
    """One member's declared curve as a CDF on ``inputs.cdf_size`` points.

    Raises ``ValueError`` / ``RuntimeError`` from the PCHIP build, which each caller reports
    in its own vocabulary: a dossier warns and drops the row, a replay drops the member.
    """
    anchors, _conflicts = declared_anchors(percentile_pairs)
    block = _parse_block_in_body(reasoning, "numeric") if reasoning else None
    below = above = None
    if isinstance(block, NumericStructured):
        tails = _tail_probabilities_from_block(
            block, open_lower_bound=inputs.open_lower, open_upper_bound=inputs.open_upper
        )
        below = tails.below.value if inputs.open_lower and tails.below.status == "valid" else None
        above = tails.above.value if inputs.open_upper and tails.above.status == "valid" else None
    if below is not None or above is not None:
        question = NumericQuestion(
            question_text="Member replay",
            lower_bound=inputs.lower_bound,
            upper_bound=inputs.upper_bound,
            open_lower_bound=inputs.open_lower,
            open_upper_bound=inputs.open_upper,
            cdf_size=inputs.cdf_size,
        )
        percentiles = [Percentile(percentile=level / 100.0, value=value) for level, value in anchors.items()]
        distribution = build_numeric_distribution(
            percentiles,
            question,
            inputs.zero_point,
            p_below_lower_bound=below,
            p_above_upper_bound=above,
        )
        return [point.percentile for point in distribution.get_cdf()]
    cdf, _diagnostics = generate_pchip_cdf(
        anchors,
        open_upper_bound=inputs.open_upper,
        open_lower_bound=inputs.open_lower,
        upper_bound=inputs.upper_bound,
        lower_bound=inputs.lower_bound,
        zero_point=inputs.zero_point,
        min_step=inputs.min_step,
        max_step=inputs.max_step,
        num_points=inputs.cdf_size,
        p_below_lower_bound=below,
        p_above_upper_bound=above,
    )
    return cdf


def score_member_cdf(cdf: Sequence[float] | np.ndarray, inputs: MemberScoringInputs) -> float:
    """Metaculus log score of a rebuilt CDF against the record's resolution."""
    return numeric_log_score(
        list(cdf),
        inputs.resolution,
        inputs.lower_bound,
        inputs.upper_bound,
        open_lower_bound=inputs.open_lower,
        open_upper_bound=inputs.open_upper,
        zero_point=inputs.zero_point,
    )


def postprocess_replay_cdf(aggregate: np.ndarray, inputs: MemberScoringInputs) -> np.ndarray:
    """Legalize a pointwise CDF aggregate the way the publish path legalizes one.

    Clamp, monotonize, pin the tails to the open-bound minimum, then ``safe_cdf_bounds`` with
    the grid's own step limits. Production's ``_postprocess_ensemble_cdf`` additionally
    ramp-tilts a CDF whose thinnest bin is under the min step before that call, so the two
    can differ on a record where the ramp fired (docs, "Replaying a member subset").
    """
    heights = np.maximum.accumulate(np.clip(np.asarray(aggregate, dtype=float), 0.0, 1.0))
    heights[0] = max(heights[0], OPEN_TAIL_MIN_MASS) if inputs.open_lower else 0.0
    heights[-1] = min(heights[-1], 1.0 - OPEN_TAIL_MIN_MASS) if inputs.open_upper else 1.0
    return safe_cdf_bounds(
        heights,
        open_lower=inputs.open_lower,
        open_upper=inputs.open_upper,
        min_step=inputs.min_step,
        max_step=inputs.max_step,
    )


def theoretical_floor_score(record: dict, *, n_points: int | None = None) -> float | None:
    """Lowest score the SERVER's constraints permit on this record's grid.

    Context only, never the censoring detector: the pipeline's ramp and repair passes never
    land exactly on the theoretical minimum, so thresholding on this value detects nothing
    (docs, "Detecting a censored member score"). None when the resolution fell outside a
    CLOSED bound, where no legal forecast could have put mass there.
    """
    inputs = member_scoring_inputs(record, n_points=n_points)
    if inputs is None:
        return None
    n_inbound = inputs.cdf_size - 1
    bucket = _resolution_bucket(inputs, n_inbound)
    if bucket == 0:
        return 50.0 * math.log(OPEN_TAIL_MIN_MASS / BOUNDARY_BASELINE) if inputs.open_lower else None
    if bucket == n_inbound + 1:
        return 50.0 * math.log(OPEN_TAIL_MIN_MASS / BOUNDARY_BASELINE) if inputs.open_upper else None
    n_open = int(inputs.open_lower) + int(inputs.open_upper)
    baseline = (1.0 - BOUNDARY_BASELINE * n_open) / n_inbound
    return 50.0 * math.log(inputs.min_step / baseline)


def censoring_ratio(cdf: Sequence[float] | np.ndarray, inputs: MemberScoringInputs) -> float | None:
    """Mass this CDF put on the resolution bin, over the smallest mass it was ALLOWED to put there.

    A ratio near 1 means the member is pinned at the floor, so its log score is censored.
    None when the bin has no positive floor: a closed bound pins the boundary mass to 0, and
    there is nothing to be censored against.
    """
    heights = np.asarray(cdf, dtype=float)
    n_inbound = len(heights) - 1
    bucket = _resolution_bucket(inputs, n_inbound)
    if bucket == 0:
        mass, floor = float(heights[0]), OPEN_TAIL_MIN_MASS if inputs.open_lower else 0.0
    elif bucket == n_inbound + 1:
        mass, floor = float(1.0 - heights[-1]), OPEN_TAIL_MIN_MASS if inputs.open_upper else 0.0
    else:
        mass, floor = float(np.diff(heights)[bucket - 1]), grid_step_constraints(len(heights))[0]
    return None if floor <= 0 else mass / floor


def is_censored(
    cdf: Sequence[float] | np.ndarray, inputs: MemberScoringInputs, *, tol: float = CENSOR_RATIO_TOL
) -> bool | None:
    """True when this CDF's resolution-bin mass sits at the GRID's constraint floor.

    Compared against the grid's floor rather than the member's own thinnest step, which looks
    equivalent and is not (docs, "Detecting a censored member score").
    """
    ratio = censoring_ratio(cdf, inputs)
    return None if ratio is None else bool(ratio <= tol)


def mc_median_probs(record: dict, members: Mapping[str, Any], models: Sequence[str]) -> list[float] | None:
    """The published multiple-choice median over ``models``, positionally aligned to the record's options."""
    return _mc_aggregate_probs(record, members, models, method="median")


def replay_subset_score(
    record: dict,
    members: Mapping[str, Any],
    models: Sequence[str],
    *,
    method: str,
    n_points: int | None = None,
) -> float | None:
    """Metaculus log score of the aggregate of ``models`` alone.

    ``method`` is ``"median"`` (what production publishes) or ``"mean"``. ``n_points``
    overrides the numeric grid. None when the subset is empty or nothing in it could be
    rebuilt and scored.
    """
    if method not in AGGREGATION_METHODS:
        raise ValueError(f"unrecognized aggregation {method=}; expected one of {AGGREGATION_METHODS}")
    if not models:
        return None
    q_type = record["type"]
    if q_type == "binary":
        values = [float(members[model]) for model in models]
        aggregate = aggregate_binary_median(values) if method == "median" else aggregate_binary_mean(values)
        return binary_log_score(aggregate, record["resolution_parsed"])
    if q_type == "multiple_choice":
        options: list[str] = record["options"]
        probs = _mc_aggregate_probs(record, members, models, method=method)
        return None if probs is None else mc_log_score(probs, options.index(record["resolution_parsed"]))
    return _replay_numeric(record, members, models, method=method, n_points=n_points)


def replay_solo_score(
    record: dict, value: Any, *, n_points: int | None = None, model: str | None = None
) -> float | None:
    """Metaculus log score of one member's own forecast, with no aggregation applied.

    A member's declared distribution is scored as declared: production's clamps belong to the
    aggregate, so applying them here would score a ballot nobody submitted. None when a
    multiple-choice ballot carries no mass, or a numeric curve cannot be rebuilt or scored.
    """
    q_type = record["type"]
    if q_type == "binary":
        return binary_log_score(float(value), record["resolution_parsed"])
    if q_type == "multiple_choice":
        options: list[str] = record["options"]
        total = sum(value[option] for option in options)
        if total <= 0:
            return None
        return mc_log_score([value[option] / total for option in options], options.index(record["resolution_parsed"]))
    inputs = member_scoring_inputs(record, n_points=n_points)
    if inputs is None:
        return None
    try:
        return score_member_cdf(member_cdf(value, inputs, reasoning=member_reasoning(record, model)), inputs)
    except (ValueError, RuntimeError, ZeroDivisionError):
        return None


def is_replayable(record: dict) -> bool:
    """True when the record carries a resolution this module can score a forecast against."""
    q_type = record["type"]
    if q_type == "binary":
        return isinstance(record.get("resolution_parsed"), bool)
    if q_type == "multiple_choice":
        options = record.get("options") or []
        return isinstance(record.get("resolution_parsed"), str) and record["resolution_parsed"] in options
    if q_type in NUMERIC_RECORD_TYPES:
        return resolve_numeric_record_to_score_inputs(record) is not None
    return False


def _resolution_bucket(inputs: MemberScoringInputs, n_inbound: int) -> int:
    return resolution_to_bucket_index(
        inputs.resolution,
        inputs.lower_bound,
        inputs.upper_bound,
        n_inbound=n_inbound,
        zero_point=inputs.zero_point,
    )


def _mc_aggregate_probs(
    record: dict, members: Mapping[str, Any], models: Sequence[str], *, method: str
) -> list[float] | None:
    options: list[str] = record["options"]
    ballots = [
        PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=option, probability=members[model][option]) for option in options
            ]
        )
        for model in models
    ]
    if not ballots:
        return None
    aggregate = (
        aggregate_multiple_choice_median(ballots) if method == "median" else aggregate_multiple_choice_mean(ballots)
    )
    by_option = {option.option_name: option.probability for option in aggregate.predicted_options}
    return [by_option[option] for option in options]


def _replay_numeric(
    record: dict,
    members: Mapping[str, Any],
    models: Sequence[str],
    *,
    method: str,
    n_points: int | None,
) -> float | None:
    inputs = member_scoring_inputs(record, n_points=n_points)
    if inputs is None:
        return None
    cdfs: list[list[float]] = []
    for model in models:
        try:
            cdfs.append(member_cdf(members[model], inputs, reasoning=member_reasoning(record, model)))
        except (ValueError, RuntimeError):
            continue
    if not cdfs:
        return None
    stacked = np.vstack(cdfs)
    aggregate = np.median(stacked, axis=0) if method == "median" else stacked.mean(axis=0)
    try:
        return score_member_cdf(postprocess_replay_cdf(aggregate, inputs), inputs)
    except (ValueError, ZeroDivisionError):
        return None
