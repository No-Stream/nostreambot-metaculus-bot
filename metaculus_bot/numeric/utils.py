"""Numeric aggregation and helper utilities used by TemplateForecaster.

This module centralises logic for combining numeric forecasts and constructing
user-friendly bound messages so that the core forecaster class stays small.
"""

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
from forecasting_tools import PredictedOptionList
from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    Percentile,
)
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.constants import NUM_RAMP_K_FACTOR, PMF_ABOVE_RANGE_KEY, PMF_BELOW_RANGE_KEY
from metaculus_bot.mc_processing import clamp_and_renormalize_probs
from metaculus_bot.numeric.config import OPEN_TAIL_MIN_MASS, PCHIP_CDF_POINTS, grid_step_constraints
from metaculus_bot.numeric.date_axis import EpochDateQuestion, format_epoch
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid, safe_cdf_bounds
from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution
from metaculus_bot.numeric.pmf_cdf import server_rounded_pmf
from metaculus_bot.numeric.validation import resolve_zero_point

__all__ = [
    "aggregate_binary_mean",
    "aggregate_numeric",
    "bound_messages",
    "clamp_and_renormalize_mc",
    "nominal_bounds",
    "pmf_bound_messages",
]


logger = logging.getLogger(__name__)


def aggregate_binary_mean(predictions: Sequence[float]) -> float:
    """Return the mean of *binary* forecasts rounded to three decimals.

    This matches the old behaviour from `TemplateForecaster`.
    """

    if not predictions:
        raise ValueError("Cannot aggregate empty list of binary predictions")

    mean_prediction = sum(predictions) / len(predictions)
    return round(mean_prediction, 3)


def _pin_endpoints(p_vals: np.ndarray, question: NumericQuestion) -> None:
    """Pin CDF endpoints in-place according to open/closed bound semantics."""
    if question.open_lower_bound:
        p_vals[0] = max(p_vals[0], OPEN_TAIL_MIN_MASS)
    else:
        p_vals[0] = 0.0
    if question.open_upper_bound:
        p_vals[-1] = min(p_vals[-1], 1.0 - OPEN_TAIL_MIN_MASS)
    else:
        p_vals[-1] = 1.0


def _postprocess_ensemble_cdf(
    p_vals: np.ndarray,
    question: NumericQuestion,
    method_label: str,
) -> NumericDistribution:
    """Shared CDF post-processing for both mean and median aggregation.

    Pins the endpoints, enforces monotonicity, ramp-smooths any sub-min-step bin and routes
    the result through ``safe_cdf_bounds`` with the step limits of the grid the CDF is on;
    that last pass is load-bearing, since the ramp (or a raw concentrated median) can leave
    bins over the grid-scaled max step and crash ``Percentile`` validation downstream.
    Nothing is resampled: ``aggregate_numeric`` already aligned every member to the
    question's own grid. The result is labelled with the question's own value axis (the
    same ``zero_point`` the per-model builds resolve), so ``declared_percentiles`` and
    ``get_cdf()`` agree with each other and with every member. ``method_label`` is used
    only in log and marker text. Detail: ``docs/numeric_pipeline.md``, Step 9.
    """
    p_vals = np.clip(p_vals, 0.0, 1.0)
    p_vals = np.maximum.accumulate(p_vals)
    _pin_endpoints(p_vals, question)

    min_step_required, max_step_required = grid_step_constraints(len(p_vals))

    # Compared as the server compares it, so one-ULP noise in a pointwise mean of exact-min-step members cannot ramp the grid.
    diffs_before = server_rounded_pmf(p_vals)
    min_delta_before = float(np.min(diffs_before)) if len(diffs_before) else 1.0
    if min_delta_before < min_step_required:
        ramp = np.linspace(0.0, min_step_required * NUM_RAMP_K_FACTOR, len(p_vals))
        p_vals = np.maximum.accumulate(p_vals + ramp)
        _pin_endpoints(p_vals, question)

        diffs_after = server_rounded_pmf(p_vals)
        min_delta_after = float(np.min(diffs_after)) if len(diffs_after) else 1.0
        logger.warning(
            "Ensemble CDF ramp smoothing (%s) | Q %s | URL %s | min_prob_delta_before=%.8f | min_prob_delta_after=%.8f",
            method_label,
            question.id_of_question,
            question.page_url,
            min_delta_before,
            min_delta_after,
        )

    p_vals = safe_cdf_bounds(
        p_vals,
        open_lower=question.open_lower_bound,
        open_upper=question.open_upper_bound,
        min_step=min_step_required,
        max_step=max_step_required,
        question_id=question.id_of_question,
        model_name=f"ensemble_{method_label}",
    )

    zero_point = resolve_zero_point(question)
    value_grid = build_cdf_value_grid(question.lower_bound, question.upper_bound, zero_point, len(p_vals))
    declared_percentiles = [
        Percentile(percentile=float(p), value=float(v)) for v, p in zip(value_grid, p_vals, strict=True)
    ]
    return create_pchip_numeric_distribution(
        pchip_cdf=list(map(float, p_vals)),
        percentile_list=declared_percentiles,
        question=question,
        zero_point=zero_point,
    )


def _canonical_cdf_length(question: NumericQuestion) -> int:
    """Points the question's submitted CDF has: its ``cdf_size``.

    Mirrors ``build_numeric_distribution``'s target, so per-model CDFs and the
    ensemble CDF live on the same grid by construction. An out-of-range value
    raises LOUDLY rather than silently substituting 201: a ``cdf_size`` below 2 is
    a malformed question, and the substitution would publish a 201-point CDF
    against a grid the platform never declared.
    """
    target = int(question.cdf_size)
    if target < 2:
        raise ValueError(f"NumericQuestion.cdf_size must be >= 2 to define a CDF grid, got {target}")
    return target


def _cdf_heights_on_canonical_grid(
    prediction: NumericDistribution,
    n_points: int,
    question: NumericQuestion,
    model_index: int,
) -> np.ndarray:
    """One model's CDF heights, resampled to ``n_points`` if it arrived on another grid.

    Grid index ``i`` means the same thing in every CDF — Metaculus bucket
    ``i / (n - 1)`` of the question's range — so a length mismatch is resolved by
    interpolating in that shared cdf-location space, NOT in value space: the PCHIP
    grid and forecasting-tools' fallback builder compute the same value axis by
    different formulas, equal in exact arithmetic but not in the last float bits, so
    only the bucket index is shared by construction.
    """
    heights = np.asarray([float(p.percentile) for p in prediction.get_cdf()], dtype=float)
    if heights.size < 2:
        raise ValueError(f"Model {model_index} contributed a {heights.size}-point CDF; cannot aggregate")
    if heights.size == n_points:
        return heights
    logger.warning(
        "NUMERIC_AGGREGATE_GRID_MISMATCH: question=%s model_index=%d got_points=%d expected_points=%d — "
        "resampling in cdf-location space before aggregation",
        question.id_of_question,
        model_index,
        heights.size,
        n_points,
    )
    return np.interp(
        np.linspace(0.0, 1.0, n_points),
        np.linspace(0.0, 1.0, heights.size),
        heights,
    )


def aggregate_numeric(
    predictions: Sequence[NumericDistribution],
    question: NumericQuestion,
    method: str | Literal["mean", "median"] = "mean",
) -> NumericDistribution:
    """Aggregate ``predictions`` by ``method`` (``"mean"`` or ``"median"``), pointwise in CDF space.

    Every model contributes to every grid point, and the aggregation is POSITIONAL (grid
    index ``i`` across all models): grouping on the float ``value`` axis silently medianed
    over a SUBSET, because the PCHIP grid and forecasting-tools' fallback grid agree in exact
    arithmetic but not in the last bits (about 225 distinct x-values for 201 buckets, a
    quarter of them short of ``n`` contributors, nothing recording it). A model that
    genuinely arrives on a different-length grid is resampled first (logged, see
    ``_cdf_heights_on_canonical_grid``). ``question`` supplies the grid and the bound
    flags. Detail: ``docs/numeric_pipeline.md``, Step 9.
    """

    if not predictions:
        raise ValueError("Cannot aggregate empty list of numeric predictions")

    if method not in ("mean", "median"):
        raise ValueError(f"Invalid aggregation method: {method}")

    n_points = _canonical_cdf_length(question)
    heights = np.vstack(
        [
            _cdf_heights_on_canonical_grid(prediction, n_points, question, index)
            for index, prediction in enumerate(predictions)
        ]
    )
    if heights.shape != (len(predictions), n_points):
        raise ValueError(f"Aligned CDF matrix has shape {heights.shape}, expected {(len(predictions), n_points)}")

    p_vals = heights.mean(axis=0) if method == "mean" else np.median(heights, axis=0)
    return _postprocess_ensemble_cdf(p_vals, question, method_label=method)


def nominal_bounds(question: NumericQuestion) -> tuple[float, float]:
    """Return (upper, lower) nominal/displayed bounds; derive from half-step for discrete Qs.

    The half-step branch requires ``cdf_size > 1`` — the API derives cdf_size as
    ``inbound_outcome_count + 1`` so real questions are always >= 2, but a degenerate
    stub would otherwise divide by zero. It is never entered for a date question: the
    epoch adapter (``date_axis.as_epoch_question``) always carries the nominal bounds the
    API declared, because Mantic labels a date bin by its LEFT edge and the centre-aligned
    derivation below would shift both displayed dates by half a day.
    """
    nominal_upper = getattr(question, "nominal_upper_bound", None)
    nominal_lower = getattr(question, "nominal_lower_bound", None)
    cdf_size = question.cdf_size
    if nominal_upper is None and nominal_lower is None and cdf_size > 1 and cdf_size != PCHIP_CDF_POINTS:
        step = (question.upper_bound - question.lower_bound) / (cdf_size - 1)
        nominal_upper = question.upper_bound - step / 2
        nominal_lower = question.lower_bound + step / 2
    upper = nominal_upper if nominal_upper is not None else question.upper_bound
    lower = nominal_lower if nominal_lower is not None else question.lower_bound
    return upper, lower


def _displayed_bounds(question: NumericQuestion) -> tuple[str | float, str | float]:
    """``(upper, lower)`` as a forecaster reads them: the nominal bounds, rendered as dates on the epoch adapter."""
    upper_bound_value, lower_bound_value = nominal_bounds(question)
    if isinstance(question, EpochDateQuestion):
        return (
            format_epoch(upper_bound_value, question.date_granularity),
            format_epoch(lower_bound_value, question.date_granularity),
        )
    return upper_bound_value, lower_bound_value


def bound_messages(question: NumericQuestion, *, stated_tail_fields: bool = False) -> tuple[str, str]:
    """Return upper & lower bound helper messages for numeric prompts.

    For discrete questions, if nominal bounds are missing, derive them using half-step logic.
    On the epoch adapter of a date question the bounds read as dates, not epoch floats.
    """

    upper_bound_number, lower_bound_number = _displayed_bounds(question)

    if question.open_upper_bound:
        upper_bound_message = (
            f"The upper bound is open: {upper_bound_number} is the top of the displayed range, not a hard limit, "
            f"so the outcome can resolve above {upper_bound_number}. Your percentiles are the ONLY way you express "
            f"probability mass, including mass beyond the displayed range. To put N% of your probability above the "
            f"open ceiling, place that fraction of your percentiles above it: if you believe there is a ~75% chance "
            f"the outcome exceeds {upper_bound_number}, then your P50 (median) must be ABOVE {upper_bound_number} and "
            f"only your lower percentiles (P1, P2.5, P5, P10, P20) sit inside or below the range. Put percentiles at "
            f"or above {upper_bound_number} where you actually believe the value lies, even far outside the displayed "
            f"range. Do not pile percentiles at the boundary."
        )
    else:
        upper_bound_message = f"The upper bound is closed: the outcome can not be higher than {upper_bound_number}."

    if question.open_lower_bound:
        lower_bound_message = (
            f"The lower bound is open: {lower_bound_number} is the bottom of the displayed range, not a hard limit, "
            f"so the outcome can resolve below {lower_bound_number}. Your percentiles are the ONLY way you express "
            f"probability mass, including mass beyond the displayed range. To put N% of your probability below the "
            f"open floor, place that fraction of your percentiles below it: if you believe there is a ~75% chance the "
            f"outcome is below {lower_bound_number}, then your P50 (median) must be BELOW {lower_bound_number} and "
            f"only your upper percentiles (P80, P90, P95, P97.5, P99) sit inside or above the range. Put percentiles "
            f"at or below {lower_bound_number} where you actually believe the value lies, even far outside the "
            f"displayed range. Do not pile percentiles at the boundary."
        )
    else:
        lower_bound_message = f"The lower bound is closed: the outcome can not be lower than {lower_bound_number}."
    if stated_tail_fields:
        if question.open_upper_bound:
            upper_bound_message = (
                f"The upper bound is open, so the outcome can resolve above {upper_bound_number}; "
                "make your percentiles consistent with `prob_above_upper_bound`."
            )
        if question.open_lower_bound:
            lower_bound_message = (
                f"The lower bound is open, so the outcome can resolve below {lower_bound_number}; "
                "make your percentiles consistent with `prob_below_lower_bound`."
            )
    return upper_bound_message, lower_bound_message


def pmf_bound_messages(question: NumericQuestion) -> tuple[str, str]:
    """The ``(upper, lower)`` bound messages of a per-bin prompt, where the reserved keys carry out-of-range mass.

    The twin of :func:`bound_messages` for a question elicited per bin (``config.elicit_per_bin``):
    same displayed bounds, same order, but the percentile wording ("your percentiles are the ONLY way
    you express probability mass") would be wrong here, since an open bound's mass is the
    ``above_range`` / ``below_range`` key and a closed bound has no such key at all.
    """
    upper, lower = _displayed_bounds(question)

    if question.open_upper_bound:
        upper_bound_message = (
            f"The upper bound is open: {upper} is the top of the displayed range, not a hard limit. "
            f"`{PMF_ABOVE_RANGE_KEY}` is the probability that the outcome resolves above {upper}; it is scored as "
            f"its own outcome, so give it your honest probability, however large."
        )
    else:
        upper_bound_message = (
            f"The upper bound is closed: the outcome cannot be higher than {upper}, "
            f"and there is no `{PMF_ABOVE_RANGE_KEY}` key."
        )

    if question.open_lower_bound:
        lower_bound_message = (
            f"The lower bound is open: {lower} is the bottom of the displayed range, not a hard limit. "
            f"`{PMF_BELOW_RANGE_KEY}` is the probability that the outcome resolves below {lower}; it is scored as "
            f"its own outcome, so give it your honest probability, however large."
        )
    else:
        lower_bound_message = (
            f"The lower bound is closed: the outcome cannot be lower than {lower}, "
            f"and there is no `{PMF_BELOW_RANGE_KEY}` key."
        )
    return upper_bound_message, lower_bound_message


def clamp_and_renormalize_mc(
    predicted_option_list: PredictedOptionList,
) -> PredictedOptionList:
    """Clamp MC option probabilities into [MC_PROB_MIN, MC_PROB_MAX] and renormalize in-place.

    Delegates the math to ``clamp_and_renormalize_probs`` so the drift-free clamp is a
    single source shared with every pre-construction clamp site. Renormalization can no
    longer push a floored option back below the floor (the ``0.984 + 8x0.002`` case), so
    every option lands in [MC_PROB_MIN, MC_PROB_MAX] with the list summing to 1.0 —
    **provided ``n * MC_PROB_MIN < 1.0``** (n <= 100 at the 0.01 floor). Above that no
    in-bounds sum-1 solution exists and the delegate returns sub-floor values by necessity;
    see ``clamp_and_renormalize_probs`` for the full contract and the ft-validator
    consequences. Returns the same `PredictedOptionList` for convenience.
    """
    clamped = clamp_and_renormalize_probs([option.probability for option in predicted_option_list.predicted_options])
    for option, probability in zip(predicted_option_list.predicted_options, clamped, strict=True):
        option.probability = probability

    return predicted_option_list
