"""Numeric percentile sanitisation and distribution construction helpers."""

from __future__ import annotations

import logging

from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric.bounds_clamping import (
    calculate_bounds_buffer,
    clamp_values_to_bounds,
    log_cluster_spreading_summary,
    log_corrections_summary,
    log_heavy_clamping_diagnostics,
)
from metaculus_bot.numeric.cluster_processing import (
    apply_cluster_spreading,
    apply_jitter_for_duplicates,
    compute_cluster_parameters,
    detect_count_like_pattern,
    ensure_strictly_increasing_bounded,
    is_degenerate_cluster,
)
from metaculus_bot.numeric.config import (
    PCHIP_CDF_POINTS,
    TAIL_WIDEN_K_TAIL,
    TAIL_WIDEN_SPAN_FLOOR_GAMMA,
    TAIL_WIDEN_TAIL_START,
    TAIL_WIDENING_ENABLE,
    grid_step_constraints,
)
from metaculus_bot.numeric.diagnostics import log_pchip_fallback, validate_cdf_construction
from metaculus_bot.numeric.pchip_cdf import (
    StatedTailDiagnostics,
    build_cdf_value_grid,
    generate_pchip_cdf,
    percentiles_to_pchip_format,
)
from metaculus_bot.numeric.pchip_processing import (
    create_fallback_numeric_distribution,
    create_pchip_numeric_distribution,
    generate_pchip_cdf_with_smoothing,
)
from metaculus_bot.numeric.tail_widening import widen_declared_percentiles
from metaculus_bot.numeric.validation import (
    filter_to_standard_percentiles,
    resolve_zero_point,
    sort_by_percentile_level,
    validate_percentile_count_and_values,
)

logger = logging.getLogger(__name__)


def sanitize_percentiles(
    percentile_list: list[Percentile],
    question: NumericQuestion,
    *,
    model_name: str = "",
) -> tuple[list[Percentile], float | None]:
    """Filter, validate, sort, jitter, and optionally widen percentile declarations.

    ``model_name`` only labels the ``NUMERIC_DEGENERATE_DECLARATION`` marker (the
    forecaster whose declaration collapsed). All three production callers pass it —
    ``forecaster_runners`` the forecaster, ``aggregation_pipeline`` and
    ``ablation.run_stacker`` the stacker — so a ``model=unknown`` in the archive means a
    NEW caller forgot to, not that the field is unavailable.
    """

    filtered = filter_to_standard_percentiles(percentile_list)
    validate_percentile_count_and_values(filtered)
    ordered = sort_by_percentile_level(filtered)
    adjusted = _apply_jitter_and_clamp(ordered, question, model_name=model_name)
    widened = _maybe_widen_tails(adjusted, question)
    return widened, resolve_zero_point(question)


def build_numeric_distribution(
    percentile_list: list[Percentile],
    question: NumericQuestion,
    zero_point: float | None,
    *,
    model_name: str = "",
    p_below_lower_bound: float | None = None,
    p_above_upper_bound: float | None = None,
    tail_diagnostics: StatedTailDiagnostics | None = None,
) -> NumericDistribution:
    """Create a numeric distribution, with a legacy fallback when no tail is stated.

    ``model_name`` only labels the ``CDF_MAXSTEP_CLIP`` marker (whose declaration the
    platform's per-bin cap had to clip), the same way ``sanitize_percentiles`` labels
    ``NUMERIC_DEGENERATE_DECLARATION``.
    """

    if question.cdf_size != PCHIP_CDF_POINTS:
        prediction = _build_discrete_distribution(
            percentile_list,
            question,
            zero_point,
            question.cdf_size,
            model_name=model_name,
            p_below_lower_bound=p_below_lower_bound,
            p_above_upper_bound=p_above_upper_bound,
            tail_diagnostics=tail_diagnostics,
        )
        validate_cdf_construction(prediction, question)
        return prediction

    try:
        pchip_cdf, _smoothing_applied, _aggressive = generate_pchip_cdf_with_smoothing(
            percentile_list,
            question,
            zero_point,
            model_name=model_name,
            p_below_lower_bound=p_below_lower_bound,
            p_above_upper_bound=p_above_upper_bound,
            tail_diagnostics=tail_diagnostics,
        )
        prediction = create_pchip_numeric_distribution(pchip_cdf, percentile_list, question, zero_point)
    # Broad on purpose: narrowing this would turn a recoverable build failure into a dropped
    # forecast. A failed build with stated bound probabilities is retried without them, so the
    # member keeps its forecast on the pre-stated-tail path and the marker records the discard.
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except
        if p_below_lower_bound is not None or p_above_upper_bound is not None:
            logger.warning(
                "Stated-tail CDF build failed for Q %s (%s); rebuilding without stated tails",
                getattr(question, "id_of_question", "N/A"),
                exc,
            )
            if tail_diagnostics is not None:
                tail_diagnostics.stated_tails_discarded = True
            return build_numeric_distribution(percentile_list, question, zero_point, model_name=model_name)
        log_pchip_fallback(question, exc)
        prediction = create_fallback_numeric_distribution(percentile_list, question, zero_point, model_name=model_name)

    validate_cdf_construction(prediction, question)

    return prediction


def _build_discrete_distribution(
    percentile_list: list[Percentile],
    question: NumericQuestion,
    zero_point: float | None,
    target_cdf_size: int,
    *,
    model_name: str = "",
    p_below_lower_bound: float | None = None,
    p_above_upper_bound: float | None = None,
    tail_diagnostics: StatedTailDiagnostics | None = None,
) -> NumericDistribution:
    """Build a PCHIP distribution directly on the question's own non-201 grid.

    Built ONCE on the grid that publishes: a provisional 201-point build would apply
    the 201-grid 0.2 cap and emit a phantom ``CDF_MAXSTEP_CLIP`` for a forecast whose
    published coarse grid has a looser cap and never clipped. A build failure here
    escapes (no ft fallback), exactly as the post-provisional resample did before.
    ``declared_percentiles`` is overwritten with the CDF on the question's value axis
    (geometric on a ``zero_point`` question), the same axis ``get_cdf()`` reports.
    """
    min_step, max_step = grid_step_constraints(target_cdf_size)
    pchip_percentiles = percentiles_to_pchip_format(percentile_list)
    resampled_cdf, _ = generate_pchip_cdf(
        percentile_values=pchip_percentiles,
        open_upper_bound=question.open_upper_bound,
        open_lower_bound=question.open_lower_bound,
        upper_bound=question.upper_bound,
        lower_bound=question.lower_bound,
        zero_point=zero_point,
        min_step=min_step,
        max_step=max_step,
        num_points=target_cdf_size,
        question_id=question.id_of_question,
        question_url=question.page_url,
        model_name=model_name,
        p_below_lower_bound=p_below_lower_bound,
        p_above_upper_bound=p_above_upper_bound,
        tail_diagnostics=tail_diagnostics,
    )
    value_grid = build_cdf_value_grid(question.lower_bound, question.upper_bound, zero_point, target_cdf_size)
    declared_percentiles = [
        Percentile(percentile=float(p), value=float(v)) for v, p in zip(value_grid, resampled_cdf, strict=True)
    ]
    prediction = create_pchip_numeric_distribution(
        pchip_cdf=list(map(float, resampled_cdf)),
        percentile_list=declared_percentiles,
        question=question,
        zero_point=zero_point,
    )
    logger.info(
        "Discrete build in build_numeric_distribution | Q %s | built directly on the %d-point grid",
        question.id_of_question,
        target_cdf_size,
    )
    return prediction


def _apply_jitter_and_clamp(
    percentile_list: list[Percentile],
    question: NumericQuestion,
    *,
    model_name: str = "",
) -> list[Percentile]:
    range_size = question.upper_bound - question.lower_bound
    buffer = calculate_bounds_buffer(question)

    values = [p.value for p in percentile_list]
    modified_values = list(values)

    count_like = detect_count_like_pattern(values)
    span = (max(values) - min(values)) if values else 0.0
    value_eps, _base_delta, spread_delta = compute_cluster_parameters(range_size, count_like, span)

    modified_values, clusters_applied = apply_cluster_spreading(
        modified_values,
        question,
        value_eps=value_eps,
        spread_delta=spread_delta,
        range_size=range_size,
    )

    if is_degenerate_cluster(values, value_eps):
        # A point mass. Withheld on the continuous grid, spread inside its bin and published
        # where the bins are the outcome space: ``apply_cluster_spreading`` has the why.
        logger.warning(
            "NUMERIC_DEGENERATE_DECLARATION: question=%s model=%s n_unique=%d span=%.6g value_eps=%.6g "
            "spread_applied=%s",
            question.id_of_question,
            model_name or "unknown",
            len({float(v) for v in values}),
            span,
            value_eps,
            "true" if clusters_applied > 0 else "false",
        )

    modified_values = apply_jitter_for_duplicates(modified_values, question, range_size, percentile_list)
    modified_values, corrections_made = clamp_values_to_bounds(modified_values, percentile_list, question, buffer)

    log_cluster_spreading_summary(
        modified_values,
        values,
        question,
        clusters_applied=clusters_applied,
        spread_delta=spread_delta,
        count_like=count_like,
    )
    log_corrections_summary(modified_values, values, question, corrections_made)
    log_heavy_clamping_diagnostics(modified_values, values, question)

    modified_values = ensure_strictly_increasing_bounded(modified_values, question, range_size)

    return [Percentile(value=v, percentile=p.percentile) for v, p in zip(modified_values, percentile_list, strict=True)]


def _maybe_widen_tails(percentile_list: list[Percentile], question: NumericQuestion) -> list[Percentile]:
    if not TAIL_WIDENING_ENABLE:
        return percentile_list
    return widen_declared_percentiles(
        percentile_list,
        question,
        k_tail=TAIL_WIDEN_K_TAIL,
        tail_start=TAIL_WIDEN_TAIL_START,
        span_floor_gamma=TAIL_WIDEN_SPAN_FLOOR_GAMMA,
    )


__all__ = ["build_numeric_distribution", "sanitize_percentiles"]
