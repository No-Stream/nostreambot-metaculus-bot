"""PCHIP interpolation processing for numeric CDFs."""

from __future__ import annotations

import logging
from itertools import pairwise

import numpy as np
from forecasting_tools import NumericDistribution
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.constants import NUM_MAX_STEP, NUM_MIN_PROB_STEP, NUM_RAMP_K_FACTOR
from metaculus_bot.numeric.config import (
    OPEN_TAIL_MIN_MASS,
    PCHIP_CDF_POINTS,
    grid_step_constraints,
)
from metaculus_bot.numeric.date_axis import EpochDateQuestion
from metaculus_bot.numeric.pchip_cdf import (
    StatedTailDiagnostics,
    build_cdf_value_grid,
    enforce_min_steps,
    safe_cdf_bounds,
)

logger = logging.getLogger(__name__)


# Module-level counters for PCHIP enforcement statistics (per run)
_pchip_stats: dict[str, int] = {
    "total_attempts": 0,
    "successful_without_enforcement": 0,
    "required_aggressive_enforcement": 0,
    "failed_entirely": 0,
}


def reset_pchip_stats() -> None:
    """Reset PCHIP statistics counters (call at start of each run)."""
    # Deliberate module-global run counter (same pattern as the fallback counters in
    # fallback_openrouter). The reset REBINDS rather than mutates, so `global` is required;
    # readers get a copy from get_pchip_stats(), so nothing holds the old dict.
    global _pchip_stats  # noqa: PLW0603
    _pchip_stats = {
        "total_attempts": 0,
        "successful_without_enforcement": 0,
        "required_aggressive_enforcement": 0,
        "failed_entirely": 0,
    }


def get_pchip_stats() -> dict[str, int]:
    """Get current PCHIP statistics."""
    return _pchip_stats.copy()


def log_pchip_summary() -> None:
    """Log comprehensive PCHIP enforcement statistics."""
    stats = get_pchip_stats()
    if stats["total_attempts"] == 0:
        logger.info("PCHIP Summary: No PCHIP attempts in this run")
        return

    success_rate = 100.0 * stats["successful_without_enforcement"] / stats["total_attempts"]
    enforcement_rate = 100.0 * stats["required_aggressive_enforcement"] / stats["total_attempts"]
    failure_rate = 100.0 * stats["failed_entirely"] / stats["total_attempts"]

    logger.info(
        "PCHIP Summary | total_attempts=%d | successful_without_enforcement=%d (%.1f%%) | required_aggressive_enforcement=%d (%.1f%%) | failed_entirely=%d (%.1f%%)",
        stats["total_attempts"],
        stats["successful_without_enforcement"],
        success_rate,
        stats["required_aggressive_enforcement"],
        enforcement_rate,
        stats["failed_entirely"],
        failure_rate,
    )


def generate_pchip_cdf_with_smoothing(
    percentile_list: list[Percentile],
    question: NumericQuestion,
    zero_point: float | None,
    *,
    model_name: str = "",
    p_below_lower_bound: float | None = None,
    p_above_upper_bound: float | None = None,
    tail_diagnostics: StatedTailDiagnostics | None = None,
) -> tuple[list[float], bool, bool]:
    """Generate PCHIP CDF with optional ramp smoothing.

    ``model_name`` only labels the ``CDF_MAXSTEP_CLIP`` marker (see ``safe_cdf_bounds``).
    """
    from metaculus_bot.numeric.pchip_cdf import (  # noqa: PLC0415  # function-scoped: call-time lookup keeps tests patching metaculus_bot.numeric.pchip_cdf.* effective
        generate_pchip_cdf,
        percentiles_to_pchip_format,
    )

    _pchip_stats["total_attempts"] += 1

    pchip_percentiles = percentiles_to_pchip_format(percentile_list)

    try:
        pchip_cdf, aggressive_enforcement_used = generate_pchip_cdf(
            percentile_values=pchip_percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=zero_point,
            min_step=NUM_MIN_PROB_STEP,
            num_points=PCHIP_CDF_POINTS,
            question_id=question.id_of_question,
            question_url=question.page_url,
            model_name=model_name,
            p_below_lower_bound=p_below_lower_bound,
            p_above_upper_bound=p_above_upper_bound,
            tail_diagnostics=tail_diagnostics,
        )

        if aggressive_enforcement_used:
            _pchip_stats["required_aggressive_enforcement"] += 1
        else:
            _pchip_stats["successful_without_enforcement"] += 1

    except (ValueError, RuntimeError):
        _pchip_stats["failed_entirely"] += 1
        raise

    fixed_endpoints = (
        (pchip_cdf[0], pchip_cdf[-1]) if p_below_lower_bound is not None or p_above_upper_bound is not None else None
    )
    smoothing_applied = _apply_ramp_smoothing(
        pchip_cdf, question, model_name=model_name, fixed_endpoints=fixed_endpoints
    )

    _validate_pchip_cdf(pchip_cdf, question)
    _log_pchip_success(pchip_cdf, question, smoothing_applied)

    return pchip_cdf, smoothing_applied, aggressive_enforcement_used


def _apply_ramp_smoothing(
    pchip_cdf: list[float],
    question: NumericQuestion,
    *,
    model_name: str = "",
    fixed_endpoints: tuple[float, float] | None = None,
) -> bool:
    """Apply ramp smoothing to enforce minimum step size."""
    diffs_before = np.diff(pchip_cdf)
    min_delta_before = float(np.min(diffs_before)) if len(diffs_before) else 1.0

    if min_delta_before < NUM_MIN_PROB_STEP:
        ramp = np.linspace(0.0, NUM_MIN_PROB_STEP * NUM_RAMP_K_FACTOR, len(pchip_cdf))
        smoothed = np.maximum.accumulate(np.array(pchip_cdf) + ramp)

        # Re-pin endpoints to respect open/closed bounds semantics
        if not question.open_lower_bound:
            smoothed[0] = 0.0
        else:
            smoothed[0] = max(smoothed[0], OPEN_TAIL_MIN_MASS)
        if not question.open_upper_bound:
            smoothed[-1] = 1.0
        else:
            smoothed[-1] = min(smoothed[-1], 1.0 - OPEN_TAIL_MIN_MASS)

        if fixed_endpoints is not None:
            lower_probability, upper_probability = fixed_endpoints
            smoothed[0], smoothed[-1] = fixed_endpoints
            smoothed = enforce_min_steps(
                smoothed,
                NUM_MIN_PROB_STEP,
                lower_cap=lower_probability,
                upper_cap=upper_probability,
            )

        # Enforce max-step constraint post-smoothing
        smoothed = safe_cdf_bounds(
            smoothed,
            open_lower=question.open_lower_bound,
            open_upper=question.open_upper_bound,
            question_id=question.id_of_question,
            model_name=model_name,
        )
        pchip_cdf[:] = smoothed.tolist()

        diffs_after = np.diff(pchip_cdf)
        min_delta_after = float(np.min(diffs_after)) if len(diffs_after) else 1.0
        logger.warning(
            "CDF ramp smoothing for Q %s | URL %s | min_prob_delta_before=%.8f | min_prob_delta_after=%.8f | k_factor=%.1f",
            question.id_of_question,
            question.page_url,
            min_delta_before,
            min_delta_after,
            NUM_RAMP_K_FACTOR,
        )
        return True

    return False


def _validate_pchip_cdf(pchip_cdf: list[float], question: NumericQuestion) -> None:
    """Validate PCHIP CDF meets all requirements."""
    if len(pchip_cdf) != PCHIP_CDF_POINTS:
        raise ValueError(f"PCHIP CDF has {len(pchip_cdf)} points, expected {PCHIP_CDF_POINTS}")

    if not all(0.0 <= p <= 1.0 for p in pchip_cdf):
        invalid_probs = [p for p in pchip_cdf if not (0.0 <= p <= 1.0)]
        raise ValueError(f"PCHIP CDF contains invalid probabilities outside [0,1]: {invalid_probs}")

    if not all(a <= b for a, b in pairwise(pchip_cdf)):
        raise ValueError("PCHIP CDF is not monotonic")

    min_step = np.min(np.diff(pchip_cdf))
    if min_step < NUM_MIN_PROB_STEP - 1e-10:
        raise ValueError(f"PCHIP CDF violates minimum step requirement: {min_step:.8f} < {NUM_MIN_PROB_STEP:.8f}")

    max_step = np.max(np.diff(pchip_cdf))
    if max_step > NUM_MAX_STEP + 1e-6:
        raise ValueError(f"PCHIP CDF violates maximum step requirement: {max_step:.8f} > {NUM_MAX_STEP:.8f}")

    if not question.open_lower_bound and abs(pchip_cdf[0]) > 1e-6:
        raise ValueError(f"PCHIP CDF closed lower bound violation: {pchip_cdf[0]} != 0.0")

    if not question.open_upper_bound and abs(pchip_cdf[-1] - 1.0) > 1e-6:
        raise ValueError(f"PCHIP CDF closed upper bound violation: {pchip_cdf[-1]} != 1.0")

    if question.open_lower_bound and pchip_cdf[0] < OPEN_TAIL_MIN_MASS:
        raise ValueError(f"PCHIP CDF open lower bound violation: {pchip_cdf[0]} < {OPEN_TAIL_MIN_MASS}")

    if question.open_upper_bound and pchip_cdf[-1] > 1.0 - OPEN_TAIL_MIN_MASS:
        raise ValueError(f"PCHIP CDF open upper bound violation: {pchip_cdf[-1]} > {1.0 - OPEN_TAIL_MIN_MASS}")


def _log_pchip_success(pchip_cdf: list[float], question: NumericQuestion, smoothing_applied: bool) -> None:
    """Log successful PCHIP CDF generation."""
    min_step = np.min(np.diff(pchip_cdf))
    max_step = np.max(np.diff(pchip_cdf))

    logger.info(
        "PCHIP OK for Q %s | points=%d | min_step=%.8f | max_step=%.8f | smoothing=%s | open_bounds=(%s,%s)",
        question.id_of_question,
        len(pchip_cdf),
        min_step,
        max_step,
        smoothing_applied,
        question.open_lower_bound,
        question.open_upper_bound,
    )


def create_pchip_numeric_distribution(
    pchip_cdf: list[float],
    percentile_list: list[Percentile],
    question: NumericQuestion,
    zero_point: float | None,
) -> NumericDistribution:
    """Create a custom NumericDistribution that uses PCHIP CDF.

    ``is_date`` is set from the question: on the epoch adapter of a date question it is what
    makes forecasting-tools' ``make_readable_prediction`` render the published comment's values
    as UTC dates instead of epoch floats. Nothing else reads it.
    """

    class PchipNumericDistribution(NumericDistribution):
        def __init__(self, pchip_cdf_values, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pchip_cdf_values = pchip_cdf_values

        def get_cdf(self) -> list[Percentile]:
            """Return the pre-computed PCHIP CDF as Percentile objects.

            forecasting-tools 0.2.92's publish and aggregate paths read
            ``get_cdf()`` (``.cdf`` is a deprecated shim on the base class), so
            we override the *method* — not just the property — to guarantee our
            PCHIP output is what gets submitted, never the base-class builder's.
            """
            # Probability heights were evaluated on this canonical question grid.
            question_values = build_cdf_value_grid(
                self.lower_bound,
                self.upper_bound,
                self.zero_point,
                len(self._pchip_cdf_values),
            )
            return [
                Percentile(percentile=prob_val, value=question_val)
                for question_val, prob_val in zip(question_values, self._pchip_cdf_values, strict=True)
            ]

        @property
        def cdf(self) -> list[Percentile]:
            """Deprecated alias for :meth:`get_cdf` (returns identical data)."""
            return self.get_cdf()

    return PchipNumericDistribution(
        pchip_cdf_values=pchip_cdf,
        declared_percentiles=percentile_list,
        open_upper_bound=question.open_upper_bound,
        open_lower_bound=question.open_lower_bound,
        upper_bound=question.upper_bound,
        lower_bound=question.lower_bound,
        zero_point=zero_point,
        cdf_size=question.cdf_size,
        is_date=isinstance(question, EpochDateQuestion),
        # Our CDF is already the final, min/max-step- and bound-enforced submission,
        # exposed via the get_cdf() override above. strict_validation=False stops the
        # 0.2.92 validators from (a) rejecting our beyond-open-bound percentile
        # convention (_check_too_far_from_bounds) and (b) mutating declared_percentiles
        # (_check_and_update_repeating_values) that diagnostics + spread metrics read
        # verbatim; standardize_cdf=False keeps the base machinery from ever
        # re-standardizing a distribution we already finalized.
        strict_validation=False,
        standardize_cdf=False,
    )


def create_fallback_numeric_distribution(
    percentile_list: list[Percentile],
    question: NumericQuestion,
    zero_point: float | None,
    *,
    model_name: str = "",
) -> NumericDistribution:
    """Create fallback NumericDistribution when PCHIP fails.

    Wraps forecasting-tools' native CDF builder (``get_cdf()``) but runs its output
    through ``safe_cdf_bounds`` whatever the bound shape. Metaculus rejects open-bound
    CDFs with ``cdf[0] < OPEN_TAIL_MIN_MASS`` / ``cdf[-1] > 1 - OPEN_TAIL_MIN_MASS`` and caps the per-bin step on
    every question, so the legal range and the grid's step limits are enforced here
    rather than trusting the raw builder output. The closed/closed case used to be
    handed back raw: a tight declaration (normal, sd 0.6 on a [0, 100] grid) produced
    a 0.297 bin against the server's 0.2 cap and an HTTP 400 on the lone-survivor
    path, on the modal Mantic date shape. ``safe_cdf_bounds`` pins nothing on a
    closed bound and ``enforce_min_steps`` caps at [0.0, 1.0], so the closed
    endpoints survive it exactly.

    ``standardize_cdf=False`` keeps ``get_cdf()`` on the non-standardizing raw
    linear-interpolation path (the 0.2.54 behavior this fallback was written
    against); the endpoint/step enforcement is our own ``safe_cdf_bounds`` pass,
    not upstream's ``_standardize_cdf``.

    KNOWN AVAILABILITY GAP (2026-08-25 sentinel audit, unfixed): on a log-scaled
    (``zero_point``) question, upstream's ``get_cdf()`` can itself raise on a
    float-epsilon overshoot of 1.0 — so the fallback is unavailable on exactly the
    question shape where PCHIP is most likely to have failed in the first place. That
    fails FAST (the forecaster is dropped and attributed), which is why it was left: a
    drop is not a fabricated distribution. Fixing it means reaching into upstream's
    builder, which is its own change.
    """

    # create_fallback_numeric_distribution builds exactly ONE instance, so a closure
    # cell is a safe per-distribution cache. Memoizing keeps CDF_MAXSTEP_CLIP at one
    # marker per BUILD rather than one per accessor read (validate + diagnostics +
    # aggregate + publish all call get_cdf()), matching the PCHIP sibling's count.
    cached_cdf: list[Percentile] | None = None

    class BoundSafeNumericDistribution(NumericDistribution):
        def get_cdf(self) -> list[Percentile]:
            nonlocal cached_cdf
            if cached_cdf is not None:
                return cached_cdf
            base = super().get_cdf()
            probs = np.array([p.percentile for p in base], dtype=float)
            # Scale the min/max-step constraints to the actual grid length. On a coarse
            # discrete grid (cdf_size < 201) the 201-grid defaults (max_step=0.2) would
            # wrongly clip each bin to 20%; grid_step_constraints mirrors the server's
            # per-bin rules so the fallback matches the pipeline's discrete-build path.
            min_step, max_step = grid_step_constraints(len(base))
            safe = safe_cdf_bounds(
                probs,
                self.open_lower_bound,
                self.open_upper_bound,
                min_step=min_step,
                max_step=max_step,
                question_id=question.id_of_question,
                model_name=model_name,
            )
            cached_cdf = [
                Percentile(percentile=float(prob), value=p.value) for prob, p in zip(safe, base, strict=False)
            ]
            return cached_cdf

        @property
        def cdf(self) -> list[Percentile]:
            """Deprecated alias for :meth:`get_cdf` (returns identical data)."""
            return self.get_cdf()

    return BoundSafeNumericDistribution(
        declared_percentiles=percentile_list,
        open_upper_bound=question.open_upper_bound,
        open_lower_bound=question.open_lower_bound,
        upper_bound=question.upper_bound,
        lower_bound=question.lower_bound,
        zero_point=zero_point,
        cdf_size=question.cdf_size,
        is_date=isinstance(question, EpochDateQuestion),
        # strict_validation=False: preserve the beyond-range declared percentiles
        # verbatim (no _check_too_far_from_bounds rejection, no
        # _check_and_update_repeating_values mutation). standardize_cdf=False: keep
        # super().get_cdf() on the raw linear-interp path — our safe_cdf_bounds pass
        # in get_cdf() owns endpoint/step enforcement.
        strict_validation=False,
        standardize_cdf=False,
    )
