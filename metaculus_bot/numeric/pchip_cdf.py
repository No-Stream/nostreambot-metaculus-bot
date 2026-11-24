"""
PCHIP-based CDF construction for robust numeric forecasting.

Based on the battle-tested implementation from panchul (Q2 2025 competition winner).
Provides smooth, monotonic CDF construction with strict constraints enforcement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from metaculus_bot.constants import NUM_MAX_STEP, NUM_MIN_PROB_STEP
from metaculus_bot.numeric.config import OPEN_TAIL_MIN_MASS

logger = logging.getLogger(__name__)

# Headroom on the uniform-mixture weight so the blended CDF clears the min-step
# constraint rather than landing exactly on it (mirrors ``discrete_snap``).
_ALPHA_SAFETY_MARGIN: float = 1.1

# How far above ``max_step`` a step must sit to count as a violation, and how little
# displaced mass is left before packing stops chasing it.
_STEP_TOLERANCE: float = 1e-12
_MASS_TOLERANCE: float = 1e-15
# When packing gives up, how much unplaced mass makes it a real failure rather than
# accumulated float drift. Deliberately far looser than _MASS_TOLERANCE (a ~200-ring walk
# can drift a few times 1e-15) and far below the min-step, so nothing meaningful is lost.
_UNPLACED_MASS_TOLERANCE: float = 1e-9
# How far below ``min_step`` an adjacent step may sit before it counts as a violation. One
# tolerance for the rebuild trigger, the rebuild's range check, its post-check and the final
# assertion: when only the last two carried it, a 1e-18 step deficit tripped the trigger and
# the range check then refused a range 1e-16 short, dropping a forecast that put almost all
# of its mass beyond an open bound (Mantic edge-case review, 2026-09). The server rounds the
# PMF to 9 decimals, so a deficit this small is invisible to it.
_MIN_STEP_TOLERANCE: float = 1e-10


@dataclass
class StatedTailDiagnostics:
    """Repairs made while applying explicit probabilities at the question bounds."""

    lower_dropped_percentiles: tuple[float, ...] = ()
    upper_dropped_percentiles: tuple[float, ...] = ()
    lower_clamped: bool = False
    upper_clamped: bool = False
    lower_applied_probability: float | None = None
    upper_applied_probability: float | None = None
    stated_tails_discarded: bool = False


@dataclass(frozen=True)
class _MaxStepRepair:
    """A max-step repair's output plus what the cap displaced (the marker's fields)."""

    cdf: np.ndarray
    clipped_mass: float
    over_cap_bins: int
    bins_displaced: int
    max_offset_bins: int


def _pack_excess_nearest_first(steps: np.ndarray, max_step: float) -> None:
    """Clip each over-cap bin to ``max_step`` and pour its excess into the NEAREST bins, in place.

    Nearest-first rather than proportional-to-slack: on a fine grid every other bin has
    almost the full cap free, so slack-proportional allocation spreads a clipped spike
    near-uniformly over the whole grid. q45065 published 47% of its mass above 35 deaths
    where all three forecasters had declared ~2% there. The cap itself is the platform's
    (``0.2 * 200 / N``), so the spike is genuinely unpublishable; where the excess lands
    is ours, and adjacent bins are the legal shape closest to what was declared.
    """
    n_bins = steps.size
    for source in np.flatnonzero(steps > max_step + _STEP_TOLERANCE):
        excess = float(steps[source]) - max_step
        steps[source] = max_step
        for offset in range(1, n_bins):
            if excess <= _MASS_TOLERANCE:
                break
            ring = [j for j in (source - offset, source + offset) if 0 <= j < n_bins]
            room = np.maximum(max_step - steps[ring], 0.0)
            total_room = float(room.sum())
            if total_room <= 0.0:
                continue
            if total_room <= excess:
                # ``+= room`` rather than ``= max_step``: a ring member can itself be an
                # over-cap bin awaiting its own turn as ``source``, and assigning the cap
                # would silently delete its excess.
                steps[ring] += room
                excess -= total_room
            else:
                # Split the remainder across the ring in proportion to headroom, so a
                # symmetric declaration does not acquire a one-sided tail at the last ring.
                steps[ring] += excess * room / total_room
                excess = 0.0
        if excess > _UNPLACED_MASS_TOLERANCE:
            raise RuntimeError(
                f"Failed to redistribute CDF probability mass within max step constraint "
                f"(unplaced excess={excess:.12f}, bins={n_bins}, max_step={max_step})"
            )


def _displacement_stats(before: np.ndarray, after: np.ndarray, over_cap: np.ndarray) -> tuple[int, int]:
    """``(bins that received displaced mass, farthest such bin's distance from a clipped bin)``."""
    received = np.flatnonzero(after > before + _STEP_TOLERANCE)
    sources = np.flatnonzero(over_cap)
    if received.size == 0 or sources.size == 0:
        return 0, 0
    offsets = np.abs(received[:, None] - sources[None, :]).min(axis=1)
    return int(received.size), int(offsets.max())


def _redistribute_excess_probability(cdf: np.ndarray, max_step: float) -> _MaxStepRepair:
    """Clip every over-cap step and pack the displaced mass into the nearest bins with headroom.

    Total mass and monotonicity survive by construction: packing only ADDS to bins that
    had headroom, so the min-step cannot be tightened either. Callers must hand in a
    MONOTONE CDF (every production caller pre-monotonizes); the ``np.maximum(raw_steps,
    0.0)`` floor below is a defensive last resort, not a supported path — it inflates
    total mass, and the endpoint is restored only for monotone input.
    """
    raw_steps = np.diff(cdf)
    was_monotone = not bool(np.any(raw_steps < 0.0))
    steps = np.maximum(raw_steps, 0.0)
    over_cap = steps > max_step + _STEP_TOLERANCE
    clipped_mass = float(np.sum(steps[over_cap] - max_step))

    before = steps.copy()
    _pack_excess_nearest_first(steps, max_step)
    bins_displaced, max_offset_bins = _displacement_stats(before, steps, over_cap)

    new_cdf = np.empty_like(cdf)
    new_cdf[0] = cdf[0]
    new_cdf[1:] = cdf[0] + np.cumsum(steps)
    if was_monotone:
        # The repair only MOVES mass between bins, so a monotone input's endpoint is
        # unchanged by contract; assigning it back keeps a closed upper bound exactly 1.0
        # instead of one ulp under, which Metaculus compares for equality.
        new_cdf[-1] = cdf[-1]
    return _MaxStepRepair(
        cdf=new_cdf,
        clipped_mass=clipped_mass,
        over_cap_bins=int(np.count_nonzero(over_cap)),
        bins_displaced=bins_displaced,
        max_offset_bins=max_offset_bins,
    )


def safe_cdf_bounds(
    cdf: np.ndarray,
    open_lower: bool,
    open_upper: bool,
    *,
    min_step: float = NUM_MIN_PROB_STEP,
    max_step: float = NUM_MAX_STEP,
    question_id: int | str | None = None,
    model_name: str = "",
) -> np.ndarray:
    """
    Ensure CDF respects Metaculus boundary constraints:
    • For *open* bounds: cdf[0] ≥ ``OPEN_TAIL_MIN_MASS``, cdf[-1] ≤ ``1 - OPEN_TAIL_MIN_MASS``
    • No single step may exceed ``max_step``
    • Adjacent steps stay ≥ ``min_step`` (re-enforced after pin+cummax)

    ``min_step`` / ``max_step`` default to the 201-point-grid constants
    (``NUM_MIN_PROB_STEP`` / ``NUM_MAX_STEP``). A caller building a CDF on a non-201 grid
    (a discrete question with ``cdf_size != 201``) must pass the grid-scaled values so the
    per-bin constraints match the server's ``round(0.01 / N, 9)`` min-step and
    ``0.2 * 200 / N`` max-step formulas, where ``N = cdf_size - 1``.

    ``question_id`` / ``model_name`` only label the ``CDF_MAXSTEP_CLIP`` marker emitted
    when the max-step cap displaces mass; they are absent on the ablation/pooling callers,
    where the repair is not a production forecast.
    """
    # Work on a copy to avoid mutating callers unexpectedly
    cdf = cdf.copy()

    # Pin tails to legal open-bound limits
    if open_lower:
        cdf[0] = max(cdf[0], OPEN_TAIL_MIN_MASS)
    if open_upper:
        cdf[-1] = min(cdf[-1], 1.0 - OPEN_TAIL_MIN_MASS)

    # Enforce the maximum step rule
    pre_max_step = float(np.max(np.diff(cdf))) if cdf.size > 1 else 0.0
    if pre_max_step > max_step + _STEP_TOLERANCE:
        repair = _redistribute_excess_probability(cdf, max_step)
        cdf = repair.cdf
        # WARN, not DEBUG: this repair reshaped 47% of q45065's published mass and left no
        # trace in the run logs, so the 2026-07-15 "repair-tier WARNs never fire" audit
        # never saw it. NOT alertable — the cap is the platform's, not a bot defect.
        logger.warning(
            "CDF_MAXSTEP_CLIP: question=%s model=%s clipped_mass=%.6f over_cap_bins=%d "
            "bins_displaced=%d max_offset_bins=%d pre_max_step=%.6f max_step=%.6f",
            question_id,
            model_name or "unknown",
            repair.clipped_mass,
            repair.over_cap_bins,
            repair.bins_displaced,
            repair.max_offset_bins,
            pre_max_step,
            max_step,
        )

    # Ensure monotonicity and clamp to legal probability range
    np.maximum.accumulate(cdf, out=cdf)
    np.clip(cdf, 0.0, 1.0, out=cdf)

    # Re-apply open bounds in case redistribution nudged them
    if open_lower:
        cdf[0] = max(cdf[0], OPEN_TAIL_MIN_MASS)
    if open_upper:
        cdf[-1] = min(cdf[-1], 1.0 - OPEN_TAIL_MIN_MASS)

    if cdf.size > 1:
        np.maximum.accumulate(cdf, out=cdf)
        # Pinning cdf[0] up to the open-tail minimum + cummax flattens any prefix below it into 0-step bins.
        upper_cap = 1.0 - OPEN_TAIL_MIN_MASS if open_upper else 1.0
        lower_cap = OPEN_TAIL_MIN_MASS if open_lower else 0.0
        cdf = enforce_min_steps(cdf, min_step, upper_cap=upper_cap, lower_cap=lower_cap)

    return cdf


def enforce_strict_increasing(
    percentile_dict: dict[int | float, float],
) -> dict[int | float, float]:
    """Ensure strictly increasing values by adding tiny jitter if necessary."""
    sorted_items = sorted(percentile_dict.items())
    last_val = -float("inf")
    new_dict = {}

    for p, v in sorted_items:
        # A tiny epsilon above the previous value when the declaration repeats or dips.
        adjusted = last_val + 1e-8 if v <= last_val else v
        new_dict[p] = adjusted
        last_val = adjusted

    return new_dict


def enforce_min_steps(
    y_values: np.ndarray,
    min_step: float,
    *,
    upper_cap: float = 1.0,
    lower_cap: float = 0.0,
) -> np.ndarray:
    """Enforce minimum step size between adjacent points (panchul-style sweep).

    Forward pass lifts each point to be at least ``prev + min_step`` (capped at
    ``upper_cap``). When the upper cap pins the last point before the grid
    ends, a backward pass pulls earlier points down so every step still meets
    ``min_step``. Used by both the PCHIP pipeline and the mixture-CDF builder
    to keep CDFs strictly increasing under the Metaculus min-step constraint.
    """
    n = len(y_values)
    result = y_values.copy()
    for i in range(1, n):
        if result[i] < result[i - 1] + min_step:
            result[i] = result[i - 1] + min_step
        if result[i] > upper_cap:
            result[i] = upper_cap
    for j in range(n - 2, -1, -1):
        if result[j] > result[j + 1] - min_step:
            result[j] = result[j + 1] - min_step
        if result[j] < lower_cap:
            result[j] = lower_cap
    return result


def build_cdf_value_grid(
    lower_bound: float,
    upper_bound: float,
    zero_point: float | None,
    num_points: int = 201,
) -> np.ndarray:
    """Build the CDF evaluation value grid: linear for linear-scaled questions, geometric
    when ``zero_point`` is set (log-scaled questions).

    This is the canonical grid the production CDF lives on and the grid the Metaculus
    scorer buckets resolutions against (see ``scoring_common.resolution_to_bucket_index``).
    Offline pooling primitives that consume x-values (Vincentization, CRPS) must build their
    grid here so a zero_point question's pooled CDF aligns with the scorer's buckets.

    The geometric branch matches the Metaculus backend's non-linear spacing:
    ``x(t) = lower + (upper - lower) * (ratio**t - 1) / (ratio - 1)`` where
    ``ratio = (upper - zero_point) / (lower - zero_point)`` and ``t`` ranges over a uniform
    [0, 1] grid of ``num_points`` points.
    """
    t = np.linspace(0, 1, num_points)

    if zero_point is None:
        # Linear grid
        return lower_bound + (upper_bound - lower_bound) * t

    # Non-linear grid based on zero_point
    ratio = (upper_bound - zero_point) / (lower_bound - zero_point)
    # Handle potential numerical issues
    if abs(ratio - 1.0) < 1e-10:
        return lower_bound + (upper_bound - lower_bound) * t
    return lower_bound + (upper_bound - lower_bound) * ((ratio**t - 1) / (ratio - 1))


def _validate_pchip_bounds(
    percentile_values: dict[int | float, float],
    lower_bound: float,
    upper_bound: float,
    zero_point: float | None,
) -> None:
    """Reject inputs the interpolation cannot be built on at all."""
    if not percentile_values:
        raise ValueError("Empty percentile values dictionary")

    if upper_bound <= lower_bound:
        raise ValueError(f"Upper bound ({upper_bound}) must be greater than lower bound ({lower_bound})")

    if zero_point is not None and (abs(zero_point - lower_bound) < 1e-6 or abs(zero_point - upper_bound) < 1e-6):
        raise ValueError(f"zero_point ({zero_point}) too close to bounds [{lower_bound}, {upper_bound}]")


def _clean_percentile_values(percentile_values: dict[int | float, float]) -> dict[float, float]:
    """Drop out-of-range percentile LABELS and reject unusable percentile VALUES.

    The KEY filter is a genuine filter and must stay one: a caller that hands a whole CDF
    over as prob*100 labels legitimately includes exactly 0.0 and 100.0, and those are
    dropped here because the boundary points are re-added below from the question's own
    bound semantics. ``test_boundary_labels_stay_silently_filtered`` pins the build as
    identical with and without them.

    A bad VALUE is the opposite case and raises. Silently skipping it built a
    12-of-13-point CDF while ``declared_percentiles`` still advertised 13 — a distribution
    missing an anchor the model declared, with nothing recording the loss. It is
    reachable: ``json.loads`` accepts a bare ``NaN``, and the strictly-increasing check
    (``value <= prev``) is False for NaN, so NaN used to pass the block schema and publish.
    Every caller already handles ValueError (``build_numeric_distribution`` falls back to
    forecasting-tools' builder, which re-validates), and the extraction ladder's own
    finiteness check now closes the upstream path.
    """
    cleaned: dict[float, float] = {}
    for label, raw_value in percentile_values.items():
        try:
            label_float = float(label)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"non-numeric percentile label {label!r} in declared percentiles") from exc
        if not (0 < label_float < 100):
            continue  # Boundary/out-of-range labels: dropped by design (see above).
        try:
            value_float = float(raw_value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"non-numeric value {raw_value!r} at percentile {label_float}") from exc
        if not np.isfinite(value_float):
            raise ValueError(f"non-finite value {value_float} at percentile {label_float}")
        cleaned[label_float] = value_float

    if len(cleaned) < 2:
        raise ValueError(f"Need at least 2 valid percentile points (got {len(cleaned)})")
    return cleaned


def _nudge_duplicate_values_apart(percentile_values: dict[float, float]) -> dict[float, float]:
    """Offset repeated values by 1e-9 so PCHIP sees a strictly increasing x-axis."""
    sorted_items = sorted(percentile_values.items())
    last_value = -float("inf")

    for label, value in sorted_items:
        adjusted = last_value + 1e-9 if value <= last_value else value
        percentile_values[label] = adjusted
        last_value = adjusted

    return percentile_values


def _percentile_arrays_with_boundaries(
    percentile_values: dict[float, float],
    *,
    open_lower_bound: bool,
    open_upper_bound: bool,
    lower_bound: float,
    upper_bound: float,
    tail_probabilities: tuple[float, float] | None = None,
    p_below_lower_bound: float | None = None,
    p_above_upper_bound: float | None = None,
    tail_diagnostics: StatedTailDiagnostics | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add boundary anchors, preferring stated tail probabilities over conflicting percentiles."""
    percentile_labels, declared_values = zip(*sorted(percentile_values.items()), strict=False)
    percentiles = np.array(percentile_labels) / 100.0  # Convert to [0,1] range
    values = np.array(declared_values)

    if np.any(np.diff(values) <= 0):
        raise ValueError("Percentile values must be strictly increasing after de-duplication")

    if tail_probabilities is not None:
        lower_probability, upper_probability = tail_probabilities
        lower_conflicts = np.zeros(len(values), dtype=bool)
        upper_conflicts = np.zeros(len(values), dtype=bool)
        if p_below_lower_bound is not None:
            lower_conflicts = (
                ((values > lower_bound) & (percentiles < lower_probability))
                | ((values < lower_bound) & (percentiles > lower_probability))
                | (values == lower_bound)
            )
            if tail_diagnostics is not None:
                tail_diagnostics.lower_dropped_percentiles = tuple(
                    label for label, dropped in zip(percentile_labels, lower_conflicts, strict=True) if dropped
                )
        if p_above_upper_bound is not None:
            upper_level = 1.0 - upper_probability
            upper_conflicts = (
                ((values < upper_bound) & (percentiles > upper_level))
                | ((values > upper_bound) & (percentiles < upper_level))
                | (values == upper_bound)
            )
            if tail_diagnostics is not None:
                tail_diagnostics.upper_dropped_percentiles = tuple(
                    label for label, dropped in zip(percentile_labels, upper_conflicts, strict=True) if dropped
                )
        keep = ~(lower_conflicts | upper_conflicts)
        # The omitted side retains its legacy endpoint unless grid feasibility
        # required reducing its tail mass to accommodate the stated side.
        values = np.concatenate(([lower_bound, upper_bound], values[keep]))
        percentiles = np.concatenate(([lower_probability, 1.0 - upper_probability], percentiles[keep]))
        order = np.argsort(values, kind="stable")
        values, percentiles = values[order], percentiles[order]
        unique = np.concatenate(([True], np.diff(values) > 0.0))
        return percentiles[unique], values[unique]

    if not open_lower_bound and lower_bound < values[0] - 1e-9:
        percentiles = np.insert(percentiles, 0, 0.0)
        values = np.insert(values, 0, lower_bound)

    if not open_upper_bound and upper_bound > values[-1] + 1e-9:
        percentiles = np.append(percentiles, 1.0)
        values = np.append(values, upper_bound)

    return percentiles, values


def _feasible_tail_probabilities(
    cdf: np.ndarray,
    *,
    p_below_lower_bound: float | None,
    p_above_upper_bound: float | None,
    open_lower_bound: bool,
    open_upper_bound: bool,
    min_step: float,
    diagnostics: StatedTailDiagnostics,
) -> tuple[float, float]:
    """Keep stated tails within the endpoint and grid-mass constraints."""
    for probability in (p_below_lower_bound, p_above_upper_bound):
        if probability is not None and (not np.isfinite(probability) or not 0.0 <= probability <= 1.0):
            raise ValueError(f"Invalid stated tail probability: {probability}")

    lower_floor = OPEN_TAIL_MIN_MASS if open_lower_bound else 0.0
    upper_floor = OPEN_TAIL_MIN_MASS if open_upper_bound else 0.0
    tail_capacity = 1.0 - (len(cdf) - 1) * min_step
    if tail_capacity < lower_floor + upper_floor:
        raise ValueError("Grid minimum steps leave no room for the required open-tail probabilities")

    lower = float(cdf[0]) if p_below_lower_bound is None else p_below_lower_bound
    upper = 1.0 - float(cdf[-1]) if p_above_upper_bound is None else p_above_upper_bound
    lower = float(np.clip(lower, lower_floor, tail_capacity - upper_floor)) if open_lower_bound else 0.0
    upper = float(np.clip(upper, upper_floor, tail_capacity - lower_floor)) if open_upper_bound else 0.0

    if lower + upper > tail_capacity:
        if p_below_lower_bound is None:
            lower = tail_capacity - upper
        elif p_above_upper_bound is None:
            upper = tail_capacity - lower
        else:
            mass_above_floors = lower + upper - lower_floor - upper_floor
            available_above_floors = tail_capacity - lower_floor - upper_floor
            scale = available_above_floors / mass_above_floors
            lower = lower_floor + (lower - lower_floor) * scale
            upper = upper_floor + (upper - upper_floor) * scale

    if p_below_lower_bound is not None:
        diagnostics.lower_applied_probability = lower
        diagnostics.lower_clamped = lower != p_below_lower_bound
    if p_above_upper_bound is not None:
        diagnostics.upper_applied_probability = upper
        diagnostics.upper_clamped = upper != p_above_upper_bound
    return lower, upper


def _evaluate_pchip_on_grid(
    percentiles: np.ndarray,
    values: np.ndarray,
    *,
    open_lower_bound: bool,
    open_upper_bound: bool,
    lower_bound: float,
    upper_bound: float,
    zero_point: float | None,
    num_points: int,
) -> np.ndarray:
    """Interpolate the declaration onto the submission grid and pin closed bounds."""
    # Log scaling is appropriate when all values are positive and the lower bound is too.
    use_log = np.all(values > 0) and zero_point is None and lower_bound > 0
    x_vals = np.log(values) if use_log else values

    try:
        spline = PchipInterpolator(x_vals, percentiles, extrapolate=True)
    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except  # intentional: logged fallback to linear interp
        logger.warning("PchipInterpolator failed, falling back to linear interpolation", exc_info=True)

        def spline(x):
            return np.interp(x, x_vals, percentiles)

    # Generate the grid (linear, or geometric when zero_point is set) and evaluate,
    # clamping the evaluation points to avoid extrapolation issues.
    cdf_x = build_cdf_value_grid(lower_bound, upper_bound, zero_point, num_points)
    eval_x = np.log(cdf_x) if use_log else cdf_x
    eval_x_clamped = np.clip(eval_x, x_vals[0], x_vals[-1])

    cdf_y = spline(eval_x_clamped).clip(0.0, 1.0)
    cdf_y = np.maximum.accumulate(cdf_y)

    if not open_lower_bound:
        cdf_y[0] = 0.0
    if not open_upper_bound:
        cdf_y[-1] = 1.0

    return cdf_y


def _blend_with_uniform(cdf_y: np.ndarray, min_step: float, num_points: int) -> np.ndarray:
    """Mix in just enough of a uniform CDF to pre-satisfy the min-step constraint.

    This is the mechanism that keeps every downstream repair tier off real forecasts
    (same approach as ``discrete_snap``); the tiers below it are pathological-input
    guards, not routine passes.
    """
    total_range = float(cdf_y[-1] - cdf_y[0])
    min_alpha = min_step * num_points / total_range * _ALPHA_SAFETY_MARGIN if total_range > 1e-12 else 1.0
    alpha = min(1.0, min_alpha)
    uniform_cdf = np.linspace(float(cdf_y[0]), float(cdf_y[-1]), num_points)
    return (1.0 - alpha) * cdf_y + alpha * uniform_cdf


def _ramp_tail_to_one(cdf_y: np.ndarray, overflow_idx: int) -> None:
    """Replace the saturated tail with a uniform ramp up to 1.0, in place."""
    steps_remaining = len(cdf_y) - overflow_idx
    for i in range(overflow_idx, len(cdf_y)):
        t = (i - overflow_idx) / max(1, steps_remaining - 1)
        cdf_y[i] = min(1.0, cdf_y[overflow_idx - 1] + (1.0 - cdf_y[overflow_idx - 1]) * t)


def _pull_earlier_points_down(cdf_y: np.ndarray, from_idx: int, floor_idx: int, min_step: float) -> None:
    """Walk backwards from ``from_idx`` making room for ``min_step`` above each point."""
    for j in range(from_idx - 1, floor_idx - 1, -1):
        max_allowed = cdf_y[j + 1] - min_step
        if cdf_y[j] > max_allowed:
            cdf_y[j] = max_allowed


def _reenforce_min_steps_from(cdf_y: np.ndarray, start_idx: int, min_step: float) -> None:
    """Lift each point after ``start_idx`` to ``prev + min_step``, back-filling on overflow."""
    for i in range(start_idx + 1, len(cdf_y)):
        if cdf_y[i] >= cdf_y[i - 1] + min_step:
            continue
        cdf_y[i] = cdf_y[i - 1] + min_step
        if cdf_y[i] <= 1.0:
            continue
        cdf_y[i] = 1.0
        _pull_earlier_points_down(cdf_y, i, start_idx, min_step)


def _redistribute_saturated_tail(cdf_y: np.ndarray, min_step: float) -> None:
    """Legacy panchul redistribution for a CDF that overshot 1.0 before the grid end.

    Kept as its own pass because the ramp shape is specific to the PCHIP path.
    ``enforce_min_steps``' ``upper_cap=1.0`` already caps the forward sweep, so on the
    current pipeline this is a no-op guard rather than a live repair tier.
    """
    if cdf_y[-1] <= 1.0:
        return

    overflow_indices = np.where(cdf_y > 1.0)[0]
    if len(overflow_indices) == 0:
        return

    overflow_idx = int(overflow_indices[0])
    _ramp_tail_to_one(cdf_y, overflow_idx)
    _reenforce_min_steps_from(cdf_y, overflow_idx, min_step)


def _rebuild_with_min_steps(
    cdf_y: np.ndarray,
    min_step: float,
    *,
    open_lower_bound: bool,
    open_upper_bound: bool,
    question_id: int | str | None,
    question_url: str | None,
) -> np.ndarray:
    """Last repair tier: rebuild the CDF from its own step SHAPE with min-step floors.

    Fires only when the min-step is still violated after ``safe_cdf_bounds``. AGENTS.md
    records zero fires across 1182 archived numeric forecasts — on the production
    201-point grid ``_blend_with_uniform`` gets there first. It stays reachable on a
    coarse grid whose available range is exactly saturated by the min-step.

    Raises:
        ValueError: the CDF range cannot hold one ``min_step`` per bin (beyond
            ``_MIN_STEP_TOLERANCE``; a float-epsilon shortfall rebuilds).
        RuntimeError: the rebuild itself failed to satisfy the min-step.
    """
    steps = np.diff(cdf_y)
    violated_steps = np.sum(steps < min_step)
    logger.warning(
        "PCHIP minimum step enforcement required for Q %s | URL %s | violated_steps=%d/%d (%.1f%%) | min_step_found=%.8f | min_step_required=%.8f | available_range=%.6f | required_range=%.6f",
        question_id or "N/A",
        question_url or "N/A",
        violated_steps,
        len(steps),
        100.0 * violated_steps / len(steps),
        np.min(steps),
        min_step,
        cdf_y[-1] - cdf_y[0],
        (len(cdf_y) - 1) * min_step,
    )

    # Create a strictly monotonic sequence over the legal endpoint range.
    start_val = cdf_y[0] if open_lower_bound else 0.0
    end_val = min(cdf_y[-1], 1.0) if open_upper_bound else 1.0

    available_range = end_val - start_val
    required_range = (len(cdf_y) - 1) * min_step

    if required_range > available_range + _MIN_STEP_TOLERANCE:
        raise ValueError(
            f"Cannot satisfy minimum step requirement: need {required_range:.6f} "
            f"but only have {available_range:.6f} available in CDF range"
        )

    new_cdf = np.zeros_like(cdf_y)
    new_cdf[0] = start_val

    if len(cdf_y) > 2:
        # Keep the original shape but floor every step, then hand out what is left over.
        orig_shape = np.diff(cdf_y)
        orig_shape = np.maximum(orig_shape, min_step)
        orig_shape = orig_shape / np.sum(orig_shape)

        extra_steps = (available_range - required_range) * orig_shape
        for i in range(1, len(new_cdf)):
            new_cdf[i] = new_cdf[i - 1] + min_step + extra_steps[i - 1]
    else:
        # Simple linear spacing if original shape is unavailable
        for i in range(1, len(new_cdf)):
            new_cdf[i] = new_cdf[i - 1] + (available_range / (len(new_cdf) - 1))

    if np.any(np.diff(new_cdf) < min_step - _MIN_STEP_TOLERANCE):
        raise RuntimeError("Internal error: Step size enforcement failed")

    new_steps = np.diff(new_cdf)
    logger.info(
        "PCHIP aggressive enforcement completed for Q %s | URL %s | new_min_step=%.8f | new_max_step=%.8f | total_range_redistributed=%.6f | shape_preserved=True",
        question_id or "N/A",
        question_url or "N/A",
        np.min(new_steps),
        np.max(new_steps),
        available_range,
    )
    return new_cdf


def _assert_pchip_constraints(
    cdf_y: np.ndarray,
    min_step: float,
    *,
    open_lower_bound: bool,
    open_upper_bound: bool,
) -> None:
    """Fail loudly rather than submit a CDF the Metaculus validators would reject."""
    if np.any(np.diff(cdf_y) < min_step - _MIN_STEP_TOLERANCE):
        problematic_indices = np.where(np.diff(cdf_y) < min_step - _MIN_STEP_TOLERANCE)[0]
        raise RuntimeError(
            f"Failed to enforce minimum step size at indices: {problematic_indices}, "
            f"values: {np.diff(cdf_y)[problematic_indices]}"
        )

    if not open_lower_bound and abs(cdf_y[0]) > 1e-10:
        raise RuntimeError(f"Failed to enforce lower bound: {cdf_y[0]}")

    if not open_upper_bound and abs(cdf_y[-1] - 1.0) > 1e-10:
        raise RuntimeError(f"Failed to enforce upper bound: {cdf_y[-1]}")


def generate_pchip_cdf(
    percentile_values: dict[int | float, float],
    *,
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None = None,
    min_step: float = NUM_MIN_PROB_STEP,
    max_step: float = NUM_MAX_STEP,
    num_points: int = 201,
    question_id: int | str | None = None,
    question_url: str | None = None,
    model_name: str = "",
    p_below_lower_bound: float | None = None,
    p_above_upper_bound: float | None = None,
    tail_diagnostics: StatedTailDiagnostics | None = None,
) -> tuple[list[float], bool]:
    """
    Generate a robust continuous CDF using PCHIP interpolation with strict constraint enforcement.

    Based on the panchul implementation with enhancements for robustness. ``percentile_values``
    maps percentiles in (0, 100) to values; ``zero_point`` enables non-linear grid scaling.
    Returns ``(cdf_values, aggressive_enforcement_used)`` where the second element flags whether
    aggressive step enforcement was required to satisfy the min-step constraint.

    ``min_step`` / ``max_step`` default to the 201-point-grid constants. A caller building a CDF
    on a non-201 grid (a discrete question with ``num_points != 201``) must pass the grid-scaled
    values (see ``numeric.config.grid_step_constraints``) so the per-bin constraints match the
    server's ``round(0.01 / N, 9)`` min-step and ``0.2 * 200 / N`` max-step, where
    ``N = num_points - 1``. Passing the 201-grid ``max_step`` (0.2) on a coarse discrete grid
    wrongly clips each bin to 20% and shoves the excess onto higher bins.

    ``model_name`` only labels the ``CDF_MAXSTEP_CLIP`` marker (which forecaster's
    declaration the cap had to clip); callers that build an ensemble CDF pass a label for
    the aggregation stage instead.

    Explicit tail probabilities add bound anchors and remove contradictory percentile
    declarations. They are clamped only as required by open-tail floors and the grid's
    minimum total interior mass; ``tail_diagnostics`` records the applied values and
    dropped labels. Omitted probabilities keep the legacy construction.

    Raises:
        ValueError: If input validation fails
        RuntimeError: If constraint enforcement fails
    """
    _validate_pchip_bounds(percentile_values, lower_bound, upper_bound, zero_point)
    cleaned = _nudge_duplicate_values_apart(_clean_percentile_values(percentile_values))

    percentiles, values = _percentile_arrays_with_boundaries(
        cleaned,
        open_lower_bound=open_lower_bound,
        open_upper_bound=open_upper_bound,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    cdf_y = _evaluate_pchip_on_grid(
        percentiles,
        values,
        open_lower_bound=open_lower_bound,
        open_upper_bound=open_upper_bound,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        zero_point=zero_point,
        num_points=num_points,
    )

    if p_below_lower_bound is not None or p_above_upper_bound is not None:
        diagnostics = tail_diagnostics if tail_diagnostics is not None else StatedTailDiagnostics()
        lower_probability, upper_probability = _feasible_tail_probabilities(
            cdf_y,
            p_below_lower_bound=p_below_lower_bound,
            p_above_upper_bound=p_above_upper_bound,
            open_lower_bound=open_lower_bound,
            open_upper_bound=open_upper_bound,
            min_step=min_step,
            diagnostics=diagnostics,
        )
        percentiles, values = _percentile_arrays_with_boundaries(
            cleaned,
            open_lower_bound=open_lower_bound,
            open_upper_bound=open_upper_bound,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            tail_probabilities=(lower_probability, upper_probability),
            p_below_lower_bound=p_below_lower_bound,
            p_above_upper_bound=p_above_upper_bound,
            tail_diagnostics=diagnostics,
        )
        cdf_y = _evaluate_pchip_on_grid(
            percentiles,
            values,
            open_lower_bound=open_lower_bound,
            open_upper_bound=open_upper_bound,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            zero_point=zero_point,
            num_points=num_points,
        )

    # Repair ladder, cheapest first: uniform pre-mix, min-step sweep, saturated-tail
    # ramp, then boundary + max-step enforcement.
    cdf_y = _blend_with_uniform(cdf_y, min_step, num_points)
    cdf_y = enforce_min_steps(cdf_y, min_step, upper_cap=1.0, lower_cap=0.0)
    _redistribute_saturated_tail(cdf_y, min_step)
    cdf_y = safe_cdf_bounds(
        cdf_y,
        open_lower_bound,
        open_upper_bound,
        min_step=min_step,
        max_step=max_step,
        question_id=question_id,
        model_name=model_name,
    )

    aggressive_enforcement_used = bool(np.any(np.diff(cdf_y) < min_step - _MIN_STEP_TOLERANCE))
    if aggressive_enforcement_used:
        cdf_y = _rebuild_with_min_steps(
            cdf_y,
            min_step,
            open_lower_bound=open_lower_bound,
            open_upper_bound=open_upper_bound,
            question_id=question_id,
            question_url=question_url,
        )

    _assert_pchip_constraints(
        cdf_y,
        min_step,
        open_lower_bound=open_lower_bound,
        open_upper_bound=open_upper_bound,
    )

    return cdf_y.tolist(), aggressive_enforcement_used


def percentiles_to_pchip_format(percentiles: list) -> dict[float, float]:
    """
    Convert forecasting-tools Percentile objects to PCHIP input format.

    Args:
        percentiles: List of Percentile objects with .percentile and .value attributes

    Returns:
        Dictionary mapping percentile (0-100) to value
    """
    result = {}
    for p in percentiles:
        percentile_key = p.percentile * 100  # Convert from [0,1] to [0,100]
        result[percentile_key] = p.value
    return result
