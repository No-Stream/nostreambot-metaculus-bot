"""The floor on the mass a PUBLISHED aggregate places beyond each open bound, per platform.

Metaculus gets ``METACULUS_OUT_OF_RANGE_TAIL_FLOOR`` (2%) since 2026-11-24: the structural 1% this
pipeline builds whenever P1 sits inside the range cost about -80 on each of two fall 2026 escapes,
and over our history escapes ran at or above the 2% floor's 1.44%-per-side breakeven (receipt:
``scratch/trace_evals_2026-11-23/tail_mass_investigation.md`` in the artifacts repo).

Mantic (the Crucible competition, a fork of the Metaculus platform) scores a resolution beyond an
OPEN bound against a fixed 0.05 reference, ``50 * ln(mass / 0.05)``: 5% in that bucket scores 0 and
the structural 1% this pipeline publishes whenever every declared percentile sits inside the range
scores -80.5. Its question writers set the ranges themselves and are paid for bot disagreement; in
Series 1 half the date questions, a quarter of the discrete and an eighth of the numeric ones
resolved outside the displayed range. So on a Mantic question the published aggregate carries at
least ``MANTIC_OUT_OF_RANGE_TAIL_FLOOR`` beyond each open bound. Applied to every Series 1
competitor's own published distributions the same floor cost at most 1.9 points per question on
average and gained up to 9.1 for thin-tailed bots (``scratch_docs_and_planning/
mantic_adversarial_candidates_2026-09-08.md``, section 1).

Where it applies: ``forecaster._aggregate_predictions``, the one seam every aggregation path (median,
mean, base-combine, stacker, single survivor) returns through, so it is the last touch before the
publish gate, the comment and the ``NUMERIC_AGGREGATE`` marker read the distribution. Nowhere else:
per-member forecasts are never touched
(their ``MEMBER_FORECAST`` tails keep measuring what the models declared), and the backtest and
ablation harnesses replay Metaculus questions and receive the same 2% floor.

What it does, per side: an open tail under the floor is raised toward it, an open tail already at or
above it is left alone (a tail is never reduced), and a closed bound stays at its exact 0.0 or 1.0
(the server requires that). A tail rises only as far as the other tail leaves room for. The server
needs every one of the grid's ``N`` steps to be at least ``round(0.01 / N, 9)``, so the interior
keeps at least ``N`` times that, and each open tail is raised to
``min(floor, 1 - the other tail - that interior mass)``: the floor itself whenever the other tail is
thin too (the interior keeps 0.90), less when the other tail is fat (98% below the lower bound on
the 201-point grid leaves the upper tail room to rise from 0.1% to 1%), and nothing at all when the
interior already sits at its minimum. Both sides settle from the same two raw endpoints, which is
enough because a fat tail's target is its own raw value. Without the cap a both-open aggregate with
98% below the lower bound had its upper tail set to 5%, so ``cdf[-1] = 0.95 < cdf[0] = 0.98`` and
the rebuild raised at the aggregation seam, which has no fallback: the question was forfeited (codex
second-opinion review, 2026-09-08). A confident aggregate beyond one open bound is a realistic Mantic
shape, since half its date questions resolved above the range.

The interior is rescaled affinely between the new endpoints, which keeps it monotone, fixes the
median of a symmetric shape in place and can only shrink steps, so the platform's max step cannot be
newly violated; it CAN push the bins that sat exactly on the min step (the uniform-mixture tails of
a concentrated PCHIP build) below it, so the pipeline's min-step sweep runs again with the new
endpoints as its caps. With the endpoints feasible that sweep needs nothing more: its forward pass
lifts each bin to the min step and its backward pass pulls the run down from the upper cap, and the
lower cap is only ever reached exactly. A lift under ``_LIFT_TOLERANCE`` (the server's 10-decimal CDF
rounding, so invisible to it) is not a move: an aggregate whose interior already sits at the minimum
arrives with its endpoints a few 1e-15 off it, accumulated over the grid's steps, and without the
tolerance the seam rebuilt the distribution for that noise and logged the tail's own raw mass as the
floor applied. ``tests/test_out_of_range_floor.py`` proves the server's rules on the result for
every distinct grid Mantic has published.

The raw (pre-floor) tails are kept beside the published ones so the floor can be benchmarked on
this bot's own forecasts once live telemetry accumulates: the marker records both, and its
``tail_floor`` is the level the moved tails were actually raised to (the floor, or the capped level).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from forecasting_tools import MetaculusQuestion, NumericDistribution, Percentile

from metaculus_bot.constants import (
    MANTIC_OUT_OF_RANGE_TAIL_FLOOR,
    METACULUS_OUT_OF_RANGE_TAIL_FLOOR,
    PLATFORM_MANTIC,
)
from metaculus_bot.numeric.config import grid_step_constraints
from metaculus_bot.numeric.date_axis import numeric_view
from metaculus_bot.numeric.pchip_cdf import enforce_min_steps
from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution
from metaculus_bot.question_platform import question_platform

__all__ = [
    "FlooredCdf",
    "TailFloorOutcome",
    "floor_out_of_range_tails",
    "floor_published_tails",
    "tail_floor_for_platform",
]

# The smallest lift of a tail that counts as a move: the server's 10-decimal CDF rounding (module docstring).
_LIFT_TOLERANCE: float = 1e-10


@dataclass(frozen=True)
class FlooredCdf:
    """A CDF's heights after the floor, with the mass beyond each bound before and after.

    ``raw`` and ``published`` are ``(below the lower bound, above the upper bound)``, i.e.
    ``cdf[0]`` and ``1 - cdf[-1]``. ``floor`` is the level the moved tails were raised to: the
    nominal floor, or less where the other tail left the interior no more than its min-step mass;
    ``0.0`` when neither endpoint moved, in which case ``cdf`` is the input array itself.
    """

    cdf: np.ndarray
    raw: tuple[float, float]
    published: tuple[float, float]
    floor: float


@dataclass(frozen=True)
class TailFloorOutcome:
    """What the seam publishes and logs: the distribution to publish and the marker's tail fields.

    ``distribution`` is the input object itself when nothing was floored. ``floor`` is the level the
    moved tails were raised to (``FlooredCdf.floor``), ``0.0`` when none moved (a
    closed-bound question, tails already at or above the floor, or an interior already
    at its minimum).
    """

    distribution: NumericDistribution
    cdf_size: int
    raw: tuple[float, float]
    published: tuple[float, float]
    floor: float


def tail_floor_for_platform(question: MetaculusQuestion) -> float:
    """``MANTIC_OUT_OF_RANGE_TAIL_FLOOR`` on a Mantic question, ``METACULUS_OUT_OF_RANGE_TAIL_FLOOR`` otherwise."""
    if question_platform(question) == PLATFORM_MANTIC:
        return MANTIC_OUT_OF_RANGE_TAIL_FLOOR
    return METACULUS_OUT_OF_RANGE_TAIL_FLOOR


def floor_out_of_range_tails(cdf: np.ndarray, *, open_lower: bool, open_upper: bool, floor: float) -> FlooredCdf:
    """Raise each OPEN tail of a server-legal CDF toward ``floor``, as far as the other tail allows; pure.

    ``cdf`` is the heights the server would receive (length ``inbound bins + 1``, monotone, endpoints
    already pinned for the bound flags). A floor of ``0.0`` is a no-op that hands the input back.
    """
    low, high = float(cdf[0]), float(cdf[-1])
    raw = (low, 1.0 - high)
    min_step, _ = grid_step_constraints(cdf.size)
    least_interior = (cdf.size - 1) * min_step
    lifted_low = min(floor, high - least_interior)
    lifted_high = min(floor, 1.0 - low - least_interior)
    lower_moved = open_lower and lifted_low > low + _LIFT_TOLERANCE
    upper_moved = open_upper and lifted_high > raw[1] + _LIFT_TOLERANCE
    if not (lower_moved or upper_moved):
        return FlooredCdf(cdf=cdf, raw=raw, published=raw, floor=0.0)

    target_low = lifted_low if lower_moved else low
    target_high = 1.0 - lifted_high if upper_moved else high
    rescaled = target_low + (cdf - low) * ((target_high - target_low) / (high - low))
    rescaled[0] = target_low
    rescaled[-1] = target_high
    legal = enforce_min_steps(rescaled, min_step, upper_cap=target_high, lower_cap=target_low)
    applied = max(lifted_low if lower_moved else 0.0, lifted_high if upper_moved else 0.0)
    return FlooredCdf(cdf=legal, raw=raw, published=(float(legal[0]), 1.0 - float(legal[-1])), floor=applied)


def floor_published_tails(aggregated: NumericDistribution, question: MetaculusQuestion) -> TailFloorOutcome:
    """Apply the platform's tail floor to the aggregate about to be published.

    Materialises the CDF once (the same read the marker's ``cdf_size`` and tails come from). When an
    endpoint moves, the distribution is rebuilt on the question's numeric view with the floored
    heights as both its CDF and its ``declared_percentiles``, on the value axis the aggregate already
    carried, so the comment, the spread readers and the submission all see one distribution; the
    view is the epoch adapter on a date question, which keeps ``is_date`` and the date rendering.
    """
    heights = aggregated.get_cdf()
    cdf = np.fromiter((point.percentile for point in heights), dtype=float, count=len(heights))
    result = floor_out_of_range_tails(
        cdf,
        open_lower=aggregated.open_lower_bound,
        open_upper=aggregated.open_upper_bound,
        floor=tail_floor_for_platform(question),
    )
    if result.floor == 0.0:
        return TailFloorOutcome(
            distribution=aggregated, cdf_size=cdf.size, raw=result.raw, published=result.raw, floor=0.0
        )

    declared = [
        Percentile(percentile=float(probability), value=float(point.value))
        for point, probability in zip(heights, result.cdf, strict=True)
    ]
    rebuilt = create_pchip_numeric_distribution(
        pchip_cdf=[float(probability) for probability in result.cdf],
        percentile_list=declared,
        question=numeric_view(question),
        zero_point=aggregated.zero_point,
    )
    return TailFloorOutcome(
        distribution=rebuilt,
        cdf_size=cdf.size,
        raw=result.raw,
        published=result.published,
        floor=result.floor,
    )
