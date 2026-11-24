"""The Mantic-only out-of-range tail floor (``metaculus_bot/numeric/out_of_range_floor.py``).

Mantic scores an outcome beyond an OPEN bound against a fixed 5% reference (``50 * ln(mass / 0.05)``):
5% there scores 0, the structural 1% this pipeline publishes when every percentile sits inside the
range scores -80.5. Mantic's writers set the ranges and half its date questions resolved above the
displayed range in Series 1, so the PUBLISHED aggregate of a Mantic numeric, discrete or date question
gets each open tail raised to ``MANTIC_OUT_OF_RANGE_TAIL_FLOOR``; Metaculus aggregates receive their own lower floor and per-member forecasts are never touched.

A tail rises only as far as the other tail leaves room for: the server needs every one of the grid's
N steps to be at least ``round(0.01 / N, 9)``, so when one open tail already holds most of the mass
the other is raised to ``min(floor, 1 - fat tail - N * min step)`` rather than to the floor. Without
that cap a both-open aggregate with 98% below the lower bound had its upper tail set to 5%, which put
``cdf[-1]`` under ``cdf[0]`` and made the rebuild raise at the aggregation seam, forfeiting the
question (codex second-opinion review, 2026-09-08).

Three layers are pinned here: the pure array function (its semantics and the server's CDF rules on
its output), the same rules on every distinct grid Mantic has ever published (the recorded corpus,
distilled into ``tests/data/mantic_cdf_grids_2026_09_08.json``), and the forecaster seam that applies
it to the published aggregate and records raw and published tails on ``NUMERIC_AGGREGATE``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from forecasting_tools import NumericDistribution, NumericQuestion
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import DateQuestion
from scipy.stats import norm

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import (
    MANTIC_OUT_OF_RANGE_TAIL_FLOOR,
    METACULUS_OUT_OF_RANGE_TAIL_FLOOR,
    PLATFORM_MANTIC,
    PLATFORM_METACULUS,
)
from metaculus_bot.numeric.config import (
    OPEN_TAIL_MIN_MASS,
    PMF_FLOOR_MARGIN,
    STANDARD_PERCENTILES,
    grid_step_constraints,
)
from metaculus_bot.numeric.date_axis import as_epoch_question, to_epoch
from metaculus_bot.numeric.out_of_range_floor import (
    FlooredCdf,
    floor_out_of_range_tails,
    floor_published_tails,
    tail_floor_for_platform,
)
from metaculus_bot.numeric.pchip_cdf import safe_cdf_bounds
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric.pmf_cdf import build_pmf_distribution
from metaculus_bot.numeric.utils import aggregate_numeric
from metaculus_bot.question_platform import question_platform
from tests.pipeline_test_helpers import (
    ORACLE_BIN_COUNTS,
    assert_server_accepts_cdf,
    cdf_heights,
    distribution_from_heights,
    make_count_question,
    make_e2e_bot,
    make_real_date_question,
    make_real_numeric_question,
    on_mantic,
)

FLOOR = MANTIC_OUT_OF_RANGE_TAIL_FLOOR
GRIDS_PATH = Path(__file__).parent / "data" / "mantic_cdf_grids_2026_09_08.json"


def _least_interior_mass(cdf_size: int) -> float:
    """The mass a legal CDF on this grid must keep between its endpoints: every step at least the min step."""
    min_step, _ = grid_step_constraints(cdf_size)
    return (cdf_size - 1) * min_step


def _legal_cdf(raw_heights: np.ndarray, *, open_lower: bool, open_upper: bool) -> np.ndarray:
    """Make raw heights server-legal the way the ensemble post-processing does: endpoint pins, then
    ``safe_cdf_bounds`` with the grid's own step limits; asserts the server accepts the result."""
    heights = np.maximum.accumulate(np.clip(np.asarray(raw_heights, dtype=float), 0.0, 1.0))
    heights[0] = max(heights[0], 0.001) if open_lower else 0.0
    heights[-1] = min(heights[-1], 0.999) if open_upper else 1.0
    min_step, max_step = grid_step_constraints(len(heights))
    legal = safe_cdf_bounds(heights, open_lower, open_upper, min_step=min_step, max_step=max_step)
    assert_server_accepts_cdf(legal, cdf_size=len(legal), open_lower=open_lower, open_upper=open_upper)
    return legal


def _normal_cdf(n_points: int, *, centre: float, sd: float) -> np.ndarray:
    """A normal CDF sampled on the unit grid: ``centre`` and ``sd`` are fractions of the range."""
    return norm.cdf(np.linspace(0.0, 1.0, n_points), loc=centre, scale=sd)


def _fat_lower_tail(n_points: int, mass_below: float) -> np.ndarray:
    """``mass_below`` of the mass under the lower bound, the structural 0.1% above the upper, linear between."""
    return np.linspace(mass_below, 0.999, n_points)


def _fat_upper_tail(n_points: int, mass_above: float) -> np.ndarray:
    """The mirror image: ``mass_above`` beyond the upper bound, the structural 0.1% below the lower."""
    return np.linspace(0.001, 1.0 - mass_above, n_points)


SHAPES: dict[str, Callable[[int], np.ndarray]] = {
    "concentrated_middle": lambda n: _normal_cdf(n, centre=0.5, sd=0.03),
    "piled_on_lower_edge": lambda n: _normal_cdf(n, centre=0.05, sd=0.02),
    "piled_on_upper_edge": lambda n: _normal_cdf(n, centre=0.95, sd=0.02),
    "flat": lambda n: np.linspace(0.0, 1.0, n),
    "fat_both_tails": lambda n: _normal_cdf(n, centre=0.5, sd=0.6),
    "fat_upper_tail_only": lambda n: _normal_cdf(n, centre=0.9, sd=0.3),
    # One tail already fat, the other structural: 94% leaves exactly the floor of room, 99.9% none at all.
    "mass_below_lower_94pct": lambda n: _fat_lower_tail(n, 0.94),
    "mass_below_lower_95pct": lambda n: _fat_lower_tail(n, 0.95),
    "mass_below_lower_98pct": lambda n: _fat_lower_tail(n, 0.98),
    "mass_below_lower_99.9pct": lambda n: np.full(n, 0.999),
    "mass_above_upper_94pct": lambda n: _fat_upper_tail(n, 0.94),
    "mass_above_upper_95pct": lambda n: _fat_upper_tail(n, 0.95),
    "mass_above_upper_98pct": lambda n: _fat_upper_tail(n, 0.98),
    "mass_above_upper_99.9pct": lambda n: np.full(n, 0.001),
}
"""Every aggregate shape a Mantic question can publish, up to the grid, as raw heights on the unit grid."""


def _shape(name: str, n_points: int) -> np.ndarray:
    return SHAPES[name](n_points)


def _assert_open_tail_follows_the_rule(raw: float, published: float, *, interior_after: float, least_interior: float):
    """Never lowered; at or above the floor it is left alone; under it, raised to the floor unless the
    other tail leaves the interior only its minimum, in which case it rises exactly that far."""
    assert published >= raw, "a tail is never lowered"
    if raw >= FLOOR:
        assert published == raw, "a tail already at or above the floor keeps its exact mass"
        return
    assert published <= FLOOR + 1e-12, "a tail is never raised above the floor"
    if published < FLOOR - 1e-12:
        assert interior_after == pytest.approx(least_interior), (
            "short of the floor only when the interior is at its minimum"
        )


def _assert_floored_result_is_legal(result: FlooredCdf, before: np.ndarray, *, open_lower: bool, open_upper: bool):
    """The server's rules, the per-side tail rule, and ``floor`` as the level the moved tails were
    raised to (the floor, or less where the other tail capped it), ``0.0`` when none moved."""
    after = result.cdf
    assert len(after) == len(before)
    assert np.all(np.diff(after) > 0.0), "the floored CDF must be strictly increasing"
    assert_server_accepts_cdf(after, cdf_size=len(after), open_lower=open_lower, open_upper=open_upper)

    raw_low, raw_high = float(before[0]), 1.0 - float(before[-1])
    assert result.raw == (raw_low, raw_high)
    assert result.published == (float(after[0]), 1.0 - float(after[-1]))

    interior_after = float(after[-1] - after[0])
    least_interior = _least_interior_mass(len(before))
    if open_lower:
        _assert_open_tail_follows_the_rule(
            raw_low, result.published[0], interior_after=interior_after, least_interior=least_interior
        )
    else:
        assert after[0] == 0.0 == before[0]
    if open_upper:
        _assert_open_tail_follows_the_rule(
            raw_high, result.published[1], interior_after=interior_after, least_interior=least_interior
        )
    else:
        assert after[-1] == 1.0 == before[-1]

    lower_moved, upper_moved = bool(after[0] != before[0]), bool(after[-1] != before[-1])
    if not (lower_moved or upper_moved):
        assert result.floor == 0.0
        return
    assert 0.0 < result.floor <= FLOOR
    if lower_moved:
        assert result.published[0] == pytest.approx(result.floor)
    if upper_moved:
        assert result.published[1] == pytest.approx(result.floor)


class TestFloorSemantics:
    """The array function on the standard 201-point grid, one rule per test."""

    def test_both_open_tails_under_the_floor_are_raised_to_exactly_the_floor(self) -> None:
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=True, open_upper=True)
        assert (before[0], before[-1]) == (0.001, 0.999), "the structural 1% tails are the input"

        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)

        _assert_floored_result_is_legal(result, before, open_lower=True, open_upper=True)
        assert result.floor == FLOOR
        assert result.published == (FLOOR, 1.0 - (1.0 - FLOOR))
        assert result.published[1] == pytest.approx(FLOOR)

    def test_the_interior_is_rescaled_not_shifted_so_the_median_stays_put(self) -> None:
        """An affine map of [0.001, 0.999] onto [0.05, 0.95] fixes 0.5; the min-step re-lift of the tail
        bins then moves the crossing by a few thousandths of a bin, nothing a reader could see."""
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=True, open_upper=True)
        after = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR).cdf
        grid = np.arange(len(before), dtype=float)
        crossing_before = float(np.interp(0.5, before, grid))
        crossing_after = float(np.interp(0.5, after, grid))
        assert crossing_before == pytest.approx(100.0)
        assert abs(crossing_after - crossing_before) < 0.01

    def test_an_open_lower_and_closed_upper_bound_floors_the_low_side_only(self) -> None:
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=True, open_upper=False)
        result = floor_out_of_range_tails(before, open_lower=True, open_upper=False, floor=FLOOR)
        _assert_floored_result_is_legal(result, before, open_lower=True, open_upper=False)
        assert result.cdf[0] == FLOOR
        assert result.cdf[-1] == 1.0

    def test_a_closed_lower_and_open_upper_bound_floors_the_high_side_only(self) -> None:
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=False, open_upper=True)
        result = floor_out_of_range_tails(before, open_lower=False, open_upper=True, floor=FLOOR)
        _assert_floored_result_is_legal(result, before, open_lower=False, open_upper=True)
        assert result.cdf[0] == 0.0
        assert result.cdf[-1] == 1.0 - FLOOR

    def test_two_closed_bounds_return_the_input_unchanged(self) -> None:
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=False, open_upper=False)
        result = floor_out_of_range_tails(before, open_lower=False, open_upper=False, floor=FLOOR)
        assert result.floor == 0.0
        assert np.array_equal(result.cdf, before)
        assert result.raw == result.published == (0.0, 0.0)

    def test_tails_already_at_or_above_the_floor_are_never_reduced(self) -> None:
        before = _legal_cdf(_shape("fat_both_tails", 201), open_lower=True, open_upper=True)
        assert before[0] > FLOOR
        assert 1.0 - before[-1] > FLOOR
        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)
        assert result.floor == 0.0
        assert np.array_equal(result.cdf, before)

    def test_one_thin_and_one_fat_tail_floors_only_the_thin_side(self) -> None:
        before = _legal_cdf(_shape("fat_upper_tail_only", 201), open_lower=True, open_upper=True)
        assert before[0] < FLOOR < 1.0 - before[-1]
        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)
        _assert_floored_result_is_legal(result, before, open_lower=True, open_upper=True)
        assert result.cdf[0] == FLOOR
        assert result.cdf[-1] == before[-1], "the fat side keeps its exact mass"

    def test_a_flat_cdf_floors_both_sides_and_stays_flat_inside(self) -> None:
        before = _legal_cdf(_shape("flat", 201), open_lower=True, open_upper=True)
        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)
        _assert_floored_result_is_legal(result, before, open_lower=True, open_upper=True)
        interior_steps = np.diff(result.cdf)[1:-1]
        assert interior_steps.max() - interior_steps.min() < 1e-9, "an affine map keeps a uniform interior uniform"

    def test_a_zero_floor_is_a_no_op_on_any_shape(self) -> None:
        for name in SHAPES:
            before = _legal_cdf(_shape(name, 201), open_lower=True, open_upper=True)
            result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=0.0)
            assert result.floor == 0.0, name
            assert np.array_equal(result.cdf, before), name

    def test_the_input_array_is_not_mutated(self) -> None:
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=True, open_upper=True)
        snapshot = before.copy()
        floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)
        assert np.array_equal(before, snapshot)

    def test_the_min_step_is_re_enforced_because_the_shrink_alone_breaks_it(self) -> None:
        """A concentrated aggregate carries long runs of bins sitting EXACTLY at the server's minimum
        step (the uniform-mixture tails of the PCHIP build). The affine shrink scales every step by
        the same factor under one, so those bins land below the minimum and the server would reject
        the submission; the floor re-runs the pipeline's min-step sweep, which is what makes the
        result legal. Pinned by showing the naive shrink fails where the function passes."""
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=True, open_upper=True)
        min_step, _ = grid_step_constraints(len(before))
        assert np.isclose(np.diff(before), min_step).sum() > 100, "the input carries min-step runs"

        scale = (1.0 - 2.0 * FLOOR) / (before[-1] - before[0])
        naive = FLOOR + (before - before[0]) * scale
        with pytest.raises(AssertionError, match="step below server min"):
            assert_server_accepts_cdf(naive, cdf_size=len(naive), open_lower=True, open_upper=True)

        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)
        _assert_floored_result_is_legal(result, before, open_lower=True, open_upper=True)


_FAT_TAIL_MASSES = (0.98, 0.945)
"""The codex review's two shapes: 98% beyond one bound made the naive floor non-monotone, 94.5% left zero-width steps."""
_GRIDS_UNDER_TEST = (13, 201, 451, 2001)
"""Mantic's date grid, the standard grid, the discrete corpus's largest and the platform's largest."""


class TestAFatTailLeavesLittleRoomForTheFloor:
    """One open tail already holds most of the mass; the other rises only as far as the interior allows.

    The server needs every one of the grid's N steps to be at least ``round(0.01 / N, 9)``, so the
    interior keeps at least N times that, and the thin tail is raised to
    ``min(floor, 1 - fat tail - that interior)``: about 1% behind a 98% tail, 4.5% behind a 94.5%
    one, and nothing when the interior is already at its minimum. The fat tail is never reduced.
    Before this cap the 98% shape came back with ``cdf[-1] = 0.95 < cdf[0] = 0.98`` and the seam's
    rebuild raised ``ValidationError``, forfeiting the question.
    """

    @pytest.mark.parametrize("cdf_size", _GRIDS_UNDER_TEST)
    @pytest.mark.parametrize("mass_below", _FAT_TAIL_MASSES)
    def test_a_fat_lower_tail_caps_how_far_the_upper_tail_rises(self, cdf_size: int, mass_below: float) -> None:
        before = _legal_cdf(_fat_lower_tail(cdf_size, mass_below), open_lower=True, open_upper=True)
        assert (before[0], before[-1]) == (mass_below, 0.999), "the legaliser leaves this shape as built"

        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)

        _assert_floored_result_is_legal(result, before, open_lower=True, open_upper=True)
        room = 1.0 - mass_below - _least_interior_mass(cdf_size)
        assert 0.001 < room < FLOOR
        assert result.cdf[0] == mass_below, "the fat tail is never reduced"
        assert result.published[1] == pytest.approx(room), "the thin tail rises exactly as far as the interior allows"
        assert result.floor == pytest.approx(room)
        assert float(result.cdf[-1] - result.cdf[0]) == pytest.approx(_least_interior_mass(cdf_size))

    @pytest.mark.parametrize("cdf_size", _GRIDS_UNDER_TEST)
    @pytest.mark.parametrize("mass_above", _FAT_TAIL_MASSES)
    def test_a_fat_upper_tail_caps_how_far_the_lower_tail_rises(self, cdf_size: int, mass_above: float) -> None:
        before = _legal_cdf(_fat_upper_tail(cdf_size, mass_above), open_lower=True, open_upper=True)
        assert before[0] == 0.001
        assert 1.0 - before[-1] == pytest.approx(mass_above)

        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)

        _assert_floored_result_is_legal(result, before, open_lower=True, open_upper=True)
        room = 1.0 - mass_above - _least_interior_mass(cdf_size)
        assert 0.001 < room < FLOOR
        assert result.cdf[-1] == before[-1], "the fat tail is never reduced"
        assert result.published[0] == pytest.approx(room)
        assert result.floor == pytest.approx(room)
        assert float(result.cdf[-1] - result.cdf[0]) == pytest.approx(_least_interior_mass(cdf_size))

    @pytest.mark.parametrize("cdf_size", _GRIDS_UNDER_TEST)
    def test_an_interior_already_at_its_minimum_leaves_the_cdf_untouched(self, cdf_size: int) -> None:
        """99.9% below the lower bound legalises to the tightest interior the server accepts, and a fat
        tail is never reduced, so there is no room at all: the input comes back as is."""
        before = _legal_cdf(np.full(cdf_size, 0.999), open_lower=True, open_upper=True)
        assert float(before[-1] - before[0]) == pytest.approx(_least_interior_mass(cdf_size))

        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)

        assert result.floor == 0.0
        assert result.cdf is before
        assert result.raw == result.published

    def test_a_94_percent_tail_leaves_exactly_the_floor_of_room_on_the_standard_grid(self) -> None:
        before = _legal_cdf(_fat_lower_tail(201, 0.94), open_lower=True, open_upper=True)
        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)
        _assert_floored_result_is_legal(result, before, open_lower=True, open_upper=True)
        assert result.floor == FLOOR
        assert result.published[1] == pytest.approx(FLOOR)
        assert float(result.cdf[-1] - result.cdf[0]) == pytest.approx(_least_interior_mass(201))

    @pytest.mark.parametrize("cdf_size", _GRIDS_UNDER_TEST)
    @pytest.mark.parametrize("mass_beyond", _FAT_TAIL_MASSES)
    def test_a_fat_open_tail_beside_a_closed_bound_is_left_alone(self, cdf_size: int, mass_beyond: float) -> None:
        fat_lower = _legal_cdf(_fat_lower_tail(cdf_size, mass_beyond), open_lower=True, open_upper=False)
        fat_upper = _legal_cdf(_fat_upper_tail(cdf_size, mass_beyond), open_lower=False, open_upper=True)
        for before, open_lower, open_upper in ((fat_lower, True, False), (fat_upper, False, True)):
            result = floor_out_of_range_tails(before, open_lower=open_lower, open_upper=open_upper, floor=FLOOR)
            _assert_floored_result_is_legal(result, before, open_lower=open_lower, open_upper=open_upper)
            assert result.floor == 0.0
            assert np.array_equal(result.cdf, before)


def _corpus_grids() -> list[tuple[int, bool, bool, int]]:
    payload = json.loads(GRIDS_PATH.read_text(encoding="utf-8"))
    return [
        (g["inbound_outcome_count"] + 1, g["open_lower_bound"], g["open_upper_bound"], g["questions"])
        for g in payload["grids"]
    ]


class TestEveryManticGrid:
    """The server rules hold on every distinct grid Mantic has published, for every shape.

    The fixture holds each ``(inbound_outcome_count, open_lower_bound, open_upper_bound)`` triple
    among the 524 continuous questions of the recorded 2026-09-08 corpus: 88 grids from 3 to 450
    bins, both-open, one-open and both-closed. The floor acts on heights alone, so the value axis
    is not part of a grid's identity.
    """

    def test_the_fixture_covers_the_corpus(self) -> None:
        grids = _corpus_grids()
        assert len(grids) == 88
        assert sum(questions for *_, questions in grids) == 524
        assert {(lo, hi) for _, lo, hi, _ in grids} == {(True, True), (False, True), (True, False), (False, False)}
        assert (min(n for n, *_ in grids), max(n for n, *_ in grids)) == (4, 451)

    @pytest.mark.parametrize("shape_name", sorted(SHAPES))
    def test_the_floored_cdf_passes_the_server_on_every_grid(self, shape_name: str) -> None:
        for cdf_size, open_lower, open_upper, _ in _corpus_grids():
            before = _legal_cdf(_shape(shape_name, cdf_size), open_lower=open_lower, open_upper=open_upper)
            result = floor_out_of_range_tails(before, open_lower=open_lower, open_upper=open_upper, floor=FLOOR)
            try:
                _assert_floored_result_is_legal(result, before, open_lower=open_lower, open_upper=open_upper)
            except AssertionError as exc:
                raise AssertionError(f"{shape_name} on cdf_size={cdf_size} open=({open_lower},{open_upper})") from exc

    def test_a_thin_tailed_shape_ends_with_the_floor_beyond_every_open_bound(self) -> None:
        for cdf_size, open_lower, open_upper, _ in _corpus_grids():
            before = _legal_cdf(_shape("concentrated_middle", cdf_size), open_lower=open_lower, open_upper=open_upper)
            after = floor_out_of_range_tails(before, open_lower=open_lower, open_upper=open_upper, floor=FLOOR).cdf
            if open_lower:
                assert after[0] >= FLOOR, (cdf_size, open_lower, open_upper)
            if open_upper:
                assert 1.0 - after[-1] >= FLOOR - 1e-12, (cdf_size, open_lower, open_upper)

    def test_a_fat_tailed_shape_still_rises_on_every_both_open_grid(self) -> None:
        """98% beyond one bound: the other tail leaves its structural 0.1% on every grid, and the fat
        tail never moves. Both-open grids only; beside a closed bound there is nothing to raise."""
        for cdf_size, open_lower, open_upper, _ in _corpus_grids():
            if not (open_lower and open_upper):
                continue
            before = _legal_cdf(_shape("mass_below_lower_98pct", cdf_size), open_lower=True, open_upper=True)
            result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)
            assert result.cdf[0] == before[0], cdf_size
            assert result.cdf[-1] < before[-1], cdf_size
            assert 0.0 < result.floor < FLOOR, cdf_size


_OPEN_BOUND_SHAPES = (
    pytest.param(False, True, id="closed-open"),
    pytest.param(True, True, id="open-open"),
)


class TestPooledPerBinMembersUnderTheFloor:
    """The modal live path for a coarse open-bound Mantic grid: three per-bin builds, the mean pool, then the floor.

    Every bin a member declared 0 on leaves the builder at ``min_step + PMF_FLOOR_MARGIN`` and each open
    tail at ``OPEN_TAIL_MIN_MASS + PMF_FLOOR_MARGIN``. Raising a tail to 5% rescales the interior by about
    0.90, which pushes every one of those bins UNDER the min step, so acceptance rests on the floor's
    min-step sweep restoring them to exactly the min step with no margin left, and on the server's
    9-decimal PMF rounding not landing them short of it.
    """

    @pytest.mark.parametrize(("open_lower", "open_upper"), _OPEN_BOUND_SHAPES)
    @pytest.mark.parametrize("n_bins", ORACLE_BIN_COUNTS)
    def test_the_floor_moves_and_the_server_accepts_the_result(
        self, n_bins: int, open_lower: bool, open_upper: bool
    ) -> None:
        question = make_count_question(n_bins, open_lower=open_lower, open_upper=open_upper)
        believed = sorted({0, n_bins // 2, n_bins - 1})
        members: list[NumericDistribution] = []
        for k in believed:
            declared = [0.0] * (n_bins + 2)
            declared[1 + k] = 1.0
            members.append(build_pmf_distribution(declared, question))
        pooled = aggregate_numeric(members, question, "mean")
        builder_tail = OPEN_TAIL_MIN_MASS + PMF_FLOOR_MARGIN
        raw_tails = (builder_tail if open_lower else 0.0, builder_tail)
        assert (cdf_heights(pooled)[0], 1.0 - cdf_heights(pooled)[-1]) == pytest.approx(raw_tails, abs=1e-14)

        outcome = floor_published_tails(pooled, question)

        assert outcome.floor == FLOOR, "both tails sat at the builder's floor, so this is the moving case"
        assert outcome.raw == pytest.approx(raw_tails, abs=1e-14)
        assert outcome.published == pytest.approx((FLOOR if open_lower else 0.0, FLOOR))
        published = cdf_heights(outcome.distribution)
        assert_server_accepts_cdf(published, cdf_size=question.cdf_size, open_lower=open_lower, open_upper=open_upper)
        min_step, _ = grid_step_constraints(question.cdf_size)
        pmf = np.diff(published)
        zero_bins = [k for k in range(n_bins) if k not in believed]
        assert pmf[zero_bins] == pytest.approx(min_step, abs=1e-12), "the margin is consumed: exactly the min step"
        assert np.round(pmf[zero_bins], 9).tolist() == [min_step] * len(zero_bins)
        assert all(pmf[k] > 0.25 for k in believed), "each believed bin keeps the pool's third, scaled by the interior"


def _normal_members(question: NumericQuestion, centres: tuple[float, ...], sd: float) -> list[NumericDistribution]:
    """Three members with normal percentiles at ``centres``: inside the range they carry the structural
    1% tails, beyond a bound they pile the aggregate's mass past it (the shape the feasibility cap is for)."""
    members: list[NumericDistribution] = []
    for centre in centres:
        declared = [
            Percentile(percentile=p, value=float(norm.ppf(p, loc=centre, scale=sd))) for p in STANDARD_PERCENTILES
        ]
        sanitized, zero_point = sanitize_percentiles(declared, question, model_name="test-model")
        members.append(build_numeric_distribution(sanitized, question, zero_point, model_name="test-model"))
    return members


def _aggregate_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("NUMERIC_AGGREGATE:")]


def _fields(line: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in line.split(": ", 1)[1].split(" "))


class TestPlatformGate:
    def test_each_platform_gets_its_own_floor(self) -> None:
        mantic = on_mantic(make_real_numeric_question(open_lower_bound=True))
        assert question_platform(mantic) == PLATFORM_MANTIC
        assert tail_floor_for_platform(mantic) == FLOOR

        metaculus = make_real_numeric_question(open_lower_bound=True)
        assert question_platform(metaculus) == PLATFORM_METACULUS
        assert tail_floor_for_platform(metaculus) == METACULUS_OUT_OF_RANGE_TAIL_FLOOR == 0.02

    def test_the_date_adapter_keeps_the_platform(self) -> None:
        date_question = make_real_date_question(open_upper_bound=True)
        assert tail_floor_for_platform(date_question) == FLOOR
        assert tail_floor_for_platform(as_epoch_question(date_question)) == FLOOR


class TestTheForecasterSeam:
    """``TemplateForecaster._aggregate_predictions``, the one seam every aggregation path returns through."""

    @pytest.mark.asyncio
    async def test_a_mantic_numeric_aggregate_publishes_the_floor_beyond_each_open_bound(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        question = on_mantic(make_real_numeric_question(open_lower_bound=True, open_upper_bound=True))
        members = _normal_members(question, centres=(10.0, 10.4, 9.7), sd=1.0)
        member_heights_before = [cdf_heights(m) for m in members]

        bot = make_e2e_bot(AggregationStrategy.MEDIAN)
        aggregated = await bot._aggregate_predictions(cast(list[PredictionTypes], members), question)

        assert isinstance(aggregated, NumericDistribution)
        published = cdf_heights(aggregated)
        assert published[0] >= FLOOR
        assert 1.0 - published[-1] >= FLOOR - 1e-12
        assert_server_accepts_cdf(published, cdf_size=len(published), open_lower=True, open_upper=True)
        assert (aggregated.open_lower_bound, aggregated.open_upper_bound) == (True, True)
        assert (aggregated.lower_bound, aggregated.upper_bound) == (question.lower_bound, question.upper_bound)
        assert [p.value for p in aggregated.declared_percentiles] == [p.value for p in aggregated.get_cdf()]
        assert [p.percentile for p in aggregated.declared_percentiles] == list(map(float, published))
        for before, member in zip(member_heights_before, members, strict=True):
            assert np.array_equal(before, cdf_heights(member))

        (line,) = _aggregate_lines(caplog)
        fields = _fields(line)
        assert fields["qtype"] == "numeric"
        assert fields["cdf_size"] == "201"
        assert (fields["oor_low_raw"], fields["oor_high_raw"]) == ("0.010000", "0.010000")
        assert (fields["oor_low"], fields["oor_high"]) == ("0.050000", "0.050000")
        assert fields["tail_floor"] == "0.050000"

    @pytest.mark.asyncio
    async def test_a_metaculus_aggregate_with_structural_tails_is_raised_to_the_metaculus_floor(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        question = make_real_numeric_question(open_lower_bound=True, open_upper_bound=True)
        assert question_platform(question) == PLATFORM_METACULUS
        members = _normal_members(question, centres=(10.0, 10.4, 9.7), sd=1.0)
        expected = cdf_heights(aggregate_numeric(members, question, method="median"))

        bot = make_e2e_bot(AggregationStrategy.MEDIAN)
        aggregated = await bot._aggregate_predictions(cast(list[PredictionTypes], members), question)

        assert isinstance(aggregated, NumericDistribution)
        assert (expected[0], expected[-1]) == pytest.approx((0.01, 0.99)), "the combiner builds structural 1% tails"
        published = cdf_heights(aggregated)
        assert (published[0], published[-1]) == pytest.approx(
            (METACULUS_OUT_OF_RANGE_TAIL_FLOOR, 1.0 - METACULUS_OUT_OF_RANGE_TAIL_FLOOR)
        )
        (line,) = _aggregate_lines(caplog)
        fields = _fields(line)
        assert (fields["oor_low_raw"], fields["oor_high_raw"]) == ("0.010000", "0.010000")
        assert (fields["oor_low"], fields["oor_high"]) == ("0.020000", "0.020000")
        assert fields["tail_floor"] == "0.020000"

    def test_the_same_object_comes_back_when_nothing_is_floored(self) -> None:
        question = make_real_numeric_question(open_lower_bound=False, open_upper_bound=False)
        members = _normal_members(question, centres=(10.0, 10.4, 9.7), sd=1.0)
        aggregated = aggregate_numeric(members, question, method="median")

        outcome = floor_published_tails(aggregated, question)

        assert outcome.distribution is aggregated
        assert outcome.floor == 0.0
        assert outcome.raw == outcome.published
        assert outcome.raw == pytest.approx((0.0, 0.0))
        assert outcome.cdf_size == 201

    @pytest.mark.asyncio
    async def test_a_mantic_date_aggregate_floors_its_open_upper_bound_and_stays_a_date(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        question: DateQuestion = make_real_date_question(open_lower_bound=False, open_upper_bound=True)
        epoch_question = as_epoch_question(question)
        day = datetime(2026, 9, 16, tzinfo=UTC)
        centres = tuple(to_epoch(day + timedelta(hours=h)) for h in (12, 13, 11))
        members = _normal_members(epoch_question, centres=centres, sd=4 * 3600.0)

        bot = make_e2e_bot(AggregationStrategy.MEDIAN)
        aggregated = await bot._aggregate_predictions(cast(list[PredictionTypes], members), question)

        assert isinstance(aggregated, NumericDistribution)
        assert aggregated.is_date is True, "the rebuilt aggregate must still render as dates in the comment"
        published = cdf_heights(aggregated)
        assert len(published) == 13
        assert published[0] == 0.0, "a closed lower bound is never moved"
        assert 1.0 - published[-1] >= FLOOR - 1e-12
        assert_server_accepts_cdf(published, cdf_size=13, open_lower=False, open_upper=True)
        assert int(np.argmax(np.diff(published))) == 8
        (line,) = _aggregate_lines(caplog)
        fields = _fields(line)
        assert (fields["qtype"], fields["cdf_size"]) == ("date", "13")
        assert fields["oor_low"] == fields["oor_low_raw"] == "0.000000"
        assert (fields["oor_high_raw"], fields["oor_high"]) == ("0.010000", "0.050000")
        assert fields["tail_floor"] == "0.050000"

    def test_the_codex_shape_rebuilds_into_a_distribution_instead_of_raising(self) -> None:
        """98% below the lower bound, 0.1% above the upper, on the standard grid: the shape whose naive
        floor put ``cdf[-1]`` under ``cdf[0]`` and made this rebuild raise ``ValidationError``."""
        question = on_mantic(make_real_numeric_question(open_lower_bound=True, open_upper_bound=True))
        before = _legal_cdf(_fat_lower_tail(201, 0.98), open_lower=True, open_upper=True)
        aggregated = distribution_from_heights(before, question)

        outcome = floor_published_tails(aggregated, question)

        assert outcome.distribution is not aggregated
        published = cdf_heights(outcome.distribution)
        assert_server_accepts_cdf(published, cdf_size=201, open_lower=True, open_upper=True)
        room = 1.0 - 0.98 - _least_interior_mass(201)
        assert published[0] == 0.98
        assert 1.0 - published[-1] == pytest.approx(room)
        assert outcome.raw == (0.98, pytest.approx(0.001))
        assert outcome.published == (0.98, pytest.approx(room))
        assert outcome.floor == pytest.approx(room)
        assert outcome.cdf_size == 201

    @pytest.mark.asyncio
    async def test_members_piled_below_the_range_publish_a_capped_upper_tail_and_an_honest_marker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every member's thirteen percentiles sit below the range, so the aggregate carries about 96%
        beyond the open lower bound and the builder's 1% above the upper. The upper tail can rise only
        to what the interior's min-step mass leaves (about 2.6%), not to 5%, and the marker records
        that level as the floor applied."""
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        question = on_mantic(make_real_numeric_question(open_lower_bound=True, open_upper_bound=True))
        members = _normal_members(question, centres=(-1.8, -1.6, -2.0), sd=1.0)
        raw = cdf_heights(aggregate_numeric(members, question, method="median"))
        assert raw[0] > 0.95
        assert 1.0 - raw[-1] == pytest.approx(0.01)
        room = 1.0 - raw[0] - _least_interior_mass(201)
        assert 0.01 < room < FLOOR

        bot = make_e2e_bot(AggregationStrategy.MEDIAN)
        aggregated = await bot._aggregate_predictions(cast(list[PredictionTypes], members), question)

        assert isinstance(aggregated, NumericDistribution)
        published = cdf_heights(aggregated)
        assert_server_accepts_cdf(published, cdf_size=201, open_lower=True, open_upper=True)
        assert published[0] == raw[0], "the fat tail is never reduced"
        assert 1.0 - published[-1] == pytest.approx(room)
        (line,) = _aggregate_lines(caplog)
        fields = _fields(line)
        assert fields["oor_low"] == fields["oor_low_raw"] == f"{raw[0]:.6f}"
        assert fields["oor_high_raw"] == "0.010000"
        assert fields["oor_high"] == fields["tail_floor"] == f"{room:.6f}"

    @pytest.mark.asyncio
    async def test_a_date_aggregate_piled_past_its_open_upper_bound_still_rebuilds_as_a_date(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Both bounds open, every member centred a few hours after the range ends: the upper tail is fat
        and the lower rises exactly as far as the 13-point grid's interior allows, through the epoch
        adapter. A 12-bin Mantic date question pools by the pointwise MEAN (``elicit_per_bin``), so the
        raw reference is the members' mean. The centres sit close enough to the bound that the interior
        holds more than its minimum, which is what leaves the floor room to move; members entirely beyond
        the range legalise to an interior already at its minimum, the untouched case pinned above."""
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        question: DateQuestion = make_real_date_question(open_lower_bound=True, open_upper_bound=True)
        epoch_question = as_epoch_question(question)
        range_end = datetime(2026, 9, 20, tzinfo=UTC)
        centres = tuple(to_epoch(range_end + timedelta(hours=h)) for h in (8, 9, 7))
        members = _normal_members(epoch_question, centres=centres, sd=4 * 3600.0)
        raw = cdf_heights(aggregate_numeric(members, epoch_question, method="mean"))
        assert 1.0 - raw[-1] > 0.95
        assert raw[-1] - raw[0] > _least_interior_mass(13), "the interior holds more than its minimum"

        bot = make_e2e_bot(AggregationStrategy.MEDIAN)
        aggregated = await bot._aggregate_predictions(cast(list[PredictionTypes], members), question)

        assert isinstance(aggregated, NumericDistribution)
        assert aggregated.is_date is True
        published = cdf_heights(aggregated)
        assert_server_accepts_cdf(published, cdf_size=13, open_lower=True, open_upper=True)
        assert published[-1] == raw[-1], "the fat tail is never reduced"
        assert raw[0] < published[0] < FLOOR
        assert float(published[-1] - published[0]) == pytest.approx(_least_interior_mass(13))
        (line,) = _aggregate_lines(caplog)
        fields = _fields(line)
        assert (fields["qtype"], fields["cdf_size"]) == ("date", "13")
        assert fields["oor_high"] == fields["oor_high_raw"]
        assert fields["oor_low"] == fields["tail_floor"] == f"{published[0]:.6f}"
