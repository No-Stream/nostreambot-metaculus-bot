"""Stated out-of-range probabilities control percentile-path CDF endpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

import metaculus_bot.numeric.pipeline as pipeline
from metaculus_bot.numeric.config import OPEN_TAIL_MIN_MASS, STANDARD_PERCENTILES, grid_step_constraints
from metaculus_bot.numeric.pchip_cdf import StatedTailDiagnostics, generate_pchip_cdf
from metaculus_bot.numeric.pipeline import build_numeric_distribution
from metaculus_bot.structured_output_schema import _read_declared_tail_probability
from tests.pipeline_test_helpers import assert_server_accepts_cdf


def _generate(percentiles: dict[int | float, float], **kwargs: Any) -> tuple[list[float], StatedTailDiagnostics]:
    diagnostics = StatedTailDiagnostics()
    cdf, _ = generate_pchip_cdf(
        percentiles,
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=True,
        open_upper_bound=True,
        tail_diagnostics=diagnostics,
        **kwargs,
    )
    return cdf, diagnostics


class TestStatedTailAnchors:
    def test_stated_probabilities_override_conflicts_and_interpolate_from_bounds(self) -> None:
        cdf, diagnostics = _generate(
            {1: 10.0, 2.5: 15.0, 5: 20.0, 50: 50.0, 90: 80.0, 97.5: 90.0, 99: 95.0},
            p_below_lower_bound=0.04,
            p_above_upper_bound=0.06,
        )

        assert cdf[0] == pytest.approx(0.04, abs=1e-12)
        assert cdf[-1] == pytest.approx(0.94, abs=1e-12)
        assert diagnostics.lower_dropped_percentiles == (1.0, 2.5)
        assert diagnostics.upper_dropped_percentiles == (97.5, 99.0)
        assert not diagnostics.lower_clamped
        assert not diagnostics.upper_clamped
        # Bound-to-P5 is interpolated instead of a flat prefix repaired to min steps.
        min_step, _ = grid_step_constraints(len(cdf))
        assert cdf[20] - cdf[0] > 20 * min_step

    def test_conflicting_beyond_bound_and_on_bound_anchors_are_dropped(self) -> None:
        cdf, diagnostics = _generate(
            {1: -10.0, 2.5: 0.0, 5: 10.0, 50: 50.0, 95: 90.0, 97.5: 100.0, 99: 110.0},
            p_below_lower_bound=0.005,
            p_above_upper_bound=0.005,
        )

        assert cdf[0] == pytest.approx(0.005, abs=1e-12)
        assert cdf[-1] == pytest.approx(0.995, abs=1e-12)
        assert diagnostics.lower_dropped_percentiles == (1.0, 2.5)
        assert diagnostics.upper_dropped_percentiles == (97.5, 99.0)

    def test_missing_opposite_side_keeps_legacy_endpoint(self) -> None:
        percentiles: dict[int | float, float] = {1: 10.0, 5: 20.0, 50: 50.0, 95: 80.0, 99: 90.0}
        legacy, _ = _generate(percentiles)
        cdf, diagnostics = _generate(percentiles, p_below_lower_bound=0.04)

        assert cdf[0] == pytest.approx(0.04, abs=1e-12)
        assert cdf[-1] == legacy[-1]
        assert diagnostics.upper_applied_probability is None
        assert diagnostics.upper_dropped_percentiles == ()

    def test_matching_probability_at_a_different_value_is_kept(self) -> None:
        cdf, diagnostics = _generate(
            {1: 10.0, 4: 20.0, 50: 50.0, 94: 80.0, 99: 90.0},
            p_below_lower_bound=0.04,
            p_above_upper_bound=0.06,
        )

        assert cdf[0] == pytest.approx(0.04, abs=1e-12)
        assert cdf[-1] == pytest.approx(0.94, abs=1e-12)
        assert diagnostics.lower_dropped_percentiles == (1.0,)
        assert diagnostics.upper_dropped_percentiles == (99.0,)

    def test_missing_tail_arguments_keep_legacy_result(self) -> None:
        percentiles: dict[int | float, float] = {1: 10.0, 50: 50.0, 99: 90.0}
        legacy, _ = _generate(percentiles)
        explicit_none, diagnostics = _generate(percentiles, p_below_lower_bound=None, p_above_upper_bound=None)

        assert explicit_none == legacy
        assert legacy[0] == 0.01
        assert legacy[-1] == 0.99
        assert diagnostics == StatedTailDiagnostics()

    def test_probabilities_below_platform_floor_are_clamped(self) -> None:
        cdf, diagnostics = _generate(
            {1: 10.0, 50: 50.0, 99: 90.0},
            p_below_lower_bound=0.0,
            p_above_upper_bound=0.0,
        )

        assert cdf[0] == OPEN_TAIL_MIN_MASS
        assert cdf[-1] == 1.0 - OPEN_TAIL_MIN_MASS
        assert diagnostics.lower_clamped
        assert diagnostics.upper_clamped

    @pytest.mark.parametrize("num_points", [9, 201, 451])
    def test_nearly_exhausted_interior_is_clamped_to_grid_capacity(self, num_points: int) -> None:
        min_step, max_step = grid_step_constraints(num_points)
        cdf, diagnostics = _generate(
            {1: 10.0, 50: 50.0, 99: 90.0},
            p_below_lower_bound=0.6,
            p_above_upper_bound=0.395,
            num_points=num_points,
            min_step=min_step,
            max_step=max_step,
        )

        assert cdf[-1] - cdf[0] == pytest.approx((num_points - 1) * min_step, abs=1e-12)
        assert np.min(np.diff(cdf)) >= min_step - 1e-10
        assert diagnostics.lower_clamped
        assert diagnostics.upper_clamped
        assert diagnostics.lower_applied_probability == pytest.approx(cdf[0], abs=1e-12)
        assert diagnostics.upper_applied_probability == pytest.approx(1.0 - cdf[-1], abs=1e-12)

    def test_missing_side_yields_before_stated_side_for_grid_feasibility(self) -> None:
        cdf, diagnostics = _generate({1: 10.0, 50: 50.0, 99: 90.0}, p_below_lower_bound=0.985)

        assert cdf[0] == pytest.approx(0.985, abs=1e-12)
        assert cdf[-1] == pytest.approx(0.995, abs=1e-12)
        assert not diagnostics.lower_clamped

    def test_missing_upper_side_stays_monotone_when_all_percentiles_are_above_range(self) -> None:
        cdf, diagnostics = _generate({1: 200.0, 50: 300.0, 99: 400.0}, p_below_lower_bound=0.04)

        assert cdf[0] == pytest.approx(0.04, abs=1e-12)
        assert cdf[-1] == pytest.approx(0.05, abs=1e-12)
        assert np.min(np.diff(cdf)) >= grid_step_constraints(len(cdf))[0] - 1e-10
        assert diagnostics.lower_dropped_percentiles == (1.0,)

    def test_missing_lower_side_stays_monotone_when_all_percentiles_are_below_range(self) -> None:
        cdf, diagnostics = _generate({1: -300.0, 50: -200.0, 99: -100.0}, p_above_upper_bound=0.04)

        assert cdf[0] == pytest.approx(0.95, abs=1e-12)
        assert cdf[-1] == pytest.approx(0.96, abs=1e-12)
        assert np.min(np.diff(cdf)) >= grid_step_constraints(len(cdf))[0] - 1e-10
        assert diagnostics.upper_dropped_percentiles == (99.0,)

    @pytest.mark.parametrize("invalid_probability", [-0.1, 1.1, float("nan"), float("inf")])
    def test_builder_rejects_invalid_probabilities(self, invalid_probability: float) -> None:
        with pytest.raises(ValueError, match="tail probability"):
            _generate({1: 10.0, 50: 50.0, 99: 90.0}, p_below_lower_bound=invalid_probability)


@pytest.mark.parametrize("cdf_size", [9, 201, 451])
@pytest.mark.parametrize("zero_point", [None, -1.0])
def test_distribution_build_keeps_stated_tail_probabilities(cdf_size: int, zero_point: float | None) -> None:
    question = NumericQuestion(
        id_of_question=123,
        id_of_post=123,
        page_url="https://example.com/questions/123",
        question_text="How many?",
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=True,
        open_upper_bound=True,
        cdf_size=cdf_size,
        zero_point=zero_point,
    )
    percentiles = [Percentile(percentile=level, value=100.0 * level) for level in STANDARD_PERCENTILES]
    diagnostics = StatedTailDiagnostics()

    distribution = build_numeric_distribution(
        percentiles,
        question,
        zero_point,
        p_below_lower_bound=0.04,
        p_above_upper_bound=0.06,
        tail_diagnostics=diagnostics,
    )
    cdf = distribution.get_cdf()

    assert cdf[0].percentile == pytest.approx(0.04, abs=1e-12)
    assert cdf[-1].percentile == pytest.approx(0.94, abs=1e-12)
    assert_server_accepts_cdf(
        np.array([point.percentile for point in cdf]),
        cdf_size=cdf_size,
        open_lower=True,
        open_upper=True,
    )


def test_smoothing_keeps_the_applied_endpoints_on_a_saturated_grid() -> None:
    question = NumericQuestion(
        question_text="How many?",
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=True,
        open_upper_bound=True,
    )
    percentiles = [Percentile(percentile=level, value=100.0 * level) for level in STANDARD_PERCENTILES]
    diagnostics = StatedTailDiagnostics()
    distribution = build_numeric_distribution(
        percentiles,
        question,
        None,
        p_below_lower_bound=0.6,
        p_above_upper_bound=0.395,
        tail_diagnostics=diagnostics,
    )
    cdf = distribution.get_cdf()

    assert cdf[0].percentile == pytest.approx(diagnostics.lower_applied_probability, abs=1e-12)
    assert 1.0 - cdf[-1].percentile == pytest.approx(diagnostics.upper_applied_probability, abs=1e-12)
    assert_server_accepts_cdf(
        np.array([point.percentile for point in cdf]),
        cdf_size=question.cdf_size,
        open_lower=True,
        open_upper=True,
    )


def test_stated_tail_build_failure_keeps_member_without_stated_tails(monkeypatch: pytest.MonkeyPatch) -> None:
    question = NumericQuestion(
        question_text="How many?",
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=True,
        open_upper_bound=True,
    )
    percentiles = [Percentile(percentile=level, value=100.0 * level) for level in STANDARD_PERCENTILES]

    def fail_build(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("cannot build stated anchors")

    monkeypatch.setattr("metaculus_bot.numeric.pipeline.generate_pchip_cdf_with_smoothing", fail_build)
    diagnostics = StatedTailDiagnostics()
    prediction = build_numeric_distribution(
        percentiles, question, None, p_below_lower_bound=0.04, tail_diagnostics=diagnostics
    )
    assert diagnostics.stated_tails_discarded
    assert len(prediction.cdf) == question.cdf_size


@pytest.mark.parametrize("cdf_size", [9, 201])
def test_failed_stated_build_discards_diagnostics_and_retries(cdf_size: int, monkeypatch: pytest.MonkeyPatch) -> None:
    question = NumericQuestion(
        question_text="How many?",
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=True,
        open_upper_bound=True,
        cdf_size=cdf_size,
    )
    percentiles = [Percentile(percentile=level, value=100.0 * level) for level in STANDARD_PERCENTILES]
    diagnostics = StatedTailDiagnostics()
    target = "_build_discrete_distribution" if cdf_size != 201 else "generate_pchip_cdf_with_smoothing"
    original = getattr(pipeline, target)

    def fail_stated(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("p_above_upper_bound") is not None:
            diagnostics.upper_dropped_percentiles = (99.0,)
            diagnostics.upper_applied_probability = 0.1
            raise RuntimeError("stated build failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, target, fail_stated)
    prediction = build_numeric_distribution(
        percentiles, question, None, p_above_upper_bound=0.1, tail_diagnostics=diagnostics
    )
    assert len(prediction.get_cdf()) == cdf_size
    assert diagnostics.stated_tails_discarded
    assert diagnostics.upper_dropped_percentiles == ()
    assert diagnostics.upper_applied_probability is None


@pytest.mark.parametrize(("value", "expected"), [(" 0.08 ", 0.08), (True, None), ("nan", None), ("bad", None)])
def test_declared_tail_accepts_finite_numeric_strings(value: object, expected: float | None) -> None:
    assert _read_declared_tail_probability(value) == expected
