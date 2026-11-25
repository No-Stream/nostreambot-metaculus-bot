"""The member-replay path: rebuilding one member's CDF, replaying a subset aggregate,
and detecting a score censored at the grid's constraint floor.

Every record here is synthetic, so nothing reads the archive or the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from metaculus_bot.aggregation_strategies import aggregate_binary_median
from metaculus_bot.numeric.config import OPEN_TAIL_MIN_MASS, PCHIP_CDF_POINTS, grid_step_constraints
from metaculus_bot.performance_analysis.member_replay import (
    CENSOR_RATIO_TOL,
    censoring_ratio,
    is_censored,
    is_replayable,
    mc_median_probs,
    member_cdf,
    member_scoring_inputs,
    postprocess_replay_cdf,
    record_cdf_size,
    replay_solo_score,
    replay_subset_score,
    score_member_cdf,
    theoretical_floor_score,
)
from metaculus_bot.scoring_common import binary_log_score, mc_log_score


def _numeric_record(
    resolution: float | str,
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 100.0,
    open_lower: bool = False,
    open_upper: bool = False,
    inbound_outcome_count: int | None = 200,
    q_type: str = "numeric",
) -> dict:
    scaling: dict = {"range_min": lower_bound, "range_max": upper_bound, "zero_point": None}
    if inbound_outcome_count is not None:
        scaling["inbound_outcome_count"] = inbound_outcome_count
    return {
        "question_id": 1,
        "post_id": 1,
        "type": q_type,
        "resolution_parsed": resolution,
        "scaling": scaling,
        "open_lower_bound": open_lower,
        "open_upper_bound": open_upper,
        "our_forecast_values": [i / 200 for i in range(201)],
    }


def _binary_record(*, resolution: bool = True) -> dict:
    return {"question_id": 2, "post_id": 2, "type": "binary", "resolution_parsed": resolution}


def _mc_record(options: list[str], resolved: str) -> dict:
    return {
        "question_id": 3,
        "post_id": 3,
        "type": "multiple_choice",
        "options": options,
        "resolution_parsed": resolved,
    }


def _curve(median: float, half_width: float) -> list[tuple[float, float]]:
    """A monotone 11-anchor curve centred on ``median``, dense enough to PCHIP."""
    labels = [2.5, 5.0, 10.0, 20.0, 40.0, 50.0, 60.0, 80.0, 90.0, 95.0, 97.5]
    return [(label, median + half_width * (label - 50.0) / 47.5) for label in labels]


class TestRecordCdfSize:
    """The grid a member's CDF is rebuilt on, which sets the log score's baseline."""

    def test_inbound_outcome_count_wins(self):
        assert record_cdf_size({"scaling": {"inbound_outcome_count": 41}}) == 42

    def test_falls_back_to_the_published_cdf_length(self):
        assert record_cdf_size({"scaling": {}, "our_forecast_values": [0.0] * 11}) == 11

    def test_falls_back_to_the_standard_grid(self):
        assert record_cdf_size({"scaling": {}}) == PCHIP_CDF_POINTS

    def test_a_published_list_too_short_to_be_a_cdf_is_not_a_grid_size(self):
        assert record_cdf_size({"scaling": {}, "our_forecast_values": [0.0, 1.0]}) == PCHIP_CDF_POINTS


class TestMemberCdf:
    """One member's declared curve, rebuilt on the record's own grid."""

    def test_built_on_the_records_own_grid(self):
        record = _numeric_record(50.0, inbound_outcome_count=10)
        inputs = member_scoring_inputs(record)
        assert inputs is not None
        assert len(member_cdf(_curve(50.0, 20.0), inputs)) == 11

    def test_n_points_overrides_the_records_grid(self):
        record = _numeric_record(50.0, inbound_outcome_count=10)
        inputs = member_scoring_inputs(record, n_points=PCHIP_CDF_POINTS)
        assert inputs is not None
        assert len(member_cdf(_curve(50.0, 20.0), inputs)) == PCHIP_CDF_POINTS

    def test_a_coarse_grid_scores_a_sharp_forecast_lower_than_the_201_default(self):
        record = _numeric_record(50.0, inbound_outcome_count=10)
        own = member_scoring_inputs(record)
        wide = member_scoring_inputs(record, n_points=PCHIP_CDF_POINTS)
        assert own is not None
        assert wide is not None
        on_own_grid = score_member_cdf(member_cdf(_curve(50.0, 5.0), own), own)
        on_201 = score_member_cdf(member_cdf(_curve(50.0, 5.0), wide), wide)
        assert on_201 > on_own_grid + 1.0

    def test_a_curve_with_one_anchor_cannot_be_rebuilt(self):
        record = _numeric_record(50.0)
        inputs = member_scoring_inputs(record)
        assert inputs is not None
        with pytest.raises((ValueError, RuntimeError)):
            member_cdf([(50.0, 50.0)], inputs)

    def test_no_scoring_inputs_without_bounds(self):
        assert member_scoring_inputs({"type": "numeric", "scaling": {}, "resolution_parsed": 5.0}) is None


class TestReplaySoloScore:
    """A member's own forecast, scored as the member declared it."""

    def test_binary_matches_the_shared_log_score(self):
        assert replay_solo_score(_binary_record(), 0.8) == pytest.approx(binary_log_score(0.8, True))

    def test_multiple_choice_renormalizes_without_clamping(self):
        record = _mc_record(["a", "b"], "a")
        expected = mc_log_score([0.6 / 0.9, 0.3 / 0.9], 0)
        assert replay_solo_score(record, {"a": 0.6, "b": 0.3}) == pytest.approx(expected)

    def test_multiple_choice_with_no_mass_is_unscoreable(self):
        assert replay_solo_score(_mc_record(["a", "b"], "a"), {"a": 0.0, "b": 0.0}) is None

    def test_a_tighter_curve_on_the_resolution_scores_better(self):
        record = _numeric_record(50.0)
        tight = replay_solo_score(record, _curve(50.0, 5.0))
        loose = replay_solo_score(record, _curve(50.0, 40.0))
        assert tight is not None
        assert loose is not None
        assert tight > loose

    def test_an_unbuildable_curve_is_none_rather_than_an_exception(self):
        assert replay_solo_score(_numeric_record(50.0), [(50.0, 50.0)]) is None


class TestReplaySubsetScore:
    """The aggregate of an arbitrary member subset, recombined as production recombines it."""

    def test_binary_median_matches_the_production_aggregator(self):
        record = _binary_record()
        members = {"a": 0.2, "b": 0.5, "c": 0.9}
        expected = binary_log_score(aggregate_binary_median([0.2, 0.5, 0.9]), True)
        assert replay_subset_score(record, members, ["a", "b", "c"], method="median") == pytest.approx(expected)

    def test_binary_mean_rounds_to_three_decimals_like_production(self):
        record = _binary_record()
        members = {"a": 0.1, "b": 0.2, "c": 0.4}
        assert replay_subset_score(record, members, ["a", "b", "c"], method="mean") == pytest.approx(
            binary_log_score(0.233, True)
        )

    def test_leaving_a_member_out_moves_the_aggregate(self):
        record = _binary_record()
        members = {"a": 0.2, "b": 0.5, "c": 0.9}
        full = replay_subset_score(record, members, ["a", "b", "c"], method="median")
        without_a = replay_subset_score(record, members, ["b", "c"], method="median")
        assert full is not None
        assert without_a is not None
        assert without_a > full

    def test_an_empty_subset_has_no_aggregate(self):
        assert replay_subset_score(_binary_record(), {"a": 0.5}, [], method="median") is None

    def test_an_unrecognized_method_raises(self):
        with pytest.raises(ValueError, match="unrecognized aggregation"):
            replay_subset_score(_binary_record(), {"a": 0.5}, ["a"], method="geometric")

    def test_multiple_choice_median_goes_through_the_production_clamp(self):
        record = _mc_record(["a", "b", "c"], "c")
        members = {model: {"a": 0.5, "b": 0.5, "c": 0.0} for model in ("x", "y")}
        probs = mc_median_probs(record, members, ["x", "y"])
        assert probs is not None
        assert probs[2] > 0.0

    def test_numeric_median_agrees_with_a_hand_built_pointwise_median(self):
        record = _numeric_record(50.0)
        inputs = member_scoring_inputs(record)
        assert inputs is not None
        members = {"a": _curve(45.0, 10.0), "b": _curve(50.0, 10.0), "c": _curve(55.0, 10.0)}
        stacked = np.vstack([member_cdf(members[model], inputs) for model in ("a", "b", "c")])
        expected = score_member_cdf(postprocess_replay_cdf(np.median(stacked, axis=0), inputs), inputs)
        assert replay_subset_score(record, members, ["a", "b", "c"], method="median") == pytest.approx(expected)

    def test_a_member_that_cannot_be_rebuilt_is_dropped_from_the_aggregate(self):
        record = _numeric_record(50.0)
        members = {"good": _curve(50.0, 10.0), "broken": [(50.0, 50.0)]}
        both = replay_subset_score(record, members, ["good", "broken"], method="median")
        alone = replay_subset_score(record, members, ["good"], method="median")
        assert both is not None
        assert both == pytest.approx(alone)

    def test_a_subset_with_nothing_rebuildable_has_no_aggregate(self):
        record = _numeric_record(50.0)
        assert replay_subset_score(record, {"broken": [(50.0, 50.0)]}, ["broken"], method="median") is None


class TestPostprocessReplayCdf:
    """A pointwise aggregate legalized the way the publish path legalizes one."""

    def test_open_tails_are_pinned_to_the_servers_minimum(self):
        record = _numeric_record(50.0, open_lower=True, open_upper=True)
        inputs = member_scoring_inputs(record)
        assert inputs is not None
        legal = postprocess_replay_cdf(np.linspace(0.0, 1.0, 201), inputs)
        assert legal[0] >= OPEN_TAIL_MIN_MASS
        assert legal[-1] <= 1.0 - OPEN_TAIL_MIN_MASS

    def test_closed_tails_are_pinned_to_zero_and_one(self):
        record = _numeric_record(50.0)
        inputs = member_scoring_inputs(record)
        assert inputs is not None
        legal = postprocess_replay_cdf(np.linspace(0.2, 0.8, 201), inputs)
        assert legal[0] == 0.0
        assert legal[-1] == 1.0

    def test_a_degenerate_step_function_comes_back_inside_the_grids_step_limits(self):
        record = _numeric_record(50.0)
        inputs = member_scoring_inputs(record)
        assert inputs is not None
        min_step, max_step = grid_step_constraints(201)
        cliff = np.concatenate([np.zeros(100), np.ones(101)])
        steps = np.diff(postprocess_replay_cdf(cliff, inputs))
        assert steps.min() >= min_step - 1e-12
        assert steps.max() <= max_step + 1e-9

    def test_a_non_monotone_aggregate_is_made_monotone(self):
        record = _numeric_record(50.0)
        inputs = member_scoring_inputs(record)
        assert inputs is not None
        wobbly = np.linspace(0.0, 1.0, 201).copy()
        wobbly[100] = 0.1
        assert np.all(np.diff(postprocess_replay_cdf(wobbly, inputs)) >= 0.0)


class TestCensoring:
    """Whether a member's numeric log score is pinned at the grid's constraint floor."""

    def _floor_cdf(self, n_points: int, floor_bin: int) -> np.ndarray:
        """A CDF holding exactly the min step on ``floor_bin`` and the rest spread evenly."""
        min_step, _ = grid_step_constraints(n_points)
        steps = np.full(n_points - 1, (1.0 - min_step) / (n_points - 2))
        steps[floor_bin] = min_step
        return np.concatenate([[0.0], np.cumsum(steps)])

    def test_a_member_pinned_at_the_floor_is_flagged(self):
        record = _numeric_record(50.0, inbound_outcome_count=200)
        inputs = member_scoring_inputs(record)
        assert inputs is not None
        cdf = self._floor_cdf(201, 99)
        assert censoring_ratio(cdf, inputs) == pytest.approx(1.0)
        assert is_censored(cdf, inputs) is True

    def test_the_floor_bin_next_to_the_resolution_is_not_the_resolutions_bin(self):
        record = _numeric_record(50.0, inbound_outcome_count=200)
        inputs = member_scoring_inputs(record)
        assert inputs is not None
        assert is_censored(self._floor_cdf(201, 98), inputs) is False

    def test_a_bad_but_informative_member_is_not_flagged(self):
        record = _numeric_record(50.0)
        inputs = member_scoring_inputs(record)
        assert inputs is not None
        far_off = member_cdf(_curve(95.0, 3.0), inputs)
        ratio = censoring_ratio(far_off, inputs)
        assert ratio is not None
        assert is_censored(far_off, inputs) is (ratio <= CENSOR_RATIO_TOL)

    def test_a_flat_forecast_sits_far_above_the_floor(self):
        record = _numeric_record(50.0)
        inputs = member_scoring_inputs(record)
        assert inputs is not None
        flat = np.linspace(0.0, 1.0, 201)
        assert censoring_ratio(flat, inputs) == pytest.approx(1.0 / grid_step_constraints(201)[0] / 200.0)
        assert is_censored(flat, inputs) is False

    def test_a_closed_bound_has_no_floor_to_be_censored_against(self):
        record = _numeric_record("above_upper_bound", open_upper=False)
        inputs = member_scoring_inputs(record)
        assert inputs is not None
        assert censoring_ratio(np.linspace(0.0, 1.0, 201), inputs) is None
        assert is_censored(np.linspace(0.0, 1.0, 201), inputs) is None

    def test_an_open_upper_tail_is_censored_against_the_open_tail_minimum(self):
        record = _numeric_record("above_upper_bound", open_upper=True)
        inputs = member_scoring_inputs(record)
        assert inputs is not None
        cdf = np.linspace(0.0, 1.0 - OPEN_TAIL_MIN_MASS, 201)
        assert censoring_ratio(cdf, inputs) == pytest.approx(1.0)
        assert is_censored(cdf, inputs) is True


class TestTheoreticalFloorScore:
    """The lowest score the server's own constraints permit, reported for context."""

    def test_an_interior_bin_uses_the_grids_min_step(self):
        record = _numeric_record(50.0)
        n_inbound = PCHIP_CDF_POINTS - 1
        min_step, _ = grid_step_constraints(PCHIP_CDF_POINTS)
        expected = 50.0 * np.log(min_step / (1.0 / n_inbound))
        assert theoretical_floor_score(record) == pytest.approx(expected)

    def test_the_interior_floor_is_the_same_on_every_grid(self):
        """The server's min step is 0.01/inbound, which cancels the 1/inbound uniform baseline."""
        coarse = _numeric_record(50.0, inbound_outcome_count=10)
        standard = _numeric_record(50.0)
        assert theoretical_floor_score(coarse) == pytest.approx(theoretical_floor_score(standard))
        assert theoretical_floor_score(standard) == pytest.approx(50.0 * np.log(0.01))

    def test_an_out_of_range_resolution_under_a_closed_bound_has_no_floor(self):
        assert theoretical_floor_score(_numeric_record("above_upper_bound", open_upper=False)) is None

    def test_an_out_of_range_resolution_under_an_open_bound_has_the_boundary_floor(self):
        record = _numeric_record("above_upper_bound", open_upper=True)
        assert theoretical_floor_score(record) == pytest.approx(50.0 * np.log(OPEN_TAIL_MIN_MASS / 0.05))

    def test_no_floor_without_scoring_inputs(self):
        assert theoretical_floor_score({"type": "numeric", "scaling": {}, "resolution_parsed": 5.0}) is None


class TestIsReplayable:
    """Which archived records carry a resolution a forecast can be scored against."""

    def test_a_resolved_binary_is_replayable(self):
        assert is_replayable(_binary_record()) is True

    def test_an_unresolved_binary_is_not(self):
        assert is_replayable({"type": "binary", "resolution_parsed": None}) is False

    def test_a_multiple_choice_resolving_to_a_declared_option_is_replayable(self):
        assert is_replayable(_mc_record(["a", "b"], "a")) is True

    def test_a_multiple_choice_resolving_off_its_option_list_is_not(self):
        assert is_replayable(_mc_record(["a", "b"], "c")) is False

    def test_a_numeric_with_bounds_and_a_number_is_replayable(self):
        assert is_replayable(_numeric_record(50.0)) is True

    def test_a_discrete_record_counts_as_numeric(self):
        assert is_replayable(_numeric_record(50.0, q_type="discrete")) is True

    def test_a_numeric_without_bounds_is_not(self):
        assert is_replayable({"type": "numeric", "scaling": {}, "resolution_parsed": 5.0}) is False

    def test_an_unrecognized_type_is_not(self):
        assert is_replayable({"type": "date", "resolution_parsed": 5.0}) is False


def test_member_replay_uses_fenced_stated_tail() -> None:
    inputs = member_scoring_inputs(_numeric_record("above_upper_bound", open_upper=True))
    assert inputs is not None
    body = '```json\n{"question_type": "numeric", "prob_above_upper_bound": 0.10, "declared_percentiles": {"0.1": 30, "0.5": 50, "0.9": 70}}\n```'
    cdf = member_cdf(_curve(50.0, 20.0), inputs, reasoning=body)
    assert 1.0 - cdf[-1] == pytest.approx(0.10, abs=1e-8)


def test_solo_and_subset_replay_attribute_stated_tail_to_the_member() -> None:
    record = _numeric_record("above_upper_bound", open_upper=True)
    body = (
        '```json\n{"question_type": "numeric", "prob_above_upper_bound": 0.10, '
        '"declared_percentiles": {"0.1": 30, "0.5": 50, "0.9": 70}}\n```'
    )
    record["comment_text"] = f"## R1: Forecaster 1 Reasoning\nModel: openrouter/test/model\n\n{body}"
    curve = _curve(50.0, 20.0)
    inputs = member_scoring_inputs(record)
    assert inputs is not None
    expected = score_member_cdf(member_cdf(curve, inputs, reasoning=body), inputs)
    assert replay_solo_score(record, curve, model="model") == pytest.approx(expected)
    legacy_score = replay_solo_score(record, curve)
    assert legacy_score is not None
    assert legacy_score < expected
    subset_score = replay_subset_score(record, {"model": curve}, ["model"], method="median")
    cdf = member_cdf(curve, inputs, reasoning=body)
    expected_subset = score_member_cdf(postprocess_replay_cdf(np.asarray(cdf), inputs), inputs)
    assert subset_score == pytest.approx(expected_subset)
