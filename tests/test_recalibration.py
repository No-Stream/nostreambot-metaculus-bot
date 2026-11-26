"""Offline forward-chaining recalibration tests."""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN, MC_PROB_MIN
from metaculus_bot.mc_processing import clamp_and_renormalize_probs
from metaculus_bot.performance_analysis.recalibration import (
    RECALIBRATION_ERAS,
    fit_binary_isotonic,
    fit_binary_platt,
    fit_mc_power,
    forward_chaining_recalibration,
    transform_mc_power,
)
from metaculus_bot.scoring_common import spot_peer_delta


def binary_record(index: int, era: str, probability: float = 0.8, outcome: bool = True) -> dict:
    return {
        "question_id": index,
        "bot_comment_created_at": (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index)).isoformat(),
        "type": "binary",
        "config_era": era if era in {"pre_flip", "post_flip"} else "triple_era",
        "triple_subera": era if era.startswith("triple_") else None,
        "our_prob_yes": probability,
        "resolution_parsed": outcome,
        "metadata": {"actual_resolve_time": f"2026-10-{index % 28 + 1:02d}T00:00:00Z"},
    }


def training_binary(era: str = "pre_flip") -> list[dict]:
    return [binary_record(index, era, 0.2 if index < 40 else 0.8, index % 4 == 0) for index in range(80)]


def mc_record(index: int, era: str, winner: str = "A") -> dict:
    record = binary_record(index, era)
    record.update(type="multiple_choice", options=["A", "B"], our_forecast_values=[0.8, 0.2], resolution_parsed=winner)
    return record


class TestFits:
    def test_platt_matches_two_group_binomial_mle(self) -> None:
        probabilities = [0.2] * 100 + [0.8] * 100
        outcomes = [True] * 30 + [False] * 70 + [True] * 60 + [False] * 40
        fitted = fit_binary_platt(probabilities, outcomes)
        assert fitted.predict(0.2) == pytest.approx(0.3, abs=1e-7)
        assert fitted.predict(0.8) == pytest.approx(0.6, abs=1e-7)
        assert fitted.predict(0) == BINARY_PROB_MIN
        assert fitted.predict(1) == BINARY_PROB_MAX

    def test_isotonic_pools_ties_and_uses_step_interpolation(self) -> None:
        fitted = fit_binary_isotonic([0.2, 0.8, 0.2, 0.8], [False, True, True, True])
        shuffled = fit_binary_isotonic([0.8, 0.2, 0.8, 0.2], [True, True, True, False])
        assert fitted == shuffled
        assert fitted.predict(0.1) == pytest.approx(0.5)
        assert fitted.predict(0.5) == pytest.approx(0.5)
        assert fitted.predict(0.8) == BINARY_PROB_MAX
        assert fitted.predict(1) == BINARY_PROB_MAX

    def test_power_mle_is_question_level_and_preserves_order(self) -> None:
        fitted = fit_mc_power([[0.8, 0.2]] * 100, [0] * 60 + [1] * 40)
        expected_alpha = math.log(0.6 / 0.4) / math.log(0.8 / 0.2)
        assert fitted == pytest.approx(expected_alpha, abs=1e-6)
        assert transform_mc_power([0.8, 0.2], fitted) == pytest.approx([0.6, 0.4], abs=1e-6)

    def test_power_floor_then_renormalize(self) -> None:
        transformed = transform_mc_power([0.98, 0.01, 0.01], 20)
        assert sum(transformed) == pytest.approx(1)
        powered = np.array([0.98**20, 0.01**20, 0.01**20])
        expected = clamp_and_renormalize_probs((powered / powered.sum()).tolist())
        assert transformed == pytest.approx(expected)

    def test_power_transform_matches_production_floor_repair(self) -> None:
        probabilities = [0.92] + [0.01] * 8

        transformed = transform_mc_power(probabilities, 2.0)

        assert all(probability >= MC_PROB_MIN for probability in transformed)
        powered = np.array([probability**2 for probability in probabilities])
        expected = clamp_and_renormalize_probs((powered / powered.sum()).tolist())
        assert transformed == pytest.approx(expected)


class TestForwardChaining:
    def test_first_era_late_half_trains_only_on_early_half(self) -> None:
        records = [binary_record(index, "pre_flip", 0.2 if index % 2 else 0.8, index % 3 == 0) for index in range(100)]
        records.append(binary_record(101, "post_flip", 0.8))
        result = forward_chaining_recalibration(records, draws=50)
        late_rows = [row for row in result.rows if row.era == "pre_flip (late half)"]
        assert len(late_rows) == 2
        assert all(row.n_train == 50 for row in late_rows)
        assert all(row.n_test == 50 for row in late_rows)
        assert all(row.train_eras == ("pre_flip (early half)",) for row in late_rows)

        changed_late_outcomes = [
            {**record, "resolution_parsed": not record["resolution_parsed"]}
            if record["config_era"] == "pre_flip" and record["question_id"] >= 50
            else record
            for record in records
        ]
        changed = forward_chaining_recalibration(changed_late_outcomes, draws=50)
        assert [row.parameters for row in late_rows] == [
            row.parameters for row in changed.rows if row.era == "pre_flip (late half)"
        ]
        assert result.first_era_split_boundary == "2026-01-01T00:00:49.500000+00:00"

    def test_first_era_split_boundary_is_deterministic(self) -> None:
        records = [binary_record(index, "pre_flip") for index in range(6)]
        forward = forward_chaining_recalibration(records, draws=50)
        reversed_result = forward_chaining_recalibration(list(reversed(records)), draws=50)
        assert forward.first_era_split_boundary == reversed_result.first_era_split_boundary
        assert forward.era_order == reversed_result.era_order
        assert forward.rows == reversed_result.rows

    def test_missing_first_era_forecast_timestamp_is_counted(self) -> None:
        record = binary_record(0, "pre_flip")
        record["bot_comment_created_at"] = None
        result = forward_chaining_recalibration([record], draws=50)
        assert result.exclusions["missing forecast timestamp"] == 1
        assert result.rows == ()

    def test_chronological_training_no_future_or_test_era_outcomes(self) -> None:
        records = [
            *(
                binary_record(index, "pre_flip", 0.2 if index % 80 < 40 else 0.8, index % 80 % 4 == 0)
                for index in range(160)
            ),
            binary_record(200, "post_flip", 0.8),
        ]
        future = [binary_record(200 + index, "triple_fall_config", 0.8, False) for index in range(80)]
        before = forward_chaining_recalibration(records, draws=50)
        after = forward_chaining_recalibration(list(reversed(records + future)), draws=50)
        assert before.era_order == RECALIBRATION_ERAS
        assert RECALIBRATION_ERAS.index("post_flip") < RECALIBRATION_ERAS.index("triple_pre_market")
        original = [row for row in before.rows if row.era == "post_flip"]
        added = [row for row in after.rows if row.era == "post_flip"]
        assert original == added
        for row in original:
            assert row.train_eras == ("pre_flip (early half)", "pre_flip (late half)")
            assert row.n_train == 160
            assert row.n_test == 1
            assert row.status == "evaluated"
            assert row.mean_delta == pytest.approx(spot_peer_delta(old_prob=0.8, new_prob=0.25, question_type="binary"))

    def test_minimum_training_skips_count_test_eras_and_questions(self) -> None:
        result = forward_chaining_recalibration([*training_binary()[:49], binary_record(100, "post_flip")])
        for pooled in result.pooled:
            if pooled.question_type == "binary":
                assert pooled.n_test == 0
                assert pooled.skipped_eras == 3
                assert pooled.skipped_questions == 50
        assert all(row.status == "insufficient training" for row in result.rows)

    def test_pooled_delta_and_interval_use_questions_across_test_eras(self) -> None:
        records = [
            binary_record(index, "pre_flip", 0.2 if index % 80 < 40 else 0.8, index % 80 % 4 == 0)
            for index in range(160)
        ]
        records += [binary_record(100 + index, "post_flip", outcome=index % 3 == 0) for index in range(20)]
        records += [binary_record(200 + index, "triple_pre_market", outcome=index % 2 == 0) for index in range(10)]
        result = forward_chaining_recalibration(records, draws=50)
        for pooled in result.pooled:
            if pooled.question_type != "binary":
                continue
            evaluated = [row for row in result.rows if row.method == pooled.method and row.status == "evaluated"]
            assert pooled.n_test == 110
            assert pooled.sum_delta == pytest.approx(
                sum(row.sum_delta for row in evaluated if row.sum_delta is not None)
            )
            assert pooled.mean_delta == pytest.approx(pooled.sum_delta / 110)
            assert pooled.interval is not None
        assert result == forward_chaining_recalibration(records, draws=50)

    def test_mc_uses_winner_probability_and_no_spot_peer_field(self) -> None:
        records = [mc_record(index, "pre_flip", "A" if index < 60 else "B") for index in range(100)]
        records += [mc_record(200, "post_flip", "B")]
        result = forward_chaining_recalibration(records)
        row = next(row for row in result.rows if row.era == "post_flip")
        assert row.n_train == 100
        assert row.n_test == 1
        assert row.mean_delta == pytest.approx(
            spot_peer_delta(old_prob=0.2, new_prob=0.4, question_type="multiple_choice"), abs=1e-4
        )

    def test_missing_eras_and_invalid_observations_are_counted(self) -> None:
        missing_era = binary_record(100, "pre_flip")
        missing_era["config_era"] = None
        missing_subera = binary_record(101, "triple_pre_market")
        missing_subera["triple_subera"] = None
        invalid = binary_record(102, "pre_flip", float("nan"))
        result = forward_chaining_recalibration([missing_era, missing_subera, invalid])
        assert result.exclusions == {"missing era": 2, "invalid binary forecast": 1}
        assert result.rows == ()

    @pytest.mark.parametrize(("field", "value"), [("config_era", "wrong"), ("triple_subera", "wrong")])
    def test_unknown_era_fails(self, field: str, value: str) -> None:
        record = binary_record(1, "triple_pre_market")
        record[field] = value
        with pytest.raises(ValueError, match="era"):
            forward_chaining_recalibration([record])


class TestHeldOutSynthetic:
    @pytest.mark.parametrize(("slope", "minimum_gain"), [(0.25, 5.0), (1.0, -0.3)])
    def test_seeded_binary_holdout(self, slope: float, minimum_gain: float) -> None:
        rng = np.random.default_rng(20261125)
        records = []
        for index in range(10000):
            probability = float(rng.uniform(0.02, 0.98))
            truth = 1 / (1 + math.exp(-slope * math.log(probability / (1 - probability))))
            records.append(
                binary_record(
                    index, "pre_flip" if index < 5000 else "post_flip", probability, bool(rng.random() < truth)
                )
            )
        result = forward_chaining_recalibration(records, draws=50)
        row = next(row for row in result.rows if row.method == "platt" and row.era == "post_flip")
        assert row.parameters is not None
        assert row.parameters["slope"] == pytest.approx(slope, abs=0.08)
        assert row.mean_delta is not None
        assert row.mean_delta > minimum_gain
        if slope == 1:
            assert abs(row.mean_delta) < 0.3

    def test_power_mle_can_reverse_option_order(self) -> None:
        alpha = fit_mc_power([[0.8, 0.2]] * 100, [0] * 40 + [1] * 60)
        assert alpha < 0
        assert transform_mc_power([0.8, 0.2], alpha) == pytest.approx([0.4, 0.6], abs=1e-6)
