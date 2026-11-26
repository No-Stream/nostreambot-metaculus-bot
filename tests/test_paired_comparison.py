from __future__ import annotations

import numpy as np
import pytest

from metaculus_bot.performance_analysis.paired_comparison import (
    Likelihood,
    PairedScoreRow,
    bayesian_bootstrap,
    compare_paired,
    fit_model,
    leave_one_cluster_out,
    prepare_paired_data,
    sample_prior,
    sample_size_statement,
    summarize_fit,
)
from metaculus_bot.performance_analysis.posterior_summary import sampling_diagnostics


def rows_for(
    values: list[float], *, clusters: list[str] | None = None, members: list[str] | None = None
) -> list[PairedScoreRow]:
    clusters = clusters or [f"event-{index // 4}" for index in range(len(values))]
    return [
        PairedScoreRow(
            f"unit-{index}",
            clusters[index],
            ("binary", "discrete", "multiple_choice", "numeric")[index % 4],
            0,
            value,
            None if members is None else members[index],
        )
        for index, value in enumerate(values)
    ]


def test_prepare_indexes_units_clusters_groups_and_members() -> None:
    data = prepare_paired_data(
        [
            PairedScoreRow("u1", "event-b", "binary", 0, 1.0, "model-a"),
            PairedScoreRow("u1", "event-b", "binary", 1, 2.0, "model-a"),
            PairedScoreRow("u2", "event-a", "numeric", 0, 3.0, "model-b"),
        ]
    )
    assert data.units == ("u1", "u2")
    assert data.clusters == ("event-a", "event-b")
    assert data.groups == ("binary", "numeric")
    assert data.members == ("model-a", "model-b")
    assert data.unit_index.tolist() == [0, 0, 1]
    assert data.cluster_index.tolist() == [1, 1, 0]
    assert data.group_index.tolist() == [0, 0, 1]
    assert data.member_index is not None
    assert data.member_index.tolist() == [0, 0, 1]
    assert data.differences.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [PairedScoreRow("u", "e", "g", 0, float("nan"))],
        [PairedScoreRow("u", "e1", "g", 0, 1), PairedScoreRow("u", "e2", "g", 1, 1)],
        [PairedScoreRow("u", "e", "g1", 0, 1), PairedScoreRow("u", "e", "g2", 1, 1)],
        [PairedScoreRow("u", "e", "g", 0, 1), PairedScoreRow("u", "e", "g", 0, 2)],
        [PairedScoreRow("u1", "e", "g", 0, 1, "m"), PairedScoreRow("u2", "e", "g", 0, 1)],
    ],
)
def test_prepare_rejects_bad_rows(rows: list[PairedScoreRow]) -> None:
    with pytest.raises(ValueError, match=r".+"):
        prepare_paired_data(rows)


def test_prior_predictive_differences_are_believable() -> None:
    paired_rows = [
        PairedScoreRow(
            f"unit-{index // 2}",
            f"event-{index // 8}",
            ("binary", "discrete", "multiple_choice", "numeric")[index // 2 % 4],
            index % 2,
            0.0,
        )
        for index in range(58 * 2)
    ]
    data = prepare_paired_data(paired_rows)
    inference = sample_prior(data, likelihood="student_t", draws=2_000, seed=17)
    effect = inference.prior["effect"].values.ravel()
    differences = inference.prior_predictive["difference"].values.ravel()
    assert 0.03 < np.mean(np.abs(effect) > 10) < 0.07
    assert 5 < np.median(np.abs(differences)) < 20
    assert np.mean(np.abs(differences) > 100) < 0.02


@pytest.mark.slow
@pytest.mark.parametrize("likelihood", ["student_t", "normal"])
def test_recovers_a_planted_effect(likelihood: Likelihood) -> None:
    rng = np.random.default_rng(14)
    paired_rows = [
        PairedScoreRow(
            f"unit-{index // 2}",
            f"event-{index // 8}",
            ("binary", "discrete", "multiple_choice", "numeric")[index // 2 % 4],
            index % 2,
            float(value),
        )
        for index, value in enumerate(6 + rng.normal(0, 3, 40 * 2))
    ]
    data = prepare_paired_data(paired_rows)
    inference = fit_model(data, likelihood=likelihood, draws=300, tune=300, chains=2, seed=123)
    report = summarize_fit(data, inference, likelihood=likelihood, allow_diagnostics_warnings=True)
    assert report["effect"]["interval_90"][0] < 6 < report["effect"]["interval_90"][1]
    assert report["effect"]["p_above_zero"] > 0.95


@pytest.mark.slow
def test_a_null_effect_interval_contains_zero() -> None:
    rng = np.random.default_rng(15)
    paired_rows = [
        PairedScoreRow(
            f"unit-{index // 2}",
            f"event-{index // 8}",
            ("binary", "discrete", "multiple_choice", "numeric")[index // 2 % 4],
            index % 2,
            float(value),
        )
        for index, value in enumerate(rng.normal(0, 3, 40 * 2))
    ]
    data = prepare_paired_data(paired_rows)
    inference = fit_model(data, likelihood="student_t", draws=300, tune=300, chains=2, seed=124)
    report = summarize_fit(data, inference, likelihood="student_t", allow_diagnostics_warnings=True)
    assert report["effect"]["interval_90"][0] <= 0 <= report["effect"]["interval_90"][1]


@pytest.mark.slow
def test_the_normal_fit_follows_rare_large_reversals_more_than_student_t() -> None:
    rng = np.random.default_rng(16)
    values = [*rng.normal(0, 3, 40).tolist(), 80.0, 80.0, 80.0]
    result = compare_paired(rows_for(values), draws=300, tune=300, chains=2, seed=125, allow_diagnostics_warnings=True)
    assert result["normal"]["effect"]["mean"] > result["student_t"]["effect"]["mean"]
    assert result["fits_disagree"] is True
    assert [row["unit_id"] for row in result["influential_units"][:3]] == ["unit-40", "unit-41", "unit-42"]


@pytest.mark.slow
def test_member_offsets_are_estimated_when_rows_carry_members() -> None:
    paired_rows = [
        PairedScoreRow(
            f"unit-{index}",
            f"event-{index // 4}",
            ("binary", "discrete", "multiple_choice", "numeric")[index % 4],
            0,
            (2 if member == "model-a" else -2),
            member,
        )
        for index in range(30)
        for member in ("model-a", "model-b")
    ]
    data = prepare_paired_data(paired_rows)
    inference = fit_model(data, likelihood="normal", draws=300, tune=300, chains=2, seed=126)
    result = summarize_fit(data, inference, likelihood="normal", allow_diagnostics_warnings=True)
    assert set(result["members"]) == {"model-a", "model-b"}


def test_bootstrap_resamples_events_not_questions() -> None:
    values = [0.0, 0.0, 10.0, 10.0, 0.0, 0.0]
    shared_events = rows_for(values, clusters=["a", "a", "b", "b", "c", "c"])
    separate_events = rows_for(values)
    shared = bayesian_bootstrap(shared_events, draws=5_000, seed=42)
    shared_again = bayesian_bootstrap(shared_events, draws=5_000, seed=42)
    separate = bayesian_bootstrap(separate_events, draws=5_000, seed=42)
    assert shared == shared_again
    assert np.std(shared["effect"]["draws"]) > np.std(separate["effect"]["draws"])


def test_bootstrap_probabilities_on_constant_data() -> None:
    result = bayesian_bootstrap(rows_for([3.0] * 20), draws=100, seed=42)
    assert result["effect"]["mean"] == 3.0
    assert result["effect"]["p_above_zero"] == 1.0
    assert result["effect"]["p_gain_above"]["2.0"] == 1.0
    assert result["effect"]["p_loss_above"]["2.0"] == 0.0


def test_leave_one_cluster_out_names_the_outlier_event() -> None:
    result = leave_one_cluster_out(rows_for([1.0, 2.0, 3.0, 20.0], clusters=["a", "b", "c", "outlier-event"]))
    assert result[0]["cluster"] == "outlier-event"


def test_sample_size_arithmetic() -> None:
    assert sample_size_statement(16.0, 36.0)["required_units"] == 23


def test_failed_diagnostics_raise_unless_allowed() -> None:
    import arviz as az

    inference = az.from_dict(
        {
            "posterior": {"effect": np.array([[0.0] * 20, [10.0] * 20])},
            "sample_stats": {"diverging": np.zeros((2, 20), dtype=bool)},
        }
    )
    with pytest.raises(ValueError, match="diagnostics failed"):
        sampling_diagnostics(inference, allow_diagnostics_warnings=False)
    assert sampling_diagnostics(inference, allow_diagnostics_warnings=True)["passed"] is False
