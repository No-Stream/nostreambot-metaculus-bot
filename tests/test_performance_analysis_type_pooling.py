import json

import arviz as az
import numpy as np
import pytest

from metaculus_bot.performance_analysis.cohorts import EXCLUSION_COHORTS
from metaculus_bot.performance_analysis.type_pooling import (
    _diagnostics,
    fit_model,
    make_report,
    prepare_data,
    render_report,
)


def record(qid: int, qtype: str, era: str, score: float | None, submitted: str | None, resolved: str | None) -> dict:
    return {
        "question_id": qid,
        "type": qtype,
        "config_era": era,
        "metaculus_scores": {"spot_peer_score": score},
        "bot_comment_created_at": submitted,
        "metadata": {"actual_resolve_time": resolved},
    }


def test_preparation_indexes_scores_days_and_missing_horizons() -> None:
    rows = [
        record(1, "binary", "old", 5, "2026-01-01T00:00:00Z", "2026-01-03T01:00:00Z"),
        record(2, "multiple_choice", "new", -3, None, "2026-01-03T23:00:00Z"),
        record(3, "binary", "new", None, None, None),
        record(4, "binary", "old", 9, "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z"),
    ]
    data = prepare_data(rows, era_field="config_era")
    assert data.types == ("binary", "multiple_choice")
    assert data.eras == ("new", "old")
    assert data.type_index.tolist() == [0, 1, 0]
    assert data.era_index.tolist() == [1, 0, 1]
    assert data.day_index.tolist() == [0, 0, 1]
    assert data.horizon_observed.tolist() == [True, False, True]
    assert data.standardized_horizon[1] == 0
    assert data.dropped_missing_score == 1
    assert data.missing_horizon == 1


def test_selection_and_invalid_horizon() -> None:
    rows = [record(1, "binary", "old", 2, "2026-01-05T00:00:00Z", "2026-01-03T00:00:00Z")]
    with pytest.raises(ValueError, match="horizon"):
        prepare_data(rows, era_field="config_era")
    with pytest.raises(ValueError, match=r"no.*records"):
        prepare_data(rows, era_field="config_era", eras=["other"])


@pytest.mark.slow
def test_synthetic_interaction_recovery() -> None:
    rows = []
    for era, qtype, mean in [
        ("old", "binary", 0),
        ("old", "multiple_choice", 0),
        ("new", "binary", 0),
        ("new", "multiple_choice", 50),
    ]:
        for index in range(20):
            rows.append(
                record(
                    len(rows) + 1,
                    qtype,
                    era,
                    mean + (index % 5 - 2),
                    "2026-01-01T00:00:00Z",
                    f"2026-01-{3 + index % 10:02d}T00:00:00Z",
                )
            )
    data = prepare_data(rows)
    inference = fit_model(data, draws=200, tune=200, chains=2, seed=1234)
    report = make_report(data, inference, allow_diagnostics_warnings=True)
    contrast = next(row for row in report["contrasts"] if row["era"] == "new")
    assert contrast["interval_95"][0] > 20
    assert contrast["probability_below_zero"] < 0.01
    interaction = inference.posterior["interaction"].values
    binary, multiple_choice = data.types.index("binary"), data.types.index("multiple_choice")
    old, new = data.eras.index("old"), data.eras.index("new")
    difference_in_differences = (
        interaction[..., multiple_choice, new]
        - interaction[..., binary, new]
        - interaction[..., multiple_choice, old]
        + interaction[..., binary, old]
    )
    assert np.quantile(difference_in_differences, 0.025) > 20
    assert "multiple_choice" in render_report(report)


def test_strict_exclusion_and_utc_day() -> None:
    excluded_id = int(next(iter(next(cohort for cohort in EXCLUSION_COHORTS.values() if cohort))))
    rows = [
        record(excluded_id, "binary", "old", 3, "2026-01-01T00:00:00Z", "2026-01-03T23:00:00-02:00"),
        record(999999999, "binary", "old", 4, "2026-01-01T00:00:00Z", "2026-01-04T01:00:00Z"),
    ]
    data = prepare_data(rows, strict=True)
    assert data.excluded == 1
    assert data.scores.tolist() == [4]
    assert data.days == ("2026-01-04",)


def test_missing_resolution_day_fails() -> None:
    with pytest.raises(ValueError, match="resolution time"):
        prepare_data([record(1, "binary", "old", 3, "2026-01-01T00:00:00Z", None)])


def test_diagnostic_failure_and_warning_override() -> None:
    inference = az.from_dict(
        {
            "posterior": {"alpha": np.array([[0.0] * 20, [10.0] * 20])},
            "sample_stats": {"diverging": np.zeros((2, 20), dtype=bool)},
        }
    )
    with pytest.raises(ValueError, match="diagnostics failed"):
        _diagnostics(inference, allow_diagnostics_warnings=False)
    result = _diagnostics(inference, allow_diagnostics_warnings=True)
    assert result["passed"] is False
    json.dumps(result, allow_nan=False)
