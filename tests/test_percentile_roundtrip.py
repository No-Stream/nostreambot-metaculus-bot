"""Round-trip telemetry through real percentile builds."""

import logging
from pathlib import Path

import numpy as np
import pytest
from forecasting_tools import NumericQuestion, Percentile

from metaculus_bot.forecaster_runners import build_guarded_numeric_distribution
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.date_axis import as_epoch_question
from metaculus_bot.numeric.roundtrip import format_percentile_roundtrip_marker
from scripts.telemetry.markers import MARKER_SPECS, parse_log_text
from tests.pipeline_test_helpers import (
    distribution_from_heights,
    make_count_question,
    make_real_date_question,
    make_real_numeric_question,
)


def _points(values: list[float]) -> list[Percentile]:
    return [Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, values, strict=True)]


def _records(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return parse_log_text(
        caplog.text, run_id="1", workflow="test", artifact="test", run_date="2026-10-06", log_file="test.log"
    )["percentile_roundtrip"]


def test_count_declaration_survives_grid_replacement(caplog: pytest.LogCaptureFixture) -> None:
    question = make_count_question(35, page_url="https://www.metaculus.com/questions/700/")
    stated = _points([0, 0, 0, 0, 0, 0, 2, 3, 7, 12, 18, 25, 32])
    with caplog.at_level(logging.INFO):
        member = build_guarded_numeric_distribution(stated, question, model_name="test member")
    (record,) = _records(caplog)
    assert "stage" not in record
    assert record["qtype"] == "discrete"
    assert record["platform"] == "metaculus"
    assert record["model"] == "test member"
    assert record["point_count"] == len(stated)
    assert record["max_abs_drift"] == pytest.approx(0.004248159, abs=1e-9)
    assert record["p"] == 0.4
    assert record["v"] == 0
    assert record["cdf_at_v"] == pytest.approx(0.395751841, abs=1e-9)
    for probability, expected_drift in [(0.1, 0.0), (0.2, 0.0), (0.4, 0.004248159)]:
        line = format_percentile_roundtrip_marker(
            member, [Percentile(percentile=probability, value=0)], question, model="plateau"
        )
        point_record = parse_log_text(
            line, run_id="1", workflow="test", artifact="test", run_date="2026-10-06", log_file="test.log"
        )["percentile_roundtrip"][0]
        assert point_record["max_abs_drift"] == pytest.approx(expected_drift, abs=1e-9)
    assert record["max_abs_drift"] == pytest.approx(abs(record["cdf_at_v"] - record["p"]), abs=1e-6)


def test_linear_continuous_roundtrip(caplog: pytest.LogCaptureFixture) -> None:
    question = make_real_numeric_question(upper_bound=100, open_upper_bound=False)
    stated = _points([100 * p for p in STANDARD_PERCENTILES])
    with caplog.at_level(logging.INFO):
        build_guarded_numeric_distribution(stated, question, model_name="linear")
    (record,) = _records(caplog)
    assert record["max_abs_drift"] == pytest.approx(0, abs=1e-6)
    assert record["qtype"] == "numeric"


def test_fine_count_maxstep_clip(caplog: pytest.LogCaptureFixture) -> None:
    question = make_count_question(250, page_url="https://www.metaculus.com/questions/700/")
    stated = _points([0 if p <= 0.6 else 249 * p for p in STANDARD_PERCENTILES])
    with caplog.at_level(logging.INFO):
        build_guarded_numeric_distribution(stated, question, model_name="spike")
    assert "CDF_MAXSTEP_CLIP:" in caplog.text
    (record,) = _records(caplog)
    assert record["max_abs_drift"] == pytest.approx(0.44, abs=1e-9)
    assert record["p"] == 0.6
    assert record["v"] == 0
    assert record["cdf_at_v"] == pytest.approx(0.16, abs=1e-9)
    assert record["max_abs_drift"] == pytest.approx(abs(record["cdf_at_v"] - record["p"]), abs=1e-6)


def test_marker_registry_and_documentation() -> None:
    spec = next(spec for spec in MARKER_SPECS if spec.name == "percentile_roundtrip")
    question: NumericQuestion = make_real_numeric_question(upper_bound=100, open_upper_bound=False)
    stated = _points([100 * p for p in STANDARD_PERCENTILES])
    prediction = build_guarded_numeric_distribution(stated, question, model_name="linear")
    line = format_percentile_roundtrip_marker(prediction, stated, question, model="linear")
    match = spec.regex.search(line)
    assert match is not None
    assert float(match["max_abs_drift"]) == pytest.approx(0, abs=1e-6)
    assert "stage" not in match.groupdict()
    assert spec.qid_kind == "question_id"
    documentation = (Path(__file__).parents[1] / "docs/telemetry_markers.md").read_text()
    marker_documentation = documentation.split("### PERCENTILE_ROUNDTRIP", maxsplit=1)[1]
    assert "build_guarded_numeric_distribution" in marker_documentation
    assert "_aggregate_predictions" not in marker_documentation
    assert "stage=" not in marker_documentation
    assert "DiscreteQuestion" in marker_documentation
    assert "NUMERIC_AGGREGATE" in marker_documentation


def test_date_axis_uses_epoch_values(caplog: pytest.LogCaptureFixture) -> None:
    question = as_epoch_question(make_real_date_question(cdf_size=201))
    stated = _points(
        [question.lower_bound + p * (question.upper_bound - question.lower_bound) for p in STANDARD_PERCENTILES]
    )
    with caplog.at_level(logging.INFO):
        build_guarded_numeric_distribution(stated, question, model_name="date")
    (record,) = _records(caplog)
    assert record["qtype"] == "date"
    assert record["max_abs_drift"] == pytest.approx(0, abs=1e-6)


def test_log_grid_evaluation_uses_built_value_axis() -> None:
    question = make_real_numeric_question(lower_bound=1, upper_bound=1000, zero_point=0, open_upper_bound=False)
    stated = _points([1000**p for p in STANDARD_PERCENTILES])
    member = build_guarded_numeric_distribution(stated, question, model_name="log")
    # Between actual geometric grid points, the shared PIT helper interpolates in value space.
    grid = member.get_cdf()
    value = (grid[70].value + grid[71].value) / 2
    target = (grid[70].percentile + grid[71].percentile) / 2
    line = format_percentile_roundtrip_marker(member, [Percentile(percentile=0.3, value=value)], question, model="log")
    record = parse_log_text(
        line, run_id="1", workflow="test", artifact="test", run_date="2026-10-06", log_file="test.log"
    )["percentile_roundtrip"][0]
    assert record["cdf_at_v"] == pytest.approx(target, abs=1e-9)
    assert record["max_abs_drift"] == pytest.approx(abs(target - 0.3), abs=1e-9)


@pytest.mark.parametrize(
    ("value", "probability", "expected_height"),
    [
        (-0.5, 0.15, 0.15),
        (0.5, 0.3, 0.3),
        (0.5, 0.1, 0.2),
        (-1, 0.05, 0.05),
        (-1, 0.2, 0.1),
        (3.5, 0.95, 0.95),
        (3.5, 0.6, 0.8),
        (4, 0.6, 0.8),
        (1, 0.7, 0.5),
    ],
)
def test_discrete_bin_edges_and_open_tails(value: float, probability: float, expected_height: float) -> None:
    question = make_count_question(4, open_lower=True)
    member = distribution_from_heights(np.array([0.1, 0.2, 0.5, 0.7, 0.8]), question)
    line = format_percentile_roundtrip_marker(
        member, [Percentile(percentile=probability, value=value)], question, model="edges"
    )
    record = parse_log_text(
        line, run_id="1", workflow="test", artifact="test", run_date="2026-10-06", log_file="test.log"
    )["percentile_roundtrip"][0]
    assert record["platform"] == "mantic"
    assert record["cdf_at_v"] == pytest.approx(expected_height, abs=1e-9)
    assert record["max_abs_drift"] == pytest.approx(abs(expected_height - probability), abs=1e-9)


def test_log_discrete_uses_actual_bin_edges() -> None:
    question = make_count_question(35).model_copy(update={"lower_bound": 1.0, "upper_bound": 1000.0, "zero_point": 0.0})
    stated = _points([1000**p for p in STANDARD_PERCENTILES])
    member = build_guarded_numeric_distribution(stated, question, model_name="log discrete")
    grid = member.get_cdf()
    probability = grid[10].percentile + (grid[11].percentile - grid[10].percentile) / 4
    value = (grid[10].value + grid[11].value) / 2
    line = format_percentile_roundtrip_marker(
        member, [Percentile(percentile=probability, value=value)], question, model="log discrete"
    )
    record = parse_log_text(
        line, run_id="1", workflow="test", artifact="test", run_date="2026-10-06", log_file="test.log"
    )["percentile_roundtrip"][0]
    assert record["cdf_at_v"] == pytest.approx(probability, abs=1e-9)
    assert record["max_abs_drift"] == 0
