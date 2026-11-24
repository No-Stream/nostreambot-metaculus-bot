"""Stated tail probabilities through the real forecast and aggregation pipeline.

Only the external research and LLM calls are replaced. The framework lifecycle,
member fan-out, extraction, percentile repair, CDF construction, final combination
and report construction all run as they do for a forecast.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
from forecasting_tools import GeneralLlm, NumericQuestion
from forecasting_tools.data_models.numeric_report import NumericReport
from forecasting_tools.data_models.questions import DateQuestion

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import METACULUS_OUT_OF_RANGE_TAIL_FLOOR
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.date_axis import as_epoch_question
from tests.pipeline_test_helpers import (
    assert_server_accepts_cdf,
    cdf_heights,
    make_count_question,
    make_e2e_bot,
    make_real_date_question,
    make_real_numeric_question,
    metaculus_url,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.e2e]


def _percentiles(question: NumericQuestion) -> dict[str, float]:
    span = question.upper_bound - question.lower_bound
    return {
        str(percentile): question.lower_bound + span * (0.1 + 0.8 * index / (len(STANDARD_PERCENTILES) - 1))
        for index, percentile in enumerate(STANDARD_PERCENTILES)
    }


def _response(
    question: NumericQuestion,
    *,
    tails: dict[str, object] | None = None,
    outcome_type: str = "continuous",
    percentiles: dict[str, float] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "question_type": "numeric",
        "declared_percentiles": _percentiles(question) if percentiles is None else percentiles,
        "outcome_type": outcome_type,
    }
    if tails is not None:
        payload.update(tails)
    return "The forecast is centered in the displayed range.\n\n```json\n" + json.dumps(payload) + "\n```"


async def _forecast(
    question: NumericQuestion | DateQuestion,
    response: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[NumericReport, list[str]]:
    bot = make_e2e_bot(AggregationStrategy.MEDIAN, n_forecasters=2, min_forecasters_to_publish=2)
    monkeypatch.setattr(
        bot, "run_research", AsyncMock(return_value="Historical observations span the displayed range.")
    )
    calls: list[str] = []
    declared = json.loads(response.split("```json\n", 1)[1].split("\n```", 1)[0])["declared_percentiles"]

    async def invoke(_llm: GeneralLlm, prompt: str, **_kwargs: Any) -> str:
        if prompt.startswith("Extract the structured data from the text below."):
            calls.append("parser")
            if "Return is_discrete_integer=" in prompt:
                return json.dumps({"is_discrete_integer": '"discrete_integer"' in response})
            return json.dumps(
                {"percentiles": [{"percentile": float(level), "value": value} for level, value in declared.items()]}
            )
        assert "STRUCTURED FORECAST" in prompt, f"Unexpected LLM prompt: {prompt[:200]}"
        if isinstance(question, DateQuestion):
            assert '"question_type": "date"' in prompt
            assert question.lower_bound.strftime("%Y-%m-%d") in prompt
        calls.append("forecaster")
        return response

    monkeypatch.setattr(GeneralLlm, "invoke", invoke)
    reports = await bot.forecast_questions([question], return_exceptions=True)
    assert len(reports) == 1
    report = reports[0]
    assert isinstance(report, NumericReport), f"Forecast failed: {report!r}"
    assert "FORECASTERS_USED=2/2" in report.explanation
    assert_server_accepts_cdf(
        cdf_heights(report.prediction),
        cdf_size=question.cdf_size,
        open_lower=question.open_lower_bound,
        open_upper=question.open_upper_bound,
    )
    return report, calls


def _tail_markers(caplog: pytest.LogCaptureFixture, *, bound: str, status: str) -> list[dict[str, str]]:
    markers = [
        dict(re.findall(r"(\w+)=([^\s]+)", record.getMessage()))
        for record in caplog.records
        if record.getMessage().startswith("NUMERIC_STATED_TAIL")
    ]
    return [marker for marker in markers if marker["bound"] == bound and marker["status"] == status]


async def test_open_lower_four_percent_survives_the_full_forecast(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    question = make_real_numeric_question(48001, upper_bound=100.0, open_lower_bound=True, open_upper_bound=False)
    caplog.set_level(logging.INFO)
    report, calls = await _forecast(question, _response(question, tails={"prob_below_lower_bound": 0.04}), monkeypatch)

    heights = cdf_heights(report.prediction)
    assert heights[0] == pytest.approx(0.04, abs=1e-9)
    assert heights[-1] == 1.0
    assert calls == ["forecaster", "forecaster"]
    markers = _tail_markers(caplog, bound="lower", status="used")
    assert len(markers) == 2
    assert all(float(marker["stated"]) == 0.04 for marker in markers)
    assert all(float(marker["resulting_mass"]) == pytest.approx(0.04, abs=1e-9) for marker in markers)


async def test_closed_bounds_ignore_stated_tails_without_changing_the_forecast(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    question = make_real_numeric_question(48002, upper_bound=100.0, open_lower_bound=False, open_upper_bound=False)
    caplog.set_level(logging.INFO)
    legacy, _ = await _forecast(question, _response(question), monkeypatch)
    caplog.clear()
    report, calls = await _forecast(
        question,
        _response(question, tails={"prob_below_lower_bound": 0.04, "prob_above_upper_bound": 0.03}),
        monkeypatch,
    )

    np.testing.assert_array_equal(cdf_heights(report.prediction), cdf_heights(legacy.prediction))
    assert calls == ["forecaster", "forecaster"]
    assert not [record for record in caplog.records if record.getMessage().startswith("NUMERIC_STATED_TAIL")]


async def test_missing_stated_tails_keep_legacy_cdf_and_emit_missing_markers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    question = make_real_numeric_question(48003, upper_bound=100.0, open_lower_bound=True, open_upper_bound=True)
    caplog.set_level(logging.INFO)
    report, calls = await _forecast(question, _response(question), monkeypatch)

    assert cdf_heights(report.prediction)[0] == pytest.approx(METACULUS_OUT_OF_RANGE_TAIL_FLOOR, abs=1e-9)
    assert cdf_heights(report.prediction)[-1] == pytest.approx(1.0 - METACULUS_OUT_OF_RANGE_TAIL_FLOOR, abs=1e-9)
    assert calls == ["forecaster", "forecaster"]
    for bound in ("lower", "upper"):
        markers = _tail_markers(caplog, bound=bound, status="missing")
        assert len(markers) == 2
        assert all(int(marker["dropped_percentiles"]) == 0 for marker in markers)
        assert all(float(marker["resulting_mass"]) == pytest.approx(0.01) for marker in markers), "members keep 1%"


async def test_stated_lower_tail_overrides_conflicting_in_range_p1(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    question = make_real_numeric_question(48004, upper_bound=100.0, open_lower_bound=True, open_upper_bound=False)
    caplog.set_level(logging.INFO)
    declared = _percentiles(question)
    assert declared[str(STANDARD_PERCENTILES[0])] > question.lower_bound
    report, _ = await _forecast(
        question,
        _response(question, tails={"prob_below_lower_bound": 0.04}, percentiles=declared),
        monkeypatch,
    )

    assert cdf_heights(report.prediction)[0] == pytest.approx(0.04, abs=1e-9)
    markers = _tail_markers(caplog, bound="lower", status="used")
    assert len(markers) == 2
    assert all(
        int(marker["dropped_percentiles"]) == sum(level <= 0.04 for level in STANDARD_PERCENTILES) for marker in markers
    )


@pytest.mark.parametrize(("lower_mass", "upper_mass"), [(0.6, 0.4), (0.7, 0.4)])
async def test_invalid_tail_sum_keeps_members_and_uses_legacy_cdf(
    lower_mass: float,
    upper_mass: float,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    question = make_real_numeric_question(48005, upper_bound=100.0, open_lower_bound=True, open_upper_bound=True)
    caplog.set_level(logging.INFO)
    legacy, _ = await _forecast(question, _response(question), monkeypatch)
    caplog.clear()
    report, calls = await _forecast(
        question,
        _response(
            question,
            tails={"prob_below_lower_bound": lower_mass, "prob_above_upper_bound": upper_mass},
        ),
        monkeypatch,
    )

    np.testing.assert_array_equal(cdf_heights(report.prediction), cdf_heights(legacy.prediction))
    assert calls == ["forecaster", "forecaster"]
    assert len(_tail_markers(caplog, bound="lower", status="invalid")) == 2
    assert len(_tail_markers(caplog, bound="upper", status="invalid")) == 2


async def test_stated_tails_survive_a_metaculus_discrete_grid(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    question = make_count_question(8, qid=48006, page_url=metaculus_url(48006), open_lower=True, open_upper=True)
    question.open_time = datetime(2026, 1, 1)
    question.scheduled_resolution_time = datetime(2026, 12, 31)
    caplog.set_level(logging.INFO)
    report, calls = await _forecast(
        question,
        _response(
            question,
            tails={"prob_below_lower_bound": 0.04, "prob_above_upper_bound": 0.03},
            outcome_type="discrete_integer",
        ),
        monkeypatch,
    )

    heights = cdf_heights(report.prediction)
    assert len(heights) == question.cdf_size
    assert heights[0] == pytest.approx(0.04, abs=1e-9)
    assert heights[-1] == pytest.approx(0.97, abs=1e-9)
    assert calls == ["forecaster", "forecaster"]
    assert len(_tail_markers(caplog, bound="lower", status="used")) == 2
    assert len(_tail_markers(caplog, bound="upper", status="used")) == 2


async def test_open_lower_four_percent_survives_the_date_forecast_pipeline(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    question = make_real_date_question(
        48007,
        lower_bound=datetime(2027, 1, 1, tzinfo=UTC),
        upper_bound=datetime(2027, 2, 1, tzinfo=UTC),
        open_lower_bound=True,
        open_upper_bound=False,
        cdf_size=201,
        date_granularity="",
    )
    question.page_url = metaculus_url(48007)
    declared = {
        level: datetime.fromtimestamp(value, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        for level, value in _percentiles(as_epoch_question(question)).items()
    }
    response = (
        "The event date is most likely inside January.\n\n```json\n"
        + json.dumps(
            {
                "question_type": "date",
                "declared_percentiles": declared,
                "prob_below_lower_bound": 0.04,
            }
        )
        + "\n```"
    )
    caplog.set_level(logging.INFO)
    report, calls = await _forecast(question, response, monkeypatch)

    heights = cdf_heights(report.prediction)
    assert heights[0] == pytest.approx(0.04, abs=1e-9)
    assert heights[-1] == 1.0
    assert report.prediction.is_date is True
    assert report.prediction.get_cdf()[0].value == question.lower_bound.timestamp()
    assert calls == ["forecaster", "forecaster"]
    markers = _tail_markers(caplog, bound="lower", status="used")
    assert len(markers) == 2
    assert all(float(marker["resulting_mass"]) == pytest.approx(0.04, abs=1e-9) for marker in markers)
    extractions = [
        record.getMessage() for record in caplog.records if record.getMessage().startswith("EXTRACTION_RUNG:")
    ]
    assert len(extractions) == 2
    assert all("qtype=date rung=block" in message for message in extractions)
