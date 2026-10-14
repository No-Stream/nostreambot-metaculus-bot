import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import GeneralLlm, MetaculusQuestion, PredictedOptionList, ReasonedPrediction
from forecasting_tools.data_models.forecast_report import ResearchWithPredictions
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile as FTPercentile
from forecasting_tools.data_models.questions import ConditionalQuestion, DateQuestion
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from main import TemplateForecaster
from metaculus_bot.constants import (
    FORECASTS_SECTION_CHAR_LIMIT,
    PER_QUESTION_WALL_CLOCK_DEADLINE,
    RESEARCH_SECTION_CHAR_LIMIT,
)
from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
from metaculus_bot.research import timeseries_anchor as ts_anchor
from metaculus_bot.value_extraction import ExtractionOutcome, McForecast


def _stub_open_time() -> datetime:
    return datetime.now() - timedelta(days=30)


def _stub_resolve_time() -> datetime:
    return datetime.now() + timedelta(days=365)


# `asyncio` is used by the soft-deadline test below; an explicit no-op reference
# prevents the formatter from pruning the import when the only usage sits far
# below the import block (formatter heuristic).
_ASYNCIO_SLEEP = asyncio.sleep


@pytest.fixture
def mock_metaculus_question():
    question = MagicMock(spec=MetaculusQuestion)
    question.page_url = "http://example.com/question"
    question.api_json = {"question": {}}
    question.question_text = "Test Question"
    question.background_info = "Background info"
    question.resolution_criteria = "Resolution criteria"
    question.fine_print = "Fine print"
    question.unit_of_measure = "units"
    question.id_of_question = 123  # Add a mock ID for testing
    question.open_time = _stub_open_time()
    question.scheduled_resolution_time = _stub_resolve_time()
    return question


@pytest.mark.asyncio
async def test_template_forecaster_init_with_forecasters(mock_general_llm):
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config)

    assert bot._forecaster_llms == llms_config["forecasters"]
    assert bot.predictions_per_research_report == 2
    assert bot.get_llm("default") == mock_general_llm  # Should be the first forecaster


@pytest.mark.asyncio
async def test_template_forecaster_init_without_forecasters():
    llms_config = {
        "default": GeneralLlm(model="test_default"),
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config, predictions_per_research_report=3)

    assert not bot._forecaster_llms
    assert bot.predictions_per_research_report == 3
    assert cast(GeneralLlm, bot.get_llm("default")).model == "test_default"


@pytest.mark.asyncio
async def test_template_forecaster_init_no_llms_provided():
    with pytest.raises(ValueError, match=r"Either 'forecasters' or a 'default' LLM must be provided."):
        TemplateForecaster(llms=None)


@pytest.mark.asyncio
async def test_template_forecaster_init_missing_required_llms():
    # Test missing parser and researcher
    incomplete_llms: dict[str, str | GeneralLlm] = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
    }
    with pytest.raises(ValueError, match="Missing required LLM purposes: parser, researcher"):
        TemplateForecaster(llms=incomplete_llms)

    # Test missing just researcher
    incomplete_llms = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
    }
    with pytest.raises(ValueError, match="Missing required LLM purposes: researcher"):
        TemplateForecaster(llms=incomplete_llms)


@pytest.mark.asyncio
async def test_template_forecaster_init_forecasters_not_list():
    llms_config = {
        "forecasters": "not_a_list",
        "default": GeneralLlm(model="test_default"),
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    with patch("metaculus_bot.llm_setup.logger.warning") as mock_warning:
        bot = TemplateForecaster(llms=llms_config)
        mock_warning.assert_called_once_with("'forecasters' key in llms must be a list of GeneralLlm objects.")
        assert not bot._forecaster_llms
        assert bot.predictions_per_research_report == 1  # Default value from parent class


@pytest.mark.asyncio
async def test_research_and_make_predictions_with_forecasters(mock_binary_question, mock_general_llm):
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1)

    # Mock internal methods
    bot._get_notepad = AsyncMock(
        return_value=MagicMock(total_research_reports_attempted=0, total_predictions_attempted=0)
    )
    bot.run_research = AsyncMock(return_value="mock research")
    bot._make_prediction = AsyncMock(return_value=ReasonedPrediction(prediction_value=0.5, reasoning="test"))
    bot._gather_results_and_exceptions = AsyncMock(
        return_value=(
            [
                ReasonedPrediction(prediction_value=0.5, reasoning="test"),
                ReasonedPrediction(prediction_value=0.6, reasoning="test2"),
            ],
            [],
            None,
        )
    )

    # Wrap _forecaster_with_soft_deadline so we can count invocations. The
    # soft-deadline wrapper is the new per-forecaster entrypoint — it delegates
    # to _make_prediction internally. Tests mocking _gather_results_and_exceptions
    # short-circuit execution, so we assert on the wrapper being called (once
    # per forecaster) rather than the inner _make_prediction.
    bot._forecaster_with_soft_deadline = AsyncMock(
        return_value=ReasonedPrediction(prediction_value=0.5, reasoning="test")
    )

    result = await bot._research_and_make_predictions(mock_binary_question)

    bot._get_notepad.assert_called_once_with(mock_binary_question)
    # The question travels positionally; the per-question time budget rides alongside
    # it as a keyword so the research phase can bound itself (time_budget.py).
    (research_args, research_kwargs) = bot.run_research.call_args
    assert research_args == (mock_binary_question,)
    assert research_kwargs["time_budget"].total_s == PER_QUESTION_WALL_CLOCK_DEADLINE
    # Forecasters receive run_research output verbatim — there is no whole-corpus
    # summarization pass. AskNews-only summarization lives in the orchestrator
    # (see test_research_orchestrator.py).
    assert bot._forecaster_with_soft_deadline.call_count == 2  # Called once for each forecaster
    # Trailing chart_b64 is None here (TS_ANCHOR_CHART_ENABLED off in the test env).
    bot._forecaster_with_soft_deadline.assert_any_call(
        mock_binary_question,
        "mock research",
        mock_general_llm,
        qid=mock_binary_question.id_of_question,
        chart_b64=None,
    )
    assert isinstance(result, ResearchWithPredictions)
    assert len(result.predictions) == 2


@pytest.mark.asyncio
async def test_diagnostics_seam_forecasters_clean_comment_carries_block(mock_general_llm, mock_binary_question):
    """Diagnostics seam: forecaster prompts receive the clean research text while the
    comment-bound research_report gets the provider-diagnostics block re-appended."""
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1)

    diagnostics_block = "---\n\n## Provider Diagnostics\n\n- asknews: ok | 100 chars | 50 ms"
    bot._get_notepad = AsyncMock(
        return_value=MagicMock(total_research_reports_attempted=0, total_predictions_attempted=0)
    )
    bot.run_research = AsyncMock(return_value="clean research")
    bot._research.pop_provider_diagnostics = MagicMock(return_value=diagnostics_block)
    bot._forecaster_with_soft_deadline = AsyncMock(
        return_value=ReasonedPrediction(prediction_value=0.5, reasoning="test")
    )

    result = await bot._research_and_make_predictions(mock_binary_question)

    # Forecasters got the clean text (no diagnostics).
    for call in bot._forecaster_with_soft_deadline.call_args_list:
        assert call[0][1] == "clean research"
    # The comment-bound research_report carries the block, appended after the research body.
    assert "## Provider Diagnostics" in result.research_report
    assert result.research_report.index("clean research") < result.research_report.index("## Provider Diagnostics")
    bot._research.pop_provider_diagnostics.assert_called_once_with(mock_binary_question.id_of_question)


@pytest.mark.asyncio
async def test_research_and_make_predictions_without_forecasters(mock_binary_question):
    llms_config = {
        "default": GeneralLlm(model="test_default"),
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config, predictions_per_research_report=1)

    # Mock the super method call
    with patch(
        "forecasting_tools.forecast_bots.forecast_bot.ForecastBot._research_and_make_predictions",
        new_callable=AsyncMock,
    ) as mock_super_method:
        mock_super_method.return_value = ResearchWithPredictions(
            research_report="super research",
            summary_report="super summary",
            predictions=[ReasonedPrediction(prediction_value=0.6, reasoning="super test")],
        )
        result = await bot._research_and_make_predictions(mock_binary_question)
        mock_super_method.assert_called_once_with(mock_binary_question)
        assert isinstance(result, ResearchWithPredictions)
        assert result.research_report == "super research"


@pytest.mark.asyncio
async def test_make_prediction_with_provided_llm(mock_binary_question, mock_general_llm):
    llms_config: dict[str, str | GeneralLlm] = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config)
    bot._get_notepad = AsyncMock(return_value=MagicMock(total_predictions_attempted=0))
    bot._run_forecast_on_binary = AsyncMock(
        return_value=ReasonedPrediction(prediction_value=0.7, reasoning="binary forecast")
    )

    result = await bot._make_prediction(mock_binary_question, "some research", mock_general_llm)

    bot._get_notepad.assert_called_once_with(mock_binary_question)
    # Trailing chart_b64 is None (TS_ANCHOR_CHART_ENABLED off in the test env).
    bot._run_forecast_on_binary.assert_called_once_with(mock_binary_question, "some research", mock_general_llm, None)
    assert result.prediction_value == 0.7
    assert "Model: mock_model" in result.reasoning
    assert "binary forecast" in result.reasoning


@pytest.mark.asyncio
async def test_make_prediction_without_provided_llm(mock_binary_question):
    mock_default_llm = MagicMock(spec=GeneralLlm)
    mock_default_llm.model = "default_mock_model"
    mock_default_llm.invoke = AsyncMock(return_value="default reasoning")

    llms_config = {
        "default": mock_default_llm,
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config)
    bot._get_notepad = AsyncMock(return_value=MagicMock(total_predictions_attempted=0))
    bot._run_forecast_on_binary = AsyncMock(
        return_value=ReasonedPrediction(prediction_value=0.8, reasoning="default binary forecast")
    )
    bot.get_llm = MagicMock(return_value=mock_default_llm)

    result = await bot._make_prediction(mock_binary_question, "some research")

    bot._get_notepad.assert_called_once_with(mock_binary_question)
    bot.get_llm.assert_called_once_with("default", "llm")
    # Trailing chart_b64 is None (TS_ANCHOR_CHART_ENABLED off in the test env).
    bot._run_forecast_on_binary.assert_called_once_with(mock_binary_question, "some research", mock_default_llm, None)
    assert result.prediction_value == 0.8
    assert "Model: default_mock_model" in result.reasoning
    assert "default binary forecast" in result.reasoning


@pytest.mark.asyncio
async def test_run_forecast_on_binary_uses_provided_llm(mock_binary_question, mock_general_llm):
    llms_config: dict[str, str | GeneralLlm] = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config)

    # Patch the ladder seam so the forecaster LLM is exercised without hitting
    # the real parser: extract_binary returns a canned block-rung outcome.
    async def _fake_extract(*_args, **_kwargs) -> ExtractionOutcome[float]:
        return ExtractionOutcome(value=0.65, rung="block", block_present=True)

    with patch("metaculus_bot.forecaster_runners.extract_binary", side_effect=_fake_extract) as mock_extract:
        result = await bot._run_forecast_on_binary(mock_binary_question, "some research", mock_general_llm)
        mock_general_llm.invoke.assert_called_once()
        mock_extract.assert_called_once()
        assert result.prediction_value == 0.65
        assert "mock reasoning" in result.reasoning


@pytest.mark.asyncio
async def test_run_forecast_on_multiple_choice_uses_provided_llm(mock_metaculus_question, mock_general_llm):
    llms_config: dict[str, str | GeneralLlm] = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config)
    mock_metaculus_question.options = ["A", "B"]

    # Patch the ladder seam for MC. Construct a well-formed PredictedOptionList
    # so the downstream clamp/renormalize step doesn't reject it.

    fake_pol = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=0.6),
            PredictedOption(option_name="B", probability=0.4),
        ]
    )

    async def _fake_extract(*_args, **_kwargs) -> ExtractionOutcome[McForecast]:
        return ExtractionOutcome(
            value=McForecast(fake_pol, [o.probability for o in fake_pol.predicted_options]),
            rung="block",
            block_present=True,
        )

    with patch("metaculus_bot.forecaster_runners.extract_mc", side_effect=_fake_extract) as mock_extract:
        result = await bot._run_forecast_on_multiple_choice(mock_metaculus_question, "some research", mock_general_llm)
        mock_general_llm.invoke.assert_called_once()
        mock_extract.assert_called_once()
        assert result.prediction_value is not None
        assert "mock reasoning" in result.reasoning


@pytest.mark.asyncio
async def test_run_forecast_on_numeric_uses_provided_llm(mock_metaculus_question, mock_general_llm):
    llms_config: dict[str, str | GeneralLlm] = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config)

    # Mock bound_messages and structured_output to return a valid percentile list

    fake_percentiles = [
        FTPercentile(value=v, percentile=p)
        for v, p in zip(
            [0.25, 0.5, 1, 2, 4, 5, 6, 7, 8, 9, 9.5, 9.75, 9.9],
            [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99],
            strict=True,
        )
    ]
    # Provide minimal numeric bounds attributes expected by NumericDistribution.from_question
    mock_metaculus_question.open_upper_bound = False
    mock_metaculus_question.open_lower_bound = False
    mock_metaculus_question.upper_bound = 100
    mock_metaculus_question.lower_bound = 0
    mock_metaculus_question.zero_point = None
    mock_metaculus_question.cdf_size = 201

    # Percentile extraction now routes through the ladder seam (extract_numeric);
    # only the OutcomeTypeResult classification still goes through parse_structured
    # (and only when the block-first read in run_numeric_forecast doesn't find an
    # outcome_type in the reasoning, which is the case for "mock reasoning").
    async def _fake_extract_numeric(*_args, **_kwargs) -> ExtractionOutcome[list[FTPercentile]]:
        return ExtractionOutcome(value=fake_percentiles, rung="block", block_present=True)

    with (
        patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("", "")) as mock_bounds,
        patch(
            "metaculus_bot.forecaster_runners.parse_structured",
            side_effect=[OutcomeTypeResult(is_discrete_integer=False)],
        ) as mock_struct,
        patch(
            "metaculus_bot.forecaster_runners.extract_numeric",
            side_effect=_fake_extract_numeric,
        ) as mock_extract,
    ):
        result = await bot._run_forecast_on_numeric(mock_metaculus_question, "some research", mock_general_llm)
        mock_general_llm.invoke.assert_called_once()
        mock_bounds.assert_called_once()
        assert mock_struct.call_count == 1  # outcome type classification only; percentiles route through the ladder
        mock_extract.assert_called_once()
        assert result.prediction_value is not None
        assert "mock reasoning" in result.reasoning


def test_format_methods_preserve_long_outputs_for_complete_comment_allocation():
    llms_config = {
        "default": GeneralLlm(model="test_default"),
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config)

    long_research_body = "Line\n" + "A" * (RESEARCH_SECTION_CHAR_LIMIT + 500)
    long_reasoning = "Reasoning\n" + "B" * (FORECASTS_SECTION_CHAR_LIMIT + 800)
    research_with_predictions = ResearchWithPredictions(
        research_report=f"# Deep Dive\n{long_research_body}",
        summary_report="Summary",
        predictions=[ReasonedPrediction(prediction_value=0.5, reasoning=long_reasoning)],
    )

    formatted_research = bot._format_main_research(1, research_with_predictions)
    assert formatted_research.startswith("## Report 1 Research")
    assert len(formatted_research) > RESEARCH_SECTION_CHAR_LIMIT
    assert "A" * 500 in formatted_research

    formatted_rationales = bot._format_forecaster_rationales(1, research_with_predictions)
    assert formatted_rationales.startswith("## R1: Forecaster 1 Reasoning")
    assert len(formatted_rationales) > FORECASTS_SECTION_CHAR_LIMIT
    assert "B" * 800 in formatted_rationales


# ---------------------------------------------------------------------------
# F4: real _forecaster_with_soft_deadline timeout branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forecaster_with_soft_deadline_times_out_and_bumps_counter(
    mock_binary_question, mock_general_llm, monkeypatch: pytest.MonkeyPatch
):
    """The real wrapper must raise TimeoutError and bump _forecasters_dropped_count
    when _make_prediction exceeds FORECASTER_SOFT_DEADLINE. Prior tests replaced
    the wrapper with an AsyncMock, so the asyncio.wait_for branch was untested.
    """
    llms_config = {
        "forecasters": [mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1)

    # Tighten the deadline to a fraction of a second so the test is fast.
    monkeypatch.setattr("metaculus_bot.forecaster.FORECASTER_SOFT_DEADLINE", 0.05)

    async def slow_make_prediction(question, research, llm, chart_b64=None):
        await asyncio.sleep(5)
        return ReasonedPrediction(prediction_value=0.5, reasoning="never returned")

    bot._make_prediction = AsyncMock(side_effect=slow_make_prediction)

    assert bot._forecasters_dropped_count == 0
    with pytest.raises(asyncio.TimeoutError):
        await bot._forecaster_with_soft_deadline(
            mock_binary_question, "research", mock_general_llm, qid=mock_binary_question.id_of_question
        )
    assert bot._forecasters_dropped_count == 1


# ---------------------------------------------------------------------------
# F5: min-forecasters guard raise path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_min_forecasters_guard_raises_runtime_error_when_exception_group_none(
    mock_binary_question, mock_general_llm
):
    """If only 1/2 forecasters succeed and threshold is 3, with no exception_group,
    the guard raises RuntimeError and bumps _questions_failed_to_publish.
    """
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    # Warning expected on construction (threshold exceeds ensemble size); this is
    # the exact scenario we want to test.
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=3)

    bot._get_notepad = AsyncMock(
        return_value=MagicMock(total_research_reports_attempted=0, total_predictions_attempted=0)
    )
    bot.run_research = AsyncMock(return_value="mock research")
    # Both forecaster tasks succeed; threshold is 3, so we get "Only 2/2" and the
    # min-forecasters guard fires.
    bot._forecaster_with_soft_deadline = AsyncMock(
        return_value=ReasonedPrediction(prediction_value=0.5, reasoning="ok")
    )

    assert bot._questions_failed_to_publish == 0
    with pytest.raises(RuntimeError, match="Only 2/2 forecasters succeeded"):
        await bot._research_and_make_predictions(mock_binary_question)
    assert bot._questions_failed_to_publish == 1


@pytest.mark.asyncio
async def test_min_forecasters_guard_reraises_exception_group_when_present(mock_binary_question, mock_general_llm):
    """If only 1/2 forecasters succeed and exception_group is non-None, the
    re-raise path preserves the exception chain by delegating to the framework
    helper (which raises an ExceptionGroup).
    """
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=3)

    bot._get_notepad = AsyncMock(
        return_value=MagicMock(total_research_reports_attempted=0, total_predictions_attempted=0)
    )
    bot.run_research = AsyncMock(return_value="mock research")
    # First forecaster succeeds; second raises. Below the 3/2 threshold, the
    # exception group is wrapped and re-raised.
    call_count = {"n": 0}

    async def mixed_results(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ReasonedPrediction(prediction_value=0.5, reasoning="ok")
        raise RuntimeError("forecaster 2 failed")

    bot._forecaster_with_soft_deadline = cast(Any, mixed_results)

    with pytest.raises(ExceptionGroup) as exc_info:  # 3.11+ builtin
        await bot._research_and_make_predictions(mock_binary_question)

    # The framework helper wraps the exception group with a prepended message
    # but preserves the original wrapped exceptions.
    assert any(isinstance(e, RuntimeError) and "forecaster 2 failed" in str(e) for e in exc_info.value.exceptions)
    assert bot._questions_failed_to_publish == 1


# ---------------------------------------------------------------------------
# F5b: exception-dropped forecaster counts as degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exception_dropped_forecaster_counts_as_degradation(mock_binary_question, mock_general_llm):
    """A forecaster that finishes by raising (not a timeout, not a cancel) must
    bump _forecasters_dropped_count so cli.py's alertable exit fires.

    Regression for the 2026-07-19 silent-degradation bug: a numeric forecaster
    got message.content=None from OpenRouter, the AssertionError propagated, the
    forecaster was dropped, and the question published on the surviving models —
    but the drop was invisible to CI because only soft-deadline timeouts and
    wall-clock cancels bumped the counter. min=1 keeps the survivor publishing
    so the exception-drop counter is isolated from _questions_failed_to_publish.
    """
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1)

    bot._get_notepad = AsyncMock(
        return_value=MagicMock(total_research_reports_attempted=0, total_predictions_attempted=0)
    )
    bot.run_research = AsyncMock(return_value="mock research")

    # One forecaster succeeds; the other raises AssertionError mid-prediction
    # (mirrors the real message.content=None -> AssertionError failure).
    call_count = {"n": 0}

    async def one_raises(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ReasonedPrediction(prediction_value=0.5, reasoning="ok")
        raise AssertionError("message.content was None")

    bot._forecaster_with_soft_deadline = cast(Any, one_raises)

    assert bot._forecasters_dropped_count == 0
    assert bot.alertable_count == 0

    result = await bot._research_and_make_predictions(mock_binary_question)

    # Survivor still publishes (1/2 >= min 1), so no question-failed bump.
    assert len(result.predictions) == 1
    assert bot._questions_failed_to_publish == 0
    # The exception-dropped forecaster is now counted as degradation.
    assert bot._forecasters_dropped_count == 1
    assert bot.alertable_count == 1
    # The error still flows to the errors list ("Minor Errors" reporting unchanged).
    assert any("AssertionError" in e and "message.content was None" in e for e in result.errors)


@pytest.mark.asyncio
async def test_degraded_publish_states_its_survivor_count_in_the_log(mock_binary_question, mock_general_llm, caplog):
    """A degraded publish must SAY it was degraded, in the run log.

    This is the case the marker exists for. At a ``min_forecasters_to_publish`` of 1
    a 1-of-2 publish is a success: it exits zero, and the only line that had ever
    named a survivor count fires solely BELOW the minimum, so it stays silent. The
    count reached just the published Metaculus comment (``FORECASTERS_USED``), which
    is never logged. So the operator reading a run log could not tell a thinned
    ensemble from a healthy one without counting ``EXTRACTION_RUNG`` lines and
    deduping model slugs.

    Asserts the SURVIVED count and the configured total, so a degraded run is
    self-evidently degraded from one grep.
    """
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1)
    bot._get_notepad = AsyncMock(
        return_value=MagicMock(total_research_reports_attempted=0, total_predictions_attempted=0)
    )
    bot.run_research = AsyncMock(return_value="mock research")

    call_count = {"n": 0}

    async def one_raises(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # The "Model:" prefix is what _make_prediction stamps in production, and
            # what the marker reads to name survivors.
            return ReasonedPrediction(prediction_value=0.5, reasoning="Model: openrouter/openai/gpt-5.6-sol\n\nok")
        raise AssertionError("message.content was None")

    bot._forecaster_with_soft_deadline = cast(Any, one_raises)

    with caplog.at_level("INFO"):
        result = await bot._research_and_make_predictions(mock_binary_question)

    assert len(result.predictions) == 1
    survived = [m for m in (r.getMessage() for r in caplog.records) if "FORECASTERS_SURVIVED:" in m]
    assert survived, "a degraded publish emitted no survivor-count line"
    assert any("survived=1/2" in m for m in survived), f"expected survived=1/2 on a 1-of-2 publish; got: {survived}"
    # Names the survivor, not the roster: reporting the configured models here would
    # relabel this degraded run as full.
    assert any("models=gpt-5.6-sol" in m for m in survived), f"expected only the survivor named; got: {survived}"


@pytest.mark.asyncio
async def test_soft_deadline_timeout_counted_once_through_gather(
    mock_binary_question, mock_general_llm, monkeypatch: pytest.MonkeyPatch
):
    """A soft-deadline timeout must count exactly once end-to-end.

    The timeout is bumped at its raise site in _forecaster_with_soft_deadline,
    then the failed task also lands in the gather's done-loop. Pins the
    no-double-count invariant: the done-loop excludes asyncio.TimeoutError so the
    same drop isn't counted twice.
    """
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1)

    bot._get_notepad = AsyncMock(
        return_value=MagicMock(total_research_reports_attempted=0, total_predictions_attempted=0)
    )
    bot.run_research = AsyncMock(return_value="mock research")

    # Drive the REAL soft-deadline wrapper (don't patch it) so its timeout path
    # runs; patch the inner _make_prediction: one fast, one past the deadline.
    monkeypatch.setattr("metaculus_bot.forecaster.FORECASTER_SOFT_DEADLINE", 0.05)
    call_count = {"n": 0}

    async def make_pred(question, research, llm, chart_b64=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ReasonedPrediction(prediction_value=0.5, reasoning="ok")
        await asyncio.sleep(5)
        return ReasonedPrediction(prediction_value=0.5, reasoning="never")

    bot._make_prediction = AsyncMock(side_effect=make_pred)

    result = await bot._research_and_make_predictions(mock_binary_question)

    assert len(result.predictions) == 1
    # Counted once (at the raise site), not twice (raise site + done-loop).
    assert bot._forecasters_dropped_count == 1
    assert bot.alertable_count == 1


def _bot_with_one_forecaster(mock_general_llm) -> TemplateForecaster:
    llms_config: dict[str, Any] = {
        "forecasters": [mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    return TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1)


class TestMetaculusClientSeam:
    """``metaculus_client`` passes straight through to the framework, which uses it for the
    tournament fetch and every publish POST; the Mantic run mode injects a ``ManticClient`` here."""

    def test_injected_client_reaches_the_framework(self, mock_general_llm):
        client = MetaculusClient(base_url="https://platform.invalid/api", token="t" * 40)
        llms_config: dict[str, Any] = {
            "forecasters": [mock_general_llm],
            "summarizer": "mock_summarizer_model",
            "parser": "mock_parser_model",
            "researcher": "mock_researcher_model",
            "default": "mock_default_model",
        }
        bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1, metaculus_client=client)
        assert bot.metaculus_client is client

    def test_default_is_the_framework_metaculus_client(self, mock_general_llm):
        bot = _bot_with_one_forecaster(mock_general_llm)
        assert type(bot.metaculus_client) is MetaculusClient


class TestUnsupportedQuestionTypes:
    """The type guard at the top of ``forecast_questions`` DROPS conditional questions with one
    WARNING rather than raising; every entry path (tournament fetch, URL list) funnels through
    it. Date questions pass it (Mantic's pool is 41% date questions)."""

    async def test_conditional_questions_are_skipped_with_a_warning(self, mock_general_llm, caplog):
        bot = _bot_with_one_forecaster(mock_general_llm)
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.forecaster"):
            # A real (unvalidated) instance, not a spec'd mock: the guard names the dropped TYPE.
            reports = await bot.forecast_questions([ConditionalQuestion.model_construct()])
        assert reports == []
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "Skipping 1 unsupported question(s)" in message and "ConditionalQuestion" in message for message in warnings
        ), warnings

    async def test_a_date_question_passes_the_guard(self, mock_general_llm, caplog, monkeypatch):
        forwarded: list[list[MetaculusQuestion]] = []

        async def capture(self, questions, return_exceptions=False):
            forwarded.append(list(questions))
            return []

        monkeypatch.setattr(ForecastBot, "forecast_questions", capture)
        date_question = MagicMock(spec=DateQuestion)
        date_question.already_forecasted = False
        date_question.close_time = datetime.now(UTC) + timedelta(days=1)
        bot = _bot_with_one_forecaster(mock_general_llm)
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.forecaster"):
            await bot.forecast_questions([cast(MetaculusQuestion, date_question)])
        assert forwarded == [[date_question]]
        assert not [r for r in caplog.records if "unsupported" in r.getMessage()]


class TestResearchChartSideChannel:
    """_pull_research_chart pops the TS-anchor chart from the provider's per-session
    cache and returns it — but only when the chart flag is on. Off (the prod default),
    it never touches the provider module and leaves the cache intact."""

    def test_pull_returns_chart_and_drains_cache_when_flag_on(self, mock_general_llm, monkeypatch):
        monkeypatch.setenv("TS_ANCHOR_CHART_ENABLED", "true")
        ts_anchor._reset_session_caches()
        ts_anchor._session_charts[777] = "Zm9v"  # a stashed chart for qid 777
        try:
            bot = _bot_with_one_forecaster(mock_general_llm)
            chart = bot._pull_research_chart(777)

            assert chart == "Zm9v"
            assert 777 not in ts_anchor._session_charts  # popped exactly once, cache drained
        finally:
            ts_anchor._reset_session_caches()

    def test_pull_returns_none_and_leaves_cache_when_flag_off(self, mock_general_llm, monkeypatch):
        monkeypatch.delenv("TS_ANCHOR_CHART_ENABLED", raising=False)
        ts_anchor._reset_session_caches()
        ts_anchor._session_charts[778] = "Zm9v"  # present, but the flag gate must skip it
        try:
            bot = _bot_with_one_forecaster(mock_general_llm)
            chart = bot._pull_research_chart(778)

            assert chart is None
            assert ts_anchor._session_charts[778] == "Zm9v"  # flag-off leaves the provider cache untouched
        finally:
            ts_anchor._reset_session_caches()

    def test_pull_returns_none_when_no_chart_stashed(self, mock_general_llm, monkeypatch):
        monkeypatch.setenv("TS_ANCHOR_CHART_ENABLED", "true")
        ts_anchor._reset_session_caches()
        bot = _bot_with_one_forecaster(mock_general_llm)
        assert bot._pull_research_chart(779) is None
