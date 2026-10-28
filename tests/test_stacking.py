"""Unit tests for stacking functionality."""

import asyncio
from datetime import datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.prompts import (
    stacking_binary_prompt,
    stacking_multiple_choice_prompt,
    stacking_numeric_prompt,
)
from tests.conftest import gather_predictions_stub

# Type alias matching ForecastBot.__init__'s `llms` parameter. The bot also accepts a
# "forecasters" list value at runtime (see TemplateForecaster setup), which the upstream
# annotation does not model, so test literals are cast to this alias at each call site.
LlmsConfig = dict[str, str | GeneralLlm] | None

# The pipeline `run_stacking` / `_aggregate_predictions` reasoned-prediction params are typed with
# the framework's PredictionTypes union; test fixtures build concrete float predictions.
ReasonedPredictionList = list[ReasonedPrediction[PredictionTypes]]


def _stub_open_time() -> datetime:
    return datetime.now() - timedelta(days=30)


def _stub_resolve_time() -> datetime:
    return datetime.now() + timedelta(days=365)


class TestStackingConfiguration:
    """Tests for stacking LLM configuration in TemplateForecaster."""

    def test_stacking_bot_creation_success(self):
        """Test successful creation of stacking bot."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        bot = TemplateForecaster(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            publish_reports_to_metaculus=False,
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm, test_llm],
                    "stacker": test_llm,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
            is_benchmarking=True,
        )

        assert bot.aggregation_strategy == AggregationStrategy.STACKING
        assert len(bot._forecaster_llms) == 2
        assert bot._pipeline.stacker_llm is not None
        assert bot._pipeline.stacker_llm.model == "test-model"

        bot.aggregation_strategy = AggregationStrategy.MEAN
        bot.stacking_fallback_on_failure = False
        bot.stacking_randomize_order = False

        assert bot._pipeline.strategy == AggregationStrategy.MEAN
        assert bot._pipeline.stacking_fallback_on_failure is False
        assert bot._pipeline.stacking_randomize_order is False

    def test_stacking_without_stacker_llm_fails(self):
        """Test that stacking without stacker LLM fails at runtime."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        # Bot creation should succeed (no stacker LLM configured)
        bot = TemplateForecaster(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            publish_reports_to_metaculus=False,
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm],
                    # Note: no "stacker" key
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
        )

        # Bot creation succeeds but no stacker LLM is configured.
        assert bot._pipeline.stacker_llm is None

        # The error should occur when trying to aggregate predictions

        with pytest.raises(ValueError, match="STACKING aggregation strategy requires a stacker LLM"):
            asyncio.run(
                bot._aggregate_predictions(
                    predictions=[0.5],
                    question=Mock(),
                    research="test research",
                    reasoned_predictions=[Mock()],
                )
            )

    def test_invalid_stacker_llm_type(self):
        """Test that invalid stacker LLM type is handled gracefully."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        bot = TemplateForecaster(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            publish_reports_to_metaculus=False,
            aggregation_strategy=AggregationStrategy.MEAN,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm],
                    "stacker": "not-an-llm",  # Invalid type
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
        )

        # Should not crash but stacker should be None due to warning
        assert bot._pipeline.stacker_llm is None

    def test_stacking_parameters(self):
        """Test stacking-specific parameters."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        # Test default values
        bot1 = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm],
                    "stacker": test_llm,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
        )

        assert bot1.stacking_fallback_on_failure is True
        assert bot1.stacking_randomize_order is True

        # Test custom values
        bot2 = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm],
                    "stacker": test_llm,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
            stacking_fallback_on_failure=False,
            stacking_randomize_order=False,
        )

        assert bot2.stacking_fallback_on_failure is False
        assert bot2.stacking_randomize_order is False


class TestStackingPrompts:
    """Tests for stacking prompt generation."""

    def test_stacking_binary_prompt(self):
        """Test binary stacking prompt generation."""
        question = BinaryQuestion(
            question_text="Will it rain tomorrow?",
            background_info="Weather question",
            resolution_criteria="Resolves YES if it rains",
            fine_print="",
            page_url="https://test.com/1",
            id_of_question=1,
            open_time=_stub_open_time(),
            scheduled_resolution_time=_stub_resolve_time(),
        )

        research = "Weather forecast shows 50% chance of rain."
        base_predictions = [
            "Analysis 1: Based on weather patterns, I estimate 60% probability.",
            "Analysis 2: Meteorological data suggests 40% chance.",
        ]

        prompt = stacking_binary_prompt(question, research, base_predictions)

        assert "Will it rain tomorrow?" in prompt
        assert "Weather forecast shows 50% chance" in prompt
        assert "Model 1 Analysis:" in prompt
        assert "Model 2 Analysis:" in prompt
        assert "Analysis 1: Based on weather patterns" in prompt
        assert "Analysis 2: Meteorological data" in prompt
        # Post-refactor: the trailing "Probability: ZZ%" prose line is gone;
        # `posterior_prob` in the STRUCTURED FORECAST JSON block is the only
        # forecast surface. Assert on the block instead.
        assert "Probability: ZZ%" not in prompt
        assert '"question_type": "binary"' in prompt
        assert "posterior_prob" in prompt

    def test_stacking_multiple_choice_prompt(self):
        """Test multiple choice stacking prompt generation."""
        question = MultipleChoiceQuestion(
            question_text="What color will the ball be?",
            options=["Red", "Blue", "Green"],
            background_info="Ball color question",
            resolution_criteria="Based on final ball color",
            fine_print="",
            page_url="https://test.com/2",
            id_of_question=2,
            open_time=_stub_open_time(),
            scheduled_resolution_time=_stub_resolve_time(),
        )

        research = "Previous balls were mostly red."
        base_predictions = [
            "Red seems most likely at 50%, Blue 30%, Green 20%",
            "I think Blue is undervalued: Red 40%, Blue 40%, Green 20%",
        ]

        prompt = stacking_multiple_choice_prompt(question, research, base_predictions)

        assert "What color will the ball be?" in prompt
        assert "Red', 'Blue', 'Green" in prompt
        assert "Previous balls were mostly red" in prompt
        assert "Model 1 Analysis:" in prompt
        assert "Red seems most likely" in prompt
        # Post-refactor: the STRUCTURED FORECAST JSON block uses real option
        # names as `option_probs` keys — the trailing "{opt}: NN%" prose lines
        # are gone. Strict parsers still get exact-string matches from the
        # JSON keys.
        assert '"question_type": "multiple_choice"' in prompt
        assert '"Red"' in prompt
        assert '"Blue"' in prompt
        assert '"Green"' in prompt
        assert "Red: NN%" not in prompt
        assert "Blue: NN%" not in prompt
        assert "Green: NN%" not in prompt

    def test_stacking_numeric_prompt(self):
        """Test numeric stacking prompt generation."""
        question = NumericQuestion(
            question_text="How many people will attend?",
            lower_bound=0.0,
            upper_bound=1000.0,
            open_lower_bound=False,
            open_upper_bound=False,
            background_info="Attendance question",
            resolution_criteria="Based on final count",
            fine_print="",
            page_url="https://test.com/3",
            id_of_question=3,
            open_time=_stub_open_time(),
            scheduled_resolution_time=_stub_resolve_time(),
        )

        research = "Historical attendance averages 500 people."
        base_predictions = [
            "Based on trends, I expect around 400-600 people.",
            "Considering weather, probably 300-500 range.",
        ]
        lower_bound_msg = "Lower bound: 0"
        upper_bound_msg = "Upper bound: 1000"

        prompt = stacking_numeric_prompt(
            question,
            research,
            base_predictions,
            lower_bound_message=lower_bound_msg,
            upper_bound_message=upper_bound_msg,
        )

        assert "How many people will attend?" in prompt
        assert "Historical attendance averages 500" in prompt
        assert "Model 1 Analysis:" in prompt
        assert "Based on trends, I expect around 400-600" in prompt
        assert "Lower bound: 0" in prompt
        assert "Upper bound: 1000" in prompt
        # Post-refactor: the trailing "Percentile 5: [value]" prose lines are
        # gone; `declared_percentiles` in the STRUCTURED FORECAST JSON block
        # (keyed by "0.01" .. "0.99") is the only forecast surface.
        assert '"question_type": "numeric"' in prompt
        assert '"declared_percentiles"' in prompt
        assert '"0.05"' in prompt
        assert '"0.5"' in prompt
        assert '"0.95"' in prompt
        assert "Percentile 5:" not in prompt
        assert "Percentile 99:" not in prompt


class TestModelNameStripping:
    """Tests for model name stripping and order randomization."""

    def test_model_name_stripping(self):
        """Test that model names are properly stripped from reasoning."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm],
                    "stacker": test_llm,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
            stacking_randomize_order=False,  # Disable for predictable testing
        )

        # Create test predictions with model names
        reasoned_predictions = [
            ReasonedPrediction(
                prediction_value=0.6,
                reasoning="Model: gpt-4\n\nThis is my analysis of the situation...",
            ),
            ReasonedPrediction(
                prediction_value=0.4,
                reasoning="Model: claude-3\n\nI disagree with the above analysis...",
            ),
        ]

        # Test the model name stripping logic
        base_predictions = []
        for pred in reasoned_predictions:
            reasoning = pred.reasoning
            if reasoning.startswith("Model: "):
                lines = reasoning.split("\n", 2)
                if len(lines) >= 3 and lines[1] == "":
                    reasoning = lines[2]
            base_predictions.append(reasoning)

        assert base_predictions[0] == "This is my analysis of the situation..."
        assert base_predictions[1] == "I disagree with the above analysis..."
        assert "Model: gpt-4" not in base_predictions[0]
        assert "Model: claude-3" not in base_predictions[1]

    def test_order_randomization_setting(self):
        """Test that order randomization setting is respected."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        # Test randomization enabled
        bot1 = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm],
                    "stacker": test_llm,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
            stacking_randomize_order=True,
        )
        assert bot1.stacking_randomize_order is True

        # Test randomization disabled
        bot2 = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm],
                    "stacker": test_llm,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
            stacking_randomize_order=False,
        )
        assert bot2.stacking_randomize_order is False


class TestStackingIntegration:
    """Integration tests for stacking functionality."""

    @pytest.mark.asyncio
    async def test_stacking_validation_errors(self):
        """Test that stacking validation raises appropriate errors."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        bot = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm],
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                    # Note: no stacker LLM
                },
            ),
        )

        # Test missing stacker LLM
        with pytest.raises(ValueError, match="STACKING aggregation strategy requires a stacker LLM"):
            await bot._aggregate_predictions(
                predictions=[0.5],
                question=Mock(),
                research="test research",
                reasoned_predictions=[Mock()],
            )

        # Add stacker LLM
        bot._pipeline.stacker_llm = test_llm

        # Test missing reasoned predictions
        with pytest.raises(
            ValueError,
            match="STACKING aggregation strategy requires reasoned predictions",
        ):
            await bot._aggregate_predictions(
                predictions=[0.5],
                question=Mock(),
                research="test research",
                reasoned_predictions=None,
            )

        # Test missing research
        with pytest.raises(ValueError, match="STACKING aggregation strategy requires research context"):
            await bot._aggregate_predictions(
                predictions=[0.5],
                question=Mock(),
                research=None,
                reasoned_predictions=[Mock()],
            )

    @pytest.mark.asyncio
    async def test_stacking_fallback_behavior(self):
        """Test fallback behavior when stacking fails."""
        from metaculus_bot.llm_configs import STACKER_FALLBACK_LLM

        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        # Create bot with fallback enabled
        bot = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm],
                    "stacker": test_llm,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
            stacking_fallback_on_failure=True,
        )

        # Mock pipeline.run_stacking to raise an exception. After primary fails we
        # fall through to fallback LLM then to MEDIAN aggregation; patch the
        # fallback stacker constructor call path so both stacker attempts
        # raise, landing us in MEDIAN.
        mock_question = Mock(spec=BinaryQuestion)
        mock_question.id_of_question = 42
        with (
            patch.object(
                bot._pipeline, "run_stacking", side_effect=RuntimeError("Stacking failed")
            ) as mock_run_stacking,
            patch.object(bot._pipeline, "strategy", AggregationStrategy.STACKING),
        ):
            result = await bot._aggregate_predictions(
                predictions=[0.4, 0.6],
                question=mock_question,
                research="test research",
                reasoned_predictions=[Mock(), Mock()],
            )
            # Median of [0.4, 0.6] = 0.5 — both the primary and fallback
            # stackers fail, so we end up in the MEDIAN branch (was MEAN
            # before; changed as part of the fallback hardening).
            assert result == 0.5
            # F8: assert the degradation counters were tripped as expected.
            # Primary fails → primary counter, then fallback is attempted
            # (counter tripped at invocation time) and also fails.
            assert bot._pipeline.counters.stacker_primary_failed_count == 1
            assert bot._pipeline.counters.stacker_fallback_used_count == 1
            assert bot._pipeline.counters.stacker_fallback_failed_count == 1
            # Two pipeline.run_stacking calls expected: primary + fallback.
            assert mock_run_stacking.call_count == 2
            # The second call must have received the fallback LLM override.
            assert mock_run_stacking.call_args_list[1].kwargs["stacker_llm_override"] is STACKER_FALLBACK_LLM

    @pytest.mark.asyncio
    async def test_stacking_fallback_success_middle_path(self):
        """F7: Primary stacker fails, fallback stacker succeeds.

        The middle path (primary raises → fallback succeeds → return fallback value)
        was untested before: prior tests either covered primary success or both
        failing. This asserts the counters reflect "one failure, one successful
        recovery, no MEDIAN fallback needed".
        """
        from metaculus_bot.llm_configs import STACKER_FALLBACK_LLM

        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        bot = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm],
                    "stacker": test_llm,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
            stacking_fallback_on_failure=True,
        )

        mock_question = Mock(spec=BinaryQuestion)
        mock_question.id_of_question = 99
        # side_effect sequence: primary raises, fallback returns 0.7.
        with patch.object(
            bot._pipeline, "run_stacking", side_effect=[RuntimeError("primary failed"), 0.7]
        ) as mock_run_stacking:
            result = await bot._aggregate_predictions(
                predictions=[0.4, 0.6],
                question=mock_question,
                research="test research",
                reasoned_predictions=[Mock(), Mock()],
            )

        assert result == 0.7, "Fallback stacker's value must be returned (not MEDIAN=0.5)."
        assert bot._pipeline.counters.stacker_primary_failed_count == 1
        assert bot._pipeline.counters.stacker_fallback_used_count == 1
        assert bot._pipeline.counters.stacker_fallback_failed_count == 0
        assert mock_run_stacking.call_count == 2
        # Verify the fallback call received the fallback LLM as override.
        assert mock_run_stacking.call_args_list[1].kwargs["stacker_llm_override"] is STACKER_FALLBACK_LLM

    @pytest.mark.asyncio
    async def test_stacking_no_fallback_raises_error(self):
        """Test that stacking without fallback raises errors."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        bot = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm],
                    "stacker": test_llm,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
            stacking_fallback_on_failure=False,  # No fallback
        )

        # Real BinaryQuestion (not Mock(spec=...)) because _aggregate_predictions
        # reads question.id_of_question to assert non-None and key into
        # self._pipeline.outcomes on the success/fallback paths. A Mock(spec=...)
        # returns a MagicMock for id_of_question, which would key the dict with
        # a Mock object instead of an int — fine for the no-fallback path tested
        # here (which raises before the dict write), but a real question is
        # future-proof if this test ever exercises the success path.
        question = BinaryQuestion(
            question_text="Will it happen?",
            background_info="bg",
            resolution_criteria="rc",
            fine_print="",
            page_url="https://test.com/no-fallback",
            id_of_question=999,
            open_time=_stub_open_time(),
            scheduled_resolution_time=_stub_resolve_time(),
        )

        # Mock pipeline.run_stacking to raise an exception
        with (
            patch.object(bot._pipeline, "run_stacking", side_effect=RuntimeError("Stacking failed")),
            pytest.raises(RuntimeError, match="Stacking failed"),
        ):
            await bot._aggregate_predictions(
                predictions=[0.4, 0.6],
                question=question,
                research="test research",
                reasoned_predictions=[Mock(), Mock()],
            )


class TestStackingMethods:
    """Test individual stacking methods."""

    @pytest.mark.asyncio
    async def test_run_stacking_question_type_routing(self):
        """Test that pipeline.run_stacking routes to the correct method per question type."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        bot = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm],
                    "stacker": test_llm,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
        )

        reasoned_preds: ReasonedPredictionList = [ReasonedPrediction(prediction_value=0.6, reasoning="test")]

        # Mock the stacking helper functions to check routing and return values
        with (
            patch("metaculus_bot.stacking.run_stacking_binary", return_value=(0.5, "meta")) as mock_binary,
            patch("metaculus_bot.stacking.run_stacking_mc", return_value=(Mock(), "meta")) as mock_mc,
            patch(
                "metaculus_bot.stacking.run_stacking_numeric",
                return_value=(
                    [
                        Percentile(value=1.0, percentile=0.01),
                        Percentile(value=2.5, percentile=0.025),
                        Percentile(value=5.0, percentile=0.05),
                        Percentile(value=10.0, percentile=0.10),
                        Percentile(value=20.0, percentile=0.20),
                        Percentile(value=40.0, percentile=0.40),
                        Percentile(value=50.0, percentile=0.50),
                        Percentile(value=60.0, percentile=0.60),
                        Percentile(value=80.0, percentile=0.80),
                        Percentile(value=90.0, percentile=0.90),
                        Percentile(value=95.0, percentile=0.95),
                        Percentile(value=97.5, percentile=0.975),
                        Percentile(value=99.0, percentile=0.99),
                    ],
                    "meta",
                ),
            ) as mock_numeric,
        ):
            # Test binary question routing
            binary_question = Mock(spec=BinaryQuestion)
            binary_question.id_of_question = 101
            binary_question.page_url = "https://example.com/q/101"
            binary_question.question_text = "Test binary question?"
            binary_question.background_info = "Test background"
            binary_question.resolution_criteria = "Test resolution criteria"
            binary_question.fine_print = "Test fine print"
            result = await bot._pipeline.run_stacking(binary_question, "research", reasoned_preds)
            mock_binary.assert_called_once()
            mock_mc.assert_not_called()
            mock_numeric.assert_not_called()
            assert result == 0.5

            # Verify the call arguments for the helper: (stacker_llm, parser_llm, question)
            # positionally, then research + base_texts keyword-only.
            assert mock_binary.call_args.args[2] == binary_question
            assert mock_binary.call_args.kwargs["research"] == "research"
            assert mock_binary.call_args.kwargs["base_texts"] == ["test"]

            # Reset mocks
            mock_binary.reset_mock()
            mock_mc.reset_mock()
            mock_numeric.reset_mock()

            # Test multiple choice question routing
            mc_question = Mock(spec=MultipleChoiceQuestion)
            mc_question.id_of_question = 102
            mc_question.question_text = "Test MC question?"
            mc_question.background_info = "Test background"
            mc_question.resolution_criteria = "Test resolution criteria"
            mc_question.fine_print = "Test fine print"
            mc_question.page_url = "https://example.com/q/102"
            mc_question.options = ["A", "B", "C"]
            result = await bot._pipeline.run_stacking(mc_question, "research", reasoned_preds)
            mock_binary.assert_not_called()
            mock_mc.assert_called_once()
            mock_numeric.assert_not_called()

            # Reset mocks
            mock_binary.reset_mock()
            mock_mc.reset_mock()
            mock_numeric.reset_mock()

            # Test numeric question routing
            numeric_question = Mock(spec=NumericQuestion)
            numeric_question.id_of_question = 103
            numeric_question.question_text = "Test numeric question?"
            numeric_question.background_info = "Test background"
            numeric_question.resolution_criteria = "Test resolution criteria"
            numeric_question.fine_print = "Test fine print"
            numeric_question.page_url = "https://example.com/q/103"
            numeric_question.upper_bound = 100
            numeric_question.lower_bound = 0
            numeric_question.open_upper_bound = False
            numeric_question.open_lower_bound = False
            numeric_question.unit_of_measure = "units"
            numeric_question.zero_point = None
            numeric_question.cdf_size = 201
            # spec=NumericQuestion does not expose Pydantic field names, and nominal_bounds
            # reads these two directly (None means "fall back to the hard bounds").
            numeric_question.nominal_upper_bound = None
            numeric_question.nominal_lower_bound = None
            result = await bot._pipeline.run_stacking(numeric_question, "research", reasoned_preds)
            mock_binary.assert_not_called()
            mock_mc.assert_not_called()
            mock_numeric.assert_called_once()
            # Numeric path returns a NumericDistribution
            from forecasting_tools.data_models.numeric_report import (
                NumericDistribution as FTNumericDistribution,
            )

            assert isinstance(result, FTNumericDistribution)

    @pytest.mark.asyncio
    async def test_run_stacking_unsupported_question_type(self):
        """Test that unsupported question types raise an error from pipeline.run_stacking."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        bot = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm],
                    "stacker": test_llm,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
        )

        # A conditional question is the one framework type the bot never stacks (date
        # questions stack through the numeric path since 2026-09).
        from forecasting_tools.data_models.questions import ConditionalQuestion

        unsupported_question = Mock(spec=ConditionalQuestion)
        unsupported_question.id_of_question = 104
        unsupported_question.page_url = "https://example.com/q/104"
        reasoned_preds: ReasonedPredictionList = [ReasonedPrediction(prediction_value=0.6, reasoning="test")]

        with pytest.raises(ValueError, match="Unsupported question type for stacking"):
            await bot._pipeline.run_stacking(unsupported_question, "research", reasoned_preds)

    def test_stacking_model_name_stripping_edge_cases(self):
        """Test edge cases in model name stripping logic."""
        # Test reasoning without model prefix
        reasoning1 = "This is analysis without model prefix"
        if reasoning1.startswith("Model: "):
            lines = reasoning1.split("\n", 2)
            if len(lines) >= 3 and lines[1] == "":
                reasoning1 = lines[2]
        assert reasoning1 == "This is analysis without model prefix"

        # Test reasoning with model prefix but no empty line
        reasoning2 = "Model: gpt-4\nThis continues immediately"
        if reasoning2.startswith("Model: "):
            lines = reasoning2.split("\n", 2)
            if len(lines) >= 3 and lines[1] == "":
                reasoning2 = lines[2]
            # Should not strip if no empty line
        assert "Model: gpt-4" in reasoning2

        # Test reasoning with model prefix and empty line
        reasoning3 = "Model: claude-3\n\nThis is the actual analysis"
        if reasoning3.startswith("Model: "):
            lines = reasoning3.split("\n", 2)
            if len(lines) >= 3 and lines[1] == "":
                reasoning3 = lines[2]
        assert reasoning3 == "This is the actual analysis"


class TestStackingResearchAndMakePredictions:
    """Test stacking integration in _research_and_make_predictions."""

    @pytest.mark.asyncio
    async def test_stacking_in_research_and_make_predictions(self):
        """Test that stacking is properly integrated in _research_and_make_predictions."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        bot = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm, test_llm],  # 2 forecasters
                    "stacker": test_llm,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
            min_forecasters_to_publish=1,
        )

        question = Mock()

        # Mock predictions from forecasters. Declared before the patch block because
        # the gather stub is built from them (see gather_predictions_stub in conftest:
        # stubbing _forecaster_with_soft_deadline with an AsyncMock makes each inline
        # call in _research_and_make_predictions create a real coroutine, and the
        # gather stand-in has to close them or they surface as "never awaited"
        # RuntimeWarnings pinned to an unrelated test).
        pred1 = ReasonedPrediction(prediction_value=0.6, reasoning="Analysis 1")
        pred2 = ReasonedPrediction(prediction_value=0.8, reasoning="Analysis 2")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="test research"),
            patch.object(
                bot,
                "_gather_predictions_with_wall_clock",
                new=gather_predictions_stub(([pred1, pred2], [], None)),
            ),
            patch.object(bot._pipeline, "stack_predictions", return_value=0.7) as mock_aggregate,
            patch.object(
                bot,
                "_forecaster_with_soft_deadline",
                new=AsyncMock(return_value=ReasonedPrediction(prediction_value=0.5, reasoning="stub")),
            ),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0)

            # Call the method
            result = await bot._research_and_make_predictions(question)

            # Verify stacking path was taken
            mock_aggregate.assert_called_once()
            call_args = mock_aggregate.call_args

            # Check that predictions were passed correctly
            assert call_args[0][0] == [0.6, 0.8]  # prediction values
            assert call_args[1]["research"] == "test research"
            assert call_args[1]["reasoned_predictions"] == [pred1, pred2]

            # Check that result contains single aggregated prediction
            assert len(result.predictions) == 1
            assert result.predictions[0].prediction_value == 0.7
            assert "Stacked prediction" in result.predictions[0].reasoning

    @pytest.mark.asyncio
    async def test_non_stacking_preserves_multiple_predictions(self):
        """Test that non-stacking strategies preserve multiple predictions."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        bot = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.MEAN,  # Not stacking
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm, test_llm],
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
            min_forecasters_to_publish=1,
        )

        question = Mock()

        pred1 = ReasonedPrediction(prediction_value=0.6, reasoning="Analysis 1")
        pred2 = ReasonedPrediction(prediction_value=0.8, reasoning="Analysis 2")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="test research"),
            patch.object(
                bot,
                "_gather_predictions_with_wall_clock",
                new=gather_predictions_stub(([pred1, pred2], [], None)),
            ),
            patch.object(
                bot,
                "_forecaster_with_soft_deadline",
                new=AsyncMock(return_value=ReasonedPrediction(prediction_value=0.5, reasoning="stub")),
            ),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0)

            result = await bot._research_and_make_predictions(question)

            # Should preserve both predictions for framework to aggregate
            assert len(result.predictions) == 2
            assert result.predictions[0].prediction_value == 0.6
            assert result.predictions[1].prediction_value == 0.8


class TestStackingBenchmarkConfiguration:
    """Test stacking configuration in benchmark context."""

    def test_benchmark_stacking_configuration(self):
        """Test that stacking bots are properly configured for benchmarking."""
        from main import TemplateForecaster as BenchmarkForecaster
        from metaculus_bot.aggregation_strategies import AggregationStrategy

        # Test configuration similar to what's used in community_benchmark.py
        test_llm = GeneralLlm(model="test-model", temperature=0.0)

        # Simulate base models and stacking model configuration
        base_forecasters = [test_llm, test_llm, test_llm]  # 3 base models
        stacker = test_llm

        stacking_bot = BenchmarkForecaster(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            publish_reports_to_metaculus=False,
            folder_to_save_reports_to=None,
            skip_previously_forecasted_questions=False,
            research_provider=None,
            max_questions_per_run=None,
            is_benchmarking=True,
            research_cache={},
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": base_forecasters,
                    "stacker": stacker,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
            max_concurrent_research=2,
            stacking_fallback_on_failure=False,  # Fail in benchmarking
            stacking_randomize_order=True,  # Avoid position bias
        )

        assert stacking_bot.aggregation_strategy == AggregationStrategy.STACKING
        assert len(stacking_bot._forecaster_llms) == 3  # All base models
        assert stacking_bot._pipeline.stacker_llm is not None
        assert stacking_bot.stacking_fallback_on_failure is False
        assert stacking_bot.stacking_randomize_order is True
        assert stacking_bot.is_benchmarking is True


class TestStackingGuardsAndReasoning:
    """Additional tests for STACKING guard path and reasoning propagation."""

    @pytest.mark.asyncio
    async def test_stacking_guard_single_element(self):
        """STACKING base aggregator returns pre-stacked single prediction as-is."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)
        bot = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
        )

        result = await bot._aggregate_predictions(predictions=[0.42], question=Mock())
        assert result == 0.42

    @pytest.mark.asyncio
    async def test_stacking_guard_multi_element_mean_binary(self):
        """STACKING guard averages multiple pre-stacked binary predictions by mean."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)
        bot = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
        )

        result = await bot._aggregate_predictions(predictions=[0.4, 0.6], question=Mock())
        assert result == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_stacking_guard_multi_element_mean_mc(self):
        """STACKING guard mean-aggregates multiple pre-stacked MC predictions."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)
        bot = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
        )
        pred1 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.7),
                PredictedOption(option_name="B", probability=0.3),
            ]
        )
        pred2 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.5),
                PredictedOption(option_name="B", probability=0.5),
            ]
        )
        result = await bot._aggregate_predictions(predictions=[pred1, pred2], question=Mock())
        assert isinstance(result, PredictedOptionList)
        # Expect mean: A=(0.7+0.5)/2=0.6, B=0.4
        a_prob = next(o.probability for o in result.predicted_options if o.option_name == "A")
        b_prob = next(o.probability for o in result.predicted_options if o.option_name == "B")
        assert a_prob == pytest.approx(0.6)
        assert b_prob == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_stacking_guard_multi_element_mean_numeric(self):
        """STACKING guard mean-aggregates numeric distributions."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)
        bot = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
        )
        # Build a minimal numeric question and two distributions on same bounds
        num_q = Mock(spec=NumericQuestion)
        num_q.id_of_question = 999
        # A Metaculus page_url, set explicitly because spec=NumericQuestion does not expose Pydantic
        # field names: the publish path reads the platform off it for the Mantic-only tail floor.
        num_q.page_url = "https://www.metaculus.com/questions/999/"
        num_q.open_upper_bound = False
        num_q.open_lower_bound = False
        num_q.upper_bound = 100.0
        num_q.lower_bound = 0.0
        num_q.zero_point = None
        num_q.cdf_size = 201

        pcts1 = [
            Percentile(value=10.0, percentile=0.1),
            Percentile(value=90.0, percentile=0.9),
        ]
        pcts2 = [
            Percentile(value=20.0, percentile=0.1),
            Percentile(value=80.0, percentile=0.9),
        ]
        d1 = NumericDistribution(
            declared_percentiles=pcts1,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
            cdf_size=201,
        )
        d2 = NumericDistribution(
            declared_percentiles=pcts2,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
            cdf_size=201,
        )

        result = await bot._aggregate_predictions(predictions=[d1, d2], question=num_q)
        # The result should be a NumericDistribution on same bounds with a CDF of 201 points
        assert isinstance(result, NumericDistribution)
        assert len(result.cdf) == 201

    @pytest.mark.asyncio
    async def test_stacker_reasoning_propagated(self):
        """Stacker meta-analysis AND base model reasoning should both be preserved
        in the final ReasonedPrediction so residual analysis can still attribute
        per-model forecasts even when stacking overrode the base aggregation."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)
        bot = TemplateForecaster(
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                LlmsConfig,
                {
                    "forecasters": [test_llm, test_llm],
                    "stacker": test_llm,
                    "default": test_llm,
                    "parser": test_llm,
                    "researcher": test_llm,
                    "summarizer": test_llm,
                },
            ),
            min_forecasters_to_publish=1,
        )

        question = Mock(spec=BinaryQuestion)
        question.id_of_question = 999

        pred1 = ReasonedPrediction(prediction_value=0.6, reasoning="Analysis 1")
        pred2 = ReasonedPrediction(prediction_value=0.8, reasoning="Analysis 2")

        # Mock internals to avoid network
        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="test research"),
            patch.object(
                bot,
                "_gather_predictions_with_wall_clock",
                new=gather_predictions_stub(([pred1, pred2], [], None)),
            ),
            patch.object(bot._pipeline, "run_stacking", return_value=0.7),
            patch.object(
                bot,
                "_forecaster_with_soft_deadline",
                new=AsyncMock(return_value=ReasonedPrediction(prediction_value=0.5, reasoning="stub")),
            ),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)

            # Pre-store meta-analysis as if produced by stacker LLM
            bot._pipeline.meta_reasoning[999] = "Meta-analysis text"

            result = await bot._research_and_make_predictions(question)

            assert len(result.predictions) == 1
            assert result.predictions[0].prediction_value == 0.7
            reasoning = result.predictions[0].reasoning
            assert "Meta-analysis text" in reasoning
            assert "Analysis 1" in reasoning
            assert "Analysis 2" in reasoning
            assert "## Stacker Meta-Analysis" in reasoning
            assert "## Base Model Reasoning" in reasoning


class TestStackerTransientRetry:
    """The stacker invoke is wrapped in the elapsed-gated transient retry.

    STACKER_LLM / STACKER_FALLBACK_LLM are allowed_tries=1 by design (prefer the
    cross-provider fallback over retrying a stalled provider). The wrapper adds
    recovery from instant aiohttp blips (litellm #14895) WITHOUT retrying a slow
    stall — a slow stall still propagates so the fallback chain engages.
    """

    @staticmethod
    def _binary_question() -> "BinaryQuestion":
        q = Mock(spec=BinaryQuestion)
        q.id_of_question = 555
        q.page_url = "https://example.com/q/555"
        q.question_text = "Will it rain?"
        q.background_info = "bg"
        q.resolution_criteria = "rc"
        q.fine_print = "fp"
        q.open_time = _stub_open_time()
        q.scheduled_resolution_time = _stub_resolve_time()
        return cast(BinaryQuestion, q)

    @pytest.mark.asyncio
    async def test_fast_stacker_blip_retries_then_succeeds(self) -> None:
        """A fast litellm.Timeout on the stacker invoke is retried; the next call wins."""
        import litellm.exceptions as litellm_exc

        from metaculus_bot.stacking import run_stacking_binary

        stacker = Mock(spec=GeneralLlm)
        stacker.invoke = AsyncMock(
            side_effect=[litellm_exc.Timeout("blip", model="m", llm_provider="openrouter"), "meta text"]
        )
        stacker.model = "stub-stacker"
        parser = Mock(spec=GeneralLlm)

        from metaculus_bot.value_extraction import ExtractionOutcome

        with (
            patch("metaculus_bot.llm_retry.asyncio.sleep", new=AsyncMock()),
            patch(
                "metaculus_bot.stacking.extract_binary",
                new=AsyncMock(return_value=ExtractionOutcome(value=0.42, rung="block", block_present=True)),
            ),
        ):
            value, meta = await run_stacking_binary(
                stacker,
                parser,
                self._binary_question(),
                research="research",
                base_texts=["base reasoning"],
                stacker_wall_timeout=500.0,
            )

        assert value == 0.42
        assert meta == "meta text"
        assert stacker.invoke.await_count == 2

    @pytest.mark.asyncio
    async def test_slow_stacker_stall_not_retried(self) -> None:
        """A stacker invoke that stalls past the elapsed gate is NOT retried — it propagates.

        Patches the elapsed clock so the failure reads as slow; the stacker layer
        must surface the exception (one invoke) so the pipeline's fallback chain
        engages rather than the wrapper silently retrying a multi-minute stall.
        """
        import litellm.exceptions as litellm_exc

        from metaculus_bot.llm_retry import TRANSIENT_RETRY_MAX_ELAPSED_S
        from metaculus_bot.stacking import run_stacking_binary

        stacker = Mock(spec=GeneralLlm)
        stacker.invoke = AsyncMock(side_effect=litellm_exc.Timeout("stall", model="m", llm_provider="openrouter"))
        parser = Mock(spec=GeneralLlm)

        clock = iter([0.0] + [TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0] * 10)

        with (
            patch("metaculus_bot.llm_retry.time.monotonic", lambda: next(clock)),
            pytest.raises(litellm_exc.Timeout),
        ):
            await run_stacking_binary(
                stacker,
                parser,
                self._binary_question(),
                research="research",
                base_texts=["base reasoning"],
                stacker_wall_timeout=500.0,
            )

        assert stacker.invoke.await_count == 1


class TestStackingNumericParseNotes:
    """The numeric stacker's parser instruction must list the full 13-percentile set."""

    @staticmethod
    def _numeric_question(*, open_lower_bound: bool = False, open_upper_bound: bool = False) -> "NumericQuestion":
        q = Mock(spec=NumericQuestion)
        q.id_of_question = 777
        q.page_url = "https://example.com/q/777"
        q.question_text = "How many?"
        q.background_info = "bg"
        q.resolution_criteria = "rc"
        q.fine_print = "fp"
        q.unit_of_measure = "USD"
        q.lower_bound = 0.0
        q.upper_bound = 1000.0
        q.open_lower_bound = open_lower_bound
        q.open_upper_bound = open_upper_bound
        q.zero_point = None
        q.cdf_size = 201
        q.nominal_upper_bound = None
        q.nominal_lower_bound = None
        q.open_time = _stub_open_time()
        q.scheduled_resolution_time = _stub_resolve_time()
        return cast(NumericQuestion, q)

    @staticmethod
    async def _capture_parse_notes(question: "NumericQuestion") -> str:
        """Run run_stacking_numeric with the extraction ladder stubbed and return the parser notes."""
        from metaculus_bot.stacking import run_stacking_numeric
        from metaculus_bot.value_extraction import ExtractionOutcome

        stacker = Mock(spec=GeneralLlm)
        stacker.invoke = AsyncMock(return_value="meta text")
        stacker.model = "stub-stacker"
        parser = Mock(spec=GeneralLlm)

        captured: dict[str, str] = {}

        async def _capture(*_args, prompt_notes: str = "", **_kwargs):
            captured["notes"] = prompt_notes
            return ExtractionOutcome(value=[], rung="block", block_present=True)

        with patch("metaculus_bot.stacking.extract_numeric", new=AsyncMock(side_effect=_capture)):
            await run_stacking_numeric(
                stacker,
                parser,
                question,
                research="research",
                base_texts=["base reasoning"],
                lower_bound_message="lower msg",
                upper_bound_message="upper msg",
                stacker_wall_timeout=500.0,
            )

        return captured["notes"]

    @pytest.mark.asyncio
    async def test_numeric_parse_notes_list_all_13_generated(self) -> None:
        """run_stacking_numeric passes a parser instruction naming all 13 labels (incl. 1 and 99)."""
        from metaculus_bot.numeric.config import STANDARD_PERCENTILES_CSV

        notes = await self._capture_parse_notes(self._numeric_question())
        assert STANDARD_PERCENTILES_CSV in notes
        assert STANDARD_PERCENTILES_CSV == "1,2.5,5,10,20,40,50,60,80,90,95,97.5,99"
        assert "13 percentiles" in notes
        assert "2.5,5,10,20,40,50,60,80,90,95,97.5." not in notes

    @pytest.mark.asyncio
    async def test_numeric_parse_notes_open_bound_never_clamps(self) -> None:
        """Open-bound questions must NOT get a hard "within [lo, hi]" clamp in the parser notes.

        The stacker follows open-bound guidance to place mass beyond the displayed range;
        an obedient parser must extract those out-of-range values verbatim instead of
        clamping them back in. Delegating to build_parse_notes gives us that open/closed
        awareness (thoroughly matrixed in tests/test_forecaster_runners_parse_notes.py).
        """
        notes = await self._capture_parse_notes(self._numeric_question(open_lower_bound=True))

        assert "within [" not in notes
        assert "resolve below" in notes
        assert "never clamp" in notes
