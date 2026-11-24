"""Tests for the extracted aggregation pipeline module.

Exercises AggregationPipeline's three main paths:
1. Base-combine re-entry (stacking already done, parent class calls aggregate)
2. Stacking fallback chain (primary -> fallback LLM -> median)
3. Simple MEAN/MEDIAN aggregation
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.aggregation_pipeline import AggregationCounters, AggregationPipeline
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import THIN_PUBLISH_BINARY_CEIL, THIN_PUBLISH_BINARY_FLOOR
from metaculus_bot.member_forecast import (
    NUMERIC_COMBINE_METHOD_SINGLE,
    NUMERIC_COMBINE_METHOD_STACKED,
    NUMERIC_COMBINE_METHOD_UNRECORDED,
)
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.date_axis import EpochDateQuestion, numeric_view
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric.utils import aggregate_numeric
from tests.conftest import make_mock_numeric_question
from tests.mantic_fakes import load_preseason_date_question
from tests.pipeline_test_helpers import (
    assert_server_accepts_cdf,
    cdf_heights,
    certain_of_bin,
    make_count_question,
    make_e2e_bot,
    make_real_numeric_question,
    metaculus_url,
    on_mantic,
    pmf_of,
)


def _make_binary_question(qid: int = 100) -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will X happen?",
        id_of_question=qid,
        page_url="https://metaculus.com/questions/100",
        api_json={"type": "binary"},
    )


def _make_mc_question(qid: int = 200) -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which will happen?",
        id_of_question=qid,
        page_url="https://metaculus.com/questions/200",
        options=["A", "B", "C"],
        api_json={"type": "multiple_choice"},
    )


def _make_pipeline(
    strategy: AggregationStrategy = AggregationStrategy.CONDITIONAL_STACKING,
    stacking_fallback_on_failure: bool = True,
) -> AggregationPipeline:
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    return AggregationPipeline(
        strategy=strategy,
        stacker_llm=test_llm,
        parser_llm=test_llm,
        stacking_fallback_on_failure=stacking_fallback_on_failure,
        stacking_randomize_order=False,
        stacking_spread_thresholds={"binary": 0.15, "mc": 0.20, "numeric": 0.15},
        discrete_integer_votes=defaultdict(list),
    )


class TestAggregationCounters:
    def test_initial_values_are_zero(self):
        counters = AggregationCounters()
        assert counters.stacking_expected_combine_count == 0
        assert counters.stacking_unexpected_combine_count == 0
        assert counters.stacking_fallback_count == 0
        assert counters.stacker_primary_failed_count == 0
        assert counters.stacker_fallback_used_count == 0
        assert counters.stacker_fallback_failed_count == 0
        assert counters.conditional_stacking_triggered_count == 0
        assert counters.conditional_stacking_skipped_count == 0
        assert counters.conditional_stacking_skipped_single_forecaster_count == 0
        assert counters.conditional_stacking_crux_failures == 0
        assert counters.conditional_stacking_search_failures == 0


class TestBaseCombineReentry:
    """Test the base-combine path: reasoned_predictions=None and research=None."""

    @pytest.mark.asyncio
    async def test_empty_predictions_do_not_consume_expected_combine_state(self) -> None:
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=106)
        pipeline.register_expected_base_combine(question)

        with pytest.raises(ValueError, match="Cannot aggregate empty list of predictions"):
            pipeline.base_combine(
                predictions=[],
                question=question,
            )

        assert pipeline.expected_base_combines == {106}
        assert pipeline.counters == AggregationCounters()

    @pytest.mark.asyncio
    async def test_single_prediction_returns_as_is(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=101)
        pipeline.register_expected_base_combine(question)

        result = pipeline.base_combine(
            predictions=[0.42],
            question=question,
        )

        assert result == 0.42
        assert pipeline.counters.stacking_expected_combine_count == 1

    @pytest.mark.asyncio
    async def test_unexpected_combine_increments_counter(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=102)
        # Do NOT register expected combine

        result = pipeline.base_combine(
            predictions=[0.55],
            question=question,
        )

        assert result == 0.55
        assert pipeline.counters.stacking_unexpected_combine_count == 1

    @pytest.mark.asyncio
    async def test_multiple_binary_uses_median_for_conditional_stacking(self):
        pipeline = _make_pipeline(strategy=AggregationStrategy.CONDITIONAL_STACKING)
        question = _make_binary_question(qid=103)
        pipeline.register_expected_base_combine(question)

        result = pipeline.base_combine(
            predictions=[0.30, 0.50, 0.70],
            question=question,
        )

        # Median of [0.30, 0.50, 0.70] = 0.50
        assert result == 0.50

    @pytest.mark.asyncio
    async def test_multiple_binary_uses_mean_for_stacking(self):
        pipeline = _make_pipeline(strategy=AggregationStrategy.STACKING)
        question = _make_binary_question(qid=104)
        pipeline.register_expected_base_combine(question)

        result = pipeline.base_combine(
            predictions=[0.30, 0.50, 0.70],
            question=question,
        )

        # Mean of [0.30, 0.50, 0.70] = 0.50 (coincidentally same as median here)
        assert abs(cast(float, result) - 0.50) < 0.01

    @pytest.mark.asyncio
    async def test_multiple_mc_uses_median_for_conditional_stacking(self):
        pipeline = _make_pipeline(strategy=AggregationStrategy.CONDITIONAL_STACKING)
        question = _make_mc_question(qid=105)
        pipeline.register_expected_base_combine(question)

        pol1 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.5),
                PredictedOption(option_name="B", probability=0.3),
                PredictedOption(option_name="C", probability=0.2),
            ]
        )
        pol2 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.6),
                PredictedOption(option_name="B", probability=0.2),
                PredictedOption(option_name="C", probability=0.2),
            ]
        )
        pol3 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.7),
                PredictedOption(option_name="B", probability=0.1),
                PredictedOption(option_name="C", probability=0.2),
            ]
        )

        result = pipeline.base_combine(
            predictions=[pol1, pol2, pol3],
            question=question,
        )

        assert isinstance(result, PredictedOptionList)
        probs = {o.option_name: o.probability for o in result.predicted_options}
        # Median of A: [0.5, 0.6, 0.7] = 0.6
        # Median of B: [0.3, 0.2, 0.1] = 0.2
        # Median of C: [0.2, 0.2, 0.2] = 0.2
        # After renormalization: A=0.6, B=0.2, C=0.2
        assert abs(probs["A"] - 0.6) < 0.01
        assert abs(probs["B"] - 0.2) < 0.01


class TestStackingFallbackChain:
    """Test the primary -> fallback -> median chain."""

    @pytest.mark.asyncio
    async def test_empty_predictions_fail_before_context_validation_or_state_changes(self) -> None:
        pipeline = _make_pipeline()
        pipeline.stacker_llm = None
        pipeline.outcomes[200] = "skipped"
        question = _make_binary_question(qid=200)

        with pytest.raises(ValueError, match="Cannot aggregate empty list of predictions"):
            await pipeline.stack_predictions(
                predictions=[],
                question=question,
                research=None,
                reasoned_predictions=None,
                aggregated_tool_output=None,
            )

        assert pipeline.outcomes == {200: "skipped"}
        assert pipeline.counters == AggregationCounters()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_cancelling_primary_attempt_propagates_without_recording_a_failure(self) -> None:
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=205)
        predictions: list[PredictionTypes] = [0.20, 0.80]
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.20, reasoning="Model: m1\n\nLow"),
            ReasonedPrediction(prediction_value=0.80, reasoning="Model: m2\n\nHigh"),
        ]
        attempt_started = asyncio.Event()
        attempt_cancelled = asyncio.Event()

        async def wait_for_cancellation(*args: Any, **kwargs: Any) -> PredictionTypes:
            attempt_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                attempt_cancelled.set()
            raise AssertionError("unreachable")

        with patch.object(pipeline, "run_stacking", side_effect=wait_for_cancellation):
            aggregate_task = asyncio.create_task(
                pipeline.stack_predictions(
                    predictions=predictions,
                    question=question,
                    research="test research",
                    reasoned_predictions=reasoned,
                )
            )
            await attempt_started.wait()
            aggregate_task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await aggregate_task

        assert attempt_cancelled.is_set()
        assert 205 not in pipeline.outcomes
        assert pipeline.counters == AggregationCounters()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_cancelling_fallback_attempt_preserves_primary_failure_counts(self) -> None:
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=206)
        predictions: list[PredictionTypes] = [0.20, 0.80]
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.20, reasoning="Model: m1\n\nLow"),
            ReasonedPrediction(prediction_value=0.80, reasoning="Model: m2\n\nHigh"),
        ]
        fallback_started = asyncio.Event()
        fallback_cancelled = asyncio.Event()
        attempt_count = 0

        async def fail_primary_then_wait(*args: Any, **kwargs: Any) -> PredictionTypes:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise RuntimeError("primary failed")
            fallback_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                fallback_cancelled.set()
            raise AssertionError("unreachable")

        with patch.object(pipeline, "run_stacking", side_effect=fail_primary_then_wait):
            aggregate_task = asyncio.create_task(
                pipeline.stack_predictions(
                    predictions=predictions,
                    question=question,
                    research="test research",
                    reasoned_predictions=reasoned,
                )
            )
            await fallback_started.wait()
            aggregate_task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await aggregate_task

        assert fallback_cancelled.is_set()
        assert 206 not in pipeline.outcomes
        assert pipeline.counters == AggregationCounters(
            stacker_primary_failed_count=1,
            stacker_fallback_used_count=1,
        )

    @pytest.mark.asyncio
    async def test_primary_success_sets_outcome(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=201)
        predictions: list[PredictionTypes] = [0.20, 0.80, 0.50]
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.20, reasoning="Model: m1\n\nLow"),
            ReasonedPrediction(prediction_value=0.80, reasoning="Model: m2\n\nHigh"),
            ReasonedPrediction(prediction_value=0.50, reasoning="Model: m3\n\nMid"),
        ]

        with patch.object(pipeline, "run_stacking", new=AsyncMock(return_value=0.65)):
            result = await pipeline.stack_predictions(
                predictions=predictions,
                question=question,
                research="test research",
                reasoned_predictions=reasoned,
            )

        assert abs(cast(float, result) - 0.65) < 0.01
        assert pipeline.outcomes[201] == "primary"

    @pytest.mark.asyncio
    async def test_primary_failure_invokes_fallback(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=202)
        predictions: list[PredictionTypes] = [0.20, 0.80]
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.20, reasoning="Model: m1\n\nLow"),
            ReasonedPrediction(prediction_value=0.80, reasoning="Model: m2\n\nHigh"),
        ]

        call_count = 0

        async def mock_run_stacking(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("primary timed out")
            return 0.55

        with patch.object(pipeline, "run_stacking", side_effect=mock_run_stacking):
            result = await pipeline.stack_predictions(
                predictions=predictions,
                question=question,
                research="test research",
                reasoned_predictions=reasoned,
            )

        assert abs(cast(float, result) - 0.55) < 0.01
        assert pipeline.outcomes[202] == "fallback_llm"
        assert pipeline.counters.stacker_primary_failed_count == 1
        assert pipeline.counters.stacker_fallback_used_count == 1

    @pytest.mark.asyncio
    async def test_both_fail_uses_median(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=203)
        predictions: list[PredictionTypes] = [0.20, 0.80, 0.50]
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.20, reasoning="Model: m1\n\nLow"),
            ReasonedPrediction(prediction_value=0.80, reasoning="Model: m2\n\nHigh"),
            ReasonedPrediction(prediction_value=0.50, reasoning="Model: m3\n\nMid"),
        ]

        with patch.object(pipeline, "run_stacking", side_effect=TimeoutError("timed out")):
            result = await pipeline.stack_predictions(
                predictions=predictions,
                question=question,
                research="test research",
                reasoned_predictions=reasoned,
            )

        # Median of [0.20, 0.50, 0.80] = 0.50
        assert abs(cast(float, result) - 0.50) < 0.01
        assert pipeline.outcomes[203] == "fallback_median"
        assert pipeline.counters.stacker_primary_failed_count == 1
        assert pipeline.counters.stacker_fallback_used_count == 1
        assert pipeline.counters.stacker_fallback_failed_count == 1

    @pytest.mark.asyncio
    async def test_fallback_disabled_raises(self):
        pipeline = _make_pipeline(stacking_fallback_on_failure=False)
        question = _make_binary_question(qid=204)
        predictions: list[PredictionTypes] = [0.20, 0.80]
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.20, reasoning="Model: m1\n\nLow"),
            ReasonedPrediction(prediction_value=0.80, reasoning="Model: m2\n\nHigh"),
        ]

        with (
            patch.object(pipeline, "run_stacking", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await pipeline.stack_predictions(
                predictions=predictions,
                question=question,
                research="test research",
                reasoned_predictions=reasoned,
            )


class TestSimpleAggregation:
    """Test MEAN/MEDIAN strategies (no stacking)."""

    def test_empty_predictions_do_not_change_state(self) -> None:
        pipeline = _make_pipeline(strategy=AggregationStrategy.MEAN)
        pipeline.outcomes[300] = "existing"

        with pytest.raises(ValueError, match="Cannot aggregate empty list of predictions"):
            pipeline.simple_combine(predictions=[], question=_make_binary_question(qid=300))

        assert pipeline.outcomes == {300: "existing"}
        assert pipeline.counters == AggregationCounters()

    @pytest.mark.asyncio
    async def test_mean_binary(self):
        pipeline = _make_pipeline(strategy=AggregationStrategy.MEAN)
        question = _make_binary_question(qid=301)

        result = pipeline.simple_combine(
            predictions=[0.20, 0.40, 0.60],
            question=question,
        )

        # Mean of [0.20, 0.40, 0.60] = 0.40
        assert abs(cast(float, result) - 0.40) < 0.01

    @pytest.mark.asyncio
    async def test_median_binary(self):
        pipeline = _make_pipeline(strategy=AggregationStrategy.MEDIAN)
        question = _make_binary_question(qid=302)

        result = pipeline.simple_combine(
            predictions=[0.20, 0.40, 0.60],
            question=question,
        )

        # Median of [0.20, 0.40, 0.60] = 0.40
        assert abs(cast(float, result) - 0.40) < 0.01

    @pytest.mark.asyncio
    async def test_mean_mc(self):
        pipeline = _make_pipeline(strategy=AggregationStrategy.MEAN)
        question = _make_mc_question(qid=303)

        pol1 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.4),
                PredictedOption(option_name="B", probability=0.4),
                PredictedOption(option_name="C", probability=0.2),
            ]
        )
        pol2 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.6),
                PredictedOption(option_name="B", probability=0.2),
                PredictedOption(option_name="C", probability=0.2),
            ]
        )

        result = pipeline.simple_combine(predictions=[pol1, pol2], question=question)

        assert isinstance(result, PredictedOptionList)
        probs = {o.option_name: o.probability for o in result.predicted_options}
        # Mean of A: [0.4, 0.6] = 0.5, B: [0.4, 0.2] = 0.3, C: [0.2, 0.2] = 0.2
        assert abs(probs["A"] - 0.5) < 0.01
        assert abs(probs["B"] - 0.3) < 0.01
        assert abs(probs["C"] - 0.2) < 0.01


class TestRunStacking:
    """Test the run_stacking dispatch per question type."""

    @pytest.mark.asyncio
    async def test_binary_dispatches_to_stacking_module(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=401)
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.30, reasoning="Model: m1\n\nReasons"),
            ReasonedPrediction(prediction_value=0.70, reasoning="Model: m2\n\nReasons"),
        ]

        with patch("metaculus_bot.aggregation_pipeline.stacking.run_stacking_binary") as mock_stack:
            mock_stack.return_value = (0.55, "Meta analysis text")
            result = await pipeline.run_stacking(question, "research", reasoned)

        assert result == 0.55
        assert pipeline.meta_reasoning[401] == "Meta analysis text"

    @pytest.mark.asyncio
    async def test_mc_dispatches_to_stacking_module(self):
        pipeline = _make_pipeline()
        question = _make_mc_question(qid=402)
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(
                prediction_value=PredictedOptionList(
                    predicted_options=[
                        PredictedOption(option_name="A", probability=0.5),
                        PredictedOption(option_name="B", probability=0.3),
                        PredictedOption(option_name="C", probability=0.2),
                    ]
                ),
                reasoning="Model: m1\n\nReasons",
            ),
        ]

        expected_pol = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.6),
                PredictedOption(option_name="B", probability=0.25),
                PredictedOption(option_name="C", probability=0.15),
            ]
        )

        with patch("metaculus_bot.aggregation_pipeline.stacking.run_stacking_mc") as mock_stack:
            mock_stack.return_value = (expected_pol, "MC meta text")
            result = await pipeline.run_stacking(question, "research", reasoned)

        assert result == expected_pol
        assert pipeline.meta_reasoning[402] == "MC meta text"

    @pytest.mark.asyncio
    async def test_a_date_question_stacks_on_its_epoch_view(self, caplog):
        """The date branch is reachable only with NUMERIC_STACKING_ENABLED on (off in prod), so no
        live run exercises the ``numeric_view`` hand-off: the stacker must be handed the epoch
        adapter, the built distribution must keep the date axis, and the stacker's MEMBER_FORECAST
        line must say ``qtype=date``."""
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        pipeline = _make_pipeline()
        question = load_preseason_date_question()
        qid = question.id_of_question
        assert qid is not None
        window_start = datetime(2026, 9, 16, 2, tzinfo=UTC).timestamp()
        window_seconds = 20 * 3600
        epoch_percentiles = [
            Percentile(percentile=p, value=window_start + window_seconds * i / (len(STANDARD_PERCENTILES) - 1))
            for i, p in enumerate(STANDARD_PERCENTILES)
        ]
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.5, reasoning="Model: m1\n\nEarly."),
            ReasonedPrediction(prediction_value=0.5, reasoning="Model: m2\n\nLate."),
        ]

        with patch(
            "metaculus_bot.aggregation_pipeline.stacking.run_stacking_numeric",
            new=AsyncMock(return_value=(epoch_percentiles, "date meta")),
        ) as stacker:
            result = await pipeline.run_stacking(question, "research", reasoned)

        handed = stacker.call_args.args[2]
        assert isinstance(handed, EpochDateQuestion)
        assert handed.id_of_question == qid
        assert isinstance(result, NumericDistribution)
        assert result.is_date is True
        assert pipeline.meta_reasoning[qid] == "date meta"
        (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("MEMBER_FORECAST:")]
        assert f"question={qid} " in line
        assert " role=stacker " in line
        assert " qtype=date " in line


class TestThresholdLookup:
    def test_binary_threshold(self):
        pipeline = _make_pipeline()
        q = _make_binary_question()
        assert pipeline.get_threshold_for_question(q) == 0.15

    def test_mc_threshold(self):
        pipeline = _make_pipeline()
        q = _make_mc_question()
        assert pipeline.get_threshold_for_question(q) == 0.20

    def test_numeric_threshold(self):
        pipeline = _make_pipeline()
        q = make_mock_numeric_question(id_of_question=500)
        assert pipeline.get_threshold_for_question(q) == 0.15


class TestRegisterExpectedBaseCombine:
    def test_registers_and_discards(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=600)

        pipeline.register_expected_base_combine(question)
        assert 600 in pipeline.expected_base_combines

    def test_requires_question_id(self):
        pipeline = _make_pipeline()
        question = BinaryQuestion(
            question_text="Test",
            id_of_question=None,
            page_url="http://example.com",
            api_json={"type": "binary"},
        )

        with pytest.raises(AssertionError):
            pipeline.register_expected_base_combine(question)


class TestStackerNumericAttribution:
    @pytest.mark.asyncio
    async def test_sanitize_percentiles_receives_the_stacker_model_name(self):
        """NUMERIC_DEGENERATE_DECLARATION attributes by ``model_name``, and the telemetry
        archive reads ``model=unknown`` as "a caller forgot to pass it" — so the stacker
        path must wire its own model through, mirroring the forecaster_runners wiring test
        (test_sanitize_percentiles_receives_the_forecaster_model_name)."""
        pipeline = _make_pipeline()
        question = make_mock_numeric_question(id_of_question=301, cdf_size=201)
        percentiles = [
            Percentile(percentile=p, value=5.0 + 90.0 * i / (len(STANDARD_PERCENTILES) - 1))
            for i, p in enumerate(STANDARD_PERCENTILES)
        ]
        reasoned = cast(
            "list[ReasonedPrediction[PredictionTypes]]",
            [
                ReasonedPrediction(prediction_value=0.5, reasoning="Model: m1\n\nLow."),
                ReasonedPrediction(prediction_value=0.5, reasoning="Model: m2\n\nHigh."),
            ],
        )

        with (
            patch(
                "metaculus_bot.aggregation_pipeline.stacking.run_stacking_numeric",
                new=AsyncMock(return_value=(percentiles, "meta")),
            ),
            patch(
                "metaculus_bot.aggregation_pipeline.sanitize_percentiles",
                wraps=sanitize_percentiles,
            ) as spy,
            patch(
                "metaculus_bot.aggregation_pipeline.build_numeric_distribution",
                wraps=build_numeric_distribution,
            ) as build_spy,
        ):
            await pipeline.run_stacking(question, "research", reasoned)

        assert spy.call_args.kwargs["model_name"] == "test-model"
        # A lost model_name kwarg attributes CDF_MAXSTEP_CLIP to model=unknown silently.
        assert build_spy.call_args.kwargs["model_name"] == "test-model"


class TestThinPublishFloorInBaseCombine:
    """The single-survivor binary floor lives in ``_base_combine``'s ``len == 1`` branch.

    That branch serves two very different objects: the lone RAW member a
    ``single_forecaster`` skip hands through, and the single PRE-STACKED output of a
    fired stacker. Only the first has no aggregation behind it, so the trigger is the
    skip reason ``route_after_forecasts`` writes for exactly that event, never the
    prediction count alone. Bounds are THIN_PUBLISH_BINARY_FLOOR / _CEIL; the per-model
    record is untouched because the clamp is applied to the returned aggregate only.
    """

    @staticmethod
    def _single_survivor(pipeline: AggregationPipeline, question: BinaryQuestion) -> None:
        """Stamp the state the single-forecaster short-circuit leaves behind."""
        pipeline.register_expected_base_combine(question)
        assert question.id_of_question is not None
        pipeline.skip_reasons[question.id_of_question] = "single_forecaster"

    @pytest.mark.asyncio
    async def test_lone_low_call_is_floored_and_the_marker_names_raw_and_clamped(self, caplog):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=901)
        self._single_survivor(pipeline, question)

        with caplog.at_level("WARNING", logger="metaculus_bot.aggregation_pipeline"):
            result = pipeline.base_combine(predictions=[0.03], question=question)

        assert result == THIN_PUBLISH_BINARY_FLOOR
        markers = [r for r in caplog.records if r.getMessage().startswith("THIN_PUBLISH_FLOOR:")]
        assert len(markers) == 1
        assert markers[0].levelname == "WARNING"
        assert markers[0].getMessage() == "THIN_PUBLISH_FLOOR: question=901 raw=0.0300 clamped=0.0500 survivors=1"

    @pytest.mark.asyncio
    async def test_lone_high_call_is_ceilinged(self, caplog):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=902)
        self._single_survivor(pipeline, question)

        with caplog.at_level("WARNING", logger="metaculus_bot.aggregation_pipeline"):
            result = pipeline.base_combine(predictions=[0.97], question=question)

        assert result == THIN_PUBLISH_BINARY_CEIL
        assert [r.getMessage() for r in caplog.records if r.getMessage().startswith("THIN_PUBLISH_FLOOR:")] == [
            "THIN_PUBLISH_FLOOR: question=902 raw=0.9700 clamped=0.9500 survivors=1"
        ]

    @pytest.mark.asyncio
    async def test_lone_call_inside_the_band_is_untouched_and_silent(self, caplog):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=903)
        self._single_survivor(pipeline, question)

        with caplog.at_level("WARNING", logger="metaculus_bot.aggregation_pipeline"):
            result = pipeline.base_combine(predictions=[0.30], question=question)

        assert result == 0.30
        # Silence means nothing moved; the single-survivor EVENT is already observable
        # via FORECASTERS_SURVIVED and the skip reason.
        assert not [r for r in caplog.records if "THIN_PUBLISH_FLOOR" in r.getMessage()]

    @pytest.mark.asyncio
    async def test_the_skip_reason_is_left_for_the_comment_builder(self):
        # _create_unified_explanation pops it to publish STACKER_SKIP_REASON; the floor
        # only reads it. Consuming it here would erase the marker from the comment.
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=904)
        self._single_survivor(pipeline, question)

        pipeline.base_combine(predictions=[0.03], question=question)

        assert pipeline.skip_reasons[904] == "single_forecaster"
        assert pipeline.counters.stacking_expected_combine_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("members", "expected_median"),
        [
            ([0.03, 0.10, 0.15], 0.10),
            # The sharper case: a multi-member median BELOW the floor publishes as is.
            # The floor must never touch a multi-member publish — the receipt priced a
            # global [0.05, 0.95] over 408 binaries at -52.02 spot peer, 50 losses to 1 win.
            ([0.02, 0.03, 0.04], 0.03),
        ],
    )
    async def test_multi_member_publishes_are_never_floored(self, members, expected_median, caplog):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=905)
        pipeline.register_expected_base_combine(question)
        pipeline.skip_reasons[905] = "spread_below_threshold"

        with caplog.at_level("WARNING", logger="metaculus_bot.aggregation_pipeline"):
            result = pipeline.base_combine(predictions=list(members), question=question)

        assert result == pytest.approx(expected_median)
        assert not [r for r in caplog.records if "THIN_PUBLISH_FLOOR" in r.getMessage()]

    @pytest.mark.asyncio
    async def test_two_single_survivor_reports_combine_without_the_floor(self, caplog):
        # research_reports_per_question > 1 (no entrypoint configures it, but the framework
        # supports it): two reports each surviving on one forecaster both stamp
        # "single_forecaster", yet the framework hands their two values to one base combine.
        # Two members ARE an aggregation, so the reason alone must not floor them — the
        # trigger is the reason AND a lone value together.
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=910)
        pipeline.register_expected_base_combine(question)
        pipeline.skip_reasons[910] = "single_forecaster"

        with caplog.at_level("WARNING", logger="metaculus_bot.aggregation_pipeline"):
            result = pipeline.base_combine(predictions=[0.03, 0.04], question=question)

        assert result == pytest.approx(0.035)
        assert not [r for r in caplog.records if "THIN_PUBLISH_FLOOR" in r.getMessage()]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strategy", [AggregationStrategy.STACKING, AggregationStrategy.CONDITIONAL_STACKING])
    async def test_a_single_pre_stacked_output_is_not_floored(self, strategy, caplog):
        # The stacked path registers an expected base combine but writes NO skip reason,
        # so its lone output through this same branch passes untouched even at 0.03: the
        # stacker already aggregated the ensemble.
        pipeline = _make_pipeline(strategy=strategy)
        question = _make_binary_question(qid=906)
        pipeline.register_expected_base_combine(question)

        with caplog.at_level("WARNING", logger="metaculus_bot.aggregation_pipeline"):
            result = pipeline.base_combine(predictions=[0.03], question=question)

        assert result == 0.03
        assert not [r for r in caplog.records if "THIN_PUBLISH_FLOOR" in r.getMessage()]

    @pytest.mark.asyncio
    async def test_a_fired_stack_leaves_the_skip_reason_untouched(self):
        """The floor only READS ``skip_reasons``; no aggregation path writes or clears it.

        ``skip_reasons`` belongs to the routing step and to the comment builder that pops
        it, and the stacked path deliberately stays out of it. Mutating it here to defend
        against a reason orphaned by a mid-run crash would cost the reachable case
        instead: see ``test_a_discarded_stack_attempt_leaves_a_sibling_reports_reason``.
        """
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=907)
        pipeline.skip_reasons[907] = "single_forecaster"
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.02, reasoning="Model: m1\n\nLow"),
            ReasonedPrediction(prediction_value=0.04, reasoning="Model: m2\n\nLow"),
        ]

        with patch.object(pipeline, "run_stacking", new=AsyncMock(return_value=0.03)):
            stacked = await pipeline.stack_predictions(
                predictions=[0.02, 0.04], question=question, research="research", reasoned_predictions=reasoned
            )

        assert stacked == 0.03
        assert pipeline.skip_reasons[907] == "single_forecaster"
        assert pipeline.outcomes[907] == "primary"

    @pytest.mark.asyncio
    async def test_a_discarded_stack_attempt_leaves_a_sibling_reports_reason(self, caplog):
        """A failed stack must not consume the reason another report legitimately wrote.

        ``research_reports_per_question > 1`` (no entrypoint configures it, but the
        framework supports it and this bot sets ``required_successful_predictions=0.0``,
        so a partial report survival still publishes) has every report share ONE pipeline
        instance. Report A survives on a single forecaster and writes
        ``single_forecaster``; report B tries to stack, raises, and its whole task is
        dropped from ``valid_prediction_set``. The framework then flattens report A's lone
        value into one base combine, where that reason is the only thing left saying the
        published value has no aggregation behind it — so it has to still be there, and
        still be there afterwards for the STACKER_SKIP_REASON comment marker.
        """
        pipeline = _make_pipeline(stacking_fallback_on_failure=False)
        question = _make_binary_question(qid=911)
        pipeline.skip_reasons[911] = "single_forecaster"  # report A's routing
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.30, reasoning="Model: m1\n\nmid"),
            ReasonedPrediction(prediction_value=0.60, reasoning="Model: m2\n\nmid"),
        ]

        with (
            patch.object(pipeline, "run_stacking", new=AsyncMock(side_effect=RuntimeError("stacker down"))),
            pytest.raises(RuntimeError),
        ):
            await pipeline.stack_predictions(  # report B, discarded
                predictions=[0.30, 0.60], question=question, research="research", reasoned_predictions=reasoned
            )
        assert pipeline.skip_reasons[911] == "single_forecaster"

        pipeline.register_expected_base_combine(question)
        with caplog.at_level("WARNING", logger="metaculus_bot.aggregation_pipeline"):
            published = pipeline.base_combine(predictions=[0.03], question=question)

        assert published == THIN_PUBLISH_BINARY_FLOOR
        assert [r.getMessage() for r in caplog.records if r.getMessage().startswith("THIN_PUBLISH_FLOOR:")] == [
            "THIN_PUBLISH_FLOOR: question=911 raw=0.0300 clamped=0.0500 survivors=1"
        ]
        assert pipeline.skip_reasons[911] == "single_forecaster"

    @pytest.mark.asyncio
    async def test_a_lone_mc_survivor_is_returned_as_is(self):
        pipeline = _make_pipeline()
        question = _make_mc_question(qid=908)
        pipeline.register_expected_base_combine(question)
        pipeline.skip_reasons[908] = "single_forecaster"
        lone = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.03),
                PredictedOption(option_name="B", probability=0.47),
                PredictedOption(option_name="C", probability=0.50),
            ]
        )

        result = pipeline.base_combine(predictions=[lone], question=question)

        assert result is lone

    @pytest.mark.asyncio
    async def test_a_lone_numeric_survivor_keeps_its_snap_path(self):
        # The floor branch must sit beside, not in front of, the discrete-integer snap the
        # single-survivor numeric path relies on (forecaster.py hands the raw distribution
        # through and this branch is where a DISCRETE majority gets honoured).
        pipeline = _make_pipeline()
        question = make_mock_numeric_question(id_of_question=909)
        pipeline.register_expected_base_combine(question)
        pipeline.skip_reasons[909] = "single_forecaster"
        pipeline.discrete_integer_votes[909] = [True]
        lone = MagicMock(spec=NumericDistribution)
        snapped = MagicMock(spec=NumericDistribution)

        with patch("metaculus_bot.aggregation_pipeline.maybe_snap_to_integers", return_value=snapped) as snap:
            result = pipeline.base_combine(predictions=[lone], question=question)

        assert result is snapped
        snap.assert_called_once_with(lone, question, [True])
        assert 909 not in pipeline.discrete_integer_votes


# ---------------------------------------------------------------------------
# Per-bin members: the linear opinion pool, and the NUMERIC_AGGREGATE ``method`` record
# ---------------------------------------------------------------------------


def _sharp_members(question: MetaculusQuestion, bins: tuple[int, ...]) -> list[NumericDistribution]:
    """One member certain of each bin in ``bins``, on the question's own grid."""
    view = numeric_view(question)
    return [certain_of_bin(view, k) for k in bins]


def _percentile_members(question: NumericQuestion) -> list[NumericDistribution]:
    """Three percentile-elicited members spread over the middle of the question's range."""
    span = question.upper_bound - question.lower_bound
    members = []
    for shift in (0.0, 0.02, -0.015):
        declared = [
            Percentile(percentile=p, value=question.lower_bound + span * (0.4 + shift + 0.4 * p))
            for p in STANDARD_PERCENTILES
        ]
        sanitized, zero_point = sanitize_percentiles(declared, question, model_name="test-model")
        members.append(build_numeric_distribution(sanitized, question, zero_point, model_name="test-model"))
    return members


_MANTIC_DATE_651 = on_mantic(load_preseason_date_question())


class TestNumericCombineStrategy:
    """Per-bin members are pooled by the pointwise MEAN of their CDFs; everything else keeps its strategy.

    The pointwise MEDIAN of three sharp CDFs that disagree is the middle member's CDF outright:
    0.99 on its bin and the platform floor on the bins the other two believed, which a log score in
    the resolved bin punishes hardest (about -230 Series 1 points on post 651's 12-bin grid). The
    mean of the CDFs is the CDF of the mixture PMF, so every believed bin keeps a third of the mass
    a member gave it. The elicitation is decided per question (``elicit_per_bin``), so an ensemble
    never mixes the two kinds of member and the rule is a property of the question alone.
    """

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            pytest.param(_MANTIC_DATE_651, AggregationStrategy.MEAN, id="mantic-12-bin-date"),
            pytest.param(make_count_question(11, open_upper=False), AggregationStrategy.MEAN, id="mantic-11-bin-count"),
            pytest.param(
                make_count_question(450, open_upper=False), AggregationStrategy.MEDIAN, id="mantic-450-bin-count"
            ),
            pytest.param(
                make_count_question(11, page_url=metaculus_url(700), open_upper=False),
                AggregationStrategy.MEDIAN,
                id="metaculus-11-bin-count",
            ),
            pytest.param(make_real_numeric_question(), AggregationStrategy.MEDIAN, id="metaculus-201-point"),
            pytest.param(load_preseason_date_question(), AggregationStrategy.MEDIAN, id="12-bin-date-off-mantic"),
            pytest.param(_make_binary_question(), AggregationStrategy.MEDIAN, id="binary"),
            pytest.param(_make_mc_question(), AggregationStrategy.MEDIAN, id="multiple-choice"),
        ],
    )
    def test_the_median_becomes_the_mean_only_for_a_per_bin_question(self, question, expected) -> None:
        pipeline = _make_pipeline()
        assert pipeline._numeric_combine_strategy(question, AggregationStrategy.MEDIAN) is expected

    def test_a_mean_stays_a_mean(self) -> None:
        pipeline = _make_pipeline(strategy=AggregationStrategy.STACKING)
        for question in (_MANTIC_DATE_651, make_real_numeric_question(), _make_binary_question()):
            assert pipeline._numeric_combine_strategy(question, AggregationStrategy.MEAN) is AggregationStrategy.MEAN

    def test_base_combine_pools_three_sharp_per_bin_members(self, caplog) -> None:
        """Members certain of bins 3, 5 and 7 publish about a third on each; the median would have
        published 0.99 on bin 5 and the floor on 3 and 7."""
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        pipeline = _make_pipeline()
        question = _MANTIC_DATE_651
        pipeline.register_expected_base_combine(question)
        members = _sharp_members(question, (3, 5, 7))

        combined = pipeline.base_combine(cast("list[PredictionTypes]", members), question)

        assert isinstance(combined, NumericDistribution)
        pooled = pmf_of(combined)
        assert pooled[[3, 5, 7]] == pytest.approx([0.3308, 0.3308, 0.3308], abs=1e-3)
        assert pooled[[0, 1, 2, 4, 6, 8, 9, 10, 11]].max() < 0.001
        heights = cdf_heights(combined)
        assert_server_accepts_cdf(heights, cdf_size=13, open_lower=False, open_upper=False)
        assert combined.is_date is True
        assert pipeline.numeric_combine_methods == {question.id_of_question: "mean"}
        assert "STACKING base combine: numeric mean aggregation" in caplog.text

    def test_base_combine_keeps_the_median_for_a_mantic_percentile_grid(self) -> None:
        """450 bins is far above the per-bin threshold, so a Mantic question this wide stays on
        percentiles and its members are medianed byte for byte, exactly as before Wave C."""
        pipeline = _make_pipeline()
        question = make_count_question(450, open_upper=False)
        members = _percentile_members(question)
        expected = cdf_heights(aggregate_numeric(members, question, "median"))

        combined = pipeline.base_combine(cast("list[PredictionTypes]", members), question)

        assert isinstance(combined, NumericDistribution)
        assert np.array_equal(cdf_heights(combined), expected)
        assert pipeline.numeric_combine_methods[700] == "median"

    def test_base_combine_keeps_the_median_byte_for_byte_for_percentile_members(self) -> None:
        pipeline = _make_pipeline()
        question = make_real_numeric_question()
        members = _percentile_members(question)
        expected = cdf_heights(aggregate_numeric(members, question, "median"))

        combined = pipeline.base_combine(cast("list[PredictionTypes]", members), question)

        assert isinstance(combined, NumericDistribution)
        assert np.array_equal(cdf_heights(combined), expected)
        assert pipeline.numeric_combine_methods == {question.id_of_question: "median"}

    def test_the_median_fallback_pools_per_bin_members_too(self) -> None:
        pipeline = _make_pipeline()
        question = _MANTIC_DATE_651
        members = _sharp_members(question, (3, 5, 7))

        combined = pipeline._median_fallback(cast("list[PredictionTypes]", members), question)

        assert isinstance(combined, NumericDistribution)
        assert pmf_of(combined)[[3, 5, 7]] == pytest.approx([0.3308] * 3, abs=1e-3)
        assert pipeline.numeric_combine_methods == {question.id_of_question: "mean"}

    def test_simple_combine_pools_per_bin_members_under_the_median_strategy(self) -> None:
        pipeline = _make_pipeline(strategy=AggregationStrategy.MEDIAN)
        question = _MANTIC_DATE_651
        members = _sharp_members(question, (3, 5, 7))

        combined = pipeline.simple_combine(cast("list[PredictionTypes]", members), question)

        assert isinstance(combined, NumericDistribution)
        assert pmf_of(combined)[[3, 5, 7]] == pytest.approx([0.3308] * 3, abs=1e-3)
        assert pipeline.numeric_combine_methods == {question.id_of_question: "mean"}


class TestNumericCombineMethodRecording:
    """Every numeric return path writes ``numeric_combine_methods``; the forecaster seam pops it.

    ``mean`` / ``median`` are the combiner's own tokens; ``stacked`` is a stacker-adopted
    distribution; ``single`` is the lone raw member the min-forecasters=1 short-circuit hands
    through. A pre-stacked output that re-enters ``base_combine`` keeps the record the stacker path
    wrote. Binary and MC questions never write it.
    """

    @staticmethod
    def _stacked_reasoned() -> list[ReasonedPrediction[PredictionTypes]]:
        return [
            ReasonedPrediction(prediction_value=0.5, reasoning="Model: m1\n\nEarly."),
            ReasonedPrediction(prediction_value=0.5, reasoning="Model: m2\n\nLate."),
        ]

    def test_base_combine_records_the_median_under_conditional_stacking(self) -> None:
        pipeline = _make_pipeline()
        question = make_real_numeric_question()
        pipeline.base_combine(cast("list[PredictionTypes]", _percentile_members(question)), question)
        assert pipeline.numeric_combine_methods == {question.id_of_question: "median"}

    def test_base_combine_records_the_mean_under_plain_stacking(self) -> None:
        pipeline = _make_pipeline(strategy=AggregationStrategy.STACKING)
        question = make_real_numeric_question()
        pipeline.base_combine(cast("list[PredictionTypes]", _percentile_members(question)), question)
        assert pipeline.numeric_combine_methods == {question.id_of_question: "mean"}

    @pytest.mark.parametrize("strategy", [AggregationStrategy.MEAN, AggregationStrategy.MEDIAN])
    def test_simple_combine_records_its_effective_strategy(self, strategy: AggregationStrategy) -> None:
        pipeline = _make_pipeline(strategy=strategy)
        question = make_real_numeric_question()
        pipeline.simple_combine(cast("list[PredictionTypes]", _percentile_members(question)), question)
        assert pipeline.numeric_combine_methods == {question.id_of_question: strategy.value}

    def test_a_lone_raw_survivor_records_single(self) -> None:
        pipeline = _make_pipeline()
        question = make_real_numeric_question()
        qid = question.id_of_question
        assert qid is not None
        pipeline.register_expected_base_combine(question)
        pipeline.skip_reasons[qid] = "single_forecaster"
        (member,) = _percentile_members(question)[:1]

        combined = pipeline.base_combine([member], question)

        assert combined is member
        assert pipeline.numeric_combine_methods == {qid: NUMERIC_COMBINE_METHOD_SINGLE}
        assert NUMERIC_COMBINE_METHOD_SINGLE == "single"

    def test_a_lone_per_bin_survivor_records_single_as_well(self) -> None:
        pipeline = _make_pipeline()
        question = _MANTIC_DATE_651
        (member,) = _sharp_members(question, (8,))
        combined = pipeline.base_combine([member], question)
        assert combined is member
        assert pipeline.numeric_combine_methods == {question.id_of_question: "single"}

    @pytest.mark.asyncio
    async def test_a_stacked_distribution_records_stacked_and_survives_the_re_entry(self) -> None:
        """``stacking_route`` calls ``stack_predictions`` itself and then registers the expected
        base combine; the framework re-enters ``_aggregate_predictions`` with the single pre-stacked
        output, which must not overwrite the stacker's record with ``single``."""
        pipeline = _make_pipeline()
        question = make_real_numeric_question()
        qid = question.id_of_question
        assert qid is not None
        members = _percentile_members(question)
        stacked = members[1]

        with patch.object(pipeline, "run_stacking", new=AsyncMock(return_value=stacked)):
            adopted = await pipeline.stack_predictions(
                cast("list[PredictionTypes]", members),
                question,
                research="research",
                reasoned_predictions=self._stacked_reasoned(),
            )
        assert adopted is stacked
        assert pipeline.outcomes[qid] == "primary"
        assert pipeline.numeric_combine_methods == {qid: NUMERIC_COMBINE_METHOD_STACKED}
        assert NUMERIC_COMBINE_METHOD_STACKED == "stacked"

        pipeline.register_expected_base_combine(question)
        re_entered = pipeline.base_combine([adopted], question)

        assert re_entered is stacked
        assert pipeline.numeric_combine_methods == {qid: "stacked"}

    @pytest.mark.asyncio
    async def test_the_fallback_stacker_records_stacked_too(self) -> None:
        pipeline = _make_pipeline()
        question = make_real_numeric_question()
        members = _percentile_members(question)
        attempts = 0

        async def fail_primary(*args: Any, **kwargs: Any) -> PredictionTypes:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("primary failed")
            return members[2]

        with patch.object(pipeline, "run_stacking", side_effect=fail_primary):
            adopted = await pipeline.stack_predictions(
                cast("list[PredictionTypes]", members),
                question,
                research="research",
                reasoned_predictions=self._stacked_reasoned(),
            )

        assert adopted is members[2]
        assert pipeline.outcomes == {question.id_of_question: "fallback_llm"}
        assert pipeline.numeric_combine_methods == {question.id_of_question: "stacked"}

    @pytest.mark.asyncio
    async def test_a_failed_stack_records_the_fallback_median(self) -> None:
        pipeline = _make_pipeline()
        question = make_real_numeric_question()
        members = _percentile_members(question)

        with patch.object(pipeline, "run_stacking", new=AsyncMock(side_effect=RuntimeError("down"))):
            combined = await pipeline.stack_predictions(
                cast("list[PredictionTypes]", members),
                question,
                research="research",
                reasoned_predictions=self._stacked_reasoned(),
            )

        assert isinstance(combined, NumericDistribution)
        assert pipeline.outcomes == {question.id_of_question: "fallback_median"}
        assert pipeline.numeric_combine_methods == {question.id_of_question: "median"}

    def test_binary_and_mc_questions_never_write_the_record(self) -> None:
        pipeline = _make_pipeline()
        binary = _make_binary_question(qid=901)
        pipeline.base_combine([0.3, 0.5, 0.7], binary)
        mc = _make_mc_question(qid=902)
        pols = [
            PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="A", probability=a),
                    PredictedOption(option_name="B", probability=b),
                    PredictedOption(option_name="C", probability=1.0 - a - b),
                ]
            )
            for a, b in ((0.5, 0.3), (0.6, 0.2))
        ]
        pipeline.base_combine(cast("list[PredictionTypes]", pols), mc)
        pipeline.simple_combine([0.3, 0.5], binary)
        assert pipeline.numeric_combine_methods == {}


class TestTheAggregateMarkerMethod:
    """``TemplateForecaster._aggregate_predictions`` pops the record and ends NUMERIC_AGGREGATE with it."""

    @staticmethod
    def _aggregate_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.getMessage().startswith("NUMERIC_AGGREGATE:")]

    @pytest.mark.asyncio
    async def test_a_percentile_question_logs_method_median(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        bot = make_e2e_bot(AggregationStrategy.CONDITIONAL_STACKING)
        question = make_real_numeric_question()

        aggregated = await bot._aggregate_predictions(
            cast("list[PredictionTypes]", _percentile_members(question)), question
        )

        assert isinstance(aggregated, NumericDistribution)
        (line,) = self._aggregate_lines(caplog)
        assert line.startswith(f"NUMERIC_AGGREGATE: question={question.id_of_question} qtype=numeric cdf_size=201 ")
        assert line.endswith(" tail_floor=0.020000 method=median"), "the Metaculus tail floor lifts the open upper tail"
        assert bot._pipeline.numeric_combine_methods == {}, (
            "popped at the seam, so a record never outlives its question"
        )

    @pytest.mark.asyncio
    async def test_a_per_bin_question_logs_method_mean_and_publishes_the_pool(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        bot = make_e2e_bot(AggregationStrategy.CONDITIONAL_STACKING)
        question = _MANTIC_DATE_651

        aggregated = await bot._aggregate_predictions(
            cast("list[PredictionTypes]", _sharp_members(question, (3, 5, 7))), question
        )

        assert isinstance(aggregated, NumericDistribution)
        assert pmf_of(aggregated)[[3, 5, 7]] == pytest.approx([0.3308] * 3, abs=1e-3)
        (line,) = self._aggregate_lines(caplog)
        assert line == (
            f"NUMERIC_AGGREGATE: question={question.id_of_question} qtype=date cdf_size=13 oor_low=0.000000 "
            "oor_high=0.000000 oor_low_raw=0.000000 oor_high_raw=0.000000 tail_floor=0.000000 method=mean"
        )
        assert bot._pipeline.numeric_combine_methods == {}

    @pytest.mark.asyncio
    async def test_a_path_that_never_recorded_logs_unrecorded_instead_of_forfeiting(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``unrecorded`` is a bug signal, never an expected value: a combine path that forgot to
        write the dict still publishes, and the archive says so."""
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        bot = make_e2e_bot(AggregationStrategy.CONDITIONAL_STACKING)
        question = make_real_numeric_question()
        (member,) = _percentile_members(question)[:1]

        with patch.object(bot._pipeline, "base_combine", return_value=member):
            aggregated = await bot._aggregate_predictions([member], question)

        assert isinstance(aggregated, NumericDistribution), "the floor may rebuild the object; it still publishes"
        (line,) = self._aggregate_lines(caplog)
        assert line.endswith(f" method={NUMERIC_COMBINE_METHOD_UNRECORDED}")
        assert NUMERIC_COMBINE_METHOD_UNRECORDED == "unrecorded"
