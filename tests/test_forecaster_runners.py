"""Tests for metaculus_bot.forecaster_runners — extracted per-type forecast functions.

Exercises the public runners (run_binary_forecast, run_mc_forecast, run_numeric_forecast,
run_date_forecast) to verify they produce the same results as the original
TemplateForecaster methods they replaced.

The runners now delegate value extraction to the deterministic-first
``value_extraction`` ladder (block → repair → llm). Tests here mostly patch the
top-level ladder entrypoints (``extract_binary``, ``extract_mc``,
``extract_numeric``, ``extract_pmf``) so we exercise the caller's post-processing without
re-testing ladder internals. A handful of integration-style tests feed a
rationale with a real fenced JSON block so the ladder runs end-to-end and we
can assert an ``EXTRACTION_RUNG`` telemetry line was emitted.

``TestPerBinBranch`` covers the per-bin path a coarse Mantic grid takes (``elicit_per_bin``):
the numeric and date runners branch into ``_run_pmf_forecast`` and none of the percentile
machinery (discrete vote, sanitizer, PCHIP build, unit-mismatch guard) is called.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import litellm.exceptions as litellm_exc
import numpy as np
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
from forecasting_tools.ai_models.ai_utils.openai_utils import VisionMessageData
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile
from pydantic import ValidationError

from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN, MANTIC_SITE_URL, PMF_ABOVE_RANGE_KEY
from metaculus_bot.exceptions import UnitMismatchError
from metaculus_bot.forecaster_runners import (
    build_pmf_parse_notes,
    run_binary_forecast,
    run_date_forecast,
    run_mc_forecast,
    run_numeric_forecast,
)
from metaculus_bot.llm_retry import TRANSIENT_RETRY_MAX_ELAPSED_S
from metaculus_bot.numeric.config import PCHIP_CDF_POINTS
from metaculus_bot.numeric.date_axis import EpochDateQuestion, as_epoch_question
from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
from metaculus_bot.numeric.pmf_grid import PmfGrid, pmf_grid
from metaculus_bot.value_extraction import ExtractionOutcome, McForecast, PmfForecast
from tests.pipeline_test_helpers import (
    assert_server_accepts_cdf,
    cdf_heights,
    distribution_from_heights,
    make_count_question,
    make_real_date_question,
    make_real_numeric_question,
    server_min_step,
)


@pytest.fixture
def parser_llm():
    return GeneralLlm(model="test-parser")


@pytest.fixture
def forecaster_llm():
    return GeneralLlm(model="test-forecaster")


@pytest.fixture
def binary_question():
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = 1001
    q.page_url = "https://metaculus.com/questions/1001"
    q.question_text = "Will X happen?"
    return q


@pytest.fixture
def mc_question():
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.id_of_question = 2001
    q.page_url = "https://metaculus.com/questions/2001"
    q.question_text = "Which outcome?"
    q.options = ["Option A", "Option B", "Option C"]
    return q


@pytest.fixture
def numeric_question():
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = 3001
    q.page_url = "https://metaculus.com/questions/3001"
    q.lower_bound = 0
    q.upper_bound = 1000
    q.open_lower_bound = False
    q.open_upper_bound = True
    q.unit_of_measure = "widgets"
    q.cdf_size = PCHIP_CDF_POINTS  # the standard continuous grid: the per-bin gate reads it
    return q


def _binary_outcome(value: float) -> ExtractionOutcome[float]:
    return ExtractionOutcome(value=value, rung="block", block_present=True)


def _mc_outcome(pol: PredictedOptionList) -> ExtractionOutcome[McForecast]:
    return ExtractionOutcome(
        value=McForecast(pol, [o.probability for o in pol.predicted_options]), rung="block", block_present=True
    )


def _make_option_list(options: list[tuple[str, float]]) -> PredictedOptionList:
    return PredictedOptionList(predicted_options=[PredictedOption(option_name=n, probability=p) for n, p in options])


def _member_forecast_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("MEMBER_FORECAST:")]


def _built_numeric_prediction() -> NumericDistribution:
    question = make_real_numeric_question(upper_bound=1000)
    return distribution_from_heights(np.linspace(0, 0.99, question.cdf_size), question)


_STANDARD_PERCENTILES: list[Percentile] = [
    Percentile(percentile=p / 100, value=v)
    for p, v in zip(
        [2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5],
        [50, 100, 150, 200, 350, 450, 550, 700, 800, 900, 950],
        strict=True,
    )
]


class TestRunBinaryForecast:
    @pytest.mark.asyncio
    async def test_returns_reasoned_prediction_with_clamped_value(
        self, binary_question, forecaster_llm, parser_llm
    ) -> None:
        """Binary forecast clamps to [BINARY_PROB_MIN, BINARY_PROB_MAX] and returns ReasonedPrediction."""
        reasoning_text = "Analysis: likely yes.\n\nProbability: 75%"

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=_binary_outcome(0.75)),
            ),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert isinstance(result, ReasonedPrediction)
        assert result.prediction_value == 0.75
        assert result.reasoning == reasoning_text

    @pytest.mark.asyncio
    async def test_clamps_below_minimum(
        self, binary_question, forecaster_llm, parser_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Values below BINARY_PROB_MIN are clamped up, and the MEMBER_FORECAST line
        keeps the pre-clamp value beside the published one."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster_runners")
        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="Very unlikely")),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=_binary_outcome(0.001)),
            ),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert result.prediction_value == BINARY_PROB_MIN
        assert _member_forecast_lines(caplog) == [
            f"MEMBER_FORECAST: question=1001 model=test-forecaster role=member qtype=binary "
            f"raw=0.001 published={BINARY_PROB_MIN}"
        ]

    @pytest.mark.asyncio
    async def test_clamps_above_maximum(
        self, binary_question, forecaster_llm, parser_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Values above BINARY_PROB_MAX are clamped down, with the raw value on the marker."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster_runners")
        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="Nearly certain")),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=_binary_outcome(0.999)),
            ),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert result.prediction_value == BINARY_PROB_MAX
        assert _member_forecast_lines(caplog) == [
            f"MEMBER_FORECAST: question=1001 model=test-forecaster role=member qtype=binary "
            f"raw=0.999 published={BINARY_PROB_MAX}"
        ]

    @pytest.mark.asyncio
    async def test_extraction_rung_logged_via_real_ladder(
        self, binary_question, forecaster_llm, parser_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Integration-style: no ladder patch; a rationale with a valid binary block
        goes through ``extract_binary`` end-to-end, clamps to BINARY_PROB_MAX, and
        emits an ``EXTRACTION_RUNG`` INFO line on the ``metaculus_bot.value_extraction``
        logger.
        """
        caplog.set_level(logging.INFO, logger="metaculus_bot.value_extraction")
        reasoning_text = 'Nearly certain.\n\n```json\n{"question_type": "binary", "posterior_prob": 0.99}\n```'

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert result.prediction_value == BINARY_PROB_MAX  # 0.99 clamped down
        rung_msgs = [r.getMessage() for r in caplog.records if "EXTRACTION_RUNG" in r.getMessage()]
        assert len(rung_msgs) == 1
        msg = rung_msgs[0]
        assert "question=1001" in msg
        assert "model=test-forecaster" in msg
        assert "qtype=binary" in msg
        assert "rung=block" in msg
        assert "block_present=True" in msg


class TestForecasterBroadRetry:
    """The forecaster invoke is wrapped in the broad, 30s-gated retry (Round-2).

    Forecaster GeneralLlm instances are set allowed_tries=1 in llm_configs.py, so
    this wrapper is their SOLE retry layer: it recovers a fast blip but obeys the
    universal "no retry after 30s" deadline-safety rule that forecasting-tools'
    un-gated tenacity could not.
    """

    @pytest.mark.asyncio
    async def test_fast_blip_on_forecaster_invoke_retries_then_succeeds(
        self, binary_question, forecaster_llm, parser_llm
    ) -> None:
        """A fast litellm.Timeout on forecaster_llm.invoke is retried; the next call wins."""
        invoke = AsyncMock(
            side_effect=[
                litellm_exc.Timeout("blip", model="m", llm_provider="openrouter"),
                "Analysis: likely yes.\n\nProbability: 70%",
            ]
        )
        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch("metaculus_bot.llm_retry.asyncio.sleep", new=AsyncMock()),
            patch.object(forecaster_llm, "invoke", new=invoke),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=_binary_outcome(0.70)),
            ),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert result.prediction_value == 0.70
        assert invoke.await_count == 2

    @pytest.mark.asyncio
    async def test_slow_forecaster_failure_not_retried(self, binary_question, forecaster_llm, parser_llm) -> None:
        """A forecaster invoke that fails past the 30s gate is NOT retried — it propagates."""
        invoke = AsyncMock(side_effect=litellm_exc.Timeout("stall", model="m", llm_provider="openrouter"))
        clock = iter([0.0] + [TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0] * 20)

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch("metaculus_bot.llm_retry.time.monotonic", lambda: next(clock)),
            patch.object(forecaster_llm, "invoke", new=invoke),
            pytest.raises(litellm_exc.Timeout),
        ):
            await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert invoke.await_count == 1


def _last_invoke_arg(invoke: AsyncMock) -> object:
    """Return the single positional arg of the most recent await, narrowing the
    Optional ``await_args`` for the type checker."""
    await_args = invoke.await_args
    assert await_args is not None, "invoke was never awaited"
    (arg,) = await_args.args
    return arg


class TestForecasterChartVision:
    """The optional TS-anchor chart image (chart_b64) turns the forecaster invoke
    input into a ``VisionMessageData``; absent it stays a bare prompt string. All
    three runners share the ``_forecaster_input`` helper, so binary covers the wiring
    and one numeric case confirms it threads through the numeric path too.
    """

    @pytest.mark.asyncio
    async def test_no_chart_invokes_with_bare_prompt(self, binary_question, forecaster_llm, parser_llm) -> None:
        invoke = AsyncMock(return_value="Analysis.\n\nProbability: 40%")
        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="PROMPT"),
            patch.object(forecaster_llm, "invoke", new=invoke),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=_binary_outcome(0.40)),
            ),
        ):
            await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert _last_invoke_arg(invoke) == "PROMPT"  # plain string, not VisionMessageData

    @pytest.mark.asyncio
    async def test_chart_wraps_invoke_in_vision_message(self, binary_question, forecaster_llm, parser_llm) -> None:
        invoke = AsyncMock(return_value="Analysis.\n\nProbability: 40%")
        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="PROMPT"),
            patch.object(forecaster_llm, "invoke", new=invoke),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=_binary_outcome(0.40)),
            ),
        ):
            await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm, chart_b64="ZmFrZQ==")

        called_arg = _last_invoke_arg(invoke)
        assert isinstance(called_arg, VisionMessageData)
        assert called_arg.prompt == "PROMPT"
        assert called_arg.b64_image == "ZmFrZQ=="
        assert called_arg.image_resolution == "low"

    @pytest.mark.asyncio
    async def test_numeric_runner_wraps_chart(self, numeric_question, forecaster_llm, parser_llm) -> None:
        invoke = AsyncMock(return_value="reasoning")
        mock_parse_structured = AsyncMock(return_value=OutcomeTypeResult(is_discrete_integer=False))
        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="PROMPT"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper", "lower")),
            patch.object(forecaster_llm, "invoke", new=invoke),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=mock_parse_structured),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
                ),
            ),
            patch("metaculus_bot.forecaster_runners.sanitize_percentiles", return_value=(_STANDARD_PERCENTILES, None)),
            patch(
                "metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=_built_numeric_prediction()
            ) as mock_build,
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
            patch("metaculus_bot.forecaster_runners.log_open_bound_piling_diagnostics"),
        ):
            await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm, chart_b64="ZmFrZQ==")

        called_arg = _last_invoke_arg(invoke)
        assert isinstance(called_arg, VisionMessageData)
        assert called_arg.b64_image == "ZmFrZQ=="
        # A lost model_name kwarg attributes CDF_MAXSTEP_CLIP to model=unknown silently.
        assert mock_build.call_args.kwargs["model_name"] == forecaster_llm.model


class TestRunMcForecast:
    @pytest.mark.asyncio
    async def test_returns_reasoned_prediction_with_option_list(self, mc_question, forecaster_llm, parser_llm) -> None:
        """MC forecast returns a ReasonedPrediction with PredictedOptionList."""
        reasoning_text = "Option A most likely."
        option_list = _make_option_list([("Option A", 0.6), ("Option B", 0.3), ("Option C", 0.1)])

        with (
            patch("metaculus_bot.forecaster_runners.multiple_choice_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
            patch(
                "metaculus_bot.forecaster_runners.extract_mc",
                new=AsyncMock(return_value=_mc_outcome(option_list)),
            ),
            patch("metaculus_bot.forecaster_runners.clamp_and_renormalize_mc", return_value=option_list),
        ):
            result = await run_mc_forecast(mc_question, "research", forecaster_llm, parser_llm)

        assert isinstance(result, ReasonedPrediction)
        assert result.prediction_value == option_list
        assert result.reasoning == reasoning_text

    @pytest.mark.asyncio
    async def test_real_ladder_maps_block_options_onto_question(self, mc_question, forecaster_llm, parser_llm) -> None:
        """Integration-style: rationale with a valid MC block goes through the real
        ``extract_mc`` ladder (no patch) and the returned option list maps onto
        ``question.options`` in canonical order.
        """
        reasoning_text = (
            "Option A is most likely.\n\n"
            "```json\n"
            '{"question_type": "multiple_choice", "option_probs": '
            '{"Option A": 0.6, "Option B": 0.3, "Option C": 0.1}}\n'
            "```"
        )

        with (
            patch("metaculus_bot.forecaster_runners.multiple_choice_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
        ):
            result = await run_mc_forecast(mc_question, "research", forecaster_llm, parser_llm)

        assert isinstance(result, ReasonedPrediction)
        pol = result.prediction_value
        assert isinstance(pol, PredictedOptionList)
        names = [o.option_name for o in pol.predicted_options]
        assert set(names) == {"Option A", "Option B", "Option C"}
        total = sum(o.probability for o in pol.predicted_options)
        assert abs(total - 1.0) < 1e-6


class TestRunNumericForecast:
    @pytest.mark.asyncio
    async def test_percentile_branch_returns_prediction_and_discrete_vote(
        self, numeric_question, forecaster_llm, parser_llm
    ) -> None:
        """Numeric forecast returns (prediction, discrete_vote) tuple via the percentile branch."""
        reasoning_text = "OUTCOME_TYPE: DISCRETE\n\nPercentile 2.5: 50"
        mock_prediction = _built_numeric_prediction()

        mock_parse_structured = AsyncMock(return_value=OutcomeTypeResult(is_discrete_integer=True))

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=mock_parse_structured),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
                ),
            ),
            patch(
                "metaculus_bot.forecaster_runners.sanitize_percentiles",
                return_value=(_STANDARD_PERCENTILES, None),
            ),
            patch(
                "metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=mock_prediction
            ) as mock_build,
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
        ):
            prediction, discrete_vote = await run_numeric_forecast(
                numeric_question, "research", forecaster_llm, parser_llm
            )

        assert prediction.prediction_value == mock_prediction
        assert discrete_vote is True
        # A lost model_name kwarg attributes CDF_MAXSTEP_CLIP to model=unknown silently.
        assert mock_build.call_args.kwargs["model_name"] == forecaster_llm.model

    @pytest.mark.asyncio
    async def test_sanitize_percentiles_receives_the_forecaster_model_name(
        self, numeric_question, forecaster_llm, parser_llm
    ) -> None:
        """The NUMERIC_DEGENERATE_DECLARATION marker attributes a collapsed declaration
        to the forecaster that made it, and that attribution exists only because this
        caller passes ``model_name`` — a ``model=unknown`` in the archive means a caller
        forgot to (see the ``sanitize_percentiles`` docstring)."""
        mock_parse_structured = AsyncMock(return_value=OutcomeTypeResult(is_discrete_integer=False))

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="reasoning")),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=mock_parse_structured),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
                ),
            ),
            patch(
                "metaculus_bot.forecaster_runners.sanitize_percentiles",
                return_value=(_STANDARD_PERCENTILES, None),
            ) as mock_sanitize,
            patch(
                "metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=_built_numeric_prediction()
            ),
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
        ):
            await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm)

        assert mock_sanitize.call_args.kwargs["model_name"] == forecaster_llm.model

    @pytest.mark.asyncio
    async def test_unit_mismatch_raises(self, numeric_question, forecaster_llm, parser_llm) -> None:
        """When detect_unit_mismatch returns True, raises UnitMismatchError."""
        mock_parse_structured = AsyncMock(return_value=OutcomeTypeResult(is_discrete_integer=False))

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="reasoning")),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=mock_parse_structured),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
                ),
            ),
            patch(
                "metaculus_bot.forecaster_runners.sanitize_percentiles",
                return_value=(_STANDARD_PERCENTILES, None),
            ),
            patch(
                "metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=_built_numeric_prediction()
            ),
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(True, "off by 1000x")),
            pytest.raises(UnitMismatchError, match="off by 1000x"),
        ):
            await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm)

    @pytest.mark.asyncio
    async def test_c3_block_read_skips_parser_call_for_outcome_type(
        self, numeric_question, forecaster_llm, parser_llm
    ) -> None:
        """F7: when the rationale's structured JSON block declares outcome_type,
        discrete_vote is read from the block and parse_structured is NOT called for
        OutcomeTypeResult. Percentiles come from the ladder (patched); parse_structured
        is patched to a sentinel that fails the test if invoked.
        """
        reasoning_text = (
            "Some rationale text.\n"
            "```json\n"
            '{"question_type": "numeric", "outcome_type": "discrete_integer",'
            ' "declared_percentiles": {"0.1": 10.0, "0.5": 50.0, "0.9": 90.0}}\n'
            "```\n"
        )

        mock_parse_structured = AsyncMock(
            side_effect=AssertionError("parse_structured should not be called for outcome_type when block declares it")
        )

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=mock_parse_structured),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
                ),
            ),
            patch(
                "metaculus_bot.forecaster_runners.sanitize_percentiles",
                return_value=(_STANDARD_PERCENTILES, None),
            ),
            patch(
                "metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=_built_numeric_prediction()
            ),
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
        ):
            _, discrete_vote = await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm)

        assert discrete_vote is True
        assert mock_parse_structured.await_count == 0  # confirms parse_structured was never called

    @pytest.mark.asyncio
    async def test_a_misspelled_outcome_type_falls_back_to_the_parser_call(
        self, numeric_question, forecaster_llm, parser_llm
    ) -> None:
        """A stray spelling costs ONE parser call and keeps the percentiles on rung 1.

        The schema reads an unrecognised outcome_type as absent (2026-09-02), so the block
        still validates and ``_resolve_discrete_vote`` takes exactly the OutcomeTypeResult
        fallback it already has for a block that declares nothing. Under the bare Literal
        this same rationale failed the whole numeric block, which sent the FORECAST to the
        LLM salvage rung and fired the parser call anyway.
        """
        reasoning_text = (
            "Some rationale text.\n"
            "```json\n"
            '{"question_type": "numeric", "outcome_type": "integer",'
            ' "declared_percentiles": {"0.1": 10.0, "0.5": 50.0, "0.9": 90.0}}\n'
            "```\n"
        )
        mock_parse_structured = AsyncMock(return_value=OutcomeTypeResult(is_discrete_integer=True))

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=mock_parse_structured),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
                ),
            ),
            patch(
                "metaculus_bot.forecaster_runners.sanitize_percentiles",
                return_value=(_STANDARD_PERCENTILES, None),
            ),
            patch(
                "metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=_built_numeric_prediction()
            ),
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
        ):
            _, discrete_vote = await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm)

        assert discrete_vote is True
        assert mock_parse_structured.await_count == 1

    @pytest.mark.asyncio
    async def test_discrete_vote_none_when_parse_fails(self, numeric_question, forecaster_llm, parser_llm) -> None:
        """When OUTCOME_TYPE parsing fails, discrete_vote is None."""
        mock_parse_structured = AsyncMock(side_effect=ValidationError.from_exception_data(title="test", line_errors=[]))

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="reasoning")),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=mock_parse_structured),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
                ),
            ),
            patch(
                "metaculus_bot.forecaster_runners.sanitize_percentiles",
                return_value=(_STANDARD_PERCENTILES, None),
            ),
            patch(
                "metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=_built_numeric_prediction()
            ),
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
        ):
            _, discrete_vote = await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm)

        assert discrete_vote is None


_MANTIC_URL = f"{MANTIC_SITE_URL}/questions/5001/"
_METACULUS_URL = "https://www.metaculus.com/questions/5001/"

# Post 651's shape: 80% on bin 8, the rest over the other trading days, 0 on the weekend bins 4, 5 and 11.
_DATE_DECLARED = [0.0, 0.01, 0.01, 0.02, 0.03, 0.0, 0.0, 0.05, 0.06, 0.80, 0.01, 0.01, 0.0, 0.0]
_DATE_WEEKEND_BINS = (4, 5, 11)
# An 11-bin count question with an open ceiling: certain of 3, a token on above_range.
_COUNT_DECLARED = [0.0, 0.0, 0.0, 0.05, 0.85, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]


def _pmf_outcome(declared: list[float]) -> ExtractionOutcome[PmfForecast]:
    return ExtractionOutcome(value=PmfForecast(declared=list(declared)), rung="block", block_present=True)


def _compact(values: list[float]) -> str:
    return json.dumps(values, separators=(",", ":"))


@contextmanager
def _percentile_path_never_runs() -> Iterator[None]:
    """Patches that fail the test if the per-bin branch touches the percentile machinery."""
    never = AsyncMock(side_effect=AssertionError("the per-bin branch must not take the percentile path"))
    with (
        patch("metaculus_bot.forecaster_runners._resolve_discrete_vote", new=never),
        patch("metaculus_bot.forecaster_runners.extract_numeric", new=never),
        patch("metaculus_bot.forecaster_runners.extract_date", new=never),
        patch("metaculus_bot.forecaster_runners.parse_structured", new=never),
        patch("metaculus_bot.forecaster_runners.sanitize_percentiles", side_effect=AssertionError("sanitize")),
        patch("metaculus_bot.forecaster_runners.build_numeric_distribution", side_effect=AssertionError("pchip")),
        patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", side_effect=AssertionError("unit guard")),
    ):
        yield


@contextmanager
def _percentile_path_stubs() -> Iterator[None]:
    """The percentile runner's collaborators stubbed, for a question that must NOT take the per-bin branch."""
    with (
        patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="PERCENTILE PROMPT"),
        patch(
            "metaculus_bot.forecaster_runners.parse_structured",
            new=AsyncMock(return_value=OutcomeTypeResult(is_discrete_integer=False)),
        ),
        patch("metaculus_bot.forecaster_runners.sanitize_percentiles", return_value=(_STANDARD_PERCENTILES, None)),
        patch("metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=_built_numeric_prediction()),
        patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
        patch("metaculus_bot.forecaster_runners.log_final_prediction"),
        patch("metaculus_bot.forecaster_runners.log_open_bound_piling_diagnostics"),
    ):
        yield


class TestPerBinBranch:
    """A coarse Mantic grid is elicited per bin: prompt, ``extract_pmf``, ``build_pmf_distribution``.

    The gate is ``numeric.config.elicit_per_bin``: Mantic host, an outcome-space grid, at most 31
    bins. On that path there are no percentiles, so nothing that consumes percentiles may run: the
    discrete vote (a paid parser call on a block that has no ``outcome_type``), the sanitizer, the
    PCHIP build and the unit-mismatch guard (which on grid input would be fail-open). The
    ``MEMBER_FORECAST`` line carries ``elicitation=pmf`` with the platform's ``N + 2`` PMF as both
    ``raw`` (as declared) and ``published`` (after the floor blend).
    """

    @pytest.mark.asyncio
    async def test_a_12_bin_mantic_date_question_takes_the_per_bin_path(
        self, forecaster_llm, parser_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster_runners")
        question = make_real_date_question()
        invoke = AsyncMock(return_value="per-bin reasoning")
        extract = AsyncMock(return_value=_pmf_outcome(_DATE_DECLARED))
        with (
            _percentile_path_never_runs(),
            patch("metaculus_bot.forecaster_runners.pmf_prompt", return_value="PMF PROMPT") as prompt,
            patch.object(forecaster_llm, "invoke", new=invoke),
            patch("metaculus_bot.forecaster_runners.extract_pmf", new=extract),
        ):
            result = await run_date_forecast(question, "research", forecaster_llm, parser_llm)

        assert _last_invoke_arg(invoke) == "PMF PROMPT"
        prompt.assert_called_once()
        view = prompt.call_args.args[0]  # the prompt renders the epoch view, as date_prompt does
        assert isinstance(view, EpochDateQuestion)
        assert view.id_of_question == question.id_of_question
        assert prompt.call_args.args[1] == "research"

        extract.assert_awaited_once()
        assert extract.await_args is not None
        assert extract.await_args.args[0] == "per-bin reasoning"
        grid = extract.await_args.args[1]
        assert isinstance(grid, PmfGrid)
        assert grid.labels == tuple(f"2026-09-{day:02d}" for day in range(8, 20))
        assert extract.await_args.args[2] is parser_llm
        assert extract.await_args.kwargs["question_id"] == question.id_of_question
        assert extract.await_args.kwargs["model_name"] == forecaster_llm.model
        assert extract.await_args.kwargs["prompt_notes"] == build_pmf_parse_notes(grid)

        prediction = result.prediction_value
        assert isinstance(prediction, NumericDistribution)
        assert prediction.is_date is True
        assert result.reasoning == "per-bin reasoning"
        heights = cdf_heights(prediction)
        assert len(heights) == 13
        assert heights[0] == 0.0
        assert heights[-1] == 1.0
        assert_server_accepts_cdf(heights, cdf_size=13, open_lower=False, open_upper=False)
        mass = np.diff(heights)
        assert int(np.argmax(mass)) == 8
        assert mass[8] > 0.78
        for weekend_bin in _DATE_WEEKEND_BINS:
            assert mass[weekend_bin] == pytest.approx(server_min_step(12), abs=2e-9)

        (line,) = _member_forecast_lines(caplog)
        assert line.startswith(
            f"MEMBER_FORECAST: question={question.id_of_question} model=test-forecaster role=member qtype=date "
            f"raw={_compact(_DATE_DECLARED)} published="
        )
        assert line.endswith(" oor_low=0.000000 oor_high=0.000000 elicitation=pmf")
        published = json.loads(line.split(" published=", 1)[1].split(" ", 1)[0])
        assert len(published) == 14
        assert published[0] == 0.0
        assert published[-1] == 0.0
        assert published[1:-1] == pytest.approx(mass.tolist())

    @pytest.mark.asyncio
    async def test_an_11_bin_mantic_count_question_takes_the_per_bin_path_and_casts_no_vote(
        self, forecaster_llm, parser_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The numeric runner's return keeps its shape: ``(prediction, None)``, no discrete vote."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster_runners")
        question = make_count_question(11)
        extract = AsyncMock(return_value=_pmf_outcome(_COUNT_DECLARED))
        with (
            _percentile_path_never_runs(),
            patch("metaculus_bot.forecaster_runners.pmf_prompt", return_value="PMF PROMPT") as prompt,
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="count reasoning")),
            patch("metaculus_bot.forecaster_runners.extract_pmf", new=extract),
        ):
            result, discrete_vote = await run_numeric_forecast(question, "research", forecaster_llm, parser_llm)

        assert discrete_vote is None
        assert prompt.call_args.args[0] is question  # the view of a numeric question is the question itself
        grid = extract.await_args.args[1]  # type: ignore[union-attr]
        assert grid.keys == (*(str(count) for count in range(11)), PMF_ABOVE_RANGE_KEY)

        heights = cdf_heights(result.prediction_value)
        assert len(heights) == 12
        assert heights[0] == 0.0
        assert_server_accepts_cdf(heights, cdf_size=12, open_lower=False, open_upper=True)
        mass = np.diff(heights)
        assert int(np.argmax(mass)) == 3
        assert 1.0 - heights[-1] == pytest.approx(0.05, abs=0.001)

        (line,) = _member_forecast_lines(caplog)
        assert " qtype=numeric " in line
        assert f" raw={_compact(_COUNT_DECLARED)} " in line
        assert line.endswith(" elicitation=pmf")
        oor_high = float(line.rsplit("oor_high=", 1)[1].split(" ", 1)[0])
        assert oor_high == pytest.approx(1.0 - heights[-1], abs=1e-6)

    @pytest.mark.asyncio
    async def test_the_declaration_is_handed_to_the_builder_with_the_model_name(
        self, forecaster_llm, parser_llm
    ) -> None:
        """The call sequence, with the builder stubbed: the ladder's declared vector goes to
        ``build_pmf_distribution`` unchanged, and the model name rides along for the clip marker."""
        question = make_count_question(11)
        built = MagicMock(spec=NumericDistribution)
        with (
            _percentile_path_never_runs(),
            patch("metaculus_bot.forecaster_runners.pmf_prompt", return_value="PMF PROMPT"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="count reasoning")),
            patch(
                "metaculus_bot.forecaster_runners.extract_pmf",
                new=AsyncMock(return_value=_pmf_outcome(_COUNT_DECLARED)),
            ),
            patch("metaculus_bot.forecaster_runners.build_pmf_distribution", return_value=built) as build,
            patch("metaculus_bot.forecaster_runners.published_pmf", return_value=_COUNT_DECLARED),
            patch("metaculus_bot.forecaster_runners.out_of_range_mass", return_value=(0.0, 0.05)),
            patch("metaculus_bot.forecaster_runners.log_final_prediction") as final_log,
        ):
            result, _ = await run_numeric_forecast(question, "research", forecaster_llm, parser_llm)

        assert result.prediction_value is built
        build.assert_called_once_with(_COUNT_DECLARED, question, model_name=forecaster_llm.model)
        final_log.assert_called_once_with(built, question)

    @pytest.mark.asyncio
    async def test_a_declaration_the_builder_refuses_propagates(self, forecaster_llm, parser_llm) -> None:
        """A guard fails shut: mass declared in a closed tail raises out of the runner, never publishes."""
        question = make_count_question(11)
        below_a_closed_floor = [0.3, *_COUNT_DECLARED[1:]]
        with (
            _percentile_path_never_runs(),
            patch("metaculus_bot.forecaster_runners.pmf_prompt", return_value="PMF PROMPT"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="count reasoning")),
            patch(
                "metaculus_bot.forecaster_runners.extract_pmf",
                new=AsyncMock(return_value=_pmf_outcome(below_a_closed_floor)),
            ),
            pytest.raises(ValueError, match="closed lower bound"),
        ):
            await run_numeric_forecast(question, "research", forecaster_llm, parser_llm)

    @pytest.mark.parametrize(
        "question",
        [
            pytest.param(
                make_real_numeric_question().model_copy(update={"page_url": _MANTIC_URL}),
                id="mantic-201-point-continuous-grid",
            ),
            pytest.param(make_count_question(200), id="mantic-200-bin-discrete"),
            pytest.param(make_count_question(11, page_url=_METACULUS_URL), id="metaculus-11-bin-discrete"),
        ],
    )
    @pytest.mark.asyncio
    async def test_the_branch_is_not_taken_off_the_gate(
        self, question: NumericQuestion, forecaster_llm, parser_llm
    ) -> None:
        """A 201-point continuous grid is not an outcome space, 200 bins is above the threshold, and
        Metaculus is not in ``PMF_ELICITATION_PLATFORMS``: each stays on percentiles, byte for byte."""
        never = AsyncMock(side_effect=AssertionError("extract_pmf must not run off the per-bin gate"))
        with (
            _percentile_path_stubs(),
            patch("metaculus_bot.forecaster_runners.pmf_prompt", side_effect=AssertionError("pmf_prompt")),
            patch("metaculus_bot.forecaster_runners.extract_pmf", new=never),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="percentile reasoning")) as invoke,
            patch("metaculus_bot.forecaster_runners.extract_numeric") as extract_numeric,
        ):
            extract_numeric.return_value = ExtractionOutcome(
                value=_STANDARD_PERCENTILES, rung="block", block_present=True
            )
            await run_numeric_forecast(question, "research", forecaster_llm, parser_llm)

        assert _last_invoke_arg(invoke) == "PERCENTILE PROMPT"
        extract_numeric.assert_awaited_once()
        never.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_metaculus_hosted_date_question_stays_on_percentiles(self, forecaster_llm, parser_llm) -> None:
        """Same 12-bin grid as post 651, Metaculus host: the platform clause alone keeps it on percentiles."""
        question = make_real_date_question().model_copy(
            update={"page_url": "https://www.metaculus.com/questions/4001/"}
        )
        never = AsyncMock(side_effect=AssertionError("extract_pmf must not run on a Metaculus question"))
        with (
            patch("metaculus_bot.forecaster_runners.date_prompt", return_value="DATE PROMPT"),
            patch("metaculus_bot.forecaster_runners.extract_pmf", new=never),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="date reasoning")),
            patch("metaculus_bot.forecaster_runners.extract_date") as extract_date,
            patch("metaculus_bot.forecaster_runners.build_guarded_numeric_distribution") as build,
        ):
            extract_date.return_value = ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
            await run_date_forecast(question, "research", forecaster_llm, parser_llm)

        extract_date.assert_awaited_once()
        build.assert_called_once()
        never.assert_not_awaited()


class TestBuildPmfParseNotes:
    def test_lists_every_key_verbatim_and_the_count(self) -> None:
        grid = pmf_grid(make_count_question(11))
        notes = build_pmf_parse_notes(grid)
        for key in grid.keys:
            assert f"'{key}'" in notes
        assert f"exactly {len(grid.keys)} keys" in notes
        assert "'label'" in notes
        assert "'probability'" in notes
        assert PMF_ABOVE_RANGE_KEY in notes
        assert "below_range" not in notes  # the lower bound is closed: no such key to spell

    def test_an_unstated_key_is_left_out_never_written_as_zero(self) -> None:
        """The notes used to say both "never fill it in" and "a key the forecaster gave no probability is 0",
        and a parser obeying the second invented an ``above_range: 0`` that then satisfied the every-key rule.
        A key the forecaster never priced is left out, so ``extract_pmf`` drops the member instead."""
        notes = build_pmf_parse_notes(pmf_grid(make_count_question(11)))
        assert "gave no probability" not in notes
        assert "leave that key out" in notes
        assert "never" in notes.lower()
        assert "fill in" in notes
        assert "ruled out is 0" in notes  # an explicit "cannot happen" IS a stated probability

    def test_a_date_grid_spells_dates_and_no_epoch_second(self) -> None:
        grid = pmf_grid(as_epoch_question(make_real_date_question()))
        notes = build_pmf_parse_notes(grid)
        assert "'2026-09-08'" in notes
        assert "'2026-09-19'" in notes
        assert "1789776000" not in notes
        assert "above_range" not in notes
        assert "below_range" not in notes

    def test_both_open_bounds_spell_both_reserved_keys(self) -> None:
        question = make_real_numeric_question(open_lower_bound=True, open_upper_bound=True).model_copy(
            update={"page_url": _MANTIC_URL, "cdf_size": 21}
        )
        notes = build_pmf_parse_notes(pmf_grid(question))
        assert "'below_range'" in notes
        assert "'above_range'" in notes
