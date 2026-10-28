"""Smoke tests for backtest.py CLI entry point."""

import argparse
import asyncio
import math
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from forecasting_tools import BinaryQuestion
from forecasting_tools.data_models.binary_report import BinaryReport

import backtest
from backtest import _build_parser, _filter_bots
from metaculus_bot.backtest.question_prep import BacktestQuestionSet
from metaculus_bot.backtest.scoring import GroundTruth
from metaculus_bot.benchmark.bot_factory import (
    BENCHMARK_BOT_CONFIG,
    DEFAULT_HELPER_LLMS,
    INDIVIDUAL_MODEL_SPECS,
    create_individual_bots,
)
from metaculus_bot.constants import (
    BACKTEST_DEFAULT_RESOLVED_AFTER,
    BACKTEST_DEFAULT_TOURNAMENT,
    CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD,
    CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD,
    CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD,
)


def test_import_backtest():
    """Module can be imported without errors."""
    assert callable(backtest.run_backtest)


def test_cli_argument_parsing():
    """CLI argument parsing with different combinations."""
    parser = _build_parser()

    args = parser.parse_args([])
    assert args.num_questions == 20
    assert args.include_models is None
    assert args.exclude_models is None

    args = parser.parse_args(["--num-questions", "50"])
    assert args.num_questions == 50

    args = parser.parse_args(["--include-models", "gpt", "qwen"])
    assert args.include_models == ["gpt", "qwen"]

    args = parser.parse_args(["--exclude-models", "grok"])
    assert args.exclude_models == ["grok"]

    args = parser.parse_args(["--resolved-after", "2025-06-01"])
    assert args.resolved_after == "2025-06-01"

    args = parser.parse_args(["--tournament", "my-tournament"])
    assert args.tournament == "my-tournament"


def test_cli_defaults_use_constants():
    """Default CLI values come from constants module."""
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.resolved_after == BACKTEST_DEFAULT_RESOLVED_AFTER
    assert args.tournament == BACKTEST_DEFAULT_TOURNAMENT


def test_backtest_bots_expose_every_init_parameter_to_get_config():
    """Benchmarker snapshots each bot with ``get_config()``, which reads every ``__init__`` parameter back.

    A parameter the bot does not store crashes every backtest before its first forecast.
    """
    bots = create_individual_bots(
        cast("Any", INDIVIDUAL_MODEL_SPECS),
        DEFAULT_HELPER_LLMS,
        BENCHMARK_BOT_CONFIG,
        batch_size=1,
        research_cache={},
    )

    for bot in bots:
        config = bot.get_config()
        assert config["stacking_spread_thresholds"] == {
            "binary": CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD,
            "mc": CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD,
            "numeric": CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD,
        }
        assert config["research_sink"] is None


def test_model_filtering_include():
    """Include filter keeps only matching bots."""
    bot_a = Mock()
    bot_a.name = "gpt-5.1"
    bot_b = Mock()
    bot_b.name = "qwen3-235b"
    bot_c = Mock()
    bot_c.name = "deepseek-3.2"

    filtered = _filter_bots([bot_a, bot_b, bot_c], include_models=["gpt", "qwen"], exclude_models=None)
    assert len(filtered) == 2
    assert cast(Mock, filtered[0]).name == "gpt-5.1"
    assert cast(Mock, filtered[1]).name == "qwen3-235b"


def test_model_filtering_exclude():
    """Exclude filter removes matching bots."""
    bot_a = Mock()
    bot_a.name = "gpt-5.1"
    bot_b = Mock()
    bot_b.name = "qwen3-235b"
    bot_c = Mock()
    bot_c.name = "deepseek-3.2"

    filtered = _filter_bots([bot_a, bot_b, bot_c], include_models=None, exclude_models=["deepseek"])
    assert len(filtered) == 2
    assert cast(Mock, filtered[0]).name == "gpt-5.1"
    assert cast(Mock, filtered[1]).name == "qwen3-235b"


def test_model_filtering_no_filters():
    """No filters returns all bots."""
    bot_a = Mock()
    bot_a.name = "gpt-5.1"

    filtered = _filter_bots([bot_a], include_models=None, exclude_models=None)
    assert len(filtered) == 1


def test_model_filtering_empty_result_raises():
    """Filtering out all bots raises ValueError."""
    bot_a = Mock()
    bot_a.name = "gpt-5.1"

    with pytest.raises(ValueError, match="No bots remaining"):
        _filter_bots([bot_a], include_models=["nonexistent"], exclude_models=None)


@patch("backtest.save_backtest_data")
@patch("backtest.generate_backtest_report")
@patch("backtest.Benchmarker")
@patch("backtest.MonetaryCostManager")
@patch("backtest.typeguard.check_type", side_effect=lambda val, _type: val)
@patch("backtest.create_individual_bots")
@patch("backtest.apply_scoring_patches")
@patch("backtest.screen_research_for_leakage")
@patch("backtest.fetch_resolved_questions")
def test_run_backtest_full_flow(
    mock_fetch: AsyncMock,
    mock_leakage: AsyncMock,
    mock_patches: Mock,
    mock_create_bots: Mock,
    mock_check_type: Mock,
    mock_cost_manager: Mock,
    mock_benchmarker_class: Mock,
    mock_gen_report: Mock,
    mock_save_data: Mock,
) -> None:
    """Offline backtest flow preserves the question, ground truth, scores, and report payload."""
    question = BinaryQuestion(
        question_text="Will X happen?",
        id_of_question=123,
        id_of_post=123,
        page_url="https://example.com/123",
        background_info="",
        resolution_criteria="Resolves yes if X happens.",
        fine_print="",
        published_time=None,
        close_time=None,
    )

    ground_truth = GroundTruth(
        question_id=123,
        question_type="binary",
        resolution=True,
        resolution_string="Yes",
        community_prediction=0.7,
        actual_resolution_time=None,
        question_text="Will X happen?",
    )

    question_set = BacktestQuestionSet(
        questions=[question],
        ground_truths={123: ground_truth},
    )
    mock_fetch.return_value = question_set

    mock_leakage.return_value = ([question], {123: ground_truth}, {})

    mock_bot = Mock()
    mock_bot.name = "test-bot"
    mock_create_bots.return_value = [mock_bot]

    mock_cost_mgr_instance = Mock()
    mock_cost_mgr_instance.__enter__ = Mock(return_value=mock_cost_mgr_instance)
    mock_cost_mgr_instance.__exit__ = Mock(return_value=None)
    mock_cost_mgr_instance.current_usage = "$0.01"
    mock_cost_manager.return_value = mock_cost_mgr_instance

    forecast_report = BinaryReport(
        question=question,
        prediction=0.8,
        explanation="# Forecast reasoning",
        price_estimate=0.01,
        minutes_taken=1.0,
        errors=[],
    )
    benchmark_result = SimpleNamespace(name="test-bot", forecast_reports=[forecast_report])

    mock_benchmarker = Mock()
    mock_benchmarker.run_benchmark = AsyncMock(return_value=[benchmark_result])
    mock_benchmarker_class.return_value = mock_benchmarker

    mock_gen_report.return_value = "# Backtest Report\n..."
    mock_save_data.return_value = "backtests/data.json"

    args = argparse.Namespace(
        num_questions=1,
        resolved_after="2025-12-01",
        tournament="fall-aib-2025",
        include_models=None,
        exclude_models=None,
        research_dir=None,
    )

    asyncio.run(backtest.run_backtest(args))

    mock_fetch.assert_awaited_once_with(
        total_questions=1,
        resolved_after="2025-12-01",
        tournament="fall-aib-2025",
    )
    mock_leakage.assert_awaited_once_with([question], {123: ground_truth})
    mock_create_bots.assert_called_once_with(
        backtest.INDIVIDUAL_MODEL_SPECS,
        backtest.DEFAULT_HELPER_LLMS,
        backtest.BENCHMARK_BOT_CONFIG,
        batch_size=backtest.BENCHMARK_BATCH_SIZE,
        research_cache={},
    )
    mock_patches.assert_called_once()
    mock_benchmarker_class.assert_called_once_with(
        questions_to_use=[question],
        forecast_bots=[mock_bot],
        file_path_to_save_reports="backtests/",
        concurrent_question_batch_size=backtest.BENCHMARK_BATCH_SIZE,
    )
    mock_benchmarker.run_benchmark.assert_awaited_once_with()

    mock_gen_report.assert_called_once()
    generated_args, generated_kwargs = mock_gen_report.call_args
    assert generated_args[1] is question_set
    assert generated_kwargs == {"output_path": "backtests/backtest_report.md"}
    generated_results = generated_args[0]
    assert len(generated_results) == 1
    assert generated_results[0].bot_name == "test-bot"
    assert generated_results[0].num_questions == 1
    assert generated_results[0].num_scored == 1
    assert generated_results[0].num_failed == 0
    scores_by_metric = {score.metric_name: score for score in generated_results[0].scores}
    assert set(scores_by_metric) == {"brier", "log_score"}
    assert {score.question_id for score in generated_results[0].scores} == {123}
    assert scores_by_metric["brier"].bot_score == pytest.approx(0.04)
    assert scores_by_metric["brier"].community_score == pytest.approx(0.09)
    assert scores_by_metric["log_score"].bot_score == pytest.approx(100.0 * (math.log2(0.8) + 1.0))
    assert scores_by_metric["log_score"].community_score == pytest.approx(100.0 * (math.log2(0.7) + 1.0))
    mock_save_data.assert_called_once_with(question_set, generated_results, output_dir="backtests")
