"""Tests for producer-side comment construction in main.py and the
performance-analysis collector.

Covers three just-added paths that were previously untested:

1. ``TemplateForecaster._create_unified_explanation`` appends both a
   tri-state ``<!-- STACKER_OUTCOME=primary|fallback_llm|fallback_median|skipped -->``
   marker AND a legacy ``<!-- STACKED=true/false -->`` marker (for one round
   of back-compat with older parsers) only when the aggregation strategy is
   STACKING or CONDITIONAL_STACKING. The value comes from
   ``self._pipeline.outcomes.pop(qid, None)``; missing entries default to
   ``"fallback_median"`` (conservative — "stacker did not really succeed").

2. ``TemplateForecaster._format_and_expand_research_summary`` annotates
   ``*Forecaster N*`` bullets with the model name pulled from each forecast's
   ``Model: ...`` reasoning prefix. This has to survive comment trimming so
   downstream parsers can recover per-model attribution from the summary
   alone.

3. ``metaculus_bot.performance_analysis.collector._process_single_question``
   wires the new post-data/was-stacked/per-model-numeric-percentiles/
   score-data fields onto each record.

The end-to-end test at the bottom round-trips the producer (main.py) and
consumer (performance_analysis.parsing) together — the critical confidence
check that the two sides actually work in concert.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    MultipleChoiceReport,
    NumericDistribution,
    NumericQuestion,
    NumericReport,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.forecast_report import ResearchWithPredictions
from forecasting_tools.data_models.numeric_report import Percentile

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.comment.formatting import build_unified_explanation
from metaculus_bot.comment.markers import (
    EXPLANATION_DETAIL_UNKNOWN_MARKER,
    FORECASTERS_USED_MARKER_RE,
    STACKED_MARKER_FALSE,
    STACKED_MARKER_TRUE,
    STACKER_OUTCOME_FALLBACK_LLM,
    STACKER_OUTCOME_FALLBACK_MEAN,
    STACKER_OUTCOME_FALLBACK_MEDIAN,
    STACKER_OUTCOME_PRIMARY,
    STACKER_OUTCOME_SKIPPED,
    STACKER_OUTCOME_SKIPPED_CONFIG_OFF,
)
from metaculus_bot.constants import COMMENT_CHAR_LIMIT
from metaculus_bot.performance_analysis.collector import _process_post, _process_single_question
from metaculus_bot.performance_analysis.parsing import (
    parse_explanation_detail_status,
    parse_per_model_forecasts,
    parse_per_model_numeric_percentiles,
    parse_per_model_reasoning_text,
    parse_stacked_marker,
    parse_stacker_outcome_marker,
    parse_stacker_skip_reason_marker,
)
from metaculus_bot.prompts import SUMMARIZER_SOFT_FAIL_BANNER
from metaculus_bot.research.agentic.artifact import render_findings
from metaculus_bot.research.agentic.types import Finding
from metaculus_bot.research.provider_diagnostics import ProviderResult, format_provider_diagnostics_block
from metaculus_bot.stacking import combine_stacker_and_base_reasoning
from tests.conftest import gather_predictions_stub

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_bot(strategy: AggregationStrategy, n_forecasters: int = 2, research_reports: int = 1) -> TemplateForecaster:
    """Create a TemplateForecaster with the minimal LLM config for the strategy.

    STACKING and CONDITIONAL_STACKING require a stacker LLM; CONDITIONAL_STACKING
    additionally requires an analyzer. MEAN/MEDIAN just need forecasters and the
    default helpers. We always pass an analyzer to keep the helper uniform — the
    extra key is harmless for non-conditional strategies.
    """
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    llms: dict[str, str | GeneralLlm | list[GeneralLlm]] = {
        "forecasters": [test_llm] * n_forecasters,
        "stacker": test_llm,
        "analyzer": test_llm,
        "default": test_llm,
        "parser": test_llm,
        "researcher": test_llm,
        "summarizer": test_llm,
    }
    return TemplateForecaster(
        research_reports_per_question=research_reports,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        aggregation_strategy=strategy,
        llms=llms,  # type: ignore[arg-type]
        is_benchmarking=True,
    )


def _make_binary_question(qid: int = 12345) -> MagicMock:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = qid
    q.question_text = "Will it happen?"
    q.page_url = f"https://metaculus.com/questions/{qid}/"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    return q


_BASE_EXPLANATION = "# SUMMARY\n\nBase explanation body."


# ---------------------------------------------------------------------------
# _create_unified_explanation marker injection
# ---------------------------------------------------------------------------


class TestStackedMarkerInjection:
    """Exercises TemplateForecaster._create_unified_explanation.

    The parent ForecastBot._create_unified_explanation is patched to return a
    fixed base string so the test doesn't need a real report/prediction stack.
    """

    @pytest.mark.parametrize(
        "strategy",
        [AggregationStrategy.MEAN, AggregationStrategy.MEDIAN],
    )
    def test_no_marker_for_non_stacking_strategies(self, strategy: AggregationStrategy):
        bot = _make_bot(strategy)
        q = _make_binary_question()

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert "STACKED=" not in out
        assert STACKED_MARKER_TRUE not in out
        assert STACKED_MARKER_FALSE not in out

    def test_non_stacking_cleans_stale_state_defensively(self):
        # If the dict accidentally has an entry for this question, the pop
        # must still clear it regardless of strategy, to avoid leaking state
        # across questions within a run.
        bot = _make_bot(AggregationStrategy.MEAN)
        q = _make_binary_question()
        bot._pipeline.outcomes[q.id_of_question] = "primary"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert q.id_of_question not in bot._pipeline.outcomes

    def test_skip_reason_marker_emitted_and_state_popped(self):
        # The mid-chain link of the STACKER_SKIP_REASON pipeline: stacking_route
        # records the reason on the bot dict, and THIS method is the only place it
        # is popped and handed to build_unified_explanation. Without this test,
        # dropping the skip_reason= pass-through (or the pop) leaves every published
        # comment without the marker while both chain ends stay green.
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question()
        bot._pipeline.outcomes[q.id_of_question] = "skipped"
        bot._pipeline.skip_reasons[q.id_of_question] = "single_forecaster"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert "<!-- STACKER_SKIP_REASON=single_forecaster -->" in out
        assert parse_stacker_skip_reason_marker(out) == "single_forecaster"
        # State cleaned: a stale entry must not leak onto the next question.
        assert q.id_of_question not in bot._pipeline.skip_reasons

    def test_stacking_primary_emits_primary_and_legacy_true_markers(self):
        # primary outcome → tri-state STACKER_OUTCOME=primary AND legacy STACKED=true
        # (the legacy marker stays during one round of back-compat).
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question()
        bot._pipeline.outcomes[q.id_of_question] = "primary"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert STACKER_OUTCOME_PRIMARY in out
        assert STACKED_MARKER_TRUE in out
        assert STACKED_MARKER_FALSE not in out
        assert parse_stacker_outcome_marker(out) == "primary"
        assert parse_stacked_marker(out) is True
        # State cleaned
        assert q.id_of_question not in bot._pipeline.outcomes

    def test_stacking_fallback_llm_emits_fallback_llm_and_legacy_true(self):
        # fallback_llm outcome (primary stacker failed; fallback LLM succeeded)
        # → STACKER_OUTCOME=fallback_llm AND legacy STACKED=true (since the
        # ensemble value still came from a stacker LLM, not MEDIAN).
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question()
        bot._pipeline.outcomes[q.id_of_question] = "fallback_llm"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert STACKER_OUTCOME_FALLBACK_LLM in out
        assert STACKED_MARKER_TRUE in out
        assert parse_stacker_outcome_marker(out) == "fallback_llm"
        assert parse_stacked_marker(out) is True

    def test_stacking_fallback_median_emits_fallback_median_and_legacy_false(self):
        # fallback_median outcome (BOTH stacker LLMs failed; MEDIAN was used)
        # → STACKER_OUTCOME=fallback_median AND legacy STACKED=false. This is
        # the load-bearing fix from the May 2026 analysis: previously this path
        # silently emitted STACKED=true, contaminating treatment-effect cuts.
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question()
        bot._pipeline.outcomes[q.id_of_question] = "fallback_median"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert STACKER_OUTCOME_FALLBACK_MEDIAN in out
        assert STACKED_MARKER_FALSE in out
        assert STACKED_MARKER_TRUE not in out
        assert parse_stacker_outcome_marker(out) == "fallback_median"
        assert parse_stacked_marker(out) is False

    def test_stacking_skipped_emits_skipped_and_legacy_false(self):
        # skipped outcome (CONDITIONAL_STACKING below threshold; never
        # invoked the stacker) → STACKER_OUTCOME=skipped AND legacy STACKED=false.
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        q = _make_binary_question()
        bot._pipeline.outcomes[q.id_of_question] = "skipped"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert STACKER_OUTCOME_SKIPPED in out
        assert STACKED_MARKER_FALSE in out
        assert STACKED_MARKER_TRUE not in out
        assert parse_stacker_outcome_marker(out) == "skipped"
        assert parse_stacked_marker(out) is False

    def test_stacking_skipped_config_off_emits_marker_and_legacy_false(self):
        # skipped_config_off (CONDITIONAL_STACKING above threshold, but the
        # per-type <TYPE>_STACKING_ENABLED gate was off) → STACKER_OUTCOME=
        # skipped_config_off AND legacy STACKED=false. Round-trips through the
        # parser so residual pulls can tell config-suppression from
        # below-threshold skips without git archaeology.
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        q = _make_binary_question()
        bot._pipeline.outcomes[q.id_of_question] = "skipped_config_off"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert STACKER_OUTCOME_SKIPPED_CONFIG_OFF in out
        assert STACKED_MARKER_FALSE in out
        assert STACKED_MARKER_TRUE not in out
        assert parse_stacker_outcome_marker(out) == "skipped_config_off"
        assert parse_stacked_marker(out) is False

    def test_stacking_budget_skip_emits_fallback_mean_marker(self):
        # F15: under AggregationStrategy.STACKING, the wall-clock budget-skip
        # path forces base-combine via MEAN (not MEDIAN — see main.py:1308-1314).
        # That path must therefore emit STACKER_OUTCOME=fallback_mean so
        # downstream residual-analysis cuts that bucket on aggregation strategy
        # don't conflate it with the MEDIAN-fallback bucket. The CONDITIONAL_
        # STACKING budget-skip path (covered by the next test) keeps
        # fallback_median because that strategy's base-combine uses MEDIAN.
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question()
        bot._pipeline.outcomes[q.id_of_question] = "fallback_mean"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert STACKER_OUTCOME_FALLBACK_MEAN in out
        assert STACKED_MARKER_FALSE in out
        assert STACKED_MARKER_TRUE not in out
        assert parse_stacker_outcome_marker(out) == "fallback_mean"
        assert parse_stacked_marker(out) is False

    def test_conditional_stacking_budget_skip_keeps_fallback_median_marker(self):
        # F15 control case: CONDITIONAL_STACKING budget-skip still uses MEDIAN
        # for base-combine, so the fallback_median marker remains accurate
        # there. This test pins that the marker is NOT changed under
        # CONDITIONAL_STACKING by the F15 fix.
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        q = _make_binary_question()
        bot._pipeline.outcomes[q.id_of_question] = "fallback_median"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert STACKER_OUTCOME_FALLBACK_MEDIAN in out
        assert STACKED_MARKER_FALSE in out
        assert STACKED_MARKER_TRUE not in out
        assert parse_stacker_outcome_marker(out) == "fallback_median"
        assert parse_stacked_marker(out) is False

    def test_conditional_stacking_primary_emits_primary_marker(self):
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        q = _make_binary_question()
        bot._pipeline.outcomes[q.id_of_question] = "primary"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert STACKER_OUTCOME_PRIMARY in out
        assert STACKED_MARKER_TRUE in out
        assert parse_stacker_outcome_marker(out) == "primary"
        assert parse_stacked_marker(out) is True

    def test_conditional_stacking_missing_dict_entry_raises(self):
        # Every reachable code path in _aggregate_predictions populates
        # pipeline outcomes before _create_unified_explanation runs. A missing
        # entry under STACKING/CONDITIONAL_STACKING means a real bug — fail
        # loudly rather than silently mislabel as fallback_median.
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        q = _make_binary_question()
        assert q.id_of_question not in bot._pipeline.outcomes

        with (
            patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION),
            pytest.raises(AssertionError, match="stacker_outcome must be provided"),
        ):
            bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

    def test_qid_none_under_stacking_raises(self):
        # Defensive branch in _create_unified_explanation: if id_of_question is
        # None, popped is None and the fail-fast assert fires. This shouldn't
        # happen in practice (upstream assert in _research_and_make_predictions
        # guarantees a non-None qid) but lock the behavior.
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question()
        # Bypass the pydantic id_of_question setter via __dict__ (model_construct
        # would also work but this keeps the test minimal).
        object.__setattr__(q, "id_of_question", None)

        with (
            patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION),
            pytest.raises(AssertionError, match="stacker_outcome must be provided"),
        ):
            bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

    def test_marker_survives_trim_comment(self):
        # trim_comment preserves the tail when truncating; the marker is
        # appended at the very end, so it must survive even when the base
        # text is pushed over the comment char limit.
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question()
        bot._pipeline.outcomes[q.id_of_question] = "primary"
        huge_base = "# SUMMARY\n" + ("X" * (COMMENT_CHAR_LIMIT + 1000))

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=huge_base):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert len(out) <= COMMENT_CHAR_LIMIT
        assert STACKED_MARKER_TRUE in out
        assert parse_stacked_marker(out) is True

    def test_annotated_bullets_survive_trim(self):
        # Load-bearing invariant: when a comment overflows COMMENT_CHAR_LIMIT,
        # the structured summary-and-tail preserving trim keeps the Forecasts
        # summary bullets intact (so downstream parse_per_model_forecasts still
        # gets per-model attribution) AND the tail STACKED marker.
        #
        # Build a full fake base comment whose rationale body is huge enough to
        # force trim_comment to fire. The summary has three annotated bullets
        # and the "### Research Summary" marker that _trim_preserving_summary_and_tail
        # anchors on. If annotation wiring ran before trim (as in production),
        # the annotated bullets are already present in the base_text; the trim
        # just needs to preserve them.
        #
        # NOTE: model strings here (and in the parametrized fixtures below) are
        # *illustrative* — they need to be plausible names that the parser can
        # recognize, but the specific identity of "gpt-5.5" / "claude-opus-4.7" /
        # "gemini-3.1-pro-preview" doesn't drive what's under test (trim
        # preservation + per-model regex extraction). On forecaster rotation,
        # update them to current names but verify the parser logic still
        # matches the new shape (provider/family-name/version pattern).
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question()
        bot._pipeline.outcomes[q.id_of_question] = "primary"

        summary_head = (
            "# SUMMARY\n"
            "*Question*: Will X?\n\n"
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: 72.0%\n"
            "*Forecaster 2 (claude-opus-4.7)*: 68.0%\n"
            "*Forecaster 3 (gemini-3.1-pro-preview)*: 80.0%\n\n"
            "### Research Summary\nshort research.\n\n"
            "================================================================================\n"
            "FORECAST SECTION:\n\n"
            "## R1: Forecaster 1 Reasoning\nModel: openrouter/openai/gpt-5.5\n\n"
        )
        huge_rationale = "X" * (COMMENT_CHAR_LIMIT + 50_000)
        base_text = summary_head + huge_rationale

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=base_text):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        # Trim fired (output shorter than input).
        assert len(out) < len(base_text)
        assert len(out) <= COMMENT_CHAR_LIMIT
        # Summary head preserved through the trim.
        assert "*Forecaster 1 (gpt-5.5)*: 72.0%" in out
        assert "*Forecaster 2 (claude-opus-4.7)*: 68.0%" in out
        assert "*Forecaster 3 (gemini-3.1-pro-preview)*: 80.0%" in out
        # Marker block appended at the end survives. The STACKED marker is
        # followed by a TOOLS_USED marker since Workstream C activation, so
        # the trailing token is TOOLS_USED=true/false — STACKED still appears,
        # just not as the last line.
        assert STACKED_MARKER_TRUE in out
        assert parse_stacked_marker(out) is True
        # Parser recovers per-model attribution keyed by model name, not "Forecaster N".
        per_model = parse_per_model_forecasts(out)
        assert per_model == {
            "gpt-5.5": "72.0%",
            "claude-opus-4.7": "68.0%",
            "gemini-3.1-pro-preview": "80.0%",
        }


# ---------------------------------------------------------------------------
# _format_and_expand_research_summary model annotation
# ---------------------------------------------------------------------------


_PARENT_SUMMARY = (
    "## Report 1 Summary\n"
    "### Forecasts\n"
    "*Forecaster 1*: 72.0%\n"
    "*Forecaster 2*: 68.0%\n"
    "*Forecaster 3*: 80.0%\n\n"
    "### Research Summary\n"
    "Some research.\n"
)


def _make_prediction_with_model(
    prob: float,
    model_tag: str | None,
    body: str = "analysis...",
) -> ReasonedPrediction:
    reasoning = body if model_tag is None else f"Model: {model_tag}\n\n{body}"
    return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)


class TestFormatAndExpandResearchSummaryAnnotation:
    """Exercises TemplateForecaster._format_and_expand_research_summary.

    The parent ForecastBot._format_and_expand_research_summary is patched to
    return a canned summary so we can focus on the annotation wiring.
    """

    def _call(self, predictions: list[ReasonedPrediction], parent_return: str = _PARENT_SUMMARY) -> str:
        research = ResearchWithPredictions(
            research_report="raw",
            summary_report="summary",
            errors=[],
            predictions=predictions,
        )

        with patch.object(
            ForecastBot,
            "_format_and_expand_research_summary",
            return_value=parent_return,
        ):
            return TemplateForecaster._format_and_expand_research_summary(
                report_number=1,
                report_type=BinaryReport,
                predicted_research=research,
            )

    def test_all_three_forecasts_get_annotated(self):
        predictions = [
            _make_prediction_with_model(0.72, "openrouter/openai/gpt-5.5"),
            _make_prediction_with_model(0.68, "openrouter/anthropic/claude-opus-4.7"),
            _make_prediction_with_model(0.80, "openrouter/google/gemini-3.1-pro-preview"),
        ]
        out = self._call(predictions)
        assert "*Forecaster 1 (gpt-5.5)*: 72.0%" in out
        assert "*Forecaster 2 (claude-opus-4.7)*: 68.0%" in out
        assert "*Forecaster 3 (gemini-3.1-pro-preview)*: 80.0%" in out

    def test_one_forecast_missing_model_prefix_leaves_that_bullet_unannotated(self):
        predictions = [
            _make_prediction_with_model(0.72, "openrouter/openai/gpt-5.5"),
            _make_prediction_with_model(0.68, None),
            _make_prediction_with_model(0.80, "openrouter/google/gemini-3.1-pro-preview"),
        ]
        out = self._call(predictions)
        assert "*Forecaster 1 (gpt-5.5)*: 72.0%" in out
        assert "*Forecaster 2*: 68.0%" in out
        assert "*Forecaster 2 (" not in out
        assert "*Forecaster 3 (gemini-3.1-pro-preview)*: 80.0%" in out

    def test_all_forecasts_missing_model_prefix_leaves_all_bullets_unannotated(self):
        predictions = [
            _make_prediction_with_model(0.72, None),
            _make_prediction_with_model(0.68, None),
            _make_prediction_with_model(0.80, None),
        ]
        out = self._call(predictions)
        assert "*Forecaster 1*: 72.0%" in out
        assert "*Forecaster 2*: 68.0%" in out
        assert "*Forecaster 3*: 80.0%" in out
        assert "(" not in out.split("### Forecasts")[1].split("### Research Summary")[0]

    def test_bullet_count_less_than_forecast_count_known_indices_still_annotated(self):
        # Edge case: stacking collapses predictions down to a single bullet
        # even though multiple base models fed in. Indices beyond what the
        # parent returned simply have no bullet to annotate — no crash.
        parent_return = "## Report 1 Summary\n### Forecasts\n*Forecaster 1*: 72.0%\n\n### Research Summary\nstuff\n"
        predictions = [
            _make_prediction_with_model(0.72, "openrouter/openai/gpt-5.5"),
            _make_prediction_with_model(0.68, "openrouter/anthropic/claude-opus-4.7"),
            _make_prediction_with_model(0.80, "openrouter/google/gemini-3.1-pro-preview"),
        ]
        out = self._call(predictions, parent_return=parent_return)
        assert "*Forecaster 1 (gpt-5.5)*: 72.0%" in out
        assert "Forecaster 2" not in out
        assert "Forecaster 3" not in out

    def test_bullet_count_more_than_forecast_count_unknown_indices_left_alone(self):
        # Converse edge case: parent returned more bullets than we have
        # predictions for. Known indices (1, 2) annotated, unknown (3) left alone.
        parent_return = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1*: 72.0%\n"
            "*Forecaster 2*: 68.0%\n"
            "*Forecaster 3*: 80.0%\n"
            "*Forecaster 4*: 65.0%\n\n"
            "### Research Summary\nstuff\n"
        )
        predictions = [
            _make_prediction_with_model(0.72, "openrouter/openai/gpt-5.5"),
            _make_prediction_with_model(0.68, "openrouter/anthropic/claude-opus-4.7"),
        ]
        out = self._call(predictions, parent_return=parent_return)
        assert "*Forecaster 1 (gpt-5.5)*: 72.0%" in out
        assert "*Forecaster 2 (claude-opus-4.7)*: 68.0%" in out
        assert "*Forecaster 3*: 80.0%" in out
        assert "*Forecaster 4*: 65.0%" in out


# ---------------------------------------------------------------------------
# Collector _process_single_question
# ---------------------------------------------------------------------------


def _make_post_data(category_name: str = "Politics") -> dict:
    return {
        "id": 999,
        "title": "Some question",
        "projects": {"category": [{"name": category_name}]},
    }


def _make_binary_q_dict(
    qid: int = 101,
    resolution: str | None = "yes",
    forecast_values: list[float] | None = None,
    score_data: dict | None = None,
) -> dict:
    if forecast_values is None:
        forecast_values = [0.3, 0.7]
    my_forecasts = {
        "latest": {"forecast_values": forecast_values},
    }
    if score_data is not None:
        my_forecasts["score_data"] = score_data
    return {
        "id": qid,
        "type": "binary",
        "resolution": resolution,
        "my_forecasts": my_forecasts,
        "scaling": {},
        "open_lower_bound": False,
        "open_upper_bound": False,
        "options": None,
        "title": "Will X?",
        "nr_forecasters": 42,
        "open_time": "2024-01-01T00:00:00Z",
        "actual_resolve_time": "2024-06-01T00:00:00Z",
        "scheduled_resolve_time": "2024-06-01T00:00:00Z",
    }


def _make_numeric_q_dict(
    qid: int = 202,
    resolution: str = "42.5",
    cdf: list[float] | None = None,
) -> dict:
    # Build a plausible 201-point CDF roughly centered at 50 with range [0, 100]
    if cdf is None:
        cdf = [i / 200.0 for i in range(201)]  # uniform, 0.0 .. 1.0
    return {
        "id": qid,
        "type": "numeric",
        "resolution": resolution,
        "my_forecasts": {"latest": {"forecast_values": cdf}},
        "scaling": {"range_min": 0.0, "range_max": 100.0, "zero_point": None},
        "open_lower_bound": False,
        "open_upper_bound": False,
        "options": None,
        "title": "What will X be?",
        "nr_forecasters": 10,
        "open_time": "2024-01-01T00:00:00Z",
        "actual_resolve_time": "2024-06-01T00:00:00Z",
        "scheduled_resolve_time": "2024-06-01T00:00:00Z",
    }


def _make_mc_q_dict(
    qid: int = 303,
    resolution: str = "Option A",
    options: list[str] | None = None,
    forecast_values: list[float] | None = None,
) -> dict:
    if options is None:
        options = ["Option A", "Option B"]
    if forecast_values is None:
        forecast_values = [0.8, 0.2]
    return {
        "id": qid,
        "type": "multiple_choice",
        "resolution": resolution,
        "my_forecasts": {"latest": {"forecast_values": forecast_values}},
        "scaling": {},
        "open_lower_bound": False,
        "open_upper_bound": False,
        "options": options,
        "title": "Which option?",
        "nr_forecasters": 10,
        "open_time": "2024-01-01T00:00:00Z",
        "actual_resolve_time": "2024-06-01T00:00:00Z",
        "scheduled_resolve_time": "2024-06-01T00:00:00Z",
    }


class TestForecastersUsedDisclosure:
    """The published comment must state the ensemble size actually used, so a
    degraded publish (a dropped model) is distinguishable in the durable record
    from a genuine roster change (both otherwise look like "fewer than N bullets").

    Producer side: TemplateForecaster._create_unified_explanation reports the
    contributor count recorded by _research_and_make_predictions, and n_configured
    from the roster. Consumer side: _process_post parses the marker off the comment
    and _process_single_question carries it onto the record for residual analysis.
    """

    def _bot(self):
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)  # 2 forecasters configured
        bot._pipeline.outcomes[12345] = "skipped"
        return bot

    async def _run_fanout(self, bot, question, prediction_values: list[float]) -> ResearchWithPredictions:
        """Run the real _research_and_make_predictions once and return its collection.

        Everything outside ensemble-size bookkeeping is stubbed: research, the
        forecaster fan-out (whose return value IS the surviving per-model
        prediction list), crux extraction, targeted search, and the stacker
        aggregate.
        """
        predictions = [
            ReasonedPrediction(prediction_value=value, reasoning=f"Model: openrouter/provider/m{i}\nbody")
            for i, value in enumerate(prediction_values)
        ]
        with (
            patch.object(
                bot, "_get_notepad", new=AsyncMock(return_value=MagicMock(total_research_reports_attempted=0))
            ),
            patch.object(bot, "run_research", new=AsyncMock(return_value="research body")),
            patch.object(
                bot,
                "_forecaster_with_soft_deadline",
                new=AsyncMock(return_value=ReasonedPrediction(prediction_value=0.5, reasoning="stub")),
            ),
            patch.object(
                bot, "_gather_predictions_with_wall_clock", new=gather_predictions_stub((predictions, [], None))
            ),
            patch("metaculus_bot.stacking_route.extract_disagreement_crux", new=AsyncMock(return_value="the crux")),
            patch("metaculus_bot.stacking_route.run_targeted_search", new=AsyncMock(return_value="targeted research")),
            patch.object(bot, "_aggregate_predictions", new=AsyncMock(return_value=0.6)),
        ):
            return await bot._research_and_make_predictions(question)

    def _build_comment(self, bot, question, collections: list[ResearchWithPredictions]) -> str:
        """Build the published comment from collections the fan-out already returned."""
        # The real _aggregate_predictions sets this; _run_fanout mocks it out, and
        # build_unified_explanation asserts on its presence under stacking.
        bot._pipeline.outcomes.setdefault(question.id_of_question, "primary")
        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            return bot._create_unified_explanation(question, collections, 0.6, 0.01, 1.0)

    async def _publish_via_pipeline(self, bot, question, prediction_values: list[float]):
        """The producer path end to end: fan out, then build the comment from what the
        fan-out returned. Returns (collection, comment_text).
        """
        collection = await self._run_fanout(bot, question, prediction_values)
        return collection, self._build_comment(bot, question, [collection])

    @pytest.mark.asyncio
    async def test_stacked_publish_discloses_every_contributing_forecaster(self):
        """The stacked path publishes ONE aggregated prediction, so counting the
        returned collection reported FORECASTERS_USED=1/3 on a healthy three-model
        run — manufacturing exactly the false degradation reading this marker exists
        to rule out. The count must come from the fan-out, not the collection.
        """
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING, n_forecasters=3)
        q = _make_binary_question()
        # Binary range 0.75 >> the 0.15 threshold, so stacking fires.
        collection, comment = await self._publish_via_pipeline(bot, q, [0.10, 0.50, 0.85])

        assert bot._pipeline.counters.conditional_stacking_triggered_count == 1, (
            "precondition: stacking must have fired"
        )
        assert len(collection.predictions) == 1, "precondition: stacking collapses the ensemble to one aggregate"
        match = FORECASTERS_USED_MARKER_RE.search(comment)
        assert match is not None
        assert match.groups() == ("3", "3")

    @pytest.mark.asyncio
    async def test_base_combine_publish_discloses_survivors_of_configured(self):
        """Non-stacked path (spread at/below threshold): two survivors of three
        configured must read 2/3, agreeing with the stacked path's meaning of
        n_used.
        """
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING, n_forecasters=3)
        q = _make_binary_question()
        collection, comment = await self._publish_via_pipeline(bot, q, [0.45, 0.55])

        assert bot._pipeline.counters.conditional_stacking_skipped_count == 1, (
            "precondition: stacking must have been skipped"
        )
        assert len(collection.predictions) == 2
        match = FORECASTERS_USED_MARKER_RE.search(comment)
        assert match is not None
        assert match.groups() == ("2", "3")

    @pytest.mark.asyncio
    async def test_single_forecaster_short_circuit_discloses_one_of_configured(self):
        """The n==1 short-circuit (skips spread + stacking) must also report the
        fan-out count: 1 of 3, the genuinely degraded publish.
        """
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING, n_forecasters=3)
        q = _make_binary_question()
        collection, comment = await self._publish_via_pipeline(bot, q, [0.42])

        assert len(collection.predictions) == 1
        match = FORECASTERS_USED_MARKER_RE.search(comment)
        assert match is not None
        assert match.groups() == ("1", "3")

    @pytest.mark.asyncio
    async def test_multi_report_publish_counts_forecasters_from_every_report(self):
        """With research_reports_per_question > 1 the framework fans out once per
        report and hands every collection to one comment, so the count has to
        accumulate across reports: it is defined as the number of per-model summary
        bullets the comment carries. Assigning instead of accumulating would report
        the LAST report's survivors (3) on a comment showing 5 bullets, which reads
        as a drop that did not happen.

        Note the denominator stays the ROSTER size, so a multi-report run publishes
        used > configured (5/3 here). No entrypoint configures more than one report
        (cli.py, benchmark/bot_factory.py and ablation/forecasters.py all pin 1), so
        production never emits that shape.
        """
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING, n_forecasters=3, research_reports=2)
        q = _make_binary_question()
        # Both spreads sit at/below the 0.15 threshold, so each report base-combines
        # and keeps its own per-model predictions (2 bullets, then 3).
        first = await self._run_fanout(bot, q, [0.45, 0.55])
        second = await self._run_fanout(bot, q, [0.44, 0.50, 0.56])

        comment = self._build_comment(bot, q, [first, second])

        assert [len(first.predictions), len(second.predictions)] == [2, 3]
        match = FORECASTERS_USED_MARKER_RE.search(comment)
        assert match is not None
        assert match.groups() == ("5", "3")

    def _collection(self, n_predictions: int) -> ResearchWithPredictions:
        return ResearchWithPredictions(
            research_report="# RESEARCH\nbody",
            summary_report="_summary_",
            errors=[],
            predictions=[
                ReasonedPrediction(prediction_value=0.6, reasoning=f"Model: openrouter/provider/m{i}\nr")
                for i in range(n_predictions)
            ],
        )

    def test_degraded_publish_discloses_used_of_configured(self):
        bot = self._bot()
        q = _make_binary_question()
        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [self._collection(1)], 0.6, 0.01, 1.0)
        # 1 of 2 configured contributed — a dropped model, disclosed.
        match = FORECASTERS_USED_MARKER_RE.search(out)
        assert match is not None
        assert match.groups() == ("1", "2")

    def test_full_ensemble_discloses_all_used(self):
        bot = self._bot()
        q = _make_binary_question()
        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [self._collection(2)], 0.6, 0.01, 1.0)
        match = FORECASTERS_USED_MARKER_RE.search(out)
        assert match is not None
        assert match.groups() == ("2", "2")

    def test_delegated_run_discloses_parent_fanout_width_not_zero(self):
        """With no "forecasters" roster the bot delegates to the parent, whose
        fan-out width is predictions_per_research_report. The configured count must
        report that width — an empty roster previously published `3/0`, inverting
        the used-under-configured invariant that residual analysis reads.
        """
        test_llm = GeneralLlm(model="test-model", temperature=0.0)
        bot = TemplateForecaster(
            research_reports_per_question=1,
            predictions_per_research_report=3,
            publish_reports_to_metaculus=False,
            aggregation_strategy=AggregationStrategy.MEAN,
            llms={"default": test_llm, "parser": test_llm, "researcher": test_llm, "summarizer": test_llm},  # type: ignore[arg-type]
            is_benchmarking=True,
        )
        assert not bot._forecaster_llms, "precondition: this is the delegated path"
        q = _make_binary_question()
        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [self._collection(3)], 0.6, 0.01, 1.0)
        match = FORECASTERS_USED_MARKER_RE.search(out)
        assert match is not None
        assert match.groups() == ("3", "3")

    def test_record_carries_parsed_ensemble_size(self):
        post = _make_post_data()
        q = _make_binary_q_dict(forecast_values=[0.3, 0.7])
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text=("# SUMMARY\nfoo\n<!-- FORECASTERS_USED=2/3 -->\n<!-- EXPLANATION_DETAIL=unknown -->\n"),
            comment_id=5,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=None,
            post_data=post,
            forecasters_used=(2, 3),
        )
        assert rec is not None
        assert rec["forecasters_used"] == 2
        assert rec["forecasters_configured"] == 3
        assert rec["explanation_detail_status"] == "unknown"

    def test_process_post_parses_marker_off_the_comment_onto_the_record(self):
        """The parse → pass → materialize chain, driven from the top.

        The two tests below hand ``forecasters_used`` to _process_single_question
        pre-parsed, so neither notices if _process_post stops parsing the marker.
        This one starts from raw comment text, which is what a pull actually has.
        """
        post_data = {
            "id": 771,
            "title": "Will it happen?",
            "projects": {"category": [{"name": "Politics"}]},
            "question": {
                "id": 551,
                "type": "binary",
                "resolution": "yes",
                "my_forecasts": {"latest": {"forecast_values": [0.3, 0.7]}},
                "scaling": {},
                "open_lower_bound": False,
                "open_upper_bound": False,
                "title": "Will it happen?",
                "nr_forecasters": 40,
                "open_time": "2024-01-01T00:00:00Z",
                "actual_resolve_time": "2024-06-01T00:00:00Z",
                "scheduled_resolve_time": "2024-06-01T00:00:00Z",
            },
        }
        comment_text = "## Report 1 Summary\n### Forecasts\n*Forecaster 1 (m1)*: 70.0%\n<!-- FORECASTERS_USED=2/3 -->\n"
        comment_lookup = {771: {"id": 4242, "on_post": 771, "text": comment_text, "created_at": "2024-05-01T00:00:00Z"}}

        records = _process_post(post_data, comment_lookup)

        assert len(records) == 1
        assert records[0]["forecasters_used"] == 2
        assert records[0]["forecasters_configured"] == 3

    def test_record_ensemble_size_none_when_marker_absent(self):
        post = _make_post_data()
        q = _make_binary_q_dict(forecast_values=[0.3, 0.7])
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="# SUMMARY\nno marker\n",
            comment_id=5,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=None,
            post_data=post,
            forecasters_used=None,
        )
        assert rec is not None
        assert rec["forecasters_used"] is None
        assert rec["forecasters_configured"] is None


class TestCollectorProcessSingleQuestion:
    """Exercises metaculus_bot.performance_analysis.collector._process_single_question.

    The contract under test:
    - Binary/numeric/MC scoring is populated when resolution + forecast exist.
    - was_stacked, per_model_forecasts, per_model_numeric_percentiles, and
      metaculus_scores are carried onto the record verbatim.
    - category comes from post_data.projects.category[0].name.
    - None is returned when resolution_raw is missing or parse_resolution
      flags the question for skipping.
    """

    def test_binary_question_populates_scores_and_prob_yes(self):
        post = _make_post_data()
        q = _make_binary_q_dict(forecast_values=[0.3, 0.7])
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="# SUMMARY\nfoo\n",
            comment_id=5,
            per_model={"gpt-5.5": "70.0%"},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is not None
        assert rec["type"] == "binary"
        assert rec["our_prob_yes"] == 0.7
        assert rec["brier_score"] is not None
        assert rec["log_score"] is not None
        assert rec["was_stacked"] is True
        assert rec["per_model_forecasts"] == {"gpt-5.5": "70.0%"}
        assert rec["per_model_numeric_percentiles"] == {}

    def test_numeric_question_populates_numeric_log_score_and_percentiles(self):
        post = _make_post_data()
        q = _make_numeric_q_dict(resolution="42.5")
        per_model_percentiles = {"gpt-5.5": [(2.5, 10.0), (50.0, 42.0), (97.5, 85.0)]}
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="# SUMMARY\n",
            comment_id=7,
            per_model={},
            per_model_numeric_percentiles=per_model_percentiles,
            was_stacked=False,
            post_data=post,
        )
        assert rec is not None
        assert rec["type"] == "numeric"
        assert rec["numeric_log_score"] is not None
        assert rec["our_prob_yes"] is None
        assert rec["per_model_numeric_percentiles"] == per_model_percentiles
        assert rec["was_stacked"] is False

    def test_multiple_choice_question_populates_mc_log_score(self):
        post = _make_post_data()
        q = _make_mc_q_dict(
            resolution="Option A",
            options=["Option A", "Option B"],
            forecast_values=[0.8, 0.2],
        )
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="# SUMMARY\n",
            comment_id=9,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=None,
            post_data=post,
        )
        assert rec is not None
        assert rec["type"] == "multiple_choice"
        assert rec["mc_log_score"] is not None
        assert rec["was_stacked"] is None

    def test_mc_question_stores_full_option_probs_via_process_post(self):
        """_process_post stores full option probability dicts for MC questions.

        This is the load-bearing fix: previously only the first option line was
        captured, making MC residual analysis impossible.
        """
        mc_comment_text = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: \n"
            "- Option A: 60.0%\n"
            "- Option B: 25.0%\n"
            "- Option C: 15.0%\n"
            "\n"
            "*Forecaster 2 (claude-opus-4.7)*: \n"
            "- Option A: 55.0%\n"
            "- Option B: 30.0%\n"
            "- Option C: 15.0%\n"
            "\n"
            "### Research Summary\n"
            "research here\n"
        )
        post_data = {
            "id": 888,
            "title": "Which option wins?",
            "projects": {"category": [{"name": "Politics"}]},
            "question": {
                "id": 444,
                "type": "multiple_choice",
                "resolution": "Option A",
                "my_forecasts": {"latest": {"forecast_values": [0.6, 0.25, 0.15]}},
                "scaling": {},
                "open_lower_bound": False,
                "open_upper_bound": False,
                "options": ["Option A", "Option B", "Option C"],
                "title": "Which option wins?",
                "nr_forecasters": 50,
                "open_time": "2024-01-01T00:00:00Z",
                "actual_resolve_time": "2024-06-01T00:00:00Z",
                "scheduled_resolve_time": "2024-06-01T00:00:00Z",
            },
        }
        comment_lookup = {
            888: {"id": 9999, "on_post": 888, "text": mc_comment_text, "created_at": "2024-05-01T00:00:00Z"}
        }

        records = _process_post(post_data, comment_lookup)
        assert len(records) == 1
        rec = records[0]
        assert rec["type"] == "multiple_choice"
        # The fix: per_model_forecasts should contain full option dicts, not single strings.
        per_model = rec["per_model_forecasts"]
        assert "gpt-5.5" in per_model
        assert "claude-opus-4.7" in per_model
        assert per_model["gpt-5.5"] == {"Option A": 0.60, "Option B": 0.25, "Option C": 0.15}
        assert per_model["claude-opus-4.7"] == {"Option A": 0.55, "Option B": 0.30, "Option C": 0.15}

    def test_binary_question_per_model_unchanged_with_mc_fix(self):
        """Binary per_model_forecasts remains as raw strings (regression guard)."""
        binary_comment_text = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: 72.0%\n"
            "*Forecaster 2 (claude-opus-4.7)*: 68.0%\n"
            "\n"
            "### Research Summary\n"
            "research here\n"
        )
        post_data = {
            "id": 889,
            "title": "Will X happen?",
            "projects": {"category": [{"name": "Science"}]},
            "question": _make_binary_q_dict(),
        }
        comment_lookup = {
            889: {"id": 9998, "on_post": 889, "text": binary_comment_text, "created_at": "2024-05-01T00:00:00Z"}
        }

        records = _process_post(post_data, comment_lookup)
        assert len(records) == 1
        rec = records[0]
        assert rec["type"] == "binary"
        per_model = rec["per_model_forecasts"]
        # Binary: values remain as raw strings (the existing contract).
        assert per_model == {"gpt-5.5": "72.0%", "claude-opus-4.7": "68.0%"}

    @pytest.mark.parametrize("flag", [True, False, None])
    def test_was_stacked_carried_verbatim(self, flag: bool | None):
        post = _make_post_data()
        q = _make_binary_q_dict()
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=flag,
            post_data=post,
        )
        assert rec is not None
        assert rec["was_stacked"] is flag

    def test_metaculus_scores_populated_from_my_forecasts_score_data(self):
        post = _make_post_data()
        score_data = {
            "peer_score": 5.2,
            "spot_peer_score": 3.1,
            "baseline_score": 12.0,
            "spot_baseline_score": 10.0,
            "coverage": 0.95,
            "weighted_coverage": 0.88,
            "relative_legacy_score": 0.05,
        }
        q = _make_binary_q_dict(score_data=score_data)
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is not None
        assert rec["metaculus_scores"] == score_data

    def test_metaculus_scores_none_when_score_data_missing(self):
        post = _make_post_data()
        q = _make_binary_q_dict()
        assert "score_data" not in q["my_forecasts"]
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is not None
        assert rec["metaculus_scores"] is None

    def test_category_pulled_from_projects_first_entry(self):
        post = _make_post_data(category_name="Sports")
        q = _make_binary_q_dict()
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is not None
        assert rec["metadata"]["category"] == "Sports"

    def test_category_none_when_projects_empty(self):
        post = {"id": 1, "title": "t", "projects": {"category": []}}
        q = _make_binary_q_dict()
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is not None
        assert rec["metadata"]["category"] is None

    def test_returns_none_when_resolution_missing(self):
        post = _make_post_data()
        q = _make_binary_q_dict(resolution=None)
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is None

    def test_returns_none_when_parse_resolution_flags_skip(self):
        # parse_resolution marks "annulled" and "ambiguous" with should_skip=True.
        post = _make_post_data()
        q = _make_binary_q_dict(resolution="annulled")
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is None


# ---------------------------------------------------------------------------
# per_base_model_forecasts field on collector records
# ---------------------------------------------------------------------------


class TestPerBaseModelForecastsOnRecord:
    """Exercises per_base_model_forecasts field wiring in _process_post.

    When a stacked binary comment contains base-model reasoning blocks with
    probability lines, the collector must populate per_base_model_forecasts
    with the per-base-model values so downstream stacker_detection can
    compute the counterfactual median.
    """

    def test_stacked_binary_populates_per_base_model_forecasts(self):
        base_predictions = [
            ReasonedPrediction(
                prediction_value=0.72,
                reasoning="Model: openrouter/openai/gpt-5.5\n\nAnalysis.\n\nProbability: 72%",
            ),
            ReasonedPrediction(
                prediction_value=0.68,
                reasoning="Model: openrouter/anthropic/claude-opus-4.7\n\nAnalysis.\n\nProbability: 68%",
            ),
            ReasonedPrediction(
                prediction_value=0.75,
                reasoning="Model: openrouter/google/gemini-3.1-pro-preview\n\nAnalysis.\n\nProbability: 75%",
            ),
        ]
        meta_text = "Stacker meta.\n\nProbability: 71%"
        combined = combine_stacker_and_base_reasoning(meta_text, base_predictions)

        comment_text = (
            "# SUMMARY\n"
            "*Question*: Will X?\n\n"
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (stacker-model)*: 71.0%\n\n"
            "### Research Summary\nresearch.\n\n"
            "================================================================================\n"
            "FORECAST SECTION:\n\n"
            f"## R1: Forecaster 1 Reasoning\n{combined}\n"
            "<!-- STACKED=true -->\n"
        )

        post_data = {
            "id": 900,
            "title": "Stacked binary question",
            "projects": {"category": [{"name": "Science"}]},
            "question": _make_binary_q_dict(qid=501, forecast_values=[0.29, 0.71]),
        }
        comment_lookup = {900: {"id": 8888, "on_post": 900, "text": comment_text, "created_at": "2024-05-15T00:00:00Z"}}

        records = _process_post(post_data, comment_lookup)
        assert len(records) == 1
        rec = records[0]
        assert rec["was_stacked"] is True
        # The critical new field:
        per_base = rec["per_base_model_forecasts"]
        assert per_base != {}
        assert per_base["gpt-5.5"] == "72.0%"
        assert per_base["claude-opus-4.7"] == "68.0%"
        assert per_base["gemini-3.1-pro-preview"] == "75.0%"

    def test_non_stacked_binary_has_empty_per_base_model_forecasts(self):
        comment_text = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: 72.0%\n"
            "*Forecaster 2 (claude-opus-4.7)*: 68.0%\n\n"
            "### Research Summary\nresearch.\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "Analysis.\n\nProbability: 72%\n\n"
            "## R1: Forecaster 2 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "Analysis.\n\nProbability: 68%\n"
        )
        post_data = {
            "id": 901,
            "title": "Non-stacked question",
            "projects": {"category": [{"name": "Science"}]},
            "question": _make_binary_q_dict(qid=502),
        }
        comment_lookup = {901: {"id": 8889, "on_post": 901, "text": comment_text, "created_at": "2024-05-15T00:00:00Z"}}

        records = _process_post(post_data, comment_lookup)
        assert len(records) == 1
        rec = records[0]
        assert rec["per_base_model_forecasts"] == {}

    def test_stacked_mc_populates_per_base_model_option_dicts(self):
        base_predictions = [
            ReasonedPrediction(
                prediction_value=0.6,
                reasoning=("Model: openrouter/openai/gpt-5.5\n\nAnalysis:\n- Option A: 60.0%\n- Option B: 40.0%"),
            ),
            ReasonedPrediction(
                prediction_value=0.55,
                reasoning=(
                    "Model: openrouter/anthropic/claude-opus-4.7\n\nAnalysis:\n- Option A: 55.0%\n- Option B: 45.0%"
                ),
            ),
        ]
        meta_text = "Stacker:\n- Option A: 58.0%\n- Option B: 42.0%"
        combined = combine_stacker_and_base_reasoning(meta_text, base_predictions)

        comment_text = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (stacker)*: \n"
            "- Option A: 58.0%\n"
            "- Option B: 42.0%\n\n"
            "### Research Summary\nresearch.\n\n"
            "================================================================================\n"
            "FORECAST SECTION:\n\n"
            f"## R1: Forecaster 1 Reasoning\n{combined}\n"
            "<!-- STACKED=true -->\n"
        )

        post_data = {
            "id": 902,
            "title": "Stacked MC question",
            "projects": {"category": [{"name": "Politics"}]},
            "question": _make_mc_q_dict(
                qid=503,
                options=["Option A", "Option B"],
                forecast_values=[0.58, 0.42],
            ),
        }
        comment_lookup = {902: {"id": 8890, "on_post": 902, "text": comment_text, "created_at": "2024-05-15T00:00:00Z"}}

        records = _process_post(post_data, comment_lookup)
        assert len(records) == 1
        rec = records[0]
        per_base = rec["per_base_model_forecasts"]
        assert per_base["gpt-5.5"] == {"Option A": 0.60, "Option B": 0.40}
        assert per_base["claude-opus-4.7"] == {"Option A": 0.55, "Option B": 0.45}


# ---------------------------------------------------------------------------
# End-to-end integration: producer (main.py) + consumer (parsing.py)
# ---------------------------------------------------------------------------


class TestProducerConsumerRoundTrip:
    """The critical integration check: run a fake stacked pipeline through
    main.py's comment construction, then parse the output back with the
    performance_analysis parser and confirm per-model attributions survive."""

    def test_stacked_binary_roundtrip_recovers_models_and_marker(self):
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question(qid=555)
        bot._pipeline.outcomes[q.id_of_question] = "primary"

        predictions = [
            _make_prediction_with_model(0.72, "openrouter/openai/gpt-5.5", body="gpt analysis"),
            _make_prediction_with_model(0.68, "openrouter/anthropic/claude-opus-4.7", body="claude analysis"),
            _make_prediction_with_model(0.80, "openrouter/google/gemini-3.1-pro-preview", body="gemini analysis"),
        ]
        research = ResearchWithPredictions(
            research_report="raw research",
            summary_report="summary",
            errors=[],
            predictions=predictions,
        )

        # Produce the annotated summary via the annotation wiring.
        parent_summary = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1*: 72.0%\n"
            "*Forecaster 2*: 68.0%\n"
            "*Forecaster 3*: 80.0%\n\n"
            "### Research Summary\nr\n"
        )
        with patch.object(
            ForecastBot,
            "_format_and_expand_research_summary",
            return_value=parent_summary,
        ):
            annotated_summary = TemplateForecaster._format_and_expand_research_summary(
                report_number=1,
                report_type=BinaryReport,
                predicted_research=research,
            )

        # Wrap the annotated summary in a full base explanation so
        # _create_unified_explanation treats it like a real comment.
        base = f"# SUMMARY\n*Question*: ?\n\n{annotated_summary}\n# RESEARCH\nbody\n"
        with patch.object(ForecastBot, "_create_unified_explanation", return_value=base):
            full_comment = bot._create_unified_explanation(q, [research], 0.5, 0.01, 1.0)

        # Producer: marker present, summary annotated.
        assert STACKED_MARKER_TRUE in full_comment
        # Consumer: parse it back.
        assert parse_stacked_marker(full_comment) is True
        per_model = parse_per_model_forecasts(full_comment)
        assert per_model == {
            "gpt-5.5": "72.0%",
            "claude-opus-4.7": "68.0%",
            "gemini-3.1-pro-preview": "80.0%",
        }

    def test_roundtrip_with_stacker_combined_reasoning(self):
        # Real stacking shape: the framework collapses base predictions into ONE
        # aggregated prediction whose reasoning is produced by
        # combine_stacker_and_base_reasoning(). The per-base-model percentiles
        # are only recoverable from the combined reasoning body (the summary
        # bullet shows only the stacker's aggregate).
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question(qid=777)
        bot._pipeline.outcomes[q.id_of_question] = "primary"

        # Base predictions, one per ensemble member, each tagged with Model: by
        # _make_prediction in production (here we build the tag manually).
        base_predictions = [
            ReasonedPrediction(
                prediction_value=0.40,
                reasoning=(
                    "Model: openrouter/openai/gpt-5.5\n\n"
                    "gpt body.\n\n"
                    "Percentile 2.5: 10.0\n"
                    "Percentile 50: 40.0\n"
                    "Percentile 97.5: 80.0\n"
                ),
            ),
            ReasonedPrediction(
                prediction_value=0.50,
                reasoning=(
                    "Model: openrouter/anthropic/claude-opus-4.7\n\n"
                    "claude body.\n\n"
                    "Percentile 2.5: 15.0\n"
                    "Percentile 50: 50.0\n"
                    "Percentile 97.5: 90.0\n"
                ),
            ),
            ReasonedPrediction(
                prediction_value=0.60,
                reasoning=(
                    "Model: openrouter/google/gemini-3.1-pro-preview\n\n"
                    "gemini body.\n\n"
                    "Percentile 2.5: 20.0\n"
                    "Percentile 50: 60.0\n"
                    "Percentile 97.5: 100.0\n"
                ),
            ),
        ]
        meta_text = (
            "Stacker consolidated the three base forecasts.\n\n"
            "Percentile 2.5: 15.0\n"
            "Percentile 50: 50.0\n"
            "Percentile 97.5: 90.0\n"
        )
        combined = combine_stacker_and_base_reasoning(meta_text, base_predictions)
        aggregated = ReasonedPrediction(prediction_value=0.50, reasoning=combined)

        # Production: ResearchWithPredictions has length 1 for stacked questions.
        research = ResearchWithPredictions(
            research_report="raw research",
            summary_report="summary",
            errors=[],
            predictions=[aggregated],
        )
        assert len(research.predictions) == 1

        # Summary bullet shows only the stacker's aggregate (not per-base-model).
        # _format_and_expand_research_summary wraps a parent-returned summary
        # and tries to annotate based on per-prediction Model: prefixes; the
        # aggregated prediction starts with "## Stacker Meta-Analysis", no
        # Model: prefix, so Forecaster 1 stays unannotated.
        parent_summary = "## Report 1 Summary\n### Forecasts\n*Forecaster 1*: 50.0%\n\n### Research Summary\nr\n"
        with patch.object(
            ForecastBot,
            "_format_and_expand_research_summary",
            return_value=parent_summary,
        ):
            annotated_summary = TemplateForecaster._format_and_expand_research_summary(
                report_number=1,
                report_type=BinaryReport,
                predicted_research=research,
            )
        # No Model: prefix on stacker meta → no annotation; bullet left alone.
        assert "*Forecaster 1*: 50.0%" in annotated_summary

        # Build the base unified comment as the framework would: summary +
        # rationales section with a single R1 block wrapping the combined body.
        base_unified = (
            "# SUMMARY\n*Question*: ?\n\n"
            f"{annotated_summary}\n"
            "================================================================================\n"
            "FORECAST SECTION:\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            f"{combined}\n"
        )
        with patch.object(ForecastBot, "_create_unified_explanation", return_value=base_unified):
            # Prevent accidental use of aggregated_prediction_value by passing a
            # placeholder; the bot's method just concatenates the marker.
            full_comment = bot._create_unified_explanation(
                q,
                [research],
                0.5,
                0.01,
                1.0,
            )

        # Producer: stacked marker present.
        assert STACKED_MARKER_TRUE in full_comment
        assert parse_stacked_marker(full_comment) is True

        # Consumer 1: per_model_forecasts reflects ONLY the stacker's bullet
        # (no per-base attribution in the summary).
        forecasts = parse_per_model_forecasts(full_comment)
        assert forecasts == {"Forecaster 1": "50.0%"}

        # Consumer 2: percentiles recover the 3 base models via the
        # stacker-combined-body handler that splits on the delimiter.
        percentiles = parse_per_model_numeric_percentiles(full_comment)
        # Expect at minimum the 3 base models keyed by display name.
        assert "gpt-5.5" in percentiles
        assert "claude-opus-4.7" in percentiles
        assert "gemini-3.1-pro-preview" in percentiles
        assert percentiles["gpt-5.5"] == [(2.5, 10.0), (50.0, 40.0), (97.5, 80.0)]
        assert percentiles["claude-opus-4.7"] == [(2.5, 15.0), (50.0, 50.0), (97.5, 90.0)]
        assert percentiles["gemini-3.1-pro-preview"] == [
            (2.5, 20.0),
            (50.0, 60.0),
            (97.5, 100.0),
        ]

        # Consumer 3: reasoning text recovers 4 entries — stacker + 3 base.
        # The stacker entry is keyed anonymously as "Forecaster 1" since the
        # combined body begins with "## Stacker Meta-Analysis" (no Model: line
        # at the top of the R1 section for the stacker).
        reasoning = parse_per_model_reasoning_text(full_comment)
        assert len(reasoning) == 4
        assert "gpt-5.5" in reasoning
        assert "claude-opus-4.7" in reasoning
        assert "gemini-3.1-pro-preview" in reasoning
        # Stacker key — falls back since stacker's meta has no Model: prefix.
        assert "Forecaster 1" in reasoning
        assert "Stacker consolidated" in reasoning["Forecaster 1"]
        assert "gpt body." in reasoning["gpt-5.5"]
        assert "claude body." in reasoning["claude-opus-4.7"]
        assert "gemini body." in reasoning["gemini-3.1-pro-preview"]


# ---------------------------------------------------------------------------
# Regression: oversized comments must construct report objects without raising
# the framework's "Explanation must start with a '#'" ValidationError.
# (2026-06-05 crash: binary Q578 + MC Q20683 forecast fine but crashed on report
# construction because the trimmed explanation lost its leading "#".)
# ---------------------------------------------------------------------------


class TestOversizedCommentReportConstruction:
    """The trimmed explanation must always satisfy the framework validator.

    forecast_report.py validate_explanation_starts_with_hash rejects any
    explanation not starting with '#'. We build an oversized 6-model unified
    explanation (research bloated past COMMENT_CHAR_LIMIT) through the real
    build_unified_explanation path, then construct the concrete report types to
    prove no ValidationError fires — the exact 2026-06-05 crash on Q578/Q20683.
    """

    def _oversized_explanation(self, question: MetaculusQuestion) -> str:
        summary = "\n".join(f"*Forecaster {i}*: {60 + i}.0%" for i in range(1, 7))
        rationales = "\n".join(
            f"## R1: Forecaster {i} Reasoning\nModel: openrouter/provider/m{i}\nrationale body {i}" for i in range(1, 7)
        )
        base = (
            "# SUMMARY\n*Question*: will X?\n\n"
            "## Report 1 Summary\n### Forecasts\n"
            f"{summary}\n\n"
            "### Research Summary\n_Full research in the RESEARCH section below._\n\n"
            "# RESEARCH\n## Report 1 Research\n" + ("research_token " * 30_000) + "\n\n# FORECASTS\n" + rationales
        )
        out = build_unified_explanation(base, question, AggregationStrategy.CONDITIONAL_STACKING, "primary")
        assert len(out) <= COMMENT_CHAR_LIMIT
        assert out.lstrip().startswith("#")
        return out

    def test_binary_report_constructs_from_oversized_comment(self) -> None:
        question = BinaryQuestion(
            question_text="Will it happen?", background_info="bg", resolution_criteria="rc", fine_print=""
        )
        explanation = self._oversized_explanation(question)
        # Must not raise ValidationError on the explanation field.
        report = BinaryReport(question=question, explanation=explanation, prediction=0.73)
        assert report.explanation.lstrip().startswith("#")

    def test_mc_report_constructs_from_oversized_comment(self) -> None:
        question = MultipleChoiceQuestion(
            question_text="Which option?",
            background_info="bg",
            resolution_criteria="rc",
            fine_print="",
            options=["Yes", "No"],
        )
        explanation = self._oversized_explanation(question)
        prediction = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Yes", probability=0.6),
                PredictedOption(option_name="No", probability=0.4),
            ]
        )
        report = MultipleChoiceReport(question=question, explanation=explanation, prediction=prediction)
        assert report.explanation.lstrip().startswith("#")


class TestResearchNotDuplicatedInComment:
    """Guard the forecaster.py dedup fix: research must appear ONCE in the
    published comment (under # RESEARCH), not twice (also under ### Research
    Summary). Reverting summary_report back to the full corpus would double the
    comment size and reintroduce the trim pressure that caused the crash.

    We drive the framework's real _create_unified_explanation with a stub
    summary_report (what forecaster.py now sets) + full research_report, and
    assert the research sentinel appears exactly once.
    """

    def test_research_appears_once_when_summary_report_is_stub(self) -> None:
        bot = _make_bot(AggregationStrategy.MEAN)
        q = BinaryQuestion(
            question_text="Will it happen?", background_info="bg", resolution_criteria="rc", fine_print=""
        )
        research_sentinel = "UNIQUE_RESEARCH_SENTINEL_42"
        collection = ResearchWithPredictions(
            research_report=f"## Web Research\n{research_sentinel} body text",
            summary_report="_Full research in the RESEARCH section below._",
            errors=[],
            predictions=[ReasonedPrediction(prediction_value=0.6, reasoning="Model: x\nreasoning")],
        )
        explanation = bot._create_unified_explanation(q, [collection], 0.6, 0.01, 1.0)
        assert explanation.count(research_sentinel) == 1, "research must not be duplicated into the summary section"
        # The stub pointer rides in the summary section; research lives under # RESEARCH.
        assert "_Full research in the RESEARCH section below._" in explanation
        assert "### Research Summary" in explanation


class TestOfflineAssemblySmoke:
    """Full _create_unified_explanation path with a 6-model stacked collection —
    mirrors what a real test_questions run produces for Q578/Q20683, but offline
    (no API spend). Two cases: a healthy comment (research present once, real
    headings, under limit) and an oversized one (validator invariant + markers +
    report construction hold even when the trim sacrifices research).
    """

    def _bot_and_question(self):
        # Roster size matches the six-model collection below, so the ensemble-size
        # marker this path publishes is the one a real six-model run would carry.
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING, n_forecasters=6)
        q = BinaryQuestion(
            question_text="Will it happen?", background_info="bg", resolution_criteria="rc", fine_print=""
        )
        q.id_of_question = 578  # _create_unified_explanation reads the outcome by qid
        bot._pipeline.outcomes[q.id_of_question] = "primary"
        return bot, q

    def _research_report(self, sentinel: str, *, token_reps: int) -> str:
        # Mirror the orchestrator's output: h2 provider headers with in-body
        # headings already demoted to >= h3 (by _demote_inner_headings). The
        # framework's report_sections_to_markdown renormalizes cleanly when the
        # first section is the minimum level, so no [Hashtag] fallback fires.
        provider_a = (
            "## News Articles (AskNews)\n"
            "### Historical Context\n"
            + (f"{sentinel} " + "asknews_token " * token_reps)
            + "\n### Recent Developments\nmore"
        )
        provider_b = "## Web Research (Native Search)\n### Findings\n" + ("native_token " * token_reps)
        return f"{provider_a}\n\n---\n\n{provider_b}"

    def _predictions(self, *, token_reps: int):
        return [
            ReasonedPrediction(
                prediction_value=0.60 + i * 0.01,
                reasoning=f"Model: openrouter/provider/model-{i}\n" + (f"rationale {i} " * token_reps),
            )
            for i in range(1, 7)
        ]

    def test_healthy_6model_comment_assembles_with_research_and_real_headings(self) -> None:
        bot, q = self._bot_and_question()
        sentinel = "RESEARCH_ONCE_SENTINEL"
        collection = ResearchWithPredictions(
            research_report=self._research_report(sentinel, token_reps=300),
            summary_report="_Full research in the RESEARCH section below._",
            errors=[],
            predictions=self._predictions(token_reps=300),
        )

        explanation = bot._create_unified_explanation(q, [collection], 0.65, 1.23, 4.5)

        assert explanation.lstrip().startswith("#")
        assert len(explanation) <= COMMENT_CHAR_LIMIT
        # Healthy comment: research carried exactly once (the dedup fix), and the
        # provider's in-body headings render as real markdown (no degrade).
        assert explanation.count(sentinel) == 1
        assert "[Hashtag]" not in explanation
        assert "### Historical Context" in explanation
        # Residual-analysis markers present.
        assert "<!-- STACKER_OUTCOME=primary -->" in explanation
        assert "<!-- STACKED=true -->" in explanation
        assert "<!-- FORECASTERS_USED=6/6 -->" in explanation

    def test_oversized_6model_comment_holds_invariant_and_constructs(self) -> None:
        bot, q = self._bot_and_question()
        # Large rationales force the comment well past the limit; research is
        # sacrificed first by design, so we don't assert it survives — only that
        # the invariant, markers, and report construction hold.
        collection = ResearchWithPredictions(
            research_report=self._research_report("dropped", token_reps=4_000),
            summary_report="_Full research in the RESEARCH section below._",
            errors=[],
            predictions=self._predictions(token_reps=4_000),
        )

        explanation = bot._create_unified_explanation(q, [collection], 0.65, 1.23, 4.5)

        assert explanation.lstrip().startswith("#"), "validator invariant — the crash contract"
        assert len(explanation) <= COMMENT_CHAR_LIMIT
        assert "[Hashtag]" not in explanation
        assert "<!-- STACKER_OUTCOME=primary -->" in explanation
        assert "<!-- STACKED=true -->" in explanation
        # Trim survival on a REALISTIC oversized comment — sections the trimmer
        # recognizes, so the research-first strategy fires rather than the
        # last-resort header-and-tail one that the synthetic filler test in
        # test_comment_formatting.py exercises. A degraded publish is most likely on
        # a long comment, so this is where losing the disclosure would hurt most.
        assert "<!-- FORECASTERS_USED=6/6 -->" in explanation
        # The exact 2026-06-05 crash: report construction must not raise.
        report = BinaryReport(question=q, explanation=explanation, prediction=0.65)
        assert report.explanation.lstrip().startswith("#")

    def test_research_above_section_limit_survives_when_complete_comment_fits(self) -> None:
        bot = _make_bot(AggregationStrategy.MEAN, n_forecasters=1)
        q = BinaryQuestion(
            question_text="Will it happen?", background_info="bg", resolution_criteria="rc", fine_print=""
        )
        middle_marker = "FITTING_RESEARCH_MIDDLE_SENTINEL"
        research = (
            "## Full Research\n"
            + ("research detail " * 900)
            + middle_marker
            + (" research detail" * 3_500)
            + "\nFITTING_RESEARCH_TAIL_SENTINEL"
        )
        collection = ResearchWithPredictions(
            research_report=research,
            summary_report="_Full research in the RESEARCH section below._",
            errors=[],
            predictions=[
                ReasonedPrediction(
                    prediction_value=0.65,
                    reasoning=(
                        "Model: openrouter/provider/model-1\n\n"
                        "A concise rationale.\n\n"
                        '```json\n{"probability": 0.65}\n```'
                    ),
                )
            ],
        )

        explanation = bot._create_unified_explanation(q, [collection], 0.65, 0.01, 1.0)

        assert len(explanation) < COMMENT_CHAR_LIMIT
        assert middle_marker in explanation
        assert "FITTING_RESEARCH_TAIL_SENTINEL" in explanation
        assert "[... trimmed for length]" not in explanation

    def test_multi_collection_overflow_uses_rationale_aware_budget_before_framework_guard(self) -> None:
        bot = _make_bot(AggregationStrategy.MEAN, n_forecasters=2, research_reports=2)
        q = BinaryQuestion(
            question_text="Will it happen?", background_info="bg", resolution_criteria="rc", fine_print=""
        )
        collections = [
            ResearchWithPredictions(
                research_report=f"## Research Collection {index}\n" + (f"research_{index} " * 5_000),
                summary_report="_Full research in the RESEARCH section below._",
                errors=[],
                predictions=[
                    ReasonedPrediction(
                        prediction_value=0.6 + index / 10,
                        reasoning=(
                            f"Model: openrouter/provider/model-{index}\n\n"
                            + (f"rationale_{index} " * 9_000)
                            + f'\n\n```json\n{{"forecast": {index}}}\n```'
                        ),
                    )
                ],
            )
            for index in (1, 2)
        ]

        source_comment = bot._create_comment(q, collections, 0.65, 0.01, 1.0)
        assert len(source_comment) > 150_000

        explanation = bot._create_unified_explanation(q, collections, 0.65, 0.01, 1.0)

        assert len(explanation) <= COMMENT_CHAR_LIMIT
        assert "# FORECASTS" in explanation
        assert "Model: openrouter/provider/model-1" in explanation
        assert "Model: openrouter/provider/model-2" in explanation
        assert explanation.count("```json") == 2
        # The collector intentionally parses report 1 only; both report blocks remain in the
        # published comment even though report 2 is outside that parser's current contract.
        assert set(parse_per_model_reasoning_text(explanation)) == {"model-1"}
        assert "<!-- FORECASTERS_USED=2/2 -->" in explanation
        report = BinaryReport(question=q, explanation=explanation, prediction=0.65)
        assert report.prediction == 0.65

    def test_multi_collection_overflow_preserves_each_pending_lead_and_diagnostics_block(self) -> None:
        bot = _make_bot(AggregationStrategy.MEAN, n_forecasters=2, research_reports=2)
        q = BinaryQuestion(
            question_text="Will it happen?", background_info="bg", resolution_criteria="rc", fine_print=""
        )
        collections: list[ResearchWithPredictions] = []
        pending_leads = [
            "Verify whether the first collection's source changed.",
            "Check whether the second collection's source has a newer update.",
        ]
        lost_source_tokens = [
            "lost=briefing_1:error(timeout)",
            "lost=briefing_2:error(timeout)",
        ]

        for index, pending_lead in enumerate(pending_leads, start=1):
            diagnostics = format_provider_diagnostics_block(
                [
                    ProviderResult(
                        name=f"collection_{index}",
                        status="fallback",
                        chars=1_000,
                        latency_ms=250,
                        details={"sources": {f"briefing_{index}": "error(timeout)"}},
                    )
                ]
            )
            collections.append(
                ResearchWithPredictions(
                    research_report=(
                        f"Optional evidence from collection {index}. " * 5_000
                        + f"\n\n    Indented source note from collection {index}."
                        + f"\n\n```text\n    Code block line from collection {index}.\n```"
                        + "\n\n"
                        + render_findings([], [pending_lead])
                        + "\n\n"
                        + diagnostics
                    ),
                    summary_report=f"Summary from collection {index}.",
                    errors=[],
                    predictions=[
                        ReasonedPrediction(
                            prediction_value=0.6 + index / 10,
                            reasoning=(
                                f"Model: openrouter/provider/model-{index}\n\n"
                                + (f"Rationale from collection {index}. " * 20)
                                + f'\n\n```json\n{{"forecast": {index}}}\n```'
                            ),
                        )
                    ],
                )
            )

        source_comment = bot._create_comment(q, collections, 0.65, 0.01, 1.0)
        assert len(source_comment) > 150_000

        explanation = bot._create_unified_explanation(q, collections, 0.65, 0.01, 1.0)

        assert len(explanation) <= COMMENT_CHAR_LIMIT
        assert "Verify whether the first collection's source changed." in explanation
        assert "Check whether the second collection's source has a newer update." in explanation
        assert all(token in explanation for token in lost_source_tokens)
        assert EXPLANATION_DETAIL_UNKNOWN_MARKER not in explanation
        assert "Model: openrouter/provider/model-1" in explanation
        assert "Model: openrouter/provider/model-2" in explanation

    def test_stacked_combined_overflow_keeps_base_model_attribution(self) -> None:
        bot = _make_bot(AggregationStrategy.STACKING, n_forecasters=3)
        q = BinaryQuestion(
            question_text="Will it happen?", background_info="bg", resolution_criteria="rc", fine_print=""
        )
        q.id_of_question = 779
        bot._pipeline.outcomes[q.id_of_question] = "primary"
        base_predictions = [
            ReasonedPrediction(
                prediction_value=value,
                reasoning=(
                    f"Model: openrouter/provider/stacked-{index}\n\n"
                    + (f"stacked_rationale_{index} " * 5_000)
                    + f'\n\n```json\n{{"forecast": {value}}}\n```'
                ),
            )
            for index, value in enumerate((0.4, 0.5, 0.6), start=1)
        ]
        combined = combine_stacker_and_base_reasoning("Stacker summary.\n\nPercentile 50: 0.5\n", base_predictions)
        collection = ResearchWithPredictions(
            research_report="short research",
            summary_report="summary",
            errors=[],
            predictions=[ReasonedPrediction(prediction_value=0.5, reasoning=combined)],
        )
        source_comment = bot._create_comment(q, [collection], 0.5, 0.01, 1.0)
        assert len(source_comment) > 150_000

        explanation = bot._create_unified_explanation(q, [collection], 0.5, 0.01, 1.0)

        assert len(explanation) <= COMMENT_CHAR_LIMIT
        parsed_reasoning = parse_per_model_reasoning_text(explanation)
        assert {"stacked-1", "stacked-2", "stacked-3"}.issubset(parsed_reasoning)
        assert explanation.count("```json") == 3
        assert "<!-- STACKED=true -->" in explanation

    def test_unfittable_rationale_marks_detail_unknown_and_keeps_prediction(self) -> None:
        bot = _make_bot(AggregationStrategy.MEAN, n_forecasters=1)
        q = BinaryQuestion(
            question_text="Will it happen?", background_info="bg", resolution_criteria="rc", fine_print=""
        )
        original_reasoning = (
            "Model: openrouter/provider/oversized-json\n\n"
            "A rationale whose structured output exceeds the complete comment budget.\n\n"
            + '```json\n{"evidence": "'
            + ("x" * COMMENT_CHAR_LIMIT)
            + '"}\n```'
        )
        collection = ResearchWithPredictions(
            research_report="Complete research evidence remains in the collection.",
            summary_report="Full research is in the research section.",
            errors=[],
            predictions=[ReasonedPrediction(prediction_value=0.65, reasoning=original_reasoning)],
        )

        explanation = bot._create_unified_explanation(q, [collection], 0.65, 0.01, 1.0)
        assert len(explanation) <= COMMENT_CHAR_LIMIT
        assert EXPLANATION_DETAIL_UNKNOWN_MARKER in explanation
        assert "missing model attribution and reasoning are unknown" in explanation
        assert parse_explanation_detail_status(explanation) == "unknown"
        assert parse_per_model_reasoning_text(explanation) == {}
        assert "*Final Prediction*: 65.0%" in explanation
        assert collection.predictions[0].reasoning == original_reasoning
        assert collection.research_report == "Complete research evidence remains in the collection."
        report = BinaryReport(question=q, explanation=explanation, prediction=0.65)
        assert report.prediction == 0.65

    def test_numeric_report_fallback_keeps_multiline_public_forecast(self) -> None:
        bot = _make_bot(AggregationStrategy.MEAN, n_forecasters=1)
        q = NumericQuestion(
            question_text="What will the value be?",
            background_info="background",
            resolution_criteria="criteria",
            fine_print="",
            lower_bound=0,
            upper_bound=100,
            open_lower_bound=False,
            open_upper_bound=False,
            cdf_size=3,
        )
        prediction = NumericDistribution.from_question(
            [
                Percentile(percentile=0.1, value=10),
                Percentile(percentile=0.5, value=50),
                Percentile(percentile=0.9, value=90),
            ],
            q,
            standardize_cdf=False,
        )
        original_reasoning = (
            "Model: openrouter/provider/oversized-numeric-json\n\n"
            + '```json\n{"evidence": "'
            + ("x" * COMMENT_CHAR_LIMIT)
            + '"}\n```'
        )
        collection = ResearchWithPredictions(
            research_report="Numeric evidence remains in the research report.",
            summary_report="Numeric evidence summary.",
            errors=[],
            predictions=[ReasonedPrediction(prediction_value=prediction, reasoning=original_reasoning)],
        )

        explanation = bot._create_unified_explanation(q, [collection], prediction, 0.01, 1.0)
        readable_prediction = DataOrganizer.get_readable_prediction(prediction)
        report = NumericReport(question=q, explanation=explanation, prediction=prediction)

        assert len(explanation) <= COMMENT_CHAR_LIMIT
        assert EXPLANATION_DETAIL_UNKNOWN_MARKER in explanation
        assert parse_explanation_detail_status(explanation) == "unknown"
        assert "missing model attribution and reasoning are unknown" in explanation
        assert parse_per_model_reasoning_text(explanation) == {}
        assert readable_prediction in explanation
        assert "10.00% chance of value below 10.0" in explanation
        assert "50.00% chance of value below 50.0" in explanation
        assert "90.00% chance of value below 90.0" in explanation
        assert report.prediction == prediction
        assert len(report.prediction.declared_percentiles) == 3
        assert report.prediction.declared_percentiles == prediction.declared_percentiles
        assert "<!-- FORECASTERS_USED=1/1 -->" in explanation

    def test_multiple_choice_fallback_keeps_long_multiline_labels_when_they_fit(self) -> None:
        bot = _make_bot(AggregationStrategy.MEAN, n_forecasters=1)
        option_names = ["A detailed outcome label " * 8, "An alternative outcome label " * 8]
        q = MultipleChoiceQuestion(
            question_text="Which outcome?",
            background_info="background",
            resolution_criteria="criteria",
            fine_print="",
            options=option_names,
        )
        prediction = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=option_names[0], probability=0.6),
                PredictedOption(option_name=option_names[1], probability=0.4),
            ]
        )
        original_reasoning = (
            "Model: openrouter/provider/oversized-mc-json\n\n"
            + '```json\n{"evidence": "'
            + ("x" * COMMENT_CHAR_LIMIT)
            + '"}\n```'
        )
        collection = ResearchWithPredictions(
            research_report="Multiple-choice evidence remains in the research report.",
            summary_report="Multiple-choice evidence summary.",
            errors=[],
            predictions=[ReasonedPrediction(prediction_value=prediction, reasoning=original_reasoning)],
        )

        explanation = bot._create_unified_explanation(q, [collection], prediction, 0.01, 1.0)
        readable_prediction = DataOrganizer.get_readable_prediction(prediction)
        report = MultipleChoiceReport(question=q, explanation=explanation, prediction=prediction)

        assert len(explanation) <= COMMENT_CHAR_LIMIT
        assert EXPLANATION_DETAIL_UNKNOWN_MARKER in explanation
        assert parse_explanation_detail_status(explanation) == "unknown"
        assert parse_per_model_reasoning_text(explanation) == {}
        assert readable_prediction.strip() in explanation
        assert f"- {option_names[0]}: 60.0%" in explanation
        assert f"- {option_names[1]}: 40.0%" in explanation
        assert "<!-- FORECASTERS_USED=1/1 -->" in explanation
        assert [option.option_name for option in report.prediction.predicted_options] == option_names
        assert [option.probability for option in report.prediction.predicted_options] == [0.6, 0.4]

    def test_supported_large_multiple_choice_report_uses_bounded_forecast_reference(self) -> None:
        bot = _make_bot(AggregationStrategy.MEAN, n_forecasters=1)
        option_names = [
            f"Option {index:04d}: " + ("x" * (200 - len(f"Option {index:04d}: "))) for index in range(1_000)
        ]
        q = MultipleChoiceQuestion(
            question_text="Which outcome?",
            background_info="background",
            resolution_criteria="criteria",
            fine_print="",
            options=option_names,
        )
        prediction = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=option_name, probability=0.001) for option_name in option_names
            ]
        )
        collection = ResearchWithPredictions(
            research_report="Multiple-choice evidence remains in the research report.",
            summary_report="Multiple-choice evidence summary.",
            errors=[],
            predictions=[
                ReasonedPrediction(
                    prediction_value=prediction,
                    reasoning='Model: openrouter/provider/large-mc-json\n\n```json\n{"options": 1000}\n```',
                )
            ],
        )

        explanation = bot._create_unified_explanation(q, [collection], prediction, 0.01, 1.0)
        report = MultipleChoiceReport(question=q, explanation=explanation, prediction=prediction)

        assert len(explanation) <= COMMENT_CHAR_LIMIT
        assert "*Final Prediction*: the full forecast remains in the published report." in explanation
        assert EXPLANATION_DETAIL_UNKNOWN_MARKER in explanation
        assert parse_explanation_detail_status(explanation) == "unknown"
        assert parse_per_model_reasoning_text(explanation) == {}
        assert "<!-- FORECASTERS_USED=1/1 -->" in explanation
        assert all(
            option.option_name == expected
            for option, expected in zip(report.prediction.predicted_options, option_names, strict=True)
        )
        assert all(option.probability == pytest.approx(0.001) for option in report.prediction.predicted_options)
        assert collection.research_report == "Multiple-choice evidence remains in the research report."

    def test_incomplete_fallback_keeps_disclosures_and_prediction(self) -> None:
        bot = _make_bot(AggregationStrategy.MEAN, n_forecasters=1)
        q = BinaryQuestion(
            question_text="Will it happen?", background_info="bg", resolution_criteria="rc", fine_print=""
        )
        correction_claim = "The briefing was materially wrong: " + ("x" * COMMENT_CHAR_LIMIT)
        unresolved_source_check = "Confirm the current source still reports the same value."
        correction = render_findings(
            [
                Finding(
                    claim=correction_claim,
                    source_url="https://example.invalid/report",
                    quote="The primary source reports a different value.",
                    date="2026-10-01",
                    retrieved_how="read_document",
                    topic="labor",
                    discrepancy=True,
                    verification_tier="snippet",
                )
            ],
            [unresolved_source_check],
        )
        diagnostics = format_provider_diagnostics_block(
            [
                ProviderResult(
                    name="asknews",
                    status="fallback",
                    chars=1_000,
                    latency_ms=250,
                    fallback_provider="openrouter",
                    details={"sources": {"articles": "ok(2)", "summarizer": "error(timeout)"}},
                )
            ]
        )
        research = "\n\n".join([SUMMARIZER_SOFT_FAIL_BANNER, correction, diagnostics])
        original_reasoning = (
            "Model: openrouter/provider/oversized-json\n\n"
            + '```json\n{"evidence": "'
            + ("x" * COMMENT_CHAR_LIMIT)
            + '"}\n```'
        )
        collection = ResearchWithPredictions(
            research_report=research,
            summary_report="Full research remains in the research section.",
            errors=[],
            predictions=[ReasonedPrediction(prediction_value=0.65, reasoning=original_reasoning)],
        )

        explanation = bot._create_unified_explanation(q, [collection], 0.65, 0.01, 1.0)
        report = BinaryReport(question=q, explanation=explanation, prediction=0.65)

        assert len(explanation) <= COMMENT_CHAR_LIMIT
        assert report.prediction == 0.65
        assert "*Final Prediction*: 65.0%" in explanation
        assert EXPLANATION_DETAIL_UNKNOWN_MARKER in explanation
        assert parse_explanation_detail_status(explanation) == "unknown"
        assert "missing model attribution and reasoning are unknown" in explanation
        assert parse_per_model_reasoning_text(explanation) == {}
        assert SUMMARIZER_SOFT_FAIL_BANNER in explanation
        assert "Driver-reported corrections (limited evidence)" in explanation
        assert (
            "These claims rely partly on snippets or generated answers; recheck the underlying sources." in explanation
        )
        assert "[Correction details omitted for length.]" in explanation
        assert "Pending leads:" in explanation
        assert f"- {unresolved_source_check}" in explanation
        assert "## Provider Diagnostics" in explanation
        assert "lost=summarizer:error(timeout)" in explanation
        assert "FORECASTERS_USED=1/1" in explanation
        assert correction_claim not in explanation
        assert collection.research_report == research

    def test_incomplete_fallback_marks_pending_lead_details_omitted_when_too_long(self) -> None:
        bot = _make_bot(AggregationStrategy.MEAN, n_forecasters=1)
        q = BinaryQuestion(
            question_text="Will it happen?", background_info="bg", resolution_criteria="rc", fine_print=""
        )
        unresolved_source_check = "Confirm the current source still reports the same value: " + (
            "x" * COMMENT_CHAR_LIMIT
        )
        research = render_findings([], [unresolved_source_check])
        collection = ResearchWithPredictions(
            research_report=research,
            summary_report="Current research summary.",
            errors=[],
            predictions=[
                ReasonedPrediction(
                    prediction_value=0.65,
                    reasoning='Model: openrouter/provider/model-1\n\n```json\n{"probability": 0.65}\n```',
                )
            ],
        )

        explanation = bot._create_unified_explanation(q, [collection], 0.65, 0.01, 1.0)
        report = BinaryReport(question=q, explanation=explanation, prediction=0.65)

        assert len(explanation) <= COMMENT_CHAR_LIMIT
        assert report.prediction == 0.65
        assert "*Final Prediction*: 65.0%" in explanation
        assert "Pending leads:" in explanation
        assert "[Pending lead details omitted for length.]" in explanation
        assert unresolved_source_check not in explanation
        assert EXPLANATION_DETAIL_UNKNOWN_MARKER in explanation
        assert parse_explanation_detail_status(explanation) == "unknown"
        assert collection.research_report == research

    def test_overflow_preserves_corrections_soft_fail_and_provider_diagnostics(self) -> None:
        bot = _make_bot(AggregationStrategy.MEAN, n_forecasters=1)
        q = BinaryQuestion(
            question_text="Will it happen?", background_info="bg", resolution_criteria="rc", fine_print=""
        )
        correction_claim = "The briefing says 7%; the cited source reports 5.2%."
        unresolved_source_check = "Confirm the current source still reports the same value."
        corrections = render_findings(
            [
                Finding(
                    claim=correction_claim,
                    source_url="https://example.invalid/report",
                    quote="The unemployment rate was 5.2%.",
                    date="2026-10-01",
                    retrieved_how="read_document",
                    topic="labor",
                    discrepancy=True,
                    verification_tier="snippet",
                )
            ],
            [unresolved_source_check],
        )
        diagnostics = format_provider_diagnostics_block(
            [
                ProviderResult(
                    name="asknews",
                    status="fallback",
                    chars=1_000,
                    latency_ms=250,
                    fallback_provider="openrouter",
                    details={"sources": {"articles": "ok(2)", "summarizer": "error(timeout)"}},
                )
            ]
        )
        research = "\n\n".join(
            [
                "## News Articles (AskNews)",
                SUMMARIZER_SOFT_FAIL_BANNER,
                "Two article summaries remain available.",
                corrections,
                diagnostics,
            ]
        )
        collection = ResearchWithPredictions(
            research_report=research,
            summary_report="Full research remains in the research section.",
            errors=[],
            predictions=[
                ReasonedPrediction(
                    prediction_value=0.65,
                    reasoning=(
                        "Model: openrouter/provider/model-1\n\n"
                        + ("Long rationale detail. " * 15_000)
                        + '\n\n```json\n{"probability": 0.65}\n```'
                    ),
                )
            ],
        )

        explanation = bot._create_unified_explanation(q, [collection], 0.65, 0.01, 1.0)

        assert len(explanation) <= COMMENT_CHAR_LIMIT
        assert SUMMARIZER_SOFT_FAIL_BANNER in explanation
        assert correction_claim in explanation
        assert (
            "These claims rely partly on snippets or generated answers; recheck the underlying sources." in explanation
        )
        assert "Pending leads:" in explanation
        assert f"- {unresolved_source_check}" in explanation
        assert "lost=summarizer:error(timeout)" in explanation
        assert "Model: openrouter/provider/model-1" in explanation
        assert '{"probability": 0.65}' in explanation
        assert collection.research_report == research

    @pytest.mark.asyncio
    async def test_forecast_questions_aggregates_and_constructs_overflow_report(self) -> None:
        bot = _make_bot(AggregationStrategy.MEAN, n_forecasters=2)
        q = BinaryQuestion(
            question_text="Will it happen?", background_info="bg", resolution_criteria="rc", fine_print=""
        )
        q.id_of_question = 780
        research = "## Complete research\n" + ("research evidence " * 3_000)
        reasoned_predictions = [
            ReasonedPrediction(
                prediction_value=value,
                reasoning=(
                    f"Model: openrouter/provider/model-{index}\n\n"
                    + (f"rationale_{index} " * 8_000)
                    + f'\n\n```json\n{{"probability": {value}}}\n```'
                ),
            )
            for index, value in enumerate((0.4, 0.8), start=1)
        ]
        delivered_research: list[str] = []
        prediction_index = 0

        async def return_prediction(question, research_text, llm, *, qid, chart_b64=None):
            nonlocal prediction_index
            delivered_research.append(research_text)
            prediction = reasoned_predictions[prediction_index]
            prediction_index += 1
            return prediction

        with (
            patch.object(bot, "run_research", new=AsyncMock(return_value=research)),
            patch.object(bot, "_forecaster_with_soft_deadline", new=AsyncMock(side_effect=return_prediction)),
        ):
            reports = await bot.forecast_questions([q])

        assert len(reports) == 1
        report = reports[0]
        assert isinstance(report, BinaryReport)
        assert report.prediction == pytest.approx(0.6)
        assert len(report.explanation) <= COMMENT_CHAR_LIMIT
        assert len(delivered_research) == 2
        assert all(delivered == research for delivered in delivered_research)
        assert "Model: openrouter/provider/model-1" in report.explanation
        assert "Model: openrouter/provider/model-2" in report.explanation
        assert report.explanation.count("```json") == 2
        assert "<!-- FORECASTERS_USED=2/2 -->" in report.explanation
