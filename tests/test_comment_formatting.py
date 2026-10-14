"""Comment section formatting and marker preservation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.comment.formatting import (
    build_unified_explanation,
    format_research_summary_with_models,
)
from metaculus_bot.comment.markers import (
    FORECASTERS_USED_MARKER_RE,
    STACKED_MARKER_FALSE,
    STACKED_MARKER_TRUE,
    STACKER_OUTCOME_FALLBACK_LLM,
    STACKER_OUTCOME_FALLBACK_MEAN,
    STACKER_OUTCOME_FALLBACK_MEDIAN,
    STACKER_OUTCOME_PRIMARY,
    STACKER_OUTCOME_SKIPPED,
    STACKER_OUTCOME_SKIPPED_CONFIG_OFF,
    STACKER_SKIP_REASONS,
    TOOLS_USED_MARKER_FALSE,
    TOOLS_USED_MARKER_TRUE,
)
from metaculus_bot.constants import (
    COMMENT_CHAR_LIMIT,
    SUMMARY_SECTION_CHAR_LIMIT,
)


class TestFormatResearchSummaryWithModels:
    def test_injects_model_names_into_bullets(self):

        base_text = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1*: 72.0%\n"
            "*Forecaster 2*: 68.0%\n"
            "### Research Summary\nSome research.\n"
        )
        predictions = [
            MagicMock(reasoning="Model: openrouter/openai/gpt-5.5\n\nanalysis"),
            MagicMock(reasoning="Model: openrouter/anthropic/claude-opus-4.7\n\nanalysis"),
        ]
        result = format_research_summary_with_models(base_text, predictions)
        assert "*Forecaster 1 (gpt-5.5)*: 72.0%" in result
        assert "*Forecaster 2 (claude-opus-4.7)*: 68.0%" in result

    def test_missing_model_prefix_leaves_bullet_unannotated(self):

        base_text = "## Report 1 Summary\n### Forecasts\n*Forecaster 1*: 72.0%\n### Research Summary\nSome research.\n"
        predictions = [MagicMock(reasoning="just analysis, no model prefix")]
        result = format_research_summary_with_models(base_text, predictions)
        assert "*Forecaster 1*: 72.0%" in result
        assert "(" not in result.split("### Forecasts")[1].split("### Research Summary")[0]

    def test_preserves_oversized_text_for_complete_comment_allocation(self):

        huge_text = "## Report 1 Summary\n" + ("X" * (SUMMARY_SECTION_CHAR_LIMIT + 5000))
        predictions: list = []
        result = format_research_summary_with_models(huge_text, predictions)
        assert result == huge_text


class TestBuildUnifiedExplanation:
    def _make_question(
        self,
        cls: type[BinaryQuestion | NumericQuestion | MultipleChoiceQuestion] = BinaryQuestion,
        qid: int = 12345,
    ) -> MagicMock:
        q = MagicMock(spec=cls)
        q.id_of_question = qid
        return q

    def test_non_stacking_strategy_just_trims(self):

        base_text = "# SUMMARY\n\nBody text."
        result = build_unified_explanation(
            base_text=base_text,
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.MEAN,
            stacker_outcome=None,
        )
        assert "STACKED=" not in result
        assert "STACKER_OUTCOME=" not in result
        assert "Body text." in result

    def test_median_strategy_just_trims(self):

        base_text = "# SUMMARY\nContent."
        result = build_unified_explanation(
            base_text=base_text,
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.MEDIAN,
            stacker_outcome=None,
        )
        assert "STACKED=" not in result

    def test_stacking_primary_emits_correct_markers(self):

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.STACKING,
            stacker_outcome="primary",
        )
        assert STACKER_OUTCOME_PRIMARY in result
        assert STACKED_MARKER_TRUE in result
        assert STACKED_MARKER_FALSE not in result

    def test_stacking_fallback_llm_emits_correct_markers(self):

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.STACKING,
            stacker_outcome="fallback_llm",
        )
        assert STACKER_OUTCOME_FALLBACK_LLM in result
        assert STACKED_MARKER_TRUE in result

    def test_stacking_fallback_median_emits_correct_markers(self):

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_outcome="fallback_median",
        )
        assert STACKER_OUTCOME_FALLBACK_MEDIAN in result
        assert STACKED_MARKER_FALSE in result
        assert STACKED_MARKER_TRUE not in result

    def test_stacking_fallback_mean_emits_correct_markers(self):

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.STACKING,
            stacker_outcome="fallback_mean",
        )
        assert STACKER_OUTCOME_FALLBACK_MEAN in result
        assert STACKED_MARKER_FALSE in result
        assert STACKED_MARKER_TRUE not in result

    def test_stacking_skipped_emits_correct_markers(self):

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_outcome="skipped",
        )
        assert STACKER_OUTCOME_SKIPPED in result
        assert STACKED_MARKER_FALSE in result
        assert STACKED_MARKER_TRUE not in result

    def test_stacking_skipped_config_off_emits_correct_markers(self):
        # skipped_config_off: spread exceeded the threshold but the per-type
        # <TYPE>_STACKING_ENABLED gate was off. Distinguished from plain
        # "skipped" (spread below threshold) so residual analysis doesn't need
        # git archaeology to re-attribute config-off suppressions.

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_outcome="skipped_config_off",
        )
        assert STACKER_OUTCOME_SKIPPED_CONFIG_OFF in result
        assert STACKED_MARKER_FALSE in result
        assert STACKED_MARKER_TRUE not in result

    def test_skip_reason_marker_rides_each_skip_outcome(self):
        # The additive STACKER_SKIP_REASON companion: plain "skipped" conflates
        # spread-below-threshold with the single-forecaster short-circuit (which
        # computes no spread at all), so the reason gets its own marker while
        # STACKER_OUTCOME stays byte-stable for existing parsers.

        # Every reason in the frozenset, not a hand-picked subset: the config-off
        # reason rides the config-off outcome; every other reason rides plain "skipped".
        for reason in sorted(STACKER_SKIP_REASONS):
            outcome = "skipped_config_off" if reason == "config_off" else "skipped"
            result = build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(),
                aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
                stacker_outcome=outcome,
                skip_reason=reason,
            )
            assert f"<!-- STACKER_SKIP_REASON={reason} -->" in result
            assert f"<!-- STACKER_OUTCOME={outcome} -->" in result

    def test_skip_reason_absent_when_not_supplied(self):
        # Back-compat: every pre-field comment (and every non-skip outcome) omits
        # the marker entirely rather than emitting a placeholder value.

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_outcome="skipped",
        )
        assert "STACKER_SKIP_REASON" not in result

    def test_unknown_skip_reason_raises_valueerror(self):

        with pytest.raises(ValueError, match="Unknown stacker skip reason"):
            build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(),
                aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
                stacker_outcome="skipped",
                skip_reason="bogus_reason",
            )

    def test_unknown_outcome_raises_valueerror(self):

        with pytest.raises(ValueError, match="Unknown stacker outcome"):
            build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(),
                aggregation_strategy=AggregationStrategy.STACKING,
                stacker_outcome="bogus_value",
            )

    def test_stacking_with_none_outcome_raises_assertion(self):

        with pytest.raises(AssertionError, match="stacker_outcome must be provided"):
            build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(),
                aggregation_strategy=AggregationStrategy.STACKING,
                stacker_outcome=None,
            )

    def test_tools_used_marker_emitted_for_binary_when_enabled(self):

        with patch("metaculus_bot.comment.formatting._tool_runner_feature_enabled", return_value=True):
            result = build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(cls=BinaryQuestion),
                aggregation_strategy=AggregationStrategy.STACKING,
                stacker_outcome="primary",
            )
        assert TOOLS_USED_MARKER_TRUE in result

    def test_tools_used_marker_false_when_disabled(self):

        with patch("metaculus_bot.comment.formatting._tool_runner_feature_enabled", return_value=False):
            result = build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(cls=BinaryQuestion),
                aggregation_strategy=AggregationStrategy.STACKING,
                stacker_outcome="primary",
            )
        assert TOOLS_USED_MARKER_FALSE in result

    def test_numeric_question_type_passed_to_feature_enabled(self):

        with patch("metaculus_bot.comment.formatting._tool_runner_feature_enabled", return_value=False) as mock_fe:
            build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(cls=NumericQuestion),
                aggregation_strategy=AggregationStrategy.STACKING,
                stacker_outcome="primary",
            )
        mock_fe.assert_called_once_with("numeric")

    def test_mc_question_type_passed_to_feature_enabled(self):

        with patch("metaculus_bot.comment.formatting._tool_runner_feature_enabled", return_value=False) as mock_fe:
            build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(cls=MultipleChoiceQuestion),
                aggregation_strategy=AggregationStrategy.STACKING,
                stacker_outcome="primary",
            )
        mock_fe.assert_called_once_with("multiple_choice")

    def test_trims_oversized_comment(self):

        huge_base = "# SUMMARY\n" + ("X" * (COMMENT_CHAR_LIMIT + 5000))
        result = build_unified_explanation(
            base_text=huge_base,
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.STACKING,
            stacker_outcome="primary",
        )
        assert len(result) <= COMMENT_CHAR_LIMIT
        assert STACKER_OUTCOME_PRIMARY in result

    # -----------------------------------------------------------------------
    # FORECASTERS_USED ensemble-size disclosure (n contributed / N configured)
    # -----------------------------------------------------------------------

    def test_forecasters_used_marker_degraded_ensemble(self):

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_outcome="skipped_config_off",
            n_used=2,
            n_configured=3,
        )
        match = FORECASTERS_USED_MARKER_RE.search(result)
        assert match is not None
        assert match.group(1) == "2"
        assert match.group(2) == "3"
        # Additive: existing markers are untouched.
        assert STACKER_OUTCOME_SKIPPED_CONFIG_OFF in result

    def test_forecasters_used_marker_full_ensemble_distinguishable(self):
        # The whole point: a full-ensemble comment (3/3) is distinguishable from a
        # degraded one (2/3), so a missing bullet can't be confused with a roster change.

        full = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_outcome="skipped",
            n_used=3,
            n_configured=3,
        )
        degraded = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_outcome="skipped",
            n_used=2,
            n_configured=3,
        )
        assert "<!-- FORECASTERS_USED=3/3 -->" in full
        assert "<!-- FORECASTERS_USED=2/3 -->" in degraded

    def test_forecasters_used_marker_single_forecaster(self):
        # MIN_FORECASTERS_TO_PUBLISH=1: a lone-survivor publish is disclosed too.

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_outcome="skipped",
            n_used=1,
            n_configured=3,
        )
        assert "<!-- FORECASTERS_USED=1/3 -->" in result

    def test_forecasters_used_marker_on_non_stacking_strategy(self):
        # Ensemble size is orthogonal to stacking: the marker rides even on the
        # non-stacking early-return path (backtests use MEAN).

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.MEAN,
            stacker_outcome=None,
            n_used=2,
            n_configured=3,
        )
        assert "<!-- FORECASTERS_USED=2/3 -->" in result
        # Non-stacking path still emits no stacker markers.
        assert "STACKER_OUTCOME=" not in result

    def test_forecasters_used_marker_absent_when_counts_not_provided(self):
        # Back-compat: callers that don't pass counts (existing tests, legacy
        # paths) get no marker — never a spurious 0/0.

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.STACKING,
            stacker_outcome="primary",
        )
        assert "FORECASTERS_USED=" not in result

    def test_forecasters_used_marker_survives_trimming(self):
        # A disclosure that gets middle-trimmed away on a 150k comment is
        # worthless: it must ride the preserved tail alongside the stacker markers.

        huge_base = "# SUMMARY\n" + ("X" * (COMMENT_CHAR_LIMIT + 5000))
        result = build_unified_explanation(
            base_text=huge_base,
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_outcome="skipped_config_off",
            n_used=2,
            n_configured=3,
        )
        assert len(result) <= COMMENT_CHAR_LIMIT
        assert "<!-- FORECASTERS_USED=2/3 -->" in result

    @pytest.mark.parametrize(
        "question_cls",
        [BinaryQuestion, NumericQuestion, MultipleChoiceQuestion],
        ids=["binary", "numeric", "multiple_choice"],
    )
    @pytest.mark.parametrize(
        ("strategy", "stacker_outcome"),
        [
            (AggregationStrategy.CONDITIONAL_STACKING, "skipped"),
            (AggregationStrategy.MEAN, None),
        ],
        ids=["stacking_branch", "non_stacking_branch"],
    )
    def test_forecasters_used_marker_on_every_supported_question_type(
        self,
        question_cls: type[BinaryQuestion | NumericQuestion | MultipleChoiceQuestion],
        strategy: AggregationStrategy,
        stacker_outcome: str | None,
    ):
        """Every supported question type and aggregation branch must disclose ensemble size."""

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(cls=question_cls),
            aggregation_strategy=strategy,
            stacker_outcome=stacker_outcome,
            n_used=2,
            n_configured=3,
        )
        assert "<!-- FORECASTERS_USED=2/3 -->" in result

    def test_forecasters_used_marker_survives_trimming_on_non_stacking_path(self):
        """The disclosure must survive trimming without trailing stacker markers."""

        huge_base = "# SUMMARY\n" + ("X" * (COMMENT_CHAR_LIMIT + 5000))
        result = build_unified_explanation(
            base_text=huge_base,
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.MEAN,
            stacker_outcome=None,
            n_used=2,
            n_configured=3,
        )
        assert len(result) <= COMMENT_CHAR_LIMIT
        assert "<!-- FORECASTERS_USED=2/3 -->" in result
