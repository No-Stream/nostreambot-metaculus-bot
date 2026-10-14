"""Tests for metaculus_bot.performance_analysis.parsing.

The parser maps `*Forecaster N*: value` bullets in the summary section of a
Metaculus bot comment to the model that actually produced each forecast.
Ground truth for the index->model mapping comes from `Model: openrouter/...`
lines inside the `## R1: Forecaster N Reasoning` blocks further down in the
comment, injected by main.py when it wraps each prediction's reasoning.

Attribution must never silently mislabel — if the Model: lines are missing
or malformed, the parser should return an anonymized fallback (Forecaster N
keys), never guess from a configured roster.

Per-parser unit coverage lives here. The real-comment cohort machinery — the
checked-in miniature that is the CI floor, the local breadth sweep, and the
fall-aib-2025 fixture — lives in ``tests/test_performance_analysis_attribution.py``.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import Mock

import pytest

from metaculus_bot.comment.markers import (
    STACKED_BASE_REASONING_HEADER,
    STACKER_META_ANALYSIS_HEADER,
    STACKER_SKIP_REASONS,
)
from metaculus_bot.performance_analysis.parsing import (
    _MC_OPTION_LINE_RE,
    _iter_per_model_blocks,
    _parse_probability,
    _summary_section_for_bullets,
    _validate_percentile_labels,
    annotate_forecaster_bullets_with_models,
    detect_historical_stacker_signature,
    extract_model_display_name_from_reasoning,
    parse_explanation_detail_status,
    parse_forecaster_model_map,
    parse_forecasters_used_marker,
    parse_inferred_stacker_outcome,
    parse_per_base_model_forecasts,
    parse_per_model_forecasts,
    parse_per_model_mc_option_probs,
    parse_per_model_numeric_percentiles,
    parse_per_model_reasoning_text,
    parse_resolution,
    parse_stacked_marker,
    parse_stacker_outcome_marker,
    parse_stacker_skip_reason_marker,
)
from metaculus_bot.stacking import combine_stacker_and_base_reasoning

# ---------------------------------------------------------------------------
# parse_stacked_marker
# ---------------------------------------------------------------------------


class TestParseStackedMarker:
    def test_true_marker(self):
        assert parse_stacked_marker("# SUMMARY\n...\n<!-- STACKED=true -->\n") is True

    def test_false_marker(self):
        assert parse_stacked_marker("# SUMMARY\n...\n<!-- STACKED=false -->\n") is False

    def test_absent_marker_returns_none(self):
        assert parse_stacked_marker("# SUMMARY\nNo marker here\n") is None

    def test_case_insensitive(self):
        assert parse_stacked_marker("<!-- stacked=TRUE -->") is True
        assert parse_stacked_marker("<!-- STACKED=FALSE -->") is False

    def test_whitespace_tolerant(self):
        assert parse_stacked_marker("<!--   STACKED=true   -->") is True


# ---------------------------------------------------------------------------
# parse_stacker_outcome_marker (new tri-state marker)
# ---------------------------------------------------------------------------


class TestParseStackerOutcomeMarker:
    def test_primary_marker(self):
        assert parse_stacker_outcome_marker("...\n<!-- STACKER_OUTCOME=primary -->\n") == "primary"

    def test_fallback_llm_marker(self):
        assert parse_stacker_outcome_marker("...\n<!-- STACKER_OUTCOME=fallback_llm -->\n") == "fallback_llm"

    def test_fallback_median_marker(self):
        # The load-bearing case: pre-fix, this path silently emitted STACKED=true.
        assert parse_stacker_outcome_marker("...\n<!-- STACKER_OUTCOME=fallback_median -->\n") == "fallback_median"

    def test_skipped_marker(self):
        assert parse_stacker_outcome_marker("...\n<!-- STACKER_OUTCOME=skipped -->\n") == "skipped"

    def test_skipped_config_off_marker(self):
        # The alternation-order case: "skipped" must not shadow the longer
        # "skipped_config_off" literal (config-suppressed vs below-threshold).
        assert (
            parse_stacker_outcome_marker("...\n<!-- STACKER_OUTCOME=skipped_config_off -->\n") == "skipped_config_off"
        )

    def test_absent_marker_returns_none(self):
        assert parse_stacker_outcome_marker("# SUMMARY\nNo marker here\n") is None

    def test_case_insensitive(self):
        assert parse_stacker_outcome_marker("<!-- stacker_outcome=PRIMARY -->") == "primary"

    def test_whitespace_tolerant(self):
        assert parse_stacker_outcome_marker("<!--  STACKER_OUTCOME=fallback_llm  -->") == "fallback_llm"


class TestParseStackerSkipReasonMarker:
    """The additive skip-reason marker: what disambiguates q44870's single-forecaster
    short-circuit from a spread-below-threshold skip in the published record."""

    def test_every_registered_reason_parses(self):
        # Iterates the emit side's frozenset so a reason added there (spread_undefined,
        # wall_clock_budget were once missed here) is covered the moment it lands.
        for reason in sorted(STACKER_SKIP_REASONS):
            text = f"...\n<!-- STACKER_OUTCOME=skipped -->\n<!-- STACKER_SKIP_REASON={reason} -->\n"
            assert parse_stacker_skip_reason_marker(text) == reason

    def test_absent_marker_returns_none(self):
        assert parse_stacker_skip_reason_marker("...\n<!-- STACKER_OUTCOME=skipped -->\n") is None

    def test_case_insensitive_and_lowercased(self):
        assert parse_stacker_skip_reason_marker("<!-- stacker_skip_reason=SINGLE_FORECASTER -->") == (
            "single_forecaster"
        )


# ---------------------------------------------------------------------------
# parse_forecasters_used_marker (ensemble-size disclosure)
# ---------------------------------------------------------------------------


class TestParseForecastersUsedMarker:
    def test_degraded_ensemble(self):
        # (used, configured) — 2 of 3 contributed: a dropped model, NOT a roster change.
        assert parse_forecasters_used_marker("...\n<!-- FORECASTERS_USED=2/3 -->\n") == (2, 3)

    def test_full_ensemble(self):
        assert parse_forecasters_used_marker("...\n<!-- FORECASTERS_USED=3/3 -->\n") == (3, 3)

    def test_single_forecaster(self):
        assert parse_forecasters_used_marker("...\n<!-- FORECASTERS_USED=1/3 -->\n") == (1, 3)

    def test_absent_marker_returns_none(self):
        # Old-era comments predate the marker: unknown ensemble size, not a lie.
        assert parse_forecasters_used_marker("# SUMMARY\nNo marker here\n") is None

    def test_whitespace_tolerant(self):
        assert parse_forecasters_used_marker("<!--   FORECASTERS_USED=2/3   -->") == (2, 3)


class TestParseExplanationDetailStatus:
    def test_explicit_incomplete_detail_is_unknown(self):
        assert parse_explanation_detail_status("# SUMMARY\n<!-- EXPLANATION_DETAIL=unknown -->") == "unknown"

    def test_unmarked_historical_comment_stays_unclassified(self):
        assert parse_explanation_detail_status("# SUMMARY\nOld comment") is None
        assert parse_explanation_detail_status(None) is None


# ---------------------------------------------------------------------------
# detect_historical_stacker_signature (body shape, no marker)
# ---------------------------------------------------------------------------


class TestDetectHistoricalStackerSignature:
    """Detection of pre-marker stacked comments (April 2026 vintage and earlier).

    The historical signature is `## R1: Forecaster 1 Reasoning` immediately
    followed by `## (Stacker )?Meta-Analysis`. This shape is produced by the
    stacking commit (2026-04-02, c6d1ab3) and survived the rename in
    95c4fff (2026-04-27). Comments published before either commit landed
    don't carry the signature; comments where Forecaster 1's body is
    ordinary model reasoning don't carry it either.
    """

    def test_historical_meta_analysis_header(self):
        # Pre-rename: `## Meta-Analysis` (no "Stacker" prefix).
        comment = (
            "# SUMMARY\nstuff\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.5\n\n"
            "## Meta-Analysis\n\n"
            "Synthesis of 6 base models below.\n"
        )
        assert detect_historical_stacker_signature(comment) is True

    def test_current_stacker_meta_analysis_header(self):
        # Post-rename: `## Stacker Meta-Analysis`.
        comment = (
            "# SUMMARY\nstuff\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "## Stacker Meta-Analysis\n\n"
            "Synthesis of 6 base models below.\n"
        )
        assert detect_historical_stacker_signature(comment) is True

    def test_h1_meta_analysis_and_synthesis_header(self):
        # Earliest stacker variant emitted ``# Meta-Analysis and Synthesis``
        # as an H1 heading. Observed in April 2026 records of the
        # spring-aib-2026 dataset (e.g., post 43155).
        comment = (
            "# SUMMARY\nstuff\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.4\n\n"
            "# Meta-Analysis and Synthesis\n"
            "## Model Agreement Analysis\n"
            "..."
        )
        assert detect_historical_stacker_signature(comment) is True

    def test_no_meta_header_in_first_section_returns_false(self):
        # Plain stacking-skipped or non-stacking comment: Forecaster 1 has its
        # own analysis structure, not a meta-analysis.
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "Probability: 72%\nReasoning: ...\n\n"
            "## R1: Forecaster 2 Reasoning\nProbability: 75%\n"
        )
        assert detect_historical_stacker_signature(comment) is False

    def test_meta_analysis_inside_individual_forecaster_body_returns_false(self):
        # A model's own reasoning structure may include a "## Meta-Analysis"
        # heading several paragraphs in — that's not the stacker shape; the
        # FIRST heading after the section header is what matters.
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "## Source Analysis\nstuff\n\n"
            "## Meta-Analysis\nlocal sub-section\n"
        )
        assert detect_historical_stacker_signature(comment) is False

    def test_no_forecaster_1_section_returns_false(self):
        assert detect_historical_stacker_signature("# SUMMARY\nno forecasts here\n") is False

    def test_whitespace_tolerant(self):
        comment = "## R1: Forecaster 1 Reasoning\n\n\nModel:openrouter/openai/gpt-5.5\n## Stacker Meta-Analysis\n"
        assert detect_historical_stacker_signature(comment) is True


# ---------------------------------------------------------------------------
# parse_inferred_stacker_outcome (combined: marker → legacy → historical-body)
# ---------------------------------------------------------------------------


class TestParseInferredStackerOutcome:
    """End-to-end stacker-outcome detection across all three sources of signal.

    Verifies that the combined parser handles every combination a real dataset
    can throw at it: post-fix tri-state markers (preferred), legacy STACKED=
    markers (one-round back-compat), and pre-marker historical body shapes.
    """

    def test_marker_outcome_preferred_when_present(self):
        comment = (
            "## R1: Forecaster 1 Reasoning\nModel: x\n\n## Meta-Analysis\nstuff\n"
            "<!-- STACKER_OUTCOME=fallback_llm -->\n"
            "<!-- STACKED=true -->\n"
        )
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "fallback_llm"
        assert source == "marker_outcome"

    def test_marker_outcome_fallback_median_distinguished_from_legacy(self):
        # The bug fix: fallback_median should be readable distinctly, not
        # collapsed to legacy STACKED=true (the pre-fix behavior).
        comment = "stuff\n<!-- STACKER_OUTCOME=fallback_median -->\n"
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "fallback_median"
        assert source == "marker_outcome"

    def test_legacy_marker_true_maps_to_primary(self):
        # Legacy STACKED=true cannot distinguish primary from fallback_llm,
        # so it conservatively maps to primary (the most common case).
        comment = "stuff\n<!-- STACKED=true -->\n"
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "primary"
        assert source == "marker_legacy"

    def test_legacy_marker_false_maps_to_skipped(self):
        comment = "stuff\n<!-- STACKED=false -->\n"
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "skipped"
        assert source == "marker_legacy"

    def test_historical_body_with_no_marker_infers_primary(self):
        comment = (
            "# SUMMARY\nstuff\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.5\n\n"
            "## Meta-Analysis\nSynthesis of base models.\n"
        )
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "primary"
        assert source == "historical_body"

    def test_no_signal_returns_none(self):
        comment = "## R1: Forecaster 1 Reasoning\nModel: openrouter/openai/gpt-5.5\n\nProbability: 72%\n"
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome is None
        assert source == "none"

    def test_marker_outcome_takes_precedence_over_historical_body(self):
        # Belt-and-suspenders: even if both signals are present (marker added
        # to a comment that already has the historical body shape), the
        # explicit marker wins.
        comment = (
            "## R1: Forecaster 1 Reasoning\nModel: x\n## Stacker Meta-Analysis\nbody\n"
            "<!-- STACKER_OUTCOME=fallback_median -->\n"
        )
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "fallback_median"
        assert source == "marker_outcome"

    def test_legacy_marker_takes_precedence_over_historical_body(self):
        # Precedence ordering is marker_outcome > marker_legacy > historical_body.
        # If a comment has both STACKED=false (legacy "skipped") and a historical
        # body shape that would otherwise be inferred as primary, the legacy
        # marker wins. Source must reflect this so downstream filters can tell
        # marker-derived outcomes from body-shape-derived ones.
        comment = "## R1: Forecaster 1 Reasoning\nModel: x\n## Stacker Meta-Analysis\nbody\n<!-- STACKED=false -->\n"
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "skipped"
        assert source == "marker_legacy"


# ---------------------------------------------------------------------------
# parse_per_model_numeric_percentiles
# ---------------------------------------------------------------------------


class TestParsePerModelNumericPercentiles:
    def test_extracts_percentile_blocks(self):
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "analysis...\n\n"
            "Percentile 2.5: 100.0\n"
            "Percentile 50: 150.0\n"
            "Percentile 97.5: 200.0\n\n"
            "## R1: Forecaster 2 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "more analysis...\n\n"
            "Percentile 2.5: 110.0\n"
            "Percentile 50: 160.0\n"
            "Percentile 97.5: 210.0\n"
        )
        result = parse_per_model_numeric_percentiles(comment)
        assert result["gpt-5.5"] == [(2.5, 100.0), (50.0, 150.0), (97.5, 200.0)]
        assert result["claude-opus-4.7"] == [(2.5, 110.0), (50.0, 160.0), (97.5, 210.0)]

    def test_section_without_percentiles_skipped(self):
        # Stacker meta-block with no distribution should not appear in result
        comment = (
            "## R1: Forecaster 1 Reasoning\nModel: openrouter/openai/gpt-5.5\n\nJust text reasoning, no distribution.\n"
        )
        assert parse_per_model_numeric_percentiles(comment) == {}

    def test_missing_model_line_uses_anonymized_key(self):
        comment = "## R1: Forecaster 1 Reasoning\nNo Model line.\n\nPercentile 50: 42.0\n"
        assert parse_per_model_numeric_percentiles(comment) == {"Forecaster 1": [(50.0, 42.0)]}

    def test_empty_comment(self):
        assert parse_per_model_numeric_percentiles("") == {}


# ---------------------------------------------------------------------------
# Forward-fix: inline model name in summary bullets
# ---------------------------------------------------------------------------


class TestExtractModelDisplayNameFromReasoning:
    def test_happy_path(self):
        reasoning = "Model: openrouter/openai/gpt-5.5\n\nActual reasoning text."
        assert extract_model_display_name_from_reasoning(reasoning) == "gpt-5.5"

    def test_whitespace_variations(self):
        reasoning = "Model:   openrouter/anthropic/claude-opus-4.7  \n\nreasoning."
        assert extract_model_display_name_from_reasoning(reasoning) == "claude-opus-4.7"

    def test_no_prefix_returns_none(self):
        assert extract_model_display_name_from_reasoning("Just reasoning, no prefix.") is None

    def test_empty_value_returns_none(self):
        assert extract_model_display_name_from_reasoning("Model:   \n\nreasoning.") is None

    def test_only_prefix(self):
        assert extract_model_display_name_from_reasoning("Model: openrouter/openai/gpt-5.5") == "gpt-5.5"


class TestAnnotateForecasterBulletsWithModels:
    def test_adds_model_name_to_each_bullet(self):
        text = "### Forecasts\n*Forecaster 1*: 73.0%\n*Forecaster 2*: 75.0%\n"
        annotated = annotate_forecaster_bullets_with_models(text, {1: "gpt-5.5", 2: "claude-opus-4.7"})
        assert "*Forecaster 1 (gpt-5.5)*: 73.0%" in annotated
        assert "*Forecaster 2 (claude-opus-4.7)*: 75.0%" in annotated

    def test_idempotent_when_already_annotated(self):
        text = "*Forecaster 1 (gpt-5.5)*: 73.0%\n"
        annotated = annotate_forecaster_bullets_with_models(text, {1: "gpt-5.5"})
        assert annotated == text

    def test_indices_not_in_map_left_unchanged(self):
        text = "*Forecaster 1*: 73.0%\n*Forecaster 2*: 75.0%\n"
        annotated = annotate_forecaster_bullets_with_models(text, {1: "gpt-5.5"})
        assert "*Forecaster 1 (gpt-5.5)*: 73.0%" in annotated
        assert "*Forecaster 2*: 75.0%" in annotated

    def test_round_trip_parses_back_to_model_names(self):
        # The full round-trip: annotate a summary-only text, then parse it back.
        # Simulates the comment-trimmed case where the rationales section is
        # cut off but the summary survives.
        text = (
            "## Report 1 Summary\n### Forecasts\n*Forecaster 1*: 73.0%\n*Forecaster 2*: 75.0%\n[... trimmed for length]"
        )
        annotated = annotate_forecaster_bullets_with_models(text, {1: "gpt-5.5", 2: "claude-opus-4.7"})
        parsed = parse_per_model_forecasts(annotated)
        assert parsed == {"gpt-5.5": "73.0%", "claude-opus-4.7": "75.0%"}

    def test_survives_comment_trim_path(self):
        # End-to-end: annotated summary + trimmed rationale section still
        # yields correct per-model attribution.
        summary_only = (
            "*Forecaster 1 (gpt-5.5)*: 73.0%\n*Forecaster 2 (claude-opus-4.7)*: 75.0%\n[... trimmed for length]"
        )
        parsed = parse_per_model_forecasts(summary_only)
        assert parsed == {"gpt-5.5": "73.0%", "claude-opus-4.7": "75.0%"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_comment(bullets: list[tuple[int, str]], rationales: list[tuple[int, str]]) -> str:
    """Build a minimal bot-comment-shaped string.

    bullets:   list of (idx, value) for the Report 1 Summary section
    rationales: list of (idx, model) for the R1 Forecaster N Reasoning blocks
    """
    bullet_lines = "\n".join(f"*Forecaster {i}*: {v}" for i, v in bullets)
    rationale_sections = "\n\n".join(
        f"## R1: Forecaster {i} Reasoning\nModel: {m}\n\nreasoning text here." for i, m in rationales
    )
    return f"""# SUMMARY
*Question*: ?

## Report 1 Summary
### Forecasts
{bullet_lines}


### Research Summary
Some research text.

================================================================================
FORECAST SECTION:

{rationale_sections}
"""


def _build_base_pred(model_path: str, extra_body: str) -> Mock:
    """Build a Mock ReasonedPrediction-shaped object with the bot's Model: prefix.

    main.TemplateForecaster._make_prediction injects the prefix on real predictions;
    we replicate that here so combine_stacker_and_base_reasoning produces a realistic
    shape.
    """
    pred = Mock()
    pred.reasoning = f"Model: {model_path}\n\n{extra_body}"
    return pred


def _build_stacked_comment(
    combined_reasoning: str,
    stacker_bullet_name: str = "stacker-gpt",
    bullet_value: str = "72.0%",
) -> str:
    """Wrap a stacker-combined body in a plausible Metaculus comment skeleton.

    The production flow produces exactly ONE summary bullet for stacked questions
    (the aggregate value) and ONE R1 header whose body is the combined reasoning
    emitted by combine_stacker_and_base_reasoning.
    """
    return f"""# SUMMARY
*Question*: Will stacking improve calibration?

## Report 1 Summary
### Forecasts
*Forecaster 1 ({stacker_bullet_name})*: {bullet_value}

### Research Summary
Some research.

================================================================================
FORECAST SECTION:

## R1: Forecaster 1 Reasoning
{combined_reasoning}

<!-- STACKED=true -->
"""


# ---------------------------------------------------------------------------
# parse_forecaster_model_map
# ---------------------------------------------------------------------------


class TestParseForecasterModelMap:
    def test_standard_five_model_ensemble(self):
        comment = _build_comment(
            bullets=[(1, "73.0%"), (2, "75.0%"), (3, "85.0%"), (4, "92.0%"), (5, "88.0%")],
            rationales=[
                (1, "openrouter/openai/gpt-5.2"),
                (2, "openrouter/openai/gpt-5.1"),
                (3, "openrouter/anthropic/claude-4.6-opus"),
                (4, "openrouter/anthropic/claude-opus-4.5"),
                (5, "openrouter/google/gemini-3.1-pro-preview"),
            ],
        )
        result = parse_forecaster_model_map(comment)
        assert result == {
            1: "gpt-5.2",
            2: "gpt-5.1",
            3: "claude-4.6-opus",
            4: "claude-opus-4.5",
            5: "gemini-3.1-pro-preview",
        }

    def test_model_names_with_dots_and_hyphens(self):
        comment = _build_comment(
            bullets=[(1, "50%")],
            rationales=[(1, "openrouter/openai/gpt-5.5")],
        )
        assert parse_forecaster_model_map(comment) == {1: "gpt-5.5"}

    def test_long_multi_dot_name(self):
        comment = _build_comment(
            bullets=[(1, "50%")],
            rationales=[(1, "openrouter/google/gemini-3.1-pro-preview")],
        )
        assert parse_forecaster_model_map(comment) == {1: "gemini-3.1-pro-preview"}

    def test_ignores_report_2_rationales(self):
        # Only R1 matters — R2 is a separate research report; attribution should
        # come from R1.
        comment = _build_comment(
            bullets=[(1, "60%")],
            rationales=[(1, "openrouter/openai/gpt-5.2")],
        )
        # Append a bogus R2 with a different model
        comment += "\n\n## R2: Forecaster 1 Reasoning\nModel: openrouter/openai/gpt-4\n\nother."
        assert parse_forecaster_model_map(comment) == {1: "gpt-5.2"}

    def test_missing_model_lines_returns_empty(self):
        comment = _build_comment(
            bullets=[(1, "60%")],
            rationales=[],
        )
        assert parse_forecaster_model_map(comment) == {}

    def test_malformed_model_line_skipped(self):
        # Line says "Model:" but no value — must not crash, must return empty
        comment = """## R1: Forecaster 1 Reasoning
Model:

## R1: Forecaster 2 Reasoning
Model: openrouter/openai/gpt-5.2

"""
        # Forecaster 1 is malformed; Forecaster 2 is fine
        result = parse_forecaster_model_map(comment)
        assert result == {2: "gpt-5.2"}

    def test_stray_model_line_in_research_ignored(self):
        # A Model: line appearing outside a Forecaster Reasoning header (e.g.,
        # research quoting a model name) must not confuse attribution.
        comment = (
            "### Research Summary\n"
            "Model: openrouter/openai/gpt-99-fake\n"
            "This was cited in research.\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.2\n\n"
            "real reasoning.\n"
        )
        assert parse_forecaster_model_map(comment) == {1: "gpt-5.2"}

    def test_whitespace_variations(self):
        # Tabs, extra spaces, trailing whitespace on Model line must still parse.
        comment = "## R1: Forecaster 1 Reasoning\n  Model:   openrouter/openai/gpt-5.2   \n\nreasoning\n"
        assert parse_forecaster_model_map(comment) == {1: "gpt-5.2"}

    def test_empty_comment(self):
        assert parse_forecaster_model_map("") == {}


# ---------------------------------------------------------------------------
# parse_per_model_forecasts — integration of model map + bullets
# ---------------------------------------------------------------------------


class TestParsePerModelForecasts:
    def test_happy_path_keys_by_model_name(self):
        comment = _build_comment(
            bullets=[(1, "73.0%"), (2, "75.0%"), (3, "85.0%")],
            rationales=[
                (1, "openrouter/openai/gpt-5.2"),
                (2, "openrouter/openai/gpt-5.1"),
                (3, "openrouter/anthropic/claude-4.6-opus"),
            ],
        )
        result = parse_per_model_forecasts(comment)
        assert result == {
            "gpt-5.2": "73.0%",
            "gpt-5.1": "75.0%",
            "claude-4.6-opus": "85.0%",
        }

    def test_bullets_without_rationales_falls_back_to_anonymized(self):
        # Pre-injection comments (if any exist) or future breakage — don't
        # guess, return anonymized keys so analysis can still aggregate
        # overall but per-model numbers are correctly absent.
        comment = """## Report 1 Summary
### Forecasts
*Forecaster 1*: 73.0%
*Forecaster 2*: 75.0%
"""
        result = parse_per_model_forecasts(comment)
        assert result == {"Forecaster 1": "73.0%", "Forecaster 2": "75.0%"}

    def test_partial_model_lines_mixes_named_and_anonymized(self):
        # Forecaster 1 has Model: line, Forecaster 2 does not.
        comment = """## Report 1 Summary
### Forecasts
*Forecaster 1*: 73.0%
*Forecaster 2*: 75.0%

## R1: Forecaster 1 Reasoning
Model: openrouter/openai/gpt-5.2

reasoning.
"""
        result = parse_per_model_forecasts(comment)
        assert result == {"gpt-5.2": "73.0%", "Forecaster 2": "75.0%"}

    def test_empty_comment(self):
        assert parse_per_model_forecasts("") == {}

    def test_explicit_model_names_override_is_supported(self):
        # Back-compat: callers can pass model_names to force a specific roster
        # (useful for tests or external pipelines). When passed, skip the
        # Model:-line lookup.
        comment = """*Forecaster 1*: 73.0%
*Forecaster 2*: 75.0%
"""
        result = parse_per_model_forecasts(comment, model_names=["alpha", "beta"])
        assert result == {"alpha": "73.0%", "beta": "75.0%"}


# ---------------------------------------------------------------------------
# Stacked-comment parsing — per-base-model recovery from the combined R1 body.
# ---------------------------------------------------------------------------


class TestParsePerModelNumericPercentilesStacked:
    """Per-base-model percentile recovery from stacker-combined comments.

    When stacking fires, all base predictions collapse into one ReasonedPrediction,
    so there's only one R1 header; the base reasonings are folded into its body
    under the "## Base Model Reasoning (inputs to stacker)" delimiter. The parser
    must split on that delimiter and attribute percentiles per base model.
    """

    def test_three_base_models_each_with_percentiles(self):
        base_names = [
            "openrouter/openai/gpt-5.5",
            "openrouter/anthropic/claude-opus-4.7",
            "openrouter/google/gemini-3.1-pro-preview",
        ]
        base_preds = []
        for i, path in enumerate(base_names):
            body = (
                f"analysis for model {i}\n\n"
                f"Percentile 10: {100 + 10 * i}\n"
                f"Percentile 25: {125 + 10 * i}\n"
                f"Percentile 50: {150 + 10 * i}\n"
                f"Percentile 75: {175 + 10 * i}\n"
                f"Percentile 90: {200 + 10 * i}"
            )
            base_preds.append(_build_base_pred(path, body))
        meta_text = "My stacker analysis: considered the three inputs, leaned toward median-ish."
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined)

        result = parse_per_model_numeric_percentiles(comment)
        # Three per-base-model entries — one per sub-block.
        assert set(result.keys()) == {"gpt-5.5", "claude-opus-4.7", "gemini-3.1-pro-preview"}
        assert result["gpt-5.5"] == [(10.0, 100.0), (25.0, 125.0), (50.0, 150.0), (75.0, 175.0), (90.0, 200.0)]
        assert result["claude-opus-4.7"] == [(10.0, 110.0), (25.0, 135.0), (50.0, 160.0), (75.0, 185.0), (90.0, 210.0)]
        assert result["gemini-3.1-pro-preview"] == [
            (10.0, 120.0),
            (25.0, 145.0),
            (50.0, 170.0),
            (75.0, 195.0),
            (90.0, 220.0),
        ]

    def test_stacker_meta_percentiles_included_under_stacker_name(self):
        # Some stackers explicitly restate the final distribution in prose.
        # Those percentile lines should be captured under the stacker's model
        # name (the R1 header's Model: line, or "Forecaster 1" fallback when
        # absent).
        base_pred = _build_base_pred(
            "openrouter/openai/gpt-5.5",
            "base prose\n\nPercentile 50: 100",
        )
        meta_text = "Final stacker distribution:\nPercentile 10: 90\nPercentile 50: 110\nPercentile 90: 140"
        combined = combine_stacker_and_base_reasoning(meta_text, [base_pred])
        comment = _build_stacked_comment(combined)

        result = parse_per_model_numeric_percentiles(comment)
        # Base model's single percentile line
        assert result["gpt-5.5"] == [(50.0, 100.0)]
        # Stacker's prose percentiles — the R1 body has no Model: line (the
        # aggregated_prediction's reasoning starts with ## Stacker Meta-Analysis),
        # so the stacker falls back to "Forecaster 1".
        assert result["Forecaster 1"] == [(10.0, 90.0), (50.0, 110.0), (90.0, 140.0)]

    def test_non_stacked_comment_matches_existing_behavior(self):
        # Sanity: when the delimiter is absent, the stacker branch must not
        # trigger, and per-R1-header attribution behaves as before.
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "analysis...\n\nPercentile 50: 42\n\n"
            "## R1: Forecaster 2 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "analysis...\n\nPercentile 50: 55\n"
        )
        assert parse_per_model_numeric_percentiles(comment) == {
            "gpt-5.5": [(50.0, 42.0)],
            "claude-opus-4.7": [(50.0, 55.0)],
        }

    def test_mixed_r1_blocks_stacked_and_unstacked(self):
        # Unusual (cross-report) shape: one R1 block is stacker-combined, another
        # is a plain per-forecaster section. Each should be handled on its own
        # terms without cross-contamination.
        base_pred = _build_base_pred(
            "openrouter/openai/gpt-5.5",
            "base prose\n\nPercentile 50: 200",
        )
        combined = combine_stacker_and_base_reasoning("stacker meta", [base_pred])
        comment = (
            f"## R1: Forecaster 1 Reasoning\n{combined}\n"
            "## R1: Forecaster 2 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "plain analysis\n\nPercentile 50: 300\n"
        )
        result = parse_per_model_numeric_percentiles(comment)
        # Stacked block contributes the base model's name
        assert result["gpt-5.5"] == [(50.0, 200.0)]
        # Non-stacked block keyed by its own Model: line
        assert result["claude-opus-4.7"] == [(50.0, 300.0)]
        # Fallback "Forecaster N" should NOT appear — the stacker meta-text
        # has no percentile lines so it's not attributed.
        assert "Forecaster 1" not in result


class TestParsePerModelReasoningTextStacked:
    """Per-base-model prose recovery from stacker-combined comments."""

    def test_returns_entry_per_base_model_plus_stacker(self):
        base_names = [
            "openrouter/openai/gpt-5.5",
            "openrouter/anthropic/claude-opus-4.7",
            "openrouter/google/gemini-3.1-pro-preview",
        ]
        base_preds = [_build_base_pred(name, f"reasoning body for model {i}") for i, name in enumerate(base_names)]
        meta_text = "My stacker meta-analysis prose."
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined)

        result = parse_per_model_reasoning_text(comment)
        # 3 base models + stacker = 4 entries. Stacker's R1 body has no Model:
        # line, so it falls back to "Forecaster 1".
        assert set(result.keys()) == {"gpt-5.5", "claude-opus-4.7", "gemini-3.1-pro-preview", "Forecaster 1"}
        for i, name in enumerate(["gpt-5.5", "claude-opus-4.7", "gemini-3.1-pro-preview"]):
            assert f"reasoning body for model {i}" in result[name]
        assert "stacker meta-analysis prose" in result["Forecaster 1"].lower()

    def test_non_stacked_comment_matches_existing_behavior(self):
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "plain reasoning A.\n\n"
            "## R1: Forecaster 2 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "plain reasoning B.\n"
        )
        result = parse_per_model_reasoning_text(comment)
        assert "plain reasoning A" in result["gpt-5.5"]
        assert "plain reasoning B" in result["claude-opus-4.7"]
        assert set(result.keys()) == {"gpt-5.5", "claude-opus-4.7"}


class TestParsePerModelForecastsStacked:
    """parse_per_model_forecasts only captures the summary bullet.

    For stacked comments the bot publishes ONE summary bullet (the stacker's
    aggregate), not one per base model. So this function correctly returns
    a single-entry dict — the per-base-model values are never written to the
    summary and are recoverable only from the reasoning prose via
    parse_per_model_reasoning_text / parse_per_model_numeric_percentiles.
    Documenting this behavior explicitly so a future reader doesn't misread
    the single-entry return as a bug.
    """

    def test_stacked_comment_returns_single_entry_for_stacker(self):
        base_preds = [
            _build_base_pred("openrouter/openai/gpt-5.5", "base A"),
            _build_base_pred("openrouter/anthropic/claude-opus-4.7", "base B"),
        ]
        combined = combine_stacker_and_base_reasoning("stacker prose", base_preds)
        comment = _build_stacked_comment(combined, stacker_bullet_name="stacker-model", bullet_value="72.0%")

        result = parse_per_model_forecasts(comment)
        # Exactly one entry — the stacker's aggregate. This is BY DESIGN: the
        # bot does not publish per-base-model bullets when stacking fires.
        assert result == {"stacker-model": "72.0%"}


class TestStackedCommentEndToEnd:
    """End-to-end: synthesize a stacker-combined comment and run every parser."""

    def test_full_parse_pipeline(self):
        base_names = [
            "openrouter/openai/gpt-5.5",
            "openrouter/anthropic/claude-opus-4.7",
        ]
        base_preds = []
        for i, path in enumerate(base_names):
            body = f"reasoning body {i}\n\nPercentile 50: {150 + 10 * i}\nPercentile 90: {200 + 10 * i}"
            base_preds.append(_build_base_pred(path, body))
        meta_text = "Stacker: I averaged the medians with a slight rightward lean."
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined, stacker_bullet_name="stacker-gpt-5.5", bullet_value="165.0")

        # parse_stacked_marker — trivially True
        assert parse_stacked_marker(comment) is True

        # parse_per_model_forecasts — only the stacker's aggregate
        forecasts = parse_per_model_forecasts(comment)
        assert forecasts == {"stacker-gpt-5.5": "165.0"}

        # parse_per_model_numeric_percentiles — per-base-model distributions
        percentiles = parse_per_model_numeric_percentiles(comment)
        assert percentiles["gpt-5.5"] == [(50.0, 150.0), (90.0, 200.0)]
        assert percentiles["claude-opus-4.7"] == [(50.0, 160.0), (90.0, 210.0)]

        # parse_per_model_reasoning_text — per-base-model prose plus stacker's
        reasoning = parse_per_model_reasoning_text(comment)
        assert "reasoning body 0" in reasoning["gpt-5.5"]
        assert "reasoning body 1" in reasoning["claude-opus-4.7"]
        # Stacker's meta text — no Model: line in combined body, so key falls
        # back to "Forecaster 1".
        assert "averaged the medians" in reasoning["Forecaster 1"]


# ---------------------------------------------------------------------------
# parse_resolution — unchanged behavior, but worth a guard
# ---------------------------------------------------------------------------


class TestParseResolution:
    def test_binary_yes(self):
        assert parse_resolution("yes", "binary") == (True, False)

    def test_binary_no(self):
        assert parse_resolution("no", "binary") == (False, False)

    def test_annulled(self):
        _, skip = parse_resolution("annulled", "binary")
        assert skip is True

    def test_numeric_float(self):
        val, skip = parse_resolution("42.5", "numeric")
        assert val == 42.5
        assert skip is False

    def test_numeric_above_upper(self):
        val, skip = parse_resolution("above_upper_bound", "numeric")
        assert val == "above_upper_bound"
        assert skip is False


# ---------------------------------------------------------------------------
# Shared-constant producer/consumer round-trip. The stacker-combined comment
# format is emitted by metaculus_bot.stacking.combine_stacker_and_base_reasoning
# and re-parsed by the performance-analysis pipeline. If these ever drift, the
# collector silently loses per-base-model attribution on stacked questions, so
# guard the shared header literals explicitly.
# ---------------------------------------------------------------------------


class TestStackedHeaderConstantRoundTrip:
    def test_combine_output_contains_shared_constants(self):
        base_pred = _build_base_pred("openrouter/openai/gpt-5.5", "base prose")
        combined = combine_stacker_and_base_reasoning("stacker meta prose", [base_pred])
        assert STACKER_META_ANALYSIS_HEADER in combined
        assert STACKED_BASE_REASONING_HEADER in combined

    def test_parser_splits_on_shared_constant(self):
        # Verify parser ingests the same literal the producer emits — no drift.
        base_pred = _build_base_pred("openrouter/openai/gpt-5.5", "base prose\n\nPercentile 50: 42")
        combined = combine_stacker_and_base_reasoning("stacker meta", [base_pred])
        comment = _build_stacked_comment(combined)
        parsed = parse_per_model_numeric_percentiles(comment)
        assert parsed["gpt-5.5"] == [(50.0, 42.0)]


# ---------------------------------------------------------------------------
# _iter_per_model_blocks: centralizes the R1-walking boilerplate shared by
# parse_per_model_numeric_percentiles and parse_per_model_reasoning_text. Keep
# a minimal shape check so a future refactor doesn't silently change attribution.
# ---------------------------------------------------------------------------


class TestIterPerModelBlocks:
    def test_plain_body_yields_single_non_stacker_entry_per_r1(self):
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "prose A\n\n"
            "## R1: Forecaster 2 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "prose B\n"
        )
        blocks = list(_iter_per_model_blocks(comment))
        assert len(blocks) == 2
        names = [b[0] for b in blocks]
        is_meta_flags = [b[2] for b in blocks]
        assert names == ["gpt-5.5", "claude-opus-4.7"]
        assert is_meta_flags == [False, False]
        # Bodies must have the Model: prefix stripped.
        assert "prose A" in blocks[0][1]
        assert not blocks[0][1].lstrip().startswith("Model:")

    def test_stacker_combined_body_yields_meta_then_bases(self):
        base_preds = [
            _build_base_pred("openrouter/openai/gpt-5.5", "base A prose"),
            _build_base_pred("openrouter/anthropic/claude-opus-4.7", "base B prose"),
        ]
        combined = combine_stacker_and_base_reasoning("stacker meta prose", base_preds)
        comment = _build_stacked_comment(combined)
        blocks = list(_iter_per_model_blocks(comment))
        # 1 stacker-meta entry + 2 base entries
        assert len(blocks) == 3
        assert blocks[0][2] is True  # stacker meta
        assert blocks[1][2] is False
        assert blocks[2][2] is False
        assert "stacker meta prose" in blocks[0][1]
        assert blocks[1][0] == "gpt-5.5"
        assert blocks[2][0] == "claude-opus-4.7"


# ---------------------------------------------------------------------------
# F10 — _FORECASTER_RE must only match summary-section bullets, not stray
# ``*Forecaster N*: value`` strings appearing inside reasoning prose. Without
# this, an LLM that quotes a forecaster's value back ("...*Forecaster 3*: 50%
# would have been wrong") gets parsed as if it were a real bullet, contaminating
# rank/disagreement metrics.
# ---------------------------------------------------------------------------


class TestForecasterReBoundedToSummarySection:
    def test_stray_forecaster_pattern_in_reasoning_not_parsed(self):
        # A bullet-shaped string appearing INSIDE Forecaster 1's reasoning must
        # NOT be picked up as a real summary bullet. Only the bullets in
        # ``## Report 1 Summary`` count.
        comment = _build_comment(
            bullets=[(1, "73.0%"), (2, "75.0%")],
            rationales=[
                (1, "openrouter/openai/gpt-5.5"),
                (2, "openrouter/anthropic/claude-opus-4.7"),
            ],
        )
        # Inject a stray "bullet" inside the rationale prose - this should be
        # ignored. Without anchoring this becomes a third forecaster.
        comment = comment.replace(
            "reasoning text here.",
            "reasoning text here. Note: *Forecaster 99*: 12.0% would have been wrong.",
            1,
        )
        result = parse_per_model_forecasts(comment)
        # Only the two real summary bullets — not "Forecaster 99".
        assert set(result.keys()) == {"gpt-5.5", "claude-opus-4.7"}
        assert "Forecaster 99" not in result

    def test_real_summary_bullets_still_parsed(self):
        # Belt-and-suspenders: confirm anchoring doesn't break the normal path.
        comment = _build_comment(
            bullets=[(1, "73.0%"), (2, "75.0%"), (3, "80.0%")],
            rationales=[
                (1, "openrouter/openai/gpt-5.5"),
                (2, "openrouter/anthropic/claude-opus-4.7"),
                (3, "openrouter/google/gemini-3.1-pro-preview"),
            ],
        )
        result = parse_per_model_forecasts(comment)
        assert result == {"gpt-5.5": "73.0%", "claude-opus-4.7": "75.0%", "gemini-3.1-pro-preview": "80.0%"}

    def test_no_summary_marker_falls_back_with_warning(self, caplog):
        # When the comment has no ``### Research Summary`` marker (truncated /
        # malformed), the parser must still be able to read summary bullets if
        # they exist BEFORE any stray reasoning text — and emit a warning so
        # operators notice. We accept the legacy unanchored behavior in this
        # path because there's no clean section to bound to.

        comment = "*Forecaster 1*: 73.0%\n*Forecaster 2*: 75.0%\n"
        with caplog.at_level(logging.WARNING):
            result = parse_per_model_forecasts(comment)
        assert result == {"Forecaster 1": "73.0%", "Forecaster 2": "75.0%"}

    def test_bullets_in_research_summary_section_still_excluded(self):
        # Edge case: research prose between ``### Research Summary`` and the
        # rationale section sometimes quotes forecaster-shaped strings. The
        # cleanest split is on the section-end marker (Research Summary or the
        # forecast section divider).
        comment = _build_comment(
            bullets=[(1, "73.0%")],
            rationales=[(1, "openrouter/openai/gpt-5.5")],
        )
        # Inject a stray bullet shape inside the research summary section.
        comment = comment.replace(
            "Some research text.",
            "Some research text.\nQuote: *Forecaster 99*: 99% (this is a quote, not a bullet).",
        )
        result = parse_per_model_forecasts(comment)
        assert "Forecaster 99" not in result
        assert result == {"gpt-5.5": "73.0%"}


# ---------------------------------------------------------------------------
# F11 — _parse_probability must reject implausible values that would be silent
# parse errors. Currently ``"1.5"`` becomes 0.015 (almost certainly wrong),
# which silently enters Brier/spread calcs. Tighten the heuristic so:
#   - Explicit "%" → divide by 100 always
#   - Bare value in [0, 1] → leave alone
#   - Bare value in [1.5, 100] → treat as percentage, divide by 100
#   - Bare value in (1, 1.5) → ambiguous, reject (None) - was a probability
#     >1.0 (impossible) but too small to be a reasonable percentage
#   - Bare value > 100 → reject (None)
# ---------------------------------------------------------------------------


class TestParseProbabilityHeuristic:
    def test_decimal_probability_unchanged(self):
        assert _parse_probability("0.72") == pytest.approx(0.72)

    def test_explicit_percentage(self):
        assert _parse_probability("72.0%") == pytest.approx(0.72)

    def test_explicit_percentage_with_value_below_1_5_still_works(self):
        # ``"1.0%"`` is unambiguously 1% because of the explicit % sign.
        assert _parse_probability("1.0%") == pytest.approx(0.01)

    def test_bare_value_above_threshold_treated_as_percentage(self):
        # ``"72"`` (no %) is almost certainly the bot's missing-% form.
        assert _parse_probability("72") == pytest.approx(0.72)

    def test_bare_value_at_threshold_lower_bound(self):
        # The lower bound for "treat as percentage" — 1.5 is unambiguous.
        assert _parse_probability("1.5") == pytest.approx(0.015)

    def test_bare_value_in_ambiguous_range_rejected(self):
        # ``"1.2"`` (no %) is ambiguous: too high to be a valid probability,
        # too low to confidently be a percentage. Reject as parse error.
        assert _parse_probability("1.2") is None

    def test_bare_value_just_above_one_rejected(self):
        # ``"1.01"`` shouldn't silently become 0.0101 — that's almost certainly
        # a parse error (a real bullet would be 0.01 or 1%).
        assert _parse_probability("1.01") is None

    def test_bare_value_above_one_hundred_rejected(self):
        # ``"150"`` could be a percentage (impossible) or some other quantity.
        # Either way, not a valid probability.
        assert _parse_probability("150") is None

    def test_negative_value_rejected(self):
        assert _parse_probability("-0.5") is None

    def test_unparseable_string_returns_none(self):
        assert _parse_probability("not a number") is None
        assert _parse_probability("") is None

    def test_zero_and_one_boundaries_accepted(self):
        assert _parse_probability("0") == pytest.approx(0.0)
        assert _parse_probability("0.0") == pytest.approx(0.0)
        assert _parse_probability("1") == pytest.approx(1.0)
        assert _parse_probability("1.0") == pytest.approx(1.0)

    def test_one_hundred_percent_accepted(self):
        assert _parse_probability("100%") == pytest.approx(1.0)
        assert _parse_probability("100") == pytest.approx(1.0)

    def test_zero_percent_accepted(self):
        # 0% explicit is a real corner: divide by 100 still yields 0.
        assert _parse_probability("0%") == pytest.approx(0.0)

    def test_warning_emitted_when_skipping_ambiguous(self, caplog):
        # WARNING-level log when we drop an out-of-range value — operators
        # should be able to grep for these to spot upstream parse drift.

        with caplog.at_level(logging.WARNING):
            assert _parse_probability("1.2") is None
        # We don't strictly require a specific message format, just that
        # something was logged at WARNING when we dropped a parseable but
        # implausible value.
        assert any("1.2" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# parse_per_model_mc_option_probs — MC full option vector extraction
# ---------------------------------------------------------------------------


class TestParsePerModelMcOptionProbs:
    """Extracts per-forecaster option probability vectors from MC comment bullets.

    The bug: parse_per_model_forecasts only captures the first line after
    ``*Forecaster N*:``, which for MC is just the first option (e.g.
    ``- Option A: 40.0%``). This parser captures ALL option lines per
    forecaster and returns ``{model: {option: probability}}``.
    """

    def test_basic_mc_two_forecasters(self):

        comment = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: \n"
            "- Option A: 40.0%\n"
            "- Option B: 30.0%\n"
            "- Option C: 20.0%\n"
            "- Option D: 10.0%\n"
            "\n"
            "*Forecaster 2 (claude-opus-4.7)*: \n"
            "- Option A: 35.0%\n"
            "- Option B: 25.0%\n"
            "- Option C: 25.0%\n"
            "- Option D: 15.0%\n"
            "\n"
            "### Research Summary\n"
            "Some research here\n"
        )
        result = parse_per_model_mc_option_probs(comment)
        assert result == {
            "gpt-5.5": {"Option A": 0.40, "Option B": 0.30, "Option C": 0.20, "Option D": 0.10},
            "claude-opus-4.7": {"Option A": 0.35, "Option B": 0.25, "Option C": 0.25, "Option D": 0.15},
        }

    def test_returns_empty_for_binary_comment(self):

        comment = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: 72.0%\n"
            "*Forecaster 2 (claude-opus-4.7)*: 68.0%\n"
            "\n"
            "### Research Summary\n"
            "Some research here\n"
        )
        result = parse_per_model_mc_option_probs(comment)
        assert result == {}

    def test_empty_comment(self):

        assert parse_per_model_mc_option_probs("") == {}

    def test_options_with_special_characters_in_name(self):

        comment = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: \n"
            "- Yes (>50%): 60.0%\n"
            "- No (<=50%): 30.0%\n"
            "- Ambiguous / unclear: 10.0%\n"
            "\n"
            "### Research Summary\n"
        )
        result = parse_per_model_mc_option_probs(comment)
        assert result == {
            "gpt-5.5": {"Yes (>50%)": 0.60, "No (<=50%)": 0.30, "Ambiguous / unclear": 0.10},
        }

    def test_model_map_from_rationales(self):
        """Attribution uses Model: lines from R1 sections when inline name is absent."""

        comment = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1*: \n"
            "- Yes: 70.0%\n"
            "- No: 30.0%\n"
            "\n"
            "### Research Summary\n"
            "stuff\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "reasoning here\n"
        )
        result = parse_per_model_mc_option_probs(comment)
        assert result == {"gpt-5.5": {"Yes": 0.70, "No": 0.30}}

    def test_probabilities_sum_approximately_one(self):
        """Sanity: parsed probabilities should sum to ~1.0 for well-formed comments."""

        comment = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: \n"
            "- A: 25.0%\n"
            "- B: 25.0%\n"
            "- C: 25.0%\n"
            "- D: 25.0%\n"
            "\n"
            "### Research Summary\n"
        )
        result = parse_per_model_mc_option_probs(comment)
        total = sum(result["gpt-5.5"].values())
        assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# parse_per_base_model_forecasts — per-base-model probability recovery from
# stacker-combined reasoning blocks. When stacking fires, the summary bullet
# shows only the stacker's aggregate; base-model individual forecasts are
# recoverable only from the reasoning body (each base sub-block contains a
# "Probability: X%" or "Final probability: X%" line for binary, or option
# lines for MC). This parser extracts those into a field that downstream
# stacker_detection.py can use for counterfactual-median analysis.
# ---------------------------------------------------------------------------


class TestParsePerBaseModelForecasts:
    """Per-base-model probability extraction from stacker-combined reasoning bodies."""

    def test_binary_stacked_comment_extracts_five_base_model_probs(self):
        base_names = [
            "openrouter/openai/gpt-5.5",
            "openrouter/anthropic/claude-opus-4.7",
            "openrouter/google/gemini-3.1-pro-preview",
            "openrouter/x-ai/grok-4.1-fast",
            "openrouter/openai/gpt-5.4",
        ]
        base_preds = []
        for i, path in enumerate(base_names):
            prob = 60 + i * 5
            body = f"My analysis of this question.\n\nProbability: {prob}%"
            base_preds.append(_build_base_pred(path, body))
        meta_text = "Stacker analysis: weighted average of base models.\n\nProbability: 72%"
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined)

        result = parse_per_base_model_forecasts(comment, "binary")
        assert len(result) == 5
        assert result["gpt-5.5"] == "60.0%"
        assert result["claude-opus-4.7"] == "65.0%"
        assert result["gemini-3.1-pro-preview"] == "70.0%"
        assert result["grok-4.1-fast"] == "75.0%"
        assert result["gpt-5.4"] == "80.0%"

    def test_binary_stacked_comment_handles_decimal_probability_format(self):
        base_preds = [
            _build_base_pred("openrouter/openai/gpt-5.5", "Analysis.\n\nFinal probability: 0.72"),
            _build_base_pred("openrouter/anthropic/claude-opus-4.7", "Analysis.\n\nProbability: 65%"),
        ]
        meta_text = "Meta.\n\nProbability: 70%"
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined)

        result = parse_per_base_model_forecasts(comment, "binary")
        assert len(result) == 2
        assert result["gpt-5.5"] == "72.0%"
        assert result["claude-opus-4.7"] == "65.0%"

    def test_binary_extracts_last_probability_when_multiple_present(self):
        body = "Initial estimate: Probability: 40%\nAfter adjusting for base rates...\nFinal probability: 55%"
        base_preds = [_build_base_pred("openrouter/openai/gpt-5.5", body)]
        meta_text = "Meta.\n\nProbability: 55%"
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined)

        result = parse_per_base_model_forecasts(comment, "binary")
        # Should extract the LAST probability (the final answer)
        assert result["gpt-5.5"] == "55.0%"

    def test_mc_stacked_comment_extracts_per_base_model_option_dicts(self):
        base_preds = [
            _build_base_pred(
                "openrouter/openai/gpt-5.5",
                "My analysis:\n- Option A: 60.0%\n- Option B: 25.0%\n- Option C: 15.0%",
            ),
            _build_base_pred(
                "openrouter/anthropic/claude-opus-4.7",
                "My analysis:\n- Option A: 55.0%\n- Option B: 30.0%\n- Option C: 15.0%",
            ),
        ]
        meta_text = "Stacker meta:\n- Option A: 58.0%\n- Option B: 27.0%\n- Option C: 15.0%"
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined)

        result = parse_per_base_model_forecasts(comment, "multiple_choice")
        assert len(result) == 2
        assert result["gpt-5.5"] == {"Option A": 0.60, "Option B": 0.25, "Option C": 0.15}
        assert result["claude-opus-4.7"] == {"Option A": 0.55, "Option B": 0.30, "Option C": 0.15}

    def test_numeric_returns_empty_dict(self):
        base_preds = [
            _build_base_pred(
                "openrouter/openai/gpt-5.5",
                "analysis\n\nPercentile 50: 42",
            ),
        ]
        meta_text = "Stacker meta"
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined)

        result = parse_per_base_model_forecasts(comment, "numeric")
        assert result == {}

    def test_discrete_returns_empty_dict(self):
        base_preds = [
            _build_base_pred(
                "openrouter/openai/gpt-5.5",
                "analysis\n\nPercentile 50: 10",
            ),
        ]
        meta_text = "Stacker meta"
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined)

        result = parse_per_base_model_forecasts(comment, "discrete")
        assert result == {}

    def test_non_stacked_comment_returns_empty_dict(self):
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "analysis...\n\nProbability: 72%\n\n"
            "## R1: Forecaster 2 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "analysis...\n\nProbability: 68%\n"
        )
        result = parse_per_base_model_forecasts(comment, "binary")
        assert result == {}

    def test_excludes_stacker_meta_block(self):
        base_preds = [
            _build_base_pred("openrouter/openai/gpt-5.5", "Analysis.\n\nProbability: 72%"),
        ]
        meta_text = "Stacker analysis.\n\nProbability: 75%"
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined)

        result = parse_per_base_model_forecasts(comment, "binary")
        # Only the base model, NOT the stacker meta
        assert len(result) == 1
        assert "gpt-5.5" in result
        assert "Forecaster 1" not in result

    def test_empty_comment_returns_empty_dict(self):
        assert parse_per_base_model_forecasts("", "binary") == {}
        assert parse_per_base_model_forecasts("", "multiple_choice") == {}

    def test_binary_with_about_prefix(self):
        base_preds = [
            _build_base_pred("openrouter/openai/gpt-5.5", "Analysis.\n\nProbability: about 72%"),
        ]
        meta_text = "Meta."
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined)

        result = parse_per_base_model_forecasts(comment, "binary")
        assert result["gpt-5.5"] == "72.0%"


# ---------------------------------------------------------------------------
# Regression guard: the new RESEARCH-section blocks added this session
# (Prediction Market Snapshot, Provider Diagnostics, URL Context Fetches,
# financial_routing comment) must NOT bleed into the MC option-probability
# parsers.
#
# WHY THIS TEST EXISTS — the latent coupling
# ------------------------------------------
# ``_MC_OPTION_LINE_RE`` (parsing.py ~line 577) is
#     (?m)^[ \t]*-\s+(.+?):\s+([0-9]+(?:\.[0-9]+)?)\s*%
# which matches ANY ``- <name>: NN%`` bullet at column 0. The Prediction Market
# Snapshot block emits ``### Resolution criteria / rules`` bullets shaped EXACTLY
# like that — e.g. ``- **Polymarket** <url>: ...resolves YES if X... 55%``. That
# line WOULD be captured as a spurious MC option ("**Polymarket** <url>: ...resolves
# YES if X..." → 0.55) if the regex ever ran over it.
#
# It is safe TODAY only because the two MC-option parsers are SCOPED to sections
# OUTSIDE the research blob:
#   * ``parse_per_model_mc_option_probs`` scans only ``_summary_section_for_bullets``
#     — the prefix BEFORE ``### Research Summary`` — and only the region between
#     consecutive ``*Forecaster N*:`` bullets within it.
#   * ``parse_per_base_model_forecasts`` scans only the ``## R1:`` reasoning bodies
#     (via ``_iter_per_model_blocks``) and only returns non-empty when a
#     stacker-combined body is present.
# The snapshot lives in ``# RESEARCH`` (after ``### Research Summary``, before
# ``# FORECASTS``), so neither parser's scope reaches it.
#
# If a future change widens that scope, reorders the framework sections, or drops
# the ``### Research Summary`` boundary, the snapshot's ``: NN%`` bullets would be
# silently mis-parsed as MC option probabilities and corrupt MC calibration
# metrics. This test FAILS LOUDLY if that protection ever regresses.
# ---------------------------------------------------------------------------

# The booby-trap lines: Prediction Market Snapshot ``### Resolution criteria /
# rules`` bullets shaped exactly like an MC option line (``- <name>: NN%``).
# These sit inside the RESEARCH blob and must never surface as MC options.
_POLYMARKET_TRAP_LINE = "- **Polymarket** <https://polymarket.com/event/foo>: resolves YES if X happens: 55%"
_KALSHI_TRAP_LINE = "- **Kalshi** <https://kalshi.com/markets/bar>: market settles to the higher tier: 47%"
# Manifold's rules text is a USER-AUTHORED description, fetched per candidate by the detail
# enrichment step, and such descriptions routinely OPEN with a percentage. Both positions are
# pinned here because the obvious guess about this regex is WRONG: it is not anchored to
# end-of-line, and `(.+?)` is non-greedy, so a percentage ANYWHERE after the bullet's first
# colon matches — a leading one matches on the URL prefix alone. So there is no "safe" position
# for a percentage inside a rules bullet, and the section boundary is genuinely the only
# protection, not a belt on top of a shape rule.
_MANIFOLD_LEADING_PERCENT_TRAP_LINE = (
    "- **manifold** <https://manifold.markets/footyFan/brazil-2026>: 45% chance per the creator's own model"
)
_MANIFOLD_TRAILING_PERCENT_TRAP_LINE = (
    "- **manifold** <https://manifold.markets/footyFan/tier-2026>: resolves to the tier above: 45%"
)

# A realistic STACKED multiple-choice bot comment carrying ALL FOUR new research
# blocks. Section order mirrors the framework's unified layout
# (SUMMARY, then Research Summary, then RESEARCH, then FORECASTS) and the
# stacker-combined R1 body produced by combine_stacker_and_base_reasoning.
_MC_COMMENT_WITH_NEW_RESEARCH_BLOCKS = f"""# SUMMARY
*Question*: Which tier will the policy land in?
*Final Prediction*: Option A 58.0%, Option B 42.0%

## Report 1 Summary
### Forecasts
*Forecaster 1 (stacker-claude-opus-4.5)*:
- Option A: 58.0%
- Option B: 42.0%

### Research Summary
Stacked across the base models below.

# RESEARCH
## Report 1 Research
## News Articles (AskNews)
Recent coverage of the policy debate.

---

## Provider Diagnostics
- asknews: ok | 4210 chars | 8120 ms
- native_search: ok | 9044 chars | 41210 ms
- gemini_search: ok | 2210 chars | 15330 ms

---

## Prediction Market Snapshot
| platform | title | prob | vol | close | conf |
|---|---|---|---|---|---|
| Polymarket | Will the policy land in Tier A | 0.55 | 12000 | 2026-07-01 | 0.80 |
| Kalshi | Policy tier resolution | 0.47 | 4000 | 2026-07-01 | 0.62 |

##### Resolution criteria / rules
{_POLYMARKET_TRAP_LINE}
{_KALSHI_TRAP_LINE}
{_MANIFOLD_LEADING_PERCENT_TRAP_LINE}
{_MANIFOLD_TRAILING_PERCENT_TRAP_LINE}

---

## Web Research (Google Search via Gemini)
Gemini grounded results.

### URL Context Fetches
_url_context: none_

<!-- financial_routing: fred=[] tickers=['ACME'] extracted=['ACME'] unknown=[] -->

# FORECASTS
================================================================================
FORECAST SECTION:

## R1: Forecaster 1 Reasoning
{{combined_reasoning}}

<!-- STACKED=true -->
<!-- STACKER_OUTCOME=primary -->
"""


def _mc_stacked_comment_with_research_blocks() -> str:
    """Build the synthetic STACKED MC comment with the combined R1 body filled in.

    The two base models each emit a real MC option vector inside their reasoning
    body — that's the legitimate per-base-model content the parsers SHOULD find.
    The Prediction Market Snapshot's option-shaped rules lines live in the
    research blob and must NOT be picked up.
    """
    base_preds = [
        _build_base_pred(
            "openrouter/openai/gpt-5.5",
            "My MC analysis:\n- Option A: 60.0%\n- Option B: 40.0%",
        ),
        _build_base_pred(
            "openrouter/anthropic/claude-opus-4.7",
            "My MC analysis:\n- Option A: 56.0%\n- Option B: 44.0%",
        ),
    ]
    meta_text = "Stacker meta:\n- Option A: 58.0%\n- Option B: 42.0%"
    combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
    return _MC_COMMENT_WITH_NEW_RESEARCH_BLOCKS.format(combined_reasoning=combined)


class TestNewResearchBlocksDoNotLeakIntoMcParsers:
    """Lock down MC parsers against the option-shaped Prediction Market rules lines."""

    def test_snapshot_rules_lines_are_actually_mc_option_shaped(self):
        # Guard the premise of this whole class: confirm the trap lines really
        # DO match the MC-option regex. If the regex is ever rewritten so these
        # no longer match, the scoping protection is moot and this test (and the
        # ones below) should be revisited.

        assert _MC_OPTION_LINE_RE.search(_POLYMARKET_TRAP_LINE) is not None
        assert _MC_OPTION_LINE_RE.search(_KALSHI_TRAP_LINE) is not None
        assert _MC_OPTION_LINE_RE.search(_MANIFOLD_TRAILING_PERCENT_TRAP_LINE) is not None
        # A LEADING percentage is option-shaped TOO, which is the correction worth carrying: the
        # regex is unanchored at the end and non-greedy at the start, so it matches on the URL
        # prefix plus the first `: NN%` it can find. There is therefore no safe position for a
        # percentage in a rules bullet, venue rules text is deliberately NOT sanitized (a market
        # whose real rules read "Threshold: 45%" renders verbatim), and the `### Research Summary`
        # boundary asserted below is the ONLY thing standing between these lines and MC
        # calibration.
        assert _MC_OPTION_LINE_RE.search(_MANIFOLD_LEADING_PERCENT_TRAP_LINE) is not None

    def test_mc_option_probs_excludes_snapshot_rules_lines(self):
        comment = _mc_stacked_comment_with_research_blocks()
        result = parse_per_model_mc_option_probs(comment)

        # Only the stacker's summary-bullet option vector is returned.
        assert result == {"stacker-claude-opus-4.5": {"Option A": 0.58, "Option B": 0.42}}

        # No snapshot-derived option ever appears, under any key.
        for options in result.values():
            assert not any("Polymarket" in opt for opt in options)
            assert not any("Kalshi" in opt for opt in options)
            assert not any("manifold" in opt for opt in options)
            assert 0.55 not in options.values()
            assert 0.47 not in options.values()
            assert 0.45 not in options.values()

    def test_per_base_model_forecasts_recovers_only_true_base_models(self):
        comment = _mc_stacked_comment_with_research_blocks()
        result = parse_per_base_model_forecasts(comment, "multiple_choice")

        # Exactly the two real base models, with their R1-body option vectors.
        assert result == {
            "gpt-5.5": {"Option A": 0.60, "Option B": 0.40},
            "claude-opus-4.7": {"Option A": 0.56, "Option B": 0.44},
        }

        # The snapshot rules lines live in the research blob, not in any R1
        # reasoning body, so no Polymarket/Kalshi-keyed option leaks in.
        for options in result.values():
            assert isinstance(options, dict)
            assert not any("Polymarket" in opt for opt in options)
            assert not any("Kalshi" in opt for opt in options)
            assert not any("manifold" in opt for opt in options)

    def test_per_model_forecasts_unaffected_by_new_blocks(self):
        # The binary/summary per-model parser keys off the single stacker bullet
        # in the summary prefix; the new research blocks must not perturb it.
        comment = _mc_stacked_comment_with_research_blocks()
        result = parse_per_model_forecasts(comment)

        # The MC stacker bullet's first-line value is empty (options follow on
        # the next lines), so the captured value is empty — but the KEY must be
        # exactly the stacker, with no spurious snapshot-derived entries.
        assert set(result.keys()) == {"stacker-claude-opus-4.5"}
        assert "Polymarket" not in result
        assert "Kalshi" not in result

    def test_summary_boundary_is_what_protects_the_parsers(self):
        # Make the protection mechanism explicit: the snapshot trap lines must
        # fall AFTER the ``### Research Summary`` boundary, i.e. outside the
        # region the MC option parser scopes to. If a refactor ever moves the
        # boundary or the snapshot, this assertion flags it before the
        # mis-parse can happen.

        comment = _mc_stacked_comment_with_research_blocks()
        summary_prefix = _summary_section_for_bullets(comment)
        assert _POLYMARKET_TRAP_LINE not in summary_prefix
        assert _KALSHI_TRAP_LINE not in summary_prefix
        assert _MANIFOLD_TRAILING_PERCENT_TRAP_LINE not in summary_prefix
        # And the legitimate stacker bullet IS inside the scoped prefix.
        assert "*Forecaster 1 (stacker-claude-opus-4.5)*:" in summary_prefix


# ---------------------------------------------------------------------------
# Block-first parsing (2026-07 prompt change) — the forecaster prompts now emit
# forecast values ONLY inside the fenced ```json STRUCTURED FORECAST block; the
# trailing prose value lines are removed. Historical comments still carry the
# prose lines, so the reasoning-body parsers must try block-first per body and
# fall back to the prose regexes. The tests below exercise BOTH paths per
# function plus block-over-prose precedence when both are present.
# ---------------------------------------------------------------------------


def _fenced_json_block(payload: dict) -> str:
    return "```json\n" + json.dumps(payload) + "\n```"


def _numeric_block(percentiles: dict[str, float]) -> str:
    return _fenced_json_block({"question_type": "numeric", "declared_percentiles": percentiles})


def _binary_block(posterior_prob: float) -> str:
    return _fenced_json_block({"question_type": "binary", "posterior_prob": posterior_prob})


def _mc_block(option_probs: dict[str, float]) -> str:
    return _fenced_json_block({"question_type": "multiple_choice", "option_probs": option_probs})


class TestBlockFirstNumericPercentiles:
    """parse_per_model_numeric_percentiles: JSON block first, prose regex fallback."""

    def test_post_change_comment_parses_via_block(self):
        # No prose "Percentile X: V" lines at all — values live only in the block.
        block = _numeric_block({"0.025": 100.0, "0.1": 120.0, "0.5": 150.0, "0.9": 180.0})
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "analysis prose, no trailing value lines.\n\n"
            f"{block}\n"
        )
        result = parse_per_model_numeric_percentiles(comment)
        # Block decimals (0.025, 0.1, ...) are converted to the raw prose-label
        # convention (2.5, 10, ...) so downstream consumers see one shape.
        assert result == {"gpt-5.5": [(2.5, 100.0), (10.0, 120.0), (50.0, 150.0), (90.0, 180.0)]}

    def test_block_labels_are_exact_despite_float_noise(self):
        # 0.1 * 100 is 10.000000000000002 in IEEE 754 — the converter must
        # round so stacker_detection's label lookup (10.0/50.0/90.0) matches.
        block = _numeric_block({"0.1": 1.0, "0.5": 2.0, "0.9": 3.0})
        comment = f"## R1: Forecaster 1 Reasoning\nModel: openrouter/openai/gpt-5.5\n\n{block}\n"
        result = parse_per_model_numeric_percentiles(comment)
        assert result == {"gpt-5.5": [(10.0, 1.0), (50.0, 2.0), (90.0, 3.0)]}

    def test_pre_change_comment_still_parses_via_prose(self):
        # Historical shape: prose lines, no JSON block.
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "analysis...\n\n"
            "Percentile 10: 120.0\n"
            "Percentile 50: 150.0\n"
            "Percentile 90: 180.0\n"
        )
        result = parse_per_model_numeric_percentiles(comment)
        assert result == {"gpt-5.5": [(10.0, 120.0), (50.0, 150.0), (90.0, 180.0)]}

    def test_block_wins_over_prose_when_both_present(self):
        # Transition-era comment: block AND prose lines. Block values differ so
        # the assertion proves precedence, not coincidence.
        block = _numeric_block({"0.1": 125.0, "0.5": 155.0, "0.9": 185.0})
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            f"{block}\n\n"
            "Percentile 10: 120.0\n"
            "Percentile 50: 150.0\n"
            "Percentile 90: 180.0\n"
        )
        result = parse_per_model_numeric_percentiles(comment)
        assert result == {"gpt-5.5": [(10.0, 125.0), (50.0, 155.0), (90.0, 185.0)]}

    def test_invalid_block_falls_back_to_prose(self):
        # A fenced json block that fails schema validation (missing the
        # required 0.1/0.5/0.9 percentiles) must not eat the prose lines.
        bad_block = _fenced_json_block({"question_type": "numeric", "declared_percentiles": {"0.5": 150.0}})
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            f"{bad_block}\n\n"
            "Percentile 10: 120.0\n"
            "Percentile 50: 150.0\n"
            "Percentile 90: 180.0\n"
        )
        result = parse_per_model_numeric_percentiles(comment)
        assert result == {"gpt-5.5": [(10.0, 120.0), (50.0, 150.0), (90.0, 180.0)]}

    def test_stacked_comment_mixed_eras_per_base_model(self):
        # One base model is post-change (block only), one pre-change (prose
        # only) — per-body try-then-fallback must handle both in one comment.
        block_body = "block-era analysis.\n\n" + _numeric_block({"0.1": 10.0, "0.5": 20.0, "0.9": 30.0})
        prose_body = "prose-era analysis.\n\nPercentile 10: 11\nPercentile 50: 21\nPercentile 90: 31"
        base_preds = [
            _build_base_pred("openrouter/openai/gpt-5.5", block_body),
            _build_base_pred("openrouter/anthropic/claude-opus-4.7", prose_body),
        ]
        combined = combine_stacker_and_base_reasoning("stacker meta, no values", base_preds)
        comment = _build_stacked_comment(combined)
        result = parse_per_model_numeric_percentiles(comment)
        assert result["gpt-5.5"] == [(10.0, 10.0), (50.0, 20.0), (90.0, 30.0)]
        assert result["claude-opus-4.7"] == [(10.0, 11.0), (50.0, 21.0), (90.0, 31.0)]


class TestBlockFirstPerBaseModelBinary:
    """parse_per_base_model_forecasts binary branch: block first, prose fallback."""

    def test_post_change_stacked_comment_parses_via_block(self):
        base_preds = [
            _build_base_pred("openrouter/openai/gpt-5.5", "analysis, no prose value.\n\n" + _binary_block(0.62)),
            _build_base_pred(
                "openrouter/anthropic/claude-opus-4.7", "analysis, no prose value.\n\n" + _binary_block(0.57)
            ),
        ]
        combined = combine_stacker_and_base_reasoning("stacker meta", base_preds)
        comment = _build_stacked_comment(combined)
        result = parse_per_base_model_forecasts(comment, "binary")
        # Same "{prob*100:.1f}%" shape the prose path produces.
        assert result == {"gpt-5.5": "62.0%", "claude-opus-4.7": "57.0%"}

    def test_block_wins_over_prose_when_both_present(self):
        body = "Initial thoughts.\n\nProbability: 55%\n\n" + _binary_block(0.62)
        base_preds = [_build_base_pred("openrouter/openai/gpt-5.5", body)]
        combined = combine_stacker_and_base_reasoning("stacker meta", base_preds)
        comment = _build_stacked_comment(combined)
        result = parse_per_base_model_forecasts(comment, "binary")
        assert result == {"gpt-5.5": "62.0%"}

    def test_invalid_block_falls_back_to_prose(self):
        # posterior_prob out of range fails validation → prose line still read.
        bad_block = _fenced_json_block({"question_type": "binary", "posterior_prob": 1.7})
        body = "Analysis.\n\nProbability: 55%\n\n" + bad_block
        base_preds = [_build_base_pred("openrouter/openai/gpt-5.5", body)]
        combined = combine_stacker_and_base_reasoning("stacker meta", base_preds)
        comment = _build_stacked_comment(combined)
        result = parse_per_base_model_forecasts(comment, "binary")
        assert result == {"gpt-5.5": "55.0%"}

    def test_pre_change_prose_only_unchanged(self):
        base_preds = [_build_base_pred("openrouter/openai/gpt-5.5", "Analysis.\n\nProbability: 72%")]
        combined = combine_stacker_and_base_reasoning("stacker meta", base_preds)
        comment = _build_stacked_comment(combined)
        result = parse_per_base_model_forecasts(comment, "binary")
        assert result == {"gpt-5.5": "72.0%"}


class TestBlockFirstPerBaseModelMultipleChoice:
    """parse_per_base_model_forecasts MC branch: block first, prose fallback."""

    def test_post_change_stacked_comment_parses_via_block(self):
        base_preds = [
            _build_base_pred(
                "openrouter/openai/gpt-5.5",
                "analysis, no option lines.\n\n" + _mc_block({"Option A": 0.61, "Option B": 0.39}),
            ),
        ]
        combined = combine_stacker_and_base_reasoning("stacker meta", base_preds)
        comment = _build_stacked_comment(combined)
        result = parse_per_base_model_forecasts(comment, "multiple_choice")
        assert result == {"gpt-5.5": {"Option A": 0.61, "Option B": 0.39}}

    def test_block_wins_over_prose_when_both_present(self):
        body = "My analysis:\n- Option A: 55.0%\n- Option B: 45.0%\n\n" + _mc_block(
            {"Option A": 0.61, "Option B": 0.39}
        )
        base_preds = [_build_base_pred("openrouter/openai/gpt-5.5", body)]
        combined = combine_stacker_and_base_reasoning("stacker meta", base_preds)
        comment = _build_stacked_comment(combined)
        result = parse_per_base_model_forecasts(comment, "multiple_choice")
        assert result == {"gpt-5.5": {"Option A": 0.61, "Option B": 0.39}}

    def test_invalid_block_falls_back_to_prose(self):
        # option_probs not summing to ~1.0 fails validation → prose lines used.
        bad_block = _fenced_json_block(
            {"question_type": "multiple_choice", "option_probs": {"Option A": 0.5, "Option B": 0.1}}
        )
        body = "My analysis:\n- Option A: 55.0%\n- Option B: 45.0%\n\n" + bad_block
        base_preds = [_build_base_pred("openrouter/openai/gpt-5.5", body)]
        combined = combine_stacker_and_base_reasoning("stacker meta", base_preds)
        comment = _build_stacked_comment(combined)
        result = parse_per_base_model_forecasts(comment, "multiple_choice")
        assert result == {"gpt-5.5": {"Option A": 0.55, "Option B": 0.45}}


class TestMcSummaryBulletsStayProseOnly:
    """parse_per_model_mc_option_probs reads bot-RENDERED summary bullets.

    The framework formats each parsed prediction via
    MultipleChoiceReport.make_readable_prediction into ``- {option}: {prob}%``
    lines regardless of what the model wrote in its rationale, so the summary
    region keeps its prose form after the block-only prompt change. The parser
    must keep reading the bullets and must NOT be diverted by a JSON block
    living in the R1 reasoning body.
    """

    def test_summary_bullets_parsed_even_when_r1_body_is_block_only(self):
        r1_block = _mc_block({"Option A": 0.99, "Option B": 0.01})  # decoy values
        comment = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: \n"
            "- Option A: 40.0%\n"
            "- Option B: 60.0%\n"
            "\n"
            "### Research Summary\n"
            "stuff\n\n"
            "================================================================================\n"
            "FORECAST SECTION:\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            f"block-era rationale.\n\n{r1_block}\n"
        )
        result = parse_per_model_mc_option_probs(comment)
        # Summary bullets (the bot's rendered values), NOT the R1 body's block.
        assert result == {"gpt-5.5": {"Option A": 0.40, "Option B": 0.60}}


# ---------------------------------------------------------------------------
# Tolerant raw-JSON salvage rung (strict block > prose > tolerant).
#
# Historical 2026-05/06-era blocks carry retired tier-2 scaffold fields
# (mixture_components, tails, distribution_family_hint), which is what still needs
# this rung. The other historical offender, an edge-value concentration=0.0, is
# now read as absent by the strict MC schema (2026-09-02, when the field left the
# prompt), so those comments resolve one rung earlier. parse_structured_block's
# extra="forbid" schemas return None for the whole block, and block-only
# rationales (gemini-3.1-pro in that era) have no prose value lines either —
# so those models silently vanished from recovered per-model data. That parse
# loss produced the false "gemini missed 5 of 45 summer questions" finding in
# the 2026-07-15 ensemble screening (4 of the 5 were recovery artifacts; only
# qid 44136 was a real soft-deadline drop). The payload shapes below are taken
# verbatim from the affected comments (qids 43652/43746/43656).
# ---------------------------------------------------------------------------


class TestTolerantBlockSalvage:
    def test_numeric_block_with_retired_tier2_fields_salvaged(self):
        # Gemini's actual Q43652-era block shape: strict-invalid (retired
        # fields + only 3 declared percentiles), and NO prose value lines.
        block = _fenced_json_block(
            {
                "question_type": "numeric",
                "declared_percentiles": {"0.1": -15.5, "0.5": -0.2, "0.9": 16.5},
                "distribution_family_hint": "normal",
                "tails": {"below_min_expected": 0.05, "above_max_expected": 0.06},
                "mixture_components": [{"weight": 1.0, "mean": -0.5, "sd": 9.0}],
            }
        )
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/google/gemini-3.1-pro-preview\n\n"
            f"block-only rationale, no prose lines.\n\n{block}\n"
        )
        result = parse_per_model_numeric_percentiles(comment)
        assert result == {"gemini-3.1-pro-preview": [(10.0, -15.5), (50.0, -0.2), (90.0, 16.5)]}

    def test_numeric_prose_still_wins_over_tolerant_salvage(self):
        # Transition-era: strict-invalid block AND prose lines. Prose keeps
        # precedence (tolerant is strictly the LAST rung).
        block = _fenced_json_block(
            {
                "question_type": "numeric",
                "declared_percentiles": {"0.1": 999.0, "0.5": 1000.0, "0.9": 1001.0},
                "mixture_components": [],
            }
        )
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            f"{block}\n\n"
            "Percentile 10: 120.0\n"
            "Percentile 50: 150.0\n"
            "Percentile 90: 180.0\n"
        )
        result = parse_per_model_numeric_percentiles(comment)
        assert result == {"gpt-5.5": [(10.0, 120.0), (50.0, 150.0), (90.0, 180.0)]}

    def test_numeric_garbage_keys_and_nonfinite_values_skipped(self):
        block = _fenced_json_block(
            {
                "question_type": "numeric",
                "declared_percentiles": {"0.5": 150.0, "not-a-pct": 1.0, "1.5": 2.0},
                "tails": {},
            }
        )
        comment = f"## R1: Forecaster 1 Reasoning\nModel: openrouter/openai/gpt-5.5\n\n{block}\n"
        result = parse_per_model_numeric_percentiles(comment)
        assert result == {"gpt-5.5": [(50.0, 150.0)]}

    def test_mc_block_with_zero_concentration_recovers(self):
        # Gemini's actual Q43656-era MC block. concentration=0.0 used to fail the strict
        # >0 validator, which pushed this comment onto the tolerant rung (its trailing
        # prose lines lack the "- " bullet prefix the regex fallback needs); since
        # 2026-09-02 the strict schema reads the value as absent and resolves it directly.
        # Either way the recovered ballot is the same, which is what this pins.
        block = _fenced_json_block(
            {
                "question_type": "multiple_choice",
                "option_probs": {"Option A": 0.6, "Option B": 0.4},
                "other_mass": 0.0,
                "concentration": 0.0,
            }
        )
        base_preds = [
            _build_base_pred(
                "openrouter/google/gemini-3.1-pro-preview",
                f"block-only rationale.\n\n{block}\n\nOption A: 60%\nOption B: 40%",
            ),
        ]
        combined = combine_stacker_and_base_reasoning("stacker meta", base_preds)
        comment = _build_stacked_comment(combined)
        result = parse_per_base_model_forecasts(comment, "multiple_choice")
        assert result == {"gemini-3.1-pro-preview": {"Option A": 0.6, "Option B": 0.4}}

    def test_binary_block_with_retired_scenario_shape_salvaged(self):
        # Old scenarios shape (description/conditional_yes instead of
        # name/probability) fails extra="forbid"; posterior_prob is salvaged.
        block = _fenced_json_block(
            {
                "question_type": "binary",
                "posterior_prob": 0.62,
                "scenarios": [{"description": "conviction upheld", "conditional_yes": 0.98}],
            }
        )
        base_preds = [
            _build_base_pred("openrouter/google/gemini-3.1-pro-preview", f"block-only rationale.\n\n{block}"),
        ]
        combined = combine_stacker_and_base_reasoning("stacker meta", base_preds)
        comment = _build_stacked_comment(combined)
        result = parse_per_base_model_forecasts(comment, "binary")
        assert result == {"gemini-3.1-pro-preview": "62.0%"}

    def test_conflicting_question_type_not_salvaged(self):
        # A binary-typed block in a numeric parse context must not be salvaged.
        block = _fenced_json_block({"question_type": "binary", "posterior_prob": 0.5})
        comment = f"## R1: Forecaster 1 Reasoning\nModel: openrouter/openai/gpt-5.5\n\n{block}\n"
        assert parse_per_model_numeric_percentiles(comment) == {}

    def test_malformed_json_not_salvaged(self):
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            '```json\n{"question_type": "numeric", "declared_percentiles": {"0.5": \n```\n'
        )
        assert parse_per_model_numeric_percentiles(comment) == {}


# ---------------------------------------------------------------------------
# Double-scaled percentile-label guard (_validate_percentile_labels).
#
# parse_per_model_numeric_percentiles emits labels in the PERCENT convention
# (2.5, 5, ..., 97.5, 99) — always strictly inside (0, 100). A forecaster that
# writes its block keys in percent form (2.5, 5, ...) instead of the decimal
# convention (0.025, 0.05, ...) has them multiplied by 100 by the block readers,
# producing double-scaled labels (250, 500, ..., 9750). That was the qid=43684
# grok-4.3 record the 2026-07-15 coherence study had to hand-drop. The guard
# passes clean labels through, deterministically rescales an exact
# canonical-set*100 double-scaling, and rejects any other out-of-range set.
# ---------------------------------------------------------------------------

# Canonical percent-form label sets (mirrors STANDARD_PERCENTILES * 100).
_CANONICAL_LABELS_11 = [2.5, 5.0, 10.0, 20.0, 40.0, 50.0, 60.0, 80.0, 90.0, 95.0, 97.5]
_CANONICAL_LABELS_13 = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 50.0, 60.0, 80.0, 90.0, 95.0, 97.5, 99.0]


def _pairs(labels: list[float]) -> list[tuple[float, float]]:
    """Build (label, value) pairs with monotone-increasing dummy values."""
    return [(label, float(i)) for i, label in enumerate(labels)]


def _prose_percentile_lines(labels: list[float], values: list[float]) -> str:
    """Render ``Percentile L: V`` prose lines (labels emitted without trailing .0)."""
    return "\n".join(f"Percentile {label:g}: {value}" for label, value in zip(labels, values, strict=True))


class TestValidatePercentileLabels:
    """Unit coverage for the three branches of the label guard."""

    def test_valid_11_point_set_passes_through_unchanged(self):
        pairs = _pairs(_CANONICAL_LABELS_11)
        assert _validate_percentile_labels(pairs, model="gpt-5.5") == pairs

    def test_valid_13_point_set_passes_through_unchanged(self):
        pairs = _pairs(_CANONICAL_LABELS_13)
        assert _validate_percentile_labels(pairs, model="gpt-5.5") == pairs

    def test_exact_11_point_x100_is_rescaled(self):
        double_scaled = _pairs([label * 100.0 for label in _CANONICAL_LABELS_11])
        result = _validate_percentile_labels(double_scaled, model="grok-4.3", question_id=43684)
        assert result == _pairs(_CANONICAL_LABELS_11)

    def test_exact_13_point_x100_is_rescaled(self):
        double_scaled = _pairs([label * 100.0 for label in _CANONICAL_LABELS_13])
        result = _validate_percentile_labels(double_scaled, model="gpt-5.5")
        assert result == _pairs(_CANONICAL_LABELS_13)

    def test_out_of_range_non_canonical_set_rejected(self):
        # Labels out of (0, 100) whose /100 is not a canonical set → drop.
        garbage = [(250.0, 1.0), (777.0, 2.0), (9999.0, 3.0)]
        assert _validate_percentile_labels(garbage, model="gpt-5.5") is None

    def test_label_at_zero_boundary_rejected(self):
        # 0 is not strictly inside (0, 100); /100 = {0.0, ...} is not canonical.
        with_zero = [(0.0, 1.0), (50.0, 2.0), (97.5, 3.0)]
        assert _validate_percentile_labels(with_zero, model="gpt-5.5") is None

    def test_label_at_hundred_boundary_rejected(self):
        with_hundred = [(2.5, 1.0), (50.0, 2.0), (100.0, 3.0)]
        assert _validate_percentile_labels(with_hundred, model="gpt-5.5") is None

    def test_partial_double_scaling_rejected_not_rescaled(self):
        # A mix of correct (2.5) and double-scaled (500) labels does not /100 to
        # a canonical set, so it must be rejected rather than silently rescaled.
        mixed = [(2.5, 1.0), (500.0, 2.0), (97.5, 3.0)]
        assert _validate_percentile_labels(mixed, model="gpt-5.5") is None

    def test_rescale_emits_warning_naming_model_and_labels(self, caplog):
        double_scaled = _pairs([label * 100.0 for label in _CANONICAL_LABELS_11])
        with caplog.at_level(logging.WARNING):
            _validate_percentile_labels(double_scaled, model="grok-4.3", question_id=43684)
        messages = [r.getMessage() for r in caplog.records]
        assert any("Rescaling" in m and "grok-4.3" in m and "43684" in m for m in messages)

    def test_reject_emits_warning_naming_model_and_labels(self, caplog):
        with caplog.at_level(logging.WARNING):
            _validate_percentile_labels([(250.0, 1.0), (777.0, 2.0)], model="gpt-5.5")
        messages = [r.getMessage() for r in caplog.records]
        assert any("Rejecting" in m and "gpt-5.5" in m and "250.0" in m for m in messages)


class TestParsePerModelNumericPercentilesLabelGuard:
    """End-to-end: the label guard fires inside parse_per_model_numeric_percentiles."""

    def test_qid_43684_real_shape_passes_through_via_prose(self):
        # Faithful reproduction of the qid=43684 grok-4.3 comment_archive record:
        # a numeric block whose keys are in PERCENT form (2.5, 5, ...) plus the
        # matching prose value lines. The strict schema rejects the block (keys
        # > 1, missing 0.1/0.5/0.9, retired fields) AND the tolerant salvage rung
        # drops it (its keys fail the 0<pct<1 guard), so the prose lines win and
        # yield clean in-range labels — the PASSTHROUGH branch, NOT rescale. The
        # 250..9750 double-scaling only ever arose in the coherence study's own
        # guard-less raw-JSON reader, never in this production parser.
        labels = _CANONICAL_LABELS_11
        values = [-10.0, -8.0, -6.0, -3.5, -0.5, 1.0, 2.5, 5.5, 8.0, 10.0, 12.0]
        block = _fenced_json_block(
            {
                "question_type": "numeric",
                # Percent-form keys, exactly as grok-4.3 emitted them.
                "declared_percentiles": {f"{label:g}": value for label, value in zip(labels, values, strict=True)},
                "distribution_family_hint": "normal",
                "tails": {"below_min_expected": 0.03, "above_max_expected": 0.03},
                "mixture_components": None,
            }
        )
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/x-ai/grok-4.3\n\n"
            f"analysis prose.\n\n{block}\n\n{_prose_percentile_lines(labels, values)}\n"
        )
        result = parse_per_model_numeric_percentiles(comment, question_id=43684)
        assert result == {"grok-4.3": list(zip(labels, values, strict=True))}

    def test_double_scaled_prose_labels_are_rescaled(self):
        # If the double-scaled labels DO reach extraction (a model emitting
        # ``Percentile 250: V`` prose), the guard rescales the exact 11-point
        # canonical-set*100 back to canonical.
        canonical_values = [-10.0, -8.0, -6.0, -3.5, -0.5, 1.0, 2.5, 5.5, 8.0, 10.0, 12.0]
        double_scaled_labels = [label * 100.0 for label in _CANONICAL_LABELS_11]
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/x-ai/grok-4.3\n\n"
            f"analysis.\n\n{_prose_percentile_lines(double_scaled_labels, canonical_values)}\n"
        )
        result = parse_per_model_numeric_percentiles(comment, question_id=43684)
        assert result == {"grok-4.3": list(zip(_CANONICAL_LABELS_11, canonical_values, strict=True))}

    def test_genuine_garbage_labels_drop_the_model(self):
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "Percentile 250: 1\nPercentile 777: 2\nPercentile 9999: 3\n"
        )
        assert parse_per_model_numeric_percentiles(comment) == {}

    def test_clean_prose_unaffected_by_guard(self):
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "Percentile 2.5: 100.0\nPercentile 50: 150.0\nPercentile 97.5: 200.0\n"
        )
        result = parse_per_model_numeric_percentiles(comment)
        assert result == {"gpt-5.5": [(2.5, 100.0), (50.0, 150.0), (97.5, 200.0)]}
