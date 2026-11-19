"""Tests for metaculus_bot.performance_analysis.stacker_detection module.

Verifies the multi-signal stacker-detection helper that combines explicit flags,
body markers, spread thresholds, and production-vs-median deltas to classify
whether a given historical record was stacked.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from metaculus_bot.performance_analysis.stacker_detection import (
    compute_production_vs_median_delta,
    detect_stacker_fired,
    exceeded_spread_threshold,
    get_stacker_outcome_field,
    has_stacker_body_marker,
    has_was_stacked_flag,
    member_spread,
)

# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestHasWasStackedFlag:
    def test_true(self):
        assert has_was_stacked_flag({"was_stacked": True}) is True

    def test_false(self):
        assert has_was_stacked_flag({"was_stacked": False}) is False

    def test_missing(self):
        assert has_was_stacked_flag({"question_id": 123}) is None

    def test_none_value(self):
        assert has_was_stacked_flag({"was_stacked": None}) is None


class TestGetStackerOutcomeField:
    def test_primary(self):
        assert get_stacker_outcome_field({"stacker_outcome": "primary"}) == "primary"

    def test_fallback_llm(self):
        assert get_stacker_outcome_field({"stacker_outcome": "fallback_llm"}) == "fallback_llm"

    def test_fallback_median(self):
        assert get_stacker_outcome_field({"stacker_outcome": "fallback_median"}) == "fallback_median"

    def test_skipped(self):
        assert get_stacker_outcome_field({"stacker_outcome": "skipped"}) == "skipped"

    def test_missing(self):
        assert get_stacker_outcome_field({"question_id": 123}) is None

    def test_none_value(self):
        assert get_stacker_outcome_field({"stacker_outcome": None}) is None

    def test_empty_string(self):
        assert get_stacker_outcome_field({"stacker_outcome": ""}) is None


class TestHasStackerBodyMarker:
    def test_stacker_outcome_marker(self):
        record = {"comment_text": "some text <!-- STACKER_OUTCOME=primary --> more text"}
        assert has_stacker_body_marker(record) is True

    def test_stacker_outcome_fallback_llm(self):
        record = {"comment_text": "text <!-- STACKER_OUTCOME=fallback_llm --> end"}
        assert has_stacker_body_marker(record) is True

    def test_stacker_outcome_skipped(self):
        # skipped means it did NOT fire — this should return False
        record = {"comment_text": "text <!-- STACKER_OUTCOME=skipped --> end"}
        assert has_stacker_body_marker(record) is False

    def test_stacker_outcome_skipped_config_off(self):
        # config-suppressed skip (per-type gate off despite high spread) —
        # the stacker did NOT fire, so this must classify as median.
        record = {"comment_text": "text <!-- STACKER_OUTCOME=skipped_config_off --> end"}
        assert has_stacker_body_marker(record) is False

    def test_stacker_outcome_fallback_median(self):
        # fallback_median means stacker failed, fell back to median — NOT stacker
        record = {"comment_text": "text <!-- STACKER_OUTCOME=fallback_median --> end"}
        assert has_stacker_body_marker(record) is False

    def test_stacked_true_legacy(self):
        record = {"comment_text": "text <!-- STACKED=true --> end"}
        assert has_stacker_body_marker(record) is True

    def test_stacked_false_legacy(self):
        record = {"comment_text": "text <!-- STACKED=false --> end"}
        assert has_stacker_body_marker(record) is False

    def test_tools_used_alone_not_stacker(self):
        # TOOLS_USED alone doesn't indicate stacking
        record = {"comment_text": "text <!-- TOOLS_USED=true --> end"}
        assert has_stacker_body_marker(record) is None

    def test_historical_body_signature(self):
        record = {
            "comment_text": (
                "## R1: Forecaster 1 Reasoning\n"
                "Model: openrouter/anthropic/claude-opus-4.5\n"
                "## Stacker Meta-Analysis\n"
                "The models disagree..."
            )
        }
        assert has_stacker_body_marker(record) is True

    def test_no_marker(self):
        record = {"comment_text": "just a regular comment with no markers"}
        assert has_stacker_body_marker(record) is None

    def test_missing_comment_text(self):
        record = {"question_id": 123}
        assert has_stacker_body_marker(record) is None


class TestComputeProductionVsMedianDelta:
    def test_binary_record(self):
        record = {
            "type": "binary",
            "our_prob_yes": 0.70,
            "per_model_forecasts": {
                "gpt-5.5": "60%",
                "claude-opus-4.7": "80%",
                "gemini-3.1-pro": "70%",
            },
        }
        # median of [0.6, 0.7, 0.8] = 0.7, delta = |0.70 - 0.70| = 0.0
        assert compute_production_vs_median_delta(record) == pytest.approx(0.0, abs=1e-6)

    def test_binary_record_with_delta(self):
        record = {
            "type": "binary",
            "our_prob_yes": 0.85,
            "per_model_forecasts": {
                "gpt-5.5": "60%",
                "claude-opus-4.7": "80%",
                "gemini-3.1-pro": "70%",
            },
        }
        # median of [0.6, 0.7, 0.8] = 0.7, delta = |0.85 - 0.70| = 0.15
        assert compute_production_vs_median_delta(record) == pytest.approx(0.15, abs=1e-6)

    def test_missing_production_value(self):
        record = {
            "type": "binary",
            "per_model_forecasts": {"gpt-5.5": "60%", "claude-opus-4.7": "80%"},
        }
        assert compute_production_vs_median_delta(record) is None

    def test_insufficient_models(self):
        record = {
            "type": "binary",
            "our_prob_yes": 0.70,
            "per_model_forecasts": {"gpt-5.5": "60%"},
        }
        assert compute_production_vs_median_delta(record) is None

    def test_numeric_record_returns_none_when_no_cdf(self):
        # Numeric delta requires CDF comparison which is complex; without
        # our_forecast_values it should return None
        record = {
            "type": "numeric",
            "per_model_numeric_percentiles": {"a": [[2.5, 10], [50, 50], [97.5, 90]]},
        }
        assert compute_production_vs_median_delta(record) is None


class TestComputeProductionVsMedianDeltaBaseModelField:
    """Tests for per_base_model_forecasts preference in compute_production_vs_median_delta."""

    def test_prefers_per_base_model_forecasts_over_collapsed_stacker(self):
        """Stacked record: per_model_forecasts has only the stacker singleton,
        per_base_model_forecasts has the original base models."""
        record = {
            "type": "binary",
            "our_prob_yes": 0.85,
            "per_model_forecasts": {"stacker": "85%"},
            "per_base_model_forecasts": {"gpt-5.5": "60%", "claude": "80%", "gemini": "70%"},
        }
        # median of base [0.6, 0.7, 0.8] = 0.7, delta = |0.85 - 0.70| = 0.15
        assert compute_production_vs_median_delta(record) == pytest.approx(0.15, abs=1e-6)

    def test_empty_per_base_model_falls_back_to_per_model(self):
        """Empty per_base_model_forecasts should fall back to per_model_forecasts."""
        record = {
            "type": "binary",
            "our_prob_yes": 0.85,
            "per_model_forecasts": {"a": "60%", "b": "80%", "c": "70%"},
            "per_base_model_forecasts": {},
        }
        # median of [0.6, 0.7, 0.8] = 0.7, delta = |0.85 - 0.70| = 0.15
        assert compute_production_vs_median_delta(record) == pytest.approx(0.15, abs=1e-6)

    def test_missing_per_base_model_falls_back_to_per_model(self):
        """Older records without per_base_model_forecasts still work."""
        record = {
            "type": "binary",
            "our_prob_yes": 0.85,
            "per_model_forecasts": {"a": "60%", "b": "80%", "c": "70%"},
        }
        assert compute_production_vs_median_delta(record) == pytest.approx(0.15, abs=1e-6)

    def test_stacker_singleton_without_base_returns_none(self):
        """Stacked record with collapsed singleton and no base field → None (< 2 probs)."""
        record = {
            "type": "binary",
            "our_prob_yes": 0.85,
            "per_model_forecasts": {"stacker": "85%"},
        }
        assert compute_production_vs_median_delta(record) is None


class TestExceededSpreadThreshold:
    def test_binary_high_spread(self):
        record = {
            "type": "binary",
            "per_model_forecasts": {
                "gpt-5.5": "90%",
                "claude-opus-4.7": "50%",
                "gemini-3.1-pro": "70%",
            },
        }
        # spread = 0.9 - 0.5 = 0.4 > 0.15
        assert exceeded_spread_threshold(record) is True

    def test_binary_low_spread(self):
        record = {
            "type": "binary",
            "per_model_forecasts": {
                "gpt-5.5": "70%",
                "claude-opus-4.7": "75%",
                "gemini-3.1-pro": "72%",
            },
        }
        # spread = 0.75 - 0.70 = 0.05 < 0.15
        assert exceeded_spread_threshold(record) is False

    def test_binary_at_threshold(self):
        # Use values that compute to exactly the threshold in float
        # 0.70 - 0.55 = 0.15000000000000002 in float64 (> threshold due to precision)
        # In production, the strict > comparison means float-precision boundary cases trigger.
        # Use values that clearly don't exceed: 0.72 - 0.58 = 0.14
        record = {
            "type": "binary",
            "per_model_forecasts": {
                "gpt-5.5": "58%",
                "claude-opus-4.7": "72%",
            },
        }
        # spread = 0.72 - 0.58 = 0.14 < 0.15
        assert exceeded_spread_threshold(record) is False

    def test_missing_per_model(self):
        record = {"type": "binary"}
        assert exceeded_spread_threshold(record) is None

    def test_mc_high_spread(self):
        # MC option vectors live in per_model_forecasts as {model: {option: prob}}
        # (what collector._process_post emits via parse_per_model_mc_option_probs).
        record = {
            "type": "multiple_choice",
            "per_model_forecasts": {
                "gpt-5.5": {"A": 0.8, "B": 0.2},
                "claude-opus-4.7": {"A": 0.5, "B": 0.5},
            },
        }
        # max option spread: A = 0.8-0.5=0.3, B = 0.5-0.2=0.3. max=0.3 > 0.20
        assert exceeded_spread_threshold(record) is True

    def test_mc_low_spread(self):
        record = {
            "type": "multiple_choice",
            "per_model_forecasts": {
                "gpt-5.5": {"A": 0.6, "B": 0.4},
                "claude-opus-4.7": {"A": 0.65, "B": 0.35},
            },
        }
        # max option spread: A = 0.65-0.6=0.05, B = 0.4-0.35=0.05. max=0.05 < 0.20
        assert exceeded_spread_threshold(record) is False

    def test_mc_prefers_per_base_model_forecasts_on_stacked_records(self):
        # On stacked records per_model_forecasts collapses to the stacker's
        # single aggregate; per_base_model_forecasts carries the real base-model
        # option vectors and must win (mirrors the binary branch).
        record = {
            "type": "multiple_choice",
            "per_model_forecasts": {
                "stacker": {"A": 0.65, "B": 0.35},
            },
            "per_base_model_forecasts": {
                "gpt-5.5": {"A": 0.8, "B": 0.2},
                "claude-opus-4.7": {"A": 0.5, "B": 0.5},
            },
        }
        # base-model spread: A = 0.3 > 0.20; the collapsed per_model_forecasts
        # alone would return None (only one model).
        assert exceeded_spread_threshold(record) is True

    def test_mc_string_forecasts_return_none(self):
        # If the MC option parser found nothing, the collector falls back to
        # single-string forecasts — no option vectors to compute a spread from.
        record = {
            "type": "multiple_choice",
            "per_model_forecasts": {
                "gpt-5.5": "A (60%)",
                "claude-opus-4.7": "A (55%)",
            },
        }
        assert exceeded_spread_threshold(record) is None

    def test_numeric_high_spread_closed_bounds(self):
        # Standard 11-percentile lists (raw 0-100 labels) on a [0, 100] range.
        # P10: |40-20|/100=0.20; P50: |60-40|/100=0.20; P90: |80-60|/100=0.20 > 0.15.
        std = [2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5]
        model_a = [[p, v] for p, v in zip(std, [10, 15, 20, 25, 35, 40, 45, 55, 60, 65, 70], strict=True)]
        model_b = [[p, v] for p, v in zip(std, [30, 35, 40, 45, 55, 60, 65, 75, 80, 85, 90], strict=True)]
        record = {
            "type": "numeric",
            "per_model_numeric_percentiles": {"a": model_a, "b": model_b},
            "scaling": {"range_min": 0.0, "range_max": 100.0},
        }
        assert exceeded_spread_threshold(record) is True

    def test_numeric_restated_sparse_curve_does_not_clear_the_anchor_floor(self):
        # MIN_SCOREABLE_ANCHORS counts DISTINCT labels everywhere else (via
        # declared_anchors); gating on the raw PAIR count here let a 5-anchor curve
        # restated twice (10 pairs — comment prose restates whole sets) pose as a
        # scoreable member. With the sparse member excluded only one map remains, no
        # spread is computable, and the verdict is None rather than a measured spread.
        std = [2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5]
        dense = [[p, v] for p, v in zip(std, [10, 15, 20, 25, 35, 40, 45, 55, 60, 65, 70], strict=True)]
        sparse_once = [[10.0, 40.0], [25.0, 50.0], [50.0, 60.0], [75.0, 70.0], [90.0, 80.0]]
        record = {
            "type": "numeric",
            "per_model_numeric_percentiles": {"a": dense, "b": sparse_once * 2},
            "scaling": {"range_min": 0.0, "range_max": 100.0},
        }
        assert exceeded_spread_threshold(record) is None

    def test_numeric_spread_is_percentile_label_invariant(self):
        # Regression: the numeric spread must key P10/P50/P90 by LABEL, not by the
        # positional index [2, 5, 8]. Prepend P1 and append P99 (the future 13-set)
        # so positional [2,5,8] would read P5/P40/P80 instead. Only the P50 value
        # differs from its neighbors here, so a positional bug (reading P40) would
        # compute a DIFFERENT spread. Label lookup must be invariant.
        labels = [1, 2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5, 99]
        # Both models identical except P50: a=50, b=90 -> P50 spread = |90-50|/100 = 0.40.
        # All other labels identical -> 0 spread. P40 (positional index 5 in 13-set)
        # is identical between models, so a positional bug would see 0 and return False.
        vals_a = [5, 10, 15, 20, 30, 40, 50, 55, 70, 80, 85, 90, 95]
        vals_b = [5, 10, 15, 20, 30, 40, 90, 55, 70, 80, 85, 90, 95]
        model_a = [[p, v] for p, v in zip(labels, vals_a, strict=True)]
        model_b = [[p, v] for p, v in zip(labels, vals_b, strict=True)]
        record = {
            "type": "numeric",
            "per_model_numeric_percentiles": {"a": model_a, "b": model_b},
            "scaling": {"range_min": 0.0, "range_max": 100.0},
        }
        # Label lookup finds the P50 disagreement (0.40 > 0.15) -> True.
        # A positional-[2,5,8] bug would read P40 (identical) and return False.
        assert exceeded_spread_threshold(record) is True


class TestExceededSpreadThresholdBaseModelField:
    """Tests for per_base_model_forecasts preference in exceeded_spread_threshold."""

    def test_prefers_base_models_high_spread(self):
        """Spread computed over base models even when per_model_forecasts is a singleton."""
        record = {
            "type": "binary",
            "per_model_forecasts": {"stacker": "75%"},
            "per_base_model_forecasts": {"gpt-5.5": "90%", "claude": "50%", "gemini": "70%"},
        }
        # spread over base = 0.9 - 0.5 = 0.4 > 0.15
        assert exceeded_spread_threshold(record) is True

    def test_prefers_base_models_low_spread(self):
        """Low spread among base models → False."""
        record = {
            "type": "binary",
            "per_model_forecasts": {"stacker": "72%"},
            "per_base_model_forecasts": {"gpt-5.5": "70%", "claude": "74%", "gemini": "72%"},
        }
        # spread = 0.74 - 0.70 = 0.04 < 0.15
        assert exceeded_spread_threshold(record) is False

    def test_empty_base_field_falls_back(self):
        """Empty per_base_model_forecasts falls back to per_model_forecasts for spread."""
        record = {
            "type": "binary",
            "per_model_forecasts": {"gpt-5.5": "90%", "claude": "50%"},
            "per_base_model_forecasts": {},
        }
        # spread = 0.9 - 0.5 = 0.4 > 0.15
        assert exceeded_spread_threshold(record) is True

    def test_singleton_stacker_without_base_returns_none(self):
        """Collapsed stacker singleton with no base field → None (< 2 probs)."""
        record = {
            "type": "binary",
            "per_model_forecasts": {"stacker": "75%"},
        }
        assert exceeded_spread_threshold(record) is None


# ---------------------------------------------------------------------------
# Main detect_stacker_fired tests
# ---------------------------------------------------------------------------


class TestDetectStackerFired:
    def test_was_stacked_true_confirmed_stacker(self):
        """Signal 1: explicit was_stacked=True → confirmed_stacker."""
        record = {"was_stacked": True, "type": "binary"}
        assert detect_stacker_fired(record) == "confirmed_stacker"

    def test_was_stacked_false_low_spread_confirmed_median(self):
        """was_stacked=False AND low spread → confirmed_median."""
        record = {
            "was_stacked": False,
            "type": "binary",
            "per_model_forecasts": {
                "gpt-5.5": "70%",
                "claude-opus-4.7": "72%",
            },
        }
        assert detect_stacker_fired(record) == "confirmed_median"

    def test_was_stacked_false_but_high_spread_and_production_differs(self):
        """was_stacked=False but high spread + production differs from median → likely_stacker.

        This is the "flag may be wrong" case — the record says no stacking but
        the evidence says otherwise.
        """
        record = {
            "was_stacked": False,
            "type": "binary",
            "our_prob_yes": 0.85,
            "per_model_forecasts": {
                "gpt-5.5": "60%",
                "claude-opus-4.7": "80%",
                "gemini-3.1-pro": "50%",
            },
        }
        # median = 0.60, spread = 0.80-0.50 = 0.30 > 0.15, |0.85 - 0.60| = 0.25 > 0.05
        assert detect_stacker_fired(record) == "likely_stacker"

    def test_no_flags_low_spread_likely_median(self):
        """No explicit flags + low spread → likely_median."""
        record = {
            "type": "binary",
            "our_prob_yes": 0.71,
            "per_model_forecasts": {
                "gpt-5.5": "70%",
                "claude-opus-4.7": "72%",
                "gemini-3.1-pro": "71%",
            },
        }
        assert detect_stacker_fired(record) == "likely_median"

    def test_no_flags_high_spread_production_matches_median(self):
        """No flags + high spread + production matches median → likely_median.

        Stacker may have failed and fallen back to median.
        """
        record = {
            "type": "binary",
            "our_prob_yes": 0.70,
            "per_model_forecasts": {
                "gpt-5.5": "50%",
                "claude-opus-4.7": "90%",
                "gemini-3.1-pro": "70%",
            },
        }
        # median = 0.70, spread = 0.9-0.5 = 0.4 > 0.15, |0.70 - 0.70| = 0.0 <= 0.05
        assert detect_stacker_fired(record) == "likely_median"

    def test_no_fields_unknown(self):
        """Completely empty record → unknown."""
        record = {"question_id": 999}
        assert detect_stacker_fired(record) == "unknown"

    def test_body_marker_stacker_outcome_primary(self):
        """Signal 3: body marker STACKER_OUTCOME=primary → confirmed_stacker."""
        record = {
            "type": "binary",
            "comment_text": "some text <!-- STACKER_OUTCOME=primary --> more",
        }
        assert detect_stacker_fired(record) == "confirmed_stacker"

    def test_body_marker_stacker_outcome_fallback_llm(self):
        """STACKER_OUTCOME=fallback_llm also means stacker fired."""
        record = {
            "type": "binary",
            "comment_text": "text <!-- STACKER_OUTCOME=fallback_llm --> end",
        }
        assert detect_stacker_fired(record) == "confirmed_stacker"

    def test_stacker_outcome_field_primary(self):
        """Signal 2: stacker_outcome field present → confirmed_stacker."""
        record = {"stacker_outcome": "primary", "type": "binary"}
        assert detect_stacker_fired(record) == "confirmed_stacker"

    def test_stacker_outcome_field_skipped(self):
        """stacker_outcome=skipped + low spread → confirmed_median."""
        record = {
            "stacker_outcome": "skipped",
            "type": "binary",
            "per_model_forecasts": {
                "gpt-5.5": "70%",
                "claude-opus-4.7": "72%",
            },
        }
        assert detect_stacker_fired(record) == "confirmed_median"

    def test_stacker_outcome_field_fallback_median(self):
        """stacker_outcome=fallback_median → confirmed_median."""
        record = {
            "stacker_outcome": "fallback_median",
            "type": "binary",
        }
        assert detect_stacker_fired(record) == "confirmed_median"

    def test_numeric_record_high_spread_stacker_marker(self):
        """Numeric record with stacker marker → confirmed_stacker."""
        record = {
            "type": "numeric",
            "comment_text": "text <!-- STACKED=true --> end",
        }
        assert detect_stacker_fired(record) == "confirmed_stacker"

    def test_custom_threshold(self):
        """Custom default_threshold for production-vs-median delta."""
        record = {
            "type": "binary",
            "our_prob_yes": 0.73,
            "per_model_forecasts": {
                "gpt-5.5": "60%",
                "claude-opus-4.7": "80%",
                "gemini-3.1-pro": "70%",
            },
        }
        # median = 0.70, spread = 0.80-0.60 = 0.20 > 0.15
        # |0.73 - 0.70| = 0.03
        # With default_threshold=0.05: 0.03 < 0.05, so likely_median (production matches)
        assert detect_stacker_fired(record, default_threshold=0.05) == "likely_median"
        # With default_threshold=0.02: 0.03 > 0.02, so likely_stacker
        assert detect_stacker_fired(record, default_threshold=0.02) == "likely_stacker"

    def test_historical_body_signature_confirmed(self):
        """Historical body signature (pre-marker era) → confirmed_stacker."""
        record = {
            "type": "binary",
            "comment_text": (
                "## R1: Forecaster 1 Reasoning\n"
                "Model: openrouter/anthropic/claude-opus-4.5\n"
                "## Stacker Meta-Analysis\n"
                "The models show disagreement on this question..."
            ),
        }
        assert detect_stacker_fired(record) == "confirmed_stacker"

    def test_cross_signal_rescues_via_per_base_model_forecasts(self):
        """Cross-signal fires correctly: no explicit flags, but per_base_model_forecasts
        shows real disagreement and production differs from base median → likely_stacker.

        This is the key scenario where the old code would return 'unknown' because
        per_model_forecasts collapsed to a stacker singleton (< 2 entries → None
        for both spread and delta). With per_base_model_forecasts populated, the
        cross-signal now correctly identifies the stacking.
        """
        record = {
            "type": "binary",
            "our_prob_yes": 0.82,
            "per_model_forecasts": {"stacker": "82%"},
            "per_base_model_forecasts": {"gpt-5.5": "60%", "claude": "90%", "gemini": "70%"},
        }
        # Base median = 0.70, spread = 0.9-0.6 = 0.30 > 0.15, |0.82-0.70| = 0.12 > 0.05
        assert detect_stacker_fired(record) == "likely_stacker"

    def test_cross_signal_base_models_agree_likely_median(self):
        """Base models agree (low spread) even though per_model_forecasts is a singleton."""
        record = {
            "type": "binary",
            "our_prob_yes": 0.71,
            "per_model_forecasts": {"stacker": "71%"},
            "per_base_model_forecasts": {"gpt-5.5": "70%", "claude": "72%", "gemini": "71%"},
        }
        # spread = 0.72-0.70 = 0.02 < 0.15 → likely_median
        assert detect_stacker_fired(record) == "likely_median"


class TestNumericSpreadDenominator:
    """The numeric branch normalizes by the question range when it has one, and by the
    members' own median P10-P90 span otherwise. Which denominator is used decides the
    verdict, so both paths and the degenerate case are pinned."""

    _LABELS: ClassVar[list[float]] = [2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5]

    @classmethod
    def _curve(cls, values: list[float]) -> list[list[float]]:
        return [[p, v] for p, v in zip(cls._LABELS, values, strict=True)]

    def test_open_bounds_normalize_by_the_member_median_span(self):
        # Open bounds rule out the range denominator. Member spans are 60 each, so the
        # median P90-P10 denominator is (median P90 80) - (median P10 20) = 60, and the
        # P50 spread of 20 normalizes to 0.33 > 0.15.
        record = {
            "type": "numeric",
            "per_model_numeric_percentiles": {
                "a": self._curve([5, 8, 10, 15, 25, 30, 35, 45, 50, 55, 60]),
                "b": self._curve([25, 28, 30, 35, 45, 50, 55, 65, 70, 75, 80]),
            },
            "open_lower_bound": True,
            "open_upper_bound": True,
            "scaling": {"range_min": 0.0, "range_max": 1000.0},
        }
        assert exceeded_spread_threshold(record) is True

    def test_a_wide_range_denominator_shrinks_the_same_disagreement_below_threshold(self):
        # Identical member curves as above, but closed bounds on a [0, 1000] range:
        # the same 20-unit P50 gap normalizes to 0.02, under the threshold.
        record = {
            "type": "numeric",
            "per_model_numeric_percentiles": {
                "a": self._curve([5, 8, 10, 15, 25, 30, 35, 45, 50, 55, 60]),
                "b": self._curve([25, 28, 30, 35, 45, 50, 55, 65, 70, 75, 80]),
            },
            "scaling": {"range_min": 0.0, "range_max": 1000.0},
        }
        assert exceeded_spread_threshold(record) is False

    def test_a_non_positive_denominator_is_unmeasurable(self):
        # Open bounds with every member declaring one value: median P90 == median P10,
        # so the fallback denominator is 0 and no spread can be normalized.
        flat = self._curve([50.0] * 11)
        record = {
            "type": "numeric",
            "per_model_numeric_percentiles": {"a": flat, "b": flat},
            "open_lower_bound": True,
            "open_upper_bound": True,
            "scaling": {},
        }
        assert exceeded_spread_threshold(record) is None

    def test_fewer_than_two_member_curves_is_unmeasurable(self):
        record = {
            "type": "numeric",
            "per_model_numeric_percentiles": {"a": self._curve([5, 8, 10, 15, 25, 30, 35, 45, 50, 55, 60])},
            "scaling": {"range_min": 0.0, "range_max": 100.0},
        }
        assert exceeded_spread_threshold(record) is None

    def test_discrete_is_treated_like_numeric(self):
        record = {
            "type": "discrete",
            "per_model_numeric_percentiles": {
                "a": self._curve([10, 15, 20, 25, 35, 40, 45, 55, 60, 65, 70]),
                "b": self._curve([30, 35, 40, 45, 55, 60, 65, 75, 80, 85, 90]),
            },
            "scaling": {"range_min": 0.0, "range_max": 100.0},
        }
        assert exceeded_spread_threshold(record) is True

    def test_an_unrecognized_question_type_is_unmeasurable(self):
        assert exceeded_spread_threshold({"type": "trinary"}) is None


class TestMemberSpread:
    """The spread value the stacking trigger compares, exposed so the trace index can sort on it.

    Each expected value is computed by hand from the fixture, in the unit the type uses.
    """

    def test_binary_is_the_probability_range_over_base_models(self):
        record = {
            "type": "binary",
            "per_model_forecasts": {"stacker": "60%"},
            "per_base_model_forecasts": {"a": "20%", "b": "35%", "c": "50%"},
        }
        assert member_spread(record) == pytest.approx(0.30)

    def test_multiple_choice_is_the_widest_per_option_range(self):
        record = {
            "type": "multiple_choice",
            "per_model_forecasts": {"a": {"X": 0.7, "Y": 0.3}, "b": {"X": 0.4, "Y": 0.6}, "c": {"X": 0.5, "Y": 0.5}},
        }
        assert member_spread(record) == pytest.approx(0.30)

    def test_numeric_is_the_widest_key_percentile_range_over_the_question_width(self):
        labels = [1, 2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95]
        low = [[label, float(label)] for label in labels]
        high = [[label, float(label) + (30.0 if label == 90 else 10.0)] for label in labels]
        record = {
            "type": "numeric",
            "per_model_numeric_percentiles": {"a": low, "b": high},
            "scaling": {"range_min": 0.0, "range_max": 200.0},
        }
        # P10 and P50 differ by 10, P90 by 30, on a 200-wide closed question.
        assert member_spread(record) == pytest.approx(30.0 / 200.0)

    def test_one_member_has_no_spread(self):
        assert member_spread({"type": "binary", "per_model_forecasts": {"a": "40%"}}) is None

    def test_a_date_question_has_no_spread_rule(self):
        assert member_spread({"type": "date"}) is None
