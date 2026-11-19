"""Multi-signal stacker detection for historical performance records.

Combines explicit flags, comment-body markers, spread thresholds, and
production-vs-median deltas to classify whether an LLM stacker fired on a
given resolved-question record. Designed to be importable from notebooks,
scripts, and ablation analyses.

Signal hierarchy (strongest first):
1. Explicit ``was_stacked`` field (authoritative log).
2. ``stacker_outcome`` field present and non-null.
3. Bot comment body marker (STACKER_OUTCOME, STACKED, historical body signature).
4. Spread > production threshold AND production differs materially from median.
5. Spread <= production threshold → likely_median.
6. None of the above → unknown.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Callable
from typing import Literal

from metaculus_bot.comment.markers import (
    HISTORICAL_STACKER_SIGNATURE_RE,
    STACKED_MARKER_RE,
    STACKER_OUTCOME_RE,
)
from metaculus_bot.constants import (
    CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD,
    CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD,
    CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD,
)
from metaculus_bot.performance_analysis.parsing import (
    MIN_SCOREABLE_ANCHORS,
    _parse_probability,
    declared_anchors,
)

logger: logging.Logger = logging.getLogger(__name__)

DetectorVerdict = Literal[
    "confirmed_stacker",
    "confirmed_median",
    "likely_stacker",
    "likely_median",
    "unknown",
]

# Stacker outcomes that indicate the stacker LLM actually produced a value
_STACKER_FIRED_OUTCOMES: frozenset[str] = frozenset({"primary", "fallback_llm"})

# Stacker outcomes that indicate the stacker did NOT produce a value.
# "skipped" = spread at/below threshold; "skipped_config_off" = spread exceeded
# the threshold but the per-type <TYPE>_STACKING_ENABLED gate was off.
_STACKER_MEDIAN_OUTCOMES: frozenset[str] = frozenset(
    {"skipped", "skipped_config_off", "fallback_median", "fallback_mean"}
)


# ---------------------------------------------------------------------------
# Signal extractors
# ---------------------------------------------------------------------------


def has_was_stacked_flag(record: dict) -> bool | None:
    """Check if the record has an explicit ``was_stacked`` boolean field.

    Returns True/False if the field is a bool, None if missing or null.
    """
    value = record.get("was_stacked")
    if isinstance(value, bool):
        return value
    return None


def get_stacker_outcome_field(record: dict) -> str | None:
    """Return the ``stacker_outcome`` field value if present and non-empty.

    Returns None if the field is missing, null, or empty string.
    """
    value = record.get("stacker_outcome")
    if value is None or value == "":
        return None
    return str(value)


def has_stacker_body_marker(record: dict) -> bool | None:
    """Check comment body for stacker markers.

    Returns:
        True if a marker indicates the stacker fired (primary, fallback_llm, STACKED=true, historical signature).
        False if a marker indicates the stacker did NOT fire (skipped, fallback_median, STACKED=false).
        None if no relevant marker is found or comment_text is absent.
    """
    comment_text = record.get("comment_text")
    if not comment_text:
        return None

    # Check STACKER_OUTCOME marker first (most specific)
    outcome_match = STACKER_OUTCOME_RE.search(comment_text)
    if outcome_match is not None:
        outcome = outcome_match.group(1).lower()
        if outcome in _STACKER_FIRED_OUTCOMES:
            return True
        if outcome in _STACKER_MEDIAN_OUTCOMES:
            return False

    # Check legacy STACKED marker
    stacked_match = STACKED_MARKER_RE.search(comment_text)
    if stacked_match is not None:
        return stacked_match.group(1).lower() == "true"

    # Check historical body signature (pre-marker era)
    if HISTORICAL_STACKER_SIGNATURE_RE.search(comment_text) is not None:
        return True

    return None


def compute_production_vs_median_delta(record: dict) -> float | None:
    """Compute |production_probability - median_of_per_model_probabilities|.

    Only works for binary records currently. Returns None if insufficient data.

    When ``per_base_model_forecasts`` is present (populated for stacked records
    by the collector), prefer it for the median computation since
    ``per_model_forecasts`` on stacked records collapses to the stacker's single
    aggregated value.
    """
    question_type = record.get("type", "")

    if question_type == "binary":
        prod_prob = record.get("our_prob_yes")
        if prod_prob is None:
            return None

        per_base_model = record.get("per_base_model_forecasts") or {}
        per_model = per_base_model if per_base_model else (record.get("per_model_forecasts") or {})
        probs = [_parse_probability(v) for v in per_model.values()]
        probs = [p for p in probs if p is not None]
        if len(probs) < 2:
            return None

        median_prob = statistics.median(probs)
        return abs(float(prod_prob) - median_prob)

    # Numeric/MC delta computation is more complex (CDF comparison);
    # return None for now — the spread threshold alone is the primary signal
    # for those types.
    return None


# Percentile LABELS (raw 0-100 numbers as parsed from comments) at which numeric
# disagreement is measured: the 10th, 50th, and 90th. Looked up by label so growing the
# standard percentile set can't shift them.
_NUMERIC_KEY_LABELS: tuple[float, ...] = (10.0, 50.0, 90.0)


def base_or_per_model_forecasts(record: dict) -> dict:
    """Per-base-model forecasts when present, else the (possibly collapsed) per-model ones.

    On a stacked record ``per_model_forecasts`` holds only the stacker's single
    aggregate, so the base-model field is preferred wherever it was recovered.
    """
    per_base_model = record.get("per_base_model_forecasts") or {}
    return per_base_model if per_base_model else (record.get("per_model_forecasts") or {})


def _binary_spread(record: dict) -> float | None:
    """The members' binary probability RANGE."""
    probs = [p for p in (_parse_probability(v) for v in base_or_per_model_forecasts(record).values()) if p is not None]
    if len(probs) < 2:
        return None
    return max(probs) - min(probs)


def _mc_spread(record: dict) -> float | None:
    """The widest PER-OPTION spread across the members.

    The collector emits MC option vectors as {model: {option: prob}}. A non-dict value
    means the MC option parser found nothing and the collector fell back to single-string
    forecasts — no option vectors, so nothing to measure.
    """
    model_option_dicts = [v for v in base_or_per_model_forecasts(record).values() if isinstance(v, dict)]
    if len(model_option_dicts) < 2:
        return None

    option_probs: dict[str, list[float]] = {}
    for model_options in model_option_dicts:
        for name, prob in model_options.items():
            if prob is not None:
                option_probs.setdefault(name, []).append(float(prob))

    spreads = [max(probs) - min(probs) for probs in option_probs.values() if len(probs) >= 2]
    if not spreads:
        return None
    return max(spreads)


def _scoreable_label_maps(record: dict) -> list[dict[float, float]]:
    """Member label→value maps dense enough to measure, and carrying every key label.

    This branch consumes lenient historical data (>= MIN_SCOREABLE_ANCHORS percentiles,
    possibly non-standard), so a per-model label dict is used rather than the strict
    PercentileSet value object. The floor is the shared constant in ``parsing``, counted
    the way its docstring defines it — DISTINCT labels via ``declared_anchors``, exactly
    as ``ranking_cohort`` and ``analysis.max_step_clamp_screen`` count it. Comment prose
    can restate a member's whole set, so the raw pair count overstates density.
    """
    model_maps: list[dict[float, float]] = []
    for model_pcts in (record.get("per_model_numeric_percentiles") or {}).values():
        anchors, _ = declared_anchors(model_pcts)
        if len(anchors) < MIN_SCOREABLE_ANCHORS:
            continue
        label_to_value = {round(label, 6): value for label, value in anchors.items()}
        if all(round(label, 6) in label_to_value for label in _NUMERIC_KEY_LABELS):
            model_maps.append(label_to_value)
    return model_maps


def _closed_question_width(scaling: dict, *, open_lower: object, open_upper: object) -> float | None:
    """The question's own finite width, or None when a bound is open, absent, or infinite."""
    if open_lower or open_upper:
        return None
    range_min = scaling.get("range_min")
    range_max = scaling.get("range_max")
    if range_min is None or range_max is None:
        return None
    if not math.isfinite(range_min) or not math.isfinite(range_max):
        return None
    return range_max - range_min


def _numeric_spread_denominator(record: dict, model_maps: list[dict[float, float]]) -> float:
    """What the raw percentile spread is normalized by: the question range, else the
    members' own median P10-P90 span (the only scale available on an open bound).
    """
    scaling = record.get("scaling") or {}
    question_width = _closed_question_width(
        scaling,
        open_lower=record.get("open_lower_bound", scaling.get("open_lower_bound", False)),
        open_upper=record.get("open_upper_bound", scaling.get("open_upper_bound", False)),
    )
    if question_width is not None:
        return question_width

    p10_values = [m[round(10.0, 6)] for m in model_maps]
    p90_values = [m[round(90.0, 6)] for m in model_maps]
    return statistics.median(p90_values) - statistics.median(p10_values)


def _numeric_spread(record: dict) -> float | None:
    """The widest NORMALIZED percentile spread at the 10th, 50th and 90th percentiles."""
    if len(record.get("per_model_numeric_percentiles") or {}) < 2:
        return None

    model_maps = _scoreable_label_maps(record)
    if len(model_maps) < 2:
        return None

    denominator = _numeric_spread_denominator(record, model_maps)
    if denominator <= 0:
        return None

    max_normalized_spread = 0.0
    for label in _NUMERIC_KEY_LABELS:
        values_at_pct = [m[round(label, 6)] for m in model_maps]
        normalized = (max(values_at_pct) - min(values_at_pct)) / denominator
        max_normalized_spread = max(max_normalized_spread, normalized)

    return max_normalized_spread


_SPREAD_BY_TYPE: dict[str, tuple[Callable[[dict], float | None], float]] = {
    "binary": (_binary_spread, CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD),
    "multiple_choice": (_mc_spread, CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD),
    "numeric": (_numeric_spread, CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD),
    "discrete": (_numeric_spread, CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD),
}


def member_spread(record: dict) -> float | None:
    """How far apart the record's members are, in the unit the stacking trigger compares.

    A probability range on binary, the widest per-option range on multiple choice, and on
    numeric and discrete the widest 10th/50th/90th-percentile range divided by the question's
    width (or by the members' median P10-P90 span when a bound is open). The three units differ,
    so spreads compare only within one question type. None when fewer than two members carry a
    measurable value.
    """
    spread_rule = _SPREAD_BY_TYPE.get(record.get("type", ""))
    return None if spread_rule is None else spread_rule[0](record)


def exceeded_spread_threshold(record: dict) -> bool | None:
    """Check whether per-model spread exceeds the production trigger threshold.

    Uses the same logic as ``compute_binary_spread_from_record`` and
    ``compute_numeric_spread_from_record`` from the historical residual script,
    but rewritten to operate on the raw record dict directly.

    Returns True if spread > threshold, False if spread <= threshold,
    None if spread cannot be computed.
    """
    spread_rule = _SPREAD_BY_TYPE.get(record.get("type", ""))
    if spread_rule is None:
        return None
    spread_of, threshold = spread_rule
    spread = spread_of(record)
    return None if spread is None else spread > threshold


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------


def detect_stacker_fired(record: dict, *, default_threshold: float = 0.05) -> DetectorVerdict:
    """Detect whether an LLM stacker fired on this resolved-question record.

    Combines multiple signals in priority order to return a confidence-weighted
    verdict. See module docstring for the full signal hierarchy.

    Parameters
    ----------
    record : dict
        A single performance record (as loaded from cached JSON datasets).
    default_threshold : float
        Material-difference threshold for signal #4: |production - median| must
        exceed this to indicate the stacker produced a different value. Default 0.05.

    Returns
    -------
    DetectorVerdict
        One of: confirmed_stacker, confirmed_median, likely_stacker, likely_median, unknown.
    """
    was_stacked = has_was_stacked_flag(record)
    explicit = _verdict_from_explicit_signals(record, was_stacked)
    if explicit is not None:
        return explicit
    return _verdict_from_spread_signals(record, was_stacked, default_threshold)


def _verdict_from_explicit_signals(record: dict, was_stacked: bool | None) -> DetectorVerdict | None:
    """Signals 1-3 (flag, outcome field, body marker). None = none of them decided."""
    # Signal 1: explicit was_stacked flag (authoritative)
    if was_stacked is True:
        return "confirmed_stacker"

    # Signal 2: stacker_outcome field
    stacker_outcome = get_stacker_outcome_field(record)
    if stacker_outcome is not None:
        if stacker_outcome in _STACKER_FIRED_OUTCOMES:
            return "confirmed_stacker"
        if stacker_outcome in _STACKER_MEDIAN_OUTCOMES:
            return "confirmed_median"

    # Signal 3: body markers. A marker saying the stacker did NOT fire is confirmation on
    # its own, whatever ``was_stacked`` holds (True was already returned above).
    body_marker = has_stacker_body_marker(record)
    if body_marker is True:
        return "confirmed_stacker"
    if body_marker is False:
        return "confirmed_median"

    # An explicitly-False was_stacked with no body marker is not decisive on its own —
    # it is combined with the spread signals by the caller.
    return None


def _verdict_from_spread_signals(
    record: dict,
    was_stacked: bool | None,
    default_threshold: float,
) -> DetectorVerdict:
    """Signals 4-5: per-model spread plus how far production sits from the members' median."""
    spread_exceeded = exceeded_spread_threshold(record)
    delta = compute_production_vs_median_delta(record)
    production_differs = spread_exceeded is True and delta is not None and delta > default_threshold

    if was_stacked is False:
        # Explicit flag says no stacking. But check if evidence contradicts.
        return "likely_stacker" if production_differs else "confirmed_median"

    # No explicit flags or body markers from here on — rely on spread + delta
    if spread_exceeded is False:
        return "likely_median"
    if spread_exceeded is True:
        # High spread but production matches median — stacker may have failed/fallen back
        return "likely_stacker" if production_differs else "likely_median"

    # spread_exceeded is None (cannot compute) and no other signals
    return "unknown"


__all__ = [
    "DetectorVerdict",
    "base_or_per_model_forecasts",
    "compute_production_vs_median_delta",
    "detect_stacker_fired",
    "exceeded_spread_threshold",
    "get_stacker_outcome_field",
    "has_stacker_body_marker",
    "has_was_stacked_flag",
    "member_spread",
]
