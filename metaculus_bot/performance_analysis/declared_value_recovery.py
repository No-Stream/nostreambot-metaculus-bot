"""Recover one model's declared forecast value out of one reasoning body.

The three-rung extraction ladder, kept whole because the rungs only mean
anything as a sequence: the strict fenced-``json`` STRUCTURED FORECAST block
first, the historical prose value lines (``Probability: NN%``,
``Percentile X: V``, ``- Option: NN%``) second, and a tolerant raw-JSON salvage
of strict-invalid old-era blocks last. The percentile-label guard that runs
after every rung lives here too, since all three feed it. Split out of
``parsing`` so the ladder and its era-specific rationale sit together, away from
the attribution walk that decides WHOSE body is being read.
"""

import json
import logging
import math
import re
from typing import Literal, TypeGuard

from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.percentile_set import PERCENTILE_KEY_DECIMALS
from metaculus_bot.structured_output_schema import (
    BinaryStructured,
    MultipleChoiceStructured,
    NumericStructured,
    StructuredBlock,
    extract_json_block,
    extract_json_block_candidates,
    parse_structured_block,
)

logger: logging.Logger = logging.getLogger(__name__)

# Matches a leading signed decimal number, optionally followed by '%'. Used by
# ``_parse_probability`` to pull a value out of a forecast string that may have
# surrounding prose (e.g. ``"about 72%"``) — standard comment bullets contain
# just the bare value, but we tolerate stray text since the downstream cost of
# dropping a parseable forecast is higher than the cost of accepting a weird one.
_PROBABILITY_VALUE_RE: re.Pattern[str] = re.compile(r"(-?[0-9]+(?:\.[0-9]+)?)\s*%?")


def _parse_probability(raw: str) -> float | None:
    """Parse a per-model forecast string into a probability in [0, 1].

    Accepts values like ``"72.0%"`` and ``"0.72"``. Heuristic for bare numbers:

    * Explicit ``%`` in the source string → always divide by 100.
    * Bare value in ``[0, 1]`` → treat as already-scaled probability.
    * Bare value in ``[1.5, 100]`` → treat as percentage (the bot's
      missing-``%`` form on values like ``"72"``); divide by 100.
    * Bare value in ``(1.0, 1.5)`` → ambiguous (too high for a valid
      probability, too low to confidently be a percentage). Reject as a
      parse error to avoid silently corrupting Brier/spread calculations
      with a ~2× scaled value (the F11 bug pre-fix).
    * Bare value > 100 or value < 0 → reject as out-of-range parse error.

    Returns ``None`` for any rejected case and logs a WARNING when an
    in-range but ambiguous value is dropped, so operators can grep the
    skip rate for upstream parse drift.
    """
    match = _PROBABILITY_VALUE_RE.search(raw)
    if match is None:
        return None
    try:
        num = float(match.group(1))
    except ValueError:
        return None
    has_percent = "%" in raw
    if has_percent:
        scaled = num / 100.0
    elif 0.0 <= num <= 1.0:
        scaled = num
    elif 1.5 <= num <= 100.0:
        scaled = num / 100.0
    else:
        # Either negative, in the (1.0, 1.5) ambiguous zone, or > 100.
        # The (1.0, 1.5) zone is the load-bearing fix: ``"1.2"`` was being
        # silently coerced to 0.012 — a parse error masquerading as a tiny
        # probability that contaminated Brier / spread metrics. Drop it
        # explicitly with a WARNING so the skip rate is auditable.
        logger.warning(
            "Dropping out-of-range probability value: %r (parsed as %s; "
            "neither a valid decimal probability nor a confident percentage)",
            raw,
            num,
        )
        return None
    if scaled < 0.0 or scaled > 1.0:
        return None
    return scaled


# Matches lines like "Percentile 2.5: 1234.5" (trailing whitespace OK).
_PERCENTILE_LINE_RE: re.Pattern[str] = re.compile(
    r"^\s*Percentile\s+([0-9]+(?:\.[0-9]+)?)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*$",
    re.MULTILINE,
)

# Matches MC option lines like: ``- Option A: 40.0%`` or ``- Yes (>50%): 60.0%``
# Captures (option_name, numeric_value). The option name runs from after ``- ``
# to the last ``:`` before the numeric value. Anchored at start-of-line.
_MC_OPTION_LINE_RE: re.Pattern[str] = re.compile(r"(?m)^[ \t]*-\s+(.+?):\s+([0-9]+(?:\.[0-9]+)?)\s*%")

_PROBABILITY_LINE_RE: re.Pattern[str] = re.compile(
    r"(?i)(?:final\s+)?probability\s*:\s*(.+)",
)


def _extract_last_probability_from_body(body_text: str) -> float | None:
    """Extract the LAST probability value from a base-model reasoning body.

    Scans for lines matching "Probability: X%" or "Final probability: X%"
    (case-insensitive) and returns the last one found — that's typically the
    model's final answer after any intermediate estimates.
    """
    last_prob: float | None = None
    for match in _PROBABILITY_LINE_RE.finditer(body_text):
        raw_value = match.group(1).strip()
        parsed = _parse_probability(raw_value)
        if parsed is not None:
            last_prob = parsed
    return last_prob


# ---------------------------------------------------------------------------
# Structured-block extraction (block-first, prose-regex fallback)
#
# Since 2026-07 the forecaster prompts emit forecast values ONLY inside the
# fenced ```json STRUCTURED FORECAST block; the trailing prose value lines
# ("Probability: NN%", "Percentile X: V", "- Option: NN%") are gone from new
# rationales. Historical comments carry the prose lines and often no block, so
# the per-model reasoning-body parsers try the JSON block first and fall back to
# the prose regexes — no era detection, just per-body try-then-fallback.
#
# NOTE: this applies only to parsers that read the R1 REASONING bodies (via
# ``_iter_per_model_blocks``). ``parse_per_model_mc_option_probs`` reads the
# SUMMARY bullet region, which is bot-rendered display text
# (``forecasting_tools.forecast_bots.forecast_bot`` formats each parsed
# prediction via ``make_readable_prediction`` → ``- {option}: {prob}%`` lines),
# unaffected by the prompt change — it stays prose-only.
# ---------------------------------------------------------------------------


def _parse_block_in_body(
    body_text: str,
    question_type: Literal["binary", "numeric", "multiple_choice"],
) -> StructuredBlock | None:
    """Parse the structured JSON block in a per-model reasoning body, if any.

    Pre-checks for a fenced block so historical prose-only bodies skip the
    full parse (and its per-body "no JSON block" logging) entirely.
    ``parse_structured_block`` returns None on malformed JSON / validation
    failure, so callers just fall back to the prose regex on None.
    """
    candidates = extract_json_block_candidates(body_text)
    if all(_declares_other_known_type(raw_block, question_type) for raw_block in candidates):
        return None
    return parse_structured_block(body_text, question_type)


def _declares_other_known_type(raw_block: str, question_type: str) -> bool:
    """True when the block names a different known question type.

    Callers probe a comment of unknown type for each type in turn, so a binary block read as numeric
    is "not this type", not a parse failure; without this check the strict parser logs a mismatch
    WARNING for every member of every non-numeric comment the collector reads. The caller skips the
    parse only when EVERY candidate names another type, so a trailing recap block of another type
    cannot hide a valid earlier block, the case ``parse_structured_block`` walks candidates for.
    """
    try:
        payload = json.loads(raw_block)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    declared_type = payload.get("question_type")
    return declared_type in _KNOWN_BLOCK_QUESTION_TYPES and declared_type != question_type


def _numeric_percentiles_from_block(body_text: str) -> list[tuple[float, float]] | None:
    """Return sorted (percentile, value) pairs from the body's JSON block, or None.

    The block stores percentile keys as decimals (0.025, 0.5, 0.9) while the
    prose convention downstream expects the raw percent labels the regex path
    captures (2.5, 50, 90) — convert so consumers see identical shapes
    regardless of comment era. Rounded to 6 places to cancel float noise
    (0.1 * 100 == 10.000000000000002), matching the label rounding in
    ``stacker_detection``.
    """
    block = _parse_block_in_body(body_text, "numeric")
    if not isinstance(block, NumericStructured) or not block.declared_percentiles:
        return None
    return sorted((round(pct * 100.0, 6), value) for pct, value in block.declared_percentiles.items())


def _binary_prob_from_block(body_text: str) -> float | None:
    """Return the declared posterior probability from the body's JSON block, or None."""
    block = _parse_block_in_body(body_text, "binary")
    if not isinstance(block, BinaryStructured):
        return None
    return block.posterior_prob


def _mc_option_probs_from_block(body_text: str) -> dict[str, float] | None:
    """Return {option_name: probability} from the body's JSON block, or None."""
    block = _parse_block_in_body(body_text, "multiple_choice")
    if not isinstance(block, MultipleChoiceStructured):
        return None
    return dict(block.option_probs)


# ---------------------------------------------------------------------------
# Tolerant raw-JSON salvage (last rung, after strict block AND prose regex)
#
# Historical blocks (pre-2026-07-08 prompt era) carry retired tier-2 scaffold
# fields (``mixture_components``, ``tails``, ``distribution_family_hint``, the
# old ``scenarios`` shape). The retired KEYS are what still needs this rung; the
# other historical offender, an edge-value ``concentration: 0.0``, does not — since
# 2026-09-02 the strict MC schema reads an unusable ``concentration`` / ``other_mass``
# as absent rather than rejecting the block, so those comments now resolve on the
# strict rung. ``parse_structured_block`` uses extra="forbid"
# schemas, so it returns None for the ENTIRE block even though the declared
# forecast values inside are fine. Models that emitted block-only rationales
# with no prose value lines (gemini-3.1-pro in the 2026-05/06 era) then vanish
# from recovered per-model data entirely — the 2026-07-15 ensemble screening's
# false "gemini missed 5 of 45 summer questions" finding was exactly this
# (4 of the 5 were parse losses, verified against GHA run logs; only 1 was a
# real soft-deadline drop). The salvage below reads just the value field from
# the raw JSON, faithful to what the model emitted, mirroring the tolerant
# read pinned in scratch/coherence_2026-07-15/schema_README.md.
#
# Precedence is deliberately strict-block > prose > tolerant: transition-era
# comments where a sparse/invalid block coexists with full prose value lines
# keep resolving from prose exactly as before.
# ---------------------------------------------------------------------------

_KNOWN_BLOCK_QUESTION_TYPES: frozenset[str] = frozenset({"binary", "numeric", "multiple_choice"})


def _tolerant_block_payload(
    body_text: str,
    question_type: Literal["binary", "numeric", "multiple_choice"],
) -> dict | None:
    """Raw ``json.loads`` of the body's fenced block, bypassing schema validation.

    Returns the payload dict, or None when there is no block, the JSON is
    malformed, or the block declares a *conflicting* known question_type
    (an unknown/absent question_type passes — field extraction filters it).
    """
    raw = extract_json_block(body_text)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    declared_type = payload.get("question_type")
    if declared_type in _KNOWN_BLOCK_QUESTION_TYPES and declared_type != question_type:
        return None
    return payload


def _is_finite_number(value: object) -> TypeGuard[float]:
    """True for real int/float values (bool excluded, NaN/inf excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numeric_percentiles_from_block_tolerant(body_text: str) -> list[tuple[float, float]] | None:
    """Salvage (percentile, value) pairs from a strict-invalid numeric block.

    Same decimal→raw-percent label conversion as the strict reader. Keys must
    coerce to float in (0, 1); values must be finite numbers. Returns None when
    nothing usable survives.

    A PARTIAL set is a legitimate return here, deliberately: recovering 3 of 13 anchors off
    an old-era block is the whole point of this rung, and whether 3 is enough depends on what
    the caller does with the curve. Completeness is therefore a CONSUMER gate — see
    ``MIN_SCOREABLE_ANCHORS`` (in ``parsing``), which drops sparse curves before they are
    PCHIP'd into a full CDF and log-scored beside 11-anchor members. A consumer that scores
    these without that gate is measuring interpolation, not the model.
    """
    payload = _tolerant_block_payload(body_text, "numeric")
    if payload is None:
        return None
    declared = payload.get("declared_percentiles")
    if not isinstance(declared, dict):
        return None
    pairs: list[tuple[float, float]] = []
    for key, value in declared.items():
        try:
            pct = float(key)
        except (TypeError, ValueError):
            continue
        if not (0.0 < pct < 1.0) or not _is_finite_number(value):
            continue
        pairs.append((round(pct * 100.0, 6), float(value)))
    return sorted(pairs) if pairs else None


def _binary_prob_from_block_tolerant(body_text: str) -> float | None:
    """Salvage the posterior probability from a strict-invalid binary block."""
    payload = _tolerant_block_payload(body_text, "binary")
    if payload is None:
        return None
    prob = payload.get("posterior_prob")
    if not _is_finite_number(prob) or not (0.0 <= prob <= 1.0):
        return None
    return float(prob)


def _mc_option_probs_from_block_tolerant(body_text: str) -> dict[str, float] | None:
    """Salvage {option_name: probability} from a strict-invalid MC block.

    Per-value bounds are enforced but the strict sum≈1 check is NOT — the
    downstream consumers (option-set match + renormalization) handle scale.
    """
    payload = _tolerant_block_payload(body_text, "multiple_choice")
    if payload is None:
        return None
    raw_options = payload.get("option_probs")
    if not isinstance(raw_options, dict):
        return None
    options: dict[str, float] = {}
    for key, value in raw_options.items():
        if not isinstance(key, str) or not key.strip():
            return None
        if not _is_finite_number(value) or not (0.0 <= value <= 1.0):
            return None
        options[key] = float(value)
    if not options or sum(options.values()) <= 0.0:
        return None
    return options


# ---------------------------------------------------------------------------
# Percentile-label validation (guard against double-scaled labels)
#
# ``parse_per_model_numeric_percentiles`` emits labels in the PERCENT convention
# (2.5, 5, ..., 97.5, 99) — every legitimate label is strictly inside (0, 100).
# A forecaster that writes its block keys in percent form (2.5, 5, ...) instead
# of the decimal convention (0.025, 0.05, ...) has them multiplied by 100 by the
# block readers, producing double-scaled labels (250, 500, ..., 9750). That was
# the qid=43684 grok-4.3 record the 2026-07-15 coherence study had to hand-drop
# (drop_reason=malformed_percentile_labels_out_of_range). The guard below runs
# after extraction and either passes clean labels through, deterministically
# rescales an exact canonical-set*100 match, or rejects the model's percentiles.
# ---------------------------------------------------------------------------

# Canonical label sets in the percent convention (STANDARD_PERCENTILES * 100).
# The 13-point set is the current standard; the 11-point subset (P1/P99 dropped)
# is the pre-2026-07-07 archive-era shape. Both are derived from the decimal-form
# single source of truth so a percentile-set change stays in lockstep.
_CANONICAL_PERCENT_LABELS_13: frozenset[float] = frozenset(
    round(p * 100.0, PERCENTILE_KEY_DECIMALS) for p in STANDARD_PERCENTILES
)
# P1 (0.01) and P99 (0.99) are the finer tail anchors added when the set grew 11->13.
_CANONICAL_PERCENT_LABELS_11: frozenset[float] = _CANONICAL_PERCENT_LABELS_13 - {
    round(0.01 * 100.0, PERCENTILE_KEY_DECIMALS),
    round(0.99 * 100.0, PERCENTILE_KEY_DECIMALS),
}
_CANONICAL_PERCENT_LABEL_SETS: tuple[frozenset[float], ...] = (
    _CANONICAL_PERCENT_LABELS_13,
    _CANONICAL_PERCENT_LABELS_11,
)


def _validate_percentile_labels(
    percentiles: list[tuple[float, float]],
    *,
    model: str,
    question_id: int | str | None = None,
) -> list[tuple[float, float]] | None:
    """Guard extracted percentile labels against the double-scaled failure mode.

    Runs after label extraction/normalization (block, prose, or tolerant rung).
    The percent-label convention puts every legitimate label strictly inside
    ``(0, 100)`` (P1..P99). This guard:

    * returns the list unchanged when every label is in ``(0, 100)``;
    * deterministically divides every label by 100 (with a WARNING) when the set
      is EXACTLY a canonical label set * 100 (11- or 13-point) — a safe,
      reversible correction of percent-vs-decimal double scaling, not a guess;
    * otherwise returns ``None`` (with a WARNING naming the model and offending
      labels) so the caller drops the model rather than emit out-of-range labels
      into downstream calibration.

    ``question_id`` is optional log context; callers that lack a qid pass None.
    """
    labels = [label for label, _ in percentiles]
    if all(0.0 < label < 100.0 for label in labels):
        return percentiles

    rescaled_label_set = frozenset(round(label / 100.0, PERCENTILE_KEY_DECIMALS) for label in labels)
    if rescaled_label_set in _CANONICAL_PERCENT_LABEL_SETS:
        logger.warning(
            "Rescaling double-scaled percentile labels (label/100 == canonical set): "
            "question=%s model=%s offending_labels=%s",
            question_id,
            model,
            sorted(labels),
        )
        return [(round(label / 100.0, PERCENTILE_KEY_DECIMALS), value) for label, value in percentiles]

    logger.warning(
        "Rejecting numeric percentiles with out-of-range labels: question=%s model=%s offending_labels=%s",
        question_id,
        model,
        sorted(labels),
    )
    return None


def _base_model_probability(body_text: str) -> float | None:
    """One base-model block's binary probability, strict block → prose → tolerant salvage."""
    prob = _binary_prob_from_block(body_text)
    if prob is None:
        prob = _extract_last_probability_from_body(body_text)
    if prob is None:
        # Tolerant salvage for historical strict-invalid blocks with
        # no prose value lines (see the salvage-rung comment above).
        prob = _binary_prob_from_block_tolerant(body_text)
    return prob


def _base_model_option_probs(body_text: str) -> dict[str, float]:
    """One base-model block's MC option probabilities; empty when none can be read."""
    options = _mc_option_probs_from_block(body_text)
    if options is None:
        options = {
            opt_match.group(1).strip(): float(opt_match.group(2)) / 100.0
            for opt_match in _MC_OPTION_LINE_RE.finditer(body_text)
        }
    if not options:
        # Tolerant salvage, e.g. blocks with concentration=0.0 that
        # the strict validator rejects wholesale.
        options = _mc_option_probs_from_block_tolerant(body_text) or {}
    return options
