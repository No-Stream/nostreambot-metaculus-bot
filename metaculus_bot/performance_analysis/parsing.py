"""Comment text parsing and resolution parsing utilities.

Per-model attribution
---------------------
Metaculus bot comments contain two relevant sections:

1. A summary with ``*Forecaster N*: value`` bullets (one per ensemble member).
2. Rationales sections ``## R1: Forecaster N Reasoning`` whose first line is
   ``Model: openrouter/<provider>/<model>``, injected by
   ``main.TemplateForecaster._run_forecast_on`` when wrapping each prediction.

We key per-model results by the model display name pulled from the rationales'
``Model:`` line, not by the current ``FORECASTER_MODEL_NAMES`` list. This
avoids the failure mode where a roster change in ``llm_configs.py`` silently
relabels historical forecasts.

Stacked comments
----------------
When a question is stacked, the framework collapses all base predictions into a
single ``ReasonedPrediction`` — so there's only ONE ``## R1: Forecaster 1 Reasoning``
block, whose body contains both the stacker's meta-analysis and the base models'
reasonings folded in under a ``## Base Model Reasoning (inputs to stacker)`` sub-header.
See ``metaculus_bot.stacking.combine_stacker_and_base_reasoning`` for the exact
format. Per-base-model parsers below split on that delimiter to recover
attribution for each base model; the summary bullets only show the stacker's
aggregate, so per-base-model values are only recoverable from reasoning prose.

What lives here vs. next door
-----------------------------
The parsers every consumer calls stay in this module — ``parse_per_model_*``,
``parse_per_base_model_forecasts``, ``parse_resolution``, plus the shared
``MIN_SCOREABLE_ANCHORS`` floor and ``declared_anchors``. The mechanical layers
under them live next door and are re-exported here where an outside caller
already imports them from this path:

* ``performance_analysis.comment_sections`` — the summary/rationale boundary
  markers, the bullet + ``## R1:``/``Model:`` regexes, the stacker-combined body
  split, the positional ``Forecaster N`` keys, and the per-model block walk.
* ``performance_analysis.declared_value_recovery`` — the strict-block → prose →
  tolerant-salvage ladder for one reasoning body, and the percentile-label guard.
* ``performance_analysis.marker_readers`` — the ``STACKED`` / ``STACKER_OUTCOME``
  / ``STACKER_SKIP_REASON`` / ``FORECASTERS_USED`` trailer markers.
"""

import logging
import re
from collections.abc import Sequence

from metaculus_bot.comment.markers import EXPLANATION_DETAIL_UNKNOWN_MARKER
from metaculus_bot.performance_analysis.comment_sections import (
    _FORECASTER_RE,
    _iter_per_model_blocks,
    _split_stacker_combined_body,  # noqa: F401  # re-export: comment/trimming.py and tests/test_comment_trimming.py document this attribution path by name
    _summary_section_for_bullets,
    anonymous_model_key,
    extract_model_display_name_from_reasoning,  # noqa: F401  # re-export: forecaster.py, comment/formatting.py and the __init__ surface import it from this module
    is_anonymous_model_key,  # noqa: F401  # re-export: analysis.py, ranking_cohort.py and the __init__ surface import it from this module
    parse_forecaster_model_map,
)
from metaculus_bot.performance_analysis.declared_value_recovery import (
    _MC_OPTION_LINE_RE,
    _PERCENTILE_LINE_RE,
    _base_model_option_probs,
    _base_model_probability,
    _numeric_percentiles_from_block,
    _numeric_percentiles_from_block_tolerant,
    _parse_probability,  # noqa: F401  # re-export: analysis.py, audit.py and stacker_detection.py import it from this module
    _validate_percentile_labels,
)
from metaculus_bot.performance_analysis.marker_readers import (
    detect_historical_stacker_signature,  # noqa: F401  # re-export: the __init__ surface imports it from this module
    parse_forecasters_used_marker,  # noqa: F401  # re-export: collector.py imports it from this module
    parse_inferred_stacker_outcome,  # noqa: F401  # re-export: collector.py and scripts/derive_mini_comment_fixture.py import it from this module
    parse_stacked_marker,  # noqa: F401  # re-export: collector.py and scripts/derive_mini_comment_fixture.py import it from this module
    parse_stacker_outcome_marker,  # noqa: F401  # re-export: the __init__ surface imports it from this module
    parse_stacker_skip_reason_marker,  # noqa: F401  # re-export: collector.py imports it from this module
)

logger: logging.Logger = logging.getLogger(__name__)

SKIP_RESOLUTIONS: frozenset[str] = frozenset({"annulled", "ambiguous"})


def parse_explanation_detail_status(comment_text: str | None) -> str | None:
    """Return ``unknown`` when the published comment explicitly omits rationale detail.

    ``None`` means the comment carries no completeness marker, which is the only honest result
    for historical comments and complete comments that predate this disclosure.
    """
    if comment_text is None:
        return None
    return "unknown" if EXPLANATION_DETAIL_UNKNOWN_MARKER in comment_text else None


# Minimum DISTINCT percentile labels a member's declared curve needs before anything
# rebuilds a distribution from it and compares that against another member's. A sparse
# recovery gets treated as a distribution the model never declared: on q43729 a 3-anchor
# curve ranked #1 at +92.01 against five 11-anchor siblings, and on q43826 the same shape
# ranked LAST at -135.86 — a ~96-point artifact either way, which is what made "gemini was
# catastrophically worse" a scoring-path artifact in that question's dossier.
#
# It lives HERE, beside the recovery that produces these curves, because consumers across
# several modules gate on it — ``ranking_cohort``, ``analysis``'s ``max_step_clamp_screen``,
# ``stacker_detection.exceeded_spread_threshold``, and ``audit`` — and each used to carry
# its own literal 9. A shared leaf is the one home every consumer can import without cycles.
MIN_SCOREABLE_ANCHORS: int = 9


def declared_anchors(pairs: Sequence[Sequence[float]]) -> tuple[dict[float, float], int]:
    """``(label -> value, n_conflicting_restatements)`` for one declared curve.

    Percentile lines are recovered from comment prose, and a member sometimes restates its
    whole set (one archived curve carries a byte-identical 11-point set twice, arriving as 22
    pairs). Keying by label is what the PCHIP build already does, so the count that matters
    for density is the number of DISTINCT labels, never the number of lines — a 3-anchor set
    restated three times is still a 3-anchor set. A restatement that disagrees with itself is
    counted so the caller can report it; the dict build otherwise takes the last value
    silently.
    """
    anchors: dict[float, float] = {}
    conflicts = 0
    for pair in pairs:
        label, value = float(pair[0]), float(pair[1])
        if label in anchors and anchors[label] != value:
            conflicts += 1
        anchors[label] = value
    return anchors, conflicts


def parse_per_model_numeric_percentiles(
    comment_text: str,
    model_names: list[str] | None = None,
    question_id: int | str | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Extract per-forecaster percentile lists from a numeric/discrete comment.

    Walks each ``## R1: Forecaster N Reasoning`` block and collects every
    ``Percentile P: V`` line inside it. Returns a dict mapping model display
    name → list of (percentile, value) tuples (percentiles are the raw numbers
    from the text, e.g. 2.5, 5, 10 — not normalized to [0, 1]).

    Attribution mirrors ``parse_per_model_forecasts``:
    1. explicit ``model_names`` (indexed 1..N),
    2. ``Model:`` line inside each R1 section,
    3. ``Forecaster N`` anonymized fallback.

    Empty dict if no sections match. Sections without percentile lines are
    skipped (stacker meta-blocks that only reason, no distribution).

    Every extracted percentile set is run through ``_validate_percentile_labels``
    before it is returned: labels outside ``(0, 100)`` are either deterministically
    rescaled (when the set is an exact canonical-set*100 double-scaling) or the
    model is dropped, so downstream calibration never sees out-of-range labels.
    ``question_id`` is optional context threaded into that guard's WARNING logs.

    Stacked blocks
    --------------
    When a single R1 block contains ``## Base Model Reasoning (inputs to stacker)``
    (the stacker-combined format emitted by
    ``metaculus_bot.stacking.combine_stacker_and_base_reasoning``), this function
    treats each base-model sub-block as its own attribution target: the dict may
    therefore contain more entries than there are R1 headers. Percentile lines
    appearing in the stacker meta-analysis portion (above the delimiter) are
    also captured under the stacker's model name when present — some stackers
    explicitly restate ``Percentile X: Y`` in their prose and those are useful
    signal.
    """
    result: dict[str, list[tuple[float, float]]] = {}
    for key, body_text, _is_stacker_meta in _iter_per_model_blocks(comment_text, model_names):
        percentiles = _numeric_percentiles_from_block(body_text)
        if percentiles is None:
            percentiles = [(float(m.group(1)), float(m.group(2))) for m in _PERCENTILE_LINE_RE.finditer(body_text)]
        if not percentiles:
            # Tolerant salvage: historical blocks with retired tier-2 fields
            # fail the strict schema, and block-only rationales have no prose
            # lines — without this rung those models vanish from recovery.
            percentiles = _numeric_percentiles_from_block_tolerant(body_text) or []
        if not percentiles:
            continue
        validated = _validate_percentile_labels(percentiles, model=key, question_id=question_id)
        if validated is None:
            continue
        result[key] = validated
    return result


def annotate_forecaster_bullets_with_models(
    text: str,
    model_names_by_index: dict[int, str],
) -> str:
    """Rewrite ``*Forecaster N*: value`` bullets to include the model name.

    Idempotent: bullets that already include a parenthesized name are left
    untouched. Indices without a known model are also left untouched.
    """

    def _replace(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        existing_name = m.group(2)
        value = m.group(3)
        if existing_name is not None:
            return m.group(0)
        name = model_names_by_index.get(idx)
        if name is None:
            return m.group(0)
        return f"*Forecaster {idx} ({name})*: {value}"

    return _FORECASTER_RE.sub(_replace, text)


def parse_per_model_reasoning_text(
    comment_text: str,
    model_names: list[str] | None = None,
) -> dict[str, str]:
    """Extract per-forecaster reasoning prose from ``## R1: Forecaster N Reasoning`` blocks.

    Returns ``{model_display_name: body_text}``. The leading ``Model: ...`` line
    is stripped so the body is just the prose the model produced. Sections
    whose body is empty after stripping are skipped.

    Attribution mirrors ``parse_per_model_numeric_percentiles`` / ``parse_per_model_forecasts``:

    1. explicit ``model_names`` (indexed 1..N),
    2. ``Model:`` line inside each R1 section,
    3. ``Forecaster N`` anonymized fallback.

    Stacked blocks
    --------------
    When a single R1 block contains ``## Base Model Reasoning (inputs to stacker)``
    (the stacker-combined format from
    ``metaculus_bot.stacking.combine_stacker_and_base_reasoning``), the dict will
    include one entry per base model found after the delimiter plus one entry
    for the stacker's own meta-analysis. This means the returned dict can have
    more entries than there are R1 headers.
    """
    result: dict[str, str] = {}
    for key, body_text, _is_stacker_meta in _iter_per_model_blocks(comment_text, model_names):
        if not body_text:
            continue
        result[key] = body_text
    return result


def parse_per_model_forecasts(
    comment_text: str,
    model_names: list[str] | None = None,
) -> dict[str, str]:
    """Extract per-model predictions from the summary section of a comment.

    Returns dict mapping model name → raw value string (e.g. ``"72.0%"``).

    Attribution sources, in order of preference:

    1. ``model_names`` argument, if provided (back-compat; indexed 1..N).
    2. ``Model:`` lines in R1 rationales sections (primary path).
    3. Anonymized fallback keys (``"Forecaster N"``) if neither is available
       — better to leave a forecast unattributed than mislabel it.

    Only returns per-BASE-model forecasts for UNSTACKED questions. For stacked
    questions the bot publishes a single summary bullet with the stacker's
    aggregate value (the base models' individual forecasts are never written to
    the summary), so this function returns ``{stacker_model: aggregate_value}``
    — that single entry. To recover per-base-model info from a stacked comment
    use ``parse_per_model_reasoning_text`` and
    ``parse_per_model_numeric_percentiles``, which operate on the combined
    reasoning body.
    """
    if model_names is not None:
        fallback_map: dict[int, str] = {i + 1: name for i, name in enumerate(model_names)}
    else:
        fallback_map = parse_forecaster_model_map(comment_text)

    summary_text = _summary_section_for_bullets(comment_text)
    result: dict[str, str] = {}
    for match in _FORECASTER_RE.finditer(summary_text):
        idx = int(match.group(1))
        inline_name = match.group(2)
        value = match.group(3).strip()
        key = inline_name.strip() if inline_name is not None else (fallback_map.get(idx) or anonymous_model_key(idx))
        result[key] = value
    return result


def parse_per_model_mc_option_probs(
    comment_text: str,
    model_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Extract per-forecaster MC option probability vectors from a comment.

    For multiple-choice questions the bot emits multi-line bullets::

        *Forecaster 1 (gpt-5.5)*:
        - Option A: 40.0%
        - Option B: 30.0%
        ...

    This function captures ALL option lines per forecaster and returns
    ``{model_display_name: {option_name: probability}}`` where probability
    is in [0, 1].

    Returns an empty dict for binary/numeric comments (no option lines found)
    or for empty input.

    Attribution uses the same fallback chain as ``parse_per_model_forecasts``:
    inline name in bullet > ``Model:`` line in R1 section > ``Forecaster N``.

    Deliberately prose-only (no structured-block path): this function reads
    the SUMMARY bullet region, which is bot-rendered display text — the
    framework formats each parsed prediction via
    ``MultipleChoiceReport.make_readable_prediction`` into
    ``- {option}: {prob}%`` lines (``forecasting_tools/forecast_bots/
    forecast_bot.py::_format_and_expand_research_summary``). The 2026-07
    block-only prompt change alters what the MODEL writes in its R1 rationale,
    not how the bot renders the summary, so these bullets keep their prose
    form across eras. The forecaster's JSON block lives in the R1 reasoning
    body and is handled by the ``_iter_per_model_blocks`` consumers.
    """
    if not comment_text:
        return {}

    if model_names is not None:
        fallback_map: dict[int, str] = {i + 1: name for i, name in enumerate(model_names)}
    else:
        fallback_map = parse_forecaster_model_map(comment_text)

    summary_text = _summary_section_for_bullets(comment_text)

    # Find all forecaster bullet positions, then collect option lines between them.
    bullet_matches = list(_FORECASTER_RE.finditer(summary_text))
    if not bullet_matches:
        return {}

    result: dict[str, dict[str, float]] = {}
    for i, match in enumerate(bullet_matches):
        idx = int(match.group(1))
        inline_name = match.group(2)

        key = inline_name.strip() if inline_name is not None else (fallback_map.get(idx) or anonymous_model_key(idx))

        # Region: the captured first-line value (group 3) plus everything
        # until the next bullet or end of summary. Group 3 matters because for
        # MC the regex consumes the first option line (e.g. "- Option A: 40.0%")
        # as part of the (.+) capture.
        first_line = match.group(3)
        after_match_end = bullet_matches[i + 1].start() if i + 1 < len(bullet_matches) else len(summary_text)
        region = first_line + "\n" + summary_text[match.end() : after_match_end]

        # Parse option lines in this region.
        options: dict[str, float] = {}
        for opt_match in _MC_OPTION_LINE_RE.finditer(region):
            option_name = opt_match.group(1).strip()
            prob_pct = float(opt_match.group(2))
            options[option_name] = prob_pct / 100.0

        if options:
            result[key] = options

    return result


def parse_per_base_model_forecasts(
    comment_text: str,
    q_type: str,
) -> dict[str, str | dict[str, float]]:
    """Extract per-base-model forecasts from a stacker-combined reasoning body.

    For binary questions: returns ``{model_name: "XX.X%"}`` — one entry per
    base-model sub-block, extracted from the LAST "Probability: X%" or
    "Final probability: X%" line in each block's prose.

    For MC questions: returns ``{model_name: {option: probability}}`` — one
    entry per base-model sub-block, extracted from ``- Option: XX.X%`` lines.

    For numeric/discrete: returns ``{}`` — those question types use
    ``parse_per_model_numeric_percentiles`` which already handles stacked bodies.

    Returns ``{}`` for non-stacked comments (no base-model sub-blocks found).
    """
    if not comment_text:
        return {}
    if q_type in ("numeric", "discrete"):
        return {}

    result: dict[str, str | dict[str, float]] = {}
    for model_name, body_text, is_stacker_meta in _iter_per_model_blocks(comment_text):
        if is_stacker_meta:
            continue

        if q_type == "binary":
            prob = _base_model_probability(body_text)
            if prob is not None:
                result[model_name] = f"{prob * 100:.1f}%"

        elif q_type == "multiple_choice":
            options = _base_model_option_probs(body_text)
            if options:
                result[model_name] = options

    # Only return non-empty if we found a stacker-combined body (i.e., there
    # was at least one is_stacker_meta=True block). For non-stacked comments,
    # _iter_per_model_blocks yields blocks but none with is_stacker_meta=True,
    # so we'd be extracting from plain per-forecaster blocks (which are already
    # captured by parse_per_model_forecasts). Return empty to avoid duplication.
    has_stacker_meta = any(is_meta for _, _, is_meta in _iter_per_model_blocks(comment_text))
    if not has_stacker_meta:
        return {}
    return result


def parse_resolution(
    resolution_raw: str,
    question_type: str,
) -> tuple[bool | float | str | None, bool]:
    """Parse raw resolution string into a typed value.

    Returns (parsed_value, should_skip). should_skip=True means this question
    should be excluded from analysis.
    """
    if resolution_raw in SKIP_RESOLUTIONS:
        return None, True

    if question_type == "binary":
        if resolution_raw == "yes":
            return True, False
        if resolution_raw == "no":
            return False, False
        logger.warning(f"Unexpected binary resolution: {resolution_raw!r}")
        return None, True

    if question_type in ("numeric", "discrete"):
        if resolution_raw == "above_upper_bound":
            return "above_upper_bound", False
        if resolution_raw == "below_lower_bound":
            return "below_lower_bound", False
        try:
            return float(resolution_raw), False
        except (ValueError, TypeError):
            logger.warning(f"Unparseable numeric resolution: {resolution_raw!r}")
            return None, True

    if question_type == "multiple_choice":
        return resolution_raw, False

    logger.warning(f"Unknown question type: {question_type!r}")
    return None, True
