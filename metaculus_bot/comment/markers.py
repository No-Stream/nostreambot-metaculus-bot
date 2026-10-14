"""Shared constants + regex for the stacker-outcome HTML-comment markers.

Keeping the literals + regex in one module avoids silent producer/consumer drift
if either side changes the comment shape, so this docstring is where the meaning
of every marker value lives and the code below carries at most a one-line
pointer. The markers are injected into each published Metaculus comment by the
bot's ``_create_unified_explanation`` override (see ``main.py``) and parsed back
out by the residual-analysis collector via
``metaculus_bot.performance_analysis.parsing.parse_stacked_marker``.

STACKER_OUTCOME
---------------

``STACKER_OUTCOME=<primary|fallback_llm|fallback_median|fallback_mean|skipped|skipped_config_off>``
is the tri-state-plus marker, and it replaces the lossy ``STACKED=true/false``
marker, which silently mislabeled a median fallback as ``STACKED=true`` and could
not express the conditional-stacking skip path at all.

* ``primary`` and ``fallback_llm`` mean a stacker LLM produced the value.
* ``fallback_median`` means no stacker LLM did (both failed, or
  CONDITIONAL_STACKING's wall-clock budget skipped them) and the members were
  combined under that strategy's MEDIAN rule.
* ``fallback_mean`` is the same outcome on the regular STACKING budget-skip path,
  whose base combine is MEAN (F15). STACKING's budget-skip path base-combines by
  MEAN where CONDITIONAL_STACKING's does so by MEDIAN, and the original
  ``fallback_median`` marker mislabeled it, so the separate value is what makes
  the rule that ran readable from the published record.
* ``skipped`` means the conditional-stacking trigger short-circuited the stacker
  because spread stayed at/below the threshold.
* ``skipped_config_off`` means spread EXCEEDED the threshold but the per-type
  ``<TYPE>_STACKING_ENABLED`` env gate was off, so the stacker was deliberately
  bypassed: config-suppressed rather than spread-suppressed. (2026-07 residual
  round: 22 numeric "skipped" suppressions had to be re-attributed to config-off
  via git archaeology — this value makes the reason durable in the published
  record.) Comments published before this value shipped collapse both skip
  reasons into ``skipped``; disambiguating those requires the workflow-yaml flag
  history.

On a binary or MC question the outcome names the rule that ran. On a numeric or
date question it names only the STRATEGY's rule, and the rule that ran is the
run-log ``NUMERIC_AGGREGATE ... method=`` field
(``metaculus_bot/member_forecast.py``), which is authoritative: a question
elicited per bin (``numeric.config.elicit_per_bin``, a Mantic grid of
``PMF_ELICITATION_MAX_BINS`` bins or fewer) is pooled by the pointwise MEAN
under every outcome, ``fallback_median`` and the two skips included, and a
percentile question under a MEAN-strategy run carries ``method=mean`` with no
stacker outcome at all.

``STACKER_OUTCOME_RE`` lists ``skipped_config_off`` ahead of ``skipped`` in the
alternation so the longer literal wins on first match rather than relying on
backtracking after the ``skipped`` branch fails the trailing ``-->``. The
import-time asserts at the bottom of this module pin that ordering (the full
literal must be captured, not just its ``skipped`` prefix) along with every other
literal/regex pair, so an edit that breaks the invariant fails at import rather
than silently weeks later when the collector runs against real comments.

STACKED (legacy)
----------------

``STACKED=<true|false>`` is the legacy binary marker, derived from the new
outcome (true ↔ outcome ∈ {primary, fallback_llm}, false ↔ outcome ∈ {skipped,
skipped_config_off, fallback_median, fallback_mean}). Both families coexist on
every stacked comment for one round of back-compat so external parsers don't
break the day this fix lands; new analyses should prefer the STACKER_OUTCOME
variant. ``STACKED_MARKER_RE`` tolerates surrounding whitespace and casing so
accidental reformatting of already-published comments (e.g. a markdown editor
normalizing whitespace) doesn't silently desync the collector.

STACKER_SKIP_REASON
-------------------

An additive companion to STACKER_OUTCOME, emitted only on the skip paths. The
outcome value ``skipped`` conflates two mechanisms — spread at/below threshold,
and the single-forecaster short-circuit that never computes a spread at all
(q44870, the first resolved instance, is indistinguishable from a low-spread skip
in the outcome alone) — so the reason gets its own marker rather than a new
outcome value: STACKER_OUTCOME stays byte-stable for every existing parser of the
legacy value. ``config_off`` restates skipped_config_off's reason so the field is
self-contained. ``spread_undefined`` is the fifth and rarest: the spread could
not be MEASURED at all (a non-positive normalizing denominator — see
spread_metrics' SPREAD_UNDEFINED WARN), so the routing decision rests on no
measurement. It must not read as ``spread_below_threshold``, which is an
affirmative "the models agreed".

TOOLS_USED
----------

The probabilistic-tools activation marker, emitted alongside STACKED by
``_create_unified_explanation`` so residual analysis can distinguish
tool-augmented runs from vanilla stacking runs.

FORECASTERS_USED
----------------

The ensemble-size disclosure marker, injected into every published comment by
``build_unified_explanation``. It records how many forecasters CONTRIBUTED (== the
number of ``*Forecaster N*`` summary bullets) out of how many were CONFIGURED
this run. Without it a comment carrying two bullets is ambiguous between a
3-model ensemble where one model never answered (a drop) and a genuine 2-model
era (a roster change) — CLAUDE.md's "fewer than N bullets" gotcha. It rides the
preserved comment tail alongside the STACKER_OUTCOME markers, so it survives the
150k middle-trim. Cause of any drop stays in the run-log FORECASTER_DROPS
telemetry (operational detail), not the public comment.

EXPLANATION_DETAIL
------------------

``EXPLANATION_DETAIL=unknown`` is emitted only by the bounded incomplete-comment fallback. It
means that the public explanation could not retain all rationale detail, so missing model
attribution or reasoning must remain unknown to downstream readers. Unmarked historical comments
do not claim complete detail.

Stacker section headers
-----------------------

``STACKER_META_ANALYSIS_HEADER`` and ``STACKED_BASE_REASONING_HEADER`` are
emitted by ``metaculus_bot.stacking.combine_stacker_and_base_reasoning`` inside
the single R1 body for stacked questions, and shared with
``metaculus_bot.performance_analysis.parsing``, which splits the body on
``STACKED_BASE_REASONING_HEADER`` to recover per-base-model attribution.

``BASE_MODEL_SUBBLOCK_SPLIT_RE`` splits the base-model portion of a
stacker-combined R1 body into per-model sub-blocks, one per
``Model: openrouter/<provider>/<name>`` line the bot injects ahead of each base
reasoning. Two consumers share it so they can never drift:
``metaculus_bot.performance_analysis.parsing`` reads it to recover
per-base-model attribution, and ``metaculus_bot.comment.trimming`` uses the same
split so a length-trim shrinks each base sub-block from within — keeping every
base model paired with its own ``Model:`` line and JSON forecast block. The value
is required to contain ``/`` so a narrative line like ``Model: previous version``
inside a base reasoning's prose can't be mistaken for a sub-block boundary: the
bot-injected prefix is always a slash-delimited OpenRouter path (e.g.
``Model: openrouter/openai/gpt-5.5``).

Historical stacker signature
----------------------------

``HISTORICAL_STACKER_META_HEADER`` and ``HISTORICAL_STACKER_SIGNATURE_RE``
recover stacked comments published before the explicit ``STACKED=`` /
``STACKER_OUTCOME=`` markers existed. Three header variants are in the wild (all
stacker-only): ``## Stacker Meta-Analysis`` (current), ``## Meta-Analysis``
(older), and ``# Meta-Analysis and Synthesis`` (earliest H1). Match condition:
the meta header is the FIRST heading after ``## R1: Forecaster 1 Reasoning``
(modulo a possible ``Model:`` line). A bare ``## Meta-Analysis`` deeper in a body
isn't signal — that pattern shows up inside individual non-stacker forecaster
reasoning bodies too.
"""

from __future__ import annotations

import re

# Legacy binary marker, kept one round for external parsers; see the module docstring.
STACKED_MARKER_TRUE: str = "<!-- STACKED=true -->"
STACKED_MARKER_FALSE: str = "<!-- STACKED=false -->"

# Whitespace- and case-tolerant so a reformatted published comment still parses.
STACKED_MARKER_RE: re.Pattern[str] = re.compile(
    r"<!--\s*STACKED=(true|false)\s*-->",
    re.IGNORECASE,
)

# Tri-state-plus marker replacing STACKED=; every value's meaning is in the module docstring.
STACKER_OUTCOME_PRIMARY: str = "<!-- STACKER_OUTCOME=primary -->"
STACKER_OUTCOME_FALLBACK_LLM: str = "<!-- STACKER_OUTCOME=fallback_llm -->"
STACKER_OUTCOME_FALLBACK_MEDIAN: str = "<!-- STACKER_OUTCOME=fallback_median -->"
# F15: the STACKING budget-skip path base-combines by MEAN; see the module docstring.
STACKER_OUTCOME_FALLBACK_MEAN: str = "<!-- STACKER_OUTCOME=fallback_mean -->"
STACKER_OUTCOME_SKIPPED: str = "<!-- STACKER_OUTCOME=skipped -->"
# Config-suppressed rather than spread-suppressed; see the module docstring.
STACKER_OUTCOME_SKIPPED_CONFIG_OFF: str = "<!-- STACKER_OUTCOME=skipped_config_off -->"

# Alternation order is load-bearing: skipped_config_off before skipped (module docstring).
STACKER_OUTCOME_RE: re.Pattern[str] = re.compile(
    r"<!--\s*STACKER_OUTCOME=(primary|fallback_llm|fallback_median|fallback_mean|skipped_config_off|skipped)\s*-->",
    re.IGNORECASE,
)

# Additive companion to STACKER_OUTCOME on the skip paths; reasons in the module docstring.
STACKER_SKIP_REASONS: frozenset[str] = frozenset(
    {"spread_below_threshold", "spread_undefined", "config_off", "single_forecaster", "wall_clock_budget"}
)

STACKER_SKIP_REASON_RE: re.Pattern[str] = re.compile(
    r"<!--\s*STACKER_SKIP_REASON="
    r"(spread_below_threshold|spread_undefined|config_off|single_forecaster|wall_clock_budget)\s*-->",
    re.IGNORECASE,
)


def format_stacker_skip_reason_marker(reason: str) -> str:
    """Render the skip-reason marker, rejecting values the regex could not parse back."""
    if reason not in STACKER_SKIP_REASONS:
        raise ValueError(f"Unknown stacker skip reason {reason!r}; expected one of {sorted(STACKER_SKIP_REASONS)}")
    return f"<!-- STACKER_SKIP_REASON={reason} -->"


# Probabilistic-tools activation marker; see the module docstring.
TOOLS_USED_MARKER_TRUE: str = "<!-- TOOLS_USED=true -->"
TOOLS_USED_MARKER_FALSE: str = "<!-- TOOLS_USED=false -->"

# Used only when the comment's essential structure exceeds the publication budget.
EXPLANATION_DETAIL_UNKNOWN_MARKER: str = "<!-- EXPLANATION_DETAIL=unknown -->"

TOOLS_USED_MARKER_RE: re.Pattern[str] = re.compile(
    r"<!--\s*TOOLS_USED=(true|false)\s*-->",
    re.IGNORECASE,
)

# Ensemble-size disclosure: n contributed of N configured; see the module docstring.
FORECASTERS_USED_MARKER_PREFIX: str = "FORECASTERS_USED"

FORECASTERS_USED_MARKER_RE: re.Pattern[str] = re.compile(
    r"<!--\s*FORECASTERS_USED=(\d+)/(\d+)\s*-->",
    re.IGNORECASE,
)


def format_forecasters_used_marker(n_used: int, n_configured: int) -> str:
    """Render the ensemble-size marker: n contributed of N configured (``n/N``)."""
    return f"<!-- {FORECASTERS_USED_MARKER_PREFIX}={n_used}/{n_configured} -->"


# Stacked-body section headers, shared with the residual-analysis parser (module docstring).
STACKER_META_ANALYSIS_HEADER: str = "## Stacker Meta-Analysis"
STACKED_BASE_REASONING_HEADER: str = "## Base Model Reasoning (inputs to stacker)"

# Per-base-model sub-block boundary, shared by parsing and trimming (module docstring).
BASE_MODEL_SUBBLOCK_SPLIT_RE: re.Pattern[str] = re.compile(
    r"(?m)^[ \t]*Model:[ \t]*([^\n]*/[^\n]*?)[ \t]*$",
)

# Pre-marker stacked-comment signature; its three header variants are in the module docstring.
HISTORICAL_STACKER_META_HEADER: str = "## Meta-Analysis"

HISTORICAL_STACKER_SIGNATURE_RE: re.Pattern[str] = re.compile(
    r"##\s+R1:\s+Forecaster\s+1\s+Reasoning"
    r"(?:\s*\n\s*Model:[^\n]*)?"
    r"\s*\n+"
    r"#{1,2}\s+(?:Stacker\s+)?Meta-Analysis\b",
    re.IGNORECASE,
)

# Fail at import if any literal/regex pair desyncs, not weeks later in the collector.
assert STACKED_MARKER_RE.search(STACKED_MARKER_TRUE) is not None, (
    f"STACKED_MARKER_RE does not match STACKED_MARKER_TRUE={STACKED_MARKER_TRUE!r}"
)
assert STACKED_MARKER_RE.search(STACKED_MARKER_FALSE) is not None, (
    f"STACKED_MARKER_RE does not match STACKED_MARKER_FALSE={STACKED_MARKER_FALSE!r}"
)
assert STACKER_OUTCOME_RE.search(STACKER_OUTCOME_PRIMARY) is not None, (
    f"STACKER_OUTCOME_RE does not match STACKER_OUTCOME_PRIMARY={STACKER_OUTCOME_PRIMARY!r}"
)
assert STACKER_OUTCOME_RE.search(STACKER_OUTCOME_FALLBACK_LLM) is not None, (
    f"STACKER_OUTCOME_RE does not match STACKER_OUTCOME_FALLBACK_LLM={STACKER_OUTCOME_FALLBACK_LLM!r}"
)
assert STACKER_OUTCOME_RE.search(STACKER_OUTCOME_FALLBACK_MEDIAN) is not None, (
    f"STACKER_OUTCOME_RE does not match STACKER_OUTCOME_FALLBACK_MEDIAN={STACKER_OUTCOME_FALLBACK_MEDIAN!r}"
)
assert STACKER_OUTCOME_RE.search(STACKER_OUTCOME_FALLBACK_MEAN) is not None, (
    f"STACKER_OUTCOME_RE does not match STACKER_OUTCOME_FALLBACK_MEAN={STACKER_OUTCOME_FALLBACK_MEAN!r}"
)
assert STACKER_OUTCOME_RE.search(STACKER_OUTCOME_SKIPPED) is not None, (
    f"STACKER_OUTCOME_RE does not match STACKER_OUTCOME_SKIPPED={STACKER_OUTCOME_SKIPPED!r}"
)
_skipped_config_off_match = STACKER_OUTCOME_RE.search(STACKER_OUTCOME_SKIPPED_CONFIG_OFF)
assert _skipped_config_off_match is not None, (
    f"STACKER_OUTCOME_RE does not match STACKER_OUTCOME_SKIPPED_CONFIG_OFF={STACKER_OUTCOME_SKIPPED_CONFIG_OFF!r}"
)
# Guard the alternation order: the full literal must be captured, not its "skipped" prefix.
assert _skipped_config_off_match.group(1) == "skipped_config_off", (
    f"STACKER_OUTCOME_RE captured {_skipped_config_off_match.group(1)!r} from "
    f"{STACKER_OUTCOME_SKIPPED_CONFIG_OFF!r}; expected 'skipped_config_off'"
)
del _skipped_config_off_match
assert all(
    (_m := STACKER_SKIP_REASON_RE.search(format_stacker_skip_reason_marker(_reason))) is not None
    and _m.group(1) == _reason
    for _reason in STACKER_SKIP_REASONS
), "STACKER_SKIP_REASON_RE does not round-trip its own formatter output"
assert TOOLS_USED_MARKER_RE.search(TOOLS_USED_MARKER_TRUE) is not None, (
    f"TOOLS_USED_MARKER_RE does not match TOOLS_USED_MARKER_TRUE={TOOLS_USED_MARKER_TRUE!r}"
)
assert TOOLS_USED_MARKER_RE.search(TOOLS_USED_MARKER_FALSE) is not None, (
    f"TOOLS_USED_MARKER_RE does not match TOOLS_USED_MARKER_FALSE={TOOLS_USED_MARKER_FALSE!r}"
)
_forecasters_used_match = FORECASTERS_USED_MARKER_RE.search(format_forecasters_used_marker(2, 3))
assert _forecasters_used_match is not None, "FORECASTERS_USED_MARKER_RE does not match its own formatter output"
assert _forecasters_used_match.group(1) == "2", (
    f"FORECASTERS_USED_MARKER_RE captured {_forecasters_used_match.groups()!r}; expected ('2', '3')"
)
assert _forecasters_used_match.group(2) == "3", (
    f"FORECASTERS_USED_MARKER_RE captured {_forecasters_used_match.groups()!r}; expected ('2', '3')"
)
del _forecasters_used_match
assert STACKER_META_ANALYSIS_HEADER.startswith("## "), (
    f"STACKER_META_ANALYSIS_HEADER must be a markdown H2 header, got {STACKER_META_ANALYSIS_HEADER!r}"
)
assert STACKED_BASE_REASONING_HEADER.startswith("## "), (
    f"STACKED_BASE_REASONING_HEADER must be a markdown H2 header, got {STACKED_BASE_REASONING_HEADER!r}"
)

__all__ = [
    "BASE_MODEL_SUBBLOCK_SPLIT_RE",
    "EXPLANATION_DETAIL_UNKNOWN_MARKER",
    "FORECASTERS_USED_MARKER_PREFIX",
    "FORECASTERS_USED_MARKER_RE",
    "HISTORICAL_STACKER_META_HEADER",
    "HISTORICAL_STACKER_SIGNATURE_RE",
    "STACKED_BASE_REASONING_HEADER",
    "STACKED_MARKER_FALSE",
    "STACKED_MARKER_RE",
    "STACKED_MARKER_TRUE",
    "STACKER_META_ANALYSIS_HEADER",
    "STACKER_OUTCOME_FALLBACK_LLM",
    "STACKER_OUTCOME_FALLBACK_MEAN",
    "STACKER_OUTCOME_FALLBACK_MEDIAN",
    "STACKER_OUTCOME_PRIMARY",
    "STACKER_OUTCOME_RE",
    "STACKER_OUTCOME_SKIPPED",
    "STACKER_OUTCOME_SKIPPED_CONFIG_OFF",
    "STACKER_SKIP_REASONS",
    "STACKER_SKIP_REASON_RE",
    "TOOLS_USED_MARKER_FALSE",
    "TOOLS_USED_MARKER_RE",
    "TOOLS_USED_MARKER_TRUE",
    "format_forecasters_used_marker",
    "format_stacker_skip_reason_marker",
]
