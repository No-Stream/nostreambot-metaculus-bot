"""Attribution guards for metaculus_bot.performance_analysis.parsing.

The cohort machinery, split out of ``tests/test_performance_analysis_parsing.py``
(which keeps the small per-parser unit classes). Two cohorts inherit the SAME
assertions from ``_RealCommentAttributionChecks`` so they cannot drift:

* ``TestMiniFixtureAttribution`` — the deterministic CI floor over the checked-in
  ``tests/data/performance_comments_mini.jsonl``. NOT skip-gated, by design; its
  trim-side counterpart is ``TestAgainstCheckedInMiniComments`` in
  ``tests/test_comment_trimming.py``.
* ``TestRealDataRegression`` — the broad local sweep over the gitignored
  ``scratch/performance_data.json``, skip-gated because that pull is absent in CI.

``TestFallAib2025Fixture`` pins the parser against a synthesized older-roster
comment, guarding the fall-aib-2025 Platt-fit parse-rate gate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from metaculus_bot.performance_analysis.parsing import (
    _parse_probability,
    parse_per_model_forecasts,
    parse_stacked_marker,
)

# Imported (not re-implemented) so the shape the fixture was DERIVED under and the
# shape the test RE-CHECKS can never drift into two different definitions. Same
# reason for ``parser_outputs``: it is both the fixture's faithfulness invariant
# and the exact expectation pinned below, so the two cannot cover different parsers.
from scripts.derive_mini_comment_fixture import comment_shape, parser_outputs

# ---------------------------------------------------------------------------
# Real-data regression test
# ---------------------------------------------------------------------------


# Big local performance pull. Gitignored, untracked, and rewritten by every
# ``spring-aib-2026`` collector run, so it is absent in CI and its contents shift
# between local runs. It stays the BROAD sweep (283 records, every era) but can
# never be the only cohort — see ``_MINI_FIXTURE_PATH``.
_PERF_DATA_PATH = Path(__file__).parent.parent / "scratch" / "performance_data.json"

# Checked-in miniature: one record per distinct comment shape, distilled from the
# big pull by ``scripts/derive_mini_comment_fixture.py`` under a hard invariant —
# every parser in ``parser_outputs`` returns identical output on the miniature and
# on its full-size source. (``parse_per_model_reasoning_text`` is public but out
# of scope by design: the shrink elides rationale prose, which is what it returns,
# so it diverges on every record while its key set survives.)
# This is the DETERMINISTIC CI floor for the same
# assertions, so the attribution guard runs on every PR instead of only for
# whoever happens to have pulled the big file. ``.jsonl`` (not ``.json``)
# because ``.gitignore`` carries a repo-wide ``*.json`` rule with no negations.
_MINI_FIXTURE_PATH = Path(__file__).parent / "data" / "performance_comments_mini.jsonl"

# Independent oracle for "does this comment carry a recoverable Model: line for
# forecaster index N". Deliberately NOT reusing the production
# ``_R1_MODEL_RE`` / ``parse_forecaster_model_map``: the whole point is to catch
# a regression *in* that regex, and an oracle built from it would move in
# lockstep and see nothing. Written instead from the documented comment format
# (``## R1: Forecaster N Reasoning`` immediately followed by ``Model: <path>``,
# injected by ``forecaster.TemplateForecaster._make_prediction``). Verified to
# agree with the production map on all 283 archived records at the time of
# writing, so a future divergence is signal, not noise.
_ORACLE_R1_MODEL_RE = re.compile(
    r"^[ \t]*##[ \t]+R1:[ \t]+Forecaster[ \t]+(\d+)[ \t]+Reasoning[ \t]*\r?\n[ \t]*Model:[ \t]*(\S[^\n]*)$",
    re.MULTILINE,
)

_ANONYMIZED_KEY_RE = re.compile(r"Forecaster (\d+)")

# The bare literal every summary bullet starts with. Used to tell "the parser
# found nothing because there is nothing to find" (the trim removed the whole
# `## Report 1 Summary / ### Forecasts` block) from "the parser found nothing
# although bullets are right there" — the second is a regression, the first is
# not. Deliberately a plain substring, not ``_FORECASTER_RE``: a regression IN
# that regex is exactly what this has to stay sensitive to.
_SUMMARY_BULLET_LITERAL = "*Forecaster"


def _oracle_recoverable_indices(comment: str) -> set[int]:
    """Forecaster indices whose model name is present in ``comment``.

    An index in this set MUST come back from the parser as a real model name —
    the attribution source is right there in the text. An index absent from it
    is structurally unattributable and correctly falls back to ``Forecaster N``.
    """
    return {
        int(match.group(1))
        for match in _ORACLE_R1_MODEL_RE.finditer(comment)
        if match.group(2).strip().rsplit("/", 1)[-1].strip()
    }


class _RealCommentAttributionChecks:
    """Attribution assertions run against a cohort of real archived comments.

    Subclasses supply ``records``. Both the checked-in miniature (deterministic,
    runs in CI) and the big local pull (broad, local-only) inherit the SAME
    assertions, so the two cohorts can't drift apart — a rule that holds only for
    the local sweep would be a rule CI never enforces.

    The checks make no assumption about which roster produced the comments. They
    assert only that every forecast whose model name is RECOVERABLE from the
    comment text comes back named, never index-anonymized `Forecaster N`.
    """

    def test_records_with_a_recoverable_model_name_parse_to_it(self, records):
        # A `Forecaster N` key is only a parser bug when the comment actually
        # contains that index's `Model:` line. Two archived cohorts legitimately
        # lack one and MUST fall back to the anonymized key:
        #
        #  * comments trimmed at the Metaculus char limit, which can erase a
        #    `Model:` line mid-rationale;
        #  * the 2026-04-04..04-13 stacking-era cohort, where a fired stacker
        #    published its aggregate via a `ReasonedPrediction` constructed
        #    directly in `_aggregate_predictions` — bypassing the
        #    `Model: {llm.model}` prefix that `_make_prediction` adds to every
        #    per-forecaster rationale. Those comments carry no `Model:` line
        #    anywhere, so no parser can name them.
        #
        # Keying on per-index recoverability instead of either cohort marker
        # makes the assertion both stricter and era-agnostic: it now flags a
        # dropped attribution even inside a trimmed comment, which the old
        # "not trimmed" proxy waved through.
        total = 0
        fully_parsed = 0
        unattributable = []
        no_bullets = []
        parser_bad = []
        for rec in records:
            comment = rec.get("comment_text") or ""
            if not comment:
                continue
            total += 1
            parsed = parse_per_model_forecasts(comment)
            recoverable = _oracle_recoverable_indices(comment)
            anonymized = [k for k in parsed if k.startswith("Forecaster ")]
            dropped = [
                key
                for key in anonymized
                if (match := _ANONYMIZED_KEY_RE.fullmatch(key)) and int(match.group(1)) in recoverable
            ]
            if not parsed:
                # Parsed NOTHING. Counting this as "clean" (it has no anonymized
                # keys, after all) would let a parser that returned {} for every
                # comment pass at a 1.00 ratio — measured, on both cohorts. It is
                # not a clean parse and it is not an attribution failure either;
                # it gets its own bucket. Legitimate ONLY when the comment has no
                # summary bullet to find; otherwise the parser missed one.
                if _SUMMARY_BULLET_LITERAL in comment:
                    parser_bad.append((rec["post_id"], ["<no forecasts parsed from a comment that has bullets>"]))
                else:
                    no_bullets.append(rec["post_id"])
            elif not anonymized:
                fully_parsed += 1
            elif dropped:
                parser_bad.append((rec["post_id"], dropped))
            else:
                unattributable.append(rec["post_id"])
        assert not parser_bad, (
            "Parser dropped a forecast it had the evidence to recover "
            f"(anonymized despite a present Model: line, or found no bullets at all): {parser_bad[:10]}"  # HARNESS-SCAN-EXEMPT-subsampling  # error-message display truncation, not data subsampling
        )
        # Coverage floor over the records that HAVE an attribution source, so a
        # shift in the unattributable cohort's size can't mask a real drop.
        # Bullet-less records are excluded from BOTH sides: they can't be parsed
        # cleanly, so leaving them in the denominator alone would penalize the
        # parser for a trim, and leaving them in the numerator was the hole above.
        attributable = total - len(unattributable) - len(no_bullets)
        assert attributable > 0, "No records carry an attribution source; fixture is unusable"
        assert fully_parsed / attributable >= 0.90, (
            f"Only {fully_parsed}/{attributable} attributable records parsed cleanly "
            f"({len(unattributable)} of {total} records carry no Model: line at all; "
            f"{len(no_bullets)} carry no summary bullets at all)"
        )

    def test_stacking_era_cohort_is_structurally_unattributable(self, records):
        # Guards the premise of the exemption above: the 2026-04 stacking-era
        # comments are unattributable because the text carries NO `Model:` line,
        # not because the parser gave up. If a future fixture refresh brings in
        # comments that do carry one, this fails and the exemption gets re-derived
        # rather than silently swallowing a real regression.
        for rec in records:
            comment = rec.get("comment_text") or ""
            if not comment:
                continue
            parsed = parse_per_model_forecasts(comment)
            if not [k for k in parsed if k.startswith("Forecaster ")]:
                continue
            if _oracle_recoverable_indices(comment):
                # Has a source for SOME index; the per-index assertion above owns
                # whether the specific anonymized index was recoverable.
                continue
            assert "Model:" not in comment, (
                f"post {rec['post_id']} was treated as unattributable but contains a Model: "
                "line the oracle did not match — refresh the oracle or the exemption"
            )


# Exact per-record parse expectations for the checked-in miniature, keyed by
# ``post_id``. GENERATED — refresh with:
#
#     uv run python scripts/derive_mini_comment_fixture.py --emit-expectations
#
# Why exact values and not just the clean-parse ratio in
# ``test_records_with_a_recoverable_model_name_parse_to_it``: that check passes at
# a floor below 1.0, so over a twelve-record fixture it tolerates one silently
# wrong record. Because the fixture is PINNED, the expectation here can instead be
# the complete parse of every record, and one moved value fails. Both checks stay:
# the ratio catches a broad collapse, this catches a single record.
#
# CHARACTERIZATION, not specification: every value below was produced by running
# the real parsers over the fixture, so regenerating after a parser change will
# happily bless that change. A diff here is a signal to READ the diff and confirm
# each moved value is intended.
#
# Two entries look wrong and are not. ``parse_per_model_forecasts`` returns
# whatever text follows the bullet's colon, and only binary questions put a
# scalar there: multiple-choice bullets are followed by a newline and an option
# list, so the value is that list's FIRST line (``- 0: 12.0%``), and numeric
# bullets carry the literal ``Probability distribution:`` ahead of their
# percentile lines. Pinning those strings pins real current behavior; the
# per-type values a caller actually wants live in
# ``parse_per_model_mc_option_probs`` and ``parse_per_model_numeric_percentiles``,
# pinned alongside them.

_EXPECTED_PARSES_BY_POST = {
    41517: {
        "parse_per_model_forecasts": {
            "gpt-5.2": "- 0: 12.0%",
            "gpt-5": "- 0: 10.0%",
            "claude-opus-4.5": "- 0: 5.0%",
            "gemini-3-flash-preview": "- 0: 5.0%",
            "gemini-3-pro-preview": "- 0: 2.0%",
        },
        "parse_forecaster_model_map": {
            1: "gpt-5.2",
            2: "gpt-5",
            3: "claude-opus-4.5",
            4: "gemini-3-flash-preview",
            5: "gemini-3-pro-preview",
        },
        "parse_per_model_numeric_percentiles": {},
        "parse_per_model_mc_option_probs": {
            "gpt-5.2": {"0": 0.12, "1": 0.3, "2": 0.38, "3+": 0.2},
            "gpt-5": {"0": 0.1, "1": 0.56, "2": 0.26, "3+": 0.08},
            "claude-opus-4.5": {"0": 0.05, "1": 0.33, "2": 0.42, "3+": 0.2},
            "gemini-3-flash-preview": {"0": 0.05, "1": 0.28, "2": 0.46, "3+": 0.21},
            "gemini-3-pro-preview": {"0": 0.02, "1": 0.5, "2": 0.3, "3+": 0.18},
        },
        "parse_inferred_stacker_outcome": (None, "none"),
        "parse_stacked_marker": None,
    },
    41518: {
        "parse_per_model_forecasts": {
            "gpt-5.2": "28.0%",
            "gpt-5": "20.0%",
            "claude-opus-4.5": "15.0%",
            "gemini-3-flash-preview": "28.0%",
            "gemini-3-pro-preview": "12.0%",
        },
        "parse_forecaster_model_map": {
            1: "gpt-5.2",
            2: "gpt-5",
            3: "claude-opus-4.5",
            4: "gemini-3-flash-preview",
            5: "gemini-3-pro-preview",
        },
        "parse_per_model_numeric_percentiles": {},
        "parse_per_model_mc_option_probs": {},
        "parse_inferred_stacker_outcome": (None, "none"),
        "parse_stacked_marker": None,
    },
    41537: {
        "parse_per_model_forecasts": {
            "gpt-5.2": "Probability distribution:",
            "gpt-5": "Probability distribution:",
            "gemini-3-flash-preview": "Probability distribution:",
            "gemini-3-pro-preview": "Probability distribution:",
        },
        "parse_forecaster_model_map": {
            1: "gpt-5.2",
            2: "gpt-5",
            3: "gemini-3-flash-preview",
            4: "gemini-3-pro-preview",
        },
        "parse_per_model_numeric_percentiles": {
            "gpt-5.2": [
                (2.5, 73000000000.0),
                (5.0, 74000000000.0),
                (10.0, 75500000000.0),
                (20.0, 77500000000.0),
                (40.0, 79300000000.0),
                (50.0, 80200000000.0),
                (60.0, 81000000000.0),
                (80.0, 83000000000.0),
                (90.0, 85000000000.0),
                (95.0, 86800000000.0),
                (97.5, 88000000000.0),
            ],
            "gpt-5": [
                (2.5, 76800000000.0),
                (5.0, 77400000000.0),
                (10.0, 78000000000.0),
                (20.0, 78800000000.0),
                (40.0, 79800000000.0),
                (50.0, 80300000000.0),
                (60.0, 80800000000.0),
                (80.0, 81800000000.0),
                (90.0, 82500000000.0),
                (95.0, 83100000000.0),
                (97.5, 83700000000.0),
            ],
            "gemini-3-flash-preview": [
                (2.5, 75200000000.0),
                (5.0, 76800000000.0),
                (10.0, 78100000000.0),
                (20.0, 79200000000.0),
                (40.0, 80250000000.0),
                (50.0, 80700000000.0),
                (60.0, 81150000000.0),
                (80.0, 82400000000.0),
                (90.0, 83600000000.0),
                (95.0, 84800000000.0),
                (97.5, 86500000000.0),
            ],
            "gemini-3-pro-preview": [
                (2.5, 78500000000.0),
                (5.0, 79200000000.0),
                (10.0, 79800000000.0),
                (20.0, 80500000000.0),
                (40.0, 81400000000.0),
                (50.0, 81800000000.0),
                (60.0, 82300000000.0),
                (80.0, 83400000000.0),
                (90.0, 84200000000.0),
                (95.0, 85000000000.0),
                (97.5, 86200000000.0),
            ],
        },
        "parse_per_model_mc_option_probs": {},
        "parse_inferred_stacker_outcome": (None, "none"),
        "parse_stacked_marker": None,
    },
    41841: {
        "parse_per_model_forecasts": {
            "gpt-5.2": "Probability distribution:",
            "gpt-5": "Probability distribution:",
            "gemini-3-flash-preview": "Probability distribution:",
            "gemini-3-pro-preview": "Probability distribution:",
        },
        "parse_forecaster_model_map": {
            1: "gpt-5.2",
            2: "gpt-5",
            3: "gemini-3-flash-preview",
            4: "gemini-3-pro-preview",
        },
        "parse_per_model_numeric_percentiles": {
            "gpt-5.2": [
                (2.5, 55.6),
                (5.0, 56.8),
                (10.0, 58.4),
                (20.0, 60.6),
                (40.0, 62.9),
                (50.0, 64.0),
                (60.0, 65.1),
                (80.0, 67.2),
                (90.0, 68.5),
                (95.0, 69.3),
                (97.5, 70.0),
            ],
            "gpt-5": [
                (2.5, 56.5),
                (5.0, 57.5),
                (10.0, 58.5),
                (20.0, 60.0),
                (40.0, 62.0),
                (50.0, 63.0),
                (60.0, 64.0),
                (80.0, 66.0),
                (90.0, 67.2),
                (95.0, 68.5),
                (97.5, 69.8),
            ],
            "gemini-3-flash-preview": [
                (2.5, 57.5),
                (5.0, 58.8),
                (10.0, 60.1),
                (20.0, 61.8),
                (40.0, 63.6),
                (50.0, 64.4),
                (60.0, 65.2),
                (80.0, 66.9),
                (90.0, 68.3),
                (95.0, 69.4),
                (97.5, 70.3),
            ],
            "gemini-3-pro-preview": [
                (2.5, 56.5),
                (5.0, 57.4),
                (10.0, 58.6),
                (20.0, 59.9),
                (40.0, 61.5),
                (50.0, 62.3),
                (60.0, 63.1),
                (80.0, 65.2),
                (90.0, 66.8),
                (95.0, 67.9),
                (97.5, 68.8),
            ],
        },
        "parse_per_model_mc_option_probs": {},
        "parse_inferred_stacker_outcome": (None, "none"),
        "parse_stacked_marker": None,
    },
    41848: {
        "parse_per_model_forecasts": {},
        "parse_forecaster_model_map": {
            1: "gpt-5.2",
            2: "gpt-5",
            3: "claude-opus-4.5",
            4: "gemini-3-flash-preview",
            5: "gemini-3-pro-preview",
        },
        "parse_per_model_numeric_percentiles": {},
        "parse_per_model_mc_option_probs": {},
        "parse_inferred_stacker_outcome": (None, "none"),
        "parse_stacked_marker": None,
    },
    41871: {
        "parse_per_model_forecasts": {},
        "parse_forecaster_model_map": {
            1: "gpt-5.2",
            2: "gpt-5",
            3: "claude-opus-4.5",
            4: "gemini-3-flash-preview",
            5: "gemini-3-pro-preview",
        },
        "parse_per_model_numeric_percentiles": {
            "gpt-5.2": [
                (2.5, 6100000000.0),
                (5.0, 6200000000.0),
                (10.0, 6400000000.0),
                (20.0, 6700000000.0),
                (40.0, 7000000000.0),
                (50.0, 7200000000.0),
                (60.0, 7350000000.0),
                (80.0, 7650000000.0),
                (90.0, 7800000000.0),
                (95.0, 7900000000.0),
                (97.5, 7970000000.0),
            ],
            "gpt-5": [
                (2.5, 6250000000.0),
                (5.0, 6350000000.0),
                (10.0, 6475000000.0),
                (20.0, 6625000000.0),
                (40.0, 6825000000.0),
                (50.0, 6950000000.0),
                (60.0, 7075000000.0),
                (80.0, 7300000000.0),
                (90.0, 7475000000.0),
                (95.0, 7600000000.0),
                (97.5, 7725000000.0),
            ],
            "claude-opus-4.5": [
                (2.5, 6100000000.0),
                (5.0, 6200000000.0),
                (10.0, 6350000000.0),
                (20.0, 6550000000.0),
                (40.0, 6850000000.0),
                (50.0, 7050000000.0),
                (60.0, 7200000000.0),
                (80.0, 7450000000.0),
                (90.0, 7650000000.0),
                (95.0, 7800000000.0),
                (97.5, 7950000000.0),
            ],
            "gemini-3-flash-preview": [
                (2.5, 6150000000.0),
                (5.0, 6300000000.0),
                (10.0, 6500000000.0),
                (20.0, 6750000000.0),
                (40.0, 7000000000.0),
                (50.0, 7100000000.0),
                (60.0, 7200000000.0),
                (80.0, 7450000000.0),
                (90.0, 7650000000.0),
                (95.0, 7800000000.0),
                (97.5, 7950000000.0),
            ],
            "gemini-3-pro-preview": [
                (2.5, 6250000000.0),
                (5.0, 6450000000.0),
                (10.0, 6600000000.0),
                (20.0, 6800000000.0),
                (40.0, 7050000000.0),
                (50.0, 7150000000.0),
                (60.0, 7250000000.0),
                (80.0, 7550000000.0),
                (90.0, 7750000000.0),
                (95.0, 7900000000.0),
                (97.5, 7980000000.0),
            ],
        },
        "parse_per_model_mc_option_probs": {},
        "parse_inferred_stacker_outcome": (None, "none"),
        "parse_stacked_marker": None,
    },
    42108: {
        "parse_per_model_forecasts": {
            "gpt-5.2": "37.0%",
            "gpt-5.1": "42.0%",
            "claude-4.6-opus": "35.0%",
            "claude-opus-4.5": "38.0%",
            "gemini-3-pro-preview": "55.0%",
        },
        "parse_forecaster_model_map": {
            1: "gpt-5.2",
            2: "gpt-5.1",
            3: "claude-4.6-opus",
            4: "claude-opus-4.5",
            5: "gemini-3-pro-preview",
        },
        "parse_per_model_numeric_percentiles": {},
        "parse_per_model_mc_option_probs": {},
        "parse_inferred_stacker_outcome": (None, "none"),
        "parse_stacked_marker": None,
    },
    42110: {
        "parse_per_model_forecasts": {
            "Forecaster 1": "- Maria Lazar: 10.0%",
            "gpt-5.1": "- Maria Lazar: 9.0%",
            "claude-4.6-opus": "- Maria Lazar: 4.0%",
            "claude-opus-4.5": "- Maria Lazar: 9.0%",
            "gemini-3-pro-preview": "- Maria Lazar: 6.0%",
        },
        "parse_forecaster_model_map": {
            2: "gpt-5.1",
            3: "claude-4.6-opus",
            4: "claude-opus-4.5",
            5: "gemini-3-pro-preview",
        },
        "parse_per_model_numeric_percentiles": {},
        "parse_per_model_mc_option_probs": {
            "Forecaster 1": {"Maria Lazar": 0.1, "Chris Taylor": 0.89, "Someone else": 0.01},
            "gpt-5.1": {"Maria Lazar": 0.09, "Chris Taylor": 0.89, "Someone else": 0.02},
            "claude-4.6-opus": {"Maria Lazar": 0.04, "Chris Taylor": 0.95, "Someone else": 0.01},
            "claude-opus-4.5": {"Maria Lazar": 0.09, "Chris Taylor": 0.9, "Someone else": 0.01},
            "gemini-3-pro-preview": {"Maria Lazar": 0.06, "Chris Taylor": 0.93, "Someone else": 0.01},
        },
        "parse_inferred_stacker_outcome": (None, "none"),
        "parse_stacked_marker": None,
    },
    42113: {
        "parse_per_model_forecasts": {
            "Forecaster 1": "12.0%",
            "gpt-5.1": "7.0%",
            "claude-4.6-opus": "1.0%",
            "claude-opus-4.5": "9.0%",
            "gemini-3-pro-preview": "1.0%",
        },
        "parse_forecaster_model_map": {
            2: "gpt-5.1",
            3: "claude-4.6-opus",
            4: "claude-opus-4.5",
            5: "gemini-3-pro-preview",
        },
        "parse_per_model_numeric_percentiles": {},
        "parse_per_model_mc_option_probs": {},
        "parse_inferred_stacker_outcome": (None, "none"),
        "parse_stacked_marker": None,
    },
    43053: {
        "parse_per_model_forecasts": {"Forecaster 1": "- Ismaïl Omar Guelleh officially declared winner: 92.0%"},
        "parse_forecaster_model_map": {},
        "parse_per_model_numeric_percentiles": {},
        "parse_per_model_mc_option_probs": {
            "Forecaster 1": {
                "Ismaïl Omar Guelleh officially declared winner": 0.92,
                "Another candidate officially declared winner": 0.02,
                "Vote held but no final official result before May 1": 0.02,
                "Election postponed or cancelled": 0.04,
            }
        },
        "parse_inferred_stacker_outcome": ("primary", "historical_body"),
        "parse_stacked_marker": None,
    },
    43054: {
        "parse_per_model_forecasts": {"Forecaster 1": "10.0%"},
        "parse_forecaster_model_map": {},
        "parse_per_model_numeric_percentiles": {},
        "parse_per_model_mc_option_probs": {},
        "parse_inferred_stacker_outcome": (None, "none"),
        "parse_stacked_marker": None,
    },
    43059: {
        "parse_per_model_forecasts": {"Forecaster 1": "Probability distribution:"},
        "parse_forecaster_model_map": {},
        "parse_per_model_numeric_percentiles": {
            "Forecaster 1": [
                (2.5, 87.0),
                (5.0, 95.0),
                (10.0, 105.0),
                (20.0, 119.0),
                (40.0, 137.0),
                (50.0, 146.0),
                (60.0, 155.0),
                (80.0, 172.0),
                (90.0, 184.0),
                (95.0, 193.0),
                (97.5, 197.0),
            ]
        },
        "parse_per_model_mc_option_probs": {},
        "parse_inferred_stacker_outcome": ("primary", "historical_body"),
        "parse_stacked_marker": None,
    },
}


class TestMiniFixtureAttribution(_RealCommentAttributionChecks):
    """The CI floor: same attribution checks, on the checked-in miniature.

    Not skipif-gated — the fixture is committed, so these run on every PR. If the
    file is ever lost, the failure is a loud missing-file error rather than a
    silent skip, which is the whole point of this class existing.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def records(cls):
        with _MINI_FIXTURE_PATH.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_fixture_covers_every_shape_the_parsers_branch_on(self, records):
        # The miniature is only a meaningful floor if it still spans the comment
        # shapes the parsers treat differently. Each record carries the shape it
        # was selected for; assert the set is intact and duplicate-free so a
        # careless hand-edit or a stale regeneration can't quietly narrow
        # coverage to, say, only well-formed modern comments.
        shapes = [rec["_shape"] for rec in records]
        shape_keys = [tuple(sorted(shape.items())) for shape in shapes]
        assert len(shape_keys) == len(set(shape_keys)), f"duplicate shapes in miniature: {shapes}"

        # Axes are read BY NAME: positional access would silently check the wrong
        # axis if ``comment_shape`` ever gained or reordered a field.
        expected_values = {
            "attribution": {"map", "nomap"},
            "trim": {"trim", "intact"},
            "boundary_marker": {"res", "nores"},
            "naming": {"named", "anon"},
            # Every question type the pipeline publishes, so a type-specific
            # parser regression can't hide behind a binary-only fixture.
            "question_type": {"binary", "numeric", "discrete", "multiple_choice"},
        }
        assert {axis for shape in shapes for axis in shape} == set(expected_values), (
            "miniature shape axes do not match what this test knows how to check; "
            "comment_shape changed — update expected_values"
        )
        for axis, expected in expected_values.items():
            observed = {shape[axis] for shape in shapes}
            assert observed == expected, f"miniature must cover every {axis} value: expected {expected}, got {observed}"

    def test_miniature_still_matches_its_recorded_shape(self, records):
        # Guards the fixture against edits that change what a record exercises.
        # `_shape` is metadata written at derivation time; recomputing it from the
        # comment text catches a hand-edited comment whose label no longer fits.
        for rec in records:
            recomputed = comment_shape(rec)
            assert recomputed == rec["_shape"], (
                f"post {rec['post_id']} no longer matches its recorded shape: "
                f"{rec['_shape']} -> {recomputed}; regenerate with "
                "scripts/derive_mini_comment_fixture.py"
            )

    def test_expectation_table_covers_exactly_the_fixture(self, records):
        # A record with no table entry must FAIL rather than go unchecked, and a
        # table entry for a record the fixture no longer has is dead weight that
        # hides a shrunken fixture. Compared as sets so both directions report.
        assert {rec["post_id"] for rec in records} == set(_EXPECTED_PARSES_BY_POST), (
            "expectation table and miniature disagree on which posts exist; regenerate with "
            "scripts/derive_mini_comment_fixture.py --emit-expectations"
        )

    def test_each_record_parses_to_its_exact_expectation(self, records):
        # The exact-value floor. Every OTHER check on this fixture is aggregate
        # (a >=0.90 clean-parse ratio) or structural (shape coverage, shape
        # consistency), and none of them says WHICH indices resolve to WHICH
        # model names — so a single record silently losing a model, or reporting
        # a perturbed probability, passes them all. Asserting the whole parse
        # dict per record is what makes one wrong value a failure.
        #
        # Compares the full multi-parser mapping, not just
        # ``parse_per_model_forecasts``: the fixture deliberately spans all four
        # question types, and pinning only the binary-shaped parser would leave
        # the numeric/MC/stacker-inference parsers on aggregate-only coverage —
        # the same hole this closes. ``parser_outputs`` is imported rather than
        # re-listed so the pinned set cannot drift from the set the fixture's
        # faithfulness invariant was established under.
        for rec in records:
            post_id = rec["post_id"]
            assert parser_outputs(rec["comment_text"]) == _EXPECTED_PARSES_BY_POST[post_id], (
                f"post {post_id} no longer parses to its pinned expectation; if the change is "
                "intended, review each moved value then regenerate with "
                "scripts/derive_mini_comment_fixture.py --emit-expectations"
            )


@pytest.mark.skipif(
    not _PERF_DATA_PATH.exists(),
    reason=(
        f"local-only breadth cohort: {_PERF_DATA_PATH.name} is a gitignored collector pull "
        "(rewritten by every spring-aib-2026 run), absent in CI. The same attribution checks "
        "run unconditionally in TestMiniFixtureAttribution over the checked-in "
        "tests/data/performance_comments_mini.jsonl, which is the CI floor."
    ),
)
class TestRealDataRegression(_RealCommentAttributionChecks):
    """The broad local sweep: same checks across all 283 records, every era.

    Local-only by nature (the pull is gitignored and rewritten by the collector).
    Kept because breadth catches shapes the miniature hasn't been taught yet;
    ``TestMiniFixtureAttribution`` is what guarantees the checks run at all.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def records(cls):
        # Re-check existence rather than trusting the class-level skipif: that
        # predicate is evaluated once at COLLECTION, and the collector rewrites
        # this file in place, so a pull landing mid-run would turn a skip into a
        # FileNotFoundError at setup.
        if not _PERF_DATA_PATH.exists():
            pytest.skip("big local performance pull disappeared between collection and setup")
        with open(_PERF_DATA_PATH) as f:
            return json.load(f)

    def test_known_sample_post_matches_expected_models(self, records):
        # post 42631 (Oscar winner question) is in the March cohort — sampled
        # in plan exploration and known to use the old roster.
        sample = next((r for r in records if r["post_id"] == 42631), None)
        if sample is None:
            pytest.skip("sample post 42631 not in data")
        assert sample is not None  # narrows for the type checker (pytest.skip raises)
        if "openrouter/openai/gpt-5.2" not in sample["comment_text"]:
            pytest.skip("post 42631 no longer from March roster; refresh test fixture or delete")
        parsed = parse_per_model_forecasts(sample["comment_text"])
        expected_values = {
            "gpt-5.2": "56.0%",
            "gpt-5.1": "57.0%",
            "claude-4.6-opus": "52.0%",
            "claude-opus-4.5": "52.0%",
            "gemini-3.1-pro-preview": "58.0%",
        }
        assert parsed == expected_values


# ---------------------------------------------------------------------------
# F12 — fall-aib-2025 fixture: the binary Platt fit was unstable between
# fall-aib-2025 and spring-aib-2026 (slope drift > 0.3). The Platt plan
# documented an >=80% parse-rate stability gate. Lock in the parser's
# performance on a representative fall-aib-2025 comment so we catch any
# regression that would silently invalidate the fit.
# ---------------------------------------------------------------------------


# Synthesized from a real fall-aib-2025 binary comment shape (post 41137-era):
# 6-model ensemble with the older roster (gpt-5.1, o3, claude-sonnet-4.5,
# grok-4.1-fast, qwen3-235b-a22b-thinking-2507, kimi-k2-0905). Structurally
# identical to a current-vintage comment, just with different model paths
# and slightly older summary header content.
FALL_AIB_2025_FIXTURE = """# SUMMARY
*Question*: Will a sixth contentious case be opened at the International Court of Justice in 2025?
*Final Prediction*: 21.0%
*Total Cost*: $0.0775
*Time Spent*: 8.64 minutes


## Report 1 Summary
### Forecasts
*Forecaster 1*: 25.0%
*Forecaster 2*: 33.0%
*Forecaster 3*: 22.0%
*Forecaster 4*: 10.0%
*Forecaster 5*: 20.0%
*Forecaster 6*: 10.0%


### Research Summary
Brief research summary covering recent ICJ news.

================================================================================
FORECAST SECTION:

## R1: Forecaster 1 Reasoning
Model: openrouter/openai/gpt-5.1

Analysis text. Final probability: 25%

## R1: Forecaster 2 Reasoning
Model: openrouter/openai/o3

Analysis text. Final probability: 33%

## R1: Forecaster 3 Reasoning
Model: openrouter/anthropic/claude-sonnet-4.5

Analysis text. Final probability: 22%

## R1: Forecaster 4 Reasoning
Model: openrouter/x-ai/grok-4.1-fast

Analysis text. Final probability: 10%

## R1: Forecaster 5 Reasoning
Model: openrouter/qwen/qwen3-235b-a22b-thinking-2507

Analysis text. Final probability: 20%

## R1: Forecaster 6 Reasoning
Model: openrouter/moonshotai/kimi-k2-0905

Analysis text. Final probability: 10%

<!-- STACKED=false -->
"""


class TestFallAib2025Fixture:
    def test_parse_rate_meets_eighty_percent_threshold(self):
        # All 6 bullets must resolve to NAMED model keys (no anonymized
        # ``Forecaster N`` placeholders). Parse rate = 6/6 = 100%, well
        # above the 80% gate.
        result = parse_per_model_forecasts(FALL_AIB_2025_FIXTURE)
        named_keys = [k for k in result if not k.startswith("Forecaster ")]
        total = len(result)
        assert total == 6, f"Expected 6 bullets, parsed {total}"
        parse_rate = len(named_keys) / total
        assert parse_rate >= 0.80, f"Parse rate {parse_rate:.0%} below 80% gate"

    def test_old_roster_model_names_extracted(self):
        # Specific check: the older roster names must come through cleanly.
        result = parse_per_model_forecasts(FALL_AIB_2025_FIXTURE)
        expected_models = {
            "gpt-5.1",
            "o3",
            "claude-sonnet-4.5",
            "grok-4.1-fast",
            "qwen3-235b-a22b-thinking-2507",
            "kimi-k2-0905",
        }
        assert set(result.keys()) == expected_models

    def test_values_attributed_correctly(self):
        result = parse_per_model_forecasts(FALL_AIB_2025_FIXTURE)
        assert result["gpt-5.1"] == "25.0%"
        assert result["o3"] == "33.0%"
        assert result["claude-sonnet-4.5"] == "22.0%"
        assert result["grok-4.1-fast"] == "10.0%"
        assert result["qwen3-235b-a22b-thinking-2507"] == "20.0%"
        assert result["kimi-k2-0905"] == "10.0%"

    def test_probabilities_parse_to_valid_range(self):
        # The full Platt fit pipeline runs _parse_probability on each value;
        # confirm none of them get dropped by the F11 heuristic tightening.
        result = parse_per_model_forecasts(FALL_AIB_2025_FIXTURE)
        parsed = {k: _parse_probability(v) for k, v in result.items()}
        for model, prob in parsed.items():
            assert prob is not None, f"{model} dropped by parser"
            assert 0.0 <= prob <= 1.0

    def test_legacy_stacked_marker_parsed(self):
        # Fall-aib-2025 comments use the legacy STACKED= marker. Confirm
        # the marker reader still picks it up.
        assert parse_stacked_marker(FALL_AIB_2025_FIXTURE) is False
