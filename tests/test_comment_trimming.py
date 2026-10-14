"""Tests for comment.trimming.

The structured trim path (preserve summary head + tail) protects
residual-analysis parsers that extract per-forecaster predictions and the
STACKED marker from bot comments. Breaking this silently would degrade every
future performance-analysis pull, so the tests here combine unit coverage,
end-to-end coverage against realistic comment structure, and an explicit
parser-compatibility check that runs the live parsing.py regexes.
"""

import json
import math
from typing import ClassVar

import pytest
from forecasting_tools import ReasonedPrediction

from metaculus_bot.comment.markers import EXPLANATION_DETAIL_UNKNOWN_MARKER, STACKED_BASE_REASONING_HEADER
from metaculus_bot.comment.trimming import (
    TRIM_NOTICE,
    TrimConfig,
    _allocate_block_budgets,
    trim_comment,
    trim_section,
)
from metaculus_bot.constants import (
    COMMENT_CHAR_LIMIT,
    FORECASTS_SECTION_CHAR_LIMIT,
    RESEARCH_SECTION_CHAR_LIMIT,
    SUMMARY_SECTION_CHAR_LIMIT,
)
from metaculus_bot.prompts import SUMMARIZER_SOFT_FAIL_BANNER
from metaculus_bot.research.provider_diagnostics import PROVIDER_DIAGNOSTICS_HEADER
from metaculus_bot.stacking import combine_stacker_and_base_reasoning

# ---------------------------------------------------------------------------
# Legacy section trim (unchanged behavior — keeps header, uses tail)
# ---------------------------------------------------------------------------


def test_trim_section_preserves_header_and_tail() -> None:
    header = "## Report 1 Summary"
    # "unit-test-section" has no known suffix, so it routes to the generic
    # section_limit default (FORECASTS_SECTION_CHAR_LIMIT).
    body_length = FORECASTS_SECTION_CHAR_LIMIT + 512
    body = "A" * body_length
    original = f"{header}\n{body}"

    trimmed = trim_section(original, "unit-test-section")

    assert trimmed.splitlines()[0] == header
    assert TRIM_NOTICE in trimmed.splitlines()[1]
    assert len(trimmed) == FORECASTS_SECTION_CHAR_LIMIT

    available = FORECASTS_SECTION_CHAR_LIMIT - len(header) - len(TRIM_NOTICE) - 2
    assert available > 0
    expected_tail = body[-available:]
    assert trimmed.endswith(expected_tail)


def test_trim_section_noop_when_within_limit() -> None:
    original = "## Report 1 Summary\nshort body"
    trimmed = trim_section(original, "unit-test-noop")
    assert trimmed == original


def test_overflow_preserves_optional_research_before_single_newline_diagnostics_boundary() -> None:
    large_paragraph = "x" * 330
    recent_evidence = "Recent primary-source update remains available."
    comment = _build_sectioned_comment(
        summary_body="*Forecaster 1*: 65%",
        research_body=(
            f"{large_paragraph}\n\n{recent_evidence}\n{PROVIDER_DIAGNOSTICS_HEADER}\nlost=summarizer:error(timeout)"
        ),
        forecasts_body=(
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/provider/model-1\n\n"
            + ("A concise rationale. " * 8)
            + '\n\n```json\n{"probability": 0.65}\n```'
        ),
    )
    assert len(comment) > 800

    trimmed = trim_comment(comment, config=TrimConfig(comment_limit=800))

    assert len(trimmed) <= 800
    assert recent_evidence in trimmed
    assert "lost=summarizer:error(timeout)" in trimmed


def test_incomplete_fallback_stays_within_limit_when_disclosures_alone_overflow() -> None:
    # Each diagnostics block runs to the end of its collection, so enough collections make the
    # disclosures alone exceed the budget even after every other part has been compacted.
    limit = 2_000
    diagnostics = f"{PROVIDER_DIAGNOSTICS_HEADER}\n" + "\n".join(
        f"- provider_{index}: fallback | lost=source_{index}:error(timeout)" for index in range(40)
    )
    research_body = f"evidence 1\n\n{diagnostics}" + "".join(
        f"\n## Report {report} Research\nevidence {report}\n\n{diagnostics}" for report in range(2, 6)
    )
    comment = _build_sectioned_comment(
        summary_body="*Forecaster 1*: 65%",
        research_body=research_body,
        forecasts_body=(
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/provider/model-1\n\n"
            '```json\n{"evidence": "' + ("x" * limit) + '"}\n```'
        ),
    )

    trimmed = trim_comment(comment, config=TrimConfig(comment_limit=limit))

    assert len(trimmed) <= limit
    assert trimmed.startswith("# SUMMARY")
    assert EXPLANATION_DETAIL_UNKNOWN_MARKER in trimmed
    assert "<!-- STACKED=true -->" in trimmed
    assert "<!-- TOOLS_USED=false -->" in trimmed


def test_report_heading_inside_research_content_does_not_split_collections() -> None:
    # The framework renders collection content at heading level 3 or deeper, so only a level-2
    # "## Report N Research" line opens a collection. A deeper look-alike stays inside report 1,
    # and report 1's diagnostics block (which runs to the end of its collection) is kept whole.
    diagnostics = f"{PROVIDER_DIAGNOSTICS_HEADER}\n- asknews: fallback | lost=summarizer:error(timeout)"
    research_body = (
        ("optional evidence " * 500)
        + f"\n\n{diagnostics}\n\n### Report 2 Research\nLook-alike heading kept with the diagnostics. "
        + ("diagnostic detail " * 500)
    )
    comment = _build_sectioned_comment(
        summary_body="*Forecaster 1*: 65%",
        research_body=research_body,
        forecasts_body=(
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/provider/model-1\n\n"
            'A concise rationale.\n\n```json\n{"probability": 0.65}\n```'
        ),
    )

    trimmed = trim_comment(comment, config=TrimConfig(comment_limit=len(comment) - 3_000))

    assert "lost=summarizer:error(timeout)" in trimmed
    assert "Look-alike heading kept with the diagnostics." in trimmed
    assert "Model: openrouter/provider/model-1" in trimmed


# ---------------------------------------------------------------------------
# Fallback: tail-only trim when the summary marker is absent
# ---------------------------------------------------------------------------


def test_trim_final_comment_falls_back_to_tail_when_no_marker() -> None:
    # Payload with neither "### Research Summary" nor # RESEARCH / # FORECASTS
    # structure must take the last-resort header-preserving fallback. This is
    # the exact crash scenario from the 2026-06-05 log: a "# SUMMARY ..." blob
    # with no recognizable sections that overflows the limit. The fallback must
    # preserve the leading "#" line so the framework's
    # validate_explanation_starts_with_hash validator never rejects it.
    header = "# SUMMARY"
    payload = header + "\n" + ("0123456789" * math.ceil((COMMENT_CHAR_LIMIT + 250) / 10))
    trimmed = trim_comment(payload)

    assert trimmed.lstrip().startswith("#"), "output must remain '#'-leading (validator invariant)"
    assert trimmed.startswith(header)
    assert TRIM_NOTICE in trimmed
    assert len(trimmed) == COMMENT_CHAR_LIMIT
    # The tail of the original payload must survive after header + notice.
    available = COMMENT_CHAR_LIMIT - len(header) - len(TRIM_NOTICE) - 2
    assert available > 0
    assert trimmed.endswith(payload[-available:])


def test_trim_final_comment_noop_when_short() -> None:
    payload = "Concise explanation"
    trimmed = trim_comment(payload)
    assert trimmed == payload


# ---------------------------------------------------------------------------
# Structured trim (new path): preserves summary + tail, drops middle
# ---------------------------------------------------------------------------


def _build_realistic_comment(
    *,
    summary_bullets: list[tuple[int, str, str]],  # (idx, model, value)
    rationale_filler_per_model: int,
    num_rationales: int | None = None,
    research_filler: int = 0,
    stacked: bool | None = None,
) -> str:
    """Build a bot comment that mirrors the real production structure.

    Structure matches main.TemplateForecaster._create_unified_explanation:
      # SUMMARY
      ## Report 1 Summary
      ### Forecasts
        *Forecaster N*: value  (one bullet per ensemble member)
      ### Research Summary
        ...research body...
      ## R1: Forecaster N Reasoning
        Model: openrouter/.../<name>
        ...reasoning body...
      <!-- STACKED=<bool> -->   (optional, only for stacking strategies)
    """
    if num_rationales is None:
        num_rationales = len(summary_bullets)

    lines = [
        "# SUMMARY",
        "*Question*: will X?",
        "",
        "## Report 1 Summary",
        "### Forecasts",
    ]
    for idx, _, value in summary_bullets:
        lines.append(f"*Forecaster {idx}*: {value}")
    lines.append("")
    lines.append("### Research Summary")
    if research_filler:
        lines.append("research_token " * research_filler)
    lines.append("")
    for idx, model, _ in summary_bullets[:num_rationales]:
        lines.append(f"## R1: Forecaster {idx} Reasoning")
        lines.append(f"Model: openrouter/provider/{model}")
        lines.append("")
        lines.append("reasoning_token " * rationale_filler_per_model)
        lines.append("")
    if stacked is True:
        lines.append("<!-- STACKED=true -->")
    elif stacked is False:
        lines.append("<!-- STACKED=false -->")
    return "\n".join(lines)


SUMMARY_BULLETS = [
    (1, "gpt-5.2", "73.0%"),
    (2, "gpt-5.4", "68.0%"),
    (3, "claude-opus-4.6", "71.0%"),
    (4, "claude-opus-4.5", "70.0%"),
    (5, "gemini-3.1-pro-preview", "65.0%"),
    (6, "grok-4.1-fast", "74.0%"),
]


class TestStructuredTrim:
    def test_noop_when_within_limit(self) -> None:
        comment = _build_realistic_comment(
            summary_bullets=SUMMARY_BULLETS,
            rationale_filler_per_model=100,
            stacked=True,
        )
        assert len(comment) < COMMENT_CHAR_LIMIT
        assert trim_comment(comment) == comment

    def test_preserves_summary_and_tail_when_overflowing(self) -> None:
        # Inflate the research section so the comment exceeds the limit.
        comment = _build_realistic_comment(
            summary_bullets=SUMMARY_BULLETS,
            rationale_filler_per_model=800,
            research_filler=25_000,  # ~375k chars of filler, forces overflow
            stacked=True,
        )
        assert len(comment) > COMMENT_CHAR_LIMIT, "test precondition: must overflow"

        trimmed = trim_comment(comment)

        # Every summary bullet must survive.
        for idx, _, value in SUMMARY_BULLETS:
            assert f"*Forecaster {idx}*: {value}" in trimmed, f"Forecaster {idx} summary bullet lost"

        # The trim notice must be present exactly once (signalling the middle gap).
        assert trimmed.count(TRIM_NOTICE) == 1

        # The STACKED marker must survive (it's at the tail).
        assert "<!-- STACKED=true -->" in trimmed

        # Total length must respect the comment limit.
        assert len(trimmed) <= COMMENT_CHAR_LIMIT

    def test_stacked_false_marker_preserved(self) -> None:
        comment = _build_realistic_comment(
            summary_bullets=SUMMARY_BULLETS,
            rationale_filler_per_model=800,
            research_filler=25_000,
            stacked=False,
        )
        trimmed = trim_comment(comment)
        assert "<!-- STACKED=false -->" in trimmed

    def test_missing_stacked_marker_ok(self) -> None:
        # Non-stacking strategies emit no marker; trim must still succeed.
        comment = _build_realistic_comment(
            summary_bullets=SUMMARY_BULLETS,
            rationale_filler_per_model=800,
            research_filler=25_000,
            stacked=None,
        )
        assert len(comment) > COMMENT_CHAR_LIMIT
        trimmed = trim_comment(comment)
        assert TRIM_NOTICE in trimmed
        # All summary bullets still present.
        for idx, _, value in SUMMARY_BULLETS:
            assert f"*Forecaster {idx}*: {value}" in trimmed

    def test_head_over_budget_falls_back_to_header_and_tail(self) -> None:
        # Absurdly-long summary head — e.g. a runaway model dump before
        # Research Summary — with no # RESEARCH/# FORECASTS structure. The
        # summary_and_tail path bails (head exceeds head_budget), so we fall
        # through to the header-preserving last-resort trim. It must keep the
        # leading "# SUMMARY" line (validator invariant) rather than prepend
        # the notice and drop the header.
        cfg = TrimConfig(head_budget=500)  # tight head cap to force fallback
        bloated_summary = "*Forecaster 1*: 50%\n" + ("x" * 10_000)
        comment = (
            f"# SUMMARY\n{bloated_summary}\n### Research Summary\n" + ("body " * 40_000) + "\n<!-- STACKED=true -->\n"
        )
        assert len(comment) > cfg.comment_limit
        trimmed = trim_comment(comment, config=cfg)
        assert trimmed.startswith("# SUMMARY"), "header-preserving fallback must keep the leading '#' line"
        assert TRIM_NOTICE in trimmed
        assert len(trimmed) <= cfg.comment_limit

    def test_middle_gets_dropped_not_head_or_tail(self) -> None:
        # Build a comment where the middle contains a unique marker that
        # must NOT appear in the trimmed output, while head and tail
        # markers MUST appear.
        head_sentinel = "HEAD_SENTINEL_XYZ"
        middle_sentinel = "MIDDLE_SENTINEL_XYZ"
        tail_sentinel = "TAIL_SENTINEL_XYZ"
        comment = (
            f"# SUMMARY\n*Forecaster 1*: 50%\n{head_sentinel}\n"
            f"### Research Summary\n"
            + middle_sentinel
            + "\n"
            + ("padding " * 40_000)
            + "\n"
            + tail_sentinel
            + "\n<!-- STACKED=true -->\n"
        )
        assert len(comment) > COMMENT_CHAR_LIMIT
        trimmed = trim_comment(comment)
        assert head_sentinel in trimmed
        assert tail_sentinel in trimmed
        assert middle_sentinel not in trimmed


# ---------------------------------------------------------------------------
# Section-aware research-first trim (primary strategy)
# ---------------------------------------------------------------------------


def _build_sectioned_comment(
    *,
    summary_body: str,
    research_body: str,
    forecasts_body: str,
    trailing_markers: str = "<!-- STACKED=true -->\n<!-- TOOLS_USED=false -->",
) -> str:
    """Mirror the framework's unified comment: # SUMMARY / # RESEARCH / # FORECASTS.

    Matches forecast_bot.py:538-550 — the comment always opens with # SUMMARY
    and carries # RESEARCH and # FORECASTS as top-level (h1) sections, with the
    residual-analysis markers trailing after the FORECASTS rationales.
    """
    return (
        "# SUMMARY\n"
        "*Question*: will X?\n\n"
        "## Report 1 Summary\n"
        "### Forecasts\n"
        f"{summary_body}\n\n"
        "### Research Summary\n"
        "_Full research in the RESEARCH section below._\n\n"
        "# RESEARCH\n"
        "## Report 1 Research\n"
        f"{research_body}\n\n"
        "# FORECASTS\n"
        f"{forecasts_body}\n"
        f"{trailing_markers}\n"
    )


def _build_sectioned_comment_with_length(length: int) -> str:
    forecasts_body = (
        "## R1: Forecaster 1 Reasoning\n"
        "Model: openrouter/provider/model-1\n\n"
        "A concise rationale.\n\n"
        '```json\n{"probability": 0.65}\n```'
    )
    base = _build_sectioned_comment(
        summary_body="*Forecaster 1*: 65%",
        research_body="",
        forecasts_body=forecasts_body,
    )
    research_characters = length - len(base)
    assert research_characters >= 0
    if research_characters:
        paragraph_count = math.ceil((research_characters + 2) / 502)
        content_characters = research_characters - 2 * (paragraph_count - 1)
        paragraph_size, extra_characters = divmod(content_characters, paragraph_count)
        research_body = "\n\n".join(
            "R" * (paragraph_size + (index < extra_characters)) for index in range(paragraph_count)
        )
    else:
        research_body = ""
    comment = _build_sectioned_comment(
        summary_body="*Forecaster 1*: 65%",
        research_body=research_body,
        forecasts_body=forecasts_body,
    )
    assert len(comment) == length
    return comment


@pytest.mark.parametrize(
    ("source_length", "should_trim"),
    [
        (COMMENT_CHAR_LIMIT - 1, False),
        (COMMENT_CHAR_LIMIT, False),
        (COMMENT_CHAR_LIMIT + 1, True),
        (150_000, True),
        (150_001, True),
    ],
)
def test_complete_comment_budget_boundary_preserves_rationale(source_length: int, should_trim: bool) -> None:
    comment = _build_sectioned_comment_with_length(source_length)

    trimmed = trim_comment(comment)

    assert len(trimmed) <= COMMENT_CHAR_LIMIT
    if not should_trim:
        assert trimmed == comment
    else:
        assert trimmed != comment
        assert TRIM_NOTICE in trimmed
    assert "*Forecaster 1*: 65%" in trimmed
    assert "Model: openrouter/provider/model-1" in trimmed
    assert '{"probability": 0.65}' in trimmed
    assert "<!-- STACKED=true -->" in trimmed


class TestSummarizerDisclosuresSurviveTrimming:
    """Both halves of the summarizer-degradation disclosure must reach the reader.

    Two trims act on the RESEARCH section with OPPOSITE polarity, and the two
    disclosures sit at opposite ends of it, so before the pin each trim killed one:

    * ``trim_section`` (research budget, ``preserve_header=True``) keeps header +
      TAIL, discarding the ``⚠ RAW UNSCREENED ARTICLES`` banner that
      ``_degraded_to_raw_articles`` PREPENDS to the AskNews body.
    * ``trim_comment``'s ``research_first`` strategy keeps the FRONT of the research
      body, discarding the APPENDED ``## Provider Diagnostics`` block — which is
      where the ``summarizer:error(...)`` token lives.

    Exposure is not hypothetical: 160 of 963 archived bundles already exceed the
    research budget, and a summarizer soft-fail is itself what pushes a bundle over
    (raw articles run longer than the briefing), so the banner was most likely
    trimmed precisely on the questions where it fired. The durable machine-readable
    record survives either way (the archive sink writes before any trimming, and the
    per-source token has its own archive field); what was lost is the human-readable
    disclosure on the published comment.
    """

    def test_banner_survives_an_overflowing_research_section(self) -> None:
        # The banner sits INSIDE the body (the orchestrator prepends it to the AskNews
        # provider's text, so an assembled bundle carries it after that provider's own
        # header — not at the body's first byte), and the research trim keeps header +
        # tail, so without the pin this is exactly the disclosure that gets dropped.
        body = "\n".join(f"Article {i}: raw unscreened article text." for i in range(2_000))
        section = "## Report 1 Research\n## News Articles (AskNews)\n" + SUMMARIZER_SOFT_FAIL_BANNER + f"\n\n{body}"
        assert len(section) > RESEARCH_SECTION_CHAR_LIMIT, "precondition: section must overflow"

        trimmed = trim_section(section, "report_1_research")

        assert len(trimmed) <= RESEARCH_SECTION_CHAR_LIMIT
        assert "RAW UNSCREENED ARTICLES" in trimmed, "the degradation banner must survive the research trim"
        assert TRIM_NOTICE in trimmed
        # The pin must not cost the section its header or its leading '#'.
        assert trimmed.lstrip().startswith("#")
        assert "## Report 1 Research" in trimmed

    def test_banner_pinned_even_when_it_leads_the_body(self) -> None:
        # The other placement: banner at the body's first byte (a bundle whose only
        # provider is AskNews, so no provider header precedes it).
        body = "\n".join(f"Article {i}: raw unscreened article text." for i in range(2_000))
        section = f"## Report 1 Research\n{SUMMARIZER_SOFT_FAIL_BANNER}\n\n{body}"
        assert len(section) > RESEARCH_SECTION_CHAR_LIMIT, "precondition: section must overflow"

        trimmed = trim_section(section, "report_1_research")

        assert len(trimmed) <= RESEARCH_SECTION_CHAR_LIMIT
        assert "RAW UNSCREENED ARTICLES" in trimmed
        assert trimmed.lstrip().startswith("#")

    def test_untrimmed_research_section_is_byte_identical(self) -> None:
        # The pin is a trim-time repair, not a rewrite: a section within budget must
        # pass through untouched whether or not it carries a banner.
        section = f"## Report 1 Research\n{SUMMARIZER_SOFT_FAIL_BANNER}\n\nshort body"
        assert trim_section(section, "report_1_research") == section

    def test_diagnostics_block_survives_the_research_first_comment_trim(self) -> None:
        # research_first keeps the FRONT of the research body, so the appended
        # diagnostics block — and the summarizer:error token that names WHICH source
        # was lost — is what gets dropped here.
        diagnostics = (
            "---\n\n## Provider Diagnostics\n\n"
            "- asknews: ok | 41000 chars | 8000 ms | sources=0/1 | lost=summarizer:error(TimeoutError)"
        )
        research = f"{SUMMARIZER_SOFT_FAIL_BANNER}\n\n" + ("research_token " * 30_000) + f"\n\n{diagnostics}"
        comment = _build_sectioned_comment(
            summary_body="*Forecaster 1*: 5.0%",
            research_body=research,
            forecasts_body="## R1: Forecaster 1 Reasoning\nModel: openrouter/openai/gpt-5.6-sol\nbody",
        )
        assert len(comment) > COMMENT_CHAR_LIMIT, "precondition: comment must overflow"

        trimmed = trim_comment(comment)

        assert len(trimmed) <= COMMENT_CHAR_LIMIT
        assert trimmed.lstrip().startswith("#")
        assert "## Provider Diagnostics" in trimmed, "the diagnostics block must survive the comment trim"
        assert "summarizer:error(TimeoutError)" in trimmed, (
            "the per-source token naming the lost source must survive — it is the half of the "
            "disclosure that says WHICH source degraded"
        )
        # The banner already survived this strategy; it must keep doing so.
        assert "RAW UNSCREENED ARTICLES" in trimmed


class TestResearchFirstTrim:
    """The primary strategy: shrink # RESEARCH before SUMMARY/FORECASTS."""

    def test_research_shrunk_first_summary_and_forecasts_survive(self) -> None:
        summary = "\n".join(f"*Forecaster {i}*: {60 + i}.0%" for i in range(1, 7))
        # Sentinel sits at the END of the research body — research is summary-
        # style and front-loaded, so the trim keeps the front and drops the
        # tail, taking this sentinel with it.
        research = ("research_token " * 30_000) + " RESEARCH_SENTINEL"  # forces overflow
        forecasts = "FORECASTS_SENTINEL\n" + "\n".join(
            f"## R1: Forecaster {i} Reasoning\nModel: openrouter/provider/m{i}\nrationale body" for i in range(1, 7)
        )
        comment = _build_sectioned_comment(summary_body=summary, research_body=research, forecasts_body=forecasts)
        assert len(comment) > COMMENT_CHAR_LIMIT, "precondition: must overflow"

        trimmed = trim_comment(comment)

        assert len(trimmed) <= COMMENT_CHAR_LIMIT
        assert trimmed.lstrip().startswith("#")
        # SUMMARY survives whole: every bullet + the Research Summary marker.
        assert "### Research Summary" in trimmed
        for i in range(1, 7):
            assert f"*Forecaster {i}*: {60 + i}.0%" in trimmed
        # FORECASTS survives whole.
        assert "FORECASTS_SENTINEL" in trimmed
        for i in range(1, 7):
            assert f"## R1: Forecaster {i} Reasoning" in trimmed
        # Research middle is sacrificed.
        assert "RESEARCH_SENTINEL" not in trimmed
        assert TRIM_NOTICE in trimmed
        # Trailing residual-analysis markers survive.
        assert "<!-- STACKED=true -->" in trimmed
        assert "<!-- TOOLS_USED=false -->" in trimmed

    def test_summary_and_tail_fallback_when_head_plus_tail_overflow(self) -> None:
        # When head + tail alone overflow (massive FORECASTS body), research-
        # first bails (research_budget < 0) and the summary-and-tail path takes
        # over. The trim must still produce a valid '#'-leading string within
        # the limit, preserving the SUMMARY head and the trailing markers.
        summary = "\n".join(f"*Forecaster {i}*: {60 + i}.0%" for i in range(1, 7))
        research = "research " * 5_000
        forecasts = "FORECASTS_HEAD\n" + ("fcast_token " * 30_000)  # huge tail
        comment = _build_sectioned_comment(summary_body=summary, research_body=research, forecasts_body=forecasts)
        assert len(comment) > COMMENT_CHAR_LIMIT

        trimmed = trim_comment(comment)

        assert len(trimmed) <= COMMENT_CHAR_LIMIT
        assert trimmed.lstrip().startswith("#")
        # SUMMARY is preserved even in this extreme case.
        assert "### Research Summary" in trimmed
        for i in range(1, 7):
            assert f"*Forecaster {i}*: {60 + i}.0%" in trimmed
        # Research body is gone entirely.
        assert "research research" not in trimmed
        assert TRIM_NOTICE in trimmed
        # Trailing residual-analysis markers (at the very end) survive.
        assert "<!-- STACKED=true -->" in trimmed
        assert "<!-- TOOLS_USED=false -->" in trimmed


# ---------------------------------------------------------------------------
# Leading-'#' invariant: for any '#'-leading input (the framework's contract —
# every comment opens with "# SUMMARY"), trim_comment's output also starts with
# '#', so validate_explanation_starts_with_hash can never reject it. This is the
# contract that makes the 2026-06-05 crash structurally impossible.
# ---------------------------------------------------------------------------


def _invariant_shapes() -> list[str]:
    big = COMMENT_CHAR_LIMIT + 50_000
    return [
        # Sectioned, with Research Summary marker, just over the limit.
        _build_sectioned_comment(
            summary_body="*Forecaster 1*: 50%",
            research_body="r " * (big // 2),
            forecasts_body="rationale",
        ),
        # Sectioned, no trailing markers.
        _build_sectioned_comment(
            summary_body="*Forecaster 1*: 50%",
            research_body="r " * (big // 2),
            forecasts_body="rationale",
            trailing_markers="",
        ),
        # Has ### Research Summary but no # RESEARCH/# FORECASTS structure.
        "# SUMMARY\n*Forecaster 1*: 50%\n### Research Summary\n" + ("body " * (big // 5)),
        # The exact crash shape: # SUMMARY blob, no markers, overflowing.
        "# SUMMARY\n" + ("x" * big),
        # Leading "\n# SUMMARY..." with no sections — the framework's
        # clean_indents output starts with a newline, so the last-resort trim
        # must skip it and keep the "#" header line.
        "\n# SUMMARY\n" + ("x" * big),
        # Degenerate single-line '# x' that somehow overflows.
        "# " + ("y" * big),
        # No leading newline structure at all (still starts with #).
        "#" + ("z" * big),
    ]


@pytest.mark.parametrize("shape", _invariant_shapes())
def test_trim_comment_output_always_starts_with_hash(shape: str) -> None:
    out = trim_comment(shape)
    assert out.lstrip().startswith("#"), f"invariant violated for shape starting {shape[:40]!r}"
    assert len(out) <= COMMENT_CHAR_LIMIT


# ---------------------------------------------------------------------------
# End-to-end: trimmed comments remain parseable by residual-analysis regexes
# ---------------------------------------------------------------------------


class TestParserCompatibility:
    """Guard against silent breakage of the parsers that consume bot comments.

    If a trim reduces what these parsers can extract, historical performance
    analyses become inconsistent with live ones, which was the specific
    motivation for keeping the summary head intact.
    """

    def _comment(self, *, stacked: bool | None = True) -> str:
        return _build_realistic_comment(
            summary_bullets=SUMMARY_BULLETS,
            rationale_filler_per_model=600,
            research_filler=25_000,
            stacked=stacked,
        )

    def test_parse_per_model_forecasts_survives_trim(self) -> None:
        from metaculus_bot.performance_analysis.parsing import parse_per_model_forecasts

        comment = self._comment()
        assert len(comment) > COMMENT_CHAR_LIMIT
        trimmed = trim_comment(comment)

        # Original extraction — baseline
        original_parsed = parse_per_model_forecasts(comment)
        trimmed_parsed = parse_per_model_forecasts(trimmed)

        assert original_parsed, "precondition: parser must find something in the original"
        # Trim must not lose any model's summary forecast.
        assert trimmed_parsed == original_parsed, (
            f"Trim broke per-model parsing. orig={original_parsed} trimmed={trimmed_parsed}"
        )

    def test_parse_stacked_marker_survives_trim(self) -> None:
        from metaculus_bot.performance_analysis.parsing import parse_stacked_marker

        for flag in (True, False):
            comment = self._comment(stacked=flag)
            assert len(comment) > COMMENT_CHAR_LIMIT
            trimmed = trim_comment(comment)
            assert parse_stacked_marker(trimmed) is flag, f"STACKED={flag} marker lost during trim"

    def test_parse_forecaster_model_map_at_least_partial(self) -> None:
        # This one's more permissive: the R1 rationale blocks sit in the
        # middle of the comment and may be dropped by trimming. We don't
        # require ALL of them to survive — just that whatever survives
        # uses real model names (not anonymized fallbacks). Losing model
        # attribution silently would be a regression; missing some
        # rationales entirely is acceptable and already handled by
        # anonymized fallback in the parser.
        from metaculus_bot.performance_analysis.parsing import parse_forecaster_model_map

        comment = self._comment()
        trimmed = trim_comment(comment)
        # The tail-preserving trim is likely to retain the LAST few
        # rationales. Whatever it retains must have valid model names.
        trimmed_map = parse_forecaster_model_map(trimmed)
        for idx, name in trimmed_map.items():
            assert name, f"Forecaster {idx} attribution degraded to '{name}'"
            assert not name.startswith("Forecaster"), f"Forecaster {idx} attribution degraded to '{name}'"


# ---------------------------------------------------------------------------
# Regression guards on real historical comments
# ---------------------------------------------------------------------------


class TestAgainstRealHistoricalData:
    """Run the trim on actual comments from scratch/analysis_2026-04/.

    These comments are the exact strings the bot emitted and Metaculus stored;
    trimming them shouldn't produce behavior meaningfully different from what
    production emitted. This catches environment-specific edge cases that
    synthetic comments might miss.

    LOCAL-ONLY, and that is acceptable only because ``TestAgainstCheckedInMiniComments``
    below replays the same inflate-trim-reparse invariants over the checked-in
    miniature on every PR. This class is the breadth cohort (~90k-char real
    comments, dozens of them); the miniature is the floor that cannot skip. Do not
    let this be the only home for an invariant.
    """

    def _load_real_comments(self) -> list[str]:
        """Return only the string comment_text values from the Q2 dataset."""
        import json
        from pathlib import Path

        path = Path(__file__).parent.parent / "scratch" / "analysis_2026-04" / "performance_data.json"
        if not path.exists():
            pytest.skip(
                f"local-only breadth cohort: {path} is a gitignored Q2-2026 collector pull, absent in CI. "
                "The same trim/reparse invariants run unconditionally in "
                "TestAgainstCheckedInMiniComments over tests/data/performance_comments_mini.jsonl."
            )
        with path.open() as f:
            records = json.load(f)
        return [r["comment_text"] for r in records if isinstance(r.get("comment_text"), str)]

    def test_no_real_comment_gets_worse_parsing_after_re_trim(self) -> None:
        """Re-trimming an already-trimmed comment must not make things worse.

        The stored comments were trimmed by production before being saved.
        Passing them through trim_comment again should be a no-op (they're
        all ≤ limit). But even if a future change altered the limit, the
        parser output on the re-trimmed version should not REGRESS vs. the
        stored version.
        """
        from metaculus_bot.performance_analysis.parsing import (
            parse_forecaster_model_map,
            parse_per_model_forecasts,
            parse_stacked_marker,
        )

        comments = self._load_real_comments()
        assert comments, "precondition: some real comments must exist"

        for text in comments:
            if len(text) > COMMENT_CHAR_LIMIT:
                pytest.fail(f"Stored comment longer than limit ({len(text)} > {COMMENT_CHAR_LIMIT}) — data corrupt?")

            trimmed = trim_comment(text)
            assert trimmed == text, "trim_comment must be a no-op on comments already within the limit"

            # Parsers behave identically (trivially, since no-op).
            assert parse_per_model_forecasts(trimmed) == parse_per_model_forecasts(text)
            assert parse_stacked_marker(trimmed) == parse_stacked_marker(text)
            assert parse_forecaster_model_map(trimmed) == parse_forecaster_model_map(text)

    def test_inflated_real_comment_still_parses_after_trim(self) -> None:
        """Take a real comment, inflate it past the limit, and confirm the
        trim preserves everything the parsers need."""
        from metaculus_bot.performance_analysis.parsing import (
            parse_per_model_forecasts,
            parse_stacked_marker,
        )

        comments = self._load_real_comments()

        # The Q2 dataset always contains multiple healthy large comments with
        # the expected structure. If this filter returns nothing, it's a real
        # regression (data format change, trim limit change, or dataset
        # corruption) — not a test-environment issue — so assert.
        original = next(
            (
                c
                for c in comments
                if len(c) > 90_000
                and "[... trimmed for length]" not in c
                and "### Research Summary" in c
                and "*Forecaster 1*:" in c
            ),
            None,
        )
        assert original is not None, (
            "No healthy ≥90k-char comment with expected structure found in Q2 dataset. "
            "If the dataset is intact, this indicates a structural regression in the comment format."
        )

        original_forecasts = parse_per_model_forecasts(original)
        original_stacked = parse_stacked_marker(original)
        assert original_forecasts, "precondition: original must have parseable forecasts"

        # Inflate the research section until the comment overflows
        marker = "### Research Summary"
        idx = original.find(marker)
        assert idx >= 0
        inflated = (
            original[: idx + len(marker)] + "\n" + ("filler_token " * 15_000) + "\n" + original[idx + len(marker) :]
        )
        assert len(inflated) > COMMENT_CHAR_LIMIT, "precondition: inflated must overflow"

        trimmed = trim_comment(inflated)
        assert len(trimmed) <= COMMENT_CHAR_LIMIT

        # Summary forecasts must round-trip unchanged.
        assert parse_per_model_forecasts(trimmed) == original_forecasts
        # STACKED marker behavior must match (tail is preserved either way).
        assert parse_stacked_marker(trimmed) == original_stacked

    def test_every_real_comment_inflated_holds_invariant_and_parses(self) -> None:
        """Replay over ALL stored comments, not just one hand-picked example.

        For every real comment in the Q2 dataset: inflate its research section
        past the limit, trim, and assert the trimmed output (a) stays within the
        limit, (b) starts with '#' (the validator invariant — the exact contract
        whose violation crashed Q578/Q20683), and (c) round-trips the per-model
        forecasts and STACKED marker that residual analysis depends on. This is
        the closest thing to a production replay without spending API credits.
        """
        from metaculus_bot.performance_analysis.parsing import (
            parse_per_model_forecasts,
            parse_stacked_marker,
        )

        comments = self._load_real_comments()
        assert comments, "precondition: some real comments must exist"

        marker = "### Research Summary"
        checked = 0
        for original in comments:
            # Only comments with the structured marker are inflatable in a way
            # that mirrors production; skip the rare ones without it (already
            # covered by the synthetic no-marker invariant test).
            idx = original.find(marker)
            if idx < 0:
                continue

            original_forecasts = parse_per_model_forecasts(original)
            original_stacked = parse_stacked_marker(original)

            inflated = (
                original[: idx + len(marker)] + "\n" + ("filler_token " * 15_000) + "\n" + original[idx + len(marker) :]
            )
            assert len(inflated) > COMMENT_CHAR_LIMIT, "precondition: inflated must overflow"

            trimmed = trim_comment(inflated)

            assert len(trimmed) <= COMMENT_CHAR_LIMIT, f"trim exceeded limit ({len(trimmed)})"
            assert trimmed.lstrip().startswith("#"), "validator invariant violated on a real comment"
            # Per-model forecasts in the SUMMARY head must survive unchanged.
            if original_forecasts:
                assert parse_per_model_forecasts(trimmed) == original_forecasts
            # STACKED marker (tail) must survive unchanged.
            assert parse_stacked_marker(trimmed) == original_stacked
            checked += 1

        assert checked >= 50, f"expected to exercise dozens of real comments, only hit {checked}"


class TestAgainstCheckedInMiniComments:
    """The CI floor for the real-comment trim replay above.

    ``TestAgainstRealHistoricalData`` reads ``scratch/analysis_2026-04/`` — a
    gitignored local pull — so it skips entirely in CI and the trim/parser
    round-trip it guards goes unenforced on every PR. This class replays the same
    invariants over the checked-in miniature
    (``tests/data/performance_comments_mini.jsonl``, one record per distinct
    comment shape, built by ``scripts/derive_mini_comment_fixture.py``).

    The miniature comments are small by construction, so the no-op-trim assertion
    is trivially true for them and is not repeated here. What DOES transfer is the
    inflation replay: inflating a real comment's research section past the limit
    and trimming it exercises the same section-budget and block-trim code paths
    regardless of the seed comment's original size.
    """

    def _load_mini_comments(self) -> list[str]:
        from pathlib import (
            Path,  # HARNESS-SCAN-EXEMPT-function-level-import  # matches this file's local style
        )

        path = Path(__file__).parent / "data" / "performance_comments_mini.jsonl"
        with path.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        return [rec["comment_text"] for rec in records if isinstance(rec.get("comment_text"), str)]

    def test_inflated_mini_comments_hold_invariant_and_parse(self) -> None:
        from metaculus_bot.performance_analysis.parsing import (
            parse_per_model_forecasts,
            parse_stacked_marker,
        )

        comments = self._load_mini_comments()
        assert comments, "precondition: checked-in miniature must contain comments"

        marker = "### Research Summary"
        checked = 0
        for original in comments:
            idx = original.find(marker)
            if idx < 0:
                continue
            original_forecasts = parse_per_model_forecasts(original)
            if not original_forecasts:
                continue
            original_stacked = parse_stacked_marker(original)

            inflated = (
                original[: idx + len(marker)] + "\n" + ("filler_token " * 15_000) + "\n" + original[idx + len(marker) :]
            )
            assert len(inflated) > COMMENT_CHAR_LIMIT, "precondition: inflated must overflow"

            trimmed = trim_comment(inflated)

            assert len(trimmed) <= COMMENT_CHAR_LIMIT, f"trim exceeded limit ({len(trimmed)})"
            assert trimmed.lstrip().startswith("#"), "validator invariant violated on a real comment"
            assert parse_per_model_forecasts(trimmed) == original_forecasts
            assert parse_stacked_marker(trimmed) == original_stacked
            checked += 1

        # Every shape carrying the marker plus parseable bullets; a drop here means
        # the miniature narrowed or the trim stopped round-tripping.
        assert checked >= 10, f"expected the miniature's marker-bearing comments, only hit {checked}"


# ---------------------------------------------------------------------------
# Per-section budgets + block-aware FORECASTS trim (Model:-line preservation)
#
# The FORECASTS (rationales) section gets its own large budget and is trimmed
# block-by-block, so a per-forecaster overflow no longer destroys the first
# forecaster's ``Model:`` attribution line (measured 29/29 in July 2026) or its
# JSON forecast block — both parsed by the residual pipeline.
# ---------------------------------------------------------------------------


_RATIONALE_MODELS: list[tuple[int, str]] = [
    (1, "gpt-5.6-sol"),
    (2, "gpt-5.5"),
    (3, "claude-opus-4.7"),
    (4, "claude-opus-4.8"),
    (5, "gemini-3.1-pro-preview"),
    (6, "grok-4.5"),
]


def _numeric_json_block(base: float) -> str:
    """A schema-valid numeric STRUCTURED FORECAST block (the 3 required percentiles)."""
    payload = {
        "question_type": "numeric",
        "declared_percentiles": {"0.1": base, "0.5": base + 10.0, "0.9": base + 20.0},
    }
    return "```json\n" + json.dumps(payload) + "\n```"


def _rationale_block(idx: int, model: str, *, prose_tokens: int, base: float) -> str:
    """One ``## R1: Forecaster N Reasoning`` block: header + Model line + prose + JSON block."""
    return (
        f"## R1: Forecaster {idx} Reasoning\n"
        f"Model: openrouter/provider/{model}\n"
        f"{'reasoning_token ' * prose_tokens}\n"
        f"{_numeric_json_block(base)}"
    )


def _rationales_section(prose_tokens: int) -> str:
    return "\n\n".join(_rationale_block(i, m, prose_tokens=prose_tokens, base=100.0 + i) for i, m in _RATIONALE_MODELS)


class TestSectionBudgetRouting:
    """trim_section picks the budget (and block-awareness) from the section name."""

    def test_rationales_route_to_forecasts_budget(self) -> None:
        rationales = _rationales_section(prose_tokens=1200)
        assert len(rationales) > FORECASTS_SECTION_CHAR_LIMIT, "precondition: section must overflow"
        trimmed = trim_section(rationales, "report_1_rationales")
        assert len(trimmed) <= FORECASTS_SECTION_CHAR_LIMIT

    def test_research_routes_to_research_budget(self) -> None:
        research = "## Report 1 Research\n" + ("research_token " * 6_000)
        assert len(research) > RESEARCH_SECTION_CHAR_LIMIT
        trimmed = trim_section(research, "report_1_research")
        assert len(trimmed) <= RESEARCH_SECTION_CHAR_LIMIT
        assert TRIM_NOTICE in trimmed

    def test_summary_routes_to_summary_budget(self) -> None:
        summary = "## Report 1 Summary\n" + ("X" * (SUMMARY_SECTION_CHAR_LIMIT + 2_000))
        trimmed = trim_section(summary, "report_1_summary")
        assert len(trimmed) <= SUMMARY_SECTION_CHAR_LIMIT


class TestRationaleBlockAwareTrim:
    """An over-budget FORECASTS section keeps every block's Model: line + JSON block."""

    def test_all_six_model_lines_and_blocks_survive(self) -> None:
        from metaculus_bot.performance_analysis.parsing import (
            parse_forecaster_model_map,
            parse_per_model_numeric_percentiles,
            parse_per_model_reasoning_text,
        )

        rationales = _rationales_section(prose_tokens=1500)  # ~145k, well over budget
        assert len(rationales) > FORECASTS_SECTION_CHAR_LIMIT

        trimmed = trim_section(rationales, "report_1_rationales")
        assert len(trimmed) <= FORECASTS_SECTION_CHAR_LIMIT
        assert trimmed.startswith("## R1: Forecaster 1 Reasoning")
        # The middle prose is sacrificed, so the trim notice fires.
        assert TRIM_NOTICE in trimmed

        # Wrap in the # FORECASTS header the R1 parsers expect.
        comment = f"# FORECASTS\n{trimmed}\n"

        model_map = parse_forecaster_model_map(comment)
        assert len(model_map) == 6, f"expected all 6 Model: lines, got {model_map}"
        for i, model in _RATIONALE_MODELS:
            assert model_map[i] == model

        percentiles = parse_per_model_numeric_percentiles(comment)
        assert len(percentiles) == 6, f"expected all 6 JSON blocks recovered, got {sorted(percentiles)}"

        bodies = parse_per_model_reasoning_text(comment)
        assert len(bodies) == 6


# ---------------------------------------------------------------------------
# Stacker-combined FORECASTS trim (per-base-model attribution)
#
# When stacking fires, combine_stacker_and_base_reasoning folds the stacker's
# meta-analysis and every base model's reasoning into ONE ``## R1: Forecaster 1
# Reasoning`` block. The pre-fix single-body trim kept only the LAST json block
# in the whole combined body and orphaned it from its ``Model:`` line, so
# parsing._split_stacker_combined_body re-attributed that trailing model's
# forecast values to the last surviving base model — silent misattribution in
# residual analysis. Stacking is prod-disabled today, but the trim runs in
# backtests/ablation and is armed for any stacking re-enable, so it must keep
# each base model paired with its own line and forecast block.
# ---------------------------------------------------------------------------


class TestStackerCombinedBlockTrim:
    """An over-budget stacker-combined R1 body keeps each base model's own
    ``Model:`` line and json forecast block — no cross-model misattribution."""

    _STACKER_META_BASE = 300.0
    # Well-separated bases so a misattributed block is unmistakable in the
    # recovered percentiles (a neighbor's values would be off by hundreds).
    _BASE_MODELS: ClassVar[list[tuple[str, float]]] = [
        ("openrouter/openai/gpt-5.6-sol", 100.0),
        ("openrouter/anthropic/claude-opus-4.8", 500.0),
        ("openrouter/google/gemini-3.1-pro-preview", 900.0),
    ]

    def _combined_body(self, *, prose_tokens: int) -> str:
        """A single ``## R1: Forecaster 1 Reasoning`` block for a stacked question.

        Built exactly as production does: base predictions carry the bot-injected
        ``Model:`` prefix, each with a DISTINCT numeric forecast block, folded via
        combine_stacker_and_base_reasoning under the stacker meta-analysis.
        """
        prose = "reasoning_token " * prose_tokens
        base_preds = [
            ReasonedPrediction(
                prediction_value=base,
                reasoning=f"Model: {path}\n\n{prose}\n{_numeric_json_block(base)}",
            )
            for path, base in self._BASE_MODELS
        ]
        meta_text = f"stacker synthesis prose.\n{prose}\n{_numeric_json_block(self._STACKER_META_BASE)}"
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        return f"## R1: Forecaster 1 Reasoning\n{combined}"

    @staticmethod
    def _expected(base: float) -> list[tuple[float, float]]:
        # _numeric_json_block emits percentiles {0.1, 0.5, 0.9} → percent labels.
        return [(10.0, base), (50.0, base + 10.0), (90.0, base + 20.0)]

    def test_each_base_model_keeps_own_line_and_block_after_trim(self) -> None:
        from metaculus_bot.performance_analysis.parsing import (
            parse_per_model_numeric_percentiles,
            parse_per_model_reasoning_text,
        )

        # ~130k across 4 sub-blocks — large enough that the pre-fix single-body
        # trim would cut off a later base model's Model: line and orphan its
        # json block onto a neighbor.
        text = self._combined_body(prose_tokens=2000)
        assert len(text) > FORECASTS_SECTION_CHAR_LIMIT, "precondition: must overflow"

        trimmed = trim_section(text, "report_1_rationales")
        assert len(trimmed) <= FORECASTS_SECTION_CHAR_LIMIT
        # Middle prose sacrificed → the notice fires.
        assert TRIM_NOTICE in trimmed
        # The R1 header and the stacker delimiter survive verbatim; the parser's
        # stacker-body detection keys on the delimiter.
        assert trimmed.startswith("## R1: Forecaster 1 Reasoning")
        assert STACKED_BASE_REASONING_HEADER in trimmed
        # Every base model keeps its Model: attribution line.
        for path, _ in self._BASE_MODELS:
            assert f"Model: {path}" in trimmed, f"lost Model: line for {path}"

        # Value-equality per model is the assertion that catches misattribution:
        # each base model must recover ITS OWN distinct percentiles.
        comment = f"# FORECASTS\n{trimmed}\n"
        percentiles = parse_per_model_numeric_percentiles(comment)
        assert percentiles["gpt-5.6-sol"] == self._expected(100.0)
        assert percentiles["claude-opus-4.8"] == self._expected(500.0)
        assert percentiles["gemini-3.1-pro-preview"] == self._expected(900.0)
        # The stacker portion has no Model: line, so its own block is recovered
        # under the "Forecaster 1" fallback key.
        assert percentiles["Forecaster 1"] == self._expected(self._STACKER_META_BASE)

        # Per-base-model prose stays attributable for all three.
        bodies = parse_per_model_reasoning_text(comment)
        for name in ("gpt-5.6-sol", "claude-opus-4.8", "gemini-3.1-pro-preview"):
            assert name in bodies, f"lost reasoning body for {name}"

    def test_under_budget_stacked_body_untouched(self) -> None:
        text = self._combined_body(prose_tokens=50)
        assert len(text) < FORECASTS_SECTION_CHAR_LIMIT, "precondition: must fit under budget"
        assert trim_section(text, "report_1_rationales") == text


class TestFullCommentPreservesAttribution:
    """End-to-end: a >150k raw comment trims to under the ceiling with attribution intact."""

    def test_within_budgets_and_fully_parseable(self) -> None:
        from metaculus_bot.performance_analysis.parsing import (
            parse_forecaster_model_map,
            parse_per_model_forecasts,
            parse_per_model_numeric_percentiles,
        )

        bullets = "\n".join(f"*Forecaster {i} ({m})*: Median: {100 + i}" for i, m in _RATIONALE_MODELS)
        raw_summary = f"## Report 1 Summary\n### Forecasts\n{bullets}\n\n### Research Summary\n_stub_"
        raw_research = "## Report 1 Research\n" + ("research_token " * 6_000)
        raw_rationales = _rationales_section(prose_tokens=1500)

        raw_total = len(raw_summary) + len(raw_research) + len(raw_rationales)
        assert raw_total > COMMENT_CHAR_LIMIT, "precondition: untrimmed comment must exceed the ceiling"

        summary = trim_section(raw_summary, "report_1_summary")
        research = trim_section(raw_research, "report_1_research")
        rationales = trim_section(raw_rationales, "report_1_rationales")

        assert len(summary) <= SUMMARY_SECTION_CHAR_LIMIT
        assert len(research) <= RESEARCH_SECTION_CHAR_LIMIT
        assert len(rationales) <= FORECASTS_SECTION_CHAR_LIMIT

        # Assemble like forecast_bot._create_unified_explanation, then apply the
        # whole-comment trim as build_unified_explanation does.
        assembled = (
            "# SUMMARY\n*Question*: will X happen?\n\n"
            f"{summary}\n\n"
            f"# RESEARCH\n{research}\n\n"
            f"# FORECASTS\n{rationales}\n"
            "<!-- STACKED=false -->\n"
        )
        final = trim_comment(assembled)
        assert len(final) <= COMMENT_CHAR_LIMIT

        forecasts = parse_per_model_forecasts(final)
        assert len(forecasts) == 6, f"lost summary bullets: {forecasts}"

        model_map = parse_forecaster_model_map(final)
        assert len(model_map) == 6, f"lost Model: lines: {model_map}"
        for i, model in _RATIONALE_MODELS:
            assert model_map[i] == model

        percentiles = parse_per_model_numeric_percentiles(final)
        assert len(percentiles) == 6, f"lost per-model JSON blocks: {sorted(percentiles)}"


# ---------------------------------------------------------------------------
# Water-fill allocator (_allocate_block_budgets)
#
# The block-aware FORECASTS trim splits its budget across per-forecaster blocks
# via this allocator. The tests above build near-equal blocks, so ``fits`` is
# always empty and the allocator degenerates to a flat ``total // n`` split —
# the redistribution branch (keep small blocks whole, hand the freed budget to
# the large ones) is never exercised there. These tests hit it directly. The
# load-bearing invariant: under a shortfall the budgets sum to EXACTLY ``total``
# — a reclaim/remainder accounting bug that overshot would push the assembled
# comment past COMMENT_CHAR_LIMIT and Metaculus would reject the submission.
# ---------------------------------------------------------------------------


class TestAllocateBlockBudgets:
    def test_tiny_block_kept_whole_freed_budget_redistributed(self) -> None:
        # One tiny block beside several large ones, with a budget well below
        # their combined size — the common prod case (a soft-deadline-truncated
        # rationale next to full-length ones).
        sizes = [100, 40_000, 40_000, 40_000]
        total = 60_000
        assert total < sum(sizes), "precondition: shortfall — exercises the redistribution branch"

        budgets = _allocate_block_budgets(sizes, total)

        # (a) whole budget consumed exactly — nothing lost or invented.
        assert sum(budgets) == total
        # (b) the tiny block is kept whole (redistribution didn't trim it).
        assert budgets[0] == 100
        # (c) each large block gets strictly more than the flat even split,
        # because the tiny block's unused share was handed back to them.
        flat_share = total // len(sizes)
        for i in (1, 2, 3):
            assert budgets[i] > flat_share
        # No block is ever budgeted above its own size.
        for size, budget in zip(sizes, budgets, strict=True):
            assert budget <= size

    def test_multi_round_water_fill_keeps_unequal_small_blocks_whole(self) -> None:
        # Genuinely multi-round: block 1 (25k) does NOT fit the round-1 share of
        # 20k (= 60k // 3), but once block 0's budget is reclaimed the round-2
        # share rises to ~29.5k (= 59k // 2) and block 1 then fits. Both small
        # blocks must survive whole; only the largest is trimmed.
        sizes = [1_000, 25_000, 40_000]
        total = 60_000
        assert total < sum(sizes)

        budgets = _allocate_block_budgets(sizes, total)

        assert sum(budgets) == total
        assert budgets[0] == 1_000  # round-1 fit, kept whole
        assert budgets[1] == 25_000  # round-2 fit, kept whole
        assert budgets[2] < 40_000  # the only genuinely overflowing block is trimmed
        assert budgets[2] > total // len(sizes)  # ...and gets more than the flat share

    def test_budget_at_least_sum_keeps_every_block_whole(self) -> None:
        # Surplus budget: everything fits, so each block keeps its full size and
        # the total is deliberately NOT fully allocated.
        sizes = [100, 200, 300]
        budgets = _allocate_block_budgets(sizes, total=10_000)
        assert budgets == sizes

    def test_single_block_trimmed_to_total(self) -> None:
        assert _allocate_block_budgets([5_000], total=3_000) == [3_000]

    def test_single_block_kept_whole_when_it_fits(self) -> None:
        assert _allocate_block_budgets([5_000], total=8_000) == [5_000]
