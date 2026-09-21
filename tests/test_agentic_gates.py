"""Gate tests for the agentic gap-fill v2 driver loop.

The four gates that constrain what the low-effort driver may do and what may
reach a forecaster, split out of ``tests/test_agentic_loop.py`` (which keeps the
loop mechanics — budgets, deadlines, tool dispatch, dedup, telemetry):

* ``TestProvenanceGate`` — W3: hard URL check + warn-only quote check on findings.
* ``TestVerificationTierStamping`` — W4: code-derived verification tiers, end to end.
* ``TestResearchPlanGate`` — W1: no external tool call before ``set_research_plan``.
* ``TestConcludeGate`` — W2: no early conclude until gap accounting clears.

Shared scaffolding (``FakeLlm`` and the loop-driving helpers) lives in
``tests/agentic_fakes.py`` so this file and ``test_agentic_loop.py`` drive the
loop identically.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from metaculus_bot.research.agentic.loop import (
    _SPAN_JOINER_MAX_CHARS,
    _normalize_quote_text,
    _quote_is_grounded,
    run_agentic_loop,
)
from metaculus_bot.research.agentic.types import ToolOutcome
from tests.agentic_fakes import FakeLlm
from tests.agentic_fakes import gap_accounting as _accounting
from tests.agentic_fakes import loop_config as _config
from tests.agentic_fakes import plan_call as _plan_call
from tests.agentic_fakes import response as _response
from tests.agentic_fakes import tool_call as _tool_call
from tests.agentic_fakes import tool_messages as _tool_messages
from tests.agentic_fakes import tool_spec as _tool_spec


def _finding(source_url: str, *, quote: str = "Verbatim quote from the source.", discrepancy: bool = False) -> dict:
    return {
        "claim": "The board published the minutes on July 1.",
        "source_url": source_url,
        "quote": quote,
        "date": "2026-07-01",
        "retrieved_how": "fetch",
        "topic": "minutes",
        "discrepancy": discrepancy,
    }


def _search_returning(content: str, links: list[str] | None = None):
    async def _search(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown=content, links=links or [], method="search")

    return _search


def _returns_method(content: str, *, method: str, status: str = "ok", links: list[str] | None = None):
    """A tool handler returning a ToolOutcome with a specific method/status —
    used by the W4 tier-stamping tests to drive fetched vs snippet vs failed
    retrievals through the loop."""

    async def _handler(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown=content, links=links or [], method=method, status=status)

    return _handler


class TestProvenanceGate:
    """Tier-1 hard URL gate + Tier-2 warn-only quote check on agentic findings.

    A finding is rendered under a "supersedes-the-briefing" banner shown to
    every base forecaster, so a hallucinated/mistyped citation from the
    low-effort driver would silently override correct research. The URL check is
    a hard gate; the quote check only warns."""

    @pytest.mark.asyncio
    async def test_local_navigation_inventory_cannot_ground_a_finding(self) -> None:
        inventory_url = "https://agency.example/workbook.xlsx#sheet=Totals"
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("inventory", "fetch", {"url": inventory_url})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "record",
                            "record_findings",
                            {"findings": [_finding(inventory_url, quote="Totals sheet inventory entry")]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done", "conclude")]),
            ]
        )
        inventory = _returns_method(
            f"Totals sheet inventory entry: {inventory_url}", method="local_navigation", links=[inventory_url]
        )

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("fetch", inventory)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 0
        assert result.telemetry.provenance_rejections == 1
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_local_navigation_does_not_erase_provenance_earned_by_search(self) -> None:
        source_url = "https://agency.example/report"
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("search", "search_web", {"query": "report"})]),
                _response(tool_calls=[_tool_call("inventory", "inventory", {"url": source_url})]),
                _response(
                    tool_calls=[
                        _tool_call("record", "record_findings", {"findings": [_finding(source_url, quote="Minutes")]})
                    ]
                ),
                _response(tool_calls=[_tool_call("done", "conclude")]),
            ]
        )
        search = _search_returning(f"{source_url} Minutes")
        inventory = _returns_method(f"Navigation only: {source_url}", method="local_navigation", links=[source_url])

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_web", search), _tool_spec("inventory", inventory)],
            _config(max_steps=6, max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.provenance_rejections == 0
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_hallucinated_url_is_rejected_and_counted(self) -> None:
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://hallucinated.example/nowhere")]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop("system", "briefing with no URLs", [], _config(), llm_call=fake_llm)

        assert "The board published the minutes" not in result.findings_markdown
        assert result.telemetry.findings_count == 0
        assert result.telemetry.provenance_rejections == 1
        # The rejection reason is fed back through the existing "Rejected:" channel.
        rejection = _tool_messages(result)[0]["content"]
        assert "findings[0] rejected" in rejection
        assert "did not appear in any tool result or the briefing" in rejection

    @pytest.mark.asyncio
    async def test_url_from_tool_result_is_accepted(self) -> None:
        """A URL the driver retrieved via a tool (here, embedded in search
        result content) grounds a non-discrepancy finding."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "minutes"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://agency.example/minutes")]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning("Found it at https://agency.example/minutes — details follow.")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert "The board published the minutes" in result.findings_markdown
        assert result.telemetry.findings_count == 1
        assert result.telemetry.provenance_rejections == 0

    @pytest.mark.asyncio
    async def test_briefing_url_accepted_for_non_discrepancy(self) -> None:
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _tool_call("rec1", "record_findings", {"findings": [_finding("https://gov.example/report")]})
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system", "briefing cites https://gov.example/report as a source", [], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.provenance_rejections == 0

    @pytest.mark.asyncio
    async def test_briefing_only_url_rejected_for_discrepancy(self) -> None:
        """A discrepancy must rest on a fresh primary-source check, so a URL that
        appears ONLY in the briefing (never in a tool result) does not ground it."""
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://gov.example/report", discrepancy=True)]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system", "briefing cites https://gov.example/report as a source", [], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 0
        assert result.telemetry.provenance_rejections == 1
        rejection = _tool_messages(result)[0]["content"]
        assert "discrepancy source_url" in rejection
        assert "fresh primary-source check" in rejection

    @pytest.mark.asyncio
    async def test_tool_sourced_url_accepted_for_discrepancy(self) -> None:
        """The same discrepancy is accepted when the URL DID come from a tool."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://gov.example/report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://gov.example/report", discrepancy=True)]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        # The fetch's URL ARGUMENT is what the driver saw — enough to ground it,
        # even though the result body carries no URL.
        fetch = _search_returning("The report body, no links here.")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("fetch", fetch)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.provenance_rejections == 0

    @pytest.mark.asyncio
    async def test_url_variant_normalizes_to_seen_url_and_is_accepted(self) -> None:
        """Trailing slash, utm params, and a fragment on the cited URL all
        normalize to the tool-seen URL, so the finding is grounded."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://Agency.example/report/?utm_source=news#section-2")]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning("See https://agency.example/report for the figures.")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.provenance_rejections == 0

    @pytest.mark.asyncio
    async def test_quote_mismatch_warns_but_accepts(self, caplog: pytest.LogCaptureFixture) -> None:
        """A quote not found in the tool contents is warn-only: the finding is
        still accepted, quote_mismatch_warnings increments, and a WARN is logged."""
        caplog.set_level(logging.WARNING, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        quote="A sentence that never appears in the tool result body.",
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning("See https://agency.example/report — the body says something else entirely.")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1  # accepted despite the quote miss
        assert result.telemetry.provenance_rejections == 0
        assert result.telemetry.quote_mismatch_warnings == 1
        assert any(
            "GAP_FILL_V2 quote_mismatch" in r.getMessage()
            for r in caplog.records
            if r.name == "metaculus_bot.research.agentic.loop"
        )

    async def _run_quote_check(self, *, source_body: str, quote: str) -> Any:
        """Drive the standard plan -> search -> record -> conclude loop with one
        finding whose ``quote`` is spot-checked against ``source_body`` (which
        carries the citation URL so the hard provenance gate accepts the finding)."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://agency.example/report", quote=quote)]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning(f"https://agency.example/report — {source_body}")
        return await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

    @pytest.mark.asyncio
    async def test_stitched_quote_with_bounded_joiners_is_grounded(self) -> None:
        """The four stitch joiners the 2026-08-24 residual round measured as 65% of
        all warnings ever emitted: ``and``, semicolon, comma, and a short narration
        fragment between two verbatim spans. Each span appears verbatim in the
        source but separated there by unrelated text, so before the bounded-connective
        boundary none of these could ever pass (0 of 156 on a corpus containing
        every span verbatim)."""
        # Real shape from run_31868372733: two FEC table rows the driver joined
        # with " and ", each verbatim in the source but non-adjacent there.
        row_a = '"MARC L. ANDREESSEN | CONTRIBUTION | 02/11/2026 | 12500000.00"'
        row_b = '"BENJAMIN HOROWITZ | CONTRIBUTION | 02/11/2026 | 12500000.00"'
        source_body = (
            "Filing index. MARC L. ANDREESSEN | CONTRIBUTION | 02/11/2026 | 12500000.00 "
            "(receipt 881). Additional rows follow the schedule appendix. "
            "BENJAMIN HOROWITZ | CONTRIBUTION | 02/11/2026 | 12500000.00 (receipt 882)."
        )
        for quote in (
            f"{row_a} and {row_b}",
            f"{row_a}; {row_b}",
            f"{row_a}, {row_b}",
            f"{row_a} and later {row_b}",
        ):
            result = await self._run_quote_check(source_body=source_body, quote=quote)
            assert result.telemetry.findings_count == 1
            assert result.telemetry.quote_mismatch_warnings == 0, quote

    @pytest.mark.asyncio
    async def test_stitched_quote_with_fabricated_span_still_warns(self) -> None:
        """The broadening must not rubber-stamp: a stitched quote whose second span
        never appears in the source keeps warning — every weight-bearing span is
        still checked independently."""
        result = await self._run_quote_check(
            source_body="MARC L. ANDREESSEN | CONTRIBUTION | 02/11/2026 | 12500000.00 (receipt 881).",
            quote='"MARC L. ANDREESSEN | CONTRIBUTION | 02/11/2026 | 12500000.00" and '
            '"A FABRICATED DONOR | CONTRIBUTION | 02/11/2026 | 99900000.00"',
        )
        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 1

    @pytest.mark.asyncio
    async def test_long_narration_joiner_still_reads_as_one_span_and_warns(self) -> None:
        """The connective is BOUNDED (24 chars): a full narration sentence between
        spans — the briefing-vs-fetched discrepancy shape, whose form guarantees a
        mismatch by construction — still reads as one span and stays a warning."""
        result = await self._run_quote_check(
            source_body="Musk has personally donated more than $85 million toward the midterms.",
            quote='The briefing says: "Elon Musk — $90.6 million." The fetched report instead says: '
            '"Musk has personally donated more than $85 million toward the midterms."',
        )
        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 1

    @pytest.mark.asyncio
    async def test_fabricated_figure_in_the_joiner_still_warns(self) -> None:
        """The connective between two verbatim spans is CHECKED when it carries a
        digit. `re.split` on a group-less boundary discarded the joiner entirely, so
        a fabricated figure riding between two genuine spans grounded cleanly on the
        spans alone — the exact hole the digit clause exists to close."""
        result = await self._run_quote_check(
            source_body="Total revenue was 4.2 billion dollars for the quarter. Unrelated filler follows here. "
            "The prior fiscal year closed with lower totals across segments.",
            quote='"Total revenue was 4.2 billion dollars" up 47.3% from "the prior fiscal year closed"',
        )
        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 1

    @pytest.mark.asyncio
    async def test_genuine_figure_in_the_joiner_grounds(self) -> None:
        """The digit-gated connective check is a grounding requirement, not a ban:
        when the joiner's figure really is in the source, the stitched quote passes."""
        result = await self._run_quote_check(
            source_body="Total revenue was 4.2 billion dollars for the quarter, up 47.3% from a year earlier. "
            "The prior fiscal year closed with lower totals across segments.",
            quote='"Total revenue was 4.2 billion dollars" up 47.3% from "the prior fiscal year closed"',
        )
        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_span_ending_in_whitespace_before_its_glyph_still_splits(self) -> None:
        """The whitespace-only boundary survives as its own alternative: the bounded
        connective's `(?<=\\S)` lookbehind fails when the first span ends in
        whitespace before its closing glyph, so without the old clause this shape
        reads as one contiguous span and warns — the 2026-07-28 false-positive
        class, reintroduced."""
        result = await self._run_quote_check(
            source_body="Contribution schedules list MARC L. ANDREESSEN in the filing index. Later pages "
            "name BENJAMIN HOROWITZ under the same schedule.",
            quote='"MARC L. ANDREESSEN in the filing " "name BENJAMIN HOROWITZ under"',
        )
        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_the_joiner_bound_splits_at_the_constant_and_not_one_past_it(self) -> None:
        """Pins _SPAN_JOINER_MAX_CHARS at N and N+1 so the empirically-tuned bound
        cannot drift silently: a connective of exactly the bound still splits (both
        verbatim spans ground, digit-free narration unchecked), one char past it
        reads as a single span and warns."""

        source_body = (
            "Contribution schedules list MARC L. ANDREESSEN in the filing index. Later pages "
            "name BENJAMIN HOROWITZ under the same schedule."
        )
        span_a = '"MARC L. ANDREESSEN in the filing"'
        span_b = '"name BENJAMIN HOROWITZ under"'

        at_bound = "x" * _SPAN_JOINER_MAX_CHARS
        result = await self._run_quote_check(source_body=source_body, quote=f"{span_a}{at_bound}{span_b}")
        assert result.telemetry.quote_mismatch_warnings == 0

        past_bound = "x" * (_SPAN_JOINER_MAX_CHARS + 1)
        result = await self._run_quote_check(source_body=source_body, quote=f"{span_a}{past_bound}{span_b}")
        assert result.telemetry.quote_mismatch_warnings == 1

    @pytest.mark.asyncio
    async def test_quote_mismatch_warning_dedupes_on_resubmission(self) -> None:
        """The driver re-lists banked findings (record_findings, then the same
        finding again in a later record_findings): an unmatched quote warns ONCE per
        (source_url, quote) per run — 5.5% of archived warnings were exact
        re-submissions inflating the per-question density."""
        finding = _finding(
            "https://agency.example/report",
            quote="A sentence that never appears in the tool result body.",
        )
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(tool_calls=[_tool_call("rec1", "record_findings", {"findings": [finding]})]),
                _response(tool_calls=[_tool_call("rec2", "record_findings", {"findings": [finding]})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning("See https://agency.example/report — the body says something else entirely.")

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_web", search)],
            _config(max_steps=6),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1  # second submission deduped at banking
        assert result.telemetry.quote_mismatch_warnings == 1

    @pytest.mark.asyncio
    async def test_two_distinct_unmatched_quotes_on_one_url_both_warn(self) -> None:
        """The dedup key is (source_url, quote), not source_url: two DIFFERENT
        ungrounded quotes citing the same page are two separate problems and must
        both be counted. Keying on the URL alone would hide every quote after the
        first on a page the driver cites repeatedly — the common case, since one
        fetched document backs several findings."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        quote="A first sentence that is absent from the tool result body.",
                                    ),
                                    _finding(
                                        "https://agency.example/report",
                                        quote="A second sentence that is also absent from the body.",
                                    ),
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning("See https://agency.example/report — the body says something else entirely.")

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_web", search)],
            _config(),
            llm_call=fake_llm,
        )

        assert result.telemetry.quote_mismatch_warnings == 2

    @pytest.mark.asyncio
    async def test_all_short_nonnumeric_spans_fall_back_to_the_whole_quote(self) -> None:
        """When every split piece is a sub-floor digit-free fragment, no piece
        carries weight — and the fallback tests the WHOLE normalized quote instead
        of passing for free. Without it the broadened boundary would auto-pass any
        quote chopped into short fragments, which is the cheapest way to defeat the
        check entirely."""
        absent = await self._run_quote_check(
            source_body="The filing lists one fact in the appendix; nothing else was disclosed.",
            quote='"one fact" and "two"',
        )
        assert absent.telemetry.findings_count == 1
        assert absent.telemetry.quote_mismatch_warnings == 1

        present = await self._run_quote_check(
            source_body="The filing lists one fact and two others in the appendix.",
            quote='"one fact" and "two"',
        )
        assert present.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_quote_found_in_tool_content_emits_no_warning(self) -> None:
        """A quote present (modulo whitespace/quote-glyph normalization) in the
        tool content does not warn."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        quote="The rate was 4.1 percent.",
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        # Curly quotes + extra whitespace in the source still match the finding.
        search = _search_returning(
            "https://agency.example/report says:  “The rate   was 4.1 percent.”  Full text follows."
        )

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_glyph_wrapped_quote_is_grounded(self) -> None:
        """A finding whose quote is WRAPPED in quote glyphs (the shape the driver
        actually emits) must still ground against unwrapped source text.

        Regression: `_normalize_quote_text` used to SUBSTITUTE every glyph with a
        straight apostrophe rather than stripping it, so a wrapped quote
        normalized to `'the rate was 4.1 percent.'` — with leading/trailing
        apostrophes absent from the source — and the substring test failed on
        text that was genuinely verbatim. In the 2026-07-25 prod run this fired
        on 22 of 22 findings, making the anti-fabrication check dead.
        """
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        quote="“The rate was 4.1 percent.”",
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning("https://agency.example/report says: The rate was 4.1 percent. Full text follows.")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_glyph_separated_sentences_ground_per_span(self) -> None:
        """Two separately-quoted sentences joined by a SPACE must ground per-span
        when both are verbatim in the source but separated there by other text.

        Regression: the split recognized only an ellipsis, so a glyph-joined pair
        was tested as one contiguous substring and could never match — the source
        has "Methods: ..." between the two sentences. In the 2026-07-28 prod run
        this fired 8 times, all false positives, on exactly this Pearce-Raftery
        abstract shape. The split therefore runs on the RAW quote: normalization
        deletes the glyphs that mark the boundary.
        """
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "lifespan"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        quote=(
                                            "“Background: We consider the problem of quantifying the human "
                                            "lifespan using a statistical approach that probabilistically "
                                            "forecasts the maximum reported age at death (MRAD) through 2100.” "
                                            "“We estimate the probabilities that a person lives to at least age "
                                            "126, 128, or 130 this century, as 89%, 44%, and 13%, respectively.”"
                                        ),
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning(
            "https://agency.example/report abstract: Background: We consider the problem of quantifying the "
            "human lifespan using a statistical approach that probabilistically forecasts the maximum "
            "reported age at death (MRAD) through 2100. Methods: We fit a Bayesian hierarchical model to "
            "records from 19 countries. Results: We estimate the probabilities that a person lives to at "
            "least age 126, 128, or 130 this century, as 89%, 44%, and 13%, respectively."
        )

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_web", search)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_ellipsis_joined_quote_grounds_per_span(self) -> None:
        """An ellipsis-joined quote grounds when EVERY span appears in the tool
        contents, so the driver's documented eliding style is not a false alarm.

        A contiguous substring test can never satisfy a stitched quote; the check
        splits on ellipsis and requires each span to ground independently.
        """
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "shares"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        quote='"Windows | 56.61%" ... "Linux | 4.36%"',
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning(
            "https://agency.example/report table: Windows | 56.61% then Unknown | 21.45% then Linux | 4.36% end."
        )

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_ellipsis_quote_with_absent_span_still_warns(self) -> None:
        """Per-span grounding must not become a rubber stamp: if any span is
        absent from the tool contents, the mismatch still warns."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "shares"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        quote='"Windows | 56.61%" ... "Linux | 99.99%"',
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning(
            "https://agency.example/report table: Windows | 56.61% then Unknown | 21.45% then Linux | 4.36% end."
        )

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 1

    @pytest.mark.asyncio
    async def test_whitespace_only_quote_is_grounded(self) -> None:
        """A whitespace-only quote normalizes to empty — nothing to verify — so it
        grounds without warning. Only a truly blank quote passes for free; a short
        non-blank quote is still substring-tested (see the fallback below)."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://agency.example/report", quote="   ")]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning("See https://agency.example/report for the figures.")

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_web", search)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_glyphs_on_both_sides_ground(self) -> None:
        """When the SOURCE text is itself wrapped in quote glyphs, a glyph-wrapped
        finding quote still grounds — deletion is symmetric, so both sides converge
        (the quote uses curly single quotes, the source straight doubles)."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        # The curly single quotes ARE the fixture: this test pins that
                                        # glyph deletion converges when the two sides use different
                                        # quote families, so they must not be flattened to ASCII.
                                        quote="‘The rate was 4.1 percent.’",  # noqa: RUF001
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning('https://agency.example/report quotes: "The rate was 4.1 percent." verbatim.')

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_web", search)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_unicode_ellipsis_joined_quote_grounds_per_span(self) -> None:
        """The Unicode ellipsis (…) is a span boundary too, so a quote elided with
        it grounds per-span exactly like the literal '...' form."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "shares"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        quote='"Windows | 56.61%" … "Linux | 4.36%"',
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning(
            "https://agency.example/report table: Windows | 56.61% then Unknown | 21.45% then Linux | 4.36% end."
        )

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_web", search)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_ellipsis_spans_across_separate_tool_outputs_ground(self) -> None:
        """The quote corpus concatenates every tool result, so an ellipsis-joined
        quote grounds even when its two spans came from two different retrievals."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "windows"})]),
                _response(tool_calls=[_tool_call("s2", "search_news", {"query": "linux"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        quote='"Windows | 56.61%" ... "Linux | 4.36%"',
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        web = _search_returning("https://agency.example/report first cell: Windows | 56.61% here.")
        news = _search_returning("A separate wire reports the tail cell: Linux | 4.36% today.")

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_web", web), _tool_spec("search_news", news)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_short_fabricated_numeric_span_still_warns(self) -> None:
        """A fabricated NUMBER riding alongside one long genuine span must warn.

        Regression (forge panel, proven by execution): the span filter DROPPED
        sub-floor spans instead of checking them, so a quote pairing a real long
        clause with an invented short figure grounded cleanly. Numbers are the
        high-risk case — short, fabricated, and what a forecaster acts on.
        """
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "rate"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        quote="The unemployment rate reached a new high this quarter ... 47.3%",
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning(
            "https://agency.example/report — The unemployment rate reached a new high this quarter, "
            "officials said. The figure was 3.9%."
        )

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1  # still accepted — warn-only
        assert result.telemetry.quote_mismatch_warnings == 1

    @pytest.mark.asyncio
    async def test_short_genuine_numeric_span_does_not_warn(self) -> None:
        """A short numeric span that IS present must not warn — otherwise the fix
        would false-positive on every real table-cell elision."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "rate"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        quote="The unemployment rate reached a new high this quarter ... 3.9%",
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning(
            "https://agency.example/report — The unemployment rate reached a new high this quarter, "
            "officials said. The figure was 3.9%."
        )

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_provenance_counters_surface_in_log_line(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://hallucinated.example/nowhere")]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        await run_agentic_loop("system", "briefing with no URLs", [], _config(), llm_call=fake_llm)

        marker = next(r.getMessage() for r in caplog.records if "GAP_FILL_V2:" in r.getMessage())
        assert "provenance_rejections=1" in marker
        assert "quote_mismatch_warnings=0" in marker


# Every stitch joiner that appears in the archived quote_mismatch corpus (341
# distinct quotes across the GHA run logs in `backtests/gha_artifact_store/`,
# mined 2026-08-25), with its occurrence count. The archive is gitignored, so
# these shapes are transcribed here to keep the sensitivity result reproducible
# in CI; the SPANS are synthetic, only the joiner is corpus-verbatim, which is
# all the boundary regex keys on.
#
# The 7 shapes marked "was 0/N" below are the ones the pre-2026-08-24
# whitespace-only clause could not split at all: every one of them warned even
# when both spans sat verbatim in the source. The rest were already handled by
# the ellipsis / whitespace clauses and are pinned here so broadening the regex
# never quietly loses them.
_ARCHIVED_STITCH_JOINERS: list[tuple[str, int, bool]] = [
    ('"; "', 198, True),  # was 0/198 — semicolon joiner, the single most common shape
    ("...", 155, False),
    ("”; “", 115, True),  # was 0/115 — curly-glyph semicolon
    ("” ... “", 52, False),
    ('", "', 50, True),  # was 0/50 — comma joiner
    ("” “", 49, False),
    ('" ... "', 40, False),
    ("” and “", 34, True),  # was 0/34 — the "and" joiner
    ('" "', 32, False),
    ("`; `", 27, True),  # was 0/27 — backtick spans (markdown table cells)
    ('" and "', 25, True),  # was 0/25
    ("…", 19, False),
    ('": "', 18, True),  # was 0/18 — colon joiner (label: value)
    ("”\n\n“", 13, False),
    ("” … “", 13, False),
]

_CORPUS_SPAN_A = "The agency published the schedule on 11 February 2026"
_CORPUS_SPAN_B = "the review board recorded no objections that quarter"
# Both spans verbatim, but NON-ADJACENT — the property that makes a stitched
# quote impossible to ground as one contiguous substring.
_CORPUS_BODY = _normalize_quote_text(
    f"Filing index. {_CORPUS_SPAN_A}, per the docket. Several unrelated paragraphs of "
    f"procedural text follow here. Later, {_CORPUS_SPAN_B}, closing the item."
)


class TestArchivedStitchShapesGround:
    """Corpus regression pins for the span-boundary broadening (2026-08-24).

    The measured failure was total: 0 of 156 archived multi-span quotes could
    ground on a corpus containing every span verbatim, so quote_mismatch was
    reporting the driver's punctuation rather than its honesty. These pin the real
    joiner shapes at the unit level — the loop-driven tests above cover the same
    property end to end for a few of them, but only a table over the whole corpus
    catches a regex edit that fixes one shape by dropping another.
    """

    @pytest.mark.parametrize(
        ("joiner", "archived_count", "missed_before"),
        [pytest.param(*case, id=repr(case[0])) for case in _ARCHIVED_STITCH_JOINERS],
    )
    def test_stitched_shape_grounds_when_both_spans_are_verbatim(
        self, joiner: str, archived_count: int, missed_before: bool
    ) -> None:
        del archived_count, missed_before  # documentation; see the table above
        quote = f'"{_CORPUS_SPAN_A}{joiner}{_CORPUS_SPAN_B}"'
        assert _quote_is_grounded(quote, _CORPUS_BODY)

    @pytest.mark.parametrize(
        ("joiner", "archived_count", "missed_before"),
        [pytest.param(*case, id=repr(case[0])) for case in _ARCHIVED_STITCH_JOINERS],
    )
    def test_stitched_shape_still_warns_when_a_span_is_fabricated(
        self, joiner: str, archived_count: int, missed_before: bool
    ) -> None:
        """Same shapes, second span invented: broadening the boundary must not turn
        the check into a rubber stamp for anything containing a joiner."""
        del archived_count, missed_before
        quote = f'"{_CORPUS_SPAN_A}{joiner}A FABRICATED CLAUSE THAT IS ABSENT FROM THE SOURCE"'
        assert not _quote_is_grounded(quote, _CORPUS_BODY)


class TestVerificationTierStamping:
    """W4 end-to-end: the loop stamps each banked finding's verification_tier
    from the URL->best-method map — CODE-derived, not driver-claimed. A search-
    then-fetch upgrades the tier; a failed retrieval leaves a discrepancy
    snippet-tier and demoted (the 131.3 fix). Observed via the rendered markdown
    (tier token on topic findings; block placement on discrepancies)."""

    @pytest.mark.asyncio
    async def test_fetch_sourced_finding_renders_fetched_tier(self) -> None:
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://agency.example/report"})]),
                _response(
                    tool_calls=[
                        _tool_call("rec1", "record_findings", {"findings": [_finding("https://agency.example/report")]})
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        fetch = _returns_method("The report body.", method="plain")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("fetch", fetch)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert "[verification: fetched]" in result.findings_markdown
        assert "[verification: snippet]" not in result.findings_markdown

    @pytest.mark.asyncio
    async def test_search_sourced_finding_renders_snippet_tier(self) -> None:
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call("rec1", "record_findings", {"findings": [_finding("https://agency.example/report")]})
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _returns_method("See https://agency.example/report for the figures.", method="search")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert "[verification: snippet]" in result.findings_markdown
        assert "[verification: fetched]" not in result.findings_markdown

    @pytest.mark.asyncio
    async def test_search_then_fetch_upgrades_tier_to_fetched(self) -> None:
        """A URL first surfaced in a search snippet, then actually fetched,
        upgrades to fetched (best-tier wins) — even though search ran first."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://agency.example/report"})]),
                _response(
                    tool_calls=[
                        _tool_call("rec1", "record_findings", {"findings": [_finding("https://agency.example/report")]})
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _returns_method("See https://agency.example/report — snippet only.", method="search")
        fetch = _returns_method("The full report body.", method="plain")

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_web", search), _tool_spec("fetch", fetch)],
            _config(),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        assert "[verification: fetched]" in result.findings_markdown
        assert "[verification: snippet]" not in result.findings_markdown

    @pytest.mark.asyncio
    async def test_failed_fetch_then_snippet_leaves_discrepancy_demoted(self) -> None:
        """The 131.3 scenario end-to-end: the direct fetch 403s (no tier), a
        search snippet then surfaces the contradicting figure, and the discrepancy
        banked on that URL is snippet-tier — so it renders in the DEMOTED block,
        never superseding the briefing."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://gov.example/figure"})]),
                _response(tool_calls=[_tool_call("s1", "search_news", {"query": "figure"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://gov.example/figure",
                                        quote="A snippet reports the figure is 131.3.",
                                        discrepancy=True,
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        # The direct fetch is blocked (403) — grants no tier but still marks the
        # URL tool-seen, so the discrepancy clears the provenance gate.
        blocked_fetch = _returns_method("403 Forbidden", method="plain", status="blocked")
        # The search snippet surfaces the contradicting figure at the same URL.
        snippet_search = _returns_method("https://gov.example/figure says the figure is 131.3.", method="news")

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("fetch", blocked_fetch), _tool_spec("search_news", snippet_search)],
            _config(),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        md = result.findings_markdown
        assert "### ⚠ Possible corrections (snippet-sourced — recheck advised)" in md
        assert "### ⚠ Corrections to the briefing" not in md
        assert "do NOT supersede the briefing" in md

    @pytest.mark.asyncio
    async def test_driver_supplied_tier_is_overwritten_by_code(self) -> None:
        """A driver that puts verification_tier in its record_findings payload
        cannot self-promote: the loop overwrites it from the URL->method map. A
        snippet-only URL falsely claimed "fetched" renders as snippet."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    {
                                        **_finding("https://agency.example/report"),
                                        "verification_tier": "fetched",  # driver lie
                                    }
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _returns_method("See https://agency.example/report for the figures.", method="search")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        # Code-derived snippet wins over the driver's "fetched" claim.
        assert "[verification: snippet]" in result.findings_markdown
        assert "[verification: fetched]" not in result.findings_markdown

    @pytest.mark.asyncio
    async def test_relist_after_fetch_upgrade_restamps_banked_discrepancy(self) -> None:
        """Bank-then-fetch-then-relist: a discrepancy banked from a search
        snippet, whose URL is THEN successfully fetched, upgrades to fetched
        authority when the driver re-lists it in conclude — the duplicate branch
        restamps the stored finding instead of leaving it snippet-stamped, so
        the verified correction renders in the supersede block, not demoted."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_news", {"query": "figure"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://gov.example/figure", discrepancy=True)]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://gov.example/figure"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "done1",
                            "conclude",
                            {"final_findings": [_finding("https://gov.example/figure", discrepancy=True)]},
                        )
                    ]
                ),
            ]
        )
        snippet_search = _returns_method("https://gov.example/figure says the figure is 131.3.", method="news")
        fetch = _returns_method("The full primary-source page.", method="plain")

        # cap=0 keeps the W2 conclude gate out of the way (this test's subject is
        # the duplicate-restamp, not gap accounting).
        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_news", snippet_search), _tool_spec("fetch", fetch)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        md = result.findings_markdown
        assert "### ⚠ Corrections to the briefing" in md
        assert "### ⚠ Possible corrections (snippet-sourced — recheck advised)" not in md
        conclude_message = _tool_messages(result)[-1]
        assert "Skipped 1 final finding(s) already recorded earlier in this run." in conclude_message["content"]

    def test_bank_duplicate_restamps_tier_upgrade_only(self) -> None:
        """_bank_findings unit: a duplicate re-record restamps the stored finding
        from the (upgraded) tier map, counts as a duplicate, and never appends;
        a re-record without a tier change leaves the stored finding as-is."""
        from metaculus_bot.research.agentic.loop import _bank_findings, _LoopState
        from metaculus_bot.research.agentic.types import Finding

        state = _LoopState(messages=[], started_at_s=0.0, deadline_at_s=100.0)
        finding = Finding.model_validate(_finding("https://gov.example/figure", discrepancy=True))
        state.url_best_tier["https://gov.example/figure"] = "snippet"

        assert _bank_findings(state, [finding]) == (1, 0)
        assert state.findings[0].verification_tier == "snippet"

        # Same tier map: duplicate skipped, stamp unchanged.
        assert _bank_findings(state, [finding]) == (0, 1)
        assert state.findings[0].verification_tier == "snippet"

        # Tier upgraded between banks: the re-record restamps the stored copy.
        state.url_best_tier["https://gov.example/figure"] = "fetched"
        assert _bank_findings(state, [finding]) == (0, 1)
        assert len(state.findings) == 1
        assert state.findings[0].verification_tier == "fetched"


class TestResearchPlanGate:
    """W1: set_research_plan is required before any external tool call. Until it
    runs, external calls are rejected with a nudge; the plan-nudge cap forces a
    soft-continue (plan_skipped) so a driver that never plans can't wedge. The
    plan also emits the GHOST_PRE / GHOST_PRE_JSON pre-research telemetry."""

    @pytest.mark.asyncio
    async def test_external_tool_before_plan_is_rejected_with_nudge(self) -> None:
        invoked: list[str] = []

        async def search_web(**kwargs: Any) -> ToolOutcome:
            invoked.append(kwargs.get("query", ""))
            return ToolOutcome(content_markdown="ran", method="search")

        # Turn 1 calls an external tool with no plan set -> rejected. Turn 2 sets
        # the plan. Turn 3's search then runs. Turn 4 concludes.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("early", "search_web", {"query": "too soon"})]),
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "now allowed"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # W1 test: cap=0 disables the W2 conclude gate so the early conclude here
        # isn't blocked (the plan-gate nudge is what this exercises).
        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        # The premature call was rejected: its handler never ran, and its tool
        # message carries the plan-required nudge.
        assert invoked == ["now allowed"]
        early_message = _tool_messages(result)[0]
        assert early_message["tool_call_id"] == "early"
        assert "status: error" in early_message["content"]
        assert "call set_research_plan first" in early_message["content"]
        # A rejected pre-plan call is NOT counted against the tool-call budget.
        assert result.telemetry.per_tool_counts.get("search_web") == 1
        assert result.telemetry.plan_skipped is False
        assert "The" not in result.findings_markdown or result.telemetry.findings_count == 0

    @pytest.mark.asyncio
    async def test_plan_accepted_then_external_tool_runs(self) -> None:
        invoked: list[str] = []

        async def search_web(**kwargs: Any) -> ToolOutcome:
            invoked.append(kwargs.get("query", ""))
            return ToolOutcome(content_markdown="ran", method="search")

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "allowed"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # W1 test: cap=0 disables the W2 conclude gate (early conclude, no
        # gap_accounting) so the plan-accepted path is what's under test.
        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert invoked == ["allowed"]
        assert result.telemetry.plan_gaps == 1
        assert result.telemetry.plan_skipped is False

    @pytest.mark.asyncio
    async def test_plan_emits_ghost_pre_json_marker(self, caplog: pytest.LogCaptureFixture) -> None:
        """The dry-run forecast passed to set_research_plan is logged as GHOST_PRE
        / GHOST_PRE_JSON — the pre-research counterpart to the concluding ghost."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _plan_call(
                            gaps=[{"id": "g1", "question": "q?"}],
                            sensitive_assumptions=["the incumbent runs again", "turnout stays flat"],
                            dry_run_forecast={"question_type": "binary", "posterior_prob": 0.37},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # cap=0 disables the W2 gate: this test asserts the GHOST_PRE telemetry,
        # not the conclude gate, and concludes without research/accounting.
        await run_agentic_loop(
            "system",
            "user",
            [],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
            log_prefix="question=https://q/1/ ",
        )

        pre_json_lines = [m.getMessage() for m in caplog.records if "GHOST_PRE_JSON:" in m.getMessage()]
        assert len(pre_json_lines) == 1
        blob = pre_json_lines[0].split("GHOST_PRE_JSON:", 1)[1].strip()
        assert json.loads(blob) == {"qtype": "binary", "prob": 0.37}
        # The GHOST_PRE summary line carries the plan shape and the question ref.
        pre_lines = [m.getMessage() for m in caplog.records if "GHOST_PRE:" in m.getMessage()]
        # GHOST_PRE_JSON also contains "GHOST_PRE" as a substring, so filter it out.
        pre_summary = [m for m in pre_lines if "GHOST_PRE_JSON:" not in m]
        assert len(pre_summary) == 1
        assert "gaps=1" in pre_summary[0]
        assert "sensitive_assumptions=2" in pre_summary[0]
        assert "question=https://q/1/" in pre_summary[0]

    @pytest.mark.asyncio
    async def test_ghost_pre_json_suppressed_when_no_dry_run(self, caplog: pytest.LogCaptureFixture) -> None:
        """No dry_run_forecast -> the GHOST_PRE summary still fires but the JSON
        companion is suppressed (nothing to serialize), mirroring the ghost path."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # cap=0 disables the W2 gate: GHOST_PRE-suppression telemetry is the subject.
        await run_agentic_loop("system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm)

        assert any("GHOST_PRE:" in m.getMessage() for m in caplog.records)
        assert not any("GHOST_PRE_JSON:" in m.getMessage() for m in caplog.records)
        # No dry run supplied at all is the LEGITIMATE suppression: no WARN.
        assert not any("GHOST_PRE_JSON suppressed" in m.getMessage() for m in caplog.records)

    @pytest.mark.asyncio
    async def test_ghost_pre_json_suppression_on_unparseable_dry_run_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A dry run that WAS supplied but fails to parse into a structured forecast
        (the observed prod case: flat declared percentiles failing schema validation,
        run 30718626314) suppresses GHOST_PRE_JSON — and that loss must be countable,
        because it drops exactly the flattest pre-research views, biasing the archived
        zero-move rate. GHOST_PRE still fires; the JSON half is replaced by a WARN."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _plan_call(
                            gaps=[{"id": "g1", "question": "q?"}],
                            dry_run_forecast={"not_a_recognizable_block": True},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        await run_agentic_loop("system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm)

        assert any("GHOST_PRE:" in m.getMessage() for m in caplog.records)
        assert not any("GHOST_PRE_JSON:" in m.getMessage() for m in caplog.records)
        warns = [r for r in caplog.records if "GHOST_PRE_JSON suppressed" in r.getMessage()]
        assert len(warns) == 1
        assert warns[0].levelno == logging.WARNING

    @pytest.mark.asyncio
    async def test_ghost_pre_json_suppression_on_non_dict_dry_run_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """The other half of the same loss: a driver that answers `dry_run_forecast`
        with a bare scalar rather than a forecast block. It is discarded by the
        isinstance check before any parse is attempted, so only the "was something
        supplied at all" test separates it from the legitimate no-dry-run case — and
        it must count, for the same non-random-loss reason."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _tool_call(
                            "plan0",
                            "set_research_plan",
                            {"gaps": [{"id": "g1", "question": "q?"}], "dry_run_forecast": "0.42"},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        await run_agentic_loop("system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm)

        assert not any("GHOST_PRE_JSON:" in m.getMessage() for m in caplog.records)
        warns = [r for r in caplog.records if "GHOST_PRE_JSON suppressed" in r.getMessage()]
        assert len(warns) == 1
        assert warns[0].levelno == logging.WARNING

    @pytest.mark.asyncio
    async def test_plan_nudge_cap_soft_continues_without_plan(self, caplog: pytest.LogCaptureFixture) -> None:
        """A driver that never plans hits the plan-nudge cap (2), after which the
        loop soft-continues: the external tool runs un-gated and plan_skipped is
        logged. This is the anti-wedge guard."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        invoked: list[str] = []

        async def search_web(**kwargs: Any) -> ToolOutcome:
            invoked.append(kwargs.get("query", ""))
            return ToolOutcome(content_markdown="ran", method="search")

        # Three external-only turns: turns 1 and 2 are plan-rejected (2 nudges ==
        # cap), turn 3 runs un-gated, turn 4 concludes.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("c1", "search_web", {"query": "q1"})]),
                _response(tool_calls=[_tool_call("c2", "search_web", {"query": "q2"})]),
                _response(tool_calls=[_tool_call("c3", "search_web", {"query": "q3"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("search_web", search_web)], _config(max_steps=6), llm_call=fake_llm
        )

        # First two calls were rejected pre-plan; the third ran once the cap
        # forced a soft-continue.
        assert invoked == ["q3"]
        assert result.telemetry.plan_skipped is True
        assert result.telemetry.plan_gaps == 0
        marker = next(m.getMessage() for m in caplog.records if "GAP_FILL_V2:" in m.getMessage())
        assert "plan_skipped=True" in marker
        assert "plan_gaps=0" in marker

    @pytest.mark.asyncio
    async def test_unaddressed_gaps_appear_in_budget_line(self) -> None:
        """After a plan is set, the per-turn budget line lists the plan's gap ids
        as the driver's outstanding work-list (W1 coarse accounting)."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="ran", method="search")

        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _plan_call(gaps=[{"id": "gap-a", "question": "q1?"}, {"id": "gap-b", "question": "q2?"}])
                    ]
                ),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # cap=0 disables the W2 gate: the budget-line work-list (W1) is the subject.
        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        # The search result's budget line carries both gap ids.
        search_message = _tool_messages(result)[1]
        budget_line = next(line for line in search_message["content"].splitlines() if line.startswith("[budget: "))
        assert "unaddressed_gaps=[gap-a, gap-b]" in budget_line

    @pytest.mark.asyncio
    async def test_gap_list_capped_at_max_gaps(self) -> None:
        """set_research_plan drops gaps beyond config.max_gaps (the ranked tail),
        keeping the top-N; plan_gaps reflects the kept count."""
        gaps = [{"id": f"g{i}", "question": f"q{i}?"} for i in range(6)]
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=gaps)]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # cap=0 disables the W2 gate: the max_gaps cap (W1) is the subject here.
        result = await run_agentic_loop(
            "system", "user", [], _config(max_tool_calls=14, max_conclude_gate_rejections=0), llm_call=fake_llm
        )

        # Default max_gaps is 4; only the top 4 survive.
        assert result.telemetry.plan_gaps == 4
        plan_message = _tool_messages(result)[0]
        assert "kept the top 4 of 6 gaps" in plan_message["content"]

    @pytest.mark.asyncio
    async def test_zero_gap_plan_is_rejected_and_w1_gate_stays_armed(self) -> None:
        """F3a: a zero-gap set_research_plan is rejected — research_plan is left
        None so W1's external-tool gate stays armed. A follow-up external tool
        call is still plan-rejected; the empty plan did not open the gate."""
        invoked: list[str] = []

        async def search_web(**kwargs: Any) -> ToolOutcome:
            invoked.append(kwargs.get("query", ""))
            return ToolOutcome(content_markdown="ran", method="search")

        # Turn 1: empty-gap plan (rejected). Turn 2: an external tool that must
        # still be plan-gated (only 1 nudge so far, cap not yet hit). Turn 3: a
        # real plan. Turn 4: the external tool finally runs. Turn 5: conclude.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call("p0", gaps=[])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "too soon"})]),
                _response(tool_calls=[_plan_call("p1", gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("s2", "search_web", {"query": "now allowed"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=7, max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        # The empty plan was rejected and did NOT open the gate, so the turn-2
        # search was plan-rejected; only the post-real-plan search ran.
        assert invoked == ["now allowed"]
        plan_reject = _tool_messages(result)[0]
        assert plan_reject["tool_call_id"] == "p0"
        assert "status: error" in plan_reject["content"]
        assert "Research plan rejected" in plan_reject["content"]
        early_search = _tool_messages(result)[1]
        assert early_search["tool_call_id"] == "s1"
        assert "call set_research_plan first" in early_search["content"]
        assert result.telemetry.plan_gaps == 1
        assert result.telemetry.plan_skipped is False

    @pytest.mark.asyncio
    async def test_replan_to_empty_does_not_clobber_a_prior_valid_plan(self) -> None:
        """F3a corollary: once a valid plan is set, a later empty re-plan is
        rejected rather than clobbering it — plan_gaps keeps the real count."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call("p0", gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_plan_call("p1", gaps=[])]),
                _response(tool_calls=[_tool_call("done1", "conclude", {"gap_accounting": _accounting("g1")})]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [], _config(max_steps=6, max_conclude_gate_rejections=0), llm_call=fake_llm
        )

        assert result.telemetry.plan_gaps == 1
        replan = _tool_messages(result)[1]
        assert "Research plan rejected" in replan["content"]


class TestConcludeGate:
    """W2: the conclude gate blocks an EARLY conclusion until the driver's
    gap_accounting covers every plan gap, the run made ≥1 external call per gap,
    and the fetch floor is met (top-2 gaps cite a fetch, OR ≥2 fetches/reads).
    A rejection continues the loop (no stop, no banking) and bumps
    conclude_gate_rejections; the cap accepts unconditionally after 2 rejections;
    must_conclude (deadline) and plan_skipped bypass it entirely."""

    @pytest.mark.asyncio
    async def test_reject_missing_gap_id_then_accept_on_complete_accounting(self) -> None:
        """A conclude whose accounting omits a plan gap is rejected (loop
        continues); a follow-up conclude that covers every gap is accepted."""

        async def fetch(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="body", method="rendered")

        # Plan has two gaps; two fetches meet the per-gap call count + fetch floor.
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(
                    tool_calls=[
                        _tool_call("f1", "fetch", {"url": "https://a.example"}),
                        _tool_call("f2", "fetch", {"url": "https://b.example"}),
                    ]
                ),
                # First conclude accounts for only g1 -> rejected.
                _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1")})]),
                # Second conclude accounts for both -> accepted.
                _response(tool_calls=[_tool_call("c2", "conclude", {"gap_accounting": _accounting("g1", "g2")})]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("fetch", fetch)], _config(max_steps=8), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 1
        first_conclude = _tool_messages(result)[-2]
        assert "Conclude rejected" in first_conclude["content"]
        assert "missing entries for gap(s): g2" in first_conclude["content"]
        # The accepted conclude actually stopped the loop.
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_reject_too_few_external_calls(self) -> None:
        """Two plan gaps but only one external tool call: the ≥1-call-per-gap
        invariant is unmet, so the conclude is rejected even with full accounting."""

        async def fetch(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="body", method="rendered")

        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://a.example"})]),
                # cap=0 keeps the rejection observable (the loop would otherwise
                # loop; here it soft-fails after the single scripted conclude).
                _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1", "g2")})]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("fetch", fetch)],
            _config(max_steps=6, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 1
        conclude_message = _tool_messages(result)[-1]
        assert "only 1 external tool call(s) made for 2 plan gap(s)" in conclude_message["content"]
        # Rejected -> the loop did not conclude early; it soft-failed on exhaustion.
        assert result.telemetry.concluded_early is False

    @pytest.mark.asyncio
    async def test_reject_when_both_fetch_floor_clauses_fail(self) -> None:
        """One gap, one external call, but it's a search (no fetch/read) and the
        accounting doesn't cite a fetch: both fetch-floor clauses fail -> reject."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "c1",
                            "conclude",
                            {"gap_accounting": _accounting("g1", actions="searched only, no primary source")},
                        )
                    ]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=6, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 1
        conclude_message = _tool_messages(result)[-1]
        assert "fetch floor unmet" in conclude_message["content"]

    @pytest.mark.asyncio
    async def test_accept_via_per_gap_fetch_clause(self) -> None:
        """Fetch-floor clause (a): the top-ranked gap's accounting cites a fetch,
        even though the run's own fetch count is below the global floor (one
        rendered fetch). The conclude is accepted."""

        async def fetch(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="body", method="rendered")

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://a.example"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "c1",
                            "conclude",
                            {"gap_accounting": _accounting("g1", actions="fetched the primary source PDF")},
                        )
                    ]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("fetch", fetch)], _config(max_steps=6), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_accept_via_global_fetch_count_clause(self) -> None:
        """Fetch-floor clause (b): the run made ≥2 fetches/reads, so a conclude
        whose accounting does NOT cite a fetch per gap is still accepted."""

        async def fetch(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="body", method="rendered")

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(
                    tool_calls=[
                        _tool_call("f1", "fetch", {"url": "https://a.example"}),
                        _tool_call("f2", "fetch", {"url": "https://b.example"}),
                    ]
                ),
                _response(
                    tool_calls=[
                        _tool_call(
                            "c1",
                            "conclude",
                            {"gap_accounting": _accounting("g1", actions="reviewed the search snippets")},
                        )
                    ]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("fetch", fetch)], _config(max_steps=6), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_reject_when_prose_cites_fetch_but_no_successful_retrieval(self) -> None:
        """Fetch-floor tightening: the top-ranked gaps' accounting cites fetch/read
        verbs, but every fetch 403'd (zero successful fetched-tier retrievals). The
        per-gap prose clause no longer clears the floor on its own -> reject."""

        async def fetch(**_: Any) -> ToolOutcome:
            # Every fetch 403s (status=blocked): no fetched tier is granted, so
            # url_best_tier stays empty and fetches_reads is 0 despite the calls.
            return ToolOutcome(content_markdown="Fetch blocked with HTTP 403.", method="plain", status="blocked")

        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(
                    tool_calls=[
                        _tool_call("f1", "fetch", {"url": "https://a.example"}),
                        _tool_call("f2", "fetch", {"url": "https://b.example"}),
                    ]
                ),
                _response(
                    tool_calls=[
                        _tool_call(
                            "c1",
                            "conclude",
                            {
                                "gap_accounting": _accounting(
                                    "g1", actions="fetched the primary source but received 403"
                                )
                                + _accounting("g2", actions="read_document timed out")
                            },
                        )
                    ]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("fetch", fetch)],
            _config(max_steps=6, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 1
        conclude_message = _tool_messages(result)[-1]
        assert "fetch floor unmet" in conclude_message["content"]
        # made 0 — both fetches 403'd, so no primary source was reached even though
        # the per-gap accounting narrates a fetch/read for each top gap.
        assert "made 0" in conclude_message["content"]
        assert result.telemetry.concluded_early is False

    @pytest.mark.asyncio
    async def test_accept_with_one_successful_fetch_below_global_floor(self) -> None:
        """The tightened per-gap clause still serves its purpose: one load-bearing
        fetched-tier retrieval plus honest per-gap fetch citations is accepted even
        though the run's fetch count (1) is below the global 2-fetch floor."""

        async def fetch(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="body", method="rendered")

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        # Two gaps -> the per-gap external floor needs >=2 external calls: one
        # successful fetch (fetched tier) plus one search (snippet tier). That
        # leaves fetches_reads=1, below the global floor of 2, so acceptance rides
        # on the per-gap fetch citations backed by the single real retrieval.
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(
                    tool_calls=[
                        _tool_call("f1", "fetch", {"url": "https://a.example"}),
                        _tool_call("s1", "search_web", {"query": "x"}),
                    ]
                ),
                _response(
                    tool_calls=[
                        _tool_call(
                            "c1",
                            "conclude",
                            {
                                "gap_accounting": _accounting("g1", actions="fetched the primary source PDF")
                                + _accounting("g2", actions="read the appendix table")
                            },
                        )
                    ]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("fetch", fetch), _tool_spec("search_web", search_web)],
            _config(max_steps=6),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_locally_read_documents_clear_the_fetch_floor(self) -> None:
        """A document read from bytes we hold counts toward the floor like any other retrieval.

        The floor counts fetched-tier URLs, so the local rung's two methods have to be in
        ``provenance._METHOD_TO_TIER`` or a run that read two primary-source PDFs without
        spending a model call would be told it never fetched anything and be sent back out.
        """

        async def fetch(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="the report's own text", method="pdf_local")

        async def read_document(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="[p.4] the passage asked for", method="digest_local")

        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(
                    tool_calls=[
                        _tool_call("f1", "fetch", {"url": "https://a.example/report.pdf"}),
                        _tool_call("r1", "read_document", {"url": "https://b.example/filing.pdf", "ask": "revenue?"}),
                    ]
                ),
                _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1", "g2")})]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("fetch", fetch), _tool_spec("read_document", read_document)],
            _config(max_steps=6, max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_must_conclude_bypasses_gate(self) -> None:
        """Budget exhaustion (_must_conclude) overrides the gate: a conclude with
        no gap_accounting at all is accepted, and the rejection counter stays 0."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        # max_tool_calls=2: plan (1) + one search (2) exhausts the budget, so the
        # conclude turn is in must-conclude mode and the gate is bypassed.
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                _response(tool_calls=[_tool_call("c1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("search_web", search_web)], _config(max_tool_calls=2), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_rejection_cap_accepts_after_two_rejections(self) -> None:
        """After max_conclude_gate_rejections (2) blocked concludes, the third is
        accepted unconditionally so a pathological driver can't loop forever."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        # A single gap, one snippet-only search, and three bare concludes that all
        # fail the fetch floor. The first two are rejected; the third is accepted
        # by the cap.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                _response(
                    tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
                _response(
                    tool_calls=[_tool_call("c2", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
                _response(
                    tool_calls=[_tool_call("c3", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("search_web", search_web)], _config(max_steps=8), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 2
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_plan_skipped_bypasses_gate(self) -> None:
        """When the plan-nudge cap forced a soft-continue (plan_skipped), there is
        no plan to enforce, so the conclude gate is bypassed."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        # Two plan-less external turns hit the nudge cap; turn 3 runs un-gated;
        # turn 4 concludes with no accounting -> accepted (plan_skipped bypass).
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("c1", "search_web", {"query": "q1"})]),
                _response(tool_calls=[_tool_call("c2", "search_web", {"query": "q2"})]),
                _response(tool_calls=[_tool_call("c3", "search_web", {"query": "q3"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("search_web", search_web)], _config(max_steps=8), llm_call=fake_llm
        )

        assert result.telemetry.plan_skipped is True
        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_conclude_with_no_plan_is_rejected(self) -> None:
        """F3b: conclude is not plan-gated, so a driver could conclude on turn 1
        with no plan set (plan_skipped still False). The gate now rejects that —
        planning can't be skipped by never calling set_research_plan."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        # Turn 1 concludes with no plan -> rejected; turn 2's bare conclude is
        # accepted by the rejection cap (=1), proving the loop can't wedge.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("c1", "conclude")]),
                _response(tool_calls=[_tool_call("c2", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=6, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 1
        first_conclude = _tool_messages(result)[0]
        assert "Conclude rejected" in first_conclude["content"]
        assert "no research plan was set" in first_conclude["content"]
        # The cap accepted the second conclude — no wedge.
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_valid_plan_after_nudge_cap_re_arms_the_gate(self) -> None:
        """F3c: plan_skipped was checked before the plan and never cleared, so a
        valid plan set after the plan-nudge cap fired stayed permanently exempt.
        Now a real plan re-arms the gate even once plan_skipped is True."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        # Turns 1-2: plan-less external calls hit the nudge cap (plan_skipped=True).
        # Turn 3: a real plan registers. Turn 4: an early conclude with no
        # gap_accounting must now be rejected (the gate is re-armed despite
        # plan_skipped). Turn 5: the cap (=1) accepts, so no wedge.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("c1", "search_web", {"query": "q1"})]),
                _response(tool_calls=[_tool_call("c2", "search_web", {"query": "q2"})]),
                _response(tool_calls=[_plan_call("p1", gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("d1", "conclude")]),
                _response(tool_calls=[_tool_call("d2", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=8, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.plan_skipped is True
        assert result.telemetry.plan_gaps == 1
        # The gate re-armed: the first post-plan conclude was rejected on the
        # missing gap accounting, not waved through by the stale plan_skipped.
        assert result.telemetry.conclude_gate_rejections == 1
        rejected_conclude = _tool_messages(result)[-2]
        assert "Conclude rejected" in rejected_conclude["content"]
        assert "missing entries for gap(s): g1" in rejected_conclude["content"]
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_conclude_gate_rejections_surface_in_marker(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                # One rejected conclude (fetch floor unmet), then the cap accepts.
                _response(
                    tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
                _response(
                    tool_calls=[_tool_call("c2", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
            ]
        )

        await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=8, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        marker = next(r.getMessage() for r in caplog.records if "GAP_FILL_V2:" in r.getMessage())
        assert "conclude_gate_rejections=1" in marker
