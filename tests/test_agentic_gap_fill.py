"""Integration tests for the gap-fill v2 seam (research/agentic_gap_fill.py).

Covers the seam contract (flag-gating, benchmarking hard-off, soft-fail) plus
the orchestrator-level wiring (v1+v2 concurrent sections, archive payload).
Everything is mocked — zero LLM/network calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import GeneralLlm

from metaculus_bot.prompts import MARKET_SNAPSHOT_SECTION_HEADER, binary_prompt
from metaculus_bot.research.agentic import run_agentic_loop as _real_run_agentic_loop
from metaculus_bot.research.agentic.driver_prompt import _template_skeleton, build_system_prompt
from metaculus_bot.research.agentic.types import LoopTelemetry, ToolOutcome, ToolSpec
from metaculus_bot.research.agentic_gap_fill import run_gap_fill_v2
from metaculus_bot.research.orchestrator import ResearchOrchestrator
from metaculus_bot.research.persistence import ResearchPersistenceWriter
from tests.agentic_fakes import FakeLlm
from tests.agentic_fakes import plan_call as _plan_call
from tests.agentic_fakes import response as _response
from tests.agentic_fakes import tool_call as _tool_call
from tests.pipeline_test_helpers import (
    make_real_binary_question,
    make_real_date_question,
    make_real_mc_question,
    make_real_numeric_question,
)

BUNDLE = "## News Articles (AskNews)\nSome first-pass research prose about unemployment."


_FINDING = {
    "claim": "BLS reported the unemployment rate at 4.1% for June 2026.",
    "source_url": "https://www.bls.gov/news.release/empsit.nr0.htm",
    "quote": "The unemployment rate was 4.1 percent in June.",
    "date": "2026-07-03",
    "retrieved_how": "fetch of BLS release",
    "topic": "labor-market",
}

_GHOST_BLOCK = '```json\n{"question_type": "binary", "posterior_prob": 0.12}\n```'


def _happy_path_llm() -> FakeLlm:
    """Plan (W1 gate opener), one fetch step, then conclude with a finding, then the ghost turn.

    The fetch is a real fetched-tier retrieval of the finding's source URL, so the
    conclude satisfies both the finding's provenance and the W2 fetch floor: the
    top-gap ``gap_accounting`` cites a fetch in ``actions_taken`` AND >=1 successful
    fetched-tier retrieval landed (prose alone no longer clears the floor).
    """
    return FakeLlm(
        [
            _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "What is the June 2026 rate?"}])]),
            _response(tool_calls=[_tool_call("c1", "fetch", {"url": _FINDING["source_url"]})]),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "conclude",
                        {
                            "final_findings": [_FINDING],
                            "gap_accounting": [
                                {
                                    "gap_id": "g1",
                                    "actions_taken": "fetched the BLS release and confirmed the June rate",
                                    "status": "resolved",
                                }
                            ],
                        },
                    )
                ]
            ),
            _response(content=_GHOST_BLOCK),
        ]
    )


def _fake_tools() -> list[ToolSpec]:
    async def _search(**_: Any) -> ToolOutcome:
        """Surfaces the finding's source URL like a real search result, so the provenance gate accepts _FINDING."""
        return ToolOutcome(
            content_markdown="result: BLS release found",
            links=[_FINDING["source_url"]],
            status="ok",
        )

    async def _fetch(**_: Any) -> ToolOutcome:
        """A rendered fetch of the source page: tiers the URL "fetched", clearing the provenance and fetch-floor gates."""
        return ToolOutcome(
            content_markdown="BLS release: The unemployment rate was 4.1 percent in June.",
            method="rendered",
            status="ok",
        )

    return [
        ToolSpec(
            name="search_web",
            description="fake exa",
            parameters={"type": "object", "properties": {}, "additionalProperties": True},
            handler=_search,
            timeout_s=1.0,
        ),
        ToolSpec(
            name="fetch",
            description="fake fetch",
            parameters={"type": "object", "properties": {}, "additionalProperties": True},
            handler=_fetch,
            timeout_s=1.0,
        ),
    ]


def _patch_loop_internals(fake_llm: FakeLlm):
    """Patch the seam's LLM binding and tool construction (no network, no keys)."""
    return (
        patch("metaculus_bot.research.agentic.loop.build_default_llm_call", return_value=fake_llm),
        patch("metaculus_bot.research.agentic_gap_fill.build_gap_fill_tools", return_value=_fake_tools()),
    )


class TestRunGapFillV2Seam:
    @pytest.mark.asyncio
    async def test_flag_off_returns_empty_with_zero_llm_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GAP_FILL_V2_ENABLED", raising=False)
        fake_llm = _happy_path_llm()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch, tools_patch:
            result = await run_gap_fill_v2(make_real_binary_question(), BUNDLE, is_benchmarking=False)
        assert result == ""
        assert fake_llm.calls == []

    @pytest.mark.asyncio
    async def test_benchmarking_returns_empty_with_zero_llm_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        fake_llm = _happy_path_llm()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch, tools_patch:
            result = await run_gap_fill_v2(make_real_binary_question(), BUNDLE, is_benchmarking=True)
        assert result == ""
        assert fake_llm.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "question_factory",
        [make_real_binary_question, make_real_mc_question, make_real_numeric_question, make_real_date_question],
        ids=["binary", "mc", "numeric", "date"],
    )
    async def test_happy_path_returns_findings_and_emits_markers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        question_factory,
    ) -> None:
        """Every type the bot forecasts passes the ``_SupportedQuestion`` gate: with the Mantic workflow
        running v2 and date questions about 40% of that pool, a date type dropped from the union would
        forecast them with no agentic pass and no test would notice."""
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        question = question_factory()
        fake_llm = _happy_path_llm()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch, tools_patch:
            result = await run_gap_fill_v2(question, BUNDLE, is_benchmarking=False)

        assert "## Agentic Research Findings" in result
        assert _FINDING["claim"] in result
        messages = [record.getMessage() for record in caplog.records]
        assert any("GAP_FILL_V2:" in m for m in messages)
        assert any("GHOST_FORECAST:" in m for m in messages)
        # The block must parse (non-empty summary): a broken parse degrades to qtype=unknown with summary="".
        ghost_lines = [m for m in messages if "GHOST_FORECAST:" in m]
        assert any("qtype=binary" in m and "summary=posterior_prob=0.1200" in m for m in ghost_lines)
        # log_prefix carries the question reference on the marker lines.
        assert any(f"question={question.page_url}" in m for m in messages if "GAP_FILL_V2:" in m)

    @pytest.mark.asyncio
    async def test_user_brief_embeds_question_and_bundle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The frozen prefix must carry resolution criteria, the real template, and the bundle."""
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        question = make_real_numeric_question()
        fake_llm = _happy_path_llm()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch, tools_patch:
            await run_gap_fill_v2(question, BUNDLE, is_benchmarking=False)

        first_messages = fake_llm.calls[0]["messages"]
        assert first_messages[0]["role"] == "system"
        assert "research analyst" in first_messages[0]["content"]
        brief = first_messages[1]["content"]
        assert question.resolution_criteria in brief
        assert BUNDLE in brief
        # Real numeric bounds/units render in the skeleton (OPEN_BOUND_PILING lesson).
        assert "Units: %" in brief
        assert "The upper bound is open" in brief

    @pytest.mark.asyncio
    async def test_driver_config_carries_the_question_ref_the_log_prefix_uses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The transport stamps ``LoopConfig.question_ref`` on each call's ledger metadata so a
        PROMPT_SIZE_ALERT names its question in the same ref space as the ghost markers."""
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        question = make_real_binary_question()
        fake_llm = _happy_path_llm()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch as build_llm_call, tools_patch:
            await run_gap_fill_v2(question, BUNDLE, is_benchmarking=False)

        (config,) = build_llm_call.call_args.args
        assert config.question_ref == question.page_url

    @pytest.mark.asyncio
    async def test_seam_exception_soft_fails_to_empty(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        caplog.set_level(logging.ERROR, logger="metaculus_bot.research.agentic_gap_fill")
        errors: list[BaseException] = []
        with patch(
            "metaculus_bot.research.agentic_gap_fill.build_gap_fill_tools",
            side_effect=RuntimeError("tool construction exploded"),
        ):
            result = await run_gap_fill_v2(
                make_real_binary_question(), BUNDLE, is_benchmarking=False, on_error=errors.append
            )
        assert result == ""
        assert any("Gap-fill v2 seam failed" in record.getMessage() for record in caplog.records)
        # on_error is the orchestrator's only crash signal for this path (no marker, no payload): counter path (b).
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    @pytest.mark.asyncio
    async def test_on_error_not_called_on_flag_off_skip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """on_error is a CRASH signal, not a skip signal: a clean flag-off early
        return must never fire it (else every non-v2 run would falsely alert)."""
        monkeypatch.delenv("GAP_FILL_V2_ENABLED", raising=False)
        errors: list[BaseException] = []
        result = await run_gap_fill_v2(
            make_real_binary_question(), BUNDLE, is_benchmarking=False, on_error=errors.append
        )
        assert result == ""
        assert errors == []

    @pytest.mark.asyncio
    async def test_archive_sink_receives_transcript_and_telemetry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        captured: list[dict] = []
        fake_llm = _happy_path_llm()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch, tools_patch:
            await run_gap_fill_v2(
                make_real_binary_question(),
                BUNDLE,
                is_benchmarking=False,
                archive_sink=captured.append,
            )
        assert len(captured) == 1
        payload = captured[0]
        assert payload["telemetry"]["findings_count"] == 1
        assert payload["telemetry"]["tool_calls"] == 3  # set_research_plan + fetch + conclude
        roles = [message["role"] for message in payload["transcript"]]
        assert roles[0] == "system"
        assert "tool" in roles
        # Archived as a dict, not a GhostForecast, so the payload stays an opaque JSON blob.
        ghost = payload["ghost"]
        assert isinstance(ghost, dict)
        assert ghost["qtype"] == "binary"
        assert ghost["parsed_summary"] == "posterior_prob=0.1200"
        assert _GHOST_BLOCK.split("\n", 1)[1].rsplit("\n", 1)[0] in ghost["raw_text"]

    @pytest.mark.asyncio
    async def test_archive_sink_ghost_none_when_ghost_phase_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No ghost turn (loop concludes without an explicit conclude call that
        triggers the ghost phase) → payload carries ghost=None, not a missing key."""
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        captured: list[dict] = []
        # A search then two no-tool-call turns (nudge, then stop): conclude never runs, so the ghost phase is skipped.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("c1", "search_web", {"query": "q"})]),
                _response(content="no more tool calls"),
                _response(content="still nothing to do"),
            ]
        )
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch, tools_patch:
            await run_gap_fill_v2(
                make_real_binary_question(),
                BUNDLE,
                is_benchmarking=False,
                archive_sink=captured.append,
            )
        assert len(captured) == 1
        assert "ghost" in captured[0]
        assert captured[0]["ghost"] is None


class TestDriverSystemPromptBaseRateTriage:
    """The BASE-RATE bullet must triage: research uncertain / niche / CONDITIONAL
    rates, skip common-knowledge ones instead of burning budget re-verifying."""

    def test_system_prompt_carries_research_and_skip_triggers(self) -> None:
        prompt = build_system_prompt("2026-07-16")
        assert "BASE-RATE targets" in prompt
        assert "RESEARCH the rate when ANY of these hold" in prompt
        assert "CONDITIONAL" in prompt
        assert "SKIP the lookup" in prompt
        assert "common knowledge" in prompt
        assert "real denominator and count" in prompt

    def test_base_rate_bullet_carries_process_change_check(self) -> None:
        """BTF-2 lesson: a rate from a changed regime (new gatekeeper, rule, or
        coalition) is itself a finding — the bullet must prompt the process-still-
        holds check, not just the denominator lookup."""
        # Collapse whitespace so assertions don't depend on line-wrap positions.
        collapsed = " ".join(build_system_prompt("2026-07-16").split())
        assert "check whether the process that generated the class still holds" in collapsed
        assert "same decision-maker, same rule or procedure, same coalition" in collapsed
        assert "A rate drawn from a changed regime is itself a finding worth recording" in collapsed


class TestDriverSystemPromptLiveDataSource:
    """Nebraska/Texas lesson (44554 vs 44556): a question resolving off a live
    tracker needs that tracker's CURRENT reading, and a target phrased as the
    tracker's value on the resolution date cannot be answered by any search. The
    numeric default already covers a measured quantity; this bullet covers the
    same failure on binary and MC questions that resolve off an instrument."""

    def test_system_prompt_makes_the_current_reading_a_verify_target(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert "resolves off a live data source" in collapsed
        assert "tracker, index, average, counter, or dashboard" in collapsed
        assert "its CURRENT reading, together with the date it was last updated, is a verify target" in collapsed

    def test_system_prompt_bans_future_dated_targets(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert "Never make a target of what that source will read on the resolution date" in collapsed


class TestDriverSystemPromptCatalyst:
    """BTF-2 lesson: exact-entity search misses scheduled 'catalyst' events that
    change what the deciding actor wants (their case: 17 searches about a bill,
    never searched the summit that made the government want it passed). The
    CATALYST bullet must push a calendar/agenda search, not an entity search,
    and stay detached (no read on what an empty calendar implies)."""

    def test_system_prompt_carries_catalyst_target(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-16").split())
        assert "CATALYST targets" in collapsed
        assert "spend 1-2 searches on the calendar around the question" in collapsed
        assert "scheduled event, deadline, or process change inside the question window" in collapsed
        assert "changes what the key actor wants" in collapsed
        # Search the surrounding agenda, not the entity — the whole point.
        assert "search the surrounding agenda, not the entity name" in collapsed
        # An empty calendar is a recordable finding too.
        assert 'a dated "no scheduled catalyst found inside the window" is a finding' in collapsed

    def test_catalyst_bullet_stays_detached(self) -> None:
        """The empty-calendar finding must be stated plainly with no read on the
        outcome — consistent with the detachment lint (bans likely/suggests/
        indicates in findings). The old draft's 'supports the status-quo lean'
        would editorialize likelihood; the shipped text must not."""
        collapsed = " ".join(build_system_prompt("2026-07-16").split())
        assert "stated plainly with no read on what it means for the outcome" in collapsed
        assert "supports the status-quo lean" not in collapsed
        assert "supports the lean" not in collapsed


class TestDriverSystemPromptResearchPlan:
    """W1: STEP 1 directs the driver to keep the dry run private but emit its
    outputs via set_research_plan, and to build the gap list from the briefing
    alone (v1-analyzer style) with both verify- and fill-targets."""

    def test_step1_requires_set_research_plan(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert "set_research_plan" in collapsed
        assert "REQUIRED before any research tool" in collapsed
        assert "external tool calls are rejected until you call it" in collapsed

    def test_step1_keeps_dry_run_private_but_emits_outputs(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert "This reasoning stays PRIVATE" in collapsed
        assert "dry-run forecast as the template's STRUCTURED FORECAST block" in collapsed
        assert "sensitive assumptions that would most move that forecast if wrong" in collapsed

    def test_step1_gap_list_from_briefing_alone_with_both_target_kinds(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert "Build the gap list from the BRIEFING ALONE" in collapsed
        assert "what load-bearing fact is missing or unverified" in collapsed
        assert "rank the most forecast-moving gap first" in collapsed
        # Both verify-targets (assumptions to check) AND fill-targets (absent facts).
        assert "verify-targets (assumptions to check against a primary source)" in collapsed
        assert "fill-targets (facts the briefing simply does not contain)" in collapsed

    def test_step2_references_the_ranked_gaps(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert "Work your ranked gaps in order" in collapsed
        assert "per-turn budget line shows the time, tool calls and turns you have left" in collapsed
        assert "lists your plan's gap ids" in collapsed
        # The old label claimed a live "outstanding" list the loop cannot compute (findings carry no gap id).
        assert "outstanding gaps" not in collapsed

    def test_step2_carries_derivation_license(self) -> None:
        """W3: STEP 2 grants a narrow synthesis license — the driver may put
        decision-relevant arithmetic in the ``derivation`` field, with every input
        quoted. The saw-tooth-style per-year bound table is the exemplar shape."""
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert "YOU MAY DERIVE" in collapsed
        assert "put the arithmetic in the finding's `derivation` field" in collapsed
        # Every input number must be a quoted value in the same finding.
        assert "Every input number in the derivation must ALSO appear" in collapsed
        assert "no likelihood language, no new facts" in collapsed
        # Exemplar: derive a per-year bound table from quoted record data.
        assert "per-year bound table" in collapsed


class TestDriverSystemPromptConcludeGate:
    """W2: STEP 3 must describe the conclude gate — the required gap_accounting
    (per-gap gap_id / actions_taken / status), the three statuses, and what gets
    an early conclude rejected (missing gap, too few calls, unmet fetch floor)."""

    def test_step3_requires_gap_accounting(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert "conclude REQUIRES a `gap_accounting` list" in collapsed
        assert "one entry per gap in your research plan" in collapsed
        # The three per-entry fields are named.
        assert "gap_id:" in collapsed
        assert "actions_taken:" in collapsed
        assert "status:" in collapsed

    def test_step3_lists_the_three_statuses(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert "`resolved`" in collapsed
        assert "`unresolved_parked`" in collapsed
        assert "`not_decision_relevant_on_inspection`" in collapsed

    def test_step3_describes_what_gets_rejected(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert "An EARLY conclude" in collapsed
        assert "REJECTED" in collapsed
        assert "a plan gap is missing from the accounting" in collapsed
        assert "fewer external tool calls than you have plan gaps" in collapsed
        assert "the fetch floor is unmet" in collapsed
        assert "Snippet-only research does not clear this floor" in collapsed
        # The deadline exemption is stated, but not as a loophole to coast to.
        assert "forced deadline conclusion is exempt" in collapsed
        assert "do not coast to the deadline to dodge it" in collapsed


class TestDriverSystemPromptMotivation:
    """W2: the driver runs at low reasoning effort, so the system prompt carries
    a motivation block explaining WHY thoroughness and verification are valued —
    a wrong fact is costlier than a missing one; an unverified snippet is a
    liability; depth on the few decision-relevant gaps beats breadth; fetch
    primary sources before contradicting the briefing."""

    def test_prompt_frames_wrong_facts_as_costly(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert "A wrong fact does more damage than a missing one" in collapsed
        # The concrete blow-up: an unverified snippet swung a whole ensemble.
        assert "swung an entire ensemble the wrong way" in collapsed

    def test_prompt_frames_unverified_snippet_as_liability(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert "an unconfirmed snippet is a liability, not a finding" in collapsed
        assert "a search excerpt is a lead, not a confirmation" in collapsed

    def test_prompt_values_depth_over_breadth_and_primary_sources(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert "real depth on the two or three gaps that actually move the forecast" in collapsed
        assert "pull the primary source and read the operative language yourself" in collapsed


class TestDriverSystemPromptSnippetDemotion:
    """W4: the discrepancy rule warns the driver that a snippet-only discrepancy
    is demoted (won't supersede the briefing) and to fetch the primary source
    before contradicting it — synergy with W2's fetch floor. The demotion is
    code-enforced (loop-stamped tier); the prompt just tells the driver why."""

    def test_discrepancy_rule_warns_snippet_only_is_demoted(self) -> None:
        collapsed = " ".join(build_system_prompt("2026-07-21").split())
        assert 'A discrepancy sourced only from search snippets will be demoted to "possible corrections"' in collapsed
        assert "will NOT supersede the briefing" in collapsed
        assert "if you intend to contradict the briefing, fetch the primary source first" in collapsed


class TestDriverTemplateSkeletonCarriesTheMarketClause:
    """The skeleton is the panel's real prompt with only the research slot placeholdered, and
    the driver dry-runs the forecast against it. The market-reading clause is gated on the
    research carrying MARKET_SNAPSHOT_SECTION_HEADER, so a header-free placeholder silently
    dropped a clause every prod question's real prompt carries (the provider emits that header
    whenever it rendered anything, deliberate-empty sentence included). The placeholder now
    carries the header; these pins keep the two from drifting apart again."""

    _CLAUSE_PHRASE = "the liquidity warning governs"

    def test_binary_skeleton_carries_the_market_reading_rules(self) -> None:
        skeleton = _template_skeleton(make_real_binary_question())
        assert MARKET_SNAPSHOT_SECTION_HEADER in skeleton
        assert self._CLAUSE_PHRASE in skeleton

    def test_mc_and_numeric_skeletons_carry_it_too(self) -> None:
        for question in (make_real_mc_question(), make_real_numeric_question()):
            skeleton = _template_skeleton(question)
            assert MARKET_SNAPSHOT_SECTION_HEADER in skeleton
            assert self._CLAUSE_PHRASE in skeleton

    def test_a_header_free_render_still_omits_the_clause(self) -> None:
        """The gate itself is unchanged: research with no market section (a benchmark run, a
        flag-off run, a soft-failed provider) gets no clause. Pinned here so a future fix to
        the skeleton cannot be mistaken for the gate having been loosened."""
        prompt = binary_prompt(make_real_binary_question(), "## News Articles (AskNews)\nSome prose.")
        assert MARKET_SNAPSHOT_SECTION_HEADER not in prompt
        assert self._CLAUSE_PHRASE not in prompt


@pytest.fixture
def mock_llm() -> GeneralLlm:
    return GeneralLlm(model="test/model", temperature=0.0)


class TestOrchestratorBothFlags:
    """Orchestrator-level wiring: v1 and v2 sections coexist; archive carries the trace."""

    @pytest.mark.asyncio
    async def test_both_sections_render_and_v2_appends_after_v1(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(
            # Long enough to clear GAP_FILL_MIN_RESEARCH_CHARS (200) so the v1 gate opens.
            return_value="First-pass research prose long enough to pass the gap-fill min-chars gate. " * 4
        )
        v2_section = "## Agentic Research Findings\n\n### labor-market\n\nClaim: BLS says 4.1%."

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass",
                new_callable=AsyncMock,
                return_value="v1 gap-fill addendum text",
            ) as v1_mock,
            patch(
                "metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2",
                new_callable=AsyncMock,
                return_value=v2_section,
            ) as v2_mock,
        ):
            research = await orch.run_research(make_real_binary_question())

        v1_mock.assert_awaited_once()
        v2_mock.assert_awaited_once()
        assert "## Targeted Gap-Fill (second pass)" in research
        assert "## Agentic Research Findings" in research
        assert research.index("## Targeted Gap-Fill (second pass)") < research.index("## Agentic Research Findings")

    @pytest.mark.asyncio
    async def test_v2_flag_off_leaves_v1_path_untouched(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.delenv("GAP_FILL_V2_ENABLED", raising=False)

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(
            # Long enough to clear GAP_FILL_MIN_RESEARCH_CHARS (200) so the v1 gate opens.
            return_value="First-pass research prose long enough to pass the gap-fill min-chars gate. " * 4
        )

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass",
                new_callable=AsyncMock,
                return_value="v1 gap-fill addendum text",
            ),
            patch(
                "metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2",
                new_callable=AsyncMock,
            ) as v2_mock,
        ):
            research = await orch.run_research(make_real_binary_question())

        # The stage gates on the v2 flag before awaiting the seam, so the seam is never called when the flag is off.
        assert v2_mock.await_count == 0
        assert "## Targeted Gap-Fill (second pass)" in research
        assert "## Agentic Research Findings" not in research

    @pytest.mark.asyncio
    async def test_v2_import_error_leaves_v1_addendum_intact(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A v2 module-level import failure must degrade only v2, never v1.

        Regression guard for the merged-guard bug: with one shared try block, a
        broken v2 import zeroed v1's addendum too even with the v2 flag off in
        every other respect.
        """
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(
            return_value="First-pass research prose long enough to pass the gap-fill min-chars gate. " * 4
        )

        # None in sys.modules makes the stage's function-level import raise ImportError, as a broken v2 tree would.
        monkeypatch.setitem(sys.modules, "metaculus_bot.research.agentic_gap_fill", None)

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass",
                new_callable=AsyncMock,
                return_value="v1 gap-fill addendum text",
            ),
        ):
            research = await orch.run_research(make_real_binary_question())

        assert "## Targeted Gap-Fill (second pass)" in research
        assert "v1 gap-fill addendum text" in research
        assert "## Agentic Research Findings" not in research

    @pytest.mark.asyncio
    async def test_v2_runtime_error_leaves_v1_addendum_intact(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(
            return_value="First-pass research prose long enough to pass the gap-fill min-chars gate. " * 4
        )

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass",
                new_callable=AsyncMock,
                return_value="v1 gap-fill addendum text",
            ),
            patch(
                "metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2",
                new_callable=AsyncMock,
                side_effect=RuntimeError("v2 exploded"),
            ),
        ):
            research = await orch.run_research(make_real_binary_question())

        assert "## Targeted Gap-Fill (second pass)" in research
        assert "## Agentic Research Findings" not in research

    @pytest.mark.asyncio
    async def test_v1_runtime_error_leaves_v2_findings_intact(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(
            return_value="First-pass research prose long enough to pass the gap-fill min-chars gate. " * 4
        )

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass",
                new_callable=AsyncMock,
                side_effect=RuntimeError("v1 exploded"),
            ),
            patch(
                "metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2",
                new_callable=AsyncMock,
                return_value="## Agentic Research Findings\n\nClaim: BLS says 4.1%.",
            ),
        ):
            research = await orch.run_research(make_real_binary_question())

        assert "## Targeted Gap-Fill (second pass)" not in research
        assert "## Agentic Research Findings" in research

    @pytest.mark.asyncio
    async def test_archive_payload_carries_gap_fill_v2_key(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

        captured: dict = {}

        def sink(**kwargs) -> None:
            captured.update(kwargs)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            allow_research_fallback=False,
            research_sink=sink,
        )
        provider = AsyncMock(return_value="research prose")
        trace = {"transcript": [{"role": "system", "content": "x"}], "telemetry": {"steps": 2}}

        async def _fake_v2(
            question, bundle, *, is_benchmarking, archive_sink=None, ghost_context_sink=None, on_error=None
        ):
            if archive_sink is not None:
                archive_sink(trace)
            return "## Agentic Research Findings\n\nClaim: something."

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch("metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2", side_effect=_fake_v2),
        ):
            await orch.run_research(make_real_binary_question())

        assert captured["gap_fill_v2"] == trace

    @pytest.mark.asyncio
    async def test_archive_payload_omits_key_when_v2_off(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
        monkeypatch.delenv("GAP_FILL_V2_ENABLED", raising=False)

        captured: dict = {}

        def sink(**kwargs) -> None:
            captured.update(kwargs)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            allow_research_fallback=False,
            research_sink=sink,
        )
        provider = AsyncMock(return_value="research prose")
        with patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]):
            await orch.run_research(make_real_binary_question())

        assert captured["gap_fill_v2"] is None

    def test_persistence_writer_serializes_gap_fill_v2(self, tmp_path) -> None:
        """Writer round-trips the v2 trace and omits the key when absent."""
        writer = ResearchPersistenceWriter(run_mode="tournament", platform="metaculus", tournament_id="t", run_id="r")
        trace = {"transcript": [{"role": "system", "content": "x"}], "telemetry": {"steps": 3}}
        writer.record(
            qid=1,
            page_url="https://www.metaculus.com/questions/1/",
            question_text="Q?",
            research_text="research",
            providers_used=["asknews"],
            gap_fill_used=False,
            gap_fill_v2=trace,
        )
        writer.record(
            qid=2,
            page_url="https://www.metaculus.com/questions/2/",
            question_text="Q2?",
            research_text="research",
            providers_used=["asknews"],
            gap_fill_used=False,
        )
        out = writer.flush(output_dir=str(tmp_path))
        assert out is not None
        records = [json.loads(line) for line in out.read_text().strip().splitlines()]
        assert records[0]["gap_fill_v2"] == trace
        assert "gap_fill_v2" not in records[1]


async def _run_v2_through_orchestrator_with_driver(orch: ResearchOrchestrator, driver: Any) -> LoopTelemetry:
    """Run the REAL agentic loop end-to-end through the orchestrator with a stub
    ``driver``, returning the loop's telemetry.

    ``run_agentic_loop`` is wrapped (patched only in the seam's namespace, so the
    wrapper still calls the genuine loop) to capture the ``LoopResult.telemetry``
    for assertions the orchestrator counter alone can't make — ``deadline_hit`` and
    ``error``. ``build_default_llm_call`` is stubbed to ``driver`` (the loop's first
    ``llm_call``), tools are faked, and a single ``AsyncMock`` stands in for the
    research providers — mirroring the other crash-counter tests. The REAL loop's
    ``except asyncio.TimeoutError`` classification runs, which is the point.
    """
    captured: dict[str, Any] = {}

    async def _capturing_loop(*args: Any, **kwargs: Any) -> Any:
        result = await _real_run_agentic_loop(*args, **kwargs)
        captured["telemetry"] = result.telemetry
        return result

    provider = AsyncMock(return_value="research prose")
    with (
        patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
        patch("metaculus_bot.research.agentic_gap_fill.run_agentic_loop", _capturing_loop),
        patch("metaculus_bot.research.agentic.loop.build_default_llm_call", return_value=driver),
        patch("metaculus_bot.research.agentic_gap_fill.build_gap_fill_tools", return_value=_fake_tools()),
    ):
        await orch.run_research(make_real_binary_question())
    return captured["telemetry"]


class TestGapFillV2CrashCounter:
    """The alertable crash counter (orchestrator.gap_fill_v2_error_count).

    Detection backstop for the CLASS of bug the fastapi eager-import defect
    exposed: v2 crashed on step 0, soft-failed to "", and emitted a marker
    byte-identical to a legitimate idle run — nothing reddened CI. A genuine
    crash must now bump an alertable counter; an idle "found nothing" run and a
    deadline hit must NOT. Three mutually-exclusive crash paths, one bump each
    (see gap_fill_stages._run_gap_fill_v2). A dead-on-arrival bug bumps it
    on every question; a rare transient bumps it once (accepted false alarm).
    """

    @pytest.mark.asyncio
    async def test_seam_construction_crash_bumps_counter_once(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Path (b) end-to-end: the REAL run_gap_fill_v2 seam crashes in tool
        construction, soft-fails to "", and its on_error callback bumps the
        orchestrator counter exactly once (no payload -> post-gather check is a no-op)."""
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(return_value="research prose")

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.agentic_gap_fill.build_gap_fill_tools",
                side_effect=RuntimeError("tool construction exploded"),
            ),
        ):
            research = await orch.run_research(make_real_binary_question())

        assert "## Agentic Research Findings" not in research  # soft-failed, still published
        assert orch.gap_fill_v2_error_count == 1

    @pytest.mark.asyncio
    async def test_loop_internal_soft_fail_bumps_counter_via_telemetry_error(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Path (a): the loop ran but hit its catch-all soft-fail, stamping
        telemetry.error and returning normally. The orchestrator's post-gather
        check reads the archived telemetry error and bumps the counter."""
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(return_value="research prose")

        async def _fake_v2(
            question, bundle, *, is_benchmarking, archive_sink=None, ghost_context_sink=None, on_error=None
        ):
            """Mirrors the loop's soft-fail: swallows the crash, stamps telemetry.error, archives the payload, returns ""."""
            if archive_sink is not None:
                archive_sink(
                    {"transcript": [], "telemetry": {"steps": 0, "error": "APIConnectionError('boom')"}, "ghost": None}
                )
            return ""

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch("metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2", side_effect=_fake_v2),
        ):
            await orch.run_research(make_real_binary_question())

        assert orch.gap_fill_v2_error_count == 1

    @pytest.mark.asyncio
    async def test_import_escape_error_bumps_counter(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Path (c): an error escaping the seam entirely (e.g. a broken import)
        is caught by _run_v2's own except, which counts it directly."""
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(return_value="research prose")

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2",
                new_callable=AsyncMock,
                side_effect=RuntimeError("v2 escaped its own soft-fail"),
            ),
        ):
            await orch.run_research(make_real_binary_question())

        assert orch.gap_fill_v2_error_count == 1

    @pytest.mark.asyncio
    async def test_idle_run_does_not_bump_counter(self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch) -> None:
        """Negative control: a clean run that legitimately found nothing (telemetry
        error is None) must NOT bump the counter — otherwise every empty-findings
        question would falsely redden CI. This is the byte-identical-marker case."""
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(return_value="research prose")

        async def _fake_v2(
            question, bundle, *, is_benchmarking, archive_sink=None, ghost_context_sink=None, on_error=None
        ):
            if archive_sink is not None:
                archive_sink({"transcript": [], "telemetry": {"steps": 0, "error": None}, "ghost": None})
            return ""  # nothing to research — a legitimate empty run, not a crash

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch("metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2", side_effect=_fake_v2),
        ):
            await orch.run_research(make_real_binary_question())

        assert orch.gap_fill_v2_error_count == 0

    @pytest.mark.asyncio
    async def test_genuine_deadline_through_real_loop_does_not_bump_counter(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Negative control through the REAL loop: a driver that runs past the wall
        deadline is a genuine deadline hit, not a crash — deadline_hit=True, no
        stamped error, counter stays 0. An expected-degradation deadline must not
        redden CI (the class's 'a deadline hit must NOT [bump]' contract)."""
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        # A tiny wall deadline, kept above _DEADLINE_SLOP_S so the elapsed time still classifies as a real deadline.
        monkeypatch.setattr("metaculus_bot.research.agentic_gap_fill.GAP_FILL_V2_WALL_DEADLINE", 1.0)

        async def _sleeping_driver(_messages: Any, _tools: Any) -> Any:
            await asyncio.sleep(30)  # cancelled by the outer wait_for deadline
            raise AssertionError("driver should have been cancelled by the deadline")

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        telemetry = await _run_v2_through_orchestrator_with_driver(orch, _sleeping_driver)

        assert telemetry.deadline_hit is True
        assert telemetry.error is None
        assert orch.gap_fill_v2_error_count == 0

    @pytest.mark.asyncio
    async def test_spurious_inner_timeout_bumps_counter_and_is_not_a_deadline(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression for the TimeoutError misclassification: a bare (builtin)
        TimeoutError raised INSIDE the driver — well before the wall deadline — is a
        crash, not a deadline. On Python 3.11+ asyncio.TimeoutError IS builtin
        TimeoutError, so pre-fix code stamped deadline_hit=True and never counted it.
        Post-fix: error is stamped, deadline_hit=False, and the counter bumps once
        via the orchestrator's post-gather archive-telemetry path."""
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        # Default wall deadline: the driver raises at once, so elapsed ~0 forces the crash classification.

        async def _timeout_driver(_messages: Any, _tools: Any) -> Any:
            raise TimeoutError("simulated connection-level timeout")

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        telemetry = await _run_v2_through_orchestrator_with_driver(orch, _timeout_driver)

        assert telemetry.deadline_hit is False
        assert telemetry.error is not None
        assert "TimeoutError" in telemetry.error
        assert orch.gap_fill_v2_error_count == 1
