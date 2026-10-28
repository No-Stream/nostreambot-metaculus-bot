"""The v1 ghost through the seam and the stage (research/agentic_gap_fill.py, research/gap_fill_stages.py).

Gap-fill v1 and v2 run concurrently, so the v2 loop never sees v1's section. The loop hands out its
pre-ghost context (``GhostContext``) through the seam's ``ghost_context_sink``; once both passes have
landed, the stage re-asks the driver with gap-fill v1's section in the brief, on the same tool list
with tools forbidden, and the answer is archived beside the plain ghost as ``ghost_v1``. The pair
measures v1's marginal value on the driver (docs/agentic_gap_fill.md "The ghost forecast"). Shares its
fixtures with tests/test_agentic_gap_fill.py; zero LLM or network calls.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import GeneralLlm

from metaculus_bot.research.agentic.types import GhostContext
from metaculus_bot.research.agentic_gap_fill import run_gap_fill_v2, run_gap_fill_v2_ghost_v1
from metaculus_bot.research.orchestrator import ResearchOrchestrator
from tests.agentic_fakes import FakeLlm
from tests.agentic_fakes import response as _response
from tests.agentic_fakes import tool_call as _tool_call
from tests.pipeline_test_helpers import make_real_binary_question
from tests.test_agentic_gap_fill import BUNDLE, _happy_path_llm, _patch_loop_internals


@pytest.fixture
def mock_llm() -> GeneralLlm:
    return GeneralLlm(model="test/model", temperature=0.0)


_V1_GHOST_BLOCK = '```json\n{"question_type": "binary", "posterior_prob": 0.20}\n```'
_V1_ADDENDUM = "The BLS calendar confirms the July release date; no revision to June."


def _happy_path_llm_with_v1_ghost() -> FakeLlm:
    """The happy path plus one more scripted answer, the v1 ghost's."""
    fake_llm = _happy_path_llm()
    fake_llm._responses.append(_response(content=_V1_GHOST_BLOCK))
    return fake_llm


class TestRunGapFillV2GhostV1Seam:
    """The v1 ghost through the seam: the loop hands out its pre-ghost context, and the seam re-asks the driver
    with gap-fill v1's section in the brief, on the same tool list with tools forbidden."""

    @pytest.mark.asyncio
    async def test_ghost_context_sink_receives_the_context_only_when_the_plain_ghost_ran(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        contexts: list[GhostContext] = []
        fake_llm = _happy_path_llm()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch, tools_patch:
            await run_gap_fill_v2(
                make_real_binary_question(), BUNDLE, is_benchmarking=False, ghost_context_sink=contexts.append
            )

        (context,) = contexts
        plain_ghost_call = fake_llm.calls[-1]
        assert context.messages == plain_ghost_call["messages"][:-1]
        assert context.tools_json == plain_ghost_call["tools"]

    @pytest.mark.asyncio
    async def test_no_plain_ghost_means_the_sink_is_never_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        contexts: list[GhostContext] = []
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
                make_real_binary_question(), BUNDLE, is_benchmarking=False, ghost_context_sink=contexts.append
            )
        assert contexts == []

    @pytest.mark.asyncio
    async def test_v1_ghost_carries_v1s_section_in_its_brief_and_logs_its_own_markers(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        question = make_real_binary_question()
        contexts: list[GhostContext] = []
        fake_llm = _happy_path_llm_with_v1_ghost()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch, tools_patch:
            await run_gap_fill_v2(question, BUNDLE, is_benchmarking=False, ghost_context_sink=contexts.append)
            ghost_v1 = await run_gap_fill_v2_ghost_v1(question, contexts[0], _V1_ADDENDUM)

        *_, plain_ghost_call, v1_call = fake_llm.calls
        brief = v1_call["messages"][-1]["content"]
        assert f"## Targeted Gap-Fill (second pass)\n\n{_V1_ADDENDUM}" in brief
        assert "STRUCTURED FORECAST block" in brief
        assert v1_call["messages"][:-1] == plain_ghost_call["messages"][:-1]
        assert v1_call["tools"] == plain_ghost_call["tools"]
        assert v1_call["tool_choice"] == "none"

        assert ghost_v1 is not None
        assert (ghost_v1.qtype, ghost_v1.parsed_summary) == ("binary", "posterior_prob=0.2000")
        messages = [record.getMessage() for record in caplog.records]
        assert f"question={question.page_url} GHOST_FORECAST_V1: qtype=binary summary=posterior_prob=0.2000" in messages
        assert any(m.startswith(f"question={question.page_url} GHOST_FORECAST_V1_JSON:") for m in messages)


class TestV1GhostThroughTheStage:
    """Orchestrator-level wiring: the v1 ghost runs after the gather, from the REAL loop's context."""

    @pytest.mark.asyncio
    async def test_v1_ghost_runs_after_both_passes_with_v1s_section_and_is_archived(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """v1 and v2 run concurrently, so the loop never sees v1's section; the stage issues the v1 ghost after
        the gather, from the REAL loop's context, and the payload the archive receives carries it as ghost_v1."""
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        captured: dict = {}

        def sink(**kwargs) -> None:
            captured.update(kwargs)

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, research_sink=sink)
        provider = AsyncMock(
            return_value="First-pass research prose long enough to pass the gap-fill min-chars gate. " * 4
        )
        fake_llm = _happy_path_llm_with_v1_ghost()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass",
                new_callable=AsyncMock,
                return_value="v1 gap-fill addendum text",
            ),
            llm_patch,
            tools_patch,
        ):
            research = await orch.run_research(make_real_binary_question())

        assert "## Targeted Gap-Fill (second pass)" in research
        assert "## Agentic Research Findings" in research
        v1_call = fake_llm.calls[-1]
        assert "## Targeted Gap-Fill (second pass)\n\nv1 gap-fill addendum text" in v1_call["messages"][-1]["content"]
        assert v1_call["tool_choice"] == "none"
        messages = [record.getMessage() for record in caplog.records]
        assert len([m for m in messages if "GHOST_FORECAST_V1: " in m]) == 1
        assert len([m for m in messages if "GHOST_FORECAST: " in m]) == 1
        ghost_v1 = captured["gap_fill_v2"]["ghost_v1"]
        assert (ghost_v1["qtype"], ghost_v1["parsed_summary"]) == ("binary", "posterior_prob=0.2000")

    @pytest.mark.asyncio
    async def test_no_v1_section_means_no_v1_ghost(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With gap-fill v1 off there is nothing to pair against, so no second call, no marker, and a
        None placeholder in the archive payload rather than a missing key."""
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        captured: dict = {}

        def sink(**kwargs) -> None:
            captured.update(kwargs)

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, research_sink=sink)
        provider = AsyncMock(return_value="research prose")
        fake_llm = _happy_path_llm()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            llm_patch,
            tools_patch,
        ):
            await orch.run_research(make_real_binary_question())

        assert len(fake_llm.calls) == 4
        assert not any("GHOST_FORECAST_V1" in record.getMessage() for record in caplog.records)
        assert captured["gap_fill_v2"]["ghost_v1"] is None
