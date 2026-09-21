from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from metaculus_bot.research.agentic.loop import _summarize_ghost, run_agentic_loop
from metaculus_bot.research.agentic.types import ToolOutcome
from metaculus_bot.structured_output_schema import NumericStructured
from tests.agentic_fakes import FakeLlm
from tests.agentic_fakes import gap_accounting as _accounting
from tests.agentic_fakes import loop_config as _config
from tests.agentic_fakes import plan_call as _plan_call
from tests.agentic_fakes import response as _response
from tests.agentic_fakes import tool_call as _tool_call
from tests.agentic_fakes import tool_messages as _tool_messages
from tests.agentic_fakes import tool_spec as _tool_spec


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, delta: float) -> None:
        self.value += delta


def _tool_names(tools_json: list[dict[str, Any]] | None) -> list[str]:
    return [] if tools_json is None else [tool["function"]["name"] for tool in tools_json]


@pytest.mark.asyncio
async def test_happy_path_records_findings_and_emits_telemetry(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")

    async def fetch(**_: Any) -> ToolOutcome:
        # Reach a primary source: a rendered fetch tiers the URL "fetched", so the
        # conclude gate's fetch floor is satisfied by a real retrieval — not by the
        # accounting's fetch-verb prose alone.
        return ToolOutcome(content_markdown="Authoritative page text.", method="rendered")

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "When was the report published?"}])]),
            _response(tool_calls=[_tool_call("fetch1", "fetch", {"url": "https://example.com"})]),
            _response(
                tool_calls=[
                    _tool_call(
                        "record1",
                        "record_findings",
                        {
                            "findings": [
                                {
                                    "claim": "The report was published on July 1.",
                                    "source_url": "https://example.com",
                                    "quote": "Published on July 1, 2026.",
                                    "date": "2026-07-01",
                                    "retrieved_how": "fetch",
                                    "topic": "timeline",
                                }
                            ]
                        },
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "done1",
                        "conclude",
                        {
                            "pending_leads": ["Check the appendix."],
                            # W2 conclude gate: one gap, so its accounting must
                            # cover g1 and (per-gap fetch-floor clause) cite a
                            # fetch/read action in actions_taken.
                            "gap_accounting": [
                                {
                                    "gap_id": "g1",
                                    "actions_taken": "fetched the primary source and confirmed the date",
                                    "status": "resolved",
                                }
                            ],
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
        _config(),
        llm_call=fake_llm,
    )

    assert "## Agentic Research Findings" in result.findings_markdown
    assert "The report was published on July 1." in result.findings_markdown
    assert result.telemetry.model == "openai/gpt-5.6-luna"
    assert result.telemetry.steps == 4
    assert result.telemetry.tool_calls == 4
    assert result.telemetry.per_tool_counts == {
        "set_research_plan": 1,
        "fetch": 1,
        "record_findings": 1,
        "conclude": 1,
    }
    assert result.telemetry.plan_gaps == 1
    assert result.telemetry.plan_skipped is False
    assert result.telemetry.rendered_fetches == 1
    assert result.telemetry.dup_tool_calls == 0
    assert result.telemetry.findings_count == 1
    assert result.telemetry.pending_leads_count == 1
    assert result.telemetry.concluded_early is True
    marker_lines = [record.getMessage() for record in caplog.records if "GAP_FILL_V2:" in record.getMessage()]
    assert len(marker_lines) == 1
    # Plan-§6 marker shape: model + per-surface counters, grep-able in run_logs.
    marker = marker_lines[0]
    assert "model=openai/gpt-5.6-luna" in marker
    assert "searches=0" in marker
    assert "fetches=1" in marker
    assert "rendered=1" in marker
    assert "reads=0" in marker
    assert "dup_tool_calls=0" in marker
    assert "plan_gaps=1" in marker
    assert "plan_skipped=False" in marker
    assert "conclude_gate_rejections=0" in marker


@pytest.mark.asyncio
async def test_duplicate_tool_calls_counted_and_warned(caplog: pytest.LogCaptureFixture) -> None:
    """Exact-duplicate (tool, normalized-args) repeats bump dup_tool_calls and
    append a gentle warning to the duplicate's tool result — no enforcement."""
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")

    async def search_web(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="done", method="search")

    # Same args with shuffled key order still count as the same call.
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report", "extra": 1})]),
            _response(tool_calls=[_tool_call("s2", "search_web", {"extra": 1, "query": "report"})]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("search_web", search_web)],
        _config(),
        llm_call=fake_llm,
    )

    assert result.telemetry.dup_tool_calls == 1
    # tool_messages[0] is the set_research_plan result; the two searches follow.
    tool_messages = _tool_messages(result)
    first, second = tool_messages[1], tool_messages[2]
    assert "already made earlier in this run" not in first["content"]
    assert "already made earlier in this run" in second["content"]
    # The duplicate still executed — counter + warning only, no enforcement.
    assert "done" in second["content"]
    marker = next(record.getMessage() for record in caplog.records if "GAP_FILL_V2:" in record.getMessage())
    assert "dup_tool_calls=1" in marker


@pytest.mark.asyncio
async def test_throttled_fetch_retry_is_not_flagged_a_duplicate() -> None:
    """A throttled fetch's identical retry must not carry the duplicate note.

    The throttle message asks the driver to call fetch on the same URL again
    later in the run, and the outcome is deliberately not cached, so the retry
    really does re-request the page. Telling it "its result will not have
    changed" would steer it off the URL this handling exists to recover.
    """

    async def fetch(**_: Any) -> ToolOutcome:
        return ToolOutcome(
            content_markdown="Rate limited: https://ogimet.com returned HTTP 200, but its body is a short "
            'interstitial carrying the throttle phrase "wait 20 seconds".',
            method="throttled",
            status="throttled",
        )

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://ogimet.com/q"})]),
            _response(tool_calls=[_tool_call("f2", "fetch", {"url": "https://ogimet.com/q"})]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("fetch", fetch)],
        # The W2 conclude gate is beside the point here and a throttled fetch
        # reaches no primary source, so keep it out of the way rather than
        # scripting an extra turn.
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
    )

    # tool_messages[0] is the set_research_plan result; the two fetches follow.
    tool_messages = _tool_messages(result)
    first, second = tool_messages[1], tool_messages[2]
    assert "already made earlier in this run" not in first["content"]
    assert "already made earlier in this run" not in second["content"]
    assert "Rate limited" in second["content"]
    assert result.telemetry.dup_tool_calls == 0
    # The retry is still a real call against the budget — eviction is advisory
    # bookkeeping only, so max_tool_calls keeps capping a throttle spin.
    assert result.telemetry.per_tool_counts["fetch"] == 2


@pytest.mark.asyncio
async def test_rendered_fetch_outcomes_counted_in_telemetry() -> None:
    async def fetch(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="rendered body", method="rendered")

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://example.com"})]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("fetch", fetch)],
        _config(),
        llm_call=fake_llm,
    )

    assert result.telemetry.per_tool_counts["fetch"] == 1
    assert result.telemetry.rendered_fetches == 1


@pytest.mark.asyncio
async def test_parallel_tool_calls_execute_concurrently_and_append_budget_line() -> None:
    started: list[str] = []
    finished: list[str] = []
    release = asyncio.Event()

    def _parallel_handler(name: str) -> Any:
        async def _handler(**_: Any) -> ToolOutcome:
            started.append(name)
            if len(started) == 3:
                release.set()
            await release.wait()
            finished.append(name)
            return ToolOutcome(content_markdown=f"{name} done", method="plain")

        return _handler

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(
                tool_calls=[
                    _tool_call("a", "alpha"),
                    _tool_call("b", "beta"),
                    _tool_call("c", "gamma"),
                ]
            ),
            _response(tool_calls=[_tool_call("done", "conclude")]),
        ]
    )

    result = await asyncio.wait_for(
        run_agentic_loop(
            "system",
            "user",
            [
                _tool_spec("alpha", _parallel_handler("alpha")),
                _tool_spec("beta", _parallel_handler("beta")),
                _tool_spec("gamma", _parallel_handler("gamma")),
            ],
            # cap=0 disables the W2 conclude gate: this test exercises parallel
            # execution + budget-line rendering, and concludes without accounting.
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        ),
        timeout=0.5,
    )

    assert started == ["alpha", "beta", "gamma"]
    assert set(finished) == {"alpha", "beta", "gamma"}
    # tool_messages[0] is the set_research_plan result; the parallel batch follows.
    tool_messages = _tool_messages(result)[1:4]
    assert [message["name"] for message in tool_messages] == ["alpha", "beta", "gamma"]
    budget_lines = [
        next(line for line in message["content"].splitlines() if line.startswith("[budget: "))
        for message in tool_messages
    ]
    assert len(set(budget_lines)) == 1
    assert budget_lines[0].startswith("[budget: ")
    # plan (1) + alpha/beta/gamma (3) = 4 tool calls used. The default plan's gap
    # id trails as the W1 work-list suffix.
    assert "4/14 tool calls used" in budget_lines[0]


@pytest.mark.asyncio
async def test_budget_threshold_switches_to_conclude_mode_and_shrinks_tools() -> None:
    clock = FakeClock()

    async def search_web(**_: Any) -> ToolOutcome:
        clock.advance(95.0)
        return ToolOutcome(content_markdown="late result", method="search")

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("search1", "search_web")]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("search_web", search_web)],
        _config(wall_deadline_s=100.0, conclude_threshold_s=10.0),
        llm_call=fake_llm,
        now=clock,
    )

    # The plan turn (tool message [0]) runs at full budget; the search (message
    # [1]) advances the clock to 5s remaining, tipping into must-conclude mode.
    # The default plan's gap id trails as the W1 work-list suffix.
    assert (
        "[budget: 5s remaining — you must conclude now unaddressed_gaps=[g1]]" in _tool_messages(result)[1]["content"]
    )
    # calls[2] is the conclude turn — the first LLM call made after the clock
    # crossed the threshold, so its tool schema is shrunk to internals only.
    assert _tool_names(fake_llm.calls[2]["tools"]) == ["set_research_plan", "record_findings", "conclude"]


@pytest.mark.asyncio
async def test_deadline_mid_tool_still_returns_banked_findings() -> None:
    async def slow_tool(**_: Any) -> ToolOutcome:
        await asyncio.sleep(0.2)
        return ToolOutcome(content_markdown="too slow", method="plain")

    # Turn 1 sets the plan (so the external slow_tool clears the W1 gate); turn 2
    # banks a finding and fires the slow tool that trips the wall deadline.
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(
                tool_calls=[
                    _tool_call(
                        "record1",
                        "record_findings",
                        {
                            "findings": [
                                {
                                    "claim": "The statute took effect in 2025.",
                                    "source_url": "https://example.com/statute",
                                    "quote": "This act takes effect in 2025.",
                                    "topic": "law",
                                }
                            ]
                        },
                    ),
                    _tool_call("slow1", "slow_tool"),
                ]
            ),
        ]
    )

    result = await run_agentic_loop(
        "system",
        # Briefing embeds the cited URL, so the finding is provenance-grounded
        # (a non-discrepancy finding may cite a briefing URL).
        "briefing cites https://example.com/statute",
        [_tool_spec("slow_tool", slow_tool, timeout_s=1.0)],
        _config(wall_deadline_s=0.05, max_steps=3),
        llm_call=fake_llm,
    )

    assert "The statute took effect in 2025." in result.findings_markdown
    assert result.telemetry.deadline_hit is True
    assert result.ghost is None


@pytest.mark.asyncio
async def test_tool_timeout_is_reported_and_loop_continues() -> None:
    async def slow_tool(**_: Any) -> ToolOutcome:
        await asyncio.sleep(0.05)
        return ToolOutcome(content_markdown="never seen", method="plain")

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("slow1", "slow_tool")]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("slow_tool", slow_tool, timeout_s=0.01)],
        _config(),
        llm_call=fake_llm,
    )

    # tool_messages[0] is the plan result; the timed-out slow_tool is [1].
    assert "status: timeout" in _tool_messages(result)[1]["content"]
    assert result.telemetry.steps == 3


@pytest.mark.asyncio
async def test_record_findings_rejects_lint_and_banks_clean_findings() -> None:
    fake_llm = FakeLlm(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "record1",
                        "record_findings",
                        {
                            "findings": [
                                {
                                    "claim": "This likely resolves soon.",
                                    "source_url": "https://example.com/bad",
                                    "quote": "Likely to resolve soon.",
                                    "topic": "general",
                                },
                                {
                                    "claim": "The board published the minutes.",
                                    "source_url": "https://example.com/good",
                                    "quote": "Minutes were published on Tuesday.",
                                    "topic": "minutes",
                                },
                            ]
                        },
                    )
                ]
            ),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    # Briefing embeds the clean finding's URL so it clears the provenance gate;
    # the lint rejection is what this test exercises.
    result = await run_agentic_loop(
        "system", "briefing cites https://example.com/good", [], _config(), llm_call=fake_llm
    )

    assert "The board published the minutes." in result.findings_markdown
    assert "This likely resolves soon." not in result.findings_markdown
    assert result.telemetry.lint_rejections == 1
    assert "findings[0] rejected" in _tool_messages(result)[0]["content"]


@pytest.mark.asyncio
async def test_derivation_banned_phrase_is_not_a_lint_rejection() -> None:
    """W3: a banned-register phrase confined to the ``derivation`` field is the
    arithmetic-synthesis carve-out — it clears the lint, banks, and does NOT bump
    lint_rejections. The same phrase in ``claim`` would still be rejected."""
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "What are the yearly records?"}])]),
            _response(
                tool_calls=[
                    _tool_call(
                        "record1",
                        "record_findings",
                        {
                            "findings": [
                                {
                                    "claim": "Per-year oldest-human bound reconstructed from the quoted record.",
                                    "source_url": "https://example.com/grg",
                                    "quote": "1997: 122; 1998: 116; 1999: 119.",
                                    "date": "2026-07-01",
                                    "retrieved_how": "fetch",
                                    "topic": "oldest-human bounds",
                                    "derivation": (
                                        "1997 max 122; the annual record probably tops the prior year by 1 — "
                                        "arithmetic only, inputs above."
                                    ),
                                }
                            ]
                        },
                    )
                ]
            ),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    # max_conclude_gate_rejections=0 disables the W2 gate: this test exercises
    # the W3 lint carve-out, not the conclude gate, and concludes without
    # research/accounting.
    result = await run_agentic_loop(
        "system",
        "briefing cites https://example.com/grg",
        [],
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
    )

    assert result.telemetry.lint_rejections == 0
    assert "Per-year oldest-human bound reconstructed from the quoted record." in result.findings_markdown
    assert "Derived analysis (arithmetic from quoted sources)" in result.findings_markdown


_FINDING_UN = {
    "claim": "The UN projects a population peak of ~10.3 billion in the mid-2080s.",
    "source_url": "https://example.com/un",
    "quote": "peak of around 10.3 billion people in the mid-2080s",
    "date": "2024-07-11",
    "retrieved_how": "fetch",
    "topic": "demographics",
}
_FINDING_NASA = {
    "claim": "NASA reports no known >140m asteroid with significant 100-year impact risk.",
    "source_url": "https://example.com/nasa",
    "quote": "no known asteroid larger than 140 meters",
    "date": "2023-09-08",
    "retrieved_how": "fetch",
    "topic": "asteroids",
}


@pytest.mark.asyncio
async def test_findings_recorded_then_relisted_in_conclude_are_not_duplicated() -> None:
    """Regression (Q578): a driver that banks findings with record_findings and
    then re-lists the SAME findings in conclude's final_findings must not double
    them (observed: 8 findings rendered 16 times). Banking is idempotent by
    full-field identity."""
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_tool_call("rec1", "record_findings", {"findings": [_FINDING_UN, _FINDING_NASA]})]),
            _response(tool_calls=[_tool_call("done1", "conclude", {"final_findings": [_FINDING_UN, _FINDING_NASA]})]),
        ]
    )

    # cap=0 disables the W2 conclude gate: this test never plans (dedup is the
    # subject) and concludes without gap accounting.
    result = await run_agentic_loop(
        "system",
        "briefing cites https://example.com/un and https://example.com/nasa",
        [],
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
    )

    md = result.findings_markdown
    assert md.count("The UN projects a population peak") == 1
    assert md.count("NASA reports no known") == 1
    assert result.telemetry.findings_count == 2
    conclude_message = _tool_messages(result)[-1]
    assert "Skipped 2 final finding(s) already recorded earlier in this run." in conclude_message["content"]
    assert "Concluded with 0 final finding(s)" in conclude_message["content"]


@pytest.mark.asyncio
async def test_findings_sharing_source_but_distinct_claim_are_both_kept() -> None:
    """Dedup must key on the WHOLE finding, not just source_url: two findings
    from the same page with different claims/topics are genuinely distinct and
    must both survive (the Q578 log had exactly this shape for the NASA page)."""
    long_claim = {
        **_FINDING_NASA,
        "topic": "asteroids (with NEO Surveyor context)",
        "claim": "NASA reports no known >140m asteroid impact risk and says NEO Surveyor will accelerate discovery.",
    }
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_tool_call("rec1", "record_findings", {"findings": [_FINDING_NASA, long_claim]})]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system", "briefing cites https://example.com/nasa", [], _config(), llm_call=fake_llm
    )

    assert result.telemetry.findings_count == 2
    assert "NEO Surveyor" in result.findings_markdown


@pytest.mark.asyncio
async def test_cap_exhaustion_restricts_next_step_to_internal_tools() -> None:
    async def search_web(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="done", method="search")

    # Budget of 2: the plan call (1) plus one external search (2) exhaust it, so
    # the conclude turn's tool schema shrinks to internals only.
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("search1", "search_web")]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("search_web", search_web)],
        _config(max_tool_calls=2),
        llm_call=fake_llm,
    )

    # calls[2] is the conclude turn — offered only internal tools once the cap hit.
    assert _tool_names(fake_llm.calls[2]["tools"]) == ["set_research_plan", "record_findings", "conclude"]
    assert result.telemetry.tool_calls == 3


_CLEAN_FINDING = {
    "claim": "The report was published on July 1.",
    "source_url": "https://example.com",
    "quote": "Published on July 1, 2026.",
    "topic": "timeline",
}


@pytest.mark.asyncio
async def test_external_tool_bind_error_extra_key_becomes_error_and_loop_continues() -> None:
    """An LLM-emitted extra kwarg makes spec.handler(**arguments) raise TypeError
    at bind time (async-def binds eagerly). It must surface as a status=error
    tool result inside the per-tool boundary, not crash the batch gather and
    abort the whole pass. The loop keeps running and banked findings survive."""
    invoked: list[str] = []

    async def strict_tool(*, query: str) -> ToolOutcome:
        invoked.append(query)
        return ToolOutcome(content_markdown="ran", method="search")

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("bad1", "strict_tool", {"query": "x", "unexpected": 1})]),
            _response(tool_calls=[_tool_call("record1", "record_findings", {"findings": [_CLEAN_FINDING]})]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "briefing cites https://example.com",
        [_tool_spec("strict_tool", strict_tool)],
        _config(),
        llm_call=fake_llm,
    )

    assert result.telemetry.steps == 4  # loop ran all four turns — no abort
    # tool_messages[0] is the plan result; the bind-error tool result is [1].
    bad_message = _tool_messages(result)[1]
    assert bad_message["tool_call_id"] == "bad1"
    assert "status: error" in bad_message["content"]
    assert "TypeError" in bad_message["content"]
    assert invoked == []  # handler body never ran; failure was at bind time
    assert "The report was published on July 1." in result.findings_markdown


@pytest.mark.asyncio
async def test_external_tool_bind_error_missing_key_becomes_error_and_loop_continues() -> None:
    """A missing required kwarg raises TypeError at bind time too; same contract:
    error tool result, loop continues, banked findings survive."""
    invoked: list[str] = []

    async def strict_tool(*, query: str) -> ToolOutcome:
        invoked.append(query)
        return ToolOutcome(content_markdown="ran", method="search")

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("bad1", "strict_tool", {})]),  # no `query`
            _response(tool_calls=[_tool_call("record1", "record_findings", {"findings": [_CLEAN_FINDING]})]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "briefing cites https://example.com",
        [_tool_spec("strict_tool", strict_tool)],
        _config(),
        llm_call=fake_llm,
    )

    assert result.telemetry.steps == 4
    # tool_messages[0] is the plan result; the bind-error tool result is [1].
    bad_message = _tool_messages(result)[1]
    assert bad_message["tool_call_id"] == "bad1"
    assert "status: error" in bad_message["content"]
    assert "TypeError" in bad_message["content"]
    assert invoked == []
    assert "The report was published on July 1." in result.findings_markdown


@pytest.mark.asyncio
async def test_batch_clamped_to_remaining_budget_rejects_overflow_external_calls() -> None:
    """A single turn can emit more calls than budget allows (parallel_tool_calls).
    The batch is clamped to the remaining external-call slots: in-budget calls run,
    overflow external calls are rejected with a synthetic budget-exhausted error,
    every tool_call_id still gets exactly one response, and internal tools
    (record_findings) keep working past the ceiling."""
    invoked: list[str] = []

    async def search_web(*, query: str) -> ToolOutcome:
        invoked.append(query)
        return ToolOutcome(content_markdown=f"result {query}", method="search")

    # Budget of 3: the plan call (1) leaves two external slots, so q1/q2 run and
    # the overflow q3 is budget-rejected — the shape this test exercises.
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(
                tool_calls=[
                    _tool_call("c1", "search_web", {"query": "q1"}),
                    _tool_call("c2", "search_web", {"query": "q2"}),
                    _tool_call("c3", "search_web", {"query": "q3"}),
                ]
            ),
            _response(tool_calls=[_tool_call("record1", "record_findings", {"findings": [_CLEAN_FINDING]})]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "briefing cites https://example.com",
        [_tool_spec("search_web", search_web)],
        _config(max_tool_calls=3),
        llm_call=fake_llm,
    )

    # Only the two in-budget external calls executed; the third never bound.
    assert set(invoked) == {"q1", "q2"}
    assert len(invoked) == 2
    assert result.telemetry.per_tool_counts["search_web"] == 2

    # tool_messages[0] is the plan result; the clamped parallel batch is [1:4].
    batch_messages = _tool_messages(result)[1:4]
    assert [message["tool_call_id"] for message in batch_messages] == ["c1", "c2", "c3"]

    rejected_message = next(message for message in batch_messages if message["tool_call_id"] == "c3")
    assert "status: error" in rejected_message["content"]
    assert "budget is exhausted" in rejected_message["content"]
    accepted_messages = [message for message in batch_messages if message["tool_call_id"] in {"c1", "c2"}]
    assert all("budget is exhausted" not in message["content"] for message in accepted_messages)

    # record_findings ran past the ceiling — internal tools are not budget-gated.
    assert "The report was published on July 1." in result.findings_markdown


@pytest.mark.asyncio
async def test_append_only_message_history_is_preserved_across_steps() -> None:
    async def search_web(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="done", method="search")

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("search1", "search_web")]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("search_web", search_web)],
        _config(),
        llm_call=fake_llm,
    )

    for earlier, later in zip(fake_llm.calls, fake_llm.calls[1:], strict=False):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]


@pytest.mark.asyncio
async def test_soft_fail_preserves_banked_findings_on_broken_llm_response() -> None:
    fake_llm = FakeLlm(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "record1",
                        "record_findings",
                        {
                            "findings": [
                                {
                                    "claim": "The agency released the bulletin.",
                                    "source_url": "https://example.com/bulletin",
                                    "quote": "Bulletin released Friday.",
                                    "topic": "bulletin",
                                }
                            ]
                        },
                    )
                ]
            ),
            object(),
        ]
    )

    result = await run_agentic_loop(
        "system", "briefing cites https://example.com/bulletin", [], _config(), llm_call=fake_llm
    )

    assert "The agency released the bulletin." in result.findings_markdown
    assert result.ghost is None


@pytest.mark.asyncio
async def test_soft_fail_returns_empty_result_on_injected_loop_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    async def search_web(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="done", method="search")

    fake_llm = FakeLlm([_response(tool_calls=[_tool_call("search1", "search_web")])])
    monkeypatch.setattr(
        "metaculus_bot.research.agentic.loop._tool_schemas",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("search_web", search_web)],
        _config(),
        llm_call=fake_llm,
    )

    assert result.findings_markdown == ""
    assert result.ghost is None


@pytest.mark.asyncio
async def test_ghost_phase_runs_after_conclude_and_logs_marker(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
    ghost_text = 'analysis\n```json\n{"question_type":"binary","posterior_prob":0.42}\n```'
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_tool_call("done1", "conclude")]),
            _response(content=ghost_text),
        ]
    )

    result = await run_agentic_loop(
        "system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm, ghost_prompt="ghost now"
    )

    assert result.ghost is not None
    assert result.ghost.qtype == "binary"
    assert result.ghost.parsed_summary == "posterior_prob=0.4200"
    assert any("GHOST_FORECAST:" in record.getMessage() for record in caplog.records)


def _ghost_json_payload(caplog: pytest.LogCaptureFixture) -> dict:
    """Extract and parse the single GHOST_FORECAST_JSON marker line from captured logs."""
    lines = [record.getMessage() for record in caplog.records if "GHOST_FORECAST_JSON:" in record.getMessage()]
    assert len(lines) == 1, f"expected exactly one GHOST_FORECAST_JSON line, got {lines}"
    blob = lines[0].split("GHOST_FORECAST_JSON:", 1)[1].strip()
    return json.loads(blob)


@pytest.mark.asyncio
async def test_ghost_phase_emits_full_fidelity_json_marker_binary(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
    ghost_text = 'analysis\n```json\n{"question_type":"binary","posterior_prob":0.42}\n```'
    fake_llm = FakeLlm([_response(tool_calls=[_tool_call("d", "conclude")]), _response(content=ghost_text)])

    await run_agentic_loop(
        "system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm, ghost_prompt="ghost now"
    )

    payload = _ghost_json_payload(caplog)
    assert payload == {"qtype": "binary", "prob": 0.42}


@pytest.mark.asyncio
async def test_ghost_phase_emits_full_fidelity_json_marker_mc(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
    ghost_text = 'analysis\n```json\n{"question_type":"multiple_choice","option_probs":{"Blue":0.3,"Red":0.7}}\n```'
    fake_llm = FakeLlm([_response(tool_calls=[_tool_call("d", "conclude")]), _response(content=ghost_text)])

    await run_agentic_loop(
        "system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm, ghost_prompt="ghost now"
    )

    payload = _ghost_json_payload(caplog)
    assert payload == {"qtype": "multiple_choice", "option_probs": {"Blue": 0.3, "Red": 0.7}}


@pytest.mark.asyncio
async def test_ghost_phase_emits_full_fidelity_json_marker_numeric(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
    ghost_text = (
        'analysis\n```json\n{"question_type":"numeric","declared_percentiles":{"0.1":10.0,"0.5":20.5,"0.9":30.0}}\n```'
    )
    fake_llm = FakeLlm([_response(tool_calls=[_tool_call("d", "conclude")]), _response(content=ghost_text)])

    await run_agentic_loop(
        "system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm, ghost_prompt="ghost now"
    )

    payload = _ghost_json_payload(caplog)
    # JSON serializes float keys as strings; the full percentile set survives round-trip.
    assert payload == {
        "qtype": "numeric",
        "declared_percentiles": {"0.1": 10.0, "0.5": 20.5, "0.9": 30.0},
        "median": 20.5,
    }


@pytest.mark.asyncio
async def test_ghost_phase_suppresses_json_marker_when_unparseable(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
    fake_llm = FakeLlm([_response(tool_calls=[_tool_call("d", "conclude")]), _response(content="no block here")])

    await run_agentic_loop(
        "system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm, ghost_prompt="ghost now"
    )

    # Legacy marker still fires (qtype=unknown); the JSON companion is suppressed.
    assert any("GHOST_FORECAST:" in r.getMessage() for r in caplog.records)
    assert not any("GHOST_FORECAST_JSON:" in r.getMessage() for r in caplog.records)


class TestSummarizeGhost:
    """Branch coverage for _summarize_ghost (MC + numeric; binary is covered
    by test_ghost_phase_runs_after_conclude_and_logs_marker above). Also asserts
    the third tuple element — the full-fidelity forecast payload."""

    def test_multiple_choice_formats_sorted_option_probs(self) -> None:
        raw = (
            "analysis\n```json\n"
            '{"question_type": "multiple_choice", "option_probs": {"Zeta": 0.5, "Alpha": 0.3, "Mid": 0.2}}'
            "\n```"
        )

        qtype, summary, payload = _summarize_ghost(raw)

        assert qtype == "multiple_choice"
        assert summary == "Alpha=0.300, Mid=0.200, Zeta=0.500"
        assert payload == {"qtype": "multiple_choice", "option_probs": {"Zeta": 0.5, "Alpha": 0.3, "Mid": 0.2}}

    def test_numeric_reports_median(self) -> None:
        raw = (
            "analysis\n```json\n"
            '{"question_type": "numeric", "declared_percentiles": {"0.1": 10.0, "0.5": 20.5, "0.9": 30.0}}'
            "\n```"
        )

        qtype, summary, payload = _summarize_ghost(raw)

        assert qtype == "numeric"
        assert summary == "median=20.5"
        assert payload == {
            "qtype": "numeric",
            "declared_percentiles": {0.1: 10.0, 0.5: 20.5, 0.9: 30.0},
            "median": 20.5,
        }

    def test_numeric_missing_median_yields_empty_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The median-None guard. Unreachable through a schema-valid parse
        (NumericStructured requires percentile 0.5), so build the block via
        model_construct and stub the parse to return it."""
        block = NumericStructured.model_construct(question_type="numeric", declared_percentiles={0.1: 10.0, 0.9: 30.0})

        def fake_parse(_raw_text: str, qtype: str) -> Any:
            return block if qtype == "numeric" else None

        monkeypatch.setattr("metaculus_bot.research.agentic.loop.parse_structured_block", fake_parse)

        qtype, summary, payload = _summarize_ghost("whatever")

        assert qtype == "numeric"
        assert summary == ""
        # Payload still carries the full (median-less) percentile set for scoring.
        assert payload == {"qtype": "numeric", "declared_percentiles": {0.1: 10.0, 0.9: 30.0}, "median": None}

    def test_unparseable_text_reports_unknown(self) -> None:
        qtype, summary, payload = _summarize_ghost("no structured block here")

        assert qtype == "unknown"
        assert summary == ""
        assert payload is None

    @pytest.mark.parametrize(
        ("qtype", "block"),
        [
            ("numeric", '{"question_type":"numeric","declared_percentiles":{"0.1":10.0,"0.5":20.5,"0.9":30.0}}'),
            ("multiple_choice", '{"question_type":"multiple_choice","option_probs":{"Blue":0.3,"Red":0.7}}'),
            ("binary", '{"question_type":"binary","posterior_prob":0.42}'),
        ],
    )
    def test_no_qtype_mismatch_warnings_on_declared_block(
        self, caplog: pytest.LogCaptureFixture, qtype: str, block: str
    ) -> None:
        """Regression (BUG 2): the probe used to try all three qtypes, tripping
        the shared parser's question_type-mismatch WARN for the non-matching
        ones (5 spurious WARNs/run). Reading the declared type first parses only
        the matching type — zero mismatch WARNs on the expected path."""
        caplog.set_level(logging.WARNING, logger="metaculus_bot.structured_output_schema")

        result_qtype, _summary, _payload = _summarize_ghost(f"analysis\n```json\n{block}\n```")

        assert result_qtype == qtype
        mismatch_warns = [r for r in caplog.records if "question_type mismatch" in r.getMessage()]
        assert mismatch_warns == []


@pytest.mark.asyncio
async def test_ghost_phase_numeric_emits_no_qtype_mismatch_warnings(caplog: pytest.LogCaptureFixture) -> None:
    """End-to-end guard: a numeric ghost run through the full loop leaves no
    question_type-mismatch WARN in the structured_output_schema logger."""
    caplog.set_level(logging.WARNING, logger="metaculus_bot.structured_output_schema")
    ghost_text = (
        'analysis\n```json\n{"question_type":"numeric","declared_percentiles":{"0.1":10.0,"0.5":20.5,"0.9":30.0}}\n```'
    )
    fake_llm = FakeLlm([_response(tool_calls=[_tool_call("d", "conclude")]), _response(content=ghost_text)])

    result = await run_agentic_loop(
        "system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm, ghost_prompt="ghost now"
    )

    assert result.ghost is not None
    assert result.ghost.qtype == "numeric"
    assert [r for r in caplog.records if "question_type mismatch" in r.getMessage()] == []


class TestHarvestVerificationTiers:
    """W4: the per-call tier harvester. Fetched-class methods (document/rendered/
    plain/cache) tier only the URLs actually retrieved (the arguments) "fetched";
    snippet-class methods (search/news) tier every surfaced URL "snippet"; a
    failed/blocked outcome or a non-retrieval method grants no tier at all."""

    def test_fetched_class_methods_tier_argument_urls_fetched(self) -> None:
        from metaculus_bot.research.agentic.loop import _harvest_verification_tiers

        # pdf_local and digest_local belong in this class because both mean we decoded the bytes
        # the host served (pypdf text, then a deterministic passage selection), which is a
        # stronger claim than "document" — a model's reading of a URL we never saw ourselves.
        for method in ("document", "rendered", "plain", "cache", "pdf_local", "digest_local"):
            tiers = _harvest_verification_tiers(
                "fetch",
                {"url": "https://x.example/a"},
                ToolOutcome(content_markdown="body", method=method),
            )
            assert tiers == {"https://x.example/a": "fetched"}, method

    def test_fetched_class_does_not_tier_body_or_link_urls(self) -> None:
        """A fetch's outbound links and in-body URLs are leads, not pages we
        read — only the requested URL earns the fetched tier from this call."""
        from metaculus_bot.research.agentic.loop import _harvest_verification_tiers

        tiers = _harvest_verification_tiers(
            "fetch",
            {"url": "https://x.example/a"},
            ToolOutcome(
                content_markdown="body cites https://z.example/c",
                links=["https://y.example/b"],
                method="plain",
            ),
        )
        assert tiers == {"https://x.example/a": "fetched"}

    def test_read_document_ask_url_is_not_tiered_fetched(self) -> None:
        """F1: read_document takes a free-text ``ask``. A URL the driver types
        into ``ask`` (e.g. "compare with https://other") was never retrieved —
        only the ``url`` page was — so only ``url`` earns the fetched tier."""
        from metaculus_bot.research.agentic.loop import _harvest_verification_tiers

        tiers = _harvest_verification_tiers(
            "read_document",
            {"url": "https://x.example/a", "ask": "how does this compare to https://other.example/b?"},
            ToolOutcome(content_markdown="the answer", method="document"),
        )
        assert tiers == {"https://x.example/a": "fetched"}

    def test_search_class_methods_tier_all_surfaced_urls_snippet(self) -> None:
        from metaculus_bot.research.agentic.loop import _harvest_verification_tiers

        for method in ("search", "news"):
            tiers = _harvest_verification_tiers(
                "search_web",
                {"query": "q"},
                ToolOutcome(
                    content_markdown="see https://x.example/a",
                    links=["https://y.example/b"],
                    method=method,
                ),
            )
            assert tiers == {"https://x.example/a": "snippet", "https://y.example/b": "snippet"}, method

    def test_failed_outcome_grants_no_tier(self) -> None:
        """The 131.3 mechanism: a fetch that 403s (status != ok) confers no
        authority, so a later search snippet of the same fact stays snippet."""
        from metaculus_bot.research.agentic.loop import _harvest_verification_tiers

        tiers = _harvest_verification_tiers(
            "fetch",
            {"url": "https://x.example/a"},
            ToolOutcome(content_markdown="403", method="plain", status="blocked"),
        )
        assert tiers == {}

    def test_internal_tool_and_unknown_method_grant_no_tier(self) -> None:
        from metaculus_bot.research.agentic.loop import _harvest_verification_tiers

        assert (
            _harvest_verification_tiers("record_findings", {}, ToolOutcome(content_markdown="ok", method="internal"))
            == {}
        )
        # document_needed is an intermediate fetch state, not a real retrieval.
        assert (
            _harvest_verification_tiers(
                "fetch", {"url": "https://x.example/a"}, ToolOutcome(content_markdown="", method="document_needed")
            )
            == {}
        )


class TestActionsCiteFetch:
    """F2: the per-gap fetch-floor self-report clause matches fetch/read verbs on
    word boundaries. A bare substring test fired on "already"/"spread" (contain
    "read") and on the honest failed-fetch note "could not fetch the source" —
    both cleared the floor with zero real retrievals."""

    def test_narration_words_containing_read_do_not_clear(self) -> None:
        from metaculus_bot.research.agentic.loop import _actions_cite_fetch

        for phrase in (
            "already covered by the briefing",
            "spread the estimates across scenarios",
            "the thread was inconclusive",
            "ready to conclude",
            "readily available in the digest",
        ):
            assert _actions_cite_fetch(phrase) is False, phrase

    def test_failed_or_attempted_fetch_note_does_not_clear(self) -> None:
        from metaculus_bot.research.agentic.loop import _actions_cite_fetch

        for phrase in (
            "could not fetch the primary source",
            "tried to fetch but it 403'd",
            "will fetch next run",
        ):
            assert _actions_cite_fetch(phrase) is False, phrase

    def test_real_fetch_read_mentions_clear(self) -> None:
        from metaculus_bot.research.agentic.loop import _actions_cite_fetch

        for phrase in (
            "read the PDF and confirmed the figure",
            "fetched the release",
            "fetches the dataset and cross-checked",
            "fetching the report resolved it",
            "used read_document on the filing",
            "reads the appendix table",
        ):
            assert _actions_cite_fetch(phrase) is True, phrase


class TestFetchFloorCountsSuccessfulRetrievals:
    """F4: the global fetch-floor clause counts successful fetched-tier
    retrievals (url_best_tier), not fetch/read tool CALLS. per_tool_counts
    increments at accept time regardless of outcome, so two 403'd fetches would
    otherwise clear the floor though they reached nothing — the 131.3 mechanism."""

    @pytest.mark.asyncio
    async def test_two_blocked_fetches_do_not_clear_the_global_floor(self) -> None:
        async def fetch(**_: Any) -> ToolOutcome:
            # Both fetches 403 (status=blocked): no fetched tier is granted, so
            # url_best_tier stays empty and the global floor is unmet.
            return ToolOutcome(content_markdown="Fetch blocked with HTTP 403.", method="plain", status="blocked")

        # One gap; the accounting cites "searched" (no fetch verb), so the per-gap
        # clause fails too. With two fetch CALLS but zero successful retrievals,
        # the conclude must be rejected on the fetch floor.
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
                    tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
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
        # made 0 — the two calls happened but neither retrieved a primary source.
        assert "made 0" in conclude_message["content"]


class TestNormalizeUrl:
    def test_lowercases_scheme_and_host_only(self) -> None:
        from metaculus_bot.research.agentic.loop import _normalize_url

        assert _normalize_url("HTTPS://Example.COM/Path") == "https://example.com/Path"

    def test_strips_trailing_slash_fragment_and_trackers(self) -> None:
        from metaculus_bot.research.agentic.loop import _normalize_url

        assert (
            _normalize_url("https://example.com/report/?utm_source=x&id=7&gclid=abc#frag")
            == "https://example.com/report?id=7"
        )

    def test_rejects_non_http_scheme(self) -> None:
        from metaculus_bot.research.agentic.loop import _normalize_url

        assert _normalize_url("ftp://example.com/file") is None
        assert _normalize_url("not a url") is None
