"""The gap-fill v2 loop's per-run state, its per-turn records, and the budget over them.

``_LoopState`` is the single mutable record every subsystem folds into — banked
findings, the provenance URL sets, the URL->verification-tier map, the telemetry
counters and the research plan. Exactly one instance exists per run, created in
``run_agentic_loop``; ``_ToolCall`` and ``_ToolExecutionResult`` are the per-turn
records that flow through admission, dispatch and absorption.

The assistant-message parsers sit here because they are what PRODUCES those
records: one LLM response becomes a normalized assistant message plus a list of
``_ToolCall``s. The budget helpers sit here because each is arithmetic over the
same state — how much wall clock is left, whether the driver must conclude now,
and the ``[budget: ...]`` line appended to every tool message.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from metaculus_bot.research.agentic.types import (
    Finding,
    LoopConfig,
    LoopTelemetry,
    ResearchPlan,
)
from metaculus_bot.research.image_assets import ImageView


@dataclass(slots=True)
class _LoopState:
    messages: list[dict[str, Any]]
    started_at_s: float
    deadline_at_s: float
    telemetry: LoopTelemetry = field(default_factory=LoopTelemetry)
    findings: list[Finding] = field(default_factory=list)
    # Canonical-JSON identity -> index in ``findings`` for already-banked
    # findings, so re-recording the same finding (record_findings then a re-list
    # in conclude's final_findings) restamps the stored copy's tier instead of
    # double-appending. See gates._bank_findings.
    seen_finding_keys: dict[str, int] = field(default_factory=dict)
    pending_leads: list[str] = field(default_factory=list)
    seen_tool_calls: set[tuple[str, str]] = field(default_factory=set)
    # Provenance gate accumulators (see provenance._normalize_url). Normalized
    # URLs the driver actually saw via a TOOL this run — fetch/read call
    # arguments plus every URL in a tool result's content/links. Discrepancy
    # findings must cite one of these (a fresh primary-source check).
    tool_seen_urls: set[str] = field(default_factory=set)
    # Normalized URL -> best retrieval tier seen this run ("fetched" outranks
    # "snippet"), stamped onto each finding's verification_tier at banking time
    # (W4). Only successful (status=ok) tool outcomes contribute a tier, so a
    # 403'd fetch never grants "fetched" authority — a later search snippet of
    # the same fact lands "snippet" and its discrepancy is demoted. A URL only
    # in the briefing (never retrieved) is absent here -> untiered finding.
    url_best_tier: dict[str, str] = field(default_factory=dict)
    # Normalized URLs embedded in the frozen briefing bundle. Non-discrepancy
    # findings may cite these too; discrepancies may NOT.
    briefing_urls: set[str] = field(default_factory=set)
    # Concatenated, normalized tool-result contents — the corpus the warn-only
    # quote spot-check searches.
    tool_content_normalized: str = ""
    # (source_url, quote) pairs already warned this run. The driver re-lists its
    # banked findings in conclude's final_findings, so without this an unmatched
    # quote warns twice and inflates the per-question density (5.5% of all
    # archived warnings were exact re-submissions, 2026-08-24 residual round).
    warned_quote_keys: set[tuple[str, str]] = field(default_factory=set)
    nudged_for_no_action: bool = False
    explicit_conclude: bool = False
    stop_loop: bool = False
    # Per-run log prefix (question ref), so internal-tool handlers can emit
    # markers keyed the same way as the ghost phase (GHOST_PRE at plan-set time).
    log_prefix: str = ""
    # Turn-one research plan (W1). None until set_research_plan runs; external
    # tool calls are rejected with _PLAN_REQUIRED_NUDGE until it exists (unless
    # the plan-nudge cap forces a soft-continue). W2 reads research_plan.gaps.
    research_plan: ResearchPlan | None = None
    # Count of external tool calls rejected by the plan gate, and whether the cap
    # was hit so we soft-continued without a plan (telemetry plan_skipped).
    plan_nudges: int = 0
    plan_skipped: bool = False
    # Pixel identity is the attachment identity; sources retain every URL alias
    # that produced those pixels without attaching the image more than once.
    image_views_by_id: dict[str, ImageView] = field(default_factory=dict)
    image_sources_by_id: dict[str, set[str]] = field(default_factory=dict)
    image_observations_by_id: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    delivered_image_ids: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class _ToolExecutionResult:
    tool_call_id: str
    tool_name: str
    content: str
    method: str = ""
    # Provenance harvested from an EXTERNAL tool call (never internal
    # record_findings/conclude, whose echoed rejection text would otherwise let
    # a hallucinated URL launder itself into the seen-set): the normalized URLs
    # the driver saw/requested this call, and the normalized result text the
    # warn-only quote check searches. Accumulated into loop state post-gather.
    provenance_urls: list[str] = field(default_factory=list)
    provenance_text: str = ""
    # Normalized URL -> verification tier this call established (W4). Only
    # populated for successful retrievals: a fetched-class call tiers the URLs it
    # actually retrieved (its arguments) "fetched"; a snippet-class call tiers
    # every URL it surfaced "snippet". Merged into state.url_best_tier (best-tier
    # wins) post-gather. Empty when the outcome granted no retrieval authority.
    provenance_tiers: dict[str, str] = field(default_factory=dict)
    image_views: list[ImageView] = field(default_factory=list)


def _get_field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _parse_response_message(response: Any) -> dict[str, Any]:
    choices = _get_field(response, "choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices[0].message")
    message = _get_field(choices[0], "message")
    if message is None:
        raise ValueError("LLM response missing choices[0].message")

    content = _get_field(message, "content")
    tool_calls_raw = _get_field(message, "tool_calls")
    tool_calls: list[dict[str, Any]] = []
    if tool_calls_raw:
        if not isinstance(tool_calls_raw, list):
            raise ValueError("assistant tool_calls was not a list")
        tool_calls = [_normalize_tool_call(entry, index) for index, entry in enumerate(tool_calls_raw)]

    assistant: dict[str, Any] = {"role": "assistant", "content": content if isinstance(content, str) else ""}
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    return assistant


def _normalize_tool_call(raw: Any, index: int) -> dict[str, Any]:
    function = _get_field(raw, "function")
    name = _get_field(function, "name")
    arguments = _get_field(function, "arguments")
    if not isinstance(name, str) or not name:
        raise ValueError(f"assistant tool call {index} missing function.name")
    call_id = _get_field(raw, "id")
    normalized: dict[str, Any] = {
        "id": call_id if isinstance(call_id, str) and call_id else f"tool_{index}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments if isinstance(arguments, str) else "{}",
        },
    }
    return normalized


def _extract_tool_calls(assistant_message: dict[str, Any]) -> list[_ToolCall]:
    raw_calls = assistant_message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    return [
        _ToolCall(
            id=str(entry["id"]),
            name=str(entry["function"]["name"]),
            arguments=str(entry["function"]["arguments"]),
        )
        for entry in raw_calls
    ]


def _remaining_s(state: _LoopState, now: Callable[[], float]) -> float:
    return max(0.0, state.deadline_at_s - now())


def _must_conclude(state: _LoopState, config: LoopConfig, now: Callable[[], float]) -> bool:
    """True once any of the three budgets leaves only the concluding turn.

    The step cap counts too: the final turn is reserved for ``conclude``, so a driver that
    researches until told to stop still concludes explicitly (gap accounting, ghost phase)
    rather than being cut off mid-research (Q14333 smoke, 2026-09-22).
    """
    return (
        _remaining_s(state, now) < config.conclude_threshold_s
        or state.telemetry.tool_calls >= config.max_tool_calls
        or state.telemetry.steps >= config.max_steps - 1
    )


def _plan_gaps_suffix(state: _LoopState) -> str:
    """The plan's gap ids for the budget line (W1), the work-list conclude's gap_accounting must cover (W2).

    A static list of every plan gap, not a live-shrinking count: findings carry no gap id,
    so the loop cannot know which gaps are done, and the driver tracks that itself.
    """
    if state.research_plan is None or not state.research_plan.gaps:
        return ""
    gap_ids = ", ".join(gap.id for gap in state.research_plan.gaps)
    return f" plan_gaps=[{gap_ids}]"


def _budget_line(state: _LoopState, config: LoopConfig, now: Callable[[], float]) -> str:
    remaining = int(_remaining_s(state, now))
    gaps = _plan_gaps_suffix(state)
    if _must_conclude(state, config, now):
        return f"\n[budget: {remaining}s remaining — you must conclude now{gaps}]"
    return (
        f"\n[budget: {remaining}s remaining, "
        f"{state.telemetry.tool_calls}/{config.max_tool_calls} tool calls used, "
        f"{state.telemetry.steps}/{config.max_steps} turns used{gaps}]"
    )
