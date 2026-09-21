"""Turning one assistant turn's tool calls into exactly one tool message each.

Three stages, in order. ADMISSION (``_admit_tool_calls``) applies the plan gate,
the call budget and duplicate detection, bumping the accepted calls' counters
before any handler runs. ABSORPTION (``_absorb_tool_results``) folds a batch's
provenance URLs, verification tiers and per-method counters into the loop state.
EMISSION (``_append_tool_messages``) writes one tool message per
``tool_call_id`` in the assistant's original order — anything else and the next
LLM turn 400s. The content builders for a refused, unknown or failed call live
here too, since a rejection has to render in exactly the shape a real outcome
does.

The per-CALL handler dispatch stays in ``loop.py``: it reaches the internal
``set_research_plan`` / ``record_findings`` / ``conclude`` handlers, which emit
the loop's own telemetry markers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from metaculus_bot.constants import GAP_FILL_V2_TOOL_BUDGET_LINE_RESERVE_CHARS
from metaculus_bot.research.agentic.gates import _PLAN_REQUIRED_NUDGE
from metaculus_bot.research.agentic.image_messages import image_reference_message, image_source_urls
from metaculus_bot.research.agentic.loop_state import (
    _LoopState,
    _ToolCall,
    _ToolExecutionResult,
)
from metaculus_bot.research.agentic.provenance import _TIER_RANK, _normalize_quote_text
from metaculus_bot.research.agentic.tool_schemas import _INTERNAL_TOOL_NAMES
from metaculus_bot.research.agentic.types import LoopConfig, ToolOutcome


def _truncate_content(content: str, max_chars: int) -> tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False
    marker = f"\n[truncated at {len(content)} chars]"
    clipped = content[: max(0, max_chars - len(marker))].rstrip()
    return f"{clipped}{marker}", True


def _tool_metadata_lines(tool_name: str, outcome: ToolOutcome, max_chars: int) -> list[str]:
    lines = [f"tool: {tool_name}", f"status: {outcome.status}"]
    if outcome.method:
        lines.append(f"method: {outcome.method}")
    if outcome.links:
        links = [f"- {link}" for link in outcome.links]
        candidate = [*lines, "links:"]
        metadata_limit = max_chars // 2
        for link in links:
            if len("\n".join([*candidate, link])) > metadata_limit:
                break
            candidate.append(link)
        if len(candidate) > len(lines) + 1:
            lines = candidate
    if outcome.truncated and "[truncated at " not in outcome.content_markdown:
        lines.append("truncated: true")
    return lines


def tool_content_body_budget(tool_name: str, outcome: ToolOutcome, max_chars: int) -> int:
    """Characters available for a body after the exact bounded metadata envelope."""
    metadata = "\n".join(_tool_metadata_lines(tool_name, outcome, max_chars))
    return max(0, max_chars - len(metadata) - 2 - GAP_FILL_V2_TOOL_BUDGET_LINE_RESERVE_CHARS)


def _format_tool_content(tool_name: str, outcome: ToolOutcome, max_chars: int) -> str:
    body_budget = tool_content_body_budget(tool_name, outcome, max_chars)
    body, truncated = _truncate_content(outcome.content_markdown, body_budget)
    effective = outcome.model_copy(update={"content_markdown": body, "truncated": outcome.truncated or truncated})
    lines = _tool_metadata_lines(tool_name, effective, max_chars)
    revised_budget = max(
        0,
        max_chars - len("\n".join(lines)) - 2 - GAP_FILL_V2_TOOL_BUDGET_LINE_RESERVE_CHARS,
    )
    if len(body) > revised_budget:
        body, _ = _truncate_content(outcome.content_markdown, revised_budget)
        effective = effective.model_copy(update={"content_markdown": body, "truncated": True})
        lines = _tool_metadata_lines(tool_name, effective, max_chars)
    if effective.content_markdown:
        lines.append("")
        lines.append(effective.content_markdown)
    return "\n".join(lines)[:max_chars]


def _with_budget_line(content: str, budget_line: str, max_chars: int) -> str:
    """Put the bounded budget status in the header, leaving continuation markers last."""
    budget_capacity = max(0, max_chars - len(content))
    bounded_budget_line = budget_line[: min(GAP_FILL_V2_TOOL_BUDGET_LINE_RESERVE_CHARS, budget_capacity)]
    header, separator, body = content.partition("\n\n")
    return f"{header}{bounded_budget_line}\n\n{body}" if separator else f"{content}{bounded_budget_line}"


def _parse_arguments(arguments: str) -> dict[str, Any]:
    if not arguments.strip():
        return {}
    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _tool_error_result(tool_call: _ToolCall, message: str, max_result_chars: int) -> _ToolExecutionResult:
    """Synthesize the error tool-response for a call that never reached its handler."""
    outcome = ToolOutcome(content_markdown=message, method="internal", status="error")
    return _ToolExecutionResult(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=_format_tool_content(tool_call.name, outcome, max_result_chars),
    )


def _normalized_call_key(tool_call: _ToolCall) -> tuple[str, str]:
    """(tool, normalized-args) identity for exact-duplicate detection.

    JSON args are re-serialized with sorted keys so key-order shuffles still
    count as the same call; unparseable args fall back to the raw string.
    """
    try:
        normalized = json.dumps(json.loads(tool_call.arguments or "{}"), sort_keys=True)
    except (json.JSONDecodeError, ValueError):
        normalized = tool_call.arguments
    return (tool_call.name, normalized)


_DUPLICATE_CALL_WARNING = (
    "\n[note: this exact tool call was already made earlier in this run — "
    "its result will not have changed. Vary the query/URL or move on.]"
)


def _budget_rejected_content(tool_name: str, config: LoopConfig) -> str:
    outcome = ToolOutcome(
        content_markdown=(
            f"Tool call rejected: the {config.max_tool_calls}-call research budget is exhausted. "
            "No further external tool calls will run — call conclude to finish, "
            "or record_findings to bank what you already have."
        ),
        method="internal",
        status="error",
    )
    return _format_tool_content(tool_name, outcome, config.max_result_chars)


def _plan_rejected_content(tool_name: str, config: LoopConfig) -> str:
    outcome = ToolOutcome(
        content_markdown=f"Tool call rejected: {_PLAN_REQUIRED_NUDGE}.",
        method="internal",
        status="error",
    )
    return _format_tool_content(tool_name, outcome, config.max_result_chars)


@dataclass
class _AdmittedCalls:
    """Which calls in one batch run, and why each of the rest was refused."""

    accepted: list[_ToolCall]
    duplicate_call_ids: set[str]
    rejected_call_ids: set[str]
    plan_rejected_call_ids: set[str]


def _admit_tool_calls(tool_calls: list[_ToolCall], state: _LoopState, config: LoopConfig) -> _AdmittedCalls:
    """Apply the plan gate, the call budget, and duplicate detection to one batch.

    Mutates ``state`` exactly as the inline version did — the accepted calls'
    counters are bumped here, before any handler runs, and the plan nudge is
    recorded once per gated batch.
    """
    admitted = _AdmittedCalls(
        accepted=[], duplicate_call_ids=set(), rejected_call_ids=set(), plan_rejected_call_ids=set()
    )

    # Plan gate (W1): external tool calls are rejected until set_research_plan
    # has run. Checked once per batch (before gather), so a parallel batch of
    # external calls emitted before any plan is all rejected together and counts
    # as a single nudge — a driver gets config.max_plan_nudges turns to plan,
    # after which the loop soft-continues (plan_skipped) rather than wedging.
    # Internal tools (set_research_plan/record_findings/conclude) are never
    # plan-gated, so the driver can always plan, bank, or finish.
    plan_gate_active = state.research_plan is None and not state.plan_skipped

    # Clamp the batch to the remaining call slots. With parallel_tool_calls a
    # single turn can emit more calls than budget allows; without this an
    # over-budget batch executes (and bills) every external call, overshooting
    # the max_tool_calls anytime ceiling. Internal bookkeeping tools
    # (record_findings/conclude) are never rejected so the driver can always
    # bank/finish. Rejected calls are NOT counted as executed, so
    # telemetry.tool_calls stays consistent with _must_conclude's gate.
    for tool_call in tool_calls:
        is_internal = tool_call.name in _INTERNAL_TOOL_NAMES
        if not is_internal and plan_gate_active:
            admitted.plan_rejected_call_ids.add(tool_call.id)
            continue
        if not is_internal and state.telemetry.tool_calls >= config.max_tool_calls:
            admitted.rejected_call_ids.add(tool_call.id)
            continue

        state.telemetry.tool_calls += 1
        state.telemetry.per_tool_counts[tool_call.name] = state.telemetry.per_tool_counts.get(tool_call.name, 0) + 1
        call_key = _normalized_call_key(tool_call)
        if call_key in state.seen_tool_calls:
            state.telemetry.dup_tool_calls += 1
            admitted.duplicate_call_ids.add(tool_call.id)
        else:
            state.seen_tool_calls.add(call_key)
        admitted.accepted.append(tool_call)

    # Record the plan nudge (once per gated batch) and flip to soft-continue
    # once the cap is hit, so the NEXT batch's external calls run un-gated.
    if admitted.plan_rejected_call_ids:
        state.plan_nudges += 1
        if state.plan_nudges >= config.max_plan_nudges:
            state.plan_skipped = True
            state.telemetry.plan_skipped = True
    return admitted


def _absorb_tool_results(state: _LoopState, results: Sequence[_ToolExecutionResult]) -> None:
    """Fold one batch's provenance, verification tiers, and counters into loop state."""
    provenance_texts: list[str] = []
    for result in results:
        if result.method == "rendered":
            state.telemetry.rendered_fetches += 1
        # Accumulate provenance so a LATER turn's record_findings/conclude can
        # verify a finding's source_url against what the driver actually
        # retrieved. Internal tools contribute nothing (see
        # provenance._harvest_provenance).
        state.tool_seen_urls.update(result.provenance_urls)
        # Merge per-call verification tiers, keeping the best tier seen per URL
        # (fetched outranks snippet) — a URL first seen via search then fetched
        # upgrades to "fetched" (W4).
        for url, tier in result.provenance_tiers.items():
            existing = state.url_best_tier.get(url)
            if existing is None or _TIER_RANK[tier] > _TIER_RANK[existing]:
                state.url_best_tier[url] = tier
        if result.provenance_text:
            provenance_texts.append(result.provenance_text)
        for image_view in result.image_views:
            observation = image_view.metadata()
            observations = state.image_observations_by_id.setdefault(image_view.image_id, [])
            if observation not in observations:
                observations.append(observation)
            existing_view = state.image_views_by_id.get(image_view.image_id)
            if existing_view is None:
                state.image_views_by_id[image_view.image_id] = image_view
            else:
                parent_page_urls = tuple(dict.fromkeys((*existing_view.parent_page_urls, *image_view.parent_page_urls)))
                state.image_views_by_id[image_view.image_id] = replace(existing_view, parent_page_urls=parent_page_urls)
            state.image_sources_by_id.setdefault(image_view.image_id, set()).update(image_source_urls(image_view))
    if provenance_texts:
        state.tool_content_normalized = _normalize_quote_text(
            f"{state.tool_content_normalized} {' '.join(provenance_texts)}"
        )


def _append_tool_messages(
    tool_calls: list[_ToolCall],
    admitted: _AdmittedCalls,
    results: Sequence[_ToolExecutionResult],
    *,
    state: _LoopState,
    config: LoopConfig,
    budget_line: str,
) -> None:
    """Emit exactly one tool message per tool_call_id, in the assistant's original order.

    Anything else and the next LLM turn 400s. Rejected calls get a synthetic
    plan-gate / budget-exhausted error response. This is also the only stage that
    can see an outcome, so it owns the one duplicate-detection exemption the
    admission stage cannot make: a throttled fetch's key is evicted from
    ``state.seen_tool_calls`` here (see the inline comment below).
    """
    results_by_id = {result.tool_call_id: result for result in results}
    delivered_image_ids: list[str] = []
    for tool_call in tool_calls:
        if tool_call.id in admitted.plan_rejected_call_ids:
            content = _plan_rejected_content(tool_call.name, config)
        elif tool_call.id in admitted.rejected_call_ids:
            content = _budget_rejected_content(tool_call.name, config)
        else:
            result = results_by_id[tool_call.id]
            # Forget a throttled call so its retry isn't called a duplicate. The
            # throttle outcome is deliberately not cached and its message tells the
            # driver to fetch the same URL again later in the run
            # (tools._throttled_fetch_outcome), so _DUPLICATE_CALL_WARNING's "its
            # result will not have changed. Vary the query/URL or move on." is false
            # exactly here and steers the driver off the URL. Advisory bookkeeping
            # only: max_tool_calls still caps a throttle spin, a re-throttled retry
            # re-registers at admission and is evicted again here, and a retry that
            # succeeds leaves its key in place so a THIRD identical call is still
            # warned.
            if result.method == "throttled":
                state.seen_tool_calls.discard(_normalized_call_key(tool_call))
            content = result.content + (_DUPLICATE_CALL_WARNING if tool_call.id in admitted.duplicate_call_ids else "")
        state.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": _with_budget_line(content, budget_line, config.max_result_chars),
            }
        )
        if tool_call.id not in admitted.rejected_call_ids and tool_call.id not in admitted.plan_rejected_call_ids:
            result = results_by_id[tool_call.id]
            for image_view in result.image_views:
                if (
                    image_view.image_id not in state.delivered_image_ids
                    and image_view.image_id not in delivered_image_ids
                ):
                    delivered_image_ids.append(image_view.image_id)
    if delivered_image_ids:
        state.messages.append(image_reference_message(delivered_image_ids))
        state.delivered_image_ids.update(delivered_image_ids)
