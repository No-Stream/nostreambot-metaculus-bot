"""The gap-fill v2 driver loop: the turn loop, its internal tool handlers, and the ghost phase.

``run_agentic_loop`` is the entry point and ``_run_loop_body`` the turn loop;
between them they own message management, the soft-fail/timeout wrapper, and the
``GAP_FILL_V2`` completion marker. The cohesive subsystems the loop drives live
in siblings — ``tool_schemas`` (the advertised tool list), ``loop_state`` (state,
per-turn records, budget), ``provenance`` (URL/quote grounding + tiers),
``gates`` (W1-W4), ``dispatch`` (batch admission, absorption, tool messages) and
``artifact`` (the rendered findings section).

What stays here rather than moving out is load-bearing, not leftover:

* The three internal tool handlers (``set_research_plan`` / ``record_findings`` /
  ``conclude``) and ``_validate_findings_payload``, because they log this
  module's telemetry markers — ``GHOST_PRE`` / ``GHOST_PRE_JSON`` and the
  ``GAP_FILL_V2 quote_mismatch`` WARN are grepped and asserted on the
  ``...agentic.loop`` logger, so relocating them would silently re-key them.
* The per-CALL handler dispatch, which reaches those handlers (moving it to
  ``dispatch`` would close an import cycle).
* The ghost phase and ``_summarize_ghost``, whose ``parse_structured_block``
  binding is a monkeypatch surface read through this module's globals.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from collections.abc import Callable
from typing import Any, Literal, get_args

from pydantic import ValidationError

from metaculus_bot.numeric.date_axis import format_epoch, to_epoch
from metaculus_bot.question_types import QuestionType
from metaculus_bot.research.agentic.artifact import detachment_lint, render_findings
from metaculus_bot.research.agentic.dispatch import (
    _absorb_tool_results,
    _admit_tool_calls,
    _append_tool_messages,
    _format_tool_content,
    _parse_arguments,
    _tool_error_result,
)
from metaculus_bot.research.agentic.gates import (
    _PLAN_REQUIRED_NUDGE,
    _actions_cite_fetch,  # noqa: F401  # re-export: tests/test_agentic_loop.py imports it from this module
    _apply_findings_telemetry,
    _bank_findings,
    _check_image_provenance,
    _check_url_provenance,
    _coerce_pending_leads,
    _coerce_planned_gaps,
    _evaluate_conclude_gate,
    _FindingsValidation,
)
from metaculus_bot.research.agentic.image_messages import bind_image_materialization
from metaculus_bot.research.agentic.llm import LlmCall, build_default_llm_call
from metaculus_bot.research.agentic.loop_state import (
    _budget_line,
    _extract_tool_calls,
    _LoopState,
    _must_conclude,
    _parse_response_message,
    _ToolCall,
    _ToolExecutionResult,
)
from metaculus_bot.research.agentic.provenance import (
    _SPAN_JOINER_MAX_CHARS,  # noqa: F401  # re-export: tests/test_agentic_gates.py imports it from this module
    _harvest_provenance,
    _harvest_verification_tiers,
    _iter_normalized_urls,
    _method_to_tier,  # noqa: F401  # re-export: tests/test_agentic_tools.py imports it from this module
    _normalize_quote_text,  # noqa: F401  # re-export: tests/test_agentic_gates.py imports it from this module
    _normalize_url,  # noqa: F401  # re-export: tests/test_agentic_loop.py imports it from this module
    _quote_is_grounded,
)
from metaculus_bot.research.agentic.tool_schemas import (
    _INTERNAL_TOOL_NAMES,
    _INTERNAL_TOOL_TIMEOUT_S,
    _tool_schemas,
)
from metaculus_bot.research.agentic.types import (
    Finding,
    GhostContext,
    GhostForecast,
    LoopConfig,
    LoopResult,
    ResearchPlan,
    ToolOutcome,
    ToolSpec,
)
from metaculus_bot.structured_output_schema import (
    BinaryStructured,
    DateStructured,
    MultipleChoiceStructured,
    NumericStructured,
    extract_json_block,
    parse_structured_block,
)

logger = logging.getLogger(__name__)

_NUDGE = "call conclude or use tools"

# The two ghost markers; the v1 ghost's pair mirrors the plain ghost's shapes so one scorer reads both.
_GHOST_MARKER = "GHOST_FORECAST"
_GHOST_V1_MARKER = "GHOST_FORECAST_V1"
_GHOST_TIMEOUT_S = 60.0


def _validate_findings_payload(
    raw_findings: Any,
    state: _LoopState,
    *,
    label: Literal["findings", "final_findings"],
) -> _FindingsValidation:
    if raw_findings is None:
        return _FindingsValidation([], [], 0, 0, 0)
    if not isinstance(raw_findings, list):
        return _FindingsValidation([], [f"{label} must be a list"], 0, 0, 0)

    accepted: list[Finding] = []
    rejected: list[str] = []
    lint_rejections = 0
    provenance_rejections = 0
    quote_mismatch_warnings = 0
    for index, raw_finding in enumerate(raw_findings):
        try:
            finding = Finding.model_validate(raw_finding)
        except ValidationError as exc:
            rejected.append(f"{label}[{index}] invalid: {exc.errors()[0]['msg']}")
            continue
        violations = detachment_lint(finding)
        if violations:
            lint_rejections += 1
            rejected.append(f"{label}[{index}] rejected: {'; '.join(violations)}")
            continue
        image_provenance_reason = _check_image_provenance(finding, state)
        if image_provenance_reason is not None:
            provenance_rejections += 1
            rejected.append(f"{label}[{index}] rejected: {image_provenance_reason}")
            continue
        provenance_reason = _check_url_provenance(finding, state) if finding.evidence_kind == "text" else None
        if provenance_reason is not None:
            provenance_rejections += 1
            rejected.append(f"{label}[{index}] rejected: {provenance_reason}")
            continue
        # Warn-only and deduped per run: see docs/agentic_gap_fill.md "The findings gates".
        if finding.evidence_kind == "text" and not _quote_is_grounded(finding.quote, state.tool_content_normalized):
            warned_key = (finding.source_url, finding.quote)
            if warned_key not in state.warned_quote_keys:
                state.warned_quote_keys.add(warned_key)
                quote_mismatch_warnings += 1
                logger.warning(
                    "GAP_FILL_V2 quote_mismatch: source_url=%s quote=%r not found verbatim in tool contents",
                    finding.source_url,
                    finding.quote,
                )
        accepted.append(finding)
    return _FindingsValidation(accepted, rejected, lint_rejections, provenance_rejections, quote_mismatch_warnings)


async def _record_findings_tool(state: _LoopState, arguments: dict[str, Any]) -> ToolOutcome:
    validation = _validate_findings_payload(arguments.get("findings"), state, label="findings")
    banked, duplicates = _bank_findings(state, validation.accepted)
    _apply_findings_telemetry(state, validation)

    lines = [f"Recorded {banked} finding(s)."]
    if duplicates:
        lines.append(f"Skipped {duplicates} finding(s) already recorded earlier in this run.")
    if validation.rejected:
        lines.append("Rejected:")
        lines.extend(f"- {item}" for item in validation.rejected)
    return ToolOutcome(content_markdown="\n".join(lines), method="internal")


def _summarize_dry_run(dry_run_forecast: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize the plan's dry-run forecast dict into the GHOST_FORECAST_JSON payload
    shape, reusing the ghost parser so GHOST_PRE_JSON and GHOST_FORECAST_JSON are
    directly comparable. Returns ``None`` when absent or unparseable (the JSON
    marker line is then suppressed, same as the ghost path)."""
    if not isinstance(dry_run_forecast, dict):
        return None
    # Fenced so the tested _summarize_ghost path parses it identically to a real ghost.
    raw_text = f"```json\n{json.dumps(dry_run_forecast, separators=(',', ':'))}\n```"
    _qtype, _summary, forecast = _summarize_ghost(raw_text)
    return forecast


async def _set_research_plan_tool(state: _LoopState, arguments: dict[str, Any], config: LoopConfig) -> ToolOutcome:
    """Register the driver's turn-one research plan (W1).

    Stores ``state.research_plan`` (W2 reads its gaps for conclude-time
    accounting) and emits the GHOST_PRE / GHOST_PRE_JSON telemetry — the pre-
    research counterpart to the concluding ghost, so the pre/post delta measures
    whether v2's research moved its own view.
    """
    gaps, gap_issues = _coerce_planned_gaps(arguments.get("gaps"), max_gaps=config.max_gaps)
    # Rejected, never stored, so the W1 and W2 gates stay armed: docs/agentic_gap_fill.md "The findings gates".
    if not gaps:
        notes = "; ".join(gap_issues) if gap_issues else "provide at least one ranked research gap"
        return ToolOutcome(
            content_markdown=f"Research plan rejected: {_PLAN_REQUIRED_NUDGE} ({notes}).",
            method="internal",
            status="error",
        )

    sensitive = arguments.get("sensitive_assumptions")
    sensitive_assumptions = [item for item in sensitive if isinstance(item, str)] if isinstance(sensitive, list) else []
    raw_dry_run_forecast = arguments.get("dry_run_forecast")
    dry_run_forecast = raw_dry_run_forecast if isinstance(raw_dry_run_forecast, dict) else None

    state.research_plan = ResearchPlan(
        dry_run_forecast=dry_run_forecast,
        sensitive_assumptions=sensitive_assumptions,
        gaps=gaps,
    )
    state.telemetry.plan_gaps = len(gaps)

    forecast = _summarize_dry_run(dry_run_forecast)
    logger.info(
        "%sGHOST_PRE: gaps=%s sensitive_assumptions=%s",
        state.log_prefix,
        len(gaps),
        len(sensitive_assumptions),
    )
    if forecast is not None:
        logger.info("%sGHOST_PRE_JSON: %s", state.log_prefix, json.dumps(forecast, separators=(",", ":")))
    elif raw_dry_run_forecast is not None:
        # Non-random loss, not just a suppressed line: see docs/agentic_gap_fill.md on the ghost forecast.
        logger.warning(
            "%sGHOST_PRE_JSON suppressed: dry_run_forecast did not parse into a structured forecast; "
            "this question's ghost pair will have no pre-research half",
            state.log_prefix,
        )

    lines = [f"Research plan set: {len(gaps)} gap(s), {len(sensitive_assumptions)} sensitive assumption(s)."]
    if gap_issues:
        lines.append("Notes:")
        lines.extend(f"- {item}" for item in gap_issues)
    return ToolOutcome(content_markdown="\n".join(lines), method="internal")


async def _conclude_tool(
    state: _LoopState, arguments: dict[str, Any], config: LoopConfig, now: Callable[[], float]
) -> ToolOutcome:
    gate_block = _evaluate_conclude_gate(state, arguments, config, now)
    if gate_block is not None:
        return gate_block

    validation = _validate_findings_payload(arguments.get("final_findings"), state, label="final_findings")
    pending_leads, pending_errors = _coerce_pending_leads(arguments.get("pending_leads"))
    banked, duplicates = _bank_findings(state, validation.accepted)
    state.pending_leads = pending_leads
    _apply_findings_telemetry(state, validation)
    state.explicit_conclude = True
    state.stop_loop = True

    lines = [f"Concluded with {banked} final finding(s) and {len(pending_leads)} pending lead(s)."]
    if duplicates:
        lines.append(f"Skipped {duplicates} final finding(s) already recorded earlier in this run.")
    if validation.rejected or pending_errors:
        lines.append("Rejected:")
        lines.extend(f"- {item}" for item in [*validation.rejected, *pending_errors])
    return ToolOutcome(content_markdown="\n".join(lines), method="internal")


async def _run_tool_handler(
    tool_call: _ToolCall,
    arguments: dict[str, Any],
    timeout_s: float,
    *,
    tools_by_name: dict[str, ToolSpec],
    state: _LoopState,
    config: LoopConfig,
    now: Callable[[], float],
) -> ToolOutcome:
    """Dispatch one tool call under its timeout, folding every failure into an outcome."""
    try:
        # Bind inside the boundary: a bad argument becomes an error outcome, not a batch abort.
        if tool_call.name == "set_research_plan":
            handler = _set_research_plan_tool(state, arguments, config)
        elif tool_call.name == "record_findings":
            handler = _record_findings_tool(state, arguments)
        elif tool_call.name == "conclude":
            handler = _conclude_tool(state, arguments, config, now)
        else:
            handler = tools_by_name[tool_call.name].handler(**arguments)
        raw_outcome = await asyncio.wait_for(handler, timeout=timeout_s)
        return ToolOutcome.model_validate(raw_outcome)
    except TimeoutError:
        return ToolOutcome(
            content_markdown=f"Tool timed out after {timeout_s:.2f}s.",
            method="internal",
            status="timeout",
        )
    except ValidationError as exc:
        return ToolOutcome(
            content_markdown=f"Tool returned invalid outcome: {exc.errors()[0]['msg']}",
            method="internal",
            status="error",
        )
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # tool-execution boundary: any tool failure becomes an error outcome, never a loop crash
        return ToolOutcome(
            content_markdown=f"{type(exc).__name__}: {exc}",
            method="internal",
            status="error",
        )


async def _execute_one_tool_call(
    tool_call: _ToolCall,
    tools_by_name: dict[str, ToolSpec],
    state: _LoopState,
    config: LoopConfig,
    *,
    now: Callable[[], float],
) -> _ToolExecutionResult:
    try:
        arguments = _parse_arguments(tool_call.arguments)
    except (json.JSONDecodeError, ValueError) as exc:
        return _tool_error_result(tool_call, f"Invalid tool arguments: {exc}", config.max_result_chars)

    if tool_call.name in _INTERNAL_TOOL_NAMES:
        timeout_s = _INTERNAL_TOOL_TIMEOUT_S
    else:
        spec = tools_by_name.get(tool_call.name)
        if spec is None:
            return _tool_error_result(tool_call, f"Unknown tool: {tool_call.name}", config.max_result_chars)
        timeout_s = spec.timeout_s

    outcome = await _run_tool_handler(
        tool_call,
        arguments,
        timeout_s,
        tools_by_name=tools_by_name,
        state=state,
        config=config,
        now=now,
    )

    provenance_urls, provenance_text = _harvest_provenance(tool_call.name, arguments, outcome)
    provenance_tiers = _harvest_verification_tiers(tool_call.name, arguments, outcome)
    return _ToolExecutionResult(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=_format_tool_content(tool_call.name, outcome, config.max_result_chars),
        method=outcome.method,
        provenance_urls=provenance_urls,
        provenance_text=provenance_text,
        provenance_tiers=provenance_tiers,
        image_views=outcome.image_views,
    )


async def _execute_tool_batch(
    tool_calls: list[_ToolCall],
    *,
    tools_by_name: dict[str, ToolSpec],
    state: _LoopState,
    config: LoopConfig,
    now: Callable[[], float],
) -> None:
    admitted = _admit_tool_calls(tool_calls, state, config)
    results = await asyncio.gather(
        *[_execute_one_tool_call(tool_call, tools_by_name, state, config, now=now) for tool_call in admitted.accepted]
    )
    _absorb_tool_results(state, results)
    _append_tool_messages(
        tool_calls,
        admitted,
        results,
        state=state,
        config=config,
        budget_line=_budget_line(state, config, now),
    )


def _freeze_result(
    state: _LoopState,
    findings_markdown: str,
    ghost: GhostForecast | None,
    ghost_context: GhostContext | None = None,
) -> LoopResult:
    state.telemetry.findings_count = len(state.findings)
    state.telemetry.pending_leads_count = len(state.pending_leads)
    state.telemetry.concluded_early = state.explicit_conclude and not state.telemetry.deadline_hit
    state.telemetry.wall_s = max(0.0, state.telemetry.wall_s)
    return LoopResult(
        findings_markdown=findings_markdown,
        ghost=ghost,
        telemetry=state.telemetry,
        transcript=copy.deepcopy(state.messages),
        ghost_context=ghost_context,
        image_views=list(state.image_views_by_id.values()),
        image_sources={image_id: sorted(sources) for image_id, sources in state.image_sources_by_id.items()},
        image_observations=copy.deepcopy(state.image_observations_by_id),
    )


def _log_completion(state: _LoopState, log_prefix: str) -> None:
    """Emit this run's single GAP_FILL_V2 marker line (fields: docs/agentic_gap_fill.md "Telemetry")."""
    per_tool = state.telemetry.per_tool_counts
    searches = per_tool.get("search_news", 0) + per_tool.get("search_web", 0)
    logger.info(
        "%sGAP_FILL_V2: model=%s steps=%s tool_calls=%s searches=%s fetches=%s rendered=%s reads=%s "
        "dup_tool_calls=%s deadline_hit=%s concluded_early=%s wall_s=%.2f findings=%s "
        "pending_leads=%s lint_rejections=%s provenance_rejections=%s quote_mismatch_warnings=%s "
        "plan_gaps=%s plan_skipped=%s conclude_gate_rejections=%s error=%s",
        log_prefix,
        state.telemetry.model,
        state.telemetry.steps,
        state.telemetry.tool_calls,
        searches,
        per_tool.get("fetch", 0),
        state.telemetry.rendered_fetches,
        per_tool.get("read_document", 0),
        state.telemetry.dup_tool_calls,
        state.telemetry.deadline_hit,
        state.explicit_conclude and not state.telemetry.deadline_hit,
        state.telemetry.wall_s,
        len(state.findings),
        len(state.pending_leads),
        state.telemetry.lint_rejections,
        state.telemetry.provenance_rejections,
        state.telemetry.quote_mismatch_warnings,
        state.telemetry.plan_gaps,
        state.telemetry.plan_skipped,
        state.telemetry.conclude_gate_rejections,
        state.telemetry.error,
    )


# Derived, not restated: a question type added to the shared Literal reaches the ghost parser too.
_GHOST_QTYPES: tuple[QuestionType, ...] = get_args(QuestionType)


def _declared_qtype(raw_text: str) -> QuestionType | None:
    """Peek the ghost block's self-declared ``question_type`` without parsing it.

    The ghost emits exactly one structured block that names its own
    ``question_type`` (the schema discriminator). Reading it lets
    ``_summarize_ghost`` parse only the matching type instead of trying all
    three — the two non-matching attempts would otherwise trip
    ``parse_structured_payload``'s ``question_type`` mismatch guard, which WARNs
    by construction (a numeric ghost logged 2 such WARNs, an MC ghost 1). Returns
    ``None`` when no block is present, the JSON is malformed, or the declared
    type is missing/unsupported, so the caller can fall back to trying every type.
    """
    raw = extract_json_block(raw_text)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    declared = payload.get("question_type")
    return declared if declared in _GHOST_QTYPES else None


def _summarize_ghost(raw_text: str) -> tuple[str, str, dict[str, Any] | None]:
    """Parse the ghost's structured block once into ``(qtype, legacy_summary, forecast)``.

    ``legacy_summary`` is the lossy human-readable string kept for the
    back-compat ``GHOST_FORECAST`` marker. ``forecast`` is the full,
    deterministically-parsable payload for the ``GHOST_FORECAST_JSON`` marker:
    the posterior probability (binary), the complete option->prob dict (MC), or
    the complete percentile->value dict plus the median (numeric). ``None`` when
    no block parses — the JSON marker line is then suppressed and only the
    legacy ``unknown``/``""`` marker is emitted.

    Only the block's self-declared ``question_type`` is parsed (see
    ``_declared_qtype``); the all-types fallback runs only when no type could be
    read, keeping the mismatch-WARN path off the normal ghost flow.
    """
    declared = _declared_qtype(raw_text)
    candidate_qtypes = (declared,) if declared is not None else _GHOST_QTYPES
    for qtype in candidate_qtypes:
        block = parse_structured_block(raw_text, qtype)
        if isinstance(block, BinaryStructured):
            prob = float(block.posterior_prob)
            return "binary", f"posterior_prob={prob:.4f}", {"qtype": "binary", "prob": prob}
        if isinstance(block, MultipleChoiceStructured):
            option_probs = {name: float(prob) for name, prob in block.option_probs.items()}
            summary = ", ".join(f"{name}={prob:.3f}" for name, prob in sorted(option_probs.items()))
            return "multiple_choice", summary, {"qtype": "multiple_choice", "option_probs": option_probs}
        if isinstance(block, NumericStructured) and block.declared_percentiles:
            declared = {float(pct): float(value) for pct, value in block.declared_percentiles.items()}
            median = declared.get(0.5)
            summary = "" if median is None else f"median={median}"
            return "numeric", summary, {"qtype": "numeric", "declared_percentiles": declared, "median": median}
        if isinstance(block, DateStructured):
            # Epoch seconds, the axis the date pipeline forecasts on, so the ghost scores like a published CDF.
            declared = {float(pct): to_epoch(moment) for pct, moment in block.declared_percentiles.items()}
            median = declared.get(0.5)
            summary = "" if median is None else f"median={format_epoch(median, '')}"
            return "date", summary, {"qtype": "date", "declared_percentiles": declared, "median": median}
    return "unknown", "", None


async def _run_ghost_phase(
    *,
    state: _LoopState,
    ghost_prompt: str,
    tools_json: list[dict[str, Any]],
    llm_call: LlmCall,
    log_prefix: str,
) -> GhostForecast | None:
    """Log the driver's post-research dry-run forecast (telemetry only, never published).

    Interpretation guardrail (FUTURE.md "Score the archived gap-fill v2 ghost
    forecasts"): the ghost is a SAME-MODEL (terra-low driver) counterfactual, NOT a
    panel proxy. It measures whether the v2 findings alone, forecast by one cheap
    model, land near truth — scored OFFLINE against resolution by
    ``scripts/score_ghosts.py``. It genuinely diverges from the published ensemble
    median (measured 2026-08-24 on 39 triple-era binaries: |ghost - panel| median
    8 pp, non-zero on 37 of 39) but with NO systematic confidence direction — the
    ghost-minus-panel confidence delta was -3.13 pp mean, CI95 [-6.37, +0.25], so the
    old "a single low-effort model is over-decisive by construction" prediction is
    unsupported and a negative ghost delta must not be explained away as expected
    over-decisiveness. Either way, ghost-vs-published divergence is NOT an alarm
    signal and must not gate a run — score it against resolution, not the panel.
    """
    state.messages.append({"role": "user", "content": ghost_prompt})
    assistant_message = await _complete_ghost(state.messages, tools_json, llm_call, log_prefix, phase="Ghost phase")
    if assistant_message is None:
        return None
    state.messages.append(assistant_message)
    return _log_ghost(assistant_message, log_prefix, _GHOST_MARKER)


async def _complete_ghost(
    messages: list[dict[str, Any]],
    tools_json: list[dict[str, Any]],
    llm_call: LlmCall,
    log_prefix: str,
    *,
    phase: str,
) -> dict[str, Any] | None:
    """One tool-forbidden ghost completion on the research turns' tool list, or None on timeout or failure."""
    try:
        # Same tool list as the last research turn, tools forbidden via tool_choice: the cached prefix keeps matching.
        response = await asyncio.wait_for(llm_call(messages, tools_json, tool_choice="none"), timeout=_GHOST_TIMEOUT_S)
        return _parse_response_message(response)
    except TimeoutError:
        logger.warning("%s%s timed out after %.0fs", log_prefix, phase, _GHOST_TIMEOUT_S)
        return None
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # telemetry-only phase: a failed ghost must never cost the run its banked findings
        logger.warning("%s%s failed: %s: %s", log_prefix, phase, type(exc).__name__, exc)
        return None


def _log_ghost(assistant_message: dict[str, Any], log_prefix: str, marker: str) -> GhostForecast:
    """Parse a ghost answer once and log its lossy summary line plus the additive full-fidelity JSON companion."""
    raw_text = assistant_message["content"]
    qtype, parsed_summary, forecast = _summarize_ghost(raw_text)
    ghost = GhostForecast(qtype=qtype, raw_text=raw_text, parsed_summary=parsed_summary)
    logger.info("%s%s: qtype=%s summary=%s", log_prefix, marker, ghost.qtype, ghost.parsed_summary)
    if forecast is not None:
        logger.info("%s%s_JSON: %s", log_prefix, marker, json.dumps(forecast, separators=(",", ":")))
    return ghost


async def run_ghost_v1(context: GhostContext, ghost_prompt: str, *, log_prefix: str = "") -> GhostForecast | None:
    """The v1 ghost: the plain ghost's brief plus gap-fill v1's section, branched off the pre-ghost transcript.

    Sends the loop's transcript up to, not including, the plain ghost's prompt, so the driver
    answers without seeing its own first ghost, on the same tool list with ``tool_choice="none"``
    so the cached prefix keeps matching. Telemetry only, never published, never raises; logs
    ``GHOST_FORECAST_V1`` and ``GHOST_FORECAST_V1_JSON`` in the plain ghost's shapes. What the
    pair measures: docs/agentic_gap_fill.md "The ghost forecast".
    """
    messages = [*context.messages, {"role": "user", "content": ghost_prompt}]
    assistant_message = await _complete_ghost(
        messages, context.tools_json, context.llm_call, log_prefix, phase="v1 ghost phase"
    )
    if assistant_message is None:
        return None
    return _log_ghost(assistant_message, log_prefix, _GHOST_V1_MARKER)


async def _run_loop_body(
    *,
    state: _LoopState,
    tools: list[ToolSpec],
    config: LoopConfig,
    llm_call: LlmCall,
    ghost_prompt: str | None,
    log_prefix: str,
    now: Callable[[], float],
) -> LoopResult:
    tools_by_name = {tool.name: tool for tool in tools}
    offered_tools: list[dict[str, Any]] = []

    while not state.stop_loop and state.telemetry.steps < config.max_steps:
        offered_tools = _tool_schemas(tools, _must_conclude(state, config, now))
        response = await llm_call(state.messages, offered_tools)
        assistant_message = _parse_response_message(response)
        state.messages.append(assistant_message)
        state.telemetry.steps += 1

        tool_calls = _extract_tool_calls(assistant_message)
        if tool_calls:
            await _execute_tool_batch(tool_calls, tools_by_name=tools_by_name, state=state, config=config, now=now)
            continue

        if state.nudged_for_no_action:
            state.stop_loop = True
            break

        state.messages.append({"role": "user", "content": _NUDGE})
        state.nudged_for_no_action = True

    state.telemetry.wall_s = now() - state.started_at_s
    findings_markdown = render_findings(state.findings, state.pending_leads)
    ghost: GhostForecast | None = None
    ghost_context: GhostContext | None = None
    if ghost_prompt is not None and state.explicit_conclude:
        # Captured before the ghost turn, so a second ghost branches off the same cached prefix without seeing the first.
        ghost_context = GhostContext(
            messages=copy.deepcopy(state.messages), tools_json=offered_tools, llm_call=llm_call
        )
        ghost = await _run_ghost_phase(
            state=state, ghost_prompt=ghost_prompt, tools_json=offered_tools, llm_call=llm_call, log_prefix=log_prefix
        )

    _log_completion(state, log_prefix)
    return _freeze_result(state, findings_markdown, ghost, ghost_context if ghost is not None else None)


# Jitter allowance for the deadline-versus-inner-timeout test below; see docs/agentic_gap_fill.md.
_DEADLINE_SLOP_S = 0.5


def _finalize_loop_exit(state: _LoopState, now_fn: Callable[[], float], log_prefix: str) -> LoopResult:
    """Shared soft-fail tail for the deadline-hit and crash exits of the outer
    ``wait_for``: stamp elapsed wall time, render whatever findings were banked,
    emit the completion marker, and freeze. Callers set ``deadline_hit`` /
    ``error`` on ``state.telemetry`` before calling."""
    state.telemetry.wall_s = now_fn() - state.started_at_s
    findings_markdown = render_findings(state.findings, state.pending_leads)
    _log_completion(state, log_prefix)
    return _freeze_result(state, findings_markdown, None)


async def run_agentic_loop(
    system_prompt: str,
    user_brief: str,
    tools: list[ToolSpec],
    config: LoopConfig,
    *,
    llm_call: LlmCall | None = None,
    ghost_prompt: str | None = None,
    log_prefix: str = "",
    now: Callable[[], float] | None = None,
) -> LoopResult:
    now_fn = now or time.monotonic
    base_call = llm_call or build_default_llm_call(config)
    state = _LoopState(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_brief},
        ],
        started_at_s=now_fn(),
        deadline_at_s=now_fn() + config.wall_deadline_s,
        log_prefix=log_prefix,
    )
    state.telemetry.model = config.model
    call = bind_image_materialization(base_call, state.image_views_by_id, state.image_sources_by_id)
    # Briefing URLs ground non-discrepancy findings only: docs/agentic_gap_fill.md "The findings gates".
    state.briefing_urls = set(_iter_normalized_urls(user_brief))

    try:
        return await asyncio.wait_for(
            _run_loop_body(
                state=state,
                tools=tools,
                config=config,
                llm_call=call,
                ghost_prompt=ghost_prompt,
                log_prefix=log_prefix,
                now=now_fn,
            ),
            timeout=config.wall_deadline_s,
        )
    except asyncio.CancelledError:
        raise
    except TimeoutError as exc:
        # Classify by elapsed wall time: an inner connection timeout lands in this except too.
        elapsed = now_fn() - state.started_at_s
        if elapsed >= config.wall_deadline_s - _DEADLINE_SLOP_S:
            state.telemetry.deadline_hit = True
        else:
            logger.exception("%sAgentic loop hit a non-deadline TimeoutError; soft-failing", log_prefix)
            # Newline-sanitize: the marker regex captures error= to end-of-line.
            state.telemetry.error = repr(exc).replace("\n", " ")
        return _finalize_loop_exit(state, now_fn, log_prefix)
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # sanctioned package boundary: mirror v1 soft-fail contract and never raise past the harness except on cancellation
        logger.exception("%sAgentic loop failed; soft-failing to banked findings if any", log_prefix)
        # error= is the only field separating a crash from an idle run; sanitize for the marker regex.
        state.telemetry.error = repr(exc).replace("\n", " ")
        return _finalize_loop_exit(state, now_fn, log_prefix)


__all__ = ["LlmCall", "run_agentic_loop", "run_ghost_v1"]
