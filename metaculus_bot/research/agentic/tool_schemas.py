"""The tool protocol the gap-fill v2 driver sees each turn.

Two things live here. First, the tools the LOOP implements itself —
``set_research_plan`` / ``record_findings`` / ``conclude`` — named in
``_INTERNAL_TOOL_NAMES`` with their own short timeout: they never leave the
process, never count against the research budget, and never contribute
provenance. That tuple is the single authority for the membership test, which
provenance harvesting, the per-gap research floor and batch admission all key
off; a second copy would drift. Second, the JSON-schema builders for the tool
list advertised on every LLM turn (internal tools always, the research tools
only until the driver must conclude).

``_tool_schemas`` is a monkeypatch surface for the loop's soft-fail test, so
``loop.py`` imports it and calls it through its own module global.
"""

from __future__ import annotations

from typing import Any

from metaculus_bot.research.agentic.types import ToolSpec

_INTERNAL_TOOL_TIMEOUT_S = 5.0
_INTERNAL_TOOL_NAMES = ("set_research_plan", "record_findings", "conclude")


def _tool_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _internal_tool_schemas() -> list[dict[str, Any]]:
    finding_schema = {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "source_url": {"type": "string"},
                        "quote": {"type": "string"},
                        "evidence_kind": {"type": "string", "enum": ["text", "image"], "default": "text"},
                        "image_id": {
                            "type": "string",
                            "description": "For image evidence, the image_id delivered by a preceding tool result.",
                        },
                        "visual_observation": {
                            "type": "string",
                            "description": (
                                "For image evidence, state exactly what you visually observed and label numerical "
                                "values as transcribed or estimated. This records your interpretation; delivery "
                                "provenance does not machine-verify the observation."
                            ),
                        },
                        "date": {"type": "string"},
                        "retrieved_how": {"type": "string"},
                        "topic": {"type": "string"},
                        "discrepancy": {"type": "boolean"},
                        "derivation": {
                            "type": "string",
                            "description": (
                                "OPTIONAL arithmetic-only synthesis over THIS finding's quoted numbers "
                                "(a derived table, bound, or rate). Every input number must appear as a "
                                "quoted value with URL in this finding's quote/source. Arithmetic and its "
                                "result only — no likelihood language, no new facts."
                            ),
                        },
                    },
                    "required": ["claim", "source_url"],
                    "allOf": [
                        {
                            "if": {"properties": {"evidence_kind": {"const": "image"}}, "required": ["evidence_kind"]},
                            "then": {"required": ["image_id", "visual_observation"]},
                            "else": {"required": ["quote"]},
                        }
                    ],
                    "additionalProperties": True,
                },
            }
        },
        "required": ["findings"],
        "additionalProperties": False,
    }
    conclude_schema = {
        "type": "object",
        "properties": {
            "pending_leads": {"type": "array", "items": {"type": "string"}},
            "final_findings": {
                "type": "array",
                "items": finding_schema["properties"]["findings"]["items"],
            },
            "gap_accounting": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "gap_id": {"type": "string"},
                        "actions_taken": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": [
                                "resolved",
                                "unresolved_parked",
                                "not_decision_relevant_on_inspection",
                            ],
                        },
                    },
                    "required": ["gap_id", "actions_taken", "status"],
                    "additionalProperties": True,
                },
                "description": (
                    "REQUIRED before concluding early: one entry per research-plan gap "
                    "(gap_id, what you did, and its status). An early conclude is rejected "
                    "until every plan gap is accounted for and the fetch floor is met."
                ),
            },
        },
        "additionalProperties": False,
    }
    plan_schema = {
        "type": "object",
        "properties": {
            "dry_run_forecast": {
                "type": "object",
                "description": (
                    "Your private dry-run forecast as the panel's STRUCTURED FORECAST block "
                    "(same shape as the template: question_type + posterior_prob / option_probs / "
                    "declared_percentiles). Telemetry only — never shown to the panel."
                ),
                "additionalProperties": True,
            },
            "sensitive_assumptions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 assumptions that would most move your forecast if wrong.",
            },
            "gaps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "question": {"type": "string"},
                        "why_decision_relevant": {"type": "string"},
                    },
                    "required": ["id", "question"],
                    "additionalProperties": True,
                },
                "description": (
                    "Ranked research gaps (most forecast-moving first): verify-targets "
                    "(assumptions to check) AND fill-targets (facts absent from the briefing). "
                    "At least one gap is required — a plan with no gaps is rejected."
                ),
            },
        },
        "required": ["gaps"],
        "additionalProperties": False,
    }
    return [
        _tool_schema(
            "set_research_plan",
            "Register your turn-one research plan (dry-run forecast, sensitive assumptions, ranked gaps). "
            "REQUIRED before any research tool — external tool calls are rejected until this is set.",
            plan_schema,
        ),
        _tool_schema(
            "record_findings",
            "Bank detached findings. Text evidence requires a quote; visual evidence requires a delivered image_id "
            "and visual_observation tied to that image's exact source URL. Claims must stay citation-only and avoid "
            "likelihood or verdict language. Optional derivation carries arithmetic over quoted text evidence.",
            finding_schema,
        ),
        _tool_schema(
            "conclude",
            "Finish the loop, optionally banking final findings and leaving pending leads for follow-up telemetry.",
            conclude_schema,
        ),
    ]


def _tool_schemas(tools: list[ToolSpec], must_conclude: bool) -> list[dict[str, Any]]:
    internal = _internal_tool_schemas()
    if must_conclude:
        return internal
    return internal + [_tool_schema(tool.name, tool.description, tool.parameters) for tool in tools]
