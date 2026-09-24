"""The agentic loop's data types: tool contracts, findings, the plan, and the loop's config, telemetry and result.

Field rationale lives in docs/agentic_gap_fill.md ("The findings gates", "The bounds", "Telemetry",
"The ghost forecast"); each field here carries at most one line of why.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

from metaculus_bot.constants import (
    GAP_FILL_V2_CONCLUDE_THRESHOLD,
    GAP_FILL_V2_MAX_TOOL_CALLS,
    GAP_FILL_V2_WALL_DEADLINE,
)
from metaculus_bot.research.image_assets import ImageView

if TYPE_CHECKING:
    from metaculus_bot.research.agentic.llm import LlmCall


class ToolOutcome(BaseModel):
    content_markdown: str
    links: list[str] = Field(default_factory=list)
    method: str = ""
    status: str = "ok"
    truncated: bool = False
    image_views: list[ImageView] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[ToolOutcome]]
    timeout_s: float


class Finding(BaseModel):
    """One citation-grounded finding; the W3 derivation and W4 tier rules: docs/agentic_gap_fill.md "The findings gates"."""

    claim: str
    source_url: str
    quote: str = ""
    date: str = ""
    retrieved_how: str = ""
    topic: str = "general"
    discrepancy: bool = False
    # Arithmetic over the finding's own quoted numbers only, so it is exempt from the detachment lint (W3).
    derivation: str | None = None
    # Code-stamped at banking time from the URL's best method seen, never driver-claimed (W4).
    verification_tier: Literal["fetched", "snippet"] | None = None
    evidence_kind: Literal["text", "image"] = "text"
    image_id: str | None = None
    visual_observation: str | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> Finding:
        if self.evidence_kind == "text" and not self.quote:
            raise ValueError("text evidence requires quote")
        if self.evidence_kind == "image":
            if not self.image_id:
                raise ValueError("image evidence requires image_id")
            if not self.visual_observation or not self.visual_observation.strip():
                raise ValueError("image evidence requires visual_observation")
        return self


class GhostForecast(BaseModel):
    qtype: str
    raw_text: str
    parsed_summary: str = ""


class PlannedGap(BaseModel):
    """One ranked research gap the driver commits to in set_research_plan (W1).

    ``id`` is the driver-chosen handle W2's conclude-time accounting keys on;
    ``question`` is the factual question to resolve; ``why_decision_relevant``
    is the ranking rationale (the trailing slot holds the least forecast-moving
    gap). Both verify-targets (assumptions to check) and fill-targets (facts
    absent from the briefing) live here — the debiased union of v1's and v2's
    targeting questions.
    """

    id: str
    question: str
    why_decision_relevant: str = ""


class ResearchPlan(BaseModel):
    """The driver's turn-one research plan, emitted via set_research_plan (W1).

    The private dry run stays in the driver's reasoning; its outputs surface
    here. ``dry_run_forecast`` is the qtype-shaped structured-block payload (same
    schema family as the ghost) used only for the GHOST_PRE telemetry delta;
    ``sensitive_assumptions`` are the 3-5 assumptions that would most move the
    forecast if wrong; ``gaps`` is the ranked work-list (capped at
    ``GAP_FILL_V2_MAX_GAPS``). W2 reads ``gaps`` for conclude-time accounting.
    """

    dry_run_forecast: dict[str, Any] | None = None
    sensitive_assumptions: list[str] = Field(default_factory=list)
    gaps: list[PlannedGap] = Field(default_factory=list)


# A plan gap's terminal disposition at conclude time (W2); the gate needs an entry per gap, not a particular status.
GapStatus = Literal["resolved", "unresolved_parked", "not_decision_relevant_on_inspection"]


class GapAccountingEntry(BaseModel):
    """One plan gap's disposition, supplied in conclude's ``gap_accounting`` (W2).

    ``gap_id`` keys back to a ``PlannedGap.id`` from the turn-one research plan;
    ``actions_taken`` is the driver's free-text note of what it did for this gap
    (searches run, pages fetched, why it parked it); ``status`` is the terminal
    disposition. The conclude gate rejects an early conclusion unless every plan
    gap id appears here (plus the global tool-call and fetch-floor invariants);
    see loop._conclude_tool.
    """

    gap_id: str
    actions_taken: str = ""
    status: GapStatus = "resolved"


@dataclass(slots=True)
class LoopConfig:
    """The loop's knobs; the budget defaults derive from the constants the seam passes in, so the two cannot disagree."""

    model: str
    reasoning_effort: str = "medium"
    max_tool_calls: int = GAP_FILL_V2_MAX_TOOL_CALLS
    wall_deadline_s: float = GAP_FILL_V2_WALL_DEADLINE
    conclude_threshold_s: float = GAP_FILL_V2_CONCLUDE_THRESHOLD
    max_result_chars: int = 8000
    max_steps: int = 20
    # Ranked gaps set_research_plan keeps (W1); the driver ranks them, so the dropped tail is the least valuable.
    max_gaps: int = 4
    # Plan-gate rejections before the loop soft-continues unplanned, so a driver that never plans cannot wedge it (W1).
    max_plan_nudges: int = 2
    # Conclude-gate rejections before an early conclusion is accepted anyway (W2); budget exhaustion bypasses the gate.
    max_conclude_gate_rejections: int = 2
    # The question ref the transport stamps on each call's ledger metadata, so PROMPT_SIZE_ALERT can name it.
    question_ref: str | None = None


@dataclass(slots=True)
class LoopTelemetry:
    """The GAP_FILL_V2 completion marker's fields; what each one counts: docs/agentic_gap_fill.md "Telemetry"."""

    model: str = ""
    steps: int = 0
    tool_calls: int = 0
    per_tool_counts: dict[str, int] = field(default_factory=dict)
    rendered_fetches: int = 0
    dup_tool_calls: int = 0
    deadline_hit: bool = False
    concluded_early: bool = False
    wall_s: float = 0.0
    findings_count: int = 0
    pending_leads_count: int = 0
    lint_rejections: int = 0
    provenance_rejections: int = 0
    quote_mismatch_warnings: int = 0
    plan_gaps: int = 0
    plan_skipped: bool = False
    conclude_gate_rejections: int = 0
    # None on a healthy run AND on a deadline hit; the one field that tells a step-0 crash from an idle run.
    error: str | None = None


@dataclass(slots=True)
class GhostContext:
    """What a second ghost needs to branch off the loop's cached prefix without seeing the first ghost.

    ``messages`` is the transcript up to, not including, the plain ghost's prompt; ``tools_json``
    is the tool list the last research turn offered; ``llm_call`` is the transport the loop used.
    """

    messages: list[dict[str, Any]]
    tools_json: list[dict[str, Any]]
    llm_call: LlmCall


@dataclass(slots=True)
class LoopResult:
    findings_markdown: str
    ghost: GhostForecast | None
    telemetry: LoopTelemetry
    transcript: list[dict[str, Any]]
    # Present only when the plain ghost ran, so every v1 ghost has its pair.
    ghost_context: GhostContext | None = None
    # Unique pixel payloads plus all source/final URL aliases needed by the asset manifest.
    image_views: list[ImageView] = field(default_factory=list)
    image_sources: dict[str, list[str]] = field(default_factory=dict)
    # Every distinct byte-free source/crop observation, even when multiple inputs normalize to one PNG.
    image_observations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
