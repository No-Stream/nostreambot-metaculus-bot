"""The gates that constrain the gap-fill v2 driver, and the banking they guard.

Four numbered work-items from the v2 plan converge here:

* **W1** — the plan gate's rejection text (``_PLAN_REQUIRED_NUDGE``) and the
  ranked-gap coercion behind ``set_research_plan``.
* **W2** — the conclude gate: per-gap accounting, the one-external-call-per-gap
  invariant, and the fetch floor that stops a run concluding on snippets alone.
* **W3** — the hard ``source_url`` provenance check every finding must clear.
* **W4** — the code-derived verification tier stamped onto a finding at banking
  time, plus the idempotent banking itself.

``_validate_findings_payload`` — the driver-facing entry point that runs the
detachment lint, this module's provenance check and the warn-only quote
spot-check — deliberately stays in ``loop.py``: its ``GAP_FILL_V2
quote_mismatch`` WARN is asserted on the ``...agentic.loop`` logger, so moving it
would silently re-key that marker.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, NamedTuple

from pydantic import ValidationError

from metaculus_bot.research.agentic.loop_state import _LoopState, _must_conclude
from metaculus_bot.research.agentic.provenance import _normalize_url, _outranks
from metaculus_bot.research.agentic.tool_schemas import _INTERNAL_TOOL_NAMES
from metaculus_bot.research.agentic.types import (
    Finding,
    GapAccountingEntry,
    LoopConfig,
    PlannedGap,
    ToolOutcome,
)

# Returned in place of an external tool's result until set_research_plan has run
# (W1). Mirrors loop._NUDGE mechanics: the driver sees why the call was rejected and
# what to do instead. Capped at LoopConfig.max_plan_nudges (then soft-continue).
_PLAN_REQUIRED_NUDGE = (
    "call set_research_plan first — register your dry-run forecast, sensitive "
    "assumptions, and ranked research gaps before using any research tool"
)


class _FindingsValidation(NamedTuple):
    accepted: list[Finding]
    rejected: list[str]
    lint_rejections: int
    provenance_rejections: int
    quote_mismatch_warnings: int


def _check_url_provenance(finding: Finding, state: _LoopState) -> str | None:
    """Return a rejection reason if the finding's source_url isn't grounded, else None.

    A non-discrepancy finding is grounded when its normalized URL appears
    anywhere the driver could have seen it — a tool result this run OR the
    frozen briefing. A discrepancy finding must rest on a fresh primary-source
    check (per the driver prompt), so only a TOOL-sourced URL counts; a
    briefing-embedded URL does not.
    """
    normalized = _normalize_url(finding.source_url)
    if normalized is None:
        return f"source_url {finding.source_url!r} is not a valid http(s) URL"
    if normalized in state.tool_seen_urls:
        return None
    if finding.discrepancy:
        return (
            f"discrepancy source_url {finding.source_url!r} was not seen in any tool result this run; "
            "a discrepancy must rest on a fresh primary-source check (search/fetch/read), "
            "not a URL already in the briefing"
        )
    if normalized in state.briefing_urls:
        return None
    return (
        f"source_url {finding.source_url!r} did not appear in any tool result or the briefing this run; "
        "cite a URL you actually retrieved"
    )


def _check_image_provenance(finding: Finding, state: _LoopState) -> str | None:
    """Require visual evidence to name pixels delivered on an earlier model turn and their real source."""
    if finding.evidence_kind != "image":
        return None
    assert finding.image_id is not None
    if finding.image_id not in state.delivered_image_ids:
        return f"image_id {finding.image_id!r} was not delivered to the driver on a preceding model turn"
    registered_sources = state.image_sources_by_id.get(finding.image_id, set())
    if finding.source_url not in registered_sources:
        return (
            f"source_url {finding.source_url!r} is not a registered source or final URL for "
            f"image_id {finding.image_id!r}"
        )
    return None


def _coerce_pending_leads(raw_pending_leads: Any) -> tuple[list[str], list[str]]:
    if raw_pending_leads is None:
        return [], []
    if not isinstance(raw_pending_leads, list):
        return [], ["pending_leads must be a list of strings"]

    pending_leads: list[str] = []
    issues: list[str] = []
    for index, item in enumerate(raw_pending_leads):
        if isinstance(item, str):
            pending_leads.append(item)
        else:
            issues.append(f"pending_leads[{index}] invalid: expected string")
    return pending_leads, issues


def _stamp_verification_tier(finding: Finding, state: _LoopState) -> Finding:
    """Return a copy of ``finding`` with ``verification_tier`` set from the
    URL->best-tier map (W4) — CODE-derived, so any driver-supplied tier is
    overwritten. A source_url the driver never retrieved through a tool (e.g. a
    briefing-only URL) is absent from the map and stays untiered (None)."""
    normalized = _normalize_url(finding.source_url)
    tier = state.url_best_tier.get(normalized) if normalized is not None else None
    return finding.model_copy(update={"verification_tier": tier})


def _bank_findings(state: _LoopState, accepted: list[Finding]) -> tuple[int, int]:
    """Append findings to ``state.findings``, skipping ones already banked this run.

    Returns ``(banked, duplicates)``. Banking is idempotent by full-field
    identity — the finding's canonical JSON serialization EXCLUDING the
    loop-stamped ``verification_tier``: a driver that records findings
    incrementally with ``record_findings`` and then re-lists the same ones in
    ``conclude``'s ``final_findings`` (observed on Q578 — 8 findings rendered 16
    times) no longer doubles the list. Serializing every OTHER field keeps
    genuinely distinct findings that happen to share a source/quote (a different
    ``claim`` or ``topic``) separate, and stays correct if ``Finding`` gains a
    field. Each banked finding carries the code-derived tier (W4).

    A duplicate still RESTAMPS the stored finding's tier: when the URL's tier
    upgraded between the two banks (banked from a search snippet, then the same
    URL was successfully fetched, then re-listed in conclude), the affirmative
    re-record carries the fetched authority forward — otherwise a verified
    discrepancy would stay demoted to the "possible corrections" block. Only an
    explicit re-record upgrades (never a bare fetch alone: the driver may have
    fetched, seen the claim was wrong, and deliberately not re-listed it), and
    the best-tier merge in ``url_best_tier`` makes the restamp monotonic — a
    tier never downgrades.
    """
    banked = 0
    duplicates = 0
    for finding in accepted:
        # Key on the driver-stable identity (tier is loop-stamped, so exclude it)
        # so dedup survives a between-bank tier upgrade.
        key = finding.model_dump_json(exclude={"verification_tier"})
        stamped = _stamp_verification_tier(finding, state)
        existing_index = state.seen_finding_keys.get(key)
        if existing_index is not None:
            duplicates += 1
            if _outranks(stamped.verification_tier, state.findings[existing_index].verification_tier):
                state.findings[existing_index] = stamped
            continue
        state.seen_finding_keys[key] = len(state.findings)
        state.findings.append(stamped)
        banked += 1
    return banked, duplicates


def _apply_findings_telemetry(state: _LoopState, validation: _FindingsValidation) -> None:
    state.telemetry.lint_rejections += validation.lint_rejections
    state.telemetry.provenance_rejections += validation.provenance_rejections
    state.telemetry.quote_mismatch_warnings += validation.quote_mismatch_warnings


def _coerce_planned_gaps(raw_gaps: Any, *, max_gaps: int) -> tuple[list[PlannedGap], list[str]]:
    """Validate the plan's ``gaps`` list, capping at ``max_gaps`` (the tail — least
    forecast-moving — is dropped since the driver ranks them). Returns
    ``(gaps, issues)``; malformed entries are skipped with a reason."""
    if raw_gaps is None or not isinstance(raw_gaps, list):
        return [], ["gaps must be a list of {id, question, why_decision_relevant}"]
    gaps: list[PlannedGap] = []
    issues: list[str] = []
    for index, raw_gap in enumerate(raw_gaps):
        try:
            gaps.append(PlannedGap.model_validate(raw_gap))
        except ValidationError as exc:
            issues.append(f"gaps[{index}] invalid: {exc.errors()[0]['msg']}")
    if len(gaps) > max_gaps:
        issues.append(f"kept the top {max_gaps} of {len(gaps)} gaps (ranked); dropped the rest")
        gaps = gaps[:max_gaps]
    return gaps, issues


# Fetch floor (W2, clause c): the run must reach primary sources, not stop at
# search snippets. The global fallback clause is satisfied at this many
# successful fetched-tier retrievals; the per-gap clause reads the driver's own
# actions_taken note for a fetch/read mention.
_FETCH_FLOOR_MIN_CALLS = 2
# Word-boundary match on fetch/read verbs so common narration words don't count:
# a bare substring "read" fired on "already"/"spread"/"thread"/"ready", and
# "could not fetch the source" (an honest failed-fetch note) satisfied the floor
# with zero real retrievals (F2). \b anchors kill the "read"-in-a-word matches;
# the fetch verb is required CONJUGATED (fetched/fetches/fetching) so bare
# present-tense "fetch" — overwhelmingly negated/attempted ("could not fetch",
# "try to fetch") in a past-actions note — doesn't clear, while any real
# completed fetch/read mention still does. (bare `read` stays: its past tense is
# spelled identically, so "read the PDF" must count.)
_FETCH_ACTION_RE = re.compile(r"\b(?:fetch(?:ed|es|ing)|read(?:s|ing|_document)?)\b", re.IGNORECASE)


def _external_tool_call_count(state: _LoopState) -> int:
    """Accepted EXTERNAL tool calls this run — the internal bookkeeping tools
    (set_research_plan / record_findings / conclude) don't count toward the
    per-gap research floor."""
    return sum(count for name, count in state.telemetry.per_tool_counts.items() if name not in _INTERNAL_TOOL_NAMES)


def _coerce_gap_accounting(raw_accounting: Any) -> list[GapAccountingEntry]:
    """Validate conclude's ``gap_accounting`` into entries, skipping malformed
    ones (a skipped entry just reads as a still-unaccounted gap in the gate)."""
    if not isinstance(raw_accounting, list):
        return []
    entries: list[GapAccountingEntry] = []
    for raw_entry in raw_accounting:
        try:
            entries.append(GapAccountingEntry.model_validate(raw_entry))
        except ValidationError:
            continue
    return entries


def _actions_cite_fetch(actions_taken: str) -> bool:
    """True when the driver's free-text ``actions_taken`` mentions fetching or
    reading a source (the fetch floor's per-gap self-report clause). This prose
    signal is no longer trusted on its own: a failure note ("fetched the source
    but got a 403") also matches, so the conclude gate now counts it only
    alongside >=1 successful fetched-tier retrieval (see ``_conclude_gate_debts``);
    the telemetry-based clause is the robust one. Matches fetch/read verbs on word
    boundaries so "already"/"spread"/"could not fetch" don't false-positive."""
    return bool(_FETCH_ACTION_RE.search(actions_taken))


def _conclude_gate_debts(state: _LoopState, entries: list[GapAccountingEntry]) -> list[str]:
    """Outstanding conclude-gate requirements (W2); empty means the early
    conclusion may proceed. Only called when a research plan with gaps exists.

    Enforces: (a) every plan gap appears in the accounting; (b) at least one
    external tool call per plan gap (the loop can't attribute calls to gaps, so
    this is the cheap global invariant); (c) the fetch floor — either the top-2
    ranked gaps' accounting each cites a fetch/read action AND the run made >=1
    successful fetched-tier retrieval, OR the run made ``_FETCH_FLOOR_MIN_CALLS``+
    fetches/reads. The >=1-retrieval conjunct stops prose alone (which can cite a
    fetch verb even in a failure note) from clearing the floor with zero pages
    reached.
    """
    plan = state.research_plan
    # Caller guards both; asserted separately to keep the type narrow.
    assert plan is not None
    assert plan.gaps
    gaps = plan.gaps
    debts: list[str] = []

    accounted_ids = {entry.gap_id for entry in entries}
    missing_ids = [gap.id for gap in gaps if gap.id not in accounted_ids]
    if missing_ids:
        debts.append(f"gap_accounting is missing entries for gap(s): {', '.join(missing_ids)}")

    external_calls = _external_tool_call_count(state)
    if external_calls < len(gaps):
        debts.append(
            f"only {external_calls} external tool call(s) made for {len(gaps)} plan gap(s) — "
            "research each gap at least once before concluding"
        )

    # Count only successfully-retrieved primary sources, not fetch/read tool
    # CALLS: per_tool_counts increments at accept time regardless of outcome, so
    # two 403'd fetches would clear the floor though they reached nothing — the
    # exact 131.3 mechanism, and exactly what W4's tier map already excludes
    # (url_best_tier is populated only from status=="ok" fetched-class outcomes).
    # Reusing it makes W2's "fetched" agree with W4's (F4). (Distinct URLs, so a
    # single page fetched twice counts once — the intent is breadth of primary
    # sources reached.)
    fetches_reads = sum(1 for tier in state.url_best_tier.values() if tier == "fetched")
    entry_by_id = {entry.gap_id: entry for entry in entries}
    # Top-2 ranked gaps: a finite-N fetch-floor clause, not subsampling.
    top_gaps = gaps[:2]  # HARNESS-SCAN-EXEMPT-subsampling
    top_cite_fetch = bool(top_gaps) and all(
        gap.id in entry_by_id and _actions_cite_fetch(entry_by_id[gap.id].actions_taken) for gap in top_gaps
    )
    # The per-gap prose clause counts ONLY alongside >=1 real successful retrieval:
    # a fetch-verb note ("fetched the source but got a 403") also matches the
    # actions regex, so prose alone could clear the floor with zero pages reached.
    # Requiring one real fetched-tier retrieval closes that hole while preserving
    # the clause's purpose (one load-bearing fetch + honest per-gap notes beats the
    # global 2-fetch bar). The global clause remains a fetch-count-only fallback.
    if not ((top_cite_fetch and fetches_reads >= 1) or fetches_reads >= _FETCH_FLOOR_MIN_CALLS):
        debts.append(
            "fetch floor unmet: a top-ranked gap's fetch/read_document citation now counts only "
            f"alongside at least one successful fetched-tier retrieval (the run made {fetches_reads}); "
            f"otherwise make {_FETCH_FLOOR_MIN_CALLS}+ fetches/reads — fetch a primary source before "
            "concluding on the load-bearing gaps"
        )
    return debts


def _evaluate_conclude_gate(
    state: _LoopState, arguments: dict[str, Any], config: LoopConfig, now: Callable[[], float]
) -> ToolOutcome | None:
    """The W2 anti-satisficing gate. Returns a blocking error ToolOutcome (loop
    continues, ``conclude_gate_rejections`` bumped) when an EARLY conclusion
    hasn't met the work-list floor, else ``None`` (let the conclude proceed).

    Bypasses entirely — never blocks — when the deadline/budget forces a
    conclusion (``_must_conclude``) or once the rejection cap
    (``max_conclude_gate_rejections``) is hit (both guarantee no wedge).

    When a real research plan exists (gaps registered) its work-list floor is
    enforced regardless of ``plan_skipped`` — a valid plan set AFTER the plan-
    nudge cap fired re-arms the gate (F3c: ``plan_skipped`` is checked only in the
    no-plan branch below, not before the plan). When NO plan was set, a driver
    that legitimately soft-continued past the plan-nudge cap (``plan_skipped``) is
    let through, but an early conclude with no plan at all is rejected with a
    plan-first nudge (F3b: a driver can't skip planning by never calling it).
    """
    if _must_conclude(state, config, now):
        return None
    if state.telemetry.conclude_gate_rejections >= config.max_conclude_gate_rejections:
        return None

    plan = state.research_plan
    if plan is not None and plan.gaps:
        debts = _conclude_gate_debts(state, _coerce_gap_accounting(arguments.get("gap_accounting")))
    elif state.plan_skipped:
        return None
    else:
        debts = ["no research plan was set — call set_research_plan with ranked gaps before concluding"]
    if not debts:
        return None

    state.telemetry.conclude_gate_rejections += 1
    lines = [
        "Conclude rejected — resolve the outstanding research debt below before concluding "
        "(a forced deadline conclusion overrides this):"
    ]
    lines.extend(f"- {debt}" for debt in debts)
    return ToolOutcome(content_markdown="\n".join(lines), method="internal", status="error")
