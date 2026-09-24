from __future__ import annotations

import re
from collections import defaultdict

from metaculus_bot.research.agentic.types import Finding

_BANNED_REGISTER_PATTERNS: tuple[str, ...] = (
    r"likely",
    r"unlikely",
    r"probably",
    r"probable",
    r"suggests",
    r"indicates that",
    r"we believe",
    r"we expect",
    r"expected to (?:win|pass|fail|resolve)",
    r"points to (?:a|the)",
    r"this implies",
    r"bullish",
    r"bearish",
    r"i think",
    r"in our view",
    r"odds are",
)
_BANNED_REGISTER_RE = re.compile(r"\b(?:" + "|".join(_BANNED_REGISTER_PATTERNS) + r")\b", re.IGNORECASE)

# Label prefixing a finding that carries a ``derivation`` (W3) — flags the
# arithmetic synthesis as ours so the panel weights it accordingly.
_DERIVED_ANALYSIS_LABEL = "Derived analysis (arithmetic from quoted sources)"

# Corrections block headers + banners (W4). Discrepancy findings split by their
# loop-stamped ``verification_tier``: a "fetched" discrepancy (we pulled the
# primary source) keeps supersede authority; a "snippet"/untiered one (seen only
# in a search excerpt, or via a failed retrieval) is demoted so a snippet-median
# "correction" can no longer override the briefing for every forecaster (the
# 131.3 failure mode).
_SUPERSEDE_HEADER = "### ⚠ Corrections to the briefing"
_SUPERSEDE_BANNER = (
    "The sourced findings below contradict the research briefing and supersede the corresponding briefing content."
)
_DEMOTED_HEADER = "### ⚠ Possible corrections (snippet-sourced — recheck advised)"
_DEMOTED_BANNER = (
    "The findings below contradict the briefing but rest only on search-snippet evidence, not a fetched primary "
    "source. They do NOT supersede the briefing — treat them as leads that contradict it, and recheck against the "
    "primary source before relying on them."
)


def detachment_lint(finding: Finding) -> list[str]:
    # Driver-authored ``claim``, ``topic`` and visual observation are scanned. ``derivation`` (W3) is
    # deliberately EXEMPT — it is arithmetic-only synthesis over the finding's
    # own quoted numbers (a saw-tooth-style bound/rate table), which needs no
    # likelihood language and would false-positive on incidental register words
    # inside a computation. Do NOT add ``derivation`` here; the contract that
    # keeps it safe (every input quoted, arithmetic only, no new facts) is a
    # prompt/render-side convention, not a lint rule. ``quote`` stays unscanned
    # too — it is verbatim source text, not the driver's own register.
    violations: list[str] = []
    authored_fields = [("claim", finding.claim), ("topic", finding.topic)]
    if finding.visual_observation is not None:
        authored_fields.append(("visual_observation", finding.visual_observation))
    for field_name, value in authored_fields:
        matches = [match.group(0) for match in _BANNED_REGISTER_RE.finditer(value)]
        for match in matches:
            violations.append(f"{field_name} contains banned register {match!r}")
    return violations


def _quote_lines(quote: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in quote.splitlines() or [""]]


def _retrieved_how_line(finding: Finding) -> str:
    """Render the retrieved-how line with the loop-stamped verification tier
    appended compactly (W4). A tier of None (a source never retrieved through a
    tool) is omitted, so plain briefing-grounded findings read as before."""
    if finding.verification_tier is None:
        return f"Retrieved how: {finding.retrieved_how}"
    return f"Retrieved how: {finding.retrieved_how} [verification: {finding.verification_tier}]"


def _finding_body_lines(finding: Finding, *, include_tier: bool) -> list[str]:
    """Render one finding's body (label, claim, source, quote, derivation, date,
    retrieved-how) — the single shared renderer for BOTH the corrections blocks
    and the topic blocks, so a derived correction keeps its arithmetic support in
    either path (F5). A finding can be both ``discrepancy=True`` and carry a
    ``derivation`` (the literal 131.3 shape: "briefing says 133; arithmetic from
    quoted sources gives 131.3"), and that evidence must not vanish when the
    finding routes to a corrections block.

    ``include_tier`` controls only the retrieved-how line: topic findings append
    the loop-stamped verification tier inline (W4); correction findings omit it —
    their block header already states the tier. Deterministic: the derivation
    label/line are emitted iff the field is set, in stable field order.
    """
    body: list[str] = []
    if finding.derivation:
        # W3: label our arithmetic synthesis so the panel weights it as a derived
        # analysis, not a source claim.
        body.append(_DERIVED_ANALYSIS_LABEL)
    body.append(f"Claim: {finding.claim}")
    body.append(f"Source: {finding.source_url}")
    if finding.evidence_kind == "image":
        body.append(f"Image ID: {finding.image_id}")
        body.append(
            "Visual observation (driver-reported from delivered pixels; transcribed/estimated as stated): "
            f"{finding.visual_observation}"
        )
    else:
        body.append("Quote:")
        body.extend(_quote_lines(finding.quote))
    if finding.derivation:
        body.append(f"Derivation: {finding.derivation}")
    body.append(f"Date: {finding.date}")
    body.append(_retrieved_how_line(finding) if include_tier else f"Retrieved how: {finding.retrieved_how}")
    return body


def _append_correction_block(lines: list[str], header: str, banner: str, corrections: list[Finding]) -> None:
    """Append one corrections block (header + banner + each finding) to ``lines``.

    Shared by the fetched (supersede) and snippet-sourced (demoted) blocks so the
    two stay byte-for-byte parallel except for their header/banner. Corrections
    omit the tier token (the block header already states it) but keep the derived-
    analysis label/derivation via the shared body renderer (F5).
    """
    lines.append("")
    lines.append(header)
    lines.append(banner)
    lines.append("")
    for finding in corrections:
        lines.extend(_finding_body_lines(finding, include_tier=False))
        lines.append("")
    lines.pop()


def render_findings(findings: list[Finding], pending_leads: list[str]) -> str:
    if not findings and not pending_leads:
        return ""

    # W4: discrepancies split by loop-stamped tier. Only a "fetched" discrepancy
    # (we pulled the primary source) keeps supersede authority; "snippet" and
    # untiered (None — never retrieved through a tool, or via a failed fetch)
    # are demoted so a snippet-sourced "correction" can't override the briefing
    # for every forecaster. Insertion order is preserved within each block
    # (deterministic).
    supersede_corrections = [f for f in findings if f.discrepancy and f.verification_tier == "fetched"]
    demoted_corrections = [f for f in findings if f.discrepancy and f.verification_tier != "fetched"]
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        if finding.discrepancy:
            continue
        grouped[finding.topic].append(finding)

    lines = ["## Agentic Research Findings"]
    if supersede_corrections:
        _append_correction_block(lines, _SUPERSEDE_HEADER, _SUPERSEDE_BANNER, supersede_corrections)
    if demoted_corrections:
        _append_correction_block(lines, _DEMOTED_HEADER, _DEMOTED_BANNER, demoted_corrections)

    for topic in sorted(grouped):
        lines.append("")
        lines.append(f"### {topic}")
        for finding in grouped[topic]:
            lines.extend(_finding_body_lines(finding, include_tier=True))
            lines.append("")
        lines.pop()

    if pending_leads:
        lines.append("")
        lines.append("Pending leads:")
        for lead in pending_leads:
            lines.append(f"- {lead}")

    return "\n".join(lines)
