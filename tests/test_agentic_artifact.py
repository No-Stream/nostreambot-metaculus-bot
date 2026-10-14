from __future__ import annotations

from typing import Literal

from metaculus_bot.research.agentic.artifact import (
    PENDING_LEADS_END_MARKER,
    PENDING_LEADS_START_MARKER,
    detachment_lint,
    render_findings,
)
from metaculus_bot.research.agentic.types import Finding


def test_render_findings_empty_returns_empty_string() -> None:
    assert render_findings([], []) == ""


def test_render_findings_is_deterministic_and_topic_sorted() -> None:
    findings = [
        Finding(
            claim="Treasury released the report.",
            source_url="https://example.com/b",
            quote="Report published on July 1.",
            date="2026-07-01",
            retrieved_how="fetch",
            topic="zeta",
        ),
        Finding(
            claim="The ministry published a corrigendum.",
            source_url="https://example.com/a1",
            quote="Corrigendum posted the next day.",
            date="2026-07-02",
            retrieved_how="read_document",
            topic="alpha",
        ),
        Finding(
            claim="A regulator archived the filing.",
            source_url="https://example.com/a2",
            quote="Archived filing includes the relevant clause.",
            date="2026-07-03",
            retrieved_how="fetch",
            topic="alpha",
        ),
    ]

    rendered = render_findings(findings, ["Check the appendix PDF."])

    assert rendered.startswith("## Agentic Research Findings")
    assert rendered.index("### alpha") < rendered.index("### zeta")
    assert rendered.index("The ministry published a corrigendum.") < rendered.index("A regulator archived the filing.")
    assert "Pending leads:\n- Check the appendix PDF." in rendered
    assert (
        f"{PENDING_LEADS_START_MARKER}\nPending leads:\n- Check the appendix PDF.\n{PENDING_LEADS_END_MARKER}"
        in rendered
    )


def test_render_findings_hoists_discrepancies_to_corrections_block_first() -> None:
    findings = [
        Finding(
            claim="The briefing says the rate is 7%, but the source says 5.2%.",
            source_url="https://example.com/correction",
            quote="The unemployment rate was 5.2%.",
            date="2026-07-10",
            retrieved_how="fetch",
            topic="labor",
            discrepancy=True,
            # Historical Findings can still carry a fetched tier. Rendering keeps
            # the correction visible above topic evidence without claiming authority.
            verification_tier="fetched",
        ),
        Finding(
            claim="The ministry published a bulletin.",
            source_url="https://example.com/bulletin",
            quote="Bulletin issued July 10.",
            date="2026-07-10",
            retrieved_how="fetch",
            topic="labor",
        ),
    ]

    rendered = render_findings(findings, [])

    assert "### ⚠ Driver-reported corrections to the briefing" in rendered
    assert rendered.index("### ⚠ Driver-reported corrections to the briefing") < rendered.index("### labor")
    assert (
        "Compare these claims with the briefing and each cited excerpt; source attribution does not verify interpretation."
        in rendered
    )
    assert "The briefing says the rate is 7%, but the source says 5.2%." in rendered
    assert "supersede" not in rendered.lower()


def _discrepancy(
    tier: Literal["fetched", "snippet"] | None, *, claim: str, url: str = "https://example.com/correction"
) -> Finding:
    return Finding(
        claim=claim,
        source_url=url,
        quote="The source states the corrected figure.",
        date="2026-07-10",
        retrieved_how="fetch",
        topic="labor",
        discrepancy=True,
        verification_tier=tier,
    )


def test_fetched_discrepancy_is_driver_reported_for_comparison() -> None:
    """A fetched excerpt is attributable evidence, not semantic verification."""
    rendered = render_findings([_discrepancy("fetched", claim="Briefing says 7%; source says 5.2%.")], [])

    assert "### ⚠ Driver-reported corrections to the briefing" in rendered
    assert "### ⚠ Driver-reported corrections (limited evidence)" not in rendered
    assert (
        "Compare these claims with the briefing and each cited excerpt; source attribution does not verify interpretation."
        in rendered
    )
    assert "Briefing says 7%; source says 5.2%." in rendered
    assert "supersede" not in rendered.lower()


def test_snippet_discrepancy_renders_limited_evidence_block() -> None:
    """Snippet and generated-answer corrections remain visibly less complete."""
    rendered = render_findings([_discrepancy("snippet", claim="Briefing says 133; a snippet says 131.3.")], [])

    assert "### ⚠ Driver-reported corrections (limited evidence)" in rendered
    assert "### ⚠ Driver-reported corrections to the briefing" not in rendered
    assert "These claims rely partly on snippets or generated answers; recheck the underlying sources." in rendered
    assert "Briefing says 133; a snippet says 131.3." in rendered
    assert "supersede" not in rendered.lower()


def test_untiered_discrepancy_is_demoted() -> None:
    """A discrepancy with no tier (None — its URL was never retrieved through a
    tool, or only via a failed fetch) is demoted conservatively, same as snippet."""
    rendered = render_findings([_discrepancy(None, claim="Briefing says X; an untiered source says Y.")], [])

    assert "### ⚠ Driver-reported corrections (limited evidence)" in rendered
    assert "### ⚠ Driver-reported corrections to the briefing" not in rendered


def test_mixed_discrepancies_render_both_evidence_levels_before_topics() -> None:
    """A run with fetched and limited-evidence discrepancies preserves both."""
    findings = [
        _discrepancy("snippet", claim="Snippet-only correction.", url="https://example.com/snippet"),
        _discrepancy("fetched", claim="Fetched correction.", url="https://example.com/fetched"),
        Finding(
            claim="The ministry published a bulletin.",
            source_url="https://example.com/bulletin",
            quote="Bulletin issued July 10.",
            date="2026-07-10",
            retrieved_how="fetch",
            topic="labor",
            verification_tier="fetched",
        ),
    ]

    rendered = render_findings(findings, [])

    fetched_at = rendered.index("### ⚠ Driver-reported corrections to the briefing")
    limited_at = rendered.index("### ⚠ Driver-reported corrections (limited evidence)")
    topic_at = rendered.index("### labor")
    assert fetched_at < limited_at < topic_at
    assert "Fetched correction." in rendered
    assert "Snippet-only correction." in rendered
    assert "The ministry published a bulletin." in rendered
    assert "supersede" not in rendered.lower()


def test_verification_tier_renders_compactly_on_topic_findings() -> None:
    """W4: a non-discrepancy finding carrying a tier shows it inline alongside
    retrieved_how, so the panel sees each finding's retrieval quality."""
    findings = [
        Finding(
            claim="The agency reported the figure.",
            source_url="https://example.com/fetched",
            quote="The figure is 4.1 percent.",
            date="2026-07-01",
            retrieved_how="fetch",
            topic="labor",
            verification_tier="fetched",
        ),
        Finding(
            claim="A secondary source echoed a figure.",
            source_url="https://example.com/snippet",
            quote="Reported at about 4 percent.",
            date="2026-07-02",
            retrieved_how="search_web",
            topic="labor",
            verification_tier="snippet",
        ),
    ]

    rendered = render_findings(findings, [])

    assert "Retrieved how: fetch [verification: fetched]" in rendered
    assert "Retrieved how: search_web [verification: snippet]" in rendered


def test_untiered_topic_finding_omits_verification_token() -> None:
    """A finding with no tier (briefing-grounded, never tool-retrieved) renders
    the plain retrieved-how line — no stray verification token."""
    rendered = render_findings(
        [
            Finding(
                claim="The ministry published a bulletin.",
                source_url="https://example.com/bulletin",
                quote="Bulletin issued July 10.",
                date="2026-07-10",
                retrieved_how="fetch",
                topic="labor",
            )
        ],
        [],
    )

    assert "Retrieved how: fetch" in rendered
    assert "[verification:" not in rendered


def test_render_findings_without_discrepancies_has_no_corrections_block() -> None:
    rendered = render_findings(
        [
            Finding(
                claim="The ministry published a bulletin.",
                source_url="https://example.com/bulletin",
                quote="Bulletin issued July 10.",
                date="2026-07-10",
                retrieved_how="fetch",
                topic="labor",
            )
        ],
        [],
    )

    assert "### ⚠ Driver-reported corrections to the briefing" not in rendered
    assert rendered.startswith("## Agentic Research Findings\n\n### labor")


def test_detachment_lint_flags_discrepancy_claims_too() -> None:
    violations = detachment_lint(
        Finding(
            claim="The briefing likely overstates the rate.",
            source_url="https://example.com/correction",
            quote="The rate was 5.2%.",
            discrepancy=True,
        )
    )

    assert "claim contains banned register 'likely'" in violations


def test_detachment_lint_flags_claim_and_topic_only() -> None:
    finding = Finding(
        claim="This likely resolves in 2026.",
        source_url="https://example.com",
        quote="Analyst note says the odds are improving.",
        topic="Bullish updates",
    )

    violations = detachment_lint(finding)

    assert "claim contains banned register 'likely'" in violations
    assert "topic contains banned register 'Bullish'" in violations
    assert all("odds are" not in violation for violation in violations)


def test_detachment_lint_exempts_derivation_field() -> None:
    """W3: the ``derivation`` field is arithmetic-only synthesis (saw-tooth-class
    tables/bounds) and is exempt from the banned-register scan. The SAME phrase
    that trips the lint in ``claim`` passes when it appears only in derivation."""
    banned_phrase = "the annual record probably tops the prior year by 1"
    exempt = Finding(
        claim="Yearly oldest-human records reconstructed from the quoted table.",
        source_url="https://example.com/table",
        quote="1997: 122; 1998: 116; 1999: 119.",
        derivation=banned_phrase,
    )
    linted = Finding(
        claim=banned_phrase,
        source_url="https://example.com/table",
        quote="1997: 122; 1998: 116; 1999: 119.",
    )

    assert detachment_lint(exempt) == []
    assert any("probably" in violation for violation in detachment_lint(linted))


def test_render_labels_derived_findings() -> None:
    """W3: a finding carrying a derivation gets a visible label in its topic block
    so forecasters weight it as our synthesis, not a source claim."""
    findings = [
        Finding(
            claim="Per-year upper bound on the oldest living human, from the quoted record.",
            source_url="https://example.com/grg",
            quote="Verified oldest person 1990-2000: 122, 122, 122, 122, 119, 117, 116, 115, 114, 113.",
            date="2026-07-01",
            retrieved_how="fetch",
            topic="oldest-human bounds",
            derivation="Max verified age per calendar year: 1990=122 ... 2000=113; annual step never exceeds +1.",
        )
    ]

    rendered = render_findings(findings, [])

    assert "### oldest-human bounds" in rendered
    assert "Derived analysis (arithmetic from quoted sources)" in rendered
    assert "Max verified age per calendar year" in rendered
    # The label sits inside the topic block, not before it.
    assert rendered.index("### oldest-human bounds") < rendered.index(
        "Derived analysis (arithmetic from quoted sources)"
    )


def test_fetched_discrepancy_renders_derivation_in_driver_reported_block() -> None:
    """F5: a finding that is both discrepancy=True and carries a derivation — the
    literal 131.3 shape ("briefing says 133; arithmetic from quoted sources gives
    131.3") — routes to the fetched correction block and must KEEP its arithmetic support:
    the derived-analysis label and the Derivation line, not just Claim/Quote."""
    finding = Finding(
        claim="Briefing says 133; the quoted sources sum to 131.3.",
        source_url="https://example.com/correction",
        quote="Line items: 40.0, 51.3, 40.0.",
        date="2026-07-10",
        retrieved_how="fetch",
        topic="labor",
        discrepancy=True,
        verification_tier="fetched",
        derivation="40.0 + 51.3 + 40.0 = 131.3, not the 133 the briefing states.",
    )

    rendered = render_findings([finding], [])

    assert "### ⚠ Driver-reported corrections to the briefing" in rendered
    # The arithmetic support survives in the corrections block.
    assert "Derived analysis (arithmetic from quoted sources)" in rendered
    assert "Derivation: 40.0 + 51.3 + 40.0 = 131.3, not the 133 the briefing states." in rendered
    # The label/derivation sit inside the corrections block, above the topic blocks.
    assert rendered.index("### ⚠ Driver-reported corrections to the briefing") < rendered.index("Derived analysis")
    assert "supersede" not in rendered.lower()


def test_snippet_discrepancy_renders_derivation_in_demoted_block() -> None:
    """A snippet-tier derived discrepancy keeps its derivation in the limited-evidence block."""
    finding = Finding(
        claim="Briefing says 133; a snippet's numbers give 131.3.",
        source_url="https://example.com/snippet",
        quote="Reported components: 40, 51.3, 40.",
        date="2026-07-10",
        retrieved_how="search_web",
        topic="labor",
        discrepancy=True,
        verification_tier="snippet",
        derivation="40 + 51.3 + 40 = 131.3.",
    )

    rendered = render_findings([finding], [])

    assert "### ⚠ Driver-reported corrections (limited evidence)" in rendered
    assert "Derived analysis (arithmetic from quoted sources)" in rendered
    assert "Derivation: 40 + 51.3 + 40 = 131.3." in rendered


def test_derived_correction_render_is_deterministic() -> None:
    """Determinism guard extended to derived corrections: repeated renders of a
    derived discrepancy produce byte-identical output."""
    finding = Finding(
        claim="Briefing says 133; arithmetic gives 131.3.",
        source_url="https://example.com/correction",
        quote="Components: 40.0, 51.3, 40.0.",
        date="2026-07-10",
        retrieved_how="fetch",
        topic="labor",
        discrepancy=True,
        verification_tier="fetched",
        derivation="40.0 + 51.3 + 40.0 = 131.3.",
    )

    assert render_findings([finding], []) == render_findings([finding], [])


def test_render_omits_derivation_label_for_plain_findings() -> None:
    """A finding with no derivation renders exactly as before — no stray label."""
    rendered = render_findings(
        [
            Finding(
                claim="The ministry published a bulletin.",
                source_url="https://example.com/bulletin",
                quote="Bulletin issued July 10.",
                date="2026-07-10",
                retrieved_how="fetch",
                topic="labor",
            )
        ],
        [],
    )

    assert "Derived analysis" not in rendered


def test_render_with_derivation_is_deterministic() -> None:
    """Determinism guard extended to the derivation label: repeated renders of the
    same findings produce byte-identical output."""
    findings = [
        Finding(
            claim="Derived bound table.",
            source_url="https://example.com/a",
            quote="Row values: 10, 11, 12.",
            date="2026-07-01",
            retrieved_how="fetch",
            topic="alpha",
            derivation="10 + 1 = 11; 11 + 1 = 12.",
        ),
        Finding(
            claim="A plain source claim.",
            source_url="https://example.com/b",
            quote="Report published.",
            date="2026-07-02",
            retrieved_how="fetch",
            topic="beta",
        ),
    ]

    assert render_findings(findings, ["lead"]) == render_findings(findings, ["lead"])
