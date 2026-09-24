"""The reasoning rules and template steps inside the three base forecaster prompts.

Every surviving rule from the 2026-09-02 de-bloat has a presence pin on its wording here and
every removed constant or phrase an absence pin, which is what makes AGENTS.md's claim to that
effect true. These are string assertions on the rendered prompt, not tests of the pipeline that
calls it, so a rule can be reworded only by updating the pin that names it.
"""

import json
from datetime import UTC, datetime

import pytest

from metaculus_bot.prompts import (
    _HISTORY_DISCHARGED_RULE,
    _SOFT_CLOCK_RULE,
    binary_prompt,
    gap_fill_analyzer_prompt,
    multiple_choice_prompt,
    numeric_prompt,
    stacking_binary_prompt,
    stacking_multiple_choice_prompt,
    stacking_numeric_prompt,
)
from metaculus_bot.research.gemini_attribution import UNVERIFIED_ATTRIBUTION_MARKER
from tests.prompt_builders import (
    _binary_prompt_text,
    _binary_q,
    _extract_last_json_block,
    _flat,
    _mc_prompt_text,
    _mc_q,
    _numeric_prompt_text,
    _numeric_q,
    _stacked_prompt_texts,
)


class TestForecastingWindowAnchor:
    """Every forecasting prompt must surface the open date, today, and
    resolution date so the LLM anchors on the forecasting window and does
    NOT treat pre-open historical events as already-resolved (e.g., the
    classic 1945-Japan-detonation → "Will a detonation occur by 2030?"
    auto-YES error)."""

    # -- binary ------------------------------------------------------------

    def test_binary_injects_open_and_resolution_dates(self) -> None:
        q = _binary_q(
            open_time=datetime(2026, 1, 15),
            resolve_time=datetime(2030, 12, 31),
        )
        result = binary_prompt(q, research="r")
        assert "2026-01-15" in result
        assert "2030-12-31" in result
        assert "Forecasting window" in result
        assert "days ago" in result
        assert "days from now" in result
        assert "BEFORE the open date" in result

    def test_binary_pre_open_rule_is_stated_twice_not_three_times(self) -> None:
        """The pre-open footgun has cost the bot badly, so the rule is deliberately stated
        TWICE: the forecasting-window line ("events before the open date do NOT resolve YES")
        and the status-quo derivation's demand to name the specific POST-OPEN event. The third
        statement, a 447-char 0a restatement with the 1945-detonation worked example, was
        retired as pure repetition (its receipt is the window line's own docstring)."""
        result = binary_prompt(_binary_q(), research="r")
        assert "BEFORE the open date" in result
        assert "POST-OPEN event" in result
        assert "1945" not in result
        assert "pre-dating the open date" not in result
        assert "open timestamp" not in result

    def test_binary_asserts_on_missing_open_time(self) -> None:
        """Missing timestamps are a data bug, not a graceful-degrade path."""
        q = _binary_q()
        q.open_time = None
        with pytest.raises(AssertionError):
            binary_prompt(q, research="r")

    def test_binary_asserts_on_missing_scheduled_resolution(self) -> None:
        q = _binary_q()
        q.scheduled_resolution_time = None
        with pytest.raises(AssertionError):
            binary_prompt(q, research="r")

    # -- multiple choice ---------------------------------------------------

    def test_multiple_choice_injects_window(self) -> None:
        q = _mc_q(
            open_time=datetime(2025, 3, 1),
            resolve_time=datetime(2027, 3, 1),
        )
        result = multiple_choice_prompt(q, research="r")
        assert "2025-03-01" in result
        assert "2027-03-01" in result
        assert "Forecasting window" in result
        assert "BEFORE the open date" in result

    def test_multiple_choice_asserts_on_missing_timestamps(self) -> None:
        q = _mc_q()
        q.open_time = None
        with pytest.raises(AssertionError):
            multiple_choice_prompt(q, research="r")

    # -- numeric -----------------------------------------------------------

    def test_numeric_injects_window(self) -> None:
        q = _numeric_q(
            open_time=datetime(2024, 6, 1),
            resolve_time=datetime(2026, 6, 1),
        )
        result = numeric_prompt(q, research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        assert "2024-06-01" in result
        assert "2026-06-01" in result
        assert "Forecasting window" in result
        assert "BEFORE the open date" in result

    def test_numeric_asserts_on_missing_timestamps(self) -> None:
        q = _numeric_q()
        q.scheduled_resolution_time = None
        with pytest.raises(AssertionError):
            numeric_prompt(q, research="r", lower_bound_message="lbm", upper_bound_message="ubm")


class TestSourceProvenanceLadder:
    """The Source-analysis section of every forecaster prompt must carry the
    provenance trust ladder: rank claims by proximity to the primary record and
    adjust by source motivation (discount self-interest, treat statements
    against interest as strong evidence). These are lightweight "did the text
    land" guards — we expect them to break when the prompt is revised."""

    def _assert_ladder_present(self, prompt: str) -> None:
        # Collapse whitespace so assertions don't depend on where clean_indents wraps lines.
        lowered = " ".join(prompt.lower().split())
        assert "proximity to the primary record" in lowered
        assert "against the speaker's interest" in lowered
        # The A-D DEFINITIONS live once, in the research-side _SOURCE_TIER_TAG_INSTRUCTION, and
        # the briefing arrives carrying the tags (99 of 1,069 archive records, every one since the
        # tagging landed in prod). The forecaster ladder names the tag shape and keeps only the two
        # USAGE clauses the tag instruction does not state: (C) facts-not-framing, (D) suggestive only.
        assert "arrive tagged by source tier" in lowered
        assert "[a: ...]" in lowered
        assert "use their cited facts, not their framing" in lowered
        assert "suggestive only" in lowered
        assert "government statistics, regulatory filings" not in lowered, (
            "the tier definitions were re-teaching what the research-side tag instruction already defines"
        )

    def _assert_unverified_attribution_defined(self, prompt: str) -> None:
        """`[unverified attribution]` reaches the forecaster in gemini research sections
        (`research/gemini_attribution.py` writes it over a tier tag that names no outlet, or
        names one the grounding record cannot back), and it lands on a bundle whose ladder tells the model to
        weight by tier. Undefined, it is a token the forecaster has to guess at, on exactly the
        claims where the guess matters."""
        lowered = " ".join(prompt.lower().split())
        assert "[unverified attribution]" in lowered
        # Two causes since 2026-09-24: a tag naming no outlet (a class description such as
        # "official") is rewritten as well as a named outlet the record cannot back.
        assert "whose tag named no outlet, or one the research pipeline could not match" in lowered
        assert "could not match against its own retrieval record" in lowered
        assert "whose named outlet the research pipeline could not match" not in lowered
        # The two halves that keep it from reading as "this fact is false" or as a tier grade.
        assert "the claim itself may still be correct" in lowered
        assert "untiered, unattributed evidence rather than as a named outlet's authority" in lowered

    def test_binary_prompt_defines_the_unverified_attribution_token(self) -> None:
        self._assert_unverified_attribution_defined(binary_prompt(_binary_q(), research="r"))

    def test_multiple_choice_prompt_defines_the_unverified_attribution_token(self) -> None:
        self._assert_unverified_attribution_defined(multiple_choice_prompt(_mc_q(), research="r"))

    def test_numeric_prompt_defines_the_unverified_attribution_token(self) -> None:
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        self._assert_unverified_attribution_defined(result)

    def test_the_definition_matches_the_marker_the_rewriter_actually_writes(self) -> None:
        """Pinned against the emitting constant rather than a copied string: if the rewriter's
        wording changes, the forecaster-facing definition has to move with it or the prompt
        defines a token no section carries."""
        assert f"[{UNVERIFIED_ATTRIBUTION_MARKER}]" in binary_prompt(_binary_q(), research="r")

    def test_binary_prompt_carries_provenance_ladder(self) -> None:
        self._assert_ladder_present(binary_prompt(_binary_q(), research="r"))

    def test_multiple_choice_prompt_carries_provenance_ladder(self) -> None:
        self._assert_ladder_present(multiple_choice_prompt(_mc_q(), research="r"))

    def test_numeric_prompt_carries_provenance_ladder(self) -> None:
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        self._assert_ladder_present(result)

    def test_numeric_status_quo_derivation_is_the_one_anchor_to_latest_statement(self) -> None:
        """The numeric prompt used to tell the model five times how to pick a center: the
        step-0 status-quo derivation, a step-1 "centered near this value" push, a step-3
        "status-quo outcome" line, step-3 trend continuation and a step-7 trajectory check.
        One anchor statement (step 0, which already says to move off the latest measurement
        only for a named post-open event) and one trend statement (step 3) remain."""
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        lowered = " ".join(result.lower().split())
        assert lowered.count("most recent authoritative measurement") == 1
        assert lowered.count("trend continuation") == 1
        assert "centered near this value" not in lowered
        assert "status-quo outcome" not in lowered
        assert "trajectory check" not in lowered
        # Step 1 no longer claims a "data anchor" it does not carry.
        assert "source analysis and data anchor" not in lowered


class TestNullResultReadingClause:
    """A search that found nothing licenses "could not find evidence of X", never
    "X did not happen". On qid 44799 four of six forecasters read the gap-fill
    resolver's "I found no authoritative public record" as "the permit is absent",
    and the two members that discounted it scored best in the ensemble. Every base
    forecaster prompt must carry the reading rule; the gap-fill analyzer carries the
    auditor's version of it."""

    def _assert_forecaster_clause_present(self, prompt: str) -> None:
        # Collapse whitespace so assertions don't depend on where clean_indents wraps lines.
        lowered = " ".join(prompt.lower().split())
        assert "read a null search result as a null search result" in lowered
        assert "could not find evidence of x" in lowered
        # Coverage-conditioned strength.
        assert "weight the absence by how well the topic is covered" in lowered
        # The retired third bullet ("weaker still where the actor has already demonstrated the
        # behavior") carried no receipt of its own and pushed the wrong way on qid 43837, where
        # Metaculus had announced eleven prior tournaments, none was found, and the answer was NO.
        assert "already demonstrated the capability" not in lowered

    def test_binary_prompt_carries_null_result_clause(self) -> None:
        self._assert_forecaster_clause_present(binary_prompt(_binary_q(), research="r"))

    def test_multiple_choice_prompt_carries_null_result_clause(self) -> None:
        self._assert_forecaster_clause_present(multiple_choice_prompt(_mc_q(), research="r"))

    def test_numeric_prompt_carries_null_result_clause(self) -> None:
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        self._assert_forecaster_clause_present(result)

    def test_clause_sits_with_the_evidence_weighting_rubric(self) -> None:
        """The clause is an evidence-weighting rule, so it must land after the
        Strong/Moderate/Weak rubric rather than displacing it."""
        prompt = binary_prompt(_binary_q(), research="r")
        rubric_at = prompt.index("Weak: anecdotes")
        clause_at = prompt.index("Read a null search result")
        assert rubric_at < clause_at

    def test_stacking_prompts_do_not_carry_the_clause(self) -> None:
        """Scope guard: the clause ships to the base prompts only (stacking is
        prod-disabled, so the diff stays minimal)."""
        stacked = stacking_binary_prompt(_binary_q(), research="r", base_predictions=["a1", "a2"])
        assert "Read a null search result" not in stacked

    def test_gap_fill_analyzer_carries_the_auditor_version(self) -> None:
        """Auditor phrasing, not forecaster phrasing: the analyzer must not bank a
        first-pass "found nothing" as an established negative, and must point the
        gap at the authoritative place that would hold the record."""
        prompt = gap_fill_analyzer_prompt(
            "Will the permit be granted?",
            "Resolves YES if the regulator lists the permit.",
            "See the regulator's public register.",
            "First pass: no authoritative public record found.",
            is_benchmarking=False,
        )
        lowered = " ".join(prompt.lower().split())
        assert "null results are search outcomes" in lowered
        assert "not as an established negative fact" in lowered
        assert "name that source in the search query" in lowered


class TestOutsideViewRubricRules:
    """The outside-view corrections from the 2026-09-01 residual round, each stated ONCE.

    qid 43837: six members applied a monthly announcement rate over the FULL question
    window when 16 days had already elapsed event-free, then OR-ed it with a specific
    scheduled path the rate already contained. The correction survives in two places
    that were already there: the remaining-exposure sentence rides the binary
    conditional-hazard bullet (and stands alone as one MC bullet, since MC has no
    hazard bullet), and the disjointness half lives in the binary union line. The
    standalone `_REMAINING_EXPOSURE_RULE` that restated both was retired.

    qid 44557: four of six wrote a 17-25% base rate and published 35-55% on soft
    schedule signals. What survives is the SIZE: the existing "Anchor on your math"
    bullet now says a move of more than about 15 points needs a named reason. The
    standalone `_ANCHOR_CONSISTENCY_RULE` was retired — its first bullet was the fourth
    request to state the anchor, and its second ("do not move off your number on a
    general feeling that history counsels caution") priced at zero on the whole archive
    and would suppress good moves.

    qid 44561: all six built a "no failure announced yet, so Poisson(1.0)" schedule
    model instead of the pooled FDIC failure rate; `_COUNT_IN_PERIOD_REFERENCE_CLASS`
    stays verbatim in all three prompts.

    Receipts: scratch/prompt_bloat_audit_2026-09-02.md, scratch/prompt_debloat_2026-09-02/receipts.md."""

    # -- remaining exposure (binary, MC): once each --------------------------

    _EXPOSURE_KEY = "rates apply to the exposure that remains"

    def _assert_remaining_exposure_once(self, prompt: str) -> None:
        flat = _flat(prompt)
        assert flat.count(self._EXPOSURE_KEY) == 1, "the exposure rule must be stated exactly once"
        assert "apply it from now until the deadline" in flat
        assert "treating the elapsed event-free part of the window as observed" in flat
        # The retired constant's phrasings must not come back beside the surviving statement.
        assert "outside-view rates apply to the exposure" not in flat
        assert "the two must be disjoint" not in flat
        assert "remove that path's own instances from the rate before combining" not in flat

    def test_binary_prompt_states_remaining_exposure_once_inside_the_hazard_bullet(self) -> None:
        prompt = _binary_prompt_text()
        self._assert_remaining_exposure_once(prompt)
        # Folded INTO the conditional-hazard bullet, which is the same rule for recurring events.
        flat = _flat(prompt)
        bullet_at = flat.index(self._EXPOSURE_KEY)
        window = flat[bullet_at : bullet_at + 900]
        # The check keeps its label: without it the trailing skip token names something the
        # bullet never introduced, and "Otherwise write ..." reads as scoping over the
        # exposure sentence, which applies to every rate-based question and is never skipped.
        assert "conditional-hazard check: for a recurring event with a history of inter-arrival gaps" in window
        assert "no event in the t days already elapsed" in window
        assert "non-recurring, conditional-hazard skipped" in window

    def test_multiple_choice_prompt_states_remaining_exposure_once(self) -> None:
        self._assert_remaining_exposure_once(_mc_prompt_text())

    def test_binary_union_line_requires_disjoint_paths(self) -> None:
        """The union instruction itself is what invited the qid 43837 double count, so
        the disjointness requirement lives where the union is computed — and only there,
        since MC never computes a union."""
        flat = _flat(_binary_prompt_text())
        assert "1 - product of (1-p_i)" in flat
        assert "union only over paths that cannot be the same event" in flat
        assert "union only over paths" not in _flat(_mc_prompt_text())

    # -- anchor adherence (binary, MC): one rule, with a size ----------------

    _ANCHOR_SIZE = "more than about 15 points"

    def _assert_single_anchor_rule_with_size(self, prompt: str) -> None:
        flat = _flat(prompt)
        assert "anchor on your math" in flat
        assert flat.count(self._ANCHOR_SIZE) == 1, "the 15-point size belongs to exactly one bullet"
        assert "named, specific" in flat
        # The retired `_ANCHOR_CONSISTENCY_RULE` phrasings.
        assert "state the outside-view number you computed" not in flat
        assert "more than about 15 percentage points" not in flat
        assert "do not move off your own number on a general feeling" not in flat
        assert "history counsels caution" not in flat

    def test_binary_prompt_has_one_anchor_rule_sized_at_15_points(self) -> None:
        self._assert_single_anchor_rule_with_size(_binary_prompt_text())

    def test_multiple_choice_prompt_has_one_anchor_rule_sized_at_15_points(self) -> None:
        self._assert_single_anchor_rule_with_size(_mc_prompt_text())

    # -- count-in-period reference class (all three types) -----------------

    def _assert_count_in_period(self, prompt: str) -> None:
        flat = _flat(prompt)
        assert "how many events of a kind occur in a period" in flat
        assert "pooled realized rate of that event over the longest comparable history" in flat
        # A known-candidate schedule updates the rate rather than replacing it.
        assert "it does not replace it" in flat

    def test_binary_prompt_carries_count_in_period_class(self) -> None:
        self._assert_count_in_period(_binary_prompt_text())

    def test_multiple_choice_prompt_carries_count_in_period_class(self) -> None:
        self._assert_count_in_period(_mc_prompt_text())

    def test_numeric_prompt_carries_count_in_period_class(self) -> None:
        """Count questions arrive as all three question types, so this one rule ships to
        the numeric prompt too — unlike the exposure rule, which is probability-shaped."""
        self._assert_count_in_period(_numeric_prompt_text())

    # -- placement + scope -------------------------------------------------

    def test_outside_view_rules_sit_in_the_outside_view_phase(self) -> None:
        """Both outside-view rules must land inside PHASE 1, before the inside-view update,
        so the model reads them while computing the number. The anchor size rides the
        final-rationale bullet in PHASE 2, where the final number is written."""
        prompt = _binary_prompt_text()
        phase1_at = prompt.index("PHASE 1: OUTSIDE VIEW")
        phase2_at = prompt.index("PHASE 2: INSIDE VIEW UPDATE")
        for phrase in ("For questions asking how many events", "Rates apply to the exposure that REMAINS"):
            assert phase1_at < prompt.index(phrase) < phase2_at, f"{phrase!r} outside PHASE 1"
        assert prompt.index("Anchor on your math") > phase2_at

    def test_stacking_prompts_do_not_carry_the_rules(self) -> None:
        """Same scope guard as the null-result clause: base prompts only, since stacking
        is prod-disabled."""
        for prompt in _stacked_prompt_texts():
            flat = _flat(prompt)
            assert self._EXPOSURE_KEY not in flat
            assert self._ANCHOR_SIZE not in flat
            assert "anchor on your math" not in flat
            assert "how many events of a kind occur in a period" not in flat

    def test_numeric_prompt_does_not_carry_the_probability_shaped_rules(self) -> None:
        """Scope: the exposure rule and the 15-point size are worded in probabilities, while
        the numeric prompt anchors on a range. Only the reference-class rule ships there."""
        flat = _flat(_numeric_prompt_text())
        assert self._EXPOSURE_KEY not in flat
        assert self._ANCHOR_SIZE not in flat


class TestSoftClockAndHistoryDischargedRules:
    """The two Phase 1 rules the 2026-09-02 failure-mode audit produced
    (scratch/failure_mode_audit_2026-09-02/AUDIT_SYNTHESIS.md; Items A and C of
    scratch_docs_and_planning/announced_unscheduled_fix_plan_2026-09-02.md).

    `_SOFT_CLOCK_RULE` (lens A, the announced-but-unscheduled event). On "will X happen before
    D" questions whose only route to X is a target date the responsible actor has ANNOUNCED but
    is not BOUND to, members decomposed P(target lands in window) x P(X | target) and set the
    first term near 1 because the target had been announced. 52 of 815 STRICT records (6.4%;
    8.3% of binaries; coder kappa 0.74); on the 37 flagged binaries the bot published a mean 0.44
    for events that happened 8% of the time, 13 records above 0.5 resolved NO and none went the
    other way, and flagged records score 18.7 spot-peer points worse (95% CI 5.9 to 33.4).
    Receipts 43837 / 44424 / 44557; the contrast is 45217, where a statutory clock existed,
    members computed the date and scored +45. The "measured record of meeting" carve-out is
    load-bearing: on qid 42305 a weekly bulletin with a measured publication lag WAS a binding
    clock and a near-1 timing term was right. It supersedes the two 2026-09-02 rules Item B
    retired, and it deliberately adds no structured-block field.

    `_HISTORY_DISCHARGED_RULE` (lens C, history repeats past an acknowledged change). A member
    names a reason the historical cadence has been discharged and keeps it as the center anyway:
    12.1% of rationales, about 7 spot per flagged record (95% CI 2.7 to 12.2), the pattern failed
    in 13 of 13 live-triple fires; coder agreement 0.59 and partly hindsight-contaminated, so
    upper bounds. Shipped on the plan's recommendation with the operator's final say pending.

    Both ship to binary and MC only and sit in Phase 1 beside the reference-class bullets."""

    _RUBRIC = "Strong: multiple independent sources"
    _COUNT_IN_PERIOD = "For questions asking how many events"
    _SOFT_CLOCK_OPENER = "A target date the responsible actor has not bound itself to"
    _HISTORY_OPENER = "If your own analysis names a reason the historical cadence has been discharged"

    def _assert_verbatim_once(self, prompt: str, constant: str) -> None:
        flat_prompt = _flat(prompt)
        flat_constant = _flat(constant)
        assert flat_constant in flat_prompt, "the rule must land verbatim (modulo line wrapping)"
        assert flat_prompt.count(flat_constant) == 1, "the rule must be stated exactly once"

    # -- presence, verbatim, once (binary and MC) -----------------------------

    def test_binary_prompt_carries_the_soft_clock_rule_verbatim(self) -> None:
        self._assert_verbatim_once(_binary_prompt_text(), _SOFT_CLOCK_RULE)

    def test_multiple_choice_prompt_carries_the_soft_clock_rule_verbatim(self) -> None:
        self._assert_verbatim_once(_mc_prompt_text(), _SOFT_CLOCK_RULE)

    def test_binary_prompt_carries_the_history_discharged_rule_verbatim(self) -> None:
        self._assert_verbatim_once(_binary_prompt_text(), _HISTORY_DISCHARGED_RULE)

    def test_multiple_choice_prompt_carries_the_history_discharged_rule_verbatim(self) -> None:
        self._assert_verbatim_once(_mc_prompt_text(), _HISTORY_DISCHARGED_RULE)

    # -- wording pins: each load-bearing clause, by name ----------------------

    def test_soft_clock_rule_keeps_every_load_bearing_clause(self) -> None:
        """Each clause earns its place: the carve-out keeps a measured cadence (q42305) from
        being read as soft; "as its own number" is the move the audit measured; the list of
        things that do NOT raise it is the q44557 / q43837 evidence class; "say which clock"
        is what q45217's members did; the parenthetical is the receipt (principle 3 of the fix
        plan: a receipted rule carries its reason)."""
        flat = _flat(_SOFT_CLOCK_RULE)
        assert "no statute, no contract, no published schedule it has a measured record of meeting" in flat
        assert "is evidence that a target exists, not that it will hold" in flat
        assert "price the probability that the target lands inside the question window as its own number" in flat
        assert "derived from that actor's record of slips and scrubs for this kind of event" in flat
        assert "an announcement, plan, tracker page or partner page does not raise it" in flat
        assert "where a binding clock exists, compute the date from it and say which clock" in flat
        assert "forecasts averaged 44% on events that happened 8% of the time" in flat

    def test_history_discharged_rule_is_conditional_on_the_members_own_acknowledgment(self) -> None:
        """The condition is the member's OWN written acknowledgment, so the rule cannot fire on
        a question where nothing has changed; and the cadence becomes a BOUND, not a center,
        which is the whole correction. The receipt rides as a short parenthetical, and it names
        its own sample: the label had coder agreement 0.59 and partial hindsight contamination, so
        a bare "0 of 13" would read to a forecaster as a law when it is an ordinary draw at the
        17-18% held rate the older era bands show."""
        flat = _flat(_HISTORY_DISCHARGED_RULE)
        assert flat.startswith("• if your own analysis names a reason the historical cadence has been discharged")
        assert "(its driver was met, the deadline passed, the rule changed)" in flat
        assert "that cadence is a bound on your estimate, not its center" in flat
        assert "state the post-change estimate and what it rests on" in flat
        assert "in a small audit of this bot's own past forecasts" in flat
        assert "held in none of 13 recent cases" in flat

    def test_rules_do_not_revive_the_retired_anchor_consistency_wording(self) -> None:
        """Item B retired "do not move off your number when history counsels caution" because it
        pulled against exactly this correction; the two new rules must not smuggle it back."""
        for prompt in (_binary_prompt_text(), _mc_prompt_text()):
            flat = _flat(prompt)
            assert "history counsels caution" not in flat
            assert "do not move off your own number on a general feeling" not in flat

    # -- placement: Phase 1, after the reference-class bullets, before the rubric ----

    @pytest.mark.parametrize(
        ("build", "timeframe_step"),
        [
            pytest.param(_binary_prompt_text, "3) Timeframe reasoning", id="binary"),
            pytest.param(_mc_prompt_text, "(3) Timeframe reasoning", id="multiple_choice"),
        ],
    )
    def test_rules_sit_in_the_reference_class_step_of_phase_1(self, build, timeframe_step: str) -> None:
        """Both rules are about how to read a base rate you just computed, so they land inside
        the reference-class step (after `_COUNT_IN_PERIOD_REFERENCE_CLASS`, before the timeframe
        step), inside PHASE 1 and before the Strong/Moderate/Weak evidence rubric in PHASE 2.
        Soft-clock first, history-discharged directly after it."""
        prompt = build()
        phase1_at = prompt.index("PHASE 1: OUTSIDE VIEW")
        count_at = prompt.index(self._COUNT_IN_PERIOD)
        soft_clock_at = prompt.index(self._SOFT_CLOCK_OPENER)
        history_at = prompt.index(self._HISTORY_OPENER)
        timeframe_at = prompt.index(timeframe_step)
        rubric_at = prompt.index(self._RUBRIC)
        phase2_at = prompt.index("PHASE 2: INSIDE VIEW UPDATE")
        assert phase1_at < count_at < soft_clock_at < history_at < timeframe_at < phase2_at < rubric_at

    # -- scope: not numeric, not stacking, no block field ---------------------------

    def test_numeric_prompt_does_not_carry_the_rules(self) -> None:
        """Both rules are worded in probabilities over an event-by-deadline question; the
        numeric prompt anchors on a range and gets neither."""
        flat = _flat(_numeric_prompt_text())
        assert _flat(self._SOFT_CLOCK_OPENER) not in flat
        assert _flat(self._HISTORY_OPENER) not in flat
        assert "measured record of meeting" not in flat
        assert "historical cadence has been discharged" not in flat

    def test_stacking_prompts_do_not_carry_the_rules(self) -> None:
        """Same scope guard as every other base-prompt rule: stacking is prod-disabled."""
        for prompt in _stacked_prompt_texts():
            flat = _flat(prompt)
            assert _flat(self._SOFT_CLOCK_OPENER) not in flat
            assert _flat(self._HISTORY_OPENER) not in flat
            assert "measured record of meeting" not in flat
            assert "historical cadence has been discharged" not in flat

    def test_rules_add_no_structured_block_field(self) -> None:
        """The schema audit rejected a `target_holds_probability` / `clock_type` slot: the block
        is written after the forecast is fixed, so a field there cannot scaffold the reasoning,
        and the number lives in the rationale. The example blocks stay exactly the forecast."""
        assert set(json.loads(_extract_last_json_block(_binary_prompt_text()))) == {"question_type", "posterior_prob"}
        assert set(json.loads(_extract_last_json_block(_mc_prompt_text()))) == {"question_type", "option_probs"}
        for constant in (_SOFT_CLOCK_RULE, _HISTORY_DISCHARGED_RULE):
            assert "target_holds_probability" not in constant
            assert "clock_type" not in constant

    def test_rules_keep_the_base_prompts_mode_agnostic(self) -> None:
        """The benchmarking leakage guard lives on the research side; the forecaster prompts stay
        mode-agnostic, so neither rule may name a market venue or crowd source."""
        for constant in (_SOFT_CLOCK_RULE, _HISTORY_DISCHARGED_RULE):
            for venue in ("Polymarket", "Kalshi", "Manifold", "PredictIt", "Metaculus", "CME FedWatch"):
                assert venue not in constant


class TestStatusQuoDerivation:
    """Every forecaster prompt must open PHASE 0 with a mandatory status-quo
    DERIVATION — a question the model answers itself before reviewing any
    research — not a rule/warning. Rules-as-warnings get argued away as
    boilerplate (qid 42024: 4/5 models dismissed the "not yet satisfied"
    line); a derivation the model writes in its own words is stickier."""

    def _assert_derivation_present(self, prompt: str) -> None:
        lowered = " ".join(prompt.lower().split())
        assert "status-quo derivation" in lowered
        # The model must state the platform-state premise in its own words...
        assert "open and unresolved as of" in lowered
        # ...with today's date interpolated so the statement is concrete. The
        # prompt uses UTC (see prompts._forecasting_window_str, which uses
        # datetime.now(timezone.utc) to stay tz-aware against ft's aware question
        # datetimes), so assert the UTC date too — a naive datetime.now() flakes
        # in the evening-local/next-day-UTC window.
        assert datetime.now(UTC).strftime("%Y-%m-%d") in prompt
        # Moving off the status quo requires naming a post-open trigger.
        assert "post-open event" in lowered
        # And an explicit commitment about the window.
        assert "no qualifying event has yet occurred inside the window" in lowered

    def _assert_derivation_before_outside_view(self, prompt: str) -> None:
        idx_derivation = prompt.find("Status-quo derivation")
        idx_phase1 = prompt.find("PHASE 1")
        assert idx_derivation >= 0
        assert idx_phase1 >= 0
        assert idx_derivation < idx_phase1, "status-quo derivation must come before the outside view"

    def test_binary_prompt_has_status_quo_derivation_first(self) -> None:
        prompt = binary_prompt(_binary_q(), research="r")
        self._assert_derivation_present(prompt)
        self._assert_derivation_before_outside_view(prompt)
        # In binary, the derivation must be the TOP of PHASE 0 — before the resolution check.
        assert prompt.find("Status-quo derivation") < prompt.find("Resolution check")

    def test_multiple_choice_prompt_has_status_quo_derivation_first(self) -> None:
        prompt = multiple_choice_prompt(_mc_q(), research="r")
        self._assert_derivation_present(prompt)
        self._assert_derivation_before_outside_view(prompt)

    def test_numeric_prompt_has_status_quo_derivation_first(self) -> None:
        prompt = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        self._assert_derivation_present(prompt)
        self._assert_derivation_before_outside_view(prompt)

    def test_binary_resolution_check_retained_after_derivation(self) -> None:
        """Regression: adding the derivation must not displace the 0a resolution
        check or the 0b decomposition."""
        prompt = binary_prompt(_binary_q(), research="r")
        assert "Resolution check" in prompt
        assert "Resolution decomposition" in prompt


class TestConjunctiveCriteriaPricing:
    """Change #2: the binary prompt upgrades the qualitative Boolean-product
    decomposition with a NUMERIC pricing table — placed LATE in the flow
    (after evidence review and red-team) so the red-team can causally affect
    the clause probabilities. PHASE 0b keeps only the qualitative listing."""

    def test_binary_prompt_has_pricing_table_section(self) -> None:
        prompt = binary_prompt(_binary_q(), research="r")
        lowered = " ".join(prompt.lower().split())
        assert "conjunctive criteria pricing" in lowered
        assert "one row per resolution clause" in lowered
        assert "product" in lowered

    def test_pricing_table_comes_after_red_team_and_before_final_rationale(self) -> None:
        """ORDERING IS LOAD-BEARING: clause numbers must be emitted after the
        red-team/bear-bull section so red-teaming can move them, and before
        the final rationale that reconciles against the product."""
        prompt = binary_prompt(_binary_q(), research="r")
        idx_red_team = prompt.find("Red-team both")
        idx_pricing = prompt.find("Conjunctive criteria pricing")
        idx_final = prompt.find("Final rationale and calibration")
        assert 0 <= idx_red_team < idx_pricing < idx_final

    def test_phase0b_keeps_listing_but_defers_numbers(self) -> None:
        """PHASE 0b still lists/structures the clauses (early structure
        identification is fine) but must explicitly defer the probabilities
        to the late pricing step."""
        prompt = binary_prompt(_binary_q(), research="r")
        assert "Resolution decomposition" in prompt
        assert "Boolean product" in prompt
        lowered = " ".join(prompt.lower().split())
        assert "do not assign probabilities to the clauses yet" in lowered
        # The deferral must appear inside 0b, i.e. before PHASE 1.
        idx_defer = prompt.find("Do NOT assign probabilities to the clauses yet")
        assert 0 <= idx_defer < prompt.find("PHASE 1")

    def test_reconciliation_requires_named_clause_dependence(self) -> None:
        """pgodzinai 42855 failure mode: a computed clause product coexisting
        with free-form narrative adjustment gets nullified (82% computed →
        87% via 'season-specific upward adjustment'). Any deviation from the
        product must operate through the clause probabilities themselves, a
        named clause dependence, or a corrected clause decomposition; overrides
        that route around the clauses are explicitly forbidden. Kept through the
        2026-09 de-bloat by operator decision, with its reason attached: the
        criteria stay consumed as constraints."""
        prompt = binary_prompt(_binary_q(), research="r")
        lowered = " ".join(prompt.lower().split())
        assert "you have exactly three valid moves" in lowered
        assert "revise the clause probabilities themselves and recompute" in lowered
        assert "name a specific dependence between clauses" in lowered
        # The third sanctioned move: the decomposition itself was wrong.
        assert "revise the clause decomposition from 0b" in lowered
        assert (
            "all hedging and adjustment must operate through the clauses, their dependence, or a corrected "
            "decomposition, not around them" in lowered
        )
        assert "so the criteria stay consumed as constraints" in lowered
        assert "if none applies, stay at the product" in lowered

    def test_clause_product_is_named_as_the_anchor_on_multi_clause_questions(self) -> None:
        """Three rules used to say how the final number may differ from a computed one and
        none said WHICH computation anchors when a Step-2 base rate and a 5b clause product
        both exist. One sentence where the product is computed now says: the product, because
        it is more specific than the step-2 base rate, which is the comparand a bare "more
        specific" left the model to infer; the three valid moves are the ways to leave it."""
        prompt = binary_prompt(_binary_q(), research="r")
        lowered = " ".join(prompt.lower().split())
        assert 'this product is the number the "anchor on your math" check in step 6 anchors to' in lowered
        assert "because it is more specific than the step-2 base rate" in lowered
        # The anchor bullet names the product among the computations it covers.
        assert "clause product" in lowered
        # And the sentence sits in 5b, before the reconciliation bullet.
        assert prompt.index("more specific than the step-2 base rate") < prompt.index("exactly three valid moves")


class TestTemplateStatesEachCheckOnce:
    """The binary and MC templates said several things twice: an odds check and a small-delta
    check that both asked how a ±10% shift would sit; a trailing "Brief checklist" whose items
    re-asked for outputs the template had already produced (0b paraphrases the criteria, step 2
    states the base rate, step 4 lists the evidence, step 5 red-teams). The two checklist items
    with no template twin, the bait-and-switch check and the consistency line, moved INTO the
    template as its final numbered step: an instruction that shapes the answer's structure
    belongs in the template, not in a post-hoc reminder. MC also carried three statements of
    "sum to 100%", two of them in integer percent while the schema demands decimals summing to
    1.0; the schema line is the one the parser reads, so it is the one that stays. The
    every-option floor was on the percent scale too ("a probability (1-99%)", "at least 1%") and
    now interpolates MC_PROB_MIN, since a percent-scale option_probs is hard-rejected and drops
    the whole ballot to the paid LLM salvage rung (q44558)."""

    def test_final_checks_is_the_last_template_step_in_every_base_prompt(self) -> None:
        for prompt in (_binary_prompt_text(), _mc_prompt_text(), _numeric_prompt_text()):
            flat = _flat(prompt)
            assert flat.count("bait-and-switch check") == 1
            assert "brief checklist" not in flat
            # Retired checklist echoes of template outputs.
            assert "paraphrase the resolution criteria" not in flat
            assert "paraphrase options" not in flat
            assert "top 3-5 evidence items" not in flat
            assert "top 3 to 5 evidence items" not in flat
            assert "blind-spot scenario most likely" not in flat
            assert "blind-spot statement" not in flat
            assert "blind spot scenario and expected effect" not in flat
            assert "state the outside-view base rate you anchored to" not in flat
            assert "state the outside-view distribution used as anchor" not in flat
            assert "state the outside view baseline used" not in flat
            # Placement: inside the template, after the final-rationale step, before the block.
            final_checks_at = flat.index("final checks")
            assert flat.index("final rationale") < final_checks_at < flat.index("structured forecast")

    def test_binary_final_checks_keep_the_consistency_line(self) -> None:
        flat = _flat(_binary_prompt_text())
        assert 'consistency line: "x out of 100 times, [criteria] happens." sensible?' in flat

    def test_mc_final_checks_keep_the_consistency_line(self) -> None:
        flat = _flat(_mc_prompt_text())
        assert "most likely: __; least likely: __; coherent with rationale?" in flat

    def test_numeric_final_checks_keep_units_and_the_percentile_consistency_line(self) -> None:
        """Numeric adds the units bullet: the unit-mismatch guard withholds a forecaster that gets
        the units wrong, so it is the one checklist item with a pipeline consequence."""
        flat = _flat(_numeric_prompt_text())
        final_checks = flat[flat.index("final checks") :]
        assert "units: what are the units of the output values and why?" in final_checks
        assert "which percentile corresponds to the status quo or trend" in final_checks

    def test_numeric_step_seven_asks_for_a_central_estimate_not_a_probability(self) -> None:
        """ "My base rate was X% ... moving to Y%" was a probability template copy-pasted onto a
        distribution question (the sibling numeric odds check was cut for the same reason)."""
        flat = _flat(_numeric_prompt_text())
        assert (
            "state your outside-view central estimate and range, then say what the current evidence moved and why"
            in flat
        )
        assert "my base rate was x%" not in flat
        assert "odds check" not in flat

    def test_numeric_outcome_type_is_defined_once_in_the_schema_notes(self) -> None:
        """The field was defined three times (step 9, schema notes, example). The schema note is
        the definition; step 9 is a one-line pointer to it."""
        flat = _flat(_numeric_prompt_text())
        assert "outcome type classification" not in flat
        assert flat.count("counts, rankings, number of events") == 1
        assert "record it in `outcome_type`" in flat
        assert "definition in the schema notes" in flat

    def test_odds_and_delta_are_one_check(self) -> None:
        for prompt in (_binary_prompt_text(), _mc_prompt_text()):
            flat = _flat(prompt)
            assert flat.count("odds and delta check") == 1
            assert "odds check:" not in flat
            assert "small-delta check" not in flat
            assert "9:1" in flat

    def test_mc_states_the_sum_constraint_once_in_decimals(self) -> None:
        prompt = _mc_prompt_text()
        flat = _flat(prompt)
        assert "sum to 1.0" in flat
        assert "sum to 100" not in flat
        assert "use integers" not in flat
        assert "remember:" not in flat
        # The every-option requirement is extraction-critical and stays, but on the DECIMAL
        # scale: a percent-scale option_probs is hard-rejected by _check_option_probs and
        # cannot be repaired, so the whole ballot drops to the paid LLM salvage rung.
        assert "you must assign a probability to every single option" in flat
        assert "assign it at least 0.01" in flat
        assert "1-99%" not in flat


class TestKeptScaffoldingBullets:
    """Six bullets the 2026-09-02 de-bloat deliberately KEPT as reasoning scaffolding but
    that no other test named, so a later cut could have taken any of them and stayed green.
    AGENTS.md claims every surviving rule has a presence pin on its wording; these are the
    pins that make that claim true. Each phrase is the bullet's own opening clause, so a
    reword lands here rather than in a vague substring."""

    def test_binary_worked_examples_keep_their_reason(self) -> None:
        """Step 0b's meta-justification is an operator keep: it tells the model WHY it is
        writing two worked examples, which is what stops them becoming a paraphrase."""
        assert (
            "this is mechanical bait-and-switch protection: it forces the resolution criteria to be "
            "consumed as structured constraints rather than treated as a prose paraphrase"
            in _flat(_binary_prompt_text())
        )

    def test_binary_and_numeric_keep_the_question_specific_base_rate_bullet(self) -> None:
        assert (
            "question-specific base rate: the relevant base rate is the historical frequency for questions "
            "like this one" in _flat(_binary_prompt_text())
        )
        assert (
            "question-specific base rate: anchor on the historical frequency, trend, or variance for this "
            "specific indicator" in _flat(_numeric_prompt_text())
        )

    def test_binary_keeps_the_trajectory_check(self) -> None:
        """Kept in binary and removed from numeric, where it restated step-3 trend
        continuation. The numeric absence is pinned by
        test_numeric_status_quo_derivation_is_the_one_anchor_to_latest_statement."""
        flat = _flat(_binary_prompt_text())
        assert (
            'trajectory check: consider whether the "status quo" means "nothing changes" or '
            '"the current trajectory reaches its natural conclusion"' in flat
        )
        assert "justify predictions that diverge from the most likely trajectory" in flat

    def test_mc_keeps_the_blind_spot_and_calibration_audit_bullets(self) -> None:
        flat = _flat(_mc_prompt_text())
        assert (
            "blind-spot consideration: if the resolution is unexpected, what would likely be the reason, "
            "and how should that affect confidence spreads?" in flat
        )
        assert "calibration audit: if one option is genuinely dominant, commit to it" in flat

    def test_numeric_keeps_the_small_delta_check(self) -> None:
        """Numeric's own delta check survives the binary/MC odds-and-delta merge: there is no
        odds check on a distribution question for it to merge with."""
        assert "small delta check: would +/- 10 percent on key percentiles still fit the reasoning?" in _flat(
            _numeric_prompt_text()
        )


class TestResolutionMetricEcho:
    """PHASE 0 resolution-metric echo: when the resolution criteria name an
    official statistical series, force the forecaster to name the exact
    resolving series and enumerate its variants (component vs total, etc.)
    before forecasting. Motivated by qid 44211 — all six models priced the
    USBP-apprehensions *component* of a series that resolves on the *total*,
    even though the research carried the wedge, the historical conversion, and
    an explicit provider warning. Scoped to the numeric prompt (which also
    serves discrete-integer questions — outcome_type is decided in the block,
    there is no separate discrete prompt) and the binary prompt. MC and the
    prod-disabled stacking prompts are deliberately untouched."""

    _HEADER = "resolution-metric echo (named-series questions only)"
    _INERT = "no named series, metric echo skipped"

    def _assert_core_block(self, prompt: str) -> None:
        c = _flat(prompt)
        assert self._HEADER in c
        assert "name the exact series that resolves this question" in c
        assert "enumerate the plausible variants" in c
        assert "component vs total" in c
        # Inert escape: questions with no named series skip the step.
        assert self._INERT in c
        # Don't-discard-on-one-implausible-estimate: the gemini-31k poisoning lesson.
        assert "do not discard a candidate variant" in c
        assert "recompute the candidate from its components" in c

    def test_numeric_prompt_has_metric_echo(self) -> None:
        prompt = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        self._assert_core_block(prompt)
        c = _flat(prompt)
        # Numeric reconciles against the displayed range — but NOT as an oracle
        # (the 44211 trap: the true total was the bounds midpoint).
        assert "reconcile each candidate against the displayed range" in c
        assert 'do not read "inside the range" as confirming' in c
        # Points at both numeric-available research sections (synergy, not duplication).
        assert "## Resolution Source Snapshot" in prompt
        assert "## Time Series Anchor" in prompt

    def test_binary_prompt_has_metric_echo(self) -> None:
        prompt = binary_prompt(_binary_q(), research="r")
        self._assert_core_block(prompt)
        c = _flat(prompt)
        # Binary reconciles against the criteria's threshold/comparison (no displayed range).
        assert "reconcile each candidate against the threshold or comparison" in c
        # Points at the resolution-source snapshot only — the TS anchor is numeric-only.
        assert "## Resolution Source Snapshot" in prompt
        assert "## Time Series Anchor" not in prompt

    def test_binary_metric_echo_after_decomposition_before_phase1(self) -> None:
        prompt = binary_prompt(_binary_q(), research="r")
        idx_decomp = prompt.find("Resolution decomposition")
        idx_echo = prompt.find("Resolution-metric echo")
        idx_phase1 = prompt.find("PHASE 1")
        assert 0 <= idx_decomp < idx_echo < idx_phase1

    def test_numeric_metric_echo_after_status_quo_before_phase1(self) -> None:
        prompt = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        idx_sq = prompt.find("Status-quo derivation")
        idx_echo = prompt.find("Resolution-metric echo")
        idx_phase1 = prompt.find("PHASE 1")
        assert 0 <= idx_sq < idx_echo < idx_phase1

    def test_mc_prompt_omits_metric_echo(self) -> None:
        # MC questions resolve to an enumerated option, not a value read off a
        # named statistical series, so the component-vs-total ambiguity does not
        # arise. The measured miss family (44211 numeric, 42018/41801 binary)
        # contains zero MC cases — scoping decision locked here.
        prompt = multiple_choice_prompt(_mc_q(), research="r")
        assert "Resolution-metric echo" not in prompt

    def test_stacking_prompts_omit_metric_echo(self) -> None:
        # Stacking is prod-disabled; those prompts are left untouched.
        binary = stacking_binary_prompt(_binary_q(), research="r", base_predictions=["a1", "a2"])
        mc = stacking_multiple_choice_prompt(_mc_q(), research="r", base_predictions=["a1", "a2"])
        numeric = stacking_numeric_prompt(
            _numeric_q(),
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message="lbm",
            upper_bound_message="ubm",
        )
        for p in (binary, mc, numeric):
            assert "Resolution-metric echo" not in p
