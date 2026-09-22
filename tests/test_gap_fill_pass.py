"""Tests for the second-pass gap-fill pipeline in ``targeted_research``.

Mocks ``_run_analyzer`` at the module level and ``build_native_search_llm`` (the
per-gap resolver now runs OpenAI native web search via OpenRouter). The helper
``_patch_resolver`` returns an LLM whose ``.invoke`` is the supplied AsyncMock,
so each test still captures the per-gap prompt and controls the result. No live
API calls.
"""

import asyncio
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import MetaculusQuestion

from metaculus_bot.constants import GAP_FILL_MAX_GAPS
from metaculus_bot.research.targeted import (
    DROP_ALREADY_ANSWERED,
    DROP_NOT_ANSWERABLE,
    DROP_OVER_CAP,
    DROP_REASONS,
    DROP_SAME_NEED,
    DROP_SCHEMA,
    _parse_gap_list,
    _run_analyzer,
    run_gap_fill_pass,
    triage_gaps,
)
from scripts.telemetry.markers import MARKER_SPECS


@dataclass
class MockQuestion:
    """Minimal stand-in for MetaculusQuestion for gap-fill tests.

    Duck-typed to match the attribute access in ``run_gap_fill_pass``: the analyzer and the
    per-gap resolver both read ``question_text``, ``resolution_criteria`` and ``fine_print``.
    """

    question_text: str = "Will X happen by 2026?"
    resolution_criteria: str | None = "Resolves YES if X happens before Dec 31, 2026."
    fine_print: str | None = "See bls.gov for data."
    id_of_question: int = 42
    page_url: str = "https://example.com/q/42"
    # The MC ballot, None on non-MC questions, matching MetaculusQuestion; read via getattr at the call site.
    options: list[str] | None = None

    _unused: dict[str, str] = field(default_factory=dict)


def _q(mock: MockQuestion) -> MetaculusQuestion:
    """Cast a MockQuestion to MetaculusQuestion for static type checkers.

    The real code only uses duck-typed attribute access, so this is a runtime
    no-op; it exists solely to keep Pyright happy about the function signature.
    """
    return cast(MetaculusQuestion, mock)


def _gap(
    text: str,
    search_query: str | None = None,
    why_matters: str = "",
    *,
    answerable_now: object = True,
    already_in_first_pass: object = False,
    same_need_as: object = None,
) -> dict[str, Any]:
    """An analyzer gap as ``_parse_gap_list`` emits it, graded to pass triage unless a grade is overridden."""
    return {
        "gap": text,
        "search_query": text if search_query is None else search_query,
        "why_matters": why_matters,
        "answerable_now": answerable_now,
        "already_in_first_pass": already_in_first_pass,
        "same_need_as": same_need_as,
    }


def _gap_without(field: str) -> dict[str, Any]:
    """An otherwise-passing gap with one grade key absent, as a model that omits null-valued keys emits it."""
    gap = _gap("g")
    del gap[field]
    return gap


@contextmanager
def _patch_resolver(invoke: AsyncMock) -> Iterator[MagicMock]:
    """Patch ``build_native_search_llm`` so the per-gap resolver uses ``invoke``.

    The resolver now does ``llm = build_native_search_llm(...); await llm.invoke(prompt)``.
    Tests historically asserted against the per-gap search AsyncMock directly, so
    we wrap it in a stub LLM whose ``.invoke`` is that AsyncMock — the prompt is
    still captured on ``invoke`` exactly as before. Yields the builder mock so
    callers can assert the model slug / reasoning_effort args.
    """
    stub_llm = MagicMock()
    stub_llm.invoke = invoke
    builder = MagicMock(return_value=stub_llm)
    with patch("metaculus_bot.research.targeted.build_native_search_llm", builder):
        yield builder


# ---------------------------------------------------------------------------
# _parse_gap_list unit tests
# ---------------------------------------------------------------------------


class TestParseGapList:
    """Cover the various shapes of analyzer output the parser must tolerate."""

    @pytest.mark.parametrize("raw", ["", "   \n  "])
    def test_empty_string_raises(self, raw: str) -> None:
        with pytest.raises(ValueError, match="empty response"):
            _parse_gap_list(raw)

    def test_plain_valid_json(self) -> None:
        raw = '{"gaps": [{"gap": "g1", "why_matters": "wm1", "search_query": "sq1"}]}'
        out = _parse_gap_list(raw)

        assert len(out) == 1
        assert out[0]["gap"] == "g1"
        assert out[0]["search_query"] == "sq1"
        assert out[0]["why_matters"] == "wm1"

    def test_json_fenced_code_block(self) -> None:
        raw = '```json\n{"gaps": [{"gap": "fenced gap", "why_matters": "wm", "search_query": "sq"}]}\n```'
        out = _parse_gap_list(raw)

        assert len(out) == 1
        assert out[0]["gap"] == "fenced gap"

    def test_json_with_trailing_commentary(self) -> None:
        raw = 'Here is the output:\n{"gaps": [{"gap": "g", "why_matters": "wm", "search_query": "sq"}]}\n\nHope that helps!'
        out = _parse_gap_list(raw)

        assert len(out) == 1
        assert out[0]["gap"] == "g"

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_gap_list("not json at all, just words")

    def test_a_slot_without_gap_text_is_kept_empty_for_triage_to_drop(self) -> None:
        """The parser never compacts the list: ``same_need_as`` is a position in the ANALYZER's list, so a
        skipped slot would shift every later pointer onto the wrong gap. An item with no gap text keeps
        its slot as an empty one and triage drops it as schema drift."""
        raw = '{"gaps": [{"why_matters": "wm", "search_query": "sq"}]}'

        out = _parse_gap_list(raw)

        assert out == [{"gap": "", "search_query": "sq", "why_matters": "wm"}]
        assert [d["reason"] for d in triage_gaps(out, max_gaps=4).dropped] == [DROP_SCHEMA]

    def test_a_non_dict_item_is_kept_as_an_empty_slot(self) -> None:
        raw = '{"gaps": ["just a string", {"gap": "g2", "search_query": "q2", "why_matters": "w2"}]}'

        out = _parse_gap_list(raw)

        assert out == [
            {"gap": "", "search_query": "", "why_matters": ""},
            {"gap": "g2", "search_query": "q2", "why_matters": "w2"},
        ]

    def test_gap_missing_search_query_falls_back_to_gap_text(self) -> None:
        raw = '{"gaps": [{"gap": "my gap text", "why_matters": "wm"}]}'
        out = _parse_gap_list(raw)

        assert len(out) == 1
        assert out[0]["gap"] == "my gap text"
        assert out[0]["search_query"] == "my gap text"

    def test_parser_never_clips(self) -> None:
        """The cap lives in ``triage_gaps`` and applies AFTER the grade filter, so a dropped gap never
        displaces a kept one; a parse-time clip would pre-empt that."""
        gap_objs = [_gap(f"g{i}", f"q{i}", f"wm{i}") for i in range(GAP_FILL_MAX_GAPS + 3)]

        out = _parse_gap_list(json.dumps({"gaps": gap_objs}))

        assert [g["gap"] for g in out] == [f"g{i}" for i in range(GAP_FILL_MAX_GAPS + 3)]

    def test_grade_fields_pass_through_verbatim(self) -> None:
        """The parser hands the three grade fields to triage exactly as the analyzer typed them; triage,
        not the parser, decides what a mistyped grade means."""
        raw = json.dumps(
            {
                "gaps": [
                    _gap("g", "q", "w", answerable_now=False, already_in_first_pass=True, same_need_as=1),
                    _gap("h", "r", "x", answerable_now="yes", already_in_first_pass=None, same_need_as="1"),
                ]
            }
        )

        out = _parse_gap_list(raw)

        assert out[0]["answerable_now"] is False
        assert out[0]["already_in_first_pass"] is True
        assert out[0]["same_need_as"] == 1
        assert out[1]["answerable_now"] == "yes"
        assert out[1]["already_in_first_pass"] is None
        assert out[1]["same_need_as"] == "1"

    def test_absent_grade_fields_stay_absent(self) -> None:
        """A pre-grade payload carries no grade keys, and the parser adds none, so triage can tell an
        omitted grade from an explicit null."""
        raw = '{"gaps": [{"gap": "g1", "why_matters": "wm1", "search_query": "sq1"}]}'

        out = _parse_gap_list(raw)

        assert out == [{"gap": "g1", "search_query": "sq1", "why_matters": "wm1"}]

    def test_unfenced_with_brace_inside_string_value(self) -> None:
        """F11: braces embedded in string values must not truncate extraction.

        The analyzer sometimes returns unfenced JSON with trailing commentary.
        A naive brace counter closes the object at the first `}` it sees —
        even one inside a string literal — producing invalid JSON that then
        silently parses to ``[]``. The string-literal-aware helper preserves
        the full payload.
        """
        raw = (
            "Here is the gap analysis:\n"
            '{"gaps": [{"gap": "What does the phrase \\"ends with a } brace\\" mean?", '
            '"search_query": "q1", "why_matters": "wm1"}]}\n\n'
            "Hope that helps!"
        )
        out = _parse_gap_list(raw)

        assert len(out) == 1
        # The gap text must include the brace that would have been chopped off.
        assert "} brace" in out[0]["gap"]
        assert out[0]["search_query"] == "q1"

    def test_unfenced_with_trailing_prose_containing_braces(self) -> None:
        """F16: trailing prose after a well-formed JSON object can contain
        braces of its own (e.g. "{a, b} covers edge cases"). The balanced-brace
        fallback must stop at the end of the first balanced block and not
        attempt to extend through trailing commentary, which would break the
        parser. The Wave 1 fix covered the string-literal-in-value case; this
        asserts the sibling trailing-brace case is handled too.
        """
        raw = '{"gaps":[{"gap":"g","search_query":"q","why_matters":"w"}]}\n\nNote: the set {a, b} covers edge cases'
        out = _parse_gap_list(raw)

        assert len(out) == 1
        assert out[0]["gap"] == "g"
        assert out[0]["search_query"] == "q"
        assert out[0]["why_matters"] == "w"


# ---------------------------------------------------------------------------
# triage_gaps unit tests
# ---------------------------------------------------------------------------


class TestTriageGaps:
    """The grade-based lean-out of gap-fill v1 (operator ruling, 2026-09-09). The analyzer grades every
    gap on three fields and code drops the failing ones before any resolver call. Receipt: about a
    third of v1's resolver calls bought nothing on the archive (future-dated asks 18% of gaps, re-fetched
    first-pass readings on 47% of the forced current-reading slots, a paraphrase pair on one question in
    three; scratch/cost_pass_2026-09-09/v1_gap_redundancy/REDUNDANCY.md), while a positional cap of 2
    dropped the useful gap on 4 of 6 traced questions, so the filter is by grade, never by position."""

    def test_all_four_pass(self) -> None:
        gaps = [_gap(f"g{i}") for i in range(1, 5)]

        triage = triage_gaps(gaps, max_gaps=4)

        assert triage.kept == gaps
        assert triage.dropped == []
        assert triage.listed == 4

    def test_each_grade_drops_with_its_own_reason(self) -> None:
        gaps = [
            _gap("what the tracker will show on the resolution date", answerable_now=False),
            _gap("the reading the briefing already dates", already_in_first_pass=True),
            _gap("a distinct, live fact"),
        ]

        triage = triage_gaps(gaps, max_gaps=4)

        assert triage.kept == [gaps[2]]
        assert [(d["position"], d["reason"]) for d in triage.dropped] == [
            (1, DROP_NOT_ANSWERABLE),
            (2, DROP_ALREADY_ANSWERED),
        ]

    def test_a_gap_earns_exactly_one_reason_graded_before_deduped(self) -> None:
        """One reason per gap, so the marker's per-reason counts partition ``listed``."""
        gaps = [_gap("g1"), _gap("g1 again, but future-dated", answerable_now=False, same_need_as=1)]

        triage = triage_gaps(gaps, max_gaps=4)

        assert [d["reason"] for d in triage.dropped] == [DROP_NOT_ANSWERABLE]

    def test_paraphrase_of_a_kept_gap_is_dropped(self) -> None:
        gaps = [_gap("the dashboard reading"), _gap("the monthly summary of that reading", same_need_as=1)]

        triage = triage_gaps(gaps, max_gaps=4)

        assert triage.kept == [gaps[0]]
        assert [(d["position"], d["reason"]) for d in triage.dropped] == [(2, DROP_SAME_NEED)]

    def test_paraphrase_chain_through_a_dropped_paraphrase_is_dropped(self) -> None:
        """Gap 3 restates gap 2, which restates gap 1: the pointer is followed to the NEED, and the need is
        the one gap 1 is already searching, so both restatements go."""
        gaps = [_gap("g1"), _gap("g2", same_need_as=1), _gap("g3", same_need_as=2)]

        triage = triage_gaps(gaps, max_gaps=4)

        assert triage.kept == [gaps[0]]
        assert [(d["position"], d["reason"]) for d in triage.dropped] == [(2, DROP_SAME_NEED), (3, DROP_SAME_NEED)]

    def test_paraphrase_of_an_unanswerable_gap_becomes_the_needs_carrier(self) -> None:
        """The analyzer's commonest pair is a future-dated ask and its present-tense rewording. Nothing is
        searching the need once the ask is dropped, so the rewording is kept; a third restatement is not."""
        gaps = [
            _gap("what the tracker will show on 2026-09-30", answerable_now=False),
            _gap("what the tracker shows now", same_need_as=1),
            _gap("the tracker's reading via its mirror", same_need_as=2),
        ]

        triage = triage_gaps(gaps, max_gaps=4)

        assert triage.kept == [gaps[1]]
        assert [(d["position"], d["reason"]) for d in triage.dropped] == [
            (1, DROP_NOT_ANSWERABLE),
            (3, DROP_SAME_NEED),
        ]

    def test_paraphrase_of_a_first_pass_reading_is_dropped_whatever_its_own_grade_says(self) -> None:
        """A need the first pass already answers stays answered under any rewording."""
        gaps = [
            _gap("the tracker reading", already_in_first_pass=True),
            _gap("the tracker's summary page", same_need_as=1),
        ]

        triage = triage_gaps(gaps, max_gaps=4)

        assert triage.kept == []
        assert [d["reason"] for d in triage.dropped] == [DROP_ALREADY_ANSWERED, DROP_SAME_NEED]

    def test_a_malformed_slot_ahead_of_a_pointer_keeps_the_positions_aligned(self) -> None:
        """The analyzer list [malformed, A future-dated, B distinct, C same_need_as=2 (A's present-tense
        rewording)]: C's pointer must still name A, so C is kept as A's carrier. A parser that dropped
        the malformed slot would renumber and C's pointer would land on B."""
        gaps = [
            {"gap": "", "search_query": "", "why_matters": ""},
            _gap("what the tracker will show on 2026-09-30", answerable_now=False),
            _gap("an unrelated base rate"),
            _gap("what the tracker shows now", same_need_as=2),
        ]

        triage = triage_gaps(gaps, max_gaps=4)

        assert [g["gap"] for g in triage.kept] == ["an unrelated base rate", "what the tracker shows now"]
        assert [(d["position"], d["reason"]) for d in triage.dropped] == [(1, DROP_SCHEMA), (2, DROP_NOT_ANSWERABLE)]

    def test_two_siblings_of_an_unanswerable_gap_share_one_carrier(self) -> None:
        """Both restatements point at the same future-dated gap rather than at each other: the first
        becomes the need's carrier, and the second is a restatement of a need now being searched."""
        gaps = [
            _gap("what the tracker will show on 2026-09-30", answerable_now=False),
            _gap("what the tracker shows now", same_need_as=1),
            _gap("the tracker's reading via its mirror", same_need_as=1),
        ]

        triage = triage_gaps(gaps, max_gaps=4)

        assert triage.kept == [gaps[1]]
        assert [(d["position"], d["reason"]) for d in triage.dropped] == [
            (1, DROP_NOT_ANSWERABLE),
            (3, DROP_SAME_NEED),
        ]

    def test_two_siblings_of_a_schema_dropped_gap_share_one_carrier(self) -> None:
        gaps = [
            {"gap": "ungraded", "search_query": "q", "why_matters": ""},
            _gap("graded restatement", same_need_as=1),
            _gap("another graded restatement", same_need_as=1),
        ]

        triage = triage_gaps(gaps, max_gaps=4)

        assert triage.kept == [gaps[1]]
        assert [(d["position"], d["reason"]) for d in triage.dropped] == [(1, DROP_SCHEMA), (3, DROP_SAME_NEED)]

    def test_paraphrase_of_a_schema_dropped_gap_stands_on_its_own_grades(self) -> None:
        """An ungraded gap says nothing about whether its need is covered, so a graded restatement of it is
        judged on its own fields."""
        gaps = [{"gap": "ungraded", "search_query": "q", "why_matters": ""}, _gap("graded restatement", same_need_as=1)]

        triage = triage_gaps(gaps, max_gaps=4)

        assert triage.kept == [gaps[1]]
        assert [(d["position"], d["reason"]) for d in triage.dropped] == [(1, DROP_SCHEMA)]

    @pytest.mark.parametrize(
        "gap",
        [
            pytest.param({"gap": "g", "search_query": "q", "why_matters": ""}, id="no grade fields at all"),
            pytest.param(_gap_without("answerable_now"), id="answerable_now key omitted"),
            pytest.param(_gap_without("already_in_first_pass"), id="already_in_first_pass key omitted"),
            pytest.param(_gap("g", answerable_now=None), id="answerable_now null"),
            pytest.param(_gap("g", answerable_now="true"), id="answerable_now as a string"),
            pytest.param(_gap("g", already_in_first_pass=None), id="already_in_first_pass null"),
            pytest.param(_gap("g", already_in_first_pass=0), id="already_in_first_pass as an int"),
            pytest.param(_gap("g", same_need_as=True), id="same_need_as a bool"),
            pytest.param(_gap("g", same_need_as=1.0), id="same_need_as a float"),
            pytest.param(_gap("g", same_need_as="1"), id="same_need_as a string"),
            pytest.param(_gap("g", same_need_as=1), id="same_need_as pointing at itself"),
            pytest.param(_gap("g", same_need_as=2), id="same_need_as pointing forward"),
            pytest.param(_gap("g", same_need_as=0), id="same_need_as zero"),
        ],
    )
    def test_schema_drift_drops_the_gap(self, gap: dict[str, Any]) -> None:
        """A guard fails SHUT. A grade the analyzer omitted or mistyped is never read as passing, because a
        grade that defaulted to passing would spend exactly the money the grade exists to save."""
        triage = triage_gaps([gap], max_gaps=4)

        assert triage.kept == []
        assert triage.dropped == [{**gap, "position": 1, "reason": DROP_SCHEMA}]

    def test_an_omitted_same_need_as_key_reads_as_no_pointer(self) -> None:
        """An omitted null pointer remains compatible with archived analyzer output. A
        pointer the analyzer did type is still validated (see ``test_schema_drift_drops_the_gap``)."""
        gap = _gap_without("same_need_as")

        triage = triage_gaps([gap], max_gaps=4)

        assert triage.kept == [gap]
        assert triage.dropped == []

    def test_cap_applies_after_the_filter(self) -> None:
        """Six listed, two failing: all four survivors are searched. A parse-time clip to four would have
        kept the first four and searched only two of them."""
        gaps = [
            _gap("g1", answerable_now=False),
            _gap("g2"),
            _gap("g3", already_in_first_pass=True),
            _gap("g4"),
            _gap("g5"),
            _gap("g6"),
        ]

        triage = triage_gaps(gaps, max_gaps=4)

        assert [g["gap"] for g in triage.kept] == ["g2", "g4", "g5", "g6"]
        assert triage.dropped_for(DROP_OVER_CAP) == 0

    def test_survivors_past_the_cap_are_dropped_as_over_cap(self) -> None:
        gaps = [_gap(f"g{i}") for i in range(1, 7)]

        triage = triage_gaps(gaps, max_gaps=4)

        assert [g["gap"] for g in triage.kept] == ["g1", "g2", "g3", "g4"]
        assert [(d["position"], d["reason"]) for d in triage.dropped] == [(5, DROP_OVER_CAP), (6, DROP_OVER_CAP)]

    def test_counts_partition_the_list(self) -> None:
        gaps = [
            _gap("g1", answerable_now=False),
            _gap("g2"),
            _gap("g3", same_need_as=2),
            {"gap": "g4", "search_query": "q4", "why_matters": ""},
            _gap("g5", already_in_first_pass=True),
            _gap("g6"),
            _gap("g7"),
        ]

        triage = triage_gaps(gaps, max_gaps=2)

        assert triage.listed == 7
        assert len(triage.kept) == 2
        assert {reason: triage.dropped_for(reason) for reason in DROP_REASONS} == {
            DROP_NOT_ANSWERABLE: 1,
            DROP_ALREADY_ANSWERED: 1,
            DROP_SAME_NEED: 1,
            DROP_SCHEMA: 1,
            DROP_OVER_CAP: 1,
        }

    def test_empty_list(self) -> None:
        triage = triage_gaps([], max_gaps=4)

        assert triage.kept == []
        assert triage.dropped == []
        assert triage.listed == 0


# ---------------------------------------------------------------------------
# run_gap_fill_pass integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_gaps_returns_empty_string() -> None:
    """Analyzer returns no gaps → addendum is "" and the search fn is never called."""
    question = MockQuestion()

    fake_search = AsyncMock(return_value="should not be called")
    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=[])),
        _patch_resolver(fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()


@pytest.mark.asyncio
async def test_two_gaps_run_in_parallel() -> None:
    """Analyzer returns 2 gaps → 2 native-search calls that run concurrently.

    The slow-first, fast-second ordering guards against a sequential refactor: if the
    code ever did ``for gap in gaps: await llm.invoke(...)``, gap 1 would
    finish before gap 2. With ``asyncio.gather``, the short sleep wins the race.
    """
    question = MockQuestion()

    gaps = [
        _gap("gap one text", "q1", "wm1"),
        _gap("gap two text", "q2", "wm2"),
    ]

    completion_order: list[str] = []

    async def search_side_effect(prompt: str) -> str:
        if "q1" in prompt:
            await asyncio.sleep(0.1)
            completion_order.append("q1")
            return "result for gap 1"
        if "q2" in prompt:
            await asyncio.sleep(0.01)
            completion_order.append("q2")
            return "result for gap 2"
        raise AssertionError(f"unexpected prompt: {prompt!r}")

    fake_search = AsyncMock(side_effect=search_side_effect)

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=gaps)),
        _patch_resolver(fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert fake_search.await_count == 2
    assert "### Gap 1: gap one text" in out
    assert "### Gap 2: gap two text" in out
    assert "result for gap 1" in out
    assert "result for gap 2" in out
    # Concurrency check: the fast q2 search must land before the slow q1; a sequential run would give ["q1", "q2"].
    assert completion_order == ["q2", "q1"]


_TRIAGE_SPEC = next(spec for spec in MARKER_SPECS if spec.name == "gap_fill_v1_triage")


def _triage_markers(caplog: pytest.LogCaptureFixture) -> list[dict[str, str]]:
    """Every emitted GAP_FILL_V1_TRIAGE line, parsed by the registry's own regex so the emitter and the spec
    fail side by side here rather than only against the hand-typed literal in tests/test_telemetry_markers.py."""
    lines = [rec.message for rec in caplog.records if rec.message.startswith("GAP_FILL_V1_TRIAGE:")]
    parsed = [_TRIAGE_SPEC.regex.search(line) for line in lines]
    assert all(parsed), f"a GAP_FILL_V1_TRIAGE line does not match its MarkerSpec: {lines}"
    return [match.groupdict() for match in parsed if match is not None]


def _triage_record(question: str, listed: int, kept: int, **dropped: int) -> dict[str, str]:
    """The expected parse of one marker line; every reason not named counts zero."""
    counts = {f"dropped_{reason}": str(dropped.get(reason, 0)) for reason in DROP_REASONS}
    return {"question": question, "listed": str(listed), "kept": str(kept), **counts}


def _triage_levels(caplog: pytest.LogCaptureFixture) -> list[int]:
    return [rec.levelno for rec in caplog.records if rec.message.startswith("GAP_FILL_V1_TRIAGE:")]


def test_marker_spec_fields_are_the_drop_reasons() -> None:
    """The emitter derives its field set and order from DROP_REASONS while the spec spells them out; a new
    reason has to land in both, or the harvester silently records zero rows."""
    assert set(_TRIAGE_SPEC.regex.groupindex) - {"question", "listed", "kept"} == {f"dropped_{r}" for r in DROP_REASONS}


@pytest.mark.asyncio
async def test_resolver_runs_exactly_the_survivors_in_order(caplog: pytest.LogCaptureFixture) -> None:
    """Four listed, two dropped before any spend: the resolver is invoked once per survivor in analyzer
    order, the addendum numbers the survivors 1..K (that index is the index into the raw record's
    ``gaps``), and the GAP_FILL_V1_TRIAGE marker carries one count per drop reason."""
    question = MockQuestion()
    gaps = [
        _gap("future reading", "q1", "wm1", answerable_now=False),
        _gap("live reading", "q2", "wm2"),
        _gap("live reading, restated", "q3", "wm3", same_need_as=2),
        _gap("distinct base rate", "q4", "wm4"),
    ]
    fake_search = AsyncMock(side_effect=["r2", "r4"])

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=gaps)),
        _patch_resolver(fake_search) as builder,
        caplog.at_level(logging.INFO, logger="metaculus_bot.research.targeted"),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    prompts = [call.args[0] for call in fake_search.await_args_list]
    assert len(prompts) == 2
    assert builder.call_count == 2
    assert "q2" in prompts[0]
    assert "q1" not in prompts[0]
    assert "q4" in prompts[1]
    assert "### Gap 1: live reading\n" in out
    assert "r2" in out
    assert "### Gap 2: distinct base rate\n" in out
    assert "r4" in out
    assert "future reading" not in out
    assert "restated" not in out
    assert _triage_markers(caplog) == [_triage_record("42", listed=4, kept=2, not_answerable=1, same_need=1)]


@pytest.mark.asyncio
async def test_every_gap_dropped_returns_empty_and_counts_them(caplog: pytest.LogCaptureFixture) -> None:
    """Prose must never stand in for an absent section: with nothing left to search the pass returns "",
    builds no resolver, and the marker's kept=0 beside its reason counts is the loss record (v1 has no
    ProviderResult and no lost= token; docs/research.md)."""
    question = MockQuestion()
    gaps = [
        _gap("g1", "q1", answerable_now=False),
        _gap("g2", "q2", already_in_first_pass=True),
        _gap("g3", "q3", same_need_as=2),
        {"gap": "g4", "search_query": "q4", "why_matters": ""},
    ]
    fake_search = AsyncMock(return_value="should not be called")

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=gaps)),
        _patch_resolver(fake_search) as builder,
        patch("metaculus_bot.research.targeted.record_raw_research") as rec,
        caplog.at_level(logging.INFO, logger="metaculus_bot.research.targeted"),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()
    builder.assert_not_called()
    assert _triage_markers(caplog) == [
        _triage_record("42", listed=4, kept=0, not_answerable=1, in_first_pass=1, same_need=1, schema=1)
    ]
    # Legitimate grades are the filter working, so the marker stays at INFO.
    assert _triage_levels(caplog) == [logging.INFO]
    # The dropped list is the whole audit trail on this path, so the raw record is written with it.
    payload = rec.call_args.kwargs["payload"]
    assert payload["gaps"] == []
    assert payload["results"] == []
    assert [(d["position"], d["reason"]) for d in payload["dropped"]] == [
        (1, DROP_NOT_ANSWERABLE),
        (2, DROP_ALREADY_ANSWERED),
        (3, DROP_SAME_NEED),
        (4, DROP_SCHEMA),
    ]


@pytest.mark.asyncio
async def test_every_gap_failing_the_schema_warns(caplog: pytest.LogCaptureFixture) -> None:
    """An analyzer that stopped emitting the grades is v1 gone dark while it still bills, the same outcome
    as a dead analyzer, so the marker line is a WARNING then and only then."""
    question = MockQuestion()
    gaps = [
        {"gap": "g1", "search_query": "q1", "why_matters": ""},
        {"gap": "g2", "search_query": "q2", "why_matters": ""},
    ]

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=gaps)),
        _patch_resolver(AsyncMock()),
        caplog.at_level(logging.INFO, logger="metaculus_bot.research.targeted"),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    assert _triage_markers(caplog) == [_triage_record("42", listed=2, kept=0, schema=2)]
    assert _triage_levels(caplog) == [logging.WARNING]


@pytest.mark.asyncio
async def test_dropped_gaps_are_logged_with_position_reason_and_text(caplog: pytest.LogCaptureFixture) -> None:
    """The marker carries counts; the per-gap line carries what was dropped, so a run log can be read by eye."""
    question = MockQuestion()
    gaps = [_gap("what the tracker will show on 2026-09-30", "q1", answerable_now=False), _gap("live", "q2")]

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=gaps)),
        _patch_resolver(AsyncMock(return_value="r")),
        caplog.at_level(logging.INFO, logger="metaculus_bot.research.targeted"),
    ):
        await run_gap_fill_pass(_q(question), "first-pass research")

    drop_lines = [rec.message for rec in caplog.records if rec.message.startswith("GapFill: dropped gap")]
    assert drop_lines == [
        "GapFill: dropped gap #1 reason=not_answerable same_need_as=None: what the tracker will show on 2026-09-30"
    ]


@pytest.mark.asyncio
async def test_analyzer_with_no_gaps_emits_a_zero_triage_marker(caplog: pytest.LogCaptureFixture) -> None:
    """listed=0 is "the analyzer answered and found nothing", a different record from a dead analyzer."""
    question = MockQuestion()

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=[])),
        _patch_resolver(AsyncMock()),
        patch("metaculus_bot.research.targeted.record_raw_research") as rec,
        caplog.at_level(logging.INFO, logger="metaculus_bot.research.targeted"),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    assert _triage_markers(caplog) == [_triage_record("42", listed=0, kept=0)]
    assert _triage_levels(caplog) == [logging.INFO]
    # A raw record exists whenever the analyzer answered, so archive presence counts filter on non-empty gaps.
    assert rec.call_args.kwargs["payload"] == {"gaps": [], "results": [], "dropped": []}


@pytest.mark.asyncio
async def test_analyzer_failure_emits_no_triage_marker(caplog: pytest.LogCaptureFixture) -> None:
    """A dead analyzer is GAP_FILL_ANALYZER_FAILED alone; a listed=0 triage line beside it would read as a
    legitimately empty gap list."""
    question = MockQuestion()

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(side_effect=TimeoutError())),
        _patch_resolver(AsyncMock()),
        caplog.at_level(logging.INFO, logger="metaculus_bot.research.targeted"),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    assert _triage_markers(caplog) == []
    assert any(rec.message.startswith("GAP_FILL_ANALYZER_FAILED:") for rec in caplog.records)


@pytest.mark.asyncio
async def test_cap_applies_after_the_grade_filter(caplog: pytest.LogCaptureFixture) -> None:
    """``GAP_FILL_MAX_GAPS`` caps the SURVIVORS: with two of GAP_FILL_MAX_GAPS + 2 gaps failing a grade,
    every survivor is searched. The old parse-time clip kept the first GAP_FILL_MAX_GAPS and would have
    searched only GAP_FILL_MAX_GAPS - 2 of them."""
    question = MockQuestion()
    oversized_count = GAP_FILL_MAX_GAPS + 2
    gaps = [_gap(f"gap {i}", f"q{i}", f"wm{i}") for i in range(oversized_count)]
    gaps[0]["answerable_now"] = False
    gaps[1]["already_in_first_pass"] = True
    fake_search = AsyncMock(side_effect=[f"res{i}" for i in range(2, oversized_count)])

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=gaps)),
        _patch_resolver(fake_search),
        caplog.at_level(logging.INFO, logger="metaculus_bot.research.targeted"),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert fake_search.await_count == GAP_FILL_MAX_GAPS
    assert f"### Gap {GAP_FILL_MAX_GAPS}: gap {oversized_count - 1}" in out
    assert f"res{oversized_count - 1}" in out
    assert _triage_markers(caplog)[0]["dropped_over_cap"] == "0"


@pytest.mark.asyncio
async def test_survivors_past_the_cap_are_not_searched(caplog: pytest.LogCaptureFixture) -> None:
    question = MockQuestion()
    oversized_count = GAP_FILL_MAX_GAPS + 2
    gaps = [_gap(f"gap {i}", f"q{i}", f"wm{i}") for i in range(oversized_count)]
    fake_search = AsyncMock(side_effect=[f"res{i}" for i in range(oversized_count)])

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=gaps)),
        _patch_resolver(fake_search),
        caplog.at_level(logging.INFO, logger="metaculus_bot.research.targeted"),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert fake_search.await_count == GAP_FILL_MAX_GAPS
    assert f"### Gap {GAP_FILL_MAX_GAPS + 1}:" not in out
    assert _triage_markers(caplog)[0]["dropped_over_cap"] == "2"


@pytest.mark.asyncio
async def test_raw_record_carries_the_survivors_and_the_dropped_gaps() -> None:
    """The raw research log is where the next redundancy audit reads the analyzer's grading: the kept gaps
    (aligned with ``results`` and with the addendum's Gap N index) plus every dropped gap with its analyzer
    position and reason."""
    question = MockQuestion()
    kept = _gap("live reading", "q2", "wm2")
    dropped = _gap("future reading", "q1", "wm1", answerable_now=False)

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=[dropped, kept])),
        _patch_resolver(AsyncMock(return_value="r2")),
        patch("metaculus_bot.research.targeted.record_raw_research") as rec,
    ):
        await run_gap_fill_pass(_q(question), "first-pass research")

    rec.assert_called_once()
    payload = rec.call_args.kwargs["payload"]
    assert payload["gaps"] == [kept]
    assert payload["results"] == ["r2"]
    assert payload["dropped"] == [{**dropped, "position": 1, "reason": DROP_NOT_ANSWERABLE}]


@pytest.mark.asyncio
async def test_malformed_analyzer_output_soft_fails() -> None:
    """Malformed output preserves forecasting but reports a stage failure."""
    question = MockQuestion()
    errors: list[BaseException] = []
    analyzer = MagicMock(invoke=AsyncMock(return_value="not valid JSON"))
    fake_search = AsyncMock(return_value="should not be called")
    with (
        patch("metaculus_bot.fallback_openrouter.build_llm_with_openrouter_fallback", return_value=analyzer),
        _patch_resolver(fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research", on_error=errors.append)

    assert out == ""
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    fake_search.assert_not_called()


@pytest.mark.asyncio
async def test_analyzer_prompt_carries_the_mc_ballot() -> None:
    """The gap analyzer must see an MC question's option list: a "no coverage of candidate X"
    gap is only findable when the analyzer knows the candidates (q44952 — no research stage
    ever saw the ballot, and the winner went unsearched)."""
    question = MockQuestion(options=["Mir Kim", "Hunter Feuerstein", "Other"])

    stub_llm = MagicMock()
    stub_llm.invoke = AsyncMock(return_value='{"gaps": []}')
    with patch(
        "metaculus_bot.fallback_openrouter.build_llm_with_openrouter_fallback",
        MagicMock(return_value=stub_llm),
    ):
        await _run_analyzer(_q(question), "first-pass research", is_benchmarking=False)

    assert stub_llm.invoke.await_args is not None
    prompt = stub_llm.invoke.await_args.args[0]
    assert "Options (in resolution order): Mir Kim | Hunter Feuerstein | Other" in prompt


@pytest.mark.asyncio
async def test_analyzer_returns_every_listed_gap_unclipped() -> None:
    """The cap lives in triage, after the grade filter; a clip restored at the parse call would let a
    dropped gap displace a kept one again with every other test still green."""
    question = MockQuestion()
    listed = [_gap(f"g{i}", f"q{i}") for i in range(1, GAP_FILL_MAX_GAPS + 3)]

    stub_llm = MagicMock()
    stub_llm.invoke = AsyncMock(return_value=json.dumps({"gaps": listed}))
    with patch(
        "metaculus_bot.fallback_openrouter.build_llm_with_openrouter_fallback",
        MagicMock(return_value=stub_llm),
    ):
        gaps = await _run_analyzer(_q(question), "first-pass research", is_benchmarking=False)

    assert gaps == listed


@pytest.mark.asyncio
async def test_resolver_prompt_carries_the_resolution_criteria_and_fine_print() -> None:
    """The per-gap resolver reads the criteria it is asked about. On q44267 it saw only the title,
    ruled which of two published figures resolved the question from a sister question's wording,
    and every forecaster ratified it (-95.66 spot peer); the analyzer had seen both fields all along."""
    question = MockQuestion(
        resolution_criteria="Resolves as the count detected in the ADIZ.",
        fine_print="Synced with the original question.",
    )
    gap = _gap("Which figure resolves it?", "ADIZ count definition", "w")

    fake_search = AsyncMock(return_value="found it")
    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=[gap])),
        _patch_resolver(fake_search),
    ):
        await run_gap_fill_pass(_q(question), "first-pass research")

    assert fake_search.await_args is not None
    prompt = fake_search.await_args.args[0]
    assert (
        "Resolution criteria (what the question actually resolves on):\nResolves as the count detected in the ADIZ."
        in prompt
    )
    assert "Fine print:\nSynced with the original question." in prompt


@pytest.mark.asyncio
async def test_partial_search_failure_returns_successful_results(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """3 gaps, middle one raises → addendum contains gap 1 and gap 3 only; gap 2 logged."""
    question = MockQuestion()

    gaps = [
        _gap("g1", "q1", "wm1"),
        _gap("g2", "q2", "wm2"),
        _gap("g3", "q3", "wm3"),
    ]

    async def search_side_effect(prompt: str) -> str:
        await asyncio.sleep(0)
        if "q2" in prompt:
            raise RuntimeError("boom for gap 2")
        if "q1" in prompt:
            return "result 1"
        if "q3" in prompt:
            return "result 3"
        raise AssertionError(f"unexpected prompt: {prompt!r}")

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=gaps)),
        _patch_resolver(AsyncMock(side_effect=search_side_effect)),
        caplog.at_level(logging.WARNING, logger="metaculus_bot.research.targeted"),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert "### Gap 1: g1" in out
    assert "result 1" in out
    assert "### Gap 3: g3" in out
    assert "result 3" in out
    # Gap 2 must NOT appear in the addendum.
    assert "### Gap 2:" not in out
    # And the failure was logged as a warning.
    assert any("gap #2" in rec.message.lower() or "boom" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_all_searches_fail_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    """All native searches raise → addendum is "" and each failure logs a warning.

    The soft-fail contract depends on failures being observable in logs, so we assert
    exactly one warning per failed gap with the "gap #<n>" marker.
    """
    question = MockQuestion()

    gaps = [
        _gap("g1", "q1", "wm1"),
        _gap("g2", "q2", "wm2"),
    ]
    fake_search = AsyncMock(side_effect=RuntimeError("all fail"))

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=gaps)),
        _patch_resolver(fake_search),
        caplog.at_level(logging.WARNING, logger="metaculus_bot.research.targeted"),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""

    gap_failure_records = [
        rec for rec in caplog.records if rec.levelno == logging.WARNING and "gap #" in rec.message.lower()
    ]
    assert len(gap_failure_records) == len(gaps)
    assert any("gap #1" in rec.message.lower() for rec in gap_failure_records)
    assert any("gap #2" in rec.message.lower() for rec in gap_failure_records)


@pytest.mark.asyncio
async def test_benchmarking_flag_threaded_to_analyzer_and_searches() -> None:
    """is_benchmarking=True: analyzer + each gap search receives the benchmarking warning."""
    question = MockQuestion()

    gaps = [
        _gap("g1", "q1", "wm1"),
        _gap("g2", "q2", "wm2"),
    ]
    fake_analyzer = AsyncMock(return_value=gaps)
    fake_search = AsyncMock(side_effect=["r1", "r2"])

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", fake_analyzer),
        _patch_resolver(fake_search),
    ):
        await run_gap_fill_pass(_q(question), "first-pass", is_benchmarking=True)

    # Analyzer was called with is_benchmarking=True.
    fake_analyzer.assert_awaited_once()
    assert fake_analyzer.call_args.kwargs["is_benchmarking"] is True

    # Each per-gap search prompt includes the benchmarking warning string.
    for call in fake_search.call_args_list:
        prompt = call.args[0]
        assert "benchmarking run" in prompt


@pytest.mark.asyncio
async def test_resolver_builds_native_search_llm_with_sol_low() -> None:
    """The per-gap resolver runs OpenAI native search on gpt-6-sol at low effort.

    Locks the 2026-06-25 migration off direct-Google grounded Gemini: every gap
    resolution must build a native-search LLM with the GAP_FILL_RESOLVER_MODEL
    slug (gpt-5.6-terra since the 2026-07-20 sol→terra flip, then gpt-6-sol on the
    2026-09-22 GPT-6 migration since Terra has no GPT-6 successor). Effort stays low
    (Round-2): the resolver was the ~5-min critical-path bottleneck, and low is
    ~4.5× faster (native_search v3 bench). Pinned to the constant so it stays a
    canary if either the model or effort changes again.
    """
    from metaculus_bot.constants import (
        GAP_FILL_RESOLVER_MODEL,
        GAP_FILL_RESOLVER_REASONING_EFFORT,
    )  # HARNESS-SCAN-EXEMPT-function-level-import  # constants pinned in the one test that asserts them

    question = MockQuestion()
    gaps = [_gap("g1", "q1", "wm1")]
    fake_search = AsyncMock(return_value="resolved")

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=gaps)),
        _patch_resolver(fake_search) as builder,
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert "resolved" in out
    builder.assert_called_once()
    call = builder.call_args
    model_arg = call.args[0] if call.args else call.kwargs.get("model_slug")
    assert model_arg == GAP_FILL_RESOLVER_MODEL == "openai/gpt-6-sol"
    assert call.kwargs["reasoning_effort"] == GAP_FILL_RESOLVER_REASONING_EFFORT == "low"


@pytest.mark.asyncio
async def test_resolver_enforces_wall_clock_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung per-gap resolver invoke must be bounded by NATIVE_SEARCH_WALL_TIMEOUT.

    The 2026-06-25 migration added ``asyncio.wait_for(llm.invoke(...), timeout=...)``
    to ``_resolve_single_gap`` — a backstop the old google-genai grounded path
    lacked. It shares the build_native_search_llm config (and therefore the same
    OpenRouter whitespace-drip pathology, 2026-05-20 incident) with the targeted
    search and native_search provider, whose backstop is locked in by
    test_targeted_research.test_enforces_wall_clock_timeout. This mirrors that
    test for the gap-fill resolver: a resolver that sleeps past the cap is
    cancelled, the TimeoutError is captured by gather(return_exceptions=True),
    and the pass soft-fails to "" rather than hanging the whole question.
    """
    monkeypatch.setattr("metaculus_bot.research.targeted.NATIVE_SEARCH_WALL_TIMEOUT", 0.05)

    question = MockQuestion()
    gaps = [_gap("g1", "q1", "wm1")]

    async def hang(_prompt: str) -> str:
        """Sleep well past the 0.05s wall-clock cap; the test passes only if wait_for cancels it first."""
        await asyncio.sleep(5)
        return "should never reach here"

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=gaps)),
        _patch_resolver(AsyncMock(side_effect=hang)),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""


@pytest.mark.asyncio
async def test_analyzer_timeout_returns_empty() -> None:
    """_run_analyzer raising TimeoutError → soft-fail with "" and no search calls."""
    question = MockQuestion()

    fake_search = AsyncMock()
    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(side_effect=TimeoutError())),
        _patch_resolver(fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()


@pytest.mark.asyncio
async def test_analyzer_missing_key_returns_empty() -> None:
    """_run_analyzer raising ValueError (e.g., missing API key) → "" and no searches."""
    question = MockQuestion()

    fake_search = AsyncMock()
    with (
        patch(
            "metaculus_bot.research.targeted._run_analyzer",
            AsyncMock(side_effect=ValueError("API key must be set")),
        ),
        _patch_resolver(fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()


@pytest.mark.asyncio
async def test_analyzer_gemini_api_error_returns_empty() -> None:
    """google-genai APIError from the analyzer → soft-fail with "" (no raising out).

    Covers ClientError (4xx) and ServerError (5xx) since both subclass APIError.
    """
    import httpx  # HARNESS-SCAN-EXEMPT-function-level-import  # httpx response faked only by this test
    from google.genai.errors import APIError  # HARNESS-SCAN-EXEMPT-function-level-import  # faked only by this test

    question = MockQuestion()

    # APIError signature: APIError(code, response_json, response=None). We fake a 500 payload.
    response = httpx.Response(status_code=500, text='{"error": {"message": "internal"}}')
    fake_exc = APIError(code=500, response_json={"error": {"message": "internal"}}, response=response)

    fake_search = AsyncMock()
    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(side_effect=fake_exc)),
        _patch_resolver(fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()


@pytest.mark.asyncio
async def test_analyzer_httpx_error_returns_empty() -> None:
    """Raw httpx.HTTPError from the analyzer → soft-fail with "" (covers mid-SDK network failures)."""
    import httpx  # HARNESS-SCAN-EXEMPT-function-level-import  # httpx error faked only by this test

    question = MockQuestion()
    fake_search = AsyncMock()
    with (
        patch(
            "metaculus_bot.research.targeted._run_analyzer",
            AsyncMock(side_effect=httpx.ConnectError("connection refused")),
        ),
        _patch_resolver(fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()


@pytest.mark.asyncio
async def test_analyzer_os_error_returns_empty() -> None:
    """OSError (DNS / socket) from the analyzer → soft-fail with ""."""
    question = MockQuestion()
    fake_search = AsyncMock()
    with (
        patch(
            "metaculus_bot.research.targeted._run_analyzer",
            AsyncMock(side_effect=OSError("name resolution failed")),
        ),
        _patch_resolver(fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()


# ---------------------------------------------------------------------------
# Prompt wiring (resolution_criteria / fine_print thread through)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyzer_receives_resolution_criteria_and_fine_print() -> None:
    """The analyzer call passes through resolution_criteria + fine_print from the question."""
    question = MockQuestion(
        question_text="Will bitcoin exceed $200k?",
        resolution_criteria="Resolves YES if BTC USD > 200k on Jan 1 2027.",
        fine_print="Data source: Coinbase Pro BTC-USD.",
    )

    fake_analyzer = AsyncMock(return_value=[])
    fake_search = AsyncMock()
    with (
        patch("metaculus_bot.research.targeted._run_analyzer", fake_analyzer),
        _patch_resolver(fake_search),
    ):
        await run_gap_fill_pass(_q(question), "some first-pass research")

    # _run_analyzer is called with (question, first_pass_research, is_benchmarking=...)
    fake_analyzer.assert_awaited_once()
    positional_q = fake_analyzer.call_args.args[0]
    assert positional_q.resolution_criteria == "Resolves YES if BTC USD > 200k on Jan 1 2027."
    assert positional_q.fine_print == "Data source: Coinbase Pro BTC-USD."
