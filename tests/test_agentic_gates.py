"""Gate tests for the agentic gap-fill v2 driver loop.

The four gates that constrain what the low-effort driver may do and what may
reach a forecaster, split out of ``tests/test_agentic_loop.py`` (which keeps the
loop mechanics — budgets, deadlines, tool dispatch, dedup, telemetry):

* ``TestProvenanceGate`` — findings cite only code-delivered evidence references.
* ``TestEvidenceRetrievalMetadata`` — source text and provider summaries retain their origin.
* ``TestResearchPlanGate`` — W1: no external tool call before ``set_research_plan``.
* ``TestConcludeGate`` — W2: no early conclude until gap accounting clears.

Shared scaffolding (``FakeLlm`` and the loop-driving helpers) lives in
``tests/agentic_fakes.py`` so this file and ``test_agentic_loop.py`` drive the
loop identically.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pytest

from metaculus_bot.constants import (
    GAP_FILL_V2_EVIDENCE_FINDING_MAX_CHARS,
    GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS,
)
from metaculus_bot.research.agentic.loop import _bank_findings, _LoopState, run_agentic_loop
from metaculus_bot.research.agentic.provenance import (
    _SPAN_JOINER_MAX_CHARS,
    _normalize_quote_text,
    _quote_is_grounded,
)
from metaculus_bot.research.agentic.types import EvidencePassage, Finding, FindingSupport, ToolOutcome
from tests.agentic_fakes import FakeLlm
from tests.agentic_fakes import gap_accounting as _accounting
from tests.agentic_fakes import loop_config as _config
from tests.agentic_fakes import plan_call as _plan_call
from tests.agentic_fakes import response as _response
from tests.agentic_fakes import tool_call as _tool_call
from tests.agentic_fakes import tool_messages as _tool_messages
from tests.agentic_fakes import tool_spec as _tool_spec


def _finding(
    *,
    claim: str = "The board published the minutes on July 1.",
    evidence_refs: tuple[str, ...] = ("ev_1",),
    discrepancy: bool = False,
) -> dict:
    return {
        "claim": claim,
        "evidence_refs": list(evidence_refs),
        "date": "2026-07-01",
        "topic": "minutes",
        "discrepancy": discrepancy,
    }


def _search_returning(content: str, links: list[str] | None = None):
    async def _search(**kwargs: Any) -> ToolOutcome:
        source_url = next(iter(links or []), None) or kwargs.get("url")
        if source_url is None:
            match = re.search(r"https?://[^\s<>\"']+", content)
            source_url = match.group(0).rstrip(".,;:!?)]}") if match else None
        if source_url is None:
            raise AssertionError("test search result must identify the source for its evidence passage")
        method = "plain" if kwargs.get("url") else "search"
        origin = "source_text" if method == "plain" else "provider_summary"
        return ToolOutcome(
            content_markdown=content,
            links=links or [source_url],
            method=method,
            evidence_passages=[
                EvidencePassage(
                    source_url=source_url,
                    text=content,
                    retrieval_method=method,
                    content_origin=origin,
                )
            ],
        )

    return _search


def _returns_method(content: str, *, method: str, status: str = "ok", links: list[str] | None = None):
    """A tool handler returning a ToolOutcome with a specific method/status —
    used by the W4 tier-stamping tests to drive fetched vs snippet vs failed
    retrievals through the loop."""

    async def _handler(**kwargs: Any) -> ToolOutcome:
        source_url = kwargs.get("url") or next(iter(links or []), None)
        if source_url is None:
            match = re.search(r"https?://[^\s<>\"']+", content)
            source_url = match.group(0).rstrip(".,;:!?)]}") if match else None
        evidence_passages = []
        if status == "ok" and method != "local_navigation" and source_url is not None:
            origin = "provider_summary" if method in {"search", "news", "web"} else "source_text"
            evidence_passages = [
                EvidencePassage(
                    source_url=source_url,
                    text=content,
                    retrieval_method=method,
                    content_origin=origin,
                )
            ]
        return ToolOutcome(
            content_markdown=content,
            links=links or ([source_url] if source_url else []),
            method=method,
            status=status,
            evidence_passages=evidence_passages,
        )

    return _handler


async def _run_evidence_proposal(
    outcome: ToolOutcome,
    findings: list[dict[str, Any]],
    *,
    tool_name: str = "search_web",
    briefing: str = "briefing with no URLs",
) -> tuple[Any, FakeLlm]:
    async def source_tool(**_: Any) -> ToolOutcome:
        return outcome

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("source", tool_name)]),
            _response(tool_calls=[_tool_call("record", "record_findings", {"findings": findings})]),
            _response(tool_calls=[_tool_call("done", "conclude")]),
        ]
    )
    result = await run_agentic_loop(
        "system",
        briefing,
        [_tool_spec(tool_name, source_tool)],
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
    )
    return result, fake_llm


class TestProvenanceGate:
    """Text findings cite code-delivered passages; the loop owns their source and quote."""

    @pytest.mark.asyncio
    async def test_local_navigation_inventory_cannot_ground_a_finding(self) -> None:
        inventory_url = "https://agency.example/workbook.xlsx#sheet=Totals"
        outcome = ToolOutcome(
            content_markdown=f"Totals sheet inventory entry: {inventory_url}",
            method="local_navigation",
            links=[inventory_url],
        )
        result, _ = await _run_evidence_proposal(outcome, [_finding()], tool_name="inventory")

        assert result.telemetry.findings_count == 0
        assert result.telemetry.provenance_rejections == 1
        assert "unknown or undelivered evidence_ref" in _tool_messages(result)[2]["content"]
        assert result.delivered_evidence == []

    @pytest.mark.asyncio
    async def test_local_navigation_does_not_erase_evidence_delivered_by_search(self) -> None:
        source_url = "https://agency.example/report"
        search = _search_returning(f"{source_url} Minutes")
        inventory = _returns_method(
            f"Navigation only: {source_url}",
            method="local_navigation",
            links=[source_url],
        )
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("search", "search_web", {"query": "report"})]),
                _response(tool_calls=[_tool_call("inventory", "inventory", {"url": source_url})]),
                _response(tool_calls=[_tool_call("record", "record_findings", {"findings": [_finding()]})]),
                _response(tool_calls=[_tool_call("done", "conclude")]),
            ]
        )
        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_web", search), _tool_spec("inventory", inventory)],
            _config(max_steps=6, max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.provenance_rejections == 0
        assert result.delivered_evidence[0].source_url == source_url

    @pytest.mark.asyncio
    async def test_legacy_raw_quote_proposal_is_rejected(self) -> None:
        source_url = "https://agency.example/report"
        outcome = ToolOutcome(
            content_markdown="Minutes were published on July 1.",
            method="plain",
            evidence_passages=[
                EvidencePassage(
                    source_url=source_url,
                    text="Minutes were published on July 1.",
                    retrieval_method="plain",
                    content_origin="source_text",
                )
            ],
        )
        legacy_finding = {
            "claim": "The board published the minutes.",
            "source_url": source_url,
            "quote": "Minutes were published on July 1.",
            "retrieved_how": "fetch",
        }
        result, _ = await _run_evidence_proposal(outcome, [legacy_finding], tool_name="fetch")

        assert result.telemetry.findings_count == 0
        assert result.telemetry.provenance_rejections == 1
        assert "source evidence fields are code-owned: quote, retrieved_how" in _tool_messages(result)[2]["content"]
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_delivered_passage_is_materialized_without_quote_normalization(self) -> None:
        source_url = "https://agency.example/report.pdf"
        awkward_text = "The extraction joins containsn and preserves sex-\nspecific notation with 10² cases."
        outcome = ToolOutcome(
            content_markdown="The report contains a table.",
            method="read_document",
            evidence_passages=[
                EvidencePassage(
                    source_url=source_url,
                    text=awkward_text,
                    retrieval_method="read_document",
                    content_origin="source_text",
                )
            ],
        )
        result, _ = await _run_evidence_proposal(outcome, [_finding(claim="The report includes this passage.")])

        expected_quote = "\n".join("> " + line if line else ">" for line in awkward_text.splitlines())
        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 0
        assert source_url in result.findings_markdown
        assert expected_quote in result.findings_markdown
        assert "containsn" in result.findings_markdown
        assert "sex-\nspecific" not in result.findings_markdown
        assert "sex-\n> specific" in result.findings_markdown
        assert "10²" in result.findings_markdown
        assert "10^2" not in result.findings_markdown

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "passage_text",
        [
            pytest.param(
                '"MARC L. ANDREESSEN | CONTRIBUTION | 02/11/2026 | 12500000.00"; '
                '"BENJAMIN HOROWITZ | CONTRIBUTION | 02/11/2026 | 12500000.00"',
                id="semicolon-separated-table-rows",
            ),
            pytest.param(
                '"MARC L. ANDREESSEN | CONTRIBUTION | 02/11/2026 | 12500000.00", '
                '"BENJAMIN HOROWITZ | CONTRIBUTION | 02/11/2026 | 12500000.00"',
                id="comma-separated-table-rows",
            ),
            pytest.param("`Windows | 56.61%` and `Linux | 4.36%`", id="markdown-table-cells"),
            pytest.param(
                "Background: a study followed records.\n\nResults: 89%, 44%, and 13%.", id="blank-line-boundary"
            ),
            pytest.param("The rate parameter of the ex-\nponential distribution is 0.733.", id="hyphenated-word-wrap"),
            pytest.param("Deaths during 2020\u2013\n2100 across all 13 countries.", id="wrapped-numeric-range"),
            pytest.param("signiﬁcantly distorting the number of people alive", id="typographic-ligature"),
            pytest.param("The estimate rose to 10² cases per site.", id="superscript-digit"),
            pytest.param("The rate is 4.1 percent.", id="short-numeric-fact"),
            pytest.param("Total revenue was 4.2 billion dollars, up 47.3% year over year.", id="multiple-figures"),
            pytest.param("Results: Windows 56.61%; unknown 21.45%; Linux 4.36%.", id="intervening-table-row"),
            pytest.param("\u2018The rate was 4.1 percent.\u2019", id="curly-quote-glyphs"),
            pytest.param("“The rate   was 4.1 percent.”", id="source-whitespace-and-glyphs"),
            pytest.param("First observed value 10.3. ... Later value 12.8.", id="ellipsis-separated-values"),
            pytest.param("First observed value 10.3. … Later value 12.8.", id="unicode-ellipsis"),
            pytest.param("The count was 12\nper 100,000 residents.", id="line-wrapped-denominator"),
            pytest.param("Table 2: \tRate\tCount\n2025\t5.2%\t1,240", id="tabular-spacing"),
            pytest.param("It containsn a compacted run and observeP capitalization.", id="compacted-pdf-runs"),
            pytest.param("The value was 10-\n20 percent, not 1020 percent.", id="numeric-hyphen-preserved"),
            pytest.param("The report covers ages 5\u20139 and 10\u201314 years.", id="ranges-and-units"),
        ],
    )
    async def test_delivered_reference_preserves_extracted_text_exactly(self, passage_text: str) -> None:
        outcome = ToolOutcome(
            content_markdown="Source passage available.",
            method="read_document",
            evidence_passages=[
                EvidencePassage(
                    source_url="https://agency.example/report.pdf",
                    text=passage_text,
                    retrieval_method="read_document",
                    content_origin="source_text",
                )
            ],
        )
        result, _ = await _run_evidence_proposal(outcome, [_finding()])

        expected_quote = "\n".join("> " + line if line else ">" for line in passage_text.splitlines())
        assert result.telemetry.findings_count == 1
        assert expected_quote in result.findings_markdown
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_source_url_cannot_be_restamped_away_from_referenced_passage(self) -> None:
        source_a = "https://agency.example/a"
        source_b = "https://agency.example/b"
        outcome = ToolOutcome(
            content_markdown="The rate was 5.2%.",
            method="search",
            evidence_passages=[
                EvidencePassage(
                    source_url=source_a,
                    text="The rate was 5.2%.",
                    retrieval_method="search",
                    content_origin="provider_summary",
                )
            ],
        )
        result, _ = await _run_evidence_proposal(outcome, [{**_finding(), "source_url": source_b}])

        assert result.telemetry.findings_count == 0
        assert result.telemetry.provenance_rejections == 1
        assert "source_url disagrees with cited evidence_ref source" in _tool_messages(result)[2]["content"]

    @pytest.mark.asyncio
    async def test_adapter_source_url_is_rendered_exactly(self) -> None:
        source_url = "https://agency.example/report/?utm_source=archive#table-2"
        outcome = ToolOutcome(
            content_markdown="A search result points to the report.",
            method="search",
            evidence_passages=[
                EvidencePassage(
                    source_url=source_url,
                    text="The table reports 5.2%.",
                    retrieval_method="search",
                    content_origin="provider_summary",
                )
            ],
        )
        result, _ = await _run_evidence_proposal(outcome, [_finding()])

        assert result.telemetry.findings_count == 1
        assert source_url in result.findings_markdown
        assert "https://agency.example/report" not in result.findings_markdown.replace(source_url, "")

    @pytest.mark.asyncio
    async def test_reference_rejection_counters_remain_in_completion_marker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        outcome = ToolOutcome(
            content_markdown="The report states the figure.",
            method="search",
            evidence_passages=[
                EvidencePassage(
                    source_url="https://agency.example/report",
                    text="The report states the figure.",
                    retrieval_method="search",
                    content_origin="provider_summary",
                )
            ],
        )
        result, _ = await _run_evidence_proposal(
            outcome,
            [{**_finding(), "source_url": "https://agency.example/other"}],
        )

        marker = next(record.getMessage() for record in caplog.records if "GAP_FILL_V2:" in record.getMessage())
        assert result.telemetry.provenance_rejections == 1
        assert "provenance_rejections=1" in marker
        assert "quote_mismatch_warnings=0" in marker

    @pytest.mark.asyncio
    @pytest.mark.parametrize("discrepancy", [False, True], ids=["ordinary-finding", "discrepancy"])
    async def test_briefing_url_without_a_delivered_passage_cannot_ground_text(self, discrepancy: bool) -> None:
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _tool_call(
                            "record",
                            "record_findings",
                            {"findings": [_finding(discrepancy=discrepancy)]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done", "conclude")]),
            ]
        )
        result = await run_agentic_loop(
            "system",
            "briefing cites https://gov.example/report as a source",
            [],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 0
        assert result.telemetry.provenance_rejections == 1
        assert "unknown or undelivered evidence_ref" in _tool_messages(result)[0]["content"]

    @pytest.mark.asyncio
    async def test_multiple_sources_keep_each_passage_and_origin_attached(self) -> None:
        source_a = "https://agency.example/report"
        source_b = "https://reader.example/answer"
        outcome = ToolOutcome(
            content_markdown="Two relevant results.",
            method="search",
            evidence_passages=[
                EvidencePassage(
                    source_url=source_a,
                    text="The report states the rate was 5.2%.",
                    retrieval_method="read_document",
                    content_origin="source_text",
                ),
                EvidencePassage(
                    source_url=source_b,
                    text="The reader answer reports the same figure.",
                    retrieval_method="read_document",
                    content_origin="generated_answer",
                ),
            ],
        )
        result, _ = await _run_evidence_proposal(
            outcome,
            [{**_finding(claim="Both delivered observations report 5.2%.", evidence_refs=("ev_1", "ev_2"))}],
        )

        assert result.telemetry.findings_count == 1
        assert source_a in result.findings_markdown
        assert source_b in result.findings_markdown
        assert "Evidence (source text; read_document; ev_1)" in result.findings_markdown
        assert "Evidence (generated answer; read_document; ev_2)" in result.findings_markdown
        assert "The report states the rate was 5.2%." in result.findings_markdown
        assert "The reader answer reports the same figure." in result.findings_markdown

    @pytest.mark.asyncio
    async def test_oversize_passage_is_not_delivered_or_citable(self) -> None:
        source_url = "https://agency.example/report"
        outcome = ToolOutcome(
            content_markdown="The source was fetched.",
            method="plain",
            evidence_passages=[
                EvidencePassage(
                    source_url=source_url,
                    text="A" * (GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS + 1),
                    retrieval_method="plain",
                    content_origin="source_text",
                )
            ],
        )
        result, _ = await _run_evidence_proposal(outcome, [_finding()], tool_name="fetch")

        assert result.telemetry.findings_count == 0
        assert result.delivered_evidence == []
        assert "unknown or undelivered evidence_ref" in _tool_messages(result)[2]["content"]
        assert "No citable passage was delivered" in _tool_messages(result)[1]["content"]

    @pytest.mark.asyncio
    async def test_combined_passages_over_finding_limit_are_rejected(self) -> None:
        outcome = ToolOutcome(
            content_markdown="Two delivered passages.",
            method="plain",
            evidence_passages=[
                EvidencePassage(
                    source_url="https://agency.example/a",
                    text="A" * GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS,
                    retrieval_method="plain",
                    content_origin="source_text",
                ),
                EvidencePassage(
                    source_url="https://agency.example/b",
                    text="B" * GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS,
                    retrieval_method="plain",
                    content_origin="source_text",
                ),
            ],
        )
        finding = _finding(claim="The two passages jointly support the claim.", evidence_refs=("ev_1", "ev_2"))
        result, _ = await _run_evidence_proposal(outcome, [finding])

        assert result.telemetry.findings_count == 0
        assert result.telemetry.provenance_rejections == 1
        assert f"exceeds {GAP_FILL_V2_EVIDENCE_FINDING_MAX_CHARS} characters" in _tool_messages(result)[2]["content"]


# These helpers remain useful for reading and checking historical raw-quote
# records. Live text findings use delivered references and never call this
# matcher; the cases below preserve its standalone behavior for legacy readers.
_ARCHIVED_STITCH_JOINERS: list[tuple[str, int, bool]] = [
    ('"; "', 198, True),
    ("...", 155, False),
    ("”; “", 115, True),
    ("” ... “", 52, False),
    ('", "', 50, True),
    ("” “", 49, False),
    ('" ... "', 40, False),
    ("” and “", 34, True),
    ('" "', 32, False),
    ("`; `", 27, True),
    ('" and "', 25, True),
    ("…", 19, False),
    ('": "', 18, True),
    ("”\n\n“", 13, False),
    ("” … “", 13, False),
]

_CORPUS_SPAN_A = "The agency published the schedule on 11 February 2026"
_CORPUS_SPAN_B = "the review board recorded no objections that quarter"
_CORPUS_BODY = _normalize_quote_text(
    f"Filing index. {_CORPUS_SPAN_A}, per the docket. Several unrelated paragraphs of "
    f"procedural text follow here. Later, {_CORPUS_SPAN_B}, closing the item."
)


class TestArchivedStitchShapesGround:
    """Preserve the historical quote matcher checks without admitting raw live proposals."""

    @pytest.mark.parametrize(
        ("joiner", "archived_count", "missed_before"),
        [pytest.param(*case, id=repr(case[0])) for case in _ARCHIVED_STITCH_JOINERS],
    )
    def test_stitched_shape_grounds_when_both_spans_are_verbatim(
        self, joiner: str, archived_count: int, missed_before: bool
    ) -> None:
        del archived_count, missed_before
        quote = f'"{_CORPUS_SPAN_A}{joiner}{_CORPUS_SPAN_B}"'
        assert _quote_is_grounded(quote, _CORPUS_BODY)

    @pytest.mark.parametrize(
        ("joiner", "archived_count", "missed_before"),
        [pytest.param(*case, id=repr(case[0])) for case in _ARCHIVED_STITCH_JOINERS],
    )
    def test_stitched_shape_rejects_a_fabricated_span(
        self, joiner: str, archived_count: int, missed_before: bool
    ) -> None:
        del archived_count, missed_before
        quote = f'"{_CORPUS_SPAN_A}{joiner}A FABRICATED CLAUSE THAT IS ABSENT FROM THE SOURCE"'
        assert not _quote_is_grounded(quote, _CORPUS_BODY)

    def test_long_narration_joiner_still_reads_as_one_span_and_warns(self) -> None:
        source_body = _normalize_quote_text("Musk has personally donated more than $85 million toward the midterms.")
        quote = (
            'The briefing says: "Elon Musk — $90.6 million." The fetched report instead says: '
            '"Musk has personally donated more than $85 million toward the midterms."'
        )
        assert not _quote_is_grounded(quote, source_body)

    def test_span_ending_in_whitespace_before_its_glyph_still_splits(self) -> None:
        source_body = _normalize_quote_text(
            "Contribution schedules list MARC L. ANDREESSEN in the filing index. Later pages "
            "name BENJAMIN HOROWITZ under the same schedule."
        )
        quote = '"MARC L. ANDREESSEN in the filing " "name BENJAMIN HOROWITZ under"'
        assert _quote_is_grounded(quote, source_body)

    def test_the_joiner_bound_splits_at_the_constant_and_not_one_past_it(self) -> None:
        source_body = _normalize_quote_text(
            "Contribution schedules list MARC L. ANDREESSEN in the filing index. Later pages "
            "name BENJAMIN HOROWITZ under the same schedule."
        )
        span_a = '"MARC L. ANDREESSEN in the filing"'
        span_b = '"name BENJAMIN HOROWITZ under"'
        at_bound = "x" * _SPAN_JOINER_MAX_CHARS
        past_bound = "x" * (_SPAN_JOINER_MAX_CHARS + 1)
        assert _quote_is_grounded(f"{span_a}{at_bound}{span_b}", source_body)
        assert not _quote_is_grounded(f"{span_a}{past_bound}{span_b}", source_body)

    def test_all_short_nonnumeric_spans_fall_back_to_the_whole_quote(self) -> None:
        quote = '"one fact" and "two"'
        absent = _normalize_quote_text("The filing lists one fact in the appendix; nothing else was disclosed.")
        present = _normalize_quote_text("The filing lists one fact and two others in the appendix.")
        assert not _quote_is_grounded(quote, absent)
        assert _quote_is_grounded(quote, present)

    def test_short_fabricated_numeric_span_still_warns(self) -> None:
        source_body = _normalize_quote_text(
            "https://agency.example/report — The unemployment rate reached a new high this quarter, "
            "officials said. The figure was 3.9%."
        )
        quote = "The unemployment rate reached a new high this quarter ... 47.3%"
        assert not _quote_is_grounded(quote, source_body)


class TestLineLevelQuoteShapesGround:
    def test_newline_stitched_lines_ground_when_each_line_is_verbatim(self) -> None:
        quote = f"{_CORPUS_SPAN_B}\n{_CORPUS_SPAN_A}"
        assert _quote_is_grounded(quote, _CORPUS_BODY)

    def test_newline_stitched_quote_rejects_a_fabricated_line(self) -> None:
        quote = f"{_CORPUS_SPAN_A}\nA FABRICATED CLAUSE THAT IS ABSENT FROM THE SOURCE"
        assert not _quote_is_grounded(quote, _CORPUS_BODY)

    def test_newline_stitched_quote_rejects_a_fabricated_figure(self) -> None:
        quote = f"{_CORPUS_SPAN_A}\n47.3%"
        assert not _quote_is_grounded(quote, _CORPUS_BODY)

    def test_line_break_hyphenation_in_the_source_grounds_the_joined_word(self) -> None:
        corpus = _normalize_quote_text("the rate parameter of the ex-\nponential distribution is 0.733")
        assert _quote_is_grounded("the rate parameter of the exponential distribution is 0.733", corpus)

    def test_wrapped_numeric_range_in_the_source_grounds_the_unbroken_range(self) -> None:
        corpus = _normalize_quote_text(
            "individuals dying in the period 2020\u2013\n2100 from any of the 13 IDL countries"
        )
        assert _quote_is_grounded(
            "individuals dying in the period 2020\u20132100 from any of the 13 IDL countries", corpus
        )

    def test_typographic_ligature_in_the_quote_grounds_on_plain_letters(self) -> None:
        corpus = _normalize_quote_text("significantly distorting the number of people alive")
        assert _quote_is_grounded("signiﬁcantly distorting the number of people alive", corpus)

    def test_superscript_digits_are_not_folded(self) -> None:
        corpus = _normalize_quote_text("the estimate rose to 10² cases per site")
        assert not _quote_is_grounded("the estimate rose to 102 cases per site", corpus)

    def test_rejoining_never_alters_digits(self) -> None:
        corpus = _normalize_quote_text("estimates ranged from 10-\n20 percent across sites")
        assert not _quote_is_grounded("estimates ranged from 1020 percent across sites", corpus)


class TestEvidenceRetrievalMetadata:
    """Evidence references render the retrieval method and content origin delivered by tools."""

    @pytest.mark.asyncio
    async def test_fetched_source_text_is_shown_with_its_method(self) -> None:
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://agency.example/report"})]),
                _response(tool_calls=[_tool_call("rec1", "record_findings", {"findings": [_finding()]})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        fetch = _returns_method("The report body.", method="plain")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("fetch", fetch)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert "Evidence (source text; plain; ev_1)" in result.findings_markdown
        assert "The report body." in result.findings_markdown

    @pytest.mark.asyncio
    async def test_search_summary_is_shown_with_its_origin_and_method(self) -> None:
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(tool_calls=[_tool_call("rec1", "record_findings", {"findings": [_finding()]})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _returns_method("See https://agency.example/report for the figures.", method="search")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert "Evidence (provider summary; search; ev_1)" in result.findings_markdown
        assert "See https://agency.example/report for the figures." in result.findings_markdown

    @pytest.mark.asyncio
    async def test_search_then_fetch_finding_cites_the_fetched_passage(self) -> None:
        """A finding recorded after both retrievals cites the full source text."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://agency.example/report"})]),
                _response(
                    tool_calls=[
                        _tool_call("rec1", "record_findings", {"findings": [_finding(evidence_refs=("ev_2",))]})
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _returns_method("See https://agency.example/report — snippet only.", method="search")
        fetch = _returns_method("The full report body.", method="plain")

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_web", search), _tool_spec("fetch", fetch)],
            _config(),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        assert "Evidence (source text; plain; ev_2)" in result.findings_markdown
        assert "The full report body." in result.findings_markdown
        assert "Evidence (provider summary; search; ev_1)" not in result.findings_markdown

    @pytest.mark.asyncio
    async def test_failed_fetch_then_snippet_leaves_discrepancy_demoted(self) -> None:
        """The 131.3 scenario end-to-end: the direct fetch 403s (no tier), a
        search snippet then surfaces the contradicting figure, and the discrepancy
        banked on that URL is snippet-tier — so it renders in the DEMOTED block,
        never superseding the briefing."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://gov.example/figure"})]),
                _response(tool_calls=[_tool_call("s1", "search_news", {"query": "figure"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding(discrepancy=True)]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        # The direct fetch is blocked (403) — grants no tier but still marks the
        # URL tool-seen, so the discrepancy clears the provenance gate.
        blocked_fetch = _returns_method("403 Forbidden", method="plain", status="blocked")
        # The search snippet surfaces the contradicting figure at the same URL.
        snippet_search = _returns_method("https://gov.example/figure says the figure is 131.3.", method="news")

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("fetch", blocked_fetch), _tool_spec("search_news", snippet_search)],
            _config(),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        md = result.findings_markdown
        assert "### ⚠ Driver-reported corrections (limited evidence)" in md
        assert "### ⚠ Driver-reported corrections to the briefing" not in md
        assert "recheck the underlying sources" in md
        assert "Evidence (provider summary; news; ev_1)" in md
        assert "supersede" not in md.lower()

    @pytest.mark.asyncio
    async def test_driver_supplied_tier_is_rejected_as_code_owned(self) -> None:
        """A text proposal cannot set the code-owned verification tier."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    {
                                        **_finding(),
                                        "verification_tier": "fetched",  # code-owned field
                                    }
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _returns_method("See https://agency.example/report for the figures.", method="search")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 0
        assert result.telemetry.provenance_rejections == 1
        assert "verification_tier" in _tool_messages(result)[2]["content"]

    @pytest.mark.asyncio
    async def test_explicit_replacement_uses_the_newly_fetched_evidence(self) -> None:
        """A changed evidence reference replaces the earlier finding explicitly.

        The finding keeps one stable identity, and the rendered correction remains
        driver-reported so forecasters compare it with the briefing and excerpt.
        """
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_news", {"query": "figure"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding(discrepancy=True)]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://gov.example/figure"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "done1",
                            "conclude",
                            {
                                "final_findings": [
                                    {
                                        **_finding(discrepancy=True, evidence_refs=("ev_2",)),
                                        "replaces_finding_id": "finding_1",
                                    }
                                ]
                            },
                        )
                    ]
                ),
            ]
        )
        snippet_search = _returns_method("https://gov.example/figure says the figure is 131.3.", method="news")
        fetch = _returns_method("The full primary-source page.", method="plain")

        # cap=0 keeps the W2 conclude gate out of the way (this test exercises
        # explicit replacement, not gap accounting).
        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_news", snippet_search), _tool_spec("fetch", fetch)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        md = result.findings_markdown
        assert "### ⚠ Driver-reported corrections to the briefing" in md
        assert "### ⚠ Driver-reported corrections (limited evidence)" not in md
        assert "The full primary-source page." in md
        assert "supersede" not in md.lower()
        assert "Finding ID: finding_1" in md
        conclude_message = _tool_messages(result)[-1]
        assert "Concluded with 1 final finding(s)" in conclude_message["content"]
        assert "Skipped 1 final finding(s)" not in conclude_message["content"]

    def test_duplicate_keeps_tier_bound_to_its_cited_support(self) -> None:
        """A later fetch does not restamp a finding that still cites a snippet."""
        state = _LoopState(messages=[], started_at_s=0.0, deadline_at_s=100.0)
        finding = Finding(
            claim="The briefing reports one figure and the source reports another.",
            source_url="https://gov.example/figure",
            quote="A snippet reports the figure.",
            retrieved_how="search_news",
            discrepancy=True,
            supports=[
                FindingSupport(
                    source_url="https://gov.example/figure",
                    quote="A snippet reports the figure.",
                    retrieved_how="search_news",
                    content_origin="provider_summary",
                    evidence_ref="ev_1",
                )
            ],
        )
        state.url_best_tier["https://gov.example/figure"] = "snippet"

        assert _bank_findings(state, [finding]) == (1, 0)
        assert state.findings[0].verification_tier == "snippet"

        # Same tier map: duplicate skipped, stamp unchanged.
        assert _bank_findings(state, [finding]) == (0, 1)
        assert state.findings[0].verification_tier == "snippet"

        # A later URL fetch does not change which evidence this finding cites.
        state.url_best_tier["https://gov.example/figure"] = "fetched"
        assert _bank_findings(state, [finding]) == (0, 1)
        assert len(state.findings) == 1
        assert state.findings[0].verification_tier == "snippet"


class TestResearchPlanGate:
    """W1: set_research_plan is required before any external tool call. Until it
    runs, external calls are rejected with a nudge; the plan-nudge cap forces a
    soft-continue (plan_skipped) so a driver that never plans can't wedge. The
    plan also emits the GHOST_PRE / GHOST_PRE_JSON pre-research telemetry."""

    @pytest.mark.asyncio
    async def test_external_tool_before_plan_is_rejected_with_nudge(self) -> None:
        invoked: list[str] = []

        async def search_web(**kwargs: Any) -> ToolOutcome:
            invoked.append(kwargs.get("query", ""))
            return ToolOutcome(content_markdown="ran", method="search")

        # Turn 1 calls an external tool with no plan set -> rejected. Turn 2 sets
        # the plan. Turn 3's search then runs. Turn 4 concludes.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("early", "search_web", {"query": "too soon"})]),
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "now allowed"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # W1 test: cap=0 disables the W2 conclude gate so the early conclude here
        # isn't blocked (the plan-gate nudge is what this exercises).
        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        # The premature call was rejected: its handler never ran, and its tool
        # message carries the plan-required nudge.
        assert invoked == ["now allowed"]
        early_message = _tool_messages(result)[0]
        assert early_message["tool_call_id"] == "early"
        assert "status: error" in early_message["content"]
        assert "call set_research_plan first" in early_message["content"]
        # A rejected pre-plan call is NOT counted against the tool-call budget.
        assert result.telemetry.per_tool_counts.get("search_web") == 1
        assert result.telemetry.plan_skipped is False
        assert "The" not in result.findings_markdown or result.telemetry.findings_count == 0

    @pytest.mark.asyncio
    async def test_plan_accepted_then_external_tool_runs(self) -> None:
        invoked: list[str] = []

        async def search_web(**kwargs: Any) -> ToolOutcome:
            invoked.append(kwargs.get("query", ""))
            return ToolOutcome(content_markdown="ran", method="search")

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "allowed"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # W1 test: cap=0 disables the W2 conclude gate (early conclude, no
        # gap_accounting) so the plan-accepted path is what's under test.
        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert invoked == ["allowed"]
        assert result.telemetry.plan_gaps == 1
        assert result.telemetry.plan_skipped is False

    @pytest.mark.asyncio
    async def test_plan_emits_ghost_pre_json_marker(self, caplog: pytest.LogCaptureFixture) -> None:
        """The dry-run forecast passed to set_research_plan is logged as GHOST_PRE
        / GHOST_PRE_JSON — the pre-research counterpart to the concluding ghost."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _plan_call(
                            gaps=[{"id": "g1", "question": "q?"}],
                            sensitive_assumptions=["the incumbent runs again", "turnout stays flat"],
                            dry_run_forecast={"question_type": "binary", "posterior_prob": 0.37},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # cap=0 disables the W2 gate: this test asserts the GHOST_PRE telemetry,
        # not the conclude gate, and concludes without research/accounting.
        await run_agentic_loop(
            "system",
            "user",
            [],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
            log_prefix="question=https://q/1/ ",
        )

        pre_json_lines = [m.getMessage() for m in caplog.records if "GHOST_PRE_JSON:" in m.getMessage()]
        assert len(pre_json_lines) == 1
        blob = pre_json_lines[0].split("GHOST_PRE_JSON:", 1)[1].strip()
        assert json.loads(blob) == {"qtype": "binary", "prob": 0.37}
        # The GHOST_PRE summary line carries the plan shape and the question ref.
        pre_lines = [m.getMessage() for m in caplog.records if "GHOST_PRE:" in m.getMessage()]
        # GHOST_PRE_JSON also contains "GHOST_PRE" as a substring, so filter it out.
        pre_summary = [m for m in pre_lines if "GHOST_PRE_JSON:" not in m]
        assert len(pre_summary) == 1
        assert "gaps=1" in pre_summary[0]
        assert "sensitive_assumptions=2" in pre_summary[0]
        assert "question=https://q/1/" in pre_summary[0]

    @pytest.mark.asyncio
    async def test_ghost_pre_json_suppressed_when_no_dry_run(self, caplog: pytest.LogCaptureFixture) -> None:
        """No dry_run_forecast -> the GHOST_PRE summary still fires but the JSON
        companion is suppressed (nothing to serialize), mirroring the ghost path."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # cap=0 disables the W2 gate: GHOST_PRE-suppression telemetry is the subject.
        await run_agentic_loop("system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm)

        assert any("GHOST_PRE:" in m.getMessage() for m in caplog.records)
        assert not any("GHOST_PRE_JSON:" in m.getMessage() for m in caplog.records)
        # No dry run supplied at all is the LEGITIMATE suppression: no WARN.
        assert not any("GHOST_PRE_JSON suppressed" in m.getMessage() for m in caplog.records)

    @pytest.mark.asyncio
    async def test_ghost_pre_json_suppression_on_unparseable_dry_run_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A dry run that WAS supplied but fails to parse into a structured forecast
        (the observed prod case: flat declared percentiles failing schema validation,
        run 30718626314) suppresses GHOST_PRE_JSON — and that loss must be countable,
        because it drops exactly the flattest pre-research views, biasing the archived
        zero-move rate. GHOST_PRE still fires; the JSON half is replaced by a WARN."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _plan_call(
                            gaps=[{"id": "g1", "question": "q?"}],
                            dry_run_forecast={"not_a_recognizable_block": True},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        await run_agentic_loop("system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm)

        assert any("GHOST_PRE:" in m.getMessage() for m in caplog.records)
        assert not any("GHOST_PRE_JSON:" in m.getMessage() for m in caplog.records)
        warns = [r for r in caplog.records if "GHOST_PRE_JSON suppressed" in r.getMessage()]
        assert len(warns) == 1
        assert warns[0].levelno == logging.WARNING

    @pytest.mark.asyncio
    async def test_ghost_pre_json_suppression_on_non_dict_dry_run_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """The other half of the same loss: a driver that answers `dry_run_forecast`
        with a bare scalar rather than a forecast block. It is discarded by the
        isinstance check before any parse is attempted, so only the "was something
        supplied at all" test separates it from the legitimate no-dry-run case — and
        it must count, for the same non-random-loss reason."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _tool_call(
                            "plan0",
                            "set_research_plan",
                            {"gaps": [{"id": "g1", "question": "q?"}], "dry_run_forecast": "0.42"},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        await run_agentic_loop("system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm)

        assert not any("GHOST_PRE_JSON:" in m.getMessage() for m in caplog.records)
        warns = [r for r in caplog.records if "GHOST_PRE_JSON suppressed" in r.getMessage()]
        assert len(warns) == 1
        assert warns[0].levelno == logging.WARNING

    @pytest.mark.asyncio
    async def test_plan_nudge_cap_soft_continues_without_plan(self, caplog: pytest.LogCaptureFixture) -> None:
        """A driver that never plans hits the plan-nudge cap (2), after which the
        loop soft-continues: the external tool runs un-gated and plan_skipped is
        logged. This is the anti-wedge guard."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        invoked: list[str] = []

        async def search_web(**kwargs: Any) -> ToolOutcome:
            invoked.append(kwargs.get("query", ""))
            return ToolOutcome(content_markdown="ran", method="search")

        # Three external-only turns: turns 1 and 2 are plan-rejected (2 nudges ==
        # cap), turn 3 runs un-gated, turn 4 concludes.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("c1", "search_web", {"query": "q1"})]),
                _response(tool_calls=[_tool_call("c2", "search_web", {"query": "q2"})]),
                _response(tool_calls=[_tool_call("c3", "search_web", {"query": "q3"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("search_web", search_web)], _config(max_steps=6), llm_call=fake_llm
        )

        # First two calls were rejected pre-plan; the third ran once the cap
        # forced a soft-continue.
        assert invoked == ["q3"]
        assert result.telemetry.plan_skipped is True
        assert result.telemetry.plan_gaps == 0
        marker = next(m.getMessage() for m in caplog.records if "GAP_FILL_V2:" in m.getMessage())
        assert "plan_skipped=True" in marker
        assert "plan_gaps=0" in marker

    @pytest.mark.asyncio
    async def test_plan_gaps_appear_in_budget_line(self) -> None:
        """After a plan is set, the per-turn budget line lists the plan's gap ids,
        the work-list conclude's gap_accounting must cover (W1)."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="ran", method="search")

        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _plan_call(gaps=[{"id": "gap-a", "question": "q1?"}, {"id": "gap-b", "question": "q2?"}])
                    ]
                ),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # cap=0 disables the W2 gate: the budget-line work-list (W1) is the subject.
        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        # The search result's budget line carries both gap ids.
        search_message = _tool_messages(result)[1]
        budget_line = next(line for line in search_message["content"].splitlines() if line.startswith("[budget: "))
        assert "plan_gaps=[gap-a, gap-b]" in budget_line
        assert "unaddressed_gaps" not in budget_line

    @pytest.mark.asyncio
    async def test_gap_list_capped_at_max_gaps(self) -> None:
        """set_research_plan drops gaps beyond config.max_gaps (the ranked tail),
        keeping the top-N; plan_gaps reflects the kept count."""
        gaps = [{"id": f"g{i}", "question": f"q{i}?"} for i in range(6)]
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=gaps)]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # cap=0 disables the W2 gate: the max_gaps cap (W1) is the subject here.
        result = await run_agentic_loop(
            "system", "user", [], _config(max_tool_calls=14, max_conclude_gate_rejections=0), llm_call=fake_llm
        )

        # Default max_gaps is 4; only the top 4 survive.
        assert result.telemetry.plan_gaps == 4
        plan_message = _tool_messages(result)[0]
        assert "kept the top 4 of 6 gaps" in plan_message["content"]

    @pytest.mark.asyncio
    async def test_zero_gap_plan_is_rejected_and_w1_gate_stays_armed(self) -> None:
        """F3a: a zero-gap set_research_plan is rejected — research_plan is left
        None so W1's external-tool gate stays armed. A follow-up external tool
        call is still plan-rejected; the empty plan did not open the gate."""
        invoked: list[str] = []

        async def search_web(**kwargs: Any) -> ToolOutcome:
            invoked.append(kwargs.get("query", ""))
            return ToolOutcome(content_markdown="ran", method="search")

        # Turn 1: empty-gap plan (rejected). Turn 2: an external tool that must
        # still be plan-gated (only 1 nudge so far, cap not yet hit). Turn 3: a
        # real plan. Turn 4: the external tool finally runs. Turn 5: conclude.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call("p0", gaps=[])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "too soon"})]),
                _response(tool_calls=[_plan_call("p1", gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("s2", "search_web", {"query": "now allowed"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=7, max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        # The empty plan was rejected and did NOT open the gate, so the turn-2
        # search was plan-rejected; only the post-real-plan search ran.
        assert invoked == ["now allowed"]
        plan_reject = _tool_messages(result)[0]
        assert plan_reject["tool_call_id"] == "p0"
        assert "status: error" in plan_reject["content"]
        assert "Research plan rejected" in plan_reject["content"]
        early_search = _tool_messages(result)[1]
        assert early_search["tool_call_id"] == "s1"
        assert "call set_research_plan first" in early_search["content"]
        assert result.telemetry.plan_gaps == 1
        assert result.telemetry.plan_skipped is False

    @pytest.mark.asyncio
    async def test_replan_to_empty_does_not_clobber_a_prior_valid_plan(self) -> None:
        """F3a corollary: once a valid plan is set, a later empty re-plan is
        rejected rather than clobbering it — plan_gaps keeps the real count."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call("p0", gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_plan_call("p1", gaps=[])]),
                _response(tool_calls=[_tool_call("done1", "conclude", {"gap_accounting": _accounting("g1")})]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [], _config(max_steps=6, max_conclude_gate_rejections=0), llm_call=fake_llm
        )

        assert result.telemetry.plan_gaps == 1
        replan = _tool_messages(result)[1]
        assert "Research plan rejected" in replan["content"]


class TestConcludeGate:
    """W2: the conclude gate blocks an EARLY conclusion until the driver's
    gap_accounting covers every plan gap, the run made ≥1 external call per gap,
    and the fetch floor is met (top-2 gaps cite a fetch, OR ≥2 fetches/reads).
    A rejection continues the loop (no stop, no banking) and bumps
    conclude_gate_rejections; the cap accepts unconditionally after 2 rejections;
    must_conclude (deadline) and plan_skipped bypass it entirely."""

    @pytest.mark.asyncio
    async def test_reject_missing_gap_id_then_accept_on_complete_accounting(self) -> None:
        """A conclude whose accounting omits a plan gap is rejected (loop
        continues); a follow-up conclude that covers every gap is accepted."""

        async def fetch(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="body", method="rendered")

        # Plan has two gaps; two fetches meet the per-gap call count + fetch floor.
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(
                    tool_calls=[
                        _tool_call("f1", "fetch", {"url": "https://a.example"}),
                        _tool_call("f2", "fetch", {"url": "https://b.example"}),
                    ]
                ),
                # First conclude accounts for only g1 -> rejected.
                _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1")})]),
                # Second conclude accounts for both -> accepted.
                _response(tool_calls=[_tool_call("c2", "conclude", {"gap_accounting": _accounting("g1", "g2")})]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("fetch", fetch)], _config(max_steps=8), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 1
        first_conclude = _tool_messages(result)[-2]
        assert "Conclude rejected" in first_conclude["content"]
        assert "missing entries for gap(s): g2" in first_conclude["content"]
        # The accepted conclude actually stopped the loop.
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_reject_too_few_external_calls(self) -> None:
        """Two plan gaps but only one external tool call: the ≥1-call-per-gap
        invariant is unmet, so the conclude is rejected even with full accounting."""

        async def fetch(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="body", method="rendered")

        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://a.example"})]),
                # cap=0 keeps the rejection observable (the loop would otherwise
                # loop; here it soft-fails after the single scripted conclude).
                _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1", "g2")})]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("fetch", fetch)],
            _config(max_steps=6, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 1
        conclude_message = _tool_messages(result)[-1]
        assert "only 1 external tool call(s) made for 2 plan gap(s)" in conclude_message["content"]
        # Rejected -> the loop did not conclude early; it soft-failed on exhaustion.
        assert result.telemetry.concluded_early is False

    @pytest.mark.asyncio
    async def test_reject_when_both_fetch_floor_clauses_fail(self) -> None:
        """One gap, one external call, but it's a search (no fetch/read) and the
        accounting doesn't cite a fetch: both fetch-floor clauses fail -> reject."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "c1",
                            "conclude",
                            {"gap_accounting": _accounting("g1", actions="searched only, no primary source")},
                        )
                    ]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=6, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 1
        conclude_message = _tool_messages(result)[-1]
        assert "fetch floor unmet" in conclude_message["content"]

    @pytest.mark.asyncio
    async def test_accept_via_per_gap_fetch_clause(self) -> None:
        """Fetch-floor clause (a): the top-ranked gap's accounting cites a fetch,
        even though the run's own fetch count is below the global floor (one
        rendered fetch). The conclude is accepted."""

        async def fetch(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="body", method="rendered")

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://a.example"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "c1",
                            "conclude",
                            {"gap_accounting": _accounting("g1", actions="fetched the primary source PDF")},
                        )
                    ]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("fetch", fetch)], _config(max_steps=6), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_accept_via_global_fetch_count_clause(self) -> None:
        """Fetch-floor clause (b): the run made ≥2 fetches/reads, so a conclude
        whose accounting does NOT cite a fetch per gap is still accepted."""

        async def fetch(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="body", method="rendered")

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(
                    tool_calls=[
                        _tool_call("f1", "fetch", {"url": "https://a.example"}),
                        _tool_call("f2", "fetch", {"url": "https://b.example"}),
                    ]
                ),
                _response(
                    tool_calls=[
                        _tool_call(
                            "c1",
                            "conclude",
                            {"gap_accounting": _accounting("g1", actions="reviewed the search snippets")},
                        )
                    ]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("fetch", fetch)], _config(max_steps=6), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_reject_when_prose_cites_fetch_but_no_successful_retrieval(self) -> None:
        """Fetch-floor tightening: the top-ranked gaps' accounting cites fetch/read
        verbs, but every fetch 403'd (zero successful fetched-tier retrievals). The
        per-gap prose clause no longer clears the floor on its own -> reject."""

        async def fetch(**_: Any) -> ToolOutcome:
            # Every fetch 403s (status=blocked): no fetched tier is granted, so
            # url_best_tier stays empty and fetches_reads is 0 despite the calls.
            return ToolOutcome(content_markdown="Fetch blocked with HTTP 403.", method="plain", status="blocked")

        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(
                    tool_calls=[
                        _tool_call("f1", "fetch", {"url": "https://a.example"}),
                        _tool_call("f2", "fetch", {"url": "https://b.example"}),
                    ]
                ),
                _response(
                    tool_calls=[
                        _tool_call(
                            "c1",
                            "conclude",
                            {
                                "gap_accounting": _accounting(
                                    "g1", actions="fetched the primary source but received 403"
                                )
                                + _accounting("g2", actions="read_document timed out")
                            },
                        )
                    ]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("fetch", fetch)],
            _config(max_steps=6, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 1
        conclude_message = _tool_messages(result)[-1]
        assert "fetch floor unmet" in conclude_message["content"]
        # made 0 — both fetches 403'd, so no primary source was reached even though
        # the per-gap accounting narrates a fetch/read for each top gap.
        assert "made 0" in conclude_message["content"]
        assert result.telemetry.concluded_early is False

    @pytest.mark.asyncio
    async def test_accept_with_one_successful_fetch_below_global_floor(self) -> None:
        """The tightened per-gap clause still serves its purpose: one load-bearing
        fetched-tier retrieval plus honest per-gap fetch citations is accepted even
        though the run's fetch count (1) is below the global 2-fetch floor."""

        async def fetch(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="body", method="rendered")

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        # Two gaps -> the per-gap external floor needs >=2 external calls: one
        # successful fetch (fetched tier) plus one search (snippet tier). That
        # leaves fetches_reads=1, below the global floor of 2, so acceptance rides
        # on the per-gap fetch citations backed by the single real retrieval.
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(
                    tool_calls=[
                        _tool_call("f1", "fetch", {"url": "https://a.example"}),
                        _tool_call("s1", "search_web", {"query": "x"}),
                    ]
                ),
                _response(
                    tool_calls=[
                        _tool_call(
                            "c1",
                            "conclude",
                            {
                                "gap_accounting": _accounting("g1", actions="fetched the primary source PDF")
                                + _accounting("g2", actions="read the appendix table")
                            },
                        )
                    ]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("fetch", fetch), _tool_spec("search_web", search_web)],
            _config(max_steps=6),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_locally_read_documents_clear_the_fetch_floor(self) -> None:
        """A document read from bytes we hold counts toward the floor like any other retrieval.

        The floor counts fetched-tier URLs, so the local rung's two methods have to be in
        ``provenance._METHOD_TO_TIER`` or a run that read two primary-source PDFs without
        spending a model call would be told it never fetched anything and be sent back out.
        """

        async def fetch(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="the report's own text", method="pdf_local")

        async def read_document(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="[p.4] the passage asked for", method="digest_local")

        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(
                    tool_calls=[
                        _tool_call("f1", "fetch", {"url": "https://a.example/report.pdf"}),
                        _tool_call("r1", "read_document", {"url": "https://b.example/filing.pdf", "ask": "revenue?"}),
                    ]
                ),
                _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1", "g2")})]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("fetch", fetch), _tool_spec("read_document", read_document)],
            _config(max_steps=6, max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_must_conclude_bypasses_gate(self) -> None:
        """Budget exhaustion (_must_conclude) overrides the gate: a conclude with
        no gap_accounting at all is accepted, and the rejection counter stays 0."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        # max_tool_calls=2: plan (1) + one search (2) exhausts the budget, so the
        # conclude turn is in must-conclude mode and the gate is bypassed.
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                _response(tool_calls=[_tool_call("c1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("search_web", search_web)], _config(max_tool_calls=2), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_rejection_cap_accepts_after_two_rejections(self) -> None:
        """After max_conclude_gate_rejections (2) blocked concludes, the third is
        accepted unconditionally so a pathological driver can't loop forever."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        # A single gap, one snippet-only search, and three bare concludes that all
        # fail the fetch floor. The first two are rejected; the third is accepted
        # by the cap.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                _response(
                    tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
                _response(
                    tool_calls=[_tool_call("c2", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
                _response(
                    tool_calls=[_tool_call("c3", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("search_web", search_web)], _config(max_steps=8), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 2
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_plan_skipped_bypasses_gate(self) -> None:
        """When the plan-nudge cap forced a soft-continue (plan_skipped), there is
        no plan to enforce, so the conclude gate is bypassed."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        # Two plan-less external turns hit the nudge cap; turn 3 runs un-gated;
        # turn 4 concludes with no accounting -> accepted (plan_skipped bypass).
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("c1", "search_web", {"query": "q1"})]),
                _response(tool_calls=[_tool_call("c2", "search_web", {"query": "q2"})]),
                _response(tool_calls=[_tool_call("c3", "search_web", {"query": "q3"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("search_web", search_web)], _config(max_steps=8), llm_call=fake_llm
        )

        assert result.telemetry.plan_skipped is True
        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_conclude_with_no_plan_is_rejected(self) -> None:
        """F3b: conclude is not plan-gated, so a driver could conclude on turn 1
        with no plan set (plan_skipped still False). The gate now rejects that —
        planning can't be skipped by never calling set_research_plan."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        # Turn 1 concludes with no plan -> rejected; turn 2's bare conclude is
        # accepted by the rejection cap (=1), proving the loop can't wedge.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("c1", "conclude")]),
                _response(tool_calls=[_tool_call("c2", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=6, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 1
        first_conclude = _tool_messages(result)[0]
        assert "Conclude rejected" in first_conclude["content"]
        assert "no research plan was set" in first_conclude["content"]
        # The cap accepted the second conclude — no wedge.
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_valid_plan_after_nudge_cap_re_arms_the_gate(self) -> None:
        """F3c: plan_skipped was checked before the plan and never cleared, so a
        valid plan set after the plan-nudge cap fired stayed permanently exempt.
        Now a real plan re-arms the gate even once plan_skipped is True."""

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        # Turns 1-2: plan-less external calls hit the nudge cap (plan_skipped=True).
        # Turn 3: a real plan registers. Turn 4: an early conclude with no
        # gap_accounting must now be rejected (the gate is re-armed despite
        # plan_skipped). Turn 5: the cap (=1) accepts, so no wedge.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("c1", "search_web", {"query": "q1"})]),
                _response(tool_calls=[_tool_call("c2", "search_web", {"query": "q2"})]),
                _response(tool_calls=[_plan_call("p1", gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("d1", "conclude")]),
                _response(tool_calls=[_tool_call("d2", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=8, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.plan_skipped is True
        assert result.telemetry.plan_gaps == 1
        # The gate re-armed: the first post-plan conclude was rejected on the
        # missing gap accounting, not waved through by the stale plan_skipped.
        assert result.telemetry.conclude_gate_rejections == 1
        rejected_conclude = _tool_messages(result)[-2]
        assert "Conclude rejected" in rejected_conclude["content"]
        assert "missing entries for gap(s): g1" in rejected_conclude["content"]
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_conclude_gate_rejections_surface_in_marker(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")

        async def search_web(**_: Any) -> ToolOutcome:
            return ToolOutcome(content_markdown="snippet", method="search")

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                # One rejected conclude (fetch floor unmet), then the cap accepts.
                _response(
                    tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
                _response(
                    tool_calls=[_tool_call("c2", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
            ]
        )

        await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=8, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        marker = next(r.getMessage() for r in caplog.records if "GAP_FILL_V2:" in r.getMessage())
        assert "conclude_gate_rejections=1" in marker
