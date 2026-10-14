from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from metaculus_bot.research.agentic import fetch_outcomes
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.agentic.dispatch import _materialize_delivered_evidence
from metaculus_bot.research.agentic.loop_state import _LoopState, _ToolExecutionResult
from metaculus_bot.research.agentic.types import EvidencePassage, LoopConfig
from metaculus_bot.research.known_api import adapters, backends
from metaculus_bot.research.known_api.result import KnownApiResult
from metaculus_bot.research.market_retrieval.types import MarketChild, MarketMatch
from metaculus_bot.research.source_evidence import SourceExcerpt


@pytest.mark.asyncio
async def test_search_web_exposes_only_provider_highlights_with_explicit_result_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_url = "https://agency.example/report"
    monkeypatch.setenv("EXA_API_KEY", "offline-test-key")
    monkeypatch.setattr(
        agentic_tools,
        "_call_exa_search",
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    title="Agency report",
                    url=result_url,
                    highlights=["The published estimate was 5.2% in 2025."],
                ),
                SimpleNamespace(
                    title="Unbound result",
                    url="",
                    highlights=["See https://agency.example/other for the 7.1% figure."],
                ),
            ]
        ),
    )

    outcome = await agentic_tools.search_web("published estimate")

    assert outcome.evidence_passages == [
        EvidencePassage(
            source_url=result_url,
            text="The published estimate was 5.2% in 2025.",
            retrieval_method="search",
            content_origin="provider_summary",
        )
    ]


@pytest.mark.asyncio
async def test_search_news_marks_provider_summary_and_uses_article_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article_url = "https://news.example/story"
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "offline-test-id")
    monkeypatch.setenv("ASKNEWS_SECRET", "offline-test-secret")
    monkeypatch.setattr(
        agentic_tools,
        "_call_asknews_search",
        AsyncMock(
            return_value=[
                {
                    "eng_title": "Published estimate",
                    "article_url": article_url,
                    "summary": "The estimate was 5.2% in 2025.",
                }
            ]
        ),
    )

    outcome = await agentic_tools.search_news("published estimate")

    assert outcome.evidence_passages == [
        EvidencePassage(
            source_url=article_url,
            text="The estimate was 5.2% in 2025.",
            retrieval_method="news",
            content_origin="provider_summary",
        )
    ]


@pytest.mark.asyncio
async def test_fetch_binds_exact_delivered_paragraphs_to_the_final_url(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_url = "https://agency.example/redirect"
    final_url = "https://agency.example/report"
    first_paragraph = "The annual estimate was 5.2% in 2025."
    second_paragraph = "The report used a sample of 4,100 households."
    text = f"{first_paragraph}\n\n{second_paragraph}"
    monkeypatch.setattr(
        agentic_tools,
        "_fetch_via_ladder",
        AsyncMock(
            return_value=fetch_outcomes.PlainFetchResult(
                status="ok",
                method="rendered",
                text=text,
                links=[],
                url=final_url,
            )
        ),
    )

    outcome = await agentic_tools.fetch(requested_url)

    assert outcome.evidence_passages == [
        EvidencePassage(
            source_url=final_url,
            text=first_paragraph,
            retrieval_method="rendered",
            content_origin="source_text",
        ),
        EvidencePassage(
            source_url=final_url,
            text=second_paragraph,
            retrieval_method="rendered",
            content_origin="source_text",
        ),
    ]
    assert all(passage.text in outcome.content_markdown for passage in outcome.evidence_passages)


def test_fetch_marks_paragraph_cut_by_start_char_as_clipped() -> None:
    text = "Complete paragraph.\n\nA paragraph continued after the starting offset."

    outcome = agentic_tools._render_fetch_outcome(
        "https://agency.example/report",
        text,
        [],
        method="plain",
        start_char=5,
    )

    assert outcome.evidence_passages[0].text == "ete paragraph."
    assert outcome.evidence_passages[0].clipped is True
    assert outcome.evidence_passages[1].clipped is False


def test_fetch_marks_tail_window_cut_as_clipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agentic_tools, "tool_content_body_budget", lambda *_args, **_kwargs: 250)
    first_paragraph = "Complete paragraph."
    second_paragraph = "A partial paragraph " + "x" * 300
    text = f"{first_paragraph}\n\n{second_paragraph}"

    outcome = agentic_tools._render_fetch_outcome(
        "https://agency.example/report",
        text,
        [],
        method="plain",
        start_char=0,
    )

    assert outcome.evidence_passages[0].text == first_paragraph
    assert outcome.evidence_passages[0].clipped is False
    assert outcome.evidence_passages[1].clipped is True
    assert outcome.evidence_passages[1].text in outcome.content_markdown


def test_fetch_does_not_treat_presentation_leads_as_source_text() -> None:
    url = "https://agency.example/report"
    lead = "Source: https://agency.example/lead"
    excerpt = SourceExcerpt(
        source_url=url,
        text="The agency report gives a 5.2% estimate.",
        retrieval_method="rendered",
    )
    outcome = agentic_tools._render_fetch_outcome(
        url,
        f"{lead}\n\nUnreadable embed disclosure.\n\n{excerpt.text}",
        ["https://agency.example/lead"],
        method="rendered",
        start_char=0,
        evidence_passages=[excerpt],
    )

    assert [passage.text for passage in outcome.evidence_passages] == [excerpt.text]
    assert lead in outcome.content_markdown
    assert "Unreadable embed disclosure." in outcome.content_markdown


def test_fetch_with_empty_structured_evidence_does_not_fall_back_to_markdown_scanning() -> None:
    outcome = agentic_tools._render_fetch_outcome(
        "https://agency.example/report",
        "Source: https://agency.example/lead\n\nThe page mentions 5.2%.",
        ["https://agency.example/lead"],
        method="rendered",
        start_char=0,
        evidence_passages=[],
    )

    assert outcome.evidence_passages == []


def test_local_document_uses_raw_captured_html_text_without_presentation_leads() -> None:
    result = fetch_outcomes.PlainFetchResult(
        status="ok",
        method="rendered",
        text="Source: https://agency.example/lead\n\nUnreadable embed disclosure.\n\nArticle passage.",
        links=["https://agency.example/lead"],
        url="https://agency.example/report",
        source_text="Article passage.",
    )

    held = agentic_tools._held_from_result(result.url, result)

    assert held.text == "Article passage."


def test_clipped_window_passage_stays_marked_in_materialized_tool_result() -> None:
    outcome = agentic_tools._render_fetch_outcome(
        "https://agency.example/report",
        "Complete paragraph.\n\nA paragraph continued after the starting offset.",
        [],
        method="plain",
        start_char=5,
    )
    state = _LoopState(messages=[], started_at_s=0.0, deadline_at_s=100.0)
    result = _ToolExecutionResult(
        tool_call_id="call_fetch",
        tool_name="fetch",
        content="",
        outcome=outcome,
    )

    delivered_content = _materialize_delivered_evidence(result, state, LoopConfig(model="test-model"))

    assert "clipped: true" in delivered_content
    assert state.delivered_evidence["ev_1"].clipped is True


def test_local_digest_adapter_binds_candidate_passages_to_url_and_fetched_method() -> None:
    text = "The report gives a 5.2% estimate for 2025."
    excerpt = SourceExcerpt(source_url="", text=text, retrieval_method="bm25_candidate")

    passages = agentic_tools._digest_evidence_passages(
        [excerpt],
        block=f"### Local digest\n\n{text}",
        source_url="https://agency.example/report",
    )

    assert passages == [
        EvidencePassage(
            source_url="https://agency.example/report",
            text=text,
            retrieval_method="digest_local",
            content_origin="source_text",
        )
    ]


@pytest.mark.asyncio
async def test_paid_document_read_is_labeled_as_a_generated_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://agency.example/scanned-report.pdf"
    answer = "The report gives a 5.2% estimate for 2025."
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test-key")
    monkeypatch.setattr(
        agentic_tools, "_acquire_local_document", AsyncMock(return_value=agentic_tools.local_document.HeldDocument())
    )
    monkeypatch.setattr(agentic_tools, "_url_context_robots_skip", AsyncMock(return_value=False))
    monkeypatch.setattr(agentic_tools, "_take_paid_document_read_attempt", lambda _url, _ctx: True)
    monkeypatch.setattr(
        agentic_tools, "_run_document_read_sync", lambda _url, _ask: (answer, 1, ["URL_RETRIEVAL_STATUS_SUCCESS"])
    )

    outcome = await agentic_tools.read_document(url, "What estimate is reported?")

    assert outcome.evidence_passages == [
        EvidencePassage(
            source_url=url,
            text=answer,
            retrieval_method="document",
            content_origin="generated_answer",
        )
    ]


def test_known_api_adapter_preserves_structured_source_associations() -> None:
    first_url = "https://fred.stlouisfed.org/series/DGS30"
    second_url = "https://finance.yahoo.com/quote/%5EGSPC/history/"
    result = SimpleNamespace(
        status="ok",
        content_markdown="See https://wrong.example.\n5.2%\n4,100 households.",
        source_url=first_url,
        links=[first_url, second_url],
        evidence_passages=[
            SimpleNamespace(
                source_url=first_url,
                text="5.2%",
                retrieval_method="known_api",
                content_origin="source_text",
                clipped=True,
            ),
            SimpleNamespace(
                source_url=second_url,
                text="4,100 households.",
                retrieval_method="known_api",
                content_origin="provider_summary",
                clipped=False,
            ),
        ],
    )

    outcome = adapters.to_tool_outcome(result)  # type: ignore[arg-type]

    assert outcome.evidence_passages == [
        EvidencePassage(
            source_url=first_url,
            text="5.2%",
            retrieval_method="known_api",
            content_origin="source_text",
            clipped=True,
        ),
        EvidencePassage(
            source_url=second_url,
            text="4,100 households.",
            retrieval_method="known_api",
            content_origin="provider_summary",
        ),
    ]


def test_known_api_adapter_does_not_infer_a_source_from_markdown_or_links() -> None:
    result = KnownApiResult(
        status="ok",
        content_markdown="A result mentions https://agency.example/report and gives 5.2%.",
        source_url="",
        links=["https://agency.example/report"],
    )

    outcome = adapters.to_tool_outcome(result)

    assert outcome.evidence_passages == []


def test_known_api_fetch_adapter_preserves_structured_passages() -> None:
    passage = SourceExcerpt(
        source_url="https://fred.stlouisfed.org/series/DGS30",
        text="  - 2025-01-01: 5.2%",
        retrieval_method="known_api",
        content_origin="source_text",
    )
    result = KnownApiResult(
        status="ok",
        content_markdown=f"### DGS30\n{passage.text}",
        source_url=passage.source_url,
        evidence_passages=[passage],
    )

    fetch_result = adapters.to_fetch_result(result, url=passage.source_url)

    assert fetch_result.evidence_passages == [passage]


def test_fred_backend_returns_observation_rows_as_source_bound_passages() -> None:
    series = pd.Series([5.2, 5.4], index=pd.to_datetime(["2025-01-01", "2025-02-01"]))

    class FredMetadata:
        def get_series_info(self, _series_id: str) -> pd.Series:
            return pd.Series({"title": "Thirty-year Treasury rate", "units": "Percent"})

    result = backends._render_fred(
        "DGS30",
        series,
        "https://fred.stlouisfed.org/series/DGS30",
        fred=FredMetadata(),
        start=None,
        end=None,
        first_release=False,
    )

    assert len(result.evidence_passages) == 2
    assert [passage.source_url for passage in result.evidence_passages] == [
        "https://fred.stlouisfed.org/series/DGS30",
        "https://fred.stlouisfed.org/series/DGS30",
    ]
    assert all(passage.content_origin == "structured_data" for passage in result.evidence_passages)
    assert all(passage.text in result.content_markdown for passage in result.evidence_passages)
    assert all(
        "DGS30 (Thirty-year Treasury rate); units: Percent; 2025-" in passage.text
        for passage in result.evidence_passages
    )


def test_fred_search_returns_series_catalogue_rows_with_canonical_series_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = pd.DataFrame(
        {"title": ["Thirty-year yield"], "frequency": ["Daily"], "units": ["Percent"]},
        index=["DGS30"],
    )

    class FakeFred:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "offline-test-key"

        def search(self, _query: str, *, limit: int) -> pd.DataFrame:
            assert limit == backends.FRED_SEARCH_LIMIT
            return hits

    monkeypatch.setattr(backends.fred_rendering, "Fred", FakeFred)

    result = backends._fred_search_sync("treasury yield", "offline-test-key")

    assert len(result.evidence_passages) == 1
    passage = result.evidence_passages[0]
    assert passage.source_url == "https://fred.stlouisfed.org/series/DGS30"
    assert passage.text == "- DGS30: Thirty-year yield (Daily; Percent)"
    assert passage.content_origin == "structured_data"
    assert passage.text in result.content_markdown


def test_yahoo_backend_returns_observation_rows_as_source_bound_passages(monkeypatch: pytest.MonkeyPatch) -> None:
    series = pd.Series([100.0, 102.0], index=pd.to_datetime(["2025-01-01", "2025-02-01"]))
    monkeypatch.setattr(backends.ts_fetch, "fetch_series", lambda *_args, **_kwargs: series)

    result = backends._yahoo_history_sync("^GSPC", start=None, end=None, column="Close")

    assert len(result.evidence_passages) == 2
    assert all(
        passage.source_url == "https://finance.yahoo.com/quote/^GSPC/history/" for passage in result.evidence_passages
    )
    assert all(passage.content_origin == "structured_data" for passage in result.evidence_passages)
    assert all(passage.text in result.content_markdown for passage in result.evidence_passages)
    assert all(
        "^GSPC (Yahoo Finance Close); units: not supplied" in passage.text for passage in result.evidence_passages
    )


def test_market_backend_associates_visible_child_rows_with_the_parent_market_url() -> None:
    market_url = "https://polymarket.com/event/first"
    match = _market_match("First market", market_url)
    match.children = (MarketChild(title="Outcome A", implied_prob_yes=0.4),)

    result = backends._render_matches([match], "polymarket")

    assert len(result.evidence_passages) == 2
    assert [passage.source_url for passage in result.evidence_passages] == [market_url, market_url]
    assert all(passage.text in result.content_markdown for passage in result.evidence_passages)
    assert "Outcome A" in result.evidence_passages[1].text


@pytest.mark.asyncio
async def test_edgar_document_returns_bounded_source_text_passages(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://www.sec.gov/Archives/edgar/data/123/filing.txt"
    document_text = "First filed paragraph.\n\nSecond filed paragraph."

    async def fake_filing_document(_session: object, requested_url: str) -> SimpleNamespace:
        assert requested_url == url
        return SimpleNamespace(url=url, content_type="text/plain", body=document_text.encode())

    monkeypatch.setattr(backends.sec_edgar, "filing_document", fake_filing_document)

    result = await backends._edgar_document(None, url)

    assert [passage.text for passage in result.evidence_passages] == [
        "First filed paragraph.",
        "Second filed paragraph.",
    ]
    assert all(passage.source_url == url for passage in result.evidence_passages)
    assert all(passage.content_origin == "source_text" for passage in result.evidence_passages)
    assert all(passage.text in result.content_markdown for passage in result.evidence_passages)


@pytest.mark.asyncio
async def test_edgar_submission_rows_bind_to_the_fetched_submissions_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filing_url = "https://www.sec.gov/Archives/edgar/data/123/filing.htm"
    filing = SimpleNamespace(
        form="10-K",
        filing_date="2025-02-15",
        report_date="2024-12-31",
        primary_document_url=filing_url,
    )

    async def fake_company_submissions(_session: object, _cik: str) -> SimpleNamespace:
        return SimpleNamespace(cik=123, name="Example Corp", filings=[filing])

    monkeypatch.setattr(backends.sec_edgar, "company_submissions", fake_company_submissions)

    result = await backends._edgar_submissions(None, "0000123")

    assert len(result.evidence_passages) == 1
    passage = result.evidence_passages[0]
    submissions_url = f"{backends.sec_edgar.DATA_BASE_URL}/submissions/CIK0000000123.json"
    assert passage.source_url == submissions_url
    assert passage.source_url != filing_url
    assert passage.text == f"- 10-K filed 2025-02-15 (period 2024-12-31): {filing_url}"
    assert passage.content_origin == "structured_data"
    assert passage.text in result.content_markdown


def test_market_backend_keeps_each_row_bound_to_its_own_market_url() -> None:
    first_url = "https://polymarket.com/event/first"
    second_url = "https://polymarket.com/event/second"
    matches = [
        _market_match("First market", first_url),
        _market_match("Second market", second_url),
    ]

    result = backends._render_matches(matches, "polymarket")

    assert len(result.evidence_passages) == 2
    assert [passage.source_url for passage in result.evidence_passages] == [first_url, second_url]
    assert all(passage.content_origin == "structured_data" for passage in result.evidence_passages)
    assert all(passage.text in result.content_markdown for passage in result.evidence_passages)
    assert first_url not in result.evidence_passages[1].text
    assert second_url not in result.evidence_passages[0].text


def _market_match(title: str, url: str) -> MarketMatch:
    return MarketMatch(
        platform="polymarket",
        market_title=title,
        market_url=url,
        implied_prob_yes=0.52,
        bid=None,
        ask=None,
        spread=None,
        volume_24h=None,
        close_time=None,
        is_resolved=False,
        match_confidence=1.0,
        raw_rules="Resolves yes when the event occurs.",
    )
