"""Shared page-digest behavior for fresh, cached and held HTML reads."""

from __future__ import annotations

import re
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from metaculus_bot.research.agentic import local_document
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.document_text import PdfText
from metaculus_bot.research.fetch_ladder import classify, run_cache
from metaculus_bot.research.fetch_ladder.context import LadderContext
from metaculus_bot.research.fetch_ladder.digest import LadderDigest
from metaculus_bot.research.fetch_ladder.ladder import fetch_url
from metaculus_bot.research.fetch_ladder.policy import (
    GAP_FILL_DOCUMENT_POLICY,
    GAP_FILL_FETCH_POLICY,
    RESOLUTION_SOURCE_POLICY,
)
from metaculus_bot.research.fetch_ladder.verdict import PageExtraction
from metaculus_bot.research.fetch_markers import fetch_marker_line
from metaculus_bot.research.resolution_fetch_result import FetchResult
from metaculus_bot.research.wayback import WaybackSnapshot
from tests.resolution_source_fakes import FakeResponse, FakeSession

_URL = "https://tracker.example.com/digest"


def _long_page() -> str:
    return "Opening context about the tracker.\n\n" + ("The tracker reports 917 admissions this week. " * 220)


@pytest.mark.asyncio
async def test_long_html_digest_receives_full_text_and_reapplies_leads_and_cap_on_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The digest sees the canonical extraction on both fresh and cached presentation."""
    full_text = _long_page()
    calls: list[tuple[str, str, float]] = []

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        calls.append((text, query, budget_seconds))
        return LadderDigest(
            passages=["The tracker reports 917 admissions this week."],
            passages_returned=4,
            passages_grounded=3,
            fallback_used=False,
            method="llm_extractive",
        )

    monkeypatch.setattr(classify, "_extract_page_text", lambda *_args, **_kwargs: PageExtraction(text=full_text))
    monkeypatch.setattr(classify, "render_inline_chart_data", lambda *_args, **_kwargs: "Chart data lead")
    monkeypatch.setattr(classify, "unreadable_data_embed_providers", lambda *_args, **_kwargs: ["Infogram"])
    session = FakeSession({_URL: FakeResponse(200, body=b"<html><body>page</body></html>")})
    policy = replace(RESOLUTION_SOURCE_POLICY, digest=digest)

    fresh = await fetch_url(
        _URL,
        policy=policy,
        ctx=LadderContext(query="weekly admissions", session=session, host_sems={}),
    )
    cached = await fetch_url(
        _URL,
        policy=policy,
        ctx=LadderContext(query="weekly admissions", session=session, host_sems={}),
    )

    assert len(calls) == 2
    assert all(text == full_text for text, _query, _budget in calls)
    assert [query for _text, query, _budget in calls] == ["weekly admissions", "weekly admissions"]
    assert all(0.0 < budget <= policy.total_wall_s for _text, _query, budget in calls)
    for result in (fresh, cached):
        assert result.status == "success"
        assert result.text.startswith("Chart data lead\n\n")
        assert "Infogram embed(s)" in result.text
        assert "The tracker reports 917 admissions this week." in result.text
        assert len(result.text) <= policy.per_url_max_chars  # type: ignore[arg-type]
        assert result.passages_returned == 4
        assert result.passages_grounded == 3
        assert result.fallback_used is False
    assert cached.cache_hit is True
    assert session.requested == [_URL]


@pytest.mark.asyncio
async def test_fresh_html_digest_receives_wall_time_remaining_after_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_text = _long_page()
    budgets: list[float] = []

    def slow_extract(*_args: object, **_kwargs: object) -> PageExtraction:
        time.sleep(0.01)
        return PageExtraction(text=full_text)

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        del text, query
        budgets.append(budget_seconds)
        return LadderDigest(
            passages=["The tracker reports 917 admissions this week."],
            passages_returned=1,
            passages_grounded=1,
            fallback_used=False,
            method="llm_extractive",
        )

    monkeypatch.setattr(classify, "_extract_page_text", slow_extract)
    monkeypatch.setattr(classify, "render_inline_chart_data", lambda *_args: "")
    monkeypatch.setattr(classify, "unreadable_data_embed_providers", lambda *_args: [])
    policy = replace(RESOLUTION_SOURCE_POLICY, digest=digest)
    classified = await classify._classify_html_body(
        b"<html><body>page</body></html>",
        _URL,
        "text/html",
        http_status=200,
        query="weekly admissions",
        remaining_wall_s=0.2,
        pol=policy,
    )

    assert classified.result.status == "success"
    assert len(budgets) == 1
    assert 0.0 < budgets[0] < 0.2


@pytest.mark.asyncio
async def test_html_digest_is_skipped_when_presentation_has_used_the_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = AsyncMock(side_effect=AssertionError("an expired HTML budget must not call the digest seat"))
    artifact = run_cache.HtmlRead(
        url=_URL,
        http_status=200,
        content_type="text/html",
        extraction=PageExtraction(text=_long_page()),
        chart_block="",
        datawrapper_charts=(),
        unreadable_embeds=(),
        links=(),
        routing_body=b"<html>",
    )
    result = await artifact.present_html(
        replace(RESOLUTION_SOURCE_POLICY, digest=digest),
        query="weekly admissions",
        route="direct",
        now=datetime.now(UTC),
        budget_seconds=0.0,
    )

    assert result is not None
    assert result.status == "success"
    digest.assert_not_awaited()


@pytest.mark.asyncio
async def test_short_html_does_not_invoke_digest_and_has_no_digest_marker_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        del query, budget_seconds
        calls.append(text)
        raise AssertionError("short HTML must stay on the ordinary presentation path")

    short_text = "Official tracker report.\n\n" + ("The official tracker reports 917 admissions this week. " * 12)
    monkeypatch.setattr(classify, "_extract_page_text", lambda *_args, **_kwargs: PageExtraction(text=short_text))
    session = FakeSession({_URL: FakeResponse(200, body=b"<html><body>page</body></html>")})
    result = await fetch_url(
        _URL,
        policy=replace(RESOLUTION_SOURCE_POLICY, digest=digest),
        ctx=LadderContext(query="admissions", session=session, host_sems={}),
    )

    assert result.status == "success"
    assert result.text == short_text
    assert calls == []
    assert result.passages_returned is None
    assert result.passages_grounded is None
    assert result.fallback_used is None


@pytest.mark.asyncio
async def test_empty_html_digest_selection_keeps_the_capped_page() -> None:
    """An empty digest selection must not replace a readable page with no-match prose."""
    cap = 300

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        del text, query, budget_seconds
        return LadderDigest(
            passages=[],
            passages_returned=0,
            passages_grounded=0,
            fallback_used=False,
            method="bm25",
        )

    artifact = run_cache.HtmlRead(
        url=_URL,
        http_status=200,
        content_type="text/html",
        extraction=PageExtraction(text=_long_page()),
        chart_block="",
        datawrapper_charts=(),
        unreadable_embeds=(),
        links=(),
        routing_body=b"<html>",
    )
    result = await artifact.present_html(
        replace(RESOLUTION_SOURCE_POLICY, digest=digest, per_url_max_chars=cap),
        query="",
        route="direct",
        now=datetime.now(UTC),
        budget_seconds=RESOLUTION_SOURCE_POLICY.total_wall_s,
    )

    assert result is not None
    assert result.status == "success"
    assert "The tracker reports 917 admissions this week." in result.text
    assert "No passage in this document matched the query." not in result.text
    assert len(result.text) <= cap
    assert result.passages_returned is None
    assert result.passages_grounded is None
    assert result.fallback_used is None


@pytest.mark.asyncio
async def test_wayback_html_cache_hit_uses_the_same_digest_seat_and_preserves_archive_lead() -> None:
    full_text = _long_page()
    calls: list[tuple[str, str, float]] = []

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        calls.append((text, query, budget_seconds))
        return LadderDigest(
            passages=["The tracker reports 917 admissions this week."],
            passages_returned=1,
            passages_grounded=1,
            fallback_used=False,
            method="llm_extractive",
        )

    now = datetime(2026, 9, 10, tzinfo=UTC)
    artifact = run_cache.HtmlRead(
        url=_URL,
        http_status=200,
        content_type="text/html",
        extraction=PageExtraction(text=full_text),
        chart_block="",
        datawrapper_charts=(),
        unreadable_embeds=(),
        links=(),
        routing_body=b"<html>",
    )
    wayback = run_cache.WaybackRead(
        url=_URL,
        snapshot=WaybackSnapshot(captured_at=now - timedelta(days=2), inner_url=_URL),
        artifact=artifact,
        live_status="blocked",
        live_http_status=403,
        live_content_type="text/html",
        live_failure_class="http_403",
        live_exc=None,
        live_server="example",
    )
    policy = replace(RESOLUTION_SOURCE_POLICY, digest=digest)
    run_cache.clear()
    try:
        run_cache.put(_URL, wayback, route="wayback")
        result = await run_cache.get(
            _URL,
            policy=policy,
            query="weekly admissions",
            now=now,
            budget_s=policy.total_wall_s,
        )
    finally:
        run_cache.clear()

    assert result is not None
    assert result.cache_hit is True
    assert result.route == "wayback"
    assert "Archived copy" in result.text
    assert "The tracker reports 917 admissions this week." in result.text
    assert len(calls) == 1
    assert calls[0][0] == full_text
    assert calls[0][1] == "weekly admissions"


@pytest.mark.asyncio
async def test_loop_fetch_keeps_long_html_paginated_and_does_not_digest_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_text = _long_page()

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        del text, query, budget_seconds
        raise AssertionError("ordinary fetch pagination must not invoke the page digest")

    monkeypatch.setattr(classify, "_extract_page_text", lambda *_args, **_kwargs: PageExtraction(text=full_text))
    monkeypatch.setattr(agentic_tools, "GAP_FILL_FETCH_POLICY", replace(GAP_FILL_FETCH_POLICY, digest=digest))
    session = FakeSession({_URL: FakeResponse(200, body=b"<html><body>page</body></html>")})
    monkeypatch.setattr(
        agentic_tools,
        "_per_call_ctx",
        lambda _question_ctx, *, query: LadderContext(query=query, session=session, host_sems={}),
    )

    first = await agentic_tools.fetch(_URL, question_topic="weekly admissions")
    continuation = re.search(
        r"\n\[truncated at (\d+) of \d+ chars — call again with start_char=(\d+)\]$", first.content_markdown
    )
    assert continuation is not None
    next_offset = int(continuation.group(2))
    assert next_offset == int(continuation.group(1))
    second = await agentic_tools.fetch(_URL, start_char=next_offset, question_topic="weekly admissions")

    assert first.method == "plain"
    assert first.truncated is True
    assert next_offset < 8_000
    assert first.content_markdown[: continuation.start()] + second.content_markdown == full_text.strip()
    assert "917 admissions" in second.content_markdown
    assert second.method == "cache"
    assert session.requested == [_URL]


@pytest.mark.asyncio
async def test_held_flat_html_uses_digest_seat_with_remaining_read_document_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held_text = "Opening context.\n\n" + ("The tracker reports 917 admissions this week. " * 100)
    calls: list[tuple[str, str, float]] = []

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        calls.append((text, query, budget_seconds))
        return LadderDigest(
            passages=["The tracker reports 917 admissions this week."],
            passages_returned=2,
            passages_grounded=2,
            fallback_used=False,
            method="llm_extractive",
        )

    monkeypatch.setattr(
        agentic_tools,
        "GAP_FILL_DOCUMENT_POLICY",
        replace(GAP_FILL_DOCUMENT_POLICY, digest=digest),
    )
    monkeypatch.setattr(
        agentic_tools,
        "_acquire_local_document",
        AsyncMock(return_value=local_document.HeldDocument(text=held_text)),
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    outcome = await agentic_tools.read_document(_URL, "weekly admissions")

    assert outcome.method == "digest_local"
    assert "The tracker reports 917 admissions this week." in outcome.content_markdown
    assert len(calls) == 1
    text, query, budget = calls[0]
    assert text == held_text
    assert query == "weekly admissions"
    assert 0.0 < budget <= agentic_tools._READ_DOCUMENT_TOTAL_BUDGET_S


@pytest.mark.asyncio
async def test_subfloor_flat_no_match_still_falls_through_to_paid_reader_when_opening_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = local_document.HeldDocument(text="Opening page context with no matching fact.")
    digest = AsyncMock(
        return_value=LadderDigest(
            passages=["Opening page context with no matching fact."],
            passages_returned=0,
            passages_grounded=0,
            fallback_used=True,
            method="digest_local",
        )
    )
    monkeypatch.setattr(agentic_tools, "GAP_FILL_DOCUMENT_POLICY", replace(GAP_FILL_DOCUMENT_POLICY, digest=digest))
    monkeypatch.setattr(agentic_tools, "_acquire_local_document", AsyncMock(return_value=held))
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(agentic_tools, "_url_context_robots_skip", AsyncMock(return_value=False))
    monkeypatch.setattr(
        agentic_tools,
        "_run_document_read_sync",
        MagicMock(return_value=("Paid reader answer.", 1, ["SUCCESS"])),
    )

    outcome = await agentic_tools.read_document(_URL, "weekly admissions")

    assert outcome.method == "document"
    assert outcome.content_markdown == "Paid reader answer."
    digest.assert_not_awaited()


@pytest.mark.asyncio
async def test_subfloor_flat_matching_passage_stays_local_even_when_digest_grounding_count_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = local_document.HeldDocument(text="The tracker reports 917 admissions this week.")
    digest = AsyncMock(
        return_value=LadderDigest(
            passages=["The tracker reports 917 admissions this week."],
            passages_returned=0,
            passages_grounded=0,
            fallback_used=True,
            method="digest_local",
        )
    )
    monkeypatch.setattr(agentic_tools, "GAP_FILL_DOCUMENT_POLICY", replace(GAP_FILL_DOCUMENT_POLICY, digest=digest))
    monkeypatch.setattr(agentic_tools, "_acquire_local_document", AsyncMock(return_value=held))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    outcome = await agentic_tools.read_document(_URL, "weekly admissions")

    assert outcome.method == "digest_local"
    assert "Document: https://tracker.example.com/digest" in outcome.content_markdown
    assert "917 admissions" in outcome.content_markdown
    digest.assert_awaited_once()


@pytest.mark.asyncio
async def test_held_pdf_keeps_page_aware_digest_and_never_uses_flat_digest_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = PdfText(
        page_count=2,
        pages_read=2,
        pages=("The first page reports unemployment at 4.1 percent.", "The second page reports revisions."),
        truncated_by="",
        outline=(),
    )
    digest = AsyncMock(side_effect=AssertionError("PDFs must use the page-aware local digest"))
    monkeypatch.setattr(agentic_tools, "GAP_FILL_DOCUMENT_POLICY", replace(GAP_FILL_DOCUMENT_POLICY, digest=digest))
    monkeypatch.setattr(
        agentic_tools,
        "_acquire_local_document",
        AsyncMock(return_value=local_document.held_pdf(pdf)),
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    outcome = await agentic_tools.read_document(_URL, "unemployment revisions")

    assert outcome.method == "digest_local"
    assert "[p.1]" in outcome.content_markdown or "[p.2]" in outcome.content_markdown
    digest.assert_not_awaited()


def test_digest_marker_fields_precede_the_caller_tail() -> None:
    line = fetch_marker_line(
        FetchResult(
            url=_URL,
            status="success",
            text="passage",
            http_status=200,
            content_type="text/html",
            passages_returned=4,
            passages_grounded=3,
            fallback_used=False,
        ),
        qid=123,
        caller="resolution_source",
    )

    assert "passages_returned=4 passages_grounded=3 fallback_used=False" in line
    assert line.endswith("caller=resolution_source")
