"""Public fetch/read_document behavior over parsed local sources."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from metaculus_bot.research.agentic import ladder_adapter
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.agentic.dispatch import _format_tool_content
from metaculus_bot.research.agentic.fetch_outcomes import PlainFetchResult
from metaculus_bot.research.agentic.types import ToolOutcome
from metaculus_bot.research.document_text import DocumentDigest
from metaculus_bot.research.fetch_ladder.context import LadderContext, LocalReadReceipt
from metaculus_bot.research.source_documents import ParsedSource, SourceMember, SourceSection

_URL = "https://example.com/results.zip"


def _archive() -> ParsedSource:
    return ParsedSource(
        kind="archive",
        sections=(
            SourceSection("notes.txt", None, "Background only."),
            SourceSection(
                "results.xlsx", "Summary", "Sheet: Summary\nrow 1: A1=month | B1=value\nrow 2: A2=May | B2=42"
            ),
            SourceSection("results.xlsx", "Details", "Sheet: Details\nrow 1: A1=June | B1=47"),
        ),
        members=(
            SourceMember("notes.txt", 16, True, kind="text"),
            SourceMember("results.xlsx", 500, True, kind="workbook"),
            SourceMember("scan.bin", 20, False, reason="unsupported archive member type"),
        ),
    )


def _plain(**overrides: object) -> PlainFetchResult:
    values: dict[str, object] = {
        "status": "ok",
        "method": ladder_adapter.LOCAL_NAVIGATION_METHOD,
        "text": "inventory from ladder",
        "links": [],
        "url": _URL,
        "content_type": "application/zip",
        "local_kind": "archive",
        "navigation_only": True,
    }
    values.update(overrides)
    return PlainFetchResult(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_fetch_archive_navigates_then_reads_exact_member_and_sheet(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _archive()
    monkeypatch.setattr(agentic_tools, "_fetch_via_ladder", AsyncMock(return_value=_plain()))
    monkeypatch.setattr(agentic_tools.run_cache, "local_source_for", lambda _url: source)

    inventory = await agentic_tools.fetch(_URL)
    member_inventory = await agentic_tools.fetch(_URL, member="results.xlsx")
    selected = await agentic_tools.fetch(_URL, member="results.xlsx", sheet="Summary")

    assert inventory.method == ladder_adapter.LOCAL_NAVIGATION_METHOD
    assert "results.xlsx" in inventory.content_markdown
    assert "B2=42" not in inventory.content_markdown
    assert member_inventory.method == ladder_adapter.LOCAL_NAVIGATION_METHOD
    assert "Sheet: Summary" in member_inventory.content_markdown
    assert "B2=42" not in member_inventory.content_markdown
    assert selected.method == ladder_adapter.LOCAL_SOURCE_METHOD
    assert "[Member: results.xlsx | Sheet: Summary]" in selected.content_markdown
    assert "B2=42" in selected.content_markdown


@pytest.mark.asyncio
async def test_fetch_local_selection_errors_are_typed_and_never_reenter_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch_via_ladder = AsyncMock(return_value=_plain())
    monkeypatch.setattr(agentic_tools, "_fetch_via_ladder", fetch_via_ladder)
    monkeypatch.setattr(agentic_tools.run_cache, "local_source_for", lambda _url: _archive())

    outcome = await agentic_tools.fetch(_URL, sheet="Summary")

    assert outcome.status == "error"
    assert outcome.method == "local_selection"
    assert "Select an archive member" in outcome.content_markdown
    fetch_via_ladder.assert_awaited_once()


@pytest.mark.asyncio
async def test_read_document_queries_all_local_sections_without_paid_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agentic_tools,
        "_acquire_local_document",
        AsyncMock(return_value=agentic_tools.local_document.HeldDocument(source=_archive())),
    )
    paid_reader = AsyncMock(side_effect=AssertionError("parsed local source must never use the paid reader"))
    monkeypatch.setattr(agentic_tools, "_run_document_read_sync", paid_reader)

    outcome = await agentic_tools.read_document(_URL, "June value")

    assert outcome.status == "ok"
    assert outcome.method == agentic_tools.local_document.DIGEST_LOCAL_METHOD
    assert "Searched 3 readable sections" in outcome.content_markdown
    assert "June" in outcome.content_markdown
    paid_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_document_local_refusal_never_uses_paid_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    refusal = _plain(
        status="error",
        method="plain",
        text="Local archive read refused: expanded size limit exceeded.",
        local_read_refused=True,
        navigation_only=False,
    )
    monkeypatch.setattr(
        agentic_tools,
        "_acquire_local_document",
        AsyncMock(return_value=agentic_tools.local_document.HeldDocument(local_refusal=refusal)),
    )
    paid_reader = AsyncMock(side_effect=AssertionError("typed local refusal must be terminal"))
    monkeypatch.setattr(agentic_tools, "_run_document_read_sync", paid_reader)

    outcome = await agentic_tools.read_document(_URL, "value")

    assert outcome.status == "error"
    assert outcome.method == "plain"
    assert "expanded size limit" in outcome.content_markdown
    paid_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_digest_timeout_is_terminal_and_never_uses_paid_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()

    def slow_digest(*args: object, **kwargs: object) -> DocumentDigest:
        del args, kwargs
        release.wait(timeout=1)
        return DocumentDigest(block="late digest", passages=1)

    paid_reader = MagicMock(side_effect=AssertionError("timed-out local digest must never use the paid reader"))
    monkeypatch.setattr(
        agentic_tools,
        "_acquire_local_document",
        AsyncMock(return_value=agentic_tools.local_document.HeldDocument(source=_archive())),
    )
    monkeypatch.setattr(agentic_tools.source_presentation, "digest_source", slow_digest)
    monkeypatch.setattr(agentic_tools, "pdf_parse_semaphore", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(agentic_tools, "_READ_DOCUMENT_TOTAL_BUDGET_S", 0.01)
    monkeypatch.setattr(agentic_tools, "_run_document_read_sync", paid_reader)

    outcome = await agentic_tools.read_document(_URL, "value")
    release.set()

    assert outcome.status == "error"
    assert outcome.method == "local_selection"
    assert "timed out" in outcome.content_markdown
    paid_reader.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_local_digest_holds_worker_capacity_until_thread_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    guard = threading.Lock()
    running = 0
    peak_running = 0

    def slow_digest(*args: object, **kwargs: object) -> DocumentDigest:
        del args, kwargs
        nonlocal running, peak_running
        with guard:
            running += 1
            peak_running = max(peak_running, running)
        started.set()
        release.wait(timeout=2)
        with guard:
            running -= 1
        return DocumentDigest(block="digest", passages=1)

    gate = asyncio.Semaphore(1)
    monkeypatch.setattr(agentic_tools.source_presentation, "digest_source", slow_digest)
    monkeypatch.setattr(agentic_tools, "pdf_parse_semaphore", lambda: gate)

    cancelled = asyncio.create_task(
        agentic_tools._local_source_digest_outcome(
            _URL,
            "value",
            _archive(),
            member=None,
            sheet=None,
            budget_seconds=1,
        )
    )
    await asyncio.to_thread(started.wait, 1)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    queued = asyncio.create_task(
        agentic_tools._local_source_digest_outcome(
            _URL,
            "value",
            _archive(),
            member=None,
            sheet=None,
            budget_seconds=1,
        )
    )
    await asyncio.sleep(0.05)
    assert peak_running == 1
    release.set()
    outcome = await queued

    assert outcome.status == "ok"
    assert peak_running == 1


@pytest.mark.asyncio
async def test_outer_acquisition_wall_still_hands_generic_timeout_to_paid_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow_ordinary_read(
        _url: str,
        *,
        ctx: LadderContext | None,
        local_read_receipt: LocalReadReceipt,
    ) -> agentic_tools.local_document.HeldDocument:
        del ctx, local_read_receipt
        await asyncio.sleep(10)
        raise AssertionError("outer wall did not cancel ordinary local acquisition")

    paid_reader = MagicMock(return_value=("Paid reader answer.", 1, ["SUCCESS"]))
    monkeypatch.setattr(agentic_tools, "_run_local_document_ladder", slow_ordinary_read)
    monkeypatch.setattr(agentic_tools, "_LOCAL_DOCUMENT_BUDGET_S", 0.01)
    monkeypatch.setattr(agentic_tools, "_url_context_robots_skip", AsyncMock(return_value=False))
    monkeypatch.setattr(agentic_tools, "_run_document_read_sync", paid_reader)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    outcome = await asyncio.wait_for(agentic_tools.read_document(_URL, "value"), timeout=1)

    assert outcome.status == "ok"
    assert outcome.method == "document"
    assert outcome.content_markdown == "Paid reader answer."
    paid_reader.assert_called_once()


@pytest.mark.asyncio
async def test_outer_wall_after_local_bytes_is_terminal_and_never_pays(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow_local_parse(
        _url: str,
        *,
        ctx: LadderContext | None,
        local_read_receipt: LocalReadReceipt,
    ) -> agentic_tools.local_document.HeldDocument:
        del ctx
        local_read_receipt.encountered = True
        await asyncio.sleep(10)
        raise AssertionError("outer wall did not cancel local parsing wait")

    paid_reader = MagicMock(side_effect=AssertionError("timed-out local source must never use the paid reader"))
    monkeypatch.setattr(agentic_tools, "_run_local_document_ladder", slow_local_parse)
    monkeypatch.setattr(agentic_tools, "_LOCAL_DOCUMENT_BUDGET_S", 0.01)
    monkeypatch.setattr(agentic_tools, "_run_document_read_sync", paid_reader)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    outcome = await asyncio.wait_for(agentic_tools.read_document(_URL, "value"), timeout=1)

    assert outcome.status == "error"
    assert outcome.method == "local_selection"
    assert "bytes were acquired" in outcome.content_markdown
    paid_reader.assert_not_called()


@pytest.mark.asyncio
async def test_outer_wall_detects_local_source_cached_during_acquisition(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow_cache_presentation(
        _url: str,
        *,
        ctx: LadderContext | None,
        local_read_receipt: LocalReadReceipt,
    ) -> agentic_tools.local_document.HeldDocument:
        del ctx, local_read_receipt
        await asyncio.sleep(10)
        raise AssertionError("outer wall did not cancel cached source presentation")

    cache_reads = iter((None, _archive()))
    monkeypatch.setattr(agentic_tools.run_cache, "local_source_for", lambda _url: next(cache_reads))
    monkeypatch.setattr(agentic_tools, "_run_local_document_ladder", slow_cache_presentation)
    monkeypatch.setattr(agentic_tools, "_LOCAL_DOCUMENT_BUDGET_S", 0.01)

    held = await asyncio.wait_for(agentic_tools._acquire_local_document(_URL), timeout=1)

    assert held.local_refusal is not None
    assert held.local_refusal.local_read_refused is True


def test_fetch_continuations_fit_final_dispatch_budget_without_losing_text() -> None:
    text = "x" * 17_000
    links = ["https://example.com/" + "a" * 200, "https://example.com/" + "b" * 200]
    starts = [0]
    reconstructed = ""

    while starts:
        start = starts.pop()
        outcome = agentic_tools._render_fetch_outcome(_URL, text, links, method="plain", start_char=start)
        dispatched = _format_tool_content("fetch", outcome, 8_000)
        assert len(dispatched) <= 8_000
        marker_start = outcome.content_markdown.find("\n[truncated at ")
        body = outcome.content_markdown if marker_start < 0 else outcome.content_markdown[:marker_start]
        reconstructed += body
        if outcome.truncated:
            marker = outcome.content_markdown[marker_start:]
            starts.append(int(marker.split("start_char=", 1)[1].split("]", 1)[0]))

    assert reconstructed == text


@pytest.mark.asyncio
async def test_long_archive_inventory_paginates_to_late_exact_member(monkeypatch: pytest.MonkeyPatch) -> None:
    members = tuple(
        SourceMember(f"nested/{index:03d}-" + "long-name-" * 10 + ".txt", 10, True, kind="text") for index in range(128)
    )
    late_member = members[-1].name
    source = ParsedSource(
        kind="archive",
        sections=tuple(SourceSection(member.name, None, f"content for {member.name}") for member in members),
        members=members,
    )
    monkeypatch.setattr(agentic_tools, "_fetch_via_ladder", AsyncMock(return_value=_plain()))
    monkeypatch.setattr(agentic_tools.run_cache, "local_source_for", lambda _url: source)

    first = await agentic_tools.fetch(_URL)
    pages = [first]
    while pages[-1].truncated:
        continuation_start = int(pages[-1].content_markdown.split("start_char=", 1)[1].split("]", 1)[0])
        pages.append(await agentic_tools.fetch(_URL, start_char=continuation_start))
    selected = await agentic_tools.fetch(_URL, member=late_member)

    assert first.truncated is True
    assert late_member not in first.content_markdown
    assert late_member in pages[-1].content_markdown
    assert selected.method == ladder_adapter.LOCAL_SOURCE_METHOD
    assert f"content for {late_member}" in selected.content_markdown


def test_public_source_tool_schemas_advertise_exact_selectors() -> None:
    by_name = {tool.name: tool.parameters for tool in agentic_tools.build_gap_fill_tools("topic")}

    assert set(by_name["fetch"]["properties"]) == {"url", "start_char", "member", "sheet"}
    assert set(by_name["read_document"]["properties"]) == {"url", "ask", "member", "sheet"}
    assert by_name["fetch"]["required"] == ["url"]
    assert by_name["read_document"]["required"] == ["url", "ask"]
    assert by_name["view_image"]["required"] == ["url"]
    assert by_name["view_image"]["properties"]["crop"]["minItems"] == 4


def test_formatter_caps_large_link_metadata_even_when_body_is_empty() -> None:
    outcome = ToolOutcome(
        content_markdown="",
        method="plain",
        links=[f"https://example.com/{index}/" + "x" * 1_000 for index in range(25)],
    )

    assert len(_format_tool_content("fetch", outcome, 8_000)) <= 8_000
