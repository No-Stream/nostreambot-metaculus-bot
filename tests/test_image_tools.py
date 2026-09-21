"""Public same-agent image viewing over bytes retained by the shared ladder."""

from __future__ import annotations

import asyncio
import threading
from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from metaculus_bot.constants import GAP_FILL_IMAGE_MAX_VIEWS
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.agentic.fetch_outcomes import PlainFetchResult
from metaculus_bot.research.agentic.image_tools import ImageViewState, view_image
from metaculus_bot.research.fetch_ladder.run_cache import ImageSource
from metaculus_bot.research.image_assets import ImageView
from metaculus_bot.research.image_leads import ImageLead

_URL = "https://example.com/chart.png"
_FINAL_URL = "https://cdn.example.com/chart.png"


def _png(color: tuple[int, int, int]) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (20, 10), color).save(stream, format="PNG")
    return stream.getvalue()


def _plain_image() -> PlainFetchResult:
    return PlainFetchResult(
        status="ok",
        method="local_navigation",
        text="Raster image retained for local viewing.",
        links=[],
        url=_FINAL_URL,
        content_type="image/png",
        local_kind="image",
        navigation_only=True,
    )


@pytest.mark.asyncio
async def test_view_image_fetches_only_through_ladder_and_returns_pixels_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ImageSource(url=_FINAL_URL, body=_png((255, 0, 0)), content_type="image/png")
    acquire = AsyncMock(return_value=_plain_image())
    monkeypatch.setattr(
        "metaculus_bot.research.agentic.image_tools.run_cache.image_source_for",
        lambda url: source if url in {_URL, _FINAL_URL} else None,
    )

    outcome = await view_image(_URL, state=ImageViewState(), acquire=acquire)

    acquire.assert_awaited_once_with(_URL)
    assert outcome.status == "ok"
    assert outcome.method == "image_local"
    assert outcome.links == [_URL, _FINAL_URL]
    assert len(outcome.image_views) == 1
    assert outcome.image_views[0].source_url == _URL
    assert outcome.image_views[0].final_url == _FINAL_URL
    assert "image_id=" in outcome.content_markdown
    assert "pixels" not in outcome.content_markdown.lower()
    assert outcome.image_views[0].png_bytes not in outcome.content_markdown.encode()


@pytest.mark.asyncio
async def test_view_image_validates_crop_in_oriented_source_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    source = ImageSource(url=_FINAL_URL, body=_png((0, 255, 0)), content_type="image/png")
    monkeypatch.setattr("metaculus_bot.research.agentic.image_tools.run_cache.image_source_for", lambda _url: source)

    outcome = await view_image(
        _URL,
        crop=[1, 2, 11, 8],
        state=ImageViewState(),
        acquire=AsyncMock(return_value=_plain_image()),
    )

    assert outcome.image_views[0].crop == (1, 2, 11, 8)
    assert outcome.image_views[0].original_width == 20
    assert outcome.image_views[0].original_height == 10
    assert outcome.image_views[0].width == 10
    assert outcome.image_views[0].height == 6


@pytest.mark.asyncio
async def test_image_lead_retains_requested_and_final_parent_pages_separately_from_image_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ImageSource(url=_FINAL_URL, body=_png((0, 0, 255)), content_type="image/png")
    monkeypatch.setattr("metaculus_bot.research.agentic.image_tools.run_cache.image_source_for", lambda _url: source)
    state = ImageViewState()
    await state.remember_leads(
        ("https://example.com/report", "https://cdn.example.com/report"),
        (ImageLead(url=_URL, filename="chart.png"),),
    )

    outcome = await view_image(_URL, state=state, acquire=AsyncMock(return_value=_plain_image()))

    assert outcome.image_views[0].parent_page_urls == (
        "https://cdn.example.com/report",
        "https://example.com/report",
    )
    assert outcome.links == [_URL, _FINAL_URL]


@pytest.mark.asyncio
async def test_redirected_html_lead_retains_both_parent_urls_through_public_fetch_and_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_page = "https://example.com/report"
    final_page = "https://cdn.example.com/report"
    page = PlainFetchResult(
        status="ok",
        method="plain",
        text="Report page",
        links=[],
        url=final_page,
        content_type="text/html",
        image_leads=(ImageLead(url=_URL, filename="chart.png"),),
    )
    source = ImageSource(url=_FINAL_URL, body=_png((0, 0, 255)), content_type="image/png")
    monkeypatch.setattr(agentic_tools, "_fetch_via_ladder", AsyncMock(side_effect=[page, _plain_image()]))
    monkeypatch.setattr("metaculus_bot.research.agentic.image_tools.run_cache.image_source_for", lambda _url: source)
    tools = agentic_tools.build_gap_fill_tools("topic")
    fetch_tool = next(tool for tool in tools if tool.name == "fetch")
    view_tool = next(tool for tool in tools if tool.name == "view_image")

    await fetch_tool.handler(url=requested_page)
    outcome = await view_tool.handler(url=_URL)

    assert outcome.image_views[0].parent_page_urls == (final_page, requested_page)


@pytest.mark.asyncio
async def test_public_fetch_then_read_document_deliver_cached_pixels_without_paid_or_second_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ImageSource(url=_FINAL_URL, body=_png((20, 40, 60)), content_type="image/png")
    acquire = AsyncMock(return_value=_plain_image())
    paid_reader = AsyncMock(side_effect=AssertionError("direct image entrypoints must never use the paid reader"))
    monkeypatch.setattr(agentic_tools, "_fetch_via_ladder", acquire)
    monkeypatch.setattr(agentic_tools, "_run_document_read_sync", paid_reader)
    monkeypatch.setattr("metaculus_bot.research.agentic.image_tools.run_cache.image_source_for", lambda _url: source)
    monkeypatch.setattr(agentic_tools.run_cache, "image_source_for", lambda _url: source)
    tools = agentic_tools.build_gap_fill_tools("topic")
    fetch_tool = next(tool for tool in tools if tool.name == "fetch")
    read_tool = next(tool for tool in tools if tool.name == "read_document")

    fetched = await fetch_tool.handler(url=_URL)
    read = await read_tool.handler(url=_URL, ask="What does the chart show?")

    assert fetched.method == read.method == "image_local"
    assert fetched.status == read.status == "ok"
    assert len(fetched.image_views) == len(read.image_views) == 1
    assert fetched.image_views[0].png_bytes == read.image_views[0].png_bytes
    acquire.assert_awaited_once()
    paid_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_document_malformed_image_is_terminal_without_paid_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    source = ImageSource(url=_URL, body=b"not an image", content_type="image/png")
    paid_reader = AsyncMock(side_effect=AssertionError("malformed local image must never use the paid reader"))
    monkeypatch.setattr(agentic_tools, "_run_document_read_sync", paid_reader)
    monkeypatch.setattr(agentic_tools.run_cache, "image_source_for", lambda _url: source)
    monkeypatch.setattr("metaculus_bot.research.agentic.image_tools.run_cache.image_source_for", lambda _url: source)
    read_tool = next(tool for tool in agentic_tools.build_gap_fill_tools("topic") if tool.name == "read_document")

    outcome = await read_tool.handler(url=_URL, ask="What does it show?")

    assert outcome.status == "error"
    assert outcome.method == "image_local"
    assert "corrupt, unsupported" in outcome.content_markdown
    paid_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_read_document_and_view_image_share_four_distinct_view_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urls = [f"https://example.com/chart-{index}.png" for index in range(GAP_FILL_IMAGE_MAX_VIEWS + 1)]
    sources = {
        url: ImageSource(url=url, body=_png((index, index, index)), content_type="image/png")
        for index, url in enumerate(urls)
    }

    async def acquire(url: str, **_kwargs: object) -> PlainFetchResult:
        return PlainFetchResult(
            status="ok",
            method="local_navigation",
            text="retained",
            links=[],
            url=url,
            content_type="image/png",
            local_kind="image",
            navigation_only=True,
        )

    paid_reader = AsyncMock(side_effect=AssertionError("direct image entrypoints must never use the paid reader"))
    monkeypatch.setattr(agentic_tools, "_fetch_via_ladder", acquire)
    monkeypatch.setattr(agentic_tools, "_run_document_read_sync", paid_reader)
    monkeypatch.setattr("metaculus_bot.research.agentic.image_tools.run_cache.image_source_for", sources.get)
    monkeypatch.setattr(agentic_tools.run_cache, "image_source_for", sources.get)
    tools = agentic_tools.build_gap_fill_tools("topic")
    fetch_tool = next(tool for tool in tools if tool.name == "fetch")
    read_tool = next(tool for tool in tools if tool.name == "read_document")
    view_tool = next(tool for tool in tools if tool.name == "view_image")

    outcomes = [
        await fetch_tool.handler(url=urls[0]),
        await read_tool.handler(url=urls[1], ask="chart"),
        await view_tool.handler(url=urls[2]),
        await fetch_tool.handler(url=urls[3]),
        await read_tool.handler(url=urls[4], ask="chart"),
    ]

    assert all(outcome.status == "ok" for outcome in outcomes[:GAP_FILL_IMAGE_MAX_VIEWS])
    assert outcomes[-1].status == "error"
    assert outcomes[-1].method == "image_local"
    assert "already used its 4 distinct image views" in outcomes[-1].content_markdown
    paid_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_distinct_view_budget_is_concurrency_safe_but_duplicate_pixels_are_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ImageViewState()
    bodies = {_URL + str(index): _png((index, index, index)) for index in range(GAP_FILL_IMAGE_MAX_VIEWS + 1)}
    monkeypatch.setattr(
        "metaculus_bot.research.agentic.image_tools.run_cache.image_source_for",
        lambda url: ImageSource(url=url, body=bodies[url], content_type="image/png"),
    )

    async def acquire(url: str) -> PlainFetchResult:
        return PlainFetchResult(
            status="ok",
            method="local_navigation",
            text="retained",
            links=[],
            url=url,
            local_kind="image",
            navigation_only=True,
        )

    outcomes = await asyncio.gather(
        *(view_image(url, state=state, acquire=acquire) for url in bodies),
    )

    assert sum(outcome.status == "ok" for outcome in outcomes) == GAP_FILL_IMAGE_MAX_VIEWS
    assert sum(outcome.status == "error" for outcome in outcomes) == 1
    accepted_url = next(url for url, outcome in zip(bodies, outcomes, strict=True) if outcome.status == "ok")
    duplicate = await view_image(accepted_url, state=state, acquire=acquire)
    assert duplicate.status == "ok"


@pytest.mark.asyncio
async def test_cancelled_normalization_holds_worker_capacity_until_thread_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    running = 0
    peak_running = 0
    guard = threading.Lock()

    def slow_normalize(
        body: bytes,
        *,
        source_url: str,
        final_url: str,
        crop: tuple[int, int, int, int] | None,
        parent_page_urls: tuple[str, ...],
    ) -> ImageView:
        del parent_page_urls
        nonlocal running, peak_running
        with guard:
            running += 1
            peak_running = max(peak_running, running)
        started.set()
        release.wait(timeout=2)
        with guard:
            running -= 1
        return ImageView("id", source_url, final_url, "sha", 1, 1, 1, 1, crop, body)

    monkeypatch.setattr("metaculus_bot.research.agentic.image_tools.normalize_image", slow_normalize)
    state = ImageViewState(worker_gate=asyncio.Semaphore(1))
    source = ImageSource(_URL, b"pixels", "image/png")
    monkeypatch.setattr("metaculus_bot.research.agentic.image_tools.run_cache.image_source_for", lambda _url: source)
    acquire = AsyncMock(return_value=_plain_image())

    cancelled = asyncio.create_task(view_image(_URL, state=state, acquire=acquire))
    await asyncio.to_thread(started.wait, 1)
    cancelled.cancel()
    queued = asyncio.create_task(view_image(_URL, state=state, acquire=acquire))
    await asyncio.sleep(0.05)
    assert peak_running == 1
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await queued
    assert peak_running == 1


@pytest.mark.asyncio
async def test_view_image_never_escalates_a_non_image_or_local_refusal() -> None:
    state = ImageViewState()
    not_image = AsyncMock(return_value=PlainFetchResult("ok", "plain", "page", [], _URL, content_type="text/html"))
    refused = AsyncMock(
        return_value=PlainFetchResult(
            "error", "plain", "SVG images are unsupported.", [], _URL, local_read_refused=True
        )
    )

    page_outcome = await view_image(_URL, state=state, acquire=not_image)
    refused_outcome = await view_image(_URL, state=state, acquire=refused)

    assert page_outcome.status == "error"
    assert refused_outcome.status == "error"
    assert "unsupported" in refused_outcome.content_markdown


@pytest.mark.asyncio
async def test_repeated_invalid_images_do_not_exhaust_normalization_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    source = ImageSource(_URL, b"invalid", "image/png")
    monkeypatch.setattr("metaculus_bot.research.agentic.image_tools.run_cache.image_source_for", lambda _url: source)
    state = ImageViewState(worker_gate=asyncio.Semaphore(1))
    acquire = AsyncMock(return_value=_plain_image())

    refused = [await view_image(_URL, state=state, acquire=acquire) for _ in range(4)]
    source = ImageSource(_URL, _png((10, 20, 30)), "image/png")
    accepted = await view_image(_URL, state=state, acquire=acquire)

    assert all(outcome.status == "error" for outcome in refused)
    assert accepted.status == "ok"
