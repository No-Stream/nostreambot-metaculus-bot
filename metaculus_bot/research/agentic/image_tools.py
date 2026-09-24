"""Public same-agent image viewing over bytes retained by the shared fetch ladder."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from metaculus_bot.constants import GAP_FILL_IMAGE_MAX_VIEWS
from metaculus_bot.research.agentic.fetch_outcomes import PlainFetchResult
from metaculus_bot.research.agentic.types import ToolOutcome
from metaculus_bot.research.fetch_ladder import run_cache
from metaculus_bot.research.http_fetch import pdf_parse_semaphore
from metaculus_bot.research.image_assets import ImageReadError, ImageView, normalize_image
from metaculus_bot.research.image_leads import ImageLead

ImageAcquire = Callable[[str], Awaitable[PlainFetchResult]]


class ImageViewState:
    """One tool closure's normalization gate and distinct-pixel budget."""

    def __init__(self, *, worker_gate: asyncio.Semaphore | None = None) -> None:
        self._worker_gate = worker_gate
        self._view_lock = asyncio.Lock()
        self._image_ids: set[str] = set()
        self._parent_pages_by_image_url: dict[str, set[str]] = {}

    async def remember_leads(self, parent_page_urls: tuple[str, ...], leads: tuple[ImageLead, ...]) -> None:
        """Retain page provenance separately from the image's own URL aliases."""
        async with self._view_lock:
            for lead in leads:
                self._parent_pages_by_image_url.setdefault(lead.url, set()).update(parent_page_urls)

    async def normalize(
        self,
        body: bytes,
        *,
        source_url: str,
        final_url: str,
        crop: tuple[int, int, int, int] | None,
    ) -> ImageView:
        """Normalize off-loop without freeing a cancelled worker's slot early."""
        worker_gate = self._worker_gate or pdf_parse_semaphore()
        await worker_gate.acquire()
        async with self._view_lock:
            parent_page_urls = tuple(sorted(self._parent_pages_by_image_url.get(source_url, ())))
        worker = asyncio.create_task(
            asyncio.to_thread(
                normalize_image,
                body,
                source_url=source_url,
                final_url=final_url,
                crop=crop,
                parent_page_urls=parent_page_urls,
            )
        )
        worker.add_done_callback(lambda finished: self._normalization_finished(finished, worker_gate))
        view = await asyncio.shield(worker)

        async with self._view_lock:
            if view.image_id not in self._image_ids and len(self._image_ids) >= GAP_FILL_IMAGE_MAX_VIEWS:
                raise ImageReadError(
                    f"this research run has already used its {GAP_FILL_IMAGE_MAX_VIEWS} distinct image views"
                )
            self._image_ids.add(view.image_id)
        return view

    @staticmethod
    def _normalization_finished(worker: asyncio.Task[ImageView], worker_gate: asyncio.Semaphore) -> None:
        worker_gate.release()
        if worker.cancelled():
            return
        worker.exception()


def _crop_tuple(crop: list[int] | tuple[int, ...] | None) -> tuple[int, int, int, int] | None:
    if crop is None:
        return None
    if len(crop) != 4:
        raise ImageReadError("crop must contain exactly four integer coordinates")
    values = tuple(crop)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ImageReadError("crop must contain exactly four integer coordinates")
    return values  # type: ignore[return-value]


async def view_image(
    url: str,
    crop: list[int] | tuple[int, ...] | None = None,
    *,
    state: ImageViewState,
    acquire: ImageAcquire,
) -> ToolOutcome:
    """Fetch through the shared ladder, normalize retained raster bytes, and return one view."""
    plain = await acquire(url)
    if plain.status != "ok":
        return ToolOutcome(content_markdown=plain.text, method=plain.method, status=plain.status)
    return await view_acquired_image(url, plain, crop, state=state)


async def view_acquired_image(
    url: str,
    plain: PlainFetchResult,
    crop: list[int] | tuple[int, ...] | None = None,
    *,
    state: ImageViewState,
) -> ToolOutcome:
    """Normalize an image result already acquired and retained by the shared ladder."""
    if plain.local_kind != "image":
        return ToolOutcome(
            content_markdown=(
                f"Image view refused: {url} did not resolve to a supported raster image "
                f"(received {plain.content_type or 'unknown content type'})."
            ),
            method="image_local",
            status="error",
        )

    source = run_cache.image_source_for(url) or run_cache.image_source_for(plain.url)
    if source is None:
        return ToolOutcome(
            content_markdown="Image view failed: the shared fetch ladder did not retain image bytes.",
            method="image_local",
            status="error",
        )

    try:
        crop_tuple = _crop_tuple(crop)
        view = await state.normalize(
            source.body,
            source_url=url,
            final_url=source.url,
            crop=crop_tuple,
        )
    except ImageReadError as error:
        return ToolOutcome(
            content_markdown=f"Image view refused: {error.reason}.",
            method="image_local",
            status="error",
        )

    metadata = json.dumps(view.metadata(), ensure_ascii=False, separators=(",", ":"))
    links = list(dict.fromkeys((view.source_url, view.final_url)))
    return ToolOutcome(
        content_markdown=f"image_id={view.image_id}\nmetadata={metadata}",
        links=links,
        method="image_local",
        image_views=[view],
    )
