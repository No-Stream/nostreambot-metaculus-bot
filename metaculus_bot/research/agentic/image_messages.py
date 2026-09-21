"""Keep image bytes out of loop state and resolve references only for an LLM request."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Mapping
from typing import Any

from metaculus_bot.research.agentic.llm import LlmCall
from metaculus_bot.research.image_assets import ImageView

_IMAGE_REFERENCE_TYPE = "image_reference"
_IMAGE_BATCH_LABEL = "Images returned by the preceding tool calls follow."


def image_reference_message(image_ids: Iterable[str]) -> dict[str, Any]:
    """Build the byte-free message stored after a batch's textual tool messages."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": _IMAGE_BATCH_LABEL},
            *[{"type": _IMAGE_REFERENCE_TYPE, "image_id": image_id} for image_id in image_ids],
        ],
    }


def image_source_urls(image_view: ImageView) -> set[str]:
    return {url for url in (image_view.source_url, image_view.final_url) if url}


def _materialize_messages(
    messages: list[dict[str, Any]],
    image_views_by_id: Mapping[str, ImageView],
    image_sources_by_id: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list) or not any(
            isinstance(block, dict) and block.get("type") == _IMAGE_REFERENCE_TYPE for block in content
        ):
            materialized.append(message)
            continue

        materialized_blocks: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != _IMAGE_REFERENCE_TYPE:
                materialized_blocks.append(block)
                continue
            image_id = block.get("image_id")
            if not isinstance(image_id, str) or image_id not in image_views_by_id:
                raise ValueError(f"Unknown image reference: {image_id!r}")
            image_view = image_views_by_id[image_id]
            metadata = image_view.metadata() | {"source_urls": sorted(image_sources_by_id.get(image_id, set()))}
            materialized_blocks.extend(
                [
                    {
                        "type": "text",
                        "text": "Image metadata: " + json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(image_view.png_bytes).decode("ascii")
                        },
                    },
                ]
            )
        materialized.append(message | {"content": materialized_blocks})
    return materialized


def bind_image_materialization(
    llm_call: LlmCall,
    image_views_by_id: Mapping[str, ImageView],
    image_sources_by_id: Mapping[str, set[str]],
) -> LlmCall:
    """Decorate either the production transport or a test double with reference resolution."""

    async def call_with_images(
        messages: list[dict[str, Any]],
        tools_json: list[dict[str, Any]] | None,
        *,
        tool_choice: str | None = None,
    ) -> Any:
        materialized = _materialize_messages(messages, image_views_by_id, image_sources_by_id)
        if tool_choice is None:
            return await llm_call(materialized, tools_json)
        return await llm_call(materialized, tools_json, tool_choice=tool_choice)

    return call_with_images
