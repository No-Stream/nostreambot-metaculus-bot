"""Neutral records for source excerpts, structured rows, and provider text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SourceExcerpt:
    """Delivered text with its source, retrieval path, and content origin.

    ``structured_data`` is an adapter-rendered row, which may omit fields such
    as units; ``provider_summary`` and ``generated_answer`` are not source text.
    """

    source_url: str
    text: str
    retrieval_method: str
    content_origin: Literal["source_text", "structured_data", "provider_summary", "generated_answer"] = "source_text"
    clipped: bool = False
