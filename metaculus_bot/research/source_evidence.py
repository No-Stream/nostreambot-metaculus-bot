"""Neutral records for source excerpts, structured rows, and provider text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from metaculus_bot.constants import GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS


def bounded_passages(text: str, *, tabular: bool = False) -> list[str]:
    """Return verbatim bounded passages, splitting oversized text at natural boundaries."""
    passages: list[str] = []
    for paragraph in text.split("\n\n"):
        if not paragraph.strip():
            continue
        if len(paragraph) <= GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS:
            passages.append(paragraph)
            continue
        for line in paragraph.splitlines():
            if line.strip() and len(line) <= GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS:
                passages.append(line)
            elif line.strip() and not tabular:
                passages.extend(_sentence_passages(line))
    return passages


def _sentence_passages(line: str) -> list[str]:
    boundaries = [match.end() for match in re.finditer(r"(?<=[.!?])\s+", line)] + [len(line)]
    passages: list[str] = []
    chunk_start = 0
    sentence_start = 0
    for end in boundaries:
        if end - chunk_start > GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS:
            if sentence_start > chunk_start:
                passages.append(line[chunk_start:sentence_start])
            chunk_start = sentence_start
        if end - sentence_start > GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS:
            chunk_start = end
        sentence_start = end
    if chunk_start < len(line):
        passages.append(line[chunk_start:])
    return passages


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
