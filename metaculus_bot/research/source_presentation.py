"""Local inventories and query excerpts over complete, named source sections."""

from __future__ import annotations

from collections.abc import Sequence

from metaculus_bot.constants import DOCUMENT_DIGEST_TOP_K
from metaculus_bot.research.document_text import DocumentDigest, render_flat_passages, select_passages
from metaculus_bot.research.source_documents import ParsedSource, SourceSection


def select_source_sections(
    source: ParsedSource, *, member: str | None = None, sheet: str | None = None
) -> tuple[SourceSection, ...]:
    if member is not None:
        if source.kind != "archive":
            raise ValueError("member is only valid for an archive")
        entry = next((entry for entry in source.members if entry.name == member), None)
        if entry is None:
            raise ValueError(f"Unknown archive member: {member}")
        if not entry.readable:
            raise ValueError(f"Unreadable archive member {member}: {entry.reason or 'unsupported format'}")
    elif sheet is not None and source.kind == "archive":
        raise ValueError("Select an archive member before selecting a sheet")
    sections = tuple(section for section in source.sections if member is None or section.member == member)
    if sheet is not None:
        sections = tuple(section for section in sections if section.sheet == sheet)
        if not sections:
            raise ValueError(f"Unknown sheet: {sheet}")
    return sections


def section_label(section: SourceSection) -> str:
    parts = []
    if section.member is not None:
        parts.append(f"Member: {section.member}")
    if section.sheet is not None:
        parts.append(f"Sheet: {section.sheet}")
    return " | ".join(parts) or "Document text"


def source_inventory(source: ParsedSource, *, member: str | None = None) -> str:
    lines = ["Navigation only; select content to read before citing it as evidence."]
    if source.kind == "archive" and member is None:
        lines.append("Archive members (use the exact member name):")
        for entry in source.members:
            status = (entry.kind or "readable") if entry.readable else f"unsupported: {entry.reason or 'format'}"
            lines.append(f"- {entry.name} ({entry.size_bytes} bytes; {status})")
    else:
        sections = select_source_sections(source, member=member)
        lines.append("Sheets (use the exact sheet name):")
        for section in sections:
            dimensions = ""
            if section.rows is not None and section.columns is not None:
                dimensions = f"; {section.rows} rows, {section.columns} columns"
            lines.append(f"- {section_label(section)} ({len(section.text)} chars{dimensions})")
    lines.extend(source.notices)
    return "\n".join(lines)


def source_text(sections: Sequence[SourceSection]) -> str:
    return "\n\n".join(f"[{section_label(section)}]\n{section.text}" for section in sections)


def digest_source(
    source: ParsedSource,
    *,
    query: str,
    source_url: str,
    max_chars: int,
    member: str | None = None,
    sheet: str | None = None,
) -> DocumentDigest:
    sections = select_source_sections(source, member=member, sheet=sheet)
    starts: list[int] = []
    chunks: list[str] = []
    offset = 0
    for section in sections:
        starts.append(offset)
        chunk = section.text + "\n"
        chunks.append(chunk)
        offset += len(chunk)
    joined = "".join(chunks)
    passages = select_passages(joined, query, top_k=DOCUMENT_DIGEST_TOP_K, page_breaks=starts)
    labeled = []
    for passage in passages:
        assert passage.page is not None
        section = sections[passage.page - 1]
        section_start = starts[passage.page - 1]
        labeled.append(
            f"[{section_label(section)}; chars {passage.start - section_start}:{passage.end - section_start}]\n"
            f"{passage.text}"
        )
    unread_members = sum(not entry.readable for entry in source.members)
    coverage = f"Searched {len(sections)} readable sections; {unread_members} archive members not read."
    if not labeled:
        labeled = [f"No matching passage in the readable sections for: {query}."]
    block = render_flat_passages(
        [coverage, *source.notices, *labeled],
        query=query,
        max_chars=max_chars,
        source_url=source_url,
        source_chars=len(joined),
    )
    return DocumentDigest(block=block, passages=len(passages))
