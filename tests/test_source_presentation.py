from __future__ import annotations

import pytest

from metaculus_bot.research.source_documents import ParsedSource, SourceMember, SourceSection
from metaculus_bot.research.source_presentation import (
    digest_source,
    select_source_sections,
    source_inventory,
    source_text,
)


@pytest.fixture
def archive() -> ParsedSource:
    return ParsedSource(
        kind="archive",
        sections=(
            SourceSection("old.csv", None, "date,cdi\n2025-01-01,2\n"),
            SourceSection("current.xlsx", "Regions", "Row\tA\tB\n1\tRegion\tDrought index\n2\tZurich\t5\n", 2, 2),
            SourceSection("current.xlsx", "Forecast", "Row\tA\tB\n1\tRegion\tForecast\n2\tZurich\t3\n", 2, 2),
        ),
        members=(
            SourceMember("old.csv", 24, True, "text"),
            SourceMember("current.xlsx", 250, True, "workbook"),
            SourceMember("nested.zip", 100, False, "archive", "Nested archives are unsupported"),
        ),
    )


def test_inventory_is_navigation_not_a_first_member_read(archive: ParsedSource) -> None:
    text = source_inventory(archive)
    assert "Navigation only" in text
    assert "old.csv" in text
    assert "current.xlsx" in text
    assert "nested.zip" in text
    assert "unsupported" in text
    assert "2025-01-01" not in text
    selected = source_inventory(archive, member="current.xlsx")
    assert "Regions" in selected
    assert "Forecast" in selected
    assert "Zurich" not in selected


def test_exact_member_and_sheet_selection_keeps_labels(archive: ParsedSource) -> None:
    sections = select_source_sections(archive, member="current.xlsx", sheet="Regions")
    assert len(sections) == 1
    text = source_text(sections)
    assert "current.xlsx" in text
    assert "Regions" in text
    assert "Zurich\t5" in text
    assert "Forecast" not in text


@pytest.mark.parametrize(
    ("member", "sheet"),
    [("missing.csv", None), ("nested.zip", None), ("current.xlsx", "missing"), (None, "Regions")],
)
def test_bad_selectors_fail_explicitly(archive: ParsedSource, member: str | None, sheet: str | None) -> None:
    with pytest.raises(ValueError, match=r"Unknown|Unreadable|Select an archive member"):
        select_source_sections(archive, member=member, sheet=sheet)


def test_digest_labels_every_selected_section_and_respects_cap(archive: ParsedSource) -> None:
    digest = digest_source(archive, query="Zurich drought", source_url="https://example.org/data.zip", max_chars=1000)
    assert digest.passages > 0
    assert "current.xlsx" in digest.block
    assert "Regions" in digest.block
    assert "Zurich" in digest.block
    assert len(digest.block) <= 1000
    assert "p." not in digest.block


def test_digest_finds_late_rows_without_only_scanning_first_window() -> None:
    source = ParsedSource(
        kind="workbook",
        sections=(SourceSection(None, "History", "Old observations\n" * 3000 + "Row 3001: Zurich drought index 5"),),
    )
    digest = digest_source(source, query="Zurich drought", source_url="https://example.org/data.xlsx", max_chars=1200)
    assert "Zurich drought index 5" in digest.block
    assert "History" in digest.block
    assert len(digest.block) <= 1200


def test_no_match_is_not_an_unreadability_claim(archive: ParsedSource) -> None:
    digest = digest_source(
        archive, query="unrelated volcanology", source_url="https://example.org/data.zip", max_chars=800
    )
    assert digest.passages == 0
    assert "No matching passage" in digest.block
    assert "unsupported" not in digest.block.lower()
