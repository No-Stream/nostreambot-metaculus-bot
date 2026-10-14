from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
from docx import Document

from metaculus_bot.constants import RAW_RESEARCH_LOG_DIR_ENV, RAW_RESEARCH_LOG_ENABLED_ENV
from metaculus_bot.research import raw_log
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.fetch_ladder import classify, run_cache
from metaculus_bot.research.fetch_ladder.verdict import PageExtraction
from metaculus_bot.research.raw_log import record_source_capture
from metaculus_bot.research.source_capture_persistence import (
    SourceCaptureIngestError,
    ingest_source_capture_records,
)
from scripts.download_raw_research import harvest_raw_logs_from_dir, merge_and_write
from tests.resolution_source_fakes import FakeResponse, FakeSession
from tests.test_document_text import build_text_pdf


@pytest.fixture
def enabled_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(RAW_RESEARCH_LOG_ENABLED_ENV, "true")
    monkeypatch.setenv(RAW_RESEARCH_LOG_DIR_ENV, str(tmp_path / "run_logs"))
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    return tmp_path / "run_logs"


def _raw_capture_record(log_dir: Path) -> dict[str, object]:
    log_path = log_dir / "raw_research_12345.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    return next(record for record in records if record["provider"] == "source_capture")


class TestRecordSourceCapture:
    def test_writes_bounded_content_addressed_sidecars_and_metadata(self, enabled_log_dir: Path) -> None:
        raw_body = b"<html><body>real source</body></html>"
        parsed_text = "First page\nSecond page"

        capture = record_source_capture(
            qid=808,
            source_url="https://example.org/source?token=private#section",
            retrieval_method="rendered",
            content_type="text/html; charset=utf-8",
            parser_version="trafilatura 2.2.0",
            raw_body=raw_body,
            parsed_text=parsed_text,
            truncated_by="",
            refusal_reason="",
        )

        assert capture is not None
        assert capture["source_url"] == "https://example.org/source"
        assert (
            capture["source_identity_sha256"]
            == hashlib.sha256(b"https://example.org/source?token=private#section").hexdigest()
        )
        assert capture["capture_status"] == "complete"
        assert capture["raw_sha256"] == hashlib.sha256(raw_body).hexdigest()
        assert capture["parsed_sha256"] == hashlib.sha256(parsed_text.encode()).hexdigest()
        raw_asset = enabled_log_dir / str(capture["raw_asset"])
        parsed_asset = enabled_log_dir / str(capture["parsed_asset"])
        assert raw_asset.read_bytes() == raw_body
        assert parsed_asset.read_text(encoding="utf-8") == parsed_text
        assert raw_asset.name.startswith(str(capture["raw_sha256"]))
        assert parsed_asset.name.startswith(str(capture["parsed_sha256"]))

        log_record = _raw_capture_record(enabled_log_dir)
        assert log_record["qid"] == 808
        assert log_record["phase"] == capture["capture_identity_sha256"]
        assert log_record["payload"] == capture
        serialized_log = (enabled_log_dir / "raw_research_12345.jsonl").read_text(encoding="utf-8")
        assert "real source" not in serialized_log
        assert "private" not in serialized_log

    def test_refuses_over_cap_assets_without_silent_truncation(
        self, enabled_log_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(raw_log, "DOCUMENT_TEXT_PDF_MAX_BYTES", 4)
        monkeypatch.setattr(raw_log, "LOCAL_SOURCE_MAX_CHARS", 3)
        capture = record_source_capture(
            qid=808,
            source_url="https://example.org/large.pdf",
            retrieval_method="direct_http",
            content_type="application/pdf",
            parser_version="pypdf",
            raw_body=b"oversize",
            parsed_text="too long",
            truncated_by="pages",
            refusal_reason="",
        )

        assert capture is not None
        assert capture["capture_status"] == "refused"
        assert capture["raw_capture_status"] == "refused"
        assert capture["parsed_capture_status"] == "refused"
        assert capture["raw_asset"] is None
        assert capture["parsed_asset"] is None
        assert capture["truncated_by"] == "pages"
        assert list((enabled_log_dir / "source_captures").glob("*")) == []

    @pytest.mark.asyncio
    async def test_fetch_capture_survives_a_cached_document_read_without_refetch(
        self,
        enabled_log_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_cache.clear()
        url = "https://example.org/cached-report"
        raw_body = b"<html><body>Captured tracker report.</body></html>"
        parsed_text = "The tracker reports 917 admissions this week.\n" * 18
        monkeypatch.setattr(classify, "_extract_page_text", lambda *_args, **_kwargs: PageExtraction(text=parsed_text))
        monkeypatch.setattr(classify, "render_inline_chart_data", lambda *_args: "")
        monkeypatch.setattr(classify, "unreadable_data_embed_providers", lambda *_args: [])
        session = FakeSession({url: FakeResponse(200, body=raw_body)})
        question_ctx = agentic_tools.question_ladder_context(session=session)

        fetched = await agentic_tools.fetch(url, question_topic="weekly admissions", ctx=question_ctx)
        assert fetched.status == "ok"
        capture_record = _raw_capture_record(enabled_log_dir)
        capture = capture_record["payload"]
        assert isinstance(capture, dict)
        assert (enabled_log_dir / str(capture["raw_asset"])).read_bytes() == raw_body
        assert (enabled_log_dir / str(capture["parsed_asset"])).read_text(encoding="utf-8") == parsed_text

        document = await agentic_tools.read_document(url, "How many admissions?", ctx=question_ctx)

        assert document.method == "digest_local"
        assert "917 admissions" in document.content_markdown
        assert session.requested == [url]
        assert len((enabled_log_dir / "raw_research_12345.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    @pytest.mark.asyncio
    async def test_word_document_capture_keeps_raw_and_parsed_content(
        self,
        enabled_log_dir: Path,
    ) -> None:
        run_cache.clear()
        url = "https://example.org/cached-report.docx"
        buffer = BytesIO()
        word_document = Document()
        word_document.add_paragraph("Quarterly revenue grew 3.2 percent.")
        word_document.save(buffer)
        raw_body = buffer.getvalue()
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        session = FakeSession({url: FakeResponse(200, body=raw_body, content_type=content_type)})
        question_ctx = agentic_tools.question_ladder_context(session=session)

        fetched = await agentic_tools.fetch(url, question_topic="revenue growth", ctx=question_ctx)

        assert fetched.status == "ok"
        capture = _raw_capture_record(enabled_log_dir)["payload"]
        assert isinstance(capture, dict)
        assert capture["content_type"] == content_type
        assert (enabled_log_dir / str(capture["raw_asset"])).read_bytes() == raw_body
        parsed_text = (enabled_log_dir / str(capture["parsed_asset"])).read_text(encoding="utf-8")
        assert "Quarterly revenue grew 3.2 percent." in parsed_text

        document = await agentic_tools.read_document(url, "What changed?", ctx=question_ctx)

        assert document.status == "ok"
        assert session.requested == [url]
        assert len((enabled_log_dir / "raw_research_12345.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    @pytest.mark.asyncio
    async def test_pdf_capture_keeps_raw_and_parsed_content(
        self,
        enabled_log_dir: Path,
    ) -> None:
        run_cache.clear()
        url = "https://example.org/cached-report.pdf"
        raw_body = build_text_pdf([["The report states that 917 admissions occurred this week."]])
        session = FakeSession({url: FakeResponse(200, body=raw_body, content_type="application/pdf")})
        question_ctx = agentic_tools.question_ladder_context(session=session)

        fetched = await agentic_tools.fetch(url, question_topic="admissions", ctx=question_ctx)

        assert fetched.status == "ok"
        capture = _raw_capture_record(enabled_log_dir)["payload"]
        assert isinstance(capture, dict)
        assert capture["content_type"] == "application/pdf"
        assert capture["page_count"] == 1
        assert capture["pages_read"] == 1
        assert (enabled_log_dir / str(capture["raw_asset"])).read_bytes() == raw_body
        parsed_text = (enabled_log_dir / str(capture["parsed_asset"])).read_text(encoding="utf-8")
        assert "917 admissions" in parsed_text

        document = await agentic_tools.read_document(url, "How many admissions?", ctx=question_ctx)

        assert document.status == "ok"
        assert session.requested == [url]
        assert len((enabled_log_dir / "raw_research_12345.jsonl").read_text(encoding="utf-8").splitlines()) == 1


class TestIngestSourceCaptureRecords:
    def test_validates_hashes_and_copies_sidecars_to_raw_archive(self, enabled_log_dir: Path, tmp_path: Path) -> None:
        raw_body = b"<html>source</html>"
        parsed_text = "source text"
        capture = record_source_capture(
            qid=808,
            source_url="https://example.org/source",
            retrieval_method="direct_http",
            content_type="text/html",
            parser_version="trafilatura 2.2.0",
            raw_body=raw_body,
            parsed_text=parsed_text,
        )
        assert capture is not None
        raw_record = _raw_capture_record(enabled_log_dir)

        ingest_source_capture_records([raw_record], enabled_log_dir, tmp_path / "archive" / "raw")

        archive_root = tmp_path / "archive" / "raw"
        assert (archive_root / str(capture["raw_asset"])).read_bytes() == raw_body
        assert (archive_root / str(capture["parsed_asset"])).read_text(encoding="utf-8") == parsed_text

    def test_rejects_path_traversal(self, enabled_log_dir: Path, tmp_path: Path) -> None:
        capture = record_source_capture(
            qid=808,
            source_url="https://example.org/source",
            retrieval_method="direct_http",
            content_type="text/html",
            parser_version="trafilatura 2.2.0",
            raw_body=b"<html>source</html>",
            parsed_text="source text",
        )
        assert capture is not None
        capture["raw_asset"] = "../outside.html"
        raw_record = _raw_capture_record(enabled_log_dir)
        raw_record["payload"] = capture

        with pytest.raises(SourceCaptureIngestError, match="portable"):
            ingest_source_capture_records([raw_record], enabled_log_dir, tmp_path / "archive")

    def test_rejects_corrupt_content_addressed_asset(self, enabled_log_dir: Path, tmp_path: Path) -> None:
        capture = record_source_capture(
            qid=808,
            source_url="https://example.org/source",
            retrieval_method="direct_http",
            content_type="text/html",
            parser_version="trafilatura 2.2.0",
            raw_body=b"<html>source</html>",
            parsed_text="source text",
        )
        assert capture is not None
        raw_asset = enabled_log_dir / str(capture["raw_asset"])
        raw_asset.write_bytes(b"corrupt-content-xxx")

        with pytest.raises(SourceCaptureIngestError, match="hash mismatch"):
            ingest_source_capture_records([_raw_capture_record(enabled_log_dir)], enabled_log_dir, tmp_path / "archive")

    def test_raw_research_harvest_copies_source_assets_into_raw_archive(
        self, enabled_log_dir: Path, tmp_path: Path
    ) -> None:
        raw_body = b"<html>source</html>"
        capture = record_source_capture(
            qid=808,
            source_url="https://example.org/source",
            retrieval_method="direct_http",
            content_type="text/html",
            parser_version="trafilatura 2.2.0",
            raw_body=raw_body,
            parsed_text="source text",
        )
        assert capture is not None

        archive_dir = tmp_path / "research_archive" / "raw"
        harvested = harvest_raw_logs_from_dir(enabled_log_dir, source_archive_dir=archive_dir)

        assert "12345" in harvested
        assert (archive_dir / str(capture["raw_asset"])).read_bytes() == raw_body

    def test_same_source_keeps_distinct_capture_identities_and_deduplicates_exact_repeats(
        self, enabled_log_dir: Path, tmp_path: Path
    ) -> None:
        source_url = "https://example.org/source?id=1"
        first_capture = record_source_capture(
            qid=808,
            source_url=source_url,
            retrieval_method="direct_http",
            content_type="text/html",
            parser_version="trafilatura 2.2.0",
            raw_body=b"<html>direct source</html>",
            parsed_text="direct source",
        )
        rendered_capture = record_source_capture(
            qid=808,
            source_url=source_url,
            retrieval_method="rendered",
            content_type="text/html",
            parser_version="trafilatura 2.2.0",
            raw_body=b"<html>rendered source</html>",
            parsed_text="rendered source",
        )
        repeated_capture = record_source_capture(
            qid=808,
            source_url=source_url,
            retrieval_method="rendered",
            content_type="text/html",
            parser_version="trafilatura 2.2.0",
            raw_body=b"<html>rendered source</html>",
            parsed_text="rendered source",
        )
        assert first_capture is not None
        assert rendered_capture is not None
        assert repeated_capture is not None

        archive_dir = tmp_path / "research_archive" / "raw"
        harvested = harvest_raw_logs_from_dir(enabled_log_dir, source_archive_dir=archive_dir)
        totals = merge_and_write(archive_dir, harvested)
        archived_records = [json.loads(line) for line in (archive_dir / "12345.jsonl").read_text().splitlines()]

        assert totals["12345"] == 2
        assert len(archived_records) == 2
        assert len({record["phase"] for record in archived_records}) == 2
        assert {record["payload"]["raw_sha256"] for record in archived_records} == {
            first_capture["raw_sha256"],
            rendered_capture["raw_sha256"],
        }
