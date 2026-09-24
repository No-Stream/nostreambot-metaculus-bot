"""Tests for normalized image persistence and portable archive ingestion."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from metaculus_bot.research import agentic_gap_fill
from metaculus_bot.research.agentic.types import LoopTelemetry
from metaculus_bot.research.image_assets import ImageView
from metaculus_bot.research.image_persistence import (
    ImageAssetIngestError,
    ingest_record_images,
    persist_image_views,
)
from metaculus_bot.research.persistence import ResearchPersistenceWriter
from scripts import download_research, sync_all
from tests.pipeline_test_helpers import make_real_binary_question

# A valid 1 x 1 RGB PNG, kept as actual image bytes rather than an opaque fake payload.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jc9sAAAAASUVORK5CYII="
)
_PNG_SHA256 = hashlib.sha256(_PNG_BYTES).hexdigest()


def _image_view(
    source_url: str,
    final_url: str | None = None,
    parent_page_urls: tuple[str, ...] = (),
    *,
    source_sha256: str = "a" * 64,
    original_dimensions: tuple[int, int] = (1, 1),
    dimensions: tuple[int, int] = (1, 1),
    crop: tuple[int, int, int, int] | None = None,
) -> ImageView:
    return ImageView(
        image_id=_PNG_SHA256,
        source_url=source_url,
        final_url=final_url or source_url,
        source_sha256=source_sha256,
        original_width=original_dimensions[0],
        original_height=original_dimensions[1],
        width=dimensions[0],
        height=dimensions[1],
        crop=crop,
        png_bytes=_PNG_BYTES,
        parent_page_urls=parent_page_urls,
    )


def _media_record(asset_path: str, png_sha256: str = _PNG_SHA256) -> dict[str, Any]:
    return {
        "qid": 12345,
        "run_id": "100",
        "gap_fill_v2": {
            "transcript": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_reference", "image_id": _PNG_SHA256},
                        {"type": "text", "text": 'Image metadata: {"image_id": "' + _PNG_SHA256 + '"}'},
                    ],
                }
            ],
            "images": [
                {
                    "image_id": _PNG_SHA256,
                    "png_sha256": png_sha256,
                    "byte_count": len(_PNG_BYTES),
                    "source_urls": ["https://source.test/chart", "https://cdn.test/chart"],
                    "asset_path": asset_path,
                }
            ],
        },
    }


def _use_stored_artifact(monkeypatch, run_dir: Path) -> None:
    selection = SimpleNamespace(by_run={"100": {}}, expired=[])
    monkeypatch.setattr(download_research, "select_artifacts", lambda *_args, **_kwargs: selection)

    def _persisted_run_dirs(*_args: Any, **_kwargs: Any):
        yield "100", {}, run_dir

    monkeypatch.setattr(download_research, "persisted_run_dirs", _persisted_run_dirs)


class TestPersistImageViews:
    def test_repeated_png_hash_is_written_once_with_merged_aliases_and_portable_metadata(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "research_outputs"
        image_views = [
            _image_view(
                "https://source.test/chart",
                "https://cdn.test/chart",
                ("https://source.test/article-one",),
            ),
            _image_view(
                "https://mirror.test/chart",
                "https://cdn.test/chart",
                ("https://mirror.test/article-two",),
            ),
        ]

        manifest = persist_image_views(
            image_views,
            {
                _PNG_SHA256: [
                    "https://source.test/chart",
                    "https://cdn.test/chart",
                    "https://mirror.test/chart",
                ]
            },
            output_dir=images_dir,
        )

        assert len(manifest) == 1
        assert manifest[0]["image_id"] == _PNG_SHA256
        assert manifest[0]["png_sha256"] == _PNG_SHA256
        assert manifest[0]["asset_path"] == f"media/{_PNG_SHA256}.png"
        assert manifest[0]["source_urls"] == [
            "https://cdn.test/chart",
            "https://mirror.test/chart",
            "https://source.test/chart",
        ]
        assert manifest[0]["parent_page_urls"] == [
            "https://mirror.test/article-two",
            "https://source.test/article-one",
        ]
        assert set(manifest[0]["parent_page_urls"]).isdisjoint(manifest[0]["source_urls"])
        assert manifest[0]["byte_count"] == len(_PNG_BYTES)
        assert "png_bytes" not in manifest[0]
        assert str(images_dir) not in json.dumps(manifest)
        saved_images = list((images_dir / "media").glob("*.png"))
        assert saved_images == [images_dir / "media" / f"{_PNG_SHA256}.png"]
        assert saved_images[0].read_bytes() == _PNG_BYTES

    def test_same_png_keeps_each_distinct_source_and_crop_observation(self, tmp_path: Path) -> None:
        first_view = _image_view(
            "https://source-one.test/chart",
            "https://cdn.test/chart",
            ("https://source-one.test/article",),
            source_sha256="a" * 64,
            original_dimensions=(4, 4),
            dimensions=(1, 1),
            crop=(0, 0, 1, 1),
        )
        second_view = _image_view(
            "https://source-two.test/chart",
            "https://cdn.test/chart",
            ("https://source-two.test/article",),
            source_sha256="b" * 64,
            original_dimensions=(8, 6),
            dimensions=(1, 1),
            crop=(3, 2, 4, 3),
        )

        manifest = persist_image_views(
            [first_view],
            {_PNG_SHA256: [first_view.source_url, second_view.source_url, first_view.final_url]},
            output_dir=tmp_path,
            image_observations={_PNG_SHA256: [first_view.metadata(), first_view.metadata(), second_view.metadata()]},
        )

        image_manifest = manifest[0]
        assert image_manifest["image_id"] == _PNG_SHA256
        assert image_manifest["representative_metadata"] == first_view.metadata()
        assert len(image_manifest["observations"]) == 2
        assert [observation["source_sha256"] for observation in image_manifest["observations"]] == [
            "a" * 64,
            "b" * 64,
        ]
        assert [observation["crop"] for observation in image_manifest["observations"]] == [
            (0, 0, 1, 1),
            (3, 2, 4, 3),
        ]
        assert image_manifest["source_urls"] == [
            "https://cdn.test/chart",
            "https://source-one.test/chart",
            "https://source-two.test/chart",
        ]
        assert image_manifest["parent_page_urls"] == [
            "https://source-one.test/article",
            "https://source-two.test/article",
        ]
        assert "source_sha256" not in image_manifest
        assert "png_bytes" not in json.dumps(image_manifest)
        assert list((tmp_path / "media").glob("*.png")) == [tmp_path / "media" / f"{_PNG_SHA256}.png"]

    def test_loop_seam_persists_images_before_archiving_byte_free_transcript(self, tmp_path: Path, monkeypatch) -> None:
        output_dir = tmp_path / "research_outputs"
        image_view = _image_view(
            "https://source.test/chart",
            "https://cdn.test/chart",
            ("https://source.test/article",),
        )
        transcript = _media_record(f"media/{_PNG_SHA256}.png")["gap_fill_v2"]["transcript"]
        loop_result = SimpleNamespace(
            findings_markdown="findings",
            transcript=transcript,
            telemetry=LoopTelemetry(),
            ghost=None,
            ghost_context=None,
            image_views=[image_view],
            image_sources={_PNG_SHA256: ["https://source.test/chart", "https://cdn.test/chart"]},
            image_observations={_PNG_SHA256: [image_view.metadata()]},
        )
        captured: list[dict[str, Any]] = []

        @asynccontextmanager
        async def _fake_session():
            yield object()

        async def _fake_loop(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return loop_result

        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        monkeypatch.setattr(agentic_gap_fill.guard, "_get_session", _fake_session)
        monkeypatch.setattr(agentic_gap_fill, "build_system_prompt", lambda _today: "system")
        monkeypatch.setattr(agentic_gap_fill, "build_user_brief", lambda _question, _bundle: "brief")
        monkeypatch.setattr(agentic_gap_fill, "build_gap_fill_tools", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(agentic_gap_fill, "run_agentic_loop", _fake_loop)

        question = make_real_binary_question()
        findings = asyncio.run(
            agentic_gap_fill.run_gap_fill_v2(
                question,
                "bundle",
                is_benchmarking=False,
                archive_sink=captured.append,
                image_output_dir=output_dir,
            )
        )

        assert findings == "findings"
        assert len(captured) == 1
        payload = captured[0]
        assert payload["images"][0]["asset_path"] == f"media/{_PNG_SHA256}.png"
        assert payload["images"][0]["parent_page_urls"] == ["https://source.test/article"]
        assert "png_bytes" not in payload["images"][0]
        transcript_json = json.dumps(payload["transcript"])
        assert _PNG_BYTES not in transcript_json.encode()
        assert base64.b64encode(_PNG_BYTES).decode("ascii") not in transcript_json
        assert {block["type"] for block in transcript[0]["content"]} == {"image_reference", "text"}
        assert (output_dir / "media" / f"{_PNG_SHA256}.png").read_bytes() == _PNG_BYTES

        writer = ResearchPersistenceWriter("tournament", "metaculus", "t", "100")
        writer.record(
            qid=12345,
            page_url="https://www.metaculus.com/questions/12345/",
            question_text="Will the chart matter?",
            research_text=findings,
            providers_used=[],
            gap_fill_used=True,
            gap_fill_v2=payload,
        )
        jsonl_path = writer.flush(output_dir=str(output_dir))
        assert jsonl_path is not None
        archived = json.loads(jsonl_path.read_text())
        assert archived["gap_fill_v2"]["images"][0]["asset_path"] == f"media/{_PNG_SHA256}.png"
        assert archived["gap_fill_v2"]["images"][0]["parent_page_urls"] == ["https://source.test/article"]
        assert "png_bytes" not in json.dumps(archived)


class TestIngestRecordImages:
    def test_artifact_sidecar_is_hash_verified_copied_and_resolves_from_archive_root(self, tmp_path: Path) -> None:
        artifact_root = tmp_path / "artifact"
        artifact_research_dir = artifact_root / "research_outputs"
        artifact_media_path = artifact_research_dir / "media" / f"{_PNG_SHA256}.png"
        artifact_media_path.parent.mkdir(parents=True)
        artifact_media_path.write_bytes(_PNG_BYTES)
        archive_root = tmp_path / "research_archive"
        record = _media_record(f"media/{_PNG_SHA256}.png")

        ingest_record_images(record, artifact_research_dir, archive_root)

        canonical_path = archive_root / record["gap_fill_v2"]["images"][0]["asset_path"]
        assert canonical_path.read_bytes() == _PNG_BYTES
        assert Path(record["gap_fill_v2"]["images"][0]["asset_path"]).is_absolute() is False
        assert canonical_path.relative_to(archive_root).as_posix() == f"media/{_PNG_SHA256}.png"

    def test_repeated_ingestion_is_idempotent_and_does_not_rewrite_the_jsonl_record(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        artifact_research_dir = tmp_path / "artifact" / "research_outputs"
        sidecar = artifact_research_dir / "media" / f"{_PNG_SHA256}.png"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_bytes(_PNG_BYTES)
        original = json.dumps(_media_record(f"media/{_PNG_SHA256}.png"), separators=(",", ":")) + "\n"
        jsonl_path = artifact_research_dir / "research_100.jsonl"
        jsonl_path.write_text(original)
        _use_stored_artifact(monkeypatch, tmp_path / "artifact")
        archive_root = tmp_path / "research_archive"

        records = download_research.download_research_artifacts(
            "example/repo",
            0,
            store_dir=tmp_path / "store",
            from_store=True,
            output_dir=archive_root,
        )
        first_copy = (archive_root / "media" / f"{_PNG_SHA256}.png").read_bytes()
        download_research.download_research_artifacts(
            "example/repo",
            0,
            store_dir=tmp_path / "store",
            from_store=True,
            output_dir=archive_root,
        )

        assert records == [_media_record(f"media/{_PNG_SHA256}.png")]
        assert first_copy == _PNG_BYTES
        assert jsonl_path.read_text() == original

    def test_legacy_records_are_unchanged_and_archive_rebuild_is_byte_idempotent(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        artifact_research_dir = tmp_path / "artifact" / "research_outputs"
        artifact_research_dir.mkdir(parents=True)
        legacy_record = {"qid": 456, "run_id": "100", "research_text": "old archive payload"}
        original_jsonl = '{ "qid" : 456, "run_id" : "100", "research_text" : "old archive payload" }\n'
        jsonl_path = artifact_research_dir / "research_100.jsonl"
        jsonl_path.write_text(original_jsonl)
        _use_stored_artifact(monkeypatch, tmp_path / "artifact")
        archive_root = tmp_path / "research_archive"

        records = download_research.download_research_artifacts(
            "example/repo",
            0,
            store_dir=tmp_path / "store",
            from_store=True,
            output_dir=archive_root,
        )
        download_research.build_archive(records, archive_root)
        by_qid_path = archive_root / "by_qid" / "456.jsonl"
        first_archive_bytes = by_qid_path.read_bytes()
        download_research.build_archive(records, archive_root)

        assert records == [legacy_record]
        assert "gap_fill_v2" not in records[0]
        assert jsonl_path.read_text() == original_jsonl
        assert by_qid_path.read_bytes() == first_archive_bytes

    def test_missing_referenced_media_is_reported_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(ImageAssetIngestError, match="Missing referenced image media"):
            ingest_record_images(
                _media_record(f"media/{_PNG_SHA256}.png"),
                tmp_path / "artifact" / "research_outputs",
                tmp_path / "research_archive",
            )

    def test_hash_mismatch_is_reported_and_not_copied(self, tmp_path: Path) -> None:
        artifact_research_dir = tmp_path / "artifact" / "research_outputs"
        sidecar = artifact_research_dir / "media" / f"{_PNG_SHA256}.png"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_bytes(b"not the normalized png")
        archive_root = tmp_path / "research_archive"

        with pytest.raises(ImageAssetIngestError, match="hash mismatch"):
            ingest_record_images(_media_record(f"media/{_PNG_SHA256}.png"), artifact_research_dir, archive_root)

        assert not (archive_root / "media" / f"{_PNG_SHA256}.png").exists()

    def test_download_phase_ingests_from_each_jsonl_parent(self, tmp_path: Path, monkeypatch) -> None:
        artifact_research_dir = tmp_path / "artifact" / "research_outputs"
        sidecar = artifact_research_dir / "media" / f"{_PNG_SHA256}.png"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_bytes(_PNG_BYTES)
        (artifact_research_dir / "research_100.jsonl").write_text(
            json.dumps(_media_record(f"media/{_PNG_SHA256}.png")) + "\n"
        )
        _use_stored_artifact(monkeypatch, tmp_path / "artifact")
        archive_root = tmp_path / "research_archive"

        records = download_research.download_research_artifacts(
            "example/repo", 0, store_dir=tmp_path / "store", from_store=True, output_dir=archive_root
        )

        assert records[0]["qid"] == 12345
        assert (archive_root / "media" / f"{_PNG_SHA256}.png").read_bytes() == _PNG_BYTES

    def test_sync_all_ingests_media_into_its_configured_archive(self, tmp_path: Path, monkeypatch) -> None:
        run_dir = tmp_path / "artifact"
        artifact_research_dir = run_dir / "research_outputs"
        sidecar = artifact_research_dir / "media" / f"{_PNG_SHA256}.png"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_bytes(_PNG_BYTES)
        (artifact_research_dir / "research_100.jsonl").write_text(
            json.dumps(_media_record(f"media/{_PNG_SHA256}.png")) + "\n"
        )
        selection = SimpleNamespace(by_run={"100": {}}, expired=[], total_artifacts=1)
        monkeypatch.setattr(sync_all, "select_artifacts", lambda *_args, **_kwargs: selection)

        def _persisted_run_dirs(*_args: Any, **_kwargs: Any):
            yield "100", {"name": "research-100"}, run_dir

        monkeypatch.setattr(sync_all, "persisted_run_dirs", _persisted_run_dirs)
        monkeypatch.setattr(sync_all, "resolve_workflow_map", lambda *_args, **_kwargs: {})
        archive_root = tmp_path / "research_archive"

        summary = sync_all.run_sync(
            "example/repo",
            0,
            research_dir=archive_root,
            backfill_dir=tmp_path / "backfill",
            telemetry_dir=tmp_path / "telemetry",
            raw_dir=tmp_path / "raw",
            store_dir=tmp_path / "store",
            from_store=True,
        )

        assert summary.research_questions == 1
        assert (archive_root / "media" / f"{_PNG_SHA256}.png").read_bytes() == _PNG_BYTES
        assert (
            json.loads((archive_root / "latest" / "12345.json").read_text())["gap_fill_v2"]["images"][0]["asset_path"]
            == f"media/{_PNG_SHA256}.png"
        )
