"""Verify and archive content-addressed source captures referenced by raw research logs."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from metaculus_bot.research.raw_log import SOURCE_CAPTURE_DIR, SOURCE_CAPTURE_PROVIDER, SOURCE_CAPTURE_SCHEMA_VERSION

logger = logging.getLogger(__name__)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RAW_CAPTURE_EXTENSIONS = ("bin", "html", "pdf", "txt")
_PARSED_CAPTURE_EXTENSIONS = ("txt",)


class SourceCaptureIngestError(RuntimeError):
    """A raw-research record references a missing or inconsistent source capture."""


def ingest_source_capture_records(records: Iterable[Mapping[str, Any]], source_dir: Path, archive_dir: Path) -> int:
    """Copy referenced source files after validating their portable path, size, and digest.

    Manifest records stay in the existing ``raw_research_<run_id>.jsonl`` log. Their asset
    paths are relative to that log's parent, which is why ``source_dir`` is the directory
    containing that file and the archive paths stay relative to the raw archive directory.
    """
    copied = 0
    for record in records:
        if record.get("provider") != SOURCE_CAPTURE_PROVIDER:
            continue
        payload = _capture_payload(record)
        for prefix in ("raw", "parsed"):
            if payload.get(f"{prefix}_capture_status") == "captured":
                _ingest_capture_asset(payload, prefix, source_dir, archive_dir)
                copied += 1
    return copied


def _capture_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise SourceCaptureIngestError("Invalid source capture manifest: payload must be an object")
    if record.get("truncated") or payload.get("schema_version") != SOURCE_CAPTURE_SCHEMA_VERSION:
        raise SourceCaptureIngestError("Invalid source capture manifest: truncated or unsupported schema")
    return payload


def _ingest_capture_asset(payload: Mapping[str, Any], prefix: str, source_dir: Path, archive_dir: Path) -> None:
    asset_path, digest = _validate_asset(payload, prefix)
    byte_count = payload[f"{prefix}_byte_count"]
    if type(byte_count) is not int or byte_count < 0:
        raise SourceCaptureIngestError(f"Invalid source capture {prefix} byte count")
    source_asset = source_dir.joinpath(*asset_path.parts)
    try:
        content = source_asset.read_bytes()
    except FileNotFoundError as exc:
        raise SourceCaptureIngestError(f"Missing referenced source capture: {source_asset}") from exc
    if len(content) != byte_count:
        raise SourceCaptureIngestError(
            f"Source capture byte count mismatch: {source_asset} expected {byte_count}, got {len(content)}"
        )
    actual_digest = hashlib.sha256(content).hexdigest()
    if actual_digest != digest:
        raise SourceCaptureIngestError(
            f"Source capture hash mismatch: {source_asset} expected {digest}, got {actual_digest}"
        )
    destination = archive_dir.joinpath(*asset_path.parts)
    _copy_verified_capture(content, destination, digest, asset_path.as_posix())


def _validate_asset(payload: Mapping[str, Any], prefix: str) -> tuple[PurePosixPath, str]:
    """The asset's relative path and digest, accepted only as exactly ``source_captures/<sha256>.<extension>``."""
    asset_path_value = payload.get(f"{prefix}_asset")
    digest = payload.get(f"{prefix}_sha256")
    if not isinstance(asset_path_value, str) or not isinstance(digest, str):
        raise SourceCaptureIngestError(f"Invalid source capture {prefix} asset reference")
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise SourceCaptureIngestError(f"Invalid source capture {prefix} SHA-256")
    extensions = _PARSED_CAPTURE_EXTENSIONS if prefix == "parsed" else _RAW_CAPTURE_EXTENSIONS
    if asset_path_value not in {f"{SOURCE_CAPTURE_DIR}/{digest}.{extension}" for extension in extensions}:
        raise SourceCaptureIngestError(f"Invalid portable source capture path: {asset_path_value!r}")
    return PurePosixPath(asset_path_value), digest


def _copy_verified_capture(content: bytes, destination: Path, expected_digest: str, relative_path: str) -> None:
    if destination.exists():
        existing_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if existing_digest != expected_digest:
            raise SourceCaptureIngestError(
                f"Archived source capture hash mismatch: {destination} expected {expected_digest}, got {existing_digest}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(content)
        os.replace(temporary_path, destination)
    except OSError as exc:
        logger.error("Could not archive source capture %s: %s", relative_path, exc)
        raise SourceCaptureIngestError(f"Could not archive source capture {relative_path}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
