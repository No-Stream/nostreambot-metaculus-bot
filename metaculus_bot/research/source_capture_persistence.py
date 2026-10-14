"""Verify and archive content-addressed source captures referenced by raw research logs."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

logger = logging.getLogger(__name__)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_CAPTURE_PROVIDER = "source_capture"
_SOURCE_CAPTURE_DIRECTORY = "source_captures"
_SOURCE_CAPTURE_EXTENSIONS = frozenset({"bin", "html", "pdf", "txt"})


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
        if record.get("provider") != _SOURCE_CAPTURE_PROVIDER:
            continue
        payload = _capture_payload(record)
        for prefix in ("raw", "parsed"):
            if payload.get(f"{prefix}_capture_status") != "captured":
                continue
            copied += _ingest_capture_asset(payload, prefix, source_dir, archive_dir)
    return copied


def _capture_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise SourceCaptureIngestError("Invalid source capture manifest: payload must be an object")
    if record.get("truncated") or payload.get("schema_version") != 1:
        raise SourceCaptureIngestError("Invalid source capture manifest: truncated or unsupported schema")
    return payload


def _ingest_capture_asset(payload: Mapping[str, Any], prefix: str, source_dir: Path, archive_dir: Path) -> int:
    asset_path = _validate_asset(payload, prefix)
    digest = str(payload[f"{prefix}_sha256"])
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
    return 1


def _validate_asset(payload: Mapping[str, Any], prefix: str) -> PurePosixPath:
    asset_path_value = payload.get(f"{prefix}_asset")
    digest = payload.get(f"{prefix}_sha256")
    if not isinstance(asset_path_value, str) or not isinstance(digest, str):
        raise SourceCaptureIngestError(f"Invalid source capture {prefix} asset reference")
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise SourceCaptureIngestError(f"Invalid source capture {prefix} SHA-256")
    asset_path = PurePosixPath(asset_path_value)
    expected_paths = {f"{_SOURCE_CAPTURE_DIRECTORY}/{digest}.{extension}" for extension in _SOURCE_CAPTURE_EXTENSIONS}
    portable = (
        asset_path_value in expected_paths
        and not asset_path.is_absolute()
        and not PureWindowsPath(asset_path_value).is_absolute()
        and asset_path.as_posix() == asset_path_value
    )
    parsed_extension_is_valid = prefix != "parsed" or asset_path.suffix == ".txt"
    if not portable or not parsed_extension_is_valid:
        raise SourceCaptureIngestError(f"Invalid portable source capture path: {asset_path_value!r}")
    return asset_path


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
