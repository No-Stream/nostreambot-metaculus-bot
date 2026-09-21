"""Write normalized research images as content-addressed sidecars."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from metaculus_bot.research.image_assets import ImageView

logger = logging.getLogger(__name__)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_DIRECTORY = "media"


class ImageAssetIngestError(RuntimeError):
    """A research record references image media that is missing or inconsistent."""


def persist_image_views(
    image_views: Sequence[ImageView],
    image_sources: Mapping[str, Sequence[str]],
    *,
    output_dir: Path | str = "research_outputs",
    image_observations: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Write PNG sidecars once and return a byte-free, relative-path manifest.

    Image identity and the on-disk filename both use the SHA-256 digest of the normalized
    PNG bytes. If multiple source aliases normalize to the same image, they share one
    file and one manifest entry.
    """
    output_path = Path(output_dir)
    manifest_by_hash: dict[str, dict[str, Any]] = {}
    observations_by_hash: dict[str, dict[str, dict[str, Any]]] = {}

    for image_view in image_views:
        png_sha256 = hashlib.sha256(image_view.png_bytes).hexdigest()
        metadata = image_view.metadata()
        if image_view.image_id != png_sha256 or metadata["png_sha256"] != png_sha256:
            raise ValueError("ImageView identity must equal the SHA-256 of its normalized PNG bytes")

        asset_path = f"{_MEDIA_DIRECTORY}/{png_sha256}.png"
        output_path_for_image = output_path / asset_path
        if png_sha256 not in manifest_by_hash:
            if output_path_for_image.exists():
                existing_sha256 = hashlib.sha256(output_path_for_image.read_bytes()).hexdigest()
                if existing_sha256 != png_sha256:
                    raise ImageAssetIngestError(
                        f"Existing research image has a hash mismatch: {output_path_for_image} "
                        f"expected {png_sha256}, got {existing_sha256}"
                    )
            else:
                output_path_for_image.parent.mkdir(parents=True, exist_ok=True)
                output_path_for_image.write_bytes(image_view.png_bytes)

            manifest_by_hash[png_sha256] = {
                "image_id": image_view.image_id,
                "png_sha256": png_sha256,
                "byte_count": len(image_view.png_bytes),
                "representative_metadata": metadata,
                "source_urls": [],
                "parent_page_urls": [],
                "asset_path": asset_path,
                "observations": [],
            }
            observations_by_hash[png_sha256] = {}

        image_manifest = manifest_by_hash[png_sha256]
        source_urls = set(image_manifest["source_urls"])
        source_urls.update(image_sources.get(image_view.image_id, ()))
        parent_page_urls = set(image_manifest["parent_page_urls"])
        candidate_observations = list(image_observations.get(image_view.image_id, ())) if image_observations else []
        candidate_observations.append(metadata)
        for candidate in candidate_observations:
            observation = _validate_observation(candidate, png_sha256)
            observation_key = json.dumps(observation, sort_keys=True, separators=(",", ":"))
            observations_by_hash[png_sha256].setdefault(observation_key, observation)
            source_urls.update(url for url in (observation.get("source_url"), observation.get("final_url")) if url)
            parent_page_urls.update(observation.get("parent_page_urls", ()))

        image_manifest["source_urls"] = sorted(source_urls)
        image_manifest["parent_page_urls"] = sorted(parent_page_urls)

    for png_sha256, observations in observations_by_hash.items():
        manifest_by_hash[png_sha256]["observations"] = [observations[key] for key in sorted(observations)]

    return [manifest_by_hash[png_sha256] for png_sha256 in sorted(manifest_by_hash)]


def _validate_observation(observation: Mapping[str, Any], expected_png_sha256: str) -> dict[str, Any]:
    metadata = dict(observation)
    if "png_bytes" in metadata:
        raise ValueError("Image observations must not include normalized PNG bytes")
    if metadata.get("image_id") != expected_png_sha256 or metadata.get("png_sha256") != expected_png_sha256:
        raise ValueError("Image observation identity must equal the SHA-256 of its normalized PNG bytes")
    return metadata


def ingest_record_images(record: dict[str, Any], source_dir: Path, archive_dir: Path) -> None:
    """Copy and verify a record's PNG sidecars into the canonical research archive.

    ``asset_path`` is portable and relative to the JSONL's parent in an artifact. The
    same path remains relative to the archive root after ingestion, so the record itself
    does not need an artifact-specific or machine-specific rewrite.
    """
    gap_fill_v2 = record.get("gap_fill_v2")
    if not isinstance(gap_fill_v2, dict):
        return
    image_entries = gap_fill_v2.get("images")
    if image_entries is None:
        return
    if not isinstance(image_entries, list):
        raise ImageAssetIngestError("Invalid research image manifest: images must be a list")

    for image_entry in image_entries:
        portable_path, png_sha256 = _validate_image_entry(image_entry)
        source_image_path = source_dir.joinpath(*portable_path.parts)
        archive_image_path = archive_dir.joinpath(*portable_path.parts)
        _copy_verified_image(source_image_path, archive_image_path, png_sha256, portable_path.as_posix())


def _validate_image_entry(image_entry: object) -> tuple[PurePosixPath, str]:
    if not isinstance(image_entry, dict):
        raise ImageAssetIngestError("Invalid research image manifest: each image entry must be an object")

    asset_path_value = image_entry.get("asset_path")
    png_sha256 = image_entry.get("png_sha256")
    image_id = image_entry.get("image_id")
    if not isinstance(asset_path_value, str) or not isinstance(png_sha256, str) or not isinstance(image_id, str):
        raise ImageAssetIngestError(
            "Invalid research image manifest: asset_path, image_id, and png_sha256 are required"
        )
    if _SHA256_PATTERN.fullmatch(png_sha256) is None:
        raise ImageAssetIngestError(f"Invalid normalized PNG SHA-256 in research image manifest: {png_sha256!r}")
    if image_id != png_sha256:
        raise ImageAssetIngestError(
            f"Research image identity mismatch: image_id {image_id!r} does not equal png_sha256 {png_sha256!r}"
        )

    portable_path = PurePosixPath(asset_path_value)
    if not _is_portable_asset_path(portable_path, asset_path_value, png_sha256):
        raise ImageAssetIngestError(f"Invalid portable research image path: {asset_path_value!r}")
    return portable_path, png_sha256


def _is_portable_asset_path(portable_path: PurePosixPath, asset_path: str, png_sha256: str) -> bool:
    expected_parts = (_MEDIA_DIRECTORY, f"{png_sha256}.png")
    return (
        not portable_path.is_absolute()
        and not PureWindowsPath(asset_path).is_absolute()
        and ".." not in portable_path.parts
        and portable_path.as_posix() == asset_path
        and portable_path.parts == expected_parts
    )


def _copy_verified_image(
    source_image_path: Path,
    archive_image_path: Path,
    expected_sha256: str,
    relative_asset_path: str,
) -> None:
    try:
        png_bytes = source_image_path.read_bytes()
    except FileNotFoundError as exc:
        message = f"Missing referenced image media: {source_image_path} (record expects {relative_asset_path})"
        logger.error(message)
        raise ImageAssetIngestError(message) from exc

    actual_sha256 = hashlib.sha256(png_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        message = (
            f"Referenced image media hash mismatch: {source_image_path} expected {expected_sha256}, got {actual_sha256}"
        )
        logger.error(message)
        raise ImageAssetIngestError(message)

    if archive_image_path.exists():
        archived_sha256 = hashlib.sha256(archive_image_path.read_bytes()).hexdigest()
        if archived_sha256 != expected_sha256:
            message = (
                f"Canonical research image hash mismatch: {archive_image_path} expected {expected_sha256}, "
                f"got {archived_sha256}"
            )
            logger.error(message)
            raise ImageAssetIngestError(message)
        return

    archive_image_path.parent.mkdir(parents=True, exist_ok=True)
    archive_image_path.write_bytes(png_bytes)
