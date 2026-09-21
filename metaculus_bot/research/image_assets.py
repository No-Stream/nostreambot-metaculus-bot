"""Pure normalization for image evidence returned by research fetches."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Any, cast

from PIL import GifImagePlugin, Image, ImageOps, PngImagePlugin, WebPImagePlugin

from metaculus_bot.constants import (
    GAP_FILL_IMAGE_MAX_BYTES,
    GAP_FILL_IMAGE_MAX_EDGE,
    GAP_FILL_IMAGE_MAX_PIXELS,
    GAP_FILL_IMAGE_MAX_SOURCE_PIXELS,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
)

_SUPPORTED_FORMATS: frozenset[str] = frozenset({"PNG", "JPEG", "WEBP", "BMP", "GIF"})
_PIL_IMAGE_ERRORS: tuple[type[Exception], ...] = (OSError, SyntaxError, ValueError, EOFError)


class ImageReadError(ValueError):
    """An image body could not be normalized within the supported image contract."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ImageView:
    image_id: str
    source_url: str
    final_url: str
    source_sha256: str
    original_width: int
    original_height: int
    width: int
    height: int
    crop: tuple[int, int, int, int] | None
    png_bytes: bytes
    parent_page_urls: tuple[str, ...] = ()

    def metadata(self) -> dict[str, Any]:
        """Return JSON-ready image metadata without carrying the normalized bytes."""
        return {
            "image_id": self.image_id,
            "source_url": self.source_url,
            "final_url": self.final_url,
            "parent_page_urls": self.parent_page_urls,
            "source_sha256": self.source_sha256,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "width": self.width,
            "height": self.height,
            "crop": self.crop,
            "png_sha256": self.image_id,
            "byte_count": len(self.png_bytes),
        }


def _validate_crop(crop: tuple[int, int, int, int] | None, width: int, height: int) -> None:
    if crop is None:
        return
    if (
        not isinstance(crop, tuple)
        or len(crop) != 4
        or any(not isinstance(value, int) or isinstance(value, bool) for value in crop)
    ):
        raise ImageReadError("crop must contain exactly four integer coordinates")
    left, top, right, bottom = crop
    if left < 0 or right <= left or right > width:
        raise ImageReadError("crop must be a non-empty rectangle within the oriented source image")
    if top < 0 or bottom <= top or bottom > height:
        raise ImageReadError("crop must be a non-empty rectangle within the oriented source image")


def _output_dimensions(width: int, height: int) -> tuple[int, int]:
    scale = min(
        1.0,
        GAP_FILL_IMAGE_MAX_EDGE / max(width, height),
        math.sqrt(GAP_FILL_IMAGE_MAX_PIXELS / (width * height)),
    )
    return max(1, int(width * scale)), max(1, int(height * scale))


def _to_rgb(image: Image.Image) -> Image.Image:
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    if not has_alpha:
        result = image.convert("RGB")
        result.info.clear()
        return result

    rgba = image if image.mode == "RGBA" else image.convert("RGBA")
    owns_rgba = rgba is not image
    alpha = rgba.getchannel("A")
    white_background = Image.new("RGB", rgba.size, "white")
    try:
        white_background.paste(rgba, mask=alpha)
        white_background.info.clear()
        return white_background
    except (OSError, ValueError):
        white_background.close()
        raise
    finally:
        if owns_rgba:
            rgba.close()
        alpha.close()


def _open_image(body: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(body))
    except (Image.DecompressionBombError, *_PIL_IMAGE_ERRORS) as exc:
        raise ImageReadError("source image is corrupt, unsupported, or exceeds Pillow's safety limit") from exc


def _validate_source_image(source_image: Image.Image) -> None:
    if source_image.format not in _SUPPORTED_FORMATS:
        raise ImageReadError(f"unsupported source image format: {source_image.format or 'unknown'}")

    if source_image.format in {"GIF", "PNG", "WEBP"}:
        try:
            image_with_frames = cast(
                GifImagePlugin.GifImageFile | PngImagePlugin.PngImageFile | WebPImagePlugin.WebPImageFile,
                source_image,
            )
            frame_count = image_with_frames.n_frames
        except _PIL_IMAGE_ERRORS as exc:
            raise ImageReadError("source image is corrupt while checking animation frames") from exc
        if frame_count > 1:
            raise ImageReadError("animated GIF, PNG, and WebP images are unsupported")

    source_width, source_height = source_image.size
    if source_width < 1 or source_height < 1:
        raise ImageReadError("source image has invalid dimensions")
    if source_width * source_height > GAP_FILL_IMAGE_MAX_SOURCE_PIXELS:
        raise ImageReadError(f"source image exceeds the {GAP_FILL_IMAGE_MAX_SOURCE_PIXELS}-pixel decoding limit")


def _orient_image(source_image: Image.Image) -> Image.Image:
    try:
        source_image.load()
        return ImageOps.exif_transpose(source_image)
    except (Image.DecompressionBombError, *_PIL_IMAGE_ERRORS) as exc:
        raise ImageReadError("source image pixels are corrupt or cannot be decoded") from exc


def _encode_oriented_image(
    oriented_image: Image.Image, crop: tuple[int, int, int, int] | None
) -> tuple[bytes, int, int, int, int]:
    original_width, original_height = oriented_image.size
    _validate_crop(crop, original_width, original_height)

    working_image = oriented_image.crop(crop) if crop is not None else oriented_image
    owns_working_image = crop is not None
    try:
        output_width, output_height = _output_dimensions(*working_image.size)
        rgb_image = _to_rgb(working_image)
        try:
            if (output_width, output_height) != working_image.size:
                resized_image = rgb_image.resize(
                    (output_width, output_height),
                    resample=Image.Resampling.LANCZOS,
                )
                rgb_image.close()
                rgb_image = resized_image

            with io.BytesIO() as normalized_stream:
                rgb_image.save(normalized_stream, format="PNG", optimize=False)
                png_bytes = normalized_stream.getvalue()
        except (OSError, ValueError) as exc:
            raise ImageReadError("normalized PNG could not be encoded") from exc
        finally:
            rgb_image.close()
    except ImageReadError:
        raise
    except (Image.DecompressionBombError, *_PIL_IMAGE_ERRORS) as exc:
        raise ImageReadError("source image pixels could not be normalized") from exc
    finally:
        if owns_working_image:
            working_image.close()

    return png_bytes, original_width, original_height, output_width, output_height


def _validate_body(body: bytes) -> None:
    if not isinstance(body, bytes):
        raise ImageReadError("source image body must be bytes")
    if len(body) > RESOLUTION_SOURCE_MAX_RESPONSE_BYTES:
        raise ImageReadError(f"source image exceeds the {RESOLUTION_SOURCE_MAX_RESPONSE_BYTES}-byte input limit")


def _validate_png_size(png_bytes: bytes) -> None:
    if len(png_bytes) > GAP_FILL_IMAGE_MAX_BYTES:
        raise ImageReadError(
            f"normalized PNG exceeds the {GAP_FILL_IMAGE_MAX_BYTES}-byte output limit; provide a tighter explicit crop"
        )


def normalize_image(
    body: bytes,
    *,
    source_url: str,
    final_url: str,
    crop: tuple[int, int, int, int] | None = None,
    parent_page_urls: tuple[str, ...] = (),
) -> ImageView:
    """Normalize supported static image bytes into a bounded, metadata-free PNG.

    Crop coordinates are ``(left, top, right, bottom)`` in the image's displayed
    orientation after EXIF transforms. Images are never enlarged or automatically cropped.
    """
    _validate_body(body)
    source_sha256 = hashlib.sha256(body).hexdigest()
    source_image = _open_image(body)
    with source_image:
        _validate_source_image(source_image)
        oriented_image = _orient_image(source_image)

        close_oriented_image = oriented_image is not source_image
        try:
            png_bytes, original_width, original_height, output_width, output_height = _encode_oriented_image(
                oriented_image, crop
            )
        finally:
            if close_oriented_image:
                oriented_image.close()

    _validate_png_size(png_bytes)

    image_id = hashlib.sha256(png_bytes).hexdigest()
    return ImageView(
        image_id=image_id,
        source_url=source_url,
        final_url=final_url,
        source_sha256=source_sha256,
        original_width=original_width,
        original_height=original_height,
        width=output_width,
        height=output_height,
        crop=crop,
        png_bytes=png_bytes,
        parent_page_urls=tuple(dict.fromkeys(parent_page_urls)),
    )
