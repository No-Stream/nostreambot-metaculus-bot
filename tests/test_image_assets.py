"""Pure normalization for image assets extracted from cited web pages."""

from __future__ import annotations

import hashlib
import io
import struct
import zlib
from typing import Any, cast

import pytest
from PIL import Image, ImageDraw

from metaculus_bot.constants import (
    GAP_FILL_IMAGE_MAX_EDGE,
    GAP_FILL_IMAGE_MAX_PIXELS,
    GAP_FILL_IMAGE_MAX_SOURCE_PIXELS,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
)
from metaculus_bot.research.image_assets import ImageReadError, normalize_image


def _encode_image(image: Image.Image, image_format: str, **save_options: Any) -> bytes:
    output = io.BytesIO()
    image.save(output, format=image_format, **save_options)
    return output.getvalue()


def _large_png_header(width: int, height: int) -> bytes:
    """Make an openable PNG header whose pixel data would be truncated if decoded."""

    def _chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + chunk_data)
        return struct.pack(">I", len(chunk_data)) + chunk_type + chunk_data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    incomplete_scanline = zlib.compress(b"\x00\x00\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", incomplete_scanline) + _chunk(b"IEND", b"")


class TestNormalizeImage:
    @pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP", "BMP", "GIF"])
    def test_supported_static_formats_become_deterministic_png(self, image_format: str) -> None:
        image = Image.new("RGB", (12, 8), "white")
        ImageDraw.Draw(image).line((0, 7, 11, 0), fill="black", width=2)
        body = _encode_image(image, image_format)

        normalized = normalize_image(body, source_url="https://source.test/chart", final_url="https://cdn.test/chart")
        repeated = normalize_image(body, source_url="https://other.test/chart", final_url="https://cdn.test/chart")

        assert normalized.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        assert normalized.image_id == hashlib.sha256(normalized.png_bytes).hexdigest()
        assert normalized.png_bytes == repeated.png_bytes
        assert normalized.image_id == repeated.image_id
        assert normalized.source_sha256 == hashlib.sha256(body).hexdigest()
        assert (normalized.original_width, normalized.original_height) == (12, 8)
        assert (normalized.width, normalized.height, normalized.crop) == (12, 8, None)
        assert normalized.parent_page_urls == ()

    def test_exif_orientation_is_applied_before_dimensions_and_crop_validation(self) -> None:
        image = Image.new("RGB", (4, 6), "white")
        exif = Image.Exif()
        exif[274] = 6
        body = _encode_image(image, "JPEG", exif=exif)

        normalized = normalize_image(
            body,
            source_url="https://source.test/chart",
            final_url="https://cdn.test/chart",
            crop=(0, 0, 6, 4),
        )

        assert (normalized.original_width, normalized.original_height) == (6, 4)
        assert (normalized.width, normalized.height) == (6, 4)
        assert normalized.crop == (0, 0, 6, 4)

    def test_alpha_and_palette_transparency_are_composited_over_white(self) -> None:
        rgba_image = Image.new("RGBA", (3, 1))
        rgba_image.putdata([(0, 0, 0, 0), (255, 0, 0, 128), (0, 0, 255, 255)])
        rgba_body = _encode_image(rgba_image, "PNG")

        rgba_normalized = normalize_image(rgba_body, source_url="a", final_url="b")
        with Image.open(io.BytesIO(rgba_normalized.png_bytes)) as decoded_rgba:
            assert decoded_rgba.mode == "RGB"
            assert [decoded_rgba.getpixel((x, 0)) for x in range(3)] == [
                (255, 255, 255),
                (255, 127, 127),
                (0, 0, 255),
            ]

        palette_image = Image.new("P", (2, 1))
        palette_image.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
        palette_image.putdata([0, 1])
        palette_body = _encode_image(palette_image, "PNG", transparency=bytes([0, 255]))

        palette_normalized = normalize_image(palette_body, source_url="a", final_url="b")
        with Image.open(io.BytesIO(palette_normalized.png_bytes)) as decoded_palette:
            assert decoded_palette.mode == "RGB"
            assert [decoded_palette.getpixel((x, 0)) for x in range(2)] == [(255, 255, 255), (255, 0, 0)]

    def test_palette_images_use_high_quality_resampling_when_downscaled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        image = Image.new("P", (8, 1))
        image.putpalette([255, 0, 0, 0, 0, 255] + [0] * 762)
        image.putdata([0, 1] * 4)
        body = _encode_image(image, "PNG")
        monkeypatch.setattr("metaculus_bot.research.image_assets.GAP_FILL_IMAGE_MAX_EDGE", 2)

        normalized = normalize_image(body, source_url="a", final_url="b")

        with Image.open(io.BytesIO(normalized.png_bytes)) as decoded:
            colors = [cast(tuple[int, int, int], decoded.getpixel((x, 0))) for x in range(2)]
        assert any(red > 0 and blue > 0 for red, _green, blue in colors)

    def test_downscale_preserves_aspect_ratio_and_obeys_both_output_limits(self) -> None:
        image = Image.new("RGB", (3000, 1800), "white")
        ImageDraw.Draw(image).line((0, 1799, 2999, 0), fill="black", width=3)
        body = _encode_image(image, "PNG")

        normalized = normalize_image(body, source_url="a", final_url="b")

        assert (normalized.original_width, normalized.original_height) == (3000, 1800)
        assert max(normalized.width, normalized.height) <= GAP_FILL_IMAGE_MAX_EDGE
        assert normalized.width * normalized.height <= GAP_FILL_IMAGE_MAX_PIXELS
        assert normalized.width < normalized.original_width
        assert normalized.height < normalized.original_height
        assert normalized.width / normalized.height == pytest.approx(3000 / 1800, rel=0.002)

    def test_oversized_pixel_header_is_rejected_before_pixel_data_is_decoded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = _large_png_header(GAP_FILL_IMAGE_MAX_SOURCE_PIXELS + 1, 1)

        def fail_if_decoded(_image: Image.Image) -> None:
            raise AssertionError("source pixels were decoded before the dimension guard")

        monkeypatch.setattr(Image.Image, "load", fail_if_decoded)
        with pytest.raises(ImageReadError, match="pixel"):
            normalize_image(body, source_url="a", final_url="b")

    @pytest.mark.parametrize(
        "body",
        [
            b"<svg xmlns='http://www.w3.org/2000/svg'></svg>",
            b"not an image",
            b"\x89PNG\r\n\x1a\ncorrupt",
        ],
        ids=["svg", "unknown", "corrupt"],
    )
    def test_svg_unknown_and_corrupt_inputs_are_rejected(self, body: bytes) -> None:
        with pytest.raises(ImageReadError):
            normalize_image(body, source_url="a", final_url="b")

    @pytest.mark.parametrize("image_format", ["GIF", "PNG", "WEBP"])
    def test_animated_inputs_are_rejected(self, image_format: str) -> None:
        first_frame = Image.new("RGB", (4, 4), "white")
        second_frame = Image.new("RGB", (4, 4), "black")
        body = _encode_image(
            first_frame, image_format, save_all=True, append_images=[second_frame], duration=100, loop=0
        )

        with pytest.raises(ImageReadError, match="animated"):
            normalize_image(body, source_url="a", final_url="b")

    @pytest.mark.parametrize(
        "crop",
        [
            (True, 0, 1, 1),
            (0, 0, 1),
            (-1, 0, 2, 2),
            (2, 0, 2, 2),
            (0, 0, 5, 2),
            (0, 0, 2.5, 2),
        ],
        ids=["boolean-coordinate", "wrong-arity", "negative", "empty", "out-of-bounds", "non-integer"],
    )
    def test_invalid_crop_is_rejected(self, crop: tuple[int, ...]) -> None:
        body = _encode_image(Image.new("RGB", (4, 3), "white"), "PNG")

        with pytest.raises(ImageReadError, match="crop"):
            normalize_image(body, source_url="a", final_url="b", crop=crop)  # type: ignore[arg-type]

    def test_a_valid_crop_uses_explicit_source_coordinates_without_centering(self) -> None:
        image = Image.new("RGB", (6, 2), "white")
        for x, color in enumerate(("red", "red", "green", "green", "blue", "blue")):
            ImageDraw.Draw(image).line((x, 0, x, 1), fill=color)
        body = _encode_image(image, "PNG")

        normalized = normalize_image(body, source_url="a", final_url="b", crop=(2, 0, 4, 2))
        with Image.open(io.BytesIO(normalized.png_bytes)) as decoded:
            assert decoded.size == (2, 2)
            assert {decoded.getpixel((x, y)) for x in range(2) for y in range(2)} == {(0, 128, 0)}
        assert normalized.crop == (2, 0, 4, 2)

    def test_source_and_output_byte_caps_fail_with_actionable_crop_guidance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ImageReadError, match="byte"):
            normalize_image(b"x" * (RESOLUTION_SOURCE_MAX_RESPONSE_BYTES + 1), source_url="a", final_url="b")

        body = _encode_image(Image.new("RGB", (20, 20), "black"), "PNG")
        monkeypatch.setattr("metaculus_bot.research.image_assets.GAP_FILL_IMAGE_MAX_BYTES", 32)
        with pytest.raises(ImageReadError, match="crop"):
            normalize_image(body, source_url="a", final_url="b")

    def test_metadata_omits_png_bytes_and_carries_stable_hashes_and_dimensions(self) -> None:
        body = _encode_image(Image.new("RGB", (10, 5), "white"), "PNG")
        parent_page_urls = ("https://source.test/page", "https://source.test/other", "https://source.test/page")
        normalized = normalize_image(
            body,
            source_url="https://source.test",
            final_url="https://cdn.test",
            parent_page_urls=parent_page_urls,
        )

        metadata = normalized.metadata()

        assert "png_bytes" not in metadata
        assert metadata["image_id"] == normalized.image_id
        assert metadata["png_sha256"] == normalized.image_id
        assert metadata["byte_count"] == len(normalized.png_bytes)
        assert metadata["source_sha256"] == hashlib.sha256(body).hexdigest()
        assert metadata["source_url"] == "https://source.test"
        assert metadata["final_url"] == "https://cdn.test"
        assert normalized.parent_page_urls == ("https://source.test/page", "https://source.test/other")
        assert metadata["parent_page_urls"] == normalized.parent_page_urls
        assert metadata["original_width"] == 10
        assert metadata["original_height"] == 5
        assert metadata["width"] == 10
        assert metadata["height"] == 5
        assert metadata["crop"] is None
