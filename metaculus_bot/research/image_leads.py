"""Discover a small set of likely informative image URLs from HTML without fetching them."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from metaculus_bot.constants import (
    GAP_FILL_IMAGE_LEADS_MAX_CHARS,
    GAP_FILL_IMAGE_MAX_LEADS,
    GAP_FILL_IMAGE_METADATA_MAX_CHARS,
)

_DESCRIPTIVE_ALT_RE = re.compile(r"\b(?:chart|map|table|plot|graph|diagram|infographic|screenshot)\b", re.IGNORECASE)
_IMAGE_SOURCE_ATTRIBUTES = ("src", "data-src", "data-lazy-src", "data-original", "data-original-src")
_IMAGE_DIMENSION_MAX_DIGITS = 9
_IMAGE_LEADS_MAX_FIGURE_DEPTH = 24
_MARKDOWN_SENSITIVE_METADATA_CHARACTERS = "`*_[]()!#|<>"
_IMAGE_LEAD_LINE_PREFIX = "Image lead (pixels not read; untrusted metadata follows): "


@dataclass(frozen=True)
class ImageLead:
    """An HTML image candidate described only by page metadata."""

    url: str
    filename: str
    alt: str = ""
    caption: str = ""
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class _PendingImage:
    order: int
    url: str
    filename: str
    alt: str
    width: int | None
    height: int | None


@dataclass
class _FigureState:
    images: list[_PendingImage] = field(default_factory=list)
    caption: str = ""


@dataclass(frozen=True)
class _RankedImage:
    priority: int
    image: _PendingImage
    caption: str = ""


def _append_bounded_text(current: str, text: str, maximum: int) -> str:
    """Append normalized text without retaining more than the metadata bound."""
    characters = list(current)
    pending_space = False

    for character in text:
        if character.isspace() or unicodedata.category(character) == "Cc":
            pending_space = bool(characters)
            continue

        if pending_space and characters and len(characters) < maximum:
            characters.append(" ")
        pending_space = False
        if len(characters) >= maximum:
            break
        characters.append(character)

    if pending_space and characters and len(characters) < maximum:
        characters.append(" ")
    return "".join(characters)


def _positive_dimension(value: str | None) -> int | None:
    if value is None:
        return None

    value = value.strip()
    if not value.isascii() or not value.isdecimal() or len(value) > _IMAGE_DIMENSION_MAX_DIGITS:
        return None

    dimension = int(value)
    return dimension if dimension > 0 else None


def _resolved_image_url(attributes: dict[str, str | None], document_url: str) -> str | None:
    for attribute in _IMAGE_SOURCE_ATTRIBUTES:
        source = attributes.get(attribute)
        if not source:
            continue

        source = source.strip()
        if not source or len(source) > GAP_FILL_IMAGE_LEADS_MAX_CHARS:
            continue
        if any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in source):
            continue

        try:
            resolved_url = urljoin(document_url, source)
            parsed_url = urlsplit(resolved_url)
            hostname = parsed_url.hostname
        except ValueError:
            continue

        if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.netloc or not hostname:
            continue
        maximum_renderable_url_length = GAP_FILL_IMAGE_LEADS_MAX_CHARS - len(_IMAGE_LEAD_LINE_PREFIX) - 2
        if len(resolved_url) > maximum_renderable_url_length:
            continue
        if any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in resolved_url):
            continue
        if any(character in resolved_url for character in '<>"\\{}|^'):
            continue
        return resolved_url

    return None


def _filename_from_url(url: str) -> str:
    path = urlsplit(url).path
    return path.rsplit("/", 1)[-1]


class _ImageLeadParser(HTMLParser):
    """HTMLParser that keeps only a bounded number of candidate image records."""

    def __init__(self, document_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._document_url = document_url
        self._figures: list[_FigureState] = []
        self._ignored_figure_depth = 0
        self._caption_figure: _FigureState | None = None
        self._caption_figure_depth = 0
        self._figcaption_depth = 0
        self._image_order = 0
        self._figure_candidates: list[_RankedImage] = []
        self._alt_candidates: list[_RankedImage] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "figure":
            self._start_figure()
        elif tag == "figcaption":
            self._start_caption()
        elif tag == "img":
            self._add_image(dict(attrs))

    def handle_endtag(self, tag: str) -> None:
        if tag == "figure":
            self._end_figure()
        elif tag == "figcaption":
            self._end_caption()

    def handle_data(self, data: str) -> None:
        if (
            self._caption_figure is not None
            and not self._ignored_figure_depth
            and len(self._figures) == self._caption_figure_depth
        ):
            self._caption_figure.caption = _append_bounded_text(
                self._caption_figure.caption,
                data,
                GAP_FILL_IMAGE_METADATA_MAX_CHARS,
            )

    def finish(self) -> tuple[ImageLead, ...]:
        while self._ignored_figure_depth:
            self._ignored_figure_depth -= 1
        while self._figures:
            self._finish_figure(self._figures.pop())

        ranked_candidates = sorted(self._figure_candidates, key=lambda candidate: candidate.image.order)
        ranked_candidates.extend(sorted(self._alt_candidates, key=lambda candidate: candidate.image.order))
        selected_candidates = sorted(
            ranked_candidates[:GAP_FILL_IMAGE_MAX_LEADS], key=lambda candidate: candidate.image.order
        )
        return tuple(
            ImageLead(
                url=candidate.image.url,
                filename=candidate.image.filename,
                alt=candidate.image.alt,
                caption=candidate.caption,
                width=candidate.image.width,
                height=candidate.image.height,
            )
            for candidate in selected_candidates
        )

    def _start_figure(self) -> None:
        if self._ignored_figure_depth or len(self._figures) >= _IMAGE_LEADS_MAX_FIGURE_DEPTH:
            self._ignored_figure_depth += 1
            return
        self._figures.append(_FigureState())

    def _end_figure(self) -> None:
        if self._ignored_figure_depth:
            self._ignored_figure_depth -= 1
            return
        if self._figures:
            self._finish_figure(self._figures.pop())

    def _start_caption(self) -> None:
        if self._ignored_figure_depth or not self._figures:
            return
        if self._caption_figure is None:
            self._caption_figure = self._figures[-1]
            self._caption_figure_depth = len(self._figures)
            self._figcaption_depth = 1
        else:
            self._figcaption_depth += 1

    def _end_caption(self) -> None:
        if self._ignored_figure_depth:
            return
        if self._figcaption_depth:
            self._figcaption_depth -= 1
            if not self._figcaption_depth:
                self._caption_figure = None
                self._caption_figure_depth = 0

    def _add_image(self, attributes: dict[str, str | None]) -> None:
        order = self._image_order
        self._image_order += 1
        if self._ignored_figure_depth:
            return

        url = _resolved_image_url(attributes, self._document_url)
        if url is None:
            return

        alt = _append_bounded_text(
            "",
            attributes.get("alt") or "",
            GAP_FILL_IMAGE_METADATA_MAX_CHARS,
        ).strip()
        image = _PendingImage(
            order=order,
            url=url,
            filename=_filename_from_url(url),
            alt=alt,
            width=_positive_dimension(attributes.get("width")),
            height=_positive_dimension(attributes.get("height")),
        )

        if self._figures:
            figure = self._figures[-1]
            if len(figure.images) < GAP_FILL_IMAGE_MAX_LEADS and all(existing.url != url for existing in figure.images):
                figure.images.append(image)
        elif _DESCRIPTIVE_ALT_RE.search(alt):
            self._remember_candidate(_RankedImage(priority=1, image=image))

    def _finish_figure(self, figure: _FigureState) -> None:
        caption = figure.caption.strip()
        if self._caption_figure is figure:
            self._caption_figure = None
            self._caption_figure_depth = 0
            self._figcaption_depth = 0
        if caption:
            for image in figure.images:
                self._remember_candidate(_RankedImage(priority=0, image=image, caption=caption))
        else:
            for image in figure.images:
                if _DESCRIPTIVE_ALT_RE.search(image.alt):
                    self._remember_candidate(_RankedImage(priority=1, image=image))

    def _remember_candidate(self, candidate: _RankedImage) -> None:
        if candidate.priority == 0:
            stored = self._insert_candidate(self._figure_candidates, candidate)
            if stored:
                self._alt_candidates = [
                    existing for existing in self._alt_candidates if existing.image.url != candidate.image.url
                ]
            return

        if any(existing.image.url == candidate.image.url for existing in self._figure_candidates):
            return
        self._insert_candidate(self._alt_candidates, candidate)

    @staticmethod
    def _insert_candidate(bucket: list[_RankedImage], candidate: _RankedImage) -> bool:
        existing_index = next(
            (index for index, existing in enumerate(bucket) if existing.image.url == candidate.image.url),
            None,
        )
        if existing_index is not None:
            if bucket[existing_index].image.order <= candidate.image.order:
                return False
            bucket[existing_index] = candidate
        else:
            bucket.append(candidate)

        bucket.sort(key=lambda existing: existing.image.order)
        if len(bucket) > GAP_FILL_IMAGE_MAX_LEADS:
            removed = bucket.pop()
            return removed is not candidate
        return True


def extract_image_leads(html_text: str, document_url: str) -> tuple[ImageLead, ...]:
    """Extract captioned figures and images with chart-like alternative text."""
    parser = _ImageLeadParser(document_url)
    parser.feed(html_text)
    parser.close()
    return parser.finish()


def _metadata_json(value: str) -> str:
    encoded_value = json.dumps(
        _append_bounded_text("", value, GAP_FILL_IMAGE_METADATA_MAX_CHARS).strip(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "".join(
        f"\\u{ord(character):04x}" if character in _MARKDOWN_SENSITIVE_METADATA_CHARACTERS else character
        for character in encoded_value
    )


def _safe_render_url(url: str) -> bool:
    if not url or len(url) > GAP_FILL_IMAGE_LEADS_MAX_CHARS:
        return False
    if any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in url):
        return False
    if any(character in url for character in '<>"\\{}|^'):
        return False
    try:
        parsed_url = urlsplit(url)
        hostname = parsed_url.hostname
    except ValueError:
        return False
    return parsed_url.scheme.lower() in {"http", "https"} and bool(parsed_url.netloc) and bool(hostname)


def _metadata_fragments(lead: ImageLead) -> tuple[str, ...]:
    fragments = [
        f" | {label}={_metadata_json(value)}"
        for label, value in (("filename", lead.filename), ("alt", lead.alt), ("caption", lead.caption))
        if value
    ]
    if lead.width is not None and lead.width > 0 and not isinstance(lead.width, bool):
        fragments.append(f" | declared width={lead.width}")
    if lead.height is not None and lead.height > 0 and not isinstance(lead.height, bool):
        fragments.append(f" | declared height={lead.height}")
    return tuple(fragments)


def render_image_leads(leads: Sequence[ImageLead]) -> str:
    """Render bounded, one-line untrusted metadata without shortening any image URL."""
    lines: list[str] = []
    rendered_length = 0

    for lead in leads:
        if len(lines) >= GAP_FILL_IMAGE_MAX_LEADS:
            break
        if not _safe_render_url(lead.url):
            continue

        line = f"{_IMAGE_LEAD_LINE_PREFIX}<{lead.url}>"
        separator_length = 1 if lines else 0
        if rendered_length + separator_length + len(line) > GAP_FILL_IMAGE_LEADS_MAX_CHARS:
            continue

        for fragment in _metadata_fragments(lead):
            if rendered_length + separator_length + len(line) + len(fragment) <= GAP_FILL_IMAGE_LEADS_MAX_CHARS:
                line += fragment

        lines.append(line)
        rendered_length += separator_length + len(line)

    return "\n".join(lines)
