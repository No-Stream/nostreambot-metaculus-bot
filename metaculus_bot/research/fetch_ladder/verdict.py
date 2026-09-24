"""What a caller makes of a body the ladder read: the one seat the two callers differ in.

The READ is shared and identical for both callers — the decode, the ARIA rewrite, the calibrated
two-pass extraction, the line-shape metric, the inline-chart read, the meta-refresh detection, the
pypdf parse, the link collection. What differs is the VERDICT: which branch a body's shape earns,
and whether what came back counts as content. The resolution-source fetcher publishes only text
that clears a 400-character chrome floor and the line-shape metric, because its section is
captioned primary grading evidence; the gap-fill v2 driver is handed any non-empty extraction and
judges it itself, and a short one escalates to the browser.

So :class:`LadderPolicy` carries ONE seat rather than a flag per difference, and the two
implementations below are the whole of it. Which verdict decides what, and why the floors belong
here rather than in the classifier: ``docs/architecture.md``, "What a verdict decides".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol

from metaculus_bot.constants import (
    DOCUMENT_DIGEST_TOP_K,
    RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS,
    RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS,
    RESOLUTION_SOURCE_JS_WALL_MIN_CHARS,
)
from metaculus_bot.research.document_text import PdfText, digest_pdf, disclosed_page_text, has_text_layer
from metaculus_bot.research.rendered_fetch import is_json_content_type
from metaculus_bot.research.resolution_fetch_result import (
    PDF_CONTENT_TYPES,
    FetchStatus,
    FetchStatusReason,
)
from metaculus_bot.research.source_documents import is_local_source

_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
_RAW_TEXT_CONTENT_TYPES = ("text/plain", "text/csv", "text/tab-separated-values", "text/tsv")
_HTML_STRIPPED_TEXT_CONTENT_TYPES = ("text/plain", "text/csv")
_IMAGE_CONTENT_TYPE_PREFIXES = ("image/",)
_IMAGE_MAGIC_BYTES = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM")

# Which branch of the classifier a body takes; only the gap-fill verdict routes to the two terminals.
BodyRoute = Literal["html", "text", "document", "source", "image", "invalid_image", "svg", "unsupported"]


def _is_raster_body(body: bytes) -> bool:
    stripped = body.lstrip()
    return stripped.startswith(_IMAGE_MAGIC_BYTES) or (
        len(stripped) >= 12 and stripped.startswith(b"RIFF") and stripped[8:12] == b"WEBP"
    )


def _is_svg(content_type: str, body: bytes) -> bool:
    return "image/svg+xml" in content_type or body.lstrip().lower().startswith(b"<svg")


def _is_strict_json_octet_stream(content_type: str, body: bytes) -> bool:
    if content_type.partition(";")[0].strip().lower() != "application/octet-stream":
        return False
    try:
        json.loads(body.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return True


def looks_like_js_wall(text: str) -> bool:
    """A 200 OK whose extracted text is shorter than the JS-wall threshold is a
    strong signal the page needs JS to render — Tier-2 candidate."""
    return len(text.strip()) < RESOLUTION_SOURCE_JS_WALL_MIN_CHARS


def looks_like_page_chrome(text: str) -> bool:
    """True when an extraction is too thin to be anything but chrome around the content.

    The floor is what the ``no_resolving_content`` verdict rests on; a named embed provider only
    says WHERE the content went (`embed_shell` vs `thin_page`). The 2026-09-02 calibration behind
    the 400-character elbow, and why the rule is ungated: ``docs/constants.md``
    "RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS".
    """
    return len(text.strip()) < RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS


def content_share(text: str) -> float:
    """Share of an extraction's characters that sit in content-shaped lines.

    Content is table rows (lines starting with ``|``, whatever their length: a price-history
    table is rows of 10-char cells) and lines of at least
    ``RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS``; every other line is chrome-shaped. Lines are
    stripped and blank ones dropped first. One pass over the extracted text, no second parse. What
    a line-shape rule buys and gives up: ``docs/constants.md``
    "RESOLUTION_SOURCE_CONTENT_SHARE_MIN".
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    total = sum(len(line) for line in lines)
    if total == 0:
        return 0.0
    content = sum(
        len(line) for line in lines if line.startswith("|") or len(line) >= RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS
    )
    return content / total


def _no_content_verdict(
    extracted: str | None, unreadable_embeds: list[str]
) -> tuple[FetchStatus, FetchStatusReason | None]:
    """Which withhold a 200 with no readable content earns, and why.

    Order is load-bearing and unchanged since the chrome floor generalised: a named routeless
    embed is the most specific thing we can say (`embed_shell` — the numbers exist and we have no
    route to them), the JS-wall floor keeps its own much lower threshold and its position in the
    middle so the chrome floor cannot swallow that population, and `thin_page` is everything else:
    under the floor, or over it on chrome alone (the line-shape metric in
    :func:`classify._extract_page_text`).
    """
    if unreadable_embeds:
        # Datawrapper is exempt from the embed scan, so a walled tracker still hops as `js_wall`.
        return "no_resolving_content", "embed_shell"
    if extracted is None or looks_like_js_wall(extracted):
        return "js_wall", None
    return "no_resolving_content", "thin_page"


def _pdf_unreadable_reason(pdf: PdfText) -> FetchStatusReason:
    """Why a document we read the bytes of yielded no text.

    ``encrypted`` / ``malformed`` come from the parse; ``no_text_layer`` is a document
    that parsed fine and carries images instead of text, which is the ONE shape a paid
    document read could still rescue.
    """
    if pdf.unreadable_reason == "encrypted":
        return "encrypted"
    if pdf.unreadable_reason == "malformed":
        return "malformed"
    return "no_text_layer"


@dataclass(frozen=True, slots=True)
class PageExtraction:
    """What the extractor policy decided for one HTML body (:func:`classify._extract_page_text`).

    ``text`` is what the verdict sees: the precision re-extraction when it rescued the page, else
    the default one (None when nothing extracted). ``chrome_metric_withheld`` marks a default
    extraction that cleared the chrome floor and failed the line-shape metric with no rescue, so
    the verdict withholds that text (the page itself still publishes when a chart block carries
    its numbers); ``precision_rescued`` marks a text that came from the fallback. Both ride the
    result into ``details["counts"]``.
    """

    text: str | None
    chrome_metric_withheld: bool = False
    precision_rescued: bool = False


@dataclass(frozen=True, slots=True)
class HtmlVerdict:
    """One caller's reading of a classified HTML body: the status, and the text it publishes."""

    status: FetchStatus
    status_reason: FetchStatusReason | None
    published_text: str


@dataclass(frozen=True, slots=True)
class DocumentVerdict:
    """One caller's reading of a parsed document: the digest block, or the whole text, or neither."""

    status: FetchStatus
    status_reason: FetchStatusReason | None
    text: str


class LadderVerdict(Protocol):
    """The seat :class:`LadderPolicy` carries: everything the two callers judge differently."""

    def unread_route(self, content_type: str) -> BodyRoute | None:
        """The branch this content type earns without reading the body, or None to read it."""
        ...

    def body_route(self, content_type: str, body: bytes) -> BodyRoute:
        """Which branch a body this caller has read takes."""
        ...

    def html(self, extraction: PageExtraction, *, chart_block: str, unreadable_embeds: list[str]) -> HtmlVerdict:
        """What this caller makes of one HTML body's extraction."""
        ...

    def document(self, pdf: PdfText, *, query: str, max_chars: int | None, source_url: str) -> DocumentVerdict:
        """What this caller serves from a parsed document. Runs inside the parse's own thread hop."""
        ...


@dataclass(frozen=True, slots=True)
class ResolutionSourceVerdict:
    """The fetcher's reading: content type decides the branch, and a page must clear both floors."""

    def unread_route(self, content_type: str) -> BodyRoute | None:
        """None always: this caller reads every 200 it is given, and routes on the header."""
        del content_type
        return None

    def body_route(self, content_type: str, body: bytes) -> BodyRoute:
        if is_local_source(body, content_type):
            return "source"
        if _is_svg(content_type, body):
            return "svg"
        if _is_raster_body(body):
            return "image"
        if _is_image_content_type(content_type):
            return "invalid_image"
        if _is_strict_json_octet_stream(content_type, body):
            return "text"
        if any(ct in content_type for ct in _HTML_CONTENT_TYPES):
            return "html"
        if (
            is_json_content_type(content_type)
            or _is_xml_content_type(content_type)
            or any(ct in content_type for ct in _RAW_TEXT_CONTENT_TYPES)
        ):
            return "text"
        # Everything else goes to the document branch, whose `%PDF-` sniff decides what these bytes are.
        return "document"

    def html(self, extraction: PageExtraction, *, chart_block: str, unreadable_embeds: list[str]) -> HtmlVerdict:
        extracted = extraction.text
        if (extraction.chrome_metric_withheld or looks_like_page_chrome(extracted or "")) and not chart_block:
            status, reason = _no_content_verdict(extracted, unreadable_embeds)
            return HtmlVerdict(status=status, status_reason=reason, published_text="")
        return HtmlVerdict(
            status="success",
            status_reason=None,
            published_text="" if extraction.chrome_metric_withheld else (extracted or ""),
        )

    def document(self, pdf: PdfText, *, query: str, max_chars: int | None, source_url: str) -> DocumentVerdict:
        if not has_text_layer(pdf):
            return DocumentVerdict("unreadable_document", _pdf_unreadable_reason(pdf), "")
        digest = digest_pdf(pdf, query=query, top_k=DOCUMENT_DIGEST_TOP_K, max_chars=max_chars, source_url=source_url)
        if not digest.passages:
            # A document read END TO END that discusses nothing the ask names (see the doc).
            return DocumentVerdict("no_resolving_content", "no_matching_passage", "")
        return DocumentVerdict("success", None, digest.block)


@dataclass(frozen=True, slots=True)
class GapFillVerdict:
    """The driver's reading: the bytes decide the branch, and any non-empty extraction is content."""

    def unread_route(self, content_type: str) -> BodyRoute | None:
        """Read every body so supported raster bytes can be retained for same-agent viewing."""
        del content_type
        return None

    def body_route(self, content_type: str, body: bytes) -> BodyRoute:
        if _is_pdf_content_type(content_type) or body.lstrip().startswith(b"%PDF-"):
            return "document"
        if is_local_source(body, content_type):
            return "source"
        if _is_svg(content_type, body):
            return "svg"
        if _is_raster_body(body):
            return "image"
        if _is_image_content_type(content_type):
            return "invalid_image"
        if any(ct in content_type for ct in _HTML_CONTENT_TYPES) or b"<html" in body.lower():
            return "html"
        if (
            is_json_content_type(content_type)
            or _is_xml_content_type(content_type)
            or any(ct in content_type for ct in _RAW_TEXT_CONTENT_TYPES)
            or not content_type
            or _is_strict_json_octet_stream(content_type, body)
        ):
            return "text"
        return "unsupported"

    def html(self, extraction: PageExtraction, *, chart_block: str, unreadable_embeds: list[str]) -> HtmlVerdict:
        del unreadable_embeds
        published = "" if extraction.chrome_metric_withheld else (extraction.text or "").strip()
        if published or chart_block:
            return HtmlVerdict(status="success", status_reason=None, published_text=published)
        # `js_wall` rather than a withhold: the browser is the next rung and the adapter says `empty`.
        return HtmlVerdict(status="js_wall", status_reason=None, published_text="")

    def document(self, pdf: PdfText, *, query: str, max_chars: int | None, source_url: str) -> DocumentVerdict:
        del query, max_chars
        if not has_text_layer(pdf):
            return DocumentVerdict("unreadable_document", _pdf_unreadable_reason(pdf), "")
        # Held rather than digested here: the driver's own ask arrives later, in `read_document`.
        return DocumentVerdict("success", None, disclosed_page_text(pdf))


def _is_pdf_content_type(content_type: str) -> bool:
    return any(token in content_type for token in PDF_CONTENT_TYPES)


def _is_xml_content_type(content_type: str) -> bool:
    media_type = content_type.partition(";")[0].strip().lower()
    return media_type in ("application/xml", "text/xml") or media_type.endswith("+xml")


def _is_image_content_type(content_type: str) -> bool:
    return content_type.startswith(_IMAGE_CONTENT_TYPE_PREFIXES)


RESOLUTION_SOURCE_VERDICT: LadderVerdict = ResolutionSourceVerdict()
GAP_FILL_VERDICT: LadderVerdict = GapFillVerdict()
