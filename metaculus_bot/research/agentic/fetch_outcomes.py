"""The agentic fetch result type and the platform refusal shared by its ladder adapter.

The shared fetch ladder owns HTTP response classification. This module keeps the driver-facing
``PlainFetchResult`` vocabulary, bounded HTML link extraction, the document-needed placeholder,
and the question-platform self-reference refusal that the gap-fill caller applies before the
shared ladder is entered.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from metaculus_bot.constants import MANTIC_HOST, METACULUS_HOST
from metaculus_bot.research.image_leads import ImageLead
from metaculus_bot.research.resolution_fetch_result import LocalKind
from metaculus_bot.research.resolution_url_scan import is_metaculus_self_ref

_FETCH_LINK_CAP = 25


@dataclass(slots=True)
class PlainFetchResult:
    status: str
    method: str
    text: str
    links: list[str]
    url: str
    content_type: str | None = None
    escalate_rendered: bool = False
    # Set only by a host's response, never by a refusal this ladder made itself (see the doc).
    http_status: int | None = None
    # A throttled body is withheld from ``text``; these fields carry only the existing marker's
    # evidence and the rung that obtained it.
    throttle_phrase: str | None = None
    throttle_chars: int | None = None
    throttle_method: str | None = None
    local_kind: LocalKind | None = None
    navigation_only: bool = False
    local_read_refused: bool = False
    image_leads: tuple[ImageLead, ...] = ()


class _LinkCollector(HTMLParser):
    def __init__(self, *, base_url: str, cap: int) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._cap = cap
        self._links: list[str] = []
        self._seen: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if len(self._links) >= self._cap or tag.lower() != "a":
            return
        href = None
        for name, value in attrs:
            if name.lower() == "href":
                href = value
                break
        if not href:
            return
        absolute = urljoin(self._base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            return
        if absolute in self._seen:
            return
        self._seen.add(absolute)
        self._links.append(absolute)

    @property
    def links(self) -> list[str]:
        return list(self._links)


def _extract_links_from_html(html: str, base_url: str) -> list[str]:
    parser = _LinkCollector(base_url=base_url, cap=_FETCH_LINK_CAP)
    parser.feed(html)
    parser.close()
    return parser.links


# Named rather than spelled at each site: three producers and two consumers branch on it.
DOCUMENT_NEEDED_METHOD = "document_needed"
_DOCUMENT_NEEDED_MSG = "This URL is a PDF — use read_document(url, ask) to read it."


def _document_needed_result(current_url: str, content_type: str) -> PlainFetchResult:
    """The escalate-to-a-document-read outcome, for the three rungs that can reach it.

    ``status="ok"`` with a method the tier map does not carry: nothing has been read yet, so
    this can never be stamped ``fetched``, but it is not a failure either — the fetch handler
    reads the method and escalates.
    """
    return PlainFetchResult(
        status="ok",
        method=DOCUMENT_NEEDED_METHOD,
        text=_DOCUMENT_NEEDED_MSG,
        links=[],
        url=current_url,
        content_type=content_type or None,
    )


_PLATFORM_FETCH_BLOCK_MSG = (
    "Metaculus and Mantic pages are already reflected in the question brief; "
    f"do not fetch {METACULUS_HOST} or {MANTIC_HOST} URLs."
)


def _fetch_plain_url_block(url: str) -> PlainFetchResult | None:
    """Reject a URL the plain rung must not dial, or None when it is fetchable.

    Runs on the caller-supplied URL and again on every redirect hop, so a 3xx cannot walk into a
    target the initial check would have refused. What both rungs it closes would otherwise reach,
    and why the paid reader has to honour it too: docs/agentic_gap_fill.md "Why the question
    platforms' own hosts are refused".
    """
    if is_metaculus_self_ref(url):
        return PlainFetchResult(
            status="blocked",
            method="plain",
            text=_PLATFORM_FETCH_BLOCK_MSG,
            links=[],
            url=url,
        )
    return None
