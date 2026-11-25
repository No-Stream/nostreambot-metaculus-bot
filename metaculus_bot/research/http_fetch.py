"""Shared aiohttp HTTP utilities for research providers.

Right-sized extraction (2026-07 resolution-source plan): only the genuinely
generic pieces live here — session construction, a size-capped body read, and
provider-agnostic HTML/URL helpers (the two embed scans below: Datawrapper
charts, shared so the resolution-source Tier-2 hop and any future
agentic-fetch integration can't drift on the route, and the routeless
data-embed providers a page can hide its numbers behind). Retry/backoff logic
stays provider-private (prediction_market's is JSON-API shaped; the
resolution-source fetcher deliberately does no retries).
"""

from __future__ import annotations

import asyncio
import codecs
import html as html_entities
import ipaddress
import logging
import re
import socket
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp
import aiohttp.abc
import aiohttp.resolver
import certifi

logger = logging.getLogger(__name__)

# Sentinel used only as default for AddressFamily; referenced in FilteringResolver.resolve.
_DEFAULT_FAMILY: socket.AddressFamily = socket.AF_INET


IpAddr = ipaddress.IPv4Address | ipaddress.IPv6Address


class FilteringResolver(aiohttp.abc.AbstractResolver):
    """DNS resolver that vets each resolved IP against a caller-supplied predicate.

    Motivation: :func:`is_public_http_url` resolves the target host and rejects
    on a private-IP hit, but ``aiohttp.TCPConnector`` performs its OWN
    resolution at connect time. A DNS-rebinding server that returned a public
    IP to the preflight and a private IP to the connect would slip past the
    guard (classic TOCTOU). This resolver runs at connect time — the same
    layer as aiohttp's DNS cache — so every IP actually dialed has been vetted
    and any redirect hop is re-resolved through the same filter.

    ``disallow(ip)`` returns True to REJECT an address. Callers pass e.g.
    ``_ip_is_disallowed`` from ``resolution_source.py``.

    If every resolved address is filtered out, ``resolve`` raises ``OSError``
    (mirroring how ``getaddrinfo``-based resolvers surface unusable results,
    so the fetch layer's existing except-clause catches it uniformly).
    """

    def __init__(
        self,
        *,
        disallow: Callable[[IpAddr], bool],
        inner: aiohttp.abc.AbstractResolver | None = None,
    ) -> None:
        self._disallow = disallow
        self._inner: aiohttp.abc.AbstractResolver = inner or aiohttp.resolver.ThreadedResolver()

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = _DEFAULT_FAMILY,
    ) -> list[Any]:
        raw = await self._inner.resolve(host, port, family)
        survivors: list[Any] = []
        for entry in raw:
            ip_str = entry["host"]
            try:
                parsed = ipaddress.ip_address(ip_str)
            except ValueError:
                # Non-parseable address string — treat conservatively as disallowed.
                continue
            if not self._disallow(parsed):
                survivors.append(entry)
        if not survivors:
            raise OSError(f"all resolved addresses disallowed for {host}")
        return survivors

    async def close(self) -> None:
        await self._inner.close()


# Shared redirect policy for the SSRF-guarded fetchers (resolution_source and
# research.agentic.tools) — both follow redirects manually so each hop can be
# re-guarded, and both must follow the SAME policy or the two fetchers drift
# for the same URL. Real-world sources chain at most 1-2 hops; 5 leaves slack
# for tracker redirects while keeping the per-hop re-guard cost bounded.
MAX_REDIRECTS: int = 5
REDIRECT_STATUSES: frozenset[int] = frozenset({301, 302, 303, 307, 308})


# Per-header and per-status-line byte cap for every session this module builds. aiohttp's
# default is 8,190 B, which is smaller than the Content-Security-Policy header real sources
# send (measured: who.int 8,765 B, visitwales.com 9,697 B) — and a header over the cap
# rejects the response before any body is read, so the page arrives as `error http=None`,
# indistinguishable from a host that never answered. 64 KiB is far above anything observed
# while still bounding what one response's headers can buffer.
_MAX_HEADER_BYTES: int = 65536


# Safari-like UA + full Accept / Accept-Language / Accept-Encoding.
# FINDINGS (resolution_source_probe): this exact header set recovered
# 6 extra sources vs Chrome-UA-only (38/50 vs 32/50).
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # The pair the measurement above was taken with, pinned as measured for the resolution-source
    # fetcher: widening what THIS header set asks for would change the negotiation on every fetch
    # the 38/50 figure covers, and no measurement covers the wider set. `br` and `zstd` became
    # DECODABLE when both decoders were declared in pyproject — needed for a body we never
    # negotiate at all, the Wayback rung's `id_` replay, which carries whatever encoding the
    # origin sent the archive's crawler. Pinning here does NOT mean nothing we send asks for
    # them: a session built without explicit headers (`build_session(headers=None)`, which is
    # what the prediction-market JSON-API providers use) inherits aiohttp's own default
    # `Accept-Encoding`, and `_gen_default_accept_encoding()` widens that to
    # `gzip, deflate, br, zstd` the moment both decoders are importable (aiohttp 3.14.3,
    # measured 2026-09-04). Both facts are pinned by
    # tests/test_http_fetch.py::TestUnadvertisedContentEncoding.
    "Accept-Encoding": "gzip, deflate",
}


def build_session(
    *,
    timeout_s: float,
    connector_limit: int = 20,
    headers: dict[str, str] | None = None,
    resolver: aiohttp.abc.AbstractResolver | None = None,
) -> aiohttp.ClientSession:
    """Construct a fresh aiohttp session with total + sock_read timeouts and a connection cap.

    ``headers=None`` (the default) adds no session-level headers — prediction_market's
    JSON-API calls rely on that; the resolution-source fetcher passes BROWSER_HEADERS. "No
    session-level headers" is not "no headers": each request then carries aiohttp's own
    defaults, whose ``Accept-Encoding`` advertises ``br`` and ``zstd`` as well as the
    ``gzip, deflate`` pair now that both decoders are installed (see ``BROWSER_HEADERS``).

    ``resolver=None`` (the default) uses aiohttp's built-in ThreadedResolver.
    Callers that need to vet resolved IPs (SSRF-sensitive fetchers) pass a
    :class:`FilteringResolver` so aiohttp's own connect-time DNS lookup goes
    through the same predicate as the preflight guard — closing the classic
    DNS-rebinding TOCTOU.

    TLS trust is pinned to certifi's bundle rather than left to whatever store the
    machine happens to carry. Measured 2026-09-03: trade.gov, a cited government source
    that fetched fine when it was archived, failed the handshake here with
    ``CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain`` against
    the default store and succeeded against certifi's — so which sources are reachable
    was a property of the machine, and a source lost that way is indistinguishable in
    telemetry from a dead host.

    The header-size caps are raised from aiohttp's 8,190-byte default because two
    corpus hosts send a Content-Security-Policy header larger than that (who.int
    8,765 B, visitwales.com 9,697 B) and aiohttp rejects the whole response before any
    body is read, landing as ``error http=None``. At 64 KiB who.int returns a readable
    200 and visitwales an honest 404.
    """
    timeout = aiohttp.ClientTimeout(total=timeout_s, sock_read=timeout_s)
    connector_kwargs: dict[str, Any] = {
        "limit": connector_limit,
        "ssl": ssl.create_default_context(cafile=certifi.where()),
    }
    if resolver is not None:
        connector_kwargs["resolver"] = resolver
    connector = aiohttp.TCPConnector(**connector_kwargs)
    return aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers=headers,
        max_line_size=_MAX_HEADER_BYTES,
        max_field_size=_MAX_HEADER_BYTES,
    )


# ---------------------------------------------------------------------------
# Per-host politeness gate (one in-flight request per netloc)
# ---------------------------------------------------------------------------
#
# The map has to outlive a single provider call. The Tier-1 fetcher used to build a
# fresh dict per call, so six questions running concurrently each got their own
# semaphore for the same host and hit it six times at once — the opposite of the
# politeness the semaphore exists to provide, and a plausible contributor to the 403s
# the escalation ladder is being built to route around.
#
# Scoped to the RUNNING event loop, not to the process: an `asyncio.Semaphore` binds to
# the loop that first blocks on it and raises from any other, so a second `asyncio.run`
# in the same process (a backtest question loop, the test suite) would otherwise crash
# on contention. Clearing on a loop change keeps "shared across every concurrent
# question", which is what one loop means here, without that hazard.
#
# The tradeoff, priced after the fact: sharing the map serializes same-host requests ACROSS
# the concurrent questions, inside a per-question wall that was not raised and that discards
# pages which already fetched when it fires — so a question that loses the queue can lose its
# whole section rather than one page. The acquire wait is deliberately unbounded; backlog task-14 "The 45 s resolution-source wall discards pages that already succeeded"
# holds both remedies (partial harvest, or a budget-bounded wait) as operator calls.
_HOST_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
_HOST_SEMAPHORE_LOOP: asyncio.AbstractEventLoop | None = None


def host_semaphores() -> dict[str, asyncio.Semaphore]:
    """The shared netloc -> ``Semaphore(1)`` map for the running event loop."""
    global _HOST_SEMAPHORE_LOOP  # noqa: PLW0603  # module-level cache of the loop the map is bound to
    loop = asyncio.get_running_loop()
    if loop is not _HOST_SEMAPHORE_LOOP:
        _HOST_SEMAPHORES.clear()
        _HOST_SEMAPHORE_LOOP = loop
    return _HOST_SEMAPHORES


def reset_host_semaphores() -> None:
    """Drop every cached semaphore. For tests, so one test's gate can't hold another's."""
    global _HOST_SEMAPHORE_LOOP  # noqa: PLW0603  # paired with host_semaphores' cache
    _HOST_SEMAPHORES.clear()
    _HOST_SEMAPHORE_LOOP = None


# ---------------------------------------------------------------------------
# Shared PDF-parse gate (at most two documents parsing at once, loop-wide)
# ---------------------------------------------------------------------------
#
# Every route that parses a fetched document contends here — the Tier-1 resolution-source
# rung and the gap-fill v2 local-document ladder — because the bound has to hold across
# them, not inside each. The rationale is the one `agentic/local_document.py` states for
# its own cap verbatim: pypdf decodes every content stream, so a parse is CPU-bound AND
# holds its body for the duration; a Tier-1 fan-out is up to RESOLUTION_SOURCE_MAX_URLS
# per question across DEFAULT_MAX_CONCURRENT_RESEARCH questions, so unbounded that is
# ~30 bodies of up to DOCUMENT_TEXT_PDF_MAX_BYTES plus their parse arenas — a MemoryError
# no soft-fail boundary catches. Two slots covers a real burst while bounding the peak,
# and measurement says more would not buy throughput anyway: pypdf is pure Python, so six
# concurrent parses of a 220-page document took 10.20 s against 1.66 s solo (6.13x on a
# 10-core machine) while starving the loop that every other provider's I/O runs on.
#
# Deliberately hardcoded rather than a constants.py entry: it is a property of pypdf and
# the default ThreadPoolExecutor's width, not a tuning knob anyone should reach for
# without re-measuring the numbers above.
_PDF_PARSE_SLOTS = 2
_PDF_PARSE_SEMAPHORE: asyncio.Semaphore | None = None
_PDF_PARSE_SEMAPHORE_LOOP: asyncio.AbstractEventLoop | None = None


def pdf_parse_semaphore() -> asyncio.Semaphore:
    """The shared ``Semaphore(2)`` bounding concurrent document parses on the running loop.

    Loop-scoped for the same reason :func:`host_semaphores` is: an ``asyncio.Semaphore``
    binds to the loop that first blocks on it and raises from any other, so a second
    ``asyncio.run`` in one process (a backtest's question loop, the test suite) would
    otherwise crash on contention.
    """
    global _PDF_PARSE_SEMAPHORE, _PDF_PARSE_SEMAPHORE_LOOP  # noqa: PLW0603  # module-level cache of the loop's gate
    loop = asyncio.get_running_loop()
    if _PDF_PARSE_SEMAPHORE is None or loop is not _PDF_PARSE_SEMAPHORE_LOOP:
        _PDF_PARSE_SEMAPHORE = asyncio.Semaphore(_PDF_PARSE_SLOTS)
        _PDF_PARSE_SEMAPHORE_LOOP = loop
    return _PDF_PARSE_SEMAPHORE


def reset_pdf_parse_semaphore() -> None:
    """Drop the cached parse gate. For tests, so one test's held slot can't gate another's."""
    global _PDF_PARSE_SEMAPHORE, _PDF_PARSE_SEMAPHORE_LOOP  # noqa: PLW0603  # paired with pdf_parse_semaphore's cache
    _PDF_PARSE_SEMAPHORE = None
    _PDF_PARSE_SEMAPHORE_LOOP = None


def semaphore_for_host(url: str, sems: dict[str, asyncio.Semaphore]) -> asyncio.Semaphore:
    """Get-or-create the ``Semaphore(1)`` gating requests to ``url``'s netloc.

    Takes the map explicitly so both callers keep their own scope: Tier-1 passes
    :func:`host_semaphores` (shared across concurrent questions) and the gap-fill v2
    loop passes its own module global.
    """
    host = urlparse(url).netloc
    sem = sems.get(host)
    if sem is None:
        sem = asyncio.Semaphore(1)
        sems[host] = sem
    return sem


_READ_CHUNK_BYTES = 65536


async def read_body_capped(resp: Any, *, max_bytes: int, label: str) -> bytes | None:
    """Read a response body incrementally, rejecting bodies over ``max_bytes``.

    Streams DECOMPRESSED bytes via ``resp.content.iter_chunked`` (aiohttp's
    ``DeflateBuffer`` feeds the same stream ``resp.read()`` consumes, so no
    chunked/gzip truncation) and aborts as soon as the running total exceeds
    the cap. The cap therefore bounds peak memory DURING the read — a huge or
    gzip-bombed response from an untrusted URL can't buffer fully before the
    guard fires.

    Returns None on an oversized body (logged at WARNING with ``label``).
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in resp.content.iter_chunked(_READ_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            logger.warning(f"{label} response too large ({total} bytes read > {max_bytes}); dropping")
            return None
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Text decoding (raw-body branches: JSON / text/plain / text/csv / CSV datasets)
# ---------------------------------------------------------------------------
#
# A blanket `body.decode("utf-8", errors="replace")` turns a UTF-16 or
# Windows-1252 body into mojibake that still type-checks as text, and the
# caller then renders `0�.�4�2�` to a forecaster as grading evidence. The
# response ALREADY carries the two facts needed to decode it properly — a BOM
# and/or a declared charset — so both are honored, and whatever survives is
# scored so a caller can refuse a body it could not decode at all.

_CHARSET_RE = re.compile(r"charset\s*=\s*\"?([\w.:+-]+)\"?", re.IGNORECASE)

# BOM -> codec. UTF-32-LE (`ff fe 00 00`) must be tested BEFORE UTF-16-LE
# (`ff fe`), which is a prefix of it. The `utf-16` / `utf-32` / `utf-8-sig`
# codecs consume the BOM themselves, so the decoded text carries no U+FEFF.
_BOM_CODECS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

# Above this share of undecodable characters, the decode failed rather than the
# text being slightly dirty, and the caller should refuse the body. The two
# regimes are far apart (measured on synthetic bodies; the cases are pinned in
# `tests/test_http_fetch.py`): a body decoded with the wrong codec runs ~0.5 —
# every second byte of BOM-less UTF-16
# ASCII is a NUL, and a UTF-16 BOM read as UTF-8 is two U+FFFDs up front — while
# real text carrying a few mis-encoded punctuation marks runs 0.03 on a 37-char
# line and far less on a real page. 0.10 sits in the empty gap between them, so
# it never costs us a page over one bad smart quote.
MAX_UNDECODABLE_CHAR_RATIO = 0.10


def _bom_codec(body: bytes) -> str | None:
    for bom, codec in _BOM_CODECS:
        if body.startswith(bom):
            return codec
    return None


def _declared_charset(content_type: str | None) -> str | None:
    if not content_type:
        return None
    match = _CHARSET_RE.search(content_type)
    return match.group(1) if match else None


def undecodable_char_ratio(text: str) -> float:
    """Share of ``text`` that carries no content because the decode failed.

    Counts U+FFFD (the replacement char `errors="replace"` substitutes for bytes
    the codec rejected) AND NUL. Both are needed: high bytes from an undeclared
    Windows-1252 body surface as U+FFFD, while BOM-less UTF-16 ASCII decodes as
    valid-but-meaningless NUL-interleaved text with zero replacement chars.
    """
    if not text:
        return 0.0
    return (text.count("�") + text.count("\x00")) / len(text)


def decode_text_body(body: bytes, content_type: str | None) -> tuple[str, float]:
    """Decode a raw body to ``(text, undecodable_char_ratio)``.

    Codec precedence: BOM (the body's own self-description, and the shape Excel
    exports CSV in), then the response's declared ``charset=``, then UTF-8.
    ``errors="replace"`` throughout — a partly-undecodable body still has to
    produce text, because the ratio is how the caller decides whether to keep it.
    An unknown declared charset falls back to UTF-8 rather than raising: a
    typo'd charset label is not a reason to lose the body.
    """
    codec = _bom_codec(body) or _declared_charset(content_type) or "utf-8"
    try:
        text = body.decode(codec, errors="replace")
    except LookupError:
        logger.info(f"unknown charset {codec!r} declared; decoding as utf-8")
        text = body.decode("utf-8", errors="replace")
    return text, undecodable_char_ratio(text)


# ---------------------------------------------------------------------------
# Datawrapper embed detection (resolution-source Tier-2 second hop)
# ---------------------------------------------------------------------------
#
# Poll-tracker pages routinely lock their daily series inside a Datawrapper
# iframe, which trafilatura drops at every setting — the resolving data never
# reaches the research bundle (qids 44858 / 44841, 2026-08-24 dossiers). The
# helpers here find the embeds in RAW page HTML and build the one URL that is
# safe to fetch.
#
# Route mechanism (live-verified against natesilver.net trackers 2026-08-25):
#
# - Page HTML embeds `datawrapper.dwcdn.net/<chart_id>/<version>/` with the
#   version PINNED at page-authoring time. The pinned version goes stale and
#   `<pinned>/dataset.csv` keeps serving its old snapshot with HTTP 200
#   (observed 5 and 14 MONTHS stale on the two real trackers). Reaching the
#   live version from a pinned one takes a 26+ hop chain of meta-refresh/JS
#   stubs that plain HTTP cannot follow. The versioned dataset route must
#   therefore NEVER be derived from page HTML.
# - `static.dwcdn.net/data/<chart_id>.csv` is version-free and always serves
#   the latest published dataset (Last-Modified refreshes on each republish;
#   observed hourly on live trackers). It is the URL behind the chart's own
#   "Get the data" affordance — the artifact resolution fine print names.

# Chart ids are 5 alphanumeric chars. Matches both host forms
# (`datawrapper.dwcdn.net/<id>/…`, `static.dwcdn.net/data/<id>.csv`) and
# tolerates JSON-escaped slashes (`\/`) — Substack pages carry the embed URL
# inside an HTML-escaped JSON `data-attrs` attribute.
_DATAWRAPPER_CHART_ID_RE = re.compile(
    r"(?:datawrapper|static)\.dwcdn\.net\\?/(?:data\\?/)?([A-Za-z0-9]{5})(?![A-Za-z0-9])"
)

# Title forms, in the two shapes observed in the wild:
# - JSON attrs (Substack `data-attrs`, HTML-escaped or plain):
#   `&quot;title&quot;:&quot;Do Americans …&quot;` — sits AFTER the URL. JSON
#   strings are double-quote-delimited only; a bare `'` inside the value
#   (e.g. "Trump's net approval …") is title text, not a terminator.
# - Datawrapper's own responsive embed: `<iframe title="…" … src="…">` —
#   sits BEFORE the URL, `=`-separated. The closing quote must match the
#   opening one, for the same apostrophe reason.
_DW_TITLE_JSON_RE = re.compile(r"(?:&quot;|\")title(?:&quot;|\")\s*:\s*(?:&quot;|\")(.{1,300}?)(?:&quot;|\")")
_DW_TITLE_ATTR_RE = re.compile(r"title\s*=\s*(?:\"([^\"]{1,300})\"|'([^']{1,300})')")

# Window sizes bracket the observed layouts: the Substack JSON title lands
# ~380 chars after the URL (two thumbnail URLs in between); an iframe's
# `title=` attribute sits within the same tag just before `src=`.
_DW_TITLE_FORWARD_WINDOW = 700
_DW_TITLE_BACKWARD_WINDOW = 300


@dataclass(frozen=True)
class DatawrapperChartRef:
    """A Datawrapper chart referenced by a fetched page, in document order."""

    chart_id: str
    title: str | None


def _within_one_tag(html_text: str, start: int, end: int) -> bool:
    """True when ``html_text[start:end]`` crosses no tag boundary.

    The proximity window alone cannot tell "this iframe's own title" from "a share
    button's title 200 chars earlier". A `>` or `<` between the two means they live on
    different elements, so the candidate is not this chart's title.
    """
    return "<" not in html_text[start:end] and ">" not in html_text[start:end]


def extract_datawrapper_charts(html_text: str) -> list[DatawrapperChartRef]:
    """Scan RAW page HTML for Datawrapper chart embeds.

    Must run on the raw HTML, not extracted main text — trafilatura emits
    neither iframe ``src`` attributes nor embed-script URLs at any setting
    (verified against the live tracker pages, 2026-08-24 dossier work).

    Returns first-seen-deduped refs in document order (tracker pages put the
    hero/resolving chart first). Titles are best-effort: the JSON-attrs form
    is searched forward of the URL (bounded at the next embed so a titleless
    chart can't steal its neighbour's), the iframe-attribute form backward.

    Both forms are additionally anchored to the URL's own TAG: a candidate with a
    tag boundary between it and the chart URL is rejected. Without that, the window
    alone let an unrelated `title="Share on X"` or an `og:title` on a neighbouring
    element render as the chart's identity inside the Tier-2 lead — the lead names
    the chart it is serving data for, so a borrowed title is a false claim about
    which series the forecaster is reading. Both real layouts sit inside one tag
    (the Substack `data-attrs` JSON blob, and an iframe's own `title=`), so the
    anchor costs nothing on them.
    """
    if not html_text:
        return []
    matches = list(_DATAWRAPPER_CHART_ID_RE.finditer(html_text))
    charts: list[DatawrapperChartRef] = []
    seen: set[str] = set()
    for i, m in enumerate(matches):
        chart_id = m.group(1)
        if chart_id in seen:
            continue
        seen.add(chart_id)

        forward_end = m.end() + _DW_TITLE_FORWARD_WINDOW
        if i + 1 < len(matches):
            forward_end = min(forward_end, matches[i + 1].start())
        raw_title: str | None = None
        json_match = _DW_TITLE_JSON_RE.search(html_text, m.end(), forward_end)
        if json_match is not None and _within_one_tag(html_text, m.end(), json_match.start()):
            raw_title = json_match.group(1)
        else:
            backward_start = max(0, m.start() - _DW_TITLE_BACKWARD_WINDOW)
            if i > 0:
                backward_start = max(backward_start, matches[i - 1].end())
            attr_matches = [
                match
                for match in _DW_TITLE_ATTR_RE.finditer(html_text, backward_start, m.start())
                if _within_one_tag(html_text, match.end(), m.start())
            ]
            if attr_matches:
                # The attr pattern alternates on quote style; exactly one group is set.
                raw_title = attr_matches[-1].group(1) or attr_matches[-1].group(2)

        title = html_entities.unescape(raw_title) if raw_title else None
        charts.append(DatawrapperChartRef(chart_id=chart_id, title=title))
    return charts


# ---------------------------------------------------------------------------
# Third-party data embeds we have NO route to (resolution-source Tier-1)
# ---------------------------------------------------------------------------
#
# The Datawrapper scan above exists because trafilatura drops embeds; these
# providers are the same failure with no second hop behind it. Flourish's own
# developer docs state the mechanism: "A Flourish embed is a placeholder plus a
# script that builds the chart in the browser. AI assistants and crawlers
# usually read a page's raw HTML once and don't run JavaScript, so to them the
# chart doesn't exist."
#
# qids 44554/44556 (2026-08-31 dossiers): racetothewh.com/senate/26 returned
# HTTP 200 and extracted 2.9k chars of forecast background, while the resolving
# Nebraska polling average lived in two Infogram iframes. The fetch reported an
# unqualified `success` and the section rendered under the "primary grading
# evidence" caveat with zero polling numbers in it, byte-identical across three
# questions. Naming the providers is what lets a caller either withhold an
# embed-only page (`no_resolving_content`) or tell the forecaster the numbers
# are not in the text it did get.
#
# Datawrapper is deliberately NOT here: it has a live-dataset route (the Tier-2
# hop), so its embeds are readable and its outcome is carried by the hop's own
# FetchStatus.
#
# Each provider matches either its embed-container marker (the class/element the
# loader script looks for) or its own host, with `\\?/` after the host so a
# JSON-escaped embed URL inside a `data-attrs` blob still matches — the same
# tolerance the Datawrapper id regex carries.
_DATA_EMBED_PROVIDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("infogram", re.compile(r"infogram-embed|infogram-async|(?:e\.)?infogram\.com\\?/", re.IGNORECASE)),
    ("flourish", re.compile(r"flourish-embed|(?:public\.)?flourish\.studio\\?/|flo\.uri\.sh\\?/", re.IGNORECASE)),
    (
        "tableau",
        re.compile(
            r"tableauPlaceholder|tableauViz|<tableau-viz|public\.tableau\.com\\?/"
            r"|tableauusercontent\.com\\?/|tableau\.com\\?/views\\?/",
            re.IGNORECASE,
        ),
    ),
)


def unreadable_data_embed_providers(html_text: str) -> list[str]:
    """Names of third-party data-embed providers referenced by ``html_text``.

    Runs on RAW page HTML for the same reason the Datawrapper scan does:
    trafilatura emits neither iframe ``src`` attributes nor embed-script URLs at
    any setting, so extracted text can never reveal that a chart was there.

    One entry per provider, ordered by where each first appears. Only providers
    with no fetch route of their own are reported (see the note above on Datawrapper's exemption), so a
    non-empty return means "this page displays data we cannot read".
    """
    if not html_text:
        return []
    hits: list[tuple[int, str]] = []
    for provider, pattern in _DATA_EMBED_PROVIDER_PATTERNS:
        match = pattern.search(html_text)
        if match is not None:
            hits.append((match.start(), provider))
    return [provider for _, provider in sorted(hits)]


# ---------------------------------------------------------------------------
# Meta-refresh redirects (a hop no HTTP status announces)
# ---------------------------------------------------------------------------
#
# cdc.gov's surveillance pages answer 200 with a 234-340 byte stub whose only content is
# `<meta http-equiv="refresh" content="0; url=...">` pointing at the real page. The
# manual redirect loop never sees it — there is no 3xx and no `Location` header — so the
# fetch classified the stub as a JS wall and the resolving numbers were never fetched.
#
# The target is returned RAW (not joined against a base) so the caller keeps ownership of
# resolution and of the SSRF re-guard every derived URL has to pass: this module has no
# business deciding what is safe to fetch.
# Bounded because the driver reads the list and a link farm would fill its window with hrefs.
PAGE_LINK_CAP = 25


class _LinkCollector(HTMLParser):
    """Absolute, de-duplicated http(s) hrefs of a page, up to ``cap``."""

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


def extract_page_links(html_text: str, base_url: str, *, cap: int = PAGE_LINK_CAP) -> list[str]:
    """The page's outbound links, resolved against ``base_url``: the DOCUMENT the DOM came from.

    Shared because both callers' HTML classification collects them off one parse; only the
    gap-fill preset keeps them (``LadderPolicy.collect_links``), since the driver navigates by
    them and the fetcher never does.
    """
    parser = _LinkCollector(base_url=base_url, cap=cap)
    parser.feed(html_text)
    parser.close()
    return parser.links


_META_TAG_RE = re.compile(r"<meta\s([^>]*)>", re.IGNORECASE)
_HTTP_EQUIV_REFRESH_RE = re.compile(r"http-equiv\s*=\s*[\"']?\s*refresh\b", re.IGNORECASE)
_CONTENT_ATTR_RE = re.compile(r"content\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))", re.IGNORECASE)
# `content="0; url=/real/page"`, `content='0;URL=…'`, and the unquoted-inner-value form.
# The delay is deliberately ignored: a long delay is still a redirect, and the pages that
# use one for a "you are being redirected" interstitial carry no resolving content either.
_REFRESH_URL_RE = re.compile(r"""\burl\s*=\s*['\"]?([^'\"\s;]+)""", re.IGNORECASE)


def meta_refresh_target(html_text: str) -> str | None:
    """The (possibly relative) URL a ``<meta http-equiv="refresh">`` tag points at, or None.

    Attribute order is not assumed — both ``http-equiv``-first and ``content``-first
    spellings occur — and HTML entities in the target are unescaped, since a query string
    written ``&amp;`` in markup has to be dialled as ``&``. The first such tag wins: a page
    with two conflicting refresh targets has a browser race in it, and taking the first is
    what a browser does.

    A ``;`` ends the target, which is the delimiter the unquoted ``content=0;url=/p`` form
    needs; the cost is a target carrying a literal semicolon (``;jsessionid=``), which no
    observed stub has.
    """
    if not html_text:
        return None
    for tag in _META_TAG_RE.finditer(html_text):
        attrs = tag.group(1)
        if not _HTTP_EQUIV_REFRESH_RE.search(attrs):
            continue
        content = _CONTENT_ATTR_RE.search(attrs)
        if content is None:
            continue
        # Unescaped BEFORE the url match, not after: markup writes the target's own
        # quotes as `&#x27;` and its query separators as `&amp;`, and matching first
        # would stop at the `;` ending the entity — returning a bare `'` for
        # `content="0; URL=&#x27;/page&#x27;"` and truncating `?a=1&amp;b=2` at the `&`.
        value = html_entities.unescape(content.group(1) or content.group(2) or content.group(3) or "")
        url_match = _REFRESH_URL_RE.search(value)
        if url_match is None:
            continue
        target = url_match.group(1).strip()
        if target:
            return target
    return None


# ---------------------------------------------------------------------------
# ARIA tables (a real table wearing a div costume)
# ---------------------------------------------------------------------------
#
# cdc.gov's outbreak stat blocks are `<div role="table">` / `role="row"` /
# `role="rowheader"` / `role="cell"`, which is valid accessible markup and completely
# invisible to trafilatura's table handling: it flattens the block to whichever values
# happened to sit inside a `<p>` and drops the rest, so the cyclosporiasis page rendered
# "17,180 / 2 / 48 plus the District of Columbia" with no labels and no hospitalization
# count at all (922, in a bare `<div role="cell">`). Rewritten to real table tags, the
# same page extracts "| Hospitalizations | 922 |".
#
# `rowgroup` is mapped too even though ARIA-wise it is optional scaffolding: the CDC block
# has one, and a `<tr>` whose parent is neither `table` nor `tbody` is dropped by lxml's
# HTML parser, so leaving it a div would defeat the whole rewrite.
_ARIA_ROLE_TAGS: dict[str, str] = {
    "table": "table",
    "grid": "table",
    "rowgroup": "tbody",
    "row": "tr",
    "rowheader": "th",
    "columnheader": "th",
    "cell": "td",
    "gridcell": "td",
}
# One `[^>]*` for the attribute region, deliberately not quote-aware: a `>` inside an
# attribute value truncates our view of that ONE tag (costing at most a rewrite we would
# have made), whereas a quote-aware form desynchronises for the rest of the document the
# moment a page carries an unbalanced quote — which real pages, and truncated captures of
# them, routinely do. Single quantifier, so no backtracking cliff either (the measured
# 3.4 s-at-200-KiB shape in `resolution_body_text` needed two).
_ARIA_TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)([^>]*)>")
_ARIA_ROLE_ATTR_RE = re.compile(r"""(?:^|\s)role\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE)
# Tags that never close, so they must not go on the nesting stack.
_VOID_HTML_TAGS: frozenset[str] = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)


def rewrite_aria_tables(html_text: str) -> str | None:
    """``html_text`` with ARIA-table roles rewritten to real table tags, or None.

    None means no role-bearing element was found, which lets the caller hand trafilatura
    the ORIGINAL bytes — so a page with no ARIA table extracts byte-identically to before
    this rung existed, and the encoding detection stays trafilatura's job.

    Nesting is tracked with an explicit stack rather than by pairing a tag with the next
    close of its name: every element in the CDC block is a ``div``, so "the next
    ``</div>``" is the innermost cell, not the table. A close tag matches the nearest
    open of the same name and discards whatever was left unclosed above it, which is how
    the unclosed ``<p>`` inside a cell is absorbed. Anything still open at the end is
    rewritten WITHOUT a close tag: a truncated capture (and plenty of live HTML) never
    closes its outer divs, and lxml's recovering parser closes them for us — whereas
    leaving the outer ``role="table"`` div alone would strand every rewritten ``<tr>``
    outside a table and lose the block entirely.
    """
    if not html_text:
        return None
    edits = _aria_table_edits(html_text)
    if not edits:
        return None
    pieces: list[str] = []
    cursor = 0
    for start, end, replacement in sorted(edits):
        pieces.append(html_text[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(html_text[cursor:])
    return "".join(pieces)


def _aria_role_tag(attrs: str) -> str | None:
    """The real tag name an element's ``role`` attribute maps to, if any."""
    role_match = _ARIA_ROLE_ATTR_RE.search(attrs)
    if role_match is None:
        return None
    role = (role_match.group(1) or role_match.group(2) or role_match.group(3) or "").strip().lower()
    return _ARIA_ROLE_TAGS.get(role)


def _aria_table_edits(html_text: str) -> list[tuple[int, int, str]]:
    """``(start, end, replacement)`` spans rewriting every ARIA-table tag pair."""
    # (tag name, mapped tag or None, open-tag start, open-tag end)
    stack: list[tuple[str, str | None, int, int]] = []
    edits: list[tuple[int, int, str]] = []
    for tag in _ARIA_TAG_RE.finditer(html_text):
        name = tag.group(2).lower()
        if tag.group(1):
            _close_aria_tag(stack, edits, name, tag)
            continue
        attrs = tag.group(3)
        if name in _VOID_HTML_TAGS or attrs.rstrip().endswith("/"):
            continue
        stack.append((name, _aria_role_tag(attrs), tag.start(), tag.end()))
    edits.extend(
        (open_start, open_end, f"<{mapped}>") for _name, mapped, open_start, open_end in stack if mapped is not None
    )
    return edits


def _close_aria_tag(
    stack: list[tuple[str, str | None, int, int]],
    edits: list[tuple[int, int, str]],
    name: str,
    tag: re.Match[str],
) -> None:
    """Pair a close tag with the nearest open of the same name, recording the rewrite.

    Whatever sat above the match was left unclosed (the ``<p>`` inside a CDC cell) and is
    dropped with it. A close tag matching nothing on the stack is ignored.
    """
    for depth in range(len(stack) - 1, -1, -1):
        if stack[depth][0] != name:
            continue
        _, mapped, open_start, open_end = stack[depth]
        if mapped is not None:
            edits.append((open_start, open_end, f"<{mapped}>"))
            edits.append((tag.start(), tag.end(), f"</{mapped}>"))
        del stack[depth:]
        return


_DATAWRAPPER_CHART_ID_SHAPE = re.compile(r"[A-Za-z0-9]{5}\Z")


def datawrapper_live_data_url(chart_id: str) -> str:
    """The version-free, always-latest dataset URL for a Datawrapper chart.

    This is the ONLY dataset route callers may fetch (see the mechanism note
    above): the versioned `datawrapper.dwcdn.net/<id>/<version>/dataset.csv`
    form serves the pinned — potentially months-stale — snapshot when the
    version comes from page HTML. Rejects ids that don't match the 5-char
    shape so a crafted page can't turn the template into path traversal.
    """
    if not _DATAWRAPPER_CHART_ID_SHAPE.match(chart_id):
        raise ValueError(f"not a Datawrapper chart id: {chart_id!r}")
    return f"https://static.dwcdn.net/data/{chart_id}.csv"


def parse_http_last_modified(value: str) -> datetime | None:
    """Parse an HTTP ``Last-Modified`` header into an aware UTC datetime.

    Returns None on a malformed value — callers treat unverifiable freshness
    the same as stale (the serve-live-or-nothing rule).
    """
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


# Enough to carry an upstream JSON error object or the title of an HTML error page.
ERROR_SNIPPET_BYTES: int = 2048


async def read_body_snippet(resp: Any, *, max_bytes: int = ERROR_SNIPPET_BYTES) -> str:
    """Decode at most ``max_bytes`` of a body, for a log line on a non-200.

    ``resp.text()`` reads and DECOMPRESSES the whole body before the caller slices
    it, so a CDN 429/502 serving a multi-megabyte HTML error page (or a gzip bomb)
    defeats the very memory ceiling the callers enforce on their success path.
    Reads bounded chunks and stops, leaving the rest of the body unread.
    """
    buf = bytearray()
    async for chunk in resp.content.iter_chunked(max_bytes):
        buf += chunk
        if len(buf) >= max_bytes:
            break
    return bytes(buf[:max_bytes]).decode("utf-8", errors="replace")
