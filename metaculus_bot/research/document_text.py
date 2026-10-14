"""Deterministic PDF text extraction and lexical passage selection.

Stdlib and pypdf only, with no I/O of its own: bytes in, structured text out. Both
consumers — the Tier-1 resolution-source fetcher, which drops every PDF unread, and the
gap-fill v2 research loop, which sends every PDF to a paid Gemini ``url_context`` read —
already hold the bytes by the time they need the text, so acquiring them is the caller's
job and keeping this module free of network and model calls is what lets either one use
it without inheriting the other's dependencies.

Why deterministic extraction comes first: measured 2026-09-03, local pypdf pulled 833,450
chars out of a 6.7 MB 220-page PDF in 5.3 s and the passage the driver was looking for was
in it, while the paid reader returned nothing for the same file. So the order is acquire
the text locally, select the relevant passages locally, and spend a model call only on a
document we genuinely cannot read (``unreadable_reason``, or an empty
``has_text_layer``, which together separate "we could not parse this" from "this is a scan
with no text layer at all").

``is_pdf_body`` is the shared ``%PDF-`` magic check used by the fetch ladder's document
classifier. Keeping the check here leaves the pure text foundation independent of the ladder's
transport and caller adapters.

Extraction is CPU-bound (pypdf parses and decodes every content stream), so an async
caller must run ``extract_pdf_text`` in a thread — ``asyncio.to_thread`` — never inline on
the event loop.
"""

from __future__ import annotations

import io
from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import log
from time import monotonic
from typing import Literal

from pypdf import PdfReader
from pypdf import filters as pypdf_filters
from pypdf.errors import DependencyError, PyPdfError
from pypdf.generic import Destination

from metaculus_bot.research.source_evidence import SourceExcerpt

# Everything the reader and page-text paths raise on a mangled PDF, enumerated rather than
# blanket-caught. pypdf's own tree hangs off PyPdfError (PdfReadError, PdfStreamError,
# EmptyFileError, FileNotDecryptedError, LimitReachedError, ParseError), but DependencyError
# sits outside it, and the parser also raises plain stdlib types from inside the same paths
# (counted in pypdf 6.16.2 across _reader / _page / _doc_common / generic / _utils / filters:
# ValueError x23, TypeError x12, NotImplementedError x6, KeyError x5, IndexError x3, and the
# two Unicode errors under UnicodeError). The handful of bare `raise Exception` sites in
# pypdf are deliberately NOT in here: those are pypdf bugs and swallowing one would hide a
# real defect. Four of the five guarded blocks below wrap exactly one pypdf call with none
# of this module's own logic inside it, so a KeyError from our code still crashes the way it
# should. _read_outline is the exception: its guard spans _walk_outline, twenty lines of our
# own traversal. That is sound today because every raise site inside the walk is pypdf's (the
# Destination attribute reads and get_destination_page_number), while the walk's own
# statements — the explicit stack, the isinstance dispatch, the entry-cap comparison — can
# raise none of these types. The maintenance rule that keeps it sound: a subscript, an index
# or arithmetic added to that walk belongs OUTSIDE the try, or a bug in our own code starts
# degrading to an empty outline instead of crashing.
_PDF_ERRORS: tuple[type[Exception], ...] = (
    PyPdfError,
    DependencyError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    NotImplementedError,
    UnicodeError,
)

# pypdf decodes a page's whole content stream into memory before tokenising it, and the cost
# is linear in the DECODED size, at roughly 0.5-0.8 s of CPU and 35-55 MB of RSS per decoded
# megabyte: a 27 KB file carrying 9 MB of compressed-away operators took 6.95 s here and
# yielded 1.2 M chars of "text". None of the module's other bounds catch that. The caller's
# body-size limit caps the COMPRESSED bytes, max_pages does not help when one page carries the
# whole stream, and max_seconds is only checked between pages (see _read_pages), so the one
# bound that reaches inside a single page is pypdf's own decoded-output cap, which its
# `filters.decompress` docstring documents as ours to lower. A breach arrives as
# LimitReachedError, a PyPdfError already in _PDF_ERRORS, so an over-large page costs that page
# ("") exactly like a decode failure and its siblings still read.
#
# 8 MB bounds one page's parse arena to ~300 MB at that slope, and sits far above any real
# content stream: the 6.7 MB / 220-page document this module was measured on yielded 833,450
# chars in total, ~3.8 KB a page. pypdf's MAX_DECLARED_STREAM_LENGTH is deliberately left
# alone: it caps the declared /Length, which is ENCODED bytes — the wrong quantity, and an
# encoded cap below the caller's body limit would refuse streams whose decoded size is fine.
#
# Set once at import and process-global, which is what pypdf's knobs are; every consumer of
# this module wants the same bound, and nothing else in the bot reads PDFs.
PDF_MAX_DECODED_STREAM_BYTES = 8_000_000

pypdf_filters.ZLIB_MAX_OUTPUT_LENGTH = PDF_MAX_DECODED_STREAM_BYTES
pypdf_filters.LZW_MAX_OUTPUT_LENGTH = PDF_MAX_DECODED_STREAM_BYTES
pypdf_filters.RUN_LENGTH_MAX_OUTPUT_LENGTH = PDF_MAX_DECODED_STREAM_BYTES
pypdf_filters.MAX_ARRAY_BASED_STREAM_OUTPUT_LENGTH = PDF_MAX_DECODED_STREAM_BYTES

_PDF_MAGIC = b"%PDF-"

# A page carrying fewer than this many non-whitespace chars is running-header noise (a page
# number, a footer stamp), not a text layer, so a scan whose OCR-free pages emit a stray
# glyph still reads as "no text layer" rather than as a readable document.
TEXT_LAYER_MIN_CHARS = 40

# Okapi BM25, hand-rolled because the whole retrieval step is ~40 lines and a search dependency
# for that is not worth the supply-chain surface. k1 is standard; b is 0, which turns length
# normalisation OFF. BM25's b rewards a short unit for the same term frequency, which is right
# when the units are documents of genuinely different size and wrong here: our windows are
# uniform by construction (``_segment_windows`` caps each at ``window_chars``), so the length
# differences left are our own segmentation residue — a page break ends a window wherever it
# falls, so a 220-page PDF yields roughly one short tail window per page (a heading, a footer
# stamp, a one-line page). At b=0.75 those fragments outrank the paragraphs that carry the
# figure: a one-token heading page scored 0.3245 against 0.1268 for the 75-token paragraph
# holding the same term once, and took the top digest slot.
BM25_K1 = 1.5
BM25_B = 0.0

# Query terms carrying no retrieval signal, dropped before scoring. Two families, one reason:
# IDF here is computed over the windows of the SINGLE document being read, so a term present in
# most windows is driven to ~0 and a verbose query ends up scored by whichever of its words
# happen to be rare in this document. The query is the question text plus its resolution
# criteria, so the rare words are the procedural vocabulary every criteria carries ("resolves",
# "according", "published") and the URL scaffolding of its cited sources ("https", "www"), while
# the topical terms — the ones that say what the document must be about — are the common ones.
# Measured: four criteria-boilerplate pages took all four top slots on a 31-page report with the
# row carrying the resolving figure fifth, and on a real EIA outlook PDF the windows holding it
# ranked 152nd. Only structural and procedural words are listed; domain nouns a criteria happens
# to use ("total", "revised", "surveillance") stay in, because they are what a table row
# carrying the answer says. Quoted block rather than a set literal (hence the SIM905 exemption
# below): a 150-word literal formats to one word per line, where the block keeps each family on
# its own line and a diff shows which family a word joined.
_QUERY_STOPWORDS = frozenset(
    """
    a an the and or but if then than so as of to in on at by for from with within without
    into over under between before after during about above below up down out off again
    is are was were be been being am do does did doing has have had having will would shall
    should can could may might must not no nor only also very more most other others such
    any all each both few some same too own per via
    this that these those it its they them their there here which who whom whose what when
    where why how
    question questions resolve resolves resolved resolution resolutions criteria criterion
    metaculus
    according publish published publishes publishing report reported reporting reports
    source sources page pages date dates data
    yes maybe
    https http www com org gov net html htm pdf aspx index php
    """.split()  # noqa: SIM905
)

# The default in effect: nothing passes ``window_chars`` today, so this is the window size the
# digest actually uses. It mirrors DOCUMENT_DIGEST_WINDOW_CHARS in constants.py, which is the
# knob a caller would pass, and a test pins the two equal so the default can never drift from
# the configured one.
DEFAULT_WINDOW_CHARS = 600

# Outline entries the digest renders before collapsing the rest into a count. A 220-page
# report's bookmark tree runs to hundreds of entries, which would crowd out the passages
# the digest exists to carry.
DIGEST_MAX_OUTLINE_ENTRIES = 25

# One wording for "we read this document and it does not discuss the ask", used with the
# query appended and, on a blank query, alone.
_NO_MATCH_SENTENCE = "No passage in the text read matched the query"

UnreadableReason = Literal["", "encrypted", "malformed"]
TruncationCause = Literal["", "pages", "seconds"]


@dataclass(frozen=True)
class PdfText:
    """Per-page text of one PDF, plus what stopped the read.

    ``pages`` holds one entry per page actually read, stripped of surrounding whitespace,
    with ``""`` for a page that carries no text layer (or whose own content stream failed to
    decode — a partial read keeps the pages that worked). ``pages_read`` is therefore
    ``len(pages)`` and ``page_count`` is what the document declares, so
    ``pages_read < page_count`` plus ``truncated_by`` says which bound bit.

    ``unreadable_reason`` distinguishes the two ways a document yields nothing: ``"scanned"``
    is not one of its values because a scan parses fine and is reported as pages present and
    all empty (``has_text_layer`` is False), while ``"encrypted"`` / ``"malformed"`` mean we
    never got a page list at all.
    """

    page_count: int
    pages_read: int
    pages: tuple[str, ...]
    truncated_by: TruncationCause
    outline: tuple[tuple[str, int], ...]
    unreadable_reason: UnreadableReason = ""


@dataclass(frozen=True)
class Passage:
    """One scored window of a document, addressed by char offset into the joined text.

    ``text`` is exactly ``source[start:end]``. ``page`` is the 1-based page the passage sits
    on, or None when the caller supplied no page offsets; where they were supplied a window
    never straddles a page break, so the number is exact rather than a best guess.
    """

    score: float
    start: int
    end: int
    text: str
    page: int | None


@dataclass(frozen=True)
class DocumentDigest:
    """A rendered digest block plus how many passages it carries.

    The count rides along because it is the only thing that says whether a digest ANSWERED
    the ask: zero passages means the document does not discuss what was asked, which reads
    identically to a successful read in the block itself (the "no passage matched" sentence)
    and is what the gap-fill v2 loop's ``AGENTIC_FETCH_LOCAL_DOC`` marker reports.
    """

    block: str
    passages: int
    outcome: Literal["answered", "no_match", "partial_no_match", "unreadable", "empty_text"] = "answered"
    scope: str = "complete_document"
    evidence_passages: tuple[SourceExcerpt, ...] = ()


def is_pdf_body(body: bytes) -> bool:
    """True when these bytes are a PDF, by the ``%PDF-`` header the format mandates.

    Tolerates leading whitespace, which some servers prepend, the way pypdf's own header
    scan does.
    """
    return body.lstrip().startswith(_PDF_MAGIC)


def extract_pdf_text(body: bytes, *, max_pages: int, max_seconds: float) -> PdfText:
    """Per-page text of ``body``, bounded by page count and a page-boundary clock, never raising.

    Both bounds are the caller's: a resolution source we cannot read in ``max_seconds``
    costs the research phase more than it is worth, and ``max_pages`` bounds the pathological
    case (a 4,000-page appendix dump) independently of how fast the machine is. A bad PDF
    comes back as a ``PdfText`` carrying ``unreadable_reason`` rather than an exception,
    because both callers are soft-failing research providers and a malformed source document
    must never take a forecast run down.

    CPU-bound; run it in a thread from async code.
    """
    reader, reason = _open_reader(body)
    if reader is None:
        return PdfText(page_count=0, pages_read=0, pages=(), truncated_by="", outline=(), unreadable_reason=reason)

    page_count = _page_count(reader)
    if page_count is None:
        return PdfText(page_count=0, pages_read=0, pages=(), truncated_by="", outline=(), unreadable_reason="malformed")

    outline = _read_outline(reader)
    pages, truncated_by = _read_pages(reader, page_count=page_count, max_pages=max_pages, max_seconds=max_seconds)
    return PdfText(
        page_count=page_count,
        pages_read=len(pages),
        pages=pages,
        truncated_by=truncated_by,
        outline=outline,
        unreadable_reason="",
    )


def has_text_layer(pdf: PdfText) -> bool:
    """True when at least one read page carries real text rather than header noise.

    False on a scan, which is the signal a caller uses to escalate to a paid document read:
    that is the one case where a model is the only way to get the content.
    """
    return any(_visible_char_count(page) >= TEXT_LAYER_MIN_CHARS for page in pdf.pages)


def truncation_note(pdf: PdfText) -> str:
    """What stopped the read, as one clause to append to a line, or ``""`` when nothing did.

    Public because two renderers have to say this the same way. The digest header appends it
    to its own page counts here; gap-fill v2's ``pdf_local`` fetch result serves the joined
    page text with no header at all, and a partial read that does not say so reads as the whole
    document — a driver then reports an absence over pages nobody read. Sharing one derivation
    is what keeps the two wordings from drifting apart.

    The leading ``"; "`` is part of the clause, so a caller rendering it as its own line brings
    its own lead-in; ``pages_read`` rides inside the wording because the bound that bit is only
    meaningful next to how far the read got.
    """
    return {
        "pages": f"; stopped at the {pdf.pages_read}-page read cap",
        "seconds": f"; stopped after {pdf.pages_read} pages on the extraction time budget",
    }.get(pdf.truncated_by, "")


def joined_page_text(pdf: PdfText) -> tuple[str, tuple[int, ...]]:
    """The read pages as one string, plus each page's start offset in it.

    Pages join on a blank line so a page boundary is also a paragraph boundary for
    ``select_passages``, and the returned offsets are what its ``page_breaks`` argument
    expects, so page attribution never has to be reconstructed by a caller.
    """
    breaks: list[int] = []
    cursor = 0
    for page in pdf.pages:
        breaks.append(cursor)
        cursor += len(page) + 2  # the "\n\n" separator below
    return "\n\n".join(pdf.pages), tuple(breaks)


def disclosed_page_text(pdf: PdfText) -> str:
    """The read pages as one string, led by a note when they are not the whole document.

    Both writers of a PDF's flat text go through here, because that text is served with no header
    of its own — a gap-fill ``pdf_local`` fetch window, and a later digest of a page held as text
    — and a partial read that does not say so reads as the whole document. The wording is
    :func:`truncation_note`'s, so this and the digest header cannot drift apart.
    """
    text = joined_page_text(pdf)[0].strip()
    note = truncation_note(pdf)
    if not note:
        return text
    return f"[Partial document read: {pdf.page_count} pages{note}]\n\n{text}"


def select_passages(
    text: str,
    query: str,
    *,
    top_k: int,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    page_breaks: Sequence[int] | None = None,
) -> list[Passage]:
    """The ``top_k`` windows of ``text`` most relevant to ``query``, by BM25, highest first.

    Deterministic and lexical on purpose: the selection step decides which few hundred words
    of a 220-page document a forecaster or the research driver sees, and a model call there
    would put an unauditable choice on the critical path of every PDF. Ties break on the
    LONGER window first and then on position, earliest first, so re-running on the same inputs
    returns the same passages in the same order. Length is the first tiebreak because with
    length normalisation off (see ``BM25_B``) a heading and the paragraph under it can score
    exactly equal, and of two equally-scoring windows the one carrying more context is the
    more useful passage.

    Returns ``[]`` when no window contains any query term — an empty result means "this
    document does not discuss what you asked", which is information, where handing back the
    first ``top_k`` windows would dress the front matter up as an answer.

    Supplying ``page_breaks`` also makes a page boundary a hard window boundary, which is what
    keeps ``Passage.page`` exact: without it, two short pages pack into one window and the
    passage is labelled with the page it starts on even when the match is a page later. That
    costs a phrase straddling a page break, which BM25 then scores as two partial windows —
    the right trade, since a ``[p.N]`` label a forecaster may cite has to be true.
    """
    # Sorted once here so the windowing and the page labels read the same order, whatever a
    # caller hands in; `joined_page_text` already returns them ascending.
    breaks = tuple(sorted(page_breaks)) if page_breaks else None
    spans = _window_spans(text, window_chars, page_breaks=breaks)
    if not spans or top_k <= 0:
        return []

    query_tokens = _unique_tokens(query)
    if not query_tokens:
        return []

    window_tokens = [_tokenise(text[start:end]) for start, end in spans]
    scores = _bm25_scores(window_tokens, query_tokens)
    ranked = sorted(
        ((score, spans[i]) for i, score in enumerate(scores) if score > 0.0),
        key=lambda item: (-item[0], -(item[1][1] - item[1][0]), item[1][0]),
    )
    return [
        Passage(
            score=score,
            start=start,
            end=end,
            text=text[start:end],
            page=_page_for_offset(start, breaks),
        )
        for score, (start, end) in ranked[:top_k]
    ]


def render_document_digest(pdf: PdfText, *, query: str, top_k: int, max_chars: int | None, source_url: str) -> str:
    """:func:`digest_pdf`'s block, for a caller that needs only the text.

    Kept only until the Tier-1 resolution-source fetcher moves to :func:`digest_pdf`, which it
    has to in order to record whether the digest matched anything; delete it then.
    """
    return digest_pdf(pdf, query=query, top_k=top_k, max_chars=max_chars, source_url=source_url).block


def digest_pdf(pdf: PdfText, *, query: str, top_k: int, max_chars: int | None, source_url: str) -> DocumentDigest:
    """The forecaster/driver-facing block for one document: what it is, then what it says.

    Deterministic and I/O-free, so the same ``PdfText`` always renders the same block. The
    header states what was read and what was not, because a digest of the first 400 pages of
    a 900-page document that does not say so is a silent partial read.

    The no-text-layer branch fires on a document with no text AT ALL, not on a failed
    ``has_text_layer`` — that function's 40-char floor is the caller's escalate-to-a-paid-read
    signal, and reusing it here would withhold the whole content of a one-line PDF ("Q3
    unemployment rate: 4.1%") as if it were a scan. Terse-but-real text is rendered; a scan
    whose only text is page numbers simply matches no query and says so.
    """
    sections = [_digest_header(pdf, source_url=source_url)]
    if pdf.unreadable_reason:
        sections.append(f"The document could not be parsed ({pdf.unreadable_reason}); no text was extracted.")
        return DocumentDigest(
            block=_truncate_digest("\n\n".join(sections), max_chars),
            passages=0,
            outcome="unreadable",
            scope="unreadable",
        )

    scope = "partial_document" if pdf.truncated_by else "complete_document"
    sections.extend(_digest_outline(pdf.outline))
    text, page_breaks = joined_page_text(pdf)
    if not text.strip():
        sections.append("No extractable text layer: the pages carry images rather than text.")
        return DocumentDigest(
            block=_truncate_digest("\n\n".join(sections), max_chars),
            passages=0,
            outcome="empty_text",
            scope=scope,
        )

    passages = select_passages(text, query, top_k=top_k, page_breaks=page_breaks)
    sections.append(_digest_passages(passages, query=query))
    block = _truncate_digest("\n\n".join(sections), max_chars)
    return DocumentDigest(
        block=block,
        passages=len(passages),
        outcome="answered" if passages else ("partial_no_match" if pdf.truncated_by else "no_match"),
        scope=scope,
        evidence_passages=_rendered_evidence(passages, block, source_url=source_url),
    )


def digest_text(text: str, *, query: str, top_k: int, max_chars: int | None, source_url: str) -> DocumentDigest:
    """The same digest for a document held as flat text — a fetched web page, not a PDF.

    Page-less by construction, so its passages carry no ``[p.N]`` label and the header states
    a char count instead of a page count. Everything else is :func:`digest_pdf`'s — the BM25
    selection, the no-match sentence, the truncation marker — so the two shapes read the same
    way to whoever consumes them, which is what lets one caller serve a PDF and an HTML page
    through one code path.
    """
    passages = select_passages(text, query, top_k=top_k)
    block = render_flat_passages(
        [passage.text for passage in passages],
        query=query,
        max_chars=max_chars,
        source_url=source_url,
        source_chars=len(text),
    )
    return DocumentDigest(
        block=block,
        passages=len(passages),
        outcome="answered" if passages else ("no_match" if text.strip() else "empty_text"),
        scope="complete_text",
        evidence_passages=_rendered_evidence(passages, block, source_url=source_url),
    )


def _rendered_evidence(passages: list[Passage], block: str, *, source_url: str) -> tuple[SourceExcerpt, ...]:
    """The selected passages that survived into the rendered ``block`` intact, as BM25 evidence."""
    return tuple(
        SourceExcerpt(source_url=source_url, text=passage.text, retrieval_method="bm25")
        for passage in passages
        if passage.text in block
    )


def render_flat_passages(
    passages: Sequence[str],
    *,
    query: str,
    max_chars: int | None,
    source_url: str | None = None,
    source_chars: int | None = None,
) -> str:
    """Render selected page-less passages, optionally with the held-document header."""
    if (source_url is None) != (source_chars is None):
        raise ValueError("source_url and source_chars must be provided together")
    cleaned = [passage.strip() for passage in passages if passage.strip()]
    block = _digest_passages(
        [Passage(score=0.0, start=0, end=len(passage), text=passage, page=None) for passage in cleaned],
        query=query,
    )
    if source_url is not None and source_chars is not None:
        block = f"Document: {source_url}\n{source_chars} chars of text, no page structure\n\n{block}"
    return _truncate_digest(block, max_chars)


# --- PDF reading -------------------------------------------------------------------------


def _open_reader(body: bytes) -> tuple[PdfReader | None, UnreadableReason]:
    """A reader for ``body``, or None plus the reason it is unreadable."""
    try:
        reader = PdfReader(io.BytesIO(body))
    except _PDF_ERRORS:
        return None, "malformed"
    if not reader.is_encrypted:
        return reader, ""
    # An empty user password is the common "encrypted for print restrictions only" case and
    # reads normally; a real password means the bytes stay opaque to us.
    try:
        decrypted = bool(reader.decrypt(""))
    except _PDF_ERRORS:
        decrypted = False
    return (reader, "") if decrypted else (None, "encrypted")


def _page_count(reader: PdfReader) -> int | None:
    """The declared page count, or None when the page tree itself does not parse."""
    try:
        return len(reader.pages)
    except _PDF_ERRORS:
        return None


def _read_pages(
    reader: PdfReader, *, page_count: int, max_pages: int, max_seconds: float
) -> tuple[tuple[str, ...], TruncationCause]:
    """Text of the first ``max_pages`` pages, stopping early once ``max_seconds`` elapses.

    ``max_seconds`` is a BETWEEN-PAGES CHECKPOINT, not an elapsed bound: the clock is read
    after each page returns, so a page already in progress runs to completion however long it
    takes and the total can overrun the budget by one page's work. That is deliberate for the
    small-budget case — a budget too small for even one page still returns that page, because
    forward progress beats an empty result and the caller learns the bound bit anyway from
    ``truncated_by``. What keeps the overrun finite is PDF_MAX_DECODED_STREAM_BYTES, which
    bounds the decoded bytes a single page's content stream can cost; without it one page can
    outrun any budget (F44's receipt: 9 MB of decoded operators for ~7 s of CPU). A
    before-each-page check would add nothing, since the same page still has to finish.

    The clock also starts here, which is late. ``extract_pdf_text`` reads the declared page count
    and walks the whole bookmark outline first, and neither of those steps is inside this budget;
    call them the prologue. pypdf's own caps keep the prologue finite (100,000 outline entries,
    100 levels of nesting), and a body built to maximize both measured a ceiling around 16 s. So
    the real worst case for one parse is the prologue plus ``max_seconds`` plus one page, and a
    caller that cancels the surrounding coroutine stops none of it, because the worker runs in a
    thread. Recorded in FUTURE.md under "The PDF parse overruns ``max_seconds``".
    """
    deadline = monotonic() + max_seconds
    limit = min(page_count, max_pages)
    pages: list[str] = []
    for index in range(limit):
        pages.append(_page_text(reader, index))
        if monotonic() >= deadline and index + 1 < page_count:
            return tuple(pages), "seconds"
    return tuple(pages), "pages" if limit < page_count else ""


def _page_text(reader: PdfReader, index: int) -> str:
    """One page's text, or ``""`` when that page has no text layer or fails to decode.

    A page whose content stream is corrupt is reported the same way as a page with no text:
    the whole point of a per-page tuple is that one bad page does not cost the other 219.
    """
    try:
        return reader.pages[index].extract_text().strip()
    except _PDF_ERRORS:
        return ""


def _read_outline(reader: PdfReader) -> tuple[tuple[str, int], ...]:
    """The bookmark tree flattened to (title, 1-based page) in document order.

    Nesting is flattened rather than indented because the digest uses the outline to say what
    is in the document and on which page, and a caller that wants the hierarchy can read the
    PDF itself. An outline whose destinations do not resolve — a linearised file whose page
    objects moved, a tree pypdf cannot walk — yields ``()``: a partial outline with silently
    wrong page numbers is worse than none, since the digest's own page labels come from the
    text offsets and would then contradict it.
    """
    try:
        return tuple(_walk_outline(reader, reader.outline))
    except _PDF_ERRORS:
        return ()


# Counts entries APPENDED, so a nested list or an untitled Destination is popped without
# incrementing it. That makes this a cap on what the walk can hand the digest, and not a cap on
# how much of pypdf's tree the walk visits; pypdf's own OUTLINE_MAX_ENTRIES = 100_000 is the only
# bound on the visit itself. Recorded in FUTURE.md under "The PDF parse overruns `max_seconds`".
OUTLINE_MAX_ENTRIES_WALKED = 2_000


def _walk_outline(reader: PdfReader, items: Sequence[object]) -> list[tuple[str, int]]:
    """Depth-first walk of pypdf's nested list of Destinations, without recursion.

    Iterative on purpose: the bookmark tree is attacker-shaped input, and a recursive walk over
    a pathologically deep tree raises RecursionError, which is not a pypdf error and would break
    this module's never-raises contract. The explicit stack cannot exhaust the interpreter's, and
    the entry cap bounds how many titled entries a wide tree can produce; the digest shows
    DIGEST_MAX_OUTLINE_ENTRIES of these anyway.

    An untitled entry is skipped: the digest renders the outline as "title (p.N)", and a
    bookmark with no title carries no information a reader can act on.
    """
    entries: list[tuple[str, int]] = []
    stack: list[object] = list(reversed(items))
    while stack and len(entries) < OUTLINE_MAX_ENTRIES_WALKED:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
            continue
        if not isinstance(item, Destination):
            continue
        title = (item.title or "").strip()
        page_index = reader.get_destination_page_number(item)
        if title and page_index is not None and page_index >= 0:
            entries.append((title, page_index + 1))
    return entries


def _visible_char_count(page: str) -> int:
    return sum(1 for char in page if not char.isspace())


# --- Passage selection -------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric runs. Punctuation and case carry no retrieval signal here.

    Alphanumeric in the Unicode sense, not ASCII: resolution sources include non-English
    pages, and an ASCII-only class would shred "Ürgüp" into two meaningless fragments and
    then fail to match the same word in the query.
    """
    tokens: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _unique_tokens(query: str) -> list[str]:
    """Scoring terms of ``query``: deduplicated, in first-appearance order, stopwords dropped.

    A repeated term in a short query is emphasis a user did not mean to encode as a weight,
    and dedup keeps the score independent of how the ask was phrased. The stopword filter is
    ``_QUERY_STOPWORDS``, and it applies HERE only — ``select_passages`` is the single
    consumer, and the digest header still renders the caller's query verbatim, so what the
    reader is told was asked stays exactly what was asked.

    A query made ENTIRELY of stopwords keeps its raw tokens: filtering it to nothing would
    turn "will this resolve yes" into a query that matches nothing at all, silently reporting
    that the document does not discuss it. Scoring such a query on procedural words is weak,
    but it is what this selector did before the filter existed, so nothing regresses.
    """
    tokens = list(dict.fromkeys(_tokenise(query)))
    scoring = [token for token in tokens if token not in _QUERY_STOPWORDS]
    return scoring or tokens


def _bm25_scores(window_tokens: list[list[str]], query_tokens: list[str]) -> list[float]:
    """Okapi BM25 of every window against the query terms. No window length is read: b is 0."""
    n_windows = len(window_tokens)
    if not any(window_tokens):
        return [0.0] * n_windows

    counters = [Counter(tokens) for tokens in window_tokens]
    idf = {
        term: log(1 + (n_windows - df + 0.5) / (df + 0.5))
        for term, df in ((term, sum(1 for c in counters if term in c)) for term in query_tokens)
    }
    return [_window_score(counter, query_tokens=query_tokens, idf=idf) for counter in counters]


def _window_score(counter: Counter[str], *, query_tokens: list[str], idf: dict[str, float]) -> float:
    score = 0.0
    for term in query_tokens:
        freq = counter.get(term, 0)
        if not freq:
            continue  # a term absent from this window contributes nothing, absent everywhere or not
        # BM25's length-normalisation factor, (1 - b + b * length / avg_length), is exactly
        # (1 - b) at b=0, so no window length is read at all. See BM25_B for why b is 0.
        denominator = freq + BM25_K1 * (1 - BM25_B)
        score += idf[term] * freq * (BM25_K1 + 1) / denominator
    return score


def _window_spans(text: str, window_chars: int, *, page_breaks: Sequence[int] | None = None) -> list[tuple[int, int]]:
    """Char spans covering ``text``, never crossing a page break, in document order."""
    if window_chars <= 0:
        return []
    spans: list[tuple[int, int]] = []
    for start, end in _page_segments(text, page_breaks):
        spans.extend(_segment_windows(text, start, end, window_chars))
    return spans


def _page_segments(text: str, page_breaks: Sequence[int] | None) -> list[tuple[int, int]]:
    """``text`` cut at the page starts, or one segment when the caller gave none."""
    if not page_breaks:
        return [(0, len(text))]
    bounds = sorted({0, len(text), *(offset for offset in page_breaks if 0 < offset < len(text))})
    return list(pairwise(bounds))


def _segment_windows(text: str, start: int, end: int, window_chars: int) -> list[tuple[int, int]]:
    """One page's spans, split on paragraph boundaries where it has them.

    Packing consecutive short paragraphs up to ``window_chars`` keeps a scored window big
    enough to carry context, and a paragraph longer than the budget (a wall-of-text page, or
    a PDF whose extraction produced no blank lines at all) falls back to fixed windows cut at
    whitespace.
    """
    spans: list[tuple[int, int]] = []
    pending: tuple[int, int] | None = None
    for para_start, para_end in _paragraph_spans(text, start, end):
        if para_end - para_start > window_chars:
            if pending is not None:
                spans.append(pending)
                pending = None
            spans.extend(_split_long_span(text, para_start, para_end, window_chars))
        elif pending is None:
            pending = (para_start, para_end)
        elif para_end - pending[0] <= window_chars:
            pending = (pending[0], para_end)
        else:
            spans.append(pending)
            pending = (para_start, para_end)
    if pending is not None:
        spans.append(pending)
    return spans


def _paragraph_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Non-empty paragraph spans within ``[start, end)``, whitespace-trimmed, in order."""
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        break_at = _next_blank_line(text, cursor, end)
        stop = end if break_at is None else break_at[0]
        trimmed = _trim_span(text, cursor, stop)
        if trimmed is not None:
            spans.append(trimmed)
        if break_at is None:
            break
        cursor = break_at[1]
    return spans


def _next_blank_line(text: str, start: int, length: int) -> tuple[int, int] | None:
    """Span of the next run of whitespace containing two or more newlines, or None."""
    cursor = start
    while cursor < length:
        if text[cursor] != "\n":
            cursor += 1
            continue
        run_end = cursor
        newlines = 0
        while run_end < length and text[run_end].isspace():
            if text[run_end] == "\n":
                newlines += 1
            run_end += 1
        if newlines >= 2:
            return cursor, run_end
        cursor = max(run_end, cursor + 1)
    return None


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """``(start, end)`` narrowed past surrounding whitespace, or None when it holds none."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _split_long_span(text: str, start: int, end: int, window_chars: int) -> list[tuple[int, int]]:
    """A too-long paragraph cut into ``window_chars`` windows, at whitespace where possible."""
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + window_chars, end)
        if stop < end:
            pivot = _last_space(text, cursor + (window_chars * 3) // 4, stop)
            if pivot > cursor:
                stop = pivot
        trimmed = _trim_span(text, cursor, stop)
        if trimmed is not None:
            spans.append(trimmed)
        cursor = stop if stop > cursor else cursor + 1
    return spans


def _last_space(text: str, low: int, high: int) -> int:
    """Index of the last whitespace char in ``[low, high)``, or -1 — the preferred cut point."""
    for index in range(high - 1, low - 1, -1):
        if text[index].isspace():
            return index
    return -1


def _page_for_offset(offset: int, page_breaks: Sequence[int] | None) -> int | None:
    """The 1-based page containing ``offset``, given ascending page-start offsets."""
    if not page_breaks:
        return None
    return max(1, bisect_right(page_breaks, offset))


# --- Digest rendering --------------------------------------------------------------------


def _digest_header(pdf: PdfText, *, source_url: str) -> str:
    chars = sum(len(page) for page in pdf.pages)
    note = truncation_note(pdf)
    return (
        f"Document: {source_url}\n{pdf.page_count} pages, {pdf.pages_read} read, {chars} chars of text extracted{note}"
    )


def _digest_outline(outline: tuple[tuple[str, int], ...]) -> list[str]:
    if not outline:
        return []
    shown = outline[:DIGEST_MAX_OUTLINE_ENTRIES]
    lines = ["Outline:"] + [f"  {title} (p.{page})" for title, page in shown]
    if len(outline) > len(shown):
        lines.append(f"  ... {len(outline) - len(shown)} further outline entries")
    return ["\n".join(lines)]


def _digest_passages(passages: list[Passage], *, query: str) -> str:
    if not passages:
        # The sentence has to read as a statement about the DOCUMENT either way: a blank query
        # left "matched the query:" dangling, which reads as a render that lost its own text
        # rather than as "this document does not discuss what was asked".
        asked = query.strip()
        return f"{_NO_MATCH_SENTENCE}: {asked}" if asked else f"{_NO_MATCH_SENTENCE}."
    header = f"Most relevant passages for: {query}"
    # A page-less passage is labelled "[passage]" rather than "[p.?]": the flat-text digest has
    # no pages to number at all, and "?" would read as a page we failed to identify.
    body = [f"[p.{p.page}] {p.text}" if p.page is not None else f"[passage] {p.text}" for p in passages]
    return "\n\n".join([header, *body])


def _truncate_digest(block: str, max_chars: int | None) -> str:
    """``block`` bounded to ``max_chars``, with the cut disclosed in the text; None is unbounded.

    The marker is never dropped to make room, even when ``max_chars`` is smaller than the
    marker itself: a block that silently ends mid-passage reads as a complete document
    digest, which is exactly the failure the marker exists to prevent.
    """
    if max_chars is None or len(block) <= max_chars:
        return block
    marker = f"[digest truncated at {max_chars} chars]"
    keep = max(0, max_chars - len(marker) - 1)  # -1 for the newline joining the two
    kept = block[:keep].rstrip()
    return f"{kept}\n{marker}" if kept else marker
