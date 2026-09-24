"""Reading a document the fetch ladder already holds, instead of paying a model to read it.

The gap-fill v2 ladder used to classify a PDF as ``document_needed`` from its Content-Type
alone, before a single byte was read, and hand it to a paid Gemini ``url_context`` call. Over
the 2026 summer season that was 191 reader calls, nine documents over 100k tokens carried 67%
of all the tokens retrieved, and on the one document where both routes were tried local pypdf
pulled 833,450 chars in 5.3 s while the paid read returned nothing at all. So the order is now
acquire the bytes, extract the text locally, select passages locally, and spend a reader call
only on a document we genuinely cannot read.

This module owns the held-document representation and digest rendering that sit between the
ladder spine in ``tools.py`` and the pure text machinery in ``research/document_text.py``:

* :class:`HeldDocument` — the text, parsed PDF structure, or refusal the free ladder holds.
* :func:`digest_held` — the passage digest for a document we hold, page-wise for a PDF and
  flat for an HTML page, rendered in one shape either way.
* :func:`exceeds_url_context_size_gate` — the hard floor on the one paid call in the ladder.
* :func:`log_local_document_read` — the ``AGENTIC_FETCH_LOCAL_DOC`` telemetry marker.

It deliberately does NOT import ``tools``: the dependency runs one way (``tools`` →
``local_document`` → ``document_text``), so the extraction and the digest stay testable
without standing up the ladder, its aiohttp session or its Chromium rung.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from metaculus_bot.constants import (
    DOCUMENT_TEXT_PDF_MAX_BYTES,
    URL_CONTEXT_SIZE_GATE_TOKENS,
)
from metaculus_bot.research.agentic.fetch_outcomes import PlainFetchResult
from metaculus_bot.research.document_text import (
    DocumentDigest,
    PdfText,
    digest_pdf,
    digest_text,
    disclosed_page_text,
)
from metaculus_bot.research.source_documents import ParsedSource

logger = logging.getLogger(__name__)

# ``ToolOutcome.method`` values this rung produces. Both map to the ``fetched`` verification
# tier in ``provenance._METHOD_TO_TIER`` — we decoded the bytes the host served, which is a
# stronger claim than a model's paraphrase of them — so both names are load-bearing and are
# pinned by the tier tests rather than spelled inline at their call sites.
PDF_LOCAL_METHOD = "pdf_local"
DIGEST_LOCAL_METHOD = "digest_local"
# NOT tiered: nothing was read. Its own name rather than a bare "error" so the driver's
# outcome, and the archived run log, say WHY the paid reader was skipped as well.
OVERSIZE_DOCUMENT_METHOD = "oversize_document"

# chars / 4, the estimator the season's reader sizing was measured with.
_CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass(frozen=True, slots=True)
class HeldDocument:
    """What the free ladder holds for one URL: its text, its page structure, or neither.

    ``text`` is the whole document as one string (a PDF's pages joined, or a page's extracted
    main text) — the pagination and digest source. ``pdf`` is present only when we parsed a
    PDF, and carries the page offsets that make a digest's ``[p.N]`` labels exact; it is set
    even for a scan, where ``text`` is empty, because "we looked locally and there is no text
    layer" is exactly what tells a later call to stop trying for free. ``oversize`` means the
    body was refused before parsing, which is a reason NOT to escalate rather than a reason to.
    ``refused_landing`` is the ladder's own ``blocked`` result for a URL that LED somewhere it
    must not dial (a 3xx onto a question-platform host): the paid reader dials from Google's
    address and would follow the same hop, so the refusal is held for it to honour too.
    """

    text: str = ""
    pdf: PdfText | None = None
    oversize: bool = False
    refused_landing: PlainFetchResult | None = None
    source: ParsedSource | None = None
    local_refusal: PlainFetchResult | None = None

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


def held_pdf(pdf: PdfText) -> HeldDocument:
    """The held form of a parsed document: its joined text plus the page structure.

    A scan comes back with page structure and no text, which is the shape that tells a caller
    the free route is exhausted rather than untried.
    """
    return HeldDocument(text=disclosed_page_text(pdf), pdf=pdf)


_OVERSIZE_DOCUMENT_MSG = (
    "Document too large to read: {url} is a document over the {mib} MiB local-read cap, so "
    "nothing was read from it and no model read was attempted either — a document this size "
    "costs more to have a model retrieve than any answer it could return is worth. Find a "
    "smaller source, or a page that summarises this one."
)


def oversize_message(url: str) -> str:
    return _OVERSIZE_DOCUMENT_MSG.format(url=url, mib=DOCUMENT_TEXT_PDF_MAX_BYTES // (1024 * 1024))


def digest_held(held: HeldDocument, *, ask: str, top_k: int, max_chars: int, source_url: str) -> DocumentDigest:
    """The passage digest for a document we hold, page-wise where we have pages.

    One entry point for both shapes so the driver reads one format whether the ask landed on
    a PDF or an HTML page; the PDF branch is the richer one (outline, per-passage page
    numbers) and is preferred whenever a parse is in hand.
    """
    if held.pdf is not None:
        return digest_pdf(held.pdf, query=ask, top_k=top_k, max_chars=max_chars, source_url=source_url)
    return digest_text(held.text, query=ask, top_k=top_k, max_chars=max_chars, source_url=source_url)


def exceeds_url_context_size_gate(text: str) -> bool:
    """True when text we ALREADY hold is too big to be worth a paid ``url_context`` read.

    A hard floor on the one paid call in this rung rather than a live branch: the ladder above
    serves any URL whose text it holds from the local digest, so a document this size should
    never reach the reader at all. It is enforced anyway because the failure it prevents is
    the season's worst reader spend — the nine archived documents past this bound carried 67%
    of all tokens the reader retrieved, and the largest of them returned nothing for the money.
    """
    return len(text) // _CHARS_PER_TOKEN_ESTIMATE > URL_CONTEXT_SIZE_GATE_TOKENS


def log_local_document_read(url: str, *, method: str, chars: int, pages: int | None, passages: int | None) -> None:
    """Emit ``AGENTIC_FETCH_LOCAL_DOC``: one line per document served without a model call.

    ``chars`` is the local text we HELD, not the window or block handed to the driver, so the
    figure is comparable across both methods and against the size gate above. ``pages`` is
    ``n/a`` for a page with no page structure and ``passages`` is ``n/a`` for a ``pdf_local``
    fetch, which serves the text itself and selects nothing.
    """
    logger.info(
        f"AGENTIC_FETCH_LOCAL_DOC: url={url} method={method} chars={chars} "
        f"pages={'n/a' if pages is None else pages} passages={'n/a' if passages is None else passages}"
    )
