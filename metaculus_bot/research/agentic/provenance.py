"""Provenance and retrieval-quality primitives for the gap-fill v2 driver loop.

Everything here answers one of two questions about a URL a driver-supplied
finding cites: did the driver actually SEE it through a tool this run (the hard
provenance gate), and HOW WELL did it see it — a page we fetched, or a search
snippet (the W4 verification tier that decides whether a discrepancy may
supersede the briefing). The URL/quote normalizers live beside the per-call
harvesters that feed the loop's accumulators because the gate and the tier must
agree on one URL set: splitting them is how a URL ends up provenance-seen with
no tier, or tiered without having been seen (F7).

Pure functions over one tool call's arguments and outcome — no loop state, no
logging. ``loop.py`` folds each call's harvest into ``_LoopState``, and
``gates.py`` reads the accumulators back when it validates a finding.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from metaculus_bot.research.agentic.tool_schemas import _INTERNAL_TOOL_NAMES
from metaculus_bot.research.agentic.types import ToolOutcome

# Provenance gate (source_url grounding + quote spot-check). A finding is
# rendered under a "supersedes-the-briefing" banner and shown to every base
# forecaster, so a hallucinated/mistyped citation from the low-effort driver
# would silently override correct research. The URL check is a HARD gate; the
# quote check is WARN-ONLY (read_document paraphrases and ellipsis-joined
# quotes make a hard quote gate too false-positive-prone). The banner says
# "sourced" (not "verified") because only the URL is gated — see artifact.py.
_URL_IN_TEXT_RE = re.compile(r"https?://[^\s<>\"'\)\]]+")
_URL_TRAILING_PUNCT = ".,;:)]}>\"'"
_TRACKER_PARAM_PREFIXES = ("utm_",)
_TRACKER_PARAM_NAMES = frozenset({"gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src", "igshid", "spm"})
# All straight/curly quote glyphs and backticks are DELETED during normalization
# so a driver quote wrapped in glyphs still matches unwrapped source text — both
# sides run through _normalize_quote_text, so deletion is symmetric.
_QUOTE_GLYPHS_RE = re.compile(r"[\"'‘’“”`]")  # noqa: RUF001  # the curly glyphs ARE the pattern; ASCII-ifying would stop matching them
_WHITESPACE_RE = re.compile(r"\s+")

# Retrieval-quality tiers (W4). ToolOutcome.method records HOW a URL's content
# reached the driver; we collapse those method values into two tiers so a
# finding stamped from the URL->best-method map carries honest authority. A
# "fetched" URL is one whose actual page/document we pulled (document/rendered/
# plain/cache/pdf_local/digest_local); a "snippet" URL was only seen in a
# search/news result. A discrepancy resting on a snippet must NOT supersede the
# briefing (the 131.3 failure mode: a crowd-median "correction" from a search
# snippet after the direct fetch 403'd, which every forecaster then adopted).
# Methods absent here (internal bookkeeping, error/blocked/throttled/oversize
# outcomes, the intermediate document_needed state) contribute no tier — the URL
# still counts for the provenance gate, but the finding stays untiered and a
# discrepancy on it is demoted, conservatively. See artifact.render_findings.
#
# The two local-document methods sit in the "fetched" class deliberately: we
# decoded the bytes the host served us (pypdf text, then a deterministic BM25
# passage selection), which is a STRONGER claim than "document" — a model's
# reading of a URL it fetched itself and we never saw.
_METHOD_TO_TIER: dict[str, str] = {
    "document": "fetched",
    "rendered": "fetched",
    "plain": "fetched",
    # The impersonated retry read the host's own bytes through the shared impersonated rung;
    # absent from here a page it really did retrieve would stay untiered and its discrepancy
    # silently demoted below the briefing, the 131.3 failure mode above.
    "impersonate": "fetched",
    # The archive served the host's own bytes with the age disclosed in-text (the shared Wayback rung).
    "wayback": "fetched",
    # The page's own JSON feed, harvested during its render and served directly (the shared derived rung).
    "derived_api": "fetched",
    "cache": "fetched",
    "pdf_local": "fetched",
    "digest_local": "fetched",
    "known_api": "fetched",
    "image_local": "fetched",
    "local": "fetched",
    "search": "snippet",
    "news": "snippet",
}
# fetched outranks snippet, so a URL seen via search THEN fetched upgrades.
_TIER_RANK: dict[str, int] = {"snippet": 0, "fetched": 1}


def _outranks(candidate: str | None, incumbent: str | None) -> bool:
    """True when ``candidate`` is a strictly better verification tier than
    ``incumbent``. None (untiered — the URL was never retrieved through a tool)
    ranks below every real tier."""
    return (-1 if candidate is None else _TIER_RANK[candidate]) > (-1 if incumbent is None else _TIER_RANK[incumbent])


def _method_to_tier(method: str) -> str | None:
    """Map a ToolOutcome.method to a verification tier ("fetched"/"snippet"), or
    None when the method isn't a real content retrieval (internal bookkeeping,
    error/blocked states) and so grants no retrieval authority."""
    return _METHOD_TO_TIER.get(method)


def _normalize_url(url: str) -> str | None:
    """Canonicalize a URL for provenance comparison.

    Lowercases scheme + host, drops the fragment, strips a trailing slash from
    the path, and removes common tracker query params (``utm_*``, ``gclid``,
    ``fbclid``, ...). Returns ``None`` for non-http(s) or unparseable input, so
    those never count as provenance.
    """
    candidate = url.strip().rstrip(_URL_TRAILING_PUNCT)
    try:
        parts = urlsplit(candidate)
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return None
    host = parts.netloc.lower()
    if not host:
        return None
    path = parts.path.rstrip("/")
    kept_params = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(_TRACKER_PARAM_PREFIXES) and key.lower() not in _TRACKER_PARAM_NAMES
    ]
    return urlunsplit((scheme, host, path, urlencode(kept_params), ""))


def _iter_normalized_urls(text: str) -> Iterator[str]:
    """Yield the normalized form of every http(s) URL found in free text."""
    for match in _URL_IN_TEXT_RE.finditer(text):
        normalized = _normalize_url(match.group(0))
        if normalized is not None:
            yield normalized


def _normalize_quote_text(text: str) -> str:
    """Lowercase, collapse whitespace, and DELETE quote glyphs for substring matching.

    Deletion (not substitution) keeps the normalizer symmetric. The driver wraps
    its ``quote`` field in glyphs by convention while source text usually is not
    wrapped, so substituting a placeholder apostrophe left the quote with leading/
    trailing apostrophes the source lacked and the substring test missed genuinely
    verbatim content. Both the quote and the tool corpus run through here, so
    deleting glyphs on both sides makes the wrapped and unwrapped forms converge.
    """
    return _WHITESPACE_RE.sub(" ", _QUOTE_GLYPHS_RE.sub("", text)).strip().lower()


# Span boundaries in a driver quote — the ways it stitches non-contiguous
# source text, none of which any single contiguous substring test could satisfy:
#   1. An ellipsis: the literal "..." (three-or-more dots) or the Unicode "…".
#      The driver elides mid-quote ("<span A> ... <span B>").
#   2. A QUOTE-GLYPH boundary: a CLOSING glyph joined to an OPENING glyph by a
#      BOUNDED non-glyph connective (up to 24 chars, lazily matched). The original
#      2026-07-28 clause accepted whitespace only (`"<A>" "<B>"`, added after 8
#      all-false-positive warnings in that prod run); the 2026-08-24 residual
#      round measured that 65% of all 365 warnings ever emitted carried a joiner
#      the whitespace-only clause missed — `"<A>" and "<B>"`, `"<A>"; "<B>"`,
#      `"<A>", "<B>"`, or a short narration fragment — and 0 of the 156
#      multi-span stitched quotes could pass on a corpus containing every span
#      verbatim. The bounded connective admits those; a longer joiner (a full
#      narration sentence) still reads as one span and stays a warning.
#      The lookarounds are what make "closing then opening" real: the first glyph
#      must be preceded by non-whitespace (span text ends at it) and the second
#      followed by non-whitespace (span text starts at it). Without them the
#      residual round's literal `[glyph][^glyph]{0,24}?[glyph]` form consumes any
#      quoted span whose CONTENT is <=24 chars as a "boundary" — deleting exactly
#      the short table cells (`"Windows | 56.61%"`) the 10-char floor below was
#      tuned to keep — and the pinned ellipsis shapes in tests/test_agentic_gates.py
#      fail. The whitespace-only clause survives as its OWN alternative, because
#      the lookarounds do not subsume it: a span ending in whitespace before its
#      closing glyph (`"span A " "span B"`) fails `(?<=\S)` and would otherwise
#      read as one contiguous span and warn — the exact false-positive class the
#      2026-07-28 clause was added to eliminate. Whitespace-only can never consume
#      a span with non-whitespace content, so it is safe alongside.
# The whole alternation is ONE capturing group, deliberately: `re.split` DISCARDS
# unmatched separators, and a discarded glyph-boundary connective is up to
# _SPAN_JOINER_MAX_CHARS of driver text that never gets checked — a fabricated
# figure riding in the joiner between two verbatim spans (`"<A>" up 47.3% from
# "<B>"`) would ground cleanly on the spans alone, defeating the digit clause
# whose own comment calls that "the whole risk". With the capture, split returns
# the connectives at the ODD indices and `_quote_is_grounded` checks any that
# carry a digit. Digit-FREE connectives stay unchecked whatever their length:
# a connective is driver narration by construction, not source text, so requiring
# it in the corpus would manufacture exactly the false positives the bounded
# connective was measured to fix (65% of all 365 warnings ever emitted).
# This regex is applied to the RAW quote, BEFORE normalization, and each resulting
# span normalized on its own. That order is load-bearing: _normalize_quote_text
# DELETES quote glyphs, so normalizing first destroys the very boundary clause 2
# looks for. Splitting raw is safe for clause 1 too — normalization collapses
# whitespace and deletes glyphs but never touches runs of dots (verified by
# execution across the quote shapes in tests/test_agentic_loop.py).
# Two intra-word apostrophes within 24 chars of each other ("Musk's ... Trump's")
# DO now split — harmless by construction, since every contiguous piece of a
# genuinely verbatim quote is itself verbatim, and sub-floor non-numeric fragments
# are not trusted as positive evidence anyway (the floor + digit clauses below
# are unchanged).
#
# The connective bound is empirically tuned, not principled: the 2026-08-24 round
# measured the real joiners (" and ", "; ", ", ", short narration fragments) well
# under it and full narration sentences well over it. Named so the boundary tests
# can pin it at N and N+1 instead of restating a magic number.
_SPAN_JOINER_MAX_CHARS = 24
# The RUF001 suppressions below are load-bearing: the curly quote glyphs ARE the
# pattern, so ASCII-ifying them would stop matching the text they exist to split.
_SPAN_BOUNDARY_RE = re.compile(
    r"(\.{3,}|…"
    r"|[\"'‘’“”`]\s*[\"'‘’“”`]"  # noqa: RUF001
    rf"|(?<=\S)[\"'‘’“”`][^\"'‘’“”`]{{0,{_SPAN_JOINER_MAX_CHARS}}}?[\"'‘’“”`](?=\S))"  # noqa: RUF001
)
# Minimum normalized length for a split span to be grounded on its own.
# Below this a span is a bare token or punctuation run that appears in arbitrary
# text, so trusting it per-span would rubber-stamp the check. Set to 10 rather
# than the ~25 a prose-calibrated analysis suggested: real findings elide compact
# TABLE CELLS ("Windows | 56.61%" is 16 normalized chars, "Linux | 4.36%" is 13),
# which a 25-floor would drop — collapsing every stitched quote back to the
# whole-quote fallback that a contiguous test can never satisfy, recreating the
# dead-check symptom for the ellipsis subset. 10 sits above the single bare tokens
# around those cells ("Windows"/"Unknown" are 7) so an isolated common word can't
# ground a fabricated stitch, and below the real cell spans so genuine elisions
# still verify per-span.
_MIN_GROUNDING_SPAN_CHARS = 10
# A sub-floor span carrying a digit is checked anyway rather than skipped. The
# floor exists so a short span is not TRUSTED as positive evidence; it must not
# mean a short span is IGNORED. Numbers are the whole risk: they are short, they
# are what gets fabricated, and they are what a forecaster acts on. A quote
# pairing a real long clause with an invented figure ("<real clause> ... 47.3%")
# otherwise grounded cleanly on the clause alone (forge panel, reproduced by
# execution). Short spans WITHOUT a digit stay unchecked — a bare connective
# fragment appears in arbitrary text, so requiring it would only add noise.
_DIGIT_RE = re.compile(r"\d")


def _quote_is_grounded(quote: str, tool_content_normalized: str) -> bool:
    """True when the finding's quote is grounded in the tool contents.

    An empty quote is treated as grounded — there is nothing to verify. The driver
    stitches non-contiguous source text two ways — eliding with an ellipsis, and
    joining separately-quoted sentences with a space — so the quote is split on
    ``_SPAN_BOUNDARY_RE`` and every span that carries evidentiary weight must
    appear in the tool contents independently (a plain unstitched quote is one
    span — the whole thing). The split runs on the RAW quote and each span is
    normalized afterwards, because normalization deletes the quote glyphs that
    mark a glyph boundary; see the ``_SPAN_BOUNDARY_RE`` block. A span carries
    weight when it clears ``_MIN_GROUNDING_SPAN_CHARS`` or contains a digit; the
    digit clause closes the hole where a fabricated figure rode alongside a
    genuine long clause.

    The regex's one capture group makes ``re.split`` return the matched boundaries
    at the ODD indices, and those get the narrower rule: a boundary connective is
    driver narration rather than source text, so it is checked only when it
    carries a digit (the fabrication risk the digit clause exists for) and is
    otherwise ignored whatever its length — demanding narration verbatim in the
    corpus would manufacture the false positives the bounded connective was
    measured to fix. When no piece carries weight — the quote is all short
    non-numeric fragments — the whole normalized quote is tested as a substring so
    a short quote is never auto-passed; only a truly empty quote passes for free.
    """
    normalized_quote = _normalize_quote_text(quote)
    if not normalized_quote:
        return True
    spans: list[str] = []
    for index, part in enumerate(_SPAN_BOUNDARY_RE.split(quote)):
        normalized = _normalize_quote_text(part)
        if not normalized:
            continue
        if index % 2 == 1:  # a captured boundary: connective narration, digit-gated
            if _DIGIT_RE.search(normalized):
                spans.append(normalized)
        elif len(normalized) >= _MIN_GROUNDING_SPAN_CHARS or _DIGIT_RE.search(normalized):
            spans.append(normalized)
    if not spans:
        return normalized_quote in tool_content_normalized
    return all(span in tool_content_normalized for span in spans)


def _surfaced_urls(arguments: dict[str, Any], outcome: ToolOutcome) -> list[str]:
    """Normalized URLs an EXTERNAL tool call actually surfaced.

    Exactly the ``url`` the driver asked the tool to retrieve (fetch/read_document
    take one — a search's ``query`` is a search term, not a retrieval target)
    plus every URL in the result body and link list. A URL merely typed into a
    free-text argument (read_document's ``ask``, a ``query``) was NOT retrieved by
    the tool, so it is deliberately excluded: that is what stops a URL a driver
    pastes into ``ask`` from laundering itself into provenance or a "fetched" tier
    (F1). Both ``_harvest_provenance`` and the snippet tier path share this so a
    URL can never be provenance-seen without a matching tier, or vice versa (F7).
    """
    urls: list[str] = []
    requested = arguments.get("url")
    if isinstance(requested, str):
        urls.extend(_iter_normalized_urls(requested))
    urls.extend(_iter_normalized_urls(outcome.content_markdown))
    for link in outcome.links:
        normalized = _normalize_url(link)
        if normalized is not None:
            urls.append(normalized)
    return urls


def _harvest_provenance(tool_name: str, arguments: dict[str, Any], outcome: ToolOutcome) -> tuple[list[str], str]:
    """Collect the normalized URLs and result text a single EXTERNAL tool call surfaced.

    Internal bookkeeping tools contribute nothing: their echoed content restates
    the driver's own rejected findings, so harvesting them would let a
    hallucinated URL launder itself into ``tool_seen_urls``.
    """
    # Navigation inventories expose possible members/sheets/images without
    # reading their contents. Keep the inventory visible in the tool message,
    # but never let its URLs or labels ground a finding or quote.
    if tool_name in _INTERNAL_TOOL_NAMES or outcome.method == "local_navigation":
        return [], ""
    return _surfaced_urls(arguments, outcome), outcome.content_markdown


def _harvest_verification_tiers(tool_name: str, arguments: dict[str, Any], outcome: ToolOutcome) -> dict[str, str]:
    """Assign a retrieval tier to the URLs this EXTERNAL tool call established (W4).

    Only a successful (``status == "ok"``) outcome grants a tier — a 403'd/
    blocked fetch confers no authority, which is the exact 131.3 mechanism (the
    real fetch failed, so a later search snippet must not inherit "fetched").

    A **fetched-class** call (document/rendered/plain/cache) tiers ONLY the page
    it actually retrieved — the ``url`` argument the driver asked for — as
    "fetched". The explicit known-API tools have no ``url`` argument, so they tier
    only the canonical source links their backend adapter owns. A URL merely named
    in free text or an ordinary fetch result's body/links is a lead, not a page we
    read, so it earns no tier from that call (F1: an ``ask``-URL must not inherit
    fetched authority). A **snippet-class**
    call (search/news) tiers every URL it surfaced (the exact set provenance
    harvests — requested ``url``, body, links) as "snippet": the driver saw only
    the excerpt, never the page.
    """
    if tool_name in _INTERNAL_TOOL_NAMES or outcome.status != "ok":
        return {}
    tier = _method_to_tier(outcome.method)
    if tier is None:
        return {}
    if tier == "fetched":
        # Only the requested page (the `url` argument) counts as retrieved.
        requested = arguments.get("url")
        if isinstance(requested, str):
            return dict.fromkeys(_iter_normalized_urls(requested), tier)
        if outcome.method == "known_api":
            canonical_links = (_normalize_url(link) for link in outcome.links)
            return {link: tier for link in canonical_links if link is not None}
        return {}
    # snippet: every surfaced URL was seen only as an excerpt. Reuse provenance's
    # URL set so tier and provenance can't drift (F7).
    return dict.fromkeys(_surfaced_urls(arguments, outcome), tier)
