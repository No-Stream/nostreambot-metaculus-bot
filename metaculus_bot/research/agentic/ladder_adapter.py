"""One ``FetchResult``, read as this ladder's own ``PlainFetchResult``: the loop's half of the ladder.

The gap-fill v2 tools run on the shared fetch ladder (``research/fetch_ladder/``), which speaks the
resolution-source vocabulary: twelve ``FetchStatus`` values, eight ``FetchRoute`` values, and a
``status_reason`` where a status has more than one rule behind it. The driver reads six statuses and
a ``method`` the verification-tier map keys on. This module is the ONE place those two vocabularies
meet, so a status the ladder gains cannot silently become an ``ok`` the loop stamps ``fetched``.

Five facts of the tool contract it exists to preserve, each of which a test pins:

* ``ok`` means content was read and nothing else, because ``provenance._harvest_verification_tiers``
  grants the ``fetched`` tier on status alone.
* ``method`` is a ``provenance._METHOD_TO_TIER`` token for a real read and one the map does not
  carry for a non-read (``empty``, ``document_needed``, ``oversize_document``, ``throttled``).
* ``http_status`` is set only by a host's response, never by a refusal the ladder made itself.
* ``escalate_rendered`` is the caller's thin-content signal, with a chart block pinning it off.
* Links resolve against the DOCUMENT url, after a client-side redirect.

The mapping table itself, and which cell each of the loop's messages comes from:
``docs/agentic_gap_fill.md`` "The shared fetch ladder, and the adapter over it".
"""

from __future__ import annotations

from metaculus_bot.research.agentic.fetch_outcomes import (
    _PLATFORM_FETCH_BLOCK_MSG,
    DOCUMENT_NEEDED_METHOD,
    PlainFetchResult,
    _document_needed_result,
)
from metaculus_bot.research.agentic.local_document import OVERSIZE_DOCUMENT_METHOD, PDF_LOCAL_METHOD, oversize_message
from metaculus_bot.research.http_fetch import REDIRECT_STATUSES
from metaculus_bot.research.resolution_fetch_result import FetchResult, FetchRoute, FetchStatus

# Total by construction, so a status the ladder gains is a type error rather than a silent `ok`.
_LOOP_STATUS: dict[FetchStatus, str] = {
    "success": "ok",
    "throttled": "throttled",
    "blocked": "blocked",
    "ssrf_blocked": "blocked",
    "not_found": "error",
    "error": "error",
    "unsupported_type": "error",
    "js_wall": "empty",
    "empty_body": "empty",
    "no_resolving_content": "empty",
    # `ok` with a method the tier map does not carry: nothing was read, and `fetch` escalates on it.
    "unreadable_document": "ok",
    # Neither is reachable under the gap-fill presets; the table is what makes that a fact.
    "stale_data": "error",
    "ungrounded": "error",
}

# Which rung produced a success, as the method the tier map reads (`_failure_method` for the rest).
_LOOP_METHOD: dict[FetchRoute, str] = {
    "direct": "plain",
    "known_api": "known_api",
    # The hop's target was read as an ordinary page, which is what the driver is being handed.
    "meta_refresh": "plain",
    "impersonate": "impersonate",
    "pdf_local": PDF_LOCAL_METHOD,
    "derived_api": "derived_api",
    "rendered": "rendered",
    "wayback": "wayback",
    "url_context": "document",
}

_NON_PUBLIC_INITIAL_MSG = "Blocked non-public or unsupported URL."
_NON_PUBLIC_HOP_MSG = "Blocked non-public redirect target."
_OVER_CAP_MSG = "Fetch body exceeded the size limit."
_REDIRECT_LIMIT_MSG = "Redirect limit exceeded."
_NO_TEXT_MSG = "Plain fetch returned no extractable text."
_UNDECODABLE_MSG = "Plain fetch could not decode the body as text."
LOCAL_SOURCE_METHOD = "local"
LOCAL_NAVIGATION_METHOD = "local_navigation"


def as_plain_result(result: FetchResult, *, requested_url: str) -> PlainFetchResult:
    """``result`` as the loop's own outcome, with the message the driver reads for it.

    ``requested_url`` is the URL the tool was called with, which is how the two non-public
    refusals are told apart: the ladder refuses the initial URL and a derived hop with the same
    ``ssrf_blocked`` status, and only the URL on the result says which.
    """
    if result.status == "success":
        if result.navigation_only:
            method = LOCAL_NAVIGATION_METHOD
        elif result.local_kind is not None:
            method = LOCAL_SOURCE_METHOD
        else:
            method = "cache" if result.cache_hit else _LOOP_METHOD[result.route]
        return PlainFetchResult(
            status="ok",
            method=method,
            text=result.text,
            links=list(result.links),
            url=result.url,
            content_type=result.content_type,
            escalate_rendered=result.escalate_rendered,
            http_status=result.http_status,
            local_kind=result.local_kind,
            navigation_only=result.navigation_only,
            local_read_refused=result.local_read_refused,
            image_leads=result.image_leads,
        )
    if result.status == "throttled":
        return PlainFetchResult(
            status="throttled",
            method="throttled",
            text="",
            links=[],
            url=result.url,
            content_type=result.content_type,
            http_status=result.http_status,
            throttle_phrase=result.throttle_phrase,
            throttle_chars=result.throttle_chars,
            throttle_method=_LOOP_METHOD[result.route],
        )
    if _needs_a_reader(result):
        return _document_needed_result(result.url, result.content_type or "")
    return PlainFetchResult(
        status=_loop_status(result),
        method=_failure_method(result),
        text=_failure_text(result, requested_url=requested_url),
        links=list(result.links),
        url=result.url,
        content_type=result.content_type,
        escalate_rendered=result.escalate_rendered,
        http_status=result.http_status,
        local_kind=result.local_kind,
        navigation_only=result.navigation_only,
        local_read_refused=result.local_read_refused,
        image_leads=result.image_leads,
    )


def _loop_status(result: FetchResult) -> str:
    """Which of the driver's six statuses ``result`` is, table first and one reason after it.

    A body that arrived as mojibake is ``empty`` rather than the ``error`` its ``unsupported_type``
    would otherwise map to: what we hold is replacement characters rather than the page, and the
    browser's own charset sniffing is the next rung, which is what ``empty`` keeps reachable.
    """
    if result.status == "unsupported_type" and result.status_reason == "undecodable_body":
        return "empty"
    return _LOOP_STATUS[result.status]


def _needs_a_reader(result: FetchResult) -> bool:
    """Whether this is a document only a model read could turn into text.

    A document whose bytes we read and could not decode (a scan, an encrypted or malformed file),
    and a declared image, whose bytes buy nothing a local rung can read.
    """
    if result.local_read_refused:
        return False
    if result.status == "unreadable_document":
        return True
    return result.status == "unsupported_type" and result.status_reason == "image_needs_reader"


def _failure_method(result: FetchResult) -> str:
    """The method a non-read carries: never one the verification-tier map knows."""
    if result.status_reason == "oversize_document":
        return OVERSIZE_DOCUMENT_METHOD
    return "plain"


def _failure_text(result: FetchResult, *, requested_url: str) -> str:
    """What the driver is told about a fetch that read nothing, keyed on the status.

    One message per shape the driver can act on differently: retry later, find another host, use
    ``read_document``, or stop. The ladder's own ``text`` is empty on every non-success, so these
    are the loop's own messages rather than a pass-through.
    """
    if result.status == "ssrf_blocked":
        return _NON_PUBLIC_INITIAL_MSG if result.url == requested_url else _NON_PUBLIC_HOP_MSG
    if result.status == "blocked":
        # No host status means the refusal was ours: a redirect onto a question platform.
        if result.http_status is None:
            return _PLATFORM_FETCH_BLOCK_MSG
        return f"Fetch blocked with HTTP {result.http_status}."
    if result.status in ("js_wall", "empty_body", "no_resolving_content"):
        return _NO_TEXT_MSG
    if result.status == "unsupported_type":
        if result.status_reason == "undecodable_body":
            return _UNDECODABLE_MSG
        return f"Unsupported content type: {result.content_type or 'unknown'}"
    if result.status == "not_found":
        return f"Fetch failed with HTTP {result.http_status}."
    return _error_text(result)


def _error_text(result: FetchResult) -> str:
    """Why an ``error`` read nothing: the shapes the ladder can end an `error` on, in order.

    A withheld archive capture and an ungrounded paid read also land here, both unreachable under
    the gap-fill presets, and both keep the cited host's own status, which is the honest reading.
    """
    if result.status_reason == "oversize_document":
        return oversize_message(result.url)
    if result.http_status in REDIRECT_STATUSES:
        return f"Malformed redirect from {result.url}"
    if result.http_status == 200:
        return _OVER_CAP_MSG
    if result.http_status is not None:
        return f"Fetch failed with HTTP {result.http_status}."
    if result.exc:
        return f"Fetch error: {result.exc}"
    return _REDIRECT_LIMIT_MSG


def method_is_a_document_escalation(method: str) -> bool:
    """Whether ``method`` is the ladder telling ``fetch`` to escalate to ``read_document``."""
    return method == DOCUMENT_NEEDED_METHOD
