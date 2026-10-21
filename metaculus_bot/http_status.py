"""Shared reader for the HTTP status forecasting-tools hides behind a re-raise.

ft's ``raise_for_status_with_additional_info`` re-raises a fresh, message-only
``requests.HTTPError`` (no ``response`` attached) and chains the original via
``raise ... from e``. Both MetaculusClient-patching modules — ``fetch_hardening``
and ``publish_hardening`` — need the real status to make their (deliberately
different) retry decisions, so the chain walk lives here ONCE; a correction to it
cannot land on one surface only.
"""

from __future__ import annotations

import http.client
import re
from collections.abc import Iterator

from requests import exceptions as req_exc
from urllib3 import exceptions as ul3_exc

# ft's message-only HTTPError embeds the status as literal text ("Status code:
# 405"). Anchored on that phrase so an unrelated 3-digit number in a server
# message can't be misread as a status.
_STATUS_IN_MESSAGE = re.compile(r"Status code:\s*(?P<status>\d{3})\b")

# ft wraps at most once, so a short walk suffices; the bound also guards against
# a pathological self-referencing chain.
_CAUSE_CHAIN_DEPTH = 4


def iter_cause_chain(exc: BaseException, max_depth: int = _CAUSE_CHAIN_DEPTH) -> Iterator[BaseException]:
    """``exc`` plus its ``__cause__`` links, bounded and cycle-safe."""
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(max_depth):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        yield current
        current = current.__cause__


def http_status_from_exception(exc: BaseException) -> int | None:
    """Best-effort HTTP status for a requests-shaped exception, or None.

    Prefers a typed ``http_status`` attribute, then a real ``response.status_code``
    anywhere in the cause chain, then ft's own message text.
    """
    for cause in iter_cause_chain(exc):
        status = getattr(cause, "http_status", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
        status = getattr(getattr(cause, "response", None), "status_code", None)
        if isinstance(status, int):
            return status
    match = _STATUS_IN_MESSAGE.search(str(exc))
    return int(match.group("status")) if match else None


# Transport-level failures worth another attempt regardless of any HTTP status.
# ProtocolError/RemoteDisconnected sit beside requests' own types because requests
# re-raises urllib3's (and httplib's) exceptions unwrapped.
_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    req_exc.ConnectionError,
    req_exc.Timeout,
    ul3_exc.ProtocolError,
    http.client.RemoteDisconnected,
)

_TRANSIENT_FETCH_STATUSES = frozenset({429, 500, 502, 503, 504})

# Consulted only when no status could be read anywhere: a statusless transient
# failure almost always self-describes as a throttle or a timeout.
_TRANSIENT_MESSAGE_TOKENS = ("too many requests", "timed out", "timeout")


def is_transient_question_fetch_error(exc: BaseException) -> bool:
    """True when a question-fetch failure is worth another attempt.

    This is the BENCHMARK/BACKTEST outer-loop retry policy (community_benchmark's
    and backtest/question_prep's fetch loops), deliberately WITHOUT the 403
    carve-out — that belongs to the hardened prod fetch path
    (``fetch_hardening._is_retryable``), and the two policies must not be
    mistaken for one.

    Transport errors anywhere in the cause chain are retryable. Otherwise the
    decision keys on the real status via :func:`http_status_from_exception`
    (which reads ``response.status_code`` or ft's anchored "Status code: NNN"
    message text) — never on bare digits in the message, which ft's re-raise
    pollutes with URLs, question ids, and echoed response bodies. Only a fully
    statusless exception falls back to a narrow throttle/timeout text check.
    """
    if any(isinstance(cause, _TRANSPORT_ERRORS) for cause in iter_cause_chain(exc)):
        return True
    status = http_status_from_exception(exc)
    if status is not None:
        return status in _TRANSIENT_FETCH_STATUSES
    message = str(exc).lower()
    return any(token in message for token in _TRANSIENT_MESSAGE_TOKENS)
