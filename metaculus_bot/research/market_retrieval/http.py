"""Everything between the socket and a typed row: the bounded GET, the body caps, the
per-query failure accounting, and the scalar coercions the venue parsers apply to
untrusted JSON.

Split out of ``venues.py`` because none of it is venue-specific — the four venue paths sit
on top of this one transport layer — and because the coercions have the widest reuse in the
package. Built on ``research/http_fetch.py``, which owns the session factory and the
streaming body reads this module's caps are enforced with.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiohttp

from metaculus_bot.research.http_fetch import read_body_capped, read_body_snippet
from metaculus_bot.research.market_retrieval.types import MarketMatch, SettlementSource, _FetchTally
from metaculus_bot.run_status import RUN_STATUS

logger = logging.getLogger(__name__)

# Per-HTTP-call cap. The snapshot-level wall sits outside it.
PLATFORM_HTTP_TIMEOUT = 10.0

# One bounded backoff between retries. Two attempts is the measured sufficiency: the venue
# fan-out already issues several queries per venue, so a persistent outage is visible
# without a third attempt and a transient one is covered by the second.
HTTP_RETRY_BACKOFF_SECS = 0.5

# Hard cap on one buffered response body (the search endpoints don't paginate).
MAX_RESPONSE_BYTES = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# Scalar coercions over untrusted JSON
# ---------------------------------------------------------------------------


def safe_float(value: Any) -> float | None:
    # The isfinite guard is the same one `safe_int` needs and for a sharper reason: json.loads
    # accepts bare NaN/Infinity literals, and NaN defeats every comparison in `_liquidity_label`,
    # so a row whose volume arrived as NaN falls through to the strongest label and renders
    # `signal=deep` — presenting missing data to a forecaster as the best possible liquidity
    # evidence. None renders `no-liquidity-data`, which is what a missing figure means.
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def parse_iso(value: Any) -> datetime | None:
    """An ISO timestamp as a UTC-AWARE datetime, or None.

    Normalized HERE, at the one boundary all four venue parsers pass through, so no comparison
    site downstream has to remember: ``fromisoformat`` returns an aware datetime for the ``Z``
    form and a NAIVE one for a bare timestamp or a date-only string, and mixing the two makes
    ``max(closes)`` over an event's nested markets raise ``TypeError: can't compare offset-naive
    and offset-aware datetimes``. That runs inside `to_thread` with no guard, so it reaches the
    snapshot-level net and zeroes ALL FOUR venues for the question — and the offending event
    stays in the 6h catalogue cache, so every later question repeats it.

    A naive value is TREATED as UTC — the same assumption ``assemble_pool``'s ``as_of`` makes.
    Not an assertion that every venue publishes UTC: PredictIt's ``dateEnd`` is historically
    Eastern, and only the date-granular rendering makes that <=5h skew immaterial. Behaviour is
    unchanged for rendering, since attaching a tzinfo does not shift the wall clock.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_iso_guarded(value: Any) -> datetime | None:
    """``parse_iso`` behind a ``YYYY-MM-DD`` prefix guard, for a field with sentinels.

    PredictIt's ``dateEnd`` ships the literal ``"NA"`` and ``"N/A"`` no-close sentinels, which
    the prefix guard rejects outright — nothing without a real calendar date in front reaches
    the parser. (7-digit fractional seconds, the other odd shape in that dump, need no help:
    ``fromisoformat`` has accepted arbitrary fraction lengths since 3.11 and truncates to
    microseconds.)

    Anything else with a valid date prefix but an unparseable tail — a named-zone suffix, an
    out-of-range field — falls back to the DATE alone rather than to nothing. That is all the
    ranker needs to tell same-window from different-window, so the truncation loses nothing
    and the field never raises.
    """
    text = str(value or "").strip()
    if len(text) < 10 or text[4] != "-" or text[7] != "-":
        return None
    # Calendar-date field slices, not data sampling.
    if not (text[:4].isdigit() and text[5:7].isdigit() and text[8:10].isdigit()):  # HARNESS-SCAN-EXEMPT-subsampling
        return None
    return parse_iso(text) or parse_iso(text[:10])  # HARNESS-SCAN-EXEMPT-subsampling


def settlement_sources(raw: Any) -> tuple[SettlementSource, ...]:
    """A venue's ``[{name, url}, ...]`` block, verbatim, as typed rows.

    Only the two fields, so the raw-research archive's shape cannot silently pick up
    whatever else the venue starts shipping in that block.
    """
    if not isinstance(raw, list):
        return ()
    out: list[SettlementSource] = []
    for source in raw:
        if not isinstance(source, dict):
            continue
        name = str(source.get("name") or "").strip()
        url = str(source.get("url") or "").strip()
        if name or url:
            out.append(SettlementSource(name=name, url=url))
    return tuple(out)


# ---------------------------------------------------------------------------
# Bounded GET
# ---------------------------------------------------------------------------


async def read_json_capped(resp: Any, label: str) -> Any | None:
    """Parse a body as JSON, rejecting anything over MAX_RESPONSE_BYTES.

    ONE read path, deliberately: the bounded stream off ``resp.content``, never
    ``resp.json()``. A second branch keyed on which methods the response happens to expose
    would decide the production read by what a test double implements — and it did: gating on
    ``.read`` sent every stubbed venue response down an uncapped ``.json()``, leaving the one
    thing standing between a runaway upstream and the heap unreachable from any test.

    Returns None on a decode failure or an oversized body; the caller logs.
    """
    raw = await read_body_capped(resp, max_bytes=MAX_RESPONSE_BYTES, label=label)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
        logger.warning(f"{label} JSON decode failed: {exc}")
        return None


@dataclass(frozen=True, slots=True)
class _RetryableStatus:
    """One attempt's verdict when the status says "ask again": the status, and nothing else.

    A sentinel rather than ``None`` because ``None`` is already the *give-up* answer a decoded
    body can legitimately be, so the retry decision has to be a distinct type or the two collapse.
    """

    status: int


@dataclass(frozen=True, slots=True)
class _NonRetryableStatus:
    """One terminal HTTP status that should be returned to the caller as no data."""

    status: int


async def _read_one_attempt(
    session: Any,
    url: str,
    params: dict[str, str],
    *,
    client_timeout: aiohttp.ClientTimeout,
    retryable: set[int],
    label: str,
) -> Any | _RetryableStatus | _NonRetryableStatus | None:
    """One GET: the parsed body, ``None`` on a non-retryable outcome, or a ``_RetryableStatus``.

    Split out of ``http_get_with_backoff`` so the retry bookkeeping and the response reading are
    each one level of nesting deep. One incidental improvement falls out: the response is released
    when this returns, so a retryable status no longer holds it open across the backoff sleep.
    """
    async with session.get(url, params=params, timeout=client_timeout) as resp:
        status = resp.status
        if status in retryable or status >= 500:
            return _RetryableStatus(status)
        if status != 200:
            snippet = await read_body_snippet(resp)
            logger.warning(f"{label} HTTP {status} non-retryable: {snippet}")
            return _NonRetryableStatus(status)
        return await read_json_capped(resp, label)


async def http_get_with_backoff(
    session: Any,
    url: str,
    params: dict[str, str],
    *,
    max_attempts: int,
    retryable_statuses: Iterable[int] | None = None,
    label: str,
    component: str | None = None,
) -> Any | None:
    """GET ``url`` with ``max_attempts`` and one bounded backoff between retries.

    Returns the parsed JSON body on 200, or None on retry exhaustion / a non-200. Caps the
    body at MAX_RESPONSE_BYTES so a runaway upstream cannot blow up memory.

    Elapsed time is bounded by ``max_attempts`` alone — ``max_attempts * PLATFORM_HTTP_TIMEOUT``
    plus ``(max_attempts - 1)`` backoffs, which is what ``_HTTP_STAGE_WORST`` in the seam
    module computes and folds into the snapshot budget. A separate cumulative-sleep budget
    would add nothing over that: at every call site here the backoff total is one 0.5s sleep.

    ``retryable_statuses`` defaults to (403, 429, 500, 502, 503, 504); anything >= 500 is
    retryable regardless.
    """
    retryable: set[int] = set(retryable_statuses or (403, 429, 500, 502, 503, 504))
    timeout = aiohttp.ClientTimeout(total=PLATFORM_HTTP_TIMEOUT, sock_read=PLATFORM_HTTP_TIMEOUT)
    if component is not None:
        RUN_STATUS.record_source_lookup_attempt(component)

    for attempt in range(max_attempts):
        try:
            outcome = await _read_one_attempt(
                session, url, params, client_timeout=timeout, retryable=retryable, label=label
            )
        except (TimeoutError, aiohttp.ClientError) as exc:
            if await _retry_transport_error(exc, attempt=attempt, max_attempts=max_attempts, label=label):
                continue
            _record_transport_error(component, exc)
            return None
        should_retry, response = await _process_fetch_outcome(
            outcome,
            attempt=attempt,
            max_attempts=max_attempts,
            label=label,
            component=component,
        )
        if should_retry:
            continue
        return response
    return None


async def _retry_transport_error(
    exc: TimeoutError | aiohttp.ClientError, *, attempt: int, max_attempts: int, label: str
) -> bool:
    if attempt + 1 >= max_attempts:
        logger.warning(f"{label} transient error after {attempt + 1} attempts: {exc}")
        return False
    logger.warning(
        f"{label} transient error: {exc}; retry {attempt + 2}/{max_attempts} after {HTTP_RETRY_BACKOFF_SECS:.2f}s"
    )
    await asyncio.sleep(HTTP_RETRY_BACKOFF_SECS)
    return True


async def _process_fetch_outcome(
    outcome: Any | _RetryableStatus | _NonRetryableStatus | None,
    *,
    attempt: int,
    max_attempts: int,
    label: str,
    component: str | None,
) -> tuple[bool, Any | None]:
    if isinstance(outcome, _RetryableStatus):
        if attempt + 1 >= max_attempts:
            logger.warning(f"{label} HTTP {outcome.status} after {attempt + 1} attempts; giving up")
            if component is not None:
                RUN_STATUS.record_source_http_failure(component, http_status=outcome.status)
            return False, None
        logger.warning(
            f"{label} HTTP {outcome.status}; retry {attempt + 2}/{max_attempts} after {HTTP_RETRY_BACKOFF_SECS:.2f}s"
        )
        await asyncio.sleep(HTTP_RETRY_BACKOFF_SECS)
        return True, None
    if isinstance(outcome, _NonRetryableStatus):
        if component is not None:
            RUN_STATUS.record_source_http_failure(component, http_status=outcome.status)
        return False, None
    if outcome is None and component is not None:
        RUN_STATUS.record_source_failure(component, error_type="provider_failure")
    return False, outcome


def _record_transport_error(component: str | None, exc: TimeoutError | aiohttp.ClientError) -> None:
    if component is None:
        return
    error_type = "timeout" if isinstance(exc, TimeoutError) else "connection_error"
    RUN_STATUS.record_source_failure(component, error_type=error_type)


def flatten_results(results: Sequence[Any], platform: str) -> tuple[list[MarketMatch], _FetchTally]:
    """Flatten per-query search results and tally how many sub-queries were lost.

    A leaf search signals upstream failure with ``None`` and a successful-but-empty parse
    with ``[]``, so anything that is not a list — a raised task or a ``None`` — counts as a
    lost sub-query rather than being logged and discarded. One venue's 403 on one long
    query therefore degrades THAT QUERY and nothing else: not the venue's other queries,
    and not the question.
    """
    out: list[MarketMatch] = []
    queries_ok = 0
    queries_failed = 0
    for result in results:
        if isinstance(result, list):
            out.extend(result)
            queries_ok += 1
        elif isinstance(result, Exception):
            logger.warning(f"{platform} query task raised: {result}")
            queries_failed += 1
        else:
            queries_failed += 1
    return out, _FetchTally(queries_ok, queries_failed)
