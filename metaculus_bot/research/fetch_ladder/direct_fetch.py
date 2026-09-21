"""The plain fetch every rung is measured against: one bounded redirect loop, one hop at a time.

``_fetch_direct`` is the ladder's own baseline route and also the transport three later rungs
borrow: the derived-feed GET, the Wayback snapshot and the paid rung's robots pre-check each issue
their request through it, which is what gives a derived URL the same SSRF preflight, per-host gate
and budget-clamped hop timeout a cited one gets. It sits UNDER the rungs rather than beside the
dispatcher for exactly that reason, so the package's dependencies run one way.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

import aiohttp

from metaculus_bot.constants import RESOLUTION_SOURCE_HTTP_TIMEOUT, RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S
from metaculus_bot.research.fetch_ladder import classify, context, guard
from metaculus_bot.research.http_fetch import MAX_REDIRECTS
from metaculus_bot.research.resolution_fetch_result import FetchResult

logger = logging.getLogger(__name__)


async def _fetch_one_hop(
    session: Any, current_url: str, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult | str:
    """ONE GET against ``current_url`` under its host semaphore: terminal result or next URL.

    The request's timeout is the REMAINING wall budget rather than the session's flat
    ``RESOLUTION_SOURCE_HTTP_TIMEOUT``, and it is computed AFTER the semaphore is acquired so
    a hop that queued behind a slow host does not then help itself to a fresh 20 s. This is
    the one choke point every hop passes through — the initial GET, each 3xx hop and the
    meta-refresh hop — so clamping here is what makes the budget arithmetic the rest of the
    ladder does actually bind: a hop admitted with 3 s left (the meta-refresh rung's floor)
    could otherwise run the full 20 s, overshoot ``RESOLUTION_SOURCE_WALL_TIMEOUT`` and let
    the provider's outer ``wait_for`` discard every sibling page that had already fetched.
    Monotonically <= the old 20 s, and an expiry lands on the existing ``TimeoutError`` path,
    so overrunning costs this one URL rather than the question.

    BOTH ``ClientTimeout`` fields are set because a per-request timeout REPLACES the
    session's wholesale rather than merging with it.

    A cited PDF is the one branch whose work does NOT finish inside the two contexts: it
    comes back as a :class:`_PendingDocument` and is parsed after both have exited, because
    that parse is seconds of CPU and the host gate is loop-wide (see
    :class:`_PendingDocument`). The HTML branch's ``to_thread`` hops still run inside the
    semaphore and the open response, and since the extractor policy they can be TWO
    trafilatura passes rather than one: the second is skipped under
    ``RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S`` of remaining wall, which bounds the
    worst case without moving the work. Moving it would trade a measured hazard for an
    unmeasured restructure (FUTURE.md carries the entry); the meta-refresh hop that follows
    the classification needs the decoded text inside this loop either way.
    """
    async with guard._sem_for_host(host_sems, current_url):
        hop_timeout_s = min(
            RESOLUTION_SOURCE_HTTP_TIMEOUT, max(ctx.rung_budget_s(), RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S)
        )
        try:
            async with session.get(
                current_url,
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=hop_timeout_s, sock_read=hop_timeout_s),
            ) as resp:
                outcome = await classify._resolution_response_outcome(resp, current_url, ctx)
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.info(f"resolution_source fetch error for {current_url}: {type(e).__name__}: {e}")
            return FetchResult(
                url=current_url,
                status="error",
                text="",
                http_status=None,
                content_type=None,
                failure_class=classify._network_failure_class(e),
                exc=type(e).__name__,
            )
    if isinstance(outcome, classify._PendingDocument):
        return await classify._finish_document(outcome, ctx)
    if isinstance(outcome, classify._PendingSource):
        return await classify._finish_source(outcome, ctx)
    return outcome


async def _fetch_direct(
    session: Any, url: str, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult:
    """Fetch a single URL directly, holding the per-host politeness semaphore hop by hop.

    Content-type routing:
      * HTML → ARIA-table rewrite + trafilatura extraction (via to_thread), the
        inline-chart rung, then the chrome / JS-wall checks and the meta-refresh hop.
      * JSON → capped raw body, no pretty-print (the data IS the content).
      * text/plain, text/csv → capped raw body.
      * anything else, including a missing/empty Content-Type header → capped read,
        then the ``%PDF-`` magic check: a document is read locally and rendered as a
        query-relevant digest, and anything else is ``unsupported_type`` as before.

    Politeness: each hop acquires the semaphore for THAT hop's host around its single GET,
    the body read on a terminal response, and the HTML branch's extraction, and releases it
    before following a redirect. A cited PDF's parse is the one thing deliberately outside
    the hold — it comes back as a ``_PendingDocument`` and is parsed after the semaphore is
    released, because the gate is loop-wide and the parse is seconds of CPU. Keying per hop
    — not on the original URL's host — preserves one-request-per-host when chains from
    different initial hosts converge on the same final host; the strict per-hop
    acquire/release pairing means an A→B→A chain never re-acquires a semaphore it still
    holds (asyncio semaphores are not reentrant).

    SSRF guard: rejects non-public URLs (private / loopback / link-local IPs,
    userinfo tricks, non-http(s) schemes) BEFORE any network I/O and again on
    every hop target, whether it came from a ``Location`` header or a meta-refresh
    tag (:func:`_hop_refusal` is the one place both checks live, and
    :func:`_vetted_hop_target` maps its verdict onto the terminal result). The
    connect-time :class:`FilteringResolver` (see :func:`_get_session`) provides the
    actual DNS-rebinding boundary; these preflight checks are fast-fail
    observability so we surface ``ssrf_blocked`` without opening a session. Hops of
    both shapes are followed in-band and share the one ``MAX_REDIRECTS`` cap.

    No retries (Tier 1 anti-goal). Any aiohttp/asyncio error becomes ``error``. Escalation
    beyond this route is :func:`_escalate_unresolved`'s job, so this function stays exactly
    what it always was: the plain fetch, terminal on its own outcome.
    """
    # Guard the initial URL before any network I/O.
    if not await guard.is_public_http_url(url):
        logger.warning(f"resolution_source ssrf_blocked (initial url): {urlparse(url).netloc}")
        return FetchResult(
            url=url,
            status="ssrf_blocked",
            text="",
            http_status=None,
            content_type=None,
        )

    current_url = url
    # Bounded redirect loop. Each iteration issues ONE GET with
    # allow_redirects=False under the current hop's host semaphore; a redirect
    # status (or a meta-refresh stub) resolves the next URL, re-guards, and loops
    # (each hop releases its semaphore before the next acquires its own — no
    # nesting, so no self-deadlock on revisited hosts).
    # Non-redirect responses fall through to the content-type routing below.
    for _hop in range(MAX_REDIRECTS + 1):
        outcome = await _fetch_one_hop(session, current_url, host_sems, ctx)
        if isinstance(outcome, FetchResult):
            return outcome
        current_url = outcome

    # Fell out of the loop -> exceeded MAX_REDIRECTS.
    logger.info(f"resolution_source redirect chain exceeded {MAX_REDIRECTS} hops (final={current_url})")
    return FetchResult(
        url=current_url,
        status="error",
        text="",
        http_status=None,
        content_type=None,
    )
