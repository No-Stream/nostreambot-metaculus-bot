"""The dispatcher: one URL's direct fetch, then the rungs its outcome earns, in one order.

``fetch_url`` is the entry point both callers use, ``_fetch_one`` is the whole ladder for one
URL, ``_escalate_unresolved`` is the rung order and the browser gate, and ``_run_rung`` is the
bracket that closes each rung's attempts on that rung's own wall and outcome rather than the
ladder's. Nothing here dials anything itself: :mod:`direct_fetch` and :mod:`rungs` own every
request. Why the rungs sit in this order: ``docs/architecture.md``, "The shared fetch ladder".
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import replace
from typing import Any

from metaculus_bot.research.fetch_ladder import context, direct_fetch, guard, run_cache, rungs
from metaculus_bot.research.fetch_ladder.policy import LadderPolicy
from metaculus_bot.research.http_fetch import host_semaphores
from metaculus_bot.research.resolution_fetch_result import FetchResult, FetchRoute, FetchStatus

logger = logging.getLogger(__name__)


async def _run_rung(
    ctx: context.LadderContext, fallback: FetchStatus, rung: Awaitable[FetchResult | None]
) -> FetchResult | None:
    """Await one rung and close the attempts it opened with that rung's own wall and outcome.

    The one home for the bracket every dispatcher site used to copy by hand: read
    ``len(ctx.rungs)`` before the rung runs, await it, then close every attempt opened since
    with the status that stood once it was over — its result's, or ``fallback`` (the status it
    left standing) when it declined (:meth:`LadderContext.close_rungs`). Structural rather than
    stylistic: a rung awaited without the bracket still returned its result, and its attempt fell
    through to :func:`_stamped_with_route`'s last-resort close, which stamps the ladder's FINAL
    status and the whole-ladder wall — the two figures the per-rung close exists to keep apart,
    with the marker parsing either way. ``rung`` is the coroutine created at the call site, which
    runs none of its code until it is awaited here, so the length is read first.
    """
    first_new = len(ctx.rungs)
    result = await rung
    ctx.close_rungs(first_new, fallback if result is None else result.status)
    return result


def _rung_declined_by_policy(ctx: context.LadderContext, rung: FetchRoute, direct: FetchResult, url: str) -> bool:
    """True once a rung this caller does not carry has recorded its skip (``rungs_enabled``)."""
    if rung in ctx.policy.rungs_enabled:
        return False
    ctx.skip_rung(rung, direct.status, url, "rung_not_enabled")
    return True


async def _escalate_via_browser(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult | None:
    """One question's derived-feed-then-browser escalation on this host, under one gate.

    Why the gate is held across the PAIR, and why the fast-path decline sits here rather than
    inside the rung: ``docs/architecture.md`` "Why the rungs sit in this order".
    """
    async with ctx.shared.browser_escalation_gate(url):
        if not _rung_declined_by_policy(ctx, "derived_api", direct, url):
            derived = await _run_rung(
                ctx, direct.status, rungs._derived_api_rung(session, url, direct, host_sems=host_sems, ctx=ctx)
            )
            if derived is not None:
                return derived
        if ctx.fast_path:
            context._skip_for_fast_path(ctx, "rendered", direct, url)
            return None
        if _rung_declined_by_policy(ctx, "rendered", direct, url):
            return None
        return await _run_rung(ctx, direct.status, rungs._rendered_rung(url, direct, host_sems, ctx))


async def _escalate_thin_success(
    url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult:
    """The browser, on a success this caller's verdict judged too thin to be the page.

    Only a caller with a thin-content floor reaches this branch (``escalate_rendered``, which the
    fetcher's verdict never sets), and only the browser is tried: the archive and the paid reader
    answer a page we could not read AT ALL, and this one we did. A render that declines or reads
    nothing leaves the thin text standing, which is the whole point of escalating on a success.
    """
    if ctx.fast_path:
        context._skip_for_fast_path(ctx, "rendered", direct, url)
        return direct
    if _rung_declined_by_policy(ctx, "rendered", direct, url):
        return direct
    async with ctx.shared.browser_escalation_gate(url):
        rendered = await _run_rung(ctx, direct.status, rungs._rendered_rung(url, direct, host_sems, ctx))
    return rendered if rendered is not None else direct


async def _escalate_impersonated(
    session: Any,
    url: str,
    impersonated: FetchResult,
    *,
    host_sems: dict[str, asyncio.Semaphore],
    ctx: context.LadderContext,
) -> FetchResult | None:
    """Apply this caller's browser policy to a body recovered by impersonation."""
    if impersonated.status == "throttled":
        return impersonated
    if impersonated.status == "success":
        if impersonated.escalate_rendered:
            return await _escalate_thin_success(url, impersonated, host_sems=host_sems, ctx=ctx)
        return impersonated
    if rungs._rendered_rung_applies(impersonated):
        return await _escalate_via_browser(session, url, impersonated, host_sems=host_sems, ctx=ctx)
    return None


async def _escalate_offsite(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult:
    """The two rungs whose product is not the live host's bytes: the archive, then the paid read.

    Reached only for the statuses the browser rungs do not claim (``_wayback_rung_applies``). The
    archive's ``stale_data`` withhold is kept as the FALLBACK rather than returned early, so the
    paid rung below is still reachable for that page; the paid rung is asked about the DIRECT
    outcome for the same reason (``docs/architecture.md``, "Why the rungs sit in this order").
    """
    wayback = None
    if not _rung_declined_by_policy(ctx, "wayback", direct, url):
        wayback = await _run_rung(
            ctx, direct.status, rungs._wayback_rung(session, url, direct, host_sems=host_sems, ctx=ctx)
        )
        if wayback is not None and wayback.status == "success":
            return wayback
    if not _rung_declined_by_policy(ctx, "url_context", direct, url):
        read = await _run_rung(
            ctx, direct.status, rungs._url_context_rung(session, url, direct, host_sems=host_sems, ctx=ctx)
        )
        if read is not None:
            return read
    return wayback if wayback is not None else direct


async def _escalate_unresolved(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult:
    """Run the escalation rungs a direct fetch's outcome earns, cheapest first.

    Returns the FIRST rung's rescue, or ``direct`` unchanged when every rung declines or fails; a
    rung that fired and produced nothing still leaves its attempt on the context, and each rung is
    closed the moment its result is known (:func:`_run_rung`) so its attempt carries its own wall
    and outcome rather than the ladder's. A rung the caller's policy does not carry records a
    ``rung_not_enabled`` skip and is not reached. The Wayback withhold as fallback, the
    archive-readable convention behind an attempt with no rescue, and which rungs use ``session``
    at all: ``docs/architecture.md`` "What the dispatcher returns, and what it carries forward".
    """
    if direct.status == "throttled":
        return direct
    if _terminal_local_read(direct):
        return direct
    if direct.status == "success":
        if not direct.escalate_rendered:
            return direct
        return await _escalate_thin_success(url, direct, host_sems=host_sems, ctx=ctx)
    # First because it is free and its triggers are disjoint from the browser's (see the doc).
    if not _rung_declined_by_policy(ctx, "impersonate", direct, url):
        impersonated = await _run_rung(
            ctx, direct.status, rungs._impersonate_rung(url, direct, host_sems=host_sems, ctx=ctx)
        )
        if impersonated is not None:
            escalated = await _escalate_impersonated(session, url, impersonated, host_sems=host_sems, ctx=ctx)
            if escalated is not None:
                return escalated
    if rungs._rendered_rung_applies(direct):
        escalated = await _escalate_via_browser(session, url, direct, host_sems=host_sems, ctx=ctx)
        if escalated is not None:
            return escalated
    return await _escalate_offsite(session, url, direct, host_sems=host_sems, ctx=ctx)


async def _fetch_one(
    session: Any, url: str, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext | None = None
) -> FetchResult:
    """Fetch a single URL directly, then escalate what the direct route could not read.

    ``ctx`` carries the question text a PDF digest ranks passages against, the wall-clock
    origin each rung bounds itself with, and the rung attempts stamped onto the returned
    result. It defaults to a fresh one so the fetch surface can still be driven with three
    arguments, which is what every existing caller and test does.
    """
    ctx = context.LadderContext() if ctx is None else ctx
    direct = await direct_fetch._fetch_direct(session, url, host_sems, ctx)
    # The direct fetch's own rungs are over, and its status is what they left standing.
    ctx.close_rungs(0, direct.status)
    escalated = await _escalate_unresolved(session, url, direct, host_sems=host_sems, ctx=ctx)
    if direct.chrome_metric_withheld:
        # A fact about this URL's ladder rather than about one result (see the doc).
        escalated.chrome_metric_withheld = True
    return context._stamped_with_route(escalated, ctx)


def _terminal_local_read(result: FetchResult) -> bool:
    return result.local_read_refused or result.local_kind in ("archive", "workbook", "word")


def _store_complete_read(url: str, result: FetchResult, ctx: context.LadderContext) -> None:
    capture = ctx.read_capture_for(result)
    if capture is None or result.route == "url_context":
        return
    complete_source = isinstance(capture.artifact, run_cache.SourceRead)
    if (result.status == "success" or complete_source) and run_cache.cacheable(capture.artifact):
        run_cache.put(url, capture.artifact, route=capture.route)


async def _try_known_api(url: str, ctx: context.LadderContext) -> FetchResult | None:
    """Run rung 0 inside the current URL's remaining wall, or decline it."""
    callback = ctx.policy.known_api
    if callback is None:
        return None
    remaining_s = ctx.rung_budget_s()
    if remaining_s <= 0:
        return None
    try:
        return await asyncio.wait_for(callback(url), timeout=remaining_s)
    except TimeoutError:
        logger.warning("known_api rung exceeded the remaining wall budget for %s", url)
        return None


async def fetch_url(url: str, *, policy: LadderPolicy, ctx: context.LadderContext) -> FetchResult:
    """Fetch one URL through the whole ladder under ``policy``: the entry point both callers use.

    Rung 0 first (``policy.known_api``, a public API that answers this URL exactly, with no page
    fetch at all), then the ladder proper. The policy is bound onto the context here rather than
    threaded through the rungs, so every rung reads ``ctx.policy`` and no rung signature carries
    a second argument.

    ``ctx.session`` and ``ctx.host_sems`` are how a caller that already holds a session and the
    process-wide politeness map keeps them: with a session per URL the fetcher's connector limits
    would change. A context naming neither gets a session opened and closed for this URL alone
    and the process-wide map, which is what a caller with one fetch in hand wants.
    """
    host_sems = ctx.host_sems if ctx.host_sems is not None else host_semaphores()
    bound = replace(ctx, policy=policy, host_sems=host_sems)
    translated = await _try_known_api(url, bound)
    if translated is not None:
        return translated
    try:
        cached = await run_cache.get(
            url,
            policy=policy,
            query=ctx.query,
            now=ctx.now,
            budget_s=bound.rung_budget_s(),
        )
    except TimeoutError:
        logger.warning("fetch cache presentation exceeded this URL's remaining wall budget: %s", url)
        return FetchResult(
            url=url,
            status="error",
            text="",
            http_status=None,
            content_type=None,
            cache_hit=True,
        )
    if cached is not None:
        if cached.route not in ("direct", "meta_refresh") or (
            cached.status == "success" and not cached.escalate_rendered
        ):
            return cached
        if bound.session is not None:
            escalated = await _escalate_unresolved(bound.session, url, cached, host_sems=host_sems, ctx=bound)
        else:
            async with guard._get_session() as session:
                escalated = await _escalate_unresolved(session, url, cached, host_sems=host_sems, ctx=bound)
        result = context._stamped_with_route(escalated, bound)
        _store_complete_read(url, result, bound)
        return result
    if bound.session is not None:
        result = await _fetch_one(bound.session, url, host_sems, bound)
    else:
        async with guard._get_session() as session:
            result = await _fetch_one(session, url, host_sems, replace(bound, session=session))
    _store_complete_read(url, result, bound)
    return result
