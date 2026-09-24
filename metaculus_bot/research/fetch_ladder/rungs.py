"""The escalation rungs a direct fetch's outcome earns, cheapest first, each one self-bounding.

One block per rung: its trigger predicate, the gates it clears before spending anything, the
transport call, and the classification of whatever came back. The impersonated retry, the
derived-API feed, the headless-Chromium render, the Wayback snapshot and the paid Gemini read all
live here; a rung declines by returning None, and the outcome it was handed then stands.
:mod:`ladder` is what orders them and closes the attempts they open.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import replace
from typing import Any
from urllib.parse import urlparse

from metaculus_bot.constants import (
    DOCUMENT_TEXT_PDF_MAX_BYTES,
    GAP_FILL_V2_READER_MODEL,
    GAP_FILL_V2_READER_THINKING_LEVEL,
    GOOGLE_API_KEY_ENV,
    RESOLUTION_SOURCE_DERIVED_API_MIN_BUDGET_S,
    RESOLUTION_SOURCE_HTTP_TIMEOUT,
    RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
    RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S,
    RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS,
    RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV,
    RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS,
    RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S,
    RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS,
    RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S,
    RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS,
    env_flag_enabled,
)
from metaculus_bot.research import derived_api, impersonated_fetch, resolution_presentation
from metaculus_bot.research.fetch_ladder import classify, context, direct_fetch, guard, run_cache
from metaculus_bot.research.fetch_ladder.policy import LadderPolicy
from metaculus_bot.research.http_fetch import decode_text_body
from metaculus_bot.research.impersonated_fetch import (
    ImpersonateBudgetExhausted,
    ImpersonateDeclined,
    ImpersonatedResponse,
    ImpersonateTransportError,
    ImpersonateUnpinnable,
    fetch_impersonated,
)
from metaculus_bot.research.rendered_fetch import (
    RENDER_EXIT_RESERVE_MS,
    RENDER_SETTLE_MS,
    RENDER_TIMEOUT_MS,
    MemoScope,
    RenderBudgetExpired,
    RenderDomOverCeiling,
    RenderedPage,
    RenderOffHost,
    RenderTimeout,
    is_json_content_type,
    note_rendered_no_text,
    render_page,
    rendered_to_nothing,
)
from metaculus_bot.research.resolution_fetch_result import (
    _NON_OK_FETCH_STATUS,
    FetchResult,
    FetchStatus,
    FetchStatusReason,
    RungAttempt,
    server_header_token,
    vacuous_body_status,
)
from metaculus_bot.research.robots_policy import ROBOTS_FETCH_TIMEOUT_S, google_extended_blocks_url
from metaculus_bot.research.url_context_reader import NOT_ADDRESSED_SENTINEL, run_url_context_read
from metaculus_bot.research.wayback import (
    WaybackSnapshot,
    innermost_url,
    parse_snapshot_url,
    snapshot_age_days,
    wayback_lead,
    wayback_snapshot_url,
)

logger = logging.getLogger(__name__)


def _impersonate_rung_applies(direct: FetchResult) -> bool:
    """Whether a browser's TLS fingerprint could plausibly turn ``direct`` into a readable page.

    ``blocked`` with an HTTP status in ``impersonated_fetch.IMPERSONATE_TRIGGER_STATUSES`` (403
    today), and both halves are load-bearing. ``blocked`` has four producers:
    ``_NON_OK_FETCH_STATUS`` maps 403, 406 and 429 to it, and :func:`_vetted_hop_target` returns
    it for a Metaculus self-reference hop carrying the REDIRECT's 301 or 302. The 403 test
    excludes that last case exactly, which matters: handing a URL this module refused to a second
    transport is the bypass the guard exists to prevent.

    The trigger set is the TRANSPORT's, read as a module attribute at call time rather than
    imported, for two reasons. It is the one trigger both fetchers share (gap-fill v2's ``fetch``
    and ``read_document`` ladders key on the same object), so the population cannot drift between
    them; and the test packages empty that attribute to decline the rung by default and restore the
    transport's own object in the tests that exercise it, which only bites on a read that resolves
    the name at call time.

    Excluded on purpose, each for its own reason. 429 is a throttle, not a fingerprint verdict,
    and retrying at once with a different fingerprint against a host that just asked us to slow
    down is the one shape where the retry could make our position worse; the 2026-09-04
    diagnostic measured impersonation helping only on 403s. 406 is a content-negotiation refusal,
    and impersonation changes the ``Accept`` headers as a side effect, so a 406 rung would be an
    untested guess. 401 is an authentication requirement no fingerprint changes, and is not even a
    ``blocked`` shape: absent from ``_NON_OK_FETCH_STATUS``, it falls through to ``error``. A 200
    carrying a challenge or throttle interstitial is not a representable trigger today, because
    Tier 1 has no throttle-phrase check (only gap-fill v2's ``fetch_outcomes`` has one) and such a
    page classifies as ``js_wall`` or ``thin_page``; FUTURE.md carries that entry, and the
    ``error`` with ``failure_class="tls"`` widening nothing has measured.
    """
    return direct.status == "blocked" and direct.http_status in impersonated_fetch.IMPERSONATE_TRIGGER_STATUSES


async def _impersonated_body_outcome(response: ImpersonatedResponse, ctx: context.LadderContext) -> FetchResult:
    """Classify a body the impersonated retry read, through the ladder's ONE body router.

    :func:`classify._classify_body` is the same router a direct 200 takes, so a rescued page is
    indistinguishable downstream from a directly-fetched one and a route the caller's verdict
    gains reaches this rung with it rather than needing a second copy here.

    ``http_status`` is the IMPERSONATED response's 200, not the direct 403. The bytes came with a
    200 and that is the honest record: a rescue's fetch line reads ``status=ok http=200
    route=impersonate`` with no ``failure_class``, and the fact that the direct fetch was refused
    lives on the escalation line's ``from_status=blocked``, exactly as a Wayback rescue reports the
    snapshot's own status. This diverges from :func:`_rendered_rung`, which passes the direct
    status because there the direct GET also got a 200.

    No meta-refresh hop. :func:`_resolution_html_outcome` runs one after a no-content
    classification; following it from here would mean deciding which transport dials the target
    and re-entering the whole hop loop from inside a rung, so this router calls
    :func:`_classify_html_body` directly (FUTURE.md carries the entry).

    The page's own context, not :func:`_aux_ctx`. The Wayback and derived-feed rungs use the
    child context because they fetch a DIFFERENT URL on the page's behalf and must not let that
    URL's inner rungs hijack the page's route. These bytes are the cited page's own, so a
    ``pdf_local`` attempt belongs on the page's record, and a document rescue reads
    ``route=pdf_local``: the accounting a meta-refresh hop onto a PDF already produces, in the
    file's own words "the hop got us the bytes, the local read is what the text came from".
    """
    # `_classify_body` rather than its hop-following sibling: this rung owns no redirect loop, and
    # following one would mean deciding which transport dials the target (FUTURE.md has the entry).
    outcome = await classify._classify_body(
        response.body, response.url, response.content_type, ctx, http_status=response.status
    )
    if isinstance(outcome, classify._PendingDocument):
        return await classify._finish_document(outcome, ctx)
    if isinstance(outcome, classify._PendingSource):
        return await classify._finish_source(outcome, ctx)
    return outcome


async def _impersonate_or_record_the_skip(
    retry_url: str, budget_s: float, host_sems: dict[str, asyncio.Semaphore], attempt: RungAttempt
) -> ImpersonatedResponse | None:
    """Dial the transport, or turn its decline into the record the attempt should carry.

    Two declines are SKIPS rather than fired attempts, each stamped on the attempt already started
    rather than appended as a second one, the pattern :func:`_render_or_record_the_skip` uses for
    its skips. :class:`ImpersonateUnpinnable` is a hop whose host would not pin to a vetted public
    address; the pin can fail on the FIRST hop, where nothing was dialed, or on a later redirect
    hop, where the earlier hops were, so the skip says the pin failed on some hop, not that no
    wall was spent. :class:`ImpersonateBudgetExhausted` is the wall running out while the
    transport waited on a pre-dial await (the vetting lookup, a redirect re-guard, the host gate):
    nothing was dialed on that hop, so it is the ``wall_budget`` skip the pre-gate floor records,
    and not a fired attempt whose ``blocked`` outcome would read as the host refusing the
    fingerprint. Every other :class:`ImpersonateDeclined` leaves the attempt fired, so the
    dispatcher closes it on the direct status and the archive reads ``route=impersonate
    status=blocked``: we tried the fingerprint and this is still the answer. Logged at the level
    the shape deserves. A transport failure at INFO, as the direct path's own are, because a reset
    or a handshake failure is a fact about the host rather than about this rung; the spent
    budget at WARNING like every other wall skip; a refused hop, an oversized body, a redirect
    chain past the cap or a pin that did not hold at WARNING, and the transport already logged
    that last one at ERROR.

    The two body caps are the direct path's own: ``RESOLUTION_SOURCE_MAX_RESPONSE_BYTES`` for a
    page and ``DOCUMENT_TEXT_PDF_MAX_BYTES`` for a declared PDF, exactly the pair
    :func:`_resolution_pdf_outcome` reads under, so a cited PDF between the two is read on this
    rung as the direct fetch would have read it rather than declined as oversized.
    """
    netloc = urlparse(retry_url).netloc
    try:
        return await fetch_impersonated(
            retry_url,
            host_sems=host_sems,
            deadline_monotonic_s=time.monotonic() + budget_s,
            per_hop_timeout_s=RESOLUTION_SOURCE_HTTP_TIMEOUT,
            max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
            document_max_bytes=DOCUMENT_TEXT_PDF_MAX_BYTES,
        )
    except ImpersonateUnpinnable as exc:
        logger.warning("resolution_source: the impersonated retry of %s could not pin its host (%s)", netloc, exc)
        attempt.skipped_reason = "impersonate_unpinnable"
    except ImpersonateBudgetExhausted as exc:
        logger.warning("resolution_source: skipping the impersonated retry for %s: %s", netloc, exc)
        attempt.skipped_reason = "wall_budget"
    except ImpersonateTransportError as exc:
        logger.info(
            "resolution_source: the impersonated retry of %s failed in transport (failure_class=%s exc=%s)",
            netloc,
            exc.failure_class,
            exc.exc,
        )
    except ImpersonateDeclined as exc:
        logger.warning(
            "resolution_source: the impersonated retry of %s produced nothing (%s: %s)",
            netloc,
            type(exc).__name__,
            exc,
        )
    return None


def _impersonate_dial_budget_s(budget_s: float, pol: LadderPolicy) -> float:
    """The wall the whole retry gets: this rung's remaining budget, capped where the caller caps it.

    A caller whose own wall is far longer than one page's worth (the loop's 90 s ``fetch``) bounds
    the retry at one plain hop's timeout, so a slow redirect chain cannot spend a whole tool budget
    inside one transport. The fetcher caps nothing here, because its wall IS one question's worth.
    """
    if pol.impersonate_dial_wall_s is None:
        return budget_s
    return min(budget_s, pol.impersonate_dial_wall_s)


async def _impersonate_rung(
    url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult | None:
    """Re-dial a page that answered our aiohttp client 403, presenting a real browser's fingerprint.

    Measured 2026-09-04 from a GitHub Actions runner (``scripts/probes/fetch_diagnostic.py``): four
    Akamai-fronted federal hosts (bls.gov twice, one of them a PDF, cdc.gov and fsis.usda.gov)
    answered the bot's own client 403 and the same GET through ``curl_cffi`` with Chrome
    impersonation 200, so that refusal was a TLS and HTTP/2 fingerprint verdict and is recoverable
    client-side. The four hosts that also refused the impersonated GET (Cloudflare, CloudFront and
    DataDome fronts) are the egress-IP population and stay the Wayback and paid rungs' business.
    Free: no key, no model call, no spend.

    The transport is :mod:`metaculus_bot.research.impersonated_fetch`, which carries the SSRF
    invariants itself because libcurl never touches aiohttp's connect-time ``FilteringResolver``:
    it pre-resolves the host through the repo's own vetting predicate, pins the connection to the
    vetted address, re-guards and re-pins every redirect hop under the shared ``MAX_REDIRECTS``
    cap, and caps the body as the direct fetch does: the page cap for every body, and one re-dial
    under the document cap when a declared PDF aborts on the first, so a cited PDF between the two
    caps (one of the four measured recoverable URLs is a PDF) is read here as the direct path
    would have read it (:func:`_impersonate_or_record_the_skip` names the two constants).

    Ordered gates, cheapest first, after the trigger (:func:`_impersonate_rung_applies`). The kill
    switch (``impersonated_fetch.impersonation_enabled``, the transport's own reading of
    ``RESOLUTION_SOURCE_IMPERSONATE_ENABLED``, ON by default in code unlike the paid rung's
    default-off) records ``impersonate_disabled``. The URL dialed is ``direct.url``, the hop that
    ANSWERED 403 rather than the cited URL, because :func:`_resolution_status_outcome` sets ``url``
    to the answering hop and that is the URL the host actually refused; when the two differ the
    landing is re-vetted through :func:`_landing_refused`, and a refusal is a decline with no
    attempt, the same helper :func:`_rendered_rung` runs. The rung's ATTEMPT stays keyed on the
    cited ``url``, which is what the escalation line names. The per-run memo
    (``impersonated_fetch.impersonation_refused``, keyed by the HOST that answered a block plus the
    exact URL dialed to reach it) records ``impersonate_host_refused``: a host that answered the
    impersonated client with a block status is not going to answer the next cited URL on it
    differently in the same run, and a chain that ended in one is not walked twice, while a host
    that merely redirected into the block keeps its other URLs. The memo is process-global and shared with
    gap-fill v2, whose ``fetch`` and ``read_document`` ladders write it too, so the earlier
    refusal it records may have been a v2 fetch of a URL no question ever cited. Then the wall
    budget, through :meth:`LadderContext.claim_rung_budget` with the meta-refresh hop's floor: one
    GET against a host that just answered us, no launch and no gate contended process-wide.

    Deliberately NO fast-path skip (:func:`_skip_for_fast_path`). That token separates "the
    question's close left no room for a browser" from "this rung ran out of the provider's own
    clock" and its docstring reserves it for the two EXPENSIVE rungs; this rung costs exactly what
    the meta-refresh hop costs, and the cheap rungs run on the fast path unchanged. Do not add the
    gate for symmetry.

    Outcomes. A decline from the transport (:class:`ImpersonateDeclined`) leaves the attempt FIRED
    and :func:`_run_rung` closes it on the direct status, which is the ``route=impersonate
    status=blocked`` record the archive wants: we tried the fingerprint and this is still the
    answer. The one exception is :class:`ImpersonateUnpinnable`, a hop whose host would not pin
    to a vetted public address, which is stamped on the attempt already started as its own skip
    rather than appended as a second one (the pattern :func:`_rendered_rung` uses for
    ``render_non_200``). On the first hop that is near-impossible in practice, since the direct
    fetch resolved this host through the filtering resolver moments earlier; on a later redirect
    hop it is a target the direct fetch never resolved, so the skip means the pin failed on some
    hop rather than that nothing was dialed. A non-200 answer stamps the rung's own outcome
    (``blocked`` for a still-403, so the escalation line reads ``rung=impersonate
    outcome=blocked``; ``not_found`` for a 404 or 410; ``error`` for anything else), memoizes the
    host when the status is block-shaped (``impersonated_fetch.IMPERSONATE_BLOCK_STATUSES``: the
    three ``blocked`` rows of the status table plus 401 and 503, which stamp ``error`` here and
    still switch the host off), and returns None. A 200 goes through
    :func:`_impersonated_body_outcome`, the same classification a direct 200 gets. A success or
    throttle is returned. An unreadable result is returned only when the caller's verdict sets
    ``escalate_rendered``; this restores gap-fill's browser follow-up while resolution-source
    retains its original direct refusal for the off-site rungs.
    """
    if not _impersonate_rung_applies(direct):
        return None
    if not impersonated_fetch.impersonation_enabled():
        ctx.skip_rung("impersonate", direct.status, url, "impersonate_disabled")
        return None
    retry_url = direct.url
    if await guard._landing_refused(retry_url, url, action="re-dialing"):
        return None
    if impersonated_fetch.impersonation_refused(retry_url):
        ctx.skip_rung("impersonate", direct.status, url, "impersonate_host_refused")
        return None
    budget_s = ctx.claim_rung_budget("impersonate", direct.status, url, RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S)
    if budget_s is None:
        return None
    budget_s = _impersonate_dial_budget_s(budget_s, ctx.policy)
    attempt = ctx.start_rung("impersonate", direct.status, url)
    response = await _impersonate_or_record_the_skip(retry_url, budget_s, host_sems, attempt)
    if response is None:
        return None
    netloc = urlparse(retry_url).netloc
    if response.status != 200:
        outcome = _NON_OK_FETCH_STATUS.get(response.status, "error")
        # The memo write is the transport's rule (`IMPERSONATE_BLOCK_STATUSES`: a 404 says the path
        # is gone, which says nothing about the host's view of our fingerprint), keyed on the host
        # that ANSWERED plus the exact URL dialed: the impersonated client follows redirects itself,
        # so the block can come from a later hop's netloc, and that is the host that refused us,
        # while the dialed host merely redirected and keeps its other URLs.
        impersonated_fetch.note_refusal_if_block_shaped(
            dialed_url=retry_url, answered_url=response.url, status=response.status
        )
        attempt.outcome = outcome
        # The Server header names which CDN refused the impersonated GET (a host that refuses both
        # clients is otherwise indistinguishable from one whose fingerprint scoring changed), and
        # the elapsed time separates an edge's instant refusal from a challenge that ran the clock.
        logger.info(
            "resolution_source: the impersonated retry of %s was answered %d by %s (%s; server=%s elapsed=%.1fs); "
            "the direct result stands",
            netloc,
            response.status,
            urlparse(response.url).netloc,
            outcome,
            server_header_token(response.server) or "none",
            response.elapsed_s,
        )
        return None
    result = await _impersonated_body_outcome(response, ctx)
    if result.chrome_metric_withheld:
        # The metric withheld the impersonated body's extraction. `chrome_metric_withholds` counts a
        # withhold anywhere on the URL's ladder, and a 403 direct fetch had no body for the metric
        # to withhold, so the fact is stamped on the direct result: `_fetch_one` carries it from
        # there onto whatever this ladder leaves standing, the direct result when nothing rescues
        # the page or a later rung's result when one does. Idempotent on a rescue, whose own
        # result already carries the flag.
        direct.chrome_metric_withheld = True
    # The rung's own verdict, stamped before deciding: a body that classified as unreadable is a
    # fact about the page the escalation line has to keep even though the direct status stands.
    attempt.outcome = result.status
    if result.status in ("success", "throttled") or result.escalate_rendered:
        return result
    logger.info(
        "resolution_source: the impersonated retry of %s got a 200 that classified as %s; the direct result stands",
        netloc,
        result.status,
    )
    return None


def _rendered_rung_applies(direct: FetchResult) -> bool:
    """Whether a browser could plausibly turn ``direct`` into readable content.

    Three triggers, all pages that answered 200 with nothing this caller can use: ``js_wall``
    (the population the rung was measured on), the ``thin_page`` shape of
    ``no_resolving_content``, and a result the caller's own verdict marked
    ``escalate_rendered`` — a success too short to be the page, which only the gap-fill preset
    produces. Why ``embed_shell`` and ``blocked`` are NOT triggers: ``docs/architecture.md``,
    "Why the rungs sit in this order".
    """
    if direct.escalate_rendered or direct.status == "js_wall":
        return True
    return direct.status == "no_resolving_content" and direct.status_reason == "thin_page"


async def _render_or_record_the_skip(
    url: str,
    budget_s: float,
    host_sems: dict[str, asyncio.Semaphore],
    attempt: RungAttempt,
    *,
    memo_scope: MemoScope,
) -> RenderedPage | None:
    """Run the transport under the rung's wall bound; on anything but a page, record why.

    Every way the transport can stop short of a rendered page lands on ``attempt.skipped_reason``
    here, so the rung itself reads as one statement per outcome. The mapping, and why each token
    is what it is, is the :func:`_rendered_rung` docstring's business; this function only
    applies it.
    """
    goto_timeout_ms = int(min(RENDER_TIMEOUT_MS, budget_s * 1000) - RENDER_SETTLE_MS)
    try:
        page = await asyncio.wait_for(
            render_page(
                url,
                memo_scope=memo_scope,
                host_gate=guard._sem_for_host(host_sems, url),
                goto_timeout_ms=goto_timeout_ms,
                # The transport's exit (shared teardown bound, launch, driver stop) runs AFTER
                # this deadline and has to land inside the wait_for below, so the deadline is
                # the budget less that reserve. Strictly safer: it can only shorten the goto or
                # decline earlier at the transport's own navigation floor.
                deadline_monotonic_s=time.monotonic() + budget_s - RENDER_EXIT_RESERVE_MS / 1000,
                # Recording the page's own XHR costs one buffered body per response inside the
                # render task, which is why the transport keeps it off by default — here it is
                # exactly the rung's fallback, so it is worth the bytes.
                harvest_json=True,
            ),
            timeout=budget_s,
        )
    except RenderBudgetExpired:
        # The budget ran out in the queue behind the two gates: nothing rendered, so it is the same
        # skip the pre-gate check records, not a cut-off render and not a missing browser.
        attempt.skipped_reason = "wall_budget"
        return None
    except RenderTimeout as exc:
        # The transport's own DOM-read bound: a browser ran and the page kept navigating (or the
        # transport re-raised it for a URL it already cut off this run). A fact about the page.
        logger.warning(
            "resolution_source: the rendered rung for %s was cut off by the transport (%.1fs budget): %s; "
            "leaving the direct result",
            urlparse(url).netloc,
            budget_s,
            exc,
        )
        attempt.skipped_reason = "render_timeout"
        return None
    except TimeoutError:
        # The rung's own wait_for above: the render was still queued behind the two gates when
        # the budget ran out, or the transport overran its exit reserve. Neither says anything
        # about the page, so it is the wall binding, the same skip the pre-gate check records.
        # Ordered after the two transport exceptions, which both subclass this.
        logger.warning(
            "resolution_source: the rendered rung for %s outlived its %.1fs wall budget before the transport "
            "answered; leaving the direct result",
            urlparse(url).netloc,
            budget_s,
        )
        attempt.skipped_reason = "wall_budget"
        return None
    except RenderDomOverCeiling:
        # Chromium rendered the page and the DOM is over `RENDERED_DOM_MAX_CHARS`: a fact about
        # the page, kept out of `renderer_unavailable` so the install-failed signal stays clean.
        # The transport already logged the size.
        attempt.skipped_reason = "render_dom_too_large"
        return None
    except RenderOffHost:
        # Chromium's main frame landed on a host other than the pinned one, a server-side redirect
        # the route guard never sees, so the transport refused the DOM unread on its pre-read check,
        # or discarded it unpublished when the navigation committed during the read itself. Also a
        # fact about the page, and nothing from that render is published either way. The transport
        # already logged both hosts.
        attempt.skipped_reason = "render_off_host"
        return None
    if page is None:
        # The transport declines with ONE signal for several causes — Playwright missing or
        # broken, a host that will not pin to a public IP, or a browser error — and its own
        # WARN/DEBUG lines say which. Recorded as a SKIP rather than a fired rung because nothing
        # was rendered: it then claims no `route=` and emits no escalation line, while keeping
        # the measured wall_s that says what the declined launch cost.
        attempt.skipped_reason = "renderer_unavailable"
    return page


async def _rendered_rung(
    url: str, direct: FetchResult, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult | None:
    """Render an unreadable page in headless Chromium and re-classify it, or None.

    Runs from the escalation ladder, outside the aiohttp response context, so no response is
    held open across a 12-35 s render. It does NOT run outside the per-host gate: the transport
    re-acquires the same loop-wide ``Semaphore(1)`` for the URL's host and holds it across the
    launch-cap queue, the launch, the navigation, the settle and the teardown, because Chromium
    dials that host itself. Both acquires are unbounded by design (FUTURE.md item 5), which is
    why the transport recomputes the navigation budget only once both are held.

    Self-bounding on the shared pattern: skipped below ``RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S``
    of remaining wall, and the navigation gets the remaining budget less the settle, capped at
    the transport's own 35 s — as a CEILING. The transport tightens it after the gates to what is
    actually left of the DEADLINE handed to it less the settle and the DOM read, or declines
    under its own floor before a browser is launched, so a goto that runs its budget out can
    still be settled and read. That deadline is the remaining budget LESS the transport's exit
    reserve (``RENDER_EXIT_RESERVE_MS``: the shared teardown bound plus a second for the launch
    and the driver stop), because the transport spends those after its DOM is in hand and this
    rung's own bound has to fit them too. Degrading to the direct result costs one page;
    overrunning the provider's outer ``wait_for`` costs every page the question already fetched.

    The whole transport call — queue, launch, navigation, DOM read and teardown — is ALSO held
    to the remaining budget with ``asyncio.wait_for``. That bounds when this rung stops WAITING,
    not when the transport stops RUNNING: ``wait_for`` cancels the render and then awaits its
    unwinding teardown, so the reserve above is what keeps that teardown inside the wall, and a
    render that runs every bound out hands its DOM back before the cut instead of being
    cancelled in its own exit. The receipt is ogimet.com (2026-09-03): the goto timed out at
    33 s as designed and ``page.content()`` then blocked for 40 s more, for a 76 s render against
    a 45 s wall. The two bounds are recorded apart, by the exception class the transport raises.
    Its own DOM-read bound raises :class:`RenderTimeout`: a browser ran and the page kept
    navigating, which is a fact about the page and is its own skip, ``render_timeout``, rather
    than ``renderer_unavailable`` — a cut-off render says nothing about whether Chromium works,
    and must not latch that warning. This rung's outer bound raises a bare ``TimeoutError``: the
    render was still queued behind the two gates, or the transport overran its exit reserve,
    neither of which is about the page, so it is recorded as ``wall_budget`` like the pre-gate
    floor check and the post-gate :class:`RenderBudgetExpired`. The direct result is what stands
    either way. The transport memoises a timed-out URL itself, at the raise site and only when a
    browser actually ran; with the reserve in place its DOM-read bound lands before this rung's
    outer cut even in the salvage shape (goto ran its budget out), so the memo is written there
    too. A memoised URL re-raises on the next question, so it is recorded the same way again.

    The rendered DOM re-enters :func:`_classify_html_body`, so a rescued page gets the same
    chart read, ARIA rewrite, floors and disclosure leads as a directly-fetched one — unless the
    browser was answered with something other than a 200. The direct GET got a 200 for this URL
    (that is the trigger), so a non-200 main-frame status is the edge telling the browser apart,
    and its markup (a 403 or 429 interstitial routinely clears the chrome floor) is not the page:
    the rung leaves the direct result standing and does not memoise, because a 429 is retryable;
    that is its own skip, ``render_non_200``. A DOM over ``RENDERED_DOM_MAX_CHARS`` is likewise a
    fact about the page and its own skip, ``render_dom_too_large`` (the transport raises
    :class:`RenderDomOverCeiling` for it), so neither inflates ``renderer_unavailable``. A main
    frame that landed on a host other than the one the browser's DNS pin covers is refused by the
    transport unread on its pre-read check, or discarded unpublished when the navigation commits
    during the read itself (:class:`RenderOffHost`), and is its own skip too, ``render_off_host``:
    a server-side redirect the route guard never sees, so a fact about the page rather than the
    install, and nothing from that render is published either way.
    When the DOM STILL carries nothing, the JSON the page fetched for itself is the last free
    route (:func:`_derived_api_from_harvest`) — a JavaScript dashboard's numbers arrive over XHR
    and are in its HTML at no wait condition. Only once that fails too is the URL memoized
    (:func:`note_rendered_no_text`), so a second URL on the same page in this run does not spend
    another launch to learn the same thing; that memo hit is its own skip, ``rendered_no_text``,
    so it never inflates the count the operator reads as the Chromium install having failed.

    The browser is handed ``direct.url``, the URL the direct fetch LANDED on once its redirect
    hops were followed and re-guarded, rather than the cited ``url``: the pin then covers the
    host that actually serves the content, which is also the host the landing check holds the
    browser to, and a page whose canonical form is one ordinary hop away (``example.com`` to
    ``www.example.com``) is not refused for taking it. When the two differ, the landing is
    re-vetted through :func:`_landing_refused`, the one home of that re-vet (shared with the
    impersonated retry, which dials ``direct.url`` for the same reason), because this is where
    the URL the browser dials is decided; the refusal is a decline rather than a terminal
    result, so this site returns None with no attempt. The render memos are keyed on
    the URL rendered; the classifier's base, and so the ``FetchResult.url`` a rescue carries onto
    the ``RESOLUTION_SOURCE_FETCH`` line and into the published ``### <url>`` heading, is the URL
    the browser's main frame LANDED on (``RenderedPage.document_url``: the direct fetch's final
    hop, or a same-host hop past it); the rung's attempt stays keyed on the cited ``url``, which
    is what the escalation line names, so a per-URL join between the two lines keys on the
    escalation line; and the harvested feed is remembered for the cited URL's host, which is the
    host the next cited URL asks :func:`_derived_api_rung` about.
    """
    if not _rendered_rung_applies(direct):
        return None
    render_url = direct.url
    if await guard._landing_refused(render_url, url, action="rendering"):
        return None
    memo_scope = ctx.policy.render_memo_scope
    if rendered_to_nothing(render_url, memo_scope=memo_scope):
        ctx.skip_rung("rendered", direct.status, url, "rendered_no_text")
        return None
    budget_s = ctx.claim_rung_budget("rendered", direct.status, url, RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S)
    if budget_s is None:
        return None
    attempt = ctx.start_rung("rendered", direct.status, url)
    page = await _render_or_record_the_skip(render_url, budget_s, host_sems, attempt, memo_scope=memo_scope)
    if page is None:
        return None
    if page.http_status is not None and page.http_status != 200:
        logger.warning(
            "resolution_source: the browser was answered %d for %s where the direct GET got 200; "
            "not reading that page as content",
            page.http_status,
            urlparse(render_url).netloc,
        )
        # Its own skip, not a fired rung: nothing about the page was read, so the attempt claims
        # no route and emits no escalation line, and the count keeps "Chromium refused where our
        # GET was not" measurable. No memo, because a 429 has to stay re-requestable.
        attempt.skipped_reason = "render_non_200"
        return None
    classified = await classify._classify_html_body(
        page.html.encode("utf-8", errors="replace"),
        # The document the DOM came from: `final_url` when the navigation committed (same host as
        # `render_url` by construction, the path may differ after a same-host client-side redirect
        # or meta refresh), so relative links and the published section URL name the real
        # document, as the direct path's last hop and the `meta_refresh` route already do.
        page.document_url,
        page.content_type or "text/html",
        # The direct fetch's status, not the browser's: this page answered 200 and carried no
        # text, which is the fact the record should keep. Chromium reports no status at all
        # when a goto timed out and the DOM was salvaged, and a non-200 never reaches here.
        http_status=direct.http_status if direct.http_status is not None else 200,
        # What the browser left of the wall: the render spent the rest, and the extractor's
        # optional second pass declines under its floor rather than overrunning the provider.
        query=ctx.query,
        remaining_wall_s=ctx.rung_budget_s(),
        pol=ctx.policy,
    )
    classify.capture_html_read(ctx, classified)
    if classified.result.chrome_metric_withheld:
        # The metric withheld the rendered DOM's extraction. `chrome_metric_withholds` counts a
        # withhold anywhere on the URL's ladder, and a js_wall direct fetch had nothing for the
        # metric to withhold, so the fact is stamped on the direct result: `_fetch_one` carries
        # it from there onto whatever this ladder leaves standing, the direct result when nothing
        # rescues the page or the harvested feed when it does.
        direct.chrome_metric_withheld = True
    # The render's own verdict, stamped before the harvest gets its turn: when the harvested
    # feed rescues the page, the ladder's result is `success` and the closer would otherwise
    # credit the render with a rescue the DOM never delivered.
    attempt.outcome = classified.result.status
    if classified.result.status in ("success", "throttled"):
        return classified.result
    derived = _derived_api_from_harvest(url, direct, page, ctx)
    if derived is not None:
        return derived
    note_rendered_no_text(render_url, memo_scope=memo_scope)
    return None


def _derived_api_from_harvest(
    url: str, direct: FetchResult, page: RenderedPage, ctx: context.LadderContext
) -> FetchResult | None:
    """Serve the JSON the rendered page fetched for itself, when the DOM carried nothing.

    Its own rung attempt rather than part of the render's, because ``route`` is the LAST rung
    that fired and ``derived_api`` is what actually produced the text — the render only found
    the endpoint. The endpoint is also remembered for the host, so a later cited URL on it can
    GET the feed without a second launch (:func:`_derived_api_rung`).

    Declines silently when nothing was harvested or the biggest body carries no usable content
    (:func:`vacuous_body_status`): a body we could not decode must never become the page's
    content on a section captioned primary grading evidence.
    """
    harvested = derived_api.largest_json(page.json_responses)
    if harvested is None:
        return None
    raw, undecodable_ratio = decode_text_body(harvested.body, "application/json")
    if vacuous_body_status(raw, undecodable_ratio, require_csv_rows=False) is not None:
        return None
    derived_api.remember_endpoint(url, harvested.url)
    endpoint = derived_api.DerivedEndpoint(endpoint_url=harvested.url, discovered_on=url)
    ctx.start_rung("derived_api", direct.status, url)
    return _derived_api_result(url, endpoint, raw, http_status=direct.http_status, ctx=ctx)


def _derived_api_result(
    url: str, endpoint: derived_api.DerivedEndpoint, raw: str, *, http_status: int | None, ctx: context.LadderContext
) -> FetchResult:
    """One derived-feed result: the provenance lead, then the budgeted JSON.

    The lead LEADS and its cost comes out of the per-URL cap
    (:func:`resolution_presentation._lead_then_capped_body`),
    because a feed served with its provenance line trimmed off is a JSON blob nobody can check.
    """
    artifact = run_cache.TextRead(
        url=url,
        text=raw,
        http_status=http_status,
        content_type="application/json",
        lead=derived_api.derived_api_lead(endpoint, url),
    )
    result = artifact.present(ctx.policy, query=ctx.query, route="direct", now=ctx.now)
    if result is None:
        raise RuntimeError("fresh derived-API artifact was rejected by the policy that classified it")
    if result.status == "success":
        ctx.capture_read(result, artifact)
    return result


async def _derived_api_rung(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult | None:
    """GET a JSON feed an earlier render on this host already found, before launching a browser.

    This is the whole point of remembering the endpoint: a host with several cited URLs in one
    run pays for one Chromium launch, not one per URL. It runs BEFORE the rendered rung for the
    same reason every ladder here is ordered cheapest-first — one GET against a known endpoint
    is a rounding error next to a browser launch. Within a question that holds even when the
    URLs are fetched concurrently, because the dispatcher runs this rung and the browser rung
    under one per-host gate (:meth:`QuestionRungBudget.browser_escalation_gate`), so a same-host
    sibling asks for the endpoint only after the first render has had its chance to record it.

    The GET goes through :func:`_fetch_direct`, so it inherits the SSRF preflight, the
    connect-time filtering resolver, the redirect re-guard, the per-host gate and the
    budget-clamped hop timeout unchanged. A feed that fails hands the URL on to the browser.
    """
    if not _rendered_rung_applies(direct):
        return None
    endpoint = derived_api.endpoint_for(url)
    if endpoint is None:
        return None
    if ctx.claim_rung_budget("derived_api", direct.status, url, RESOLUTION_SOURCE_DERIVED_API_MIN_BUDGET_S) is None:
        return None
    ctx.start_rung("derived_api", direct.status, url)
    logger.info(
        f"resolution_source derived_api: {urlparse(url).netloc} -> {endpoint.endpoint_url} "
        f"(found on {endpoint.discovered_on}, direct read was {direct.status})"
    )
    feed = await direct_fetch._fetch_direct(session, endpoint.endpoint_url, host_sems, context._aux_ctx(ctx))
    if feed.status == "throttled":
        return replace(feed, url=url)
    if feed.status != "success":
        return None
    if not is_json_content_type(feed.content_type or ""):
        # The same gate the harvest half applies at discovery, because a remembered endpoint is
        # not a promise about what it answers NEXT time: one came back 200 with an HTML "session
        # expired" portal page, which the lead below would have introduced as the JSON feed the
        # page loads its figures from. Declining hands the URL to the browser, whose own harvest
        # is gated the same way.
        logger.info(
            "resolution_source derived_api: %s answered %r rather than JSON — not served as the feed",
            endpoint.endpoint_url,
            feed.content_type,
        )
        return None
    return _derived_api_result(url, endpoint, feed.text, http_status=feed.http_status, ctx=ctx)


# A page the archive can plausibly substitute for: the host refused us, never answered, or says the
# URL is gone. Why `js_wall`, `no_resolving_content` and `ssrf_blocked` are excluded:
# docs/architecture.md "Why the rungs sit in this order".
_WAYBACK_TRIGGER_STATUSES: frozenset[FetchStatus] = frozenset({"blocked", "error", "not_found"})


# A body an archived copy is no more readable than the live one, so the archive adds nothing: a
# declared image, which only a model read can turn into text at all.
_WAYBACK_EXCLUDED_REASONS: frozenset[FetchStatusReason] = frozenset({"image_needs_reader"})


def _wayback_rung_applies(direct: FetchResult, pol: LadderPolicy) -> bool:
    """Whether the archive is a plausible substitute for ``direct``, for THIS caller.

    The shared trigger set plus whatever the caller adds to it (the loop also substitutes for a
    body it could not read at all), minus the reasons an archived copy cannot help with.
    ``wayback_needs_host_refusal`` then drops a ``blocked`` that carries no host status, which is a
    refusal WE made: handing that to a third-party fetcher is the bypass ``ssrf_blocked``'s
    exclusion prevents, one reason over.
    """
    if direct.status not in (_WAYBACK_TRIGGER_STATUSES | pol.wayback_extra_trigger_statuses):
        return False
    if direct.status_reason in _WAYBACK_EXCLUDED_REASONS:
        return False
    return not (pol.wayback_needs_host_refusal and direct.status == "blocked" and direct.http_status is None)


async def _wayback_snapshot_result(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult | None:
    """Fetch the archive's freshest capture of ``url`` and serve it, or withhold it.

    The fetch goes through :func:`_fetch_direct`, so the snapshot is classified by exactly the
    path a live page is — including the chart read and the chrome floor — and inherits the SSRF
    preflight, the per-hop re-guard and the budget-clamped hop timeout. What comes back extra is
    the FINAL URL, which is where the archive puts the 14-digit capture timestamp.

    Three outcomes, in this order, and the order is the design.

    The inner URL is UNWRAPPED (repeatedly, since a capture OF a capture presents
    ``web.archive.org`` as its own inner host) and re-checked first through
    :func:`_hop_refusal`, the one home of the two checks every derived URL owes, because a
    hostname check on ``web.archive.org/web/…/metaculus.com/…`` sails past every self-reference
    filter in the pipeline: an archived Metaculus page in front of a forecaster is the question
    quoting itself. Then a snapshot the archive could not serve at all (no capture, or a capture that
    404s) DECLINES: there is no archived copy, which is a different fact from a stale one, and
    the direct route's own status says more about the source than a fact about the archive would.
    Only a capture we actually READ and cannot date, or can date and it is too old, is withheld
    as ``stale_data`` — because the disclosure that makes a snapshot admissible is its age, and a
    copy with no usable date cannot carry it. The direct status is not lost by that swap either:
    the ``RESOLUTION_SOURCE_ESCALATION`` line for this rung carries ``from_status``, and the
    withhold keeps the direct fetch's HTTP status and failure diagnostics, so the
    ``RESOLUTION_SOURCE_FETCH`` line it replaces the direct result on still says which host
    refused us and from which CDN.
    """
    snapshot_ctx = context._aux_ctx(ctx)
    snapshot = await direct_fetch._fetch_direct(
        session, wayback_snapshot_url(url, now=ctx.now), host_sems, snapshot_ctx
    )
    parsed = parse_snapshot_url(snapshot.url)
    captured_of = None if parsed is None else innermost_url(parsed.inner_url)
    if captured_of is not None and await guard._hop_refusal(captured_of) is not None:
        logger.warning(
            "resolution_source wayback refused: snapshot of %s wraps a URL we do not fetch (%s)",
            urlparse(url).netloc,
            urlparse(captured_of).netloc,
        )
        return None
    if snapshot.status != "success":
        # Two different facts, and the archive's own redirect is what tells them apart: a
        # request it never redirected onto a dated capture URL means it holds no capture, while
        # a capture URL we did land on and could not use means it holds one we cannot read.
        # Both used to log "no archived copy served", so apnews.com — a capture served in full
        # whose extraction was 355 chars of AP boilerplate — read as an empty archive.
        logger.info(
            "resolution_source wayback: %s for %s (%s)",
            "no archived copy served" if parsed is None else "an archived capture was served but is unusable",
            urlparse(url).netloc,
            snapshot.status,
        )
        return None
    age_days = None if parsed is None else snapshot_age_days(parsed, ctx.now)
    max_age_days = ctx.policy.wayback_max_age_days
    if max_age_days is None:
        # No bound: the capture date is surfaced and the caller's own reader weighs it, so an
        # undatable capture is a decline rather than a withhold (there is nothing to disclose).
        if parsed is None or age_days is None:
            return None
    elif parsed is None or age_days is None or age_days > max_age_days:
        logger.warning(
            "resolution_source wayback: capture for %s is not usable (final=%s, age=%s) — withheld as stale",
            urlparse(url).netloc,
            snapshot.url,
            "undatable" if age_days is None else f"{age_days:.1f}d",
        )
        return FetchResult(
            url=url,
            status="stale_data",
            text="",
            # The cited HOST's status and diagnostics, not the archive's: this verdict replaces
            # the direct result on the FETCH line, where `http=200` was the archive answering
            # and the missing `failure_class` / `server` undercounted the blocked population
            # the ladder exists for. Only the success below reports the snapshot's own status,
            # because those bytes are the archive's.
            http_status=direct.http_status,
            content_type=direct.content_type,
            failure_class=direct.failure_class,
            exc=direct.exc,
            server=direct.server,
            # A verdict names its own rung. The dispatcher otherwise stamps the LAST rung that
            # fired, and the paid rung fires after this one: a stale capture the reader then
            # failed to improve on came back `route=url_context status=stale_data`, a status that
            # rung cannot produce, on the field that partitions the archive by route.
            route="wayback",
        )
    # The lead LEADS and its cost comes out of the per-URL cap
    # (:func:`resolution_presentation._lead_then_capped_body`):
    # an archived page whose age line has been trimmed off is being passed off as the live one.
    assert parsed is not None
    assert age_days is not None
    lead = wayback_lead(parsed, age_days, direct.status)
    result = FetchResult(
        url=url,
        status="success",
        text=resolution_presentation._lead_then_capped_body(lead, snapshot.text, url, cap=ctx.policy.per_url_max_chars),
        http_status=snapshot.http_status,
        content_type=snapshot.content_type,
        datawrapper_charts=snapshot.datawrapper_charts,
        unreadable_embeds=snapshot.unreadable_embeds,
        status_reason=snapshot.status_reason,
        chrome_metric_withheld=snapshot.chrome_metric_withheld,
        precision_rescued=snapshot.precision_rescued,
        links=snapshot.links,
        passages_returned=snapshot.passages_returned,
        passages_grounded=snapshot.passages_grounded,
        fallback_used=snapshot.fallback_used,
    )
    _capture_wayback_read(
        ctx,
        result=result,
        snapshot_ctx=snapshot_ctx,
        snapshot=snapshot,
        direct=direct,
        parsed=parsed,
    )
    return result


def _capture_wayback_read(
    ctx: context.LadderContext,
    *,
    result: FetchResult,
    snapshot_ctx: context.LadderContext,
    snapshot: FetchResult,
    direct: FetchResult,
    parsed: WaybackSnapshot,
) -> None:
    snapshot_artifact = snapshot_ctx.artifact_for(snapshot)
    if snapshot_artifact is None:
        return
    ctx.capture_read(
        result,
        run_cache.WaybackRead(
            url=result.url,
            snapshot=parsed,
            artifact=snapshot_artifact,
            live_status=direct.status,
            live_http_status=direct.http_status,
            live_content_type=direct.content_type,
            live_failure_class=direct.failure_class,
            live_exc=direct.exc,
            live_server=direct.server,
        ),
    )


async def _wayback_rung(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult | None:
    """Try the Wayback Machine for a page our own address could not reach.

    Bounded three ways, because this rung's cost is concentrated rather than spread: below
    ``RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S`` of remaining wall it is skipped, at most
    ``RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS`` snapshots are fetched per question, and every
    snapshot contends on the one ``web.archive.org`` host gate — which is the documented trade
    for the politeness that gate exists to provide.
    """
    if not _wayback_rung_applies(direct, ctx.policy):
        return None
    if ctx.claim_rung_budget("wayback", direct.status, url, RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S) is None:
        return None
    if not ctx.shared.take_wayback_attempt():
        logger.warning(
            "resolution_source: skipping the wayback rung for %s — this question's %d snapshot attempt(s) are spent",
            urlparse(url).netloc,
            RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS,
        )
        ctx.skip_rung("wayback", direct.status, url, "wayback_cap")
        return None
    ctx.start_rung("wayback", direct.status, url)
    return await _wayback_snapshot_result(session, url, direct, host_sems=host_sems, ctx=ctx)


# What the paid reader is allowed to be asked about. Tested against the DIRECT outcome (an
# archive withhold on the way down does not change it — see `_escalate_unresolved`): everything
# the free ladder left unresolved EXCEPT the outcomes where a model-mediated read cannot help or
# must not be tried. A 404/410 has no page to read, an empty or undecodable body and an unreadable
# document are bytes we DID get (only `no_text_layer` could ever be rescued, and that is v2's
# `read_document` job on a URL the driver chose), and `ssrf_blocked` is a URL WE refused — handing
# that to a third-party fetcher is exactly the bypass the guard exists to prevent, which is why it
# is excluded here and not merely unlisted.
# Two outcomes inside the set are excluded by REASON rather than by status — see
# :func:`_url_context_rung_applies` — so this set is the ceiling on the population, not the
# population itself.
_URL_CONTEXT_TRIGGER_STATUSES: frozenset[FetchStatus] = frozenset(
    {"blocked", "js_wall", "error", "no_resolving_content"}
)

# The reasons that take an outcome OUT of the population above. Scoped on the reason rather than by
# dropping the status, because the statuses they ride are otherwise exactly what the rung exists
# for: `embed_shell` and `thin_page` are pages our client genuinely could not read, and a 403
# `blocked` is the rung's whole reason to exist.
#   `no_matching_passage` — a document we read END TO END whose passage selection matched no query
#   term. Its bytes were never the problem (we hold its full text and its outline), so paying
#   Gemini to re-read the same PDF buys nothing.
#   `metaculus_self_ref` — a redirect WE refused because it landed on the question platform's own
#   site. The rung is handed the CITED url, so Gemini would follow the same redirect and read the
#   page we refused: a paid read that by construction returns nothing new, and on Mantic the other
#   bots' forecasts read in as grading evidence. The same bypass `ssrf_blocked` is kept out of the
#   trigger set to prevent, closed here by reason because the self-reference's status is `blocked`
#   by contract.
_URL_CONTEXT_EXCLUDED_REASONS: frozenset[FetchStatusReason] = frozenset({"no_matching_passage", "metaculus_self_ref"})


def _url_context_rung_applies(direct: FetchResult) -> bool:
    """Whether a model-mediated read could plausibly resolve ``direct``.

    The trigger statuses above, minus the outcomes inside them a paid read cannot help with or
    must not be tried on (``_URL_CONTEXT_EXCLUDED_REASONS``, which says why for each).
    """
    if direct.status not in _URL_CONTEXT_TRIGGER_STATUSES:
        return False
    return direct.status_reason not in _URL_CONTEXT_EXCLUDED_REASONS


def _url_context_lead(live_status: FetchStatus) -> str:
    """The MANDATORY disclosure a model-mediated read carries.

    Both clauses are the point. It says WHY this route was taken, so a forecaster knows the host
    refused us rather than that we chose a model over a fetch. And it says the text is not a copy
    of the page — every other section in this snapshot is bytes the host served, and reading a
    paraphrase under the same "primary grading evidence" caption without that line would overstate
    what was retrieved by exactly the amount that matters.
    """
    return (
        f"[Read via Gemini url_context because the live page could not be fetched ({live_status}); "
        f"model-mediated, not a byte-for-byte copy.]"
    )


async def _fetch_robots_txt(
    session: Any, robots_url: str, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> str | None:
    """Read one robots.txt through THIS path's own fetch; None when we could not read it.

    Goes through :func:`_fetch_direct` rather than a second client, so the SSRF preflight, the
    connect-time filtering resolver, the per-hop redirect re-guard, the per-host gate and the
    budget-clamped hop timeout all apply to a request this pre-check makes. That path also
    CLASSIFIES, so a host serving robots.txt as HTML can come back withheld under the chrome
    floor — which reads as "no directives", i.e. proceed and pay, the only direction an
    unreadable robots.txt is allowed to fail in.

    Bounded at ``ROBOTS_FETCH_TIMEOUT_S`` on top of the hop's own clamp, the same bound gap-fill
    v2 gives the identical read: the hop clamp is the remaining WALL (up to 20 s) and the
    per-host gate in front of it is an unbounded acquire, and neither is a sensible price for a
    pre-check whose only job is to avoid one paid call. A timeout reads as unreadable.
    """
    try:
        result = await asyncio.wait_for(
            direct_fetch._fetch_direct(session, robots_url, host_sems, context._aux_ctx(ctx)), ROBOTS_FETCH_TIMEOUT_S
        )
    except TimeoutError:
        logger.info(
            "resolution_source: robots.txt pre-check for %s did not answer in %.1fs", robots_url, ROBOTS_FETCH_TIMEOUT_S
        )
        return None
    return result.text if result.status == "success" else None


async def _url_context_robots_skip(
    session: Any, url: str, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> bool:
    """True when ``url``'s host tells ``Google-Extended`` to stay out of that path.

    Only the PAID rung consults this: the free rungs dial from our own client under our own user
    agent, and this bot's reading of ``Content-Signal: use=reference`` is that reference use is
    permitted. The per-host cache lives in ``robots_policy`` and is shared with gap-fill v2's
    reader, so a host reached by both paths in one run is read once.
    """
    return await google_extended_blocks_url(
        url, fetch_text=lambda robots_url: _fetch_robots_txt(session, robots_url, host_sems, ctx)
    )


async def _url_context_admission(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> tuple[str, float] | None:
    """Every gate the paid read has to clear, in increasing cost order; ``(api_key, budget_s)`` or None.

    The trigger population, the flag (default off in code, on in every bot workflow), the question's
    time-budget fast path, the API key, the wall budget, then the per-host ``Google-Extended``
    robots pre-check — the one gate that costs a request — and the wall budget AGAIN. The robots
    check is worth a request of its own because a host that disallows that token refuses the
    fetch server-side — proven live 2026-09-03, where the same call that retrieved a
    robots-allowed host came back ``URL_RETRIEVAL_STATUS_ERROR`` on
    internationalaisafetyreport.org — so the read would be spend with a known-zero return.

    The budget is checked twice because the pre-check sits between reading it and spending it,
    and it can eat real time: an unbounded per-host gate acquire and then up to
    ``ROBOTS_FETCH_TIMEOUT_S``. The read runs in a thread, which ``wait_for`` cannot cancel, so
    the client-side ceiling is the only thing that returns the worker — and a ceiling sized off
    the figure read BEFORE the pre-check could outlive the provider's wall while the money is
    spent on a result nothing reads. The second check costs nothing (the read has not started),
    and the budget returned here is the one the ceiling and the ``wait_for`` are sized off.
    """
    if not _url_context_rung_applies(direct):
        return None
    if not env_flag_enabled(RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV):
        return None
    if ctx.fast_path:
        # After the flag and before the key: recorded only for a rung that was ARMED, so a
        # flag-off run never reports spend avoided on a rung that could not have fired.
        context._skip_for_fast_path(ctx, "url_context", direct, url)
        return None
    api_key = os.getenv(GOOGLE_API_KEY_ENV)
    if not api_key:
        logger.info(
            "resolution_source: url_context rung is enabled but %s is not set — skipping %s",
            GOOGLE_API_KEY_ENV,
            urlparse(url).netloc,
        )
        ctx.skip_rung("url_context", direct.status, url, "no_api_key")
        return None
    if ctx.claim_rung_budget("url_context", direct.status, url, RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S) is None:
        return None
    if await _url_context_robots_skip(session, url, host_sems, ctx):
        logger.info(f"RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP: url={url} host={urlparse(url).netloc}")
        ctx.skip_rung("url_context", direct.status, url, "robots_disallowed")
        return None
    budget_s = ctx.claim_rung_budget(
        "url_context",
        direct.status,
        url,
        RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S,
        note=" after the robots pre-check",
    )
    if budget_s is None:
        return None
    # Last, and only for a read that cleared every cheaper gate, so a slot is spent on a read
    # that is actually about to fire — not on one robots or the wall already declined. Mirrors
    # the Wayback per-question cap: a question citing several dead sources pays at most
    # RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS times inside one provider wall.
    if not ctx.shared.take_url_context_attempt():
        logger.info(
            "resolution_source: skipping the url_context rung for %s — this question's %d paid read(s) are spent",
            urlparse(url).netloc,
            RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS,
        )
        ctx.skip_rung("url_context", direct.status, url, "url_context_cap")
        return None
    return api_key, budget_s


def _withheld_reply_preview(reply: str) -> str:
    """The head of a paid reply we are DISCARDING, collapsed onto one log line.

    Whitespace-collapsed because a model's answer arrives with newlines and a multi-line log
    record is what makes a run log unreadable, and bounded by
    ``RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS`` because the point is to audit what the read
    said, not to keep it.
    """
    collapsed = " ".join(reply.split())
    if len(collapsed) <= RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS:
        return collapsed
    return f"{collapsed[:RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS]}…"


async def _url_context_rung(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult | None:
    """Ask Gemini to read a page our own client could not, or decline.

    The LAST rung and the only paid one, so every gate is checked before a cent is spent
    (:func:`_url_context_admission`, which also explains why the wall budget is read twice and
    why the figure it hands back is the one that sizes the read).

    Zero successful retrievals DISCARDS the text and reports ``ungrounded``. Gemini answers
    fluently out of parametric memory when every retrieval failed, and this section is captioned
    primary grading evidence, so a fluent unsourced answer here is the Q38195 failure with a
    forecaster-facing blast radius. That is the same floor ``gemini_search`` and v2's
    ``read_document`` apply, for the same reason.

    An answer that opens with ``NOT_ADDRESSED_SENTINEL`` is WITHHELD as ``no_resolving_content``
    / ``not_addressed``. The prompt asks for that opening when the retrieved page does not discuss
    the ask, so it is the designed non-answer, and rendered under the url_context lead it was
    prose standing in for an absent section, the shape :func:`_finish_document` closes for a PDF
    with ``no_matching_passage``. The page was retrieved (so it is not ``ungrounded``) and the
    read was paid for, so the verdict stays on the record as this rung's own rather than declining
    to the direct result.
    """
    admitted = await _url_context_admission(session, url, direct, host_sems=host_sems, ctx=ctx)
    if admitted is None:
        return None
    api_key, budget_s = admitted
    ctx.start_rung("url_context", direct.status, url)
    try:
        text, n_retrievals, statuses = await asyncio.wait_for(
            asyncio.to_thread(
                run_url_context_read,
                url,
                ctx.query,
                api_key=api_key,
                role="resolution_source",
                model=GAP_FILL_V2_READER_MODEL,
                thinking_level=GAP_FILL_V2_READER_THINKING_LEVEL,
                # The client-side ceiling is what returns the worker: wait_for cancels this
                # coroutine and not the thread it is waiting on. Sized off the remaining budget
                # so the read cannot outlive the provider's own wall by more than the margin.
                timeout_ms=int(max(0.0, budget_s - ctx.policy.rung_wall_margin_s) * 1000),
                attempts=RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS,
            ),
            timeout=budget_s,
        )
    except TimeoutError:
        logger.warning("resolution_source url_context read timed out for %s", urlparse(url).netloc)
        return None
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # paid-rung soft-fail boundary: a dead reader leaves the direct result, never takes the provider down
        logger.warning(
            "resolution_source url_context read failed for %s: %s: %s",
            urlparse(url).netloc,
            type(exc).__name__,
            exc,
        )
        return None
    if n_retrievals == 0 or not text.strip():
        # Spelled parallel to the gap-fill v2 reader's AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED (and
        # gemini_search's GEMINI_UNGROUNDED_SUPPRESSED), so the three suppression rates read as
        # one family. `statuses` carries every reported url_retrieval_status; `none` means the
        # SDK attached no url_metadata entry at all. A registered marker spec, named
        # resolution_source_urlcontext_ungrounded_suppressed in scripts/telemetry/markers.py, so
        # the spelling and the always-present `statuses=` field are a data contract.
        logger.warning(
            f"RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url={url} statuses={','.join(statuses) or 'none'}"
        )
        if text.strip():
            # The suppressed answer itself, on its own unregistered line (see the
            # `not_addressed` twin below for why a withheld reply is kept at all). Only when
            # there IS one: the same branch fires on an empty reply, where there is nothing to
            # audit.
            logger.info(f"url_context ungrounded reply for {urlparse(url).netloc}: {_withheld_reply_preview(text)}")
        return FetchResult(
            url=url,
            status="ungrounded",
            text="",
            # The host's status and diagnostics stay on a verdict that served nothing: a
            # model-mediated read has no status of its own, and the FETCH line this result
            # replaces the direct one on is where "which host refused us" is counted.
            http_status=direct.http_status,
            content_type=direct.content_type,
            failure_class=direct.failure_class,
            exc=direct.exc,
            server=direct.server,
        )
    answer = text.strip()
    if answer.startswith(NOT_ADDRESSED_SENTINEL):
        # A registered marker spec like its two URLCONTEXT siblings, named
        # resolution_source_urlcontext_not_addressed in scripts/telemetry/markers.py; `host=`
        # because the rollout question is which hosts Gemini can
        # reach but finds nothing on.
        logger.warning(f"RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED: url={url} host={urlparse(url).netloc}")
        # What the withheld read actually SAID, on a separate unregistered line so the marker's
        # own shape stays a data contract. Without it the verdict is unauditable: "the page does
        # not discuss this" and "the model read the bot-challenge page it was served" reach this
        # branch identically, and the text that tells them apart was being dropped on the floor.
        logger.info(f"url_context not_addressed reply for {urlparse(url).netloc}: {_withheld_reply_preview(answer)}")
        return FetchResult(
            url=url,
            status="no_resolving_content",
            text="",
            http_status=direct.http_status,
            content_type=direct.content_type,
            status_reason="not_addressed",
            failure_class=direct.failure_class,
            exc=direct.exc,
            server=direct.server,
        )
    # The lead LEADS and is budgeted out of the cap
    # (:func:`resolution_presentation._lead_then_capped_body`): a model's
    # answer rendered without the disclosure reads as the page itself.
    lead = _url_context_lead(direct.status)
    return FetchResult(
        url=url,
        status="success",
        text=resolution_presentation._lead_then_capped_body(lead, answer, url, cap=ctx.policy.per_url_max_chars),
        http_status=direct.http_status,
        content_type="text/plain",
    )
