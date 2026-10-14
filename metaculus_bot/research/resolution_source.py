"""Resolution-source provider adapter.

Selects URLs named by a question's resolution criteria, fetches them through the shared ladder
under ``RESOLUTION_SOURCE_POLICY``, and presents the returned evidence to the research pipeline.
This module owns provider-level orchestration: benchmarking and enablement gates, the shared
session and host-politeness map, the per-question rung budget, the optional Datawrapper dataset
hop, telemetry, diagnostics, and raw research records. The factory also applies the provider's
question-level wall and fast-path controls.

The fetch transport, body classification, escalation rungs, and per-rung deadlines live in
``research/fetch_ladder/``. See ``docs/architecture.md`` ("The shared fetch ladder") and
``docs/research.md`` ("Resolution-source fetcher") for those contracts and their boundaries.

``strip_markdown_escapes``, ``looks_like_csv_rows`` and ``format_resolution_sections`` remain
public re-exports for the Tier-1 test and presentation surfaces.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

import aiohttp
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    RESOLUTION_SOURCE_DATAWRAPPER_HOP_WALL_MARGIN_S,
    RESOLUTION_SOURCE_DATAWRAPPER_MIN_HOP_BUDGET_S,
    RESOLUTION_SOURCE_ENABLED_ENV,
    RESOLUTION_SOURCE_MAX_URLS,
    RESOLUTION_SOURCE_WALL_TIMEOUT,
    env_flag_enabled,
)
from metaculus_bot.research import fetch_markers, resolution_datawrapper, resolution_presentation
from metaculus_bot.research.fetch_ladder import classify, context, guard, ladder, policy
from metaculus_bot.research.fetch_ladder.policy import LADDER_CALLER_RESOLUTION_SOURCE
from metaculus_bot.research.http_fetch import DatawrapperChartRef, datawrapper_live_data_url, host_semaphores
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.raw_log import record_raw_research
from metaculus_bot.research.resolution_fetch_result import (
    FetchResult,
    RungSkipReason,
    _fetch_result_sources,
    looks_like_csv_rows,  # noqa: F401  # re-export: the Tier-1 suite imports the row-shape check from this module path
)
from metaculus_bot.research.resolution_presentation import format_resolution_sections  # noqa: F401  # public re-export
from metaculus_bot.research.resolution_url_scan import (
    extract_source_urls,
    is_fred_url,
    is_metaculus_self_ref,
    is_yahoo_ticker_url,
    strip_markdown_escapes,  # noqa: F401  # re-export: the Tier-1 suite imports the markdown unescaper from this module path
)

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Pure helpers — no I/O
# ---------------------------------------------------------------------------


def select_fetchable_urls(criteria: str | None, fine_print: str | None) -> list[str]:
    """Compose the fetchable URL list from a question's resolution criteria + fine print.

    Skips self-refs (the question platform's own site: metaculus.com or
    competitions.mantic.com), FRED, and Yahoo ticker URLs — those either add no new
    info or are covered by another provider. Caps at ``RESOLUTION_SOURCE_MAX_URLS``
    AFTER the skip filter so a run of leading self-refs / FRED / Yahoo URLs (questions
    often list their own page first) doesn't starve the real sources out of the
    fetch budget.
    """
    combined = f"{criteria or ''}\n\n{fine_print or ''}"
    urls = extract_source_urls(combined)
    filtered = [u for u in urls if not (is_metaculus_self_ref(u) or is_fred_url(u) or is_yahoo_ticker_url(u))]
    return filtered[:RESOLUTION_SOURCE_MAX_URLS]


async def _fetch_datawrapper_dataset(
    session: Any,
    chart: DatawrapperChartRef,
    parent_url: str,
    host_sems: dict[str, asyncio.Semaphore],
) -> FetchResult:
    """Tier-2 hop: fetch one Datawrapper chart's LIVE dataset CSV.

    Fetches ONLY the version-free ``static.dwcdn.net/data/<id>.csv`` route —
    never a page-pinned versioned ``dataset.csv``, which serves months-stale
    snapshots as HTTP 200 (see the route mechanism note in ``http_fetch``).

    Freshness guard (serve live or nothing): the dataset's ``Last-Modified``
    must be within ``RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS`` of now.
    Older, missing, or unparseable → ``stale_data`` with no text, so a dead
    chart can never masquerade as the live resolving series. The publish
    timestamp is also rendered into the section so forecasters see the data's
    age even when it passes.

    Content-Type is deliberately NOT gated here: we constructed the URL from a
    shape-validated chart id, the endpoint serves CSV (its versioned sibling
    labels the same bytes ``application/octet-stream``), and the body read is
    size-capped either way. Redirects are unexpected on this CDN and map to
    ``error`` rather than being followed.
    """
    url = datawrapper_live_data_url(chart.chart_id)
    # Uniform SSRF preflight (dwcdn is a public CDN — no exemptions added; the
    # connect-time FilteringResolver stays the real boundary).
    if not await guard.is_public_http_url(url):
        logger.warning(f"resolution_source ssrf_blocked (datawrapper hop): {urlparse(url).netloc}")
        return FetchResult(
            url=url,
            status="ssrf_blocked",
            text="",
            http_status=None,
            content_type=None,
            chart_id=chart.chart_id,
            chart_title=chart.title,
            parent_url=parent_url,
        )

    async with guard._sem_for_host(host_sems, url):
        try:
            async with session.get(url, allow_redirects=False) as resp:
                return await resolution_datawrapper._datawrapper_dataset_outcome(resp, chart, parent_url, url)
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.info(f"resolution_source datawrapper hop {chart.chart_id} error: {type(e).__name__}: {e}")
            return FetchResult(
                url=url,
                status="error",
                text="",
                http_status=None,
                content_type=None,
                chart_id=chart.chart_id,
                chart_title=chart.title,
                parent_url=parent_url,
                failure_class=classify._network_failure_class(e),
                exc=type(e).__name__,
            )


async def _fetch_datawrapper_datasets_in_order(
    session: Any,
    picks: list[tuple[int, DatawrapperChartRef]],
    page_results: list[FetchResult],
    host_sems: dict[str, asyncio.Semaphore],
) -> list[FetchResult]:
    """Fetch selected chart datasets in document order.

    Each chart uses the same CDN host, and its host semaphore already serializes the
    requests. Awaiting each preflight and fetch in turn prevents a slower DNS check
    from letting a later chart overtake the hero chart.
    """
    results: list[FetchResult] = []
    for idx, chart in picks:
        results.append(await _fetch_datawrapper_dataset(session, chart, page_results[idx].url, host_sems))
    return results


async def fetch_resolution_sources(
    urls: list[str], *, query: str = "", fast_path: bool = False, fetch_policy: policy.LadderPolicy | None = None
) -> list[FetchResult]:
    """Fetch each URL under per-netloc Semaphore(1) politeness, then hop to
    the live datasets of any Datawrapper charts the fetched pages embed.

    ``query`` is the question's title plus resolution criteria. It never touches the
    network; its one job is ranking which passages of a cited PDF a forecaster sees.
    Empty is legitimate (a caller with no question text in hand) and simply means a
    document renders its header and outline with no passages. ``fast_path`` is the
    question's time-budget thin-window mode; it rides every URL's :class:`LadderContext`
    and makes the two expensive rungs decline (see there).
    ``fetch_policy`` optionally replaces the production policy for focused diagnostics; the
    known-API rung is added to whichever policy is selected.

    Distinct hosts run concurrently up to the connector limit; same-host
    requests serialize (politeness — e.g. StatCan asks Crawl-delay: 2). The
    host-semaphore map is now the PROCESS-WIDE one
    (:func:`http_fetch.host_semaphores`, scoped to the running loop) rather than a
    fresh dict per call: with one map per call, six questions fetching the same host
    concurrently each held their own semaphore and hit it six times at once. Every
    ``ladder.fetch_url`` task shares it, so each hop contends on ITS host's semaphore —
    chains from different initial hosts that converge on one final host still
    serialize there; the Tier-2 dataset fetches contend on the dwcdn host's semaphore
    the same way. Session is closed in ``finally``.

    Sharing the map buys that politeness at the cost of CROSS-QUESTION serialization: a
    same-host queue now forms across the concurrent questions, inside a
    ``RESOLUTION_SOURCE_WALL_TIMEOUT`` that was not raised and that discards work which
    already succeeded when it fires, so a question that loses the queue can lose every
    page it had already fetched rather than just the contended one (reproduced; the
    archived tail says 3 of 23 all-fail fetches ran the full per-request timeout). The
    acquire wait itself is deliberately unbounded — see FUTURE.md item 5, where both
    remedies (partial harvest, or a budget-bounded wait) are the operator's call.

    Teardown race guard (F5): the outer factory wraps this call in
    ``asyncio.wait_for``. When the wall-clock timeout fires, wait_for cancels
    this coroutine. Page fetches run as explicit tasks and are canceled and
    drained in ``finally`` before the session closes. The ordered Datawrapper
    hop is awaited directly, so cancellation reaches its active fetch before
    the same session closes.
    """
    # backends imports market_retrieval settlement helpers, which import this module's public URL
    # extractor. Keep the known-API wiring at the call boundary to break that genuine cycle.
    from metaculus_bot.research.known_api import backends, wiring  # noqa: PLC0415  # real circular import

    host_sems = host_semaphores()
    tasks: list[asyncio.Task[FetchResult]] = []
    started = time.monotonic()

    session_cm = guard._get_session()
    async with session_cm as session:
        try:
            # One context per URL: the rung attempts belong to that URL's result, while
            # the query and the wall-clock origin are the same for all of them.
            # ONE shared rung budget across this question's URLs, and one per-URL context each:
            # the Wayback cap is per question (every snapshot shares one host gate), while the
            # rung attempts belong to the URL they were spent on.
            shared_budget = context.QuestionRungBudget()
            known_api_policy = replace(
                policy.RESOLUTION_SOURCE_POLICY if fetch_policy is None else fetch_policy,
                known_api=wiring.build_known_api_fetcher(
                    session=session,
                    kalshi_detail_budget=backends.KalshiGetBudget(),
                ),
            )
            page_tasks = [
                asyncio.create_task(
                    ladder.fetch_url(
                        u,
                        policy=known_api_policy,
                        ctx=context.LadderContext(
                            query=query,
                            started=started,
                            shared=shared_budget,
                            fast_path=fast_path,
                            session=session,
                            host_sems=host_sems,
                        ),
                    )
                )
                for u in urls
            ]
            tasks.extend(page_tasks)
            page_results = list(await asyncio.gather(*page_tasks, return_exceptions=False))

            picks = resolution_datawrapper._select_datawrapper_charts(page_results)
            if not picks:
                return page_results
            # The hop is a SECOND network phase inside the provider's single 45s wall,
            # and its datasets share one CDN host, so the per-host politeness semaphore
            # serializes them — worst case MAX_CHARTS x the 20s HTTP timeout, on top of
            # whatever the page phase already spent. Unbounded, a slow CDN tail would
            # blow the outer wall and cancel the WHOLE provider, discarding Tier-1
            # pages that already fetched. So the hop gets only the wall budget the
            # pages left behind (minus a margin so this path returns before the outer
            # wait_for fires), degrades to the pages on its own timeout, and is skipped
            # outright when less than one typical CDN fetch's worth remains. Typical
            # cost is trivial — a poll CSV is tens of KB off a CDN, sub-second-to-~2s
            # per dataset (the validation receipts' live runs) — so the bound exists
            # for the tail, which is exactly what a wall cap is for.
            hop_budget_s = (
                RESOLUTION_SOURCE_WALL_TIMEOUT
                - (time.monotonic() - started)
                - RESOLUTION_SOURCE_DATAWRAPPER_HOP_WALL_MARGIN_S
            )
            if hop_budget_s < RESOLUTION_SOURCE_DATAWRAPPER_MIN_HOP_BUDGET_S:
                logger.warning(
                    "resolution_source: skipping the datawrapper hop (%d chart(s)) — %.1fs of wall "
                    "budget left; serving %d Tier-1 page result(s) without datasets",
                    len(picks),
                    hop_budget_s,
                    len(page_results),
                )
                return page_results
            dataset_results: list[FetchResult] = []
            try:
                async with asyncio.timeout(hop_budget_s):
                    dataset_results = await _fetch_datawrapper_datasets_in_order(
                        session, picks, page_results, host_sems
                    )
            except TimeoutError:
                logger.warning(
                    "resolution_source: datawrapper hop timed out after %.1fs; serving %d Tier-1 "
                    "page result(s) without datasets",
                    hop_budget_s,
                    len(page_results),
                )
                return page_results
            return resolution_datawrapper._interleave_dataset_results(page_results, picks, dataset_results)
        finally:
            # Whether we exit normally or via cancellation, cancel any still-
            # running task and let them settle before the session closes.
            # (No-op cost when everything already finished successfully.)
            for t in tasks:
                if not t.done():
                    t.cancel()
            # return_exceptions=True: drained tasks may surface CancelledError,
            # which is expected here.
            await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _log_fetch_outcome_markers(qid: int | None, results: list[FetchResult]) -> None:
    """Emit ONE greppable ``RESOLUTION_SOURCE_FETCH`` line per fetched URL.

    Per-URL outcomes used to live only in free-text log lines and in the published
    comment's provider-diagnostics block, so a cut like "cdc.gov is 0 successes in
    1,069 fetch records" meant re-scraping run logs that expire from GHA at 90
    days. This is the harvested form (spec ``resolution_source_fetch``,
    ``scripts/telemetry/markers.py``); the free-text outcome lines it replaces were
    deleted rather than kept beside it, so no fetch is logged twice.

    Emitted here, at the per-question aggregation point, because that is where the
    question id exists — threading it down through ``fetch_resolution_sources`` /
    ``ladder._fetch_one`` / the ladder's classification helpers would change the signature
    of the whole monkeypatched fetch surface to carry a value only a log line reads.

    Tier-2 dataset hops ride the same marker and are identified by their url, which
    is always ``static.dwcdn.net/data/<chart_id>.csv`` — that host is reachable no
    other way, so a query can partition cited pages from hop artifacts on it.
    ``status`` is the shared token (``ok`` for a success, else the verbatim
    ``FetchStatus``) and ``embeds`` names the routeless data-embed providers found in
    the page's raw HTML, which is what makes an unreadable-embed page queryable even
    when its prose made it a success.

    ``reason`` is appended only where the status alone is ambiguous —
    ``no_resolving_content``'s ``embed_shell`` vs ``thin_page`` vs the
    ``no_matching_passage`` of a document read in full that discusses nothing the question
    asks about vs the paid reader's ``not_addressed``, ``unreadable_document``'s
    ``no_text_layer`` vs ``encrypted`` vs
    ``malformed``, and the ``budget_skipped`` / ``parse_contention`` that say an
    ``unsupported_type`` was a document we were holding.
    ``route`` is appended only when a ladder rung produced the outcome. Both are
    appended rather than always emitted so every line the archive already holds stays
    byte-identical and an absent field keeps meaning "this does not apply", not "old
    record"; both sit at the tail in the order the marker spec's optional groups do.

    Each rung that FIRED also gets one ``RESOLUTION_SOURCE_ESCALATION`` line. The
    fetch line above carries only the final outcome, so on its own it cannot say
    whether a rung rescued the page or what the attempt cost — and ``wall_s`` is what
    decides whether a rung earns its latency under a close-derived time budget. Both
    ``outcome`` and ``wall_s`` are the RUNG's own (``RungAttempt``): the status that stood
    once that rung was over and what that rung alone cost, so on a page where a dead feed
    GET was followed by a rescuing render the first line reads the direct status and the
    second reads ``success``, and neither is billed for the other's latency. The ``url``
    on an escalation line is the URL the rung was invoked ON, which for a meta-refresh hop
    is the stub rather than the target the fetch line names, and for a ``route=rendered``
    rescue is the cited URL while the fetch line names the URL the browser landed on (since
    2026-09-04; before that boundary the two agreed, so a per-URL join against the cited URL
    is exact only for earlier records and otherwise keys on the escalation line).
    """
    for r in results:
        logger.info(fetch_markers.fetch_marker_line(r, qid=qid, caller=LADDER_CALLER_RESOLUTION_SOURCE))
        for line in fetch_markers.escalation_marker_lines(r, qid=qid, caller=LADDER_CALLER_RESOLUTION_SOURCE):
            logger.info(line)


def _rung_counts(results: list[FetchResult]) -> dict[str, int]:
    """Per-rung attempt counts for ``details["counts"]``.

    Zeroes are kept: they render nothing in the diagnostics line but survive into the
    archive, which is what makes "the rung existed and never fired" distinguishable
    from "this record predates the rung".
    """
    attempts = [attempt for r in results for attempt in r.rung_attempts]
    fired_by_rung = Counter(attempt.rung for attempt in attempts if not attempt.skipped_reason)
    # Typed on `RungSkipReason` so every literal key indexed below is checked: a misspelt reason
    # is a type error rather than a permanently-zero count silently absent from the archive.
    skips_by_reason: Counter[RungSkipReason | Literal[""]] = Counter(
        attempt.skipped_reason for attempt in attempts if attempt.skipped_reason
    )
    budget_skips_by_rung = Counter(attempt.rung for attempt in attempts if attempt.skipped_reason == "wall_budget")
    return {
        "meta_refresh_hops": fired_by_rung["meta_refresh"],
        # No `impersonate_rescues` key beside this: rescues are read off `route=` on the fetch
        # marker, which already partitions the population by rung. The only per-rung rescue count
        # in this dict is `chrome_metric_withholds_rescued`, and that exists because it needs a
        # join the marker cannot express.
        "impersonate_attempts": fired_by_rung["impersonate"],
        "pdf_documents_read": fired_by_rung["pdf_local"],
        "rendered_attempts": fired_by_rung["rendered"],
        "derived_api_reads": fired_by_rung["derived_api"],
        "wayback_attempts": fired_by_rung["wayback"],
        "url_context_reads": fired_by_rung["url_context"],
        "rung_budget_skips": skips_by_reason["wall_budget"],
        # The same skips broken out per rung, because the aggregate cannot say WHICH rung the
        # provider's wall is binding on — and at the paid flag's rollout "how often does the
        # paid rung get starved by the pages before it" is the question. The total stays as it
        # is, since the archive already reads it.
        **{f"{rung}_budget_skips": budget_skips_by_rung[rung] for rung in context._BUDGET_GATED_RUNGS},
        # Its own count: a rung a caller never runs reads otherwise as one no page ever earned.
        "rung_not_enabled_skips": skips_by_reason["rung_not_enabled"],
        # Its own count rather than folded into the budget skips: a document left unread
        # because two others were already parsing says the 2-slot gate is the binding
        # constraint, which is a different thing to fix than a question that ran late.
        "pdf_contention_skips": skips_by_reason["parse_contention"],
        # Also its own count, for the same reason: a browser rung that never rendered because
        # Chromium is missing on the runner (the install step is `continue-on-error` in every
        # workflow, so its absence is by design) says something different from a question that
        # ran out of wall, and both are invisible in `rendered_attempts`.
        "renderer_unavailable_skips": skips_by_reason["renderer_unavailable"],
        # Its own count too: a render that launched and was cut off by the transport's DOM-read
        # cap is a page that keeps navigating, which is a fact about the page, whereas the two
        # counts above are facts about the runner and about the question's clock (the rung's own
        # outer cut, which fires while the render is still queued, is a `wall_budget` skip).
        # Folding it into either would hide the population the ogimet receipt (2026-09-03) is
        # the first member of.
        "render_timeout_skips": skips_by_reason["render_timeout"],
        # Its own count: the browser was answered a non-200 where the direct GET got a 200, the
        # edge telling Chromium apart. Counted with the fired renders it read as a render that
        # produced chrome again, and the rate at which the runner's browser is refused is the
        # question the escalation ladder's case rests on.
        "render_non_200_skips": skips_by_reason["render_non_200"],
        # Its own count: Chromium rendered the page and the DOM was over `RENDERED_DOM_MAX_CHARS`,
        # a fact about the page that used to be folded into `renderer_unavailable_skips`, where it
        # pointed triage at the Playwright install.
        "render_dom_too_large_skips": skips_by_reason["render_dom_too_large"],
        # Its own count: Chromium's main frame landed on a host other than the pinned one (a
        # server-side redirect hop the route guard never sees), so the transport refused the DOM
        # unread. A fact about the page, and the one count that says how often a cited page sends
        # the browser somewhere else, which is what prices the host-equality rule.
        "render_off_host_skips": skips_by_reason["render_off_host"],
        # And its own count: a browser rung skipped because an earlier question in this run
        # already rendered the same URL to nothing is the memo doing its job, not a runner without
        # Chromium — folded into `renderer_unavailable_skips` it inflated the install-failed signal.
        "rendered_no_text_skips": skips_by_reason["rendered_no_text"],
        # Also its own count: a question that spent its two snapshot attempts on earlier
        # cited URLs is a question whose per-question cap is binding, which is a different
        # thing to tune than a question that ran out of wall.
        "wayback_cap_skips": skips_by_reason["wayback_cap"],
        # The paid rung's analogue: a question that spent its per-question paid-read budget on
        # earlier cited URLs, which is the spend cap binding rather than the wall or the flag.
        "url_context_cap_skips": skips_by_reason["url_context_cap"],
        # Its own count: an expensive rung declined because the QUESTION's close-derived budget
        # put it on the fast path, which is a fact about the question's window rather than about
        # the provider's own 45 s wall (`rung_budget_skips`) — the two are tuned differently.
        "fast_path_skips": skips_by_reason["fast_path"],
        # Its own count because it is the free pre-check EARNING its request: a host that
        # disallows Google-Extended refuses the read server-side, so this is spend avoided
        # rather than a page lost, and it must not read as a failure.
        "url_context_robots_skips": skips_by_reason["robots_disallowed"],
        # Its own count because it is a MISCONFIGURATION rather than a tuning signal: with the
        # flag on and GOOGLE_API_KEY unset the paid rung fires nowhere, and without this key
        # that run is byte-identical in the archive to one with the flag off.
        "url_context_no_api_key_skips": skips_by_reason["no_api_key"],
        # Its own count for the same reason: with the kill switch off the impersonated retry fires
        # nowhere, and without this key that run is byte-identical in the archive to one where no
        # cited page ever earned the retry.
        "impersonate_disabled_skips": skips_by_reason["impersonate_disabled"],
        # Its own count: the impersonated retry declined because a hop's host would not pin to a
        # vetted public address. On the first hop nothing was dialed, and since the direct fetch
        # resolved that host through the filtering resolver moments earlier a nonzero count means
        # DNS disagreed with itself (a flake, or a rebinding host that flipped); on a later redirect
        # hop the earlier hops WERE dialed and the target is one the direct fetch never resolved, so
        # the DNS-disagreement reading does not apply to that case.
        "impersonate_unpinnable_skips": skips_by_reason["impersonate_unpinnable"],
        # Its own count: the per-run memo declining a cited URL on a host that already answered the
        # impersonated client with a block status this run, which is the memo doing its job rather
        # than a failure, the distinction `rendered_no_text_skips` draws for the browser. The memo
        # is process-global and shared with gap-fill v2, so the earlier refusal may have been a v2
        # fetch of a URL no question cited.
        "impersonate_host_refused_skips": skips_by_reason["impersonate_host_refused"],
        # The extractor policy's decisions, per cited URL. `chrome_metric_withholds`: the
        # line-shape metric withheld an HTML extraction of the URL somewhere on its ladder — the
        # final result's own (including a chart-rescued page, whose chart block still published
        # without that text) or the direct fetch's, carried onto the rung result that replaced it
        # (`ladder._fetch_one`). `chrome_metric_withholds_rescued`: the subset a rung past the direct
        # fetch then served (`route` is not `direct` and the result is a success) — the policy's
        # headline win, a menu tree withheld and the price table rendered, which summed off the
        # rescue's own flag alone reached neither key. `precision_fallback_rescues`: the published
        # text is the precision re-extraction, taken after the default one failed the metric.
        "chrome_metric_withholds": sum(1 for r in results if r.chrome_metric_withheld),
        "chrome_metric_withholds_rescued": sum(
            1 for r in results if r.chrome_metric_withheld and r.status == "success" and r.route != "direct"
        ),
        "precision_fallback_rescues": sum(1 for r in results if r.precision_rescued),
    }


def _document_query(question: MetaculusQuestion) -> str:
    """The text a cited document's passages are ranked against.

    Title plus resolution criteria, because those are the two fields that say what the
    question is graded on — and the ranking is BM25 over the document, so the criteria's
    own vocabulary ("laboratory-confirmed cases", "final revised estimate") is exactly
    what should pull the right paragraph out of a 220-page report. Fine print is left
    out: it is mostly procedural boilerplate about ambiguity and annulment, which would
    dilute the term set with words no relevant passage contains.
    """
    return f"{question.question_text or ''} {question.resolution_criteria or ''}".strip()


def resolution_source_provider(is_benchmarking: bool = False, *, fast_path: bool = False) -> ResearchCallable:
    """Factory returning the async ResearchCallable for the resolution-source fetcher.

    Gating (both hard):

    - ``is_benchmarking=True`` short-circuits to ``""`` (leakage guard — current
      page content post-dates any backtest window, same rationale as the
      prediction-market provider).
    - Env flag ``RESOLUTION_SOURCE_ENABLED`` must be truthy.

    ``fast_path`` is the question's time-budget thin-window mode, the same flag the
    orchestrator uses to drop the slow search providers. This provider is cheap and
    hard-capped, so it still runs; what the flag changes is that the two EXPENSIVE rungs of
    its escalation ladder — a 12-35 s Chromium launch and the paid reader — decline, recorded
    under ``counts["fast_path_skips"]``.

    Returns section BODY only; the orchestrator prepends the ``## Resolution
    Source Snapshot`` header. Inner ``### {url}`` headers stay at h3 — the
    orchestrator's heading demotion only touches h1/h2, and h3 is already
    correctly nested under the h2 provider header.
    """

    async def _fetch(question: MetaculusQuestion) -> str:
        if is_benchmarking:
            return ""
        if not env_flag_enabled(RESOLUTION_SOURCE_ENABLED_ENV):
            return ""

        urls = select_fetchable_urls(question.resolution_criteria, question.fine_print)
        if not urls:
            return ""

        try:
            results = await asyncio.wait_for(
                fetch_resolution_sources(urls, query=_document_query(question), fast_path=fast_path),
                timeout=RESOLUTION_SOURCE_WALL_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(f"resolution_source: wall-clock timeout after {RESOLUTION_SOURCE_WALL_TIMEOUT}s")
            return ""

        # CITED pages only. A withheld Tier-2 dataset is a hop artifact, not an
        # unfetched cited URL, and counting it here inflated the ratio with
        # by-design withholds (`stale_data`) on exactly the tracker questions the
        # hop serves. Datasets get their own count so both stay readable.
        cited = [r for r in results if r.chart_id is None]
        unfetched = [r for r in cited if r.status != "success"]
        n_datasets_withheld = sum(1 for r in results if r.chart_id is not None and r.status != "success")
        if unfetched or n_datasets_withheld:
            # Counts rather than a verdict. This line used to assert the ladder "rescued none of
            # them" whenever anything was unfetched, which is false on the ordinary mixed
            # question — Wayback serving one host while a second stays walled — and the summary is
            # where a run log gets read first. A rescue is a success no direct fetch produced,
            # the same reading `_rung_counts` takes; the statuses say what the losses actually
            # were instead of naming two of them by hand.
            n_rescued = sum(1 for r in cited if r.status == "success" and r.route != "direct")
            lost_statuses = ",".join(sorted({r.status for r in unfetched})) or "none"
            logger.info(
                f"resolution_source: {len(unfetched)}/{len(cited)} cited urls unfetched "
                f"({lost_statuses}); {n_rescued} rescued by a later rung; "
                f"{n_datasets_withheld} embedded dataset(s) withheld",
            )
        qid = getattr(question, "id_of_question", None)
        _log_fetch_outcome_markers(qid, results)
        record_raw_research(qid=qid, provider="resolution_source", payload=results)
        # Per-URL outcome map for the diagnostics block: even when the provider
        # returns a non-empty notice (all URLs failed → status `ok`), this surfaces
        # WHICH sources were lost so the block doesn't read as fully healthy.
        record_provider_detail(
            qid,
            "resolution_source",
            {"sources": _fetch_result_sources(results), "counts": _rung_counts(results)},
        )
        return resolution_presentation.format_resolution_sections(results, datetime.now(UTC))

    return _fetch
