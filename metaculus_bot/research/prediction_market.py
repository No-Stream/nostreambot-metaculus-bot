"""Prediction-market research provider: the seam module for ranked market retrieval.

Retrieves markets from Polymarket + Kalshi + Manifold + PredictIt that bear on a Metaculus
question, and returns a `MarketSnapshot` the forecaster reads as a peer cross-check. The
retrieval machinery lives in `metaculus_bot.research.market_retrieval`; this module owns the
provider seam, the snapshot orchestrator, both telemetry markers, and the one retry-wrapped LLM
invocation the two LLM stages share.

Four stages per question:

    1a  PREFETCH      Kalshi's complete open-events catalogue + PredictIt's whole dump
    1b  QUERY AUTHOR  one LLM call adding domain vocabulary  (concurrent with 1a)
    2   VENUE SEARCH  Manifold + Polymarket, every query, per-query failure isolation
    2.5 ENRICH        one Manifold detail GET per candidate, for the rules text
    3   POOL ASSEMBLY three channels unioned; channel order IS the ranking
    4   RANK          one LLM call over the whole pool, up to 8 rows, model's order

The stages that make no LLM call live next door: `market_retrieval.session_state` holds the
per-session caches, the degradation counters, the aiohttp session factory and the two catalogue
prefetches, and `market_retrieval.snapshot_stages` holds the per-question context, the source
ledger, venue search, pool assembly and the post-rank accounting. Both are re-exported here,
because this module is the seam every outside consumer and every test patches against.

Two measured facts shaped this, and both are things NOT to "improve" (the bake-off receipts
are in `scratch/bakeoff_run_2026-08-03/results/`):

- **Selection binds, not generation.** A perfect ranker over the pool that already exists
  reaches 14/16 questions; the same pool's deterministic top-4 reaches 5/16. So generation is
  recall-maximal — no score floor anywhere — and all the judgment sits in one ranking call.
- **The render step was the weak link, not the LLM.** In the previous design a keep/drop
  auditor voted KEEP on 58 of 58 near-identical rows and a venue round-robin then evicted 43
  of them. There is no venue quota, no fairness interleave and no post-hoc re-scoring in this
  pipeline, at any stage.

Two behaviours the old keyword/fuzzy design had are deliberately gone. There is no fuzzy score
FLOOR (it discarded the adjacent-cut markets that carry most of the evidential value; the
fuzzy scorer survives as a way to ORDER Kalshi's ~9,762 events down to a promptable 100). And
the provider path passes `as_of=None`: the filter dropped every market closing before the
question resolved, which is precisely the "same quantity, adjacent month" class the ranked arm
scored most of its wins on, and 20 of 47 archived runs had Polymarket fetch candidates and
render nothing because of it. The parameter and the cache key survive for explicit callers
(backtests, replay tooling), where leakage defence is real; the drop itself now happens inside
pool assembly, so an ineligible row frees its width slot instead of being deleted after the
width has already been spent.

Soft-fail on every path. This provider returns an empty snapshot on any failure and never
raises — a broken venue API must never break a forecast — so the degradation counters below
are the ONLY route by which an outage reddens CI.

One analysis note follows from the ranker being allowed to say nothing. A question where it
DELIBERATELY returns zero rows over a non-empty pool now renders a one-sentence "no sufficiently
relevant market among N candidates" notice (see `format_snapshot_for_research`), so the
`## Prediction Market Snapshot` header is present and the provider diagnostics read `ok` rather
than `empty`. Records from before that change (pre 2026-08-24) render no section at all on those
questions, so a header-scan `providers_used` reconstruction drops the provider there while an
artifact record still lists it under `providers_attempted` — a fall in prediction-market presence
across old records can mean "the ranker declined" rather than "the provider broke". The
`MARKET_RANKING:` line's `outcome=` field distinguishes the two in every era.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import aiohttp
import openai
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    KALSHI_CATALOGUE_WALL_TIMEOUT,
    MARKET_QUERY_AUTHOR_BACKOFFS,
    MARKET_QUERY_AUTHOR_WALL_TIMEOUT,
    MARKET_RANKER_BACKOFFS,
    MARKET_RANKER_WALL_TIMEOUT,
    PREDICTION_MARKET_TIMEOUT,
    PREDICTION_MARKETS_ENABLED_ENV,
    env_flag_enabled,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.llm_configs import MARKET_QUERY_AUTHOR_LLM_CONFIG, MARKET_RANKER_LLM_CONFIG
from metaculus_bot.llm_retry import invoke_with_transient_retry
from metaculus_bot.research.market_retrieval import generation, ranking, rendering
from metaculus_bot.research.market_retrieval.http import HTTP_RETRY_BACKOFF_SECS, PLATFORM_HTTP_TIMEOUT
from metaculus_bot.research.market_retrieval.queries import (
    build_query_author_prompt,
    dedupe_queries,
    deterministic_queries,
    parse_query_author,
)

# The process-scoped state lives in `session_state`, imported here because THIS module is the
# patch surface: `prediction_market._get_session` is what tests replace and what
# `fetch_market_snapshot` reads, and `prediction_market._KALSHI_CACHE` is the same dict object the
# prefetches read. The names with no local caller are re-exports for the orchestrator's
# degradation properties — which import them from HERE at call time, so a test patching them here
# is still what the orchestrator sees — and for the test suite.
from metaculus_bot.research.market_retrieval.session_state import (
    _KALSHI_CACHE,  # noqa: F401  # re-export: tests assert on the cache's TTL-pinning behaviour
    _KALSHI_CATALOGUE_IN_FLIGHT,  # noqa: F401  # re-export: tests assert the single-flight guard is reset with its cache
    _PREDICTIT_CACHE,  # noqa: F401  # re-export
    _PREDICTIT_UNIVERSE_IN_FLIGHT,  # noqa: F401  # re-export
    _SNAPSHOT_CACHE,
    _bump_kalshi_catalogue_failure,  # noqa: F401  # re-export: tests seed the counter through the seam
    _bump_source_loss,
    _get_session,
    _kalshi_catalogue,
    _predictit_universe,
    _reset_session_caches,  # noqa: F401  # re-export: the test-suite + session-start reset hook
    kalshi_catalogue_fetch_failures,  # noqa: F401  # re-export: read by the orchestrator's alertable count
    prediction_market_source_losses,  # noqa: F401  # re-export: read by the orchestrator's alertable count
    reset_series_degradation_counter,  # noqa: F401  # re-export: called at run start by forecast_questions
    reset_source_loss_counter,  # noqa: F401  # re-export: called at run start by forecast_questions
)
from metaculus_bot.research.market_retrieval.snapshot_stages import (
    _assemble_pool_stage,
    _pool_positions,
    _PrefetchResult,
    _record_venue_health,
    _run_venue_search_stage,
    _SnapshotContext,
)

# The row types and the liquidity vocabulary live in the market_retrieval package; this module
# re-exports them because it is the seam every consumer imports from — `raw_log`'s `asdict`,
# `provider_health`'s `getattr`, and the test suite all reach them through
# `metaculus_bot.research.prediction_market`.
from metaculus_bot.research.market_retrieval.types import (
    LIQUIDITY_DECENT_USD,  # noqa: F401  # re-export: consumed by the liquidity-contract tests
    LIQUIDITY_DEEP_USD,  # noqa: F401  # re-export
    LIQUIDITY_STRONG_USD,  # noqa: F401  # re-export
    MANIFOLD_HIGH_BETTORS,  # noqa: F401  # re-export
    MANIFOLD_THIN_BETTORS,  # noqa: F401  # re-export
    MarketChild,  # noqa: F401  # re-export: the multi-outcome sub-row, archived inside MarketMatch
    MarketMatch,
    MarketSnapshot,
    ScalarEstimate,  # noqa: F401  # re-export: a scalar market's value, archived inside MarketMatch
    SettlementSource,  # noqa: F401  # re-export: archive shape + test constructions
    _FetchTally,
    _liquidity_label,  # noqa: F401  # re-export
)
from metaculus_bot.research.provider_diagnostics import is_lost_source, record_provider_detail
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.raw_log import record_raw_research
from metaculus_bot.time_utils import _as_utc

logger = logging.getLogger(__name__)


# Worst-case wall clock for one snapshot, computed from the same constants the stages use so a
# raised wall cannot silently outgrow PREDICTION_MARKET_TIMEOUT. A previous 31.0-versus-30.0
# overrun shipped once, which is why the arithmetic is code rather than a comment: stages 1a and
# 1b run concurrently so the chain takes their max, and the rest is serial. Per-stage worst case
# is `(len(backoffs) + 1) * wall + sum(backoffs)`.
def _llm_stage_worst(wall_timeout: float, backoffs: tuple[float, ...]) -> float:
    return (len(backoffs) + 1) * wall_timeout + sum(backoffs)


_HTTP_STAGE_WORST = PLATFORM_HTTP_TIMEOUT * 2 + HTTP_RETRY_BACKOFF_SECS
SNAPSHOT_STAGE_BUDGET_S: float = (
    max(
        _llm_stage_worst(MARKET_QUERY_AUTHOR_WALL_TIMEOUT, MARKET_QUERY_AUTHOR_BACKOFFS),
        KALSHI_CATALOGUE_WALL_TIMEOUT,
        _HTTP_STAGE_WORST,
    )
    + _HTTP_STAGE_WORST
    + generation.MANIFOLD_DETAIL_WALL_S
    + _llm_stage_worst(MARKET_RANKER_WALL_TIMEOUT, MARKET_RANKER_BACKOFFS)
)


# ---------------------------------------------------------------------------
# The one LLM invocation helper, shared by both LLM stages
# ---------------------------------------------------------------------------


async def _invoke_market_llm(
    config: dict, prompt: str, *, wall_timeout: float, backoffs: tuple[float, ...], label: str
) -> str:
    """Invoke one market-retrieval LLM stage under the elapsed-gated retry wrapper.

    ONE helper for both stages, deliberately: it is a single patch point for the tests, and it
    makes it impossible for the query author and the ranker to drift on retry policy. Each
    stage passes its OWN wall and backoff ladder, which is what keeps the serial chain of
    worst cases inside `SNAPSHOT_STAGE_BUDGET_S`.

    Both configs pin `allowed_tries=1`, so this wrapper is the SOLE retry layer: it gives the
    call its own wall cap (otherwise the snapshot-level `wait_for` is the only bound and a
    stalled stage takes the whole snapshot down with it) and replaces forecasting-tools'
    un-gated `random.uniform(5, 10)` tenacity sleep with one bounded, elapsed-gated backoff
    that never fires on a deterministic client error.

    Returns `""` on any transient failure. Both callers' parsers treat an empty completion as
    unusable, so the soft-fail and the unparseable-output path are one path rather than two.
    """
    # Constructor errors are config bugs (bad model slug, missing API key wiring) and should
    # crash loudly. Only `.invoke` is expected to face transient LLM errors.
    llm = build_llm_with_openrouter_fallback(**config)
    try:
        return await invoke_with_transient_retry(
            lambda: llm.invoke(prompt),
            wall_timeout=wall_timeout,
            label=label,
            backoffs=backoffs,
        )
    # `openai.APIError` and NOT `litellm.exceptions.APIError`: every litellm transport exception
    # (Timeout, RateLimitError, APIConnectionError, InternalServerError, ServiceUnavailableError)  # noqa: ERA001  # prose list of exception classes, not code
    # subclasses the openai root but NOT litellm's own APIError, so catching the latter caught
    # only a bare `APIError` and let every realistic provider blip escape — past `_rank_pool`'s
    # `except RankingUnusable` to the snapshot-level net, discarding the WHOLE snapshot where the
    # documented fail-open would have rendered the pool-order slate. Same idiom as
    # `research/orchestrator.py` and `ablation/leakage_screen.py`. Deliberately NOT `Exception`:
    # the TypeError/AttributeError family must still crash (§2 fail-fast), and `CancelledError`
    # subclasses neither root so the `wait_for` boundary stays intact.
    except (TimeoutError, openai.APIError, RuntimeError):
        logger.warning(f"{label} LLM call failed", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Stage 1b — the query author (concurrent with 1a)
# ---------------------------------------------------------------------------


async def _authored_queries(title: str, resolution_criteria: str) -> tuple[tuple[str, ...], str]:
    """Extra search queries from one LLM call, ADDITIVE to the deterministic set.

    Additive rather than replacing is load-bearing rather than stylistic: a replacing author's
    failure mode is an empty query set, which is indistinguishable from "no markets exist" —
    the silent failure that hid the Manifold breakage for 17+ days. Additive, its worst case is
    "no gain", so `()` costs no recall and is still reported as a lost source (a permanently
    dead author would otherwise be invisible, which is the exact failure class the degradation
    counters exist for).

    Runs concurrently with the catalogue prefetches because those need no queries at all — the
    catalogue IS the venue, and the settlement join keys on domains.
    """
    completion = await _invoke_market_llm(
        MARKET_QUERY_AUTHOR_LLM_CONFIG,
        build_query_author_prompt(title, resolution_criteria),
        wall_timeout=MARKET_QUERY_AUTHOR_WALL_TIMEOUT,
        backoffs=MARKET_QUERY_AUTHOR_BACKOFFS,
        label="market_query_author",
    )
    extra = parse_query_author(completion)
    if not extra:
        return (), "error(unusable)"
    return extra, f"ok({len(extra)})"


# ---------------------------------------------------------------------------
# Stage 4 — ranking
# ---------------------------------------------------------------------------


async def _rank_pool(question: Any, pool: generation.PoolResult) -> tuple[list[MarketMatch], str, str, int]:
    """One LLM call over the whole pool. Returns ``(rows, token, outcome, prompt_chars)``.

    Fails open to the pool-order top rows, which is literally the head of what the model was
    shown — so a fail-open is a truncation of the ranker's own input rather than a different
    pipeline. `outcome` is `ranked` / `failopen` / `empty`, for the telemetry line.

    An EMPTY ARRAY from the model is a valid answer, not a failure: width is the model's choice
    in 0..8, it took 3 and 6 rows on the two measured true negatives against a fixed-8 arm's
    12, and conflating `[]` with unusable output would delete the whole adaptive-width
    mechanism. Everything else the parser cannot read as a ranking fails open — output that is
    not a JSON array of ranking objects (`RankingUnusable`), and a non-empty array yielding no
    usable row (`RankingShapeRegression`, e.g. a renamed index key). The two are logged with a
    `reason=` that tells them apart, because one says the model emitted prose and the other says
    our prompt/parser contract broke.
    """
    if not pool.candidates:
        # Nothing to rank, so the stage does not run and there is no LLM call to lose. A
        # non-loss token: an empty pool is the venues' story to tell (their own tokens, and
        # Signal C for the catalogues), not a ranking failure.
        return [], "none", "empty", 0

    prompt = ranking.build_ranker_prompt(
        ranking.RankerQuestion(
            title=getattr(question, "title", None) or getattr(question, "question_text", "") or "",
            qtype=type(question).__name__,
            unit=str(getattr(question, "unit_of_measure", "") or ""),
            resolution_criteria=getattr(question, "resolution_criteria", "") or "",
            fine_print=getattr(question, "fine_print", "") or "",
        ),
        pool.candidates,
    )
    completion = await _invoke_market_llm(
        MARKET_RANKER_LLM_CONFIG,
        prompt,
        wall_timeout=MARKET_RANKER_WALL_TIMEOUT,
        backoffs=MARKET_RANKER_BACKOFFS,
        label="market_ranker",
    )
    try:
        picks = ranking.parse_ranking(completion, len(pool.candidates))
    except ranking.RankingUnusable as exc:
        # A marker line rather than prose, because the archive has to be able to count these: a
        # fail-open renders retrieval order under `[ranking unavailable]`, and the run logs it
        # sits in expire from GHA at 90 days. `reason=shape_regression` is the one that means OUR
        # contract broke (a renamed index key would otherwise have passed silently as `ok(0)`);
        # `reason=unreadable` means the model emitted something that is not a ranking array.
        reason = "shape_regression" if isinstance(exc, ranking.RankingShapeRegression) else "unreadable"
        logger.warning(
            f"MARKET_RANKING_DEGRADED: question={getattr(question, 'id_of_question', None)} "
            f"pool={len(pool.candidates)} reason={reason} "
            f"detail=falling back to retrieval order; {exc}"
        )
        return ranking.fail_open_slate(pool.candidates), f"error({type(exc).__name__})", "failopen", len(prompt)
    # Read directly, not through getattr-with-a-default: both fields are declared on
    # MetaculusQuestion, and a default would turn a future rename into a cap that silently
    # never fires again on a surface that already fires on zero archived rows. A question
    # whose `open_time` is genuinely None is still handled — `cap_stale_top_tier` no-ops on
    # it, which is the replay-tool case. The `getattr` above and elsewhere in this module
    # are the module's older convention on a `question: Any` parameter; sweeping them is
    # tracked separately, so the two shapes sit side by side here on purpose.
    ranked_rows = ranking.cap_stale_top_tier(
        ranking.apply_picks(pool.candidates, picks),
        question_open_time=question.open_time,
    )
    _log_tier_caps(question.id_of_question, ranked_rows)
    return ranked_rows, f"ok({len(picks)})", "ranked", len(prompt)


def _log_tier_caps(qid: int | None, ranked_rows: Sequence[MarketMatch]) -> None:
    """One INFO line per question whose top-tier grade the staleness cap refused, and none otherwise.

    Silent on the overwhelmingly common no-cap case, so a line in the run log means the ranker
    graded a long-closed market as same-date. The note itself also rides the rendered table and the
    archived snapshot (`MarketMatch.tier_cap_note`), which is what makes the incidence answerable
    offline after the 90-day GHA log expiry; this line is the prod-log half.
    """
    capped = [row for row in ranked_rows if row.tier_cap_note]
    if not capped:
        return
    logger.info(
        f"MARKET_TIER_CAPPED: question={qid} rows={len(capped)} "
        f"capped={','.join(f'{row.platform}@{row.rank}' for row in capped)}"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _as_of_cache_key(as_of: datetime | None) -> str:
    """Render `as_of` as a cache-key string, normalizing through the shared `_as_utc`.

    Two callers passing the same instant spelled differently (naive-UTC vs `+00:00`) must
    collide on one key, so the normalization has to match the one `assemble_pool` applies
    to the same value — hence both going through `time_utils._as_utc` rather than each
    respelling the naive-vs-aware branch.
    """
    if as_of is None:
        return "none"
    return _as_utc(as_of).isoformat()


async def fetch_market_snapshot(
    question: Any,
    *,
    platforms: tuple[str, ...] = ("polymarket", "kalshi", "manifold", "predictit"),
    timeout: float = PREDICTION_MARKET_TIMEOUT,  # noqa: ASYNC109
    as_of: datetime | None = None,
) -> MarketSnapshot:
    """Run the four-stage pipeline and return the ranked snapshot.

    Soft-fails on any error: returns an empty `MarketSnapshot` + a WARNING. A broken
    prediction-market API should never break a forecast run.

    The `timeout` default is the same constant the provider path passes, because any lower
    default is one the pipeline cannot succeed under — stage 1a alone outruns the 5.0s this
    carried while `SNAPSHOT_STAGE_BUDGET_S` grew past 130s, so a direct caller (backtest,
    replay tool) omitting the argument got a guaranteed `error(timeout)` and a source-loss bump.

    `as_of` (backtest leakage defence) drops candidates whose `close_time` is at or before it.
    The provider path passes None — see the module docstring for why that filter was actively
    destroying the evidence class this pipeline exists to surface — and explicit callers that
    genuinely need leakage defence pass their own instant.
    """
    qid = getattr(question, "id_of_question", None)
    cache_key = (qid, _as_of_cache_key(as_of)) if qid is not None else None
    if cache_key is not None:
        cached_snap = _SNAPSHOT_CACHE.get(cache_key)
        if cached_snap is not None:
            return cached_snap

    # Session lifecycle: create the aiohttp session at the orchestrator level so cleanup
    # happens OUTSIDE the wait_for cancellation boundary. wait_for kills inner work cleanly,
    # then the surrounding context manager runs session.close().
    session_cm = _get_session()
    try:
        async with session_cm as session:
            try:
                snapshot = await asyncio.wait_for(
                    _fetch_market_snapshot_impl(question, session=session, platforms=platforms, as_of=as_of),
                    timeout=timeout,
                )
            except TimeoutError:
                logger.warning(f"Prediction-market snapshot TIMEOUT after {timeout}s for qid={qid}")
                # A whole-provider failure needs its own loss token: with an empty `sources`
                # map the diagnostics line renders no suffix at all, so a dead snapshot is
                # indistinguishable from one that was never asked for. It also counts toward
                # alertable_count — losing every source at once is strictly worse than losing
                # one, which already alerts.
                _bump_source_loss()
                return MarketSnapshot(matches=[], sources={"snapshot": "error(timeout)"})
    except Exception as e:  # HARNESS-SCAN-EXEMPT-broad-except
        # Outer safety net; should not normally fire -- investigate if seen. Re-raising after
        # logging would defeat the soft-fail contract the rest of the bot depends on, so we
        # swallow + log here. Inner narrow handlers in the venue helpers cover the common paths.
        logger.warning("Prediction-market snapshot FAILED (soft-fail returning empty)", exc_info=True)
        _bump_source_loss()
        return MarketSnapshot(matches=[], sources={"snapshot": f"error({type(e).__name__})"})

    if cache_key is not None:
        _SNAPSHOT_CACHE[cache_key] = snapshot
    return snapshot


async def _run_prefetch_stage(ctx: _SnapshotContext) -> _PrefetchResult:
    """Stage 1a PREFETCH ‖ stage 1b QUERY AUTHOR, concurrently."""
    stage_one: dict[str, asyncio.Task[Any]] = {
        "query_author": asyncio.create_task(_authored_queries(ctx.title, ctx.resolution_criteria))
    }
    if "kalshi" in ctx.platforms:
        stage_one["kalshi"] = asyncio.create_task(_kalshi_catalogue(ctx.session, qid=ctx.qid))
    if "predictit" in ctx.platforms:
        stage_one["predictit"] = asyncio.create_task(_predictit_universe(ctx.session, qid=ctx.qid))
    outcomes = dict(zip(stage_one, await asyncio.gather(*stage_one.values(), return_exceptions=True), strict=True))

    def _outcome(name: str, default: Any) -> Any:
        """One stage-one outcome, converting a residual raised error into a lost source.

        The helpers each soft-fail on the errors their transports raise, so anything landing
        here is a residual (a transport error escaping a stub, a bug). Reporting it as this
        source's loss rather than letting it kill the snapshot matches what the old per-platform
        narrow catch did, and keeps one broken venue from silencing the other three.
        """
        outcome = outcomes.get(name, default)
        if isinstance(outcome, BaseException):
            logger.warning(f"Market stage {name} raised (soft-fail): {type(outcome).__name__}: {outcome}")
            ctx.ledger.record_loss(name, f"error({type(outcome).__name__})")
            return default
        return outcome

    extra_queries, author_token = _outcome("query_author", ((), "error(unusable)"))
    kalshi_events, kalshi_token = _outcome("kalshi", ([], "none"))
    predictit_markets, predictit_tally = _outcome("predictit", ([], _FetchTally()))

    ctx.ledger.report("query_author", author_token)
    if "kalshi" in ctx.platforms:
        ctx.ledger.report("kalshi", kalshi_token)

    return _PrefetchResult(
        extra_queries=extra_queries,
        kalshi_events=kalshi_events,
        predictit_markets=predictit_markets,
        predictit_tally=predictit_tally,
    )


async def _rank_stage(
    ctx: _SnapshotContext,
    pool: generation.PoolResult,
    pool_by_venue: dict[str, list[MarketMatch]],
) -> list[MarketMatch]:
    """Stage 4 RANK, then emit the degradation WARN and both telemetry markers."""
    ranked_rows, ranking_token, outcome, prompt_chars = await _rank_pool(ctx.question, pool)
    ctx.ledger.report("ranking", ranking_token)

    if ctx.ledger.lost:
        # One WARN naming every degraded source: this counter reddens CI, and a red run whose
        # cause isn't named in the log is what teaches people to ignore alerts.
        logger.warning(f"Prediction-market sources degraded (alertable): {', '.join(ctx.ledger.lost)}")

    _log_ranking_telemetry(ctx.qid, pool, ranked_rows, outcome=outcome, prompt_chars=prompt_chars)
    _record_venue_health(
        ctx.qid, pool, ranked_rows, pool_by_venue=pool_by_venue, sources=ctx.ledger.tokens, platforms=ctx.platforms
    )
    return ranked_rows


async def _fetch_market_snapshot_impl(
    question: Any,
    *,
    session: aiohttp.ClientSession,
    platforms: tuple[str, ...],
    as_of: datetime | None,
) -> MarketSnapshot:
    """The four stages. See the module docstring for the shape and why it is that shape."""
    ctx = _SnapshotContext.build(question, session=session, platforms=platforms, as_of=as_of)
    prefetch = await _run_prefetch_stage(ctx)
    all_queries = dedupe_queries([*deterministic_queries(ctx.title), *prefetch.extra_queries])
    venue_search_results = await _run_venue_search_stage(ctx, all_queries)
    pool, pool_by_venue = await _assemble_pool_stage(ctx, all_queries, prefetch, venue_search_results)
    ranked_rows = await _rank_stage(ctx, pool, pool_by_venue)
    return MarketSnapshot(
        matches=ranked_rows,
        sources=ctx.ledger.tokens,
        pool_size=len(pool.candidates),
        # The instant the render dates its staleness disclosures against. `as_of` when a caller
        # supplied one (a backtest's simulated present, where pool assembly has already dropped
        # everything closing at or before it), otherwise now. Stamped on the snapshot rather than
        # read from the clock in the renderer so a replay of the archived payload reproduces what
        # the forecaster saw instead of re-aging every row against the replay's own clock.
        forecast_time=as_of or datetime.now(UTC),
    )


def _log_ranking_telemetry(
    qid: int | None,
    pool: generation.PoolResult,
    ranked_rows: list[MarketMatch],
    *,
    outcome: str,
    prompt_chars: int,
) -> None:
    """One INFO line per question, carrying every ranked row's `(venue, pool_index, rank)`.

    The pool index is the point: it is the post-ship instrument for the two questions this port
    deliberately left open. Whether the ranker's attention decays down a ~400-candidate prompt
    (the measured pick-index distribution says it does NOT, which is why the input stays
    venue-grouped rather than interleaved), and whether Manifold detail enrichment changes
    which rows get picked. Both answer themselves from prod logs instead of another bake-off.
    """
    rendered = ",".join(f"{row.platform}:{index}@{row.rank}" for index, row in _pool_positions(pool, ranked_rows))
    logger.info(
        f"MARKET_RANKING: question={qid} pool={len(pool.candidates)} outcome={outcome} "
        f"rows={len(ranked_rows)} prompt_chars={prompt_chars} rendered={rendered or 'none'}"
    )


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def _log_child_render_telemetry(qid: int | None, stats: rendering.ChildRenderStats) -> None:
    """One INFO line per rendered snapshot, beside `MARKET_RANKING`.

    `withheld` is why it exists: the Kalshi no-price spread threshold is calibrated on eleven fixture
    strikes, so its prod incidence has to be a query rather than a guess, and the same field counts the
    Polymarket placeholder legs and Manifold untouched priors the parsers now blank. `max_stage` and
    `ladder_chars` say whether `LADDER_SECTION_MAX_CHARS` binds on real slates. `named` + `collapsed`
    should always equal `outcomes` — that is the completeness invariant, so a line where they disagree
    is a render bug rather than a tuning signal.

    A SEPARATE marker rather than extra fields on `MARKET_RANKING`, because the harvester's
    `market_ranking` regex is not end-anchored: a new line keeps `scripts/telemetry/markers.py` purely
    additive instead of re-cutting a spec other tracks are editing this round.
    """
    logger.info(
        f"MARKET_CHILD_RENDER: question={qid} families={stats.families} full_rows={stats.full_rows} "
        f"ladder_rows={stats.ladder_rows} outcomes={stats.outcomes} named={stats.named} "
        f"collapsed={stats.collapsed} withheld={stats.withheld} max_stage={stats.max_stage} "
        f"ladder_chars={stats.ladder_chars}"
    )


def format_snapshot_for_research(snapshot: MarketSnapshot, *, qid: int | None = None) -> str:
    """The markdown block for the research bundle, or `""` when there is nothing to show.

    A thin delegate to `rendering.render_snapshot_with_stats`; the degraded-ranking marker is derived
    from the snapshot's own `ranking` source token rather than passed in, so the render is
    reproducible from an archived snapshot alone. `qid` is telemetry-only — it labels the
    `MARKET_CHILD_RENDER` line and changes nothing about the text, so a caller replaying an archived
    snapshot can omit it.

    One zero-row case renders a sentence instead of `""`: a ranker that reviewed a non-empty
    pool and deliberately kept nothing (`ranking: ok(0)` — the adaptive-width empty answer).
    Before this, that section vanished wholesale and read exactly like a provider outage, while
    the forecaster prompt still shipped the market-weighting clauses for a table that wasn't
    there (q45200: healthy 381-candidate pool, correct zero-row answer, no `## Prediction Market
    Snapshot` header at all). The gate is deliberately narrow — `ok(0)` is emitted only by a
    successful ranking call returning an empty array, and `pool_size > 0` excludes the
    nothing-to-rank case — so every failure path (`error(...)`, timeout, empty pool) still
    returns `""` and still reads as `status="empty"` downstream. Note the flip side: a
    deliberate-zero question now records `prediction_market: ok` in the provider diagnostics
    rather than `empty`, which is the honest label — the provider did contribute a judgment.

    That claim rests on `ok(0)` meaning ONE thing, which is why `parse_ranking` raises on a
    non-empty array it can read no row from (`RankingShapeRegression`, 2026-08-26). While that
    case also returned `[]`, this sentence asserted "none was judged to bear on it" over output
    whose shape we had failed to parse.
    """
    rendered, child_stats = rendering.render_snapshot_with_stats(
        snapshot, ranking_degraded=is_lost_source(snapshot.sources.get("ranking", ""))
    )
    if rendered:
        _log_child_render_telemetry(qid, child_stats)
        return rendered
    if snapshot.sources.get("ranking") == "ok(0)" and snapshot.pool_size > 0:
        return rendering.render_no_relevant_market_line(snapshot.pool_size)
    return ""


# ---------------------------------------------------------------------------
# ResearchCallable factory (plugged into _select_research_providers)
# ---------------------------------------------------------------------------


def prediction_market_provider(is_benchmarking: bool = False) -> ResearchCallable:
    """Factory returning an async research callable for prediction-market data.

    The returned callable accepts a `MetaculusQuestion` and uses its full API: `id_of_question`
    for caching, `title` / `resolution_criteria` / `fine_print` for the query author, the
    settlement-source join and the ranker prompt.

    Gated on the PREDICTION_MARKETS_ENABLED env flag; disabled returns "".

    When ``is_benchmarking=True`` the provider hard-disables regardless of the env flag. There
    is no orchestrator-level backstop, so this check IS the backtest defence: markets retain
    their last-trade price after resolution, and the ``as_of`` filter alone was never
    sufficient (a market that closes between ``as_of`` and now still leaks). See CLAUDE.md and
    the ``gemini_search_provider`` / ``native_search_provider`` precedents.
    """
    if PREDICTION_MARKET_TIMEOUT < SNAPSHOT_STAGE_BUDGET_S:
        # A stale `PREDICTION_MARKET_TIMEOUT=30` left in someone's .env would otherwise surface
        # only as a generic snapshot timeout on every question, with the real cause invisible.
        logger.warning(
            f"PREDICTION_MARKET_TIMEOUT={PREDICTION_MARKET_TIMEOUT}s is BELOW the pipeline's worst-case "
            f"stage sum of {SNAPSHOT_STAGE_BUDGET_S}s — snapshots will time out under load. Raise the env "
            f"override (or unset it to take the default)."
        )

    async def _fetch(question: MetaculusQuestion) -> str:
        if is_benchmarking:
            return ""
        if not env_flag_enabled(PREDICTION_MARKETS_ENABLED_ENV):
            return ""

        snapshot = await fetch_market_snapshot(question, as_of=None, timeout=PREDICTION_MARKET_TIMEOUT)
        # Surface per-source outcomes so the orchestrator's diagnostics line shows partial
        # degradation (a live venue while a sub-source silently died). Recorded here at the
        # ResearchCallable boundary so it's keyed to the qid the orchestrator pops; no-op when
        # qid is None.
        record_provider_detail(
            getattr(question, "id_of_question", None),
            "prediction_market",
            {"sources": snapshot.sources},
        )
        record_raw_research(
            qid=getattr(question, "id_of_question", None),
            provider="prediction_market",
            payload=snapshot,
        )
        return format_snapshot_for_research(snapshot, qid=getattr(question, "id_of_question", None))

    return _fetch
