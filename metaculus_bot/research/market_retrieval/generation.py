"""Pool assembly: three retrieval channels unioned into one ranked candidate list.

**Channel order IS the ranking.** ``add()`` keeps first-seen position, so a structural
settlement-source hit always precedes a venue-index hit, which always precedes a fuzzy fill,
with no cross-channel score comparison anywhere — which matters because the scores are not
comparable (1.0 for a structural hit, an inverted venue rank, a 0-100 fuzzy score). That
ordering is used for two things and they must stay the same one: the order candidates are
PRESENTED to the ranker in, and the deterministic fail-open slate. Presenting in the
fail-open order is what makes a fail-open a truncation of what the model was shown rather
than a different pipeline.

WITHIN a channel every venue's rows carry that channel's own ordering signal, so each is sorted
by it before the width can cut: ``fuzzy_best`` for the settlement join, the venue's inverted
rank for venue search, ``fuzzy_best_many`` for the Kalshi universe, dump order for PredictIt
(which has no width). No channel is compared against another.

The design is recall-maximal on purpose. The bake-off measured that **selection, not
generation, is the binding constraint**: a perfect ranker over this pool reaches 14/16
questions while the deterministic top-4 of the same pool scores 5/16. So nothing here filters on
RELEVANCE — there is no score floor, and the retired ``KALSHI_MIN_FUZZY_SCORE`` /
``PREDICTIT_MIN_FUZZY_SCORE`` floors are absent rather than defaulted, because a floor is
what discarded the adjacent-cut markets that carry most of the evidential value. The one
exception is ELIGIBILITY: an explicit ``as_of`` drops a candidate that already closed, which is
leakage defence rather than a relevance judgment, and it happens inside ``add()`` so the width
still means "N eligible candidates" (as a post-hoc filter it deleted rows the width had already
spent its slots on).

Everything CPU-bound runs in ONE ``asyncio.to_thread`` hop, and that is not hygiene:
``publish_hardening.py`` documents how a pinned event loop misattributes forecaster
soft-deadline drops to the forecasters. Building the settlement-domain index belongs in the
same hop — it walks the current ~15.2k-event catalogue and is not memoized.

The hop only works because the scan itself releases the GIL. As a per-event Python loop the
full-catalogue fuzzy scan measured ~0.45-0.59s of GIL-HOLDING bytecode (9,762 events x ~17
queries), so the offload bought nothing — it converted one freeze into sustained starvation,
with loop lag p50 at 15-20ms single-question and 56ms at 6 concurrent. Batched through
``fuzzy_best_many`` (``rapidfuzz.process.cdist``, which threads internally) the same work is
~0.06s and the loop returns to idle. The settlement index walk and the catalogue sort are still
real CPU, which is why the hop stays; row CONSTRUCTION is not, since the universe channel yields
lazily and only the rows inside the width are ever built.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from metaculus_bot.research.market_retrieval import venues
from metaculus_bot.research.market_retrieval.http import flatten_results
from metaculus_bot.research.market_retrieval.queries import fuzzy_best, fuzzy_best_many
from metaculus_bot.research.market_retrieval.ranking import RULES_CHARS
from metaculus_bot.research.market_retrieval.settlement_join import question_domains, settlement_domain_index
from metaculus_bot.research.market_retrieval.types import MarketMatch, _FetchTally
from metaculus_bot.time_utils import _as_utc

logger = logging.getLogger(__name__)

# Venue presentation order, and therefore the pool's flat order. Kalshi's block comes first
# because it is the widest; the ranker prompt states explicitly that the blocks are ordered
# by venue and NOT by value, and the measured pick-index distribution bears that out (pick
# rate RISES with index across kalshi -> polymarket -> manifold).
VENUE_ORDER: tuple[str, ...] = ("kalshi", "polymarket", "manifold", "predictit")

# Per-venue pool width. **PredictIt is deliberately absent**: its whole universe is 197
# markets from one GET, so there is nothing to select and a width would only throw evidence
# away — the same "cannot come back by way of a defaulted argument" logic as the deleted
# score floors. A venue missing from this dict is unbounded, which is the statement.
#
# Kalshi's 100 is what fits in a prompt out of ~15,200 open events (2026-10-17), so something has to
# order them; that is all `fuzzy_best` is for here. The two venue-search widths are ceilings on
# the union across queries (each query's own response is capped at the endpoint's `limit`).
RETRIEVAL_WIDTH: dict[str, int] = {
    "kalshi": 100,
    "polymarket": 60,
    "manifold": 60,
}

# Manifold detail-enrichment dials. Bounded at <= 60 fetches by construction (the Manifold
# pool width), 10 at a time, and the WHOLE fan-out sits under one wall — without it the worst
# case is 6 waves x 10s = 60s, which blows the snapshot's time budget on rules text.
MANIFOLD_DETAIL_CONCURRENCY = 10
MANIFOLD_DETAIL_WALL_S = 10.0
# Derived from the ranker's own Manifold cap rather than restated: this is where the text is
# STORED, so a store cap below the prompt cap silently caps the prompt too — declaring both as
# 300 made raising only the ranker's a no-op. The render path has its own, separate, smaller cap.
MANIFOLD_DETAIL_RULES_CHARS = RULES_CHARS["manifold"]

CHANNEL_SETTLEMENT_JOIN = "settlement_join"
CHANNEL_VENUE_SEARCH = "venue_search"
CHANNEL_UNIVERSE_FUZZY = "universe_fuzzy"


@dataclass(frozen=True, slots=True)
class PoolResult:
    """The assembled pool plus everything the caller needs to report on it.

    ``candidates`` is flat and venue-major, so a row's list index is its ranker prompt index
    and the head of the list is the fail-open slate.

    ``per_venue_counts`` is what provider-health records as ``candidates_pre_filter``: the
    per-venue POOL count, not the rendered count. Field-presence has to be measured over the
    pool rather than over the ranked rows, or a legitimate 6-row render that happens to
    exclude Polymarket records a 100%-dead ``open_interest`` field and reddens CI on the
    ranker's judgment.

    ``per_venue_tally`` is the one degradation signal: a venue is degraded iff its tally has a
    failed sub-fetch, and the caller turns that into the source token. There is deliberately no
    separate ``degraded_venues`` list — two spellings of the same predicate is how they drift.

    ``channel_counts`` and ``settlement_domains`` are TELEMETRY, read by the ``market pool:`` log
    line and the tests, never by a decision. ``channel_counts`` counts what entered the pool
    pre-width, so on a venue whose search overfills its width it exceeds the candidate count by
    design; the Kalshi universe channel is the exception, since it is consumed only to the width.
    """

    candidates: tuple[MarketMatch, ...] = ()
    per_venue_counts: dict[str, int] = field(default_factory=dict)
    per_venue_tally: dict[str, _FetchTally] = field(default_factory=dict)
    channel_counts: dict[str, int] = field(default_factory=dict)
    settlement_domains: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EnrichmentResult:
    """How the Manifold detail fan-out went.

    ``n_ok`` counts detail GETs that came BACK, not rows whose rules text changed: a market
    with a genuinely empty description is a success. The caller reports a lost source only at
    ``n_ok == 0 and n_attempted > 0`` — a partially-lost fan-out costs no recall, only rules
    text, so bumping the degradation counter on it would alert on nothing actionable.
    """

    n_attempted: int = 0
    n_ok: int = 0


def _dedup_key(match: MarketMatch) -> tuple[str, str]:
    """``(venue, casefolded venue-native id)``, falling back to the title.

    The venue-native id is the right key — it is stable across channels, where
    ``market_url`` is not (Kalshi's join channel knows the event ticker; its URL is
    series-level and shared by several events). The title fallback only matters for a row a
    venue shipped without an id, and keying those on the title at least keeps two distinct
    ones distinct instead of collapsing every id-less row into a single slot.
    """
    return (match.platform, (match.venue_market_id or match.market_title).strip().casefold())


def _settlement_join_channel(
    criteria_text: str, queries: Sequence[str], kalshi_events: Sequence[dict[str, Any]]
) -> tuple[list[MarketMatch], tuple[str, ...]]:
    """Kalshi events settling on a publisher the question's own resolution text names.

    Reaches the markets whose titles share almost no vocabulary with the question — the
    class a fuzzy scorer structurally cannot see. Re-ranked WITHIN the channel by
    ``fuzzy_best``, because the join itself has no ranking signal and returns catalogue
    order: leaving it unranked would make the fail-open slate measure the alphabet.
    """
    domains = question_domains(criteria_text)
    if not domains or not kalshi_events:
        return [], tuple(sorted(domains))

    index = settlement_domain_index(kalshi_events)
    joined: dict[str, dict[str, Any]] = {}
    for domain in sorted(domains):
        for event in index.get(domain, ()):
            joined.setdefault(str(event.get("event_ticker") or ""), event)

    scored: list[tuple[float, MarketMatch]] = []
    for event in joined.values():
        match = venues.kalshi_event_match(event, match_confidence=1.0, channel=CHANNEL_SETTLEMENT_JOIN)
        if match is None:
            continue
        scored.append((fuzzy_best(list(queries), match.market_title, match.raw_rules), match))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [match for _, match in scored], tuple(sorted(domains))


def _kalshi_universe_channel(queries: Sequence[str], kalshi_events: Sequence[dict[str, Any]]) -> Iterator[MarketMatch]:
    """The full Kalshi catalogue in ``fuzzy_best_many`` order, yielded LAZILY.

    Scoring is whole-catalogue (that is what makes the ordering global), but construction is
    not: building a ``MarketMatch`` for all ~9,762 events spent it on the ~9,662 rows the width
    then discarded. Yielding lets the caller stop at the width instead, which over a synthetic
    9,762-event catalogue measured 0.162s -> 0.028s and 6.7 MB -> 1.3 MB peak for the same
    100-row pool, with the yielded order verified identical to the eager list's.

    The generator carries no cap of its own, deliberately — the join channel's rows dedup
    against these BEFORE the width slice, so a cap here could leave the pool under-filled.
    Truncation stays where it was, in ``assemble_pool``.
    """
    usable: list[dict[str, Any]] = []
    titles: list[str] = []
    rules: list[str] = []
    for event in kalshi_events:
        if not isinstance(event, dict):
            continue
        title = event.get("title") or event.get("sub_title") or ""
        if not title:
            continue
        usable.append(event)
        titles.append(title)
        rules.append(venues.kalshi_event_rules(event))

    scores = fuzzy_best_many(list(queries), titles, rules)
    # Stable, so equal scores keep catalogue order — the same tie-break the per-event loop had.
    # Keyed on the score alone so the sort never compares two event dicts.
    for score, event in sorted(zip(scores, usable, strict=True), key=lambda pair: pair[0], reverse=True):
        match = venues.kalshi_event_match(event, match_confidence=score, channel=CHANNEL_UNIVERSE_FUZZY)
        if match is not None:
            yield match


def _venue_search_channel(
    venue_search_results: Mapping[str, Sequence[list[MarketMatch] | BaseException | None]],
) -> tuple[list[MarketMatch], dict[str, _FetchTally]]:
    """Every venue's search hits in ``VENUE_ORDER``, plus the per-venue lost-sub-query tally.

    A venue absent from the mapping ran no search at all and gets no tally; a venue whose entry
    holds a ``None`` or a raised task ran one and lost sub-queries, which is the distinction
    ``flatten_results`` keeps and the only degradation signal the pool carries.
    """
    search_rows: list[MarketMatch] = []
    tallies: dict[str, _FetchTally] = {}
    for venue in VENUE_ORDER:
        results = venue_search_results.get(venue)
        if results is None:
            continue
        matches, tally = flatten_results(list(results), venue)
        # Ordered by the venue's OWN inverted rank before anything can cut it. `flatten_results`
        # concatenates the per-query lists in QUERY order, so an early query's rank-9 row sat ahead
        # of a later query's rank-0 exact hit and took its width slot — a truncation bias keyed on
        # which query happened to be issued first. Stable, so equal ranks keep query precedence:
        # the deterministic query set is ordered precision-descending and that tiebreak is worth
        # keeping. Only this channel is re-ordered; the settlement join has already re-ranked
        # itself with `fuzzy_best`, and re-ordering a whole venue POOL would break channel order.
        matches.sort(key=lambda match: match.match_confidence, reverse=True)
        search_rows.extend(matches)
        tallies[venue] = tally
    return search_rows, tallies


def _predictit_universe_channel(predictit_markets: Sequence[dict[str, Any]]) -> list[MarketMatch]:
    """Every PredictIt market, in dump order. No scorer and no width, deliberately.

    197 markets is a prompt block, not a corpus, so there is nothing to rank and nothing to
    cut. A consequence worth knowing: provider-health's no-contribution signal can never
    fire for PredictIt (``candidates_pre_filter`` is always 197), which leaves
    ``record_catalogue_size`` as its only health signal.
    """
    out: list[MarketMatch] = []
    for market in predictit_markets:
        if not isinstance(market, dict):
            continue
        match = venues.predictit_market_match(market, match_confidence=1.0, channel=CHANNEL_UNIVERSE_FUZZY)
        if match is not None:
            out.append(match)
    return out


def _fill_pools(
    *,
    join_rows: Sequence[MarketMatch],
    search_rows: Sequence[MarketMatch],
    kalshi_universe: Iterator[MarketMatch],
    predictit_rows: Sequence[MarketMatch],
    as_of_utc: datetime | None,
) -> tuple[dict[str, list[MarketMatch]], dict[str, int]]:
    """Every channel's rows into per-venue pools, first-seen-wins. Returns ``(pools, channel_counts)``.

    **The call order below IS the ranking** — see the module docstring. ``pools`` is pre-width, so
    ``channel_counts`` counts what entered the pool rather than what survived the cut.
    """
    pools: dict[str, list[MarketMatch]] = {venue: [] for venue in VENUE_ORDER}
    seen: set[tuple[str, str]] = set()
    channel_counts: dict[str, int] = {}

    def add(match: MarketMatch) -> None:
        if match.platform not in pools:
            logger.warning(f"pool assembly: dropping row from unknown venue {match.platform!r}")
            return
        # Ahead of every write below, so an ineligible row consumes no width, no dedup slot and no
        # channel count. `parse_iso` returns aware datetimes for all four venues, and `as_of_utc`
        # went through `_as_utc` in `assemble_pool`, so the comparison cannot raise on a mixed pair.
        if as_of_utc is not None and match.close_time is not None and match.close_time <= as_of_utc:
            return
        key = _dedup_key(match)
        if key in seen:
            return
        seen.add(key)
        pools[match.platform].append(match)
        channel_counts[match.retrieval_channel] = channel_counts.get(match.retrieval_channel, 0) + 1

    for match in join_rows:
        add(match)
    for match in search_rows:
        add(match)
    # The Kalshi universe generator is consumed only as far as the width; see the channel's own
    # docstring for the measurement. The stop condition reads the POOL count rather than the rows
    # consumed, so a dedup hit or an `as_of` drop still frees its slot — and a venue absent from
    # `RETRIEVAL_WIDTH` stays unbounded here too.
    kalshi_width = RETRIEVAL_WIDTH.get("kalshi")
    for match in kalshi_universe:
        if kalshi_width is not None and len(pools["kalshi"]) >= kalshi_width:
            break
        add(match)
    for match in predictit_rows:
        add(match)
    return pools, channel_counts


def _apply_widths(pools: Mapping[str, list[MarketMatch]]) -> tuple[list[MarketMatch], dict[str, int]]:
    """The pool cut to ``RETRIEVAL_WIDTH``, flat and venue-major, plus the counts kept per venue.

    A venue absent from ``RETRIEVAL_WIDTH`` is unbounded, which is PredictIt's whole-universe
    statement rather than an oversight.
    """
    candidates: list[MarketMatch] = []
    per_venue_counts: dict[str, int] = {}
    for venue in VENUE_ORDER:
        width = RETRIEVAL_WIDTH.get(venue)
        rows = pools[venue] if width is None else pools[venue][:width]
        per_venue_counts[venue] = len(rows)
        candidates.extend(rows)
    return candidates, per_venue_counts


def assemble_pool(
    *,
    criteria_text: str,
    queries: Sequence[str],
    kalshi_events: Sequence[dict[str, Any]],
    predictit_markets: Sequence[dict[str, Any]],
    venue_search_results: Mapping[str, Sequence[list[MarketMatch] | BaseException | None]],
    as_of: datetime | None = None,
) -> PoolResult:
    """The synchronous, I/O-free half of pool assembly. Call it via ``build_pool``.

    Named and public rather than a closure inside ``build_pool`` so the channel ordering, the
    dedup and the widths are testable without an event loop — and so there is exactly one
    thing to hand ``asyncio.to_thread``.

    ``criteria_text`` is the question's ``resolution_criteria`` + ``fine_print``, concatenated
    by the caller: the settlement join extracts URLs from it, and the fine print is where a
    question often names the actual release page.

    ``kalshi_events`` is the PROJECTED catalogue (``venues.kalshi_prefetch_events``), not raw
    ``/events`` payloads. Pass ``[]`` for a lost pull — the caller owns that distinction, since
    it is the one holding the ``CataloguePull`` token.

    ``venue_search_results`` is per-venue, per-query, in the leaf fetchers' ``None``-vs-``[]``
    contract: ``None`` is an upstream failure, ``[]`` a search that parsed to nothing. That
    distinction is what separates a degraded venue from a venue with nothing to say, and it
    cannot be recovered from the pool afterwards. A raised query may be passed through as the
    exception itself (what ``asyncio.gather(return_exceptions=True)`` hands back); it counts as
    one lost sub-query, same as ``None``.

    ``as_of`` (backtest leakage defence; None on the provider path) drops a candidate that closed
    at or before that instant. It is tested INSIDE ``add()``, ahead of the width slice, so
    ``RETRIEVAL_WIDTH`` means "100 ELIGIBLE candidates" on both paths and a dropped row frees its
    slot. As a post-hoc filter over the already-truncated pool this lost eligible evidence
    outright: measured, 150 Kalshi events whose 120 highest-scoring rows close before ``as_of``
    truncated to 100, then filtered to ZERO, while 30 eligible candidates sat unused in the
    catalogue. Filtering here also keeps ``per_venue_counts`` and ``channel_counts`` describing
    the pool that actually exists — the post-hoc filter zeroed the former, which fed
    ``candidates_pre_filter=0`` into provider health, exactly the shape that field exists to
    prevent alerting on.
    """
    as_of_utc = None if as_of is None else _as_utc(as_of)
    join_rows, domains = _settlement_join_channel(criteria_text, queries, kalshi_events)
    search_rows, tallies = _venue_search_channel(venue_search_results)
    pools, channel_counts = _fill_pools(
        join_rows=join_rows,
        search_rows=search_rows,
        kalshi_universe=_kalshi_universe_channel(queries, kalshi_events),
        predictit_rows=_predictit_universe_channel(predictit_markets),
        as_of_utc=as_of_utc,
    )
    candidates, per_venue_counts = _apply_widths(pools)

    degraded = [venue for venue, tally in tallies.items() if tally.failed]
    logger.info(
        f"market pool: n={len(candidates)} per_venue={per_venue_counts} channels={channel_counts} "
        f"domains={list(domains)} degraded={degraded or 'none'}"
    )
    return PoolResult(
        candidates=tuple(candidates),
        per_venue_counts=per_venue_counts,
        per_venue_tally=tallies,
        channel_counts=channel_counts,
        settlement_domains=domains,
    )


async def build_pool(
    *,
    criteria_text: str,
    queries: Sequence[str],
    kalshi_events: Sequence[dict[str, Any]],
    predictit_markets: Sequence[dict[str, Any]],
    venue_search_results: Mapping[str, Sequence[list[MarketMatch] | BaseException | None]],
    as_of: datetime | None = None,
) -> PoolResult:
    """``assemble_pool`` off the event loop. One hop, so the loop is yielded once, not N times.

    See the module docstring for why the offload is load-bearing rather than tidy: a pinned loop
    shows up as forecaster soft-deadline drops somewhere else entirely. Post-batching the fuzzy
    scan is ~0.06s (it was ~0.45s of GIL-holding bytecode, which the hop could not hide), and the
    settlement index walk plus the catalogue sort are what is left to offload.
    """
    return await asyncio.to_thread(
        assemble_pool,
        criteria_text=criteria_text,
        queries=queries,
        kalshi_events=kalshi_events,
        predictit_markets=predictit_markets,
        venue_search_results=venue_search_results,
        as_of=as_of,
    )


async def enrich_manifold(
    candidates: Sequence[MarketMatch],
    session: Any,
    *,
    concurrency: int = MANIFOLD_DETAIL_CONCURRENCY,
    wall_s: float = MANIFOLD_DETAIL_WALL_S,
) -> EnrichmentResult:
    """Fill in Manifold rules text and multi-outcome answers from each candidate's detail record.

    In place, on the pool rows. The Manifold search listing carries neither field, so without
    this every Manifold candidate reaches the ranker title-only — and the ranker's stated "single
    most reliable cue" is the settlement/rules text. Runs post-dedup, on the pool, which bounds
    it at the Manifold width by construction; running it inside the search would scale with the
    query count instead.

    The answers matter more than the rules text now that the search sends
    ``contractType=ALL``: a multi-outcome row's ``probability`` is null, so this fan-out is the
    ONLY place its price can come from. Both fields ride the same GET, so lifting the
    ``BINARY`` recall ceiling cost no extra request.

    Soft-fails at every level. A lost detail GET leaves that row title-only — no rules text, no
    answers, a ``-`` in the rendered ``prob`` column — and the whole fan-out sits under
    ``wall_s`` returning whatever completed, because both fields are worth a bounded wait and
    neither is worth the snapshot.
    """
    rows = [row for row in candidates if row.platform == "manifold" and row.venue_market_id]
    if not rows:
        return EnrichmentResult()

    semaphore = asyncio.Semaphore(concurrency)
    n_ok = 0

    async def _enrich_one(row: MarketMatch) -> None:
        nonlocal n_ok
        async with semaphore:
            detail = await venues.manifold_market_detail(session, row.venue_market_id)
        if detail is None:
            return
        n_ok += 1
        text = str(detail.get("textDescription") or "").strip()
        # A description that merely restates the beginning of the title is not rules text; it
        # would spend ranker tokens repeating the same line's own title back at it.
        if text and not row.market_title.strip().startswith(text):
            row.raw_rules = text[:MANIFOLD_DETAIL_RULES_CHARS]
        # Unconditional: a BINARY detail carries no answers array, so these write the `()` the
        # fields already hold rather than needing a branch to say so. Two reads of one array,
        # because the two consumers want different slices of it: `top_answers` is the ranker's
        # one-line segment (three leaders, no volume) and `children` is the render's sub-rows
        # (every answer, with its own volume).
        row.top_answers = venues.manifold_top_answers(detail)
        row.children = venues.manifold_answer_children(detail)

    gathered = asyncio.gather(*(_enrich_one(row) for row in rows), return_exceptions=True)
    try:
        for outcome in await asyncio.wait_for(gathered, wall_s):
            if isinstance(outcome, BaseException):
                logger.warning(f"Manifold detail enrichment raised: {type(outcome).__name__}: {outcome}")
    except TimeoutError:
        logger.warning(
            f"Manifold detail enrichment hit its {wall_s}s wall with {n_ok}/{len(rows)} returned; "
            f"keeping what completed"
        )
    logger.info(f"manifold detail enrichment: attempted={len(rows)} ok={n_ok}")
    return EnrichmentResult(n_attempted=len(rows), n_ok=n_ok)
