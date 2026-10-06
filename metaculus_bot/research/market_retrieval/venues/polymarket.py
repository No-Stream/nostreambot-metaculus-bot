"""Polymarket: Gamma's public search, parsed from either the events or the markets list.

``events_status=active`` is load-bearing rather than tidy: nothing downstream filters on close
date any more (the pipeline passes ``as_of=None``), so without the parameter ``Ethereum`` puts 7
closed events out of 10 straight into the pool.

**A multi-market EVENT quotes no price of its own.** An event is a set of mutually-exclusive
outcomes ("How many Fed rate cuts in 2026?" → 0, 1, 2, ... cuts), each a separate nested market
with its own book, so no single number answers the event's title. Reading ``markets[0]`` for the
price legs — what this module did until 2026-08-05 — rendered the FIRST child's probability under
the EVENT's title: on this package's own committed fixture, "How many Fed rate cuts in 2026?" with
prob 0.89, which is the "will no cuts happen" child. The forecaster prompts tell the model to
anchor on a matched market's price, so that row mislabelled an anchor. Every nested market now
renders as its own ``MarketChild`` sub-row instead, and the event's price legs are ``None``.

**An untouched leg quotes no price.** Gamma's ``outcomePrices`` default is ``["0.5","0.5"]``, so a
placeholder ("Candidate A", "Party B", "Other") reads 0.5 with zero trading — 155 of the archive's
1,839 ranked-era child outcomes. ``_priced_or_none`` blanks exactly that shape, everywhere this module reads a
price, since a fabricated 0.5 that sorts to the front of a render is worse than no row at all.

**ABSENT ``outcomePrices`` is Gamma's OTHER way of saying the same thing, and needs no fix**
(investigated 2026-08-26 after a QA run flagged the sparsity; closed as no-defect). Re-running this
module's own search shape, 118 of 174 nested markets carried no ``outcomePrices`` — and all 118 were
``active: false`` with ``volumeNum: 0``; across three queries, ZERO price-less legs had any volume
(0 of 119). The ``/markets/<id>`` detail endpoint serves ``outcomePrices: null`` for them too, with
``bestBid: 0 / bestAsk: 1`` — an empty book whose CLOB midpoint is the synthetic $0.50 the blanking
rule above exists to refuse. So a Polymarket detail fan-out (the Manifold-style enrichment this
venue deliberately does not have) could only manufacture that 0.50: every informative, traded leg
already ships a price, and a price-less leg collapses into the ladder's "unquoted" group with a
count, which is the honest render for an outcome nobody has priced. Not a ``fields=`` projection
artifact either — we send none.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from metaculus_bot.research.market_retrieval.http import http_get_with_backoff, parse_iso, safe_float
from metaculus_bot.research.market_retrieval.types import MarketChild, MarketMatch
from metaculus_bot.research.market_retrieval.venues._shared import RULES_TEXT_MAX_CHARS, VENUE_SEARCH_LIMIT

logger = logging.getLogger(__name__)

POLYMARKET_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"

POLYMARKET_MAX_ATTEMPTS = 2

# Gamma's own default for a leg nobody has touched: `outcomePrices` reads `["0.5","0.5"]` on a
# placeholder ("Candidate A", "Party B", "Other"), so exactly this value with no trading is the
# venue's absence of a price rather than a price. Compared exactly rather than as a band, because
# any real trade moves the leg off it — and 155 of the 1,839 archived ranked-era child outcomes sit here.
POLYMARKET_UNTOUCHED_PRICE = 0.5


def _prob_from_prices(prices: Any) -> float | None:
    if isinstance(prices, str):
        try:
            parsed = json.loads(prices)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(parsed, list) and parsed:
            try:
                return float(parsed[0])
            except (TypeError, ValueError):
                return None
    if isinstance(prices, list) and prices:
        try:
            return float(prices[0])
        except (TypeError, ValueError):
            return None
    return None


def _polymarket_market_volume(market: dict[str, Any]) -> float | None:
    """One nested market's all-time volume: ``volumeNum``, else the string-typed ``volume``."""
    total_volume = safe_float(market.get("volumeNum"))
    return total_volume if total_volume is not None else safe_float(market.get("volume"))


def _priced_or_none(price: float | None, *, total_volume: float | None, open_interest: float | None) -> float | None:
    """Gamma's ``outcomePrices`` default is ``["0.5","0.5"]``, so an untouched leg quotes exactly 0.5.

    Returned as ``None`` — the leg has no price, and rendering 0.5 told a forecaster the crowd was
    split 50/50 on an outcome nobody had ever traded. It was not a cosmetic defect: q45189's own
    snapshot spent four of its FL-22 primary rows on ``Candidate A 0.50, B 0.50, C 0.50, D 0.50``
    while the four real candidates (Carbonara 0.42, Burck 0.29, Keiser 0.15, Askar 0.12) went
    unrendered, and the archive holds 155 such children against 3 traded exact-0.5 legs.

    Exactly-0.5 WITH trading is a real 50/50 and keeps its price, which is what the volume and
    open-interest arguments are for. Both are consulted rather than volume alone because a Gamma leg
    can omit ``volumeNum``/``volume`` entirely (59 archived children) while still carrying open
    interest.
    """
    if price is None or price != POLYMARKET_UNTOUCHED_PRICE:
        return price
    if (total_volume or 0.0) > 0.0 or (open_interest or 0.0) > 0.0:
        return price
    return None


def polymarket_event_children(markets: Sequence[dict[str, Any]]) -> tuple[MarketChild, ...]:
    """One child per nested market, in GAMMA'S ARRAY ORDER, each priced off its OWN ``outcomePrices``.

    The title is ``groupItemTitle`` ("0 (0 bps)", "1 (25 bps)") in preference to ``question``
    ("Will no Fed rate cuts happen in 2026?"): it is the label Polymarket's own event page shows
    against each outcome, it is a third the length, and under the parent's title it reads as the
    ladder rung it is. ``question`` is the fallback for the events that ship no group label.

    **Unsorted, and that is the contract.** Presentation belongs to the renderer, which sorts a copy
    by ``child_render_order_key`` for the full sub-rows and reads this order for the ladder row that
    names every remaining outcome. Sorting here was a survival decision taken where it could not see
    the render budget, and Gamma's array order is what an event's outcome ladder means something in.

    A leg at Gamma's untouched ``0.5`` default reports no price (``_priced_or_none``) and says so via
    ``price_withheld``. Each child's ``bestBid`` and ``bestAsk`` travel with its probability when
    Gamma publishes both. A market with no usable title is dropped rather than rendered as a blank
    row; it would spend a child slot saying nothing.
    """
    children: list[MarketChild] = []
    for market in markets:
        title = market.get("groupItemTitle") or market.get("question") or market.get("title") or ""
        if not title:
            continue
        total_volume = _polymarket_market_volume(market)
        open_interest = safe_float(market.get("openInterest"))
        raw_price = _prob_from_prices(market.get("outcomePrices"))
        price = _priced_or_none(raw_price, total_volume=total_volume, open_interest=open_interest)
        children.append(
            MarketChild(
                title=str(title),
                implied_prob_yes=price,
                total_volume=total_volume,
                open_interest=open_interest,
                is_resolved=bool(market.get("closed")) or bool(market.get("resolved")),
                close_time=parse_iso(market.get("endDate") or market.get("end_date_iso") or ""),
                quote_low=safe_float(market.get("bestBid")),
                quote_high=safe_float(market.get("bestAsk")),
                price_withheld=price is None and raw_price is not None,
            )
        )
    return tuple(children)


@dataclass(frozen=True, slots=True)
class _EventLegs:
    """The price, money and children columns one Gamma event contributes.

    One return value for the three mutually-exclusive shapes an event can have (a family, a
    single market, no markets at all), so ``_polymarket_event_match`` reads the shape once and
    cannot end up with half of one shape's legs and half of another's.
    """

    implied: float | None = None
    bid: float | None = None
    ask: float | None = None
    vol_24h: float | None = None
    total_volume: float | None = None
    liquidity: float | None = None
    open_interest: float | None = None
    price_withheld: bool = False
    children: tuple[MarketChild, ...] = ()


def _polymarket_single_market_legs(
    market: dict[str, Any], *, event_volume: float | None, event_open_interest: float | None
) -> _EventLegs:
    """The legs of a ONE-market event, read off that market with the event as the fallback.

    Event and market ask the same question here, so the price legs are the market's own — the
    shape that predates the 2026-08-05 family split and renders exactly as it always has.
    """
    bid = safe_float(market.get("bestBid"))
    ask = safe_float(market.get("bestAsk"))
    # volumeNum is Gamma's all-time total; fall back to the event-level or market-level
    # `volume` when it is absent.
    total_volume = safe_float(market.get("volumeNum"))
    if total_volume is None:
        total_volume = event_volume if event_volume is not None else safe_float(market.get("volume"))
    liquidity = safe_float(market.get("liquidityNum"))
    if liquidity is None:
        liquidity = safe_float(market.get("liquidity"))
    # Market-level first (more specific when present), event-level as the fallback that
    # actually populates on today's payloads.
    open_interest = safe_float(market.get("openInterest"))
    if open_interest is None:
        open_interest = event_open_interest
    # AFTER the money legs, because the untouched-default guard reads them — and it reads the
    # legs as finally resolved (event-level fallbacks included) rather than the market's own
    # fields, so a placeholder leg inside a genuinely traded event keeps its price.
    raw_price = _prob_from_prices(market.get("outcomePrices"))
    implied = _priced_or_none(raw_price, total_volume=total_volume, open_interest=open_interest)
    return _EventLegs(
        implied=implied,
        bid=bid,
        ask=ask,
        vol_24h=safe_float(market.get("volume24hr")),
        total_volume=total_volume,
        liquidity=liquidity,
        open_interest=open_interest,
        price_withheld=implied is None and raw_price is not None,
    )


def _polymarket_event_legs(event: dict[str, Any]) -> _EventLegs:
    """Which of the three event shapes this is, and the legs that shape publishes.

    A MULTI-market event quotes no price of its own — see the module docstring for the mislabelled
    anchor that came of reading ``markets[0]`` — and its money columns come from the EVENT rather
    than from a sum over the children. That is not a stylistic preference: Gamma's public-search
    response TRUNCATES the nested markets list, so on the committed fixture the event reports
    $46.2M of volume against $9.1M across the two children it shipped. Summing the visible subset
    would understate the family by 5x.
    """
    volume = safe_float(event.get("volume"))
    # Gamma carries `openInterest` on the EVENT, not the nested market (5/5 live events vs
    # 0/42 nested, verified 2026-08-03) — reading it only at the market level left the
    # rendered OI column blank on every Polymarket row ever archived.
    event_open_interest = safe_float(event.get("openInterest"))
    markets = [market for market in (event.get("markets") or []) if isinstance(market, dict)]

    if len(markets) > 1:
        return _EventLegs(
            vol_24h=safe_float(event.get("volume24hr")),
            total_volume=volume,
            liquidity=safe_float(event.get("liquidity")),
            open_interest=event_open_interest,
            children=polymarket_event_children(markets),
        )
    if markets:
        return _polymarket_single_market_legs(markets[0], event_volume=volume, event_open_interest=event_open_interest)
    return _EventLegs(total_volume=volume, open_interest=event_open_interest)


def _polymarket_event_match(event: dict[str, Any], rank: int) -> MarketMatch:
    """One Gamma event as a candidate row: a single-outcome event's own price, or its children's.

    Which legs the row carries is ``_polymarket_event_legs``' decision; everything here is the
    event-level fields every shape shares.
    """
    title = event.get("title") or event.get("question") or ""
    slug = str(event.get("slug") or "")
    legs = _polymarket_event_legs(event)
    bid, ask = legs.bid, legs.ask

    return MarketMatch(
        platform="polymarket",
        market_title=title,
        market_url=f"https://polymarket.com/event/{slug}" if slug else "",
        implied_prob_yes=legs.implied,
        bid=bid,
        ask=ask,
        spread=(ask - bid) if (bid is not None and ask is not None) else None,
        # No lifetime-volume substitute when `volume24hr` is absent. It was silently
        # carrying all-time volume on 25 of 122 archived rows — a recency field holding a
        # number with no recency, which is only harmless while nothing reads it. None says
        # "the venue didn't publish a 24h figure", which is what happened.
        volume_24h=legs.vol_24h,
        close_time=parse_iso(event.get("endDate") or event.get("end_date_iso") or ""),
        is_resolved=bool(event.get("closed")) or bool(event.get("resolved")),
        match_confidence=100.0 - rank,
        raw_rules=(event.get("description") or "")[:RULES_TEXT_MAX_CHARS],
        total_volume=legs.total_volume,
        liquidity=legs.liquidity,
        open_interest=legs.open_interest,
        venue_market_id=slug,
        retrieval_channel="venue_search",
        children=legs.children,
        price_withheld=legs.price_withheld,
    )


def _polymarket_market_match(market: dict[str, Any], rank: int) -> MarketMatch:
    slug = str(market.get("slug") or "")
    total_volume = _polymarket_market_volume(market)
    liquidity = safe_float(market.get("liquidityNum"))
    if liquidity is None:
        liquidity = safe_float(market.get("liquidity"))
    open_interest = safe_float(market.get("openInterest"))
    # The top-level-markets fallback branch of the search response reads the same `outcomePrices`
    # field as the event branch, so it takes the same untouched-default guard — a placeholder row
    # reaching the pool through this path is the same fabrication by another route.
    raw_price = _prob_from_prices(market.get("outcomePrices"))
    implied = _priced_or_none(raw_price, total_volume=total_volume, open_interest=open_interest)
    return MarketMatch(
        platform="polymarket",
        market_title=market.get("question") or market.get("title") or "",
        market_url=f"https://polymarket.com/market/{slug}" if slug else "",
        implied_prob_yes=implied,
        bid=safe_float(market.get("bestBid")),
        ask=safe_float(market.get("bestAsk")),
        spread=None,
        volume_24h=safe_float(market.get("volume24hr")),
        close_time=parse_iso(market.get("endDate") or ""),
        is_resolved=bool(market.get("closed")),
        match_confidence=100.0 - rank,
        raw_rules=(market.get("description") or "")[:RULES_TEXT_MAX_CHARS],
        total_volume=total_volume,
        liquidity=liquidity,
        open_interest=open_interest,
        venue_market_id=slug,
        retrieval_channel="venue_search",
        price_withheld=implied is None and raw_price is not None,
    )


def parse_polymarket_matches(payload: Any, *, width: int) -> list[MarketMatch] | None:
    """Parse a Gamma public-search response into candidate rows, venue-rank order.

    Gamma returns ``{events: [...], markets: [...]}``; events are preferred and the
    top-level markets list is the fallback for when there were none. BOTH branches read
    ``openInterest``, because both are live paths and the field sits in a different place in
    each.

    ``match_confidence`` is the venue's OWN rank inverted, never a local re-score: the point
    of the venue-index channel is to stop trying to out-rank the index.

    ``None`` when the TOP-LEVEL shape is not the documented dict, honouring the package's
    ``None``-means-fetch-failed contract. ``http_get_with_backoff`` has already turned every
    non-200 and every undecodable body into ``None`` before this runs, so the only payload that
    reaches here shapeless is a 200 whose contract changed — a loss, not a search that found
    nothing. Returning ``[]`` for it counted as a clean query in ``flatten_results`` and
    published as a benign ``none``. Per-ROW malformation is different and still skips the row:
    one unreadable event among ten is not a venue-wide failure.
    """
    if not isinstance(payload, dict):
        logger.warning(f"Polymarket returned a {type(payload).__name__} payload, not the documented object")
        return None

    out: list[MarketMatch] = []
    for rank, event in enumerate(payload.get("events") or []):
        if len(out) >= width:
            break
        if isinstance(event, dict):
            out.append(_polymarket_event_match(event, rank))
    if out:
        return out

    for rank, market in enumerate(payload.get("markets") or []):
        if len(out) >= width:
            break
        if isinstance(market, dict):
            out.append(_polymarket_market_match(market, rank))
    return out


async def polymarket_search(session: Any, query: str, *, width: int) -> list[MarketMatch] | None:
    """Search Polymarket for one query. ``None`` when the fetch failed OR the shape changed.

    ``events_status=active`` is load-bearing, not tidiness: nothing downstream filters on
    close date any more, so without it ``Ethereum`` puts 7 closed events out of 10 into the
    pool.

    The None-vs-``[]`` split is the whole degradation contract: a retry-exhausted 503 would
    otherwise arrive at the caller as an ordinary empty result and publish as a benign
    ``none``. A 200 whose top-level shape is not Gamma's object is the same class of loss, and
    the parser reports it the same way.
    """
    payload = await http_get_with_backoff(
        session,
        POLYMARKET_SEARCH_URL,
        {"q": query, "limit_per_type": str(VENUE_SEARCH_LIMIT), "events_status": "active"},
        max_attempts=POLYMARKET_MAX_ATTEMPTS,
        label=f"Polymarket q={query[:40]!r}",  # HARNESS-SCAN-EXEMPT-subsampling  # log-label truncation
        component="polymarket",
    )
    if payload is None:
        return None
    return parse_polymarket_matches(payload, width=width)
