"""PredictIt: the whole ~197-market dump, one row per market, quoting no price.

Parses ``dateEnd`` into ``close_time`` and renders contract names. The whole universe reaches the
pool, so there is no per-question query to select a contract with and pricing an arbitrary one
would misreport the market; the contract NAMES are most of a PredictIt market's semantic content,
since its own title is often just "Which party will win ...".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from metaculus_bot.research.market_retrieval.http import http_get_with_backoff, parse_iso_guarded, safe_float
from metaculus_bot.research.market_retrieval.types import MarketChild, MarketMatch

logger = logging.getLogger(__name__)

PREDICTIT_URL = "https://www.predictit.org/api/marketdata/all/"

PREDICTIT_MAX_ATTEMPTS = 2

# PredictIt bundles several binary contracts per market, and the contract names are most of its
# semantic content. Eight covers a crowded primary field without letting one market's ballot dominate
# the ranker prompt.
PREDICTIT_CONTRACTS_RENDERED = 8


async def predictit_prefetch(session: Any) -> list[dict[str, Any]] | None:
    """Fetch the full PredictIt market dump. ``None`` when the fetch itself failed.

    One unpaginated ``/marketdata/all/`` GET, ~197 markets, no auth and no query param — so
    "prefetch the universe" is one request's worth of work, and the ranked design shows all
    of it rather than fuzzy-selecting 20.

    A wrong TOP-LEVEL shape — not an object, or no ``markets`` array in it — is also ``None``.
    ``http_get_with_backoff`` has already turned every non-200 and every undecodable body into
    ``None``, so what reaches here shapeless is a 200 whose contract changed, and the caller
    CACHES a successful universe for 6h: as ``[]`` one malformed response pinned an empty
    universe as healthy for the whole TTL, and every later question in the run then read the
    venue back as fine with no HTTP and no counter bump. Individual malformed MARKETS are still
    filtered out rather than failing the pull, mirroring the two search parsers.
    """
    payload = await http_get_with_backoff(
        session,
        PREDICTIT_URL,
        {},
        max_attempts=PREDICTIT_MAX_ATTEMPTS,
        label="PredictIt prefetch",
        component="predictit",
    )
    if payload is None:
        return None
    if not isinstance(payload, dict):
        logger.warning(f"PredictIt returned a {type(payload).__name__} payload, not the documented object")
        return None

    markets = payload.get("markets")
    if not isinstance(markets, list):
        logger.warning("PredictIt payload carried no top-level 'markets' list; treating as a lost dump")
        return None
    return [market for market in markets if isinstance(market, dict)]


def predictit_contract_names(market: dict[str, Any]) -> str:
    """The contract list for one PredictIt market, comma-joined.

    PredictIt bundles one binary contract per outcome, and the contract NAMES are most of
    the market's semantic content — its own title is often just "Which party will win ...".
    So they are both the fuzzy-scoring text and the rules text the ranker reads. One helper
    for both, so the two cannot describe the same market differently.
    """
    contracts = market.get("contracts")
    if not isinstance(contracts, list):
        return ""
    names = [
        str(contract.get("name") or "")
        for contract in contracts[:PREDICTIT_CONTRACTS_RENDERED]  # HARNESS-SCAN-EXEMPT-subsampling
        if isinstance(contract, dict)
    ]
    return ", ".join(name for name in names if name)


def predictit_contract_children(contracts: Sequence[dict[str, Any]]) -> tuple[MarketChild, ...]:
    """Each contract as its own sub-row, in BALLOT ORDER, priced off its own ``lastTradePrice``.

    A PredictIt market is a ballot — "Which party will win ..." with one binary contract per outcome
    — so it has no single probability, and this is where its prices finally reach a forecaster:
    ``predictit_market_match`` has always (correctly) refused to quote one arbitrary contract as the
    market's, which until 2026-08-05 meant the whole ballot rendered priceless.

    ``lastTradePrice`` rather than ``bestBuyYesCost``: the last trade is a price somebody paid, while
    the best ask is one side of a book that on a thin PredictIt contract can sit far from it.

    Ballot order, unsorted, because PredictIt publishes no per-contract volume to rank by (its dump
    carries no volume field at all — the same absence that makes every PredictIt row read
    ``no-liquidity-data``) and its own order is the meaningful one: seat-count rungs and candidate
    lists arrive ordered, and re-sorting a ladder by price would scramble it.

    That reasoning was right and is now the rule for every venue: as of 2026-08-25 the other three
    parsers stopped sorting too, and the renderer owns presentation. The half of the old asymmetry
    that DID cost something here is gone with it — truncation used to cut the end of a ballot, which
    on this venue meant the last contracts on the paper rather than the cheap tail. Nothing is cut
    now; a family's remaining outcomes are all named in one ladder row, in this order.
    """
    return tuple(
        MarketChild(
            title=title,
            implied_prob_yes=safe_float(contract.get("lastTradePrice")),
            # A missing status is not evidence of settlement, matching the market-level derivation.
            is_resolved=(contract.get("status") or "").lower() not in ("", "open"),
            close_time=parse_iso_guarded(contract.get("dateEnd")),
        )
        for contract in contracts
        if (title := str(contract.get("name") or contract.get("shortName") or ""))
    )


def predictit_market_match(market: dict[str, Any], *, match_confidence: float, channel: str) -> MarketMatch | None:
    """One PredictIt market as a candidate row. None when it has no usable name.

    One row per market, matching Kalshi's one-row-per-event, and NO price. The whole
    ~197-market universe reaches the pool, so there is no per-question query to pick a
    contract with; relevance is the ranker's job, and quoting one arbitrary contract's price
    on a multi-contract ballot would misreport the market. Rendered close coverage therefore
    reads ~26% — the universe's own rate, not the ~64% a fuzzy pre-filter selects for.

    The prices are not lost, though: since 2026-08-05 every contract renders as its own sub-row via
    ``predictit_contract_children``, which is what makes the market-level ``None`` a statement about
    the ballot rather than a hole in the row.

    Both event-level derivations read EVERY contract rather than ``contracts[0]``, mirroring
    ``kalshi_event_match``: a market whose contracts are ``[Closed, Open]`` is live, and reading
    the first one makes the verdict depend on the order untrusted external JSON happened to
    arrive in — reverse the list and the same market flips. Both fields are load-bearing now:
    ``status`` is a rendered column AND a ranker prompt signal, telling the model a RESOLVED
    price is a realized outcome rather than a forecast. Derived from the contracts rather than
    the market-level ``status`` field (present on 197/197 live markets, ignored here) so both
    venues answer the question the same way, in one place.
    """
    name = market.get("name") or ""
    short_name = market.get("shortName") or ""
    if not name and not short_name:
        return None

    raw_contracts = market.get("contracts")
    # Filtered to dicts ONCE and gated on the filtered list. Folding the isinstance check inside
    # the `all()` instead would take `all()` over an EMPTY sequence when every contract is a
    # non-dict — reporting RESOLVED where an unusable contracts block must stay unresolved.
    contracts = [entry for entry in raw_contracts if isinstance(entry, dict)] if isinstance(raw_contracts, list) else []
    statuses = [(entry.get("status") or "").lower() for entry in contracts]
    # A missing status is not evidence of settlement, so it reads as open.
    is_resolved = bool(statuses) and all(status not in ("", "open") for status in statuses)
    closes = [parsed for entry in contracts if (parsed := parse_iso_guarded(entry.get("dateEnd"))) is not None]

    contract_names = predictit_contract_names(market)
    market_id = market.get("id")
    return MarketMatch(
        platform="predictit",
        market_title=name or short_name,
        market_url=market.get("url") or "",
        implied_prob_yes=None,
        bid=None,
        ask=None,
        spread=None,
        volume_24h=None,
        close_time=max(closes) if closes else None,
        is_resolved=is_resolved,
        match_confidence=match_confidence,
        raw_rules=f"contracts: {contract_names}" if contract_names else "",
        venue_market_id="" if market_id is None else str(market_id),
        retrieval_channel=channel,
        children=predictit_contract_children(contracts),
    )
