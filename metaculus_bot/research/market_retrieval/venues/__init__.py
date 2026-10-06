"""The four venue fetch/parse paths: raw venue JSON in, ``MarketMatch`` rows out.

One module per venue, with no cross-venue calls anywhere — each is its own endpoint, its own
payload shape and its own bounds. Only ``VENUE_SEARCH_LIMIT`` and ``RULES_TEXT_MAX_CHARS`` span
venues, and they live in ``_shared``. This package root re-exports every name so
``from ...market_retrieval.venues import X`` and ``venues.X`` keep reading the same as when this
was one module; the per-venue design notes live in each submodule's docstring.

**A test that PATCHES a CONSTANT must target the submodule, not this package.** The re-exports
below bind values, so ``setattr(venues, "KALSHI_PAGE_MAX_BYTES", 32)`` would rebind a copy nothing
reads while the real bound stays live — patch ``venues.kalshi`` instead. Patching a FUNCTION on
this package still works for callers that go through the package namespace, which is why
``generation.py``, ``session_state.py`` and ``snapshot_stages.py`` import the package and call
``venues.fn(...)``.

Every path here serves a pipeline that hands its WHOLE candidate pool to one ranking call, which
is what makes the per-venue decisions the shape they are. Recall is generation's job, selection
is the ranker's, and no path here fuzzy-selects, score-floors or width-caps on the venue's
behalf.

Two structural rules hold throughout. **Every function takes ``session`` as a parameter** — the
session factory stays in the seam module, where four test files patch it. And **nothing here
reads or writes a cache**: the caches are module globals in the seam module that the orchestrator
imports by name, so the caller owns the TTL and these modules stay pure I/O plus parse. That is
why the Kalshi catalogue pull reports ``complete`` rather than writing ``_KALSHI_CACHE`` itself.
"""

from __future__ import annotations

from metaculus_bot.research.market_retrieval.venues import kalshi, manifold, polymarket, predictit
from metaculus_bot.research.market_retrieval.venues._shared import (
    NO_PRICE_SPREAD,
    RULES_TEXT_MAX_CHARS,
    VENUE_SEARCH_LIMIT,
)
from metaculus_bot.research.market_retrieval.venues.kalshi import (
    KALSHI_EVENT_FIELDS,
    KALSHI_EVENTS_URL,
    KALSHI_MARKET_FIELDS,
    KALSHI_NESTED_HEAD_ONLY_FIELDS,
    KALSHI_NESTED_TAIL_FIELDS,
    KALSHI_PAGE_MAX_ATTEMPTS,
    KALSHI_PAGE_MAX_BYTES,
    KALSHI_RESOLVED_STATUSES,
    CataloguePull,
    kalshi_event_match,
    kalshi_event_rules,
    kalshi_event_usd_liquidity,
    kalshi_prefetch_events,
    kalshi_price_strike,
    kalshi_strike_children,
    kalshi_strike_price,
    kalshi_tradeable_strikes,
    kalshi_usd_liquidity,
)
from metaculus_bot.research.market_retrieval.venues.manifold import (
    MANIFOLD_ANSWER_TEXT_MAX_CHARS,
    MANIFOLD_MARKET_URL,
    MANIFOLD_MAX_ATTEMPTS,
    MANIFOLD_PRICED_OUTCOME_TYPE,
    MANIFOLD_SEARCH_URL,
    MANIFOLD_TOP_ANSWERS_RENDERED,
    MANIFOLD_UNTOUCHED_PROBABILITY,
    manifold_answer_children,
    manifold_market_detail,
    manifold_rules_text,
    manifold_scalar_estimate,
    manifold_search,
    manifold_top_answers,
    parse_manifold_matches,
)
from metaculus_bot.research.market_retrieval.venues.polymarket import (
    POLYMARKET_MAX_ATTEMPTS,
    POLYMARKET_SEARCH_URL,
    POLYMARKET_UNTOUCHED_PRICE,
    parse_polymarket_matches,
    polymarket_event_children,
    polymarket_search,
)
from metaculus_bot.research.market_retrieval.venues.predictit import (
    PREDICTIT_CONTRACTS_RENDERED,
    PREDICTIT_MAX_ATTEMPTS,
    PREDICTIT_URL,
    predictit_contract_children,
    predictit_contract_names,
    predictit_market_match,
    predictit_prefetch,
)

__all__ = [  # noqa: RUF022  # grouped by venue with section comments, not alphabetical
    # The submodules themselves, so a test can patch a CONSTANT where it is actually read.
    "kalshi",
    "manifold",
    "polymarket",
    "predictit",
    # Shared bounds
    "RULES_TEXT_MAX_CHARS",
    "VENUE_SEARCH_LIMIT",
    # Kalshi
    "CataloguePull",
    "KALSHI_EVENTS_URL",
    "KALSHI_EVENT_FIELDS",
    "KALSHI_MARKET_FIELDS",
    "KALSHI_NESTED_HEAD_ONLY_FIELDS",
    "KALSHI_NESTED_TAIL_FIELDS",
    "KALSHI_PAGE_MAX_ATTEMPTS",
    "KALSHI_PAGE_MAX_BYTES",
    "NO_PRICE_SPREAD",
    "KALSHI_RESOLVED_STATUSES",
    "kalshi_event_match",
    "kalshi_event_rules",
    "kalshi_event_usd_liquidity",
    "kalshi_prefetch_events",
    "kalshi_price_strike",
    "kalshi_strike_children",
    "kalshi_strike_price",
    "kalshi_tradeable_strikes",
    "kalshi_usd_liquidity",
    # Manifold
    "MANIFOLD_ANSWER_TEXT_MAX_CHARS",
    "MANIFOLD_MARKET_URL",
    "MANIFOLD_MAX_ATTEMPTS",
    "MANIFOLD_PRICED_OUTCOME_TYPE",
    "MANIFOLD_SEARCH_URL",
    "MANIFOLD_TOP_ANSWERS_RENDERED",
    "MANIFOLD_UNTOUCHED_PROBABILITY",
    "manifold_answer_children",
    "manifold_market_detail",
    "manifold_rules_text",
    "manifold_scalar_estimate",
    "manifold_search",
    "manifold_top_answers",
    "parse_manifold_matches",
    # Polymarket
    "POLYMARKET_MAX_ATTEMPTS",
    "POLYMARKET_SEARCH_URL",
    "POLYMARKET_UNTOUCHED_PRICE",
    "parse_polymarket_matches",
    "polymarket_event_children",
    "polymarket_search",
    # PredictIt
    "PREDICTIT_CONTRACTS_RENDERED",
    "PREDICTIT_MAX_ATTEMPTS",
    "PREDICTIT_URL",
    "predictit_contract_children",
    "predictit_contract_names",
    "predictit_market_match",
    "predictit_prefetch",
]
