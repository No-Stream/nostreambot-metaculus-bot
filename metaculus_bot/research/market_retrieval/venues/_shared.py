"""The bounds and rules more than one venue parser reads.

Everything else in this package is venue-local by construction: a constant that only Kalshi's
catalogue pull or only Manifold's answers array cares about lives in that venue's module, so
this file stays the short list of what genuinely spans venues.
"""

from __future__ import annotations

from metaculus_bot.research.market_retrieval.types import MarketChild

# The venue-search endpoints' own `limit`. NOT a retrieval width: the pool's per-venue width
# is generation's business, and the parsers take theirs as an explicit argument, so no hard
# slice in a parser can silently cap a wider pool — a `payload[:10]` left in one would make
# "width 60" mean 10 per query with nothing to see at the call site.
VENUE_SEARCH_LIMIT = 10

# Rules-text retention at parse time. Generous here and tightened per venue in the ranker prompt:
# this bound only stops a pathological row from being carried around.
RULES_TEXT_MAX_CHARS = 2000

# A book this wide does not imply a price, so a venue reports none rather than its midpoint:
# `kalshi_strike_price`, and Polymarket's untraded legs, whose Gamma price IS that midpoint (a live
# 0.001/0.498 book quoted 0.2495). Calibration is thin and deliberately generous: the only committed
# live-book captures quote real spreads of 0.01-0.10 (11 live strikes across the three Kalshi
# fixtures), an empty book is `yes_bid 0.0000` / `yes_ask 1.0000` (spread 1.0), and the Kalshi module
# documents empty books as 1,063 of 1,066 settled strikes versus 44 of 71,413 live ones. 0.40 sits 4x
# above the widest observed real spread, so it fires on a degenerate book and not on a genuinely wide
# quote. The `MARKET_CHILD_RENDER` marker's `withheld=` field reports how often it fires in prod, so
# this can be tightened on measured data instead of guesswork.
NO_PRICE_SPREAD = 0.40


def child_render_order_key(child: MarketChild) -> tuple[bool, float]:
    """The FULL-ROW presentation order for child outcomes: open first, then price-descending.

    **It no longer decides what survives a budget, and nothing outside ``rendering`` may call it.**
    That is the contract as of 2026-08-25, and it is what the whole ordering argument turned on. The
    renderer used to truncate a family from the end, so a parser sorting its own children was
    choosing which prices a forecaster would ever see — a presentation decision taken where it could
    not see the budget it had to fit. It is also unanswerable as posed: on a mutually-exclusive
    bracket family price-descending is right (the leaders are the mass), and on a cumulative
    threshold ladder it is close to worst-possible (it fronts the deep in-the-money rungs at 0.99 and
    cuts the crossing region). A 50-rung gold ladder rendered ``Above $3451.99 0.99 / Above $3771.99
    0.99 / Above $3291.99 0.99`` under it. PredictIt refused to sort for exactly that reason and its
    docstring said so; three venues sorting and one not was the tell.

    The renderer now names every remaining outcome in one ladder row, so nothing is dropped and this
    key only chooses which outcomes keep a full sub-row's liquidity / close / status cells. Open
    before settled because a settled rung's price is a realized outcome rather than a forecast — on a
    Manifold threshold ladder the crossed rungs settle to exactly 1.0 while the market stays open, so
    price alone would spend the full rows on the part of the ladder that is no longer a forecast.
    Price-descending within each group because the priced rungs are the market's answer, whatever
    traded. A child with no quote (including one whose manufactured default the venue refused) sorts
    with the zero-priced rungs, so a real price always outranks an absent one.

    Callers rely on Python's sort STABILITY: equal-priced (and quoteless) children keep the venue's
    own catalogue order — Kalshi's threshold order, Gamma's array order, Manifold's answer order —
    which is what keeps the render deterministic across runs.
    """
    return (child.is_resolved, -(child.implied_prob_yes or 0.0))
