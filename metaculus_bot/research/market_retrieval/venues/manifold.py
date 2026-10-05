"""Manifold: search plus a per-candidate detail GET, because a multi-outcome row is priceless.

``contractType=ALL``, and a multi-outcome row's price arrives from a SECOND request. ``BINARY``
was a measured ~30% recall ceiling (27 of 89 labeled-wanted markets are MULTIPLE_CHOICE /
MULTI_NUMERIC / NUMBER / DATE). Those rows come back with ``probability`` null and no ``answers``
key, so their whole price lives in the per-candidate detail GET the enrichment hook already fires.

That one array is read TWICE, for two surfaces with different room: ``manifold_top_answers`` takes
three leaders for the ranker's one-line candidate, and ``manifold_answer_children`` takes every
answer with its own volume for the rendered ``↳`` sub-rows.

``outcomeType`` IS read, in exactly one place: the price read in ``parse_manifold_matches``, as an
ALLOW-LIST on ``BINARY``. A ``PSEUDO_NUMERIC`` market is why. It trades a VALUE on a ``[min, max]``
scale, publishes no ``answers`` array — so the discriminator above reads it as BINARY — and still
publishes a market-level ``probability``, which on that contract is the value's normalized POSITION
on the scale rather than a probability of anything. Live 2026-08-05, the market Metaculus Q14333's
ranker graded ``same_quantity_same_date`` rendered ``prob`` 0.48 where its ``value`` was 120.97
years, and ``probability * max == value`` to full float precision.

A third read of the same array is a REFUSAL: an answer at exactly 0.5 with zero volume is sitting at
its untouched prior, so ``_priced_or_none`` reports no price for it. Both surfaces take that rule, and
they dispose of it differently — the ranker segment DROPS the answer (a one-line candidate has no room
to say "unpriced") while the sub-row keeps it and renders a blank price, which is itself information.

An allow-list rather than a ``PSEUDO_NUMERIC`` deny-list because the two failures are not
symmetric. A type this parser has not met yet loses a real price VISIBLY — ``prob`` renders ``-``,
and a WARN fires whenever a non-null ``probability`` is declined with no readable ``value`` to show
instead — where a deny-list would print the next such type's scale position as a probability,
silently, exactly as this one did. ``STONK`` is the standing evidence that "the next such type"
is not hypothetical: it is a share-price contract that also carries no ``answers``, and it only
escapes this bug because it happens to publish ``probability`` null today.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from metaculus_bot.research.market_retrieval.http import http_get_with_backoff, safe_float, safe_int
from metaculus_bot.research.market_retrieval.types import MarketChild, MarketMatch, ScalarEstimate
from metaculus_bot.research.market_retrieval.venues._shared import RULES_TEXT_MAX_CHARS, VENUE_SEARCH_LIMIT

logger = logging.getLogger(__name__)

MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
MANIFOLD_MARKET_URL = "https://api.manifold.markets/v0/market"

MANIFOLD_MAX_ATTEMPTS = 2

# The ONE ``outcomeType`` whose market-level ``probability`` is a probability — see the module
# docstring for why this is an allow-list. Every other type either publishes that field as null
# (MULTIPLE_CHOICE, MULTI_NUMERIC, NUMBER, DATE, POLL, STONK) or publishes it as a scale position
# (PSEUDO_NUMERIC), and neither belongs in a column headed `prob`.
MANIFOLD_PRICED_OUTCOME_TYPE = "BINARY"

# A multi-outcome Manifold market's leading answers, kept off its detail payload for the RANKER's
# candidate line. Three, because they become one pipe-separated segment of a one-line candidate and
# the leaders carry the shape of the distribution; the tail, where a threshold ladder's 17 rungs
# live, is what the rendered `↳` sub-rows keep instead (`manifold_answer_children`), since a table
# row has room a prompt line does not.
#
# 60 chars because measured answer texts are short (10-13 on the committed fixture: `Over $4.60`,
# `$3.80 - $4.19`, `Nov-Dec 2026`), so the cap only bounds a pathological answer. It applies to the
# sub-row titles too, under the renderer's own tighter `CHILD_TITLE_MAX_CHARS`.
MANIFOLD_TOP_ANSWERS_RENDERED = 3
MANIFOLD_ANSWER_TEXT_MAX_CHARS = 60

# A freshly-created Manifold answer sits at its 0.5 PRIOR until somebody bets. Exactly this value
# with zero volume is therefore the venue's absence of a price rather than a price, and rendering it
# told a forecaster the crowd was split 50/50 on a rung nobody had touched. Compared exactly, because
# any bet moves the answer off it.
MANIFOLD_UNTOUCHED_PROBABILITY = 0.5


def _priced_or_none(probability: float | None, *, volume: float | None) -> float | None:
    """``None`` for an answer sitting at its untouched 0.5 prior; the probability otherwise.

    Gated on ``volume`` ALONE, unlike the Polymarket sibling: Manifold publishes no per-answer open
    interest, so volume is the only trading evidence an answer carries. And the field must be PRESENT
    and zero — a missing ``volume`` is not evidence of no trading, and blanking on its absence would
    delete real prices from any payload shape that omits it.
    """
    if probability is None or probability != MANIFOLD_UNTOUCHED_PROBABILITY:
        return probability
    return None if volume is not None and volume == 0.0 else probability


def _walk_tiptap_text(node: Any) -> list[str]:
    """Depth-first collect every leaf ``text`` string from a TipTap/ProseMirror doc."""
    out: list[str] = []
    if isinstance(node, dict):
        text = node.get("text")
        if isinstance(text, str) and text:
            out.append(text)
        content = node.get("content")
        if isinstance(content, list):
            for child in content:
                out.extend(_walk_tiptap_text(child))
    elif isinstance(node, list):
        for child in node:
            out.extend(_walk_tiptap_text(child))
    return out


def manifold_rules_text(market: dict[str, Any]) -> str:
    """A Manifold market's rules text: ``textDescription``, else the flattened doc, else "".

    Deliberately NO fall back to the question title: that renders a candidate line whose
    ``rules:`` segment repeats its own title, spending ranker tokens on nothing. The
    enrichment hook is what fills real description text in, and a blank here is the honest
    signal that it has not run or found nothing.
    """
    text_description = market.get("textDescription")
    if isinstance(text_description, str) and text_description.strip():
        return text_description[:RULES_TEXT_MAX_CHARS]

    description = market.get("description")
    if isinstance(description, dict):
        collected = _walk_tiptap_text(description)
        return " ".join(collected)[:RULES_TEXT_MAX_CHARS] if collected else ""
    if isinstance(description, str) and description.strip():
        return description[:RULES_TEXT_MAX_CHARS]
    return ""


def manifold_scalar_estimate(market: dict[str, Any]) -> ScalarEstimate | None:
    """A scalar market's traded value and its scale, or ``None`` when there is no value to show.

    Read straight off the SEARCH listing, which was the surprise: a ``PSEUDO_NUMERIC`` row arrives
    from ``/search-markets`` carrying ``value``, ``min``, ``max`` and ``isLogScale`` alongside the
    misleading ``probability`` (live-verified 2026-08-05), so both detecting the shape and pricing it
    are free — this adds no request, and the detail GET the enrichment hook fires is not involved.

    ``value`` is taken verbatim and NEVER derived from ``probability``. That is safe arithmetic on a
    linear market (``probability * max == value`` exactly) and catastrophic on a log one, where the
    same product overstates the venue's own figure by 29x on a 2000-250000 scale and 6554x on a
    1-10000000 one. Reading the field is correct on both and needs no branch on the scale, so
    ``is_log_scale`` is carried for the FORECASTER's benefit — where a value sits between its bounds
    means something different on a log axis — and not for any computation here.

    A row with no readable ``value`` returns ``None`` and renders no price at all, rather than
    falling back to the one number that is definitely wrong.
    """
    value = safe_float(market.get("value"))
    if value is None:
        return None
    return ScalarEstimate(
        value=value,
        minimum=safe_float(market.get("min")),
        maximum=safe_float(market.get("max")),
        is_log_scale=bool(market.get("isLogScale")),
    )


def parse_manifold_matches(payload: Any, *, width: int) -> list[MarketMatch] | None:
    """Parse a Manifold search response into candidate rows, venue-rank order.

    Close time and every liquidity field are read off EACH SEARCH ROW. That is the fix for
    the blank close/liquidity cells the bake-off measured on 52 of 94 Manifold rows: those
    came from looking each row up in a cached universe it was never in.

    ``None`` when the TOP-LEVEL shape is not the documented array, honouring the package's
    ``None``-means-fetch-failed contract. ``http_get_with_backoff`` has already turned every
    non-200 and every undecodable body into ``None`` before this runs, so a shapeless payload
    here is a 200 whose contract changed — a loss, not a search that found nothing, and ``[]``
    published it as a benign ``none``. Per-ROW malformation still skips the row: one unreadable
    market among ten is not a venue-wide failure.
    """
    if not isinstance(payload, list):
        logger.warning(f"Manifold returned a {type(payload).__name__} payload, not the documented array")
        return None

    out: list[MarketMatch] = []
    for rank, market in enumerate(payload):
        if len(out) >= width:
            break
        if not isinstance(market, dict):
            continue
        slug = market.get("slug") or ""
        creator = market.get("creatorUsername") or ""
        url = f"https://manifold.markets/{creator}/{slug}" if slug and creator else (market.get("url") or "")

        close_ms = market.get("closeTime")
        close_time: datetime | None = None
        if isinstance(close_ms, (int, float)):
            try:
                close_time = datetime.fromtimestamp(float(close_ms) / 1000.0, tz=UTC)
            except (OverflowError, OSError, ValueError):
                close_time = None

        # Either a probability or a scalar value, never both — the whole point of the allow-list.
        raw_probability: float | None = None
        if market.get("outcomeType") == MANIFOLD_PRICED_OUTCOME_TYPE:
            # Same untouched-prior blanking the two ANSWER surfaces get: a fresh binary
            # market is created at exactly 0.50 with zero volume, and that manufactured
            # number would render in the PARENT `prob` cell — the one the prompt tells
            # models to anchor on (the sweep's one missed same-class site; the archive
            # holds a live specimen at 0.5 / volume 0.0).
            raw_probability = safe_float(market.get("probability"))
            implied_prob_yes = _priced_or_none(raw_probability, volume=safe_float(market.get("volume")))
            scalar_estimate = None
        else:
            implied_prob_yes = None
            scalar_estimate = manifold_scalar_estimate(market)
            if scalar_estimate is None and market.get("probability") is not None:
                # The unhandled shape, and the reason the allow-list is safe to prefer: a type
                # publishing a market-level number this parser cannot interpret loses its price
                # LOUDLY here rather than rendering it under a `prob` header. No type does this
                # today (PSEUDO_NUMERIC ships `value`; every other non-BINARY type ships
                # `probability` null), so a line in the run log means Manifold shipped a new one.
                logger.warning(
                    f"Manifold {market.get('outcomeType')!r} market carries a market-level probability "
                    f"but no readable value; rendering no price for {market.get('id')!r}"
                )

        out.append(
            MarketMatch(
                platform="manifold",
                market_title=market.get("question") or "",
                market_url=url,
                implied_prob_yes=implied_prob_yes,
                bid=None,
                ask=None,
                spread=None,
                volume_24h=safe_float(market.get("volume24Hours")),
                close_time=close_time,
                is_resolved=bool(market.get("isResolved")),
                match_confidence=100.0 - rank,
                raw_rules=manifold_rules_text(market),
                total_volume=safe_float(market.get("volume")),
                liquidity=safe_float(market.get("totalLiquidity")),
                num_bettors=safe_int(market.get("uniqueBettorCount")),
                venue_market_id=str(market.get("id") or ""),
                retrieval_channel="venue_search",
                scalar_estimate=scalar_estimate,
                # Same contract as the Polymarket parent: withheld means "the venue
                # published a number and this repo refused it", so the render's
                # `withheld=` telemetry counts the blanked parent too.
                price_withheld=implied_prob_yes is None and raw_probability is not None,
            )
        )
    return out


async def manifold_search(session: Any, query: str, *, width: int) -> list[MarketMatch] | None:
    """Search Manifold for one query. ``None`` when the fetch failed OR the shape changed.

    ``contractType=ALL`` rather than ``BINARY``, which was a ~30% recall ceiling: 27 of the 89
    labeled-wanted Manifold markets are MULTIPLE_CHOICE / MULTI_NUMERIC / NUMBER / DATE, and no
    query, width or enrichment could reach them while the parameter was pinned. On this module's
    own fixture term the same search returns 6 BINARY under ``BINARY`` and 6 BINARY + 2
    MULTIPLE_CHOICE under ``ALL``.

    A multi-outcome row arrives with ``probability`` NULL and NO ``answers`` key — the search
    response carries no per-answer data at all — so it would reach the ranker priceless. The
    answers come from the per-candidate detail GET the enrichment hook already fires
    (``manifold_top_answers`` below); a row whose detail was lost stays title-only and renders
    ``-`` for its price, the same soft-fail the rules text has.
    """
    payload = await http_get_with_backoff(
        session,
        MANIFOLD_SEARCH_URL,
        {"term": query, "contractType": "ALL", "limit": str(VENUE_SEARCH_LIMIT)},
        max_attempts=MANIFOLD_MAX_ATTEMPTS,
        retryable_statuses=(429, 500, 502, 503, 504),
        label=f"Manifold q={query[:40]!r}",  # HARNESS-SCAN-EXEMPT-subsampling  # log-label truncation
        component="manifold",
    )
    if payload is None:
        return None
    return parse_manifold_matches(payload, width=width)


async def manifold_market_detail(session: Any, market_id: str) -> dict[str, Any] | None:
    """One market's full record, for the ``textDescription`` and ``answers`` the listing omits.

    ``None`` on any failure, which the enrichment hook treats as "leave the row title-only":
    a lost detail GET costs no recall, only rules text and — on a multi-outcome row — its only
    price. One attempt, no retry — 60 of these fire per question under a 10s fan-out wall, and a
    row's description is not worth spending that wall on twice.
    """
    payload = await http_get_with_backoff(
        session,
        f"{MANIFOLD_MARKET_URL}/{market_id}",
        {},
        max_attempts=1,
        retryable_statuses=(429, 500, 502, 503, 504),
        label=f"Manifold detail id={market_id[:40]!r}",  # HARNESS-SCAN-EXEMPT-subsampling  # log-label truncation
    )
    return payload if isinstance(payload, dict) else None


def manifold_top_answers(detail: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    """A multi-outcome market's leading ``(answer_text, probability)`` pairs, highest first.

    The detail payload's ``answers`` array is where ALL of a non-BINARY market's price
    information lives: the search listing omits the array entirely and its market-level
    ``probability`` is null, so before this a multi-outcome row reached the forecaster with a
    title and nothing else.

    The array IS the outcome-type discriminator for the ANSWERS path, which is why nothing here
    reads ``outcomeType``: a BINARY detail carries no ``answers`` key at all, while MULTIPLE_CHOICE,
    MULTI_NUMERIC, NUMBER and DATE all publish the same ``{text, probability}`` answer shape
    (live-verified across all five, 2026-08-05), so one read covers the whole ~30% of Manifold
    the ``contractType`` flip reaches without a second field to keep in sync.

    That still holds, and it is now the narrower of two claims: the PRICE read in
    ``parse_manifold_matches`` does consult ``outcomeType``, because a scalar ``PSEUDO_NUMERIC``
    market publishes no answers either and would otherwise pass for BINARY. The distinction is that
    the discriminator here is asked "are there several prices?", which the array answers exactly,
    while the price read is asked "is this one number a probability?", which it cannot.

    The sort's STABILITY is load-bearing rather than incidental. A threshold ladder resolves its
    crossed rungs to probability exactly 1.0 — 10 of 17 answers on the committed fixture — so the
    leaders are routinely a tie, and only keeping the array's own order among equals makes the
    pick deterministic across runs. ``reverse=True`` preserves that order rather than inverting
    it, per the sort's documented guarantee.

    An answer without usable text or a readable probability is dropped rather than carried blank,
    mirroring ``predictit_contract_names``: an empty label spends ranker tokens saying nothing, and a
    non-finite probability would defeat every comparison in the sort and then render as ``nan%``.

    An answer at its untouched 0.5 prior with zero volume is dropped under the SAME rule
    (``_priced_or_none``), and this surface matters upstream of the render: these pairs are what the
    RANKER sees of a multi-outcome Manifold market, so a defaulted answer here distorts which markets
    get selected at all — a 0.5 answer would routinely lead the three-leader segment and describe the
    market to the ranker as an even split. Dropping rather than carrying it blank is this function's
    existing rule for an answer with nothing to say.

    The sort is unchanged and stays here: it picks the LEADERS for a one-line prompt segment, which is
    a selection rather than a presentation order, so the "parsers do not sort" rule that moved the
    render order into ``rendering`` does not reach it.
    """
    answers = detail.get("answers")
    if not isinstance(answers, list):
        return ()

    scored: list[tuple[str, float]] = []
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        # Answer text is user-authored, so whitespace is collapsed here at the boundary — a
        # newline would break both the one-line candidate line and the one-line rules bullet.
        text = " ".join(str(answer.get("text") or "").split())
        probability = _priced_or_none(safe_float(answer.get("probability")), volume=safe_float(answer.get("volume")))
        if not text or probability is None:
            continue
        scored.append((text[:MANIFOLD_ANSWER_TEXT_MAX_CHARS], probability))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return tuple(scored[:MANIFOLD_TOP_ANSWERS_RENDERED])


def manifold_answer_children(detail: dict[str, Any]) -> tuple[MarketChild, ...]:
    """Every answer as its own sub-row, in the ANSWERS ARRAY's own order. ``()`` on a BINARY market.

    The render-side counterpart to ``manifold_top_answers``, off the same ``answers`` array and with
    the same discriminator (a BINARY detail carries no such key), differing in two ways that follow
    from where each one goes. This keeps ALL the answers rather than the leading
    ``MANIFOLD_TOP_ANSWERS_RENDERED``, because the renderer has its own child budget and a
    three-answer cap tuned for a one-line ranker segment would throw away rungs the table has room
    for. And each answer keeps its own ``volume`` — the sub-row has a column for it, the ranker line
    does not.

    **Unsorted, and that is the contract.** This venue's ordering set the pattern the other two
    copied, and the whole pattern moved into ``rendering`` on 2026-08-25: the renderer sorts a copy by
    ``child_render_order_key`` for the full sub-rows and reads THIS order for the ladder row naming
    every remaining outcome. A parser sorting its children was deciding what survived a render budget
    it could not see, and Manifold's array order is the ladder's own threshold order.

    Nothing about the old ordering argument was wrong, only misplaced: a threshold ladder settles its
    crossed rungs to exactly 1.0 while the market stays open (10 of 17 on this module's committed
    fixture), so a price-only order would still spend the full rows on the part of the ladder that is
    no longer a forecast. ``child_render_order_key`` still queues settled rungs behind open ones for
    exactly that reason — it just does it where the budget is visible.

    An answer at its untouched 0.5 prior with zero volume carries no price (``_priced_or_none``) and
    says so via ``price_withheld``. It is still RENDERED, unlike in ``manifold_top_answers``: a table
    row can say "this rung exists and nobody has priced it", which is real information, where a
    one-line ranker segment cannot.

    ``num_bettors`` is the MARKET's ``uniqueBettorCount``, copied onto every child: Manifold scores
    participation on bettors and publishes no per-answer count, so this is the honest figure for each
    answer (they share one bettor pool) and it makes each sub-row's ``signal`` cell agree with its
    parent's instead of reading a false ``no-liquidity-data``. Each answer's OWN ``volume`` outranks
    it in the label, though — a zero there reads ``thin`` whatever the market's pool is
    (``liquidity_label_from_fields``), because the shared pool cannot vouch for a rung nobody bet on,
    and the untouched-prior answers this parser blanks a price for were reading ``high``.
    """
    answers = detail.get("answers")
    if not isinstance(answers, list):
        return ()

    num_bettors = safe_int(detail.get("uniqueBettorCount"))
    children: list[MarketChild] = []
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        text = " ".join(str(answer.get("text") or "").split())
        raw_probability = safe_float(answer.get("probability"))
        volume = safe_float(answer.get("volume"))
        if not text or raw_probability is None:
            continue
        probability = _priced_or_none(raw_probability, volume=volume)
        children.append(
            MarketChild(
                title=text[:MANIFOLD_ANSWER_TEXT_MAX_CHARS],
                implied_prob_yes=probability,
                total_volume=volume,
                num_bettors=num_bettors,
                # A resolved answer carries a `resolution` verdict ("YES"/"NO"); an open one carries
                # null. On a threshold ladder the crossed rungs settle individually while the market
                # stays open, so this is per-answer rather than the market's own flag.
                is_resolved=bool(answer.get("resolution")),
                price_withheld=probability is None,
            )
        )
    return tuple(children)
