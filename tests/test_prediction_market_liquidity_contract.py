"""Regression guards for the venue payload CONTRACT: field names, units, and query shape.

Every fixture here is a VERBATIM TRIM of a live payload captured 2026-08-03
(`tests/data/kalshi_live_markets_2026_08_03.json`,
`tests/data/polymarket_live_events_2026_08_03.json`), never a hand-authored dict.
That is the whole point of this file. The pre-existing Kalshi fixture in
`test_prediction_market_provider.py` was written in the same commit as the parser and
encoded the same guessed field names (`volume`, `open_interest`), so the pair was
self-consistent and permanently green while the shipped label read
`no-liquidity-data` on 100% of Kalshi rows for three weeks. A fixture authored from
the same belief as the parser cannot test the parser.

Covered here:

- Kalshi reads `volume_fp` / `open_interest_fp` (the live names; the bare names are
  absent from the current API spec entirely).
- Kalshi contract counts are converted to USD before meeting the USD thresholds, with
  the two quantities using the multipliers their own units imply.
- The rejected FUTURE.md "fall back to volume_24h" fix stays rejected: on the real
  long-horizon markets this bot forecasts, 24h volume is ~0, so that fallback labels a
  deep market "thin".
- `liquidity_dollars` is deprecated upstream and always "0.0000", so nothing may score
  off it.
- Polymarket carries `openInterest` on the EVENT, not the nested market.
- Manifold's query set includes a short-enough term to survive its strict-conjunction
  search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from metaculus_bot.research.market_retrieval.queries import (
    _RELEVANCE_STOPWORDS,
    deterministic_queries,
    manifold_relaxation_terms,
)
from metaculus_bot.research.market_retrieval.venues import (
    kalshi_event_match,
    kalshi_event_usd_liquidity,
    kalshi_usd_liquidity,
    parse_polymarket_matches,
)
from metaculus_bot.research.prediction_market import (
    LIQUIDITY_DECENT_USD,
    LIQUIDITY_DEEP_USD,
    LIQUIDITY_STRONG_USD,
    _liquidity_label,
)

_DATA = Path(__file__).parent / "data"


# Fixtures: verbatim trims of live payloads


@pytest.fixture(scope="module")
def kalshi_live_events() -> list[dict]:
    """Three real Kalshi events (2026-08-03), one per liquidity shape we must get right."""
    payload = json.loads((_DATA / "kalshi_live_markets_2026_08_03.json").read_text())
    return payload["events"]


@pytest.fixture(scope="module")
def polymarket_live_payload() -> dict[str, Any]:
    """Two real Gamma events (2026-08-03) carrying event-level `openInterest`."""
    return json.loads((_DATA / "polymarket_live_events_2026_08_03.json").read_text())


def _event_by_role(events: list[dict], role: str) -> dict:
    for ev in events:
        if ev.get("_fixture_role") == role:
            return ev
    raise AssertionError(f"fixture lost its {role!r} event — regenerate it from a live payload")


def _match_for(events: list[dict], role: str):
    """The fixture event as a candidate row, plus its first nested market.

    No score argument and no floor: ranked retrieval builds a row for EVERY catalogue event and
    lets the ranker select, so the parse is unconditional and this helper cannot accidentally be
    testing a filter.
    """
    ev = _event_by_role(events, role)
    match = kalshi_event_match(ev, match_confidence=1.0, channel="universe_fuzzy")
    assert match is not None, f"fixture event {role!r} produced no candidate row"
    return match, ev["markets"][0]


# D1 — Kalshi field names


class TestKalshiLiquidityFieldNames:
    """The live Kalshi API carries `volume_fp` / `open_interest_fp`. The bare names
    `volume` / `open_interest` are absent from the spec (verified against
    docs.kalshi.com 2026-08-03) and were never present in any archived payload."""

    def test_reads_fp_suffixed_volume_and_open_interest(self, kalshi_live_events):
        """Pins the field NAME and the multiplier together.

        `volume_fp` is a contract count, so the parsed value is that count times trade
        price; `open_interest_fp` is collateral, so it is the count times notional. Both
        legs must be non-None (pre-fix they were None) and must track their own raw field
        (a mixed-up mapping would fail even though both are populated).
        """
        match, raw = _match_for(kalshi_live_events, "deep")
        price = (float(raw["yes_bid_dollars"]) + float(raw["yes_ask_dollars"])) / 2.0
        notional = float(raw["notional_value_dollars"])
        assert match.total_volume == pytest.approx(float(raw["volume_fp"]) * price)
        assert match.open_interest == pytest.approx(float(raw["open_interest_fp"]) * notional)
        # volume_24h is reported as-is (it is a display column, not a threshold input),
        # and it is the one field name that was always correct.
        assert match.volume_24h == pytest.approx(float(raw["volume_24h_fp"]))

    def test_deep_market_is_labelled_deep_not_no_data(self, kalshi_live_events):
        """735k contracts at $0.83 is the single deepest market in the live catalogue.
        Pre-fix this rendered `no-liquidity-data`."""
        match, _ = _match_for(kalshi_live_events, "deep")
        assert _liquidity_label(match) == "deep"

    def test_rejected_volume_24h_fallback_would_mislabel_this_market(self, kalshi_live_events):
        """Pins the REJECTION of the FUTURE.md one-line fix, not just the bug.

        `POWER-28-DH-DS-DP` has 105,980 contracts of lifetime volume and 59,025 of open
        interest at $0.50, but `volume_24h_fp` is exactly 0.00 — the normal state for the
        long-horizon markets this bot forecasts (only 189 of 1,504 live markets have any
        24h volume at all). Scoring off 24h volume would call it `thin`, which the
        forecaster prompt reads as "discount this as noisy".
        """
        match, raw = _match_for(kalshi_live_events, "v24h_trap")
        assert float(raw["volume_24h_fp"]) == 0.0, "fixture no longer exercises the 24h-volume trap"
        assert _liquidity_label(match) != "thin"
        assert _liquidity_label(match) != "no-liquidity-data"

    def test_deprecated_liquidity_dollars_is_not_a_scoring_input(self, kalshi_live_events):
        """`liquidity_dollars` is marked deprecated upstream and documented to always
        return "0.0000". Every fixture market carries that constant, so a label that is
        correct anyway proves nothing scored off it."""
        for role in ("deep", "v24h_trap", "oi_only"):
            match, raw = _match_for(kalshi_live_events, role)
            assert raw["liquidity_dollars"] == "0.0000", "upstream changed; re-read the deprecation note"
            assert _liquidity_label(match) != "no-liquidity-data"

    def test_open_interest_collateral_carries_a_low_priced_market(self, kalshi_live_events):
        """`KXTAIWANLVL4-26JUL01` holds 119,177 contracts of open interest at $0.001.

        Turnover (volume x price) is ~$202, but open interest is COLLATERAL: both sides of
        each contract together post the full $1 notional, so ~$119k of capital is actually
        locked in this market. Scaling open interest by trade price — the obvious
        one-multiplier shortcut — would call it `thin`.
        """
        match, raw = _match_for(kalshi_live_events, "oi_only")
        notional = float(raw["notional_value_dollars"])
        collateral = float(raw["open_interest_fp"]) * notional
        assert collateral > LIQUIDITY_DEEP_USD, "fixture no longer exercises the collateral case"
        assert _liquidity_label(match) == "deep"

    def test_usd_thresholds_receive_usd_not_contract_counts(self, kalshi_live_events):
        """The thresholds are named *_USD and Polymarket feeds them real dollars, so
        Kalshi must not feed them raw contract counts (median trade price is $0.16, so
        counts overstate turnover ~6x)."""
        match, raw = _match_for(kalshi_live_events, "v24h_trap")
        price = (float(raw["yes_bid_dollars"]) + float(raw["yes_ask_dollars"])) / 2.0
        turnover = float(raw["volume_fp"]) * price
        assert match.total_volume is not None
        assert match.total_volume == pytest.approx(turnover, rel=1e-6)
        assert match.total_volume < float(raw["volume_fp"]), "contract counts reached the USD thresholds unconverted"

    def test_thresholds_are_the_shared_usd_pair(self):
        """No per-venue threshold split — one USD ladder for both real-money venues."""
        assert LIQUIDITY_DECENT_USD < LIQUIDITY_STRONG_USD < LIQUIDITY_DEEP_USD


# D1 sibling — the same units contract at EVENT-FAMILY scope


@pytest.fixture(scope="module")
def kalshi_family_events() -> list[dict]:
    """Two real multi-strike Kalshi events (2026-08-05), captured for the family-scope legs."""
    payload = json.loads((_DATA / "kalshi_family_liquidity_2026_08_05.json").read_text())
    return payload["events"]


def _family(events: list[dict], ticker: str) -> dict:
    for event in events:
        if event.get("event_ticker") == ticker:
            return event
    raise AssertionError(f"fixture lost {ticker!r} — regenerate it from a live payload")


class TestKalshiFamilyLiquidityUnits:
    """Every event above is single-strike, so they pin the units on ONE market. A Kalshi event is
    normally a threshold FAMILY (86.5% of the catalogue), and on 2026-08-05 the liquidity legs
    stopped reading ``nested[0]`` and became a per-strike sum. These pin the same field-name and
    multiplier contract at that wider scope, off a payload captured for it.

    The rendered consequence of the scope change — a family whose label flips from ``thin`` to
    ``deep`` — is pinned in ``test_market_retrieval_venues.py``. What is here is the arithmetic."""

    def test_each_strike_converts_at_its_own_price_and_notional(self, kalshi_family_events):
        """The sum is over per-strike DOLLARS, not over counts.

        ``KXGOVWINS-27JAN01`` trades its three strikes at $0.335, $0.12 and $0.535, so the shortcut
        of adding the contract counts and multiplying once is off by a third ($69,204 at the first
        strike's price against a true $107,079) while still producing a plausible-looking figure in
        a rendered column. Recomputed from the raw strings rather than asserted against a constant,
        so a fixture edit cannot quietly move the expectation with the code.
        """
        event = _family(kalshi_family_events, "KXGOVWINS-27JAN01")
        strikes = event["markets"]
        assert [market["status"] for market in strikes] == ["active"] * 3, "fixture must be all-open here"

        expected_volume = sum(
            float(market["volume_fp"]) * (float(market["yes_bid_dollars"]) + float(market["yes_ask_dollars"])) / 2.0
            for market in strikes
        )
        expected_open_interest = sum(
            float(market["open_interest_fp"]) * float(market["notional_value_dollars"]) for market in strikes
        )
        volume, open_interest = kalshi_event_usd_liquidity(strikes)

        assert volume == pytest.approx(expected_volume)
        assert open_interest == pytest.approx(expected_open_interest)

        summed_counts = sum(float(market["volume_fp"]) for market in strikes)
        first_price = (float(strikes[0]["yes_bid_dollars"]) + float(strikes[0]["yes_ask_dollars"])) / 2.0
        assert volume != pytest.approx(summed_counts * first_price), "counts were summed before conversion"

    def test_a_single_strike_family_is_exactly_its_one_strike(self, kalshi_live_events):
        """The n==1 case, which is what ties this class to the rest of the file.

        Every fixture in ``TestKalshiLiquidityFieldNames`` is a one-strike event, so the sum has to
        reduce to ``kalshi_usd_liquidity`` on that strike or those tests are pinning arithmetic the
        shipped path no longer runs. Asserted as an identity across all three roles rather than
        against copied numbers.
        """
        for role in ("deep", "v24h_trap", "oi_only"):
            event = _event_by_role(kalshi_live_events, role)
            (only,) = event["markets"]
            assert kalshi_event_usd_liquidity(event["markets"]) == kalshi_usd_liquidity(only), role

    def test_a_family_publishing_no_counts_still_reports_no_data(self, kalshi_family_events):
        """The ``None`` vs ``0.0`` distinction ``_liquidity_label`` needs, preserved through the sum.

        ``sum([])`` is ``0``, so the obvious implementation turns "upstream sent no volume field" into
        a real zero — which relabels ``no-liquidity-data`` as ``thin``, i.e. tells the forecaster the
        market is illiquid rather than unmeasured. Uses the real payload with the two count fields
        removed, so only the absence differs from the case above.
        """
        event = _family(kalshi_family_events, "KXGOVWINS-27JAN01")
        countless = [
            {key: value for key, value in market.items() if key not in {"volume_fp", "open_interest_fp"}}
            for market in event["markets"]
        ]

        assert kalshi_event_usd_liquidity(countless) == (None, None)

        match = kalshi_event_match({**event, "markets": countless}, match_confidence=1.0, channel="universe_fuzzy")
        assert match is not None
        assert _liquidity_label(match) == "no-liquidity-data"


# D1 sibling — Polymarket open interest nesting level


class TestPolymarketOpenInterestNesting:
    def test_open_interest_read_from_the_event(self, polymarket_live_payload):
        """Gamma puts `openInterest` on the event (5/5 live events) and not on the nested
        market (0/42), so the market-level read yielded None on every row ever archived."""
        ev = polymarket_live_payload["events"][0]
        assert "openInterest" not in ev["markets"][0], "fixture no longer exercises the nesting defect"
        matches = parse_polymarket_matches(polymarket_live_payload, width=10)
        assert matches
        assert matches[0].open_interest == pytest.approx(float(ev["openInterest"]))

    def test_market_level_open_interest_still_wins_when_present(self, polymarket_live_payload):
        """The event-level read is a fallback added on top of the market-level one, not a
        replacement: a market that does carry its own openInterest is more specific."""
        payload = json.loads(json.dumps(polymarket_live_payload))
        ev = payload["events"][0]
        ev["markets"][0]["openInterest"] = 4242.0
        matches = parse_polymarket_matches(payload, width=10)
        assert matches is not None
        assert matches[0].open_interest == pytest.approx(4242.0)

    def test_top_level_markets_fallback_branch_also_reads_open_interest(self):
        """The `events`-empty fallback branch has its own openInterest read; a fix that
        only touches the event branch re-blanks the column whenever Gamma returns
        top-level markets."""
        payload = {
            "events": [],
            "markets": [
                {
                    "question": "Will the Fed cut rates in 2026?",
                    "slug": "fed-cut-2026",
                    "outcomePrices": '["0.61", "0.39"]',
                    "volumeNum": 250000.0,
                    "openInterest": 90000.0,
                }
            ],
        }
        matches = parse_polymarket_matches(payload, width=10)
        assert matches is not None
        assert matches[0].open_interest == pytest.approx(90000.0)
        assert _liquidity_label(matches[0]) == "deep"


# D2 — Manifold query shape


def _content_tokens(term: str) -> int:
    """Manifold requires every CONTENT token to appear in a market's text; stopwords are
    measured not to constrain the match (see the D2 diagnosis), so they don't count."""
    return len([t for t in term.split() if t.lower().strip(".,'") not in _RELEVANCE_STOPWORDS])


class TestManifoldQueryLength:
    """Manifold's `/v0/search-markets` treats `term` as a strict conjunction: every
    content token must appear in a market's text, and one absent token returns `[]`.
    Measured cliff sits at ~4 content tokens. The shipped query set is 9-15-token
    question sentences (S2) plus 5-8-token LLM phrases, so it was nearly always
    unsatisfiable — Manifold contributed zero rows to any bundle for 17+ days while
    reporting healthy."""

    def test_relaxation_ladder_descends_to_a_satisfiable_length(self):
        title = "What will the seasonally adjusted unemployment rate be in Australia for July 2026?"
        terms = manifold_relaxation_terms(title)
        assert terms, "no relaxation rungs produced"
        assert [_content_tokens(t) for t in terms] == sorted((_content_tokens(t) for t in terms), reverse=True), (
            "rungs must descend from most specific to most general"
        )
        assert _content_tokens(terms[-1]) <= 2
        assert _content_tokens(terms[0]) <= 3

    def test_ladder_keeps_the_salient_entity_not_the_scaffolding(self):
        terms = manifold_relaxation_terms("Will a bot finish in the top 5 of the Summer 2026 Metaculus Cup?")
        assert any("Metaculus" in t for t in terms)
        assert not any(t.lower().startswith("will ") for t in terms)

    def test_ladder_drops_duplicate_tokens(self):
        """A title repeating an entity must not yield 'Sturgis Sturgis Motorcycle' — a
        duplicated token spends a rung slot without narrowing anything."""
        terms = manifold_relaxation_terms(
            "How many vehicles will enter Sturgis for the 86th Annual Sturgis Motorcycle Rally?"
        )
        for term in terms:
            lowered = [t.lower() for t in term.split()]
            assert len(lowered) == len(set(lowered)), f"duplicate token in rung {term!r}"

    def test_ladder_is_empty_for_a_title_with_no_content_tokens(self):
        assert manifold_relaxation_terms("Will it?") == []

    def test_the_deterministic_set_leads_with_the_full_length_query(self):
        """The relaxation rungs are ADDITIVE, not a replacement, and they come after the
        precise queries.

        Every rung is now issued unconditionally in parallel rather than walked until one lands,
        because recall is monotone decreasing in token count while precision is monotone
        increasing — the pool wants the precise rung's hits AND the general rung's, and the
        ranker sorts them out. So the full title must still be first and still be issued: a
        short-query-first design loses the high-precision rows a question that already matches at
        full length earns."""
        title = "Will Anthropic release Claude Opus 5 by August 31, 2026?"
        queries = deterministic_queries(title)

        assert queries[0] == title
        for rung in manifold_relaxation_terms(title):
            assert rung in queries
            assert queries.index(rung) > 0
