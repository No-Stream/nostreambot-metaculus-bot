"""The four venue fetch/parse paths, at the four decisions the ranked pipeline rests on.

Each of these is a place where a plausible-looking simplification stays green while quietly
costing recall or misreporting a row:

- The Kalshi ``/events`` stream projects each page down to exactly the retained fields, and a
  200 carrying ``{"events": null}`` reads as a LOST catalogue rather than an empty one — the
  failure that would otherwise pin an empty index for a 6h TTL.
- An event's ``close_time`` is the MAX over its nested markets and it is resolved only when
  ALL of them are. Reading ``nested[0]`` instead misclassifies ~305 live event families as
  RESOLVED on the frozen universe, and both fields are a rendered column and a ranker input.
- PredictIt derives ``close_time`` and ``is_resolved`` from EVERY contract, through a date
  guard that survives the ``NA``/``N/A`` sentinels the dump ships.
- Polymarket sends ``events_status=active``; nothing downstream filters on close date any
  more, so without it a search puts mostly-closed events straight into the pool.

Payload fixtures are real captures, deliberately: the 2026-07-12 Kalshi liquidity regression
shipped because a hand-written fixture was authored from the same wrong belief as the parser,
so the pair agreed and the suite stayed green.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from metaculus_bot.research.market_retrieval import venues
from metaculus_bot.research.market_retrieval.http import (
    flatten_results,
    parse_iso,
    parse_iso_guarded,
    safe_float,
)
from metaculus_bot.research.market_retrieval.types import MarketMatch, ScalarEstimate, _liquidity_label
from tests.market_retrieval_fakes import FakeResponse, FakeSession

_PAYLOADS_PATH = Path(__file__).parent / "data" / "prediction_market_venue_payloads.json"
_MULTI_CLOSE_PATH = Path(__file__).parent / "data" / "kalshi_multi_close_event_2026_08_04.json"
_FAMILY_LIQUIDITY_PATH = Path(__file__).parent / "data" / "kalshi_family_liquidity_2026_08_05.json"
_MANIFOLD_MULTI_OUTCOME_PATH = Path(__file__).parent / "data" / "manifold_multi_outcome_2026_08_05.json"
_MANIFOLD_PSEUDO_NUMERIC_PATH = Path(__file__).parent / "data" / "manifold_pseudo_numeric_2026_08_05.json"


@pytest.fixture(scope="module")
def captured_payloads() -> dict[str, Any]:
    return json.loads(_PAYLOADS_PATH.read_text())


@pytest.fixture(scope="module")
def manifold_multi_outcome() -> dict[str, Any]:
    """A live ``contractType=ALL`` search plus three market details, one per shape that matters.

    The search half shows what a multi-outcome row arrives WITHOUT (a probability, an answers
    array); the detail half shows where its price actually lives, across a MULTIPLE_CHOICE
    ladder, a bucketed MULTI_NUMERIC market, and a BINARY control that carries no answers at all.
    """
    return json.loads(_MANIFOLD_MULTI_OUTCOME_PATH.read_text())


@pytest.fixture(scope="module")
def manifold_pseudo_numeric() -> dict[str, Any]:
    """Live search rows for the scalar shape, kept OUT of the multi-outcome capture above.

    A sibling file rather than two more rows in ``manifold_multi_outcome``, because that fixture
    documents itself as ONE search response for one term, in the venue's own rank order, and four
    assertions across two suites read it that way (``{"BINARY", "MULTIPLE_CHOICE"}``, the
    every-non-BINARY-row-is-priceless sweep, its venue-rank ordering, and provider-health's
    multi-outcome count). Splicing a row from a different search into it would have broken all four
    and made its provenance false.

    Three shapes: the linear market that exposed the bug, two log-scale markets (the receipt that
    the value must be READ rather than recomputed), and a ``STONK`` row for the type that publishes
    neither answers nor a price.
    """
    return json.loads(_MANIFOLD_PSEUDO_NUMERIC_PATH.read_text())


@pytest.fixture(scope="module")
def multi_close_page() -> dict[str, Any]:
    return json.loads(_MULTI_CLOSE_PATH.read_text())


@pytest.fixture(scope="module")
def family_liquidity_page() -> dict[str, Any]:
    """Two live events pinning the family-scope derivations where they change a RENDERED cell.

    ``KXGOVWINS-27JAN01`` is thin on ``nested[0]`` and deep as a family; ``KXNETANYAHUPARDON-26``
    has collapsed to one live strike that is NOT the first, so a positional price read quotes a
    settled strike's realized figure.
    """
    return json.loads(_FAMILY_LIQUIDITY_PATH.read_text())


def _event(page: dict[str, Any], ticker: str) -> dict[str, Any]:
    for event in page["events"]:
        if event.get("event_ticker") == ticker:
            return copy.deepcopy(event)
    raise AssertionError(f"fixture carries no event {ticker!r}")


class TestKalshiEventProjection:
    """The streamed ``/events`` projection: what it keeps, and what it refuses to cache."""

    @pytest.mark.asyncio
    async def test_projection_retains_exactly_the_listed_fields_per_tier(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """A raw market object carries ~38 keys and the catalogue is held for 6h, so the
        projection is the memory bound. This asserts the DROP, not just the keep — which is why
        the fixture is a raw page rather than a pre-projected one.

        Both TIERS are pinned, and the tail is the head MINUS the rules text: every other field is
        read across nested markets (`close_time` by the max derivation, `status` by all-resolved and
        by the tradeable-strike rule, the money fields by the family liquidity sum). The two tuples
        are asserted against the literal `rules_primary` rather than against each other, so the
        derivation cannot vacuously satisfy its own test — a tail that lost the money fields (the
        pre-2026-08-05 shape) would keep the subset relation and re-break the liquidity sum. The
        fixture's 4-market event is what makes the tiers distinguishable at all.
        """
        raw_market_keys = set(multi_close_page["events"][0]["markets"][0])
        assert len(raw_market_keys) > 20, "fixture should be a RAW page, or this proves nothing"
        assert any(len(event["markets"]) > 1 for event in multi_close_page["events"]), (
            "fixture needs a multi-market event, or the tail tier is never exercised"
        )
        assert set(venues.KALSHI_NESTED_TAIL_FIELDS) == set(venues.KALSHI_MARKET_FIELDS) - {"rules_primary"}
        # The per-strike sub-row labels have to survive on EVERY strike, not just the head: they are
        # what each `↳` row is titled, and the tail is where all but one strike lives.
        assert {"yes_sub_title", "ticker"} <= set(venues.KALSHI_NESTED_TAIL_FIELDS)

        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload=multi_close_page)})
        pull = await venues.kalshi_prefetch_events(session)

        assert pull.token == ""
        assert pull.complete is True
        assert len(pull.events) == len(multi_close_page["events"])
        for event in pull.events:
            assert set(event) == {*venues.KALSHI_EVENT_FIELDS, "settlement_sources", "markets"}
            for index, market in enumerate(event["markets"]):
                expected = venues.KALSHI_MARKET_FIELDS if index == 0 else venues.KALSHI_NESTED_TAIL_FIELDS
                assert set(market) == set(expected), f"{event['event_ticker']} market {index}"

    @pytest.mark.asyncio
    async def test_the_tail_tier_retains_every_field_the_family_liquidity_sum_reads(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """The projection and the liquidity sum are one contract, and this is where they meet.

        `kalshi_event_usd_liquidity` converts EVERY tradeable strike at its own price and notional,
        so a tail missing any money field silently zeroes that strike's contribution — no error, no
        log, just a smaller number in a rendered column. The check is behavioural rather than a
        field-list comparison: it runs the real streamed projection and asserts the sum computed
        off the PROJECTED events equals the one computed off the raw fixture. That is what a
        field-list assertion cannot do, because it cannot know which fields the arithmetic reads.
        """
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload=multi_close_page)})
        pull = await venues.kalshi_prefetch_events(session)

        for projected in pull.events:
            raw = _event(multi_close_page, projected["event_ticker"])
            assert len(raw["markets"]) == len(projected["markets"])
            assert venues.kalshi_event_usd_liquidity(projected["markets"]) == venues.kalshi_event_usd_liquidity(
                raw["markets"]
            ), f"{projected['event_ticker']}: the projection dropped a field the liquidity sum reads"

    @pytest.mark.asyncio
    async def test_settlement_sources_are_projected_from_the_event_level(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """``settlement_sources`` lives on the EVENT; every nested market's copy is null, so a
        projection reading the market level would ship a permanently empty join key."""
        raw = _event(multi_close_page, "USCLIMATE")
        assert raw["settlement_sources"], "fixture must carry event-level sources"
        assert all(market.get("settlement_sources") is None for market in raw["markets"])

        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload=multi_close_page)})
        pull = await venues.kalshi_prefetch_events(session)

        projected = next(event for event in pull.events if event["event_ticker"] == "USCLIMATE")
        assert projected["settlement_sources"] == [
            {"name": source["name"], "url": source["url"]} for source in raw["settlement_sources"]
        ]

    @pytest.mark.asyncio
    async def test_a_ticketless_event_is_dropped_as_the_page_streams(self) -> None:
        """The projection's one ROW-level drop, and the only branch in it nothing else guards.

        The event ticker is the pool's dedup key and the settlement index's key, so a ticketless
        event is unreferenceable downstream: it would take a Kalshi width slot and a dedup slot
        while nothing could point at it, and `kalshi_event_match` would build it a URL-less row.
        The sibling event is what makes this a per-row drop rather than a lost page — a projection
        that bailed on the whole body would satisfy an assertion about the bad event alone.
        """
        page = {
            "events": [
                {"event_ticker": "", "title": "no ticker at all", "markets": [{"status": "active"}]},
                {"event_ticker": "KEEPER", "title": "has a ticker", "markets": [{"status": "active"}]},
            ],
            "cursor": "",
        }
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload=page)})

        pull = await venues.kalshi_prefetch_events(session)

        assert [event["event_ticker"] for event in pull.events] == ["KEEPER"]
        assert (pull.token, pull.complete) == ("", True), "one unusable row is not a degraded pull"

    @pytest.mark.asyncio
    async def test_a_200_carrying_a_null_events_array_is_a_lost_catalogue(self) -> None:
        """The failure this check exists for: a 200 whose body has no ``events`` array yields
        zero items exactly like an empty exchange, and caching that pins an empty index for
        6h and silently disables both Kalshi channels."""
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload={"events": None, "cursor": ""})})
        pull = await venues.kalshi_prefetch_events(session)

        assert pull.events == []
        assert pull.token == "error(no_events_array)"
        assert pull.complete is False, "an incomplete pull must never be cached for the TTL"

    @pytest.mark.asyncio
    async def test_a_genuinely_empty_catalogue_is_distinguishable_from_a_lost_one(self) -> None:
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload={"events": [], "cursor": ""})})
        pull = await venues.kalshi_prefetch_events(session)

        assert (pull.events, pull.token, pull.complete) == ([], "", True)

    @pytest.mark.asyncio
    async def test_a_rate_limited_page_stops_pagination_without_retrying(self) -> None:
        """Re-asking a rate limiter 0.5s later is a second violation, not a retry, so a 429
        stops the pull dead and refuses to cache — otherwise a throttled exchange gets pinned
        as a short catalogue for the 6h TTL."""
        session = FakeSession(
            {venues.KALSHI_EVENTS_URL: [FakeResponse(429, text="slow down"), FakeResponse(200, payload={"events": []})]}
        )
        pull = await venues.kalshi_prefetch_events(session)

        assert pull.events == []
        assert pull.token == "error(http_429)"
        assert pull.complete is False
        assert pull.tally.failed == 1
        assert session._call_counts[venues.KALSHI_EVENTS_URL] == 1, "a 429 must not be re-issued"

    @pytest.mark.asyncio
    async def test_a_page_over_the_byte_ceiling_is_dropped_and_not_retried(
        self, monkeypatch: pytest.MonkeyPatch, multi_close_page: dict[str, Any]
    ) -> None:
        """The last-resort guard against a runaway or compressed-bomb body, which had no
        behavioural test at all — neutering it (`if False`) left the whole suite green.

        Non-retryability is half the invariant: a retryable size cap doubles requests against a
        rate-limited exchange for a body that will be just as oversized the second time. The
        other half is `complete=False`, since the seam caches on exactly that flag and a
        `complete=True` here would pin a truncated catalogue for the 6h TTL. The real 64 MB
        ceiling is impractical to materialize, so the constant is patched down — on the
        `venues.kalshi` SUBMODULE, since the `venues` package re-export is a separate binding
        the stream never reads.
        """
        monkeypatch.setattr(venues.kalshi, "KALSHI_PAGE_MAX_BYTES", 32)
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload=multi_close_page)})

        pull = await venues.kalshi_prefetch_events(session)

        assert pull.token == "dropped(size_cap)"
        assert pull.complete is False
        assert pull.events == []
        assert session._call_counts[venues.KALSHI_EVENTS_URL] == 1, "an oversized body must not be re-requested"

    @pytest.mark.asyncio
    async def test_a_body_that_dies_mid_stream_is_a_lost_page_not_an_empty_one(self) -> None:
        """The worst of the untested branches. Mutated to return a clean EMPTY page instead of
        `error(parse)`, a garbled body pins an empty events list for the 6h TTL with `token="none"`
        and the degradation counter NOT bumped — the exact failure the sibling `{"events": null}`
        test exists to prevent, and it never reddens CI.

        The truncated body fires the parse branch at the default ceiling, and it is the first and
        only call site of the `raw_content` hook `FakeResponse` was built for.
        """
        truncated = b'{"events": [{"event_ticker": "A", "title": "x"'
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, raw_content=truncated)})

        pull = await venues.kalshi_prefetch_events(session)

        assert pull.token == "error(parse)"
        assert pull.complete is False
        assert pull.events == []
        assert session._call_counts[venues.KALSHI_EVENTS_URL] == 1, "a malformed body must not be re-requested"

    @pytest.mark.asyncio
    async def test_a_retry_that_would_not_fit_the_wall_is_not_taken(self, caplog: pytest.LogCaptureFixture) -> None:
        """The retry-budget guard: a 503 IS retryable, but sleeping the backoff would consume a
        wall that has almost nothing left, so the pull gives up and reports the transport error
        it actually saw.

        Anchored on THIS guard rather than the loop-top deadline check, which is behaviourally
        shadowed by the per-attempt `attempt_budget <= 0.0` branch — deleting the loop-top check
        leaves token, complete, tally and call count byte-identical, so a test written against it
        would pass on its own mutant. Delete the guard under test instead and the pull sleeps
        0.5s and reports `error(wall_timeout)`, which is what the token assertion catches.
        """
        session = FakeSession(
            {venues.KALSHI_EVENTS_URL: [FakeResponse(503, text="down"), FakeResponse(503, text="still down")]}
        )

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.market_retrieval.venues.kalshi"):
            pull = await venues.kalshi_prefetch_events(session, wall_timeout=0.01)

        assert pull.token == "error(http_503)", "the wall must not relabel the transport failure"
        assert pull.complete is False
        assert session._call_counts[venues.KALSHI_EVENTS_URL] == 1
        messages = "\n".join(caplog.messages)
        assert "retry budget exhausted" in messages
        assert "retry 2/" not in messages, "no backoff may be taken when the wall cannot fit it"

    @pytest.mark.asyncio
    async def test_pagination_follows_the_cursor_and_accumulates_every_page(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        first = {"events": multi_close_page["events"][:1], "cursor": "page2"}
        second = {"events": multi_close_page["events"][1:], "cursor": ""}
        session = FakeSession(
            {venues.KALSHI_EVENTS_URL: [FakeResponse(200, payload=first), FakeResponse(200, payload=second)]}
        )
        pull = await venues.kalshi_prefetch_events(session)

        assert [event["event_ticker"] for event in pull.events] == ["KXMARRIAGESTYLESKRAVITZ-HSZK", "USCLIMATE"]
        assert pull.tally.ok == 2
        assert pull.complete is True

    @pytest.mark.asyncio
    async def test_an_empty_page_with_an_open_cursor_is_a_truncated_pull_not_a_finished_one(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """Only an EXHAUSTED cursor ends pagination on its own terms.

        A mid-pagination page that returns zero rows while still handing back a token has stopped
        short of the catalogue. Treating that as the end reported `complete=True`, so the seam
        pinned the pages collected so far — here 1 of an unknown number — for the whole 6h TTL, and
        every later question in the run read a short universe back as a healthy `ok(n)` with no
        HTTP and no counter bump. The runaway-bound re-check cannot cover it: that fires only on
        `max_pages` / `event_limit`, and one page reaches neither.
        """
        first = {"events": multi_close_page["events"][:1], "cursor": "page2"}
        empty_with_more = {"events": [], "cursor": "page3"}
        session = FakeSession(
            {
                venues.KALSHI_EVENTS_URL: [
                    FakeResponse(200, payload=first),
                    FakeResponse(200, payload=empty_with_more),
                ]
            }
        )

        pull = await venues.kalshi_prefetch_events(session, page_sleep_s=0.0)

        assert [event["event_ticker"] for event in pull.events] == ["KXMARRIAGESTYLESKRAVITZ-HSZK"], (
            "the pages that did arrive must still reach the question that paid for them"
        )
        assert pull.complete is False, "a truncated pull must never be cached for the TTL"
        assert pull.token == "dropped(empty_page)"
        assert pull.tally.ok == 2

    @pytest.mark.asyncio
    async def test_an_empty_page_that_EXHAUSTS_the_cursor_is_a_clean_finish(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """The other side of the rule, and why it keys on the CURSOR rather than the row count.

        A final page carrying no rows and no token is how a catalogue whose size is a multiple of
        the page size ends. Reading that as truncated would refuse to cache every such pull and
        redden CI on a perfectly normal boundary.
        """
        first = {"events": multi_close_page["events"][:1], "cursor": "page2"}
        last = {"events": [], "cursor": ""}
        session = FakeSession(
            {venues.KALSHI_EVENTS_URL: [FakeResponse(200, payload=first), FakeResponse(200, payload=last)]}
        )

        pull = await venues.kalshi_prefetch_events(session, page_sleep_s=0.0)

        assert len(pull.events) == 1
        assert (pull.token, pull.complete) == ("", True)

    @pytest.mark.asyncio
    async def test_the_page_bound_stops_a_runaway_cursor_and_reports_it_incomplete(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """A cursor that never empties must terminate on the page bound, not spin — and the
        catalogue it leaves is TRUNCATED, so the pull is not complete. Reporting `complete=True`
        here would pin a short universe for the 6h TTL behind a green `ok(n)`, which is reachable
        with no code change at all: Kalshi halving its effective page size puts ~38% of the
        universe past the 120-page bound."""
        forever = {"events": multi_close_page["events"][:1], "cursor": "always-more"}
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload=forever)})
        # page_sleep_s=0.0: this test is about the BOUND, not the throttle courtesy the real
        # 0.25s default buys, and three real sleeps per bound test is dead suite time.
        pull = await venues.kalshi_prefetch_events(session, max_pages=3, page_sleep_s=0.0)

        assert pull.tally.ok == 3
        assert len(pull.events) == 3
        assert pull.complete is False
        assert pull.token == "dropped(runaway_bound)"

    @pytest.mark.asyncio
    async def test_the_event_limit_also_reports_the_pull_incomplete(self, multi_close_page: dict[str, Any]) -> None:
        """The sibling bound. Growth past `event_limit` truncates the catalogue exactly as the
        page bound does, and the two must not disagree on what `complete` means."""
        forever = {"events": multi_close_page["events"][:1], "cursor": "always-more"}
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload=forever)})
        pull = await venues.kalshi_prefetch_events(session, event_limit=2, page_sleep_s=0.0)

        assert len(pull.events) == 2
        assert pull.complete is False
        assert pull.token == "dropped(runaway_bound)"


class TestKalshiEventDerivations:
    """The four event-level derivations, none of which may read ``nested[0]``."""

    def test_close_time_is_the_max_over_nested_markets(self, multi_close_page: dict[str, Any]) -> None:
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        nested_closes = [market["close_time"] for market in event["markets"]]
        assert len(set(nested_closes)) == 4, "fixture must have DIFFERING closes or this proves nothing"

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.close_time is not None
        assert match.close_time.strftime("%Y-%m-%d") == "2028-01-01"
        assert match.close_time.strftime("%Y-%m-%d") != nested_closes[0][:10]

    def test_close_time_survives_a_nested_market_missing_its_zulu_suffix(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """The mixed-awareness trap, at the site that actually compares two datetimes.

        `fromisoformat` returns an AWARE datetime for `...Z` and a NAIVE one for a bare
        timestamp, and `max()` over the two raises `TypeError: can't compare offset-naive and
        offset-aware`. It happens inside `to_thread` with no guard, so it reaches the
        snapshot-level net and zeroes all four venues for the question — and the offending event
        stays in the 6h cache, so every later question repeats it. `parse_iso` normalizes at the
        boundary, which covers all four venue parsers at once.
        """
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        latest = max(market["close_time"] for market in event["markets"])
        event["markets"][0]["close_time"] = event["markets"][0]["close_time"].replace("Z", "")

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.close_time is not None
        assert match.close_time.strftime("%Y-%m-%d") == latest[:10]

    def test_an_event_is_resolved_only_when_every_nested_market_is(self, multi_close_page: dict[str, Any]) -> None:
        """``nested[0]`` is ``finalized`` here while the other three are ``active``, so the old
        read reports a live event family RESOLVED — measured on ~305 of them."""
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        assert event["markets"][0]["status"] == "finalized"
        assert {market["status"] for market in event["markets"][1:]} == {"active"}

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.is_resolved is False

    def test_an_all_settled_event_is_resolved(self, multi_close_page: dict[str, Any]) -> None:
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        for market in event["markets"]:
            market["status"] = "settled"

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.is_resolved is True

    def test_liquidity_sums_every_tradeable_strike_rather_than_reading_the_first(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """An event is a threshold family, so its money is the family's, summed per strike.

        `nested[0]` here holds $155 of the family's $201 volume — a 1.3x understatement, mild by
        catalogue standards (median 2.6x, 140x at p90). The magnitude is not the point on this
        fixture; the SCOPE is, and `test_the_family_liquidity_sum_changes_the_rendered_label` pins a
        case where the same scope difference moves a rendered cell.

        The expected values are computed from the per-strike legs rather than hardcoded, because
        `kalshi_usd_liquidity`'s arithmetic is pinned exhaustively in
        `test_prediction_market_liquidity_contract.py` and duplicating it here would just give a
        formula change two places to be wrong in. What this asserts is which strikes are IN, and
        `> nested[0]` alone is what makes it fail on the old read.
        """
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        tradeable = event["markets"][1:]  # nested[0] is `finalized`, the other three `active`
        assert [market["status"] for market in tradeable] == ["active"] * 3
        expected_volume = sum(venues.kalshi_usd_liquidity(market)[0] or 0.0 for market in tradeable)
        expected_oi = sum(venues.kalshi_usd_liquidity(market)[1] or 0.0 for market in tradeable)
        first_volume, first_oi = venues.kalshi_usd_liquidity(event["markets"][0])

        match = venues.kalshi_event_match(event, match_confidence=0.5, channel="universe_fuzzy")

        assert match is not None
        assert match.total_volume is not None
        assert match.open_interest is not None
        assert match.total_volume == pytest.approx(expected_volume)
        assert match.open_interest == pytest.approx(expected_oi)
        assert first_volume is not None
        assert first_oi is not None
        assert match.total_volume > first_volume, "the sum must exceed its own first strike"
        assert match.open_interest > first_oi

    def test_a_settled_strikes_empty_book_is_kept_out_of_the_sum(self, multi_close_page: dict[str, Any]) -> None:
        """The reason the sum's scope is LIVE strikes rather than all of them.

        A settled Kalshi market publishes an empty book (`yes_bid 0.0000` / `yes_ask 1.0000` on
        1,063 of 1,066 settled strikes inside open events), so the bid/ask midpoint reads a
        fabricated $0.50 for it — 16x this strike's own $0.031 last trade. Including it would put
        $155 of invented turnover into a $201 family and move 9 events' rendered liquidity label,
        which is the same class of error as the `nested[0]` read this replaced.

        Asserted through the difference an all-strikes sum WOULD produce, so a scope regression
        fails rather than merely changing a number nobody pinned.
        """
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        settled = event["markets"][0]
        assert settled["status"] == "finalized"
        assert (settled["yes_bid_dollars"], settled["yes_ask_dollars"]) == ("0.0000", "1.0000")
        settled_volume, settled_oi = venues.kalshi_usd_liquidity(settled)
        assert settled_volume is not None
        assert settled_oi is not None

        all_strikes = venues.kalshi_event_usd_liquidity(event["markets"])
        live_only = venues.kalshi_event_usd_liquidity(event["markets"][1:])

        assert venues.kalshi_event_usd_liquidity(event["markets"]) == live_only
        assert all_strikes[0] is not None
        assert live_only[0] is not None
        assert all_strikes[0] == pytest.approx(live_only[0]), (
            "a settled strike's $0.50 phantom midpoint reached the family sum"
        )

    def test_an_all_settled_family_falls_back_to_its_own_strikes(self, multi_close_page: dict[str, Any]) -> None:
        """The empty-set branch: dropping settled strikes must not blank a resolved row's columns.

        No frozen-universe event takes this path (a fully-settled family leaves the open-events
        catalogue), so it is reachable only through the other retrieval channels — which is exactly
        why it needs a test rather than an assumption. `no-liquidity-data` on a market that traded
        $200 would tell a forecaster the venue publishes no figures.
        """
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        for market in event["markets"]:
            market["status"] = "settled"

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.is_resolved is True
        assert match.total_volume == pytest.approx(venues.kalshi_event_usd_liquidity(event["markets"])[0])
        assert match.total_volume is not None
        assert match.total_volume > 0.0
        assert _liquidity_label(match) != "no-liquidity-data"

    def test_a_multi_strike_family_quotes_no_family_price_and_shows_every_strikes_own(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """Three live strikes at $0.285, $0.35 and $0.445 have no single probability, so the FAMILY
        reports none and each strike reports its own instead.

        Both halves are the contract. Quoting one threshold's price under the event's title says
        "35%" about a question the row never asked, so the family-level legs stay `None`. But
        withholding them and stopping there — what this did until the strike children landed — left
        8,103 of 9,417 frozen-universe events rendering a bare `-` with their prices one field away.
        """
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        assert sum(1 for market in event["markets"] if market["status"] == "active") == 3

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert (match.implied_prob_yes, match.bid, match.ask, match.spread) == (None, None, None, None)
        assert match.volume_24h is None, "24h volume belongs to the quoted strike, and none is quoted"
        assert match.total_volume is not None, "withholding the price must not withhold the money"
        assert [(child.title, child.implied_prob_yes) for child in match.children] == [
            ("Before Jan 1, 2027", pytest.approx(0.285)),
            ("Before Jul 1, 2027", pytest.approx(0.35)),
            ("Before Jan 1, 2028", pytest.approx(0.445)),
        ]

    def test_the_strike_children_arrive_in_the_catalogues_own_order(self, multi_close_page: dict[str, Any]) -> None:
        """Unsorted, which is the 2026-08-25 inversion of this assertion.

        It used to pin price-descending, because the renderer truncated a long ladder from the end and
        the parser's order therefore decided which rungs a forecaster would ever see. Nothing is
        truncated now, so the parser has no survival decision to make — and the order it destroyed
        carries real meaning: Kalshi's nested array is threshold/date-ordered, which is the only order a
        cumulative ladder reads in. Presentation moved to the renderer, which sorts a copy for the full
        rows and reads this order for the ladder.
        """
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        expected = [market["yes_sub_title"] for market in event["markets"] if market["status"] == "active"]

        children = venues.kalshi_strike_children(event["markets"])

        assert [child.title for child in children] == expected
        assert expected == ["Before Jan 1, 2027", "Before Jul 1, 2027", "Before Jan 1, 2028"], (
            "the fixture's own array is threshold-ordered, which is what makes this order the meaningful one"
        )
        prices = [child.implied_prob_yes or 0.0 for child in children]
        assert prices != sorted(prices, reverse=True), "and it is NOT price order, or this proves nothing"

    def test_an_open_strike_with_an_empty_book_reports_no_price(self) -> None:
        """A midpoint is only a price when a book supports it.

        An empty Kalshi book is bid 0.0000 / ask 1.0000, whose midpoint is a synthetic $0.50 nobody
        quoted. `kalshi_tradeable_strikes` already scoped SETTLED strikes out for exactly this reason —
        calling the result "a synthetic $0.50 nobody traded at" — while an OPEN strike with an empty book
        had no guard at all, which is how a rendered table came to tell a forecaster the market prices
        `P(diesel > $5.40) = 0.50` with diesel near $4.70. The raw book still rides along so the render
        can show what was quoted.
        """
        strikes = [
            {
                "yes_sub_title": "Above $5.40",
                "status": "active",
                "yes_bid_dollars": "0.0000",
                "yes_ask_dollars": "1.0000",
                "volume_fp": "0",
                "open_interest_fp": "0",
            },
            {
                "yes_sub_title": "Above $4.60",
                "status": "active",
                "yes_bid_dollars": "0.5100",
                "yes_ask_dollars": "0.5400",
                "volume_fp": "900",
                "open_interest_fp": "900",
            },
        ]

        children = venues.kalshi_strike_children(strikes)

        by_label = {child.title: child for child in children}
        assert by_label["Above $5.40"].implied_prob_yes is None
        assert by_label["Above $5.40"].price_withheld is True
        assert (by_label["Above $5.40"].quote_low, by_label["Above $5.40"].quote_high) == (0.0, 1.0)
        assert by_label["Above $4.60"].implied_prob_yes == pytest.approx(0.525)
        assert by_label["Above $4.60"].price_withheld is False

    @pytest.mark.parametrize(
        ("bid", "ask", "priced"),
        [
            ("0.0000", "1.0000", False),  # no book at all
            ("0.3000", "1.0000", False),  # quoted, but far wider than any observed real spread
            ("0.4000", "0.8000", False),  # exactly at the threshold, which is inclusive
            ("0.4000", "0.7900", True),  # just inside it
            ("0.3000", "0.4000", True),  # the widest real spread on any committed capture
        ],
    )
    def test_the_no_price_spread_threshold_is_inclusive(self, bid: str, ask: str, priced: bool) -> None:
        """`KALSHI_NO_PRICE_SPREAD` is the weakest number in this change — calibrated on eleven live
        strikes across the committed captures, whose real spreads run 0.01 to 0.10, so 0.40 sits 4x above
        the widest observed one. Its exact boundary is pinned here because the run-log `withheld=` field
        is what will retune it, and a silently-shifted comparison would make that measurement
        uninterpretable."""
        market = {"yes_bid_dollars": bid, "yes_ask_dollars": ask}

        assert (venues.kalshi_strike_price(market) is not None) is priced

    def test_the_committed_live_books_all_stay_priced(self) -> None:
        """The false-positive guard, over every live TWO-SIDED book in every committed Kalshi
        capture (all three files, 11 books): the threshold must blank degenerate books and
        nothing else. Iterating one capture proved 5 of the 11 the docstring claimed — a
        threshold retune could have blanked a real book in the other two files with this green."""
        capture_files = (
            "kalshi_multi_close_event_2026_08_04.json",
            "kalshi_live_markets_2026_08_03.json",
            "kalshi_family_liquidity_2026_08_05.json",
        )
        live: list[dict[str, Any]] = []
        for name in capture_files:
            page = json.loads((Path(__file__).parent / "data" / name).read_text())
            live.extend(
                market
                for event in page["events"]
                for market in event.get("markets", [])
                if market.get("status") == "active"
                and market.get("yes_bid_dollars") is not None
                and market.get("yes_ask_dollars") is not None
            )
        assert len(live) >= 11, "the committed captures must actually contain the 11 live books"

        for market in live:
            spread = float(market["yes_ask_dollars"]) - float(market["yes_bid_dollars"])
            assert spread < venues.kalshi.KALSHI_NO_PRICE_SPREAD, market.get("yes_sub_title")
            assert venues.kalshi_strike_price(market) is not None, market.get("yes_sub_title")

    def test_the_dollar_conversion_deliberately_does_not_take_the_no_price_rule(self) -> None:
        """`kalshi_usd_liquidity` still midpoints an empty book, and that is the intended asymmetry.

        The spread guard corrects a PRICE; this function converts a contract COUNT into dollars, so
        applying the same rule here would delete a real count rather than fix a bad number. Blanking
        the empty-book midpoint would blank the `total_vol` and `OI` cells on any no-book strike that
        has traded, which is a strictly worse cell than a stale conversion. The one half of the
        no-price rule the function DOES take (since 2026-08-25) is the no-price-at-all case: with
        neither a book nor a last trade there is nothing to convert by, so volume dollars are None
        rather than the old `price or 0.0` manufactured `$0`. The details its docstring calls
        unpinned are pinned here too: the `last_price_dollars` fallback, the `or 1.0` notional
        default, and the truthiness check that routes a genuine 0.0 midpoint to the last trade.
        """
        empty_book = {"volume_fp": "1000", "open_interest_fp": "400", "yes_bid_dollars": "0.0000"}
        empty_book |= {"yes_ask_dollars": "1.0000", "notional_value_dollars": "1.0000"}
        no_quotes = {"volume_fp": "1000", "last_price_dollars": "0.3000", "notional_value_dollars": "1.0000"}
        zero_midpoint = {
            "volume_fp": "1000",
            "yes_bid_dollars": "0.0000",
            "yes_ask_dollars": "0.0000",
            "last_price_dollars": "0.2000",
        }
        traded_no_price = {"volume_fp": "1000", "open_interest_fp": "400"}

        assert venues.kalshi_strike_price(empty_book) is None, "the PRICE is still refused"
        assert venues.kalshi_usd_liquidity(empty_book) == (pytest.approx(500.0), pytest.approx(400.0))
        assert venues.kalshi_usd_liquidity(no_quotes)[0] == pytest.approx(300.0)
        # No notional field at all: open interest converts at the $1.00 default rather than vanishing.
        assert venues.kalshi_usd_liquidity({"open_interest_fp": "400"})[1] == pytest.approx(400.0)
        assert venues.kalshi_usd_liquidity(zero_midpoint)[0] == pytest.approx(200.0)
        # A real contract count with no price to convert by is UNKNOWN dollars, not $0: open
        # interest still converts (notional has a documented $1.00 default) but volume is None.
        assert venues.kalshi_usd_liquidity(traded_no_price) == (None, pytest.approx(400.0))

    def test_a_settled_strike_is_never_rendered_as_a_child(self, multi_close_page: dict[str, Any]) -> None:
        """The same scope the money legs use, for the same reason: a settled Kalshi market publishes
        an EMPTY book (bid 0.0000 / ask 1.0000), so its midpoint is a synthetic $0.50 nobody traded
        at. Rendering that as a sub-row would invent a price, which is worse than the withholding
        the children replace. Belt and braces since 2026-08-25 — the spread guard would blank that
        book anyway — but the scope still earns its place: a settled strike's realized figures are not
        the family's forecast."""
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        settled = event["markets"][0]
        assert (settled["status"], settled["yes_bid_dollars"], settled["yes_ask_dollars"]) == (
            "finalized",
            "0.0000",
            "1.0000",
        )

        children = venues.kalshi_strike_children(event["markets"])

        assert settled["yes_sub_title"] not in {child.title for child in children}
        assert all(child.implied_prob_yes != pytest.approx(0.5) for child in children)

    def test_each_strike_child_carries_its_own_usd_liquidity(self, family_liquidity_page: dict[str, Any]) -> None:
        """A strike's own conversion, not a share of the family sum: the three governorship strikes
        convert at wildly different prices, so per-strike dollars are the only honest figure. The sum
        of the children equals the family total, which is what makes both readings consistent."""
        event = _event(family_liquidity_page, "KXGOVWINS-27JAN01")

        children = venues.kalshi_strike_children(event["markets"])
        family_volume, family_open_interest = venues.kalshi_event_usd_liquidity(event["markets"])

        assert len(children) == 3
        assert sum(child.total_volume or 0.0 for child in children) == pytest.approx(family_volume)
        assert sum(child.open_interest or 0.0 for child in children) == pytest.approx(family_open_interest)
        by_label = {child.title: child for child in children}
        for market in event["markets"]:
            child = by_label[market["yes_sub_title"]]
            volume, open_interest = venues.kalshi_usd_liquidity(market)
            assert (child.total_volume, child.open_interest) == (pytest.approx(volume), pytest.approx(open_interest))

    def test_a_strike_child_falls_back_to_the_ticker_when_the_label_is_missing(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """`yes_sub_title` is present on every strike in every committed capture, but it is not a
        documented guarantee and a labelless sub-row would render blank. The ticker is opaque and it
        is the strike's primary key, so it always exists."""
        event = _event(multi_close_page, "USCLIMATE")
        event["markets"][0].pop("yes_sub_title")

        children = venues.kalshi_strike_children(event["markets"])

        # Membership, not position: the price sort decides where the labelless strike lands.
        assert event["markets"][0]["ticker"] in {child.title for child in children}

    def test_a_collapsed_family_renders_no_children_because_it_quotes_a_price(
        self, family_liquidity_page: dict[str, Any]
    ) -> None:
        """The exact complement of `kalshi_price_strike`: a family either quotes one price at the
        family level or quotes every strike's beneath it, never both and never neither. This one has
        collapsed to a single live strike, so the price belongs on the row itself."""
        event = _event(family_liquidity_page, "KXNETANYAHUPARDON-26")

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.implied_prob_yes is not None
        assert match.children == ()
        assert venues.kalshi_price_strike(event["markets"]) is not None

    @pytest.mark.parametrize("ticker", ["KXMARRIAGESTYLESKRAVITZ-HSZK", "USCLIMATE"])
    def test_a_family_never_carries_both_a_price_and_children(
        self, multi_close_page: dict[str, Any], ticker: str
    ) -> None:
        """The invariant the render depends on: a row with children shows a dash in `prob`, so a row
        carrying both would render its own price AND its outcomes' — two answers to one question."""
        event = _event(multi_close_page, ticker)

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert (match.implied_prob_yes is None) is bool(match.children)

    def test_a_collapsed_family_quotes_its_live_strike_not_the_first(
        self, family_liquidity_page: dict[str, Any]
    ) -> None:
        """The case a positional read gets exactly backwards.

        `KXNETANYAHUPARDON-26` has settled its July strike and left the November one trading, so the
        family asks one live question again and a price is honest. But the live strike is
        `nested[1]`: reading position 0 quotes the settled strike's realized $0.031 as the row's
        current probability, when the market is actually at $0.175. `nested[0]` is the settled one on
        essentially every collapsed family, because Kalshi orders strikes by date — so the positional
        read is not merely unreliable here, it is reliably wrong.
        """
        event = _event(family_liquidity_page, "KXNETANYAHUPARDON-26")
        settled, live = event["markets"]
        assert (settled["status"], live["status"]) == ("finalized", "active")

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.bid == safe_float(live["yes_bid_dollars"])
        assert match.ask == safe_float(live["yes_ask_dollars"])
        assert match.implied_prob_yes == pytest.approx(0.175)
        assert match.implied_prob_yes != pytest.approx(safe_float(settled["last_price_dollars"]))
        assert match.is_resolved is False, "one live strike means the family is still open"

    def test_the_family_liquidity_sum_changes_the_rendered_label(self, family_liquidity_page: dict[str, Any]) -> None:
        """Where the scope change stops being an arithmetic detail and becomes a forecaster-visible
        claim.

        `KXGOVWINS-27JAN01` carries $3,387 of volume on its first strike and $107,079 across the
        three, all live. The `signal` column reads `thin` on the first and `deep` on the family, and
        the prompt tells forecasters to discount thin markets as noisy and anchor on deep ones — so
        the old read told them to discount a $107k market. Only 2 events on the frozen universe make
        that flip at 2-4 all-open strikes, which is why this is a committed live capture rather than
        a hand-built dict.
        """
        event = _event(family_liquidity_page, "KXGOVWINS-27JAN01")
        assert [market["status"] for market in event["markets"]] == ["active"] * 3
        first_volume, first_oi = venues.kalshi_usd_liquidity(event["markets"][0])

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.total_volume == pytest.approx(107_079.02, rel=1e-6)
        assert match.open_interest == pytest.approx(31_470.14, rel=1e-6)
        assert _liquidity_label(match) == "deep"

        first_only = copy.deepcopy(match)
        first_only.total_volume, first_only.open_interest = first_volume, first_oi
        assert _liquidity_label(first_only) == "thin", "fixture no longer straddles a label boundary"

    def test_a_single_strike_event_still_quotes_its_price(self, captured_payloads: dict[str, Any]) -> None:
        """The other side of the rule: a one-strike event and its strike ask the same question, so
        the price is honest and must survive. 13.5% of the catalogue is single-strike."""
        event = copy.deepcopy(captured_payloads["kalshi_events"]["events"][0])
        assert len(event["markets"]) == 1, "fixture must be single-market or this proves nothing"
        only = event["markets"][0]

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        bid, ask = safe_float(only["yes_bid_dollars"]), safe_float(only["yes_ask_dollars"])
        assert bid is not None, "fixture must quote a two-sided book"
        assert ask is not None, "fixture must quote a two-sided book"

        assert match is not None
        assert (match.bid, match.ask) == (bid, ask)
        assert match.implied_prob_yes == pytest.approx((bid + ask) / 2.0)
        assert match.spread == pytest.approx(ask - bid)
        assert match.price_withheld is False

    def test_a_single_strike_event_with_no_book_withholds_its_price_and_keeps_the_book(
        self, captured_payloads: dict[str, Any]
    ) -> None:
        """The PARENT-row half of the no-manufactured-price rule, and the reason one edit to
        `kalshi_strike_price` fixes both sites.

        A one-strike family quotes that strike's price as the market's own, on a row the ranker stamped
        with a relation tier and the forecaster prompts tell a model to anchor on — so an empty book here
        rendered a synthetic $0.50 as the strongest available evidence. The blanking must not take the
        BOOK with it: `bid` / `ask` / `spread` are the evidence FOR the refusal, not a duplicate of the
        price, and they are what lets the render show `0.00-1.00` rather than a bare dash.
        """
        event = copy.deepcopy(captured_payloads["kalshi_events"]["events"][0])
        assert len(event["markets"]) == 1, "fixture must be single-market or this proves nothing"
        event["markets"][0]["yes_bid_dollars"] = "0.0000"
        event["markets"][0]["yes_ask_dollars"] = "1.0000"

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.implied_prob_yes is None
        assert match.price_withheld is True
        assert (match.bid, match.ask) == (0.0, 1.0)
        assert match.spread == pytest.approx(1.0)
        assert match.children == (), "still a one-strike family, so no sub-rows"

    def test_the_row_carries_the_ranked_retrieval_fields(self, multi_close_page: dict[str, Any]) -> None:
        event = _event(multi_close_page, "USCLIMATE")

        match = venues.kalshi_event_match(event, match_confidence=71.5, channel="settlement_join")

        assert match is not None
        assert match.venue_market_id == "USCLIMATE"
        assert match.sub_title == event["sub_title"]
        assert [source.url for source in match.settlement_sources] == [
            source["url"] for source in event["settlement_sources"]
        ]
        assert match.match_confidence == 71.5
        assert match.retrieval_channel == "settlement_join"

    def test_an_event_without_a_title_yields_no_row(self) -> None:
        assert venues.kalshi_event_match({"event_ticker": "X"}, match_confidence=1.0, channel="x") is None

    def test_kalshi_usd_liquidity_uses_the_captured_payload_units(self, captured_payloads: dict[str, Any]) -> None:
        """Volume scales by trade price and open interest by notional value. Reading the bare
        ``volume`` / ``open_interest`` names — which the spec does not define — blanked the
        label on 100% of Kalshi rows once already."""
        market = captured_payloads["kalshi_events"]["events"][0]["markets"][0]
        volume_usd, oi_usd = venues.kalshi_usd_liquidity(market)

        volume_fp = safe_float(market["volume_fp"])
        oi_fp = safe_float(market["open_interest_fp"])
        notional = safe_float(market["notional_value_dollars"]) or 1.0
        assert volume_fp is not None
        assert oi_fp is not None
        assert oi_usd == pytest.approx(oi_fp * notional)
        assert volume_usd is not None, "volume converts at a sub-dollar price"
        assert volume_usd <= volume_fp, "volume converts at a sub-dollar price"

    def test_absent_counts_read_as_no_data_rather_than_zero(self) -> None:
        assert venues.kalshi_usd_liquidity({}) == (None, None)
        assert venues.kalshi_usd_liquidity({"volume_fp": "0", "last_price_dollars": "0.5"})[0] == 0.0


class TestPolymarket:
    @pytest.mark.asyncio
    async def test_the_search_sends_events_status_active(self, captured_payloads: dict[str, Any]) -> None:
        """Nothing downstream filters on close date, so this parameter is the only thing keeping
        resolved markets out of the pool: ``Ethereum`` returns 7 closed events out of 10 without
        it."""
        seen: dict[str, Any] = {}

        def handler(params: dict[str, Any]) -> FakeResponse:
            seen.update(params)
            return FakeResponse(200, payload=captured_payloads["polymarket_search"])

        session = FakeSession({venues.POLYMARKET_SEARCH_URL: handler})

        await venues.polymarket_search(session, "ethereum", width=60)

        assert seen["events_status"] == "active"
        assert seen["q"] == "ethereum"

    def test_the_event_branch_reads_open_interest(self, captured_payloads: dict[str, Any]) -> None:
        """Gamma carries ``openInterest`` on the EVENT (5/5 live events vs 0/42 nested), which
        is why reading it only at the market level blanked the OI column on every archived row."""
        payload = copy.deepcopy(captured_payloads["polymarket_search"])
        payload["events"][0]["openInterest"] = 4321.0
        payload["events"][0]["markets"][0].pop("openInterest", None)

        rows = venues.parse_polymarket_matches(payload, width=60)

        assert rows is not None
        assert rows[0].open_interest == 4321.0
        assert rows[0].platform == "polymarket"

    def test_a_multi_market_event_prices_each_child_and_not_itself(self, captured_payloads: dict[str, Any]) -> None:
        """The mislabelled anchor this whole expansion starts from, pinned on the payload that
        produced it.

        The fixture's first event is titled "How many Fed rate cuts in 2026?" and its first nested
        market is "Will no Fed rate cuts happen in 2026?" at 0.888. Reading `markets[0]` for the
        price legs rendered 0.89 under the EVENT's title — and the forecaster prompts tell the model
        to anchor on a matched market's price, so that row mislabelled an anchor rather than merely
        losing one. Each child now carries its own `groupItemTitle` and its own `outcomePrices`.
        """
        event = captured_payloads["polymarket_search"]["events"][0]
        assert event["title"] == "How many Fed rate cuts in 2026?"
        assert event["markets"][0]["question"] == "Will no Fed rate cuts happen in 2026?"

        rows = venues.parse_polymarket_matches(captured_payloads["polymarket_search"], width=60)

        assert rows is not None
        assert rows[0].market_title == "How many Fed rate cuts in 2026?"
        assert (rows[0].implied_prob_yes, rows[0].bid, rows[0].ask, rows[0].spread) == (None, None, None, None)
        assert [(child.title, child.implied_prob_yes) for child in rows[0].children] == [
            ("0 (0 bps)", pytest.approx(0.888)),
            ("1 (25 bps)", pytest.approx(0.065)),
        ]

    def test_a_multi_market_events_money_comes_off_the_event_not_a_child_sum(
        self, captured_payloads: dict[str, Any]
    ) -> None:
        """Gamma's public-search response TRUNCATES the nested markets list, so the children are a
        subset of the event's outcomes: this event reports $46.2M of volume against $9.1M across the
        two markets it shipped. Summing the visible subset would understate the family 5x, and
        reading `markets[0]` (the pre-expansion behaviour) understated it by nearly 7x."""
        event = captured_payloads["polymarket_search"]["events"][0]
        child_volume_sum = sum(safe_float(market["volumeNum"]) or 0.0 for market in event["markets"])

        rows = venues.parse_polymarket_matches(captured_payloads["polymarket_search"], width=60)

        assert rows is not None
        assert rows[0].total_volume == pytest.approx(safe_float(event["volume"]))
        assert rows[0].open_interest == pytest.approx(safe_float(event["openInterest"]))
        assert rows[0].volume_24h == pytest.approx(safe_float(event["volume24hr"]))
        assert rows[0].total_volume is not None
        assert rows[0].total_volume > 4 * child_volume_sum

    def test_an_absent_24h_volume_stays_none_rather_than_carrying_lifetime_volume(
        self, captured_payloads: dict[str, Any]
    ) -> None:
        """`volume24hr` is optional on Gamma events (absent on 25 of 122 archived rows). The old
        fallback silently substituted the LIFETIME `volume` — a recency field holding a number
        with no recency, which is only harmless while nothing reads it. Absence must stay None."""
        payload = copy.deepcopy(captured_payloads["polymarket_search"])
        event = payload["events"][0]
        del event["volume24hr"]
        for market in event["markets"]:
            market.pop("volume24hr", None)
        assert safe_float(event["volume"]) is not None, "fixture must carry lifetime volume or this proves nothing"

        rows = venues.parse_polymarket_matches(payload, width=60)

        assert rows is not None
        assert rows[0].volume_24h is None
        assert rows[0].total_volume == pytest.approx(safe_float(event["volume"]))

    def test_a_single_market_event_is_unchanged_and_gains_no_children(self, captured_payloads: dict[str, Any]) -> None:
        """The superset half: event and market ask the same question when there is only one, so the
        price legs stay the market's and the row renders exactly as it always has."""
        payload = copy.deepcopy(captured_payloads["polymarket_search"])
        event = payload["events"][1]
        assert len(event["markets"]) == 1, "fixture must be single-market or this proves nothing"

        rows = venues.parse_polymarket_matches(payload, width=60)

        assert rows is not None
        assert rows[1].children == ()
        # Gamma ships `outcomePrices` as a JSON-encoded array of STRINGS, so the float conversion is
        # part of what the parser is being pinned to do.
        assert rows[1].implied_prob_yes == pytest.approx(float(json.loads(event["markets"][0]["outcomePrices"])[0]))
        assert rows[1].bid == safe_float(event["markets"][0]["bestBid"])
        assert rows[1].total_volume == pytest.approx(safe_float(event["markets"][0]["volumeNum"]))

    def test_the_children_arrive_in_gammas_array_order(self, captured_payloads: dict[str, Any]) -> None:
        """Unsorted, which is the 2026-08-25 inversion of this assertion.

        It used to pin price-descending because the renderer truncated from the end. Nothing is truncated
        now — the ladder row names every remaining outcome — so the parser has no survival decision to
        make and Gamma's array order, which is the order an event's own outcome ladder reads in, survives
        to the renderer. Reversing the payload therefore DOES reverse the children, and the renderer is
        where the presentation order is applied.
        """
        payload = copy.deepcopy(captured_payloads["polymarket_search"])
        payload["events"][0]["markets"].reverse()

        rows = venues.parse_polymarket_matches(payload, width=60)

        assert rows is not None
        assert [child.title for child in rows[0].children] == ["1 (25 bps)", "0 (0 bps)"]

    def test_nested_market_children_keep_their_own_bid_ask_book(self, captured_payloads: dict[str, Any]) -> None:
        event = captured_payloads["polymarket_search"]["events"][0]
        markets = event["markets"]
        assert len(markets) > 1

        children = venues.polymarket_event_children(markets)

        assert [(child.quote_low, child.quote_high) for child in children] == [
            (safe_float(market.get("bestBid")), safe_float(market.get("bestAsk"))) for market in markets
        ]

    def test_an_untouched_placeholder_leg_reports_no_price(self) -> None:
        """Gamma's `outcomePrices` default is `["0.5","0.5"]`, so a placeholder leg quotes exactly 0.5
        with nothing behind it — 155 of the archive's 1,839 ranked-era child outcomes.

        This is the shape that made `4e342da` a regression rather than a fix: q45189's own snapshot spent
        four rows on `Candidate A 0.50, B 0.50, C 0.50, D 0.50` while the four real FL-22 candidates
        (Carbonara 0.42, Burck 0.29, Keiser 0.15, Askar 0.12) went unrendered, because a fabricated 0.50
        outranks every one of them under a price-descending order.
        """

        def outcome(title: str, price: float, **fields: Any) -> dict[str, Any]:
            return {"groupItemTitle": title, "outcomePrices": json.dumps([str(price)]), **fields}

        markets = [
            outcome("Candidate A", 0.5, volumeNum=0.0, openInterest=0.0),
            outcome("Candidate B", 0.5),  # no money fields at all
            outcome("Carbonara", 0.42, volumeNum=41_000.0),
        ]

        children = venues.polymarket_event_children(markets)

        by_label = {child.title: child for child in children}
        assert by_label["Candidate A"].implied_prob_yes is None
        assert by_label["Candidate A"].price_withheld is True
        assert by_label["Candidate B"].implied_prob_yes is None
        assert by_label["Carbonara"].implied_prob_yes == pytest.approx(0.42)
        assert by_label["Carbonara"].price_withheld is False

    @pytest.mark.parametrize(
        ("fields", "priced"),
        [
            ({"volumeNum": 0.0, "openInterest": 0.0}, False),
            ({}, False),
            ({"volumeNum": 1_200.0}, True),
            ({"openInterest": 900.0}, True),
            ({"volumeNum": 0.0, "openInterest": 4.0}, True),
        ],
    )
    def test_a_traded_exact_half_keeps_its_price(self, fields: dict[str, Any], priced: bool) -> None:
        """The false-positive guard, and it is a measurement rather than a worry: the archive holds 3
        exact-0.5 legs WITH trading against 155 without. A real 50/50 is a real price, so the guard turns
        on trading evidence and not on the number. Open interest counts as well as volume, because a Gamma
        leg can omit the volume field entirely (59 archived children) while still carrying OI."""
        market = {"groupItemTitle": "coin flip", "outcomePrices": json.dumps(["0.5"]), **fields}

        children = venues.polymarket_event_children([market])

        assert (children[0].implied_prob_yes is not None) is priced

    @pytest.mark.parametrize("price", ["0.49", "0.51", "0.500001"])
    def test_a_price_beside_the_default_is_a_real_price_however_thin(self, price: str) -> None:
        """The guard compares to Gamma's default EXACTLY, and that is the whole reason it is safe.

        Any trade moves a leg off `["0.5","0.5"]`, so a price merely NEAR 0.5 is one somebody quoted —
        and a tolerance here would silently delete real prices from every leg that happens to trade
        close to even. A no-volume leg is a thin market, which the `signal` column already says.
        """
        market = {"groupItemTitle": "close to even", "outcomePrices": json.dumps([price]), "volumeNum": 0.0}

        children = venues.polymarket_event_children([market])

        assert children[0].implied_prob_yes == pytest.approx(float(price))
        assert children[0].price_withheld is False

    def test_a_single_market_events_own_price_takes_the_same_guard(self) -> None:
        """The PARENT-row half. A single-market event quotes its market's price as the row's own, so an
        untouched placeholder reaching the pool through that branch is the same fabrication under a
        relation tier."""
        event = {
            "title": "Party B nominee?",
            "slug": "party-b",
            "markets": [{"question": "Party B nominee?", "outcomePrices": json.dumps(["0.5"]), "volumeNum": 0.0}],
        }

        rows = venues.parse_polymarket_matches({"events": [event]}, width=60)

        assert rows is not None
        assert rows[0].implied_prob_yes is None
        assert rows[0].price_withheld is True

    def test_the_top_level_markets_branch_takes_the_same_guard(self) -> None:
        """The fallback path Gamma uses when a search returns no events reads the same field, so it
        cannot be the one place a manufactured 0.5 survives."""
        payload = {"markets": [{"question": "Placeholder", "outcomePrices": json.dumps(["0.5"]), "volumeNum": 0.0}]}

        rows = venues.parse_polymarket_matches(payload, width=60)

        assert rows is not None
        assert rows[0].implied_prob_yes is None
        assert rows[0].price_withheld is True

    def test_a_settled_leg_is_marked_resolved_in_the_array_order(self) -> None:
        """A nested Polymarket market can individually be closed while the event stays open (Kalshi
        pre-scopes its strikes to live ones, so this venue is where the distinction shows). The flag is
        per-leg, and it is what the RENDERER's order key reads to queue realized prices behind the rungs
        still carrying uncertainty — the parser itself keeps Gamma's order."""

        def outcome(title: str, price: float, *, closed: bool = False) -> dict[str, Any]:
            return {"groupItemTitle": title, "outcomePrices": json.dumps([str(price)]), "closed": closed}

        markets = [
            outcome("0 cuts", 1.00, closed=True),
            outcome("1 cut", 0.62),
            outcome("2 cuts", 0.25),
            outcome("3+ cuts", 0.08),
        ]

        children = venues.polymarket_event_children(markets)

        assert [child.title for child in children] == ["0 cuts", "1 cut", "2 cuts", "3+ cuts"]
        assert [child.is_resolved for child in children] == [True, False, False, False]

    def test_a_child_falls_back_to_its_question_when_it_has_no_group_label(
        self, captured_payloads: dict[str, Any]
    ) -> None:
        """`groupItemTitle` is a third the length and reads as the ladder rung it is, but the live
        capture in `polymarket_live_events_2026_08_03.json` ships it null, so the full binary
        question is the fallback rather than a theoretical one."""
        payload = copy.deepcopy(captured_payloads["polymarket_search"])
        payload["events"][0]["markets"][0]["groupItemTitle"] = None

        rows = venues.parse_polymarket_matches(payload, width=60)

        assert rows is not None
        assert rows[0].children[0].title == "Will no Fed rate cuts happen in 2026?"

    def test_a_leg_with_no_usable_label_is_dropped_rather_than_rendered_blank(self) -> None:
        """A nameless leg would spend a sub-row saying nothing, and the ladder row would name it as a
        bare price. The guard survived the 2026-08-25 rewrite of this loop (a comprehension with a
        walrus became an explicit loop when the price guard was added), and the surviving legs keep
        Gamma's order around the hole."""
        markets = [
            {"groupItemTitle": "1 cut", "outcomePrices": json.dumps(["0.62"])},
            {"outcomePrices": json.dumps(["0.20"])},  # no groupItemTitle, question or title
            {"groupItemTitle": "2 cuts", "outcomePrices": json.dumps(["0.18"])},
        ]

        children = venues.polymarket_event_children(markets)

        assert [child.title for child in children] == ["1 cut", "2 cuts"]

    def test_the_markets_fallback_branch_also_reads_open_interest(self) -> None:
        payload = {
            "events": [],
            "markets": [
                {
                    "question": "Will X happen?",
                    "slug": "will-x-happen",
                    "openInterest": 999.0,
                    "volumeNum": 12345.0,
                    "description": "rules text",
                    "closed": True,
                }
            ],
        }

        rows = venues.parse_polymarket_matches(payload, width=60)

        assert rows is not None
        assert len(rows) == 1
        assert rows[0].open_interest == 999.0
        assert rows[0].total_volume == 12345.0
        assert rows[0].is_resolved is True
        assert rows[0].venue_market_id == "will-x-happen"

    def test_the_score_is_the_inverted_venue_rank(self, captured_payloads: dict[str, Any]) -> None:
        rows = venues.parse_polymarket_matches(captured_payloads["polymarket_search"], width=60)

        assert rows is not None
        assert [row.match_confidence for row in rows] == [100.0 - rank for rank in range(len(rows))]
        assert all(row.retrieval_channel == "venue_search" for row in rows)

    def test_width_bounds_the_parse(self, captured_payloads: dict[str, Any]) -> None:
        payload = copy.deepcopy(captured_payloads["polymarket_search"])
        payload["events"] = payload["events"] * 5

        rows = venues.parse_polymarket_matches(payload, width=3)

        assert rows is not None
        assert len(rows) == 3

    def test_a_wrong_top_level_shape_reads_as_a_lost_fetch_not_an_empty_search(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 200 whose top-level shape is not Gamma's object is a LOSS, and ``[]`` hid that.

        ``http_get_with_backoff`` has already converted every non-200 and every undecodable body
        to ``None`` by the time this runs, so the only payload reaching here shapeless is a 200
        whose contract changed. As ``[]`` that counted as a successful query in
        ``flatten_results`` and published as a benign ``none`` token — the venue read exactly
        like one whose index found nothing, which is the shape that hid a dead Manifold for 17
        days. The WARN has to name the venue and the shape, because the operator's next question
        is which venue broke.
        """
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.market_retrieval.venues.polymarket"):
            assert venues.parse_polymarket_matches(["not", "a", "dict"], width=60) is None

        assert "Polymarket returned a list payload" in "\n".join(caplog.messages)

    def test_a_malformed_ROW_is_skipped_without_losing_the_venue(self) -> None:
        """The other side of the top-level rule: one unreadable event among several is not a
        venue-wide failure, so the row is dropped and its siblings still reach the pool."""
        payload = {"events": ["not a dict", {"title": "Will X happen?", "slug": "x"}]}

        rows = venues.parse_polymarket_matches(payload, width=60)

        assert rows is not None
        assert [row.market_title for row in rows] == ["Will X happen?"]


class TestManifold:
    def test_close_and_liquidity_come_off_each_search_row(self, captured_payloads: dict[str, Any]) -> None:
        """The bake-off measured blank close/liquidity on 52 of 94 Manifold rows because it
        looked each row up in a cached universe it was never in. Everything is read off the
        search row itself now."""
        payload = captured_payloads["manifold_search"]
        rows = venues.parse_manifold_matches(payload, width=60)

        assert rows, "fixture must carry Manifold search rows"
        for row, raw in zip(rows, payload, strict=False):
            assert row.num_bettors == raw.get("uniqueBettorCount")
            assert row.total_volume == safe_float(raw.get("volume"))
            if isinstance(raw.get("closeTime"), (int, float)):
                assert row.close_time is not None
            assert row.venue_market_id == str(raw["id"])

    def test_the_score_is_the_inverted_venue_rank(self, captured_payloads: dict[str, Any]) -> None:
        rows = venues.parse_manifold_matches(captured_payloads["manifold_search"], width=60)

        assert rows is not None
        assert [row.match_confidence for row in rows] == [100.0 - rank for rank in range(len(rows))]

    def test_the_caller_owns_the_row_width(self) -> None:
        """Width is the caller's, never a slice inside the parser: a ``payload[:10]`` here would
        make "width 60" mean 10 per query, with nothing to see at the call site."""
        payload = [{"id": f"m{i}", "question": f"Q{i}", "probability": 0.5} for i in range(25)]

        wide = venues.parse_manifold_matches(payload, width=25)
        narrow = venues.parse_manifold_matches(payload, width=4)

        assert wide is not None
        assert len(wide) == 25
        assert narrow is not None
        assert len(narrow) == 4

    def test_a_wrong_top_level_shape_reads_as_a_lost_fetch_not_an_empty_search(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Manifold's half of the top-level-shape rule; see the Polymarket sibling for why ``[]``
        was the wrong answer. This venue is where the cost was actually paid: it contributed zero
        rows for 17+ days and every channel read it as healthy."""
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.market_retrieval.venues.manifold"):
            assert venues.parse_manifold_matches({"markets": []}, width=60) is None

        assert "Manifold returned a dict payload" in "\n".join(caplog.messages)

    def test_a_malformed_ROW_is_skipped_without_losing_the_venue(self) -> None:
        rows = venues.parse_manifold_matches(["not a dict", {"id": "m1", "question": "Q"}], width=60)

        assert rows is not None
        assert [row.venue_market_id for row in rows] == ["m1"]

    @pytest.mark.asyncio
    async def test_the_search_asks_for_every_contract_type(self, captured_payloads: dict[str, Any]) -> None:
        """The parameter that lifts a measured ~30% recall ceiling.

        ``BINARY`` made 27 of 89 labeled-wanted Manifold markets — every MULTIPLE_CHOICE,
        MULTI_NUMERIC, NUMBER and DATE one — unreachable by any query or width, so this single
        literal is the whole lever. It is asserted rather than left to the fixture because a
        revert to ``BINARY`` would still return rows and would still look healthy everywhere else.
        """
        seen: dict[str, Any] = {}

        def handler(params: dict[str, Any]) -> FakeResponse:
            seen.update(params)
            return FakeResponse(200, payload=captured_payloads["manifold_search"])

        session = FakeSession({venues.MANIFOLD_SEARCH_URL: handler})

        await venues.manifold_search(session, "gas prices", width=60)

        assert seen["contractType"] == "ALL"
        assert seen["term"] == "gas prices"

    def test_rules_text_does_not_fall_back_to_the_title(self) -> None:
        """A title-as-rules row spends ranker tokens repeating its own title; a blank is the
        honest signal that enrichment has not filled it in."""
        assert venues.manifold_rules_text({"question": "Will X happen?"}) == ""

    def test_rules_text_flattens_a_tiptap_description(self) -> None:
        doc = {"type": "doc", "content": [{"type": "p", "content": [{"text": "resolves per"}, {"text": "NOAA"}]}]}

        assert venues.manifold_rules_text({"description": doc}) == "resolves per NOAA"

    @pytest.mark.asyncio
    async def test_a_detail_fetch_returns_none_on_a_non_dict_body(self) -> None:
        session = FakeSession({venues.MANIFOLD_MARKET_URL: FakeResponse(200, payload=["not", "a", "dict"])})

        assert await venues.manifold_market_detail(session, "abc") is None


class TestManifoldMultiOutcome:
    """The rows the ``contractType=BINARY`` pin used to hide, and where their price comes from.

    Two halves of one contract. The SEARCH response is what the pool is built from, and a
    multi-outcome row arrives from it with no probability and no answers — so a parser that read
    only the search would put a priceless row in front of a forecaster. The DETAIL response is
    where the answers live, which is why the enrichment hook (already firing once per Manifold
    candidate for rules text) is the whole render path.
    """

    def test_a_multi_outcome_search_row_parses_with_no_probability_but_keeps_its_liquidity(
        self, manifold_multi_outcome: dict[str, Any]
    ) -> None:
        """The rows the flip admits, straight off a live ``contractType=ALL`` response.

        ``probability`` is null on every one, so ``implied_prob_yes`` stays None and the rendered
        ``prob`` column reads ``-``: a multi-outcome market has no single yes-price to quote, and
        inventing one (the first answer's, say) would put a threshold's probability under the
        market's own title. Everything the liquidity label keys on is present regardless, so these
        rows are labelled exactly like the BINARY ones.
        """
        payload = manifold_multi_outcome["search_all"]
        rows = venues.parse_manifold_matches(payload, width=60)

        assert rows is not None
        multi = [(row, raw) for row, raw in zip(rows, payload, strict=False) if raw["outcomeType"] != "BINARY"]
        assert len(multi) == 2, "fixture must carry multi-outcome rows, or this proves nothing"
        for row, raw in multi:
            assert raw.get("probability") is None
            assert "answers" not in raw
            assert row.implied_prob_yes is None
            assert row.num_bettors == raw["uniqueBettorCount"]
            assert row.close_time is not None
            assert _liquidity_label(row) != "no-liquidity-data"
            assert row.top_answers == (), "the search response has no answer data to populate from"

    def test_a_multi_outcome_row_is_scored_and_ordered_exactly_like_a_binary_one(
        self, manifold_multi_outcome: dict[str, Any]
    ) -> None:
        """No venue-rank penalty for being multi-outcome: `match_confidence` is the venue's own
        inverted rank, and re-scoring these rows down would reintroduce the local re-ranking the
        ranked design exists to remove."""
        payload = manifold_multi_outcome["search_all"]
        rows = venues.parse_manifold_matches(payload, width=60)

        assert rows is not None
        assert [row.match_confidence for row in rows] == [100.0 - rank for rank in range(len(rows))]
        assert [row.market_title for row in rows] == [raw["question"] for raw in payload]

    def test_the_top_answers_come_off_the_detail_payload_highest_first(
        self, manifold_multi_outcome: dict[str, Any]
    ) -> None:
        """A bucketed MULTI_NUMERIC market, whose leaders are distinct — so this pins the ORDER
        as well as the values, which the tied MULTIPLE_CHOICE ladder below cannot."""
        answers = venues.manifold_top_answers(manifold_multi_outcome["detail_multi_numeric"])

        assert [text for text, _ in answers] == ["$3.80 - $4.19", "$3.40 - $3.79", "$4.20 - $4.59"]
        assert [round(prob, 4) for _, prob in answers] == [0.5083, 0.2465, 0.1602]

    def test_the_answer_count_is_capped_at_the_leaders(self, manifold_multi_outcome: dict[str, Any]) -> None:
        """17 answers reach a 3-answer segment: a threshold ladder would otherwise spend the
        bullet's whole character budget on rungs nobody is trading."""
        detail = manifold_multi_outcome["detail_multiple_choice"]

        assert len(detail["answers"]) > venues.MANIFOLD_TOP_ANSWERS_RENDERED
        assert len(venues.manifold_top_answers(detail)) == venues.MANIFOLD_TOP_ANSWERS_RENDERED

    def test_tied_leading_answers_keep_the_arrays_own_order(self, manifold_multi_outcome: dict[str, Any]) -> None:
        """A resolved threshold ladder prices its crossed rungs at exactly 1.0 — 10 of 17 here —
        so the leaders are a TIE and the pick is only deterministic because the sort is stable and
        `reverse=True` preserves the order of equals rather than inverting it. A `sorted(...)[::-1]`
        or a set-based dedup would pass a values-only assertion and reorder the segment run to run.
        """
        detail = manifold_multi_outcome["detail_multiple_choice"]
        tied = [answer["text"] for answer in detail["answers"] if answer["probability"] == 1]

        assert len(tied) > venues.MANIFOLD_TOP_ANSWERS_RENDERED, "fixture lost its tie"
        answers = venues.manifold_top_answers(detail)
        assert [text for text, _ in answers] == tied[: venues.MANIFOLD_TOP_ANSWERS_RENDERED]
        assert {prob for _, prob in answers} == {1.0}
        assert venues.manifold_top_answers(detail) == answers, "the pick must not vary between calls"

    def test_a_binary_detail_yields_no_answers_at_all(self, manifold_multi_outcome: dict[str, Any]) -> None:
        """The discriminator, on a real BINARY detail: no ``answers`` key exists, so reading the
        array is enough to tell the shapes apart and nothing needs to consult ``outcomeType``."""
        detail = manifold_multi_outcome["detail_binary"]

        assert "answers" not in detail
        assert detail["probability"] is not None
        assert venues.manifold_top_answers(detail) == ()

    @pytest.mark.parametrize(
        "detail",
        [
            {},
            {"answers": None},
            {"answers": []},
            {"answers": "not a list"},
            {"answers": ["not a dict"]},
            {"answers": [{"text": "No price", "probability": None}]},
            {"answers": [{"text": "", "probability": 0.5}]},
            {"answers": [{"text": "   ", "probability": 0.5}]},
            {"answers": [{"probability": 0.5}]},
        ],
    )
    def test_an_unusable_answers_block_yields_nothing_rather_than_a_blank_row(self, detail: dict[str, Any]) -> None:
        """An answer with no label or no readable probability is dropped, not rendered empty.

        The empty-label case is the one with teeth: ``answers: (50%)`` would be tolerable, but the
        un-parenthesised form of it is exactly the ``: NN%`` shape the per-model MC option parser
        reads, so a blank label is worth refusing at the boundary rather than downstream.
        """
        assert venues.manifold_top_answers(detail) == ()

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_a_non_finite_answer_probability_is_dropped(self, literal: str) -> None:
        """`json.loads` accepts these bare literals, and NaN defeats every comparison in the
        sort — so one would silently decide the order and then render as `nan%`."""
        detail = {
            "answers": [
                {"text": "Poisoned", "probability": json.loads(literal)},
                {"text": "Real", "probability": 0.25},
            ]
        }

        assert venues.manifold_top_answers(detail) == (("Real", 0.25),)

    def test_answer_text_is_capped_and_flattened(self) -> None:
        """Answer text is user-authored: a newline would break both the one-line candidate line
        and the one-line rules bullet, and an essay-length answer would eat the bullet budget."""
        detail = {
            "answers": [{"text": "A" * 500, "probability": 0.9}, {"text": "two\nlines\there", "probability": 0.8}]
        }

        answers = venues.manifold_top_answers(detail)

        assert answers[0] == ("A" * venues.MANIFOLD_ANSWER_TEXT_MAX_CHARS, 0.9)
        assert answers[1] == ("two lines here", 0.8)

    def test_the_answer_children_keep_every_rung_with_its_own_volume(
        self, manifold_multi_outcome: dict[str, Any]
    ) -> None:
        """The render-side read of the same array, and where it differs from the ranker's.

        `top_answers` keeps three leaders and no volume, because it becomes one segment of a
        one-line candidate line. The `↳` sub-rows have a volume column and their own budget, so they
        keep the WHOLE ladder — throwing 14 of 17 rungs away here would discard the distribution's
        shape to satisfy a cap tuned for a different surface.
        """
        detail = manifold_multi_outcome["detail_multiple_choice"]

        children = venues.manifold_answer_children(detail)

        assert len(children) == len(detail["answers"]) == 17
        assert len(children) > len(venues.manifold_top_answers(detail))
        by_label = {child.title: child for child in children}
        for answer in detail["answers"]:
            assert by_label[answer["text"]].total_volume == pytest.approx(answer["volume"])
            assert by_label[answer["text"]].implied_prob_yes == pytest.approx(answer["probability"])

    def test_the_answer_children_arrive_in_the_arrays_own_order(self, manifold_multi_outcome: dict[str, Any]) -> None:
        """Unsorted, which is the 2026-08-25 inversion of this assertion.

        This venue's probability-descending order set the pattern the other two copied, and the whole
        pattern moved into the renderer: the parser was deciding what survived a render budget it could
        not see, and Manifold's answers array is the ladder's own threshold order. Nothing is truncated
        now, so that order carries meaning instead of consequence.
        """
        detail = manifold_multi_outcome["detail_multi_numeric"]

        children = venues.manifold_answer_children(detail)

        assert [child.title for child in children] == [answer["text"] for answer in detail["answers"]]
        probabilities = [child.implied_prob_yes or 0.0 for child in children]
        assert probabilities != sorted(probabilities, reverse=True), "and NOT probability order"
        assert children[0].title == "Below $3.00", "the array's own first rung, not its most probable"

    def test_an_untouched_answer_at_the_prior_reports_no_price(self) -> None:
        """A freshly-created Manifold answer sits at its 0.5 PRIOR until somebody bets, so exactly 0.5
        with zero volume is the venue's absence of a price. Rendering it told a forecaster the crowd was
        split 50/50 on a rung nobody had touched.

        Gated on volume ALONE, unlike the Polymarket sibling: Manifold publishes no per-answer open
        interest, so volume is the only trading evidence an answer carries. A MISSING volume is not
        evidence of no trading, and blanking on its absence would delete real prices.
        """
        detail = {
            "uniqueBettorCount": 47,
            "answers": [
                {"text": "untouched", "probability": 0.5, "volume": 0.0},
                {"text": "real coin flip", "probability": 0.5, "volume": 900.0},
                {"text": "no volume field", "probability": 0.5},
                {"text": "ordinary rung", "probability": 0.27, "volume": 12.0},
            ],
        }

        children = venues.manifold_answer_children(detail)

        by_label = {child.title: child for child in children}
        assert by_label["untouched"].implied_prob_yes is None
        assert by_label["real coin flip"].implied_prob_yes == pytest.approx(0.5)
        assert by_label["no volume field"].implied_prob_yes == pytest.approx(0.5)
        assert by_label["ordinary rung"].implied_prob_yes == pytest.approx(0.27)
        assert [child.price_withheld for child in children] == [True, False, False, False]

    def test_an_untouched_binary_market_reports_no_parent_price(self) -> None:
        """The third same-class surface, missed by the answer-level sweep: a fresh BINARY
        market is created at exactly 0.50 with zero volume, and that manufactured number
        rendered as the PARENT `prob` cell — the one the prompt tells models to anchor on
        (the archive holds a live specimen at 0.5 / volume 0.0). Same volume-gated blanking,
        same `price_withheld` accounting as the answer surfaces."""
        payload = [
            {
                "id": "m-fresh",
                "question": "Fresh at the prior?",
                "outcomeType": "BINARY",
                "probability": 0.5,
                "volume": 0.0,
            },
            {
                "id": "m-coin",
                "question": "Real coin flip?",
                "outcomeType": "BINARY",
                "probability": 0.5,
                "volume": 900.0,
            },
            {"id": "m-thin", "question": "Ordinary?", "outcomeType": "BINARY", "probability": 0.27, "volume": 12.0},
        ]

        rows = venues.parse_manifold_matches(payload, width=60)

        assert rows is not None
        by_title = {row.market_title: row for row in rows}
        assert by_title["Fresh at the prior?"].implied_prob_yes is None
        assert by_title["Fresh at the prior?"].price_withheld is True
        assert by_title["Real coin flip?"].implied_prob_yes == pytest.approx(0.5)
        assert by_title["Real coin flip?"].price_withheld is False
        assert by_title["Ordinary?"].implied_prob_yes == pytest.approx(0.27)

    def test_an_answer_the_parser_cannot_read_is_skipped_without_losing_its_siblings(self) -> None:
        """The two guards inside the answer loop, which the 2026-08-25 price rewrite edited around: an
        entry that is not a dict and an answer with no text or no probability are both skipped, and the
        readable rungs keep the array's order across the hole. A skip that fell through would put a
        blank-titled row into a family whose whole point is that every outcome is named."""
        detail = {
            "answers": [
                {"text": "Below $3.00", "probability": 0.21, "volume": 30.0},
                "not a dict at all",
                {"text": "", "probability": 0.4},
                {"text": "no probability"},
                {"text": "Above $3.00", "probability": 0.79, "volume": 90.0},
            ]
        }

        children = venues.manifold_answer_children(detail)

        assert [child.title for child in children] == ["Below $3.00", "Above $3.00"]

    @pytest.mark.parametrize("probability", [0.49, 0.51, 0.500001])
    def test_a_probability_beside_the_prior_is_a_real_price_however_thin(self, probability: float) -> None:
        """Compared to the 0.5 prior EXACTLY, because any bet moves an answer off it.

        The Manifold sibling of the Polymarket guard, and the same reasoning: a tolerance would blank
        real prices on every answer that happens to trade near even, while the volume column already
        tells a forecaster how thin the rung is.
        """
        detail = {"answers": [{"text": "close to even", "probability": probability, "volume": 0.0}]}

        children = venues.manifold_answer_children(detail)

        assert children[0].implied_prob_yes == pytest.approx(probability)
        assert children[0].price_withheld is False

    def test_an_untouched_answer_is_dropped_from_the_ranker_segment(self) -> None:
        """The two surfaces dispose of a blanked answer differently, and the asymmetry is deliberate.

        ``manifold_top_answers`` feeds the RANKER's one-line candidate, upstream of the render — so a
        defaulted 0.5 there does not merely misprice a row, it distorts which markets get selected at all
        (0.5 leads the three-leader segment and describes the market to the ranker as an even split). A
        one-line segment has no room to say "unpriced", so the answer is dropped, matching that function's
        existing rule for an answer with nothing to say. A table row CAN say it, so the child survives.
        """
        detail = {
            "uniqueBettorCount": 47,
            "answers": [
                {"text": "untouched", "probability": 0.5, "volume": 0.0},
                {"text": "ordinary rung", "probability": 0.27, "volume": 12.0},
            ],
        }

        assert venues.manifold_top_answers(detail) == (("ordinary rung", 0.27),)
        assert [child.title for child in venues.manifold_answer_children(detail)] == ["untouched", "ordinary rung"]

    def test_a_settled_rung_keeps_its_place_and_carries_its_flag(self, manifold_multi_outcome: dict[str, Any]) -> None:
        """A threshold ladder settles its crossed rungs to exactly 1.0 while the market stays open — 10
        of this fixture's 17 answers — and those rungs are the floor the series has already passed.

        The parser reports the flag per answer and leaves each rung where the array put it. Queueing
        realized prices behind the rungs that still carry uncertainty is the RENDERER's job now, which is
        where the budget they compete for is visible; the reasoning was never wrong, only misplaced.
        """
        detail = manifold_multi_outcome["detail_multiple_choice"]
        settled = sum(1 for answer in detail["answers"] if answer.get("resolution"))
        assert settled > len(detail["answers"]) / 2, "fixture must be settled-heavy, or this proves nothing"

        children = venues.manifold_answer_children(detail)

        flags = [child.is_resolved for child in children]
        assert flags == [bool(answer.get("resolution")) for answer in detail["answers"]]
        assert flags != sorted(flags), "the parser must NOT hoist the open rungs — the renderer does that"
        assert sum(flags) == settled

    def test_a_market_publishing_answers_publishes_no_market_level_price(
        self, manifold_multi_outcome: dict[str, Any]
    ) -> None:
        """The venue convention the render invariant rests on, pinned against the live capture.

        Three of the four adapters enforce "a row with outcomes has no single price" structurally.
        Manifold's ANSWERS path still does not — the enrichment hook attaches children without
        touching `implied_prob_yes` — because the venue itself guarantees it: a multi-outcome market
        arrives with `probability` null and a BINARY one carries no `answers` key. If a payload
        refresh ever shipped both, a row would render its own price AND its outcomes', which is two
        answers to one question and makes the table's legend false. This is where that would be
        caught.

        The PRICE side is no longer left to the venue's convention: `parse_manifold_matches` reads a
        market-level `probability` only on a BINARY row. That is what `PSEUDO_NUMERIC` forced — a
        scalar market publishes no answers AND a non-null `probability`, so it satisfied this
        convention while still carrying a number that is not a price (see
        `TestManifoldScalarValueMarkets`).
        """
        for key in ("detail_multiple_choice", "detail_multi_numeric", "detail_binary"):
            detail = manifold_multi_outcome[key]
            has_answers = isinstance(detail.get("answers"), list)
            assert has_answers is (detail.get("probability") is None), key

        # And on the SEARCH side, which is where the row's `implied_prob_yes` actually comes from.
        # Keyed on `outcomeType`, which the price read now consults — so for the non-BINARY rows this
        # pins the parser's own allow-list, while for the BINARY ones it still pins a payload fact
        # (that Manifold ships them a real probability, without which the venue would go priceless).
        search = manifold_multi_outcome["search_all"]
        assert {market["outcomeType"] for market in search} == {"BINARY", "MULTIPLE_CHOICE"}
        for row, market in zip(venues.parse_manifold_matches(search, width=60) or [], search, strict=False):
            assert (row.implied_prob_yes is None) is (market["outcomeType"] != "BINARY"), row.market_title

    def test_an_answer_child_carries_the_markets_bettor_count(self, manifold_multi_outcome: dict[str, Any]) -> None:
        """The one field that is legitimately the PARENT's. Manifold scores participation on unique
        bettors and publishes no per-answer count, so a child with `num_bettors=None` would render
        `no-liquidity-data` — telling a forecaster the venue publishes no volume figures, on a venue
        that publishes a volume for every answer. Every answer really does share one bettor pool."""
        detail = manifold_multi_outcome["detail_multiple_choice"]
        assert detail["uniqueBettorCount"] > 0

        children = venues.manifold_answer_children(detail)

        assert {child.num_bettors for child in children} == {detail["uniqueBettorCount"]}

    def test_a_settled_rung_is_marked_resolved_while_the_market_stays_open(
        self, manifold_multi_outcome: dict[str, Any]
    ) -> None:
        """A threshold ladder settles its crossed rungs individually — 10 of this fixture's 17
        answers resolved YES while the market itself is open — so `status` is read per answer. A
        rung's realized 1.0 is evidence about what happened, not a forecast, and the `status` column
        is the only thing that says so."""
        detail = manifold_multi_outcome["detail_multiple_choice"]
        assert detail["isResolved"] is False

        children = venues.manifold_answer_children(detail)

        resolved = {child.title for child in children if child.is_resolved}
        assert resolved == {answer["text"] for answer in detail["answers"] if answer.get("resolution")}
        assert 0 < len(resolved) < len(children), "fixture must mix settled and open rungs"

    def test_a_binary_market_has_no_answer_children(self, manifold_multi_outcome: dict[str, Any]) -> None:
        """The same discriminator `top_answers` uses: a BINARY detail carries no `answers` key at
        all, so nothing here reads `outcomeType`."""
        detail = manifold_multi_outcome["detail_binary"]
        assert "answers" not in detail

        assert venues.manifold_answer_children(detail) == ()


class TestManifoldScalarValueMarkets:
    """``PSEUDO_NUMERIC``: the market whose ``probability`` field is not a probability.

    It trades a VALUE on a ``[min, max]`` scale, publishes no ``answers`` array — so the
    array-presence discriminator the multi-outcome path rests on reads it as BINARY — and still
    ships a market-level ``probability``, which on this contract is the value's normalized position
    on that scale. In production the row Metaculus Q14333's ranker graded ``same_quantity_same_date``
    rendered ``prob`` 0.48 on a market whose estimate was 121 years.

    Two of these tests assert arithmetic over the fixture's own figures rather than parser output.
    That is the point: they establish what the payload MEANS, so the parser and the fixture cannot
    agree on a wrong reading the way the 2026-07-12 Kalshi liquidity regression's hand-built pair did.
    """

    def test_the_probability_field_is_the_values_position_on_the_scale(
        self, manifold_pseudo_numeric: dict[str, Any]
    ) -> None:
        """The receipt, exact to the last bit: ``probability * max == value``.

        And the two facts that let it through — no ``answers`` key, so the multi-outcome
        discriminator sees a BINARY market, and a non-null ``probability``, so it prices.
        """
        market = manifold_pseudo_numeric["search_pseudo_numeric"][0]

        assert market["outcomeType"] == "PSEUDO_NUMERIC"
        assert market["probability"] * market["max"] == market["value"]
        assert "answers" not in market
        assert market["probability"] is not None

    def test_a_scalar_row_carries_its_value_and_no_probability(self, manifold_pseudo_numeric: dict[str, Any]) -> None:
        """The fix. ``implied_prob_yes`` stays empty — nothing here is a yes-price — and the value
        rides its own field, where no formatter can render it as one."""
        payload = manifold_pseudo_numeric["search_pseudo_numeric"]
        rows = venues.parse_manifold_matches(payload, width=60)

        assert rows is not None
        assert len(rows) == 1
        row = rows[0]
        assert row.implied_prob_yes is None
        assert row.scalar_estimate == ScalarEstimate(
            value=120.96691732988944, minimum=0.0, maximum=250.0, is_log_scale=False
        )
        # Everything that is NOT the price is untouched by the allow-list, so a scalar row is still a
        # first-class candidate: same venue rank, same liquidity label, same close date.
        assert row.match_confidence == 100.0
        assert row.num_bettors == payload[0]["uniqueBettorCount"]
        assert _liquidity_label(row) == "thin"
        assert row.close_time is not None

    def test_the_value_is_read_from_the_venue_and_never_recomputed(
        self, manifold_pseudo_numeric: dict[str, Any]
    ) -> None:
        """Why ``value`` is read rather than derived, on the two live log-scale markets.

        ``probability * max`` is exact on a linear market, which is what makes it a tempting
        one-liner. On these it overstates the venue's own figure by 29x and 6554x, and interpolating
        between the bounds in log space does not reproduce it either — the mapping is Manifold's, so
        the field is the only honest source and no branch on ``isLogScale`` can rescue arithmetic.
        """
        payload = manifold_pseudo_numeric["search_log_scale"]
        rows = venues.parse_manifold_matches(payload, width=60)

        assert rows is not None
        assert len(rows) == 2
        for row, market in zip(rows, payload, strict=False):
            assert market["isLogScale"] is True
            assert row.implied_prob_yes is None
            estimate = row.scalar_estimate
            assert estimate is not None
            assert estimate.value == market["value"]
            assert estimate.is_log_scale is True
            assert estimate.scale_label == "log scale"
            # The number the arithmetic shortcut would have produced, and how far off it is.
            assert market["probability"] * market["max"] > 20 * estimate.value

    def test_the_naive_linear_map_reproduces_value_exactly_on_every_linear_market(
        self, manifold_pseudo_numeric: dict[str, Any]
    ) -> None:
        """The trap, stated as a passing assertion. This is the test that explains the next one.

        ``min + probability * (max - min)`` IS what Manifold computes for a LINEAR market — exactly,
        to the last bit, on every one of the 11 linear markets in the captured listing (worst relative
        error 2.2e-16). So the one-line recompute looks correct under any test written from linear
        payloads, which is how a wrong implementation of this would ship. Pinned deliberately: if
        someone reads only the log-scale test below, this one tells them why reading `value` is not
        merely a style preference.
        """
        linear = [
            market
            for market in manifold_pseudo_numeric["search_scale_extremes"]
            if not market["isLogScale"] and market["value"]
        ]
        assert len(linear) == 2, "fixture must carry linear markets, or this proves nothing"

        for market in linear:
            naive = market["min"] + market["probability"] * (market["max"] - market["min"])
            assert naive == pytest.approx(market["value"], rel=1e-12)

    def test_a_log_scale_value_is_nothing_like_the_naive_linear_map(
        self, manifold_pseudo_numeric: dict[str, Any]
    ) -> None:
        """The regression guard against a future "simplification" into a recompute.

        On the real markets captured here the naive map that is exact on linear markets is off by
        2.4x on the 2040 world-population market — showing a forecaster 19 billion people where the
        market says 7.8 billion — and by 389,585x on a log market whose value is 1. Manifold's log
        mapping is its own: interpolating between the bounds in log space does not reproduce it
        either (see the fixture's provenance).

        Concretely: this fails the moment `manifold_scalar_estimate` stops reading `value` and starts
        computing one, which no amount of linear-market coverage would catch.
        """
        payload = [market for market in manifold_pseudo_numeric["search_scale_extremes"] if market["isLogScale"]]
        assert len(payload) == 2, "fixture must carry log-scale markets, or this proves nothing"
        rows = venues.parse_manifold_matches(payload, width=60)

        assert rows is not None
        for row, market in zip(rows, payload, strict=False):
            estimate = row.scalar_estimate
            assert estimate is not None
            assert estimate.value == market["value"], "the venue's own figure, verbatim"
            naive = market["min"] + market["probability"] * (market["max"] - market["min"])
            assert naive != pytest.approx(estimate.value, rel=0.1)
            assert naive > 2 * estimate.value

        # The two extremes those rows represent, named so a future reader sees the range of the hazard.
        by_value = {row.scalar_estimate.value: row for row in rows if row.scalar_estimate is not None}
        assert 7_848_589_347.056932 in by_value, "the world-population market (2.4x)"
        assert 1.0 in by_value, "the market whose value is 1 against a naive 389,585"

    def test_a_scalar_market_on_a_unit_scale_is_indistinguishable_by_magnitude(
        self, manifold_pseudo_numeric: dict[str, Any]
    ) -> None:
        """A REAL live market on ``min=0, max=1``, trading a value of 0.468.

        This is why the rendered cell carries a `value` label instead of relying on the number
        looking too large to be a probability. Pre-fix this row rendered `0.47` in a column headed
        `prob`, and no forecaster — and no reviewer — could have told from the number that it was not
        one. It is also the reason the parser's allow-list keys on `outcomeType` rather than on any
        property of the value itself: there is no such property.
        """
        payload = [
            market
            for market in manifold_pseudo_numeric["search_scale_extremes"]
            if market["min"] == 0 and market["max"] == 1
        ]
        assert len(payload) == 1, "fixture must carry the unit-scale market, or this proves nothing"
        rows = venues.parse_manifold_matches(payload, width=60)

        assert rows is not None
        estimate = rows[0].scalar_estimate
        assert rows[0].implied_prob_yes is None
        assert estimate is not None
        assert estimate.value == 0.4680001051790792
        assert 0.0 <= estimate.value <= 1.0, "shaped exactly like a probability, and not one"

    def test_a_type_with_neither_answers_nor_a_value_stays_priceless(
        self, manifold_pseudo_numeric: dict[str, Any]
    ) -> None:
        """A live ``STONK`` row: the second Manifold type that looks BINARY to the array
        discriminator. It publishes ``probability`` null today, so it renders ``-`` under any design
        — but it is the standing reason the price read is an allow-list rather than a
        ``PSEUDO_NUMERIC`` deny-list, since a deny-list would price it the day that changes."""
        payload = manifold_pseudo_numeric["search_stonk"]
        assert payload[0]["outcomeType"] == "STONK"
        assert "answers" not in payload[0]

        rows = venues.parse_manifold_matches(payload, width=60)

        assert rows is not None
        assert len(rows) == 1
        assert rows[0].implied_prob_yes is None
        assert rows[0].scalar_estimate is None

    def test_a_binary_row_still_prices_off_its_probability(self, manifold_multi_outcome: dict[str, Any]) -> None:
        """The regression guard on the 688-in-1000 case. The allow-list narrows which rows carry a
        price; it must not change the price a BINARY row carries, or the whole venue goes dark."""
        payload = manifold_multi_outcome["search_all"]
        rows = venues.parse_manifold_matches(payload, width=60)

        assert rows is not None
        binary = [
            (row, raw)
            for row, raw in zip(rows, payload, strict=False)
            if raw["outcomeType"] == venues.MANIFOLD_PRICED_OUTCOME_TYPE
        ]
        assert len(binary) == 2, "fixture must carry BINARY rows, or this proves nothing"
        for row, raw in binary:
            assert row.implied_prob_yes == raw["probability"]
            assert row.scalar_estimate is None

    def test_a_market_level_number_this_parser_cannot_read_is_declined_loudly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The allow-list's visible-failure half, on the shape no live type has yet.

        This is what makes the allow-list the safe default: a future scalar type loses its price
        with a line in the run log, where a deny-list would have printed its scale position under
        the `prob` header and said nothing. No shape in the fixtures reaches this branch.
        """
        payload = [{"id": "future1", "question": "Q", "outcomeType": "SOMETHING_NEW", "probability": 0.7}]

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.market_retrieval.venues.manifold"):
            rows = venues.parse_manifold_matches(payload, width=60)

        assert rows is not None
        assert rows[0].implied_prob_yes is None
        assert rows[0].scalar_estimate is None
        assert "market-level probability" in caplog.text
        assert "SOMETHING_NEW" in caplog.text

    def test_the_live_payloads_reach_that_warning_on_no_row(
        self,
        manifold_pseudo_numeric: dict[str, Any],
        manifold_multi_outcome: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The warning above is a signal, so it has to be silent on every shape that exists: a
        PSEUDO_NUMERIC row has a value to show, and every other non-BINARY type ships `probability`
        null. One line per Manifold row would make the whole marker unreadable."""
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.market_retrieval.venues.manifold"):
            for key, fixture in (
                ("search_pseudo_numeric", manifold_pseudo_numeric),
                ("search_log_scale", manifold_pseudo_numeric),
                ("search_stonk", manifold_pseudo_numeric),
                ("search_all", manifold_multi_outcome),
            ):
                assert venues.parse_manifold_matches(fixture[key], width=60) is not None

        assert caplog.text == ""

    @pytest.mark.parametrize("market", [{}, {"value": None}, {"value": "not a number"}, {"value": float("nan")}])
    def test_an_unreadable_value_yields_no_estimate_rather_than_a_blank_one(self, market: dict[str, Any]) -> None:
        """No value means no price. A `ScalarEstimate` holding None, or NaN, would render `value nan`
        in a column a forecaster is told to anchor on — `safe_float` is what keeps the field finite."""
        assert venues.manifold_scalar_estimate(market) is None

    def test_an_estimate_keeps_its_value_when_the_venue_omits_the_bounds(self) -> None:
        """The bounds are the venue's to omit and the value stands without them, so a missing scale
        costs the scale note and not the number."""
        estimate = venues.manifold_scalar_estimate({"value": 42.5})

        assert estimate is not None
        assert estimate == ScalarEstimate(value=42.5, minimum=None, maximum=None, is_log_scale=False)
        assert estimate.bounds_text() == ""

    def test_negative_bounds_are_joined_with_to_rather_than_a_hyphen(self) -> None:
        """Manifold's scales are routinely negative (live: -15 to 2, -48 to 48, -4 to 4), and
        `-15--2` is unreadable. Asserted on the type because both the table cell and the ranker's
        candidate line render it from here."""
        estimate = venues.manifold_scalar_estimate({"value": -1, "min": -15, "max": 2})

        assert estimate is not None
        assert estimate.bounds_text() == "-15 to 2"

    def test_wide_bounds_stay_exact_and_grouped_rather_than_becoming_an_exponent(self) -> None:
        """A bound is a parameter the market's author chose, so it is printed exact and grouped.

        Both halves matter on the scales that exist. Rounding would misstate the scale — a live `max`
        of 20,000,000,000 needs 11 significant digits — and `%g`'s automatic exponent would print one
        bound of the same scale as `1e+07` while its sibling reads `1`, which is one scale written two
        ways.
        """
        estimate = venues.manifold_scalar_estimate({"value": 609, "min": 1, "max": 10_000_000})

        assert estimate is not None
        assert estimate.bounds_text() == "1 to 10,000,000"


class TestScalarCoercions:
    """The coercions every venue parser sits on, at the boundary where they are enforced."""

    def test_parse_iso_returns_an_aware_datetime_for_both_shapes(self) -> None:
        """The `Z` form and a bare timestamp must come back mutually comparable, or a `max()`
        over an event's nested closes raises and takes the whole snapshot down."""
        aware = parse_iso("2026-11-01T00:00:00Z")
        naive_input = parse_iso("2026-11-01T00:00:00")

        assert aware is not None
        assert naive_input is not None
        assert aware.tzinfo is not None
        assert naive_input.tzinfo is not None
        assert naive_input == aware, "a naive value is TREATED as UTC, not shifted"
        assert max([aware, naive_input]) == aware

    def test_parse_iso_does_not_shift_the_rendered_wall_clock(self) -> None:
        """Attaching a tzinfo must leave every `%Y-%m-%d` render byte-identical — that is what
        makes the normalization safe to do at the boundary rather than per comparison site."""
        parsed = parse_iso("2026-11-01T23:30:00")

        assert parsed is not None
        assert parsed.strftime("%Y-%m-%d") == "2026-11-01"

    def test_a_date_only_string_is_also_aware(self) -> None:
        """The shape `parse_iso_guarded` falls back to, which is how PredictIt rows reach the
        pool naive today."""
        parsed = parse_iso("2026-11-01")

        assert parsed is not None
        assert parsed.tzinfo is not None

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_safe_float_rejects_non_finite_values(self, literal: str) -> None:
        """`json.loads` accepts these bare literals, and NaN defeats every comparison in
        `_liquidity_label` — so a row whose volume arrived as NaN would fall through to the
        strongest label and render `signal=deep`, presenting missing data to a forecaster as the
        best possible liquidity evidence. None renders `no-liquidity-data` instead."""
        value = json.loads(literal)

        assert safe_float(value) is None
        assert _liquidity_label(_row_with_volume(value)) == "no-liquidity-data"

    def test_safe_float_keeps_a_real_zero(self) -> None:
        """The distinction the liquidity signal rests on: absent is None, a genuinely
        zero-volume brand-new market is 0.0 and renders `thin`."""
        assert safe_float(0) == 0.0
        assert safe_float("0.0000") == 0.0


def _row_with_volume(volume: Any) -> MarketMatch:
    return MarketMatch(
        platform="kalshi",
        market_title="M",
        market_url="https://kalshi.com/markets/M",
        implied_prob_yes=None,
        bid=None,
        ask=None,
        spread=None,
        volume_24h=None,
        close_time=None,
        is_resolved=False,
        match_confidence=1.0,
        raw_rules="",
        total_volume=safe_float(volume),
        open_interest=None,
    )


class TestPredictIt:
    @pytest.mark.parametrize("sentinel", ["NA", "N/A", None, "", "not-a-date", "2026-11", "20261103"])
    def test_a_value_with_no_calendar_date_prefix_yields_no_close_time(self, sentinel: Any) -> None:
        """The prefix guard is what keeps PredictIt's ``NA``/``N/A`` no-close sentinels — and any
        other non-date the dump might ship — from reaching the parser at all."""
        assert parse_iso_guarded(sentinel) is None

    def test_a_seven_digit_fraction_keeps_its_FULL_timestamp(self) -> None:
        """``fromisoformat`` has accepted arbitrary fraction lengths since 3.11, so this shape
        parses whole rather than degrading to the date — asserting only ``%Y-%m-%d`` here would
        hold for BOTH outcomes and so pin neither.

        Pinned because the fallback is lossy on purpose: if a future guard tightened the tail and
        pushed this shape down to the date rung, `close_time` on every fractional PredictIt row
        would silently jump backwards by up to a day.
        """
        parsed = parse_iso_guarded("2026-11-03T23:59:59.1234567")

        assert parsed == datetime(2026, 11, 3, 23, 59, 59, 123456, tzinfo=UTC)

    @pytest.mark.parametrize(
        "unparseable_tail",
        [
            "2026-11-03T23:59:59 EST",  # a named zone, which fromisoformat has never accepted
            "2026-11-03T25:00:00",  # hour out of range
            "2026-11-03T23:59:60",  # leap second
            "2026-11-03T23:59:59.",  # a truncated fraction
        ],
    )
    def test_a_valid_date_with_an_unparseable_tail_degrades_to_the_date(self, unparseable_tail: str) -> None:
        """The rung the guard exists for, and the only one that can actually fire: a real
        calendar date whose TAIL `fromisoformat` refuses. Truncating to the date keeps the row's
        close window — all the ranker reads — where returning None would render `closes:` blank
        and losing the guard would raise inside `to_thread` and zero all four venues."""
        parsed = parse_iso_guarded(unparseable_tail)

        assert parsed == datetime(2026, 11, 3, tzinfo=UTC)

    def test_a_plain_timestamp_keeps_its_time_of_day(self) -> None:
        assert parse_iso_guarded("2026-11-03T23:59:59") == datetime(2026, 11, 3, 23, 59, 59, tzinfo=UTC)

    def test_the_row_carries_the_parsed_close_time(self, captured_payloads: dict[str, Any]) -> None:
        market = copy.deepcopy(captured_payloads["predictit_all"]["markets"][0])
        market["contracts"][0]["dateEnd"] = "2026-12-31T23:59:59.9999999"

        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.close_time is not None
        assert match.close_time.strftime("%Y-%m-%d") == "2026-12-31"

    def test_each_contract_becomes_a_child_with_its_own_last_trade_price(
        self, captured_payloads: dict[str, Any]
    ) -> None:
        """Where a PredictIt ballot's prices finally reach a forecaster.

        The market level has always (correctly) refused to quote one arbitrary contract as the
        market's — the whole ~197-market universe reaches the pool, so there is no per-question query
        to pick one with — which until the sub-rows landed meant every PredictIt row rendered
        priceless. `lastTradePrice` rather than `bestBuyYesCost`: the last trade is a price somebody
        paid, while the best ask on a thin contract can sit far from it.
        """
        market = captured_payloads["predictit_all"]["markets"][0]

        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.implied_prob_yes is None, "a ballot has no single probability"
        assert [(child.title, child.implied_prob_yes) for child in match.children] == [
            (contract["name"], pytest.approx(contract["lastTradePrice"])) for contract in market["contracts"]
        ]

    def test_each_contract_child_keeps_its_yes_bid_ask_book(self, captured_payloads: dict[str, Any]) -> None:
        market = captured_payloads["predictit_all"]["markets"][0]

        children = venues.predictit_contract_children(market["contracts"])

        assert [(child.quote_low, child.quote_high) for child in children] == [
            (safe_float(contract.get("bestSellYesCost")), safe_float(contract.get("bestBuyYesCost")))
            for contract in market["contracts"]
        ]

    def test_the_contract_children_keep_ballot_order(self, captured_payloads: dict[str, Any]) -> None:
        """Unsorted, unlike the real-money venues: PredictIt's dump carries no per-contract volume to
        rank by — the same absence that makes every PredictIt row read `no-liquidity-data` — and its
        own order is the meaningful one, since seat-count rungs and candidate lists arrive ordered."""
        market = copy.deepcopy(captured_payloads["predictit_all"]["markets"][0])
        market["contracts"] = [
            {"name": "192 or fewer", "lastTradePrice": 0.21, "status": "Open"},
            {"name": "193 to 197", "lastTradePrice": 0.09, "status": "Open"},
            {"name": "198 to 202", "lastTradePrice": 0.40, "status": "Open"},
        ]

        children = venues.predictit_contract_children(market["contracts"])

        assert [child.title for child in children] == ["192 or fewer", "193 to 197", "198 to 202"]

    def test_a_settled_contract_is_marked_resolved_per_contract(self, captured_payloads: dict[str, Any]) -> None:
        """Same rule as the market-level derivation, one level down: a missing status is not evidence
        of settlement, so it reads open."""
        market = copy.deepcopy(captured_payloads["predictit_all"]["markets"][0])
        market["contracts"] = [
            {"name": "Settled", "lastTradePrice": 1.0, "status": "Closed"},
            {"name": "Trading", "lastTradePrice": 0.4, "status": "Open"},
            {"name": "Unstated", "lastTradePrice": 0.4},
        ]

        children = venues.predictit_contract_children(market["contracts"])

        assert [(child.title, child.is_resolved) for child in children] == [
            ("Settled", True),
            ("Trading", False),
            ("Unstated", False),
        ]

    def test_a_contract_child_parses_its_own_close_date_through_the_guard(self) -> None:
        """Each contract carries its own `dateEnd`, and PredictIt's `NA` sentinel is the common case
        (the committed dump ships it on every contract), so the guard has to run per child too."""
        contracts = [
            {"name": "Dated", "lastTradePrice": 0.5, "dateEnd": "2026-11-03T23:59:59"},
            {"name": "Sentinel", "lastTradePrice": 0.5, "dateEnd": "NA"},
        ]

        children = venues.predictit_contract_children(contracts)

        assert children[0].close_time == datetime(2026, 11, 3, 23, 59, 59, tzinfo=UTC)
        assert children[1].close_time is None

    def test_contract_names_become_the_rules_text(self, captured_payloads: dict[str, Any]) -> None:
        """A PredictIt market's own title is often just "Which party will win ...", so the
        contract names are most of its semantic content."""
        market = copy.deepcopy(captured_payloads["predictit_all"]["markets"][0])
        market["contracts"] = [{"name": f"Candidate {i}"} for i in range(12)]

        names = venues.predictit_contract_names(market)
        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert names.count(",") == venues.PREDICTIT_CONTRACTS_RENDERED - 1
        assert match is not None
        assert match.raw_rules == f"contracts: {names}"

    def test_no_price_is_reported(self, captured_payloads: dict[str, Any]) -> None:
        """The whole universe reaches the ranker, so there is no query to select a contract
        with, and pricing an arbitrary one would misreport the market."""
        market = copy.deepcopy(captured_payloads["predictit_all"]["markets"][0])

        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert (match.implied_prob_yes, match.bid, match.ask, match.spread) == (None, None, None, None)
        assert match.venue_market_id == str(market["id"])

    def test_a_nameless_market_yields_no_row(self) -> None:
        assert venues.predictit_market_match({"id": 7}, match_confidence=1.0, channel="x") is None

    def test_a_market_whose_every_contract_is_closed_is_resolved(self) -> None:
        market = {"id": 1, "name": "M", "contracts": [{"name": "c", "status": "Closed"}]}

        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.is_resolved is True

    @pytest.mark.parametrize("order", [("Closed", "Open"), ("Open", "Closed")])
    def test_one_open_contract_keeps_the_market_live_in_either_order(self, order: tuple[str, str]) -> None:
        """The `nested[0]` bug this file already fixes for Kalshi. Reading the first contract made
        the verdict depend on the order untrusted external JSON arrived in — the SAME market read
        RESOLVED one way and open the other — and the ranker prompt tells the model a RESOLVED
        price is a realized outcome rather than a forecast, so it deprioritizes a live market."""
        market = {
            "id": 1,
            "name": "M",
            "contracts": [
                {"name": "a", "status": order[0], "dateEnd": "2026-07-01T00:00:00"},
                {"name": "b", "status": order[1], "dateEnd": "2026-12-31T00:00:00"},
            ],
        }

        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.is_resolved is False
        assert match.close_time is not None
        assert match.close_time.strftime("%Y-%m-%d") == "2026-12-31", "close_time is the max over contracts"

    def test_a_contracts_block_holding_no_usable_entry_is_not_resolved(self) -> None:
        """The empty-sequence trap: `all()` over nothing is True, so folding the dict filter
        inside the conjunction would report a market with no readable contract as SETTLED."""
        market = {"id": 1, "name": "M", "contracts": ["nonsense", 7, None]}

        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.is_resolved is False
        assert match.close_time is None

    def test_a_contract_with_no_status_reads_as_open(self) -> None:
        """A missing field is not evidence of settlement."""
        market = {"id": 1, "name": "M", "contracts": [{"name": "c"}]}

        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.is_resolved is False

    @pytest.mark.asyncio
    async def test_prefetch_keeps_the_none_versus_empty_contract(self) -> None:
        lost = FakeSession({venues.PREDICTIT_URL: [FakeResponse(503, text="down"), FakeResponse(503, text="down")]})
        assert await venues.predictit_prefetch(lost) is None

        empty = FakeSession({venues.PREDICTIT_URL: FakeResponse(200, payload={"markets": []})})
        assert await venues.predictit_prefetch(empty) == [], "a genuinely empty dump is Signal C's business"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("shapeless_body", [{"nope": []}, {"markets": "not a list"}, ["not", "an", "object"]])
    async def test_a_wrong_top_level_shape_is_a_lost_dump_not_an_empty_universe(self, shapeless_body: Any) -> None:
        """The caller CACHES a successful dump for 6h, which is what makes ``[]`` here expensive.

        A 200 whose top-level shape changed parsed to an empty universe that
        ``_predictit_universe`` then pinned as healthy for the TTL, so every later question in the
        run read the venue back as fine with no HTTP and no counter bump. Signal C does fire on
        the first question's empty observation, but the cache outlives the finding. ``None`` puts
        it on the fetch-failed path instead: no cache write, one source loss.
        """
        session = FakeSession({venues.PREDICTIT_URL: FakeResponse(200, payload=shapeless_body)})

        assert await venues.predictit_prefetch(session) is None

    @pytest.mark.asyncio
    async def test_a_malformed_MARKET_is_filtered_without_losing_the_dump(self) -> None:
        session = FakeSession(
            {venues.PREDICTIT_URL: FakeResponse(200, payload={"markets": ["nonsense", {"id": 1, "name": "M"}]})}
        )

        assert await venues.predictit_prefetch(session) == [{"id": 1, "name": "M"}]


class TestVenueRetryBudget:
    """Each buffered venue gets a SECOND attempt, pinned on a literal 2.

    The sibling assertion in the provider suite compares the call count to
    ``*_MAX_ATTEMPTS`` itself, which is tautological: dropping all three constants to 1
    leaves it — and the whole suite — green while silently retiring every venue retry. A
    transient blip on a single query then costs that query's rows outright.
    """

    @pytest.mark.parametrize(
        ("url", "fetch", "attempts", "empty_body"),
        [
            (
                venues.MANIFOLD_SEARCH_URL,
                lambda session: venues.manifold_search(session, "q", width=60),
                venues.MANIFOLD_MAX_ATTEMPTS,
                [],
            ),
            (
                venues.POLYMARKET_SEARCH_URL,
                lambda session: venues.polymarket_search(session, "q", width=60),
                venues.POLYMARKET_MAX_ATTEMPTS,
                {"events": [], "markets": []},
            ),
            (venues.PREDICTIT_URL, venues.predictit_prefetch, venues.PREDICTIT_MAX_ATTEMPTS, {"markets": []}),
        ],
    )
    @pytest.mark.asyncio
    async def test_a_transient_first_attempt_is_retried_and_the_second_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, url: str, fetch: Any, attempts: int, empty_body: Any
    ) -> None:
        monkeypatch.setattr("metaculus_bot.research.market_retrieval.http.HTTP_RETRY_BACKOFF_SECS", 0.0)
        session = FakeSession({url: [FakeResponse(503, text="down"), FakeResponse(200, payload=empty_body)]})

        assert await fetch(session) == [], "the second attempt's body must reach the parser"
        assert session._call_counts[url] == 2, "a transient first attempt must be retried"
        assert attempts >= 2, "the constant must permit the retry this test just took"


class TestBufferedBodyCap:
    """``MAX_RESPONSE_BYTES``, driven through the three venues that buffer a whole body.

    Previously unreachable from any test: ``read_json_capped`` chose between the capped stream
    and a bare ``resp.json()`` on whether the response exposed ``.read``, and the fake did not,
    so every stubbed venue took the UNCAPPED branch — production's memory ceiling was decided
    by what a test double implemented, and neutering the cap left the suite green.
    """

    @pytest.mark.parametrize(
        ("url", "fetch"),
        [
            (venues.MANIFOLD_SEARCH_URL, lambda session: venues.manifold_search(session, "q", width=60)),
            (venues.POLYMARKET_SEARCH_URL, lambda session: venues.polymarket_search(session, "q", width=60)),
            (venues.PREDICTIT_URL, venues.predictit_prefetch),
        ],
    )
    @pytest.mark.asyncio
    async def test_an_oversized_body_reads_as_a_failed_fetch_not_an_empty_one(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, url: str, fetch: Any
    ) -> None:
        """Every buffered venue sits on the one cap, and a breach must return ``None``: as ``[]``
        it would publish as a benign ``none`` token and read as a venue whose index found
        nothing. The 10 MiB ceiling is impractical to materialize, so it is patched down."""
        monkeypatch.setattr("metaculus_bot.research.market_retrieval.http.MAX_RESPONSE_BYTES", 64)
        session = FakeSession({url: FakeResponse(200, payload=[{"id": "m", "question": "x" * 500}])})

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.http_fetch"):
            assert await fetch(session) is None

        assert "response too large" in "\n".join(caplog.messages)

    @pytest.mark.asyncio
    async def test_a_body_under_the_cap_parses_whole(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The other half of the cap: it must not truncate a legitimate body. A capped read that
        returned its partial buffer instead of None would hand `json.loads` a half object and
        surface as an unexplained decode failure on the largest real payloads."""
        payload = [{"id": f"m{index}", "question": f"Q{index}", "probability": 0.5} for index in range(40)]
        monkeypatch.setattr(
            "metaculus_bot.research.market_retrieval.http.MAX_RESPONSE_BYTES", len(json.dumps(payload).encode())
        )
        session = FakeSession({venues.MANIFOLD_SEARCH_URL: FakeResponse(200, payload=payload)})

        rows = await venues.manifold_search(session, "q", width=60)

        assert rows is not None
        assert [row.venue_market_id for row in rows] == [str(entry["id"]) for entry in payload]


def _manifold_row(title: str) -> MarketMatch:
    return MarketMatch(
        platform="manifold",
        market_title=title,
        market_url="",
        implied_prob_yes=None,
        bid=None,
        ask=None,
        spread=None,
        volume_24h=None,
        close_time=None,
        is_resolved=False,
        match_confidence=1.0,
        raw_rules="",
    )


class TestPerQueryFailureIsolation:
    """``None`` is an upstream failure, ``[]`` a search that parsed to nothing. Collapsing the
    two publishes a total outage as a benign ``none``."""

    def test_a_lost_query_is_tallied_without_losing_its_siblings(self) -> None:
        results = [[_manifold_row("a")], None, [_manifold_row("b")], ValueError("boom")]
        rows, tally = flatten_results(results, "manifold")

        assert [row.market_title for row in rows] == ["a", "b"]
        assert (tally.ok, tally.failed) == (2, 2)

    def test_an_all_empty_fan_out_is_a_clean_zero(self) -> None:
        rows, tally = flatten_results([[], []], "manifold")

        assert rows == []
        assert (tally.ok, tally.failed) == (2, 0)
