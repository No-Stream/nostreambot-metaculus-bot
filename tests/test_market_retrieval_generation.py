"""Pool assembly: channel order, dedup, widths, degradation, and the enrichment hook.

Two of these tests exist because of specific ways the port could ship looking done while
measuring nothing:

- **The widths must actually be wide.** In the shipped provider the render cap doubles as the
  retrieval width (``top_k = max_matches_per_platform + 2``), so a ranked pipeline that
  inherited it would rank a 5-candidate pool. ``test_the_kalshi_channel_is_wide`` asserts the
  pool is far larger than the old caps against a catalogue that has the rows to fill it.
- **PredictIt's whole universe must reach the pool.** Its 197 markets come from one GET, so a
  width there would throw evidence away for nothing, and a re-added one would be invisible in
  every other test.

The remaining ones guard the ordering contract: channel order IS the ranking, and the pool's
head is the fail-open slate, so a channel that sorts differently or a dedup that keeps the
LAST occurrence silently changes what a degraded run publishes.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Literal

import pytest

from metaculus_bot.research.market_retrieval import generation
from metaculus_bot.research.market_retrieval.generation import (
    CHANNEL_SETTLEMENT_JOIN,
    CHANNEL_UNIVERSE_FUZZY,
    CHANNEL_VENUE_SEARCH,
    RETRIEVAL_WIDTH,
    VENUE_ORDER,
    assemble_pool,
    build_pool,
    enrich_manifold,
)
from metaculus_bot.research.market_retrieval.ranking import RULES_CHARS
from metaculus_bot.research.market_retrieval.types import MarketMatch, _FetchTally

Platform = Literal["polymarket", "kalshi", "manifold", "predictit"]

# The question text the settlement-join channel keys on: a real BLS host, so the registrable
# domain is bls.gov and the join is exercised end to end rather than stubbed.
BLS_CRITERIA = "Resolves per the release at https://data.bls.gov/timeseries/LNS14000000"


def _kalshi_event(
    ticker: str, *, title: str, settles: str | None = None, rules: str = "", closes: str = "2026-12-31T00:00:00Z"
) -> dict[str, Any]:
    """A projected Kalshi event, in the shape the streaming projection emits."""
    return {
        "event_ticker": ticker,
        "series_ticker": ticker.split("-")[0],
        "title": title,
        "sub_title": "",
        "settlement_sources": [{"name": "src", "url": settles}] if settles else [],
        "markets": [
            {
                "rules_primary": rules,
                "close_time": closes,
                "yes_bid_dollars": "0.40",
                "yes_ask_dollars": "0.45",
                "last_price_dollars": "0.42",
                "notional_value_dollars": "1.0",
                "volume_fp": "1000",
                "open_interest_fp": "500",
                "volume_24h_fp": "10",
                "status": "active",
            }
        ],
    }


def _predictit_market(market_id: int, name: str) -> dict[str, Any]:
    return {
        "id": market_id,
        "name": name,
        "url": f"https://www.predictit.org/markets/detail/{market_id}",
        "contracts": [{"name": "Yes", "status": "Open", "dateEnd": "2026-11-03T23:59:59.9999999"}],
    }


def _search_row(platform: Platform, market_id: str, title: str, rank: int = 0) -> MarketMatch:
    return MarketMatch(
        platform=platform,
        market_title=title,
        market_url=f"https://example.test/{market_id}",
        implied_prob_yes=0.5,
        bid=None,
        ask=None,
        spread=None,
        volume_24h=None,
        close_time=None,
        is_resolved=False,
        match_confidence=100.0 - rank,
        raw_rules="",
        venue_market_id=market_id,
        retrieval_channel=CHANNEL_VENUE_SEARCH,
    )


def _pool(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "criteria_text": "",
        "queries": ["unemployment rate"],
        "kalshi_events": [],
        "predictit_markets": [],
        "venue_search_results": {},
    }
    kwargs.update(overrides)
    return assemble_pool(**kwargs)


class TestChannelOrder:
    """Channel order is the ranking: settlement join, then venue index, then fuzzy fill."""

    def test_a_settlement_hit_precedes_a_fuzzy_fill_for_the_same_query_set(self) -> None:
        """The whole point of the join channel: it reaches a market whose title shares no
        vocabulary with the question, and a fuzzy scorer would rank that market last."""
        joined = _kalshi_event("JOINED-1", title="Nonsense words nobody queried", settles="https://data.bls.gov/x")
        fuzzy = _kalshi_event("FUZZY-1", title="US unemployment rate above 4%")

        result = _pool(criteria_text=BLS_CRITERIA, kalshi_events=[fuzzy, joined])

        tickers = [row.venue_market_id for row in result.candidates]
        assert tickers.index("JOINED-1") < tickers.index("FUZZY-1")
        assert result.candidates[0].retrieval_channel == CHANNEL_SETTLEMENT_JOIN
        assert result.settlement_domains == ("bls.gov",)

    def test_the_venue_index_channel_precedes_the_fuzzy_fill(self) -> None:
        """Both land in the pool; only the ORDER separates them, and that order is what the
        fail-open slate cuts on."""
        result = _pool(
            kalshi_events=[_kalshi_event("K-1", title="unemployment rate")],
            venue_search_results={"manifold": [[_search_row("manifold", "m1", "unemployment")]]},
        )

        channels = [row.retrieval_channel for row in result.candidates]
        assert set(channels) == {CHANNEL_VENUE_SEARCH, CHANNEL_UNIVERSE_FUZZY}
        assert result.channel_counts == {CHANNEL_VENUE_SEARCH: 1, CHANNEL_UNIVERSE_FUZZY: 1}

    def test_the_flat_pool_is_venue_major_in_venue_order(self) -> None:
        """The ranker prompt indexes into this list, so its order is the prompt's order."""
        result = _pool(
            kalshi_events=[_kalshi_event("K-1", title="a")],
            predictit_markets=[_predictit_market(1, "b")],
            venue_search_results={
                "polymarket": [[_search_row("polymarket", "p1", "c")]],
                "manifold": [[_search_row("manifold", "m1", "d")]],
            },
        )

        platforms = [row.platform for row in result.candidates]
        assert platforms == ["kalshi", "polymarket", "manifold", "predictit"]
        assert platforms == [venue for venue in VENUE_ORDER if venue in platforms]

    def test_the_settlement_channel_is_re_ranked_within_itself(self) -> None:
        """The join returns catalogue order, which carries no evidential signal — leaving it
        unranked would make the fail-open slate measure the alphabet."""
        weak = _kalshi_event("AAA-FIRST", title="Something unrelated entirely", settles="https://data.bls.gov/a")
        strong = _kalshi_event("ZZZ-LAST", title="US unemployment rate above 4%", settles="https://www.bls.gov/b")

        result = _pool(criteria_text=BLS_CRITERIA, kalshi_events=[weak, strong])

        assert [row.venue_market_id for row in result.candidates][:2] == ["ZZZ-LAST", "AAA-FIRST"]


class TestDedup:
    def test_dedup_keeps_the_first_occurrence(self) -> None:
        """First-seen wins, because first-seen is the higher-priority channel. Keeping the last
        occurrence would let a fuzzy fill overwrite a structural hit's channel and score."""
        event = _kalshi_event("DUPE-1", title="US unemployment rate", settles="https://data.bls.gov/x")

        result = _pool(criteria_text=BLS_CRITERIA, kalshi_events=[event])

        assert len(result.candidates) == 1
        assert result.candidates[0].retrieval_channel == CHANNEL_SETTLEMENT_JOIN
        assert result.candidates[0].match_confidence == 1.0

    def test_the_dedup_key_is_case_insensitive_on_the_venue_id(self) -> None:
        rows = [_search_row("manifold", "AbC", "one"), _search_row("manifold", "abc", "two")]

        result = _pool(venue_search_results={"manifold": [rows]})

        assert [row.market_title for row in result.candidates] == ["one"]

    def test_the_same_id_at_two_venues_is_two_candidates(self) -> None:
        rows = {
            "manifold": [[_search_row("manifold", "shared-id", "manifold row")]],
            "polymarket": [[_search_row("polymarket", "shared-id", "polymarket row")]],
        }

        result = _pool(venue_search_results=rows)

        assert len(result.candidates) == 2


class TestWidths:
    def test_the_kalshi_channel_is_wide(self) -> None:
        """The regression guard for the port's biggest silent-failure mode: the shipped
        provider's retrieval width is its RENDER cap plus two, so a ranked pipeline that
        inherited it would rank five candidates and measure nothing."""
        events = [_kalshi_event(f"K-{i}", title=f"US unemployment rate scenario {i}") for i in range(150)]

        result = _pool(kalshi_events=events)

        assert result.per_venue_counts["kalshi"] == RETRIEVAL_WIDTH["kalshi"] == 100
        assert result.per_venue_counts["kalshi"] > 20, "far wider than the old render-cap-derived width"

    def test_only_the_kept_width_of_kalshi_rows_is_ever_constructed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scoring is whole-catalogue; construction is not. Building a row for every event spent
        it on the ~9,662 the width discards — measured 0.162s / 6.7 MB against 0.028s / 1.3 MB over
        a 9,762-event catalogue — so the channel yields and the caller stops at the width."""
        events = [_kalshi_event(f"K-{i}", title=f"US unemployment rate scenario {i}") for i in range(400)]
        real_match = generation.venues.kalshi_event_match
        calls = 0

        def counting_match(event: dict[str, Any], **kwargs: Any) -> MarketMatch | None:
            nonlocal calls
            calls += 1
            return real_match(event, **kwargs)

        monkeypatch.setattr(generation.venues, "kalshi_event_match", counting_match)

        result = _pool(kalshi_events=events)

        assert result.per_venue_counts["kalshi"] == RETRIEVAL_WIDTH["kalshi"]
        assert calls <= RETRIEVAL_WIDTH["kalshi"] + 1, "one row past the width at most, not 400"

    def test_the_lazy_channel_still_scores_the_whole_catalogue(self) -> None:
        """The laziness must not become a truncated SCAN: the ordering is only global because
        every event is scored, so the best row has to win even when it sits last in the dump."""
        events = [_kalshi_event(f"K-{i}", title=f"Something unrelated {i}") for i in range(300)]
        events.append(_kalshi_event("BEST", title="US unemployment rate above 4%"))

        result = _pool(kalshi_events=events)

        assert result.candidates[0].venue_market_id == "BEST"

    def test_the_venue_search_widths_bound_the_union_across_queries(self) -> None:
        per_query = [[_search_row("manifold", f"m{i}", f"row {i}") for i in range(10)] for _ in range(10)]
        distinct = [[_search_row("manifold", f"q{q}-{i}", f"row {q}-{i}") for i in range(10)] for q in range(10)]

        assert _pool(venue_search_results={"manifold": per_query}).per_venue_counts["manifold"] == 10
        assert (
            _pool(venue_search_results={"manifold": distinct}).per_venue_counts["manifold"]
            == RETRIEVAL_WIDTH["manifold"]
        )

    def test_the_venue_search_width_cuts_on_venue_RANK_not_on_query_order(self) -> None:
        """Which query was issued first must not decide which rows survive the width.

        `flatten_results` concatenates the per-query lists in query order, so before the sort an
        early query's rank-9 rows sat ahead of a later query's rank-0 exact hit and took its slot.
        The width here is filled entirely by query one, so the second query's top hit is the row
        that gets evicted — the exact defect, and the one row a forecaster most wants to see.

        The first query starts at rank 1 so its rows never TIE the second query's rank-0 hit:
        equal ranks legitimately keep query precedence (the sibling test pins that), and a tie
        here would make the assertion depend on the tiebreak instead of on the rank ordering.
        """
        width = RETRIEVAL_WIDTH["manifold"]
        first_query = [_search_row("manifold", f"q1-{i}", f"weak row {i}", rank=i + 1) for i in range(width)]
        second_query = [_search_row("manifold", "exact", "the exact match", rank=0)]

        result = _pool(venue_search_results={"manifold": [first_query, second_query]})

        ids = [row.venue_market_id for row in result.candidates]
        assert result.per_venue_counts["manifold"] == width
        assert "exact" in ids, "a later query's rank-0 hit was truncated behind an earlier query's tail"
        assert ids[0] == "exact", "the highest venue rank leads the venue's block"
        assert ids[-1] != f"q1-{width - 1}", "the weakest row must be the one the width evicts"

    def test_equal_venue_ranks_keep_query_precedence(self) -> None:
        """The sort is STABLE, deliberately: the deterministic query set is ordered
        precision-descending, so among rows the venue ranked equally the earlier query's row is
        the better bet and that tiebreak is worth keeping."""
        first_query = [_search_row("manifold", "from-q1", "row", rank=0)]
        second_query = [_search_row("manifold", "from-q2", "row", rank=0)]

        result = _pool(venue_search_results={"manifold": [first_query, second_query]})

        assert [row.venue_market_id for row in result.candidates] == ["from-q1", "from-q2"]

    def test_the_settlement_join_channel_still_leads_the_higher_ranked_search_rows(self) -> None:
        """The re-order is WITHIN the venue-search channel: it must not promote a search row past
        a structural settlement hit, which is the one cross-channel ordering guarantee."""
        joined = _kalshi_event("JOINED", title="Nonsense nobody queried", settles="https://data.bls.gov/x")
        search = [_search_row("kalshi", "SEARCHED", "US unemployment rate", rank=0)]

        result = _pool(criteria_text=BLS_CRITERIA, kalshi_events=[joined], venue_search_results={"kalshi": [search]})

        assert [row.venue_market_id for row in result.candidates] == ["JOINED", "SEARCHED"]

    def test_predictit_has_no_width_and_the_whole_universe_reaches_the_pool(self) -> None:
        """197 markets from one GET: there is nothing to select, so a width would only discard
        evidence. Its absence from RETRIEVAL_WIDTH is the statement."""
        markets = [_predictit_market(i, f"Market {i}") for i in range(197)]

        result = _pool(predictit_markets=markets)

        assert "predictit" not in RETRIEVAL_WIDTH
        assert result.per_venue_counts["predictit"] == 197

    def test_predictit_rows_keep_the_dump_order(self) -> None:
        """No scorer on this channel, so the pool order is the dump order."""
        markets = [_predictit_market(i, f"Market {i}") for i in range(5)]

        result = _pool(predictit_markets=markets)

        assert [row.venue_market_id for row in result.candidates] == ["0", "1", "2", "3", "4"]


class TestAsOfEligibility:
    """The backtest leakage guard, applied where the width can still be filled.

    Unreachable on the provider path (it passes None, and the provider hard-disables under
    `is_benchmarking`), so this is the only guard the first real backtest or replay caller has.
    """

    def test_a_closed_row_frees_its_width_slot_instead_of_being_deleted_after_the_cut(self) -> None:
        """The whole point of filtering inside `add()`. As a post-hoc filter over the
        already-truncated pool, 150 events whose 120 highest-scoring rows close before `as_of`
        truncated to 100 and then filtered to ZERO — while 30 eligible candidates sat unused in the
        catalogue. It also zeroed `per_venue_counts`, which feeds `candidates_pre_filter` into
        provider health: exactly the shape that field exists to prevent alerting on.

        The closed rows score HIGHER here (they carry the query terms verbatim), so they win the
        fuzzy ordering and would occupy every slot of the width if eligibility were checked later.
        """
        width = RETRIEVAL_WIDTH["kalshi"]
        closed = [
            _kalshi_event(f"OLD-{i}", title=f"US unemployment rate scenario {i}", closes="2026-01-31T00:00:00Z")
            for i in range(width + 20)
        ]
        open_rows = [
            _kalshi_event(f"NEW-{i}", title=f"US unemployment rate later scenario {i}", closes="2026-12-31T00:00:00Z")
            for i in range(30)
        ]

        result = _pool(kalshi_events=[*closed, *open_rows], as_of=datetime(2026, 5, 1, tzinfo=UTC))

        assert result.per_venue_counts["kalshi"] == 30, "every eligible candidate must survive the width"
        assert {row.venue_market_id for row in result.candidates} == {f"NEW-{i}" for i in range(30)}
        assert result.channel_counts[CHANNEL_UNIVERSE_FUZZY] == 30, "counts describe the pool that exists"

    def test_a_naive_as_of_is_read_as_utc(self) -> None:
        """An explicit caller can still hand over a naive instant; it must not raise on comparison
        against the aware close times `parse_iso` produces."""
        events = [_kalshi_event("OLD", title="unemployment rate", closes="2026-01-31T00:00:00Z")]

        assert _pool(kalshi_events=events, as_of=datetime(2026, 5, 1)).per_venue_counts["kalshi"] == 0

    def test_no_as_of_keeps_every_row(self) -> None:
        """The provider path. None must be a true no-op, not a filter with a permissive default."""
        events = [_kalshi_event("OLD", title="unemployment rate", closes="2026-01-31T00:00:00Z")]

        assert _pool(kalshi_events=events).per_venue_counts["kalshi"] == 1


class TestDegradation:
    """`per_venue_tally` is the only degradation signal the pool carries — a venue is degraded
    iff its tally holds a failed sub-fetch, which is what the caller turns into a source token."""

    def test_a_venue_whose_search_failed_contributes_nothing_and_reports_degraded(self) -> None:
        """A lost sub-fetch and a genuine no-match both arrive as an empty list, so the
        distinction has to be carried around the pool rather than inferred from it."""
        result = _pool(venue_search_results={"manifold": [None], "polymarket": [[]]})

        assert result.per_venue_counts["manifold"] == 0
        assert result.per_venue_tally["manifold"] == _FetchTally(ok=0, failed=1)
        assert result.per_venue_tally["polymarket"] == _FetchTally(ok=1, failed=0)

    def test_a_partially_lost_fan_out_keeps_what_returned(self) -> None:
        rows = [_search_row("manifold", "m1", "kept")]

        result = _pool(venue_search_results={"manifold": [rows, None]})

        assert [row.venue_market_id for row in result.candidates] == ["m1"]
        assert result.per_venue_tally["manifold"] == _FetchTally(ok=1, failed=1)

    def test_an_empty_catalogue_yields_an_empty_pool_without_raising(self) -> None:
        result = _pool(criteria_text=BLS_CRITERIA)

        assert result.candidates == ()
        assert result.per_venue_counts == dict.fromkeys(VENUE_ORDER, 0)
        assert result.per_venue_tally == {}

    def test_no_question_urls_means_no_settlement_channel(self) -> None:
        event = _kalshi_event("K-1", title="unemployment", settles="https://data.bls.gov/x")

        result = _pool(criteria_text="No links here at all.", kalshi_events=[event])

        assert result.settlement_domains == ()
        assert result.candidates[0].retrieval_channel == CHANNEL_UNIVERSE_FUZZY


class TestBuildPoolOffloadsTheCpuWork:
    """The full-catalogue fuzzy scan is ~0.45s of BLOCKING CPU, and a pinned event loop shows
    up as forecaster soft-deadline drops attributed to the forecasters."""

    @pytest.mark.asyncio
    async def test_assembly_runs_in_exactly_one_to_thread_hop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[Any] = []
        real_to_thread = asyncio.to_thread

        async def spy(func: Any, /, *args: Any, **kwargs: Any) -> Any:
            calls.append(func)
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", spy)

        result = await build_pool(
            criteria_text=BLS_CRITERIA,
            queries=["unemployment"],
            kalshi_events=[_kalshi_event("K-1", title="US unemployment rate")],
            predictit_markets=[_predictit_market(1, "Market")],
            venue_search_results={},
        )

        assert calls == [assemble_pool], "one hop, not one per channel"
        assert len(result.candidates) == 2


class TestManifoldEnrichment:
    """The search listing carries no description, so without this every Manifold candidate
    reaches the ranker title-only — and the ranker's own "single most reliable cue" is the
    rules text."""

    @pytest.mark.asyncio
    async def test_the_description_becomes_the_rules_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        row = _search_row("manifold", "m1", "Will gas prices exceed $4?")

        async def fake_detail(_session: Any, market_id: str) -> dict[str, Any]:
            assert market_id == "m1"
            return {"textDescription": "Resolves per the AAA national average."}

        monkeypatch.setattr(generation.venues, "manifold_market_detail", fake_detail)

        outcome = await enrich_manifold([row], object())

        assert row.raw_rules == "Resolves per the AAA national average."
        assert (outcome.n_attempted, outcome.n_ok) == (1, 1)

    @pytest.mark.asyncio
    async def test_a_description_that_only_restates_the_title_is_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        row = _search_row("manifold", "m1", "Will gas prices exceed $4 by June?")

        async def fake_detail(_session: Any, _market_id: str) -> dict[str, Any]:
            return {"textDescription": "Will gas prices exceed $4"}

        monkeypatch.setattr(generation.venues, "manifold_market_detail", fake_detail)

        outcome = await enrich_manifold([row], object())

        assert row.raw_rules == "", "a prefix of the title is not rules text"
        assert outcome.n_ok == 1, "the GET still returned; only the text was uninformative"

    @pytest.mark.asyncio
    async def test_the_stored_text_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        row = _search_row("manifold", "m1", "Q")

        async def fake_detail(_session: Any, _market_id: str) -> dict[str, Any]:
            return {"textDescription": "x" * 5000}

        monkeypatch.setattr(generation.venues, "manifold_market_detail", fake_detail)

        await enrich_manifold([row], object())

        # The store cap IS the ranker's, not a second 300: this is where the text is stored, so a
        # store cap below the prompt cap silently caps the prompt too.
        assert len(row.raw_rules) == generation.MANIFOLD_DETAIL_RULES_CHARS == RULES_CHARS["manifold"]

    @pytest.mark.asyncio
    async def test_a_lost_detail_leaves_that_row_title_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Per-row soft-fail: a lost detail GET costs rules text, never recall, so it must not
        take the sibling rows or the snapshot down with it."""
        rows = [_search_row("manifold", "m1", "one"), _search_row("manifold", "m2", "two")]

        async def fake_detail(_session: Any, market_id: str) -> dict[str, Any] | None:
            return None if market_id == "m1" else {"textDescription": "real rules"}

        monkeypatch.setattr(generation.venues, "manifold_market_detail", fake_detail)

        outcome = await enrich_manifold(rows, object())

        assert rows[0].raw_rules == ""
        assert rows[1].raw_rules == "real rules"
        assert (outcome.n_attempted, outcome.n_ok) == (2, 1)

    @pytest.mark.asyncio
    async def test_a_raising_detail_does_not_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [_search_row("manifold", "m1", "one"), _search_row("manifold", "m2", "two")]

        async def fake_detail(_session: Any, market_id: str) -> dict[str, Any]:
            if market_id == "m1":
                raise RuntimeError("boom")
            return {"textDescription": "real rules"}

        monkeypatch.setattr(generation.venues, "manifold_market_detail", fake_detail)

        outcome = await enrich_manifold(rows, object())

        assert rows[1].raw_rules == "real rules"
        assert outcome.n_ok == 1

    @pytest.mark.asyncio
    async def test_the_wall_returns_whatever_completed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a wall on the WHOLE fan-out the worst case is 6 waves x 10s = 60s, which
        blows the snapshot's time budget on rules text."""
        rows = [_search_row("manifold", f"m{i}", f"row {i}") for i in range(4)]
        cancelled: list[str] = []

        async def fake_detail(_session: Any, market_id: str) -> dict[str, Any]:
            if market_id == "m0":
                return {"textDescription": "fast rules"}
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(market_id)
                raise
            return {"textDescription": "never arrives"}

        monkeypatch.setattr(generation.venues, "manifold_market_detail", fake_detail)

        outcome = await enrich_manifold(rows, object(), concurrency=4, wall_s=0.05)

        assert rows[0].raw_rules == "fast rules"
        assert [row.raw_rules for row in rows[1:]] == ["", "", ""]
        assert (outcome.n_attempted, outcome.n_ok) == (4, 1)
        assert set(cancelled) == {"m1", "m2", "m3"}, "the detail wall must cancel outstanding GET tasks"

    @pytest.mark.asyncio
    async def test_concurrency_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [_search_row("manifold", f"m{i}", f"row {i}") for i in range(20)]
        in_flight = 0
        peak = 0

        async def fake_detail(_session: Any, _market_id: str) -> dict[str, Any]:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1
            return {"textDescription": "rules"}

        monkeypatch.setattr(generation.venues, "manifold_market_detail", fake_detail)

        await enrich_manifold(rows, object(), concurrency=3)

        assert peak <= 3

    @pytest.mark.asyncio
    async def test_nothing_to_enrich_reports_no_attempt(self) -> None:
        """`none` rather than a loss: the caller must not report a lost source when there was
        no Manifold candidate to fetch a detail for."""
        outcome = await enrich_manifold([_search_row("kalshi", "K-1", "row")], object())

        assert (outcome.n_attempted, outcome.n_ok) == (0, 0)

    @pytest.mark.asyncio
    async def test_a_manifold_row_without_an_id_is_skipped(self) -> None:
        outcome = await enrich_manifold([_search_row("manifold", "", "row")], object())

        assert outcome.n_attempted == 0


class TestManifoldMultiOutcomeEnrichment:
    """A multi-outcome row's ANSWERS ride the same detail GET as its rules text.

    That coincidence is why lifting the ``contractType=BINARY`` recall ceiling cost no extra
    request: the search response publishes no probability and no answer array for these rows, so
    the fan-out that was already fetching descriptions is the only place their price can come
    from. Which also means the soft-fail matters more than it did — a lost GET now costs a row its
    price, not just its rules text.
    """

    @pytest.mark.asyncio
    async def test_the_answers_and_the_rules_text_come_from_one_get(self, monkeypatch: pytest.MonkeyPatch) -> None:
        row = _search_row("manifold", "m1", "How high will the US national average gas price get in 2026?")
        calls: list[str] = []

        async def fake_detail(_session: Any, market_id: str) -> dict[str, Any]:
            calls.append(market_id)
            return {
                "textDescription": "Resolves per the AAA national average.",
                "answers": [
                    {"text": "Over $4.60", "probability": 0.4992},
                    {"text": "Over $4.65", "probability": 0.3723},
                    {"text": "Over $4.70", "probability": 0.2765},
                    {"text": "Over $4.90", "probability": 0.1212},
                ],
            }

        monkeypatch.setattr(generation.venues, "manifold_market_detail", fake_detail)

        outcome = await enrich_manifold([row], object())

        assert calls == ["m1"], "one request must carry both fields"
        assert row.raw_rules == "Resolves per the AAA national average."
        assert row.top_answers == (("Over $4.60", 0.4992), ("Over $4.65", 0.3723), ("Over $4.70", 0.2765))
        assert outcome.n_ok == 1

    @pytest.mark.asyncio
    async def test_a_binary_row_gains_no_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The write is unconditional, so this pins that a BINARY detail leaves the field empty on
        its own evidence — no ``answers`` key — rather than needing a branch to say so."""
        row = _search_row("manifold", "m1", "Will gas prices exceed $4?")

        async def fake_detail(_session: Any, _market_id: str) -> dict[str, Any]:
            return {"textDescription": "Resolves per AAA.", "probability": 0.86}

        monkeypatch.setattr(generation.venues, "manifold_market_detail", fake_detail)

        await enrich_manifold([row], object())

        assert row.raw_rules == "Resolves per AAA."
        assert row.top_answers == ()

    @pytest.mark.asyncio
    async def test_a_lost_detail_leaves_a_multi_outcome_row_priceless_rather_than_wrong(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The per-row soft-fail, at the case it now costs the most: the row keeps its title,
        liquidity and close date, renders ``-`` for its price, and must not take its sibling down.
        A guessed stand-in price would be worse than the dash — nothing here has one to guess."""
        rows = [_search_row("manifold", "m1", "lost"), _search_row("manifold", "m2", "kept")]

        kept = {"answers": [{"text": "$3.80 - $4.19", "probability": 0.5083}]}

        async def fake_detail(_session: Any, market_id: str) -> dict[str, Any] | None:
            return None if market_id == "m1" else kept

        monkeypatch.setattr(generation.venues, "manifold_market_detail", fake_detail)

        outcome = await enrich_manifold(rows, object())

        assert (rows[0].raw_rules, rows[0].top_answers) == ("", ())
        assert rows[1].top_answers == (("$3.80 - $4.19", 0.5083),)
        assert (outcome.n_attempted, outcome.n_ok) == (2, 1)

    @pytest.mark.asyncio
    async def test_a_detail_carrying_answers_but_no_description_still_populates_the_price(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two fields are independent: 5 of 40 measured markets publish an empty description,
        and one of those must not lose its answers to a shared early return."""
        row = _search_row("manifold", "m1", "Price of Ethereum (ETH) by 2030?")

        async def fake_detail(_session: Any, _market_id: str) -> dict[str, Any]:
            return {"textDescription": "", "answers": [{"text": "9000-10000", "probability": 0.224}]}

        monkeypatch.setattr(generation.venues, "manifold_market_detail", fake_detail)

        await enrich_manifold([row], object())

        assert row.raw_rules == ""
        assert row.top_answers == (("9000-10000", 0.224),)
