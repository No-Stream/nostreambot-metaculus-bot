"""Market metadata survives venue adapters and reaches the rendered snapshot."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from metaculus_bot.research.market_retrieval.ranking import RENDER_BUDGET
from metaculus_bot.research.market_retrieval.rendering import (
    CHILD_TITLE_MAX_CHARS,
    MAX_CHILD_ROWS_PER_MARKET,
    RAW_BULLET_BODY_MAX_CHARS,
    TITLE_MAX_CHARS,
    WHY_CHARS,
    render_snapshot,
)
from metaculus_bot.research.market_retrieval.types import MarketChild, MarketSnapshot
from metaculus_bot.research.market_retrieval.venues import (
    KALSHI_EVENTS_URL,
    KALSHI_MARKET_FIELDS,
    kalshi_event_match,
    kalshi_prefetch_events,
    kalshi_usd_liquidity,
    parse_manifold_matches,
    parse_polymarket_matches,
)
from tests.market_retrieval_fakes import FakeResponse, FakeSession
from tests.test_market_retrieval_generation import Platform
from tests.test_market_retrieval_rendering import (
    _BUDGET_FORECAST_TIME,
    _CAP_NOTE,
    MARKET_SNAPSHOT_MAXED_RENDER_CHAR_BUDGET,
    _row,
    _table_rows,
)

_PAYLOADS_PATH = Path(__file__).parent / "data" / "prediction_market_venue_payloads.json"
_KALSHI_FAMILY_PATH = Path(__file__).parent / "data" / "kalshi_family_liquidity_2026_08_05.json"


@pytest.fixture(scope="module")
def venue_payloads() -> dict[str, Any]:
    return json.loads(_PAYLOADS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def kalshi_family_page() -> dict[str, Any]:
    return json.loads(_KALSHI_FAMILY_PATH.read_text(encoding="utf-8"))


def _rendered_rows(snapshot: MarketSnapshot) -> list[dict[str, str]]:
    return _table_rows(render_snapshot(snapshot))


def _metadata_fields(cell: str) -> dict[str, str]:
    """Parse a rendered metadata cell (`label value label value`) into label -> value."""
    body = cell.split(": ", 1)[1] if ": " in cell else cell
    tokens = body.split(" ")
    return dict(zip(tokens[::2], tokens[1::2], strict=True))


class TestPolymarketMetadata:
    def test_single_market_metadata_reaches_the_parent_row(self, venue_payloads: dict[str, Any]) -> None:
        """A single-market event uses its market's volume, liquidity, price moves, and update time."""
        event = next(event for event in venue_payloads["polymarket_search"]["events"] if len(event["markets"]) == 1)
        market = event["markets"][0]

        rows = parse_polymarket_matches({"events": [event], "markets": []}, width=1)

        assert rows is not None
        match = rows[0]
        assert match.volume_24h == pytest.approx(float(market["volume24hr"]))
        assert match.liquidity == pytest.approx(float(market["liquidityNum"]))
        assert match.last_update_time == datetime.fromisoformat(market["updatedAt"].replace("Z", "+00:00"))
        assert match.price_change_1d == pytest.approx(float(market["oneDayPriceChange"]))
        assert match.price_change_1w == pytest.approx(float(market["oneWeekPriceChange"]))

        forecast_time = datetime(2026, 8, 4, tzinfo=UTC)
        rendered = _rendered_rows(MarketSnapshot(matches=[match], forecast_time=forecast_time))[0]
        metadata = _metadata_fields(rendered["metadata"])

        assert set(metadata) == {"v24", "liq", "upd", "1d", "1w"}
        assert metadata["1d"].startswith("+")
        assert metadata["1w"].startswith("-")

    def test_multi_market_metadata_is_kept_on_each_child(self, venue_payloads: dict[str, Any]) -> None:
        """Each child keeps its own figures rather than inheriting the first outcome's metadata."""
        event = next(event for event in venue_payloads["polymarket_search"]["events"] if len(event["markets"]) > 1)

        rows = parse_polymarket_matches({"events": [event], "markets": []}, width=1)

        assert rows is not None
        match = rows[0]
        assert len(match.children) == len(event["markets"])
        for child, market in zip(match.children, event["markets"], strict=True):
            assert child.volume_24h == pytest.approx(float(market["volume24hr"]))
            assert child.liquidity == pytest.approx(float(market["liquidityNum"]))
            assert child.last_update_time == datetime.fromisoformat(market["updatedAt"].replace("Z", "+00:00"))
            expected_one_day_change = market.get("oneDayPriceChange")
            expected_one_week_change = market.get("oneWeekPriceChange")
            assert child.price_change_1d == (
                pytest.approx(float(expected_one_day_change)) if expected_one_day_change is not None else None
            )
            assert child.price_change_1w == (
                pytest.approx(float(expected_one_week_change)) if expected_one_week_change is not None else None
            )

        forecast_time = datetime(2026, 8, 4, tzinfo=UTC)
        rendered_children = [
            row
            for row in _rendered_rows(MarketSnapshot(matches=[match], forecast_time=forecast_time))
            if row["platform"] == "↳"
        ]
        assert len(rendered_children) == len(event["markets"])
        for child_row in rendered_children:
            metadata = _metadata_fields(child_row["metadata"])
            assert {"v24", "liq", "upd"} <= set(metadata)
            assert metadata["upd"].endswith(("m", "h", "d"))
            assert all(metadata[label].startswith(("+", "-")) for label in ("1d", "1w") if label in metadata)

    def test_non_finite_market_metadata_renders_as_missing(self, venue_payloads: dict[str, Any]) -> None:
        event = copy.deepcopy(
            next(event for event in venue_payloads["polymarket_search"]["events"] if len(event["markets"]) == 1)
        )
        market = event["markets"][0]
        market["volume24hr"] = float("nan")
        market["liquidityNum"] = float("nan")
        market["liquidity"] = float("nan")
        market["oneDayPriceChange"] = float("nan")
        market["oneWeekPriceChange"] = float("nan")

        rows = parse_polymarket_matches({"events": [event], "markets": []}, width=1)

        assert rows is not None
        match = rows[0]
        assert match.volume_24h is None
        assert match.liquidity is None
        assert match.price_change_1d is None
        assert match.price_change_1w is None
        metadata = _rendered_rows(MarketSnapshot(matches=[match]))[0]["metadata"]

        assert metadata == "-"


class TestKalshiMetadata:
    @pytest.mark.asyncio
    async def test_catalogue_projection_and_children_keep_per_strike_metadata(
        self, kalshi_family_page: dict[str, Any]
    ) -> None:
        """The streaming projection retains fields used by both a market row and strike sub-rows."""
        raw_event = next(
            event for event in kalshi_family_page["events"] if event["event_ticker"] == "KXGOVWINS-27JAN01"
        )
        session = FakeSession({KALSHI_EVENTS_URL: FakeResponse(200, payload=kalshi_family_page)})

        pull = await kalshi_prefetch_events(session)

        assert pull.complete is True
        event = next(event for event in pull.events if event["event_ticker"] == raw_event["event_ticker"])
        assert {"volume_24h_fp", "updated_time"} <= set(KALSHI_MARKET_FIELDS)
        for projected_market, raw_market in zip(event["markets"], raw_event["markets"], strict=True):
            assert projected_market["volume_24h_fp"] == raw_market["volume_24h_fp"]
            assert projected_market["updated_time"] == raw_market["updated_time"]

        match = kalshi_event_match(event, match_confidence=100.0, channel="venue_search")

        assert match is not None
        assert len(match.children) == len(raw_event["markets"])
        for child, raw_market in zip(match.children, raw_event["markets"], strict=True):
            expected_volume_24h, _ = kalshi_usd_liquidity({**raw_market, "volume_fp": raw_market["volume_24h_fp"]})
            assert child.volume_24h == pytest.approx(expected_volume_24h)
            assert child.last_update_time == datetime.fromisoformat(raw_market["updated_time"].replace("Z", "+00:00"))

        forecast_time = datetime(2026, 11, 1, tzinfo=UTC)
        rendered_children = [
            row
            for row in _rendered_rows(MarketSnapshot(matches=[match], forecast_time=forecast_time))
            if row["platform"] == "↳"
        ]
        assert len(rendered_children) == len(raw_event["markets"])
        for row in rendered_children:
            metadata = _metadata_fields(row["metadata"])
            assert "v24" in metadata
            assert "trd" not in metadata
            assert metadata["upd"].endswith("d")
            assert metadata["prev"].startswith(("+", "-"))
            assert "1w" not in metadata


class TestManifoldMetadata:
    def test_trade_and_update_ages_use_the_snapshot_forecast_time(self, venue_payloads: dict[str, Any]) -> None:
        """A later edit does not make a market look recently traded."""
        market = copy.deepcopy(venue_payloads["manifold_search"][0])
        market["lastBetTime"] = int(datetime(2026, 8, 2, 12, tzinfo=UTC).timestamp() * 1000)
        market["lastUpdatedTime"] = int(datetime(2026, 8, 4, 12, tzinfo=UTC).timestamp() * 1000)
        forecast_time = datetime(2026, 8, 5, 12, tzinfo=UTC)

        rows = parse_manifold_matches([market], width=1)

        assert rows is not None
        match = rows[0]
        assert match.num_bettors == int(market["uniqueBettorCount"])
        assert match.volume_24h == pytest.approx(float(market["volume24Hours"]))
        assert match.liquidity is None  # Manifold totalLiquidity is past subsidy, not depth
        assert match.last_trade_time == datetime(2026, 8, 2, 12, tzinfo=UTC)
        assert match.last_update_time == datetime(2026, 8, 4, 12, tzinfo=UTC)

        rendered = _rendered_rows(MarketSnapshot(matches=[match], forecast_time=forecast_time))[0]
        assert rendered["metadata"].startswith("MANA: ")
        metadata = _metadata_fields(rendered["metadata"])

        assert metadata == {"v24": "0", "bt": "7", "trd": "3d"}


# Each venue's parser fills a fixed subset of the metadata fields (venues/*.py), so the worst case is the
# largest REAL subset at maximal values, not every field at once, which no venue produces.
_MAXED_METADATA_BY_VENUE: dict[Platform, tuple[str, ...]] = {
    "kalshi": ("volume_24h", "last_update_time", "price_change_since_previous"),
    "polymarket": ("volume_24h", "liquidity", "last_update_time", "price_change_1d", "price_change_1w"),
    "manifold": (
        "volume_24h",
        "num_bettors",
        "last_trade_time",
        "last_update_time",
        "price_change_1d",
        "price_change_1w",
    ),
}


@pytest.mark.parametrize("platform", sorted(_MAXED_METADATA_BY_VENUE))
def test_populated_metadata_on_every_row_and_child_fits_the_existing_render_budget(platform: Platform) -> None:
    forecast_time = _BUDGET_FORECAST_TIME
    maxed_values = {
        "volume_24h": 123_456_789.0,
        "liquidity": 987_654_321.0,
        "num_bettors": 654_321,
        "last_trade_time": forecast_time - timedelta(days=60),
        "last_update_time": forecast_time - timedelta(days=1),
        "price_change_1d": 0.55,
        "price_change_since_previous": 0.55,
        "price_change_1w": -0.23,
    }
    metadata = {field: maxed_values[field] for field in _MAXED_METADATA_BY_VENUE[platform]}
    children = tuple(
        replace(
            MarketChild(
                title=f"{'C' * (CHILD_TITLE_MAX_CHARS - 2)}{index:02d}",
                implied_prob_yes=0.91 - 0.03 * index,
                total_volume=123_456_789.0,
                open_interest=98_765_432.0,
                close_time=datetime(2026, 12, 31, tzinfo=UTC),
            ),
            **metadata,
        )
        for index in range(MAX_CHILD_ROWS_PER_MARKET)
    )
    match = replace(
        _row(
            platform=platform,
            title="M" * TITLE_MAX_CHARS,
            tier="same_quantity_other_cut",
            why="W" * WHY_CHARS,
            rules="R" * RAW_BULLET_BODY_MAX_CHARS,
            url="https://kalshi.com/markets/" + "T" * 40,
            prob=None,
            volume=123_456_789.0,
            oi=98_765_432.0,
            close=datetime(2026, 12, 31, tzinfo=UTC),
            children=children,
            cap_note=_CAP_NOTE,
        ),
        **metadata,
    )
    matches = [match for _ in range(RENDER_BUDGET)]

    rendered = render_snapshot(MarketSnapshot(matches=matches, forecast_time=forecast_time))

    assert len(rendered) < MARKET_SNAPSHOT_MAXED_RENDER_CHAR_BUDGET


class TestPriceChangeFormatting:
    @pytest.mark.parametrize(
        ("change", "expected"),
        [
            (-1.11e-16, "+0"),  # Manifold reports float noise for an untraded answer
            (5.55e-17, "+0"),
            (0.0123, "+1.2"),
            (-0.55, "-55"),
            (0.004, "+0.4"),
        ],
    )
    def test_changes_render_as_points_rounded_to_a_tenth(self, change: float, expected: str) -> None:
        match = replace(_row("manifold"), price_change_1d=change, price_change_1w=change)

        metadata = _metadata_fields(_rendered_rows(MarketSnapshot(matches=[match], forecast_time=None))[0]["metadata"])

        assert metadata["1d"] == expected
        assert metadata["1w"] == expected
