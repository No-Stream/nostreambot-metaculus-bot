"""Kalshi price movement uses consecutive last trades, not the current book midpoint."""

import copy
import json
from datetime import datetime
from pathlib import Path

import pytest

from metaculus_bot.research.market_retrieval.venues import (
    KALSHI_EVENTS_URL,
    kalshi_event_match,
    kalshi_prefetch_events,
    parse_polymarket_matches,
)
from tests.market_retrieval_fakes import FakeResponse, FakeSession


@pytest.mark.asyncio
async def test_kalshi_trade_change_survives_catalogue_projection() -> None:
    path = Path(__file__).parent / "data" / "kalshi_family_liquidity_2026_08_05.json"
    page = json.loads(path.read_text(encoding="utf-8"))
    event = next(event for event in page["events"] if event["event_ticker"] == "KXGOVWINS-27JAN01")
    for market in event["markets"]:
        market["last_price_dollars"] = "0.67"
        market["previous_price_dollars"] = "0.41"
        market["yes_bid_dollars"] = "0.20"
        market["yes_ask_dollars"] = "0.40"
    pull = await kalshi_prefetch_events(FakeSession({KALSHI_EVENTS_URL: FakeResponse(200, payload=page)}))
    projected = next(event for event in pull.events if event["event_ticker"] == "KXGOVWINS-27JAN01")
    assert all(market["previous_price_dollars"] == "0.41" for market in projected["markets"])
    family = kalshi_event_match(projected, match_confidence=1.0, channel="venue_search")
    assert family is not None
    assert family.price_change_since_previous is None
    assert all(child.price_change_since_previous == pytest.approx(0.26) for child in family.children)
    single = copy.deepcopy(projected)
    single["markets"] = single["markets"][:1]
    match = kalshi_event_match(single, match_confidence=1.0, channel="venue_search")
    assert match is not None
    assert match.price_change_since_previous == pytest.approx(0.26)
    assert match.price_change_1d is None
    for field in ("last_price_dollars", "previous_price_dollars"):
        for invalid in (None, "NaN"):
            invalid_event = copy.deepcopy(single)
            invalid_event["markets"][0][field] = invalid
            match = kalshi_event_match(invalid_event, match_confidence=1.0, channel="venue_search")
            assert match is not None
            assert match.price_change_since_previous is None


def test_collapsed_kalshi_family_does_not_mislabel_one_strikes_metadata() -> None:
    path = Path(__file__).parent / "data" / "kalshi_family_liquidity_2026_08_05.json"
    page = json.loads(path.read_text(encoding="utf-8"))
    event = next(event for event in page["events"] if event["event_ticker"] == "KXGOVWINS-27JAN01")
    event["markets"] = event["markets"][:2]
    event["markets"][0]["status"] = "finalized"
    event["markets"][1].update(
        status="active",
        volume_24h_fp="37",
        updated_time="2026-07-21T18:49:13Z",
        last_price_dollars="0.67",
        previous_price_dollars="0.41",
    )
    family = kalshi_event_match(event, match_confidence=1.0, channel="venue_search")
    assert family is not None
    assert family.volume_24h is None
    assert family.last_update_time is None
    assert family.price_change_since_previous is None
    single = copy.deepcopy(event)
    single["markets"] = single["markets"][1:]
    match = kalshi_event_match(single, match_confidence=1.0, channel="venue_search")
    assert match is not None
    assert match.volume_24h == pytest.approx(37 * 0.12)
    assert match.last_update_time is not None
    assert match.price_change_since_previous == pytest.approx(0.26)


def test_polymarket_multi_market_update_comes_from_event() -> None:
    path = Path(__file__).parent / "data" / "prediction_market_venue_payloads.json"
    payload = json.loads(path.read_text(encoding="utf-8"))["polymarket_search"]
    event = next(event for event in payload["events"] if len(event["markets"]) > 1)
    matches = parse_polymarket_matches({"events": [event]}, width=1)
    assert matches is not None
    assert matches[0].last_update_time == datetime.fromisoformat(event["updatedAt"].replace("Z", "+00:00"))
