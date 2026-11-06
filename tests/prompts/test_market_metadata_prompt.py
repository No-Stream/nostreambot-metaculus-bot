"""Presence pins for the market metadata legend seen in forecaster prompts."""

from datetime import UTC, datetime

from metaculus_bot.prompts import binary_prompt
from metaculus_bot.research.market_retrieval.rendering import (
    MARKET_SIGNAL_LEGEND,
    TABLE_COLUMNS,
    render_snapshot,
)
from metaculus_bot.research.market_retrieval.types import MarketSnapshot
from metaculus_bot.research.market_retrieval.venues import kalshi_event_match
from tests.prompt_builders import _binary_q


def test_metadata_header_and_legend_reach_the_forecaster_prompt() -> None:
    assert "metadata" in TABLE_COLUMNS
    assert "`metadata`: v24 24h volume, liq liquidity" in MARKET_SIGNAL_LEGEND
    assert "prev Kalshi change vs its previous price (window unspecified)" in MARKET_SIGNAL_LEGEND
    assert "USD (Kalshi converted)" in MARKET_SIGNAL_LEGEND
    assert (
        "A stale last trade or wide bid-ask range makes a price less reliable; a large recent move often means news."
        in MARKET_SIGNAL_LEGEND
    )
    assert "PredictIt" in MARKET_SIGNAL_LEGEND
    assert "Kalshi 24h vol is contracts" not in MARKET_SIGNAL_LEGEND

    match = kalshi_event_match(
        {
            "event_ticker": "LEGEND-PIN",
            "title": "A market for the metadata legend",
            "markets": [],
        },
        match_confidence=1.0,
        channel="venue_search",
    )
    assert match is not None
    snapshot = MarketSnapshot(matches=[match], forecast_time=datetime(2026, 11, 4, tzinfo=UTC))
    research = render_snapshot(snapshot)
    prompt = binary_prompt(_binary_q(), research=research)

    assert MARKET_SIGNAL_LEGEND in research
    assert MARKET_SIGNAL_LEGEND in prompt
