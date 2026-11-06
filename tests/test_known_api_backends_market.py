"""The prediction-market known-API backend: Kalshi/Polymarket/Manifold/PredictIt as a snapshot.

Every venue read is faked. The venue search functions and the Kalshi detail GET are patched on
the module so no test opens a socket; the run's already-pulled Kalshi catalogue and PredictIt dump
are passed in (the per-question seam the loop wiring will fill).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest

from metaculus_bot.research.known_api import backends
from metaculus_bot.research.market_retrieval import venues
from metaculus_bot.research.market_retrieval.types import MarketMatch

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _match(title: str, prob: float, *, platform: str = "polymarket") -> MarketMatch:
    return MarketMatch(
        platform=platform,  # type: ignore[arg-type]
        market_title=title,
        market_url=f"https://example.com/{title.replace(' ', '-')}",
        implied_prob_yes=prob,
        bid=None,
        ask=None,
        spread=None,
        volume_24h=None,
        close_time=datetime(2026, 12, 31, tzinfo=UTC),
        is_resolved=False,
        match_confidence=1.0,
        raw_rules="",
    )


class TestVenueSearch:
    async def test_polymarket_search_renders_a_snapshot(self, monkeypatch: pytest.MonkeyPatch):
        async def _fake_search(session, query, *, width):
            return [_match("Fed cuts rates in December", 0.62)]

        monkeypatch.setattr(venues, "polymarket_search", _fake_search)

        result = await backends.market_snapshot(venue="polymarket", market="fed rate cut", session=object())

        assert result.status == "ok"
        assert "Fed cuts rates in December" in result.content_markdown

    async def test_manifold_search(self, monkeypatch: pytest.MonkeyPatch):
        async def _fake_search(session, query, *, width):
            return [_match("Nobel Peace Prize 2026", 0.1, platform="manifold")]

        monkeypatch.setattr(venues, "manifold_search", _fake_search)

        result = await backends.market_snapshot(venue="manifold", market="nobel peace prize", session=object())

        assert result.status == "ok"
        assert "Nobel Peace Prize 2026" in result.content_markdown

    async def test_no_matches_is_not_found(self, monkeypatch: pytest.MonkeyPatch):
        async def _empty(session, query, *, width):
            return []

        monkeypatch.setattr(venues, "polymarket_search", _empty)

        result = await backends.market_snapshot(venue="polymarket", market="nothing here", session=object())

        assert result.status == "not_found"

    async def test_unknown_venue_is_error(self):
        result = await backends.market_snapshot(venue="betfair", market="anything", session=object())

        assert result.status == "error"
        assert "betfair" in result.content_markdown


class TestKalshi:
    _CATALOGUE: ClassVar[list[dict[str, Any]]] = [
        {"title": "US unemployment rate in August 2026", "event_ticker": "KXU3-26AUG", "markets": []},
        {"title": "Will it rain in London", "event_ticker": "KXRAIN", "markets": []},
    ]

    async def test_free_text_fuzzy_matches_the_catalogue_without_a_request(self, monkeypatch: pytest.MonkeyPatch):
        def _no_get(*args, **kwargs):
            raise AssertionError("free-text Kalshi must not fetch")

        monkeypatch.setattr(backends, "_kalshi_fetch_json", _no_get)

        result = await backends.market_snapshot(
            venue="kalshi", market="unemployment rate August", session=object(), kalshi_catalogue=self._CATALOGUE
        )

        assert result.status == "ok"
        assert "unemployment rate in August" in result.content_markdown.lower() or "KXU3" in result.content_markdown

    async def test_free_text_without_a_catalogue_is_empty(self):
        result = await backends.market_snapshot(venue="kalshi", market="unemployment rate", session=object())

        assert result.status == "empty"

    async def test_ticker_reads_the_event_endpoint(self, monkeypatch: pytest.MonkeyPatch):
        calls: list[str] = []

        async def _fake_get(session, url, **_):
            calls.append(url)
            return {"event": {"title": "US unemployment August", "event_ticker": "KXU3-26AUG", "markets": []}}

        monkeypatch.setattr(backends, "_kalshi_fetch_json", _fake_get)

        result = await backends.market_snapshot(venue="kalshi", market="KXU3-26AUG", session=object())

        assert result.status == "ok"
        assert "US unemployment August" in result.content_markdown
        assert any("events/KXU3-26AUG" in url for url in calls)

    async def test_ticker_falls_back_to_the_market_endpoint(self, monkeypatch: pytest.MonkeyPatch):
        async def _fake_get(session, url, **_):
            if "/events/" in url:
                return None
            return {"market": {"title": "Single strike market", "ticker": "KXONE-26DEC", "status": "active"}}

        monkeypatch.setattr(backends, "_kalshi_fetch_json", _fake_get)

        result = await backends.market_snapshot(venue="kalshi", market="KXONE-26DEC", session=object())

        assert result.status == "ok"
        assert "Single strike market" in result.content_markdown

    async def test_ticker_not_found(self, monkeypatch: pytest.MonkeyPatch):
        async def _none(session, url, **_):
            return None

        monkeypatch.setattr(backends, "_kalshi_fetch_json", _none)

        result = await backends.market_snapshot(venue="kalshi", market="KXNOPE-26DEC", session=object())

        assert result.status == "not_found"

    async def test_a_lowercase_ticker_still_reads_the_event_endpoint(self, monkeypatch: pytest.MonkeyPatch):
        calls: list[str] = []

        async def _fake_get(session, url, **_):
            calls.append(url)
            return {"event": {"title": "US unemployment August", "event_ticker": "KXU3-26AUG", "markets": []}}

        monkeypatch.setattr(backends, "_kalshi_fetch_json", _fake_get)

        result = await backends.market_snapshot(venue="kalshi", market="kxu3-26aug", session=object())

        assert result.status == "ok"
        assert any("events/KXU3-26AUG" in url for url in calls)

    async def test_detail_get_budget_caps_at_four_across_calls(self):
        """Patch the session, not _kalshi_fetch_json, so the real budget check inside it runs."""

        class _Resp:
            status = 404

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        class _CountingSession:
            def __init__(self) -> None:
                self.gets = 0

            def get(self, url, timeout=None):
                self.gets += 1
                return _Resp()

        session = _CountingSession()
        budget = backends.KalshiGetBudget()

        for _ in range(5):
            await backends.market_snapshot(
                venue="kalshi", market="KXQ-26DEC", session=session, kalshi_detail_budget=budget
            )

        assert session.gets == backends.MAX_KALSHI_DETAIL_GETS

    async def test_polymarket_outage_is_error_not_no_match(self, monkeypatch: pytest.MonkeyPatch):
        async def _outage(session, query, *, width):
            return None

        monkeypatch.setattr(venues, "polymarket_search", _outage)

        result = await backends.market_snapshot(venue="polymarket", market="fed", session=object())

        assert result.status == "error"

    async def test_five_row_cap(self, monkeypatch: pytest.MonkeyPatch):
        async def _many(session, query, *, width):
            return [_match(f"Market {i}", 0.5) for i in range(12)]

        monkeypatch.setattr(venues, "polymarket_search", _many)

        result = await backends.market_snapshot(venue="polymarket", market="anything", session=object())

        assert sum(line.startswith("| polymarket|") for line in result.content_markdown.splitlines()) == (
            backends.MARKET_SNAPSHOT_ROWS
        )


class TestPredictIt:
    _DUMP: ClassVar[list[dict[str, Any]]] = [
        {"name": "Which party wins the 2026 Senate", "contracts": [{"name": "Republicans", "status": "open"}]},
        {"name": "Unrelated market", "contracts": [{"name": "X", "status": "open"}]},
    ]

    async def test_matches_the_cached_dump(self):
        result = await backends.market_snapshot(
            venue="predictit", market="Senate 2026", session=object(), predictit_markets=self._DUMP
        )

        assert result.status == "ok"
        assert "Senate" in result.content_markdown

    async def test_without_a_dump_is_empty(self):
        result = await backends.market_snapshot(venue="predictit", market="Senate", session=object())

        assert result.status == "empty"
