"""The known-API rung wiring: fanout, provenance, and shared market resources."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from metaculus_bot.research import resolution_source
from metaculus_bot.research.agentic import provenance
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.fetch_ladder.context import LadderContext
from metaculus_bot.research.known_api import backends, wiring
from metaculus_bot.research.known_api.result import KnownApiResult
from metaculus_bot.research.resolution_fetch_result import FetchResult
from tests.resolution_source_fakes import FakeSession

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "url",
    [
        "https://finance.yahoo.com/quote/SPCX/history/?period1=1784592000&period2=1785024000",
        "https://query1.finance.yahoo.com/v8/finance/chart/SPCX?period1=1784592000&period2=1785024000&interval=1d",
    ],
)
async def test_yahoo_exclusive_end_is_excluded_from_rendered_history(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    ceilings: list[date] = []

    def fetch_series(spec: Any, ceiling: date, *, lookback_years: int) -> pd.Series:
        ceilings.append(ceiling)
        return pd.Series([100.0, 200.0], index=pd.to_datetime(["2026-07-25", "2026-07-26"]))

    monkeypatch.setattr(backends.ts_fetch, "fetch_series", fetch_series)
    fetcher = wiring.build_known_api_fetcher(session=object())
    result = await fetcher(url)

    assert ceilings == [date(2026, 7, 25)]
    assert result is not None
    assert result.status == "success"
    assert "2026-07-25" in result.text
    assert "2026-07-26" not in result.text


def _ok(series_id: str) -> KnownApiResult:
    source_url = f"https://fred.stlouisfed.org/series/{series_id}"
    return KnownApiResult(
        status="ok",
        content_markdown=f"### {series_id}\nvalue",
        source_url=source_url,
        links=[source_url],
    )


async def test_rung_zero_fans_out_both_fred_ids_and_preserves_links(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    async def _fred(
        *,
        series_id: str | None = None,
        search: str | None = None,
        start: date | None = None,
        end: date | None = None,
        first_release: bool = False,
    ) -> KnownApiResult:
        seen.append(
            {
                "series_id": series_id,
                "search": search,
                "start": start,
                "end": end,
                "first_release": first_release,
            }
        )
        assert series_id is not None
        return _ok(series_id)

    monkeypatch.setattr(backends, "fred_series", _fred)
    fetcher = wiring.build_known_api_fetcher(session=object())

    result = await fetcher(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS30,DGS10&cosd=2026-06-01&coed=2026-07-31"
    )

    assert isinstance(result, FetchResult)
    assert result.status == "success"
    assert result.route == "known_api"
    assert result.links == [
        "https://fred.stlouisfed.org/series/DGS30",
        "https://fred.stlouisfed.org/series/DGS10",
    ]
    assert "### DGS30" in result.text
    assert "### DGS10" in result.text
    assert [call["series_id"] for call in seen] == ["DGS30", "DGS10"]
    assert all(call["start"] == date(2026, 6, 1) for call in seen)
    assert all(call["end"] == date(2026, 7, 31) for call in seen)


async def test_rung_zero_declines_partial_fred_fanout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []

    async def _fred(**kwargs: Any) -> KnownApiResult:
        series_id = kwargs["series_id"]
        seen.append(series_id)
        if series_id == "DGS10":
            return KnownApiResult(status="not_found", content_markdown="missing", source_url="")
        return _ok(series_id)

    monkeypatch.setattr(backends, "fred_series", _fred)
    fetcher = wiring.build_known_api_fetcher(session=object())

    result = await fetcher("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS30,DGS10")

    assert result is None
    assert seen == ["DGS30", "DGS10"]


async def test_explicit_market_tool_and_rung_zero_share_session_and_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    session = object()
    ctx = LadderContext(session=session)
    specs = agentic_tools.build_gap_fill_tools("topic", ctx=ctx)
    assert [spec.name for spec in specs] == [
        "search_news",
        "search_web",
        "fetch",
        "read_document",
        "view_image",
        "fred_series",
        "yahoo_history",
        "market_snapshot",
    ]

    seen: list[tuple[object, backends.KalshiGetBudget]] = []

    async def _market(
        *, venue: str, market: str, session: object, kalshi_detail_budget: backends.KalshiGetBudget, **kwargs: Any
    ) -> KnownApiResult:
        del venue, market, kwargs
        assert kalshi_detail_budget.take()
        seen.append((session, kalshi_detail_budget))
        return KnownApiResult(status="ok", content_markdown="snapshot", source_url="https://kalshi.com/markets/X")

    monkeypatch.setattr(backends, "market_snapshot", _market)
    market = next(spec for spec in specs if spec.name == "market_snapshot")
    await market.handler(venue="kalshi", market="KXU3-26AUG")

    assert ctx.policy.known_api is not None
    rung_result = await ctx.policy.known_api("https://kalshi.com/markets/KXU3-26AUG")

    assert rung_result is not None
    assert seen[0][0] is session
    assert seen[1][0] is session
    assert seen[0][1] is seen[1][1]
    assert seen[0][1].remaining == backends.MAX_KALSHI_DETAIL_GETS - 2


async def test_resolution_source_uses_known_api_rung_before_page_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://kalshi.com/markets/KXU3-26AUG"
    session = FakeSession({})
    seen_sessions: list[object] = []

    async def _market(*, venue: str, market: str, session: object, **kwargs: Any) -> KnownApiResult:
        del venue, market, kwargs
        seen_sessions.append(session)
        return KnownApiResult(
            status="ok",
            content_markdown="### Kalshi snapshot\nprice",
            source_url=url,
            links=[url],
        )

    monkeypatch.setattr(resolution_source.guard, "_get_session", lambda: session)
    monkeypatch.setattr(backends, "market_snapshot", _market)

    results = await resolution_source.fetch_resolution_sources([url])

    assert [result.route for result in results] == ["known_api"]
    assert results[0].text == "### Kalshi snapshot\nprice"
    assert results[0].links == [url]
    assert seen_sessions == [session]
    assert session.requested == []


async def test_known_api_is_a_fetched_provenance_method() -> None:
    assert provenance._method_to_tier("known_api") == "fetched"
