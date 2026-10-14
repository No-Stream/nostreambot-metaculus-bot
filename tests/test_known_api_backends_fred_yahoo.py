"""The FRED and Yahoo known-API backends: deterministic reads, neutral results, never raise.

Every fetch is faked. The FRED path patches ``fred_rendering.Fred`` and ``ts_fetch.fetch_series``
(the repo rule: the patch target is ``fred_rendering``, not ``financial_data`` -- fredapi's real
class carries the identical literals, so a patch at the wrong module stays green while proving
nothing). The autouse network guard means an unpatched call fails loudly rather than dialing.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import pytest

from metaculus_bot.research import fred_rendering, ts_fetch
from metaculus_bot.research.known_api import backends
from metaculus_bot.research.known_api.backends import MAX_OBSERVATIONS

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _series(values: dict[str, float]) -> pd.Series:
    index = pd.DatetimeIndex([pd.Timestamp(day) for day in values])
    return pd.Series(list(values.values()), index=index, dtype="float64")


class _FakeFred:
    def __init__(self, data: pd.Series | Exception, *, title: str = "Some Title") -> None:
        self._data = data
        self._title = title

    def get_series(self, series_id: str, observation_start=None, observation_end=None, **kwargs):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data

    def get_series_info(self, series_id: str):
        return pd.Series({"title": self._title})


class TestFredSeriesKeyed:
    def _patch_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRED_API_KEY", "test-key")

    async def test_renders_a_windowed_block(self, monkeypatch: pytest.MonkeyPatch):
        self._patch_key(monkeypatch)
        data = _series({"2026-06-01": 4.20, "2026-06-02": 4.25, "2026-06-03": 4.30})
        monkeypatch.setattr(fred_rendering, "Fred", lambda api_key: _FakeFred(data, title="30Y Treasury"))

        result = await backends.fred_series(series_id="DGS30")

        assert result.status == "ok"
        assert "DGS30" in result.content_markdown
        assert "30Y Treasury" in result.content_markdown
        assert "4.3" in result.content_markdown
        assert result.source_url == "https://fred.stlouisfed.org/series/DGS30"
        assert result.links == ["https://fred.stlouisfed.org/series/DGS30"]

    async def test_unknown_series_is_not_found(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
        self._patch_key(monkeypatch)
        caplog.set_level(logging.WARNING, logger=backends.__name__)
        error = ValueError("Bad Request. The series does not exist.")
        monkeypatch.setattr(fred_rendering, "Fred", lambda api_key: _FakeFred(error))

        result = await backends.fred_series(series_id="DEXBOUS")

        assert result.status == "not_found"
        assert "does not exist" in result.content_markdown.lower()
        assert "FRED_UNKNOWN_SERIES: series_id=DEXBOUS proposed_by=gap_fill_driver" in caplog.text

    async def test_empty_series_is_empty(self, monkeypatch: pytest.MonkeyPatch):
        self._patch_key(monkeypatch)
        monkeypatch.setattr(fred_rendering, "Fred", lambda api_key: _FakeFred(pd.Series([], dtype="float64")))

        result = await backends.fred_series(series_id="DGS30")

        assert result.status == "empty"

    async def test_transport_error_names_the_exception(self, monkeypatch: pytest.MonkeyPatch):
        self._patch_key(monkeypatch)
        monkeypatch.setattr(fred_rendering, "Fred", lambda api_key: _FakeFred(OSError("connection reset")))

        result = await backends.fred_series(series_id="DGS30")

        assert result.status == "error"
        assert "OSError" in result.content_markdown

    async def test_window_caps_at_the_max_newest_kept(self, monkeypatch: pytest.MonkeyPatch):
        self._patch_key(monkeypatch)
        dates = pd.date_range("2000-01-01", periods=MAX_OBSERVATIONS + 50, freq="D")
        big = pd.Series(range(len(dates)), index=dates, dtype="float64")
        monkeypatch.setattr(fred_rendering, "Fred", lambda api_key: _FakeFred(big))

        # An explicit wide window keeps the whole series in range, so the cap is what bounds it.
        result = await backends.fred_series(series_id="DGS30", start=date(2000, 1, 1))

        rendered_rows = result.content_markdown.count("\n  - ")
        assert rendered_rows == MAX_OBSERVATIONS
        assert f"capped at {MAX_OBSERVATIONS}" in result.content_markdown
        # Newest kept: the first rendered row is the series' last date, the last row is 399 days earlier.
        newest = dates[-1].strftime("%Y-%m-%d")
        oldest_kept = dates[-MAX_OBSERVATIONS].strftime("%Y-%m-%d")
        first_row = result.content_markdown.split("\n  - ")[1]
        assert newest in first_row
        assert oldest_kept in result.content_markdown

    async def test_empty_series_with_a_window_is_empty_not_a_crash(self, monkeypatch: pytest.MonkeyPatch):
        self._patch_key(monkeypatch)
        # FRED returns an empty (RangeIndex) series for a window with no observations.
        monkeypatch.setattr(fred_rendering, "Fred", lambda api_key: _FakeFred(pd.Series([], dtype="float64")))

        result = await backends.fred_series(series_id="CSUSHPISA", start=date(2026, 9, 1), end=date(2026, 9, 10))

        assert result.status == "empty"

    async def test_first_release_label_only_when_the_comparison_renders(self, monkeypatch: pytest.MonkeyPatch):
        self._patch_key(monkeypatch)
        data = _series({"2026-06-01": 4.2, "2026-07-01": 4.3})
        monkeypatch.setattr(fred_rendering, "Fred", lambda api_key: _FakeFred(data, title="UNRATE"))
        # No first-release table available: the header must not claim one over current-vintage rows.
        monkeypatch.setattr(fred_rendering, "_fetch_fred_first_releases", lambda *a, **k: None)

        result = await backends.fred_series(series_id="UNRATE", first_release=True)

        assert result.status == "ok"
        assert "first-release" not in result.content_markdown

    async def test_default_window_is_the_last_thirty(self, monkeypatch: pytest.MonkeyPatch):
        self._patch_key(monkeypatch)
        dates = pd.date_range("2026-01-01", periods=90, freq="D")
        data = pd.Series(range(len(dates)), index=dates, dtype="float64")
        monkeypatch.setattr(fred_rendering, "Fred", lambda api_key: _FakeFred(data))

        result = await backends.fred_series(series_id="DGS30")

        assert result.content_markdown.count("\n  - ") == backends.DEFAULT_OBSERVATIONS


class TestFredSeriesKeyless:
    async def test_falls_back_to_ts_fetch_when_no_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        data = _series({"2026-06-01": 4.20, "2026-06-02": 4.25})
        monkeypatch.setattr(ts_fetch, "fetch_series", lambda spec, ceiling, **kw: data)

        result = await backends.fred_series(series_id="DGS10")

        assert result.status == "ok"
        assert "DGS10" in result.content_markdown
        assert "4.25" in result.content_markdown


class TestFredSearch:
    async def test_renders_search_hits(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        hits = pd.DataFrame(
            {
                "title": ["10-Year Treasury", "10Y-2Y Spread"],
                "frequency": ["Daily", "Daily"],
                "units": ["Percent", "Percent"],
            },
            index=["DGS10", "T10Y2Y"],
        )

        class _SearchFred(_FakeFred):
            def search(self, text: str, limit: int = 1000, **kwargs):
                return hits

        monkeypatch.setattr(fred_rendering, "Fred", lambda api_key: _SearchFred(pd.Series([], dtype="float64")))

        result = await backends.fred_series(search="10 year treasury")

        assert result.status == "ok"
        assert "DGS10" in result.content_markdown
        assert "T10Y2Y" in result.content_markdown


class TestYahooHistory:
    async def test_renders_a_windowed_history(self, monkeypatch: pytest.MonkeyPatch):
        data = _series({"2026-09-05": 7600.0, "2026-09-08": 7636.36})
        monkeypatch.setattr(ts_fetch, "fetch_series", lambda spec, ceiling, **kw: data)

        result = await backends.yahoo_history(ticker="^GSPC")

        assert result.status == "ok"
        assert "^GSPC" in result.content_markdown
        assert "7636.36" in result.content_markdown
        assert result.source_url == "https://finance.yahoo.com/quote/^GSPC/history/"

    async def test_delisted_ticker_is_not_found(self, monkeypatch: pytest.MonkeyPatch):
        def _raise(spec, ceiling, **kw):
            raise ts_fetch.FetchError("empty history (bad or delisted ticker?)")

        monkeypatch.setattr(ts_fetch, "fetch_series", _raise)

        result = await backends.yahoo_history(ticker="NOPE")

        assert result.status == "not_found"
        assert "delisted" in result.content_markdown.lower() or "not_found" in result.content_markdown.lower()

    async def test_column_is_passed_through(self, monkeypatch: pytest.MonkeyPatch):
        seen: dict[str, str] = {}

        def _capture(spec, ceiling, **kw):
            seen["column"] = spec.column
            return _series({"2026-09-08": 12.3})

        monkeypatch.setattr(ts_fetch, "fetch_series", _capture)

        await backends.yahoo_history(ticker="^VIX", column="High")

        assert seen["column"] == "High"

    async def test_transport_error_is_error(self, monkeypatch: pytest.MonkeyPatch):
        def _raise(spec, ceiling, **kw):
            raise OSError("connection reset")

        monkeypatch.setattr(ts_fetch, "fetch_series", _raise)

        result = await backends.yahoo_history(ticker="^GSPC")

        assert result.status == "error"
        assert "OSError" in result.content_markdown

    async def test_a_wide_window_widens_the_fetch_lookback(self, monkeypatch: pytest.MonkeyPatch):
        seen: dict[str, int] = {}

        def _capture(spec, ceiling, *, lookback_years=15, **kw):
            seen["lookback_years"] = lookback_years
            return _series({"2026-09-08": 12.3})

        monkeypatch.setattr(ts_fetch, "fetch_series", _capture)

        await backends.yahoo_history(ticker="^GSPC", start=date(2005, 1, 1), end=date(2026, 1, 1))

        assert seen["lookback_years"] > 15


class TestFredArguments:
    async def test_neither_series_id_nor_search_is_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FRED_API_KEY", "test-key")

        result = await backends.fred_series()

        assert result.status == "error"

    async def test_search_without_a_key_is_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)

        result = await backends.fred_series(search="treasury")

        assert result.status == "error"
