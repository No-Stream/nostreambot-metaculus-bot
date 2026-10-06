"""Tests for the time-series-anchor provider: render, estimator math, provider, guards.

All HTTP is mocked. Two seams, matching the module's own layering:

- Provider soft-fail tests patch ``ts_fetch._http_get`` (the single synchronous HTTP seam
  that returns raw CSV bytes) so the real ``fetch_series`` parse + leakage guard runs and a
  genuine data error propagates into the provider's soft-fail path.
- Render / provider / guard tests monkeypatch ``timeseries_anchor.fetch_series`` with a
  canned synthetic series, so no network and a deterministic band. That target is where
  ``build_anchor_section`` LOOKS the symbol up, which is what makes the patch bite.

Coverage (one behavior per test):
- Render (``ts_render``): latest-value first line + P10/P50/P90 band line (single); both
  legs + band (spread); model_target=False withholds the band; derived-target labelling;
  the realized-max floor line; section char budget truncates.
- Estimator math (``ts_estimators``): horizon arithmetic, change/max bands (both the
  additive and log branches), the relative-return spread series, n_eff.
- Provider: disabled flag -> "" (even when routable); non-numeric question -> "";
  is_benchmarking=True uses ``question.open_time`` as the fetch ceiling (does NOT
  short-circuit like prediction_market — this provider is backtest-safe); env-flag gate is
  checked BEFORE the is_benchmarking as_of logic; malformed fetch -> "" + WARNING;
  two calls -> byte-identical output (determinism).
- Guards: the bounds-overlap magnitude backstop, the no-band-no-section rule, and the
  chart side-channel's success-path-only stashing.

The fetch layer itself is covered in ``test_ts_fetch.py`` and routing in
``test_ts_routing.py``; question mocks and the ``_http_get`` fake live in
``tests/ts_anchor_fakes.py``.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from forecasting_tools import BinaryQuestion

from metaculus_bot.constants import (
    FINANCIAL_VARIANCE_RATIO_FLOOR,
    FINANCIAL_VARIANCE_RATIO_LAG,
    FINANCIAL_VARIANCE_RATIO_MIN_RETURNS,
)
from metaculus_bot.research import timeseries_anchor as ts
from metaculus_bot.research import ts_fetch as tf
from metaculus_bot.research import ts_render as tsrender
from metaculus_bot.research.timeseries_anchor import (
    _band_misses_bounds,
    _reset_session_caches,
    build_anchor_section,
    timeseries_anchor_provider,
)
from metaculus_bot.research.ts_estimators import (
    CALENDAR_DAYS_PER_YEAR,
    MONTHLY_CLOCK,
    TRADING_DAYS_PER_YEAR,
    SeriesClock,
    _build_spread_series,
    _detect_freq,
    _empirical_change_band,
    _empirical_max_band,
    _horizon_end_date,
    _n_eff,
    annualized_realized_vol_pct,
    clock_matches_cadence,
    horizon_steps,
    multi_period_annualized_vol_pct,
    observed_periods_per_year,
    series_clock,
    variance_ratio,
)
from metaculus_bot.research.ts_fetch import FRED_CSV_URL, SeriesSpec
from metaculus_bot.research.ts_render import (
    _apply_derivation,
    _realized_max_floor,
    _render_single,
    _render_spread,
    _truncate_section,
)
from metaculus_bot.research.ts_routing import _Route, route_question
from scripts.telemetry.markers import MARKER_SPECS
from tests.ts_anchor_fakes import (
    _DGS10_RC,
    FakeHttp,
    _csv,
    _make_numeric_q,
    noise_dominated_close_series,
    random_walk_close_series,
)


# Test isolation: the provider keeps a rendered-section cache and the fetch layer
# keeps a parsed-series cache. Both bleed across tests otherwise.
@pytest.fixture(autouse=True)
def _reset_provider_caches():
    _reset_session_caches()
    yield
    _reset_session_caches()


# Synthetic series factories.


def _daily_positive_series(name: str, *, seed: int = 0, end: str = "2026-06-30", years: int = 6) -> pd.Series:
    """A strictly-positive daily business-day series, deterministic per seed."""
    end_ts = pd.Timestamp(end)
    idx = pd.bdate_range(end_ts - pd.Timedelta(days=round(years * 365.25)), end_ts)
    rng = np.random.default_rng(seed)
    walk = 20.0 + np.cumsum(rng.normal(0.0, 0.3, len(idx)))
    return pd.Series(np.abs(walk) + 8.0, index=idx, name=name)


def _twenty_four_seven_series(name: str, *, seed: int = 0, end: str = "2026-06-30", years: int = 6) -> pd.Series:
    """A strictly-positive series with a bar EVERY calendar day — the crypto shape.

    Same generator as ``_daily_positive_series`` but on ``date_range(freq="D")`` instead of
    ``bdate_range``, so ``_detect_freq`` still reads "daily" while the observed density is 1.0
    rows/day rather than 5/7. That gap is the whole defect class."""
    end_ts = pd.Timestamp(end)
    idx = pd.date_range(end=end_ts, periods=round(years * CALENDAR_DAYS_PER_YEAR), freq="D")
    rng = np.random.default_rng(seed)
    walk = 20.0 + np.cumsum(rng.normal(0.0, 0.3, len(idx)))
    return pd.Series(np.abs(walk) + 8.0, index=idx, name=name)


def _monthly_series(name: str, *, seed: int = 0, end: str = "2026-06-01", n: int = 96) -> pd.Series:
    """A strictly-positive monthly (month-start) series, deterministic per seed. n months
    ending at ``end`` — long enough that a small monthly horizon leaves ample overlap."""
    idx = pd.date_range(end=pd.Timestamp(end), periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    walk = 200.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    return pd.Series(np.abs(walk) + 50.0, index=idx, name=name)


def _weekly_series(name: str, *, seed: int = 0, end: str = "2026-06-26", n: int = 300) -> pd.Series:
    """A strictly-positive weekly series — the GASREGW shape, and the only frequency whose
    step noun and horizon divisor differ from both daily bases."""
    idx = pd.date_range(end=pd.Timestamp(end), periods=n, freq="W-FRI")
    rng = np.random.default_rng(seed)
    walk = 3.0 + np.cumsum(rng.normal(0.0, 0.02, n))
    return pd.Series(np.abs(walk) + 1.0, index=idx, name=name)


# Render.
class TestRenderSingle:
    def test_latest_value_first_line_and_band_line(self):
        series = _daily_positive_series("^VIX")
        route = _Route(kind="single", spec=SeriesSpec(source="yfinance", series_id="^VIX"), label="CBOE VIX")

        out, band = _render_single(series, route=route, ceiling=date(2026, 6, 30), calendar_days=14)

        assert band is not None  # a model-target series longer than the horizon renders a band
        first_line = out.splitlines()[0]
        assert first_line.startswith("**CBOE VIX** — latest ")
        assert "as of 2026-06-30" in first_line
        assert "P10 / P50 / P90 →" in out
        assert PROVENANCE_MARKER in out
        # The band line reports both the raw overlapping-window count and the ~independent
        # count (n_obs // h). A daily 14-day horizon -> h=10 trading days.
        assert "overlapping windows" in out
        assert "independent" in out
        h = horizon_steps(series_clock(pd.DatetimeIndex(series.index)), 14)
        assert f"~{series.size // h:,} independent" in out

    def test_stale_series_renames_the_range_instead_of_claiming_52_weeks(self):
        """A discontinued series has no observation inside the trailing year, and the whole
        history is the only band there is — but rendering it as a "52-week range" states a
        recency the numbers don't have (a 2019 high reads as this year's)."""
        series = _daily_positive_series("^DEAD", end="2024-06-28", years=2)
        route = _Route(kind="single", spec=SeriesSpec(source="yfinance", series_id="^DEAD"), label="Dead index")

        out, _ = _render_single(series, route=route, ceiling=date(2026, 6, 30), calendar_days=14)

        first_date = pd.DatetimeIndex(series.index).min().strftime("%Y-%m-%d")
        assert "52-week range" not in out
        assert f"full-history range ({first_date} to 2024-06-28; no observation inside the trailing year)" in out
        # The band itself still renders — the fallback carries real information.
        assert "of the way up the range" in out or "range is flat" in out

    def test_a_current_series_keeps_the_52_week_label(self):
        series = _daily_positive_series("^VIX")
        route = _Route(kind="single", spec=SeriesSpec(source="yfinance", series_id="^VIX"), label="CBOE VIX")

        out, _ = _render_single(series, route=route, ceiling=date(2026, 6, 30), calendar_days=14)

        assert "- 52-week range:" in out
        assert "full-history range" not in out

    def test_note_rendered_and_band_withheld_when_not_model_target(self):
        series = _daily_positive_series("PAYEMS")
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="fred", series_id="PAYEMS", revises=True),
            label="Total nonfarm payrolls",
            model_target=False,
            note="This is the payrolls LEVEL series.",
        )

        out, band = _render_single(series, route=route, ceiling=date(2026, 6, 30), calendar_days=30)

        assert band is None  # model_target=False -> no band computed, so nothing to return
        assert "- Note: This is the payrolls LEVEL series." in out
        # model_target=False -> no empirical band emitted at all.
        assert "P10 / P50 / P90" not in out
        assert "empirical band" not in out.lower()

    def test_todays_bar_marked_in_progress_on_a_live_ceiling(self, monkeypatch: pytest.MonkeyPatch):
        """A latest bar dated the fetch ceiling, rendered the same day, is still forming —
        and the empirical band anchors on it, so the header must say so. ``_today_utc``
        is frozen to the fixture's ceiling to simulate the live case deterministically."""
        series = _daily_positive_series("^VIX")
        route = _Route(kind="single", spec=SeriesSpec(source="yfinance", series_id="^VIX"), label="CBOE VIX")
        monkeypatch.setattr(tsrender, "_today_utc", lambda: date(2026, 6, 30))

        out, _ = _render_single(series, route=route, ceiling=date(2026, 6, 30), calendar_days=14)

        assert "as of 2026-06-30 — today's bar, in progress;" in out.splitlines()[0]

    def test_no_in_progress_marker_when_the_ceiling_is_a_past_date(self):
        """The benchmark shape: the bar is dated the ceiling, but the render runs later
        (the wall clock is NOT frozen), so the same-dated bar is a completed historical
        one and must not be called in-progress."""
        series = _daily_positive_series("^VIX")
        route = _Route(kind="single", spec=SeriesSpec(source="yfinance", series_id="^VIX"), label="CBOE VIX")

        out, _ = _render_single(series, route=route, ceiling=date(2026, 6, 30), calendar_days=14)

        assert "in progress" not in out
        assert "⚠" not in out  # a latest AT the ceiling is fresh, not stale

    def test_stale_daily_latest_gets_a_note_and_a_warn(self, caplog: pytest.LogCaptureFixture):
        """A daily series whose newest bar is years older than the ceiling: the 52-week
        fallback already renames the range, but the header's latest value — the number
        the band is applied to — needs its own staleness flag, in render and run logs."""
        series = _daily_positive_series("^DEAD", end="2024-06-28", years=2)
        route = _Route(kind="single", spec=SeriesSpec(source="yfinance", series_id="^DEAD"), label="Dead index")

        with caplog.at_level(logging.WARNING):
            out, _ = _render_single(series, route=route, ceiling=date(2026, 6, 30), calendar_days=14)

        age = (date(2026, 6, 30) - date(2024, 6, 28)).days
        assert f"⚠ Latest observation is {age} days old" in out
        assert f"FINANCIAL_STALE_LATEST: surface=ts_anchor symbol=^DEAD age_d={age} cadence=trading-day" in caplog.text


class TestRenderSpread:
    def test_renders_both_legs_and_band(self):
        series_a = _daily_positive_series("CL=F", seed=1)
        series_b = _daily_positive_series("^GSPC", seed=2) * 40.0  # distinct level
        route = _Route(
            kind="spread",
            spec=SeriesSpec(source="yfinance", series_id="CL=F"),
            label="CL=F",
            spec_b=SeriesSpec(source="yfinance", series_id="^GSPC"),
            label_b="^GSPC",
        )

        out, band = _render_spread(series_a, series_b, route=route, calendar_days=14)

        assert band is not None  # the spread always emits a band; returned for the bounds backstop
        assert len(band) == 3  # P10/P50/P90
        assert "Relative-return spread: CL=F vs ^GSPC" in out
        assert "- CL=F latest:" in out
        assert "- ^GSPC latest:" in out
        assert "- CL=F recent:" in out
        assert "- ^GSPC recent:" in out
        assert "relative-return band" in out
        assert "P10 / P50 / P90 →" in out
        # The spread band line also reports the overlapping + ~independent window counts.
        assert "overlapping windows" in out
        assert "independent" in out
        # §g: spread sections carry an explicit mean-zero-prior disclaimer.
        assert "mean-zero by construction" in out
        assert "not a directional signal" in out


class TestRoutedValueFieldRendering:
    def test_max_question_names_routed_high_field(self):
        route = route_question(_make_numeric_q(question_text="What is the highest VIX value this year?"))
        assert route is not None
        assert route.value_field == "High"
        assert route.is_max is True

        out, _ = _render_single(
            _daily_positive_series("^VIX"), route=route, ceiling=date(2026, 6, 30), calendar_days=14
        )

        assert "(High, as of 2026-06-30; series frequency: daily)" in out.splitlines()[0]

    def test_ordinary_question_names_routed_close_field(self):
        route = route_question(_make_numeric_q(question_text="What will the price of gold be on the date?"))
        assert route is not None
        assert route.value_field == "Close"
        assert route.is_max is False

        out, _ = _render_single(
            _daily_positive_series("GC=F"), route=route, ceiling=date(2026, 6, 30), calendar_days=14
        )

        assert "(Close, as of 2026-06-30; series frequency: daily)" in out.splitlines()[0]

    def test_unregistered_url_cited_ticker_names_routed_close_field(self):
        route = route_question(
            _make_numeric_q(
                question_text="What will the ALFA share price be?",
                resolution_criteria="Tracks https://finance.yahoo.com/quote/ALFA at the resolution date.",
            )
        )
        assert route is not None
        assert route.label == "ALFA"
        assert route.value_field == "Close"

        out, _ = _render_single(
            _daily_positive_series("ALFA"), route=route, ceiling=date(2026, 6, 30), calendar_days=14
        )

        assert "(Close, as of 2026-06-30; series frequency: daily)" in out.splitlines()[0]

    def test_derived_yfinance_target_names_column_without_repeating_frequency(self):
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="yfinance", series_id="TEST=F", column="High"),
            label="Test monthly high",
            derivation="monthly_avg",
        )

        out, _ = _render_single(_weekly_series("TEST=F"), route=route, ceiling=date(2026, 6, 30), calendar_days=30)

        first_line = out.splitlines()[0]
        assert "(High, monthly average of the higher-frequency series; from raw level " in first_line
        assert "daily High" not in first_line

    def test_paired_route_names_close_field_for_both_legs(self):
        route = route_question(
            _make_numeric_q(
                question_text="Will CL=F's returns exceed ^GSPC's over the window?",
                resolution_criteria=(
                    "Compares https://finance.yahoo.com/quote/CL=F and https://finance.yahoo.com/quote/%5EGSPC."
                ),
            )
        )
        assert route is not None
        assert route.kind == "spread"
        assert route.value_field == "Close"
        assert route.value_field_b == "Close"

        out, _ = _render_spread(
            _daily_positive_series("CL=F", seed=1),
            _daily_positive_series("^GSPC", seed=2),
            route=route,
            calendar_days=14,
        )

        cl_latest_line = next(line for line in out.splitlines() if line.startswith("- CL=F latest: "))
        gspc_latest_line = next(line for line in out.splitlines() if line.startswith("- ^GSPC latest: "))
        assert cl_latest_line.endswith("(Close, as of 2026-06-30)")
        assert gspc_latest_line.endswith("(Close, as of 2026-06-30)")

    def test_fred_latest_line_keeps_existing_format(self):
        route = route_question(
            _make_numeric_q(
                question_text="What will the 10-year treasury yield be?",
                resolution_criteria=_DGS10_RC,
            )
        )
        assert route is not None
        assert route.spec.source == "fred"
        assert route.value_field is None

        out, _ = _render_single(
            _daily_positive_series("DGS10"), route=route, ceiling=date(2026, 6, 30), calendar_days=14
        )

        first_line = out.splitlines()[0]
        assert first_line.startswith(f"**{route.label}** — latest ")
        assert first_line.endswith("(as of 2026-06-30; series frequency: daily)")
        assert "daily Close" not in first_line
        assert "daily High" not in first_line


# Derived-target math: hand-confirmed reference values from the replay (Phase A).
class TestDerivedTargets:
    def test_mom_diff_scaled_first_difference(self):
        # PAYEMS-style: [100,110,105] x1000 -> diffs [10000, -5000].
        idx = pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"])
        s = pd.Series([100.0, 110.0, 105.0], index=idx)
        out = _apply_derivation(s, "mom_diff", 1000.0)
        assert out.tolist() == pytest.approx([10000.0, -5000.0])

    def test_mom_pct_percent_change(self):
        # CPI-style: [100,110,105] -> MoM % [10.0, -4.5455].
        idx = pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"])
        s = pd.Series([100.0, 110.0, 105.0], index=idx)
        out = _apply_derivation(s, "mom_pct", 1.0)
        assert out.tolist() == pytest.approx([10.0, -4.545454545454546])

    def test_monthly_avg_of_weekly(self):
        # Gasoline-style: weekly [3,4,5] in Jan + [15] in Feb -> {Jan 4.0, Feb 15.0}.
        idx = pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19", "2026-02-02"])
        w = pd.Series([3.0, 4.0, 5.0, 15.0], index=idx)
        out = _apply_derivation(w, "monthly_avg", 1.0)
        assert [str(d.date()) for d in out.index] == ["2026-01-01", "2026-02-01"]
        assert out.tolist() == pytest.approx([4.0, 15.0])

    def test_level_scale_millions_to_billions(self):
        # BOPGTB-style unit conversion: millions of USD -> billions via scale=0.001.
        idx = pd.to_datetime(["2026-01-01"])
        out = _apply_derivation(pd.Series([-81800.0], index=idx), "level", 0.001)
        assert out.tolist() == pytest.approx([-81.8])

    def test_level_scale_one_is_identity(self):
        idx = pd.to_datetime(["2026-01-01", "2026-02-01"])
        s = pd.Series([4.1, 4.3], index=idx)
        out = _apply_derivation(s, "level", 1.0)
        pd.testing.assert_series_equal(out, s)


class TestRealizedMaxFloor:
    def test_floor_is_elapsed_window_max(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="D")
        s = pd.Series([10.0, 12.0, 11.0, 15.0, 13.0, 9.0, 8.0, 7.0, 6.0, 5.0], index=idx)
        # Max over the elapsed portion [window_start, ceiling] = 15.0 (the fourth obs).
        floor = _realized_max_floor(s, window_start=date(2026, 1, 1), ceiling=date(2026, 1, 5))
        assert floor == pytest.approx(15.0)

    def test_no_floor_when_window_not_yet_open(self):
        # Benchmark path: window_start == ceiling -> no elapsed portion, no floor.
        idx = pd.date_range("2026-01-01", periods=5, freq="D")
        s = pd.Series([10.0, 12.0, 11.0, 15.0, 13.0], index=idx)
        assert _realized_max_floor(s, window_start=date(2026, 1, 3), ceiling=date(2026, 1, 3)) is None
        assert _realized_max_floor(s, window_start=None, ceiling=date(2026, 1, 3)) is None


class TestEffectiveWindowCount:
    """n_eff ~= n_obs // h captures that overlapping windows share observations, so
    the independent-sample count is far below the raw overlapping-window count at
    long horizons. Floored at 1 for degenerate inputs."""

    def test_long_horizon_collapses_to_few_independent_windows(self):
        # 15 years of daily data, 1-year (h=252 trading-day) horizon -> ~15 independent.
        assert _n_eff(15 * 252, 252) == 15

    def test_short_horizon_keeps_many_independent_windows(self):
        assert _n_eff(3780, 10) == 378

    def test_floored_at_one(self):
        # Fewer observations than the horizon (never happens where the band renders,
        # but the count must stay honest rather than go to zero).
        assert _n_eff(5, 10) == 1
        assert _n_eff(0, 10) == 1


class TestRenderDerived:
    def test_mom_diff_labels_derived_quantity_and_history(self):
        # A monthly level series routed through mom_diff x1000 must render the DERIVED
        # change values, not the raw level, and label them clearly.
        series = _monthly_series("PAYEMS")
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="fred", series_id="PAYEMS", revises=True),
            label="Nonfarm payrolls — MoM change",
            derivation="mom_diff",
            scale=1000.0,
        )

        out, band = _render_single(series, route=route, ceiling=date(2026, 6, 1), calendar_days=30)

        assert band is not None
        assert "latest derived value" in out
        assert "month-over-month change" in out
        assert "from raw level" in out  # the raw level is still surfaced for context
        assert "(derived)" in out  # the history block is labeled as derived values
        assert "P10 / P50 / P90 →" in out
        assert "52-week range" not in out  # the level-only 52w line is skipped for derived Qs

    def test_mom_pct_renders_percent_change_band(self):
        series = _monthly_series("CPIAUCSL", seed=3)
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="fred", series_id="CPIAUCSL", revises=True),
            label="CPI MoM % change",
            derivation="mom_pct",
        )

        out, band = _render_single(series, route=route, ceiling=date(2026, 6, 1), calendar_days=30)

        assert band is not None
        assert "month-over-month % change" in out
        assert "P10 / P50 / P90 →" in out

    def test_max_window_realized_floor_line_when_window_started(self):
        # A daily High series, max framing, window already open before the ceiling ->
        # the realized-max floor line appears and lifts the band.
        series = _daily_positive_series("BTC-USD", end="2026-06-30")
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="yfinance", series_id="BTC-USD", column="High"),
            label="Bitcoin highest",
            is_max=True,
        )

        out, band = _render_single(
            series,
            route=route,
            ceiling=date(2026, 6, 30),
            calendar_days=30,
            window_start=date(2026, 1, 1),  # window opened months before the ceiling
        )

        assert band is not None
        assert "Realized max so far this window" in out
        assert "HARD LOWER BOUND" in out
        assert "forward-max" in out

    def test_max_window_no_floor_line_when_window_not_open(self):
        series = _daily_positive_series("^VIX", end="2026-06-30")
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="yfinance", series_id="^VIX", column="High"),
            label="VIX max",
            is_max=True,
        )
        # Benchmark path: window_start == ceiling -> no elapsed portion, no floor line.
        out, band = _render_single(
            series,
            route=route,
            ceiling=date(2026, 6, 30),
            calendar_days=14,
            window_start=date(2026, 6, 30),
        )
        assert band is not None
        assert "Realized max so far this window" not in out
        assert "forward-max" in out


class TestTruncateSection:
    def test_section_char_budget_enforced(self, monkeypatch):
        # Patched on ts_render, which is where _truncate_section reads the budget.
        monkeypatch.setattr(tsrender, "TS_ANCHOR_SECTION_MAX_CHARS", 120)
        text = "line\n" * 200  # ~1000 chars, well over the shrunken budget

        out = _truncate_section(text)

        assert len(out) <= 120
        assert out.endswith("[truncated — time-series anchor section budget]")

    def test_under_budget_passthrough(self):
        text = "short section"
        assert _truncate_section(text) == text


PROVENANCE_MARKER = "Statistical extrapolation of the resolution series' own history"


# Estimator math: known input -> hand-computed expected output. These pin the
# band/horizon arithmetic against silent mutations (e.g. a base/fwd swap or a
# swapped horizon constant) that a string-presence render test cannot see.
class TestEstimatorMath:
    def test_horizon_steps_matches_documented_formula(self):
        # daily on the trading-day basis: round(days * 252/365); weekly: round(days/7);
        # monthly: round(days/30.4375).
        trading = SeriesClock(freq="daily", periods_per_year=TRADING_DAYS_PER_YEAR)
        assert horizon_steps(trading, 30) == 21  # round(30 * 252 / 365) = round(20.7123)
        assert horizon_steps(SeriesClock(freq="weekly", periods_per_year=TRADING_DAYS_PER_YEAR), 21) == 3
        assert horizon_steps(SeriesClock(freq="monthly", periods_per_year=TRADING_DAYS_PER_YEAR), 90) == 3

    def test_horizon_steps_floored_at_one(self):
        # round(5 / 30.4375) = round(0.164) = 0 -> floored to 1 (never a 0-step horizon).
        assert horizon_steps(SeriesClock(freq="monthly", periods_per_year=TRADING_DAYS_PER_YEAR), 5) == 1

    def test_change_band_additive_ramp_is_exactly_h_step(self):
        # Constant-step additive ramp: every overlapping h-step change equals h*step
        # exactly, so P10=P50=P90 collapse. anchor=0 -> band returns the raw change,
        # which must be +h*step (a base/fwd swap would flip the sign to -h*step).
        step = 10.0
        y = np.arange(0, 20, dtype="float64") * step + 10.0  # 10, 20, 30, ...
        h = 3
        p10, p50, p90 = _empirical_change_band(y, h, use_log=False, anchor=0.0)
        assert p10 == pytest.approx(h * step)
        assert p50 == pytest.approx(h * step)
        assert p90 == pytest.approx(h * step)

    def test_change_band_log_branch_constant_ratio(self):
        # Constant-ratio positive series y = 100 * 1.01^t: every h-step log change is
        # exactly h*log(1.01), so the log-multiplicative band collapses to last*1.01^h.
        t = np.arange(0, 51, dtype="float64")
        ratio = 1.01
        y = 100.0 * ratio**t
        h = 3
        last = float(y[-1])
        expected = last * ratio**h
        p10, p50, p90 = _empirical_change_band(y, h, use_log=True, anchor=last)
        assert p10 == pytest.approx(expected)
        assert p50 == pytest.approx(expected)
        assert p90 == pytest.approx(expected)

    def test_max_band_hand_computed_window_max(self):
        # y=[1,3,2,5,4], h=2. Each window spans h+1=3 points (an h-step horizon, matching
        # the change band's y[i+h]-vs-y[i] span): [1,3,2],[3,2,5],[2,5,4] ->
        # window_max=[3,5,5], win_anchor=y[:3]=[1,3,2], diffs=[2,2,3].
        # sorted diffs=[2,2,3]; numpy linear quantiles at (.10,.50,.90) over n=3:
        #   .10 -> pos 0.2 -> 2.0;  .50 -> pos 1.0 -> 2.0;  .90 -> pos 1.8 -> 2.8.
        # anchor last=10 -> (12.0, 12.0, 12.8). A window_max/anchor swap flips the
        # diffs negative and this fails; a length-h (not h+1) window regresses to the
        # old [2,0,3,0] -> (10.0, 11.0, 12.7).
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        p10, p50, p90 = _empirical_max_band(y, 2, use_log=False, last=10.0)
        assert p10 == pytest.approx(12.0)
        assert p50 == pytest.approx(12.0)
        assert p90 == pytest.approx(12.8)

    def test_max_band_log_branch_hand_computed(self):
        # F12: the use_log=True branch is the production path for every strictly-positive
        # financial series (VIX/BTC/gold "highest value" questions) — hand-pin it.
        # y=[1,2,1,4,2], h=2. Windows span h+1=3 points: [1,2,1],[2,1,4],[1,4,2] ->
        # window_max=[2,4,4], win_anchor=y[:3]=[1,2,1].
        # log-ratios = [ln(2/1), ln(4/2), ln(4/1)] = [ln2, ln2, 2*ln2].
        # sorted=[ln2, ln2, 2*ln2]; numpy linear quantiles over n=3:
        #   .10 -> pos 0.2 -> ln2;  .50 -> pos 1.0 -> ln2;  .90 -> pos 1.8 -> 1.8*ln2.
        # last=10, band = last*exp(r) -> (10*2, 10*2, 10*2^1.8) = (20.0, 20.0, ~34.822).
        y = np.array([1.0, 2.0, 1.0, 4.0, 2.0])
        p10, p50, p90 = _empirical_max_band(y, 2, use_log=True, last=10.0)
        assert p10 == pytest.approx(20.0)
        assert p50 == pytest.approx(20.0)
        assert p90 == pytest.approx(10.0 * 2.0**1.8)  # ~34.822

    def test_build_spread_series_relative_returns(self):
        # a=[10,20,25], b=[5,5,10] on aligned dates. rel = 100*[(logA-logA0)-(logB-logB0)]:
        #   t0 -> 0
        #   t1 -> 100*(log2 - 0)        = 100*ln(2)    ~= 69.3147
        #   t2 -> 100*(log2.5 - log2)   = 100*ln(1.25) ~= 22.3144
        idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        a = pd.Series([10.0, 20.0, 25.0], index=idx)
        b = pd.Series([5.0, 5.0, 10.0], index=idx)
        spread = _build_spread_series(a, b)
        assert spread.iloc[0] == pytest.approx(0.0)
        assert spread.iloc[1] == pytest.approx(100.0 * float(np.log(2.0)))
        assert spread.iloc[2] == pytest.approx(100.0 * float(np.log(1.25)))

    def test_build_spread_series_disjoint_calendar_raises(self):
        # No overlapping dates -> inner join empty -> ValueError (the documented contract).
        a = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
        b = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-02-01", "2026-02-02"]))
        with pytest.raises(ValueError, match="no overlapping dates"):
            _build_spread_series(a, b)


# The calendar/frequency assumption class (the q44882 sqrt(252) defect and its ts-anchor
# siblings). Every test here contrasts a business-day series against a 24/7 one whose bars land
# every calendar day: `_detect_freq` reads both as "daily", so `freq` alone cannot drive a
# calendar<->row conversion and `SeriesClock` carries the observed density alongside it.
class TestSeriesClockAndCalendarBases:
    def test_detect_freq_cannot_tell_the_two_daily_shapes_apart(self):
        # The reason SeriesClock exists: the median gap is 1.0 day for BOTH a business-day
        # series (gaps 1,1,1,1,3) and a 24/7 one, so freq is blind to a 365/252 = 1.45x
        # difference in rows per year. If this ever stops holding, the clock can be simplified.
        business = _daily_positive_series("^GSPC")
        continuous = _twenty_four_seven_series("BTC-USD")
        assert _detect_freq(pd.DatetimeIndex(business.index)) == "daily"
        assert _detect_freq(pd.DatetimeIndex(continuous.index)) == "daily"

    def test_clock_reads_the_two_bases_off_the_index(self):
        business = series_clock(pd.DatetimeIndex(_daily_positive_series("^GSPC").index))
        continuous = series_clock(pd.DatetimeIndex(_twenty_four_seven_series("BTC-USD").index))
        assert (business.freq, business.periods_per_year, business.step_unit) == (
            "daily",
            TRADING_DAYS_PER_YEAR,
            "trading-day",
        )
        assert (continuous.freq, continuous.periods_per_year, continuous.step_unit) == (
            "daily",
            CALENDAR_DAYS_PER_YEAR,
            "calendar-day",
        )

    def test_clock_leaves_weekly_and_monthly_on_the_nominal_basis(self):
        # Their horizon conversions are already calendar-honest (/7, /30.4375), so the density
        # read is not consulted and must not perturb them.
        monthly = series_clock(pd.DatetimeIndex(_monthly_series("CPIAUCSL").index))
        assert (monthly.freq, monthly.periods_per_year, monthly.step_unit) == (
            "monthly",
            TRADING_DAYS_PER_YEAR,
            "month",
        )

    def test_horizon_on_a_24_7_series_is_the_calendar_window_itself(self):
        # One step IS one calendar day, so a 90-day question is a 90-step horizon. The shipped
        # formula gave round(90 * 252/365) = 62 — a 62-day change band presented as 90 days,
        # ~sqrt(90/62) = 1.20x too narrow on exactly the most volatile asset class we route.
        continuous = series_clock(pd.DatetimeIndex(_twenty_four_seven_series("BTC-USD").index))
        assert horizon_steps(continuous, 90) == 90
        assert horizon_steps(continuous, 31) == 31
        business = series_clock(pd.DatetimeIndex(_daily_positive_series("^GSPC").index))
        assert horizon_steps(business, 90) == 62  # round(90 * 252/365) — unchanged

    def test_horizon_end_date_inverts_horizon_steps_on_the_same_clock(self):
        # These two conversions used to be wrong in OPPOSITE directions by the same 365/252, so
        # on a 24/7 series they cancelled: the chart ribbon ended at about the right date while
        # the band it drew was a 62-day band labelled 90. Pin the round-trip on BOTH bases so a
        # future fix to one alone fails here instead of skewing the chart.
        as_of = pd.Timestamp("2026-06-30")
        for series in (_daily_positive_series("^GSPC"), _twenty_four_seven_series("BTC-USD")):
            clock = series_clock(pd.DatetimeIndex(series.index))
            for calendar_days in (31, 90, 180):
                h = horizon_steps(clock, calendar_days)
                end = _horizon_end_date(as_of, clock, h)
                assert abs((end - as_of).days - calendar_days) <= 1, (clock, calendar_days, end)

    def test_density_read_degrades_to_the_trading_day_basis_when_unmeasurable(self):
        # Same fail-safe contract financial_data's fixtures pin: too few rows, a non-datetime
        # index, or a zero-span index all keep the historical 252 behavior.
        short = pd.date_range(end="2026-06-30", periods=10, freq="D")
        assert observed_periods_per_year(short) == TRADING_DAYS_PER_YEAR
        assert observed_periods_per_year(pd.RangeIndex(50)) == TRADING_DAYS_PER_YEAR
        assert observed_periods_per_year(pd.DatetimeIndex(["2026-06-30"] * 50)) == TRADING_DAYS_PER_YEAR

    def test_density_read_splits_at_six_bars_a_week_not_at_five(self):
        # The threshold is the whole discriminator between the two bases, and it sits at 6/7
        # rows per day precisely so an exchange series with holidays (0.690) and a six-session
        # week both stay on 252 while only a genuinely 24/7 series (1.0) crosses. A six-day
        # trading week is the nearest real shape below the line, so pin it: nudging the
        # threshold down to 5/7 would silently re-annualize every equity series on 365.
        six_day_week = pd.DatetimeIndex(
            [d for d in pd.date_range(end="2026-06-30", periods=700, freq="D") if d.weekday() != 6]
        )
        assert observed_periods_per_year(six_day_week) == TRADING_DAYS_PER_YEAR
        every_day = pd.date_range(end="2026-06-30", periods=700, freq="D")
        assert observed_periods_per_year(every_day) == CALENDAR_DAYS_PER_YEAR

    def test_density_read_needs_a_fortnight_of_bars_before_it_overrules_the_default(self):
        # The minimum-rows guard: a fortnight of bars cannot distinguish 5/7 from 7/7, so
        # anything shorter keeps the historical basis. Pinned at N-1 and N because an off-by-one
        # here is invisible in output — it just quietly re-annualizes short crypto frames.
        for n_rows, expected in ((13, TRADING_DAYS_PER_YEAR), (14, CALENDAR_DAYS_PER_YEAR)):
            index = pd.date_range(end="2026-06-30", periods=n_rows, freq="D")
            assert observed_periods_per_year(index) == expected, n_rows

    def test_clock_and_horizon_on_a_weekly_series_convert_on_sevens(self):
        # Weekly is the frequency whose step noun and horizon divisor differ from BOTH daily
        # bases, and the density read must stay unconsulted on it (a weekly index reads ~0.14
        # rows/day, well under the split, so a stray read would be harmless here but wrong in
        # principle — the basis field is nominal for non-daily clocks).
        clock = series_clock(pd.DatetimeIndex(_weekly_series("GASREGW").index))
        assert (clock.freq, clock.periods_per_year, clock.step_unit) == ("weekly", TRADING_DAYS_PER_YEAR, "week")
        assert horizon_steps(clock, 90) == 13  # round(90 / 7)
        assert horizon_steps(clock, 3) == 1  # floored at one step
        as_of = pd.Timestamp("2026-06-30")
        assert _horizon_end_date(as_of, clock, 13) == as_of + pd.Timedelta(weeks=13)

    def test_horizon_end_date_on_the_monthly_clock_steps_calendar_months(self):
        # The derived-target clock (MoM change / MoM % / monthly average). One step is one
        # calendar month, so the ribbon end is a DateOffset — not h * 30.4375 days, which would
        # drift a day or two per quarter against the month the question actually resolves in.
        as_of = pd.Timestamp("2026-06-30")
        assert MONTHLY_CLOCK.step_unit == "month"
        assert _horizon_end_date(as_of, MONTHLY_CLOCK, 3) == pd.Timestamp("2026-09-30")
        assert horizon_steps(MONTHLY_CLOCK, 90) == 3  # round(90 / 30.4375)

    def test_weekly_render_names_weeks_and_omits_the_daily_only_vol_line(self):
        # `step_unit` reaching the rendered band line through its real caller (`_band_line`),
        # on the one frequency where the noun is neither "trading-day" nor "calendar-day".
        # The vol note is daily-only, so a weekly series must not grow one — annualizing 30
        # WEEKLY returns on either daily basis would be off by ~sqrt(52/252).
        weekly = _weekly_series("GASREGW")
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="fred", series_id="GASREGW"),
            label="US regular gasoline ($/gal)",
        )

        out, band = _render_single(weekly, route=route, ceiling=date(2026, 6, 30), calendar_days=90)

        assert band is not None
        assert "all 13-week change windows" in out
        assert "trading-day" not in out
        assert "calendar-day" not in out
        assert "volatility" not in out

    def test_realized_vol_line_annualizes_on_the_observed_density(self):
        continuous = _twenty_four_seven_series("BTC-USD")
        clock = series_clock(pd.DatetimeIndex(continuous.index))
        returns = continuous.pct_change().dropna().tail(tsrender.REALIZED_VOL_WINDOW)
        expected = float(returns.std() * np.sqrt(CALENDAR_DAYS_PER_YEAR) * 100.0)
        shipped_252 = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)

        lines = tsrender._realized_vol_lines(continuous, clock, symbol="BTC-USD")

        assert lines == [f"- 30-calendar-day annualized realized volatility: {expected:.1f}%"]
        # The defect was worth a factor of sqrt(365/252) = 1.2035; make sure the old number is
        # genuinely different at the rendered precision rather than a rounding coincidence.
        assert f"{shipped_252:.1f}" != f"{expected:.1f}"
        assert expected == pytest.approx(shipped_252 * float(np.sqrt(365 / 252)))

    def test_realized_vol_line_labels_trading_days_on_an_exchange_series(self):
        # 30 rows is six calendar weeks here, so "30-day" was itself a row count posing as a
        # calendar window. The number is unchanged from the shipped sqrt(252) behavior.
        business = _daily_positive_series("^GSPC")
        clock = series_clock(pd.DatetimeIndex(business.index))
        returns = business.pct_change().dropna().tail(tsrender.REALIZED_VOL_WINDOW)
        expected = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)

        assert tsrender._realized_vol_lines(business, clock, symbol="^GSPC") == [
            f"- 30-trading-day annualized realized volatility: {expected:.1f}%"
        ]

    def test_rendered_band_on_a_24_7_series_uses_calendar_steps_and_says_so(self):
        continuous = _twenty_four_seven_series("BTC-USD")
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="yfinance", series_id="BTC-USD"),
            label="Bitcoin price ($)",
        )
        calendar_days = 90

        out, band = _render_single(continuous, route=route, ceiling=date(2026, 6, 30), calendar_days=calendar_days)

        assert band is not None
        # h == the calendar window, and the label names calendar steps rather than trading days.
        assert f"all {calendar_days}-calendar-day change windows" in out
        assert "trading-day" not in out
        # And the band really is the 90-step band, materially wider than the 62-step one the
        # shipped horizon produced.
        y = continuous.to_numpy(dtype="float64")
        last = float(continuous.iloc[-1])
        expected = _empirical_change_band(y, calendar_days, use_log=True, anchor=last)
        assert band == pytest.approx(expected)
        shipped = _empirical_change_band(y, round(calendar_days * 252 / 365), use_log=True, anchor=last)
        assert (band[2] - band[0]) > (shipped[2] - shipped[0])

    def test_business_day_render_is_unchanged_by_the_clock(self):
        # The 252 path must stay byte-identical apart from the vol label's unit word, so the
        # fix cannot have moved any exchange-traded question's band.
        business = _daily_positive_series("^VIX")
        route = _Route(kind="single", spec=SeriesSpec(source="yfinance", series_id="^VIX"), label="VIX")

        out, band = _render_single(business, route=route, ceiling=date(2026, 6, 30), calendar_days=90)

        assert band is not None
        assert "all 62-trading-day change windows" in out  # round(90 * 252/365)
        y = business.to_numpy(dtype="float64")
        expected = _empirical_change_band(y, 62, use_log=True, anchor=float(business.iloc[-1]))
        assert band == pytest.approx(expected)

    def test_chart_ribbon_and_band_agree_on_the_calendar_horizon(self, monkeypatch):
        # The chart path has its own copy of the freq -> h -> horizon_end walk, so it gets its
        # own guard: on a 24/7 series the ribbon must end at as_of + calendar_days AND the band
        # it draws must be the calendar-step band (pre-fix the ribbon was right by cancellation
        # while the band was the 62-step one).
        continuous = _twenty_four_seven_series("BTC-USD", end="2026-03-15")
        as_of = datetime(2026, 3, 15, tzinfo=UTC)
        resolves = datetime(2026, 6, 13, tzinfo=UTC)  # 90 calendar days out
        calendar_days = (resolves.date() - as_of.date()).days
        captured: dict[str, object] = {}

        monkeypatch.setenv("TS_ANCHOR_CHART_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: continuous)
        monkeypatch.setattr(
            "metaculus_bot.research.ts_chart.render_anchor_chart",
            lambda *_a, **kw: captured.update(kw) or "FAKE_PNG_BASE64",
        )
        q = _make_numeric_q(
            qid=9317,
            resolution_criteria="Resolves per https://finance.yahoo.com/quote/BTC-USD/history/",
            scheduled_resolution_time=resolves,
            lower_bound=0.0,
            upper_bound=10000.0,
        )

        out = build_anchor_section(q, as_of)

        assert out
        assert ts._session_charts.get(9317) == "FAKE_PNG_BASE64"
        assert captured["horizon_end"] == pd.Timestamp("2026-03-15") + pd.Timedelta(days=calendar_days)
        y = continuous.to_numpy(dtype="float64")
        expected = _empirical_change_band(y, calendar_days, use_log=True, anchor=float(continuous.iloc[-1]))
        assert captured["band"] == pytest.approx(expected)

    def test_spread_band_reads_the_density_of_the_JOINED_series(self):
        # `_render_spread` has its own freq -> h walk, on the INNER-JOINED index. Two 24/7 legs
        # join to a calendar-daily series, so the horizon is the calendar window; a mixed
        # 24/7-plus-exchange pair joins down to business days and keeps the 252 basis. The
        # shipped code used 252 unconditionally, so it was right only by luck on mixed pairs.
        leg_a = _twenty_four_seven_series("BTC-USD", seed=1)
        leg_b = _twenty_four_seven_series("ETH-USD", seed=2)
        route = _Route(
            kind="spread",
            spec=SeriesSpec(source="yfinance", series_id="BTC-USD"),
            label="BTC-USD",
            spec_b=SeriesSpec(source="yfinance", series_id="ETH-USD"),
            label_b="ETH-USD",
        )

        out, _band = _render_spread(leg_a, leg_b, route=route, calendar_days=90)

        assert "Forward 90-calendar-day relative-return band" in out

        mixed_out, _ = _render_spread(leg_a, _daily_positive_series("^GSPC", seed=3), route=route, calendar_days=90)
        assert "Forward 62-trading-day relative-return band" in mixed_out


# Row-wise month-over-month derivations are only "month-over-month" when one row is one month.
def _quarterly_series(name: str, *, seed: int = 0, end: str = "2026-04-01", n: int = 60) -> pd.Series:
    """A strictly-positive quarterly (quarter-start) series — the GDPC1 shape."""
    idx = pd.date_range(end=pd.Timestamp(end), periods=n, freq="QS")
    rng = np.random.default_rng(seed)
    walk = 20000.0 + np.cumsum(rng.normal(50.0, 120.0, n))
    return pd.Series(np.abs(walk) + 500.0, index=idx, name=name)


def _semiannual_series(name: str, *, seed: int = 0, n: int = 40) -> pd.Series:
    """~183-day cadence — the crack between the quarterly and annual buckets."""
    idx = pd.date_range(end=pd.Timestamp("2026-04-01"), periods=n, freq="183D")
    rng = np.random.default_rng(seed)
    walk = 100.0 + np.cumsum(rng.normal(0.0, 2.0, n))
    return pd.Series(np.abs(walk) + 10.0, index=idx, name=name)


class TestCoarseCadenceClocks:
    """Quarterly/annual series get real buckets with calendar-honest steps.

    Before them, `_detect_freq` topped out at "monthly": a quarterly series (GDPC1, ~92-day
    gaps) classified monthly and every horizon converted at 30.4375 days per step — a 3x-too-
    wide band under false "monthly" labels ("series frequency: monthly", "Last monthly
    observations", "3-month change windows" for what is a 3-QUARTER band), reachable through
    any coarse FRED series cited by URL (`_single_url_route` builds a model-target level route
    for every unregistered id). The same row-count-as-calendar defect class as the 24/7
    sqrt(252), in the low-frequency direction.
    """

    def test_quarterly_and_annual_series_get_their_own_buckets(self):
        quarterly = _quarterly_series("GDPC1")
        annual = pd.Series(
            np.linspace(100.0, 200.0, 30),
            index=pd.date_range(end="2026-01-01", periods=30, freq="YS"),
        )
        q_clock = series_clock(pd.DatetimeIndex(quarterly.index))
        a_clock = series_clock(pd.DatetimeIndex(annual.index))
        assert (q_clock.freq, q_clock.step_unit) == ("quarterly", "quarter")
        assert (a_clock.freq, a_clock.step_unit) == ("annual", "year")

    def test_quarterly_horizons_convert_on_calendar_quarters(self):
        clock = series_clock(pd.DatetimeIndex(_quarterly_series("GDPC1").index))
        # A 90-day question is ONE quarterly step (the monthly bucket read it as 3 steps —
        # a 276-day band presented as 90 days); a year is four.
        assert horizon_steps(clock, 90) == 1
        assert horizon_steps(clock, 365) == 4

    def test_horizon_end_date_inverts_on_the_coarse_clocks_too(self):
        as_of = pd.Timestamp("2026-06-30")
        quarterly = series_clock(pd.DatetimeIndex(_quarterly_series("GDPC1").index))
        assert abs((_horizon_end_date(as_of, quarterly, 1) - as_of).days - 91) <= 3
        annual = SeriesClock(freq="annual", periods_per_year=TRADING_DAYS_PER_YEAR)
        assert abs((_horizon_end_date(as_of, annual, 1) - as_of).days - 365) <= 1

    def test_gdpc1_shape_renders_a_quarter_band_under_honest_labels(self, monkeypatch):
        """End-to-end through the URL branch (the verified exposure): a quarterly FRED series
        must render quarter-labelled history and a quarter-step band, not the monthly trio of
        false statements."""
        quarterly = _quarterly_series("GDPC1")
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: quarterly)
        q = _make_numeric_q(
            qid=9451,
            resolution_criteria="Resolves per https://fred.stlouisfed.org/series/GDPC1 for the quarter.",
            scheduled_resolution_time=datetime(2026, 9, 28, tzinfo=UTC),
            lower_bound=0.0,
            upper_bound=100000.0,
        )

        out = build_anchor_section(q, datetime(2026, 6, 30, tzinfo=UTC))

        assert "series frequency: quarterly" in out
        assert "Last quarterly observations" in out
        assert "-quarter change windows" in out
        assert "monthly" not in out

    def test_a_cadence_the_buckets_misdescribe_is_refused_a_band(self, monkeypatch, caplog):
        """The fail-safe for the cracks: a ~183-day semiannual series lands in the annual
        bucket, where one 365.25-day step spans TWO real observations — a band too narrow by
        the same factor. `clock_matches_cadence` refuses it and the band-None guard drops the
        section rather than serving a mis-converted quantity."""
        semiannual = _semiannual_series("BOGUS1")
        clock = series_clock(pd.DatetimeIndex(semiannual.index))
        assert clock.freq == "annual"
        assert clock_matches_cadence(clock, pd.DatetimeIndex(semiannual.index)) is False

        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: semiannual)
        q = _make_numeric_q(
            qid=9452,
            resolution_criteria="Resolves per https://fred.stlouisfed.org/series/BOGUS1 on the date.",
            scheduled_resolution_time=datetime(2026, 9, 28, tzinfo=UTC),
        )

        with caplog.at_level(logging.WARNING):
            out = build_anchor_section(q, datetime(2026, 6, 30, tzinfo=UTC))

        assert out == ""
        assert any("cadence" in record.getMessage() for record in caplog.records)

    def test_matching_cadences_all_pass_the_guard(self):
        for series in (
            _daily_positive_series("^GSPC"),
            _twenty_four_seven_series("BTC-USD"),
            _monthly_series("CPIAUCSL"),
            _quarterly_series("GDPC1"),
        ):
            index = pd.DatetimeIndex(series.index)
            assert clock_matches_cadence(series_clock(index), index) is True, series.name


class TestAnchorRealizedVolNoiseFlag:
    """The anchor's own one-day-return volatility line carries the same noise screen.

    This is the second place the bot annualizes ONE-step returns for a forecaster, and the
    anchor routes to any URL-cited Yahoo ticker — so q44797's pegged cross would render an
    equally inflated figure here. The change BAND is a different story: it is built from
    h-step changes at h of tens of observations, where independent quote noise contributes
    ~2/h of the variance, so the flag must say the band is unaffected rather than casting
    doubt on the whole section.
    """

    @staticmethod
    def _clock() -> SeriesClock:
        return SeriesClock(freq="daily", periods_per_year=TRADING_DAYS_PER_YEAR)

    def test_a_clean_series_renders_one_unqualified_line(self):
        lines = tsrender._realized_vol_lines(random_walk_close_series(seed=3), self._clock(), symbol="CSUSHPISA")

        assert len(lines) == 1
        assert lines[0].startswith("- 30-trading-day annualized realized volatility:")
        assert "noise-suspect" not in lines[0]

    def test_a_noise_dominated_series_flags_and_offers_the_robust_figure(self):
        noisy = noise_dominated_close_series(seed=3)
        robust = multi_period_annualized_vol_pct(
            noisy,
            lag=FINANCIAL_VARIANCE_RATIO_LAG,
            periods_per_year=TRADING_DAYS_PER_YEAR,
            min_returns=FINANCIAL_VARIANCE_RATIO_MIN_RETURNS,
        )
        assert robust is not None

        lines = tsrender._realized_vol_lines(noisy, self._clock(), symbol="CSUSHPISA")

        assert len(lines) == 2
        assert lines[0].endswith("— noise-suspect")
        assert "Vendor-noise flag" in lines[1]
        assert f"the volatility is {robust:.1f}%" in lines[1]
        assert "The change band above is unaffected" in lines[1]

    def test_the_emitted_marker_line_parses_under_the_spec(self, caplog: pytest.LogCaptureFixture):
        """This surface had NO emitter-side assertion at all — deleting its `logger.info` was
        green, and so was any field reorder, while the archive harvested nothing. The parser
        tests could not catch it: they parse hand-typed strings whose comments only claim to
        match what the emitter produces. Same pattern as test_ts_routing's ts_anchor_route test.

        `long_vol` reads None here because the anchor computes no long-horizon window; the
        field is present because both surfaces share one emitter and one shape.
        """
        noisy = noise_dominated_close_series(seed=3)
        clock = self._clock()
        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.ts_render"):
            tsrender._realized_vol_lines(noisy, clock, symbol="CSUSHPISA")

        ratio = variance_ratio(
            noisy, lag=FINANCIAL_VARIANCE_RATIO_LAG, min_returns=FINANCIAL_VARIANCE_RATIO_MIN_RETURNS
        )
        robust = multi_period_annualized_vol_pct(
            noisy,
            lag=FINANCIAL_VARIANCE_RATIO_LAG,
            periods_per_year=clock.periods_per_year,
            min_returns=FINANCIAL_VARIANCE_RATIO_MIN_RETURNS,
        )
        short = annualized_realized_vol_pct(
            noisy, window=tsrender.REALIZED_VOL_WINDOW, periods_per_year=clock.periods_per_year
        )
        assert ratio is not None
        assert robust is not None
        assert short is not None

        (line,) = [r.getMessage() for r in caplog.records if "FINANCIAL_NOISE_FLAG" in r.getMessage()]
        spec = next(s for s in MARKER_SPECS if s.name == "financial_noise_flag")
        match = spec.regex.search(line)
        assert match is not None
        assert match.groupdict() == {
            "surface": "ts_anchor",
            "symbol": "CSUSHPISA",
            "vr_lag": str(FINANCIAL_VARIANCE_RATIO_LAG),
            "vr": f"{ratio:.3f}",
            "floor": str(FINANCIAL_VARIANCE_RATIO_FLOOR),
            "short_vol": f"{short:.1f}",
            "long_vol": "None",
            "robust_vol": str(round(robust, 1)),
        }

    def test_a_short_series_renders_nothing(self):
        short = random_walk_close_series(seed=3, n=20)
        assert tsrender._realized_vol_lines(short, self._clock(), symbol="CSUSHPISA") == []


class TestAnchorValueFormatter:
    """`_fmt` renders the anchor's levels, history rows, 52-week range and band quantiles.

    It was the last `.4g`-class precision sibling: `:,.1f` between 100 and 10,000 rounded a
    Case-Shiller level of 331.893 to "331.9" on a question with 0.02-point buckets, and
    `:,.0f` above that dropped an index level's decimals entirely — while the FRED block in
    the OTHER provider rendered the same observation at full precision in the same bundle
    (q44944 carried both providers, with true prints 331.020 / 331.172 / 332.105 / 327.462).
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (331.893, "331.893"),  # the motivating case: was "331.9"
            (334.5123456, "334.512"),  # capped at three decimals, not six
            (20150.55, "20,150.55"),  # was "20,151" — decimals dropped above 10,000
            (6699580.0, "6,699,580"),  # WALCL: thousands-separated, never scientific
            (1200.0, "1,200"),  # rstrip halts at the "." — integer zeros survive
            (100.0, "100"),  # the branch boundary
            (-331.893, "-331.893"),  # sign does not change the branch
            (4.2, "4.2"),  # below 100 keeps 4 significant figures
            (0.00012345, "0.0001234"),  # a MoM % change must not round to "0.000"
        ],
    )
    def test_the_four_magnitude_bands(self, value: float, expected: str) -> None:
        assert tsrender._fmt(value) == expected


class TestSharedVolEstimator:
    """The one vol definition (`annualized_realized_vol_pct`), after the q44882 defect was
    fixed in one of its two byte-identical copies weeks before the other."""

    def test_matches_the_hand_computed_value(self):
        series = _twenty_four_seven_series("BTC-USD")
        expected = float(series.pct_change().dropna().tail(30).std() * np.sqrt(365) * 100.0)
        assert annualized_realized_vol_pct(series, window=30, periods_per_year=365) == pytest.approx(expected)

    def test_returns_none_below_the_window(self):
        # 30 closes yield 29 returns — a "30-observation" vol computed on 29 would wear the
        # window's label without its sample size.
        short = pd.Series(
            np.linspace(100.0, 110.0, 30),
            index=pd.date_range(end="2026-06-30", periods=30, freq="D"),
        )
        assert annualized_realized_vol_pct(short, window=30, periods_per_year=365) is None


class TestVarianceRatio:
    """`variance_ratio` is the vendor-noise screen behind financial_data's noise flag.

    q44797 handed six forecasters a 17.8% "volatility" computed on a pegged cross whose
    daily returns were 79% quote noise; the honest like-for-like figure off the liquid
    anchor was ~10.6%. The screen has to separate those two regimes at the sample size the
    provider actually holds (~265 daily bars) and stay silent on a clean series, since a
    false flag would demote a perfectly good short-window volatility.
    """

    def test_clean_random_walk_reads_near_one_and_clears_the_floor(self):
        clean = random_walk_close_series(seed=3)
        ratio = variance_ratio(
            clean,
            lag=FINANCIAL_VARIANCE_RATIO_LAG,
            min_returns=FINANCIAL_VARIANCE_RATIO_MIN_RETURNS,
        )
        assert ratio is not None
        # A walk's null value is 1; the null sd of VR(5) at n~264 is ~0.13, so anything in
        # this band is ordinary sampling noise and must NOT fire the flag.
        assert 0.7 <= ratio <= 1.3
        assert ratio > FINANCIAL_VARIANCE_RATIO_FLOOR

    def test_quote_noise_dominated_series_falls_below_the_floor(self):
        noisy = noise_dominated_close_series(seed=3)
        ratio = variance_ratio(
            noisy,
            lag=FINANCIAL_VARIANCE_RATIO_LAG,
            min_returns=FINANCIAL_VARIANCE_RATIO_MIN_RETURNS,
        )
        assert ratio is not None
        # iid quote noise contributing a fraction p of return variance drives VR(5) to
        # 1 - 0.8p; at p ~ 2/3 that is ~0.47, the value the 44797 verification measured on
        # USDSZL=X. The floor must sit between this and the clean case above.
        assert ratio == pytest.approx(0.47, abs=0.12)
        assert ratio < FINANCIAL_VARIANCE_RATIO_FLOOR

    def test_the_floor_operating_point_across_twenty_seeds(self):
        """One seed proves nothing about a threshold, and the two distributions genuinely
        overlap in the tails, so the honest claim is an operating point rather than perfect
        separation. Measured over seeds 0-19 (scratch/next_season_bundle_2026-09/item10/
        vr_calibration.py): clean VR mean 1.005, min 0.636; noisy VR mean 0.465, max 0.617.
        At the 0.60 floor that is 0 of 20 clean series flagged and 18 of 20 noisy ones
        caught. A false positive costs a demoted-but-still-printed short-window figure; a
        false negative reproduces q44797, so the asymmetry favours keeping the floor high
        enough to catch most noise and low enough to never flag a walk."""
        flagged_clean = 0
        flagged_noisy = 0
        for seed in range(20):
            clean = variance_ratio(
                random_walk_close_series(seed=seed),
                lag=FINANCIAL_VARIANCE_RATIO_LAG,
                min_returns=FINANCIAL_VARIANCE_RATIO_MIN_RETURNS,
            )
            noisy = variance_ratio(
                noise_dominated_close_series(seed=seed),
                lag=FINANCIAL_VARIANCE_RATIO_LAG,
                min_returns=FINANCIAL_VARIANCE_RATIO_MIN_RETURNS,
            )
            assert clean is not None
            assert noisy is not None
            flagged_clean += clean < FINANCIAL_VARIANCE_RATIO_FLOOR
            flagged_noisy += noisy < FINANCIAL_VARIANCE_RATIO_FLOOR
        assert flagged_clean == 0, "a clean random walk must never be called vendor noise"
        assert flagged_noisy >= 17, f"the screen caught only {flagged_noisy}/20 noise-dominated series"

    def test_the_thirty_row_vol_window_is_refused_not_estimated(self):
        """The 44797 verification's §11 constraint: the null sd of VR(5) is ~0.40 at n=30,
        so the statistic is uninformative on the window the volatility line uses. Refusing
        it is the only honest answer — a number there would read as measurement."""
        thirty_rows = noise_dominated_close_series(seed=1, n=31)
        assert (
            variance_ratio(
                thirty_rows,
                lag=FINANCIAL_VARIANCE_RATIO_LAG,
                min_returns=FINANCIAL_VARIANCE_RATIO_MIN_RETURNS,
            )
            is None
        )

    def test_a_momentum_series_reads_above_one(self):
        """Sign check in the other direction. Returns with positive autocorrelation (rho =
        0.3) persist rather than reverse, so VR(5) sits above 1 — around 1.6 by
        1 + (2/q)*sum_k (q-k)*rho^k. Confirms the statistic is measuring return
        autocorrelation and not just any departure from a straight line."""
        rng = np.random.default_rng(5)
        shocks = rng.normal(0.0, 0.006, 300)
        returns = np.zeros(300)
        for i in range(1, 300):
            returns[i] = 0.3 * returns[i - 1] + shocks[i]
        momentum = pd.Series(
            16.4 * np.exp(np.cumsum(returns)),
            index=pd.bdate_range(end="2026-07-17", periods=300),
        )
        ratio = variance_ratio(momentum, lag=5, min_returns=FINANCIAL_VARIANCE_RATIO_MIN_RETURNS)
        assert ratio is not None
        assert ratio > 1.2

    def test_hand_computed_ratio_on_a_short_deterministic_series(self):
        """VR(2) = 1.5375 on log prices [0, 1, 2, 4, 7, 11, 16], by hand.

        Every other numeric assertion on this estimator is a tolerance band on a seeded
        pseudo-random series, and both it and `multi_period_annualized_vol_pct` read their
        multi-period returns from the same helper — so the sqrt(VR) identity between them is
        algebraically vacuous on the lag window, and a consistent slip in the window shifts
        both together. Four plausible mutations of `_overlapping_log_return_sample` left the
        whole suite green: a lag off-by-one, numpy's default ddof=0 on the multi variance, a
        NON-overlapping window (`lp[lag::lag]`), and simple instead of log returns.

        The arithmetic, all exact in binary except the final division:
          1-step log returns [1, 1, 2, 3, 4, 5], mean 8/3, sum of squared deviations 120/9,
            so var(ddof=1) = 120/45 = 8/3.
          2-step log returns [2, 3, 5, 7, 9], mean 26/5, sum of squared deviations 32.8,
            so var(ddof=1) = 8.2.
          VR(2) = 8.2 / (2 * 8/3) = 1.5375.
        A non-overlapping window gives 2.3125, ddof=0 gives 1.23.
        """
        series = pd.Series(np.exp(np.array([0.0, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0])))
        assert variance_ratio(series, lag=2, min_returns=6) == pytest.approx(8.2 / (2 * 8 / 3))

    def test_an_alternating_series_reads_exactly_zero_at_the_reversal_lag(self):
        """Lag sensitivity, which the series above lacks (it is nearly lag-invariant).

        On log prices [0, 1, 0, 1, ...] every 2-step return is zero — perfect mean reversion
        — so VR(2) is 0, while the 3-step returns are [1, -1, 1, -1, 1, -1] with
        var(ddof=1) = 6/5 = 1.2 against a 1-step var of 8/7, giving VR(3) = 1.2 / (3 * 8/7) =
        0.35. A lag off-by-one flips the first value from 0 to 1.0, which no tolerance band on
        a random series would have caught.
        """
        series = pd.Series(np.exp(np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])))
        # Tolerance rather than `== 0.0`: the 2-step differences are zero only because
        # log(exp(x)) round-trips exactly here, which is a libm property, not arithmetic.
        assert variance_ratio(series, lag=2, min_returns=6) == pytest.approx(0.0, abs=1e-12)
        assert variance_ratio(series, lag=3, min_returns=6) == pytest.approx(1.2 / (3 * 8 / 7))

    def test_hand_computed_multi_period_volatility_on_the_same_series(self):
        """`multi_period_annualized_vol_pct` = sd(lag-step) / sqrt(lag) * sqrt(252) * 100.

        On the same log prices the 2-step returns have var(ddof=1) = 8.2, so the annualized
        figure is sqrt(8.2 / 2 * 252) * 100 = 3214.3428... — an absurd volatility, which is
        the point: the series is a synthetic ramp chosen so the arithmetic is checkable by
        hand rather than a tolerance band on a random walk. This is the number the flagged
        block promotes to the headline and tells a forecaster to size intervals from, so it
        needs one assertion that is not the vacuous sqrt(VR) identity.
        """
        series = pd.Series(np.exp(np.array([0.0, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0])))
        vol = multi_period_annualized_vol_pct(series, lag=2, periods_per_year=252, min_returns=6)
        assert vol == pytest.approx(math.sqrt(8.2 / 2 * 252) * 100)

    def test_a_constant_step_series_returns_none_rather_than_float_noise(self):
        """A series with no measurable return variation — an administratively fixed quote,
        or a constant-log-step ramp — makes the ratio a quotient of float rounding noise. An
        exact ramp read 0.369 before the degeneracy guard, i.e. a confident noise flag
        manufactured out of the last bits of two mantissas."""
        ramp = pd.Series(
            100.0 * np.exp(np.linspace(0.0, 0.5, 300)),
            index=pd.bdate_range(end="2026-07-17", periods=300),
        )
        assert variance_ratio(ramp, lag=5, min_returns=FINANCIAL_VARIANCE_RATIO_MIN_RETURNS) is None

    def test_a_non_positive_value_returns_none_rather_than_a_nan(self):
        """Log returns are undefined at or below zero (a spread series crosses it), and a
        nan ratio compares False against the floor — a silent no-flag."""
        crosses_zero = pd.Series(
            np.linspace(-1.0, 1.0, 200),
            index=pd.bdate_range(end="2026-07-17", periods=200),
        )
        assert variance_ratio(crosses_zero, lag=5, min_returns=120) is None

    def test_a_flat_series_returns_none_rather_than_dividing_by_zero(self):
        flat = pd.Series(
            np.full(200, 5.0),
            index=pd.bdate_range(end="2026-07-17", periods=200),
        )
        assert variance_ratio(flat, lag=5, min_returns=120) is None


class TestDerivationFrequencyInvariant:
    def test_mom_derivations_reject_a_non_monthly_source(self):
        weekly = pd.Series(
            np.linspace(3.0, 4.0, 60),
            index=pd.date_range(end="2026-06-29", periods=60, freq="W"),
            name="GASREGW",
        )
        for derivation in ("mom_diff", "mom_pct"):
            with pytest.raises(ValueError, match="must be one month"):
                _apply_derivation(weekly, derivation, 1.0)

    def test_mom_derivations_accept_a_monthly_source(self):
        monthly = _monthly_series("PAYEMS", n=48)
        assert len(_apply_derivation(monthly, "mom_diff", 1000.0)) == 47
        assert len(_apply_derivation(monthly, "mom_pct", 1.0)) == 47

    def test_monthly_avg_resamples_before_any_row_wise_step(self):
        # The weekly -> monthly derivation is calendar-based (resample), so it is exempt from the
        # invariant above and must still work on the weekly series that rejected mom_*.
        weekly = pd.Series(
            np.linspace(3.0, 4.0, 60),
            index=pd.date_range(end="2026-06-29", periods=60, freq="W"),
            name="GASREGW",
        )
        out = _apply_derivation(weekly, "monthly_avg", 1.0)
        assert _detect_freq(pd.DatetimeIndex(out.index)) == "monthly"

    @pytest.mark.asyncio
    async def test_a_non_monthly_source_soft_fails_the_section_not_the_run(self, monkeypatch, caplog):
        """The invariant's whole point is that it withholds a mislabelled quantity — so it has to
        reach the provider as a soft-fail, not as an exception that takes the research fan-out
        down. Wired through the real caller: a MoM CPI question citing the CPIAUCSL URL routes to
        mom_pct, and the fetch hands back a WEEKLY-cadence frame (the hazard shape — a registry
        entry declaring mom_pct whose source series is not monthly). The provider catches
        ValueError, so this pins the invariant against being "fixed" into a bare crash."""
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: _weekly_series("CPIAUCSL"))
        question = _make_numeric_q(
            qid=9911,
            question_text="What will be the month-over-month percent change in US headline CPI for December 2026?",
            resolution_criteria="Resolves per https://fred.stlouisfed.org/series/CPIAUCSL on the release date.",
            lower_bound=-1.0,
            upper_bound=2.0,
        )

        with caplog.at_level(logging.WARNING):
            result = await timeseries_anchor_provider()(question)

        assert result == ""
        assert any("soft-fail for qid=9911" in r.message and "ValueError" in r.message for r in caplog.records)


# Provider factory (flag gating, benchmark ceiling, soft-fail, determinism).
class TestProviderFactory:
    @pytest.mark.asyncio
    async def test_disabled_flag_returns_empty_even_when_routable(self, monkeypatch):
        """Env-flag gate: with TS_ANCHOR_ENABLED unset the provider short-circuits to ""
        WITHOUT touching fetch_series, even for a cleanly-routable question."""
        monkeypatch.delenv("TS_ANCHOR_ENABLED", raising=False)
        fetch_spy = MagicMock(side_effect=AssertionError("fetch_series must not run when disabled"))
        monkeypatch.setattr(ts, "fetch_series", fetch_spy)

        provider = timeseries_anchor_provider()
        result = await provider(_make_numeric_q(resolution_criteria=_DGS10_RC))

        assert result == ""
        fetch_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_numeric_question_returns_empty(self, monkeypatch):
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        binary_q = MagicMock(spec=BinaryQuestion)
        binary_q.id_of_question = 9
        binary_q.resolution_criteria = _DGS10_RC

        provider = timeseries_anchor_provider()
        assert await provider(binary_q) == ""

    @pytest.mark.asyncio
    async def test_enabled_flag_routes_fetches_and_renders(self, monkeypatch):
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: _daily_positive_series("DGS10"))

        provider = timeseries_anchor_provider()
        out = await provider(_make_numeric_q(resolution_criteria=_DGS10_RC))

        assert isinstance(out, str)
        assert out  # non-empty section
        # DGS10 is a registry entry, so URL routing carries the registry's descriptive
        # label (F10 fix) rather than the bare series_id.
        assert out.splitlines()[0].startswith("**10-Year Treasury constant-maturity yield (%)** — latest ")
        assert "P10 / P50 / P90 →" in out

    @pytest.mark.asyncio
    async def test_is_benchmarking_uses_open_time_as_ceiling(self, monkeypatch):
        """Backtest-safe path: is_benchmarking=True does NOT short-circuit (unlike
        prediction_market) — it pins the fetch ceiling to question.open_time so series
        data known at forecast time IS the answer without leaking the resolution."""
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        captured_ceilings: list[date] = []

        def _capturing_fetch(spec, ceiling, **_kwargs):
            captured_ceilings.append(ceiling)
            return _daily_positive_series("DGS10", end="2026-03-10")

        monkeypatch.setattr(ts, "fetch_series", _capturing_fetch)

        open_time = datetime(2026, 3, 15, tzinfo=UTC)
        q = _make_numeric_q(resolution_criteria=_DGS10_RC, open_time=open_time)

        provider = timeseries_anchor_provider(is_benchmarking=True)
        out = await provider(q)

        assert out  # still ran (not short-circuited)
        assert captured_ceilings == [open_time.date()]  # ceiling pinned to open_time

    @pytest.mark.asyncio
    async def test_env_flag_gate_precedes_is_benchmarking_logic(self, monkeypatch):
        """Ordering mirror: the env-flag gate is evaluated BEFORE the is_benchmarking
        as_of branch, so a disabled flag returns "" without ever reading open_time."""
        monkeypatch.delenv("TS_ANCHOR_ENABLED", raising=False)
        fetch_spy = MagicMock(side_effect=AssertionError("must not fetch when flag disabled"))
        monkeypatch.setattr(ts, "fetch_series", fetch_spy)

        # open_time deliberately absent — if the is_benchmarking branch ran first it would
        # log a warning; the flag gate must return "" before that.
        q = _make_numeric_q(resolution_criteria=_DGS10_RC)
        q.open_time = None

        provider = timeseries_anchor_provider(is_benchmarking=True)
        assert await provider(q) == ""
        fetch_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_fetch_soft_fails_to_empty_with_warning(self, monkeypatch, caplog):
        """A genuine fetch/data error (here: HTML instead of CSV) soft-fails to "" + WARNING;
        it never raises out of the provider."""
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        monkeypatch.setattr(tf, "_http_get", FakeHttp({FRED_CSV_URL: b"<html>bad series id</html>"}))

        provider = timeseries_anchor_provider()
        with caplog.at_level(logging.WARNING):
            result = await provider(_make_numeric_q(resolution_criteria=_DGS10_RC))

        assert result == ""
        assert any("soft-fail" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_deterministic_output_across_calls(self, monkeypatch):
        """Same question + same series -> byte-identical section. Reset caches between
        the two calls so the second recomputes rather than reading the section cache."""
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: _daily_positive_series("DGS10"))
        q = _make_numeric_q(resolution_criteria=_DGS10_RC)

        provider = timeseries_anchor_provider()
        first = await provider(q)
        _reset_session_caches()
        second = await provider(q)

        assert first == second
        assert first  # not the empty soft-fail

    @pytest.mark.asyncio
    async def test_leaky_fetch_soft_fails_to_empty(self, monkeypatch, caplog):
        """A post-ceiling row triggers the fetch layer's LeakageError, which the provider
        catches and soft-fails to "" — the render never reflects the leaked observation."""
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        # The provider (live) uses as_of=now; a 2099 row is unambiguously post-ceiling.
        rows = [("2026-06-01", "4.20"), ("2099-01-01", "9.99")]
        monkeypatch.setattr(tf, "_http_get", FakeHttp({FRED_CSV_URL: _csv("DGS10", rows)}))

        provider = timeseries_anchor_provider()
        with caplog.at_level(logging.WARNING):
            result = await provider(_make_numeric_q(resolution_criteria=_DGS10_RC))

        assert result == ""
        assert "9.99" not in result

    @pytest.mark.asyncio
    async def test_missing_scheduled_resolution_time_returns_empty(self, monkeypatch):
        """build_anchor_section needs a real scheduled_resolution_time to size the horizon;
        without one it returns "" before fetching."""
        monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
        fetch_spy = MagicMock(side_effect=AssertionError("must not fetch without a horizon"))
        monkeypatch.setattr(ts, "fetch_series", fetch_spy)

        q = _make_numeric_q(resolution_criteria=_DGS10_RC, scheduled_resolution_time=None)
        provider = timeseries_anchor_provider()

        assert await provider(q) == ""
        fetch_spy.assert_not_called()


# Bounds-overlap backstop: a rendered P10-P90 band lying ENTIRELY outside the question's
# displayed range is a gross units/magnitude mismatch (level-vs-derived, wrong country), so
# build_anchor_section drops the section rather than feed a wrong-units anchor to the
# forecasters. Open / non-finite bounds impose no constraint on that side.
class TestBoundsBackstop:
    def test_band_none_never_misses(self):
        # No band rendered (not a model target, or horizon exceeds history) -> nothing to check.
        assert _band_misses_bounds(_make_numeric_q(lower_bound=0.0, upper_bound=1.0), None) is False

    def test_band_below_closed_bounds_misses(self):
        # The canonical bug shape: a ~0.3 band (MoM-%) vs an index-level range [250, 350].
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0)
        assert _band_misses_bounds(q, (0.1, 0.3, 0.5)) is True

    def test_band_above_closed_bounds_misses(self):
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0)
        assert _band_misses_bounds(q, (400.0, 450.0, 500.0)) is True

    def test_band_overlapping_bounds_does_not_miss(self):
        q = _make_numeric_q(lower_bound=0.0, upper_bound=1.0)
        assert _band_misses_bounds(q, (0.1, 0.3, 0.5)) is False

    def test_open_lower_bound_loosens_but_does_not_remove_the_constraint(self):
        # An OPEN lower edge means the outcome CAN settle below it, so the constraint has to
        # loosen — but mapping the edge to -inf removed it entirely, which disarmed the whole
        # backstop on the ~95% of real numeric questions that carry two open bounds. A
        # ~0.3-magnitude band against an index-level range is still the canonical wrong-units
        # shape whether or not the edge is open.
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0, open_lower_bound=True)
        assert _band_misses_bounds(q, (0.1, 0.3, 0.5)) is True

    def test_open_upper_bound_loosens_but_does_not_remove_the_constraint(self):
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0, open_upper_bound=True)
        assert _band_misses_bounds(q, (400.0, 450.0, 500.0)) is True

    def test_band_just_beyond_an_open_edge_is_tolerated(self):
        # The other half of "loosens": a band sitting a little past an OPEN edge is a normal
        # forecast (the outcome really can settle out there), not a units error. Only a band
        # clear of the range by more than the tolerance margin is suppressed.
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0, open_lower_bound=True)
        assert _band_misses_bounds(q, (240.0, 245.0, 249.0)) is False

    def test_band_just_beyond_a_closed_edge_still_misses(self):
        # A CLOSED edge means the outcome cannot settle beyond it, so no tolerance is granted
        # there and the original zero-overlap rule stands unchanged.
        q = _make_numeric_q(lower_bound=250.0, upper_bound=350.0)
        assert _band_misses_bounds(q, (240.0, 245.0, 249.0)) is True

    def test_non_finite_bounds_impose_no_constraint(self):
        # A non-finite edge carries no span to scale a tolerance against, so it stays a
        # genuine no-constraint case (unlike an open-but-finite edge).
        q = _make_numeric_q(lower_bound=float("-inf"), upper_bound=float("inf"))
        assert _band_misses_bounds(q, (0.1, 0.3, 0.5)) is False

    def test_wrong_quantity_band_on_a_two_open_bound_question_is_caught(self):
        # qid 44868 ("the NOB spread", resolves in basis points around 53, displayed
        # [29.5, 70.5] with BOTH bounds open). These are the two wrong-quantity bands a
        # single-leg or unscaled route would produce. Pre-fix both returned False — the exact
        # hole that left the 548ba88 magnitude guard unable to police open-bound questions.
        q = _make_numeric_q(lower_bound=29.5, upper_bound=70.5, open_lower_bound=True, open_upper_bound=True)
        assert _band_misses_bounds(q, (4.40, 4.68, 4.95)) is True  # single-leg 10-year LEVEL, in percent
        assert _band_misses_bounds(q, (0.35, 0.53, 0.72)) is True  # spread in percent, not basis points

    def test_correct_quantity_band_on_the_same_question_is_kept(self):
        # Same question, the CORRECT basis-point spread band -> must not be suppressed.
        q = _make_numeric_q(lower_bound=29.5, upper_bound=70.5, open_lower_bound=True, open_upper_bound=True)
        assert _band_misses_bounds(q, (35.0, 53.0, 72.0)) is False

    @pytest.mark.parametrize(
        ("qid", "band", "lower_bound", "upper_bound"),
        [
            # Every band the anchor actually published in prod, read verbatim from the
            # archived `## Time Series Anchor` sections, with that question's real bounds.
            # All four carry two OPEN bounds, so they are precisely the population the
            # tolerance margin newly constrains — a too-tight threshold silently suppresses
            # healthy anchors, and silent suppression is the failure direction that would go
            # unnoticed. q44944's band sits partly above its own stale displayed range, which
            # is what makes it the binding case rather than a decorative one.
            (44943, (3.992, 4.200, 4.312), 3.9, 4.5),
            (44944, (330.9, 332.3, 335.4), 328.0, 332.0),
            (45114, (4.194, 4.336, 4.535), 4.2, 4.8),
            (45172, (18.70, 22.49, 30.83), 15.0, 25.0),
        ],
    )
    def test_real_published_bands_are_not_suppressed(
        self, qid: int, band: tuple[float, float, float], lower_bound: float, upper_bound: float
    ):
        q = _make_numeric_q(
            qid=qid,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            open_lower_bound=True,
            open_upper_bound=True,
        )
        assert _band_misses_bounds(q, band) is False

    def test_build_anchor_section_skips_on_bounds_mismatch(self, monkeypatch, caplog):
        # End-to-end: a ~0.3-magnitude series routed onto a level question with an
        # index-level range [250, 350] -> the section is dropped with a WARN.
        flat = pd.Series(
            np.full(400, 0.3, dtype="float64"),
            index=pd.bdate_range("2024-01-01", periods=400),
            name="DGS10",
        )
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: flat)
        q = _make_numeric_q(resolution_criteria=_DGS10_RC, lower_bound=250.0, upper_bound=350.0)

        with caplog.at_level(logging.WARNING):
            out = build_anchor_section(q, datetime(2026, 3, 15, tzinfo=UTC))

        assert out == ""
        assert any("zero overlap with question bounds" in r.message for r in caplog.records)

    def test_build_anchor_section_renders_when_band_within_bounds(self, monkeypatch):
        # Same series, but a range that contains the ~0.3 band -> the section renders.
        flat = pd.Series(
            np.full(400, 0.3, dtype="float64"),
            index=pd.bdate_range("2024-01-01", periods=400),
            name="DGS10",
        )
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: flat)
        q = _make_numeric_q(resolution_criteria=_DGS10_RC, lower_bound=0.0, upper_bound=1.0)

        out = build_anchor_section(q, datetime(2026, 3, 15, tzinfo=UTC))

        assert out  # non-empty; the band ~0.3 lies within [0, 1]
        assert "P10 / P50 / P90 →" in out

    # The spread path is now wired through the same backstop as the single path (part 2 of
    # the two-ticker fix). A relative-return-worded two-ticker question PASSES the wording
    # gate, but if its displayed bounds are a wrong-unit range (a ratio's [60, 110], not a
    # ±pp band) the backstop still drops the section — belt-and-suspenders behind the gate.
    @staticmethod
    def _spread_fetch(spec, _ceiling, **_kwargs):
        if spec.series_id == "CL=F":
            return _daily_positive_series("CL=F", seed=1)
        return _daily_positive_series("^GSPC", seed=2) * 40.0

    @staticmethod
    def _relret_two_ticker_q(*, lower_bound: float, upper_bound: float, qid: int = 8201) -> MagicMock:
        # Passes the relative-return wording gate (routes to spread), 14-day horizon -> a
        # ±few-pp mean-zero band.
        qt = "How much will CL=F's returns exceed ^GSPC's over the window?"
        rc = "return(https://finance.yahoo.com/quote/CL=F) minus return(https://finance.yahoo.com/quote/%5EGSPC)."
        return _make_numeric_q(
            qid=qid,
            question_text=qt,
            resolution_criteria=rc,
            # 14 calendar days past the as_of every caller below passes.
            scheduled_resolution_time=datetime(2026, 3, 29, tzinfo=UTC),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def test_build_anchor_section_spread_skips_on_bounds_mismatch(self, monkeypatch, caplog):
        # gold/silver-ratio-shaped bounds [60, 110] vs a ±few-pp mean-zero spread band ->
        # dropped by the newly-wired backstop with a spread-specific WARN.
        monkeypatch.setattr(ts, "fetch_series", self._spread_fetch)
        q = self._relret_two_ticker_q(lower_bound=60.0, upper_bound=110.0)

        with caplog.at_level(logging.WARNING):
            out = build_anchor_section(q, datetime(2026, 3, 15, tzinfo=UTC))

        assert out == ""
        assert any("zero overlap with question bounds" in r.message for r in caplog.records)

    def test_build_anchor_section_spread_renders_when_band_within_bounds(self, monkeypatch):
        # Same spread, but a pp-scale range [-50, 50] that contains the ±few-pp band -> renders.
        monkeypatch.setattr(ts, "fetch_series", self._spread_fetch)
        q = self._relret_two_ticker_q(lower_bound=-50.0, upper_bound=50.0, qid=8202)

        out = build_anchor_section(q, datetime(2026, 3, 15, tzinfo=UTC))

        assert out  # non-empty; the ±few-pp band lies within [-50, 50]
        assert "Relative-return spread: CL=F vs ^GSPC" in out
        assert "P10 / P50 / P90 →" in out


# A band is the anchor's whole quantitative payload, and the numeric prompt's anchor clause
# fires on the section HEADER alone — so a section without a band tells the forecaster to
# expect a P10/P50/P90 range that isn't there. Prod shipped exactly that once (qid 44803, a
# newly-listed ticker whose history was shorter than the horizon): a bare label, a 52-week
# range whose low WAS the latest value, and "empirical band withheld".
class TestNoBandNoSection:
    # A 28-calendar-day horizon is 19 TRADING steps on a daily series, so the history has to
    # be shorter than that (not merely shorter than 28) to reach the withheld branch.
    _HORIZON_CALENDAR_DAYS = 28
    _HISTORY_OBSERVATIONS = 15
    _AS_OF = datetime(2026, 3, 15, tzinfo=UTC)
    _RESOLVES = datetime(2026, 4, 12, tzinfo=UTC)  # _HORIZON_CALENDAR_DAYS past _AS_OF

    @classmethod
    def _short_history(cls) -> pd.Series:
        return pd.Series(
            np.linspace(150.0, 119.0, cls._HISTORY_OBSERVATIONS),
            index=pd.bdate_range("2026-02-23", periods=cls._HISTORY_OBSERVATIONS),
            name="SPCX",
        )

    def test_fixture_history_is_genuinely_shorter_than_the_horizon(self):
        # The fixture only exercises the withheld branch if y.size <= h; assert that rather
        # than trusting the arithmetic, so a constant change can't quietly make the tests
        # below pass for the wrong reason.
        clock = series_clock(pd.DatetimeIndex(self._short_history().index))
        assert horizon_steps(clock, self._HORIZON_CALENDAR_DAYS) >= self._HISTORY_OBSERVATIONS
        assert (self._RESOLVES.date() - self._AS_OF.date()).days == self._HORIZON_CALENDAR_DAYS

    def test_section_suppressed_when_horizon_exceeds_history(self, monkeypatch, caplog):
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: self._short_history())
        q = _make_numeric_q(
            qid=44803,
            resolution_criteria=_DGS10_RC,
            scheduled_resolution_time=self._RESOLVES,
            lower_bound=50.0,
            upper_bound=250.0,
        )

        with caplog.at_level(logging.INFO):
            out = build_anchor_section(q, self._AS_OF)

        assert out == ""
        assert any("no empirical band" in r.message for r in caplog.records)

    def test_render_still_reports_the_withheld_sentinel(self):
        # The suppression belongs in build_anchor_section, not in _render_single: the renderer
        # keeps returning band=None plus its explanatory line so the caller (and any future
        # coverage instrumentation) can tell "no band" from "unroutable".
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="yfinance", series_id="SPCX"),
            label="SPCX",
        )
        out, band = _render_single(
            self._short_history(),
            route=route,
            ceiling=self._AS_OF.date(),
            calendar_days=self._HORIZON_CALENDAR_DAYS,
        )
        assert band is None
        assert "empirical band withheld" in out

    def test_section_renders_when_history_covers_the_horizon(self, monkeypatch):
        # Control: same question shape, enough history -> band and section both survive, so
        # the suppression keys on the missing band rather than on anything else here.
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: _daily_positive_series("SPCX"))
        q = _make_numeric_q(
            resolution_criteria=_DGS10_RC,
            scheduled_resolution_time=self._RESOLVES,
            lower_bound=0.0,
            upper_bound=1000.0,
        )

        out = build_anchor_section(q, self._AS_OF)

        assert out
        assert "P10 / P50 / P90 →" in out

    def test_non_model_target_route_still_renders_without_a_band(self, monkeypatch):
        # The suppression is gated on model_target, which is what separates the two reasons a
        # band can be absent. A NON-model-target route is deliberately band-free (context-only
        # history plus its explanatory note), so dropping that gate would silently delete a
        # whole intended section shape rather than only the broken one.
        route = _Route(
            kind="single",
            spec=SeriesSpec(source="fred", series_id="PAYEMS", revises=True),
            label="Total nonfarm payrolls",
            model_target=False,
            note="This is the payrolls LEVEL series.",
        )
        monkeypatch.setattr(ts, "route_question", lambda _q: route)
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: _daily_positive_series("PAYEMS"))
        q = _make_numeric_q(scheduled_resolution_time=self._RESOLVES, lower_bound=0.0, upper_bound=1000.0)

        out = build_anchor_section(q, self._AS_OF)

        assert out
        assert "P10 / P50 / P90" not in out
        assert "- Note: This is the payrolls LEVEL series." in out


# Chart side-channel must respect the bounds backstop: build_anchor_section stashes the
# chart only on the success path, so a bounds-rejected (wrong-units) section leaves nothing
# for forecaster.py `_pull_research_chart` to attach to the base forecasters' vision message.
class TestChartBackstop:
    def _flat_series(self) -> pd.Series:
        return pd.Series(
            np.full(400, 0.3, dtype="float64"),
            index=pd.bdate_range("2024-01-01", periods=400),
            name="DGS10",
        )

    def test_chart_not_stashed_on_bounds_reject(self, monkeypatch):
        # Chart flag ON + a band that misses the bounds -> section suppressed AND no chart
        # stashed (the render is never even attempted on the reject path).
        monkeypatch.setenv("TS_ANCHOR_CHART_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: self._flat_series())
        monkeypatch.setattr(
            "metaculus_bot.research.ts_chart.render_anchor_chart",
            MagicMock(side_effect=AssertionError("chart must not render on a bounds-rejected section")),
        )
        q = _make_numeric_q(resolution_criteria=_DGS10_RC, lower_bound=250.0, upper_bound=350.0)

        out = build_anchor_section(q, datetime(2026, 3, 15, tzinfo=UTC))

        assert out == ""
        assert ts._session_charts == {}  # nothing stashed for _pull_research_chart to attach

    def test_chart_stashed_on_success_path(self, monkeypatch):
        # The move to the success path must still stash the chart when the band is in-bounds.
        monkeypatch.setenv("TS_ANCHOR_CHART_ENABLED", "true")
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: self._flat_series())
        monkeypatch.setattr(
            "metaculus_bot.research.ts_chart.render_anchor_chart",
            lambda *_a, **_k: "FAKE_PNG_BASE64",
        )
        q = _make_numeric_q(qid=8123, resolution_criteria=_DGS10_RC, lower_bound=0.0, upper_bound=1.0)

        out = build_anchor_section(q, datetime(2026, 3, 15, tzinfo=UTC))

        assert out  # non-empty success path
        assert ts._session_charts.get(8123) == "FAKE_PNG_BASE64"
