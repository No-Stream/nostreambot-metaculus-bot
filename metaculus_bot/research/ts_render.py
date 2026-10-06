"""Section rendering for the time-series-anchor provider.

Turns a routed series (``ts_routing._Route``) plus its fetched history into the markdown
``## Time Series Anchor`` section a forecaster reads: a headline latest value, a
multi-resolution history, a 52-week range, and the horizon-matched empirical band from
``ts_estimators``. Self-budgeted — the per-resolution row caps and
``TS_ANCHOR_SECTION_MAX_CHARS`` in ``constants.py`` cap table and section sizes.

Both renderers return ``(text, band)``: the band is the P10/P50/P90 actually rendered (the
floor-lifted one for a max-window question), so ``timeseries_anchor``'s bounds-overlap
backstop checks exactly what the forecaster sees. ``_render_single`` returns ``None`` for the
band when none was emitted — not a model target, or the horizon exceeds available history.

Also owns the derived-target transforms (``_apply_derivation``): every question resolves on
some derivation of the raw series, and the level case (scale 1.0) is the identity.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from metaculus_bot.constants import (
    FINANCIAL_VARIANCE_RATIO_FLOOR,
    FINANCIAL_VARIANCE_RATIO_LAG,
    TS_ANCHOR_LOOKBACK_YEARS,
    TS_ANCHOR_MONTHLY_TABLE_ROWS,
    TS_ANCHOR_NATIVE_TABLE_ROWS,
    TS_ANCHOR_SECTION_MAX_CHARS,
    TS_ANCHOR_SPREAD_LOOKBACK_YEARS,
    TS_ANCHOR_WEEKLY_TABLE_ROWS,
)
from metaculus_bot.research.noise_flag import noise_flag_line, screen_for_quote_noise
from metaculus_bot.research.ts_estimators import (
    MONTHLY_CLOCK,
    Freq,
    SeriesClock,
    _build_spread_series,
    _detect_freq,
    _empirical_change_band,
    _empirical_max_band,
    _n_eff,
    annualized_realized_vol_pct,
    clock_matches_cadence,
    horizon_steps,
    series_clock,
    stale_latest_age_days,
)
from metaculus_bot.research.ts_routing import Derivation, _Route

logger = logging.getLogger(__name__)

# Trailing OBSERVATIONS behind the annualized-vol note — a smoothing choice, not a calendar
# window, which is why the rendered label names the step unit ("30 trading days" vs "30 days")
# instead of claiming 30 calendar days on both bases. Deliberately not converted to a calendar
# window: the row count sets the estimator's sample size, and rescaling it to 21 rows on
# exchange-traded series would degrade every equity vol reading to make a label true, when
# naming the unit makes it true for free.
REALIZED_VOL_WINDOW = 30

PROVENANCE_FOOTER = (
    "Statistical extrapolation of the resolution series' own history; blind to news, "
    "events, and policy — weigh against the rest of the research."
)

# Human phrasing for the derived-quantity label line and the history-block header.
_DERIVED_TARGET_DESC: dict[Derivation, str] = {
    "mom_diff": "month-over-month change (first difference of the level)",
    "mom_pct": "month-over-month % change",
    "monthly_avg": "monthly average of the higher-frequency series",
}
_DERIVED_HISTORY_HEADER: dict[Derivation, str] = {
    "mom_diff": "Last monthly MoM changes (derived)",
    "mom_pct": "Last monthly MoM % changes (derived)",
    "monthly_avg": "Last monthly averages (derived)",
}


def _today_utc() -> date:
    """Wall-clock UTC date, module-level so tests can freeze it.

    The renderer only gets a ``ceiling`` (the fetch as-of date), which equals today on a
    live run but is a past date under benchmarking — where the same-dated bar is a
    COMPLETED historical one. Comparing the ceiling against the real wall clock is what
    lets the partial-bar marker fire only when the bar can actually still be forming.
    """
    return datetime.now(UTC).date()


def _fmt(v: float) -> str:
    """Published precision on levels, thousands-separated, 4 significant figures below 100.

    Fixed-point above 100 with trailing zeros stripped. The old `:,.1f` rounded a Case-Shiller
    level of 331.893 to "331.9" on a question with 0.02-point buckets, and `:,.0f` above 10,000
    dropped an index level's decimals entirely — the same `.4g`-class defect fixed at the FRED
    render sites, in the other renderer of the same series (both providers append
    unconditionally, so one bundle stated two different values for one observation).

    Capped at three decimals rather than ``number_format.format_decimal_value``'s six because
    this formatter also renders the empirical P10/P50/P90 band, where six decimals on an
    estimate would be fabricated precision; three is enough to carry a level's published digits
    (and the band's quantiles, which a forecaster sizes an interval from, are worth up to 0.05
    index points — roughly 1-2 buckets on the questions this feeds). Below 100 stays `:.4g` so a
    small month-over-month % change or a percentage-point spread band keeps its significant
    figures instead of rounding to "0.000".
    """
    if abs(float(v)) >= 100:
        # rstrip("0") halts at the ".", so an integer's own zeros are never eaten
        # (1200.0 -> "1,200"); the same idiom number_format.format_decimal_value uses.
        return f"{v:,.3f}".rstrip("0").rstrip(".")
    return f"{v:.4g}"


def _history_lines(series: pd.Series, n: int, header: str) -> str:
    tail = series.tail(n)
    dates = pd.DatetimeIndex(tail.index).strftime("%Y-%m-%d")
    values = tail.to_numpy(dtype="float64")
    rows = [f"  - {d}: {_fmt(v)}" for d, v in zip(dates, values, strict=True)]
    return f"- {header}:\n" + "\n".join(rows)


def _downsample_last(series: pd.Series, rule: str) -> pd.Series:
    """Keep the last real observation within each calendar period, KEEPING its true
    observation date. Unlike ``resample(...).last()`` this never labels a row by a
    bucket-end date that postdates the ceiling — load-bearing for a leakage-safe
    provider (a Sunday week-end or a month-end label after the fetch ceiling would
    look like future data even though the value is genuine)."""
    periods = pd.DatetimeIndex(series.index).to_period(rule)
    keep = ~periods.duplicated(keep="last")
    return series[keep]


def _multi_res_history(series: pd.Series, freq: Freq, monthly_header: str = "Last monthly observations") -> list[str]:
    """Native + coarser down-samples per frequency, using the per-resolution row caps.

    ``monthly_header`` overrides the native-monthly block header so derived-quantity
    questions (which collapse to a monthly series) label their history as the derived
    values, not raw observations."""
    blocks: list[str] = []
    if freq == "daily":
        blocks.append(_history_lines(series, TS_ANCHOR_NATIVE_TABLE_ROWS, "Last daily observations"))
        weekly = _downsample_last(series, "W")
        blocks.append(_history_lines(weekly, TS_ANCHOR_WEEKLY_TABLE_ROWS, "Weekly (last obs of week)"))
        monthly = _downsample_last(series, "M")
        blocks.append(_history_lines(monthly, TS_ANCHOR_MONTHLY_TABLE_ROWS, "Monthly (last obs of month)"))
    elif freq == "weekly":
        blocks.append(_history_lines(series, TS_ANCHOR_WEEKLY_TABLE_ROWS, "Last weekly observations"))
        monthly = _downsample_last(series, "M")
        blocks.append(_history_lines(monthly, TS_ANCHOR_MONTHLY_TABLE_ROWS, "Monthly (last obs of month)"))
    elif freq == "monthly":
        blocks.append(_history_lines(series, TS_ANCHOR_MONTHLY_TABLE_ROWS, monthly_header))
    else:  # quarterly / annual: native rows only, honestly labelled — no finer resolution exists
        blocks.append(_history_lines(series, TS_ANCHOR_MONTHLY_TABLE_ROWS, f"Last {freq} observations"))
    return blocks


# Derived-target transforms (ported from the replay's build_target_series).


def _apply_derivation(series: pd.Series, derivation: Derivation, scale: float) -> pd.Series:
    """Turn the raw fetched series into the quantity the question resolves on.

    Mirrors the Phase-A replay's ``build_target_series`` level family:
      - level        → raw × scale (scale=1.0 is a no-op; BOPGTB uses 0.001, millions→billions).
      - mom_diff      → month-over-month first difference × scale (PAYEMS ×1000, thousands→persons).
      - mom_pct       → month-over-month % change (×100), first NaN dropped.
      - monthly_avg   → calendar-month mean of a higher-frequency series (weekly gasoline → month).

    The two ``mom_*`` branches are ROW-based (``diff()`` / ``shift(1)``), so "month-over-month"
    is only true when one row IS one month. Every registry entry that declares them names a
    monthly FRED series (CPIAUCSL, PAYEMS) and ``monthly_avg`` resamples to months first, so the
    invariant holds today — but a future entry declaring ``mom_pct`` on the weekly GASREGW would
    silently render a week-over-week change under a month-over-month label, the same
    row-count-as-calendar defect as the rest of this class. Checked rather than assumed: a
    non-monthly source raises, and the provider soft-fails the section instead of publishing a
    mislabelled quantity.
    """
    if derivation == "level":
        return series * scale if scale != 1.0 else series
    if derivation == "monthly_avg":
        return series.resample("MS").mean().dropna()
    if derivation in ("mom_diff", "mom_pct"):
        source_freq = _detect_freq(pd.DatetimeIndex(series.index))
        if source_freq != "monthly":
            raise ValueError(
                f"derivation {derivation!r} is a row-wise month-over-month change but the source "
                f"series is {source_freq}; one row must be one month for the label to be true"
            )
        if derivation == "mom_diff":
            return series.diff().dropna() * scale
        return ((series / series.shift(1) - 1.0) * 100.0).dropna()
    raise ValueError(f"unhandled derivation {derivation!r}")  # unreachable via Literal


def _realized_max_floor(series: pd.Series, window_start: date | None, ceiling: date) -> float | None:
    """Max already observed within the elapsed part of a max-window question's window.

    A forward max can only rise, so the max over ``[window_start, ceiling]`` is a hard
    lower bound on the answer once the window has started (live case: window opened in the
    past, ``ceiling`` = now). Returns None when the window hasn't opened yet (benchmark
    case: ``ceiling`` = open_time, no elapsed portion) or no observation falls inside it.
    Uses ``open_time`` as ``window_start``; for calendar-scoped questions (e.g. 'highest in
    2025') that understates the true window, giving a looser — but still valid — lower bound.
    """
    if window_start is None or window_start >= ceiling:
        return None
    idx = pd.DatetimeIndex(series.index)
    elapsed = series[(idx >= pd.Timestamp(window_start)) & (idx <= pd.Timestamp(ceiling))]
    if elapsed.empty:
        return None
    return float(elapsed.max())


def _fifty_two_week_line(series: pd.Series, ceiling: date, last: float) -> str:
    """The trailing-year high/low band, or the whole series' band NAMED as such.

    A stale or discontinued series has no observation inside the trailing year, and
    falling back to the full history under a "52-week range" label states a recency the
    numbers do not have — a 2019 high reads as this year's. The fallback still carries
    real information (it is the only band there is), so it renders, with its own dates
    in the label instead.
    """
    cutoff = pd.Timestamp(ceiling) - pd.Timedelta(days=365)
    window = series[series.index >= cutoff]
    label = "52-week range"
    if window.empty:
        window = series
        dates = pd.DatetimeIndex(window.index)
        label = (
            f"full-history range ({dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}; "
            "no observation inside the trailing year)"
        )
    low = float(window.min())
    high = float(window.max())
    span = high - low
    pct = f"{(last - low) / span * 100:.0f}% of the way up the range" if span > 0 else "range is flat"
    return f"- {label}: {_fmt(low)} – {_fmt(high)} (latest sits {pct})"  # noqa: RUF001  # en dash is deliberate range typography in rendered research


def _realized_vol_lines(series: pd.Series, clock: SeriesClock, *, symbol: str) -> list[str]:
    """Annualized realized volatility over the last ``REALIZED_VOL_WINDOW`` observations,
    plus the vendor-noise flag when the series' variance ratio trips the screen.

    Annualizes on the series' OBSERVED density, not a fixed sqrt(252): a 24/7 series prints
    ~365 bars a year, so the trading-day factor understated its volatility by
    sqrt(365/252) = 1.2035x — the q44882 (ETH-USD) defect, whose twin in ``financial_data`` was
    fixed in e6ae276 while this copy kept the constant. The estimator now lives once in
    ``ts_estimators.annualized_realized_vol_pct`` so the two surfaces cannot be corrected
    separately again; only the sentence wording is local. The label names the step unit for
    the same reason: 30 rows is six calendar weeks on an exchange-traded series, so calling it
    "30-day" was a row count masquerading as a calendar window.

    The noise screen is ``research.noise_flag``, shared verbatim with ``financial_data``
    (same estimator, same ``FINANCIAL_VARIANCE_RATIO_*`` thresholds, same telemetry line),
    because this is the other place the bot annualizes ONE-day returns for a forecaster:
    q44797's pegged cross would render an equally inflated figure here, and the anchor can
    route to any URL-cited Yahoo ticker. Only the sentence is local. The BANDS above it
    are structurally immune — they are empirical h-step change quantiles at h of tens of
    observations, where independent quote noise contributes ~2/h of the variance — so the
    flag says so rather than casting doubt on the whole section.
    """
    annualized = annualized_realized_vol_pct(
        series, window=REALIZED_VOL_WINDOW, periods_per_year=clock.periods_per_year
    )
    if annualized is None:
        return []
    vol_line = f"- {REALIZED_VOL_WINDOW}-{clock.step_unit} annualized realized volatility: {annualized:.1f}%"
    screen = screen_for_quote_noise(series, periods_per_year=clock.periods_per_year)
    if screen is None:
        return [vol_line]
    robust_vol = screen.robust_vol_pct
    robust_clause = (
        ""
        if robust_vol is None
        else (
            f" Measured on overlapping {FINANCIAL_VARIANCE_RATIO_LAG}-{clock.step_unit} returns, over which that "
            f"reversing component cancels, the volatility is {robust_vol:.1f}%."
        )
    )
    # No long-horizon window on this surface — the anchor renders one volatility, so the
    # shared line's long_vol reads None here (`surface` is what tells that apart from a
    # yfinance series too short to hold one).
    logger.info(noise_flag_line(screen, surface="ts_anchor", symbol=symbol, short_vol=annualized, long_vol=None))
    return [
        f"{vol_line} — noise-suspect",
        f"- ⚠ Vendor-noise flag: variance ratio VR({FINANCIAL_VARIANCE_RATIO_LAG}) = {screen.ratio:.2f} over this "
        f"series' history. A random walk reads ~1.0; below {FINANCIAL_VARIANCE_RATIO_FLOOR:.2f} most of each "
        "step's move is reversed the next, which inflates any volatility computed from one-step returns."
        f"{robust_clause} The change band above is unaffected — it is built from multi-observation changes, over "
        "which that noise cancels.",
    ]


def _band_line(
    kind: str,
    clock: SeriesClock,
    h: int,
    *,
    lookback_years: int,
    band: tuple[float, float, float],
    last: float,
    n_windows: int,
    n_eff: int,
) -> str:
    unit = clock.step_unit
    p10, p50, p90 = band
    return (
        f"- Horizon-matched empirical band (empirical P10/P50/P90 of all {h}-{unit} {kind} windows "
        f"in the last ~{lookback_years} years — {n_windows:,} overlapping windows, ~{n_eff:,} "
        f"independent — applied to the latest value {_fmt(last)}):\n"
        f"  - P10 / P50 / P90 → {_fmt(p10)} / {_fmt(p50)} / {_fmt(p90)}"
    )


def _latest_value_lines(
    route: _Route,
    series: pd.Series,
    derived: pd.Series,
    *,
    clock: SeriesClock,
    ceiling: date,
) -> list[str]:
    """Header line (latest value + as-of date), the route note, and the stale-latest warning."""
    raw_last = float(series.iloc[-1])
    raw_last_ts = pd.DatetimeIndex(series.index)[-1]
    raw_last_date = raw_last_ts.strftime("%Y-%m-%d")
    # A bar dated the fetch ceiling, rendered on the very day it is forming, is today's
    # still-in-progress bar (live runs set ceiling == today; a backtest's past-dated
    # ceiling matches a completed historical bar, which _today_utc keeps unmarked). The
    # empirical band anchors on this value, so the reader is told it can still move.
    partial_suffix = " — today's bar, in progress" if raw_last_ts.date() == ceiling == _today_utc() else ""
    last = float(derived.iloc[-1])
    freq: Freq = clock.freq
    field_prefix = f"{route.value_field}, " if route.value_field is not None else ""
    if route.derivation != "level":
        parts: list[str] = [
            f"**{route.label}** — latest derived value {_fmt(last)} "
            f"({field_prefix}{_DERIVED_TARGET_DESC[route.derivation]}; from raw level {_fmt(raw_last)} "
            f"as of {raw_last_date}{partial_suffix}; effective series frequency: {freq})"
        ]
    else:
        parts = [
            f"**{route.label}** — latest {_fmt(last)} "
            f"({field_prefix}as of {raw_last_date}{partial_suffix}; series frequency: {freq})"
        ]
    if route.note:
        parts.append(f"- Note: {route.note}")
    if freq == "daily":
        stale_age = stale_latest_age_days(raw_last_ts.date(), ceiling, clock.periods_per_year)
        if stale_age is not None:
            parts.append(
                f"- ⚠ Latest observation is {stale_age} days old — beyond what a {clock.step_unit} "
                "cadence explains; treat the latest value (and the band anchored on it) as stale."
            )
            logger.warning(
                f"FINANCIAL_STALE_LATEST: surface=ts_anchor symbol={route.spec.series_id} "
                f"age_d={stale_age} cadence={clock.step_unit}"
            )
    return parts


def _band_section_lines(
    derived: pd.Series,
    route: _Route,
    *,
    clock: SeriesClock,
    h: int,
    use_log: bool,
    ceiling: date,
    window_start: date | None,
) -> tuple[list[str], tuple[float, float, float] | None]:
    """The empirical-band lines plus the band actually rendered (floor-lifted on a max window).

    The band is ``None`` when none was emitted: not a model target, a series cadence the
    freq buckets misdescribe, or a horizon longer than the available history.
    """
    parts: list[str] = []
    last = float(derived.iloc[-1])
    y = derived.to_numpy(dtype="float64")
    # A forward-window-max question (from title framing OR a High-column yfinance route)
    # resolves on the max over the window, not the period-end level.
    is_max = route.is_max or route.value_field == "High"
    band: tuple[float, float, float] | None = None
    cadence_ok = clock_matches_cadence(clock, pd.DatetimeIndex(derived.index))
    if not cadence_ok and route.model_target:
        # A cadence the freq buckets misdescribe (a ~183-day semiannual series in the
        # "annual" bucket, a 2-4-day series on the daily basis) would convert the horizon
        # by the same wrong factor its gap disagrees by — a mis-sized band under a
        # confident label. Withhold the band; the caller's band-None guard then drops the
        # section rather than serving a wrong quantity.
        logger.warning(
            "ts_anchor: series cadence (median gap) disagrees >1.5x with the %s clock's %.1f-day step "
            "for %s — withholding the empirical band rather than mis-converting the horizon",
            clock.freq,
            clock.nominal_step_days,
            route.label,
        )
    if route.model_target and y.size > h and cadence_ok:
        n_eff = _n_eff(int(y.size), h)
        if is_max:
            band = _empirical_max_band(y, h, use_log=use_log, last=last)
            floor = _realized_max_floor(derived, window_start, ceiling)
            if floor is not None:
                band = (max(band[0], floor), max(band[1], floor), max(band[2], floor))
                parts.append(
                    f"- Realized max so far this window: {_fmt(floor)} — a HARD LOWER BOUND on the answer "
                    f"(the resolution window has already started; a forward max can only rise from here)."
                )
            # sliding_window_view over length-(h+1) windows yields y.size - h forward windows.
            kind = "forward-max"
        else:
            band = _empirical_change_band(y, h, use_log=use_log, anchor=last)
            # y[:-h] vs y[h:] pairs yield y.size - h overlapping h-step changes.
            kind = "change"
        parts.append(
            _band_line(
                kind,
                clock,
                h,
                lookback_years=TS_ANCHOR_LOOKBACK_YEARS,
                band=band,
                last=last,
                n_windows=int(y.size) - h,
                n_eff=n_eff,
            )
        )
    elif route.model_target:
        parts.append(f"- (Horizon {h} exceeds available history; empirical band withheld.)")
    return parts, band


def _render_single(
    series: pd.Series,
    *,
    route: _Route,
    ceiling: date,
    calendar_days: int,
    window_start: date | None = None,
) -> tuple[str, tuple[float, float, float] | None]:
    """Return the rendered section text and the P10/P50/P90 band actually rendered (the
    floor-lifted band for a max-window question), or ``None`` when no band was emitted
    (not a model target, or the horizon exceeds available history). The caller uses the
    band for the bounds-overlap backstop so it checks exactly what the forecaster sees."""
    # Everything downstream operates on the DERIVED quantity the question resolves on
    # (level×scale for plain/unit-converted levels; MoM change / MoM % / monthly average
    # for the derived shapes). For plain level (scale=1.0) `derived` IS `series`, so the
    # level path is byte-identical to before.
    derived = _apply_derivation(series, route.derivation, route.scale)
    is_derived = route.derivation != "level"
    # MoM change / MoM % / monthly-average all collapse to a monthly effective clock (enforced
    # in `_apply_derivation`, which refuses a non-monthly source for the row-wise mom_* shapes);
    # the horizon and band are computed on that, matching the replay's build_target_series.
    # Everything else reads BOTH clock facts off the series: `freq` alone can't tell a
    # business-day series from a 24/7 one, and the horizon conversion needs that (see
    # `SeriesClock`).
    clock = MONTHLY_CLOCK if is_derived else series_clock(pd.DatetimeIndex(derived.index))
    h = horizon_steps(clock, calendar_days)
    use_log = bool(np.all(derived.to_numpy(dtype="float64") > 0.0))

    parts = _latest_value_lines(route, series, derived, clock=clock, ceiling=ceiling)
    if is_derived:
        parts.extend(_multi_res_history(derived, clock.freq, monthly_header=_DERIVED_HISTORY_HEADER[route.derivation]))
    else:
        parts.extend(_multi_res_history(derived, clock.freq))
        parts.append(_fifty_two_week_line(derived, ceiling, float(derived.iloc[-1])))

    band_lines, band = _band_section_lines(
        derived,
        route,
        clock=clock,
        h=h,
        use_log=use_log,
        ceiling=ceiling,
        window_start=window_start,
    )
    parts.extend(band_lines)

    if clock.freq == "daily" and use_log:
        parts.extend(_realized_vol_lines(derived, clock, symbol=route.spec.series_id))

    parts.append(f"\n_{PROVENANCE_FOOTER}_")
    return "\n".join(parts), band


def _render_spread(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    route: _Route,
    calendar_days: int,
) -> tuple[str, tuple[float, float, float]]:
    """Return the rendered spread section text and its P10/P50/P90 relative-return band
    (pp). Mirrors ``_render_single``'s (text, band) shape so ``build_anchor_section`` can
    run the same ``_band_misses_bounds`` bounds-overlap backstop on the spread path. The
    spread always emits a band (a too-short history raises ValueError first), so the band
    is never None here."""
    spread_series = _build_spread_series(series_a, series_b)  # raises ValueError on bad legs
    clock = series_clock(pd.DatetimeIndex(spread_series.index))
    h = horizon_steps(clock, calendar_days)
    y = spread_series.to_numpy(dtype="float64")
    if y.size <= h:
        raise ValueError(f"spread history length {y.size} too short for horizon {h}")
    # Re-anchor to 0 at the forecast ceiling: the band is the forward-window
    # relative return (pp), which is what the question resolves on.
    band = _empirical_change_band(y, h, use_log=False, anchor=0.0)

    last_a = float(series_a.iloc[-1])
    last_b = float(series_b.iloc[-1])
    date_a = pd.DatetimeIndex(series_a.index)[-1].strftime("%Y-%m-%d")
    date_b = pd.DatetimeIndex(series_b.index)[-1].strftime("%Y-%m-%d")
    field_prefix_a = f"{route.value_field}, " if route.value_field is not None else ""
    field_prefix_b = f"{route.value_field_b}, " if route.value_field_b is not None else ""
    parts: list[str] = [
        f"**Relative-return spread: {route.label} vs {route.label_b}** "
        f"(ret[{route.label}] − ret[{route.label_b}] over the forecast window, in percentage points)",  # noqa: RUF001  # minus sign is deliberate math typography in rendered research
        f"- {route.label} latest: {_fmt(last_a)} ({field_prefix_a}as of {date_a})",
        f"- {route.label_b} latest: {_fmt(last_b)} ({field_prefix_b}as of {date_b})",
    ]
    parts.append(_history_lines(series_a, TS_ANCHOR_NATIVE_TABLE_ROWS, f"{route.label} recent"))
    parts.append(_history_lines(series_b, TS_ANCHOR_NATIVE_TABLE_ROWS, f"{route.label_b} recent"))
    unit = clock.step_unit
    # y[:-h] vs y[h:] pairs yield y.size - h overlapping h-step changes.
    parts.append(
        f"- Forward {h}-{unit} relative-return band (pp, empirical over the last "
        f"~{TS_ANCHOR_SPREAD_LOOKBACK_YEARS} years — {int(y.size) - h:,} overlapping windows, "
        f"~{_n_eff(int(y.size), h):,} independent):\n"
        f"  - P10 / P50 / P90 → {_fmt(band[0])} / {_fmt(band[1])} / {_fmt(band[2])}"
    )
    parts.append(
        "- Relative-return spreads are ~mean-zero by construction; the band is an honest prior, "
        "not a directional signal."
    )
    parts.append(f"\n_{PROVENANCE_FOOTER}_")
    return "\n".join(parts), band


def _truncate_section(text: str) -> str:
    """Hard char-budget backstop (row caps normally keep it well under)."""
    if len(text) <= TS_ANCHOR_SECTION_MAX_CHARS:
        return text
    marker = "\n[truncated — time-series anchor section budget]"
    return text[: TS_ANCHOR_SECTION_MAX_CHARS - len(marker)].rstrip() + marker
