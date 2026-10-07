"""Time-series anchor research provider (Phase B).

For a numeric Metaculus question whose resolution series is a fetchable
FRED/yfinance series, render a deterministic empirical anchor: the latest value,
a multi-resolution history, a 52-week range, and a horizon-matched empirical band
built ONLY from the series' own past. No LLM, no statsforecast, no model selection
— the Phase-A offline replay (``scratch/ts_anchor_replay_2026-07-16/synthesis.md``)
found CV-gated model picks beat naive out-of-sample only 43% of the time, while the
naive empirical h-step-change band is sharper AND better tail-calibrated than what
we publish (published cov@10 was 0.03 vs a 0.10 target; the anchor's was 0.18). The
band's value is grounding the forecaster and pulling in our over-wide low tail.

Backtest-safe — the FIRST research provider that is. Unlike prediction_market /
resolution_source (which hard-disable under ``is_benchmarking``), this provider
runs in benchmarks by pinning ``as_of`` to ``question.open_time`` and fetching the
series point-in-time up to that date (ALFRED vintages for revising macro series),
so series data known at forecast time IS the answer without leaking the resolution.

This module owns the provider factory, the per-session caches, and the two guards that
decide whether a routed question actually gets a section — both of them "no payload, no
section" rules. The magnitude backstop (``_band_misses_bounds``) drops a band whose values
are the wrong quantity for the question's displayed range, and a question whose history is
too short for its horizon renders no band at all, which also drops the section — the numeric
prompt's anchor clause fires on the section header alone, so a headline with no band promises
a P10/P50/P90 range that isn't there.

The rest of the stack lives in siblings, layered leaf-first so the imports run one way:

- ``ts_fetch`` — point-in-time, leakage-proof fetching (FRED / ALFRED vintages / yfinance).
- ``ts_estimators`` — the pure-numpy band math and horizon conversions.
- ``ts_routing`` — deterministic question → series routing (URL extraction, then a curated
  keyword registry) and the ``_Route`` it returns.
- ``ts_render`` — section rendering, the derived-target transforms, and the char budget.
- ``ts_chart`` — the optional matplotlib chart, imported lazily off the cold path.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
from forecasting_tools.data_models.questions import MetaculusQuestion, NumericQuestion

from metaculus_bot.constants import (
    TS_ANCHOR_CHART_ENABLED_ENV,
    TS_ANCHOR_ENABLED_ENV,
    TS_ANCHOR_LOOKBACK_YEARS,
    TS_ANCHOR_OPEN_BOUND_SPAN_TOLERANCE,
    TS_ANCHOR_SPREAD_LOOKBACK_YEARS,
    TS_ANCHOR_TIMEOUT,
    env_flag_enabled,
)
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.ts_estimators import (
    _empirical_change_band,
    _horizon_end_date,
    horizon_steps,
    series_clock,
)
from metaculus_bot.research.ts_fetch import (
    FetchError,
    _reset_politeness_clock,
    _reset_series_cache,
    fetch_series,
)
from metaculus_bot.research.ts_render import (
    _apply_derivation,
    _fmt,
    _render_single,
    _render_spread,
    _truncate_section,
)
from metaculus_bot.research.ts_routing import _Route, route_question

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-session cache
# ---------------------------------------------------------------------------

# Rendered-section cache keyed (qid, as_of_iso) — the as_of leg keeps a backtest
# at one as-of from reusing a section computed at another.
_SECTION_CACHE: dict[tuple[int, str], str] = {}

# Chart side-channel: qid -> base64 PNG. Populated by build_anchor_section only
# when TS_ANCHOR_CHART_ENABLED is on AND the question routed to a single LEVEL
# series (v1 skips max-window / spread charts). The provider returns only the
# text section; the forecaster pulls the image from here to attach to each base
# model's vision message. Never read by the stacker / summarizer / gap-fill.
_session_charts: dict[int, str] = {}


def _reset_session_caches() -> None:
    """Clear the section + chart caches, the series cache, and the fetch pacing clock.

    The pacing clock rides along (tests + session start) so a fresh session's first fetch
    isn't made to wait out the last one's interval.
    """
    _SECTION_CACHE.clear()
    _session_charts.clear()
    _reset_series_cache()
    _reset_politeness_clock()


def _as_of_iso(as_of: datetime) -> str:
    aware = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat()


def _maybe_stash_single_chart(
    series: pd.Series, *, route: _Route, question: MetaculusQuestion, as_of: datetime, calendar_days: int
) -> None:
    """Render + stash the anchor chart for a single plain-LEVEL question when the chart
    flag is on. No-op for max-window / spread / derived-target / no-band cases (v1 charts
    only the plain level shape, where the ribbon reads cleanly — a derived MoM/monthly-avg
    band is on a different quantity than the level history this charts). Chart-render
    failures are swallowed so a plotting hiccup never breaks the text section — the chart
    is a strict add-on and the caller's soft-fail only guards the text. An unimportable
    chart module (matplotlib absent under --no-dev) degrades the same way but logs at
    ERROR: that's a misconfigured flag flip affecting the whole run, not one question.
    """
    if not env_flag_enabled(TS_ANCHOR_CHART_ENABLED_ENV):
        return
    qid = getattr(question, "id_of_question", None)
    if qid is None:
        return
    is_max = route.is_max or route.value_field == "High"
    # v1: chart only plain level questions. Derived targets (MoM diff / MoM % / monthly
    # avg) render + fit the band on a derived quantity; the level-shape chart would
    # mislead by pairing a level history with a change-quantity ribbon.
    if is_max or route.derivation != "level" or not route.model_target:
        return

    # Chart the derived (unit-scaled) series so the ribbon matches the text band exactly;
    # for plain level (scale=1.0) this is a no-op and identical to before.
    charted = _apply_derivation(series, route.derivation, route.scale)
    # Only plain-LEVEL routes reach here (derived targets returned above), so the charted
    # series is the fetched one and its own clock — resolution AND observed density — is what
    # both the horizon and the ribbon's calendar end must be computed on.
    clock = series_clock(pd.DatetimeIndex(charted.index))
    h = horizon_steps(clock, calendar_days)
    y = charted.to_numpy(dtype="float64")
    if y.size <= h:
        return  # band withheld in the text too; nothing to chart
    last = float(charted.iloc[-1])
    use_log = bool(np.all(y > 0.0))
    band = _empirical_change_band(y, h, use_log=use_log, anchor=last)

    as_of_ts = pd.Timestamp(as_of).tz_localize(None) if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of)
    try:
        # Import inside the guard: matplotlib is a dev-only dependency, and the bot
        # workflows install with --no-dev, so flipping the chart flag on in prod must
        # degrade to the text-only anchor rather than kill the provider.
        from metaculus_bot.research.ts_chart import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # matplotlib kept off the cold path
            render_anchor_chart,
        )
    except ImportError as exc:  # HARNESS-SCAN-EXEMPT-defensive-fallback
        # A missing chart module is a run-level misconfiguration (the flag is on but
        # matplotlib isn't installed), not a per-question render hiccup — log it at
        # ERROR so it stands out from the WARN below, then degrade to text-only.
        logger.error(
            "ts_anchor: chart module unavailable — matplotlib is dev-only and absent under "
            "`uv sync --no-dev`, so no chart will attach for ANY question this run: %s",
            exc,
        )
        return

    try:
        _session_charts[qid] = render_anchor_chart(
            charted,
            as_of=as_of_ts,
            horizon_end=_horizon_end_date(as_of_ts, clock, h),
            band=band,
            title=route.label,
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning("ts_anchor: chart render failed for qid=%s (%s): %s", qid, type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# Magnitude backstop
# ---------------------------------------------------------------------------


def _finite_bound(v: float) -> bool:
    """A displayed edge imposes a real constraint only when finite; some questions carry
    ``±inf`` bounds, which mean "no constraint on this side"."""
    return bool(np.isfinite(v))


def _band_misses_bounds(question: NumericQuestion, band: tuple[float, float, float] | None) -> bool:
    """Generic units/magnitude backstop: True when the rendered P10-P90 band lies ENTIRELY
    outside the question's displayed range — a gross mismatch (level-vs-derived, or a
    wrong-country CPI magnitude) meaning the anchor describes a different quantity than the
    one that resolves. Returns False when no band was rendered — nothing to check.

    A CLOSED edge is absolute: the outcome cannot settle past it, so any band clear of it is
    disqualifying. An OPEN edge only LOOSENS the constraint — the outcome can settle somewhat
    beyond a displayed edge, so a band sitting just outside is an ordinary forecast rather than
    a units error, but one sitting far outside is still the wrong quantity. Treating open as
    "no constraint at all" disarmed this guard on the ~95% of numeric questions that carry two
    open bounds, which is how a percent-unit band on a basis-point question could have shipped
    unchecked. A non-finite edge stays a genuine no-constraint case: with no finite span there
    is nothing to scale a tolerance against.
    """
    if band is None:
        return False
    lower_finite = _finite_bound(question.lower_bound)
    upper_finite = _finite_bound(question.upper_bound)
    # Tolerance is a multiple of the displayed span, so it is unit-agnostic; it needs both
    # edges finite to be defined at all.
    margin = (
        TS_ANCHOR_OPEN_BOUND_SPAN_TOLERANCE * (question.upper_bound - question.lower_bound)
        if lower_finite and upper_finite
        else np.inf
    )
    eff_lower = -np.inf if not lower_finite else question.lower_bound - (margin if question.open_lower_bound else 0.0)
    eff_upper = np.inf if not upper_finite else question.upper_bound + (margin if question.open_upper_bound else 0.0)
    p10, _p50, p90 = band
    return bool(p90 < eff_lower or p10 > eff_upper)


# ---------------------------------------------------------------------------
# Synchronous core
# ---------------------------------------------------------------------------


def build_anchor_section(question: NumericQuestion, as_of: datetime) -> str:
    """Synchronous core: route, fetch point-in-time, render. Returns "" if unroutable.

    Runs in a worker thread (all HTTP is blocking). Raises FetchError/ValueError on
    a genuine data problem — the async wrapper soft-fails those to "".
    """
    scheduled = getattr(question, "scheduled_resolution_time", None)
    if not isinstance(scheduled, datetime):
        return ""
    ceiling = as_of.date()
    calendar_days = max(1, (scheduled.date() - ceiling).days)

    route = route_question(question)
    if route is None:
        return ""

    if route.kind == "spread":
        return _build_spread_anchor_section(question, route, ceiling, calendar_days)
    return _build_single_anchor_section(question, route, as_of, ceiling=ceiling, calendar_days=calendar_days)


def _build_spread_anchor_section(question: NumericQuestion, route: _Route, ceiling: date, calendar_days: int) -> str:
    """Fetch both legs, render the relative-return spread, and apply the bounds backstop."""
    assert route.spec_b is not None  # kind=="spread" always sets spec_b
    series_a = fetch_series(route.spec, ceiling, lookback_years=TS_ANCHOR_SPREAD_LOOKBACK_YEARS)
    series_b = fetch_series(route.spec_b, ceiling, lookback_years=TS_ANCHOR_SPREAD_LOOKBACK_YEARS)
    spread_text, spread_band = _render_spread(series_a, series_b, route=route, calendar_days=calendar_days)
    if _band_misses_bounds(question, spread_band):
        logger.warning(
            "ts_anchor: relative-return spread band P10/P90 %s/%s has zero overlap with question "
            "bounds [%s, %s] (open_lower=%s open_upper=%s) — legs=%s/%s qid=%s url=%s; skipping "
            "section (likely a wrong-quantity two-ticker question — ratio / price-difference / level)",
            _fmt(spread_band[0]),
            _fmt(spread_band[2]),
            question.lower_bound,
            question.upper_bound,
            question.open_lower_bound,
            question.open_upper_bound,
            route.spec.series_id,
            route.spec_b.series_id,
            question.id_of_question,
            question.page_url,
        )
        return ""
    return _truncate_section(spread_text)


def _build_single_anchor_section(
    question: NumericQuestion, route: _Route, as_of: datetime, *, ceiling: date, calendar_days: int
) -> str:
    """Fetch one series, render it, and apply both no-payload-no-section guards."""
    series = fetch_series(route.spec, ceiling, lookback_years=TS_ANCHOR_LOOKBACK_YEARS)
    # For a forward-max question whose window has already opened, the max over the
    # elapsed portion is a hard lower bound. Approximate window_start by open_time; when
    # ceiling == open_time (benchmark path) there's no elapsed portion and no floor.
    open_time = getattr(question, "open_time", None)
    window_start = open_time.date() if isinstance(open_time, datetime) else None
    text, band = _render_single(
        series, route=route, ceiling=ceiling, calendar_days=calendar_days, window_start=window_start
    )
    if band is None and route.model_target:
        # No band, no section. The band IS the anchor's quantitative payload, and the numeric
        # prompt's anchor clause fires on the section header alone (prompts.py
        # `_ts_anchor_evidence_clause`), so a bandless section tells the forecaster to expect a
        # P10/P50/P90 range that isn't there and leaves only a bare label plus a history table.
        # Live case: a newly-listed ticker whose history was shorter than the horizon.
        # Gated on model_target because that flag is what distinguishes the two ways
        # _render_single returns no band: a NON-model-target route is deliberately band-free
        # (context-only history), whereas a model-target route promised a band and came up short.
        logger.info(
            "ts_anchor: no empirical band for qid=%s (series=%s derivation=%s; history shorter than "
            "the horizon) — skipping section rather than emitting a header the prompt's anchor "
            "clause promises a band for",
            question.id_of_question,
            route.spec.series_id,
            route.derivation,
        )
        return ""
    if _band_misses_bounds(question, band):
        assert band is not None  # _band_misses_bounds returns False for a None band
        logger.warning(
            "ts_anchor: empirical band P10/P90 %s/%s has zero overlap with question bounds "
            "[%s, %s] (open_lower=%s open_upper=%s) — series=%s derivation=%s qid=%s url=%s; "
            "skipping section (likely wrong units / country)",
            _fmt(band[0]),
            _fmt(band[2]),
            question.lower_bound,
            question.upper_bound,
            question.open_lower_bound,
            question.open_upper_bound,
            route.spec.series_id,
            route.derivation,
            question.id_of_question,
            question.page_url,
        )
        return ""
    # Stash the chart only on the success path: a bounds-rejected section returns "" but a
    # chart stashed here would still be pulled onto every base forecaster's vision message
    # (forecaster.py `_pull_research_chart`), defeating the backstop. Success-only also
    # skips a wasted matplotlib render on rejects.
    _maybe_stash_single_chart(series, route=route, question=question, as_of=as_of, calendar_days=calendar_days)
    return _truncate_section(text)


# ---------------------------------------------------------------------------
# ResearchCallable factory
# ---------------------------------------------------------------------------


def timeseries_anchor_provider(is_benchmarking: bool = False) -> ResearchCallable:
    """Factory for the time-series-anchor research callable.

    Gated on ``TS_ANCHOR_ENABLED`` (double-gated: orchestrator selects on the flag,
    and ``_fetch`` re-checks it). Unlike other backtest-sensitive providers this one
    does NOT hard-disable under ``is_benchmarking`` — it is leakage-safe by pinning
    ``as_of`` to ``question.open_time`` (live: ``datetime.now(UTC)``) and fetching the
    series point-in-time up to that date, with ALFRED vintages for revising macro
    series. Soft-fails to "" on any per-question fetch/data error or timeout.
    """

    async def _fetch(question: MetaculusQuestion) -> str:
        if not env_flag_enabled(TS_ANCHOR_ENABLED_ENV):
            return ""
        if not isinstance(question, NumericQuestion):
            return ""

        if is_benchmarking:
            open_time = getattr(question, "open_time", None)
            if not isinstance(open_time, datetime):
                logger.warning(
                    "ts_anchor: is_benchmarking but qid=%s has no open_time; skipping (leakage-safe)",
                    getattr(question, "id_of_question", None),
                )
                return ""
            as_of = open_time
        else:
            as_of = datetime.now(UTC)

        qid = getattr(question, "id_of_question", None)
        cache_key = (qid, _as_of_iso(as_of)) if qid is not None else None
        if cache_key is not None and cache_key in _SECTION_CACHE:
            return _SECTION_CACHE[cache_key]

        try:
            section = await asyncio.wait_for(
                asyncio.to_thread(build_anchor_section, question, as_of),
                timeout=TS_ANCHOR_TIMEOUT,
            )
        except (TimeoutError, FetchError, ValueError) as exc:
            logger.warning("ts_anchor: soft-fail for qid=%s (%s): %s", qid, type(exc).__name__, exc)
            return ""

        if cache_key is not None:
            _SECTION_CACHE[cache_key] = section
        return section

    return _fetch
