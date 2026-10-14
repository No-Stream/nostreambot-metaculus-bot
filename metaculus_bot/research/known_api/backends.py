"""The known-API backends: FRED, Yahoo Finance, prediction markets and SEC EDGAR.

Each backend takes a resolved identifier (a FRED id, a Yahoo symbol, a venue + market, or an
EDGAR call), reads it through the same client code the research providers use, and returns a
neutral :class:`KnownApiResult`. No LLM, no paid key. A backend NEVER raises to its caller:
an unknown id is ``not_found`` with the provider's message, an empty window is ``empty``, a
transport or quota failure is ``error`` naming the exception class. Detail: docs/research.md
"Known-API registry".

The bounds are the ones the 2026-09-09 cost pass measured a need for: a windowed read capped at
400 observations newest-kept, a 15 s client timeout on each FRED or Yahoo HTTP request, five
market rows, and at most four Kalshi detail GETs per question (a :class:`KalshiGetBudget` the
caller constructs per question, since the loop has no per-question object yet -- see
:func:`market_snapshot`). The FRED and Yahoo operations also have a 15 s async response bound;
that declines a slow result but cannot terminate a Python worker thread, while the client timeout
ensures a stalled socket eventually releases it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import islice
from typing import Any, cast
from xml.etree.ElementTree import ParseError

import aiohttp
import pandas as pd
import trafilatura

from metaculus_bot.constants import (
    DOCUMENT_TEXT_MAX_PAGES,
    DOCUMENT_TEXT_MAX_SECONDS,
    FRED_API_KEY_ENV,
    GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS,
    SEC_EDGAR_CONTACT_EMAIL_ENV,
)
from metaculus_bot.research import document_text, fred_rendering, sec_edgar, ts_fetch
from metaculus_bot.research.known_api.result import KnownApiResult
from metaculus_bot.research.known_api.translate import KnownApiCall
from metaculus_bot.research.market_retrieval import generation, queries, rendering, venues
from metaculus_bot.research.market_retrieval.http import PLATFORM_HTTP_TIMEOUT, read_json_capped
from metaculus_bot.research.market_retrieval.types import MarketMatch, MarketSnapshot
from metaculus_bot.research.number_format import format_decimal_change, format_decimal_value
from metaculus_bot.research.source_evidence import SourceExcerpt

logger = logging.getLogger(__name__)

# The async bound declines a slow result; the FRED and Yahoo clients carry the transport bound.
FRED_YAHOO_CALL_TIMEOUT_S = 15.0
# The default read is the recent window; the driver widens it with start/end when it needs history.
DEFAULT_OBSERVATIONS = 30
# The hard cap: a windowed read never renders more than this, newest kept.
MAX_OBSERVATIONS = 400
# FRED free-text search returns at most this many candidate series.
FRED_SEARCH_LIMIT = 5
# A market snapshot renders at most this many rows.
MARKET_SNAPSHOT_ROWS = 5
# The per-question ceiling on Kalshi detail GETs, enforced by the counting budget below.
MAX_KALSHI_DETAIL_GETS = 4
# ".../trade-api/v2", derived off the catalogue URL so the base cannot drift from the venue's.
KALSHI_API_BASE = venues.KALSHI_EVENTS_URL.rsplit("/", 1)[0]
# A Kalshi market ticker: alphanumerics and dashes with a dash, no whitespace (free text is fuzzy-matched).
_KALSHI_TICKER_SHAPE = re.compile(r"[A-Za-z0-9]+-[A-Za-z0-9-]+\Z")
# A filing document's extracted text is capped at the gap-fill loop's own per-result char budget.
EDGAR_DOC_MAX_CHARS = 8000
# A company page renders this many of its most recent filings (SEC serves them newest first).
EDGAR_FILINGS_ROWS = 15


@dataclass
class KalshiGetBudget:
    """A per-question ceiling on Kalshi detail GETs.

    A semaphore would cap concurrency, not the total; the gap-fill driver calls sequentially, so
    only a counter enforces "at most N GETs per question". The caller builds one per question.
    """

    remaining: int = MAX_KALSHI_DETAIL_GETS

    def take(self) -> bool:
        """Consume one GET's budget; False when the per-question ceiling is already reached."""
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


def _today() -> date:
    return datetime.now(tz=UTC).date()


# ts_fetch's own default lookback; a wider explicit window needs a wider fetch or it silently truncates.
DEFAULT_LOOKBACK_YEARS = 15


def _lookback_years(start: date | None, end: date | None) -> int:
    """Enough lookback for ts_fetch to reach ``start``, else its default; keyed/keyless can't disagree."""
    if start is None:
        return DEFAULT_LOOKBACK_YEARS
    span_days = ((end or _today()) - start).days
    return max(DEFAULT_LOOKBACK_YEARS, span_days // 365 + 2)


def _window(series: pd.Series, *, start: date | None, end: date | None) -> pd.Series:
    """Slice ``series`` to ``[start, end]`` then to the recent/cap bound, newest observations kept.

    With no ``start`` the default is the last :data:`DEFAULT_OBSERVATIONS`; an explicit window is
    honoured and then capped at :data:`MAX_OBSERVATIONS`, so a wide range can never render an
    unbounded table.
    """
    clean = series.dropna().sort_index()
    # An empty FRED series carries a RangeIndex, so a Timestamp comparison would raise; bail first.
    if clean.empty:
        return clean
    if start is not None:
        clean = clean[clean.index >= pd.Timestamp(start)]
    if end is not None:
        clean = clean[clean.index <= pd.Timestamp(end)]
    limit = MAX_OBSERVATIONS if start is not None else min(DEFAULT_OBSERVATIONS, MAX_OBSERVATIONS)
    return clean.tail(limit)


def _summary_lines(series: pd.Series) -> list[str]:
    latest = float(series.iloc[-1])
    latest_date = pd.Timestamp(series.index[-1]).strftime("%Y-%m-%d")
    lines = [f"- Latest: {format_decimal_value(latest)} ({latest_date})"]
    if len(series) >= 2:
        previous = float(series.iloc[-2])
        change = latest - previous
        lines.append(f"- Previous: {format_decimal_value(previous)} (change {format_decimal_change(change)})")
    return lines


def _observation_lines(series: pd.Series, *, total: int, series_label: str, units: str) -> list[str]:
    """Newest-first observation rows, with a caption when the window was capped."""
    newest_first = series.iloc[::-1]
    caption = f"- Observations (newest first, {len(series)} of {total}"
    # Only claim the cap when it actually bound the result, not on a short default-window read.
    caption += (
        f"; capped at {MAX_OBSERVATIONS})" if len(series) == MAX_OBSERVATIONS and total > MAX_OBSERVATIONS else ")"
    )
    rows = [
        f"  - {series_label}; units: {units}; {cast(pd.Timestamp, ts).strftime('%Y-%m-%d')}: "
        f"{format_decimal_value(float(value))}"
        for ts, value in newest_first.items()
    ]
    return [caption + "\n" + "\n".join(rows)]


def _render_series_block(
    *, header: str, source_url: str, series: pd.Series, total: int, series_label: str, units: str
) -> str:
    parts = [
        header,
        f"Source: {source_url}",
        *_summary_lines(series),
        *_observation_lines(series, total=total, series_label=series_label, units=units),
    ]
    return "\n".join(parts)


def _series_evidence_passages(
    series: pd.Series, source_url: str, *, series_label: str, units: str
) -> list[SourceExcerpt]:
    """Expose each delivered API observation with the series context needed to read its value."""
    passages: list[SourceExcerpt] = []
    for timestamp, value in series.iloc[::-1].items():
        row = (
            f"  - {series_label}; units: {units}; {cast(pd.Timestamp, timestamp).strftime('%Y-%m-%d')}: "
            f"{format_decimal_value(float(value))}"
        )
        if len(row) <= GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS:
            passages.append(
                SourceExcerpt(
                    source_url=source_url,
                    text=row,
                    retrieval_method="known_api",
                    content_origin="structured_data",
                )
            )
    return passages


async def _bounded(sync_call: Callable[[], KnownApiResult], *, label: str) -> KnownApiResult:
    """Run a blocking backend in a thread with a bounded response wait."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(sync_call), timeout=FRED_YAHOO_CALL_TIMEOUT_S)
    except TimeoutError:
        logger.warning("known_api %s timed out after %ss", label, FRED_YAHOO_CALL_TIMEOUT_S)
        return KnownApiResult(
            status="error", content_markdown=f"{label} timed out after {FRED_YAHOO_CALL_TIMEOUT_S}s.", source_url=""
        )


def _column(frame: pd.DataFrame, name: str) -> list[Any]:
    """A search column's values, or blanks when FRED omitted it, aligned to the frame's rows."""
    return frame[name].tolist() if name in frame.columns else [""] * len(frame)


def _fred_search_sync(text: str, api_key: str) -> KnownApiResult:
    fred = fred_rendering.Fred(api_key=api_key)
    # Fred.search caps at FRED_SEARCH_LIMIT, so the whole frame renders; columns zip, no row loop.
    hits = fred.search(text, limit=FRED_SEARCH_LIMIT)
    if not isinstance(hits, pd.DataFrame) or hits.empty:
        return KnownApiResult(
            status="empty", content_markdown=f"FRED search for {text!r} matched no series.", source_url=""
        )
    records = list(
        zip(
            list(hits.index),
            _column(hits, "title"),
            _column(hits, "frequency"),
            _column(hits, "units"),
            strict=True,
        )
    )
    rows = [f"FRED series matching {text!r}:"]
    rendered_rows = [f"- {series_id}: {title} ({frequency}; {units})" for series_id, title, frequency, units in records]
    rows += rendered_rows
    passages = [
        SourceExcerpt(
            source_url=f"https://fred.stlouisfed.org/series/{series_id}",
            text=row,
            retrieval_method="known_api",
            content_origin="structured_data",
        )
        for (series_id, _title, _frequency, _units), row in zip(records, rendered_rows, strict=True)
        if len(row) <= GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS
    ]
    return KnownApiResult(
        status="ok",
        content_markdown="\n".join(rows),
        source_url="https://fred.stlouisfed.org/",
        evidence_passages=passages,
    )


def _fred_series_keyed_sync(
    series_id: str, api_key: str, *, start: date | None, end: date | None, first_release: bool
) -> KnownApiResult:
    source_url = f"https://fred.stlouisfed.org/series/{series_id}"
    fred = fred_rendering.Fred(api_key=api_key)
    # A bad id raises ValueError("... does not exist"); fred_series's runner classifies it not_found.
    data = fred.get_series(series_id, observation_start=start, observation_end=end)
    return _render_fred(series_id, data, source_url, fred=fred, start=start, end=end, first_release=first_release)


def _fred_series_keyless_sync(series_id: str, *, start: date | None, end: date | None) -> KnownApiResult:
    source_url = f"https://fred.stlouisfed.org/series/{series_id}"
    # revises=False: a live read wants current values, matching the keyed path (this is not a backtest).
    spec = ts_fetch.SeriesSpec(source="fred", series_id=series_id)
    data = ts_fetch.fetch_series(spec, end or _today(), lookback_years=_lookback_years(start, end))
    return _render_fred(series_id, data, source_url, fred=None, start=start, end=end, first_release=False)


def _render_fred(
    series_id: str,
    data: pd.Series,
    source_url: str,
    *,
    fred: Any,
    start: date | None,
    end: date | None,
    first_release: bool,
) -> KnownApiResult:
    total = int(data.dropna().shape[0])
    windowed = _window(data, start=start, end=end)
    if windowed.empty:
        return KnownApiResult(
            status="empty",
            content_markdown=f"FRED series {series_id!r} has no observations in the window.",
            source_url=source_url,
        )
    title, units = _fred_series_metadata(fred, series_id)
    # The rendered observations are the current vintage; label the first-release table only if it renders.
    note = _fred_first_release_note(fred, series_id, windowed) if (first_release and fred is not None) else ""
    header = f"### {series_id} ({title})" + (" [with first-release comparison]" if note else "")
    series_label = f"{series_id} ({title})"
    block = (
        _render_series_block(
            header=header,
            source_url=source_url,
            series=windowed,
            total=total,
            series_label=series_label,
            units=units,
        )
        + note
    )
    return KnownApiResult(
        status="ok",
        content_markdown=block,
        source_url=source_url,
        links=[source_url],
        evidence_passages=_series_evidence_passages(windowed, source_url, series_label=series_label, units=units),
    )


def _fred_series_metadata(fred: Any, series_id: str) -> tuple[str, str]:
    if fred is None:
        return series_id, "not supplied by the keyless CSV response"
    try:
        info = fred.get_series_info(series_id)
    except (ValueError, OSError, ParseError):  # title is cosmetic enrichment; fall back to the id
        return series_id, "not returned by FRED metadata"
    if isinstance(info, pd.DataFrame):
        title = str(info["title"].iloc[0]) if "title" in info.columns else series_id
        units = str(info["units"].iloc[0]) if "units" in info.columns else "not returned by FRED metadata"
        return title, units
    if isinstance(info, pd.Series):
        title = str(info["title"]) if "title" in info.index else series_id
        units = str(info["units"]) if "units" in info.index else "not returned by FRED metadata"
        return title, units
    return series_id, "not returned by FRED metadata"


def _fred_first_release_note(fred: Any, series_id: str, windowed: pd.Series) -> str:
    observation_start = windowed.index[0]
    first_releases = fred_rendering._fetch_fred_first_releases(fred, series_id, observation_start)
    if first_releases is None or first_releases.empty:
        return ""
    return "\n" + "\n".join(fred_rendering._first_release_lines(windowed, first_releases))


async def fred_series(
    *,
    series_id: str | None = None,
    search: str | None = None,
    start: date | None = None,
    end: date | None = None,
    first_release: bool = False,
) -> KnownApiResult:
    """One FRED series over a date window, or a free-text search over FRED's catalogue.

    ``search`` runs ``Fred.search`` (five hits: id, title, frequency, units). ``series_id`` reads
    the observations through the keyed API when ``FRED_API_KEY`` is set (with the initial-release
    table when ``first_release``), and through the keyless ``fredgraph`` CSV otherwise. One series
    per call; the window is capped at :data:`MAX_OBSERVATIONS`.
    """
    api_key = os.getenv(FRED_API_KEY_ENV)
    source_url = f"https://fred.stlouisfed.org/series/{series_id}" if series_id else ""

    def _run() -> KnownApiResult:
        try:
            if search:
                if not api_key:
                    return KnownApiResult(
                        status="error",
                        content_markdown=f"FRED search needs {FRED_API_KEY_ENV}, which is unset.",
                        source_url="",
                    )
                return _fred_search_sync(search, api_key)
            if not series_id:
                return KnownApiResult(
                    status="error", content_markdown="fred_series needs a series_id or a search string.", source_url=""
                )
            if api_key:
                return _fred_series_keyed_sync(series_id, api_key, start=start, end=end, first_release=first_release)
            return _fred_series_keyless_sync(series_id, start=start, end=end)
        except ts_fetch.FetchError as exc:
            return KnownApiResult(
                status="not_found", content_markdown=f"FRED {series_id!r}: {exc}", source_url=source_url
            )
        except ValueError as exc:
            if fred_rendering.is_unknown_fred_series_error(exc):
                logger.warning(
                    "FRED_UNKNOWN_SERIES: series_id=%s proposed_by=gap_fill_driver",
                    series_id,
                )
                return KnownApiResult(
                    status="not_found",
                    content_markdown=f"FRED has no series {series_id!r}: {exc}",
                    source_url=source_url,
                )
            logger.warning("known_api FRED %s failed (ValueError): %s", series_id or search, exc)
            return KnownApiResult(
                status="error", content_markdown=f"FRED fetch failed (ValueError): {exc}", source_url=source_url
            )
        # ParseError (a non-XML error body) subclasses SyntaxError, not OSError; name it or it escapes.
        except (OSError, ParseError) as exc:
            logger.warning("known_api FRED %s failed (%s): %s", series_id or search, type(exc).__name__, exc)
            return KnownApiResult(
                status="error",
                content_markdown=f"FRED fetch failed ({type(exc).__name__}): {exc}",
                source_url=source_url,
            )

    return await _bounded(_run, label=f"FRED {search or series_id}")


def _yahoo_history_sync(ticker: str, *, start: date | None, end: date | None, column: str) -> KnownApiResult:
    source_url = f"https://finance.yahoo.com/quote/{ticker}/history/"
    spec = ts_fetch.SeriesSpec(source="yfinance", series_id=ticker, column=column)  # type: ignore[arg-type]
    data = ts_fetch.fetch_series(spec, end or _today(), lookback_years=_lookback_years(start, end))
    total = int(data.dropna().shape[0])
    windowed = _window(data, start=start, end=end)
    if windowed.empty:
        return KnownApiResult(
            status="empty",
            content_markdown=f"Yahoo Finance {ticker!r} has no {column} data in the window.",
            source_url=source_url,
        )
    header = f"### {ticker} (Yahoo Finance {column})"
    series_label = f"{ticker} (Yahoo Finance {column})"
    units = "not supplied by the yfinance history response"
    block = _render_series_block(
        header=header,
        source_url=source_url,
        series=windowed,
        total=total,
        series_label=series_label,
        units=units,
    )
    return KnownApiResult(
        status="ok",
        content_markdown=block,
        source_url=source_url,
        links=[source_url],
        evidence_passages=_series_evidence_passages(windowed, source_url, series_label=series_label, units=units),
    )


async def yahoo_history(
    *,
    ticker: str,
    start: date | None = None,
    end: date | None = None,
    column: str = "Close",
) -> KnownApiResult:
    """One Yahoo Finance symbol's adjusted price history over a date window.

    Reads through ``ts_fetch.fetch_series`` (the yfinance client the financial-data provider
    uses), ``column`` in Close/High/Low/Open. One ticker per call, window capped at
    :data:`MAX_OBSERVATIONS`; an empty history (a bad or delisted ticker) is ``not_found``.
    """
    source_url = f"https://finance.yahoo.com/quote/{ticker}/history/"

    def _run() -> KnownApiResult:
        try:
            return _yahoo_history_sync(ticker, start=start, end=end, column=column)
        except ts_fetch.FetchError as exc:
            return KnownApiResult(
                status="not_found", content_markdown=f"Yahoo Finance {ticker!r}: {exc}", source_url=source_url
            )
        except (OSError, ValueError) as exc:
            logger.warning("known_api Yahoo %s failed (%s): %s", ticker, type(exc).__name__, exc)
            return KnownApiResult(
                status="error",
                content_markdown=f"Yahoo fetch failed ({type(exc).__name__}): {exc}",
                source_url=source_url,
            )

    return await _bounded(_run, label=f"Yahoo {ticker}")


async def _kalshi_fetch_json(session: Any, url: str, *, budget: KalshiGetBudget | None = None) -> dict | None:
    """One Kalshi detail GET under the per-question budget; None on an exhausted budget or any non-200."""
    if budget is not None and not budget.take():
        return None
    timeout = aiohttp.ClientTimeout(total=PLATFORM_HTTP_TIMEOUT, sock_read=PLATFORM_HTTP_TIMEOUT)
    try:
        async with session.get(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            body = await read_json_capped(resp, label=f"kalshi {url}")
    except (TimeoutError, aiohttp.ClientError):
        return None
    return body if isinstance(body, dict) else None


async def _kalshi_by_ticker(session: Any, ticker: str, budget: KalshiGetBudget | None) -> MarketMatch | None:
    """One Kalshi event by ticker, from the event endpoint falling to the market endpoint."""
    event_url = f"{KALSHI_API_BASE}/events/{ticker}?with_nested_markets=true"
    data = await _kalshi_fetch_json(session, event_url, budget=budget)
    if data:
        event = dict(data.get("event") or data)
        event["markets"] = event.get("markets") or data.get("markets") or []
        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="known_api")
        if match is not None:
            return match
    market_data = await _kalshi_fetch_json(session, f"{KALSHI_API_BASE}/markets/{ticker}", budget=budget)
    if not market_data:
        return None
    market = market_data.get("market") or market_data
    single = {
        "event_ticker": ticker,
        "title": market.get("title") or market.get("yes_sub_title") or ticker,
        "sub_title": "",
        "settlement_sources": [],
        "markets": [market],
    }
    return venues.kalshi_event_match(single, match_confidence=1.0, channel="known_api")


def _kalshi_fuzzy(catalogue: list[dict[str, Any]], query: str) -> list[MarketMatch]:
    """The catalogue's best matches for a free-text query, zero requests.

    Reuses the prediction-market provider's own scored generator so the two cannot drift (its
    docstring warns that a second copy silently diverges); ``islice`` builds only the rows kept.
    """
    return list(islice(generation._kalshi_universe_channel([query], catalogue), MARKET_SNAPSHOT_ROWS))


def _predictit_ranked(
    scored: list[tuple[float, Any]], builder: Callable[[float, Any], MarketMatch | None]
) -> list[MarketMatch]:
    """The top PredictIt rows a fuzzy score put first, built into rows, capped at the snapshot width."""
    ordered = sorted(scored, key=lambda pair: pair[0], reverse=True)[:MARKET_SNAPSHOT_ROWS]
    return [match for score, item in ordered if (match := builder(score, item)) is not None]


def _predictit_matches(dump: list[dict[str, Any]], query: str) -> list[MarketMatch]:
    """The cached PredictIt dump's best matches for a free-text query, scored over market names."""
    usable = [market for market in dump if isinstance(market, dict) and (market.get("name") or market.get("shortName"))]
    names = [str(market.get("name") or market.get("shortName")) for market in usable]
    scores = queries.fuzzy_best_many([query], names, names)
    return _predictit_ranked(
        list(zip(scores, usable, strict=True)),
        lambda score, market: venues.predictit_market_match(market, match_confidence=score, channel="known_api"),
    )


def _search_matches(rows: list[MarketMatch] | None, venue: str) -> KnownApiResult:
    """Adapt a venue search result: None is an outage (``error``), ``[]`` is a genuine no-match."""
    if rows is None:
        return KnownApiResult(status="error", content_markdown=f"{venue} search failed.", source_url="")
    return _render_matches(rows, venue)


def _render_matches(matches: list[MarketMatch], venue: str) -> KnownApiResult:
    """A market snapshot rendered from the matched rows, or ``not_found`` when nothing matched."""
    rows = matches[:MARKET_SNAPSHOT_ROWS]
    if not rows:
        return KnownApiResult(status="not_found", content_markdown=f"No {venue} market matched.", source_url="")
    snapshot = MarketSnapshot(matches=rows, pool_size=len(rows), forecast_time=datetime.now(tz=UTC))
    links = [row.market_url for row in rows if row.market_url]
    evidence_passages: list[SourceExcerpt] = []
    child_allowances = rendering._child_allowances(rows)
    for match, allowance in zip(rows, child_allowances, strict=True):
        if not match.market_url:
            continue
        parent_row = rendering._table_row(rendering._row_cells(match, snapshot.forecast_time))
        if len(parent_row) <= GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS:
            evidence_passages.append(
                SourceExcerpt(
                    source_url=match.market_url,
                    text=parent_row,
                    retrieval_method="known_api",
                    content_origin="structured_data",
                )
            )
        child_positions = rendering._full_row_order(match.children)[:allowance]
        for position in child_positions:
            child_row = rendering._table_row(
                rendering._child_cells(match.platform, match.children[position], snapshot.forecast_time)
            )
            if len(child_row) <= GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS:
                evidence_passages.append(
                    SourceExcerpt(
                        source_url=match.market_url,
                        text=child_row,
                        retrieval_method="known_api",
                        content_origin="structured_data",
                    )
                )
    return KnownApiResult(
        status="ok",
        content_markdown=rendering.render_snapshot(snapshot),
        source_url=links[0] if links else "",
        links=links,
        evidence_passages=evidence_passages,
    )


async def market_snapshot(
    *,
    venue: str,
    market: str,
    session: Any,
    kalshi_catalogue: list[dict[str, Any]] | None = None,
    predictit_markets: list[dict[str, Any]] | None = None,
    kalshi_detail_budget: KalshiGetBudget | None = None,
) -> KnownApiResult:
    """One prediction-market snapshot for a venue + market, from a venue id or free text.

    Kalshi resolves a ticker through the event endpoint (falling to the market endpoint), bounded to
    ``kalshi_detail_budget`` (a per-question counting budget the caller constructs, since the loop
    has no per-question object yet), and free text over the run's already-pulled catalogue with zero
    new requests. Polymarket and Manifold search; PredictIt reads the run's cached dump. A venue
    outage (search returns None) is ``error``, distinct from a genuine no-match. Up to
    :data:`MARKET_SNAPSHOT_ROWS` rows.
    """
    venue = venue.lower()
    if venue == "polymarket":
        return _search_matches(await venues.polymarket_search(session, market, width=MARKET_SNAPSHOT_ROWS), venue)
    if venue == "manifold":
        return _search_matches(await venues.manifold_search(session, market, width=MARKET_SNAPSHOT_ROWS), venue)
    if venue == "kalshi":
        if _KALSHI_TICKER_SHAPE.match(market):
            budget = kalshi_detail_budget or KalshiGetBudget()
            match = await _kalshi_by_ticker(session, market.upper(), budget)
            return _render_matches([match] if match is not None else [], venue)
        if not kalshi_catalogue:
            return KnownApiResult(
                status="empty", content_markdown="Kalshi catalogue unavailable for a free-text lookup.", source_url=""
            )
        return _render_matches(_kalshi_fuzzy(kalshi_catalogue, market), venue)
    if venue == "predictit":
        if not predictit_markets:
            return KnownApiResult(status="empty", content_markdown="PredictIt dump unavailable.", source_url="")
        return _render_matches(_predictit_matches(predictit_markets, market), venue)
    return KnownApiResult(status="error", content_markdown=f"Unknown venue {venue!r}.", source_url="")


def _extract_filing_text(doc: sec_edgar.FilingDocument) -> str:
    """A filing document's readable text: PDF via document_text, HTML via trafilatura, else decoded."""
    if document_text.is_pdf_body(doc.body):
        pdf = document_text.extract_pdf_text(
            doc.body, max_pages=DOCUMENT_TEXT_MAX_PAGES, max_seconds=DOCUMENT_TEXT_MAX_SECONDS
        )
        text, _pages = document_text.joined_page_text(pdf)
        return text[:EDGAR_DOC_MAX_CHARS]
    decoded = doc.body.decode("utf-8", errors="replace")
    if "html" in (doc.content_type or "").lower():
        return (trafilatura.extract(decoded) or "")[:EDGAR_DOC_MAX_CHARS]
    return decoded[:EDGAR_DOC_MAX_CHARS]


async def _edgar_submissions(session: Any, cik: str) -> KnownApiResult:
    subs = await sec_edgar.company_submissions(session, cik)
    source_url = f"{sec_edgar.DATA_BASE_URL}/submissions/CIK{sec_edgar.pad_cik(subs.cik)}.json"
    lines = [
        f"### {subs.name} (CIK {subs.cik})",
        f"Source: {source_url}",
        "Recent filings (form, filed, period, document):",
    ]
    filings = subs.filings[:EDGAR_FILINGS_ROWS]
    rendered_rows = [
        f"- {filing.form} filed {filing.filing_date} (period {filing.report_date}): {filing.primary_document_url}"
        for filing in filings
    ]
    lines += rendered_rows
    evidence_passages = [
        SourceExcerpt(
            source_url=source_url,
            text=row,
            retrieval_method="known_api",
            content_origin="structured_data",
        )
        for filing, row in zip(filings, rendered_rows, strict=True)
        if filing.primary_document_url and len(row) <= GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS
    ]
    return KnownApiResult(
        status="ok",
        content_markdown="\n".join(lines),
        source_url=source_url,
        links=[source_url],
        evidence_passages=evidence_passages,
    )


async def _edgar_document(session: Any, url: str) -> KnownApiResult:
    doc = await sec_edgar.filing_document(session, url)
    text = await asyncio.to_thread(_extract_filing_text, doc)
    if not text.strip():
        return KnownApiResult(
            status="empty", content_markdown=f"SEC EDGAR filing {url} carried no extractable text.", source_url=doc.url
        )
    evidence_passages = [
        SourceExcerpt(
            source_url=doc.url,
            text=paragraph,
            retrieval_method="known_api",
            content_origin="source_text",
        )
        for paragraph in text.split("\n\n")
        if paragraph.strip() and len(paragraph) <= GAP_FILL_V2_EVIDENCE_PASSAGE_MAX_CHARS
    ]
    return KnownApiResult(
        status="ok",
        content_markdown=text,
        source_url=doc.url,
        links=[doc.url],
        evidence_passages=evidence_passages,
    )


async def edgar(call: KnownApiCall) -> KnownApiResult | None:
    """One SEC EDGAR read for a translated EDGAR URL, or None to decline (fall through to the ladder).

    Declines (returns None) when ``SEC_EDGAR_CONTACT_EMAIL`` is unset, because the fair-access client
    refuses to dial without a contact and the page fetch is the right fallback. With a contact, a
    company page reads the filings table and an Archives document reads its extracted text. A reached
    EDGAR error (an HTTP status, quota) is ``error`` naming it, not a decline.
    """
    if not os.getenv(SEC_EDGAR_CONTACT_EMAIL_ENV, "").strip():
        return None
    try:
        async with sec_edgar.edgar_session() as session:
            if call.edgar_kind == "company_submissions" and call.edgar_arg:
                return await _edgar_submissions(session, call.edgar_arg)
            if call.edgar_kind == "filing_document" and call.edgar_url:
                return await _edgar_document(session, call.edgar_url)
    except sec_edgar.SecEdgarContactUnsetError:
        return None
    # ValueError covers a malformed CIK from pad_cik; without it the never-raise contract breaks.
    except (sec_edgar.SecEdgarError, ValueError) as exc:
        logger.warning("known_api EDGAR %s failed (%s): %s", call.canonical_url, type(exc).__name__, exc)
        return KnownApiResult(status="error", content_markdown=f"SEC EDGAR error: {exc}", source_url=call.canonical_url)
    except (TimeoutError, aiohttp.ClientError) as exc:
        logger.warning("known_api EDGAR %s failed (%s): %s", call.canonical_url, type(exc).__name__, exc)
        return KnownApiResult(
            status="error",
            content_markdown=f"SEC EDGAR fetch failed ({type(exc).__name__}): {exc}",
            source_url=call.canonical_url,
        )
    return None
