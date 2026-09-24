"""Financial data research provider using yfinance and FRED.

Fetches real price/indicator data for questions involving trackable financial metrics.
Follows the same factory-function-returning-ResearchCallable pattern as other providers.
"""

import asyncio
import logging
import os
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import pandas as pd
import yfinance
from forecasting_tools import GeneralLlm
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    FINANCIAL_CLASSIFIER_MODEL,
    FINANCIAL_CLASSIFIER_TIMEOUT,
    FINANCIAL_VARIANCE_RATIO_FLOOR,
    FINANCIAL_VARIANCE_RATIO_LAG,
    FINANCIAL_YFINANCE_LOOKBACK_DAYS,
    FINANCIAL_YFINANCE_RECENT_DAYS,
    FRED_API_KEY_ENV,
    MAX_FINANCIAL_IDENTIFIERS,
    TS_ANCHOR_HTTP_TIMEOUT,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.llm_retry import invoke_with_transient_retry
from metaculus_bot.research.currency_pegs import peg_disclosure_lines, peg_for_ticker
from metaculus_bot.research.fred_rendering import UnknownFredSeries, _fetch_fred_data, _fetch_fred_data_ceiling
from metaculus_bot.research.fx_identifiers import is_fx_identifier
from metaculus_bot.research.known_api import parse as known_api_parse
from metaculus_bot.research.noise_flag import noise_flag_line, screen_for_quote_noise
from metaculus_bot.research.provider_diagnostics import is_lost_source, record_provider_detail
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.ts_estimators import (
    CALENDAR_DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    annualized_realized_vol_pct,
    daily_step_unit,
    observed_periods_per_year,
    stale_latest_age_days,
)

logger: logging.Logger = logging.getLogger(__name__)

# Full-string char-class guards (same classes the extraction regexes enforce), used
# to sanitize classifier-emitted IDs — which come from comma-splitting with NO
# char validation — before they reach the fetch set and the HTML-comment marker.
# A `-->` in a classifier token would otherwise close the comment and leak its tail
# as visible markdown in the published Metaculus comment.
_TICKER_CHARS_RE = re.compile(r"^[A-Za-z0-9.^=\-]+$")
_FRED_CHARS_RE = re.compile(r"^[A-Za-z0-9_]+$")

# Single source of truth for the reference identifiers: id -> human label, grouped
# by category. The KNOWN_* frozensets AND the CLASSIFIER_PROMPT reference table are
# both DERIVED from these dicts, so the allowlist and the prompt cannot drift apart.
# Frozensets feed the soft-fail flagging (unrecognized classifier IDs surface as
# WARNINGs + in the routing marker, never silently dropped); labels feed the prompt.
_TICKER_GROUPS: dict[str, dict[str, str]] = {
    "Stock indices": {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "Nasdaq",
        "^RUT": "Russell 2000",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei",
    },
    "Stocks": {
        "AAPL": "",
        "MSFT": "",
        "GOOGL": "",
        "AMZN": "",
        "NVDA": "",
        "TSLA": "",
        "META": "",
        "BRK-B": "",
        "JPM": "",
        "V": "",
    },
    "Commodities": {
        "CL=F": "crude oil",
        "GC=F": "gold",
        "SI=F": "silver",
        "NG=F": "natural gas",
        "HG=F": "copper",
    },
    "Crypto": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
    },
    "Bonds/Rates": {
        "^TNX": "10Y Treasury yield",
        "^FVX": "5Y Treasury yield",
        "^TYX": "30Y Treasury yield",
    },
    "Currencies": {
        "EURUSD=X": "",
        "GBPUSD=X": "",
        "USDJPY=X": "",
        "DX-Y.NYB": "US Dollar Index",
    },
}
_FRED_GROUPS: dict[str, dict[str, str]] = {
    "labor": {"UNRATE": "unemployment rate", "PAYEMS": "nonfarm payrolls"},
    "inflation": {"CPIAUCSL": "CPI all items", "CPILFESL": "core CPI", "PCEPI": "PCE price index"},
    "output": {"GDP": "gross domestic product", "GDPC1": "real GDP"},
    "rates": {"FEDFUNDS": "federal funds rate", "DFF": "daily fed funds"},
    "treasury": {"DGS10": "10Y Treasury rate", "DGS2": "2Y Treasury rate", "T10Y2Y": "10Y-2Y spread"},
    "housing": {"CSUSHPISA": "Case-Shiller home price index", "HOUST": "housing starts"},
    "consumer": {"UMCSENT": "consumer sentiment", "RSAFS": "retail sales"},
    "money": {"M2SL": "M2 money supply", "WALCL": "Fed balance sheet"},
}

TICKER_LABELS: dict[str, str] = {tid: label for group in _TICKER_GROUPS.values() for tid, label in group.items()}
FRED_LABELS: dict[str, str] = {sid: label for group in _FRED_GROUPS.values() for sid, label in group.items()}
KNOWN_TICKERS: frozenset[str] = frozenset(TICKER_LABELS)
KNOWN_FRED_SERIES: frozenset[str] = frozenset(FRED_LABELS)


def _render_ticker(tid: str, label: str) -> str:
    """Render `id (label)` when a label exists, else the bare id (stocks have none)."""
    return f"{tid} ({label})" if label else tid


def _build_ticker_reference_lines() -> str:
    """Build the prompt's grouped ticker reference lines from _TICKER_GROUPS."""
    return "\n".join(
        f"{group_name}: " + ", ".join(_render_ticker(tid, label) for tid, label in members.items())
        for group_name, members in _TICKER_GROUPS.items()
    )


def _build_fred_reference_lines() -> str:
    """Build the prompt's FRED reference bullet lines from _FRED_GROUPS."""
    return "\n".join(
        "- " + ", ".join(f"{sid} ({label})" for sid, label in members.items()) for members in _FRED_GROUPS.values()
    )


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    """Drop duplicates while keeping first-seen order (dict preserves insertion)."""
    return list(dict.fromkeys(items))


def _cap_identifiers(
    tickers: list[str],
    fred_series: list[str],
    extracted: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Bound the total fetch set to ``MAX_FINANCIAL_IDENTIFIERS``, dropping classifier IDs first.

    Each identifier becomes its own ``asyncio.to_thread``, and the fan-out was
    previously unbounded — one over-eager classification could queue arbitrarily many
    blocking calls into the process-wide default executor that ts_fetch,
    resolution_source, the agentic fetch ladder, and the /auth/key probe all share.
    Queued tasks burn their ``wait_for`` budget without executing, so the damage lands on
    unrelated providers and other questions.

    Deliberately asymmetric about WHAT it drops: URL-EXTRACTED identifiers are kept
    unconditionally, because they are the load-bearing guarantee that the source a
    question actually resolves on is fetched even when the classifier misroutes (the
    invariant the merged-set bail below rests on). Only classifier-added extras are
    trimmed, from the tail, and the drop is logged so an over-eager classification is
    visible rather than silent.
    """
    total = len(tickers) + len(fred_series)
    if total <= MAX_FINANCIAL_IDENTIFIERS:
        return (tickers, fred_series)

    extracted_tickers = set(
        extracted["tickers"]
    )  # HARNESS-SCAN-EXEMPT-object-explosion: short id list, not a frame column
    extracted_fred = set(
        extracted["fred_series"]
    )  # HARNESS-SCAN-EXEMPT-object-explosion: short id list, not a frame column
    budget = MAX_FINANCIAL_IDENTIFIERS - len(extracted_tickers) - len(extracted_fred)
    dropped: list[str] = []

    def keep(identifiers: list[str], protected: set[str]) -> list[str]:
        nonlocal budget
        kept: list[str] = []
        for identifier in identifiers:
            if identifier in protected:
                kept.append(identifier)  # extracted: never dropped
            elif budget > 0:
                kept.append(identifier)
                budget -= 1
            else:
                dropped.append(identifier)
        return kept

    kept_tickers = keep(tickers, extracted_tickers)
    kept_fred = keep(fred_series, extracted_fred)
    logger.warning(
        "financial_data: capped fetch set at %d identifiers (%d requested); dropped classifier-only "
        "IDs %s. URL-extracted IDs are never dropped.",
        MAX_FINANCIAL_IDENTIFIERS,
        total,
        dropped,
    )
    return (kept_tickers, kept_fred)


def _sanitize_classifier_ids(items: list[str], char_re: re.Pattern[str], kind: str) -> list[str]:
    """Drop classifier IDs that don't fully match the extraction char class (log-and-skip).

    Classifier IDs come from comma-splitting with no char validation, so a garbled
    token (e.g. one containing `-->`) could close the routing HTML comment and leak
    its tail as visible markdown, or be sent pointlessly to yfinance/FRED. Filter to
    the same char classes the URL-extraction regexes enforce, warning on each drop so
    a malformed classifier emission is visible rather than silently dropped.
    """
    sanitized: list[str] = []
    for item in items:
        if char_re.fullmatch(item):
            sanitized.append(item)
        else:
            logger.warning("financial classifier emitted malformed %s %r — dropping", kind, item)
    return sanitized


def extract_financial_identifiers_from_criteria(text: str) -> dict[str, list[str]]:
    """Deterministically extract the resolving FRED series / Yahoo tickers cited in ``text``.

    Shares the known-API registry's parsers (``known_api.parse``) so the extraction and the
    rung-0 translation cannot drift: it reads the series page, the graph CSV/XLS and ALFRED
    forms for FRED, and the quote pages (regional hosts included) and the chart data endpoint for
    Yahoo. Returns ``{"tickers": [...], "fred_series": [...]}``, deduped, order-preserving.
    """
    return {"tickers": known_api_parse.yahoo_symbols(text), "fred_series": known_api_parse.fred_series_ids(text)}


# Exchange-rate routing rule, interpolated into CLASSIFIER_PROMPT below.
#
# The reference tables carry NO exchange-rate FRED series at all and only three currency crosses, so
# on a question about any other currency the classifier had nothing to route to and invented an id:
# q45363 got `DEXBOUS`, which does not exist on FRED, with no Yahoo cross beside it, and therefore no
# financial block at all on a currency question (see fx_identifiers for the full receipt). Naming
# Yahoo's two FX spellings and forbidding an unsourced FRED FX id is the fix at the cause, because the
# currency's ISO code is not recoverable downstream: FRED's country codes are not ISO currency codes,
# and the `BO` in `DEXBOUS` is a country rather than a currency, so the classifier is the only step in
# the pipeline that can name the pair.
_FX_ROUTING_RULE = """EXCHANGE-RATE QUESTIONS: route these to a Yahoo FX ticker, never to a FRED series unless the resolution criteria name one by URL. Yahoo serves every cross in two spellings that mean opposite directions: USD<ISO>=X (equivalently <ISO>=X) quotes units of the foreign currency per US dollar, and <ISO>USD=X quotes US dollars per unit of it. Emit the spelling that matches how the question quotes the rate - a question asking how many bolivianos one US dollar buys is USDBOB=X, a question asking the dollar value of one boliviano is BOBUSD=X. This holds for every currency, including ones absent from the reference table above.

NEVER invent a FRED series ID. Emit only IDs listed in the reference table above, or ones a URL in the resolution criteria names explicitly."""

# Built from _TICKER_GROUPS / _FRED_GROUPS so the prompt's reference table and the
# KNOWN_* allowlist share one source of truth (the f-string-injected blocks carry no
# `{...}` of their own; the trailing {question_text}/{resolution_criteria}/{fine_print}
# remain str.format placeholders filled by _classify_financial_question).
CLASSIFIER_PROMPT = f"""You are a classifier that determines whether a forecasting question involves financial markets or economic indicators that can be looked up via stock/index/commodity tickers or FRED economic data series.

Respond in EXACTLY this format (3 lines, no extra text):
FINANCIAL: YES or NO
TICKERS: comma-separated yfinance tickers, or NONE
FRED_SERIES: comma-separated FRED series IDs, or NONE

REFERENCE TABLE of common tickers and FRED series:

{_build_ticker_reference_lines()}

FRED series:
{_build_fred_reference_lines()}

{_FX_ROUTING_RULE}

Only output YES if there are specific tickers or FRED series that would provide useful data.
If the question is about general economic trends without specific measurable indicators, output NO.

Question: {{question_text}}

Resolution criteria (may name the exact resolving source/series):
{{resolution_criteria}}
{{fine_print}}"""


async def _classify_financial_question(
    question_text: str,
    classifier_llm: GeneralLlm,
    resolution_criteria: str = "",
    fine_print: str = "",
) -> tuple[dict[str, list[str]] | None, str | None]:
    """Use an LLM to determine if a question involves trackable financial/economic data.

    Returns ``(classification, error_reason)``: the classification is
    ``{"tickers": [...], "fred_series": [...]}`` or None when the question isn't
    financial, and ``error_reason`` is the exception type name when the call FAILED
    (None otherwise). Resolution criteria / fine print are passed through so the
    classifier also sees the resolving source (belt-and-suspenders to the deterministic
    extraction).

    The two None cases have to be told apart by the caller: "read as non-financial" is a
    normal outcome, while a dead classifier (a model retirement — the 2026-05-15 grok 404
    precedent — a schema change, a quota) is a lost source that must render on the
    diagnostics line. They used to be indistinguishable, so a persistently broken
    classifier looked exactly like a question with no financial angle.
    """
    try:
        prompt = CLASSIFIER_PROMPT.format(
            question_text=question_text,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
        )
        # Wrapped with the elapsed-gated transient retry (litellm #14895): the
        # classifier LLM is built allowed_tries=1, so this wrapper is its sole
        # retry layer and also supplies the wall-clock cap the call previously
        # lacked. FINANCIAL_CLASSIFIER_TIMEOUT doubles as the wall cap (no
        # separate constant exists; the per-request timeout is the natural bound).
        response = await invoke_with_transient_retry(
            lambda: classifier_llm.invoke(prompt),
            wall_timeout=FINANCIAL_CLASSIFIER_TIMEOUT,
            label="financial_classifier",
        )
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # provider soft-fail boundary, logged
        logger.warning("Financial classifier LLM call failed", exc_info=True)
        return (None, type(exc).__name__)

    return (_parse_classifier_response(response), None)


def _parse_classifier_response(response: str) -> dict[str, list[str]] | None:
    """Parse the structured 3-line classifier response into a dict or None."""
    lines: dict[str, str] = {}
    for raw_line in response.strip().splitlines():
        line = raw_line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            lines[key.strip().upper()] = value.strip().upper()

    financial_flag = lines.get("FINANCIAL", "")
    if not financial_flag.startswith("YES"):
        return None

    tickers = _parse_csv_field(lines.get("TICKERS", "NONE"))
    fred_series = _parse_csv_field(lines.get("FRED_SERIES", "NONE"))

    if not tickers and not fred_series:
        return None

    logger.debug(f"Financial classifier: {tickers=}, {fred_series=}")
    return {"tickers": tickers, "fred_series": fred_series}


def _parse_csv_field(raw: str) -> list[str]:
    """Parse a comma-separated field, returning [] for 'NONE' or empty."""
    if not raw or raw == "NONE":
        return []
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item and item != "NONE"]


def _yfinance_history(ticker_obj: Any, window_end: datetime, *, is_benchmarking: bool) -> pd.DataFrame:
    """Daily bars from ``FINANCIAL_YFINANCE_LOOKBACK_DAYS`` before ``window_end``.

    Benchmarking additionally ceilings the window at ``window_end``; live leaves
    ``end`` unset so yfinance includes today's partial bar.
    """
    start = (window_end - timedelta(days=FINANCIAL_YFINANCE_LOOKBACK_DAYS)).date()
    if is_benchmarking:
        end = (window_end + timedelta(days=1)).date()  # yfinance end is EXCLUSIVE → +1d makes as_of inclusive
        return ticker_obj.history(start=start.isoformat(), end=end.isoformat(), timeout=TS_ANCHOR_HTTP_TIMEOUT)
    return ticker_obj.history(start=start.isoformat(), timeout=TS_ANCHOR_HTTP_TIMEOUT)


def _yfinance_latest_lines(
    ticker: str,
    close: pd.Series,
    reference_date: date,
    *,
    name: str,
    periods_per_year: int,
    is_benchmarking: bool,
) -> list[str]:
    """Header, optional name, the dated latest price, and the stale-observation warning."""
    parts = [f"### {ticker}"]
    if name:
        parts.append(f"**{name}**")
    current_price = close.iloc[-1]
    latest_obs_date = cast(pd.Timestamp, close.index[-1]).date()
    # Every "latest" carries its observation date: an undated headline reads as live
    # even when Yahoo's newest bar is days old (a weekend Friday close, or a
    # null-close hole silently dropped from the frame).
    latest_line = f"- Latest price: {current_price:.2f} (as of {latest_obs_date.isoformat()})"
    if not is_benchmarking and latest_obs_date == reference_date:
        latest_line += " — today's bar, in progress"
    parts.append(latest_line)

    stale_age = stale_latest_age_days(latest_obs_date, reference_date, periods_per_year)
    if stale_age is not None:
        unit = daily_step_unit(periods_per_year)
        parts.append(
            f"- ⚠ Latest observation is {stale_age} days old — beyond what a {unit} cadence "
            "explains; treat the latest price as stale."
        )
        logger.warning(
            f"FINANCIAL_STALE_LATEST: surface=financial_data symbol={ticker} age_d={stale_age} cadence={unit}"
        )
    return parts


def _volatility_lines(close: pd.Series, periods_per_year: int, *, symbol: str) -> list[str]:
    """Annualized volatility at two horizons, plus the vendor-noise flag when it fires.

    Two horizons because a single 30-row window is both a noisy estimate and, on a thin
    series, a systematically inflated one: q44797 published a 43-day forecast sized off a
    17.8% figure computed on 30 rows of a pegged cross, where the same series over a year
    read 15.2% and the liquid anchor read 10.6-12.9%.

    The noise flag is the variance-ratio screen (``FINANCIAL_VARIANCE_RATIO_*``). When it
    fires the ordering inverts: the noise-robust volatility (measured on multi-day returns,
    over which the reversing component cancels) leads, the long window follows, and the
    short window comes last carrying the noise-suspect label — so the number nearest the top
    is the one a forecaster should size an interval from. Unflagged, the short window stays
    first, exactly as before.
    """
    unit = daily_step_unit(periods_per_year)
    # Volatility over the trailing FINANCIAL_YFINANCE_RECENT_DAYS observations — the
    # shared estimator (ts_estimators), so this line and the anchor stack's vol note
    # cannot drift apart again; None when the return sample is shorter than the window
    # (a vol wearing the window's label without its sample size). Name the step unit:
    # FINANCIAL_YFINANCE_RECENT_DAYS is a ROW count, which is six calendar weeks on an
    # exchange-traded series and 30 calendar days on a 24/7 one, so a bare "30-day" label
    # was itself a row count posing as a calendar window.
    short_vol = annualized_realized_vol_pct(
        close, window=FINANCIAL_YFINANCE_RECENT_DAYS, periods_per_year=periods_per_year
    )
    if short_vol is None:
        return []
    short_line = f"- {FINANCIAL_YFINANCE_RECENT_DAYS}-{unit} annualized volatility: {short_vol:.1f}%"

    # The long horizon: one year of returns, or everything the fetch actually holds when
    # that is less. Capped at a year so the label stays a period a forecaster can reason
    # about, and skipped when it would not clear the short window (two windows of the same
    # length are one number printed twice).
    long_window = min(len(close) - 1, periods_per_year)
    long_vol = (
        annualized_realized_vol_pct(close, window=long_window, periods_per_year=periods_per_year)
        if long_window > FINANCIAL_YFINANCE_RECENT_DAYS
        else None
    )
    long_line = None if long_vol is None else f"- {long_window}-{unit} annualized volatility: {long_vol:.1f}%"

    screen = screen_for_quote_noise(close, periods_per_year=periods_per_year)
    if screen is None:
        return [short_line] if long_line is None else [short_line, long_line]

    robust_vol = screen.robust_vol_pct
    flagged = [
        f"- ⚠ Vendor-noise flag: variance ratio VR({FINANCIAL_VARIANCE_RATIO_LAG}) = {screen.ratio:.2f} over "
        f"{len(close) - 1} {unit} returns. A random walk reads ~1.0; below "
        f"{FINANCIAL_VARIANCE_RATIO_FLOOR:.2f} most of each day's move is reversed the next, which is quote "
        "noise on an illiquid or fixed cross rather than genuine price movement, and it inflates every "
        "volatility computed from one-day returns."
    ]
    if robust_vol is not None:
        flagged.append(
            f"- Noise-robust annualized volatility, from overlapping {FINANCIAL_VARIANCE_RATIO_LAG}-"
            f"{unit} returns (the horizon over which that reversing component cancels): "
            f"{robust_vol:.1f}% — size intervals from THIS figure, not the one-day-return ones below."
        )
    if long_line is not None:
        flagged.append(f"{long_line} (from one-day returns, noise included)")
    flagged.append(f"{short_line} (from one-day returns, noise included; noise-suspect)")
    logger.info(
        noise_flag_line(screen, surface="financial_data", symbol=symbol, short_vol=short_vol, long_vol=long_vol)
    )
    return flagged


def _yfinance_stats_lines(close: pd.Series, periods_per_year: int, *, symbol: str) -> list[str]:
    """Period returns, annualized volatility, and the 52-week range.

    ``symbol`` is threaded purely for the noise flag's telemetry line, which is otherwise
    anonymous: nothing on ``close`` names the ticker (its ``.name`` is "Close") and the fetch
    fans out one thread per identifier, so line order does not identify it either.
    """
    parts: list[str] = []
    # Period returns
    returns_section = _compute_period_returns(close, periods_per_year)
    if returns_section:
        parts.append(returns_section)

    parts.extend(_volatility_lines(close, periods_per_year, symbol=symbol))

    # 52-week range, windowed by DATE like the period returns: a row-count slice
    # under a fixed "52-week" label spans ~13 months on a gapped 24/7 series and
    # over a year on a holiday-bearing exchange index. Never empty — the last
    # observation is always inside its own trailing year.
    year_slice = close[close.index >= close.index[-1] - pd.Timedelta(days=CALENDAR_DAYS_PER_YEAR)]
    low_52w = year_slice.min()
    high_52w = year_slice.max()
    parts.append(f"- 52-week range: {low_52w:.2f} - {high_52w:.2f}")
    return parts


def _yfinance_fundamentals_lines(close: pd.Series, info: dict, *, is_benchmarking: bool) -> list[str]:
    """The .info fundamentals block (live only) and the last five closes."""
    parts: list[str] = []
    # Optional fundamentals from .info (live only; `info` is {} under benchmarking).
    if is_benchmarking:
        parts.append("- Fundamentals: [omitted under backtest — .info has no historical mode]")
    else:
        fundamentals = _format_fundamentals(info)
        if fundamentals:
            parts.append(fundamentals)

    # Last 5 closing prices
    last_5 = close.tail(5)
    closing_lines = [
        f"  - {cast(pd.Timestamp, date).strftime('%Y-%m-%d')}: {price:.2f}" for date, price in last_5.items()
    ]
    parts.append("- Last 5 closes:\n" + "\n".join(closing_lines))
    return parts


def _fetch_yfinance_data(ticker: str, *, as_of: datetime | None = None, is_benchmarking: bool = False) -> str:
    """One ticker's markdown block, plus the peg anchor's block when the pair is pegged.

    Sync function -- caller wraps in asyncio.to_thread(). Returns "" when the ticker's own
    fetch fails, exactly as before; a peg anchor that fails to fetch degrades to a visible
    one-line notice rather than taking the pegged pair's block down with it.

    The anchor is rendered BESIDE the pegged pair, never substituted for it: the question
    resolves on the pegged cross, so its own quote has to stay on the page. Only the
    interpretation changes, and the peg disclosure inside the block says so.
    """
    block = _render_yfinance_block(ticker, as_of=as_of, is_benchmarking=is_benchmarking)
    if not block:
        return ""
    peg = peg_for_ticker(ticker)
    if peg is None or peg.anchor_ticker is None:
        return block
    # Recursion is bounded at one level by construction: no anchor ticker is itself a key of
    # currency_pegs.HARD_PEG_ANCHORS (asserted in tests), so the anchor's own render finds no peg.
    anchor_block = _render_yfinance_block(peg.anchor_ticker, as_of=as_of, is_benchmarking=is_benchmarking)
    if not anchor_block:
        logger.warning(f"peg anchor fetch returned nothing for {peg.anchor_ticker=} (pegged {ticker=})")
        return (
            f"{block}\n- ⚠ The peg anchor `{peg.anchor_ticker}` could not be fetched, so no clean read of "
            "this pair's dynamics is available in this block."
        )
    return (
        f"{block}\n\n_Peg anchor for {ticker} — the liquid cross the peg is fixed to. Its volatility and "
        f"percent moves are the honest read of USD/{peg.currency}'s; its price LEVELS are a different "
        f"quantity unless the peg is at par._\n{anchor_block}"
    )


def _render_yfinance_block(ticker: str, *, as_of: datetime | None = None, is_benchmarking: bool = False) -> str:
    """Fetch price data and key metrics for a single ticker via yfinance.

    Returns formatted markdown or "" on any failure.

    Both paths fetch by explicit calendar start date, ``as_of`` -
    ``FINANCIAL_YFINANCE_LOOKBACK_DAYS`` (``as_of`` defaults to now; the live provider
    passes now explicitly). A bare ``period="Nd"`` is deliberately avoided: Yahoo's
    chart API reads that custom range as N trading BARS for listed assets but ~N
    calendar DATES for 24/7 ones — one integer under two unit systems. Live
    (``is_benchmarking=False``) leaves ``end`` unset (yfinance defaults it to now, so
    today's partial bar stays included) and reads the live ``.info`` fundamentals.
    Backtest (``is_benchmarking=True``) ceilings the history at ``as_of`` via ``end``
    and SKIPs the ``.info`` call entirely — ``.info`` has no historical mode, so its
    market cap / P/E / current price would leak TODAY's values into a question
    resolved months ago. The latest price then comes from the last ceilinged close,
    and every derived stat (returns / vol / 52wk) computes off that frame.
    """
    try:
        ticker_obj = yfinance.Ticker(ticker)
        if is_benchmarking:
            assert as_of is not None, "benchmarking yfinance fetch requires as_of"
        window_end = as_of if as_of is not None else datetime.now(UTC)
        history = _yfinance_history(ticker_obj, window_end, is_benchmarking=is_benchmarking)

        if history.empty:
            logger.warning(f"yfinance returned empty history for {ticker=}")
            return ""

        # Skip the live `.info` call under benchmarking (no historical mode → leakage).
        info: dict = {} if is_benchmarking else (ticker_obj.info or {})
        # Yahoo can serve a bar whose close is null (a consolidation hole); yfinance's
        # keepna=False default usually deletes the row, but a NaN that does arrive would
        # poison every tail-anchored stat below, so drop them here too.
        close = history["Close"].dropna()
        if close.empty:
            logger.warning(f"yfinance returned no non-null closes for {ticker=}")
            return ""
        # yfinance serves listed-asset bars on a tz-aware exchange-local index while
        # window_end is UTC; the two dates disagree for part of every day, so age and
        # the partial-bar check compare in the index's own timezone (a 00:03 UTC cron
        # would otherwise inflate every US-equity age by one).
        index_tz = cast(pd.DatetimeIndex, close.index).tz
        reference_date = window_end.astimezone(index_tz).date() if index_tz is not None else window_end.date()

        periods_per_year = observed_periods_per_year(close.index)
        parts = _yfinance_latest_lines(
            ticker,
            close,
            reference_date,
            name=info.get("shortName", ""),
            periods_per_year=periods_per_year,
            is_benchmarking=is_benchmarking,
        )
        # Before the stats, because it changes how every one of them reads.
        peg = peg_for_ticker(ticker)
        if peg is not None:
            parts.extend(peg_disclosure_lines(peg))
        parts.extend(_yfinance_stats_lines(close, periods_per_year, symbol=ticker))
        parts.extend(_yfinance_fundamentals_lines(close, info, is_benchmarking=is_benchmarking))
        return "\n".join(parts)

    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except  # provider soft-fail boundary, logged
        logger.warning(f"yfinance fetch failed for {ticker=}", exc_info=True)
        return ""


# Annualization bases for a daily-bar series: exchange-traded assets print ~252
# rows/year (5 trading days a week), 24/7 markets (crypto) print ~365. Annualizing
# a 24/7 series with sqrt(252) understates vol by ~17% (sqrt(252/365) ~= 0.83), and
# row-count windows labeled "1y"/"52-week" then span only ~8.2 calendar months —
# both bit q44882 (ETH-USD). The two constants AND the observed-density read that picks
# between them are the package's shared definitions, imported from ts_estimators above, so a
# correction cannot miss a copy — the ts-anchor stack had exactly that miss until 2026-08-25.

# Calendar days back behind each period-return label, on BOTH bases. Each count is
# resolved to a target DATE and matched at-or-before — never used as a row offset,
# because a gapped index (Yahoo's dropped null-close bar, an unscheduled closure)
# shifts every row offset by the hole count, which is how a "1d" return spanning 53
# hours once rendered on a BTC-USD snapshot. Calendar days rather than business days
# deliberately: `pd.offsets.BDay` skips only weekends and knows nothing about market
# HOLIDAYS, so a business-day 1y target on a NYSE-shaped index landed ~352 calendar
# days back under a label claiming a year, at slip 0 — silently. With calendar
# targets, every label is a true calendar period on both bases, and the weekend or
# holiday under a 252-basis target is absorbed by the at-or-before match within the
# slip grace below.
_PERIOD_TARGET_DAYS: list[tuple[str, int]] = [
    ("1d", 1),
    ("1w", 7),
    ("1m", 30),
    ("3m", 91),
    ("6m", 182),
    ("1y", CALENDAR_DAYS_PER_YEAR),
]

# Calendar days a period-return match may land before its target date before the label
# discloses the actual span. On the 252 basis a target can fall on a weekend day with a
# market holiday under the adjacent Friday — a 3-day slip that is routine, not a
# mislabel. On the 365 basis every date should print a bar, so ANY slip is a data gap
# the label must own up to.
_PERIOD_SLIP_GRACE_DAYS: dict[int, int] = {TRADING_DAYS_PER_YEAR: 3, CALENDAR_DAYS_PER_YEAR: 0}


def _compute_period_returns(close: pd.Series, periods_per_year: int) -> str:
    """Compute returns over standard periods via date-based at-or-before lookups.

    ``periods_per_year`` is REQUIRED (no trading-day default): it selects the slip
    grace that separates routine weekend/holiday slippage from a data gap, so a caller
    that forgets it would silently reproduce the q44882 mislabelling on a 24/7 series.
    Each label's start value is the newest observation at or before ``last - days``
    (the same pattern as the FRED YoY lookup below); a label whose match lands beyond
    the basis's slip grace discloses the actual span ("1d (actual 2d)") instead of
    wearing a period it doesn't cover. A label with no observation at or before its
    target is omitted, exactly as the old too-few-rows guard omitted it.
    """
    last_ts = cast(pd.Timestamp, close.index[-1])
    end_price = close.iloc[-1]
    lines = []
    for label, days in _PERIOD_TARGET_DAYS:
        target = last_ts - pd.Timedelta(days=days)
        prior = close.loc[:target]
        if prior.empty:
            continue
        start_ts = cast(pd.Timestamp, prior.index[-1])
        pct_change = (end_price / prior.iloc[-1] - 1) * 100
        slipped = (target - start_ts).days > _PERIOD_SLIP_GRACE_DAYS[periods_per_year]
        span = f" (actual {(last_ts - start_ts).days}d)" if slipped else ""
        lines.append(f"  - {label}{span}: {pct_change:+.2f}%")
    if lines:
        return "- Period returns:\n" + "\n".join(lines)
    return ""


def _format_fundamentals(info: dict) -> str:
    """Extract optional fundamental metrics from yfinance .info dict."""
    lines = []
    pe = info.get("trailingPE")
    if pe is not None:
        lines.append(f"  - P/E ratio: {pe:.1f}")
    market_cap = info.get("marketCap")
    if market_cap is not None:
        if market_cap >= 1e12:
            lines.append(f"  - Market cap: ${market_cap / 1e12:.2f}T")
        elif market_cap >= 1e9:
            lines.append(f"  - Market cap: ${market_cap / 1e9:.2f}B")
        else:
            lines.append(f"  - Market cap: ${market_cap / 1e6:.0f}M")
    fwd_eps = info.get("forwardEps")
    if fwd_eps is not None:
        lines.append(f"  - Forward EPS: {fwd_eps:.2f}")
    if lines:
        return "- Fundamentals:\n" + "\n".join(lines)
    return ""


def _resolve_fetch_as_of(question: MetaculusQuestion, *, is_benchmarking: bool) -> datetime | None:
    """The window ceiling for every fetch, or None when benchmarking cannot be made leakage-safe.

    Ceiling every fetch to open_time under benchmarking (leakage-safe, mirrors
    timeseries_anchor). A missing open_time can't be ceilinged, so we bail rather
    than risk fetching today's data into a resolved question.
    """
    if not is_benchmarking:
        return datetime.now(UTC)
    open_time = getattr(question, "open_time", None)
    if not isinstance(open_time, datetime):
        logger.warning(
            "financial_data: is_benchmarking but qid=%s has no open_time; skipping (leakage-safe)",
            getattr(question, "id_of_question", None),
        )
        return None
    return open_time


FRED_SKIPPED_NO_KEY_TOKEN = "skipped(no_fred_api_key)"  # noqa: S105  # diagnostics source token, not a credential


def _build_financial_fetch_jobs(
    tickers: list[str],
    fred_series: list[str],
    *,
    as_of: datetime,
    is_benchmarking: bool,
    resolving_fred_series: frozenset[str] = frozenset(),
) -> tuple[list[tuple[str, asyncio.Task]], dict[str, str]]:
    """Spawn one fetch task per identifier, each paired with the ticker/FRED id it fetches.

    ``resolving_fred_series`` holds the UPPER-CASED ids the resolution criteria cited by URL
    — the series a question actually grades against — which is what earns the extra
    first-release/vintage fetch on the live path.

    Returns ``(jobs, not_fetched)``, where ``not_fetched`` maps every identifier that never
    became a job to its ``details["sources"]`` loss token. Only the missing-``FRED_API_KEY``
    case populates it today, and it exists because a skipped series used to leave NO source
    token at all: N requested series vanished from the diagnostics map, so the line read as
    fully healthy while nothing had been fetched for them.
    """
    jobs: list[tuple[str, asyncio.Task]] = [
        (
            ticker,
            asyncio.ensure_future(
                asyncio.to_thread(_fetch_yfinance_data, ticker, as_of=as_of, is_benchmarking=is_benchmarking)
            ),
        )
        for ticker in tickers
    ]
    if is_benchmarking:
        # Keyless ceilinged path — no FRED_API_KEY needed, works in CI, and returns
        # point-in-time vintages instead of today's revisions.
        jobs.extend(
            (series_id, asyncio.ensure_future(asyncio.to_thread(_fetch_fred_data_ceiling, series_id, as_of)))
            for series_id in fred_series
        )
        return (jobs, {})
    fred_api_key = os.getenv(FRED_API_KEY_ENV)
    if fred_api_key:
        jobs.extend(
            (
                series_id,
                asyncio.ensure_future(
                    asyncio.to_thread(
                        _fetch_fred_data,
                        series_id,
                        fred_api_key,
                        is_resolving_source=series_id.upper() in resolving_fred_series,
                    )
                ),
            )
            for series_id in fred_series
        )
    elif fred_series:
        logger.info(f"FRED_API_KEY not set, skipping {len(fred_series)} FRED series fetches")
        return (jobs, dict.fromkeys(fred_series, FRED_SKIPPED_NO_KEY_TOKEN))
    return (jobs, {})


async def _gather_financial_results(jobs: list[tuple[str, asyncio.Task]]) -> tuple[list[str], dict[str, str]]:
    """Await every fetch, returning the non-empty sections plus a per-identifier outcome map.

    Per-identifier outcome for the diagnostics block: a requested ticker/FRED
    series that errored or returned no data is a lost source, so a partial
    financial fetch stays visible even when other identifiers succeed.

    A FRED id that does not exist gets its own ``unknown_series`` token rather than the generic
    ``error``: it means an id was hallucinated (or a resolution URL is dead), which is a defect to
    chase, while ``error`` is an upstream failure to retry. The WARN naming the id was already
    emitted at the fetch site, so nothing is logged again here.
    """
    results = await asyncio.gather(*(task for _, task in jobs), return_exceptions=True)
    non_empty_results: list[str] = []
    sources: dict[str, str] = {}
    for (identifier, _), result in zip(jobs, results, strict=True):
        if isinstance(result, UnknownFredSeries):
            sources[identifier] = "unknown_series"
            continue
        if isinstance(result, Exception):
            logger.warning(f"Financial data fetch task failed: {result}")
            sources[identifier] = "error"
            continue
        if isinstance(result, str) and result.strip():
            non_empty_results.append(result)
            sources[identifier] = "ok"
        else:
            sources[identifier] = "empty"
    return non_empty_results, sources


def financial_data_provider(is_benchmarking: bool = False) -> ResearchCallable:
    """Factory function returning an async research callable for financial/economic data.

    The callable:
    1. Classifies whether the question involves financial data (via cheap LLM).
    2. Fetches relevant data from yfinance and/or FRED in parallel.
    3. Combines results into structured markdown.

    FRED gracefully degrades if FRED_API_KEY is not set.

    Backtest-safe like ``timeseries_anchor``: under ``is_benchmarking`` every fetch is
    ceilinged to ``question.open_time`` (NOT the resolution time — post-open data can
    contain the resolution). Both modes fetch yfinance by explicit start date
    (``as_of`` - ``FINANCIAL_YFINANCE_LOOKBACK_DAYS``); benchmarking additionally
    ceilings the window via ``end`` and skips the leaky ``.info`` fundamentals. FRED:
    live uses the fredapi path, while benchmarking routes through the keyless
    ``ts_fetch`` fredgraph / ALFRED-vintage path so revised macro series return the
    vintage known at forecast time rather than today's revisions.
    """
    classifier_llm = build_llm_with_openrouter_fallback(
        model=FINANCIAL_CLASSIFIER_MODEL,
        role="financial_classifier",
        # temperature=None defers reasoning models to provider defaults; redundant
        # on ft 0.2.92 (GeneralLlm ctor default is already None). No top_p.
        temperature=None,
        # No max_tokens (2026-09-22): reasoning tokens counted against the old 500 cap; the timeout bounds a runaway.
        reasoning={"effort": "low"},
        timeout=FINANCIAL_CLASSIFIER_TIMEOUT,
        # allowed_tries=1 so the elapsed-gated transient retry in
        # _classify_financial_question is the SOLE retry layer. Without this the
        # builder defaults to allowed_tries=2, whose unguarded tenacity would
        # retry a slow stall — the exact failure mode the elapsed gate prevents.
        allowed_tries=1,
    )

    async def _fetch(question: MetaculusQuestion) -> str:
        as_of = _resolve_fetch_as_of(question, is_benchmarking=is_benchmarking)
        if as_of is None:
            return ""

        extracted = extract_financial_identifiers_from_criteria(
            f"{question.resolution_criteria or ''}\n{question.fine_print or ''}"
        )

        classification, classifier_error = await _classify_financial_question(
            question.question_text,
            classifier_llm,
            resolution_criteria=question.resolution_criteria or "",
            fine_print=question.fine_print or "",
        )
        qid = getattr(question, "id_of_question", None)
        # Built once and re-seeded into the final record at the _assemble_financial_output
        # call below: this ``if`` is the ONLY producer of a ``classifier`` key, since
        # ``sources`` otherwise holds identifier keys only.
        classifier_loss = {"classifier": f"error({classifier_error})"} if classifier_error is not None else {}
        if classifier_loss:
            # Recorded HERE, above the empty-fetch-set early return below, because that
            # return happens before the per-identifier record_provider_detail call at the
            # end — so a dead classifier on a question with no extracted identifiers
            # produced NO diagnostics detail at all and read as "no financial angle".
            # record_provider_detail assigns the whole dict, so a later record for this qid
            # REPLACES this one rather than merging — which is why the same token is merged
            # into the per-identifier map below rather than left to survive on its own. It
            # used to be dropped there, so a dead classifier beside a fetched identifier
            # archived a record byte-identical to a fully healthy provider.
            record_provider_detail(qid, "financial_data", {"sources": dict(classifier_loss)})

        # Sanitize classifier IDs (F2) BEFORE they reach the fetch set, marker, or
        # unknown-flagging: they come from comma-splitting with no char validation,
        # unlike the regex-constrained extracted IDs. A malformed token (e.g. one
        # containing `-->`) would otherwise leak into the HTML-comment marker.
        classifier_tickers = _sanitize_classifier_ids(
            classification["tickers"] if classification else [], _TICKER_CHARS_RE, "ticker"
        )
        classifier_fred = _sanitize_classifier_ids(
            classification["fred_series"] if classification else [], _FRED_CHARS_RE, "FRED series"
        )

        # Extraction is ADDITIVE, not a replacement: the classifier may legitimately
        # add a Yahoo proxy for richer context, but extracted IDs guarantee the
        # resolving series is in the fetch set.
        tickers = _dedupe_preserving_order(classifier_tickers + extracted["tickers"])
        fred_series = _dedupe_preserving_order(classifier_fred + extracted["fred_series"])
        tickers, fred_series = _cap_identifiers(tickers, fred_series, extracted)

        # The deterministic extraction is the load-bearing guarantee: even when the
        # classifier returns None (question read as non-financial) or misroutes, the
        # source the question RESOLVES ON must still fire. So we only bail when the
        # merged fetch set is empty.
        if not tickers and not fred_series:
            return ""

        # Soft-fail loudly: classifier IDs not in the reference allowlist (and not
        # independently confirmed by extraction) are flagged but still fetched —
        # a valid-but-unlisted series shouldn't be dropped, just made visible.
        unknown = _flag_unknown_classifier_ids(classifier_tickers, classifier_fred, extracted)

        jobs, not_fetched = _build_financial_fetch_jobs(
            tickers,
            fred_series,
            as_of=as_of,
            is_benchmarking=is_benchmarking,
            resolving_fred_series=frozenset(sid.upper() for sid in extracted["fred_series"]),
        )

        non_empty_results, sources = await _gather_financial_results(jobs)
        sources.update(not_fetched)
        return _assemble_financial_output(
            non_empty_results,
            {**classifier_loss, **sources},
            qid=qid,
            marker=_build_routing_marker(fred_series, tickers, extracted, unknown),
        )

    return _fetch


def _assemble_financial_output(
    non_empty_results: list[str],
    sources: dict[str, str],
    *,
    qid: int | None,
    marker: str,
) -> str:
    """Join the fetched blocks and record the per-identifier detail; "" when nothing was fetched.

    A section with nothing in it is ABSENT, and the per-identifier loss tokens in
    ``details["sources"]`` are the whole record -- the same answer AskNews gives when both of its
    search phases come back with no articles (``research/providers.py``). Prose must never stand in
    for that absence, including on the exchange-rate question this helper's own predicates exist
    for: a non-empty return flips the orchestrator's status from ``empty`` to ``ok``, counts the
    provider in ``providers_succeeded``, and defeats every downstream empty guard at once, which is
    why AskNews' old ``No articles were found`` sentence was removed rather than reworded.

    ``sources`` is per-IDENTIFIER with one exception: the caller merges in a dead classifier's
    ``classifier`` token, since ``record_provider_detail`` assigns the whole dict and this record
    would otherwise overwrite the earlier classifier-only one. It is not an identifier, so the FX
    count below skips it, while ``is_lost_source`` still renders it in the diagnostics ``lost=``
    suffix.

    ``counts["fx_identifiers_empty"]`` is what carries q45363's signal instead: how many attempted
    EXCHANGE-RATE identifiers came back with nothing in them (shape by ``is_fx_identifier``, outcome
    by the canonical ``is_lost_source``, so ``empty`` / ``unknown_series`` / ``error`` /
    ``skipped(no_fred_api_key)`` all count). It is recorded on every path, so a 0 means the check ran
    rather than that it never did, and it does not depend on whether the section rendered -- an FX
    identifier lost beside a ticker that rendered fine is the same partial gap and reads the same
    way.
    """
    empty_fx_identifiers = sum(
        1 for identifier, token in sources.items() if is_fx_identifier(identifier) and is_lost_source(token)
    )
    record_provider_detail(
        qid,
        "financial_data",
        {"sources": sources, "counts": {"fx_identifiers_empty": empty_fx_identifiers}},
    )
    body = "\n\n".join(non_empty_results)
    return body + marker if body else ""


def _flag_unknown_classifier_ids(
    classifier_tickers: list[str],
    classifier_fred: list[str],
    extracted: dict[str, list[str]],
) -> list[str]:
    """Return classifier IDs not in the reference allowlist nor confirmed by extraction.

    Logs a WARNING per unrecognized ID (soft-fail loudly). Caller still fetches them.
    """
    unknown: list[str] = []
    for ticker in classifier_tickers:
        if ticker not in KNOWN_TICKERS and ticker not in extracted["tickers"]:
            unknown.append(ticker)
            logger.warning("financial classifier emitted unrecognized ticker %r — fetching anyway but flagging", ticker)
    for series_id in classifier_fred:
        if series_id not in KNOWN_FRED_SERIES and series_id not in extracted["fred_series"]:
            unknown.append(series_id)
            logger.warning(
                "financial classifier emitted unrecognized FRED series %r — fetching anyway but flagging", series_id
            )
    return unknown


def _build_routing_marker(
    fred_series: list[str],
    tickers: list[str],
    extracted: dict[str, list[str]],
    unknown: list[str],
) -> str:
    """Build a forecaster-invisible, greppable routing marker (Part D observability).

    An HTML comment is invisible in rendered markdown but survives verbatim into
    research_text — the cached blob, the persisted artifact, and the Metaculus
    comment — so the routing decision is durable and auditable without changing
    the ResearchCallable signature.
    """
    return (
        f"\n\n<!-- financial_routing: fred=[{','.join(fred_series)}] tickers=[{','.join(tickers)}] "
        f"extracted_fred=[{','.join(extracted['fred_series'])}] "
        f"extracted_tickers=[{','.join(extracted['tickers'])}] unknown=[{','.join(unknown)}] -->"
    )
