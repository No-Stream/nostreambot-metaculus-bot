"""The rendered snapshot: a markdown table in the ranker's order, plus the rules bullets.

Seven things here are contracts rather than formatting choices:

- **Zero rendered rows returns ``""`` before any preamble is emitted.** That early return is
  what produces ``status="empty"`` downstream and the attempted-vs-succeeded distinction
  residual analysis reads off the archive. Under ranked selection a zero-row render is a
  legitimate outcome (the model is allowed to say nothing bears on the question), so this is
  a hot path, not a theoretical one. The seam (``prediction_market.format_snapshot_for_research``)
  substitutes ``render_no_relevant_market_line`` for exactly that deliberate-zero case, so the
  forecaster can tell a considered empty answer from an outage; every failure path still
  returns ``""``.
- **The rows are rendered in the ranker's order, verbatim.** No venue interleave, no
  fairness pass, no per-venue cap, no re-scoring. Round-robin venue fairness is exactly what
  lost 43 of 58 wanted rows in the measurement this port is built on.
- **A multi-outcome family's price distribution reaches the forecaster WHOLE.** No outcome is ever
  dropped: the leading outcomes get full ``↳`` sub-rows and every remaining outcome is named, with
  its own price, in one ``↳ [remaining N]`` ladder row. That is the 2026-08-25 reversal, and it is
  the module's most load-bearing contract. A family is a distribution over its own outcome space,
  its forecast content is the SHAPE of that distribution, and no subset carries a shape — so
  truncating from the end was answering the wrong question. Measured: 108 of 162 archived families
  (67%) were truncated, and truncation correlated WITH relevance (81% of ``same_quantity_other_cut``
  families versus 50% of ``weak`` ones). On q45189 all three forecasters read the one surviving
  bracket of a ten-bracket margin ladder as an equality constraint and cut the resolving bucket
  below their own prior. Presentation lives HERE, not in the venue parsers: the renderer sorts a copy
  by ``child_render_order_key`` for the full rows and reads the venue's catalogue order for the
  ladder. Under character pressure the ladder collapses groups in increasing order of forecast
  content (unquoted, then settled, then the cheapest open outcomes), and every collapsed group states
  its count and its summed price — a counted set, never a silent cut.
- **A number in the ``prob`` column is a price the venue actually quoted.** Three venues publish a
  ~0.50 default for an outcome nobody has quoted — an empty Kalshi book is ``0.0000``/``1.0000``, a
  Polymarket placeholder leg is Gamma's ``["0.5","0.5"]``, a fresh Manifold answer sits at its 0.5
  prior — and 192 of 1,839 archived ranked-era child outcomes were in that class. The venue parsers blank those
  at parse time, so a fabricated 0.50 never reaches the sort, the render, or a group's summed price.
  A usable price appears with its venue-reported bid/ask range when both sides are known; a blanked
  price renders the range alone (``0.00-1.00``), which cannot be read as a point probability.
- **A ``close`` date that has already passed says so, in the cell.** The column is the venue's
  TRADING close, and on Kalshi it trails settlement by a median +317 days on the markets this bot
  has rendered, so a forecaster told to "verify the resolution date" is checking against a date that
  means something else. Worse, a market can be long dead and still read ``status=open``: Manifold's
  close dates are soft, so ``is_resolved`` stays false past them, and q45163's best-ranked row closed
  five months before the forecast with nothing in the table saying so. ``_close_cell`` appends
  ``(Nd ago)`` against the snapshot's own ``forecast_time``, and the legend carries the
  close-is-not-settlement caveat. The date itself never leaves the cell — it is what the
  date-verification instruction is checked against, and an age cannot answer that.
- **Every number in the ``prob`` column is a probability, unless it is labelled as something
  else.** The one exception is a scalar market's own value, and it renders as ``value N (scale LOW
  to HIGH)`` for exactly that reason — the forecaster prompts tell a model to anchor on this cell,
  so an unlabelled non-probability here is not a cosmetic defect. Manifold shipped one: a
  ``PSEUDO_NUMERIC`` market's ``probability`` field is a position on its value scale, and the cell
  read ``0.48`` on a market whose estimate was 120.97 years. ``_price_cell`` is the only writer.
- **The rules-bullet section keeps its byte shape** (``- **{platform}** <{url}>: {body}`` at
  h3). The MC per-model option parser scans for ``- <name>: NN%`` and the bullets sit close
  enough to a per-model region for that to matter, so nothing percentage-shaped may ever be
  APPENDED to a bullet. Nothing is: a body is the market's rules text verbatim. A multi-outcome
  market's answer probabilities used to ride inside it, parenthesised for exactly that reason,
  and moved out to the sub-rows — see ``_bullet_body``.

The preamble/legend strings mirror the ones the provider ships today, minus the fuzzy-match
vocabulary the ranker replaces. Their asserted substrings are load-bearing: the ranked design
changes how a row is chosen, not what a forecaster is told to verify.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from metaculus_bot.research.market_retrieval.ranking import (
    DEGRADED_RANKING_MARKER,
    STRONG_TIERS,
    TIER_UNSPECIFIED,
    WHY_CHARS,
)
from metaculus_bot.research.market_retrieval.types import (
    MarketChild,
    MarketMatch,
    MarketSnapshot,
    ScalarEstimate,
    _child_liquidity_label,
    _liquidity_label,
    format_scalar_number,
)
from metaculus_bot.research.market_retrieval.venues._shared import child_render_order_key
from metaculus_bot.time_utils import _as_utc

# Significant digits for a scalar market's VALUE, which is deliberately shorter than the bounds
# beside it (`types.SCALAR_BOUND_SIG_DIGITS`, which lives there because the ranker's candidate line
# renders the same bounds). The two differ because the numbers do: a value is a noisy crowd estimate,
# so six digits already say more than it means (120.96691732988944 -> `120.967`) and a short cell
# reads better, while a bound is a parameter the market's author chose and rounding it would
# misstate the market's own scale.
SCALAR_VALUE_SIG_DIGITS = 6

# Separate from, and smaller than, the ranker prompt's per-venue caps: this text ships to the
# forecaster inside a research section that has its own character budget, and the two must not be
# collapsed into one constant. A bullet is now rules text alone — a multi-outcome market's outcome
# names and prices moved out of the body and into `↳` sub-rows on 2026-08-05, where each gets its
# own price, volume, close date and status instead of a parenthesised percentage.
RAW_BULLET_BODY_MAX_CHARS = 200

# The `↳` sub-row bounds. Three separate ceilings, because they fail differently. Since 2026-08-25
# none of them decides whether an outcome's PRICE reaches the forecaster — the ladder row carries
# every remaining outcome — so all three now bound only how many outcomes keep a FULL sub-row, i.e.
# their own volume / OI / signal / close / status cells.
#
# `MAX_CHILD_ROWS_PER_MARKET` bounds ONE market: a Manifold threshold ladder ships 17 answers and a
# Kalshi strike family can carry more, and past the first handful the rungs are the tail of a
# distribution whose shape the leaders already gave. It only binds on a slate with few
# child-bearing markets.
#
# `MAX_CHILD_ROWS_PER_SNAPSHOT` bounds the SECTION, and it is the one that has to exist: 86.5% of
# the Kalshi catalogue is multi-strike, so a realistic 8-row slate is mostly parents with children,
# and the per-market cap alone would license 80 sub-rows — roughly doubling a section that goes to
# the expensive forecaster models on every question. Its value is chosen against `RENDER_BUDGET`
# rather than picked: while it is >= `RENDER_BUDGET`, no multi-outcome market can render priceless,
# and that guarantee is now doubly true — a family granted zero full rows still renders its ladder
# row. 24 -> 14 on 2026-08-25, which is where the ladder's characters come from: once a ladder row
# carries every price, a full sub-row buys only liquidity/close/status detail at ~4x the characters
# per outcome, so shifting 10 slots into the ladder allowance is the cheapest way to pay for
# completeness. Each full row a family loses keeps its price in the ladder row and gives up only its
# liquidity cells.
#
# 14 rather than the 16 the design's grid picked, and the extra 2 are what the cumulative-ladder
# collapse rule cost. That rule needs a legend clause naming the `+N off certainty by under X` group
# (~150 chars of fixed overhead on every snapshot), and at 16 the committed MAXED budget was left with
# 7 characters of slack — a budget that cannot absorb one more word is not a budget. The design's own
# grid measured 14 as the BETTER completeness setting anyway (2 of 42 archived snapshots compacting
# versus 5, and 4 families under a 0.95 named-price share versus 6), because the slots move from
# per-outcome detail to the ladder allowance that keeps outcomes named.
#
# `CHILD_TITLE_MAX_CHARS` is well under the parent's `TITLE_MAX_CHARS`: an outcome label is a rung,
# not a question ("Republican Party", "Before Nov 1, 2026", "0 (0 bps)", "$3.80 - $4.19" — 9-18
# chars measured across all four venues' committed fixtures), so this only bounds a pathological one.
# It bounds a ladder TERM's label too, for the same reason.
MAX_CHILD_ROWS_PER_MARKET = 10
MAX_CHILD_ROWS_PER_SNAPSHOT = 14
CHILD_TITLE_MAX_CHARS = 48

# The ladder row's bounds. A per-ROW cap cannot bound a SECTION, which is why there are two: on the
# committed MAXED fixture eight 436-char ladder rows each fit their own 600-char cap and the section
# landed at 13,306 against a 10,600 budget. So the ladder gets a section-level allowance, spent by
# escalating the LARGEST title one compaction stage at a time until the total fits — deterministic,
# order-independent, and it compacts where it buys the most characters.
#
# 1,400 is chosen against the archive's own ladder-title totals (median 240 chars per snapshot, p90
# 1,313, max 3,198): it leaves 37 of 42 archived snapshots entirely uncompacted and compacts the five
# widest exactly where the collapsed outcomes carry ~0 summed price. Raising it to 2,400 puts the
# MAXED fixture over its committed budget; the full grid is in the design's §6.
LADDER_SECTION_MAX_CHARS = 1_400
LADDER_ROW_MAX_CHARS = 600

# The escalating floor a ladder row collapses its least informative OPEN outcomes at, once collapsing
# the unquoted and settled groups has not freed enough characters. `0.0` is the no-op that keeps every
# open outcome named; the last entry is the most aggressive collapse any stage can ask for.
LADDER_PRICE_FLOORS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.50)

# Summed open price above which a family is read as a CUMULATIVE threshold ladder rather than a
# mutually-exclusive partition. A partition's prices sum to ~1 by construction (q45189's ten margin
# brackets sum to 0.965; fees and spreads put a real one in roughly 0.95-1.05), while a threshold
# ladder's nested prices are SURVIVAL probabilities and sum to roughly the rung count times the average
# survival — a median 1.46 across the archived Kalshi families, and 25.4 on the 50-rung gold ladder. So
# 1.2 separates the two populations with room on both sides, and a family that sums BELOW 1 (a truncated
# Gamma list) correctly reads as a partition.
#
# The distinction only matters when something must collapse, which is why it appears here and not in the
# ordering key: on a partition the informative outcomes are the highest-priced ones, and on a cumulative
# ladder they are the ones nearest the CROSSING — a 0.99 "above $3251" rung on a gold ladder trading near
# $4400 is a near-certainty carrying no forecast content at all.
LADDER_CUMULATIVE_PRICE_SUM = 1.2

# Compaction stages, in increasing order of what they cost the forecaster.
#
# 0 names every remaining outcome. 1 collapses the UNQUOTED ones (no price, nothing to say). 2
# collapses the SETTLED ones — deliberately not before that, because a Manifold threshold ladder
# settles its crossed rungs to exactly 1.0 while the market stays open (10 of 17 on that module's
# committed fixture), so those titles are the floor the series has already passed and the group names
# its LAST member for exactly that reason. 3 and up walk `LADDER_PRICE_FLOORS`, collapsing open
# outcomes cheapest-first.
_LADDER_STAGE_COLLAPSE_UNQUOTED = 1
_LADDER_STAGE_COLLAPSE_SETTLED = 2
LADDER_MAX_STAGE = _LADDER_STAGE_COLLAPSE_SETTLED + len(LADDER_PRICE_FLOORS) - 1

# The stage label for the hard bound: a family no floor can compact (200 open rungs all priced 0.90)
# keeps the highest-priced terms that fit and closes with a counted, summed remainder. Sentinel
# rather than `LADDER_MAX_STAGE + 1` so the `MARKET_CHILD_RENDER` marker's `max_stage=` field says
# "this fell off the end of the ladder" rather than reading as one more ordinary stage.
LADDER_HARD_BOUND_STAGE = 99

# Characters the hard bound reserves for the `[remaining N] ` prefix and the
# `+N more (N priced, X.XX summed)` tail it appends, so the greedy term fill cannot spend the
# whole row cap on terms alone (worst-case tail ~40 chars incl. the ` / ` separator).
_LADDER_HARD_BOUND_RESERVE = 60

# The shortest a hard-bounded ladder row can be asked to get, because it always names at least its
# highest-priced remaining outcome: `[remaining NNN] ` (<= 17) + one term at `CHILD_TITLE_MAX_CHARS`
# plus a price and a resolved flag (<= 55) + ` / +NNN more (NNN priced, NNN.NN summed)` (<= 40)
# = ~112 chars. 120 leaves ~8 chars of margin without pretending the floor is tighter than the
# arithmetic allows.
#
# It is what makes `LADDER_SECTION_MAX_CHARS` a real bound rather than a target: the section binds
# exactly while a slate has no more than `LADDER_SECTION_MAX_CHARS // LADDER_MIN_ROW_CHARS` ladder
# rows, which is 11 against a `RENDER_BUDGET` of 8 (asserted in the render-budget tests). Past that
# many families the floor wins and the section runs over, which is the honest tradeoff — the
# alternative is dropping a family's ladder row, and an unnamed outcome is the defect this design
# exists to remove.
LADDER_MIN_ROW_CHARS = 120

# The continuation glyph, in the `platform` cell: a sub-row's venue is its parent's, one line up, and
# repeating it would spend ~10 chars a row restating that. It occupies the platform column rather
# than prefixing the title so the title cell holds nothing but the outcome's own label.
CHILD_ROW_MARKER = "↳"

TABLE_COLUMNS: tuple[str, ...] = (
    "platform",
    "title",
    "prob",
    "total_vol",
    "OI",
    "signal",
    "close",
    "status",
    "relation",
    "why",
)

TITLE_MAX_CHARS = 80

# Shared trailing legend, shipped INSIDE the research section. The liquidity half is unchanged;
# the retired `relevance` column's content-overlap explanation is replaced by ONE sentence per
# new axis — the ranker's relation + why, and what RESOLVED means about a price. This text is
# fixed overhead on every rendered snapshot and the snapshot goes to the expensive forecaster
# models, so it earns its length: every label a cell can hold is named (a legend that omits one
# teaches forecasters to guess at it) and nothing else is.
#
# This legend is the ONLY place the forecaster is taught the table's notation: the prompts'
# market clause (`prompts._MARKET_READING_RULES`) used to restate the liquidity weighting, the
# tier order, RESOLVED and the `↳` glyph, and now carries policy only. So the `no-liquidity-data`
# sentence here must stay in sync with `market_retrieval.types._liquidity_label`: that label means
# "this venue publishes no volume figures" (only PredictIt now), an absence of measurement rather
# than a measurement of thinness, and a forecaster that collapses it into "thin" discounts a
# market for the wrong reason.
MARKET_SIGNAL_LEGEND = (
    "The `signal` column labels each market's liquidity/participation "
    "(thin/decent/deep for real-money venues, thin/decent/high for Manifold's play-money bettor count); "
    "`total_vol` and `OI` are that market's traded volume and open interest in approximate USD on the "
    "real-money venues, and play-money mana on Manifold. "
    "`no-liquidity-data` means the venue publishes no volume figures at all (PredictIt) — it says nothing "
    "about how liquid the market is, so treat it as unknown rather than as thin. Treat "
    "deep/high-liquidity markets as a strong anchor and discount thin ones (low volume, few participants) as noisy. "
    "Rows are ordered by EVIDENTIAL VALUE, best first, and `relation` grades how each bears on THIS question — "
    "`same_quantity_same_date`, then `same_quantity_other_cut`, then `driver_or_consequence`, then `weak` "
    "(`unspecified` if ungraded); only the first two measure the quantity asked about, and `why` is the "
    "one-phrase reason. A `why` cell opening `demoted from same-date:` means our arithmetic withdrew the ranker's "
    "top grade from that row: treat that price as background evidence, not an anchor. "
    "`status` is `open` or `RESOLVED`; a RESOLVED price is a realized outcome, not a forecast. "
    "`close` is the venue's TRADING close, not its settlement date (on Kalshi the two are months apart), so never "
    "read it as this market's resolution date. `(Nd ago)` means the close date had already passed when this forecast "
    "was made: treat that price as stale even where `status` still reads open. "
    "A `↳` row is one OUTCOME of the market above it (strike, bracket, ballot line) with its own price; the parent "
    "row has none, so anchor on the outcome that matches this question. "
    "A `↳ [remaining N]` row names every one of that market's remaining outcomes with its own price, so no outcome "
    "is hidden — a `+N unquoted` / `+N settled` / `+N under X` / `+N off certainty by under X` group is a counted "
    "set with its summed price, not a silent cut (the last one groups a threshold ladder's near-certain AND "
    "near-impossible rungs, keeping those nearest its crossing). "
    "`PRICE (bid-ask LO-HI)` shows a price and the venue's two-sided book; `LO-HI` alone means there is a book "
    "but no usable price, not 50/50. "
    "A `prob` cell prefixed `value` is a SCALAR market's estimate of the quantity itself, in the market's own units "
    "on the scale shown beside it — not a probability."
)

# Strong-evidence framing — used when at least one rendered row measures the same quantity
# (relation tier 1 or 2). "MAY be relevant" and "verify each market's resolution criteria" are
# asserted verbatim elsewhere; the old "the match below is fuzzy" clause is gone, because the
# rows are now selected by a model reading each market's rules rather than by word overlap. The
# old "a poorly-matched market may be worth little or nothing" clause is gone too — the
# per-row `relation` grade now says that per row, and saying it twice is the verbosity the
# forecaster-facing budget exists to stop.
MARKET_PREAMBLE_STRONG = (
    "The following prediction markets MAY be relevant — each was selected and ranked for THIS question, so "
    "verify each market's resolution criteria, resolution date, and topic against THIS question before "
    "weighting. A market whose criteria and date match this question is extremely strong evidence — anchor on "
    "its price; on a related but different event, date, or threshold, name the specific mismatch and discount "
    "accordingly. "
)

# Neutral framing — used when NO rendered row measures the same quantity, so the table is
# context at best. "may all be off-topic" is asserted verbatim elsewhere.
MARKET_PREAMBLE_NEUTRAL = (
    "The following prediction markets were retrieved for this question and may all be off-topic — none was "
    "judged to measure the quantity THIS question asks about, so treat them as leads to verify, not as "
    "evidence. Weight a market only after you confirm its resolution criteria, date, and topic match this "
    "question; otherwise disregard it. "
)


def render_no_relevant_market_line(pool_size: int) -> str:
    """The section body for a DELIBERATE zero-row ranking over a non-empty candidate pool.

    Without it the section vanishes wholesale on an adaptive-width-zero answer, which reads
    exactly like a provider outage — the run log records the difference (``outcome=ranked
    rows=0``) but the forecaster prompt still ships the full relation/liquidity weighting
    clauses for a table that isn't there. One sentence closes that: the empty table becomes a
    considered judgment the forecaster can lean on rather than an absence to guess about. Only
    the ranker's own empty answer earns it; every failure path still renders nothing (see
    ``prediction_market.format_snapshot_for_research``).
    """
    return (
        f"No sufficiently relevant market among {pool_size} candidates — prediction markets were "
        "retrieved and reviewed for this question, and none was judged to bear on it. This is a "
        "deliberate empty result, not a provider outage."
    )


def _cell(text: str, *, limit: int | None = None) -> str:
    """One table cell: pipes neutralised, newlines flattened, optionally truncated.

    A raw `|` from a market title or a model-authored phrase would split the row into extra
    columns and silently shift every later cell — which is why the rendering tests select
    cells BY HEADER NAME rather than by index.
    """
    cleaned = (text or "").replace("|", "/").replace("\n", " ").strip()
    return cleaned if limit is None else cleaned[:limit]


def _price_cell(
    implied_prob_yes: float | None,
    scalar_estimate: ScalarEstimate | None,
    quote_low: float | None = None,
    quote_high: float | None = None,
) -> str:
    """The ``prob`` cell: price with known book, price alone, scalar value, range, or ``-``.

    A scalar market's number is prefixed with ``value`` and followed by its scale precisely so it
    CANNOT be read as a probability. A bare figure would not be enough: the failure this fixes
    rendered ``0.48`` for a market whose value was 120.97, and the reason that got through is that
    ``0.48`` is a perfectly plausible probability. Magnitude alone does not save the general case
    either — a scalar market on a 0-to-1 scale trades values that look exactly like probabilities —
    so the label does the work rather than the reader's arithmetic.

    When the venue reports both book sides, the cell shows them beside the price as ``PRICE (bid-ask
    LO-HI)``. The range gives the forecaster the quote's uncertainty around that price: a 0.24-0.61
    book is materially different from a tight 0.52-0.54 one even when their midpoints look plausible.
    When the price is absent, the range remains by itself — an empty Kalshi book is ``0.00-1.00``,
    and a quoted but unusably wide one might be ``0.30-1.00`` — so no midpoint is implied.

    A row can never hold a probability AND a scalar value — the venue parser fills one or the other.
    A scalar value whose venue omitted book bounds renders labelled and without them: the number is
    the market's answer and the scale is context for it.
    """
    if implied_prob_yes is not None:
        price = f"{implied_prob_yes:.2f}"
        if quote_low is not None and quote_high is not None:
            return f"{price} (bid-ask {quote_low:.2f}-{quote_high:.2f})"
        return price
    if scalar_estimate is not None:
        value = format_scalar_number(scalar_estimate.value, sig_digits=SCALAR_VALUE_SIG_DIGITS)
        bounds = scalar_estimate.bounds_text()
        return f"value {value} ({scalar_estimate.scale_label} {bounds})" if bounds else f"value {value}"
    if quote_low is not None and quote_high is not None:
        return f"{quote_low:.2f}-{quote_high:.2f}"
    return "-"


def _close_cell(close_time: datetime | None, forecast_time: datetime | None) -> str:
    """The ``close`` cell: the date, plus how long ago it passed when it is in the past.

    A bare date makes a forecaster do the subtraction, and on the case this fixes there was nothing
    to prompt them to try: q45163's rank-0 row closed 2026-02-27 against an 08-09 snapshot and
    rendered ``status=open`` beside it, because Manifold's soft close dates leave ``is_resolved``
    false — so the table's only cue that its best-ranked market was five months dead was a date in a
    column full of dates. ``(162d ago)`` is the cue.

    It says the DATE has passed and not that trading stopped, which is the only thing we know: the
    same soft-close behaviour that hid this defect means a Manifold market can still be tradeable
    past its close date, so ``closed 162d ago`` would be an overclaim on exactly the venue that
    produced the case. The legend carries the reading ("treat that price as stale even where
    ``status`` still reads open") and the shorter suffix saves 7 chars a row, which matters on a
    slate that can carry 22 dated rows.

    The date STAYS. It is what the preamble's "verify each market's resolution date against THIS
    question" instruction is checked against, and a relative age cannot answer that.

    ``timedelta.days`` floors, so a close a few hours old reads as 0 and discloses nothing (a market
    that stopped trading this morning is not stale) and a future close goes negative and discloses
    nothing. Both sides normalize through ``_as_utc``: venue close times are UTC-aware by
    construction (``http.parse_iso``), and a naive ``forecast_time`` from a caller would otherwise
    raise on the subtraction and take the whole render down.
    """
    if close_time is None:
        return "-"
    rendered = close_time.strftime("%Y-%m-%d")
    if forecast_time is None:
        return rendered
    days_since_close = (_as_utc(forecast_time) - _as_utc(close_time)).days
    return f"{rendered} ({days_since_close}d ago)" if days_since_close >= 1 else rendered


def _priced_cells(
    *,
    implied_prob_yes: float | None,
    total_volume: float | None,
    open_interest: float | None,
    close_time: datetime | None,
    is_resolved: bool,
    signal: str,
    forecast_time: datetime | None,
    scalar_estimate: ScalarEstimate | None = None,
    quote_low: float | None = None,
    quote_high: float | None = None,
) -> dict[str, str]:
    """The six columns a parent row and a ``↳`` sub-row format identically.

    Shared so the two can never disagree about how a price or a date reads: a sub-row that rounded
    its probability to three places, or wrote its dates the other way round, would look like a
    different kind of number rather than the same measurement one level down.

    ``scalar_estimate`` defaults to absent because only a parent row can carry one — no venue
    publishes a scalar OUTCOME inside a multi-outcome market — but it is formatted here rather than
    at the parent's call site so there stays exactly one place that decides what the ``prob`` column
    may contain. The quote bounds ride through for the same reason: a single-strike Kalshi FAMILY and a
    strike sub-row can show their price alongside the book, or show only the book when their midpoint is refused.

    ``forecast_time`` is required rather than defaulted so no call site can silently render dates
    with no staleness read: the value may legitimately be ``None`` (an archived snapshot written
    before the field existed), and that has to be a decision the caller states.
    """
    return {
        "prob": _price_cell(implied_prob_yes, scalar_estimate, quote_low, quote_high),
        "total_vol": f"{total_volume:.0f}" if total_volume is not None else "-",
        "OI": f"{open_interest:.0f}" if open_interest is not None else "-",
        "signal": signal,
        "close": _close_cell(close_time, forecast_time),
        "status": "RESOLVED" if is_resolved else "open",
    }


def _why_cell(match: MarketMatch) -> str:
    """The ``why`` cell: the tier-cap note, then the ranker's phrase, inside ONE ``WHY_CHARS`` budget.

    The note LEADS and the ranker's phrase yields to it, because the note is our own arithmetic
    refusing the model's grade and a cell that truncated it away would show a demoted tier with no
    stated reason. What the note does NOT get is a budget of its own: it is forecaster-facing prose
    in a per-row cell, and this section's char budgets are measured on a slate where every cell sits
    at its cap, so an exempt note simply was not covered by them — three capped rows on such a slate
    crossed the maxed budget and five crossed the ``RESEARCH_SECTION_CHAR_LIMIT / 4`` ceiling. The
    note is generated by ``ranking._tier_cap_note`` at ~62 characters, so sharing the budget leaves
    the ranker's phrase ~56 of its 120 on a demoted row and costs the uncapped rows nothing.
    """
    note = _cell(match.tier_cap_note, limit=WHY_CHARS)
    if not note:
        return _cell(match.relevance_label, limit=WHY_CHARS) or "-"
    # Two characters for the ". " join below, so note + join + label lands exactly at WHY_CHARS.
    label = _cell(match.relevance_label, limit=max(WHY_CHARS - len(note) - 2, 0))
    return f"{note}. {label}" if label else note


def _row_cells(match: MarketMatch, forecast_time: datetime | None) -> dict[str, str]:
    return {
        "platform": match.platform,
        "title": _cell(match.market_title, limit=TITLE_MAX_CHARS),
        **_priced_cells(
            implied_prob_yes=match.implied_prob_yes,
            total_volume=match.total_volume,
            open_interest=match.open_interest,
            close_time=match.close_time,
            is_resolved=match.is_resolved,
            signal=_liquidity_label(match),
            forecast_time=forecast_time,
            scalar_estimate=match.scalar_estimate,
            # The parent's book, populated only where the venue publishes one.
            quote_low=match.bid,
            quote_high=match.ask,
        ),
        "relation": _cell(match.relation_tier) or TIER_UNSPECIFIED,
        "why": _why_cell(match),
    }


def _child_cells(platform: str, child: MarketChild, forecast_time: datetime | None) -> dict[str, str]:
    """One outcome's sub-row: its own price and liquidity, its parent's venue and relation.

    ``relation`` and ``why`` are the RANKER's grades and the ranker never saw this outcome — it
    graded the market. Repeating the parent's grade on every sub-row would spend ~30 chars a row
    implying a judgement that was not made, so both read ``-`` and the indentation says whose
    grade applies.
    """
    return {
        "platform": CHILD_ROW_MARKER,
        "title": _cell(child.title, limit=CHILD_TITLE_MAX_CHARS),
        **_priced_cells(
            implied_prob_yes=child.implied_prob_yes,
            total_volume=child.total_volume,
            open_interest=child.open_interest,
            close_time=child.close_time,
            is_resolved=child.is_resolved,
            signal=_child_liquidity_label(platform, child),
            forecast_time=forecast_time,
            quote_low=child.quote_low,
            quote_high=child.quote_high,
        ),
        "relation": "-",
        "why": "-",
    }


def _child_allowances(matches: Sequence[MarketMatch]) -> list[int]:
    """How many FULL sub-rows each row may render, filling ONE ROUND AT A TIME across the whole slate.

    So every multi-outcome market shows its leading outcome — the highest-priced open strike or
    bracket — before any market shows a second, and a market with fewer outcomes than the
    round count hands its unused slots back to the others. Draining the budget in rank order instead
    would spend it all on the top two rows and leave rows 6-8 rendering a bare ``-``, which is the
    exact defect this whole expansion exists to remove.

    Since 2026-08-25 this rations only the liquidity/close/status DETAIL: every outcome past a
    family's allowance is still named with its price in that family's ladder row, so an allowance of
    zero costs a family its detail cells and nothing else.

    NOT the venue round-robin the module docstring forbids, and the distinction is worth being
    precise about because the vocabulary collides. That one reordered the RANKED ROWS for venue
    fairness and cost 43 of 58 wanted rows. This changes no row's position and drops no row: the
    slate is still the ranker's order verbatim, and the only thing shared out is how deep each
    market's own outcome list is allowed to go.
    """
    allowances = [0] * len(matches)
    budget = MAX_CHILD_ROWS_PER_SNAPSHOT
    for _ in range(MAX_CHILD_ROWS_PER_MARKET):
        for index, match in enumerate(matches):
            if budget <= 0:
                return allowances
            if allowances[index] < len(match.children):
                allowances[index] += 1
                budget -= 1
    return allowances


def _table_row(cells: dict[str, str]) -> str:
    """One markdown row, columns in ``TABLE_COLUMNS`` order. A ``KeyError`` here is the point:
    every row shape — parent, sub-row, marker — must fill every column or the table skews."""
    return "| " + " | ".join(cells[column] for column in TABLE_COLUMNS) + " |"


def _open_priced(children: Sequence[MarketChild]) -> list[MarketChild]:
    """The OPEN, PRICED children — the one definition of the membership every disclosed
    price figure (a group's sum AND its stated count) is computed over, so the two cannot
    drift apart."""
    return [child for child in children if child.implied_prob_yes is not None and not child.is_resolved]


def _open_price_total(children: Sequence[MarketChild]) -> float:
    """The summed quoted prices of the OPEN children — every price figure the ladder discloses.

    Sums ``_open_priced``, so a collapse group's summed price, its stated count, and the hard
    bound's remainder all read one membership rule. Resolved children are excluded because a
    settled rung's price is a realized outcome rather than forecast content, and reporting it
    as a summed price a forecaster is not seeing would overstate what the collapse cost.
    Unquoted children contribute nothing, which is why blanking the venues' manufactured 0.50
    defaults corrects these figures for free: a fabricated price used to inflate them.

    A raw sum rather than a share of the family, deliberately. A cumulative Kalshi threshold ladder's
    nested survival prices routinely sum past 1.0 (median 1.46 across the archived Kalshi families),
    so nothing here can be labelled "mass" — and the number is now attached to a NAMED, counted group
    rather than standing in for outcomes the forecaster cannot see at all, so it reads as arithmetic
    about that group instead of a claim about the distribution.
    """
    return sum(child.implied_prob_yes for child in _open_priced(children) if child.implied_prob_yes is not None)


@dataclass(frozen=True, slots=True)
class _LadderRow:
    """One family's ``[remaining N]`` row: its rendered title and what that title accounts for.

    ``named + collapsed`` always equals the number of outcomes the row covers, which is the
    completeness invariant stated as arithmetic — every remaining outcome is either named with its own
    price or inside a group that states its count.
    """

    title: str
    stage: int
    named: int
    collapsed: int


def _is_cumulative_family(children: Sequence[MarketChild]) -> bool:
    """Whether a family's prices are a cumulative threshold ladder rather than a partition.

    One cheap read of data already in hand, and it exists only to decide what a COLLAPSE gives up —
    see ``LADDER_CUMULATIVE_PRICE_SUM`` for the arithmetic that separates the two populations.
    """
    return _open_price_total(children) > LADDER_CUMULATIVE_PRICE_SUM


def _forecast_content(child: MarketChild, *, cumulative: bool) -> float:
    """How much forecast content one outcome's price carries. Higher survives a collapse longer.

    On a PARTITION it is the price: the leaders are the mass, and a 0.02 bracket says little.

    On a CUMULATIVE ladder it is the distance from certainty, ``min(p, 1-p)``. This is the point the
    design's §2b makes and the reason a price-ranked collapse is close to worst-possible on that shape:
    a 50-rung gold ladder's top-priced rungs are ``Above $3251.99 0.99 / Above $3691.99 0.99 / ...`` —
    eight near-certainties, zero information — while ``Above $4411.99 0.46``, the crossing that answers
    the question, sits in the middle of the price range. Keeping by price names the first set and counts
    the second. Both tails of such a ladder are uninformative and the middle is the answer.

    ``-1.0`` for an unquoted outcome, so it collapses before anything with a price at every floor
    (including the ``0.0`` no-op floor, which no real price falls below).
    """
    price = child.implied_prob_yes
    if price is None:
        return -1.0
    return min(price, 1.0 - price) if cumulative else price


def _quoted_price(child: MarketChild) -> float:
    """The child's quoted price, refusing an unquoted one LOUDLY.

    Every caller reaches this behind an ``implied_prob_yes is not None`` filter, so a None
    here means that filter regressed. The point of not writing ``or 0.0`` at the call site is
    that such a regression must not turn into a rendered ``0.00`` — that is exactly the
    withheld-price-as-real-zero shape 58175a7 fixed at the venue parsers.
    """
    if child.implied_prob_yes is None:
        raise ValueError(f"unquoted outcome reached price arithmetic: {child.title!r}")
    return child.implied_prob_yes


def _ladder_content_key(child: MarketChild, *, cumulative: bool) -> tuple[bool, float]:
    """Collapse-survival order: open before settled, then most-informative first."""
    return (child.is_resolved, -_forecast_content(child, cumulative=cumulative))


def _ladder_term(child: MarketChild) -> str:
    """One outcome inside a ladder row: its label and its price, or its label and a dash.

    ``R`` marks a settled outcome — the ``status`` column a full sub-row has is not available here, and
    a realized 1.00 read as a forecast is exactly the confusion the parent table's RESOLVED label
    exists to prevent. Two characters rather than the word, because this string is repeated once per
    outcome inside a section that ships to the expensive forecaster models on every question.
    """
    title = _cell(child.title, limit=CHILD_TITLE_MAX_CHARS)
    if child.implied_prob_yes is None:
        return f"{title} -"
    return f"{title} {child.implied_prob_yes:.2f}{' R' if child.is_resolved else ''}"


def _ladder_join(total: int, parts: Sequence[str]) -> str:
    """The row's title: the remaining count, then the terms and groups, slash-separated."""
    return f"[remaining {total}] " + " / ".join(parts)


def _settled_group(settled: Sequence[MarketChild]) -> str:
    """The collapsed settled group: its count, its realized price span, and the last rung it names.

    The group names its LAST member in family order. On an ordered threshold ladder that title is
    the crossed floor — Manifold settles crossed rungs to exactly 1.0 while the market stays open —
    so collapsing them to a bare count would delete the floor the series has already passed.

    Prices come through ``_quoted_price`` rather than an ``or 0.0``: every caller filters on
    ``implied_prob_yes is not None``, so a fallback here would be dead code covering exactly the
    shape 58175a7 fixed — a withheld price re-entering the arithmetic as a real 0.00. A regression
    in that filter now raises instead of rendering a fabricated span.
    """
    prices = [_quoted_price(child) for child in settled]
    low, high = min(prices), max(prices)
    span = f"{low:.2f}" if low == high else f"{low:.2f}-{high:.2f}"
    last = _cell(settled[-1].title, limit=CHILD_TITLE_MAX_CHARS)
    return f"+{len(settled)} settled at {span}, last {last}"


def _cheap_group(cheap: Sequence[MarketChild], *, floor: float, cumulative: bool) -> str:
    """The collapsed low-content group: its count, the floor it fell under, and its summed price.

    "off by" rather than "under" on a cumulative ladder, because the group is BOTH of its tails —
    the near-certain rungs and the near-impossible ones — and calling a 0.99 rung "under 0.20"
    would read as a price claim that is simply false.
    """
    label = "off certainty by under" if cumulative else "under"
    return f"+{len(cheap)} {label} {floor:.2f} ({_open_price_total(cheap):.2f} summed)"


def _ladder_at_stage(rest: Sequence[MarketChild], *, stage: int) -> _LadderRow:
    """The ladder title for ``rest`` at one compaction stage. ``rest`` is in FAMILY order.

    Terms are emitted in family order and only MEMBERSHIP changes with the stage, so a ladder always
    reads as the venue's own ladder — on an ordered threshold family that is the threshold order, and
    a price-sorted reading of a cumulative ladder is unintelligible.

    **All bookkeeping is by POSITION, never by object identity.** A family can legitimately hold two
    equal outcomes, and the committed ``TestRenderBudget._maxed_rows`` fixture holds ten references to
    ONE ``MarketChild`` instance, so an ``id()``-keyed set silently merges them. That is not
    hypothetical: it is what made the design's costing pass report the maxed fixture at 9,642 chars
    when the real figure was 13,306.

    What each collapsed group has to say is ``_settled_group`` / ``_cheap_group``'s business; the
    settled one's last-member rule in particular is stated there.

    Which OPEN outcomes a floor takes is decided by ``_forecast_content``, which reads the family's shape:
    on a partition the cheapest go first, and on a cumulative ladder BOTH tails go before the crossing.
    That is the one place shape has to be read, and only because a collapse is the only point at which
    forecast content is actually given up.
    """
    unquoted = [index for index, child in enumerate(rest) if child.implied_prob_yes is None]
    settled = [index for index, child in enumerate(rest) if child.implied_prob_yes is not None and child.is_resolved]
    live = [index for index, child in enumerate(rest) if child.implied_prob_yes is not None and not child.is_resolved]

    collapse_unquoted = stage >= _LADDER_STAGE_COLLAPSE_UNQUOTED
    collapse_settled = stage >= _LADDER_STAGE_COLLAPSE_SETTLED
    floor_index = min(max(stage - _LADDER_STAGE_COLLAPSE_SETTLED, 0), len(LADDER_PRICE_FLOORS) - 1)
    floor = LADDER_PRICE_FLOORS[floor_index]
    cumulative = _is_cumulative_family(rest)
    cheap = [index for index in live if _forecast_content(rest[index], cumulative=cumulative) < floor]
    if live and len(cheap) == len(live):
        # The most informative live outcome is never collapsed, whatever the floor. A row that named
        # nothing would be a bare count — the silent cut this whole design removes — and the top floor
        # can otherwise take a whole family: 200 rungs at 0.90 all sit 0.10 off certainty, so a 0.50
        # floor swallows every one of them. Same guarantee `_ladder_hard_bound` makes, for the same
        # reason; `LADDER_MIN_ROW_CHARS` is the arithmetic that follows from both.
        leader = min(live, key=lambda index: _ladder_content_key(rest[index], cumulative=cumulative))
        cheap = [index for index in cheap if index != leader]

    keep = {index for index in live if index not in set(cheap)}
    if not collapse_unquoted:
        keep |= set(unquoted)
    if not collapse_settled:
        keep |= set(settled)

    parts = [_ladder_term(child) for index, child in enumerate(rest) if index in keep]
    collapsed = 0
    if collapse_unquoted and unquoted:
        parts.append(f"+{len(unquoted)} unquoted")
        collapsed += len(unquoted)
    if collapse_settled and settled:
        parts.append(_settled_group([rest[index] for index in settled]))
        collapsed += len(settled)
    if cheap:
        parts.append(_cheap_group([rest[index] for index in cheap], floor=floor, cumulative=cumulative))
        collapsed += len(cheap)

    return _LadderRow(title=_ladder_join(len(rest), parts), stage=stage, named=len(keep), collapsed=collapsed)


def _ladder_hard_bound(rest: Sequence[MarketChild], *, cap: int = LADDER_ROW_MAX_CHARS) -> _LadderRow:
    """The last resort: keep the most informative terms that fit ``cap``, count and sum the remainder.

    Reached in two ways, and both are narrow. A family no floor can compact — 200 outcomes at exactly
    0.50 all sit the maximum distance from certainty, so no stage touches them. And a family whose
    stage-compacted title still exceeds its SHARE of the section allowance, which is the tier
    ``_fit_ladder_section`` falls to.

    Terms are emitted in family order; only which ones survive is decided, by ``_forecast_content``, so a
    cumulative ladder keeps its crossing region rather than the zero-information near-certainties a
    price-ranked selection would have named.

    The most informative term is ALWAYS named, even when it alone exceeds ``cap``. A row that named
    nothing would be a bare count, which is the silent cut this design removes; ``LADDER_MIN_ROW_CHARS``
    is the arithmetic that follows from this guarantee.
    """
    cumulative = _is_cumulative_family(rest)
    by_content = sorted(
        range(len(rest)), key=lambda position: _ladder_content_key(rest[position], cumulative=cumulative)
    )

    # and the loop below then adds every further term that fits.
    kept: set[int] = set(by_content[:1])  # HARNESS-SCAN-EXEMPT-subsampling
    for index in by_content[1:]:
        trial = kept | {index}
        body = " / ".join(_ladder_term(rest[position]) for position in sorted(trial))
        if len(body) > cap - _LADDER_HARD_BOUND_RESERVE:
            break
        kept = trial

    dropped = [child for index, child in enumerate(rest) if index not in kept]
    parts = [_ladder_term(rest[index]) for index in sorted(kept)]
    # The count the sum covers, stated. `_open_price_total` sums only the OPEN, PRICED
    # members (a settled rung's price is a realized outcome, an unquoted one has none),
    # so a bare `+12 more (0.35 summed)` could hide 8 outcomes settled at 1.00 — the
    # reader has no way to tell 0.35-across-12 from 0.35-across-4. The stage path splits
    # into named per-kind groups instead; the hard bound is already at its character
    # ceiling, so it names the denominator rather than adding rows (a 200-rung worst case
    # renders "+199 more (199 priced, 145.00 summed)", 37 chars against a 60-char reserve).
    # `_open_priced` is the same filter `_open_price_total` sums, by construction.
    priced = _open_priced(dropped)
    parts.append(f"+{len(dropped)} more ({len(priced)} priced, {_open_price_total(dropped):.2f} summed)")
    return _LadderRow(
        title=_ladder_join(len(rest), parts),
        stage=LADDER_HARD_BOUND_STAGE,
        named=len(kept),
        collapsed=len(dropped),
    )


def _fit_ladder_row(rest: Sequence[MarketChild], *, min_stage: int = 0) -> _LadderRow:
    """The lowest compaction stage at or above ``min_stage`` whose title fits one row's cap."""
    for stage in range(min_stage, LADDER_MAX_STAGE + 1):
        row = _ladder_at_stage(rest, stage=stage)
        if len(row.title) <= LADDER_ROW_MAX_CHARS:
            return row
    return _ladder_hard_bound(rest)


def _fit_ladder_section(rests: Sequence[Sequence[MarketChild]]) -> list[_LadderRow]:
    """Every family's ladder row, compacted until the SECTION's summed titles fit their allowance.

    Two tiers, because the first one alone does not bind.

    **Stage escalation.** Escalate the LONGEST title one stage at a time: that buys the most
    characters per stage, is independent of the order the families arrive in, and compacts exactly the
    family whose collapsed outcomes are least likely to matter (a 3,198-char title is a 200-rung
    ladder, not a five-bracket partition). Ties break to the lowest index so the render is
    byte-identical across calls.

    **Per-family hard bound**, when every stage is spent and the section is still over. The escalation
    loop can exit over budget, because a stage only helps when there is something cheap to collapse: a
    family of outcomes at exactly 0.50 reaches ``LADDER_MAX_STAGE`` untouched, and under the
    price-ranked collapse this replaced so did a cumulative near-certainty ladder — eight of those held
    3,672 chars of titles against a 1,400 allowance and rendered 12,951 chars against a committed 10,600
    budget and an 11,250 section ceiling. The character ceilings are hard, so each over-share row falls
    to ``_ladder_hard_bound`` at its equal share of the allowance. Every outcome is still named or inside
    a counted, summed group, which is the invariant that matters.

    This tier is a deliberate addition to the design, which specified the escalation loop alone and
    separately asserted the section allowance holds. Those two cannot both be true on that shape, and
    the budget derivation depends on the allowance binding.
    """
    rows = [_fit_ladder_row(rest) for rest in rests]
    while sum(len(row.title) for row in rows) > LADDER_SECTION_MAX_CHARS:
        escalatable = [index for index, row in enumerate(rows) if row.stage < LADDER_MAX_STAGE]
        if not escalatable:
            break
        target = max(escalatable, key=lambda index: (len(rows[index].title), -index))
        rows[target] = _fit_ladder_row(rests[target], min_stage=rows[target].stage + 1)

    if not rows or sum(len(row.title) for row in rows) <= LADDER_SECTION_MAX_CHARS:
        return rows
    share = max(LADDER_SECTION_MAX_CHARS // len(rows), LADDER_MIN_ROW_CHARS)
    return [
        row if len(row.title) <= share else _ladder_hard_bound(rests[index], cap=share)
        for index, row in enumerate(rows)
    ]


def _ladder_cells(title: str) -> dict[str, str]:
    """The ladder row as a table row: every other column a dash.

    Shaped as a table row rather than a trailing note because a bare line between rows would end the
    markdown table and orphan everything after it. Its own columns are empty because the row is a list
    of prices, not one measurement — a volume or close date here would have to belong to some
    particular outcome, and it belongs to all of them.
    """
    return {**dict.fromkeys(TABLE_COLUMNS, "-"), "platform": CHILD_ROW_MARKER, "title": title}


@dataclass(frozen=True, slots=True)
class ChildRenderStats:
    """What one snapshot's ``↳`` rows accounted for — the ``MARKET_CHILD_RENDER`` marker's payload.

    ``withheld`` is why this exists. The Kalshi spread threshold is calibrated on eleven fixture
    strikes, so its prod incidence has to be a query rather than a guess; ``max_stage`` and
    ``ladder_chars`` say whether ``LADDER_SECTION_MAX_CHARS`` binds in practice. ``named +
    collapsed == outcomes`` is the completeness invariant, so a marker line where those disagree is a
    render bug rather than a tuning signal.
    """

    families: int = 0
    full_rows: int = 0
    ladder_rows: int = 0
    outcomes: int = 0
    named: int = 0
    collapsed: int = 0
    withheld: int = 0
    max_stage: int = 0
    ladder_chars: int = 0


def _bullet_body(match: MarketMatch) -> str:
    """A bullet's text: the market's rules, verbatim and flattened to one line.

    A multi-outcome market's answers used to LEAD this body as ``answers: Over $4.60 (50%), ...``,
    because they were the row's only price and the table had nowhere to put them. The ``↳`` sub-rows
    are that place now, and each answer gets a real ``prob`` cell plus its own volume, close date and
    status there, so carrying them here too would pay twice for the same numbers and crowd out the
    rules text the sub-rows cannot show. That also retires the parenthesisation guard: the reason
    ``(50%)`` was parenthesised is that the per-model MC option parser reads ``- <name>: NN%`` and a
    bare percentage after a colon could arm that shape. Nothing the formatter adds to a bullet is
    percentage-shaped any more, and a table cell cannot be mistaken for a bullet.

    One duplication is knowingly left: a PredictIt bullet reads ``contracts: A, B, C`` and its
    sub-rows name the same contracts. The names are in ``raw_rules`` because the RANKER needs them
    (a PredictIt title is often just "Which party will win ..."), so splitting them out would change
    a candidate line to save ~150 chars on the venue that contributes the fewest rendered rows.
    """
    return (match.raw_rules or "").strip().replace("\n", " ")


def _preamble(matches: Sequence[MarketMatch]) -> str:
    """Strong when any rendered row measures the same quantity, neutral otherwise.

    The selector is the ranker's own top tier, replacing the content-overlap + confidence bar
    it retired. A fail-open slate carries no tier on any row, so it selects neutral — which
    is right: an unranked slate must never get an authoritative header.
    """
    return MARKET_PREAMBLE_STRONG if any(m.relation_tier in STRONG_TIERS for m in matches) else MARKET_PREAMBLE_NEUTRAL


def _full_row_order(children: Sequence[MarketChild]) -> list[int]:
    """Child POSITIONS in full-sub-row presentation order: open first, then price-descending.

    Positions rather than children, because everything downstream has to distinguish two equal
    outcomes — and the same tuple can legitimately hold one instance twice. ``sorted`` is stable on
    the index, so equal-keyed children keep the venue's own catalogue order.
    """
    return sorted(range(len(children)), key=lambda index: child_render_order_key(children[index]))


def _withheld_prices(matches: Sequence[MarketMatch]) -> int:
    """Rows and outcomes whose price this repo REFUSED as manufactured, parents included."""
    return sum(1 for match in matches if match.price_withheld) + sum(
        1 for match in matches for child in match.children if child.price_withheld
    )


def render_snapshot_with_stats(
    snapshot: MarketSnapshot, *, ranking_degraded: bool = False
) -> tuple[str, ChildRenderStats]:
    """The markdown block plus what its ``↳`` rows accounted for. ``("", stats)`` when there is nothing.

    ``ranking_degraded`` prefixes the degraded marker, because the preamble and legend both
    tell the forecaster the rows are in evidential order — which is false for a fail-open
    slate, and a silently-wrong ordering claim is worse than a visibly degraded one.

    A multi-outcome market contributes one row, a full ``↳`` sub-row for each outcome its allowance
    covered, and — when any outcome is left over — ONE ``↳ [remaining N]`` ladder row naming every one
    of them with its price. Nothing is dropped. The rules-bullet section stays ONE bullet per market
    regardless: every outcome inside a market shares its settlement rule, so a bullet per sub-row
    would repeat the same text up to ten times.

    The stats form is the primary and ``render_snapshot`` the wrapper, so the seam can log the
    ``MARKET_CHILD_RENDER`` marker without every existing call site and test having to learn about it.
    """
    matches = snapshot.matches
    if not matches:
        return "", ChildRenderStats()

    allowances = _child_allowances(matches)
    # Two passes, because the ladder's SECTION budget is a property of the whole slate: every family's
    # title has to exist before any of them can be compacted.
    full_positions: list[list[int]] = []
    leftovers: list[list[MarketChild]] = []
    for match, allowance in zip(matches, allowances, strict=False):
        order = _full_row_order(match.children)
        keep = set(order[:allowance])
        full_positions.append(order[:allowance])
        leftovers.append([child for index, child in enumerate(match.children) if index not in keep])
    ladder_families = [index for index, rest in enumerate(leftovers) if rest]
    ladder_rows = dict(
        zip(ladder_families, _fit_ladder_section([leftovers[index] for index in ladder_families]), strict=False)
    )

    lines: list[str] = []
    if ranking_degraded:
        lines.append(DEGRADED_RANKING_MARKER)
        lines.append("")
    lines.append(_preamble(matches) + MARKET_SIGNAL_LEGEND)
    lines.append("")
    lines.append("| " + " | ".join(TABLE_COLUMNS) + " |")
    lines.append("|" + "---|" * len(TABLE_COLUMNS))
    for index, match in enumerate(matches):
        lines.append(_table_row(_row_cells(match, snapshot.forecast_time)))
        lines.extend(
            _table_row(_child_cells(match.platform, match.children[position], snapshot.forecast_time))
            for position in full_positions[index]
        )
        ladder = ladder_rows.get(index)
        if ladder is not None:
            lines.append(_table_row(_ladder_cells(ladder.title)))

    lines.append("")
    lines.append("### Resolution criteria / rules")
    for match in matches:
        body = _bullet_body(match)
        if len(body) > RAW_BULLET_BODY_MAX_CHARS:
            body = body[:RAW_BULLET_BODY_MAX_CHARS] + "..."
        # An empty body used to render a bare `- **manifold** <url>: ` line, which reads as
        # "this market publishes no resolution criteria" — a claim about the market rather
        # than about our retrieval. 6 of 146 archived rows (all Manifold, whose description
        # field is optional) rendered that way. Naming the gap keeps a forecaster from
        # discounting a market for saying nothing when it was us who carried nothing.
        if not body:
            body = "[rules unavailable — venue published no description]"
        link = f" <{match.market_url}>" if match.market_url else ""
        lines.append(f"- **{match.platform}**{link}: {body}")

    full_rows = sum(len(positions) for positions in full_positions)
    stats = ChildRenderStats(
        families=sum(1 for match in matches if match.children),
        full_rows=full_rows,
        ladder_rows=len(ladder_rows),
        outcomes=sum(len(match.children) for match in matches),
        named=full_rows + sum(row.named for row in ladder_rows.values()),
        collapsed=sum(row.collapsed for row in ladder_rows.values()),
        withheld=_withheld_prices(matches),
        max_stage=max((row.stage for row in ladder_rows.values()), default=0),
        ladder_chars=sum(len(row.title) for row in ladder_rows.values()),
    )
    return "\n".join(lines), stats


def render_snapshot(snapshot: MarketSnapshot, *, ranking_degraded: bool = False) -> str:
    """``render_snapshot_with_stats``' text half — what every caller but the telemetry seam wants."""
    return render_snapshot_with_stats(snapshot, ranking_degraded=ranking_degraded)[0]
