"""Row and snapshot types for the prediction-market provider, plus the liquidity labels.

This module imports nothing else from this repo, deliberately: it sits at the bottom of the
package's dependency graph so the venue, generation, ranking and rendering modules can all
depend on the row type without a cycle back through the seam module. Everything here is
re-exported from ``metaculus_bot.research.prediction_market``, which is where the archive
serializer, ``provider_health`` and the test suite import it from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

# Liquidity / participation signal-label thresholds. Low-volume markets are
# often bot-dominated (roughly sub-$10k), so a "thin" label is a real noise
# warning, not a formality. These cutoffs are a tunable first pass, not
# calibrated values — the "thin" ceiling sits at $5k deliberately conservatively.
#
# The unit is USD, and that is load-bearing rather than incidental: Polymarket's
# `volumeNum` / `openInterest` are already USD, while Kalshi reports CONTRACT COUNTS
# (`FixedPointCount`, "market volume in contracts" — docs.kalshi.com, 2026-08-03). One
# shared threshold pair across both venues therefore requires converting Kalshi's
# counts to dollars at the point of parse; see `_kalshi_usd_liquidity`.
# Tier floors: under DECENT is thin, then decent, strong, and deep from DEEP up. Operator-set 2026-10-06
# after Mantic's Polymarket backtest (a strong AI forecaster beat the price clearly below ~$10k of
# volume and drew level near $100k): docs/research.md "Liquidity tiers".
LIQUIDITY_DECENT_USD = 1_000.0
LIQUIDITY_STRONG_USD = 10_000.0
LIQUIDITY_DEEP_USD = 100_000.0
MANIFOLD_THIN_BETTORS = 20
MANIFOLD_HIGH_BETTORS = 100

# Significant digits for a scalar market's BOUNDS, wherever they render. Wide enough to stay exact on
# any author-chosen scale: a real Manifold scale runs as high as `max` 20,000,000,000 (11 digits), and
# rounding a bound misstates the market's own scale. The VALUE beside them is formatted separately and
# shorter (`rendering.SCALAR_VALUE_SIG_DIGITS`) because it is a noisy estimate rather than a chosen
# parameter.
SCALAR_BOUND_SIG_DIGITS = 12


def format_scalar_number(value: float, *, sig_digits: int) -> str:
    """One scalar figure for a forecaster: rounded to ``sig_digits``, grouped, never in exponent form.

    Significant digits rather than decimal places, because these scales span four orders of magnitude
    in each direction — live Manifold markets run from ``0.5 to 2.5`` up to ``1e6 to 2e10`` — and any
    fixed decimal precision serves one end while making nonsense of the other.

    Plain grouped decimal rather than ``%g``'s automatic exponent, because a cell reading
    ``value 7.84859e+09 (log scale 1000000 to 2e+10)`` writes the same scale two ways and asks a
    reader to parse exponents; ``value 7,848,590,000 (log scale 1,000,000 to 20,000,000,000)`` is the
    world population, legibly. The trailing zeros are a rounding artefact rather than a precision
    claim, which is the accepted cost — nobody reads a population ending in five zeros as exact, and
    the alternative asks every reader to do the exponent arithmetic instead.

    ``is_integer`` rather than ``value == int(value)``: it returns ``False`` for a non-finite float
    instead of raising ``OverflowError``, so a hand-built estimate carrying ``inf`` degrades to the
    string ``inf`` rather than taking down the whole rendered snapshot. Nothing the venue parser
    builds can be non-finite (``safe_float`` filters it), so this is about not being the crash site.
    """
    rounded = float(f"{value:.{sig_digits}g}")
    return f"{int(rounded):,}" if rounded.is_integer() else f"{rounded:,}"


@dataclass(frozen=True, slots=True)
class SettlementSource:
    """One publisher a market settles on: the venue's own `{name, url}` pair, verbatim.

    A dataclass rather than the raw payload dict so `dataclasses.asdict` walks it into a plain
    `{"name": ..., "url": ...}` for the raw-research archive, and so the two fields cannot
    silently pick up whatever else a venue starts shipping in that block.
    """

    name: str = ""
    url: str = ""


@dataclass(frozen=True, slots=True)
class ScalarEstimate:
    """A scalar market's traded VALUE in the question's own units — never a probability.

    Manifold's ``PSEUDO_NUMERIC`` contracts trade a number on a bounded ``[minimum, maximum]``
    scale: an age in years, a stock price, a seat count. Such a market publishes a market-level
    ``probability`` like a BINARY one does, but it means something entirely different — the value's
    normalized POSITION on that scale (0.4839 on a 0-250 age market whose value is 120.97) — so
    reading it as a price rendered a meaningless 0.48 under a title asking for an age.

    Deliberately NOT stored in ``MarketMatch.implied_prob_yes``, and this type exists to make that
    impossible rather than merely discouraged: the two numbers answer different questions, share no
    units, and only one of them belongs in a column a forecaster is told to anchor on. A row carries
    one or the other, never both.

    ``value`` is the venue's own computed figure, read verbatim and never recomputed from
    ``probability``. On a log-scale market the two disagree by 29x-6554x (measured live 2026-08-05;
    see the committed ``manifold_pseudo_numeric`` fixture), and no interpolation between the bounds
    reproduces the venue's mapping either — so the field is the only honest source. It is built only
    from ``safe_float`` output, so it is finite.

    The bounds are the market author's own choice of scale and carry real forecasting information —
    a value of 121 against a maximum of 250 is a different signal from the same value against a
    maximum of 130 — so they are rendered beside it. They are optional because they are the venue's
    to omit, and the value stands alone without them.
    """

    value: float
    minimum: float | None = None
    maximum: float | None = None
    is_log_scale: bool = False

    @property
    def scale_label(self) -> str:
        """The word that precedes the bounds. ``log`` changes how a value's POSITION between them
        reads, which is the only thing the scale type affects downstream — the value itself is in
        question units either way."""
        return "log scale" if self.is_log_scale else "scale"

    def bounds_text(self) -> str:
        """``"0 to 250"``, or ``""`` when the venue omitted either bound.

        Lives on the type because two surfaces render it — the table's ``prob`` cell and the ranker's
        candidate line — and the same market's scale must not read two ways depending on which one is
        looking. Joined with ``to`` rather than a hyphen because Manifold's scales are routinely
        negative (live: -15 to 2, -48 to 48, -4 to 4), and ``-15--2`` is unreadable.
        """
        if self.minimum is None or self.maximum is None:
            return ""
        low = format_scalar_number(self.minimum, sig_digits=SCALAR_BOUND_SIG_DIGITS)
        high = format_scalar_number(self.maximum, sig_digits=SCALAR_BOUND_SIG_DIGITS)
        return f"{low} to {high}"


@dataclass(frozen=True, slots=True)
class MarketChild:
    """One tradeable OUTCOME inside a multi-outcome parent, with its own correctly-matched price.

    A Polymarket event, a Kalshi strike family, a multi-outcome Manifold market and a PredictIt
    ballot are all one *market* to the ranker and several *prices* to a forecaster. The parent row
    carries no probability in every one of those cases — there is no single number that answers the
    parent's own title — so before this type existed the prices were either withheld or, on
    Polymarket, taken from ``markets[0]`` and rendered under the EVENT's title: the row read
    "How many Fed rate cuts in 2026? 0.89" where 0.89 was the "will no cuts happen" child.

    Deliberately leaner than ``MarketMatch``: a child is never ranked, never deduped and never
    joined on a settlement source, so it carries no id, url, tier or channel — the renderer fills
    the ``relation`` and ``why`` cells with a dash for exactly that reason. What it does carry is
    every cell that describes a PRICE: the number itself, the liquidity behind it, when it closes
    and whether it has already settled.

    ``num_bettors`` is the one field that is legitimately the PARENT's rather than the child's.
    Manifold scores participation on unique bettors and publishes no per-answer count, so its
    children inherit the market's — every answer really does share one bettor pool, and inheriting
    it renders each child the same honest label as its parent instead of a false
    ``no-liquidity-data`` on a venue that does publish per-answer volume. It is only ever the
    weaker half of the label: an answer whose OWN ``total_volume`` is present and zero reads
    ``thin`` regardless of the pool it sits in (``liquidity_label_from_fields``), because a price
    nobody has traded has no crowd behind it.

    ADDITIVE-ONLY past ``close_time``, for the same reason ``MarketMatch`` is: ``raw_log`` archives
    the whole snapshot through ``dataclasses.asdict``, under an envelope whose ``schema_version`` is
    shared across every provider, so a removal or a reorder changes the archive with no version to
    bump.

    The last three fields are the 2026-08-25 no-manufactured-price change:

    - ``quote_low`` / ``quote_high`` are the venue's own two-sided book, carried so the renderer can
      show it beside a price or by itself when the price is unusable. That distinguishes "nobody is
      quoting this rung" (``0.00-1.00``) from "quoted, very wide" (``0.30-1.00``), which
      ``implied_prob_yes is None`` alone cannot. Kalshi, Polymarket and PredictIt populate these
      fields from their available child quote legs; a range renders only when both sides are present.
      Other venues leave them empty.
    - ``price_withheld`` marks a price this repo REFUSED because the venue manufactured it — a
      Kalshi strike with no real book, a Polymarket placeholder leg at Gamma's ``["0.5","0.5"]``
      default, a Manifold answer sitting at its untouched 0.5 prior with zero volume. Separate from
      ``implied_prob_yes is None`` because the renderer and the telemetry both need to tell a refusal
      from "the venue published no price at all", and the two look identical without it.
    """

    title: str
    implied_prob_yes: float | None = None
    total_volume: float | None = None
    open_interest: float | None = None
    num_bettors: int | None = None
    is_resolved: bool = False
    close_time: datetime | None = None
    quote_low: float | None = None
    quote_high: float | None = None
    price_withheld: bool = False


@dataclass
class MarketMatch:
    """One market a forecaster might read as a peer cross-check.

    ADDITIVE-ONLY. Every field below the first twelve was added after a consumer already
    depended on the shape: `raw_log` archives the whole snapshot via `dataclasses.asdict` under
    an envelope whose `schema_version` is shared across all providers (so a removal changes the
    archive with no version to bump), `provider_health` reads the liquidity fields by `getattr`
    name, and several test sites construct the row with twelve positional arguments. Add at the
    end, always defaulted; never remove or reorder.

    `match_confidence` predates the ranked-retrieval design and no longer means "fuzzy
    relevance". It now carries the score of whichever RETRIEVAL CHANNEL found the row — 1.0 for
    a settlement-source join (a structural hit has no score to report), the inverted venue rank
    for a venue-index hit, the `fuzzy_best` score for an enumerable-universe hit — so it is not
    comparable across channels and is not a relevance signal. Relevance is `relation_tier` and
    `relevance_label`, both from the ranker.
    """

    platform: Literal["polymarket", "kalshi", "manifold", "predictit"]
    market_title: str
    market_url: str
    implied_prob_yes: float | None
    bid: float | None
    ask: float | None
    spread: float | None
    volume_24h: float | None
    close_time: datetime | None
    is_resolved: bool
    match_confidence: float
    raw_rules: str
    # Liquidity / participation fields. Previously received-but-discarded; now
    # parsed so the formatter can label how informative each crowd signal is.
    total_volume: float | None = None
    liquidity: float | None = None
    open_interest: float | None = None
    num_bettors: int | None = None
    # Ranked-retrieval fields. A venue parser fills `venue_market_id`, `sub_title` and
    # `settlement_sources`; pool assembly fills `retrieval_channel`; the ranker fills the rest.
    # The venue-native id (kalshi event_ticker, polymarket slug, manifold contract id, predictit
    # market id) is the pool's dedup key and the enrichment hook's join key — before this field
    # existed it was only recoverable by parsing `market_url`.
    venue_market_id: str = ""
    relation_tier: str = ""
    relevance_label: str = ""
    rank: int | None = None
    retrieval_channel: str = ""
    sub_title: str = ""
    settlement_sources: tuple[SettlementSource, ...] = ()
    # A multi-outcome Manifold market's leading `(answer_text, probability)` pairs, for the RANKER's
    # one-line candidate line: `implied_prob_yes` is null on every non-BINARY Manifold market, so
    # without these the candidate reaches the ranker with a title and nothing else. Empty on every
    # BINARY row and on a row whose detail GET was lost. Plain pairs rather than a nested dataclass
    # because the archive walks them into JSON arrays either way and there is no venue payload here
    # to guard against extra keys.
    #
    # Deliberately NOT what the render reads — `children` is, and it keeps the whole answer array
    # with each answer's own volume. Two reads of one payload, because a one-line prompt segment and
    # a table row want different slices; see `manifold_answer_children`.
    top_answers: tuple[tuple[str, float], ...] = ()
    # The multi-outcome expansion. A venue adapter fills this when the market has SEVERAL prices and
    # the parent title has none, and the renderer emits one indented sub-row per entry. The ranker's
    # view stays flat — a candidate line describes the parent alone and a pick costs one slot — so
    # this field is written by the venue parsers and read only by `rendering`.
    #
    # THE VENUE'S OWN CATALOGUE ORDER, verbatim: Kalshi's threshold-ordered nested array, Gamma's
    # event array, Manifold's answers array, PredictIt's ballot. Presentation belongs to `rendering`,
    # which sorts a copy by `venues/_shared.child_render_order_key` for the full sub-rows and reads THIS
    # order for the ladder row that names every remaining outcome. `rendering` is the only consumer of
    # the order (`generation` writes the Manifold enrichment and reads nothing). A row with children
    # carries `implied_prob_yes=None` on every venue — the invariant that makes "the parent has no
    # single probability" a fact about the data rather than a rendering convention.
    #
    # ⚠ THE ARCHIVED ORDER CHANGES MEANING AT 2026-08-25, and there is no schema version to bump (the
    # `raw_log` envelope's `schema_version` is shared across every provider). Records written BEFORE
    # that date preserve the RENDER order the parser imposed — open-first price-descending on the three
    # price-bearing venues, traded-size before `4e342da` — and records after it preserve the venue's
    # CATALOGUE order. Any replay that reconstructs "what the forecaster saw" from the archive must key
    # on the record's timestamp; a replay that re-sorts is comparing two different things.
    children: tuple[MarketChild, ...] = ()
    # A SCALAR market's traded value, on the only venue that has one (Manifold `PSEUDO_NUMERIC`).
    # Mutually exclusive with `implied_prob_yes` by construction in the venue parser, for the reason
    # `ScalarEstimate` documents: the same payload field carries a probability on a BINARY market and
    # a scale position on a scalar one, and a row that carried both would put two incompatible
    # numbers in one `prob` cell. Empty on every other row and every other venue.
    scalar_estimate: ScalarEstimate | None = None
    # Set when the venue DID publish a number for this row's own price and we refused it as
    # manufactured — the single-strike Kalshi family whose one strike has no real book, or a
    # single-market Polymarket event sitting at Gamma's `["0.5","0.5"]` default. `MarketChild`
    # carries the same flag for the same reason (see its docstring): `implied_prob_yes is None`
    # cannot distinguish a refusal from a venue that quotes nothing, and the `MARKET_CHILD_RENDER`
    # marker has to count refusals to say how often the Kalshi spread threshold fires in prod.
    price_withheld: bool = False
    # Why this row's `relation_tier` is one rung BELOW what the ranker asked for, written by
    # `ranking.cap_stale_top_tier` and empty on every row it did not touch. Separate from
    # `relevance_label` on purpose: that field is the ranker's own phrase, verbatim, and
    # overwriting it would make "what the model actually said" unrecoverable from the archive.
    # Separate from `relation_tier` for the mirror reason — the tier cell has to stay one of the
    # four vocabulary words, since `STRONG_TIERS` membership (which picks the preamble) and every
    # tier-conditioned residual cut test it by equality.
    tier_cap_note: str = ""


@dataclass
class MarketSnapshot:
    matches: list[MarketMatch] = field(default_factory=list)
    # Per-source outcome tokens ({source_name: token}) for the provider-diagnostics line.
    # SEVEN keys on a complete run: the four venues, plus `query_author` (the additive query
    # stage), `manifold_detail` (the rules-text enrichment fan-out) and `ranking` (the one
    # selection call). Plus `snapshot` INSTEAD of all of them on the whole-provider failure
    # paths, where the snapshot timed out or blew up before any source reported.
    #
    # A token starting with "ok"/"none" is benign; anything else (e.g. "dropped(size_cap)",
    # "error(...)", "partial(1/2)") is a LOST source. `none` means the source was reached
    # successfully and had nothing to contribute — an outage must never land there, or the
    # published line reads healthy through a blackout. Two `none`s are load-bearing and worth
    # knowing: `manifold_detail: none` means there were no Manifold candidates to enrich, and
    # `ranking: none` means the pool was empty so the ranking call never ran. See
    # provider_diagnostics.
    sources: dict[str, str] = field(default_factory=dict)
    # How many candidates the ranker was shown. ADDITIVE (archived via `asdict` alongside
    # `sources`, so removal/reorder changes the raw-archive shape with no version to bump).
    # What it exists for: `ranking: ok(0)` alone cannot say "nothing bore on the question"
    # versus "there was nothing to rank" — the formatter renders the deliberate-zero notice
    # only when a non-empty pool was reviewed, and this is the N that notice quotes. 0 on
    # every whole-provider failure path (timeout, outer-except), matching their empty sources.
    pool_size: int = 0
    # The instant the forecast is being made — `as_of` where a caller supplied one (a backtest's
    # simulated present), otherwise the instant the snapshot was fetched. ADDITIVE, same
    # `asdict`-archived reason as `pool_size`.
    #
    # It lives on the SNAPSHOT rather than being read from the clock inside the renderer because
    # the render has to be reproducible from an archived snapshot alone (the same contract that
    # keeps the degraded-ranking marker derived from `sources`): a replay months later would
    # otherwise stamp every row `(N days ago)` against ITS clock and report staleness the
    # forecaster was never shown. `as_of` first for the mirror case — with a cutoff supplied, pool
    # assembly has already dropped everything closing at or before it, so no row can be stale
    # relative to that instant and no disclosure fires, which is correct.
    #
    # None means "unknown", and the renderer then prints the bare close date with no staleness
    # claim: archived snapshots written before this field existed replay exactly as they rendered.
    forecast_time: datetime | None = None


@dataclass(frozen=True, slots=True)
class _FetchTally:
    """How many of a platform's HTTP sub-fetches came back vs. were lost.

    Carried alongside the matches because an upstream outage and a genuine no-match
    both arrive as an empty match list; without these counts the diagnostics token
    cannot tell them apart (see `_platform_source_token`). The unit is one sub-fetch
    at each platform's natural granularity: a keyword query for Polymarket/Manifold,
    a page for the Kalshi events prefetch, the single dump for PredictIt.
    """

    ok: int = 0
    failed: int = 0


def liquidity_label_from_fields(
    platform: str,
    *,
    total_volume: float | None,
    open_interest: float | None,
    num_bettors: int | None,
) -> str:
    """Label how informative a price is, given its liquidity/participation.

    Real-money venues (Polymarket, Kalshi) score on dollar volume / open interest;
    Manifold (play-money) scores on unique bettor count instead. A thin market is
    a noise warning, and the four real-money tiers grade how far a price can be leaned on.

    Takes loose fields rather than a row so a ``MarketChild`` sub-row is labelled by the SAME rule
    as its parent. Two labelling paths would let a Kalshi strike and its family disagree about what
    "thin" means — and one of them would drift, which is why Manifold's zero-own-volume rule lives
    here rather than in the child-only wrapper.
    """
    if platform == "predictit":
        # PredictIt exposes no volume/liquidity/OI fields in its all-markets dump.
        return "no-liquidity-data"

    if platform == "manifold":
        # Own volume, PRESENT and zero, overrides the bettor count. On a CHILD sub-row the bettor
        # count is the PARENT's — Manifold publishes no per-answer count — so a market with 150
        # bettors labelled every one of its untouched answers `high`, including the ones
        # `_priced_or_none` had just refused a price for as sitting at their untouched prior (62 of
        # 399 archived Manifold children, 15.5%, rendered decent/high with zero own volume). A
        # price nobody traded has no crowd behind it whatever the market's pool is. Absent volume
        # is NOT evidence of no trading, so the same `is not None` gate `_priced_or_none` uses
        # applies here, and the bettor thresholds below are untouched — a parent row carries its
        # own market volume, so a zero there means the same thing it means on a child.
        if total_volume is not None and total_volume == 0.0:
            return "thin"
        if num_bettors is None:
            return "no-liquidity-data"
        if num_bettors < MANIFOLD_THIN_BETTORS:
            return "thin"
        if num_bettors <= MANIFOLD_HIGH_BETTORS:
            return "decent"
        return "high"

    # Real-money venues: score on the larger of total volume and open interest.
    if total_volume is None and open_interest is None:
        return "no-liquidity-data"
    return _real_money_label(max(total_volume or 0.0, open_interest or 0.0))


def _real_money_label(score_usd: float) -> str:
    """The real-money tier for a market whose larger of volume and open interest is ``score_usd``."""
    tier_ceilings = ((LIQUIDITY_DECENT_USD, "thin"), (LIQUIDITY_STRONG_USD, "decent"), (LIQUIDITY_DEEP_USD, "strong"))
    return next((label for ceiling, label in tier_ceilings if score_usd < ceiling), "deep")


def _liquidity_label(m: MarketMatch) -> str:
    """``liquidity_label_from_fields`` for a whole row. The name every consumer already imports."""
    return liquidity_label_from_fields(
        m.platform,
        total_volume=m.total_volume,
        open_interest=m.open_interest,
        num_bettors=m.num_bettors,
    )


def _child_liquidity_label(platform: str, child: MarketChild) -> str:
    """``liquidity_label_from_fields`` for one sub-row, under its parent's platform rule."""
    return liquidity_label_from_fields(
        platform,
        total_volume=child.total_volume,
        open_interest=child.open_interest,
        num_bettors=child.num_bettors,
    )
