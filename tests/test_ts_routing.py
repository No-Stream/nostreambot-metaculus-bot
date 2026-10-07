"""Tests for the deterministic question -> series routing (``research/ts_routing.py``).

Routing is text-only: no network, no LLM, no fetch — every test here calls
``route_question`` on a question mock (``tests.ts_anchor_fakes``) and asserts the resolved
``_Route`` (or a skip plus its log line). The one exception is
``test_the_backstop_alone_would_not_have_caught_those``, which calls the provider's
magnitude backstop directly to pin WHY the two-leg wording guard is not redundant with it.

Coverage (one behavior per test):
- URL branch: FRED URL, Yahoo URL (single), two Yahoo tickers -> spread, ALFRED-vintage
  defaults, registry derivation/scale carried through a cited URL, quantity-gate skips,
  ambiguous URL -> None.
- Keyword branch: every registry entry that carries a gate or a near-collision, plus
  miss -> None, ambiguous keyword -> None, "highest" -> High column.
- Derivation gating: a non-"level" entry (CPI mom_pct, PAYEMS mom_diff, gasoline
  monthly_avg) fires only when the wording asks for that derived quantity.
- Gasoline split: the monthly-average and weekly-level siblings are exact complements.
- UST-10Y wording: the widened DGS10 keywords plus the route-level two-leg/change guard.
- Gate vocabulary: registry-driven — every ``require_any_keywords`` token on every gated
  entry has to route, so a token added without a routing check fails here.
- Change-guard scoping: two-leg tokens veto unconditionally, %-change tokens only when the
  route's own band isn't a percent change, and the URL branch reads the title only.
- Routing marker: one ``TS_ANCHOR_ROUTE`` line per call, one distinct step label per branch,
  each line re-parsed through the harvester's own MarkerSpec regex.
"""

from __future__ import annotations

import inspect
import logging
import re
from typing import ClassVar

import pytest

from metaculus_bot.research import ts_routing as tsr
from metaculus_bot.research.timeseries_anchor import _band_misses_bounds
from metaculus_bot.research.ts_routing import route_question
from scripts.telemetry.markers import MARKER_SPECS
from tests.ts_anchor_fakes import _DGS10_RC, _make_discrete_q, _make_numeric_q


# Routing.
class TestRouting:
    def test_routes_via_fred_url(self):
        route = route_question(_make_numeric_q(resolution_criteria=_DGS10_RC))
        assert route is not None
        assert route.kind == "single"
        assert route.spec.source == "fred"
        assert route.spec.series_id == "DGS10"
        assert route.spec.revises is False  # DGS10 is in the non-revising allowlist

    def test_routes_via_fred_url_revising_series_uses_alfred_spec(self):
        # The URL pins CPIAUCSL, but its mom_pct derivation is now gated: the question must
        # ask for the MoM quantity, else the URL route skips (see the YoY-URL test below).
        # A MoM CPI question citing the URL routes and, being a revising series, uses ALFRED.
        qt = "What will month-over-month CPI inflation be in the United States for December 2026?"
        rc = "Resolves per https://fred.stlouisfed.org/series/CPIAUCSL."
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert route.spec.series_id == "CPIAUCSL"
        assert route.derivation == "mom_pct"  # registry derivation carried through the URL route
        assert route.spec.revises is True  # CPIAUCSL revises -> ALFRED vintage fetch

    def test_url_cited_unlisted_series_defaults_to_alfred_vintage(self):
        # A revising series NOT in the (small) non-revising allowlist — INDPRO revises
        # heavily but was never enumerated. The allowlist default (revises=True) sends
        # it to ALFRED point-in-time, so a plain-fredgraph revision leak is impossible.
        rc = "Resolves per https://fred.stlouisfed.org/series/INDPRO on the resolution date."
        route = route_question(_make_numeric_q(resolution_criteria=rc))
        assert route is not None
        assert route.spec.series_id == "INDPRO"
        assert route.spec.revises is True  # unlisted -> fail-safe ALFRED vintage

    def test_url_cited_registry_series_carries_registry_derivation(self):
        # F10: a CHANGE-phrased question citing the PAYEMS FRED URL (resolves on the MoM
        # jobs-added change, scaled x1000) inherits the registry's derivation/scale/label,
        # NOT a raw unscaled level band. The mom_diff derivation is now gated on the URL
        # route too (see the level-phrased-URL test below), so the question must ask for the
        # change quantity for this to route.
        qt = "What will the change in nonfarm payroll employment be for December 2026?"
        rc = "Resolves per https://fred.stlouisfed.org/series/PAYEMS on the release date."
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert route.spec.series_id == "PAYEMS"
        assert route.derivation == "mom_diff"  # from the registry, not the 'level' default
        assert route.scale == 1000.0
        assert "payrolls" in route.label.lower()  # registry label, not the bare series_id

    def test_url_cited_registry_series_skips_when_quantity_gate_fails(self):
        # A PAYEMS FRED URL on a payroll-LEVEL question (no change language): the mom_diff
        # derivation doesn't fit the level quantity, so the URL route skips entirely rather
        # than fall back to a misleading raw-level band.
        qt = "What will total nonfarm payroll employment (in thousands of persons) be?"
        rc = "Resolves per https://fred.stlouisfed.org/series/PAYEMS on the release date."
        assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None

    def test_url_cited_registry_series_carries_unit_scale(self):
        # BOPGTB is a plain level but unit-scaled (millions -> billions). A cited URL must
        # still carry scale=0.001 from the registry.
        rc = "Resolves per https://fred.stlouisfed.org/series/BOPGTB on the release date."
        route = route_question(_make_numeric_q(resolution_criteria=rc))
        assert route is not None
        assert route.spec.series_id == "BOPGTB"
        assert route.derivation == "level"
        assert route.scale == pytest.approx(0.001)

    def test_url_cited_unregistered_series_keeps_level_defaults(self):
        # INDPRO is not in the registry -> URL routing keeps the dataclass defaults
        # (level, scale=1.0, label==series_id): the fix only carries metadata for MATCHES.
        rc = "Resolves per https://fred.stlouisfed.org/series/INDPRO on the release date."
        route = route_question(_make_numeric_q(resolution_criteria=rc))
        assert route is not None
        assert route.spec.series_id == "INDPRO"
        assert route.derivation == "level"
        assert route.scale == 1.0
        assert route.label == "INDPRO"

    def test_routes_via_yahoo_url_single(self):
        # %5E is the URL-encoded caret for ^VIX; the extractor url-decodes before matching.
        rc = "Tracks https://finance.yahoo.com/quote/%5EVIX at close."
        route = route_question(_make_numeric_q(resolution_criteria=rc))
        assert route is not None
        assert route.kind == "single"
        assert route.spec.source == "yfinance"
        assert route.spec.series_id == "^VIX"
        assert route.value_field == "Close"

    def test_two_yahoo_tickers_route_to_spread(self):
        # The relative-return family (all 47 observed two-ticker questions phrase it "X's
        # returns exceed Y's"): two Yahoo tickers + relative-return wording -> spread.
        qt = "How much will CL=F's returns exceed ^GSPC's over the window?"
        rc = "return(https://finance.yahoo.com/quote/CL=F) minus return(https://finance.yahoo.com/quote/%5EGSPC)."
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert route.kind == "spread"
        assert route.spec.series_id == "CL=F"
        assert route.value_field == "Close"
        assert route.spec_b is not None
        assert route.spec_b.series_id == "^GSPC"
        assert route.value_field_b == "Close"

    def test_two_yahoo_tickers_relative_outperform_wording_routes_to_spread(self):
        # "outperform" is part of the relative-return keyword set -> spread.
        qt = "Will NVDA outperform AAPL over the window?"
        rc = "Compares https://finance.yahoo.com/quote/NVDA and https://finance.yahoo.com/quote/AAPL."
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert route.kind == "spread"
        assert route.spec.series_id == "NVDA"
        assert route.spec_b is not None
        assert route.spec_b.series_id == "AAPL"

    def test_two_yahoo_tickers_ratio_wording_skips(self, caplog):
        # A gold/silver RATIO question (~85x) cites two Yahoo tickers but asks for a ratio,
        # not a relative return -> the mean-zero pp band would be wrong-unit, so skip + log.
        qt = "What will the gold-to-silver price ratio be on the resolution date?"
        rc = "Ratio of https://finance.yahoo.com/quote/GC=F to https://finance.yahoo.com/quote/SI=F."
        with caplog.at_level(logging.INFO):
            assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None
        assert any("no relative-return wording" in r.message for r in caplog.records)

    def test_two_yahoo_tickers_price_difference_wording_skips(self, caplog):
        # A price-DIFFERENCE question (dollars) cites two Yahoo tickers -> skip (wrong unit).
        qt = "What will the price of gold minus the price of silver be, in dollars?"
        rc = "https://finance.yahoo.com/quote/GC=F minus https://finance.yahoo.com/quote/SI=F."
        with caplog.at_level(logging.INFO):
            assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None
        assert any("no relative-return wording" in r.message for r in caplog.records)

    def test_two_yahoo_tickers_single_level_wording_skips(self):
        # A single ticker's LEVEL, with a second ticker cited only as context -> the level
        # (dollars) is wrong-unit for the spread band, and there's no relret wording -> skip.
        qt = "What will the closing price of NVDA be, with AAPL cited for comparison?"
        rc = "See https://finance.yahoo.com/quote/NVDA and https://finance.yahoo.com/quote/AAPL."
        assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None

    def test_two_yahoo_tickers_return_to_level_wording_skips(self, caplog):
        # "return TO <level>" is a LEVEL question (dollars), not a relative return: the bare
        # "return" substring used to route this to the mean-zero pp spread band (wrong-unit).
        # The word-boundary gate's "(?!\s+to\b)" lookahead excludes it -> skip + log.
        qt = "Will the price of NVDA return to $400 before AAPL does?"
        rc = "Compares https://finance.yahoo.com/quote/NVDA and https://finance.yahoo.com/quote/AAPL."
        with caplog.at_level(logging.INFO):
            assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None
        assert any("no relative-return wording" in r.message for r in caplog.records)

    def test_two_yahoo_tickers_singular_return_phrasing_routes_to_spread(self):
        # Guards against a plural-only fix: the SINGULAR "return(URL) minus return(URL)"
        # criteria wording (no plural "returns" anywhere in the text) must still route to
        # spread -- "return(" is followed by "(", not " to", so the lookahead lets it match.
        qt = "Where will the CL=F versus ^GSPC figure land over the window?"
        rc = "return(https://finance.yahoo.com/quote/CL=F) minus return(https://finance.yahoo.com/quote/%5EGSPC)."
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert route.kind == "spread"
        assert route.spec.series_id == "CL=F"
        assert route.spec_b is not None
        assert route.spec_b.series_id == "^GSPC"

    def test_routes_via_template_keyword(self):
        route = route_question(_make_numeric_q(question_text="Where will the 10-year treasury yield close?"))
        assert route is not None
        assert route.spec.series_id == "DGS10"
        assert "10-Year Treasury" in route.label

    def test_miss_returns_none(self):
        assert route_question(_make_numeric_q(question_text="Who wins the 2028 election?")) is None

    def test_ambiguous_url_returns_none(self, caplog):
        # Two DIFFERENT fred series cited -> not a 2-ticker spread, not a single -> ambiguous.
        rc = "https://fred.stlouisfed.org/series/DGS10 and https://fred.stlouisfed.org/series/UNRATE"
        with caplog.at_level(logging.INFO):
            assert route_question(_make_numeric_q(resolution_criteria=rc)) is None
        assert any("ambiguous URL routing" in r.message for r in caplog.records)

    def test_ambiguous_keyword_returns_none(self, caplog):
        # Two ungated level entries both match (10-year treasury + high yield spread) ->
        # ambiguous -> None. (The CPI entry, which used to serve this test, is now gated on
        # MoM language, so it no longer matches a bare "cpi versus ..." string.)
        q = _make_numeric_q(question_text="the 10-year treasury yield versus the high yield spread")
        with caplog.at_level(logging.INFO):
            assert route_question(q) is None
        assert any("ambiguous keyword routing" in r.message for r in caplog.records)

    def test_highest_framing_selects_high_column(self):
        q = _make_numeric_q(question_text="What is the highest VIX value this year?")
        route = route_question(q)
        assert route is not None
        assert route.spec.column == "High"
        assert route.value_field == "High"
        assert route.is_max is True

    def test_sp500_keyword_routes_to_gspc_level(self):
        route = route_question(_make_numeric_q(question_text="Where will the S&P 500 close on Dec 31?"))
        assert route is not None
        assert route.spec.source == "yfinance"
        assert route.spec.series_id == "^GSPC"
        assert route.derivation == "level"

    def test_bitcoin_highest_routes_to_btc_high_column_max(self):
        route = route_question(_make_numeric_q(question_text="What is the highest price of Bitcoin in 2025?"))
        assert route is not None
        assert route.spec.series_id == "BTC-USD"
        assert route.spec.column == "High"  # max/highest framing -> daily High
        assert route.value_field == "High"
        assert route.label == "Bitcoin price ($)"
        assert route.is_max is True

    def test_silver_highest_routes_to_high_column(self):
        route = route_question(_make_numeric_q(question_text="Highest silver price per troy oz in April?"))
        assert route is not None
        assert route.spec.series_id == "SI=F"
        assert route.spec.column == "High"

    def test_gold_level_routes_to_close_column(self):
        route = route_question(_make_numeric_q(question_text="What will the price of gold be on the date?"))
        assert route is not None
        assert route.spec.series_id == "GC=F"
        assert route.spec.column == "Close"  # no max framing -> Close
        assert route.value_field == "Close"
        assert route.label == "Gold front-month futures ($/oz)"

    def test_case_shiller_routes_and_revises(self):
        route = route_question(
            _make_numeric_q(question_text="What will the Case-Shiller national home price index be?")
        )
        assert route is not None
        assert route.spec.series_id == "CSUSHPISA"
        assert route.spec.revises is True  # not in the non-revising allowlist -> ALFRED vintage

    def test_fed_funds_upper_routes_and_non_revising(self):
        route = route_question(
            _make_numeric_q(question_text="What will the federal funds target range upper limit be?")
        )
        assert route is not None
        assert route.spec.series_id == "DFEDTARU"
        assert route.spec.revises is True  # DFEDTARU is NOT in the non-revising allowlist

    def test_average_weekly_hours_routes_and_revises(self):
        route = route_question(_make_numeric_q(question_text="What will average weekly hours (all employees) be?"))
        assert route is not None
        assert route.spec.series_id == "AWHAETP"
        assert route.spec.revises is True

    def test_australia_unemployment_routes_to_aus_series(self):
        route = route_question(_make_numeric_q(question_text="What will the Australian unemployment rate be?"))
        assert route is not None
        assert route.spec.series_id == "LRHUTTTTAUM156S"

    def test_us_unemployment_excludes_australia_entry(self):
        # "unemployment rate" alone must route to US UNRATE, not double-match the AUS entry.
        route = route_question(_make_numeric_q(question_text="What will the US unemployment rate be?"))
        assert route is not None
        assert route.spec.series_id == "UNRATE"

    def test_payrolls_routes_to_mom_diff_with_scale(self):
        route = route_question(_make_numeric_q(question_text="How many nonfarm payrolls jobs were added?"))
        assert route is not None
        assert route.spec.series_id == "PAYEMS"
        assert route.derivation == "mom_diff"
        assert route.scale == 1000.0
        assert route.model_target is True  # no longer a level-with-caveat skip

    def test_cpi_routes_to_mom_pct(self):
        route = route_question(_make_numeric_q(question_text="What will month-over-month CPI inflation be?"))
        assert route is not None
        assert route.spec.series_id == "CPIAUCSL"
        assert route.derivation == "mom_pct"

    def test_gasoline_routes_to_monthly_avg(self):
        route = route_question(
            _make_numeric_q(question_text="What will the monthly average regular gasoline price be?")
        )
        assert route is not None
        assert route.spec.series_id == "GASREGW"
        assert route.derivation == "monthly_avg"

    def test_goods_trade_balance_routes_with_unit_scale(self):
        route = route_question(_make_numeric_q(question_text="What will the US advance goods trade balance be?"))
        assert route is not None
        assert route.spec.series_id == "BOPGTB"
        assert route.derivation == "level"
        assert route.scale == pytest.approx(0.001)  # millions of USD -> billions


# Derivation gate: a NON-"level" registry entry (CPI mom_pct, PAYEMS mom_diff, gasoline
# monthly_avg) must only fire when the question actually asks for that derived quantity.
# The overreach it fixes: an index-LEVEL, year-over-year, or foreign-country CPI question
# (real tournament questions — UK q41640, Egypt q41634) used to route to US CPIAUCSL
# mom_pct and get an empirical band in the wrong units / wrong country. Phrasings below
# mirror the real tournament questions (scratch/ts_anchor_gate_2026-07-16/ts_labeled.json).
class TestDerivationGating:
    def test_cpi_intended_mom_us_question_routes(self):
        # qid 39567/41681 — the intended recurring US MoM headline-CPI family.
        qt = (
            "What will the seasonally adjusted month over month headline CPI inflation be in the "
            "United States in the following months? (Sep-25)"
        )
        route = route_question(_make_numeric_q(question_text=qt))
        assert route is not None
        assert route.spec.series_id == "CPIAUCSL"
        assert route.derivation == "mom_pct"

    def test_cpi_index_level_question_skips(self):
        qt = "What will the CPI-U index level (1982-84=100) be for December 2026?"
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_cpi_yoy_question_skips(self):
        qt = "What will the year-over-year CPI inflation rate be in the United States in December 2026?"
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_uk_cpi_12_month_yoy_question_skips(self):
        # qid 41640 — routes to US CPIAUCSL mom_pct today; wrong units AND wrong country.
        qt = (
            "What will be the UK Consumer Prices Index (CPI) 12-month rate (year-over-year percent change) "
            "for March 2026, as reported by the UK Office for National Statistics (ONS)?"
        )
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_egypt_urban_cpi_question_skips(self):
        # qid 41634 — foreign CPI; must not inherit the US MoM-% band.
        qt = (
            "What is Egypt's annual inflation rate (year-over-year percent change) for the Urban "
            "Consumer Price Index (Urban CPI) for March 2026?"
        )
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_cpi_url_yoy_question_skips(self):
        # A YoY question citing the CPIAUCSL FRED URL: the URL route is gated too, so it
        # skips rather than hand back a US-MoM-% band for a year-over-year quantity.
        qt = "What will the year-over-year CPI inflation rate be for December 2026?"
        rc = "Resolves per https://fred.stlouisfed.org/series/CPIAUCSL on the release date."
        assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None

    def test_payrolls_change_intended_phrasing_routes(self):
        # qid 40100/38829 — the intended recurring "change in nonfarm payroll employment" family.
        qt = "What will be the change in seasonally adjusted nonfarm payroll employment in the following months? (Dec-25)"
        route = route_question(_make_numeric_q(question_text=qt))
        assert route is not None
        assert route.spec.series_id == "PAYEMS"
        assert route.derivation == "mom_diff"
        assert route.scale == 1000.0

    def test_payrolls_level_question_skips(self):
        qt = "What will total nonfarm payroll employment (in thousands of persons) be in December 2026?"
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_gasoline_point_in_time_routes_to_weekly_level_not_monthly_avg(self):
        # GASREGW is WEEKLY, so a point-in-time gasoline question resolves on the weekly
        # LEVEL. Before the level sibling existed the monthly gate correctly refused this
        # question and there was nowhere else for it to land, so the whole point-in-time
        # gasoline family went dark (qid 45082, missed in prod). It must now route to level
        # — and never inherit monthly_avg, which is the 548ba88 rule.
        qt = "What will the price of regular gasoline be on December 31, 2026?"
        route = route_question(_make_numeric_q(question_text=qt))
        assert route is not None
        assert route.spec.series_id == "GASREGW"
        assert route.derivation == "level"

    def test_gasoline_for_the_month_intended_routes(self):
        # qid 41795/41791/41785 — the intended "national average price ... for the month of X" family.
        qt = (
            "What will be the US national average price for regular gasoline (dollars per gallon) "
            "for the month of April 2026?"
        )
        route = route_question(_make_numeric_q(question_text=qt))
        assert route is not None
        assert route.spec.series_id == "GASREGW"
        assert route.derivation == "monthly_avg"

    def test_gasoline_national_average_spot_routes_to_level_not_monthly_avg(self):
        # The GEOGRAPHIC "national average" descriptor is in every gasoline question; on a
        # point-in-time question it must NOT select monthly_avg (the bug F4 fixed — the old
        # bare-"average" gate matched here, and the ~$3/gal band can't be caught by bounds).
        # It now lands on the weekly level instead of nowhere.
        qt = "What will the national average price of regular gasoline be on July 31, 2026?"
        route = route_question(_make_numeric_q(question_text=qt))
        assert route is not None
        assert route.spec.series_id == "GASREGW"
        assert route.derivation == "level"


# The two gasoline entries are exact complements on the same four month-scoped tokens: the
# monthly_avg entry REQUIRES one, the level entry EXCLUDES all four. That is what makes the
# split safe rather than lucky — a point-in-time question cannot inherit monthly_avg (the
# 548ba88 rule) and a month-scoped question cannot inherit level, so neither derivation can
# leak into the other's family. It also means they can never co-match, which matters because
# two matches make route_question skip as ambiguous — strictly worse than the old behavior.
class TestGasolineDerivationSplit:
    _MONTH_SCOPED_TOKENS = (
        "for the month",
        "monthly average",
        "during the month",
        "monthly",
        "in the month",
        "calendar month",
    )

    @pytest.mark.parametrize(
        ("text", "expected_derivation"),
        [
            # Month-scoped phrasings -> monthly_avg, one per gate token (the tuple below is
            # asserted against the registry, so this list has to grow with it).
            ("average price for regular gasoline for the month of April 2026", "monthly_avg"),
            ("the monthly average price of regular gasoline in April 2026", "monthly_avg"),
            ("the price of regular gasoline during the month of April 2026", "monthly_avg"),
            ("the monthly regular gasoline price for April 2026", "monthly_avg"),
            ("the price of regular gasoline in the month of April 2026", "monthly_avg"),
            ("the regular gasoline price over the calendar month of April 2026", "monthly_avg"),
            # Point-in-time phrasings -> level.
            ("the price of regular gasoline on August 17, 2026", "level"),
            ("the national average price of regular gasoline on December 31, 2026", "level"),
            ("regular gasoline price as of the resolution date", "level"),
        ],
    )
    def test_each_phrasing_selects_exactly_one_derivation(self, text: str, expected_derivation: str):
        route = route_question(_make_numeric_q(question_text=text))
        assert route is not None, f"{text!r} routed nowhere"
        assert route.spec.series_id == "GASREGW"
        assert route.derivation == expected_derivation

    @pytest.mark.parametrize(
        "text",
        [
            "average price for regular gasoline for the month of April 2026",
            "the price of regular gasoline on August 17, 2026",
        ],
    )
    def test_exactly_one_gasoline_entry_matches(self, text: str):
        # Guards the ambiguity trap directly: if both entries ever matched, route_question
        # would log "ambiguous keyword routing" and skip, silently un-routing the family
        # this change exists to route.
        gasoline_hits = [
            e for e in tsr._TEMPLATE_REGISTRY if e.series_id == "GASREGW" and tsr._entry_matches(e, text.lower())
        ]
        assert len(gasoline_hits) == 1, [e.derivation for e in gasoline_hits]

    def test_real_prod_question_text_routes_to_level(self):
        # Verbatim qid 45082, the question that went dark in prod. Its title says only "gas
        # price" (which matches no keyword); the KEYWORD hit comes from the resolution
        # criteria's "Regular Gasoline", which is why a title-only check mis-diagnosed this
        # as a missing-keyword bug. route_question reads title + criteria + fine_print.
        qt = "What will be the average gas price in the U.S. on August 17, 2026?"
        rc = (
            "This question resolves as the average price of U.S. Regular Gasoline in dollars per gallon "
            "for August 17, 2026 according to the U.S. Energy Information Administration's Gasoline and "
            "Diesel Fuel Update for or closest to that date."
        )
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert route.spec.series_id == "GASREGW"
        assert route.derivation == "level"

    def test_level_entry_gate_tokens_mirror_the_monthly_entry(self):
        # The safety argument is "exact complements on the same tokens". If someone edits one
        # entry's tokens without the other, the complement breaks and a phrasing can fall
        # through both gates again (exactly how the point-in-time family went dark). Pin it.
        monthly = next(e for e in tsr._TEMPLATE_REGISTRY if e.series_id == "GASREGW" and e.derivation == "monthly_avg")
        level = next(e for e in tsr._TEMPLATE_REGISTRY if e.series_id == "GASREGW" and e.derivation == "level")
        assert monthly.require_any_keywords == self._MONTH_SCOPED_TOKENS
        assert level.exclude_keywords == self._MONTH_SCOPED_TOKENS
        # Same keyword surface, so neither family can be reachable only through the other.
        assert monthly.keywords == level.keywords

    def test_discrete_question_routes_identically_to_its_numeric_twin(self):
        # DiscreteQuestion subclasses NumericQuestion and routing is text-only, so a discrete
        # question with registry-matching text must route the same. Pinned because "all 10
        # discrete questions routed to nothing" was read as a discrete-specific gate; it was
        # question composition (poll aggregators and bespoke trackers), not a code path.
        qt = "What will the price of regular gasoline be on December 31, 2026?"
        numeric = route_question(_make_numeric_q(question_text=qt))
        discrete = route_question(_make_discrete_q(question_text=qt))
        assert numeric is not None
        assert discrete is not None
        assert (discrete.spec.series_id, discrete.derivation) == (numeric.spec.series_id, numeric.derivation)


# UST-10Y wording. The recurring "ending value of the UST 10Y Yield for these biweekly
# periods" family (qids 43931, 43650, 42143, 40788, 40216, 40089, 39915, 39590, 39509) names
# the series in a form none of the DGS10 keywords covered. But all 9 DO cite the DGS10 FRED
# URL in their resolution criteria and therefore already routed pre-fix through the URL
# branch, verified on the real pulled text — so these tokens are WORDING ROBUSTNESS for the
# case where a future sibling drops the link, and they recover ZERO currently-observed
# questions. (The earlier claim that the family carried no routable URL came from a
# title-only probe; the archive stores no resolution_criteria at all, so it could not have
# been the evidence either way.)
class TestUstTenYearWording:
    @pytest.mark.parametrize(
        "title",
        [
            "What will be the ending value of the UST 10Y Yield for these biweekly periods of Q2 2026? (Jun 15 - Jun 26)",
            "What will be the ending value of the UST 10Y Yield for the following biweekly periods? (Sep 29 - Oct 10)",
        ],
    )
    def test_ust_10y_wording_routes_to_dgs10_level(self, title: str):
        route = route_question(_make_numeric_q(question_text=title))
        assert route is not None
        assert route.spec.series_id == "DGS10"
        assert route.derivation == "level"

    def test_nob_spread_question_still_skips(self):
        # The load-bearing half. q44868 ("the NOB spread") cites TWO FRED URLs (DGS30 and
        # DGS10) and resolves on their DIFFERENCE in basis points. Widening DGS10's keywords
        # must not turn it into a single-leg 10-year LEVEL anchor — that would publish a
        # ~4.7-percent band on a question resolving around 53 basis points. The URL branch
        # runs before the registry and rejects it as ambiguous, which is what keeps the
        # widening safe; this test fails if anyone reorders those branches.
        qt = "What will be the NOB spread on August 14, 2026?"
        rc = (
            "This question resolves as the difference, in basis points, between the yield on 30-Year "
            "U.S. Treasury Securities on August 14, 2026 as presented by FRED at "
            "https://fred.stlouisfed.org/series/DGS30 and the yield on 10-Year U.S. Treasury Securities "
            "on that date as presented by FRED at https://fred.stlouisfed.org/series/DGS10."
        )
        assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None

    @pytest.mark.parametrize(
        "question_text",
        [
            "What will be the UST 10Y Yield minus the UST 2Y Yield on August 14, 2026?",
            "By how many basis points will the UST 10Y Yield change between July 1 and August 14, 2026?",
            "What will the UST 10-Year spread over Bunds be in August 2026?",
            # The same shapes on the PRE-EXISTING keywords: the hazard was never specific to
            # the two UST tokens, so the guard sits on the route, not on those keywords.
            "What will be the 10-year treasury yield minus the 2-year treasury yield on August 14, 2026?",
            "By how many bps will the 10-year yield move by August 14, 2026?",
        ],
    )
    def test_two_leg_or_change_wording_with_no_url_does_not_route(self, question_text: str):
        # The hazard the NOB test above does NOT cover: the same quantity mismatch with no
        # extractable FRED URL, so the URL branch's ambiguity check never runs and the keyword
        # route is the only gate. The magnitude backstop cannot catch these either (next test).
        assert route_question(_make_numeric_q(question_text=question_text)) is None

    def test_the_backstop_alone_would_not_have_caught_those(self):
        # Pins WHY the wording guard is required rather than redundant with the backstop.
        # If a future change makes the backstop catch this shape, this test fails and the
        # guard can be reconsidered on evidence instead of taste. (The backstop itself lives
        # in timeseries_anchor and is covered there; this reaches across the seam on purpose.)
        level_band_in_percent = (4.40, 4.68, 4.95)
        bps_change_question = _make_numeric_q(
            question_text="basis-point change",
            lower_bound=-50.0,
            upper_bound=50.0,
            open_lower_bound=True,
            open_upper_bound=True,
        )
        assert _band_misses_bounds(bps_change_question, level_band_in_percent) is False

    def test_the_guard_does_not_disarm_ambiguity_detection(self, caplog):
        # The reason this is a ROUTE-level guard and not per-entry `exclude_keywords`. An
        # exclude removes the entry from the match list the ambiguity check counts, so
        # excluding DGS10 on this two-leg question left HY-OAS as the SOLE match and it routed
        # to one leg — turning a correct ambiguous-skip into a wrong single-series anchor. The
        # guard runs after that check, so it can only ever turn a route into a skip.
        q = _make_numeric_q(question_text="the 10-year treasury yield versus the high yield spread")
        with caplog.at_level(logging.INFO):
            assert route_question(q) is None
        assert any("ambiguous keyword routing" in r.message for r in caplog.records)

    def test_a_natively_published_spread_series_still_routes(self):
        # The guard must not refuse an entry whose OWN series is the spread. HY OAS is
        # published as a spread, so "spread" in the wording describes exactly what it serves.
        route = route_question(
            _make_numeric_q(question_text="What will the ICE BofA high yield spread be on August 14, 2026?")
        )
        assert route is not None
        assert route.spec.series_id == "BAMLH0A0HYM2"

    def test_plain_level_wording_still_routes(self):
        # The guard must not cost the family the keywords were widened for.
        route = route_question(
            _make_numeric_q(question_text="What will be the ending value of the UST 10Y Yield on August 14, 2026?")
        )
        assert route is not None
        assert (route.spec.series_id, route.derivation) == ("DGS10", "level")


# Two live prod misses from a residual round, pinned on the questions' VERBATIM text (title +
# resolution criteria + fine print — route_question reads all three, and both defects were
# invisible to title-only probes). q45401 is the derivation-gate vocabulary defect; q45362 is
# the URL branch's missing change-vs-level guard.
class TestLiveProdRoutingMisses:
    _Q45401_TITLE = "How many jobs will the U.S. economy add in August 2026?"
    _Q45401_RC = (
        "This question resolves as the increase in number of employees on nonfarm payrolls, "
        "seasonally adjusted, reported by the U.S. Bureau of Labor Statistics at its"
        "[ Employment Situation Summary](https://www.bls.gov/news.release/empsit.nr0.htm) for August 2026."
    )
    _Q45401_FP = (
        "Specifically, resolution is based on the numbers reported at [Table B-1. Employees on nonfarm "
        "payrolls by industry sector and selected industry detail](https://www.bls.gov/news.release/empsit.t17.htm) "
        "for *Total nonfarm* employees, *Seasonally adjusted*, by subtracting the July 2026 number from the "
        "August 2026 number.\n\nFor example, the number reported in the June 2026 release was 158,984,000 for "
        "June 2026 and 158,927,000 for May 2026. Hence the increase was 57,000."
    )
    _Q45362_TITLE = (
        "What will be the percentage change in the S&P 500 from close of Aug 26, 2026 to open of Aug 31, 2026?"
    )
    _Q45362_RC = (
        "This question will resolve as the percentage change from the non-adjusted closing price of the "
        "S\\&P 500 Index (^GSPC) according to [Yahoo! Finance](https://uk.finance.yahoo.com/quote/%5EGSPC/history/) "
        "on August 26, 2026, to the open price of the S\\&P 500 on August 31, 2026. These prices are the last "
        "before the symposium and the first afterwards.\n\nFor example, if the dates were June 3 (\\$7,553.68 "
        "close) and June 8 (\\$7,440.57 open) the resolution would be -1.49742%."
    )

    def test_q45401_jobs_added_phrasing_routes_to_payems_mom_diff(self):
        # The live confirmed miss: "the economy will ADD jobs" / "the INCREASE in employees" is
        # the mom_diff quantity in vocabulary the gate didn't know ("change"/"added"/"gain").
        # route_question rejected it at the derivation gate for the whole triple era.
        q = _make_discrete_q(
            question_text=self._Q45401_TITLE, resolution_criteria=self._Q45401_RC, fine_print=self._Q45401_FP
        )
        route = route_question(q)
        assert route is not None
        assert route.spec.series_id == "PAYEMS"
        assert route.derivation == "mom_diff"
        assert route.scale == 1000.0

    def test_q45362_pct_change_question_citing_a_level_url_skips(self, caplog):
        # The wrong route: a %-change question citing the ^GSPC Yahoo URL was handed a LEVEL
        # band, and only the magnitude backstop (a numeric heuristic — a change question whose
        # bounds overlapped the level band would sail through) stopped the section. The wording
        # guard now covers the single-URL route.
        q = _make_numeric_q(question_text=self._Q45362_TITLE, resolution_criteria=self._Q45362_RC)
        with caplog.at_level(logging.INFO):
            assert route_question(q) is None
        assert any("skipping rather than anchoring a single-series band" in r.message for r in caplog.records)

    def test_a_payroll_level_question_still_skips_despite_the_widened_vocabulary(self):
        # " add " is space-delimited so the widening cannot arm on "address"/"additional",
        # and a LEVEL question mentioning an address must not inherit the mom_diff band.
        qt = (
            "What will total nonfarm payroll employment (in thousands of persons) be in December 2026? "
            "Resolution per the BLS release; send correspondence to the Bureau's Washington address."
        )
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_a_plain_gspc_level_question_still_routes_through_its_url(self):
        # The guard must only fire on change/difference wording — the ordinary level family
        # citing the same URL keeps its anchor.
        rc = "Resolves per the closing value at https://finance.yahoo.com/quote/%5EGSPC on the date."
        route = route_question(_make_numeric_q(question_text="Where will the S&P 500 close?", resolution_criteria=rc))
        assert route is not None
        assert (route.spec.series_id, route.derivation) == ("^GSPC", "level")

    def test_a_spread_native_series_url_is_exempt_from_the_url_guard(self):
        # Same exemption as the keyword branch: HY OAS is published AS a spread, so "spread"
        # in the wording describes exactly what the cited series serves.
        qt = "What will the ICE BofA US high yield spread be on August 14, 2026?"
        rc = "Resolves per https://fred.stlouisfed.org/series/BAMLH0A0HYM2 on that date."
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert route.spec.series_id == "BAMLH0A0HYM2"

    def test_pct_change_wording_with_no_url_skips_on_the_keyword_branch_too(self):
        # The widened guard vocabulary applies to both branches: the same S&P %-change shape
        # phrased without a URL reaches the keyword registry and must skip there.
        qt = "What will be the percentage change in the S&P 500 over the last week of August 2026?"
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_payems_change_family_is_untouched_by_the_widened_guard(self):
        # Bare "change" is deliberately NOT a guard token — it is the vocabulary the mom_diff
        # family legitimately routes on. Only percentage-qualified change wording skips.
        qt = "What will be the change in seasonally adjusted nonfarm payroll employment in the following months? (Dec-25)"
        route = route_question(_make_numeric_q(question_text=qt))
        assert route is not None
        assert (route.spec.series_id, route.derivation) == ("PAYEMS", "mom_diff")


# The change-vs-level guard's scoping: the %-change tokens veto only routes whose band is NOT
# itself a percent change, and on the URL branch the guard reads the TITLE only. Without the
# derivation scoping, the guard and the widened CPI MoM vocabulary shipped in the same commit
# cancel each other out — a MoM CPI question worded the way BLS words the series ("percent
# change") passes the widened quantity gate and is then killed by the widened guard.
class TestChangeGuardScoping:
    def test_cpi_mom_percent_change_wording_routes_on_the_keyword_branch(self):
        # BLS's own name for the quantity is "seasonally adjusted 1-month percent change in
        # CPI-U" — the mom_pct band IS the quantity this wording asks for, so the guard must
        # not veto it.
        qt = (
            "What will be the seasonally adjusted month-over-month percent change in US headline CPI for December 2026?"
        )
        route = route_question(_make_numeric_q(question_text=qt))
        assert route is not None
        assert (route.spec.series_id, route.derivation) == ("CPIAUCSL", "mom_pct")

    def test_cpi_mom_percentage_change_wording_routes_through_its_url(self):
        qt = "What will be the month-over-month percentage change in US headline CPI for December 2026?"
        rc = "Resolves per https://fred.stlouisfed.org/series/CPIAUCSL on the release date."
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert (route.spec.series_id, route.derivation) == ("CPIAUCSL", "mom_pct")

    def test_a_percentage_change_payroll_question_still_skips(self):
        # mom_diff is NOT exempt: its band is a count of persons, so %-change wording is a
        # different quantity and the guard must keep firing.
        qt = "What will be the percentage change in seasonally adjusted nonfarm payroll employment in December 2026?"
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_an_incidental_comparison_in_the_criteria_does_not_kill_a_url_anchor(self):
        # The URL-branch guard reads the TITLE only: this branch had no wording guard before
        # 2026-08-24, and its recurring families cite the resolving URL in criteria that can
        # mention a spread or comparison incidentally. A full-text scan would silently remove
        # live anchors with the suite green — the opposite-direction twin of the narrow-gate
        # defect this round fixed.
        qt = "What will be the ending value of the UST 10Y Yield on August 14, 2026?"
        rc = (
            "Resolves per https://fred.stlouisfed.org/series/DGS10 on the resolution date. "
            "Commentators often quote the 2-year vs. the 10-year, but this question uses the 10-year alone."
        )
        route = route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc))
        assert route is not None
        assert (route.spec.series_id, route.derivation) == ("DGS10", "level")

    def test_two_leg_wording_vetoes_even_the_mom_pct_family(self):
        # The mom_pct exemption is scoped to the %-CHANGE tokens only. The two-leg tokens are
        # unconditional because no single-series band of any derivation can express a difference
        # between two series — so a CPI MoM question that also asks for a comparison against
        # another leg must still skip, and the exemption must not leak across the two patterns.
        qt = (
            "What will be the month-over-month percent change in US headline CPI "
            "versus the euro-area figure for December 2026?"
        )
        assert route_question(_make_numeric_q(question_text=qt)) is None
        assert tsr._change_guard_hits(qt, "mom_pct") is True
        assert tsr._PCT_CHANGE_RE.search(qt) is not None  # the exempted half really did match

    def test_a_percentage_change_question_still_skips_the_monthly_average_family(self):
        # monthly_avg's band is $/gal, so %-change wording is a different quantity: alongside
        # mom_diff (persons) this is the second non-exempt derivation, and only mom_pct is exempt.
        qt = "What will be the percent change in the monthly average price of regular gasoline in April 2026?"
        assert route_question(_make_numeric_q(question_text=qt)) is None
        assert tsr._change_guard_hits(qt, "monthly_avg") is True
        assert tsr._change_guard_hits(qt, "mom_pct") is False  # same text, exempt derivation

    def test_the_title_scoping_does_not_weaken_the_q45362_skip(self):
        # The families the URL guard exists for state their quantity in the title; the
        # verbatim q45362 test in TestLiveProdRoutingMisses pins the real case, this pins the
        # minimal shape so the two scoping rules can't be conflated.
        qt = "What will be the percentage change in the S&P 500 over the window?"
        rc = "Resolves per https://finance.yahoo.com/quote/%5EGSPC at the close."
        assert route_question(_make_numeric_q(question_text=qt, resolution_criteria=rc)) is None


# The q45401 defect class swept across the other non-"level" entries: a derivation gate that
# only knows some of the phrasings for its quantity goes dark silently on the rest.
class TestWidenedDerivationVocabulary:
    @pytest.mark.parametrize(
        "question_text",
        [
            "What will the monthly inflation rate for the US Consumer Price Index be in December 2026?",
            "What will the m/m headline CPI inflation print be for the United States in December 2026?",
            "By how much will the US CPI rise from the previous month in the December 2026 report?",
        ],
    )
    def test_cpi_mom_phrasings_beyond_the_original_tokens_route(self, question_text: str):
        route = route_question(_make_numeric_q(question_text=question_text))
        assert route is not None
        assert (route.spec.series_id, route.derivation) == ("CPIAUCSL", "mom_pct")

    def test_cpi_yoy_excludes_still_veto_the_widened_tokens(self):
        # The excludes outrank the widened require_any vocabulary, so a YoY question that
        # happens to mention the previous month stays out of the MoM family.
        qt = (
            "What will the year-over-year CPI inflation rate be for December 2026, "
            "compared with the rate from the previous month's report?"
        )
        assert route_question(_make_numeric_q(question_text=qt)) is None

    def test_gasoline_in_the_month_phrasing_routes_to_monthly_avg(self):
        qt = "What will the average price of regular gasoline be in the month of April 2026?"
        route = route_question(_make_numeric_q(question_text=qt))
        assert route is not None
        assert (route.spec.series_id, route.derivation) == ("GASREGW", "monthly_avg")

    # One carrier sentence per gated entry, with the token interpolated verbatim (never
    # stripped — "mom " and " add " carry deliberate delimiter spaces, so stripping them would
    # test a token the registry doesn't hold). The sentences read a little stiff for the
    # multi-word tokens; the token's presence is the point, not the prose.
    _GATE_TOKEN_CARRIERS: ClassVar[dict[tuple[str, str], str]] = {
        (
            "PAYEMS",
            "mom_diff",
        ): "What will the {token} in seasonally adjusted nonfarm payroll employment be in August 2026?",
        ("CPIAUCSL", "mom_pct"): "What will US headline CPI inflation be for December 2026, on a {token} basis?",
        ("GASREGW", "monthly_avg"): "What will the price of regular gasoline be, {token} of April 2026?",
    }

    def test_every_derivation_gate_token_in_the_registry_actually_routes(self):
        # Registry-driven rather than hand-listed, because the q45401 defect IS a gate that
        # knows only some of the phrasings for its quantity: a token added without a routing
        # check is the same failure waiting to happen, and a token that never routes is dead
        # vocabulary that reads as coverage. Walking the entries' own tuples covers every future
        # addition the moment it lands.
        for entry in tsr._TEMPLATE_REGISTRY:
            if not entry.require_any_keywords:
                continue
            carrier = self._GATE_TOKEN_CARRIERS.get((entry.series_id, entry.derivation))
            assert carrier is not None, (
                f"gated entry {entry.series_id}/{entry.derivation} has no carrier sentence — add one so its "
                "gate vocabulary is exercised"
            )
            for token in entry.require_any_keywords:
                route = route_question(_make_numeric_q(question_text=carrier.format(token=token)))
                assert route is not None, f"{entry.series_id} gate token {token!r} routes nowhere"
                assert (route.spec.series_id, route.derivation) == (entry.series_id, entry.derivation), token


_GSPC_YAHOO_RC = "Resolves per https://finance.yahoo.com/quote/%5EGSPC at the close."
_TWO_TICKER_RC = (
    "Resolves per https://finance.yahoo.com/quote/BTC-USD/ and https://finance.yahoo.com/quote/ETH-USD/ history."
)
_THREE_SERIES_RC = (
    "Per https://fred.stlouisfed.org/series/DGS10 and https://fred.stlouisfed.org/series/DGS2 "
    "and https://finance.yahoo.com/quote/%5EGSPC"
)

# (step label, question_text, resolution_criteria, expected decision, expected `series=` field).
# One row per branch `_route_question_impl` / `_single_url_route` can decide on; the completeness
# test below fails if the module grows a twelfth branch and this table doesn't.
_MARKER_BRANCHES: list[tuple[str, str, str, str, str]] = [
    ("url_single", "Where will the S&P 500 close?", _GSPC_YAHOO_RC, "routed", "^GSPC"),
    # Spread series are slash-joined, in leg order.
    ("url_spread", "Will Bitcoin outperform Ethereum in returns?", _TWO_TICKER_RC, "routed", "BTC-USD/ETH-USD"),
    (
        "url_no_relative_return_wording",
        "What will the ratio of Bitcoin to Ethereum be?",
        _TWO_TICKER_RC,
        "skipped",
        "BTC-USD/ETH-USD",
    ),
    # Ambiguity is comma-joined, FRED ids before tickers.
    ("url_ambiguous", "What will the gap be?", _THREE_SERIES_RC, "skipped", "DGS10,DGS2,^GSPC"),
    (
        "url_quantity_gate",
        "What will the CPI index level be in December 2026?",
        "Per https://fred.stlouisfed.org/series/CPIAUCSL",
        "skipped",
        "CPIAUCSL",
    ),
    (
        "url_change_vs_level_guard",
        "What will be the percentage change in the S&P 500 over the window?",
        _GSPC_YAHOO_RC,
        "skipped",
        "^GSPC",
    ),
    ("kw_single", "Where will the 10-year treasury yield close?", "rc", "routed", "DGS10"),
    (
        "kw_ambiguous",
        "What will the spread between the 10-year treasury yield and the high yield spread be?",
        "rc",
        "skipped",
        "DGS10,BAMLH0A0HYM2",
    ),
    (
        "kw_change_vs_level_guard",
        "What will be the percentage change in the S&P 500 over the last week of August 2026?",
        "rc",
        "skipped",
        "^GSPC",
    ),
    # The q45401 shape: title keywords hit, the quantity gate refused. Previously byte-identical
    # to a no-keyword miss (a bare silent None), which is what hid the defect for a whole era.
    (
        "kw_derivation_gate",
        "What will total nonfarm payroll employment (in thousands of persons) be in December 2026?",
        "rc",
        "skipped",
        "PAYEMS",
    ),
    # No series is known, so the field carries the "none" sentinel rather than an empty gap.
    ("kw_no_keyword_hit", "Who wins the 2028 election?", "rc", "skipped", "none"),
]


class TestRoutingMarker:
    """Every route_question call emits one TS_ANCHOR_ROUTE line — the marker that made anchor
    coverage queryable (27 of the era's 30 route-level misses previously left no log line).
    The line shapes here must stay in sync with the `ts_anchor_route` MarkerSpec
    (scripts/telemetry/markers.py), which tests/test_telemetry_markers.py pins parser-side.
    """

    @pytest.mark.parametrize(
        ("step", "question_text", "resolution_criteria", "decision", "series"),
        _MARKER_BRANCHES,
        ids=[branch[0] for branch in _MARKER_BRANCHES],
    )
    def test_each_branch_emits_its_own_step_label_and_parses_under_the_spec(
        self, step: str, question_text: str, resolution_criteria: str, decision: str, series: str, caplog
    ):
        r"""The step label is the entire analytical value of the marker: it distinguishes "no
        keyword matched" from "the gate refused" from "the guard fired", and a mislabelled branch
        reads in the archive as a different failure mode. Asserted VERBATIM, then re-parsed
        through the harvester's own regex — which takes each field with ``\S+``, so a multi-series
        value that ever grew a space (``", ".join``) would truncate the field and silently corrupt
        the harvested series list. Both joins here are deliberately spaceless."""
        spec = next(s for s in MARKER_SPECS if s.name == "ts_anchor_route")
        question = _make_numeric_q(qid=7150, question_text=question_text, resolution_criteria=resolution_criteria)

        with caplog.at_level(logging.INFO):
            route_question(question)

        lines = [r.getMessage() for r in caplog.records if "TS_ANCHOR_ROUTE" in r.getMessage()]
        expected = f"TS_ANCHOR_ROUTE: question=7150 decision={decision} series={series} step={step}"
        assert any(expected in line for line in lines), lines
        match = spec.regex.search(lines[0])
        assert match is not None
        assert match.groupdict() == {"question": "7150", "decision": decision, "series": series, "step": step}

    def test_the_table_covers_every_step_label_the_module_can_emit(self):
        # Without this, a new branch ships with no marker test and its label goes unverified in
        # the archive. Scans the module for the step-label naming convention (url_* / kw_*) rather
        # than trusting the table to have been kept in sync by hand.
        emitted = set(re.findall(r'"((?:url|kw)_[a-z_]+)"', inspect.getsource(tsr)))
        assert emitted == {branch[0] for branch in _MARKER_BRANCHES}
