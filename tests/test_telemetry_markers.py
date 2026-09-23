# HARNESS-SCAN-EXEMPT-monolithic-file-loc  # one class per MARKER_SPECS entry; a split fragments the registry
"""Tests for the run-log telemetry marker parser (scripts/telemetry/markers.py).

Each example line below is copied from the format string in the emitting code (the source of
truth), so a producer-side change to a marker shape breaks these tests loudly instead of
silently dropping records from the archive. The full marker-to-emitter map (which module and
function emits each token, plus any per-marker caveat) lives in docs/telemetry_markers.md.
"""

import json
import os
import re
from pathlib import Path

import pytest

from metaculus_bot.comment.markers import STACKER_SKIP_REASONS
from scripts.telemetry.markers import (
    MARKER_SPECS,
    coerce_value,
    parse_log_text,
    qid_from_ref,
)

# Prod cli.py log format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
PFX = "2026-07-17 14:23:01,123 - metaculus_bot.x - INFO - "
PFX_WARN = "2026-07-17 14:30:00,456 - metaculus_bot.x - WARNING - "

EXTRACTION_RUNG_LINE = (
    PFX + "EXTRACTION_RUNG: question=12345 model=openai/gpt-5.6-sol qtype=binary rung=block block_present=True"
)
GAP_FILL_V2_LINE = (
    "2026-07-21 14:25:10,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GAP_FILL_V2: model=openai/gpt-5.6-terra "
    "steps=7 tool_calls=9 searches=4 fetches=3 rendered=1 reads=2 dup_tool_calls=0 deadline_hit=False "
    "concluded_early=True wall_s=312.44 findings=5 pending_leads=1 lint_rejections=0 "
    "provenance_rejections=1 quote_mismatch_warnings=2 plan_gaps=3 plan_skipped=False "
    "conclude_gate_rejections=1 error=None"
)
# Verbatim from research/agentic/loop.py:_log_completion; error= alone tells this crash from a healthy idle run.
GAP_FILL_V2_CRASHED_LINE = (
    "2026-07-23 14:25:10,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GAP_FILL_V2: model=openai/gpt-5.6-terra "
    "steps=0 tool_calls=0 searches=0 fetches=0 rendered=0 reads=0 dup_tool_calls=0 deadline_hit=False "
    "concluded_early=False wall_s=0.12 findings=0 pending_leads=0 lint_rejections=0 "
    "provenance_rejections=0 quote_mismatch_warnings=0 plan_gaps=0 plan_skipped=False "
    "conclude_gate_rejections=0 error=APIConnectionError(\"No module named 'fastapi'\")"
)
# Verbatim from research/agentic/loop.py:_log_completion; pre-2026-07-21 shape, still replayed by re-harvesting.
GAP_FILL_V2_LEGACY_LINE = (
    "2026-07-17 14:25:10,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GAP_FILL_V2: model=openai/gpt-5.6-terra "
    "steps=7 tool_calls=9 searches=4 fetches=3 rendered=1 reads=2 dup_tool_calls=0 deadline_hit=False "
    "concluded_early=True wall_s=312.44 findings=5 pending_leads=1 lint_rejections=0"
)
GHOST_PRE_LINE = (
    "2026-07-21 14:25:09,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GHOST_PRE: gaps=3 sensitive_assumptions=2"
)
GHOST_PRE_JSON_LINE = (
    "2026-07-21 14:25:09,001 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GHOST_PRE_JSON: "
    '{"qtype":"binary","prob":0.35}'
)
GHOST_FORECAST_LINE = (
    "2026-07-17 14:25:11,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GHOST_FORECAST: qtype=binary summary=posterior_prob=0.4200"
)
GHOST_FORECAST_JSON_LINE = (
    "2026-07-17 14:25:11,001 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GHOST_FORECAST_JSON: "
    '{"qtype":"binary","prob":0.42}'
)
# Verbatim from research/agentic/loop.py:run_ghost_v1; the plain ghost's shapes under a _V1 token.
GHOST_FORECAST_V1_LINE = (
    "2026-09-09 14:25:31,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GHOST_FORECAST_V1: qtype=binary summary=posterior_prob=0.5500"
)
GHOST_FORECAST_V1_JSON_LINE = (
    "2026-09-09 14:25:31,000 - metaculus_bot.research.agentic.loop - INFO - "
    'question=https://www.metaculus.com/questions/38975/ GHOST_FORECAST_V1_JSON: {"qtype":"binary","prob":0.55}'
)
GHOST_FORECAST_JSON_NUMERIC_LINE = (
    "2026-07-17 14:25:11,002 - metaculus_bot.research.agentic.loop - INFO - "
    "question=12 GHOST_FORECAST_JSON: "
    '{"qtype":"numeric","declared_percentiles":{"0.1":10.0,"0.5":20.5,"0.9":30.0},"median":20.5}'
)
OPEN_BOUND_PILING_LINE = (
    PFX_WARN + "OPEN_BOUND_PILING: question=51000 model=gemini-3.1-pro-preview bound=upper "
    "bin_mass=0.153 declared_edge=1000 bound_value=1000"
)
# Copied from metaculus_bot/close_margin.py:format_close_margin_marker output.
CLOSE_MARGIN_LINE = (
    PFX + "CLOSE_MARGIN: question=44620 close_time=2026-07-20T00:00:00+00:00 "
    "submitted_at=2026-07-19T13:50:00+00:00 window_s=864000 margin_s=36600 margin_frac=0.0424"
)
CLOSE_MARGIN_NA_LINE = (
    PFX + "CLOSE_MARGIN: question=44620 close_time=2026-07-20T00:00:00+00:00 "
    "submitted_at=2026-07-19T00:00:00+00:00 window_s=n/a margin_s=86400 margin_frac=n/a"
)
# Verbatim from research/prediction_market.py:_log_ranking_telemetry; the four lines below are its four shapes.
MARKET_RANKING_RANKED_LINE = (
    PFX + "MARKET_RANKING: question=44620 pool=4 outcome=ranked rows=2 prompt_chars=36412 "
    "rendered=polymarket:2@0,kalshi:0@1"
)
MARKET_RANKING_EMPTY_LINE = (
    PFX + "MARKET_RANKING: question=44620 pool=0 outcome=empty rows=0 prompt_chars=0 rendered=none"
)
MARKET_CHILD_RENDER_LINE = (
    PFX + "MARKET_CHILD_RENDER: question=45363 families=6 full_rows=14 ladder_rows=6 outcomes=158 "
    "named=71 collapsed=87 withheld=3 max_stage=5 ladder_chars=1368"
)
MARKET_RANKING_FAILOPEN_LINE = (
    PFX + "MARKET_RANKING: question=None pool=4 outcome=failopen rows=2 prompt_chars=36412 "
    "rendered=polymarket:2@0,kalshi:0@1"
)
MARKET_RANKING_UNTRACEABLE_INDEX_LINE = (
    PFX + "MARKET_RANKING: question=44620 pool=4 outcome=ranked rows=1 prompt_chars=36412 rendered=manifold:-1@0"
)
TS_ANCHOR_ROUTE_ROUTED_LINE = PFX + "TS_ANCHOR_ROUTE: question=45401 decision=routed series=PAYEMS step=kw_single"
TS_ANCHOR_ROUTE_GATE_SKIP_LINE = (
    PFX + "TS_ANCHOR_ROUTE: question=45367 decision=skipped series=PAYEMS step=kw_derivation_gate"
)
TS_ANCHOR_ROUTE_NO_HIT_LINE = (
    PFX + "TS_ANCHOR_ROUTE: question=45193 decision=skipped series=none step=kw_no_keyword_hit"
)
TS_ANCHOR_ROUTE_SPREAD_LINE = PFX + "TS_ANCHOR_ROUTE: question=44700 decision=routed series=CL=F/^GSPC step=url_spread"
# Verbatim from financial_data.py:_fetch_yfinance_data and ts_render.py:_render_single, one shared estimator.
FINANCIAL_STALE_LATEST_YFINANCE_LINE = (
    PFX_WARN + "FINANCIAL_STALE_LATEST: surface=financial_data symbol=TEST age_d=3 cadence=calendar-day"
)
FINANCIAL_STALE_LATEST_TS_ANCHOR_LINE = (
    PFX_WARN + "FINANCIAL_STALE_LATEST: surface=ts_anchor symbol=^DEAD age_d=9 cadence=trading-day"
)
CREDIT_BALANCE_LINE = PFX + "CREDIT_BALANCE: key=donated phase=start remaining=123.45 usage=4.16"
CREDIT_BALANCE_SKIP_LINE = (
    PFX_WARN + "CREDIT_BALANCE: key=personal phase=start skipped (env var OPENROUTER_API_KEY not set)"
)
# Verbatim from credit_telemetry.py:_fetch_snapshot; the HTTP probe skip when DONATED_OPENROUTER_KEY_ENABLED is off.
CREDIT_BALANCE_DONATED_DISABLED_LINE = (
    PFX + "CREDIT_BALANCE: key=donated phase=start skipped (donated routing disabled)"
)
# Verbatim from credit_telemetry.py:log_end_and_check_floor; pre-2026-07-27 shape (no source=), still re-harvested.
CREDIT_SPEND_LINE = PFX + "CREDIT_SPEND: key=donated run_delta_usd=3.34 remaining=120.11"
CREDIT_SPEND_NA_LINE = PFX + "CREDIT_SPEND: key=personal run_delta_usd=n/a remaining=n/a"
# Current shape, verbatim from credit_telemetry.log_end_and_check_floor.
CREDIT_SPEND_REMAINING_SOURCE_LINE = (
    PFX + "CREDIT_SPEND: key=donated run_delta_usd=3.34 remaining=120.11 source=remaining_delta"
)
CREDIT_SPEND_UNSETTLED_SOURCE_LINE = (
    PFX + "CREDIT_SPEND: key=personal run_delta_usd=0.00 remaining=n/a source=usage_delta_unsettled"
)
CREDIT_FLOOR_BREACH_LINE = (
    PFX_WARN + "CREDIT_FLOOR_BREACH: key=donated remaining=45.00 floor=50.00 — donated OpenRouter "
    "balance is below the early-warning floor, so ask Metaculus for a top-up before it runs dry; "
    "the key is not necessarily empty and the run completed normally. cli.main logs the exit "
    "decision unless a higher-priority degradation alert exits first."
)

_META = {
    "run_id": "999",
    "workflow": "tournament",
    "artifact": "research-999",
    "run_date": "2026-07-17T14:00:00Z",
    "log_file": "run.log",
}


def _parse_one(line: str) -> dict:
    """Parse a single line, assert exactly one record came out, return it."""
    harvested = parse_log_text(line + "\n", **_META)
    records = [r for recs in harvested.values() for r in recs]
    assert len(records) == 1, f"expected 1 record, got {records}"
    return records[0]


class TestCoerceValue:
    def test_bools(self):
        assert coerce_value("True") is True
        assert coerce_value("False") is False

    def test_none_sentinels(self):
        assert coerce_value("None") is None
        assert coerce_value("n/a") is None

    def test_ints_and_floats(self):
        assert coerce_value("7") == 7
        assert isinstance(coerce_value("7"), int)
        assert coerce_value("312.44") == 312.44
        assert coerce_value("-4.0") == -4.0

    def test_strings_stay_strings(self):
        assert coerce_value("upper") == "upper"
        assert coerce_value("openai/gpt-5.6-sol") == "openai/gpt-5.6-sol"
        assert coerce_value("binary") == "binary"


class TestQidFromRef:
    def test_url(self):
        assert qid_from_ref("https://www.metaculus.com/questions/38975/") == 38975

    def test_bare_int(self):
        assert qid_from_ref("12345") == 12345

    def test_none_sentinel(self):
        assert qid_from_ref("None") is None
        assert qid_from_ref(None) is None


class TestExtractionRung:
    def test_fields(self):
        rec = _parse_one(EXTRACTION_RUNG_LINE)
        assert rec["marker"] == "extraction_rung"
        assert rec["question"] == "12345"
        assert rec["qid"] == 12345
        assert rec["model"] == "openai/gpt-5.6-sol"
        assert rec["qtype"] == "binary"
        assert rec["rung"] == "block"
        assert rec["block_present"] is True

    def test_qid_kind_is_question_id(self):
        """EXTRACTION_RUNG logs question.id_of_question, so its records live in the QUESTION-id space, the tag
        a residual join uses to translate correctly."""
        assert _parse_one(EXTRACTION_RUNG_LINE)["qid_kind"] == "question_id"

    def test_line_timestamp_parsed(self):
        rec = _parse_one(EXTRACTION_RUNG_LINE)
        assert rec["line_ts"].startswith("2026-07-17T14:23:01")

    def test_run_metadata_attached(self):
        rec = _parse_one(EXTRACTION_RUNG_LINE)
        assert rec["run_id"] == "999"
        assert rec["workflow"] == "tournament"
        assert rec["artifact"] == "research-999"

    def test_verbatim_real_prod_line(self):
        """Copied byte-for-byte from a real prod tournament run log (run 29633926137, 2026-07-18), not
        reconstructed, to guard against the regexes drifting from the actual emitted format."""
        real = (
            "2026-07-18 06:30:01,112 - metaculus_bot.value_extraction - INFO - "
            "EXTRACTION_RUNG: question=44620 model=openrouter/x-ai/grok-4.5 qtype=binary rung=block block_present=True"
        )
        rec = _parse_one(real)
        assert rec["qid"] == 44620
        assert rec["model"] == "openrouter/x-ai/grok-4.5"
        assert rec["qtype"] == "binary"
        assert rec["rung"] == "block"
        assert rec["block_present"] is True

    def test_llm_salvage_rung(self):
        line = PFX + "EXTRACTION_RUNG: question=None model=grok-4.5 qtype=numeric rung=llm block_present=False"
        rec = _parse_one(line)
        assert rec["rung"] == "llm"
        assert rec["qid"] is None
        assert rec["block_present"] is False


# Captured from the real emitter under tests/test_value_extraction.py's unrepairable-final-block test.
BLOCK_FALLBACK_EMITTED_LINE = (
    "2026-09-09 10:15:32,417 - metaculus_bot.value_extraction - INFO - "
    "BLOCK_FALLBACK: question=11 model=m qtype=binary skipped=1 rung=block "
    "reasons=repair: candidate 1/2: repaired JSON failed schema validation"
)
# The format string with prod-shaped values: two failed candidates joined by " | ", quotes and colons inside.
BLOCK_FALLBACK_TWO_REASONS_LINE = (
    PFX + "BLOCK_FALLBACK: question=45163 model=openrouter/google/gemini-3.1-pro-preview qtype=numeric "
    "skipped=2 rung=repair reasons=block: candidate 1/3: block declares 'above_range' but the question's "
    "upper bound is closed | repair: candidate 2/3: repaired JSON failed schema validation"
)


class TestBlockFallback:
    def test_emitted_line_fields(self):
        rec = _parse_one(BLOCK_FALLBACK_EMITTED_LINE)
        assert rec["marker"] == "block_fallback"
        assert rec["question"] == "11"
        assert rec["qid"] == 11
        assert rec["model"] == "m"
        assert rec["qtype"] == "binary"
        assert rec["skipped"] == 1
        assert rec["rung"] == "block"
        assert rec["reasons"] == "repair: candidate 1/2: repaired JSON failed schema validation"

    def test_qid_kind_is_question_id(self):
        """Same emitter module and the same question_id variable as EXTRACTION_RUNG."""
        assert _parse_one(BLOCK_FALLBACK_EMITTED_LINE)["qid_kind"] == "question_id"

    def test_reasons_is_kept_verbatim_to_end_of_line(self):
        """No raw_fields entry: the block:/repair:/llm: prefix can never read as a number, bool or sentinel."""
        rec = _parse_one(BLOCK_FALLBACK_TWO_REASONS_LINE)
        assert rec["reasons"] == (
            "block: candidate 1/3: block declares 'above_range' but the question's upper bound is closed"
            " | repair: candidate 2/3: repaired JSON failed schema validation"
        )
        assert isinstance(rec["reasons"], str)

    def test_prod_shaped_fields(self):
        rec = _parse_one(BLOCK_FALLBACK_TWO_REASONS_LINE)
        assert rec["qid"] == 45163
        assert rec["model"] == "openrouter/google/gemini-3.1-pro-preview"
        assert rec["qtype"] == "numeric"
        assert rec["skipped"] == 2
        assert rec["rung"] == "repair"

    def test_does_not_steal_the_extraction_rung_line_beside_it(self):
        """_run_ladder logs EXTRACTION_RUNG right after BLOCK_FALLBACK for the same forecast; one record each."""
        harvested = parse_log_text(BLOCK_FALLBACK_EMITTED_LINE + "\n" + EXTRACTION_RUNG_LINE + "\n", **_META)
        assert len(harvested["block_fallback"]) == 1
        assert len(harvested["extraction_rung"]) == 1


class TestGapFillV2:
    def test_fields(self):
        rec = _parse_one(GAP_FILL_V2_LINE)
        assert rec["marker"] == "gap_fill_v2"
        assert rec["qid"] == 38975
        # GAP_FILL_V2's question= comes from question.page_url -> a POST id.
        assert rec["qid_kind"] == "post_id"
        assert rec["model"] == "openai/gpt-5.6-terra"
        assert rec["steps"] == 7
        assert rec["tool_calls"] == 9
        assert rec["searches"] == 4
        assert rec["fetches"] == 3
        assert rec["rendered"] == 1
        assert rec["reads"] == 2
        assert rec["dup_tool_calls"] == 0
        assert rec["deadline_hit"] is False
        assert rec["concluded_early"] is True
        assert rec["wall_s"] == 312.44
        assert rec["findings"] == 5
        assert rec["pending_leads"] == 1
        assert rec["lint_rejections"] == 0
        # The five loop counters added 2026-07-21 (see loop.py:_log_completion).
        assert rec["provenance_rejections"] == 1
        assert rec["quote_mismatch_warnings"] == 2
        assert rec["plan_gaps"] == 3
        assert rec["plan_skipped"] is False
        assert rec["conclude_gate_rejections"] == 1
        # error= (added 2026-07-23) coerces "None" to Python None on a healthy run.
        assert rec["error"] is None

    def test_crashed_run_carries_error_repr(self):
        """A v2 crash and a legitimate idle run emit byte-identical counters; only error= tells them apart, which
        is exactly why the fastapi eager-import defect was silently dead without it."""
        rec = _parse_one(GAP_FILL_V2_CRASHED_LINE)
        assert rec["marker"] == "gap_fill_v2"
        assert rec["steps"] == 0
        assert rec["tool_calls"] == 0
        assert rec["findings"] == 0
        # repr(exc) is preserved verbatim (spaces and all) — it is not coerced away.
        assert rec["error"] == "APIConnectionError(\"No module named 'fastapi'\")"

    def test_legacy_line_without_new_counters_still_harvests(self):
        """Old-format lines (pre-2026-07-21) must keep parsing on re-harvest; the five new counter fields plus
        error come through as None, not a dropped record."""
        rec = _parse_one(GAP_FILL_V2_LEGACY_LINE)
        assert rec["marker"] == "gap_fill_v2"
        assert rec["qid"] == 38975
        assert rec["lint_rejections"] == 0
        assert rec["provenance_rejections"] is None
        assert rec["quote_mismatch_warnings"] is None
        assert rec["plan_gaps"] is None
        assert rec["plan_skipped"] is None
        assert rec["conclude_gate_rejections"] is None
        assert rec["error"] is None


class TestGhostPre:
    def test_fields(self):
        rec = _parse_one(GHOST_PRE_LINE)
        assert rec["marker"] == "ghost_pre"
        # question= comes from log_prefix (question.page_url) -> a POST id.
        assert rec["qid"] == 38975
        assert rec["qid_kind"] == "post_id"
        assert rec["gaps"] == 3
        assert rec["sensitive_assumptions"] == 2

    def test_json_payload_round_trips(self):
        rec = _parse_one(GHOST_PRE_JSON_LINE)
        assert rec["marker"] == "ghost_pre_json"
        assert rec["qid"] == 38975
        assert rec["qid_kind"] == "post_id"
        # forecast_json stays a raw string (never coerced) so the scorer can json.loads it.
        assert json.loads(rec["forecast_json"]) == {"qtype": "binary", "prob": 0.35}

    def test_does_not_collide_with_ghost_forecast_pair(self):
        """GHOST_PRE: / GHOST_PRE_JSON: / GHOST_FORECAST: / GHOST_FORECAST_JSON: are four distinct tokens; each
        line must harvest as exactly its own marker under the one-marker-per-line break."""
        assert _parse_one(GHOST_PRE_LINE)["marker"] == "ghost_pre"
        assert _parse_one(GHOST_PRE_JSON_LINE)["marker"] == "ghost_pre_json"
        assert _parse_one(GHOST_FORECAST_LINE)["marker"] == "ghost_forecast"
        assert _parse_one(GHOST_FORECAST_JSON_LINE)["marker"] == "ghost_forecast_json"


class TestGhostForecast:
    def test_binary(self):
        rec = _parse_one(GHOST_FORECAST_LINE)
        assert rec["marker"] == "ghost_forecast"
        assert rec["qid"] == 38975
        assert rec["qtype"] == "binary"
        assert rec["summary"] == "posterior_prob=0.4200"

    def test_multiple_choice_summary(self):
        line = (
            "2026-07-17 14:25:11,000 - metaculus_bot.research.agentic.loop - INFO - "
            "question=12 GHOST_FORECAST: qtype=multiple_choice summary=Blue=0.300, Red=0.700"
        )
        rec = _parse_one(line)
        assert rec["qtype"] == "multiple_choice"
        assert rec["summary"] == "Blue=0.300, Red=0.700"

    def test_numeric_median_only(self):
        line = (
            "2026-07-17 14:25:11,000 - metaculus_bot.research.agentic.loop - INFO - "
            "question=12 GHOST_FORECAST: qtype=numeric summary=median=42.5"
        )
        rec = _parse_one(line)
        assert rec["qtype"] == "numeric"
        assert rec["summary"] == "median=42.5"


class TestGhostForecastJson:
    def test_binary_payload_round_trips(self):
        rec = _parse_one(GHOST_FORECAST_JSON_LINE)
        assert rec["marker"] == "ghost_forecast_json"
        # qid is carried via the log_prefix leading group, exactly like GHOST_FORECAST.
        assert rec["qid"] == 38975
        # forecast_json stays a raw string (never coerced) so the scorer can json.loads it.
        assert json.loads(rec["forecast_json"]) == {"qtype": "binary", "prob": 0.42}

    def test_numeric_payload_carries_full_percentiles(self):
        rec = _parse_one(GHOST_FORECAST_JSON_NUMERIC_LINE)
        assert rec["marker"] == "ghost_forecast_json"
        assert rec["qid"] == 12
        payload = json.loads(rec["forecast_json"])
        assert payload == {
            "qtype": "numeric",
            "declared_percentiles": {"0.1": 10.0, "0.5": 20.5, "0.9": 30.0},
            "median": 20.5,
        }

    def test_does_not_collide_with_legacy_ghost_forecast(self):
        """A legacy GHOST_FORECAST line must NOT be mis-harvested as ghost_forecast_json, and vice versa; the
        two tokens are mutually exclusive under the one-marker-per-line break."""
        assert _parse_one(GHOST_FORECAST_LINE)["marker"] == "ghost_forecast"
        assert _parse_one(GHOST_FORECAST_JSON_LINE)["marker"] == "ghost_forecast_json"


# Verbatim from research/agentic/tools.py:_throttled_fetch_outcome; docs/telemetry_markers.md "AGENTIC_FETCH_THROTTLED".
_OGIMET_2022_URL = (
    "https://www.ogimet.com/cgi-bin/gsynext?lang=en&state=United+S&rank=10&ano=2022&mes=08&day=31&hora=23&Send=send"
)
AGENTIC_FETCH_THROTTLED_LINE = (
    PFX + "AGENTIC_FETCH_THROTTLED: url=" + _OGIMET_2022_URL + " method=rendered chars=303 phrase=query per"
)


class TestGhostForecastV1:
    """The ghost re-asked with gap-fill v1's section (2026-09-09): the plain ghost's two shapes under a _V1 token."""

    def test_fields_match_the_plain_ghost_shape(self):
        rec = _parse_one(GHOST_FORECAST_V1_LINE)
        assert rec["marker"] == "ghost_forecast_v1"
        assert (rec["qtype"], rec["summary"]) == ("binary", "posterior_prob=0.5500")
        assert (rec["qid"], rec["qid_kind"]) == (38975, "post_id")

    def test_json_payload_round_trips(self):
        rec = _parse_one(GHOST_FORECAST_V1_JSON_LINE)
        assert rec["marker"] == "ghost_forecast_v1_json"
        assert json.loads(rec["forecast_json"]) == {"qtype": "binary", "prob": 0.55}
        assert (rec["qid"], rec["qid_kind"]) == (38975, "post_id")

    def test_the_six_ghost_tokens_land_in_six_files(self):
        """GHOST_PRE(_JSON), GHOST_FORECAST(_JSON) and GHOST_FORECAST_V1(_JSON) share a prefix; each line must
        harvest into exactly its own file, or a v1 ghost would silently pair with itself as a plain ghost."""
        lines = [
            GHOST_PRE_LINE,
            GHOST_PRE_JSON_LINE,
            GHOST_FORECAST_LINE,
            GHOST_FORECAST_JSON_LINE,
            GHOST_FORECAST_V1_LINE,
            GHOST_FORECAST_V1_JSON_LINE,
        ]
        harvested = parse_log_text("\n".join(lines) + "\n", **_META)
        for marker in (
            "ghost_pre",
            "ghost_pre_json",
            "ghost_forecast",
            "ghost_forecast_json",
            "ghost_forecast_v1",
            "ghost_forecast_v1_json",
        ):
            assert len(harvested[marker]) == 1, marker


class TestAgenticFetchThrottled:
    """Per-fetch throttle-interstitial record.

    Worth a spec because the event had no trace whatever before it, and its failure mode is
    looking like a success: on q45191 two throttled fetches reached the driver as
    ``status: ok``, were cached, and were replayed on its own retry. The pair of fields the
    archive needs is ``chars`` + ``phrase`` — together they say whether a prod fire was a real
    throttle or the rule over-reaching, which is what the cap and the phrase list get retuned
    on.
    """

    def test_fields(self):
        rec = _parse_one(AGENTIC_FETCH_THROTTLED_LINE)
        assert rec["marker"] == "agentic_fetch_throttled"
        # A query-string URL with & and + survives the \S+ capture whole.
        assert rec["url"] == _OGIMET_2022_URL
        # plain-vs-rendered says whether escalating spent a second same-host request.
        assert rec["method"] == "rendered"
        assert rec["chars"] == 303
        assert rec["phrase"] == "query per"

    def test_no_question_ref_so_a_join_goes_through_the_run(self):
        """The tool handlers run below the loop's log_prefix and have no question id, exactly like the credit
        markers; asserting it here keeps a future "add question=" edit from landing without also declaring a
        qid_kind."""
        rec = _parse_one(AGENTIC_FETCH_THROTTLED_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec

    def test_a_single_word_phrase_still_parses(self):
        rec = _parse_one(PFX + "AGENTIC_FETCH_THROTTLED: url=https://x.test/a method=plain chars=42 phrase=ratelimit")
        assert rec["phrase"] == "ratelimit"
        assert rec["chars"] == 42


# Verbatim from local_document.py:log_local_document_read; see docs/telemetry_markers.md "AGENTIC_FETCH_LOCAL_DOC".
AGENTIC_FETCH_LOCAL_DOC_PDF_LINE = (
    PFX + "AGENTIC_FETCH_LOCAL_DOC: url=https://internationalaisafetyreport.org/report.pdf "
    "method=pdf_local chars=833450 pages=220 passages=n/a"
)
AGENTIC_FETCH_LOCAL_DOC_DIGEST_LINE = (
    PFX + "AGENTIC_FETCH_LOCAL_DOC: url=https://example.gov/tracker method=digest_local "
    "chars=41200 pages=n/a passages=6"
)


class TestAgenticFetchLocalDoc:
    """Per-document record of a read the ladder did for free.

    Worth a spec because it is how the local-first rung gets measured at all: before it, every
    PDF the gap-fill v2 driver met went to a paid Gemini url_context read and the only trace of
    one was the month's spend. The two methods are different reads — a fetch serving a PDF's
    extracted text, and a read_document answering an ask from selected passages — so the rate
    of each, and the digest's passage count, are what say whether the free route is working.
    """

    def test_pdf_fields(self):
        rec = _parse_one(AGENTIC_FETCH_LOCAL_DOC_PDF_LINE)
        assert rec["marker"] == "agentic_fetch_local_doc"
        assert rec["url"] == "https://internationalaisafetyreport.org/report.pdf"
        assert rec["method"] == "pdf_local"
        assert rec["chars"] == 833450
        assert rec["pages"] == 220
        # A pdf_local fetch serves the text and selects nothing, so there is no count to give.
        assert rec["passages"] is None

    def test_digest_fields(self):
        rec = _parse_one(AGENTIC_FETCH_LOCAL_DOC_DIGEST_LINE)
        assert rec["method"] == "digest_local"
        assert rec["chars"] == 41200
        # An HTML page has no pages to number; the passage count is the digest's own answer.
        assert rec["pages"] is None
        assert rec["passages"] == 6

    def test_zero_passages_is_recorded_not_dropped(self):
        """The reading that matters most: the digest ran and the document does not discuss what was asked. That
        reads like any other successful read in the block itself, so a 0 here must survive as 0, not "no
        data"."""
        rec = _parse_one(
            PFX + "AGENTIC_FETCH_LOCAL_DOC: url=https://x.test/a method=digest_local chars=900 pages=3 passages=0"
        )
        assert rec["passages"] == 0

    def test_no_question_ref_so_a_join_goes_through_the_run(self):
        rec = _parse_one(AGENTIC_FETCH_LOCAL_DOC_PDF_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec


# Verbatim from research/agentic/tools.py:read_document; the real 2026-09-03 receipt: robots.txt bars Google-Extended.
AGENTIC_URLCONTEXT_ROBOTS_SKIP_LINE = (
    PFX + "AGENTIC_URLCONTEXT_ROBOTS_SKIP: url=https://internationalaisafetyreport.org/chapters/2/ "
    "host=internationalaisafetyreport.org"
)


class TestAgenticUrlContextRobotsSkip:
    """Per-URL record of a paid document read the robots pre-check refused to make.

    Worth a spec because the pre-check trades one free request per host for a possible paid call,
    and this line is the only measurement of that trade: each record is a call not billed, while a
    rate far above the handful of hosts that publish the directive would mean the group parser is
    over-matching and withholding reads we could have had.
    """

    def test_fields(self):
        rec = _parse_one(AGENTIC_URLCONTEXT_ROBOTS_SKIP_LINE)
        assert rec["marker"] == "agentic_urlcontext_robots_skip"
        assert rec["url"] == "https://internationalaisafetyreport.org/chapters/2/"
        # The verdict is cached and applied per HOST, so the host is the unit a rate is taken over.
        assert rec["host"] == "internationalaisafetyreport.org"

    def test_a_query_string_url_survives_whole(self):
        rec = _parse_one(
            PFX + "AGENTIC_URLCONTEXT_ROBOTS_SKIP: url=https://x.test/a?b=1&c=2+3 host=x.test",
        )
        assert rec["url"] == "https://x.test/a?b=1&c=2+3"

    def test_no_question_ref_so_a_join_goes_through_the_run(self):
        rec = _parse_one(AGENTIC_URLCONTEXT_ROBOTS_SKIP_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec


class TestOpenBoundPiling:
    def test_fields(self):
        rec = _parse_one(OPEN_BOUND_PILING_LINE)
        assert rec["marker"] == "open_bound_piling"
        assert rec["qid"] == 51000
        assert rec["model"] == "gemini-3.1-pro-preview"
        assert rec["bound"] == "upper"
        assert rec["bin_mass"] == 0.153
        assert rec["declared_edge"] == 1000
        assert rec["bound_value"] == 1000


class TestCloseMargin:
    def test_full_fields(self):
        rec = _parse_one(CLOSE_MARGIN_LINE)
        assert rec["marker"] == "close_margin"
        assert rec["question"] == "44620"
        assert rec["qid"] == 44620
        # ISO timestamps stay strings (float() fails, so coerce_value leaves them be).
        assert rec["close_time"] == "2026-07-20T00:00:00+00:00"
        assert rec["submitted_at"] == "2026-07-19T13:50:00+00:00"
        assert rec["window_s"] == 864000
        assert isinstance(rec["window_s"], int)
        assert rec["margin_s"] == 36600
        assert rec["margin_frac"] == 0.0424

    def test_na_window_and_frac(self):
        rec = _parse_one(CLOSE_MARGIN_NA_LINE)
        assert rec["window_s"] is None
        assert rec["margin_frac"] is None
        assert rec["margin_s"] == 86400


class TestMarketRanking:
    """The ranked-retrieval port's own post-ship instrument, so it has to reach the archive.

    `rendered`'s pool indices are what answer the two questions the port left open (ranker
    attention decay down a ~400-candidate prompt, and whether Manifold detail enrichment
    changes the picks); `prompt_chars` is the free prod distribution against the prompt
    ceiling. Run logs leave GHA at 90 days, so anything unharvested is unanswerable later.
    """

    def test_ranked_run_fields(self):
        rec = _parse_one(MARKET_RANKING_RANKED_LINE)
        assert rec["marker"] == "market_ranking"
        assert rec["pool"] == 4
        assert rec["outcome"] == "ranked"
        assert rec["rows"] == 2
        assert rec["prompt_chars"] == 36412
        # The venue:pool_index@rank list survives whole — the indices are the instrument.
        assert rec["rendered"] == "polymarket:2@0,kalshi:0@1"

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(MARKET_RANKING_RANKED_LINE)
        # prediction_market.py emits question.id_of_question, not the post id.
        assert rec["qid"] == 44620
        assert rec["qid_kind"] == "question_id"

    def test_empty_pool_run_is_distinguishable_from_a_declining_ranker(self):
        """`empty` means the pool had nothing to rank (a retrieval story), while a ranker that legitimately
        returned zero rows reads `outcome=ranked rows=0`; the two must not collapse, since only the first
        implicates the venues."""
        rec = _parse_one(MARKET_RANKING_EMPTY_LINE)
        assert rec["outcome"] == "empty"
        assert rec["pool"] == 0
        assert rec["rows"] == 0
        assert rec["prompt_chars"] == 0
        # "none" is a _NONE_SENTINELS value, so an unrendered slate reads as None.
        assert rec["rendered"] is None

    def test_failopen_run_and_absent_qid(self):
        """A fail-open renders the head of the ranker's own input, so its rows are still worth measuring; qid is
        Optional at the call site and renders as "None"."""
        rec = _parse_one(MARKET_RANKING_FAILOPEN_LINE)
        assert rec["outcome"] == "failopen"
        assert rec["rows"] == 2
        assert rec["qid"] is None

    def test_untraceable_pool_index_sentinel_survives(self):
        """-1 means the rendered row could not be matched back to a pool entry, a defect in the index recovery
        rather than a real position; it must stay visible rather than coercing into the index distribution as a
        0."""
        rec = _parse_one(MARKET_RANKING_UNTRACEABLE_INDEX_LINE)
        assert rec["rendered"] == "manifold:-1@0"


class TestMarketChildRender:
    """The multi-outcome render's own instrument, and the reason it is a SEPARATE marker.

    `market_ranking`'s regex is not end-anchored, so appending fields to that line would have
    re-cut a spec other work touches; a new marker keeps the harvester change purely additive.

    `withheld` is what the line exists for. The Kalshi no-price spread threshold that blanks an
    empty book is calibrated on eleven fixture strikes, so its prod incidence has to be a query
    rather than a guess, and the same field counts the Polymarket placeholder legs and Manifold
    untouched priors. Run logs leave GHA at 90 days, so an unharvested line is unanswerable later.
    """

    def test_the_fields_survive_harvesting(self):
        rec = _parse_one(MARKET_CHILD_RENDER_LINE)
        assert rec["marker"] == "market_child_render"
        assert rec["families"] == 6
        assert rec["full_rows"] == 14
        assert rec["ladder_rows"] == 6
        assert rec["outcomes"] == 158
        assert rec["withheld"] == 3
        assert rec["max_stage"] == 5
        assert rec["ladder_chars"] == 1368

    def test_the_completeness_invariant_is_checkable_from_the_archive(self):
        """`named + collapsed == outcomes` is what the render guarantees, so a harvested line where
        they disagree is a render bug rather than a tuning signal — which only works if both halves
        reach the archive as numbers."""
        rec = _parse_one(MARKET_CHILD_RENDER_LINE)

        assert rec["named"] + rec["collapsed"] == rec["outcomes"]

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(MARKET_CHILD_RENDER_LINE)
        # prediction_market.py emits question.id_of_question, matching its MARKET_RANKING sibling.
        assert rec["qid"] == 45363
        assert rec["qid_kind"] == "question_id"


# Verbatim from research/prediction_market.py:_rank_pool; before 2026-08-25, shape_regression wrongly reported ok(0).
MARKET_RANKING_DEGRADED_SHAPE_LINE = (
    PFX_WARN + "MARKET_RANKING_DEGRADED: question=44620 pool=17 reason=shape_regression "
    "detail=falling back to retrieval order; 3 entries yielded no usable pick (renamed index key, "
    "or every index outside a pool of 17); first={'index': 4, 'tier': 'weak'}"
)
MARKET_RANKING_DEGRADED_UNREADABLE_LINE = (
    PFX_WARN + "MARKET_RANKING_DEGRADED: question=None pool=4 reason=unreadable "
    "detail=falling back to retrieval order; empty completion"
)


class TestMarketRankingDegraded:
    """The sibling that says WHY a question rendered retrieval order.

    `market_ranking`'s `outcome=failopen` records that a fail-open happened; only this line
    separates "the model emitted something that is not a ranking array" from "our own
    prompt/parser contract broke", and the second is the regression that used to arrive as
    `ok(0)`. Run logs leave GHA at 90 days, so an archive holding one line and not the other
    cannot answer which failure a degraded question hit.
    """

    def test_shape_regression_fields(self):
        rec = _parse_one(MARKET_RANKING_DEGRADED_SHAPE_LINE)
        assert rec["marker"] == "market_ranking_degraded"
        assert rec["pool"] == 17
        assert rec["reason"] == "shape_regression"

    def test_detail_free_text_survives_verbatim(self):
        """detail holds repr(exc): spaces, parentheses, quotes, a dict repr. It belongs to _RAW_FIELDS so it is
        never coerced, and the regex is not end-anchored so a terser future form still harvests."""
        rec = _parse_one(MARKET_RANKING_DEGRADED_SHAPE_LINE)
        assert "renamed index key" in rec["detail"]
        assert rec["detail"].endswith("first={'index': 4, 'tier': 'weak'}")

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(MARKET_RANKING_DEGRADED_SHAPE_LINE)
        # prediction_market.py emits question.id_of_question, matching its two siblings.
        assert rec["qid"] == 44620
        assert rec["qid_kind"] == "question_id"

    def test_unreadable_reason_and_absent_qid(self):
        rec = _parse_one(MARKET_RANKING_DEGRADED_UNREADABLE_LINE)
        assert rec["reason"] == "unreadable"
        assert rec["qid"] is None

    def test_does_not_collide_with_the_market_ranking_spec(self):
        """MARKET_RANKING_DEGRADED contains MARKET_RANKING as a prefix, and market_ranking's spec sits EARLIER
        in MARKER_SPECS, so under the one-marker-per-line break a colon-less prefix match there would have
        swallowed every degraded line."""
        harvested = parse_log_text(
            MARKET_RANKING_DEGRADED_SHAPE_LINE + "\n" + MARKET_RANKING_RANKED_LINE + "\n", **_META
        )
        assert len(harvested["market_ranking_degraded"]) == 1
        assert len(harvested["market_ranking"]) == 1


# Verbatim from prediction_market.py:_log_tier_caps; one line per question whose top grade the staleness cap refused.
MARKET_TIER_CAPPED_LINE = PFX + "MARKET_TIER_CAPPED: question=45163 rows=1 capped=manifold@0"
MARKET_TIER_CAPPED_MULTI_LINE = PFX + "MARKET_TIER_CAPPED: question=45163 rows=2 capped=manifold@0,kalshi@3"


class TestMarketTierCapped:
    """The staleness tier cap, harvested because it fires on nothing yet.

    Zero of the 102 archived snapshots carry a cap, and q45163's own offending row was graded
    one tier below the cap's reach, so the interesting record is the FIRST one: it means the
    ranker called a long-closed market same-quantity-same-date. A marker that is expected to
    stay empty still needs a spec, or "did this ever fire in prod" becomes a re-scrape of GHA
    logs that expire at 90 days.
    """

    def test_fields(self):
        rec = _parse_one(MARKET_TIER_CAPPED_LINE)
        assert rec["marker"] == "market_tier_capped"
        assert rec["rows"] == 1
        assert rec["capped"] == "manifold@0"

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(MARKET_TIER_CAPPED_LINE)
        # prediction_market.py emits question.id_of_question, matching its three siblings.
        assert rec["qid"] == 45163
        assert rec["qid_kind"] == "question_id"

    def test_a_multi_row_cap_survives_whole(self):
        """`capped` is a comma-joined venue@rank list with no spaces, so the whole field arrives as one string;
        a per-row split is the analyst's job, not the parser's."""
        rec = _parse_one(MARKET_TIER_CAPPED_MULTI_LINE)
        assert rec["rows"] == 2
        assert rec["capped"] == "manifold@0,kalshi@3"

    def test_does_not_collide_with_the_two_market_ranking_specs(self):
        """All three tokens start `MARKET_`, and both ranking specs sit EARLIER in MARKER_SPECS, so a loose
        prefix match there would have swallowed every cap line before its own spec was reached; pinned in both
        directions, each of the three lines harvests as exactly itself."""
        harvested = parse_log_text(
            "\n".join([MARKET_TIER_CAPPED_LINE, MARKET_RANKING_RANKED_LINE, MARKET_RANKING_DEGRADED_SHAPE_LINE]) + "\n",
            **_META,
        )
        assert len(harvested["market_tier_capped"]) == 1
        assert len(harvested["market_ranking"]) == 1
        assert len(harvested["market_ranking_degraded"]) == 1


# Verbatim from numeric/pipeline.py, numeric/utils.py and spread_metrics.py; trailing prose forbids end-anchoring.
NUMERIC_DEGENERATE_DECLARATION_LINE = (
    "2026-08-25 23:07:17,044 - metaculus_bot.numeric.pipeline - WARNING - "
    "NUMERIC_DEGENERATE_DECLARATION: question=77 model=openrouter/openai/gpt-5.6-sol n_unique=1 "
    "span=0 value_eps=1e-07 spread_applied=false"
)
NUMERIC_DEGENERATE_DECLARATION_UNLABELLED_LINE = (
    "2026-08-25 23:07:17,044 - metaculus_bot.numeric.pipeline - WARNING - "
    "NUMERIC_DEGENERATE_DECLARATION: question=77 model=unknown n_unique=1 "
    "span=1.5e-06 value_eps=1e-07 spread_applied=false"
)
NUMERIC_AGGREGATE_GRID_MISMATCH_LINE = (
    "2026-08-25 23:07:17,044 - metaculus_bot.numeric.utils - WARNING - "
    "NUMERIC_AGGREGATE_GRID_MISMATCH: question=44620 model_index=2 got_points=201 expected_points=11 "
    "— resampling in cdf-location space before aggregation"
)
SPREAD_UNDEFINED_LINE = (
    "2026-08-25 23:07:17,044 - metaculus_bot.spread_metrics - WARNING - "
    "SPREAD_UNDEFINED: question=45363 qtype=numeric denominator=-0 models=3 — key-percentile spread "
    "is unmeasurable (non-positive denominator); reporting inf so it cannot read as agreement"
)
# Verbatim from numeric/pchip_cdf.py:safe_cdf_bounds, q45065's real opus-4.8 call plus the conc21_closed snap golden.
CDF_MAXSTEP_CLIP_LINE = (
    "2026-08-31 23:02:02,257 - metaculus_bot.numeric.pchip_cdf - WARNING - "
    "CDF_MAXSTEP_CLIP: question=45065 model=openrouter/anthropic/claude-opus-4.8 "
    "clipped_mass=0.517852 over_cap_bins=1 bins_displaced=4 max_offset_bins=2 "
    "pre_max_step=0.717852 max_step=0.200000"
)
CDF_MAXSTEP_CLIP_ENSEMBLE_LINE = (
    "2026-08-31 23:02:02,258 - metaculus_bot.numeric.pchip_cdf - WARNING - "
    "CDF_MAXSTEP_CLIP: question=None model=ensemble_discrete_snap "
    "clipped_mass=0.399018 over_cap_bins=3 bins_displaced=4 max_offset_bins=1 "
    "pre_max_step=0.499480 max_step=0.200000"
)
NUMERIC_PCHIP_FALLBACK_LINE = (
    "2026-06-11 21:14:03,512 - metaculus_bot.numeric.diagnostics - WARNING - "
    "Question 43913: PCHIP CDF construction failed (Percentile values must be strictly increasing), "
    "falling back to forecasting-tools default"
)
NUMERIC_PCHIP_FALLBACK_NO_ID_LINE = (
    "2026-06-11 21:14:03,512 - metaculus_bot.numeric.diagnostics - WARNING - "
    "Question N/A: PCHIP CDF construction failed (boom), falling back to forecasting-tools default"
)


class TestNumericDegenerateDeclaration:
    """A per-forecaster point-mass declaration, and what the grid made of it.

    On the 201-point continuous grid a point-mass declaration is not cluster-spread
    (``spread_applied=false``), so the unit-mismatch guard sees the model's own zero span and
    withholds the forecaster: there the count is a fabrication-ATTEMPT rate, which is why it
    needs a spec. Where the published bins are the outcome space (a discrete question or any
    non-201 grid) the collapse is spread under the one-bin cap (``spread_applied=true``) and the
    member publishes with its mass inside the bin it named. The 201-grid drop itself lands in
    FORECASTER_DROPS as an UnitMismatchError; only this line names the cause, and its
    predecessor (`Cluster spread applied`) was never harvested — which is exactly why the
    finding's prod incidence was unanswerable from the archive.
    """

    def test_fields(self):
        rec = _parse_one(NUMERIC_DEGENERATE_DECLARATION_LINE)
        assert rec["marker"] == "numeric_degenerate_declaration"
        assert rec["model"] == "openrouter/openai/gpt-5.6-sol"
        assert rec["n_unique"] == 1
        assert rec["span"] == 0
        assert rec["value_eps"] == pytest.approx(1e-07)
        assert rec["spread_applied"] is False

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(NUMERIC_DEGENERATE_DECLARATION_LINE)
        assert rec["qid"] == 77
        assert rec["qid_kind"] == "question_id"

    def test_spread_applied_true_harvests_as_a_bool(self):
        """On a grid whose bins are the outcome space the collapse IS spread and published, and the
        emitter renders the real value; the ``\\S+`` capture takes it and coercion makes it a bool."""
        rec = _parse_one(NUMERIC_DEGENERATE_DECLARATION_LINE.replace("spread_applied=false", "spread_applied=true"))
        assert rec["spread_applied"] is True

    def test_unlabelled_model_stays_a_readable_string(self):
        """The 'unknown' fallback fires when a caller passes no model_name (all three production callers pass one
        as of 8cccdaa, covering historical lines and any future caller that forgets). It is not a _NONE_SENTINELS
        member, so it must survive as a string rather than coercing to None (indistinguishable from a missing
        field)."""
        rec = _parse_one(NUMERIC_DEGENERATE_DECLARATION_UNLABELLED_LINE)
        assert rec["model"] == "unknown"
        # %.6g renders this sub-epsilon span in exponent form, which the unit-mismatch guard judges as a number.
        assert rec["span"] == pytest.approx(1.5e-06)


class TestNumericAggregateGridMismatch:
    """Expect zero records in prod; a nonzero count means a model's CDF length drifted.

    Worth harvesting because the predecessor defect was invisible: group-by-VALUE
    aggregation medianed over a rotating SUBSET of the ensemble whenever an ft-fallback
    distribution mixed with PCHIP ones, and nothing recorded the partial membership.
    """

    def test_fields(self):
        rec = _parse_one(NUMERIC_AGGREGATE_GRID_MISMATCH_LINE)
        assert rec["marker"] == "numeric_aggregate_grid_mismatch"
        assert rec["model_index"] == 2
        assert rec["got_points"] == 201
        assert rec["expected_points"] == 11

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(NUMERIC_AGGREGATE_GRID_MISMATCH_LINE)
        assert rec["qid"] == 44620
        assert rec["qid_kind"] == "question_id"


class TestCdfMaxstepClip:
    """The repair that reshaped 47% of q45065's published mass while logging at DEBUG.

    Not alertable — the per-bin cap is the platform's — but the two displacement fields
    are what make the repair's own POLICY auditable after the fact, which is exactly
    what was missing when the retired slack-proportional redistribution was diagnosed
    a month after the forecast resolved.
    """

    def test_fields(self):
        rec = _parse_one(CDF_MAXSTEP_CLIP_LINE)
        assert rec["marker"] == "cdf_maxstep_clip"
        assert rec["model"] == "openrouter/anthropic/claude-opus-4.8"
        assert rec["clipped_mass"] == pytest.approx(0.517852)
        assert rec["over_cap_bins"] == 1
        assert rec["bins_displaced"] == 4
        assert rec["max_offset_bins"] == 2
        assert rec["pre_max_step"] == pytest.approx(0.717852)
        assert rec["max_step"] == pytest.approx(0.2)

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(CDF_MAXSTEP_CLIP_LINE)
        assert rec["qid"] == 45065
        assert rec["qid_kind"] == "question_id"

    def test_ensemble_stage_label_and_absent_question(self):
        """The aggregation stages have no forecaster to name, so they label the stage; a caller with no question
        in scope renders "None", which must coerce to a real absent field rather than the string."""
        rec = _parse_one(CDF_MAXSTEP_CLIP_ENSEMBLE_LINE)
        assert rec["model"] == "ensemble_discrete_snap"
        assert rec["qid"] is None
        assert rec["over_cap_bins"] == 3


class TestNumericPchipFallback:
    """The one numeric repair surface with a confirmed prod fire (q43913's degenerate
    declaration raised out of generate_pchip_cdf), so it earns a spec where the dead
    repair-tier WARNs deliberately do not — see the M16 note in markers.py."""

    def test_fields(self):
        rec = _parse_one(NUMERIC_PCHIP_FALLBACK_LINE)
        assert rec["marker"] == "numeric_pchip_fallback"
        assert rec["error"] == "Percentile values must be strictly increasing"

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(NUMERIC_PCHIP_FALLBACK_LINE)
        assert rec["qid"] == 43913
        assert rec["qid_kind"] == "question_id"

    def test_a_questionless_emission_still_harvests(self):
        """log_pchip_fallback renders a missing id as "N/A" (a _NONE_SENTINELS member)."""
        rec = _parse_one(NUMERIC_PCHIP_FALLBACK_NO_ID_LINE)
        assert rec["qid"] is None
        assert rec["error"] == "boom"


class TestSpreadUndefined:
    def test_fields(self):
        rec = _parse_one(SPREAD_UNDEFINED_LINE)
        assert rec["marker"] == "spread_undefined"
        assert rec["qtype"] == "numeric"
        assert rec["models"] == 3
        # %.6g of a negative zero denominator renders "-0", which must read as the number it is, not a string.
        assert rec["denominator"] == 0

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(SPREAD_UNDEFINED_LINE)
        assert rec["qid"] == 45363
        assert rec["qid_kind"] == "question_id"

    def test_positive_denominator_variant_of_the_same_shape_harvests(self):
        """The guard is `denominator <= 0`, so a plain 0 is the common case; qtype is a captured field rather
        than a literal, so a future binary/MC variant needs no spec change."""
        rec = _parse_one(
            PFX_WARN + "SPREAD_UNDEFINED: question=1 qtype=numeric denominator=0 models=2 — key-percentile spread "
            "is unmeasurable (non-positive denominator); reporting inf so it cannot read as agreement"
        )
        assert rec["denominator"] == 0
        assert rec["models"] == 2


class TestTsAnchorRoute:
    """The routing marker that made anchor coverage a query instead of an offline re-run.

    route_question used to log only the ambiguous/guard branches: 27 of the triple era's 30
    route-level misses were the silent `kw_no_keyword_hit` return and left no line in 1,800
    persisted run logs. Every decision now emits one line, and losing it from the archive
    would put the next coverage audit back to reconstructing routes offline.
    """

    def test_routed_fields(self):
        rec = _parse_one(TS_ANCHOR_ROUTE_ROUTED_LINE)
        assert rec["marker"] == "ts_anchor_route"
        assert rec["decision"] == "routed"
        assert rec["series"] == "PAYEMS"
        assert rec["step"] == "kw_single"

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(TS_ANCHOR_ROUTE_ROUTED_LINE)
        assert rec["qid"] == 45401
        assert rec["qid_kind"] == "question_id"

    def test_derivation_gate_skip_names_the_refusing_entry(self):
        """The q45401 defect class: title keywords hit, the quantity gate refused. Naming the series is what
        makes the marker actionable; a bare "skipped" would collapse this back into the no-keyword miss it was
        previously indistinguishable from."""
        rec = _parse_one(TS_ANCHOR_ROUTE_GATE_SKIP_LINE)
        assert rec["decision"] == "skipped"
        assert rec["series"] == "PAYEMS"
        assert rec["step"] == "kw_derivation_gate"

    def test_a_plain_keyword_miss_reads_none_series(self):
        rec = _parse_one(TS_ANCHOR_ROUTE_NO_HIT_LINE)
        # "none" is a _NONE_SENTINELS value, so a series-less skip reads as None.
        assert rec["series"] is None
        assert rec["step"] == "kw_no_keyword_hit"

    def test_a_spread_ref_survives_whole(self):
        rec = _parse_one(TS_ANCHOR_ROUTE_SPREAD_LINE)
        assert rec["series"] == "CL=F/^GSPC"
        assert rec["step"] == "url_spread"


class TestFinancialStaleLatest:
    """The stale-"latest" disclosure, one spec for both emitting surfaces.

    Informational, not alertable: the render already tells the forecaster to treat the
    value as stale. Harvesting it is what turns "how often does each surface serve a
    stale anchor value" into a query — run logs expire from GHA at 90 days.
    """

    def test_yfinance_surface_fields(self):
        rec = _parse_one(FINANCIAL_STALE_LATEST_YFINANCE_LINE)
        assert rec["marker"] == "financial_stale_latest"
        assert rec["surface"] == "financial_data"
        assert rec["symbol"] == "TEST"
        assert rec["age_d"] == 3
        assert rec["cadence"] == "calendar-day"

    def test_ts_anchor_surface_fields(self):
        """A caret-prefixed Yahoo index symbol must survive as the string it is."""
        rec = _parse_one(FINANCIAL_STALE_LATEST_TS_ANCHOR_LINE)
        assert rec["surface"] == "ts_anchor"
        assert rec["symbol"] == "^DEAD"
        assert rec["age_d"] == 9
        assert rec["cadence"] == "trading-day"

    def test_no_question_ref(self):
        """Per-identifier, not per-question (one question can fire several), so the record carries no qid at
        all, the same shape as the credit markers."""
        rec = _parse_one(FINANCIAL_STALE_LATEST_YFINANCE_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec


# Verbatim from noise_flag.py:noise_flag_line; long_vol reads None on the anchor surface, as on a too-short series.
FINANCIAL_NOISE_FLAG_YFINANCE_LINE = (
    PFX + "FINANCIAL_NOISE_FLAG: surface=financial_data symbol=USDSZL=X vr_lag=5 vr=0.369 floor=0.6 "
    "short_vol=17.9 long_vol=15.2 robust_vol=10.8"
)
FINANCIAL_NOISE_FLAG_TS_ANCHOR_LINE = (
    PFX + "FINANCIAL_NOISE_FLAG: surface=ts_anchor symbol=CSUSHPISA vr_lag=5 vr=0.412 floor=0.6 "
    "short_vol=14.6 long_vol=None robust_vol=9.4"
)
FINANCIAL_NOISE_FLAG_NO_ESTIMATES_LINE = (
    PFX + "FINANCIAL_NOISE_FLAG: surface=financial_data symbol=USDSZL=X vr_lag=5 vr=0.369 floor=0.6 "
    "short_vol=17.9 long_vol=None robust_vol=None"
)


class TestFinancialNoiseFlag:
    """The vendor-noise flag, one spec for both emitting surfaces.

    Informational and not alertable, like its stale-latest sibling: the rendered block already
    tells the forecaster the one-day-return volatility is inflated and leads with the
    noise-robust figure instead. Harvesting it is what turns "how often does each surface
    serve a noise-dominated volatility" into a query after the 90-day GHA log expiry.
    """

    def test_yfinance_surface_fields(self):
        rec = _parse_one(FINANCIAL_NOISE_FLAG_YFINANCE_LINE)
        assert rec["marker"] == "financial_noise_flag"
        assert rec["surface"] == "financial_data"
        # Without the ticker, two flagged identifiers in one run would harvest as byte-identical anonymous records.
        assert rec["symbol"] == "USDSZL=X"
        assert rec["vr_lag"] == 5
        assert rec["vr"] == 0.369
        assert rec["floor"] == 0.6
        assert rec["short_vol"] == 17.9
        assert rec["long_vol"] == 15.2
        assert rec["robust_vol"] == 10.8

    def test_ts_anchor_surface_reads_none_for_the_long_window(self):
        """Same field set as the yfinance surface; `surface` tells "no long window" apart from "series too
        short"."""
        rec = _parse_one(FINANCIAL_NOISE_FLAG_TS_ANCHOR_LINE)
        assert rec["surface"] == "ts_anchor"
        assert rec["symbol"] == "CSUSHPISA"
        assert rec["short_vol"] == 14.6
        assert rec["robust_vol"] == 9.4
        assert rec["long_vol"] is None

    def test_a_refused_estimate_reads_none(self):
        """Both estimators return None on a series with no measurable return variation (an administratively
        fixed quote); the emitter renders that as "None", a _NONE_SENTINELS member, so it coerces to None
        instead of a fabricated 0.0."""
        rec = _parse_one(FINANCIAL_NOISE_FLAG_NO_ESTIMATES_LINE)
        assert rec["long_vol"] is None
        assert rec["robust_vol"] is None
        assert rec["short_vol"] == 17.9

    def test_no_question_ref(self):
        """Per-identifier like the stale-latest sibling, and neither call site has the question in scope, so the
        record carries no qid at all."""
        rec = _parse_one(FINANCIAL_NOISE_FLAG_YFINANCE_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec

    def test_does_not_collide_with_the_stale_latest_spec(self):
        """Both tokens start `FINANCIAL_`, and financial_stale_latest sits EARLIER in MARKER_SPECS, so a loose
        prefix match there would have swallowed every noise-flag line under the one-marker-per-line break."""
        harvested = parse_log_text(
            FINANCIAL_NOISE_FLAG_YFINANCE_LINE + "\n" + FINANCIAL_STALE_LATEST_YFINANCE_LINE + "\n", **_META
        )
        assert len(harvested["financial_noise_flag"]) == 1
        assert len(harvested["financial_stale_latest"]) == 1


# Verbatim from resolution_source.py:_log_fetch_outcome_markers; one line per fetched URL, tier told apart by CDN url.
RESOLUTION_SOURCE_FETCH_OK_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=44554 url=https://www.racetothewh.com/senate/26 "
    "status=ok http=200 embeds=none caller=resolution_source"
)
RESOLUTION_SOURCE_FETCH_EMBED_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=44554 url=https://www.racetothewh.com/senate/26 "
    "status=ok http=200 embeds=infogram,tableau caller=resolution_source"
)
RESOLUTION_SOURCE_FETCH_NO_CONTENT_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=44556 url=https://tracker.example.com/senate "
    "status=no_resolving_content http=200 embeds=infogram reason=embed_shell caller=resolution_source"
)
RESOLUTION_SOURCE_FETCH_THIN_PAGE_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=45088 url=https://data.wastewaterscan.org/ "
    "status=no_resolving_content http=200 embeds=none reason=thin_page caller=resolution_source"
)
RESOLUTION_SOURCE_FETCH_NO_MATCHING_PASSAGE_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=45363 url=https://www.bls.gov/news.release/pdf/wkstp.pdf "
    "status=no_resolving_content http=200 embeds=none reason=no_matching_passage route=pdf_local caller=resolution_source"
)
RESOLUTION_SOURCE_FETCH_BLOCKED_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=44211 url=https://www.cbp.gov/newsroom/stats "
    "status=blocked http=403 embeds=none caller=resolution_source"
)
RESOLUTION_SOURCE_FETCH_NO_RESPONSE_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=44211 url=https://slow.example.com/x status=error http=n/a embeds=none"
)
RESOLUTION_SOURCE_FETCH_DATASET_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=44841 url=https://static.dwcdn.net/data/kSCt4.csv "
    "status=ok http=200 embeds=none caller=resolution_source"
)
# `route` names the rung that produced the outcome; keyed and last, so all 4 optional-tail combinations parse.
RESOLUTION_SOURCE_FETCH_ROUTE_ONLY_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=44554 url=https://www.racetothewh.com/senate/26 "
    "status=ok http=200 embeds=none route=wayback caller=resolution_source"
)
RESOLUTION_SOURCE_FETCH_REASON_AND_ROUTE_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=44556 url=https://tracker.example.com/senate "
    "status=no_resolving_content http=200 embeds=infogram reason=embed_shell route=rendered caller=resolution_source"
)
# failure_class, exc and server are keyed and sit after route (the 403/CDN diagnostics tail), so all subsets parse.
RESOLUTION_SOURCE_FETCH_FAILURE_CLASS_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=44211 url=https://www.cbp.gov/newsroom/stats "
    "status=blocked http=403 embeds=none failure_class=http_403 server=akamaighost caller=resolution_source"
)
RESOLUTION_SOURCE_FETCH_TRANSPORT_ERROR_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=44211 url=https://slow.example.com/x "
    "status=error http=n/a embeds=none failure_class=timeout exc=ServerTimeoutError caller=resolution_source"
)
# The page digest's three counters (research/page_digest.py), keyed and last, appended only where the digest ran.
RESOLUTION_SOURCE_FETCH_DIGEST_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=44554 url=https://www.bls.gov/news.release/empsit.nr0.htm "
    "status=ok http=200 embeds=none route=direct passages_returned=6 passages_grounded=4 fallback_used=False"
)
RESOLUTION_SOURCE_FETCH_DIGEST_FALLBACK_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=44554 url=https://www.bls.gov/news.release/empsit.nr0.htm "
    "status=ok http=200 embeds=none passages_returned=0 passages_grounded=0 fallback_used=True"
)
# Every optional tail group at once, in the documented order; the spec's positional groups make this order the contract.
RESOLUTION_SOURCE_FETCH_FULL_TAIL_LINE = (
    PFX + "RESOLUTION_SOURCE_FETCH: question=44211 url=https://www.cbp.gov/newsroom/stats "
    "status=no_resolving_content http=200 embeds=none reason=thin_page route=rendered "
    "failure_class=http_5xx exc=ServerTimeoutError server=akamaighost "
    "passages_returned=3 passages_grounded=2 fallback_used=False"
)


class TestAsknewsNoArticles:
    def test_fields(self):
        """Both AskNews phases came back empty, so the provider returned the empty string (providers.py)."""
        rec = _parse_one(PFX_WARN + "ASKNEWS_NO_ARTICLES: question=45085 hot=0 historical=0")
        assert rec["marker"] == "asknews_no_articles"
        assert rec["qid"] == 45085
        assert rec["qid_kind"] == "question_id"
        assert rec["hot"] == 0
        assert rec["historical"] == 0

    def test_question_without_an_id_parses_with_qid_none(self):
        """The provider reads the id with getattr(..., None), so a question with no id logs question=None."""
        rec = _parse_one(PFX_WARN + "ASKNEWS_NO_ARTICLES: question=None hot=0 historical=0")
        assert rec["qid"] is None
        assert rec["question"] == "None"


class TestResolutionSourceFetch:
    """Per-URL fetch outcomes, harvested (item 19d).

    They used to live only in free-text log lines and the published comment's
    provider-diagnostics block, so "cdc.gov is 0 successes in 1,069 fetch records"
    meant re-scraping run logs that expire from GHA at 90 days.
    """

    def test_success_fields(self):
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_OK_LINE)
        assert rec["marker"] == "resolution_source_fetch"
        assert rec["url"] == "https://www.racetothewh.com/senate/26"
        assert rec["status"] == "ok"
        assert rec["http"] == 200
        # The `none` sentinel harvests as None rather than the string "none".
        assert rec["embeds"] is None
        assert rec["qid"] == 44554
        assert rec["qid_kind"] == "question_id"

    def test_unreadable_embeds_are_captured_on_a_successful_fetch(self):
        """The qids 44554/44556 shape: the fetch legitimately succeeded and the page carried prose, so this
        field is the only thing that makes the missing embedded figures queryable."""
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_EMBED_LINE)
        assert rec["status"] == "ok"
        assert rec["embeds"] == "infogram,tableau"

    def test_no_resolving_content_status(self):
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_NO_CONTENT_LINE)
        assert rec["status"] == "no_resolving_content"
        assert rec["http"] == 200  # the FETCH succeeded; the content did not arrive
        assert rec["embeds"] == "infogram"
        assert rec["reason"] == "embed_shell"
        assert rec["qid"] == 44556

    def test_the_thin_page_reason_separates_the_ungated_population(self):
        """q45088's 127-char SPA tab list: withheld by the same chrome floor with no embed provider anywhere in
        the raw HTML. Without `reason` the two rules are one bucket, and "how often does the floor catch a page
        the embed gate would have published?" stops being answerable from the archive."""
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_THIN_PAGE_LINE)
        assert rec["status"] == "no_resolving_content"
        assert rec["embeds"] is None
        assert rec["reason"] == "thin_page"

    def test_the_no_matching_passage_reason_separates_a_document_from_a_page(self):
        """The third `no_resolving_content` reason, and the only one that is a DOCUMENT we read end to end rather
        than a page we could not read. Without it a withheld PDF is one bucket with the chrome floor's pages, and
        "how often does a cited document not discuss its own question?" stops being answerable from the archive."""
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_NO_MATCHING_PASSAGE_LINE)
        assert rec["status"] == "no_resolving_content"
        assert rec["http"] == 200  # the fetch and the READ both succeeded; the content did not
        assert rec["reason"] == "no_matching_passage"
        assert rec["route"] == "pdf_local"
        assert rec["qid"] == 45363

    def test_lines_without_a_reason_still_parse_and_harvest_it_as_none(self):
        """Back-compat both ways: every archived line predates the field, and a fresh reason-less status omits
        it too. None means "no reason applies", which the absence has to keep meaning."""
        for line in (RESOLUTION_SOURCE_FETCH_OK_LINE, RESOLUTION_SOURCE_FETCH_BLOCKED_LINE):
            assert _parse_one(line).get("reason") is None

    def test_non_success_status_keeps_its_http_code(self):
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_BLOCKED_LINE)
        assert rec["status"] == "blocked"
        assert rec["http"] == 403

    def test_a_fetch_with_no_response_harvests_a_null_http_code(self):
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_NO_RESPONSE_LINE)
        assert rec["status"] == "error"
        assert rec["http"] is None

    def test_a_datawrapper_hop_is_identifiable_by_its_url(self):
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_DATASET_LINE)
        assert rec["url"] == "https://static.dwcdn.net/data/kSCt4.csv"
        assert rec["status"] == "ok"

    def test_route_names_the_ladder_rung_that_produced_the_outcome(self):
        """A rescued page reads `ok` on this marker exactly like an unescalated one, so without `route` "what did
        the ladder buy" is not a query."""
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_ROUTE_ONLY_LINE)
        assert rec["status"] == "ok"
        assert rec["route"] == "wayback"
        # route present with reason absent: a middle-positioned optional group would swallow route into reason.
        assert rec["reason"] is None

    def test_both_optional_tails_parse_together(self):
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_REASON_AND_ROUTE_LINE)
        assert rec["status"] == "no_resolving_content"
        assert rec["reason"] == "embed_shell"
        assert rec["route"] == "rendered"

    def test_lines_without_a_route_still_parse_and_harvest_it_as_none(self):
        """Every archived line predates the field, and a direct fetch on a provider not yet taught the ladder
        omits it too."""
        for line in (
            RESOLUTION_SOURCE_FETCH_OK_LINE,
            RESOLUTION_SOURCE_FETCH_NO_CONTENT_LINE,
            RESOLUTION_SOURCE_FETCH_DATASET_LINE,
        ):
            assert _parse_one(line).get("route") is None

    def test_an_http_failure_carries_its_class_and_server(self):
        """The egress-vs-host measurement: a 403 with the CDN that served it, so the archive can ask "how often
        is a cited host giving the runner IP a 403 from Akamai"."""
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_FAILURE_CLASS_LINE)
        assert rec["status"] == "blocked"
        assert rec["failure_class"] == "http_403"
        assert rec["server"] == "akamaighost"
        # No transport exception on an HTTP response, so `exc` stays absent.
        assert rec["exc"] is None

    def test_a_transport_error_carries_its_class_and_exception(self):
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_TRANSPORT_ERROR_LINE)
        assert rec["status"] == "error"
        assert rec["http"] is None
        assert rec["failure_class"] == "timeout"
        assert rec["exc"] == "ServerTimeoutError"
        assert rec["server"] is None

    def test_lines_without_the_failure_fields_harvest_them_as_none(self):
        """Additive at the tail: every archived line and every success predates them."""
        for line in (RESOLUTION_SOURCE_FETCH_OK_LINE, RESOLUTION_SOURCE_FETCH_BLOCKED_LINE):
            rec = _parse_one(line)
            assert rec.get("failure_class") is None
            assert rec.get("exc") is None
            assert rec.get("server") is None

    def test_the_page_digest_counters_parse_as_ints_and_a_bool(self):
        """`passages_returned` minus `passages_grounded` is the extractor's fabrication count for that page."""
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_DIGEST_LINE)
        assert rec["status"] == "ok"
        assert rec["route"] == "direct"
        assert rec["passages_returned"] == 6
        assert rec["passages_grounded"] == 4
        assert rec["fallback_used"] is False

    def test_a_digest_fallback_parses_without_a_route(self):
        """The three sit after every earlier optional group, so a line carrying them and no route cannot mis-claim."""
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_DIGEST_FALLBACK_LINE)
        assert rec["passages_returned"] == 0
        assert rec["passages_grounded"] == 0
        assert rec["fallback_used"] is True
        assert rec["route"] is None
        assert rec["reason"] is None

    def test_lines_without_the_digest_fields_harvest_them_as_none(self):
        """A page under the digest threshold, and every archived line, carry none of the three."""
        for line in (RESOLUTION_SOURCE_FETCH_OK_LINE, RESOLUTION_SOURCE_FETCH_FAILURE_CLASS_LINE):
            rec = _parse_one(line)
            assert rec.get("passages_returned") is None
            assert rec.get("passages_grounded") is None
            assert rec.get("fallback_used") is None

    def test_every_optional_tail_group_parses_together_in_the_documented_order(self):
        """The spec ships ahead of its emitter, so this line is the order the emitter must follow: a digest
        group written before `server` would parse, silently, with `server` harvested as None."""
        rec = _parse_one(RESOLUTION_SOURCE_FETCH_FULL_TAIL_LINE)
        assert rec["reason"] == "thin_page"
        assert rec["route"] == "rendered"
        assert rec["failure_class"] == "http_5xx"
        assert rec["exc"] == "ServerTimeoutError"
        assert rec["server"] == "akamaighost"
        assert rec["passages_returned"] == 3
        assert rec["passages_grounded"] == 2
        assert rec["fallback_used"] is False


# Verbatim from resolution_source.py; one line per ESCALATED rung tried after the direct route failed to read the page.
RESOLUTION_SOURCE_ESCALATION_RESCUED_LINE = (
    PFX + "RESOLUTION_SOURCE_ESCALATION: question=44556 url=https://tracker.example.com/senate "
    "from_status=js_wall rung=rendered outcome=success wall_s=12.44 caller=resolution_source"
)
RESOLUTION_SOURCE_ESCALATION_FAILED_LINE = (
    PFX + "RESOLUTION_SOURCE_ESCALATION: question=44211 url=https://www.cbp.gov/newsroom/stats "
    "from_status=blocked rung=impersonate outcome=blocked wall_s=3.07 caller=resolution_source"
)


class TestResolutionSourceEscalation:
    """Per-rung escalation attempts, harvested.

    A rung that fires often and rescues nothing has to be distinguishable from one that never
    fires at all, and `wall_s` is what decides whether a rung earns its latency on a question
    running under a close-derived time budget.
    """

    def test_fields(self):
        rec = _parse_one(RESOLUTION_SOURCE_ESCALATION_RESCUED_LINE)
        assert rec["marker"] == "resolution_source_escalation"
        assert rec["url"] == "https://tracker.example.com/senate"
        # The verbatim FetchStatus that triggered escalation, so the trigger population needs no join back.
        assert rec["from_status"] == "js_wall"
        assert rec["rung"] == "rendered"
        assert rec["outcome"] == "success"
        assert rec["wall_s"] == 12.44

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(RESOLUTION_SOURCE_ESCALATION_RESCUED_LINE)
        # resolution_source.py emits question.id_of_question, same as its fetch sibling.
        assert rec["qid"] == 44556
        assert rec["qid_kind"] == "question_id"

    def test_a_rung_that_rescued_nothing_is_still_recorded(self):
        rec = _parse_one(RESOLUTION_SOURCE_ESCALATION_FAILED_LINE)
        assert rec["rung"] == "impersonate"
        assert rec["outcome"] == "blocked"
        assert rec["wall_s"] == 3.07

    def test_does_not_collide_with_the_fetch_marker(self):
        """Both tokens start RESOLUTION_SOURCE_, and resolution_source_fetch sits EARLIER in MARKER_SPECS, so
        under the one-marker-per-line break a loose prefix match there would have swallowed every escalation
        line."""
        harvested = parse_log_text(
            RESOLUTION_SOURCE_ESCALATION_RESCUED_LINE + "\n" + RESOLUTION_SOURCE_FETCH_OK_LINE + "\n", **_META
        )
        assert len(harvested["resolution_source_escalation"]) == 1
        assert len(harvested["resolution_source_fetch"]) == 1


# Verbatim from research/resolution_source.py:_url_context_admission and _url_context_rung, registered 2026-09-04.
RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP_LINE = (
    PFX + "RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP: url=https://tracker.example.com/senate host=tracker.example.com"
)
RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_LINE = (
    PFX_WARN + "RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url=https://tracker.example.com/senate "
    "statuses=URL_RETRIEVAL_STATUS_ERROR"
)
RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_NO_STATUSES_LINE = (
    PFX_WARN + "RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url=https://tracker.example.com/senate "
    "statuses=none"
)
RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED_LINE = (
    PFX_WARN + "RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED: url=https://tracker.example.com/senate "
    "host=tracker.example.com"
)


class TestResolutionSourceUrlContextRobotsSkip:
    """The resolution-source ladder's paid read, refused by the free robots pre-check.

    Each record is a paid call NOT billed, and the rate against the handful of hosts publishing
    the Google-Extended directive is what says whether the group parser is over-matching and
    withholding reads we could have had.
    """

    def test_fields(self):
        rec = _parse_one(RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP_LINE)
        assert rec["marker"] == "resolution_source_urlcontext_robots_skip"
        assert rec["url"] == "https://tracker.example.com/senate"
        # The verdict is cached and applied per HOST, so the host is the unit a rate is taken over.
        assert rec["host"] == "tracker.example.com"

    def test_no_question_ref_so_a_join_goes_through_the_run(self):
        rec = _parse_one(RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec

    def test_does_not_collide_with_the_gap_fill_twin_or_the_ladder_markers(self):
        """The gap-fill twin shares this token's suffix and the two ladder markers share its prefix, so under
        the one-marker-per-line break each spec has to claim only its own full word."""
        harvested = parse_log_text(
            "\n".join(
                [
                    RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP_LINE,
                    AGENTIC_URLCONTEXT_ROBOTS_SKIP_LINE,
                    RESOLUTION_SOURCE_FETCH_OK_LINE,
                    RESOLUTION_SOURCE_ESCALATION_RESCUED_LINE,
                ]
            )
            + "\n",
            **_META,
        )
        assert len(harvested["resolution_source_urlcontext_robots_skip"]) == 1
        assert len(harvested["agentic_urlcontext_robots_skip"]) == 1
        assert len(harvested["resolution_source_fetch"]) == 1
        assert len(harvested["resolution_source_escalation"]) == 1


class TestResolutionSourceUrlContextUngroundedSuppressed:
    """A paid read discarded for retrieving nothing: money spent on nothing served."""

    def test_fields(self):
        rec = _parse_one(RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_LINE)
        assert rec["marker"] == "resolution_source_urlcontext_ungrounded_suppressed"
        assert rec["url"] == "https://tracker.example.com/senate"
        assert rec["statuses"] == "URL_RETRIEVAL_STATUS_ERROR"

    def test_the_none_sentinel_harvests_as_none(self):
        """`none` is the emitter's word for "the SDK attached no url_metadata at all", which is the no-data
        reading rather than a status called none."""
        assert _parse_one(RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_NO_STATUSES_LINE)["statuses"] is None

    def test_several_statuses_survive_as_one_comma_joined_string(self):
        rec = _parse_one(
            PFX_WARN + "RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url=https://x.test/a "
            "statuses=URL_RETRIEVAL_STATUS_ERROR,URL_RETRIEVAL_STATUS_UNSAFE"
        )
        assert rec["statuses"] == "URL_RETRIEVAL_STATUS_ERROR,URL_RETRIEVAL_STATUS_UNSAFE"

    def test_no_question_ref_so_a_join_goes_through_the_run(self):
        rec = _parse_one(RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec

    def test_does_not_collide_with_the_other_two_ungrounded_markers(self):
        """Three tokens end in UNGROUNDED_SUPPRESSED; each spec must claim only its own line, or the archive
        would count one suppression family as another."""
        harvested = parse_log_text(
            "\n".join(
                [RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_LINE, GEMINI_UNGROUNDED_LINE, AGENTIC_DOCUMENT_UNGROUNDED_LINE]
            )
            + "\n",
            **_META,
        )
        assert len(harvested["resolution_source_urlcontext_ungrounded_suppressed"]) == 1
        assert len(harvested["gemini_ungrounded_suppressed"]) == 1
        assert len(harvested["agentic_document_ungrounded_suppressed"]) == 1


class TestResolutionSourceUrlContextNotAddressed:
    """A paid read that retrieved the page and found nothing on the ask: a true negative, bought."""

    def test_fields(self):
        rec = _parse_one(RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED_LINE)
        assert rec["marker"] == "resolution_source_urlcontext_not_addressed"
        assert rec["url"] == "https://tracker.example.com/senate"
        # The rollout question is which hosts Gemini reaches but finds nothing on.
        assert rec["host"] == "tracker.example.com"

    def test_no_question_ref_so_a_join_goes_through_the_run(self):
        rec = _parse_one(RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec

    def test_the_three_paid_rung_lines_are_told_apart(self):
        harvested = parse_log_text(
            "\n".join(
                [
                    RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP_LINE,
                    RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_LINE,
                    RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED_LINE,
                ]
            )
            + "\n",
            **_META,
        )
        assert len(harvested["resolution_source_urlcontext_robots_skip"]) == 1
        assert len(harvested["resolution_source_urlcontext_ungrounded_suppressed"]) == 1
        assert len(harvested["resolution_source_urlcontext_not_addressed"]) == 1


# Verbatim from rendered_fetch.py:render_page's RenderOffHost boundary, new with the landing-host check on 2026-09-04.
RENDERED_FETCH_OFF_HOST_LINE = (
    PFX_WARN + "RENDERED_FETCH_OFF_HOST: scope=resolution_source pinned_host=dashboard.example.com "
    "landed_host=internal.example.net same_publisher=false"
)
RENDERED_FETCH_OFF_HOST_GAP_FILL_LINE = (
    PFX_WARN + "RENDERED_FETCH_OFF_HOST: scope=gap_fill_v2 pinned_host=dashboard.example.com "
    "landed_host=169.254.169.254 same_publisher=false"
)
# A landing with no hostname (empty authority or a non-http(s) scheme): urlparse hostname is None, rendered as no-data.
RENDERED_FETCH_OFF_HOST_NO_HOSTNAME_LINE = (
    PFX_WARN + "RENDERED_FETCH_OFF_HOST: scope=resolution_source pinned_host=dashboard.example.com "
    "landed_host=None same_publisher=false"
)
# A benign hop inside the publisher's own domain, refused by strict hostname equality: prices the strictness.
RENDERED_FETCH_OFF_HOST_SAME_PUBLISHER_LINE = (
    PFX_WARN + "RENDERED_FETCH_OFF_HOST: scope=resolution_source pinned_host=dashboard.example.com "
    "landed_host=www.dashboard.example.com same_publisher=true"
)


class TestRenderedFetchOffHost:
    """The only per-event record that a page sent headless Chromium off its DNS pin.

    The resolution-source caller counts the refusal under ``render_off_host_skips`` in its
    ``details["counts"]``, a per-question total that names neither host, and the gap-fill v2 caller
    keeps no count at all, so without this row a security-relevant event leaves nothing durable.
    """

    def test_fields(self):
        rec = _parse_one(RENDERED_FETCH_OFF_HOST_LINE)
        assert rec["marker"] == "rendered_fetch_off_host"
        # The render transport is shared, so the row has to say which caller asked for it.
        assert rec["scope"] == "resolution_source"
        assert rec["pinned_host"] == "dashboard.example.com"
        # A HOSTNAME, never the landing URL, which can carry a session token or a credential.
        assert rec["landed_host"] == "internal.example.net"
        # Another registrable domain: the security signal.
        assert rec["same_publisher"] is False

    def test_the_gap_fill_callers_render_is_told_apart_by_scope(self):
        rec = _parse_one(RENDERED_FETCH_OFF_HOST_GAP_FILL_LINE)
        assert rec["scope"] == "gap_fill_v2"
        # The IMDS shape: an IP literal is a stranger like any other host.
        assert rec["landed_host"] == "169.254.169.254"
        assert rec["same_publisher"] is False

    def test_a_landing_with_no_hostname_harvests_as_no_data(self):
        rec = _parse_one(RENDERED_FETCH_OFF_HOST_NO_HOSTNAME_LINE)
        assert rec["landed_host"] is None
        assert rec["same_publisher"] is False

    def test_a_benign_hop_inside_the_publisher_prices_the_strictness(self):
        """Strict hostname equality also refuses `example.com` to `www.example.com`; this field is what keeps
        that population from diluting the security signal."""
        rec = _parse_one(RENDERED_FETCH_OFF_HOST_SAME_PUBLISHER_LINE)
        assert rec["landed_host"] == "www.dashboard.example.com"
        assert rec["same_publisher"] is True

    def test_no_question_ref(self):
        """The transport runs per URL with no question in scope, so a join goes through the run id."""
        rec = _parse_one(RENDERED_FETCH_OFF_HOST_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec

    def test_does_not_collide_with_the_resolution_source_ladder_markers(self):
        """The refused render is the rung a resolution-source fetch escalated into, so all three lines show up
        in the same run's logs, and ``parse_log_text`` routes a line to the FIRST spec that matches: each has to
        claim only its own marker word."""
        harvested = parse_log_text(
            "\n".join(
                [
                    RENDERED_FETCH_OFF_HOST_LINE,
                    RESOLUTION_SOURCE_FETCH_OK_LINE,
                    RESOLUTION_SOURCE_ESCALATION_RESCUED_LINE,
                ]
            )
            + "\n",
            **_META,
        )
        assert len(harvested["rendered_fetch_off_host"]) == 1
        assert len(harvested["resolution_source_fetch"]) == 1
        assert len(harvested["resolution_source_escalation"]) == 1


class TestCredit:
    def test_balance(self):
        rec = _parse_one(CREDIT_BALANCE_LINE)
        assert rec["marker"] == "credit_balance"
        assert rec["key"] == "donated"
        assert rec["phase"] == "start"
        assert rec["remaining"] == 123.45
        assert rec["usage"] == 4.16

    def test_balance_skip_line_has_no_balance(self):
        rec = _parse_one(CREDIT_BALANCE_SKIP_LINE)
        assert rec["key"] == "personal"
        assert rec["phase"] == "start"
        assert rec["remaining"] is None
        assert rec["usage"] is None

    def test_donated_disabled_skip_line_has_no_balance(self):
        """A Mantic run: the donated key is never probed, and the archive must still see the run's start phase
        for that key (with no balance) rather than dropping the line."""
        rec = _parse_one(CREDIT_BALANCE_DONATED_DISABLED_LINE)
        assert rec["marker"] == "credit_balance"
        assert rec["key"] == "donated"
        assert rec["phase"] == "start"
        assert rec["remaining"] is None
        assert rec["usage"] is None

    def test_spend(self):
        rec = _parse_one(CREDIT_SPEND_LINE)
        assert rec["marker"] == "credit_spend"
        assert rec["key"] == "donated"
        assert rec["run_delta_usd"] == 3.34
        assert rec["remaining"] == 120.11

    def test_spend_na(self):
        rec = _parse_one(CREDIT_SPEND_NA_LINE)
        assert rec["run_delta_usd"] is None
        assert rec["remaining"] is None

    def test_spend_source_is_captured(self):
        rec = _parse_one(CREDIT_SPEND_REMAINING_SOURCE_LINE)
        assert rec["source"] == "remaining_delta"
        assert rec["run_delta_usd"] == 3.34

    def test_unsettled_zero_is_distinguishable_from_a_real_zero(self):
        """The whole point of source=: a 0.00 from the usage branch means OpenRouter had not settled yet, not
        that this run was free, and the delta alone cannot carry that distinction."""
        rec = _parse_one(CREDIT_SPEND_UNSETTLED_SOURCE_LINE)
        assert rec["run_delta_usd"] == 0.0
        assert rec["source"] == "usage_delta_unsettled"

    def test_pre_field_lines_parse_with_source_none(self):
        """Back-compat: a re-harvest of an older log must not drop the record. None reads correctly as "this run
        predates the field", distinct from any of the three real source values."""
        rec = _parse_one(CREDIT_SPEND_LINE)
        assert rec.get("source") is None

    def test_donated_key_state_drained_is_the_info_shape(self):
        """Verbatim from credit_telemetry.classify_donated_key_state: the one expected verdict."""
        line = (
            PFX + "DONATED_KEY_STATE: state=drained — the donated OpenRouter key spent its whole allocation "
            "with the cap itself intact. Credit-caused personal-key fallbacks are exempt from alerting only "
            "while the dated suppression window is open, i.e. before 2026-09-03; from that date on they "
            "redden CI like any other fallback."
        )
        rec = _parse_one(line)
        assert rec["marker"] == "donated_key_state"
        assert rec["state"] == "drained"
        assert "qid" not in rec

    def test_donated_key_state_other_verdicts_are_the_warning_shape(self):
        for state in ("zeroed", "revoked", "funded", "unknown"):
            line = (
                PFX_WARN + f"DONATED_KEY_STATE: state={state} — a credit-shaped donated-key failure that is NOT an "
                "expected drained wallet (zeroed = cap set to 0, revoked = key rejected, funded = the key still "
                "has money so the failure was not about credit, unknown = the probe could not answer). "
                "Personal-key fallbacks stay alertable, so this run will exit non-zero."
            )
            assert _parse_one(line)["state"] == state

    def test_probe_failed_prose_line_is_not_harvested(self):
        """The logger.exception line before an unknown verdict shares the token but carries no state=."""
        line = (
            "2026-07-17 14:30:00,456 - metaculus_bot.credit_telemetry - ERROR - "
            "DONATED_KEY_STATE: /auth/key probe failed; classifying as unknown (stays alertable)"
        )
        harvested = parse_log_text(line + "\n", **_META)
        assert all(records == [] for records in harvested.values()), harvested

    def test_floor_breach(self):
        rec = _parse_one(CREDIT_FLOOR_BREACH_LINE)
        assert rec["marker"] == "credit_floor_breach"
        assert rec["key"] == "donated"
        assert rec["remaining"] == 45.00
        assert rec["floor"] == 50.00


# Verbatim from credit_telemetry.py:log_role_spend, so a producer-side shape change breaks these loudly.
CREDIT_ROLE_SPEND_LINE = (
    PFX + "CREDIT_ROLE_SPEND: role=forecaster:openai key=donated usd=0.2030 calls=2 costed_calls=2 byok_usd=0.2000"
    " prompt_tokens=104000 completion_tokens=12500 cached_tokens=0 reasoning_tokens=11000"
    " charged_usd=0.2030 byok_calls=2 max_prompt_tokens=52000"
)
CREDIT_ROLE_SPEND_NA_LINE = (
    PFX + "CREDIT_ROLE_SPEND: role=perplexity_research key=direct usd=n/a calls=2 costed_calls=0 byok_usd=n/a"
    " prompt_tokens=0 completion_tokens=0 cached_tokens=0 reasoning_tokens=0 charged_usd=n/a byok_calls=0"
    " max_prompt_tokens=0"
)
# The pre-2026-09-09 shape, still in the archive: neither the token tail nor the charged tail.
CREDIT_ROLE_SPEND_PRE_TOKENS_LINE = (
    PFX + "CREDIT_ROLE_SPEND: role=forecaster:google key=personal usd=1.1433 calls=4 costed_calls=4 byok_usd=0.5716"
)
# The 2026-09-09 morning shape (commits 292b340 and f4fa773): both tails but no max_prompt_tokens yet.
CREDIT_ROLE_SPEND_PRE_MAX_PROMPT_LINE = (
    PFX + "CREDIT_ROLE_SPEND: role=forecaster:openai key=donated usd=0.2030 calls=2 costed_calls=2 byok_usd=0.2000"
    " prompt_tokens=104000 completion_tokens=12500 cached_tokens=0 reasoning_tokens=11000"
    " charged_usd=0.2030 byok_calls=2"
)
CREDIT_ROLE_SPEND_EMPTY_LEDGER_LINE = (
    PFX + "CREDIT_ROLE_SPEND: no successful LLM completions reached the litellm success callback this run"
)


class TestCreditRoleSpend:
    def test_row_fields(self):
        rec = _parse_one(CREDIT_ROLE_SPEND_LINE)
        assert rec["marker"] == "credit_role_spend"
        # The vendor-slot role carries a colon; it must survive as the string it is.
        assert rec["role"] == "forecaster:openai"
        assert rec["key"] == "donated"
        assert rec["usd"] == 0.2030
        assert rec["calls"] == 2
        assert rec["costed_calls"] == 2
        assert rec["byok_usd"] == 0.2000
        assert (rec["prompt_tokens"], rec["completion_tokens"]) == (104000, 12500)
        assert (rec["cached_tokens"], rec["reasoning_tokens"]) == (0, 11000)
        assert (rec["charged_usd"], rec["byok_calls"]) == (0.2030, 2)
        assert rec["max_prompt_tokens"] == 52000

    def test_row_without_max_prompt_tokens_reads_none_for_it(self):
        """The first 2026-09-09 shape carried the token and charged tails but not the packet-size
        maximum; those rows keep harvesting with ``max_prompt_tokens`` None, never a fake zero."""
        rec = _parse_one(CREDIT_ROLE_SPEND_PRE_MAX_PROMPT_LINE)
        assert (rec["charged_usd"], rec["byok_calls"]) == (0.2030, 2)
        assert rec["max_prompt_tokens"] is None

    def test_pre_token_rows_still_parse_with_the_new_fields_absent(self):
        """Both 2026-09-09 tails are optional; the 44 archived rows before them must keep harvesting,
        with the new fields read as None ("this run predates the field"), never a fake zero. This
        row is the double-counted personal-key Google slot: usd is twice byok_usd."""
        rec = _parse_one(CREDIT_ROLE_SPEND_PRE_TOKENS_LINE)
        assert (rec["role"], rec["key"], rec["usd"], rec["byok_usd"]) == (
            "forecaster:google",
            "personal",
            1.1433,
            0.5716,
        )
        assert (rec["calls"], rec["costed_calls"]) == (4, 4)
        for field in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens"):
            assert rec[field] is None, field
        assert rec["charged_usd"] is None
        assert rec["byok_calls"] is None
        assert rec["max_prompt_tokens"] is None

    def test_uncosted_row_reads_none_not_zero(self):
        """``n/a`` is the whole point of ``costed_calls``: the calls happened, the dollars are unknown, and a
        0.0 here would read as "this role is free"."""
        rec = _parse_one(CREDIT_ROLE_SPEND_NA_LINE)
        assert rec["role"] == "perplexity_research"
        assert rec["key"] == "direct"
        assert rec["usd"] is None
        assert rec["byok_usd"] is None
        assert rec["charged_usd"] is None
        assert (rec["calls"], rec["costed_calls"], rec["byok_calls"]) == (2, 0, 0)

    def test_empty_ledger_line_is_not_a_row(self):
        """The no-completions line shares the token so it is greppable, but it must not harvest as a
        (role, key) record."""
        harvested = parse_log_text(CREDIT_ROLE_SPEND_EMPTY_LEDGER_LINE + "\n", **_META)
        assert harvested["credit_role_spend"] == []

    def test_no_question_ref(self):
        """Per-run, per-role: the same qid-less shape as the other credit markers."""
        rec = _parse_one(CREDIT_ROLE_SPEND_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec

    def test_does_not_shadow_the_per_key_spend_marker(self):
        """``CREDIT_SPEND`` and ``CREDIT_ROLE_SPEND`` share a prefix; each line must land in exactly its own
        file."""
        harvested = parse_log_text("\n".join([CREDIT_SPEND_LINE, CREDIT_ROLE_SPEND_LINE]) + "\n", **_META)
        assert len(harvested["credit_spend"]) == 1
        assert len(harvested["credit_role_spend"]) == 1


# Verbatim from credit_telemetry.py:log_run_summary; the ledger folded to cost per question, once per run.
CREDIT_RUN_SUMMARY_LINE = (
    PFX + "CREDIT_RUN_SUMMARY: n_questions=4 charged_usd=8.9201 usd_per_question=2.2300"
    " donated_usd=7.5933 personal_usd=1.3268 prompt_tokens=1240000 cached_tokens=870000 cached_share=0.7016"
    " max_prompt_tokens=41176 max_prompt_role=gap_fill_v2_driver"
)
CREDIT_RUN_SUMMARY_EMPTY_LINE = (
    PFX + "CREDIT_RUN_SUMMARY: n_questions=0 charged_usd=n/a usd_per_question=n/a donated_usd=0.0000"
    " personal_usd=0.0000 prompt_tokens=0 cached_tokens=0 cached_share=n/a max_prompt_tokens=0 max_prompt_role=none"
)


class TestCreditRunSummary:
    def test_fields(self):
        rec = _parse_one(CREDIT_RUN_SUMMARY_LINE)
        assert rec["marker"] == "credit_run_summary"
        assert (rec["n_questions"], rec["charged_usd"], rec["usd_per_question"]) == (4, 8.9201, 2.23)
        assert (rec["donated_usd"], rec["personal_usd"]) == (7.5933, 1.3268)
        assert (rec["prompt_tokens"], rec["cached_tokens"], rec["cached_share"]) == (1240000, 870000, 0.7016)
        # The vendor-slot role form carries a colon in other runs; here the role must survive as a string.
        assert (rec["max_prompt_tokens"], rec["max_prompt_role"]) == (41176, "gap_fill_v2_driver")

    def test_empty_run_reads_unknown_money_and_no_role(self):
        """A run with no completions still leaves the line; its unknowns are None, never zero, and the
        ``none`` role sentinel coerces to None like the ``n/a`` dollars."""
        rec = _parse_one(CREDIT_RUN_SUMMARY_EMPTY_LINE)
        assert rec["n_questions"] == 0
        assert (rec["charged_usd"], rec["usd_per_question"], rec["cached_share"]) == (None, None, None)
        assert (rec["donated_usd"], rec["personal_usd"]) == (0.0, 0.0)
        assert rec["max_prompt_role"] is None

    def test_no_question_ref(self):
        rec = _parse_one(CREDIT_RUN_SUMMARY_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec

    def test_does_not_shadow_the_other_credit_markers(self):
        harvested = parse_log_text(
            "\n".join([CREDIT_RUN_SUMMARY_LINE, CREDIT_ROLE_SPEND_LINE, CREDIT_SPEND_LINE]) + "\n", **_META
        )
        assert len(harvested["credit_run_summary"]) == 1
        assert len(harvested["credit_role_spend"]) == 1
        assert len(harvested["credit_spend"]) == 1


# Verbatim from credit_telemetry.py:_alert_on_oversized_prompt; the v2 driver stamps a question, every other role reads n/a.
PROMPT_SIZE_ALERT_LINE = (
    PFX_WARN + "PROMPT_SIZE_ALERT: role=gap_fill_v2_driver question=https://www.metaculus.com/questions/38975/"
    " prompt_tokens=160000 threshold=150000"
)
PROMPT_SIZE_ALERT_NO_QUESTION_LINE = (
    PFX_WARN + "PROMPT_SIZE_ALERT: role=forecaster:anthropic question=n/a prompt_tokens=512000 threshold=150000"
)


class TestPromptSizeAlert:
    def test_fields_and_post_id_question_ref(self):
        rec = _parse_one(PROMPT_SIZE_ALERT_LINE)
        assert rec["marker"] == "prompt_size_alert"
        assert rec["role"] == "gap_fill_v2_driver"
        assert (rec["prompt_tokens"], rec["threshold"]) == (160000, 150000)
        # Same log_prefix ref as the ghost markers: a Metaculus post id off page_url.
        assert rec["qid"] == 38975
        assert rec["qid_kind"] == "post_id"

    def test_roster_call_without_a_question_keeps_the_field_as_none(self):
        """A roster LLM cannot stamp its question; the emitter writes ``n/a`` rather than dropping
        the field, and the sentinel harvests as None with no qid."""
        rec = _parse_one(PROMPT_SIZE_ALERT_NO_QUESTION_LINE)
        assert rec["role"] == "forecaster:anthropic"
        assert rec["question"] == "n/a"
        assert rec["qid"] is None
        assert rec["prompt_tokens"] == 512000

    def test_does_not_steal_the_credit_markers_beside_it(self):
        harvested = parse_log_text("\n".join([PROMPT_SIZE_ALERT_LINE, CREDIT_ROLE_SPEND_LINE]) + "\n", **_META)
        assert len(harvested["prompt_size_alert"]) == 1
        assert len(harvested["credit_role_spend"]) == 1


# Verbatim from credit_telemetry.py:drain_litellm_callbacks; the only record of why CREDIT_ROLE_SPEND rows under-count.
LITELLM_CALLBACK_DRAIN_TIMEOUT_LINE = (
    PFX_WARN + "LITELLM_CALLBACK_DRAIN_TIMEOUT: litellm's logging worker did not deliver its queued "
    "success callbacks within 10.0s; continuing so the run can finish. The CREDIT_ROLE_SPEND "
    "ledger below may under-count this run's last completions."
)


class TestLitellmCallbackDrainTimeout:
    """The completeness flag on a run's role ledger: a record means those rows are a lower bound.

    At most one line per run, from the drain in ``cli._forecast_with_callback_drain``'s ``finally``.
    """

    def test_fields(self):
        rec = _parse_one(LITELLM_CALLBACK_DRAIN_TIMEOUT_LINE)
        assert rec["marker"] == "litellm_callback_drain_timeout"
        # The bound the run used. Near-constant, so the row's value is its presence.
        assert rec["timeout_s"] == 10.0

    def test_no_question_ref(self):
        """Per-run, like the credit markers it qualifies."""
        rec = _parse_one(LITELLM_CALLBACK_DRAIN_TIMEOUT_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec

    def test_does_not_steal_or_lose_the_credit_spend_markers(self):
        """The WARN names CREDIT_ROLE_SPEND in its prose, the whole reason the emitter chose a distinct prefix,
        so all three lines must land in exactly their own files whichever order the specs sit in."""
        harvested = parse_log_text(
            "\n".join([LITELLM_CALLBACK_DRAIN_TIMEOUT_LINE, CREDIT_ROLE_SPEND_LINE, CREDIT_SPEND_LINE]) + "\n",
            **_META,
        )
        assert len(harvested["litellm_callback_drain_timeout"]) == 1
        assert len(harvested["credit_role_spend"]) == 1
        assert len(harvested["credit_spend"]) == 1


class TestHtmlCommentMarkers:
    def test_stacker_outcome(self):
        rec = _parse_one("<!-- STACKER_OUTCOME=primary -->")
        assert rec["marker"] == "stacker_outcome"
        assert rec["outcome"] == "primary"

    def test_stacker_outcome_skipped_config_off(self):
        """The longer literal must win over its "skipped" prefix in the alternation."""
        rec = _parse_one("<!-- STACKER_OUTCOME=skipped_config_off -->")
        assert rec["marker"] == "stacker_outcome"
        assert rec["outcome"] == "skipped_config_off"

    def test_stacker_skip_reason(self):
        """The additive skip-reason companion: the reason plain "skipped" can't express (single-forecaster skips
        compute no spread). Iterates the comment side's frozenset, the single source of truth, so a reason added
        there is uncoverable-by-omission here: this alternation is a hand-maintained duplicate with no
        import-time assert tying it back, and a dropped bucket is unrecoverable after the 90-day GHA expiry."""
        for reason in sorted(STACKER_SKIP_REASONS):
            rec = _parse_one(f"<!-- STACKER_SKIP_REASON={reason} -->")
            assert rec["marker"] == "stacker_skip_reason", reason
            assert rec["reason"] == reason

    def test_stacker_skip_reason_does_not_collide_with_stacker_outcome(self):
        """One marker per line: a comment tail carries both markers on separate lines, and each line must
        harvest as exactly its own marker."""
        harvested = parse_log_text(
            "<!-- STACKER_OUTCOME=skipped -->\n<!-- STACKER_SKIP_REASON=single_forecaster -->\n",
            **_META,
        )
        assert [r["outcome"] for r in harvested["stacker_outcome"]] == ["skipped"]
        assert [r["reason"] for r in harvested["stacker_skip_reason"]] == ["single_forecaster"]

    def test_tools_used(self):
        rec = _parse_one("<!-- TOOLS_USED=false -->")
        assert rec["marker"] == "tools_used"
        assert rec["value"] is False


class TestQidKindAcrossMarkers:
    """qid_kind names the id space of each marker's ``question`` ref, so a residual
    join can translate a query rather than silently dropping the other-keyed records.
    """

    def test_post_id_markers(self):
        assert _parse_one(GHOST_PRE_LINE)["qid_kind"] == "post_id"
        assert _parse_one(GHOST_PRE_JSON_LINE)["qid_kind"] == "post_id"
        assert _parse_one(GHOST_FORECAST_LINE)["qid_kind"] == "post_id"
        assert _parse_one(GHOST_FORECAST_JSON_LINE)["qid_kind"] == "post_id"

    def test_question_id_markers(self):
        assert _parse_one(OPEN_BOUND_PILING_LINE)["qid_kind"] == "question_id"
        assert _parse_one(CLOSE_MARGIN_LINE)["qid_kind"] == "question_id"

    def test_credit_markers_have_no_qid_kind(self):
        """No `question` ref means no id space, so the record carries neither qid nor qid_kind."""
        rec = _parse_one(CREDIT_SPEND_LINE)
        assert "qid_kind" not in rec
        assert "qid" not in rec

    def test_divergent_question_recovered_by_both_id_forms(self):
        """The real 38880/38195 divergence: EXTRACTION_RUNG carries the QUESTION id (38195), GAP_FILL_V2 the
        POST id (38880), same question. A per-marker grep on one id would miss the other; qid_kind tags each so
        a join can unify them (see tests/test_id_mapping.py::TestMarkerRecordsForQuestion)."""
        extraction = PFX + (
            "EXTRACTION_RUNG: question=38195 model=openai/gpt-5.6-sol qtype=numeric rung=block block_present=True"
        )
        gap_fill = (
            "2026-07-19 06:30:00,000 - metaculus_bot.research.agentic.loop - INFO - "
            "question=https://www.metaculus.com/questions/38880/ GAP_FILL_V2: model=openai/gpt-5.6-terra "
            "steps=7 tool_calls=9 searches=4 fetches=3 rendered=1 reads=2 dup_tool_calls=0 deadline_hit=False "
            "concluded_early=True wall_s=312.44 findings=5 pending_leads=1 lint_rejections=0"
        )
        harvested = parse_log_text(extraction + "\n" + gap_fill + "\n", **_META)
        er = harvested["extraction_rung"][0]
        gf = harvested["gap_fill_v2"][0]
        assert (er["qid"], er["qid_kind"]) == (38195, "question_id")
        assert (gf["qid"], gf["qid_kind"]) == (38880, "post_id")


class TestParseLogText:
    def test_multiple_markers_and_seq(self):
        text = "\n".join(
            [
                EXTRACTION_RUNG_LINE,
                "some unrelated log line - INFO - nothing here",
                GAP_FILL_V2_LINE,
                EXTRACTION_RUNG_LINE,  # a second extraction line -> seq 1
                CREDIT_SPEND_LINE,
            ]
        )
        harvested = parse_log_text(text, **_META)
        assert len(harvested["extraction_rung"]) == 2
        assert [r["seq"] for r in harvested["extraction_rung"]] == [
            0,
            1,
        ]
        assert len(harvested["gap_fill_v2"]) == 1
        assert len(harvested["credit_spend"]) == 1

    def test_noise_only_yields_nothing(self):
        harvested = parse_log_text("just some logs\nno markers at all\n", **_META)
        assert all(len(v) == 0 for v in harvested.values())

    def test_every_spec_has_a_filename_stem(self):
        """Guards the archive layout: one JSONL file per marker type."""
        stems = {spec.name for spec in MARKER_SPECS}
        assert "extraction_rung" in stems
        assert "ghost_forecast" in stems
        assert "credit_balance" in stems
        assert len(stems) == len(MARKER_SPECS), "marker names must be unique (one file per type)"


# Verbatim from drop_telemetry.py:emit_drop_telemetry, so a producer-side shape change breaks these loudly.
FORECASTER_DROPS_LINE = (
    PFX + "FORECASTER_DROPS: total=3 systematic=openrouter/anthropic/claude-opus-4.8 "
    'detail={"openrouter/anthropic/claude-opus-4.8":{"zero_output":2},'
    '"openrouter/google/gemini-3.1-pro-preview":{"timeout_soft_deadline":1}}'
)
FORECASTER_DROPS_CLEAN_LINE = PFX + "FORECASTER_DROPS: total=0 systematic=none detail={}"


class TestForecasterDrops:
    def test_fields(self):
        rec = _parse_one(FORECASTER_DROPS_LINE)
        assert rec["marker"] == "forecaster_drops"
        assert rec["total"] == 3
        # A '/'-laden OpenRouter slug survives the systematic field intact.
        assert rec["systematic"] == "openrouter/anthropic/claude-opus-4.8"
        # Per-run summary: no per-question ref, so no qid space is stamped.
        assert "qid" not in rec
        assert rec["qid_kind"] is None if "qid_kind" in rec else True

    def test_detail_json_round_trips(self):
        rec = _parse_one(FORECASTER_DROPS_LINE)
        # detail is never coerced (raw string), so residual analysis can json.loads it and slugs with slashes survive.
        assert json.loads(rec["detail"]) == {
            "openrouter/anthropic/claude-opus-4.8": {"zero_output": 2},
            "openrouter/google/gemini-3.1-pro-preview": {"timeout_soft_deadline": 1},
        }

    def test_clean_run_line_parses_with_zero_total(self):
        rec = _parse_one(FORECASTER_DROPS_CLEAN_LINE)
        assert rec["total"] == 0
        assert rec["systematic"] is None  # "none" sentinel coerces to None
        assert json.loads(rec["detail"]) == {}


# Verbatim from drop_telemetry.emit_drop_telemetry: one WARN per systematic model, then a prose tail after an em dash.
SYSTEMATIC_FORECASTER_FAILURE_LINE = (
    PFX_WARN + "SYSTEMATIC_FORECASTER_FAILURE: model=openrouter/anthropic/claude-opus-4.8 dropped_on_questions=2 "
    "qids=45085,45163 causes=timeout_soft_deadline:1,zero_output:1 — one model failed across multiple "
    "questions this run (likely a refusal class or routing problem, not a blip); investigate or consider "
    "pulling it from the roster."
)


class TestSystematicForecasterFailure:
    def test_fields(self):
        rec = _parse_one(SYSTEMATIC_FORECASTER_FAILURE_LINE)
        assert rec["marker"] == "systematic_forecaster_failure"
        assert rec["model"] == "openrouter/anthropic/claude-opus-4.8"
        assert rec["dropped_on_questions"] == 2
        # Two or more ids by construction, so the comma keeps the list one string rather than a lone int.
        assert rec["qids"] == "45085,45163"
        assert rec["causes"] == "timeout_soft_deadline:1,zero_output:1"

    def test_per_run_line_carries_no_question_ref(self):
        rec = _parse_one(SYSTEMATIC_FORECASTER_FAILURE_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec

    def test_prose_tail_does_not_leak_into_causes(self):
        rec = _parse_one(SYSTEMATIC_FORECASTER_FAILURE_LINE)
        assert "one model failed" not in rec["causes"]


# Verbatim from forecaster.py:_research_and_make_predictions, the per-question counterpart to FORECASTER_DROPS above.
FORECASTERS_SURVIVED_FULL_LINE = (
    PFX + "FORECASTERS_SURVIVED: question=70002 survived=3/3 models=claude-opus-4.8,gemini-3.1-pro-preview,gpt-5.6-sol"
)
FORECASTERS_SURVIVED_DEGRADED_LINE = PFX + "FORECASTERS_SURVIVED: question=14333 survived=1/3 models=gpt-5.6-sol"
FORECASTERS_SURVIVED_UNKNOWN_LINE = PFX + "FORECASTERS_SURVIVED: question=14333 survived=2/3 models=unknown"


class TestForecastersSurvived:
    def test_full_ensemble_fields(self):
        rec = _parse_one(FORECASTERS_SURVIVED_FULL_LINE)
        assert rec["marker"] == "forecasters_survived"
        assert rec["survived"] == 3
        assert rec["configured"] == 3
        assert rec["models"] == "claude-opus-4.8,gemini-3.1-pro-preview,gpt-5.6-sol"

    def test_question_ref_is_stamped_in_the_question_id_space(self):
        """forecaster.py emits question.id_of_question, not the post id; a residual join has to know which
        space to translate into."""
        rec = _parse_one(FORECASTERS_SURVIVED_FULL_LINE)
        assert rec["qid"] == 70002
        assert rec["qid_kind"] == "question_id"

    def test_degraded_run_is_distinguishable_from_a_full_one(self):
        """The whole reason the marker exists: at a low MIN_FORECASTERS_TO_PUBLISH a 1-of-3 publish exits zero,
        so the archive must be able to tell it apart."""
        rec = _parse_one(FORECASTERS_SURVIVED_DEGRADED_LINE)
        assert rec["survived"] == 1
        assert rec["configured"] == 3
        assert rec["survived"] < rec["configured"]
        assert rec["models"] == "gpt-5.6-sol"

    def test_unknown_models_sentinel_survives_as_a_string(self):
        """ "unknown" is the fallback when no prediction carried a Model: prefix; not a _NONE_SENTINELS member, so
        it must stay a readable string rather than coercing to None (indistinguishable from a missing field)."""
        rec = _parse_one(FORECASTERS_SURVIVED_UNKNOWN_LINE)
        assert rec["models"] == "unknown"


# Verbatim from extreme_call.py:format_extreme_call_markers; one line per extreme-band member of a BINARY question.
EXTREME_CALL_LONE_LINE = (
    PFX + "EXTREME_CALL: question=44874 model=gemini-3.1-pro-preview p=0.0300 side=low lone=true survivors=3"
)
EXTREME_CALL_ACCOMPANIED_LINE = (
    PFX + "EXTREME_CALL: question=44870 model=gpt-5.6-sol p=0.9700 side=high lone=false survivors=3"
)
EXTREME_CALL_SOLO_PUBLISH_LINE = (
    PFX + "EXTREME_CALL: question=44874 model=gemini-3.1-pro-preview p=0.0300 side=low lone=true survivors=1"
)
EXTREME_CALL_UNKNOWN_MODEL_LINE = (
    PFX + "EXTREME_CALL: question=44874 model=unknown p=0.0200 side=low lone=true survivors=2"
)


class TestExtremeCall:
    def test_lone_low_call_fields(self):
        rec = _parse_one(EXTREME_CALL_LONE_LINE)
        assert rec["marker"] == "extreme_call"
        assert rec["model"] == "gemini-3.1-pro-preview"
        assert rec["p"] == 0.03
        assert rec["side"] == "low"
        assert rec["lone"] is True
        assert rec["survivors"] == 3

    def test_question_ref_is_stamped_in_the_question_id_space(self):
        """forecaster.py emits question.id_of_question, the same space as forecasters_survived, so the join to
        the survivor count is free."""
        rec = _parse_one(EXTREME_CALL_LONE_LINE)
        assert rec["qid"] == 44874
        assert rec["qid_kind"] == "question_id"

    def test_accompanied_high_call_is_distinguishable_from_a_lone_one(self):
        """The whole finding lives in this field: lone extremes were right 4 of 9, accompanied ones 21 of 23. A
        lone flag harvested as a string ("true") would still filter, but `rec["lone"] is True` would silently
        select nothing."""
        rec = _parse_one(EXTREME_CALL_ACCOMPANIED_LINE)
        assert rec["lone"] is False
        assert rec["side"] == "high"
        assert rec["p"] == 0.97

    def test_single_survivor_publish_carries_its_survivor_count(self):
        """ "lone" is vacuous when the member WAS the ensemble, so a rate cut has to drop these records;
        survivors=1 is how it finds them."""
        rec = _parse_one(EXTREME_CALL_SOLO_PUBLISH_LINE)
        assert rec["survivors"] == 1
        assert rec["lone"] is True

    def test_unknown_model_sentinel_survives_as_a_string(self):
        """Same sentinel and reason as forecasters_survived's models= field: not in _NONE_SENTINELS, so it must
        not coerce to None."""
        rec = _parse_one(EXTREME_CALL_UNKNOWN_MODEL_LINE)
        assert rec["model"] == "unknown"


# Verbatim from member_forecast.py:format_member_forecast_marker; one line per forecast VALUE, raw and published.
MEMBER_FORECAST_BINARY_LINE = (
    PFX + "MEMBER_FORECAST: question=44874 model=openrouter/google/gemini-3.1-pro-preview role=member qtype=binary "
    "raw=0.005 published=0.02"
)
MEMBER_FORECAST_MC_LINE = (
    PFX + "MEMBER_FORECAST: question=45189 model=openrouter/openai/gpt-5.6-sol role=member qtype=multiple_choice "
    "raw=[0.9,0.005,0.095] published=[0.891,0.01,0.099]"
)
MEMBER_FORECAST_NUMERIC_STACKER_LINE = (
    PFX + "MEMBER_FORECAST: question=45065 model=openrouter/anthropic/claude-opus-4.8 role=stacker qtype=numeric "
    "raw=[[0.025,9.2],[0.05,9.6],[0.5,12.1]] published=[[0.025,9.2],[0.05,9.6],[0.5,12.100000001]]"
)


class TestMemberForecast:
    """The one marker that carries a member's forecast VALUE on every question.

    Before it (2026-09-02) the raw value lived only inside the published comment's fenced
    block, which is middle-trimmed and only present since 2026-05: the clip-threshold
    re-read recovered a raw binary probability for 74 of 451 resolved binaries. The two
    JSON fields are kept VERBATIM (the spec's ``raw_fields``) so a consumer ``json.loads``
    them whatever the type, rather than a binary line coercing to a float while the
    vectors stay strings.
    """

    def test_binary_fields_stay_json_literals(self):
        rec = _parse_one(MEMBER_FORECAST_BINARY_LINE)
        assert rec["marker"] == "member_forecast"
        assert rec["model"] == "openrouter/google/gemini-3.1-pro-preview"
        assert rec["role"] == "member"
        assert rec["qtype"] == "binary"
        assert rec["raw"] == "0.005"
        assert rec["published"] == "0.02"
        assert json.loads(rec["raw"]) == 0.005

    def test_mc_vector_survives_whole(self):
        """Compact JSON carries no whitespace, so `\\S+` takes the whole array and the generic comma-splitting
        key=value pattern never sees it."""
        rec = _parse_one(MEMBER_FORECAST_MC_LINE)
        assert json.loads(rec["raw"]) == [0.9, 0.005, 0.095]
        assert json.loads(rec["published"]) == [0.891, 0.01, 0.099]

    def test_numeric_stacker_pairs_and_role(self):
        rec = _parse_one(MEMBER_FORECAST_NUMERIC_STACKER_LINE)
        assert rec["role"] == "stacker"
        assert rec["qtype"] == "numeric"
        assert json.loads(rec["raw"]) == [[0.025, 9.2], [0.05, 9.6], [0.5, 12.1]]
        assert json.loads(rec["published"])[2] == [0.5, 12.100000001]

    def test_question_ref_is_a_question_id(self):
        """Every emitter passes question.id_of_question, the same space as extraction_rung and
        forecasters_survived, so the per-member join is free."""
        rec = _parse_one(MEMBER_FORECAST_BINARY_LINE)
        assert rec["qid"] == 44874
        assert rec["qid_kind"] == "question_id"

    def test_does_not_collide_with_the_thin_publish_floor_raw_field(self):
        """Both specs spell a field `raw`; the per-spec raw_fields keeps this one verbatim without turning the
        floor marker's float into a string."""
        assert _parse_one(THIN_PUBLISH_FLOOR_LOW_LINE)["raw"] == 0.03


# Verbatim from member_forecast.py:format_member_forecast_marker, 2026-09-08 tail: out-of-range mass, real post 651.
MEMBER_FORECAST_DATE_LINE = (
    PFX + "MEMBER_FORECAST: question=651 model=openrouter/openai/gpt-5.6-sol role=member qtype=date "
    "raw=[[0.01,1789560000.0],[0.5,1789603200.0],[0.99,1789646400.0]] "
    "published=[[0.01,1789560000.0],[0.5,1789603200.0],[0.99,1789646400.0]] oor_low=0.000000 oor_high=0.000000"
)
MEMBER_FORECAST_NUMERIC_TAILS_LINE = (
    PFX + "MEMBER_FORECAST: question=45065 model=openrouter/google/gemini-3.1-pro-preview role=member qtype=numeric "
    "raw=[[0.025,9.2],[0.05,9.6],[0.5,12.1]] published=[[0.025,9.2],[0.05,9.6],[0.5,12.1]] "
    "oor_low=0.000000 oor_high=0.037500"
)

# Verbatim from member_forecast.py:format_numeric_aggregate_marker; the PUBLISHED distribution's size and tail mass.
NUMERIC_AGGREGATE_DATE_LINE = (
    PFX + "NUMERIC_AGGREGATE: question=651 qtype=date cdf_size=13 oor_low=0.000000 oor_high=0.000000"
)
NUMERIC_AGGREGATE_NUMERIC_LINE = (
    PFX + "NUMERIC_AGGREGATE: question=45065 qtype=numeric cdf_size=201 oor_low=0.001000 oor_high=0.037500"
)
# The current shape (3 trailing tail-floor fields): the real post 650 (both bounds open) published at the 5% floor.
NUMERIC_AGGREGATE_FLOORED_LINE = PFX + (
    "NUMERIC_AGGREGATE: question=650 qtype=numeric cdf_size=451 oor_low=0.050000 oor_high=0.050000 "
    "oor_low_raw=0.010000 oor_high_raw=0.010000 tail_floor=0.050000"
)
NUMERIC_AGGREGATE_UNFLOORED_LINE = PFX + (
    "NUMERIC_AGGREGATE: question=45065 qtype=numeric cdf_size=201 oor_low=0.010000 oor_high=0.037500 "
    "oor_low_raw=0.010000 oor_high_raw=0.037500 tail_floor=0.000000"
)


class TestOutOfRangeMassFields:
    """The two additive tail fields (plan B10) and the aggregate marker that carries them.

    The platform scores an out-of-range resolution against a fixed 0.05 reference, and this
    pipeline publishes exactly 1% out of range whenever every percentile sits inside; these
    fields are how the archive answers whether the models already put mass beyond the bounds
    before any mechanical tail floor is considered.
    """

    def test_a_date_member_line_parses_with_both_tails(self):
        rec = _parse_one(MEMBER_FORECAST_DATE_LINE)
        assert rec["marker"] == "member_forecast"
        assert rec["qtype"] == "date"
        assert rec["oor_low"] == 0.0
        assert rec["oor_high"] == 0.0
        # The value axis is epoch seconds; the JSON literal survives whole.
        assert json.loads(rec["raw"])[1] == [0.5, 1789603200.0]

    def test_a_numeric_member_line_parses_with_both_tails(self):
        rec = _parse_one(MEMBER_FORECAST_NUMERIC_TAILS_LINE)
        assert rec["oor_low"] == 0.0
        assert rec["oor_high"] == 0.0375
        assert json.loads(rec["published"]) == [[0.025, 9.2], [0.05, 9.6], [0.5, 12.1]]

    def test_lines_without_the_tail_still_parse_and_read_none(self):
        """Every pre-2026-09-08 line and every binary / MC line: the fields are optional, and a record without
        them says None rather than a measured zero."""
        for line in (MEMBER_FORECAST_BINARY_LINE, MEMBER_FORECAST_MC_LINE, MEMBER_FORECAST_NUMERIC_STACKER_LINE):
            rec = _parse_one(line)
            assert rec["oor_low"] is None
            assert rec["oor_high"] is None

    def test_numeric_aggregate_date_line(self):
        rec = _parse_one(NUMERIC_AGGREGATE_DATE_LINE)
        assert rec["marker"] == "numeric_aggregate"
        assert rec["qtype"] == "date"
        assert rec["cdf_size"] == 13
        assert rec["oor_low"] == 0.0
        assert rec["oor_high"] == 0.0
        assert rec["qid"] == 651
        assert rec["qid_kind"] == "question_id"

    def test_numeric_aggregate_numeric_line(self):
        rec = _parse_one(NUMERIC_AGGREGATE_NUMERIC_LINE)
        assert rec["qtype"] == "numeric"
        assert rec["cdf_size"] == 201
        assert rec["oor_low"] == 0.001
        assert rec["oor_high"] == 0.0375

    def test_a_floored_mantic_aggregate_line_carries_raw_and_published_tails(self):
        rec = _parse_one(NUMERIC_AGGREGATE_FLOORED_LINE)
        assert rec["marker"] == "numeric_aggregate"
        assert rec["qid"] == 650
        assert rec["cdf_size"] == 451
        # oor_low / oor_high keep their meaning: what was PUBLISHED, here the floor itself.
        assert rec["oor_low"] == 0.05
        assert rec["oor_high"] == 0.05
        assert rec["oor_low_raw"] == 0.01
        assert rec["oor_high_raw"] == 0.01
        assert rec["tail_floor"] == 0.05

    def test_an_unfloored_aggregate_line_reads_a_zero_floor_and_equal_tails(self):
        rec = _parse_one(NUMERIC_AGGREGATE_UNFLOORED_LINE)
        assert rec["oor_low"] == rec["oor_low_raw"] == 0.01
        assert rec["oor_high"] == rec["oor_high_raw"] == 0.0375
        assert rec["tail_floor"] == 0.0

    def test_aggregate_lines_that_predate_the_floor_fields_read_none_for_them(self):
        """The three fields are one optional trailing group, so every earlier archived line still harvests, and
        a None there says "not recorded", never a measured zero floor."""
        for line in (NUMERIC_AGGREGATE_DATE_LINE, NUMERIC_AGGREGATE_NUMERIC_LINE):
            rec = _parse_one(line)
            assert rec["marker"] == "numeric_aggregate"
            assert rec["oor_low_raw"] is None
            assert rec["oor_high_raw"] is None
            assert rec["tail_floor"] is None

    def test_the_grid_mismatch_marker_is_not_claimed_by_the_aggregate_spec(self):
        """NUMERIC_AGGREGATE_GRID_MISMATCH shares the prefix; each line must land in its own spec."""
        line = PFX_WARN + (
            "NUMERIC_AGGREGATE_GRID_MISMATCH: question=45065 model_index=1 got_points=201 expected_points=13 — "
            "resampling in cdf-location space before aggregation"
        )
        rec = _parse_one(line)
        assert rec["marker"] == "numeric_aggregate_grid_mismatch"


# Verbatim from metaculus_bot/member_forecast.py: the 2026-09-09 per-bin elicitation fields (Wave C).
MEMBER_FORECAST_PMF_LINE = (
    PFX + "MEMBER_FORECAST: question=651 model=openrouter/openai/gpt-5.6-sol role=member qtype=date "
    "raw=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0] "
    "published=[0.0,0.000833334,0.000833334,0.000833334,0.000833334,0.000833334,0.000833334,0.000833334,"
    "0.000833334,0.990833326,0.000833334,0.000833334,0.000833334,0.0] oor_low=0.000000 oor_high=0.000000 "
    "elicitation=pmf"
)
NUMERIC_AGGREGATE_POOLED_LINE = PFX + (
    "NUMERIC_AGGREGATE: question=651 qtype=date cdf_size=13 oor_low=0.000000 oor_high=0.000000 "
    "oor_low_raw=0.000000 oor_high_raw=0.000000 tail_floor=0.000000 method=mean"
)
NUMERIC_AGGREGATE_MEDIAN_LINE = PFX + (
    "NUMERIC_AGGREGATE: question=650 qtype=numeric cdf_size=451 oor_low=0.050000 oor_high=0.050000 "
    "oor_low_raw=0.010000 oor_high_raw=0.010000 tail_floor=0.050000 method=median"
)
# The ladder's line for a per-bin block: ``qtype=pmf`` is the block type, not the question type.
EXTRACTION_RUNG_PMF_LINE = (
    PFX + "EXTRACTION_RUNG: question=651 model=openrouter/openai/gpt-5.6-sol qtype=pmf rung=block block_present=True"
)


class TestPerBinElicitationFields:
    """The two additive fields of Wave C: ``elicitation`` on MEMBER_FORECAST, ``method`` on NUMERIC_AGGREGATE.

    On a Mantic enumerable grid (31 bins or fewer) each member declares one probability per bin,
    and those members are pooled by the pointwise MEAN of their CDFs instead of the MEDIAN, because
    the median of three sharp CDFs that disagree is the middle member outright. ``elicitation``
    says how to read a member's ``raw`` / ``published``; ``method`` says which rule the published
    aggregate came from, on every numeric and date question. Both are optional trailing groups, so
    every archived line still harvests with the field ``None``. The member line is post 651's
    12-bin closed date grid with a member certain of 2026-09-16 (bin 8), so ``raw`` and
    ``published`` are the 14-entry ``[below, p_0, ..., p_11, above]`` vector.
    """

    def test_a_pmf_member_line_parses_with_its_vectors_tails_and_elicitation(self):
        rec = _parse_one(MEMBER_FORECAST_PMF_LINE)
        assert rec["marker"] == "member_forecast"
        assert rec["qid"] == 651
        assert rec["qtype"] == "date"
        assert rec["elicitation"] == "pmf"
        raw = json.loads(rec["raw"])
        published = json.loads(rec["published"])
        assert len(raw) == len(published) == 14
        assert raw[9] == 1.0
        assert published[9] == 0.990833326
        assert (published[0], published[-1]) == (0.0, 0.0)
        assert rec["oor_low"] == 0.0
        assert rec["oor_high"] == 0.0

    def test_member_lines_without_the_field_read_none(self):
        for line in (
            MEMBER_FORECAST_BINARY_LINE,
            MEMBER_FORECAST_MC_LINE,
            MEMBER_FORECAST_NUMERIC_STACKER_LINE,
            MEMBER_FORECAST_DATE_LINE,
            MEMBER_FORECAST_NUMERIC_TAILS_LINE,
        ):
            rec = _parse_one(line)
            assert rec["marker"] == "member_forecast"
            assert rec["elicitation"] is None

    def test_a_pooled_aggregate_line_names_the_mean(self):
        rec = _parse_one(NUMERIC_AGGREGATE_POOLED_LINE)
        assert rec["marker"] == "numeric_aggregate"
        assert rec["qid"] == 651
        assert rec["method"] == "mean"
        assert rec["tail_floor"] == 0.0

    def test_a_percentile_aggregate_line_names_the_median(self):
        rec = _parse_one(NUMERIC_AGGREGATE_MEDIAN_LINE)
        assert rec["method"] == "median"
        assert rec["tail_floor"] == 0.05
        assert rec["oor_low_raw"] == 0.01

    def test_aggregate_lines_that_predate_the_method_read_none(self):
        for line in (
            NUMERIC_AGGREGATE_DATE_LINE,
            NUMERIC_AGGREGATE_NUMERIC_LINE,
            NUMERIC_AGGREGATE_FLOORED_LINE,
            NUMERIC_AGGREGATE_UNFLOORED_LINE,
        ):
            rec = _parse_one(line)
            assert rec["marker"] == "numeric_aggregate"
            assert rec["method"] is None

    def test_the_pmf_block_type_is_admitted_on_the_extraction_rung_line(self):
        rec = _parse_one(EXTRACTION_RUNG_PMF_LINE)
        assert rec["marker"] == "extraction_rung"
        assert rec["qtype"] == "pmf"
        assert rec["rung"] == "block"
        assert rec["block_present"] is True


# Verbatim from aggregation_pipeline.py:_floor_single_survivor_binary; WARNING only when the lone value moved.
THIN_PUBLISH_FLOOR_LOW_LINE = PFX_WARN + "THIN_PUBLISH_FLOOR: question=44874 raw=0.0300 clamped=0.0500 survivors=1"
THIN_PUBLISH_FLOOR_HIGH_LINE = PFX_WARN + "THIN_PUBLISH_FLOOR: question=44870 raw=0.9700 clamped=0.9500 survivors=1"


class TestThinPublishFloor:
    def test_low_side_fields(self):
        rec = _parse_one(THIN_PUBLISH_FLOOR_LOW_LINE)
        assert rec["marker"] == "thin_publish_floor"
        # raw is the member's declared value; clamped is what was published; both harvest as floats.
        assert rec["raw"] == 0.03
        assert rec["clamped"] == 0.05
        assert rec["survivors"] == 1

    def test_high_side_fields(self):
        rec = _parse_one(THIN_PUBLISH_FLOOR_HIGH_LINE)
        assert rec["raw"] == 0.97
        assert rec["clamped"] == 0.95

    def test_question_ref_is_stamped_in_the_question_id_space(self):
        """aggregation_pipeline.py emits question.id_of_question, the same space as forecasters_survived and
        extreme_call, so the join to the survivor count and to the member's own EXTREME_CALL line is free."""
        rec = _parse_one(THIN_PUBLISH_FLOOR_LOW_LINE)
        assert rec["qid"] == 44874
        assert rec["qid_kind"] == "question_id"

    def test_does_not_collide_with_the_extreme_call_line_it_follows(self):
        """The two markers fire on the same question in the same run (the member's EXTREME_CALL at fan-out, then
        the floor at aggregation), so each must harvest into its own file with its own fields."""
        harvested = parse_log_text(EXTREME_CALL_SOLO_PUBLISH_LINE + "\n" + THIN_PUBLISH_FLOOR_LOW_LINE + "\n", **_META)
        assert [r["p"] for r in harvested["extreme_call"]] == [0.03]
        assert [r["clamped"] for r in harvested["thin_publish_floor"]] == [0.05]


# FORECASTERS_USED is an HTML comment marker (comment/markers.py); the run-log parser has a spec too, like TOOLS_USED.
FORECASTERS_USED_LINE = PFX + "<!-- FORECASTERS_USED=2/3 -->"


class TestForecastersUsed:
    def test_fields(self):
        rec = _parse_one(FORECASTERS_USED_LINE)
        assert rec["marker"] == "forecasters_used"
        assert rec["used"] == 2
        assert rec["configured"] == 3


# Verbatim from degradation_counters.py:format_degradation_summary, the line that decides CI color.
DEGRADATION_COUNTERS_LINE = (
    PFX + "Degradation counters: forecasters_dropped=2, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_failures=1, summarizer_failures=3, gap_fill_v1_errors=2, gap_fill_v2_errors=0, "
    "prediction_market_degraded=0, prediction_market_source_losses=4, provider_degradation=1, "
    "publish_attempt_failures=1, publish_skipped_closed=2, time_budget_fast_path=3, "
    "research_budget_cuts=5"
)
# The shape before the budget-cut counter shipped: ends at time_budget_fast_path (optional-group, like every tail).
DEGRADATION_COUNTERS_PRE_BUDGET_CUT_LINE = (
    PFX + "Degradation counters: forecasters_dropped=2, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_failures=1, summarizer_failures=3, gap_fill_v2_errors=0, "
    "prediction_market_degraded=0, prediction_market_source_losses=4, provider_degradation=1, "
    "publish_attempt_failures=1, publish_skipped_closed=2, time_budget_fast_path=3"
)
# The shape before the time-budget counter shipped: ends at publish_skipped_closed (same optional-group rationale).
DEGRADATION_COUNTERS_NO_BUDGET_TAIL_LINE = (
    PFX + "Degradation counters: forecasters_dropped=2, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_failures=1, summarizer_failures=3, gap_fill_v2_errors=0, "
    "prediction_market_degraded=0, prediction_market_source_losses=4, provider_degradation=1, "
    "publish_attempt_failures=1, publish_skipped_closed=2"
)
# The 2026-08-25-and-earlier shape: ends at publish_attempt_failures (no publish_skipped_closed yet).
DEGRADATION_COUNTERS_NO_SKIP_TAIL_LINE = (
    PFX + "Degradation counters: forecasters_dropped=2, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_failures=1, summarizer_failures=3, gap_fill_v2_errors=0, "
    "prediction_market_degraded=0, prediction_market_source_losses=4, provider_degradation=1, "
    "publish_attempt_failures=1"
)
# The 2026-08-24-and-earlier shape, every archived record to that date: ends at provider_degradation.
DEGRADATION_COUNTERS_NO_PUBLISH_TAIL_LINE = (
    PFX + "Degradation counters: forecasters_dropped=2, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_failures=1, summarizer_failures=3, gap_fill_v2_errors=0, "
    "prediction_market_degraded=0, prediction_market_source_losses=4, provider_degradation=1"
)
# The shape all 290 archived records carry: ends at prediction_market_source_losses, so the tail must stay optional.
DEGRADATION_COUNTERS_NO_PROVIDER_TAIL_LINE = (
    PFX + "Degradation counters: forecasters_dropped=0, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_failures=0, summarizer_failures=0, gap_fill_v2_errors=0, "
    "prediction_market_degraded=0, prediction_market_source_losses=0"
)
# Pre-rename shape (research_provider_timeouts, no summarizer_failures); re-harvested, so the tail is optional.
DEGRADATION_COUNTERS_LEGACY_LINE = (
    PFX + "Degradation counters: forecasters_dropped=0, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_timeouts=5, gap_fill_v2_errors=0, prediction_market_degraded=1"
)


class TestDegradationCounters:
    def test_all_sixteen_current_keys_parse(self):
        rec = _parse_one(DEGRADATION_COUNTERS_LINE)
        assert rec["marker"] == "degradation_counters"
        assert rec["research_budget_cuts"] == 5
        assert rec["time_budget_fast_path"] == 3
        assert rec["publish_skipped_closed"] == 2
        assert rec["forecasters_dropped"] == 2
        assert rec["questions_failed_to_publish"] == 0
        assert rec["gap_fill_v1_errors"] == 2

    def test_pre_budget_cut_line_still_harvests_everything_else(self):
        rec = _parse_one(DEGRADATION_COUNTERS_PRE_BUDGET_CUT_LINE)
        assert rec["time_budget_fast_path"] == 3
        assert "research_budget_cuts" not in rec
        assert rec["stacker_primary_failed"] == 0
        assert rec["stacker_fallback_used"] == 0
        assert rec["stacker_fallback_failed"] == 0
        assert rec["research_provider_failures"] == 1
        assert rec["summarizer_failures"] == 3
        assert rec["gap_fill_v2_errors"] == 0
        assert rec["prediction_market_degraded"] == 0
        assert rec["prediction_market_source_losses"] == 4
        assert rec["provider_degradation"] == 1
        assert rec["publish_attempt_failures"] == 1

    def test_line_without_the_budget_tail_still_harvests_everything_else(self):
        """Every record archived before the time-budget counter shipped ends at
        publish_skipped_closed, and that now-lazy group must still capture its full
        value there rather than handing a digit to backtracking."""
        rec = _parse_one(DEGRADATION_COUNTERS_NO_BUDGET_TAIL_LINE)
        assert rec["marker"] == "degradation_counters"
        assert rec["publish_skipped_closed"] == 2
        assert rec["publish_attempt_failures"] == 1
        # Absent must read as "this era didn't emit it", never as a measured zero.
        assert "time_budget_fast_path" not in rec

    def test_line_without_the_skip_tail_still_harvests_everything_else(self):
        """Every record archived before 2026-08-25 ends at publish_attempt_failures,
        and that now-lazy group must still capture its full value there rather than
        handing a digit to backtracking, while the newest key reads as absent."""
        rec = _parse_one(DEGRADATION_COUNTERS_NO_SKIP_TAIL_LINE)
        assert rec["marker"] == "degradation_counters"
        assert rec["publish_attempt_failures"] == 1
        assert rec["provider_degradation"] == 1
        # Absent must read as "this era didn't emit it", never as a measured zero.
        assert "publish_skipped_closed" not in rec
        assert "time_budget_fast_path" not in rec

    def test_line_without_the_publish_tail_still_harvests_everything_else(self):
        """Every record archived before 2026-08-24 ends at provider_degradation, and
        the lazy provider_degradation group must still capture its full value there
        (not hand a digit to backtracking) while the newer keys read as absent."""
        rec = _parse_one(DEGRADATION_COUNTERS_NO_PUBLISH_TAIL_LINE)
        assert rec["marker"] == "degradation_counters"
        assert rec["provider_degradation"] == 1
        assert rec["prediction_market_source_losses"] == 4
        # Absent must read as "this era didn't emit it", never as a measured zero.
        assert "publish_attempt_failures" not in rec
        assert "publish_skipped_closed" not in rec
        assert "time_budget_fast_path" not in rec

    def test_line_without_the_provider_degradation_tail_still_harvests_everything_else(self):
        """The load-bearing back-compat case. All 290 archived records end at
        prediction_market_source_losses, so the new tail has to be optional-group
        wrapped — a mandatory group would drop every one of them on the next
        replace-by-run re-harvest rather than harvesting the ten counters it carries.
        """
        rec = _parse_one(DEGRADATION_COUNTERS_NO_PROVIDER_TAIL_LINE)
        assert rec["marker"] == "degradation_counters"
        assert rec["forecasters_dropped"] == 0
        assert rec["research_provider_failures"] == 0
        assert rec["summarizer_failures"] == 0
        assert rec["gap_fill_v2_errors"] == 0
        assert rec["prediction_market_degraded"] == 0
        assert rec["prediction_market_source_losses"] == 0
        # Absent must read as "this era didn't emit it", never as a measured zero.
        assert "provider_degradation" not in rec
        assert "publish_attempt_failures" not in rec
        assert "publish_skipped_closed" not in rec
        assert "time_budget_fast_path" not in rec

    def test_per_run_summary_carries_no_question_ref(self):
        rec = _parse_one(DEGRADATION_COUNTERS_LINE)
        # Aggregates a whole run, so there is no id space to stamp.
        assert "qid" not in rec

    def test_pre_rename_line_still_harvests_its_leading_counters(self):
        """The pre-rename keys it shares with today's line must still come through; the renamed/added ones are
        absent rather than dropping the whole record."""
        rec = _parse_one(DEGRADATION_COUNTERS_LEGACY_LINE)
        assert rec["marker"] == "degradation_counters"
        assert rec["forecasters_dropped"] == 0
        assert rec["stacker_fallback_failed"] == 0
        assert rec["research_provider_timeouts"] == 5
        assert rec["gap_fill_v2_errors"] == 0
        assert rec["prediction_market_degraded"] == 1
        # Keys that did not exist pre-rename are ABSENT from the record, not 0 (never a "measured zero").
        assert "research_provider_failures" not in rec
        assert "summarizer_failures" not in rec
        assert "prediction_market_source_losses" not in rec

    def test_a_future_counter_harvests_with_no_spec_change(self):
        """The whole point of the tokenized tail: appending a key to format_degradation_summary must never again
        require a markers.py edit."""
        line = (
            "2026-08-25 12:00:00,000 - metaculus_bot.forecaster - INFO - "
            "Degradation counters: forecasters_dropped=0, some_future_counter=7"
        )
        rec = _parse_one(line)
        assert rec["forecasters_dropped"] == 0
        assert rec["some_future_counter"] == 7


class TestTimeBudgetLoudMarkers:
    """The four budget WARNs docs/operations.md tells the operator to grep. Without
    specs they vanished at the 90-day GHA log expiry; the mid-phase gap-fill cut on
    a non-fast-path question was recoverable from nothing else."""

    def test_time_budget_fast_path_roundtrip(self):
        line = (
            "2026-08-25 12:00:00,000 - metaculus_bot.forecaster - WARNING - "
            "TIME_BUDGET_FAST_PATH: qid=45085 budget=1140s close_time=2026-08-25 13:00:00+00:00; "
            "dropping the slow search providers and gap-fill to protect the prediction POST"
        )
        rec = _parse_one(line)
        assert rec["marker"] == "time_budget_fast_path"
        assert rec["qid"] == 45085
        assert rec["qid_kind"] == "question_id"
        assert rec["budget_s"] == pytest.approx(1140.0)
        assert rec["close_time"] == "2026-08-25 13:00:00+00:00"

    def test_research_phase_deadline_roundtrip(self):
        line = (
            "2026-08-25 12:00:00,000 - metaculus_bot.research.orchestrator - WARNING - "
            "RESEARCH_PHASE_DEADLINE: cancelled 2/6 providers after 570s (gemini_search,native_search)"
        )
        rec = _parse_one(line)
        assert rec["marker"] == "research_phase_deadline"
        assert rec["cancelled"] == 2
        assert rec["total"] == 6
        assert rec["deadline_s"] == pytest.approx(570.0)
        assert rec["providers"] == "gemini_search,native_search"
        # No question ref on this line; attribution lives in provider_results.
        assert "qid" not in rec

    def test_gap_fill_skipped_for_budget_roundtrip(self):
        line = (
            "2026-08-25 12:00:00,000 - metaculus_bot.research.orchestrator - WARNING - "
            "GAP_FILL_SKIPPED_FOR_BUDGET: question=45085 fast_path=true research_phase_remaining=n/a"
        )
        rec = _parse_one(line)
        assert rec["marker"] == "gap_fill_skipped_for_budget"
        assert rec["qid"] == 45085
        assert rec["fast_path"] is True
        assert rec["research_phase_remaining"] is None  # "n/a" coerces to None

    def test_gap_fill_cut_for_budget_roundtrip_both_passes(self):
        for gap_fill_pass in ("V1", "V2"):
            line = (
                "2026-08-25 12:00:00,000 - metaculus_bot.research.orchestrator - WARNING - "
                f"GAP_FILL_{gap_fill_pass}_CUT_FOR_BUDGET: question=45085; research phase ran out of budget"
            )
            rec = _parse_one(line)
            assert rec["marker"] == "gap_fill_cut_for_budget"
            assert rec["qid"] == 45085
            assert rec["gap_fill_pass"] == gap_fill_pass

    def test_wallclock_abort_roundtrip(self):
        """forecaster.py's shape: two of three forecasters still running; remaining_budget negative once overrun."""
        line = (
            "2026-08-25 12:19:00,000 - metaculus_bot.forecaster - WARNING - "
            "WALLCLOCK_ABORT: qid=45085 elapsed=1140.2s forecasters_completed=1/3 cancelled=2 remaining_budget=-0.2s"
        )
        rec = _parse_one(line)
        assert rec["marker"] == "wallclock_abort"
        assert rec["qid"] == 45085
        assert rec["qid_kind"] == "question_id"
        assert rec["elapsed_s"] == pytest.approx(1140.2)
        assert rec["forecasters_completed"] == 1
        assert rec["forecasters_configured"] == 3
        assert rec["cancelled"] == 2
        assert rec["remaining_budget_s"] == pytest.approx(-0.2)

    def test_stacking_skip_line_with_the_same_token_is_not_harvested(self):
        """stacking_route.py reuses the token on a prose line; its record is the wall_clock_budget STACKER_SKIP_REASON."""
        line = (
            "2026-08-25 12:19:00,000 - metaculus_bot.stacking_route - WARNING - "
            "WALLCLOCK_ABORT: skipping stacking for Q 45085; remaining=12.0s < 60s; forcing fallback_median fallback"
        )
        harvested = parse_log_text(line + "\n", **_META)
        assert all(records == [] for records in harvested.values()), harvested


# Verbatim from provider_health.py:log_provider_degradation_summary; emitted on the healthy zero, recording clean runs.
PROVIDER_DEGRADATION_LINE = (
    PFX + "PROVIDER_DEGRADATION: run=30784152530 findings=2 alertable=2 suppressed=0 "
    'detail=[{"signal":"market_field_contract","venue":"kalshi","questions":1,'
    '"fields":"total_volume,open_interest","pool_rows":40},'
    '{"signal":"catalogue_empty","venue":"predictit_markets","questions":1,"entries":0,"fetch_ok":true}]'
)
PROVIDER_DEGRADATION_CLEAN_LINE = PFX + "PROVIDER_DEGRADATION: run=local findings=0 alertable=0 suppressed=0 detail=[]"
# Current shape (2026-08-24): the observation denominators that tell a measured zero from a vacuous one.
PROVIDER_DEGRADATION_DENOMINATED_LINE = (
    PFX + "PROVIDER_DEGRADATION: run=32300000000 findings=0 alertable=0 suppressed=0 "
    "venues_observed=4 catalogues_observed=2 pool_rows=404 detail=[]"
)
PROVIDER_DEGRADATION_SUPPRESSED_LINE = (
    PFX + "PROVIDER_DEGRADATION: run=local findings=1 alertable=0 suppressed=1 "
    'detail=[{"signal":"market_field_contract","venue":"manifold","questions":1,"fields":"num_bettors",'
    '"pool_rows":12,"suppressed_until":"2026-09-10"}] '
    "(manifold:market_field_contract suppressed until 2026-09-10); run stays green on those."
)


class TestProviderDegradation:
    def test_fields(self):
        rec = _parse_one(PROVIDER_DEGRADATION_LINE)
        assert rec["marker"] == "provider_degradation"
        # A GHA run id coerces to int while the local sentinel stays a str; the archive's ``run_id`` is the join key.
        assert rec["run"] == 30784152530
        assert rec["findings"] == 2
        assert rec["alertable"] == 2
        assert rec["suppressed"] == 0

    def test_local_run_sentinel_stays_a_string(self):
        assert _parse_one(PROVIDER_DEGRADATION_CLEAN_LINE)["run"] == "local"

    def test_detail_json_round_trips(self):
        """``detail`` is a JSON array captured verbatim (it belongs to _RAW_FIELDS
        beside FORECASTER_DROPS' detail): venue and field names are delimiter-hostile
        and residual analysis json.loads it, so coercion would mangle it."""
        rec = _parse_one(PROVIDER_DEGRADATION_LINE)
        payload = json.loads(rec["detail"])
        assert [entry["signal"] for entry in payload] == ["market_field_contract", "catalogue_empty"]
        assert payload[0]["fields"] == "total_volume,open_interest"
        assert payload[1]["venue"] == "predictit_markets"

    def test_clean_run_parses_with_an_empty_detail_array(self):
        """A measured zero is signal, so the marker fires on healthy runs too and the
        parser has to accept the empty array rather than skipping the line."""
        rec = _parse_one(PROVIDER_DEGRADATION_CLEAN_LINE)
        assert rec["findings"] == 0
        assert json.loads(rec["detail"]) == []

    def test_suppressed_run_keeps_its_arithmetic_and_resume_date(self):
        """The suppression clause is free text AFTER the JSON, so ``detail`` must stop
        at the closing bracket instead of swallowing it."""
        rec = _parse_one(PROVIDER_DEGRADATION_SUPPRESSED_LINE)
        assert rec["findings"] == 1
        assert rec["alertable"] == 0
        assert rec["suppressed"] == 1
        assert json.loads(rec["detail"])[0]["suppressed_until"] == "2026-09-10"

    def test_observation_denominators_parse(self):
        rec = _parse_one(PROVIDER_DEGRADATION_DENOMINATED_LINE)
        assert rec["findings"] == 0
        assert rec["venues_observed"] == 4
        assert rec["catalogues_observed"] == 2
        assert rec["pool_rows"] == 404

    def test_pre_denominator_lines_read_absent_not_zero(self):
        """All ~1039 archived lines predate the denominators; on a re-harvest they
        must keep parsing, with the new fields None — a vacuous zero must never be
        promoted into a measured one."""
        for line in (PROVIDER_DEGRADATION_LINE, PROVIDER_DEGRADATION_CLEAN_LINE, PROVIDER_DEGRADATION_SUPPRESSED_LINE):
            rec = _parse_one(line)
            assert rec["venues_observed"] is None
            assert rec["catalogues_observed"] is None
            assert rec["pool_rows"] is None

    def test_per_run_summary_carries_no_question_ref(self):
        rec = _parse_one(PROVIDER_DEGRADATION_LINE)
        assert "qid" not in rec


# Verbatim from fallback_openrouter.py; see docs/telemetry_markers.md "PAID_PERSONAL_KEY_FALLBACK".
PAID_FALLBACK_LINE = (
    PFX_WARN + "PAID PERSONAL-KEY FALLBACK: donated OpenRouter key failed for model=openai/gpt-5.6-sol, "
    "so this call billed to the personal OPENROUTER_API_KEY instead of the free donated key. "
    "Run will complete, then exit non-zero to alert. error=APIError: litellm.APIError: "
    'OpenrouterException - {"error":{"message":"Key limit exceeded (total limit)","code":403}}'
)
PAID_FALLBACK_SUPPRESSED_LINE = (
    PFX_WARN + "PAID PERSONAL-KEY FALLBACK: donated OpenRouter key failed for model=anthropic/claude-opus-4.8, "
    "so this call billed to the personal OPENROUTER_API_KEY instead of the free donated key. "
    "Cause is a credit shortfall, so it is NOT counted as alertable until 2026-09-10 "
    "(operator is self-funding the season). error=APIError: insufficient credit"
)


class TestPaidPersonalKeyFallback:
    def test_model_and_error_are_captured(self):
        rec = _parse_one(PAID_FALLBACK_LINE)
        assert rec["marker"] == "paid_personal_key_fallback"
        assert rec["model"] == "openai/gpt-5.6-sol"
        assert rec["error_type"] == "APIError"
        # ``error`` is the exception's str; its 403 spend-cap phrase separates a drained key from a moderation refusal.
        assert "Key limit exceeded" in rec["error"]

    def test_suppressed_variant_parses_the_same(self):
        """The alert-note clause between the model and the error differs by cause, so
        the spec must not depend on its wording."""
        rec = _parse_one(PAID_FALLBACK_SUPPRESSED_LINE)
        assert rec["model"] == "anthropic/claude-opus-4.8"
        assert rec["error"] == "insufficient credit"

    def test_the_404_variant_is_not_captured_as_this_marker(self):
        """The 404 no-allowed-providers branch logs a DIFFERENT line with no
        ``PAID PERSONAL-KEY FALLBACK`` token, and it means something else (the
        donated key's provider list doesn't cover the model, not a spend problem),
        so it must not be harvested here."""
        line = (
            PFX_WARN + "Donated OpenRouter key returned 404 'no allowed providers' for model=x-ai/grok-4.5; "
            "falling back to general (paid personal) key. error=APIError: 404"
        )
        harvested = parse_log_text(line + "\n", **_META)
        assert harvested["paid_personal_key_fallback"] == []


# Verbatim from publish_hardening.py:_wrap_with_timeout_retry; see docs/telemetry_markers.md "PUBLISH_HARDENING".
PUBLISH_HARDENING_TIMEOUT_LINE = (
    PFX_WARN + "PUBLISH_HARDENING: _post_question_prediction attempt 1/2 timed out after 20s"
)
PUBLISH_HARDENING_FAILED_LINE = (
    PFX_WARN + "PUBLISH_HARDENING: _post_question_prediction attempt 2/2 failed "
    "(HTTPError: Error while posting prediction: Status code: 405. "
    'Response: {"error":"Question 45085 is already closed to forecasting !"})'
)


class TestPublishHardening:
    def test_timeout_shape(self):
        rec = _parse_one(PUBLISH_HARDENING_TIMEOUT_LINE)
        assert rec["marker"] == "publish_hardening"
        assert rec["method"] == "_post_question_prediction"
        assert rec["attempt"] == 1
        assert rec["attempts"] == 2
        assert rec["timeout_s"] == 20
        # Exactly one branch populates per record.
        assert rec["error_type"] is None
        assert rec["error"] is None

    def test_exception_shape_captures_class_and_message(self):
        rec = _parse_one(PUBLISH_HARDENING_FAILED_LINE)
        assert rec["method"] == "_post_question_prediction"
        assert rec["attempt"] == 2
        assert rec["attempts"] == 2
        assert rec["error_type"] == "HTTPError"
        assert "405" in rec["error"]
        assert rec["timeout_s"] is None

    def test_comment_post_method_parses_too(self):
        rec = _parse_one(PFX_WARN + "PUBLISH_HARDENING: post_question_comment attempt 1/2 timed out after 20s")
        assert rec["method"] == "post_question_comment"

    def test_other_publish_hardening_strings_are_not_harvested(self):
        """The module reuses the PUBLISH_HARDENING prefix in its seam-moved
        AttributeErrors and its loop-exited RuntimeError; only the per-attempt
        failure WARNs carry the ``attempt N/M`` clause, so only those harvest."""
        non_attempt_lines = [
            PFX + "Publish hardening applied: 2 MetaculusClient.post_* methods wrapped with 20s timeout + 1 retry",
            PFX_WARN + "PUBLISH_HARDENING: MetaculusClient defines no '_post_question_prediction' to patch. "
            "The forecasting-tools publish seam moved or was renamed; repoint _PATCHED_METHODS.",
            PFX_WARN + "PUBLISH_HARDENING: _post_question_prediction loop exited without running",
        ]
        for line in non_attempt_lines:
            harvested = parse_log_text(line + "\n", **_META)
            assert harvested["publish_hardening"] == [], line

    def test_per_call_marker_carries_no_question_ref(self):
        """The wrapper sees only the POST, so there is no id space to stamp."""
        rec = _parse_one(PUBLISH_HARDENING_TIMEOUT_LINE)
        assert "qid" not in rec

    def test_the_new_not_retrying_line_is_not_harvested_as_an_attempt(self):
        """The non-retryable-4xx WARN shares the prefix but carries no ``attempt N/M``
        clause on purpose — folding it into the line above would have broken the
        attempt spec's anchored shape."""
        line = (
            PFX_WARN + "PUBLISH_HARDENING: _post_question_prediction not retrying status 405 "
            "— a second identical POST cannot succeed"
        )
        assert parse_log_text(line + "\n", **_META)["publish_hardening"] == []


# Verbatim from publish_gate.py:skip_publish_if_closed; the values are q45085's real close, fetch and publish times.
PUBLISH_SKIPPED_CLOSED_LINE = (
    PFX_WARN + "PUBLISH_SKIPPED_CLOSED: question=45085 reason=close_time_passed "
    "close_time=2026-08-03T12:00:00+00:00 now=2026-08-03T12:05:06+00:00 overdue_s=306 state=open"
)


class TestPublishSkippedClosed:
    def test_close_time_passed_shape(self):
        rec = _parse_one(PUBLISH_SKIPPED_CLOSED_LINE)
        assert rec["marker"] == "publish_skipped_closed"
        assert rec["reason"] == "close_time_passed"
        assert rec["close_time"] == "2026-08-03T12:00:00+00:00"
        assert rec["now"] == "2026-08-03T12:05:06+00:00"
        assert rec["overdue_s"] == 306
        assert rec["state"] == "open"

    def test_question_ref_is_stamped_in_the_question_id_space(self):
        """publish_gate emits question.id_of_question, so a residual join must not read it as a post id (the
        two share one integer space)."""
        rec = _parse_one(PUBLISH_SKIPPED_CLOSED_LINE)
        assert rec["qid"] == 45085
        assert rec["qid_kind"] == "question_id"

    def test_state_closed_shape_with_absent_close_time(self):
        rec = _parse_one(
            PFX_WARN + "PUBLISH_SKIPPED_CLOSED: question=45093 reason=state_closed "
            "close_time=n/a now=2026-08-06T09:00:00+00:00 overdue_s=n/a state=resolved"
        )
        assert rec["reason"] == "state_closed"
        # n/a must read as absent, never as a measured zero overdue.
        assert rec["close_time"] is None
        assert rec["overdue_s"] is None

    def test_negative_overdue_parses(self):
        """An early admin close leaves close_time in the future, so overdue is negative."""
        rec = _parse_one(
            PFX_WARN + "PUBLISH_SKIPPED_CLOSED: question=45093 reason=state_closed "
            "close_time=2026-08-07T00:00:00+00:00 now=2026-08-06T09:00:00+00:00 "
            "overdue_s=-54000 state=closed"
        )
        assert rec["overdue_s"] == -54000


# Verbatim from time_budget.py; emitted for EVERY question; see docs/telemetry_markers.md "TIME_BUDGET".
TIME_BUDGET_THIN_LINE = (
    PFX + "TIME_BUDGET: question=45085 budget_s=1140 close_time=2026-08-03T12:00:00+00:00 "
    "close_limited=true fast_path=true"
)
TIME_BUDGET_ROOMY_LINE = (
    PFX + "TIME_BUDGET: question=44870 budget_s=3510 close_time=2026-07-24T15:00:00+00:00 "
    "close_limited=false fast_path=false"
)


# Verbatim from mantic.py:_log_mantic_question; values are the real Preseason 2 posts under tests/data/.
MANTIC_QUESTION_DISCRETE_LINE = (
    PFX + "MANTIC_QUESTION: post=650 question=650 type=discrete cdf_size=451 "
    "multi_resolution=true date_granularity=n/a precision=100.0"
)
MANTIC_QUESTION_BINARY_LINE = (
    PFX + "MANTIC_QUESTION: post=648 question=648 type=binary cdf_size=n/a "
    "multi_resolution=false date_granularity=n/a precision=n/a"
)
MANTIC_QUESTION_DATE_LINE = (
    PFX + "MANTIC_QUESTION: post=651 question=651 type=date cdf_size=13 "
    "multi_resolution=false date_granularity=day precision=n/a"
)
MANTIC_QUESTION_QUANTITATIVE_LINE = (
    PFX + "MANTIC_QUESTION: post=650 question=650 type=quantitative cdf_size=451 "
    "multi_resolution=true date_granularity=n/a precision=100.0"
)


class TestManticQuestion:
    """Per-question record of what Mantic's platform fork sent, harvested because the three
    Mantic-only fields live nowhere else and the wire type is rewritten before parsing."""

    def test_discrete_shape(self):
        rec = _parse_one(MANTIC_QUESTION_DISCRETE_LINE)
        assert rec["marker"] == "mantic_question"
        assert rec["post"] == 650
        assert rec["type"] == "discrete"
        assert rec["cdf_size"] == 451
        assert rec["multi_resolution"] is True
        assert rec["date_granularity"] is None
        assert rec["precision"] == pytest.approx(100.0)

    def test_question_ref_is_stamped_in_the_question_id_space(self):
        """mantic.py emits question.id_of_question as question=; the post id is its own field, so a residual
        join can key on either space without translating."""
        rec = _parse_one(MANTIC_QUESTION_DISCRETE_LINE)
        assert rec["qid"] == 650
        assert rec["qid_kind"] == "question_id"

    def test_absent_fields_read_as_none_not_as_strings(self):
        rec = _parse_one(MANTIC_QUESTION_BINARY_LINE)
        assert rec["type"] == "binary"
        assert rec["cdf_size"] is None
        assert rec["multi_resolution"] is False
        assert rec["date_granularity"] is None
        assert rec["precision"] is None

    def test_date_granularity_survives_as_a_string(self):
        rec = _parse_one(MANTIC_QUESTION_DATE_LINE)
        assert rec["type"] == "date"
        assert rec["cdf_size"] == 13
        assert rec["date_granularity"] == "day"

    def test_the_wire_type_is_kept_verbatim(self):
        """The one record of which questions arrived as Series 2 quantitative and were rewritten to discrete
        before parsing."""
        rec = _parse_one(MANTIC_QUESTION_QUANTITATIVE_LINE)
        assert rec["type"] == "quantitative"
        assert rec["cdf_size"] == 451


# Verbatim from mantic.py:_count_dropped_post, at ERROR level before the error is re-raised into the per-post loop.
_PFX_MANTIC_ERROR = "2026-09-08 14:23:01,123 - metaculus_bot.mantic - ERROR - "
MANTIC_POST_DROPPED_LINE = _PFX_MANTIC_ERROR + "MANTIC_POST_DROPPED: post=650 type=quantitative_v3 error=ValueError"
MANTIC_POST_DROPPED_NO_QUESTION_LINE = _PFX_MANTIC_ERROR + "MANTIC_POST_DROPPED: post=651 type=n/a error=KeyError"


class TestManticPostDropped:
    """A post-level census: the post never became a question, so no qid is stamped."""

    def test_unknown_type_shape(self):
        rec = _parse_one(MANTIC_POST_DROPPED_LINE)
        assert rec["marker"] == "mantic_post_dropped"
        assert rec["post"] == 650
        assert rec["type"] == "quantitative_v3"
        assert rec["error"] == "ValueError"
        assert "qid" not in rec

    def test_a_post_without_a_question_key_harvests_type_as_none(self):
        rec = _parse_one(MANTIC_POST_DROPPED_NO_QUESTION_LINE)
        assert rec["post"] == 651
        assert rec["type"] is None
        assert rec["error"] == "KeyError"


# Verbatim from mantic.py:preflight_mantic_tournaments; INFO normally, WARNING when a new slug appears.
MANTIC_TOURNAMENTS_LINE = (
    "2026-09-08 14:23:01,123 - metaculus_bot.mantic - INFO - "
    "MANTIC_TOURNAMENTS: ongoing=preseason-2 configured=preseason-2 new=none"
)
MANTIC_TOURNAMENTS_NEW_SLUG_LINE = (
    "2026-09-08 14:23:01,123 - metaculus_bot.mantic - WARNING - "
    "MANTIC_TOURNAMENTS: ongoing=preseason-2,series-2 configured=preseason-2 new=series-2"
)


class TestManticTournaments:
    def test_quiet_shape(self):
        rec = _parse_one(MANTIC_TOURNAMENTS_LINE)
        assert rec["marker"] == "mantic_tournaments"
        assert rec["ongoing"] == "preseason-2"
        assert rec["configured"] == "preseason-2"
        assert rec["new"] is None
        # Run-level: no question ref, so no qid or id space is stamped.
        assert "qid" not in rec

    def test_a_new_slug_shape_keeps_several_slugs_as_one_string(self):
        rec = _parse_one(MANTIC_TOURNAMENTS_NEW_SLUG_LINE)
        assert rec["ongoing"] == "preseason-2,series-2"
        assert rec["new"] == "series-2"


# Verbatim from cli.py; the drop term renders only when non-zero; the second line pins its place after donated_key.
RUN_SUMMARY_MANTIC_DROPS_LINE = PFX_WARN + (
    "Run completed with 1 alertable degradation event(s) (bot=0, personal_key_fallback=0 of which "
    "donated_404=0, credit=0, mantic_post_drops=1); exiting non-zero so CI marks this run red."
)
RUN_SUMMARY_DONATED_KEY_AND_MANTIC_DROPS_LINE = PFX_WARN + (
    "Run completed with 2 alertable degradation event(s) (bot=0, personal_key_fallback=1 of which "
    "donated_404=0, credit=0, donated_key=revoked, mantic_post_drops=1); exiting non-zero so CI marks this run red."
)


class TestRunAlertableSummaryManticDrops:
    def test_the_drop_term_harvests(self):
        rec = _parse_one(RUN_SUMMARY_MANTIC_DROPS_LINE)
        assert rec["marker"] == "run_alertable_summary"
        assert rec["alertable"] == 1
        assert rec["bot"] == 0
        assert rec["personal_key_fallback"] == 0
        assert rec["credit"] == 0
        assert rec["mantic_post_drops"] == 1
        assert rec["outcome"] is None

    def test_the_drop_term_follows_the_donated_key_clause(self):
        rec = _parse_one(RUN_SUMMARY_DONATED_KEY_AND_MANTIC_DROPS_LINE)
        assert rec["donated_key"] == "revoked"
        assert rec["mantic_post_drops"] == 1

    def test_a_line_without_the_term_harvests_it_as_none(self):
        rec = _parse_one(
            PFX + "Run completed clean with 0 alertable degradation event(s) (bot=0, personal_key_fallback=0 "
            "of which donated_404=0, credit=0); nothing degraded, so this run stays green."
        )
        assert rec["outcome"] == "clean"
        assert rec["mantic_post_drops"] is None


class TestTimeBudget:
    def test_thin_window_shape(self):
        rec = _parse_one(TIME_BUDGET_THIN_LINE)
        assert rec["marker"] == "time_budget"
        assert rec["budget_s"] == 1140
        assert rec["close_time"] == "2026-08-03T12:00:00+00:00"
        assert rec["close_limited"] is True
        assert rec["fast_path"] is True

    def test_roomy_window_shape_reads_the_static_budget(self):
        """The uncensored denominator: a roomy question emits this line too, so a later
        round can measure how often a window is actually thin."""
        rec = _parse_one(TIME_BUDGET_ROOMY_LINE)
        assert rec["budget_s"] == 3510
        assert rec["close_limited"] is False
        assert rec["fast_path"] is False

    def test_absent_close_time_reads_as_none(self):
        rec = _parse_one(
            PFX + "TIME_BUDGET: question=14333 budget_s=3510 close_time=n/a close_limited=false fast_path=false"
        )
        # n/a must read as absent, never as a parsed timestamp or a zero.
        assert rec["close_time"] is None

    def test_question_ref_is_stamped_in_the_question_id_space(self):
        """time_budget emits question.id_of_question, so a residual join must not read it as a post id (the two
        share one integer space)."""
        rec = _parse_one(TIME_BUDGET_THIN_LINE)
        assert rec["qid"] == 45085
        assert rec["qid_kind"] == "question_id"


# Verbatim from cli.py; emitted on BOTH exit paths; see docs/telemetry_markers.md "RUN_ALERTABLE_SUMMARY".
RUN_ALERTABLE_RED_LINE = (
    PFX_WARN + "Run completed with 3 alertable degradation event(s) (bot=2, personal_key_fallback=1 "
    "of which donated_404=1, credit=0); exiting non-zero so CI marks this run red."
)
RUN_ALERTABLE_SUPPRESSED_LINE = (
    PFX + "Run completed with 0 alertable degradation event(s) (bot=0, personal_key_fallback=7 "
    "of which donated_404=0, credit=7 with 7 credit event(s) suppressed until 2026-09-10, "
    "donated_key=drained); every fallback was a suppressed credit event, so this run stays green."
)
# The 2026-08-25 clean shape: such a run logged no line before; see docs/telemetry_markers.md "RUN_ALERTABLE_SUMMARY".
RUN_ALERTABLE_CLEAN_LINE = (
    PFX + "Run completed clean with 0 alertable degradation event(s) (bot=0, personal_key_fallback=0 "
    "of which donated_404=0, credit=0 with 0 credit event(s) suppressed until 2026-09-10); "
    "nothing degraded, so this run stays green."
)


class TestRunAlertableSummary:
    def test_red_run_fields(self):
        rec = _parse_one(RUN_ALERTABLE_RED_LINE)
        assert rec["marker"] == "run_alertable_summary"
        assert rec["alertable"] == 3
        assert rec["bot"] == 2
        assert rec["personal_key_fallback"] == 1
        assert rec["donated_404"] == 1
        assert rec["credit"] == 0
        # No suppression clause and no probe ran, so both are absent: "never probed" must not read as "unknown".
        assert rec["suppressed_credit"] is None
        assert rec["donated_key"] is None
        # A degraded line carries no phrase marker; ``outcome`` is only ever "clean".
        assert rec["outcome"] is None

    def test_suppressed_green_run_carries_the_probe_verdict(self):
        """The shape the drained-donated-key incident produced: alertable=0 with seven
        real fallbacks. The verdict is what tells a reader why nothing was counted."""
        rec = _parse_one(RUN_ALERTABLE_SUPPRESSED_LINE)
        assert rec["alertable"] == 0
        assert rec["personal_key_fallback"] == 7
        assert rec["credit"] == 7
        assert rec["suppressed_credit"] == 7
        assert rec["resume_date"] == "2026-09-10"
        assert rec["donated_key"] == "drained"
        # ``outcome`` is what says "clean"; a degraded line has no phrase marker, so it must stay absent here.
        assert rec["outcome"] is None

    def test_clean_run_is_harvested_and_flagged(self):
        """The all-clear shape harvests as the same marker, distinguishable by
        ``outcome`` rather than by its all-zero fields — a run that lost a question
        also reads all zeros (q45085's shape) and keeps the plain phrase."""
        rec = _parse_one(RUN_ALERTABLE_CLEAN_LINE)
        assert rec["marker"] == "run_alertable_summary"
        assert rec["outcome"] == "clean"
        assert rec["alertable"] == 0
        assert rec["bot"] == 0
        assert rec["personal_key_fallback"] == 0
        assert rec["donated_404"] == 0
        assert rec["credit"] == 0
        assert rec["suppressed_credit"] == 0
        assert rec["donated_key"] is None


# Verbatim from cli.py:_tournament_source, once per --only-posts run; see docs/telemetry_markers.md "ONLY_POSTS".
ONLY_POSTS_LINE = PFX + "ONLY_POSTS: requested=650 matched=650 dropped=3"
ONLY_POSTS_NO_MATCH_LINE = PFX + "ONLY_POSTS: requested=650,999 matched=none dropped=4"


class TestOnlyPosts:
    def test_one_match(self):
        rec = _parse_one(ONLY_POSTS_LINE)
        assert rec["marker"] == "only_posts"
        assert rec["requested"] == 650
        assert rec["matched"] == 650
        assert rec["dropped"] == 3
        # A run-level marker: no question ref, so no qid or id space is stamped.
        assert "qid" not in rec
        assert "qid_kind" not in rec

    def test_no_match(self):
        """Several ids stay one comma-separated string; an empty match harvests as None."""
        rec = _parse_one(ONLY_POSTS_NO_MATCH_LINE)
        assert rec["requested"] == "650,999"
        assert rec["matched"] is None
        assert rec["dropped"] == 4


# Verbatim from forecaster.py:forecast_questions; run-level; see docs/telemetry_markers.md "QUESTION_CAP_FORFEIT".
QUESTION_CAP_FORFEIT_LINE = PFX_WARN + "QUESTION_CAP_FORFEIT: platform=mantic cap=10 total=12 dropped=2 posts=7011,7012"
QUESTION_CAP_FORFEIT_ONE_POST_LINE = (
    PFX_WARN + "QUESTION_CAP_FORFEIT: platform=metaculus cap=10 total=11 dropped=1 posts=38880"
)


class TestQuestionCapForfeit:
    def test_two_forfeits(self):
        rec = _parse_one(QUESTION_CAP_FORFEIT_LINE)
        assert rec["marker"] == "question_cap_forfeit"
        assert rec["platform"] == "mantic"
        assert rec["cap"] == 10
        assert rec["total"] == 12
        assert rec["dropped"] == 2
        assert rec["posts"] == "7011,7012"
        assert "qid" not in rec
        assert "qid_kind" not in rec

    def test_one_forfeit(self):
        """A lone post id coerces to int, as on ONLY_POSTS."""
        rec = _parse_one(QUESTION_CAP_FORFEIT_ONE_POST_LINE)
        assert rec["platform"] == "metaculus"
        assert rec["dropped"] == 1
        assert rec["posts"] == 38880


# Verbatim from forecaster.py:_drop_questions_with_unreadable_forecast_history; see docs/telemetry_markers.md.
SKIP_GUARD_UNREADABLE_LINE = (
    PFX_WARN + "SKIP_GUARD_UNREADABLE: question=70011 post_id=7011 platform=mantic reason=my_forecasts_missing"
)
SKIP_GUARD_UNREADABLE_METACULUS_LINE = (
    PFX_WARN + "SKIP_GUARD_UNREADABLE: question=45465 post_id=45464 platform=metaculus reason=my_forecasts_missing"
)


class TestSkipGuardUnreadable:
    def test_fields(self):
        rec = _parse_one(SKIP_GUARD_UNREADABLE_LINE)
        assert rec["marker"] == "skip_guard_unreadable"
        assert rec["post_id"] == 7011
        assert rec["platform"] == "mantic"
        assert rec["reason"] == "my_forecasts_missing"

    def test_question_ref_is_stamped_in_the_question_id_space(self):
        """forecaster.py emits question.id_of_question beside the post id, so a join translates from question_id."""
        rec = _parse_one(SKIP_GUARD_UNREADABLE_LINE)
        assert rec["qid"] == 70011
        assert rec["qid_kind"] == "question_id"

    def test_metaculus_platform(self):
        rec = _parse_one(SKIP_GUARD_UNREADABLE_METACULUS_LINE)
        assert rec["platform"] == "metaculus"
        assert rec["qid"] == 45465
        assert rec["post_id"] == 45464


# Verbatim from research/gemini_search.py; see docs/telemetry_markers.md "GEMINI_UNGROUNDED_SUPPRESSED".
GEMINI_UNGROUNDED_LINE = PFX_WARN + "GEMINI_UNGROUNDED_SUPPRESSED: question=38195 model=gemini-3.5-flash queries=3"


class TestGeminiUngroundedSuppressed:
    def test_fields(self):
        rec = _parse_one(GEMINI_UNGROUNDED_LINE)
        assert rec["marker"] == "gemini_ungrounded_suppressed"
        assert rec["model"] == "gemini-3.5-flash"
        assert rec["queries"] == 3

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(GEMINI_UNGROUNDED_LINE)
        # gemini_search.py passes question.id_of_question, not the post id.
        assert rec["qid"] == 38195
        assert rec["qid_kind"] == "question_id"

    def test_absent_qid_coerces_to_none(self):
        """qid is Optional at the call site; "None" renders into the line verbatim."""
        rec = _parse_one(PFX_WARN + "GEMINI_UNGROUNDED_SUPPRESSED: question=None model=gemini-3.5-flash queries=0")
        assert rec["qid"] is None
        assert rec["queries"] == 0


# Verbatim from research/gemini_search.py; see docs/telemetry_markers.md "GEMINI_GROUNDING_DENSITY".
GEMINI_GROUNDING_DENSITY_LINE = PFX + "GEMINI_GROUNDING_DENSITY: question=44944 chunks=4 supports=1 chars=3535"


class TestGeminiGroundingDensity:
    def test_fields(self):
        rec = _parse_one(GEMINI_GROUNDING_DENSITY_LINE)
        assert rec["marker"] == "gemini_grounding_density"
        assert rec["chunks"] == 4
        assert rec["supports"] == 1
        assert rec["chars"] == 3535

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(GEMINI_GROUNDING_DENSITY_LINE)
        # gemini_search.py passes question.id_of_question, same as its suppression twin.
        assert rec["qid"] == 44944
        assert rec["qid_kind"] == "question_id"

    def test_absent_qid_coerces_to_none(self):
        """qid is Optional at the call site; "None" renders into the line verbatim."""
        rec = _parse_one(PFX + "GEMINI_GROUNDING_DENSITY: question=None chunks=1 supports=0 chars=812")
        assert rec["qid"] is None
        assert rec["supports"] == 0

    def test_does_not_collide_with_the_suppression_marker(self):
        """Both markers start GEMINI_ and are emitted from the same function; each spec must claim only its own
        line or one of them would be double-counted in the archive."""
        assert _parse_one(GEMINI_GROUNDING_DENSITY_LINE)["marker"] == "gemini_grounding_density"
        assert _parse_one(GEMINI_UNGROUNDED_LINE)["marker"] == "gemini_ungrounded_suppressed"


# Verbatim from gemini_search.py:_check_attributions; see docs/telemetry_markers.md "GEMINI_UNSUPPORTED_ATTRIBUTION".
GEMINI_UNSUPPORTED_ATTRIBUTION_LINE = (
    PFX + "GEMINI_UNSUPPORTED_ATTRIBUTION: question=44953 tagged=2 unsupported=1 groups=1 labels=7"
)


class TestGeminiUnsupportedAttribution:
    def test_fields(self):
        rec = _parse_one(GEMINI_UNSUPPORTED_ATTRIBUTION_LINE)
        assert rec["marker"] == "gemini_unsupported_attribution"
        assert rec["tagged"] == 2
        assert rec["unsupported"] == 1
        assert rec["groups"] == 1
        # The denominator: without it a bare ``unsupported=21`` cannot be told from a thin grounding record.
        assert rec["labels"] == 7

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(GEMINI_UNSUPPORTED_ATTRIBUTION_LINE)
        # gemini_search.py passes question.id_of_question, same as its two siblings.
        assert rec["qid"] == 44953
        assert rec["qid_kind"] == "question_id"

    def test_absent_qid_coerces_to_none(self):
        rec = _parse_one(
            PFX + "GEMINI_UNSUPPORTED_ATTRIBUTION: question=None tagged=21 unsupported=21 groups=14 labels=1"
        )
        assert rec["qid"] is None
        assert rec["unsupported"] == 21

    def test_does_not_collide_with_its_two_gemini_siblings(self):
        """All three start GEMINI_ and two of the three come out of the same function, so each spec must claim
        only its own line or the archive double-counts."""
        assert _parse_one(GEMINI_UNSUPPORTED_ATTRIBUTION_LINE)["marker"] == "gemini_unsupported_attribution"
        assert _parse_one(GEMINI_GROUNDING_DENSITY_LINE)["marker"] == "gemini_grounding_density"
        assert _parse_one(GEMINI_UNGROUNDED_LINE)["marker"] == "gemini_ungrounded_suppressed"


# Verbatim from gemini_usage.py, shared by every Gemini surface; see docs/telemetry_markers.md "GEMINI_USAGE".
GEMINI_USAGE_GROUNDED_LINE = (
    PFX + "GEMINI_USAGE: role=grounded_search model=gemini-3.5-flash prompt_tokens=1420 "
    "tool_use_prompt_tokens=8305 candidates_tokens=2011 thoughts_tokens=944 total_tokens=12680 "
    "search_queries=3 question=44944"
)
GEMINI_USAGE_READ_DOCUMENT_LINE = (
    PFX + "GEMINI_USAGE: role=read_document model=gemini-3.5-flash prompt_tokens=214 "
    "tool_use_prompt_tokens=n/a candidates_tokens=1877 thoughts_tokens=n/a total_tokens=2091 "
    "search_queries=0"
)


class TestGeminiUsage:
    """The Google AI Studio side of a run's spend.

    Grounding is metered against a monthly grounded-prompt allowance per project and billed per
    QUERY on overage, so `search_queries` is the billable unit and any feature that multiplies
    grounded calls re-eats the same pool. Before this marker none of it was in the archive.
    """

    def test_grounded_search_fields(self):
        rec = _parse_one(GEMINI_USAGE_GROUNDED_LINE)
        assert rec["marker"] == "gemini_usage"
        assert rec["role"] == "grounded_search"
        assert rec["model"] == "gemini-3.5-flash"
        assert rec["prompt_tokens"] == 1420
        assert rec["tool_use_prompt_tokens"] == 8305
        assert rec["candidates_tokens"] == 2011
        assert rec["thoughts_tokens"] == 944
        assert rec["total_tokens"] == 12680
        assert rec["search_queries"] == 3

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(GEMINI_USAGE_GROUNDED_LINE)
        # gemini_search.py passes question.id_of_question, same as its density sibling.
        assert rec["qid"] == 44944
        assert rec["qid_kind"] == "question_id"

    def test_read_document_omits_the_question_and_still_parses(self):
        """read_document is a per-URL tool running below the loop's log prefix with no question in scope, so the
        keyed tail group is what lets one spec serve both surfaces."""
        rec = _parse_one(GEMINI_USAGE_READ_DOCUMENT_LINE)
        assert rec["role"] == "read_document"
        assert rec["qid"] is None
        assert rec["prompt_tokens"] == 214
        assert rec["total_tokens"] == 2091

    def test_absent_counts_harvest_as_none_not_as_zero(self):
        """Every usage_metadata field is individually optional on the SDK response, and a count the API never
        reported must not read as a measured zero."""
        rec = _parse_one(GEMINI_USAGE_READ_DOCUMENT_LINE)
        assert rec["tool_use_prompt_tokens"] is None
        assert rec["thoughts_tokens"] is None
        # An absent web_search_queries list IS a count of none; see docs/telemetry_markers.md "GEMINI_USAGE".
        assert rec["search_queries"] == 0

    def test_a_wholly_unreported_usage_block_harvests_all_nulls(self):
        """The n/a path for search_queries specifically: the emitter renders it only when the grounding metadata
        could not be walked, never merely because no search was issued."""
        rec = _parse_one(
            PFX + "GEMINI_USAGE: role=grounded_search model=gemini-3.5-flash prompt_tokens=n/a "
            "tool_use_prompt_tokens=n/a candidates_tokens=n/a thoughts_tokens=n/a total_tokens=n/a "
            "search_queries=n/a question=None"
        )
        assert rec["marker"] == "gemini_usage"
        assert rec["qid"] is None
        assert all(
            rec[field] is None
            for field in (
                "prompt_tokens",
                "tool_use_prompt_tokens",
                "candidates_tokens",
                "thoughts_tokens",
                "total_tokens",
                "search_queries",
            )
        )

    def test_does_not_collide_with_its_gemini_siblings(self):
        """Four markers now start GEMINI_ and three come out of gemini_search.py, so each spec must claim only
        its own line or the archive double-counts."""
        harvested = parse_log_text(
            "\n".join(
                [
                    GEMINI_USAGE_GROUNDED_LINE,
                    GEMINI_GROUNDING_DENSITY_LINE,
                    GEMINI_UNSUPPORTED_ATTRIBUTION_LINE,
                    GEMINI_UNGROUNDED_LINE,
                ]
            )
            + "\n",
            **_META,
        )
        assert len(harvested["gemini_usage"]) == 1
        assert len(harvested["gemini_grounding_density"]) == 1
        assert len(harvested["gemini_unsupported_attribution"]) == 1
        assert len(harvested["gemini_ungrounded_suppressed"]) == 1


# Verbatim from agentic/tools.py:read_document; see docs/telemetry_markers.md "AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED".
AGENTIC_DOCUMENT_UNGROUNDED_LINE = (
    PFX_WARN + "AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED: url=https://example.com/filing.pdf"
)
# The optional `statuses` tail: the SDK's url_context retrieval statuses, or the `none` sentinel when it reported none.
AGENTIC_DOCUMENT_UNGROUNDED_WITH_STATUSES_LINE = (
    PFX_WARN + "AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED: url=https://example.com/filing.pdf "
    "statuses=URL_RETRIEVAL_STATUS_ERROR,URL_RETRIEVAL_STATUS_UNSAFE"
)
AGENTIC_DOCUMENT_UNGROUNDED_NO_STATUSES_LINE = (
    PFX_WARN + "AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED: url=https://example.com/filing.pdf statuses=none"
)


class TestAgenticDocumentUngroundedSuppressed:
    def test_fields(self):
        rec = _parse_one(AGENTIC_DOCUMENT_UNGROUNDED_LINE)
        assert rec["marker"] == "agentic_document_ungrounded_suppressed"
        assert rec["url"] == "https://example.com/filing.pdf"

    def test_statuses_split_a_failed_retrieval_from_no_retrieval_at_all(self):
        """A bare suppression cannot say which happened; the SDK's own status names can."""
        rec = _parse_one(AGENTIC_DOCUMENT_UNGROUNDED_WITH_STATUSES_LINE)
        assert rec["url"] == "https://example.com/filing.pdf"
        assert rec["statuses"] == "URL_RETRIEVAL_STATUS_ERROR,URL_RETRIEVAL_STATUS_UNSAFE"

    def test_the_none_sentinel_and_an_archived_line_both_harvest_as_none(self):
        """Every line the archive already holds predates the field, and `none` means the SDK reported no
        statuses: the same reading, which is why the sentinel is used."""
        for line in (AGENTIC_DOCUMENT_UNGROUNDED_LINE, AGENTIC_DOCUMENT_UNGROUNDED_NO_STATUSES_LINE):
            assert _parse_one(line).get("statuses") is None

    def test_does_not_collide_with_the_gemini_search_marker(self):
        """Both markers end in UNGROUNDED_SUPPRESSED; each spec must claim only its own line or the archive
        would double-count one of them."""
        assert _parse_one(GEMINI_UNGROUNDED_LINE)["marker"] == "gemini_ungrounded_suppressed"
        assert _parse_one(AGENTIC_DOCUMENT_UNGROUNDED_LINE)["marker"] == "agentic_document_ungrounded_suppressed"


# Verbatim from research/targeted.py:run_gap_fill_pass; see docs/telemetry_markers.md "GAP_FILL_ANALYZER_FAILED".
GAP_FILL_ANALYZER_FAILED_LINE = (
    PFX_WARN + "GAP_FILL_ANALYZER_FAILED: question=44912 error=APIError detail=404 model not found"
)


class TestGapFillAnalyzerFailed:
    def test_fields(self):
        rec = _parse_one(GAP_FILL_ANALYZER_FAILED_LINE)
        assert rec["marker"] == "gap_fill_analyzer_failed"
        assert rec["error"] == "APIError"
        # detail holds the exception str, which contains spaces — it must capture to EOL.
        assert rec["detail"] == "404 model not found"

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(GAP_FILL_ANALYZER_FAILED_LINE)
        assert rec["qid"] == 44912
        assert rec["qid_kind"] == "question_id"

    def test_detail_is_optional(self):
        """Keeps older lines (and any future terser form) parseable rather than dropped."""
        rec = _parse_one(PFX_WARN + "GAP_FILL_ANALYZER_FAILED: question=None error=TimeoutError")
        assert rec["marker"] == "gap_fill_analyzer_failed"
        assert rec["qid"] is None
        assert rec["error"] == "TimeoutError"


# Verbatim from research/targeted.py:run_gap_fill_pass; see docs/telemetry_markers.md "GAP_FILL_V1_TRIAGE".
GAP_FILL_V1_TRIAGE_LINE = (
    PFX + "GAP_FILL_V1_TRIAGE: question=44912 listed=4 kept=2 dropped_not_answerable=1 dropped_in_first_pass=0 "
    "dropped_same_need=1 dropped_schema=0 dropped_over_cap=0"
)
# Verbatim from research/gap_fill_stages.py; shares the GAP_FILL_V1_ prefix with the triage marker.
GAP_FILL_V1_CUT_LINE = PFX_WARN + "GAP_FILL_V1_CUT_FOR_BUDGET: question=44912; research phase ran out of budget"


class TestGapFillV1Triage:
    def test_fields(self):
        rec = _parse_one(GAP_FILL_V1_TRIAGE_LINE)
        assert rec["marker"] == "gap_fill_v1_triage"
        assert rec["listed"] == 4
        assert rec["kept"] == 2
        assert rec["dropped_not_answerable"] == 1
        assert rec["dropped_in_first_pass"] == 0
        assert rec["dropped_same_need"] == 1
        assert rec["dropped_schema"] == 0
        assert rec["dropped_over_cap"] == 0

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(GAP_FILL_V1_TRIAGE_LINE)
        assert rec["qid"] == 44912
        assert rec["qid_kind"] == "question_id"

    def test_a_question_with_no_gaps_is_a_record_not_an_absence(self):
        """listed=0 is the analyzer answering and finding nothing; a dead analyzer is GAP_FILL_ANALYZER_FAILED."""
        rec = _parse_one(
            PFX + "GAP_FILL_V1_TRIAGE: question=44912 listed=0 kept=0 dropped_not_answerable=0 "
            "dropped_in_first_pass=0 dropped_same_need=0 dropped_schema=0 dropped_over_cap=0"
        )
        assert rec["marker"] == "gap_fill_v1_triage"
        assert rec["listed"] == 0
        assert rec["kept"] == 0

    def test_does_not_collide_with_the_budget_cut_marker(self):
        """Both tokens start GAP_FILL_V1_; each spec must claim only its own line."""
        assert _parse_one(GAP_FILL_V1_CUT_LINE)["marker"] == "gap_fill_cut_for_budget"
        assert _parse_one(GAP_FILL_V1_TRIAGE_LINE)["marker"] == "gap_fill_v1_triage"


class TestMarkerNotInsideNoqaDirective:
    """HARNESS-SCAN-EXEMPT markers must never sit inside a ``# noqa:`` code list.

    Inside one, ruff owns the comment: RUF100's autofix deletes an unused noqa
    together with everything trailing it (how 26 markers vanished in the
    aug2026 lint expansion), and a noqa whose only "code" is a marker is an
    invalid directive that suppresses nothing. Canonical shapes are
    ``# noqa: <codes>`` followed by a separate ``# HARNESS-SCAN-EXEMPT-<kind>``
    comment, or a plain trailing marker comment with no noqa at all.
    """

    def test_no_noqa_code_list_contains_a_marker(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        noqa_code_list = re.compile(r"#\s*noqa:([^#\n]*)", re.IGNORECASE)
        offenders: list[str] = []
        for dirpath, dirnames, filenames in os.walk(repo_root):
            dirnames[:] = [
                d for d in dirnames if not (d.startswith((".", "scratch", "REFERENCE_COPY")) or d == "node_modules")
            ]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = Path(dirpath) / filename
                for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                    for match in noqa_code_list.finditer(line):
                        tokens = re.split(r"[,\s]+", match.group(1))
                        if any("HARNESS-SCAN-EXEMPT" in token for token in tokens):
                            offenders.append(f"{path.relative_to(repo_root)}:{lineno}: {line.strip()}")
        assert not offenders, (
            "HARNESS-SCAN-EXEMPT marker(s) found inside a noqa code list; move each marker "
            "into its own trailing comment (`# noqa: <codes>  # HARNESS-SCAN-EXEMPT-<kind>  # <reason>`):\n"
            + "\n".join(offenders)
        )


# Verbatim from research/gemini_search.py; see docs/telemetry_markers.md "GEMINI_GROUNDING_RETRY".
GEMINI_GROUNDING_RETRY_LINE = PFX + "GEMINI_GROUNDING_RETRY: question=45571 model=gemini-3.8-flash outcome=grounded"


class TestGeminiGroundingRetry:
    def test_fields(self):
        rec = _parse_one(GEMINI_GROUNDING_RETRY_LINE)
        assert rec["marker"] == "gemini_grounding_retry"
        assert rec["model"] == "gemini-3.8-flash"
        assert rec["outcome"] == "grounded"
        assert rec["qid"] == 45571
        assert rec["qid_kind"] == "question_id"

    @pytest.mark.parametrize("outcome", ["ungrounded", "timeout"])
    def test_every_outcome_parses(self, outcome: str):
        rec = _parse_one(PFX + f"GEMINI_GROUNDING_RETRY: question=1 model=gemini-3.8-flash outcome={outcome}")
        assert rec["outcome"] == outcome
