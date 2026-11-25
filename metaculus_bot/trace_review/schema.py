"""Column name and dtype of every table in the trace-review bundle, declared once.

The builder casts its rows to these mappings before writing and the app's reader casts again after
reading, so a column dropped, added or retyped fails loudly at the boundary instead of rendering as
an empty filter. A null always means "not recorded", never "no" (docs/trace_review.md "How to read
a null").
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype

SCHEMA_VERSION = 1

STRING = "string"
INT = "Int64"
FLOAT = "float64"
BOOL = "boolean"
DATETIME = "datetime64[us, UTC]"
# Python lists in memory, an Arrow list column on disk.
LIST = "list"

ColumnTypes = Mapping[str, str]

_PANDAS_DTYPES: dict[str, ExtensionDtype | np.dtype] = {
    STRING: pd.StringDtype(),
    INT: pd.Int64Dtype(),
    FLOAT: np.dtype("float64"),
    BOOL: pd.BooleanDtype(),
    DATETIME: pd.DatetimeTZDtype(unit="us", tz="UTC"),
}

FORECAST_COLUMNS: ColumnTypes = {
    # Identity and time
    "forecast_key": STRING,
    "platform": STRING,
    "question_id": INT,
    "post_id": INT,
    "title": STRING,
    "url": STRING,
    "tournament": STRING,
    "category": STRING,
    "question_type": STRING,
    "record_origin": STRING,
    "rescored_since_round": BOOL,
    "run_mode": STRING,
    "submitted_at": DATETIME,
    "open_time": DATETIME,
    "scheduled_resolve_time": DATETIME,
    "actual_resolve_time": DATETIME,
    "lead_days": FLOAT,
    "close_time": DATETIME,
    "close_margin_s": FLOAT,
    "close_margin_frac": FLOAT,
    # Question and outcome
    "options": LIST,
    "range_min": FLOAT,
    "range_max": FLOAT,
    "zero_point": FLOAT,
    "open_lower_bound": BOOL,
    "open_upper_bound": BOOL,
    "inbound_outcome_count": INT,
    "scaling_json": STRING,
    "resolution_raw": STRING,
    "resolved_yes": BOOL,
    "resolution_value": FLOAT,
    "resolution_out_of_range": STRING,
    "published_values": LIST,
    "published_prob_yes": FLOAT,
    "prob_on_outcome": FLOAT,
    "published_pit": FLOAT,
    "confident_wrong": BOOL,
    "brier_score": FLOAT,
    "log_score": FLOAT,
    "numeric_log_score": FLOAT,
    "mc_log_score": FLOAT,
    "type_log_score": FLOAT,
    "spot_peer_score": FLOAT,
    "peer_score": FLOAT,
    "spot_baseline_score": FLOAT,
    "baseline_score": FLOAT,
    "coverage": FLOAT,
    "headline_score": FLOAT,
    "headline_score_field": STRING,
    "question_weight": FLOAT,
    "crowd_size": INT,
    # Ensemble and cohorts
    "forecasters_used": INT,
    "forecasters_configured": INT,
    "forecasters_survived": INT,
    "n_members": INT,
    "member_models": LIST,
    "member_spread": FLOAT,
    "any_drop": BOOL,
    "drop_detail_json": STRING,
    "any_extreme_call": BOOL,
    "extraction_not_clean": BOOL,
    "aggregate_method": STRING,
    "stacker_outcome": STRING,
    "stacker_skip_reason": STRING,
    "stacker_verdict": STRING,
    "degraded_run": BOOL,
    "partial_degraded": BOOL,
    "known_bug": BOOL,
    # Research
    "research_source_class": STRING,
    "research_is_trimmed": BOOL,
    "research_chars": INT,
    "research_run_id": STRING,
    "providers_succeeded": LIST,
    "providers_failed": LIST,
    "market_snapshot_present": BOOL,
    "gap_fill_v1_used": BOOL,
    "anchor_present": BOOL,
    "anchor_confidence": STRING,
    "gfv2_present": BOOL,
    "gfv2_loop_ran": BOOL,
    "gfv2_confidence": STRING,
    "gfv2_tool_calls": INT,
    "fast_path": BOOL,
    "budget_s": FLOAT,
    "ghost_present": BOOL,
    "ghost_log_score": FLOAT,
    "ghost_score_delta": FLOAT,
    "ghost_score_delta_abs": FLOAT,
    # Config, cost, references and trait roll-ups
    "config_era": STRING,
    "triple_subera": STRING,
    "triple_subera_fine": STRING,
    "usd_per_question": FLOAT,
    "run_n_questions": INT,
    "telemetry_run_id": STRING,
    "telemetry_run_date": DATETIME,
    "n_telemetry_runs": INT,
    "comment_id": INT,
    "has_comment": BOOL,
    "comment_chars": INT,
    "explanation_detail_status": STRING,
    "n_rationales": INT,
    "rationale_chars_mean": FLOAT,
    "percent_count_mean": FLOAT,
    "hedge_per_1k_mean": FLOAT,
    "base_rate_number_count_mean": FLOAT,
    "status_quo_count_mean": FLOAT,
    "n_market_price_cited": INT,
    "n_market_absent_stated": INT,
    "n_crowd_forecast_cited": INT,
    "n_resolution_ambiguity_flagged": INT,
}

MEMBER_COLUMNS: ColumnTypes = {
    "forecast_key": STRING,
    "platform": STRING,
    "question_type": STRING,
    "model": STRING,
    "model_is_anonymous": BOOL,
    "forecaster_index": INT,
    "role": STRING,
    "value_source": STRING,
    "value_kind": STRING,
    "prob_yes": FLOAT,
    "option_probs_json": STRING,
    "percentiles_json": STRING,
    "center_value": FLOAT,
    "center_dev_from_median": FLOAT,
    "is_farthest_member": BOOL,
    "prob_on_outcome": FLOAT,
    "own_score": FLOAT,
    "score_minus_published": FLOAT,
    "score_censored": BOOL,
    "rankable": BOOL,
    "not_rankable_reason": STRING,
    "raw_value_json": STRING,
    "published_value_json": STRING,
    "clamp_moved": BOOL,
    "oor_low": FLOAT,
    "oor_high": FLOAT,
    "elicitation": STRING,
    "extraction_rung": STRING,
    "block_present": BOOL,
    "extreme_call_side": STRING,
    "has_rationale": BOOL,
    "rationale_chars": INT,
    "percent_count": INT,
    "hedge_count": INT,
    "hedge_per_1k": FLOAT,
    "base_rate_number_count": INT,
    "status_quo_count": INT,
    "resolution_criteria_count": INT,
    "market_price_cited": BOOL,
    "market_absent_stated": BOOL,
    "crowd_forecast_cited": BOOL,
    "resolution_ambiguity_flagged": BOOL,
}

TEXT_COLUMNS: ColumnTypes = {
    "forecast_key": STRING,
    "comment_text": STRING,
    "research_text": STRING,
    "gap_fill_v2_json": STRING,
    "provider_results_json": STRING,
    "provider_diagnostics_block": STRING,
    "raw_research_path": STRING,
}

TELEMETRY_COLUMNS: ColumnTypes = {
    "forecast_key": STRING,
    "marker": STRING,
    "run_id": STRING,
    "run_date": DATETIME,
    "seq": INT,
    "line_ts": STRING,
    "model": STRING,
    "payload_json": STRING,
}


def frame_from_rows(rows: Iterable[Mapping[str, object]], columns: ColumnTypes) -> pd.DataFrame:
    """One table from row dicts that carry exactly the schema's columns, cast to its dtypes.

    A row with a missing or an extra key raises, so a builder function that drifted from the
    schema fails here and not as a silently empty column in the app.
    """
    expected = set(columns)
    materialized = list(rows)
    for row in materialized:
        if set(row) != expected:
            missing, extra = sorted(expected - set(row)), sorted(set(row) - expected)
            raise ValueError(f"row does not match the schema: missing={missing} extra={extra}")
    frame = pd.DataFrame.from_records(materialized, columns=list(columns))
    return cast_frame(frame, columns)


def cast_frame(frame: pd.DataFrame, columns: ColumnTypes) -> pd.DataFrame:
    """Cast every schema column to its dtype; list columns become Python lists (or None)."""
    if list(frame.columns) != list(columns):
        raise ValueError(f"table columns differ from the schema: {sorted(set(frame.columns) ^ set(columns))}")
    cast = {}
    for column, dtype in columns.items():
        values = frame[column]
        if dtype == LIST:
            cast[column] = values.map(lambda items: None if items is None else list(items)).astype(object)
        elif dtype == DATETIME:
            cast[column] = pd.to_datetime(values, utc=True, format="ISO8601").astype(_PANDAS_DTYPES[DATETIME])
        else:
            cast[column] = values.astype(_PANDAS_DTYPES[dtype])
    return pd.DataFrame(cast, index=frame.index)
