"""One bundle row per resolved forecast, built column group by column group.

Every function takes the performance record, the forecast's chosen telemetry run, its verified
research-archive record and its member rows, and returns a small dict; none reads a file. Column
meanings and sources: docs/trace_review.md "Where each column comes from".
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from metaculus_bot.constants import PLATFORM_MANTIC
from metaculus_bot.performance_analysis.analysis import pit_on_grid
from metaculus_bot.performance_analysis.member_replay import is_replayable, replay_solo_score
from metaculus_bot.performance_analysis.platform_scores import (
    SPOT_BASELINE_FIELD,
    baseline_score,
    coverage,
    peer_score,
    ranking_score,
    spot_baseline_score,
    spot_peer_score,
)
from metaculus_bot.performance_analysis.research_tags import research_tags_for_record
from metaculus_bot.performance_analysis.scaling import NUMERIC_TYPES, cdf_and_grid
from metaculus_bot.performance_analysis.stacker_detection import detect_stacker_fired, member_spread
from metaculus_bot.time_utils import parse_iso_utc
from metaculus_bot.trace_review.config import (
    CONFIDENT_WRONG_MAX_PROB,
    CONFIDENT_WRONG_TAIL,
    MANTIC_QUESTION_URL,
    METACULUS_QUESTION_URL,
)
from metaculus_bot.trace_review.member_rows import ROLE_MEMBER, latest_line_by_model
from metaculus_bot.trace_review.sources import RunTelemetry
from metaculus_bot.trace_review.traits import TRAIT_KIND_FLAG, TRAITS

SECONDS_PER_DAY = 86_400.0
OUT_OF_RANGE_RESOLUTIONS: tuple[str, ...] = ("above_upper_bound", "below_lower_bound")
ARTIFACT_SOURCE = "artifact"
PREDICTION_MARKET_PROVIDER = "prediction_market"
PROVIDER_OK = "ok"
# Forecast-level means of these member traits; the flag traits are counted instead.
MEANED_TRAITS: tuple[str, ...] = (
    "rationale_chars",
    "percent_count",
    "hedge_per_1k",
    "base_rate_number_count",
    "status_quo_count",
)
COUNTED_TRAITS: tuple[str, ...] = tuple(trait.name for trait in TRAITS if trait.kind == TRAIT_KIND_FLAG)


@dataclass(frozen=True)
class ForecastInputs:
    """Everything one forecast row is built from."""

    record: dict
    key: str
    platform: str
    run: RunTelemetry
    archive: dict | None
    members: Sequence[dict[str, object]]
    first_marker_dates: dict[str, datetime]


def last_line(lines: Sequence[dict], marker: str) -> dict | None:
    matching = [line for line in lines if line["marker"] == marker]
    return matching[-1] if matching else None


def event_marker_flag(inputs: ForecastInputs, marker: str) -> bool | None:
    """True when the marker fired; False only when the run postdates the marker's first archived line.

    An event marker is logged only when something happens, so its absence is "no" only for a run
    that could have logged it. Every other case is "not recorded".
    """
    if last_line(inputs.run.lines, marker) is not None:
        return True
    first = inputs.first_marker_dates.get(marker)
    if inputs.run.run_date is None or first is None or inputs.run.run_date < first:
        return None
    return False


def _datetime(raw: object) -> datetime | None:
    return parse_iso_utc(raw if isinstance(raw, str) else None)


def identity_and_time(inputs: ForecastInputs) -> dict[str, object]:
    record, metadata = inputs.record, inputs.record.get("metadata") or {}
    close = last_line(inputs.run.lines, "close_margin") or {}
    submitted, resolved = (
        _datetime(record.get("bot_comment_created_at")),
        _datetime(metadata.get("actual_resolve_time")),
    )
    url_template = MANTIC_QUESTION_URL if inputs.platform == PLATFORM_MANTIC else METACULUS_QUESTION_URL
    return {
        "forecast_key": inputs.key,
        "platform": inputs.platform,
        "question_id": record["question_id"],
        "post_id": record["post_id"],
        "title": record.get("title"),
        "url": url_template.format(post_id=record["post_id"]),
        "tournament": record.get("source_tournament"),
        "category": metadata.get("category"),
        "question_type": record["type"],
        "record_origin": record["record_origin"],
        "rescored_since_round": record["rescored_since_round"],
        "run_mode": (inputs.archive or {}).get("run_mode"),
        "submitted_at": submitted,
        "open_time": _datetime(metadata.get("open_time")),
        "scheduled_resolve_time": _datetime(metadata.get("scheduled_resolve_time")),
        "actual_resolve_time": resolved,
        "lead_days": (resolved - submitted).total_seconds() / SECONDS_PER_DAY if submitted and resolved else None,
        "close_time": _datetime(close.get("close_time")),
        "close_margin_s": close.get("margin_s"),
        "close_margin_frac": close.get("margin_frac"),
    }


def type_log_score(record: dict) -> float | None:
    """The log score that belongs to the record's type: base-2 binary, ``50 ln`` numeric and MC."""
    question_type = record["type"]
    if question_type == "binary":
        return record.get("log_score")
    if question_type in NUMERIC_TYPES:
        return record.get("numeric_log_score")
    return record.get("mc_log_score")


def _published_pit(record: dict) -> float | None:
    resolution = record.get("resolution_parsed")
    grid = cdf_and_grid(record) if record["type"] in NUMERIC_TYPES else None
    if grid is None or isinstance(resolution, (str, bool)) or resolution is None:
        return None
    cdf, values = grid
    pit, _ = pit_on_grid(float(resolution), values, cdf, record.get("per_model_numeric_percentiles"))
    return pit


def _prob_on_outcome(record: dict) -> float | None:
    """Probability on what happened; on a numeric question only for an out-of-range resolution."""
    resolution, published = record.get("resolution_parsed"), record.get("our_forecast_values") or []
    question_type = record["type"]
    if question_type == "binary" and isinstance(resolution, bool) and record.get("our_prob_yes") is not None:
        prob_yes = float(record["our_prob_yes"])
        return prob_yes if resolution else 1.0 - prob_yes
    if question_type == "multiple_choice" and isinstance(resolution, str):
        options = record.get("options") or []
        return float(published[options.index(resolution)]) if resolution in options and published else None
    side = _out_of_range_side(record)
    if side is not None and published:
        return float(published[0]) if side == "below_lower_bound" else 1.0 - float(published[-1])
    return None


def _out_of_range_side(record: dict) -> str | None:
    """Which bound a numeric resolution fell past, whether the platform sent the side as a string or
    sent the value itself (q46143 resolved at 109.9M against a 110M open lower bound)."""
    resolution = record.get("resolution_parsed")
    if record["type"] not in NUMERIC_TYPES:
        return None
    if resolution in OUT_OF_RANGE_RESOLUTIONS:
        return resolution
    if not isinstance(resolution, (int, float)) or isinstance(resolution, bool):
        return None
    scaling = record.get("scaling") or {}
    if scaling.get("range_min") is not None and resolution < float(scaling["range_min"]):
        return "below_lower_bound"
    if scaling.get("range_max") is not None and resolution > float(scaling["range_max"]):
        return "above_upper_bound"
    return None


def _confident_wrong(record: dict, prob_on_outcome: float | None, pit: float | None) -> bool | None:
    if record["type"] in NUMERIC_TYPES:
        if pit is not None:
            return pit < CONFIDENT_WRONG_TAIL or pit > 1.0 - CONFIDENT_WRONG_TAIL
        return None if prob_on_outcome is None else prob_on_outcome <= CONFIDENT_WRONG_TAIL
    return None if prob_on_outcome is None else prob_on_outcome <= CONFIDENT_WRONG_MAX_PROB


def _headline(record: dict, platform: str) -> tuple[float | None, str | None]:
    """Metaculus ranks on spot peer (coverage-scaled peer only as a labelled fallback); Mantic Series
    2 ranks on spot baseline."""
    if platform == PLATFORM_MANTIC:
        value = spot_baseline_score(record)
        return value, None if value is None else SPOT_BASELINE_FIELD
    ranked = ranking_score(record)
    return (None, None) if ranked is None else (ranked.value, ranked.field)


def question_and_outcome(inputs: ForecastInputs) -> dict[str, object]:
    record = inputs.record
    scaling, resolution = record.get("scaling") or {}, record.get("resolution_parsed")
    numeric_resolution = (
        resolution if isinstance(resolution, (int, float)) and not isinstance(resolution, bool) else None
    )
    prob_on_outcome, pit = _prob_on_outcome(record), _published_pit(record)
    headline, headline_field = _headline(record, inputs.platform)
    crowd = (record.get("metadata") or {}).get("nr_forecasters")
    return {
        "options": record.get("options"),
        "range_min": scaling.get("range_min"),
        "range_max": scaling.get("range_max"),
        "zero_point": scaling.get("zero_point"),
        "open_lower_bound": record.get("open_lower_bound"),
        "open_upper_bound": record.get("open_upper_bound"),
        "inbound_outcome_count": scaling.get("inbound_outcome_count"),
        "scaling_json": json.dumps(scaling) if scaling else None,
        "resolution_raw": record.get("resolution_raw"),
        "resolved_yes": resolution if isinstance(resolution, bool) else None,
        "resolution_value": numeric_resolution,
        "resolution_out_of_range": _out_of_range_side(record),
        "published_values": record.get("our_forecast_values"),
        "published_prob_yes": record.get("our_prob_yes"),
        "prob_on_outcome": prob_on_outcome,
        "published_pit": pit,
        "confident_wrong": _confident_wrong(record, prob_on_outcome, pit),
        "brier_score": record.get("brier_score"),
        "log_score": record.get("log_score"),
        "numeric_log_score": record.get("numeric_log_score"),
        "mc_log_score": record.get("mc_log_score"),
        "type_log_score": type_log_score(record),
        "spot_peer_score": spot_peer_score(record),
        "peer_score": peer_score(record),
        "spot_baseline_score": spot_baseline_score(record),
        "baseline_score": baseline_score(record),
        "coverage": coverage(record),
        "headline_score": headline,
        "headline_score_field": headline_field,
        "question_weight": record.get("question_weight"),
        # 0 on a pre-2026-08-25 pull means the field was never read, not an empty crowd.
        "crowd_size": crowd or None,
    }


def ensemble_and_cohorts(inputs: ForecastInputs) -> dict[str, object]:
    record, lines = inputs.record, inputs.run.lines
    members = [row for row in inputs.members if row["role"] == ROLE_MEMBER]
    used, configured = record.get("forecasters_used"), record.get("forecasters_configured")
    survived = last_line(lines, "forecasters_survived")
    drops = last_line(inputs.run.run_lines, "forecaster_drops")
    rungs = latest_line_by_model(lines, "extraction_rung")
    aggregate = last_line(lines, "numeric_aggregate")
    return {
        "forecasters_used": used,
        "forecasters_configured": configured,
        "forecasters_survived": survived.get("survived") if survived else None,
        "n_members": len(members),
        "member_models": [str(row["model"]) for row in members],
        "member_spread": member_spread(record),
        "any_drop": used < configured if used is not None and configured is not None else None,
        "drop_detail_json": drops.get("detail") if drops and (drops.get("total") or 0) > 0 else None,
        "any_extreme_call": event_marker_flag(inputs, "extreme_call"),
        "extraction_not_clean": any(line.get("rung") != "block" for line in rungs.values()) if rungs else None,
        "aggregate_method": aggregate.get("method") if aggregate else None,
        "stacker_outcome": record.get("stacker_outcome"),
        "stacker_skip_reason": record.get("stacker_skip_reason"),
        "stacker_verdict": detect_stacker_fired(record),
        "degraded_run": record.get("degraded_run"),
        "partial_degraded": record.get("partial_degraded"),
        "known_bug": record.get("known_bug"),
    }


def _provider_columns(archive: dict | None) -> dict[str, object]:
    """Provider outcomes, from schema-2 artifact records only: a backfill's provider list is
    reconstructed from trimmed text and is not reliable."""
    results = (archive or {}).get("provider_results") if (archive or {}).get("source") == ARTIFACT_SOURCE else None
    if archive is None or not results:
        return {"providers_succeeded": None, "providers_failed": None, "market_snapshot_present": None}
    succeeded = list(archive.get("providers_succeeded") or [])
    attempted = list(archive.get("providers_attempted") or [])
    return {
        "providers_succeeded": succeeded,
        "providers_failed": [name for name in attempted if name not in succeeded],
        "market_snapshot_present": any(
            result.get("name") == PREDICTION_MARKET_PROVIDER and result.get("status") == PROVIDER_OK
            for result in results
        ),
    }


def ghost_value(record: dict, forecast_json: str) -> object:
    """The gap-fill ghost's forecast in the shape ``replay_solo_score`` takes, or None (date, mismatch)."""
    payload = json.loads(forecast_json)
    question_type, ghost_type = record["type"], payload.get("qtype")
    if question_type == "binary" and ghost_type == "binary":
        return float(payload["prob"])
    if question_type == "multiple_choice" and ghost_type == "multiple_choice":
        probs = payload["option_probs"]
        return probs if all(option in probs for option in record.get("options") or []) else None
    if question_type in NUMERIC_TYPES and ghost_type in NUMERIC_TYPES:
        return [[float(label) * 100.0, float(value)] for label, value in payload["declared_percentiles"].items()]
    return None


def _ghost_columns(inputs: ForecastInputs, published_score: float | None) -> dict[str, object]:
    ghost = last_line(inputs.run.lines, "ghost_forecast_json")
    value = ghost_value(inputs.record, ghost["forecast_json"]) if ghost else None
    score = replay_solo_score(inputs.record, value) if value is not None and is_replayable(inputs.record) else None
    delta = None if score is None or published_score is None else score - published_score
    return {
        "ghost_present": event_marker_flag(inputs, "ghost_forecast_json"),
        "ghost_log_score": score,
        "ghost_score_delta": delta,
        "ghost_score_delta_abs": None if delta is None else abs(delta),
    }


def research(inputs: ForecastInputs) -> dict[str, object]:
    archive, lines = inputs.archive, inputs.run.lines
    tags = research_tags_for_record(archive) if archive is not None else {}
    gap_fill = last_line(lines, "gap_fill_v2")
    budget = last_line(lines, "time_budget")
    is_artifact = archive is not None and archive.get("source") == ARTIFACT_SOURCE
    return {
        "research_source_class": archive.get("source") if archive else None,
        "research_is_trimmed": archive.get("is_trimmed") if archive else None,
        "research_chars": len(archive.get("research_text") or "") if archive else None,
        "research_run_id": archive.get("run_id") if archive else None,
        **_provider_columns(archive),
        "gap_fill_v1_used": archive.get("gap_fill_used") if is_artifact and archive else None,
        "anchor_present": tags.get("anchor_present"),
        "anchor_confidence": tags.get("anchor_confidence"),
        "gfv2_present": tags.get("gfv2_present"),
        "gfv2_loop_ran": tags.get("gfv2_loop_ran"),
        "gfv2_confidence": tags.get("gfv2_confidence"),
        "gfv2_tool_calls": gap_fill.get("tool_calls") if gap_fill else None,
        "fast_path": budget.get("fast_path") if budget else None,
        "budget_s": budget.get("budget_s") if budget else None,
        **_ghost_columns(inputs, type_log_score(inputs.record)),
    }


def _number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def _trait_rollups(members: Sequence[dict[str, object]]) -> dict[str, object]:
    with_rationale = [row for row in members if row["role"] == ROLE_MEMBER and row["has_rationale"]]
    means = {
        f"{name}_mean": statistics.fmean(_number(row[name]) for row in with_rationale) if with_rationale else None
        for name in MEANED_TRAITS
    }
    counts = {
        f"n_{name}": sum(bool(row[name]) for row in with_rationale) if with_rationale else None
        for name in COUNTED_TRAITS
    }
    return {"n_rationales": len(with_rationale), **means, **counts}


def config_cost_and_references(inputs: ForecastInputs) -> dict[str, object]:
    record = inputs.record
    credit = last_line(inputs.run.run_lines, "credit_run_summary")
    comment_text = record.get("comment_text")
    return {
        "config_era": record.get("config_era"),
        "triple_subera": record.get("triple_subera"),
        "triple_subera_fine": record.get("triple_subera_fine"),
        "usd_per_question": credit.get("usd_per_question") if credit else None,
        "run_n_questions": credit.get("n_questions") if credit else None,
        "telemetry_run_id": inputs.run.run_id,
        "telemetry_run_date": inputs.run.run_date,
        "n_telemetry_runs": inputs.run.n_runs,
        "comment_id": record.get("comment_id"),
        "has_comment": bool(comment_text),
        "comment_chars": len(comment_text) if comment_text else None,
        "explanation_detail_status": record.get("explanation_detail_status"),
        **_trait_rollups(inputs.members),
    }


def forecast_row(inputs: ForecastInputs) -> dict[str, object]:
    """One row for ``schema.FORECAST_COLUMNS``."""
    return {
        **identity_and_time(inputs),
        **question_and_outcome(inputs),
        **ensemble_and_cohorts(inputs),
        **research(inputs),
        **config_cost_and_references(inputs),
    }
