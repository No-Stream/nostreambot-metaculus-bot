"""One bundle row per forecast x ensemble member: the member's value, its own score, its rationale traits.

Values come from the comment (``per_model_numeric_percentiles`` on numeric and discrete, the base
models of ``per_model_forecasts`` on binary and multiple choice) plus any member only the
``member_forecast`` marker saw, such as a Mantic per-bin member. On a stacked record the summary
bullet is the stacker's aggregate, so it becomes a ``role == "stacker"`` row and nothing on that
record is rankable.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from metaculus_bot.performance_analysis.comment_sections import is_anonymous_model_key, parse_forecaster_model_map
from metaculus_bot.performance_analysis.member_replay import (
    is_censored,
    is_replayable,
    member_cdf,
    member_scoring_inputs,
    replay_solo_score,
    score_member_cdf,
)
from metaculus_bot.performance_analysis.parsing import _parse_probability, parse_per_model_reasoning_text
from metaculus_bot.performance_analysis.ranking_cohort import per_model_ranking_cohort
from metaculus_bot.performance_analysis.scaling import NUMERIC_TYPES
from metaculus_bot.performance_analysis.stacker_detection import base_or_per_model_forecasts, detect_stacker_fired
from metaculus_bot.trace_review.traits import TRAITS, trait_values

ROLE_MEMBER = "member"
ROLE_STACKER = "stacker"
SOURCE_COMMENT = "comment"
SOURCE_MEMBER_FORECAST = "member_forecast"
KIND_PROBABILITY = "probability"
KIND_OPTION_PROBS = "option_probs"
KIND_PERCENTILES = "percentiles"
KIND_PMF = "pmf"
CONFIRMED_STACKER = "confirmed_stacker"
NOT_RANKABLE_STACKER_ROW = "stacker_row"
NOT_RANKABLE_STACKED_RECORD = "stacked_record"
NOT_RANKABLE_ANONYMOUS = "anonymous_key"
NOT_RANKABLE_SPARSE = "sparse_anchors"
MEDIAN_LABEL = 50.0
PERCENT_PER_UNIT = 100.0
# Farthest-member flags need a median to be far from.
MIN_MEMBERS_FOR_FARTHEST = 3


@dataclass(frozen=True, slots=True)
class MemberValue:
    """One member's forecast as recovered, before scoring."""

    model: str
    role: str
    value_source: str
    value_kind: str
    value: Any


def short_model_name(model: str) -> str:
    """The comment parsers key members on the last path segment (``openrouter/x/name`` -> ``name``)."""
    return model.rsplit("/", 1)[-1].strip()


def latest_line_by_model(lines: list[dict], marker: str) -> dict[str, dict]:
    """The last line of one marker per member, keyed on the short model name."""
    by_model: dict[str, dict] = {}
    for line in lines:
        if line["marker"] == marker and line.get("model"):
            by_model[short_model_name(str(line["model"]))] = line
    return by_model


def _comment_value(raw: object, question_type: str) -> object:
    if question_type == "binary":
        return _parse_probability(raw) if isinstance(raw, str) else None
    return raw if isinstance(raw, dict) else None


def _marker_value(line: dict, record: dict) -> tuple[str, object]:
    """``(value_kind, value)`` from a ``member_forecast`` line's raw JSON, in the comment's units."""
    raw = json.loads(line["raw"])
    if line.get("elicitation") == KIND_PMF:
        return KIND_PMF, [float(mass) for mass in raw]
    question_type = record["type"]
    if question_type == "binary":
        return KIND_PROBABILITY, float(raw)
    if question_type == "multiple_choice":
        return KIND_OPTION_PROBS, dict(zip(record.get("options") or [], raw, strict=True))
    # Marker labels are fractions (0.025); comment labels are percent (2.5).
    return KIND_PERCENTILES, [[float(label) * PERCENT_PER_UNIT, float(value)] for label, value in raw]


def member_values(record: dict, lines: list[dict]) -> list[MemberValue]:
    """Every member of one forecast, comment first, then members only the marker saw."""
    question_type = record["type"]
    values: list[MemberValue] = []
    stacked = detect_stacker_fired(record) == CONFIRMED_STACKER
    if question_type in NUMERIC_TYPES:
        # A stacked comment's percentile lines are the stacker's own; its base members' curves are not published.
        role = ROLE_STACKER if stacked else ROLE_MEMBER
        for model, pairs in (record.get("per_model_numeric_percentiles") or {}).items():
            values.append(MemberValue(model, role, SOURCE_COMMENT, KIND_PERCENTILES, [list(p) for p in pairs]))
    else:
        kind = KIND_PROBABILITY if question_type == "binary" else KIND_OPTION_PROBS
        members = (record.get("per_base_model_forecasts") or {}) if stacked else base_or_per_model_forecasts(record)
        stackers = (record.get("per_model_forecasts") or {}) if stacked else {}
        for role, entries in ((ROLE_MEMBER, members), (ROLE_STACKER, stackers)):
            for model, raw in entries.items():
                values.append(MemberValue(model, role, SOURCE_COMMENT, kind, _comment_value(raw, question_type)))
    known = {value.model for value in values}
    for model, line in latest_line_by_model(lines, "member_forecast").items():
        if model not in known and line.get("role") == ROLE_MEMBER:
            kind, value = _marker_value(line, record)
            values.append(MemberValue(model, ROLE_MEMBER, SOURCE_MEMBER_FORECAST, kind, value))
    return values


def pmf_cdf(bins: list[float]) -> list[float]:
    """A per-bin vector ``[below, p_0 .. p_{N-1}, above]`` as CDF heights on the N+1 grid points."""
    return list(np.cumsum(bins)[:-1])


def _numeric_cdf(value: MemberValue, record: dict) -> list[float] | None:
    inputs = member_scoring_inputs(record)
    if inputs is None or value.value is None:
        return None
    if value.value_kind == KIND_PMF:
        cdf = pmf_cdf(value.value)
        return cdf if len(cdf) == inputs.cdf_size else None
    try:
        return member_cdf(value.value, inputs)
    except (ValueError, RuntimeError):
        return None


def own_score(value: MemberValue, record: dict) -> float | None:
    """The member's own log score as declared, the residual library's solo replay."""
    if value.value is None or not is_replayable(record):
        return None
    if value.value_kind == KIND_OPTION_PROBS:
        options = record.get("options") or []
        return replay_solo_score(record, value.value) if all(o in value.value for o in options) else None
    if value.value_kind == KIND_PMF:
        inputs = member_scoring_inputs(record)
        cdf = _numeric_cdf(value, record)
        if inputs is None or cdf is None:
            return None
        try:
            return score_member_cdf(cdf, inputs)
        except (ValueError, ZeroDivisionError):
            return None
    return replay_solo_score(record, value.value)


def _score_censored(value: MemberValue, record: dict) -> bool | None:
    if record["type"] not in NUMERIC_TYPES or not is_replayable(record):
        return None
    inputs = member_scoring_inputs(record)
    cdf = _numeric_cdf(value, record)
    return None if inputs is None or cdf is None else is_censored(cdf, inputs)


def _center_value(value: MemberValue, record: dict) -> float | None:
    """Probability (binary), probability on the published top option (multiple choice), declared median."""
    if value.value is None:
        return None
    if value.value_kind == KIND_PROBABILITY:
        return float(value.value)
    if value.value_kind == KIND_OPTION_PROBS:
        options, published = record.get("options") or [], record.get("our_forecast_values") or []
        if len(options) != len(published) or not options:
            return None
        top_option = options[int(np.argmax(published))]
        top = value.value.get(top_option)
        return None if top is None else float(top)
    if value.value_kind == KIND_PERCENTILES:
        return next((float(v) for label, v in value.value if float(label) == MEDIAN_LABEL), None)
    return None


def _prob_on_outcome(value: MemberValue, record: dict) -> float | None:
    resolution = record.get("resolution_parsed")
    if value.value is None:
        return None
    if value.value_kind == KIND_PROBABILITY and isinstance(resolution, bool):
        return float(value.value) if resolution else 1.0 - float(value.value)
    if value.value_kind == KIND_OPTION_PROBS and isinstance(resolution, str):
        total = sum(float(p) for p in value.value.values())
        return float(value.value[resolution]) / total if resolution in value.value and total > 0 else None
    return None


def _not_rankable_reason(value: MemberValue, *, stacked: bool, sparse_models: set[str]) -> str | None:
    if value.role == ROLE_STACKER:
        return NOT_RANKABLE_STACKER_ROW
    if stacked:
        return NOT_RANKABLE_STACKED_RECORD
    if is_anonymous_model_key(value.model):
        return NOT_RANKABLE_ANONYMOUS
    if value.value_source == SOURCE_COMMENT and value.model in sparse_models:
        return NOT_RANKABLE_SPARSE
    return None


def _value_columns(value: MemberValue) -> dict[str, object]:
    is_curve = value.value_kind in (KIND_PERCENTILES, KIND_PMF) and value.value is not None
    return {
        "prob_yes": float(value.value) if value.value_kind == KIND_PROBABILITY and value.value is not None else None,
        "option_probs_json": json.dumps(value.value)
        if value.value_kind == KIND_OPTION_PROBS and value.value is not None
        else None,
        "percentiles_json": json.dumps(value.value) if is_curve else None,
    }


def _marker_columns(model: str, by_marker: Mapping[str, Mapping[str, dict]]) -> dict[str, object]:
    forecast_line = by_marker["member_forecast"].get(model)
    rung_line = by_marker["extraction_rung"].get(model)
    extreme_line = by_marker["extreme_call"].get(model)
    raw = forecast_line.get("raw") if forecast_line else None
    published = forecast_line.get("published") if forecast_line else None
    return {
        "raw_value_json": raw,
        "published_value_json": published,
        "clamp_moved": json.loads(raw) != json.loads(published) if raw is not None and published is not None else None,
        "oor_low": forecast_line.get("oor_low") if forecast_line else None,
        "oor_high": forecast_line.get("oor_high") if forecast_line else None,
        "elicitation": forecast_line.get("elicitation") if forecast_line else None,
        "extraction_rung": rung_line.get("rung") if rung_line else None,
        "block_present": rung_line.get("block_present") if rung_line else None,
        "extreme_call_side": extreme_line.get("side") if extreme_line else None,
    }


def _trait_columns(rationale: str | None) -> dict[str, object]:
    if rationale is None:
        return {"has_rationale": False, **{trait.name: None for trait in TRAITS}}
    return {"has_rationale": True, **trait_values(rationale)}


@dataclass(frozen=True)
class MemberContext:
    """What every member row of one forecast shares."""

    forecast_key: str
    platform: str
    type_log_score: float | None
    rationales: dict[str, str]
    index_by_model: dict[str, int]
    by_marker: dict[str, dict[str, dict]]
    stacked: bool
    sparse_models: set[str]


def _context(
    record: dict, *, key: str, platform: str, lines: list[dict], type_log_score: float | None
) -> MemberContext:
    comment_text = record.get("comment_text") or ""
    sparse = set(per_model_ranking_cohort(record).sparse_anchors) if record["type"] in NUMERIC_TYPES else set()
    return MemberContext(
        forecast_key=key,
        platform=platform,
        type_log_score=type_log_score,
        rationales=parse_per_model_reasoning_text(comment_text) if comment_text else {},
        index_by_model={name: index for index, name in parse_forecaster_model_map(comment_text).items()},
        by_marker={m: latest_line_by_model(lines, m) for m in ("member_forecast", "extraction_rung", "extreme_call")},
        stacked=detect_stacker_fired(record) == CONFIRMED_STACKER,
        sparse_models=sparse,
    )


def _member_row(value: MemberValue, record: dict, context: MemberContext) -> dict[str, object]:
    score = own_score(value, record)
    reason = _not_rankable_reason(value, stacked=context.stacked, sparse_models=context.sparse_models)
    return {
        "forecast_key": context.forecast_key,
        "platform": context.platform,
        "question_type": record["type"],
        "model": value.model,
        "model_is_anonymous": is_anonymous_model_key(value.model),
        "forecaster_index": context.index_by_model.get(value.model),
        "role": value.role,
        "value_source": value.value_source,
        "value_kind": value.value_kind,
        **_value_columns(value),
        "center_value": _center_value(value, record),
        "center_dev_from_median": None,
        "is_farthest_member": None,
        "prob_on_outcome": _prob_on_outcome(value, record),
        "own_score": score,
        "score_minus_published": None
        if score is None or context.type_log_score is None
        else score - context.type_log_score,
        "score_censored": _score_censored(value, record),
        "rankable": reason is None,
        "not_rankable_reason": reason,
        **_marker_columns(value.model, context.by_marker),
        **_trait_columns(context.rationales.get(value.model)),
    }


def _mark_deviation_from_median(rows: list[dict[str, object]]) -> None:
    """Each member's distance from the members' median center, and which member sits farthest."""
    centered = [
        (row, center) for row in rows if row["role"] == ROLE_MEMBER and isinstance(center := row["center_value"], float)
    ]
    if not centered:
        return
    median = statistics.median(center for _, center in centered)
    for row, center in centered:
        row["center_dev_from_median"] = center - median
    deviations = [abs(center - median) for _, center in centered]
    if len(centered) < MIN_MEMBERS_FOR_FARTHEST or max(deviations) == 0.0:
        return
    for (row, _), deviation in zip(centered, deviations, strict=True):
        row["is_farthest_member"] = deviation == max(deviations)


def member_rows(
    record: dict, *, key: str, platform: str, lines: list[dict], type_log_score: float | None
) -> list[dict[str, object]]:
    """Every member row of one forecast, ready for ``schema.MEMBER_COLUMNS``."""
    context = _context(record, key=key, platform=platform, lines=lines, type_log_score=type_log_score)
    rows = [_member_row(value, record, context) for value in member_values(record, lines)]
    _mark_deviation_from_median(rows)
    return rows
