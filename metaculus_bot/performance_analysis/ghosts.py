"""Ghost-forecast markers as data: parsing, per-question selection, the post-id join and the numeric CDF build.

Gap-fill v2 logs an unpublished "ghost" forecast per question. Both ``scripts/score_ghosts.py`` (the paired
log-score read) and the loss-attribution report price ghosts against the published forecast, so the marker
reading lives here, inside the package that import-linter lets both of them import. A ghost's ``qid`` is the
record's POST id, never its question id. Scoring stays with each caller.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from metaculus_bot.mc_processing import clamp_and_renormalize_probs
from metaculus_bot.numeric.config import grid_step_constraints
from metaculus_bot.performance_analysis.clip_threshold_sweep import in_force_bounds
from metaculus_bot.performance_analysis.collector import resolve_numeric_record_to_score_inputs
from metaculus_bot.performance_analysis.round_dataset import load_marker_records
from metaculus_bot.time_utils import parse_iso_utc

_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_BINARY_RE = re.compile(rf"posterior_prob=(?P<p>{_FLOAT})")
_MEDIAN_RE = re.compile(rf"median=(?P<m>{_FLOAT})")
_MC_PAIR_RE = re.compile(rf"(?P<name>[^=,]+)=(?P<prob>{_FLOAT})")


def parse_ghost_summary(qtype: str, summary: str) -> float | dict[str, float] | None:
    """Parse a legacy ``GHOST_FORECAST`` ``summary=`` payload into a scoreable value.

    * binary          -> ``float`` posterior probability
    * multiple_choice -> ``{option_name: prob}``
    * numeric         -> ``{"median": float}`` (median only; not enough to log-score)
    * anything else / empty -> ``None``
    """
    summary = (summary or "").strip()
    if not summary:
        return None
    if qtype == "binary":
        match = _BINARY_RE.search(summary)
        return float(match.group("p")) if match else None
    if qtype == "multiple_choice":
        probs = {name.strip(): float(prob) for name, prob in _MC_PAIR_RE.findall(summary)}
        return probs or None
    if qtype == "numeric":
        match = _MEDIAN_RE.search(summary)
        return {"median": float(match.group("m"))} if match else None
    return None


def latest_ghost_per_qid(ghosts: list[dict]) -> dict[int, dict]:
    """Keep the newest ghost per question id (by run_date, then line_ts, then seq)."""
    latest: dict[int, dict] = {}
    for ghost in ghosts:
        qid = ghost.get("qid")
        if qid is None:
            continue
        key = (str(ghost.get("run_date", "")), str(ghost.get("line_ts") or ""), ghost.get("seq", 0))
        current = latest.get(qid)
        if current is None or key > (
            str(current.get("run_date", "")),
            str(current.get("line_ts") or ""),
            current.get("seq", 0),
        ):
            latest[qid] = ghost
    return latest


def normalize_json_ghost(record: dict) -> dict | None:
    """Turn a harvested ``ghost_forecast_json`` record into a normalized ghost, or None.

    The ``forecast_json`` field is a raw JSON string (never coerced by the marker
    parser). None if it is missing / malformed so the caller can fall back to any
    legacy ghost for the same qid.
    """
    raw = record.get("forecast_json")
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "qid": record.get("qid"),
        "qtype": payload.get("qtype"),
        "source": "json",
        "payload": payload,
        # Run identity: a paired read needs both halves from the SAME run (a qid can be forecast in several runs).
        "run_id": record.get("run_id"),
    }


def select_ghost_per_qid(json_ghosts: list[dict], legacy_ghosts: list[dict]) -> dict[int, dict]:
    """Pick one normalized ghost to score per qid: JSON-source preferred, legacy fallback.

    JSON ghosts carry the full forecast (deterministically parsable); legacy ghosts
    carry only the lossy summary string. Within each source we keep the latest per
    qid; a (well-formed) JSON ghost always wins over a legacy one for the same qid.
    """
    selected: dict[int, dict] = {}
    for qid, record in latest_ghost_per_qid(legacy_ghosts).items():
        selected[qid] = {
            "qid": qid,
            "qtype": record.get("qtype"),
            "source": "legacy",
            "summary": record.get("summary", ""),
        }
    for qid, record in latest_ghost_per_qid(json_ghosts).items():
        normalized = normalize_json_ghost(record)
        if normalized is not None:
            selected[qid] = normalized
    return selected


def ghost_binary_prob(ghost: dict) -> float | None:
    """Return a ghost's posterior probability from either source."""
    if ghost["source"] == "json":
        prob = ghost["payload"].get("prob")
        return float(prob) if isinstance(prob, (int, float)) and not isinstance(prob, bool) else None
    parsed = parse_ghost_summary("binary", ghost.get("summary", ""))
    return parsed if isinstance(parsed, float) else None


def ghost_mc_probs(ghost: dict) -> dict[str, float] | None:
    """Return a ghost's option-to-probability mapping from either source."""
    if ghost["source"] == "json":
        option_probs = ghost["payload"].get("option_probs")
        if not isinstance(option_probs, dict) or not option_probs:
            return None
        try:
            return {str(name): float(prob) for name, prob in option_probs.items()}
        except (TypeError, ValueError):
            return None
    parsed = parse_ghost_summary("multiple_choice", ghost.get("summary", ""))
    return parsed if isinstance(parsed, dict) else None


def clamp_ghost_to_published_bounds(ghost: dict, record: dict) -> float | dict[str, float] | None:
    """Return a binary or MC ghost after the production clamp in force at publish time."""
    question_type = record["type"]
    forecast_time = parse_iso_utc(record.get("bot_comment_created_at"))
    lower, upper = in_force_bounds(question_type, forecast_time)
    if question_type == "binary":
        probability = ghost_binary_prob(ghost)
        return min(upper, max(lower, probability)) if probability is not None else None
    if question_type == "multiple_choice":
        probabilities = ghost_mc_probs(ghost)
        if probabilities is None:
            return None
        clamped = clamp_and_renormalize_probs(list(probabilities.values()), lo=lower, hi=upper)
        return dict(zip(probabilities, clamped, strict=True))
    return None


def ghost_numeric_percentiles(ghost: dict) -> dict[float, float] | None:
    """Ghost's full percentile->value map (fraction keys in [0, 1]), or None.

    Only JSON-source ghosts carry the full percentile set; the legacy marker exposes
    a median only and so can't be turned into a CDF.
    """
    if ghost["source"] != "json":
        return None
    declared = ghost["payload"].get("declared_percentiles")
    if not isinstance(declared, dict) or len(declared) < 2:
        return None
    try:
        return {float(pct): float(value) for pct, value in declared.items()}
    except (TypeError, ValueError):
        return None


def records_by_post_id(records: list[dict]) -> dict[int, dict]:
    """Index resolved records by the post id used by ghost markers."""
    return {record["post_id"]: record for record in records}


def ghost_cdf_on_published_grid(percentiles: dict[float, float], record: dict) -> list[float] | None:
    """The ghost's CDF rebuilt on the published grid, or None when it cannot be built.

    Uses the production PCHIP builder with the published grid length and the step bounds scaled to it, so
    the ghost and the published CDF are scored on one grid. None when the record has no published CDF, no
    score inputs, or the builder rejects the declared percentiles.
    """
    published_cdf = record.get("our_forecast_values")
    if not published_cdf or len(published_cdf) < 2:
        return None
    score_inputs = resolve_numeric_record_to_score_inputs(record)
    if score_inputs is None:
        return None

    # Function-scoped so tests patching pchip_cdf.generate_pchip_cdf at its source see this call.
    from metaculus_bot.numeric.pchip_cdf import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # late import: tests patch numeric.pchip_cdf.generate_pchip_cdf at source
        generate_pchip_cdf,
    )

    _, lower, upper, zero_point = score_inputs
    pct_values = {frac * 100.0: value for frac, value in percentiles.items()}
    num_points = len(published_cdf)
    min_step, max_step = grid_step_constraints(num_points)
    try:
        ghost_cdf, _ = generate_pchip_cdf(
            pct_values,
            open_upper_bound=bool(record.get("open_upper_bound", False)),
            open_lower_bound=bool(record.get("open_lower_bound", False)),
            upper_bound=upper,
            lower_bound=lower,
            zero_point=zero_point,
            min_step=min_step,
            max_step=max_step,
            num_points=num_points,
        )
    except (ValueError, RuntimeError):
        return None
    return ghost_cdf


def load_concluding_ghosts(telemetry_dir: Path) -> dict[int, dict]:
    """Load concluding JSON and legacy ghosts from a telemetry archive."""
    json_ghosts = load_marker_records(telemetry_dir, "ghost_forecast_json")
    legacy_ghosts = load_marker_records(telemetry_dir, "ghost_forecast")
    return select_ghost_per_qid(json_ghosts, legacy_ghosts)
