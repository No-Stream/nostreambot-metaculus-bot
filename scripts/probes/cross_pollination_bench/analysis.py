"""Offline hierarchical analysis and decision rule for saved paired rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from metaculus_bot.performance_analysis.paired_comparison import PairedScoreRow, compare_paired

PRIMARY_CONTRAST = "peers_full_vs_second_look"
MEMBER_CONTRAST = "peers_full_member_own_score"
CONTRAST_NAMES = (
    PRIMARY_CONTRAST,
    "peers_full_vs_first",
    "peers_full_vs_pooled_draws",
    "peers_full_vs_blend",
    "peers_forecasts_vs_second_look",
    "second_look_vs_first",
)
CONTRASTS = CONTRAST_NAMES
STOP_GAIN_PROBABILITY_MAX = 0.10
CONFIRM_POSITIVE_PROBABILITY_MIN = 0.90
EFFECT_SIZE_POINTS = 2.0
MEMBER_MATERIAL_LOSS_POINTS = 2.0


def _paired_rows_digest(rows: dict[str, list[dict[str, Any]]]) -> str:
    serialized = json.dumps(
        [row for contrast in sorted(rows) for row in rows[contrast]],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_paired_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "contrast" not in row:
                raise ValueError(f"invalid paired row at {path}:{line_number}")
            grouped.setdefault(str(row["contrast"]), []).append(row)
    if not grouped:
        raise ValueError(f"no paired rows in {path}")
    return grouped


def _model_rows(rows: list[dict[str, Any]]) -> list[PairedScoreRow]:
    return [
        PairedScoreRow(
            unit_id=str(row["unit_id"]),
            cluster_id=str(row["cluster_id"]),
            group=str(row["group"]),
            replicate=int(row["replicate"]),
            difference=float(row["difference"]),
            member=str(row["member"]) if row.get("member") is not None else None,
        )
        for row in rows
    ]


def run_analysis(
    paired_by_contrast: dict[str, list[dict[str, Any]]],
    *,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    allow_diagnostics_warnings: bool,
) -> dict[str, Any]:
    fit_options = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "seed": seed,
        "allow_diagnostics_warnings": allow_diagnostics_warnings,
    }
    contrast_results: dict[str, Any] = {}
    for contrast in CONTRAST_NAMES:
        contrast_rows = paired_by_contrast.get(contrast, [])
        if not contrast_rows:
            raise ValueError(f"paired rows are missing contrast {contrast}")
        contrast_results[contrast] = compare_paired(_model_rows(contrast_rows), **fit_options)
    primary_rows = paired_by_contrast[PRIMARY_CONTRAST]
    complete = [row for row in primary_rows if row["complete_panel"]]
    after_release = [row for row in primary_rows if row["after_release"]]
    member_rows = paired_by_contrast.get(MEMBER_CONTRAST, [])
    if not member_rows:
        raise ValueError(f"paired rows are missing contrast {MEMBER_CONTRAST}")
    return {
        "contrasts": contrast_results,
        "primary_complete_panels": compare_paired(_model_rows(complete), **fit_options),
        "primary_after_release": compare_paired(_model_rows(after_release), **fit_options),
        "members": compare_paired(_model_rows(member_rows), **fit_options),
        "paired_rows_sha256": _paired_rows_digest(paired_by_contrast),
    }


def decide(results: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    movement = results["movement"]
    primary = analysis["contrasts"][PRIMARY_CONTRAST]["student_t"]["effect"]
    contrasts = analysis["contrasts"]
    member_effects = analysis["members"]["student_t"]["members"]
    movement_stops = movement["peers_full"]["typical_change"] <= movement["second_look"]["typical_change"]
    gain_probability = primary["p_gain_above"][str(EFFECT_SIZE_POINTS)]
    unlikely_material_gain = gain_probability < STOP_GAIN_PROBABILITY_MAX
    confirmation_conditions = {
        "positive_probability_at_least_0_90": primary["p_above_zero"] >= CONFIRM_POSITIVE_PROBABILITY_MIN,
        "no_worse_than_first_draw": contrasts["peers_full_vs_first"]["student_t"]["effect"]["mean"] >= 0,
        "no_worse_than_pooled_draws": contrasts["peers_full_vs_pooled_draws"]["student_t"]["effect"]["mean"] >= 0,
        "beats_mechanical_blend": contrasts["peers_full_vs_blend"]["student_t"]["effect"]["mean"] > 0,
        "no_member_material_loss": all(
            value["mean"] >= -MEMBER_MATERIAL_LOSS_POINTS for value in member_effects.values()
        ),
    }
    outcome = (
        "stop"
        if movement_stops or unlikely_material_gain
        else "price_production_confirmation"
        if all(confirmation_conditions.values())
        else "inconclusive"
    )
    return {
        "outcome": outcome,
        "criteria": {
            "movement_stops": movement_stops,
            "peers_full_typical_change": movement["peers_full"]["typical_change"],
            "second_look_typical_change": movement["second_look"]["typical_change"],
            "p_gain_above_2_points": gain_probability,
            "unlikely_material_gain": unlikely_material_gain,
            "p_above_zero": primary["p_above_zero"],
            "peers_full_vs_first_mean": contrasts["peers_full_vs_first"]["student_t"]["effect"]["mean"],
            "peers_full_vs_pooled_draws_mean": contrasts["peers_full_vs_pooled_draws"]["student_t"]["effect"]["mean"],
            "peers_full_vs_blend_mean": contrasts["peers_full_vs_blend"]["student_t"]["effect"]["mean"],
            "member_posterior_means": {name: value["mean"] for name, value in member_effects.items()},
            **confirmation_conditions,
        },
    }
