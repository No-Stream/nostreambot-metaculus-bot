"""Offline summaries of saved cross-pollination calls."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.probes.bench_common.records import STATUS_SCORED
from scripts.probes.cross_pollination_bench.aggregate import (
    FIRST_MEDIAN,
    MECHANICAL_BLEND,
    POOLED_FIRST_DRAWS,
    aggregate_scores,
    member_paired_rows,
    paired_rows,
)
from scripts.probes.cross_pollination_bench.manifest import Manifest
from scripts.probes.cross_pollination_bench.models import BENCH_MODELS, BenchModel
from scripts.probes.cross_pollination_bench.rows import STAGE_FIRST, UPDATE_VARIANTS, BenchCallRow

SMOKE_CACHE_SHARE_MIN = 0.80
SMOKE_COST_TOLERANCE = 0.10
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("peers_full_vs_second_look", "peers_full_median", "second_look_median"),
    ("peers_full_vs_first", "peers_full_median", FIRST_MEDIAN),
    ("peers_full_vs_pooled_draws", "peers_full_median", POOLED_FIRST_DRAWS),
    ("peers_full_vs_blend", "peers_full_median", MECHANICAL_BLEND),
    ("peers_forecasts_vs_second_look", "peers_forecasts_median", "second_look_median"),
    ("second_look_vs_first", "second_look_median", FIRST_MEDIAN),
)


def _token_stats(rows: list[BenchCallRow], model: BenchModel) -> dict[str, Any]:
    prompt = sum(row.prompt_tokens for row in rows)
    cached = sum(row.cached_tokens for row in rows)
    writes = sum(row.cache_write_tokens for row in rows)
    billed = sum(row.cost_usd or 0.0 for row in rows)
    uncached = sum(
        (row.prompt_tokens * model.input_usd_per_m + row.completion_tokens * model.output_usd_per_m) / 1_000_000
        for row in rows
    )
    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_write_tokens": writes,
        "cached_share": cached / prompt if prompt else 0.0,
        "billed_usd": billed,
        "uncached_rate_usd": uncached,
        "savings_usd": uncached - billed,
    }


def cache_summary(rows: list[BenchCallRow], models: tuple[BenchModel, ...] | list[BenchModel]) -> dict[str, Any]:
    result = {}
    for model in models:
        replied = [row for row in rows if row.model == model.label and row.reply_text is not None]
        updates = [row for row in replied if row.stage in UPDATE_VARIANTS]
        result[model.label] = {**_token_stats(replied, model), "updates": _token_stats(updates, model)}
    return result


def failure_counts(rows: list[BenchCallRow], manifest: Manifest) -> dict[str, Any]:
    by_model = {}
    models = sorted({row.model for row in rows})
    variants = sorted(set(UPDATE_VARIANTS))
    for model in models:
        by_model[model] = {
            "failed_first_calls": sum(
                row.model == model and row.stage == STAGE_FIRST and row.status != STATUS_SCORED for row in rows
            ),
            "carried_forward_updates": {
                variant: sum(
                    row.model == model and row.stage == variant and row.status != STATUS_SCORED for row in rows
                )
                for variant in variants
            },
        }
    panel_keys = {(row.question_id, row.draw) for row in rows}
    complete = sum(
        all(
            any(
                row.question_id == qid
                and row.draw == draw
                and row.model == model.label
                and row.stage == stage
                and row.status == STATUS_SCORED
                for row in rows
            )
            for model in BENCH_MODELS
            for stage in (STAGE_FIRST, *UPDATE_VARIANTS)
        )
        for qid, draw in panel_keys
    )
    return {
        "by_model": by_model,
        "complete_panels": complete,
        "missing_panels": [
            {"question_id": qid, "draw": draw}
            for qid, draw in sorted(panel_keys)
            if not any(
                r.question_id == qid and r.draw == draw and r.stage == STAGE_FIRST and r.status == STATUS_SCORED
                for r in rows
            )
        ],
    }


def smoke_checks(rows: list[BenchCallRow], models: tuple[BenchModel, ...] | list[BenchModel]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    by_call = {(row.question_id, row.model, row.draw, row.stage): row for row in rows}
    for model in models:
        for row in rows:
            if row.model != model.label:
                continue
            if row.stage == "peers_full" and row.reply_text is not None:
                first = by_call.get((row.question_id, row.model, row.draw, STAGE_FIRST))
                minimum = (first.prompt_tokens if first else row.prompt_tokens) * SMOKE_CACHE_SHARE_MIN
                checks.append(
                    {
                        "check": "cached_tokens",
                        "model": model.label,
                        "question_id": row.question_id,
                        "passed": row.cached_tokens >= minimum,
                        "actual": row.cached_tokens,
                        "minimum": minimum,
                    }
                )
                expected = (
                    (row.prompt_tokens - row.cached_tokens) * model.input_usd_per_m
                    + row.cached_tokens * model.cache_read_usd_per_m
                    + row.completion_tokens * model.output_usd_per_m
                ) / 1_000_000
                actual = row.cost_usd or 0.0
                valid_cost = any(
                    abs(actual - expected * multiplier) <= max(expected * multiplier * SMOKE_COST_TOLERANCE, 1e-9)
                    for multiplier in ((1.0, 2.0) if model.label == "deepseek" else (1.0,))
                )
                checks.append(
                    {
                        "check": "cached_cost",
                        "model": model.label,
                        "question_id": row.question_id,
                        "passed": valid_cost,
                        "actual": actual,
                        "expected": expected,
                    }
                )
            checks.append(
                {
                    "check": "rung",
                    "model": model.label,
                    "question_id": row.question_id,
                    "passed": row.rung == "block",
                    "actual": row.rung,
                }
            )
            checks.append(
                {
                    "check": "upstream_provider",
                    "model": model.label,
                    "question_id": row.question_id,
                    "passed": row.upstream_provider == model.upstream_provider,
                    "actual": row.upstream_provider,
                }
            )
    return checks


def _aggregate_summary(scores: list[Any]) -> dict[str, Any]:
    aggregate_groups: dict[str, list[Any]] = defaultdict(list)
    for score in scores:
        aggregate_groups[score.aggregate].append(score)
    aggregates = {}
    for name, values in aggregate_groups.items():
        aggregates[name] = {
            "mean_peer_points": statistics.fmean(value.peer_points for value in values),
            "n": len(values),
            "complete_panels": sum(value.complete_panel for value in values),
            "complete_panel_mean_peer_points": (
                statistics.fmean(value.peer_points for value in values if value.complete_panel)
                if any(value.complete_panel for value in values)
                else None
            ),
            "by_type": {
                qtype: {"n": len(selected), "mean_peer_points": statistics.fmean(s.peer_points for s in selected)}
                for qtype in sorted({v.qtype for v in values})
                if (selected := [v for v in values if v.qtype == qtype])
            },
            "complete_panels_by_type": {
                qtype: {"n": len(selected), "mean_peer_points": statistics.fmean(s.peer_points for s in selected)}
                for qtype in sorted({v.qtype for v in values})
                if (selected := [v for v in values if v.qtype == qtype and v.complete_panel])
            },
        }
    return aggregates


def _contrast_tables(
    scores: list[Any], manifest: Manifest, event_map: dict[int, str] | None
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    contrasts: dict[str, Any] = {}
    paired: dict[str, list[dict[str, Any]]] = {}
    for name, treatment, control in CONTRASTS:
        all_rows = paired_rows(scores, manifest, treatment=treatment, control=control, event_map=event_map)
        complete_rows = [row for row in all_rows if row["complete_panel"]]
        after = [row for row in all_rows if row["after_release"]]

        def mean(items: list[dict[str, Any]]) -> float | None:
            return statistics.fmean(row["difference"] for row in items) if items else None

        contrasts[name] = {
            "n_question_draws": len(all_rows),
            "mean_difference": mean(all_rows),
            "complete_panel_mean_difference": mean(complete_rows),
            "after_release_mean_difference": mean(after),
        }
        paired[name] = [{**row, "contrast": name} for row in all_rows]
    return contrasts, paired


def _per_question_rows(scores: list[Any], manifest: Manifest) -> list[dict[str, Any]]:
    per_question = []
    entry_map = {entry.question_id: entry for entry in manifest.questions}
    for score in scores:
        row = {
            "question_id": score.question_id,
            "draw": score.draw,
            "aggregate": score.aggregate,
            "peer_points": score.peer_points,
            "forecast": score.forecast,
        }
        entry = entry_map[score.question_id]
        if entry.qtype == "binary":
            row["realized_probability"] = score.forecast if entry.resolution_parsed else 1 - score.forecast
        elif entry.qtype == "multiple_choice":
            question = next(r for r in manifest.questions if r.question_id == score.question_id)
            options = question.question_json["options"]
            row["realized_probability"] = score.forecast[options.index(entry.resolution_parsed)]
        per_question.append(row)
    return per_question


def _spend_summary(rows: list[BenchCallRow]) -> dict[str, Any]:
    spend_by_model: dict[str, float] = defaultdict(float)
    spend_by_key: dict[str, float] = defaultdict(float)
    for row in rows:
        if row.cost_usd is not None:
            spend_by_model[row.model] += row.cost_usd
            spend_by_key[row.key_alias] += row.cost_usd
    return {
        "by_model": dict(spend_by_model),
        "by_key_alias": dict(spend_by_key),
        "measured_usd": sum(spend_by_model.values()),
    }


def _smoke_summary(rows: list[BenchCallRow], manifest: Manifest, results: dict[str, Any]) -> None:
    checks = smoke_checks(rows, BENCH_MODELS)
    results["smoke_checks"] = checks
    cost_per_call: dict[str, dict[str, float]] = {}
    for model in sorted({row.model for row in rows}):
        cost_per_call[model] = {}
        for stage, label in ((STAGE_FIRST, "first"), ("peers_full", "update")):
            selected = [
                row.cost_usd
                for row in rows
                if row.model == model
                and (row.stage == stage if label == "first" else row.stage in UPDATE_VARIANTS)
                and row.cost_usd is not None
            ]
            if selected:
                cost_per_call[model][label] = statistics.fmean(selected)
    results["measured_cost_per_call"] = cost_per_call
    results["restated_main_run_usd"] = sum(
        values.get("first", 0.0) * len(manifest.questions) * 2
        + values.get("update", 0.0) * len(manifest.questions) * 2 * len(UPDATE_VARIANTS)
        for values in cost_per_call.values()
    )


def build_results(
    rows: list[BenchCallRow], manifest: Manifest, run_meta: dict[str, Any], *, event_map: dict[int, str] | None = None
) -> dict[str, Any]:
    scores = aggregate_scores(rows, manifest)
    contrasts, paired = _contrast_tables(scores, manifest, event_map)
    member_rows = member_paired_rows(rows, manifest, variant="peers_full", event_map=event_map)
    paired["peers_full_member_own_score"] = [{**row, "contrast": "peers_full_member_own_score"} for row in member_rows]
    result = {
        "question_set": {
            "n": len(manifest.questions),
            "after_release": sum(q.after_release for q in manifest.questions),
            "by_type": dict(Counter(q.qtype for q in manifest.questions)),
        },
        "failures": failure_counts(rows, manifest),
        "aggregates": _aggregate_summary(scores),
        "contrasts": contrasts,
        "per_question": _per_question_rows(scores, manifest),
        "cache": cache_summary(rows, BENCH_MODELS),
        "spend": _spend_summary(rows),
        "paired_rows": paired,
    }
    result["spend"]["parser_usd"] = run_meta.get("parser_usd", 0.0)
    result["spend"]["measured_total_usd"] = run_meta.get("measured_usd", result["spend"]["measured_usd"])
    if run_meta.get("smoke"):
        _smoke_summary(rows, manifest, result)
    return result


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Cross-pollination bench",
        "",
        f"Questions: {results['question_set']['n']}; after release: {results['question_set']['after_release']}.",
        "",
        "## Contrasts",
        "",
        "| contrast | question-draws | mean difference | complete panels | after release |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in results["contrasts"].items():
        lines.append(
            f"| {name} | {values['n_question_draws']} | {values['mean_difference']} | {values['complete_panel_mean_difference']} | {values['after_release_mean_difference']} |"
        )
    lines += [
        "",
        "## Failures",
        "",
        json.dumps(results["failures"], sort_keys=True),
        "",
        "## Spend",
        "",
        json.dumps(results["spend"], sort_keys=True),
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    run_dir: Path,
    rows: list[BenchCallRow],
    manifest: Manifest,
    run_meta: dict[str, Any],
    *,
    event_map: dict[int, str] | None = None,
) -> None:
    results = build_results(rows, manifest, run_meta, event_map=event_map)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "SUMMARY.md").write_text(render_markdown(results), encoding="utf-8")
    paired_rows = [row for values in results["paired_rows"].values() for row in values]
    (run_dir / "paired_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in paired_rows), encoding="utf-8"
    )
