"""Bayesian partial pooling of spot-peer scores by question type and era."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from metaculus_bot.performance_analysis.collector import load_dataset
from metaculus_bot.performance_analysis.era_gap import ERA_FIELD, SECONDS_PER_DAY, in_exclusion_cohort
from metaculus_bot.performance_analysis.markdown import markdown_table
from metaculus_bot.performance_analysis.platform_scores import spot_peer_score
from metaculus_bot.time_utils import parse_iso_utc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoolingData:
    types: tuple[str, ...]
    eras: tuple[str, ...]
    days: tuple[str, ...]
    scores: np.ndarray
    type_index: np.ndarray
    era_index: np.ndarray
    day_index: np.ndarray
    standardized_horizon: np.ndarray
    horizon_observed: np.ndarray
    dropped_missing_score: int
    missing_horizon: int
    excluded: int
    log_horizon_mean: float
    log_horizon_sd: float


def prepare_data(
    records: list[dict], *, era_field: str = ERA_FIELD, eras: list[str] | None = None, strict: bool = False
) -> PoolingData:
    selected = [record for record in records if eras is None or record.get(era_field) in eras]
    excluded = sum(in_exclusion_cohort(record) for record in selected) if strict else 0
    if strict:
        selected = [record for record in selected if not in_exclusion_cohort(record)]
    scored = [(record, spot_peer_score(record)) for record in selected]
    dropped = sum(score is None for _, score in scored)
    scored = [(record, score) for record, score in scored if score is not None]
    if not scored:
        raise ValueError("no scored records in selection")
    if eras is not None:
        absent = set(eras) - {record.get(era_field) for record, _ in scored}
        if absent:
            logger.warning("Eras without scored records: %s", sorted(absent))
    if any(
        not isinstance(record.get("type"), str) or not isinstance(record.get(era_field), str) for record, _ in scored
    ):
        raise ValueError("scored records require a question type and era")
    types = tuple(sorted({record["type"] for record, _ in scored}))
    era_names = tuple(sorted({record[era_field] for record, _ in scored}))
    days, day_values, standardized, observed, mean, sd = _horizons(scored)
    missing = int((~observed).sum())
    logger.info("Dropped %d records without spot peer; excluded %d records", dropped, excluded)
    return PoolingData(
        types,
        era_names,
        days,
        np.asarray([score for _, score in scored], dtype=float),
        np.asarray([types.index(record["type"]) for record, _ in scored]),
        np.asarray([era_names.index(record[era_field]) for record, _ in scored]),
        np.asarray([days.index(day) for day in day_values]),
        standardized,
        observed,
        dropped,
        missing,
        excluded,
        mean,
        sd,
    )


def _horizons(scored: list[tuple[dict, float]]) -> tuple:
    day_values = []
    log_horizons = []
    for record, _ in scored:
        resolved = parse_iso_utc((record.get("metadata") or {}).get("actual_resolve_time"))
        if resolved is None:
            raise ValueError(f"question {record['question_id']} lacks actual resolution time for day effect")
        day_values.append(resolved.date().isoformat())
        submitted = parse_iso_utc(record.get("bot_comment_created_at"))
        if submitted is None:
            log_horizons.append(np.nan)
            continue
        horizon = (resolved - submitted).total_seconds() / SECONDS_PER_DAY
        if horizon <= 0:
            raise ValueError(f"question {record['question_id']} has nonpositive horizon")
        log_horizons.append(np.log(horizon))
    days = tuple(sorted(set(day_values)))
    horizons = np.asarray(log_horizons, dtype=float)
    observed = np.isfinite(horizons)
    mean = float(horizons[observed].mean()) if observed.any() else 0.0
    sd = float(horizons[observed].std()) if observed.any() else 0.0
    standardized = np.zeros(len(horizons))
    if sd > 0:
        standardized[observed] = (horizons[observed] - mean) / sd
    missing = int((~observed).sum())
    if missing:
        logger.warning("Omitting horizon term for %d records lacking submission time", missing)
    if sd == 0:
        logger.warning("Horizon term has no variation and is omitted")
    return days, day_values, standardized, observed, mean, sd


def fit_model(data: PoolingData, *, draws: int, tune: int, chains: int, seed: int):
    with pm.Model():
        alpha = pm.Normal("alpha", 10, 20)
        effects = {}
        for name, shape in (
            ("type", len(data.types)),
            ("era", len(data.eras)),
            ("interaction", (len(data.types), len(data.eras))),
            ("day", len(data.days)),
        ):
            tau = pm.HalfNormal(f"tau_{name}", 20)
            effects[name] = pm.Deterministic(name, tau * pm.Normal(f"z_{name}", 0, 1, shape=shape))
        sigma = pm.HalfNormal("sigma_type", 40, shape=len(data.types))
        nu = pm.Gamma("nu", 2, 0.1)
        mu = (
            alpha
            + effects["type"][data.type_index]
            + effects["era"][data.era_index]
            + effects["interaction"][data.type_index, data.era_index]
            + effects["day"][data.day_index]
        )
        if data.log_horizon_sd > 0:
            beta = pm.Normal("beta", 0, 20)
            mu = mu + beta * data.standardized_horizon
        pm.StudentT("score", nu=nu, mu=mu, sigma=sigma[data.type_index], observed=data.scores)
        inference = pm.sample(
            draws=draws, tune=tune, chains=chains, random_seed=seed, target_accept=0.99, return_inferencedata=True
        )
        pm.sample_posterior_predictive(inference, var_names=["score"], random_seed=seed, extend_inferencedata=True)
    return inference


def _interval(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "interval_90": np.quantile(values, [0.05, 0.95]).tolist(),
        "interval_95": np.quantile(values, [0.025, 0.975]).tolist(),
    }


def make_report(data: PoolingData, inference, *, allow_diagnostics_warnings: bool = False) -> dict:
    diagnostics = _diagnostics(inference, allow_diagnostics_warnings=allow_diagnostics_warnings)
    posterior = inference.posterior
    alpha = posterior["alpha"].values[..., None, None]
    cells = (
        alpha
        + posterior["type"].values[..., :, None]
        + posterior["era"].values[..., None, :]
        + posterior["interaction"].values
    )
    cell_rows, contrasts = _cell_summaries(data, cells)
    predictive = inference.posterior_predictive["score"].values
    ppc_means = predictive.mean(axis=-1)
    ppc_sds = predictive.std(axis=-1)
    return {
        "counts": {
            "scored": len(data.scores),
            "missing_score": data.dropped_missing_score,
            "missing_horizon": data.missing_horizon,
            "excluded": data.excluded,
        },
        "horizon": {"log_mean": data.log_horizon_mean, "log_sd": data.log_horizon_sd},
        "cells": cell_rows,
        "contrasts": contrasts,
        "tau_interaction": _interval(posterior["tau_interaction"].values),
        "diagnostics": diagnostics,
        "posterior_predictive": {
            "observed_mean": float(data.scores.mean()),
            "observed_sd": float(data.scores.std()),
            "mean": _interval(ppc_means),
            "sd": _interval(ppc_sds),
        },
    }


def _diagnostics(inference, *, allow_diagnostics_warnings: bool) -> dict:
    summary = cast(pd.DataFrame, az.summary(inference, kind="diagnostics", round_to="none"))
    rhat = summary["r_hat"]
    ess_bulk = summary["ess_bulk"]
    ess_tail = summary["ess_tail"]
    divergences = int(inference.sample_stats["diverging"].sum())
    sample_count = int(inference.sample_stats["diverging"].size)
    invalid = bool(rhat.isna().any() or (rhat > 1.01).any() or divergences / sample_count > 0.01)
    if invalid:
        message = "Sampling diagnostics failed: R-hat missing or above 1.01, or divergences above 1%"
        if not allow_diagnostics_warnings:
            raise ValueError(message)
        logger.warning(message)
    clean_summary = summary.replace([np.inf, -np.inf], np.nan).reset_index()
    return {
        "max_r_hat": _finite_or_none(rhat.max()),
        "min_ess_bulk": _finite_or_none(ess_bulk.min()),
        "min_ess_tail": _finite_or_none(ess_tail.min()),
        "divergences": divergences,
        "draws": sample_count,
        "passed": not invalid,
        "parameters": clean_summary.astype(object).where(clean_summary.notna(), None).to_dict(orient="records"),
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _cell_summaries(data: PoolingData, cells: np.ndarray) -> tuple[list[dict], list[dict]]:
    cell_rows = []
    for type_index, type_name in enumerate(data.types):
        for era_index, era_name in enumerate(data.eras):
            mask = (data.type_index == type_index) & (data.era_index == era_index)
            n = int(mask.sum())
            raw_mean = float(data.scores[mask].mean()) if n else None
            posterior_summary = _interval(cells[..., type_index, era_index])
            cell_rows.append(
                {
                    "type": type_name,
                    "era": era_name,
                    "n": n,
                    "raw_mean": raw_mean,
                    "posterior": posterior_summary,
                    "shrinkage": None if raw_mean is None else raw_mean - posterior_summary["mean"],
                }
            )
    contrasts = []
    for era_index, era_name in enumerate(data.eras):
        for first, second in itertools.combinations(range(len(data.types)), 2):
            # Reverse lexical order so multiple_choice minus binary has the requested direction.
            values = cells[..., second, era_index] - cells[..., first, era_index]
            contrasts.append(
                {
                    "era": era_name,
                    "type": data.types[second],
                    "minus_type": data.types[first],
                    **_interval(values),
                    "probability_below_zero": float((values < 0).mean()),
                }
            )
    return cell_rows, contrasts


def render_report(report: dict) -> str:
    lines = [
        "# Type and era partial pooling",
        "",
        f"Counts: {report['counts']}",
        "",
        "Cell estimates at mean observed log horizon, with the day effect set to zero.",
        "",
    ]
    rows = []
    for cell in report["cells"]:
        posterior = cell["posterior"]
        rows.append(
            [
                cell["type"],
                cell["era"],
                cell["n"],
                cell["raw_mean"],
                posterior["mean"],
                posterior["interval_90"],
                posterior["interval_95"],
                cell["shrinkage"],
            ]
        )
    lines.extend(
        markdown_table(
            ["Type", "Era", "N", "Raw mean", "Posterior mean", "90% interval", "95% interval", "Raw minus posterior"],
            [[str(value) for value in row] for row in rows],
        )
    )
    lines.extend(["", "## Type contrasts", ""])
    rows = [
        [
            c["era"],
            c["type"],
            c["minus_type"],
            c["mean"],
            c["interval_90"],
            c["interval_95"],
            c["probability_below_zero"],
        ]
        for c in report["contrasts"]
    ]
    lines.extend(
        markdown_table(
            ["Era", "Type", "Minus type", "Mean", "90% interval", "95% interval", "P(<0)"],
            [[str(value) for value in row] for row in rows],
        )
    )
    lines.extend(
        [
            "",
            f"Interaction scale: {report['tau_interaction']}",
            "",
            f"Diagnostics: {report['diagnostics']}",
            "",
            f"Posterior predictive: {report['posterior_predictive']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--era-field", default=ERA_FIELD)
    parser.add_argument("--eras", nargs="+")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-diagnostics-warnings", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    data = prepare_data(load_dataset(args.dataset), era_field=args.era_field, eras=args.eras, strict=args.strict)
    inference = fit_model(data, draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed)
    report = make_report(data, inference, allow_diagnostics_warnings=args.allow_diagnostics_warnings)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "type_pooling.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "type_pooling.md").write_text(render_report(report))


if __name__ == "__main__":
    main()
