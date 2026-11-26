"""Generic hierarchical analysis of paired score differences."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal, cast

import numpy as np
import pymc as pm

from metaculus_bot.performance_analysis.posterior_summary import interval_summary, sampling_diagnostics

Likelihood = Literal["student_t", "normal"]
EFFECT_PRIOR_SD = 5.0
OFFSET_SCALE_PRIOR_SD = 5.0
NOISE_SCALE_PRIOR_SD = 15.0
# Paired differences carry a point mass at exactly zero (both arms gave the same forecast). Below nu = 1 the Student-t
# likelihood of those zeros grows without bound as sigma shrinks, and the sampler collapses onto sigma = 0.
STUDENT_T_NU_FLOOR = 1.0
FIT_DISAGREEMENT_POINTS = 1.0


@dataclass(frozen=True)
class PairedScoreRow:
    unit_id: str
    cluster_id: str
    group: str
    replicate: int
    difference: float
    member: str | None = None


@dataclass(frozen=True)
class PairedData:
    units: tuple[str, ...]
    clusters: tuple[str, ...]
    groups: tuple[str, ...]
    members: tuple[str, ...]
    unit_index: np.ndarray
    cluster_index: np.ndarray
    group_index: np.ndarray
    member_index: np.ndarray | None
    differences: np.ndarray


def prepare_paired_data(rows: Sequence[PairedScoreRow]) -> PairedData:
    if not rows:
        raise ValueError("paired data requires at least one row")
    has_members = [row.member is not None for row in rows]
    if any(has_members) and not all(has_members):
        raise ValueError("rows cannot mix member and non-member values")
    unit_cluster: dict[str, str] = {}
    unit_group: dict[str, str] = {}
    seen: set[tuple[str, int, str | None]] = set()
    for row in rows:
        if not math.isfinite(row.difference):
            raise ValueError(f"non-finite difference for unit {row.unit_id}")
        if row.unit_id in unit_cluster and unit_cluster[row.unit_id] != row.cluster_id:
            raise ValueError(f"unit {row.unit_id} belongs to multiple clusters")
        if row.unit_id in unit_group and unit_group[row.unit_id] != row.group:
            raise ValueError(f"unit {row.unit_id} belongs to multiple groups")
        unit_cluster[row.unit_id] = row.cluster_id
        unit_group[row.unit_id] = row.group
        key = (row.unit_id, row.replicate, row.member)
        if key in seen:
            raise ValueError(f"repeated unit, replicate, member row: {key}")
        seen.add(key)
    units = tuple(sorted(unit_cluster))
    clusters = tuple(sorted({row.cluster_id for row in rows}))
    groups = tuple(sorted({row.group for row in rows}))
    members = tuple(sorted({row.member for row in rows if row.member is not None}))
    return PairedData(
        units=units,
        clusters=clusters,
        groups=groups,
        members=members,
        unit_index=np.asarray([units.index(row.unit_id) for row in rows], dtype=int),
        cluster_index=np.asarray([clusters.index(row.cluster_id) for row in rows], dtype=int),
        group_index=np.asarray([groups.index(row.group) for row in rows], dtype=int),
        member_index=np.asarray([members.index(row.member) for row in rows], dtype=int) if all(has_members) else None,
        differences=np.asarray([row.difference for row in rows], dtype=float),
    )


def build_model(data: PairedData, *, likelihood: Likelihood, effect_prior_sd: float = EFFECT_PRIOR_SD) -> pm.Model:
    if likelihood not in ("student_t", "normal"):
        raise ValueError(f"unknown likelihood: {likelihood}")
    coords: dict[str, tuple[str, ...]] = {
        "group_level": data.groups,
        "cluster_level": data.clusters,
        "unit_level": data.units,
        "observation": tuple(str(index) for index in range(len(data.differences))),
    }
    if data.member_index is not None:
        coords["member_level"] = data.members
    with pm.Model(coords=coords) as model:
        effect = pm.Normal("effect", 0, effect_prior_sd)
        offsets: dict[str, Any] = {}
        for name in ("group", "cluster", "unit"):
            tau = pm.HalfNormal(f"tau_{name}", OFFSET_SCALE_PRIOR_SD)
            dimension = f"{name}_level"
            offsets[name] = pm.Deterministic(name, tau * pm.Normal(f"z_{name}", 0, 1, dims=dimension), dims=dimension)
        if data.member_index is not None:
            tau = pm.HalfNormal("tau_member", OFFSET_SCALE_PRIOR_SD)
            offsets["member"] = cast(
                Any,
                pm.Deterministic("member", tau * pm.Normal("z_member", 0, 1, dims="member_level"), dims="member_level"),
            )
        mu = effect + offsets["group"][data.group_index] + offsets["cluster"][data.cluster_index]
        mu = mu + offsets["unit"][data.unit_index]
        if data.member_index is not None:
            mu = mu + offsets["member"][data.member_index]
        sigma = pm.HalfNormal("sigma", NOISE_SCALE_PRIOR_SD)
        if likelihood == "student_t":
            nu = pm.Deterministic("nu", STUDENT_T_NU_FLOOR + pm.Gamma("nu_above_floor", 2, 0.1))
            pm.StudentT("difference", nu=nu, mu=mu, sigma=sigma, observed=data.differences, dims="observation")
        else:
            pm.Normal("difference", mu=mu, sigma=sigma, observed=data.differences, dims="observation")
    return model


def sample_prior(data: PairedData, *, likelihood: Likelihood, draws: int, seed: int) -> Any:
    model = build_model(data, likelihood=likelihood)
    with model:
        return pm.sample_prior_predictive(draws=draws, random_seed=seed, return_inferencedata=True)


def fit_model(
    data: PairedData,
    *,
    likelihood: Likelihood,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    effect_prior_sd: float = EFFECT_PRIOR_SD,
) -> Any:
    model = build_model(data, likelihood=likelihood, effect_prior_sd=effect_prior_sd)
    with model:
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            target_accept=0.99,
            return_inferencedata=True,
        )


def _effect_summary(values: np.ndarray, thresholds: Sequence[float]) -> dict[str, Any]:
    result = interval_summary(values)
    result["p_above_zero"] = float((values > 0).mean())
    result["p_gain_above"] = {str(threshold): float((values > threshold).mean()) for threshold in thresholds}
    result["p_loss_above"] = {str(threshold): float((values < -threshold).mean()) for threshold in thresholds}
    result["draws"] = values.ravel().tolist()
    return result


def summarize_fit(
    data: PairedData,
    inference: Any,
    *,
    likelihood: Likelihood,
    thresholds: Sequence[float] = (2.0,),
    allow_diagnostics_warnings: bool = False,
) -> dict[str, Any]:
    diagnostics = sampling_diagnostics(inference, allow_diagnostics_warnings=allow_diagnostics_warnings)
    posterior = inference.posterior
    effect_values = posterior["effect"].values
    result: dict[str, Any] = {"likelihood": likelihood, "effect": _effect_summary(effect_values, thresholds)}
    result["groups"] = {
        name: interval_summary(effect_values + posterior["group"].sel(group_level=name).values) for name in data.groups
    }
    result["members"] = (
        {
            name: interval_summary(effect_values + posterior["member"].sel(member_level=name).values)
            for name in data.members
        }
        if data.member_index is not None
        else {}
    )
    scales = [
        name for name in ("tau_group", "tau_cluster", "tau_unit", "tau_member", "sigma", "nu") if name in posterior
    ]
    result["scales"] = {name: interval_summary(posterior[name].values) for name in scales}
    result["variance_between"] = float(
        (posterior["tau_cluster"].values ** 2 + posterior["tau_unit"].values ** 2).mean()
    )
    result["tau_cluster_squared"] = float((posterior["tau_cluster"].values ** 2).mean())
    result["tau_unit_squared"] = float((posterior["tau_unit"].values ** 2).mean())
    result["variance_within"] = float((posterior["sigma"].values ** 2).mean())
    result["diagnostics"] = diagnostics
    return result


def bayesian_bootstrap(
    rows: Sequence[PairedScoreRow], *, draws: int = 20_000, seed: int, thresholds: Sequence[float] = (2.0,)
) -> dict[str, Any]:
    data = prepare_paired_data(rows)
    if data.member_index is None:
        grouped: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            grouped.setdefault((row.unit_id, row.cluster_id), []).append(row.difference)
        units = [(cluster, float(np.mean(values))) for (_, cluster), values in grouped.items()]
    else:
        grouped_member: dict[tuple[str, str, str], list[float]] = {}
        for row in rows:
            grouped_member.setdefault((row.unit_id, row.cluster_id, row.member or ""), []).append(row.difference)
        units = [(cluster, float(np.mean(values))) for (_, cluster, _), values in grouped_member.items()]
    cluster_names = tuple(sorted({cluster for cluster, _ in units}))
    cluster_indices = np.asarray([cluster_names.index(cluster) for cluster, _ in units], dtype=np.intp)
    unit_means = np.asarray([value for _, value in units])
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(cluster_names)), size=draws)
    unit_weights = np.take(weights, cluster_indices, axis=1)
    draw_values = (unit_weights @ unit_means) / unit_weights.sum(axis=1)
    return {"effect": _effect_summary(draw_values, thresholds)}


def leave_one_cluster_out(rows: Sequence[PairedScoreRow]) -> list[dict[str, Any]]:
    prepare_paired_data(rows)
    by_unit: dict[str, list[float]] = {}
    unit_clusters: dict[str, str] = {}
    for row in rows:
        by_unit.setdefault(row.unit_id, []).append(row.difference)
        unit_clusters[row.unit_id] = row.cluster_id
    unit_means = {unit_id: float(np.mean(values)) for unit_id, values in by_unit.items()}
    by_cluster: dict[str, list[float]] = {}
    for unit_id, mean in unit_means.items():
        by_cluster.setdefault(unit_clusters[unit_id], []).append(mean)
    if len(by_cluster) < 2:
        raise ValueError("leave-one-cluster-out requires at least two clusters")
    overall = float(np.mean(list(unit_means.values())))
    output = []
    for cluster in by_cluster:
        remaining = [mean for unit_id, mean in unit_means.items() if unit_clusters[unit_id] != cluster]
        without = float(np.mean(remaining))
        output.append(
            {
                "cluster": cluster,
                "units": len(by_cluster[cluster]),
                "mean_without": without,
                "shift": without - overall,
            }
        )
    return sorted(output, key=lambda row: abs(row["shift"]), reverse=True)


def influential_units(rows: Sequence[PairedScoreRow], *, top: int = 10) -> list[dict[str, Any]]:
    prepare_paired_data(rows)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.unit_id, []).append(row.difference)
    result = [{"unit_id": unit_id, "mean": float(np.mean(values))} for unit_id, values in grouped.items()]
    return sorted(result, key=lambda row: abs(row["mean"]), reverse=True)[:top]


def sample_size_statement(
    tau_cluster_squared: float,
    tau_unit_squared: float,
    variance_within: float,
    mean_units_per_cluster: float,
    *,
    replicates_per_unit: int = 2,
    half_width: float = 2.0,
    interval: float = 0.90,
) -> dict[str, Any]:
    z = NormalDist().inv_cdf((1 + interval) / 2)

    def required(replicates: int) -> int:
        variance_of_unit_mean = (
            mean_units_per_cluster * tau_cluster_squared + tau_unit_squared + variance_within / replicates
        )
        return math.ceil(z**2 * variance_of_unit_mean / half_width**2)

    return {
        "required_units": required(replicates_per_unit),
        "replicates_per_unit": replicates_per_unit,
        "required_units_at_1_replicate": required(1),
        "required_units_at_4_replicates": required(4),
        "tau_cluster_squared": tau_cluster_squared,
        "tau_unit_squared": tau_unit_squared,
        "mean_units_per_cluster": mean_units_per_cluster,
        "half_width": half_width,
        "interval": interval,
    }


def compare_paired(
    rows: Sequence[PairedScoreRow],
    *,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    seed: int = 20261125,
    thresholds: Sequence[float] = (2.0,),
    allow_diagnostics_warnings: bool = False,
) -> dict[str, Any]:
    data = prepare_paired_data(rows)
    student_t = summarize_fit(
        data,
        fit_model(data, likelihood="student_t", draws=draws, tune=tune, chains=chains, seed=seed),
        likelihood="student_t",
        thresholds=thresholds,
        allow_diagnostics_warnings=allow_diagnostics_warnings,
    )
    normal = summarize_fit(
        data,
        fit_model(data, likelihood="normal", draws=draws, tune=tune, chains=chains, seed=seed + 1),
        likelihood="normal",
        thresholds=thresholds,
        allow_diagnostics_warnings=allow_diagnostics_warnings,
    )
    mean_difference = abs(student_t["effect"]["mean"] - normal["effect"]["mean"])
    opposite_probability = (student_t["effect"]["p_above_zero"] - 0.5) * (normal["effect"]["p_above_zero"] - 0.5) < 0
    return {
        "n_rows": len(rows),
        "n_units": len(data.units),
        "n_clusters": len(data.clusters),
        "student_t": student_t,
        "normal": normal,
        "bootstrap": bayesian_bootstrap(rows, draws=20_000, seed=seed, thresholds=thresholds),
        "leave_one_cluster_out": leave_one_cluster_out(rows),
        "fits_disagree": mean_difference > FIT_DISAGREEMENT_POINTS or opposite_probability,
        "influential_units": influential_units(rows),
        "sample_size": sample_size_statement(
            normal["tau_cluster_squared"],
            normal["tau_unit_squared"],
            normal["variance_within"],
            len(data.units) / len(data.clusters),
        ),
    }
