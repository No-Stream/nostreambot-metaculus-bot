"""Offline forward-chaining recalibration, trained only on preceding configuration eras."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import isotonic_regression, minimize
from scipy.special import expit, logsumexp

from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN, MC_PROB_MIN
from metaculus_bot.performance_analysis.eras import ERAS, NO_TS, POST_FLIP, PRE_FLIP, TRIPLE_ERA, TRIPLE_SUBERAS
from metaculus_bot.performance_analysis.loss_attribution_intervals import DEFAULT_DRAWS, DEFAULT_SEED, cluster_interval
from metaculus_bot.prob_math_utils import logit
from metaculus_bot.scoring_common import spot_peer_delta
from metaculus_bot.time_utils import parse_iso_utc

MIN_TRAINING_QUESTIONS = 50
RECALIBRATION_ERAS = (PRE_FLIP, POST_FLIP, *TRIPLE_SUBERAS)
METHODS = (("binary", "platt"), ("binary", "isotonic"), ("multiple_choice", "power"))


@dataclass(frozen=True, slots=True)
class PlattFit:
    intercept: float
    slope: float

    def predict(self, probability: float) -> float:
        return _binary_floor(float(expit(self.intercept + self.slope * logit(probability))))


@dataclass(frozen=True, slots=True)
class IsotonicFit:
    probabilities: tuple[float, ...]
    fitted: tuple[float, ...]

    def predict(self, probability: float) -> float:
        """Use the preceding fitted knot, clamping outside the training range."""
        index = max(0, bisect_right(self.probabilities, probability) - 1)
        return _binary_floor(self.fitted[index])


def _binary_floor(probability: float) -> float:
    return min(BINARY_PROB_MAX, max(BINARY_PROB_MIN, probability))


def _validate_binary(probabilities: list[float], outcomes: list[bool]) -> None:
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("Forecasts and outcomes must have equal nonzero length")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Invalid binary probability")
    if any(not isinstance(outcome, bool) for outcome in outcomes):
        raise ValueError("Binary outcomes must be bool")


def fit_binary_platt(probabilities: list[float], outcomes: list[bool]) -> PlattFit:
    """Fit unpenalized logistic maximum likelihood with an intercept and logit slope."""
    _validate_binary(probabilities, outcomes)
    if len(set(outcomes)) < 2:
        raise ValueError("Platt maximum likelihood needs both outcomes")
    logits = np.array([logit(probability) for probability in probabilities])
    observed = np.array(outcomes, dtype=float)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = parameters[0] + parameters[1] * logits
        residual = expit(linear) - observed
        return float(np.sum(np.logaddexp(0, linear) - observed * linear)), np.array(
            [np.sum(residual), np.dot(residual, logits)]
        )

    result = minimize(objective, np.array([0.0, 1.0]), jac=True, method="BFGS", options={"gtol": 1e-7})
    if not result.success:
        raise ValueError(f"Platt maximum likelihood did not converge: {result.message}")
    return PlattFit(float(result.x[0]), float(result.x[1]))


def fit_binary_isotonic(probabilities: list[float], outcomes: list[bool]) -> IsotonicFit:
    """Pool equal forecasts before weighted PAVA; retain knots for step interpolation."""
    _validate_binary(probabilities, outcomes)
    unique, inverse, counts = np.unique(probabilities, return_inverse=True, return_counts=True)
    pooled = np.bincount(inverse, weights=outcomes) / counts
    fitted = isotonic_regression(pooled, weights=counts).x
    return IsotonicFit(tuple(float(value) for value in unique), tuple(float(value) for value in fitted))


def _valid_mc(probabilities: list[float]) -> bool:
    return (
        len(probabilities) >= 2
        and all(math.isfinite(value) and 0 < value <= 1 for value in probabilities)
        and math.isclose(sum(probabilities), 1, abs_tol=1e-6)
    )


def transform_mc_power(probabilities: list[float], alpha: float) -> list[float]:
    """Raise every option to alpha, normalize, apply the production floor, then normalize."""
    if not _valid_mc(probabilities) or not math.isfinite(alpha):
        raise ValueError("Power scaling needs a probability vector and finite alpha")
    powered_logs = alpha * np.log(probabilities)
    transformed = np.exp(powered_logs - logsumexp(powered_logs))
    clamped = np.maximum(transformed, MC_PROB_MIN)
    return [float(value) for value in clamped / clamped.sum()]


def fit_mc_power(probabilities: list[list[float]], winners: list[int]) -> float:
    """Fit one unconstrained power by question-level winner log likelihood, before output floors."""
    if not probabilities or len(probabilities) != len(winners):
        raise ValueError("Ballots and winners must have equal nonzero length")
    if any(
        not _valid_mc(ballot) or not 0 <= winner < len(ballot)
        for ballot, winner in zip(probabilities, winners, strict=True)
    ):
        raise ValueError("Invalid MC ballot or winner")
    logged_ballots = [np.log(ballot) for ballot in probabilities]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        alpha = parameters[0]
        loss = 0.0
        derivative = 0.0
        for ballot, winner in zip(logged_ballots, winners, strict=True):
            scaled = alpha * ballot
            normalizer = logsumexp(scaled)
            loss += float(normalizer - scaled[winner])
            derivative += float(np.dot(np.exp(scaled - normalizer), ballot) - ballot[winner])
        return loss, np.array([derivative])

    result = minimize(objective, np.array([1.0]), jac=True, method="BFGS")
    if not result.success:
        raise ValueError(f"MC power maximum likelihood did not converge: {result.message}")
    return float(result.x[0])


@dataclass(frozen=True, slots=True)
class RecalibrationRow:
    question_type: str
    method: str
    era: str
    train_eras: tuple[str, ...]
    n_train: int
    n_test: int
    status: str
    sum_delta: float | None
    mean_delta: float | None
    interval: tuple[float, float] | None
    parameters: dict[str, float] | None


@dataclass(frozen=True, slots=True)
class PooledRecalibration:
    question_type: str
    method: str
    n_test: int
    sum_delta: float
    mean_delta: float | None
    interval: tuple[float, float] | None
    evaluated_eras: int
    skipped_eras: int
    skipped_questions: int


@dataclass(frozen=True, slots=True)
class RecalibrationResult:
    era_order: tuple[str, ...]
    rows: tuple[RecalibrationRow, ...]
    pooled: tuple[PooledRecalibration, ...]
    exclusions: dict[str, int]


@dataclass(frozen=True, slots=True)
class _Observation:
    era: str
    question_type: str
    probabilities: tuple[float, ...]
    winner: int
    cluster: str


def _record_era(record: dict) -> str | None:
    config = record.get("config_era")
    subera = record.get("triple_subera")
    if config is not None and config not in ERAS:
        raise ValueError(f"Unknown configuration era: {config}")
    if subera is not None and subera not in TRIPLE_SUBERAS:
        raise ValueError(f"Unknown triple subera: {subera}")
    if config is None or config == NO_TS:
        return None
    if config == TRIPLE_ERA:
        return subera
    if subera is not None:
        raise ValueError(f"Triple subera {subera} conflicts with configuration era {config}")
    return config


def _observations(records: list[dict]) -> tuple[list[_Observation], dict[str, int]]:
    observations: list[_Observation] = []
    exclusions: Counter[str] = Counter()
    for record in records:
        kind = record["type"]
        if kind not in {"binary", "multiple_choice"}:
            continue
        era = _record_era(record)
        if era is None:
            exclusions["missing era"] += 1
            continue
        if kind == "binary":
            probability = record.get("our_prob_yes")
            outcome = record.get("resolution_parsed")
            if probability is None or not math.isfinite(probability) or not 0 < probability < 1:
                exclusions["invalid binary forecast"] += 1
                continue
            if not isinstance(outcome, bool):
                exclusions["invalid binary outcome"] += 1
                continue
            probabilities = (float(probability), 1 - float(probability))
            winner = 0 if outcome else 1
        else:
            options = record.get("options") or []
            ballot = record.get("our_forecast_values") or []
            if len(options) != len(ballot) or not _valid_mc(ballot):
                exclusions["invalid MC forecast"] += 1
                continue
            if record.get("resolution_parsed") not in options:
                exclusions["invalid MC outcome"] += 1
                continue
            probabilities = tuple(float(value) for value in ballot)
            winner = options.index(record["resolution_parsed"])
        resolved = parse_iso_utc((record.get("metadata") or {}).get("actual_resolve_time"))
        if resolved is None:
            exclusions["missing resolution timestamp"] += 1
            continue
        observations.append(_Observation(era, kind, probabilities, winner, resolved.date().isoformat()))
    observations.sort(
        key=lambda item: (RECALIBRATION_ERAS.index(item.era), item.cluster, item.probabilities, item.winner)
    )
    return observations, dict(exclusions)


def _evaluate(
    train: list[_Observation], test: list[_Observation], method: str
) -> tuple[list[float], dict[str, float] | None]:
    binary_probabilities = [item.probabilities[0] for item in train]
    outcomes = [item.winner == 0 for item in train]
    if method == "platt":
        fitted: PlattFit | IsotonicFit = fit_binary_platt(binary_probabilities, outcomes)
        parameters = {"intercept": fitted.intercept, "slope": fitted.slope}
    elif method == "isotonic":
        fitted = fit_binary_isotonic(binary_probabilities, outcomes)
        parameters = None
    else:
        alpha = fit_mc_power([list(item.probabilities) for item in train], [item.winner for item in train])
        return [
            spot_peer_delta(
                old_prob=item.probabilities[item.winner],
                new_prob=transform_mc_power(list(item.probabilities), alpha)[item.winner],
                question_type="multiple_choice",
            )
            for item in test
        ], {"alpha": alpha}
    return [
        spot_peer_delta(
            old_prob=item.probabilities[item.winner],
            new_prob=(probability if item.winner == 0 else 1 - probability),
            question_type="binary",
        )
        for item in test
        for probability in [fitted.predict(item.probabilities[0])]
    ], parameters


def forward_chaining_recalibration(
    records: list[dict], *, seed: int = DEFAULT_SEED, draws: int = DEFAULT_DRAWS
) -> RecalibrationResult:
    """Fit preceding eras and score the next era; pool held-out question deltas per method.

    Era membership follows submission-time configuration tags. This retrospective check uses
    resolved outcomes from preceding eras; it does not reconstruct historical availability of
    those resolutions at each submission time. The caller applies source and exclusion cohorts.
    """
    observations, exclusions = _observations(records)
    rows: list[RecalibrationRow] = []
    pooled: list[PooledRecalibration] = []
    for kind, method in METHODS:
        all_deltas: list[float] = []
        all_clusters: list[str] = []
        for index, era in enumerate(RECALIBRATION_ERAS):
            previous = RECALIBRATION_ERAS[:index]
            train = [item for item in observations if item.question_type == kind and item.era in previous]
            test = [item for item in observations if item.question_type == kind and item.era == era]
            if not test:
                continue
            train_eras = tuple(label for label in previous if any(item.era == label for item in train))
            status: Literal["evaluated", "insufficient training", "single training outcome"] = "evaluated"
            if len(train) < MIN_TRAINING_QUESTIONS:
                status = "insufficient training"
            elif method == "platt" and len({item.winner for item in train}) < 2:
                status = "single training outcome"
            if status != "evaluated":
                rows.append(
                    RecalibrationRow(
                        kind, method, era, train_eras, len(train), len(test), status, None, None, None, None
                    )
                )
                continue
            deltas, parameters = _evaluate(train, test, method)
            clusters = [item.cluster for item in test]
            all_deltas.extend(deltas)
            all_clusters.extend(clusters)
            rows.append(
                RecalibrationRow(
                    kind,
                    method,
                    era,
                    train_eras,
                    len(train),
                    len(test),
                    status,
                    sum(deltas),
                    sum(deltas) / len(deltas),
                    cluster_interval(deltas, clusters, seed=seed, draws=draws),
                    parameters,
                )
            )
        pooled.append(
            PooledRecalibration(
                kind,
                method,
                len(all_deltas),
                sum(all_deltas),
                sum(all_deltas) / len(all_deltas) if all_deltas else None,
                cluster_interval(all_deltas, all_clusters, seed=seed, draws=draws),
                sum(row.method == method and row.status == "evaluated" for row in rows),
                sum(row.method == method and row.status != "evaluated" for row in rows),
                sum(row.n_test for row in rows if row.method == method and row.status != "evaluated"),
            )
        )
    return RecalibrationResult(RECALIBRATION_ERAS, tuple(rows), tuple(pooled), exclusions)
