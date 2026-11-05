"""The standing era read: spot-peer gap between two config eras, adjusted for type mix AND horizon.

Treated arm minus comparison arm in SPOT peer (the field the tournament ranks on, read through
:mod:`platform_scores`), under three controls: the pre-registered type-mix adjustment
(residualize each record on the pooled per-type mean), the same after capping the comparison
arm at the treated arm's longest submit-to-resolve lag, and a (type, lag-quintile) cell
residualization as the reweighting counterpart to the cap. Lag is ``actual_resolve_time`` minus
``bot_comment_created_at`` in days: the forecast horizon, never the batch date Metaculus set the
resolution on. Type adjustment cannot see the horizon confound, and on 2026-09-09 it moved the
headline from +10.71 [+1.72, +19.69] to +8.19 [-3.21, +19.90]. Every gap carries a cluster
bootstrap interval, one UTC day of ``actual_resolve_time`` per cluster on both arms by default or a
round's curated strong clusters via ``--clusters``. :func:`two_sided_watch` is the standing rule. Estimator derivation, the quartile receipt and the cluster convention:
``docs/performance_analysis.md`` "The era gap and the horizon confound".
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from itertools import pairwise

import numpy as np

from metaculus_bot.performance_analysis.cluster_structure import ClusterStrength
from metaculus_bot.performance_analysis.cohorts import EXCLUSION_COHORTS
from metaculus_bot.performance_analysis.collector import load_dataset
from metaculus_bot.performance_analysis.markdown import markdown_table
from metaculus_bot.performance_analysis.platform_scores import spot_peer_score
from metaculus_bot.time_utils import parse_iso_utc

logger: logging.Logger = logging.getLogger(__name__)

# Written by the round's tagging pass off bot_comment_created_at against the merge-date era map.
ERA_FIELD = "config_era"

DEFAULT_BOOTSTRAP_DRAWS = 20_000
DEFAULT_BOOTSTRAP_SEED = 20260909
CI_PERCENTILES = (2.5, 97.5)

# A concern reopens only below this many spot-peer points, with the interval excluding zero.
CONCERN_GAP_POINTS = -5.0

LAG_QUINTILE_PERCENTILES = (20, 40, 60, 80)
LAG_QUARTILE_PERCENTILES = (25, 50, 75)
SECONDS_PER_DAY = 86_400.0

WATCH_LABEL = "type-adjusted, horizon-matched"

RESOLUTION_DAY_CONVENTION = "one UTC day of actual_resolve_time per cluster"


class Verdict(StrEnum):
    CONCERN = "concern"
    FAVOURABLE = "favourable"
    NO_MEASURABLE_DIFFERENCE = "no measurable difference"


def two_sided_watch(gap: float, ci: tuple[float, float]) -> Verdict:
    """The two-sided era-difference watch, operator ruling 2026-09-09.

    ``CONCERN`` when the point estimate is below :data:`CONCERN_GAP_POINTS` (-5 spot-peer
    points) AND the 95 percent interval excludes zero. ``FAVOURABLE`` when the estimate is
    above zero and the interval excludes zero; this is reported, never flagged. Anything else
    is ``NO_MEASURABLE_DIFFERENCE``. The estimate the standing rule reads is the type-adjusted,
    horizon-matched gap under STRICT exclusions.
    """
    low, high = ci
    if gap < CONCERN_GAP_POINTS and high < 0:
        return Verdict.CONCERN
    if gap > 0 and low > 0:
        return Verdict.FAVOURABLE
    return Verdict.NO_MEASURABLE_DIFFERENCE


@dataclass(frozen=True, slots=True)
class ScoredRecord:
    question_id: str
    q_type: str
    spot: float
    lag_days: float
    cluster: str
    tournament: str | None
    cluster_labelled: bool = False


def _scored(record: dict) -> ScoredRecord | None:
    spot = spot_peer_score(record)
    submitted = parse_iso_utc(record.get("bot_comment_created_at"))
    resolved = parse_iso_utc((record.get("metadata") or {}).get("actual_resolve_time"))
    if spot is None or submitted is None or resolved is None:
        return None
    return ScoredRecord(
        question_id=str(record["question_id"]),
        q_type=record["type"],
        spot=spot,
        lag_days=(resolved - submitted).total_seconds() / SECONDS_PER_DAY,
        cluster=resolved.date().isoformat(),
        tournament=record.get("source_tournament"),
    )


@dataclass(frozen=True, slots=True)
class ClusterMap:
    """A curated question-id to cluster-id map in the round's ``cluster_structure.json`` shape.

    Reads ``qid_to_cluster`` and ``clusters[cid]["strength"]`` (strong, weak or single). Only strong
    members collapse; weak members (correlated residuals, separate draws) and unlabelled records
    are each their own cluster.
    """

    strong: dict[str, str]
    source: str

    @classmethod
    def from_cluster_structure(cls, structure: dict, *, source: str) -> ClusterMap:
        strength = {cid: cluster.get("strength") for cid, cluster in structure["clusters"].items()}
        strong = {
            qid: cid for qid, cid in structure["qid_to_cluster"].items() if strength.get(cid) == ClusterStrength.STRONG
        }
        return cls(strong=strong, source=source)

    @classmethod
    def load(cls, path: str) -> ClusterMap:
        with open(path) as f:
            return cls.from_cluster_structure(json.load(f), source=path)

    @property
    def convention(self) -> str:
        return f"curated strong clusters from {self.source}, unlabelled records their own cluster"

    def apply(self, record: ScoredRecord) -> ScoredRecord:
        cid = self.strong.get(record.question_id)
        if cid is None:
            return replace(record, cluster=f"q{record.question_id}", cluster_labelled=False)
        return replace(record, cluster=cid, cluster_labelled=True)


def in_exclusion_cohort(record: dict) -> bool:
    """Whether the record's QUESTION id sits in any standing exclusion cohort."""
    qid = str(record.get("question_id"))
    return any(qid in cohort for cohort in EXCLUSION_COHORTS.values())


@dataclass(frozen=True, slots=True)
class Arm:
    """One era's scoreable records, with the counts of what was dropped on the way in."""

    label: str
    records: tuple[ScoredRecord, ...]
    n_excluded: int
    n_unscoreable: int

    @property
    def n(self) -> int:
        return len(self.records)

    @property
    def spots(self) -> np.ndarray:
        return np.asarray([r.spot for r in self.records], dtype=float)

    @property
    def lags(self) -> np.ndarray:
        return np.asarray([r.lag_days for r in self.records], dtype=float)

    @property
    def clusters(self) -> list[str]:
        return [r.cluster for r in self.records]

    @property
    def max_lag_days(self) -> float:
        return float(self.lags.max())

    @property
    def n_cluster_labelled(self) -> int:
        return sum(1 for r in self.records if r.cluster_labelled)


def build_arm(label: str, records: Iterable[dict], *, strict: bool = False, clusters: ClusterMap | None = None) -> Arm:
    """Score an era's records; ``strict`` drops the exclusion cohorts first, ``clusters`` overrides the day key."""
    kept = list(records)
    n_excluded = 0
    if strict:
        n_before = len(kept)
        kept = [r for r in kept if not in_exclusion_cohort(r)]
        n_excluded = n_before - len(kept)
    scored = [_scored(r) for r in kept]
    usable = tuple(s for s in scored if s is not None)
    if clusters is not None:
        usable = tuple(clusters.apply(s) for s in usable)
    n_unscoreable = len(scored) - len(usable)
    if n_unscoreable:
        logger.warning(f"era arm {label}: {n_unscoreable} record(s) lack spot peer, submit time or resolve time")
    return Arm(label=label, records=usable, n_excluded=n_excluded, n_unscoreable=n_unscoreable)


def horizon_match(comparison: Arm, max_lag_days: float) -> Arm:
    """The comparison arm capped at the treated arm's longest horizon (inclusive)."""
    return Arm(
        label=f"{comparison.label}, lag <= {max_lag_days:.1f} d",
        records=tuple(r for r in comparison.records if r.lag_days <= max_lag_days),
        n_excluded=comparison.n_excluded,
        n_unscoreable=comparison.n_unscoreable,
    )


@dataclass(frozen=True, slots=True)
class ArmSummary:
    label: str
    n: int
    n_clusters: int
    n_cluster_labelled: int
    n_excluded: int
    n_unscoreable: int
    spot_mean: float
    spot_median: float
    n_negative: int
    frac_negative: float
    lag_median_days: float
    lag_max_days: float
    type_counts: dict[str, int]
    tournament_counts: dict[str, int]


def summarize_arm(arm: Arm) -> ArmSummary:
    _require_records(arm)
    spots = arm.spots
    return ArmSummary(
        label=arm.label,
        n=arm.n,
        n_clusters=len(set(arm.clusters)),
        n_cluster_labelled=arm.n_cluster_labelled,
        n_excluded=arm.n_excluded,
        n_unscoreable=arm.n_unscoreable,
        spot_mean=float(spots.mean()),
        spot_median=float(np.median(spots)),
        n_negative=int((spots < 0).sum()),
        frac_negative=float((spots < 0).mean()),
        lag_median_days=float(np.median(arm.lags)),
        lag_max_days=arm.max_lag_days,
        type_counts=dict(sorted(Counter(r.q_type for r in arm.records).items())),
        tournament_counts=dict(sorted(Counter(str(r.tournament) for r in arm.records).items())),
    )


class EmptyArmError(ValueError):
    """An era arm with no scoreable records: the read has nothing to compare and fails shut."""


def _require_records(arm: Arm) -> None:
    if not arm.records:
        raise EmptyArmError(f"era arm {arm.label!r} has no scoreable records")


@dataclass(frozen=True, slots=True)
class LagQuartileRow:
    quartile: int
    lag_low_days: float
    lag_high_days: float
    n: int
    spot_mean: float


def lag_quartiles(arm: Arm) -> list[LagQuartileRow]:
    """Spot mean by within-arm submit-to-resolve lag quartile: the horizon confound made visible."""
    _require_records(arm)
    lags, spots = arm.lags, arm.spots
    cuts = np.percentile(lags, LAG_QUARTILE_PERCENTILES)
    edges = [-np.inf, *cuts, np.inf]
    rows: list[LagQuartileRow] = []
    for quartile, (low, high) in enumerate(pairwise(edges), start=1):
        mask = (lags > low) & (lags <= high)
        if not mask.any():
            continue
        rows.append(
            LagQuartileRow(
                quartile=quartile,
                lag_low_days=float(max(low, lags.min())),
                lag_high_days=float(min(high, lags.max())),
                n=int(mask.sum()),
                spot_mean=float(spots[mask].mean()),
            )
        )
    return rows


@dataclass(frozen=True, slots=True)
class ClusterSums:
    """Per-cluster sums and counts: resampling clusters needs nothing else to rebuild a mean."""

    sums: np.ndarray
    counts: np.ndarray

    @classmethod
    def from_values(cls, values: Sequence[float], clusters: Sequence[str]) -> ClusterSums:
        by_cluster: dict[str, list[float]] = {}
        for value, cluster in zip(values, clusters, strict=True):
            by_cluster.setdefault(cluster, []).append(value)
        return cls(
            sums=np.asarray([sum(v) for v in by_cluster.values()], dtype=float),
            counts=np.asarray([len(v) for v in by_cluster.values()], dtype=float),
        )

    def resampled_means(self, rng: np.random.Generator, draws: int) -> np.ndarray:
        k = self.sums.size
        idx = rng.integers(0, k, size=(draws, k))
        return self.sums[idx].sum(axis=1) / self.counts[idx].sum(axis=1)


@dataclass(frozen=True, slots=True)
class GapEstimate:
    """A gap with its clustered interval (the verdict's input) and the by-record interval as the other bracket."""

    label: str
    n_treated: int
    n_comparison: int
    gap: float
    ci_low: float
    ci_high: float
    p_negative: float
    ci_low_by_record: float
    ci_high_by_record: float

    @property
    def verdict(self) -> Verdict:
        return two_sided_watch(self.gap, (self.ci_low, self.ci_high))

    @property
    def brackets_disagree_on_zero(self) -> bool:
        return _excludes_zero(self.ci_low, self.ci_high) != _excludes_zero(
            self.ci_low_by_record, self.ci_high_by_record
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "verdict": self.verdict, "brackets_disagree_on_zero": self.brackets_disagree_on_zero}


def _excludes_zero(low: float, high: float) -> bool:
    return low > 0 or high < 0


Residualizer = Callable[[ScoredRecord], float]


def _gap_estimate(
    label: str,
    treated: Arm,
    comparison: Arm,
    residual: Residualizer,
    *,
    draws: int,
    seed: int,
) -> GapEstimate:
    _require_records(treated)
    _require_records(comparison)
    treated_values = [residual(r) for r in treated.records]
    comparison_values = [residual(r) for r in comparison.records]
    gap = float(np.mean(treated_values) - np.mean(comparison_values))
    # Separate streams keep the by-record bracket identical under every cluster convention.
    clustered_seed, record_seed = np.random.SeedSequence(seed).spawn(2)
    clustered = _bootstrap_gaps(
        ClusterSums.from_values(treated_values, treated.clusters),
        ClusterSums.from_values(comparison_values, comparison.clusters),
        np.random.default_rng(clustered_seed),
        draws,
    )
    by_record = _bootstrap_gaps(
        ClusterSums.from_values(treated_values, _record_ids(treated)),
        ClusterSums.from_values(comparison_values, _record_ids(comparison)),
        np.random.default_rng(record_seed),
        draws,
    )
    low, high = np.percentile(clustered, CI_PERCENTILES)
    low_by_record, high_by_record = np.percentile(by_record, CI_PERCENTILES)
    return GapEstimate(
        label=label,
        n_treated=treated.n,
        n_comparison=comparison.n,
        gap=gap,
        ci_low=float(low),
        ci_high=float(high),
        p_negative=float(np.mean(clustered < 0)),
        ci_low_by_record=float(low_by_record),
        ci_high_by_record=float(high_by_record),
    )


def _record_ids(arm: Arm) -> list[str]:
    return [r.question_id for r in arm.records]


def _bootstrap_gaps(treated: ClusterSums, comparison: ClusterSums, rng: np.random.Generator, draws: int) -> np.ndarray:
    return treated.resampled_means(rng, draws) - comparison.resampled_means(rng, draws)


def _pool(treated: Arm, comparison: Arm) -> tuple[ScoredRecord, ...]:
    return treated.records + comparison.records


def _cell_means[K](records: Iterable[ScoredRecord], key: Callable[[ScoredRecord], K]) -> dict[K, float]:
    sums: dict[K, float] = {}
    counts: dict[K, int] = {}
    for r in records:
        k = key(r)
        sums[k] = sums.get(k, 0.0) + r.spot
        counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def unadjusted_gap(
    treated: Arm, comparison: Arm, *, draws: int = DEFAULT_BOOTSTRAP_DRAWS, seed: int = DEFAULT_BOOTSTRAP_SEED
) -> GapEstimate:
    """Difference of raw spot-peer means; the context every adjusted read is measured against."""
    return _gap_estimate("unadjusted", treated, comparison, lambda r: r.spot, draws=draws, seed=seed)


def type_adjusted_gap(
    treated: Arm,
    comparison: Arm,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    label: str = "type-adjusted",
    reference: Sequence[ScoredRecord] | None = None,
) -> GapEstimate:
    """The pre-registered estimator: residualize on per-type means over ``reference``, then difference the arms.

    ``reference`` defaults to the two arms pooled; the report passes the PRE-cap pool for the
    horizon-matched row so that row differs from the type-adjusted one by the cap alone.
    """
    pool = _pool(treated, comparison) if reference is None else reference
    type_means = _cell_means(pool, lambda r: r.q_type)
    return _gap_estimate(label, treated, comparison, lambda r: r.spot - type_means[r.q_type], draws=draws, seed=seed)


def type_lag_adjusted_gap(
    treated: Arm,
    comparison: Arm,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    label: str = "type x lag-quintile adjusted",
    reference: Sequence[ScoredRecord] | None = None,
) -> GapEstimate:
    """Residualize on (type, lag-quintile) cell means over ``reference``: the reweighting form of the horizon control."""
    pool = _pool(treated, comparison) if reference is None else reference
    cuts = np.percentile([r.lag_days for r in pool], LAG_QUINTILE_PERCENTILES)

    def cell(r: ScoredRecord) -> tuple[str, int]:
        return r.q_type, int(np.searchsorted(cuts, r.lag_days, side="right"))

    cell_means = _cell_means(pool, cell)
    return _gap_estimate(label, treated, comparison, lambda r: r.spot - cell_means[cell(r)], draws=draws, seed=seed)


@dataclass(frozen=True, slots=True)
class PerTypeGap:
    q_type: str
    n_treated: int
    treated_mean: float
    n_comparison: int
    comparison_mean: float
    gap: float


def per_type_gaps(treated: Arm, comparison: Arm) -> list[PerTypeGap]:
    """Raw spot-mean gap per question type, on the types both arms carry."""
    treated_means = _cell_means(treated.records, lambda r: r.q_type)
    comparison_means = _cell_means(comparison.records, lambda r: r.q_type)
    treated_counts = Counter(r.q_type for r in treated.records)
    comparison_counts = Counter(r.q_type for r in comparison.records)
    return [
        PerTypeGap(
            q_type=str(q_type),
            n_treated=treated_counts[q_type],
            treated_mean=treated_means[q_type],
            n_comparison=comparison_counts[q_type],
            comparison_mean=comparison_means[q_type],
            gap=treated_means[q_type] - comparison_means[q_type],
        )
        for q_type in sorted(treated_means.keys() & comparison_means.keys())
    ]


@dataclass(frozen=True, slots=True)
class EraGapReport:
    treated: ArmSummary
    comparison: ArmSummary
    comparison_horizon_matched: ArmSummary
    lag_quartiles: dict[str, list[LagQuartileRow]]
    gaps: list[GapEstimate]
    per_type_horizon_matched: list[PerTypeGap]
    strict: bool
    cluster_convention: str
    era_field: str
    draws: int
    seed: int

    @property
    def watch(self) -> GapEstimate:
        return next(g for g in self.gaps if g.label == WATCH_LABEL)

    def to_dict(self) -> dict:
        return {
            "treated": asdict(self.treated),
            "comparison": asdict(self.comparison),
            "comparison_horizon_matched": asdict(self.comparison_horizon_matched),
            "lag_quartiles": {label: [asdict(row) for row in rows] for label, rows in self.lag_quartiles.items()},
            "gaps": [g.to_dict() for g in self.gaps],
            "per_type_horizon_matched": [asdict(row) for row in self.per_type_horizon_matched],
            "watch": self.watch.to_dict(),
            "strict": self.strict,
            "cluster_convention": self.cluster_convention,
            "era_field": self.era_field,
            "draws": self.draws,
            "seed": self.seed,
        }


def compute_era_gap_report(
    treated: Arm,
    comparison: Arm,
    *,
    strict: bool = False,
    cluster_convention: str = RESOLUTION_DAY_CONVENTION,
    era_field: str = ERA_FIELD,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> EraGapReport:
    """Every read of the era gap on two built arms; the keyword labels record how the arms were built."""
    _require_records(treated)
    _require_records(comparison)
    matched = horizon_match(comparison, treated.max_lag_days)
    _require_records(matched)
    pool = _pool(treated, comparison)
    gaps = [
        unadjusted_gap(treated, comparison, draws=draws, seed=seed),
        type_adjusted_gap(treated, comparison, draws=draws, seed=seed),
        type_adjusted_gap(treated, matched, label=WATCH_LABEL, reference=pool, draws=draws, seed=seed),
        type_lag_adjusted_gap(treated, comparison, draws=draws, seed=seed),
        type_lag_adjusted_gap(
            treated,
            matched,
            label="type x lag-quintile adjusted, horizon-matched",
            reference=pool,
            draws=draws,
            seed=seed,
        ),
    ]
    return EraGapReport(
        treated=summarize_arm(treated),
        comparison=summarize_arm(comparison),
        comparison_horizon_matched=summarize_arm(matched),
        lag_quartiles={treated.label: lag_quartiles(treated), comparison.label: lag_quartiles(comparison)},
        gaps=gaps,
        per_type_horizon_matched=per_type_gaps(treated, matched),
        strict=strict,
        cluster_convention=cluster_convention,
        era_field=era_field,
        draws=draws,
        seed=seed,
    )


def _fmt_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in counts.items())


def _render_arms(report: EraGapReport) -> list[str]:
    header = [
        "arm",
        "n",
        "eff n (clusters)",
        "excl",
        "unscoreable",
        "spot mean",
        "spot median",
        "frac neg",
        "lag median (d)",
        "lag max (d)",
        "types",
        "tournaments",
    ]
    rows = [
        [
            s.label,
            str(s.n),
            str(s.n_clusters),
            str(s.n_excluded),
            str(s.n_unscoreable),
            f"{s.spot_mean:+.2f}",
            f"{s.spot_median:+.2f}",
            f"{s.frac_negative:.2f} ({s.n_negative}/{s.n})",
            f"{s.lag_median_days:.1f}",
            f"{s.lag_max_days:.1f}",
            _fmt_counts(s.type_counts),
            _fmt_counts(s.tournament_counts),
        ]
        for s in (report.treated, report.comparison, report.comparison_horizon_matched)
    ]
    return markdown_table(header, rows)


def _render_lag_quartiles(report: EraGapReport) -> list[str]:
    header = ["arm", "lag quartile", "lag range (d)", "n", "spot mean"]
    rows = [
        [
            label,
            f"q{row.quartile}",
            f"{row.lag_low_days:.1f} to {row.lag_high_days:.1f}",
            str(row.n),
            f"{row.spot_mean:+.2f}",
        ]
        for label, quartile_rows in report.lag_quartiles.items()
        for row in quartile_rows
    ]
    return markdown_table(header, rows)


def _render_gaps(report: EraGapReport) -> list[str]:
    header = [
        "read",
        "treated n",
        "comparison n",
        "gap",
        "95% CI (clustered)",
        "P(gap<0)",
        "95% CI (by record)",
        "verdict",
    ]
    rows = [
        [
            g.label,
            str(g.n_treated),
            str(g.n_comparison),
            f"{g.gap:+.2f}",
            f"[{g.ci_low:+.2f}, {g.ci_high:+.2f}]",
            f"{g.p_negative:.3f}",
            f"[{g.ci_low_by_record:+.2f}, {g.ci_high_by_record:+.2f}]",
            str(g.verdict),
        ]
        for g in report.gaps
    ]
    return markdown_table(header, rows)


def _render_per_type(report: EraGapReport) -> list[str]:
    header = ["type", "treated n", "treated mean", "comparison n", "comparison mean", "gap"]
    rows = [
        [
            row.q_type,
            str(row.n_treated),
            f"{row.treated_mean:+.2f}",
            str(row.n_comparison),
            f"{row.comparison_mean:+.2f}",
            f"{row.gap:+.2f}",
        ]
        for row in report.per_type_horizon_matched
    ]
    return markdown_table(header, rows)


def _render_cluster_note(report: EraGapReport) -> list[str]:
    if report.cluster_convention == RESOLUTION_DAY_CONVENTION:
        return [
            "Same-day questions need not share a world state (month-end deadlines resolve many unrelated "
            "questions together), so the clustered interval is conservative; pass `--clusters` with a round's "
            "`cluster_structure.json` to let curated strong clusters drive it. A curated interval lies between "
            "the clustered and by-record brackets."
        ]
    treated, comparison = report.treated, report.comparison
    return [
        f"Curated clusters label {treated.n_cluster_labelled}/{treated.n} treated and "
        f"{comparison.n_cluster_labelled}/{comparison.n} comparison records; the rest are their own cluster."
    ]


def _render_bracket_note(report: EraGapReport) -> list[str]:
    disagreeing = [g.label for g in report.gaps if g.brackets_disagree_on_zero]
    if not disagreeing:
        return []
    return [
        "",
        f"**Brackets disagree on zero** for: {', '.join(disagreeing)}. The clustered and by-record intervals do "
        "not agree on whether zero is inside, so the verdict on those rows depends on the cluster convention.",
    ]


def render_report(report: EraGapReport) -> str:
    policy = (
        "STRICT: exclusion cohorts dropped from both arms"
        if report.strict
        else "all records: exclusion cohorts kept in both arms"
    )
    watch = report.watch
    lines = [
        f"# Era gap: {report.treated.label} vs {report.comparison.label} ({policy})",
        "",
        f"Arms selected on `{report.era_field}`. "
        "Spot peer throughout (`platform_scores.spot_peer_score`), treated minus comparison. Lag is "
        "`actual_resolve_time` minus `bot_comment_created_at` in days, the forecast horizon. Clusters: "
        f"{report.cluster_convention}; `eff n` counts them. Bootstrap: {report.draws} draws, seed {report.seed}, "
        "both arms resampled by cluster for the interval the verdict reads, and by record for the other bracket. "
        "Intervals are the 2.5 and 97.5 percentiles.",
        "",
        *_render_cluster_note(report),
        "",
        "## Arms",
        "",
        *_render_arms(report),
        "",
        "## Spot mean by within-arm submit-to-resolve lag quartile",
        "",
        "A comparison arm whose mean falls across its quartiles was asked longer-horizon questions than the "
        "treated arm could have been; type adjustment cannot see that.",
        "",
        *_render_lag_quartiles(report),
        "",
        "## Gap under each control",
        "",
        *_render_gaps(report),
        *_render_bracket_note(report),
        "",
        f"**Standing watch** ({WATCH_LABEL}): **{watch.verdict}** at {watch.gap:+.2f} "
        f"[{watch.ci_low:+.2f}, {watch.ci_high:+.2f}]. Rule: a concern reopens only when the point estimate is "
        f"below {CONCERN_GAP_POINTS:+.0f} with an interval excluding zero; a favourable gap with an interval "
        "excluding zero is reported, never flagged; anything else is no measurable difference.",
        "",
        "## Per-type gap against the horizon-matched comparison arm",
        "",
        *_render_per_type(report),
    ]
    return "\n".join(lines)


def _era_records(data: list[dict], eras: Sequence[str], field: str) -> list[dict]:
    return [r for r in data if r.get(field) in eras]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Type-and-horizon-adjusted spot-peer gap between two config eras (read-only, offline)"
    )
    parser.add_argument(
        "--dataset", required=True, help=f"A tagged performance dataset JSON whose records carry `{ERA_FIELD}`."
    )
    parser.add_argument(
        "--treated-era", required=True, nargs="+", help="One or more eras under evaluation (the live roster)."
    )
    parser.add_argument("--comparison-era", required=True, nargs="+", help="One or more eras it is measured against.")
    parser.add_argument(
        "--era-field",
        default=ERA_FIELD,
        help=(
            "The record field carrying both arms' era labels. The tagging pass writes coarse eras to "
            "`config_era` and sub-eras to their own fields (`triple_subera`, `triple_subera_fine`), so a "
            "sub-era arm needs its field named. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Drop the exclusion cohorts (known_bug, degraded_run, partial_degraded) from BOTH arms.",
    )
    parser.add_argument(
        "--clusters",
        default=None,
        help=(
            "Optional curated cluster map in the round's cluster_structure.json shape (qid_to_cluster plus "
            "clusters[cid].strength). Strong clusters collapse to one bootstrap draw; weak and unlabelled "
            f"records stay their own cluster. Default: {RESOLUTION_DAY_CONVENTION}."
        ),
    )
    parser.add_argument("--output-json", default=None, help="Optional path to also write every number as JSON.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
    data = load_dataset(args.dataset)
    cluster_map = ClusterMap.load(args.clusters) if args.clusters else None
    convention = cluster_map.convention if cluster_map else RESOLUTION_DAY_CONVENTION
    treated_label = "+".join(args.treated_era)
    comparison_label = "+".join(args.comparison_era)
    treated = build_arm(
        treated_label,
        _era_records(data, args.treated_era, args.era_field),
        strict=args.strict,
        clusters=cluster_map,
    )
    comparison = build_arm(
        comparison_label,
        _era_records(data, args.comparison_era, args.era_field),
        strict=args.strict,
        clusters=cluster_map,
    )
    try:
        report = compute_era_gap_report(
            treated, comparison, strict=args.strict, cluster_convention=convention, era_field=args.era_field
        )
    except EmptyArmError as exc:
        # The fall read is run before its first treated question resolves; name what the field does hold.
        present = dict(Counter(str(r.get(args.era_field)) for r in data).most_common())
        sys.exit(f"{exc}; nothing to compare yet. `{args.era_field}` values in the dataset: {_fmt_counts(present)}.")

    # Logging is pinned to stderr above so the rendered report can be piped on its own.
    print(render_report(report))  # noqa: T201

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Wrote the era-gap report to {args.output_json}")


if __name__ == "__main__":
    main()
