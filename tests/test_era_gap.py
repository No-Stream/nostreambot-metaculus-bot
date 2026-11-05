"""Tests for the standing era-gap read: type-and-horizon-adjusted spot peer between two eras.

The estimator is pinned on synthetic arms where the answer is known by construction: a
comparison arm whose score falls with the submit-to-resolve lag, so the type-adjusted gap is
positive while the horizon-matched gap is exactly zero; a type mix that inflates the raw gap
while the within-type gap is zero; and clusters that collapse the bootstrap when every record
shares one resolution day.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from metaculus_bot.performance_analysis.cohorts import DEGRADED_RUN_QIDS, KNOWN_BUG_QIDS
from metaculus_bot.performance_analysis.era_gap import (
    CONCERN_GAP_POINTS,
    ERA_FIELD,
    RESOLUTION_DAY_CONVENTION,
    Arm,
    ClusterMap,
    EmptyArmError,
    EraGapReport,
    Verdict,
    build_arm,
    compute_era_gap_report,
    horizon_match,
    lag_quartiles,
    main,
    per_type_gaps,
    render_report,
    summarize_arm,
    two_sided_watch,
    type_adjusted_gap,
    type_lag_adjusted_gap,
    unadjusted_gap,
)
from metaculus_bot.time_utils import parse_iso_utc

SUBMITTED = "2026-07-01T12:00:00Z"
FAST_DRAWS = 400


def _record(
    qid: int,
    *,
    spot: float | None,
    lag_days: float,
    era: str = "treated",
    q_type: str = "binary",
    submitted: str | None = SUBMITTED,
    tournament: str = "summer-futureeval-2026",
    resolution_set_time: str | None = None,
) -> dict:
    """A collector-shaped record trimmed to the fields the era read touches."""
    resolved = parse_iso_utc(SUBMITTED)
    assert resolved is not None
    resolved += timedelta(days=lag_days)
    metadata = {"actual_resolve_time": resolved.isoformat().replace("+00:00", "Z")}
    if resolution_set_time is not None:
        metadata["resolution_set_time"] = resolution_set_time
    return {
        "question_id": qid,
        "post_id": qid,
        "type": q_type,
        ERA_FIELD: era,
        "bot_comment_created_at": submitted,
        "metadata": metadata,
        "metaculus_scores": {"spot_peer_score": spot},
        "source_tournament": tournament,
    }


def _cluster_structure(strong: dict[str, list[int]], weak: dict[str, list[int]]) -> dict:
    """The round's cluster_structure.json shape: qid_to_cluster plus clusters[cid].strength."""
    qid_to_cluster: dict[str, str] = {}
    clusters: dict[str, dict] = {}
    for strength, groups in (("strong", strong), ("weak", weak)):
        for cid, qids in groups.items():
            clusters[cid] = {"cluster_id": cid, "strength": strength}
            qid_to_cluster.update({str(q): cid for q in qids})
    return {"qid_to_cluster": qid_to_cluster, "clusters": clusters}


def _score_falls_with_lag(lag_days: float) -> float:
    return 40.0 - 0.3 * lag_days


def _horizon_confounded_arms() -> tuple[Arm, Arm]:
    """Comparison lags run 2..120 days and treated 2..40, with an identical lag-to-score law.

    Capping the comparison arm at the treated arm's maximum lag leaves exactly the treated
    arm's lag set, so the horizon-matched gap is zero by construction while the unmatched one
    is +12 (the comparison arm carries 61 mean lag days against 21).
    """
    comparison = [
        _record(1000 + i, spot=_score_falls_with_lag(lag), lag_days=lag, era="comparison")
        for i, lag in enumerate(range(2, 121, 2))
    ]
    treated = [
        _record(2000 + i, spot=_score_falls_with_lag(lag), lag_days=lag) for i, lag in enumerate(range(2, 41, 2))
    ]
    return build_arm("treated", treated), build_arm("comparison", comparison)


class TestBuildArm:
    def test_scores_lag_and_cluster_off_the_record(self):
        arm = build_arm("t", [_record(1, spot=12.5, lag_days=3.5)])
        (scored,) = arm.records
        assert scored.question_id == "1"
        assert scored.spot == pytest.approx(12.5)
        assert scored.lag_days == pytest.approx(3.5)
        # Submitted 2026-07-01T12:00Z plus 3.5 days resolves on the 5th, the cluster key.
        assert scored.cluster == "2026-07-05"

    def test_unscoreable_records_are_dropped_and_counted_not_zeroed(self):
        records = [
            _record(1, spot=10.0, lag_days=1),
            _record(2, spot=None, lag_days=1),
            _record(3, spot=10.0, lag_days=1, submitted=None),
        ]
        arm = build_arm("t", records)
        assert arm.n == 1
        assert arm.n_unscoreable == 2

    def test_strict_drops_every_exclusion_cohort_by_question_id(self):
        bug_qid = int(next(iter(KNOWN_BUG_QIDS)))
        degraded_qid = int(next(iter(DEGRADED_RUN_QIDS)))
        records = [
            _record(bug_qid, spot=-40.0, lag_days=1),
            _record(degraded_qid, spot=-100.0, lag_days=2),
            _record(1, spot=10.0, lag_days=3),
        ]
        assert build_arm("t", records).n == 3
        strict = build_arm("t", records, strict=True)
        assert strict.n == 1
        assert strict.n_excluded == 2

    def test_missing_type_raises_rather_than_filing_the_record_nowhere(self):
        record = _record(1, spot=1.0, lag_days=1)
        del record["type"]
        with pytest.raises(KeyError):
            build_arm("t", [record])

    def test_cluster_key_is_the_event_day_not_the_batch_day(self):
        """Metaculus writes resolutions in batches; the shared world state is the event, so the key is
        ``actual_resolve_time`` and ``resolution_set_time`` is ignored entirely."""
        same_event_day = [
            _record(1, spot=1.0, lag_days=3, resolution_set_time="2026-07-10T00:00:00Z"),
            _record(2, spot=1.0, lag_days=3.2, resolution_set_time="2026-07-20T00:00:00Z"),
        ]
        same_batch_day = [
            _record(3, spot=1.0, lag_days=3, resolution_set_time="2026-07-20T00:00:00Z"),
            _record(4, spot=1.0, lag_days=9, resolution_set_time="2026-07-20T00:00:00Z"),
        ]
        assert summarize_arm(build_arm("event", same_event_day)).n_clusters == 1
        assert summarize_arm(build_arm("batch", same_batch_day)).n_clusters == 2


class TestClusterMap:
    def test_only_strong_clusters_collapse(self):
        structure = _cluster_structure(strong={"outbreak": [1, 2, 3]}, weak={"regime": [4, 5]})
        cluster_map = ClusterMap.from_cluster_structure(structure, source="test")
        # Six records on one event day: the day key would make them ONE cluster.
        records = [_record(i, spot=float(i), lag_days=5) for i in range(1, 7)]
        arm = build_arm("t", records, clusters=cluster_map)
        clusters = {r.question_id: r.cluster for r in arm.records}
        assert clusters["1"] == clusters["2"] == clusters["3"] == "outbreak"
        assert clusters["4"] != clusters["5"]  # weak: correlated residuals, still separate draws
        assert clusters["6"] == "q6"  # unlabelled: its own cluster
        assert arm.n_cluster_labelled == 3
        assert summarize_arm(arm).n_clusters == 4
        assert summarize_arm(build_arm("day", records)).n_clusters == 1

    def test_curated_strong_clusters_drive_the_interval(self):
        """Every record on its own event day gives a wide day-keyed interval; one strong cluster per arm
        collapses it onto the point, so the map, not the day key, is what the bootstrap resamples."""
        treated = [_record(i, spot=float(10 + (i % 2) * 20), lag_days=1 + i) for i in range(8)]
        comparison = [_record(100 + i, spot=float((i % 2) * 20), lag_days=1 + i) for i in range(8)]
        structure = _cluster_structure(strong={"t": list(range(8)), "c": list(range(100, 108))}, weak={})
        cluster_map = ClusterMap.from_cluster_structure(structure, source="test")
        by_day = unadjusted_gap(build_arm("t", treated), build_arm("c", comparison), draws=FAST_DRAWS, seed=1)
        curated = unadjusted_gap(
            build_arm("t", treated, clusters=cluster_map),
            build_arm("c", comparison, clusters=cluster_map),
            draws=FAST_DRAWS,
            seed=1,
        )
        assert by_day.ci_low < by_day.gap < by_day.ci_high
        assert curated.ci_low == pytest.approx(curated.gap)
        assert curated.ci_high == pytest.approx(curated.gap)
        assert curated.ci_low_by_record == pytest.approx(by_day.ci_low_by_record)

    def test_load_reads_the_round_file_shape(self, tmp_path):
        path = tmp_path / "cluster_structure.json"
        path.write_text(json.dumps(_cluster_structure(strong={"s": [1]}, weak={"w": [2]})))
        cluster_map = ClusterMap.load(str(path))
        assert cluster_map.strong == {"1": "s"}
        assert str(path) in cluster_map.convention


class TestArmSummary:
    def test_effective_n_counts_distinct_resolution_days(self):
        same_day = [_record(i, spot=float(i), lag_days=5) for i in range(4)]
        spread = [_record(10 + i, spot=float(i), lag_days=5 + i) for i in range(4)]
        assert summarize_arm(build_arm("same", same_day)).n_clusters == 1
        assert summarize_arm(build_arm("spread", spread)).n_clusters == 4

    def test_distribution_fields(self):
        records = [_record(i, spot=s, lag_days=lag) for i, (s, lag) in enumerate([(-10.0, 1), (5.0, 3), (20.0, 9)])]
        summary = summarize_arm(build_arm("t", records))
        assert summary.n == 3
        assert summary.spot_mean == pytest.approx(5.0)
        assert summary.spot_median == pytest.approx(5.0)
        assert summary.n_negative == 1
        assert summary.frac_negative == pytest.approx(1 / 3)
        assert summary.lag_median_days == pytest.approx(3.0)
        assert summary.lag_max_days == pytest.approx(9.0)
        assert summary.type_counts == {"binary": 3}
        assert summary.tournament_counts == {"summer-futureeval-2026": 3}


class TestHorizonConfound:
    def test_type_adjusted_gap_is_positive_but_horizon_matched_gap_is_zero(self):
        treated, comparison = _horizon_confounded_arms()
        unmatched = type_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=1)
        assert unmatched.gap == pytest.approx(12.0)
        assert unmatched.ci_low > 0

        matched_arm = horizon_match(comparison, treated.max_lag_days)
        assert matched_arm.n == treated.n
        matched = type_adjusted_gap(treated, matched_arm, draws=FAST_DRAWS, seed=1)
        assert matched.gap == pytest.approx(0.0, abs=1e-9)
        assert matched.ci_low < 0 < matched.ci_high

    def test_lag_quintile_adjustment_attenuates_the_confound_without_capping(self):
        """Stratifying on five lag bins is coarse: it removes the between-bin trend and leaves the
        within-bin slope, so the gap shrinks from +12 to under a point rather than to zero."""
        treated, comparison = _horizon_confounded_arms()
        unmatched = type_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=1)
        adjusted = type_lag_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=1)
        assert abs(adjusted.gap) < 0.1 * unmatched.gap
        assert adjusted.ci_low < 0 < adjusted.ci_high

    def test_horizon_match_keeps_a_record_exactly_at_the_cap(self):
        comparison = build_arm("c", [_record(1, spot=1.0, lag_days=10), _record(2, spot=1.0, lag_days=10.5)])
        assert horizon_match(comparison, 10.0).n == 1

    def test_comparison_arm_lag_quartiles_fall_monotonically(self):
        _, comparison = _horizon_confounded_arms()
        rows = lag_quartiles(comparison)
        assert [row.quartile for row in rows] == [1, 2, 3, 4]
        assert sum(row.n for row in rows) == comparison.n
        means = [row.spot_mean for row in rows]
        assert means == sorted(means, reverse=True)


class TestTypeMix:
    def test_raw_gap_from_type_mix_alone_vanishes_under_type_adjustment(self):
        """Treated is half numeric (scoring 30), comparison a quarter; within type both arms tie."""

        def arm(label: str, n_binary: int, n_numeric: int, base: int) -> Arm:
            records = [_record(base + i, spot=10.0, lag_days=1 + i, q_type="binary") for i in range(n_binary)]
            records += [_record(base + 50 + i, spot=30.0, lag_days=1 + i, q_type="numeric") for i in range(n_numeric)]
            return build_arm(label, records)

        treated = arm("t", 10, 10, 100)
        comparison = arm("c", 15, 5, 300)
        assert unadjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=3).gap == pytest.approx(5.0)
        assert type_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=3).gap == pytest.approx(0.0, abs=1e-9)

    def test_per_type_gaps_only_where_both_arms_carry_the_type(self):
        treated = build_arm(
            "t",
            [_record(1, spot=20.0, lag_days=1), _record(2, spot=5.0, lag_days=2, q_type="numeric")],
        )
        comparison = build_arm(
            "c",
            [_record(3, spot=10.0, lag_days=1), _record(4, spot=0.0, lag_days=2, q_type="discrete")],
        )
        rows = per_type_gaps(treated, comparison)
        assert [row.q_type for row in rows] == ["binary"]
        assert rows[0].gap == pytest.approx(10.0)
        assert (rows[0].n_treated, rows[0].n_comparison) == (1, 1)


class TestBootstrap:
    def test_same_seed_reproduces_the_interval_and_a_new_seed_moves_it(self):
        treated, comparison = _horizon_confounded_arms()
        first = type_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=7)
        again = type_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=7)
        other = type_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=8)
        assert (first.ci_low, first.ci_high, first.p_negative) == (again.ci_low, again.ci_high, again.p_negative)
        assert (first.ci_low, first.ci_high) != (other.ci_low, other.ci_high)
        assert first.ci_low <= first.gap <= first.ci_high
        assert 0.0 <= first.p_negative <= 1.0

    def test_one_cluster_per_arm_collapses_the_interval_onto_the_point(self):
        """Resampling a single cluster returns the same records every draw, so the CI has no width.

        Pins that the bootstrap resamples CLUSTERS rather than records: eight records on one
        resolution day would otherwise give a wide interval.
        """
        treated = build_arm("t", [_record(i, spot=float(10 + i), lag_days=5) for i in range(8)])
        comparison = build_arm("c", [_record(100 + i, spot=float(i), lag_days=5) for i in range(8)])
        estimate = unadjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=1)
        assert estimate.ci_low == pytest.approx(estimate.gap)
        assert estimate.ci_high == pytest.approx(estimate.gap)


class TestTwoSidedWatch:
    def test_concern_needs_a_point_below_the_threshold_and_an_interval_excluding_zero(self):
        assert two_sided_watch(-8.0, (-15.0, -1.0)) is Verdict.CONCERN
        assert two_sided_watch(-8.0, (-15.0, 2.0)) is Verdict.NO_MEASURABLE_DIFFERENCE
        assert two_sided_watch(-3.0, (-5.0, -1.0)) is Verdict.NO_MEASURABLE_DIFFERENCE
        assert two_sided_watch(CONCERN_GAP_POINTS, (-9.0, -1.0)) is Verdict.NO_MEASURABLE_DIFFERENCE

    def test_favourable_is_reported_not_flagged(self):
        assert two_sided_watch(10.71, (1.72, 19.69)) is Verdict.FAVOURABLE
        assert two_sided_watch(8.19, (-3.21, 19.90)) is Verdict.NO_MEASURABLE_DIFFERENCE

    def test_a_positive_gap_never_reads_as_concern_whatever_its_size(self):
        assert two_sided_watch(30.0, (20.0, 40.0)) is Verdict.FAVOURABLE


class TestReport:
    def test_report_carries_every_read_and_the_watch_row(self):
        treated, comparison = _horizon_confounded_arms()
        report = compute_era_gap_report(treated, comparison, draws=FAST_DRAWS, seed=1)
        assert isinstance(report, EraGapReport)
        assert [g.label for g in report.gaps] == [
            "unadjusted",
            "type-adjusted",
            "type-adjusted, horizon-matched",
            "type x lag-quintile adjusted",
            "type x lag-quintile adjusted, horizon-matched",
        ]
        assert report.watch.label == "type-adjusted, horizon-matched"
        assert report.watch.verdict is Verdict.NO_MEASURABLE_DIFFERENCE
        assert report.comparison_horizon_matched.n == treated.n
        assert set(report.lag_quartiles) == {"treated", "comparison"}
        as_json = json.dumps(report.to_dict())
        assert "no measurable difference" in as_json

    def test_empty_arm_fails_shut(self):
        treated, _ = _horizon_confounded_arms()
        with pytest.raises(EmptyArmError, match="no scoreable records"):
            compute_era_gap_report(treated, build_arm("empty", []), draws=FAST_DRAWS, seed=1)

    def test_render_names_the_rule_and_the_watch_verdict(self):
        treated, comparison = _horizon_confounded_arms()
        text = render_report(compute_era_gap_report(treated, comparison, draws=FAST_DRAWS, seed=1))
        assert "| treated |" in text
        assert "| type-adjusted, horizon-matched |" in text
        assert "Standing watch" in text
        assert "no measurable difference" in text
        assert f"below {CONCERN_GAP_POINTS:+.0f}" in text
        assert RESOLUTION_DAY_CONVENTION in text
        assert "Brackets disagree" not in text

    def test_render_notes_when_the_brackets_disagree_on_zero(self):
        """One event day per arm collapses the clustered interval onto the point (+1, excluding zero) while
        the by-record interval on alternating +/-30 scores covers zero: the verdict depends on the convention."""
        treated = build_arm("t", [_record(i, spot=31.0 if i % 2 else -29.0, lag_days=5) for i in range(8)])
        comparison = build_arm("c", [_record(100 + i, spot=30.0 if i % 2 else -30.0, lag_days=4) for i in range(8)])
        report = compute_era_gap_report(treated, comparison, draws=FAST_DRAWS, seed=1)
        unadjusted = report.gaps[0]
        assert unadjusted.gap == pytest.approx(1.0)
        assert unadjusted.brackets_disagree_on_zero
        assert unadjusted.verdict is Verdict.FAVOURABLE
        text = render_report(report)
        assert "**Brackets disagree on zero** for: unadjusted" in text
        assert report.to_dict()["gaps"][0]["brackets_disagree_on_zero"] is True


class TestCli:
    def _write(self, tmp_path, records: list[dict]) -> str:
        path = tmp_path / "tagged.json"
        path.write_text(json.dumps(records))
        return str(path)

    def _records(self) -> list[dict]:
        bug_qid = int(next(iter(KNOWN_BUG_QIDS)))
        records = [_record(i, spot=15.0 + i, lag_days=2 + i, era="triple_era") for i in range(6)]
        records += [_record(100 + i, spot=5.0 + i, lag_days=2 + 3 * i, era="post_flip") for i in range(8)]
        records.append(_record(bug_qid, spot=-80.0, lag_days=4, era="post_flip"))
        records.append(_record(900, spot=0.0, lag_days=1, era="pre_flip"))
        return records

    def test_smoke_prints_markdown_and_writes_json(self, tmp_path, capsys):
        path = self._write(tmp_path, self._records())
        out_json = tmp_path / "era_gap.json"
        main(
            [
                "--dataset",
                path,
                "--treated-era",
                "triple_era",
                "--comparison-era",
                "post_flip",
                "--strict",
                "--output-json",
                str(out_json),
            ]
        )
        text = capsys.readouterr().out
        assert "# Era gap: triple_era vs post_flip (STRICT" in text
        assert "| post_flip | 8 |" in text  # the known-bug record dropped, and its count shown
        assert "Standing watch" in text
        written = json.loads(out_json.read_text())
        assert written["comparison"]["n_excluded"] == 1
        assert written["watch"]["label"] == "type-adjusted, horizon-matched"

    def test_without_strict_the_cohort_record_stays_in(self, tmp_path, capsys):
        path = self._write(tmp_path, self._records())
        main(["--dataset", path, "--treated-era", "triple_era", "--comparison-era", "post_flip"])
        text = capsys.readouterr().out
        assert "(all records" in text
        assert "| post_flip | 9 |" in text

    def test_unknown_era_exits_with_a_message_naming_what_the_field_holds(self, tmp_path, capsys):
        """An arm the field never carries is a clean exit, not a traceback, and the message lists the
        values the field does hold so a mistyped era name is diagnosed on the spot."""
        path = self._write(tmp_path, self._records())
        with pytest.raises(SystemExit) as exc_info:
            main(["--dataset", path, "--treated-era", "fall_config", "--comparison-era", "post_flip"])
        message = str(exc_info.value)
        assert "era arm 'fall_config' has no scoreable records" in message
        assert f"`{ERA_FIELD}` values in the dataset: post_flip 9, triple_era 6, pre_flip 1" in message
        assert capsys.readouterr().out == ""

    def test_era_field_selects_arms_on_a_sub_era_field(self, tmp_path, capsys):
        """The tagging pass writes coarse eras to config_era and sub-eras to their own fields, so the
        fall's primary read (a sub-era treated arm) needs the field named."""
        records = [_record(i, spot=20.0 + i, lag_days=2 + i % 5, era="triple_era") for i in range(11)]
        for i, record in enumerate(records):
            record["triple_subera"] = "triple_ranked_market" if i < 6 else "triple_pre_market"
        records.append(_record(50, spot=1.0, lag_days=3, era="post_flip"))  # no sub-era tag at all
        path = self._write(tmp_path, records)
        out_json = tmp_path / "sub_era.json"
        main(
            [
                "--dataset",
                path,
                "--era-field",
                "triple_subera",
                "--treated-era",
                "triple_ranked_market",
                "--comparison-era",
                "triple_pre_market",
                "--output-json",
                str(out_json),
            ]
        )
        text = capsys.readouterr().out
        assert "| triple_ranked_market | 6 |" in text
        assert "| triple_pre_market | 5 |" in text
        assert "selected on `triple_subera`" in text
        assert json.loads(out_json.read_text())["era_field"] == "triple_subera"
        with pytest.raises(SystemExit, match="no scoreable records"):
            main(["--dataset", path, "--treated-era", "triple_ranked_market", "--comparison-era", "triple_pre_market"])

    def test_multiple_comparison_labels_pool_like_a_manually_relabelled_run(self, tmp_path, capsys):
        records = [_record(i, spot=10.0 + i, lag_days=2 + i, era="treated_a") for i in range(3)]
        records += [_record(i, spot=10.0 + i, lag_days=2 + i, era="treated_b") for i in range(3, 6)]
        records += [_record(100 + i, spot=4.0 + i, lag_days=3 + i, era="pre") for i in range(5)]
        records += [_record(200 + i, spot=8.0 + i, lag_days=4 + i, era="ranked") for i in range(4)]
        records += [_record(300 + i, spot=2.0 + i, lag_days=5 + i, era="time") for i in range(3)]
        pooled_path = self._write(tmp_path, records)
        pooled_json = tmp_path / "pooled.json"
        main(
            [
                "--dataset",
                pooled_path,
                "--treated-era",
                "treated_a",
                "treated_b",
                "--comparison-era",
                "pre",
                "ranked",
                "time",
                "--output-json",
                str(pooled_json),
            ]
        )
        pooled_text = capsys.readouterr().out

        comparison_labels = {"pre", "ranked", "time"}
        treated_labels = {"treated_a", "treated_b"}
        relabelled = [
            dict(record, config_era="comparison")
            if record[ERA_FIELD] in comparison_labels
            else dict(record, config_era="treated")
            if record[ERA_FIELD] in treated_labels
            else record
            for record in records
        ]
        relabelled_path = self._write(tmp_path, relabelled)
        relabelled_json = tmp_path / "relabelled.json"
        main(
            [
                "--dataset",
                relabelled_path,
                "--treated-era",
                "treated",
                "--comparison-era",
                "comparison",
                "--output-json",
                str(relabelled_json),
            ]
        )
        relabelled_text = capsys.readouterr().out
        pooled_report_text = pooled_json.read_text().replace("pre+ranked+time", "comparison")
        pooled_report_text = pooled_report_text.replace("treated_a+treated_b", "treated")
        pooled_report = json.loads(pooled_report_text)
        relabelled_report = json.loads(relabelled_json.read_text())
        assert pooled_report == relabelled_report
        normalized_text = pooled_text.replace("pre+ranked+time", "comparison")
        normalized_text = normalized_text.replace("treated_a+treated_b", "treated")
        assert normalized_text == relabelled_text
        assert "# Era gap: treated_a+treated_b vs pre+ranked+time" in pooled_text

    def test_single_label_cli_report_text_is_unchanged(self, tmp_path, capsys):
        path = self._write(tmp_path, self._records())
        main(["--dataset", path, "--treated-era", "triple_era", "--comparison-era", "post_flip"])
        text = capsys.readouterr().out
        assert "# Era gap: triple_era vs post_flip" in text
        assert "| post_flip | 9 |" in text

    def test_empty_multi_label_arm_message_names_joined_labels(self, tmp_path, capsys):
        path = self._write(tmp_path, self._records())
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--dataset",
                    path,
                    "--treated-era",
                    "triple_era",
                    "--comparison-era",
                    "missing_one",
                    "missing_two",
                ]
            )
        assert "era arm 'missing_one+missing_two' has no scoreable records" in str(exc_info.value)
        assert capsys.readouterr().out == ""

    def test_fall_read_before_its_first_resolution_exits_cleanly(self, tmp_path, capsys):
        """The preregistered fall invocation, run while every `fall_config` question is still open: the
        treated arm selects nothing, and the module must say so rather than bootstrap an empty array."""
        records = [_record(i, spot=20.0 + i, lag_days=2 + i % 5, era="triple_era") for i in range(8)]
        for record in records:
            record["triple_subera_fine"] = "ranked_markets"
        path = self._write(tmp_path, records)
        clusters = tmp_path / "cluster_structure.json"
        clusters.write_text(json.dumps(_cluster_structure(strong={"c": [0, 1]}, weak={})))
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--dataset",
                    path,
                    "--era-field",
                    "triple_subera_fine",
                    "--treated-era",
                    "fall_config",
                    "--comparison-era",
                    "ranked_markets",
                    "--strict",
                    "--clusters",
                    str(clusters),
                ]
            )
        message = str(exc_info.value)
        assert "era arm 'fall_config' has no scoreable records; nothing to compare yet" in message
        assert "`triple_subera_fine` values in the dataset: ranked_markets 8." in message
        assert capsys.readouterr().out == ""

    def test_clusters_argument_names_the_convention_and_the_labelled_counts(self, tmp_path, capsys):
        path = self._write(tmp_path, self._records())
        clusters = tmp_path / "cluster_structure.json"
        clusters.write_text(json.dumps(_cluster_structure(strong={"abc": [0, 1, 2]}, weak={"w": [3, 4]})))
        main(
            [
                "--dataset",
                path,
                "--treated-era",
                "triple_era",
                "--comparison-era",
                "post_flip",
                "--clusters",
                str(clusters),
            ]
        )
        text = capsys.readouterr().out
        assert f"curated strong clusters from {clusters}" in text
        assert "Curated clusters label 3/6 treated and 0/9 comparison records" in text
        assert RESOLUTION_DAY_CONVENTION not in text
