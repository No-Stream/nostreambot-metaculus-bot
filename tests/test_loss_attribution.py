"""Tests for loss-attribution calculations."""

import json
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from metaculus_bot.performance_analysis.analysis import PitReading
from metaculus_bot.performance_analysis.cohorts import KNOWN_BUG_QIDS
from metaculus_bot.performance_analysis.ghosts import ghost_cdf_on_published_grid
from metaculus_bot.performance_analysis.loss_attribution import (
    build_era_report,
    calibration_reports,
    corp_decomposition,
    disagreement_report,
    era_buckets,
    floor_summary,
    ghost_comparison,
    main,
    mc_discrimination,
    pit_accounting,
    pit_bin_label,
    points_lost,
    reliability_table,
    slice_ranking,
    winner_rank,
)
from metaculus_bot.performance_analysis.loss_attribution_report import _number
from metaculus_bot.scoring_common import numeric_log_score, spot_peer_delta


class TestCorp:
    @pytest.mark.parametrize("score", ["brier", "log"])
    def test_decomposition_and_ties(self, score: Literal["brier", "log"]) -> None:
        probabilities = [0.25, 0.25, 0.75, 0.75]
        outcomes = [False, True, True, True]
        result = corp_decomposition(probabilities, outcomes, score)
        reversed_result = corp_decomposition(probabilities[::-1], outcomes[::-1], score)
        assert result == reversed_result
        assert result.mean_score == pytest.approx(result.mcb - result.dsc + result.unc)

    def test_constant_forecast_has_no_discrimination(self) -> None:
        assert corp_decomposition([0.4] * 4, [False, True, False, True], "brier").dsc == pytest.approx(0)


class TestWinnerRank:
    def test_average_rank_for_ties(self) -> None:
        assert winner_rank([0.4, 0.4, 0.2], 1) == pytest.approx(0.25)


class TestAccounting:
    def test_confidence_is_independent_of_outcome_and_bins_conserve_points(self) -> None:
        records = [
            {
                "type": "binary",
                "our_prob_yes": 0.8,
                "resolution_parsed": outcome,
                "metaculus_scores": {"spot_peer_score": peer},
                "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
            }
            for outcome, peer in [(True, 10.0), (False, -20.0)]
        ]
        before = points_lost(records, "binary", confidence=True)
        for record in records:
            del record["resolution_parsed"]
        after = points_lost(records, "binary", confidence=True)
        assert before == after
        assert sum(row.sum_spot_peer for row in before) == pytest.approx(-10)


class TestEraBuckets:
    def test_none_and_fall_are_visible(self) -> None:
        records = [
            {"triple_subera": None, "config_era": "pre_flip"},
            {"triple_subera": "triple_fall_config", "config_era": "triple_era"},
        ]
        assert list(era_buckets(records, "triple_subera")) == ["all", "(none)", "triple_fall_config"]
        assert len(era_buckets(records, "config_era")["triple_fall_config"]) == 1


class TestReliability:
    def test_bin_counts_and_observed_rate(self) -> None:
        rows = reliability_table([(0.2, True), (0.3, False), (0.8, True)], (0.0, 0.5, 1.0))
        assert [row.n for row in rows] == [2, 1]
        assert rows[0].observed == pytest.approx(0.5)
        assert rows[0].mean_forecast == pytest.approx(0.25)
        assert rows[0].observed_count == 1
        assert rows[0].expected_count == pytest.approx(0.5)
        assert rows[0].observed_minus_expected == pytest.approx(0.5)
        assert rows[0].cluster_interval is None

    def test_binary_and_multiple_choice_counts_use_existing_reliability_denominators(self) -> None:
        records = [
            {
                "type": "binary",
                "our_prob_yes": probability,
                "resolution_parsed": outcome,
                "metaculus_scores": {},
                "metadata": {"actual_resolve_time": f"2026-01-{index + 1:02d}T00:00:00Z"},
            }
            for index, (probability, outcome) in enumerate([(0.8, True), (0.2, False)])
        ] + [
            {
                "type": "multiple_choice",
                "options": ["A", "B"],
                "our_forecast_values": probabilities,
                "resolution_parsed": outcome,
                "metaculus_scores": {},
                "metadata": {"actual_resolve_time": f"2026-01-{index + 3:02d}T00:00:00Z"},
            }
            for index, (probabilities, outcome) in enumerate([([0.8, 0.2], "A"), ([0.2, 0.8], "B")])
        ]
        reports = {report.name: report for report in calibration_reports(records)}
        binary = reports["binary p_yes"].reliability
        top = reports["MC top option"].reliability
        options = reports["MC secondary: options within a ballot are dependent"].reliability
        assert sum(row.observed_count for row in binary) == 1
        assert sum(row.expected_count for row in binary) == pytest.approx(1.0)
        assert sum(row.observed_count for row in top) == 2
        assert sum(row.expected_count for row in top) == pytest.approx(1.6)
        assert sum(row.observed_count for row in options) == 2
        assert sum(row.expected_count for row in options) == pytest.approx(2.0)
        assert sum(row.n for row in options) == 4

    def test_option_interval_requires_distinct_questions(self) -> None:
        records = [
            {
                "type": "multiple_choice",
                "options": ["A", "B", "C", "D"],
                "our_forecast_values": [0.25] * 4,
                "resolution_parsed": "A",
                "metadata": {"actual_resolve_time": f"2026-01-{index + 1:02d}T00:00:00Z"},
            }
            for index in range(6)
        ]
        report = next(
            report
            for report in calibration_reports(records)
            if report.name == "MC secondary: options within a ballot are dependent"
        )
        row = report.reliability[0]
        assert row.n == 24
        assert row.n_questions == 6
        assert row.expected_count == 6
        assert row.cluster_interval is None
        assert row.jeffreys_interval is None

    def test_calibrated_draws_match_expected_counts(self) -> None:
        rng = np.random.default_rng(731)
        pairs = []
        for index in range(4000):
            probability = (0.03, 0.1, 0.3, 0.55, 0.8, 0.93)[index % 6]
            pairs.append((probability, bool(rng.random() < probability)))
        rows = reliability_table(pairs, (0.0, 0.05, 0.15, 0.35, 0.65, 0.85, 1.0))
        assert all(abs(row.observed_count - row.expected_count) < 45 for row in rows)

    @pytest.mark.parametrize(
        ("probability", "count", "days"),
        [(0.01, 40, 10), (0.99, 40, 10), (0.5, 9, 5), (0.5, 20, 4)],
    )
    def test_cluster_interval_requires_sample_days_and_expected_counts(
        self, probability: float, count: int, days: int
    ) -> None:
        row = reliability_table(
            [(probability, index % 2 == 0) for index in range(count)],
            (0.0, 1.0),
            clusters=[f"day-{index % days}" for index in range(count)],
            seed=17,
            draws=100,
        )[0]
        assert row.cluster_interval is None
        if count >= 10 and days >= 5:
            assert row.jeffreys_interval is not None

    def test_cluster_interval_resamples_whole_resolution_days(self) -> None:
        row = reliability_table(
            [(0.5, index < 12) for index in range(20)],
            (0.0, 1.0),
            clusters=[f"day-{index // 2}" for index in range(20)],
            seed=17,
            draws=1000,
        )[0]
        assert row.observed_count == 12
        assert row.expected_count == 10
        assert row.observed_minus_expected == 2
        assert row.cluster_interval == pytest.approx((-4, 8))
        assert row.jeffreys_interval is not None

    def test_perfectly_calibrated_forecast_has_zero_mcb(self) -> None:
        result = corp_decomposition(
            [0.25] * 4 + [0.75] * 4, [True, False, False, False, True, True, True, False], "brier"
        )
        assert result.mcb == pytest.approx(0)


class TestDiscrimination:
    def test_member_ballots_and_symmetric_counts(self) -> None:
        records = []
        for index, (published, member) in enumerate(
            [([0.6, 0.4], {"A": 0.3, "B": 0.7}), ([0.3, 0.7], {"A": 0.8, "B": 0.2}), ([0.6, 0.4], {"A": 0.6})]
        ):
            records.append(
                {
                    "type": "multiple_choice",
                    "question_id": index,
                    "options": ["A", "B"],
                    "resolution_parsed": "A",
                    "our_forecast_values": published,
                    "per_model_forecasts": {"model": member},
                    "metaculus_scores": {"spot_peer_score": 10.0},
                    "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
                }
            )
        rows = mc_discrimination(records)
        member_row = next(row for row in rows if row.name == "model")
        assert member_row.n == 2
        assert member_row.incomplete == 1
        assert member_row.member_only_top == 1
        assert member_row.published_only_top == 1


class TestSliceRanking:
    def test_research_classes_are_not_pooled(self) -> None:
        records = [
            {
                "type": "binary",
                "research_source_class": source,
                "anchor_present": True,
                "metaculus_scores": {"spot_peer_score": score},
                "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
            }
            for source, score in [("artifact", 10.0), ("comment_backfill", -10.0)]
        ]
        rows = slice_ranking(records)
        anchors = [row for row in rows if row.dimension == "anchor_present"]
        assert len(anchors) == 2
        assert {row.value for row in anchors} == {"artifact / True", "comment_backfill / True"}


class TestPit:
    def test_out_of_bounds_labels_preserve_side_and_openness(self) -> None:
        assert pit_bin_label(PitReading(0.0, 0.02, "low"), {"open_lower_bound": True}) == "low, open"
        assert pit_bin_label(PitReading(0.99, 1.0, "high"), {"open_upper_bound": False}) == "high, closed"
        assert pit_bin_label(PitReading.from_point(0.5), {}) == "[0.1, 0.9)"


class TestCounterfactual:
    def test_numeric_score_difference_equals_spot_delta(self) -> None:
        old_cdf = [0.1, 0.4, 0.8]
        new_cdf = [0.1, 0.3, 0.8]
        kwargs = {"open_lower_bound": True, "open_upper_bound": True}
        change = numeric_log_score(new_cdf, 0.25, 0.0, 1.0, **kwargs) - numeric_log_score(
            old_cdf, 0.25, 0.0, 1.0, **kwargs
        )
        assert change == pytest.approx(spot_peer_delta(old_prob=0.3, new_prob=0.2, question_type="numeric"))

    def test_binary_and_multiple_choice_ghosts(self) -> None:
        binary = {"type": "binary", "our_prob_yes": 0.4, "resolution_parsed": True}
        ghost = {"source": "json", "qtype": "binary", "payload": {"prob": 0.8}}
        comparison = ghost_comparison(binary, ghost)
        assert comparison is not None
        distance, delta = comparison.distance, comparison.delta
        assert distance == pytest.approx(0.4)
        assert delta == pytest.approx(spot_peer_delta(old_prob=0.4, new_prob=0.8, question_type="binary"))
        record = {
            "type": "multiple_choice",
            "options": ["A", "B"],
            "our_forecast_values": [0.4, 0.6],
            "resolution_parsed": "A",
        }
        mc_ghost = {"source": "json", "qtype": "multiple_choice", "payload": {"option_probs": {"A": 0.8, "B": 0.2}}}
        comparison = ghost_comparison(record, mc_ghost)
        assert comparison is not None
        distance, delta = comparison.distance, comparison.delta
        assert distance == pytest.approx(0.4)
        assert delta == pytest.approx(spot_peer_delta(old_prob=0.4, new_prob=0.8, question_type="multiple_choice"))


class TestCli:
    def test_writes_report_without_telemetry(self, tmp_path: Path) -> None:
        main(["--cached", "tests/fixtures/loss_attribution_records.json", "--out", str(tmp_path), "--draws", "20"])
        report = (tmp_path / "loss_attribution.md").read_text()
        assert "not run: no --telemetry-archive" in report
        assert "(n too small)" in report
        assert "observed versus expected winners" not in report
        assert (
            "| observed count | expected count | observed minus expected | 95% interval on count difference |" in report
        )
        report_data = json.loads((tmp_path / "loss_attribution.json").read_text())
        assert all("probability_mass" not in era for era in report_data["eras"])
        assert (tmp_path / "loss_attribution.json").exists()

    def test_cli_joins_ghost_by_post_id(self, tmp_path: Path) -> None:
        main(
            [
                "--cached",
                "tests/fixtures/loss_attribution_records.json",
                "--telemetry-archive",
                "tests/fixtures/loss_attribution_ghosts",
                "--out",
                str(tmp_path),
                "--draws",
                "20",
            ]
        )
        payload = json.loads((tmp_path / "loss_attribution.json").read_text())
        report = (tmp_path / "loss_attribution.md").read_text()
        assert payload["header"]["mc_joined_ghost"] == 1
        assert payload["header"]["unjoined_archived_ghosts"] == {"binary": 1}
        assert any(row["name"] == "ghost" and row["n"] == 1 for row in payload["eras"][0]["discrimination"])
        assert payload["header"]["mc_complete_member_vectors"] == 1
        all_era = next(era for era in payload["eras"] if era["era"] == "all")
        question = next(
            question for row in all_era["ghost"]["rows"] for question in row["questions"] if question["post_id"] == 201
        )
        assert question["url"] == "https://www.metaculus.com/questions/201/"
        assert "Questions in distance bins at or above 0.05, for trace review." in report
        assert report.count("Questions in distance bins at or above 0.05, for trace review.") == 1
        assert "[201](https://www.metaculus.com/questions/201/)" in report

    def test_numeric_ghost_uses_continuous_score_without_second_halving(self) -> None:
        record = {
            "type": "numeric",
            "resolution_parsed": 50.0,
            "our_forecast_values": [index / 10 for index in range(11)],
            "scaling": {"range_min": 0, "range_max": 100, "zero_point": None},
            "open_lower_bound": False,
            "open_upper_bound": False,
        }
        ghost = {
            "qtype": "numeric",
            "source": "json",
            "payload": {"declared_percentiles": {"0.1": 20, "0.5": 60, "0.9": 90}},
        }
        comparison = ghost_comparison(record, ghost)
        assert comparison is not None
        cdf = ghost_cdf_on_published_grid({0.1: 20.0, 0.5: 60.0, 0.9: 90.0}, record)
        assert cdf is not None
        expected = numeric_log_score(
            cdf, 50, 0, 100, open_lower_bound=False, open_upper_bound=False
        ) - numeric_log_score(record["our_forecast_values"], 50, 0, 100, open_lower_bound=False, open_upper_bound=False)
        assert comparison.delta == pytest.approx(expected)


class TestRealDataset:
    def test_cli_real_data(self, tmp_path: Path) -> None:
        path = Path("scratch/residual_2026-11-25_floors/perf_all_tagged.json")
        archive = Path("backtests/telemetry_archive")
        if not path.exists() or not archive.is_dir():
            pytest.skip("Real dataset or telemetry archive is not available")
        main(
            [
                "--cached",
                str(path),
                "--exclude-qids",
                "known_bug,degraded_run,partial_degraded",
                "--telemetry-archive",
                str(archive),
                "--out",
                str(tmp_path),
            ]
        )
        payload = json.loads((tmp_path / "loss_attribution.json").read_text())
        assert payload["header"]["before"]["multiple_choice"] == 130
        recalibration = payload["recalibration"]
        assert recalibration["era_order"][:4] == [
            "pre_flip (early half)",
            "pre_flip (late half)",
            "post_flip",
            "triple_pre_market",
        ]
        assert recalibration["exclusions"]["missing era"] > 0
        assert all(
            any(row["method"] == method and row["status"] == "evaluated" for row in recalibration["rows"])
            for method in ("platt", "isotonic", "power")
        )
        assert (tmp_path / "loss_attribution.md").read_text().count("## Out-of-sample recalibration") == 1
        assert payload["header"]["after"]["multiple_choice"] == 127
        assert payload["header"]["mc_fall"] == 13
        era = next(era for era in payload["eras"] if era["era"] == "all")
        assert sum(
            row["sum_spot_peer"] for row in era["accounting"]["multiple_choice: where the points went"]
        ) == pytest.approx(1934.81, abs=0.01)
        binary = next(row for row in era["calibration"] if row["name"] == "binary p_yes")["brier"]
        for name, expected in {"mean_score": 0.1258, "mcb": 0.0063, "dsc": 0.0986, "unc": 0.2181}.items():
            assert binary[name] == pytest.approx(expected, abs=5e-4)


class TestGhostExclusions:
    def test_binary_ghost_at_zero_is_priced_at_the_historical_publish_floor(self) -> None:
        record = {
            "post_id": 3,
            "type": "binary",
            "our_prob_yes": 0.5,
            "resolution_parsed": True,
            "metaculus_scores": {"spot_peer_score": 1.0},
            "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
        }
        ghost = {"qtype": "binary", "source": "json", "payload": {"prob": 0.0}}
        report = disagreement_report([record], {3: ghost}, seed=1, draws=20)
        assert report.zero_probability == 0
        assert report.counts_by_type[0].statuses == {"scored": 1}
        assert report.unscoreable == 0
        assert report.excluded_spot_peer == 0

    def test_explicit_question_id_exclusion(self, tmp_path: Path) -> None:
        main(
            [
                "--cached",
                "tests/fixtures/loss_attribution_records.json",
                "--exclude-qids",
                "100",
                "--out",
                str(tmp_path),
                "--draws",
                "20",
            ]
        )
        payload = json.loads((tmp_path / "loss_attribution.json").read_text())
        assert payload["header"]["after"] == {"multiple_choice": 1}
        assert payload["header"]["excluded"] == 1

    def test_symmetric_dissent_for_four_members(self) -> None:
        records = []
        for index, (published, members) in enumerate(
            [
                (
                    [0.3, 0.7],
                    {
                        "one": {"A": 0.8, "B": 0.2},
                        "two": {"A": 0.6, "B": 0.4},
                        "three": {"A": 0.3, "B": 0.7},
                        "four": {"A": 0.2, "B": 0.8},
                    },
                ),
                (
                    [0.8, 0.2],
                    {
                        "one": {"A": 0.2, "B": 0.8},
                        "two": {"A": 0.3, "B": 0.7},
                        "three": {"A": 0.7, "B": 0.3},
                        "four": {"A": 0.6, "B": 0.4},
                    },
                ),
            ]
        ):
            records.append(
                {
                    "type": "multiple_choice",
                    "question_id": index,
                    "options": ["A", "B"],
                    "resolution_parsed": "A",
                    "our_forecast_values": published,
                    "per_model_forecasts": members,
                    "metaculus_scores": {"spot_peer_score": 0},
                    "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
                }
            )
        rows = {row.name: row for row in mc_discrimination(records)}
        assert (rows["one"].member_only_top, rows["one"].published_only_top) == (1, 1)
        assert (rows["two"].member_only_top, rows["two"].published_only_top) == (1, 1)
        assert (rows["three"].member_only_top, rows["three"].published_only_top) == (0, 0)
        assert (rows["four"].member_only_top, rows["four"].published_only_top) == (0, 0)


class TestFloorCalibrationParity:
    def test_floor_counts_questions_options_and_hits(self) -> None:
        records = [
            {
                "type": "multiple_choice",
                "options": ["A", "B", "C"],
                "our_forecast_values": [0.01, 0.01, 0.98],
                "resolution_parsed": "A",
            }
        ]
        row = floor_summary(records)
        assert (row.questions, row.options, row.hits) == (1, 2, 1)
        assert row.expected_hits == pytest.approx(0.02)

    def test_numeric_pit_accounting_keeps_open_and_closed_tails(self) -> None:
        common = {
            "type": "numeric",
            "our_forecast_values": [index / 10 for index in range(11)],
            "scaling": {"range_min": 0, "range_max": 100, "zero_point": None},
            "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
        }
        records = [
            {
                **common,
                "resolution_parsed": "below_lower_bound",
                "open_lower_bound": True,
                "open_upper_bound": False,
                "metaculus_scores": {"spot_peer_score": -3},
            },
            {
                **common,
                "resolution_parsed": "above_upper_bound",
                "open_lower_bound": False,
                "open_upper_bound": False,
                "metaculus_scores": {"spot_peer_score": 4},
            },
            {
                **common,
                "resolution_parsed": 50.0,
                "open_lower_bound": False,
                "open_upper_bound": False,
                "metaculus_scores": {"spot_peer_score": 2},
            },
        ]
        rows = pit_accounting(records)
        assert {row.label for row in rows} == {"low, open", "high, closed", "[0.1, 0.9)"}
        assert sum(row.sum_spot_peer for row in rows) == pytest.approx(3)

    def test_missing_telemetry_directory_fails(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Telemetry archive directory does not exist"):
            main(
                [
                    "--cached",
                    "tests/fixtures/loss_attribution_records.json",
                    "--telemetry-archive",
                    str(tmp_path / "missing"),
                    "--out",
                    str(tmp_path),
                ]
            )

    @pytest.mark.parametrize(("winner", "expected_accuracy"), [("A", 1), ("B", 0)])
    def test_tied_top_scores_the_first_option_in_option_order(self, winner: str, expected_accuracy: int) -> None:
        record = {
            "type": "multiple_choice",
            "question_id": 1,
            "options": ["A", "B"],
            "resolution_parsed": winner,
            "our_forecast_values": [0.5, 0.5],
            "metaculus_scores": {"spot_peer_score": 0},
            "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
        }
        published = mc_discrimination([record])[0]
        assert published.top_accuracy == expected_accuracy
        assert published.tied_top == 1
        assert published.mean_rank == pytest.approx(0.5)

    def test_balanced_tied_ballots_show_no_miscalibration(self) -> None:
        records = [
            {
                "type": "multiple_choice",
                "question_id": index,
                "options": ["A", "B"],
                "resolution_parsed": "A" if index % 2 else "B",
                "our_forecast_values": [0.5, 0.5],
                "metaculus_scores": {"spot_peer_score": 0},
                "metadata": {"actual_resolve_time": f"2026-01-{index + 1:02d}T00:00:00Z"},
            }
            for index in range(10)
        ]
        top = next(item for item in calibration_reports(records) if item.name == "MC top option")
        assert top.brier.mcb == pytest.approx(0.0)
        assert top.log.mcb == pytest.approx(0.0)
        assert top.reliability[0].observed == pytest.approx(0.5)


class TestMultipleChoiceAccounting:
    def test_both_bin_views_conserve_score(self) -> None:
        records = [
            {
                "type": "multiple_choice",
                "options": ["A", "B"],
                "our_forecast_values": [0.3, 0.7],
                "resolution_parsed": outcome,
                "metaculus_scores": {"spot_peer_score": peer},
                "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
            }
            for outcome, peer in [("A", -20.0), ("B", 10.0)]
        ]
        for confidence in (False, True):
            assert sum(
                row.sum_spot_peer for row in points_lost(records, "multiple_choice", confidence=confidence)
            ) == pytest.approx(-10)

    def test_cohort_shorthand_excludes_question_id(self, tmp_path: Path) -> None:
        known_id = next(iter(KNOWN_BUG_QIDS))
        records = json.loads(Path("tests/fixtures/loss_attribution_records.json").read_text())
        records[0]["question_id"] = int(known_id)
        fixture = tmp_path / "records.json"
        fixture.write_text(json.dumps(records))
        main(["--cached", str(fixture), "--exclude-qids", "known_bug", "--out", str(tmp_path), "--draws", "20"])
        report = json.loads((tmp_path / "loss_attribution.json").read_text())
        assert report["header"]["cohorts"]["known_bug"] == 1
        assert report["header"]["after"] == {"multiple_choice": 1}


class TestReviewRegressions:
    def test_outcome_diagnostics_include_unscored_tied_ballot(self) -> None:
        record = {
            "type": "multiple_choice",
            "question_id": 1,
            "post_id": 1,
            "options": ["A", "B"],
            "resolution_parsed": "B",
            "our_forecast_values": [0.5, 0.5],
            "metaculus_scores": {},
            "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
        }
        report = build_era_report("all", [record], None, seed=1, draws=20)
        top = next(item for item in report.calibration if item.name == "MC top option")
        assert top.brier.n == 1
        assert top.brier.mean_score == pytest.approx(0.25)
        assert top.tied_top == 1
        row = report.discrimination[0]
        assert row.n == 1
        assert row.top_accuracy == 0
        assert row.tied_top == 1
        assert row.scored_n == 0
        assert report.exclusions["multiple_choice: by our confidence"]["no spot peer"] == 1

    def test_accounting_reports_missing_vector_and_pit(self) -> None:
        records = [
            {
                "type": "multiple_choice",
                "question_id": 1,
                "post_id": 1,
                "options": ["A", "B"],
                "resolution_parsed": "A",
                "our_forecast_values": None,
                "metaculus_scores": {"spot_peer_score": -10},
                "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
            },
            {
                "type": "numeric",
                "question_id": 2,
                "post_id": 2,
                "resolution_parsed": None,
                "our_forecast_values": None,
                "metaculus_scores": {"spot_peer_score": 2},
                "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
            },
        ]
        report = build_era_report("all", records, None, seed=1, draws=20)
        assert report.exclusions["multiple_choice: where the points went"]["no usable forecast vector"] == 1
        assert report.excluded_spot_peer["multiple_choice: where the points went"] == -10
        assert report.excluded_spot_peer["research slices"] == -8
        assert report.exclusions["numeric and discrete: placement in the distribution"]["no PIT reading"] == 1

    def test_ghost_status_and_all_distances(self) -> None:
        record = {
            "post_id": 3,
            "type": "binary",
            "our_prob_yes": 0.5,
            "resolution_parsed": True,
            "metaculus_scores": {"spot_peer_score": 1},
            "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
        }
        ghost = {"qtype": "binary", "source": "json", "payload": {"prob": 0.7}}
        result = ghost_comparison(record, ghost)
        assert result.status == "scored"
        report = disagreement_report([record], {3: ghost}, seed=1, draws=20)
        assert report.rows[0].label == "all distances"
        assert report.counts_by_type[0].statuses["scored"] == 1

    def test_ghost_bins_include_sorted_question_trace_details(self) -> None:
        records = []
        ghosts = {}
        for post_id, published, ghost_probability in [(31, 0.5, 0.65), (32, 0.5, 0.75)]:
            records.append(
                {
                    "post_id": post_id,
                    "type": "binary",
                    "our_prob_yes": published,
                    "resolution_parsed": False,
                    "metaculus_scores": {"spot_peer_score": 2.0},
                    "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
                }
            )
            ghosts[post_id] = {"qtype": "binary", "source": "json", "payload": {"prob": ghost_probability}}
        report = disagreement_report(records, ghosts, seed=1, draws=20)
        bin_row = next(row for row in report.rows if row.label == "[0.15, 0.3)")
        assert [question.post_id for question in bin_row.questions] == [32, 31]
        assert [question.distance for question in bin_row.questions] == pytest.approx([0.25, 0.15])
        assert all(question.url.endswith(f"/{question.post_id}/") for question in bin_row.questions)

    def test_negative_zero_formatting(self) -> None:
        assert _number(-0.00001) == "0.000"


def test_corp_matches_brute_force_pooled_adjacent_violators() -> None:
    probabilities = [0.1, 0.2, 0.3, 0.8]
    outcomes = [True, False, False, True]
    losses = []
    for boundaries in product((False, True), repeat=len(outcomes) - 1):
        blocks: list[list[bool]] = [[outcomes[0]]]
        for boundary, outcome in zip(boundaries, outcomes[1:], strict=True):
            if boundary:
                blocks.append([])
            blocks[-1].append(outcome)
        means = [sum(block) / len(block) for block in blocks]
        if means != sorted(means):
            continue
        losses.append(
            sum((mean - outcome) ** 2 for mean, block in zip(means, blocks, strict=True) for outcome in block)
            / len(outcomes)
        )
    result = corp_decomposition(probabilities, outcomes, "brier")
    assert result.mcb == pytest.approx(result.mean_score - min(losses))


@pytest.mark.parametrize("score", ["brier", "log"])
def test_corp_calibrated_and_constant(score: Literal["brier", "log"]) -> None:
    calibrated = corp_decomposition(
        [0.25] * 4 + [0.75] * 4, [True, False, False, False, True, True, True, False], score
    )
    constant = corp_decomposition([0.4] * 4, [False, True, False, True], score)
    assert calibrated.mcb == pytest.approx(0)
    assert constant.dsc == pytest.approx(0)


def test_positive_only_bin_has_zero_share_of_negative_spot_peer() -> None:
    records = [
        {
            "type": "binary",
            "our_prob_yes": 0.8,
            "resolution_parsed": True,
            "metaculus_scores": {"spot_peer_score": 1.0},
            "metadata": {"actual_resolve_time": "2026-01-01T00:00:00Z"},
        }
    ]
    assert points_lost(records, "binary")[0].negative_share == 0.0
