from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts.probes.bench_common.records import STATUS_SCORED
from scripts.probes.cross_pollination_bench.aggregate import FIRST_MEDIAN, AggregateScore
from scripts.probes.cross_pollination_bench.manifest import Manifest, build_manifest
from scripts.probes.cross_pollination_bench.movement import (
    contrast_by_first_spread,
    correction_counts,
    first_draw_noise,
    forecast_distance,
    likelihood_ratio,
    median_movement,
    member_changes,
    production_proxy,
    spread,
)
from scripts.probes.cross_pollination_bench.report import _contrast_tables, build_results, render_markdown
from scripts.probes.cross_pollination_bench.rows import STAGE_FIRST, UPDATE_VARIANTS, BenchCallRow
from tests.cross_pollination_bench_fakes import write_fake_round


@pytest.fixture
def manifest(tmp_path: Path) -> Manifest:
    fake = write_fake_round(tmp_path)
    return build_manifest(fake.perf_path, fake.archive_dir, event_map=None, fetch=fake.fetch)


def _row(qid: int, qtype: str, model: str, draw: int, stage: str, forecast, score: float, **kwargs) -> BenchCallRow:
    return BenchCallRow(
        question_id=qid,
        qtype=qtype,
        model=model,
        draw=draw,
        stage=stage,
        status=STATUS_SCORED,
        prompt_sha256="prompt",
        forecast=forecast,
        score=score,
        **kwargs,
    )


def test_distance_per_type() -> None:
    assert forecast_distance("binary", 0.2, 0.5) == pytest.approx(30.0)
    assert forecast_distance("multiple_choice", [0.7, 0.3], [0.4, 0.6]) == pytest.approx(30.0)
    assert forecast_distance("numeric", [0.1, 0.5, 0.9], [0.2, 0.5, 0.8]) == pytest.approx(6.6666667)
    assert forecast_distance("discrete", [0.1, 0.5, 0.9], [0.2, 0.5, 0.8]) == pytest.approx(6.6666667)


def test_typical_change_is_the_median_absolute_change_per_question_draw() -> None:
    scores = []
    for qid, variant in enumerate((0.50, 0.52, 0.60), start=1):
        scores.extend(
            [
                AggregateScore(qid, "binary", 0, FIRST_MEDIAN, 0.50, 0.0, 0.0, 3, True, 0),
                AggregateScore(qid, "binary", 0, "peers_full_median", variant, 0.0, 0.0, 3, True, 0),
            ]
        )
    movement = median_movement(scores)
    assert movement["peers_full"]["typical_change"] == pytest.approx(2.0)
    assert movement["peers_full"]["share_changed"] == pytest.approx(2 / 3)


def test_first_draw_noise_is_a_reported_scale_and_never_a_threshold() -> None:
    scores = []
    for qid, first_one, variant in ((1, 0.51, 0.52), (2, 0.61, 0.62)):
        scores.extend(
            [
                AggregateScore(qid, "binary", 1, FIRST_MEDIAN, 0.50, 0.0, 0.0, 3, True, 0),
                AggregateScore(qid, "binary", 2, FIRST_MEDIAN, first_one, 0.0, 0.0, 3, True, 0),
                AggregateScore(qid, "binary", 1, "peers_full_median", variant, 0.0, 0.0, 3, True, 0),
            ]
        )
    before = median_movement(scores)
    scores[1] = AggregateScore(1, "binary", 2, FIRST_MEDIAN, 0.80, 0.0, 0.0, 3, True, 0)
    scores[4] = AggregateScore(2, "binary", 2, FIRST_MEDIAN, 0.90, 0.0, 0.0, 3, True, 0)
    after = median_movement(scores)
    assert first_draw_noise(scores)["n_questions"] == 2
    assert first_draw_noise(scores)["median"] > 0
    assert before["peers_full"]["typical_change"] == after["peers_full"]["typical_change"]
    assert before["peers_full"]["share_changed"] == after["peers_full"]["share_changed"]


def test_spread_shrinks_when_members_converge() -> None:
    assert spread("binary", [0.2, 0.5, 0.9]) == pytest.approx(70.0)
    assert spread("binary", [0.45, 0.5, 0.55]) < spread("binary", [0.2, 0.5, 0.9])


@pytest.mark.parametrize(
    ("qtype", "first_forecast", "updated_forecast", "first_score", "updated_score"),
    [
        ("binary", 0.2, 0.8, -2.0, -0.5),
        ("multiple_choice", [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], -2.0, -0.5),
        ("numeric", [0.1, 0.3, 0.9], [0.2, 0.4, 0.8], -2.0, -0.5),
    ],
)
def test_a_member_moved_toward_the_outcome_when_its_score_rose(
    manifest: Manifest, qtype: str, first_forecast, updated_forecast, first_score: float, updated_score: float
) -> None:
    entry = next(item for item in manifest.questions if item.qtype == qtype)
    rows = [
        _row(entry.question_id, qtype, "haiku", 0, STAGE_FIRST, first_forecast, first_score),
        _row(entry.question_id, qtype, "haiku", 0, "peers_full", updated_forecast, updated_score),
    ]
    changes = member_changes(rows, manifest)["peers_full"]["haiku"]
    assert changes["share_toward_outcome"] == 1.0
    assert changes["share_away_from_outcome"] == 0.0


def test_correction_counts_by_stated_kind() -> None:
    rows = [
        _row(1, "binary", "haiku", 0, variant, 0.5, 0.0, correction_kind=kind)
        for variant, kind in zip(UPDATE_VARIANTS, ("fact", "none", "unparsed"), strict=True)
    ]
    assert correction_counts(rows)["peers_full"]["fact"] == 1
    assert correction_counts(rows)["peers_forecasts"]["none"] == 1
    assert correction_counts(rows)["second_look"]["unparsed"] == 1


def test_production_proxy_scores_production_next_to_the_first_median(manifest: Manifest) -> None:
    scores = [AggregateScore(101, "binary", 0, FIRST_MEDIAN, 0.4, 0.0, 0.0, 3, True, 0)]
    proxy = production_proxy(scores, manifest)
    assert proxy["binary"]["production_mean_peer_points"] is not None
    assert proxy["binary"]["first_median_mean_distance"] == pytest.approx(20.0)


def test_likelihood_ratio_and_numeric_rows_are_excluded(manifest: Manifest) -> None:
    assert likelihood_ratio([10.0, -5.0]) == pytest.approx(math.exp(0.05))
    scores = []
    for qid, qtype, treatment, control in (
        (101, "binary", 10.0, 0.0),
        (102, "multiple_choice", -5.0, 0.0),
        (103, "numeric", 100.0, 0.0),
    ):
        scores.extend(
            [
                AggregateScore(qid, qtype, 0, "peers_full_median", 0.4, 0.0, treatment, 3, True, 0),
                AggregateScore(qid, qtype, 0, "second_look_median", 0.4, 0.0, control, 3, True, 0),
            ]
        )
    contrasts, paired = _contrast_tables(scores, manifest, [], None)
    assert contrasts["peers_full_vs_second_look"]["likelihood_ratio_binary_and_mc"] == pytest.approx(math.exp(0.05))
    assert contrasts["peers_full_vs_second_look"]["by_type"]["numeric"]["mean_difference"] == pytest.approx(100.0)
    assert (
        contrast_by_first_spread(paired["peers_full_vs_second_look"], [], manifest)["below_median_first_spread"][
            "n_question_draws"
        ]
        == 0
    )


def test_likelihood_ratio_averages_draw_differences_within_each_question(manifest: Manifest) -> None:
    scores = []
    for qid, qtype, differences in (
        (101, "binary", (10.0, 10.0)),
        (102, "multiple_choice", (-5.0, -5.0)),
        (103, "numeric", (100.0, 100.0)),
    ):
        for draw, difference in enumerate(differences):
            scores.extend(
                [
                    AggregateScore(qid, qtype, draw, "peers_full_median", 0.4, 0.0, difference, 3, True, 0),
                    AggregateScore(qid, qtype, draw, "second_look_median", 0.4, 0.0, 0.0, 3, True, 0),
                ]
            )

    contrasts, _ = _contrast_tables(scores, manifest, [], None)

    assert contrasts["peers_full_vs_second_look"]["likelihood_ratio_binary_and_mc"] == pytest.approx(math.exp(0.05))


def test_summary_has_the_movement_sections_and_the_three_member_note(manifest: Manifest) -> None:
    result = build_results([], manifest, {})
    assert {"movement", "spread", "member_changes", "corrections", "production_proxy"} <= result.keys()
    summary = render_markdown(result)
    assert "specific to three members" in summary
