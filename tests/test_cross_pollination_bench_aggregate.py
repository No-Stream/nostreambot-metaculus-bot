from __future__ import annotations

import math
from pathlib import Path

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.questions import MetaculusQuestion, NumericQuestion

from metaculus_bot.aggregation_strategies import aggregate_multiple_choice_median
from metaculus_bot.numeric.utils import aggregate_numeric
from scripts.probes.bench_common.records import STATUS_SCORED
from scripts.probes.cross_pollination_bench.aggregate import (
    FIRST_MEDIAN,
    MECHANICAL_BLEND,
    POOLED_FIRST_DRAWS,
    aggregate_scores,
    blend_forecast,
    distribution_from_cdf,
    median_forecast,
    member_paired_rows,
    missing_panels,
    paired_rows,
    panel,
)
from scripts.probes.cross_pollination_bench.manifest import Manifest, build_manifest, rebuild_question
from scripts.probes.cross_pollination_bench.rows import STAGE_FIRST, VARIANT_PEERS_FULL, BenchCallRow
from tests.cross_pollination_bench_fakes import write_fake_round


@pytest.fixture
def round_data(tmp_path: Path) -> tuple[Manifest, dict[int, MetaculusQuestion]]:
    fake_round = write_fake_round(tmp_path)
    manifest = build_manifest(fake_round.perf_path, fake_round.archive_dir, event_map=None, fetch=fake_round.fetch)
    questions = {entry.question_id: rebuild_question(entry) for entry in manifest.questions}
    return manifest, questions


def _row(
    question_id: int,
    qtype: str,
    model: str,
    draw: int,
    stage: str,
    forecast: float | list[float] | None,
    *,
    status: str = STATUS_SCORED,
    score: float | None = 0.0,
) -> BenchCallRow:
    return BenchCallRow(
        question_id=question_id,
        qtype=qtype,
        model=model,
        draw=draw,
        stage=stage,
        status=status,
        prompt_sha256="prompt",
        forecast=forecast,
        score=score,
    )


def _first_rows(
    question_id: int, qtype: str, forecasts: list[float | list[float]], draw: int = 0
) -> list[BenchCallRow]:
    return [
        _row(question_id, qtype, f"model-{index}", draw, STAGE_FIRST, forecast)
        for index, forecast in enumerate(forecasts)
    ]


def _option_list(options: list[str], probabilities: list[float]) -> PredictedOptionList:
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=option, probability=probability)
            for option, probability in zip(options, probabilities, strict=True)
        ]
    )


def test_binary_median_is_the_production_median(round_data: tuple[Manifest, dict[int, MetaculusQuestion]]) -> None:
    _, questions = round_data
    question = questions[101]
    assert isinstance(question, BinaryQuestion)
    assert median_forecast(question, [0.2, 0.5, 0.9]) == 0.5
    assert median_forecast(question, [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]) == 0.5


def test_mc_median_matches_the_production_function(round_data: tuple[Manifest, dict[int, MetaculusQuestion]]) -> None:
    _, questions = round_data
    question = questions[102]
    assert isinstance(question, MultipleChoiceQuestion)
    ballots = [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]]
    expected = aggregate_multiple_choice_median([_option_list(question.options, ballot) for ballot in ballots])
    assert median_forecast(question, ballots) == pytest.approx(
        [option.probability for option in expected.predicted_options]
    )


@pytest.mark.parametrize("question_id", [103, 104])
def test_numeric_median_matches_aggregate_numeric(
    round_data: tuple[Manifest, dict[int, MetaculusQuestion]], question_id: int
) -> None:
    _, questions = round_data
    question = questions[question_id]
    assert isinstance(question, NumericQuestion)
    cdf_size = question.cdf_size
    cdfs = [
        [0.001 + 0.998 * index / (cdf_size - 1) for index in range(cdf_size)],
        [0.002 + 0.996 * index / (cdf_size - 1) for index in range(cdf_size)],
        [0.003 + 0.994 * index / (cdf_size - 1) for index in range(cdf_size)],
    ]
    distributions = [distribution_from_cdf(cdf, question) for cdf in cdfs]
    expected = aggregate_numeric(distributions, question, "median")
    result = median_forecast(question, cdfs)
    assert result == pytest.approx([point.percentile for point in expected.get_cdf()])


def test_blend_equals_three_quarters_mean_plus_one_quarter_median_for_three_members(
    round_data: tuple[Manifest, dict[int, MetaculusQuestion]],
) -> None:
    _, questions = round_data
    question = questions[101]
    assert isinstance(question, BinaryQuestion)
    assert blend_forecast(question, [0.2, 0.5, 0.9]) == pytest.approx(0.525, abs=1e-3)


def test_blend_of_two_members_is_their_mean(round_data: tuple[Manifest, dict[int, MetaculusQuestion]]) -> None:
    _, questions = round_data
    question = questions[101]
    assert isinstance(question, BinaryQuestion)
    assert blend_forecast(question, [0.2, 0.8]) == pytest.approx(0.5)


def test_aggregate_scores_include_the_mechanical_blend(
    round_data: tuple[Manifest, dict[int, MetaculusQuestion]],
) -> None:
    manifest, _ = round_data
    rows = _first_rows(101, "binary", [0.2, 0.5, 0.9])
    blend = next(score for score in aggregate_scores(rows, manifest) if score.aggregate == MECHANICAL_BLEND)
    assert blend.forecast == pytest.approx(0.525)


def test_a_failed_update_carries_the_first_forecast_forward(
    round_data: tuple[Manifest, dict[int, MetaculusQuestion]],
) -> None:
    manifest, _ = round_data
    rows = _first_rows(101, "binary", [0.2, 0.5, 0.9])
    rows.extend(
        _row(
            101,
            "binary",
            f"model-{index}",
            0,
            VARIANT_PEERS_FULL,
            [0.4, None, 0.8][index],
            status="build_failed" if index == 1 else STATUS_SCORED,
        )
        for index in range(3)
    )
    scores = aggregate_scores(rows, manifest)
    update = next(score for score in scores if score.aggregate == "peers_full_median")
    assert update.n_members == 3
    assert update.n_carried_forward == 1
    assert update.complete_panel is False
    assert panel(rows, 101, 0, VARIANT_PEERS_FULL)[1].carried_forward is True


def test_a_failed_first_call_removes_the_member_everywhere(
    round_data: tuple[Manifest, dict[int, MetaculusQuestion]],
) -> None:
    manifest, _ = round_data
    rows = _first_rows(101, "binary", [0.2, 0.5, 0.9])
    rows[1].status = "build_failed"
    rows.extend(_row(101, "binary", f"model-{index}", 0, VARIANT_PEERS_FULL, 0.6) for index in range(3))
    scores = aggregate_scores(rows, manifest)
    update = next(score for score in scores if score.aggregate == "peers_full_median")
    assert update.n_members == 2
    assert update.complete_panel is False
    assert all(member.model != "model-1" for member in panel(rows, 101, 0, VARIANT_PEERS_FULL))


def test_a_draw_with_no_surviving_member_is_counted_not_scored(
    round_data: tuple[Manifest, dict[int, MetaculusQuestion]],
) -> None:
    manifest, _ = round_data
    rows = [_row(101, "binary", f"model-{index}", 0, STAGE_FIRST, None, status="build_failed") for index in range(3)]
    assert aggregate_scores(rows, manifest) == []
    assert missing_panels(rows, manifest) == [{"question": 101, "draw": 0, "reason": "no_surviving_member"}]
    assert paired_rows([], manifest, treatment="peers_full_median", control=FIRST_MEDIAN) == []


def test_pooled_aggregates_use_six_forecasts(round_data: tuple[Manifest, dict[int, MetaculusQuestion]]) -> None:
    manifest, _ = round_data
    rows = _first_rows(101, "binary", [0.1, 0.2, 0.3], draw=0)
    rows.extend(_first_rows(101, "binary", [0.7, 0.8, 0.9], draw=1))
    scores = aggregate_scores(rows, manifest)
    pooled = [score for score in scores if score.aggregate == POOLED_FIRST_DRAWS]
    assert len(pooled) == 2
    assert [score.forecast for score in pooled] == [0.5, 0.5]
    assert all(score.n_members == 6 for score in pooled)


def test_peer_points_scale(round_data: tuple[Manifest, dict[int, MetaculusQuestion]]) -> None:
    manifest, _ = round_data
    rows = _first_rows(108, "binary", [0.5, 0.5, 0.5])
    rows.extend(_row(108, "binary", f"model-{index}", 0, VARIANT_PEERS_FULL, 0.8) for index in range(3))
    scores = aggregate_scores(rows, manifest)
    differences = paired_rows(scores, manifest, treatment="peers_full_median", control=FIRST_MEDIAN)
    assert differences[0]["difference"] == pytest.approx(100 * math.log(0.8 / 0.5))


def test_paired_rows_carry_event_type_and_draw(round_data: tuple[Manifest, dict[int, MetaculusQuestion]]) -> None:
    manifest, _ = round_data
    rows = _first_rows(101, "binary", [0.5, 0.5, 0.5], draw=1)
    rows.extend(_row(101, "binary", f"model-{index}", 1, VARIANT_PEERS_FULL, 0.8) for index in range(3))
    scores = aggregate_scores(rows, manifest)
    paired = paired_rows(
        scores, manifest, treatment="peers_full_median", control=FIRST_MEDIAN, event_map={101: "event-x"}
    )
    assert paired[0]["unit_id"] == "101"
    assert paired[0]["cluster_id"] == "event-x"
    assert paired[0]["group"] == "binary"
    assert paired[0]["replicate"] == 1
    assert paired[0]["complete_panel"] is True
    incomplete = _first_rows(101, "binary", [0.2, 0.5, 0.9], draw=2)
    incomplete[0].status = "build_failed"
    incomplete.extend(_row(101, "binary", f"model-{index}", 2, VARIANT_PEERS_FULL, 0.8) for index in range(3))
    paired_with_incomplete = paired_rows(
        aggregate_scores([*rows, *incomplete], manifest),
        manifest,
        treatment="peers_full_median",
        control=FIRST_MEDIAN,
        event_map={101: "event-x"},
    )
    complete_only = paired_rows(
        aggregate_scores([*rows, *incomplete], manifest),
        manifest,
        treatment="peers_full_median",
        control=FIRST_MEDIAN,
        complete_only=True,
    )
    assert len(paired_with_incomplete) == 2
    assert len(complete_only) == 1
    assert next(row for row in paired_with_incomplete if row["replicate"] == 2)["complete_panel"] is False


def test_member_rows_are_update_minus_own_first(round_data: tuple[Manifest, dict[int, MetaculusQuestion]]) -> None:
    manifest, _ = round_data
    rows = _first_rows(101, "binary", [0.2, 0.5, 0.9])
    for index, score in enumerate([1.0, 2.0, 3.0]):
        rows[index].score = score
    rows.extend(
        _row(
            101, "binary", f"model-{index}", 0, VARIANT_PEERS_FULL, [0.3, 0.4, 0.8][index], score=[1.1, 1.5, 2.0][index]
        )
        for index in range(3)
    )
    result = member_paired_rows(rows, manifest, variant=VARIANT_PEERS_FULL)
    assert [item["member"] for item in result] == ["model-0", "model-1", "model-2"]
    assert [item["difference"] for item in result] == pytest.approx(
        [0.1 * math.log(2), -0.5 * math.log(2), -1.0 * math.log(2)]
    )
