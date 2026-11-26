"""Recompute cross-pollination aggregates and scores from saved call rows."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import MetaculusQuestion, NumericQuestion

from metaculus_bot.aggregation_strategies import (
    AggregationStrategy,
    combine_binary_predictions,
    combine_multiple_choice_predictions,
    combine_numeric_predictions,
)
from metaculus_bot.mc_processing import clamp_and_renormalize_probs
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution
from metaculus_bot.numeric.validation import resolve_zero_point
from metaculus_bot.scoring_common import baseline_to_peer_factor
from scripts.probes.bench_common.records import STATUS_SCORED
from scripts.probes.bench_common.scoring import Resolution, publish_binary, score_binary, score_mc, score_numeric
from scripts.probes.cross_pollination_bench.manifest import (
    Manifest,
    ManifestQuestion,
    bench_resolution,
    rebuild_question,
)
from scripts.probes.cross_pollination_bench.rows import (
    STAGE_FIRST,
    UPDATE_VARIANTS,
    BenchCallRow,
)

FIRST_MEDIAN = "first_median"
POOLED_FIRST_DRAWS = "pooled_first_draws"
MECHANICAL_BLEND = "mechanical_blend"
PEERS_FULL_POOLED_WITH_FIRST = "peers_full_pooled_with_first"


def variant_median(variant: str) -> str:
    names = {
        "peers_full": "peers_full_median",
        "peers_forecasts": "peers_forecasts_median",
        "second_look": "second_look_median",
    }
    try:
        return names[variant]
    except KeyError as exc:
        raise ValueError(f"unknown update variant {variant!r}") from exc


@dataclass(frozen=True)
class MemberForecast:
    model: str
    forecast: float | list[float]
    score: float
    carried_forward: bool


@dataclass(frozen=True)
class AggregateScore:
    question_id: int
    qtype: str
    draw: int
    aggregate: str
    forecast: float | list[float]
    score: float
    peer_points: float
    n_members: int
    complete_panel: bool
    n_carried_forward: int


def panel(rows: Sequence[BenchCallRow], question_id: int, draw: int, stage: str) -> list[MemberForecast]:
    question_rows = [row for row in rows if row.question_id == question_id and row.draw == draw]
    first_by_model = {row.model: row for row in question_rows if row.stage == STAGE_FIRST}
    if stage == STAGE_FIRST:
        result: list[MemberForecast] = []
        for model, first in first_by_model.items():
            if first.status == STATUS_SCORED and first.forecast is not None and first.score is not None:
                result.append(MemberForecast(model, first.forecast, first.score, carried_forward=False))
        return result
    if stage not in UPDATE_VARIANTS:
        raise ValueError(f"unknown panel stage {stage!r}")
    updates = {row.model: row for row in question_rows if row.stage == stage}
    result = []
    for model, first in first_by_model.items():
        if first.status != STATUS_SCORED or first.forecast is None or first.score is None:
            continue
        update = updates.get(model)
        if (
            update is not None
            and update.status == STATUS_SCORED
            and update.forecast is not None
            and update.score is not None
        ):
            result.append(MemberForecast(model, update.forecast, update.score, carried_forward=False))
        else:
            result.append(MemberForecast(model, first.forecast, first.score, carried_forward=True))
    return result


def distribution_from_cdf(cdf: Sequence[float], question: NumericQuestion) -> NumericDistribution:
    zero_point = resolve_zero_point(question)
    values = build_cdf_value_grid(question.lower_bound, question.upper_bound, zero_point, len(cdf))
    percentiles = [
        Percentile(percentile=float(probability), value=float(value))
        for probability, value in zip(cdf, values, strict=True)
    ]
    return create_pchip_numeric_distribution(list(map(float, cdf)), percentiles, question, zero_point)


def _option_list(question: MultipleChoiceQuestion, forecast: Sequence[float]) -> PredictedOptionList:
    if len(question.options) != len(forecast):
        raise ValueError(f"question {question.id_of_question}: forecast length differs from option count")
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=option, probability=float(probability))
            for option, probability in zip(question.options, forecast, strict=True)
        ]
    )


def median_forecast(question: MetaculusQuestion, forecasts: Sequence[float | list[float]]) -> float | list[float]:
    if not forecasts:
        raise ValueError(f"question {question.id_of_question}: cannot aggregate an empty panel")
    strategy = AggregationStrategy.MEDIAN
    if isinstance(question, BinaryQuestion):
        values = []
        for forecast in forecasts:
            if isinstance(forecast, list):
                raise TypeError(f"question {question.id_of_question}: binary forecast must be a scalar")
            values.append(float(forecast))
        return publish_binary(combine_binary_predictions(values, strategy))
    if isinstance(question, MultipleChoiceQuestion):
        ballots = [_option_list(question, cast(list[float], forecast)) for forecast in forecasts]
        combined = combine_multiple_choice_predictions(ballots, strategy)
        return [float(option.probability) for option in combined.predicted_options]
    if isinstance(question, NumericQuestion):
        distributions = [distribution_from_cdf(cast(list[float], forecast), question) for forecast in forecasts]
        combined = combine_numeric_predictions(distributions, question, strategy)
        return [float(point.percentile) for point in combined.get_cdf()]
    raise TypeError(f"unsupported question class {type(question).__name__}")


def _blend_binary_forecasts(question: BinaryQuestion, forecasts: Sequence[float | list[float]]) -> list[float]:
    values: list[float] = []
    for forecast in forecasts:
        if isinstance(forecast, list):
            raise TypeError(f"question {question.id_of_question}: binary forecast must be a scalar")
        values.append(float(forecast))
    return [
        (forecast + float(statistics.median([other for peer_index, other in enumerate(values) if peer_index != index])))
        / 2
        for index, forecast in enumerate(values)
    ]


def _blend_vector_forecasts(question: MetaculusQuestion, forecasts: Sequence[float | list[float]]) -> list[list[float]]:
    vector_forecasts: list[list[float]] = []
    for forecast in forecasts:
        if not isinstance(forecast, list):
            raise TypeError(f"question {question.id_of_question}: vector forecast must be a list")
        vector_forecasts.append(forecast)
    blended: list[list[float]] = []
    for index, forecast in enumerate(vector_forecasts):
        peers = [other for peer_index, other in enumerate(vector_forecasts) if peer_index != index]
        if any(len(other) != len(forecast) for other in peers):
            raise ValueError(f"question {question.id_of_question}: member forecast vectors have different lengths")
        blended.append(
            [
                (value + float(statistics.median(other[position] for other in peers))) / 2
                for position, value in enumerate(forecast)
            ]
        )
    return blended


def blend_forecast(question: MetaculusQuestion, forecasts: Sequence[float | list[float]]) -> float | list[float]:
    if len(forecasts) < 2:
        raise ValueError("mechanical blend requires at least two members")
    blended: list[float | list[float]] = []
    if isinstance(question, BinaryQuestion):
        blended.extend(_blend_binary_forecasts(question, forecasts))
    else:
        vector_blends = _blend_vector_forecasts(question, forecasts)
        for vector in vector_blends:
            blended.append(
                clamp_and_renormalize_probs(vector) if isinstance(question, MultipleChoiceQuestion) else vector
            )
    return median_forecast(question, blended)


def score_forecast(question: MetaculusQuestion, resolution: Resolution, forecast: float | list[float]) -> float:
    if isinstance(question, BinaryQuestion):
        if isinstance(forecast, list):
            raise TypeError(f"question {question.id_of_question}: binary forecast must be a scalar")
        return score_binary(float(forecast), cast(bool, resolution))
    if isinstance(question, MultipleChoiceQuestion):
        return score_mc(cast(list[float], forecast), question.options, cast(str, resolution))
    if isinstance(question, NumericQuestion):
        if isinstance(resolution, (bool, str)):
            raise TypeError(f"question {question.id_of_question}: numeric resolution must be numeric or out of bounds")
        return score_numeric(cast(list[float], forecast), question, resolution)
    raise TypeError(f"unsupported question class {type(question).__name__}")


def _aggregate_score(
    entry: ManifestQuestion,
    question: MetaculusQuestion,
    resolution: Resolution,
    *,
    draw: int,
    aggregate: str,
    forecast: float | list[float],
    members: Sequence[MemberForecast],
    complete: bool,
) -> AggregateScore:
    score = score_forecast(question, resolution, forecast)
    peer_points = score * baseline_to_peer_factor(entry.qtype, n_options=entry.n_options)
    return AggregateScore(
        entry.question_id,
        entry.qtype,
        draw,
        aggregate,
        forecast,
        score,
        peer_points,
        len(members),
        complete,
        sum(member.carried_forward for member in members),
    )


def _aggregate_draw(
    rows: Sequence[BenchCallRow],
    entry: ManifestQuestion,
    *,
    question: MetaculusQuestion,
    resolution: Resolution,
    draw: int,
) -> tuple[list[AggregateScore], list[MemberForecast]]:
    first_members = panel(rows, entry.question_id, draw, STAGE_FIRST)
    if not first_members:
        return [], first_members
    complete_first = len(first_members) == 3
    scores = [
        _aggregate_score(
            entry,
            question,
            resolution,
            draw=draw,
            aggregate=FIRST_MEDIAN,
            forecast=median_forecast(question, [member.forecast for member in first_members]),
            members=first_members,
            complete=complete_first,
        )
    ]
    if len(first_members) >= 2:
        scores.append(
            _aggregate_score(
                entry,
                question,
                resolution,
                draw=draw,
                aggregate=MECHANICAL_BLEND,
                forecast=blend_forecast(question, [member.forecast for member in first_members]),
                members=first_members,
                complete=complete_first,
            )
        )
    for variant in UPDATE_VARIANTS:
        update_members = panel(rows, entry.question_id, draw, variant)
        if not update_members:
            continue
        complete_update = len(update_members) == 3 and not any(member.carried_forward for member in update_members)
        scores.append(
            _aggregate_score(
                entry,
                question,
                resolution,
                draw=draw,
                aggregate=variant_median(variant),
                forecast=median_forecast(question, [member.forecast for member in update_members]),
                members=update_members,
                complete=complete_update,
            )
        )
    peers_full = panel(rows, entry.question_id, draw, "peers_full")
    if peers_full:
        pooled_members = [*first_members, *peers_full]
        scores.append(
            _aggregate_score(
                entry,
                question,
                resolution,
                draw=draw,
                aggregate=PEERS_FULL_POOLED_WITH_FIRST,
                forecast=median_forecast(question, [member.forecast for member in pooled_members]),
                members=pooled_members,
                complete=len(first_members) == 3
                and len(peers_full) == 3
                and not any(member.carried_forward for member in peers_full),
            )
        )
    return scores, first_members


def _pooled_first_scores(
    entry: ManifestQuestion,
    *,
    question: MetaculusQuestion,
    resolution: Resolution,
    draws: Sequence[int],
    first_panels: Mapping[int, Sequence[MemberForecast]],
) -> list[AggregateScore]:
    if len(draws) != 2:
        return []
    pooled_members = [member for draw in draws for member in first_panels[draw]]
    if not pooled_members:
        return []
    forecast = median_forecast(question, [member.forecast for member in pooled_members])
    complete = all(len(first_panels[draw]) == 3 for draw in draws)
    return [
        _aggregate_score(
            entry,
            question,
            resolution,
            draw=draw,
            aggregate=POOLED_FIRST_DRAWS,
            forecast=forecast,
            members=pooled_members,
            complete=complete,
        )
        for draw in draws
    ]


def aggregate_scores(rows: Sequence[BenchCallRow], manifest: Manifest) -> list[AggregateScore]:
    entries = {entry.question_id: entry for entry in manifest.questions}
    question_ids = {row.question_id for row in rows}
    unknown = question_ids - entries.keys()
    if unknown:
        raise ValueError(f"call rows contain questions absent from manifest: {sorted(unknown)}")
    draws_by_question: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        if row.qtype != entries[row.question_id].qtype:
            raise ValueError(
                f"question {row.question_id}: call row type {row.qtype!r} differs from manifest type "
                f"{entries[row.question_id].qtype!r}"
            )
        if row.stage != STAGE_FIRST and row.stage not in UPDATE_VARIANTS:
            raise ValueError(f"question {row.question_id}: unknown call stage {row.stage!r}")
        draws_by_question[row.question_id].add(row.draw)
    scores: list[AggregateScore] = []
    for question_id in sorted(draws_by_question):
        draw_set = draws_by_question[question_id]
        entry = entries[question_id]
        question = rebuild_question(entry)
        resolution = bench_resolution(entry, question)
        draws = sorted(draw_set)
        first_panels: dict[int, Sequence[MemberForecast]] = {}
        for draw in draws:
            draw_scores, first_members = _aggregate_draw(
                rows, entry, question=question, resolution=resolution, draw=draw
            )
            scores.extend(draw_scores)
            first_panels[draw] = first_members
        scores.extend(
            _pooled_first_scores(
                entry,
                question=question,
                resolution=resolution,
                draws=draws,
                first_panels=first_panels,
            )
        )
    return scores


def missing_panels(rows: Sequence[BenchCallRow], manifest: Manifest) -> list[dict[str, Any]]:
    entry_ids = {entry.question_id for entry in manifest.questions}
    unknown = {row.question_id for row in rows} - entry_ids
    if unknown:
        raise ValueError(f"call rows contain questions absent from manifest: {sorted(unknown)}")
    panels = {(row.question_id, row.draw) for row in rows}
    return [
        {"question": question_id, "draw": draw, "reason": "no_surviving_member"}
        for question_id, draw in sorted(panels)
        if not panel(rows, question_id, draw, STAGE_FIRST)
    ]


def paired_rows(
    scores: Sequence[AggregateScore],
    manifest: Manifest,
    *,
    treatment: str,
    control: str,
    complete_only: bool = False,
    event_map: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    entries = {entry.question_id: entry for entry in manifest.questions}
    grouped: dict[tuple[int, int], dict[str, AggregateScore]] = defaultdict(dict)
    for score in scores:
        grouped[(score.question_id, score.draw)][score.aggregate] = score
    result: list[dict[str, Any]] = []
    for (question_id, draw), aggregates in sorted(grouped.items()):
        if treatment not in aggregates or control not in aggregates:
            continue
        treatment_score = aggregates[treatment]
        control_score = aggregates[control]
        complete = treatment_score.complete_panel and control_score.complete_panel
        if complete_only and not complete:
            continue
        entry = entries[question_id]
        event_key = event_map.get(question_id, entry.event_key) if event_map is not None else entry.event_key
        result.append(
            {
                "unit_id": str(question_id),
                "cluster_id": event_key,
                "group": treatment_score.qtype,
                "replicate": draw,
                "difference": treatment_score.peer_points - control_score.peer_points,
                "member": None,
                "complete_panel": complete,
                "after_release": entry.after_release,
            }
        )
    return result


def member_paired_rows(
    rows: Sequence[BenchCallRow],
    manifest: Manifest,
    *,
    variant: str,
    event_map: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    if variant not in UPDATE_VARIANTS:
        raise ValueError(f"unknown update variant {variant!r}")
    entries = {entry.question_id: entry for entry in manifest.questions}
    if {row.question_id for row in rows} - entries.keys():
        raise ValueError("call rows contain questions absent from manifest")
    by_key = {(row.question_id, row.draw, row.model, row.stage): row for row in rows}
    result: list[dict[str, Any]] = []
    for (question_id, draw, model, stage), update in by_key.items():
        if stage != variant:
            continue
        first = by_key.get((question_id, draw, model, STAGE_FIRST))
        if (
            first is None
            or first.status != STATUS_SCORED
            or first.score is None
            or update.status != STATUS_SCORED
            or update.score is None
        ):
            continue
        entry = entries[question_id]
        factor = baseline_to_peer_factor(entry.qtype, n_options=entry.n_options)
        complete_panel = (
            len(panel(rows, question_id, draw, STAGE_FIRST)) == 3
            and len(panel(rows, question_id, draw, variant)) == 3
            and not any(member.carried_forward for member in panel(rows, question_id, draw, variant))
        )
        result.append(
            {
                "unit_id": str(question_id),
                "cluster_id": event_map.get(question_id, entry.event_key) if event_map is not None else entry.event_key,
                "group": entry.qtype,
                "replicate": draw,
                "difference": (update.score - first.score) * factor,
                "member": model,
                "complete_panel": complete_panel,
                "after_release": entry.after_release,
            }
        )
    return result
