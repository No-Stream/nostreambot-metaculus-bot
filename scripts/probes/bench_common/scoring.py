"""Scoring a bench reply the production way: the extraction ladder, the CDF build, the platform log score.

The three per-type paths mirror ``forecaster_runners`` minus its retries: the ladder reads the fenced block
(salvaging through the parser LLM when it must), a binary value is clamped as published, a ballot is
clamped and renormalized, and a percentile set goes through the runners' own guarded build (sanitizer,
CDF build, fail-shut unit-mismatch guard) before its CDF is scored.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from forecasting_tools import BinaryQuestion, GeneralLlm, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import MetaculusQuestion, OutOfBoundsResolution

from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN
from metaculus_bot.forecaster_runners import (
    BINARY_PARSE_NOTES,
    build_guarded_numeric_distribution,
    build_mc_parse_notes,
    build_parse_notes,
)
from metaculus_bot.member_forecast import option_vector
from metaculus_bot.numeric.utils import clamp_and_renormalize_mc
from metaculus_bot.scoring_common import binary_log_score, mc_log_score, numeric_log_score
from metaculus_bot.value_extraction import extract_binary, extract_mc, extract_numeric

logger = logging.getLogger(__name__)

Resolution = bool | float | OutOfBoundsResolution | str


def _resolution_float(question: NumericQuestion, resolution: float | OutOfBoundsResolution) -> float:
    """An out-of-bounds resolution lands one unit past the bound, as ``backtest.scoring`` places it."""
    if resolution is OutOfBoundsResolution.ABOVE_UPPER_BOUND:
        return question.upper_bound + 1.0
    if resolution is OutOfBoundsResolution.BELOW_LOWER_BOUND:
        return question.lower_bound - 1.0
    return float(resolution)


def publish_binary(prob: float) -> float:
    """The probability as the runner publishes it: clamped to the platform's binary range."""
    return max(BINARY_PROB_MIN, min(BINARY_PROB_MAX, prob))


def score_binary(prob: float, outcome: bool) -> float:
    return binary_log_score(publish_binary(prob), outcome)


def score_mc(vector: Sequence[float], options: Sequence[str], correct: str) -> float:
    return mc_log_score(list(vector), list(options).index(correct))


def score_numeric(cdf: Sequence[float], question: NumericQuestion, resolution: float | OutOfBoundsResolution) -> float:
    return numeric_log_score(
        list(cdf),
        _resolution_float(question, resolution),
        question.lower_bound,
        question.upper_bound,
        open_lower_bound=question.open_lower_bound,
        open_upper_bound=question.open_upper_bound,
        zero_point=question.zero_point,
    )


def score_published(published: dict[str, Any], question: MetaculusQuestion, resolution: Resolution) -> float:
    """The score of the forecast the bot actually published, the reference level beside the bench arms."""
    if isinstance(question, BinaryQuestion):
        assert isinstance(resolution, bool)
        return score_binary(float(published["prob_yes"]), resolution)
    values = [float(v) for v in published["forecast_values"]]
    if isinstance(question, MultipleChoiceQuestion):
        assert isinstance(resolution, str)
        return score_mc(values, question.options, resolution)
    assert isinstance(question, NumericQuestion)
    assert not isinstance(resolution, (bool, str))
    return score_numeric(values, question, resolution)


@dataclass(frozen=True)
class Scored:
    """A reply's score, the value it was scored on, and which ladder rung recovered that value."""

    score: float
    forecast: Any
    rung: str
    block_present: bool
    cdf: list[float] | None = None
    stated_percentiles: list[list[float]] | None = None


async def _score_binary_reply(
    question: BinaryQuestion, outcome: bool, text: str, parser_llm: GeneralLlm, *, model_name: str
) -> Scored:
    extracted = await extract_binary(
        text, parser_llm, prompt_notes=BINARY_PARSE_NOTES, question_id=question.id_of_question, model_name=model_name
    )
    published = publish_binary(extracted.value)
    return Scored(binary_log_score(published, outcome), published, extracted.rung, extracted.block_present)


async def _score_mc_reply(
    question: MultipleChoiceQuestion, correct: str, text: str, parser_llm: GeneralLlm, *, model_name: str
) -> Scored:
    options = list(question.options)
    extracted = await extract_mc(
        text,
        options,
        parser_llm,
        prompt_notes=build_mc_parse_notes(options),
        question_id=question.id_of_question,
        model_name=model_name,
    )
    option_list = extracted.value.option_list
    try:
        option_list = clamp_and_renormalize_mc(option_list)
    except ValueError as exc:
        logger.warning(
            "MC clamp/renormalize failed for question %s, scoring the raw ballot: %s", question.id_of_question, exc
        )
    vector = option_vector(option_list)
    return Scored(score_mc(vector, options, correct), vector, extracted.rung, extracted.block_present)


async def _score_numeric_reply(
    question: NumericQuestion,
    resolution: float | OutOfBoundsResolution,
    text: str,
    parser_llm: GeneralLlm,
    *,
    model_name: str,
) -> Scored:
    """The ``forecast`` recorded is the distribution's own declaration: the sanitized percentiles on the
    201-point grid, the value-axis CDF on a coarser one (``numeric.pipeline._build_discrete_distribution``)."""
    extracted = await extract_numeric(
        text,
        parser_llm,
        prompt_notes=build_parse_notes(question),
        question_id=question.id_of_question,
        model_name=model_name,
        open_lower_bound=question.open_lower_bound,
        open_upper_bound=question.open_upper_bound,
    )
    prediction = build_guarded_numeric_distribution(
        extracted.value, question, model_name=model_name, tail_probabilities=extracted.tail_probabilities
    )
    cdf = [float(point.percentile) for point in prediction.get_cdf()]
    declared = [[float(p.percentile), float(p.value)] for p in prediction.declared_percentiles]
    stated_percentiles = [[float(point.percentile), float(point.value)] for point in extracted.value]
    return Scored(
        score_numeric(cdf, question, resolution),
        declared,
        extracted.rung,
        extracted.block_present,
        cdf,
        stated_percentiles,
    )


async def score_reply(
    question: MetaculusQuestion, resolution: Resolution, text: str, parser_llm: GeneralLlm, *, model_name: str
) -> Scored:
    """Run the extraction ladder on the reply and score the value it recovers, the production way per type.

    Raises ``ValueExtractionError`` when every rung fails, ``UnitMismatchError`` when the guard trips, and
    ``ValueError`` when the sanitizer or the CDF build rejects the declaration; the caller records each as a
    row status.
    """
    if isinstance(question, BinaryQuestion):
        assert isinstance(resolution, bool)
        return await _score_binary_reply(question, resolution, text, parser_llm, model_name=model_name)
    if isinstance(question, MultipleChoiceQuestion):
        assert isinstance(resolution, str)
        return await _score_mc_reply(question, resolution, text, parser_llm, model_name=model_name)
    assert isinstance(question, NumericQuestion)
    assert not isinstance(resolution, (bool, str))
    return await _score_numeric_reply(question, resolution, text, parser_llm, model_name=model_name)
