"""Per-type forecasting functions extracted from TemplateForecaster.

Each function takes a question, research context, a forecaster LLM (for
generation), and a parser LLM (for structured extraction), then returns
the appropriate ReasonedPrediction.

These are stateless — the caller is responsible for storing any side-effects
(like discrete integer votes for numeric questions).

The numeric and date runners share one branch: a question ``numeric.config.elicit_per_bin``
admits (a Mantic grid of 31 bins or fewer) is elicited PER BIN by ``_run_pmf_forecast``, one
probability per labelled bin, and its distribution is built straight from that declaration
(``numeric.pmf_cdf``) with none of the percentile machinery in between.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
)
from forecasting_tools.ai_models.ai_utils.openai_utils import VisionMessageData
from forecasting_tools.data_models.questions import DateQuestion
from pydantic import ValidationError

from metaculus_bot.constants import (
    BINARY_PROB_MAX,
    BINARY_PROB_MIN,
    FORECASTER_SOFT_DEADLINE,
    PMF_ABOVE_RANGE_KEY,
    PMF_BELOW_RANGE_KEY,
)
from metaculus_bot.exceptions import UnitMismatchError
from metaculus_bot.llm_retry import invoke_with_broad_retry
from metaculus_bot.member_forecast import (
    ELICITATION_PMF,
    MEMBER_FORECAST_ROLE_MEMBER,
    format_member_forecast_marker,
    option_vector,
    out_of_range_mass,
    percentile_pairs,
)
from metaculus_bot.numeric.config import EXPECTED_PERCENTILE_COUNT, STANDARD_PERCENTILES_CSV, elicit_per_bin
from metaculus_bot.numeric.date_axis import EpochDateQuestion, as_epoch_question, format_epoch, numeric_qtype
from metaculus_bot.numeric.diagnostics import log_final_prediction, log_open_bound_piling_diagnostics
from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric.pmf_cdf import build_pmf_distribution, published_pmf
from metaculus_bot.numeric.pmf_grid import PmfGrid, pmf_grid
from metaculus_bot.numeric.roundtrip import format_percentile_roundtrip_marker
from metaculus_bot.numeric.utils import bound_messages, clamp_and_renormalize_mc, pmf_bound_messages
from metaculus_bot.numeric.validation import detect_unit_mismatch
from metaculus_bot.prompts import binary_prompt, date_prompt, multiple_choice_prompt, numeric_prompt, pmf_prompt
from metaculus_bot.structured_output_schema import NumericStructured, parse_structured_block
from metaculus_bot.structured_parse import parse_structured
from metaculus_bot.value_extraction import extract_binary, extract_date, extract_mc, extract_numeric, extract_pmf

logger = logging.getLogger(__name__)


def _log_llm_output(model_name: str, question_id: int | None, reasoning: str) -> None:
    logger.info(
        f"""
\n\n
========================================
LLM OUTPUT | Model: {model_name} | Question: {question_id} | Length: {len(reasoning)} chars
========================================
{reasoning}
========================================
END LLM OUTPUT | {model_name}
========================================
\n\n
"""
    )


def _forecaster_input(prompt: str, chart_b64: str | None) -> str | VisionMessageData:
    """Return the invoke input for a forecaster: the bare prompt, or a
    ``VisionMessageData`` carrying the prompt + the time-series-anchor chart PNG.

    When ``chart_b64`` is provided (TS_ANCHOR_CHART_ENABLED on), the base model
    sees the chart alongside the prompt at low resolution — enough to read the
    band/level, cheap on tokens. All roster models are vision-capable via
    OpenRouter. The downstream parser/extraction path only ever consumes the
    model's reasoning TEXT, so it is untouched by this. Only the base forecaster
    generation sees the image — never the stacker, summarizer, or gap-fill.
    """
    if chart_b64 is None:
        return prompt
    return VisionMessageData(prompt=prompt, b64_image=chart_b64, image_resolution="low")


# The parser LLM's extraction instructions for a binary rationale (the salvage rung of the ladder).
BINARY_PARSE_NOTES: str = (
    "Return a single JSON object only. Set `prediction_in_decimal` strictly as a decimal in [0,1] "
    "(e.g., 0.17 for 17%). If the text contains 'Probability: NN%' or 'NN %', set `prediction_in_decimal` to NN/100. "
    "Do not return percentages, strings, or any extra fields."
)


def build_mc_parse_notes(options: Sequence[str]) -> str:
    """The parser LLM's extraction instructions for a multiple-choice rationale, naming the question's options."""
    return clean_indents(
        f"""
        Output a JSON array of objects with exactly these two keys per item: `option_name` (string) and `probability` (decimal in [0,1]).
        Use option names exactly from this list (case-insensitive match is OK, but prefer canonical spelling):
        {list(options)}
        Do not include any options beyond this list. If the source text prefixes with words like 'Option A:' remove the prefix.
        Ensure the probabilities approximately sum to 1.0; slight floating-point drift is OK.
        """
    )


def build_parse_notes(question: NumericQuestion) -> str:
    """Build the parser LLM's extraction instructions for a numeric question.

    The parser's only job is to EXTRACT the forecaster's declared percentile values
    and convert unit suffixes (350B -> 350000000000). It must NOT interpret or
    constrain them. For a CLOSED bound the outcome genuinely cannot cross it, so a
    hard sanity note ("at or above/below {bound}") is fine. For an OPEN bound the
    displayed bound is only the bottom/top of the shown range — the outcome can
    resolve outside it — so the parser must preserve out-of-range values verbatim
    and never clamp them into range. The downstream sanitize layer
    (numeric/bounds_clamping.py) already gates its clamps on open/closed, so this is
    the one remaining place that must respect that distinction.
    """
    unit_str = question.unit_of_measure or "base unit"

    if question.open_lower_bound:
        lower_note = (
            f"The lower bound {question.lower_bound} is only the bottom of the displayed range; the outcome can "
            f"resolve below it. If the forecaster's text states a value below {question.lower_bound}, extract that "
            "value verbatim — never clamp or round it up into range."
        )
    else:
        lower_note = f"Values are at or above the lower bound {question.lower_bound}."

    if question.open_upper_bound:
        upper_note = (
            f"The upper bound {question.upper_bound} is only the top of the displayed range; the outcome can "
            f"resolve above it. If the forecaster's text states a value above {question.upper_bound}, extract that "
            "value verbatim — never clamp or round it down into range."
        )
    else:
        upper_note = f"Values are at or below the upper bound {question.upper_bound}."

    return (
        f"Return exactly these {EXPECTED_PERCENTILE_COUNT} percentiles and no others: {STANDARD_PERCENTILES_CSV}. "
        "Do not include 0 or 100. Use keys 'percentile' (decimal in [0,1]) and 'value' (float). "
        f"Values must be in the base unit '{unit_str}'. The displayed range is [{question.lower_bound}, "
        f"{question.upper_bound}] — use it only to infer scale, not as a constraint. "
        f"{lower_note} {upper_note} "
        "If your text uses B/M/k, convert numerically to base unit (e.g., 350B → 350000000000). No suffixes."
    )


def build_date_parse_notes(question: EpochDateQuestion) -> str:
    """The parser LLM's extraction instructions for a date question.

    The date sibling of ``build_parse_notes``, with the same one job (EXTRACT the declared
    dates, never interpret them) and the same open-versus-closed distinction, which matters
    MORE here: an open upper bound is the norm on a date question, and a date after it is the
    forecaster saying "it may not happen inside the window at all", so it must be kept verbatim.
    The value format is strict ISO-8601 in UTC: a bare year or a year-month is ambiguous on a
    day grid and is refused rather than guessed.
    """
    granularity = question.date_granularity
    lower_label = format_epoch(question.lower_bound, granularity)
    upper_label = format_epoch(question.upper_bound, granularity)

    if question.open_lower_bound:
        lower_note = (
            f"The lower bound {lower_label} is only the start of the displayed range; the outcome can resolve "
            f"before it. If the forecaster's text states a date before {lower_label}, extract that date "
            "verbatim — never move it later into range."
        )
    else:
        lower_note = f"Dates are at or after the lower bound {lower_label}."

    if question.open_upper_bound:
        upper_note = (
            f"The upper bound {upper_label} is only the end of the displayed range; the outcome can resolve "
            f"after it, which is how the forecaster says the event may not happen inside the window. If the "
            f"forecaster's text states a date after {upper_label}, extract that date verbatim — never move it "
            "earlier into range."
        )
    else:
        upper_note = f"Dates are at or before the upper bound {upper_label}."

    return (
        f"Return exactly these {EXPECTED_PERCENTILE_COUNT} percentiles and no others: {STANDARD_PERCENTILES_CSV}. "
        "Do not include 0 or 100. Use keys 'percentile' (decimal in [0,1]) and 'value' (an ISO-8601 string in "
        "UTC: 'YYYY-MM-DD' for a calendar day, or 'YYYY-MM-DDTHH:MM:SSZ' when the forecaster gave a time). "
        "Never return a bare year, a year-month or a number; if the forecaster named only a month, use the date "
        f"they most plausibly meant within it. The displayed range is [{lower_label}, {upper_label}] — use it only "
        f"to read which century and year the forecaster means, not as a constraint. {lower_note} {upper_note}"
    )


def build_pmf_parse_notes(grid: PmfGrid) -> str:
    """The parser LLM's extraction instructions for a per-bin declaration.

    The per-bin sibling of ``build_parse_notes``, with the same one job: EXTRACT the probability the
    forecaster stated for each key, never interpret or rebalance it. The salvage rung has nothing
    else to spell the keys from, so they are listed verbatim: a label that matches no grid key fails
    the conversion, and every key must be present (``value_extraction.extract_pmf``). A key the
    forecaster never priced is therefore LEFT OUT rather than written as 0, so the every-key rule
    drops the member instead of publishing a tail nobody declared.
    """
    keys = ", ".join(f"'{key}'" for key in grid.keys)
    reserved: list[str] = []
    if grid.open_lower_bound:
        reserved.append(f"'{PMF_BELOW_RANGE_KEY}' is the probability that the outcome falls below the displayed range")
    if grid.open_upper_bound:
        reserved.append(f"'{PMF_ABOVE_RANGE_KEY}' is the probability that the outcome falls above the displayed range")
    reserved_note = f" {'; '.join(reserved)}." if reserved else ""
    return (
        "Return a JSON array of objects, one per key the forecaster gave a probability, each with exactly two "
        f"fields: 'label' (string) and 'probability' (decimal in [0,1]). The grid has exactly {len(grid.keys)} keys; "
        f"use these labels verbatim, in this order, and no others: {keys}. Each label names one bin of the "
        f"question's grid, spelled exactly as listed.{reserved_note} Read each probability as the forecaster stated "
        "it (a percentage becomes a decimal: 35% -> 0.35); a bin the forecaster ruled out is 0. Never interpret, "
        "rebalance or fill in a probability the forecaster did not state: leave that key out, even if the array is "
        "then shorter than the key list. The probabilities should sum to about 1.0."
    )


async def run_binary_forecast(
    question: BinaryQuestion,
    research: str,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    *,
    chart_b64: str | None = None,
) -> ReasonedPrediction[float]:
    prompt = binary_prompt(question, research)
    forecaster_input = _forecaster_input(prompt, chart_b64)
    # The forecasters' SOLE retry layer (allowed_tries=1 in llm_configs.py); the 30s gate is in llm_retry.py.
    reasoning = await invoke_with_broad_retry(
        lambda: forecaster_llm.invoke(forecaster_input),
        wall_timeout=FORECASTER_SOFT_DEADLINE,
        label="forecaster_binary",
    )
    _log_llm_output(forecaster_llm.model, question.id_of_question, reasoning)

    outcome = await extract_binary(
        reasoning,
        parser_llm,
        prompt_notes=BINARY_PARSE_NOTES,
        question_id=question.id_of_question,
        model_name=forecaster_llm.model,
    )

    decimal_pred = max(BINARY_PROB_MIN, min(BINARY_PROB_MAX, outcome.value))
    logger.info(
        format_member_forecast_marker(
            question_id=question.id_of_question,
            model=forecaster_llm.model,
            role=MEMBER_FORECAST_ROLE_MEMBER,
            qtype="binary",
            raw=outcome.value,
            published=decimal_pred,
        )
    )

    logger.info(f"Forecasted URL {question.page_url} with prediction: {decimal_pred}")
    return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)


async def run_mc_forecast(
    question: MultipleChoiceQuestion,
    research: str,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    *,
    chart_b64: str | None = None,
) -> ReasonedPrediction[PredictedOptionList]:
    prompt = multiple_choice_prompt(question, research)
    forecaster_input = _forecaster_input(prompt, chart_b64)
    # Broad, 30s-gated retry — see run_binary_forecast for the rationale.
    reasoning = await invoke_with_broad_retry(
        lambda: forecaster_llm.invoke(forecaster_input), wall_timeout=FORECASTER_SOFT_DEADLINE, label="forecaster_mc"
    )
    _log_llm_output(forecaster_llm.model, question.id_of_question, reasoning)

    outcome = await extract_mc(
        reasoning,
        list(question.options),
        parser_llm,
        prompt_notes=build_mc_parse_notes(question.options),
        question_id=question.id_of_question,
        model_name=forecaster_llm.model,
    )
    predicted_option_list = outcome.value.option_list
    try:
        predicted_option_list = clamp_and_renormalize_mc(predicted_option_list)
    except ValueError as e:
        logger.warning(f"MC clamp/renormalize failed, using raw predictions: {e}")
    logger.info(
        format_member_forecast_marker(
            question_id=question.id_of_question,
            model=forecaster_llm.model,
            role=MEMBER_FORECAST_ROLE_MEMBER,
            qtype="multiple_choice",
            raw=outcome.value.declared_probs,  # the list is clamped on construction; see McForecast
            published=option_vector(predicted_option_list),
        )
    )

    logger.info(f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}")
    return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)


async def run_numeric_forecast(
    question: NumericQuestion,
    research: str,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    *,
    chart_b64: str | None = None,
) -> tuple[ReasonedPrediction[NumericDistribution], bool | None]:
    """Run a numeric forecast and return (prediction, discrete_vote).

    The caller is responsible for storing the discrete_vote in
    _discrete_integer_votes if needed. A question elicited per bin (``elicit_per_bin``) casts no
    vote: a per-bin block has no ``outcome_type``, and the discrete snap is already skipped on every
    outcome-space grid.
    """
    if elicit_per_bin(question):
        per_bin = await _run_pmf_forecast(
            question, research, forecaster_llm, parser_llm, chart_b64=chart_b64, label="forecaster_numeric"
        )
        return per_bin, None

    upper_bound_message, lower_bound_message = bound_messages(question)
    prompt = numeric_prompt(question, research, lower_bound_message, upper_bound_message)
    forecaster_input = _forecaster_input(prompt, chart_b64)
    # Broad, 30s-gated retry — see run_binary_forecast for the rationale.
    reasoning = await invoke_with_broad_retry(
        lambda: forecaster_llm.invoke(forecaster_input),
        wall_timeout=FORECASTER_SOFT_DEADLINE,
        label="forecaster_numeric",
    )

    _log_llm_output(forecaster_llm.model, question.id_of_question, reasoning)

    qid = question.id_of_question
    discrete_vote = await _resolve_discrete_vote(reasoning, parser_llm, forecaster_llm, qid)

    parse_notes = build_parse_notes(question)

    outcome = await extract_numeric(
        reasoning,
        parser_llm,
        prompt_notes=parse_notes,
        question_id=qid,
        model_name=forecaster_llm.model,
    )
    prediction = build_guarded_numeric_distribution(outcome.value, question, model_name=forecaster_llm.model)
    return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning), discrete_vote


async def run_date_forecast(
    question: DateQuestion,
    research: str,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    *,
    chart_b64: str | None = None,
) -> ReasonedPrediction[NumericDistribution]:
    """Run a date forecast: the numeric runner on the question's epoch-seconds view.

    The prompt and the parse notes read as dates; the extracted percentiles are epoch seconds
    and go through the SAME guarded numeric build as a numeric question, on the adapter
    (``numeric.date_axis.as_epoch_question``), so the published distribution carries
    ``is_date`` and the comment renders dates. No discrete-integer vote: integer snapping on an
    epoch axis is meaningless. A coarse Mantic date grid (``elicit_per_bin`` on the epoch view,
    which carries ``page_url``) is elicited per calendar-day bin instead.
    """
    epoch_question = as_epoch_question(question)
    if elicit_per_bin(epoch_question):
        return await _run_pmf_forecast(
            epoch_question, research, forecaster_llm, parser_llm, chart_b64=chart_b64, label="forecaster_date"
        )

    upper_bound_message, lower_bound_message = bound_messages(epoch_question)
    prompt = date_prompt(epoch_question, research, lower_bound_message, upper_bound_message)
    forecaster_input = _forecaster_input(prompt, chart_b64)
    # Broad, 30s-gated retry — see run_binary_forecast for the rationale.
    reasoning = await invoke_with_broad_retry(
        lambda: forecaster_llm.invoke(forecaster_input),
        wall_timeout=FORECASTER_SOFT_DEADLINE,
        label="forecaster_date",
    )
    _log_llm_output(forecaster_llm.model, question.id_of_question, reasoning)

    outcome = await extract_date(
        reasoning,
        parser_llm,
        prompt_notes=build_date_parse_notes(epoch_question),
        question_id=question.id_of_question,
        model_name=forecaster_llm.model,
    )
    prediction = build_guarded_numeric_distribution(outcome.value, epoch_question, model_name=forecaster_llm.model)
    return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)


async def _run_pmf_forecast(
    view: NumericQuestion,
    research: str,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    *,
    chart_b64: str | None,
    label: str,
) -> ReasonedPrediction[NumericDistribution]:
    """Elicit ``view`` per bin and build the distribution straight from the declared PMF.

    ``view`` is the numeric-pipeline view of the question: the epoch adapter for a date question
    (so the marker says ``qtype=date`` and the distribution carries ``is_date``), the question
    itself otherwise. The grid, the prompt, the ladder, the build and the marker all read it.

    Nothing that consumes percentiles runs here, each for a reason. The discrete vote is a paid
    parser call on a block that has no ``outcome_type``, and the snap it feeds is already skipped
    on every outcome-space grid. ``sanitize_percentiles`` requires the 13 standard percentiles; the
    one repair a per-bin declaration needs, lifting a declared zero to the platform minimum, is the
    floor blend inside ``build_pmf_distribution``. ``detect_unit_mismatch`` guards values declared
    in the wrong unit; a per-bin declaration states no values, only mass on labelled bins, and on
    grid input its ratios pass trivially, so routing the grid through it would be a fail-open guard.
    The soft deadline, the broad retry and the drop classification are the percentile runners' own.
    """
    grid = pmf_grid(view)
    upper_bound_message, lower_bound_message = pmf_bound_messages(view)
    prompt = pmf_prompt(view, research, lower_bound_message, upper_bound_message)
    forecaster_input = _forecaster_input(prompt, chart_b64)
    # Broad, 30s-gated retry — see run_binary_forecast for the rationale.
    reasoning = await invoke_with_broad_retry(
        lambda: forecaster_llm.invoke(forecaster_input), wall_timeout=FORECASTER_SOFT_DEADLINE, label=label
    )
    _log_llm_output(forecaster_llm.model, view.id_of_question, reasoning)

    outcome = await extract_pmf(
        reasoning,
        grid,
        parser_llm,
        prompt_notes=build_pmf_parse_notes(grid),
        question_id=view.id_of_question,
        model_name=forecaster_llm.model,
    )
    prediction = build_pmf_distribution(outcome.value.declared, view, model_name=forecaster_llm.model)
    logger.info(
        format_member_forecast_marker(
            question_id=view.id_of_question,
            model=forecaster_llm.model,
            role=MEMBER_FORECAST_ROLE_MEMBER,
            qtype=numeric_qtype(view),
            raw=outcome.value.declared,
            published=published_pmf(prediction),
            out_of_range=out_of_range_mass(prediction),
            elicitation=ELICITATION_PMF,
        )
    )
    log_final_prediction(prediction, view)
    return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)


async def _resolve_discrete_vote(
    reasoning: str, parser_llm: GeneralLlm, forecaster_llm: GeneralLlm, qid: int | None
) -> bool | None:
    """This forecaster's DISCRETE-vs-CONTINUOUS vote, or None when it could not be read.

    C3: read outcome_type from the structured JSON block first (saves a parser LLM call).
    Fall back to the OutcomeTypeResult parser call when the block is missing or doesn't
    declare outcome_type.
    """
    discrete_vote: bool | None = None
    block = parse_structured_block(reasoning, "numeric")
    if isinstance(block, NumericStructured) and block.outcome_type is not None:
        discrete_vote = block.outcome_type == "discrete_integer"
    else:
        try:
            outcome_result: OutcomeTypeResult = await parse_structured(
                reasoning,
                OutcomeTypeResult,
                parser_llm,
                prompt_notes=(
                    "The forecaster classified whether this question's resolution values are discrete "
                    "integers (OUTCOME_TYPE: DISCRETE) or continuous real numbers (OUTCOME_TYPE: CONTINUOUS). "
                    "Return is_discrete_integer=true if the forecaster said DISCRETE, false if CONTINUOUS."
                ),
            )
            discrete_vote = outcome_result.is_discrete_integer
        except (ValidationError, ValueError) as e:
            logger.warning("Failed to parse OUTCOME_TYPE for Q %s | model=%s: %s", qid, forecaster_llm.model, e)

    if qid is not None:
        if discrete_vote is True:
            vote_label = "DISCRETE"
        elif discrete_vote is False:
            vote_label = "CONTINUOUS"
        else:
            vote_label = "PARSE_FAILED"
        logger.info(
            "Discrete vote for Q %s | model=%s | vote=%s",
            qid,
            forecaster_llm.model,
            vote_label,
        )

    return discrete_vote


def build_guarded_numeric_distribution(
    declared_percentiles: list[Percentile], question: NumericQuestion, *, model_name: str
) -> NumericDistribution:
    """Sanitize -> build the PCHIP CDF -> withhold on a unit mismatch.

    Shared by the numeric and date runners, and by the offline section-strip bench: on a date question
    ``question`` is the epoch-seconds adapter, which is what makes the MEMBER_FORECAST line read
    ``qtype=date`` and the built distribution carry ``is_date``. The line is emitted after the build so
    it can report the CDF's out-of-range mass, and before the guard so a withheld member still leaves
    it. ``model_name`` labels the markers.

    The unit-mismatch guard fails SHUT: it raises rather than returning a distribution, so an
    order-of-magnitude error can never reach publish.
    """
    sanitized_percentiles, zero_point = sanitize_percentiles(declared_percentiles, question, model_name=model_name)
    prediction = build_numeric_distribution(sanitized_percentiles, question, zero_point, model_name=model_name)
    logger.info(format_percentile_roundtrip_marker(prediction, declared_percentiles, question, model=model_name))
    logger.info(
        format_member_forecast_marker(
            question_id=question.id_of_question,
            model=model_name,
            role=MEMBER_FORECAST_ROLE_MEMBER,
            qtype=numeric_qtype(question),
            raw=percentile_pairs(declared_percentiles),
            published=percentile_pairs(sanitized_percentiles),
            out_of_range=out_of_range_mass(prediction),
        )
    )

    mismatch, reason = detect_unit_mismatch(sanitized_percentiles, question)
    if mismatch:
        logger.error(
            f"Unit mismatch likely for Q {getattr(question, 'id_of_question', 'N/A')} | "
            f"URL {getattr(question, 'page_url', '<unknown>')} | reason={reason}. Withholding prediction."
        )
        raise UnitMismatchError(
            f"Unit mismatch likely; {reason}. Values: {[float(p.value) for p in sanitized_percentiles]}"
        )

    log_final_prediction(prediction, question)
    log_open_bound_piling_diagnostics(prediction, question, model_name, sanitized_percentiles)
    return prediction
