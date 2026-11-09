"""Structured parser with strict json_schema and framework fallback.

Primary path: invoke a constrained LLM with response_format set to a strict
json_schema wrapper, plus provider.require_parameters=true to prevent silent
schema drops on OpenRouter.

Fallback: on any failure (validation, refusal, truncation, provider error),
fall back to forecasting_tools.structure_output (today's exact behavior).
"""

from __future__ import annotations

import logging
from typing import get_args, get_origin

from forecasting_tools import GeneralLlm, structure_output
from forecasting_tools.data_models.numeric_report import DatePercentile, Percentile
from pydantic import BaseModel, field_validator

from metaculus_bot.constants import OPENROUTER_OUTPUT_TOKEN_LIMIT
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.numeric.date_axis import parse_forecast_date
from metaculus_bot.simple_types import OptionProbability

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wrapper models for list types (response_format requires a single BaseModel)
# ---------------------------------------------------------------------------


class IsoDatePercentile(DatePercentile):
    """forecasting-tools' ``DatePercentile`` whose ``value`` is read by the repo's one date parser.

    The framework's own date template parses a ``value`` with pydantic's datetime coercion, which
    leaves a date-only string at midnight naive and a bare integer as a unix timestamp, and then
    calls ``.timestamp()`` on the naive result, which is host-local time (an 8-hour error on a
    Pacific laptop; correct on a UTC runner by accident). Routing the raw string through
    ``numeric.date_axis.parse_forecast_date`` instead gives the LLM salvage rung the same semantics as
    the block rung: strict ISO-8601, a naive time read as UTC, a date-only value at noon UTC so it
    lands inside its day bin, and a loud failure on anything else. The parser LLM's constrained
    schema still asks for a date-time string, since that is what ``DatePercentile`` declares.
    """

    @field_validator("value", mode="before")
    @classmethod
    def _parse_forecast_date(cls, value: object) -> object:
        if isinstance(value, str):
            return parse_forecast_date(value)
        raise ValueError(f"DatePercentile.value must be an ISO-8601 date string, got {value!r}")


class PercentileListWrapper(BaseModel):
    """Wrapper for list[Percentile] to satisfy json_schema response_format."""

    percentiles: list[Percentile]


class DatePercentileListWrapper(BaseModel):
    """Wrapper for list[IsoDatePercentile] to satisfy json_schema response_format."""

    percentiles: list[IsoDatePercentile]


class OptionProbabilityListWrapper(BaseModel):
    """Wrapper for list[OptionProbability] to satisfy json_schema response_format."""

    options: list[OptionProbability]


class BinProbability(BaseModel):
    """One bin of a per-bin declaration as the salvage rung reads it: the label the rationale used
    beside its probability. ``value_extraction.extract_pmf`` folds the label onto the grid's keys,
    so this carries the text as written rather than a matched bin."""

    label: str
    probability: float


class BinProbabilityListWrapper(BaseModel):
    """Wrapper for list[BinProbability] to satisfy json_schema response_format."""

    bins: list[BinProbability]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Why: one table ties an item type to its wrapper AND list field; see docs/value_extraction.md "The LLM salvage rung".
_LIST_WRAPPERS: dict[type, tuple[type[BaseModel], str]] = {
    Percentile: (PercentileListWrapper, "percentiles"),
    IsoDatePercentile: (DatePercentileListWrapper, "percentiles"),
    OptionProbability: (OptionProbabilityListWrapper, "options"),
    BinProbability: (BinProbabilityListWrapper, "bins"),
}


def _get_wrapper_type(output_type: type) -> tuple[type[BaseModel], str] | None:
    """The ``(wrapper model, list field)`` pair for a ``list[X]`` output type, else None."""
    if get_origin(output_type) is not list:
        return None
    (item_type,) = get_args(output_type)
    return _LIST_WRAPPERS.get(item_type)


def _build_constrained_llm(response_format_model: type[BaseModel], parser_model: str) -> GeneralLlm:
    """Build a parser LLM with strict json_schema response_format.

    Uses the same donated-key fallback chain as the production PARSER_LLM.
    The extra_body provider.require_parameters=true ensures OpenRouter rejects
    the request rather than silently dropping the schema.

    ``allowed_tries=1`` + ``timeout=90`` bounds the constrained primary so the
    ``structure_output`` fallback always has budget within the 600s forecaster
    soft deadline; without that cap a stuck primary can consume the entire
    deadline and cancel the coroutine before the fallback runs (F1).
    """
    return build_llm_with_openrouter_fallback(
        parser_model,
        # Why: one CREDIT_ROLE_SPEND tier for both parse paths, the same one PARSER_LLM bills to.
        role="parser",
        # Why: redundant since ft 0.2.92 defaults it to None, kept as a pin; see docs/value_extraction.md "The LLM salvage rung: design notes".
        temperature=None,
        max_tokens=OPENROUTER_OUTPUT_TOKEN_LIMIT,
        stream=False,
        timeout=90,
        allowed_tries=1,
        reasoning={"effort": "low"},
        response_format=response_format_model,
        extra_body={"provider": {"require_parameters": True}},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def parse_structured[T](
    text: str,
    output_type: type[T],
    parser_llm: GeneralLlm,
    *,
    prompt_notes: str = "",
) -> T:
    """Parse text into a pydantic model using strict json_schema, falling back to structure_output.

    Parameters
    ----------
    text:
        The forecaster reasoning to extract structured data from.
    output_type:
        The target pydantic BaseModel (or list[BaseModel] generic).
    parser_llm:
        The parser LLM instance (used only on the fallback path via structure_output).
    prompt_notes:
        Additional extraction instructions (e.g. build_parse_notes for numeric).
    """
    # Why: response_format needs a single BaseModel, so a list[X] output type is parsed through its wrapper.
    wrapper = _get_wrapper_type(output_type)
    schema_model: type[BaseModel] = output_type if wrapper is None else wrapper[0]  # type: ignore[assignment]

    # --- Primary path: constrained json_schema ---
    try:
        constrained_llm = _build_constrained_llm(schema_model, parser_llm.model)

        # Why: constrained decoding enforces the schema, so the prompt carries only the text and the notes.
        prompt_parts = [
            "Extract the structured data from the text below.",
        ]
        if prompt_notes:
            prompt_parts.append(f"\nInstructions: {prompt_notes}")
        prompt_parts.append(f"\n\nText:\n{text}")
        prompt = "\n".join(prompt_parts)

        raw_response = await constrained_llm.invoke(prompt)
        parsed = schema_model.model_validate_json(raw_response)
        if wrapper is None:
            return parsed  # type: ignore[return-value]
        _, list_field = wrapper
        return getattr(parsed, list_field)

    # Why: constrained decoding is an optimization, so ANY failure degrades to the fallback below.
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # intentional: catch-all → graceful fallback
        logger.info(
            "Constrained parse failed (%s: %s); falling back to structure_output",
            type(exc).__name__,
            str(exc)[:200],
        )

    # --- Fallback path: today's exact behavior ---
    return await structure_output(
        text_to_structure=text,
        output_type=output_type,
        model=parser_llm,
        additional_instructions=prompt_notes or None,
    )
