"""Configuration and helpers for assembling benchmark forecasting bots."""

# ruff: noqa: ERA001
# The commented-out MODEL_CATALOG / *_MODEL_SPECS entries below are a parked
# roster of benchmark arms: uncommenting an entry is the intended way to add a
# model to a sweep, so they stay in place rather than in git history.

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, cast

from forecasting_tools import GeneralLlm

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.forecaster import TemplateForecaster
from metaculus_bot.llm_configs import PARSER_LLM, RESEARCHER_LLM, SUMMARIZER_LLM

# temperature=None defers reasoning models to provider defaults; redundant on
# ft 0.2.92 (GeneralLlm ctor default is already None). top_p left unset.
MODEL_CONFIG: dict[str, Any] = {
    "max_tokens": 64_000,
    "stream": False,
    "timeout": 480,  # 8 minutes - reasoning models (o3, gpt-5.1) need extra time
    "allowed_tries": 3,
}

BENCHMARK_BOT_CONFIG: dict[str, Any] = {
    "research_reports_per_question": 1,
    "predictions_per_research_report": 1,
    "publish_reports_to_metaculus": False,
    "folder_to_save_reports_to": None,
    "skip_previously_forecasted_questions": False,
    "research_provider": None,
    "max_questions_per_run": None,
    "is_benchmarking": True,
}

DEFAULT_HELPER_LLMS: dict[str, GeneralLlm] = {
    "summarizer": SUMMARIZER_LLM,
    "parser": PARSER_LLM,
    "researcher": RESEARCHER_LLM,
}


MODEL_CATALOG: dict[str, GeneralLlm] = {
    "qwen3-235b": GeneralLlm(
        model="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
        **MODEL_CONFIG,
    ),
    "deepseek-3.2": GeneralLlm(
        model="openrouter/deepseek/deepseek-v3.2",
        **MODEL_CONFIG,
    ),
    # "kimi-k2": GeneralLlm(
    #     model="openrouter/moonshotai/kimi-k2-thinking",
    #     **MODEL_CONFIG,
    # ),
    "gpt-5.1": build_llm_with_openrouter_fallback(
        model="openrouter/openai/gpt-5.1",
        reasoning={"effort": "high"},
        **MODEL_CONFIG,
    ),
    # "o3": build_llm_with_openrouter_fallback(
    #     model="openrouter/openai/o3",
    #     reasoning={"effort": "high"},
    #     **MODEL_CONFIG,
    # ),
    # --- Models below are defined for future testing ---
    # "gpt-5.2": build_llm_with_openrouter_fallback(
    #     model="openrouter/openai/gpt-5.2",
    #     reasoning={"effort": "high"},
    #     **MODEL_CONFIG,
    # ),
    # "gemini-3-pro": GeneralLlm(
    #     model="openrouter/google/gemini-3-pro-preview",
    #     **MODEL_CONFIG,
    # ),
    # "gemini-3-flash": GeneralLlm(
    #     model="openrouter/google/gemini-3-flash-preview",
    #     **MODEL_CONFIG,
    # ),
    # "claude-opus-4.5": build_llm_with_openrouter_fallback(
    #     model="openrouter/anthropic/claude-opus-4.5",
    #     reasoning={"max_tokens": 16_000},
    #     **MODEL_CONFIG,
    # ),
    # "grok-4.1-fast": build_llm_with_openrouter_fallback(
    #     model="openrouter/x-ai/grok-4.1-fast",
    #     reasoning={"effort": "high"},
    #     **MODEL_CONFIG,
    # ),
    # "claude-sonnet-4.5": build_llm_with_openrouter_fallback(
    #     model="openrouter/anthropic/claude-sonnet-4.5",
    #     reasoning={"max_tokens": 16_000},
    #     **MODEL_CONFIG,
    # ),
    # "glm-4.7": GeneralLlm(
    #     model="openrouter/z-ai/glm-4.7",
    #     **MODEL_CONFIG,
    # ),
}

INDIVIDUAL_MODEL_SPECS: tuple[Mapping[str, str | GeneralLlm], ...] = (
    MappingProxyType({"name": "qwen3-235b", "forecaster": MODEL_CATALOG["qwen3-235b"]}),
    MappingProxyType({"name": "deepseek-3.2", "forecaster": MODEL_CATALOG["deepseek-3.2"]}),
    MappingProxyType({"name": "gpt-5.1", "forecaster": MODEL_CATALOG["gpt-5.1"]}),
    # MappingProxyType({"name": "o3", "forecaster": MODEL_CATALOG["o3"]}),
    # --- Models below are for future testing ---
    # MappingProxyType({"name": "kimi-k2", "forecaster": MODEL_CATALOG["kimi-k2"]}),
    # MappingProxyType({"name": "gpt-5.2", "forecaster": MODEL_CATALOG["gpt-5.2"]}),
    # MappingProxyType({"name": "gemini-3-pro", "forecaster": MODEL_CATALOG["gemini-3-pro"]}),
    # MappingProxyType({"name": "gemini-3-flash", "forecaster": MODEL_CATALOG["gemini-3-flash"]}),
    # MappingProxyType({"name": "claude-opus-4.5", "forecaster": MODEL_CATALOG["claude-opus-4.5"]}),
    # MappingProxyType({"name": "grok-4.1-fast", "forecaster": MODEL_CATALOG["grok-4.1-fast"]}),
    # MappingProxyType({"name": "claude-sonnet-4.5", "forecaster": MODEL_CATALOG["claude-sonnet-4.5"]}),
    # MappingProxyType({"name": "glm-4.7", "forecaster": MODEL_CATALOG["glm-4.7"]}),
)

STACKING_MODEL_SPECS: tuple[Mapping[str, GeneralLlm], ...] = (
    # MappingProxyType({"name": "stack-qwen3", "stacker": MODEL_CATALOG["qwen3-235b"]}),
    # MappingProxyType({"name": "stack-o3", "stacker": MODEL_CATALOG["o3"]}),
    # MappingProxyType({"name": "stack-claude4", "stacker": MODEL_CATALOG["claude-sonnet-4"]}),
    # MappingProxyType({"name": "stack-gpt5.1", "stacker": MODEL_CATALOG["gpt-5.1"]}),
)


def create_individual_bots(
    model_specs: Iterable[Mapping[str, str | GeneralLlm]],
    helper_llms: dict[str, GeneralLlm],
    benchmark_config: dict[str, Any],
    *,
    batch_size: int,
    research_cache: dict[int, str],
) -> list[TemplateForecaster]:
    bots: list[TemplateForecaster] = []
    for spec in model_specs:
        bot = TemplateForecaster(
            **benchmark_config,
            aggregation_strategy=AggregationStrategy.MEAN,
            llms=cast(
                "dict[str, str | GeneralLlm]",
                {"forecasters": [spec["forecaster"]], **helper_llms},
            ),
            max_concurrent_research=batch_size,
            research_cache=research_cache,
        )
        bot.name = str(spec["name"])
        bots.append(bot)
    return bots


def create_stacking_bots(
    stacking_specs: Iterable[Mapping[str, str | GeneralLlm]],
    base_forecasters: list[str | GeneralLlm],
    helper_llms: dict[str, GeneralLlm],
    benchmark_config: dict[str, Any],
    *,
    batch_size: int,
    research_cache: dict[int, str],
) -> list[TemplateForecaster]:
    bots: list[TemplateForecaster] = []
    for spec in stacking_specs:
        bot = TemplateForecaster(
            **benchmark_config,
            aggregation_strategy=AggregationStrategy.STACKING,
            llms=cast(
                "dict[str, str | GeneralLlm]",
                {
                    "forecasters": base_forecasters,
                    "stacker": spec["stacker"],
                    **helper_llms,
                },
            ),
            max_concurrent_research=batch_size,
            research_cache=research_cache,
            stacking_fallback_on_failure=False,
            stacking_randomize_order=True,
        )
        bot.name = str(spec["name"])
        bots.append(bot)
    return bots


__all__ = [
    "BENCHMARK_BOT_CONFIG",
    "DEFAULT_HELPER_LLMS",
    "INDIVIDUAL_MODEL_SPECS",
    "MODEL_CATALOG",
    "MODEL_CONFIG",
    "STACKING_MODEL_SPECS",
    "create_individual_bots",
    "create_stacking_bots",
]
