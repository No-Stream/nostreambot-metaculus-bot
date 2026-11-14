"""Offline pins for route-compatible reasoning-plus-answer allowances."""

from metaculus_bot.llm_configs import (
    DISAGREEMENT_ANALYZER_LLM,
    FORECASTER_LLMS,
    PARSER_LLM,
    STACKER_FALLBACK_LLM,
    STACKER_LLM,
    SUMMARIZER_LLM,
)
from metaculus_bot.structured_parse import PercentileListWrapper, _build_constrained_llm


def test_forecaster_budgets_preserve_timeouts_and_effort():
    for llm in FORECASTER_LLMS:
        kwargs = llm.litellm_kwargs
        is_google = "/google/" in llm.model
        assert kwargs["max_tokens"] == (65_536 if is_google else 128_000)
        assert kwargs["timeout"] == 480
        assert kwargs["stream"] is False
        if not is_google:
            assert kwargs["reasoning"] == {"effort": "xhigh"}


def test_other_production_slots_use_full_verified_allowance():
    for llm in (PARSER_LLM, SUMMARIZER_LLM, DISAGREEMENT_ANALYZER_LLM, STACKER_LLM, STACKER_FALLBACK_LLM):
        assert llm.litellm_kwargs["max_tokens"] == 128_000
    constrained = _build_constrained_llm(PercentileListWrapper, PARSER_LLM.model)
    assert constrained.litellm_kwargs["max_tokens"] == 128_000
    assert constrained.litellm_kwargs["timeout"] == 90
