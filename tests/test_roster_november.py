"""Pins for the November forecaster configuration."""

from metaculus_bot.credit_telemetry import ROLE_METADATA_KEY
from metaculus_bot.llm_configs import FORECASTER_LLMS


def test_four_forecasters_have_their_effort_and_distinct_spend_roles() -> None:
    assert len(FORECASTER_LLMS) == 4
    efforts = {llm.model: llm.litellm_kwargs["reasoning"]["effort"] for llm in FORECASTER_LLMS}
    # Argon stays at medium: at high it took 411 s of the 600 s deadline on Q46727 (2026-11-24 probe).
    assert efforts == {
        "openrouter/openai/gpt-6.1-sol": "high",
        "openrouter/anthropic/claude-opus-5.5": "high",
        "openrouter/anthropic/claude-fable-5.2": "medium",
        "openrouter/google/gemini-4-argon": "medium",
    }
    roles = [llm.litellm_kwargs["metadata"][ROLE_METADATA_KEY] for llm in FORECASTER_LLMS]
    assert len(set(roles)) == 4
