"""Pins for the November forecaster configuration."""

from metaculus_bot.credit_telemetry import ROLE_METADATA_KEY
from metaculus_bot.llm_configs import FORECASTER_LLMS


def test_four_forecasters_have_high_effort_and_distinct_spend_roles() -> None:
    assert len(FORECASTER_LLMS) == 4
    assert all(llm.litellm_kwargs["reasoning"] == {"effort": "high"} for llm in FORECASTER_LLMS)
    roles = [llm.litellm_kwargs["metadata"][ROLE_METADATA_KEY] for llm in FORECASTER_LLMS]
    assert len(set(roles)) == 4
