"""Tests for selecting the primary research provider."""

from __future__ import annotations

import pytest
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.research import providers


@pytest.mark.asyncio
async def test_other_provider_credentials_do_not_select_a_primary_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "RESEARCH_PROVIDER"):
        monkeypatch.delenv(variable, raising=False)
    for variable in ("EXA_API_KEY", "PERPLEXITY_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(variable, "configured")

    provider, name = providers.choose_provider_with_name()

    assert name == "none"
    assert await provider(MetaculusQuestion(question_text="Will X happen?", page_url="https://example.com")) == ""


@pytest.mark.parametrize("override", ["exa", "perplexity", "openrouter"])
def test_removed_provider_overrides_fall_through_to_none_without_asknews_credentials(
    monkeypatch: pytest.MonkeyPatch,
    override: str,
) -> None:
    for variable in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "EXA_API_KEY", "PERPLEXITY_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("RESEARCH_PROVIDER", override)

    _, name = providers.choose_provider_with_name()

    assert name == "none"


def test_asknews_override_requires_both_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_PROVIDER", "asknews")
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "client")
    monkeypatch.delenv("ASKNEWS_SECRET", raising=False)

    with pytest.raises(ValueError, match="RESEARCH_PROVIDER=asknews requires"):
        providers.choose_provider_with_name()


@pytest.mark.asyncio
async def test_none_override_disables_asknews_even_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_PROVIDER", "none")
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "client")
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")

    provider, name = providers.choose_provider_with_name()

    assert name == "none"
    assert await provider(MetaculusQuestion(question_text="Will X happen?", page_url="https://example.com")) == ""
