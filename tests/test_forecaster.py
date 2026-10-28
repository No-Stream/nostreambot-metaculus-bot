from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from forecasting_tools import MetaculusQuestion

from main import TemplateForecaster
from metaculus_bot.research import providers as research_providers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("credentials", "expected_provider", "expected_text", "expected_header"),
    [
        pytest.param(
            {
                "ASKNEWS_CLIENT_ID": "asknews-client",
                "ASKNEWS_SECRET": "asknews-secret",
                "EXA_API_KEY": "exa-key",
                "PERPLEXITY_API_KEY": "perplexity-key",
                "OPENROUTER_API_KEY": "openrouter-key",
            },
            "asknews",
            "AskNews briefing",
            "## News Articles (AskNews)",
            id="asknews-before-others",
        ),
        pytest.param(
            {"EXA_API_KEY": "exa-key", "PERPLEXITY_API_KEY": "perplexity-key", "OPENROUTER_API_KEY": "openrouter-key"},
            "none",
            "",
            "",
            id="other-credentials-do-not-select-primary",
        ),
        pytest.param(
            {"ASKNEWS_CLIENT_ID": "asknews-client", "EXA_API_KEY": "exa-key"},
            "none",
            "",
            "",
            id="missing-asknews-secret",
        ),
        pytest.param({}, "none", "", "", id="no_provider"),
    ],
)
async def test_run_research_uses_real_provider_priority(
    credentials: dict[str, str],
    expected_provider: str,
    expected_text: str,
    expected_header: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the production chooser while replacing each selected provider's network call."""
    for name in (
        "ASKNEWS_CLIENT_ID",
        "ASKNEWS_SECRET",
        "EXA_API_KEY",
        "PERPLEXITY_API_KEY",
        "OPENROUTER_API_KEY",
        "RESEARCH_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "NATIVE_SEARCH_ENABLED",
        "GEMINI_SEARCH_ENABLED",
        "FINANCIAL_DATA_ENABLED",
        "TS_ANCHOR_ENABLED",
        "PREDICTION_MARKETS_ENABLED",
        "RESOLUTION_SOURCE_ENABLED",
        "GAP_FILL_ENABLED",
        "GAP_FILL_V2_ENABLED",
    ):
        monkeypatch.setenv(name, "false")
    for name, value in credentials.items():
        monkeypatch.setenv(name, value)

    forecaster = TemplateForecaster(
        llms={
            "default": "mock_default_model",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
            "summarizer": "mock_summarizer",
        }
    )
    question = MetaculusQuestion(
        question_text="Test question",
        page_url="http://example.com",
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
        id_of_question=777,
    )

    asknews_call = AsyncMock(return_value="AskNews articles")
    asknews_factory = Mock(return_value=asknews_call)
    monkeypatch.setattr(research_providers, "_asknews_provider", asknews_factory)
    monkeypatch.setattr(
        forecaster._research,
        "_summarize_asknews",
        AsyncMock(return_value="AskNews briefing"),
    )
    research = await forecaster.run_research(question)

    if expected_provider == "none":
        assert research == ""
        diagnostics = forecaster._research.pop_provider_diagnostics(question.id_of_question)
        assert "- none: empty | 0 chars |" in diagnostics
    else:
        asknews_call.assert_awaited_once_with(question)
        assert expected_text in research
        assert expected_header in research

    if expected_provider == "asknews":
        asknews_factory.assert_called_once_with()
    else:
        asknews_call.assert_not_awaited()
