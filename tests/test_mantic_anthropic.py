"""Offline checks for Mantic's direct Anthropic route."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from litellm.exceptions import APIError
from litellm.llms.anthropic.chat.transformation import AnthropicConfig

from metaculus_bot.fallback_openrouter import DirectAnthropicFallbackLlm
from metaculus_bot.llm_configs import FORECASTER_LLMS, mantic_forecasters


def test_direct_request_contains_high_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("personal_ant_key", "test-key")
    direct = [llm for llm in mantic_forecasters() if isinstance(llm, DirectAnthropicFallbackLlm)]
    assert len(direct) == 2
    config = AnthropicConfig()
    for llm in direct:
        model = llm._direct_llm.model.removeprefix("anthropic/")
        params = config.map_openai_params({"reasoning_effort": "high"}, {}, model, True)
        body = config.transform_request(model, [{"role": "user", "content": "test"}], params, {}, {})
        assert body["output_config"] == {"effort": "high"}


def test_missing_key_keeps_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("personal_ant_key", raising=False)
    assert mantic_forecasters() == FORECASTER_LLMS


@pytest.mark.parametrize("status", [400, 401, 402, 403, 429, 500, 529])
async def test_direct_failure_falls_back(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    monkeypatch.setenv("personal_ant_key", "test-key")
    llm = next(llm for llm in mantic_forecasters() if isinstance(llm, DirectAnthropicFallbackLlm))
    message = "credit balance is too low" if status == 400 else "failure"
    llm._direct_llm.invoke = AsyncMock(
        side_effect=APIError(status_code=status, message=message, llm_provider="anthropic", model="test")
    )
    openrouter_invoke = AsyncMock(return_value="ok")
    monkeypatch.setattr(llm._openrouter_llm, "invoke", openrouter_invoke)
    assert await llm.invoke("prompt") == "ok"
    openrouter_invoke.assert_awaited_once_with("prompt", None)


async def test_timeout_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("personal_ant_key", "test-key")
    llm = next(llm for llm in mantic_forecasters() if isinstance(llm, DirectAnthropicFallbackLlm))
    llm._direct_llm.invoke = AsyncMock(side_effect=TimeoutError())
    openrouter_invoke = AsyncMock()
    monkeypatch.setattr(llm._openrouter_llm, "invoke", openrouter_invoke)
    with pytest.raises(TimeoutError):
        await llm.invoke("prompt")
    openrouter_invoke.assert_not_awaited()


async def test_slow_server_failure_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("personal_ant_key", "test-key")
    llm = next(llm for llm in mantic_forecasters() if isinstance(llm, DirectAnthropicFallbackLlm))
    times = iter([0.0, 31.0])
    monkeypatch.setattr("metaculus_bot.fallback_openrouter.time", SimpleNamespace(monotonic=lambda: next(times)))
    llm._direct_llm.invoke = AsyncMock(
        side_effect=APIError(status_code=529, message="overloaded", llm_provider="anthropic", model="test")
    )
    openrouter_invoke = AsyncMock()
    monkeypatch.setattr(llm._openrouter_llm, "invoke", openrouter_invoke)
    with pytest.raises(APIError):
        await llm.invoke("prompt")
    openrouter_invoke.assert_not_awaited()
