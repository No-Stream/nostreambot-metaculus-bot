"""Offline checks for Mantic's direct Anthropic route."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from litellm.exceptions import APIError
from litellm.llms.anthropic.chat.transformation import AnthropicConfig

from metaculus_bot.fallback_openrouter import DirectAnthropicFallbackLlm
from metaculus_bot.llm_configs import FORECASTER_LLMS, mantic_forecasters


def test_direct_request_carries_the_slot_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAL_ANT_KEY", "test-key")
    direct = [llm for llm in mantic_forecasters() if isinstance(llm, DirectAnthropicFallbackLlm)]
    assert len(direct) == 2
    config = AnthropicConfig()
    for llm in direct:
        slot_effort = llm._openrouter_llm.litellm_kwargs["reasoning"]["effort"]
        assert llm._direct_llm.litellm_kwargs["reasoning_effort"] == slot_effort
        model = llm._direct_llm.model.removeprefix("anthropic/")
        params = config.map_openai_params({"reasoning_effort": slot_effort}, {}, model, True)
        body = config.transform_request(model, [{"role": "user", "content": "test"}], params, {}, {})
        assert body["output_config"] == {"effort": slot_effort}


def test_missing_key_keeps_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERSONAL_ANT_KEY", raising=False)
    assert mantic_forecasters() == FORECASTER_LLMS


@pytest.mark.parametrize("status", [400, 401, 402, 403, 429, 500, 529])
async def test_direct_failure_falls_back(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    monkeypatch.setenv("PERSONAL_ANT_KEY", "test-key")
    llm = next(llm for llm in mantic_forecasters() if isinstance(llm, DirectAnthropicFallbackLlm))
    message = "credit balance is too low" if status == 400 else "failure"
    llm._direct_llm.invoke = AsyncMock(
        side_effect=APIError(status_code=status, message=message, llm_provider="anthropic", model="test")
    )
    openrouter_invoke = AsyncMock(return_value="ok")
    monkeypatch.setattr(llm._openrouter_llm, "invoke", openrouter_invoke)
    assert await llm.invoke("prompt") == "ok"
    openrouter_invoke.assert_awaited_once_with("prompt", None)


@pytest.mark.parametrize("failure", [TimeoutError(), httpx.ReadTimeout("timeout"), asyncio.CancelledError()])
async def test_timeout_does_not_fall_back(monkeypatch: pytest.MonkeyPatch, failure: BaseException) -> None:
    monkeypatch.setenv("PERSONAL_ANT_KEY", "test-key")
    llm = next(llm for llm in mantic_forecasters() if isinstance(llm, DirectAnthropicFallbackLlm))
    llm._direct_llm.invoke = AsyncMock(side_effect=failure)
    openrouter_invoke = AsyncMock()
    monkeypatch.setattr(llm._openrouter_llm, "invoke", openrouter_invoke)
    with pytest.raises(type(failure)):
        await llm.invoke("prompt")
    openrouter_invoke.assert_not_awaited()


async def test_slow_server_failure_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAL_ANT_KEY", "test-key")
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


@pytest.mark.parametrize("failure", [ValueError("invalid request"), ConnectionError("offline")])
async def test_unclassified_direct_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: Exception
) -> None:
    monkeypatch.setenv("PERSONAL_ANT_KEY", "test-key")
    llm = next(llm for llm in mantic_forecasters() if isinstance(llm, DirectAnthropicFallbackLlm))
    llm._direct_llm.invoke = AsyncMock(side_effect=failure)
    monkeypatch.setattr(llm._openrouter_llm, "invoke", AsyncMock(return_value="ok"))
    assert await llm.invoke("prompt") == "ok"
    assert "reason=other" in caplog.text
