"""Serialized request and key-routing contracts for the cross-pollination bench."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from metaculus_bot.api_key_utils import should_route_via_donated_key
from metaculus_bot.constants import (
    DONATED_OPENROUTER_KEY_ENABLED_ENV,
    OAI_ANTH_OPENROUTER_KEY_ENV,
    OPENROUTER_API_KEY_ENV,
)
from scripts.probes.bench_common import paid_call
from scripts.probes.cross_pollination_bench import request
from scripts.probes.cross_pollination_bench.models import BENCH_MODELS, BenchModel

PROMPT = "Forecast prompt with exact bytes.\n"
CACHED_PREFIX = "Forecast prompt with exact bytes."
UPDATE = "First answer and peer context."


class OpenRouterStub:
    def __init__(self) -> None:
        self.requests: list[tuple[dict[str, Any], str]] = []
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((body, self.headers.get("Authorization", "")))
                response = {
                    "id": "stub-response-id",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "provider": "DeepSeek",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "canned reply"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 7},
                        "completion_tokens_details": {"reasoning_tokens": 5},
                        "cost": 0.0123,
                    },
                }
                encoded = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                del format, args

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


@pytest.fixture
def openrouter_stub() -> Generator[OpenRouterStub, None, None]:
    stub = OpenRouterStub()
    yield stub
    stub.close()


async def call_stub(
    openrouter_stub: OpenRouterStub,
    model: BenchModel,
    messages: list[dict[str, Any]],
    *,
    session: str = "xpoll-101-haiku",
) -> Any:
    call = request.build_bench_model_call(
        model, request.BenchKeys("sk-shared", "sk-personal"), api_base=openrouter_stub.url
    )
    return await call(messages, session)


@pytest.mark.asyncio
async def test_first_call_body_is_one_block_with_the_breakpoint(openrouter_stub: OpenRouterStub) -> None:
    await call_stub(openrouter_stub, BENCH_MODELS[0], request.first_messages(PROMPT))
    body, _ = openrouter_stub.requests[0]
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": CACHED_PREFIX, "cache_control": {"type": "ephemeral"}}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [BENCH_MODELS[0], BENCH_MODELS[2]], ids=["haiku", "glm"])
async def test_update_body_keeps_the_breakpoint_on_the_first_block_only(
    openrouter_stub: OpenRouterStub, model: BenchModel
) -> None:
    await call_stub(openrouter_stub, model, request.update_messages(PROMPT, UPDATE))
    message = openrouter_stub.requests[0][0]["messages"][0]
    assert message["content"][0]["text"] == CACHED_PREFIX
    assert message["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert message["content"][1]["text"] == UPDATE
    assert "cache_control" not in message["content"][1]
    assert "cache_control" not in message


@pytest.mark.asyncio
async def test_the_deepseek_body_carries_no_marker(openrouter_stub: OpenRouterStub) -> None:
    await call_stub(openrouter_stub, BENCH_MODELS[1], request.first_messages(PROMPT))
    content = openrouter_stub.requests[0][0]["messages"][0]["content"]
    assert content == [{"type": "text", "text": CACHED_PREFIX}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", BENCH_MODELS, ids=lambda model: model.label)
async def test_body_pins_provider_session_and_reasoning(openrouter_stub: OpenRouterStub, model: BenchModel) -> None:
    await call_stub(openrouter_stub, model, request.first_messages(PROMPT), session=f"xpoll-101-{model.label}")
    body = openrouter_stub.requests[0][0]
    assert body["provider"] == {"only": [model.provider_only]}
    assert body["session_id"] == f"xpoll-101-{model.label}"
    assert body["reasoning"] == {"effort": "medium"}
    assert "reasoning_effort" not in body
    assert body["model"] == model.slug
    assert body["max_tokens"] == 64_000
    assert "tools" not in body
    assert "temperature" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_key"),
    [(BENCH_MODELS[0], "sk-shared"), (BENCH_MODELS[1], "sk-personal"), (BENCH_MODELS[2], "sk-personal")],
    ids=["haiku", "deepseek", "glm"],
)
async def test_only_haiku_uses_the_shared_key(
    openrouter_stub: OpenRouterStub, model: BenchModel, expected_key: str
) -> None:
    await call_stub(openrouter_stub, model, request.first_messages(PROMPT))
    assert openrouter_stub.requests[0][1] == f"Bearer {expected_key}"


def test_split_key_environment_scrubs_the_shared_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(request, "load_environment", lambda: None)
    monkeypatch.setattr(paid_call, "load_environment", lambda: None)
    monkeypatch.setenv(OAI_ANTH_OPENROUTER_KEY_ENV, "sk-shared")
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "sk-personal")
    monkeypatch.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, "true")
    keys = request.split_key_environment()
    assert keys == request.BenchKeys(shared="sk-shared", personal="sk-personal")
    assert OAI_ANTH_OPENROUTER_KEY_ENV not in os.environ
    assert should_route_via_donated_key("openrouter/anthropic/claude-haiku-5.5") is False
    assert paid_call.build_parser_llm(BENCH_MODELS[0].slug).litellm_kwargs.get("api_key") != "sk-shared"


@pytest.mark.parametrize(
    "missing_key", [OAI_ANTH_OPENROUTER_KEY_ENV, OPENROUTER_API_KEY_ENV], ids=["shared", "personal"]
)
def test_split_key_environment_requires_both_keys(monkeypatch: pytest.MonkeyPatch, missing_key: str) -> None:
    monkeypatch.setattr(request, "load_environment", lambda: None)
    monkeypatch.setattr(paid_call, "load_environment", lambda: None)
    monkeypatch.setenv(OAI_ANTH_OPENROUTER_KEY_ENV, "sk-shared")
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "sk-personal")
    monkeypatch.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, "true")
    monkeypatch.delenv(missing_key)
    with pytest.raises(ValueError, match=missing_key):
        request.split_key_environment()


@pytest.mark.asyncio
async def test_the_reply_keeps_usage_provider_and_cache_write(openrouter_stub: OpenRouterStub) -> None:
    reply = await call_stub(openrouter_stub, BENCH_MODELS[1], request.first_messages(PROMPT))
    assert reply.reply.tokens.cached == 80
    assert reply.cache_write_tokens == 7
    assert reply.upstream_provider == "DeepSeek"
    assert reply.usage["cost"] == 0.0123
    assert reply.reply.charged_usd == pytest.approx(0.0123)


def test_the_cached_block_is_identical_whether_or_not_it_is_the_last_block() -> None:
    first_block = request.first_messages(PROMPT)[0]["content"][0]
    update_first_block = request.update_messages(PROMPT, UPDATE)[0]["content"][0]
    assert first_block == update_first_block
    assert first_block["text"] == first_block["text"].rstrip()
