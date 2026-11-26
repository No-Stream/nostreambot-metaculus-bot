"""Cached model requests and key routing for the cross-pollination bench."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import litellm

from metaculus_bot.config import load_environment
from metaculus_bot.constants import OAI_ANTH_OPENROUTER_KEY_ENV
from metaculus_bot.credit_telemetry import DONATED_KEY_ALIAS, PERSONAL_KEY_ALIAS, llm_call_metadata
from scripts.probes.bench_common.paid_call import (
    CALL_RETRIES,
    CALL_TIMEOUT_S,
    MAX_COMPLETION_TOKENS,
    ModelReply,
    personal_key_only_environment,
    reply_from_response,
)
from scripts.probes.cross_pollination_bench.models import REASONING, BenchModel

FORECASTER_ROLE = "cross_pollination_forecaster"
CACHE_BREAKPOINT: dict[str, str] = {"type": "ephemeral"}


def _cached_prompt_block(prompt: str) -> dict[str, Any]:
    # In the 2026-11-25 smoke run, Haiku's one-block first call cached one token fewer than the same block inside
    # a two-block update (trailing newline trimmed on a request's last block), so every first update missed.
    return {"type": "text", "text": prompt.rstrip(), "cache_control": CACHE_BREAKPOINT}


def first_messages(prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [_cached_prompt_block(prompt)]}]


def update_messages(prompt: str, update_block: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                _cached_prompt_block(prompt),
                {"type": "text", "text": update_block},
            ],
        }
    ]


def message_sha256(messages: list[dict[str, Any]]) -> str:
    block_texts = [block["text"] for message in messages for block in message["content"]]
    return hashlib.sha256("\0".join(block_texts).encode("utf-8")).hexdigest()


def session_id(question_id: int, model_label: str) -> str:
    return f"xpoll-{question_id}-{model_label}"


@dataclass(frozen=True)
class BenchKeys:
    shared: str
    personal: str


def split_key_environment() -> BenchKeys:
    load_environment()
    shared_key = os.getenv(OAI_ANTH_OPENROUTER_KEY_ENV)
    if not shared_key:
        raise ValueError(f"{OAI_ANTH_OPENROUTER_KEY_ENV} must be set for Haiku bench calls")
    personal_key = personal_key_only_environment()
    return BenchKeys(shared=shared_key, personal=personal_key)


@dataclass(frozen=True)
class BenchReply:
    reply: ModelReply
    usage: dict[str, Any]
    cache_write_tokens: int
    upstream_provider: str | None
    response_id: str | None
    started_utc: str
    finished_utc: str


BenchModelCall = Callable[[list[dict[str, Any]], str], Awaitable[BenchReply]]


def key_alias(model: BenchModel) -> str:
    return DONATED_KEY_ALIAS if model.key_route == "shared" else PERSONAL_KEY_ALIAS


def build_bench_model_call(model: BenchModel, keys: BenchKeys, *, api_base: str | None = None) -> BenchModelCall:
    api_key = keys.shared if model.key_route == "shared" else keys.personal
    kwargs: dict[str, Any] = {
        "model": f"openrouter/{model.slug}",
        "api_key": api_key,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "timeout": CALL_TIMEOUT_S,
        "num_retries": CALL_RETRIES,
        "reasoning": REASONING,
        "extra_body": {"provider": {"only": [model.provider_only]}},
        "metadata": llm_call_metadata(FORECASTER_ROLE, key_alias(model)),
    }
    if api_base is not None:
        kwargs["api_base"] = api_base

    async def call(messages: list[dict[str, Any]], session: str) -> BenchReply:
        started_utc = datetime.now(UTC).isoformat()
        response = cast(
            Any,
            await litellm.acompletion(
                messages=messages,
                extra_body={**kwargs["extra_body"], "session_id": session},
                **{key: value for key, value in kwargs.items() if key != "extra_body"},
            ),
        )
        finished_utc = datetime.now(UTC).isoformat()
        usage = response.usage.model_dump()
        prompt_details = usage.get("prompt_tokens_details") or {}
        return BenchReply(
            reply=reply_from_response(response),
            usage=usage,
            cache_write_tokens=prompt_details.get("cache_write_tokens") or 0,
            upstream_provider=getattr(response, "provider", None),
            response_id=getattr(response, "id", None),
            started_utc=started_utc,
            finished_utc=finished_utc,
        )

    return call
