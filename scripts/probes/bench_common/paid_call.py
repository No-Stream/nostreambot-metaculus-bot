from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from forecasting_tools import GeneralLlm

from metaculus_bot.config import load_environment
from metaculus_bot.constants import (
    DONATED_OPENROUTER_KEY_ENABLED_ENV,
    OAI_ANTH_OPENROUTER_KEY_ENV,
    OPENROUTER_API_KEY_ENV,
)
from metaculus_bot.credit_telemetry import (
    TokenCounts,
    _CallUsage,
    _openrouter_call_usage,
    role_spend_rows,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback

MAX_COMPLETION_TOKENS = 64_000
CALL_TIMEOUT_S = 480.0
CALL_RETRIES = 2
PARSER_ROLE = "parser"
ERROR_TEXT_LIMIT = 500
ACCOUNT_REFUSAL_STATUSES = frozenset({401, 402, 403})


@dataclass(frozen=True)
class ModelReply:
    """One completion: its text, token counts, and reported charge."""

    text: str
    tokens: TokenCounts
    charged_usd: float | None


def charged_usd(usage: _CallUsage) -> float | None:
    """The money one call cost, following the production credit ledger's BYOK accounting."""
    if usage.cost_usd is None and usage.byok_upstream_usd is None:
        return None
    upstream = (usage.byok_upstream_usd or 0.0) if usage.is_byok else 0.0
    return (usage.cost_usd or 0.0) + upstream


def reply_from_response(response: Any) -> ModelReply:
    usage = _openrouter_call_usage(response)
    return ModelReply(
        text=response.choices[0].message.content or "", tokens=usage.tokens, charged_usd=charged_usd(usage)
    )


def personal_key_only_environment() -> str:
    """Load .env and remove the shared donated key before building any client."""
    load_environment()
    os.environ.pop(OAI_ANTH_OPENROUTER_KEY_ENV, None)
    os.environ[DONATED_OPENROUTER_KEY_ENABLED_ENV] = "false"
    api_key = os.getenv(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{OPENROUTER_API_KEY_ENV} must be set: this bench spends only the personal key")
    return api_key


def build_parser_llm(parser_model: str) -> GeneralLlm:
    """Build the ladder's salvage parser on the personal key at low effort."""
    return build_llm_with_openrouter_fallback(
        f"openrouter/{parser_model}",
        role=PARSER_ROLE,
        temperature=None,
        max_tokens=32_000,
        stream=False,
        timeout=300,
        allowed_tries=1,
        reasoning={"effort": "low"},
    )


def parser_ledger_usd() -> float:
    """The charges recorded for salvage-parser calls."""
    return sum(row.charged_usd or 0.0 for row in role_spend_rows() if row.role == PARSER_ROLE)


@dataclass
class SpendMeter:
    """Measured spend against the cap, including salvage-parser charges."""

    cap_usd: float
    forecaster_usd: float = 0.0
    unpriced_calls: int = 0
    refusal: str | None = None

    def add(self, reply_charged_usd: float | None) -> None:
        if reply_charged_usd is None:
            self.unpriced_calls += 1
        else:
            self.forecaster_usd += reply_charged_usd

    def refuse(self, error: str) -> None:
        self.refusal = error

    @property
    def measured_usd(self) -> float:
        return self.forecaster_usd + parser_ledger_usd()

    @property
    def stop_reason(self) -> str | None:
        if self.refusal is not None:
            return f"the provider refused the key or the model: {self.refusal}"
        if self.unpriced_calls:
            return f"{self.unpriced_calls} reply(ies) carried no charge, so spend cannot be measured against the cap"
        if self.measured_usd >= self.cap_usd:
            return f"measured spend ${self.measured_usd:.4f} reached the cap ${self.cap_usd:.2f}"
        return None
