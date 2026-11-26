"""Pinned model specifications for the cross-pollination experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from metaculus_bot.constants import (
    CROSS_POLLINATION_BENCH_DEEPSEEK_MODEL,
    CROSS_POLLINATION_BENCH_GLM_MODEL,
    CROSS_POLLINATION_BENCH_HAIKU_MODEL,
)

KeyRoute = Literal["shared", "personal"]


@dataclass(frozen=True)
class BenchModel:
    label: str
    slug: str
    provider_only: str
    upstream_provider: str
    key_route: KeyRoute
    input_usd_per_m: float
    output_usd_per_m: float
    cache_read_usd_per_m: float
    cache_write_usd_per_m: float | None


BENCH_MODELS: tuple[BenchModel, ...] = (
    BenchModel(
        "haiku", CROSS_POLLINATION_BENCH_HAIKU_MODEL, "anthropic", "Anthropic", "shared", 0.10, 0.50, 0.01, 0.125
    ),
    BenchModel(
        "deepseek", CROSS_POLLINATION_BENCH_DEEPSEEK_MODEL, "deepseek", "DeepSeek", "personal", 0.15, 0.60, 0.003, None
    ),
    BenchModel("glm", CROSS_POLLINATION_BENCH_GLM_MODEL, "z-ai", "Z.AI", "personal", 0.15, 0.50, 0.03, None),
)

REASONING: dict[str, str] = {"effort": "medium"}


def model_by_label(label: str) -> BenchModel:
    for model in BENCH_MODELS:
        if model.label == label:
            return model
    raise ValueError(f"unknown cross-pollination model label {label!r}")
