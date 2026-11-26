"""Append-only call rows used to resume and rescore the bench offline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.probes.bench_common.records import (
    STATUS_API_ERROR,
    STATUS_BUILD_FAILED,
    STATUS_EXTRACTION_FAILED,
    STATUS_SCORED,
    STATUS_SKIPPED_SPEND_CAP,
    STATUS_UNIT_MISMATCH,
)

STAGE_FIRST = "first"
VARIANT_PEERS_FULL = "peers_full"
VARIANT_PEERS_FORECASTS = "peers_forecasts"
VARIANT_SECOND_LOOK = "second_look"
UPDATE_VARIANTS = (VARIANT_PEERS_FULL, VARIANT_PEERS_FORECASTS, VARIANT_SECOND_LOOK)
STATUS_SKIPPED_NO_PEERS = "skipped_no_peers"
STATUS_REPLY_RECEIVED = "reply_received"
REPLY_STATUSES = frozenset(
    {STATUS_SCORED, STATUS_EXTRACTION_FAILED, STATUS_BUILD_FAILED, STATUS_UNIT_MISMATCH, STATUS_REPLY_RECEIVED}
)

CallKey = tuple[int, str, int, str]


@dataclass
class BenchCallRow:
    question_id: int
    qtype: str
    model: str
    draw: int
    stage: str
    status: str
    prompt_sha256: str
    score: float | None = None
    forecast: float | list[float] | None = None
    stated_percentiles: list[list[float]] | None = None
    rung: str | None = None
    block_present: bool | None = None
    reply_text: str | None = None
    correction_kind: str | None = None
    peer_order: list[str] | None = None
    usage: dict[str, Any] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    upstream_provider: str | None = None
    response_id: str | None = None
    key_alias: str = "personal"
    started_utc: str | None = None
    finished_utc: str | None = None
    elapsed_s: float = 0.0
    error: str | None = None


def call_key(row: BenchCallRow) -> CallKey:
    return row.question_id, row.model, row.draw, row.stage


def append_row(path: Path, row: BenchCallRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def load_rows(path: Path) -> list[BenchCallRow]:
    with path.open(encoding="utf-8") as handle:
        rows = [BenchCallRow(**json.loads(line)) for line in handle if line.strip()]
    return list({call_key(row): row for row in rows}.values())


def saved_replies(rows: list[BenchCallRow]) -> dict[CallKey, BenchCallRow]:
    latest = {call_key(row): row for row in rows}
    frozen_draws = {(row.question_id, row.draw) for row in latest.values() if row.stage != STAGE_FIRST}
    return {
        key: row
        for key, row in latest.items()
        if row.status in REPLY_STATUSES
        or row.stage != STAGE_FIRST
        or (
            row.stage == STAGE_FIRST
            and row.status in {STATUS_API_ERROR, STATUS_SKIPPED_SPEND_CAP}
            and (row.question_id, row.draw) in frozen_draws
        )
    }
