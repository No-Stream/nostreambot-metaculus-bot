"""Small, deterministic inputs for cross-pollination manifest tests."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import openai
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import DiscreteQuestion, MetaculusQuestion

from metaculus_bot.credit_telemetry import TokenCounts
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from scripts.probes.bench_common.paid_call import ModelReply
from scripts.probes.cross_pollination_bench.models import BenchModel, model_by_label
from scripts.probes.cross_pollination_bench.prompts import UPDATE_ROUND_MARKER
from scripts.probes.cross_pollination_bench.request import BenchReply


@dataclass
class FakeRound:
    perf_path: Path
    archive_dir: Path
    fetch: Any
    fetch_calls: list[tuple[int, int]]


def _question(qid: int, post_id: int, qtype: str, title: str) -> MetaculusQuestion:
    shared: dict[str, Any] = {
        "question_text": title,
        "id_of_question": qid,
        "id_of_post": post_id,
        "page_url": f"https://www.metaculus.com/questions/{post_id}/",
        "background_info": "Fetched background must be removed.",
        "resolution_criteria": f"Criteria for {qid}.",
        "fine_print": f"Fine print for {qid}.",
        "open_time": datetime(2026, 1, 1, tzinfo=UTC),
        "scheduled_resolution_time": datetime(2026, 12, 31, tzinfo=UTC),
        "resolution_string": f"API resolution {qid}",
    }
    if qtype == "binary":
        return BinaryQuestion(**shared)
    if qtype == "multiple_choice":
        return MultipleChoiceQuestion(options=["A", "B", "C"], **shared)
    continuous = {
        "lower_bound": 0.0,
        "upper_bound": 100.0,
        "open_lower_bound": False,
        "open_upper_bound": qid == 103,
        "zero_point": None,
        "cdf_size": 201 if qid == 103 else 102,
        "nominal_lower_bound": 0.0,
        "nominal_upper_bound": 100.0,
        "unit_of_measure": "units",
    }
    cls = DiscreteQuestion if qtype == "discrete" else NumericQuestion
    return cls(**continuous, **shared)


def write_fake_round(tmp_path: Path) -> FakeRound:
    specs = [
        (101, 201, "binary", "Same title", "2026-10-01T12:00:00+00:00", False),
        (102, 202, "multiple_choice", "Multiple choice", "2026-10-02T12:00:00+00:00", "A"),
        (103, 203, "numeric", "Numeric", "2026-10-03T12:00:00+00:00", 55.5),
        (104, 204, "discrete", "Discrete", "2026-10-04T12:00:00+00:00", "above_upper_bound"),
        (105, 205, "binary", "Same title", "2026-10-05T12:00:00+00:00", True),
        (106, 206, "binary", "On cutoff", "2026-09-17T23:00:00+00:00", False),
        (108, 208, "binary", "After release", "2026-10-18T00:01:00+00:00", True),
        (107, 207, "binary", "Missing archive", "2026-10-06T12:00:00+00:00", False),
    ]
    perf_rows: list[dict[str, Any]] = []
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    questions: dict[int, MetaculusQuestion] = {}
    for qid, post_id, qtype, title, timestamp, parsed in specs:
        if qid == 107:
            continue
        question = _question(qid, post_id, qtype, title)
        questions[qid] = question
        if qid != 107:
            our_forecast_values = [0.4, 0.3, 0.3] if qtype == "multiple_choice" else [0.01, 0.5, 0.99]
            perf_rows.append(
                {
                    "platform": "metaculus",
                    "post_id": post_id,
                    "question_id": qid,
                    "title": title,
                    "type": qtype,
                    "resolution_raw": str(parsed),
                    "resolution_parsed": parsed,
                    "our_prob_yes": 0.6 if qtype == "binary" else None,
                    "our_forecast_values": our_forecast_values,
                    "metaculus_scores": {"spot_peer_score": 1.25},
                    "options": ["A", "B", "C"] if qtype == "multiple_choice" else None,
                }
            )
        (archive_dir / f"{qid}.json").write_text(
            json.dumps(
                {
                    "platform": "metaculus",
                    "post_id": post_id,
                    "timestamp": timestamp,
                    "research_text": f"Archived research for question {qid}.",
                }
            ),
            encoding="utf-8",
        )
    perf_rows.append(
        {
            "platform": "metaculus",
            "post_id": 207,
            "question_id": 107,
            "title": "Missing archive",
            "type": "binary",
            "resolution_raw": "no",
            "resolution_parsed": False,
            "our_prob_yes": 0.6,
            "our_forecast_values": [0.4, 0.6],
            "metaculus_scores": {"spot_peer_score": 1.25},
            "options": None,
        }
    )
    (archive_dir / "106.json").write_text(
        json.dumps(
            {
                "platform": "metaculus",
                "post_id": 206,
                "timestamp": "2026-09-17T23:00:00+00:00",
                "research_text": "Cutoff research.",
            }
        ),
        encoding="utf-8",
    )
    perf_path = tmp_path / "perf.json"
    perf_path.write_text(json.dumps(perf_rows), encoding="utf-8")
    fetch_calls: list[tuple[int, int]] = []

    def fetch(post_id: int, question_id: int) -> MetaculusQuestion:
        fetch_calls.append((post_id, question_id))
        if question_id == 102:
            sibling = _question(109, post_id, "binary", "Sibling")
            return [questions[102], sibling]  # type: ignore[return-value]
        return questions[question_id]

    return FakeRound(perf_path, archive_dir, fetch, fetch_calls)


def first_reply(qtype: str, tag: str) -> str:
    if qtype == "binary":
        body = {"question_type": "binary", "posterior_prob": 0.4}
    elif qtype == "multiple_choice":
        body = {"question_type": "multiple_choice", "option_probs": {"A": 0.4, "B": 0.3, "C": 0.3}}
    else:
        body = {
            "question_type": "numeric",
            "declared_percentiles": {str(p): 10 + 80 * p for p in STANDARD_PERCENTILES},
            "outcome_type": "continuous",
        }
    return f"{tag}\n```json\n{json.dumps(body)}\n```"


def update_reply(qtype: str, tag: str, kind: str = "none") -> str:
    return f"UPDATE: prior=0.4 revised=0.4 correction={kind}\n{first_reply(qtype, tag)}"


class FakeBenchModel:
    def __init__(
        self,
        label: str,
        *,
        charged_usd: float | None = 0.001,
        price_from: BenchModel | None = None,
        price_multiplier: float = 1.0,
        cached_share: float = 0.9,
        upstream_provider: str | None = None,
        repair_only_block: bool = False,
        api_error_first_draws: frozenset[int] = frozenset(),
        unparseable: frozenset[int] = frozenset(),
        events: list[Any] | None = None,
    ) -> None:
        self.label = label
        self.charged_usd = charged_usd
        self.price_from = price_from
        self.price_multiplier = price_multiplier
        self.cached_share = cached_share
        self.upstream_provider = upstream_provider or model_by_label(label).upstream_provider
        self.repair_only_block = repair_only_block
        self.api_error_first_draws = api_error_first_draws
        self.unparseable = unparseable
        self.events = events if events is not None else []
        self.counter = 0
        self.first_counts: dict[str, int] = {}

    async def __call__(self, messages: list[dict[str, Any]], session: str) -> BenchReply:
        self.counter += 1
        call_number = self.counter
        blocks = messages[0]["content"]
        update = len(blocks) > 1 and UPDATE_ROUND_MARKER in blocks[1]["text"]
        if not update:
            self.first_counts[session] = self.first_counts.get(session, 0) + 1
        draw = self.first_counts.get(session, 0)
        index = len(self.events)
        start = time.monotonic_ns()
        self.events.append((self.label, session, messages, start, -1))
        await asyncio.sleep(0)
        end = time.monotonic_ns()
        self.events[index] = (self.label, session, messages, start, end)
        if not update and draw in self.api_error_first_draws:
            raise openai.OpenAIError("fake failure")
        prompt = blocks[0]["text"]
        qtype = (
            "multiple_choice"
            if "Multiple choice" in prompt
            else "numeric"
            if "Numeric" in prompt
            else "discrete"
            if "Discrete" in prompt
            else "binary"
        )
        tag = f"{self.label}-{call_number}"
        text = update_reply(qtype, tag) if update else first_reply(qtype, tag)
        if not update and draw in self.unparseable:
            text = tag
        if self.repair_only_block:
            text = text.replace("}\n```", ",}\n```")
        tokens = TokenCounts(
            prompt=1000, completion=300, cached=int(1000 * self.cached_share) if update else 0, reasoning=200
        )
        charge = self.charged_usd
        if self.price_from:
            charge = (
                (
                    (tokens.prompt - tokens.cached) * self.price_from.input_usd_per_m
                    + tokens.cached * self.price_from.cache_read_usd_per_m
                    + tokens.completion * self.price_from.output_usd_per_m
                )
                / 1_000_000
                * self.price_multiplier
            )
        return BenchReply(
            ModelReply(text, tokens, charge),
            {},
            0,
            self.upstream_provider,
            tag,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:01+00:00",
        )
