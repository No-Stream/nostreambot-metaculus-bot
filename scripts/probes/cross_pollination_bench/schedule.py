"""Schedule and resume each question's first forecasts and updates."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import openai
from forecasting_tools import GeneralLlm, MultipleChoiceQuestion

from metaculus_bot.exceptions import UnitMismatchError, ValueExtractionError
from scripts.probes.bench_common.paid_call import ACCOUNT_REFUSAL_STATUSES, ERROR_TEXT_LIMIT, SpendMeter
from scripts.probes.bench_common.records import (
    STATUS_API_ERROR,
    STATUS_BUILD_FAILED,
    STATUS_EXTRACTION_FAILED,
    STATUS_SCORED,
    STATUS_SKIPPED_SPEND_CAP,
    STATUS_UNIT_MISMATCH,
)
from scripts.probes.bench_common.scoring import score_reply
from scripts.probes.cross_pollination_bench.manifest import QuestionContext
from scripts.probes.cross_pollination_bench.models import BenchModel
from scripts.probes.cross_pollination_bench.prompts import (
    PeerAnswer,
    Variant,
    parse_correction_kind,
    peer_order,
    render_forecast_values,
    update_block,
)
from scripts.probes.cross_pollination_bench.request import (
    BenchModelCall,
    BenchReply,
    first_messages,
    key_alias,
    message_sha256,
    session_id,
    update_messages,
)
from scripts.probes.cross_pollination_bench.rows import (
    STAGE_FIRST,
    STATUS_REPLY_RECEIVED,
    STATUS_SKIPPED_NO_PEERS,
    UPDATE_VARIANTS,
    VARIANT_PEERS_FULL,
    VARIANT_SECOND_LOOK,
    BenchCallRow,
    CallKey,
)


@dataclass(frozen=True)
class RunConfig:
    draws: int
    variants: tuple[str, ...]
    peer_order_seed: int
    full_answers_first: bool


MAIN_RUN = RunConfig(2, UPDATE_VARIANTS, 0, full_answers_first=False)
SMOKE_RUN = RunConfig(1, UPDATE_VARIANTS, 0, full_answers_first=True)


def planned_calls(n_questions: int, n_models: int, config: RunConfig) -> int:
    return n_questions * n_models * config.draws * (1 + len(config.variants))


class _QuestionScheduler:
    def __init__(
        self,
        context: QuestionContext,
        *,
        models: Sequence[BenchModel],
        calls: Mapping[str, BenchModelCall],
        parsers: Mapping[str, GeneralLlm],
        saved: Mapping[CallKey, BenchCallRow],
        meter: SpendMeter,
        config: RunConfig,
        on_row: Callable[[BenchCallRow], None],
    ) -> None:
        self.context, self.models, self.calls, self.parsers = context, models, calls, parsers
        self.saved, self.meter, self.config, self.on_row = saved, meter, config, on_row
        self.results: list[BenchCallRow] = []
        self.first_rows: dict[tuple[str, int], BenchCallRow] = {}

    def saved_row(self, model: BenchModel, draw: int, stage: str, digest: str) -> BenchCallRow | None:
        question_id = self.context.entry.question_id
        old = self.saved.get((question_id, model.label, draw, stage))
        if old is None:
            return None
        if old.prompt_sha256 != digest:
            raise ValueError(f"question {question_id}: prompt changed for {model.label} draw {draw} {stage}")
        frozen = any(key[0] == question_id and key[2] == draw and key[3] != STAGE_FIRST for key in self.saved)
        if stage == STAGE_FIRST and old.status in (STATUS_API_ERROR, STATUS_SKIPPED_SPEND_CAP) and not frozen:
            return None
        return old

    async def score(self, row: BenchCallRow, model: BenchModel) -> None:
        assert row.reply_text is not None
        try:
            scored = await score_reply(
                self.context.question,
                self.context.resolution,
                row.reply_text,
                self.parsers[model.label],
                model_name=model.slug,
            )
        except ValueExtractionError as exc:
            row.status, row.error = STATUS_EXTRACTION_FAILED, str(exc)[:ERROR_TEXT_LIMIT]
        except UnitMismatchError as exc:
            row.status, row.error = STATUS_UNIT_MISMATCH, str(exc)[:ERROR_TEXT_LIMIT]
        except ValueError as exc:
            row.status, row.error = STATUS_BUILD_FAILED, str(exc)[:ERROR_TEXT_LIMIT]
        else:
            row.status, row.score = STATUS_SCORED, scored.score
            row.forecast = scored.cdf if scored.cdf is not None else scored.forecast
            row.stated_percentiles, row.rung, row.block_present = (
                scored.stated_percentiles,
                scored.rung,
                scored.block_present,
            )

    def record_reply(self, row: BenchCallRow, reply: BenchReply) -> None:
        self.meter.add(reply.reply.charged_usd)
        tokens = reply.reply.tokens
        row.reply_text, row.cost_usd = reply.reply.text, reply.reply.charged_usd
        row.prompt_tokens, row.completion_tokens = tokens.prompt, tokens.completion
        row.reasoning_tokens, row.cached_tokens = tokens.reasoning, tokens.cached
        row.usage, row.cache_write_tokens = reply.usage, reply.cache_write_tokens
        row.upstream_provider, row.response_id = reply.upstream_provider, reply.response_id
        row.started_utc, row.finished_utc = reply.started_utc, reply.finished_utc
        if row.stage != STAGE_FIRST:
            row.correction_kind = parse_correction_kind(reply.reply.text)
        row.status = STATUS_REPLY_RECEIVED
        self.on_row(replace(row))

    async def call(self, row: BenchCallRow, model: BenchModel, messages: list[dict[str, Any]]) -> None:
        started = time.monotonic()
        try:
            reply = await self.calls[model.label](messages, session_id(row.question_id, model.label))
        except openai.OpenAIError as exc:
            row.error = f"{type(exc).__name__}: {exc}"[:ERROR_TEXT_LIMIT]
            if getattr(exc, "status_code", None) in ACCOUNT_REFUSAL_STATUSES:
                self.meter.refuse(row.error)
        else:
            row.elapsed_s = time.monotonic() - started
            self.record_reply(row, reply)
            await self.score(row, model)
        row.elapsed_s = time.monotonic() - started

    async def execute(
        self,
        model: BenchModel,
        draw: int,
        stage: str,
        messages: list[dict[str, Any]],
        *,
        peers: list[str] | None = None,
        skip_no_peers: bool = False,
    ) -> BenchCallRow:
        digest = message_sha256(messages)
        old = self.saved_row(model, draw, stage, digest)
        if old is not None:
            row = replace(old)
            if row.status == STATUS_REPLY_RECEIVED:
                await self.score(row, model)
                self.on_row(replace(row))
            self.results.append(row)
            return row
        row = BenchCallRow(
            self.context.entry.question_id,
            self.context.entry.qtype,
            model.label,
            draw,
            stage,
            STATUS_API_ERROR,
            digest,
            peer_order=peers,
            key_alias=key_alias(model),
        )
        if (reason := self.meter.stop_reason) is not None:
            row.status, row.error = STATUS_SKIPPED_SPEND_CAP, reason
        elif skip_no_peers:
            row.status = STATUS_SKIPPED_NO_PEERS
        else:
            await self.call(row, model, messages)
        self.on_row(replace(row))
        self.results.append(row)
        return row

    async def first_for_model(self, model: BenchModel) -> None:
        for draw in range(1, self.config.draws + 1):
            self.first_rows[model.label, draw] = await self.execute(
                model, draw, STAGE_FIRST, first_messages(self.context.prompt)
            )

    async def update_for(self, model: BenchModel, draw: int, stage: str) -> None:
        own = self.first_rows[model.label, draw]
        peers = [
            other.label
            for other in self.models
            if other.label != model.label and self.first_rows[other.label, draw].status == STATUS_SCORED
        ]
        ordered = (
            peer_order(self.context.entry.question_id, model.label, peers, seed=self.config.peer_order_seed)
            if stage != VARIANT_SECOND_LOOK
            else []
        )
        options = self.context.question.options if isinstance(self.context.question, MultipleChoiceQuestion) else None
        answers = []
        for label in ordered:
            peer = self.first_rows[label, draw]
            assert peer.reply_text is not None
            assert peer.forecast is not None
            forecast = render_forecast_values(
                self.context.entry.qtype, peer.forecast, options=options, stated_percentiles=peer.stated_percentiles
            )
            answers.append(PeerAnswer(peer.reply_text, forecast))
        assert own.reply_text is not None
        no_peers = stage != VARIANT_SECOND_LOOK and not answers
        block = update_block(cast(Variant, VARIANT_SECOND_LOOK if no_peers else stage), own.reply_text, answers)
        await self.execute(
            model, draw, stage, update_messages(self.context.prompt, block), peers=ordered, skip_no_peers=no_peers
        )

    async def updates_for(self, model: BenchModel, draw: int) -> None:
        if self.first_rows[model.label, draw].status != STATUS_SCORED:
            return
        remaining = list(self.config.variants)
        if self.config.full_answers_first and VARIANT_PEERS_FULL in remaining:
            await self.update_for(model, draw, VARIANT_PEERS_FULL)
            remaining.remove(VARIANT_PEERS_FULL)
        await asyncio.gather(*(self.update_for(model, draw, variant) for variant in remaining))

    async def run(self) -> list[BenchCallRow]:
        await asyncio.gather(*(self.first_for_model(model) for model in self.models))
        await asyncio.gather(
            *(self.updates_for(model, draw) for model in self.models for draw in range(1, self.config.draws + 1))
        )
        return self.results


async def run_question(
    context: QuestionContext,
    *,
    models: Sequence[BenchModel],
    calls: Mapping[str, BenchModelCall],
    parsers: Mapping[str, GeneralLlm],
    saved: Mapping[CallKey, BenchCallRow],
    meter: SpendMeter,
    config: RunConfig,
    on_row: Callable[[BenchCallRow], None],
) -> list[BenchCallRow]:
    scheduler = _QuestionScheduler(
        context, models=models, calls=calls, parsers=parsers, saved=saved, meter=meter, config=config, on_row=on_row
    )
    return await scheduler.run()


async def run_questions(
    contexts: Sequence[QuestionContext],
    *,
    models: Sequence[BenchModel],
    calls: Mapping[str, BenchModelCall],
    parsers: Mapping[str, GeneralLlm],
    saved: Mapping[CallKey, BenchCallRow],
    meter: SpendMeter,
    config: RunConfig,
    on_row: Callable[[BenchCallRow], None],
    question_concurrency: int,
) -> list[BenchCallRow]:
    if question_concurrency < 1:
        raise ValueError("question_concurrency must be positive")
    semaphore = asyncio.Semaphore(question_concurrency)

    async def run_one(context: QuestionContext) -> list[BenchCallRow]:
        async with semaphore:
            return await run_question(
                context,
                models=models,
                calls=calls,
                parsers=parsers,
                saved=saved,
                meter=meter,
                config=config,
                on_row=on_row,
            )

    return [row for rows in await asyncio.gather(*(run_one(context) for context in contexts)) for row in rows]
