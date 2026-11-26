"""The paid half: the model call on the personal key, the spend meter, and the per-call runner.

The forecaster call goes straight through ``litellm.acompletion``, whose OpenRouter transformer asks for
usage accounting on every request, so each reply carries the charge OpenRouter made and is read by the
same reader the production ledger uses. The ladder's salvage parser is built through the repo's own
builder, which after ``personal_key_only_environment`` can only reach the personal key, and its charges
arrive through the litellm success-callback ledger. A charge OpenRouter did not report stops the run:
the meter fails shut rather than reading an unmeasurable call as free.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import litellm
import openai
from forecasting_tools import GeneralLlm

from metaculus_bot.credit_telemetry import (
    PERSONAL_KEY_ALIAS,
    llm_call_metadata,
)
from metaculus_bot.exceptions import UnitMismatchError, ValueExtractionError
from scripts.probes.bench_common.paid_call import (
    ACCOUNT_REFUSAL_STATUSES,
    CALL_RETRIES,
    CALL_TIMEOUT_S,
    ERROR_TEXT_LIMIT,
    MAX_COMPLETION_TOKENS,
    ModelReply,
    SpendMeter,
    reply_from_response,
)
from scripts.probes.bench_common.paid_call import (
    PARSER_ROLE as PARSER_ROLE,
)
from scripts.probes.bench_common.paid_call import (
    build_parser_llm as build_parser_llm,
)
from scripts.probes.bench_common.paid_call import (
    charged_usd as charged_usd,
)
from scripts.probes.bench_common.paid_call import (
    parser_ledger_usd as parser_ledger_usd,
)
from scripts.probes.bench_common.paid_call import (
    personal_key_only_environment as personal_key_only_environment,
)
from scripts.probes.bench_common.records import (
    STATUS_API_ERROR as STATUS_API_ERROR,
)
from scripts.probes.bench_common.records import (
    STATUS_BUILD_FAILED as STATUS_BUILD_FAILED,
)
from scripts.probes.bench_common.records import (
    STATUS_EXTRACTION_FAILED as STATUS_EXTRACTION_FAILED,
)
from scripts.probes.bench_common.records import (
    STATUS_SCORED,
)
from scripts.probes.bench_common.records import (
    STATUS_SKIPPED_SPEND_CAP as STATUS_SKIPPED_SPEND_CAP,
)
from scripts.probes.bench_common.records import (
    STATUS_UNIT_MISMATCH as STATUS_UNIT_MISMATCH,
)
from scripts.probes.bench_common.scoring import score_reply
from scripts.probes.section_strip_bench.plan import PlanItem
from scripts.probes.section_strip_bench.report import CallRow

logger = logging.getLogger(__name__)

FORECASTER_ROLE = "strip_bench_forecaster"
ModelCall = Callable[[str], Awaitable[ModelReply]]


def build_model_call(model: str, api_key: str, *, reasoning_effort: str | None) -> ModelCall:
    """One completion per prompt on the personal key: the prompt as the only message, no tools."""
    kwargs: dict[str, Any] = {
        "model": f"openrouter/{model}",
        "api_key": api_key,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "timeout": CALL_TIMEOUT_S,
        "num_retries": CALL_RETRIES,
        "metadata": llm_call_metadata(FORECASTER_ROLE, PERSONAL_KEY_ALIAS),
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    async def _call(prompt: str) -> ModelReply:
        response = await litellm.acompletion(messages=[{"role": "user", "content": prompt}], **kwargs)
        return reply_from_response(response)

    return _call


def _row_for(item: PlanItem, status: str) -> CallRow:
    return CallRow(
        question_id=item.question.question_id,
        qtype=item.question.qtype,
        n_options=item.question.n_options,
        arm=item.arm,
        seed=item.seed,
        status=status,
    )


async def _score_into(
    row: CallRow, item: PlanItem, reply: ModelReply, parser_llm: GeneralLlm, *, model_name: str
) -> None:
    """Fill the row from the reply: the score when the ladder and the build succeed, a status otherwise."""
    row.prompt_tokens, row.completion_tokens = reply.tokens.prompt, reply.tokens.completion
    row.reasoning_tokens, row.cached_tokens = reply.tokens.reasoning, reply.tokens.cached
    row.cost_usd, row.rationale = reply.charged_usd, reply.text
    question = item.question
    try:
        scored = await score_reply(
            question.question, question.resolution, reply.text, parser_llm, model_name=model_name
        )
    except ValueExtractionError as exc:
        row.status, row.error = STATUS_EXTRACTION_FAILED, str(exc)[:ERROR_TEXT_LIMIT]
    except UnitMismatchError as exc:
        row.status, row.error = STATUS_UNIT_MISMATCH, str(exc)[:ERROR_TEXT_LIMIT]
    except ValueError as exc:
        row.status, row.error = STATUS_BUILD_FAILED, str(exc)[:ERROR_TEXT_LIMIT]
    else:
        row.status, row.score, row.forecast = STATUS_SCORED, scored.score, scored.forecast
        row.rung, row.block_present = scored.rung, scored.block_present


async def run_item(
    item: PlanItem,
    *,
    call: ModelCall,
    parser_llm: GeneralLlm,
    model_name: str,
    meter: SpendMeter,
    semaphore: asyncio.Semaphore,
) -> CallRow:
    """One call: forecast, extract, score. Every failure mode is a row status, never the end of the run."""
    async with semaphore:
        if (stop_reason := meter.stop_reason) is not None:
            row = _row_for(item, STATUS_SKIPPED_SPEND_CAP)
            row.error = stop_reason
            return row
        started = time.monotonic()
        row = _row_for(item, STATUS_API_ERROR)
        try:
            reply = await call(item.prompt)
        except openai.OpenAIError as exc:  # litellm's whole exception family derives from openai's
            row.error = f"{type(exc).__name__}: {exc}"[:ERROR_TEXT_LIMIT]
            # litellm's own APIError carries status_code without being an APIStatusError, so read it by name.
            if getattr(exc, "status_code", None) in ACCOUNT_REFUSAL_STATUSES:
                meter.refuse(row.error)
        else:
            meter.add(reply.charged_usd)
            await _score_into(row, item, reply, parser_llm, model_name=model_name)
        row.elapsed_s = time.monotonic() - started
        return row


async def run_plan(
    plan: Sequence[PlanItem],
    *,
    call: ModelCall,
    parser_llm: GeneralLlm,
    model_name: str,
    meter: SpendMeter,
    concurrency: int,
    on_row: Callable[[CallRow], None],
) -> list[CallRow]:
    """Run every planned call under the concurrency limit, handing each finished row to ``on_row`` as it lands."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            run_item(item, call=call, parser_llm=parser_llm, model_name=model_name, meter=meter, semaphore=semaphore)
        )
        for item in plan
    ]
    rows: list[CallRow] = []
    for task in asyncio.as_completed(tasks):
        row = await task
        on_row(row)
        rows.append(row)
    return rows
