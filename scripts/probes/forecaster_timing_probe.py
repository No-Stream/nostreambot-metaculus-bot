"""Time one paid production forecaster call using archived research.

The archive contains research and a question URL, but not the question's full criteria
or numeric bounds. The probe fetches the question metadata from Metaculus before calling
one forecaster. It never publishes a forecast.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from forecasting_tools import GeneralLlm, MetaculusApi, NumericQuestion
from forecasting_tools.ai_models import general_llm as ft_general_llm

from metaculus_bot.llm_configs import FORECASTER_LLMS, FORECASTER_MODEL_NAMES
from metaculus_bot.numeric.utils import bound_messages
from metaculus_bot.prompts import numeric_prompt
from metaculus_bot.structured_output_schema import parse_structured_block

ARCHIVE_DIR = Path("backtests/research_archive/by_qid")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qid", required=True, type=int)
    parser.add_argument("--model", required=True, choices=FORECASTER_MODEL_NAMES)
    parser.add_argument("--i-accept-spend", action="store_true")
    args = parser.parse_args(argv)
    if not args.i_accept_spend:
        parser.error("This probe spends real credits; pass --i-accept-spend")
    return args


def load_probe_input(qid: int, archive_dir: Path = ARCHIVE_DIR) -> tuple[NumericQuestion, str]:
    records = [json.loads(line) for line in (archive_dir / f"{qid}.jsonl").read_text().splitlines() if line.strip()]
    record = max(records, key=lambda item: item["timestamp"])
    if record["qid"] != qid or not record["research_text"]:
        raise ValueError("Archive record has the wrong question or no research")
    question = MetaculusApi.get_question_by_url(record["page_url"])
    if not isinstance(question, NumericQuestion) or question.id_of_question != qid:
        raise ValueError("Archive question must identify a numeric Metaculus question")
    upper, lower = bound_messages(question)
    return question, numeric_prompt(question, record["research_text"], lower, upper)


async def run_probe(llm: GeneralLlm, prompt: str) -> dict[str, Any]:
    """Invoke the configured slot once and capture LiteLLM's response metadata."""
    responses: list[Any] = []
    original = ft_general_llm.acompletion

    async def capture(**kwargs: Any) -> Any:
        response = await original(**kwargs)
        responses.append(response)
        return response

    started = time.monotonic()
    with patch.object(ft_general_llm, "acompletion", capture):
        try:
            answer = await llm.invoke(prompt)
        except (AssertionError, RuntimeError):
            if not responses or responses[-1].choices[0].message.content not in (None, ""):
                raise
            answer = ""
    elapsed = time.monotonic() - started
    response = responses[-1]
    usage = response.usage
    details = usage.completion_tokens_details
    return {
        "wall_seconds": elapsed,
        "output_tokens": usage.completion_tokens,
        "reasoning_tokens": details.reasoning_tokens if details else None,
        "finish_reason": response.choices[0].finish_reason,
        "structured_forecast_parsed": parse_structured_block(answer, "numeric") is not None,
        "cost_usd": response._hidden_params.get("response_cost"),
    }


def main() -> None:
    args = parse_args()
    _, prompt = load_probe_input(args.qid)
    llm = FORECASTER_LLMS[FORECASTER_MODEL_NAMES.index(args.model)]
    print(json.dumps(asyncio.run(run_probe(llm, prompt)), indent=2))


if __name__ == "__main__":
    main()
