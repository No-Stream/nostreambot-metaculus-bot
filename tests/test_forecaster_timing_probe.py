"""Offline tests for the paid timing probe."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from forecasting_tools import NumericQuestion
from forecasting_tools.ai_models import general_llm as ft_general_llm
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from metaculus_bot.llm_configs import FORECASTER_LLMS, FORECASTER_MODEL_NAMES
from metaculus_bot.numeric.utils import bound_messages
from metaculus_bot.prompts import numeric_prompt
from scripts.probes.forecaster_timing_probe import load_probe_input, parse_args, run_probe


def test_refuses_without_spend_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--qid", "123", "--model", FORECASTER_MODEL_NAMES[0]])


def test_archive_research_builds_production_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    question = NumericQuestion(
        question_text="How many?",
        page_url="https://www.metaculus.com/questions/456/",
        id_of_question=123,
        id_of_post=456,
        resolution_criteria="A count",
        fine_print="",
        background_info="Context",
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
        scheduled_resolution_time=datetime(2027, 1, 1, tzinfo=UTC),
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=False,
    )
    (tmp_path / "123.jsonl").write_text(
        json.dumps(
            {
                "qid": 123,
                "post_id": 456,
                "page_url": question.page_url,
                "research_text": "Archived evidence",
                "timestamp": "2026-01-01",
                "run_id": "123456",
                "run_mode": "tournament",
            }
        )
        + "\n"
        # A newer comment-backfill record: lossy research and a QUESTION url, so it must never be used.
        + json.dumps(
            {
                "qid": 123,
                "on_post": 456,
                "page_url": "https://www.metaculus.com/questions/123/",
                "research_text": "Trimmed comment text",
                "timestamp": "2026-02-01",
                "run_id": "comment-9",
                "run_mode": "tournament",
            }
        )
        + "\n"
    )
    fetched_urls: list[str] = []

    def fake_fetch(url: str) -> NumericQuestion:
        fetched_urls.append(url)
        return question

    monkeypatch.setattr("scripts.probes.forecaster_timing_probe.MetaculusApi.get_question_by_url", fake_fetch)
    result_question, prompt = load_probe_input(123, tmp_path)
    upper, lower = bound_messages(question)
    assert result_question == question
    assert fetched_urls == [question.page_url]
    assert prompt == numeric_prompt(question, "Archived evidence", lower, upper)


async def test_probe_invokes_one_forecaster_and_reports_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    async def fake_completion(**kwargs: object) -> ModelResponse:
        calls.append(kwargs)
        response = ModelResponse()
        response.choices = [Choices(message=Message(role="assistant", content="answer"), finish_reason="stop", index=0)]
        response.usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)  # type: ignore[attr-defined]
        return response

    monkeypatch.setattr(ft_general_llm, "acompletion", fake_completion)
    result = await run_probe(FORECASTER_LLMS[0], "production prompt")
    assert len(calls) == 1
    assert calls[0]["messages"][0]["content"] == "production prompt"
    assert result["output_tokens"] == 5
    assert result["finish_reason"] == "stop"
    assert result["structured_forecast_parsed"] is False


async def test_probe_reports_empty_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_completion(**kwargs: object) -> ModelResponse:
        response = ModelResponse()
        response.choices = [Choices(message=Message(role="assistant", content=None), finish_reason="refusal", index=0)]
        response.usage = Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10)  # type: ignore[attr-defined]
        return response

    monkeypatch.setattr(ft_general_llm, "acompletion", fake_completion)
    result = await run_probe(FORECASTER_LLMS[0], "production prompt")
    assert result["finish_reason"] == "content_filter"  # LiteLLM normalizes refusal to content_filter.
    assert result["structured_forecast_parsed"] is False
