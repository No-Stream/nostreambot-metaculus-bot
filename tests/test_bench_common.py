from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from metaculus_bot import prompts
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from scripts.probes.bench_common.prompting import prompt_clock
from scripts.probes.bench_common.scoring import score_reply


def _common(qid: int) -> dict[str, Any]:
    return {
        "question_text": "What happens?",
        "id_of_question": qid,
        "id_of_post": qid,
        "page_url": "https://www.metaculus.com/questions/1/",
        "background_info": "",
        "resolution_criteria": "Resolves by the stated date.",
        "fine_print": "",
        "open_time": datetime(2026, 1, 1, tzinfo=UTC),
        "scheduled_resolution_time": datetime(2027, 1, 1, tzinfo=UTC),
        "api_json": {"question": {}},
    }


def _parser() -> MagicMock:
    parser = MagicMock()
    parser.model = "openrouter/test/parser"
    return parser


def test_numeric_score_carries_the_cdf_and_the_stated_percentiles() -> None:
    question = NumericQuestion(
        **_common(1),
        lower_bound=0,
        upper_bound=200,
        open_lower_bound=False,
        open_upper_bound=False,
        zero_point=None,
        cdf_size=201,
        nominal_lower_bound=0,
        nominal_upper_bound=200,
        unit_of_measure=None,
    )
    values = [float(i) for i in range(len(STANDARD_PERCENTILES))]
    declarations = {str(percentile): value for percentile, value in zip(STANDARD_PERCENTILES, values, strict=True)}
    reply = (
        "```json\n"
        + json.dumps({"question_type": "numeric", "declared_percentiles": declarations, "outcome_type": "continuous"})
        + "\n```"
    )

    scored = asyncio.run(score_reply(question, 100.0, reply, _parser(), model_name="test/model"))

    assert scored.cdf is not None
    assert scored.stated_percentiles is not None
    assert len(scored.cdf) == 201
    assert scored.cdf == sorted(scored.cdf)
    assert scored.stated_percentiles == [
        [float(p), float(v)] for p, v in zip(STANDARD_PERCENTILES, values, strict=True)
    ]


def test_binary_and_mc_scores_leave_the_new_fields_empty() -> None:
    binary = BinaryQuestion(**_common(2))
    mc = MultipleChoiceQuestion(**_common(3), options=["A", "B"])
    binary_reply = '```json\n{"question_type":"binary","posterior_prob":0.7}\n```'
    mc_reply = '```json\n{"question_type":"multiple_choice","option_probs":{"A":0.7,"B":0.3}}\n```'

    binary_score = asyncio.run(score_reply(binary, True, binary_reply, _parser(), model_name="test/model"))
    mc_score = asyncio.run(score_reply(mc, "A", mc_reply, _parser(), model_name="test/model"))

    assert binary_score.cdf is None
    assert binary_score.stated_percentiles is None
    assert mc_score.cdf is None
    assert mc_score.stated_percentiles is None


def test_prompt_clock_swaps_and_restores() -> None:
    original_window = prompts._forecasting_window_str
    original_today = prompts._today_str
    question = BinaryQuestion(**_common(4))

    try:
        with prompt_clock(lambda _question: "archived window", "2026-10-22"):
            assert prompts._today_str() == "2026-10-22"
            assert prompts._forecasting_window_str(question) == "archived window"

        assert prompts._forecasting_window_str is original_window
        assert prompts._today_str is original_today

        try:
            with prompt_clock(lambda _question: "exception window", "2026-10-22"):
                raise RuntimeError("body failed")
        except RuntimeError:
            pass
        assert prompts._forecasting_window_str is original_window
        assert prompts._today_str is original_today
    finally:
        prompts._forecasting_window_str = original_window
        prompts._today_str = original_today
