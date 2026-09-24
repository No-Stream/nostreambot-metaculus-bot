"""The prediction-market seam's two LLM stages: their config pins, budgets and concurrency.

The query author (stage 1b) and the ranker (stage 3) share one invocation helper, so what needs
pinning is that each stage carries its OWN wall timeout and backoff ladder through that helper,
that the two configs pin a single attempt with headroom over the measured completions, and that
stage 1a's catalogue pull is gathered with stage 1b rather than awaited before it.

Fakes and payload fixtures live in `tests/market_retrieval_fakes.py`.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from metaculus_bot.constants import (
    MARKET_QUERY_AUTHOR_BACKOFFS,
    MARKET_QUERY_AUTHOR_WALL_TIMEOUT,
    MARKET_RANKER_BACKOFFS,
    MARKET_RANKER_WALL_TIMEOUT,
    PREDICTION_MARKET_TIMEOUT,
)
from metaculus_bot.llm_configs import MARKET_QUERY_AUTHOR_LLM_CONFIG, MARKET_RANKER_LLM_CONFIG
from metaculus_bot.research import prediction_market as pmp
from metaculus_bot.research.market_retrieval import venues
from tests import market_retrieval_fakes as _fakes
from tests.market_retrieval_fakes import AUTHOR_JSON as _AUTHOR_JSON
from tests.market_retrieval_fakes import RANKER_CUE as _RANKER_CUE
from tests.market_retrieval_fakes import FakeSession
from tests.market_retrieval_fakes import handlers as _handlers

# Bound by assignment rather than imported — see the note in tests/test_prediction_market_transport.py.
reset_provider_caches = _fakes.reset_provider_caches
mock_question = _fakes.mock_question


def _stage_worst(wall: float, backoffs: tuple[float, ...]) -> float:
    return (len(backoffs) + 1) * wall + sum(backoffs)


class TestMarketLlmStages:
    def test_both_configs_pin_a_single_attempt(self):
        """`allowed_tries=1` is required, not decorative: the elapsed-gated `llm_retry` wrapper
        is the SOLE retry layer, and leaving this unpinned inherits forecasting-tools' default of
        2 with an un-gated `random.uniform(5, 10)` tenacity sleep inside the snapshot budget."""
        assert MARKET_RANKER_LLM_CONFIG["allowed_tries"] == 1
        assert MARKET_QUERY_AUTHOR_LLM_CONFIG["allowed_tries"] == 1

    def test_neither_config_caps_output_tokens(self):
        """A truncated completion is a TOTAL loss on both stages — the ranking fails open and the
        author's object will not parse — and reasoning tokens count against any cap. Since
        2026-09-22 neither stage sets `max_tokens`; the elapsed-gated walls bound a runaway."""
        assert "max_tokens" not in MARKET_RANKER_LLM_CONFIG
        assert "max_tokens" not in MARKET_QUERY_AUTHOR_LLM_CONFIG

    def test_each_litellm_timeout_sits_above_its_elapsed_gated_wall(self):
        """The wall is meant to be the binding bound. A litellm timeout below it would fire
        first, and the stage's own budget arithmetic would describe nothing."""
        assert MARKET_RANKER_LLM_CONFIG["timeout"] > MARKET_RANKER_WALL_TIMEOUT
        assert MARKET_QUERY_AUTHOR_LLM_CONFIG["timeout"] > MARKET_QUERY_AUTHOR_WALL_TIMEOUT

    @pytest.mark.asyncio
    async def test_each_stage_passes_its_own_budget_through_the_shared_helper(self, mock_question):
        """The call-site spy, and the more valuable half of the budget coverage.

        A constants-level inequality (tests/test_llm_retry.py) proves the numbers fit; it cannot
        prove the CALLS carry them. This does: one helper, two stages, each with its own wall and
        backoff ladder, and their serial-chain sum inside the snapshot budget.
        """
        calls: list[dict[str, Any]] = []

        # Both suppressions are structural: this stands in for an `async def` production seam, so
        # it cannot be sync and has no checkpoint to offer.
        async def _spy(config: dict, prompt: str, **kwargs: Any) -> str:
            calls.append({"config": config, **kwargs})
            return "[]" if _RANKER_CUE in prompt else _AUTHOR_JSON

        with (
            patch.object(pmp, "_invoke_market_llm", _spy),
            patch.object(pmp, "_get_session", lambda: FakeSession(_handlers())),
        ):
            await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        by_label = {call["label"]: call for call in calls}
        assert set(by_label) == {"market_query_author", "market_ranker"}

        author = by_label["market_query_author"]
        assert author["config"] is MARKET_QUERY_AUTHOR_LLM_CONFIG
        assert author["wall_timeout"] == MARKET_QUERY_AUTHOR_WALL_TIMEOUT
        assert author["backoffs"] == MARKET_QUERY_AUTHOR_BACKOFFS

        ranker = by_label["market_ranker"]
        assert ranker["config"] is MARKET_RANKER_LLM_CONFIG
        assert ranker["wall_timeout"] == MARKET_RANKER_WALL_TIMEOUT
        assert ranker["backoffs"] == MARKET_RANKER_BACKOFFS

        chain = sum(_stage_worst(call["wall_timeout"], call["backoffs"]) for call in calls)
        assert chain < PREDICTION_MARKET_TIMEOUT

    @pytest.mark.asyncio
    async def test_the_catalogue_pull_and_the_query_author_run_concurrently(self, mock_question):
        """Stage 1a and stage 1b must be GATHERED, not awaited in sequence.

        Nothing else would catch a serial wiring: the chain still fits under 150s today, so a
        serial version would pass every budget test and every behavioural test while spending
        ~20s of wall clock per question for nothing. The prefetches need no queries at all — the
        catalogue IS the venue and the settlement join keys on domains — which is what makes the
        concurrency free.
        """
        order: list[str] = []
        real_prefetch = venues.kalshi_prefetch_events

        async def _slow_prefetch(session: Any, **kwargs: Any) -> Any:
            order.append("catalogue:start")
            await asyncio.sleep(0.02)
            order.append("catalogue:end")
            return await real_prefetch(session, **kwargs)

        async def _slow_llm(config: dict, prompt: str, **_kwargs: Any) -> str:
            order.append("author:start")
            await asyncio.sleep(0.02)
            order.append("author:end")
            return _AUTHOR_JSON

        with (
            patch.object(venues, "kalshi_prefetch_events", _slow_prefetch),
            patch.object(pmp, "_invoke_market_llm", _slow_llm),
            patch.object(pmp, "_get_session", lambda: FakeSession(_handlers())),
        ):
            await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        # Both stages START before either ENDS. Which of the two is scheduled first is a
        # gather-ordering detail worth nothing; overlap is the whole invariant.
        assert set(order[:2]) == {"catalogue:start", "author:start"}, (
            f"stage 1a and 1b were awaited serially rather than gathered: {order}"
        )
