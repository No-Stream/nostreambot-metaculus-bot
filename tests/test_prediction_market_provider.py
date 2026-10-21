"""The `prediction_market_provider` factory — the research-provider entrypoint of the SEAM module.

What this file covers is the factory's own decisions: the two gates (`is_benchmarking`, the env
flag) and that neither flips the other's default, the deliberate `as_of=None` it passes, the
end-to-end fetch-and-format it hands the orchestrator, and the timeout-versus-stage-budget
arithmetic it warns about at init.

The rest of the seam's coverage lives in siblings, all of which share
`tests/market_retrieval_fakes.py` for the fake aiohttp session, the stubbed LLM stages and the
realistic venue payloads:

- `test_prediction_market_transport.py` — the shared bounded GET, and the catalogue cache/TTL.
- `test_prediction_market_llm_stages.py` — the two LLM stages' configs, budgets and concurrency.
- `test_prediction_market_snapshot.py` — the four-stage pipeline end to end, and its fail-opens.
- `test_prediction_market_formatter.py` — the formatter delegate and the liquidity labels.
- `test_prediction_market_diagnostics.py` — the seven per-source tokens, and provider health.

The retrieval machinery itself (venue parsers, pool assembly, the ranker prompt and parser, the
renderer) is unit-tested per module in `tests/test_market_retrieval_*.py`.

All HTTP is mocked via fake sessions and both LLM stages are patched, so nothing here opens a
socket or bills a key.
"""

from __future__ import annotations

import inspect
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from metaculus_bot.constants import PREDICTION_MARKET_TIMEOUT
from metaculus_bot.research import prediction_market as pmp
from metaculus_bot.research.market_retrieval.rendering import TABLE_COLUMNS
from metaculus_bot.research.prediction_market import MarketSnapshot, prediction_market_provider
from tests import market_retrieval_fakes as _fakes
from tests.market_retrieval_fakes import KALSHI_EVENTS_URL as _KALSHI_EVENTS_URL
from tests.market_retrieval_fakes import MANIFOLD_SEARCH_URL as _MANIFOLD_SEARCH_URL
from tests.market_retrieval_fakes import FakeResponse, FakeSession
from tests.market_retrieval_fakes import handlers as _handlers
from tests.market_retrieval_fakes import market_llm as _market_llm
from tests.market_retrieval_fakes import rank_one_per_venue as _rank_one_per_venue

# Shared fixtures have to be bound in THIS module's globals for pytest to find them, and bound by
# ASSIGNMENT rather than `from ... import`: pyflakes reads a same-named fixture parameter as an
# F811 redefinition of an import, so importing them would put a per-signature suppression on every
# test below. `reset_provider_caches` is autouse and resets the seam's caches around each test.
reset_provider_caches = _fakes.reset_provider_caches
mock_question = _fakes.mock_question
manifold_payload = _fakes.manifold_payload
kalshi_events_payload = _fakes.kalshi_events_payload


class TestProviderFactory:
    @pytest.mark.asyncio
    async def test_disabled_flag_returns_empty_at_the_provider_entrypoint(self, monkeypatch, mock_question):
        """Defence in depth: the orchestrator also gates registration on this flag, but the
        provider re-checks so a direct caller cannot bypass it."""
        monkeypatch.delenv("PREDICTION_MARKETS_ENABLED", raising=False)

        assert await prediction_market_provider()(mock_question) == ""

    @pytest.mark.asyncio
    async def test_is_benchmarking_short_circuits_regardless_of_the_env_flag(self, monkeypatch, mock_question):
        """There is no orchestrator-level backstop, so THIS check is the backtest defence.

        Markets retain their last-trade price after resolution, and the `as_of` filter alone was
        never sufficient — a market that closes between `as_of` and now still leaks. Mirrors the
        contract `gemini_search_provider` and `native_search_provider` use.
        """
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")

        assert await prediction_market_provider(is_benchmarking=True)(mock_question) == ""

    @pytest.mark.asyncio
    async def test_the_default_factory_call_gates_only_on_the_env_flag(self, monkeypatch, mock_question):
        """Control for the above: adding the parameter must not flip the default, which would
        silence the provider in prod where is_benchmarking is False."""
        monkeypatch.delenv("PREDICTION_MARKETS_ENABLED", raising=False)

        assert await prediction_market_provider()(mock_question) == ""

    @pytest.mark.asyncio
    async def test_the_provider_path_passes_no_as_of(self, monkeypatch, mock_question, kalshi_events_payload):
        """The provider passes `as_of=None`, deliberately, even for a question with a scheduled
        resolution.

        The old derivation (`scheduled_resolution_time - 1 day`) was worse than inert: it dropped
        every market closing before the question resolved, which is exactly the "same quantity,
        adjacent month" class that supplied most of the ranked arm's near-identical rows, and
        prod telemetry recorded the cost — 20 of 47 archived runs had Polymarket fetch candidates
        and render nothing because of it. The benchmarking guard is the leakage defence.
        """
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")
        mock_question.scheduled_resolution_time = datetime(2026, 8, 1, tzinfo=UTC)
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        captured: list[datetime | None] = []
        original = pmp.fetch_market_snapshot

        async def _capturing(question_arg: Any, *, as_of: datetime | None = None, **kwargs: Any) -> MarketSnapshot:
            captured.append(as_of)
            return await original(question_arg, as_of=as_of, **kwargs)

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", _market_llm()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
            patch.object(pmp, "fetch_market_snapshot", _capturing),
        ):
            await pmp.prediction_market_provider()(mock_question)

        assert captured == [None]

    @pytest.mark.asyncio
    async def test_the_enabled_provider_fetches_and_formats(
        self, monkeypatch, mock_question, kalshi_events_payload, manifold_payload
    ):
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
                _MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload),
            }
        )

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", _market_llm(ranking=_rank_one_per_venue)),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            out = await pmp.prediction_market_provider()(mock_question)

        assert "MAY be relevant" in out
        assert "| " + " | ".join(TABLE_COLUMNS) + " |" in out
        assert "### Resolution criteria / rules" in out

    def test_a_timeout_below_the_stage_budget_warns_loudly_at_init(self, monkeypatch, caplog):
        """A stale `PREDICTION_MARKET_TIMEOUT=30` left in someone's .env would otherwise surface
        only as a generic snapshot timeout on every question, with the real cause invisible."""
        monkeypatch.setattr(pmp, "PREDICTION_MARKET_TIMEOUT", 30.0)

        with caplog.at_level(logging.WARNING):
            prediction_market_provider()

        logged = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "BELOW the pipeline's worst-case stage sum" in logged
        assert str(pmp.SNAPSHOT_STAGE_BUDGET_S) in logged

    def test_the_default_timeout_clears_the_stage_budget(self):
        assert PREDICTION_MARKET_TIMEOUT >= pmp.SNAPSHOT_STAGE_BUDGET_S

    def test_the_snapshot_timeout_default_is_the_same_constant_the_provider_passes(self):
        """Read off the signature, because a literal default here is a guaranteed timeout.

        The provider path always passes `timeout=` explicitly, so a stale default degrades
        nothing in prod and nothing else would catch it — but a direct caller (backtest, replay
        tool) omitting the argument gets `error(timeout)` on every question plus a source-loss
        bump, since stage 1a alone outruns the 5.0s this default carried while
        `SNAPSHOT_STAGE_BUDGET_S` grew past 190s.
        """
        default = inspect.signature(pmp.fetch_market_snapshot).parameters["timeout"].default

        assert default == PREDICTION_MARKET_TIMEOUT
        assert default >= pmp.SNAPSHOT_STAGE_BUDGET_S
