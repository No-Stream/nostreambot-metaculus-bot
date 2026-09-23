"""Emit-wiring tests: each research provider calls record_raw_research with its raw payload.

The logger's own serialization/truncation/IO behavior is covered in
test_raw_research_log.py. These tests assert the WIRING — that each provider hands
its raw payload (and the correct qid/provider/phase) to record_raw_research at the
point the payload exists — by patching the function at each provider module. That
keeps them light (no file round-trip) and pins the qid-threading that AskNews and
Gemini need (their raw payload and the question live in different scopes).
"""

import asyncio
import dataclasses
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metaculus_bot.research.gemini_search import gemini_search_provider
from metaculus_bot.research.prediction_market import prediction_market_provider
from metaculus_bot.research.providers import _asknews_provider, native_search_provider
from metaculus_bot.research.resolution_source import FetchResult, resolution_source_provider
from metaculus_bot.research.targeted import run_gap_fill_pass


def _make_q(text: str = "Will X happen?", qid: int = 555) -> MagicMock:
    q = MagicMock()
    q.question_text = text
    q.id_of_question = qid
    return q


@dataclasses.dataclass
class _StubArticle:
    """AskNews-Article-shaped stub: the fields _format_single_article/dedup read."""

    eng_title: str
    summary: str
    language: str
    pub_date: datetime
    source_id: str
    article_url: str


@pytest.mark.asyncio
async def test_asknews_emits_raw_records_for_both_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")

    hot = [_StubArticle("Hot", "s", "en", datetime(2026, 7, 19, tzinfo=UTC), "src", "http://a")]
    hist = [_StubArticle("Hist", "s", "en", datetime(2026, 7, 18, tzinfo=UTC), "src", "http://b")]

    async def mock_search_news(*_args, **kwargs):
        await asyncio.sleep(0)
        resp = AsyncMock()
        resp.as_dicts = hist if kwargs.get("strategy") == "news knowledge" else hot
        return resp

    with (
        patch("asknews_sdk.AsyncAskNewsSDK") as sdk_class,
        patch("asyncio.sleep", new=AsyncMock()),
        patch("metaculus_bot.research.providers.record_raw_research") as rec,
    ):
        sdk = AsyncMock()
        sdk.news.search_news = mock_search_news
        sdk_class.return_value.__aenter__.return_value = sdk

        await _asknews_provider()(_make_q())

    by_phase = {c.kwargs["phase"]: c.kwargs for c in rec.call_args_list}
    assert set(by_phase) == {"hot", "historical"}
    assert by_phase["hot"]["provider"] == "asknews"
    assert by_phase["hot"]["qid"] == 555
    assert by_phase["hot"]["payload"] == hot
    assert by_phase["historical"]["payload"] == hist


@pytest.mark.asyncio
async def test_native_search_emits_raw_completion() -> None:
    class MockLlm:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.model = "mock"

        async def invoke(self, _prompt: str) -> str:
            await asyncio.sleep(0)
            return "RAW COMPLETION TEXT"

    with (
        patch("metaculus_bot.research.providers.build_llm_with_openrouter_fallback", MockLlm),
        patch("metaculus_bot.research.providers.record_raw_research") as rec,
    ):
        await native_search_provider()(_make_q())

    rec.assert_called_once()
    assert rec.call_args.kwargs["provider"] == "native_search"
    assert rec.call_args.kwargs["qid"] == 555
    assert rec.call_args.kwargs["payload"] == "RAW COMPLETION TEXT"


@pytest.mark.asyncio
async def test_gemini_search_emits_raw_response_with_qid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    candidate = SimpleNamespace(grounding_metadata=None, url_context_metadata=None)
    response = SimpleNamespace(text="grounded text", candidates=[candidate])
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=client),
        patch("metaculus_bot.research.gemini_search.resolve_search_redirects", new=AsyncMock(return_value={})),
        patch("metaculus_bot.research.gemini_search.record_raw_research") as rec,
    ):
        await gemini_search_provider()(_make_q())

    # A single SDK call is archived once; there is no grounding retry anymore.
    rec.assert_called_once()
    assert rec.call_args.kwargs["provider"] == "gemini_search"
    assert rec.call_args.kwargs["qid"] == 555
    assert rec.call_args.kwargs["payload"] is response


@pytest.mark.asyncio
async def test_prediction_market_emits_raw_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")

    snapshot = SimpleNamespace(matches=[], sources={})

    with (
        patch("metaculus_bot.research.prediction_market.fetch_market_snapshot", AsyncMock(return_value=snapshot)),
        patch("metaculus_bot.research.prediction_market.format_snapshot_for_research", return_value="table"),
        patch("metaculus_bot.research.prediction_market.record_raw_research") as rec,
    ):
        await prediction_market_provider(is_benchmarking=False)(_make_q())

    rec.assert_called_once()
    assert rec.call_args.kwargs["provider"] == "prediction_market"
    assert rec.call_args.kwargs["qid"] == 555
    assert rec.call_args.kwargs["payload"] is snapshot


@pytest.mark.asyncio
async def test_resolution_source_emits_raw_fetch_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")

    results = [FetchResult(url="http://x", status="success", text="body", http_status=200, content_type="text/html")]

    with (
        patch("metaculus_bot.research.resolution_source.select_fetchable_urls", return_value=["http://x"]),
        patch("metaculus_bot.research.resolution_source.fetch_resolution_sources", AsyncMock(return_value=results)),
        patch("metaculus_bot.research.resolution_source.record_raw_research") as rec,
    ):
        await resolution_source_provider(is_benchmarking=False)(_make_q())

    rec.assert_called_once()
    assert rec.call_args.kwargs["provider"] == "resolution_source"
    assert rec.call_args.kwargs["qid"] == 555
    assert rec.call_args.kwargs["payload"] is results


@pytest.mark.asyncio
async def test_gap_fill_emits_gaps_and_results() -> None:
    gaps = [
        {
            "gap": "what is X",
            "search_query": "X",
            "why_matters": "because",
            "answerable_now": True,
            "already_in_first_pass": False,
            "same_need_as": None,
        }
    ]

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", AsyncMock(return_value=gaps)),
        patch("metaculus_bot.research.targeted._resolve_single_gap", AsyncMock(return_value="answer to X")),
        patch("metaculus_bot.research.targeted.record_raw_research") as rec,
    ):
        await run_gap_fill_pass(_make_q(), "first-pass research")

    rec.assert_called_once()
    assert rec.call_args.kwargs["provider"] == "gap_fill"
    assert rec.call_args.kwargs["qid"] == 555
    payload = rec.call_args.kwargs["payload"]
    assert payload["gaps"] == gaps
    assert payload["results"] == ["answer to X"]
    assert payload["dropped"] == []
