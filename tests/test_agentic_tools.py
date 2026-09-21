from __future__ import annotations

import asyncio
import io
import logging
import socket
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from time import monotonic
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlsplit

import aiohttp
import pytest
from google.genai import types as genai_types
from PIL import Image
from pypdf import PdfWriter

from metaculus_bot.constants import (
    DOCUMENT_DIGEST_TOP_K,
    DOCUMENT_TEXT_PDF_MAX_BYTES,
    GAP_FILL_V2_READER_HTTP_ATTEMPTS,
    GAP_FILL_V2_READER_MODEL,
    GAP_FILL_V2_READER_THINKING_LEVEL,
    RESOLUTION_SOURCE_HTTP_TIMEOUT,
    RESOLUTION_SOURCE_IMPERSONATE_ENABLED_ENV,
    RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
    URL_CONTEXT_SIZE_GATE_TOKENS,
)
from metaculus_bot.research import (
    derived_api,
    document_cache,
    http_fetch,
    impersonated_fetch,
    rendered_fetch,
    robots_policy,
)
from metaculus_bot.research import providers as research_providers
from metaculus_bot.research.agentic import fetch_outcomes, local_document, provenance, tool_backends
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.agentic.loop import _harvest_verification_tiers, _method_to_tier, _tool_schemas
from metaculus_bot.research.agentic.tool_descriptions import FETCH_DESCRIPTION
from metaculus_bot.research.agentic.types import ToolOutcome
from metaculus_bot.research.document_text import extract_pdf_text
from metaculus_bot.research.fetch_ladder import classify, context, direct_fetch, run_cache, rungs, throttle, verdict
from metaculus_bot.research.fetch_ladder.policy import (
    GAP_FILL_DIRECT_POLICY,
    GAP_FILL_DOCUMENT_POLICY,
    GAP_FILL_FETCH_POLICY,
    RESOLUTION_SOURCE_POLICY,
    LadderPolicy,
)
from metaculus_bot.research.gemini_client_config import gemini_retry_sleep_allowance_s
from metaculus_bot.research.impersonated_fetch import (
    IMPERSONATE_TRIGGER_STATUSES,
    ImpersonateBodyTooLarge,
    ImpersonateBudgetExhausted,
    ImpersonateDeclined,
    ImpersonatedResponse,
    ImpersonateHopRefused,
    ImpersonatePinNotHeld,
    ImpersonateRedirectLimit,
    ImpersonateTransportError,
    impersonation_refused,
    reset_impersonation_memo,
)
from metaculus_bot.research.resolution_fetch_result import (
    _NON_OK_FETCH_STATUS,
    FetchResult,
    FetchStatus,
    FetchStatusReason,
)
from metaculus_bot.research.robots_policy import robots_txt_url
from metaculus_bot.research.wayback import wayback_snapshot_url
from scripts.telemetry.markers import MARKER_SPECS, qid_from_ref
from tests.playwright_fakes import FakeChromium, FakePage, FakePlaywrightManager, install_fake_playwright
from tests.resolution_source_fakes import _escape_config, _fake_render, _impersonated, fake_impersonated_fetch
from tests.test_document_text import build_text_pdf


class _FakeResponse:
    def __init__(self, *, status: int, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeSession:
    """Serves a queued sequence of responses; the last response repeats.

    Single-response construction keeps the original fixed-response behavior;
    multi-response construction lets redirect tests script a chain of hops.
    """

    def __init__(self, *responses: _FakeResponse) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, bool]] = []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def get(self, url: str, *, allow_redirects: bool = False, **kwargs: Any) -> _FakeResponse:
        """The next queued response, recording the URL asked for.

        ``**kwargs`` swallows the shared ladder's per-hop ``ClientTimeout``, a clamp no fake models.
        """
        del kwargs
        self.calls.append((url, allow_redirects))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _scanned_pdf() -> bytes:
    """A structurally valid PDF with a page and no text layer at all — a scan."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _static_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), "navy").save(buffer, format="PNG")
    return buffer.getvalue()


def _serve_pdf(monkeypatch: pytest.MonkeyPatch, body: bytes, *, content_type: str = "application/pdf") -> AsyncMock:
    """Wire the shared direct rung to answer one request with ``body`` under ``content_type``.

    Patches ``classify.read_body_capped`` rather than teaching the fake response object to stream,
    because the cap that read runs under is the thing the PDF rung changes and each test wants
    to state the body it is classifying, not the transport. Returns that spy, which is how a
    test counts requests: one request has to serve both a paginated fetch and a later digest.
    """
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": content_type}))
    read_body = AsyncMock(return_value=body)
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
    monkeypatch.setattr(classify, "read_body_capped", read_body)
    return read_body


_URL = "https://example.com/page"
_DIRECT_FALLBACK = FetchResult(url="", status="not_found", text="", http_status=404, content_type=None)


def _direct(
    status: FetchStatus,
    *,
    url: str = _URL,
    text: str = "",
    http_status: int | None = None,
    reason: FetchStatusReason | None = None,
    links: Sequence[str] = (),
    escalate_rendered: bool = False,
    content_type: str | None = "text/html",
) -> FetchResult:
    """One canned outcome of the shared ladder's DIRECT fetch, as the gap-fill verdict would leave it."""
    phrase = throttle.matched_throttle_phrase(text) if status == "success" else None
    return FetchResult(
        url=url,
        status="throttled" if phrase is not None else status,
        text="" if phrase is not None else text,
        http_status=http_status,
        content_type=content_type,
        status_reason=reason,
        links=list(links),
        escalate_rendered=escalate_rendered,
        throttle_phrase=phrase,
        throttle_chars=None if phrase is None else len(text.strip()),
    )


def _serve_direct(monkeypatch: pytest.MonkeyPatch, answer: FetchResult | dict[str, FetchResult]) -> list[str]:
    """Answer the shared ladder's DIRECT fetch with ``answer``, leaving every rung's gate live.

    The shared direct-fetch seam is also the one every rung's own request goes
    through — the archive snapshot, a remembered feed, the robots pre-check — so a dict keys those
    apart by URL and anything unlisted comes back ``not_found`` rather than reaching the network.
    Returns the list of URLs it was asked for, in order, which is how a test counts requests.
    """
    answers = answer if isinstance(answer, dict) else {_URL: answer}
    asked: list[str] = []

    async def _fake_direct(session: Any, url: str, host_sems: Any, ctx: Any) -> FetchResult:
        del session, host_sems
        await asyncio.sleep(0)
        asked.append(url)
        canned = answers.get(url)
        result = canned if canned is not None else replace(_DIRECT_FALLBACK, url=url)
        if result.status == "success":
            artifact = run_cache.HtmlRead(
                url=result.url,
                http_status=result.http_status,
                content_type=result.content_type,
                extraction=verdict.PageExtraction(text=result.text),
                chart_block="",
                datawrapper_charts=(),
                unreadable_embeds=(),
                links=tuple(result.links),
                routing_body=b"<html",
            )
            ctx.capture_read(result, artifact)
        return result

    monkeypatch.setattr(direct_fetch, "_fetch_direct", _fake_direct)
    return asked


def _serve_rendered(monkeypatch: pytest.MonkeyPatch, rescue: FetchResult | None) -> list[str]:
    """Answer the shared ladder's BROWSER rung with ``rescue``, or None to decline.

    The dispatcher's decision to reach the browser at all stays live, which is what the escalation
    tests are about. Returns the
    URLs the rung was invoked on.
    """
    rendered_on: list[str] = []

    async def _fake_rendered(url: str, direct: FetchResult, host_sems: Any, ctx: Any) -> FetchResult | None:
        del host_sems, ctx
        await asyncio.sleep(0)
        rendered_on.append(direct.url)
        return rescue

    monkeypatch.setattr(rungs, "_rendered_rung", _fake_rendered)
    return rendered_on


async def _fetch_direct_only(url: str) -> fetch_outcomes.PlainFetchResult:
    """One DIRECT fetch through the shared ladder, as this ladder's own result.

    The redirect loop, classification and local document read with no escalation rung
    (``GAP_FILL_DIRECT_POLICY``). Tests that pin the direct classification drive this.
    """
    return await agentic_tools._fetch_via_ladder(url, query="", pol=GAP_FILL_DIRECT_POLICY, ctx=None)


@pytest.fixture(autouse=True)
def _decline_the_impersonated_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty the impersonated retry's trigger set for every test in this module by default.

    The same switch ``tests/resolution_source/conftest.py`` flips for the Tier-1 rung, because both
    fetchers read ``impersonated_fetch.IMPERSONATE_TRIGGER_STATUSES`` at call time: a plain 403 is a
    result many tests here produce on purpose (the robots pre-check tests, the paid-reader tests),
    and with no transport double installed an unwanted fire would reach the real
    ``fetch_impersonated`` and trip the suite's ``_block_native_egress`` guard. Emptying the set
    declines before the retry looks at anything, so the plain ``blocked`` stands exactly as it did
    before the rung existed. ``TestGapFillV2ImpersonatedRetry`` restores the transport's own
    constant object and installs the shared double.
    """
    monkeypatch.setattr(impersonated_fetch, "IMPERSONATE_TRIGGER_STATUSES", frozenset())


@pytest.fixture(autouse=True)
def _reset_tool_state() -> None:
    """Drop every piece of run-scoped state the tools share, so no test inherits another's."""
    document_cache.clear_document_cache()
    http_fetch.reset_pdf_parse_semaphore()
    agentic_tools._FETCH_HOST_SEMAPHORES.clear()
    # Shared with the Tier-1 reader and the Tier-1 rungs, so both reset through their own modules.
    robots_policy.reset_robots_cache()
    derived_api.reset_derived_endpoints()
    # A FRESH launch semaphore too, so no test's event-loop binding leaks into the next.
    rendered_fetch.reset_render_state()


def test_tool_schemas_round_trip_for_public_tools() -> None:
    tools = agentic_tools.build_gap_fill_tools("topic")
    schemas = _tool_schemas(tools, must_conclude=False)

    by_name = {entry["function"]["name"]: entry["function"] for entry in schemas}
    assert by_name["search_news"]["parameters"]["required"] == ["query"]
    assert by_name["search_web"]["parameters"]["properties"]["end_published_date"]["type"] == ["string", "null"]
    assert by_name["fetch"]["parameters"]["properties"]["start_char"]["minimum"] == 0
    assert by_name["read_document"]["parameters"]["required"] == ["url", "ask"]


@pytest.mark.asyncio
async def test_search_web_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "key")
    monkeypatch.setattr(
        agentic_tools,
        "_call_exa_search",
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    title="Result title",
                    url="https://example.com/a",
                    published_date="2026-07-15",
                    highlights=["First highlight", "Second highlight"],
                )
            ]
        ),
    )

    outcome = await agentic_tools.search_web("query")

    assert outcome.status == "ok"
    assert outcome.method == "search"
    assert "Result title" in outcome.content_markdown
    assert "https://example.com/a" in outcome.content_markdown
    assert "- First highlight" in outcome.content_markdown


class _FakeHttpxAsyncClient:
    """httpx.AsyncClient double that records its constructor kwargs.

    Lets the Exa tests assert the client-side timeout was applied without a
    real socket. Instances append their kwargs to the shared ``captured`` list.
    """

    def __init__(self, captured: list[dict[str, Any]], **kwargs: Any) -> None:
        captured.append(kwargs)

    async def aclose(self) -> None:
        return None


def _patch_async_exa(monkeypatch: pytest.MonkeyPatch, searcher: MagicMock) -> list[dict[str, Any]]:
    """Wire fake ``exa_py.AsyncExa`` + ``httpx`` for a ``search_web`` test.

    ``searcher`` drives ``AsyncExa.search`` (call it to raise/return); the
    returned list captures each ``httpx.AsyncClient(**kwargs)`` construction.
    """
    captured: list[dict[str, Any]] = []

    class FakeAsyncExa:
        def __init__(self, api_key: str | None = None) -> None:
            self.base_url = "https://api.exa.ai"
            self.headers = {"x-api-key": api_key}
            self._client: Any = None

        async def search(self, **kwargs: Any) -> Any:
            return searcher(**kwargs)

    monkeypatch.setitem(sys.modules, "exa_py", SimpleNamespace(AsyncExa=FakeAsyncExa))
    monkeypatch.setitem(
        sys.modules, "httpx", SimpleNamespace(AsyncClient=lambda **kwargs: _FakeHttpxAsyncClient(captured, **kwargs))
    )
    return captured


@pytest.mark.asyncio
async def test_search_web_retries_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "key")
    searcher = MagicMock(
        side_effect=[
            RuntimeError("429 too many requests"),
            RuntimeError("rate limit"),
            SimpleNamespace(results=[SimpleNamespace(title="Recovered", url="https://example.com", highlights=[])]),
        ]
    )
    sleeps: list[float] = []

    monkeypatch.setattr("asyncio.sleep", AsyncMock(side_effect=sleeps.append))
    _patch_async_exa(monkeypatch, searcher)

    outcome = await agentic_tools.search_web("query")

    assert outcome.status == "ok"
    assert "Recovered" in outcome.content_markdown
    assert sleeps == [1.0, 4.0]
    assert searcher.call_count == 3


@pytest.mark.asyncio
async def test_search_web_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "key")
    searcher = MagicMock(side_effect=RuntimeError("429 too many requests"))

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    _patch_async_exa(monkeypatch, searcher)

    outcome = await agentic_tools.search_web("query")

    assert outcome.status == "error"
    assert "Exa search failed" in outcome.content_markdown
    assert searcher.call_count == 3


@pytest.mark.asyncio
async def test_search_web_exa_client_uses_bounded_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix 2 (Exa half): the async Exa client is built with a client-side
    timeout <= the search_web tool budget, so a hung endpoint tears the socket
    down before the loop's wait_for fires — and there is no worker thread to
    leak because the sync/to_thread path is gone."""
    monkeypatch.setenv("EXA_API_KEY", "key")
    searcher = MagicMock(return_value=SimpleNamespace(results=[]))
    captured = _patch_async_exa(monkeypatch, searcher)

    outcome = await agentic_tools.search_web("query")

    assert outcome.status == "ok"
    tool_budget = next(
        tool.timeout_s for tool in agentic_tools.build_gap_fill_tools("topic") if tool.name == "search_web"
    )
    assert len(captured) == 1
    assert "timeout" in captured[0]
    assert captured[0]["timeout"] is not None
    assert captured[0]["timeout"] <= tool_budget


@pytest.mark.asyncio
async def test_search_web_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    outcome = await agentic_tools.search_web("query")

    assert outcome.status == "error"
    assert "EXA_API_KEY" in outcome.content_markdown


@pytest.mark.asyncio
async def test_search_web_passes_end_published_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "key")
    searcher = MagicMock(return_value=SimpleNamespace(results=[]))
    _patch_async_exa(monkeypatch, searcher)

    await agentic_tools.search_web("query", end_published_date="2026-01-01")

    assert searcher.call_args.kwargs["end_published_date"] == "2026-01-01"


@pytest.mark.asyncio
async def test_search_news_happy_path_uses_gate_and_semaphore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")
    gate = AsyncMock()
    semaphore_entered = False

    class RecordingSemaphore:
        async def __aenter__(self) -> None:
            nonlocal semaphore_entered
            semaphore_entered = True

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class FakeSdk:
        async def __aenter__(self) -> FakeSdk:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        news = SimpleNamespace(
            search_news=AsyncMock(
                return_value=SimpleNamespace(
                    as_dicts=[
                        {
                            "eng_title": "Article title",
                            "pub_date": "2026-07-16",
                            "source_id": "reuters",
                            "article_url": "https://example.com/story",
                            "summary": "Short summary.",
                        }
                    ]
                )
            )
        )

    monkeypatch.setattr("metaculus_bot.research.providers._ASKNEWS_GLOBAL_SEMAPHORE", RecordingSemaphore())
    monkeypatch.setattr("metaculus_bot.research.providers._asknews_rate_gate", gate)
    monkeypatch.setitem(sys.modules, "asknews_sdk", SimpleNamespace(AsyncAskNewsSDK=lambda **_: FakeSdk()))

    outcome = await agentic_tools.search_news("query")

    assert outcome.status == "ok"
    assert outcome.method == "news"
    assert semaphore_entered is True
    gate.assert_awaited_once()
    assert "Article title" in outcome.content_markdown


class _FakeAskNewsSdk:
    """Async-context AskNews SDK double with a scripted search_news."""

    def __init__(self, search_news_mock: AsyncMock) -> None:
        self.news = SimpleNamespace(search_news=search_news_mock)

    async def __aenter__(self) -> _FakeAskNewsSdk:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _patch_asknews_env(monkeypatch: pytest.MonkeyPatch, search_news_mock: AsyncMock) -> AsyncMock:
    """Wire creds + SDK + rate gate for a _call_asknews_search test; returns the sleep recorder."""
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")
    monkeypatch.setattr("metaculus_bot.research.providers._asknews_rate_gate", AsyncMock())
    monkeypatch.setitem(
        sys.modules, "asknews_sdk", SimpleNamespace(AsyncAskNewsSDK=lambda **_: _FakeAskNewsSdk(search_news_mock))
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)
    return sleep_mock


@pytest.mark.asyncio
async def test_asknews_search_retries_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    search_news = AsyncMock(
        side_effect=[
            RuntimeError("429 too many requests"),
            SimpleNamespace(as_dicts=[{"eng_title": "Recovered article"}]),
        ]
    )
    sleep_mock = _patch_asknews_env(monkeypatch, search_news)

    articles = await agentic_tools._call_asknews_search("query")

    assert len(articles) == 1
    assert search_news.await_count == 2
    # One backoff between the two attempts, on the provider's schedule.
    expected_backoff = agentic_tools.ASKNEWS_BACKOFF_SECS * (10 + 3**1)
    assert [call.args[0] for call in sleep_mock.await_args_list] == [expected_backoff]


@pytest.mark.asyncio
async def test_asknews_search_retries_concurrency_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    search_news = AsyncMock(
        side_effect=[
            RuntimeError("concurrency limit exceeded for plan"),
            SimpleNamespace(as_dicts=[{"eng_title": "Recovered article"}]),
        ]
    )
    sleep_mock = _patch_asknews_env(monkeypatch, search_news)

    articles = await agentic_tools._call_asknews_search("query")

    assert len(articles) == 1
    assert search_news.await_count == 2
    assert sleep_mock.await_count == 1


@pytest.mark.asyncio
async def test_asknews_search_non_retryable_error_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    search_news = AsyncMock(side_effect=RuntimeError("invalid credentials"))
    sleep_mock = _patch_asknews_env(monkeypatch, search_news)

    with pytest.raises(RuntimeError, match="invalid credentials"):
        await agentic_tools._call_asknews_search("query")

    assert search_news.await_count == 1
    assert sleep_mock.await_count == 0


class _FakeAskNewsForbiddenError(Exception):
    """Stand-in for ``asknews_sdk.errors.ForbiddenError`` — matched by class name."""


# ``is_asknews_subscription_error`` keys on the class-name substring, so rename the
# attribute to the SDK's real class name (same trick as tests/test_research_providers.py).
_FakeAskNewsForbiddenError.__name__ = "ForbiddenError"


@pytest.mark.asyncio
async def test_asknews_search_subscription_inactive_raises_on_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """403011 subscription-inactive is PERMANENT, so it costs exactly one attempt.

    It used to be exempted from the fast-fail alongside rate limits, which re-rolled a
    call that can never succeed and burned the whole ``ASKNEWS_MAX_TRIES`` backoff ladder
    out of GAP_FILL_V2_WALL_DEADLINE. The primary provider's ``_is_retryable`` never
    retried it; this asserts the agentic path matches that policy.
    """
    subscription_exc = _FakeAskNewsForbiddenError("403011 - subscription is not currently active")
    assert research_providers.is_asknews_subscription_error(subscription_exc) is True
    search_news = AsyncMock(side_effect=subscription_exc)
    sleep_mock = _patch_asknews_env(monkeypatch, search_news)

    with pytest.raises(_FakeAskNewsForbiddenError, match="403011"):
        await agentic_tools._call_asknews_search("query")

    assert search_news.await_count == 1
    assert sleep_mock.await_count == 0


@pytest.mark.asyncio
async def test_asknews_search_rate_limit_exhausts_retries_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    tries = max(1, int(agentic_tools.ASKNEWS_MAX_TRIES))
    search_news = AsyncMock(side_effect=RuntimeError("429 too many requests"))
    sleep_mock = _patch_asknews_env(monkeypatch, search_news)

    with pytest.raises(RuntimeError, match="429"):
        await agentic_tools._call_asknews_search("query")

    assert search_news.await_count == tries
    assert sleep_mock.await_count == tries - 1


@pytest.mark.asyncio
async def test_search_news_missing_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
    monkeypatch.delenv("ASKNEWS_SECRET", raising=False)

    outcome = await agentic_tools.search_news("query")

    assert outcome.status == "error"
    assert "ASKNEWS_CLIENT_ID" in outcome.content_markdown


@pytest.mark.asyncio
async def test_fetch_direct_success_path_reuses_fetch_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/html"}))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
    monkeypatch.setattr(
        classify,
        "read_body_capped",
        AsyncMock(return_value=b'<html><body><a href="/a">A</a><p>Long body</p></body></html>'),
    )
    monkeypatch.setattr(
        "metaculus_bot.research.fetch_ladder.classify._extract_main_text",
        MagicMock(return_value="Rendered plain body " * 40),
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    outcome = await agentic_tools.fetch("https://example.com/page")

    assert outcome.status == "ok"
    assert outcome.method == "plain"
    assert outcome.links == ["https://example.com/a"]
    assert "Rendered plain body" in outcome.content_markdown
    assert session.calls == [("https://example.com/page", False)]


@pytest.mark.asyncio
async def test_fetch_js_wall_escalates_to_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_direct(
        monkeypatch, _direct("success", text="too short", links=["https://example.com/plain"], escalate_rendered=True)
    )
    rendered_on = _serve_rendered(
        monkeypatch,
        replace(
            _direct("success", text="rendered body", links=["https://example.com/rendered"]),
            route="rendered",
        ),
    )

    outcome = await agentic_tools.fetch(_URL)

    assert rendered_on == [_URL]
    assert outcome.method == "rendered"
    assert outcome.links == ["https://example.com/rendered"]
    assert outcome.content_markdown == "rendered body"


@pytest.mark.asyncio
async def test_fetch_thin_content_escalates_to_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_direct(monkeypatch, _direct("success", text="x" * 100, escalate_rendered=True))
    _serve_rendered(monkeypatch, replace(_direct("success", text="x" * 600), route="rendered"))

    outcome = await agentic_tools.fetch(_URL)

    assert outcome.method == "rendered"
    assert outcome.content_markdown == "x" * 600


@pytest.mark.asyncio
async def test_fetch_scanned_pdf_escalates_to_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PDF with no text layer is the paid reader's one remaining job on this rung."""
    _serve_pdf(monkeypatch, _scanned_pdf())
    read_document = AsyncMock(
        return_value=agentic_tools.ToolOutcome(content_markdown="Extracted PDF content.", method="document")
    )
    monkeypatch.setattr(agentic_tools, "read_document", read_document)

    outcome = await agentic_tools.fetch("https://example.com/file.pdf")

    assert outcome.status == "ok"
    assert outcome.method == "document"
    assert outcome.content_markdown == "Extracted PDF content."
    read_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_document_escalation_generic_ask_contains_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_pdf(monkeypatch, _scanned_pdf())
    read_document = AsyncMock(
        return_value=agentic_tools.ToolOutcome(content_markdown="Extracted PDF content.", method="document")
    )
    monkeypatch.setattr(agentic_tools, "read_document", read_document)

    tools = agentic_tools.build_gap_fill_tools("Will Nauru ratify the treaty?")
    fetch_handler = next(tool.handler for tool in tools if tool.name == "fetch")

    outcome = await fetch_handler(url="https://example.com/file.pdf")

    assert outcome.method == "document"
    read_document.assert_awaited_once_with(
        "https://example.com/file.pdf",
        "Extract the main content relevant to: Will Nauru ratify the treaty?",
        # The free rungs just ran here, so the escalation says so rather than running them again.
        ladder_exhausted=True,
        ctx=None,
    )


@pytest.mark.asyncio
async def test_fetch_pagination_second_call_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _serve_direct(
        monkeypatch,
        _direct("success", text="A" * (agentic_tools._FETCH_WINDOW_CHARS + 5), links=["https://example.com/a"]),
    )

    first = await agentic_tools.fetch(_URL)
    continuation_start = int(first.content_markdown.split("start_char=", 1)[1].split("]", 1)[0])
    second = await agentic_tools.fetch(_URL, start_char=continuation_start)

    assert first.truncated is True
    assert f"[truncated at {continuation_start} of 8005 chars" in first.content_markdown
    assert second.method == "cache"
    assert first.content_markdown[:continuation_start] + second.content_markdown == "A" * 8005
    assert asked == [_URL], "the continuation is served from the shared read cache, with no second request"


@pytest.mark.asyncio
async def test_fetch_direct_textual_branch_strips_allowlisted_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw-text branch runs the same allow-listed tag strip as the Tier-1 CSV path:
    a poll-tracker CSV's styled per-row anchors are markup the driver's result budget
    should not buy, and inequality signs in data cells must survive untouched."""
    csv_body = (
        b"date,pollster,margin\n"
        b"\"8/16 - 8/17, 2026\",<a href='https://poller.example/aug' style='color:#000'>Emerson College</a>,-12.8\n"
        b"note,a < 5 and b > 3,0.0\n"
    )
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/csv"}))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
    monkeypatch.setattr(classify, "read_body_capped", AsyncMock(return_value=csv_body))

    result = await _fetch_direct_only("https://example.com/data.csv")

    assert result.status == "ok"
    assert "Emerson College" in result.text
    assert "<a " not in result.text
    assert "style=" not in result.text
    assert "a < 5 and b > 3" in result.text


def test_extract_links_caps_at_twenty_five() -> None:
    html = "".join(f'<a href="/{index}">link{index}</a>' for index in range(30))

    links = fetch_outcomes._extract_links_from_html(html, "https://example.com/root")

    assert len(links) == 25
    assert links[0] == "https://example.com/0"
    assert links[-1] == "https://example.com/24"


@pytest.mark.asyncio
async def test_fetch_direct_follows_redirect_to_public_url(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(
        _FakeResponse(status=302, headers={"Location": "https://example.com/final"}),
        _FakeResponse(status=200, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
    monkeypatch.setattr(
        classify,
        "read_body_capped",
        AsyncMock(return_value=b"<html><body><p>Final page body</p></body></html>"),
    )
    monkeypatch.setattr(
        "metaculus_bot.research.fetch_ladder.classify._extract_main_text",
        MagicMock(return_value="Final page body " * 40),
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    result = await _fetch_direct_only("https://example.com/start")

    assert result.status == "ok"
    assert result.url == "https://example.com/final"
    assert "Final page body" in result.text
    assert session.calls == [("https://example.com/start", False), ("https://example.com/final", False)]


@pytest.mark.asyncio
async def test_fetch_direct_blocks_redirect_to_non_public_target(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(
        _FakeResponse(status=302, headers={"Location": "http://169.254.169.254/latest/meta-data/"}),
    )

    async def is_public(url: str) -> bool:
        return "169.254.169.254" not in url

    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", is_public)
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)

    result = await _fetch_direct_only("https://example.com/start")

    assert result.status == "blocked"
    assert "non-public redirect target" in result.text
    # The private hop must never be requested.
    assert session.calls == [("https://example.com/start", False)]


@pytest.mark.asyncio
async def test_fetch_direct_caps_redirect_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    hops = http_fetch.MAX_REDIRECTS + 2
    session = _FakeSession(
        *[_FakeResponse(status=302, headers={"Location": f"https://example.com/hop{i}"}) for i in range(hops)]
    )
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)

    result = await _fetch_direct_only("https://example.com/start")

    assert result.status == "error"
    assert result.text == "Redirect limit exceeded."
    assert len(session.calls) == http_fetch.MAX_REDIRECTS + 1


@pytest.mark.asyncio
async def test_fetch_direct_redirect_without_location_is_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(_FakeResponse(status=302, headers={}))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)

    result = await _fetch_direct_only("https://example.com/start")

    assert result.status == "error"
    assert "Malformed redirect" in result.text


@pytest.mark.parametrize(
    "url",
    [
        "https://www.metaculus.com/questions/12345/",
        "https://metaculus.com/q/12345",
        "https://www.metaculus.com:443/questions/12345/",  # port must not bypass the block
        "https://sub.metaculus.com/page",  # subdomain
        "https://competitions.mantic.com/questions/650/",  # the Mantic competition site: same refusal
    ],
)
@pytest.mark.asyncio
async def test_fetch_direct_blocks_metaculus_without_network(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # is_public_http_url is stubbed True so the platform URL clears the SSRF gate
    # (as it would in prod — both sites are public); the real is_metaculus_self_ref
    # then blocks it. _get_session raises if reached, proving no HTTP is attempted.
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    get_session = MagicMock(side_effect=AssertionError("must not open a session for a question-platform URL"))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", get_session)

    result = await _fetch_direct_only(url)

    assert result.status == "blocked"
    # The block text is what the driver reads, so it has to name both hosts it must not fetch.
    assert "metaculus.com" in result.text
    assert "competitions.mantic.com" in result.text
    get_session.assert_not_called()


def test_fetch_description_names_both_platform_hosts() -> None:
    """The driver reads FETCH_DESCRIPTION BEFORE it picks a URL and the block message only after
    the guard refused one, so the two must name the same hosts: a description that dropped one
    would spend fetch steps on question pages the code then refuses (``fetch`` and
    ``read_document`` alike, through the same guard). Literal pins, so the test also
    fails if the constants behind the f-strings are re-pointed at something else."""
    for host in ("metaculus.com", "competitions.mantic.com"):
        assert host in FETCH_DESCRIPTION
        assert host in fetch_outcomes._PLATFORM_FETCH_BLOCK_MSG


def test_fetch_description_redirects_data_endpoints_to_the_briefing() -> None:
    """Item D: the driver scraped FRED, Kalshi and Yahoo Finance 61 times across 23 questions
    (fetch-gap inventory, 2026-09-09) though the briefing already carries them via their APIs, so
    the description tells the driver to cite those sections instead."""
    for source in ("FRED", "Kalshi", "Polymarket", "Yahoo Finance"):
        assert source in FETCH_DESCRIPTION
    assert "financial-data and prediction-market sections" in FETCH_DESCRIPTION


def test_fetch_description_names_available_api_tools() -> None:
    for tool_name in ("fred_series", "yahoo_history", "market_snapshot"):
        assert tool_name in FETCH_DESCRIPTION
    assert "date window" in FETCH_DESCRIPTION
    assert "current prices" in FETCH_DESCRIPTION


@pytest.mark.asyncio
async def test_fetch_direct_blocks_redirect_to_metaculus(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(
        _FakeResponse(status=302, headers={"Location": "https://www.metaculus.com/questions/12345/"}),
    )
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)

    result = await _fetch_direct_only("https://example.com/start")

    assert result.status == "blocked"
    assert result.text == fetch_outcomes._PLATFORM_FETCH_BLOCK_MSG
    # The metaculus hop must never be requested (only the initial URL was GET-ed).
    assert session.calls == [("https://example.com/start", False)]


@pytest.mark.asyncio
async def test_same_host_plain_and_rendered_fetches_serialize(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan §5 politeness: a direct and a rendered fetch to the same host must
    contend on the same per-host Semaphore(1) and never run concurrently."""
    events: list[str] = []
    release_plain = asyncio.Event()
    plain_reading = asyncio.Event()

    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/html"}))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)

    async def blocking_read(resp: object, *, label: str, max_bytes: int = 0) -> bytes:
        events.append("plain_read_started")
        plain_reading.set()
        await release_plain.wait()
        events.append("plain_read_finished")
        return b"<html><body><p>Long body</p></body></html>"

    monkeypatch.setattr(classify, "read_body_capped", blocking_read)
    monkeypatch.setattr(
        "metaculus_bot.research.fetch_ladder.classify._extract_main_text",
        MagicMock(return_value="body text " * 60),
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    class _RecordingManager(FakePlaywrightManager):
        async def __aenter__(self) -> _RecordingManager:
            """Runs strictly after the transport has acquired the host gate."""
            events.append("rendered_started")
            return self

    install_fake_playwright(
        monkeypatch,
        FakePage(html="<html><body><p>rendered body</p></body></html>"),
        pinned=("example.com", "93.184.216.34"),
        manager_cls=_RecordingManager,
    )

    plain_task = asyncio.create_task(_fetch_direct_only("https://example.com/plain-page"))
    await asyncio.wait_for(plain_reading.wait(), timeout=1.0)
    assert events == ["plain_read_started"]  # the direct fetch holds the example.com gate

    rendered_url = "https://example.com/rendered-page"
    rendered_task = asyncio.create_task(
        rungs._rendered_rung(
            rendered_url,
            _direct("js_wall", url=rendered_url),
            agentic_tools._FETCH_HOST_SEMAPHORES,
            context.LadderContext(policy=GAP_FILL_FETCH_POLICY),
        )
    )
    for _ in range(3):
        await asyncio.sleep(0)
    # Rendered must be parked on the shared host gate while the direct fetch holds it.
    assert "rendered_started" not in events

    release_plain.set()
    plain_result = await plain_task
    rendered_result = await rendered_task

    assert plain_result.status == "ok"
    assert rendered_result is not None
    assert rendered_result.status == "success"
    assert events.index("plain_read_finished") < events.index("rendered_started")


@pytest.mark.asyncio
async def test_fetch_ssrf_reject_returns_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=False))

    outcome = await agentic_tools.fetch("http://127.0.0.1")

    assert outcome.status == "blocked"


@pytest.mark.asyncio
async def test_fetch_playwright_missing_degrades_to_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_direct(monkeypatch, _direct("success", text="plain body", escalate_rendered=True))
    _serve_rendered(monkeypatch, None)

    outcome = await agentic_tools.fetch(_URL)

    assert outcome.method == "plain"
    assert outcome.content_markdown == "plain body"


# ---------------------------------------------------------------------------
# No-content fetch outcome (empty-page laundering fix). A 200 OK whose page
# yields ZERO extractable text is NOT a successful fetch: it must carry a
# distinct non-"ok" status so the loop's tier stamping can never mark an unread
# page "fetched" (the companiesmarketcap.com js-wall failure, 2026-07-25).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extracted", [None, "", "   \n  \t "])
@pytest.mark.asyncio
async def test_fetch_direct_empty_extraction_returns_empty_status(
    extracted: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200-OK HTML page that extracts to nothing (or only whitespace) must
    report status="empty", not "ok" — while still flagging escalation so the
    ladder tries the rendered rung next."""
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/html"}))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
    monkeypatch.setattr(classify, "read_body_capped", AsyncMock(return_value=b"<html><body></body></html>"))
    monkeypatch.setattr(
        "metaculus_bot.research.fetch_ladder.classify._extract_main_text", MagicMock(return_value=extracted)
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    result = await _fetch_direct_only("https://example.com/js-wall")

    assert result.status == "empty"
    assert result.escalate_rendered is True
    assert "no extractable text" in result.text.lower()


@pytest.mark.asyncio
async def test_fetch_direct_thin_extraction_is_ok_not_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short-but-real extraction is genuinely read content: status stays "ok"
    (fetched-tierable) even though it's below the escalation floor. Thin != empty
    — demoting real short sources would harm legitimate official statements."""
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/html"}))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
    monkeypatch.setattr(classify, "read_body_capped", AsyncMock(return_value=b"<html><body><p>hi</p></body></html>"))
    monkeypatch.setattr(
        "metaculus_bot.research.fetch_ladder.classify._extract_main_text",
        MagicMock(return_value="Short but real official statement."),
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    result = await _fetch_direct_only("https://example.com/short")

    assert result.status == "ok"
    assert result.escalate_rendered is True  # thin -> escalate, but the content is real
    assert result.text == "Short but real official statement."


@pytest.mark.asyncio
async def test_fetch_direct_honors_declared_charset_on_textual_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A windows-1252 CSV with its charset declared decodes faithfully. The old
    forced-UTF-8 read turned every high byte into U+FFFD and shipped the
    mojibake to the driver as status="ok"."""
    body = "date,séries\n2026-08-01,0.42\n".encode("windows-1252")
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/csv; charset=windows-1252"}))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
    monkeypatch.setattr(classify, "read_body_capped", AsyncMock(return_value=body))

    result = await _fetch_direct_only("https://example.com/data.csv")

    assert result.status == "ok"
    assert "séries" in result.text
    assert "�" not in result.text


@pytest.mark.asyncio
async def test_fetch_direct_refuses_an_undecodable_textual_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A BOM-less UTF-16 body with no declared charset decodes to NUL-interleaved
    garbage — a failed decode, not text we read. It must report "empty" (never
    "ok") and escalate, so the rendered rung's browser sniffing gets a try."""
    body = "date,value\n2026-08-01,0.42\n".encode("utf-16-le")
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/plain"}))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
    monkeypatch.setattr(classify, "read_body_capped", AsyncMock(return_value=body))

    result = await _fetch_direct_only("https://example.com/data.txt")

    assert result.status == "empty"
    assert result.escalate_rendered is True
    assert "could not decode" in result.text


class TestFetchDirectTerminalStatuses:
    """Behavior pins for the non-content exit paths of the plain rung.

    Each branch here decides whether the ladder escalates, retries, or hands the
    driver a refusal, and each was previously only covered transitively through
    ``fetch``. They are pinned directly so the status/method/text triple a
    caller keys on cannot drift.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status", sorted(status for status, verdict in _NON_OK_FETCH_STATUS.items() if verdict == "blocked")
    )
    async def test_anti_bot_status_is_blocked(self, status: int, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _FakeSession(_FakeResponse(status=status, headers={"Content-Type": "text/html"}))
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)

        result = await _fetch_direct_only("https://example.com/gated")

        assert result.status == "blocked"
        assert result.method == "plain"
        assert result.text == f"Fetch blocked with HTTP {status}."
        assert result.url == "https://example.com/gated"

    @pytest.mark.asyncio
    async def test_server_error_status_is_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _FakeSession(_FakeResponse(status=503, headers={"Content-Type": "text/html"}))
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)

        result = await _fetch_direct_only("https://example.com/down")

        assert result.status == "error"
        assert result.text == "Fetch failed with HTTP 503."

    @pytest.mark.asyncio
    async def test_pdf_with_a_text_layer_is_read_locally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The rung the whole change exists for: a declared PDF is decoded, not escalated."""
        _serve_pdf(
            monkeypatch,
            build_text_pdf([["The unemployment rate was 4.1 percent in May 2026, revised from 4.0 percent."]]),
        )

        result = await _fetch_direct_only("https://example.com/report.pdf")

        assert result.status == "ok"
        assert result.method == "pdf_local"
        assert "The unemployment rate was 4.1 percent in May 2026" in result.text
        # Never escalated to the rendered rung: a browser has nothing to add to a decoded PDF,
        # and a short-but-real document is a complete read.
        assert result.escalate_rendered is False

    @pytest.mark.asyncio
    async def test_scanned_pdf_asks_for_read_document(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No text layer is the one PDF shape a model still has to read."""
        _serve_pdf(monkeypatch, _scanned_pdf())

        result = await _fetch_direct_only("https://example.com/scan.pdf")

        assert result.status == "ok"
        assert result.method == "document_needed"
        assert "read_document" in result.text

    @pytest.mark.asyncio
    async def test_pdf_magic_bytes_behind_html_content_type_are_read_locally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mislabeled body is classified off its bytes, not its Content-Type."""
        _serve_pdf(
            monkeypatch,
            build_text_pdf([["Mislabeled as HTML, but its bytes are a readable PDF document."]]),
            content_type="text/html",
        )

        result = await _fetch_direct_only("https://example.com/mislabeled")

        assert result.method == "pdf_local"
        assert "Mislabeled as HTML" in result.text

    @pytest.mark.asyncio
    async def test_oversized_body_is_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/html"}))
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
        monkeypatch.setattr(classify, "read_body_capped", AsyncMock(return_value=None))

        result = await _fetch_direct_only("https://example.com/huge")

        assert result.status == "error"
        assert result.text == "Fetch body exceeded the size limit."

    @pytest.mark.asyncio
    async def test_unsupported_content_type_is_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "application/zip"}))
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
        monkeypatch.setattr(classify, "read_body_capped", AsyncMock(return_value=b"PK\x03\x04payload"))

        result = await _fetch_direct_only("https://example.com/bundle.zip")

        assert result.status == "error"
        assert result.text == "Unsupported content type: application/zip"
        assert result.content_type == "application/zip"

    @pytest.mark.asyncio
    async def test_transport_error_is_reported_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _RaisingSession(_FakeSession):
            def get(self, url: str, *, allow_redirects: bool = False, **kwargs: Any):  # type: ignore[override]
                del kwargs
                self.calls.append((url, allow_redirects))
                raise aiohttp.ClientConnectorError(MagicMock(), OSError("refused"))

        session = _RaisingSession(_FakeResponse(status=200))
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)

        result = await _fetch_direct_only("https://example.com/unreachable")

        assert result.status == "error"
        assert result.text.startswith("Fetch error: ClientConnectorError")


@pytest.mark.asyncio
async def test_fetch_direct_redirect_to_empty_page_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A redirect chain that terminates on a 200-OK empty page is still empty —
    the final hop, not the redirect, decides the outcome."""
    session = _FakeSession(
        _FakeResponse(status=302, headers={"Location": "https://example.com/final"}),
        _FakeResponse(status=200, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
    monkeypatch.setattr(classify, "read_body_capped", AsyncMock(return_value=b"<html><body></body></html>"))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.classify._extract_main_text", MagicMock(return_value=None))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    result = await _fetch_direct_only("https://example.com/start")

    assert result.status == "empty"
    assert result.url == "https://example.com/final"


@pytest.mark.asyncio
async def test_fetch_empty_plain_failed_render_returns_empty_not_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core fix: an empty plain fetch whose rendered rung is unavailable must
    NOT be laundered back into a plain/ok success. It returns a distinct "empty"
    outcome, legible to the driver, that no tier map can promote to "fetched"."""
    url = "https://companiesmarketcap.com/berkshire-hathaway/marketcap/"
    _serve_direct(monkeypatch, {url: _direct("js_wall", url=url, escalate_rendered=True)})
    rendered_on = _serve_rendered(monkeypatch, None)

    outcome = await agentic_tools.fetch(url)

    assert rendered_on == [url], "the browser rung was reached and declined"
    assert outcome.status == "empty"
    assert outcome.method == "empty"
    assert _method_to_tier(outcome.method) is None
    # Legible to a probabilistic consumer: it must read as "nothing was read",
    # not as a thin-but-valid page it can confabulate around.
    assert "was read" in outcome.content_markdown.lower()


@pytest.mark.asyncio
async def test_fetch_empty_plain_and_empty_render_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact prod scenario: plain extracts nothing AND the rendered rung runs
    but also extracts nothing. The outcome stays "empty" rather than falling back to the
    empty plain placeholder as ok."""
    _serve_direct(monkeypatch, _direct("js_wall", escalate_rendered=True))
    empty_dom = rendered_fetch.RenderedPage(url=_URL, content_type="text/html", html="<html><body></body></html>")
    renders: list[dict[str, object]] = []
    monkeypatch.setattr(rungs, "render_page", _fake_render(empty_dom, renders))
    monkeypatch.setattr("metaculus_bot.research.fetch_ladder.classify._extract_main_text", MagicMock(return_value=None))

    outcome = await agentic_tools.fetch(_URL)

    assert [call["url"] for call in renders] == [_URL], "the browser really ran on this page"
    assert outcome.status == "empty"
    assert _method_to_tier(outcome.method) is None


@pytest.mark.asyncio
async def test_fetch_empty_plain_still_escalates_to_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty plain fetch must NOT short-circuit: the ladder still runs the
    rendered rung, and a successful render is returned as a real fetched outcome."""
    _serve_direct(monkeypatch, _direct("js_wall", escalate_rendered=True))
    rendered_on = _serve_rendered(
        monkeypatch, replace(_direct("success", text="real rendered content"), route="rendered")
    )

    outcome = await agentic_tools.fetch(_URL)

    assert outcome.status == "ok"
    assert outcome.method == "rendered"
    assert outcome.content_markdown == "real rendered content"
    assert rendered_on == [_URL]


@pytest.mark.asyncio
async def test_fetch_still_escalates_to_read_document_after_the_rungs_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ladder result only a reader can turn into text must still reach read_document.

    An unreadable document produces this shape, and the archive rung still gets its turn
    on the URL first, so the escalation has to survive a rung having run rather than only firing
    on a bare direct fetch.
    """
    url = "https://example.com/report.pdf"
    _serve_direct(
        monkeypatch,
        {url: _direct("unreadable_document", url=url, content_type="application/pdf")},
    )
    archive_asked: list[str] = []

    async def _no_capture(session: Any, page_url: str, direct: FetchResult, **kwargs: Any) -> None:
        del session, direct, kwargs
        await asyncio.sleep(0)
        archive_asked.append(page_url)

    monkeypatch.setattr(rungs, "_wayback_rung", _no_capture)
    read_document = AsyncMock(
        return_value=agentic_tools.ToolOutcome(content_markdown="Extracted doc content.", method="document")
    )
    monkeypatch.setattr(agentic_tools, "read_document", read_document)

    outcome = await agentic_tools.fetch(url)

    assert archive_asked == [url], "the archive rung had its turn before the escalation"
    assert outcome.method == "document"
    assert outcome.content_markdown == "Extracted doc content."
    read_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_fetch_cannot_earn_fetched_tier_but_real_fetch_can(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end tier check through the loop's real stamping helper: a finding
    whose source URL only ever produced an empty fetch earns NO tier (so a
    discrepancy on it can't supersede the briefing), while a genuinely read page
    earns "fetched". This is the load-bearing invariant."""
    url = "https://companiesmarketcap.com/berkshire-hathaway/marketcap/"
    _serve_direct(monkeypatch, {url: _direct("js_wall", url=url, escalate_rendered=True)})
    _serve_rendered(monkeypatch, None)

    empty_outcome = await agentic_tools.fetch(url)
    empty_tiers = _harvest_verification_tiers("fetch", {"url": url}, empty_outcome)
    assert empty_tiers == {}

    real_outcome = agentic_tools.ToolOutcome(content_markdown="Berkshire market cap is ...", method="plain")
    real_tiers = _harvest_verification_tiers("fetch", {"url": url}, real_outcome)
    assert real_tiers == {"https://companiesmarketcap.com/berkshire-hathaway/marketcap": "fetched"}


def test_empty_method_maps_to_no_tier() -> None:
    """Belt-and-suspenders: even if a future edit passed status=="ok" through, the
    "empty" method itself maps to no tier — the guard is doubly deterministic."""
    assert _method_to_tier("empty") is None
    assert _method_to_tier("plain") == "fetched"


# Verbatim, from the q45191 run's archived transcript
# (backtests/research_archive/latest/45191.json -> gap_fill_v2.transcript, the tool result
# the driver read at step 11 and again from cache at step 23). 304 chars, HTTP 200,
# status="ok", method="rendered": ogimet.com's throttle interstitial standing in for the
# 2022-08-31 daily summary the loop asked for.
_OGIMET_THROTTLE_BODY = (
    "| Professional information about meteorological conditions in the world |  |  | \n"
    "| WEATHER MODEL FORECAST METEOGRAMS INDEXES UNDECODED REPORTS TEXT INFORMATION BUFR "
    "REPORTS GRAPHIC INFORMATION OTHER Advertisements | gsynext: Limit for old data queries "
    "exceeded. Permitted a query per 20 seconds per IP |\n"
)


_OGIMET_URL = "https://www.ogimet.com/summary"


def _ogimet_page(text: str, *, escalate_rendered: bool = False, url: str = _OGIMET_URL) -> dict[str, FetchResult]:
    """One direct read of ``url`` carrying ``text``, in the form ``_serve_direct`` takes."""
    return {url: _direct("success", url=url, text=text, escalate_rendered=escalate_rendered)}


class TestThrottleInterstitialIsNotASuccess:
    """A host that throttles us answers 200 with a sentence instead of the page.

    q45191 (2026-08-10): three parallel ogimet.com fetches tripped that host's one-query-per-
    20-seconds rule, two came back as the interstitial under ``status: ok``, the run cache
    stored it, and the driver's own retry of the same URL was served the stored copy
    (``method: cache``) — so the retry it correctly made could not have succeeded. The
    exact-date reference class it published came to 4 years instead of 6, and the forecast
    under-committed to the state it had already named as the winner.

    The fix belongs on the tool, not in the prompt: that run's own pending lead reads
    "Ogimet rate-limited further historical August 31 queries (2022 and 2023)", so the driver
    had already diagnosed the throttle and still had no way back to the page.
    """

    @pytest.mark.asyncio
    async def test_a_rendered_interstitial_is_throttled_not_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The receipt's own ladder path: a too-thin direct read escalated, and the browser rung
        came back with the interstitial."""
        _serve_direct(monkeypatch, _ogimet_page("nav only", escalate_rendered=True))
        _serve_rendered(
            monkeypatch,
            replace(_direct("success", url=_OGIMET_URL, text=_OGIMET_THROTTLE_BODY), route="rendered"),
        )

        outcome = await agentic_tools.fetch(_OGIMET_URL)

        assert outcome.status == "throttled"
        assert outcome.method == "throttled"
        # The interstitial text itself must not reach the driver as content.
        assert "Limit for old data queries exceeded" not in outcome.content_markdown
        # What the driver is told to do instead: retry later, and do not read the refusal as
        # the fact being unavailable (the null-result reading that cost q44799).
        assert "again later in the run" in outcome.content_markdown
        assert "do NOT read it as evidence that the fact is unavailable" in outcome.content_markdown

    @pytest.mark.asyncio
    async def test_a_plain_interstitial_is_throttled_without_a_rendered_hop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same body arriving on the direct rung, with enough chars not to escalate."""
        _serve_direct(monkeypatch, _ogimet_page(_OGIMET_THROTTLE_BODY))
        rendered_on = _serve_rendered(monkeypatch, None)

        outcome = await agentic_tools.fetch(_OGIMET_URL)

        assert outcome.status == "throttled"
        assert rendered_on == []

    @pytest.mark.asyncio
    async def test_an_interstitial_is_never_cached_so_the_retry_refetches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The half of the fix q45191 turned on: the driver's retry must be a real request."""
        refused = _serve_direct(monkeypatch, _ogimet_page(_OGIMET_THROTTLE_BODY))

        first = await agentic_tools.fetch(_OGIMET_URL)
        assert first.status == "throttled"
        # The host has since let us through: the retry gets the page, not the stored refusal.
        served = _serve_direct(monkeypatch, _ogimet_page("31/08/2022  41.1  Phoenix Sky Harbor"))
        second = await agentic_tools.fetch(_OGIMET_URL)

        assert refused == [_OGIMET_URL]
        assert served == [_OGIMET_URL], "the retry issued its own request rather than replaying the refusal"
        assert second.status == "ok"
        assert second.method == "plain"
        assert "Phoenix Sky Harbor" in second.content_markdown

    @pytest.mark.asyncio
    async def test_a_short_legitimate_page_is_still_a_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under the char cap but with no throttle phrase: real content, cached as before.

        The size half of the rule is what keeps this safe — a one-line official statement is
        the shared HTML classifier deliberately keeps as "ok", and demoting it would
        cost more than the throttle it is trying to catch.
        """
        body = "The Ministry confirmed the vote will be held on 12 October 2026."
        assert len(body) < throttle.FETCH_THROTTLE_PAGE_MAX_CHARS
        url = "https://example.gov/statement"
        served = _serve_direct(monkeypatch, _ogimet_page(body, url=url))

        outcome = await agentic_tools.fetch(url)
        replayed = await agentic_tools.fetch(url)

        assert outcome.status == "ok"
        assert outcome.method == "plain"
        assert outcome.content_markdown == body
        assert replayed.method == "cache"
        assert served == [url]

    @pytest.mark.asyncio
    async def test_a_long_page_about_rate_limits_is_still_a_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The phrase half alone would demote a page that merely discusses throttling."""
        body = "This API returns 429 Too Many Requests once you exceed the rate limit. " * 40
        assert len(body) > throttle.FETCH_THROTTLE_PAGE_MAX_CHARS
        url = "https://example.com/api-docs"
        _serve_direct(monkeypatch, _ogimet_page(body, url=url))

        outcome = await agentic_tools.fetch(url)

        assert outcome.status == "ok"
        assert outcome.method == "plain"

    @pytest.mark.asyncio
    async def test_a_throttled_fetch_earns_no_verification_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end through the loop's real stamping helper: an interstitial can never be
        stamped ``fetched``, so a "correction" resting on it cannot supersede the briefing."""
        _serve_direct(monkeypatch, _ogimet_page(_OGIMET_THROTTLE_BODY))

        outcome = await agentic_tools.fetch(_OGIMET_URL)

        assert _harvest_verification_tiers("fetch", {"url": _OGIMET_URL}, outcome) == {}

    def test_throttled_method_maps_to_no_tier(self) -> None:
        """Belt-and-suspenders, as for "empty": even a leaked ``ok`` status grants nothing."""
        assert _method_to_tier("throttled") is None

    @pytest.mark.asyncio
    async def test_the_marker_names_the_rule_that_fired(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One greppable WARN per throttled fetch: without it the event has no trace at all,
        and ``phrase``/``chars`` are what let a prod fire be graded true or false positive."""
        _serve_direct(monkeypatch, _ogimet_page(_OGIMET_THROTTLE_BODY))

        with caplog.at_level(logging.WARNING, logger=agentic_tools.__name__):
            await agentic_tools.fetch(_OGIMET_URL)

        # 303, not the archived body's 304: `chars` is the stripped length the rule measured.
        assert (
            "AGENTIC_FETCH_THROTTLED: url=https://www.ogimet.com/summary method=plain chars=303 phrase=query per"
            in caplog.text
        )


class TestMatchedThrottlePhrase:
    """The predicate itself, anchored on the receipt and on the shapes it must not claim."""

    def test_the_q45191_body_matches_on_the_hosts_own_wording(self) -> None:
        assert throttle.matched_throttle_phrase(_OGIMET_THROTTLE_BODY) == "query per"

    @pytest.mark.parametrize(
        "body",
        [
            "429 Too Many Requests",
            "Rate limit exceeded. Retry after 30 seconds.",
            "You have made too many requests; please slow down.",
            "Limit: 60 queries per minute per API key.",
        ],
    )
    def test_common_interstitial_wordings_match(self, body: str) -> None:
        assert throttle.matched_throttle_phrase(body) is not None

    @pytest.mark.parametrize(
        "body",
        [
            "",
            "   \n  ",
            # "rate" alone is not the rule: the phrases are anchored on throttle idiom.
            "The unemployment rate fell to 4.1% in August.",
            "Growth is expected to slow down through 2027.",
            "The limit of the sequence exceeded every earlier bound.",
        ],
    )
    def test_ordinary_prose_does_not_trip_the_rule(self, body: str) -> None:
        assert throttle.matched_throttle_phrase(body) is None

    def test_a_body_over_the_cap_is_a_page_whatever_it_says(self) -> None:
        # An interstitial is a sentence. A long body carrying the same words is a page about
        # throttling, and demoting it would discard content we really did read.
        body = "Rate limit exceeded. " * 200
        assert len(body) > throttle.FETCH_THROTTLE_PAGE_MAX_CHARS
        assert throttle.matched_throttle_phrase(body) is None


def _addrinfo(ip: str) -> list[tuple[Any, ...]]:
    """A minimal getaddrinfo return; only sockaddr[0] (the IP string) is read."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


def _addrinfo6(ip: str) -> list[tuple[Any, ...]]:
    """IPv6 getaddrinfo return; sockaddr is (ip, port, flowinfo, scopeid)."""
    return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, 0, 0, 0))]


def test_host_resolver_rule_ipv4_is_bare() -> None:
    assert rendered_fetch._host_resolver_rule("example.com", "93.184.216.34") == (
        "--host-resolver-rules=MAP example.com 93.184.216.34"
    )


def test_host_resolver_rule_ipv6_is_bracketed() -> None:
    # Chromium's rule parser requires IPv6 literals bracketed in the MAP target.
    assert rendered_fetch._host_resolver_rule("example.com", "2606:2800:220:1:248:1893:25c8:1946") == (
        "--host-resolver-rules=MAP example.com [2606:2800:220:1:248:1893:25c8:1946]"
    )


@pytest.mark.asyncio
async def test_resolve_pinned_host_public_ip_returns_host_and_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=_addrinfo("93.184.216.34")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    assert await rendered_fetch.resolve_pinned_host("https://example.com/page") == ("example.com", "93.184.216.34")


@pytest.mark.asyncio
async def test_resolve_pinned_host_private_ip_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=_addrinfo("10.0.0.5")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    assert await rendered_fetch.resolve_pinned_host("https://internal.example.com/page") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_link_local_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Azure IMDS / cloud-metadata address is link-local, so the pin must fail closed."""
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=_addrinfo("169.254.169.254")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    assert await rendered_fetch.resolve_pinned_host("https://rebind.example.com/page") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_rejects_when_any_address_disallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A rebinding host that resolves to BOTH a public and a private IP must be
    # rejected wholesale (same stance as the aiohttp preflight/FilteringResolver).
    mixed = _addrinfo("93.184.216.34") + _addrinfo("127.0.0.1")
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=mixed))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    assert await rendered_fetch.resolve_pinned_host("https://mixed.example.com/page") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_ipv6_public_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=_addrinfo6("2606:2800:220:1:248:1893:25c8:1946")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    assert await rendered_fetch.resolve_pinned_host("https://v6.example.com/page") == (
        "v6.example.com",
        "2606:2800:220:1:248:1893:25c8:1946",
    )


@pytest.mark.asyncio
async def test_resolve_pinned_host_ip_literal_public_pins_to_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    # An IP-literal host needs no DNS; getaddrinfo must not even be consulted.
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(side_effect=AssertionError("getaddrinfo must not run")))

    assert await rendered_fetch.resolve_pinned_host("https://93.184.216.34/page") == ("93.184.216.34", "93.184.216.34")


@pytest.mark.asyncio
async def test_resolve_pinned_host_ip_literal_private_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(side_effect=AssertionError("getaddrinfo must not run")))

    assert await rendered_fetch.resolve_pinned_host("http://127.0.0.1/latest/meta-data/") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_userinfo_and_scheme_fail_closed() -> None:
    # Userinfo defeats hostname trust; non-http(s) schemes are never fetched.
    assert await rendered_fetch.resolve_pinned_host("https://trusted@169.254.169.254/") is None
    assert await rendered_fetch.resolve_pinned_host("ftp://example.com/x") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_dns_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(side_effect=socket.gaierror("no such host")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    assert await rendered_fetch.resolve_pinned_host("https://nxdomain.example.com/page") is None


@pytest.fixture
def _robots_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Say this host's robots.txt does not disallow ``Google-Extended``.

    The paid rung runs a one-request robots pre-check before it spends anything, and it goes
    through the shared direct fetch, so a test about the reader itself would otherwise either dial
    the network or answer the pre-check out of whatever fake body it wired for the document. Its own
    behavior is covered by ``TestUrlContextRobotsPreCheck``.
    """
    monkeypatch.setattr(agentic_tools, "_url_context_robots_skip", AsyncMock(return_value=False))


@pytest.fixture
def _no_local_document(monkeypatch: pytest.MonkeyPatch, _robots_allowed: None) -> None:
    """Make ``read_document``'s acquisition-first ladder hold nothing for the URL.

    ``read_document`` runs the free rungs (cache, plain, rendered) before it pays, so every test
    below about the PAID url_context rung has to say the free ones came back empty — otherwise
    the handler would dial the network instead of reaching the code under test. Requests
    ``_robots_allowed`` for the same reason: the pre-check ahead of the paid call is a request
    too.
    """
    monkeypatch.setattr(agentic_tools, "_acquire_local_document", AsyncMock(return_value=local_document.HeldDocument()))


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_local_document")
async def test_read_document_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))
    monkeypatch.setattr(
        agentic_tools,
        "_run_document_read_sync",
        MagicMock(return_value=("Quoted answer with dates.", 1, ["URL_RETRIEVAL_STATUS_SUCCESS"])),
    )

    outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What does it say?")

    assert outcome.status == "ok"
    assert outcome.method == "document"
    assert outcome.content_markdown == "Quoted answer with dates."


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_local_document")
async def test_read_document_genai_client_uses_bounded_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix 2 (genai half): the genai Client is built with a client-side timeout
    (ms) <= the read_document internal deadline, so a hung endpoint returns the
    to_thread worker instead of stranding it in the shared ThreadPoolExecutor.

    Patches the real ``google.genai.Client`` attribute and uses the real
    ``HttpOptions`` so the asserted timeout is the value that would ship."""
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    captured: dict[str, Any] = {}

    def fake_client(**kwargs: Any) -> Any:
        captured.update(kwargs)
        # Carries a SUCCESSFUL url_context retrieval: read_document now withholds the
        # 'fetched' tier when nothing was actually retrieved, so a metadata-less response
        # would (correctly) come back as an error and this timeout assertion would be
        # asserting on the wrong outcome.
        models = SimpleNamespace(generate_content=lambda **_: _document_response("Quoted answer."))
        return SimpleNamespace(models=models)

    monkeypatch.setattr("google.genai.Client", fake_client)
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What does it say?")

    assert outcome.status == "ok"
    http_options = captured["http_options"]
    # HttpOptions.timeout is in milliseconds; the read_document internal deadline is in seconds.
    assert http_options.timeout is not None
    assert http_options.timeout <= agentic_tools._READ_DOCUMENT_TIMEOUT_S * 1000


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_local_document")
async def test_read_document_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setattr(agentic_tools, "_READ_DOCUMENT_TIMEOUT_S", 0.01)

    async def slow_to_thread(fn, *args):
        await asyncio.sleep(0.05)
        return fn(*args)

    monkeypatch.setattr("asyncio.to_thread", slow_to_thread)
    monkeypatch.setattr(
        agentic_tools,
        "_run_document_read_sync",
        MagicMock(return_value=("late result", 1, ["URL_RETRIEVAL_STATUS_SUCCESS"])),
    )

    outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What does it say?")

    assert outcome.status == "error"
    assert "timed out" in outcome.content_markdown


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_local_document")
async def test_read_document_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What does it say?")

    assert outcome.status == "error"
    assert "GOOGLE_API_KEY" in outcome.content_markdown


class TestReadDocumentRefusesQuestionPlatformPages:
    """``read_document`` refuses a question-platform URL before any rung runs, free or paid.

    The plain and rendered rungs already refuse these through ``_fetch_plain_url_block``, but the
    paid Gemini read dials from Google's address, so until this guard a driver that met a question
    page in a search result could have it read there: on Mantic, the other bots' forecasts and
    comments included. Same message and status the ``fetch`` refusal carries, so the driver reads
    one contract.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "https://competitions.mantic.com/questions/650/",
            "https://www.metaculus.com/questions/12345/some-question/",
        ],
    )
    async def test_a_platform_page_is_blocked_before_any_fetch_or_paid_read(
        self, url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        acquire = AsyncMock(side_effect=AssertionError("the free ladder must not dial a platform page"))
        monkeypatch.setattr(agentic_tools, "_acquire_local_document", acquire)
        reader = _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document(url, "what do the other forecasters say?")

        assert outcome.status == "blocked"
        assert outcome.content_markdown == fetch_outcomes._PLATFORM_FETCH_BLOCK_MSG
        assert outcome.method == "plain"
        acquire.assert_not_awaited()
        reader.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_refusal_is_the_one_fetch_gives(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One contract for the driver: the same URL refused by either tool reads identically."""
        url = "https://competitions.mantic.com/questions/650/"
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        _no_paid_reader(monkeypatch)

        assert await agentic_tools.read_document(url, "anything") == await agentic_tools.fetch(url)

    @staticmethod
    def _redirecting_onto_the_platform(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
        """A public URL whose host 3xxes onto the competition site, served through the real plain rung."""
        session = _FakeSession(
            _FakeResponse(status=302, headers={"Location": "https://competitions.mantic.com/questions/650/"}),
        )
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
        return session

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_redirect_onto_a_platform_page_is_refused_before_the_paid_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The supplied URL clears the guard; where it LEADS does not. The free ladder refuses the hop,
        and that refusal has to reach the paid rung too: Gemini dials from Google's address and would
        follow the same redirect onto the page the guard exists to refuse, billing a read for it. The
        sibling ladder closes the same hop (``rungs._url_context_rung_applies``)."""
        session = self._redirecting_onto_the_platform(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        reader = _no_paid_reader(monkeypatch)
        rendered_on = _serve_rendered(monkeypatch, None)

        outcome = await agentic_tools.read_document("https://t.co/crucible650", "what do the other forecasters say?")

        assert outcome.status == "blocked"
        assert outcome.content_markdown == fetch_outcomes._PLATFORM_FETCH_BLOCK_MSG
        assert outcome.method == "plain"
        assert session.calls == [("https://t.co/crucible650", False)], "the platform hop itself is never dialed"
        reader.assert_not_called()
        assert rendered_on == []

    @pytest.mark.asyncio
    async def test_the_redirect_refusal_is_the_one_fetch_gives(self, monkeypatch: pytest.MonkeyPatch) -> None:
        url = "https://t.co/crucible650"
        self._redirecting_onto_the_platform(monkeypatch)
        _no_paid_reader(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")

        read = await agentic_tools.read_document(url, "anything")
        self._redirecting_onto_the_platform(monkeypatch)

        assert read == await agentic_tools.fetch(url)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_no_local_document")
    async def test_the_rest_of_the_platforms_domain_still_reaches_the_reader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``blog.mantic.com`` is an outside source: only the competition host is the platform."""
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))
        reader = MagicMock(return_value=("Quoted answer.", 1, ["URL_RETRIEVAL_STATUS_SUCCESS"]))
        monkeypatch.setattr(agentic_tools, "_run_document_read_sync", reader)

        outcome = await agentic_tools.read_document("https://blog.mantic.com/crucible-rules", "what are the rules?")

        assert outcome.status == "ok"
        assert outcome.method == "document"
        reader.assert_called_once()


def _document_response(text: str, *statuses: str) -> Any:
    """A fake Gemini response with the given text and url_context retrieval statuses.

    Defaults to one SUCCESS entry (a genuine document read). Shape mirrors the typed SDK
    models ``extract_url_context_telemetry`` reads: ``candidates[0].url_context_metadata
    .url_metadata[i].url_retrieval_status`` / ``.retrieved_url``.
    """
    statuses = statuses or ("URL_RETRIEVAL_STATUS_SUCCESS",)
    url_metadata = [
        SimpleNamespace(url_retrieval_status=status, retrieved_url=f"https://example.com/doc{i}")
        for i, status in enumerate(statuses)
    ]
    candidate = SimpleNamespace(url_context_metadata=SimpleNamespace(url_metadata=url_metadata))
    return SimpleNamespace(text=text, candidates=[candidate])


class TestReadDocumentRequiresRealRetrieval:
    """A ``document`` outcome earns the ``fetched`` tier, so it must be a real read.

    ``method="document"`` maps to ``fetched`` (``loop._METHOD_TO_TIER``), and only a
    ``fetched`` discrepancy enters the SUPERSEDE block that instructs every forecaster to
    override the briefing (``artifact.render_findings``). Gemini answers fluently from
    parametric memory when every url_context retrieval failed — the Q38195 failure mode —
    and the quote check cannot catch it here (WARN-only for this tool by design). So the
    retrieval count is the guard.
    """

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_no_local_document")
    async def test_all_retrievals_failed_withholds_the_fetched_tier(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        # Non-empty, confident-looking text plus a FAILED retrieval: exactly the shape
        # that used to be stamped `fetched` and could supersede the briefing.
        monkeypatch.setattr(
            "google.genai.Client",
            lambda **_: SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **_kw: _document_response(
                        "The filing states revenue of $4.2B.", "URL_RETRIEVAL_STATUS_ERROR"
                    )
                )
            ),
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        with caplog.at_level(logging.WARNING, logger=agentic_tools.__name__):
            outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What is revenue?")

        assert outcome.status != "ok", "an unretrieved document must not come back as a successful read"
        assert "The filing states revenue of $4.2B." not in outcome.content_markdown, (
            "the ungrounded answer text must not reach the driver as document content"
        )
        assert "AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED" in caplog.text, (
            "the suppression must be greppable in the archived run logs, mirroring GEMINI_UNGROUNDED_SUPPRESSED"
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_no_local_document")
    async def test_no_url_context_metadata_at_all_withholds_the_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # url_context never reported back (tool didn't run / SDK attached nothing). Zero
        # successful retrievals either way, so the tier is withheld the same.
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            "google.genai.Client",
            lambda **_: SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **_kw: SimpleNamespace(text="Confident recall.", candidates=[])
                )
            ),
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What is revenue?")
        assert outcome.status != "ok"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_no_local_document")
    async def test_one_success_among_failures_still_counts_as_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The guard is "did ANY retrieval land", matching gemini_search's grounded-chunk
        # floor. A partially-failed multi-URL read still rests on real retrieved content.
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            "google.genai.Client",
            lambda **_: SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **_kw: _document_response(
                        "Quoted from the filing.",
                        "URL_RETRIEVAL_STATUS_ERROR",
                        "URL_RETRIEVAL_STATUS_SUCCESS",
                    )
                )
            ),
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What is revenue?")
        assert outcome.status == "ok"
        assert outcome.method == "document"
        assert outcome.content_markdown == "Quoted from the filing."

    def test_document_method_still_maps_to_the_fetched_tier(self) -> None:
        # If this ever stopped being true the guard above would be defending nothing;
        # pin the coupling that makes it load-bearing.
        assert _method_to_tier("document") == "fetched"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_no_local_document")
    async def test_the_suppression_warn_names_the_retrieval_statuses(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Zero successes is the same number for several different problems.

        A refused fetch, a retrieval that timed out and a url_context tool that never ran
        all read as ``n_url_success == 0``, and the run log used to carry only the URL. The
        status names are what separate "this host blocked us" from "the tool did not fire".
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            "google.genai.Client",
            lambda **_: SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **_kw: _document_response(
                        "Confident recall.",
                        "URL_RETRIEVAL_STATUS_ERROR",
                        "URL_RETRIEVAL_STATUS_UNSAFE",
                    )
                )
            ),
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        with caplog.at_level(logging.WARNING, logger=agentic_tools.__name__):
            outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What is revenue?")

        assert outcome.status != "ok"
        assert (
            "AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED: url=https://example.com/file.pdf "
            "statuses=URL_RETRIEVAL_STATUS_ERROR,URL_RETRIEVAL_STATUS_UNSAFE"
        ) in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_no_local_document")
    async def test_the_suppression_warn_reads_none_when_nothing_was_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No url_metadata entry at all: the tool never reported back, which is its own case."""
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            "google.genai.Client",
            lambda **_: SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **_kw: SimpleNamespace(text="Confident recall.", candidates=[])
                )
            ),
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        with caplog.at_level(logging.WARNING, logger=agentic_tools.__name__):
            outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What is revenue?")

        assert outcome.status != "ok"
        assert "statuses=none" in caplog.text

    def test_the_backend_returns_the_status_names_beside_the_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 3-tuple is the seam that feeds the WARN above; pin its shape and order."""
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            "google.genai.Client",
            lambda **_: SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **_kw: _document_response(
                        "Quoted from the filing.",
                        "URL_RETRIEVAL_STATUS_ERROR",
                        "URL_RETRIEVAL_STATUS_SUCCESS",
                    )
                )
            ),
        )

        text, n_success, statuses = tool_backends._run_document_read_sync("https://example.com/f.pdf", "ask")

        assert text == "Quoted from the filing."
        assert n_success == 1
        assert statuses == ["URL_RETRIEVAL_STATUS_ERROR", "URL_RETRIEVAL_STATUS_SUCCESS"]


class TestReadDocumentClientConfig:
    """The reader's google-genai client: bounded retries, explicit thinking, logged spend.

    A bare ``genai.Client`` retries NOTHING (``retry_args(None)`` is
    ``stop_after_attempt(1)``), which is how two production reads died outright on a
    ``503 UNAVAILABLE``. The retry has to fit inside the existing HTTP budget rather than
    extend it, because this call runs in a ``to_thread`` worker that ``read_document``'s
    ``asyncio.wait_for`` cannot cancel — a longer worst case here means a pooled thread
    pinned for longer.
    """

    @staticmethod
    def _capture_client_kwargs(monkeypatch: pytest.MonkeyPatch, response: Any) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def fake_client(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_kw: response))

        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr("google.genai.Client", fake_client)
        return captured

    def test_retry_ladder_is_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = self._capture_client_kwargs(monkeypatch, _document_response("Quoted answer."))

        tool_backends._run_document_read_sync("https://example.com/f.pdf", "ask")

        retry_options = captured["http_options"].retry_options
        assert retry_options is not None, "without retry_options the SDK stops after one attempt"
        assert retry_options.attempts == GAP_FILL_V2_READER_HTTP_ATTEMPTS
        assert 503 in (retry_options.http_status_codes or []), "503 UNAVAILABLE is the failure this recovers"

    def test_every_attempt_plus_its_backoff_fits_the_existing_budget(self) -> None:
        """The arithmetic, pinned: the retries must not lengthen the in-thread worst case.

        ``_READ_DOCUMENT_HTTP_TIMEOUT_MS`` is the whole in-thread HTTP budget and stays where it
        was; the attempts divide it after the worst-case backoff sleeps are set aside. Today:
        2 x 26_500 + 2_000 = 55_000ms, exactly the previous single-attempt ceiling.

        What that 55 s no longer sits under is ``read_document``'s own wait, which became
        ``min(60, 65 - acquisition_elapsed)`` when the free local ladder landed ahead of the paid
        read — so 40 s at the 25 s acquisition cap. The second assertion is therefore against the
        60 s constant only (the no-acquisition case), and the money-relevant invariant is the
        separate test below.
        """
        worst_case_ms = GAP_FILL_V2_READER_HTTP_ATTEMPTS * tool_backends._READ_DOCUMENT_HTTP_PER_ATTEMPT_TIMEOUT_MS
        worst_case_ms += 1000 * gemini_retry_sleep_allowance_s(GAP_FILL_V2_READER_HTTP_ATTEMPTS)

        assert worst_case_ms <= tool_backends._READ_DOCUMENT_HTTP_TIMEOUT_MS
        assert worst_case_ms <= agentic_tools._READ_DOCUMENT_TIMEOUT_S * 1000

    def test_no_billed_attempt_is_dispatched_after_the_shortest_wait_fires(self) -> None:
        """The invariant that actually costs money, on the worst case for the wait.

        ``asyncio.wait_for`` cancels the coroutine but not the ``to_thread`` worker, so past ~10 s
        of local acquisition the worker outlives the wait by up to 15 s and finishes a call whose
        answer is discarded. That is one billed call at worst. Dispatching a NEW request after we
        stopped waiting would be a second, invisible to the ``GEMINI_USAGE`` line the read logs on
        return — so the last attempt has to START inside the shortest wait the handover can
        produce (65 - 25 = 40 s), which is what this pins: the attempts before the last one, plus
        every backoff sleep, fit in that window.
        """
        dispatch_of_last_attempt_ms = (
            GAP_FILL_V2_READER_HTTP_ATTEMPTS - 1
        ) * tool_backends._READ_DOCUMENT_HTTP_PER_ATTEMPT_TIMEOUT_MS
        dispatch_of_last_attempt_ms += 1000 * gemini_retry_sleep_allowance_s(GAP_FILL_V2_READER_HTTP_ATTEMPTS)
        shortest_wait_ms = 1000 * (agentic_tools._READ_DOCUMENT_TOTAL_BUDGET_S - agentic_tools._LOCAL_DOCUMENT_BUDGET_S)

        assert dispatch_of_last_attempt_ms <= shortest_wait_ms

    def test_thinking_level_is_set_explicitly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Quoting a fetched document back is the least reasoning-heavy Gemini call we make,
        and an unset level means the model's own default (HIGH on the Gemini 3 flash line)."""
        captured: dict[str, Any] = {}

        def fake_client(**_kwargs: Any) -> Any:
            def generate_content(**kwargs: Any) -> Any:
                captured.update(kwargs)
                return _document_response("Quoted answer.")

            return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr("google.genai.Client", fake_client)

        tool_backends._run_document_read_sync("https://example.com/f.pdf", "ask")

        thinking_config = captured["config"].thinking_config
        assert thinking_config is not None
        assert thinking_config.thinking_level == genai_types.ThinkingLevel(GAP_FILL_V2_READER_THINKING_LEVEL.upper())

    def test_token_spend_is_logged(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """This call bills the operator's personal AI Studio key and used to record nothing."""
        response = _document_response("Quoted answer.")
        response.usage_metadata = SimpleNamespace(
            prompt_token_count=8000,
            tool_use_prompt_token_count=None,
            candidates_token_count=300,
            thoughts_token_count=120,
            total_token_count=8420,
        )
        self._capture_client_kwargs(monkeypatch, response)

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.gemini_usage"):
            tool_backends._run_document_read_sync("https://example.com/f.pdf", "ask")

        assert (
            f"GEMINI_USAGE: role=read_document model={GAP_FILL_V2_READER_MODEL} prompt_tokens=8000 "
            "tool_use_prompt_tokens=n/a candidates_tokens=300 thoughts_tokens=120 total_tokens=8420 "
            "search_queries=0"
        ) in caplog.text
        assert "question=" not in caplog.text, "the document reader holds no question id to carry"


# ---------------------------------------------------------------------------
# The local-document rung (2026-09-03). A PDF is decoded and passage-selected from bytes we
# already hold, and the paid Gemini url_context reader is spent only on a document we cannot
# read at all: measured over the 2026 summer season, that was 191 reader calls, nine documents
# over 100k tokens carried 67% of the retrieved tokens, and on the one file where both routes
# were tried local pypdf pulled 833,450 chars in 5.3 s while the paid read returned nothing.
# ---------------------------------------------------------------------------


def _long_pdf() -> bytes:
    """A PDF whose text runs past one fetch window, so pagination is exercised for real."""
    line = "The reported unemployment rate for May 2026 was 4.1 percent, revised from 4.0 percent. "
    return build_text_pdf([[line] * 60, [line] * 60])


def _no_paid_reader(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the paid url_context backend with a spy that must not be called."""
    reader = MagicMock(side_effect=AssertionError("the paid reader must not run for a document we hold"))
    monkeypatch.setattr(agentic_tools, "_run_document_read_sync", reader)
    return reader


class TestLocalPdfRung:
    @pytest.mark.asyncio
    async def test_a_pdf_with_a_text_layer_is_served_locally_and_paginates(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        read_body = _serve_pdf(monkeypatch, _long_pdf())
        read_document = AsyncMock()
        monkeypatch.setattr(agentic_tools, "read_document", read_document)

        with caplog.at_level(logging.INFO, logger=local_document.__name__):
            first = await agentic_tools.fetch("https://example.gov/report.pdf")

        assert first.method == "pdf_local"
        assert _method_to_tier(first.method) == "fetched", "we decoded the bytes the host served"
        assert "unemployment rate for May 2026" in first.content_markdown
        assert first.truncated is True, "a document past one window paginates like a long page"
        read_document.assert_not_awaited()
        assert "AGENTIC_FETCH_LOCAL_DOC: url=https://example.gov/report.pdf method=pdf_local" in caplog.text
        assert "pages=2 passages=n/a" in caplog.text, "a pdf_local fetch serves the text and selects nothing"

        # The continuation is served from the run cache: no second request, no second parse.
        second = await agentic_tools.fetch("https://example.gov/report.pdf", agentic_tools._FETCH_WINDOW_CHARS)
        assert second.method == "cache"
        assert read_body.await_count == 1

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_scanned_pdfs_escalation_neither_refetches_nor_reparses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The parse that proved there is no text layer is held, so escalating costs one request."""
        read_body = _serve_pdf(monkeypatch, _scanned_pdf())
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("Model read.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        outcome = await agentic_tools.fetch("https://example.gov/scan.pdf")

        assert outcome.method == "document"
        assert read_body.await_count == 1, "the escalation reuses the held parse"

    @pytest.mark.asyncio
    async def test_a_document_over_the_byte_cap_reports_rather_than_escalating(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Too big to read locally is also too big to be worth having a model retrieve."""
        session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "application/pdf"}))
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
        monkeypatch.setattr(classify, "read_body_capped", AsyncMock(return_value=None))
        reader = _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.fetch("https://example.gov/huge.pdf")

        assert outcome.status == "error"
        assert outcome.method == "oversize_document"
        assert _method_to_tier(outcome.method) is None, "nothing was read, so no tier"
        assert "too large to read" in outcome.content_markdown.lower()
        reader.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_declared_pdf_body_is_read_under_the_document_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 6.7 MB report local extraction reads in 5.3 s is over the ordinary page cap."""
        read_body = _serve_pdf(monkeypatch, build_text_pdf([["Long enough to count as a real text layer here."]]))

        await _fetch_direct_only("https://example.gov/report.pdf")

        assert read_body.await_args is not None
        assert read_body.await_args.kwargs["max_bytes"] == DOCUMENT_TEXT_PDF_MAX_BYTES

    @pytest.mark.asyncio
    async def test_a_partial_read_says_so_in_the_text_it_serves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A truncated PDF served as ``pdf_local`` must not read as the whole document.

        Extraction stops at the page cap or the time budget and reports which in ``truncated_by``.
        The digest header discloses it; this route serves the joined page text with no header at
        all, and ``FETCH_DESCRIPTION`` tells the driver "A PDF is read here, in full text" — so a
        driver that pages to the end sees ``truncated=False`` and can state an absence over pages
        nobody read. A 405-page report served 22,290 chars without mentioning the 5 it skipped.
        """
        truncated = extract_pdf_text(_long_pdf(), max_pages=1, max_seconds=5.0)
        assert truncated.truncated_by == "pages", "the fixture has to be a genuinely partial read"
        _serve_pdf(monkeypatch, _long_pdf())
        monkeypatch.setattr(classify, "extract_pdf_text", MagicMock(return_value=truncated))

        result = await _fetch_direct_only("https://example.gov/report.pdf")

        note = "[Partial document read: 2 pages; stopped at the 1-page read cap]"
        assert result.method == local_document.PDF_LOCAL_METHOD
        assert result.text.startswith(note)
        assert "unemployment rate" in result.text, "the disclosure leads the text, it does not replace it"
        # The other writer of that text into the run cache — a later read_document digests it flat.
        assert local_document.held_pdf(truncated).text.startswith(note)
        # A complete read gets no note at all, so ordinary output is unchanged.
        whole = extract_pdf_text(_long_pdf(), max_pages=400, max_seconds=5.0)
        assert whole.truncated_by == ""
        assert not local_document.held_pdf(whole).text.startswith("[Partial")

    @pytest.mark.asyncio
    async def test_the_pypdf_gate_is_shared_with_the_tier_1_rung(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One process-wide parse gate, not one per rung.

        pypdf is pure Python, so the two rungs' parses contend for the same GIL (6 concurrent
        parses of a 220-page document took 10.2 s against 1.66 s solo) and each parse's
        ``max_seconds`` is wall-clock, so unbounded contention truncates reads on a budget
        concurrency ate rather than document size. A gate private to this module would bound
        neither.
        """
        _serve_pdf(monkeypatch, _long_pdf())
        gate = http_fetch.pdf_parse_semaphore()
        await gate.acquire()
        await gate.acquire()

        parse = asyncio.create_task(_fetch_direct_only("https://example.gov/queued.pdf"))
        for _ in range(3):
            await asyncio.sleep(0)
        assert not parse.done(), "both slots are held, so the v2 parse must be queued behind them"

        gate.release()
        result = await parse
        gate.release()

        assert result.method == "pdf_local"

    def test_the_held_parse_cache_is_run_scoped_state_the_suite_resets(self) -> None:
        # The autouse fixture calls exactly this, which is what keeps one test's held document
        # out of the next one's ladder.
        pdf = extract_pdf_text(_scanned_pdf(), max_pages=5, max_seconds=5.0)
        document_cache.cache_document("https://example.gov/a.pdf", pdf)
        assert document_cache.cached_document("https://example.gov/a.pdf") is not None

        document_cache.clear_document_cache()
        assert document_cache.cached_document("https://example.gov/a.pdf") is None


class TestReadDocumentAcquiresBeforePaying:
    """``read_document`` answers from the page's own text wherever it can get it.

    Its old shape sent every ask straight to a paid Gemini ``url_context`` call. Now the free
    rungs run first and their text is answered with a deterministic BM25 passage digest; the
    paid read is what a host that refuses us, or a document with no text layer, still needs.
    """

    @pytest.mark.asyncio
    async def test_a_fetchable_page_is_digested_with_no_model_call(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        page_text = (
            "Background on the tracker.\n\n"
            "The unemployment rate stood at 4.1 percent in May 2026.\n\n"
            "Unrelated methodology notes about seasonal adjustment.\n\n"
        ) * 3
        url = "https://example.gov/tracker"
        _serve_direct(monkeypatch, {url: _direct("success", url=url, text=page_text)})
        reader = _no_paid_reader(monkeypatch)
        # No Google key at all: the local digest is free and must not depend on one.
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with caplog.at_level(logging.INFO, logger=local_document.__name__):
            outcome = await agentic_tools.read_document(url, "unemployment rate May 2026")

        assert outcome.method == "digest_local"
        assert _method_to_tier(outcome.method) == "fetched"
        assert "4.1 percent in May 2026" in outcome.content_markdown
        assert "Most relevant passages for: unemployment rate May 2026" in outcome.content_markdown
        assert "[passage]" in outcome.content_markdown, "a page has no page numbers to claim"
        reader.assert_not_called()
        assert "method=digest_local" in caplog.text
        assert "pages=n/a" in caplog.text

    @pytest.mark.asyncio
    async def test_a_pdf_digest_carries_page_numbers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Page attribution is why a held parse beats the flat text: a cited page must be true."""
        _serve_pdf(monkeypatch, _long_pdf())
        _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document("https://example.gov/report.pdf", "unemployment rate revised")

        assert outcome.method == "digest_local"
        assert "[p.1]" in outcome.content_markdown or "[p.2]" in outcome.content_markdown

    @pytest.mark.asyncio
    async def test_text_a_fetch_already_read_is_digested_without_refetching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        read_body = _serve_pdf(monkeypatch, _long_pdf())
        await agentic_tools.fetch("https://example.gov/report.pdf")
        _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document("https://example.gov/report.pdf", "unemployment rate revised")

        assert outcome.method == "digest_local"
        assert read_body.await_count == 1, "one request served both tools"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_blocked_url_still_reaches_the_paid_reader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reader's remaining job: a host our own client cannot read from at all. A real 403
        result, so it would earn the impersonated retry; the module fixture declines that retry,
        which is the DataDome-fronted shape (sagaftra.org refused both clients on 2026-09-04)."""
        url = "https://sagaftra.org/contract"
        _serve_direct(monkeypatch, {url: _direct("blocked", url=url, http_status=403)})
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools,
            "_run_document_read_sync",
            MagicMock(return_value=("The contract states a 3.5 percent increase.", 1, ["SUCCESS"])),
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        outcome = await agentic_tools.read_document(url, "what increase is stated?")

        assert outcome.method == "document"
        assert outcome.content_markdown == "The contract states a 3.5 percent increase."

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_throttle_interstitial_is_never_digested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """q45191 again: a rate-limit sentence under HTTP 200 is not the document.

        Digesting it would put the host's refusal in front of the driver as the page's content.
        The paid reader dials from Gemini's address rather than ours, so it is the right next
        rung for exactly this case.
        """
        interstitial = "Limit for old data queries exceeded. Permitted a query per 20 seconds per IP"
        _serve_direct(monkeypatch, _ogimet_page(interstitial, escalate_rendered=True))
        # The shared ladder marks the interstitial terminal; read_document still uses its paid reader fallback.
        rendered_on = _serve_rendered(monkeypatch, None)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("Model read.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        outcome = await agentic_tools.read_document(_OGIMET_URL, "the 2022-08-31 maximum")

        assert rendered_on == []
        assert outcome.method == "document"
        assert "Limit for old data queries" not in outcome.content_markdown

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_an_image_is_not_digested_from_its_own_placeholder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``document_needed`` result is "ok" and carries our own instruction as its text.

        Digesting that would answer the ask out of the sentence telling the driver to call this
        very tool. An image also ends the free ladder: no browser reads one either, so the rung
        must not spend a Chromium launch to find that out.
        """
        session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "image/png"}))
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
        monkeypatch.setattr(classify, "read_body_capped", AsyncMock(return_value=_static_png()))
        rendered_on = _serve_rendered(monkeypatch, None)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        document_reader = MagicMock(return_value=("The chart shows 41.", 1, ["SUCCESS"]))
        monkeypatch.setattr(agentic_tools, "_run_document_read_sync", document_reader)
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        outcome = await agentic_tools.read_document("https://example.gov/chart.png", "what does the chart show?")

        assert outcome.status == "ok"
        assert outcome.method == "image_local"
        assert len(outcome.image_views) == 1
        document_reader.assert_not_called()
        assert rendered_on == []

    @pytest.mark.asyncio
    async def test_an_oversize_document_is_reported_rather_than_paid_for(self, monkeypatch: pytest.MonkeyPatch) -> None:
        url = "https://example.gov/huge.pdf"
        _serve_direct(
            monkeypatch,
            {
                url: _direct(
                    "error", url=url, reason="oversize_document", http_status=200, content_type="application/pdf"
                )
            },
        )
        reader = _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document(url, "anything")

        assert outcome.method == "oversize_document"
        assert "too large to read" in outcome.content_markdown.lower()
        reader.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_js_walled_page_is_rescued_by_the_browser_and_digested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rung read_document exists for: a page whose text only appears after JavaScript.

        The plain rung comes back with a shell, Chromium renders the real page, and the ask is
        answered from its text for free. Untested until 2026-09-03 — deleting the rendered block
        from the ladder left the whole suite green.
        """
        url = "https://example.gov/tracker"
        _serve_direct(monkeypatch, {url: _direct("success", url=url, text="Loading…", escalate_rendered=True)})
        rendered_text = (
            "Weekly tracker.\n\nThe unemployment rate stood at 4.1 percent in May 2026.\n\n"
            "Methodology notes about seasonal adjustment follow.\n\n"
        ) * 3
        _serve_rendered(monkeypatch, replace(_direct("success", url=url, text=rendered_text), route="rendered"))
        reader = _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document(url, "unemployment rate May 2026")

        assert outcome.method == "digest_local"
        assert _method_to_tier(outcome.method) == "fetched", "the browser read the host's own bytes"
        assert "4.1 percent in May 2026" in outcome.content_markdown
        reader.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_subfloor_chrome_no_passage_matched_reaches_the_paid_reader(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A JavaScript shell the browser could not rescue must not be digested (F38/D5).

        The ladder holds the shell's navigation chrome, which is under the same content floor
        ``fetch`` escalates on, and no passage of it matches the ask. Digesting it stamped an
        unread page ``fetched`` — the tier that supersedes the briefing — while the tool
        description tells the driver a zero-passage digest means the document does not discuss
        what was asked. The paid reader, which dials from Gemini's address, is the right rung.
        """
        url = "https://manifold.markets/q/some-market"
        chrome = "Home | Markets | Browse | Related questions | Sign in | Newsletter | About | Terms"
        _serve_direct(monkeypatch, {url: _direct("success", url=url, text=chrome, escalate_rendered=True)})
        _serve_rendered(monkeypatch, None)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("Model read.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        with caplog.at_level(logging.INFO, logger=local_document.__name__):
            outcome = await agentic_tools.read_document(url, "what unemployment rate did the department report for May")

        assert outcome.method == "document"
        assert "Related questions" not in outcome.content_markdown
        assert "AGENTIC_FETCH_LOCAL_DOC" not in caplog.text, (
            "the local-read marker must fire only where a digest is actually served"
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_short_page_that_answers_the_ask_is_still_digested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The floor alone would discard real short sources, so the match is the other half.

        ``fetch`` serves a thin-but-real page as a success by design (a one-line official
        statement), and here the same text answers the ask, so the free digest stands.
        """
        url = "https://example.gov/statement"
        statement = "Statement: the unemployment rate stood at 4.1 percent in May 2026."
        _serve_direct(monkeypatch, {url: _direct("success", url=url, text=statement, escalate_rendered=True)})
        _serve_rendered(monkeypatch, None)
        reader = _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document(url, "unemployment rate May 2026")

        assert outcome.method == "digest_local"
        assert "4.1 percent" in outcome.content_markdown
        reader.assert_not_called()


class TestTheDocumentedEscalationDoesNotRepeatItself:
    """``fetch`` then ``read_document`` on the same URL is the driver's documented path.

    ``READ_DOCUMENT_DESCRIPTION`` tells the driver to call it "for a URL where fetch returned
    status=blocked/js_wall/error", and ``fetch`` auto-escalates its own document results, so this
    population is the main path rather than an edge. Image bytes and empty render results are
    cached, avoiding repeated acquisition. A blocked, errored or throttled GET stays
    re-requestable because the driver is told to retry those and 429 is in the retryable block
    set.
    """

    @staticmethod
    def _wire_launch_counting_playwright(monkeypatch: pytest.MonkeyPatch) -> FakeChromium:
        """A Chromium that renders every page to an empty DOM; the returned launcher counts launches."""
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.classify._extract_main_text", MagicMock(return_value="")
        )
        return install_fake_playwright(
            monkeypatch, FakePage(html="<html><body></body></html>"), pinned=("example.gov", "93.184.216.34")
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_an_image_delivery_issues_one_request_not_two(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fetched image retains its body and delivers normalized pixels without a second GET."""
        url = "https://example.gov/chart.png"
        session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "image/png"}))
        read_body = AsyncMock(return_value=_static_png())
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
        monkeypatch.setattr(classify, "read_body_capped", read_body)
        rendered_on = _serve_rendered(monkeypatch, None)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("The chart shows 41.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        outcome = await agentic_tools.fetch(url)

        assert outcome.method == "image_local"
        assert len(outcome.image_views) == 1
        assert [requested for requested, _ in session.calls].count(url) == 1, (
            "pixel delivery must not re-GET a URL the direct rung just acquired"
        )
        assert read_body.await_count == 1, "the image body is retained once for view_image"
        assert rendered_on == []  # no browser reads an image

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_js_wall_renders_once_then_read_document_pays(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chromium = self._wire_launch_counting_playwright(monkeypatch)
        url = "https://example.gov/wall"
        asked = _serve_direct(monkeypatch, {url: _direct("js_wall", url=url, escalate_rendered=True)})
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("Model read.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        walled = await agentic_tools.fetch(url)
        assert walled.status == "empty", "a page nothing could read must not be laundered as a success"

        outcome = await agentic_tools.read_document(url, "what does the tracker report?")

        assert outcome.method == "document", "the paid reader is the rung left for a page we cannot read"
        assert len(chromium.launch_args) == 1, "the second launch would re-learn what this run already knows"
        assert asked == [url, url], (
            "the direct GET is deliberately NOT negative-cached: the driver is told to retry these URLs"
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_throttled_fetch_then_read_document_does_re_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """q45191's contract: a refusal we were served must stay re-requestable.

        The host answered 200 with a rate-limit interstitial, which is not evidence about the
        page — caching that outcome (as the pre-fix code cached its text) is what made the
        driver's own retry impossible.
        """
        interstitial = "Limit for old data queries exceeded. Permitted a query per 20 seconds per IP"
        asked = _serve_direct(monkeypatch, _ogimet_page(interstitial))
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("Model read.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        throttled = await agentic_tools.fetch(_OGIMET_URL)
        outcome = await agentic_tools.read_document(_OGIMET_URL, "the 2022-08-31 maximum")

        assert throttled.status == "throttled"
        assert outcome.method == "document"
        assert asked == [_OGIMET_URL, _OGIMET_URL], "a throttle is not a fact about the page, so nothing memoizes it"


class TestUrlContextSizeGate:
    """A document we hold is never sent to the paid reader, and the biggest are the clearest case.

    The gate rides the same branch as the text it guards, so the two cannot disagree. It is
    there because the nine archived documents past it carried 67% of the season's reader tokens
    and the largest of them returned nothing at all for the money.
    """

    def test_the_gate_reads_chars_over_four_as_tokens(self) -> None:
        at_bound = "x" * (URL_CONTEXT_SIZE_GATE_TOKENS * 4)
        assert local_document.exceeds_url_context_size_gate(at_bound) is False
        assert local_document.exceeds_url_context_size_gate(at_bound + "xxxx") is True
        assert local_document.exceeds_url_context_size_gate("") is False

    @pytest.mark.asyncio
    async def test_a_huge_held_document_is_served_locally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        held = local_document.HeldDocument(text="revision " * (URL_CONTEXT_SIZE_GATE_TOKENS + 10))
        assert local_document.exceeds_url_context_size_gate(held.text)
        monkeypatch.setattr(agentic_tools, "_acquire_local_document", AsyncMock(return_value=held))
        reader = _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document("https://example.gov/833k.pdf", "revision")

        assert outcome.method == "digest_local"
        assert len(outcome.content_markdown) <= agentic_tools._FETCH_WINDOW_CHARS
        reader.assert_not_called()

    def test_the_digest_width_is_the_configured_one(self) -> None:
        # The digest's width is a knob in constants.py, not a literal at the call site.
        digest = local_document.digest_held(
            local_document.HeldDocument(text="\n\n".join(f"paragraph {i} about revisions. " * 30 for i in range(20))),
            ask="revisions",
            top_k=DOCUMENT_DIGEST_TOP_K,
            max_chars=8000,
            source_url="https://example.gov/a",
        )
        assert digest.passages == DOCUMENT_DIGEST_TOP_K


class TestGapFillV2RendersThePlainRungsFinalUrl:
    """The browser is handed the direct fetch's POST-REDIRECT URL, ``direct.url``, never the URL
    the driver asked for. That is load-bearing since the transport pins Chromium's DNS to the host
    it is asked for and refuses a main frame that lands anywhere else: rendering the pre-redirect
    URL would pin the wrong host and then refuse the DOM when Chromium followed the same hop
    (every http-to-https, apex-to-www and shortener redirect), and gap-fill v2 would degrade on the
    resulting ``None`` with one warning line. Both of the loop's ladders reach the one rung that
    decides it, so both are driven here."""

    _REQUESTED = "https://example.com/start"
    _FINAL = "https://www.example.com/final"

    @staticmethod
    def _record_render_targets(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
        """A browser TRANSPORT that declines, recording the URL the rung asked it to render.

        One layer under the rung, because which URL the rung dials is the thing under test.
        """
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(rungs, "render_page", _fake_render(None, calls))
        # The landing differs from the cited URL, so the rung re-vets it before dialing.
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        return calls

    def _thin_direct_read_that_landed_elsewhere(self) -> FetchResult:
        return _direct("success", url=self._FINAL, text="Menu. Home.", escalate_rendered=True)

    @pytest.mark.asyncio
    async def test_the_fetch_ladder_renders_the_direct_rungs_final_url(self, monkeypatch: pytest.MonkeyPatch):
        renders = self._record_render_targets(monkeypatch)
        _serve_direct(monkeypatch, {self._REQUESTED: self._thin_direct_read_that_landed_elsewhere()})

        outcome = await agentic_tools.fetch(self._REQUESTED)

        assert [call["url"] for call in renders] == [self._FINAL]
        # The browser declined, so the thin-but-real direct read stands under its own method.
        assert outcome.method == "plain"

    @pytest.mark.asyncio
    async def test_the_document_ladder_renders_the_direct_rungs_final_url(self, monkeypatch: pytest.MonkeyPatch):
        renders = self._record_render_targets(monkeypatch)
        _serve_direct(monkeypatch, {self._REQUESTED: self._thin_direct_read_that_landed_elsewhere()})

        held = await agentic_tools._run_local_document_ladder(self._REQUESTED, ctx=None)

        assert [call["url"] for call in renders] == [self._FINAL]
        assert held.has_text

    @pytest.mark.asyncio
    async def test_through_fetch_a_scripted_302_decides_the_render_target(self, monkeypatch: pytest.MonkeyPatch):
        """End to end through the plain rung's own redirect loop, so the URL the browser is handed
        is proven to be the hop the direct fetch actually landed on rather than a value a stub
        supplied. The final page is thin, which is what sends the ladder to the browser."""
        renders = self._record_render_targets(monkeypatch)
        session = _FakeSession(
            _FakeResponse(status=302, headers={"Location": self._FINAL}),
            _FakeResponse(status=200, headers={"Content-Type": "text/html"}),
        )
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
        monkeypatch.setattr(
            classify,
            "read_body_capped",
            AsyncMock(return_value=b"<html><body><p>Menu. Home.</p></body></html>"),
        )
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.classify._extract_main_text", MagicMock(return_value="Menu. Home.")
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        await agentic_tools.fetch(self._REQUESTED)

        assert session.calls == [(self._REQUESTED, False), (self._FINAL, False)]
        assert [call["url"] for call in renders] == [self._FINAL]


# ---------------------------------------------------------------------------
# The Google-Extended robots pre-check on the paid reader (2026-09-03). Proven live: the
# verification probe's url_context call returned URL_RETRIEVAL_STATUS_ERROR on
# internationalaisafetyreport.org, whose (Cloudflare-managed) robots.txt carries the two
# directives below, while the identical call on a robots-allowed host retrieved. A retry cannot
# change a host-policy refusal, so that read is spend with a known-zero return.
# ---------------------------------------------------------------------------

_GOOGLE_EXTENDED_BLOCKED_ROBOTS = "User-agent: Google-Extended\nDisallow: /\n"
_GENERIC_CRAWLER_BLOCKED_ROBOTS = "User-agent: *\nDisallow: /\nCrawl-delay: 10\n"


def _robots_answers(
    page_url: str, robots_txt: str | None, *, page: FetchResult | None = None
) -> dict[str, FetchResult]:
    """``_serve_direct`` answers for one host: its ``/robots.txt``, and the page itself.

    The page defaults to a real host 403 (``http_status=403``), which is what leaves
    ``read_document``'s free ladder holding nothing and would earn the impersonated retry the
    module fixture declines; ``robots_txt=None`` is a policy we could not read at all.
    """
    split = urlsplit(page_url)
    robots_url = f"{split.scheme}://{split.netloc}/robots.txt"
    robots = (
        _direct("error", url=robots_url, content_type=None)
        if robots_txt is None
        else _direct("success", url=robots_url, text=robots_txt, http_status=200, content_type="text/plain")
    )
    return {
        robots_url: robots,
        page_url: page if page is not None else _direct("blocked", url=page_url, http_status=403),
    }


class TestUrlContextRobotsPreCheck:
    """One free request decides whether the paid ``url_context`` read can work at all.

    Only the ``Google-Extended`` group is consulted: our own free rungs dial under our own user
    agent and are unaffected, and this bot reads ``Content-Signal: use=reference`` as permitting
    reference use. So a host that blocks generic crawlers is still read for us, and only a host
    that names Gemini's retrieval token is skipped.
    """

    @staticmethod
    def _wire_reader(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        reader = MagicMock(return_value=("Model read.", 1, ["URL_RETRIEVAL_STATUS_SUCCESS"]))
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(agentic_tools, "_run_document_read_sync", reader)
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))
        return reader

    @pytest.mark.asyncio
    async def test_a_host_that_only_blocks_generic_crawlers_is_still_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "https://who.int/data/gho"
        asked = _serve_direct(monkeypatch, _robots_answers(url, _GENERIC_CRAWLER_BLOCKED_ROBOTS))
        reader = self._wire_reader(monkeypatch)

        outcome = await agentic_tools.read_document(url, "what does the indicator read?")

        assert outcome.method == "document"
        assert outcome.content_markdown == "Model read."
        reader.assert_called_once()
        assert "https://who.int/robots.txt" in asked

    @pytest.mark.asyncio
    async def test_a_google_extended_disallow_skips_the_paid_read(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The robots body goes through the REAL classification: a 41-character ``text/plain`` policy
        has to come back as its own text rather than as a sub-floor withhold, and the skip below is
        reachable only if it did."""
        url = "https://internationalaisafetyreport.org/chapters/2/"
        session = _FakeSession(
            _FakeResponse(status=403, headers={"Content-Type": "text/html"}),
            _FakeResponse(status=200, headers={"Content-Type": "text/plain"}),
        )
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
        monkeypatch.setattr(
            classify, "read_body_capped", AsyncMock(return_value=_GOOGLE_EXTENDED_BLOCKED_ROBOTS.encode())
        )
        reader = _no_paid_reader(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")

        with caplog.at_level(logging.INFO, logger=agentic_tools.__name__):
            outcome = await agentic_tools.read_document(url, "what does the chapter say about compute?")

        reader.assert_not_called()
        assert session.calls == [(url, False), ("https://internationalaisafetyreport.org/robots.txt", False)]
        assert outcome.status == "robots_disallowed"
        assert outcome.method == "document"
        assert _harvest_verification_tiers("read_document", {"url": url}, outcome) == {}, (
            "nothing was read, so nothing may claim the fetched tier"
        )
        assert "Retrying will not help" in outcome.content_markdown
        assert (
            "AGENTIC_URLCONTEXT_ROBOTS_SKIP: url=https://internationalaisafetyreport.org/chapters/2/ "
            "host=internationalaisafetyreport.org"
        ) in caplog.text

    @pytest.mark.asyncio
    async def test_an_unreadable_robots_txt_proceeds_to_the_reader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every ambiguity resolves toward paying: a wrong skip loses a document we could read."""
        url = "https://example.gov/report"
        _serve_direct(monkeypatch, _robots_answers(url, None))
        reader = self._wire_reader(monkeypatch)

        outcome = await agentic_tools.read_document(url, "what is the figure?")

        assert outcome.method == "document"
        reader.assert_called_once()

    @pytest.mark.asyncio
    async def test_robots_txt_is_read_once_per_host_per_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first_url = "https://internationalaisafetyreport.org/a"
        second_url = "https://internationalaisafetyreport.org/b"
        answers = _robots_answers(first_url, _GOOGLE_EXTENDED_BLOCKED_ROBOTS)
        answers[second_url] = _direct("blocked", url=second_url, http_status=403)
        asked = _serve_direct(monkeypatch, answers)
        _no_paid_reader(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")

        first = await agentic_tools.read_document(first_url, "ask one")
        second = await agentic_tools.read_document(second_url, "ask two")

        assert first.status == "robots_disallowed"
        assert second.status == "robots_disallowed"
        robots_calls = [url for url in asked if url.endswith("/robots.txt")]
        assert len(robots_calls) == 1, "the verdict is cached per host, so a run pays one request for it"

    @pytest.mark.asyncio
    async def test_the_free_rungs_are_not_gated_by_robots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pre-check guards the PAID rung only: a page our own client can read is served.

        ``internationalaisafetyreport.org``'s pages are readable by the plain fetch, and the
        operator's reading of its ``Content-Signal: use=reference`` is that reference use is
        permitted, so a disallowing robots.txt must not withhold a free read.
        """
        page = "The report states that training compute grew fourfold in 2026. " * 12
        url = "https://internationalaisafetyreport.org/chapters/2/"
        _serve_direct(
            monkeypatch,
            _robots_answers(url, _GOOGLE_EXTENDED_BLOCKED_ROBOTS, page=_direct("success", url=url, text=page)),
        )
        reader = _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document(url, "training compute growth")

        assert outcome.method == "digest_local"
        assert "training compute grew fourfold" in outcome.content_markdown
        reader.assert_not_called()


class TestGoogleExtendedRobotsRules:
    """Reading one group out of a robots.txt, biased hard toward paying rather than skipping."""

    def test_a_generic_crawler_block_is_not_a_google_extended_block(self) -> None:
        # The reason this is not ``urllib.robotparser``: its ``can_fetch("Google-Extended", ...)``
        # falls back to the ``User-agent: *`` group (verified on 3.12.12), which would skip the
        # paid read on every host that merely disallows crawlers — a far broader policy than the
        # one that was approved.
        assert robots_policy.google_extended_disallows(_GENERIC_CRAWLER_BLOCKED_ROBOTS, "/data") is False

    def test_the_receipt_host_shape_disallows_every_path(self) -> None:
        assert robots_policy.google_extended_disallows(_GOOGLE_EXTENDED_BLOCKED_ROBOTS, "/chapters/2/") is True
        assert robots_policy.google_extended_disallows(_GOOGLE_EXTENDED_BLOCKED_ROBOTS, "") is True

    def test_the_group_can_name_several_agents(self) -> None:
        robots = "User-agent: GPTBot\nUser-agent: google-extended\nDisallow: /reports\n"
        assert robots_policy.google_extended_disallows(robots, "/reports/2026") is True
        assert robots_policy.google_extended_disallows(robots, "/about") is False

    def test_rules_stop_belonging_to_a_group_at_the_next_agent_line(self) -> None:
        robots = "User-agent: Google-Extended\nDisallow: /secret\n\nUser-agent: *\nDisallow: /\n"
        assert robots_policy.google_extended_disallows(robots, "/secret/a") is True
        assert robots_policy.google_extended_disallows(robots, "/public") is False

    def test_the_longest_matching_rule_wins_and_allow_takes_a_tie(self) -> None:
        robots = "User-agent: Google-Extended\nDisallow: /docs\nAllow: /docs/public\n"
        assert robots_policy.google_extended_disallows(robots, "/docs/private") is True
        assert robots_policy.google_extended_disallows(robots, "/docs/public/a") is False
        tie = "User-agent: Google-Extended\nDisallow: /\nAllow: /\n"
        assert robots_policy.google_extended_disallows(tie, "/anything") is False

    def test_comments_and_blank_lines_are_ignored(self) -> None:
        robots = "# policy\n\nUser-agent: Google-Extended  # the AI token\nDisallow: /  # everything\n"
        assert robots_policy.google_extended_disallows(robots, "/x") is True

    def test_an_empty_disallow_allows_everything(self) -> None:
        assert robots_policy.google_extended_disallows("User-agent: Google-Extended\nDisallow:\n", "/x") is False

    @pytest.mark.parametrize("rule", ["/*/private", "/*.pdf$", "*"])
    def test_a_rule_needing_glob_matching_is_left_alone(self, rule: str) -> None:
        # Not modelled, so it cannot disallow: the read proceeds and is paid for, which is the
        # only direction an unmatched rule is allowed to fail in.
        robots = f"User-agent: Google-Extended\nDisallow: {rule}\n"
        assert robots_policy.google_extended_disallows(robots, "/reports/private/a.pdf") is False

    def test_a_trailing_star_is_the_prefix_it_decorates(self) -> None:
        robots = "User-agent: Google-Extended\nDisallow: /reports*\n"
        assert robots_policy.google_extended_disallows(robots, "/reports/2026") is True
        assert robots_policy.google_extended_disallows(robots, "/about") is False

    def test_an_empty_robots_txt_says_nothing(self) -> None:
        assert robots_policy.google_extended_disallows("", "/x") is False


# An article-shaped page well over GAP_FILL_V2_MIN_CONTENT_CHARS once extracted, so a rescue
# through it is a complete read rather than one the ladder escalates to the browser.
_IMPERSONATED_PAGE = (
    "<!doctype html><html><head><title>Work Stoppages</title></head><body><nav>Home | Data</nav><article>"
    "<h1>Major Work Stoppages in 2026</h1>"
    + "".join(
        f"<p>The Bureau of Labor Statistics counted 12 major work stoppages beginning in 2026 through August, "
        f"paragraph {index} of the summary table dated 2026-08-28, covering 1,000 or more workers each.</p>"
        for index in range(6)
    )
    + "</article></body></html>"
).encode()


class TestGapFillV2ImpersonatedRetry:
    """The direct fetch's 403, re-dialed under a real browser's TLS fingerprint by the shared rung
    (`fetch_ladder.rungs._impersonate_rung`) and mapped onto this ladder's result type.

    The transport is patched at the rung's own import seam `rungs.fetch_impersonated`, never at its
    own session, so the suite's `_block_native_egress` guard stays armed underneath every test.
    """

    _URL = "https://www.bls.gov/wsp/"
    _SECOND_URL = "https://www.bls.gov/news.release/wkstp.htm"

    @pytest.fixture(autouse=True)
    def _fresh_memo(self):
        """The host memo is process-wide by design (shared with Tier 1), so isolate it per test."""
        reset_impersonation_memo()
        yield
        reset_impersonation_memo()

    @pytest.fixture(autouse=True)
    def _arm_the_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Restore the transport's trigger set, which this module's fixture empties by default.

        The transport's constant OBJECT, not a copy, so the population asserted on here is the one
        prod and Tier 1 read.
        """
        monkeypatch.setattr(impersonated_fetch, "IMPERSONATE_TRIGGER_STATUSES", IMPERSONATE_TRIGGER_STATUSES)

    @staticmethod
    def _transport(monkeypatch: pytest.MonkeyPatch, answer: Any) -> list[dict[str, Any]]:
        """The one transport double both suites share, patched at the shared rung's import seam."""
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(rungs, "fetch_impersonated", fake_impersonated_fetch(answer, calls))
        return calls

    @staticmethod
    def _ctx() -> context.LadderContext:
        """A rung context carrying the loop's own host map and its `fetch` preset's dial cap."""
        return context.LadderContext(policy=GAP_FILL_FETCH_POLICY, host_sems=agentic_tools._FETCH_HOST_SEMAPHORES)

    def _serve_a_403(self, monkeypatch: pytest.MonkeyPatch, *, landed_on: str | None = None) -> list[str]:
        """The host answered our own client 403, on ``landed_on`` where its redirect moved us."""
        return _serve_direct(monkeypatch, {self._URL: self._blocked(landed_on or self._URL, 403)})

    @classmethod
    def _response(
        cls, status: int, *, body: bytes = b"", content_type: str = "text/html", url: str | None = None
    ) -> ImpersonatedResponse:
        return _impersonated(status, body=body, content_type=content_type, url=url or cls._URL)

    @staticmethod
    def _blocked(url: str, http_status: int | None) -> FetchResult:
        return _direct("blocked", url=url, http_status=http_status)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [403, 406, 429, 503])
    async def test_a_non_200_plain_result_carries_its_http_status(self, status: int, monkeypatch) -> None:
        session = _FakeSession(_FakeResponse(status=status, headers={"Content-Type": "text/html"}))
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)

        result = await _fetch_direct_only("https://example.com/gated")

        assert result.http_status == status

    @pytest.mark.asyncio
    async def test_the_refusals_this_ladder_makes_itself_carry_no_http_status(self, monkeypatch) -> None:
        """Both come back `blocked`, and neither is a host's verdict: the trigger must not fire on them."""

        async def is_public(url: str) -> bool:
            return "metaculus.com" in url

        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", is_public)
        session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/html"}))
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)

        non_public = await _fetch_direct_only("http://169.254.169.254/latest/meta-data/")
        self_reference = await _fetch_direct_only("https://www.metaculus.com/questions/1/")

        assert (non_public.status, non_public.http_status) == ("blocked", None)
        assert (self_reference.status, self_reference.http_status) == ("blocked", None)
        assert session.calls == [], "neither refusal issues a request"

    @pytest.mark.asyncio
    async def test_a_403_is_retried_and_a_rescue_is_the_fetch_outcome(self, monkeypatch) -> None:
        self._serve_a_403(monkeypatch)
        _serve_rendered(monkeypatch, None)
        calls = self._transport(monkeypatch, self._response(200, body=_IMPERSONATED_PAGE))

        outcome = await agentic_tools.fetch(self._URL)

        assert outcome.status == "ok"
        assert outcome.method == "impersonate"
        assert "12 major work stoppages" in outcome.content_markdown
        assert _method_to_tier(outcome.method) == "fetched", "the retry read the host's own bytes"
        (call,) = calls
        assert call["url"] == self._URL
        # Both body caps, so a declared PDF between them is read as the direct route reads one.
        assert call["host_sems"] is agentic_tools._FETCH_HOST_SEMAPHORES
        assert call["per_hop_timeout_s"] == RESOLUTION_SOURCE_HTTP_TIMEOUT
        assert call["max_bytes"] == RESOLUTION_SOURCE_MAX_RESPONSE_BYTES
        assert call["document_max_bytes"] == DOCUMENT_TEXT_PDF_MAX_BYTES
        assert call["deadline_monotonic_s"] <= monotonic() + RESOLUTION_SOURCE_HTTP_TIMEOUT

    @pytest.mark.asyncio
    @pytest.mark.parametrize("http_status", [406, 429])
    async def test_the_other_block_statuses_are_not_retried(self, http_status: int, monkeypatch) -> None:
        _serve_direct(monkeypatch, {self._URL: self._blocked(self._URL, http_status)})
        calls = self._transport(monkeypatch, self._response(200, body=_IMPERSONATED_PAGE))

        outcome = await agentic_tools.fetch(self._URL)

        assert calls == []
        assert (outcome.status, outcome.method) == ("blocked", "plain")

    @pytest.mark.asyncio
    async def test_a_url_this_ladder_refused_itself_is_never_retried(self, monkeypatch) -> None:
        """The SSRF bypass case, end to end through the real plain rung: a non-public URL and a
        Metaculus self-reference both reach `blocked` with no `http_status`, so the transport,
        whose libcurl connection never passes aiohttp's filtering resolver, is never handed them."""
        calls = self._transport(monkeypatch, self._response(200, body=_IMPERSONATED_PAGE))

        async def is_public(url: str) -> bool:
            return "metaculus.com" in url

        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard.is_public_http_url", is_public)

        non_public = await agentic_tools.fetch("http://169.254.169.254/latest/meta-data/")
        self_reference = await agentic_tools.fetch("https://www.metaculus.com/questions/1/")

        assert non_public.status == "blocked"
        assert self_reference.status == "blocked"
        assert calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "decline",
        [
            ImpersonateTransportError(failure_class="connection", exc="ConnectionError"),
            ImpersonatePinNotHeld(_URL, expected_ip="23.0.0.1", actual_ip="10.0.0.8"),
            ImpersonateHopRefused("ssrf_blocked", hop_url="http://10.0.0.8/status", from_url=_URL),
            ImpersonateBodyTooLarge(_URL, bytes_read=6_000_000, max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES),
            ImpersonateRedirectLimit(_URL, final_url=f"{_URL}?hop=6"),
            ImpersonateBudgetExhausted(waiting_on="the host gate"),
        ],
        ids=lambda decline: type(decline).__name__,
    )
    async def test_a_decline_leaves_the_blocked_outcome_byte_identical(self, decline, monkeypatch) -> None:
        """Every member of the `ImpersonateDeclined` family folds back into `None` here, so the
        driver sees exactly the `blocked` outcome it saw before the rung existed, and none of them
        says anything about the host's view of our fingerprint, so none writes the memo."""
        self._serve_a_403(monkeypatch)
        self._transport(monkeypatch, decline)

        outcome = await agentic_tools.fetch(self._URL)

        assert outcome == ToolOutcome(content_markdown="Fetch blocked with HTTP 403.", method="plain", status="blocked")
        assert _method_to_tier(outcome.method) == "fetched", (
            "unchanged: `plain` was always tiered, `blocked` never grants it"
        )
        assert impersonation_refused(self._URL) is False

    @pytest.mark.asyncio
    async def test_a_still_403_declines_and_memoizes_the_host_for_the_run(self, monkeypatch) -> None:
        calls = self._transport(monkeypatch, self._response(403))

        first = await rungs._impersonate_rung(self._URL, self._blocked(self._URL, 403), host_sems={}, ctx=self._ctx())
        second = await rungs._impersonate_rung(
            self._SECOND_URL, self._blocked(self._SECOND_URL, 403), host_sems={}, ctx=self._ctx()
        )

        assert first is None
        assert second is None
        assert len(calls) == 1, "the second URL on the host never dialed"
        assert impersonation_refused(self._URL) is True

    @pytest.mark.asyncio
    async def test_a_block_answered_by_a_redirect_target_memoizes_the_answering_host_and_the_dialed_url(
        self, monkeypatch
    ) -> None:
        """The transport follows redirects itself, so the block can come from a later hop's netloc;
        through the same transport rule Tier 1 uses, the memo bans the host that answered and the
        one URL that was dialed, and leaves the dialed host's other URLs dialable, since a host that
        merely redirected never refused us."""
        answered = "https://edge.example.net/denied"
        self._transport(monkeypatch, self._response(403, url=answered))
        direct = self._blocked(self._URL, 403)

        assert await rungs._impersonate_rung(self._URL, direct, host_sems={}, ctx=self._ctx()) is None

        assert impersonation_refused(answered) is True
        assert impersonation_refused(self._URL) is True
        assert impersonation_refused(self._SECOND_URL) is False

    @pytest.mark.asyncio
    async def test_a_404_under_impersonation_declines_without_memoizing(self, monkeypatch) -> None:
        calls = self._transport(monkeypatch, self._response(404))
        second = self._blocked(self._SECOND_URL, 403)

        assert (
            await rungs._impersonate_rung(self._URL, self._blocked(self._URL, 403), host_sems={}, ctx=self._ctx())
            is None
        )
        assert await rungs._impersonate_rung(self._SECOND_URL, second, host_sems={}, ctx=self._ctx()) is None

        assert len(calls) == 2
        assert impersonation_refused(self._URL) is False

    @pytest.mark.asyncio
    async def test_the_kill_switch_declines_before_dialing(self, monkeypatch) -> None:
        monkeypatch.setenv(RESOLUTION_SOURCE_IMPERSONATE_ENABLED_ENV, "false")
        calls = self._transport(monkeypatch, self._response(200, body=_IMPERSONATED_PAGE))
        direct = self._blocked(self._URL, 403)

        assert await rungs._impersonate_rung(self._URL, direct, host_sems={}, ctx=self._ctx()) is None
        assert calls == []

    @pytest.mark.asyncio
    async def test_the_trigger_is_the_transports_read_at_call_time(self, monkeypatch) -> None:
        """One switch for both fetchers: emptying the transport's set declines the retry here
        exactly as it declines the Tier-1 rung, with the direct `blocked` standing."""
        monkeypatch.setattr(impersonated_fetch, "IMPERSONATE_TRIGGER_STATUSES", frozenset())
        self._serve_a_403(monkeypatch)
        calls = self._transport(monkeypatch, self._response(200, body=_IMPERSONATED_PAGE))

        outcome = await agentic_tools.fetch(self._URL)

        assert calls == []
        assert (outcome.status, outcome.method) == ("blocked", "plain")

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_read_document_rescues_a_403_for_free_before_the_paid_read(self, monkeypatch) -> None:
        """The second free ladder: `read_document`'s local acquisition sits immediately in front of
        the paid `url_context` read, so a cold `read_document` on a host whose 403 is a fingerprint
        verdict must be rescued by the same retry `fetch` runs and digested for free. Before this
        wiring the ladder held nothing and the reader was paid for bytes the retry fetches."""
        self._serve_a_403(monkeypatch)
        rendered_on = _serve_rendered(monkeypatch, None)
        calls = self._transport(monkeypatch, self._response(200, body=_IMPERSONATED_PAGE))
        reader = _no_paid_reader(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")

        outcome = await agentic_tools.read_document(self._URL, "how many major work stoppages began in 2026")

        assert outcome.method == "digest_local"
        assert _method_to_tier(outcome.method) == "fetched", "the retry read the host's own bytes"
        assert "12 major work stoppages" in outcome.content_markdown
        assert [call["url"] for call in calls] == [self._URL]
        reader.assert_not_called()
        assert rendered_on == []

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_read_document_on_a_host_that_refuses_both_clients_still_pays(self, monkeypatch) -> None:
        """The DataDome shape: a still-403 under impersonation leaves the ladder holding nothing,
        the host is memoized, and the paid reader (which dials from Gemini's address) gets its turn."""
        self._serve_a_403(monkeypatch)
        calls = self._transport(monkeypatch, self._response(403))
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("Model read.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        outcome = await agentic_tools.read_document(self._URL, "how many major work stoppages began in 2026")

        assert outcome.method == "document"
        assert len(calls) == 1
        assert impersonation_refused(self._URL) is True

    @pytest.mark.asyncio
    async def test_the_document_ladder_declines_the_retry_when_its_budget_is_nearly_spent(self, monkeypatch) -> None:
        """Under `read_document` the retry runs inside the document ladder's own 25 s wall. With
        less than the rung's floor left (the same `RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S`
        Tier 1 claims) it declines without a transport call, instead of dialing a 20 s wall the
        ladder's own `wait_for` would cancel mid-transfer."""
        self._serve_a_403(monkeypatch)
        rendered_on = _serve_rendered(monkeypatch, None)
        calls = self._transport(monkeypatch, self._response(200, body=_IMPERSONATED_PAGE))
        nearly_spent = RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S - 1.0
        monkeypatch.setattr(context.LadderContext, "rung_budget_s", lambda self: nearly_spent)

        held = await agentic_tools._run_local_document_ladder(self._URL, ctx=None)

        assert calls == []
        assert held.has_text is False
        assert rendered_on == []

    @pytest.mark.asyncio
    async def test_the_document_ladder_dials_under_its_own_deadline(self, monkeypatch) -> None:
        """With budget above the floor the retry dials, and the transport's deadline is what is
        left of the ladder's own wall rather than a fresh `RESOLUTION_SOURCE_HTTP_TIMEOUT` hop that
        outlives it."""
        self._serve_a_403(monkeypatch)
        calls = self._transport(monkeypatch, self._response(200, body=_IMPERSONATED_PAGE))
        left_of_the_ladders_wall = 8.0
        monkeypatch.setattr(context.LadderContext, "rung_budget_s", lambda self: left_of_the_ladders_wall)
        before = monotonic()

        held = await agentic_tools._run_local_document_ladder(self._URL, ctx=None)

        (call,) = calls
        assert call["deadline_monotonic_s"] == pytest.approx(before + left_of_the_ladders_wall, abs=0.5)
        assert call["deadline_monotonic_s"] < monotonic() + RESOLUTION_SOURCE_HTTP_TIMEOUT - 5.0
        assert held.has_text

    @pytest.mark.asyncio
    async def test_acquire_local_document_and_its_ladder_share_one_budget(self, monkeypatch) -> None:
        """The `wait_for` that bounds acquisition and the wall every rung inside sizes itself off
        have to be ONE figure, so a rung declines under its own floor rather than being cancelled
        mid-dial."""
        seen: list[LadderPolicy] = []

        async def _record(url: str, *, policy: LadderPolicy, ctx: Any) -> FetchResult:
            del ctx
            seen.append(policy)
            await asyncio.sleep(0)
            return _direct("js_wall", url=url)

        monkeypatch.setattr(agentic_tools, "fetch_url", _record)

        await agentic_tools._acquire_local_document(self._URL)

        (policy,) = seen
        assert policy is GAP_FILL_DOCUMENT_POLICY
        assert policy.total_wall_s == agentic_tools._LOCAL_DOCUMENT_BUDGET_S

    @pytest.mark.asyncio
    async def test_fetch_keeps_the_full_wall(self, monkeypatch) -> None:
        """Under the `fetch` tool (a 90 s wall) the dial cap binds rather than the wall, so the
        retry keeps the one-plain-hop budget it always had."""
        self._serve_a_403(monkeypatch)
        _serve_rendered(monkeypatch, None)
        calls = self._transport(monkeypatch, self._response(200, body=_IMPERSONATED_PAGE))
        before = monotonic()

        await agentic_tools.fetch(self._URL)

        (call,) = calls
        assert call["deadline_monotonic_s"] == pytest.approx(before + RESOLUTION_SOURCE_HTTP_TIMEOUT, abs=0.5)

    def test_the_fetch_presets_wall_is_the_tools_own_ceiling(self) -> None:
        """The two figures are spelled apart, so one moving without the other is a drift."""
        fetch_tool = next(spec for spec in agentic_tools.build_gap_fill_tools("topic") if spec.name == "fetch")
        assert GAP_FILL_FETCH_POLICY.total_wall_s == fetch_tool.timeout_s

    @pytest.mark.asyncio
    async def test_the_retry_dials_the_plain_rungs_final_url(self, monkeypatch) -> None:
        """`direct.url` is the last hop of the direct fetch's own guarded redirect loop, the host
        that actually refused us, the same choice the browser rung makes."""
        final = "https://www.bls.gov/wsp/index.htm"
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        self._serve_a_403(monkeypatch, landed_on=final)
        _serve_rendered(monkeypatch, None)
        calls = self._transport(monkeypatch, self._response(200, body=_IMPERSONATED_PAGE, url=final))

        outcome = await agentic_tools.fetch(self._URL)

        assert [call["url"] for call in calls] == [final]
        assert outcome.method == "impersonate"

    @pytest.mark.asyncio
    async def test_an_impersonated_pdf_keeps_the_local_document_rungs_method(self, monkeypatch) -> None:
        """The bls.gov `wkstp.pdf` case: the body goes through the same local PDF rung a direct
        body does, and the `fetch` handler keys on that method, so it is not renamed."""
        pdf = build_text_pdf([["The unemployment rate was 4.1 percent in May 2026, revised from 4.0 percent."]])
        self._serve_a_403(monkeypatch)
        self._transport(monkeypatch, self._response(200, body=pdf, content_type="application/pdf"))

        outcome = await agentic_tools.fetch(self._URL)

        assert outcome.method == local_document.PDF_LOCAL_METHOD
        assert "4.1 percent in May 2026" in outcome.content_markdown

    def test_every_retrieval_method_is_tiered(self) -> None:
        """A method absent from `_METHOD_TO_TIER` grants NO verification tier, so a page the rung
        really did retrieve would stay untiered and its discrepancy silently demoted below the
        briefing (the 131.3 failure mode). Every method a real retrieval can carry is pinned here so
        the next one cannot land untiered."""
        retrieval_methods = {
            "plain",
            "rendered",
            "impersonate",
            "cache",
            "document",
            local_document.PDF_LOCAL_METHOD,
            local_document.DIGEST_LOCAL_METHOD,
        }

        assert retrieval_methods <= set(provenance._METHOD_TO_TIER)
        assert all(_method_to_tier(method) == "fetched" for method in retrieval_methods)
        # The placeholders that mean nothing was read stay untiered.
        assert _method_to_tier(fetch_outcomes.DOCUMENT_NEEDED_METHOD) is None
        assert _method_to_tier(local_document.OVERSIZE_DOCUMENT_METHOD) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("content_type", "body"),
        [
            ("text/html", _IMPERSONATED_PAGE),
            ("application/pdf", build_text_pdf([["A PDF the local rung reads whichever transport fetched it."]])),
            ("text/csv", b"date,count\n2026-08-01,11\n2026-08-02,12\n"),
            ("image/png", b"\x89PNG\r\n\x1a\nbinary"),
            # A declared image whose bytes the magic sniff does not know: the header clause is what
            # keeps the two paths agreeing, since the aiohttp path decides on the header alone.
            ("image/webp", b"RIFF\x24\x00\x00\x00WEBPVP8 binary"),
            ("image/svg+xml", b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'),
            ("application/octet-stream", b"\x89PNG\r\n\x1a\nbinary"),
        ],
    )
    async def test_the_body_classification_is_the_same_whichever_transport_read_it(
        self, content_type: str, body: bytes, monkeypatch
    ) -> None:
        """`classify` is the one copy of the classification rule: the aiohttp path (through the
        direct fetch) and the impersonated path must agree on every body shape."""
        url = "https://example.com/parity"
        session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": content_type}))
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr(classify, "read_body_capped", AsyncMock(return_value=body))

        via_aiohttp = await direct_fetch._fetch_direct(session, url, {}, self._ctx())
        via_impersonated = await rungs._impersonated_body_outcome(
            _impersonated(200, body=body, content_type=content_type, url=url), self._ctx()
        )

        assert via_aiohttp == via_impersonated


class TestPlainHtmlExtractionPolicy:
    """Item A: the loop's HTML path now runs Tier 1's free extraction steps.

    The shared HTML classifier routes an HTML body through `classify._extract_page_text`
    (the ARIA-table rewrite plus the two-pass default/precision policy) and prepends the inline
    chart-data read, and follows a `<meta http-equiv=refresh>` stub as a hop. Ported from the
    resolution-source fetcher, where 33 of 80 rendered reads served under 500 chars because the
    loop lacked these steps (fetch-gap inventory, 2026-09-09)."""

    @staticmethod
    def _serve_html(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
        session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/html"}))
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
        monkeypatch.setattr(classify, "read_body_capped", AsyncMock(return_value=body))
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    @pytest.mark.asyncio
    async def test_the_html_path_runs_the_two_pass_extraction_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The extraction is now `_extract_page_text` (which contains the ARIA rewrite and the
        precision fallback), not a single default `_extract_main_text` call."""
        body = b"<html><body><p>page</p></body></html>"
        self._serve_html(monkeypatch, body)
        spy = MagicMock(return_value=verdict.PageExtraction(text="A calibrated extraction of the page body. " * 3))
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.classify._extract_page_text", spy)

        result = await _fetch_direct_only("https://example.com/page")

        assert result.status == "ok"
        assert "A calibrated extraction of the page body." in result.text
        # Called with the decoded html, the raw bytes, the url, and the undecodable ratio (0.0 here).
        (call_args,) = spy.call_args_list
        assert call_args.args[1] == body
        assert call_args.args[2] == "https://example.com/page"
        assert call_args.args[3] == 0.0

    @pytest.mark.asyncio
    async def test_a_chrome_default_extraction_is_rescued_by_the_precision_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A default extraction that is a navigation tree (over the floor, chrome-shaped) is
        re-extracted under precision, and the precision text publishes."""
        chrome = "\n".join(["Home", "About us", "Contact", "Products", "Services", "Careers", "Blog"] * 12)
        content = (
            "The unemployment rate for May 2026 was reported at 4.1 percent, a revision from April's 4.0 "
            "that the Bureau of Labor Statistics published in its monthly employment situation release "
            "covering both the payroll survey and the household survey for the reference period. " * 2
        )
        self._serve_html(monkeypatch, b"<html><body><nav>menu</nav></body></html>")

        def fake_extract(source: Any, url: str, *, favor_precision: bool = False) -> str:
            return content if favor_precision else chrome

        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.classify._extract_main_text", fake_extract)

        result = await _fetch_direct_only("https://example.com/report")

        assert result.status == "ok"
        assert "unemployment rate for May 2026" in result.text
        assert "About us" not in result.text

    @pytest.mark.asyncio
    async def test_inline_chart_data_is_read_and_leads_the_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A page whose prose carries none of the resolving figures still serves the series from
        its inline chart config (q43949), and a page carrying a chart never escalates to render."""
        config = {"series": [{"name": "Cases", "data": [["2024", 10], ["2025", 25], ["2026", 1240]]}]}
        body = (
            f"<html><body><nav>menu</nav>"
            f'<div class="charts-highchart" data-chart="{_escape_config(config)}"></div></body></html>'
        ).encode()
        self._serve_html(monkeypatch, body)
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.classify._extract_main_text", MagicMock(return_value=None)
        )

        result = await _fetch_direct_only("https://example.com/tracker")

        assert result.status == "ok"
        assert "2026=1240" in result.text
        assert result.escalate_rendered is False

    @pytest.mark.asyncio
    async def test_an_aria_table_becomes_readable_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A `<div role="table">` stat block is rewritten to a real table before extraction, so
        its cell value survives (real trafilatura, no `_extract_main_text` patch)."""
        body = (
            b"<html><body><div role='table'>"
            b"<div role='row'><div role='columnheader'>Metric</div><div role='columnheader'>Value</div></div>"
            b"<div role='row'><div role='cell'>Hospitalizations</div><div role='cell'>922</div></div>"
            b"</div></body></html>"
        )
        self._serve_html(monkeypatch, body)

        result = await _fetch_direct_only("https://www.cdc.gov/outbreak")

        assert "Hospitalizations" in result.text
        assert "922" in result.text

    @pytest.mark.asyncio
    async def test_a_meta_refresh_stub_is_followed_as_one_hop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 200 whose only content is a `<meta http-equiv=refresh>` stub (a cdc.gov surveillance
        page) hops once to the target through this ladder's own re-guarded redirect loop."""
        stub = b"<html><head><meta http-equiv='refresh' content='0; url=/real/page'></head><body></body></html>"
        target = b"<html><body><p>resolving content</p></body></html>"
        session = _FakeSession(
            _FakeResponse(status=200, headers={"Content-Type": "text/html"}),
            _FakeResponse(status=200, headers={"Content-Type": "text/html"}),
        )
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._get_session", lambda: session)
        monkeypatch.setattr(classify, "read_body_capped", AsyncMock(side_effect=[stub, target]))
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.classify._extract_main_text",
            MagicMock(side_effect=[None, "Resolving content read from the refresh target. " * 3]),
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        result = await _fetch_direct_only("https://example.com/stub")

        assert result.status == "ok"
        assert result.url == "https://example.com/real/page"
        assert "Resolving content read from the refresh target." in result.text
        assert session.calls == [("https://example.com/stub", False), ("https://example.com/real/page", False)]


class TestTheLoopEmitsTheSharedFetchMarkers:
    """The per-URL fetch record this loop never had, on the markers the fetcher already emits.

    Both lines are data contracts: the research archive matches them by regex on the exact field
    order (``scripts/telemetry/markers.py``), so a line the loop emits has to PARSE through the
    registered spec rather than merely look right.
    """

    def _arm_a_refused_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 403 the impersonated retry is offered and declines, so both markers have content."""
        _serve_direct(monkeypatch, _direct("blocked", http_status=403))
        monkeypatch.setattr(impersonated_fetch, "IMPERSONATE_TRIGGER_STATUSES", frozenset({403}))
        monkeypatch.setattr(rungs, "fetch_impersonated", AsyncMock(side_effect=ImpersonateDeclined("declined")))

    @pytest.mark.asyncio
    async def test_a_fetch_emits_one_parseable_fetch_line_naming_this_caller(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._arm_a_refused_host(monkeypatch)

        with caplog.at_level("INFO", logger="metaculus_bot.research.agentic.tools"):
            await agentic_tools.fetch(_URL)

        (fetch_line,) = [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")]
        spec = next(s for s in MARKER_SPECS if s.name == "resolution_source_fetch")
        match = spec.regex.search(fetch_line)
        assert match is not None
        assert match.group("caller") == "gap_fill_v2"
        # `question=None`, as this loop's three event markers are: a tool call holds no question id.
        assert qid_from_ref(match.group("question")) is None
        assert match.group("status") == "blocked"
        assert match.group("http") == "403"

    @pytest.mark.asyncio
    async def test_every_rung_that_fired_emits_one_parseable_escalation_line(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._arm_a_refused_host(monkeypatch)

        with caplog.at_level("INFO", logger="metaculus_bot.research.agentic.tools"):
            await agentic_tools.fetch(_URL)

        spec = next(s for s in MARKER_SPECS if s.name == "resolution_source_escalation")
        fired = []
        for line in [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_ESCALATION:")]:
            match = spec.regex.search(line)
            assert match is not None, line
            assert match.group("caller") == "gap_fill_v2"
            fired.append(match.group("rung"))
        assert "impersonate" in fired

    @pytest.mark.asyncio
    async def test_the_robots_pre_check_is_not_recorded_as_a_fetch_the_driver_made(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """It is a gate on the paid rung, not a URL the driver asked for."""
        robots_url = robots_txt_url(_URL)
        policy_body = "User-agent: *\nAllow: /"
        _serve_direct(
            monkeypatch,
            {robots_url: _direct("success", url=robots_url, text=policy_body, content_type="text/plain")},
        )

        with caplog.at_level("INFO", logger="metaculus_bot.research.agentic.tools"):
            body = await agentic_tools._fetch_robots_txt(robots_url)

        # The verdict with no content floor is the point: a 22-character policy is its own text.
        assert body == policy_body
        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == []


class TestGapFillV2WaybackRung:
    """Item B: the Wayback Machine as v2 `fetch`'s last free rung after the impersonated retry.

    Reuses `research/wayback.py`'s pure helpers and the shared ladder's own direct fetch, so the
    snapshot GET inherits the 5 MiB body cap, the redirect vetting and the classification a live
    page gets. Unlike Tier 1 it applies NO age bound and SURFACES the capture date instead, because
    a driver-chosen URL is not a cited grading source. 133 never-read blocked URLs before the
    impersonated retry existed, plus 7 paywalled (fetch-gap inventory, 2026-09-09)."""

    _URL = "https://www.bls.gov/wsp/"
    _NOW = datetime(2026, 9, 9, tzinfo=UTC)

    @staticmethod
    def _snapshot(url: str, *, text: str = "ARCHIVED BODY of the stoppages table.", status: FetchStatus = "success"):
        return _direct(status, url=url, text=text, links=["https://www.bls.gov/a"], http_status=200)

    @staticmethod
    def _blocked(url: str, http_status: int | None) -> FetchResult:
        return _direct("blocked", url=url, http_status=http_status)

    def _question_ctx(self) -> context.LadderContext:
        """The question's ladder context with the forecast clock pinned, so a capture's age is exact."""
        return context.LadderContext(now=self._NOW, host_sems=agentic_tools._FETCH_HOST_SEMAPHORES)

    def _rung_ctx(self) -> context.LadderContext:
        return replace(self._question_ctx(), policy=GAP_FILL_FETCH_POLICY)

    def _serve_the_archive(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: FetchResult, *, page: str = ""
    ) -> list[str]:
        """The cited page refuses us and the archive answers with ``snapshot``."""
        url = page or self._URL
        return _serve_direct(
            monkeypatch,
            {url: self._blocked(url, 403), wayback_snapshot_url(url, now=self._NOW): snapshot},
        )

    def _public(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.guard.is_public_http_url", AsyncMock(return_value=True)
        )

    @pytest.mark.parametrize(
        ("direct", "applies"),
        [
            (_direct("blocked", url="https://x/y", http_status=403), True),
            (_direct("blocked", url="https://x/y", http_status=429), True),
            (_direct("error", url="https://x/y"), True),
            (_direct("blocked", url="https://x/y", http_status=None), False),
            (_direct("success", url="https://x/y", text="page"), False),
            (_direct("js_wall", url="https://x/y"), False),
        ],
    )
    def test_wayback_applies_only_to_a_host_refusal_or_error_never_our_own(
        self, direct: FetchResult, applies: bool
    ) -> None:
        assert rungs._wayback_rung_applies(direct, GAP_FILL_FETCH_POLICY) is applies

    def test_wayback_method_maps_to_the_fetched_tier(self) -> None:
        assert _method_to_tier("wayback") == "fetched"

    @pytest.mark.asyncio
    async def test_the_capture_date_is_surfaced_and_no_age_bound_is_applied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 251-day-old capture — far past Tier 1's 30-day cutoff — is still served, with its age
        disclosed for the driver to weigh."""
        self._public(monkeypatch)
        cap = "https://web.archive.org/web/20260101000000id_/https://www.bls.gov/wsp/"
        self._serve_the_archive(monkeypatch, self._snapshot(cap))

        outcome = await agentic_tools.fetch(self._URL, ctx=self._question_ctx())

        assert outcome.method == "wayback"
        assert "captured 2026-01-01" in outcome.content_markdown
        assert "251 days before this forecast" in outcome.content_markdown
        assert "ARCHIVED BODY of the stoppages table." in outcome.content_markdown

    @pytest.mark.asyncio
    async def test_a_host_refused_page_is_served_from_the_archive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._public(monkeypatch)
        cap = "https://web.archive.org/web/20250401000000id_/https://www.bls.gov/wsp/"
        self._serve_the_archive(monkeypatch, self._snapshot(cap, text="12 major work stoppages in 2024."))

        outcome = await agentic_tools.fetch(self._URL, ctx=self._question_ctx())

        assert outcome.method == "wayback"
        assert "12 major work stoppages in 2024." in outcome.content_markdown
        assert "captured 2025-04-01" in outcome.content_markdown

    @pytest.mark.asyncio
    async def test_the_archive_is_not_tried_for_a_url_we_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A `blocked` with no `http_status` is our own refusal (non-public or platform self-ref);
        handing it to the archive is the SSRF bypass the exclusion prevents."""
        asked = _serve_direct(monkeypatch, {self._URL: self._blocked(self._URL, None)})

        outcome = await agentic_tools.fetch(self._URL, ctx=self._question_ctx())

        assert outcome.status == "blocked"
        assert asked == [self._URL], "no snapshot was ever requested for a URL we refused ourselves"

    @pytest.mark.asyncio
    async def test_an_undatable_capture_declines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The archive answered the year request directly rather than a dated capture, so the copy
        cannot carry the age disclosure that makes it admissible."""
        self._public(monkeypatch)
        undated = "https://web.archive.org/web/2026id_/https://www.bls.gov/wsp/"
        self._serve_the_archive(monkeypatch, self._snapshot(undated))

        rescued = await rungs._wayback_rung(
            None, self._URL, self._blocked(self._URL, 403), host_sems={}, ctx=self._rung_ctx()
        )

        assert rescued is None

    @pytest.mark.asyncio
    async def test_an_unreadable_capture_declines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A capture whose body extracts to nothing (a JS-wall shell) is not served."""
        self._public(monkeypatch)
        cap = "https://web.archive.org/web/20260101000000id_/https://www.bls.gov/wsp/"
        self._serve_the_archive(monkeypatch, self._snapshot(cap, status="js_wall"))

        rescued = await rungs._wayback_rung(
            None, self._URL, self._blocked(self._URL, 403), host_sems={}, ctx=self._rung_ctx()
        )

        assert rescued is None

    @pytest.mark.asyncio
    async def test_a_capture_that_wraps_a_platform_page_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A capture of a metaculus.com page presents web.archive.org as its host but is refused on
        the re-guard of the inner URL (a question quoting itself)."""
        self._public(monkeypatch)
        platform = "https://www.metaculus.com/questions/1/"
        cap = f"https://web.archive.org/web/20260101000000id_/{platform}"
        self._serve_the_archive(monkeypatch, self._snapshot(cap), page=platform)

        rescued = await rungs._wayback_rung(
            None, platform, self._blocked(platform, 403), host_sems={}, ctx=self._rung_ctx()
        )

        assert rescued is None

    @pytest.mark.asyncio
    async def test_the_served_capture_is_windowed_to_the_fetch_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The archived body is served through the same 8,000-char window every fetch uses (and the
        snapshot GET goes through the shared direct fetch, so the 5 MiB body cap applies)."""
        self._public(monkeypatch)
        cap = "https://web.archive.org/web/20250401000000id_/https://www.bls.gov/wsp/"
        long_body = "x" * (agentic_tools._FETCH_WINDOW_CHARS + 5000)
        self._serve_the_archive(monkeypatch, self._snapshot(cap, text=long_body))

        outcome = await agentic_tools.fetch(self._URL, ctx=self._question_ctx())

        assert outcome.method == "wayback"
        assert outcome.truncated is True
        assert "truncated at" in outcome.content_markdown


class TestGapFillV2DerivedApiOnEmptyRender:
    """Item C: when a render's DOM extracts nothing, serve the largest same-publisher JSON feed the
    render already captured instead of returning empty. 33 of 80 rendered reads served under 500
    chars; 61 never-read dashboard URLs (fetch-gap inventory, 2026-09-09)."""

    _URL = "https://dashboard.example.gov/tracker"

    @staticmethod
    def _page(json_responses: tuple) -> rendered_fetch.RenderedPage:
        return rendered_fetch.RenderedPage(
            url=TestGapFillV2DerivedApiOnEmptyRender._URL,
            content_type="text/html",
            html="<html><body></body></html>",
            json_responses=json_responses,
            final_url=TestGapFillV2DerivedApiOnEmptyRender._URL,
        )

    def _empty_dom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "metaculus_bot.research.fetch_ladder.classify._extract_main_text", MagicMock(return_value=None)
        )
        monkeypatch.setattr("metaculus_bot.research.fetch_ladder.guard._sem_for_host", lambda *_: asyncio.Semaphore(1))
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

    def _walled(self) -> FetchResult:
        """The direct read the browser rung fires on: a 200 that carried no text."""
        return _direct("js_wall", url=self._URL, escalate_rendered=True)

    def _rung_ctx(self) -> context.LadderContext:
        return context.LadderContext(policy=GAP_FILL_FETCH_POLICY, host_sems=agentic_tools._FETCH_HOST_SEMAPHORES)

    def test_the_derived_api_serve_earns_the_fetched_tier(self) -> None:
        assert _method_to_tier("derived_api") == "fetched"

    @pytest.mark.asyncio
    async def test_an_empty_render_serves_the_largest_harvested_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._empty_dom(monkeypatch)
        feeds = (
            rendered_fetch.HarvestedJson(url="https://dashboard.example.gov/config", body=b'{"f": 1}'),
            rendered_fetch.HarvestedJson(
                url="https://dashboard.example.gov/api/data", body=b'{"cases": 1240, "as_of": "2026-08-31"}'
            ),
        )
        monkeypatch.setattr(rungs, "render_page", _fake_render(self._page(feeds), []))
        _serve_direct(monkeypatch, {self._URL: self._walled()})

        outcome = await agentic_tools.fetch(self._URL)

        assert outcome.status == "ok"
        assert outcome.method == "derived_api"
        assert '"cases": 1240' in outcome.content_markdown
        assert '{"f": 1}' not in outcome.content_markdown  # the smaller config feed is not the one served
        assert "data feed" in outcome.content_markdown
        assert "https://dashboard.example.gov/api/data" in outcome.content_markdown

    @pytest.mark.asyncio
    async def test_an_empty_render_with_no_harvested_json_still_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._empty_dom(monkeypatch)
        monkeypatch.setattr(rungs, "render_page", _fake_render(self._page(()), []))

        result = await rungs._rendered_rung(self._URL, self._walled(), {}, self._rung_ctx())

        assert result is None, "a fruitless render with no feed to harvest rescues nothing"
        assert derived_api.endpoint_for(self._URL) is None

    @pytest.mark.asyncio
    async def test_fetch_preserves_the_derived_api_method_through_escalation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rendered escalation now keeps the rung's own method, so a derived-api serve is not
        relabelled `rendered` on the way to the driver."""
        _serve_direct(monkeypatch, {self._URL: self._walled()})
        _serve_rendered(
            monkeypatch,
            replace(_direct("success", url=self._URL, text='[feed lead]\n\n{"cases": 1240}'), route="derived_api"),
        )

        outcome = await agentic_tools.fetch(self._URL)

        assert outcome.method == "derived_api"
        assert '{"cases": 1240}' in outcome.content_markdown

    @pytest.mark.asyncio
    async def test_a_remembered_endpoint_is_gotten_for_a_second_same_host_url_before_render(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A render on one dashboard page supplies the free feed for its host's next page."""
        first_url = self._URL
        second_url = "https://dashboard.example.gov/detail"
        endpoint_url = "https://dashboard.example.gov/api/data"
        self._empty_dom(monkeypatch)

        feeds = (rendered_fetch.HarvestedJson(url=endpoint_url, body=b'{"cases": 1240, "as_of": "2026-08-31"}'),)
        renders: list[dict[str, object]] = []
        monkeypatch.setattr(rungs, "render_page", _fake_render(self._page(feeds), renders))
        asked = _serve_direct(
            monkeypatch,
            {
                first_url: replace(self._walled(), url=first_url),
                second_url: replace(self._walled(), url=second_url),
                endpoint_url: _direct(
                    "success",
                    url=endpoint_url,
                    text='{"cases": 1240, "as_of": "2026-08-31"}',
                    http_status=200,
                    content_type="application/json",
                ),
            },
        )

        first = await agentic_tools.fetch(first_url)
        second = await agentic_tools.fetch(second_url)

        assert first.method == "derived_api"
        assert second.method == "derived_api"
        assert asked == [first_url, second_url, endpoint_url]
        assert [call["url"] for call in renders] == [first_url]

    @pytest.mark.asyncio
    async def test_render_memo_scopes_stay_separate_between_caller_presets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A gap-fill empty render cannot suppress the resolution-source render for the same URL."""
        self._empty_dom(monkeypatch)
        scopes: list[str] = []
        page = self._page(())

        async def _recording_render(
            url: str,
            *,
            memo_scope: str,
            host_gate: Any,
            goto_timeout_ms: int,
            deadline_monotonic_s: float | None = None,
            harvest_json: bool = False,
        ) -> rendered_fetch.RenderedPage:
            del url, host_gate, goto_timeout_ms, deadline_monotonic_s, harvest_json
            scopes.append(memo_scope)
            return page

        monkeypatch.setattr(rungs, "render_page", _recording_render)
        direct = self._walled()

        gap_fill = await rungs._rendered_rung(
            self._URL,
            direct,
            agentic_tools._FETCH_HOST_SEMAPHORES,
            context.LadderContext(policy=GAP_FILL_FETCH_POLICY),
        )
        resolution_source = await rungs._rendered_rung(
            self._URL,
            direct,
            agentic_tools._FETCH_HOST_SEMAPHORES,
            context.LadderContext(policy=RESOLUTION_SOURCE_POLICY),
        )

        assert gap_fill is None
        assert resolution_source is None
        assert scopes == ["gap_fill_v2", "resolution_source"]
        assert GAP_FILL_FETCH_POLICY.render_memo_scope != RESOLUTION_SOURCE_POLICY.render_memo_scope
        assert rendered_fetch.rendered_to_nothing(self._URL, memo_scope="gap_fill_v2") is True
        assert rendered_fetch.rendered_to_nothing(self._URL, memo_scope="resolution_source") is True
