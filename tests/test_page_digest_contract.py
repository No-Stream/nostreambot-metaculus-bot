"""Outcome and cache contracts for the long-page digest path."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime

import pytest

import metaculus_bot.research.page_digest as page_digest_module
from metaculus_bot.constants import RESOLUTION_SOURCE_WALL_TIMEOUT
from metaculus_bot.research.fetch_ladder import run_cache
from metaculus_bot.research.fetch_ladder.digest import LadderDigest
from metaculus_bot.research.fetch_ladder.policy import RESOLUTION_SOURCE_POLICY
from metaculus_bot.research.fetch_ladder.verdict import PageExtraction
from metaculus_bot.research.page_digest import digest_page
from tests.test_page_digest import PAGE, QUERY, ScriptedLlm, _passages_json

_URL = "https://report.example.gov/employment"


@pytest.mark.asyncio
async def test_valid_empty_extractor_answer_is_an_explicit_empty_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = ScriptedLlm(_passages_json())
    monkeypatch.setattr(page_digest_module, "build_llm_with_openrouter_fallback", lambda **_kwargs: llm)

    digest = await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

    assert digest.outcome == "valid_empty"
    assert digest.passages == []
    assert digest.evidence_passages == ()
    assert digest.fallback_used is False
    assert digest.scope == "prefiltered_windows"


@pytest.mark.asyncio
async def test_html_empty_answer_is_not_replaced_with_the_ordinary_page() -> None:
    async def empty_digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        del text, query, budget_seconds
        return LadderDigest(
            passages=[],
            passages_returned=0,
            passages_grounded=0,
            fallback_used=False,
            method="llm_extractive",
            outcome="valid_empty",
        )

    artifact = run_cache.HtmlRead(
        url=_URL,
        http_status=200,
        content_type="text/html",
        extraction=PageExtraction(text=PAGE),
        chart_block="",
        datawrapper_charts=(),
        unreadable_embeds=(),
        links=(),
        routing_body=b"<html>",
    )

    result = await artifact.present_html(
        replace(RESOLUTION_SOURCE_POLICY, digest=empty_digest),
        query=QUERY,
        route="direct",
        now=datetime.now(UTC),
        budget_seconds=RESOLUTION_SOURCE_POLICY.total_wall_s,
    )

    assert result is not None
    assert result.status == "no_resolving_content"
    assert result.status_reason == "no_matching_passage"
    assert result.text == ""
    assert result.evidence_passages == []
    assert result.passages_returned == 0
    assert result.fallback_used is False


@pytest.mark.asyncio
async def test_same_query_digest_is_memoized_but_another_query_is_independent() -> None:
    queries: list[str] = []

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        del text, budget_seconds
        queries.append(query)
        return LadderDigest(
            passages=[f"Selected for {query}."],
            passages_returned=1,
            passages_grounded=1,
            fallback_used=False,
            method="llm_extractive",
            outcome="answered",
        )

    artifact = run_cache.HtmlRead(
        url=_URL,
        http_status=200,
        content_type="text/html",
        extraction=PageExtraction(text=PAGE),
        chart_block="",
        datawrapper_charts=(),
        unreadable_embeds=(),
        links=(),
        routing_body=b"<html>",
    )
    policy = replace(RESOLUTION_SOURCE_POLICY, digest=digest)

    first = await artifact.present_html(
        policy, query="first query", route="direct", now=datetime.now(UTC), budget_seconds=20.0
    )
    repeated = await artifact.present_html(
        policy, query="first query", route="direct", now=datetime.now(UTC), budget_seconds=20.0
    )
    independent = await artifact.present_html(
        policy, query="second query", route="direct", now=datetime.now(UTC), budget_seconds=20.0
    )

    assert queries == ["first query", "second query"]
    assert first is not None
    assert repeated is not None
    assert independent is not None
    assert "Selected for first query." in first.text
    assert repeated.text == first.text
    assert "Selected for second query." in independent.text


@pytest.mark.asyncio
async def test_digest_memo_is_scoped_to_url_content_query_and_digest_function() -> None:
    queries: list[str] = []

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        del budget_seconds
        queries.append(f"{text}:{query}")
        return LadderDigest(
            passages=[f"Selected for {query}."],
            passages_returned=1,
            passages_grounded=1,
            fallback_used=False,
            method="llm_extractive",
            outcome="answered",
        )

    first = await run_cache.memoized_digest(_URL, PAGE, "first query", digest, budget_seconds=20.0)
    repeated = await run_cache.memoized_digest(_URL, PAGE, "first query", digest, budget_seconds=20.0)
    other_query = await run_cache.memoized_digest(_URL, PAGE, "second query", digest, budget_seconds=20.0)
    other_text = await run_cache.memoized_digest(_URL, f"{PAGE} revised", "first query", digest, budget_seconds=20.0)
    other_url = await run_cache.memoized_digest(
        "https://other.example.gov/page", PAGE, "first query", digest, budget_seconds=20.0
    )

    assert queries == [
        f"{PAGE}:first query",
        f"{PAGE}:second query",
        f"{PAGE} revised:first query",
        f"{PAGE}:first query",
    ]
    assert repeated is first
    assert other_query is not first
    assert other_text is not first
    assert other_url is not first


@pytest.mark.parametrize("outcome", ["api_error", "timeout", "empty_completion", "insufficient_budget"])
@pytest.mark.asyncio
async def test_transient_digest_outcomes_are_not_memoized(outcome: str) -> None:
    calls = 0

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        nonlocal calls
        del text, query, budget_seconds
        calls += 1
        return LadderDigest(
            passages=["Fallback passage."],
            passages_returned=0,
            passages_grounded=0,
            fallback_used=True,
            method="bm25",
            outcome=outcome,  # type: ignore[arg-type]
        )

    for _ in range(2):
        await run_cache.memoized_digest(_URL, PAGE, QUERY, digest, budget_seconds=20.0)

    assert calls == 2


@pytest.mark.asyncio
async def test_valid_empty_digest_outcome_is_memoized() -> None:
    calls = 0

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        nonlocal calls
        del text, query, budget_seconds
        calls += 1
        return LadderDigest([], 0, 0, False, "llm_extractive", outcome="valid_empty")

    first = await run_cache.memoized_digest(_URL, PAGE, QUERY, digest, budget_seconds=20.0)
    repeated = await run_cache.memoized_digest(_URL, PAGE, QUERY, digest, budget_seconds=20.0)

    assert calls == 1
    assert first is repeated


@pytest.mark.asyncio
async def test_clearing_the_run_cache_cancels_a_pending_digest_without_masking_cancellation() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        del text, query, budget_seconds
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    pending = asyncio.create_task(run_cache.memoized_digest(_URL, PAGE, QUERY, digest, budget_seconds=20.0))
    await started.wait()
    run_cache.clear()

    with pytest.raises(asyncio.CancelledError):
        await pending

    assert cancelled.is_set()
    assert not run_cache._DIGEST_WAITERS


@pytest.mark.asyncio
async def test_concurrent_same_query_shares_one_digest_call() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        nonlocal calls
        del text, budget_seconds
        calls += 1
        started.set()
        await release.wait()
        return LadderDigest(
            passages=["One selected passage."],
            passages_returned=1,
            passages_grounded=1,
            fallback_used=False,
            method="llm_extractive",
            outcome="answered",
        )

    artifact = run_cache.HtmlRead(
        url=_URL,
        http_status=200,
        content_type="text/html",
        extraction=PageExtraction(text=PAGE),
        chart_block="",
        datawrapper_charts=(),
        unreadable_embeds=(),
        links=(),
        routing_body=b"<html>",
    )
    policy = replace(RESOLUTION_SOURCE_POLICY, digest=digest)
    first = asyncio.create_task(
        artifact.present_html(policy, query="shared query", route="direct", now=datetime.now(UTC), budget_seconds=20.0)
    )
    await started.wait()
    second = asyncio.create_task(
        artifact.present_html(policy, query="shared query", route="direct", now=datetime.now(UTC), budget_seconds=20.0)
    )
    await _wait_for_digest_waiters(artifact, "shared query", digest, 2)
    release.set()

    results = await asyncio.gather(first, second)

    assert calls == 1
    assert all(result is not None and "One selected passage." in result.text for result in results)


@pytest.mark.asyncio
async def test_cancelled_digest_waiter_keeps_shared_call_for_its_remaining_waiter() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()
    calls = 0

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        nonlocal calls
        del text, query, budget_seconds
        calls += 1
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return LadderDigest([], 0, 0, False, "llm_extractive")

    artifact = run_cache.HtmlRead(
        url=_URL,
        http_status=200,
        content_type="text/html",
        extraction=PageExtraction(text=PAGE),
        chart_block="",
        datawrapper_charts=(),
        unreadable_embeds=(),
        links=(),
        routing_body=b"<html>",
    )
    policy = replace(RESOLUTION_SOURCE_POLICY, digest=digest)
    first = asyncio.create_task(
        artifact.present_html(policy, query="shared query", route="direct", now=datetime.now(UTC), budget_seconds=20.0)
    )
    await started.wait()
    second = asyncio.create_task(
        artifact.present_html(policy, query="shared query", route="direct", now=datetime.now(UTC), budget_seconds=20.0)
    )
    await _wait_for_digest_waiters(artifact, "shared query", digest, 2)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not cancelled.is_set()
    release.set()

    assert await second is not None
    assert calls == 1
    assert not cancelled.is_set()


async def _wait_for_digest_waiters(
    artifact: run_cache.HtmlRead,
    query: str,
    digest: object,
    expected: int,
) -> None:
    await asyncio.sleep(0.01)
    waiters = [
        count
        for (_, url, _, waited_query, waited_digest), count in run_cache._DIGEST_WAITERS.items()
        if url == artifact.url and waited_query == query and waited_digest is digest
    ]
    assert waiters == [expected]


@pytest.mark.asyncio
async def test_last_cancelled_digest_waiter_cancels_the_shared_call() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        del text, query, budget_seconds
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    artifact = run_cache.HtmlRead(
        url=_URL,
        http_status=200,
        content_type="text/html",
        extraction=PageExtraction(text=PAGE),
        chart_block="",
        datawrapper_charts=(),
        unreadable_embeds=(),
        links=(),
        routing_body=b"<html>",
    )
    task = asyncio.create_task(
        artifact.present_html(
            replace(RESOLUTION_SOURCE_POLICY, digest=digest),
            query="cancelled query",
            route="direct",
            now=datetime.now(UTC),
            budget_seconds=20.0,
        )
    )
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await cancelled.wait()
