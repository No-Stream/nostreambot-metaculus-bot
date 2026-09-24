"""Shared ladder behavior for bounded archives, workbooks, Word files, and raster images."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from metaculus_bot.research.agentic import ladder_adapter
from metaculus_bot.research.fetch_ladder import classify, guard, run_cache, rungs, verdict
from metaculus_bot.research.fetch_ladder.context import LadderContext
from metaculus_bot.research.fetch_ladder.ladder import fetch_url
from metaculus_bot.research.fetch_ladder.policy import GAP_FILL_FETCH_POLICY, RESOLUTION_SOURCE_POLICY, LadderPolicy
from metaculus_bot.research.fetch_ladder.verdict import PageExtraction
from metaculus_bot.research.image_leads import ImageLead
from metaculus_bot.research.resolution_fetch_result import FetchResult, FetchRoute
from metaculus_bot.research.source_documents import ParsedSource, SourceMember, SourceReadError, SourceSection
from tests.resolution_source_fakes import FakeResponse, FakeSession, _impersonated

_URL = "https://example.com/report.zip"
_FINAL_URL = "https://cdn.example.com/report.zip"


@pytest.fixture(autouse=True)
def _public_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "is_public_http_url", AsyncMock(return_value=True))


def _archive() -> ParsedSource:
    return ParsedSource(
        kind="archive",
        sections=(
            SourceSection(member="background.txt", sheet=None, text="Background material only."),
            SourceSection(
                member="results.tsv",
                sheet=None,
                text="month\thospitalizations\nAugust\t922\nSeptember\t947",
                rows=3,
                columns=2,
            ),
        ),
        members=(
            SourceMember(name="background.txt", size_bytes=25, readable=True, kind="text"),
            SourceMember(name="results.tsv", size_bytes=56, readable=True, kind="text"),
        ),
    )


@pytest.mark.asyncio
async def test_archive_is_inventory_for_gap_fill_and_query_excerpt_for_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse_threads: list[int] = []
    presentation_threads: list[int] = []
    original_present = run_cache.SourceRead.present

    def parse_source(_body: bytes, _content_type: str) -> ParsedSource:
        parse_threads.append(threading.get_ident())
        return _archive()

    def present_source(
        self: run_cache.SourceRead,
        policy: LadderPolicy,
        *,
        query: str,
        route: FetchRoute,
        now: datetime,
    ) -> FetchResult:
        presentation_threads.append(threading.get_ident())
        return original_present(self, policy, query=query, route=route, now=now)

    monkeypatch.setattr(classify, "parse_source", parse_source)
    monkeypatch.setattr(run_cache.SourceRead, "present", present_source)
    monkeypatch.setattr(verdict, "is_local_source", lambda _body, _content_type: True)
    session = FakeSession(
        {
            _URL: [
                FakeResponse(200, body=b"PK fake zip", content_type="application/zip"),
                FakeResponse(200, body=b"PK fake zip", content_type="application/zip"),
            ]
        }
    )

    gap_ctx = LadderContext(query="hospitalizations", session=session, host_sems={})
    gap_fill = await fetch_url(
        _URL,
        policy=GAP_FILL_FETCH_POLICY,
        ctx=gap_ctx,
    )
    # Force the second policy through fresh classification; cache replay is covered separately.
    run_cache.clear()
    resolution = await fetch_url(
        _URL,
        policy=RESOLUTION_SOURCE_POLICY,
        ctx=LadderContext(query="September hospitalizations", session=session, host_sems={}),
    )

    assert gap_fill.status == "success"
    assert gap_fill.local_kind == "archive"
    assert gap_ctx.local_read_receipt.encountered is True
    assert gap_fill.navigation_only is True
    assert "results.tsv" in gap_fill.text
    assert "September\t947" not in gap_fill.text
    assert resolution.status == "success"
    assert resolution.local_kind == "archive"
    assert resolution.navigation_only is False
    assert "September\t947" in resolution.text
    assert len(resolution.text) <= 6_000
    assert parse_threads
    assert all(thread_id != threading.get_ident() for thread_id in parse_threads)
    assert presentation_threads
    assert all(thread_id != threading.get_ident() for thread_id in presentation_threads)


@pytest.mark.asyncio
async def test_local_source_classification_matches_direct_and_impersonated_transports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(classify, "parse_source", lambda _body, _content_type: _archive())
    monkeypatch.setattr(verdict, "is_local_source", lambda _body, _content_type: True)
    body = b"PK fake zip"
    pending = await classify._classify_body(
        body,
        _URL,
        "application/zip",
        LadderContext(policy=GAP_FILL_FETCH_POLICY),
        http_status=200,
    )
    assert isinstance(pending, classify._PendingSource)
    direct = await classify._finish_source(pending, LadderContext(policy=GAP_FILL_FETCH_POLICY))
    impersonated = await rungs._impersonated_body_outcome(
        _impersonated(200, body=body, content_type="application/zip", url=_URL),
        LadderContext(policy=GAP_FILL_FETCH_POLICY),
    )

    assert direct == impersonated
    assert direct.local_kind == "archive"
    assert direct.navigation_only is True


@pytest.mark.asyncio
async def test_expected_local_parse_refusal_is_terminal_and_never_reaches_paid_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_body: bytes, _content_type: str) -> ParsedSource:
        raise SourceReadError("expanded_size_limit")

    paid_reader = AsyncMock(side_effect=AssertionError("local refusal must not reach url_context"))
    monkeypatch.setattr(classify, "parse_source", refuse)
    monkeypatch.setattr(verdict, "is_local_source", lambda _body, _content_type: True)
    monkeypatch.setattr(rungs, "_url_context_rung", paid_reader)
    session = FakeSession({_URL: FakeResponse(200, body=b"PK zip bomb", content_type="application/zip")})

    result = await fetch_url(
        _URL,
        policy=RESOLUTION_SOURCE_POLICY,
        ctx=LadderContext(query="hospitalizations", session=session, host_sems={}),
    )

    assert result.status == "unsupported_type"
    assert result.local_kind == "archive"
    assert result.local_read_refused is True
    assert result.navigation_only is False
    assert result.text == ""
    paid_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_archive_cache_replays_for_both_policies_and_exposes_final_url_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = _archive()
    monkeypatch.setattr(classify, "parse_source", lambda _body, _content_type: parsed)
    monkeypatch.setattr(verdict, "is_local_source", lambda _body, _content_type: True)
    session = FakeSession(
        {
            _URL: FakeResponse(302, headers={"Location": _FINAL_URL}),
            _FINAL_URL: FakeResponse(200, body=b"PK fake zip", content_type="application/zip"),
        }
    )

    initial = await fetch_url(
        _URL,
        policy=GAP_FILL_FETCH_POLICY,
        ctx=LadderContext(query="hospitalizations", session=session, host_sems={}),
    )
    cached = await fetch_url(
        _URL,
        policy=RESOLUTION_SOURCE_POLICY,
        ctx=LadderContext(query="September hospitalizations", session=session, host_sems={}),
    )

    assert initial.navigation_only is True
    assert cached.cache_hit is True
    assert cached.navigation_only is False
    assert "September\t947" in cached.text
    assert run_cache.local_source_for(_URL) is parsed
    assert run_cache.local_source_for(_FINAL_URL) is parsed
    assert session.requested == [_URL, _FINAL_URL]


@pytest.mark.asyncio
async def test_no_matching_resolution_query_still_caches_complete_parse_for_next_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse_calls = 0

    def parse_source(_body: bytes, _content_type: str) -> ParsedSource:
        nonlocal parse_calls
        parse_calls += 1
        return _archive()

    monkeypatch.setattr(classify, "parse_source", parse_source)
    monkeypatch.setattr(verdict, "is_local_source", lambda _body, _content_type: True)
    session = FakeSession({_URL: FakeResponse(200, body=b"PK fake zip", content_type="application/zip")})

    no_match = await fetch_url(
        _URL,
        policy=RESOLUTION_SOURCE_POLICY,
        ctx=LadderContext(query="unrelated astronomy phrase", session=session, host_sems={}),
    )
    inventory = await fetch_url(
        _URL,
        policy=GAP_FILL_FETCH_POLICY,
        ctx=LadderContext(session=session, host_sems={}),
    )

    assert no_match.status == "no_resolving_content"
    assert inventory.status == "success"
    assert inventory.navigation_only is True
    assert inventory.cache_hit is True
    assert parse_calls == 1
    assert session.requested == [_URL]


@pytest.mark.asyncio
async def test_inventory_method_stays_untiered_on_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classify, "parse_source", lambda _body, _content_type: _archive())
    monkeypatch.setattr(verdict, "is_local_source", lambda _body, _content_type: True)
    session = FakeSession({_URL: FakeResponse(200, body=b"PK fake zip", content_type="application/zip")})

    await fetch_url(
        _URL,
        policy=GAP_FILL_FETCH_POLICY,
        ctx=LadderContext(session=session, host_sems={}),
    )
    cached = await fetch_url(
        _URL,
        policy=GAP_FILL_FETCH_POLICY,
        ctx=LadderContext(session=session, host_sems={}),
    )
    plain = ladder_adapter.as_plain_result(cached, requested_url=_URL)

    assert cached.cache_hit is True
    assert plain.method == ladder_adapter.LOCAL_NAVIGATION_METHOD
    assert plain.navigation_only is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("image/png", b"\x89PNG\r\n\x1a\nbody"),
        ("application/octet-stream", b"\xff\xd8\xffbody"),
        ("image/gif", b"GIF89abody"),
        ("image/webp", b"RIFF\x10\x00\x00\x00WEBPbody"),
        ("image/bmp", b"BMbody"),
    ],
)
async def test_gap_fill_caches_supported_raster_bytes_for_same_agent_viewing(
    content_type: str,
    body: bytes,
) -> None:
    session = FakeSession({_URL: FakeResponse(200, body=body, content_type=content_type)})

    result = await fetch_url(
        _URL,
        policy=replace(GAP_FILL_FETCH_POLICY, rungs_enabled=frozenset()),
        ctx=LadderContext(session=session, host_sems={}),
    )

    image = run_cache.image_source_for(_URL)
    assert result.status == "success"
    assert result.local_kind == "image"
    assert result.navigation_only is True
    assert image is not None
    assert image.url == _URL
    assert image.body == body
    assert image.content_type == content_type


@pytest.mark.asyncio
async def test_resolution_still_refuses_raster_and_svg_is_an_explicit_local_refusal() -> None:
    raster_url = "https://example.com/chart.png"
    svg_url = "https://example.com/chart.svg"
    session = FakeSession(
        {
            raster_url: FakeResponse(200, body=b"\x89PNG\r\n\x1a\nbody", content_type="image/png"),
            svg_url: FakeResponse(200, body=b"<svg><text>947</text></svg>", content_type="image/svg+xml"),
        }
    )

    raster = await fetch_url(
        raster_url,
        policy=replace(RESOLUTION_SOURCE_POLICY, rungs_enabled=frozenset()),
        ctx=LadderContext(session=session, host_sems={}),
    )
    svg = await fetch_url(
        svg_url,
        policy=replace(GAP_FILL_FETCH_POLICY, rungs_enabled=frozenset()),
        ctx=LadderContext(session=session, host_sems={}),
    )

    assert raster.status == "unsupported_type"
    assert raster.local_kind is None
    assert run_cache.image_source_for(raster_url) is None
    assert svg.status == "unsupported_type"
    assert svg.local_read_refused is True
    assert run_cache.image_source_for(svg_url) is None


@pytest.mark.asyncio
async def test_declared_raster_without_supported_magic_is_refused_without_caching() -> None:
    session = FakeSession({_URL: FakeResponse(200, body=b"not a png", content_type="image/png")})

    result = await fetch_url(
        _URL,
        policy=replace(GAP_FILL_FETCH_POLICY, rungs_enabled=frozenset()),
        ctx=LadderContext(session=session, host_sems={}),
    )

    assert result.status == "unsupported_type"
    assert result.local_kind == "image"
    assert result.local_read_refused is True
    assert run_cache.image_source_for(_URL) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [GAP_FILL_FETCH_POLICY, RESOLUTION_SOURCE_POLICY])
async def test_octet_stream_routes_to_text_only_for_strict_utf8_json_and_replays_from_cache(
    policy: LadderPolicy,
) -> None:
    valid_url = "https://example.com/feed.bin"
    invalid_json_url = "https://example.com/not-json.bin"
    invalid_utf8_url = "https://example.com/bad-utf8.bin"
    body = b'{"label":"<hospitalizations>","value":947}'
    session = FakeSession(
        {
            valid_url: FakeResponse(200, body=body, content_type="application/octet-stream"),
            invalid_json_url: FakeResponse(200, body=b"value=947", content_type="application/octet-stream"),
            invalid_utf8_url: FakeResponse(
                200,
                body=b'{"label":"\xff","value":947}',
                content_type="application/octet-stream",
            ),
        }
    )
    direct_only = replace(policy, rungs_enabled=frozenset())

    valid = await fetch_url(
        valid_url,
        policy=direct_only,
        ctx=LadderContext(session=session, host_sems={}),
    )
    cached = await fetch_url(
        valid_url,
        policy=replace(
            RESOLUTION_SOURCE_POLICY if policy is GAP_FILL_FETCH_POLICY else GAP_FILL_FETCH_POLICY,
            rungs_enabled=frozenset(),
        ),
        ctx=LadderContext(session=session, host_sems={}),
    )
    invalid_json = await fetch_url(
        invalid_json_url,
        policy=direct_only,
        ctx=LadderContext(session=session, host_sems={}),
    )
    invalid_utf8 = await fetch_url(
        invalid_utf8_url,
        policy=direct_only,
        ctx=LadderContext(session=session, host_sems={}),
    )

    assert valid.status == "success"
    assert valid.text == body.decode()
    assert cached.status == "success"
    assert cached.text == body.decode()
    assert cached.cache_hit is True
    assert invalid_json.status == "unsupported_type"
    assert invalid_utf8.status == "unsupported_type"


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["text/tab-separated-values", "text/tsv"])
async def test_standard_tsv_mime_is_text_without_html_stripping(content_type: str) -> None:
    body = b"label\tvalue\n<hospitalizations>\t947\n"
    session = FakeSession({_URL: FakeResponse(200, body=body, content_type=content_type)})

    result = await fetch_url(
        _URL,
        policy=replace(GAP_FILL_FETCH_POLICY, rungs_enabled=frozenset()),
        ctx=LadderContext(session=session, host_sems={}),
    )

    assert result.status == "success"
    assert result.text == body.decode()


def test_retained_byte_eviction_counts_aliases_once(monkeypatch: pytest.MonkeyPatch) -> None:
    first_url = "https://example.com/first.png"
    first_final = "https://cdn.example.com/first.png"
    second_url = "https://example.com/second.png"
    monkeypatch.setattr(run_cache, "LOCAL_SOURCE_CACHE_MAX_BYTES", 10)
    first = run_cache.ImageRead(
        source=run_cache.ImageSource(first_final, b"123456", "image/png"),
        http_status=200,
    )
    second = run_cache.ImageRead(
        source=run_cache.ImageSource(second_url, b"abcdef", "image/png"),
        http_status=200,
    )

    run_cache.put(first_url, first, route="direct")
    assert run_cache.image_source_for(first_url) is first.source
    assert run_cache.image_source_for(first_final) is first.source
    run_cache.put(second_url, second, route="direct")

    assert run_cache.image_source_for(first_url) is None
    assert run_cache.image_source_for(first_final) is None
    assert run_cache.image_source_for(second_url) is second.source


def test_cache_entry_bound_counts_a_redirect_alias_as_one_entry() -> None:
    first_requested = "https://example.com/0.png"
    first_final = "https://cdn.example.com/0.png"
    first = run_cache.ImageRead(
        source=run_cache.ImageSource(first_final, b"0", "image/png"),
        http_status=200,
    )
    run_cache.put(first_requested, first, route="direct")
    for index in range(1, 50):
        url = f"https://example.com/{index}.png"
        run_cache.put(
            url,
            run_cache.ImageRead(run_cache.ImageSource(url, b"x", "image/png"), http_status=200),
            route="direct",
        )

    assert run_cache.image_source_for(first_requested) is first.source
    assert run_cache.image_source_for(first_final) is first.source

    last_url = "https://example.com/50.png"
    run_cache.put(
        last_url,
        run_cache.ImageRead(run_cache.ImageSource(last_url, b"x", "image/png"), http_status=200),
        route="direct",
    )
    assert run_cache.image_source_for(first_requested) is first.source
    assert run_cache.image_source_for(first_final) is first.source
    assert run_cache.image_source_for("https://example.com/1.png") is None
    assert run_cache.image_source_for(last_url) is not None


@pytest.mark.asyncio
async def test_image_leads_are_policy_neutral_in_cache_but_exposed_only_to_gap_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lead = ImageLead(url="https://example.com/chart.png", filename="chart.png", alt="Monthly chart")
    monkeypatch.setattr(classify, "extract_image_leads", lambda _html, _url: (lead,))
    monkeypatch.setattr(
        classify,
        "_extract_page_text",
        lambda *_args, **_kwargs: PageExtraction(text="Report content " * 50),
    )
    session = FakeSession({_URL: FakeResponse(200, body=b"<html><p>report</p></html>", content_type="text/html")})

    resolution = await fetch_url(
        _URL,
        policy=RESOLUTION_SOURCE_POLICY,
        ctx=LadderContext(session=session, host_sems={}),
    )
    gap_fill = await fetch_url(
        _URL,
        policy=GAP_FILL_FETCH_POLICY,
        ctx=LadderContext(session=session, host_sems={}),
    )

    assert resolution.image_leads == ()
    assert gap_fill.image_leads == (lead,)
    assert ladder_adapter.as_plain_result(gap_fill, requested_url=_URL).image_leads == (lead,)
    assert session.requested == [_URL]


@pytest.mark.asyncio
async def test_cancelled_source_parse_holds_parser_slot_until_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    gate = asyncio.Semaphore(1)

    def slow_parse(_body: bytes, _content_type: str) -> ParsedSource:
        started.set()
        assert release.wait(timeout=2)
        return _archive()

    monkeypatch.setattr(classify, "parse_source", slow_parse)
    monkeypatch.setattr(classify, "pdf_parse_semaphore", lambda: gate)
    pending = classify._PendingSource(_URL, b"PK fake zip", 200, "application/zip", "archive")
    task = asyncio.create_task(classify._finish_source(pending, LadderContext(policy=GAP_FILL_FETCH_POLICY)))
    assert await asyncio.to_thread(started.wait, 1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gate.locked()

    release.set()
    for _ in range(100):
        if not gate.locked():
            break
        await asyncio.sleep(0.01)
    assert not gate.locked()


@pytest.mark.asyncio
async def test_source_parse_deadline_returns_while_worker_keeps_parser_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    gate = asyncio.Semaphore(1)

    def slow_parse(_body: bytes, _content_type: str) -> ParsedSource:
        started.set()
        assert release.wait(timeout=2)
        return _archive()

    monkeypatch.setattr(classify, "parse_source", slow_parse)
    monkeypatch.setattr(classify, "pdf_parse_semaphore", lambda: gate)
    pending = classify._PendingSource(_URL, b"PK fake zip", 200, "application/zip", "archive")
    policy = replace(GAP_FILL_FETCH_POLICY, total_wall_s=0.05, rung_wall_margin_s=0.0)
    ctx = LadderContext(policy=policy)

    ctx.local_read_receipt.encountered = True
    result = await classify._finish_source(pending, ctx)

    assert started.is_set()
    assert result.status == "unsupported_type"
    assert result.local_read_refused is True
    assert ctx.local_read_receipt.encountered is True
    assert gate.locked()
    release.set()
    for _ in range(100):
        if not gate.locked():
            break
        await asyncio.sleep(0.01)
    assert not gate.locked()


@pytest.mark.asyncio
async def test_cancelled_cached_source_presentation_holds_gate_and_bounds_followup_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    gate = asyncio.Semaphore(1)
    calls = 0

    def slow_present(
        self: run_cache.SourceRead,
        policy: LadderPolicy,
        *,
        query: str,
        route: FetchRoute,
        now: datetime,
    ) -> FetchResult:
        del self, policy, query, route, now
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=2)
        return FetchResult(_URL, "success", "finished", 200, "application/zip")

    monkeypatch.setattr(run_cache, "pdf_parse_semaphore", lambda: gate)
    monkeypatch.setattr(run_cache.SourceRead, "present", slow_present)
    run_cache.put(
        _URL,
        run_cache.SourceRead(_URL, 200, "application/zip", _archive()),
        route="direct",
    )
    first = asyncio.create_task(
        run_cache.get(
            _URL,
            policy=GAP_FILL_FETCH_POLICY,
            query="",
            now=LadderContext().now,
            budget_s=2,
        )
    )
    assert await asyncio.to_thread(started.wait, 1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert gate.locked()

    with pytest.raises(TimeoutError):
        await run_cache.get(
            _URL,
            policy=GAP_FILL_FETCH_POLICY,
            query="",
            now=LadderContext().now,
            budget_s=0.02,
        )
    assert calls == 1

    release.set()
    for _ in range(100):
        if not gate.locked():
            break
        await asyncio.sleep(0.01)
    assert not gate.locked()
