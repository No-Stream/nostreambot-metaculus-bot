"""Tests for Google's self-cited search redirect resolver."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from metaculus_bot.research.search_redirects import is_search_redirect, resolve_search_redirects


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/opaque-token",
            True,
        ),
        (
            "http://vertexaisearch.cloud.google.com/grounding-api-redirect/opaque-token?x=1",
            True,
        ),
        ("https://other.example/grounding-api-redirect/opaque-token", False),
        ("https://vertexaisearch.cloud.google.com/other/opaque-token", False),
        ("https://vertexaisearch.cloud.google.com/grounding-api-redirect", False),
        ("vertexaisearch.cloud.google.com/grounding-api-redirect/opaque-token", False),
    ],
)
def test_is_search_redirect(url: str, expected: bool) -> None:
    assert is_search_redirect(url) is expected


@pytest.fixture
async def redirect_server() -> AsyncIterator[tuple[TestServer, list[str]]]:
    requested_paths: list[str] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        requested_paths.append(request.path)
        if request.path == "/redirect":
            return web.Response(status=302, headers={"Location": "https://example.com/article"})
        if request.path == "/redirect-relative":
            return web.Response(status=302, headers={"Location": "/article"})
        if request.path == "/redirect-ftp":
            return web.Response(status=302, headers={"Location": "ftp://example.com/article"})
        if request.path == "/ok":
            return web.Response(status=200, text="not read")
        if request.path == "/missing":
            return web.Response(status=404, text="not found")
        if request.path == "/slow":
            await asyncio.sleep(0.2)
            return web.Response(status=302, headers={"Location": "https://example.com/slow"})
        raise AssertionError(f"unexpected path: {request.path}")

    app = web.Application()
    app.router.add_get("/{tail:.*}", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        yield server, requested_paths
    finally:
        await server.close()


async def test_resolve_search_redirects_filters_statuses_and_deduplicates(
    redirect_server: tuple[TestServer, list[str]],
) -> None:
    server, requested_paths = redirect_server
    base_url = str(server.make_url("")).rstrip("/")
    redirect_url = f"{base_url}/redirect"

    resolved = await resolve_search_redirects(
        [
            redirect_url,
            redirect_url,
            f"{base_url}/redirect-relative",
            f"{base_url}/redirect-ftp",
            f"{base_url}/ok",
            f"{base_url}/missing",
        ],
        timeout_s=1.0,
    )

    assert resolved == {redirect_url: "https://example.com/article"}
    assert requested_paths.count("/redirect") == 1


async def test_resolve_search_redirects_treats_slow_request_as_unresolved(
    redirect_server: tuple[TestServer, list[str]],
) -> None:
    server, _ = redirect_server
    slow_url = str(server.make_url("/slow"))

    resolved = await resolve_search_redirects([slow_url], timeout_s=0.01)

    assert resolved == {}
