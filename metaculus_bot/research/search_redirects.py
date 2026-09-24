"""Resolve Google's self-cited search result redirect URLs."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from urllib.parse import urlsplit

import aiohttp

from metaculus_bot.research.http_fetch import REDIRECT_STATUSES, build_session

SEARCH_REDIRECT_HOST = "vertexaisearch.cloud.google.com"
SEARCH_REDIRECT_PATH_PREFIX = "/grounding-api-redirect/"


def is_search_redirect(url: str) -> bool:
    """Return whether ``url`` has Google's search redirect host and path."""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname == SEARCH_REDIRECT_HOST
        and parsed.path.startswith(SEARCH_REDIRECT_PATH_PREFIX)
    )


def _absolute_http_url(url: str | None) -> str | None:
    if url is None:
        return None
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return url


async def _resolve_one(session: aiohttp.ClientSession, url: str) -> tuple[str, str] | None:
    try:
        async with session.get(url, allow_redirects=False) as response:
            if response.status not in REDIRECT_STATUSES:
                return None
            location = _absolute_http_url(response.headers.get("Location"))
            if location is None:
                return None
            return url, location
    except (aiohttp.ClientError, TimeoutError):
        # asyncio.TimeoutError is an alias of the built-in TimeoutError on Python 3.12+.
        return None


async def resolve_search_redirects(urls: Sequence[str], *, timeout_s: float) -> dict[str, str]:
    """Resolve each distinct search redirect URL within one shared request wall."""
    unique_urls = tuple(dict.fromkeys(urls))
    if not unique_urls or timeout_s <= 0:
        return {}

    async with build_session(timeout_s=timeout_s) as session:
        resolved = await asyncio.gather(*(_resolve_one(session, url) for url in unique_urls))

    return dict(result for result in resolved if result is not None)
