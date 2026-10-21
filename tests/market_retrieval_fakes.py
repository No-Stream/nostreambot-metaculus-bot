"""Shared fakes, payload fixtures and pipeline helpers for the prediction-market seam tests.

One canonical home for the fake aiohttp session, the two stubbed LLM stages, the realistic venue
payloads and the healthy-baseline handler dict. All of it used to live inside
`tests/test_prediction_market_provider.py`, which `tests/test_market_retrieval_venues.py` then
imported from — a test module serving as the fixture library for another.

Not named `test_*` on purpose: pytest must import this module without collecting it. A fixture
defined here is picked up by any test module that binds it in its own globals, and the consumers
bind them by ASSIGNMENT (`mock_question = _fakes.mock_question`) rather than by import: pyflakes
reads a same-named fixture parameter as an F811 redefinition of an import, which would otherwise
demand a per-signature suppression on every test that takes one.

All HTTP here is faked and both LLM stages are stubbed, so nothing in this module opens a socket
or bills a key.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from metaculus_bot.research import prediction_market as pmp
from metaculus_bot.research.market_retrieval import venues
from metaculus_bot.research.prediction_market import MarketMatch, MarketSnapshot, _reset_session_caches

# ---------------------------------------------------------------------------
# Test isolation: module-level caches bleed across tests when run in certain
# orders (observed 2026-05-12: a full-suite run failed 2 tests that passed
# individually). Reset every test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_provider_caches():
    _reset_session_caches()
    yield
    _reset_session_caches()


# ---------------------------------------------------------------------------
# Fake aiohttp session
# ---------------------------------------------------------------------------


class _FakeStreamContent:
    """Stub for `resp.content`: streams a payload as JSON bytes via `iter_chunked`.

    The Kalshi catalogue pull stream-parses `resp.content.iter_chunked()`; the other JSON
    endpoints go through `read_json_capped`, which gates on a `.read` attribute FakeResponse
    deliberately lacks, so they still use the `.json()` path. `raw_content` overrides the
    serialized bytes so a test can inject malformed JSON.
    """

    def __init__(self, payload: Any, raw_content: bytes | None = None):
        if raw_content is not None:
            self._data = raw_content
        elif payload is None:
            self._data = b""
        else:
            self._data = json.dumps(payload).encode()

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        step = max(1, n)
        for i in range(0, len(self._data), step):
            yield self._data[i : i + step]


class FakeResponse:
    def __init__(self, status: int, payload: Any = None, text: str | None = None, raw_content: bytes | None = None):
        self.status = status
        self._payload = payload
        # Real aiohttp serves `.text()` and `.content` from ONE body, so an error-page stub must
        # stream its text too: the non-200 log path reads a BOUNDED snippet off `.content`, and a
        # stub holding its text outside the stream would hide a regression back to whole-body
        # `resp.text()`.
        if raw_content is None and payload is None and text:
            raw_content = text.encode()
        self.content = _FakeStreamContent(payload, raw_content=raw_content)

    async def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload

    async def text(self) -> str:
        # Faithful to aiohttp: reads the WHOLE body (the memory trap the bounded snippet read
        # exists to avoid).
        chunks = [chunk async for chunk in self.content.iter_chunked(65536)]
        return b"".join(chunks).decode("utf-8", errors="replace")

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        return None


# A handler is either a ready-made response or a callable that builds one from the GET params.
_Handler = FakeResponse | Callable[[dict[str, Any]], FakeResponse]
# Per-URL value accepted by FakeSession: a single handler or a list of them. Mapping (not dict) so
# call-site literals like {url: FakeResponse(...)} are accepted (dict is invariant in its values).
_Handlers = Mapping[str, _Handler | list[_Handler]]


class FakeSession:
    """aiohttp.ClientSession replacement dispatching GETs to handlers keyed by URL prefix."""

    def __init__(self, handlers: _Handlers):
        self._handlers: dict[str, list[_Handler]] = {
            k: (v if isinstance(v, list) else [v]) for k, v in handlers.items()
        }
        self._call_counts: dict[str, int] = dict.fromkeys(handlers, 0)
        self.closed = False

    def get(self, url: str, params: dict | None = None, **_kwargs: Any) -> FakeResponse:
        for prefix, handler_list in self._handlers.items():
            if url.startswith(prefix):
                idx = min(self._call_counts[prefix], len(handler_list) - 1)
                self._call_counts[prefix] += 1
                handler = handler_list[idx]
                if callable(handler):
                    return handler(params or {})
                return handler
        raise AssertionError(f"no handler for URL {url}")

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()


POLY_URL = "https://gamma-api.polymarket.com/public-search"
MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
MANIFOLD_DETAIL_URL = "https://api.manifold.markets/v0/market"
KALSHI_EVENTS_URL = "https://api.elections.kalshi.com/trade-api/v2/events"
PREDICTIT_URL = "https://www.predictit.org/api/marketdata/all/"


# ---------------------------------------------------------------------------
# The two LLM stages, stubbed at the `build_llm_with_openrouter_fallback` seam
# ---------------------------------------------------------------------------

# The query author's shape. `parse_query_author` strips digits at parse time, so these carry
# none — a numeric synonym would silently vanish and make the test's own intent unreadable.
AUTHOR_JSON = json.dumps({"synonyms": ["orbital launch", "Starship"], "framings": ["Starship orbit"]})

RANKER_CUE = "Rank the candidates by EVIDENTIAL VALUE"
CANDIDATE_LINE_RE = re.compile(r"^\[(\d+)\] \((\w+)\)", re.MULTILINE)


def rank_one_per_venue(prompt: str) -> str:
    """A ranking array picking the FIRST candidate of every venue block in the prompt.

    Reads the indices back out of the prompt rather than hard-coding them: `parse_ranking` drops
    out-of-range indices, so a hard-coded array would silently render fewer rows than the test
    means on a small pool and nothing at all on an empty one — a test that asserts on rows would
    then pass for the wrong reason.
    """
    first_of: dict[str, int] = {}
    for match in CANDIDATE_LINE_RE.finditer(prompt):
        first_of.setdefault(match.group(2), int(match.group(1)))
    return json.dumps(
        [{"i": index, "tier": "same_quantity_other_cut", "why": f"{venue} pick"} for venue, index in first_of.items()]
    )


def market_llm(
    *,
    ranking: str | Callable[[str], str] = "[]",
    author: str = AUTHOR_JSON,
    configs: list[dict] | None = None,
):
    """A fake LLM covering BOTH market stages, routed on the prompt each one builds.

    One stub for two stages because production has one invocation helper for two stages; a
    per-stage stub would let the two drift in the test suite while they cannot drift in prod.
    `configs` collects the constructor kwargs, which is how the config pins are asserted through
    the real call path rather than by reading the dict.
    """

    class FakeLlm:
        def __init__(self, **kwargs: Any) -> None:
            if configs is not None:
                configs.append(kwargs)

        async def invoke(self, prompt: str) -> str:
            if RANKER_CUE in prompt:
                return ranking(prompt) if callable(ranking) else ranking
            return author

    return FakeLlm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_question():
    q = MagicMock()
    q.id_of_question = 12345
    q.question_text = "Will SpaceX Starship reach orbit before July 2026?"
    q.title = "Will SpaceX Starship reach orbit before July 2026?"
    q.short_title = "Starship orbit 2026"
    q.resolution_criteria = "Resolves Yes if a SpaceX Starship reaches orbital velocity before 2026-07-01."
    # Spelled out rather than left to MagicMock's auto-attributes: the pipeline concatenates
    # fine_print into the settlement-join text and formats unit_of_measure into the ranker
    # prompt, so an auto-attribute would put a `<MagicMock ...>` repr into both.
    q.fine_print = ""
    q.unit_of_measure = ""
    q.page_url = "https://metaculus.com/q/12345"
    q.scheduled_resolution_time = None
    # Same auto-attribute reason, one stage later: `cap_stale_top_tier` subtracts this from a
    # market's close time, and a `MagicMock` there raises `TypeError` inside the ranking stage,
    # which the snapshot-level net swallows into an empty snapshot. Real questions carry
    # `datetime | None`. Tests that want the staleness cap to fire set a real datetime.
    q.open_time = None
    return q


@pytest.fixture
def polymarket_payload():
    """Realistic Polymarket gamma public-search response shape."""
    return {
        "events": [
            {
                "title": "Will SpaceX Starship reach orbit in 2026?",
                "slug": "spacex-starship-orbit-2026",
                "description": "Resolves yes if Starship reaches orbit in 2026.",
                "endDate": "2026-12-31T23:59:59Z",
                "volume": 125000.0,
                "markets": [
                    {
                        "question": "Will SpaceX Starship reach orbit in 2026?",
                        "outcomePrices": '["0.74", "0.26"]',
                        "volume24hr": 12500.0,
                        "bestBid": 0.73,
                        "bestAsk": 0.75,
                        "volumeNum": 987654.0,
                        "liquidityNum": 45000.0,
                        "openInterest": 60000.0,
                    }
                ],
            },
            {
                "title": "Will SpaceX complete 100 launches in 2026?",
                "slug": "spacex-100-launches-2026",
                "description": "Launch-count threshold market.",
                "endDate": "2026-12-31T23:59:59Z",
                "markets": [
                    {
                        "question": "Will SpaceX complete 100 launches in 2026?",
                        "outcomePrices": '["0.42", "0.58"]',
                        "volume24hr": 3000.0,
                    }
                ],
            },
        ],
        "markets": [],
    }


@pytest.fixture
def manifold_payload():
    return [
        {
            "id": "abc",
            "question": "Will Starship reach orbit before July 2026?",
            "slug": "starship-orbit-july-2026",
            "creatorUsername": "spaceFan",
            "probability": 0.62,
            "volume24Hours": 500.0,
            "volume": 8200.0,
            "totalLiquidity": 1500.0,
            "uniqueBettorCount": 42,
            "closeTime": 1782086400000,  # ms: 2026-06-22T00:00:00Z
            "textDescription": "YES if SpaceX Starship reaches orbit before July 1 2026.",
            "isResolved": False,
        }
    ]


@pytest.fixture
def kalshi_events_payload():
    """Kalshi `/events?status=open&with_nested_markets=true` shape.

    Field names match the live API (`volume_fp` / `open_interest_fp`, fixed-point STRINGS;
    `liquidity_dollars` deprecated at a constant "0.0000"). An earlier version of this fixture
    carried the bare names `volume` / `open_interest`, which exist nowhere in Kalshi's spec — it
    was written in the same commit as the parser and encoded the same wrong guess, so the pair
    stayed green while every real Kalshi row rendered `no-liquidity-data`. The field-name
    contract is additionally pinned against captured live payloads in
    `test_prediction_market_liquidity_contract.py`, which is where a rename should fail first.

    `settlement_sources` sits at the EVENT level, which is where both the projection and the
    settlement-source join read it; every market-level copy in the live payload is null.
    """
    return {
        "events": [
            {
                "event_ticker": "KXSPACEX-26",
                "title": "Will SpaceX Starship reach orbit in 2026?",
                "sub_title": "Before Dec 31 2026",
                "settlement_sources": [{"name": "SpaceX newsroom", "url": "https://www.spacex.com/updates"}],
                "markets": [
                    {
                        "ticker": "KXSPACEX-26-YES",
                        "title": "Will SpaceX Starship reach orbit in 2026?",
                        "rules_primary": "If Starship achieves orbital velocity in 2026 per SpaceX confirmation.",
                        "yes_bid_dollars": "0.68",
                        "yes_ask_dollars": "0.72",
                        "notional_value_dollars": "1.0000",
                        "volume_24h_fp": "2500.0",
                        "volume_fp": "82000.00",
                        "open_interest_fp": "15000.00",
                        "liquidity_dollars": "0.0000",
                        "close_time": "2026-12-31T23:59:59Z",
                        "status": "active",
                    }
                ],
            },
            {
                "event_ticker": "KXOTHER-1",
                "title": "Unrelated World Cup market",
                "markets": [{"ticker": "KXOTHER-1-YES", "title": "Unrelated", "status": "active"}],
            },
        ],
        "cursor": "",
    }


@pytest.fixture
def predictit_payload():
    """PredictIt `/api/marketdata/all/` shape (verified live). US-politics only, no
    volume/liquidity/OI fields anywhere."""
    return {
        "markets": [
            {
                "id": 7456,
                "name": "Will SpaceX Starship reach orbit before July 2026?",
                "shortName": "Starship orbit 2026",
                "url": "https://www.predictit.org/markets/detail/7456/starship-orbit",
                "contracts": [
                    {
                        "id": 100,
                        "name": "Yes",
                        "shortName": "Yes",
                        "lastTradePrice": 0.58,
                        "bestBuyYesCost": 0.59,
                        "bestBuyNoCost": 0.42,
                        "dateEnd": "2026-07-01T00:00:00",
                        "status": "Open",
                    }
                ],
            },
            {
                "id": 7999,
                "name": "Unrelated 2028 primary market",
                "shortName": "2028 primary",
                "url": "https://www.predictit.org/markets/detail/7999/2028-primary",
                "contracts": [
                    {"id": 200, "name": "Candidate A", "status": "Open", "dateEnd": "NA"},
                    {"id": 201, "name": "Candidate B", "status": "Open", "dateEnd": "N/A"},
                ],
            },
        ]
    }


OFF_TOPIC_KALSHI_EVENT = {
    "event_ticker": "KXWORLDCUP-26",
    "title": "Who wins the 2026 World Cup?",
    # The liquidity keys are not decoration: every real open Kalshi market carries `volume_fp` and
    # `open_interest_fp` (1,504/1,504 measured), and Signal A fires when a declared field is
    # absent from 100% of a venue's POOL rows. Without them this file's self-described HEALTHY
    # baseline parses to a 100%-dead Kalshi row, which is a `market_field_contract` finding in
    # every test that uses `handlers()`.
    "markets": [
        {
            "ticker": "KXWORLDCUP-26-BRA",
            "title": "Brazil",
            "status": "active",
            "volume_fp": "1000000",
            "open_interest_fp": "500000",
            "last_price_dollars": "0.30",
            "notional_value_dollars": "1.0000",
        }
    ],
}
OFF_TOPIC_PREDICTIT_MARKET = {
    "id": 9001,
    "name": "Who wins the 2026 World Cup?",
    "url": "https://www.predictit.org/markets/detail/9001",
    "contracts": [{"id": 1, "name": "Brazil", "status": "Open", "lastTradePrice": 0.2}],
}


def handlers(**overrides: Any) -> dict[str, Any]:
    """The healthy baseline: every venue live, every venue empty.

    Empty SEARCH results and POPULATED catalogues, deliberately. A catalogue that a successful
    fetch returns empty is a different degradation (provider-health Signal C) and stubbing one
    by default would make it fire in every test here.
    """
    baseline: dict[str, Any] = {
        POLY_URL: FakeResponse(200, {"events": [], "markets": []}),
        MANIFOLD_SEARCH_URL: FakeResponse(200, []),
        MANIFOLD_DETAIL_URL: FakeResponse(200, {}),
        KALSHI_EVENTS_URL: FakeResponse(200, {"events": [OFF_TOPIC_KALSHI_EVENT], "cursor": ""}),
        PREDICTIT_URL: FakeResponse(200, {"markets": [OFF_TOPIC_PREDICTIT_MARKET]}),
    }
    baseline.update(overrides)
    return baseline


def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero BOTH copies of the retry backoff, for tests that drive retry-exhaustion paths.

    `venues.kalshi` imports the constant by name, so patching only `http` leaves the catalogue
    pull's own retry sleeping the real 0.5s — which is exactly the kind of half-patch that makes
    a suite mysteriously slow rather than red. The second target is the SUBMODULE, not the
    `venues` package: the package re-export is a separate binding the pull never reads.
    """
    monkeypatch.setattr("metaculus_bot.research.market_retrieval.http.HTTP_RETRY_BACKOFFS_SECS", (0.0, 0.0))
    monkeypatch.setattr(venues.kalshi, "HTTP_RETRY_BACKOFF_SECS", 0.0)


async def fetch_snapshot(
    question: Any,
    handlers: dict[str, Any],
    *,
    ranking: str | Callable[[str], str] = "[]",
    author: str = AUTHOR_JSON,
    as_of: datetime | None = None,
    timeout: float = 5.0,  # noqa: ASYNC109  # mirrors the production fetch_market_snapshot timeout parameter
    configs: list[dict] | None = None,
) -> MarketSnapshot:
    """One snapshot through the real pipeline with both LLM stages stubbed."""
    with (
        patch.object(
            pmp, "build_llm_with_openrouter_fallback", market_llm(ranking=ranking, author=author, configs=configs)
        ),
        patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
    ):
        return await pmp.fetch_market_snapshot(question, timeout=timeout, as_of=as_of)


def market_row(
    title: str,
    *,
    platform: str = "polymarket",
    tier: str = "same_quantity_other_cut",
    why: str = "same series, adjacent month",
    rules: str = "rules text",
) -> MarketMatch:
    return MarketMatch(
        platform=platform,  # type: ignore[arg-type]
        market_title=title,
        market_url=f"https://example.com/{platform}",
        implied_prob_yes=0.5,
        bid=None,
        ask=None,
        spread=None,
        volume_24h=None,
        close_time=datetime(2026, 12, 31, tzinfo=UTC),
        is_resolved=False,
        match_confidence=1.0,
        raw_rules=rules,
        total_volume=100_000.0,
        open_interest=50_000.0,
        relation_tier=tier,
        relevance_label=why,
    )
