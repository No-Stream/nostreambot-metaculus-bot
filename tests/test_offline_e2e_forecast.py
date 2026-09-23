"""Offline full-pipeline e2e forecast test: the breaking-dependency tripwire.

Binary, numeric and MC each run the whole ``forecast_questions`` pipeline offline, driving the
REAL code paths of every external dependency on the forecast critical path (research, forecaster
fan-out, aggregation, gap-fill v1 and v2) and stubbing ONLY the outermost network boundary, the
socket-opening client call. That is what caught the litellm 1.92 ``tools=`` crash, which fired
only when the real call executed and which the agentic v2 loop then soft-failed to "".

Two seams do the work. ``_install_llm_router`` patches both ``acompletion`` chokepoints with a
router that reads the outgoing prompt, picks the matching canned response and forwards to real
litellm with ``mock_response``, so all real litellm import/transform/tools-path code executes.
``_install_provider_stubs`` replaces each provider's external client at its lowest boundary, with
conftest's autouse network-egress guard as the backstop. ``_assert_pipeline_ran`` is where a run
that quietly degraded gets caught.

Why each canned payload and stub is shaped the way it is, what ``_assert_pipeline_ran`` pins and
the receipt behind each signal: ``docs/architecture.md`` "The offline end-to-end test".
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asknews_sdk
import litellm
import pytest
from forecasting_tools import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.ai_models import general_llm as ft_general_llm
from forecasting_tools.data_models.forecast_report import ForecastReport

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.llm_configs import (
    DISAGREEMENT_ANALYZER_LLM,
    FORECASTER_LLMS,
    PARSER_LLM,
    RESEARCHER_LLM,
    STACKER_LLM,
    SUMMARIZER_LLM,
)
from metaculus_bot.research import gemini_search, prediction_market
from metaculus_bot.research import providers as research_providers
from metaculus_bot.research.agentic import llm as agentic_llm
from metaculus_bot.research.fetch_ladder import guard

_NOW = datetime.now(UTC)
_OPEN = _NOW - timedelta(days=30)
_RESOLVE = _NOW + timedelta(days=180)

# A fetchable URL in the resolution criteria is what exercises the resolution-source provider.
_RESOLUTION_URL = "https://data.example.gov/unemployment-report"


# ---------------------------------------------------------------------------
# Canned forecaster / stacker blocks — parse at value_extraction rung 1 (block).
# ---------------------------------------------------------------------------

_CANNED_BINARY = """\
## Analysis
The status quo holds; no post-open trigger event has occurred. Base rate is low.

```json
{"question_type": "binary", "posterior_prob": 0.22}
```
"""

_CANNED_NUMERIC = """\
## Analysis
Recent measurements cluster tightly; the distribution is centered near the latest value.
OUTCOME_TYPE: CONTINUOUS

```json
{
  "question_type": "numeric",
  "declared_percentiles": {
    "0.01": 3.0, "0.025": 3.2, "0.05": 3.4, "0.1": 3.6, "0.2": 3.8, "0.4": 4.1, "0.5": 4.3,
    "0.6": 4.5, "0.8": 5.0, "0.9": 5.6, "0.95": 6.2, "0.975": 7.0, "0.99": 7.8
  },
  "outcome_type": "continuous"
}
```
"""

_CANNED_MC = """\
## Analysis
Option A carries institutional momentum; B is the eroding status quo; C is a tail.

```json
{"question_type": "multiple_choice", "option_probs": {"Option A": 0.45, "Option B": 0.40, "Option C": 0.15}}
```
"""

_CANNED_SUMMARY_PROSE = (
    "Newest directly-relevant article: 2026-04-14. The unemployment rate stood at 4.1% in the "
    "most recent release [B: Reuters]. Initial jobless claims trended up modestly."
)

_CANNED_NATIVE_SEARCH_PROSE = (
    "The Bureau of Labor Statistics reported the unemployment rate at 4.1% in April 2026 "
    "[BLS](https://www.bls.gov/news.release/empsit.nr0.htm)."
)

# The one shape `parse_query_author` accepts; anything else costs a source and reddens the suite.
_CANNED_QUERY_AUTHOR = json.dumps(
    {"synonyms": ["jobless rate", "U-3", "household survey"], "framings": ["unemployment print", "jobs report rate"]}
)

# The ranker's tiers in value order; only the first two earn the strong-evidence preamble.
_RANKER_TIERS = ("same_quantity_same_date", "same_quantity_other_cut", "driver_or_consequence")


def _canned_ranking(prompt: str) -> str:
    """A well-formed ranking array over whatever pool this prompt actually carries.

    The indices have to be real: `parse_ranking` drops out-of-range ones, so a hard-coded
    `[0, 1, 2]` would silently render fewer rows than intended on a small pool and nothing at all
    on an empty one — and a canned array that cannot be read as JSON at all would fail open,
    bumping the source-loss counter and reddening `_assert_pipeline_ran`. The prompt states its
    own candidate count, so read it back rather than assuming the stub payloads' shape.
    """
    match = re.search(r"^(\d+) candidates from ", prompt, flags=re.MULTILINE)
    pool_size = int(match.group(1)) if match else 0
    picks = [
        {"i": index, "tier": _RANKER_TIERS[index % len(_RANKER_TIERS)], "why": f"stub pick at pool index {index}"}
        for index in range(min(pool_size, 3))
    ]
    return json.dumps(picks)


# gap-fill v1 analyzer: a single-gap JSON payload, graded to pass triage so the resolver path runs.
_CANNED_GAP_ANALYZER = json.dumps(
    {
        "gaps": [
            {
                "gap": "Latest BLS release date",
                "why_matters": "Anchors the level",
                "search_query": "BLS release",
                "answerable_now": True,
                "already_in_first_pass": False,
                "same_need_as": None,
            }
        ]
    }
)

# Providers that MUST report `ok` in the diagnostics block for these questions.
_REQUIRED_OK_PROVIDERS = frozenset({"asknews", "native_search", "gemini_search", "resolution_source"})


# ---------------------------------------------------------------------------
# LLM router — one wrapper for both acompletion chokepoints.
# ---------------------------------------------------------------------------


def _messages_text(kwargs: dict[str, Any]) -> str:
    """Concatenate all message contents so we can sniff the call type from the prompt."""
    parts: list[str] = []
    for msg in kwargs.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # Vision messages carry a list of content parts (text + image_url dicts).
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "\n".join(parts)


def _route_general_llm(kwargs: dict[str, Any]) -> str:
    """Pick the canned response for a forecasting-tools GeneralLlm call.

    Routes on prompt content. The base forecaster prompts embed a
    'STRUCTURED FORECAST' block schema and a per-type cue (percentiles / options
    line); the summarizer/native-search/gap-fill-analyzer calls carry their own
    distinctive text. Order matters — most-specific first: the forecaster branch is checked
    before the summarizer signal because the base MC and numeric prompts also contain
    "Intelligence Briefing", and the prediction-market ranker before the generic branches because
    its prompt carries a resolution-criteria header and no block, so it would otherwise fall
    through to canned prose it cannot parse. Question type is read off the prompt too: percentiles
    means numeric, an "Options (in resolution order):" line means MC, else binary.
    """
    text = _messages_text(kwargs)
    lower = text.lower()

    # Forecaster and stacker calls are the only ones carrying the fenced STRUCTURED FORECAST block.
    if "STRUCTURED FORECAST" in text:
        if "percentile" in lower:
            return _CANNED_NUMERIC
        if "options (in resolution order)" in lower:
            return _CANNED_MC
        return _CANNED_BINARY

    # The only call ranking candidates by evidential value.
    if "Rank the candidates by EVIDENTIAL VALUE" in text:
        return _canned_ranking(text)

    # Prediction-market query author: asks for a two-key JSON object of extra search queries.
    if "You are writing search queries to find prediction markets" in text:
        return _CANNED_QUERY_AUTHOR

    # gap-fill v1 analyzer: asks for a JSON {"gaps": [...]} list.
    if "research-quality auditor" in lower or '"gaps"' in text:
        return _CANNED_GAP_ANALYZER

    # AskNews summarizer: builds an "intelligence briefing" from raw <research>.
    if "intelligence briefing" in lower or "<research>" in lower:
        return _CANNED_SUMMARY_PROSE

    # Native / targeted web search, the gap-fill resolver and the perplexity fallback: no block.
    return _CANNED_NATIVE_SEARCH_PROSE


def _install_llm_router(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch BOTH acompletion chokepoints to route → real litellm + mock_response.

    Chokepoint 1: ``forecasting_tools.ai_models.general_llm.acompletion`` — every GeneralLlm call
    (forecasters, stacker, parser, summarizer, native search). Chokepoint 2:
    ``metaculus_bot.research.agentic.llm.acompletion`` — the raw-litellm agentic v2 driver (the
    ONLY ``tools=`` caller). Both forward to the REAL ``litellm.acompletion`` with
    ``mock_response`` (and, for the tools path, ``mock_tool_calls``) added, so all real litellm
    import/transform/tools-gated code executes while the network is short-circuited.

    The agentic driver needs a tool-call-shaped response, and the scripted turns
    (set_research_plan, then conclude) run its loop end to end without any external tool call
    while still driving the real ``tools=`` path on every step. ``drop_params`` must be True for
    the agentic wrapper's OpenRouter reasoning_effort handling: in prod a GeneralLlm invoke sets it
    globally, so setting it here keeps the agentic path deterministic even if it runs first.
    """
    real_acompletion = litellm.acompletion

    def _mock_completion_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Use a model LiteLLM can resolve before its mock-response branch.

        The repository roster contains future OpenRouter model IDs that the installed
        LiteLLM release does not know how to route locally. Keeping the original model
        in the outgoing call to this router still exercises the bot's model selection;
        the forwarding-only model substitution lets LiteLLM reach its offline mock path.
        """
        return {**kwargs, "model": "gpt-3.5-turbo"}

    async def general_llm_router(**kwargs: Any) -> Any:
        return await real_acompletion(**_mock_completion_kwargs(kwargs), mock_response=_route_general_llm(kwargs))

    # The scripted driver turns: set_research_plan first, then conclude.
    agentic_state = {"step": 0}

    async def agentic_router(**kwargs: Any) -> Any:
        if kwargs.get("tools"):
            step = agentic_state["step"]
            agentic_state["step"] += 1
            if step == 0:
                mock_tool_calls = [
                    {
                        "id": "plan0",
                        "type": "function",
                        "function": {
                            "name": "set_research_plan",
                            "arguments": json.dumps(
                                {"gaps": [{"id": "g1", "question": "Latest authoritative measurement?"}]}
                            ),
                        },
                    }
                ]
            else:
                mock_tool_calls = [
                    {
                        "id": "done0",
                        "type": "function",
                        "function": {
                            "name": "conclude",
                            "arguments": json.dumps(
                                {
                                    "gap_accounting": [
                                        {
                                            "gap_id": "g1",
                                            "actions_taken": "briefing already covers it",
                                            "status": "resolved",
                                        }
                                    ]
                                }
                            ),
                        },
                    }
                ]
            return await real_acompletion(
                **_mock_completion_kwargs(kwargs),
                mock_response="driving the agentic loop",
                mock_tool_calls=mock_tool_calls,
            )
        # The ghost phase calls with tools=None, and _summarize_ghost parses a plain block.
        return await real_acompletion(**_mock_completion_kwargs(kwargs), mock_response=_CANNED_BINARY)

    monkeypatch.setattr(ft_general_llm, "acompletion", general_llm_router)
    monkeypatch.setattr(agentic_llm, "acompletion", agentic_router)
    monkeypatch.setattr(litellm, "drop_params", True)


# ---------------------------------------------------------------------------
# Provider client stubs — the lowest socket-opening boundary of each provider.
# ---------------------------------------------------------------------------


class _FakeArticle:
    """Duck-typed AskNews article (attribute access used by _format_single_article)."""

    def __init__(self, title: str, url: str) -> None:
        self.eng_title = title
        self.summary = "Unemployment held at 4.1% in the latest monthly print."
        self.language = "en"
        self.pub_date = datetime(2026, 4, 14, 9, 0, tzinfo=UTC)
        self.source_id = "reuters"
        self.article_url = url


class _FakeAskNewsResponse:
    def __init__(self, articles: list[_FakeArticle]) -> None:
        self.as_dicts = articles


class _FakeAskNewsSDK:
    """Async context manager standing in for AsyncAskNewsSDK; .news.search_news is awaited."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.news = MagicMock()
        self.news.search_news = AsyncMock(
            return_value=_FakeAskNewsResponse(
                [_FakeArticle("US unemployment steady at 4.1%", "https://reuters.com/econ/us-unemployment-apr-2026")]
            )
        )

    async def __aenter__(self) -> _FakeAskNewsSDK:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


def _make_gemini_response() -> Any:
    """Minimal response with a self-cited redirect link for the offline e2e path."""

    metadata = SimpleNamespace(grounding_chunks=None, grounding_supports=None, web_search_queries=["unemployment"])
    candidate = SimpleNamespace(grounding_metadata=metadata, url_context_metadata=None)
    return SimpleNamespace(
        text=(
            "Google Search found an April 2026 unemployment rate of 4.1% "
            "[BLS Employment Situation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/offline-e2e-token)."
        ),
        candidates=[candidate],
    )


def _fake_gemini_client() -> MagicMock:
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=_make_gemini_response())
    return client


# --- aiohttp fakes for prediction-market + resolution-source (JSON + HTML) ----


class _FakeContent:
    def __init__(self, resp: _FakeHttpResponse) -> None:
        self._resp = resp

    async def iter_chunked(self, n: int) -> Any:
        body = self._resp._body
        for i in range(0, len(body), n):
            yield body[i : i + n]


class _FakeHttpResponse:
    def __init__(self, status: int = 200, *, body: bytes = b"{}", content_type: str = "application/json") -> None:
        self.status = status
        self._body = body
        self.headers = {"Content-Type": content_type}
        self.content = _FakeContent(self)

    async def read(self) -> bytes:
        return self._body

    async def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")

    async def json(self) -> Any:
        return json.loads(self._body)

    async def __aenter__(self) -> _FakeHttpResponse:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


_RESOLUTION_HTML = (
    b"<!doctype html><html><head><title>Unemployment Report</title></head><body>"
    b"<article><h1>April 2026 Employment Situation</h1>"
    b"<p>The Bureau of Labor Statistics reported the seasonally adjusted unemployment "
    b"rate at 4.1 percent for April 2026, unchanged from the prior month. Nonfarm "
    b"payroll employment rose by 175,000. The labor force participation rate held at "
    b"62.7 percent. Analysts had expected a reading near 4.2 percent, so the print was "
    b"modestly stronger than consensus. Wage growth cooled slightly year over year.</p>"
    b"</article></body></html>"
)


# Every venue returns a POPULATED off-topic payload with the liquidity fields provider_health wants.
_OFF_TOPIC_KALSHI_EVENTS = json.dumps(
    {
        "events": [
            {
                "event_ticker": "KXWORLDCUP-26",
                "title": "Who will win the 2026 FIFA World Cup?",
                "sub_title": "Tournament winner",
                "settlement_sources": [{"name": "FIFA", "url": "https://www.fifa.com/worldcup"}],
                "markets": [
                    {
                        "ticker": "KXWORLDCUP-26-BRA",
                        "title": "Brazil",
                        "rules_primary": "Resolves Yes if Brazil wins the 2026 FIFA World Cup final.",
                        "status": "active",
                        "close_time": "2026-07-19T23:59:59Z",
                        "yes_bid_dollars": "0.21",
                        "yes_ask_dollars": "0.23",
                        "notional_value_dollars": "1.0000",
                        "volume_fp": "41000.00",
                        "open_interest_fp": "9000.00",
                        "volume_24h_fp": "150.0",
                    }
                ],
            }
        ],
        "cursor": "",
    }
).encode()
_OFF_TOPIC_PREDICTIT_MARKETS = json.dumps(
    {
        "markets": [
            {
                "id": 7001,
                "name": "Who will win the 2026 FIFA World Cup?",
                "shortName": "World Cup 2026",
                "url": "https://www.predictit.org/markets/detail/7001",
                "status": "Open",
                "contracts": [
                    {
                        "id": 70011,
                        "name": "Brazil",
                        "shortName": "Brazil",
                        "status": "Open",
                        "dateEnd": "2026-07-19T23:59:59",
                        "lastTradePrice": 0.22,
                        "bestBuyYesCost": 0.23,
                        "bestBuyNoCost": 0.78,
                    }
                ],
            }
        ]
    }
).encode()
_OFF_TOPIC_POLYMARKET_SEARCH = json.dumps(
    {
        "events": [
            {
                "title": "Who will win the 2026 FIFA World Cup?",
                "slug": "world-cup-2026-winner",
                "description": "Resolves to the winner of the 2026 FIFA World Cup final.",
                "endDate": "2026-07-19T23:59:59Z",
                "openInterest": 120000.0,
                "markets": [
                    {
                        "question": "Will Brazil win the 2026 FIFA World Cup?",
                        "outcomePrices": '["0.22", "0.78"]',
                        "volumeNum": 310000.0,
                        "liquidityNum": 40000.0,
                    }
                ],
            }
        ],
        "markets": [],
    }
).encode()
_OFF_TOPIC_MANIFOLD_SEARCH = json.dumps(
    [
        {
            "id": "wc26brazil",
            "question": "Will Brazil win the 2026 FIFA World Cup?",
            "slug": "brazil-world-cup-2026",
            "creatorUsername": "footyFan",
            "probability": 0.21,
            "volume": 4200.0,
            "volume24Hours": 80.0,
            "totalLiquidity": 900.0,
            "uniqueBettorCount": 64,
            "closeTime": 1784419199000,
            "isResolved": False,
        }
    ]
).encode()
# The search listing carries no description; the detail record is where the rules text comes from.
_MANIFOLD_MARKET_DETAIL = json.dumps(
    {
        "id": "wc26brazil",
        "question": "Will Brazil win the 2026 FIFA World Cup?",
        "textDescription": "Resolves YES if Brazil lifts the trophy in the 2026 final.",
    }
).encode()


class _FakeHttpSession:
    """aiohttp.ClientSession stand-in for the prediction-market + resolution-source hosts.

    Every venue returns a populated OFF-TOPIC payload, because that is what "no relevant market"
    actually looks like upstream (the FIXTURE RATIONALE section of the module docstring has the
    receipts) — the ranker, not the transport, is what decides a candidate does not bear on the
    question. The resolution-source host returns an article-shaped HTML body so trafilatura runs.

    Manifold's two endpoints are routed separately, and the ORDER matters: the detail path
    (`/v0/market/<id>`) is checked first, because a substring test for "manifold" alone would
    serve it the search listing's array and leave every candidate title-only.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.closed = False

    def get(self, url: str, **_kwargs: Any) -> _FakeHttpResponse:
        low = url.lower()
        if "example.gov" in low or "example.com" in low:
            return _FakeHttpResponse(200, body=_RESOLUTION_HTML, content_type="text/html; charset=utf-8")
        if "manifold" in low and "/v0/market/" in low:
            return _FakeHttpResponse(200, body=_MANIFOLD_MARKET_DETAIL, content_type="application/json")
        if "manifold" in low:
            return _FakeHttpResponse(200, body=_OFF_TOPIC_MANIFOLD_SEARCH, content_type="application/json")
        if "predictit" in low:
            return _FakeHttpResponse(200, body=_OFF_TOPIC_PREDICTIT_MARKETS, content_type="application/json")
        if "kalshi" in low and "/events" in low:
            return _FakeHttpResponse(200, body=_OFF_TOPIC_KALSHI_EVENTS, content_type="application/json")
        # Polymarket public-search + anything else.
        return _FakeHttpResponse(200, body=_OFF_TOPIC_POLYMARKET_SEARCH, content_type="application/json")

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> _FakeHttpSession:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()


def _install_provider_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every provider's external client at its socket-opening boundary.

    Three patch sites are chosen rather than obvious. Both the two-phase AskNews provider and the
    agentic tools' search_news do a function-scoped ``from asknews_sdk import AsyncAskNewsSDK``, so
    patching the module attribute covers both call sites. The two-phase provider also sleeps 10.1s
    twice before its calls, so ``asyncio.sleep`` is patched inside the providers module alone,
    which keeps the test fast without touching the event loop anywhere else. For Gemini the patch
    goes on ``build_gemini_client``, the public factory the provider calls, and NOT on
    ``_cached_client_for_key``: that one is lru_cache-wrapped and conftest's autouse
    ``_clear_gemini_client_cache`` fixture calls ``.cache_clear()`` on it at teardown, so replacing
    it with a plain lambda would break teardown, while patching the caller leaves the cache intact.
    """
    monkeypatch.setattr(asknews_sdk, "AsyncAskNewsSDK", _FakeAskNewsSDK)

    # Skip the AskNews provider's real rate-gate sleeps.
    async def _noop_rate_gate() -> None:
        return None

    monkeypatch.setattr(research_providers, "_asknews_rate_gate", _noop_rate_gate)
    real_sleep = research_providers.asyncio.sleep

    async def _fast_sleep(seconds: float) -> None:
        """Collapse the provider's deliberate 10.1s throttle waits.

        The await itself stays real (a 0-sleep), so scheduling semantics are unchanged.
        """
        await real_sleep(0)

    monkeypatch.setattr(research_providers.asyncio, "sleep", _fast_sleep)

    monkeypatch.setattr(gemini_search, "build_gemini_client", _fake_gemini_client)
    monkeypatch.setattr(
        gemini_search,
        "resolve_search_redirects",
        AsyncMock(
            return_value={
                "https://vertexaisearch.cloud.google.com/grounding-api-redirect/offline-e2e-token": "https://bls.gov/report",
            }
        ),
    )

    # Prediction-market + resolution-source aiohttp sessions.
    monkeypatch.setattr(prediction_market, "_get_session", _FakeHttpSession)
    monkeypatch.setattr(guard, "_get_session", _FakeHttpSession)
    # example.gov has no DNS for the SSRF preflight; mirrors tests/resolution_source/conftest.py.
    monkeypatch.setattr(
        guard.socket,
        "getaddrinfo",
        lambda *a, **k: [(0, 0, 0, "", ("8.8.8.8", 0))],
    )
    prediction_market._reset_session_caches()


def _install_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror the prod-workflow env: every provider ENABLED + dummy keys so gates pass.

    Stacking flags are deliberately NOT left on — prod runs with stacking disabled, so
    the default-off median/skipped path is what we exercise. (conftest's autouse
    fixture sets the *_STACKING_ENABLED flags on; we set them false here to reproduce
    prod, which routes through the non-stacked aggregation.)

    The keys are dummies, each opening one gate: the AskNews creds make AskNews the primary
    provider (the prod case), GOOGLE_API_KEY opens gemini, FRED_API_KEY opens financial data, and
    only the personal OpenRouter key is set — the donated one is deleted — so the fallback wrapper
    stays single-key deterministic and the agentic router runs on one key.
    """
    for flag in (
        "NATIVE_SEARCH_ENABLED",
        "GEMINI_SEARCH_ENABLED",
        "FINANCIAL_DATA_ENABLED",
        "GAP_FILL_ENABLED",
        "GAP_FILL_V2_ENABLED",
        "PREDICTION_MARKETS_ENABLED",
        "RESOLUTION_SOURCE_ENABLED",
    ):
        monkeypatch.setenv(flag, "true")
    # Restore prod's stacking-disabled default (conftest autouse turns these on).
    for flag in ("BINARY_STACKING_ENABLED", "MC_STACKING_ENABLED", "NUMERIC_STACKING_ENABLED"):
        monkeypatch.setenv(flag, "false")
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "dummy-client")
    monkeypatch.setenv("ASKNEWS_SECRET", "dummy-secret")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-google")
    monkeypatch.setenv("FRED_API_KEY", "dummy-fred")
    monkeypatch.setenv("EXA_API_KEY", "dummy-exa")
    monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")


def _make_bot() -> TemplateForecaster:
    """Build the REAL production ensemble (llm_configs singletons), stacking-strategy
    CONDITIONAL_STACKING (the code default), min_forecasters=1, is_benchmarking=False.

    is_benchmarking MUST be False: the prediction-market and resolution-source
    providers hard-disable under benchmarking, and gap-fill v2 returns "" — we want
    all of them to run.
    """
    llms: dict[str, Any] = {
        "forecasters": FORECASTER_LLMS,
        "stacker": STACKER_LLM,
        "analyzer": DISAGREEMENT_ANALYZER_LLM,
        "summarizer": SUMMARIZER_LLM,
        "parser": PARSER_LLM,
        "researcher": RESEARCHER_LLM,
    }
    return TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,  # default False; no Metaculus post
        aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
        llms=llms,
        is_benchmarking=False,
        min_forecasters_to_publish=1,
    )


def _binary_question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will the US unemployment rate exceed 5% by December 2026?",
        id_of_question=70001,
        id_of_post=80001,
        page_url="https://www.metaculus.com/questions/70001/",
        background_info="The US unemployment rate has been between 3.4% and 4.2% for the past year.",
        resolution_criteria=(
            "Resolves YES if BLS reports a seasonally adjusted unemployment rate of 5.0% or higher "
            f"for any month through December 2026. Source: {_RESOLUTION_URL}"
        ),
        fine_print="Uses seasonally adjusted figures from the BLS Employment Situation report.",
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _numeric_question() -> NumericQuestion:
    return NumericQuestion(
        question_text="What will the US unemployment rate be in December 2026?",
        id_of_question=70002,
        id_of_post=80002,
        page_url="https://www.metaculus.com/questions/70002/",
        background_info="The US unemployment rate is reported monthly by the BLS.",
        resolution_criteria=(
            "Resolves to the seasonally adjusted U-3 rate published by BLS for December 2026. "
            f"Source: {_RESOLUTION_URL}"
        ),
        fine_print="If revised, uses the initial release value.",
        unit_of_measure="percent",
        lower_bound=0.0,
        upper_bound=20.0,
        open_lower_bound=False,
        open_upper_bound=True,
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _mc_question() -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which economic scenario is most likely for the US in 2026?",
        id_of_question=70003,
        id_of_post=80003,
        page_url="https://www.metaculus.com/questions/70003/",
        options=["Option A", "Option B", "Option C"],
        background_info="Multiple economic scenarios are possible depending on Fed policy.",
        resolution_criteria=(
            f"Resolves to the option best describing the realized outcome by year-end. Source: {_RESOLUTION_URL}"
        ),
        fine_print="Resolution determined by a panel of three economists.",
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


@pytest.fixture
def offline_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the LLM router, all provider stubs, and the prod-mirroring env."""
    _install_env(monkeypatch)
    _install_llm_router(monkeypatch)
    _install_provider_stubs(monkeypatch)


class TestOfflineE2EForecast:
    """Full offline pipeline for each question type, all providers on, no network."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_binary_full_pipeline_offline(self, offline_pipeline: None, caplog: pytest.LogCaptureFixture) -> None:
        bot = _make_bot()
        with caplog.at_level("INFO"):
            reports = await bot.forecast_questions([_binary_question()])

        assert len(reports) == 1
        report = reports[0]
        assert isinstance(report, ForecastReport)
        prediction = report.prediction
        assert isinstance(prediction, float)
        # Binary clamp is [0.02, 0.98]; median-of-3 identical 0.22 blocks = 0.22.
        assert 0.02 <= prediction <= 0.98

        _assert_pipeline_ran(caplog, bot, expect_qtype="binary")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_numeric_full_pipeline_offline(
        self, offline_pipeline: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        bot = _make_bot()
        with caplog.at_level("INFO"):
            reports = await bot.forecast_questions([_numeric_question()])

        assert len(reports) == 1
        report = reports[0]
        assert isinstance(report, ForecastReport)
        # Numeric prediction is a NumericDistribution; its published CDF is 201 points.
        cdf = report.prediction.cdf
        assert len(cdf) == 201
        # Monotonic non-decreasing CDF within [0, 1].
        probs = [pt.percentile for pt in cdf]
        assert probs == sorted(probs)
        assert 0.0 <= probs[0] <= probs[-1] <= 1.0

        _assert_pipeline_ran(caplog, bot, expect_qtype="numeric")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_mc_full_pipeline_offline(self, offline_pipeline: None, caplog: pytest.LogCaptureFixture) -> None:
        bot = _make_bot()
        with caplog.at_level("INFO"):
            reports = await bot.forecast_questions([_mc_question()])

        assert len(reports) == 1
        report = reports[0]
        assert isinstance(report, ForecastReport)
        option_probs = [opt.probability for opt in report.prediction.predicted_options]
        assert len(option_probs) == 3
        assert abs(sum(option_probs) - 1.0) < 1e-6
        assert all(0.0 <= p <= 1.0 for p in option_probs)

        _assert_pipeline_ran(caplog, bot, expect_qtype="multiple_choice")


def _assert_pipeline_ran(caplog: pytest.LogCaptureFixture, bot: TemplateForecaster, *, expect_qtype: str) -> None:
    """Assert that every real code path executed offline WITHOUT any swallowed failure.

    Each signal it pins, and the receipt behind it, is in the module docstring's
    "WHAT ``_assert_pipeline_ran`` PINS".
    """
    assert bot.alertable_count == 0, (
        f"pipeline degraded — a provider errored, a forecaster was dropped, the stacker "
        f"fell back, or gap-fill v2 crashed: alertable_count={bot.alertable_count}"
    )

    messages = [rec.getMessage() for rec in caplog.records]
    text = "\n".join(messages)

    # Forecaster value-extraction landed on rung 1 for the expected question type.
    extraction_lines = [m for m in messages if "EXTRACTION_RUNG:" in m]
    assert extraction_lines, "no EXTRACTION_RUNG telemetry — forecaster extraction did not run"
    assert any(f"qtype={expect_qtype}" in m and "rung=block" in m for m in extraction_lines), (
        f"expected a rung=block extraction for qtype={expect_qtype}; got: {extraction_lines}"
    )

    # Gap-fill v2 executed and did not crash (the fastapi tripwire).
    v2_lines = [m for m in messages if "GAP_FILL_V2:" in m]
    assert v2_lines, "no GAP_FILL_V2 marker — the agentic v2 loop never ran"
    clean_v2_lines = [m for m in v2_lines if "error=None" in m]
    assert clean_v2_lines, f"gap-fill v2 crashed (fastapi-class bug?): {v2_lines}"
    # tool_calls=0 with error=None reads like a healthy run, so require >=1; (?<!dup_) dodges dup_.
    tool_call_counts = [
        int(match.group(1)) for m in clean_v2_lines if (match := re.search(r"(?<!dup_)tool_calls=(\d+)", m))
    ]
    assert tool_call_counts, f"GAP_FILL_V2 marker has no tool_calls= field: {clean_v2_lines}"
    assert any(n > 0 for n in tool_call_counts), (
        f"gap-fill v2 ran but issued no tool calls (tool_calls=0) — driver stopped sending tools: {clean_v2_lines}"
    )

    # Gap-fill v1 graded its gap, kept it, and the resolved section reached the research bundle.
    triage_lines = [m for m in messages if "GAP_FILL_V1_TRIAGE:" in m]
    assert triage_lines, "no GAP_FILL_V1_TRIAGE marker — the gap-fill v1 analyzer/triage never ran"
    assert any("listed=1 kept=1" in m for m in triage_lines), (
        f"gap-fill v1 triage dropped the canned gap (expected listed=1 kept=1): {triage_lines}"
    )
    assert "### Gap 1: Latest BLS release date" in text, (
        f"gap-fill v1 kept its gap but no resolved gap section reached the research bundle:\n{text}"
    )

    # The stubbed providers ran end to end: every required one `ok`, and none `errored`.
    assert "Provider diagnostics" in text, "no provider-diagnostics telemetry"
    for provider in _REQUIRED_OK_PROVIDERS:
        assert f"{provider}: ok" in text, (
            f"required provider {provider!r} did not report 'ok' in diagnostics — a provider dep may have broken:\n{text}"
        )
    assert ": errored" not in text, f"a research provider errored (swallowed by the orchestrator):\n{text}"

    # The survivor count is stated positively in the log, and names the surviving models.
    survived_lines = [m for m in messages if "FORECASTERS_SURVIVED:" in m]
    assert survived_lines, "no FORECASTERS_SURVIVED telemetry — the survivor count is not in the log"
    configured = len(bot._forecaster_llms)
    assert any(f"survived={configured}/{configured}" in m for m in survived_lines), (
        f"expected a full-ensemble survivor line (survived={configured}/{configured}); got: {survived_lines}"
    )
    assert any("models=" in m and m.split("models=")[1].strip() for m in survived_lines), (
        f"FORECASTERS_SURVIVED must name the surviving models: {survived_lines}"
    )
