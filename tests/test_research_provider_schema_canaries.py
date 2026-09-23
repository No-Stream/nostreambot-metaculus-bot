"""Schema-drift canaries for research providers that hit external SDKs.

These tests construct **real pydantic models** from the upstream SDKs
(`google-genai` for Gemini grounded search, `asknews-sdk` for AskNews),
populate them with realistic-shaped data, and feed them through our
formatters. Goal: if Google or AskNews renames / removes / restructures a
field in a future SDK upgrade, pydantic validation fails at *construction*
time inside this test file — and we know to update our parser before the
next backtest.

This is the cheapest possible coverage for two providers we currently can't
verify against real APIs without burning LLM credits / news-API budget.
The trade-off: we don't catch upstream API changes that ship before the SDK
catches up. We DO catch every change that lands in a `pip install -U` cycle,
which is the dominant breakage mode.

Imports happen at module top: a missing field on the real pydantic model
causes the *test collection* to fail, which is louder than a runtime mock
silently going stale.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from asknews_sdk.dto.base import Entities
from asknews_sdk.dto.news import SearchResponseDictItem
from forecasting_tools import MetaculusQuestion
from google.genai import types as genai_types
from pydantic import AnyUrl

# Helpers — build realistic, real-typed payloads.


def _make_real_grounding_response(
    text: str,
    chunks: list[tuple[str, str | None, str | None]] | None = None,
    supports: list[tuple[int, list[int]]] | None = None,
    *,
    response_text_override: str | None = None,
) -> genai_types.GenerateContentResponse:
    """Build an actual GenerateContentResponse with grounding metadata.

    chunks: list of (uri, title, domain) — `web.uri`, `web.title`, `web.domain` populated.
    supports: list of (segment_end_index, [chunk_indices]) for inline citations.

    Uses real types so a pydantic schema break in google-genai surfaces here.
    """
    grounding_chunks: list[genai_types.GroundingChunk] | None = None
    if chunks is not None:
        grounding_chunks = [
            genai_types.GroundingChunk(web=genai_types.GroundingChunkWeb(uri=uri, title=title, domain=domain))
            for uri, title, domain in chunks
        ]

    grounding_supports: list[genai_types.GroundingSupport] | None = None
    if supports is not None:
        grounding_supports = [
            genai_types.GroundingSupport(
                segment=genai_types.Segment(start_index=0, end_index=end_idx, text=text[:end_idx]),
                grounding_chunk_indices=indices,
            )
            for end_idx, indices in supports
        ]

    metadata = genai_types.GroundingMetadata(
        grounding_chunks=grounding_chunks,
        grounding_supports=grounding_supports,
    )
    candidate = genai_types.Candidate(grounding_metadata=metadata)
    response = genai_types.GenerateContentResponse(candidates=[candidate])
    # `text` is a property derived from candidates[0].content.parts. The grounding
    # metadata path in our formatter pulls from response.text directly, so for the
    # in-process formatter test we patch the property via SimpleNamespace fallback
    # only when the caller really wants to skip the parts-construction overhead.
    # Most callers want the real path: build a Content with a single text Part.
    candidate.content = genai_types.Content(parts=[genai_types.Part(text=response_text_override or text)])
    return response


def _make_async_client(response: object) -> MagicMock:
    """Build a MagicMock client whose .aio.models.generate_content awaits to ``response``."""
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    return client


def _make_real_asknews_article(
    *,
    eng_title: str = "Headline about the topic",
    summary: str = "Body summary of the article that the formatter will surface to forecasters.",
    article_url: str = "https://example.com/news/1",
    pub_date: _dt.datetime | None = None,
    source_id: str = "Example News",
    language: str = "en",
) -> SearchResponseDictItem:
    """Construct a real AskNews `SearchResponseDictItem` with valid required fields.

    The pydantic model has ~30 fields; most have safe defaults or are populated
    here with realistic shapes. The handful of required fields without defaults
    must all be present or `model_validate(...)` raises.
    """
    if pub_date is None:
        pub_date = _dt.datetime(2026, 5, 12, 14, 0, tzinfo=_dt.UTC)
    return SearchResponseDictItem(
        article_url=cast(AnyUrl, article_url),  # pydantic coerces str → AnyUrl at construction
        article_id=uuid.uuid4(),
        classification=["news"],
        country="US",
        source_id=source_id,
        page_rank=1,
        domain_url="example.com",
        eng_title=eng_title,
        # SDK declares every Entities field as Optional with a positional Field([]) default,
        # which basedpyright/ty don't recognize as a default — all fields are optional at runtime.
        entities=Entities(),  # pyright: ignore[reportCallIssue]
        image_url=None,
        keywords=["topic", "example"],
        language=language,
        pub_date=pub_date,
        summary=summary,
        key_points=None,
        title=eng_title,
        sentiment=0,
        medoid_distance=None,
        markdown_citation=None,
        provocative="low",
        reporting_voice="objective",
        entity_relation_graph=None,
        geo_coordinates=None,
        continent="North America",
        assets=None,
        social_embeds=None,
        bias=None,
        as_string_key=f"{source_id}|{eng_title}",
    )


# Gemini grounded search — formatter pinned against real types.


class TestGeminiGroundedFormatterAgainstRealTypes:
    """End-to-end formatter coverage with real ``google-genai`` pydantic models.

    These exercise the same code paths as the existing ``SimpleNamespace``-based
    tests, but every input is a real ``genai_types.GroundingMetadata`` etc. The
    SDK's pydantic validation is the schema-drift canary: a renamed field
    causes ``GroundingChunkWeb(uri=...)`` to raise at test-collection time.
    """

    @pytest.mark.asyncio
    async def test_real_response_with_two_chunks_and_supports(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

        apple_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/apple"
        tesla_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/tesla"
        text = (
            f"Apple announced new product. [Apple source]({apple_url}) "
            f"Tesla recalled vehicles. [Tesla source]({tesla_url})"
        )
        end_apple = text.index("Apple announced new product.") + len("Apple announced new product.")
        end_tesla = text.index("Tesla recalled vehicles.") + len("Tesla recalled vehicles.")

        # Real Gemini web.uri is an opaque vertexaisearch redirect; web.domain carries
        # the real source domain, which is what the formatter must render.
        response = _make_real_grounding_response(
            text=text,
            chunks=[
                (apple_url, "Apple announces new product", "reuters.com"),
                (tesla_url, "Tesla recall", "reuters.com"),
            ],
            supports=[(end_apple, [0]), (end_tesla, [1])],
        )
        client = _make_async_client(response)

        with (
            patch("metaculus_bot.research.gemini_search.genai.Client", return_value=client),
            patch(
                "metaculus_bot.research.gemini_search.resolve_search_redirects",
                new=AsyncMock(
                    return_value={
                        apple_url: "https://reuters.com/apple",
                        tesla_url: "https://reuters.com/tesla",
                    }
                ),
            ),
        ):
            from metaculus_bot.research.gemini_search import invoke_gemini_grounded

            out = await invoke_gemini_grounded("prompt about apple and tesla")

        # Self-cited links are resolved and numbered while the real pydantic
        # Segment / GroundingSupport instances still exercise SDK schema construction.
        assert "Apple source [1]" in out
        assert "Tesla source [2]" in out

        # Sources block cites resolved domains and URLs, never the redirect blob.
        assert "### Sources" in out
        assert "[1] reuters.com — https://reuters.com/apple" in out
        assert "[2] reuters.com — https://reuters.com/tesla" in out
        assert "vertexaisearch" not in out
        assert "grounding-api-redirect" not in out

    @pytest.mark.asyncio
    async def test_real_response_with_chunks_but_no_supports(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Chunks present, supports None — sources block appended, no inline markers."""
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

        source_a_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/a"
        source_b_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/b"
        response = _make_real_grounding_response(
            text=(
                f"Some research finding [Source A]({source_a_url}) "
                f"without explicit segment mapping [Source B]({source_b_url})."
            ),
            chunks=[(source_a_url, "Source A", "example.com"), (source_b_url, None, "b.example.com")],
            supports=None,
        )
        client = _make_async_client(response)

        with (
            patch("metaculus_bot.research.gemini_search.genai.Client", return_value=client),
            patch(
                "metaculus_bot.research.gemini_search.resolve_search_redirects",
                new=AsyncMock(
                    return_value={
                        source_a_url: "https://example.com/a",
                        source_b_url: "https://b.example.com/b",
                    }
                ),
            ),
        ):
            from metaculus_bot.research.gemini_search import invoke_gemini_grounded

            out = await invoke_gemini_grounded("prompt")

        # The self-cited links are numbered even though supports was None.
        assert "Source A [1]" in out
        assert "Source B [2]" in out
        # The Sources section is appended for both resolved targets.
        assert "### Sources" in out
        assert "[1] example.com — https://example.com/a" in out
        assert "[2] b.example.com — https://b.example.com/b" in out
        assert "vertexaisearch" not in out

    @pytest.mark.asyncio
    async def test_real_response_with_empty_chunks_list_suppressed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Grounding metadata present but chunks=[] — the cited-link floor suppresses the section.

        This is the Q38195 shape against real SDK types: grounding fired (metadata present) but
        returned no chunks, so the model text is ungrounded and must not reach forecasters.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

        response = _make_real_grounding_response(
            text="Plain text with no citations.",
            chunks=[],
            supports=None,
        )
        client = _make_async_client(response)

        with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=client):
            from metaculus_bot.research.gemini_search import invoke_gemini_grounded

            out = await invoke_gemini_grounded("prompt")

        assert out == ""

    @pytest.mark.asyncio
    async def test_real_response_with_none_grounding_metadata_suppressed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Candidate with grounding_metadata=None — the cited-link floor suppresses the section.

        Grounding never fired, so the answer is pure parametric text; the floor drops it.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

        candidate = genai_types.Candidate(
            grounding_metadata=None,
            content=genai_types.Content(parts=[genai_types.Part(text="ungrounded answer")]),
        )
        response = genai_types.GenerateContentResponse(candidates=[candidate])
        client = _make_async_client(response)

        with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=client):
            from metaculus_bot.research.gemini_search import invoke_gemini_grounded

            out = await invoke_gemini_grounded("prompt")

        assert out == ""

    @pytest.mark.asyncio
    async def test_real_response_uses_genai_property_for_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sanity: the SDK's ``response.text`` property pulls from candidates[0].content.parts[0].text.

        If a future SDK changes that derivation (e.g. `parts[0].text` → `parts[0].text_value`),
        our formatter — which reads ``response.text or ""`` — would silently produce the empty
        string and we'd ship "no research" payloads to the LLM. Pin the property contract here
        so an SDK regression fails this test loudly. A cited link clears the cited-link floor so
        the body text (which is what this test asserts on) actually flows through.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

        blob = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/x"
        response = _make_real_grounding_response(
            text=f"The actual response body. [Source]({blob})",
            chunks=[(blob, "Source", "example.com")],
            supports=None,
        )
        # Guard the property contract directly:
        assert response.text == f"The actual response body. [Source]({blob})"

        client = _make_async_client(response)
        with (
            patch("metaculus_bot.research.gemini_search.genai.Client", return_value=client),
            patch(
                "metaculus_bot.research.gemini_search.resolve_search_redirects",
                new=AsyncMock(return_value={blob: "https://example.com/source"}),
            ),
        ):
            from metaculus_bot.research.gemini_search import invoke_gemini_grounded

            out = await invoke_gemini_grounded("prompt")

        assert out.startswith("The actual response body. Source [1]")


class TestGeminiToolWiringAgainstRealTypes:
    """The tools list passed to generate_content must contain real ``Tool`` objects
    with ``google_search`` and ``url_context`` attributes populated. The SDK normalizes
    our ``[{"google_search": {}}, {"url_context": {}}]`` dicts into pydantic ``Tool``
    instances; if it ever stops doing that (or renames the attribute), our formatter
    would happily continue calling the API with unrecognized tools and grounding would
    silently stop firing.
    """

    @pytest.mark.asyncio
    async def test_tools_normalize_to_real_tool_instances_with_grounding_attrs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

        response = _make_real_grounding_response(text="t", chunks=None, supports=None)
        client = _make_async_client(response)

        with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=client):
            from metaculus_bot.research.gemini_search import gemini_search_provider

            provider = gemini_search_provider()
            q = MagicMock()
            q.question_text = "Will X happen?"
            await provider(q)

        config = client.aio.models.generate_content.await_args.kwargs["config"]
        # The SDK normalizes tool dicts to real ``genai_types.Tool`` instances.
        assert all(isinstance(t, genai_types.Tool) for t in config.tools), (
            f"Expected all tools to be real genai_types.Tool instances; got {[type(t) for t in config.tools]}"
        )
        google_search_tools = [t for t in config.tools if t.google_search is not None]
        url_context_tools = [t for t in config.tools if t.url_context is not None]
        assert len(google_search_tools) == 1, "Expected exactly one tool with .google_search populated"
        assert len(url_context_tools) == 1, "Expected exactly one tool with .url_context populated"


# AskNews — formatter pinned against real SDK types.


class TestAskNewsFormatterAgainstRealTypes:
    """Formatter coverage using real ``asknews_sdk`` pydantic models.

    The bot's `_format_asknews_dual_sections` reads attributes like
    ``article.eng_title``, ``article.summary``, ``article.pub_date``,
    ``article.article_url``, ``article.source_id``. Build real
    ``SearchResponseDictItem`` instances and verify the formatted output —
    so an upstream rename (e.g. ``eng_title`` → ``english_title``) crashes
    construction here.
    """

    def test_dual_section_formatter_renders_real_articles(self) -> None:
        from metaculus_bot.research.providers import _format_asknews_dual_sections

        hist_old = _make_real_asknews_article(
            eng_title="Background article from a year ago",
            summary="Long-running context that frames the question.",
            article_url="https://example.com/news/historical-1",
            pub_date=_dt.datetime(2025, 6, 1, 12, 0, tzinfo=_dt.UTC),
            source_id="Reuters",
        )
        hist_recent = _make_real_asknews_article(
            eng_title="More recent background article",
            summary="Slightly more recent piece that still belongs in historical.",
            article_url="https://example.com/news/historical-2",
            pub_date=_dt.datetime(2026, 1, 15, 12, 0, tzinfo=_dt.UTC),
            source_id="AP",
        )
        hot = _make_real_asknews_article(
            eng_title="Breaking development today",
            summary="Hot off the press.",
            article_url="https://example.com/news/hot-1",
            pub_date=_dt.datetime(2026, 5, 13, 9, 0, tzinfo=_dt.UTC),
            source_id="BBC",
        )

        out = _format_asknews_dual_sections(hot_articles=[hot], historical_articles=[hist_old, hist_recent])

        # Both required section headers are present, in the documented order.
        assert "## Historical Context & Background" in out
        assert "## Recent Developments & Current News" in out
        assert out.index("## Historical Context & Background") < out.index("## Recent Developments & Current News")

        # Content from all three articles surfaces.
        assert "Background article from a year ago" in out
        assert "More recent background article" in out
        assert "Breaking development today" in out
        assert "Long-running context that frames the question." in out
        assert "Hot off the press." in out

        # Source attribution: source_id appears as Markdown link text.
        assert "[Reuters]" in out
        assert "[AP]" in out
        assert "[BBC]" in out

        # Sort order within historical: most recent first.
        idx_hist_recent = out.index("More recent background article")
        idx_hist_old = out.index("Background article from a year ago")
        assert idx_hist_recent < idx_hist_old, "Historical articles must be sorted most-recent-first"

    def test_cross_section_url_dedup_drops_duplicate_hot_articles(self) -> None:
        """Hot article duplicating a historical URL is dropped; historical wins."""
        from metaculus_bot.research.providers import _format_asknews_dual_sections

        url = "https://example.com/news/shared"

        hist = _make_real_asknews_article(
            eng_title="Historical version of the same story",
            summary="Historical framing.",
            article_url=url,
            pub_date=_dt.datetime(2026, 1, 1, 12, 0, tzinfo=_dt.UTC),
            source_id="HistoricalSource",
        )
        hot_dup = _make_real_asknews_article(
            eng_title="Hot version of the same story",
            summary="Hot framing of identical URL.",
            article_url=url,
            pub_date=_dt.datetime(2026, 5, 13, 12, 0, tzinfo=_dt.UTC),
            source_id="HotSource",
        )
        hot_unique = _make_real_asknews_article(
            eng_title="Genuinely new hot article",
            summary="Different URL, retained.",
            article_url="https://example.com/news/unique-hot",
            pub_date=_dt.datetime(2026, 5, 13, 13, 0, tzinfo=_dt.UTC),
            source_id="UniqueSource",
        )

        out = _format_asknews_dual_sections(
            hot_articles=[hot_dup, hot_unique],
            historical_articles=[hist],
        )

        # Historical wins on cross-section dedup.
        assert "Historical version of the same story" in out
        assert "Hot version of the same story" not in out, (
            "Hot article duplicating a historical URL should have been dropped"
        )
        # Unique hot article retained.
        assert "Genuinely new hot article" in out

    def test_empty_inputs_produce_nothing_at_all(self) -> None:
        """Both lists empty → ``""``, not a prose "no articles" sentence.

        The sentence used to be the assertion here. It was the defect: chars>0 made the
        orchestrator report ``ok``, and the summarizer wrote a briefing from it.
        """
        from metaculus_bot.research.providers import _format_asknews_dual_sections

        out = _format_asknews_dual_sections(hot_articles=[], historical_articles=[])

        assert out == ""


class TestAskNewsSDKResponseShapeContract:
    """End-to-end contract test: a real ``AsyncAskNewsSDK.news.search_news`` invocation
    returns a ``SearchResponse`` whose ``.as_dicts`` is a list of ``SearchResponseDictItem``.
    Our provider reads ``hot_response.as_dicts`` and passes the list straight to the
    formatter; if the SDK ever changes that attribute name (e.g. ``as_dicts`` →
    ``items``), the live call would 5xx-without-AttributeError but the local mock test
    would silently keep passing because the attribute name is hardcoded both sides.

    This test pins the contract by:
    1. Constructing a real ``SearchResponse`` (not a SimpleNamespace) with real items.
    2. Patching ``AsyncAskNewsSDK`` so the provider's `await sdk.news.search_news(...)`
       resolves to that real object.
    3. Asserting the formatted output mentions the article — proving the
       ``as_dicts`` access path is real-typed end-to-end.
    """

    @pytest.mark.asyncio
    async def test_provider_consumes_real_search_response_as_dicts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from asknews_sdk.dto.news import SearchResponse

        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "test-id")
        monkeypatch.setenv("ASKNEWS_SECRET", "test-secret")
        # Skip the real wait_for retries.
        monkeypatch.setattr("metaculus_bot.research.providers.ASKNEWS_WALL_TIMEOUT", 5.0)

        # Construct real SearchResponse objects (not a SimpleNamespace).
        hot_article = _make_real_asknews_article(
            eng_title="Hot story headline",
            summary="Hot story body.",
            article_url="https://example.com/news/hot-real",
            source_id="WireService",
        )
        historical_article = _make_real_asknews_article(
            eng_title="Historical story headline",
            summary="Historical story body.",
            article_url="https://example.com/news/hist-real",
            source_id="OldWire",
        )
        # hit_cache added in asknews 0.13.x (Optional[bool], defaults None at runtime);
        # pass it explicitly so basedpyright's synthesized __init__ is satisfied.
        hot_response = SearchResponse(as_dicts=[hot_article], as_string=None, offset=0, hit_cache=None)
        hist_response = SearchResponse(as_dicts=[historical_article], as_string=None, offset=0, hit_cache=None)

        # Async-context-manager mock for AsyncAskNewsSDK(...).
        fake_sdk = MagicMock()
        fake_sdk.news = MagicMock()
        # Two calls in sequence: HOT then HISTORICAL.
        fake_sdk.news.search_news = AsyncMock(side_effect=[hot_response, hist_response])

        sdk_cm = MagicMock()
        sdk_cm.__aenter__ = AsyncMock(return_value=fake_sdk)
        sdk_cm.__aexit__ = AsyncMock(return_value=None)

        # Skip the 10-second sleeps in the provider — they're not under test here.
        async def _no_sleep(_seconds: float) -> None:
            return None

        with (
            patch("asknews_sdk.AsyncAskNewsSDK", return_value=sdk_cm),
            patch("metaculus_bot.research.providers.asyncio.sleep", new=_no_sleep),
            patch("metaculus_bot.research.providers._asknews_rate_gate", new=AsyncMock()),
        ):
            from metaculus_bot.research.providers import _asknews_provider

            provider = _asknews_provider()
            # Provider duck-types `.question_text`; a SimpleNamespace stands in for the question.
            q = SimpleNamespace(question_text="Will the topic resolve YES?")
            out = await provider(cast(MetaculusQuestion, q))

        # Both articles surface, proving the .as_dicts → SearchResponseDictItem path
        # threads end-to-end with real-typed objects.
        assert "Hot story headline" in out
        assert "Historical story headline" in out
        assert "Hot story body." in out
        assert "Historical story body." in out
        assert "[WireService]" in out
        assert "[OldWire]" in out

        # Both calls fired exactly once with the question text.
        assert fake_sdk.news.search_news.await_count == 2
        for call in fake_sdk.news.search_news.await_args_list:
            assert call.kwargs["query"] == "Will the topic resolve YES?"
