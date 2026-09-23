"""Tests for the Gemini grounded search research provider.

These tests mock the google-genai SDK at the module level; no live API calls.
Patterns mirror ``tests/test_native_search_provider.py``.
"""

import asyncio
import logging
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types as genai_types

from metaculus_bot.research.gemini_search import _strip_model_citation_indices
from metaculus_bot.research.provider_diagnostics import _is_lost_source, pop_provider_detail


def _make_q(text: str) -> MagicMock:
    """Build a minimal MetaculusQuestion-shaped mock for the new ResearchCallable
    contract. Tests only care about question_text on this path."""
    q = MagicMock()
    q.question_text = text
    return q


# ---------------------------------------------------------------------------
# Canned response helpers (for grounding metadata tests)
# ---------------------------------------------------------------------------


class CannedWebChunk:
    def __init__(self, uri: str, title: str | None, domain: str | None = None) -> None:
        self.web = SimpleNamespace(uri=uri, title=title, domain=domain)


class CannedSegment:
    def __init__(self, end_index: int, text: str) -> None:
        self.end_index = end_index
        self.text = text


class CannedSupport:
    def __init__(self, seg: CannedSegment, indices: list[int]) -> None:
        self.segment = seg
        self.grounding_chunk_indices = indices


class CannedStatus:
    """A url_retrieval_status enum stand-in exposing ``.name`` like the real SDK enum."""

    def __init__(self, name: str) -> None:
        self.name = name


class CannedUrlMeta:
    """Mirror of ``google.genai.types.UrlMetadata`` (retrieved_url + url_retrieval_status)."""

    def __init__(self, retrieved_url: str | None, url_retrieval_status: object) -> None:
        self.retrieved_url = retrieved_url
        self.url_retrieval_status = url_retrieval_status


def _make_response(
    text: str,
    chunks: list[CannedWebChunk] | None = None,
    supports: list[CannedSupport] | None = None,
    url_metadata: Sequence[object] | None = None,
    web_search_queries: list[str] | None = None,
    usage_metadata: object | None = None,
    model_version: str | None = None,
) -> SimpleNamespace:
    # ``web_search_queries`` is a declared field on the real SDK GroundingMetadata
    # (Optional[list[str]]); the zero-chunk floor reads it to size its WARN, so the
    # attribute must always be present here (defaults to None) to mirror the SDK.
    metadata = SimpleNamespace(
        grounding_chunks=chunks,
        grounding_supports=supports,
        web_search_queries=web_search_queries,
    )
    url_context_metadata = SimpleNamespace(url_metadata=url_metadata) if url_metadata is not None else None
    candidate = SimpleNamespace(
        grounding_metadata=metadata,
        url_context_metadata=url_context_metadata,
    )
    # ``usage_metadata`` / ``model_version`` are declared response-level fields the
    # GEMINI_USAGE marker reads; both default to None (the SDK's own default) so a test
    # that doesn't care about spend still hands over an SDK-shaped object.
    return SimpleNamespace(
        text=text,
        candidates=[candidate],
        usage_metadata=usage_metadata,
        model_version=model_version,
    )


def _make_usage(
    prompt: int | None = 1200,
    tool_use: int | None = 340,
    candidates: int | None = 900,
    thoughts: int | None = 2600,
    total: int | None = 5040,
) -> SimpleNamespace:
    """A ``GenerateContentResponseUsageMetadata`` stand-in (every field Optional on the SDK)."""
    return SimpleNamespace(
        prompt_token_count=prompt,
        tool_use_prompt_token_count=tool_use,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        total_token_count=total,
    )


def _make_client_with_response(response: object) -> MagicMock:
    """Build a MagicMock Client whose aio.models.generate_content awaits to ``response``."""
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# build_gemini_client
# ---------------------------------------------------------------------------


def test_builder_raises_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing GOOGLE_API_KEY should raise. The grounded-search side has no
    donated/shared key path — Google AI Studio doesn't offer one — so this is
    the only key gate to test here.
    """
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    from metaculus_bot.research.gemini_search import build_gemini_client

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        build_gemini_client()


# ---------------------------------------------------------------------------
# gemini_search_provider: model selection & tool wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_uses_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no GEMINI_SEARCH_MODEL env set, default slug is used."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.delenv("GEMINI_SEARCH_MODEL", raising=False)

    # Grounded, so the one call is the whole exchange (an ungrounded response earns a retry).
    response = _make_response(
        "some research text", chunks=[CannedWebChunk(uri="https://src.example/a", title="A", domain="src.example")]
    )
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider()
        await provider(_make_q("Will X happen?"))

    assert fake_client.aio.models.generate_content.await_count == 1
    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash"
    # The question_text must actually reach the SDK (guard against broken f-string interpolation).
    assert "Will X happen?" in call_kwargs["contents"]


@pytest.mark.asyncio
async def test_provider_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """GEMINI_SEARCH_MODEL env var overrides the default."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_SEARCH_MODEL", "gemini-2.5-flash")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider()
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_provider_uses_explicit_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ``model_slug=`` param takes precedence over env var."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_SEARCH_MODEL", "gemini-2.5-flash")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider(model_slug="gemini-explicit-override")
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-explicit-override"


@pytest.mark.asyncio
async def test_provider_attaches_google_search_and_url_context_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generate_content config must include both the GoogleSearch and url_context tools."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider()
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    config = call_kwargs["config"]
    tools = list(config.tools)
    assert len(tools) == 2
    # The SDK normalizes the {"google_search": {}} / {"url_context": {}} dicts into
    # pydantic Tool objects with the corresponding attribute populated.
    google_search_configured = any(getattr(t, "google_search", None) is not None for t in tools)
    url_context_configured = any(getattr(t, "url_context", None) is not None for t in tools)
    assert google_search_configured
    assert url_context_configured


# ---------------------------------------------------------------------------
# gemini_search_provider: prompt content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_benchmarking_carve_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_benchmarking=True: prompt contains 'benchmarking run' and no market/crowd-odds ask."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider(is_benchmarking=True)
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    prompt = call_kwargs["contents"]
    assert "benchmarking run" in prompt
    assert "Market-implied or crowd odds" not in prompt


@pytest.mark.asyncio
async def test_non_benchmarking_includes_prediction_markets(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_benchmarking=False: prompt includes the market/crowd-odds bullet."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider(is_benchmarking=False)
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    prompt = call_kwargs["contents"]
    assert "Market-implied or crowd odds" in prompt
    assert "benchmarking run" not in prompt


@pytest.mark.asyncio
async def test_prompt_carries_the_mc_ballot(monkeypatch: pytest.MonkeyPatch) -> None:
    """An MC question's option list must reach the grounded-search prompt: a searching model
    can only query candidate names it has been shown (the q44952 gap — no research stage ever
    saw the ballot)."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    question = _make_q("Who will win the World Yo-Yo Contest?")
    question.options = ["Mir Kim", "Hunter Feuerstein", "Other"]

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider(is_benchmarking=False)
        await provider(question)

    prompt = fake_client.aio.models.generate_content.await_args.kwargs["contents"]
    assert "Options (in resolution order): Mir Kim | Hunter Feuerstein | Other" in prompt


# ---------------------------------------------------------------------------
# _format_grounded_response behavior (via invoke_gemini_grounded)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_citations_appended_to_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Response with grounding chunks ends with a '### Sources' block citing title + domain."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    blob = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123"
    chunks = [
        CannedWebChunk(uri=blob, title="Example One", domain="example.com"),
        CannedWebChunk(uri=blob, title="Example Two", domain="two.example.org"),
    ]
    response = _make_response("body text", chunks=chunks, supports=None)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "### Sources" in out
    assert "Example One — example.com" in out
    assert "Example Two — two.example.org" in out
    # The opaque vertexaisearch redirect blob must never reach the forecaster.
    assert "vertexaisearch" not in out
    assert "grounding-api-redirect" not in out
    # Sources comes after body text
    assert out.index("body text") < out.index("### Sources")


@pytest.mark.asyncio
async def test_sources_render_domain_not_redirect_blob(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sources cite title + domain, drop the redirect blob, and stay 1:1 with chunks.

    Covers the title+domain case, the domain-only fallback (no title), the
    title-only fallback (no domain), and a duplicate domain — which keeps its own
    index so the inline [N] markers remain valid (we do not renumber/dedup).
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    blob = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/" + ("X" * 200)
    chunks = [
        CannedWebChunk(uri=blob, title="Al Jazeera", domain="aljazeera.com"),
        CannedWebChunk(uri=blob, title=None, domain="senate.gov"),  # no title -> domain only
        CannedWebChunk(uri=blob, title="Reuters", domain=None),  # no domain -> title only
        CannedWebChunk(uri=blob, title="Al Jazeera", domain="aljazeera.com"),  # duplicate keeps its index
    ]
    response = _make_response("body text", chunks=chunks, supports=None)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "vertexaisearch" not in out
    assert "grounding-api-redirect" not in out
    assert "[1] Al Jazeera — aljazeera.com" in out
    assert "[2] senate.gov" in out
    assert "[3] Reuters" in out
    assert "[4] Al Jazeera — aljazeera.com" in out


@pytest.mark.asyncio
async def test_inline_citation_markers_inserted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A support mapping a segment end_index to chunk 0 produces a ``[1]`` marker after that offset.

    With multiple supports, reverse-iteration must preserve earlier offsets.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    text = "Alpha fact. Beta fact."
    end_alpha = text.index("Alpha fact.") + len("Alpha fact.")
    end_beta = text.index("Beta fact.") + len("Beta fact.")

    blob = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/zzz"
    chunks = [
        CannedWebChunk(uri=blob, title="A", domain="a.example.com"),
        CannedWebChunk(uri=blob, title="B", domain="b.example.com"),
    ]
    supports = [
        CannedSupport(seg=CannedSegment(end_index=end_alpha, text="Alpha fact."), indices=[0]),
        CannedSupport(seg=CannedSegment(end_index=end_beta, text="Beta fact."), indices=[1]),
    ]
    response = _make_response(text, chunks=chunks, supports=supports)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "Alpha fact.[1]" in out
    assert "Beta fact.[2]" in out
    # The sources block must also be appended whenever chunks are present — this is
    # the common production path (chunks + supports together), not chunks-only.
    assert "### Sources" in out
    assert "a.example.com" in out
    assert "vertexaisearch" not in out


@pytest.mark.asyncio
async def test_inline_citation_markers_respect_utf8_byte_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: Google's segment.end_index is a UTF-8 BYTE offset, not a codepoint index.

    Derived from the real Q578 gemini_search record
    (backtests/research_archive/raw/29718821482.jsonl). The model text contains an
    em-dash (``—``, 3 bytes / 1 codepoint) at codepoint 194, so at the first
    support's byte offset 291 the byte cursor runs 2 ahead of the codepoint cursor.
    The pre-fix code sliced the Python str at codepoint 291 and produced the
    observed mid-word corruption ``"...civilization. T[1]hese estimates"``. The
    correct byte-space splice lands the marker at the sentence boundary:
    ``"...civilization.[1] These estimates"``.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    # Verbatim opening of the Q578 raw payload text, trimmed to the minimal
    # reproducing snippet (through the first two sentences).
    text = (
        "There is no scientific consensus that humans will go extinct before 2100, "
        "but researchers and specialized forecasters have produced a wide range of "
        'probabilistic estimates for "existential risk"—defined as events that '
        "could cause human extinction or the permanent collapse of civilization. "
        "These estimates vary significantly."
    )
    # Sanity-anchor the fixture to the real payload's geometry: em-dash at
    # codepoint 194, first support end_index at byte 291.
    assert text.index("—") == 194
    byte_end_index = 291
    # The byte offset points at the space after "civilization." — one past the period.
    assert text.encode("utf-8")[:byte_end_index].decode("utf-8").endswith("collapse of civilization.")

    blob = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC"
    chunks = [CannedWebChunk(uri=blob, title="RAND", domain="rand.org")]
    supports = [CannedSupport(seg=CannedSegment(end_index=byte_end_index, text=""), indices=[0])]
    response = _make_response(text, chunks=chunks, supports=supports)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    # The marker lands cleanly at the sentence boundary...
    assert "collapse of civilization.[1] These estimates" in out
    # ...and the pre-fix mid-word corruption is gone.
    assert "T[1]hese" not in out
    assert "civilization. T[1]" not in out


@pytest.mark.asyncio
async def test_no_candidates_is_suppressed_by_the_grounded_chunk_floor(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Text with no candidates has no grounding evidence, so the floor must refuse it.

    This test used to assert the opposite ("returns its plain text") — an early return that
    walked straight past the Q38195 fabrication guard. The branch is unreachable on today's
    SDK (``response.text`` derives from a candidate), but a hole in a fabrication guard
    shouldn't rest on an SDK invariant we don't own.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = SimpleNamespace(text="plain ungrounded body", candidates=[])
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("WARNING"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out == ""
    assert "GEMINI_UNGROUNDED_SUPPRESSED" in caplog.text
    assert "queries=0" in caplog.text


@pytest.mark.asyncio
async def test_empty_response_text_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """response.text == '' short-circuits to empty, even when candidates + grounding metadata are populated.

    Isolating the ``not text`` guard: if that early-return regresses (e.g. moved below the candidates
    check), the chunks/supports path would produce a non-empty "\\n\\n### Sources\\n[1] ..." string and
    this assertion would fail.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    chunks = [CannedWebChunk(uri="https://example.com/1", title="Example One")]
    supports = [CannedSupport(seg=CannedSegment(end_index=0, text=""), indices=[0])]
    response = _make_response("", chunks=chunks, supports=supports)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out == ""


@pytest.mark.asyncio
async def test_zero_grounding_chunks_suppresses_ungrounded_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression (Q38195): grounding metadata present, chunks empty, many search queries.

    On the real Q38195 record Gemini issued 30 web-search queries and Google
    returned ZERO grounding chunks, yet the model emitted a confident, fabricated
    contract table with fake ``[primary]`` tags. Pre-fix, the formatter passed that
    ungrounded parametric text through verbatim (the fabrication reached
    forecasters). Post-fix, a response whose grounding fired-but-returned-no-chunks
    must be suppressed entirely (``""``) so the section is omitted upstream, and a
    greppable WARN must fire.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    fabricated = (
        "Generative AI has emerged as a primary driver of labor tension. "
        "Key contract expirations: Boeing IAM 837 (July 22, 2029) [primary]."
    )
    # grounding_metadata present (grounding fired), grounding_chunks empty, and
    # a populated web_search_queries list — the exact Q38195 shape.
    response = _make_response(
        fabricated,
        chunks=None,
        supports=None,
        web_search_queries=[f"query {i}" for i in range(30)],
    )
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("WARNING"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=38195)

    # The ungrounded fabrication must NOT reach the forecaster.
    assert out == ""
    assert "[primary]" not in out
    assert "Boeing" not in out
    # A greppable WARN with the query count must fire.
    assert "GEMINI_UNGROUNDED_SUPPRESSED" in caplog.text
    assert "queries=30" in caplog.text


@pytest.mark.asyncio
async def test_ungrounded_suppression_records_a_provider_loss_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suppression must be visible where degradation is read, not only in the WARN.

    Suppressing correctly still costs the whole Google research leg, and the loss used to
    be invisible in all three places the diagnostics convention teaches you to look: no
    counter moves (the provider didn't raise, so ``provider_failure_count`` stays 0 and
    ``alertable_count`` stays flat), no ``record_provider_detail`` call, and the resulting
    ``""`` maps to ProviderResult status ``empty`` — byte-identical to a healthy Gemini
    call that legitimately found nothing. If grounding degrades persistently (AI Studio
    prepaid exhaustion, an SDK contract shift), every forecast in the season loses the
    Google leg behind a normal-looking diagnostics line. So the suppression now records a
    per-source loss token, mirroring ``_degraded_to_raw_articles`` for the summarizer.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = _make_response("Ungrounded prose.", chunks=None, supports=None, web_search_queries=["q"])
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=38195)

    assert out == ""
    detail = pop_provider_detail(38195, "gemini_search")
    token = detail["sources"]["grounding"]
    # _is_lost_source treats anything not starting with "ok"/"none" as a loss, which is
    # what renders the `lost=grounding:...` segment on the diagnostics line and in the
    # schema-v2 archive.
    assert _is_lost_source(token), f"the recorded token must read as a LOST source; got {token!r}"
    assert "ungrounded" in token


@pytest.mark.asyncio
async def test_url_context_read_survives_zero_search_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful url_context read IS grounding, so the text passes the chunk floor.

    Google's search tool grounded nothing (no chunks), but url_context retrieved a
    page, so this is genuine retrieval rather than the Q38195 parametric case. The
    text comes back unmodified — there are no chunks to cite, so no inline markers
    and no ``### Sources`` block.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = _make_response(
        "Read straight off the resolving page.",
        chunks=None,
        supports=None,
        url_metadata=[CannedUrlMeta("https://gov.example/report", CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"))],
    )
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=1)

    assert "Read straight off the resolving page." in out
    assert "### Sources" not in out


@pytest.mark.asyncio
async def test_malformed_supports_fall_back_to_unspliced_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A support whose end_index is not an integer must not cost us the response.

    The splice raises TypeError partway through mutating the byte buffer; the
    formatter falls back to the ORIGINAL text (never a half-spliced buffer), still
    renders the Sources block, and leaves a greppable WARN.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    blob = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123"
    bad_support = CannedSupport(cast(CannedSegment, SimpleNamespace(end_index="not-an-int", text="x")), [0])
    response = _make_response(
        "body text",
        chunks=[CannedWebChunk(uri=blob, title="Example One", domain="example.com")],
        supports=[bad_support],
    )
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("WARNING"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out.startswith("body text")
    assert "[1] Example One — example.com" in out
    assert "could not splice inline citations" in caplog.text


# ---------------------------------------------------------------------------
# Model-authored hierarchical citation indices (_strip_model_citation_indices)
# ---------------------------------------------------------------------------


class TestStripModelCitationIndices:
    """Gemini writes its OWN hierarchical ``[2.4.1]`` indices alongside the ``[N]`` markers
    our formatter splices from real grounding metadata: 173 of 323 archived sections carry
    them and they resolve to nothing we hold, so a forecaster cannot tell which
    brackets are checkable (scratch/residual_2026-08-31/gemini_search_audit/cutB_pattern.md
    §3.1). Every example below is a shape the archived corpus actually contains, except the
    preserved-numeric cases, which pin the false-positive boundary.
    """

    def test_removes_a_lone_index_and_the_space_it_leaves(self) -> None:
        assert _strip_model_citation_indices("tag more sharks [2.4.1]. The count") == "tag more sharks. The count"

    def test_removes_a_multi_token_group(self) -> None:
        assert _strip_model_citation_indices("office [1.1.1, 1.1.2]. He") == "office. He"

    def test_keeps_a_tier_tag_and_drops_the_index_beside_it(self) -> None:
        assert _strip_model_citation_indices("path of totality [A: NASA, 1.1.2].") == "path of totality [A: NASA]."

    def test_keeps_a_tier_tag_whose_index_is_space_separated(self) -> None:
        # q45081's shape: the index sits inside the tier item with no comma before it.
        assert _strip_model_citation_indices("[A: official 2.4.1, 3.2.5].") == "[A: official]."
        assert _strip_model_citation_indices("[B: Access Newswire 1.1.4, 3.3.4].") == "[B: Access Newswire]."

    def test_keeps_a_trailing_tier_grade_when_the_index_comes_first(self) -> None:
        # q44944's shape: ``index: tier`` rather than ``tier: outlet, index``.
        assert _strip_model_citation_indices("HPI rose [1.1.8, 2.1.4: A].") == "HPI rose [A]."

    def test_preserves_semicolon_separated_tier_groups(self) -> None:
        # q44841's shape: two tier tags in one bracket, semicolon-delimited.
        stripped = _strip_model_citation_indices("[B: Forbes, 1.6.4; C: Newsweek, 3.2.2].")
        assert stripped == "[B: Forbes; C: Newsweek]."

    def test_preserves_our_spliced_markers(self) -> None:
        text = "Alpha.[1] Beta.[12] Gamma.[1, 3] Delta.[11, 12]"
        assert _strip_model_citation_indices(text) == text

    def test_preserves_bracketed_quantities_and_version_like_tokens(self) -> None:
        """The false-positive boundary. Zero of the 2,609 archived dotted-in-bracket groups
        is a quantity, but a dotted token that carries a unit, a currency prefix, a trailing
        word, a 4-digit component (a year) or a 3-digit component (an IP octet) is content,
        not a citation index, so the rule must leave every one of these alone.
        """
        for text in (
            "gasoline [3.8%] higher",
            "priced at [$1.5] a share",
            "reached [1.5 million] viewers",
            "shipped in [v2.1.3] of the tool",
            "dated [2026.08] in the filing",
            "resolved [192.168.1.1] internally",
            "the [1.5-2.0] range",
        ):
            assert _strip_model_citation_indices(text) == text, text

    def test_is_idempotent(self) -> None:
        text = "office [1.1.1, 1.1.2]. NASA [A: NASA, 1.1.2] said.[1]"
        once = _strip_model_citation_indices(text)
        assert _strip_model_citation_indices(once) == once

    def test_empties_a_table_cell_without_eating_the_pipes(self) -> None:
        assert _strip_model_citation_indices("| **2024** | 2.2% | [1.4, 1.23] |") == "| **2024** | 2.2% | |"


@pytest.mark.asyncio
async def test_splice_runs_before_the_strip_and_sources_survive_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end ordering guard. ``_splice_inline_citations`` indexes the ORIGINAL response
    text by grounding-support BYTE offsets, so the strip must run AFTER it — stripping first
    would shift every offset and land the markers mid-word. The ``### Sources`` block's own
    ``[N]`` lines must come through untouched.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    text = "OCEARCH plans a 2026 expedition [2.4.1]. It targets Nova Scotia [A: OCEARCH, 2.4.4]."
    end_first = text.index("[2.4.1].") + len("[2.4.1].")
    blob = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/zzz"
    chunks = [CannedWebChunk(uri=blob, title="OCEARCH", domain="ocearch.org")]
    supports = [CannedSupport(seg=CannedSegment(end_index=end_first, text=""), indices=[0])]
    response = _make_response(text, chunks=chunks, supports=supports)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=44872)

    assert "OCEARCH plans a 2026 expedition.[1] It targets Nova Scotia [A: OCEARCH]." in out
    assert "2.4.1" not in out
    assert "2.4.4" not in out
    assert "[1] OCEARCH — ocearch.org" in out


@pytest.mark.asyncio
async def test_url_context_only_path_is_also_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The url_context escape branch renders model text to forecasters too, and nothing is
    spliced there (no chunks), so the strip is safe and the defect is identical.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = _make_response(
        "Read straight off the resolving page [1.2.3].",
        chunks=None,
        supports=None,
        url_metadata=[CannedUrlMeta("https://gov.example/report", CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"))],
    )
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=1)

    assert "Read straight off the resolving page." in out
    assert "1.2.3" not in out


# ---------------------------------------------------------------------------
# GEMINI_GROUNDING_DENSITY telemetry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grounding_density_marker_emitted_on_the_grounded_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Post-floor, 41% of passing responses carry <=3 grounding supports and the median
    response has one support per ~872 chars — the floor-immune surface where the embellishment
    rate lives. The marker makes "did that move" a query instead of a hand audit; it is
    deliberately NOT a gate (q44944's decisive, true ICE figure came out of a 1-support
    response). ``chars`` is the RAW model text, the denominator the audit measured.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    text = "Alpha fact. Beta fact."
    blob = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/zzz"
    chunks = [
        CannedWebChunk(uri=blob, title="A", domain="a.example.com"),
        CannedWebChunk(uri=blob, title="B", domain="b.example.com"),
    ]
    supports = [CannedSupport(seg=CannedSegment(end_index=len(text), text=""), indices=[1])]
    response = _make_response(text, chunks=chunks, supports=supports)
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("INFO"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        await invoke_gemini_grounded("prompt", qid=44944)

    assert f"GEMINI_GROUNDING_DENSITY: question=44944 chunks=2 supports=1 chars={len(text)}" in caplog.text


@pytest.mark.asyncio
async def test_grounding_density_marker_absent_when_the_floor_suppresses(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A suppressed response renders nothing, so it has no density to report — and counting it
    would put ungrounded parametric text in the same cohort as grounded sections.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = _make_response("Ungrounded prose.", chunks=None, supports=None, web_search_queries=["q"])
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("INFO"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        await invoke_gemini_grounded("prompt", qid=38195)

    assert "GEMINI_GROUNDING_DENSITY" not in caplog.text
    assert "GEMINI_UNGROUNDED_SUPPRESSED" in caplog.text


@pytest.mark.asyncio
async def test_grounding_density_marker_absent_on_the_url_context_only_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The marker measures google_search grounding density, so it is scoped to responses that
    carry grounding chunks. A url_context-only response has neither chunks nor supports: its
    density is undefined rather than zero, and a ``chunks=0 supports=0`` row would pool a
    structurally different response shape into the same cohort.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = _make_response(
        "Read straight off the resolving page.",
        chunks=None,
        supports=None,
        url_metadata=[CannedUrlMeta("https://gov.example/report", CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"))],
    )
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("INFO"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=1)

    assert out.startswith("Read straight off the resolving page.")
    assert "GEMINI_GROUNDING_DENSITY" not in caplog.text


# ---------------------------------------------------------------------------
# Unsupported-attribution check (rewrite_unsupported_attributions, wired at format time)
# ---------------------------------------------------------------------------

_BLOB = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/zzz"


@pytest.mark.asyncio
async def test_unsupported_attribution_rewritten_and_counted_on_the_grounded_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """cutB's worked example, end to end: q44953 claimed ``[A: NASA]`` for the path of
    totality while its own grounded domains were perlan.is / timeanddate.com, and 76% of
    the corpus's outlet-named tier tags are that shape. The tag also carries one of
    Gemini's hierarchical indices, so this pins the ORDER too — the index strip runs first
    and the attribution check still sees the tier tag behind it.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    text = "Reykjavik lies in the path of totality [A: NASA, 1.1.2]. Cloud cover is 76% [C: Time and Date, 1.3.4]."
    chunks = [
        CannedWebChunk(uri=_BLOB, title="perlan.is", domain="perlan.is"),
        CannedWebChunk(uri=_BLOB, title="timeanddate.com", domain="timeanddate.com"),
    ]
    response = _make_response(text, chunks=chunks, supports=None)
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("INFO"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=44953)

    assert "Reykjavik lies in the path of totality [unverified attribution]." in out
    assert "NASA" not in out
    # The outlet our own record DOES name survives, and so does the sentence it decorates.
    assert "Cloud cover is 76% [C: Time and Date]." in out
    assert "GEMINI_UNSUPPORTED_ATTRIBUTION: question=44953 tagged=2 unsupported=1 groups=1 labels=2" in caplog.text
    detail = pop_provider_detail(44953, "gemini_search")
    assert detail["counts"]["unsupported_attributions"] == 1
    assert detail["counts"]["tier_tags"] == 2


@pytest.mark.asyncio
async def test_fully_supported_attributions_record_zero_and_log_no_marker(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A zero recorded on a response the check RAN over is the measurement that makes the
    incidence a rate rather than a count of complaints; the marker stays quiet so the run
    log only carries responses that lost an attribution.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = _make_response(
        "Cloud cover is 76% [C: Time and Date].",
        chunks=[CannedWebChunk(uri=_BLOB, title="timeanddate.com", domain="timeanddate.com")],
        supports=None,
    )
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("INFO"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=44953)

    assert "[C: Time and Date]" in out
    assert "GEMINI_UNSUPPORTED_ATTRIBUTION" not in caplog.text
    counts = pop_provider_detail(44953, "gemini_search")["counts"]
    assert counts["unsupported_attributions"] == 0
    assert counts["tier_tags"] == 1


@pytest.mark.asyncio
async def test_untagged_response_records_zero_tier_tags_beside_zero_unsupported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The denominator has to ride along or the recorded zero is unreadable: this response
    carried NO tier tag, the test above carried one and it was backed, and both log nothing
    because the marker is gated on ``unsupported``. Without ``tier_tags`` the two archive
    identically as ``unsupported_attributions=0``, so "the model stopped tagging" — the
    over-compliance risk the prompt's carve-out addresses (F10) — would read as "every tag
    was backed". ``tier_tags`` counts OUTLET-named items only, so a generic ``[A: official]``
    still reads 0 here; the definitive check is a grep for the tag in the archived section.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = _make_response(
        "Cloud cover is 76%, per the forecast office [A: official].",
        chunks=[CannedWebChunk(uri=_BLOB, title="timeanddate.com", domain="timeanddate.com")],
        supports=None,
    )
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("INFO"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=44953)

    assert "[A: official]" in out
    assert "GEMINI_UNSUPPORTED_ATTRIBUTION" not in caplog.text
    counts = pop_provider_detail(44953, "gemini_search")["counts"]
    assert counts == {"tier_tags": 0, "unsupported_attributions": 0}


@pytest.mark.asyncio
async def test_attribution_check_skipped_when_no_chunk_carries_a_label(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Chunks with nothing renderable (one archived section, q44802) leave the check with
    no evidence base, so every tag stands and NOTHING is recorded — an absent count reads
    as "not measured" rather than as a clean response, and rewriting here would dress our
    own render failure as the model's embellishment.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = _make_response(
        "The notice was revised [A: official, B: Reuters].",
        chunks=[CannedWebChunk(uri=_BLOB, title=None, domain=None)],
        supports=None,
    )
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("INFO"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=44802)

    assert "[A: official, B: Reuters]" in out
    assert "### Sources" not in out
    assert "GEMINI_UNSUPPORTED_ATTRIBUTION" not in caplog.text
    assert "counts" not in pop_provider_detail(44802, "gemini_search")


@pytest.mark.asyncio
async def test_attribution_check_not_applied_on_the_suppressed_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A suppressed response renders nothing, so there is no forecaster-facing attribution
    to mark and no count to record; the recorded detail stays the grounding loss token.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = _make_response("Ungrounded prose [A: NASA].", chunks=None, supports=None, web_search_queries=["q"])
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("INFO"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=38195)

    assert out == ""
    assert "GEMINI_UNSUPPORTED_ATTRIBUTION" not in caplog.text
    detail = pop_provider_detail(38195, "gemini_search")
    assert "counts" not in detail
    assert "ungrounded" in detail["sources"]["grounding"]


@pytest.mark.asyncio
async def test_attribution_check_not_applied_on_the_url_context_only_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The url_context escape branch has no grounding chunks at all, so it has no
    grounded-domain list to check against — the same no-evidence-base rule as above,
    reached by a different route.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    response = _make_response(
        "Read straight off the resolving page [A: NASA].",
        chunks=None,
        supports=None,
        url_metadata=[CannedUrlMeta("https://gov.example/report", CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"))],
    )
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        caplog.at_level("INFO"),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt", qid=1)

    assert "Read straight off the resolving page [A: NASA]." in out
    assert "GEMINI_UNSUPPORTED_ATTRIBUTION" not in caplog.text
    assert "counts" not in pop_provider_detail(1, "gemini_search")


# ---------------------------------------------------------------------------
# url_context telemetry: extract_url_context_telemetry
# ---------------------------------------------------------------------------


def test_url_context_telemetry_empty_when_metadata_absent() -> None:
    """No candidates / no url_context_metadata yield reported=False; an empty url_metadata list
    yields reported=True (the tool fired but fetched nothing). All cases give zero counts + empty list."""
    from metaculus_bot.research.url_context_telemetry import extract_url_context_telemetry

    def extract(response: object) -> tuple[bool, int, int, list[tuple[str, str]]]:
        return extract_url_context_telemetry(cast(genai_types.GenerateContentResponse, response))

    # No url_context signal at all → reported=False.
    assert extract(SimpleNamespace(text="x", candidates=[])) == (False, 0, 0, [])
    assert extract(SimpleNamespace(text="x", candidates=None)) == (False, 0, 0, [])
    # candidate present but url_context_metadata is None → still no signal.
    assert extract(_make_response("x", url_metadata=None)) == (False, 0, 0, [])
    # url_context_metadata present but url_metadata list is empty → fired-but-empty (reported=True).
    assert extract(_make_response("x", url_metadata=[])) == (True, 0, 0, [])


def test_url_context_telemetry_parses_success_and_error() -> None:
    """Two url_metadata entries (1 SUCCESS via enum-like .name, 1 ERROR via plain string).

    Proves both counts and the (status_name, url) list, and that the status is coerced
    defensively whether it arrives as an enum-with-.name or a plain string.
    """
    from metaculus_bot.research.url_context_telemetry import extract_url_context_telemetry

    url_metadata = [
        CannedUrlMeta(
            retrieved_url="https://example.com/ok",
            url_retrieval_status=CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"),
        ),
        CannedUrlMeta(
            retrieved_url="https://example.com/bad",
            url_retrieval_status="URL_RETRIEVAL_STATUS_ERROR",
        ),
    ]
    response = cast(genai_types.GenerateContentResponse, _make_response("body", url_metadata=url_metadata))

    reported, n_total, n_success, entries = extract_url_context_telemetry(response)

    assert reported is True
    assert n_total == 2
    assert n_success == 1
    assert entries == [
        ("URL_RETRIEVAL_STATUS_SUCCESS", "https://example.com/ok"),
        ("URL_RETRIEVAL_STATUS_ERROR", "https://example.com/bad"),
    ]


def test_url_context_telemetry_coerces_none_valued_fields() -> None:
    """A real ``UrlMetadata`` with both fields present-but-None coerces to ``("None", "")`` rather
    than raising — telemetry must never break the research path. ``UrlMetadata`` declares both fields
    as Optional (extra='forbid'), so a None-valued entry is the genuine degenerate case the SDK can
    emit; it is counted, not silently skipped.
    """
    from metaculus_bot.research.url_context_telemetry import extract_url_context_telemetry

    degenerate = genai_types.UrlMetadata(retrieved_url=None, url_retrieval_status=None)
    good = CannedUrlMeta(
        retrieved_url="https://example.com/ok",
        url_retrieval_status=CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"),
    )
    response = cast(genai_types.GenerateContentResponse, _make_response("body", url_metadata=[degenerate, good]))

    reported, n_total, n_success, entries = extract_url_context_telemetry(response)

    assert reported is True
    assert n_total == 2
    assert n_success == 1
    assert len(entries) == 2
    assert ("None", "") in entries
    assert ("URL_RETRIEVAL_STATUS_SUCCESS", "https://example.com/ok") in entries


# ---------------------------------------------------------------------------
# url_context telemetry: marker persisted into invoke_gemini_grounded output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_url_context_fetches_marker_in_returned_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """When url_context fetched URLs, the returned text carries a greppable subsection — but ONLY the
    SUCCESSFUL fetches are listed inline (those URLs were actually read, so they are real research
    context). A co-occurring failed fetch must stay out of the forecaster-facing text (it is captured
    in the INFO audit log instead).
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    url_metadata = [
        CannedUrlMeta(
            retrieved_url="https://example.com/ok",
            url_retrieval_status=CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"),
        ),
        CannedUrlMeta(
            retrieved_url="https://example.com/bad",
            url_retrieval_status="URL_RETRIEVAL_STATUS_ERROR",
        ),
    ]
    response = _make_response("body text", url_metadata=url_metadata)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "### URL Context Fetches" in out
    assert "URL_RETRIEVAL_STATUS_SUCCESS — https://example.com/ok" in out
    # Failed fetches must NOT appear inline — only successfully-read URLs are research context.
    assert "URL_RETRIEVAL_STATUS_ERROR" not in out
    assert "https://example.com/bad" not in out
    # The terse "none" marker must NOT appear when at least one fetch succeeded.
    assert "_url_context: none_" not in out
    # The fetch subsection must stay out of the grounding-only Sources block.
    assert "### Sources" not in out


@pytest.mark.asyncio
async def test_url_context_none_marker_when_no_fetches(monkeypatch: pytest.MonkeyPatch) -> None:
    """When url_context fired but fetched nothing (empty url_metadata), a terse greppable
    ``_url_context: none_`` marker is appended — distinguishing 'fired but empty' from
    'we don't capture it' — and NO fetch list is emitted.

    The response carries a grounding chunk so it clears the grounded-chunk floor; this
    isolates the url_context marker layer, which is orthogonal to grounding.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    chunks = [CannedWebChunk(uri="https://vertexaisearch/redirect", title="Src", domain="src.example.com")]
    response = _make_response("body text", chunks=chunks, url_metadata=[])
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "_url_context: none_" in out
    assert "### URL Context Fetches" not in out
    assert out.startswith("body text")


@pytest.mark.asyncio
async def test_no_url_context_marker_when_tool_did_not_fire(monkeypatch: pytest.MonkeyPatch) -> None:
    """When url_context produced no signal at all (no url_context_metadata on the candidate),
    the returned text must carry no url_context marker of any kind — no fetch list AND no terse
    'none' marker. This pins the requirement that an inert url_context tool never pollutes
    forecaster-facing research.

    The response carries a grounding chunk so it clears the grounded-chunk floor; the Sources block
    it produces is expected and orthogonal to the url_context marker layer under test.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    # url_metadata=None makes _make_response build url_context_metadata=None on the candidate.
    chunks = [CannedWebChunk(uri="https://vertexaisearch/redirect", title="Src", domain="src.example.com")]
    response = _make_response("body text", chunks=chunks, url_metadata=None)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out.startswith("body text")
    assert "### URL Context Fetches" not in out
    assert "_url_context: none_" not in out


@pytest.mark.asyncio
async def test_url_context_none_marker_when_all_fetches_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """When url_context fired but every retrieval FAILED, the forecaster-facing text collapses to the
    terse ``_url_context: none_`` marker — the failed URLs must NOT be listed as research context
    (only successful fetches were actually read). The failed URL is still in the INFO audit log.

    The response carries a grounding chunk so google_search grounding clears the floor; the failed
    url_context fetch is orthogonal.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    chunks = [CannedWebChunk(uri="https://vertexaisearch/redirect", title="Src", domain="src.example.com")]
    url_metadata = [
        CannedUrlMeta(
            retrieved_url="https://example.com/bad",
            url_retrieval_status=CannedStatus("URL_RETRIEVAL_STATUS_ERROR"),
        ),
    ]
    response = _make_response("body text", chunks=chunks, url_metadata=url_metadata)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "_url_context: none_" in out
    assert "### URL Context Fetches" not in out
    assert "https://example.com/bad" not in out
    assert out.startswith("body text")


# ---------------------------------------------------------------------------
# Parallel provider selection in main.py
# ---------------------------------------------------------------------------


class TestParallelProviderSelectionGemini:
    """Tests for Gemini gating via GEMINI_SEARCH_ENABLED in ``_select_research_providers``."""

    def test_select_research_providers_includes_gemini_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GEMINI_SEARCH_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        monkeypatch.delenv("NATIVE_SEARCH_ENABLED", raising=False)
        monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        mock_provider = AsyncMock(return_value="primary research")

        with patch.object(orch, "_select_research_provider", return_value=(mock_provider, "asknews")):
            providers = orch._select_research_providers()

        provider_names = [name for _, name in providers]
        assert "gemini_search" in provider_names

    def test_select_research_providers_excludes_gemini_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GEMINI_SEARCH_ENABLED", "false")
        monkeypatch.delenv("NATIVE_SEARCH_ENABLED", raising=False)
        monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        mock_provider = AsyncMock(return_value="primary research")

        with patch.object(orch, "_select_research_provider", return_value=(mock_provider, "asknews")):
            providers = orch._select_research_providers()

        provider_names = [name for _, name in providers]
        assert "gemini_search" not in provider_names


# ---------------------------------------------------------------------------
# Client HTTP configuration, thinking level, and the GEMINI_USAGE marker
# ---------------------------------------------------------------------------


class TestGeminiClientConfigAndUsage:
    """The client is configured, and every response's token spend is recorded.

    Two 2026-09 gaps, fixed together. The client was built bare — no retry options, which
    in ``google-genai`` means ``stop_after_attempt(1)``, so a transient ``503 UNAVAILABLE``
    lost the whole Google research leg — and nothing logged what the call cost on the
    operator's personal AI Studio key, which made Google-side spend reconstructable only
    from whole archived SDK responses.
    """

    @pytest.mark.asyncio
    async def test_client_carries_the_timeout_and_retry_ladder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        from metaculus_bot.constants import GEMINI_SEARCH_HTTP_ATTEMPTS, GEMINI_SEARCH_HTTP_TIMEOUT_MS

        fake_client = _make_client_with_response(_make_response("research text"))

        with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client) as client_constructor:
            from metaculus_bot.research.gemini_search import gemini_search_provider

            await gemini_search_provider()(_make_q("Will X happen?"))

        http_options = client_constructor.call_args.kwargs["http_options"]
        assert http_options.timeout == GEMINI_SEARCH_HTTP_TIMEOUT_MS
        retry_options = http_options.retry_options
        assert retry_options is not None, "a bare client retries nothing at all (stop_after_attempt(1))"
        assert retry_options.attempts == GEMINI_SEARCH_HTTP_ATTEMPTS
        assert 503 in (retry_options.http_status_codes or []), "503 UNAVAILABLE is the failure this recovers"

    def test_per_attempt_timeout_stays_under_the_outer_deadline(self) -> None:
        """The outer ``asyncio.wait_for(GEMINI_SEARCH_TIMEOUT)`` must stay the total bound.

        It is what cancels a hung call today, and adding retries must not change that. Since
        the client timeout is PER ATTEMPT, the invariant is that one attempt cannot outlast
        the outer deadline — a retry then only completes when the first attempt failed fast
        (a 503 returns in milliseconds), and a hang still ends at the outer deadline exactly
        as it does now. Sizing attempts x timeout under 360s instead would need <=176s per
        attempt, below the 150-200s AFC chains ``GEMINI_SEARCH_TIMEOUT``'s own comment
        records as legitimate.
        """
        from metaculus_bot.constants import GEMINI_SEARCH_HTTP_TIMEOUT_MS, GEMINI_SEARCH_TIMEOUT

        assert GEMINI_SEARCH_HTTP_TIMEOUT_MS / 1000 < GEMINI_SEARCH_TIMEOUT

    @pytest.mark.asyncio
    async def test_thinking_level_is_set_explicitly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Left unset, the model picks its own level — HIGH on gemini-3-flash-preview, most of the bill."""
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        from metaculus_bot.constants import GEMINI_SEARCH_THINKING_LEVEL

        fake_client = _make_client_with_response(_make_response("research text"))

        with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
            from metaculus_bot.research.gemini_search import gemini_search_provider

            await gemini_search_provider()(_make_q("Will X happen?"))

        config = fake_client.aio.models.generate_content.await_args.kwargs["config"]
        thinking_config = config.thinking_config
        assert thinking_config is not None
        # The SDK normalizes the string through its case-insensitive ThinkingLevel enum.
        assert thinking_config.thinking_level == genai_types.ThinkingLevel(GEMINI_SEARCH_THINKING_LEVEL.upper())
        # No output cap alongside it: capping tokens on a thinking model truncated silently.
        assert config.max_output_tokens is None

    @pytest.mark.asyncio
    async def test_usage_marker_logged_for_a_grounded_response(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        response = _make_response(
            "body text",
            chunks=[CannedWebChunk(uri="https://x", title="Example", domain="example.com")],
            web_search_queries=["who won"],
            usage_metadata=_make_usage(),
            model_version="gemini-3-flash-preview-002",
        )
        fake_client = _make_client_with_response(response)

        with (
            patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
            caplog.at_level(logging.INFO),
        ):
            from metaculus_bot.research.gemini_search import invoke_gemini_grounded

            out = await invoke_gemini_grounded("prompt", qid=44944)

        assert "### Sources" in out
        assert (
            "GEMINI_USAGE: role=grounded_search model=gemini-3-flash-preview-002 prompt_tokens=1200 "
            "tool_use_prompt_tokens=340 candidates_tokens=900 thoughts_tokens=2600 total_tokens=5040 "
            "search_queries=1 question=44944"
        ) in caplog.text

    @pytest.mark.asyncio
    async def test_usage_marker_logged_on_the_ungrounded_suppressed_branch(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A suppressed response was billed exactly like a useful one.

        The grounded-chunk floor throws the text away, so if the marker sat behind the
        formatting branches the wasted calls — the ones worth knowing the cost of — would be
        the only ones missing from the spend line.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        response = _make_response(
            "Ungrounded prose.",
            chunks=None,
            supports=None,
            web_search_queries=[f"query {i}" for i in range(30)],
            usage_metadata=_make_usage(thoughts=7000, total=9000),
        )
        fake_client = _make_client_with_response(response)

        with (
            patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
            caplog.at_level(logging.INFO),
        ):
            from metaculus_bot.research.gemini_search import invoke_gemini_grounded

            out = await invoke_gemini_grounded("prompt", qid=38195)

        assert out == "", "the ungrounded text is still suppressed"
        assert "GEMINI_UNGROUNDED_SUPPRESSED" in caplog.text
        assert "GEMINI_USAGE: role=grounded_search" in caplog.text
        assert "thoughts_tokens=7000" in caplog.text
        assert "total_tokens=9000" in caplog.text
        assert "search_queries=30" in caplog.text
        assert "question=38195" in caplog.text


# ---------------------------------------------------------------------------
# Grounding retry and the search-redirect loophole
# ---------------------------------------------------------------------------

_SEARCH_REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGsgBp9B9QB"


def _grounded_response(text: str = "Grounded body.") -> SimpleNamespace:
    return _make_response(text, chunks=[CannedWebChunk(uri=_SEARCH_REDIRECT, title="Src", domain="src.example")])


def _ungrounded_response(text: str = "Parametric body.") -> SimpleNamespace:
    """The gemini-3.8-flash shape: search ran (queries were billed) but no grounding metadata came back."""
    response = _make_response(text)
    response.candidates[0].grounding_metadata = None
    return response


def _client_with_responses(*responses: object) -> MagicMock:
    client = _make_client_with_response(responses[0])
    client.aio.models.generate_content = AsyncMock(side_effect=list(responses))
    return client


class TestGroundingRetry:
    """gemini-3.8-flash drops grounding metadata on about half of calls, at random per call, even when
    it searched (2026-09-22 probe: 10 of 22 calls, toolCall parts show 11-14 real queries on responses with no
    groundingMetadata). One retry recovers most of those; the second attempt shares the first's wall."""

    async def test_ungrounded_first_attempt_is_retried_and_the_grounded_retry_is_used(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        client = _client_with_responses(_ungrounded_response(), _grounded_response("Retry body."))

        with (
            patch("metaculus_bot.research.gemini_search.genai.Client", return_value=client),
            caplog.at_level(logging.INFO),
        ):
            from metaculus_bot.research.gemini_search import invoke_gemini_grounded

            out = await invoke_gemini_grounded("prompt", qid=45571)

        assert client.aio.models.generate_content.await_count == 2
        assert "Retry body." in out
        assert "Parametric body." not in out
        assert "### Sources" in out
        assert "GEMINI_GROUNDING_RETRY: question=45571 model=gemini-3.8-flash outcome=grounded" in caplog.text
        assert "GEMINI_UNGROUNDED_SUPPRESSED" not in caplog.text

    async def test_a_grounded_first_attempt_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        client = _client_with_responses(_grounded_response())

        with (
            patch("metaculus_bot.research.gemini_search.genai.Client", return_value=client),
            caplog.at_level(logging.INFO),
        ):
            from metaculus_bot.research.gemini_search import invoke_gemini_grounded

            out = await invoke_gemini_grounded("prompt", qid=1)

        assert client.aio.models.generate_content.await_count == 1
        assert "Grounded body." in out
        assert "GEMINI_GROUNDING_RETRY" not in caplog.text

    async def test_retries_once_then_suppresses(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        client = _client_with_responses(_ungrounded_response(), _ungrounded_response(), _grounded_response())

        with (
            patch("metaculus_bot.research.gemini_search.genai.Client", return_value=client),
            caplog.at_level(logging.INFO),
        ):
            from metaculus_bot.research.gemini_search import invoke_gemini_grounded

            out = await invoke_gemini_grounded("prompt", qid=2)

        assert client.aio.models.generate_content.await_count == 2
        assert out == ""
        assert "outcome=ungrounded" in caplog.text
        assert caplog.text.count("GEMINI_UNGROUNDED_SUPPRESSED") == 1
        # Both attempts were billed, so both carry a spend line.
        assert caplog.text.count("GEMINI_USAGE: role=grounded_search") == 2

    async def test_a_retry_that_outlives_the_shared_wall_suppresses_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The retry spends what is left of GEMINI_SEARCH_TIMEOUT, never a fresh one, so the provider's
        worst-case wall is unchanged; a retry that runs out of it leaves the first answer's suppression."""
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        monkeypatch.setattr("metaculus_bot.research.gemini_search.GEMINI_SEARCH_TIMEOUT", 0.2)

        attempts: list[int] = []

        async def ungrounded_then_slow(**_: object) -> SimpleNamespace:
            attempts.append(1)
            if len(attempts) == 1:
                return _ungrounded_response()
            await asyncio.sleep(5)
            return _grounded_response()

        client = _client_with_responses(_ungrounded_response())
        client.aio.models.generate_content = AsyncMock(side_effect=ungrounded_then_slow)

        with (
            patch("metaculus_bot.research.gemini_search.genai.Client", return_value=client),
            caplog.at_level(logging.INFO),
        ):
            from metaculus_bot.research.gemini_search import invoke_gemini_grounded

            out = await invoke_gemini_grounded("prompt", qid=3)

        assert len(attempts) == 2
        assert out == ""
        assert "outcome=timeout" in caplog.text
        assert "GEMINI_UNGROUNDED_SUPPRESSED" in caplog.text


class TestSearchRedirectIsNotGrounding:
    async def test_a_url_context_read_of_a_search_redirect_does_not_pass_the_floor(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Regression for the Q14333 smoke: no grounding metadata, one url_context read of a
        grounding-api-redirect URL, and 6 kB of self-tagged ``[A: official]`` text reached the panel.
        A redirect is a search hit whose metadata was dropped, the Q38195 shape, not a page the
        question pointed at, so it must not satisfy the url_context escape."""
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        redirect_read = _ungrounded_response("Self-tagged claim [A: official].")
        redirect_read.candidates[0].url_context_metadata = SimpleNamespace(
            url_metadata=[CannedUrlMeta(_SEARCH_REDIRECT, CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"))]
        )
        client = _client_with_responses(redirect_read, redirect_read)

        with (
            patch("metaculus_bot.research.gemini_search.genai.Client", return_value=client),
            caplog.at_level(logging.INFO),
        ):
            from metaculus_bot.research.gemini_search import invoke_gemini_grounded

            out = await invoke_gemini_grounded("prompt", qid=14333)

        assert out == ""
        assert "GEMINI_UNGROUNDED_SUPPRESSED" in caplog.text
