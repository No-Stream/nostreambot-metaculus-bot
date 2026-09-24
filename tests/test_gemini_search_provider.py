"""Offline tests for Gemini's self-cited search-link research provider."""

import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types as genai_types

from metaculus_bot.research import gemini_search
from metaculus_bot.research.gemini_search import _strip_model_citation_indices
from metaculus_bot.research.provider_diagnostics import _is_lost_source, pop_provider_detail

_SEARCH_REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGsgBp9B9QB"
_SEARCH_REDIRECT_TWO = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHUQAyVQ-abnnoImmTBrN9GLRlN3oZc1jKzDnDIwJH0A_vTj5tw2w-vzZTLbSsQlm-Fp3CjnxUN1BpO2lhQTyToWWym2m9VYVY7SfkkaUNHq_OYpP5PIhXvg65kuU0da4E2EQ=="
_TARGET_ONE = "https://www.example.com/report"
_TARGET_TWO = "https://news.example.org/briefing"
_FIXTURE = Path(__file__).parent / "fixtures" / "gemini_selfcite_responses.txt"


def _make_q(text: str) -> MagicMock:
    q = MagicMock()
    q.question_text = text
    q.options = None
    return q


class CannedStatus:
    """A url_retrieval_status enum stand-in exposing ``.name`` like the SDK enum."""

    def __init__(self, name: str) -> None:
        self.name = name


class CannedUrlMeta:
    """Mirror of google-genai's UrlMetadata fields used by telemetry."""

    def __init__(self, retrieved_url: str | None, url_retrieval_status: object) -> None:
        self.retrieved_url = retrieved_url
        self.url_retrieval_status = url_retrieval_status


def _make_response(
    text: str,
    *,
    url_metadata: list[object] | None = None,
    web_search_queries: list[str] | None = None,
    usage_metadata: object | None = None,
    model_version: str | None = None,
    grounding_chunks: list[object] | None = None,
) -> SimpleNamespace:
    metadata = SimpleNamespace(
        grounding_chunks=grounding_chunks,
        grounding_supports=None,
        web_search_queries=web_search_queries,
    )
    url_context_metadata = SimpleNamespace(url_metadata=url_metadata) if url_metadata is not None else None
    candidate = SimpleNamespace(
        grounding_metadata=metadata,
        url_context_metadata=url_context_metadata,
    )
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
    return SimpleNamespace(
        prompt_token_count=prompt,
        tool_use_prompt_token_count=tool_use,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        total_token_count=total,
    )


def _make_client_with_response(response: object) -> MagicMock:
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    return client


def _fixture_first_response() -> str:
    return _FIXTURE.read_text().split("\n\n### A second real response shape", maxsplit=1)[0]


def _fixture_second_response() -> str:
    return _FIXTURE.read_text().split("\n\n### A second real response shape\n", maxsplit=1)[1]


# ---------------------------------------------------------------------------
# Client construction and prompt wiring
# ---------------------------------------------------------------------------


def test_builder_raises_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        gemini_search.build_gemini_client()


@pytest.mark.asyncio
async def test_provider_uses_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.delenv("GEMINI_SEARCH_MODEL", raising=False)
    fake_client = _make_client_with_response(_make_response("some research text"))

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        await gemini_search.gemini_search_provider()(_make_q("Will X happen?"))

    assert fake_client.aio.models.generate_content.await_count == 1
    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash"
    assert "Will X happen?" in call_kwargs["contents"]


@pytest.mark.asyncio
async def test_provider_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_SEARCH_MODEL", "gemini-2.5-flash")
    fake_client = _make_client_with_response(_make_response("research text"))

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        await gemini_search.gemini_search_provider()(_make_q("Will X happen?"))

    assert fake_client.aio.models.generate_content.await_args.kwargs["model"] == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_provider_uses_explicit_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_SEARCH_MODEL", "gemini-2.5-flash")
    fake_client = _make_client_with_response(_make_response("research text"))

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        await gemini_search.gemini_search_provider(model_slug="gemini-explicit-override")(_make_q("Will X happen?"))

    assert fake_client.aio.models.generate_content.await_args.kwargs["model"] == "gemini-explicit-override"


@pytest.mark.asyncio
async def test_provider_attaches_google_search_and_url_context_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    fake_client = _make_client_with_response(_make_response("research text"))

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        await gemini_search.gemini_search_provider()(_make_q("Will X happen?"))

    tools = list(fake_client.aio.models.generate_content.await_args.kwargs["config"].tools)
    assert len(tools) == 2
    assert any(getattr(tool, "google_search", None) is not None for tool in tools)
    assert any(getattr(tool, "url_context", None) is not None for tool in tools)


@pytest.mark.asyncio
async def test_benchmarking_carve_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    fake_client = _make_client_with_response(_make_response("research text"))

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        await gemini_search.gemini_search_provider(is_benchmarking=True)(_make_q("Will X happen?"))

    prompt = fake_client.aio.models.generate_content.await_args.kwargs["contents"]
    assert "benchmarking run" in prompt
    assert "Market-implied or crowd odds" not in prompt


@pytest.mark.asyncio
async def test_non_benchmarking_includes_prediction_markets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    fake_client = _make_client_with_response(_make_response("research text"))

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        await gemini_search.gemini_search_provider(is_benchmarking=False)(_make_q("Will X happen?"))

    prompt = fake_client.aio.models.generate_content.await_args.kwargs["contents"]
    assert "Market-implied or crowd odds" in prompt
    assert "benchmarking run" not in prompt


@pytest.mark.asyncio
async def test_prompt_carries_the_mc_ballot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    fake_client = _make_client_with_response(_make_response("research text"))
    question = _make_q("Who will win the World Yo-Yo Contest?")
    question.options = ["Mir Kim", "Hunter Feuerstein", "Other"]

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        await gemini_search.gemini_search_provider(is_benchmarking=False)(question)

    prompt = fake_client.aio.models.generate_content.await_args.kwargs["contents"]
    assert "Options (in resolution order): Mir Kim | Hunter Feuerstein | Other" in prompt


# ---------------------------------------------------------------------------
# Self-cited links and the citation floor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_selfcite_fixture_links_are_numbered_and_redirects_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    text = _fixture_first_response()
    redirect_urls = list(
        dict.fromkeys(re.findall(r"https://vertexaisearch\.cloud\.google\.com/grounding-api-redirect/[^)]+", text))
    )
    targets = [_TARGET_ONE, _TARGET_TWO, "https://research.example.net/study"]
    resolution = dict(zip(redirect_urls, targets, strict=True))
    fake_client = _make_client_with_response(_make_response(text, web_search_queries=["longevity"]))

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch("metaculus_bot.research.gemini_search.resolve_search_redirects", new=AsyncMock(return_value=resolution)),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=6001)

    assert "vertexaisearch.cloud.google.com" not in out
    assert "grounding-api-redirect" not in out
    # The fixture's tier tags are class descriptions, which name no outlet, so they are
    # rewritten; the link numbers beside them survive.
    assert "[unverified attribution] [1]" in out
    assert "[unverified attribution] [3]" in out
    assert "[A: official]" not in out
    assert "### Sources" in out
    assert "[1] example.com — https://www.example.com/report" in out
    assert "[2] news.example.org — https://news.example.org/briefing" in out
    assert "[3] research.example.net — https://research.example.net/study" in out


@pytest.mark.asyncio
async def test_real_selfcite_fixture_preserves_emphasized_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    text = _fixture_second_response()
    redirect_url = next(
        iter(re.findall(r"https://vertexaisearch\.cloud\.google\.com/grounding-api-redirect/[^)]+", text))
    )
    fake_client = _make_client_with_response(_make_response(text))

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch(
            "metaculus_bot.research.gemini_search.resolve_search_redirects",
            new=AsyncMock(return_value={redirect_url: "https://demographic.example.edu/study"}),
        ),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=6014)

    assert "*Demographic Research* [1]" in out
    assert "vertexaisearch.cloud.google.com" not in out


@pytest.mark.asyncio
async def test_duplicate_resolved_targets_share_a_source_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    text = f"First [one]({_SEARCH_REDIRECT}) and second [two]({_SEARCH_REDIRECT_TWO})."
    fake_client = _make_client_with_response(_make_response(text))
    resolution = {_SEARCH_REDIRECT: _TARGET_ONE, _SEARCH_REDIRECT_TWO: _TARGET_ONE}

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch("metaculus_bot.research.gemini_search.resolve_search_redirects", new=AsyncMock(return_value=resolution)),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=6002)

    assert "First one [1] and second two [1]." in out
    assert out.count("[1] example.com — https://www.example.com/report") == 1
    assert "[2]" not in out


@pytest.mark.asyncio
async def test_unresolved_link_is_marked_without_leaking_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    text = f"Known [known]({_SEARCH_REDIRECT}) and missing [missing]({_SEARCH_REDIRECT_TWO})."
    fake_client = _make_client_with_response(_make_response(text))

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch(
            "metaculus_bot.research.gemini_search.resolve_search_redirects",
            new=AsyncMock(return_value={_SEARCH_REDIRECT: _TARGET_ONE}),
        ),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=6003)

    assert "Known known [1] and missing missing [unverified link]." in out
    assert "vertexaisearch" not in out
    assert "grounding-api-redirect" not in out
    assert "[1] example.com — https://www.example.com/report" in out


@pytest.mark.asyncio
async def test_zero_verified_links_suppresses_and_records_loss(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    text = f"Confident claim [source]({_SEARCH_REDIRECT})."
    response = _make_response(text, web_search_queries=[f"query {i}" for i in range(30)])
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch("metaculus_bot.research.gemini_search.resolve_search_redirects", new=AsyncMock(return_value={})),
        caplog.at_level(logging.INFO),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=6004)

    assert out == ""
    assert "GEMINI_SELF_CITATION: question=6004" in caplog.text
    assert "links=1 unique=1 resolved=0 unverified=1 sources=0" in caplog.text
    assert "GEMINI_UNGROUNDED_SUPPRESSED: question=6004 model=gemini-3.8-flash queries=30" in caplog.text
    detail = pop_provider_detail(6004, "gemini_search")
    assert _is_lost_source(detail["sources"]["grounding"])


@pytest.mark.asyncio
async def test_successful_url_context_link_is_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    read_url = "https://gov.example/report"
    response = _make_response(
        f"The report says [report]({read_url}).",
        url_metadata=[CannedUrlMeta(read_url, CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS"))],
    )
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch("metaculus_bot.research.gemini_search.resolve_search_redirects", new=AsyncMock(return_value={})),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=6005)

    assert "The report says report [1]." in out
    assert "[1] gov.example — https://gov.example/report" in out
    assert "### URL Context Fetches" in out


@pytest.mark.asyncio
async def test_nonredirect_link_not_read_is_unverified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    unread = "https://unread.example/article"
    text = f"Search result [search]({_SEARCH_REDIRECT}); unrelated [page]({unread})."
    fake_client = _make_client_with_response(_make_response(text))

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch(
            "metaculus_bot.research.gemini_search.resolve_search_redirects",
            new=AsyncMock(return_value={_SEARCH_REDIRECT: _TARGET_ONE}),
        ),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=6006)

    assert "search [1]" in out
    assert "page [unverified link]" in out
    assert unread not in out


@pytest.mark.asyncio
async def test_attribution_check_uses_resolved_domains(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    text = f"The claim [[A: NASA]]({_SEARCH_REDIRECT}) is disputed."
    fake_client = _make_client_with_response(_make_response(text))

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch(
            "metaculus_bot.research.gemini_search.resolve_search_redirects",
            new=AsyncMock(return_value={_SEARCH_REDIRECT: "https://timeanddate.com/eclipse"}),
        ),
        caplog.at_level(logging.INFO),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=6007)

    assert "[unverified attribution]" in out
    assert "NASA" not in out
    assert "[1] timeanddate.com — https://timeanddate.com/eclipse" in out
    assert "GEMINI_UNSUPPORTED_ATTRIBUTION: question=6007" in caplog.text


@pytest.mark.asyncio
async def test_generic_tier_tags_are_rewritten_and_counted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A class tag names no outlet, so it is rewritten like an unmatched name and counted
    apart in the provider details; the zero counts are recorded too, as measurements."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    text = f"The oldest age is 122 [[A: peer-reviewed journal]]({_SEARCH_REDIRECT}) and [B: NASA] agrees."
    fake_client = _make_client_with_response(_make_response(text))

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch(
            "metaculus_bot.research.gemini_search.resolve_search_redirects",
            new=AsyncMock(return_value={_SEARCH_REDIRECT: "https://demographic-research.org/paper"}),
        ),
        caplog.at_level(logging.INFO),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=6010)

    assert "peer-reviewed journal" not in out
    assert out.count("[unverified attribution]") == 2
    assert pop_provider_detail(6010, "gemini_search")["counts"] == {
        "tier_tags": 1,
        "generic_tier_tags": 1,
        "unsupported_attributions": 1,
    }
    assert "GEMINI_UNSUPPORTED_ATTRIBUTION: question=6010 tagged=1 unsupported=1 groups=2 labels=1 generic=1" in (
        caplog.text
    )


@pytest.mark.asyncio
async def test_self_citation_marker_reports_link_counts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    text = f"One [a]({_SEARCH_REDIRECT}) and [again]({_SEARCH_REDIRECT}) plus [b]({_SEARCH_REDIRECT_TWO})."
    fake_client = _make_client_with_response(_make_response(text))
    resolution = {_SEARCH_REDIRECT: _TARGET_ONE, _SEARCH_REDIRECT_TWO: _TARGET_TWO}

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch("metaculus_bot.research.gemini_search.resolve_search_redirects", new=AsyncMock(return_value=resolution)),
        caplog.at_level(logging.INFO),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=6008)

    assert out
    assert "GEMINI_SELF_CITATION: question=6008" in caplog.text
    assert "links=3 unique=2 resolved=2 unverified=0 sources=2" in caplog.text


@pytest.mark.asyncio
async def test_resolution_is_skipped_when_the_shared_wall_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(gemini_search, "GEMINI_SEARCH_TIMEOUT", 0.0)
    response = _make_response(f"Claim [source]({_SEARCH_REDIRECT}).")
    resolver = AsyncMock(return_value={_SEARCH_REDIRECT: _TARGET_ONE})

    with (
        patch.object(gemini_search, "build_gemini_client", return_value=MagicMock()),
        patch.object(gemini_search, "_generate_grounded", new=AsyncMock(return_value=response)),
        patch.object(gemini_search, "resolve_search_redirects", new=resolver),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=6009)

    assert out == ""
    resolver.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_sdk_call_and_no_grounding_retry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    fake_client = _make_client_with_response(_make_response("Parametric text without a link."))

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch("metaculus_bot.research.gemini_search.resolve_search_redirects", new=AsyncMock(return_value={})),
        caplog.at_level(logging.INFO),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=6010)

    assert out == ""
    assert fake_client.aio.models.generate_content.await_count == 1
    assert "grounding retry" not in caplog.text.lower()


@pytest.mark.asyncio
async def test_q38195_confident_fake_tags_without_links_are_suppressed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    fabricated = "Generative AI drove labor tension. Key contract expirations: Boeing IAM 837 [A: official]."
    response = _make_response(fabricated, web_search_queries=[f"query {i}" for i in range(30)])
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch("metaculus_bot.research.gemini_search.resolve_search_redirects", new=AsyncMock(return_value={})),
        caplog.at_level(logging.INFO),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=38195)

    assert out == ""
    assert "Boeing" not in out
    assert "GEMINI_UNGROUNDED_SUPPRESSED" in caplog.text


# ---------------------------------------------------------------------------
# Existing output and diagnostics contracts that remain alongside self-citation
# ---------------------------------------------------------------------------


class TestStripModelCitationIndices:
    def test_removes_a_lone_index_and_the_space_it_leaves(self) -> None:
        assert _strip_model_citation_indices("tag more sharks [2.4.1]. The count") == "tag more sharks. The count"

    def test_removes_a_multi_token_group(self) -> None:
        assert _strip_model_citation_indices("office [1.1.1, 1.1.2]. He") == "office. He"

    def test_keeps_a_tier_tag_and_drops_the_index_beside_it(self) -> None:
        assert _strip_model_citation_indices("path of totality [A: NASA, 1.1.2].") == "path of totality [A: NASA]."

    def test_keeps_a_tier_tag_whose_index_is_space_separated(self) -> None:
        assert _strip_model_citation_indices("[A: official 2.4.1, 3.2.5].") == "[A: official]."
        assert _strip_model_citation_indices("[B: Access Newswire 1.1.4, 3.3.4].") == "[B: Access Newswire]."

    def test_keeps_a_trailing_tier_grade_when_the_index_comes_first(self) -> None:
        assert _strip_model_citation_indices("HPI rose [1.1.8, 2.1.4: A].") == "HPI rose [A]."

    def test_preserves_semicolon_separated_tier_groups(self) -> None:
        text = _strip_model_citation_indices("[B: Forbes, 1.6.4; C: Newsweek, 3.2.2].")
        assert text == "[B: Forbes; C: Newsweek]."

    def test_preserves_our_numbered_markers(self) -> None:
        text = "Alpha.[1] Beta.[12] Gamma.[1, 3] Delta.[11, 12]"
        assert _strip_model_citation_indices(text) == text

    def test_preserves_bracketed_quantities_and_versions(self) -> None:
        for text in (
            "gasoline [3.8%] higher",
            "priced at [$1.5] a share",
            "reached [1.5 million] viewers",
            "shipped in [v2.1.3] of the tool",
            "dated [2026.08] in the filing",
            "resolved [192.168.1.1] internally",
            "the [1.5-2.0] range",
        ):
            assert _strip_model_citation_indices(text) == text

    def test_is_idempotent(self) -> None:
        text = "office [1.1.1, 1.1.2]. NASA [A: NASA, 1.1.2] said.[1]"
        once = _strip_model_citation_indices(text)
        assert _strip_model_citation_indices(once) == once

    def test_empties_a_table_cell_without_eating_the_pipes(self) -> None:
        assert _strip_model_citation_indices("| **2024** | 2.2% | [1.4, 1.23] |") == "| **2024** | 2.2% | |"


@pytest.mark.asyncio
async def test_url_context_telemetry_marker_remains_in_returned_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    read_url = "https://example.com/ok"
    response = _make_response(
        f"Body [source]({read_url}).",
        url_metadata=[
            CannedUrlMeta(read_url, CannedStatus("URL_RETRIEVAL_STATUS_SUCCESS")),
            CannedUrlMeta("https://example.com/bad", "URL_RETRIEVAL_STATUS_ERROR"),
        ],
    )
    fake_client = _make_client_with_response(response)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch("metaculus_bot.research.gemini_search.resolve_search_redirects", new=AsyncMock(return_value={})),
    ):
        out = await gemini_search.invoke_gemini_grounded("prompt", qid=6011)

    assert "### URL Context Fetches" in out
    assert "URL_RETRIEVAL_STATUS_SUCCESS — https://example.com/ok" in out
    assert "URL_RETRIEVAL_STATUS_ERROR" not in out
    assert "https://example.com/bad" not in out


class TestParallelProviderSelectionGemini:
    def test_select_research_providers_includes_gemini_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

        assert "gemini_search" in [name for _, name in providers]

    def test_select_research_providers_excludes_gemini_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

        assert "gemini_search" not in [name for _, name in providers]


class TestGeminiClientConfigAndUsage:
    @pytest.mark.asyncio
    async def test_client_carries_the_timeout_and_retry_ladder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        from metaculus_bot.constants import GEMINI_SEARCH_HTTP_ATTEMPTS, GEMINI_SEARCH_HTTP_TIMEOUT_MS

        fake_client = _make_client_with_response(_make_response("research text"))
        with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client) as constructor:
            await gemini_search.gemini_search_provider()(_make_q("Will X happen?"))

        http_options = constructor.call_args.kwargs["http_options"]
        assert http_options.timeout == GEMINI_SEARCH_HTTP_TIMEOUT_MS
        assert http_options.retry_options is not None
        assert http_options.retry_options.attempts == GEMINI_SEARCH_HTTP_ATTEMPTS
        assert 503 in (http_options.retry_options.http_status_codes or [])

    def test_per_attempt_timeout_stays_under_outer_deadline(self) -> None:
        from metaculus_bot.constants import GEMINI_SEARCH_HTTP_TIMEOUT_MS, GEMINI_SEARCH_TIMEOUT

        assert GEMINI_SEARCH_HTTP_TIMEOUT_MS / 1000 < GEMINI_SEARCH_TIMEOUT

    @pytest.mark.asyncio
    async def test_thinking_level_is_set_explicitly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        from metaculus_bot.constants import GEMINI_SEARCH_THINKING_LEVEL

        fake_client = _make_client_with_response(_make_response("research text"))
        with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
            await gemini_search.gemini_search_provider()(_make_q("Will X happen?"))

        config = fake_client.aio.models.generate_content.await_args.kwargs["config"]
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_level == genai_types.ThinkingLevel(GEMINI_SEARCH_THINKING_LEVEL.upper())
        assert config.max_output_tokens is None

    @pytest.mark.asyncio
    async def test_usage_marker_logged_for_a_cited_response(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        response = _make_response(
            f"Body [source]({_SEARCH_REDIRECT}).",
            web_search_queries=["who won"],
            usage_metadata=_make_usage(),
            model_version="gemini-3-flash-preview-002",
        )
        fake_client = _make_client_with_response(response)
        with (
            patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
            patch(
                "metaculus_bot.research.gemini_search.resolve_search_redirects",
                new=AsyncMock(return_value={_SEARCH_REDIRECT: _TARGET_ONE}),
            ),
            caplog.at_level(logging.INFO),
        ):
            out = await gemini_search.invoke_gemini_grounded("prompt", qid=6012)

        assert "### Sources" in out
        assert (
            "GEMINI_USAGE: role=grounded_search model=gemini-3-flash-preview-002 prompt_tokens=1200 "
            "tool_use_prompt_tokens=340 candidates_tokens=900 thoughts_tokens=2600 total_tokens=5040 "
            "search_queries=1 question=6012"
        ) in caplog.text

    @pytest.mark.asyncio
    async def test_usage_marker_logged_on_suppressed_branch(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        response = _make_response(
            "Ungrounded prose.",
            web_search_queries=[f"query {i}" for i in range(30)],
            usage_metadata=_make_usage(thoughts=7000, total=9000),
        )
        fake_client = _make_client_with_response(response)
        with (
            patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
            patch("metaculus_bot.research.gemini_search.resolve_search_redirects", new=AsyncMock(return_value={})),
            caplog.at_level(logging.INFO),
        ):
            out = await gemini_search.invoke_gemini_grounded("prompt", qid=6013)

        assert out == ""
        assert "GEMINI_UNGROUNDED_SUPPRESSED" in caplog.text
        assert "GEMINI_USAGE: role=grounded_search" in caplog.text
        assert "thoughts_tokens=7000" in caplog.text
        assert "total_tokens=9000" in caplog.text
