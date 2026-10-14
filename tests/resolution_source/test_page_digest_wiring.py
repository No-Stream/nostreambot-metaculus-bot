"""Production digest policies with real extraction, grounding and cache presentation."""

import html
import socket
import time

import pytest

from metaculus_bot.research import page_digest
from metaculus_bot.research.agentic import tools
from metaculus_bot.research.fetch_ladder import guard
from metaculus_bot.research.fetch_ladder.context import LadderContext
from metaculus_bot.research.fetch_ladder.ladder import fetch_url
from metaculus_bot.research.fetch_ladder.policy import (
    GAP_FILL_DOCUMENT_POLICY,
    GAP_FILL_FETCH_POLICY,
    RESOLUTION_SOURCE_POLICY,
    LadderPolicy,
)
from tests.resolution_source_fakes import FakeResponse, FakeSession
from tests.test_document_text import build_text_pdf
from tests.test_page_digest import PAGE, QUERY, TAIL_TABLE, ScriptedLlm

URL = "https://report.example.com/employment"


@pytest.fixture(autouse=True)
def _stub_page_wiring_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep these fake-fetch integration tests deterministic under mixed-path pytest runs."""

    def resolve_public_example_host(host: str, port: int | None, *args: object, **kwargs: object) -> list[tuple]:
        del host, port, args, kwargs
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 0))]

    monkeypatch.setattr(guard.socket, "getaddrinfo", resolve_public_example_host)


def _page_session() -> FakeSession:
    paragraphs = "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in PAGE.split("\n\n"))
    body = f"<html><body><article>{paragraphs}</article></body></html>".encode()
    return FakeSession({URL: FakeResponse(200, body=body, content_type="text/html")})


@pytest.mark.parametrize("policy", [RESOLUTION_SOURCE_POLICY, GAP_FILL_FETCH_POLICY, GAP_FILL_DOCUMENT_POLICY])
def test_production_callers_bind_the_real_digest(policy: LadderPolicy) -> None:
    assert policy.digest is page_digest.digest_page


async def test_fresh_cached_and_document_reads_ground_the_real_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    fabrication = "The unemployment rate was 97 percent."
    llm = ScriptedLlm(page_digest.PageDigestPassages(passages=[TAIL_TABLE, fabrication]).model_dump_json())
    monkeypatch.setattr(page_digest, "build_llm_with_openrouter_fallback", lambda **kwargs: llm)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    session = _page_session()

    fresh = await fetch_url(URL, policy=RESOLUTION_SOURCE_POLICY, ctx=LadderContext(query=QUERY, session=session))
    cached = await fetch_url(URL, policy=RESOLUTION_SOURCE_POLICY, ctx=LadderContext(query=QUERY, session=session))
    document = await tools.read_document(URL, QUERY, ctx=LadderContext(session=session))

    for result in (fresh, cached):
        assert result.status == "success"
        assert "Civilian unemployment rate | 4.2 | 4.3" in result.text
        assert fabrication not in result.text
        assert result.passages_returned == 2
        assert result.passages_grounded == 1
        assert result.fallback_used is False
        assert RESOLUTION_SOURCE_POLICY.per_url_max_chars is not None
        assert len(result.text) <= RESOLUTION_SOURCE_POLICY.per_url_max_chars
    assert cached.cache_hit
    assert document.method == "digest_local"
    assert f"Document: {URL}" in document.content_markdown
    assert "Civilian unemployment rate | 4.2 | 4.3" in document.content_markdown
    assert fabrication not in document.content_markdown
    assert session.requested == [URL]
    assert len(llm.prompts) == 1
    assert all(" ".join(TAIL_TABLE.split()) in " ".join(prompt.split()) for prompt in llm.prompts)


async def test_cached_page_uses_real_bm25_fallback_below_call_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = ScriptedLlm(AssertionError("the remaining wall cannot fund an extractor call"))
    monkeypatch.setattr(page_digest, "build_llm_with_openrouter_fallback", lambda **kwargs: llm)
    session = _page_session()
    await fetch_url(URL, policy=GAP_FILL_FETCH_POLICY, ctx=LadderContext(session=session))
    ctx = LadderContext(
        query=QUERY,
        session=session,
        started=time.monotonic() - RESOLUTION_SOURCE_POLICY.total_wall_s + 4.0,
    )

    result = await fetch_url(URL, policy=RESOLUTION_SOURCE_POLICY, ctx=ctx)

    assert result.cache_hit
    assert result.status == "success"
    assert result.fallback_used is True
    assert result.passages_returned == result.passages_grounded == 0
    assert "unemployment rate | 4.2 | 4.3" in result.text
    assert session.requested == [URL]
    assert llm.prompts == []


async def test_pdf_keeps_page_labels_without_the_flat_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = ScriptedLlm(AssertionError("PDFs must retain their page-aware digest"))
    monkeypatch.setattr(page_digest, "build_llm_with_openrouter_fallback", lambda **kwargs: llm)
    url = "https://report.example.com/report.pdf"
    session = FakeSession(
        {
            url: FakeResponse(
                200,
                body=build_text_pdf([["The unemployment rate was 4.3 percent."] * 30]),
                content_type="application/pdf",
            )
        }
    )

    result = await fetch_url(
        url, policy=RESOLUTION_SOURCE_POLICY, ctx=LadderContext(query="unemployment", session=session)
    )

    assert result.status == "success"
    assert "[p.1]" in result.text
    assert result.passages_returned is None
    assert llm.prompts == []
