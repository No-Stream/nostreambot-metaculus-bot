from metaculus_bot.research.agentic import tools
from metaculus_bot.research.agentic.tool_backends import _asknews_evidence_passages


def test_long_source_paragraph_produces_bounded_verbatim_sentences() -> None:
    sentence = "The agency observed a change in the regional forecast. "
    text = sentence * 50

    outcome = tools._render_fetch_outcome("https://example.org/report", text, [], method="plain", start_char=0)

    assert len(outcome.evidence_passages) > 1
    assert all(passage.text in text for passage in outcome.evidence_passages)
    assert sum(len(passage.text) for passage in outcome.evidence_passages) >= len(text.rstrip())


def test_long_news_summary_keeps_provider_origin_and_url() -> None:
    url = "https://example.org/article"
    summary = "The regional forecast was published. " * 60

    passages = _asknews_evidence_passages([{"article_url": url, "summary": summary}])

    assert len(passages) > 1
    assert all(passage.text in summary and passage.source_url == url for passage in passages)
    assert all(passage.content_origin == "provider_summary" for passage in passages)
