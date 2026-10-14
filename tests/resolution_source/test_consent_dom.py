"""Consent widgets are removed before extraction without dropping article content."""

from __future__ import annotations

from metaculus_bot.research.fetch_ladder.classify import _extract_page_text


def test_consent_widget_is_removed_while_cookie_research_article_is_preserved() -> None:
    article = " ".join(
        f"This cookie study reports that treatment group {index} used the service for {index + 3} days."
        for index in range(20)
    )
    consent = " ".join("We use cookies to personalize advertisements. Accept all cookies." for _ in range(80))
    html = (
        "<html><body>"
        f'<div id="cookie-consent-banner" role="dialog"><p>{consent}</p>'
        "<button>Accept all</button><button>Reject optional</button></div>"
        f"<article><h1>Cookie use in the study</h1><p>{article}</p></article>"
        "</body></html>"
    )

    extraction = _extract_page_text(html, html.encode(), "https://example.test/study", 0.0)

    assert extraction.text is not None
    assert "This cookie study reports" in extraction.text
    assert "We use cookies to personalize advertisements" not in extraction.text
    assert "Accept all cookies" not in extraction.text


def test_consent_only_page_has_no_extracted_text() -> None:
    html = (
        "<html><body>"
        '<div id="onetrust-consent-sdk" role="dialog">'
        "We use cookies to personalize content. Accept all or reject optional cookies."
        "<button>Accept all</button></div>"
        "</body></html>"
    )

    extraction = _extract_page_text(html, html.encode(), "https://example.test/consent", 0.0)

    assert extraction.text is None


def test_xhtml_with_xml_declaration_falls_back_to_original_body() -> None:
    article = " ".join(f"The agency reported {index + 10} filings in district {index}." for index in range(20))
    xhtml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        f"<article><h1>Filing report</h1><p>{article}</p></article>"
        "</body></html>"
    )

    extraction = _extract_page_text(xhtml, xhtml.encode(), "https://example.test/report.xhtml", 0.0)

    assert extraction.text is not None
    assert "The agency reported 10 filings" in extraction.text


def test_comment_only_body_does_not_raise() -> None:
    body = "<!-- nothing here -->"

    extraction = _extract_page_text(body, body.encode(), "https://example.test/empty", 0.0)

    assert extraction.text is None
