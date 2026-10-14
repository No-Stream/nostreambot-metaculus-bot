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
