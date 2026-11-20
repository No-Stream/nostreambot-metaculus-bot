"""Model-written markdown is escaped for Streamlit without touching code or losing structure."""

from __future__ import annotations

from metaculus_bot.trace_review.views import readable_markdown


def test_dollars_and_tildes_are_escaped_outside_code_and_headings_are_demoted():
    rationale = (
        "# PHASE 0\n"
        "## Status quo\n"
        "Revenue was $2,366M and the target ~$350/share, already \\$5 escaped.\n"
        "Inline `$x ~y` stays.\n"
        "```\nprice = $3 ~ish\n```\n"
    )

    rendered = readable_markdown(rationale)

    assert rendered.splitlines()[:2] == ["#### PHASE 0", "##### Status quo"]
    assert "\\$2,366M" in rendered
    assert "\\~\\$350/share" in rendered
    assert "\\$5 escaped" in rendered
    assert "\\\\$5" not in rendered
    assert "`$x ~y`" in rendered
    assert "price = $3 ~ish" in rendered


def test_text_without_headings_or_specials_is_unchanged():
    text = "Plain rationale with **bold**, a list:\n- one\n- two"
    assert readable_markdown(text) == text
