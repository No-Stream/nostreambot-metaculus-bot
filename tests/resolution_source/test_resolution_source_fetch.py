"""Network layer of the Tier-1 resolution-source fetcher.

``_fetch_one``'s branches, what the fetcher does with a page whose numbers live inside a
widget (the embed-shaped 200 and the inline-chart rung), the per-fetch telemetry marker,
and ``fetch_resolution_sources``' per-host serialization. Split out of the old single
``test_resolution_source_provider.py``; see ``test_resolution_source_helpers.py`` for the
layer map.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import UTC, datetime
from typing import ClassVar

import aiohttp
import pytest

from metaculus_bot.research import impersonated_fetch, resolution_chart_data, resolution_presentation, resolution_source
from metaculus_bot.research.fetch_ladder import classify, direct_fetch, guard, rungs, verdict
from metaculus_bot.research.fetch_ladder.context import LadderContext
from metaculus_bot.research.fetch_ladder.ladder import _fetch_one
from metaculus_bot.research.fetch_ladder.policy import RESOLUTION_SOURCE_POLICY
from metaculus_bot.research.fetch_ladder.verdict import looks_like_js_wall, looks_like_page_chrome
from metaculus_bot.research.http_fetch import host_semaphores, pdf_parse_semaphore, semaphore_for_host
from metaculus_bot.research.impersonated_fetch import IMPERSONATE_TRIGGER_STATUSES
from metaculus_bot.research.provider_diagnostics import pop_provider_detail
from metaculus_bot.research.resolution_chart_data import CHART_DATA_LEAD, render_inline_chart_data
from metaculus_bot.research.resolution_presentation import (
    _unreadable_embed_disclosure,
    format_resolution_sections,
)
from metaculus_bot.research.resolution_source import (
    FetchResult,
    _fetch_result_sources,
    _rung_counts,
    fetch_resolution_sources,
    resolution_source_provider,
)
from scripts.telemetry.markers import MARKER_SPECS
from tests.resolution_source_fakes import (
    _URL,
    FakeResponse,
    FakeSession,
    _embed_shell_page,
    _escape_config,
    _fake_render,
    _impersonated,
    _iom_shaped_page,
    _meta_refresh_stub,
    _mid_band_chart_page,
    _mock_question,
    _prose_page,
    capped_ctx,
    cdc_aria_stat_block_page,
    fake_impersonated_fetch,
)
from tests.test_document_text import build_text_pdf


class TestFetchOne:
    async def test_success_html_extracts_and_truncates(self, article_html, monkeypatch):
        """The per-URL cap is tightened here so that truncation lands too and can be verified."""
        session = FakeSession(
            {"https://news.example.com/report": FakeResponse(200, body=article_html, content_type="text/html")}
        )
        result = await _fetch_one(session, "https://news.example.com/report", {}, capped_ctx(200))
        assert result.status == "success"
        assert result.http_status == 200
        # Real trafilatura ran on the article — a known substring survives.
        assert "Bureau of Labor Statistics" in result.text
        # Per-URL truncation was applied.
        assert len(result.text) <= 200

    async def test_html_truncation_appends_marker(self, article_html, monkeypatch):
        """Live run analysis (2026-07-10) found the per-URL cap truncating mid-sentence with no marker, so
        forecasters could not tell the snapshot was partial. When truncation fires, a marker line naming the
        cap and the URL must appear, and the total text length must remain bounded by the cap."""
        cap = 200
        session = FakeSession(
            {"https://news.example.com/report": FakeResponse(200, body=article_html, content_type="text/html")}
        )
        result = await _fetch_one(session, "https://news.example.com/report", {}, capped_ctx(cap))
        assert result.status == "success"
        assert f"[truncated at {cap} chars — full source at https://news.example.com/report]" in result.text
        assert len(result.text) <= cap

    async def test_no_truncation_marker_when_fits_under_cap(self, article_html, monkeypatch):
        """An extraction that fits entirely under the cap gets NO marker appended."""
        session = FakeSession(
            {"https://news.example.com/report": FakeResponse(200, body=article_html, content_type="text/html")}
        )
        result = await _fetch_one(session, "https://news.example.com/report", {}, capped_ctx(100_000))
        assert result.status == "success"
        assert "truncated at" not in result.text

    async def test_403_maps_to_blocked(self):
        session = FakeSession({"https://blocked.example.com/x": FakeResponse(403, body=b"nope")})
        result = await _fetch_one(session, "https://blocked.example.com/x", {})
        assert result.status == "blocked"
        assert result.http_status == 403
        assert result.text == ""

    async def test_404_maps_to_not_found(self):
        session = FakeSession({"https://gone.example.com/x": FakeResponse(404)})
        result = await _fetch_one(session, "https://gone.example.com/x", {})
        assert result.status == "not_found"
        assert result.http_status == 404

    async def test_js_wall_short_html_flagged(self):
        """A 200 OK whose extracted text is short is a js_wall."""
        tiny = b"<!doctype html><html><body><div id='root'></div></body></html>"
        session = FakeSession({"https://spa.example.com/x": FakeResponse(200, body=tiny, content_type="text/html")})
        result = await _fetch_one(session, "https://spa.example.com/x", {})
        assert result.status == "js_wall"
        assert result.http_status == 200
        assert result.text == ""

    async def test_oversize_body_is_dropped(self, monkeypatch):
        """A 100-byte cap is forced here so that the body exceeds it."""
        monkeypatch.setattr(classify, "RESOLUTION_SOURCE_MAX_RESPONSE_BYTES", 100)
        oversized = b"<html><body>" + b"A" * 500 + b"</body></html>"
        session = FakeSession(
            {"https://big.example.com/x": FakeResponse(200, body=oversized, content_type="text/html")}
        )
        result = await _fetch_one(session, "https://big.example.com/x", {})
        # read_body_capped returns None -> we mark as error (no readable body).
        assert result.status == "error"
        assert result.text == ""

    async def test_timeout_maps_to_error(self):
        session = FakeSession({"https://slow.example.com/x": TimeoutError()})
        result = await _fetch_one(session, "https://slow.example.com/x", {})
        assert result.status == "error"
        assert result.http_status is None

    async def test_client_error_maps_to_error(self):
        session = FakeSession({"https://broken.example.com/x": aiohttp.ClientError("boom")})
        result = await _fetch_one(session, "https://broken.example.com/x", {})
        assert result.status == "error"
        assert result.http_status is None

    async def test_json_content_type_returns_raw_truncated(self, monkeypatch):
        cap = 200
        payload = b'{"vulnerabilities":[{"cveID":"CVE-2026-0001","description":"' + b"x" * 500 + b'"}]}'
        session = FakeSession(
            {"https://json.example.com/kev": FakeResponse(200, body=payload, content_type="application/json")}
        )
        result = await _fetch_one(session, "https://json.example.com/kev", {}, capped_ctx(cap))
        assert result.status == "success"
        assert result.content_type is not None
        assert "json" in result.content_type
        assert result.text.startswith('{"vulnerabilities')
        # Truncated -> marker appears, total bounded by cap.
        assert f"[truncated at {cap} chars — full source at https://json.example.com/kev]" in result.text
        assert len(result.text) <= cap

    @pytest.mark.parametrize(
        ("body", "content_type"),
        [
            (b"", "application/json"),
            (b"   \n\t\n ", "application/json"),
            (b"", "text/csv"),
            (b"\n\n", "text/plain"),
        ],
    )
    async def test_a_200_with_an_empty_body_is_not_a_success(self, body: bytes, content_type: str):
        """`read_body_capped` returns `b""` for an empty body and the only guard was `is None`, so
        an empty 200 shipped as `success` with `text=""`. Three things followed: an empty `###
        <url>` section rendered under the "primary grading evidence" caveat, that one result
        suppressed the all-failed "yielded no usable content" notice for every OTHER failed URL, and
        provider diagnostics reported `ok` — indistinguishable from a real fetch."""
        session = FakeSession({"https://empty.example.com/x": FakeResponse(200, body=body, content_type=content_type)})

        result = await _fetch_one(session, "https://empty.example.com/x", {})

        assert result.status == "empty_body"
        assert result.text == ""
        assert result.http_status == 200

    async def test_a_declared_charset_body_decodes_instead_of_mojibaking(self):
        """`charset=` was parsed for ROUTING and then ignored for decoding, so a Windows-1252 CSV
        rendered as grading evidence with replacement characters where its punctuation had been."""
        body = "Pollster,Approve\nO’Brien Research,44\n".encode("windows-1252")  # noqa: RUF001  # cp1252 fixture
        session = FakeSession(
            {
                "https://poll.example.com/d.csv": FakeResponse(
                    200, body=body, content_type="text/csv; charset=windows-1252"
                )
            }
        )

        result = await _fetch_one(session, "https://poll.example.com/d.csv", {})

        assert result.status == "success"
        assert "O’Brien Research" in result.text  # noqa: RUF001  # cp1252 fixture
        assert "�" not in result.text

    async def test_an_undecodable_body_is_refused_rather_than_rendered_as_mojibake(self):
        """BOM-less UTF-16 — the shape a replacement-char count alone cannot see, since every
        second byte decodes as a valid NUL. `0<?>.<?>4<?>2<?>` type-checks as text and used to
        reach the forecaster under the primary-grading-evidence caveat."""
        body = "date,value\n2026-08-01,0.42\n".encode("utf-16-le")
        session = FakeSession({"https://odd.example.com/d.csv": FakeResponse(200, body=body, content_type="text/csv")})

        result = await _fetch_one(session, "https://odd.example.com/d.csv", {})

        assert result.status == "unsupported_type"
        assert result.text == ""

    async def test_html_markup_inside_a_csv_cell_is_stripped_on_the_text_branch(self):
        """The Tier-1 half of the Datawrapper budget fix: the same class of input (a delimited
        table whose cells carry styled anchors) reaches this branch whenever a source serves its
        data as CSV directly, and the per-URL char budget should buy rows rather than markup."""
        body = b"Dates,Pollster,Approve\n8/16,<a href='https://x.test/p' style='color:#000'>Emerson College</a>,36.4\n"
        session = FakeSession(
            {"https://poll.example.com/rows.csv": FakeResponse(200, body=body, content_type="text/csv")}
        )

        result = await _fetch_one(session, "https://poll.example.com/rows.csv", {})

        assert result.status == "success"
        assert "Emerson College" in result.text
        assert "<a " not in result.text
        assert "style=" not in result.text

    async def test_json_bodies_keep_their_angle_brackets(self):
        """A JSON body's angle brackets sit inside string values that ARE the data, so the strip is
        confined to the text branches."""
        body = b'{"note": "value <a and b > c", "n": 1}'
        session = FakeSession(
            {"https://api.example.com/v": FakeResponse(200, body=body, content_type="application/json")}
        )

        result = await _fetch_one(session, "https://api.example.com/v", {})

        assert result.status == "success"
        assert result.text == body.decode("utf-8")

    async def test_a_body_that_is_not_a_document_is_still_unsupported_type(self):
        """The body IS read now (that is how the `%PDF-` magic check works), so what this
        pins is that reading it did not change the verdict for everything that is not a
        document. It replaces a pin asserting the body was never read, which was the
        behaviour that dropped every cited PDF unread."""
        session = FakeSession(
            {"https://img.example.com/chart.png": FakeResponse(200, body=b"\x89PNG\r\n", content_type="image/png")}
        )
        result = await _fetch_one(session, "https://img.example.com/chart.png", {})
        assert result.status == "unsupported_type"
        assert result.text == ""
        assert result.route == "direct"

    async def test_missing_content_type_on_a_non_document_is_unsupported_type(self):
        """A 200 OK served without a Content-Type header matches no routing prefix and reaches the document
        rung, which reads the body, finds no `%PDF-` magic, and classifies it `unsupported_type` exactly as
        before. HTML served with no content type is still not extracted: sniffing is scoped to documents,
        where the label is demonstrably unreliable and the payoff is a whole cited source."""
        resp = FakeResponse(200, body=b"<html><body>hello there</body></html>")
        del resp.headers["Content-Type"]
        session = FakeSession({"https://noct.example.com/x": resp})
        result = await _fetch_one(session, "https://noct.example.com/x", {})
        assert result.status == "unsupported_type"
        assert result.content_type is None
        assert result.text == ""


class TestEmbedShapedPages:
    """qids 44554/44556: a tracker page returned HTTP 200, extracted forecast background,
    and reported `success` while the resolving Nebraska polling average sat in two Infogram
    iframes trafilatura drops. The section published under "primary grading evidence" with
    zero polling numbers in it, byte-identical across three questions, and nothing anywhere
    said so. Two outcomes now, split by how much page text came back."""

    async def test_an_embed_shell_200_is_no_resolving_content(self, infogram_shell_html):
        session = FakeSession({"https://tracker.example.com/senate": FakeResponse(200, body=infogram_shell_html)})

        result = await _fetch_one(session, "https://tracker.example.com/senate", {})

        assert result.status == "no_resolving_content"
        assert result.status_reason == "embed_shell"
        assert result.http_status == 200  # the fetch itself succeeded; the CONTENT did not arrive
        assert result.text == ""
        assert result.unreadable_embeds == ["infogram"]

    async def test_the_same_thin_page_without_an_embed_is_withheld_as_a_thin_page(self):
        """The FLOOR is the discriminator; the embed only says where the content went.

        This test asserted the opposite when the verdict shipped — identical chrome with
        the embed markup swapped for an inert div extracted the same 167 chars and still
        published. The 2026-09-01 round found five content-free `success` renders and not
        one of them named a provider (q45088's 127-char SPA tab list, q45215's 385 chars
        of Kazakh region names), so the gate was withholding one shape of chrome and
        publishing the other.
        """
        session = FakeSession(
            {"https://plain.example.com/p": FakeResponse(200, body=_embed_shell_page("<div>chart goes here</div>"))}
        )

        result = await _fetch_one(session, "https://plain.example.com/p", {})

        assert result.status == "no_resolving_content"
        assert result.status_reason == "thin_page"
        assert result.http_status == 200  # the fetch itself succeeded; the CONTENT did not arrive
        assert result.text == ""
        assert result.unreadable_embeds == []

    async def test_prose_plus_an_unreadable_embed_keeps_the_prose_and_discloses_the_gap(
        self, tracker_with_infogram_html
    ):
        """The 44554 page itself: 2.9k chars of real background around the embed. Withholding
        it would throw away readable evidence, so the text stays and the section says plainly
        that the embedded figures are not in it — the caveat above it claims primary grading
        evidence, so an unqualified success overstated what was retrieved.

        The note LEADS the page text and says "below" because of it; as a trailer, a head-preserving trim
        deleted it, which the aggregate-trim test below pins.
        """
        session = FakeSession(
            {"https://tracker.example.com/senate/26": FakeResponse(200, body=tracker_with_infogram_html)}
        )

        result = await _fetch_one(session, "https://tracker.example.com/senate/26", {})

        assert result.status == "success"
        assert "simulate the election 50,000 times" in result.text
        assert result.unreadable_embeds == ["infogram"]
        assert "infogram embed(s) that this fetch cannot read" in result.text
        assert result.text.startswith("[This page displays data through infogram")
        assert "NOT in the page text below" in result.text

    async def test_the_disclosure_is_budgeted_inside_the_per_url_cap(self, tracker_with_infogram_html, monkeypatch):
        """The note is budgeted out of the cap (like the Tier-2 dataset lead) and never added on top of it, so
        the per-URL bound the section budget relies on still holds."""
        cap = 500
        session = FakeSession({"https://t.example.com/p": FakeResponse(200, body=tracker_with_infogram_html)})

        result = await _fetch_one(session, "https://t.example.com/p", {}, capped_ctx(cap))

        assert result.status == "success"
        assert len(result.text) <= cap
        assert "NOT in the page text below" in result.text
        # The page text is what the cap truncates; the note is not the thing cut.
        body_cap = cap - len(_unreadable_embed_disclosure(["infogram"])) - 2
        assert f"[truncated at {body_cap} chars" in result.text

    def test_the_disclosure_survives_the_aggregate_section_budget_trim(self):
        """F6: the note used to TRAIL the page text, and every truncator here preserves the
        HEAD, so the aggregate cut in `_budgeted_success_sections` deleted it outright — the
        page then rendered under the "primary grading evidence" caption with no mention of the
        unreadable embed at all, which is the q44554/44556 failure the disclosure exists to
        prevent. Sizes are derived from the prod constants so the scenario stays a REACHABLE
        one: earlier full-size pages spend most of the total, and the embed page lands last.
        """
        per_url = RESOLUTION_SOURCE_POLICY.per_url_max_chars
        assert per_url is not None
        total = resolution_presentation.RESOLUTION_SOURCE_TOTAL_MAX_CHARS
        leftover = per_url // 2  # what the embed page is left to render in
        spend = total - leftover
        filler_sizes = [per_url] * (spend // per_url)
        if spend % per_url:
            filler_sizes.append(spend % per_url)
        # Reachable on prod constants rather than hypothetical: the pages fit inside RESOLUTION_SOURCE_MAX_URLS.
        assert len(filler_sizes) + 1 <= resolution_source.RESOLUTION_SOURCE_MAX_URLS

        fillers = [
            FetchResult(
                url=f"https://p{i}.example.com/x",
                status="success",
                text="F" * size,
                http_status=200,
                content_type="text/html",
            )
            for i, size in enumerate(filler_sizes)
        ]
        embed_text = resolution_presentation._page_text_with_leads(
            "lorem ipsum " * (per_url // 2), "https://tracker.example.com/senate", ["infogram"], "", cap=per_url
        )
        embed = FetchResult(
            url="https://tracker.example.com/senate",
            status="success",
            text=embed_text,
            http_status=200,
            content_type="text/html",
            unreadable_embeds=["infogram"],
        )

        out = format_resolution_sections([*fillers, embed], datetime(2026, 9, 1, tzinfo=UTC))

        # The trim really fired — otherwise this test would pass for the wrong reason.
        assert embed_text not in out
        assert "infogram embed(s) that this fetch cannot read" in out
        assert "NOT in the page text below" in out
        # And it leads its own section, immediately under the heading.
        assert "### https://tracker.example.com/senate\n(fetched 2026-09-01)\n\n[This page displays" in out

    async def test_an_ordinary_article_carries_no_disclosure(self, article_html):
        session = FakeSession({"https://news.example.com/report": FakeResponse(200, body=article_html)})

        result = await _fetch_one(session, "https://news.example.com/report", {})

        assert result.status == "success"
        assert result.unreadable_embeds == []
        assert "cannot read" not in result.text

    def test_the_chrome_floor_sits_above_the_js_wall_floor(self):
        """Both floors read module globals so tests can retune them, and the ordering is what keeps `js_wall`
        its own population instead of a subset the chrome floor swallowed."""
        assert looks_like_page_chrome("x" * 300) is True
        assert looks_like_js_wall("x" * 300) is False
        assert looks_like_page_chrome("x" * 500) is False

    def test_a_no_resolving_content_result_is_a_loss_token_not_ok(self):
        """The diagnostics half: as `success` this reported `ok`, so the provider block read
        fully healthy on a question whose only cited source handed back no numbers."""
        results = [
            FetchResult(
                url="https://tracker.example.com/senate",
                status="no_resolving_content",
                text="",
                http_status=200,
                content_type="text/html",
                unreadable_embeds=["infogram"],
            )
        ]

        assert _fetch_result_sources(results) == {"tracker.example.com": "no_resolving_content"}

    def test_a_no_resolving_content_page_is_not_rendered_as_grading_evidence(self):
        results = [
            FetchResult(
                url="https://tracker.example.com/senate",
                status="no_resolving_content",
                text="",
                http_status=200,
                content_type="text/html",
                unreadable_embeds=["infogram"],
            )
        ]

        out = format_resolution_sections(results, datetime(2026, 9, 1, tzinfo=UTC))

        assert "### https://tracker.example.com/senate" not in out
        assert "tracker.example.com: no_resolving_content" in out
        assert "weight other evidence accordingly" in out

    def test_a_blank_no_resolving_content_result_constructs(self):
        """The success-implies-content guard must not fire on the new status: it is a FAILURE status and its
        text is empty by construction."""
        assert FetchResult(
            url="https://t.example.com/p",
            status="no_resolving_content",
            text="",
            http_status=200,
            content_type="text/html",
        )

    async def test_a_page_just_above_the_chrome_floor_still_publishes(self):
        """The elbow, from both sides. The archive census puts the shortest extraction
        that carries the resolving content at exactly 401 chars
        (myfloridaelections.com's election-date table), so the floor has to withhold at
        399 and publish at 401 or it is throwing away terse-but-real data tables.

        Trafilatura keeps the <article> paragraph verbatim, so the extraction length is the paragraph length,
        and "ab " * n is a word-shaped filler it does not collapse.
        """
        floor = verdict.RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS
        above = _prose_page("ab " * ((floor + 40) // 3))
        below = _prose_page("ab " * ((floor - 40) // 3))
        session = FakeSession(
            {"https://a.example.com/p": FakeResponse(200, body=above)}
            | {"https://b.example.com/p": FakeResponse(200, body=below)}
        )

        long_result = await _fetch_one(session, "https://a.example.com/p", {})
        short_result = await _fetch_one(session, "https://b.example.com/p", {})

        assert len(long_result.text.strip()) >= floor
        assert long_result.status == "success"
        assert short_result.status == "no_resolving_content"
        assert short_result.status_reason == "thin_page"


class TestInlineChartData:
    """qid 43949 (IOM Missing Migrants). The resolving page fetches 200 through the repo's
    own Tier-1 path and trafilatura extracts ~80k chars of incident rows and prose carrying
    none of `1342` / `Total Dead and Missing` / `2026`, because the annual series lives in a
    `data-chart` attribute. A Wayback snapshot 25 days BEFORE that forecast carries the same
    markup with 2026 = 1,240; the published forecast sat ~340 above the true level."""

    async def test_a_prose_page_gets_its_chart_series_rendered_and_stays_a_success(self):
        session = FakeSession({"https://iom.example.com/med": FakeResponse(200, body=_iom_shaped_page())})

        result = await _fetch_one(session, "https://iom.example.com/med", {})

        assert result.status == "success"
        # The resolving figure, which the prose does not carry.
        assert "2026=1240" in result.text
        assert "Total Number of Dead and Missing: 2024=2573, 2025=2185, 2026=1240" in result.text
        # The chart block LEADS, so it is the last thing any downstream trim reaches.
        assert result.text.startswith(CHART_DATA_LEAD)
        # And the page's own prose is still there under it.
        assert "context in brief" in result.text

    async def test_a_chart_only_page_is_rescued_from_the_chrome_floor(self):
        """The two fixes meet here: this page extracts 43 chars, so without the chart rung
        it is withheld as `js_wall` (43 is under the 100-char JS-wall floor, not merely
        under the 400-char chrome floor); with the rung, the numbers we recovered ARE the
        content. The mid-band pair below covers the `thin_page` counterfactual."""
        session = FakeSession(
            {"https://iom.example.com/bare": FakeResponse(200, body=_iom_shaped_page(prose="Mediterranean."))}
        )

        result = await _fetch_one(session, "https://iom.example.com/bare", {})

        assert result.status == "success"
        assert result.status_reason is None
        assert "2026=1240" in result.text

    async def test_a_mid_band_chart_page_is_rescued_from_the_thin_page_floor(self):
        """The rescue's own band: the chrome floor says withhold, the JS-wall floor says no.

        Nothing else constructs that combination — the other chart test extracts 43 chars
        (`js_wall`) and the prose one 425 (above the chrome floor) — so narrowing the
        rescue guard to `not (chart_block and looks_like_js_wall(extracted))` left the
        whole suite green while dropping the recovered 2026=1240 on every page in this
        band, which is 7 of the 8 archived sub-400 chrome extractions.
        """
        body = _mid_band_chart_page()
        extracted = classify._extract_main_text(body, "https://iom.example.com/mid") or ""
        # The fixture is only meaningful if it really sits between the two floors.
        assert looks_like_page_chrome(extracted) is True
        assert looks_like_js_wall(extracted) is False
        session = FakeSession({"https://iom.example.com/mid": FakeResponse(200, body=body)})

        result = await _fetch_one(session, "https://iom.example.com/mid", {})

        assert result.status == "success"
        assert result.status_reason is None
        assert "2026=1240" in result.text

    async def test_the_same_mid_band_page_is_withheld_when_the_chart_rung_reads_nothing(self, monkeypatch):
        """The negative twin: with the chart block empty the identical page is withheld —
        and as `thin_page`, not `js_wall`, which is what makes the rescue above the chart
        rung's doing rather than the floors'."""
        monkeypatch.setattr(classify, "render_inline_chart_data", lambda _: "")
        session = FakeSession({"https://iom.example.com/mid": FakeResponse(200, body=_mid_band_chart_page())})

        result = await _fetch_one(session, "https://iom.example.com/mid", {})

        assert result.status == "no_resolving_content"
        assert result.status_reason == "thin_page"
        assert result.text == ""

    async def test_a_malformed_chart_payload_is_ignored_rather_than_raising(self):
        """A truncated attribute (`{"series":[{"name":}]`) is what a mid-response cut or a non-JSON JS literal
        looks like. It must cost the page nothing: the prose still publishes, with no chart block and no
        exception out of the provider."""
        body = (
            "<!doctype html><html><body><article><h1>Counts</h1><p>"
            "Background prose long enough to clear the chrome floor on its own. " * 8 + "</p>"
            '<div class="charts-highchart" data-chart="{&quot;series&quot;:[{&quot;name&quot;:}]"></div>'
            "</article></body></html>"
        ).encode()
        session = FakeSession({"https://broken.example.com/p": FakeResponse(200, body=body)})

        result = await _fetch_one(session, "https://broken.example.com/p", {})

        assert result.status == "success"
        assert CHART_DATA_LEAD not in result.text
        assert "Background prose long enough" in result.text

    def test_a_config_with_no_parseable_series_renders_nothing(self):
        for html_text in (
            "",
            "<div>no charts here at all</div>",
            '<div data-chart="{}"></div>',
            '<div data-chart="{&quot;series&quot;:[]}"></div>',
            # A series whose data is callbacks / labels rather than numbers.
            '<div data-chart="{&quot;series&quot;:[{&quot;data&quot;:[&quot;n/a&quot;,null]}]}"></div>',
            # Valid JSON that is not an object.
            '<div data-chart="[1,2,3]"></div>',
        ):
            assert render_inline_chart_data(html_text) == ""

    def test_the_script_call_form_is_read_when_its_argument_is_json(self):
        html_text = (
            "<script>Highcharts.chart('container', "
            '{"title":{"text":"Weekly rate"},'
            '"xAxis":{"categories":["W1","W2"]},'
            '"series":[{"name":"Rate","data":[1.5,2.25]}]}'
            ");</script>"
        )

        out = render_inline_chart_data(html_text)

        assert "Chart 1 — Weekly rate" in out
        assert "Rate: W1=1.5, W2=2.25" in out

    def test_a_brace_inside_a_string_does_not_close_the_config_early(self):
        html_text = (
            '<script>new Highcharts.Chart({"title":{"text":"Deaths {2014-2026}"},'
            '"series":[{"name":"Total","data":[7]}]});</script>'
        )

        assert "Total: 7" in render_inline_chart_data(html_text)

    def test_point_object_and_pair_shapes_carry_their_own_labels(self):
        html_text = (
            '<div data-chart="'
            + _escape_config(
                {
                    "series": [
                        {"name": "Named", "data": [{"name": "Jan", "y": 4}, {"name": "Feb", "y": 5}]},
                        {"name": "Paired", "data": [["Q1", 10], ["Q2", 11.5]]},
                    ]
                }
            )
            + '"></div>'
        )

        out = render_inline_chart_data(html_text)

        assert "Named: Jan=4, Feb=5" in out
        assert "Paired: Q1=10, Q2=11.5" in out

    def test_a_value_too_small_to_display_renders_zero_not_minus_zero(self):
        """A tiny negative delta is a rounding artifact, not a fall. The shared
        ``number_format`` rule strips the sign off a magnitude that rounds away; this
        module's own float branch used to render it as "-0"."""
        html_text = (
            '<div data-chart="'
            + _escape_config({"series": [{"name": "Delta", "data": [["Jan", -1e-7], ["Feb", 1e-9]]}]})
            + '"></div>'
        )

        out = render_inline_chart_data(html_text)

        assert "Delta: Jan=0, Feb=0" in out
        assert "-0" not in out

    def test_a_declared_datetime_axis_renders_dates_not_epoch_millis(self):
        """Highcharts defines a datetime axis in ms since the epoch, UTC. Without the conversion a tracker's
        own daily series renders `1756771200000=42`, which is the shape most likely to matter rendered as
        noise."""
        html_text = (
            '<div data-chart="'
            + _escape_config(
                {
                    "xAxis": {"type": "datetime"},
                    "series": [{"name": "Daily", "data": [[1788220800000, 41], [1788307200000, 42]]}],
                }
            )
            + '"></div>'
        )

        assert "Daily: 2026-09-01=41, 2026-09-02=42" in render_inline_chart_data(html_text)

    def test_a_numeric_x_axis_without_the_datetime_declaration_is_left_alone(self):
        """The conversion is keyed on the axis's own declaration, never on the magnitude of the x values, so a
        chart plotting a large quantity on x is not re-dated."""
        html_text = '<div data-chart="' + _escape_config({"series": [{"data": [[1788220800000, 41]]}]}) + '"></div>'

        out = render_inline_chart_data(html_text)

        assert "1788220800000=41" in out
        assert "2026-" not in out

    def test_long_series_keep_the_newest_points_and_say_so(self):
        """The resolving value is the newest one, so the window is taken from the END."""
        n = resolution_chart_data.RESOLUTION_SOURCE_CHART_MAX_POINTS + 5
        html_text = (
            '<div data-chart="'
            + _escape_config(
                {
                    "xAxis": [{"categories": [f"m{i}" for i in range(n)]}],
                    "series": [{"name": "Monthly", "data": list(range(n))}],
                }
            )
            + '"></div>'
        )

        out = render_inline_chart_data(html_text)

        assert f"(last {resolution_chart_data.RESOLUTION_SOURCE_CHART_MAX_POINTS} of {n} points)" in out
        assert f"m{n - 1}={n - 1}" in out
        assert "m0=0" not in out

    def test_the_block_is_bounded_and_drops_whole_charts(self):
        """A half-rendered row reads like a complete series, so charts are dropped whole and the omitted count
        is stated. The bound has to hold including that note."""
        big = _escape_config(
            {
                "xAxis": [{"categories": [f"category-label-{i}" for i in range(16)]}],
                "series": [{"name": f"series-{s}", "data": [1000000 + i for i in range(16)]} for s in range(4)],
            }
        )
        html_text = "".join(f'<div data-chart="{big}"></div>' for _ in range(4))

        out = render_inline_chart_data(html_text)

        assert len(out) <= resolution_chart_data.RESOLUTION_SOURCE_CHART_BLOCK_MAX_CHARS
        assert "further chart(s) on this page omitted — chart-data budget" in out

    def test_at_most_max_charts_are_rendered(self):
        """The chart cap leaves readable charts off the page, so it is an omission like the char budget and has
        to be stated. It used to `break` silently, which made the "the omitted count is stated" claim above
        false on this exact shape."""
        one = _escape_config({"series": [{"name": "S", "data": [1]}]})
        html_text = "".join(f'<div data-chart="{one}"></div>' for _ in range(8))

        out = render_inline_chart_data(html_text)

        assert out.count("\nChart ") == resolution_chart_data.RESOLUTION_SOURCE_CHART_MAX_CHARTS
        omitted = 8 - resolution_chart_data.RESOLUTION_SOURCE_CHART_MAX_CHARTS
        assert f"[{omitted} further chart(s) on this page omitted" in out

    async def test_a_page_of_never_closing_chart_calls_costs_a_bounded_scan(self, monkeypatch):
        """The candidate bound has to count sites EXAMINED, not configs KEPT.

        A `Highcharts.chart({` whose braces never close appends no candidate, so under the
        old `len(candidates)` test every such site paid its own brace scan and the page
        cost sites x the config-char bound — quadratic, in an uncancellable thread under
        the provider's 45 s wall, where blowing the wall discards every already-fetched
        page. Asserted as outcomes (how many brace scans ran, what renders, what
        publishes) and never as elapsed time, which would be flaky on shared CI.
        """
        scans = 0
        real_balanced_object = resolution_chart_data._balanced_object

        def counting_balanced_object(text: str, start: int) -> str | None:
            nonlocal scans
            scans += 1
            return real_balanced_object(text, start)

        monkeypatch.setattr(resolution_chart_data, "_balanced_object", counting_balanced_object)
        prose = "Counts are compiled from monthly returns and revised annually. " * 8
        body = (
            "<!doctype html><html><head><title>Counts</title></head><body>"
            f"<article><h1>Counts</h1><p>{prose}</p>"
            + "<script>Highcharts.chart({'series':[1</script>" * 2000
            + "</article></body></html>"
        ).encode()

        assert render_inline_chart_data(body.decode()) == ""
        assert scans <= resolution_chart_data.RESOLUTION_SOURCE_CHART_MAX_CANDIDATES

        session = FakeSession({"https://spin.example.com/p": FakeResponse(200, body=body)})
        result = await _fetch_one(session, "https://spin.example.com/p", {})

        # The page itself is unaffected: unparseable decoration costs the chart rung a bounded scan, the page nothing.
        assert result.status == "success"
        assert CHART_DATA_LEAD not in result.text
        assert "compiled from monthly returns" in result.text

    def test_a_two_arg_chart_call_still_reaches_its_config_object(self):
        """Why the fix bounds attempts rather than refusing a `{` far from the paren: a
        real page passes its container element first, which puts the config's opening brace
        tens of chars past `(`."""
        html_text = (
            "<script>Highcharts.chart(document.getElementById('a-long-container-id-here'), "
            '{"series":[{"name":"Rate","data":[3]}]});</script>'
        )

        assert "Rate: 3" in render_inline_chart_data(html_text)

    def test_the_series_cap_keeps_the_leading_series_and_drops_the_rest(self):
        """The cap truncates silently, and nothing pinned it: the widest other fixture
        builds exactly MAX_SERIES series, so neutralizing the slice passed the suite."""
        cap = resolution_chart_data.RESOLUTION_SOURCE_CHART_MAX_SERIES
        total = cap + 2
        html_text = (
            '<div data-chart="'
            + _escape_config({"series": [{"name": f"series-{i}", "data": [100 + i]} for i in range(total)]})
            + '"></div>'
        )

        out = render_inline_chart_data(html_text)

        for i in range(cap):
            assert f"series-{i}: {100 + i}" in out
        for i in range(cap, total):
            assert f"series-{i}" not in out

    def test_the_candidate_scan_stops_at_the_bound(self):
        """Asserted on the helper because the three-chart render cap hides it: a page with
        hundreds of `data-chart` attributes renders the same three either way."""
        cap = resolution_chart_data.RESOLUTION_SOURCE_CHART_MAX_CANDIDATES
        one = _escape_config({"series": [{"name": "S", "data": [1]}]})
        page = f'<div data-chart="{one}"></div>' * (cap + 5)

        assert len(resolution_chart_data._candidate_configs(page)) == cap

    def test_a_config_over_the_char_bound_is_skipped(self, monkeypatch):
        """Both halves of the byte bound: `_render_config`'s per-config check and
        `_balanced_object`'s scan window, which read the same constant."""
        config = {"series": [{"name": "Rate", "data": list(range(12))}]}
        assert len(json.dumps(config)) > 50
        monkeypatch.setattr(resolution_chart_data, "RESOLUTION_SOURCE_CHART_MAX_CONFIG_CHARS", 50)
        html_text = '<div data-chart="' + _escape_config(config) + '"></div>'

        assert render_inline_chart_data(html_text) == ""
        assert resolution_chart_data._balanced_object("{" + "a" * 200 + "}", 0) is None

    async def test_the_chart_block_is_budgeted_inside_the_per_url_cap(self, monkeypatch):
        """Same rule as the embed disclosure: leads come OUT of the per-URL cap and never on top of it, so the
        aggregate section budget's arithmetic still holds."""
        cap = 400
        session = FakeSession({"https://iom.example.com/med": FakeResponse(200, body=_iom_shaped_page())})

        result = await _fetch_one(session, "https://iom.example.com/med", {}, capped_ctx(cap))

        assert result.status == "success"
        assert len(result.text) <= cap


class TestResolutionSourceFetchMarker:
    """Item 19d: per-URL outcomes as ONE harvested marker line (`resolution_source_fetch`).

    The outcomes used to live only in free-text logs and the comment's diagnostics block,
    so "cdc.gov is 0 successes in 1,069 fetch records" meant re-scraping GHA logs that
    expire at 90 days.
    """

    async def test_one_line_per_fetched_url_with_status_and_http_code(self, article_html, monkeypatch, caplog):
        """One line per fetched URL. The 403 carries its failure_class (server absent, since the fake response
        sends no Server header), which is what separates an egress-reputation refusal from a host fault."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {
                "https://www.bls.gov/cpi/": FakeResponse(200, body=article_html, content_type="text/html"),
                "https://cbp.gov/data": FakeResponse(403, body=b"", content_type="text/html"),
            }
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://www.bls.gov/cpi/ and https://cbp.gov/data")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        lines = [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")]
        assert lines == [
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://www.bls.gov/cpi/ status=ok http=200 embeds=none caller=resolution_source",
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://cbp.gov/data "
            "status=blocked http=403 embeds=none failure_class=http_403 caller=resolution_source",
        ]

    async def test_a_spaced_server_header_stays_one_marker_token(self, monkeypatch, caplog):
        """Emitter and parser pinned against ONE string.

        The marker spec reads the field as ``(?:\\s+server=(?P<server>\\S+))?`` with no trailing
        anchor, so a ``Server`` value that kept its space would still MATCH: the archive would
        record ``server=apache/2.4.62`` and drop ``(Debian)`` with no parse error to notice. The
        whole-line assertion pins what the emitter writes; the spec-regex assertion pins that the
        parser reads all of it back as the one value.
        """
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {
                "https://cbp.gov/data": FakeResponse(
                    403, body=b"", content_type="text/html", headers={"Server": "Apache/2.4.62 (Debian)"}
                )
            }
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://cbp.gov/data")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        (line,) = [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")]
        assert line == (
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://cbp.gov/data "
            "status=blocked http=403 embeds=none failure_class=http_403 server=apache/2.4.62_(debian) caller=resolution_source"
        )
        spec = next(s for s in MARKER_SPECS if s.name == "resolution_source_fetch")
        match = spec.regex.search(line)
        assert match is not None
        assert match.groupdict() == {
            "question": "999",
            "url": "https://cbp.gov/data",
            "status": "blocked",
            "http": "403",
            "embeds": "none",
            "reason": None,
            "route": None,
            "failure_class": "http_403",
            "exc": None,
            "server": "apache/2.4.62_(debian)",
            "passages_returned": None,
            "passages_grounded": None,
            "fallback_used": None,
            "digest_outcome": None,
            "digest_scope": None,
            "caller": "resolution_source",
        }

    async def test_the_marker_names_the_unreadable_embed_providers(
        self, tracker_with_infogram_html, monkeypatch, caplog
    ):
        """The whole point on the 44554 shape: the fetch is a legitimate `success`, so the only thing that
        makes the missing numbers queryable is this field."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {"https://www.racetothewh.com/senate/26": FakeResponse(200, body=tracker_with_infogram_html)}
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://www.racetothewh.com/senate/26")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://www.racetothewh.com/senate/26 "
            "status=ok http=200 embeds=infogram caller=resolution_source"
        ]

    async def test_the_marker_names_which_rule_withheld_the_page(self, infogram_shell_html, monkeypatch, caplog):
        """`no_resolving_content` has two rules behind it and the status alone cannot say
        which fired. `reason` is what keeps the embed-gated population (queryable since
        2026-08) separable from the thin-page one the ungated floor added."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {
                "https://tracker.example.com/senate": FakeResponse(200, body=infogram_shell_html),
                "https://data.example.com/": FakeResponse(200, body=_embed_shell_page("<div>tabs</div>")),
            }
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://tracker.example.com/senate and https://data.example.com/")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://tracker.example.com/senate "
            "status=no_resolving_content http=200 embeds=infogram reason=embed_shell caller=resolution_source",
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://data.example.com/ "
            "status=no_resolving_content http=200 embeds=none reason=thin_page caller=resolution_source",
        ]

    async def test_a_fetch_that_never_got_a_response_reports_http_n_a(self, monkeypatch, caplog):
        """A transport failure carries its class and the exception's name, so the archive tells a timeout from
        a TLS or DNS refusal without re-scraping the run log."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession({"https://slow.example.com/x": TimeoutError()})
        monkeypatch.setattr(guard, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://slow.example.com/x")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://slow.example.com/x "
            "status=error http=n/a embeds=none failure_class=timeout exc=TimeoutError caller=resolution_source"
        ]

    async def test_no_fetch_is_logged_twice(self, article_html, monkeypatch, caplog):
        """One outcome line per fetch, in one format. The free-text
        `resolution_source fetched <netloc> (<status>)` lines the marker supersedes were
        deleted rather than left beside it; what remains are REASON lines (a decode score,
        an unread content-type, an SSRF rejection) that carry what the marker cannot."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {
                "https://www.bls.gov/cpi/": FakeResponse(200, body=article_html, content_type="text/html"),
                "https://cbp.gov/data": FakeResponse(403, body=b"", content_type="text/html"),
            }
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://www.bls.gov/cpi/ and https://cbp.gov/data")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert not [m for m in caplog.messages if "resolution_source fetched" in m]

    async def test_a_followed_meta_refresh_names_its_route_and_logs_the_escalation(
        self, article_html, monkeypatch, caplog
    ):
        """`route` is what separates a page a rung rescued from one the direct read got, and
        the escalation line is the only place the trigger status and the rung's cost appear —
        the fetch line above it carries the FINAL outcome only.

        The escalation's `url` is the URL the rung was invoked ON, the stub rather than the target the fetch
        line names, because that is where the ladder engaged and what `from_status` describes.
        """
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {
                "https://cdc.example.com/surveillance": FakeResponse(200, body=_meta_refresh_stub("/data/current")),
                "https://cdc.example.com/data/current": FakeResponse(200, body=article_html),
            }
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://cdc.example.com/surveillance")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://cdc.example.com/data/current "
            "status=ok http=200 embeds=none route=meta_refresh caller=resolution_source"
        ]
        escalations = [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_ESCALATION:")]
        assert len(escalations) == 1
        assert re.fullmatch(
            r"RESOLUTION_SOURCE_ESCALATION: question=999 url=https://cdc\.example\.com/surveillance "
            r"from_status=js_wall rung=meta_refresh outcome=success wall_s=\d+\.\d\d caller=resolution_source",
            escalations[0],
        ), escalations[0]

    async def test_a_local_pdf_read_names_its_route(self, monkeypatch, caplog):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        body = build_text_pdf([["Hospitalizations reported: 922", "Deaths reported: 2 as of August 24"]])
        session = FakeSession(
            {"https://cdc.example.com/r.pdf": FakeResponse(200, body=body, content_type="application/pdf")}
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="Resolves per https://cdc.example.com/r.pdf hospitalizations")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://cdc.example.com/r.pdf "
            "status=ok http=200 embeds=none route=pdf_local caller=resolution_source"
        ]
        assert re.fullmatch(
            r"RESOLUTION_SOURCE_ESCALATION: question=999 url=https://cdc\.example\.com/r\.pdf "
            r"from_status=unsupported_type rung=pdf_local outcome=success wall_s=\d+\.\d\d caller=resolution_source",
            next(m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_ESCALATION:")),
        )

    async def test_a_document_whose_passages_matched_nothing_says_so_on_the_line(self, monkeypatch, caplog):
        """`status=ok` on a zero-passage digest was byte-identical to one carrying the
        resolving paragraph, on the surface whose contract is that success means CONTENT. The
        line now carries both halves: the withholding status AND the reason it was withheld,
        which is what separates it from a page whose chrome we could not read.

        The rung FIRED and the withhold is its outcome, which is the convention that keeps a rung that fires
        often and rescues nothing distinguishable from one that never fires.
        """
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        body = build_text_pdf([["Hospitalizations reported: 922", "Deaths reported: 2 as of August 24"]])
        session = FakeSession(
            {"https://cdc.example.com/r.pdf": FakeResponse(200, body=body, content_type="application/pdf")}
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="Resolves per https://cdc.example.com/r.pdf corn futures settlement")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://cdc.example.com/r.pdf "
            "status=no_resolving_content http=200 embeds=none reason=no_matching_passage route=pdf_local caller=resolution_source"
        ]
        assert re.fullmatch(
            r"RESOLUTION_SOURCE_ESCALATION: question=999 url=https://cdc\.example\.com/r\.pdf "
            r"from_status=unsupported_type rung=pdf_local outcome=no_resolving_content wall_s=\d+\.\d\d caller=resolution_source",
            next(m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_ESCALATION:")),
        )

    async def test_a_skipped_rung_is_counted_but_not_reported_as_an_escalation(self, monkeypatch, caplog):
        """The marker means "a rung fired". A rung that never ran for want of wall budget
        rides `details["counts"]` instead, where it stays queryable without inflating the
        rung's own fire rate.

        Four things the counts assertions below pin. The rung's own FIRE count stays zero, because a skip must
        not inflate the rate at which the rung is measured to work. The withheld page also earned a browser
        attempt, which this package's autouse fixture declines, so the second skip is the transport reporting
        itself unavailable rather than another budget skip. The same budget skip appears once more under its
        per-rung key, so the archive can say WHICH rung the wall bound, and the aggregate stays as it was.
        The last assertion is a total rather than an exact dict because the key SET grows every time a rung
        lands, and a literal there made three unrelated commits edit this one test; the total is three rather
        than two because the budget skip is counted in the aggregate AND in its per-rung key.
        """
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setattr(classify, "RESOLUTION_SOURCE_META_REFRESH_MIN_BUDGET_S", 1_000_000.0)
        session = FakeSession(
            {"https://cdc.example.com/surveillance": FakeResponse(200, body=_meta_refresh_stub("/data/current"))}
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://cdc.example.com/surveillance")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://cdc.example.com/surveillance "
            "status=js_wall http=200 embeds=none caller=resolution_source"
        ]
        assert not [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_ESCALATION:")]
        counts = pop_provider_detail(q.id_of_question, "resolution_source")["counts"]
        assert counts["rung_budget_skips"] == 1
        assert counts["meta_refresh_hops"] == 0
        assert counts["renderer_unavailable_skips"] == 1
        assert counts["meta_refresh_budget_skips"] == 1
        assert sum(counts.values()) == 3


class TestFetchResolutionSources:
    async def test_per_host_serialization(self, article_html, monkeypatch):
        """Two URLs on the same host must never fetch concurrently, while
        distinct hosts may. We track per-host in-flight counts in FakeSession
        and assert peak == 1 for the shared host."""

        # Slow the two same-host reads so their windows would overlap without the semaphore.
        original_read = FakeResponse.read
        slow_hosts_seen: dict[str, int] = {}

        async def slow_read(self: FakeResponse) -> bytes:
            """A microscopic sleep gives the event loop a chance to schedule the second same-host coroutine,
            which the semaphore must hold back."""
            host_probe = "same-host"  # marker for diagnostic only
            slow_hosts_seen[host_probe] = slow_hosts_seen.get(host_probe, 0) + 1
            await asyncio.sleep(0.01)
            return await original_read(self)  # type: ignore[misc]

        monkeypatch.setattr(FakeResponse, "read", slow_read)

        # Provide two URLs on the SAME host (must serialize) and one on a DIFFERENT host.
        session = FakeSession(
            {
                "https://a.example.com/one": FakeResponse(200, body=article_html, content_type="text/html"),
                "https://a.example.com/two": FakeResponse(200, body=article_html, content_type="text/html"),
                "https://b.example.com/three": FakeResponse(200, body=article_html, content_type="text/html"),
            }
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)

        results = await fetch_resolution_sources(
            [
                "https://a.example.com/one",
                "https://a.example.com/two",
                "https://b.example.com/three",
            ],
        )
        assert len(results) == 3
        # Same-host peak must be exactly 1 — that's the semaphore's guarantee.
        assert session.host_peak["a.example.com"] == 1
        # Other host observed at most 1 concurrent request (only one URL scheduled to it).
        assert session.host_peak.get("b.example.com", 0) == 1
        # Session was closed.
        assert session.closed is True

    async def test_redirect_convergence_serializes_on_final_host(self, article_html, monkeypatch):
        """F15 regression: two chains starting on DISTINCT hosts that both
        redirect to the SAME final host must serialize there. Keying the
        semaphore on the original URL's netloc (the old bug) gives each task
        its own semaphore, so the shared final host sees concurrency 2."""

        class SlowReadResponse(FakeResponse):
            async def read(self) -> bytes:
                """Keep the final-host GET context open long enough for the other task's GET to arrive.

                Without per-hop semaphores the two windows overlap and host_peak records 2.
                """
                await asyncio.sleep(0.01)
                return self._body

        session = FakeSession(
            {
                "https://a.example.com/one": FakeResponse(302, headers={"Location": "https://c.example.com/final"}),
                "https://b.example.com/two": FakeResponse(302, headers={"Location": "https://c.example.com/final"}),
                "https://c.example.com/final": SlowReadResponse(200, body=article_html, content_type="text/html"),
            }
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)

        results = await fetch_resolution_sources(
            ["https://a.example.com/one", "https://b.example.com/two"],
        )
        assert [r.status for r in results] == ["success", "success"]
        # The politeness guarantee holds at the CONVERGED host, not just at the original ones.
        assert session.host_peak["c.example.com"] == 1

    async def test_redirect_revisiting_initial_host_does_not_deadlock(self, article_html, monkeypatch):
        """A→B→A chain: strict per-hop acquire/release must never re-acquire a
        semaphore the task still holds (asyncio semaphores are not reentrant).
        wait_for turns a reentrancy regression into a fast TimeoutError
        instead of hanging the suite."""
        session = FakeSession(
            {
                "https://a.example.com/start": FakeResponse(302, headers={"Location": "https://b.example.com/mid"}),
                "https://b.example.com/mid": FakeResponse(302, headers={"Location": "https://a.example.com/final"}),
                "https://a.example.com/final": FakeResponse(200, body=article_html, content_type="text/html"),
            }
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)

        results = await asyncio.wait_for(
            fetch_resolution_sources(["https://a.example.com/start"]),
            timeout=5.0,
        )
        assert len(results) == 1
        assert results[0].status == "success"
        assert results[0].url == "https://a.example.com/final"

    async def test_unexpected_error_cancels_and_drains_an_in_flight_sibling(self, monkeypatch):
        """The other half of the F5 teardown guard (the wall-clock-cancellation half
        is pinned in the Datawrapper suite): when one task dies on an exception the
        fetcher does NOT catch, the gather re-raises immediately and its still-running
        siblings must be cancelled and drained BEFORE the session closes. Closing
        first is what yanks transports out from under live requests."""
        events: list[str] = []

        class _HangingResponse(FakeResponse):
            async def read(self) -> bytes:
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    events.append("sibling-settled")
                    raise
                raise AssertionError("unreachable: the sibling should be cancelled")

        class _EventSession(FakeSession):
            async def close(self) -> None:
                events.append("session-closed")
                await super().close()

        session = _EventSession(
            {
                "https://slow.example.com/x": _HangingResponse(200, body=b"", content_type="text/html"),
                # Outside the (ClientError, TimeoutError) the fetcher handles, so the gather re-raises it.
                "https://broken.example.com/y": RuntimeError("driver blew up mid-fetch"),
            }
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)

        with pytest.raises(RuntimeError, match="driver blew up mid-fetch"):
            await asyncio.wait_for(
                fetch_resolution_sources(["https://slow.example.com/x", "https://broken.example.com/y"]),
                timeout=5.0,
            )

        assert events == ["sibling-settled", "session-closed"]
        assert session.host_inflight["slow.example.com"] == 0
        assert session.closed is True


class TestAriaTableRewriteInTheFetch:
    """The end-to-end half of the ARIA rewrite (helper-level cases live in the helpers
    module): a real cdc.gov stat block reaches the forecaster with its labels attached."""

    async def test_a_cdc_stat_block_publishes_its_labelled_rows(self):
        session = FakeSession(
            {"https://www.cdc.gov/cyclosporiasis/": FakeResponse(200, body=cdc_aria_stat_block_page())}
        )

        result = await _fetch_one(session, "https://www.cdc.gov/cyclosporiasis/", {})

        assert result.status == "success"
        assert "| Hospitalizations | 922 |" in result.text
        assert result.route == "direct", "the rewrite is part of extraction, not a ladder rung"


class TestMetaRefreshHop:
    """The cdc.gov surveillance stub: HTTP 200, ~300 bytes, whose only content is a
    meta-refresh tag pointing at the page the question actually resolves on. No 3xx and no
    `Location`, so the manual redirect loop never saw it and the fetch called the stub a JS
    wall — one of the fetcher's most common silent losses."""

    async def test_the_stub_is_followed_to_the_real_page(self, article_html):
        session = FakeSession(
            {
                "https://cdc.example.com/surveillance": FakeResponse(200, body=_meta_refresh_stub("/data/current")),
                "https://cdc.example.com/data/current": FakeResponse(200, body=article_html),
            }
        )

        result = await _fetch_one(session, "https://cdc.example.com/surveillance", {})

        assert result.status == "success"
        assert result.route == "meta_refresh"
        assert "Bureau of Labor Statistics" in result.text
        assert result.url == "https://cdc.example.com/data/current"
        assert session.requested == [
            "https://cdc.example.com/surveillance",
            "https://cdc.example.com/data/current",
        ]

    async def test_the_target_is_re_guarded_against_a_private_address(self):
        """A hop target this module derived is not more trusted than a `Location` header:
        the same preflight runs on it, so a crafted resolution source cannot reach the
        instance metadata service through a refresh tag."""
        session = FakeSession(
            {
                "https://evil.example.com/p": FakeResponse(
                    200, body=_meta_refresh_stub("http://169.254.169.254/latest/meta-data/")
                )
            }
        )

        result = await _fetch_one(session, "https://evil.example.com/p", {})

        assert result.status == "ssrf_blocked"
        assert result.text == ""
        assert session.requested == ["https://evil.example.com/p"], "the target was never dialled"

    async def test_a_metaculus_target_is_refused(self):
        session = FakeSession(
            {
                "https://tracker.example.com/p": FakeResponse(
                    200, body=_meta_refresh_stub("https://www.metaculus.com/questions/999/")
                )
            }
        )

        result = await _fetch_one(session, "https://tracker.example.com/p", {})

        assert result.status == "blocked"
        assert result.status_reason == "metaculus_self_ref"
        assert session.requested == ["https://tracker.example.com/p"]

    async def test_the_hop_consumes_a_redirect_slot(self):
        """A stub pointing at itself is a refresh loop. It has to be bounded by the same
        `MAX_REDIRECTS` cap a 3xx chain is, which is the whole reason the rung returns a
        next-hop URL instead of recursing."""
        session = FakeSession({"https://loop.example.com/p": FakeResponse(200, body=_meta_refresh_stub("/p"))})

        result = await _fetch_one(session, "https://loop.example.com/p", {})

        assert result.status == "error"
        assert len(session.requested) == direct_fetch.MAX_REDIRECTS + 1

    async def test_a_page_that_already_has_content_is_served_as_is(self, article_html):
        """Some content-management systems emit a refresh tag beside real content (a
        canonical-URL nudge). The rung is only reached with nothing readable, so the page
        we already have is never thrown away for a re-fetch."""
        with_tag = article_html.replace(
            b"<body>", b'<body><meta http-equiv="refresh" content="0; url=/somewhere-else">'
        )
        session = FakeSession({"https://news.example.com/report": FakeResponse(200, body=with_tag)})

        result = await _fetch_one(session, "https://news.example.com/report", {})

        assert result.status == "success"
        assert result.route == "direct"
        assert session.requested == ["https://news.example.com/report"]

    async def test_the_hop_is_skipped_when_the_wall_budget_is_spent(self, caplog):
        """Self-bounding, because the provider's outer `wait_for` discards every page that
        already fetched when it fires: with no budget left, the stub's own verdict is worth
        more than an attempt that could cost the whole question."""
        session = FakeSession(
            {"https://cdc.example.com/surveillance": FakeResponse(200, body=_meta_refresh_stub("/data/current"))}
        )
        spent = LadderContext(started=time.monotonic() - resolution_source.RESOLUTION_SOURCE_WALL_TIMEOUT)

        with caplog.at_level("WARNING", logger="metaculus_bot.research.fetch_ladder.context"):
            result = await _fetch_one(session, "https://cdc.example.com/surveillance", {}, spent)

        assert result.status == "js_wall"
        assert result.route == "direct"
        assert session.requested == ["https://cdc.example.com/surveillance"]
        # Two skips for want of wall, each self-bounded against the same spent budget.
        assert [(a.rung, a.skipped_reason) for a in result.rung_attempts] == [
            ("meta_refresh", "wall_budget"),
            ("rendered", "wall_budget"),
        ]
        assert any("skipping the meta-refresh hop" in m for m in caplog.messages)
        assert any("skipping the rendered rung" in m for m in caplog.messages)


class TestTheHopRefusalPolicy:
    """The two checks every URL this module DERIVED owes before anything dials it, the public-URL
    preflight and the Metaculus self-reference refusal, have one home, ``_hop_refusal``. The
    terminal-result sites (a ``Location`` header, a meta-refresh tag) map its token onto a
    ``FetchResult``; the decline-shaped sites (the render landing, a Wayback capture's innermost
    URL) log and return None. Before the helper, the rendered rung shipped a third hand-rolled copy
    of the pair, and a third check added under `_vetted_hop_target`'s "the ONE place" contract would
    silently have missed the one site that decides which URL Chromium dials."""

    # Userinfo makes a URL non-public (`https://trusted@10.0.0.1/` defeats hostname trust) and the
    # host makes it a self-reference, so this one URL fails both checks with nothing patched.
    _BOTH = "https://user@www.metaculus.com/questions/999/"

    async def test_publicness_is_checked_first(self):
        """The order is a telemetry contract: a URL that is both non-public and a self-reference
        has always been recorded as ``ssrf_blocked``, never as the self-reference's ``blocked``."""
        assert await guard._hop_refusal(self._BOTH) == "ssrf_blocked"
        assert await guard._hop_refusal("https://www.metaculus.com/questions/999/") == "metaculus_self_ref"
        assert await guard._hop_refusal("https://tracker.example.com/senate") is None

    async def test_the_terminal_site_keeps_its_status_strings(self):
        """The reason is what keeps the paid rung off a URL we refused (`_url_context_rung_applies`) while the
        `blocked` status contract stays intact, and it rides the fetch marker as `reason=`."""
        blocked_both_ways = await guard._vetted_hop_target(
            self._BOTH, "https://tracker.example.com/p", content_type="", kind="redirect"
        )
        self_ref_only = await guard._vetted_hop_target(
            "https://www.metaculus.com/questions/999/",
            "https://tracker.example.com/p",
            content_type="",
            kind="redirect",
        )

        assert isinstance(blocked_both_ways, FetchResult)
        assert blocked_both_ways.status == "ssrf_blocked"
        assert blocked_both_ways.status_reason is None
        assert isinstance(self_ref_only, FetchResult)
        assert self_ref_only.status == "blocked"
        assert self_ref_only.status_reason == "metaculus_self_ref"

    async def test_the_render_landing_is_decided_by_the_same_helper(self, monkeypatch, caplog):
        """The decline site with the highest stakes routes through the helper rather than a copy: a
        refusal the helper returns for a public, non-Metaculus landing still stops the render."""
        calls: list[str] = []

        async def _refuse(candidate_url: str) -> str:
            calls.append(candidate_url)
            await asyncio.sleep(0)
            return "ssrf_blocked"

        monkeypatch.setattr(guard, "_hop_refusal", _refuse)
        renders: list[dict[str, object]] = []
        monkeypatch.setattr(rungs, "render_page", _fake_render(None, renders))
        landed = "https://www.tracker.example.com/senate"
        direct = FetchResult(url=landed, status="js_wall", text="", http_status=200, content_type="text/html")

        with caplog.at_level("WARNING", logger="metaculus_bot.research.fetch_ladder.guard"):
            result = await rungs._rendered_rung(_URL, direct, {}, LadderContext())

        assert result is None
        assert calls == [landed]
        assert renders == []
        assert [message for message in caplog.messages if "not rendering" in message]

    async def test_the_landing_re_vet_is_one_helper_with_the_cited_url_owing_nothing(self, monkeypatch, caplog):
        """`_landing_refused` is the one copy of "re-vet the landing before a second transport dials
        it". The equality short-circuit is a rule rather than an optimisation: the cited URL was
        vetted hop by hop by `_fetch_direct`, so it never reaches `_hop_refusal` here, while a
        landing that differs is a derived URL and does. The WARNING names the caller's verb."""
        calls: list[str] = []

        async def _refuse(candidate_url: str) -> str | None:
            calls.append(candidate_url)
            await asyncio.sleep(0)
            return "ssrf_blocked" if "10.0.0.8" in candidate_url else None

        monkeypatch.setattr(guard, "_hop_refusal", _refuse)
        allowed_landing = "https://www.tracker.example.com/senate"

        with caplog.at_level("WARNING", logger="metaculus_bot.research.fetch_ladder.guard"):
            same = await guard._landing_refused(_URL, _URL, action="rendering")
            allowed = await guard._landing_refused(allowed_landing, _URL, action="rendering")
            refused = await guard._landing_refused("http://10.0.0.8/status", _URL, action="re-dialing")

        assert (same, allowed, refused) == (False, False, True)
        assert calls == [allowed_landing, "http://10.0.0.8/status"]
        assert [
            message
            for message in caplog.messages
            if "not re-dialing 10.0.0.8, where the cited tracker.example.com landed" in message
        ]
        assert not any("not rendering" in message for message in caplog.messages)

    async def test_both_second_transport_rungs_decide_the_landing_through_it(self, monkeypatch):
        """The browser rung and the impersonated retry both dial `direct.url`, and both route the
        decision through the helper rather than a copy of it: a refusal it returns is a decline
        with no attempt opened and nothing dialed, on either transport."""
        seen: list[tuple[str, str, str]] = []

        async def _refused(landing_url: str, cited_url: str, *, action: str) -> bool:
            seen.append((landing_url, cited_url, action))
            await asyncio.sleep(0)
            return True

        monkeypatch.setattr(guard, "_landing_refused", _refused)
        monkeypatch.setattr(impersonated_fetch, "IMPERSONATE_TRIGGER_STATUSES", IMPERSONATE_TRIGGER_STATUSES)
        renders: list[dict[str, object]] = []
        monkeypatch.setattr(rungs, "render_page", _fake_render(None, renders))
        dials: list[dict[str, object]] = []
        monkeypatch.setattr(
            rungs,
            "fetch_impersonated",
            fake_impersonated_fetch(_impersonated(200, body=_prose_page("Whatever the host served.")), dials),
        )
        landed = "https://www.tracker.example.com/senate"
        walled = FetchResult(url=landed, status="js_wall", text="", http_status=200, content_type="text/html")
        refused = FetchResult(url=landed, status="blocked", text="", http_status=403, content_type="text/html")
        ctx = LadderContext()

        assert await rungs._rendered_rung(_URL, walled, {}, ctx) is None
        assert await rungs._impersonate_rung(_URL, refused, host_sems={}, ctx=ctx) is None

        assert seen == [(landed, _URL, "rendering"), (landed, _URL, "re-dialing")]
        assert renders == []
        assert dials == []
        assert ctx.rungs == []


class TestLocalPdfReading:
    """A cited PDF used to be the one resolution source dropped unread. Now it is read with
    pypdf and rendered as a query-ranked digest — free, deterministic, no second request."""

    _PAGES: ClassVar[list[list[str]]] = [
        ["Annual Surveillance Summary", "Contents: methods, tables, appendix"],
        ["Hospitalizations reported: 922", "Deaths reported: 2"],
    ]

    def _session(self, *, content_type: str = "application/pdf", pages: list[list[str]] | None = None) -> FakeSession:
        body = build_text_pdf(self._PAGES if pages is None else pages)
        return FakeSession(
            {"https://cdc.example.com/report.pdf": FakeResponse(200, body=body, content_type=content_type)}
        )

    async def test_a_cited_pdf_is_read_and_the_relevant_passage_rendered(self):
        session = self._session()

        result = await _fetch_one(
            session, "https://cdc.example.com/report.pdf", {}, LadderContext(query="hospitalizations reported")
        )

        assert result.status == "success"
        assert result.route == "pdf_local"
        assert "922" in result.text
        assert "[p.2]" in result.text, "the passage carries the page a forecaster could cite"
        assert result.text.startswith("Document: https://cdc.example.com/report.pdf")
        assert [(a.rung, a.from_status) for a in result.rung_attempts] == [("pdf_local", "unsupported_type")]
        assert result.rung_attempts[0].wall_s is not None

    async def test_an_undeclared_document_is_sniffed_by_its_magic_bytes(self):
        """Several government hosts serve their PDFs as `application/octet-stream`, and the
        header is exactly what the old branch trusted when it dropped them."""
        session = self._session(content_type="application/octet-stream")

        result = await _fetch_one(
            session, "https://cdc.example.com/report.pdf", {}, LadderContext(query="deaths reported")
        )

        assert result.status == "success"
        assert result.route == "pdf_local"
        assert "Deaths reported: 2" in result.text

    async def test_a_pdf_with_no_text_layer_is_unreadable_rather_than_unsupported(self):
        """The two statuses answer different questions: `unsupported_type` is a type we do
        not read, `unreadable_document` is bytes we read and could not turn into text. Only
        the second is worth a paid document read later, so they must not be the same token."""
        session = self._session(pages=[["1"]])

        result = await _fetch_one(session, "https://cdc.example.com/report.pdf", {}, LadderContext(query="anything"))

        assert result.status == "unreadable_document"
        assert result.status_reason == "no_text_layer"
        assert result.text == ""
        assert result.route == "pdf_local", "the rung ran; it just found no text"

    async def test_a_malformed_pdf_says_so(self):
        session = FakeSession(
            {
                "https://cdc.example.com/report.pdf": FakeResponse(
                    200, body=b"%PDF-1.4\nthis is not really a pdf", content_type="application/pdf"
                )
            }
        )

        result = await _fetch_one(session, "https://cdc.example.com/report.pdf", {}, LadderContext(query="anything"))

        assert result.status == "unreadable_document"
        assert result.status_reason == "malformed"

    async def test_the_digest_is_bounded_by_the_per_url_cap(self, monkeypatch):
        cap = 300
        session = self._session(pages=[["Hospitalizations reported: 922 " * 20]])

        result = await _fetch_one(
            session, "https://cdc.example.com/report.pdf", {}, capped_ctx(cap, query="hospitalizations")
        )

        assert result.status == "success"
        assert len(result.text) <= cap
        assert "digest truncated at" in result.text

    async def test_a_declared_document_gets_the_document_byte_cap_and_an_undeclared_one_does_not(self, monkeypatch):
        """The 6.7 MB receipt PDF is over the 5 MiB response cap the text branches use, so a
        DECLARED document gets the document cap. An undeclared body keeps the smaller one:
        it is far likelier to be an image than a report, and buffering 40 MiB of it per URL
        across every concurrent question buys nothing."""
        monkeypatch.setattr(classify, "RESOLUTION_SOURCE_MAX_RESPONSE_BYTES", 200)
        monkeypatch.setattr(classify, "DOCUMENT_TEXT_PDF_MAX_BYTES", 10_000_000)

        declared = await _fetch_one(
            self._session(), "https://cdc.example.com/report.pdf", {}, LadderContext(query="hospitalizations")
        )
        undeclared = await _fetch_one(
            self._session(content_type="application/octet-stream"),
            "https://cdc.example.com/report.pdf",
            {},
            LadderContext(query="hospitalizations"),
        )

        assert declared.status == "success"
        assert undeclared.status == "error", "oversize under the general cap, as before"

    async def test_an_oversize_document_is_dropped(self, monkeypatch):
        monkeypatch.setattr(classify, "DOCUMENT_TEXT_PDF_MAX_BYTES", 100)

        result = await _fetch_one(
            self._session(), "https://cdc.example.com/report.pdf", {}, LadderContext(query="hospitalizations")
        )

        assert result.status == "error"
        assert result.text == ""

    async def test_the_read_is_skipped_when_the_wall_budget_is_spent(self, caplog):
        """The parse is CPU-bound and the outer `wait_for` throws away finished work, so with
        no budget left the document is left unread rather than risking the whole question."""
        spent = LadderContext(
            query="hospitalizations", started=time.monotonic() - resolution_source.RESOLUTION_SOURCE_WALL_TIMEOUT
        )

        with caplog.at_level("WARNING", logger="metaculus_bot.research.fetch_ladder.context"):
            result = await _fetch_one(self._session(), "https://cdc.example.com/report.pdf", {}, spent)

        assert result.status == "unsupported_type"
        assert result.route == "direct"
        assert [a.skipped_reason for a in result.rung_attempts] == ["wall_budget"]
        assert any("skipping the local PDF read" in m for m in caplog.messages)
        assert result.status_reason == "budget_skipped", (
            "without the reason, a document we held and declined to read is byte-identical "
            "in the per-fetch marker to a body that was never a document"
        )

    async def test_a_query_no_passage_matches_withholds_the_document(self):
        """A document read END TO END that discusses nothing the question asks about is
        content-free, so it is withheld like any other 200 that carried nothing. Published as
        `success` its block was the header, the outline and one sentence saying nothing matched
        — prose standing in for an absent section under the "primary grading evidence" caption,
        and byte-identical in the run log to a document carrying the resolving paragraph."""
        session = self._session()

        result = await _fetch_one(
            session, "https://cdc.example.com/report.pdf", {}, LadderContext(query="corn futures settlement")
        )

        assert result.status == "no_resolving_content"
        assert result.status_reason == "no_matching_passage"
        assert result.text == ""
        # The rung still fired and owns the outcome, so "what did the ladder buy" stays a query, not a direct fetch.
        assert result.route == "pdf_local"

    async def test_a_withheld_document_is_named_in_the_failure_notice(self):
        """The forecaster-facing half: withholding must not be silent. The domain and its status
        go into the all-failed notice, which is the same treatment a chrome-only page gets."""
        session = self._session()

        result = await _fetch_one(
            session, "https://cdc.example.com/report.pdf", {}, LadderContext(query="corn futures settlement")
        )
        rendered = format_resolution_sections([result], datetime(2026, 9, 3, tzinfo=UTC))

        assert "1 resolution source(s) yielded no usable content" in rendered
        assert "cdc.example.com: no_resolving_content" in rendered
        # Nothing from the document is published, so neither is the caveat describing how a published PDF is read.
        assert "No passage in this document matched the query" not in rendered
        assert "query-relevant passages of a cited PDF" not in rendered

    async def test_a_matched_digest_carries_no_reason(self):
        """The reason is the exception, so its ABSENCE has to mean the selection found
        something rather than 'this record predates the token'."""
        session = self._session()

        result = await _fetch_one(
            session, "https://cdc.example.com/report.pdf", {}, LadderContext(query="hospitalizations reported")
        )

        assert result.status == "success"
        assert result.status_reason is None


class TestPdfParseGate:
    """At most two documents parse at once, loop-wide.

    pypdf decodes every content stream, so a parse is CPU-bound and holds its body for the
    duration. A Tier-1 fan-out alone is up to RESOLUTION_SOURCE_MAX_URLS documents per
    question across DEFAULT_MAX_CONCURRENT_RESEARCH questions, and the same gate is shared
    with the gap-fill v2 local-document ladder because the bound has to hold across both."""

    _PAGES: ClassVar[list[list[str]]] = [
        ["Annual Surveillance Summary", "Contents: methods, tables, appendix"],
        ["Hospitalizations reported: 922", "Deaths reported: 2"],
    ]

    def _session(self, url: str) -> FakeSession:
        return FakeSession({url: FakeResponse(200, body=build_text_pdf(self._PAGES), content_type="application/pdf")})

    async def test_the_host_gate_is_released_before_the_parse_runs(self, monkeypatch):
        """The per-host gate is loop-wide, so a parse held inside it blocks every other
        concurrent question's fetch of that host for the whole parse — and this population
        is a handful of government hosts, so same-host collisions are the expected case.

        The host gate is fetched get-or-create up front because the stub below reads `.locked()` off that
        object rather than looking the map up: it runs in the parse's worker thread, where there is no running
        loop, which is itself the point of the split.
        """
        url = "https://cdc.example.com/report.pdf"
        sems = host_semaphores()
        host_gate = semaphore_for_host(url, sems)
        locked_during_parse = None
        real_extract = classify.extract_pdf_text

        def observing_extract(body: bytes, **kwargs):
            nonlocal locked_during_parse
            locked_during_parse = host_gate.locked()
            return real_extract(body, **kwargs)

        monkeypatch.setattr(classify, "extract_pdf_text", observing_extract)

        result = await _fetch_one(self._session(url), url, sems, LadderContext(query="hospitalizations"))

        assert locked_during_parse is False, "the parse ran while the host was still gated"
        # The rung's own telemetry is unchanged by moving where the parse happens.
        assert result.status == "success"
        assert result.route == "pdf_local"
        assert [(a.rung, a.from_status) for a in result.rung_attempts] == [("pdf_local", "unsupported_type")]
        assert result.rung_attempts[0].wall_s is not None

    async def test_no_more_than_two_documents_parse_at_once(self, monkeypatch):
        in_flight = 0
        peak = 0
        real_extract = classify.extract_pdf_text

        def counting_extract(body: bytes, **kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                time.sleep(0.05)
                return real_extract(body, **kwargs)
            finally:
                in_flight -= 1

        monkeypatch.setattr(classify, "extract_pdf_text", counting_extract)
        hosts = [f"https://h{i}.example.com/report.pdf" for i in range(5)]

        results = await asyncio.gather(
            *(_fetch_one(self._session(url), url, {}, LadderContext(query="hospitalizations")) for url in hosts)
        )

        assert peak <= 2, f"the gate admitted {peak} concurrent parses"
        assert [r.status for r in results] == ["success"] * 5, "the bound queues parses, it does not drop them"

    async def test_a_parse_that_cannot_win_a_slot_inside_the_budget_is_skipped(self, monkeypatch, caplog):
        """Queueing behind other documents until the outer wall fires would discard every
        sibling page this question already fetched, which costs more than one unread doc."""
        gate = None

        async def hold_the_gate():
            nonlocal gate
            gate = pdf_parse_semaphore()
            await gate.acquire()
            await gate.acquire()

        await hold_the_gate()
        # Budget just above the floor, so the bounded acquire gives up almost immediately.
        spent = LadderContext(
            query="hospitalizations",
            started=time.monotonic()
            - (
                resolution_source.RESOLUTION_SOURCE_WALL_TIMEOUT
                - RESOLUTION_SOURCE_POLICY.rung_wall_margin_s
                - classify.RESOLUTION_SOURCE_PDF_MIN_BUDGET_S
                - 0.05
            ),
        )
        url = "https://busy.example.com/report.pdf"

        with caplog.at_level("WARNING", logger="metaculus_bot.research.fetch_ladder.classify"):
            result = await _fetch_one(self._session(url), url, {}, spent)

        assert result.status == "unsupported_type"
        assert result.status_reason == "parse_contention"
        assert [a.skipped_reason for a in result.rung_attempts] == ["parse_contention"]
        assert any("no parse slot" in m for m in caplog.messages)
        assert _rung_counts([result])["pdf_contention_skips"] == 1
        assert gate is not None
        gate.release()
        gate.release()


class TestSharedHostGate:
    """Politeness has to hold ACROSS questions, not just inside one.

    The map used to be rebuilt per provider call, so six questions citing the same host each
    held their own `Semaphore(1)` and hit it six times at once — the opposite of what the
    semaphore is for, on exactly the government hosts that answer our requests with 403."""

    async def test_two_concurrent_provider_calls_serialize_on_one_host(self, article_html, monkeypatch):
        class SlowReadResponse(FakeResponse):
            async def read(self) -> bytes:
                await asyncio.sleep(0.01)
                return self._body

        session = FakeSession(
            {
                "https://a.example.com/one": SlowReadResponse(200, body=article_html, content_type="text/html"),
                "https://a.example.com/two": SlowReadResponse(200, body=article_html, content_type="text/html"),
            }
        )
        monkeypatch.setattr(guard, "_get_session", lambda: session)

        await asyncio.gather(
            fetch_resolution_sources(["https://a.example.com/one"]),
            fetch_resolution_sources(["https://a.example.com/two"]),
        )

        assert session.host_peak["a.example.com"] == 1
        assert list(host_semaphores()) == ["a.example.com"], "one gate for the host, not one per call"


class TestPerHopRequestTimeout:
    """Every GET is bounded by what is left of the provider's 45 s wall, not by a flat 20 s.

    A hop admitted with a few seconds of budget left used to be free to run the session's
    full `RESOLUTION_SOURCE_HTTP_TIMEOUT`, overshoot the outer `wait_for` and take down every
    sibling page that had already fetched — the exact loss the rest of the ladder's budget
    arithmetic exists to prevent."""

    def _session(self, article_html: bytes) -> FakeSession:
        return FakeSession(
            {"https://slow.example.com/page": FakeResponse(200, body=article_html, content_type="text/html")}
        )

    async def test_a_fresh_fetch_still_gets_the_full_per_request_timeout(self, article_html):
        session = self._session(article_html)

        await _fetch_one(session, "https://slow.example.com/page", {}, LadderContext())

        assert session.get_kwargs[0]["timeout"].total == direct_fetch.RESOLUTION_SOURCE_HTTP_TIMEOUT

    async def test_a_hop_late_in_the_wall_is_clamped_to_the_remaining_budget(self, article_html):
        session = self._session(article_html)
        elapsed = 35.0
        expected = (
            resolution_source.RESOLUTION_SOURCE_WALL_TIMEOUT - elapsed - RESOLUTION_SOURCE_POLICY.rung_wall_margin_s
        )

        await _fetch_one(
            session, "https://slow.example.com/page", {}, LadderContext(started=time.monotonic() - elapsed)
        )

        timeout = session.get_kwargs[0]["timeout"]
        assert timeout.total == pytest.approx(expected, abs=0.5)
        assert timeout.total < direct_fetch.RESOLUTION_SOURCE_HTTP_TIMEOUT
        assert timeout.sock_read == timeout.total, "a per-request ClientTimeout replaces the session's, so both fields"

    async def test_a_spent_budget_still_gets_a_token_attempt_rather_than_a_zero_timeout(self, article_html):
        """A guaranteed-expired request tells us nothing a 0.5 s one does not, and a fast host
        answering inside the floor is a page we would otherwise refuse for free."""
        session = self._session(article_html)

        result = await _fetch_one(
            session,
            "https://slow.example.com/page",
            {},
            LadderContext(started=time.monotonic() - 2 * resolution_source.RESOLUTION_SOURCE_WALL_TIMEOUT),
        )

        assert session.get_kwargs[0]["timeout"].total == direct_fetch.RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S
        assert result.status == "success", "the floor is a real attempt, not a formality"
