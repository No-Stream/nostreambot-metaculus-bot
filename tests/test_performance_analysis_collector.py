"""Tests for the Metaculus collector: its HTTP retry, its two pulls, and per-record processing.

Covers ``_api_get``'s shared retry budget, the paged resolved-post pull, the public-plus-private
comment pull, and the fields ``_process_post`` stamps onto a record (the comment timestamp, the
crowd size, the stacker outcome and its provenance, the practice-post and date-question skips).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import requests

from metaculus_bot.performance_analysis import collector
from metaculus_bot.performance_analysis.collector import _process_post
from tests.performance_analysis_fakes import _binary_post, _FakeResponse


class TestApiGetRetry:
    """``_api_get`` retries a transient network failure, and its wall clock stays bounded.

    A read timeout used to propagate on its first occurrence, so one slow page abandoned a
    whole tournament sweep and wrote nothing after several minutes of paging. Nine residual
    rounds worked around it with a pasted ``resilient_pull.py``. The retry budget is SHARED
    with the 429 budget on purpose: the worst-case wall clock is then identical to the
    429-only retry this replaces, which is what makes the change strictly safer in a path
    whose overrun costs forecasts.
    """

    def _install(self, monkeypatch, outcomes: list[object]) -> tuple[list[float], list[dict]]:
        """Serve ``outcomes`` (a response or an exception instance) in order; record every sleep."""
        sleeps: list[float] = []
        calls: list[dict] = []
        remaining = list(outcomes)

        def fake_get(url: str, headers: dict | None = None, params: dict | None = None, timeout: float | None = None):
            calls.append({"url": url, "headers": dict(headers or {}), "timeout": timeout})
            outcome = remaining.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(collector.requests, "get", fake_get)
        monkeypatch.setattr(collector.time, "sleep", fake_sleep)
        return sleeps, calls

    def test_a_read_timeout_is_retried_and_the_next_attempt_wins(self, monkeypatch):
        sleeps, calls = self._install(
            monkeypatch,
            [requests.exceptions.ReadTimeout("slow page"), _FakeResponse(200, {"results": [{"id": 7}]})],
        )

        assert collector._api_get("/posts/", "token") == {"results": [{"id": 7}]}
        assert len(calls) == 2
        assert sleeps == [collector.RETRY_BACKOFF_SECS]

    def test_a_connection_error_is_retried_too(self, monkeypatch):
        self._install(
            monkeypatch, [requests.exceptions.ConnectionError("reset by peer"), _FakeResponse(200, {"results": []})]
        )

        assert collector._api_get("/comments/", "token") == {"results": []}

    def test_exhausted_transient_retries_raise_the_underlying_requests_error(self, monkeypatch):
        timeout = requests.exceptions.ReadTimeout("still slow")
        sleeps, calls = self._install(monkeypatch, [timeout] * collector.MAX_RETRIES)

        with pytest.raises(requests.exceptions.ReadTimeout):
            collector._api_get("/posts/", "token")

        # Every attempt is spent, and the last one does not sleep before giving up.
        assert len(calls) == collector.MAX_RETRIES
        assert len(sleeps) == collector.MAX_RETRIES - 1

    def test_the_transient_and_429_budgets_are_shared_so_the_attempt_ceiling_is_unchanged(self, monkeypatch):
        outcomes: list[object] = [requests.exceptions.ReadTimeout("slow"), _FakeResponse(429), _FakeResponse(429)]
        assert len(outcomes) == collector.MAX_RETRIES, "this pin assumes the three-attempt budget"
        sleeps, calls = self._install(monkeypatch, outcomes)

        with pytest.raises(requests.exceptions.RequestException):
            collector._api_get("/posts/", "token")

        assert len(calls) == collector.MAX_RETRIES
        # Worst-case wall clock: MAX_RETRIES reads at the request timeout, plus these backoffs.
        assert sleeps == [collector.RETRY_BACKOFF_SECS, collector.RETRY_BACKOFF_SECS * 2]

    def test_an_exhausted_429_names_the_rate_limit_rather_than_reporting_one_unlucky_request(self, monkeypatch):
        self._install(monkeypatch, [_FakeResponse(429)] * collector.MAX_RETRIES)

        with pytest.raises(requests.exceptions.HTTPError, match="429 rate limit"):
            collector._api_get("/posts/", "token")

    def test_a_server_error_is_not_retried(self, monkeypatch):
        _sleeps, calls = self._install(monkeypatch, [_FakeResponse(500)])

        with pytest.raises(requests.exceptions.HTTPError):
            collector._api_get("/posts/", "token")

        assert len(calls) == 1

    def test_every_attempt_carries_the_bounded_request_timeout(self, monkeypatch):
        _sleeps, calls = self._install(
            monkeypatch, [requests.exceptions.ReadTimeout("slow"), _FakeResponse(200, {"results": []})]
        )

        collector._api_get("/posts/", "token")

        assert [call["timeout"] for call in calls] == [collector.REQUEST_TIMEOUT_SECS] * 2


class TestFetchResolvedQuestions:
    """The scoring pull is the list pages and nothing else.

    Under ``with_cp=true`` a list page's question dict carries everything the collector
    reads: type, resolution, scaling, the open bounds, the timestamps and the token's own
    ``my_forecasts`` with ``forecast_values`` and ``score_data``, on a single-question post
    and on a group post's members alike (verified live 2026-09-09 against the detail
    payload, where only post-level keys the collector never reads differ). The pull used to
    page the list for ids and then GET every post one at a time behind a 0.5 s sleep, about
    99 s per hundred posts against 1.8 s for the page that already held them all. The
    pins: every page asks for ``with_cp``, the payloads come back as the list served them,
    and no per-post GET is ever issued.
    """

    def _install_pages(self, monkeypatch, pages: list[list[dict]]) -> list[dict]:
        seen: list[dict] = []
        remaining = list(pages)

        def fake_api_get(path: str, token: str, params: dict | None = None) -> dict:
            seen.append({"path": path, "token": token, "params": dict(params or {})})
            page = remaining.pop(0)
            return {"results": page, "next": "next-page" if remaining else None}

        monkeypatch.setattr(collector, "_api_get", fake_api_get)
        monkeypatch.setattr(collector, "FETCH_DELAY_SECS", 0.0)
        return seen

    def test_returns_the_list_payloads_across_pages_with_no_per_post_get(self, monkeypatch):
        first_page = [_binary_post(1, 11), _binary_post(2, 22)]
        second_page = [_binary_post(3, 33)]
        seen = self._install_pages(monkeypatch, [first_page, second_page])

        posts = collector.fetch_resolved_questions("spring-aib-2026", "token")

        assert posts == first_page + second_page
        assert [call["path"] for call in seen] == ["/posts/", "/posts/"]
        assert [call["params"]["offset"] for call in seen] == [0, collector.PAGE_SIZE]

    def test_every_list_page_asks_for_the_tokens_own_forecasts(self, monkeypatch):
        seen = self._install_pages(monkeypatch, [[_binary_post(1, 11)], [_binary_post(2, 22)]])

        collector.fetch_resolved_questions("spring-aib-2026", "personal-token")

        assert len(seen) == 2
        for call in seen:
            assert call["params"]["with_cp"] == "true"
            assert call["params"]["tournaments"] == "spring-aib-2026"
            assert call["params"]["statuses"] == "resolved"
            assert call["params"]["limit"] == collector.PAGE_SIZE
            assert call["token"] == "personal-token"

    def test_an_empty_tournament_is_one_page_and_no_posts(self, monkeypatch):
        seen = self._install_pages(monkeypatch, [[]])

        assert collector.fetch_resolved_questions("spring-aib-2026", "token") == []
        assert len(seen) == 1


class TestFetchBotComments:
    """The comment pull has to list the private comments as well as the public ones.

    The bot POSTs every comment with ``is_private: true`` and Metaculus flips older ones
    public server-side, so the default author listing served the 1,054 summer comments and
    none of the six fall ones (verified live 2026-09-09), which is a residual round whose
    records carry no comment text, no ``bot_comment_created_at`` and no per-model parse. The
    pins: both param sets are requested, the return is their union, a comment served by both
    appears once, and each listing still pages to exhaustion.
    """

    def _install_pages(
        self, monkeypatch, public_pages: list[list[dict]], private_pages: list[list[dict]]
    ) -> list[dict]:
        seen: list[dict] = []
        remaining = {False: list(public_pages), True: list(private_pages)}

        def fake_api_get(path: str, token: str, params: dict | None = None) -> dict:
            call_params = dict(params or {})
            seen.append({"path": path, "token": token, "params": call_params})
            pages = remaining[bool(call_params.get("is_private"))]
            page = pages.pop(0)
            return {"results": page, "next": "next-page" if pages else None}

        monkeypatch.setattr(collector, "_api_get", fake_api_get)
        monkeypatch.setattr(collector, "FETCH_DELAY_SECS", 0.0)
        return seen

    @staticmethod
    def _comment(comment_id: int, post_id: int) -> dict:
        return {"id": comment_id, "on_post": post_id, "text": f"*Forecaster 1*: {comment_id}%\n"}

    def test_returns_the_union_of_the_public_and_private_listings(self, monkeypatch):
        public = self._comment(1, 11)
        private = self._comment(2, 22)
        seen = self._install_pages(monkeypatch, [[public]], [[private]])

        comments = collector.fetch_bot_comments(275109, "bot-token")

        assert comments == [public, private]
        assert [call["path"] for call in seen] == ["/comments/", "/comments/"]
        assert [call["params"].get("is_private") for call in seen] == [None, "true"]
        for call in seen:
            assert call["params"]["author"] == 275109
            assert call["params"]["limit"] == collector.PAGE_SIZE
            assert call["token"] == "bot-token"

    def test_a_comment_served_by_both_listings_appears_once(self, monkeypatch):
        """Metaculus flips comments public in place, so the two listings overlap during the
        flip and a duplicate would give one post two records of the same forecast."""
        flipped = self._comment(7, 77)
        self._install_pages(monkeypatch, [[flipped, self._comment(8, 88)]], [[flipped]])

        comments = collector.fetch_bot_comments(275109, "bot-token")

        assert [c["id"] for c in comments] == [7, 8]

    def test_each_listing_pages_to_exhaustion(self, monkeypatch):
        public_pages = [[self._comment(1, 11)], [self._comment(2, 22)]]
        private_pages = [[self._comment(3, 33)], [self._comment(4, 44)], [self._comment(5, 55)]]
        seen = self._install_pages(monkeypatch, public_pages, private_pages)

        comments = collector.fetch_bot_comments(275109, "bot-token")

        assert [c["id"] for c in comments] == [1, 2, 3, 4, 5]
        public_offsets = [c["params"]["offset"] for c in seen if c["params"].get("is_private") is None]
        private_offsets = [c["params"]["offset"] for c in seen if c["params"].get("is_private") == "true"]
        assert public_offsets == [0, collector.PAGE_SIZE]
        assert private_offsets == [0, collector.PAGE_SIZE, 2 * collector.PAGE_SIZE]

    def test_an_author_with_no_comments_is_one_page_per_listing(self, monkeypatch):
        seen = self._install_pages(monkeypatch, [[]], [[]])

        assert collector.fetch_bot_comments(275109, "bot-token") == []
        assert len(seen) == 2


class TestTruncatedCommentRecovery:
    """List stubs must be replaced with the full text before record parsing."""

    @staticmethod
    def _payloads() -> dict:
        fixture_path = Path(__file__).parent / "data" / "truncated_comment_payloads.json"
        return json.loads(fixture_path.read_text())

    def test_lookup_fetches_and_uses_full_comment_payload_once(self, monkeypatch):
        payloads = self._payloads()
        list_comment = payloads["list_comment"]
        detail_comment = payloads["detail_comment"]
        calls: list[str] = []

        def fake_api_get(path: str, token: str, params: dict | None = None) -> dict:
            calls.append(path)
            assert token == "bot-token"
            return detail_comment

        monkeypatch.setattr(collector, "_api_get", fake_api_get)

        lookup = collector._build_comment_lookup([list_comment], "bot-token")
        records = _process_post(_binary_post(list_comment["on_post"], 45859), lookup)

        assert calls == [f"/comments/{list_comment['id']}/"]
        assert records[0]["comment_text"] == detail_comment["text"]
        assert records[0]["per_model_forecasts"]
        assert records[0]["forecasters_used"] == 1
        assert records[0]["forecasters_configured"] == 3

    def test_truncated_comment_without_full_text_raises(self, monkeypatch):
        list_comment = self._payloads()["list_comment"]
        monkeypatch.setattr(collector, "_api_get", lambda path, token, params=None: {"id": list_comment["id"]})

        with pytest.raises(ValueError, match=r"full comment .* has no text"):
            collector._build_comment_lookup([list_comment], "bot-token")

    def test_recognizes_stub_with_a_markdown_link_target(self, monkeypatch):
        comment = self._payloads()["list_comment_with_markdown_link"]
        monkeypatch.setattr(
            collector,
            "_api_get",
            lambda path, token, params=None: {"id": comment["id"], "text": "full comment text"},
        )

        assert collector.expand_truncated_comment(comment, "bot-token")["text"] == "full comment text"


class TestCollectorCommentCreatedAt:
    """Records produced by the collector should surface the comment's
    ``created_at`` timestamp so cohort cuts can filter by submit-date (vs the
    coarser actual_resolve_time on the question)."""

    def test_record_includes_bot_comment_created_at(self):
        post = _binary_post(1, 11)
        comment = {
            "id": 999,
            "text": "*Forecaster 1*: 70%\n",
            "on_post": 1,
            "created_at": "2026-04-30T12:34:56Z",
        }
        records = _process_post(post, {1: comment})
        assert len(records) == 1
        assert records[0]["bot_comment_created_at"] == "2026-04-30T12:34:56Z"

    def test_record_has_none_when_comment_missing(self):
        post = _binary_post(2, 22)
        records = _process_post(post, {})
        assert len(records) == 1
        assert records[0]["bot_comment_created_at"] is None

    def test_crowd_size_is_read_off_the_post_not_the_question(self):
        """``nr_forecasters`` is a POST field; reading it off the question dict with a 0
        default made it read 0 in all 2196 archived records — "never read", rendered as a
        measured empty crowd. This fixture deliberately keeps the decoy on the question."""
        post = _binary_post(5, 55)
        post["nr_forecasters"] = 170
        records = _process_post(post, {})

        assert records[0]["metadata"]["nr_forecasters"] == 170

    def test_a_post_with_no_crowd_field_reads_none_not_zero(self):
        """None means "the post didn't say", which a crowd-size cut can drop.

        A 0 would average a fabricated empty crowd into the cut, and would also silently kill
        audit.py's ``n/a`` fallback, since a real 0 is not a missing key.
        """
        post = _binary_post(6, 66)
        post.pop("nr_forecasters", None)
        records = _process_post(post, {})

        assert records[0]["metadata"]["nr_forecasters"] is None

    def test_record_has_none_when_comment_lacks_created_at(self):
        post = _binary_post(3, 33)
        comment = {"id": 1000, "text": "*Forecaster 1*: 70%\n", "on_post": 3}
        records = _process_post(post, {3: comment})
        assert len(records) == 1
        assert records[0]["bot_comment_created_at"] is None

    def test_stacker_skip_reason_marker_round_trips_onto_record(self):
        """The additive STACKER_SKIP_REASON marker must reach the record dict, because its
        documented durable path is the published comment rather than the run log."""
        post = _binary_post(4, 44)
        comment = {
            "id": 1001,
            "text": "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=skipped -->\n<!-- STACKER_SKIP_REASON=single_forecaster -->\n",
            "on_post": 4,
        }
        records = _process_post(post, {4: comment})
        assert len(records) == 1
        assert records[0]["stacker_skip_reason"] == "single_forecaster"
        assert records[0]["stacker_outcome"] == "skipped"

    def test_stacker_skip_reason_none_without_marker(self):
        post = _binary_post(5, 55)
        comment = {"id": 1002, "text": "*Forecaster 1*: 70%\n", "on_post": 5}
        records = _process_post(post, {5: comment})
        assert records[0]["stacker_skip_reason"] is None

    def test_practice_posts_produce_no_records(self):
        """Practice questions are not tournament scoring surface, so they must never enter the
        dataset; they would otherwise land in every calibration cut."""
        post = _binary_post(6, 66)
        post["title"] = "[PRACTICE] Will this be scored?"
        comment = {"id": 1003, "text": "*Forecaster 1*: 70%\n", "on_post": 6}
        assert _process_post(post, {6: comment}) == []


class TestCollectorExcludesDateQuestions:
    """The live bot forecasts date questions (on the epoch-seconds axis), so a resolved one
    reaches the collector; the residual dataset stays date-free by decision, the same exclusion
    ``backtest/question_prep.py`` and ``ablation/run_pdf.py`` carry at their seams. The skip has
    to be explicit and named in the log: ``parse_resolution``'s unknown-type fallthrough would
    file the same question as a parser bug."""

    @pytest.mark.parametrize("resolution", ["above_upper_bound", "2026-07-20T12:00:00Z"])
    def test_a_resolved_date_question_is_skipped_by_type_with_one_named_warning(self, resolution, caplog):
        """The fixture is the wire shape of a resolved date question.

        Taken from ``tests/data/mantic_series1_date_post_500_2026_09_08.json``: ``type`` is
        ``date``, the ``scaling`` bounds are epoch seconds, and the resolution is either an
        out-of-range token or an ISO timestamp.
        """
        post = _binary_post(500, 5000, resolution=resolution)
        post["question"]["type"] = "date"
        post["question"]["open_upper_bound"] = True
        post["question"]["scaling"] = {"range_min": 1781708400.0, "range_max": 1786536000.0, "zero_point": None}
        post["question"]["my_forecasts"]["latest"]["forecast_values"] = [0.0, 0.4, 0.9]
        comment = {"id": 1004, "text": "*Forecaster 1*: 2026-07-20\n", "on_post": 500}

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.performance_analysis"):
            assert _process_post(post, {500: comment}) == []

        warnings = [(r.name, r.getMessage()) for r in caplog.records if r.levelno == logging.WARNING]
        (message,) = [text for name, text in warnings if name.endswith(".collector")]
        assert "Q5000" in message
        assert "date question" in message
        assert not any("Unknown question type" in text for _, text in warnings)


class TestCollectorStackerOutcome:
    """Records produced by the collector should expose the tri-state
    ``stacker_outcome`` plus its provenance, computed from
    ``parse_inferred_stacker_outcome`` over the comment text. The legacy
    ``was_stacked`` field collapses median-fallback into False, so analyses
    that need to distinguish "stacker LLM ran" from "MEDIAN fallback" must
    consume ``stacker_outcome``.
    """

    def _run(self, post_id: int, question_id: int, comment_text: str | None) -> dict:
        post = _binary_post(post_id, question_id)
        if comment_text is None:
            records = _process_post(post, {})
        else:
            records = _process_post(post, {post_id: {"id": 999, "text": comment_text, "on_post": post_id}})
        assert len(records) == 1
        return records[0]

    def test_outcome_marker_primary(self):
        rec = self._run(1, 11, "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=primary -->\n")
        assert rec["stacker_outcome"] == "primary"
        assert rec["stacker_outcome_source"] == "marker_outcome"

    def test_outcome_marker_fallback_median_distinguished_from_skipped(self):
        """The load-bearing case: this used to round-trip as ``STACKED=true`` and
        ``was_stacked=True``, with no way to tell median-fallback from primary. The richer
        ``stacker_outcome="fallback_median"`` is now preserved on the record."""
        rec = self._run(2, 22, "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=fallback_median -->\n")
        assert rec["stacker_outcome"] == "fallback_median"
        assert rec["stacker_outcome_source"] == "marker_outcome"

    def test_outcome_marker_skipped(self):
        rec = self._run(3, 33, "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=skipped -->\n")
        assert rec["stacker_outcome"] == "skipped"
        assert rec["stacker_outcome_source"] == "marker_outcome"

    def test_outcome_marker_skipped_config_off(self):
        """A config-suppressed skip (the per-type gate off despite high spread) must survive the
        collector round-trip distinct from a plain "skipped": this is the field the
        0-of-22-numeric-suppression re-attribution needed."""
        rec = self._run(3, 33, "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=skipped_config_off -->\n")
        assert rec["stacker_outcome"] == "skipped_config_off"
        assert rec["stacker_outcome_source"] == "marker_outcome"

    def test_legacy_marker_only_maps_to_primary(self):
        rec = self._run(4, 44, "*Forecaster 1*: 70%\n<!-- STACKED=true -->\n")
        assert rec["stacker_outcome"] == "primary"
        assert rec["stacker_outcome_source"] == "marker_legacy"

    def test_legacy_marker_false_maps_to_skipped(self):
        rec = self._run(5, 55, "*Forecaster 1*: 70%\n<!-- STACKED=false -->\n")
        assert rec["stacker_outcome"] == "skipped"
        assert rec["stacker_outcome_source"] == "marker_legacy"

    def test_historical_body_inferred_primary(self):
        """A pre-marker comment from the spring-aib-2026 dataset: no ``STACKED=`` or
        ``STACKER_OUTCOME=`` marker, but the Forecaster 1 body opens with
        "## Stacker Meta-Analysis", which only the stacker pipeline produces."""
        comment = (
            "# SUMMARY\n"
            "*Forecaster 1*: 70%\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "## Stacker Meta-Analysis\n\n"
            "Synthesis of 6 base models below.\n"
        )
        rec = self._run(6, 66, comment)
        assert rec["stacker_outcome"] == "primary"
        assert rec["stacker_outcome_source"] == "historical_body"

    def test_no_signal_returns_none(self):
        rec = self._run(7, 77, "*Forecaster 1*: 70%\n")
        assert rec["stacker_outcome"] is None
        assert rec["stacker_outcome_source"] == "none"

    def test_missing_comment_returns_none(self):
        rec = self._run(8, 88, None)
        assert rec["stacker_outcome"] is None
        assert rec["stacker_outcome_source"] == "none"

    def test_outcome_marker_takes_precedence_over_legacy(self):
        """Both markers coexist for one round of back-compat, so the collector must prefer the
        richer ``STACKER_OUTCOME=`` signal, or median-fallback is silently downgraded to
        "primary"."""
        comment = "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=fallback_median -->\n<!-- STACKED=false -->\n"
        rec = self._run(9, 99, comment)
        assert rec["stacker_outcome"] == "fallback_median"
        assert rec["stacker_outcome_source"] == "marker_outcome"
