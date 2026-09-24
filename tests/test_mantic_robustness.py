"""The Mantic client's robustness rules from the 2026-09-08 readiness review.

Three rules, one class each, all in ``metaculus_bot/mantic.py``:

- **Pagination** (review item 21). The framework's default tournament fetch reads ONE page of 100
  and stops, and Mantic's paginator advertises a ``next`` link past the last page, so only walking
  offsets until an empty page can prove there is nothing more. The fetch passes a ceiling
  (``MANTIC_FETCH_QUESTION_CEILING``) with ``error_if_question_target_missed=False``.
- **Parse drops** (item 12). A post the framework cannot parse is caught by its per-post loop, logged
  as one warning and forfeited silently on every run. The client counts the drop and emits one
  ``MANTIC_POST_DROPPED`` line before re-raising, keeping fail-fast; cli reads the counter into the
  alertable arithmetic (pinned in ``tests/cli/test_cli_exit_status.py``).
- **The tournament preflight** (items 5 and 20). Two authenticated GETs before any spend, neither
  retried: the tournament list logs ``MANTIC_TOURNAMENTS`` (new-season discovery), then the configured
  tournament's own route decides whether the token's ``user_permission`` allows forecasting. The
  detail route, because the list omits an ``unlisted`` project (a new season before its first
  question) while a slug no tournament has 404s there.

``tests/test_mantic_client.py`` owns the recorded preseason fixture and the parsing seams; this module
reuses its fixture loader and never opens a socket (the autouse egress guard in conftest would refuse).
The live tournament list below is the ``GET /api/projects/tournaments/`` of 2026-09-08 with Series 2
(opened 2026-09-23) as the ongoing one, reduced to the fields the preflight reads; the detail payload
carries the same fields and more. ``series-3`` stands in for a season the constants do not name yet.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests
from forecasting_tools import ApiFilter
from forecasting_tools.helpers import metaculus_client as ft_client

from metaculus_bot import mantic
from metaculus_bot.api_preflight import ApiIdentityError
from metaculus_bot.constants import MANTIC_API_BASE_URL, MANTIC_FETCH_QUESTION_CEILING, MANTIC_TOURNAMENT_ID
from metaculus_bot.mantic import (
    ManticClient,
    get_post_drop_count,
    preflight_mantic_tournaments,
    reset_post_drop_count,
)
from scripts.telemetry.markers import parse_log_text
from tests.http_fakes import json_response, text_response
from tests.mantic_fakes import BINARY_POST_ID, DISCRETE_POST_ID, load_preseason_posts

_FAKE_TOKEN = "f" * 40
_UNPACK = "unpack_subquestions"
_MANTIC_LOGGER = "metaculus_bot.mantic"
_TOURNAMENTS_URL = f"{MANTIC_API_BASE_URL}/projects/tournaments/"
_CONFIGURED_TOURNAMENT_URL = f"{_TOURNAMENTS_URL}{MANTIC_TOURNAMENT_ID}/"
# What both routes answer to a mistyped or revoked token, live: a 403, not a 401.
_INVALID_TOKEN_STATUS = 403
_INVALID_TOKEN_BODY = {"detail": "Invalid token."}
_NOT_FOUND_BODY = {"detail": "Not found."}
# Prod cli.py log format, as in tests/test_telemetry_markers.py; the level is irrelevant to the harvest.
_LOG_PREFIX = "2026-09-08 14:23:01,123 - metaculus_bot.mantic - ERROR - "
_HARVEST_META = {
    "run_id": "999",
    "workflow": "mantic",
    "artifact": "research-999",
    "run_date": "2026-09-08T14:00:00Z",
    "log_file": "run.log",
}


def _tournament(
    slug: str, *, is_ongoing: bool, user_permission: str | None = "forecaster", bots_only: bool = True
) -> dict[str, Any]:
    return {
        "slug": slug,
        "is_ongoing": is_ongoing,
        "user_permission": user_permission,
        "bot_leaderboard_status": "bots_only" if bots_only else "exclude_and_show",
    }


# The live list (module docstring): bots-only tournaments, the configured one the only ongoing one.
_NEXT_SEASON_SLUG = "series-3"
LIVE_TOURNAMENTS = [
    _tournament("preseason-2", is_ongoing=False),
    _tournament("series-1", is_ongoing=False),
    _tournament("practice-series-1", is_ongoing=False),
    _tournament(MANTIC_TOURNAMENT_ID, is_ongoing=True),
]


def _serve(monkeypatch: pytest.MonkeyPatch, *answers: requests.Response | Exception) -> MagicMock:
    """Answer successive ``requests.get`` calls with ``answers`` in order (an exception is raised);
    returns the spy so a test can read the requests."""
    fake_get = MagicMock(side_effect=list(answers))
    monkeypatch.setattr(mantic.requests, "get", fake_get)
    return fake_get


def _serve_preflight(
    monkeypatch: pytest.MonkeyPatch, tournaments: list[dict[str, Any]], configured: dict[str, Any] | None = None
) -> MagicMock:
    """The preflight's two GETs in order: the tournament list, then the configured tournament's detail
    (``configured`` defaults to the live configured project)."""
    detail = _tournament(MANTIC_TOURNAMENT_ID, is_ongoing=True) if configured is None else configured
    return _serve(monkeypatch, json_response(tournaments), json_response(detail))


def _requested_urls(fake_get: MagicMock) -> list[str]:
    return [call.args[0] for call in fake_get.call_args_list]


def _question_stub(question_id: int) -> MagicMock:
    question = MagicMock()
    question.id_of_question = question_id
    return question


def _marker_lines(caplog: pytest.LogCaptureFixture, marker: str) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.getMessage().startswith(f"{marker}:")]


@pytest.fixture(autouse=True)
def _fresh_drop_counter():
    reset_post_drop_count()
    yield
    reset_post_drop_count()


@pytest.fixture
def client() -> ManticClient:
    return ManticClient(token=_FAKE_TOKEN)


@pytest.fixture(scope="module")
def posts_by_id() -> dict[int, dict[str, Any]]:
    return {post["id"]: post for post in load_preseason_posts()}


class TestTournamentFetchWalksEveryPage:
    """The fetch asks for a ceiling so the framework pages until an EMPTY page, not until ``next`` is null."""

    @staticmethod
    def _serve_pages(client: ManticClient, monkeypatch: pytest.MonkeyPatch, pages: dict[int, list[MagicMock]]):
        offsets_seen: list[int] = []

        def fake_grab(api_filter: ApiFilter, offset: int = 0) -> tuple[list[MagicMock], bool]:
            offsets_seen.append(offset)
            page = pages[offset]
            return page, bool(page)

        monkeypatch.setattr(client, "_grab_filtered_questions_with_offset", fake_grab)
        return offsets_seen

    def test_a_second_page_is_read_and_the_walk_stops_at_the_empty_one(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch
    ):
        page_size = client.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
        first_page = [_question_stub(index) for index in range(page_size)]
        second_page = [_question_stub(page_size + index) for index in range(3)]
        offsets_seen = self._serve_pages(
            client, monkeypatch, {0: first_page, page_size: second_page, 2 * page_size: []}
        )

        questions = client.get_all_open_questions_from_tournament(MANTIC_TOURNAMENT_ID)

        assert len(questions) == page_size + 3
        assert offsets_seen == [0, page_size, 2 * page_size]

    def test_coming_up_short_of_the_ceiling_is_the_normal_result_not_an_error(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch
    ):
        """Four open questions (the preseason) cost one extra GET of an empty page and raise nothing."""
        page_size = client.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
        preseason = [_question_stub(post_id) for post_id in (648, 649, 650, 651)]
        offsets_seen = self._serve_pages(client, monkeypatch, {0: preseason, page_size: []})

        questions = client.get_all_open_questions_from_tournament(MANTIC_TOURNAMENT_ID)

        assert [question.id_of_question for question in questions] == [648, 649, 650, 651]
        assert offsets_seen == [0, page_size]

    def test_the_ceiling_spans_several_pages(self):
        assert MANTIC_FETCH_QUESTION_CEILING >= 2 * ManticClient.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST

    def test_contrast_the_framework_default_reads_one_page_and_stops(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch
    ):
        """Why the ceiling exists: with ``num_questions`` unset the framework grabs offset 0 once."""
        page_size = client.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
        full_page = [_question_stub(index) for index in range(page_size)]
        offsets_seen = self._serve_pages(client, monkeypatch, {0: full_page, page_size: full_page})
        api_filter = ApiFilter(allowed_tournaments=[MANTIC_TOURNAMENT_ID], allowed_statuses=["open"], allowed_types=[])

        import asyncio  # HARNESS-SCAN-EXEMPT-function-level-import  # one contrast call on the framework's own coroutine

        asyncio.run(client.get_questions_matching_filter(api_filter))

        assert offsets_seen == [0]


class TestParseDropIsCountedAndLogged:
    """A post the framework rejects is counted, logged as ``MANTIC_POST_DROPPED``, and still re-raised."""

    @staticmethod
    def _with_type(post: dict[str, Any], wire_type: str) -> dict[str, Any]:
        return {**post, "question": {**post["question"], "type": wire_type}}

    def test_an_unknown_type_bumps_the_counter_logs_the_marker_and_re_raises(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ):
        post = self._with_type(posts_by_id[DISCRETE_POST_ID], "quantitative_v3")

        with caplog.at_level(logging.ERROR, logger=_MANTIC_LOGGER), pytest.raises(ValueError, match="quantitative_v3"):
            client._post_json_to_questions_while_handling_groups(post, _UNPACK)

        assert get_post_drop_count() == 1
        [record] = _marker_lines(caplog, "MANTIC_POST_DROPPED")
        assert (
            record.getMessage() == f"MANTIC_POST_DROPPED: post={DISCRETE_POST_ID} type=quantitative_v3 error=ValueError"
        )
        assert record.levelno == logging.ERROR

    def test_the_registered_spec_harvests_the_emitted_line(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ):
        """What the client EMITS is what tests/test_telemetry_markers.py pins for the registry."""
        post = self._with_type(posts_by_id[DISCRETE_POST_ID], "quantitative_v3")
        with caplog.at_level(logging.ERROR, logger=_MANTIC_LOGGER), pytest.raises(ValueError, match="quantitative_v3"):
            client._post_json_to_questions_while_handling_groups(post, _UNPACK)
        [record] = _marker_lines(caplog, "MANTIC_POST_DROPPED")

        [harvested] = parse_log_text(_LOG_PREFIX + record.getMessage() + "\n", **_HARVEST_META)["mantic_post_dropped"]

        assert harvested["post"] == DISCRETE_POST_ID
        assert harvested["type"] == "quantitative_v3"
        assert harvested["error"] == "ValueError"
        assert "qid" not in harvested

    def test_a_post_without_a_question_key_renders_type_n_a(
        self, client: ManticClient, caplog: pytest.LogCaptureFixture
    ):
        """The marker's reads are all ``.get``: the same broken post cannot make the marker raise."""
        with caplog.at_level(logging.ERROR, logger=_MANTIC_LOGGER), pytest.raises(AssertionError):
            client._post_json_to_questions_while_handling_groups({"id": 777}, _UNPACK)

        assert get_post_drop_count() == 1
        [record] = _marker_lines(caplog, "MANTIC_POST_DROPPED")
        assert record.getMessage() == "MANTIC_POST_DROPPED: post=777 type=n/a error=AssertionError"

    def test_a_post_with_no_id_at_all_still_logs(self, client: ManticClient, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.ERROR, logger=_MANTIC_LOGGER), pytest.raises(KeyError):
            client._post_json_to_questions_while_handling_groups({"question": {"id": 5}}, _UNPACK)

        [record] = _marker_lines(caplog, "MANTIC_POST_DROPPED")
        assert record.getMessage() == "MANTIC_POST_DROPPED: post=n/a type=n/a error=KeyError"

    def test_healthy_posts_leave_the_counter_at_zero(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER):
            for post in posts_by_id.values():
                client._post_json_to_questions_while_handling_groups(post, _UNPACK)

        assert get_post_drop_count() == 0
        assert _marker_lines(caplog, "MANTIC_POST_DROPPED") == []

    def test_the_counter_accumulates_across_posts_and_resets(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]]
    ):
        for post_id in (DISCRETE_POST_ID, BINARY_POST_ID):
            with pytest.raises(ValueError, match="not_a_type"):
                client._post_json_to_questions_while_handling_groups(
                    self._with_type(posts_by_id[post_id], "not_a_type"), _UNPACK
                )
        assert get_post_drop_count() == 2

        reset_post_drop_count()

        assert get_post_drop_count() == 0

    def test_through_the_frameworks_per_post_loop_the_other_posts_survive(
        self,
        client: ManticClient,
        posts_by_id: dict[int, dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """Fail-fast is kept: the framework still sees the error and logs its own warning, the healthy
        post still parses, and the counter is what turns the swallowed drop into a red run."""
        payload = {
            "next": None,
            "previous": None,
            "results": [self._with_type(posts_by_id[DISCRETE_POST_ID], "quantitative_v3"), posts_by_id[BINARY_POST_ID]],
        }
        response = MagicMock(spec=requests.Response)
        response.status_code = 200
        response.content = json.dumps(payload).encode()
        response.raise_for_status.return_value = None
        monkeypatch.setattr(ft_client.requests, "get", MagicMock(return_value=response))
        client.sleep_time_between_requests_min = 0
        client.sleep_jitter_seconds = 0
        api_filter = ApiFilter(
            allowed_tournaments=[MANTIC_TOURNAMENT_ID],
            allowed_statuses=["open"],
            allowed_types=[],
            group_question_mode=_UNPACK,
        )

        with caplog.at_level(logging.INFO):
            questions = client._get_questions_from_api(client._create_url_params_for_search(api_filter), _UNPACK)

        assert [question.id_of_post for question in questions] == [BINARY_POST_ID]
        assert get_post_drop_count() == 1
        assert any("Error processing post" in record.getMessage() for record in caplog.records)
        assert len(_marker_lines(caplog, "MANTIC_POST_DROPPED")) == 1


class TestListTournaments:
    def test_the_get_is_authenticated_bounded_and_aimed_at_the_tournament_list(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch
    ):
        fake_get = _serve(monkeypatch, json_response(LIVE_TOURNAMENTS))

        assert client.list_tournaments() == LIVE_TOURNAMENTS

        fake_get.assert_called_once()
        assert fake_get.call_args.args[0] == _TOURNAMENTS_URL
        assert fake_get.call_args.kwargs["headers"]["Authorization"] == f"Token {_FAKE_TOKEN}"
        assert fake_get.call_args.kwargs["timeout"] == client.timeout

    @pytest.mark.parametrize("status", [401, 403, 404, 500, 503])
    def test_a_non_200_fails_shut_naming_the_status(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch, status: int
    ):
        _serve(monkeypatch, json_response({"detail": "no"}, status=status))

        with pytest.raises(ApiIdentityError, match=f"status={status}"):
            client.list_tournaments()

    def test_a_200_that_is_not_a_list_fails_shut(self, client: ManticClient, monkeypatch: pytest.MonkeyPatch):
        _serve(monkeypatch, json_response({"detail": "a lander, not the API"}))

        with pytest.raises(ApiIdentityError, match="not with a JSON list"):
            client.list_tournaments()

    def test_a_200_that_is_not_json_fails_shut_with_the_body_preview(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch
    ):
        """The captive-portal shape: a 200 HTML lander. The decode failure is folded into the one
        exception the docstring promises rather than escaping as a bare ``JSONDecodeError``."""
        _serve(monkeypatch, text_response("<html><body>Sign in to the network</body></html>"))

        with pytest.raises(ApiIdentityError, match="not with JSON") as excinfo:
            client.list_tournaments()

        assert "Sign in to the network" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, requests.exceptions.JSONDecodeError)

    @pytest.mark.parametrize(
        "failure",
        [
            requests.ConnectionError("Name or service not known"),
            requests.Timeout("30s"),
            requests.exceptions.SSLError(),
        ],
        ids=lambda failure: type(failure).__name__,
    )
    def test_a_transport_failure_is_the_same_greppable_exception(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ):
        _serve(monkeypatch, failure)

        with pytest.raises(ApiIdentityError, match=type(failure).__name__) as excinfo:
            client.list_tournaments()

        assert excinfo.value.__cause__ is failure
        assert _TOURNAMENTS_URL in str(excinfo.value)


class TestGetTournament:
    """The detail route is authoritative for one project where the list is not: the list omits an
    ``unlisted`` project (the state a new season sits in before its first question), so only this
    route can say whether the configured slug exists and what the token may do on it."""

    def test_the_get_is_authenticated_bounded_and_aimed_at_the_detail_route(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch
    ):
        detail = _tournament(MANTIC_TOURNAMENT_ID, is_ongoing=True)
        fake_get = _serve(monkeypatch, json_response(detail))

        assert client.get_tournament(MANTIC_TOURNAMENT_ID) == detail

        fake_get.assert_called_once()
        assert fake_get.call_args.args[0] == _CONFIGURED_TOURNAMENT_URL
        assert fake_get.call_args.kwargs["headers"]["Authorization"] == f"Token {_FAKE_TOKEN}"
        assert fake_get.call_args.kwargs["timeout"] == client.timeout

    def test_a_404_fails_shut_naming_the_slug_and_the_constant_to_re_point(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch
    ):
        _serve(monkeypatch, json_response(_NOT_FOUND_BODY, status=404))

        with pytest.raises(ApiIdentityError, match="status=404") as excinfo:
            client.get_tournament(_NEXT_SEASON_SLUG)

        message = str(excinfo.value)
        assert f"'{_NEXT_SEASON_SLUG}'" in message
        assert "MANTIC_TOURNAMENT_ID" in message

    def test_a_rejected_token_fails_shut_naming_the_token_variable(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch
    ):
        _serve(monkeypatch, json_response(_INVALID_TOKEN_BODY, status=_INVALID_TOKEN_STATUS))

        with pytest.raises(ApiIdentityError, match=f"status={_INVALID_TOKEN_STATUS}") as excinfo:
            client.get_tournament(MANTIC_TOURNAMENT_ID)

        assert "MANTIC_TOKEN" in str(excinfo.value)

    def test_a_200_that_is_not_an_object_fails_shut(self, client: ManticClient, monkeypatch: pytest.MonkeyPatch):
        _serve(monkeypatch, json_response(LIVE_TOURNAMENTS))

        with pytest.raises(ApiIdentityError, match="not with a JSON object"):
            client.get_tournament(MANTIC_TOURNAMENT_ID)

    def test_a_transport_failure_is_the_same_greppable_exception(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch
    ):
        failure = requests.ConnectionError("connection refused")
        _serve(monkeypatch, failure)

        with pytest.raises(ApiIdentityError, match="ConnectionError") as excinfo:
            client.get_tournament(MANTIC_TOURNAMENT_ID)

        assert excinfo.value.__cause__ is failure


class TestPreflightManticTournaments:
    """Two GETs, two checks: the discovery line off the list, then the permission gate off the detail."""

    def test_the_live_tournaments_pass_and_log_the_marker_at_info(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        fake_get = _serve_preflight(monkeypatch, LIVE_TOURNAMENTS)

        with caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER):
            preflight_mantic_tournaments(client, MANTIC_TOURNAMENT_ID)

        [record] = _marker_lines(caplog, "MANTIC_TOURNAMENTS")
        assert (
            record.getMessage()
            == f"MANTIC_TOURNAMENTS: ongoing={MANTIC_TOURNAMENT_ID} configured={MANTIC_TOURNAMENT_ID} new=none"
        )
        assert record.levelno == logging.INFO
        assert _requested_urls(fake_get) == [_TOURNAMENTS_URL, _CONFIGURED_TOURNAMENT_URL]

    def test_the_registered_spec_harvests_the_emitted_line(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        _serve_preflight(monkeypatch, [*LIVE_TOURNAMENTS, _tournament(_NEXT_SEASON_SLUG, is_ongoing=True)])
        with caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER):
            preflight_mantic_tournaments(client, MANTIC_TOURNAMENT_ID)
        [record] = _marker_lines(caplog, "MANTIC_TOURNAMENTS")

        [harvested] = parse_log_text(_LOG_PREFIX + record.getMessage() + "\n", **_HARVEST_META)["mantic_tournaments"]

        assert harvested["ongoing"] == f"{MANTIC_TOURNAMENT_ID},{_NEXT_SEASON_SLUG}"
        assert harvested["configured"] == MANTIC_TOURNAMENT_ID
        assert harvested["new"] == _NEXT_SEASON_SLUG
        assert "qid" not in harvested

    def test_a_new_ongoing_bots_only_tournament_is_named_at_warning(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """The season hand-over shape: a second ongoing bots-only slug the constants have not been re-pointed at."""
        _serve_preflight(monkeypatch, [*LIVE_TOURNAMENTS, _tournament(_NEXT_SEASON_SLUG, is_ongoing=True)])

        with caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER):
            preflight_mantic_tournaments(client, MANTIC_TOURNAMENT_ID)

        [record] = _marker_lines(caplog, "MANTIC_TOURNAMENTS")
        assert (
            record.getMessage() == f"MANTIC_TOURNAMENTS: ongoing={MANTIC_TOURNAMENT_ID},{_NEXT_SEASON_SLUG} "
            f"configured={MANTIC_TOURNAMENT_ID} new={_NEXT_SEASON_SLUG}"
        )
        assert record.levelno == logging.WARNING

    def test_an_ongoing_human_tournament_is_not_new(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        _serve_preflight(
            monkeypatch, [*LIVE_TOURNAMENTS, _tournament("humans-welcome", is_ongoing=True, bots_only=False)]
        )

        with caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER):
            preflight_mantic_tournaments(client, MANTIC_TOURNAMENT_ID)

        [record] = _marker_lines(caplog, "MANTIC_TOURNAMENTS")
        assert (
            record.getMessage()
            == f"MANTIC_TOURNAMENTS: ongoing=humans-welcome,{MANTIC_TOURNAMENT_ID} configured={MANTIC_TOURNAMENT_ID} new=none"
        )
        assert record.levelno == logging.INFO

    def test_an_ended_configured_tournament_renders_ongoing_none_and_still_passes(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """After its end date the configured tournament stops being ongoing; the stale-slug red exit is cli's job
        (``_check_tournament_dates``), not this gate's, which only asks about permission."""
        ended = _tournament(MANTIC_TOURNAMENT_ID, is_ongoing=False)
        _serve_preflight(monkeypatch, [ended], configured=ended)

        with caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER):
            preflight_mantic_tournaments(client, MANTIC_TOURNAMENT_ID)

        [record] = _marker_lines(caplog, "MANTIC_TOURNAMENTS")
        assert record.getMessage() == f"MANTIC_TOURNAMENTS: ongoing=none configured={MANTIC_TOURNAMENT_ID} new=none"

    @pytest.mark.parametrize("permission", ["viewer", None, ""])
    def test_a_token_that_may_not_forecast_fails_shut_after_the_discovery_line(
        self,
        client: ManticClient,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        permission: str | None,
    ):
        """The permission is read off the DETAIL route: the list row still says ``forecaster`` here."""
        _serve_preflight(
            monkeypatch,
            LIVE_TOURNAMENTS,
            configured=_tournament(MANTIC_TOURNAMENT_ID, is_ongoing=True, user_permission=permission),
        )

        with (
            caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER),
            pytest.raises(ApiIdentityError, match=f"user_permission={permission!r}"),
        ):
            preflight_mantic_tournaments(client, MANTIC_TOURNAMENT_ID)

        assert len(_marker_lines(caplog, "MANTIC_TOURNAMENTS")) == 1

    @pytest.mark.parametrize("permission", sorted(mantic._FORECASTING_PERMISSIONS))
    def test_every_forecasting_role_passes(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch, permission: str
    ):
        _serve_preflight(
            monkeypatch,
            LIVE_TOURNAMENTS,
            configured=_tournament(MANTIC_TOURNAMENT_ID, is_ongoing=True, user_permission=permission),
        )

        preflight_mantic_tournaments(client, MANTIC_TOURNAMENT_ID)

    def test_an_unlisted_configured_tournament_passes_off_the_detail_route(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """The season hand-over: the operator re-points the slug while Mantic still has the project
        ``unlisted``, so it is absent from the list. Absence there is not evidence the project does
        not exist; the detail route answers for it and the run proceeds."""
        fake_get = _serve_preflight(
            monkeypatch, LIVE_TOURNAMENTS, configured=_tournament(_NEXT_SEASON_SLUG, is_ongoing=True)
        )

        with caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER):
            preflight_mantic_tournaments(client, _NEXT_SEASON_SLUG)

        [record] = _marker_lines(caplog, "MANTIC_TOURNAMENTS")
        assert (
            record.getMessage()
            == f"MANTIC_TOURNAMENTS: ongoing={MANTIC_TOURNAMENT_ID} configured={_NEXT_SEASON_SLUG} new={MANTIC_TOURNAMENT_ID}"
        )
        assert _requested_urls(fake_get) == [_TOURNAMENTS_URL, f"{_TOURNAMENTS_URL}{_NEXT_SEASON_SLUG}/"]

    def test_a_slug_no_tournament_has_fails_shut_after_the_discovery_line(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """The detail route 404s for a slug that does not exist; that, not absence from the list, is
        the bad-slug catch."""
        _serve(monkeypatch, json_response(LIVE_TOURNAMENTS), json_response(_NOT_FOUND_BODY, status=404))

        with (
            caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER),
            pytest.raises(ApiIdentityError, match="status=404") as excinfo,
        ):
            preflight_mantic_tournaments(client, _NEXT_SEASON_SLUG)

        assert f"'{_NEXT_SEASON_SLUG}'" in str(excinfo.value)
        assert "MANTIC_TOURNAMENT_ID" in str(excinfo.value)
        assert len(_marker_lines(caplog, "MANTIC_TOURNAMENTS")) == 1

    def test_a_rejected_token_fails_shut_before_any_discovery_line(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        fake_get = _serve(monkeypatch, json_response(_INVALID_TOKEN_BODY, status=_INVALID_TOKEN_STATUS))

        with (
            caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER),
            pytest.raises(ApiIdentityError, match=f"status={_INVALID_TOKEN_STATUS}") as excinfo,
        ):
            preflight_mantic_tournaments(client, MANTIC_TOURNAMENT_ID)

        assert "MANTIC_TOKEN" in str(excinfo.value)
        assert _marker_lines(caplog, "MANTIC_TOURNAMENTS") == []
        fake_get.assert_called_once()

    def test_a_connect_failure_stops_the_run_with_the_same_exception(
        self, client: ManticClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """DNS, TLS and connect failures stop the run before any spend either way; wrapping them
        makes the stop one greppable ``ApiIdentityError`` instead of a requests traceback."""
        failure = requests.ConnectionError("Name or service not known")
        _serve(monkeypatch, failure)

        with (
            caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER),
            pytest.raises(ApiIdentityError, match="ConnectionError") as excinfo,
        ):
            preflight_mantic_tournaments(client, MANTIC_TOURNAMENT_ID)

        assert excinfo.value.__cause__ is failure
        assert _marker_lines(caplog, "MANTIC_TOURNAMENTS") == []
