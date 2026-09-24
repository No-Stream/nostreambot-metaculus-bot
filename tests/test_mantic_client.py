"""Tests for metaculus_bot/mantic.py, the client for Mantic's competitions platform.

Mantic (competitions.mantic.com) runs a fork of the open-source Metaculus platform, so the
framework's ``MetaculusClient`` is the transport and the subclass exists for the three differences
verified by the 2026-09-08 live probe: the list filter's ``forecast_type`` vocabulary
(``quantitative`` where the framework sends ``numeric,discrete``, so the default fetch returns ZERO
quantitative questions), the ``quantitative`` question type that forecasting-tools 0.2.92 rejects
(and whose caller swallows the rejection, silently dropping the post), and the hardcoded
metaculus.com page URL.

``tests/data/mantic_preseason2_posts_2026_09_08.json`` is that probe's authenticated
``GET /api/posts/?tournaments=preseason-2`` response verbatim: four posts, one per question type,
``my_forecasts`` present with empty histories (``tests/mantic_fakes.py`` holds its path and post ids,
and derives the already-forecast state the probe could not record). No test constructs a client that
connects (the autouse egress guard in conftest would refuse it); every seam under test is reachable
through the client's own parsing and URL-parameter methods.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests
from forecasting_tools import ApiFilter, BinaryQuestion, DateQuestion, DiscreteQuestion, MultipleChoiceQuestion
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.helpers import metaculus_client as ft_client
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from metaculus_bot.constants import MANTIC_API_BASE_URL, MANTIC_SITE_URL, MANTIC_TOKEN_ENV, MANTIC_TOURNAMENT_ID
from metaculus_bot.mantic import ManticClient, build_mantic_client
from scripts.telemetry.markers import MARKER_SPECS, parse_log_text
from tests.mantic_fakes import (
    BINARY_POST_ID,
    DATE_POST_ID,
    DISCRETE_POST_ID,
    MULTIPLE_CHOICE_POST_ID,
    PRESEASON_POST_IDS,
    load_preseason_posts,
    with_prior_forecast,
)

_FAKE_TOKEN = "f" * 40
_UNPACK = "unpack_subquestions"
_MANTIC_LOGGER = "metaculus_bot.mantic"

# What the bot's own user would have standing on each question type, in the platform's shape.
_BINARY_PRIOR_FORECAST_VALUES = (0.78, 0.22)
_DISCRETE_PRIOR_FORECAST_CDF = [step / 450 for step in range(451)]

# Prod cli.py log format, as in tests/test_telemetry_markers.py.
_LOG_PREFIX = "2026-09-08 14:23:01,123 - metaculus_bot.mantic - INFO - "
_HARVEST_META = {
    "run_id": "999",
    "workflow": "mantic",
    "artifact": "research-999",
    "run_date": "2026-09-08T14:00:00Z",
    "log_file": "run.log",
}


@pytest.fixture(scope="module")
def posts_by_id() -> dict[int, dict[str, Any]]:
    return {post["id"]: post for post in load_preseason_posts()}


@pytest.fixture
def client() -> ManticClient:
    return ManticClient(token=_FAKE_TOKEN)


def _parse(client: ManticClient, post: dict[str, Any]) -> list[MetaculusQuestion]:
    return client._post_json_to_questions_while_handling_groups(post, _UNPACK)


def _parse_one(client: ManticClient, post: dict[str, Any]) -> MetaculusQuestion:
    questions = _parse(client, post)
    assert len(questions) == 1
    return questions[0]


def _as_quantitative(post: dict[str, Any]) -> dict[str, Any]:
    """The Series 2 wire shape: the preseason discrete question with Mantic's merged type."""
    quantitative = copy.deepcopy(post)
    quantitative["question"]["type"] = "quantitative"
    return quantitative


def _mantic_question_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith("MANTIC_QUESTION:")]


def _harvest(line: str) -> dict[str, Any]:
    harvested = parse_log_text(_LOG_PREFIX + line + "\n", **_HARVEST_META)
    records = [record for records in harvested.values() for record in records]
    assert len(records) == 1, f"expected exactly one harvested record, got {records}"
    return records[0]


class TestConstruction:
    def test_targets_the_mantic_api_with_the_given_token(self, monkeypatch: pytest.MonkeyPatch):
        """The framework's __init__ falls back to these two env vars; a Mantic client must never
        pick up the Metaculus token or base URL from a laptop .env that has both."""
        monkeypatch.setenv("METACULUS_TOKEN", "decoy-metaculus-token")
        monkeypatch.setenv("METACULUS_API_BASE_URL", "https://decoy.example/api")
        client = ManticClient(token=_FAKE_TOKEN)
        assert client.base_url == MANTIC_API_BASE_URL
        assert client.base_url == "https://competitions.mantic.com/api"
        assert client._get_auth_headers()["headers"]["Authorization"] == f"Token {_FAKE_TOKEN}"

    def test_is_a_metaculus_client(self, client: ManticClient):
        """ForecastBot's metaculus_client= seam and the class-level hardening both key on this."""
        assert isinstance(client, MetaculusClient)

    @pytest.mark.parametrize(
        "hardened_method",
        ["_get_questions_from_api", "_post_question_prediction", "post_question_comment"],
    )
    def test_hardened_seams_are_inherited_not_overridden(self, hardened_method: str):
        """fetch_hardening and publish_hardening patch these three names on MetaculusClient at class
        level; an override here would shadow the patched versions and run unhardened."""
        assert hardened_method not in ManticClient.__dict__


class TestBuildManticClient:
    def test_raises_naming_the_env_var_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(MANTIC_TOKEN_ENV, raising=False)
        with pytest.raises(RuntimeError, match=MANTIC_TOKEN_ENV):
            build_mantic_client()

    def test_raises_when_set_but_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(MANTIC_TOKEN_ENV, "")
        with pytest.raises(RuntimeError, match=MANTIC_TOKEN_ENV):
            build_mantic_client()

    def test_builds_a_client_on_the_token(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(MANTIC_TOKEN_ENV, _FAKE_TOKEN)
        client = build_mantic_client()
        assert isinstance(client, ManticClient)
        assert client.token == _FAKE_TOKEN
        assert client.base_url == MANTIC_API_BASE_URL


class TestTournamentFetchFilter:
    def test_fetch_omits_the_forecast_type_param(self, client: ManticClient, monkeypatch: pytest.MonkeyPatch):
        sentinel_questions: list[MetaculusQuestion] = []
        fake_fetch = AsyncMock(return_value=sentinel_questions)
        monkeypatch.setattr(client, "get_questions_matching_filter", fake_fetch)

        returned = client.get_all_open_questions_from_tournament(MANTIC_TOURNAMENT_ID)

        assert returned is sentinel_questions
        fake_fetch.assert_awaited_once()
        api_filter = fake_fetch.await_args_list[0].args[0]
        assert isinstance(api_filter, ApiFilter)
        assert api_filter.allowed_types == []
        assert api_filter.allowed_tournaments == [MANTIC_TOURNAMENT_ID]
        assert api_filter.allowed_statuses == ["open"]
        assert api_filter.group_question_mode == _UNPACK

        params = client._create_url_params_for_search(api_filter)
        assert "forecast_type" not in params
        assert params["tournaments"] == [MANTIC_TOURNAMENT_ID]
        assert params["statuses"] == ["open"]

    def test_group_question_mode_passes_through(self, client: ManticClient, monkeypatch: pytest.MonkeyPatch):
        fake_fetch = AsyncMock(return_value=[])
        monkeypatch.setattr(client, "get_questions_matching_filter", fake_fetch)
        client.get_all_open_questions_from_tournament(MANTIC_TOURNAMENT_ID, group_question_mode="exclude")
        assert fake_fetch.await_args_list[0].args[0].group_question_mode == "exclude"

    def test_contrast_the_framework_default_speaks_the_metaculus_vocabulary(self, client: ManticClient):
        """Why the override exists: the framework's default filter asks for numeric,discrete and
        Mantic answers that with nothing (its word is quantitative). If this pin ever fails the
        framework changed vocabulary and the override needs re-deriving, not deleting."""
        params = client._create_url_params_for_search(ApiFilter(allowed_statuses=["open"]))
        assert "numeric" in params["forecast_type"]
        assert "discrete" in params["forecast_type"]
        assert "quantitative" not in params["forecast_type"]


class TestParsingThePreseasonFixture:
    def test_binary(self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]]):
        question = _parse_one(client, posts_by_id[BINARY_POST_ID])
        assert isinstance(question, BinaryQuestion)

    def test_multiple_choice_keeps_all_four_options(self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]]):
        question = _parse_one(client, posts_by_id[MULTIPLE_CHOICE_POST_ID])
        assert isinstance(question, MultipleChoiceQuestion)
        assert len(question.options) == 4
        assert question.options == posts_by_id[MULTIPLE_CHOICE_POST_ID]["question"]["options"]

    def test_discrete_carries_the_451_point_grid(self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]]):
        """450 x $100 bins, both bounds open: the first Mantic grid past the Metaculus 201 cap."""
        question = _parse_one(client, posts_by_id[DISCRETE_POST_ID])
        assert isinstance(question, DiscreteQuestion)
        assert question.cdf_size == 451
        assert question.open_lower_bound is True
        assert question.open_upper_bound is True
        assert question.lower_bound == 54950.0
        assert question.upper_bound == 99950.0
        assert question.unit_of_measure == "$"

    def test_date_carries_the_13_point_grid(self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]]):
        question = _parse_one(client, posts_by_id[DATE_POST_ID])
        assert isinstance(question, DateQuestion)
        assert question.cdf_size == 13
        assert question.open_lower_bound is False
        assert question.open_upper_bound is False

    @pytest.mark.parametrize("post_id", PRESEASON_POST_IDS)
    def test_page_url_points_at_mantic(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], post_id: int
    ):
        """The framework hardcodes https://www.metaculus.com/questions/{post_id}; Mantic's shape
        (verified 200) is the trailing-slash form, which the no-slash form 308-redirects to."""
        question = _parse_one(client, posts_by_id[post_id])
        assert question.page_url == f"{MANTIC_SITE_URL}/questions/{post_id}/"
        assert question.page_url == f"https://competitions.mantic.com/questions/{post_id}/"

    @pytest.mark.parametrize("post_id", PRESEASON_POST_IDS)
    def test_post_id_equals_question_id_on_single_question_posts(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], post_id: int
    ):
        question = _parse_one(client, posts_by_id[post_id])
        assert question.id_of_post == post_id
        assert question.id_of_question == post_id

    @pytest.mark.parametrize("post_id", PRESEASON_POST_IDS)
    def test_already_forecasted_reads_the_authenticated_my_forecasts_field(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], post_id: int
    ):
        """The fixture is an authenticated read with empty histories, so the framework's
        skip-previously-forecasted path sees every question as fresh."""
        assert posts_by_id[post_id]["question"]["my_forecasts"]["history"] == []
        question = _parse_one(client, posts_by_id[post_id])
        assert question.already_forecasted is False

    def test_a_prior_forecast_parses_as_already_forecasted(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]]
    ):
        """The direction with power. An empty history reads as fresh whether ``my_forecasts`` is read
        correctly or is missing entirely (the framework derives the flag inside a blanket except that
        answers False), so only a non-empty history proves the field survives the client's parse and
        reaches the skip filter that bounds the hourly run's spend."""
        post = with_prior_forecast(posts_by_id[BINARY_POST_ID], forecast_values=_BINARY_PRIOR_FORECAST_VALUES)
        question = _parse_one(client, post)
        assert isinstance(question, BinaryQuestion)
        assert question.already_forecasted is True

    def test_the_fixture_is_the_full_preseason(self, posts_by_id: dict[int, dict[str, Any]]):
        assert set(posts_by_id) == set(PRESEASON_POST_IDS)
        assert {post["question"]["type"] for post in posts_by_id.values()} == {
            "binary",
            "multiple_choice",
            "discrete",
            "date",
        }


class TestThePriorForecastFixture:
    """``with_prior_forecast`` must change one field and nothing else, or the tests built on it
    (here and the second run in tests/test_mantic_e2e.py) would be testing the derivation."""

    def test_only_my_forecasts_differs_from_the_probe(self, posts_by_id: dict[int, dict[str, Any]]):
        original = posts_by_id[BINARY_POST_ID]
        derived = with_prior_forecast(original, forecast_values=_BINARY_PRIOR_FORECAST_VALUES)

        assert original["question"]["my_forecasts"]["history"] == [], "the shared fixture must stay untouched"
        without_forecasts = lambda post: {  # noqa: E731
            **post,
            "question": {key: value for key, value in post["question"].items() if key != "my_forecasts"},
        }
        assert without_forecasts(derived) == without_forecasts(original)

    def test_history_and_latest_carry_the_same_platform_shaped_entry(self, posts_by_id: dict[int, dict[str, Any]]):
        my_forecasts = with_prior_forecast(posts_by_id[BINARY_POST_ID], forecast_values=_BINARY_PRIOR_FORECAST_VALUES)[
            "question"
        ]["my_forecasts"]
        (entry,) = my_forecasts["history"]
        assert my_forecasts["latest"] == entry
        assert entry["question_id"] == BINARY_POST_ID
        assert entry["forecast_values"] == list(_BINARY_PRIOR_FORECAST_VALUES)
        assert entry["end_time"] is None, "a standing forecast has no end"
        assert isinstance(entry["start_time"], float), "the platform serializes forecast times as unix timestamps"


class TestQuantitativeTypeNormalization:
    def test_contrast_the_framework_rejects_the_quantitative_type(self, posts_by_id: dict[int, dict[str, Any]]):
        """The failure the override prevents: 0.2.92's DataOrganizer raises, and
        _get_questions_from_api turns that into a warning and drops the post."""
        framework_client = MetaculusClient(token=_FAKE_TOKEN)
        with pytest.raises(ValueError, match="Unknown question type: quantitative"):
            framework_client._post_json_to_questions_while_handling_groups(
                _as_quantitative(posts_by_id[DISCRETE_POST_ID]), _UNPACK
            )

    def test_quantitative_parses_as_discrete_on_the_same_grid(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]]
    ):
        question = _parse_one(client, _as_quantitative(posts_by_id[DISCRETE_POST_ID]))
        assert isinstance(question, DiscreteQuestion)
        assert question.cdf_size == 451
        assert question.lower_bound == 54950.0
        assert question.open_lower_bound is True
        assert question.page_url == f"{MANTIC_SITE_URL}/questions/{DISCRETE_POST_ID}/"

    def test_the_marker_records_the_wire_type(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER):
            _parse_one(client, _as_quantitative(posts_by_id[DISCRETE_POST_ID]))
        (line,) = _mantic_question_lines(caplog)
        assert " type=quantitative " in line

    def test_the_callers_post_json_is_left_untouched(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]]
    ):
        quantitative = _as_quantitative(posts_by_id[DISCRETE_POST_ID])
        snapshot = copy.deepcopy(quantitative)
        _parse_one(client, quantitative)
        assert quantitative == snapshot
        assert quantitative["question"]["type"] == "quantitative"

    def test_the_rewrite_keeps_the_prior_forecast(self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]]):
        """The type rewrite is the one place this client rebuilds question JSON, so it is the one
        place ``my_forecasts`` could be dropped; dropped, every quantitative question would be
        re-forecast on every hourly run."""
        post = with_prior_forecast(
            _as_quantitative(posts_by_id[DISCRETE_POST_ID]), forecast_values=_DISCRETE_PRIOR_FORECAST_CDF
        )
        question = _parse_one(client, post)
        assert isinstance(question, DiscreteQuestion)
        assert question.already_forecasted is True

    def test_group_subquestions_normalize_too(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ):
        """A group post carries its questions under group_of_questions.questions, each with its
        own type; the framework's _unpack_group_question reads these four group keys."""
        discrete_post = posts_by_id[DISCRETE_POST_ID]
        quantitative_sub = copy.deepcopy(discrete_post["question"])
        quantitative_sub["type"] = "quantitative"
        binary_sub = copy.deepcopy(posts_by_id[BINARY_POST_ID]["question"])
        group_post = {key: value for key, value in discrete_post.items() if key != "question"}
        group_post["group_of_questions"] = {
            "questions": [quantitative_sub, binary_sub],
            "fine_print": "group fine print",
            "description": "group description",
            "resolution_criteria": "group resolution criteria",
        }
        snapshot = copy.deepcopy(group_post)

        with caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER):
            questions = _parse(client, group_post)

        assert [type(q) for q in questions] == [DiscreteQuestion, BinaryQuestion]
        assert all(q.question_ids_of_group == [DISCRETE_POST_ID, BINARY_POST_ID] for q in questions)
        assert all(q.page_url == f"{MANTIC_SITE_URL}/questions/{DISCRETE_POST_ID}/" for q in questions)
        assert questions[0].background_info == "group description"
        lines = _mantic_question_lines(caplog)
        assert len(lines) == 2
        assert f" question={DISCRETE_POST_ID} type=quantitative " in lines[0]
        assert f" question={BINARY_POST_ID} type=binary " in lines[1]
        assert group_post == snapshot

    def test_excluded_groups_emit_nothing(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ):
        group_post = {key: value for key, value in posts_by_id[DISCRETE_POST_ID].items() if key != "question"}
        group_post["group_of_questions"] = {
            "questions": [_as_quantitative(posts_by_id[DISCRETE_POST_ID])["question"]],
            "fine_print": "",
            "description": "",
            "resolution_criteria": "",
        }
        with caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER):
            questions = client._post_json_to_questions_while_handling_groups(group_post, "exclude")
        assert questions == []
        assert _mantic_question_lines(caplog) == []


class TestFetchLoopEndToEnd:
    """One pass through the framework's own ``_get_questions_from_api`` on a canned HTTP response.

    That loop is where a quantitative post used to die: ``DataOrganizer`` raised, the loop logged
    "Error processing post" and moved on. Driving it with the real request path (mocked at
    ``requests.get``) pins that the override is reached through the framework's dispatch, with
    the Mantic URL, token header and parameter set the client actually sends.
    """

    def test_a_quantitative_post_survives_the_per_post_loop(
        self,
        client: ManticClient,
        posts_by_id: dict[int, dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        payload = {
            "next": None,
            "previous": None,
            "results": [_as_quantitative(posts_by_id[DISCRETE_POST_ID]), posts_by_id[BINARY_POST_ID]],
        }
        response = MagicMock(spec=requests.Response)
        response.status_code = 200
        response.content = json.dumps(payload).encode()
        response.raise_for_status.return_value = None
        fake_get = MagicMock(return_value=response)
        monkeypatch.setattr(ft_client.requests, "get", fake_get)
        # The framework's inter-request politeness sleep; zero it rather than patch time.sleep.
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

        assert [type(question) for question in questions] == [DiscreteQuestion, BinaryQuestion]
        assert not any("Error processing post" in record.getMessage() for record in caplog.records)
        assert len(_mantic_question_lines(caplog)) == 2
        assert [question.page_url for question in questions] == [
            f"{MANTIC_SITE_URL}/questions/{DISCRETE_POST_ID}/",
            f"{MANTIC_SITE_URL}/questions/{BINARY_POST_ID}/",
        ]

        fake_get.assert_called_once()
        assert fake_get.call_args.args[0] == f"{MANTIC_API_BASE_URL}/posts/"
        request_kwargs = fake_get.call_args.kwargs
        assert request_kwargs["headers"]["Authorization"] == f"Token {_FAKE_TOKEN}"
        assert "forecast_type" not in request_kwargs["params"]
        assert request_kwargs["params"]["tournaments"] == [MANTIC_TOURNAMENT_ID]
        assert request_kwargs["params"]["statuses"] == ["open"]


class TestManticQuestionMarker:
    """The MANTIC_QUESTION line and its round trip through the registered spec.

    The verbatim example lines also live in tests/test_telemetry_markers.py (the registry's own
    pin); this class checks that what the client EMITS is what that pin says.
    """

    def _emitted_line(self, client: ManticClient, post: dict[str, Any], caplog: pytest.LogCaptureFixture) -> str:
        with caplog.at_level(logging.INFO, logger=_MANTIC_LOGGER):
            _parse_one(client, post)
        (line,) = _mantic_question_lines(caplog)
        return line

    def test_discrete_line_verbatim(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ):
        line = self._emitted_line(client, posts_by_id[DISCRETE_POST_ID], caplog)
        assert line == (
            "MANTIC_QUESTION: post=650 question=650 type=discrete cdf_size=451 "
            "multi_resolution=true date_granularity=n/a precision=100.0"
        )

    def test_binary_line_renders_the_absent_fields_as_n_a(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ):
        line = self._emitted_line(client, posts_by_id[BINARY_POST_ID], caplog)
        assert line == (
            "MANTIC_QUESTION: post=648 question=648 type=binary cdf_size=n/a "
            "multi_resolution=false date_granularity=n/a precision=n/a"
        )

    def test_date_line_carries_the_granularity(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ):
        line = self._emitted_line(client, posts_by_id[DATE_POST_ID], caplog)
        assert line == (
            "MANTIC_QUESTION: post=651 question=651 type=date cdf_size=13 "
            "multi_resolution=false date_granularity=day precision=n/a"
        )

    def test_the_registered_spec_harvests_the_emitted_discrete_line(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ):
        record = _harvest(self._emitted_line(client, posts_by_id[DISCRETE_POST_ID], caplog))
        assert record["marker"] == "mantic_question"
        assert record["post"] == 650
        assert record["qid"] == 650
        assert record["qid_kind"] == "question_id"
        assert record["type"] == "discrete"
        assert record["cdf_size"] == 451
        assert record["multi_resolution"] is True
        assert record["date_granularity"] is None
        assert record["precision"] == pytest.approx(100.0)

    def test_the_registered_spec_harvests_the_absent_fields_as_none(
        self, client: ManticClient, posts_by_id: dict[int, dict[str, Any]], caplog: pytest.LogCaptureFixture
    ):
        record = _harvest(self._emitted_line(client, posts_by_id[BINARY_POST_ID], caplog))
        assert record["type"] == "binary"
        assert record["cdf_size"] is None
        assert record["multi_resolution"] is False
        assert record["date_granularity"] is None
        assert record["precision"] is None

    def test_the_spec_is_registered_in_the_question_id_space(self):
        spec = next(spec for spec in MARKER_SPECS if spec.name == "mantic_question")
        assert spec.qid_kind == "question_id"
        assert set(spec.regex.groupindex) == {
            "post",
            "question",
            "type",
            "cdf_size",
            "multi_resolution",
            "date_granularity",
            "precision",
        }
