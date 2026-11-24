"""``--mode mantic`` and the ``--only-posts`` narrowing that the paid Mantic smoke run uses.

The Mantic path's own guards in startup order: the fail-shut personal-keys check, the
mode-dependent identity preflight, the injected ``ManticClient``, and the question source.
``--only-posts`` rides along because the smoke run is how it gets exercised, though it applies
to every tournament-shaped mode.
"""

from __future__ import annotations

import logging
import sys
from typing import ClassVar, get_args
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import MetaculusApi

from metaculus_bot import mantic as mantic_module
from metaculus_bot.api_preflight import ApiIdentityError
from metaculus_bot.cli import (
    CliArgs,
    RunMode,
    _assert_personal_keys_only,
    _configure_process,
    _parse_cli_args,
    _run_forecasts,
)
from metaculus_bot.cli import main as cli_main
from metaculus_bot.constants import (
    DONATED_OPENROUTER_KEY_ENABLED_ENV,
    MANTIC_API_BASE_URL,
    MANTIC_TOKEN_ENV,
    MANTIC_TOURNAMENT_ID,
    METACULUS_CUP_ID,
    TOURNAMENT_ID,
)
from metaculus_bot.mantic import ManticClient
from scripts.telemetry.markers import MARKER_SPECS
from tests.cli_test_helpers import (
    _FAKE_MANTIC_TOKEN,
    _cli_main_test_mode,
    _configure_process_stubs,
    _filterable_bot,
    _forecaster_class,
    _mantic_env,
)


class TestAssertPersonalKeysOnly:
    """The fail-shut guard for Mantic runs. Metaculus donated ``OAI_ANTH_OPENROUTER_KEY`` for its own
    tournaments, so a run that forecasts for Mantic may spend only personal keys; the switch has to
    be off in the environment before the process starts, because the roster freezes its OpenRouter
    key at import (``llm_configs``) and ``main.py`` imports it before ``cli.main`` runs."""

    @pytest.mark.parametrize("switch", [None, "true", "1", ""])
    def test_raises_naming_the_switch_unless_it_reads_false(
        self, switch: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if switch is None:
            monkeypatch.delenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, raising=False)
        else:
            monkeypatch.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, switch)
        with pytest.raises(RuntimeError, match=DONATED_OPENROUTER_KEY_ENABLED_ENV):
            _assert_personal_keys_only()

    def test_passes_when_the_switch_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, "false")
        _assert_personal_keys_only()


class TestConfigureProcess:
    """The run mode decides which identity preflight runs. A Mantic run vets the Mantic API host and
    never contacts metaculus.com (it must not depend on Metaculus DNS health), after failing shut on
    the donated-key switch; every Metaculus mode is unchanged. The hardening patches are mode-blind."""

    METACULUS_MODES: ClassVar[list[str]] = sorted(set(get_args(RunMode)) - {"mantic"})

    @pytest.mark.parametrize("run_mode", METACULUS_MODES)
    def test_metaculus_modes_preflight_metaculus_only(self, run_mode: RunMode, monkeypatch: pytest.MonkeyPatch) -> None:
        """The switch's default (donated key on) IS the Metaculus production state, so it must not raise."""
        monkeypatch.delenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, raising=False)
        with _configure_process_stubs() as stubs:
            _configure_process(run_mode)
        stubs["verify_metaculus_api_identity"].assert_called_once_with()
        stubs["verify_api_identity"].assert_not_called()

    def test_mantic_mode_preflights_the_mantic_host_and_never_metaculus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mantic_env(monkeypatch)
        with _configure_process_stubs() as stubs:
            _configure_process("mantic")
        stubs["verify_api_identity"].assert_called_once_with(MANTIC_API_BASE_URL)
        stubs["verify_metaculus_api_identity"].assert_not_called()

    def test_mantic_mode_fails_shut_before_any_preflight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, raising=False)
        with (
            _configure_process_stubs() as stubs,
            pytest.raises(RuntimeError, match=DONATED_OPENROUTER_KEY_ENABLED_ENV),
        ):
            _configure_process("mantic")
        stubs["verify_api_identity"].assert_not_called()
        stubs["verify_metaculus_api_identity"].assert_not_called()

    @pytest.mark.parametrize("run_mode", get_args(RunMode))
    def test_hardening_is_installed_in_every_mode(self, run_mode: RunMode, monkeypatch: pytest.MonkeyPatch) -> None:
        _mantic_env(monkeypatch)
        with _configure_process_stubs() as stubs:
            _configure_process(run_mode)
        stubs["publish_hardening"].assert_called_once_with()
        stubs["fetch_hardening"].assert_called_once_with()

    @pytest.mark.parametrize("run_mode", get_args(RunMode))
    def test_the_mantic_parse_drop_counter_is_reset_at_startup(
        self, run_mode: RunMode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reset here rather than in forecast_questions, whose resets run AFTER the fetch that bumps it."""
        _mantic_env(monkeypatch)
        mantic_module._post_drop_count = 3
        with _configure_process_stubs():
            _configure_process(run_mode)
        assert mantic_module.get_post_drop_count() == 0


class TestManticQuestionSource:
    """``--mode mantic`` mirrors the tournament mode over the Mantic slug: the re-spend guard on and
    the forecast over ``MANTIC_TOURNAMENT_ID``. The stale-date check runs at startup (next class)."""

    def test_mantic_mode_forecasts_the_mantic_tournament(self) -> None:
        bot = MagicMock()
        bot.skip_previously_forecasted_questions = False
        bot.forecast_on_tournament = AsyncMock(return_value=["report"])

        with patch("metaculus_bot.cli.drain_litellm_callbacks", AsyncMock()):
            assert _run_forecasts(bot, "mantic") == ["report"]

        bot.forecast_on_tournament.assert_awaited_once_with(MANTIC_TOURNAMENT_ID, return_exceptions=True)
        assert bot.skip_previously_forecasted_questions is True


class TestOnlyPostsFilter:
    """``--only-posts`` narrows a tournament-shaped run to the listed post ids: the one-question
    paid smoke run. The question-list fetch is the one ``forecast_on_tournament`` makes, on the
    same injected client, and only the matching questions reach ``forecast_questions``. Without
    the flag every mode still forecasts through ``forecast_on_tournament`` exactly as before.
    """

    TOURNAMENT_SLUGS: ClassVar[dict[str, int | str]] = {
        "tournament": TOURNAMENT_ID,
        "minibench": MetaculusApi.CURRENT_MINIBENCH_ID,
        "quarterly_cup": METACULUS_CUP_ID,
        "metaculus_cup": METACULUS_CUP_ID,
        "mantic": MANTIC_TOURNAMENT_ID,
    }

    def test_every_tournament_shaped_mode_is_covered(self) -> None:
        """Derived from RunMode, so a new tournament-shaped mode fails here until it is listed."""
        assert set(self.TOURNAMENT_SLUGS) == set(get_args(RunMode)) - {"test_questions"}

    @pytest.mark.parametrize("run_mode", sorted(TOURNAMENT_SLUGS))
    def test_only_the_requested_posts_reach_forecast_questions(
        self, run_mode: RunMode, caplog: pytest.LogCaptureFixture
    ) -> None:
        bot = _filterable_bot(648, 650, 651, 652)
        open_questions = bot.metaculus_client.get_all_open_questions_from_tournament.return_value

        with (
            patch("metaculus_bot.cli.check_tournament_dates"),
            patch("metaculus_bot.cli.drain_litellm_callbacks", AsyncMock()),
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
        ):
            assert _run_forecasts(bot, run_mode, only_posts=frozenset({652, 650})) == ["report"]

        bot.metaculus_client.get_all_open_questions_from_tournament.assert_called_once_with(
            self.TOURNAMENT_SLUGS[run_mode]
        )
        # Fetch order, not request order, and nothing but the two asked for.
        bot.forecast_questions.assert_awaited_once_with([open_questions[1], open_questions[3]], return_exceptions=True)
        bot.forecast_on_tournament.assert_not_awaited()
        # The re-spend guard is pinned on exactly as in an unfiltered run.
        assert bot.skip_previously_forecasted_questions is True
        assert "ONLY_POSTS: requested=650,652 matched=650,652 dropped=2" in caplog.messages

    def test_the_marker_line_is_the_one_the_harvester_reads(self, caplog: pytest.LogCaptureFixture) -> None:
        """Seam pin: the archive keys off the exact spelling (scripts/telemetry/markers.py)."""
        bot = _filterable_bot(649, 650, 651)

        with (
            patch("metaculus_bot.cli.check_tournament_dates"),
            patch("metaculus_bot.cli.drain_litellm_callbacks", AsyncMock()),
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
        ):
            _run_forecasts(bot, "mantic", only_posts=frozenset({650}))

        marker_line = next(message for message in caplog.messages if message.startswith("ONLY_POSTS:"))
        spec = next(s for s in MARKER_SPECS if s.name == "only_posts")
        match = spec.regex.search(marker_line)
        assert match is not None, marker_line
        assert match.group("requested") == "650"
        assert match.group("matched") == "650"
        assert match.group("dropped") == "2"

    def test_an_unmatched_filter_warns_and_forecasts_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        """No match forecasts nothing, never the whole tournament, and says so at WARNING."""
        bot = _filterable_bot(650)

        with (
            patch("metaculus_bot.cli.check_tournament_dates"),
            patch("metaculus_bot.cli.drain_litellm_callbacks", AsyncMock()),
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
        ):
            assert _run_forecasts(bot, "mantic", only_posts=frozenset({999})) == []

        bot.forecast_questions.assert_not_awaited()
        bot.forecast_on_tournament.assert_not_awaited()
        assert "ONLY_POSTS: requested=999 matched=none dropped=1" in caplog.messages
        warning = next(record for record in caplog.records if record.levelno == logging.WARNING)
        assert "--only-posts matched none of the 1 open question(s)" in warning.getMessage()

    @pytest.mark.parametrize("run_mode", sorted(TOURNAMENT_SLUGS))
    def test_without_the_filter_every_mode_still_forecasts_on_tournament(self, run_mode: RunMode) -> None:
        bot = _filterable_bot(650)

        with (
            patch("metaculus_bot.cli.check_tournament_dates"),
            patch("metaculus_bot.cli.drain_litellm_callbacks", AsyncMock()),
        ):
            assert _run_forecasts(bot, run_mode) == ["report"]

        bot.forecast_on_tournament.assert_awaited_once_with(self.TOURNAMENT_SLUGS[run_mode], return_exceptions=True)
        bot.metaculus_client.get_all_open_questions_from_tournament.assert_not_called()
        bot.forecast_questions.assert_not_awaited()

    def test_the_flag_reaches_the_filter_through_main(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The whole argv path: ``--only-posts 650`` on a mantic run forecasts post 650 alone."""
        _mantic_env(monkeypatch)
        stub_bot = _filterable_bot(649, 650, 651)
        stub_bot.alertable_count = 0
        open_questions = stub_bot.metaculus_client.get_all_open_questions_from_tournament.return_value

        with (
            _cli_main_test_mode(alertable_count=0, stub_bot=stub_bot, mode="mantic", only_posts="650"),
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
        ):
            cli_main()

        stub_bot.metaculus_client.get_all_open_questions_from_tournament.assert_called_once_with(MANTIC_TOURNAMENT_ID)
        stub_bot.forecast_questions.assert_awaited_once_with([open_questions[1]], return_exceptions=True)
        stub_bot.forecast_on_tournament.assert_not_awaited()
        assert "ONLY_POSTS: requested=650 matched=650 dropped=2" in caplog.messages

    def test_parses_a_comma_separated_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["cli", "--mode", "mantic", "--only-posts", "650,651"])
        assert _parse_cli_args() == CliArgs(run_mode="mantic", only_posts=frozenset({650, 651}))

    def test_defaults_to_no_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["cli"])
        assert _parse_cli_args() == CliArgs(run_mode="tournament", only_posts=None)

    def test_a_non_integer_id_is_a_usage_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["cli", "--mode", "mantic", "--only-posts", "650,abc"])
        with pytest.raises(SystemExit) as exc_info:
            _parse_cli_args()
        assert exc_info.value.code == 2
        assert "comma-separated integer post ids" in capsys.readouterr().err

    def test_the_filter_is_refused_for_test_questions(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The evergreen set is not a tournament's open questions, so a filter there would be a
        silent no-op on a paid run; the parser refuses it instead."""
        monkeypatch.setattr(sys, "argv", ["cli", "--mode", "test_questions", "--only-posts", "650"])
        with pytest.raises(SystemExit) as exc_info:
            _parse_cli_args()
        assert exc_info.value.code == 2
        assert "--only-posts" in capsys.readouterr().err


class TestManticClientWiring:
    """``main`` hands the framework a ``ManticClient`` in mantic mode and nothing (so the framework
    builds its default Metaculus client) otherwise. The client is built only after the fail-shut
    check and the identity preflight, so the Mantic token is not even read before the host is vetted."""

    def test_mantic_mode_injects_a_mantic_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mantic_env(monkeypatch)
        forecaster_class = _forecaster_class()
        with _cli_main_test_mode(alertable_count=0, mode="mantic", forecaster_class=forecaster_class):
            cli_main()
        client = forecaster_class.call_args.kwargs["metaculus_client"]
        assert isinstance(client, ManticClient)
        assert client.base_url == MANTIC_API_BASE_URL

    @pytest.mark.parametrize("run_mode", sorted(set(get_args(RunMode)) - {"mantic"}))
    def test_metaculus_modes_leave_the_framework_default_client(self, run_mode: RunMode) -> None:
        forecaster_class = _forecaster_class()
        with _cli_main_test_mode(alertable_count=0, mode=run_mode, forecaster_class=forecaster_class):
            cli_main()
        assert forecaster_class.call_args.kwargs["metaculus_client"] is None

    def test_fail_shut_runs_before_the_token_is_read_or_the_host_vetted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The switch at its default: the guard raises, so no preflight GET, token read, forecaster or spend."""
        monkeypatch.delenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, raising=False)
        monkeypatch.setenv(MANTIC_TOKEN_ENV, _FAKE_MANTIC_TOKEN)
        forecaster_class = _forecaster_class()
        with (
            _cli_main_test_mode(alertable_count=0, mode="mantic", forecaster_class=forecaster_class),
            patch("metaculus_bot.cli.build_mantic_client") as build_client,
            patch("metaculus_bot.cli.verify_api_identity") as verify_api,
            pytest.raises(RuntimeError, match=DONATED_OPENROUTER_KEY_ENABLED_ENV),
        ):
            cli_main()
        build_client.assert_not_called()
        verify_api.assert_not_called()
        forecaster_class.assert_not_called()

    def test_the_client_is_built_after_the_guard_and_the_preflight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ordered, not just "each was called": guard, then preflight, then the client (which reads
        the token), then the forecaster. Two independent ``assert_called`` checks would pass with
        the token read before the host was vetted."""
        _mantic_env(monkeypatch)
        manager = MagicMock()
        forecaster_class = _forecaster_class()
        with (
            _cli_main_test_mode(alertable_count=0, mode="mantic", forecaster_class=forecaster_class),
            patch("metaculus_bot.cli._assert_personal_keys_only") as guard,
            patch("metaculus_bot.cli.verify_api_identity") as verify_api,
            patch("metaculus_bot.cli.build_mantic_client") as build_client,
        ):
            manager.attach_mock(guard, "guard")
            manager.attach_mock(verify_api, "verify_api_identity")
            manager.attach_mock(build_client, "build_mantic_client")
            manager.attach_mock(forecaster_class, "TemplateForecaster")
            cli_main()

        call_names = [name for name, _, _ in manager.mock_calls]
        order = [
            call_names.index(name)
            for name in ("guard", "verify_api_identity", "build_mantic_client", "TemplateForecaster")
        ]
        assert order == sorted(order), call_names

    def test_the_tournament_preflight_runs_on_the_built_client_before_the_forecaster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ordered like the test above: client built (token read), then the authenticated tournament
        GET on THAT client, then the forecaster. Nothing has been spent when the preflight raises."""
        _mantic_env(monkeypatch)
        manager = MagicMock()
        forecaster_class = _forecaster_class()
        with (
            _cli_main_test_mode(alertable_count=0, mode="mantic", forecaster_class=forecaster_class),
            patch("metaculus_bot.cli.build_mantic_client") as build_client,
            patch("metaculus_bot.cli.preflight_mantic_tournaments") as preflight,
        ):
            manager.attach_mock(build_client, "build_mantic_client")
            manager.attach_mock(preflight, "preflight_mantic_tournaments")
            manager.attach_mock(forecaster_class, "TemplateForecaster")
            cli_main()

        call_names = [name for name, _, _ in manager.mock_calls]
        order = [
            call_names.index(name)
            for name in ("build_mantic_client", "preflight_mantic_tournaments", "TemplateForecaster")
        ]
        assert order == sorted(order), call_names
        preflight.assert_called_once_with(build_client.return_value, MANTIC_TOURNAMENT_ID)

    def test_a_failed_tournament_preflight_stops_before_the_forecaster_is_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mantic_env(monkeypatch)
        forecaster_class = _forecaster_class()
        with (
            _cli_main_test_mode(alertable_count=0, mode="mantic", forecaster_class=forecaster_class),
            patch("metaculus_bot.cli.preflight_mantic_tournaments", side_effect=ApiIdentityError("viewer")),
            pytest.raises(ApiIdentityError, match="viewer"),
        ):
            cli_main()
        forecaster_class.assert_not_called()

    @pytest.mark.parametrize("run_mode", sorted(set(get_args(RunMode)) - {"mantic"}))
    def test_metaculus_modes_never_run_the_tournament_preflight(self, run_mode: RunMode) -> None:
        with (
            _cli_main_test_mode(alertable_count=0, mode=run_mode),
            patch("metaculus_bot.cli.preflight_mantic_tournaments") as preflight,
        ):
            cli_main()
        preflight.assert_not_called()


def test_metaculus_mode_never_builds_direct_anthropic_forecasters(monkeypatch: pytest.MonkeyPatch) -> None:
    """The personal Anthropic key is Mantic-only; a Metaculus run keeps the OpenRouter roster."""
    monkeypatch.setenv("personal_ant_key", "test-key")
    with _cli_main_test_mode(alertable_count=0), patch("metaculus_bot.cli.mantic_forecasters") as direct:
        cli_main()
    direct.assert_not_called()
