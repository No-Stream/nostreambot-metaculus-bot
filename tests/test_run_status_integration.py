from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from metaculus_bot import cli as forecast_cli
from metaculus_bot.cli import _record_cli_alert_causes
from metaculus_bot.forecaster import TemplateForecaster
from metaculus_bot.research.market_retrieval import http as market_http
from metaculus_bot.research.market_retrieval.http import http_get_with_backoff
from metaculus_bot.research.market_retrieval.session_state import (
    _predictit_universe,
    _reset_session_caches,
)
from metaculus_bot.research.market_retrieval.snapshot_stages import _platform_source_token
from metaculus_bot.run_status import RUN_STATUS

_RENDERER_PATH = Path(__file__).parents[1] / "scripts" / "forecast_run_summary.py"


def _mock_credit_fallback_census(monkeypatch: pytest.MonkeyPatch, *, alerts_active: bool) -> None:
    monkeypatch.setattr(forecast_cli, "credit_alerts_active", lambda: alerts_active)
    monkeypatch.setattr(forecast_cli, "get_generic_key_fallback_count", lambda: 1)
    monkeypatch.setattr(forecast_cli, "get_donated_404_fallback_count", lambda: 0)
    monkeypatch.setattr(forecast_cli, "get_credit_key_fallback_count", lambda: 1)
    monkeypatch.setattr(forecast_cli, "get_post_drop_count", lambda: 0)
    monkeypatch.setattr(forecast_cli, "get_probed_donated_key_state", lambda: None)
    monkeypatch.setattr(forecast_cli, "has_deprecation_alerts", lambda: False)
    monkeypatch.setattr(forecast_cli, "check_deprecation_alerts_and_exit", lambda: None)


def _record_successful_model_and_publication_counts() -> None:
    RUN_STATUS.record_model_attempt(3)
    RUN_STATUS.record_model_success(3)
    RUN_STATUS.record_publish_attempt("forecasts")
    RUN_STATUS.record_publish_success("forecasts")
    RUN_STATUS.record_publish_attempt("comments")
    RUN_STATUS.record_publish_success("comments")


def _render_runtime_status(tmp_path: Path, *, bot_outcome: str) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    summary_path = tmp_path / "step-summary.md"
    output_path = tmp_path / "step-output.txt"
    environment = {
        **os.environ,
        "GITHUB_STEP_SUMMARY": str(summary_path),
        "GITHUB_OUTPUT": str(output_path),
        "FORECAST_JOB_STATUS": "success" if bot_outcome == "success" else "failure",
        "BOT_STEP_OUTCOME": bot_outcome,
        "PLAYWRIGHT_STEP_OUTCOME": "success",
    }
    result = subprocess.run(
        [sys.executable, str(_RENDERER_PATH), "--status", str(tmp_path / "status.json")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return result, summary_path, output_path


def _write_status_for_renderer(tmp_path: Path) -> None:
    RUN_STATUS.write(tmp_path / "status.json")


def test_source_tokens_render_clean_healthy_run_and_report_a_lost_source(tmp_path: Path) -> None:
    RUN_STATUS.reset(stage="runtime")
    for provider_name, sources in (
        ("resolution_source", {"cmegroup.com": "ok"}),
        ("financial_data", {"CL=F": "ok"}),
        ("prediction_market", {"polymarket": "ok(3)"}),
    ):
        RUN_STATUS.record_provider_result(
            name=provider_name,
            status="ok",
            error_type=None,
            details={"sources": sources},
        )
    _record_successful_model_and_publication_counts()
    RUN_STATUS.set_outcome("clean")

    _write_status_for_renderer(tmp_path)
    healthy_result, healthy_summary_path, healthy_output_path = _render_runtime_status(tmp_path, bot_outcome="success")
    healthy_summary = healthy_summary_path.read_text(encoding="utf-8")

    assert healthy_result.returncode == 0
    assert "title=Published OK\n" in healthy_output_path.read_text(encoding="utf-8")
    assert "provider failure" not in healthy_summary
    assert "source checks affected" not in healthy_summary
    assert "- Cause: none recorded." in healthy_summary

    RUN_STATUS.reset(stage="runtime")
    RUN_STATUS.record_provider_result(
        name="resolution_source",
        status="ok",
        error_type=None,
        details={"sources": {"cmegroup.com": "error(ungrounded_suppressed)"}},
    )
    _record_successful_model_and_publication_counts()
    RUN_STATUS.set_outcome("degraded")

    _write_status_for_renderer(tmp_path)
    degraded_result, degraded_summary_path, degraded_output_path = _render_runtime_status(
        tmp_path, bot_outcome="failure"
    )
    degraded_summary = degraded_summary_path.read_text(encoding="utf-8")

    assert degraded_result.returncode == 0
    assert "Published OK; Resolution source provider failure" in degraded_output_path.read_text(encoding="utf-8")
    assert "Resolution source provider failure; 1/1 source checks affected" in degraded_summary


def test_suppressed_credit_fallback_stays_green_in_exit_ladder_and_renderer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    RUN_STATUS.reset(stage="runtime")
    _record_successful_model_and_publication_counts()
    _mock_credit_fallback_census(monkeypatch, alerts_active=False)
    template_bot = MagicMock(spec=TemplateForecaster, alertable_count=0)

    with caplog.at_level(logging.INFO, logger="metaculus_bot.cli"):
        forecast_cli._report_degradation_and_exit(
            template_bot,
            report_summary_error=None,
            donated_below_floor=False,
            fall_cup_reminder=False,
            mantic_tournament_stale=False,
        )

    expected_census = (
        "Run completed with 0 alertable degradation event(s) "
        "(bot=0, personal_key_fallback=1 of which donated_404=0, credit=1 with 1 credit event(s) suppressed "
        f"until {forecast_cli.CREDIT_ALERT_RESUME_DATE.isoformat()}); "
        "every fallback was a suppressed credit event, so this run stays green."
    )
    assert expected_census in [record.getMessage() for record in caplog.records]
    assert RUN_STATUS.payload()["outcome"] == "clean"
    assert RUN_STATUS.payload()["causes"] == []

    _write_status_for_renderer(tmp_path)
    result, summary_path, output_path = _render_runtime_status(tmp_path, bot_outcome="success")

    assert result.returncode == 0
    assert "Forecast run: Clean" in summary_path.read_text(encoding="utf-8")
    assert "::error" not in result.stdout
    assert "title=Published OK" in output_path.read_text(encoding="utf-8")


def test_unsuppressed_credit_fallback_stays_red_in_exit_ladder_and_renderer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    RUN_STATUS.reset(stage="runtime")
    _record_successful_model_and_publication_counts()
    _mock_credit_fallback_census(monkeypatch, alerts_active=True)
    template_bot = MagicMock(spec=TemplateForecaster, alertable_count=0)

    with pytest.raises(SystemExit) as exit_info:
        forecast_cli._report_degradation_and_exit(
            template_bot,
            report_summary_error=None,
            donated_below_floor=False,
            fall_cup_reminder=False,
            mantic_tournament_stale=False,
        )

    assert exit_info.value.code == 1
    assert RUN_STATUS.payload()["outcome"] == "degraded"
    assert RUN_STATUS.payload()["causes"] == [
        {
            "component": "credit",
            "unit": "alerts",
            "affected": 1,
            "attempted": None,
            "error_type": "provider_failure",
            "http_status": None,
        }
    ]

    _write_status_for_renderer(tmp_path)
    result, summary_path, output_path = _render_runtime_status(tmp_path, bot_outcome="failure")

    assert result.returncode == 0
    assert "Forecast run: Degraded" in summary_path.read_text(encoding="utf-8")
    assert "Credits provider failure" in summary_path.read_text(encoding="utf-8")
    assert "::error" in result.stdout
    assert "Credits provider failure" in output_path.read_text(encoding="utf-8")


class _Response:
    def __init__(self, status: int, body: bytes = b"") -> None:
        self.status = status
        self.content = self._Content(body)

    class _Content:
        def __init__(self, body: bytes) -> None:
            self.body = body

        async def iter_chunked(self, _max_bytes: int) -> AsyncIterator[bytes]:
            if self.body:
                yield self.body

    async def __aenter__(self) -> _Response:
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _Session:
    def __init__(self, *, status: int = 403, body: bytes = b"") -> None:
        self.calls = 0
        self.status = status
        self.body = body

    def get(self, *args: object, **kwargs: object) -> _Response:
        self.calls += 1
        return _Response(self.status, self.body)


@pytest.mark.asyncio
async def test_polymarket_http_failure_retains_source_and_status_without_query_text() -> None:
    RUN_STATUS.reset()

    result = await http_get_with_backoff(
        _Session(),
        "https://example.invalid/search?q=private query",
        {"q": "private question"},
        max_attempts=1,
        label="Polymarket q='private question'",
        component="polymarket",
    )

    assert result is None
    assert RUN_STATUS.payload()["causes"] == [
        {
            "component": "polymarket",
            "unit": "lookups",
            "affected": 1,
            "attempted": 1,
            "error_type": "http_error",
            "http_status": 403,
        }
    ]
    assert "private" not in str(RUN_STATUS.payload())


def test_legacy_alert_does_not_infer_attempt_count() -> None:
    RUN_STATUS.reset()
    RUN_STATUS.record_legacy_cause("polymarket", error_type="provider_failure", affected=1)

    assert RUN_STATUS.payload()["causes"][0]["attempted"] is None
    assert RUN_STATUS.payload()["causes"][0]["unit"] == "alerts"


def test_suppressed_credit_floor_does_not_make_a_clean_run_look_degraded() -> None:
    RUN_STATUS.reset()
    _record_cli_alert_causes(
        generic_fallback=0,
        donated_below_floor=True,
        alerts_active=False,
        fall_cup_reminder=False,
        mantic_tournament_stale=False,
        mantic_post_drops=0,
        deprecation_alerts=False,
    )

    assert RUN_STATUS.payload()["causes"] == []


@pytest.mark.asyncio
async def test_singleflight_predictit_failure_has_one_transport_cause_for_three_questions(monkeypatch) -> None:
    RUN_STATUS.reset()
    _reset_session_caches()
    monkeypatch.setattr(market_http, "HTTP_RETRY_BACKOFF_SECS", 0)
    session = _Session()

    outcomes = await asyncio.gather(*(_predictit_universe(session, qid=qid) for qid in (11, 12, 13)))
    for _markets, tally in outcomes:
        source_token = _platform_source_token([], tally)
        RUN_STATUS.record_provider_result(
            name="prediction_market",
            status="ok",
            error_type=None,
            details={"sources": {"predictit": source_token}},
        )

    assert session.calls == 2  # two wire retries, one logical PredictIt lookup
    assert RUN_STATUS.payload()["causes"] == [
        {
            "component": "predictit",
            "unit": "lookups",
            "affected": 1,
            "attempted": 1,
            "error_type": "http_error",
            "http_status": 403,
        },
        {
            "component": "predictit",
            "unit": "source_checks",
            "affected": 3,
            "attempted": 3,
            "error_type": "provider_failure",
            "http_status": None,
        },
    ]


@pytest.mark.asyncio
async def test_predictit_http_200_wrong_shape_is_visible_from_real_source_token(monkeypatch) -> None:
    RUN_STATUS.reset()
    _reset_session_caches()
    monkeypatch.setattr(market_http, "HTTP_RETRY_BACKOFF_SECS", 0)
    session = _Session(status=200, body=b'{"events": []}')

    outcomes = await asyncio.gather(*(_predictit_universe(session, qid=qid) for qid in (21, 22, 23)))
    assert [markets for markets, _tally in outcomes] == [[], [], []]
    for _markets, tally in outcomes:
        source_token = _platform_source_token([], tally)
        RUN_STATUS.record_provider_result(
            name="prediction_market",
            status="ok",
            error_type=None,
            details={"sources": {"predictit": source_token}},
        )

    assert session.calls == 1
    assert RUN_STATUS.payload()["causes"] == [
        {
            "component": "predictit",
            "unit": "source_checks",
            "affected": 3,
            "attempted": 3,
            "error_type": "provider_failure",
            "http_status": None,
        }
    ]
