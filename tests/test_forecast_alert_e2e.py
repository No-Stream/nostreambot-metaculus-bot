"""Runtime status and GitHub notification tests through the forecast CLI.

These tests run the real Mantic CLI and forecast pipeline. LLM, Mantic HTTP, and research
provider transport boundaries are deterministic fixtures, so no request reaches a paid API
or publishes outside the fake transport. Each completed CLI run is then passed to the actual
standalone GitHub summary renderer.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from urllib.parse import urlparse

import pytest
import requests
from forecasting_tools.ai_models import general_llm as general_llm_module
from requests.adapters import HTTPAdapter

from metaculus_bot import cli
from metaculus_bot import forecaster as forecaster_module
from metaculus_bot.api_preflight import ApiIdentityError
from metaculus_bot.constants import DONATED_OPENROUTER_KEY_ENABLED_ENV, MANTIC_TOKEN_ENV
from metaculus_bot.research.market_retrieval.venues import POLYMARKET_MAX_ATTEMPTS
from metaculus_bot.run_status import RUN_STATUS
from tests import mantic_e2e_harness as mantic
from tests import test_offline_e2e_forecast as offline
from tests.http_fakes import json_response

SUMMARY_SCRIPT = Path(__file__).parents[1] / "scripts" / "forecast_run_summary.py"
_PRIVATE_TOKEN = "fixture-token-must-not-be-persisted-123456"


def _install_cli_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    full_research: bool = False,
    model_timeout: bool = False,
    failed_comment: bool = False,
    polymarket_forbidden: bool = False,
    manifold_detail_partial: bool = False,
) -> tuple[Path, list[mantic.RecordedRequest], list[str], list[tuple[str, int]]]:
    """Run one selected Mantic question through cli.main with offline external boundaries."""
    status_path = tmp_path / "run_logs" / "status.json"
    monkeypatch.setenv("FORECAST_RUN_STATUS_PATH", str(status_path))
    monkeypatch.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, "false")
    monkeypatch.setenv(MANTIC_TOKEN_ENV, _PRIVATE_TOKEN)
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "mantic", "--only-posts", str(mantic.BINARY_POST_ID)])

    offline._install_env(monkeypatch)
    monkeypatch.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, "false")
    offline._install_llm_router(monkeypatch)

    if not full_research:

        async def stub_research(self: Any, question: Any, time_budget: Any = None) -> str:
            del self, question, time_budget
            return "## Research Summary\n\nOffline end-to-end fixture research.\n"

        monkeypatch.setattr(offline.TemplateForecaster, "run_research", stub_research)
    else:
        offline._install_provider_stubs(monkeypatch)

    if model_timeout:
        routed_acompletion = general_llm_module.acompletion
        timeout_used = False
        real_sleep = asyncio.sleep
        monkeypatch.setattr(forecaster_module, "FORECASTER_SOFT_DEADLINE", 5.0)

        async def timeout_one_forecaster(**kwargs: Any) -> Any:
            nonlocal timeout_used
            prompt_text = offline._messages_text(kwargs)
            if "STRUCTURED FORECAST" in prompt_text and not timeout_used:
                timeout_used = True
                await real_sleep(10.0)
            return await routed_acompletion(**kwargs)

        monkeypatch.setattr(general_llm_module, "acompletion", timeout_one_forecaster)

    posts = [post for post in mantic.future_dated_posts() if post["id"] == mantic.BINARY_POST_ID]
    assert len(posts) == 1

    polymarket_queries: list[str] = []
    manifold_detail_responses: list[tuple[str, int]] = []
    if polymarket_forbidden:
        base_session = offline._FakeHttpSession
        query_attempts: dict[str, int] = {}
        first_failed_query: str | None = None

        class PartialPolymarketSession(base_session):
            def get(self, url: str, **kwargs: Any) -> offline._FakeHttpResponse:
                if "polymarket" in url.lower():
                    nonlocal first_failed_query
                    query = str(kwargs["params"]["q"])
                    polymarket_queries.append(query)
                    query_attempts[query] = query_attempts.get(query, 0) + 1
                    if first_failed_query is None:
                        first_failed_query = query
                    if query == first_failed_query and query_attempts[query] <= POLYMARKET_MAX_ATTEMPTS:
                        return offline._FakeHttpResponse(403, body=b"credential details stay private")
                return super().get(url, **kwargs)

        monkeypatch.setattr(offline.prediction_market, "_get_session", PartialPolymarketSession)

    if manifold_detail_partial:
        base_session = offline._FakeHttpSession
        search_markets = json.loads(offline._OFF_TOPIC_MANIFOLD_SEARCH)
        second_market = dict(search_markets[0])
        second_market.update(
            {
                "id": "wc26argentina",
                "question": "Will Argentina win the 2026 FIFA World Cup?",
                "slug": "world-cup-2026-argentina",
            }
        )
        search_markets.append(second_market)
        failed_market_id = "wc26brazil"
        failed_market_ids: set[str] = set()

        class PartialManifoldDetailSession(base_session):
            def get(self, url: str, **kwargs: Any) -> offline._FakeHttpResponse:
                path = urlparse(url).path
                if path == "/v0/search-markets":
                    return offline._FakeHttpResponse(200, body=json.dumps(search_markets).encode())
                if path.startswith("/v0/market/"):
                    market_id = path.rsplit("/", maxsplit=1)[-1]
                    if market_id == failed_market_id and market_id not in failed_market_ids:
                        failed_market_ids.add(market_id)
                        manifold_detail_responses.append((market_id, 503))
                        return offline._FakeHttpResponse(503, body=b"temporary detail outage")
                    detail = json.loads(offline._MANIFOLD_MARKET_DETAIL)
                    detail["id"] = market_id
                    manifold_detail_responses.append((market_id, 200))
                    return offline._FakeHttpResponse(200, body=json.dumps(detail).encode())
                return super().get(url, **kwargs)

        monkeypatch.setattr(offline.prediction_market, "_get_session", PartialManifoldDetailSession)

    if failed_comment:
        original_json_response = mantic.json_response

        def fail_comment_response(
            payload: Any, *, status: int = 200, request: requests.PreparedRequest
        ) -> requests.Response:
            if urlparse(request.url or "").path.endswith("/comments/create/"):
                return original_json_response(payload, status=503, request=request)
            return original_json_response(payload, status=status, request=request)

        monkeypatch.setattr(mantic, "json_response", fail_comment_response)

    recorded_requests = mantic.install_fake_transport(monkeypatch, posts)

    # Identity validation remains on the CLI path; these are its external API boundaries.
    monkeypatch.setattr(cli, "verify_api_identity", lambda _base_url: None)
    monkeypatch.setattr(cli, "preflight_mantic_tournaments", lambda _client, _tournament_id: None)
    monkeypatch.setattr(cli, "_check_tournament_dates", lambda _run_mode: False)
    monkeypatch.setattr(cli, "check_fall_cup_reminder", lambda _logger: False)
    RUN_STATUS.reset()
    return status_path, recorded_requests, polymarket_queries, manifold_detail_responses


def _run_cli() -> int:
    try:
        cli.main()
    except SystemExit as error:
        return int(error.code or 0)
    return 0


def _read_runtime_status(status_path: Path) -> dict[str, Any]:
    assert status_path.is_file(), "cli.main must persist status even when its exit policy is non-zero"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert _PRIVATE_TOKEN not in json.dumps(status)
    assert "credential details stay private" not in json.dumps(status)
    return status


def _render_notification(tmp_path: Path, status_path: Path, *, outcome: str) -> tuple[str, str, str]:
    summary_path = tmp_path / "job-summary.md"
    output_path = tmp_path / "job-output.txt"
    env = {
        **os.environ,
        "FORECAST_RUN_STATUS_PATH": str(status_path),
        "GITHUB_STEP_SUMMARY": str(summary_path),
        "GITHUB_OUTPUT": str(output_path),
        "FORECAST_JOB_STATUS": "success" if outcome == "success" else "failure",
        "BOT_STEP_OUTCOME": outcome,
        "PLAYWRIGHT_STEP_OUTCOME": "success",
    }
    rendered = subprocess.run(
        [sys.executable, str(SUMMARY_SCRIPT), "--status", str(status_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert rendered.returncode == 0, rendered.stderr
    return (
        summary_path.read_text(encoding="utf-8"),
        output_path.read_text(encoding="utf-8"),
        rendered.stdout,
    )


@pytest.mark.e2e
def test_clean_cli_run_persists_success_and_renders_notification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status_path, recorded_requests, _, _ = _install_cli_harness(monkeypatch, tmp_path)

    assert _run_cli() == 0

    status = _read_runtime_status(status_path)
    assert status["outcome"] == "clean"
    assert status["models"]["attempted"] == status["models"]["succeeded"] == 3
    assert status["publishing"]["forecasts"] == {"attempted": 1, "succeeded": 1, "failed": 0}
    assert status["publishing"]["comments"] == {"attempted": 1, "succeeded": 1, "failed": 0}
    assert len([request for request in recorded_requests if request.path == mantic._FORECAST_PATH]) == 1
    assert len([request for request in recorded_requests if request.path == mantic._COMMENT_PATH]) == 1

    summary, output, annotation = _render_notification(tmp_path, status_path, outcome="success")
    assert "Forecast run: Clean" in summary
    assert "Models: 3/3 succeeded" in summary
    assert "Forecasts: 1/1 succeeded" in summary
    assert "Comments: 1/1 succeeded" in summary
    assert "title=Published OK" in output
    assert "::error" not in annotation


@pytest.mark.e2e
def test_partial_polymarket_loss_still_publishes_and_keeps_cli_red(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status_path, recorded_requests, polymarket_queries, _ = _install_cli_harness(
        monkeypatch, tmp_path, full_research=True, polymarket_forbidden=True
    )

    assert _run_cli() == 1

    status = _read_runtime_status(status_path)
    assert status["outcome"] == "degraded"
    polymarket_causes = [cause for cause in status["causes"] if cause["component"] == "polymarket"]
    assert polymarket_causes
    lookup_causes = [cause for cause in polymarket_causes if cause["unit"] == "lookups"]
    source_check_causes = [cause for cause in polymarket_causes if cause["unit"] == "source_checks"]
    assert len(lookup_causes) == 1
    assert lookup_causes[0]["http_status"] == 403
    assert lookup_causes[0]["attempted"] == len(set(polymarket_queries))
    assert lookup_causes[0]["attempted"] > lookup_causes[0]["affected"] > 0
    assert any(cause["error_type"] == "provider_failure" for cause in source_check_causes)
    assert status["publishing"]["forecasts"]["succeeded"] == 1
    assert status["publishing"]["comments"]["succeeded"] == 1
    assert any(request.path == mantic._FORECAST_PATH for request in recorded_requests)
    assert any(request.path == mantic._COMMENT_PATH for request in recorded_requests)

    summary, output, annotation = _render_notification(tmp_path, status_path, outcome="failure")
    assert "Polymarket HTTP 403" in summary
    assert "lookups affected" in summary
    assert "source checks affected" in summary
    assert "3/3 succeeded" in summary
    assert "Forecasts: 1/1 succeeded" in summary
    assert "Comments: 1/1 succeeded" in summary
    assert "title=Published OK; Polymarket HTTP 403" in output
    assert "::error title=Forecast run degraded" in annotation


@pytest.mark.e2e
def test_partial_manifold_detail_failure_remains_clean_and_renders_known_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status_path, recorded_requests, _, detail_responses = _install_cli_harness(
        monkeypatch, tmp_path, full_research=True, manifold_detail_partial=True
    )

    assert _run_cli() == 0

    status = _read_runtime_status(status_path)
    assert status["outcome"] == "clean"
    assert all(cause["component"] != "manifold_detail" for cause in status["causes"])
    # The offline resolution source fetch succeeds with a bare "ok" token, which is healthy.
    assert all(cause["component"] != "resolution_source" for cause in status["causes"])
    assert ("wc26brazil", 503) in detail_responses
    assert any(status_code == 200 and market_id != "wc26brazil" for market_id, status_code in detail_responses)
    assert status["models"]["attempted"] == status["models"]["succeeded"] == 3
    assert status["publishing"]["forecasts"] == {"attempted": 1, "succeeded": 1, "failed": 0}
    assert status["publishing"]["comments"] == {"attempted": 1, "succeeded": 1, "failed": 0}
    assert any(request.path == mantic._FORECAST_PATH for request in recorded_requests)
    assert any(request.path == mantic._COMMENT_PATH for request in recorded_requests)

    summary, output, annotation = _render_notification(tmp_path, status_path, outcome="success")
    assert "Forecast run: Clean" in summary
    assert "Resolution source provider failure" not in summary
    assert "Models: 3/3 succeeded" in summary
    assert "Forecasts: 1/1 succeeded" in summary
    assert "Comments: 1/1 succeeded" in summary
    assert "title=Published OK" in output
    assert "::error" not in annotation


@pytest.mark.e2e
def test_empty_asknews_source_check_remains_clean_under_existing_alert_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status_path, recorded_requests, _, _ = _install_cli_harness(monkeypatch, tmp_path, full_research=True)
    asknews_search_calls = 0

    async def return_no_articles(*_args: Any, **_kwargs: Any) -> offline._FakeAskNewsResponse:
        nonlocal asknews_search_calls
        asknews_search_calls += 1
        return offline._FakeAskNewsResponse([])

    class EmptyAskNewsSDK(offline._FakeAskNewsSDK):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.news.search_news = AsyncMock(side_effect=return_no_articles)

    monkeypatch.setattr(offline.asknews_sdk, "AsyncAskNewsSDK", EmptyAskNewsSDK)

    assert _run_cli() == 0

    status = _read_runtime_status(status_path)
    assert status["outcome"] == "clean"
    assert asknews_search_calls >= 2  # The real AskNews provider runs hot and historical searches.
    empty_asknews_checks = [
        cause
        for cause in status["causes"]
        if cause["component"] == "asknews"
        and cause["unit"] == "source_checks"
        and cause["error_type"] == "provider_failure"
    ]
    assert len(empty_asknews_checks) == 1
    assert empty_asknews_checks[0]["affected"] == empty_asknews_checks[0]["attempted"] == 1
    assert status["models"]["attempted"] == status["models"]["succeeded"] == 3
    assert status["publishing"]["forecasts"] == {"attempted": 1, "succeeded": 1, "failed": 0}
    assert status["publishing"]["comments"] == {"attempted": 1, "succeeded": 1, "failed": 0}
    assert any(request.path == mantic._FORECAST_PATH for request in recorded_requests)
    assert any(request.path == mantic._COMMENT_PATH for request in recorded_requests)

    summary, output, annotation = _render_notification(tmp_path, status_path, outcome="success")
    assert "Forecast run: Clean" in summary
    assert "AskNews provider failure; 1/1 source checks affected" in summary
    assert "Clean under existing alert policy" in summary
    assert "title=Published OK" in output
    assert "::error" not in annotation


@pytest.mark.e2e
def test_model_timeout_keeps_survivors_publishing_and_cli_red(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    status_path, recorded_requests, _, _ = _install_cli_harness(monkeypatch, tmp_path, model_timeout=True)

    assert _run_cli() == 1

    status = _read_runtime_status(status_path)
    assert status["outcome"] == "degraded"
    assert status["models"]["attempted"] == 3
    assert status["models"]["succeeded"] == 2
    assert status["models"]["timed_out"] == 1
    assert status["publishing"]["forecasts"]["succeeded"] == 1
    assert status["publishing"]["comments"]["succeeded"] == 1
    assert any(request.path == mantic._FORECAST_PATH for request in recorded_requests)
    assert any(request.path == mantic._COMMENT_PATH for request in recorded_requests)

    summary, output, annotation = _render_notification(tmp_path, status_path, outcome="failure")
    assert "Models: 2/3 succeeded, 1 timed out, 1 dropped" in summary
    assert "Forecasts: 1/1 succeeded" in summary
    assert "Comments: 1/1 succeeded" in summary
    assert "1 model timed out" in output
    assert "::error" in annotation


@pytest.mark.e2e
def test_comment_failure_after_forecast_success_is_reported_and_cli_red(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status_path, recorded_requests, _, _ = _install_cli_harness(monkeypatch, tmp_path, failed_comment=True)

    with pytest.raises(RuntimeError, match="errors occurred while forecasting"):
        cli.main()

    status = _read_runtime_status(status_path)
    assert status["outcome"] == "hard_failure"
    assert status["publishing"]["forecasts"] == {"attempted": 1, "succeeded": 1, "failed": 0}
    assert status["publishing"]["comments"]["attempted"] == 1
    assert status["publishing"]["comments"]["succeeded"] == 0
    assert status["publishing"]["comments"]["failed"] == 1
    forecast_position = next(
        index for index, request in enumerate(recorded_requests) if request.path == mantic._FORECAST_PATH
    )
    comment_position = next(
        index for index, request in enumerate(recorded_requests) if request.path == mantic._COMMENT_PATH
    )
    assert forecast_position < comment_position

    summary, output, annotation = _render_notification(tmp_path, status_path, outcome="failure")
    assert "Forecasts: 1/1 succeeded" in summary
    assert "Comments: 0/1 succeeded, 1 failed" in summary
    assert "Publication failed" in output
    assert "::error" in annotation


@pytest.mark.e2e
def test_invalid_mantic_credentials_write_setup_failure_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    status_path = tmp_path / "run_logs" / "status.json"
    monkeypatch.setenv("FORECAST_RUN_STATUS_PATH", str(status_path))
    monkeypatch.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, "false")
    monkeypatch.setenv(MANTIC_TOKEN_ENV, _PRIVATE_TOKEN)
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "mantic"])
    RUN_STATUS.reset()
    requests_seen: list[requests.PreparedRequest] = []

    def reject_authenticated_tournament_list(
        _adapter: HTTPAdapter, request: requests.PreparedRequest, **_kwargs: Any
    ) -> requests.Response:
        requests_seen.append(request)
        path = urlparse(request.url or "").path
        if path.endswith("/posts/"):
            assert "Authorization" not in request.headers
            return json_response({"results": []}, request=request)
        if path.endswith("/projects/tournaments/"):
            assert request.headers["Authorization"] == f"Token {_PRIVATE_TOKEN}"
            return json_response({"detail": "invalid token"}, status=401, request=request)
        raise AssertionError(f"unexpected setup request: {request.method} {request.url}")

    monkeypatch.setattr(HTTPAdapter, "send", reject_authenticated_tournament_list)

    with pytest.raises(ApiIdentityError, match="status=401"):
        cli.main()

    assert len(requests_seen) == 2
    status = _read_runtime_status(status_path)
    assert status["outcome"] == "hard_failure"
    assert any(
        cause["component"] == "setup" and cause["error_type"] == "authentication_error" for cause in status["causes"]
    )

    summary, output, annotation = _render_notification(tmp_path, status_path, outcome="failure")
    assert "Forecast run: Hard Failure" in summary
    assert "Setup authentication error" in summary
    assert "title=Failed; Setup authentication error" in output
    assert "::error" in annotation
