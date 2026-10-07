from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from metaculus_bot.run_status import COUNT_UNITS, STATUS_COMPONENTS, STATUS_DROP_CAUSES, STATUS_ERROR_TYPES
from scripts import forecast_run_summary

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "forecast_run_summary.py"
DEGRADED_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "forecast_alert_degraded.json"


def _status_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "outcome": "clean",
        "causes": [],
        "models": {
            "attempted": 3,
            "succeeded": 3,
            "timed_out": 0,
            "dropped": 0,
            "drop_causes": {},
        },
        "publishing": {
            "forecasts": {"attempted": 1, "succeeded": 1, "failed": 0},
            "comments": {"attempted": 1, "succeeded": 1, "failed": 0},
        },
    }


def test_renderer_allowlists_match_runtime_status_contract() -> None:
    assert set(forecast_run_summary.COMPONENT_LABELS) == STATUS_COMPONENTS
    assert set(forecast_run_summary.ERROR_LABELS) == STATUS_ERROR_TYPES
    assert forecast_run_summary.DROP_CAUSES == STATUS_DROP_CAUSES
    assert set(forecast_run_summary.COUNT_UNIT_LABELS) == COUNT_UNITS


def _run_renderer(
    tmp_path: Path,
    *,
    status: dict[str, object] | str | None,
    bot_outcome: str = "success",
    forecast_job_status: str = "success",
    playwright_outcome: str = "success",
) -> subprocess.CompletedProcess[str]:
    status_path = tmp_path / "status.json"
    if isinstance(status, dict):
        status_path.write_text(json.dumps(status), encoding="utf-8")
    elif isinstance(status, str):
        status_path.write_text(status, encoding="utf-8")

    summary_path = tmp_path / "step-summary.md"
    output_path = tmp_path / "step-output.txt"
    env = {
        **os.environ,
        "GITHUB_STEP_SUMMARY": str(summary_path),
        "GITHUB_OUTPUT": str(output_path),
        "FORECAST_JOB_STATUS": forecast_job_status,
        "BOT_STEP_OUTCOME": bot_outcome,
        "PLAYWRIGHT_STEP_OUTCOME": playwright_outcome,
    }
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--status",
            str(status_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_clean_run_writes_job_summary_and_compact_output(tmp_path: Path) -> None:
    result = _run_renderer(tmp_path, status=_status_payload())

    assert result.returncode == 0
    assert "::error" not in result.stdout
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    assert "Forecast run: Clean" in summary
    assert "Models: 3/3 succeeded" in summary
    assert "Forecasts: 1/1 succeeded" in summary
    assert "Comments: 1/1 succeeded" in summary
    assert "title=Published OK" in (tmp_path / "step-output.txt").read_text(encoding="utf-8")


def test_clean_outcome_keeps_tolerated_source_loss_without_changing_exit(tmp_path: Path) -> None:
    status = _status_payload()
    status["causes"] = [
        {
            "component": "resolution_source",
            "unit": "source_checks",
            "affected": 1,
            "attempted": 3,
            "error_type": "provider_failure",
            "http_status": 200,
        }
    ]

    result = _run_renderer(tmp_path, status=status)

    assert result.returncode == 0
    assert "::error" not in result.stdout
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    assert "Forecast run: Clean under existing alert policy" in summary
    assert "Resolution source provider failure (HTTP 200); 1/3 source checks affected" in summary
    assert "Published OK; Resolution source provider failure (HTTP 200)" in (tmp_path / "step-output.txt").read_text(
        encoding="utf-8"
    )


def test_degraded_semantic_http_200_failure_is_clear_in_job_title(tmp_path: Path) -> None:
    status = _status_payload()
    status["outcome"] = "degraded"
    status["causes"] = [
        {
            "component": "resolution_source",
            "unit": "source_checks",
            "affected": 1,
            "attempted": 3,
            "error_type": "provider_failure",
            "http_status": 200,
        }
    ]

    result = _run_renderer(tmp_path, status=status)

    assert result.returncode == 0
    assert "title=Published OK; Resolution source provider failure (HTTP 200)" in (
        tmp_path / "step-output.txt"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("unit", "component", "error_type"),
    [
        ("models", "models", "forecast_drop"),
        ("publications", "publishing", "publish_failure"),
        ("alerts", "runtime", "unknown"),
        ("unknown", "runtime", "unknown"),
        (None, "resolution_source", "provider_failure"),
    ],
)
def test_clean_status_with_alertable_or_unknown_cause_falls_back_to_unknown(
    tmp_path: Path, unit: str | None, component: str, error_type: str
) -> None:
    status = _status_payload()
    cause: dict[str, object] = {
        "component": component,
        "affected": 1,
        "attempted": 1,
        "error_type": error_type,
        "http_status": None,
    }
    if unit is not None:
        cause["unit"] = unit
    status["causes"] = [cause]

    result = _run_renderer(tmp_path, status=status, bot_outcome="success")

    assert result.returncode != 0
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    assert "Forecast run: Unknown" in summary
    assert "status evidence is unavailable" in summary.lower()
    assert "::error" in result.stdout
    assert "title=Forecast status unknown" in (tmp_path / "step-output.txt").read_text(encoding="utf-8")


def test_zero_question_run_does_not_claim_publication(tmp_path: Path) -> None:
    status = _status_payload()
    status["models"] = {
        "attempted": 0,
        "succeeded": 0,
        "timed_out": 0,
        "dropped": 0,
        "drop_causes": {},
    }
    status["publishing"] = {
        "forecasts": {"attempted": 0, "succeeded": 0, "failed": 0},
        "comments": {"attempted": 0, "succeeded": 0, "failed": 0},
    }

    result = _run_renderer(tmp_path, status=status)

    assert result.returncode == 0
    assert "title=No forecasts attempted" in (tmp_path / "step-output.txt").read_text(encoding="utf-8")


def test_partial_source_failure_reports_reduced_coverage(tmp_path: Path) -> None:
    status = json.loads(DEGRADED_FIXTURE_PATH.read_text(encoding="utf-8"))

    result = _run_renderer(tmp_path, status=status)

    assert result.returncode == 0
    assert "Polymarket HTTP 403" in result.stdout
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    assert "1/13 lookups affected" in summary
    assert "3/3 succeeded, 0 timed out, 0 dropped" in summary
    assert "::error" in result.stdout
    assert "title=Published OK; Polymarket HTTP 403" in (tmp_path / "step-output.txt").read_text(encoding="utf-8")


def test_lookup_and_per_question_check_counts_keep_separate_units(tmp_path: Path) -> None:
    status = _status_payload()
    status["outcome"] = "degraded"
    status["causes"] = [
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
            "affected": 1,
            "attempted": 3,
            "error_type": "provider_failure",
            "http_status": 200,
        },
    ]

    result = _run_renderer(tmp_path, status=status)

    assert result.returncode == 0
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    assert "PredictIt HTTP 403; 1/1 lookups affected" in summary
    assert "PredictIt provider failure (HTTP 200); 1/3 source checks affected" in summary
    assert "3/1" not in summary


def test_model_timeout_is_distinct_from_source_and_publication_failure(tmp_path: Path) -> None:
    status = _status_payload()
    status["outcome"] = "degraded"
    status["models"] = {
        "attempted": 3,
        "succeeded": 2,
        "timed_out": 1,
        "dropped": 1,
        "drop_causes": {"timeout_wall_clock": 1},
    }
    status["causes"] = [
        {
            "component": "models",
            "unit": "models",
            "affected": 1,
            "attempted": 3,
            "error_type": "forecast_drop",
            "http_status": None,
        }
    ]

    result = _run_renderer(tmp_path, status=status)

    assert result.returncode == 0
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    assert "Models: 2/3 succeeded, 1 timed out, 1 dropped" in summary
    assert "Forecasts: 1/1 succeeded" in summary
    assert "Comments: 1/1 succeeded" in summary
    assert "1 model timed out" in (tmp_path / "step-output.txt").read_text(encoding="utf-8").lower()


def test_publication_failure_names_forecasts_and_comments_separately(tmp_path: Path) -> None:
    status = _status_payload()
    status["outcome"] = "hard_failure"
    status["publishing"] = {
        "forecasts": {"attempted": 2, "succeeded": 1, "failed": 1},
        "comments": {"attempted": 2, "succeeded": 0, "failed": 2},
    }

    result = _run_renderer(tmp_path, status=status)

    assert result.returncode == 0
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    assert "Forecasts: 1/2 succeeded, 1 failed" in summary
    assert "Comments: 0/2 succeeded, 2 failed" in summary
    output = (tmp_path / "step-output.txt").read_text(encoding="utf-8")
    assert "Publication failed" in output
    assert "::error" in result.stdout


@pytest.mark.parametrize("failure_source", ["job", "bot"])
def test_clean_status_does_not_hide_failed_workflow_or_bot_step(tmp_path: Path, failure_source: str) -> None:
    bot_outcome = "failure" if failure_source == "bot" else "success"
    forecast_job_status = "failure" if failure_source == "job" else "success"

    result = _run_renderer(
        tmp_path,
        status=_status_payload(),
        bot_outcome=bot_outcome,
        forecast_job_status=forecast_job_status,
    )

    assert result.returncode == 0
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    assert "Forecast run: Hard Failure" in summary
    assert "Runtime unknown error" in summary
    assert "::error" in result.stdout


@pytest.mark.parametrize("status", [None, "{malformed"])
def test_missing_or_malformed_status_after_success_fails_observability_gate(
    tmp_path: Path, status: dict[str, object] | str | None
) -> None:
    result = _run_renderer(tmp_path, status=status, bot_outcome="success")

    assert result.returncode != 0
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    assert "Forecast run: Unknown" in summary
    assert "status evidence is unavailable" in summary.lower()
    assert "::error" in result.stdout
    assert "title=Forecast status unknown" in (tmp_path / "step-output.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("bot_outcome", "playwright_outcome", "expected"),
    [
        ("failure", "success", "bot step failed"),
        ("skipped", "failure", "setup step failed"),
    ],
)
def test_setup_or_bot_failure_without_status_keeps_upstream_failure_primary(
    tmp_path: Path, bot_outcome: str, playwright_outcome: str, expected: str
) -> None:
    result = _run_renderer(
        tmp_path,
        status=None,
        bot_outcome=bot_outcome,
        forecast_job_status="failure",
        playwright_outcome=playwright_outcome,
    )

    assert result.returncode == 0
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8").lower()
    assert "forecast run: unknown" in summary
    assert expected in summary
    assert "::error" in result.stdout


def test_unrecognized_labels_and_arbitrary_error_text_are_not_rendered(tmp_path: Path) -> None:
    status = _status_payload()
    status["outcome"] = "degraded"
    status["causes"] = [
        {
            "component": "polymarket token=super-secret raw query: will the event happen?",
            "unit": "lookups private search phrase",
            "affected": 1,
            "attempted": 1,
            "error_type": "http_error; Authorization: Bearer abc123",
            "http_status": 403,
        }
    ]

    result = _run_renderer(tmp_path, status=status)
    all_output = (
        result.stdout
        + result.stderr
        + (tmp_path / "step-summary.md").read_text(encoding="utf-8")
        + (tmp_path / "step-output.txt").read_text(encoding="utf-8")
    )

    assert "super-secret" not in all_output
    assert "will the event happen" not in all_output
    assert "abc123" not in all_output
    assert "Unknown component" in all_output
    assert "HTTP 403" in all_output
    title = next(line.removeprefix("title=") for line in all_output.splitlines() if line.startswith("title="))
    assert len(title) <= 100


def test_source_cause_with_unknown_attempt_count_stays_unknown(tmp_path: Path) -> None:
    status = _status_payload()
    status["outcome"] = "degraded"
    status["causes"] = [
        {
            "component": "polymarket",
            "unit": "lookups",
            "affected": 1,
            "attempted": None,
            "error_type": "http_error",
            "http_status": 403,
        }
    ]

    result = _run_renderer(tmp_path, status=status)

    assert result.returncode == 0
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    assert "1 lookup affected; attempt count unknown" in summary
    assert "1/1 lookups" not in summary


def test_unclassified_attempts_are_explicitly_unknown(tmp_path: Path) -> None:
    status = _status_payload()
    status["outcome"] = "degraded"
    status["models"] = {
        "attempted": 3,
        "succeeded": 1,
        "timed_out": 0,
        "dropped": 1,
        "drop_causes": {},
    }
    status["publishing"] = {
        "forecasts": {"attempted": 2, "succeeded": 1, "failed": 0},
        "comments": {"attempted": 1, "succeeded": 1, "failed": 0},
    }

    result = _run_renderer(tmp_path, status=status)

    assert result.returncode == 0
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    assert "1 outcome unknown" in summary
    assert "1/2 succeeded, 0 failed, 1 outcome unknown" in summary


def test_status_only_failure_labels_render_as_clear_display_text(tmp_path: Path) -> None:
    status = _status_payload()
    status["outcome"] = "hard_failure"
    status["causes"] = [
        {
            "component": component,
            "unit": "source_checks",
            "affected": 1,
            "attempted": 2,
            "error_type": error_type,
            "http_status": None,
        }
        for component, error_type in (
            ("gap_fill_v1", "provider_failure"),
            ("gap_fill_v2", "time_budget"),
            ("provider_health", "credit_floor"),
            ("publish_gate", "stale_tournament"),
            ("aggregation", "aggregation_failure"),
            ("credit", "configuration_reminder"),
            ("tournament", "deprecated_model"),
        )
    ]

    result = _run_renderer(tmp_path, status=status)

    assert result.returncode == 0
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    for expected_label in (
        "Gap-fill analyzer provider failure",
        "Agentic gap-fill limited time budget",
        "Provider health credit floor reached",
        "Publication gate stale tournament status",
        "Aggregation aggregation failure",
        "Credits configuration reminder",
        "Tournament status deprecated model",
    ):
        assert expected_label in summary


@pytest.mark.parametrize("invalid", ["publishing", "models", "schema_version"])
def test_contradictory_or_invalid_status_counts_fall_back_to_unknown(tmp_path: Path, invalid: str) -> None:
    status = _status_payload()
    if invalid == "publishing":
        status["publishing"] = {
            "forecasts": {"attempted": 1, "succeeded": 2, "failed": 0},
            "comments": {"attempted": 1, "succeeded": 1, "failed": 0},
        }
    elif invalid == "models":
        status["models"] = {
            "attempted": 3,
            "succeeded": 3,
            "timed_out": 1,
            "dropped": 1,
            "drop_causes": {},
        }
    else:
        status["schema_version"] = True

    result = _run_renderer(tmp_path, status=status, bot_outcome="success")

    assert result.returncode != 0
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    assert "Forecast run: Unknown" in summary
    assert "status evidence is unavailable" in summary.lower()
