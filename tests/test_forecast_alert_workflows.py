"""Failure reporting must never change the bot's red/green contract."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).resolve().parents[1] / ".github/workflows").glob("*bot*.yaml"))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_forecast_failure_remains_visible_and_summary_always_runs(path: Path) -> None:
    jobs = yaml.safe_load(path.read_text())["jobs"]
    forecast = jobs["forecast_job"]
    bot = next(step for step in forecast["steps"] if step.get("name") == "Run bot")
    assert bot["id"] == "run_bot"
    assert bot["shell"] == "bash"
    assert not bot.get("continue-on-error", False)
    assert not forecast.get("continue-on-error", False)
    assert bot["env"]["FORECAST_RUN_STATUS_PATH"] == "run_logs/status.json"
    summary = next(step for step in forecast["steps"] if step.get("id") == "run_summary")
    assert summary["if"] == "always()"
    assert "python3 scripts/forecast_run_summary.py" in summary["run"]
    assert "checkout/setup failed" in summary["run"]
    assert "title=Forecast status unknown" in summary["run"]
    assert summary["env"]["BOT_STEP_OUTCOME"] == "${{ steps.run_bot.outcome }}"
    assert summary["env"]["FORECAST_JOB_STATUS"] == "${{ job.status }}"
    assert forecast["outputs"]["title"] == "${{ steps.run_summary.outputs.title }}"
    assert forecast["steps"].index(summary) > next(
        index for index, step in enumerate(forecast["steps"]) if step.get("name") == "Upload research outputs"
    )
    report = jobs["report_status"]
    assert report["needs"] == "forecast_job"
    assert report["if"] == "always()"
    assert "needs.forecast_job.outputs.title" in report["name"]
    assert "unknown" in report["name"].lower()
    step = report["steps"][0]
    assert "needs.forecast_job.result" in step["env"]["FORECAST_RESULT"]
    assert "exit 1" in step["run"]
    assert "${{" not in step["run"]  # Outputs enter through env, never executable shell text.


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_reporting_shell_retains_failure_and_renders_untrusted_title(result: str, tmp_path: Path) -> None:
    workflow = yaml.safe_load(WORKFLOWS[0].read_text())
    step = workflow["jobs"]["report_status"]["steps"][0]
    summary = tmp_path / "summary.md"
    title = "Published OK; $(touch should-not-exist)"
    process = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={**os.environ, "FORECAST_RESULT": result, "STATUS_TITLE": title, "GITHUB_STEP_SUMMARY": str(summary)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == (0 if result == "success" else 1)
    assert title in summary.read_text()
    assert result in summary.read_text()
    assert not (tmp_path / "should-not-exist").exists()


def test_checkout_failure_still_writes_summary_and_job_output(tmp_path: Path) -> None:
    workflow = yaml.safe_load(WORKFLOWS[0].read_text())
    step = next(step for step in workflow["jobs"]["forecast_job"]["steps"] if step.get("id") == "run_summary")
    summary = tmp_path / "summary.md"
    outputs = tmp_path / "outputs"
    process = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary), "GITHUB_OUTPUT": str(outputs)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 1
    assert "unknown" in summary.read_text()
    assert "title=Forecast status unknown" in outputs.read_text()
    assert "::error" in process.stdout


def test_notification_fixture_has_no_live_bot_or_secret_inputs() -> None:
    path = WORKFLOWS[0].parent / "forecast_notification_test.yaml"
    text = path.read_text()
    workflow = yaml.safe_load(text)
    # YAML 1.1 treats GitHub's unquoted `on` as True.
    assert workflow.get("on", workflow.get(True)) == {"workflow_dispatch": None}
    assert "secrets." not in text
    assert "main.py" not in text
    assert "schedule:" not in text
    fixture = workflow["jobs"]["forecast_fixture"]
    assert fixture["outputs"]["title"] == "${{ steps.run_summary.outputs.title }}"
    assert any(step.get("run") == "exit 1" for step in fixture["steps"])
    report = workflow["jobs"]["report_status"]
    assert report["if"] == "always()"
    assert report["needs"] == "forecast_fixture"
    assert "needs.forecast_fixture.outputs.title" in report["name"]
