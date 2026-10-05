"""The structured failure evidence must survive Actions' log retention limit."""

import json
import logging
from pathlib import Path

import pytest

from metaculus_bot.run_status import RUN_STATUS, write_run_status_from_environment
from scripts.telemetry.markers import parse_log_text


def test_run_status_marker_archives_exact_persisted_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    destination = tmp_path / "status.json"
    monkeypatch.setenv("FORECAST_RUN_STATUS_PATH", str(destination))
    RUN_STATUS.reset()
    RUN_STATUS.record_research_attempt("polymarket")
    RUN_STATUS.record_research_failure("polymarket", error_type="http_error", http_status=403)
    RUN_STATUS.set_outcome("degraded")
    with caplog.at_level(logging.INFO, logger="metaculus_bot.run_status"):
        write_run_status_from_environment()
    try:
        records = parse_log_text(
            caplog.text,
            run_id="fixture",
            workflow="notification",
            artifact="research-fixture",
            run_date="2026-10-04",
            log_file="fixture.log",
        )
        assert len(records["run_status_json"]) == 1
        payload = json.loads(records["run_status_json"][0]["payload"])
        assert payload == json.loads(destination.read_text())
        assert payload["outcome"] == "degraded"
        assert payload["causes"][0]["http_status"] == 403
        assert "qid" not in records["run_status_json"][0]
    finally:
        RUN_STATUS.reset()
