"""Tests for the cron-job.org dispatcher setup (scripts/cronjob_dispatch_setup.py).

The script exists because GitHub delivers about a fifth of this repository's scheduled cron
firings, so an external dispatcher calls the workflow-dispatch endpoint instead; what is pinned
here is the exact job specs (URL, headers, body, cadence, enabled flags), the idempotency diff
(unchanged versus update, tolerant of remote-only fields, list order and header-name case), the
dry run's redaction (the GitHub token never appears in any output, the placeholder does) and the
fail-fast on ``--apply`` without both secrets.

``FakeCronJobApi`` stands in for ``requests.Session`` with the REST API's own shapes, including the
one that matters for the diff: ``GET /jobs`` returns summaries WITHOUT ``extendedData`` or
``notification`` (only ``GET /jobs/<id>`` carries them), so a diff that read the list page alone
would fail here too. No live API: ``load_environment`` is neutralized so a developer's .env cannot
leak into the missing-secret cases, and the autouse egress guard in tests/conftest.py would raise
on any real connect anyway.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Sequence
from typing import Any

import pytest
import requests

from scripts import cronjob_dispatch_setup as setup
from scripts.cronjob_dispatch_setup import (
    ACTION_CREATE,
    ACTION_UNCHANGED,
    ACTION_UPDATE,
    DISPATCH_JOBS,
    GH_DISPATCH_TOKEN_PLACEHOLDER,
    REQUEST_METHOD_POST,
    REQUEST_TIMEOUT_SECS,
    CronJobApiError,
    CronJobClient,
    DispatchJob,
    build_job_payload,
    job_matches,
    main,
    plan_actions,
)
from tests.http_fakes import json_response, text_response

FAKE_API_KEY = "cronjob-api-key-0123456789abcdef"
FAKE_GH_TOKEN = "github_pat_11ABCDEFG0123456789_secretpart"

TOURNAMENT, CUP, MANTIC = DISPATCH_JOBS


def remote_job(payload: dict[str, Any], job_id: int) -> dict[str, Any]:
    """A DetailedJob as cron-job.org returns it: our payload plus the fields the service adds."""
    job = copy.deepcopy(payload)
    job.update(
        {
            "jobId": job_id,
            "lastStatus": 0,
            "lastDuration": 0,
            "lastExecution": 0,
            "nextExecution": 1_800_000_000,
            "type": 0,
            "requestTimeout": -1,
            "redirectSuccess": False,
            "folderId": 0,
            "auth": {"enable": False, "user": "", "password": ""},
        }
    )
    job["schedule"]["expiresAt"] = 0
    job["notification"].update({"onFailureCount": 1, "onSslCertExpiry": False, "onSslCertExpirySeconds": 604800})
    return job


class FakeCronJobApi:
    """An in-memory cron-job.org behind the ``requests.Session`` surface the client uses."""

    _DETAIL_ONLY = ("extendedData", "notification", "auth")

    def __init__(self, jobs: Sequence[dict[str, Any]] = (), *, fail_with: tuple[int, str] | None = None) -> None:
        self.jobs: dict[int, dict[str, Any]] = {job["jobId"]: copy.deepcopy(job) for job in jobs}
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.headers: dict[str, str] = {}
        self._fail_with = fail_with
        self._next_id = 9000

    def request(self, method: str, url: str, *, json: dict[str, Any] | None, timeout: int) -> requests.Response:
        assert timeout == REQUEST_TIMEOUT_SECS
        assert url.startswith("https://api.cron-job.org/")
        path = url.removeprefix("https://api.cron-job.org")
        self.calls.append((method, path, json))
        if self._fail_with is not None:
            status, body = self._fail_with
            return text_response(body, status=status)
        if (method, path) == ("GET", "/jobs"):
            return json_response({"jobs": [self._summary(job) for job in self.jobs.values()], "someFailed": False})
        if method == "PUT" and path == "/jobs":
            assert json is not None
            self._next_id += 1
            self.jobs[self._next_id] = remote_job(json["job"], self._next_id)
            return json_response({"jobId": self._next_id})
        job_id = int(path.removeprefix("/jobs/"))
        if job_id not in self.jobs:
            return text_response("", status=404)
        if method == "GET":
            return json_response({"jobDetails": self.jobs[job_id]})
        assert method == "PATCH"
        assert json is not None
        self.jobs[job_id].update(copy.deepcopy(json["job"]))
        return json_response({})

    def _summary(self, job: dict[str, Any]) -> dict[str, Any]:
        return {name: value for name, value in job.items() if name not in self._DETAIL_ONLY}

    @property
    def writes(self) -> list[tuple[str, str, dict[str, Any] | None]]:
        return [call for call in self.calls if call[0] in ("PUT", "PATCH")]


def desired(job: DispatchJob) -> dict[str, Any]:
    return build_job_payload(job, FAKE_GH_TOKEN)


def payloads_printed(out: str) -> dict[str, dict[str, Any]]:
    """The dry run's JSON payloads keyed by title, parsed back from the ``--- title (file) ---`` blocks."""
    blocks = re.split(r"^--- (.+?) \(.+?\) ---$", out, flags=re.MULTILINE)[1:]
    decoder = json.JSONDecoder()
    return {title: decoder.raw_decode(body.lstrip())[0] for title, body in zip(blocks[::2], blocks[1::2], strict=True)}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """No .env load, neither secret set, no real sleeps; returns the patcher so ``secrets_env`` can build on it."""
    monkeypatch.setattr(setup, "load_environment", lambda: None)
    monkeypatch.delenv(setup.CRONJOB_API_KEY_ENV, raising=False)
    monkeypatch.delenv(setup.GH_DISPATCH_TOKEN_ENV, raising=False)
    monkeypatch.setattr(setup.time, "sleep", lambda _secs: None)
    return monkeypatch


@pytest.fixture
def secrets_env(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv(setup.CRONJOB_API_KEY_ENV, FAKE_API_KEY)
    clean_env.setenv(setup.GH_DISPATCH_TOKEN_ENV, FAKE_GH_TOKEN)


@pytest.fixture
def no_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any attempt to build a requests.Session is a test failure: these paths must never reach the API."""

    def _refuse() -> None:
        raise AssertionError("requests.Session constructed on a path that must make no request")

    monkeypatch.setattr(setup.requests, "Session", _refuse)


def install_fake_api(monkeypatch: pytest.MonkeyPatch, api: FakeCronJobApi) -> FakeCronJobApi:
    monkeypatch.setattr(setup.requests, "Session", lambda: api)
    return api


class TestJobTable:
    def test_the_three_jobs_and_their_cadence(self):
        assert [(job.title, job.workflow_file, job.minutes, job.enabled) for job in DISPATCH_JOBS] == [
            ("metaculus-bot dispatch: tournament", "run_bot_on_tournament.yaml", (2, 32), True),
            ("metaculus-bot dispatch: metaculus cup", "run_bot_on_metaculus_cup.yaml", (12, 42), True),
            ("metaculus-bot dispatch: mantic", "run_bot_on_mantic.yaml", (1, 16), True),
        ]

    def test_titles_are_unique_because_they_are_the_match_key(self):
        titles = [job.title for job in DISPATCH_JOBS]
        assert len(set(titles)) == len(titles)

    def test_minutes_never_collide_across_workflows(self):
        minutes = [minute for job in DISPATCH_JOBS for minute in job.minutes]
        assert len(set(minutes)) == len(minutes)


class TestJobPayload:
    def test_tournament_payload_exactly(self):
        assert build_job_payload(TOURNAMENT, GH_DISPATCH_TOKEN_PLACEHOLDER) == {
            "title": "metaculus-bot dispatch: tournament",
            "url": "https://api.github.com/repos/No-Stream/nostreambot-metaculus-bot/actions/workflows/run_bot_on_tournament.yaml/dispatches",
            "enabled": True,
            "saveResponses": False,
            "requestMethod": 1,
            "schedule": {
                "timezone": "UTC",
                "hours": [-1],
                "mdays": [-1],
                "minutes": [2, 32],
                "months": [-1],
                "wdays": [-1],
            },
            "extendedData": {
                "headers": {
                    "Authorization": "Bearer <GH_DISPATCH_TOKEN>",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "Content-Type": "application/json",
                },
                "body": '{"ref":"main"}',
            },
            "notification": {"onFailure": True, "onSuccess": False, "onDisable": True},
        }

    @pytest.mark.parametrize("job", DISPATCH_JOBS, ids=lambda job: job.workflow_file)
    def test_every_job_posts_a_main_dispatch_for_its_workflow(self, job: DispatchJob):
        payload = desired(job)

        assert payload["url"].endswith(f"/actions/workflows/{job.workflow_file}/dispatches")
        assert payload["requestMethod"] == REQUEST_METHOD_POST == 1
        assert payload["extendedData"]["body"] == '{"ref":"main"}'
        assert payload["extendedData"]["headers"]["Authorization"] == f"Bearer {FAKE_GH_TOKEN}"
        assert payload["schedule"]["minutes"] == list(job.minutes)
        assert payload["enabled"] is job.enabled

    def test_every_dispatch_job_is_enabled_by_default(self):
        assert {job.workflow_file: desired(job)["enabled"] for job in DISPATCH_JOBS} == {
            "run_bot_on_tournament.yaml": True,
            "run_bot_on_metaculus_cup.yaml": True,
            "run_bot_on_mantic.yaml": True,
        }


class TestJobMatches:
    def test_remote_only_fields_do_not_count_as_a_difference(self):
        assert job_matches(desired(CUP), remote_job(desired(CUP), 42))

    def test_minute_order_and_header_name_case_do_not_count(self):
        remote = remote_job(desired(CUP), 42)
        remote["schedule"]["minutes"] = [42, 12]
        remote["extendedData"]["headers"] = {
            name.lower(): value for name, value in remote["extendedData"]["headers"].items()
        }

        assert job_matches(desired(CUP), remote)

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda job: job.__setitem__("enabled", False), id="enabled"),
            pytest.param(lambda job: job["schedule"].__setitem__("minutes", [12, 43]), id="minutes"),
            pytest.param(lambda job: job["schedule"].__setitem__("hours", [3]), id="hours"),
            pytest.param(lambda job: job["extendedData"].__setitem__("body", '{"ref":"dev"}'), id="body"),
            pytest.param(
                lambda job: job["extendedData"]["headers"].__setitem__("Authorization", "Bearer rotated"),
                id="rotated-token",
            ),
            pytest.param(lambda job: job["notification"].__setitem__("onFailure", False), id="notification"),
            pytest.param(lambda job: job.__setitem__("url", "https://api.github.com/other"), id="url"),
        ],
    )
    def test_a_differing_owned_field_is_a_difference(self, mutate):
        remote = remote_job(desired(CUP), 42)
        mutate(remote)
        assert not job_matches(desired(CUP), remote)


class TestDryRun:
    @pytest.mark.usefixtures("clean_env", "no_session")
    def test_without_secrets_prints_redacted_payloads_and_no_plan(self, capsys):
        assert main([]) == 0

        out = capsys.readouterr().out
        payloads = payloads_printed(out)
        assert list(payloads) == [job.title for job in DISPATCH_JOBS]
        for job in DISPATCH_JOBS:
            assert payloads[job.title] == build_job_payload(job, GH_DISPATCH_TOKEN_PLACEHOLDER)
            assert payloads[job.title]["extendedData"]["headers"]["Authorization"] == "Bearer <GH_DISPATCH_TOKEN>"
        assert "Skipped reading the cron-job.org account: CRONJOB_API_KEY and GH_DISPATCH_TOKEN not set" in out
        assert "would-" not in out

    @pytest.mark.usefixtures("clean_env", "no_session")
    def test_one_missing_secret_names_only_that_one(self, monkeypatch, capsys):
        monkeypatch.setenv(setup.CRONJOB_API_KEY_ENV, FAKE_API_KEY)

        assert main([]) == 0

        out = capsys.readouterr().out
        assert "Skipped reading the cron-job.org account: GH_DISPATCH_TOKEN not set" in out
        assert FAKE_API_KEY not in out

    @pytest.mark.usefixtures("clean_env", "no_session")
    def test_dry_run_shows_the_mantic_payload_enabled_by_default(self, capsys):
        assert main([]) == 0
        assert payloads_printed(capsys.readouterr().out)[MANTIC.title]["enabled"] is True

    def test_obsolete_enable_mantic_option_is_rejected(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--enable-mantic"])

        assert excinfo.value.code == 2
        assert "unrecognized arguments: --enable-mantic" in capsys.readouterr().err

    @pytest.mark.usefixtures("secrets_env")
    def test_with_secrets_plans_against_the_account_without_writing(self, monkeypatch, capsys):
        stale_mantic = remote_job(desired(MANTIC), 502)
        stale_mantic["schedule"]["minutes"] = [5, 35]
        api = install_fake_api(monkeypatch, FakeCronJobApi([remote_job(desired(CUP), 501), stale_mantic]))

        assert main([]) == 0

        out = capsys.readouterr().out
        plan = [line for line in out.splitlines() if line.startswith(("would-", "unchanged"))]
        assert plan == [
            "would-create  id=-         minutes=2,32   enabled=true   metaculus-bot dispatch: tournament",
            "unchanged     id=501       minutes=12,42  enabled=true   metaculus-bot dispatch: metaculus cup",
            "would-update  id=502       minutes=1,16   enabled=true   metaculus-bot dispatch: mantic",
        ]
        assert api.writes == []
        assert [call[:2] for call in api.calls] == [("GET", "/jobs"), ("GET", "/jobs/501"), ("GET", "/jobs/502")]
        assert api.headers == {"Authorization": f"Bearer {FAKE_API_KEY}"}
        assert FAKE_GH_TOKEN not in out
        assert FAKE_API_KEY not in out
        assert out.count(GH_DISPATCH_TOKEN_PLACEHOLDER) == len(DISPATCH_JOBS)


class TestApply:
    @pytest.mark.usefixtures("clean_env", "no_session")
    def test_apply_without_secrets_fails_fast_before_any_request(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--apply"])

        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "--apply needs CRONJOB_API_KEY and GH_DISPATCH_TOKEN" in err

    @pytest.mark.usefixtures("clean_env", "no_session")
    def test_apply_with_one_secret_names_the_missing_one(self, monkeypatch, capsys):
        monkeypatch.setenv(setup.GH_DISPATCH_TOKEN_ENV, FAKE_GH_TOKEN)

        with pytest.raises(SystemExit):
            main(["--apply"])

        err = capsys.readouterr().err
        assert "--apply needs CRONJOB_API_KEY in the environment" in err
        assert FAKE_GH_TOKEN not in err

    @pytest.mark.usefixtures("secrets_env")
    def test_apply_creates_updates_and_leaves_unchanged(self, monkeypatch, capsys):
        stale_mantic = remote_job(desired(MANTIC), 502)
        stale_mantic["schedule"]["minutes"] = [5, 35]
        api = install_fake_api(monkeypatch, FakeCronJobApi([remote_job(desired(CUP), 501), stale_mantic]))
        sleeps: list[float] = []
        monkeypatch.setattr(setup.time, "sleep", sleeps.append)

        assert main(["--apply"]) == 0

        out = capsys.readouterr().out
        assert out.splitlines() == [
            "create        id=9001      minutes=2,32   enabled=true   metaculus-bot dispatch: tournament",
            "unchanged     id=501       minutes=12,42  enabled=true   metaculus-bot dispatch: metaculus cup",
            "update        id=502       minutes=1,16   enabled=true   metaculus-bot dispatch: mantic",
        ]
        assert api.writes == [
            ("PUT", "/jobs", {"job": desired(TOURNAMENT)}),
            ("PATCH", "/jobs/502", {"job": desired(MANTIC)}),
        ]
        assert sleeps == [setup.WRITE_SPACING_SECS]
        assert api.jobs[502]["enabled"] is True
        assert FAKE_GH_TOKEN not in out

    @pytest.mark.usefixtures("secrets_env")
    def test_a_second_apply_changes_nothing(self, monkeypatch, capsys):
        api = install_fake_api(monkeypatch, FakeCronJobApi())

        assert main(["--apply"]) == 0
        first_writes = list(api.writes)
        assert [call[0] for call in first_writes] == ["PUT", "PUT", "PUT"]
        capsys.readouterr()

        assert main(["--apply"]) == 0

        assert api.writes == first_writes
        assert all(line.startswith("unchanged") for line in capsys.readouterr().out.splitlines())

    @pytest.mark.usefixtures("secrets_env")
    def test_plain_apply_keeps_the_active_mantic_job_enabled(self, monkeypatch, capsys):
        api = install_fake_api(
            monkeypatch, FakeCronJobApi([remote_job(desired(job), 600 + i) for i, job in enumerate(DISPATCH_JOBS)])
        )

        assert main(["--apply"]) == 0

        assert api.writes == []
        assert api.jobs[602]["enabled"] is True
        assert "unchanged     id=602" in capsys.readouterr().out

    @pytest.mark.usefixtures("secrets_env")
    def test_non_2xx_exits_non_zero_with_status_and_body_and_no_secret(self, monkeypatch, capsys):
        install_fake_api(monkeypatch, FakeCronJobApi(fail_with=(401, '{"error":"invalid api key"}')))

        assert main(["--apply"]) == 1

        captured = capsys.readouterr()
        assert captured.err.strip() == 'cron-job.org API error: HTTP 401: {"error":"invalid api key"}'
        assert FAKE_API_KEY not in captured.out + captured.err
        assert FAKE_GH_TOKEN not in captured.out + captured.err

    @pytest.mark.usefixtures("secrets_env")
    def test_duplicate_titles_abort_instead_of_guessing(self, monkeypatch):
        api = install_fake_api(
            monkeypatch, FakeCronJobApi([remote_job(desired(CUP), 701), remote_job(desired(CUP), 702)])
        )

        with pytest.raises(
            SystemExit,
            match=r"2 cron-job.org jobs share the title 'metaculus-bot dispatch: metaculus cup' \(ids 701, 702\)",
        ):
            plan_actions(CronJobClient(FAKE_API_KEY), [desired(CUP)])

        assert api.writes == []


class TestClient:
    def test_non_2xx_raises_with_status_and_body(self, monkeypatch):
        install_fake_api(monkeypatch, FakeCronJobApi(fail_with=(429, "rate limited")))

        with pytest.raises(CronJobApiError, match="HTTP 429: rate limited") as excinfo:
            CronJobClient(FAKE_API_KEY).list_jobs()

        assert (excinfo.value.status_code, excinfo.value.body) == (429, "rate limited")

    def test_planned_action_repr_hides_the_payload(self):
        action = setup.PlannedAction(ACTION_CREATE, desired(TOURNAMENT), None)
        assert FAKE_GH_TOKEN not in repr(action)

    def test_action_kinds_are_the_three_words_the_report_prints(self):
        assert (ACTION_CREATE, ACTION_UPDATE, ACTION_UNCHANGED) == ("create", "update", "unchanged")
