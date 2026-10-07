"""Idempotent setup of the cron-job.org jobs that dispatch this repository's bot workflows.

GitHub delivers about 22% of this repository's scheduled cron firings (docs/operations.md
"Scheduling reliability"); workflow_dispatch events are not subject to that dropping, so
cron-job.org calls GitHub's workflow-dispatch endpoint twice an hour per workflow instead. A
dispatched run that finds no new question makes no LLM call and spends nothing, which is what
makes the extra firings free. COST GATE: ``--apply`` creates or changes a live schedule, and every
firing it adds is a paid, publishing bot run, so it sits behind the operator's ask-first gate
(AGENTS.md "Cost discipline"). The default dry run makes no write and needs no key; ``--apply``
reads CRONJOB_API_KEY and GH_DISPATCH_TOKEN (a fine-grained PAT, Actions read/write on this
repository only) from the environment after the repo's .env load and never prints either.

Usage:
    make cronjob_dispatch_setup                                 # dry run: payloads (token redacted) + plan
    make cronjob_dispatch_setup ARGS="--apply"                  # PAID: create/update the live jobs; ask first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import requests

from metaculus_bot.config import load_environment

CRONJOB_API_KEY_ENV = "CRONJOB_API_KEY"
GH_DISPATCH_TOKEN_ENV = "GH_DISPATCH_TOKEN"  # noqa: S105  # env var NAME, not a credential
# The dry run builds its printed payloads with this in the token's place, so no secret is redacted after the fact.
GH_DISPATCH_TOKEN_PLACEHOLDER = "<GH_DISPATCH_TOKEN>"  # noqa: S105  # a placeholder, not a credential

CRONJOB_API_BASE = "https://api.cron-job.org"
GITHUB_DISPATCH_URL = (
    "https://api.github.com/repos/No-Stream/nostreambot-metaculus-bot/actions/workflows/{workflow}/dispatches"
)
GITHUB_DISPATCH_BODY = '{"ref":"main"}'
GITHUB_API_VERSION = "2022-11-28"
REQUEST_TIMEOUT_SECS = 30
# cron-job.org rate-limits PUT /jobs to one request per second.
WRITE_SPACING_SECS = 1.0

# cron-job.org's RequestMethod enum (docs.cron-job.org/rest-api.html): 0 GET, 1 POST, 2 OPTIONS, ...
REQUEST_METHOD_POST = 1
# In every schedule list, [-1] means "every" (every hour, every day of month, ...).
SCHEDULE_EVERY = [-1]
SCHEDULE_TIMEZONE = "UTC"
SCHEDULE_FIELDS = ("timezone", "hours", "mdays", "minutes", "months", "wdays")
NOTIFICATION_FIELDS = ("onFailure", "onSuccess", "onDisable")

MANTIC_WORKFLOW_FILE = "run_bot_on_mantic.yaml"

ACTION_CREATE = "create"
ACTION_UPDATE = "update"
ACTION_UNCHANGED = "unchanged"


@dataclass(frozen=True)
class DispatchJob:
    """One cron-job.org job: fires GitHub's workflow-dispatch endpoint for ``workflow_file`` at ``minutes`` past every hour."""

    title: str
    workflow_file: str
    minutes: tuple[int, ...]
    enabled: bool


# Two firings an hour per workflow, minutes staggered so two full bot runs never share the runners and research quotas.
DISPATCH_JOBS: tuple[DispatchJob, ...] = (
    DispatchJob("metaculus-bot dispatch: tournament", "run_bot_on_tournament.yaml", (2, 32), enabled=True),
    DispatchJob("metaculus-bot dispatch: metaculus cup", "run_bot_on_metaculus_cup.yaml", (12, 42), enabled=True),
    DispatchJob("metaculus-bot dispatch: mantic", MANTIC_WORKFLOW_FILE, (1, 16), enabled=True),
)


@dataclass(frozen=True)
class PlannedAction:
    kind: str
    payload: dict[str, Any] = field(repr=False)
    job_id: int | None


class CronJobApiError(Exception):
    """A non-2xx answer from cron-job.org. Carries the response only; the request is never echoed."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def build_job_payload(job: DispatchJob, gh_token: str) -> dict[str, Any]:
    """The cron-job.org ``job`` object for ``job``, in the REST API's own field names."""
    return {
        "title": job.title,
        "url": GITHUB_DISPATCH_URL.format(workflow=job.workflow_file),
        "enabled": job.enabled,
        "saveResponses": False,
        "requestMethod": REQUEST_METHOD_POST,
        "schedule": {
            "timezone": SCHEDULE_TIMEZONE,
            "hours": SCHEDULE_EVERY,
            "mdays": SCHEDULE_EVERY,
            "minutes": list(job.minutes),
            "months": SCHEDULE_EVERY,
            "wdays": SCHEDULE_EVERY,
        },
        "extendedData": {
            "headers": {
                "Authorization": f"Bearer {gh_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
                "Content-Type": "application/json",
            },
            "body": GITHUB_DISPATCH_BODY,
        },
        "notification": {"onFailure": True, "onSuccess": False, "onDisable": True},
    }


def _comparable(job: Mapping[str, Any]) -> dict[str, Any]:
    """The fields this script owns, normalized: schedule lists unordered, header names case-insensitive."""
    schedule = job["schedule"]
    return {
        "title": job["title"],
        "url": job["url"],
        "enabled": job["enabled"],
        "saveResponses": job["saveResponses"],
        "requestMethod": job["requestMethod"],
        "schedule": {
            name: schedule[name] if name == "timezone" else sorted(schedule[name]) for name in SCHEDULE_FIELDS
        },
        "headers": {name.lower(): value for name, value in job["extendedData"]["headers"].items()},
        "body": job["extendedData"]["body"],
        "notification": {name: job["notification"][name] for name in NOTIFICATION_FIELDS},
    }


def job_matches(desired: Mapping[str, Any], remote: Mapping[str, Any]) -> bool:
    """True when the remote job already carries every field we specify; remote-only fields are ignored."""
    return _comparable(desired) == _comparable(remote)


class CronJobClient:
    """The four cron-job.org calls this script makes. Any non-2xx raises ``CronJobApiError``."""

    def __init__(self, api_key: str) -> None:
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"

    def list_jobs(self) -> list[dict[str, Any]]:
        """Job summaries (no extendedData or notification; ``job_details`` has those)."""
        return self._request("GET", "/jobs")["jobs"]

    def job_details(self, job_id: int) -> dict[str, Any]:
        return self._request("GET", f"/jobs/{job_id}")["jobDetails"]

    def create_job(self, job: Mapping[str, Any]) -> int:
        return self._request("PUT", "/jobs", body={"job": job})["jobId"]

    def update_job(self, job_id: int, job: Mapping[str, Any]) -> None:
        self._request("PATCH", f"/jobs/{job_id}", body={"job": job})

    def _request(self, method: str, path: str, *, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        response = self._session.request(method, CRONJOB_API_BASE + path, json=body, timeout=REQUEST_TIMEOUT_SECS)
        if response.status_code // 100 != 2:
            raise CronJobApiError(response.status_code, response.text)
        return response.json()


def plan_actions(client: CronJobClient, payloads: Sequence[Mapping[str, Any]]) -> list[PlannedAction]:
    """Match each desired job to the account by exact title and decide create / update / unchanged."""
    remote_by_title: dict[str, list[dict[str, Any]]] = {}
    for summary in client.list_jobs():
        remote_by_title.setdefault(summary["title"], []).append(summary)
    actions: list[PlannedAction] = []
    for payload in payloads:
        title = payload["title"]
        if title not in remote_by_title:
            actions.append(PlannedAction(ACTION_CREATE, dict(payload), None))
            continue
        matches = remote_by_title[title]
        if len(matches) > 1:
            ids = ", ".join(str(match["jobId"]) for match in matches)
            raise SystemExit(
                f"{len(matches)} cron-job.org jobs share the title {title!r} (ids {ids}); delete the extras first"
            )
        job_id = matches[0]["jobId"]
        kind = ACTION_UNCHANGED if job_matches(payload, client.job_details(job_id)) else ACTION_UPDATE
        actions.append(PlannedAction(kind, dict(payload), job_id))
    return actions


def apply_actions(client: CronJobClient, actions: Sequence[PlannedAction]) -> list[PlannedAction]:
    """Perform the creates and updates, spaced for the write rate limit; returns the actions with new ids filled in."""
    applied: list[PlannedAction] = []
    writes = 0
    for action in actions:
        if action.kind == ACTION_UNCHANGED:
            applied.append(action)
            continue
        if writes:
            time.sleep(WRITE_SPACING_SECS)
        writes += 1
        if action.kind == ACTION_CREATE:
            job_id = client.create_job(action.payload)
            applied.append(PlannedAction(ACTION_CREATE, action.payload, job_id))
            continue
        assert action.job_id is not None
        client.update_job(action.job_id, action.payload)
        applied.append(action)
    return applied


def render_action(action: PlannedAction, *, applied: bool) -> str:
    label = action.kind if applied or action.kind == ACTION_UNCHANGED else f"would-{action.kind}"
    job_id = "-" if action.job_id is None else str(action.job_id)
    minutes = ",".join(str(minute) for minute in action.payload["schedule"]["minutes"])
    enabled = str(action.payload["enabled"]).lower()
    return f"{label:<13} id={job_id:<9} minutes={minutes:<6} enabled={enabled:<6} {action.payload['title']}"


def missing_secrets() -> list[str]:
    return [name for name in (CRONJOB_API_KEY_ENV, GH_DISPATCH_TOKEN_ENV) if not os.environ.get(name)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create or update the live cron-job.org jobs. PAID: every firing is a bot run; ask the operator first.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_environment()
    missing = missing_secrets()
    if args.apply and missing:
        parser.error(f"--apply needs {' and '.join(missing)} in the environment or .env")
    if not args.apply:
        for job in DISPATCH_JOBS:
            print(f"--- {job.title} ({job.workflow_file}) ---")
            print(json.dumps(build_job_payload(job, GH_DISPATCH_TOKEN_PLACEHOLDER), indent=2))
    if missing:
        print(
            f"Skipped reading the cron-job.org account: {' and '.join(missing)} not set, so there is no create/update plan."
        )
        return 0
    gh_token = os.environ[GH_DISPATCH_TOKEN_ENV]
    payloads = [build_job_payload(job, gh_token) for job in DISPATCH_JOBS]
    client = CronJobClient(os.environ[CRONJOB_API_KEY_ENV])
    try:
        actions = plan_actions(client, payloads)
        if args.apply:
            actions = apply_actions(client, actions)
    except CronJobApiError as exc:
        print(f"cron-job.org API error: HTTP {exc.status_code}: {exc.body}", file=sys.stderr)
        return 1
    for action in actions:
        print(render_action(action, applied=args.apply))
    return 0


if __name__ == "__main__":
    sys.exit(main())
