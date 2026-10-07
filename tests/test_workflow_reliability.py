"""Guards that a hung workflow step cannot hold a concurrency group for hours.

On 2026-08-19 the Azure Ubuntu mirror failed over and ``apt-get update`` hung inside
``playwright install --with-deps`` on four tournament runs. The step carried
``continue-on-error: true``, which covers a step that FAILS and not one that HANGS, and
no step-level ``timeout-minutes``, so each hang ran to the job-level backstop of 300
minutes while holding ``concurrency.group: ${{ github.workflow }}``. GitHub allows at
most one PENDING run per group and cancels the existing pending one when a new run
arrives, so 54 of the day's 68 fires were evicted, the group was held 18 of 24 hours,
and q45374 and q45375 opened and closed without a forecast.

The fix has two halves and this module pins both, because either alone leaves the hole:

1. Every step that shells out or touches the network declares its own
   ``timeout-minutes``. A step cap kills the process, lets ``continue-on-error`` do what
   it was already there for, and — where the step is not allowed to fail — reports a
   ``failure`` rather than the ``cancelled`` a job-level timeout reports, which is
   indistinguishable from a concurrency eviction.
2. Every job declares ``timeout-minutes`` at all, so nothing inherits GitHub's 360.

The numbers are sized in scratch/residual_2026-08-24/workflow_reliability_audit.md off
measured step durations (scratch/residual_2026-08-24/track_c_step_timings.json), with
one hard floor asserted here: the ``Run bot`` cap must clear the bot's OWN per-question
contract in constants.py, or the workflow silently truncates a run in the second it was
about to publish — the forfeit these caps exist to prevent. ``test_bot_basic.yaml``
shipped exactly that inversion, a 60-minute job cap over a 3600-second contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from metaculus_bot.constants import (
    METACULUS_CLOSE_WINDOW_SECONDS,
    PER_QUESTION_WALL_CLOCK_DEADLINE,
    PUBLISH_RESERVE_SECONDS,
    TIME_BUDGET_FAST_PATH_THRESHOLD,
    WALL_CLOCK_STACKING_MIN_BUDGET,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"

# Repo-relative posix paths only: an assertion on a path derived from the developer's
# checkout passes locally by construction and fails on the first CI run.
_ALL_WORKFLOWS = sorted(p.relative_to(_REPO_ROOT).as_posix() for p in _WORKFLOW_DIR.glob("*.y*ml"))
_BOT_WORKFLOWS = sorted(p.relative_to(_REPO_ROOT).as_posix() for p in _WORKFLOW_DIR.glob("*bot*.y*ml"))
_NON_BOT_WORKFLOWS = sorted(set(_ALL_WORKFLOWS) - set(_BOT_WORKFLOWS))

# Steps that provably cannot hang, so a cap would be noise: no network, no subprocess.
_UNCAPPED_BY_DESIGN = {"Warn if Playwright install failed"}

# Worst healthy duration measured over 200 successful runs per step, in seconds
# (track_c_step_timings.json), keyed by the step's LABEL — its ``name``, or its ``uses``
# for the action steps that declare none. Matched by prefix so bumping an action's
# version does not silently drop that step from the check. Caps must clear these by the
# multiple below, or a slow runner day starts killing healthy steps.
_MEASURED_WORST_SECONDS = {
    "Check out repository": 4,
    "astral-sh/setup-uv": 3,
    "Install dependencies": 22,
    "Install Playwright Chromium": 55,
    "Upload research outputs": 2,
}
_MIN_HEADROOM_OVER_MEASURED = 5

# A hang must cost far less than the 180-minute window a tournament question is open
# for; 300 (the pre-fix value) is longer than the whole window.
_MAX_TOLERABLE_JOB_CAP_MINUTES = 90

# ci.yaml and claude.yml have no publish deadline to protect, so their caps answer to
# measured duration alone: the worst of 131 successful CI jobs is 245 s, and claude.yml
# has never run past its @claude gate. Both bounds matter. Above, GitHub's uncapped
# 360-minute default is the defect these caps exist to remove, and claude.yml is the one
# workflow that spends a paid key while it hangs; below, a cap under 5 minutes would
# start killing healthy jobs — a red required check on a slow runner day.
_MAX_NON_BOT_JOB_CAP_MINUTES = 60
_MIN_JOB_CAP_MINUTES = 5


def _workflow(rel_path: str) -> dict[str, Any]:
    return yaml.safe_load((_REPO_ROOT / rel_path).read_text())


def _steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


def _named_step(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [step for step in _steps(workflow) if step.get("name") == name]
    assert len(matches) == 1, f"expected exactly one {name!r} step, got {len(matches)}"
    return matches[0]


def _step_label(step: dict[str, Any]) -> str:
    return step.get("name") or step.get("uses", "<unnamed>")


def _labelled_step(workflow: dict[str, Any], label_prefix: str) -> dict[str, Any]:
    """The one step whose label starts with ``label_prefix`` (see _MEASURED_WORST_SECONDS)."""
    matches = [step for step in _steps(workflow) if _step_label(step).startswith(label_prefix)]
    assert len(matches) == 1, f"expected exactly one step labelled {label_prefix!r}, got {len(matches)}"
    return matches[0]


class TestEveryJobIsCapped:
    """No job may inherit GitHub's 360-minute default, in any workflow."""

    def test_the_workflow_set_is_what_we_think_it_is(self) -> None:
        # Pinned so a NEW workflow has to satisfy the invariants below rather than
        # silently escape them. .y*ml because both spellings are live (claude.yml).
        assert _ALL_WORKFLOWS == [
            ".github/workflows/ci.yaml",
            ".github/workflows/claude.yml",
            ".github/workflows/fetch_diagnostic.yaml",
            ".github/workflows/forecast_notification_test.yaml",
            ".github/workflows/run_bot_on_mantic.yaml",
            ".github/workflows/run_bot_on_metaculus_cup.yaml",
            ".github/workflows/run_bot_on_minibench.yaml",
            ".github/workflows/run_bot_on_tournament.yaml",
            ".github/workflows/test_bot.yaml",
            ".github/workflows/test_bot_basic.yaml",
        ]

    @pytest.mark.parametrize("rel_path", _ALL_WORKFLOWS)
    def test_workflow_parses(self, rel_path: str) -> None:
        # A yaml GitHub cannot parse is a workflow that silently never runs, and for the
        # cron bot workflows that is indistinguishable from cron starvation.
        assert _workflow(rel_path)["jobs"], f"{rel_path} declares no jobs"

    @pytest.mark.parametrize("rel_path", _ALL_WORKFLOWS)
    def test_every_job_declares_a_timeout(self, rel_path: str) -> None:
        for job_name, job in _workflow(rel_path)["jobs"].items():
            assert isinstance(job.get("timeout-minutes"), int), (
                f"{rel_path}:{job_name} has no timeout-minutes, so it inherits GitHub's 360-minute "
                "default — six hours of a held concurrency group per hang"
            )


class TestBotWorkflowStepsAreCapped:
    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_every_step_that_can_hang_declares_a_timeout(self, rel_path: str) -> None:
        for step in _steps(_workflow(rel_path)):
            label = _step_label(step)
            if label in _UNCAPPED_BY_DESIGN:
                continue
            assert isinstance(step.get("timeout-minutes"), int), (
                f"{rel_path}: step {label!r} shells out or hits the network with no timeout-minutes. "
                "The job cap is not a substitute: it fires late and reports `cancelled`, which reads "
                "identically to a concurrency eviction in the run list"
            )

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_step_caps_clear_the_measured_worst_case(self, rel_path: str) -> None:
        for label, worst_seconds in _MEASURED_WORST_SECONDS.items():
            cap_seconds = _labelled_step(_workflow(rel_path), label)["timeout-minutes"] * 60
            assert cap_seconds >= worst_seconds * _MIN_HEADROOM_OVER_MEASURED, (
                f"{rel_path}: {label!r} capped at {cap_seconds}s but the worst of 200 healthy runs is "
                f"{worst_seconds}s; keep {_MIN_HEADROOM_OVER_MEASURED}x headroom so a slow runner day "
                "cannot kill a healthy step"
            )

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_playwright_step_is_both_capped_and_allowed_to_fail(self, rel_path: str) -> None:
        # Either half alone leaves the 2026-08-19 hole: continue-on-error cannot see a
        # hang, and a cap without continue-on-error would turn a benign apt blip into a
        # lost question instead of a degraded rendered-fetch rung.
        step = _named_step(_workflow(rel_path), "Install Playwright Chromium")
        assert step.get("continue-on-error") is True, (
            f"{rel_path}: a Chromium install failure must not fail the run — gap-fill v2 degrades to plain fetch"
        )
        assert isinstance(step.get("timeout-minutes"), int), (
            f"{rel_path}: continue-on-error does not cover a HANG; this step needs its own cap"
        )


class TestRunBotCapRespectsTheBotsOwnContract:
    """The workflow must not cut the bot off before its own deadline machinery fires."""

    def test_the_contract_is_still_the_one_hour_cycle(self) -> None:
        # If this ever changes, the caps below have to move with it — that is the point of
        # deriving them from the constants instead of hardcoding 60.
        assert PER_QUESTION_WALL_CLOCK_DEADLINE + WALL_CLOCK_STACKING_MIN_BUDGET == METACULUS_CLOSE_WINDOW_SECONDS

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_run_bot_cap_exceeds_the_per_question_contract(self, rel_path: str) -> None:
        cap_seconds = _named_step(_workflow(rel_path), "Run bot")["timeout-minutes"] * 60
        assert cap_seconds > METACULUS_CLOSE_WINDOW_SECONDS, (
            f"{rel_path}: 'Run bot' capped at {cap_seconds}s, but a question may legitimately use "
            f"PER_QUESTION_WALL_CLOCK_DEADLINE ({PER_QUESTION_WALL_CLOCK_DEADLINE}s) and then publish "
            f"inside WALL_CLOCK_STACKING_MIN_BUDGET ({WALL_CLOCK_STACKING_MIN_BUDGET}s). Questions run "
            "concurrently, so this bound does not scale with the batch size — a tighter cap kills a "
            "slow-but-recovering run in the second it was about to publish"
        )

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_job_cap_sits_above_the_run_bot_cap(self, rel_path: str) -> None:
        workflow = _workflow(rel_path)
        run_bot_cap = _named_step(workflow, "Run bot")["timeout-minutes"]
        for job_name, job in workflow["jobs"].items():
            if not any(step.get("name") == "Run bot" for step in job.get("steps", [])):
                continue
            assert job["timeout-minutes"] > run_bot_cap, (
                f"{rel_path}:{job_name} job cap ({job['timeout-minutes']}m) does not clear the 'Run bot' "
                f"step cap ({run_bot_cap}m), so the job cap fires first and the run ends as `cancelled` "
                "with no artifact upload instead of a legible step failure"
            )

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_job_cap_is_far_shorter_than_a_question_window(self, rel_path: str) -> None:
        for job_name, job in _workflow(rel_path)["jobs"].items():
            assert job["timeout-minutes"] <= _MAX_TOLERABLE_JOB_CAP_MINUTES, (
                f"{rel_path}:{job_name} job cap is {job['timeout-minutes']}m; a hang holds the "
                "concurrency group for that long, and a tournament question is only open for 180m"
            )

    def test_bot_job_caps_do_not_drift_apart(self) -> None:
        # The bot workflows are near-identical by design, and drift is how the unsafe cap
        # hid: test_bot_basic sat at 60 while the other four sat at 300, so the one file
        # whose cap was BELOW the bot's own contract looked like the conservative one.
        caps = {
            rel_path: sorted({job["timeout-minutes"] for job in _workflow(rel_path)["jobs"].values()})
            for rel_path in _BOT_WORKFLOWS
        }
        distinct = {tuple(value) for value in caps.values()}
        assert len(distinct) == 1, f"bot workflow job caps disagree: {caps}"


class TestNonBotWorkflowCapsStayInBand:
    """ci.yaml and claude.yml were swept as the same defect class, so pin their sizing.

    fetch_diagnostic.yaml (the operator-run egress probe) joins the same band: it is
    workflow_dispatch-only and holds no key, but it does hold the concurrency group while
    it makes ~30 sequential network requests, so an uncapped hang there costs the same.

    ``TestEveryJobIsCapped`` only asks whether a cap EXISTS, which a revert to GitHub's
    360-minute default satisfies with one keystroke. These are the workflows nobody
    watches — a required check spinning for six hours, on a workflow that holds a paid
    key while it does it.
    """

    @pytest.mark.parametrize("rel_path", _NON_BOT_WORKFLOWS)
    def test_job_caps_sit_between_the_floor_and_githubs_default(self, rel_path: str) -> None:
        for job_name, job in _workflow(rel_path)["jobs"].items():
            cap = job["timeout-minutes"]
            assert _MIN_JOB_CAP_MINUTES <= cap <= _MAX_NON_BOT_JOB_CAP_MINUTES, (
                f"{rel_path}:{job_name} job cap is {cap}m, outside the "
                f"{_MIN_JOB_CAP_MINUTES}-{_MAX_NON_BOT_JOB_CAP_MINUTES}m band. The worst of 131 "
                "successful CI jobs is 245s, so anything near GitHub's 360-minute default is the "
                "hole this closed, and anything under the floor kills healthy jobs on a slow runner"
            )

    def test_the_non_bot_set_is_what_we_think_it_is(self) -> None:
        # Complement of the pinned full set: a new non-bot workflow lands in the band
        # check above rather than escaping both parametrizations.
        assert _NON_BOT_WORKFLOWS == [
            ".github/workflows/ci.yaml",
            ".github/workflows/claude.yml",
            ".github/workflows/fetch_diagnostic.yaml",
            ".github/workflows/forecast_notification_test.yaml",
        ]


class TestScheduledBotCadence:
    """Every scheduled bot workflow runs hourly, off the hour, and not on top of the others.

    Latency is what the cadence buys. A question the bot never sees before it closes is a
    forfeit — the 2026-09-01 residual round found six in the triple era — and an hourly run
    that finds no new question spends nothing, because cli pins
    ``skip_previously_forecasted_questions`` on for every tournament-shaped mode. The
    Metaculus Cup workflow sat on ``3 0 */2 * *`` (00:03 every second day) until 2026-09-03,
    which could leave a cup question unforecast for most of its window.

    Distinct minutes are the second half. The four workflows sit in SEPARATE concurrency
    groups (``group: ${{ github.workflow }}``), so a shared minute does not queue — it
    starts two or three full bot runs at once, on the same runner pool and against the same
    shared AskNews / Gemini / OpenRouter quotas.

    The Mantic workflow's entries all sit early in the hour, and the pin below derives why.
    """

    mantic_rel_path = ".github/workflows/run_bot_on_mantic.yaml"
    _MANTIC_WINDOW_SECONDS = 3600
    _MANTIC_MIN_ENTRIES = 3

    @staticmethod
    def _schedule(rel_path: str) -> list[str]:
        # PyYAML reads YAML 1.1, where the bare key `on` is the BOOLEAN True, so the
        # triggers block is not under the string "on". Accept either spelling rather than
        # hardcoding the quirk, in case a workflow ever quotes the key.
        workflow: Any = _workflow(rel_path)  # Any: the key below is a bool, not a str
        triggers = workflow.get("on") or workflow.get(True) or {}
        return [entry["cron"] for entry in triggers.get("schedule", [])]

    @property
    def scheduled(self) -> dict[str, list[str]]:
        return {rel: crons for rel in _BOT_WORKFLOWS if (crons := self._schedule(rel))}

    def test_the_scheduled_bot_set_is_what_we_think_it_is(self) -> None:
        # Derived from the files, then pinned: the two test workflows are dispatch-only
        # (spending is the operator's choice), and a new cron on one of them would show up
        # here rather than silently starting to publish on a schedule.
        assert sorted(self.scheduled) == [
            ".github/workflows/run_bot_on_mantic.yaml",
            ".github/workflows/run_bot_on_metaculus_cup.yaml",
            ".github/workflows/run_bot_on_minibench.yaml",
            ".github/workflows/run_bot_on_tournament.yaml",
        ]

    def test_every_scheduled_bot_workflow_is_hourly_and_off_the_hour(self) -> None:
        for rel_path, crons in self.scheduled.items():
            assert len(crons) >= 2, (
                f"{rel_path}: {len(crons)} cron entry(ies). GitHub silently drops schedules under "
                "runner load, so the cadence is split across several entries rather than one */N"
            )
            for cron in crons:
                minute, hour, day, month, weekday = cron.split()
                assert (hour, day, month, weekday) == ("*", "*", "*", "*"), (
                    f"{rel_path}: cron {cron!r} is not hourly. A question is open for about 180 "
                    "minutes and a run that finds nothing new costs nothing, so anything coarser "
                    "than hourly trades forfeits for no saving"
                )
                assert minute.isdigit(), f"{rel_path}: cron {cron!r} has a non-literal minute field"
                assert int(minute) != 0, (
                    f"{rel_path}: cron {cron!r} fires on the hour, into GitHub's :00 scheduling "
                    "burst; every other bot workflow deliberately sits off the hour"
                )

    def test_no_two_scheduled_bot_workflows_share_a_minute(self) -> None:
        minutes: dict[str, str] = {}
        for rel_path, crons in sorted(self.scheduled.items()):
            for cron in crons:
                minute = cron.split()[0]
                assert minute not in minutes, (
                    f"{rel_path} and {minutes[minute]} both fire at :{minute}. They are in separate "
                    "concurrency groups, so that is two simultaneous bot runs on one runner pool "
                    "and one set of shared research quotas, not a queue"
                )
                minutes[minute] = rel_path

    def test_every_mantic_entry_fires_early_enough_for_the_full_research_path(self) -> None:
        """Mantic questions open on the hour with 60-minute windows, so only early entries earn their keep.

        The per-question budget is the close time minus now minus PUBLISH_RESERVE_SECONDS, and a
        budget under TIME_BUDGET_FAST_PATH_THRESHOLD gets only the degraded research path, so an
        entry later than the derived minute buys a worse forecast and the last quarter-hour (under
        TIME_BUDGET_MIN_VIABLE_S) buys nothing. GitHub delivers about 22% of this repository's
        scheduled firings (7 to 23 of 72 a day, measured 2026-08-27 to 2026-09-07), which is why
        there are several early entries rather than one; the durable fix is an external dispatcher
        (docs/operations.md "Scheduling reliability"). Two crons at :17/:47 would have forfeited
        roughly half of all one-hour questions and given the :47 pickup only the fast path.
        """
        minutes = sorted(int(cron.split()[0]) for cron in self.scheduled[self.mantic_rel_path])
        last_full_path_minute = (
            self._MANTIC_WINDOW_SECONDS - PUBLISH_RESERVE_SECONDS - TIME_BUDGET_FAST_PATH_THRESHOLD
        ) // 60
        assert len(minutes) >= self._MANTIC_MIN_ENTRIES, (
            f"{self.mantic_rel_path} has {len(minutes)} cron entry(ies); under GitHub's measured ~22% delivery a "
            f"60-minute Mantic window needs at least {self._MANTIC_MIN_ENTRIES} early chances"
        )
        assert max(minutes) <= last_full_path_minute, (
            f"{self.mantic_rel_path} fires at {minutes}, but a pickup after :{last_full_path_minute} of a "
            "60-minute Mantic window falls under the fast-path threshold and gets only the degraded research "
            "path; later entries buy little and the last quarter-hour buys nothing"
        )


class TestFetchDiagnosticCannotSpend:
    """The one non-bot workflow anybody may dispatch is exempt from the cost gate for a reason.

    ``fetch_diagnostic.yaml`` is on the FREE side of AGENTS.md's cost gate only because it is
    structurally incapable of spending: it holds no secret, so it cannot call an LLM, reach a
    paid research provider or publish to Metaculus, and it is ``workflow_dispatch``-only, so it
    fires only when somebody chooses to fire it. Both facts were stated in a YAML comment and
    asserted nowhere, so a later edit adding ``env: OPENROUTER_API_KEY: ${{ secrets... }}`` or a
    ``schedule:`` cron would pass every other test in this module while turning a free,
    anyone-can-dispatch job into one that spends the operator's credits on a timer.
    """

    rel_path = ".github/workflows/fetch_diagnostic.yaml"

    def test_no_secret_reaches_the_job(self) -> None:
        # Raw text, not the parsed tree: a secret can arrive as job env, step env, a `with:`
        # input, an inline expression in a `run:` line or a reusable-workflow `secrets:` block,
        # and only the source text catches all of them.
        raw = (_REPO_ROOT / self.rel_path).read_text()
        assert "secrets." not in raw, (
            f"{self.rel_path} now references a secret. Its inability to spend is exactly what "
            "puts it on the free side of the cost gate (AGENTS.md) and lets anyone dispatch it; "
            "a workflow that holds a key belongs in the paid list with the operator's sign-off"
        )

    def test_it_fires_only_on_an_explicit_dispatch(self) -> None:
        # `on` is YAML 1.1 truthy, so safe_load keys the trigger block under the bool True, not
        # under the string "on" — hence the parse here rather than the module's `_workflow` helper,
        # whose dict[str, Any] signature cannot be indexed by a bool.
        parsed: Any = yaml.safe_load((_REPO_ROOT / self.rel_path).read_text())
        triggers: Any = parsed.get("on", parsed.get(True))
        assert triggers, f"{self.rel_path} declares no trigger block at all"
        assert sorted(triggers) == ["workflow_dispatch"], (
            f"{self.rel_path} triggers on {sorted(triggers)}. A `schedule:` or `push:` here would "
            "run federal-host probes from the runner IP on somebody else's cadence; every fire of "
            "this workflow is meant to be a deliberate choice"
        )


class TestPaidUrlContextRungIsArmedInEveryBotWorkflow:
    """The resolution-source ladder's one paid rung ships ON, and only where its key is wired.

    ``RESOLUTION_SOURCE_URL_CONTEXT_ENABLED`` defaults off in code, so a bot workflow that forgets
    the line runs the whole ladder free and silently forfeits the pages only Gemini's egress can
    read, while the flag WITHOUT ``GOOGLE_API_KEY`` in the same step is the ``no_api_key``
    misconfiguration, byte-identical in the archive to a flag-off run. Both halves are pinned per
    workflow, so a new bot workflow has to make the same choice deliberately rather than inherit
    the code default. The operator turned the flag on in every bot workflow on 2026-09-04; turning
    it off anywhere is a cost-gate decision (AGENTS.md), not a tidy-up.
    """

    @staticmethod
    def _bot_step_env(workflow: dict[str, Any]) -> dict[str, Any]:
        # The one step that invokes main.py, not a flattened merge across steps: a flag set on
        # the checkout or uv-setup step would satisfy a merged assertion while the bot never saw it.
        bot_steps = [step for step in _steps(workflow) if "main.py" in str(step.get("run", ""))]
        assert len(bot_steps) == 1, f"expected exactly one step invoking main.py, got {len(bot_steps)}"
        return bot_steps[0].get("env") or {}

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_the_flag_is_on_in_the_bot_step(self, rel_path: str) -> None:
        env = self._bot_step_env(_workflow(rel_path))
        assert env.get("RESOLUTION_SOURCE_URL_CONTEXT_ENABLED") == "true", (
            f"{rel_path} does not set RESOLUTION_SOURCE_URL_CONTEXT_ENABLED: 'true' on its bot step, so "
            "its resolution-source ladder runs without the paid url_context rung the other bot "
            "workflows have on"
        )

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_the_key_the_rung_bills_to_is_wired_in_the_same_step(self, rel_path: str) -> None:
        env = self._bot_step_env(_workflow(rel_path))
        assert "secrets." in str(env.get("GOOGLE_API_KEY", "")), (
            f"{rel_path} arms the paid url_context rung but wires no GOOGLE_API_KEY secret on the bot "
            "step, so every admitted read would be a no_api_key skip and the run would read in the "
            "archive exactly like one with the flag off"
        )


class TestManticWorkflowSpendsOnlyPersonalKeys:
    """The Mantic workflow must never hold the Metaculus-donated OpenRouter key, and the
    Metaculus workflows must never lose it.

    Metaculus donates ``OAI_ANTH_OPENROUTER_KEY`` for its own tournaments, so a run that
    forecasts for Mantic (a different platform) spends only the operator's personal keys.
    The code side fails shut (``cli._assert_personal_keys_only`` refuses to start while
    ``DONATED_OPENROUTER_KEY_ENABLED`` reads true), and this pin is the workflow side of the
    same rule: the secret is simply not wired, so no code-path change can reach it. Raw
    text, not the parsed tree, for the same reason as ``TestFetchDiagnosticCannotSpend``: a
    secret can arrive as job env, step env or an inline expression, and only the source
    catches all of them. The inverse pin exists because the Mantic file was copied from the
    tournament one; copying it back over a Metaculus workflow would silently move every
    Metaculus run onto the personal key.
    """

    mantic_rel_path = ".github/workflows/run_bot_on_mantic.yaml"
    _DONATED_KEY_SECRET = "OAI_ANTH_OPENROUTER_KEY"

    @property
    def mantic_raw(self) -> str:
        return (_REPO_ROOT / self.mantic_rel_path).read_text()

    def test_the_donated_key_and_the_metaculus_token_are_absent(self) -> None:
        for forbidden in (self._DONATED_KEY_SECRET, "METACULUS_TOKEN"):
            assert forbidden not in self.mantic_raw, (
                f"{self.mantic_rel_path} references {forbidden}. A Mantic run must spend only the "
                "operator's personal keys and must not depend on Metaculus credentials at all"
            )

    def test_the_mantic_token_is_wired_from_a_secret(self) -> None:
        assert "secrets.MANTIC_TOKEN" in self.mantic_raw, (
            f"{self.mantic_rel_path} does not wire MANTIC_TOKEN from a repository secret, so the "
            "client cannot authenticate and every publish would 401"
        )

    def test_both_donated_routing_flags_are_forced_off(self) -> None:
        for flag in ("DONATED_OPENROUTER_KEY_ENABLED", "GEMINI_USE_DONATED_OPENROUTER_KEY"):
            assert f"{flag}: 'false'" in self.mantic_raw, (
                f"{self.mantic_rel_path} does not set {flag}: 'false'. Both default to true in code, "
                "and the master switch is what lets cli._assert_personal_keys_only start the run"
            )

    def test_the_run_step_selects_mantic_mode(self) -> None:
        assert "--mode mantic" in self.mantic_raw, (
            f"{self.mantic_rel_path} does not pass --mode mantic, so it would forecast the Metaculus "
            "tournament with no METACULUS_TOKEN and publish nothing"
        )

    @pytest.mark.parametrize("rel_path", [p for p in _BOT_WORKFLOWS if not p.endswith("run_bot_on_mantic.yaml")])
    def test_every_metaculus_bot_workflow_still_wires_the_donated_key(self, rel_path: str) -> None:
        raw = (_REPO_ROOT / rel_path).read_text()
        assert f"{self._DONATED_KEY_SECRET}: ${{{{ secrets.{self._DONATED_KEY_SECRET} }}}}" in raw, (
            f"{rel_path} no longer wires {self._DONATED_KEY_SECRET} from its secret. Every Metaculus "
            "workflow routes OpenAI, Anthropic and Google calls through the donated key first; "
            "dropping it moves the whole run onto the personal key. If this file was overwritten "
            "with a copy of the Mantic workflow, that is the mistake this pin exists to catch"
        )
