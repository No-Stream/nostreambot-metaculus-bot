"""Guards on the scheduled archive-sync job's failure semantics.

These assert on the Makefile recipe, the launchd wrapper, and the plist rather than on
Python behavior, because that is where the 2026-06/07 data-loss bug lived: every
scheduled run from 2026-06-28 through 2026-07-26 failed and nothing noticed. The three
compounding causes, one test class each:

1. ``sync_all`` ran the Metaculus-comment backfill first and let its failure abort the
   recipe under ``set -euo pipefail``, so a failure in the half whose data never expires
   killed the half that GitHub deletes at 90 days.
2. ``run_sync.sh`` invoked make with no network-readiness wait, and launchd runs a
   missed job at the next wake — before Wi-Fi re-associates.
3. Nothing surfaced the failure: no sentinel, no notification, and the plist woke once a
   week so one bad wake cost a full week.

They are cheap string assertions on files that no test otherwise reads, which is exactly
why the regression was invisible for six weeks.
"""

from __future__ import annotations

import plistlib
import re
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MAKEFILE = _REPO_ROOT / "Makefile"
_SYNC_DIR = _REPO_ROOT / "scripts" / "research_sync"
_RUN_SYNC = _SYNC_DIR / "run_sync.sh"
# The committed artifact is the placeholder TEMPLATE, never a concrete plist: launchd needs an
# absolute ProgramArguments path, so any installable plist names one machine's checkout. Only the
# template is in git; the README's sed step generates the real file, which is gitignored.
_PLIST_TEMPLATE = _SYNC_DIR / "com.metaculusbot.research-sync.plist.template"
_REPO_ROOT_PLACEHOLDER = "__REPO_ROOT__"
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"

# Every workflow that runs the bot: four scheduled prod tournaments plus the two
# dispatch-only test workflows (test_bot / test_bot_basic both match *bot*).
# .y*ml, not .yaml: the exact-set assertion below is what forces a NEW bot workflow to
# satisfy this invariant, and a .yml-suffixed one would slip past it (claude.yml already
# uses that suffix, and both spellings are live here).
_BOT_WORKFLOWS = sorted(p.relative_to(_REPO_ROOT).as_posix() for p in _WORKFLOW_DIR.glob("*bot*.y*ml"))


def _sync_all_recipe() -> list[str]:
    """The command lines of the Makefile's ``sync_all`` recipe, in order.

    A recipe body is the tab-indented block following the target line, so we slice from
    ``sync_all:`` to the first line that is neither tab-indented nor blank.
    """
    lines = _MAKEFILE.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("sync_all:"))
    body: list[str] = []
    for line in lines[start + 1 :]:
        if not line.startswith("\t"):
            if line.strip() == "":
                continue
            break
        body.append(line.lstrip("\t"))
    return body


class TestSyncAllOrdersTheExpiringHalfSafely:
    """The GHA pull must run even when the Metaculus backfill fails."""

    def test_backfill_failure_is_non_fatal(self) -> None:
        # The `-` prefix is the fix: make reports the failure as "(ignored)" and keeps
        # going. Without it, `set -e` in run_sync.sh aborts before any artifact
        # downloads. Verified end-to-end: a `-false` recipe line exits 0 and runs the
        # rest of the recipe.
        backfill_lines = [line for line in _sync_all_recipe() if "backfill_research_from_comments.py" in line]
        assert len(backfill_lines) == 1, f"expected exactly one backfill line, got {backfill_lines}"
        assert backfill_lines[0].startswith("-"), (
            "sync_all's backfill line must be `-`-prefixed (non-fatal). Metaculus comments "
            "never expire; GHA artifacts are deleted at 90 days, so a backfill failure must "
            "not abort the GHA pull. See tests/test_research_sync_job.py docstring."
        )

    def test_backfill_still_precedes_the_gha_driver(self) -> None:
        # Ordering is load-bearing for a different reason than the `-` is: sync_all.py's
        # research build loads comments_backfill.jsonl, and there is exactly ONE build,
        # so the backfill must write that file first or its records miss this run.
        recipe = _sync_all_recipe()
        backfill_at = next(i for i, line in enumerate(recipe) if "backfill_research_from_comments.py" in line)
        driver_at = next(i for i, line in enumerate(recipe) if "scripts/sync_all.py" in line)
        assert backfill_at < driver_at, (
            "the comment backfill must precede scripts/sync_all.py so its "
            "comments_backfill.jsonl is on disk for the single research build"
        )


class TestRunSyncWaitsForTheNetwork:
    """launchd fires at the next wake, when the network is typically still down."""

    def test_network_preflight_runs_before_make(self) -> None:
        script = _RUN_SYNC.read_text()
        probe_at = script.find("if ! wait_for_network")
        # Anchor on the INVOCATION, not the bare string: `make sync_all` also appears in
        # the file's header comment, which would make the ordering assertion vacuous.
        make_at = script.find("if ! make sync_all")
        assert probe_at != -1, "run_sync.sh must call a network-readiness preflight"
        assert make_at != -1, "run_sync.sh must still invoke `make sync_all`"
        assert probe_at < make_at, "the network preflight must run BEFORE `make sync_all`"

    def test_preflight_is_bounded_and_actually_probes(self) -> None:
        script = _RUN_SYNC.read_text()
        # A bounded loop, not an unbounded wait: a job that hangs forever is its own
        # silent failure. And the probe must be a real request, so a permanently-offline
        # wake reports rather than proceeding into a doomed make.
        assert "curl -sf -m 5 https://api.github.com/" in script
        tries = re.search(r"NETWORK_WAIT_TRIES=(\d+)", script)
        sleep_s = re.search(r"NETWORK_WAIT_SLEEP_S=(\d+)", script)
        assert tries is not None, "the wait must be bounded by explicit constants"
        assert sleep_s is not None, "the wait must be bounded by explicit constants"
        assert 0 < int(tries.group(1)) * int(sleep_s.group(1)) <= 900, "bounded wait should cap out within ~15 min"


class TestSyncFailureIsVisible:
    """Six weeks of staleness happened because a failure produced no signal."""

    def test_failure_writes_a_sentinel_and_notifies(self) -> None:
        script = _RUN_SYNC.read_text()
        assert "LAST_SYNC_FAILED" in script, "a failed run must leave a greppable sentinel file"
        assert "osascript" in script, (
            "a failed run must fire a macOS notification — the dated logfile alone is what nobody read for six weeks"
        )
        assert "display notification" in script, (
            "a failed run must fire a macOS notification — the dated logfile alone is what nobody read for six weeks"
        )

    def test_sentinel_is_cleared_only_on_a_green_run(self) -> None:
        script = _RUN_SYNC.read_text()
        # A sentinel that outlives its failure is as misleading as none at all.
        assert 'rm -f "${FAILURE_SENTINEL}"' in script
        clear_at = script.find('rm -f "${FAILURE_SENTINEL}"')
        make_at = script.find("if ! make sync_all")
        assert make_at < clear_at, "the sentinel must be cleared AFTER a successful make, not before"

    def test_failure_exits_non_zero(self) -> None:
        script = _RUN_SYNC.read_text()
        assert script.count("exit 1") >= 2, "both the preflight and the make failure paths must exit non-zero"


def _workflow(rel_path: str) -> dict:
    return yaml.safe_load((_REPO_ROOT / rel_path).read_text())


def _run_bot_env(workflow: dict) -> dict[str, str]:
    """The env the step that invokes ``main.py`` actually runs under.

    Scoped to that one step (plus job-level env) rather than merged across all of them: a
    flag set on the checkout or uv-setup step would satisfy a flattened assertion while the
    bot process never saw it.
    """
    bot_steps = [
        (job, step)
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "main.py" in str(step.get("run", ""))
    ]
    assert len(bot_steps) == 1, f"expected exactly one step invoking main.py, got {len(bot_steps)}"
    job, step = bot_steps[0]
    return {**(job.get("env") or {}), **(step.get("env") or {})}


def _upload_step(workflow: dict) -> dict:
    """The single ``actions/upload-artifact`` step in a bot workflow."""
    steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
    ]
    assert len(steps) == 1, f"expected exactly one upload-artifact step, got {len(steps)}"
    return steps[0]


class TestBotWorkflowsArchiveTheirResearch:
    """Every bot workflow must persist its research AND upload it under a harvestable name.

    Both halves are needed and each was missing from the two test workflows until
    2026-08-03: they set ``RAW_RESEARCH_LOG_ENABLED`` but never
    ``PERSIST_RESEARCH_ENABLED``, and uploaded ``logs-<run_id>`` with only ``run_logs/``.
    ``sync_all.py`` gates the research harvest on the ``research-`` prefix, so even a
    correctly-written JSONL under a ``logs-*`` name contributes nothing. Measured cost:
    three runs (29718821482, 30039072456, 30321419722) whose raw provider payloads and
    telemetry markers we still hold, with no assembled per-question research at all.

    Asserted on repo-relative paths only — never a path derived from the developer's
    checkout, which passes locally by construction and fails on the first CI run.
    """

    def test_every_bot_workflow_is_covered_by_this_invariant(self) -> None:
        # Pinned as an exact set so a NEW bot workflow has to either satisfy the
        # invariant below or fail here. Silently escaping it is the failure mode that
        # cost us the three runs above.
        assert _BOT_WORKFLOWS == [
            ".github/workflows/run_bot_on_mantic.yaml",
            ".github/workflows/run_bot_on_metaculus_cup.yaml",
            ".github/workflows/run_bot_on_minibench.yaml",
            ".github/workflows/run_bot_on_tournament.yaml",
            ".github/workflows/test_bot.yaml",
            ".github/workflows/test_bot_basic.yaml",
        ]

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_raw_research_logging_implies_research_persistence(self, rel_path: str) -> None:
        env = _run_bot_env(_workflow(rel_path))
        if env.get("RAW_RESEARCH_LOG_ENABLED") != "true":
            pytest.skip(f"{rel_path} does not log raw research payloads")
        assert env.get("PERSIST_RESEARCH_ENABLED") == "true", (
            f"{rel_path} archives RAW provider payloads but not the assembled research text — "
            "the raw log is only useful next to the briefing the forecasters actually read"
        )

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_artifact_name_uses_the_harvestable_research_prefix(self, rel_path: str) -> None:
        name = _upload_step(_workflow(rel_path))["with"]["name"]
        assert name.startswith("research-"), (
            f"{rel_path} uploads as {name!r}; scripts/sync_all.py harvests research only from "
            "artifacts named research-* (RESEARCH_ARTIFACT_PREFIX)"
        )

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_artifact_path_carries_research_outputs_and_run_logs(self, rel_path: str) -> None:
        paths = _upload_step(_workflow(rel_path))["with"]["path"].split()
        assert "research_outputs/" in paths, f"{rel_path} would upload no research JSONL"
        assert "run_logs/" in paths, f"{rel_path} would upload no telemetry log"

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_upload_runs_even_when_the_bot_run_failed(self, rel_path: str) -> None:
        # Load-bearing for the crash-path flush in cli.py: the partial batch a crashed run
        # flushes only reaches the archive because this step is unconditional.
        assert _upload_step(_workflow(rel_path))["if"] == "always()"


class TestPlistSurvivesOneBadWake:
    def test_two_wakes_per_week(self) -> None:
        with _PLIST_TEMPLATE.open("rb") as handle:
            plist = plistlib.load(handle)
        intervals = plist["StartCalendarInterval"]
        assert isinstance(intervals, list), (
            "StartCalendarInterval must be a LIST of wake times — a single dict means one bad "
            "wake costs a whole week against a 90-day retention window"
        )
        assert len(intervals) >= 2
        assert {interval["Weekday"] for interval in intervals} == {0, 3}, "expect Sun + Wed wakes"

    def test_throttle_interval_prevents_a_relaunch_loop(self) -> None:
        with _PLIST_TEMPLATE.open("rb") as handle:
            plist = plistlib.load(handle)
        # run_sync.sh now exits non-zero on failure, and launchd relaunches a failed job
        # promptly by default. Throttle so a network outage gets a real retry, not a loop.
        assert plist["ThrottleInterval"] >= 600

    def test_template_points_at_the_wrapper_this_suite_checks(self) -> None:
        with _PLIST_TEMPLATE.open("rb") as handle:
            plist = plistlib.load(handle)
        # Otherwise every assertion above could pass while launchd runs something else.
        #
        # Asserted as PLACEHOLDER + repo-relative suffix, and do not "tighten" this to a
        # real path: an installable plist must carry an absolute path to one machine's
        # checkout (launchd has no notion of a repo-relative path), while _RUN_SYNC is
        # derived from __file__. Comparing those for equality passes only on the machine
        # that wrote the plist and can never pass in CI, where the checkout lives
        # somewhere else — which is exactly how this test shipped red (CI run
        # 30321344705, 2026-07-27). The template sidesteps that: the placeholder is
        # machine-independent, so this assertion means the same thing everywhere.
        program_arguments = plist["ProgramArguments"]
        assert len(program_arguments) == 1, f"expect a bare wrapper invocation, got {program_arguments}"
        invoked = program_arguments[0]
        wrapper_suffix = _RUN_SYNC.relative_to(_REPO_ROOT).as_posix()
        assert invoked == f"{_REPO_ROOT_PLACEHOLDER}/{wrapper_suffix}", (
            f"the template must invoke {_REPO_ROOT_PLACEHOLDER}/{wrapper_suffix} — the wrapper whose "
            f"contents the rest of this class asserts on, under the placeholder the README's sed step "
            f"substitutes; got {invoked!r}"
        )

    def test_template_carries_no_developer_path(self) -> None:
        # The whole point of the template: a committed absolute path leaks the operator's
        # home directory on a public fork AND is the local-green-proves-nothing trap, since
        # a path derived from one checkout passes locally by construction.
        text = _PLIST_TEMPLATE.read_text()
        assert _REPO_ROOT_PLACEHOLDER in text, "the template must keep the placeholder unsubstituted"
        for absolute_prefix in ("/Users/", "/home/"):
            assert absolute_prefix not in text, (
                f"the committed template must not contain {absolute_prefix!r} — substitute "
                f"{_REPO_ROOT_PLACEHOLDER} at install time instead (the generated .plist is gitignored)"
            )
