"""The config-era boundaries and the per-record era tags.

Two concerns kept apart. The boundary VALUES are anchored to the git history, never to the
constants themselves: an assertion that reads a constant and compares it to that constant passes
for any value it happens to hold, which is how a boundary carrying an authoring date survived a
green suite for four months. The tag VOCABULARY is asserted at literal instants, because
``era_gap --era-field triple_subera_fine`` and the clip sweep's ``pre_flip`` / ``post_flip``
slices select on the exact strings, so a renamed tag breaks a standing instrument silently.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path

import pytest

from metaculus_bot.performance_analysis import eras
from metaculus_bot.performance_analysis.eras import (
    B4E9DF0_MERGED_AT,
    DRY_KEY_FIX_MERGED_AT,
    FALL_CONFIG_MERGED_AT,
    FALL_GEMINI4_ARGON_MERGED_AT,
    FALL_GEMINI4_CARBON_MERGED_AT,
    FALL_GEMINI4_HYDROGEN_MERGED_AT,
    FALL_GPT6_OPUS55_MERGED_AT,
    FALL_GPT61_SOL_MERGED_AT,
    FALL_TARGET_MERGED_AT,
    FALL_TERRA_RESEARCH_MERGED_AT,
    FT_0292_MERGED_AT,
    GRID_SCALED_MAX_STEP_MERGED_AT,
    IMPERSONATE_RUNG_MERGED_AT,
    JULY25_MERGED_AT,
    LINTERS_MERGED_AT,
    RANKED_MARKET_MERGED_AT,
    TIME_BUDGET_MERGED_AT,
    WIDENING_FLIP_MERGED_AT,
    era_of,
    ft_unfreeze_side_of,
    post_linters_merge_of,
    triple_subera_fine_of,
    triple_subera_of,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The (merge, constant) pairs git is asked about. Every boundary belongs here.
BOUNDARY_MERGES: tuple[tuple[str, str, datetime], ...] = (
    ("0e85e1b", "WIDENING_FLIP_MERGED_AT", WIDENING_FLIP_MERGED_AT),
    ("b4e9df0", "B4E9DF0_MERGED_AT", B4E9DF0_MERGED_AT),
    ("325b1b0", "FT_0292_MERGED_AT", FT_0292_MERGED_AT),
    ("73e4782", "JULY25_MERGED_AT", JULY25_MERGED_AT),
    ("c3c91cb", "DRY_KEY_FIX_MERGED_AT", DRY_KEY_FIX_MERGED_AT),
    ("bfd5df2", "RANKED_MARKET_MERGED_AT", RANKED_MARKET_MERGED_AT),
    ("951f8e4", "TIME_BUDGET_MERGED_AT", TIME_BUDGET_MERGED_AT),
    ("eded193", "LINTERS_MERGED_AT", LINTERS_MERGED_AT),
    ("8d5a082", "FALL_CONFIG_MERGED_AT", FALL_CONFIG_MERGED_AT),
    ("a9cbe03", "IMPERSONATE_RUNG_MERGED_AT", IMPERSONATE_RUNG_MERGED_AT),
    ("660fd35", "FALL_TARGET_MERGED_AT", FALL_TARGET_MERGED_AT),
    ("021e059", "FALL_GPT6_OPUS55_MERGED_AT", FALL_GPT6_OPUS55_MERGED_AT),
    ("6489289", "FALL_GPT61_SOL_MERGED_AT", FALL_GPT61_SOL_MERGED_AT),
    ("28e82b6", "FALL_GEMINI4_ARGON_MERGED_AT", FALL_GEMINI4_ARGON_MERGED_AT),
    ("2c45ed7", "FALL_TERRA_RESEARCH_MERGED_AT", FALL_TERRA_RESEARCH_MERGED_AT),
    ("e5b3b52", "FALL_GEMINI4_HYDROGEN_MERGED_AT", FALL_GEMINI4_HYDROGEN_MERGED_AT),
    ("fdc1483", "FALL_GEMINI4_CARBON_MERGED_AT", FALL_GEMINI4_CARBON_MERGED_AT),
)

# b8d730f authored the widening flip six days before 0e85e1b merged it: the trap this file guards.
WIDENING_FLIP_AUTHORING_SHA = "b8d730f"
WIDENING_FLIP_AUTHORING_MIDNIGHT = datetime(2026, 5, 12, tzinfo=UTC)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(_REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _commit_facts(sha: str) -> tuple[str, datetime]:
    """``(full sha, committer instant in UTC)``, skipping when this clone lacks the object."""
    result = _git("log", "-1", "--format=%H %cI", sha)
    if result.returncode != 0:
        pytest.skip(f"this clone cannot resolve {sha}; CI's test job pins fetch-depth: 0 so these run there")
    full_sha, _, committed = result.stdout.strip().partition(" ")
    return full_sha, datetime.fromisoformat(committed).astimezone(UTC)


@cache
def _first_parent_shas() -> frozenset[str]:
    """Every commit on the main line: what "landed on main" means, and what prod therefore ran."""
    for ref in ("main", "origin/main"):
        result = _git("rev-list", "--first-parent", ref)
        if result.returncode == 0:
            return frozenset(result.stdout.split())
    pytest.skip("this clone carries no main ref; CI's test job pins fetch-depth: 0 so these run there")


class TestBoundariesAreMergeToMainCommitterTimestamps:
    """Boundaries are anchored to git, because prod runs from ``main``.

    A config change is live only from the moment its merge commit lands there, so each constant
    is asserted against a fact it cannot define: the committer timestamp of that merge, and the
    merge's presence on ``main``'s first-parent history.
    """

    @pytest.mark.parametrize(("sha", "label", "constant"), BOUNDARY_MERGES, ids=[m[1] for m in BOUNDARY_MERGES])
    def test_boundary_equals_its_merge_committer_timestamp(self, sha: str, label: str, constant: datetime):
        _, committed_at = _commit_facts(sha)
        assert constant == committed_at, (
            f"{label} must be {sha}'s COMMITTER timestamp ({committed_at.isoformat()}), got {constant.isoformat()}"
        )

    @pytest.mark.parametrize(("sha", "label", "constant"), BOUNDARY_MERGES, ids=[m[1] for m in BOUNDARY_MERGES])
    def test_boundary_merge_is_on_mains_first_parent_history(self, sha: str, label: str, constant: datetime):
        full_sha, _ = _commit_facts(sha)
        assert full_sha in _first_parent_shas(), (
            f"{label} is anchored to {sha}, which never landed on main directly, so prod never ran it as such"
        )

    def test_the_widening_flip_is_anchored_to_the_merge_not_the_authoring_commit(self):
        """``b8d730f`` reached main only inside ``0e85e1b``, so its own timestamp is not a boundary."""
        authoring_sha, authored_at = _commit_facts(WIDENING_FLIP_AUTHORING_SHA)
        assert authoring_sha not in _first_parent_shas()
        assert authored_at != WIDENING_FLIP_MERGED_AT
        assert authored_at < WIDENING_FLIP_MERGED_AT


class TestWideningFlipCannotRegressToTheAuthoringDate:
    """2026-05-12 midnight was retyped forward through fourteen copies of the era map.

    It is ``b8d730f``'s authoring date truncated to midnight, six days before ``0e85e1b`` landed
    the flip on main. Zero resolved records fall in that window today, so the defect is latent: a
    backfill recovering May 12-18 records would activate it.
    """

    def test_the_boundary_is_not_the_authoring_midnight(self):
        assert WIDENING_FLIP_MERGED_AT != WIDENING_FLIP_AUTHORING_MIDNIGHT

    @pytest.mark.parametrize(
        "created_at",
        [
            "2026-05-12T00:00:00Z",
            "2026-05-12T10:32:02Z",  # b8d730f's own authoring instant
            "2026-05-15T12:00:00Z",
            "2026-05-18T17:21:18Z",  # one second before 0e85e1b landed
        ],
    )
    def test_the_six_day_gap_window_tags_pre_flip(self, created_at: str):
        assert era_of({"bot_comment_created_at": created_at}) == "pre_flip"


class TestEraTag:
    """``config_era``: the coarse era, at literal instants rather than through the constants."""

    @pytest.mark.parametrize(
        ("created_at", "expected"),
        [
            ("2026-04-01T00:00:00Z", "pre_flip"),
            ("2026-05-18T17:21:19Z", "post_flip"),
            ("2026-07-01T00:00:00Z", "post_flip"),
            ("2026-07-21T17:07:36Z", "post_flip"),
            ("2026-07-21T17:07:37Z", "triple_era"),
            ("2026-09-07T12:00:00Z", "triple_era"),
        ],
    )
    def test_era_at_literal_instants(self, created_at: str, expected: str):
        assert era_of({"bot_comment_created_at": created_at}) == expected

    @pytest.mark.parametrize("raw", [None, "", "not-a-date", "2026-13-45T99:00:00Z"])
    def test_absent_or_unparseable_timestamp_is_not_attributed_to_an_era(self, raw: str | None):
        assert era_of({"bot_comment_created_at": raw}) == "no_ts"
        assert era_of({}) == "no_ts"

    @pytest.mark.parametrize(
        "boundary",
        [WIDENING_FLIP_MERGED_AT, B4E9DF0_MERGED_AT],
        ids=["widening_flip", "triple_start"],
    )
    def test_the_boundary_instant_belongs_to_the_later_era(self, boundary: datetime):
        """Half-open ``[start, end)``, asserted as a convention rather than against a date."""
        later = era_of({"bot_comment_created_at": boundary.isoformat()})
        earlier = era_of({"bot_comment_created_at": (boundary - timedelta(microseconds=1)).isoformat()})
        assert later != earlier

    def test_the_three_iso_shapes_the_archive_carries_agree(self):
        """A naive read of a -07:00 instant is seven hours off and can cross a boundary."""
        shapes = ("2026-07-21T18:07:37Z", "2026-07-21T11:07:37-07:00", "2026-07-21T18:07:37")
        assert {era_of({"bot_comment_created_at": raw}) for raw in shapes} == {"triple_era"}


class TestSuberaTags:
    """The two sub-era partitions, and the vocabulary the standing instruments select on."""

    @pytest.mark.parametrize(
        ("created_at", "coarse", "fine"),
        [
            ("2026-07-21T17:07:37Z", "triple_pre_market", "pre_ft_unfreeze"),
            ("2026-07-24T19:16:26Z", "triple_pre_market", "ft_0292"),
            ("2026-07-26T04:38:40Z", "triple_pre_market", "july25"),
            ("2026-07-28T03:07:53Z", "triple_pre_market", "post_dry_key_fix"),
            ("2026-08-06T01:28:49Z", "triple_ranked_market", "ranked_markets"),
            ("2026-08-26T17:23:30Z", "triple_time_budget", "time_budget"),
            ("2026-09-05T01:59:24Z", "triple_fall_config", "fall_config"),
            ("2026-09-05T15:31:40Z", "triple_fall_config", "fall_config"),
            ("2026-09-07T05:52:20Z", "triple_fall_config", "fall_config"),
            ("2026-09-24T20:50:53Z", "triple_fall_config", "fall_gpt6_opus55"),
            ("2026-09-29T21:36:57Z", "triple_fall_config", "fall_gpt61_sol"),
            ("2026-10-14T15:14:09Z", "triple_fall_config", "fall_gemini4_argon"),
            ("2026-10-28T20:03:31Z", "triple_fall_config", "fall_terra_research"),
            ("2026-11-13T12:43:02Z", "triple_fall_config", "fall_gemini4_hydrogen"),
            ("2026-11-13T14:36:58Z", "triple_fall_config", "fall_gemini4_carbon"),
        ],
    )
    def test_subera_at_literal_instants(self, created_at: str, coarse: str, fine: str):
        record = {"bot_comment_created_at": created_at}
        assert triple_subera_of(record) == coarse
        assert triple_subera_fine_of(record) == fine

    @pytest.mark.parametrize("created_at", ["2026-07-21T17:07:36Z", "2026-05-01T00:00:00Z", None])
    def test_a_record_outside_the_triple_era_has_no_subera(self, created_at: str | None):
        record = {"bot_comment_created_at": created_at}
        assert triple_subera_of(record) is None
        assert triple_subera_fine_of(record) is None
        assert ft_unfreeze_side_of(record) is None
        assert post_linters_merge_of(record) is None

    def test_the_three_september_merges_pool_into_one_coarse_bucket(self):
        """PRs #66 / #67 / #68 landed inside three days: separable rows would slice one config."""
        september = ["2026-09-05T01:59:24Z", "2026-09-05T15:31:40Z", "2026-09-07T05:52:20Z"]
        assert {triple_subera_of({"bot_comment_created_at": raw}) for raw in september} == {"triple_fall_config"}

    def test_the_fall_roster_merges_pool_into_one_coarse_bucket(self):
        """The six fall roster/research merges are labelled finely but pooled for the season's reads."""
        fall = [
            "2026-09-24T20:50:53Z",
            "2026-09-29T21:36:57Z",
            "2026-10-14T15:14:09Z",
            "2026-10-28T20:03:31Z",
            "2026-11-13T12:43:02Z",
            "2026-11-13T14:36:58Z",
        ]
        assert {triple_subera_of({"bot_comment_created_at": raw}) for raw in fall} == {"triple_fall_config"}

    def test_the_tag_vocabulary_the_standing_instruments_read(self):
        """``era_gap`` selects arms on these fine names and the clip sweep slices on the coarse eras."""
        assert eras.ERAS == ("pre_flip", "post_flip", "triple_era", "no_ts")
        assert eras.TRIPLE_SUBERAS == (
            "triple_pre_market",
            "triple_ranked_market",
            "triple_time_budget",
            "triple_fall_config",
        )
        assert eras.FINE_SUBERAS == (
            "pre_ft_unfreeze",
            "ft_0292",
            "july25",
            "post_dry_key_fix",
            "ranked_markets",
            "time_budget",
            "fall_config",
            "fall_gpt6_opus55",
            "fall_gpt61_sol",
            "fall_gemini4_argon",
            "fall_terra_research",
            "fall_gemini4_hydrogen",
            "fall_gemini4_carbon",
        )
        assert eras.POST_TIME_BUDGET_FINE == (
            "time_budget",
            "fall_config",
            "fall_gpt6_opus55",
            "fall_gpt61_sol",
            "fall_gemini4_argon",
            "fall_terra_research",
            "fall_gemini4_hydrogen",
            "fall_gemini4_carbon",
        )

    def test_each_subera_table_is_ordered_and_distinct(self):
        """``_subera_of`` takes the latest open row, so a duplicate or out-of-order row mis-tags."""
        for table in (eras.TRIPLE_SUBERA_TABLE, eras.FINE_SUBERA_TABLE):
            opens = [instant for _, instant in table]
            assert opens == sorted(opens)
            assert len({name for name, _ in table}) == len(table)


class TestProvenanceTags:
    @pytest.mark.parametrize(
        ("created_at", "expected"),
        [
            ("2026-07-21T17:07:37Z", "pre_ft_0292"),
            ("2026-07-24T19:16:25Z", "pre_ft_0292"),
            ("2026-07-24T19:16:26Z", "ft_0292"),
        ],
    )
    def test_ft_unfreeze_side(self, created_at: str, expected: str):
        assert ft_unfreeze_side_of({"bot_comment_created_at": created_at}) == expected

    @pytest.mark.parametrize(
        ("created_at", "expected"),
        [("2026-08-28T03:54:02Z", False), ("2026-08-28T03:54:03Z", True)],
    )
    def test_post_linters_merge_is_recorded_but_opens_no_subera(self, created_at: str, expected: bool):
        record = {"bot_comment_created_at": created_at}
        assert post_linters_merge_of(record) is expected
        assert "linters" not in str(triple_subera_fine_of(record))


class TestErasIsTheOnlyHome:
    def test_no_other_module_redeclares_a_boundary_instant(self):
        """An alias shares the object; a retyped literal is merely equal, which is the drift."""
        boundaries = {label: constant for _, label, constant in BOUNDARY_MERGES}
        offenders: list[str] = []
        package = importlib.import_module("metaculus_bot.performance_analysis")
        for module_info in pkgutil.iter_modules(package.__path__):
            if module_info.name in ("eras", "__main__"):  # importing __main__ runs the CLI
                continue
            module = importlib.import_module(f"metaculus_bot.performance_analysis.{module_info.name}")
            for attr, value in vars(module).items():
                if not isinstance(value, datetime):
                    continue
                offenders += [
                    f"{module_info.name}.{attr} duplicates eras.{label}"
                    for label, boundary in boundaries.items()
                    if value == boundary and value is not boundary
                ]
        assert offenders == [], f"import from eras.py instead of retyping the instant: {offenders}"

    def test_the_grid_scaled_max_step_gate_is_the_same_object_as_the_triple_boundary(self):
        """9f1175c rode b4e9df0, so the era split and the max-step gate are one instant."""
        assert GRID_SCALED_MAX_STEP_MERGED_AT is B4E9DF0_MERGED_AT

    def test_every_boundary_is_covered_by_the_git_anchored_pins(self):
        """A new boundary must arrive with its merge sha, or it ships unpinned."""
        declared = {
            value for name, value in vars(eras).items() if name.endswith("_MERGED_AT") and isinstance(value, datetime)
        }
        assert declared == {constant for _, _, constant in BOUNDARY_MERGES}

    def test_the_serialized_boundary_map_carries_every_instant(self):
        """``BOUNDARIES_UTC`` is what a round writes into counts_by_era.json."""
        assert set(eras.BOUNDARIES_UTC.values()) == {constant.isoformat() for _, _, constant in BOUNDARY_MERGES}
