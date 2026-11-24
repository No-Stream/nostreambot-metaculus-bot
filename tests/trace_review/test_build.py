"""The builder end to end over the fixture workspace: tables, joins, nulls, overlay, failure and cache.

Every test reads the bundle the real ``build_index`` wrote; the fixtures (conftest.py) supply
inputs only, so each assertion is about how the builder combined them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from metaculus_bot.constants import MANTIC_TOKEN_ENV, PLATFORM_MANTIC, PLATFORM_METACULUS
from metaculus_bot.performance_analysis import collector
from metaculus_bot.performance_analysis.member_replay import replay_solo_score
from metaculus_bot.trace_review.__main__ import main, parse_args
from metaculus_bot.trace_review.build import build_index
from metaculus_bot.trace_review.config import (
    BUNDLE_TABLE_FILES,
    FORECASTS_FILE,
    MANIFEST_FILE,
    MEMBERS_FILE,
    RECORD_ORIGIN_FRESH,
    RECORD_ORIGIN_ROUND,
    TELEMETRY_FILE,
    TEXTS_FILE,
)
from metaculus_bot.trace_review.member_rows import member_rows
from metaculus_bot.trace_review.schema import (
    FORECAST_COLUMNS,
    MEMBER_COLUMNS,
    TELEMETRY_COLUMNS,
    TEXT_COLUMNS,
    cast_frame,
)
from metaculus_bot.trace_review.sources import MissingPlatformTokenError, posts_needing_comments
from tests.trace_review.workspace_fixtures import (
    ANNULLED,
    COLLIDING_BINARY,
    COLLIDING_NUMERIC,
    DATE_QUESTION,
    DISCRETE,
    EARLY_BINARY,
    MANTIC_DISCRETE,
    MANTIC_MULTI_RESOLUTION,
    MANTIC_PER_BIN_MASSES,
    NAMED_MC,
    NEW_SINCE_ROUND,
    NUMERIC_BACKFILLED,
    PRACTICE,
    ROUND_POSTS,
    STACKED_MC,
    STACKED_NUMERIC,
    BuiltBundle,
    FetcherDoubles,
    Workspace,
    forecast_key,
    install_doubles,
    metaculus_fresh_posts,
)


def test_cli_skips_index_rebuild_when_scratch_has_no_round_dataset(tmp_path, monkeypatch, caplog):
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    main(["--scratch-dir", str(scratch_dir), "--skip-fresh-pull", "--output-dir", str(tmp_path / "bundle")])

    assert "no residual round dataset under scratch/; skipping trace-review index rebuild" in caplog.text


TABLE_SCHEMAS = {
    FORECASTS_FILE: FORECAST_COLUMNS,
    MEMBERS_FILE: MEMBER_COLUMNS,
    TEXTS_FILE: TEXT_COLUMNS,
    TELEMETRY_FILE: TELEMETRY_COLUMNS,
}


def metaculus(ids: tuple[int, int]) -> str:
    return forecast_key(PLATFORM_METACULUS, ids)


MANTIC_KEY = forecast_key(PLATFORM_MANTIC, MANTIC_DISCRETE)


@pytest.fixture(scope="module")
def tables(built_bundle: BuiltBundle) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(built_bundle.output_dir / name) for name in BUNDLE_TABLE_FILES}


@pytest.fixture(scope="module")
def forecasts(tables) -> pd.DataFrame:
    return tables[FORECASTS_FILE].set_index("forecast_key", drop=False)


@pytest.fixture(scope="module")
def members(tables) -> pd.DataFrame:
    return tables[MEMBERS_FILE]


@pytest.fixture(scope="module")
def telemetry(tables) -> pd.DataFrame:
    return tables[TELEMETRY_FILE]


def _bundle_bytes(output_dir: Path) -> dict[str, bytes]:
    return {name: (output_dir / name).read_bytes() for name in (*BUNDLE_TABLE_FILES, MANIFEST_FILE)}


class TestBundleShape:
    def test_four_tables_and_a_manifest_that_agree(self, built_bundle: BuiltBundle, tables):
        manifest = json.loads((built_bundle.output_dir / MANIFEST_FILE).read_text())
        for name, columns in TABLE_SCHEMAS.items():
            table = tables[name]
            assert list(table.columns) == list(columns)
            # A cast after the round trip changes nothing: every dtype survived parquet.
            pd.testing.assert_frame_equal(cast_frame(table, columns), table)
        forecasts = tables[FORECASTS_FILE]
        assert forecasts["forecast_key"].is_unique
        assert manifest["row_counts"] == {
            "forecasts": len(forecasts),
            "members": len(tables[MEMBERS_FILE]),
            "texts": len(tables[TEXTS_FILE]),
            "telemetry": len(tables[TELEMETRY_FILE]),
        }
        assert set(tables[MEMBERS_FILE]["forecast_key"]) <= set(forecasts["forecast_key"])
        assert sorted(tables[TEXTS_FILE]["forecast_key"]) == sorted(forecasts["forecast_key"])
        assert manifest["round_records"] == len(ROUND_POSTS)
        assert sum(manifest["rows_by_platform_origin"].values()) == len(forecasts)

    def test_every_member_of_a_numeric_record_comes_from_its_percentile_field(self, built_bundle, members):
        round_records = collector.load_dataset(
            str(next(built_bundle.workspace.scratch_dir.glob("*/perf_all_tagged.json")))
        )
        record = next(r for r in round_records if r["question_id"] == NUMERIC_BACKFILLED[0])
        rows = members[members["forecast_key"] == metaculus(NUMERIC_BACKFILLED)].set_index("model")

        assert set(rows.index) == set(record["per_model_numeric_percentiles"])
        for model, pairs in record["per_model_numeric_percentiles"].items():
            assert json.loads(rows.loc[model, "percentiles_json"]) == [list(pair) for pair in pairs]
            assert rows.loc[model, "own_score"] == pytest.approx(replay_solo_score(record, pairs))


class TestJoins:
    def test_each_half_of_the_id_collision_gets_only_its_own_lines(self, telemetry):
        lines = telemetry.groupby("forecast_key")["marker"].apply(set)

        # gap_fill_v2 is post-keyed: qid 41871 there is the post holding question 41609.
        assert "gap_fill_v2" in lines[metaculus(COLLIDING_NUMERIC)]
        assert "gap_fill_v2" not in lines[metaculus(COLLIDING_BINARY)]
        # extraction_rung is question-keyed: qid 41871 there is the binary question on post 42108.
        assert "extraction_rung" in lines[metaculus(COLLIDING_BINARY)]
        assert "extraction_rung" not in lines[metaculus(COLLIDING_NUMERIC)]

    def test_mantic_lines_and_research_stay_on_the_mantic_forecast(self, forecasts, telemetry, tables):
        mantic_runs = set(telemetry.loc[telemetry["forecast_key"] == MANTIC_KEY, "run_id"])
        metaculus_twin_runs = set(telemetry.loc[telemetry["forecast_key"] == metaculus(EARLY_BINARY), "run_id"])

        assert mantic_runs
        assert metaculus_twin_runs
        assert not mantic_runs & metaculus_twin_runs
        assert forecasts.loc[MANTIC_KEY, "research_source_class"] == "artifact"
        # The Mantic record sits in latest/ under the Metaculus question's id; the Metaculus record must not take it.
        assert pd.isna(forecasts.loc[metaculus(EARLY_BINARY), "research_source_class"])
        texts = tables[TEXTS_FILE].set_index("forecast_key")
        assert "CENTCOM" in texts.loc[MANTIC_KEY, "research_text"]

    def test_research_classes_stay_apart(self, forecasts, tables):
        backfilled = forecasts.loc[metaculus(NUMERIC_BACKFILLED)]
        assert backfilled["research_source_class"] == "comment_backfill"
        assert backfilled["research_is_trimmed"]
        assert pd.isna(backfilled["market_snapshot_present"])
        assert backfilled["providers_failed"] is None

        artifact = forecasts.loc[metaculus(COLLIDING_NUMERIC)]
        assert list(artifact["providers_failed"]) == ["gemini_search"]
        assert artifact["market_snapshot_present"]
        texts = tables[TEXTS_FILE].set_index("forecast_key")
        assert texts.loc[metaculus(COLLIDING_NUMERIC), "raw_research_path"].endswith(".jsonl")
        # The log-backfill record at latest/<post id> is never joined.
        assert "run log" not in str(texts.loc[metaculus(EARLY_BINARY), "research_text"])


class TestMembers:
    def test_a_stacked_record_has_stacker_rows_and_nothing_rankable(self, members):
        for ids in (STACKED_MC, STACKED_NUMERIC):
            rows = members[members["forecast_key"] == metaculus(ids)]
            assert len(rows) > 0
            assert set(rows["role"]) == {"stacker"}
            assert not rows["rankable"].any()

    def test_an_ordinary_record_ranks_named_members_and_not_anonymous_ones(self, members):
        rows = members[members["forecast_key"] == metaculus(NAMED_MC)]
        assert rows["rankable"].all()
        assert not rows["model_is_anonymous"].any()
        assert rows["forecaster_index"].notna().all()

    def test_a_sparse_curve_beside_denser_ones_is_not_rankable(self, built_bundle):
        round_records = collector.load_dataset(
            str(next(built_bundle.workspace.scratch_dir.glob("*/perf_all_tagged.json")))
        )
        record = next(r for r in round_records if r["question_id"] == NUMERIC_BACKFILLED[0])
        sparse_model = next(iter(record["per_model_numeric_percentiles"]))
        pairs = record["per_model_numeric_percentiles"][sparse_model]
        record["per_model_numeric_percentiles"][sparse_model] = [pairs[0], pairs[len(pairs) // 2], pairs[-1]]

        rows = {
            row["model"]: row
            for row in member_rows(record, key="k", platform=PLATFORM_METACULUS, lines=[], type_log_score=None)
        }

        assert rows[sparse_model]["not_rankable_reason"] == "sparse_anchors"
        assert all(row["rankable"] for model, row in rows.items() if model != sparse_model)

    def test_member_forecast_fills_raw_values_only_where_the_marker_existed(self, members):
        early = members[members["forecast_key"] == metaculus(EARLY_BINARY)]
        assert early["raw_value_json"].isna().all()
        assert early["prob_yes"].notna().all()

        late = members[members["forecast_key"] == metaculus(COLLIDING_BINARY)].set_index("model")
        assert late.loc["gpt-5.2", "raw_value_json"] == "0.37"
        assert not late.loc["gpt-5.2", "clamp_moved"]
        assert late.loc["gemini-3-pro-preview", "clamp_moved"]
        # Members the marker did not log still come from the comment, with the marker columns null.
        assert late.loc["claude-opus-4.5", "prob_yes"] == pytest.approx(0.38)
        assert pd.isna(late.loc["claude-opus-4.5", "clamp_moved"])

    def test_a_mantic_per_bin_member_comes_from_the_marker(self, members):
        row = members[members["forecast_key"] == MANTIC_KEY].iloc[0]
        assert row["value_source"] == "member_forecast"
        assert row["value_kind"] == "pmf"
        assert json.loads(row["percentiles_json"]) == MANTIC_PER_BIN_MASSES
        assert row["own_score"] is not None
        assert not pd.isna(row["own_score"])


class TestNullMeansNotRecorded:
    def test_event_markers_are_null_before_they_existed_and_false_after(self, forecasts):
        assert forecasts.loc[metaculus(DISCRETE), "any_extreme_call"]
        # Run dated after the first extreme_call line, with none of its own: a real "no".
        assert not forecasts.loc[metaculus(COLLIDING_BINARY), "any_extreme_call"]
        # Run dated before any extreme_call line existed, and a forecast with no telemetry at all.
        assert pd.isna(forecasts.loc[metaculus(EARLY_BINARY), "any_extreme_call"])
        assert pd.isna(forecasts.loc[metaculus(NAMED_MC), "any_extreme_call"])

    def test_the_ghost_is_scored_where_it_was_logged(self, forecasts):
        ghosted = forecasts.loc[metaculus(COLLIDING_BINARY)]
        assert ghosted["ghost_present"]
        assert ghosted["ghost_score_delta"] == pytest.approx(ghosted["ghost_log_score"] - ghosted["type_log_score"])
        assert not forecasts.loc[metaculus(COLLIDING_NUMERIC), "ghost_present"]

    def test_a_zero_crowd_is_not_recorded(self, forecasts):
        assert pd.isna(forecasts.loc[metaculus(EARLY_BINARY), "crowd_size"])
        assert forecasts.loc[metaculus(COLLIDING_BINARY), "crowd_size"] == 40


class TestFreshPull:
    def test_dropped_questions_are_counted_and_absent(self, built_bundle, forecasts):
        excluded = built_bundle.result.manifest["excluded"]
        assert excluded == {"date_question": 1, "multi_resolution": 1, "annulled_or_ambiguous": 1, "practice_post": 1}
        absent = {metaculus(ids) for ids in (DATE_QUESTION, ANNULLED, PRACTICE)}
        absent.add(forecast_key(PLATFORM_MANTIC, MANTIC_MULTI_RESOLUTION))
        assert not absent & set(forecasts.index)

    def test_the_overlay_adds_new_questions_once_and_flags_rescored_round_rows(self, forecasts):
        new = forecasts.loc[metaculus(NEW_SINCE_ROUND)]
        assert new["record_origin"] == RECORD_ORIGIN_FRESH
        assert pd.notna(new["config_era"])
        assert pd.isna(new["rescored_since_round"])
        assert (forecasts["record_origin"] == RECORD_ORIGIN_ROUND).sum() == len(ROUND_POSTS)

        rescored = forecasts.loc[metaculus(COLLIDING_BINARY)]
        assert rescored["rescored_since_round"]
        assert rescored["record_origin"] == RECORD_ORIGIN_ROUND
        assert not rescored["resolved_yes"]  # the round's value, not the platform's new one
        assert not forecasts.loc[metaculus(EARLY_BINARY), "rescored_since_round"]

    def test_mantic_ranks_on_spot_baseline_and_keeps_its_out_of_range_resolution(self, forecasts):
        mantic = forecasts.loc[MANTIC_KEY]
        assert mantic["headline_score_field"] == "spot_baseline_score"
        assert mantic["resolution_out_of_range"] == "above_upper_bound"
        assert mantic["prob_on_outcome"] == pytest.approx(1.0 - mantic["published_values"][-1])
        assert forecasts.loc[metaculus(COLLIDING_BINARY), "headline_score_field"] == "spot_peer_score"

    def test_only_posts_the_round_does_not_cover_need_their_comment(self, built_bundle):
        round_records = collector.load_dataset(
            str(next(built_bundle.workspace.scratch_dir.glob("*/perf_all_tagged.json")))
        )
        round_keys = {(r["question_id"], r["post_id"]) for r in round_records}

        needed = posts_needing_comments(metaculus_fresh_posts(), round_keys)

        assert needed == {NEW_SINCE_ROUND[1], DATE_QUESTION[1], ANNULLED[1], PRACTICE[1]}
        assert built_bundle.preflights  # each platform's identity was checked before its pull


class TestFailuresAndCache:
    def test_a_failed_pull_raises_and_leaves_the_previous_bundle_untouched(
        self, workspace: Workspace, tmp_path, monkeypatch
    ):
        install_doubles(monkeypatch, FetcherDoubles())
        output_dir = tmp_path / "bundle"
        build_index(workspace.config(output_dir, platforms=(PLATFORM_METACULUS,)))
        before = _bundle_bytes(output_dir)

        install_doubles(monkeypatch, FetcherDoubles(fail=True))
        with pytest.raises(ConnectionError):
            build_index(workspace.config(output_dir, platforms=(PLATFORM_METACULUS,), refresh=True))

        assert _bundle_bytes(output_dir) == before

    def test_mantic_without_its_token_fails_naming_the_way_out(self, workspace: Workspace, tmp_path, monkeypatch):
        doubles = FetcherDoubles()
        install_doubles(monkeypatch, doubles)
        monkeypatch.delenv(MANTIC_TOKEN_ENV)

        with pytest.raises(MissingPlatformTokenError, match=r"MANTIC_TOKEN.*--platforms metaculus"):
            build_index(workspace.config(tmp_path / "bundle"))
        # Metaculus alone builds without it.
        build_index(workspace.config(tmp_path / "bundle", platforms=(PLATFORM_METACULUS,)))
        assert doubles.total("posts") > 0

    def test_the_default_platforms_are_the_ones_with_a_token_so_an_unattended_sync_builds(self, monkeypatch, caplog):
        monkeypatch.delenv(MANTIC_TOKEN_ENV, raising=False)
        assert parse_args([]).platforms == (PLATFORM_METACULUS,)
        assert MANTIC_TOKEN_ENV in caplog.text  # the skip is logged, not silent

        monkeypatch.setenv(MANTIC_TOKEN_ENV, "token")
        assert parse_args([]).platforms == (PLATFORM_METACULUS, PLATFORM_MANTIC)

    def test_a_platform_named_explicitly_still_needs_its_token(self, monkeypatch):
        monkeypatch.delenv(MANTIC_TOKEN_ENV, raising=False)
        assert parse_args(["--platforms", PLATFORM_MANTIC]).platforms == (PLATFORM_MANTIC,)

    def test_a_duplicate_forecast_key_fails(self, workspace: Workspace, tmp_path, monkeypatch):
        install_doubles(monkeypatch, FetcherDoubles())
        round_path = next(workspace.scratch_dir.glob("*/perf_all_tagged.json"))
        records = json.loads(round_path.read_text())
        duplicated = tmp_path / "residual_2026-11-05" / "perf_all_tagged.json"
        duplicated.parent.mkdir()
        duplicated.write_text(json.dumps([*records, records[0]]))

        with pytest.raises(ValueError, match="duplicate forecast_key"):
            build_index(workspace.config(tmp_path / "bundle", round_path=duplicated, skip_fresh_pull=True))

    def test_a_round_with_no_recovered_rationales_builds_and_reports(
        self, workspace: Workspace, tmp_path, monkeypatch, caplog
    ):
        install_doubles(monkeypatch, FetcherDoubles())
        records = json.loads(next(workspace.scratch_dir.glob("*/perf_all_tagged.json")).read_text())
        for record in records:
            record["comment_text"] = None
        round_path = tmp_path / "residual_2026-11-05" / "perf_all_tagged.json"
        round_path.parent.mkdir()
        round_path.write_text(json.dumps(records))

        with caplog.at_level("INFO"):
            result = build_index(workspace.config(tmp_path / "bundle", round_path=round_path, skip_fresh_pull=True))

        assert all(row["share"] is None for row in result.manifest["trait_prevalence"])
        assert "TRACE_INDEX trait percent_count: not recorded of 0 rationales" in caplog.text
        assert (tmp_path / "bundle" / MANIFEST_FILE).exists()

    def test_cached_pulls_are_reused_until_refresh_and_skip_pulls_nothing(
        self, workspace: Workspace, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "bundle"
        first = FetcherDoubles()
        install_doubles(monkeypatch, first)
        build_index(workspace.config(output_dir))
        assert first.total("posts") > 0
        assert first.total("comments") == 2

        cached = FetcherDoubles()
        install_doubles(monkeypatch, cached)
        build_index(workspace.config(output_dir))
        assert cached.total("posts") == cached.total("comments") == 0

        refreshed = FetcherDoubles()
        install_doubles(monkeypatch, refreshed)
        build_index(workspace.config(output_dir, refresh=True))
        assert refreshed.calls == first.calls

        skipped = FetcherDoubles()
        install_doubles(monkeypatch, skipped)
        result = build_index(workspace.config(output_dir, skip_fresh_pull=True))
        assert not skipped.calls
        assert result.manifest["fresh_pull_skipped"]
        assert result.manifest["rows_by_platform_origin"] == {
            f"{PLATFORM_METACULUS}:{RECORD_ORIGIN_ROUND}": len(ROUND_POSTS)
        }
