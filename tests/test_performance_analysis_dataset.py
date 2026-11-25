"""Tests for the dataset the analysis reads: score healing on load, and the build's wiring.

Covers ``rescore_records`` / ``load_dataset`` self-healing a cached file's stale scores, the
research-tag and prior-diff pass-throughs inside ``build_performance_dataset``, the package's
re-export contract, and the ``zero_point`` coercion in ``resolve_numeric_record_to_score_inputs``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from metaculus_bot import performance_analysis
from metaculus_bot.performance_analysis import collector
from metaculus_bot.performance_analysis.collector import (
    build_performance_dataset,
    load_dataset,
    rescore_records,
    resolve_numeric_record_to_score_inputs,
)
from tests.performance_analysis_fakes import _binary_post


class TestRescoreRecords:
    """Stale stored scores in cached datasets must self-heal on load.

    Scores are pure functions of fields the record carries, but a cached JSON's
    score VALUES are whatever the scorer computed when the file was written — a
    scorer fix never reaches previously-saved files. The checked-in q38991
    fixture is the real record that carried a linear-bucket numeric_log_score of
    -193.29 for a month after the zero_point coercion fix, against a platform
    spot_baseline_score of 165.54.
    """

    _FIXTURE = Path(__file__).parent / "data" / "q38991_stale_zero_point_score.json"

    def _stale_record(self) -> dict:
        return json.loads(self._FIXTURE.read_text())

    def test_zero_point_record_rescores_to_platform_value(self):
        record = self._stale_record()
        assert record["scaling"]["zero_point"] == 0
        assert record["numeric_log_score"] == pytest.approx(-193.292, abs=1e-3)

        changed = rescore_records([record])

        assert changed == 1
        assert record["numeric_log_score"] == pytest.approx(record["metaculus_scores"]["spot_baseline_score"], abs=1e-9)

    def test_fresh_scores_left_untouched(self):
        record = self._stale_record()
        rescore_records([record])
        healed = record["numeric_log_score"]
        assert rescore_records([record]) == 0
        assert record["numeric_log_score"] == healed

    def test_unrecomputable_score_is_never_deleted(self):
        """Missing scaling bounds make recomputation yield None, so the stored value is kept."""
        record = self._stale_record()
        record["scaling"] = {}
        stored = record["numeric_log_score"]
        assert rescore_records([record]) == 0
        assert record["numeric_log_score"] == stored

    def test_load_dataset_heals_stale_scores(self, tmp_path: Path):
        path = tmp_path / "cached.json"
        path.write_text(json.dumps([self._stale_record()]))
        (loaded,) = load_dataset(str(path))
        assert loaded["numeric_log_score"] == pytest.approx(loaded["metaculus_scores"]["spot_baseline_score"], abs=1e-9)

    def test_malformed_record_is_skipped(self):
        assert rescore_records([{"no_type": True}, "not-a-dict"]) == 0  # type: ignore[list-item]

    def test_partial_records_are_skipped_not_crashed(self):
        """Rescoring takes arbitrary cached JSON, so every record missing a field that
        ``_compute_scores`` subscripts must be skipped rather than raise KeyError."""
        partial = [
            {"type": "binary", "resolution_parsed": True},  # no our_forecast_values
            {"type": "binary", "resolution_parsed": True, "our_forecast_values": [0.3, 0.7]},  # no our_prob_yes
            {  # numeric without open-bound flags
                "type": "numeric",
                "resolution_parsed": 5.0,
                "our_forecast_values": [0.0, 0.5, 1.0],
                "scaling": {"range_min": 0.0, "range_max": 10.0},
            },
        ]
        assert rescore_records(partial) == 0
        assert "brier_score" not in partial[0]

    def test_load_dataset_survives_partial_records(self, tmp_path: Path):
        path = tmp_path / "cached.json"
        path.write_text(json.dumps([{"type": "binary", "resolution_parsed": True}, self._stale_record()]))
        loaded = load_dataset(str(path))
        assert len(loaded) == 2
        assert loaded[1]["numeric_log_score"] == pytest.approx(
            loaded[1]["metaculus_scores"]["spot_baseline_score"], abs=1e-9
        )

    def test_load_dataset_is_idempotent_on_an_already_healed_file(self, tmp_path: Path):
        """Re-loading a file whose scores already agree with the scorer must leave every value
        byte-identical, because healing is a repair rather than a rewrite of live data."""
        healed = self._stale_record()
        rescore_records([healed])
        path = tmp_path / "healed.json"
        path.write_text(json.dumps([healed]))
        (reloaded,) = load_dataset(str(path))
        assert reloaded["numeric_log_score"] == healed["numeric_log_score"]

    def test_scoring_failure_on_a_record_without_post_id_only_warns(self, caplog):
        """Rescoring walks arbitrary cached JSON, so a record can lack post_id entirely, and the
        scoring-failure log lines must read it defensively: a subscript there turns one
        unscoreable record into a KeyError that kills the whole load."""
        unscoreable_numeric = {
            "type": "numeric",
            "resolution_parsed": 5.0,
            "our_forecast_values": [0.5],  # < 2 CDF points -> numeric_log_score raises
            "open_lower_bound": False,
            "open_upper_bound": False,
            "scaling": {"range_min": 0.0, "range_max": 10.0, "zero_point": None},
            "numeric_log_score": -1.0,
        }
        unscoreable_mc = {
            "type": "multiple_choice",
            "resolution_parsed": "B",
            "our_forecast_values": [1.0],  # fewer probabilities than options
            "options": ["A", "B"],
            "mc_log_score": -1.0,
        }
        with caplog.at_level(logging.WARNING):
            assert rescore_records([unscoreable_numeric, unscoreable_mc]) == 0

        messages = [r.getMessage() for r in caplog.records]
        assert any("Failed numeric scoring for post None" in m for m in messages)
        assert any("Failed MC scoring for post None" in m for m in messages)
        # The stored values survive: healing never deletes a score it can't recompute.
        assert unscoreable_numeric["numeric_log_score"] == -1.0
        assert unscoreable_mc["mc_log_score"] == -1.0


class TestBuildPerformanceDatasetResearchTags:
    """The dataset the analysis reads must arrive with the treatment tags stamped.

    ``attach_research_tags`` is a single call inside ``build_performance_dataset``;
    unit-testing the tagger alone leaves that pass-through deletable with a green
    suite, and every treated/untreated calibration cut reads these fields off the
    built dataset.
    """

    def test_records_carry_tags_from_the_archive_dir(self, tmp_path: Path, monkeypatch):
        (tmp_path / "11.json").write_text(
            json.dumps(
                {
                    "research_text": "## Time Series Anchor\nband\n## Agentic Research Findings\nfindings\n",
                    "source": "artifact",
                    "gap_fill_v2": {"steps": 4},
                }
            )
        )
        monkeypatch.setattr(
            collector,
            "fetch_resolved_questions",
            lambda tournament, token, **_fetch_options: [_binary_post(1, 11, score_data={})],
        )
        monkeypatch.setattr(
            collector,
            "fetch_bot_comments",
            lambda author_id, token, **_fetch_options: [{"id": 9, "text": "*Forecaster 1*: 70%\n", "on_post": 1}],
        )

        records = build_performance_dataset(tournament="t", token="fake", research_archive_dir=tmp_path)

        assert len(records) == 1
        assert records[0]["anchor_present"] is True
        assert records[0]["gfv2_present"] is True
        assert records[0]["gfv2_loop_ran"] is True
        assert records[0]["research_source_class"] == "artifact"

    def test_question_without_an_archive_record_gets_none_not_false(self, tmp_path: Path, monkeypatch):
        """Absence of evidence is not an untreated record: a missing archive file, or a whole
        missing archive, must never look like a measured False in the cuts."""
        monkeypatch.setattr(
            collector,
            "fetch_resolved_questions",
            lambda tournament, token, **_fetch_options: [_binary_post(2, 22, score_data={})],
        )
        monkeypatch.setattr(collector, "fetch_bot_comments", lambda author_id, token, **_fetch_options: [])

        records = build_performance_dataset(tournament="t", token="fake", research_archive_dir=tmp_path)

        assert records[0]["anchor_present"] is None
        assert records[0]["gfv2_present"] is None
        assert records[0]["anchor_confidence"] is None


class TestBuildPerformanceDatasetPriorDiff:
    """``prior_records=`` must actually reach the re-resolution diff.

    Same rule as the sibling class above: the call inside ``build_performance_dataset`` is a
    single line, so unit-testing ``diff_platform_rescores`` alone leaves the pass-through
    deletable with a green suite. It is the live-pull half of the q44798 detector, where
    Metaculus edits a resolution in place and no timestamp moves.
    """

    def _fetchers(self, monkeypatch, post: dict) -> None:
        monkeypatch.setattr(collector, "fetch_resolved_questions", lambda tournament, token, **_fetch_options: [post])
        monkeypatch.setattr(collector, "fetch_bot_comments", lambda author_id, token, **_fetch_options: [])

    def test_a_moved_resolution_is_tagged_on_the_built_records(self, tmp_path: Path, monkeypatch):
        self._fetchers(monkeypatch, _binary_post(3, 33, score_data={}))
        prior = [{"post_id": 3, "question_id": 33, "resolution_raw": "no", "resolution_parsed": False}]

        records = build_performance_dataset(
            tournament="t", token="fake", research_archive_dir=tmp_path, prior_records=prior
        )

        assert records[0]["platform_rescored"] is True
        assert "resolution_raw" in records[0]["platform_rescored_fields"]
        assert records[0]["prior_resolution"] == "no"

    def test_no_prior_leaves_the_tags_none_not_false(self, tmp_path: Path, monkeypatch):
        """ "Not compared" and "compared, nothing moved" are different facts, and a default build
        must produce the first, or every downstream cut reads silence as stability.

        The keys are absent rather than explicitly None on this path, which is the same "not
        compared" answer to every reader, since the diff and the renderer both use ``.get``.
        """
        self._fetchers(monkeypatch, _binary_post(4, 44, score_data={}))

        records = build_performance_dataset(tournament="t", token="fake", research_archive_dir=tmp_path)

        assert records[0].get("platform_rescored") is None
        assert records[0].get("platform_rescored_fields") is None


class TestPackageExports:
    """The residual rounds' out-of-band scripts import these off the package root, so
    the re-export list is a contract, not bookkeeping."""

    def test_new_analysis_helpers_are_re_exported(self):
        for name in (
            "attach_research_tags",
            "research_tags_for_qid",
            "research_tags_for_record",
            "max_step_clamp_screen",
            "rescore_records",
            "parse_stacker_skip_reason_marker",
        ):
            assert name in performance_analysis.__all__, name
            assert getattr(performance_analysis, name) is not None


class TestResolveNumericScoreInputsZeroPoint:
    """Regression for the zero_point sentinel bug in the record-scoring coercion:
    ``resolve_numeric_record_to_score_inputs`` must keep a serialized
    ``zero_point == 0`` (with a positive ``range_min``) as a genuine log-scale
    value, not collapse it to the linear ``None`` sentinel."""

    def _record(self, zero_point: float | int | None, range_min: float, range_max: float) -> dict:
        return {
            "type": "numeric",
            "resolution_parsed": (range_min + range_max) / 2.0,
            "scaling": {"range_min": range_min, "range_max": range_max, "zero_point": zero_point},
        }

    def test_zero_point_zero_stays_log_when_range_min_positive(self):
        """The sibling of the width_monitor fix: a log-scale question with a positive floor
        carries ``zero_point == 0``, which must survive as 0.0 so ``numeric_log_score`` buckets
        on the geometric grid."""
        inputs = resolve_numeric_record_to_score_inputs(self._record(0, 1.0, 1000.0))
        assert inputs is not None
        _res, _lo, _hi, zero_point = inputs
        assert zero_point == 0.0

    def test_zero_point_zero_dropped_when_range_min_nonpositive(self):
        """A non-positive floor rules out a log transform, so the axis is linear (None)."""
        inputs = resolve_numeric_record_to_score_inputs(self._record(0, 0.0, 100.0))
        assert inputs is not None
        assert inputs[3] is None

    def test_absent_zero_point_is_linear(self):
        inputs = resolve_numeric_record_to_score_inputs(self._record(None, 0.0, 100.0))
        assert inputs is not None
        assert inputs[3] is None

    def test_nonzero_zero_point_passthrough(self):
        inputs = resolve_numeric_record_to_score_inputs(self._record(50, 0.0, 100.0))
        assert inputs is not None
        assert inputs[3] == 50.0
