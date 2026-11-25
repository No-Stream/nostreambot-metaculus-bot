"""Tests for the ghost-forecast scorer (scripts/score_ghosts.py).

The scorer joins harvested ghost markers (gap-fill v2's unpublished dry-run forecast)
to resolved questions and computes paired log-score deltas ghost-vs-published — the
named gate for retiring gap-fill v1. Two marker sources: the full-fidelity
``GHOST_FORECAST_JSON`` (preferred; makes numeric ghosts scoreable) and the legacy
lossy ``GHOST_FORECAST`` summary. Today it finds ~0 resolved v2-era questions (v2
shipped 2026-07-17), so the n=0 path is a first-class, tested outcome.
"""

import json
import math
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

import metaculus_bot.numeric.pchip_cdf as pchip_mod
from metaculus_bot.numeric.config import grid_step_constraints
from metaculus_bot.numeric.pchip_cdf import generate_pchip_cdf
from metaculus_bot.performance_analysis.ghosts import parse_ghost_summary
from metaculus_bot.scoring_common import binary_log_score
from scripts.score_ghosts import join_and_score, main, render_report


def _legacy_ghost(qid: int, qtype: str, summary: str, run_date: str = "2026-07-17T00:00:00Z") -> dict:
    return {"marker": "ghost_forecast", "qid": qid, "qtype": qtype, "summary": summary, "run_date": run_date, "seq": 0}


def _json_ghost(qid: int, payload: dict, run_date: str = "2026-07-17T00:00:00Z", run_id: str = "run-1") -> dict:
    return {
        "marker": "ghost_forecast_json",
        "qid": qid,
        "run_id": run_id,
        "run_date": run_date,
        "seq": 0,
        "forecast_json": json.dumps(payload, separators=(",", ":")),
    }


def _pre_ghost(qid: int, payload: dict, run_date: str = "2026-07-17T00:00:00Z", run_id: str = "run-1") -> dict:
    """A harvested GHOST_PRE_JSON record — the turn-one, pre-research dry run.
    Same post-id qid space and the same ``forecast_json`` field as the concluding
    ghost (both serialize through ``_summarize_ghost``)."""
    return {
        "marker": "ghost_pre_json",
        "qid": qid,
        "run_id": run_id,
        "run_date": run_date,
        "seq": 0,
        "forecast_json": json.dumps(payload, separators=(",", ":")),
    }


def _binary_record(qid: int, resolution, our_prob_yes) -> dict:
    """A binary record whose ``question_id`` deliberately differs from ``post_id``, the scorer's join key."""
    return {
        "post_id": qid,
        "question_id": qid + 100_000,
        "type": "binary",
        "resolution_parsed": resolution,
        "our_prob_yes": our_prob_yes,
        "our_forecast_values": [our_prob_yes] if our_prob_yes is not None else None,
    }


def _numeric_record(
    qid: int,
    resolution: float,
    published_cdf: list[float],
    *,
    lower=0.0,
    upper=100.0,
    open_lower=False,
    open_upper=False,
    zero_point=None,
) -> dict:
    """A numeric record; like ``_binary_record``, its ``question_id`` differs from the ``post_id`` join key."""
    return {
        "post_id": qid,
        "question_id": qid + 100_000,
        "type": "numeric",
        "resolution_parsed": resolution,
        "our_forecast_values": published_cdf,
        "scaling": {"range_min": lower, "range_max": upper, "zero_point": zero_point},
        "open_lower_bound": open_lower,
        "open_upper_bound": open_upper,
    }


def _pchip_cdf(
    percent_percentiles: dict[float, float],
    lower=0.0,
    upper=100.0,
    *,
    open_lower=False,
    open_upper=False,
    zero_point=None,
) -> list[float]:
    """Build a 201-point CDF from percent-keyed percentiles (test helper)."""
    cdf, _ = generate_pchip_cdf(
        percent_percentiles,
        open_upper_bound=open_upper,
        open_lower_bound=open_lower,
        upper_bound=upper,
        lower_bound=lower,
        zero_point=zero_point,
    )
    return cdf


class TestParseGhostSummary:
    def test_binary(self):
        assert parse_ghost_summary("binary", "posterior_prob=0.4200") == 0.42

    def test_multiple_choice(self):
        assert parse_ghost_summary("multiple_choice", "Blue=0.300, Red=0.700") == {"Blue": 0.3, "Red": 0.7}

    def test_numeric_median(self):
        assert parse_ghost_summary("numeric", "median=42.5") == {"median": 42.5}

    def test_unknown_or_empty(self):
        assert parse_ghost_summary("unknown", "") is None
        assert parse_ghost_summary("numeric", "") is None


def test_score_ghosts_golden_report():
    """Pin the complete report for a mixed fixture, including post-id joins."""
    json_ghosts = [
        _json_ghost(1001, {"qtype": "binary", "prob": 0.001}),
        _json_ghost(1002, {"qtype": "multiple_choice", "option_probs": {"Blue": 0.2, "Red": 0.8}}),
        _json_ghost(
            1003,
            {"qtype": "numeric", "declared_percentiles": {"0.1": 10, "0.5": 40, "0.9": 80}},
        ),
    ]
    legacy_ghosts = [_legacy_ghost(1004, "binary", "posterior_prob=0.1")]
    records = [
        {**_binary_record(1001, True, 0.5), "bot_comment_created_at": "2026-08-01T00:00:00Z"},
        {
            "post_id": 1002,
            "question_id": 9002,
            "type": "multiple_choice",
            "resolution_parsed": "Red",
            "options": ["Blue", "Red"],
            "our_forecast_values": [0.5, 0.5],
        },
        _numeric_record(1003, 40.0, _pchip_cdf({10: 10, 50: 50, 90: 90})),
        _binary_record(1004, False, 0.5),
    ]
    pre_ghosts = [_pre_ghost(1001, {"qtype": "binary", "prob": 0.001})]

    report = render_report(join_and_score(json_ghosts, legacy_ghosts, records, pre_ghosts=pre_ghosts))
    assert report + "\n" == Path("tests/fixtures/score_ghosts_golden_report.txt").read_text()


class TestJoinAndScore:
    def test_n_zero_when_no_ghosts(self):
        summary = join_and_score([], [], [])
        assert summary["n_ghosts"] == 0
        assert summary["n_scored"] == 0
        assert "n=0" in render_report(summary)

    def test_legacy_binary_ghost_scored_against_resolution(self):
        legacy = [_legacy_ghost(1, "binary", "posterior_prob=0.90")]
        records = [_binary_record(1, True, 0.50)]  # ghost 0.90 beats published 0.50 on a YES
        summary = join_and_score([], legacy, records)
        assert summary["n_ghosts"] == 1
        assert summary["n_scored"] == 1
        assert summary["binary"]["n"] == 1
        assert summary["source_counts"] == {"json": 0, "legacy": 1}
        # Ghost (0.90) is more confident+correct than published (0.50) -> positive delta.
        assert summary["binary"]["mean_delta"] > 0

    def test_unmatched_ghost_not_scored(self):
        legacy = [_legacy_ghost(999, "binary", "posterior_prob=0.5")]
        summary = join_and_score([], legacy, [_binary_record(1, True, 0.5)])
        assert summary["n_ghosts"] == 1
        assert summary["n_joined"] == 0
        assert summary["n_scored"] == 0

    def test_join_keys_on_post_id_not_question_id(self):
        """Keying on the disjoint ``question_id`` instead of ``post_id`` was the old bug: it always missed."""
        ghost = [_legacy_ghost(42, "binary", "posterior_prob=0.90")]
        wrong_key = {
            "post_id": 999,  # ghost qid (42) != post_id -> no join
            "question_id": 42,  # equals ghost qid, but question_id is NOT the join key
            "type": "binary",
            "resolution_parsed": True,
            "our_prob_yes": 0.5,
            "our_forecast_values": [0.5],
        }
        summary = join_and_score([], ghost, [wrong_key])
        assert summary["n_joined"] == 0
        assert summary["n_scored"] == 0

        # Positive case: post_id == ghost qid joins even though question_id differs.
        right_key = {**wrong_key, "post_id": 42, "question_id": 42_000}
        summary = join_and_score([], ghost, [right_key])
        assert summary["n_joined"] == 1
        assert summary["n_scored"] == 1

    def test_unresolved_record_joined_but_not_scored(self):
        legacy = [_legacy_ghost(1, "binary", "posterior_prob=0.5")]
        summary = join_and_score([], legacy, [_binary_record(1, None, 0.5)])
        assert summary["n_joined"] == 1
        assert summary["n_scored"] == 0

    def test_legacy_numeric_ghost_unscoreable_median_only(self):
        """Legacy markers expose only the median, so no numeric log score is computable: an honest gap."""
        legacy = [_legacy_ghost(2, "numeric", "median=42.5")]
        records = [_numeric_record(2, 40.0, _pchip_cdf({5: 10, 50: 40, 95: 90}))]
        summary = join_and_score([], legacy, records)
        assert summary["n_joined"] == 1
        assert summary["n_scored"] == 0
        assert summary["numeric"]["n_unscoreable"] == 1
        assert summary["numeric"]["unscoreable_reasons"] == {"legacy_median_only": 1}

    def test_latest_ghost_per_qid_wins(self):
        legacy = [
            _legacy_ghost(1, "binary", "posterior_prob=0.10", run_date="2026-07-10T00:00:00Z"),
            _legacy_ghost(1, "binary", "posterior_prob=0.90", run_date="2026-07-17T00:00:00Z"),
        ]
        summary = join_and_score([], legacy, [_binary_record(1, True, 0.50)])
        # Only the latest ghost (0.90) is scored -> one scored row, positive delta.
        assert summary["n_scored"] == 1
        assert summary["binary"]["mean_delta"] > 0


class TestJsonSourceGhosts:
    def test_json_binary_scored(self):
        json_ghosts = [_json_ghost(1, {"qtype": "binary", "prob": 0.90})]
        summary = join_and_score(json_ghosts, [], [_binary_record(1, True, 0.50)])
        assert summary["source_counts"] == {"json": 1, "legacy": 0}
        assert summary["binary"]["n"] == 1
        assert summary["binary"]["mean_delta"] > 0

    def test_json_mc_scored(self):
        json_ghosts = [_json_ghost(1, {"qtype": "multiple_choice", "option_probs": {"Blue": 0.2, "Red": 0.8}})]
        record = {
            "post_id": 1,
            "question_id": 100_001,
            "type": "multiple_choice",
            "resolution_parsed": "Red",
            "options": ["Blue", "Red"],
            "our_forecast_values": [0.5, 0.5],
        }
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["multiple_choice"]["n"] == 1
        # Ghost put 0.8 on the correct option vs published 0.5 -> positive delta.
        assert summary["multiple_choice"]["mean_delta"] > 0

    def test_json_wins_over_legacy_for_same_qid(self):
        """A malformed-but-present legacy ghost and a JSON ghost on the same qid: JSON wins."""
        legacy = [_legacy_ghost(1, "binary", "posterior_prob=0.10")]
        json_ghosts = [_json_ghost(1, {"qtype": "binary", "prob": 0.90})]
        summary = join_and_score(json_ghosts, legacy, [_binary_record(1, True, 0.50)])
        assert summary["n_ghosts"] == 1
        assert summary["source_counts"] == {"json": 1, "legacy": 0}
        # The JSON ghost (0.90, correct) scored, not the legacy 0.10.
        assert summary["binary"]["mean_delta"] > 0

    def test_malformed_json_falls_back_to_legacy(self):
        legacy = [_legacy_ghost(1, "binary", "posterior_prob=0.90")]
        bad_json = {"marker": "ghost_forecast_json", "qid": 1, "run_date": "2026-07-17T00:00:00Z", "seq": 0}
        bad_json["forecast_json"] = "{not valid json"
        summary = join_and_score([bad_json], legacy, [_binary_record(1, True, 0.50)])
        assert summary["source_counts"] == {"json": 0, "legacy": 1}
        assert summary["binary"]["n"] == 1


class TestPreIdentitySplit:
    """The ghost delta split by pre/post identity (2026-08-24 residual round).

    7 of the first 12 scored ghosts were byte-identical to the turn-one PRE-research
    dry run, so a pooled delta mixes measurements of the driver's prior with
    measurements of the loop's research — the composition bias the split exists to
    surface. Only the loop_moved bucket says anything about v2's research.
    """

    def test_identical_pre_lands_in_pre_identical_bucket(self):
        payload = {"qtype": "binary", "prob": 0.30}
        json_ghosts = [_json_ghost(1, payload)]
        pre_ghosts = [_pre_ghost(1, payload)]
        summary = join_and_score(json_ghosts, [], [_binary_record(1, True, 0.50)], pre_ghosts=pre_ghosts)

        split = summary["split_by_pre_identity"]
        assert split["pre_identical"]["n"] == 1
        assert split["loop_moved"]["n"] == 0
        assert split["no_pre_marker"]["n"] == 0
        assert summary["binary"]["rows"][0]["pre_identical"] is True

    def test_moved_pre_lands_in_loop_moved_bucket(self):
        json_ghosts = [_json_ghost(1, {"qtype": "binary", "prob": 0.80})]
        pre_ghosts = [_pre_ghost(1, {"qtype": "binary", "prob": 0.30})]
        summary = join_and_score(json_ghosts, [], [_binary_record(1, True, 0.50)], pre_ghosts=pre_ghosts)

        split = summary["split_by_pre_identity"]
        assert split["loop_moved"]["n"] == 1
        assert split["pre_identical"]["n"] == 0
        assert summary["binary"]["rows"][0]["pre_identical"] is False

    def test_absent_pre_marker_is_its_own_bucket(self):
        """A run predating GHOST_PRE_JSON must read as 'nothing to compare', never
        as either identity verdict."""
        json_ghosts = [_json_ghost(1, {"qtype": "binary", "prob": 0.80})]
        summary = join_and_score(json_ghosts, [], [_binary_record(1, True, 0.50)], pre_ghosts=[])

        split = summary["split_by_pre_identity"]
        assert split["no_pre_marker"]["n"] == 1
        assert summary["binary"]["rows"][0]["pre_identical"] is None

    def test_cross_run_pre_is_not_paired(self):
        """A pre-ghost from run A must not pair with run B's concluding ghost: a run
        can emit one marker without the other (a schema-invalid dry run suppresses
        GHOST_PRE_JSON; a deadline hit banks a pre with no conclusion), so a
        qid-only lookup compares across runs and files a false identity verdict.
        Here the payloads are byte-identical, so a cross-run pair would wrongly
        read ``pre_identical``."""
        payload = {"qtype": "binary", "prob": 0.30}
        json_ghosts = [_json_ghost(1, payload, run_id="run-B")]
        pre_ghosts = [_pre_ghost(1, payload, run_id="run-A")]
        summary = join_and_score(json_ghosts, [], [_binary_record(1, True, 0.50)], pre_ghosts=pre_ghosts)

        split = summary["split_by_pre_identity"]
        assert split["no_pre_marker"]["n"] == 1
        assert split["pre_identical"]["n"] == 0
        assert summary["binary"]["rows"][0]["pre_identical"] is None

    def test_legacy_ghost_cannot_claim_an_identity(self):
        """A legacy summary-only ghost has no payload to byte-compare, so even with
        a pre record present it lands in no_pre_marker rather than a false verdict."""
        legacy = [_legacy_ghost(1, "binary", "posterior_prob=0.80")]
        pre_ghosts = [_pre_ghost(1, {"qtype": "binary", "prob": 0.80})]
        summary = join_and_score([], legacy, [_binary_record(1, True, 0.50)], pre_ghosts=pre_ghosts)

        assert summary["split_by_pre_identity"]["no_pre_marker"]["n"] == 1

    def test_split_spans_question_types(self):
        """The split pools across types (the dim doc's table shape): one identical
        binary + one moved MC land in their respective buckets with per-bucket means."""
        mc_payload = {"qtype": "multiple_choice", "option_probs": {"A": 0.7, "B": 0.3}}
        binary_payload = {"qtype": "binary", "prob": 0.30}
        json_ghosts = [_json_ghost(1, binary_payload), _json_ghost(2, mc_payload)]
        pre_ghosts = [
            _pre_ghost(1, binary_payload),  # identical
            _pre_ghost(2, {"qtype": "multiple_choice", "option_probs": {"A": 0.5, "B": 0.5}}),  # moved
        ]
        mc_record = {
            "post_id": 2,
            "question_id": 100_002,
            "type": "multiple_choice",
            "resolution_parsed": "A",
            "options": ["A", "B"],
            "our_forecast_values": [0.6, 0.4],
        }
        summary = join_and_score(json_ghosts, [], [_binary_record(1, True, 0.50), mc_record], pre_ghosts=pre_ghosts)

        split = summary["split_by_pre_identity"]
        assert split["pre_identical"]["n"] == 1
        assert split["loop_moved"]["n"] == 1
        assert split["pre_identical"]["mean_delta"] is not None
        assert split["loop_moved"]["mean_delta"] is not None

    def test_report_renders_the_split_and_names_the_prior_caveat(self):
        payload = {"qtype": "binary", "prob": 0.30}
        summary = join_and_score(
            [_json_ghost(1, payload)], [], [_binary_record(1, True, 0.50)], pre_ghosts=[_pre_ghost(1, payload)]
        )
        report = render_report(summary)
        assert "Ghost vs pre-research dry run" in report
        assert "measures the driver's prior" in report

    def test_report_omits_the_split_when_nothing_scored(self):
        summary = join_and_score([], [], [], pre_ghosts=[])
        assert "Ghost vs pre-research dry run" not in render_report(summary)

    def test_back_compat_call_without_pre_ghosts_still_carries_the_split(self):
        """Older callers (and the n=0 path) omit pre_ghosts: every scored row then
        reads no_pre_marker, and the summary shape is unchanged for consumers."""
        summary = join_and_score(
            [_json_ghost(1, {"qtype": "binary", "prob": 0.80})], [], [_binary_record(1, True, 0.50)]
        )
        assert summary["split_by_pre_identity"]["no_pre_marker"]["n"] == 1


class TestMainReadsThePreMarkerFromTheArchive:
    """``main`` must LOAD ``ghost_pre_json`` out of the archive and hand it to the join.

    Unit-testing ``join_and_score(pre_ghosts=...)`` alone leaves that load deletable
    with a green suite — and dropping it silently collapses every row into
    ``no_pre_marker``, which is exactly the pooled number the retirement gate must not
    read. This drives the real CLI (archive dir + ``--perf-json`` + ``--output``) with
    no network: ``_load_records`` reads the perf JSON off disk.
    """

    def _write_archive(self, archive_dir: Path, records: list[dict]) -> None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        by_marker: dict[str, list[dict]] = {}
        for record in records:
            by_marker.setdefault(record["marker"], []).append(record)
        for marker, marker_records in by_marker.items():
            (archive_dir / f"{marker}.jsonl").write_text(
                "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in marker_records)
            )

    def _run_main(self, tmp_path: Path, monkeypatch, ghost_records: list[dict]) -> dict:
        archive_dir = tmp_path / "telemetry_archive"
        self._write_archive(archive_dir, ghost_records)
        perf_json = tmp_path / "perf.json"
        perf_json.write_text(json.dumps([_binary_record(1, True, 0.50)]))
        out_path = tmp_path / "summary.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "score_ghosts.py",
                "--archive-dir",
                str(archive_dir),
                "--perf-json",
                str(perf_json),
                "--output",
                str(out_path),
            ],
        )
        main()
        return json.loads(out_path.read_text())

    def test_identical_pre_marker_is_read_off_disk(self, tmp_path: Path, monkeypatch):
        payload = {"qtype": "binary", "prob": 0.30}
        summary = self._run_main(tmp_path, monkeypatch, [_json_ghost(1, payload), _pre_ghost(1, payload)])
        assert summary["n_scored"] == 1
        assert summary["split_by_pre_identity"]["pre_identical"]["n"] == 1
        assert summary["split_by_pre_identity"]["no_pre_marker"]["n"] == 0

    def test_archive_without_the_pre_marker_file_still_scores(self, tmp_path: Path, monkeypatch):
        """Pre-marker-era archives have no ghost_pre_json.jsonl at all."""
        summary = self._run_main(tmp_path, monkeypatch, [_json_ghost(1, {"qtype": "binary", "prob": 0.80})])
        assert summary["n_scored"] == 1
        assert summary["split_by_pre_identity"]["no_pre_marker"]["n"] == 1


class TestNumericPairedScoring:
    def test_tight_json_numeric_ghost_beats_wide_published(self):
        """A tight ghost around the resolution out-scores a wide published CDF, so the delta is positive."""
        published_cdf = _pchip_cdf({5: 10, 25: 30, 50: 50, 75: 70, 95: 90})
        record = _numeric_record(1, 50.0, published_cdf)
        json_ghosts = [
            _json_ghost(
                1,
                {
                    "qtype": "numeric",
                    "declared_percentiles": {0.05: 45, 0.25: 48, 0.5: 50, 0.75: 52, 0.95: 55},
                    "median": 50,
                },
            )
        ]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 1
        assert summary["numeric"]["n_joined"] == 1
        assert summary["numeric"]["n_unscoreable"] == 0
        row = summary["numeric"]["rows"][0]
        assert isinstance(row["ghost_log_score"], float)
        assert isinstance(row["published_log_score"], float)
        assert row["delta"] > 0
        assert summary["n_scored"] == 1

    def test_numeric_unscoreable_when_no_published_cdf(self):
        record = {
            "post_id": 1,
            "question_id": 100_001,
            "type": "numeric",
            "resolution_parsed": 50.0,
            "our_forecast_values": None,
            "scaling": {"range_min": 0.0, "range_max": 100.0, "zero_point": None},
            "open_lower_bound": False,
            "open_upper_bound": False,
        }
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.1: 10, 0.5: 50, 0.9: 90}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 0
        assert summary["numeric"]["unscoreable_reasons"] == {"no_published_cdf": 1}

    def test_report_surfaces_numeric_coverage(self):
        published_cdf = _pchip_cdf({5: 10, 50: 50, 95: 90})
        record = _numeric_record(1, 50.0, published_cdf)
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.1: 40, 0.5: 50, 0.9: 60}})]
        summary = join_and_score(json_ghosts, [], [record])
        report = render_report(summary)
        assert "numeric coverage" in report
        assert "numeric: n=1" in report

    def test_open_lower_bound_numeric_scored(self):
        """The open-lower flag must reach both the ghost CDF build and the score, keeping both finite."""
        published_cdf = _pchip_cdf({5: 20, 50: 50, 95: 90}, open_lower=True)
        record = _numeric_record(1, 50.0, published_cdf, open_lower=True)
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.05: 25, 0.5: 55, 0.95: 85}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 1
        assert summary["numeric"]["n_unscoreable"] == 0
        row = summary["numeric"]["rows"][0]
        assert math.isfinite(row["ghost_log_score"])
        assert math.isfinite(row["published_log_score"])
        assert math.isfinite(row["delta"])

    def test_open_upper_bound_numeric_scored(self):
        """Same wiring check for the other flag."""
        published_cdf = _pchip_cdf({5: 20, 50: 50, 95: 90}, open_upper=True)
        record = _numeric_record(1, 50.0, published_cdf, open_upper=True)
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.05: 15, 0.5: 45, 0.95: 95}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 1
        assert summary["numeric"]["n_unscoreable"] == 0
        row = summary["numeric"]["rows"][0]
        assert math.isfinite(row["ghost_log_score"])
        assert math.isfinite(row["published_log_score"])
        assert math.isfinite(row["delta"])

    def test_log_scale_zero_point_numeric_scored(self):
        """A log-scaled question's ``zero_point`` must thread through the ghost CDF build and both scores."""
        published_cdf = _pchip_cdf({5: 30, 50: 100, 95: 300}, lower=1.0, upper=1000.0, zero_point=0.0)
        record = _numeric_record(1, 100.0, published_cdf, lower=1.0, upper=1000.0, zero_point=0.0)
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.05: 40, 0.5: 110, 0.95: 260}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 1
        assert summary["numeric"]["n_unscoreable"] == 0
        row = summary["numeric"]["rows"][0]
        assert math.isfinite(row["ghost_log_score"])
        assert math.isfinite(row["published_log_score"])
        assert math.isfinite(row["delta"])

    def test_numeric_unscoreable_when_scaling_missing_bounds(self):
        """Scaling without ``range_min`` yields no score inputs, so a valid CDF and ghost cannot pair."""
        published_cdf = _pchip_cdf({5: 10, 50: 50, 95: 90})
        record = {
            "post_id": 1,
            "question_id": 100_001,
            "type": "numeric",
            "resolution_parsed": 50.0,
            "our_forecast_values": published_cdf,
            "scaling": {"range_min": None, "range_max": 100.0, "zero_point": None},
            "open_lower_bound": False,
            "open_upper_bound": False,
        }
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.1: 10, 0.5: 50, 0.9: 90}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 0
        assert summary["numeric"]["unscoreable_reasons"] == {"no_score_inputs": 1}

    def test_numeric_unscoreable_when_cdf_build_fails(self):
        """Percentiles at the fraction bounds get filtered out, so the CDF build raises: cdf_build_failed."""
        published_cdf = _pchip_cdf({5: 10, 50: 50, 95: 90})
        record = _numeric_record(1, 50.0, published_cdf)
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.0: 10, 1.0: 90}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 0
        assert summary["numeric"]["unscoreable_reasons"] == {"cdf_build_failed": 1}

    def test_native_discrete_ghost_scored_on_reduced_grid(self):
        """Native-discrete questions publish on a reduced grid, so the ghost is built with the same num_points."""
        min_step = round(0.01 / 20, 9)
        published_cdf, _ = generate_pchip_cdf(
            {5: 5, 50: 10, 95: 15},
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=20.0,
            lower_bound=0.0,
            zero_point=None,
            min_step=min_step,
            num_points=21,
        )
        assert len(published_cdf) == 21  # reduced grid, not the continuous 201
        record = _numeric_record(1, 10.0, published_cdf, lower=0.0, upper=20.0)
        record["type"] = "discrete"
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.05: 6, 0.5: 11, 0.95: 14}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 1
        assert summary["numeric"]["n_unscoreable"] == 0
        row = summary["numeric"]["rows"][0]
        assert math.isfinite(row["ghost_log_score"])
        assert math.isfinite(row["published_log_score"])
        assert math.isfinite(row["delta"])


class TestGhostGridScaledMaxStep:
    """F1 regression: the ghost CDF must be rebuilt with the grid-SCALED max-step.

    On a native-discrete question the published CDF lives on a coarse grid
    (cdf_size < 201) where the server's per-bin max step relaxes above 0.2 (e.g. 1.0
    on a 9-point grid). The scorer rebuilds the ghost on that same grid; if it inherited
    the 201-grid 0.2 cap it would clip a concentrated integer's mass while the published
    side (built by the fixed prod path) stayed uncapped — an asymmetric paired log-score
    biased against the ghost on concentrated discrete questions. The ghost build must pass
    the grid-scaled (min_step, max_step) so a concentrated bin survives.
    """

    def test_concentrated_discrete_ghost_retains_bin_above_020(self, monkeypatch):
        """grok's Q38880 shape (~30% mass on integer 0) must keep P(0) above 0.25 on the 9-point grid."""
        min_step, max_step = grid_step_constraints(9)
        # Any valid 9-point published CDF fixes num_points=9; its shape feeds only the published score.
        published_cdf, _ = generate_pchip_cdf(
            {5: 1, 50: 3, 95: 6},
            open_upper_bound=True,
            open_lower_bound=False,
            upper_bound=7.5,
            lower_bound=-0.5,
            zero_point=None,
            min_step=min_step,
            max_step=max_step,
            num_points=9,
        )
        assert len(published_cdf) == 9

        captured: dict = {}
        real = pchip_mod.generate_pchip_cdf

        def spy(*args, **kwargs):
            cdf, flag = real(*args, **kwargs)
            captured["cdf"] = cdf
            captured["max_step"] = kwargs.get("max_step")
            return cdf, flag

        # _score_numeric imports generate_pchip_cdf lazily, so patching the module attribute catches the ghost build.
        monkeypatch.setattr(pchip_mod, "generate_pchip_cdf", spy)

        record = _numeric_record(1, 0.0, published_cdf, lower=-0.5, upper=7.5, open_upper=True)
        record["type"] = "discrete"
        concentrated = {0.2: 0.30, 0.4: 0.65, 0.5: 0.90, 0.8: 2.20, 0.9: 3.20, 0.99: 6.60}
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": concentrated})]
        summary = join_and_score(json_ghosts, [], [record])

        assert summary["numeric"]["n"] == 1  # scoreable
        # The ghost build received the grid-scaled max-step (1.0), not the 201-grid 0.2 cap.
        assert captured["max_step"] == pytest.approx(max_step)
        assert captured["max_step"] > 0.2
        ghost_cdf = np.asarray(captured["cdf"], dtype=float)
        assert ghost_cdf[0] == pytest.approx(0.0, abs=1e-9)  # closed lower
        p_zero = ghost_cdf[1] - ghost_cdf[0]  # F(0.5) = P(0)
        assert p_zero > 0.25, f"ghost P(0)={p_zero} was clipped by the 0.2 cap"

    def test_paired_score_symmetric_on_concentrated_discrete_grid(self):
        """Identical percentiles on the same coarse grid must give delta 0; the bug clipped only the ghost."""
        concentrated_pct = {20.0: 0.30, 40.0: 0.65, 50.0: 0.90, 80.0: 2.20, 90.0: 3.20, 99.0: 6.60}
        min_step, max_step = grid_step_constraints(9)
        published_cdf, _ = generate_pchip_cdf(
            concentrated_pct,
            open_upper_bound=True,
            open_lower_bound=False,
            upper_bound=7.5,
            lower_bound=-0.5,
            zero_point=None,
            min_step=min_step,
            max_step=max_step,
            num_points=9,
        )
        record = _numeric_record(1, 0.0, published_cdf, lower=-0.5, upper=7.5, open_upper=True)
        record["type"] = "discrete"
        ghost_declared = {p / 100.0: v for p, v in concentrated_pct.items()}
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": ghost_declared})]
        summary = join_and_score(json_ghosts, [], [record])

        assert summary["numeric"]["n"] == 1
        assert summary["numeric"]["n_unscoreable"] == 0
        row = summary["numeric"]["rows"][0]
        assert row["delta"] == pytest.approx(0.0, abs=1e-9)


class TestGhostsWithoutAScorer:
    """Gap-fill v2 emits date ghosts and nothing here can score one: the residual dataset
    excludes date questions by decision, so no date record exists to join, and the tally has no
    date arm should one ever appear. Both facts are counted and named in the report; before this
    a date ghost fell off the end of the type dispatch with no trace, so the operator read the
    missing pairs as questions still waiting on resolutions."""

    _DATE_PAYLOAD: ClassVar[dict] = {
        "qtype": "date",
        "declared_percentiles": {"0.1": 1_789_000_000.0, "0.5": 1_789_040_000.0, "0.9": 1_789_080_000.0},
        "median": 1_789_040_000.0,
    }

    def test_date_ghosts_are_counted_in_the_inventory_and_named_in_the_report(self):
        json_ghosts = [_json_ghost(1, self._DATE_PAYLOAD), _json_ghost(2, {"qtype": "binary", "prob": 0.9})]
        summary = join_and_score(json_ghosts, [], [_binary_record(2, True, 0.5)])

        assert summary["qtype_counts"] == {"binary": 1, "date": 1}
        assert summary["n_joined"] == 1  # the date ghost had no record to join
        assert summary["joined_without_scorer"] == {}
        report = render_report(summary)
        assert "by type: binary=1 date=1" in report
        assert "date ghosts cannot be scored" in report

    def test_a_joined_ghost_of_a_type_without_a_scorer_is_counted_not_dropped(self):
        date_record = {**_numeric_record(1, 1_789_040_000.0, [0.0, 0.5, 1.0]), "type": "date"}
        unparsed_legacy = [_legacy_ghost(3, "unknown", "")]
        records = [date_record, _binary_record(3, True, 0.5)]
        summary = join_and_score([_json_ghost(1, self._DATE_PAYLOAD)], unparsed_legacy, records)

        assert summary["n_joined"] == 2
        assert summary["n_scored"] == 0
        assert summary["joined_without_scorer"] == {"date": 1, "unknown": 1}
        assert "joined but no scorer for the type: date=1 unknown=1" in render_report(summary)

    def test_a_report_without_date_ghosts_says_nothing_about_them(self):
        summary = join_and_score([_json_ghost(1, {"qtype": "binary", "prob": 0.9})], [], [_binary_record(1, True, 0.5)])
        report = render_report(summary)
        assert "by type: binary=1" in report
        assert "date ghosts" not in report
        assert "no scorer" not in report


def _v1_ghost(qid: int, payload: dict, run_date: str = "2026-07-17T00:00:00Z", run_id: str = "run-1") -> dict:
    """A harvested GHOST_FORECAST_V1_JSON record: the ghost re-asked with gap-fill v1's section (2026-09-09)."""
    return {
        "marker": "ghost_forecast_v1_json",
        "qid": qid,
        "run_id": run_id,
        "run_date": run_date,
        "seq": 0,
        "forecast_json": json.dumps(payload, separators=(",", ":")),
    }


class TestSameDriverPairedReads:
    """Two ghost variants of the same driver, same question, same run, both scored against the resolution.

    Pre versus post isolates what v2's own research did to the driver; plain versus with-v1 isolates
    v1's section. Both are the same-model instrument the cost pass computed by hand on 2026-09-09
    (``scratch/cost_pass_2026-09-09/ghost_pre_post.py``), so the ensemble never confounds the read.
    """

    def test_pre_post_read_scores_the_research_effect_on_the_moved_pairs_only(self):
        json_ghosts = [
            _json_ghost(1, {"qtype": "binary", "prob": 0.80}),
            _json_ghost(2, {"qtype": "binary", "prob": 0.60}),
        ]
        pre_ghosts = [
            _pre_ghost(1, {"qtype": "binary", "prob": 0.30}),
            _pre_ghost(2, {"qtype": "binary", "prob": 0.60}),
        ]
        records = [_binary_record(1, True, 0.50), _binary_record(2, True, 0.50)]

        block = join_and_score(json_ghosts, [], records, pre_ghosts=pre_ghosts)["pre_post"]

        assert (block["n_paired"], block["n_identical"], block["moved"]["n"]) == (2, 1, 1)
        expected = binary_log_score(0.80, True) - binary_log_score(0.30, True)
        assert block["moved"]["mean_delta"] == pytest.approx(expected)
        assert block["moved"]["median_delta"] == pytest.approx(expected)
        assert (block["moved"]["n_toward"], block["moved"]["n_away"]) == (1, 0)
        assert block["moved"]["by_type"] == {"binary": block["moved"]["by_type"]["binary"]}
        assert block["moved"]["by_type"]["binary"]["n"] == 1
        assert block["moved"]["left_mean"] == pytest.approx(binary_log_score(0.30, True))
        assert block["moved"]["right_mean"] == pytest.approx(binary_log_score(0.80, True))
        assert block["moved"]["published_mean"] == pytest.approx(binary_log_score(0.50, True))
        assert block["unpaired"] == {}

    def test_partners_from_other_runs_missing_or_unscoreable_are_counted_not_paired(self):
        json_ghosts = [
            _json_ghost(1, {"qtype": "binary", "prob": 0.80}, run_id="run-B"),
            _json_ghost(2, {"qtype": "binary", "prob": 0.70}),
            _json_ghost(3, {"qtype": "binary", "prob": 0.70}),
            _json_ghost(4, {"qtype": "binary", "prob": 0.70}),
        ]
        pre_ghosts = [
            _pre_ghost(1, {"qtype": "binary", "prob": 0.30}, run_id="run-A"),
            _pre_ghost(3, {"qtype": "binary", "prob": 0.30}),
            _pre_ghost(4, {"qtype": "multiple_choice", "option_probs": {"A": 1.0}}),
        ]
        records = [_binary_record(1, True, 0.5), _binary_record(2, True, 0.5), _binary_record(4, True, 0.5)]

        block = join_and_score(json_ghosts, [], records, pre_ghosts=pre_ghosts)["pre_post"]

        assert block["n_paired"] == 0
        assert block["unpaired"] == {
            "partner from a different run": 1,
            "no partner marker": 1,
            "no resolved record": 1,
            "qtype mismatch": 1,
        }
        report = render_report(join_and_score(json_ghosts, [], records, pre_ghosts=pre_ghosts))
        assert "Pre-research dry run vs concluding ghost" in report
        assert "paired: 0 (no GHOST_PRE_JSON / GHOST_FORECAST_JSON pair on a resolved question yet)" in report

    def test_v1_pairs_measure_v1s_marginal_value_on_the_driver(self):
        json_ghosts = [
            _json_ghost(1, {"qtype": "binary", "prob": 0.40}),
            _json_ghost(2, {"qtype": "binary", "prob": 0.90}),
        ]
        v1_ghosts = [_v1_ghost(1, {"qtype": "binary", "prob": 0.70}), _v1_ghost(2, {"qtype": "binary", "prob": 0.60})]
        records = [_binary_record(1, True, 0.50), _binary_record(2, True, 0.50)]

        summary = join_and_score(json_ghosts, [], records, v1_ghosts=v1_ghosts)
        block = summary["v1_pairs"]

        assert (block["n_paired"], block["n_identical"], block["moved"]["n"]) == (2, 0, 2)
        deltas = sorted(row["delta"] for row in block["rows"])
        assert deltas[0] == pytest.approx(binary_log_score(0.60, True) - binary_log_score(0.90, True))
        assert deltas[1] == pytest.approx(binary_log_score(0.70, True) - binary_log_score(0.40, True))
        assert (block["moved"]["n_toward"], block["moved"]["n_away"]) == (1, 1)
        assert block["moved"]["sign_test_p"] == pytest.approx(1.0)
        report = render_report(summary)
        assert "Ghost with gap-fill v1 vs plain ghost" in report
        assert "paired: 2  byte-identical: 0  moved: 2" in report
        assert "toward 1 / away 1, sign test p=1.000" in report
        assert "ladder on the moved pairs: plain " in report

    def test_v1_read_reports_waiting_while_the_marker_has_no_records(self):
        summary = join_and_score([_json_ghost(1, {"qtype": "binary", "prob": 0.9})], [], [_binary_record(1, True, 0.5)])
        assert summary["v1_pairs"]["n_paired"] == 0
        assert "paired: 0 (GHOST_FORECAST_V1 ships 2026-09-09; waiting on resolutions)" in render_report(summary)


class TestMainReadsTheV1MarkerFromTheArchive(TestMainReadsThePreMarkerFromTheArchive):
    """Same load-or-it-is-deletable argument as the pre marker: ``main`` must read ``ghost_forecast_v1_json``."""

    def test_v1_ghost_is_read_off_disk_and_paired(self, tmp_path: Path, monkeypatch):
        records = [_json_ghost(1, {"qtype": "binary", "prob": 0.30}), _v1_ghost(1, {"qtype": "binary", "prob": 0.80})]
        summary = self._run_main(tmp_path, monkeypatch, records)
        assert summary["v1_pairs"]["n_paired"] == 1
        assert summary["v1_pairs"]["moved"]["n_toward"] == 1
