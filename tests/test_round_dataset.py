"""The residual round's dataset builder: tags, dedup, provenance, cohorts and the four outputs.

The receipt that the extraction is faithful is the 2026-09-09 round itself, reproduced
byte-for-byte from this code (``docs/performance_analysis.md``, "The round dataset builder"). These
tests hold the contract downstream lanes depend on: the tag vocabulary, the dedup preference, the
cumulative provenance fields, and the absence of any one round's state from the tracked spine.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from metaculus_bot.performance_analysis import round_dataset, round_outputs
from metaculus_bot.performance_analysis.eras import B4E9DF0_MERGED_AT, WIDENING_FLIP_MERGED_AT
from metaculus_bot.performance_analysis.platform_scores import spot_peer_score
from metaculus_bot.performance_analysis.round_dataset import (
    DRIFT_APPEARED,
    DRIFT_DISAPPEARED,
    DRIFT_MOVED,
    PLATFORM_DRIFT_ACCESSORS,
    SCORE_ATOL,
    TAG_FIELDS,
    RoundSpec,
    _drift_against_prior,
    build_round_dataset,
    dedup,
    heal_stored_scores,
    is_scored,
    prior_snapshot,
    score_transition,
    strip_tags,
    tag_fresh_records,
)
from metaculus_bot.performance_analysis.round_outputs import write_round_outputs

WEIGHTED_SLUG = "summer-futureeval-2026"
REUSED_SLUG = "spring-aib-2026"
PRE_FLIP_SUBMITTED = "2026-04-01T12:00:00Z"
POST_FLIP_SUBMITTED = "2026-06-01T12:00:00Z"
TRIPLE_SUBMITTED = "2026-08-01T12:00:00Z"


CURRENT_LOG_SCORE = 48.542682717024164
CURRENT_BRIER_SCORE = 0.09000000000000002


def _binary_record(
    question_id: int, post_id: int, slug: str, submitted: str, *, log_score: float = CURRENT_LOG_SCORE
) -> dict:
    """A minimally rescorable binary record: 70% on a YES resolution, scored by the live collector."""
    return {
        "question_id": question_id,
        "post_id": post_id,
        "title": f"question {question_id}",
        "type": "binary",
        "resolution_parsed": True,
        "resolution_raw": "yes",
        "our_prob_yes": 0.7,
        "our_forecast_values": [0.7],
        "log_score": log_score,
        "brier_score": CURRENT_BRIER_SCORE,
        "bot_comment_created_at": submitted,
        "source_tournament": slug,
        "metaculus_scores": {"peer_score": 4.0, "spot_peer_score": 5.0, "coverage": 1.0},
    }


def _write_round(
    tmp_path: Path,
    *,
    fresh: list[dict],
    prior: list[dict],
    weights: dict[int, float],
    changes: list[dict] | None = None,
) -> RoundSpec:
    round_dir = tmp_path / "round"
    prior_dir = tmp_path / "prior"
    telemetry_dir = tmp_path / "telemetry"
    for directory in (round_dir, prior_dir, telemetry_dir):
        directory.mkdir()
    (round_dir / f"perf_{WEIGHTED_SLUG}.json").write_text(json.dumps(fresh))
    (prior_dir / "perf_all_tagged.json").write_text(json.dumps(prior))
    (round_dir / "question_weights.json").write_text(
        json.dumps({"weights": {str(qid): weight for qid, weight in weights.items()}})
    )
    (round_dir / "platform_rescored.json").write_text(
        json.dumps({"compared": len(prior), "rescored_records": 0, "changes": changes or []})
    )
    return RoundSpec(
        round_dir=round_dir,
        label="this-round",
        prior_dir=prior_dir,
        prior_label="prior-round",
        telemetry_dir=telemetry_dir,
        weighted_slug=WEIGHTED_SLUG,
        required_slugs=(WEIGHTED_SLUG,),
        optional_slugs=("a-slug-with-no-file",),
        reused_slugs=(REUSED_SLUG,),
    )


class TestIsScored:
    def test_binary_needs_a_log_score(self) -> None:
        assert is_scored({"type": "binary", "log_score": -0.3})
        assert not is_scored({"type": "binary", "log_score": None})

    @pytest.mark.parametrize("qtype", ["numeric", "discrete"])
    def test_continuous_reads_the_numeric_log_score(self, qtype: str) -> None:
        assert is_scored({"type": qtype, "numeric_log_score": 12.0})
        assert not is_scored({"type": qtype, "log_score": -0.3})

    def test_multiple_choice_reads_its_own_field(self) -> None:
        assert is_scored({"type": "multiple_choice", "mc_log_score": -1.1})

    def test_an_unknown_type_is_never_scored(self) -> None:
        assert not is_scored({"type": None, "log_score": -0.3})


class TestStripTags:
    def test_every_owned_tag_is_removed_and_the_payload_survives(self) -> None:
        record = {"question_id": 1, **dict.fromkeys(TAG_FIELDS, "stale")}
        assert strip_tags(record) == {"question_id": 1}


class TestDedup:
    def test_a_fresh_record_wins_over_its_reused_twin_in_either_order(self) -> None:
        fresh = {"question_id": 1, "post_id": 2, "source_is_fresh_pull": True, "source_tournament": WEIGHTED_SLUG}
        reused = {"question_id": 1, "post_id": 2, "source_is_fresh_pull": False, "source_tournament": WEIGHTED_SLUG}
        assert dedup([fresh, reused]) == [fresh]
        assert dedup([reused, fresh]) == [fresh]

    def test_both_ids_form_the_key(self) -> None:
        first = {"question_id": 1, "post_id": 2, "source_is_fresh_pull": True, "source_tournament": WEIGHTED_SLUG}
        second = {"question_id": 1, "post_id": 3, "source_is_fresh_pull": True, "source_tournament": WEIGHTED_SLUG}
        assert dedup([first, second]) == [first, second]


class TestHealStoredScores:
    def test_a_stale_stored_score_is_healed_and_recorded_cumulatively(self) -> None:
        record = _binary_record(1, 2, WEIGHTED_SLUG, TRIPLE_SUBMITTED, log_score=99.0)
        record["rescored_fields_prior_rounds"] = ["numeric_log_score"]
        changes = heal_stored_scores([record], "fresh_this-round")
        assert record["log_score"] == pytest.approx(CURRENT_LOG_SCORE)
        assert record["rescored_fields_this_round"] == ["log_score"]
        assert record["rescored_fields"] == ["log_score", "numeric_log_score"]
        assert changes[0]["question_id"] == 1

    def test_a_current_score_records_no_delta(self) -> None:
        record = _binary_record(1, 2, WEIGHTED_SLUG, TRIPLE_SUBMITTED)
        record["rescored_fields_prior_rounds"] = []
        assert heal_stored_scores([record], "fresh_this-round") == []
        assert record["rescored_fields"] == []


class TestPriorViewIsFlatByConstruction:
    """The asymmetry a symmetric-looking "fix" breaks: 358 spurious `appeared` on one real round."""

    def test_a_real_record_carries_its_platform_scores_only_nested(self) -> None:
        record = _binary_record(1, 2, WEIGHTED_SLUG, TRIPLE_SUBMITTED)
        assert record["metaculus_scores"]["spot_peer_score"] == 5.0
        assert "spot_peer_score" not in record, "no archived record carries a top-level platform score"

    def test_the_prior_view_hoists_them_flat_and_keeps_no_nested_block(self) -> None:
        _, _, index = prior_snapshot([_binary_record(1, 2, WEIGHTED_SLUG, TRIPLE_SUBMITTED)], "prior-round")
        view = index[(1, 2)]
        assert view["peer_score"] == 4.0
        assert view["spot_peer_score"] == 5.0
        assert "metaculus_scores" not in view

    def test_the_accessors_read_a_live_record_and_never_the_prior_view(self) -> None:
        record = _binary_record(1, 2, WEIGHTED_SLUG, TRIPLE_SUBMITTED)
        _, _, index = prior_snapshot([record], "prior-round")
        assert spot_peer_score(record) == 5.0
        assert spot_peer_score(index[(1, 2)]) is None, "the view is flat, so an accessor cannot read it"

    def test_the_view_hoists_exactly_the_fields_the_drift_path_reads(self) -> None:
        _, _, index = prior_snapshot([_binary_record(1, 2, WEIGHTED_SLUG, TRIPLE_SUBMITTED)], "prior-round")
        assert {field for field, _ in PLATFORM_DRIFT_ACCESSORS} <= set(index[(1, 2)])

    def test_an_unmoved_platform_score_is_no_drift_through_the_real_read_paths(self) -> None:
        record = _binary_record(1, 2, WEIGHTED_SLUG, TRIPLE_SUBMITTED)
        _, _, index = prior_snapshot([_binary_record(1, 2, WEIGHTED_SLUG, TRIPLE_SUBMITTED)], "prior-round")
        assert _drift_against_prior(record, index[(1, 2)]) == ([], [])


class TestScoreTransition:
    """A one-sided null is a change. Nine archived rounds saw zero of these, and one q44798 move."""

    def test_two_nulls_are_not_a_change(self) -> None:
        assert score_transition(None, None) is None

    def test_a_score_arriving_where_there_was_none_appeared(self) -> None:
        assert score_transition(None, -5.4) == DRIFT_APPEARED

    def test_a_score_metaculus_withdrew_disappeared(self) -> None:
        assert score_transition(5.4, None) == DRIFT_DISAPPEARED

    def test_a_real_move_beyond_tolerance_moved(self) -> None:
        assert score_transition(5.406978908431959, -5.418975442292634) == DRIFT_MOVED

    def test_serialization_wiggle_inside_the_tolerance_held(self) -> None:
        assert score_transition(5.0, 5.0 + SCORE_ATOL / 2) is None

    def test_a_zero_score_is_a_value_not_an_absence(self) -> None:
        assert score_transition(0.0, None) == DRIFT_DISAPPEARED
        assert score_transition(None, 0.0) == DRIFT_APPEARED


class TestPlatformScoreDisappearing:
    """The gap this closed: a withdrawn score used to leave the drift report reading zero."""

    @pytest.fixture
    def dataset(self, tmp_path: Path) -> round_dataset.RoundDataset:
        fresh = _binary_record(101, 201, WEIGHTED_SLUG, TRIPLE_SUBMITTED)
        fresh["metaculus_scores"] = {"peer_score": 4.0, "coverage": 1.0}
        prior = [_binary_record(101, 201, WEIGHTED_SLUG, TRIPLE_SUBMITTED)]
        spec = _write_round(tmp_path, fresh=[fresh], prior=prior, weights={101: 1.0})
        return build_round_dataset(spec)

    def test_the_withdrawn_field_is_reported_as_drift(self, dataset: round_dataset.RoundDataset) -> None:
        assert [(e["field"], e["transition"], e["prior"], e["now"]) for e in dataset.platform_score_drift] == [
            ("spot_peer_score", DRIFT_DISAPPEARED, 5.0, None)
        ]

    def test_the_record_carries_the_field_in_its_rescore_tag(self, dataset: round_dataset.RoundDataset) -> None:
        assert dataset.records[0]["platform_rescored_this_round"] == ["spot_peer_score"]

    def test_the_warning_names_the_transition(self, tmp_path: Path, caplog) -> None:
        fresh = _binary_record(101, 201, WEIGHTED_SLUG, TRIPLE_SUBMITTED)
        fresh["metaculus_scores"] = {"peer_score": 4.0, "coverage": 1.0}
        spec = _write_round(
            tmp_path,
            fresh=[fresh],
            prior=[_binary_record(101, 201, WEIGHTED_SLUG, TRIPLE_SUBMITTED)],
            weights={101: 1.0},
        )
        with caplog.at_level(logging.WARNING):
            build_round_dataset(spec)
        assert any(DRIFT_DISAPPEARED in record.message for record in caplog.records)


class TestBuildRoundDataset:
    @pytest.fixture
    def built(self, tmp_path: Path) -> tuple[RoundSpec, round_dataset.RoundDataset]:
        fresh = [
            _binary_record(101, 201, WEIGHTED_SLUG, PRE_FLIP_SUBMITTED),
            _binary_record(102, 202, WEIGHTED_SLUG, POST_FLIP_SUBMITTED),
            _binary_record(103, 203, WEIGHTED_SLUG, TRIPLE_SUBMITTED),
        ]
        prior_carried = _binary_record(101, 201, WEIGHTED_SLUG, PRE_FLIP_SUBMITTED)
        prior_carried["platform_rescored"] = ["spot_peer_score"]
        prior = [
            prior_carried,
            _binary_record(900, 901, REUSED_SLUG, PRE_FLIP_SUBMITTED),
        ]
        spec = _write_round(
            tmp_path,
            fresh=fresh,
            prior=prior,
            weights={101: 1.0, 102: 0.5, 103: 1.0},
            changes=[{"question_id": 101, "post_id": 201, "field": "resolution_raw"}],
        )
        return spec, build_round_dataset(spec)

    def test_the_reused_baseline_is_carried_and_the_fresh_pull_deduped(
        self, built: tuple[RoundSpec, round_dataset.RoundDataset]
    ) -> None:
        _, dataset = built
        by_qid = {r["question_id"]: r for r in dataset.records}
        assert sorted(by_qid) == [101, 102, 103, 900]
        assert by_qid[101]["source_is_fresh_pull"] is True
        assert by_qid[101]["record_provenance"] == "fresh_this-round"
        assert by_qid[900]["record_provenance"] == "reused_prior-round_tagged"

    def test_era_tags_use_the_vocabulary_the_standing_instruments_slice_on(
        self, built: tuple[RoundSpec, round_dataset.RoundDataset]
    ) -> None:
        _, dataset = built
        eras = {r["question_id"]: r["config_era"] for r in dataset.records}
        assert eras[101] == "pre_flip"
        assert eras[102] == "post_flip"
        assert eras[103] == "triple_era"
        triple = next(r for r in dataset.records if r["question_id"] == 103)
        assert triple["triple_subera"] == "triple_pre_market"
        assert triple["triple_subera_fine"] == "post_dry_key_fix"
        assert triple["ft_unfreeze_side"] == "ft_0292"
        assert triple["post_linters_merge"] is False

    def test_novelty_weights_and_platform_provenance(self, built: tuple[RoundSpec, round_dataset.RoundDataset]) -> None:
        _, dataset = built
        by_qid = {r["question_id"]: r for r in dataset.records}
        assert by_qid[101]["is_new_since_prior"] is False
        assert by_qid[103]["is_new_since_prior"] is True
        assert by_qid[102]["question_weight"] == 0.5
        assert by_qid[900]["question_weight"] is None
        # Compared and moved, compared and unmoved, never compared: the three states must stay distinct.
        assert by_qid[101]["platform_rescored_pull_tag"] is True
        assert by_qid[101]["platform_rescored_this_round"] == ["resolution_raw"]
        assert by_qid[101]["platform_rescored"] == ["resolution_raw", "spot_peer_score"]
        assert by_qid[103]["platform_rescored_pull_tag"] is None
        assert by_qid[900]["platform_rescored_pull_tag"] is None

    def test_the_builder_writes_no_field_outside_its_declared_tag_set(
        self, built: tuple[RoundSpec, round_dataset.RoundDataset]
    ) -> None:
        _, dataset = built
        original = set(_binary_record(1, 2, WEIGHTED_SLUG, TRIPLE_SUBMITTED)) | set(round_dataset.SCORE_FIELDS)
        added = set(dataset.records[0]) - original
        assert added <= set(TAG_FIELDS), f"untagged field(s) would survive a rule change: {added - set(TAG_FIELDS)}"

    def test_an_empty_telemetry_archive_leaves_the_canonical_cohort_intact(
        self, built: tuple[RoundSpec, round_dataset.RoundDataset]
    ) -> None:
        _, dataset = built
        assert dataset.degraded.runs == []
        assert dataset.degraded.union_question_ids == set(round_dataset.DEGRADED_FULL_QUESTION_IDS)


class TestTagFreshRecords:
    """Records pulled between rounds must carry the tags the next round would give them."""

    TAGGED_FIELDS = (
        "config_era",
        "triple_subera",
        "triple_subera_fine",
        "ft_unfreeze_side",
        "post_linters_merge",
        "question_weight",
        "is_new_since_prior",
        "newly_scored",
        "degraded_run",
        "partial_degraded",
        "known_bug",
    )

    def test_the_same_record_gets_identical_tags_by_both_paths(self, tmp_path: Path) -> None:
        fresh = [
            _binary_record(101, 201, WEIGHTED_SLUG, PRE_FLIP_SUBMITTED),
            _binary_record(103, 203, WEIGHTED_SLUG, TRIPLE_SUBMITTED),
            # A canonical known-bug question, so the cohort half of the tagging is exercised.
            _binary_record(int(sorted(round_dataset.KNOWN_BUG_QUESTION_IDS)[0]), 204, WEIGHTED_SLUG, TRIPLE_SUBMITTED),
        ]
        prior = [_binary_record(101, 201, WEIGHTED_SLUG, PRE_FLIP_SUBMITTED)]
        weights = {101: 1.0, 103: 0.5}
        spec = _write_round(tmp_path, fresh=fresh, prior=prior, weights=weights)
        by_round = {r["question_id"]: r for r in build_round_dataset(spec).records}

        overlay = [{**record} for record in json.loads((spec.round_dir / f"perf_{WEIGHTED_SLUG}.json").read_text())]
        for record in overlay:
            record["source_tournament"] = WEIGHTED_SLUG
        tag_fresh_records(
            overlay,
            prior_records=prior,
            prior_label=spec.prior_label,
            weights=weights,
            weighted_slug=WEIGHTED_SLUG,
            telemetry_dir=spec.telemetry_dir,
        )

        for record in overlay:
            twin = by_round[record["question_id"]]
            assert {f: record[f] for f in self.TAGGED_FIELDS} == {f: twin[f] for f in self.TAGGED_FIELDS}
        assert any(record["known_bug"] for record in overlay)
        assert [record["is_new_since_prior"] for record in overlay] == [False, True, True]


class TestWriteRoundOutputs:
    def test_all_four_files_land_with_the_expected_shapes(self, tmp_path: Path, caplog) -> None:
        fresh = [
            _binary_record(101, 201, WEIGHTED_SLUG, TRIPLE_SUBMITTED),
            _binary_record(102, 202, WEIGHTED_SLUG, POST_FLIP_SUBMITTED),
        ]
        spec = _write_round(tmp_path, fresh=fresh, prior=[], weights={101: 1.0, 102: 1.0})
        with caplog.at_level(logging.INFO):
            write_round_outputs(build_round_dataset(spec))

        tagged = json.loads((spec.round_dir / "perf_all_tagged.json").read_text())
        assert [r["question_id"] for r in tagged] == [101, 102]
        new_since = json.loads((spec.round_dir / "new_since_prior.json").read_text())
        assert {r["question_id"] for r in new_since} == {101, 102}
        assert new_since[0]["prior_round_view"] is None
        counts = json.loads((spec.round_dir / "counts_by_era.json").read_text())
        assert counts["totals"] == {
            "records": 2,
            "scored": 2,
            "new_since_prior": 2,
            "newly_scored": 2,
            "new_union": 2,
            "triple_era_total": 1,
            "triple_era_scored": 1,
            "triple_era_clean_scored": 1,
            "triple_era_clean_scored_excl_partial": 1,
            "triple_era_clean_numeric": 0,
            "post_time_budget_subera_scored": 0,
            "newest_bot_comment_created_at": TRIPLE_SUBMITTED,
        }
        assert counts["era_by_type"]["triple_era"]["binary"] == {"total": 1, "scored": 1}
        audit = json.loads((spec.round_dir / "degraded_cohort.json").read_text())
        assert audit["resolved_hits"] == []

    def test_the_published_boundary_table_is_the_tracked_one(self, tmp_path: Path) -> None:
        spec = _write_round(
            tmp_path, fresh=[_binary_record(1, 2, WEIGHTED_SLUG, TRIPLE_SUBMITTED)], prior=[], weights={}
        )
        write_round_outputs(build_round_dataset(spec))
        counts = json.loads((spec.round_dir / "counts_by_era.json").read_text())
        assert counts["boundaries_utc"]["flip"] == WIDENING_FLIP_MERGED_AT.isoformat()
        assert counts["boundaries_utc"]["triple_start_b4e9df0"] == B4E9DF0_MERGED_AT.isoformat()


class TestNoRoundStateInTheTrackedSpine:
    """A round literal here is how ten diverging copies happened; the spine takes a RoundSpec."""

    @pytest.mark.parametrize("module", [round_dataset, round_outputs])
    def test_no_round_directory_or_label_is_hardcoded(self, module) -> None:
        source = Path(module.__file__).read_text()
        for forbidden in ("scratch/", "residual_20", "fresh_20", "reused_20"):
            assert forbidden not in source, f"{module.__name__} carries round state: {forbidden!r}"
