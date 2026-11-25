"""Tests for the research-archive treatment tags (performance_analysis/research_tags.py).

The tags drive treated/untreated calibration cuts, so the load-bearing behaviors are:
depth-agnostic header detection (artifact ``##`` vs comment-backfill ``###``), the
diagnostics-corroborated confidence ladder on a False anchor read, the gfv2
section-vs-transcript distinction (the section is the treatment marker; the loop banks
a transcript even when it soft-fails), and all-None — never False — for a question
with no archive record.
"""

import json
from pathlib import Path

from metaculus_bot.performance_analysis.research_tags import (
    attach_research_tags,
    load_verified_archive_record,
    research_tags_for_qid,
    research_tags_for_record,
)

DIAG_BLOCK_EMPTY = "## Provider Diagnostics\n- timeseries_anchor: empty | 0 chars | 30 ms\n"
DIAG_BLOCK_OK = "## Provider Diagnostics\n- timeseries_anchor: ok | 855 chars | 1171 ms\n"


def _record(
    text: str,
    *,
    source: str = "artifact",
    diagnostics_block: str | None = None,
    gap_fill_v2: dict | None = None,
    is_trimmed: bool | None = None,
    schema_version: int | None = 2,
    timestamp: str | None = "2026-08-01T00:00:00+00:00",
) -> dict:
    """One archive record. Defaults to the modern schema-v2 artifact writer.

    ``schema_version`` AND ``timestamp`` are load-bearing for the gfv2 tags: only a
    schema-v2 artifact record written after the payload-era boundary (the gap_fill_v2
    key reached main 2026-07-21, three weeks after schema v2) can carry the payload,
    so on anything else its absence is unrecorded rather than measured (see
    ``_writer_can_carry_gfv2_payload``). The default timestamp sits past the boundary.
    """
    return {
        "research_text": text,
        "source": source,
        "schema_version": schema_version,
        "provider_diagnostics_block": diagnostics_block,
        "gap_fill_v2": gap_fill_v2,
        "is_trimmed": is_trimmed,
        "timestamp": timestamp,
    }


class TestAnchorHeaderDetection:
    def test_artifact_header_form(self):
        tags = research_tags_for_record(_record("## Time Series Anchor\nFRED series band...\n"))
        assert tags["anchor_present"] is True
        assert tags["anchor_confidence"] == "header"

    def test_comment_backfill_reheaded_form(self):
        # Backfill records re-head every section one level deeper.
        tags = research_tags_for_record(
            _record("### Time Series Anchor\nFRED series band...\n", source="comment_backfill")
        )
        assert tags["anchor_present"] is True
        assert tags["research_source_class"] == "comment_backfill"

    def test_title_must_match_exactly(self):
        # A prose mention or an extended title is not the section header.
        text = "The Time Series Anchor fired.\n## Time Series Anchor Notes\n"
        assert research_tags_for_record(_record(text))["anchor_present"] is False


class TestAnchorConfidenceLadder:
    def test_diag_empty_confirms_absent(self):
        tags = research_tags_for_record(_record("## News Articles (AskNews)\n", diagnostics_block=DIAG_BLOCK_EMPTY))
        assert tags["anchor_present"] is False
        assert tags["anchor_confidence"] == "diag_confirms_absent"

    def test_diag_ok_with_missing_header_is_flagged(self):
        # The provider ran (diag ok) but the text lost the section (e.g. trimming):
        # treatment is genuinely unclear and must not read as a confirmed absence.
        tags = research_tags_for_record(_record("tail of a trimmed bundle", diagnostics_block=DIAG_BLOCK_OK))
        assert tags["anchor_present"] is False
        assert tags["anchor_confidence"] == "diag_ok_header_missing"

    def test_diag_block_read_from_research_text_fallback(self):
        # Older records carry the diagnostics block only inline in the text.
        tags = research_tags_for_record(_record("some research\n" + DIAG_BLOCK_EMPTY))
        assert tags["anchor_confidence"] == "diag_confirms_absent"

    def test_trimmed_record_without_diag_is_ambiguous(self):
        # Trimming keeps header + tail, so a leading section can be eaten silently.
        tags = research_tags_for_record(_record("tail only", is_trimmed=True))
        assert tags["anchor_confidence"] == "ambiguous_trimmed_no_diag"

    def test_untrimmed_record_without_diag(self):
        assert research_tags_for_record(_record("no sections"))["anchor_confidence"] == "absent_no_diag"


class TestGfv2SectionVsTranscript:
    def test_section_present_and_loop_ran(self):
        tags = research_tags_for_record(
            _record("## Agentic Research Findings\nfindings...\n", gap_fill_v2={"steps": 7})
        )
        assert tags["gfv2_present"] is True
        assert tags["gfv2_loop_ran"] is True
        assert tags["gfv2_confidence"] == "header"

    def test_transcript_without_section_is_not_treatment(self):
        # The v2 driver banks a transcript even when it soft-fails and contributes no
        # section (the dry-key cohort's shape). Only the SECTION marks treatment.
        tags = research_tags_for_record(_record("## News Articles (AskNews)\n", gap_fill_v2={"steps": 0}))
        assert tags["gfv2_present"] is False
        assert tags["gfv2_loop_ran"] is True
        assert tags["gfv2_confidence"] == "payload_ran_no_section"

    def test_neither(self):
        tags = research_tags_for_record(_record("## News Articles (AskNews)\n"))
        assert tags["gfv2_present"] is False
        assert tags["gfv2_loop_ran"] is False
        assert tags["gfv2_confidence"] == "payload_confirms_absent"

    def test_reheaded_gfv2_section(self):
        tags = research_tags_for_record(_record("### Agentic Research Findings\n", source="comment_backfill"))
        assert tags["gfv2_present"] is True


class TestGfv2LoopRanIsTernary:
    """The fix that keeps the planned v2 treated/untreated split honest.

    ``bool(record.get("gap_fill_v2"))`` collapsed "this writer cannot carry the field"
    into False, which put 880 archived can't-carry records into the untreated arm
    against 77 measured ones. Only a schema-v2 artifact record can carry the payload:
    that writer omits the key when the loop did not run, so there its absence IS the
    measurement, and everywhere else it is silence.
    """

    def test_comment_backfill_record_reads_none_not_false(self):
        tags = research_tags_for_record(
            _record("### News Articles (AskNews)\n", source="comment_backfill", schema_version=1)
        )
        assert tags["gfv2_loop_ran"] is None
        assert tags["gfv2_confidence"] == "absent_no_payload"

    def test_schema_v1_artifact_record_reads_none_not_false(self):
        # A live-capture record from before the payload existed. The run may well have
        # had v2 on; the writer simply had nowhere to say so.
        tags = research_tags_for_record(_record("## News Articles (AskNews)\n", schema_version=1))
        assert tags["gfv2_loop_ran"] is None

    def test_missing_schema_version_reads_none_not_false(self):
        tags = research_tags_for_record(_record("## News Articles (AskNews)\n", schema_version=None))
        assert tags["gfv2_loop_ran"] is None

    def test_trimmed_uncarryable_record_is_flagged_ambiguous(self):
        # The middle-trim keeps head + tail, so a v2 section sitting mid-bundle can be
        # eaten — the False header read is not evidence of an untreated question.
        tags = research_tags_for_record(
            _record("tail only", source="comment_backfill", schema_version=1, is_trimmed=True)
        )
        assert tags["gfv2_present"] is False
        assert tags["gfv2_loop_ran"] is None
        assert tags["gfv2_confidence"] == "ambiguous_trimmed_no_payload"

    def test_a_section_proves_the_loop_ran_whatever_the_writer(self):
        # The section IS the loop's output, so it cannot exist without a run. That
        # observation is independent of whether the writer could bank the payload.
        tags = research_tags_for_record(
            _record("### Agentic Research Findings\n", source="comment_backfill", schema_version=1)
        )
        assert tags["gfv2_loop_ran"] is True
        assert tags["gfv2_confidence"] == "header"

    def test_a_null_payload_is_not_a_recorded_loop(self):
        # The writer only ever writes a non-None payload, but a null must never read as
        # a run — that is the same key-present-but-empty trap the bool() collapse was.
        tags = research_tags_for_record(_record("## News Articles (AskNews)\n", gap_fill_v2=None))
        assert tags["gfv2_loop_ran"] is False
        assert tags["gfv2_confidence"] == "payload_confirms_absent"

    def test_a_schema_v2_artifact_from_the_pre_payload_window_reads_none(self):
        # Schema v2 landed 2026-06-28 but the gap_fill_v2 write only reached main
        # 2026-07-21 (b4e9df0) — a v2 artifact from that ~3-week window physically
        # cannot carry the key, so its silence is not "the only confident untreated
        # read"; it is the same can't-carry silence as a schema-v1 record.
        tags = research_tags_for_record(_record("## News Articles (AskNews)\n", timestamp="2026-07-05T12:00:00+00:00"))
        assert tags["gfv2_loop_ran"] is None
        assert tags["gfv2_confidence"] == "absent_no_payload"

    def test_an_undatable_v2_artifact_reads_none_not_false(self):
        # No parseable timestamp means the writer era is unknowable; the conservative
        # read is can't-carry, never a confident False.
        tags = research_tags_for_record(_record("## News Articles (AskNews)\n", timestamp=None))
        assert tags["gfv2_loop_ran"] is None
        assert tags["gfv2_confidence"] == "absent_no_payload"


class TestAbsentArchiveRecord:
    def test_missing_file_yields_all_none(self, tmp_path: Path):
        tags = research_tags_for_qid(99999, tmp_path)
        assert tags == {
            "anchor_present": None,
            "gfv2_present": None,
            "gfv2_loop_ran": None,
            "gfv2_confidence": None,
            "anchor_confidence": None,
            "research_source_class": None,
        }

    def test_none_qid_yields_all_none(self, tmp_path: Path):
        assert research_tags_for_qid(None, tmp_path)["anchor_present"] is None

    def test_unreadable_record_yields_all_none(self, tmp_path: Path):
        (tmp_path / "123.json").write_text("{not json")
        assert research_tags_for_qid(123, tmp_path)["anchor_present"] is None


class TestAttachResearchTags:
    def test_stamps_tags_by_question_id(self, tmp_path: Path):
        (tmp_path / "101.json").write_text(
            json.dumps(
                {
                    "research_text": "## Time Series Anchor\nband\n## Agentic Research Findings\nfindings\n",
                    "source": "artifact",
                    "gap_fill_v2": {"steps": 3},
                }
            )
        )
        records = [
            {"question_id": 101, "type": "numeric"},
            {"question_id": 202, "type": "binary"},  # no archive record
        ]
        attach_research_tags(records, tmp_path)
        assert records[0]["anchor_present"] is True
        assert records[0]["gfv2_present"] is True
        assert records[0]["research_source_class"] == "artifact"
        assert records[1]["anchor_present"] is None
        assert records[1]["gfv2_present"] is None

    def test_foreign_record_is_rejected_by_identity_check(self, tmp_path: Path):
        # latest/ mixes two id spaces: a file named after OUR question_id can hold a
        # DIFFERENT question's record (the measured latest/43592 collision). A loaded
        # record whose self-declared post_id contradicts ours must yield all-None,
        # never another question's treatment status.
        (tmp_path / "101.json").write_text(
            json.dumps(
                {
                    "research_text": "## Time Series Anchor\nband\n",
                    "source": "artifact",
                    "post_id": 999,  # a different question's post
                }
            )
        )
        records = [{"question_id": 101, "post_id": 555, "type": "numeric"}]
        attach_research_tags(records, tmp_path)
        assert records[0]["anchor_present"] is None

    def test_post_id_fallback_finds_post_keyed_record(self, tmp_path: Path):
        # No question-id-keyed file, but the post-id-keyed file self-identifies as
        # ours (post_id matches) — the lookup_order fallback must find it.
        (tmp_path / "555.json").write_text(
            json.dumps(
                {
                    "research_text": "## Time Series Anchor\nband\n",
                    "source": "artifact",
                    "post_id": 555,
                }
            )
        )
        records = [{"question_id": 101, "post_id": 555, "type": "numeric"}]
        attach_research_tags(records, tmp_path)
        assert records[0]["anchor_present"] is True

    def test_log_backfill_record_never_stamps_tags(self, tmp_path: Path):
        # A log_backfill record is POST-id keyed and page_url-identified only; on an
        # id collision the URL check cannot rule out a foreign question.
        (tmp_path / "101.json").write_text(
            json.dumps(
                {
                    "research_text": "## Time Series Anchor\nband\n",
                    "source": "log_backfill",
                    "page_url": "https://www.metaculus.com/questions/101/",
                }
            )
        )
        records = [{"question_id": 101, "post_id": 555, "type": "numeric"}]
        attach_research_tags(records, tmp_path)
        assert records[0]["anchor_present"] is None


class TestResearchTagsForQid:
    """The bare-qid entry point (exported, and what a one-off analysis script calls)."""

    def test_reads_the_archive_record_on_disk(self, tmp_path: Path):
        (tmp_path / "101.json").write_text(
            json.dumps(
                {
                    "research_text": "## Time Series Anchor\nband\n",
                    "source": "artifact",
                    "gap_fill_v2": {"steps": 2},
                }
            )
        )
        tags = research_tags_for_qid(101, tmp_path)
        assert tags["anchor_present"] is True
        assert tags["anchor_confidence"] == "header"
        assert tags["gfv2_present"] is False
        assert tags["gfv2_loop_ran"] is True
        assert tags["research_source_class"] == "artifact"

    def test_non_dict_record_yields_all_none(self, tmp_path: Path):
        # A JSON file holding a list (or any non-object) is not a record.
        (tmp_path / "102.json").write_text(json.dumps(["not", "a", "record"]))
        assert research_tags_for_qid(102, tmp_path)["anchor_present"] is None


class TestLogBackfillVetoOnQidLookup:
    def test_research_tags_for_qid_rejects_log_backfill(self, tmp_path: Path):
        (tmp_path / "101.json").write_text(
            json.dumps({"research_text": "## Time Series Anchor\nband\n", "source": "log_backfill"})
        )
        assert research_tags_for_qid(101, tmp_path)["anchor_present"] is None


class TestPlatformSeparation:
    """Mantic question ids share the integer space with Metaculus ids, so ``latest/<id>.json`` can
    hold a record from either platform. A record with no ``platform`` field is Metaculus: every
    archive record written before Mantic arrived is."""

    def _write_mantic_record(self, latest_dir: Path) -> None:
        (latest_dir / "101.json").write_text(
            json.dumps(
                {
                    **_record("## Time Series Anchor\nband\n"),
                    "platform": "mantic",
                    "page_url": "https://competitions.mantic.com/questions/101/",
                }
            )
        )

    def test_a_mantic_archive_record_never_joins_a_metaculus_question(self, tmp_path: Path):
        self._write_mantic_record(tmp_path)

        assert load_verified_archive_record({"question_id": 101, "post_id": 101}, tmp_path) is None

    def test_it_joins_the_mantic_question_with_the_same_id(self, tmp_path: Path):
        self._write_mantic_record(tmp_path)

        loaded = load_verified_archive_record({"question_id": 101, "post_id": 101, "platform": "mantic"}, tmp_path)

        assert loaded is not None
        assert loaded["platform"] == "mantic"

    def test_an_archive_record_without_a_platform_is_metaculus(self, tmp_path: Path):
        (tmp_path / "101.json").write_text(json.dumps(_record("text")))

        assert load_verified_archive_record({"question_id": 101}, tmp_path) is not None
        assert load_verified_archive_record({"question_id": 101, "platform": "mantic"}, tmp_path) is None
