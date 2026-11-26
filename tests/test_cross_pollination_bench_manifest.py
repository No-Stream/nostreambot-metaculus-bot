from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import DiscreteQuestion, OutOfBoundsResolution

from metaculus_bot import prompts
from metaculus_bot.constants import (
    CROSS_POLLINATION_BENCH_DEEPSEEK_MODEL,
    CROSS_POLLINATION_BENCH_GLM_MODEL,
    CROSS_POLLINATION_BENCH_HAIKU_MODEL,
)
from scripts.probes.cross_pollination_bench import manifest, models
from scripts.probes.cross_pollination_bench.manifest import (
    assign_event_keys,
    bench_resolution,
    build_contexts,
    build_manifest,
    fetch_question,
    load_manifest,
    rebuild_question,
    research_text,
    select_records,
    write_manifest,
)
from scripts.probes.cross_pollination_bench.rows import (
    REPLY_STATUSES,
    STATUS_SKIPPED_NO_PEERS,
    BenchCallRow,
    append_row,
    call_key,
    load_rows,
    saved_replies,
)
from tests.cross_pollination_bench_fakes import write_fake_round


def test_the_leak_rule_is_a_utc_date_comparison(tmp_path: Path) -> None:
    fake_round = write_fake_round(tmp_path)
    records = select_records(json.loads(fake_round.perf_path.read_text()), fake_round.archive_dir)
    assert [row[0]["question_id"] for row in records] == [101, 102, 103, 104, 105, 108]


def test_after_release_is_flagged(tmp_path: Path) -> None:
    fake_round = write_fake_round(tmp_path)
    manifest_result = build_manifest(
        fake_round.perf_path, fake_round.archive_dir, event_map=None, fetch=fake_round.fetch
    )
    assert [entry.question_id for entry in manifest_result.questions if entry.after_release] == [108]


def test_an_archive_record_for_another_post_raises(tmp_path: Path) -> None:
    fake_round = write_fake_round(tmp_path)
    archive_path = fake_round.archive_dir / "101.json"
    record = json.loads(archive_path.read_text())
    record["post_id"] = 999
    archive_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="101"):
        select_records(json.loads(fake_round.perf_path.read_text()), fake_round.archive_dir)


def test_records_written_before_the_platform_field_read_as_metaculus(tmp_path: Path) -> None:
    fake_round = write_fake_round(tmp_path)
    old_record_path = fake_round.archive_dir / "106.json"
    old_record = json.loads(old_record_path.read_text())
    del old_record["platform"], old_record["post_id"]
    old_record_path.write_text(json.dumps(old_record))
    selected_record_path = fake_round.archive_dir / "101.json"
    selected_record = json.loads(selected_record_path.read_text())
    del selected_record["platform"]
    selected_record_path.write_text(json.dumps(selected_record))
    perf_rows = json.loads(fake_round.perf_path.read_text())
    for row in perf_rows:
        if row["question_id"] in (101, 106):
            del row["platform"]
    records = select_records(perf_rows, fake_round.archive_dir)
    assert [int(row["question_id"]) for row, _ in records] == [101, 102, 103, 104, 105, 108]


def test_duplicate_titles_share_an_event_and_the_map_overrides(tmp_path: Path) -> None:
    fake_round = write_fake_round(tmp_path)
    result = build_manifest(fake_round.perf_path, fake_round.archive_dir, event_map=None, fetch=fake_round.fetch)
    by_id = {entry.question_id: entry for entry in result.questions}
    assert by_id[101].event_key == by_id[105].event_key
    event_map = {103: "x", 104: "x"}
    keys = assign_event_keys({103: "one", 104: "two"}, event_map)
    assert keys == {103: "x", 104: "x"}
    with pytest.raises(ValueError, match="999"):
        assign_event_keys({103: "one"}, {999: "x"})


def test_the_default_map_clusters_brazil_and_alaska_only() -> None:
    titles = {qid: f"Question {qid}" for qid in (46184, 46185, 45920, 46423, 46426, 46066, 46457, 46458)}
    mapped = assign_event_keys(titles, None)
    assert len(set(mapped.values())) == 5
    assert mapped[46184] == mapped[46185] == mapped[45920]
    assert mapped[46423] == mapped[46426]
    assert len({mapped[qid] for qid in (46066, 46457, 46458)}) == 3
    assert len(set(assign_event_keys(titles, {}).values())) == len(titles)
    assert len(set(assign_event_keys({101: "A", 102: "B", 103: "C", 104: "D"}, None).values())) == 4


def test_fetch_selects_the_sub_question_by_id(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake_round = write_fake_round(tmp_path)
    fetched_group = fake_round.fetch(202, 102)
    expected = fetched_group[0]
    calls: list[tuple[int, str]] = []

    def fetch_post(post_id: int, *, group_question_mode: str) -> list[object]:
        calls.append((post_id, group_question_mode))
        return fetched_group

    monkeypatch.setattr(manifest.MetaculusApi, "get_question_by_post_id", fetch_post)
    assert fetch_question(202, 102) is expected
    assert calls == [(202, "unpack_subquestions")]
    monkeypatch.setattr(
        manifest.MetaculusApi,
        "get_question_by_post_id",
        lambda post_id, *, group_question_mode: [expected],
    )
    with pytest.raises(ValueError, match="999"):
        fetch_question(202, 999)


def test_a_type_mismatch_raises(tmp_path: Path) -> None:
    fake_round = write_fake_round(tmp_path)
    original_fetch = fake_round.fetch
    fake_round.fetch = lambda post_id, question_id: (
        BinaryQuestion(question_text="wrong", id_of_question=103, id_of_post=203)
        if question_id == 103
        else original_fetch(post_id, question_id)
    )
    with pytest.raises(ValueError, match="103"):
        build_manifest(fake_round.perf_path, fake_round.archive_dir, event_map=None, fetch=fake_round.fetch)


def test_the_manifest_stores_criteria_and_a_clean_question(tmp_path: Path) -> None:
    fake_round = write_fake_round(tmp_path)
    result = build_manifest(fake_round.perf_path, fake_round.archive_dir, event_map=None, fetch=fake_round.fetch)
    entry = next(question for question in result.questions if question.question_id == 101)
    assert entry.resolution_criteria == "Criteria for 101."
    assert entry.fine_print == "Fine print for 101."
    assert entry.question_json["background_info"] == ""
    assert entry.question_json["resolution_string"] is None
    assert entry.api_resolution_string == "API resolution 101"


def test_manifest_round_trips(tmp_path: Path) -> None:
    fake_round = write_fake_round(tmp_path)
    original = build_manifest(fake_round.perf_path, fake_round.archive_dir, event_map=None, fetch=fake_round.fetch)
    path = tmp_path / "manifest.json"
    write_manifest(original, path)
    loaded = load_manifest(path)
    assert loaded == original
    questions = {entry.question_id: rebuild_question(entry) for entry in loaded.questions}
    assert isinstance(questions[101], BinaryQuestion)
    assert isinstance(questions[102], MultipleChoiceQuestion)
    assert isinstance(questions[103], NumericQuestion)
    assert isinstance(questions[104], DiscreteQuestion)
    assert questions[103].lower_bound == 0.0
    assert questions[103].upper_bound == 100.0
    assert questions[103].cdf_size == 201
    assert (
        bench_resolution(next(entry for entry in loaded.questions if entry.question_id == 101), questions[101]) is False
    )
    assert (
        bench_resolution(next(entry for entry in loaded.questions if entry.question_id == 102), questions[102]) == "A"
    )
    assert (
        bench_resolution(next(entry for entry in loaded.questions if entry.question_id == 103), questions[103]) == 55.5
    )
    assert bench_resolution(next(entry for entry in loaded.questions if entry.question_id == 104), questions[104]) is (
        OutOfBoundsResolution.ABOVE_UPPER_BOUND
    )


def test_research_text_refuses_a_changed_archive(tmp_path: Path) -> None:
    fake_round = write_fake_round(tmp_path)
    result = build_manifest(fake_round.perf_path, fake_round.archive_dir, event_map=None, fetch=fake_round.fetch)
    entry = next(question for question in result.questions if question.question_id == 101)
    (fake_round.archive_dir / "101.json").write_text(
        json.dumps({"research_text": "Changed archived research."}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="101"):
        research_text(entry, fake_round.archive_dir)


def test_the_prompt_shows_the_archive_date_and_a_blank_background(tmp_path: Path) -> None:
    fake_round = write_fake_round(tmp_path)
    result = build_manifest(fake_round.perf_path, fake_round.archive_dir, event_map=None, fetch=fake_round.fetch)
    entry = next(question for question in result.questions if question.question_id == 108)
    original_window = prompts._forecasting_window_str
    context = next(context for context in build_contexts(result, fake_round.archive_dir) if context.entry == entry)
    assert "Today: 2026-10-18" in context.prompt
    assert "Question background:\n\n" in context.prompt
    assert "Question background:\nNone" not in context.prompt
    assert "Archived research for question 108." in context.prompt
    assert prompts._forecasting_window_str is original_window


def test_saved_replies_keeps_the_last_paid_reply_per_key(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    row = BenchCallRow(101, "binary", "haiku", 0, "first", "api_error", "hash", None, None, key_alias="personal")
    append_row(path, row)
    paid = replace(row, status="scored", score=1.0, forecast=0.7)
    append_row(path, paid)
    skipped = replace(row, question_id=102, status="skipped_spend_cap")
    append_row(path, skipped)
    loaded = load_rows(path)
    result = saved_replies(loaded)
    assert result[call_key(paid)] == paid
    assert call_key(skipped) not in result
    assert STATUS_SKIPPED_NO_PEERS not in REPLY_STATUSES


def test_bench_models_pin_provider_key_and_price() -> None:
    assert tuple(model.label for model in models.BENCH_MODELS) == ("haiku", "deepseek", "glm")
    assert tuple(model.key_route for model in models.BENCH_MODELS) == ("shared", "personal", "personal")
    assert tuple(model.slug for model in models.BENCH_MODELS) == (
        CROSS_POLLINATION_BENCH_HAIKU_MODEL,
        CROSS_POLLINATION_BENCH_DEEPSEEK_MODEL,
        CROSS_POLLINATION_BENCH_GLM_MODEL,
    )
    assert (models.BENCH_MODELS[0].input_usd_per_m, models.BENCH_MODELS[0].cache_write_usd_per_m) == (0.10, 0.125)
