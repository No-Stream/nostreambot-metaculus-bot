from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from metaculus_bot import value_extraction
from scripts.probes.bench_common.paid_call import SpendMeter
from scripts.probes.cross_pollination_bench import schedule
from scripts.probes.cross_pollination_bench.manifest import build_contexts, build_manifest
from scripts.probes.cross_pollination_bench.models import BENCH_MODELS
from scripts.probes.cross_pollination_bench.request import message_sha256
from scripts.probes.cross_pollination_bench.rows import append_row, call_key, load_rows, saved_replies
from scripts.probes.cross_pollination_bench.schedule import (
    MAIN_RUN,
    SMOKE_RUN,
    planned_calls,
    run_question,
    run_questions,
)
from tests.cross_pollination_bench_fakes import FakeBenchModel, write_fake_round


@pytest.fixture
def setup_run(tmp_path: Path):
    fake = write_fake_round(tmp_path)
    manifest = build_manifest(fake.perf_path, fake.archive_dir, event_map=None, fetch=fake.fetch)
    contexts = {context.entry.question_id: context for context in build_contexts(manifest, fake.archive_dir)}
    events = []
    calls = {model.label: FakeBenchModel(model.label, events=events) for model in BENCH_MODELS}
    parsers = {model.label: MagicMock(model="parser") for model in BENCH_MODELS}
    return contexts, calls, parsers, events


def run(context, calls, parsers, *, saved=None, meter=None, config=MAIN_RUN):
    return asyncio.run(
        run_question(
            context,
            models=BENCH_MODELS,
            calls=calls,
            parsers=parsers,
            saved=saved or {},
            meter=meter or SpendMeter(10),
            config=config,
            on_row=lambda row: None,
        )
    )


def test_two_draws_make_twenty_four_calls(setup_run):
    contexts, calls, parsers, events = setup_run
    result = run(contexts[101], calls, parsers)
    assert len(result) == 24
    assert all(row.status == "scored" for row in result)
    assert planned_calls(58, 3, MAIN_RUN) == 1392
    assert planned_calls(4, 3, SMOKE_RUN) == 48
    assert len(events) == 24


def test_order_and_cached_prefix(setup_run):
    contexts, calls, parsers, events = setup_run
    run(contexts[101], calls, parsers)
    first = [event for event in events if len(event[2][0]["content"]) == 1]
    updates = [event for event in events if len(event[2][0]["content"]) == 2]
    assert max(event[4] for event in first) < min(event[3] for event in updates)
    for model in BENCH_MODELS:
        model_first = [event for event in first if event[0] == model.label]
        assert model_first[0][4] < model_first[1][3]
        model_events = [event for event in events if event[0] == model.label]
        assert len({event[1] for event in model_events}) == 1
        assert all(event[2][0]["content"][0] == model_first[0][2][0]["content"][0] for event in model_events)
        assert model_first[0][2][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_saved_replies_and_changed_prompt(setup_run):
    contexts, calls, parsers, events = setup_run
    result = run(contexts[101], calls, parsers)
    saved = {call_key(row): row for row in result}
    events.clear()
    assert len(run(contexts[101], calls, parsers, saved=saved)) == 24
    assert not events
    with pytest.raises(ValueError, match="101"):
        run(replace(contexts[101], prompt="changed"), calls, parsers, saved=saved)
    run(contexts[101], calls, parsers, saved={key: row for key, row in saved.items() if row.stage == "first"})
    assert len(events) == 18


def test_numeric_rows_store_the_cdf_and_the_stated_percentiles(setup_run):
    contexts, calls, parsers, _ = setup_run
    rows = run(contexts[103], calls, parsers, config=SMOKE_RUN)
    assert all(isinstance(row.forecast, list) and len(row.forecast) == 201 and row.stated_percentiles for row in rows)


def test_the_meter_stops_later_calls(setup_run):
    contexts, calls, parsers, events = setup_run
    rows = run(contexts[101], calls, parsers, meter=SpendMeter(0))
    assert not events
    assert all(row.status == "skipped_spend_cap" for row in rows)


def test_a_first_call_error_is_final_once_the_draw_has_updates(setup_run):
    contexts, calls, parsers, events = setup_run
    calls["haiku"].api_error_first_draws = frozenset({1})
    rows = run(contexts[101], calls, parsers)
    saved = {call_key(row): row for row in rows}
    calls["haiku"].api_error_first_draws = frozenset()
    events.clear()
    run(contexts[101], calls, parsers, saved=saved)
    assert not events
    without_updates = {key: row for key, row in saved.items() if key[2] != 1 or key[3] == "first"}
    run(contexts[101], calls, parsers, saved=without_updates)
    assert any(event[0] == "haiku" and len(event[2][0]["content"]) == 1 for event in events)


def test_a_lone_survivor_skips_the_peer_variants(setup_run):
    contexts, calls, parsers, _ = setup_run
    calls["haiku"].api_error_first_draws = frozenset({1})
    calls["deepseek"].api_error_first_draws = frozenset({1})
    rows = run(contexts[101], calls, parsers, config=SMOKE_RUN)
    assert sum(row.status == "skipped_no_peers" for row in rows) == 2
    assert any(row.stage == "second_look" and row.status == "scored" for row in rows)


def test_unpriced_reply_stops_the_run(setup_run):
    contexts, calls, parsers, _ = setup_run
    for call in calls.values():
        call.charged_usd = None
    rows = run(contexts[101], calls, parsers)
    assert any(row.status == "skipped_spend_cap" for row in rows)


def test_smoke_order_sends_full_answers_first(setup_run):
    contexts, calls, parsers, events = setup_run
    rows = run(contexts[101], calls, parsers, config=SMOKE_RUN)
    for model in BENCH_MODELS:
        full_row = next(row for row in rows if row.model == model.label and row.stage == "peers_full")
        other_rows = [
            row for row in rows if row.model == model.label and row.stage in ("peers_forecasts", "second_look")
        ]
        full_event = next(
            event for event in events if event[0] == model.label and full_row.prompt_sha256 == message_sha256(event[2])
        )
        other_events = [
            event
            for event in events
            if event[0] == model.label and any(row.prompt_sha256 == message_sha256(event[2]) for row in other_rows)
        ]
        assert all(full_event[4] < event[3] for event in other_events)


def test_reply_is_saved_before_scoring_waits_and_resumed_without_a_call(setup_run, monkeypatch):
    contexts, calls, parsers, events = setup_run
    stored = []
    original_score = schedule.score_reply

    async def interrupted_score(*args, **kwargs):
        assert stored
        assert stored[-1].reply_text
        raise asyncio.CancelledError

    monkeypatch.setattr(schedule, "score_reply", interrupted_score)

    async def interrupted():
        await run_question(
            contexts[101],
            models=BENCH_MODELS[:1],
            calls=calls,
            parsers=parsers,
            saved={},
            meter=SpendMeter(10),
            config=SMOKE_RUN,
            on_row=stored.append,
        )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(interrupted())
    saved = saved_replies(stored)
    assert len(saved) == 1
    events.clear()
    monkeypatch.setattr(schedule, "score_reply", original_score)

    async def resume():
        return await run_question(
            contexts[101],
            models=BENCH_MODELS[:1],
            calls=calls,
            parsers=parsers,
            saved=saved,
            meter=SpendMeter(10),
            config=SMOKE_RUN,
            on_row=lambda row: None,
        )

    result = asyncio.run(resume())
    assert result[0].status == "scored"
    assert all(len(event[2][0]["content"]) != 1 for event in events)


def test_saved_rows_preserve_failed_first_calls_for_draws_with_updates(setup_run):
    contexts, calls, parsers, events = setup_run
    calls["haiku"].api_error_first_draws = frozenset({1})
    rows = run(contexts[101], calls, parsers)
    saved = saved_replies(rows)
    assert saved[(101, "haiku", 1, "first")].status == "api_error"
    calls["haiku"].api_error_first_draws = frozenset()
    events.clear()
    run(contexts[101], calls, parsers, saved=saved)
    assert not events


def test_peers_come_from_the_same_draw_under_neutral_labels(setup_run):
    contexts, calls, parsers, events = setup_run
    rows = run(contexts[101], calls, parsers)
    first = {(row.model, row.draw): row for row in rows if row.stage == "first"}
    for row in rows:
        if row.stage != "peers_full" or row.draw != 2:
            continue
        event = next(
            event for event in events if event[0] == row.model and message_sha256(event[2]) == row.prompt_sha256
        )
        block = event[2][0]["content"][1]["text"]
        assert "Forecaster A" in block
        assert "Forecaster B" in block
        assert row.peer_order is not None
        for peer in row.peer_order:
            assert first[(peer, 2)].reply_text in block
            assert first[(peer, 1)].reply_text not in block


def test_a_failed_first_call_removes_the_member_for_that_draw(setup_run, monkeypatch):
    contexts, calls, parsers, events = setup_run
    calls["haiku"].unparseable = frozenset({1})

    async def no_salvage(*args, **kwargs):
        raise RuntimeError("no parser LLM in tests")

    monkeypatch.setattr(value_extraction, "parse_structured", no_salvage)
    rows = run(contexts[101], calls, parsers)
    assert next(row for row in rows if row.model == "haiku" and row.draw == 1).status == "extraction_failed"
    assert not [row for row in rows if row.model == "haiku" and row.draw == 1 and row.stage != "first"]
    assert len([row for row in rows if row.draw == 2]) == 12
    for row in rows:
        if row.draw == 1 and row.stage == "peers_full":
            block = next(
                event[2][0]["content"][1]["text"] for event in events if message_sha256(event[2]) == row.prompt_sha256
            )
            assert "Forecaster A" in block
            assert "Forecaster B" not in block
            assert "haiku-1" not in block


def test_account_refusal_stops_later_calls(setup_run):
    contexts, calls, parsers, events = setup_run
    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(403, request=request)

    async def refusal(messages, session):
        raise openai.APIStatusError("refused", response=response, body=None)

    single_calls = {**calls, "haiku": refusal}
    rows = asyncio.run(
        run_question(
            contexts[101],
            models=BENCH_MODELS[:1],
            calls=single_calls,
            parsers=parsers,
            saved={},
            meter=SpendMeter(10),
            config=MAIN_RUN,
            on_row=lambda row: None,
        )
    )
    assert [row.status for row in rows] == ["api_error", "skipped_spend_cap"]
    assert not events


def test_load_rows_returns_latest_checkpoint_per_call(setup_run, tmp_path):
    contexts, calls, parsers, _ = setup_run
    path = tmp_path / "calls.jsonl"
    asyncio.run(
        run_question(
            contexts[101],
            models=BENCH_MODELS,
            calls=calls,
            parsers=parsers,
            saved={},
            meter=SpendMeter(10),
            config=SMOKE_RUN,
            on_row=lambda row: append_row(path, row),
        )
    )
    rows = load_rows(path)
    assert len(rows) == 12
    assert all(row.status == "scored" for row in rows)


def test_saved_failed_updates_are_not_sent_again(setup_run):
    contexts, calls, parsers, events = setup_run
    rows = run(contexts[101], calls, parsers, config=SMOKE_RUN)
    saved = {call_key(row): row for row in rows}
    update = next(row for row in rows if row.stage == "peers_full")
    saved[call_key(update)] = replace(update, status="api_error", reply_text=None, score=None, forecast=None)
    events.clear()
    resumed = run(contexts[101], calls, parsers, saved=saved, config=SMOKE_RUN)
    assert not events
    assert next(row for row in resumed if call_key(row) == call_key(update)).status == "api_error"


def test_measured_cap_stops_calls_that_have_not_started(setup_run):
    contexts, calls, parsers, events = setup_run
    rows = run(contexts[101], calls, parsers, meter=SpendMeter(0.001))
    assert len(events) == 3
    assert sum(row.status == "scored" for row in rows) == 3
    assert all(row.status == "skipped_spend_cap" for row in rows if row.status != "scored")


def test_changed_update_prompt_raises_before_reuse(setup_run):
    contexts, calls, parsers, _ = setup_run
    rows = run(contexts[101], calls, parsers, config=SMOKE_RUN)
    saved = {call_key(row): row for row in rows}
    target = next(row for row in rows if row.stage == "peers_full")
    saved[call_key(target)] = replace(target, prompt_sha256="old hash")
    with pytest.raises(ValueError, match="101"):
        run(contexts[101], calls, parsers, saved=saved, config=SMOKE_RUN)


def test_questions_run_sequentially_by_default_concurrency(setup_run):
    contexts, calls, parsers, events = setup_run
    rows = asyncio.run(
        run_questions(
            [contexts[101], contexts[103]],
            models=BENCH_MODELS,
            calls=calls,
            parsers=parsers,
            saved={},
            meter=SpendMeter(10),
            config=SMOKE_RUN,
            on_row=lambda row: None,
            question_concurrency=1,
        )
    )
    assert len(rows) == 24
    first_question = [event for event in events if "xpoll-101-" in event[1]]
    second_question = [event for event in events if "xpoll-103-" in event[1]]
    assert max(event[4] for event in first_question) < min(event[3] for event in second_question)


def test_fake_repair_only_block_reaches_the_repair_rung(setup_run):
    contexts, calls, parsers, _ = setup_run
    for call in calls.values():
        call.repair_only_block = True
    rows = run(contexts[101], calls, parsers, config=SMOKE_RUN)
    assert all(row.status == "scored" and row.rung == "repair" for row in rows)


def test_fake_replies_have_distinct_per_model_call_tags(setup_run):
    contexts, calls, parsers, _ = setup_run
    rows = run(contexts[101], calls, parsers)
    for model in BENCH_MODELS:
        ids = [row.response_id for row in rows if row.model == model.label]
        assert len(set(ids)) == len(ids) == 8


def test_fake_counts_first_draws_separately_by_question(setup_run):
    contexts, calls, parsers, _ = setup_run
    calls["haiku"].api_error_first_draws = frozenset({1})
    first_rows = run(contexts[101], calls, parsers, config=SMOKE_RUN)
    second_rows = run(contexts[103], calls, parsers, config=SMOKE_RUN)
    assert next(row for row in first_rows if row.model == "haiku").status == "api_error"
    assert next(row for row in second_rows if row.model == "haiku").status == "api_error"


def test_fresh_fake_can_send_updates_from_saved_first_rows(setup_run):
    contexts, calls, parsers, _ = setup_run
    rows = run(contexts[101], calls, parsers, config=SMOKE_RUN)
    saved = {call_key(row): row for row in rows if row.stage == "first"}
    new_calls = {model.label: FakeBenchModel(model.label) for model in BENCH_MODELS}
    resumed = run(contexts[101], new_calls, parsers, saved=saved, config=SMOKE_RUN)
    assert len(resumed) == 12
    assert all(row.status == "scored" for row in resumed)
