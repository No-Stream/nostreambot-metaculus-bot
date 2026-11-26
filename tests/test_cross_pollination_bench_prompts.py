from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from forecasting_tools import BinaryQuestion

from scripts.probes.bench_common.scoring import score_reply
from scripts.probes.cross_pollination_bench.prompts import (
    PEER_LABELS,
    UPDATE_INSTRUCTION,
    PeerAnswer,
    parse_correction_kind,
    peer_order,
    render_forecast_values,
    update_block,
)


def _question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will it happen?",
        id_of_question=1,
        id_of_post=1,
        page_url="https://www.metaculus.com/questions/1/",
        background_info="",
        resolution_criteria="Resolves yes if it happens.",
        fine_print="",
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
        scheduled_resolution_time=datetime(2027, 1, 1, tzinfo=UTC),
        api_json={"question": {}},
    )


def _peers() -> list[PeerAnswer]:
    return [
        PeerAnswer(reply_text="PEER-REASONING-SENTINEL", forecast_text="Probability of Yes: 0.37"),
        PeerAnswer(reply_text="Other reasoning", forecast_text="Probability of Yes: 0.61"),
    ]


def test_the_instruction_is_identical_in_all_three_variants() -> None:
    blocks = {
        variant: update_block(variant, "OWN ANSWER", _peers())
        for variant in ("peers_full", "peers_forecasts", "second_look")
    }
    second_look_without_instruction = blocks["second_look"].removesuffix(UPDATE_INSTRUCTION)

    assert all(block.endswith(UPDATE_INSTRUCTION) for block in blocks.values())
    for variant in ("peers_full", "peers_forecasts"):
        peer_section = "── OTHER FORECASTERS ──\n" + blocks[variant].split("── OTHER FORECASTERS ──\n", 1)[
            1
        ].removesuffix(UPDATE_INSTRUCTION)
        without_peers = blocks[variant].replace(peer_section, "")
        assert without_peers.removesuffix(UPDATE_INSTRUCTION) == second_look_without_instruction


def test_second_look_shows_nothing_from_peers() -> None:
    block = update_block("second_look", "OWN ANSWER", _peers())

    assert "Forecaster A" not in block
    assert "OTHER FORECASTERS" not in block


def test_forecasts_only_shows_values_and_no_reasoning() -> None:
    block = update_block("peers_forecasts", "OWN ANSWER", _peers())

    assert "PEER-REASONING-SENTINEL" not in block
    assert "Probability of Yes: 0.37" in block


def test_full_answers_show_the_peer_reply_under_a_neutral_label() -> None:
    block = update_block("peers_full", "OWN ANSWER", _peers())

    assert f"{PEER_LABELS[0]}\nPEER-REASONING-SENTINEL" in block
    assert not any(model_name in block.lower() for model_name in ("haiku", "deepseek", "glm", "claude"))


def test_numeric_peer_values_are_the_stated_percentiles() -> None:
    rendered = render_forecast_values(
        "numeric",
        [float(value) for value in range(201)],
        options=None,
        stated_percentiles=[[0.05, 12.3], [0.5, 50.0], [0.95, 95.7]],
    )

    assert rendered.splitlines() == ["Percentile 5: 12.3", "Percentile 50: 50", "Percentile 95: 95.7"]


def test_peer_order_is_seeded_stable_and_varies_by_question() -> None:
    peers = ["alpha", "beta"]
    first_order = peer_order(1, "member", peers, seed=42)

    assert peer_order(1, "member", peers, seed=42) == first_order
    assert len({tuple(peer_order(question_id, "member", peers, seed=42)) for question_id in range(1, 51)}) == 2


def test_correction_kind_parsing() -> None:
    kinds = ("fact", "criteria", "base_rate", "reweighting", "none")

    assert [parse_correction_kind(f"UPDATE: prior=0.2 revised=0.3 correction={kind}") for kind in kinds] == list(kinds)
    assert parse_correction_kind("UPDATE: prior=0.2 revised=0.3 correction=fact\nUPDATE: correction=none") == "none"
    assert parse_correction_kind("No update line") == "unparsed"


def test_the_instruction_pins() -> None:
    assert "no longer applies" in UPDATE_INSTRUCTION
    assert "adds no evidence" in UPDATE_INSTRUCTION
    assert "even if it is unchanged" in UPDATE_INSTRUCTION
    assert "last thing you write" in UPDATE_INSTRUCTION


def test_the_last_json_block_is_the_one_scored() -> None:
    first_block = json.dumps({"question_type": "binary", "posterior_prob": 0.30})
    final_block = json.dumps({"question_type": "binary", "posterior_prob": 0.42})
    reply: str = f"```json\n{first_block}\n```\n```json\n{final_block}\n```"
    parser: MagicMock = MagicMock()
    parser.model = "openrouter/test/parser"

    scored: Any = asyncio.run(score_reply(_question(), True, reply, parser, model_name="test/model"))

    assert scored.forecast == 0.42
    assert scored.rung == "block"
