"""Optional explicit tails survive the same candidate selection as the percentiles."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.structured_output_schema import DateStructured, NumericStructured, parse_structured_payload
from metaculus_bot.value_extraction import TailProbabilities, TailProbability, extract_date, extract_numeric


def _payload(question_type: str, **tails: object) -> dict[str, object]:
    values = (
        [
            (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)).isoformat()
            for index in range(len(STANDARD_PERCENTILES))
        ]
        if question_type == "date"
        else [float(index + 1) for index in range(len(STANDARD_PERCENTILES))]
    )
    return {
        "question_type": question_type,
        "declared_percentiles": dict(zip(STANDARD_PERCENTILES, values, strict=True)),
        **tails,
    }


def _rationale(payload: dict[str, object], *, trailing_comma: bool = False) -> str:
    encoded = json.dumps(payload)
    if trailing_comma:
        encoded = encoded[:-1] + ",}"
    return f"Reasoning.\n```json\n{encoded}\n```"


@pytest.mark.parametrize("question_type", ["numeric", "date"])
class TestOptionalTailSchema:
    def test_accepts_explicit_probabilities(self, question_type: str) -> None:
        block = parse_structured_payload(
            json.dumps(_payload(question_type, prob_below_lower_bound=0.0, prob_above_upper_bound=0.2)),
            question_type,  # type: ignore[arg-type]
        )
        assert isinstance(block, (NumericStructured, DateStructured))
        assert block.prob_below_lower_bound == 0.0
        assert block.prob_above_upper_bound == 0.2

    @pytest.mark.parametrize("value", [None, True, "unknown", [], {}, -0.1, 1.0, float("inf"), float("nan")])
    def test_unusable_tail_does_not_discard_percentiles(self, question_type: str, value: object) -> None:
        block = parse_structured_payload(
            json.dumps(_payload(question_type, prob_below_lower_bound=value)),
            question_type,  # type: ignore[arg-type]
        )
        assert isinstance(block, (NumericStructured, DateStructured))
        assert block.declared_percentiles is not None
        assert len(block.declared_percentiles) == len(STANDARD_PERCENTILES)


@pytest.mark.parametrize("question_type", ["numeric", "date"])
@pytest.mark.asyncio
class TestTailExtraction:
    async def test_valid_and_missing_sides_are_separate(self, question_type: str) -> None:
        extract = extract_numeric if question_type == "numeric" else extract_date
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as salvage:
            outcome = await extract(_rationale(_payload(question_type, prob_above_upper_bound=0.2)), MagicMock())
        assert outcome.rung == "block"
        assert outcome.tail_probabilities == TailProbabilities(
            below=TailProbability(value=None, status="missing"), above=TailProbability(value=0.2, status="valid")
        )
        assert len(outcome.value) == len(STANDARD_PERCENTILES)
        salvage.assert_not_awaited()

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, TailProbability(value=None, status="invalid")),
            (True, TailProbability(value=None, status="invalid")),
            ("0.2", TailProbability(value=0.2, status="valid")),
            ("unknown", TailProbability(value=None, status="invalid")),
            ([], TailProbability(value=None, status="invalid")),
            ({}, TailProbability(value=None, status="invalid")),
        ],
    )
    async def test_tail_value_preserves_other_side(
        self, question_type: str, value: object, expected: TailProbability
    ) -> None:
        extract = extract_numeric if question_type == "numeric" else extract_date
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as salvage:
            outcome = await extract(
                _rationale(_payload(question_type, prob_below_lower_bound=value, prob_above_upper_bound=0.2)),
                MagicMock(),
            )
        assert outcome.rung == "block"
        assert outcome.tail_probabilities == TailProbabilities(
            below=expected, above=TailProbability(value=0.2, status="valid")
        )
        salvage.assert_not_awaited()

    @pytest.mark.parametrize("value", [-0.1, 1.0, 1.2, float("inf"), float("nan")])
    async def test_out_of_range_tail_keeps_stated_number(self, question_type: str, value: float) -> None:
        extract = extract_numeric if question_type == "numeric" else extract_date
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as salvage:
            outcome = await extract(_rationale(_payload(question_type, prob_below_lower_bound=value)), MagicMock())
        assert outcome.rung == "block"
        tails = outcome.tail_probabilities
        assert tails is not None
        assert tails.below.status == "invalid"
        if value != value:
            assert tails.below.value != tails.below.value
        else:
            assert tails.below.value == value
        assert tails.above == TailProbability(value=None, status="missing")
        salvage.assert_not_awaited()

    @pytest.mark.parametrize(("below", "above"), [(0.4, 0.6), (0.8, 0.4)])
    async def test_no_interior_mass_invalidates_both_tails(
        self, question_type: str, below: float, above: float
    ) -> None:
        extract = extract_numeric if question_type == "numeric" else extract_date
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as salvage:
            outcome = await extract(
                _rationale(_payload(question_type, prob_below_lower_bound=below, prob_above_upper_bound=above)),
                MagicMock(),
            )
        assert outcome.tail_probabilities == TailProbabilities(
            below=TailProbability(value=below, status="invalid"), above=TailProbability(value=above, status="invalid")
        )
        salvage.assert_not_awaited()

    async def test_zero_is_valid_and_sum_below_one_is_valid(self, question_type: str) -> None:
        extract = extract_numeric if question_type == "numeric" else extract_date
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as salvage:
            outcome = await extract(
                _rationale(_payload(question_type, prob_below_lower_bound=0, prob_above_upper_bound=0.999)),
                MagicMock(),
            )
        assert outcome.tail_probabilities == TailProbabilities(
            below=TailProbability(value=0.0, status="valid"), above=TailProbability(value=0.999, status="valid")
        )
        salvage.assert_not_awaited()

    async def test_repaired_final_candidate_supplies_percentiles_and_tails(self, question_type: str) -> None:
        extract = extract_numeric if question_type == "numeric" else extract_date
        draft = _rationale(_payload(question_type, prob_below_lower_bound=0.1))
        final = _rationale(_payload(question_type, prob_below_lower_bound=0.3), trailing_comma=True)
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as salvage:
            outcome = await extract(draft + final, MagicMock())
        assert outcome.rung == "repair"
        assert outcome.tail_probabilities is not None
        assert outcome.tail_probabilities.below == TailProbability(value=0.3, status="valid")
        salvage.assert_not_awaited()

    @pytest.mark.parametrize(("open_lower", "open_upper"), [(False, True), (True, False), (False, False)])
    async def test_closed_side_cannot_invalidate_open_side(
        self, question_type: str, open_lower: bool, open_upper: bool
    ) -> None:
        extract = extract_numeric if question_type == "numeric" else extract_date
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as salvage:
            outcome = await extract(
                _rationale(_payload(question_type, prob_below_lower_bound=0.6, prob_above_upper_bound=0.6)),
                MagicMock(),
                open_lower_bound=open_lower,
                open_upper_bound=open_upper,
            )
        assert outcome.tail_probabilities == TailProbabilities(
            below=TailProbability(value=0.6, status="valid"), above=TailProbability(value=0.6, status="valid")
        )
        salvage.assert_not_awaited()


@pytest.mark.asyncio
async def test_salvage_does_not_reuse_tails_from_rejected_block() -> None:
    rejected = _payload("numeric", prob_below_lower_bound=0.2)
    rejected["declared_percentiles"] = {0.1: 1.0, 0.5: 2.0, 0.9: 3.0}
    percentiles = [Percentile(percentile=p, value=float(index)) for index, p in enumerate(STANDARD_PERCENTILES)]
    with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(return_value=percentiles)) as salvage:
        outcome = await extract_numeric(_rationale(rejected), MagicMock())
    assert outcome.rung == "llm"
    assert outcome.tail_probabilities == TailProbabilities(
        below=TailProbability(value=None, status="missing"), above=TailProbability(value=None, status="missing")
    )
    salvage.assert_awaited_once()
