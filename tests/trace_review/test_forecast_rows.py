"""Out-of-range detection in the trace index's outcome columns."""

from __future__ import annotations

import pytest

from metaculus_bot.trace_review.forecast_rows import _out_of_range_side, _prob_on_outcome

PUBLISHED_CDF: list[float] = [0.013, 0.5, 0.974]


def _numeric_record(resolution: object) -> dict:
    return {
        "type": "numeric",
        "scaling": {"range_min": 110.0, "range_max": 125.0},
        "resolution_parsed": resolution,
        "our_forecast_values": PUBLISHED_CDF,
    }


class TestOutOfRangeSide:
    @pytest.mark.parametrize(
        ("resolution", "side"),
        [
            ("below_lower_bound", "below_lower_bound"),
            ("above_upper_bound", "above_upper_bound"),
            # Metaculus can send the value itself past an open bound (q46143: 109.9M against 110M).
            (109.9, "below_lower_bound"),
            (125.5, "above_upper_bound"),
            (110.0, None),
            (117.0, None),
        ],
    )
    def test_side_from_string_or_value(self, resolution: object, side: str | None) -> None:
        assert _out_of_range_side(_numeric_record(resolution)) == side

    def test_value_past_the_bound_gets_the_tail_probability(self) -> None:
        assert _prob_on_outcome(_numeric_record(109.9)) == pytest.approx(0.013)
        assert _prob_on_outcome(_numeric_record(125.5)) == pytest.approx(1.0 - 0.974)
        assert _prob_on_outcome(_numeric_record(117.0)) is None
