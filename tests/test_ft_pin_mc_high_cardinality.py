"""Pin test for the forecasting-tools 0.2.54 -> 0.2.92 upgrade: high-cardinality MC.

Companion to ``tests/test_ft_upgrade_seams.py`` and ``tests/test_ft_numeric_seams.py``
(Workstream B of the retired design log (git history)). Revised 2026-07-23 for the installed 0.2.92; it defines
"done" for the multiple-choice clamp seam at the upgrade.

The seam: at 0.2.92 upstream's ``PredictedOptionList`` constructor drops its
``abs(sum - 1) < 0.0001`` early return and *unconditionally* clamps every option
into ``[0.01, 0.99]``, renormalizes, and raises ``ValueError`` when any option
moves more than ``0.05``. A high-cardinality ballot (>=6 options) with one
dominant option and several near-floor options is exactly where that raise
fires: the near-floor options each jump to ``0.01``, inflating the sum enough
that renormalization drags the dominant option down by >0.05.

The fix (W4): align ``MC_PROB_MIN``/``MC_PROB_MAX`` to ``[0.01, 0.99]`` and clamp
+ renormalize BEFORE every ``PredictedOptionList`` construction (drift-free, via
``clamp_and_renormalize_probs``), so upstream's constructor is a no-op and never
raises.

Two pins:

Pin 1 (``TestHighCardinalityMcSurvivesRealPipeline`` -- an INVARIANT PIN):
drives the *real* prod MC path -- ``extract_mc`` (deterministic block rung, no
parser-LLM call) -> per-model ``clamp_and_renormalize_mc`` -> ``MEDIAN`` combine
via ``aggregation_strategies`` -- on a 7-option dominant+near-floor ballot and
asserts the resulting ``PredictedOptionList`` respects OUR
``[MC_PROB_MIN, MC_PROB_MAX]`` clamp bounds, with the sub-0.01 near-floor options
floored exactly to ``MC_PROB_MIN``. This proves no raw/unclamped list reaches the
publish boundary; the pre-construction clamp keeps our values ahead of upstream's
stricter constructor.

Pin 2 (``TestRawSubFloorConstructionRaisesOn0292`` -- a DECISION PIN, resolved):
pins that a *raw* ``PredictedOptionList`` built directly from sub-floor (0.002)
options -- the dominant+eight-near-floor shape -- now RAISES ``ValueError`` on
0.2.92 (the 0.002 options clamp to 0.01, inflating the sum so renormalization
drags the 0.984 dominant down by ~0.059), and that our pipeline
(``build_mc_prediction``) clamps that same shape BEFORE construction so it stays
in bounds without raising.
"""

from __future__ import annotations

import asyncio
import json
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOption,
    PredictedOptionList,
)

from metaculus_bot.aggregation_strategies import AggregationStrategy, combine_multiple_choice_predictions
from metaculus_bot.constants import MC_PROB_MAX, MC_PROB_MIN
from metaculus_bot.mc_processing import build_mc_prediction, clamp_and_renormalize_probs
from metaculus_bot.numeric.utils import clamp_and_renormalize_mc
from metaculus_bot.simple_types import OptionProbability
from metaculus_bot.value_extraction import extract_mc

# A 7-option ballot: one dominant option plus six near-floor options. ">=6
# options" is the cardinality the seam calls out. The near-floor share (0.006)
# sits BELOW the aligned MC_PROB_MIN=0.01 floor, so the pre-construction clamp
# floors it to MC_PROB_MIN -- the drift-free repair keeps the floored siblings at
# exactly 0.01 even under the dominant option's renormalization pull.
_DOMINANT = "Alpha"
_NEAR_FLOOR = ["Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf"]
_OPTIONS = [_DOMINANT, *_NEAR_FLOOR]


def _block_rationale(option_probs: dict[str, float]) -> str:
    """A forecaster rationale whose LAST content is a schema-valid MC STRUCTURED FORECAST block.

    Built so ``extract_mc`` resolves on rung 1 (deterministic block parse) with
    zero parser-LLM calls: the block keys map case-insensitively onto ``_OPTIONS``
    and ``option_probs`` sums to 1.0, so ``parse_structured_block`` ->
    ``_make_mc_from_block`` -> ``_validate_mc`` all pass without touching the
    network. This mirrors how production forecasts arrive (a fenced ```json block
    appended to the rationale).
    """
    payload = {"question_type": "multiple_choice", "option_probs": option_probs}
    return "Free-text reasoning about the options.\n\n```json\n" + json.dumps(payload) + "\n```\n"


def _near_floor_ballot(floor_share: float) -> dict[str, float]:
    """Ballot with each near-floor option at ``floor_share``; the dominant option takes the remainder.

    By construction the probabilities sum to exactly 1.0, so the block parser's
    renormalize-to-1.0 step is an identity and the near-floor magnitudes reach
    the clamp untouched.
    """
    dominant = 1.0 - floor_share * len(_NEAR_FLOOR)
    return {_DOMINANT: dominant, **dict.fromkeys(_NEAR_FLOOR, floor_share)}


class TestHighCardinalityMcSurvivesRealPipeline:
    """INVARIANT PIN: a >=6-option near-floor ballot survives extract -> clamp -> aggregate, bounded.

    The whole prod path yields a valid ``PredictedOptionList`` whose every option
    respects OUR ``[MC_PROB_MIN, MC_PROB_MAX]`` bounds, so no raw/unclamped list
    escapes to publish. The pre-construction clamp threads our bounds ahead of
    upstream's stricter constructor, keeping this green on 0.2.92.

    The near-floor share is 0.006, which is now BELOW the aligned ``MC_PROB_MIN``
    (0.01), so the clamp floors those options exactly to ``MC_PROB_MIN``. The
    drift-free repair in ``clamp_and_renormalize_probs`` is what guarantees they
    stay at 0.01 rather than dividing back under it when the dominant option keeps
    the post-clamp total above 1 — the exact renormalize-below-floor drift this
    pipeline used to have.
    """

    @pytest.fixture
    def parser_llm(self) -> MagicMock:
        # Rung 1 (block) must satisfy every extraction with zero parser calls. If
        # the ladder ever fell through to the LLM rung, parse_structured would
        # invoke this and the side_effect would surface it -- but the load-bearing
        # proof is the ``outcome.rung == "block"`` assertion below.
        llm = MagicMock()
        llm.invoke.side_effect = AssertionError("parser LLM must not be called; the block rung must win")
        return llm

    def _extract_and_clamp(self, option_probs: dict[str, float], parser_llm: MagicMock) -> PredictedOptionList:
        """Run the per-model prod path: extract_mc (block rung) -> clamp_and_renormalize_mc."""
        outcome = asyncio.run(
            extract_mc(
                _block_rationale(option_probs),
                list(_OPTIONS),
                parser_llm,
                question_id=4242,
                model_name="pin-model",
            )
        )
        assert outcome.rung == "block", f"expected the deterministic block rung, got {outcome.rung!r}"
        assert outcome.block_present is True
        return clamp_and_renormalize_mc(outcome.value.option_list)

    def test_per_model_extract_and_clamp_is_bounded(self, parser_llm: MagicMock) -> None:
        """One model's extract+clamp keeps all options in [MC_PROB_MIN, MC_PROB_MAX] for the near-floor ballot."""
        pol = self._extract_and_clamp(_near_floor_ballot(0.006), parser_llm)

        assert {o.option_name for o in pol.predicted_options} == set(_OPTIONS)
        assert sum(o.probability for o in pol.predicted_options) == pytest.approx(1.0)
        for option in pol.predicted_options:
            assert MC_PROB_MIN <= option.probability <= MC_PROB_MAX, (
                f"option {option.option_name!r} probability {option.probability} outside [{MC_PROB_MIN}, {MC_PROB_MAX}]"
            )

        # Non-vacuity: the near-floor options were sub-0.01 on input (0.006) and are
        # floored exactly to MC_PROB_MIN (0.01) -- so the clamp bites, the seam is
        # exercised, not dodged by accidentally comfortable probabilities.
        near_floor_probs = [o.probability for o in pol.predicted_options if o.option_name in _NEAR_FLOOR]
        assert near_floor_probs, "expected the near-floor options to be present"
        assert all(p == pytest.approx(MC_PROB_MIN) for p in near_floor_probs), (
            f"sub-0.01 near-floor options must floor to MC_PROB_MIN (0.01); got {near_floor_probs}"
        )

    def test_full_path_extract_clamp_median_is_bounded(self, parser_llm: MagicMock) -> None:
        """Full prod MC path across three models: extract -> per-model clamp -> MEDIAN combine, all bounded."""
        per_model = [self._extract_and_clamp(_near_floor_ballot(share), parser_llm) for share in (0.006, 0.007, 0.005)]

        aggregated = combine_multiple_choice_predictions(per_model, AggregationStrategy.MEDIAN)

        assert {o.option_name for o in aggregated.predicted_options} == set(_OPTIONS)
        assert sum(o.probability for o in aggregated.predicted_options) == pytest.approx(1.0)
        for option in aggregated.predicted_options:
            assert MC_PROB_MIN <= option.probability <= MC_PROB_MAX, (
                f"aggregated option {option.option_name!r} probability {option.probability} outside "
                f"[{MC_PROB_MIN}, {MC_PROB_MAX}]"
            )

        dominant_prob = next(o.probability for o in aggregated.predicted_options if o.option_name == _DOMINANT)
        near_floor_probs = [o.probability for o in aggregated.predicted_options if o.option_name in _NEAR_FLOOR]
        # The aggregate is still a dominant-plus-near-floor shape (not smeared into
        # a uniform), so the pin covers the shape the 0.2.92 raise targets.
        assert dominant_prob > 0.9, f"dominant option should remain dominant after MEDIAN combine; got {dominant_prob}"
        assert all(p == pytest.approx(MC_PROB_MIN) for p in near_floor_probs), (
            f"aggregated sub-0.01 near-floor options must floor to MC_PROB_MIN (0.01); got {near_floor_probs}"
        )


class TestRawSubFloorConstructionRaisesOn0292:
    """DECISION PIN (resolved 2026-07-23 for ft 0.2.92): raw sub-floor construction RAISES; our pipeline clamps first.

    # DECISION PIN: 0.2.54's PredictedOptionList constructor returned sub-floor
    # options verbatim (abs(sum - 1) < 0.0001 early return). 0.2.92 drops that:
    # it unconditionally clamps every option into [0.01, 0.99], renormalizes, and
    # raises ValueError when any option moves > 0.05. For the exact shape here
    # (one dominant 0.984 option + eight near-floor 0.002 options), the eight clamp
    # 0.002 -> 0.01, inflating the sum to 1.064, which drags the dominant
    # 0.984 -> ~0.925 (a ~0.059 move) and fires the raise. The W4 fix aligns
    # MC_PROB_MIN/MAX to [0.01, 0.99] and clamps BEFORE construction, so our
    # pipeline (build_mc_prediction) handles this shape without ever hitting the raise.
    """

    _DOMINANT = 0.984
    _SUB_FLOOR = 0.002
    # 8 near-floor options -> the >=6-option raise regime
    _NEAR_FLOOR_NAMES: ClassVar[list[str]] = [f"opt{i}" for i in range(8)]

    def test_raw_sub_floor_construction_raises(self) -> None:
        """A raw dominant + eight-sub-floor ballot now RAISES on 0.2.92 construction (the >0.05-move guard)."""
        with pytest.raises(ValueError, match="differs from original probability"):
            PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="Dominant", probability=self._DOMINANT),
                    *[
                        PredictedOption(option_name=name, probability=self._SUB_FLOOR)
                        for name in self._NEAR_FLOOR_NAMES
                    ],
                ]
            )

    def test_pipeline_clamps_sub_floor_shape_before_construction(self) -> None:
        """build_mc_prediction clamps the same shape BEFORE construction, so it lands in bounds without raising."""
        options = ["Dominant", *self._NEAR_FLOOR_NAMES]
        raw = [
            OptionProbability(option_name="Dominant", probability=self._DOMINANT),
            *[OptionProbability(option_name=name, probability=self._SUB_FLOOR) for name in self._NEAR_FLOOR_NAMES],
        ]

        pol = build_mc_prediction(raw, options)  # must not raise

        assert {o.option_name for o in pol.predicted_options} == set(options)
        assert sum(o.probability for o in pol.predicted_options) == pytest.approx(1.0)
        for option in pol.predicted_options:
            assert MC_PROB_MIN <= option.probability <= MC_PROB_MAX, (
                f"option {option.option_name!r} probability {option.probability} outside [{MC_PROB_MIN}, {MC_PROB_MAX}]"
            )
        # The eight sub-floor options are floored to MC_PROB_MIN; the dominant stays dominant.
        near_floor_probs = [o.probability for o in pol.predicted_options if o.option_name in self._NEAR_FLOOR_NAMES]
        assert all(p == pytest.approx(MC_PROB_MIN) for p in near_floor_probs), (
            f"sub-floor options must floor to MC_PROB_MIN (0.01); got {near_floor_probs}"
        )
        dominant_prob = next(o.probability for o in pol.predicted_options if o.option_name == "Dominant")
        assert dominant_prob > 0.9, f"dominant option should stay dominant; got {dominant_prob}"


class TestFloorFeasibilityBoundary:
    """Pin where ``clamp_and_renormalize_probs``'s in-bounds guarantee actually stops.

    Both its docstring and ``clamp_and_renormalize_mc``'s used to promise in-bounds sum-1
    output unconditionally. That is arithmetically impossible past ``n * MC_PROB_MIN >=
    1.0``: 101 options each at least 0.01 already exceed 1. These pin the real contract so
    a future reader trusts the boundary rather than the old blanket claim, and so a change
    to MC_PROB_MIN or to the degenerate branch surfaces here.

    None of this is reachable on today's Metaculus ballots (option counts are far below
    100); it is documented because the docstrings are what a caller reads.
    """

    def test_guarantee_holds_at_the_feasibility_limit(self) -> None:
        # n=100 at a 0.01 floor is exactly feasible (100 * 0.01 == 1.0 is the degenerate
        # branch's trigger, but a uniform ballot lands exactly ON the floor), so the
        # boundary for uniform input is n > 100, not n >= 100.
        out = clamp_and_renormalize_probs([1 / 100] * 100)
        assert sum(out) == pytest.approx(1.0)
        for probability in out:
            assert MC_PROB_MIN <= probability <= MC_PROB_MAX

    def test_beyond_the_limit_returns_sub_floor_values_by_necessity(self) -> None:
        # 200 uniform options: no in-bounds sum-1 solution exists, so the honest output is
        # sub-floor rather than a sum that isn't 1. Pin the value so the contract is
        # concrete.
        out = clamp_and_renormalize_probs([1 / 200] * 200)
        assert sum(out) == pytest.approx(1.0)
        assert all(probability == pytest.approx(0.005) for probability in out)
        assert min(out) < MC_PROB_MIN, "past the feasibility limit the floor cannot be honored"

    def test_uniform_high_cardinality_still_constructs(self) -> None:
        # ft clamps 0.005 -> 0.01 then renormalizes straight back to 0.005: a zero net
        # move, so its >0.05 guard does not fire. This is why sub-floor output alone is
        # not a publish-time failure.
        out = clamp_and_renormalize_probs([1 / 200] * 200)
        pol = PredictedOptionList(
            predicted_options=[PredictedOption(option_name=f"o{i}", probability=p) for i, p in enumerate(out)]
        )
        assert len(pol.predicted_options) == 200

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param([0.9] + [0.1 / 199] * 199, id="200-options-one-dominant"),
            pytest.param([0.5, 0.3] + [0.2 / 118] * 118, id="120-options-two-dominant"),
        ],
    )
    def test_high_cardinality_plus_concentration_is_what_ft_rejects(self, raw: list[float]) -> None:
        """The publish-time loss needs BOTH high cardinality and a dominant option.

        Documented as a known, unreachable-in-practice limit rather than fixed, because it
        cannot be fixed from our side: ft clamps to its own hard-coded 0.01 no matter what
        floor we use, so relaxing ours changes nothing about how far it moves the dominant
        option (measured — at n=200 ft accepts a top option only up to ~0.06, and no choice
        of our floor lifts that ceiling). Any real fix would have to cap cardinality or
        reshape the forecast, neither of which belongs in a clamp helper.
        """
        out = clamp_and_renormalize_probs(raw)
        assert sum(out) == pytest.approx(1.0), "we still return a sum-1 list"
        with pytest.raises(ValueError, match="differs from original probability"):
            PredictedOptionList(
                predicted_options=[PredictedOption(option_name=f"o{i}", probability=p) for i, p in enumerate(out)]
            )
