"""Explicit outside probabilities belong to each open side of base percentile forecasts."""

import json
from typing import Literal

import pytest

from metaculus_bot import prompts
from metaculus_bot.constants import METACULUS_OUT_OF_RANGE_TAIL_FLOOR
from tests.prompt_builders import (
    MANTIC_PAGE_URL,
    METACULUS_PAGE_URL,
    _date_prompt_text,
    _extract_last_json_block,
    _flat,
    _numeric_q,
    _open_upper_date_q,
    _pmf_prompt_text,
    _stacked_prompt_texts,
)


def _percentile_prompt(kind: Literal["numeric", "date"], *, page_url: str, open_lower: bool, open_upper: bool) -> str:
    if kind == "date":
        question = _open_upper_date_q()
    else:
        question = _numeric_q()
        question.nominal_lower_bound = 0.0
        question.nominal_upper_bound = 1000.0
    question.page_url = page_url
    question.open_lower_bound = open_lower
    question.open_upper_bound = open_upper
    if kind == "date":
        return _date_prompt_text(question)
    return prompts.numeric_prompt(question, research="r", lower_bound_message="lbm", upper_bound_message="ubm")


class TestOpenBoundProbabilityFields:
    @pytest.mark.parametrize("kind", ["numeric", "date"])
    @pytest.mark.parametrize("page_url", [METACULUS_PAGE_URL, MANTIC_PAGE_URL], ids=["metaculus", "mantic"])
    @pytest.mark.parametrize(("open_lower", "open_upper"), [(False, False), (True, False), (False, True), (True, True)])
    def test_schema_requires_only_the_open_sides(
        self, kind: Literal["numeric", "date"], page_url: str, open_lower: bool, open_upper: bool
    ) -> None:
        prompt = _percentile_prompt(kind, page_url=page_url, open_lower=open_lower, open_upper=open_upper)
        schema = json.loads(_extract_last_json_block(prompt))
        outside_probabilities = []
        for field, is_open in (
            ("prob_below_lower_bound", open_lower),
            ("prob_above_upper_bound", open_upper),
        ):
            assert (field in schema) is is_open
            assert (field in prompt) is is_open
            if is_open:
                assert 0 <= schema[field] < 1
                outside_probabilities.append(schema[field])
        assert sum(outside_probabilities) < 1

    @pytest.mark.parametrize(
        ("kind", "lower_bound", "upper_bound"),
        [
            ("numeric", "0.0", "1000.0"),
            ("date", "2026-09-01T00:00:00Z", "2027-03-01T00:00:00Z"),
        ],
    )
    def test_one_named_rule_defines_probabilities_and_their_relationship_to_percentiles(
        self, kind: Literal["numeric", "date"], lower_bound: str, upper_bound: str
    ) -> None:
        prompt = _percentile_prompt(kind, page_url=METACULUS_PAGE_URL, open_lower=True, open_upper=True)
        rule = prompts._OPEN_BOUND_PROBABILITY_RULE.format(
            fields=f"`prob_below_lower_bound` = P(outcome < {lower_bound}); "
            f"`prob_above_upper_bound` = P(outcome > {upper_bound})",
            consistency_examples="if `prob_below_lower_bound` is 0.04, your 0.01 and 0.025 percentiles must be values "
            "below the lower bound; if `prob_above_upper_bound` is 0.04, your 0.975 and 0.99 percentiles must be values "
            "above the upper bound",
        )
        assert _flat(prompt).count(_flat(rule)) == 1
        flat = _flat(rule)
        assert "required" in flat
        assert "decimal probabilities in [0, 1)" in flat
        assert "sum below 1" in flat
        assert "consistent with `declared_percentiles`" in flat
        assert "honest" in flat
        assert "reason explicitly about how likely the outcome is to fall beyond each specified bound" in flat
        assert "these values set the published probability beyond those bounds" in flat

    @pytest.mark.parametrize("granularity", ["", "day", "week"], ids=["timestamp", "day", "week"])
    def test_date_rule_names_actual_bounds_in_the_questions_date_format(self, granularity: str) -> None:
        question = _open_upper_date_q()
        question.open_lower_bound = True
        question.api_json["question"]["date_granularity"] = granularity
        prompt = _date_prompt_text(question)
        rule = prompt[prompt.index("Open-bound probabilities:") :]
        suffix = "" if granularity else "T00:00:00Z"
        assert f"`prob_below_lower_bound` = P(outcome < 2026-09-01{suffix})" in rule
        assert f"`prob_above_upper_bound` = P(outcome > 2027-03-01{suffix})" in rule

    @pytest.mark.parametrize("kind", ["numeric", "date"])
    def test_closed_bounds_do_not_receive_the_rule(self, kind: Literal["numeric", "date"]) -> None:
        prompt = _percentile_prompt(kind, page_url=MANTIC_PAGE_URL, open_lower=False, open_upper=False)
        assert "Open-bound probabilities:" not in prompt

    def test_per_bin_and_all_stacking_prompts_omit_the_rule_and_fields(self) -> None:
        for prompt in [_pmf_prompt_text(), *_stacked_prompt_texts()]:
            assert "Open-bound probabilities:" not in prompt
            assert "prob_below_lower_bound" not in prompt
            assert "prob_above_upper_bound" not in prompt

    @pytest.mark.parametrize("kind", ["numeric", "date"])
    def test_mantic_no_longer_claims_in_range_percentiles_assert_one_percent(
        self, kind: Literal["numeric", "date"]
    ) -> None:
        prompt = _percentile_prompt(kind, page_url=MANTIC_PAGE_URL, open_lower=True, open_upper=True)
        flat = _flat(prompt)
        assert "asserts a 1% chance of an out-of-range outcome" not in flat
        assert "keeping every percentile inside" not in flat


def test_example_tail_values_do_not_anchor_below_the_metaculus_floor() -> None:
    """The schema example's number anchors what models state; 0.005 sat under the structural 1%."""
    from_prompt = json.loads(
        _extract_last_json_block(
            _percentile_prompt("numeric", page_url=METACULUS_PAGE_URL, open_lower=True, open_upper=True)
        )
    )
    for field in ("prob_below_lower_bound", "prob_above_upper_bound"):
        assert from_prompt[field] >= METACULUS_OUT_OF_RANGE_TAIL_FLOOR


@pytest.mark.parametrize("kind", ["numeric", "date"])
def test_tails_come_first_and_percentiles_must_agree_with_them(kind: Literal["numeric", "date"]) -> None:
    """The model states each open tail before its percentiles, so the tail drives the percentiles."""
    text = _percentile_prompt(kind, page_url=METACULUS_PAGE_URL, open_lower=True, open_upper=True)
    example = _extract_last_json_block(text)
    assert list(json.loads(example))[:3] == ["question_type", "prob_below_lower_bound", "prob_above_upper_bound"]
    flattened = _flat(text)
    assert "decide these probabilities before your percentiles" in flattened
    assert (
        "if `prob_below_lower_bound` is 0.04, your 0.01 and 0.025 percentiles must be values below the lower bound"
        in flattened
    )
