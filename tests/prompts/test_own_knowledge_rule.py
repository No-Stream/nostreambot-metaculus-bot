"""Recalled and estimated figures are allowed but must carry a label (2026-11-24 hallucination audit)."""

import pytest

from metaculus_bot import prompts
from tests.prompt_builders import _date_prompt_text, _flat, _numeric_q, _open_upper_date_q, _stacked_prompt_texts


def _base_prompts() -> list[str]:
    numeric = prompts.numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
    return [numeric, _date_prompt_text(_open_upper_date_q())]


@pytest.mark.parametrize("index", [0, 1])
def test_base_prompts_carry_the_labelling_rule_once(index: int) -> None:
    text = _flat(_base_prompts()[index])
    assert text.count("(recalled, unverified)") == 1
    assert "never present a recalled or estimated figure as an established fact." in text


def test_stacking_prompts_do_not_carry_the_rule() -> None:
    for text in _stacked_prompt_texts():
        assert "(recalled, unverified)" not in text
