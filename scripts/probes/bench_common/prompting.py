from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import litellm
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot import prompts as prompts_module
from metaculus_bot.numeric.utils import bound_messages
from metaculus_bot.prompts import binary_prompt, multiple_choice_prompt, numeric_prompt


def render_prompt(question: MetaculusQuestion, research: str) -> str:
    """The production forecaster prompt for this question type over ``research``."""
    if isinstance(question, BinaryQuestion):
        return binary_prompt(question, research)
    if isinstance(question, MultipleChoiceQuestion):
        return multiple_choice_prompt(question, research)
    if isinstance(question, NumericQuestion):
        upper_message, lower_message = bound_messages(question, stated_tail_fields=True)
        return numeric_prompt(question, research, lower_message, upper_message)
    raise TypeError(f"no forecaster prompt for {type(question).__name__}")


def count_tokens(model: str, text: str) -> int:
    """Count tokens with litellm's tokenizer; counts may be approximate for unpublished tokenizers."""
    return litellm.token_counter(model=f"openrouter/{model}", text=text)


@contextmanager
def prompt_clock(window_for: Callable[[MetaculusQuestion], str], today: str) -> Iterator[None]:
    """Temporarily replace the production prompt clock and restore it even when prompt building fails."""
    original_window = prompts_module._forecasting_window_str
    original_today = prompts_module._today_str

    prompts_module._forecasting_window_str = window_for
    prompts_module._today_str = lambda: today
    try:
        yield
    finally:
        prompts_module._forecasting_window_str = original_window
        prompts_module._today_str = original_today
