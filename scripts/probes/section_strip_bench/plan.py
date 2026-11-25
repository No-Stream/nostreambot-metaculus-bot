"""The call plan and its price: every (question, arm, replicate) prompt, rendered before a cent is spent.

Prompts are the production builders' output with the clock anchored to the archived run date, so the
estimate reads the real token counts of the real prompts, and the dry run shows exactly what a paid run
would send.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import litellm
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot import prompts as prompts_module
from metaculus_bot.numeric.utils import bound_messages
from metaculus_bot.prompts import binary_prompt, multiple_choice_prompt, numeric_prompt
from scripts.probes.section_strip_bench.bundle import ARMS, FULL_ARM, BenchQuestion


@contextmanager
def anchored_clock(forecasting_window: str, today: str) -> Iterator[None]:
    """Render prompts as of the archived run date: the recorded window block verbatim, and its ``Today``.

    The prompt module reads ``datetime.now`` through two functions; both are swapped for the duration of a
    synchronous prompt build and restored on exit. Never hold this across an ``await``.
    """
    original_window = prompts_module._forecasting_window_str
    original_today = prompts_module._today_str

    def _recorded_window(question: MetaculusQuestion) -> str:
        del question
        return forecasting_window

    def _recorded_today() -> str:
        return today

    prompts_module._forecasting_window_str = _recorded_window
    prompts_module._today_str = _recorded_today
    try:
        yield
    finally:
        prompts_module._forecasting_window_str = original_window
        prompts_module._today_str = original_today


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


def replicate_nonce(seed: int) -> str:
    """The one line that tells replicate ``seed`` apart, since the model takes no seed parameter."""
    return f"\n\nReplicate {seed}."


@dataclass(frozen=True)
class PlanItem:
    question: BenchQuestion
    arm: str
    seed: int
    prompt: str
    prompt_tokens: int


def count_tokens(model: str, text: str) -> int:
    """tiktoken through litellm's counter; Muse Spark's own tokenizer is unpublished, so treat this as approximate."""
    return litellm.token_counter(model=f"openrouter/{model}", text=text)


def build_plan(questions: Sequence[BenchQuestion], arms: Sequence[str], seeds: int, *, model: str) -> list[PlanItem]:
    """Every (question, arm, replicate) call with its prompt already rendered, so the estimate reads real sizes."""
    plan: list[PlanItem] = []
    for bench_question in questions:
        with anchored_clock(bench_question.forecasting_window, bench_question.today):
            base_prompts = {arm: render_prompt(bench_question.question, bench_question.arms[arm]) for arm in arms}
        for arm, base in base_prompts.items():
            for seed in range(1, seeds + 1):
                prompt = base + replicate_nonce(seed)
                plan.append(PlanItem(bench_question, arm, seed, prompt, count_tokens(model, prompt)))
    return plan


@dataclass(frozen=True)
class Estimate:
    n_questions: int
    n_arms: int
    n_seeds: int
    n_calls: int
    prompt_tokens_total: int
    median_prompt_tokens: int
    output_tokens_per_call: int
    price_in_usd_per_m: float
    price_out_usd_per_m: float

    @property
    def prompt_usd(self) -> float:
        return self.prompt_tokens_total * self.price_in_usd_per_m / 1e6

    @property
    def completion_usd(self) -> float:
        return self.n_calls * self.output_tokens_per_call * self.price_out_usd_per_m / 1e6

    @property
    def total_usd(self) -> float:
        return self.prompt_usd + self.completion_usd

    @property
    def usd_per_call(self) -> float:
        return self.total_usd / self.n_calls if self.n_calls else 0.0

    def calls_under(self, cap_usd: float) -> int:
        return int(cap_usd / self.usd_per_call) if self.usd_per_call > 0 else 0

    def seeds_under(self, cap_usd: float) -> int:
        """How many replicates of the whole question-by-arm set the cap pays for at this per-call price."""
        per_seed = self.n_questions * self.n_arms
        return self.calls_under(cap_usd) // per_seed if per_seed else 0

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "prompt_usd": self.prompt_usd,
            "completion_usd": self.completion_usd,
            "total_usd": self.total_usd,
        }


def estimate_spend(plan: Sequence[PlanItem], *, output_tokens: int, price_in: float, price_out: float) -> Estimate:
    token_counts = [item.prompt_tokens for item in plan]
    return Estimate(
        n_questions=len({item.question.question_id for item in plan}),
        n_arms=len({item.arm for item in plan}),
        n_seeds=len({item.seed for item in plan}),
        n_calls=len(plan),
        prompt_tokens_total=sum(token_counts),
        median_prompt_tokens=int(statistics.median(token_counts)) if token_counts else 0,
        output_tokens_per_call=output_tokens,
        price_in_usd_per_m=price_in,
        price_out_usd_per_m=price_out,
    )


def print_plan(plan: Sequence[PlanItem], estimate: Estimate, *, model: str, cap_usd: float) -> None:
    """The plan and the estimate, printed before any decision to spend (and as the whole of a dry run)."""
    first_replicates = [item for item in plan if item.seed == 1]
    by_type = Counter(item.question.qtype for item in first_replicates if item.arm == FULL_ARM)
    arms = sorted({item.arm for item in plan}, key=ARMS.index)
    print(f"Plan: {estimate.n_questions} resolved questions ({dict(sorted(by_type.items()))})")
    print(f"  arms: {arms}; replicates: {estimate.n_seeds}")
    print(f"  calls: {estimate.n_calls}; model: {model}; replicate nonce: one appended line, no seed parameter")
    for arm in arms:
        tokens = sorted(item.prompt_tokens for item in first_replicates if item.arm == arm)
        print(
            f"  {arm:<10} median prompt tokens {int(statistics.median(tokens)):>7,} (min {tokens[0]:,}, max {tokens[-1]:,})"
        )
    print("Estimated cost of this run (tiktoken count via litellm.token_counter; the model's tokenizer is unpublished)")
    print(
        f"  prompt: {estimate.prompt_tokens_total:,} tokens at ${estimate.price_in_usd_per_m}/M = ${estimate.prompt_usd:.2f}"
    )
    print(
        f"  completion: {estimate.n_calls} x {estimate.output_tokens_per_call} tokens at "
        f"${estimate.price_out_usd_per_m}/M = ${estimate.completion_usd:.2f} "
        f"(reasoning tokens bill as completion; at 5x the assumption this line is ${5 * estimate.completion_usd:.2f})"
    )
    print(
        "  not priced here: a reply whose fenced block fails to parse triggers one salvage call to the parser model,"
        " billed to the same key and counted against the cap"
    )
    print(f"  total: ${estimate.total_usd:.2f}; cap ${cap_usd:.2f} fits {estimate.calls_under(cap_usd)} calls,")
    print(
        f"  which is {estimate.seeds_under(cap_usd)} replicates of the full {estimate.n_questions} x {estimate.n_arms} set"
    )
    print()
