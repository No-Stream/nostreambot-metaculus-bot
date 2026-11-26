"""Token and price estimates for one frozen cross-pollination run."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from scripts.probes.bench_common.prompting import count_tokens
from scripts.probes.cross_pollination_bench.manifest import Manifest, QuestionContext
from scripts.probes.cross_pollination_bench.models import BenchModel
from scripts.probes.cross_pollination_bench.schedule import RunConfig, planned_calls


@dataclass(frozen=True)
class RunEstimate:
    n_questions: int
    n_calls: int
    per_model: dict[str, dict[str, float]]
    uncached_usd: float
    cached_usd: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_questions": self.n_questions,
            "n_calls": self.n_calls,
            "per_model": self.per_model,
            "uncached_usd": self.uncached_usd,
            "cached_usd": self.cached_usd,
        }


def estimate_run(
    contexts: list[QuestionContext],
    models: tuple[BenchModel, ...] | list[BenchModel],
    config: RunConfig,
    *,
    output_tokens: int,
    answer_tokens: int,
) -> RunEstimate:
    per_model: dict[str, dict[str, float]] = {}
    total_uncached = total_cached = 0.0
    for model in models:
        uncached = cached = 0.0
        calls = 0
        for context in contexts:
            context_uncached, context_cached, context_calls = _question_cost(
                model, context, config, output_tokens=output_tokens, answer_tokens=answer_tokens
            )
            uncached += context_uncached
            cached += context_cached
            calls += context_calls
        per_model[model.label] = {"calls": calls, "uncached_usd": uncached, "cached_usd": cached}
        total_uncached += uncached
        total_cached += cached
    n_calls = planned_calls(len(contexts), len(models), config)
    return RunEstimate(len(contexts), n_calls, per_model, total_uncached, total_cached)


def _question_cost(
    model: BenchModel,
    context: QuestionContext,
    config: RunConfig,
    *,
    output_tokens: int,
    answer_tokens: int,
) -> tuple[float, float, int]:
    prompt_tokens = count_tokens(model.slug, context.prompt)
    first_calls = config.draws
    update_calls = config.draws * len(config.variants)
    first_input_tokens = prompt_tokens * first_calls
    update_input_tokens = update_calls * (prompt_tokens + answer_tokens + 400)
    update_input_tokens += config.draws * config.variants.count("peers_full") * 2 * answer_tokens
    completion_tokens = (first_calls + update_calls) * output_tokens
    uncached = (first_input_tokens + update_input_tokens) * model.input_usd_per_m
    uncached += completion_tokens * model.output_usd_per_m
    cached = first_input_tokens * model.input_usd_per_m
    cached += (update_input_tokens - update_calls * prompt_tokens) * model.input_usd_per_m
    cached += update_calls * prompt_tokens * model.cache_read_usd_per_m
    if model.cache_write_usd_per_m is not None:
        cached += prompt_tokens * model.cache_write_usd_per_m
    cached += completion_tokens * model.output_usd_per_m
    return uncached / 1_000_000, cached / 1_000_000, first_calls + update_calls


def print_plan(
    contexts: list[QuestionContext], manifest: Manifest, estimate: RunEstimate, *, cap_usd: float, smoke: bool
) -> None:
    counts = Counter(context.entry.qtype for context in contexts)
    after_release = sum(context.entry.after_release for context in contexts)
    events = Counter(context.entry.event_key for context in contexts)
    print(f"Questions: {len(contexts)} ({', '.join(f'{kind}={count}' for kind, count in sorted(counts.items()))})")
    print(f"After release: {after_release}")
    print(f"Frozen manifest: {len(manifest.questions)} questions; mode: {'smoke' if smoke else 'main'}")
    for event, count in sorted(events.items()):
        if count > 1:
            print(f"Event {event}: {count} questions")
    print(f"{estimate.n_calls} calls")
    for model, values in estimate.per_model.items():
        print(
            f"{model}: {int(values['calls'])} calls; uncached ${values['uncached_usd']:.4f}; cached ${values['cached_usd']:.4f}"
        )
    print(f"Estimated uncached total: ${estimate.uncached_usd:.4f}")
    print(f"Estimated cached total: ${estimate.cached_usd:.4f}")
    print(f"Spend cap: ${cap_usd:.2f}")
    print("Salvage parser calls are not priced in this estimate.")
