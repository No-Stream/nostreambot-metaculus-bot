"""Per-arm stacker runner for the probabilistic-tools ablation benchmark.

Reads the canonical ``(qid, model_slug)`` forecaster cache entries written by
``metaculus_bot.ablation.forecasters``, runs the tool-runner (per-rationale "Computed quantities"
plus cross-model aggregation), then dispatches to ``metaculus_bot.stacking.run_stacking_*``. The
two arms differ only by the ``PROBABILISTIC_TOOLS_ENABLED`` env-var state when the tool-runner
functions are called: ARM_STACK leaves it unset so both runners early-return ``""``; ARM_STACK_AUG
sets it to ``"1"`` so both produce real markdown for the stacker prompt. Results are cached per
``(qid, arm)``; on primary-stacker failure the runner falls back to a secondary stacker LLM, and
when both fail it caches a ``success=False`` payload so the batch wrapper continues.

Stacker choice. The default primary is ``openrouter/anthropic/claude-opus-4.5`` rather than prod's
opus-5.5 (``STACKER_LLM`` in ``llm_configs.py``, opus-4.8 -> opus-5.5 on the 2026-09-22 GPT-6/opus-5.5
migration; the slot has been Anthropic since 2026-07-20, when fable-5 left both roles), because a
gpt-5.5 primary 404ed ("no endpoints available", a data-policy guardrail) on every request from the
operator's local donated key while the GitHub-secret key worked; Anthropic models clear it. The
``gpt-6-sol`` fallback matches prod ``STACKER_FALLBACK_LLM`` (gpt-5.6-sol -> gpt-6-sol on the
2026-09-22 migration) and sits on a different provider so an Anthropic stall cannot take both
attempts down. Both go through ``build_llm_with_openrouter_fallback`` so the Metaculus-donated
OpenRouter key absorbs cost ahead of the operator's paid key; that wrapper handles the
donated-to-paid fallback on credit/auth/data-policy errors itself, so the outer
primary-to-fallback chain in ``run_stacker_for_arm`` is a defense-in-depth backstop, not the
cost control.

Cost, order of magnitude: a frontier stacker at ``reasoning={"effort": "high"}`` runs about
$0.05-0.10 per call, so a 20-question sweep (40 calls) is $2-4 and a 60-question sweep (120
calls) $6-12 worst case; the donated key usually absorbs almost all of it.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any, cast

from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

from metaculus_bot import stacking, tool_runner
from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.env import FEATURE_FLAG_ENV, probabilistic_tools_enabled
from metaculus_bot.ablation.forecasters import (
    EXPECTED_LLM_CALL_FAILURES,
    deserialize_prediction_value,
    question_type_for_serialization,
    serialize_prediction_value,
)
from metaculus_bot.ablation.stage_payload import make_error_payload, make_success_payload
from metaculus_bot.ablation.window_patch import patched_window_for_question
from metaculus_bot.aggregation_strategies import (
    AggregationStrategy,
    combine_binary_predictions,
    combine_multiple_choice_predictions,
    combine_numeric_predictions,
)
from metaculus_bot.constants import STACKER_FALLBACK_SOFT_DEADLINE, STACKER_SOFT_DEADLINE
from metaculus_bot.exceptions import UnitMismatchError
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.numeric.utils import bound_messages
from metaculus_bot.numeric.validation import detect_unit_mismatch

logger: logging.Logger = logging.getLogger(__name__)

ARM_STACK = "stack"  # LLM stacker, rationale only, no probability-math tools
ARM_STACK_AUG = "stack_aug"  # LLM stacker, rationale + computed quantities + cross-model aggregation (augmented)
ARM_PDF = "pdf"  # deterministic structured-math aggregation, no LLM (see metaculus_bot.ablation.run_pdf)
ARM_PDF_MIN1 = "pdf_min1"  # pdf arm with min_forecasters=1 (any structured output qualifies)
ARM_PDF_MIN2 = "pdf_min2"  # pdf arm with min_forecasters=2 (proper aggregation)
ARM_MEDIAN = "median"  # deterministic median over base predictions, no LLM (see metaculus_bot.ablation.run_simple_agg)
ARM_MEAN = "mean"  # deterministic mean over base predictions, no LLM (see metaculus_bot.ablation.run_simple_agg)

# Not prod's opus-5.5: a gpt-5.5 primary 404ed on the operator's local donated key; see the module docstring.
DEFAULT_STACKER_MODEL = "openrouter/anthropic/claude-opus-4.5"
# Matches prod STACKER_FALLBACK_LLM (gpt-5.6-sol -> gpt-6-sol on the 2026-09-22 migration); a different
# provider than the primary on purpose.
DEFAULT_STACKER_FALLBACK_MODEL = "openrouter/openai/gpt-6-sol"
DEFAULT_PARSER_MODEL = "openrouter/openai/gpt-oss-120b:free"

# The --lineup prod stacker, a plain GeneralLlm with no donated-key wrapper; its posture mirrors the prod forecasters.
# opus-4.8 -> opus-5.5 on the 2026-09-22 migration.
PROD_STACKER_MODEL = "openrouter/anthropic/claude-opus-5.5"
# Medium effort, no sampling params: ``temperature=None`` stops litellm injecting one, top_p and max_tokens stay unset.
_PROD_STACKER_KWARGS: dict[str, Any] = {
    "reasoning": {"effort": "medium"},
    "temperature": None,
    "stream": False,
    "timeout": 480,
    "allowed_tries": 1,
}

# Anthropic takes an explicit thinking budget (``reasoning.max_tokens``, as prod ``STACKER_LLM`` in ``llm_configs.py``).
_OPUS_STACKER_KWARGS: dict[str, Any] = {
    "reasoning": {"max_tokens": 32_000},
    "temperature": None,
    "max_tokens": 64_000,
    "stream": False,
    "timeout": 480,
    "allowed_tries": 1,
}
# OpenAI takes ``reasoning.effort`` instead; sampling params follow the repo convention (temperature=None, no top_p).
_OPENAI_STACKER_KWARGS: dict[str, Any] = {
    "reasoning": {"effort": "high"},
    "temperature": None,
    "max_tokens": 64_000,
    "stream": False,
    "timeout": 480,
    "allowed_tries": 1,
}


def _build_default_stacker_llm() -> GeneralLlm:
    """Primary ablation stacker (claude-opus-4.5 via donated-key wrapper).

    Mirrors production STACKER_LLM. Anthropic models work cleanly on the
    donated key. ``allowed_tries=1`` so we don't double-bill the donated key
    on transient stalls — the wrapper's donated→paid handling is the safety
    net.
    """
    return build_llm_with_openrouter_fallback(model=DEFAULT_STACKER_MODEL, **_OPUS_STACKER_KWARGS)


def _build_default_fallback_stacker_llm() -> GeneralLlm:
    """Fallback ablation stacker (gpt-6-sol).

    Mirrors production STACKER_FALLBACK_LLM. Different provider on purpose —
    if Anthropic is thrashing, retrying against Anthropic rarely recovers.
    Goes through the donated-key wrapper too; if the donated key data-policy
    blocks the OpenAI model (operator-specific), the wrapper's fallback to the
    paid key catches it.
    """
    return build_llm_with_openrouter_fallback(model=DEFAULT_STACKER_FALLBACK_MODEL, **_OPENAI_STACKER_KWARGS)


# Re-exports the flag helpers that moved to ``ablation.env`` (breaking a forecasters/run_stacker import cycle).
__all__ = [
    "ABLATION_MIN_FORECASTERS",
    "ARM_MEAN",
    "ARM_MEDIAN",
    "ARM_PDF",
    "ARM_PDF_MIN1",
    "ARM_PDF_MIN2",
    "ARM_STACK",
    "ARM_STACK_AUG",
    "DEFAULT_PARSER_MODEL",
    "DEFAULT_STACKER_FALLBACK_MODEL",
    "DEFAULT_STACKER_MODEL",
    "FEATURE_FLAG_ENV",
    "PROD_STACKER_MODEL",
    "probabilistic_tools_enabled",
    "run_stacker_batch",
    "run_stacker_for_arm",
]

# Held at 2 whatever prod's MIN_FORECASTERS_TO_PUBLISH is (3 -> 2 on 2026-07-20): a stricter floor drops both arms.
ABLATION_MIN_FORECASTERS = 2

# 4 chars/token on the slot's smallest window (gpt-5.5, 128k) less 30k reasoning headroom; overruns 400 both stackers.
APPROX_STACKER_CHAR_LIMIT = 4 * (128_000 - 30_000)

# Tells "not passed" from an explicit None: None means --no-stacker-fallback, which skips the fallback chain.
_UNSET: object = object()


def _truncate_long_rationales(base_texts: list[str], char_limit: int) -> list[str]:
    """Tail-truncate ``base_texts`` to fit total within ``char_limit``.

    Preserves the LAST chars of each rationale because that's where
    forecasters typically put their final probability + summary
    judgement. A head-preserving truncation would cut off the conclusion
    which is the most stacker-relevant part.
    """
    if not base_texts:
        return base_texts
    total = sum(len(t) for t in base_texts)
    if total <= char_limit:
        return base_texts
    per_rationale_limit = max(1, char_limit // len(base_texts))
    return [t[-per_rationale_limit:] if len(t) > per_rationale_limit else t for t in base_texts]


# ---------------------------------------------------------------------------
# Window-patch reentrancy lock
# ---------------------------------------------------------------------------

# Serializes ``patched_window_for_question``, a global monkey-patch that raises on re-entry, across batch calls.
_WINDOW_PATCH_LOCK: asyncio.Lock | None = None


def _get_window_patch_lock() -> asyncio.Lock:
    """Return the module-wide lock, built lazily so it binds to the running event loop."""
    global _WINDOW_PATCH_LOCK  # noqa: PLW0603  # deliberate module-global: lazily-built lock for a module-level monkey-patch
    if _WINDOW_PATCH_LOCK is None:
        _WINDOW_PATCH_LOCK = asyncio.Lock()
    return _WINDOW_PATCH_LOCK


# ---------------------------------------------------------------------------
# Forecaster filtering
# ---------------------------------------------------------------------------


def _is_finite_prediction(prediction_value: Any) -> bool:
    """Return True iff every numeric field inside ``prediction_value`` is finite.

    A free-tier forecaster whose parser produces NaN (e.g. on ``Probability:
    ?%``) used to slip through the prediction_value=None / errors=[] filter
    because Python propagates NaN through ``min``/``max`` clamps. Reject
    NaN/inf values explicitly so they don't poison the cross-model
    aggregator and bootstrap CIs downstream.
    """
    if not isinstance(prediction_value, dict):
        return False
    payload_type = prediction_value.get("type")
    if payload_type == "binary":
        prob = prediction_value.get("prob")
        return isinstance(prob, (int, float)) and math.isfinite(prob)
    if payload_type == "multiple_choice":
        options = prediction_value.get("options") or []
        for option in options:
            prob = option.get("probability") if isinstance(option, dict) else None
            if not isinstance(prob, (int, float)) or not math.isfinite(prob):
                return False
        return True
    if payload_type == "numeric":
        cdf = prediction_value.get("cdf_probabilities") or []
        if not cdf:
            return False
        return all(isinstance(value, (int, float)) and math.isfinite(value) for value in cdf)
    return False


def _surviving_forecasters(forecaster_payloads: dict[str, dict]) -> dict[str, dict]:
    """Drop forecasters with ``prediction_value=None``, errors, or NaN/inf values.

    A failed forecaster (parse error, LLM timeout, etc.) writes its payload
    with ``prediction_value=None`` and ``errors=[...]``; we filter those out
    before stacking. Also drops payloads whose numeric content carries
    NaN/inf — see ``_is_finite_prediction``.
    """
    surviving: dict[str, dict] = {}
    for slug, payload in forecaster_payloads.items():
        if payload.get("prediction_value") is None:
            continue
        if payload.get("errors"):
            continue
        if not _is_finite_prediction(payload["prediction_value"]):
            continue
        surviving[slug] = payload
    return surviving


# ---------------------------------------------------------------------------
# Stacker dispatch
# ---------------------------------------------------------------------------


async def _dispatch_stacker(
    *,
    question: MetaculusQuestion,
    research: str,
    base_texts: list[str],
    stacker_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    aggregated_tool_output: str | None,
) -> tuple[Any, str]:
    """Call the right ``stacking.run_stacking_*`` based on question type.

    ``aggregated_tool_output`` is forwarded directly to ``stacking.run_stacking_*``. The numeric
    branch mirrors ``AggregationPipeline._run_stacking_numeric`` in ``aggregation_pipeline``:
    sanitize, unit-mismatch guard, then the ``NumericDistribution`` the cache needs.
    """
    if isinstance(question, BinaryQuestion):
        return await stacking.run_stacking_binary(
            stacker_llm,
            parser_llm,
            question,
            research=research,
            base_texts=base_texts,
            aggregated_tool_output=aggregated_tool_output,
        )
    if isinstance(question, MultipleChoiceQuestion):
        return await stacking.run_stacking_mc(
            stacker_llm,
            parser_llm,
            question,
            research=research,
            base_texts=base_texts,
            aggregated_tool_output=aggregated_tool_output,
        )
    if isinstance(question, NumericQuestion):
        # Function-scoped so the sanitize_percentiles spy in test_ablation_run_stacker_dispatch.py fires at call time.
        from metaculus_bot.numeric.pipeline import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # late import: tests patch numeric.pipeline.sanitize_percentiles at source
            build_numeric_distribution,
            sanitize_percentiles,
        )

        upper_msg, lower_msg = bound_messages(question)
        perc_list, meta_text = await stacking.run_stacking_numeric(
            stacker_llm,
            parser_llm,
            question,
            research=research,
            base_texts=base_texts,
            lower_bound_message=lower_msg,
            upper_bound_message=upper_msg,
            aggregated_tool_output=aggregated_tool_output,
        )
        percentile_list, zero_point = sanitize_percentiles(list(perc_list), question, model_name=stacker_llm.model)
        mismatch, reason = detect_unit_mismatch(percentile_list, question)
        if mismatch:
            raise UnitMismatchError(
                f"Unit mismatch likely; {reason}. Values: {[float(p.value) for p in percentile_list]}"
            )
        prediction = build_numeric_distribution(percentile_list, question, zero_point, model_name=stacker_llm.model)
        return prediction, meta_text
    raise ValueError(f"Unsupported question type for stacking: {type(question).__name__}")


# ---------------------------------------------------------------------------
# M3 — Tertiary MEDIAN fallback when both stackers fail
# ---------------------------------------------------------------------------


def _median_fallback_prediction(
    question: MetaculusQuestion,
    surviving: dict[str, dict],
) -> Any:
    """Return a MEDIAN aggregation of surviving forecaster predictions.

    Mirrors ``AggregationPipeline._median_fallback``: when both primary and
    fallback stackers fail, MEDIAN-aggregate the per-forecaster
    predictions so the question still gets a publishable forecast.
    Per question type:

    * Binary: median of probabilities.
    * Multiple choice: per-option median, renormalized.
    * Numeric: pointwise median of CDFs (via combine_numeric_predictions).

    Marks the failure mode in the caller's logs; no internal logging
    here so the surrounding context (qid, arm) appears in one place.
    """
    deserialized = [
        deserialize_prediction_value(payload["prediction_value"], question) for payload in surviving.values()
    ]
    if isinstance(question, BinaryQuestion):
        return combine_binary_predictions([float(v) for v in deserialized], AggregationStrategy.MEDIAN)
    if isinstance(question, MultipleChoiceQuestion):
        return combine_multiple_choice_predictions(deserialized, AggregationStrategy.MEDIAN)
    if isinstance(question, NumericQuestion):
        return combine_numeric_predictions(deserialized, question, AggregationStrategy.MEDIAN)
    raise ValueError(f"Unsupported question type for median fallback: {type(question).__name__}")


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StackerArmCell:
    """Identity plus the shared payload fields for one (question, arm) stacker cell.

    Every payload this function writes — success, error, median fallback — carries the
    same forecaster count, computed-quantities map, cross-model aggregation and
    tools-enabled flag, and lands in the same slug-keyed cache cell. Bundling them
    keeps the write sites from drifting apart (the non-finite path once omitted
    ``stacker_slug`` and wrote to the legacy unslugged filename, so a resume re-spent
    on that question).
    """

    qid: int
    arm: str
    cache: AblationCache
    stacker_slug: str | None
    n_forecasters: int
    per_forecaster_md: dict[str, str]
    cross_model_md: str
    enable_tools: bool

    def write_error(
        self,
        *,
        reason: str,
        model_used: str | None,
        errors: list[str],
        meta_reasoning: str = "",
    ) -> dict:
        payload = make_error_payload(
            arm=self.arm,
            reason=reason,
            meta_reasoning=meta_reasoning,
            computed_quantities=self.per_forecaster_md,
            cross_model_aggregation=self.cross_model_md,
            model_used=model_used,
            n_forecasters=self.n_forecasters,
            tools_enabled=self.enable_tools,
            errors=errors,
        )
        self.cache.write_stacker_output(qid=self.qid, arm=self.arm, payload=payload, stacker_slug=self.stacker_slug)
        return payload

    def write_success(
        self,
        *,
        prediction: Any,
        meta_reasoning: str,
        model_used: str | None,
        errors: list[str],
    ) -> dict:
        payload = make_success_payload(
            arm=self.arm,
            prediction=prediction,
            meta_reasoning=meta_reasoning,
            computed_quantities=self.per_forecaster_md,
            cross_model_aggregation=self.cross_model_md,
            model_used=model_used,
            n_forecasters=self.n_forecasters,
            tools_enabled=self.enable_tools,
            errors=errors,
        )
        self.cache.write_stacker_output(qid=self.qid, arm=self.arm, payload=payload, stacker_slug=self.stacker_slug)
        return payload


def _build_stacker_inputs(
    question: MetaculusQuestion,
    surviving: dict[str, dict],
    *,
    research_blob: str,
    qid: int,
    arm: str,
) -> tuple[dict[str, str], list[str], str]:
    """Assemble the stacker prompt's inputs: (computed-quantities map, base texts, aggregation).

    Three steps, in order. Per-forecaster tool augmentation:
    ``run_tools_for_forecaster`` checks the env flag internally and returns "" when
    off, so on arm A this produces no augmentations and each rationale passes through
    unchanged (the ``## Computed quantities`` append mirrors ``TemplateForecaster._make_prediction``);
    base texts are stripped of the leading ``Model: <name>`` tag, mirroring
    ``AggregationPipeline.run_stacking``. Then the once-per-question
    cross-model aggregation, which receives the *raw* (with-Model-tag) rationales for
    parsing — the structured-block extractors don't depend on the tag, and production
    in ``stacking_route._finalize_stacked_prediction`` passes raw rationales too. Finally the prompt-size
    budget trim.
    """
    per_forecaster_md: dict[str, str] = {}
    base_texts: list[str] = []
    deserialized_values: list[Any] = []
    for slug, payload in surviving.items():
        raw_rationale = payload["reasoning"]
        # forecasters.run_forecasters_batch fixes the payload schema; a direct ``payload["model"]`` surfaces drift.
        computed_md = tool_runner.run_tools_for_forecaster(
            question=question,
            rationale=raw_rationale,
            forecaster_id=payload["model"],
        )
        if computed_md:
            augmented_rationale = f"{raw_rationale}\n\n## Computed quantities\n{computed_md}"
            per_forecaster_md[slug] = computed_md
        else:
            augmented_rationale = raw_rationale
        base_texts.append(stacking.strip_model_tag(augmented_rationale))
        deserialized_values.append(deserialize_prediction_value(payload["prediction_value"], question))

    cross_model_md = tool_runner.build_cross_model_aggregation(
        question=question,
        rationales=[p["reasoning"] for p in surviving.values()],
        prediction_values=deserialized_values,
    )
    base_texts = _rationales_within_budget(
        base_texts,
        research_blob=research_blob,
        aggregated_for_stacker=cross_model_md or None,
        qid=qid,
        arm=arm,
    )
    return per_forecaster_md, base_texts, cross_model_md


def _rationales_within_budget(
    base_texts: list[str],
    *,
    research_blob: str,
    aggregated_for_stacker: str | None,
    qid: int,
    arm: str,
) -> list[str]:
    """Tail-preserving per-rationale truncation when the assembled prompt is too big.

    Free-tier forecasters can emit 200k+ char rationales, and 4 of them stacked
    together blow past Claude/GPT context windows. Returns ``base_texts`` unchanged
    when the prompt already fits.
    """
    research_budget = len(research_blob) + len(aggregated_for_stacker or "")
    rationale_budget = APPROX_STACKER_CHAR_LIMIT - research_budget
    total_rationale_chars = sum(len(text) for text in base_texts)
    if total_rationale_chars <= max(0, rationale_budget):
        return base_texts
    logger.warning(
        "stacker | qid=%s arm=%s | prompt size %d > %d limit; truncating rationales",
        qid,
        arm,
        total_rationale_chars + research_budget,
        APPROX_STACKER_CHAR_LIMIT,
    )
    return _truncate_long_rationales(base_texts, max(1, rationale_budget))


async def _stack_with_fallback(
    question: MetaculusQuestion,
    *,
    research_blob: str,
    base_texts: list[str],
    stacker_llm: GeneralLlm,
    fallback_stacker_llm: GeneralLlm | None,
    parser_llm: GeneralLlm,
    aggregated_for_stacker: str | None,
    qid: int,
    arm: str,
) -> tuple[tuple[Any, str] | None, str | None, list[str]]:
    """Primary stacker then fallback; returns (result, which_model_used, errors).

    ``which_model_used`` is "primary", "fallback", or None when nothing produced a
    result. A None ``fallback_stacker_llm`` means --no-stacker-fallback, which skips
    the fallback chain entirely.

    ``patched_window_for_question`` is a global monkey-patch that raises on nested
    entry, so entry is serialized under the module-level asyncio.Lock: each call gets
    its own patched region without colliding with a concurrent batch call.

    Both waits mirror the primary and fallback deadlines in
    ``AggregationPipeline.stack_predictions``.
    """
    errors: list[str] = []
    async with _get_window_patch_lock():
        with patched_window_for_question(question):
            try:
                # A stuck stacker would otherwise hold the lock for litellm's whole 480 s.
                result = await asyncio.wait_for(
                    _dispatch_stacker(
                        question=question,
                        research=research_blob,
                        base_texts=base_texts,
                        stacker_llm=stacker_llm,
                        parser_llm=parser_llm,
                        aggregated_tool_output=aggregated_for_stacker,
                    ),
                    timeout=STACKER_SOFT_DEADLINE,
                )
            except EXPECTED_LLM_CALL_FAILURES as primary_exc:
                logger.exception("Primary stacker failed for qid=%s arm=%s", qid, arm)
                errors.append(f"primary: {type(primary_exc).__name__}: {primary_exc!r}")
            else:
                return result, "primary", errors

            if fallback_stacker_llm is None:
                return None, None, errors
            try:
                # Tighter deadline: a fallback is already late on the critical path.
                result = await asyncio.wait_for(
                    _dispatch_stacker(
                        question=question,
                        research=research_blob,
                        base_texts=base_texts,
                        stacker_llm=fallback_stacker_llm,
                        parser_llm=parser_llm,
                        aggregated_tool_output=aggregated_for_stacker,
                    ),
                    timeout=STACKER_FALLBACK_SOFT_DEADLINE,
                )
            except EXPECTED_LLM_CALL_FAILURES as fallback_exc:
                logger.exception("Fallback stacker failed for qid=%s arm=%s", qid, arm)
                errors.append(f"fallback: {type(fallback_exc).__name__}: {fallback_exc!r}")
                return None, None, errors
            return result, "fallback", errors


def _median_fallback_payload(
    cell: _StackerArmCell,
    *,
    question: MetaculusQuestion,
    surviving: dict[str, dict],
    model_used: str | None,
    errors: list[str],
) -> dict:
    """Tertiary MEDIAN fallback payload (mirror of the MEDIAN rung in ``AggregationPipeline.stack_predictions``).

    Both stackers failed but we still have surviving forecasters, so median-aggregate
    them: the question gets a degraded-but-publishable forecast instead of being lost
    from both arms. The ``model_used="median_fallback"`` tag lets confounder analysis
    bucket these separately from the regular primary/fallback outcomes. If the median
    itself rejects the surviving predictions, an error payload is written instead.
    """
    try:
        median_prediction = _median_fallback_prediction(question, surviving)
        payload = cell.write_success(
            prediction=serialize_prediction_value(median_prediction, question_type_for_serialization(question)),
            meta_reasoning="median_fallback: both stackers failed",
            model_used="median_fallback",
            errors=errors,
        )
    except (TypeError, ValueError) as median_exc:  # the aggregators' and serializer's own guards; a bug propagates
        logger.exception("Median fallback failed for qid=%s arm=%s", cell.qid, cell.arm)
        errors.append(f"median_fallback: {type(median_exc).__name__}: {median_exc!r}")
        return cell.write_error(reason="stacker_failed", model_used=model_used, errors=errors)
    logger.warning(
        "Median fallback engaged for qid=%s arm=%s after both stackers failed",
        cell.qid,
        cell.arm,
    )
    return payload


def _payload_from_stacker_result(
    cell: _StackerArmCell,
    *,
    question: MetaculusQuestion,
    result: tuple[Any, str],
    model_used: str | None,
    errors: list[str],
) -> dict:
    """Serialize the stacker's output into a success payload, or an error on NaN/inf.

    NaN/inf in the stacker output would corrupt cross-model aggregation, bootstrap CIs,
    and any cached downstream consumer, so it is treated the same as both stackers
    failing: an error payload, no cache pollution.
    """
    stacker_prediction, stacker_meta = result
    serialized = serialize_prediction_value(stacker_prediction, question_type_for_serialization(question))
    if not _is_finite_prediction(serialized):
        logger.error(
            "Stacker emitted non-finite prediction_value for qid=%s arm=%s; recording failure",
            cell.qid,
            cell.arm,
        )
        errors.append(f"{model_used}: stacker output contained NaN/inf")
        return cell.write_error(
            reason="stacker_nonfinite_output",
            model_used=model_used,
            errors=errors,
            meta_reasoning=stacker_meta,
        )
    return cell.write_success(
        prediction=serialized,
        meta_reasoning=stacker_meta,
        model_used=model_used,
        errors=errors,
    )


async def run_stacker_for_arm(
    question: MetaculusQuestion,
    research_blob: str,
    forecaster_payloads: dict[str, dict],
    arm: str,
    *,
    cache: AblationCache,
    stacker_llm: GeneralLlm | None = None,
    fallback_stacker_llm: GeneralLlm | object | None = _UNSET,
    parser_llm: GeneralLlm | None = None,
    stacker_slug: str | None = None,
    force: bool = False,
) -> dict:
    """Run the stacker for one arm of one question, cached per ``(qid, arm, stacker_slug)``.

    ``stacker_slug`` keys the cache filename to the active stacker so a swap (opus-4.5 free-tier
    versus opus-5.5 prod) never overwrites another stacker's results; callers derive it with
    ``model_slug_to_filename(<stacker model>)``, and ``None`` keeps the legacy ``arm_<arm>.json``
    name the tests rely on. The slug applies to every read and write here, the median-fallback
    payload included, since that is still this stacker arm's cell.

    Fewer than ``ABLATION_MIN_FORECASTERS`` surviving forecasters caches an error payload and
    returns. With ``fallback_stacker_llm=None`` (``--no-stacker-fallback``) a primary failure is
    cached and then raised so the run aborts; resume with ``--qids <remaining>``. Otherwise both
    stackers failing degrades to the MEDIAN of the surviving forecasters.
    """
    qid = question.id_of_question
    assert qid is not None, "run_stacker_for_arm requires question.id_of_question"

    if not force:
        cached = cache.read_stacker_output(qid=qid, arm=arm, stacker_slug=stacker_slug)
        if cached is not None:
            # A guaranteed checkpoint on this otherwise-sync early return, for flake8-async ASYNC910.
            await asyncio.sleep(0)
            return cached

    surviving = _surviving_forecasters(forecaster_payloads)
    if len(surviving) < ABLATION_MIN_FORECASTERS:
        payload = make_error_payload(
            arm=arm,
            reason="insufficient_forecasters",
            model_used=None,
            n_forecasters=len(surviving),
            tools_enabled=arm == ARM_STACK_AUG,
        )
        cache.write_stacker_output(qid=qid, arm=arm, payload=payload, stacker_slug=stacker_slug)
        await asyncio.sleep(0)
        return payload

    # Defaults for any LLM the caller omitted (tests pass all three); the stacker defaults ride the donated-key wrapper.
    if stacker_llm is None:
        stacker_llm = _build_default_stacker_llm()
    # _UNSET means "not specified", so build the default; an explicit None (--no-stacker-fallback) stays None.
    if fallback_stacker_llm is _UNSET:
        fallback_stacker_llm = _build_default_fallback_stacker_llm()
    if parser_llm is None:
        parser_llm = GeneralLlm(model=DEFAULT_PARSER_MODEL, allowed_tries=1)

    enable_tools = arm == ARM_STACK_AUG

    with probabilistic_tools_enabled(enable_tools):
        per_forecaster_md, base_texts, cross_model_md = _build_stacker_inputs(
            question,
            surviving,
            research_blob=research_blob,
            qid=qid,
            arm=arm,
        )
        result, stacker_model_used, errors_list = await _stack_with_fallback(
            question,
            research_blob=research_blob,
            base_texts=base_texts,
            stacker_llm=stacker_llm,
            # Only an explicit None survives the default fill above, and it means no fallback chain.
            fallback_stacker_llm=cast("GeneralLlm | None", fallback_stacker_llm),
            parser_llm=parser_llm,
            # Production passes ``aggregated_tool_output or None``; mirror that.
            aggregated_for_stacker=cross_model_md or None,
            qid=qid,
            arm=arm,
        )

    cell = _StackerArmCell(
        qid=qid,
        arm=arm,
        cache=cache,
        stacker_slug=stacker_slug,
        n_forecasters=len(surviving),
        per_forecaster_md=per_forecaster_md,
        cross_model_md=cross_model_md or "",
        enable_tools=enable_tools,
    )

    if result is None and fallback_stacker_llm is None:
        # Fail fast: a borked key aborts at qid #1 instead of failing all 88; the cached failure survives for resume.
        cell.write_error(reason="stacker_failed_no_fallback", model_used=stacker_model_used, errors=errors_list)
        raise RuntimeError(
            f"Stacker failed for qid={qid} arm={arm} with --no-stacker-fallback set. "
            f"Aborting run. Errors: {'; '.join(errors_list) if errors_list else '<no errors recorded>'}. "
            f"Resume after fixing root cause; cache has the failure payload."
        )

    if result is None:
        payload = _median_fallback_payload(
            cell, question=question, surviving=surviving, model_used=stacker_model_used, errors=errors_list
        )
    else:
        payload = _payload_from_stacker_result(
            cell, question=question, result=result, model_used=stacker_model_used, errors=errors_list
        )
    await asyncio.sleep(0)
    return payload


# ---------------------------------------------------------------------------
# Batch wrapper
# ---------------------------------------------------------------------------


async def run_stacker_batch(
    qid_to_data: dict[int, dict],
    arm: str,
    cache: AblationCache,
    *,
    stacker_llm: GeneralLlm | None = None,
    fallback_stacker_llm: GeneralLlm | object | None = _UNSET,
    parser_llm: GeneralLlm | None = None,
    stacker_slug: str | None = None,
    force: bool = False,
    concurrency: int = 2,
) -> dict[int, dict]:
    """Run the stacker for one arm across many questions.

    Returns ``{qid: payload}``. The same LLMs are reused across questions
    (built once at the call site or here, then passed straight into each
    per-question runner). Per-question failure (insufficient forecasters,
    both stackers down, etc.) is recorded in that question's payload and
    the batch continues.

    ``stacker_slug`` keys each per-question cache cell to the active stacker (see
    ``run_stacker_for_arm``). ``None`` keeps the legacy unslugged filename.
    """
    if stacker_llm is None:
        stacker_llm = _build_default_stacker_llm()
    if fallback_stacker_llm is _UNSET:
        fallback_stacker_llm = _build_default_fallback_stacker_llm()
    if parser_llm is None:
        parser_llm = GeneralLlm(model=DEFAULT_PARSER_MODEL, allowed_tries=1)

    semaphore = asyncio.Semaphore(concurrency)

    async def _one(qid: int, data: dict) -> tuple[int, dict]:
        async with semaphore:
            payload = await run_stacker_for_arm(
                question=data["question"],
                research_blob=data["research"],
                forecaster_payloads=data["forecaster_payloads"],
                arm=arm,
                cache=cache,
                stacker_llm=stacker_llm,
                fallback_stacker_llm=fallback_stacker_llm,
                parser_llm=parser_llm,
                stacker_slug=stacker_slug,
                force=force,
            )
        return qid, payload

    tasks = [_one(qid, data) for qid, data in qid_to_data.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)
