"""Forecaster fan-out runner for the probabilistic-tools ablation benchmark.

Runs N free-model forecasters per question ONCE against pre-cached Gemini
research. Caches per-(qid, model) on disk via :class:`AblationCache`. The
question-anchored window patch wraps the prediction call so prompts don't
leak today's date — important for resolved-question backtests where any
"today" reference would tip off the forecaster that the question already
resolved.

The same per-forecaster outputs feed BOTH stacker arms (tools-on, tools-off)
in the downstream A/B benchmark, so we run forecasters once per question per
backtest and cache aggressively. An expected call failure is recorded in the
forecaster's payload rather than raised, so one bad provider cannot poison the
batch; a bug is left to propagate.

Design choices:

* ``_make_prediction`` is called directly (not the soft-deadline wrapper) so
  test mocks can replace it cleanly. The runner wraps each call with a
  per-(qid, model) timer and exception capture.
* Aggregation is :class:`AggregationStrategy.MEAN` because we want raw
  per-model predictions, not stacker output — the stacker stage is downstream.
* Each bot is per-(question, forecaster) because ``research_cache`` is keyed
  by ``id_of_question`` and we want each instance's cache to contain exactly
  the one entry for the question we're forecasting.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import re
import time
from datetime import UTC, datetime
from typing import Any

import openai
from forecasting_tools import (
    GeneralLlm,
    MetaculusQuestion,
    NumericDistribution,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile
from litellm.exceptions import RateLimitError

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.ablation.env import probabilistic_tools_enabled
from metaculus_bot.ablation.forecaster_lineup import (
    build_free_forecaster_llms,
    build_free_parser_llm,
)
from metaculus_bot.ablation.window_patch import patched_window_for_question
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import FORECASTER_SOFT_DEADLINE
from metaculus_bot.exceptions import ValueExtractionError
from metaculus_bot.forecaster import TemplateForecaster
from metaculus_bot.llm_configs import RESEARCHER_LLM, SUMMARIZER_LLM
from metaculus_bot.llm_retry import llm_status_code
from metaculus_bot.mc_processing import clamp_and_renormalize_probs
from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution
from metaculus_bot.question_types import question_type_of

logger = logging.getLogger(__name__)

# An expected failure of an LLM call in this harness, as opposed to a bug (KeyError, AttributeError).
EXPECTED_LLM_CALL_FAILURES: tuple[type[Exception], ...] = (
    openai.APIError,  # the root of the whole litellm exception tree
    TimeoutError,  # a soft deadline firing
    RuntimeError,  # forecasting-tools raises this on an empty completion
    ValueError,  # a rejected value; UnitMismatchError subclasses it
    ValueExtractionError,  # every rung of the extraction ladder failed
)

# Caps the backoff fallback (no ``retry_after_seconds`` in the 429) so a stuck attempt can't park the runner.
_RATE_LIMIT_BACKOFF_CAP_SECONDS: float = 60.0

# Above the 90s+ free-tier Retry-Afters seen (a lower cap sheds forecasters), below a 3600s one that parks the batch.
_MAX_RATE_LIMIT_SLEEP_SECONDS: float = 120.0

# Matches the "gentle" preset in cli.py; the CLI always passes an explicit value, so this only serves ad-hoc callers.
_DEFAULT_MAX_RETRIES: int = 3

# Live OpenRouter 429 shape: "retry_after_seconds":13,"retry_after_seconds_raw":12.315; the int key mirrors Retry-After.
_RETRY_AFTER_REGEX = re.compile(r'"retry_after_seconds"\s*:\s*(\d+(?:\.\d+)?)')

# OpenRouter sometimes proxies a date-form (RFC 7231) Retry-After; unparsed it would fall to the 60s backoff cap.
_RETRY_AFTER_DATE_REGEX = re.compile(r'"Retry-After"\s*:\s*"([^"\d][^"]*)"')

# OpenRouter 429s carry "provider_name":"Venice" and the like; the retry log names which shared upstream is throttling.
_PROVIDER_NAME_REGEX = re.compile(r'"provider_name"\s*:\s*"([^"]+)"')


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if the exception is a 429 / upstream-provider rate limit.

    Defensive about exception class drift: matches both ``litellm.RateLimitError``
    *and* any exception whose stringified message carries ``"code":429`` or
    ``rate-limited upstream``. The upstream-rate-limit text is OpenRouter-specific
    and stable across litellm versions.
    """
    if isinstance(exc, RateLimitError):
        return True
    msg = str(exc)
    if '"code":429' in msg or "code: 429" in msg:
        return True
    return "rate-limited upstream" in msg.lower()


def _parse_retry_after_seconds(exc: BaseException) -> float | None:
    """Extract ``retry_after_seconds`` from an OpenRouter 429 payload.

    Returns None if the field isn't present (caller falls back to exponential
    backoff). Uses regex rather than json.loads because the litellm-wrapped
    string carries a JSON blob with nested braces + string-escaped JSON inside,
    which makes substring json.loads brittle.

    Two upstream forms supported:
    * Integer-seconds in ``retry_after_seconds`` (the common OpenRouter shape).
    * HTTP-date in the ``Retry-After`` header field (RFC 7231 alt form). We
      compute (target_time - now) clamped to >= 0 so a stale past-date sleeps
      nothing and retries immediately.
    """
    text = str(exc)
    match = _RETRY_AFTER_REGEX.search(text)
    if match is not None:
        return float(match.group(1))
    date_match = _RETRY_AFTER_DATE_REGEX.search(text)
    if date_match is None:
        return None
    raw_date = date_match.group(1)
    try:
        target = email.utils.parsedate_to_datetime(raw_date)
    except (ValueError, TypeError) as parse_exc:
        logger.warning("parsing Retry-After date %r: %s", raw_date, parse_exc)
        return None
    if target is None:
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    delta = (target - datetime.now(UTC)).total_seconds()
    return max(0.0, delta)


def _parse_provider_name(exc: BaseException) -> str | None:
    """Extract ``provider_name`` (e.g. ``"Venice"``) from an OpenRouter 429 payload.

    Returns None if absent. Same regex-over-json.loads tradeoff as
    ``_parse_retry_after_seconds`` — the litellm-wrapped exception string is
    not parseable JSON without significant work.
    """
    match = _PROVIDER_NAME_REGEX.search(str(exc))
    if match is None:
        return None
    return match.group(1)


def _backoff_seconds(attempt: int) -> float:
    """Jittered exponential backoff for attempt index ``attempt`` (0-indexed).

    ``2**attempt + uniform[0, 1)`` capped at ``_RATE_LIMIT_BACKOFF_CAP_SECONDS``.
    Matches the OpenRouter docs' 1s/2s/4s/8s recommendation when uncapped.
    """
    return min(_RATE_LIMIT_BACKOFF_CAP_SECONDS, (2**attempt) + random.uniform(0.0, 1.0))  # noqa: S311  # non-cryptographic backoff jitter


# The window patch is a global monkey-patch that refuses nested entry, so concurrent questions take turns inside it.
_WINDOW_PATCH_LOCK: asyncio.Lock | None = None


def _get_window_patch_lock() -> asyncio.Lock:
    """Return the module-wide lock, built lazily so it binds to the running event loop."""
    global _WINDOW_PATCH_LOCK  # noqa: PLW0603  # deliberate module-global: lazily-built lock for a module-level monkey-patch
    if _WINDOW_PATCH_LOCK is None:
        _WINDOW_PATCH_LOCK = asyncio.Lock()
    return _WINDOW_PATCH_LOCK


__all__ = [
    "deserialize_prediction_value",
    "question_type_for_serialization",
    "run_forecasters_batch",
    "run_forecasters_for_question",
    "serialize_prediction_value",
]


# ---------------------------------------------------------------------------
# Failure-stage classification
# ---------------------------------------------------------------------------


def _infer_failure_stage(exc: Exception, forecaster_model_slug: str) -> str:
    """Heuristically tag which stage of ``_make_prediction`` raised: ``forecaster``, ``parser`` or ``unknown``.

    Exception types are litellm-generic and rarely name the model, so the tag rests on the
    heuristics that survived first-light. :class:`ValueExtractionError` is the extraction ladder's
    typed terminal failure (the ladder replaced the parser LLM stage) and wins before any text is
    read. "no allowed providers" is the donated-key 404, which only the OAI-prefixed parser takes.
    A 429 status or rate-limit wording is the forecaster, because a throttled forecaster never
    reaches the parser; the wording covers SDKs that report no status. The tag is advisory for
    log readers; the full exception stays in the payload's ``errors`` list.
    """
    if isinstance(exc, ValueExtractionError):
        return "parser"
    msg = str(exc).lower()
    # The donated-key allowed-providers 404 hits only the OAI-prefixed parser; forecasters here are plain GeneralLlm.
    if "no allowed providers" in msg:
        return "parser"
    # Never ``"429" in msg``: the body embeds a 64-hex key hash and prompt text, so bare digits matched by accident.
    if llm_status_code(exc) == 429 or "rate limit" in msg or "too many requests" in msg:
        return "forecaster"
    # Parser-side errors name the parser model, so the forecaster slug in the text means the forecaster call raised.
    if forecaster_model_slug.lower() in msg:
        return "forecaster"
    return "unknown"


# ---------------------------------------------------------------------------
# Prediction-value (de)serialization
# ---------------------------------------------------------------------------


def question_type_for_serialization(question: MetaculusQuestion) -> str:
    """Return the canonical question-type discriminator string used by
    :func:`serialize_prediction_value`.

    Exposed (rather than ``_question_type``) so sibling modules in
    ``metaculus_bot.ablation`` can serialize prediction values consistently.
    The returned strings double as the wire-format discriminator in cached arm
    payloads, so they must stay aligned with ``question_types.question_type_of``.
    """
    qtype = question_type_of(question)
    if qtype is None:
        raise ValueError(f"Unsupported question type: {type(question).__name__}")
    return qtype


def serialize_prediction_value(value: Any, question_type: str) -> dict[str, Any]:
    """Convert a prediction_value from ``_make_prediction`` into a JSON-safe dict.

    Schema by question type:

    * Binary → ``{"type": "binary", "prob": float}``
    * MC → ``{"type": "multiple_choice", "options": [{"option_name", "probability"}, ...]}``
    * Numeric → full ``NumericDistribution`` round-trip (see below).

    The numeric payload captures *everything* the forecaster's pipeline computed —
    declared percentiles, the constraint-enforced 201-point CDF (probabilities), and
    the bounds/zero_point/cdf_size needed to reconstruct ``PchipNumericDistribution``
    on read. Storing the raw CDF avoids silently re-deriving via
    ``build_numeric_distribution`` at deserialize time, which would couple cached
    artifacts to whatever pipeline code happens to be running when the cache is read.
    """
    if question_type == "binary":
        return {"type": "binary", "prob": float(value)}
    if question_type == "multiple_choice":
        if not isinstance(value, PredictedOptionList):
            raise TypeError(f"Expected PredictedOptionList for MC, got {type(value).__name__}")
        return {
            "type": "multiple_choice",
            "options": [
                {"option_name": opt.option_name, "probability": float(opt.probability)}
                for opt in value.predicted_options
            ],
        }
    if question_type == "numeric":
        # Never iterate the distribution: a Pydantic model yields (field, value) tuples, which was the original bug.
        if not isinstance(value, NumericDistribution):
            raise TypeError(f"Expected NumericDistribution for numeric, got {type(value).__name__}")
        cdf_points = value.cdf
        return {
            "type": "numeric",
            "declared_percentiles": [
                {"percentile": float(p.percentile), "value": float(p.value)} for p in value.declared_percentiles
            ],
            "cdf_probabilities": [float(p.percentile) for p in cdf_points],
            "lower_bound": float(value.lower_bound),
            "upper_bound": float(value.upper_bound),
            "open_lower_bound": bool(value.open_lower_bound),
            "open_upper_bound": bool(value.open_upper_bound),
            "zero_point": float(value.zero_point) if value.zero_point is not None else None,
            "cdf_size": int(value.cdf_size) if value.cdf_size is not None else None,
        }
    raise ValueError(f"Unknown question_type for serialization: {question_type}")


def deserialize_prediction_value(payload: dict[str, Any], question: MetaculusQuestion) -> Any:
    """Inverse of ``serialize_prediction_value``. Returns the original Python type.

    For numeric payloads, reconstructs a ``PchipNumericDistribution`` using the
    cached 201-point CDF directly — does NOT re-run ``build_numeric_distribution``,
    so cached artifacts are independent of whatever pipeline code is currently
    loaded. This matters for stacker stages reading forecaster outputs that were
    cached weeks ago.

    Old payload shape (``percentiles`` instead of ``declared_percentiles``, no
    ``cdf_probabilities``) is recognized and rejected loudly — re-run the
    forecaster stage with ``--force-stages forecast`` to upgrade.
    """
    payload_type = payload["type"]
    if payload_type == "binary":
        return float(payload["prob"])
    if payload_type == "multiple_choice":
        # Clamp first: ft 0.2.92's validator raises above 0.05 on the sub-0.01 options some old-era payloads carry.
        options_payload = payload["options"]
        clamped = clamp_and_renormalize_probs([float(opt["probability"]) for opt in options_payload])
        return PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=opt["option_name"], probability=prob)
                for opt, prob in zip(options_payload, clamped, strict=True)
            ]
        )
    if payload_type == "numeric":
        if "cdf_probabilities" not in payload:
            raise ValueError(
                "Numeric forecaster payload predates the full-CDF round-trip "
                "(missing 'cdf_probabilities' key). Re-run the forecaster stage "
                "with --force-stages forecast to upgrade cached payloads."
            )
        declared = [
            Percentile(percentile=float(p["percentile"]), value=float(p["value"]))
            for p in payload["declared_percentiles"]
        ]
        cdf_probabilities: list[float] = [float(p) for p in payload["cdf_probabilities"]]
        # Bounds and cdf_size come off ``question``; the rehydrated shim carries them verbatim, keeping the value axis.
        return create_pchip_numeric_distribution(
            pchip_cdf=cdf_probabilities,
            percentile_list=declared,
            question=question,  # type: ignore[arg-type]  # NumericQuestion at runtime
            zero_point=payload.get("zero_point"),
        )
    raise ValueError(f"Unknown prediction_value payload type: {payload_type}")


# ---------------------------------------------------------------------------
# Bot construction
# ---------------------------------------------------------------------------


def _build_bot(
    *,
    question: MetaculusQuestion,
    research_blob: str,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
) -> TemplateForecaster:
    """Construct a single-forecaster TemplateForecaster keyed to one question.

    ``research_cache={qid: research_blob}`` plus ``is_benchmarking=True`` makes
    ``run_research`` short-circuit and return the cached blob without invoking
    any provider.

    ``aggregation_strategy=MEAN`` ensures the stacker doesn't fire — we just
    want the per-model rationale and prediction value from ``_make_prediction``.
    """
    qid = question.id_of_question
    if qid is None:
        raise ValueError("Question must have id_of_question for ablation forecasting")
    # ``forecasters`` holds a list, which the parent ``llms`` type ``dict[str, str | GeneralLlm]`` cannot express.
    llms: dict[str, Any] = {
        "forecasters": [forecaster_llm],
        "parser": parser_llm,
        "summarizer": SUMMARIZER_LLM,
        "researcher": RESEARCHER_LLM,
    }
    return TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        llms=llms,
        aggregation_strategy=AggregationStrategy.MEAN,
        research_provider=None,
        max_questions_per_run=None,
        is_benchmarking=True,
        max_concurrent_research=1,
        research_cache={qid: research_blob},
        min_forecasters_to_publish=1,
    )


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------


async def _predict_with_rate_limit_retries(
    bot: TemplateForecaster,
    question: MetaculusQuestion,
    research_blob: str,
    *,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    qid: int,
    max_retries: int,
) -> tuple[ReasonedPrediction | None, list[str]]:
    """Call ``_make_prediction``, retrying 429s, and return (prediction, errors).

    Attempt budget: 1 initial attempt + ``max_retries`` retries on 429. Non-429
    errors fall through after a single attempt. ``errors`` accumulates EVERY failed
    attempt's stringified exception so a postmortem can see the full sequence; on a
    success that follows transient 429s it is cleared, so downstream consumers can
    treat ``len(errors) == 0`` as "this forecaster delivered" without false positives
    from recovered retries. ``prediction`` is None iff every attempt failed.
    """
    prediction: ReasonedPrediction | None = None
    errors: list[str] = []
    for attempt in range(max_retries + 1):
        try:
            # Mirrors prod's soft deadline: a stuck forecaster once held a question for 480s x 3 litellm tries, ~24 min.
            prediction = await asyncio.wait_for(
                bot._make_prediction(question, research_blob, forecaster_llm),
                timeout=FORECASTER_SOFT_DEADLINE,
            )
            errors = []
            break
        except EXPECTED_LLM_CALL_FAILURES as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if not _is_rate_limit_error(exc):
                # A parser 404 once mis-tagged as the forecaster is why ``_infer_failure_stage`` exists.
                stage = _infer_failure_stage(exc, forecaster_llm.model)
                logger.warning(
                    "ablation forecaster failed | qid=%s | forecaster_model=%s | "
                    "likely_stage=%s | parser_model=%s | %s: %s",
                    qid,
                    forecaster_llm.model,
                    stage,
                    parser_llm.model,
                    type(exc).__name__,
                    exc,
                )
                break
            # Rate-limited path. If we've exhausted retries, log + give up.
            if attempt >= max_retries:
                logger.warning(
                    "ablation forecaster rate-limited (retries exhausted) | qid=%s | "
                    "forecaster_model=%s | attempts=%d | %s: %s",
                    qid,
                    forecaster_llm.model,
                    attempt + 1,
                    type(exc).__name__,
                    exc,
                )
                break
            await _sleep_before_rate_limit_retry(
                exc,
                forecaster_llm=forecaster_llm,
                qid=qid,
                attempt=attempt,
                max_retries=max_retries,
            )
    return prediction, errors


async def _sleep_before_rate_limit_retry(
    exc: Exception,
    *,
    forecaster_llm: GeneralLlm,
    qid: int,
    attempt: int,
    max_retries: int,
) -> None:
    """Sleep out a 429 before the next attempt, logging the wait.

    Honors ``retry_after_seconds`` when the exception carries it; jitter prevents a
    thundering-herd wake when many forecasters share an upstream provider (Venice,
    OpenInference) and all hit the same window. Capped at
    ``_MAX_RATE_LIMIT_SLEEP_SECONDS`` to bound a misbehaving upstream that signals an
    unreasonably long Retry-After (3600s / one hour observed on hot-tail throttles).
    """
    retry_after = _parse_retry_after_seconds(exc)
    if retry_after is not None:
        sleep_seconds = min(retry_after + random.uniform(0.1, 0.5), _MAX_RATE_LIMIT_SLEEP_SECONDS)  # noqa: S311  # non-cryptographic backoff jitter
    else:
        sleep_seconds = _backoff_seconds(attempt)
    logger.info(
        "ablation forecaster rate-limited (retrying) | qid=%s | "
        "forecaster_model=%s | provider=%s | attempt=%d/%d | sleep=%.2fs",
        qid,
        forecaster_llm.model,
        _parse_provider_name(exc) or "unknown",
        attempt + 1,
        max_retries,
        sleep_seconds,
    )
    await asyncio.sleep(sleep_seconds)


def _serialize_prediction_or_record_error(
    prediction: ReasonedPrediction | None,
    question: MetaculusQuestion,
    *,
    errors: list[str],
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    qid: int,
) -> tuple[Any, str]:
    """Serialize ``prediction`` for the cache payload, or record the failure in ``errors``.

    Serialization is isolated from the caller so a single forecaster's
    post-prediction failure (e.g. the ``tuple`` AttributeError that started this
    whole exercise) cannot cascade through ``asyncio.gather``'s default
    ``return_exceptions=False`` and wipe out every other forecaster's already-cached
    output for this qid. Each forecaster's payload — success or failure — must reach
    disk independently, so a failure appends to ``errors`` (in place) and yields the
    empty ``(None, "")`` pair.
    """
    if prediction is None:
        return None, ""
    try:
        qtype = question_type_for_serialization(question)
        return serialize_prediction_value(prediction.prediction_value, qtype), prediction.reasoning
    except (TypeError, ValueError) as exc:  # the serializer's own shape guards; a bug type propagates instead
        errors.append(f"{type(exc).__name__}: {exc}")
        # Tag the stage outright: the original tuple AttributeError raised here and cost a 30-minute grep detour.
        logger.warning(
            "ablation forecaster failed | qid=%s | forecaster_model=%s | "
            "likely_stage=serialize | parser_model=%s | %s: %s",
            qid,
            forecaster_llm.model,
            parser_llm.model,
            type(exc).__name__,
            exc,
        )
        return None, ""


async def _run_one_forecaster(
    question: MetaculusQuestion,
    research_blob: str,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    *,
    cache: AblationCache,
    semaphore: asyncio.Semaphore,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> tuple[str, dict[str, Any]]:
    """Run one forecaster on one question; return (model_slug_filename, payload).

    An expected call or serialization failure is captured in the payload's ``errors``
    list with ``prediction_value=None``, and the payload is written to cache before
    returning so a failure leaves a record too. A bug propagates and caches nothing.

    On ``litellm.RateLimitError`` (or any 429-shaped exception), the call is
    retried up to ``max_retries`` times. The sleep duration honors
    ``retry_after_seconds`` from the OpenRouter exception payload when present
    (plus a small jitter to dodge thundering-herd wakeups when many forecasters
    are throttled simultaneously); otherwise falls back to capped jittered
    exponential backoff. Non-429 errors are NOT retried — they fall through to
    the existing single-attempt error path.
    """
    model_slug = model_slug_to_filename(forecaster_llm.model)
    qid = question.id_of_question
    assert qid is not None, "Question must have id_of_question for ablation forecasting"

    async with semaphore:
        bot = _build_bot(
            question=question,
            research_blob=research_blob,
            forecaster_llm=forecaster_llm,
            parser_llm=parser_llm,
        )

        logger.info("ablation forecaster start | qid=%s | model=%s", qid, forecaster_llm.model)
        start = time.monotonic()
        # ``_make_prediction`` needs a notepad registered first; we skip ``_run_individual_question``, so that is ours.
        notepad = await bot._initialize_notepad(question)
        async with bot._note_pad_lock:
            bot._note_pads.append(notepad)
        try:
            prediction, errors = await _predict_with_rate_limit_retries(
                bot,
                question,
                research_blob,
                forecaster_llm=forecaster_llm,
                parser_llm=parser_llm,
                qid=qid,
                max_retries=max_retries,
            )
        finally:
            await bot._remove_notepad(question)

        duration = time.monotonic() - start

        prediction_value, reasoning = _serialize_prediction_or_record_error(
            prediction,
            question,
            errors=errors,
            forecaster_llm=forecaster_llm,
            parser_llm=parser_llm,
            qid=qid,
        )

        payload = {
            "model": forecaster_llm.model,
            "prediction_value": prediction_value,
            "reasoning": reasoning,
            "errors": errors,
            "ran_at": datetime.now(UTC).isoformat(),
            "duration_seconds": float(duration),
        }
        # A cache-write error (disk full, permissions) is deliberately not caught; it is a fatal environmental signal.
        cache.write_forecaster_output(qid=qid, model_slug=model_slug, payload=payload)
        logger.info(
            "ablation forecaster done | qid=%s | model=%s | duration=%.1fs | errors=%d",
            qid,
            forecaster_llm.model,
            duration,
            len(errors),
        )
        return model_slug, payload


async def run_forecasters_for_question(
    question: MetaculusQuestion,
    research_blob: str,
    cache: AblationCache,
    *,
    forecaster_llms: list[GeneralLlm] | None = None,
    parser_llm: GeneralLlm | None = None,
    force: bool = False,
    per_forecaster_concurrency: int = 4,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> dict[str, dict[str, Any]]:
    """Run all forecasters against one question; cache + return per-model payloads.

    Returns a dict keyed by ``model_slug_filename`` (filesystem-safe slug). A forecaster with a
    cached payload is served from cache unless ``force`` is set; the rest run under
    :func:`patched_window_for_question` so prompt-injected dates anchor to the question's
    mid-window. An expected per-forecaster failure is recorded in that payload's ``errors`` list
    and persisted with ``prediction_value=None`` while the other forecasters continue; a bug
    propagates out of this function instead.
    """
    if forecaster_llms is None:
        forecaster_llms = build_free_forecaster_llms()
    if parser_llm is None:
        parser_llm = build_free_parser_llm()

    qid = question.id_of_question
    if qid is None:
        raise ValueError("Question must have id_of_question for ablation forecasting")

    results: dict[str, dict[str, Any]] = {}
    to_run: list[GeneralLlm] = []
    for llm in forecaster_llms:
        slug = model_slug_to_filename(llm.model)
        if not force:
            cached = cache.read_forecaster_output(qid=qid, model_slug=slug)
            if cached is not None:
                results[slug] = cached
                continue
        to_run.append(llm)

    if to_run:
        semaphore = asyncio.Semaphore(per_forecaster_concurrency)
        async with _get_window_patch_lock():
            # Tools off while caching: both arms reuse the rationales; a leaked "Computed quantities" voids the A/B.
            with patched_window_for_question(question), probabilistic_tools_enabled(enabled=False):
                tasks = [
                    _run_one_forecaster(
                        question,
                        research_blob,
                        llm,
                        parser_llm,
                        cache=cache,
                        semaphore=semaphore,
                        max_retries=max_retries,
                    )
                    for llm in to_run
                ]
                for slug, payload in await asyncio.gather(*tasks):
                    results[slug] = payload

    succeeded = sum(1 for p in results.values() if p["prediction_value"] is not None)
    logger.info(
        "ablation forecaster rollup | qid=%s | succeeded=%d/%d",
        qid,
        succeeded,
        len(forecaster_llms),
    )
    return results


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


async def run_forecasters_batch(
    questions_with_research: list[tuple[MetaculusQuestion, str]],
    cache: AblationCache,
    *,
    forecaster_llms: list[GeneralLlm] | None = None,
    parser_llm: GeneralLlm | None = None,
    force: bool = False,
    per_question_concurrency: int = 2,
    per_forecaster_concurrency: int = 4,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Run forecasters on a batch of (question, research_blob) pairs.

    Returns ``{qid: {model_slug_filename: payload}}``. A question whose runner
    raises is logged at ERROR with its traceback and keyed to an empty dict, so
    one bad question cannot abort a paid run; the other questions still complete.
    """
    if forecaster_llms is None:
        forecaster_llms = build_free_forecaster_llms()
    if parser_llm is None:
        parser_llm = build_free_parser_llm()

    semaphore = asyncio.Semaphore(per_question_concurrency)
    # Late binding through the module attribute so a test monkeypatch of ``run_forecasters_for_question`` is observed.
    from metaculus_bot.ablation import forecasters as _self_module  # noqa: PLC0415, PLW0406  # late-bound patch surface

    async def _run_one(question: MetaculusQuestion, blob: str) -> dict[str, dict[str, Any]]:
        async with semaphore:
            return await _self_module.run_forecasters_for_question(
                question,
                blob,
                cache,
                forecaster_llms=forecaster_llms,
                parser_llm=parser_llm,
                force=force,
                per_forecaster_concurrency=per_forecaster_concurrency,
                max_retries=max_retries,
            )

    tasks = [_run_one(q, blob) for q, blob in questions_with_research]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    results: dict[int, dict[str, dict[str, Any]]] = {}
    for (question, _blob), outcome in zip(questions_with_research, outcomes, strict=True):
        qid = question.id_of_question
        assert qid is not None, "Question must have id_of_question for ablation forecasting"
        if isinstance(outcome, BaseException):
            # Cancellation and operator interrupts are not per-question failures.
            if not isinstance(outcome, Exception):
                raise outcome
            logger.error("ablation forecaster batch | qid=%s failed entirely", qid, exc_info=outcome)
            results[qid] = {}
        else:
            results[qid] = outcome
    return results
