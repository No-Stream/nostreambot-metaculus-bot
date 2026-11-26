"""Fetch resolved questions from past tournaments and prepare them for backtesting.

Handles question fetching, ground truth extraction, and question sanitization
so bots see clean OPEN-state questions with no resolution info.
"""

import asyncio
import copy
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from forecasting_tools import ApiFilter, MetaculusApi
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    CanceledResolution,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    QuestionState,
)

from metaculus_bot.backtest.scoring import GroundTruth
from metaculus_bot.constants import (
    BACKTEST_DEFAULT_MIN_FORECASTERS,
    BACKTEST_DEFAULT_RESOLVED_AFTER,
    BACKTEST_DEFAULT_TOURNAMENT,
    BACKTEST_OVERFETCH_RATIO,
    FETCH_RETRY_BACKOFFS,
)
from metaculus_bot.http_status import is_transient_question_fetch_error
from metaculus_bot.time_utils import _as_utc

logger: logging.Logger = logging.getLogger(__name__)

# Matches Metaculus's strict-equality count error from
# `MetaculusApi.get_questions_matching_filter`. Captured group is the actual
# resolved-question count the API has on hand. Drives the
# `_fetch_with_retries` insufficient-questions retry path: when the tournament
# holds fewer questions than the over-fetched count, retry once with the
# actual number rather than crashing the entire pipeline.
_INSUFFICIENT_QUESTIONS_RE: re.Pattern[str] = re.compile(
    r"does not match number of questions found \((\d+)\)",
    re.IGNORECASE,
)


def _parse_resolution_window(
    resolved_after: str,
    resolved_before: str | None,
) -> tuple[datetime, datetime | None]:
    """Parse resolution-window boundaries as UTC datetimes."""
    # ft 0.2.92 coerces question datetimes (incl. actual_resolution_time) to tz-aware
    # UTC at construction. The strptime boundaries are naive, so normalize them to UTC
    # too — otherwise the naive/aware ordering comparisons in _filter_and_extract raise.
    resolved_after_dt = _as_utc(datetime.strptime(resolved_after, "%Y-%m-%d"))  # noqa: DTZ007  # stamped UTC by _as_utc
    resolved_before_dt = (
        _as_utc(datetime.strptime(resolved_before, "%Y-%m-%d")) if resolved_before else None  # noqa: DTZ007  # stamped UTC by _as_utc
    )
    return resolved_after_dt, resolved_before_dt


@dataclass
class BacktestQuestionSet:
    questions: list[MetaculusQuestion]
    ground_truths: dict[int, GroundTruth]
    research_cache: dict[int, str] = field(default_factory=dict)
    fetch_metadata: dict[str, Any] = field(default_factory=dict)


async def fetch_resolved_questions(
    total_questions: int,
    *,
    resolved_after: str = BACKTEST_DEFAULT_RESOLVED_AFTER,
    tournament: str = BACKTEST_DEFAULT_TOURNAMENT,
    min_forecasters: int = BACKTEST_DEFAULT_MIN_FORECASTERS,
    resolved_before: str | None = None,
) -> BacktestQuestionSet:
    """Fetch resolved questions from a tournament, extract ground truths, prepare for backtesting.

    Over-fetches by BACKTEST_OVERFETCH_RATIO since the API can't filter by actual_resolution_time.
    Filters locally, then samples down to total_questions.

    Args:
        total_questions: Target number of clean questions to return.
        resolved_after: ISO date "YYYY-MM-DD". Questions with actual_resolution_time
            strictly less than this are excluded; the boundary is INCLUSIVE.
        tournament: Tournament slug to fetch from.
        min_forecasters: Minimum forecasters required (server-side API filter).
        resolved_before: Optional ISO date "YYYY-MM-DD" upper bound. When set, questions
            with actual_resolution_time greater than or equal to this are excluded;
            the boundary is EXCLUSIVE. Default None preserves original behavior
            (no upper bound). Together yields the half-open interval
            [resolved_after, resolved_before).
    """
    resolved_after_dt, resolved_before_dt = _parse_resolution_window(resolved_after, resolved_before)

    overfetch_count = total_questions * BACKTEST_OVERFETCH_RATIO
    logger.info(
        f"Fetching ~{overfetch_count} resolved questions from tournament '{tournament}' "
        f"(target: {total_questions}, resolved after: {resolved_after}, "
        f"resolved before: {resolved_before})"
    )

    raw_questions = await _fetch_with_retries(
        tournament=tournament,
        count=overfetch_count,
        min_forecasters=min_forecasters,
    )

    logger.info(f"API returned {len(raw_questions)} resolved questions")

    filtered_questions, ground_truths, skip_counts = _filter_and_extract(
        raw_questions=raw_questions,
        resolved_after_dt=resolved_after_dt,
        resolved_before_dt=resolved_before_dt,
    )

    logger.info(
        f"After local filtering: {len(filtered_questions)} usable questions "
        f"(skipped: {skip_counts['no_resolution_time']} no resolution time, "
        f"{skip_counts['too_early']} resolved too early, "
        f"{skip_counts['too_late']} resolved too late, "
        f"{skip_counts['canceled']} canceled/ambiguous)"
    )

    # Sample down to target if we have more
    if len(filtered_questions) > total_questions:
        sampled = random.sample(filtered_questions, total_questions)
        sampled_ids = {q.id_of_question for q in sampled}
        ground_truths = {qid: gt for qid, gt in ground_truths.items() if qid in sampled_ids}
        filtered_questions = sampled
    elif len(filtered_questions) < total_questions:
        logger.warning(
            f"Only {len(filtered_questions)} questions available after filtering "
            f"(requested {total_questions}). Proceeding with what's available."
        )

    # Prepare clean questions (strip resolution, set state=OPEN)
    clean_questions = [_prepare_question_for_backtest(q) for q in filtered_questions]
    random.shuffle(clean_questions)

    type_counts: dict[str, int] = {}
    for q in clean_questions:
        q_type = type(q).__name__
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
    logger.info(f"Final backtest question distribution: {type_counts}")

    return BacktestQuestionSet(
        questions=clean_questions,
        ground_truths=ground_truths,
        fetch_metadata={
            "tournament": tournament,
            "resolved_after": resolved_after,
            "resolved_before": resolved_before,
            "total_fetched": len(raw_questions),
            "total_filtered": len(filtered_questions),
            "total_clean": len(clean_questions),
            "type_distribution": type_counts,
            "skipped_no_resolution_time": skip_counts["no_resolution_time"],
            "skipped_too_early": skip_counts["too_early"],
            "skipped_too_late": skip_counts["too_late"],
            "skipped_canceled": skip_counts["canceled"],
        },
    )


def _filter_and_extract(
    raw_questions: list[MetaculusQuestion],
    resolved_after_dt: datetime,
    resolved_before_dt: datetime | None,
) -> tuple[list[MetaculusQuestion], dict[int, GroundTruth], dict[str, int]]:
    """Apply the standard local filtering pipeline used by both fetch entry points.

    Skips questions missing a qid, missing actual_resolution_time, outside the date window,
    or whose ground-truth extraction returns None (canceled/ambiguous/unsupported type).
    Returns (kept_questions, ground_truths_by_qid, skip_counts).
    """
    filtered_questions: list[MetaculusQuestion] = []
    ground_truths: dict[int, GroundTruth] = {}
    skip_counts = {"no_resolution_time": 0, "too_early": 0, "too_late": 0, "canceled": 0}

    for q in raw_questions:
        qid = q.id_of_question
        if qid is None:
            continue

        if q.actual_resolution_time is None:
            skip_counts["no_resolution_time"] += 1
            continue

        # API-sourced datetimes are tz-aware UTC (ft 0.2.92); locally-built test
        # fixtures may still be naive. Normalize to UTC so the window comparisons
        # against the (also-normalized) boundaries never mix naive/aware operands.
        resolved_at = _as_utc(q.actual_resolution_time)

        if resolved_at < resolved_after_dt:
            skip_counts["too_early"] += 1
            continue

        if resolved_before_dt is not None and resolved_at >= resolved_before_dt:
            skip_counts["too_late"] += 1
            continue

        gt = _extract_ground_truth(q)
        if gt is None:
            skip_counts["canceled"] += 1
            continue

        filtered_questions.append(q)
        ground_truths[qid] = gt

    return filtered_questions, ground_truths, skip_counts


@dataclass
class _CollectedQuestions:
    """Deduped questions + ground truths from a stratified multi-tournament fetch."""

    questions: list[MetaculusQuestion]
    ground_truths: dict[int, GroundTruth]
    per_tournament_raw_counts: dict[str, int]
    skip_totals: dict[str, int]


async def _collect_deduped_questions(
    tournaments: list[str],
    *,
    overfetch_count: int,
    min_forecasters: int,
    resolved_after_dt: datetime,
    resolved_before_dt: datetime | None,
) -> _CollectedQuestions:
    """Fetch + filter each tournament serially, deduping by qid (first tournament wins).

    Tournaments are walked serially rather than in parallel: the fetch is over-fetched
    and rate-limited upstream, so there is nothing to gain from fanning out. The
    unconditional checkpoint guarantees this coroutine yields control even when
    ``tournaments`` is empty (defensive — the defaults always populate it).
    """
    await asyncio.sleep(0)

    seen_qids: set[int] = set()
    collected = _CollectedQuestions(
        questions=[],
        ground_truths={},
        per_tournament_raw_counts={},
        skip_totals={"no_resolution_time": 0, "too_early": 0, "too_late": 0, "canceled": 0},
    )

    for tournament in tournaments:
        raw_questions = await _fetch_with_retries(
            tournament=tournament,
            count=overfetch_count,
            min_forecasters=min_forecasters,
        )
        collected.per_tournament_raw_counts[tournament] = len(raw_questions)
        logger.info(f"Tournament '{tournament}': API returned {len(raw_questions)} resolved questions")

        filtered_q, ground_truths, skip_counts = _filter_and_extract(
            raw_questions=raw_questions,
            resolved_after_dt=resolved_after_dt,
            resolved_before_dt=resolved_before_dt,
        )
        for reason, count in skip_counts.items():
            collected.skip_totals[reason] += count
        logger.info(
            f"Tournament '{tournament}': {len(filtered_q)} usable after filtering "
            f"(skipped: {skip_counts['no_resolution_time']} no resolution time, "
            f"{skip_counts['too_early']} too early, "
            f"{skip_counts['too_late']} too late, "
            f"{skip_counts['canceled']} canceled/ambiguous)"
        )

        for question in filtered_q:
            qid = question.id_of_question
            if qid is None or qid in seen_qids:
                continue
            seen_qids.add(qid)
            collected.questions.append(question)
            collected.ground_truths[qid] = ground_truths[qid]

    return collected


def _group_by_question_type(questions: list[MetaculusQuestion]) -> dict[str, list[MetaculusQuestion]]:
    """Bucket questions into the three scoreable types, WARNing on anything else."""
    by_type: dict[str, list[MetaculusQuestion]] = {"binary": [], "multiple_choice": [], "numeric": []}
    for question in questions:
        if isinstance(question, BinaryQuestion):
            by_type["binary"].append(question)
        elif isinstance(question, MultipleChoiceQuestion):
            by_type["multiple_choice"].append(question)
        elif isinstance(question, NumericQuestion):
            by_type["numeric"].append(question)
        else:
            logger.warning(f"Q{question.id_of_question}: unsupported question type {type(question).__name__}")
    return by_type


def _sample_per_type(
    by_type: dict[str, list[MetaculusQuestion]],
    per_type_targets: dict[str, int],
) -> tuple[list[MetaculusQuestion], dict[str, int]]:
    """Sample each type down to its target, WARNing when a bucket is undersaturated.

    An undersaturated bucket contributes everything it has rather than failing, so the
    returned per-type counts can be below the requested targets.
    """
    sampled_questions: list[MetaculusQuestion] = []
    per_type_actual: dict[str, int] = {}
    for q_type, target in per_type_targets.items():
        bucket = by_type[q_type]
        if target == 0:
            per_type_actual[q_type] = 0
            continue
        if len(bucket) < target:
            logger.warning(
                f"Only {len(bucket)} {q_type} questions available after filtering "
                f"(requested {target}). Proceeding with what's available."
            )
            chosen = bucket
        else:
            chosen = random.sample(bucket, target)
        sampled_questions.extend(chosen)
        per_type_actual[q_type] = len(chosen)
    return sampled_questions, per_type_actual


def _type_distribution(questions: list[MetaculusQuestion]) -> dict[str, int]:
    """Count of questions per concrete ft question class."""
    type_counts: dict[str, int] = {}
    for question in questions:
        type_counts[type(question).__name__] = type_counts.get(type(question).__name__, 0) + 1
    return type_counts


async def fetch_resolved_questions_stratified(
    num_binary: int,
    num_multiple_choice: int,
    num_numeric: int,
    *,
    resolved_after: str = BACKTEST_DEFAULT_RESOLVED_AFTER,
    resolved_before: str | None = None,
    tournaments: list[str] | None = None,
    min_forecasters: int = BACKTEST_DEFAULT_MIN_FORECASTERS,
) -> BacktestQuestionSet:
    """Fetch resolved questions stratified by type from one or more tournaments.

    For each tournament in `tournaments`, calls _fetch_with_retries with overfetch
    sized to total_target * BACKTEST_OVERFETCH_RATIO. Then filters by
    actual_resolution_time window and type, samples per-type to the requested counts,
    deduplicates by qid across tournaments (first tournament wins), shuffles, and
    returns a BacktestQuestionSet.

    Reuses _extract_ground_truth and _prepare_question_for_backtest as-is.

    Args:
        num_binary: Target binary count.
        num_multiple_choice: Target MC count.
        num_numeric: Target numeric count.
        resolved_after: ISO date "YYYY-MM-DD". Inclusive lower bound — boundary kept.
        resolved_before: Optional ISO date "YYYY-MM-DD". Exclusive upper bound — boundary excluded.
            Together yields the half-open interval [resolved_after, resolved_before).
        tournaments: List of tournament slugs to union. Defaults to
            [BACKTEST_DEFAULT_TOURNAMENT].
        min_forecasters: Server-side filter for minimum forecaster count.
    """
    if tournaments is None:
        tournaments = [BACKTEST_DEFAULT_TOURNAMENT]

    resolved_after_dt, resolved_before_dt = _parse_resolution_window(resolved_after, resolved_before)

    per_type_targets = {
        "binary": num_binary,
        "multiple_choice": num_multiple_choice,
        "numeric": num_numeric,
    }

    logger.info(
        f"Stratified fetch across {len(tournaments)} tournament(s) "
        f"(targets: binary={num_binary}, mc={num_multiple_choice}, numeric={num_numeric}; "
        f"resolved {resolved_after} < t < {resolved_before})"
    )

    collected = await _collect_deduped_questions(
        tournaments,
        overfetch_count=sum(per_type_targets.values()) * BACKTEST_OVERFETCH_RATIO,
        min_forecasters=min_forecasters,
        resolved_after_dt=resolved_after_dt,
        resolved_before_dt=resolved_before_dt,
    )

    by_type = _group_by_question_type(collected.questions)
    logger.info(
        f"Pre-sample buckets: binary={len(by_type['binary'])}, "
        f"mc={len(by_type['multiple_choice'])}, numeric={len(by_type['numeric'])}"
    )

    sampled_questions, per_type_actual = _sample_per_type(by_type, per_type_targets)
    sampled_ids = {q.id_of_question for q in sampled_questions}
    final_ground_truths = {qid: gt for qid, gt in collected.ground_truths.items() if qid in sampled_ids}

    clean_questions = [_prepare_question_for_backtest(q) for q in sampled_questions]
    random.shuffle(clean_questions)

    type_counts = _type_distribution(clean_questions)
    logger.info(f"Final stratified backtest question distribution: {type_counts}")

    return BacktestQuestionSet(
        questions=clean_questions,
        ground_truths=final_ground_truths,
        fetch_metadata={
            "tournaments": list(tournaments),
            "resolved_after": resolved_after,
            "resolved_before": resolved_before,
            "per_tournament_raw_counts": collected.per_tournament_raw_counts,
            "per_type_targets": per_type_targets,
            "per_type_actual": per_type_actual,
            "total_clean": len(clean_questions),
            "type_distribution": type_counts,
            "skipped_no_resolution_time": collected.skip_totals["no_resolution_time"],
            "skipped_too_early": collected.skip_totals["too_early"],
            "skipped_too_late": collected.skip_totals["too_late"],
            "skipped_canceled": collected.skip_totals["canceled"],
        },
    )


async def _fetch_with_retries(
    tournament: str,
    count: int,
    min_forecasters: int,
) -> list[MetaculusQuestion]:
    """Fetch resolved questions, retrying via the shared benchmark/backtest transient predicate."""
    api_filter = ApiFilter(
        allowed_statuses=["resolved"],
        allowed_tournaments=[tournament],
        num_forecasters_gte=min_forecasters,
        # Pin the type universe to the three types the backtest scoring path
        # (_extract_ground_truth) can handle. Pre-0.2.92 the ApiFilter default
        # excluded conditionals anyway; at 0.2.92 the default admits the new
        # conditional type, which would silently widen the backtest universe with
        # questions extraction can't score. Discrete is folded under numeric
        # (DiscreteQuestion subclasses NumericQuestion), so it needs no separate slot.
        allowed_types=["binary", "multiple_choice", "numeric"],
    )

    attempts = 0
    insufficient_retried = False
    backoffs = list(FETCH_RETRY_BACKOFFS)
    while True:
        retry_sleep_s: int = 0
        try:
            logger.info(f"Attempt {attempts + 1}: fetching {count} resolved questions from tournament '{tournament}'")
            sys.stdout.flush()
            questions = await MetaculusApi.get_questions_matching_filter(
                api_filter,
                num_questions=count,
                randomly_sample=False,
            )
            if not questions:
                raise RuntimeError("API returned 0 resolved questions")
            return questions
        except Exception as e:
            # Strict-equality count mismatch: tournament has fewer resolved questions than we
            # asked for. Retry once with the actual count. This is a different kind of "retryable"
            # from network errors, so it doesn't consume the network-retry budget. The
            # `insufficient_retried` guard prevents infinite loops if the API rejects the
            # smaller count too.
            mismatch = _INSUFFICIENT_QUESTIONS_RE.search(str(e))
            if mismatch and not insufficient_retried:
                actual_count = int(mismatch.group(1))
                if 0 < actual_count < count:
                    logger.warning(
                        "Tournament '%s' has fewer resolved questions than requested "
                        "(%d available < %d requested); retrying with count=%d",
                        tournament,
                        actual_count,
                        count,
                        actual_count,
                    )
                    sys.stdout.flush()
                    insufficient_retried = True
                    count = actual_count
                    continue  # retry without sleep, without consuming retry budget
            if attempts < 2 and is_transient_question_fetch_error(e):
                retry_sleep_s = backoffs[attempts] if attempts < len(backoffs) else backoffs[-1]
                logger.warning(
                    f"Retryable error fetching resolved questions (attempt {attempts + 1}/3): {e}. "
                    f"Backing off {retry_sleep_s}s."
                )
                sys.stdout.flush()
                attempts += 1
            else:
                logger.error(f"Failed to fetch resolved questions: {e}")
                raise RuntimeError(f"Unable to fetch resolved questions after {attempts + 1} attempts") from e
        # Sleep outside the except clause so a cancellation here doesn't discard the active exception.
        await asyncio.sleep(retry_sleep_s)


def _extract_ground_truth(question: MetaculusQuestion) -> GroundTruth | None:
    """Extract ground truth from a resolved question. Returns None for canceled/ambiguous."""
    qid = question.id_of_question
    if qid is None:
        return None

    typed_res = question.typed_resolution
    if typed_res is None:
        return None

    if isinstance(typed_res, CanceledResolution):
        return None

    # Extract community prediction where available.
    # NOTE: Metaculus removed aggregations from their list API, so CP data is no longer
    # available for newly-fetched questions. Resolved tournament questions fetched via the
    # individual question endpoint may still have this data populated.
    if isinstance(question, BinaryQuestion):
        question_type = "binary"
        community_prediction = question.community_prediction_at_access_time
    elif isinstance(question, NumericQuestion):
        question_type = "numeric"
        community_prediction = _extract_numeric_community_cdf(question)
    elif isinstance(question, MultipleChoiceQuestion):
        question_type = "multiple_choice"
        community_prediction = _extract_mc_community_probs(question)
    else:
        logger.warning(f"Q{qid}: unsupported question type {type(question).__name__}")
        return None

    # Date questions resolve to a datetime, but they hit the unsupported-type branch above
    # and return None; binary/numeric/MC resolutions never carry a datetime here. That
    # exclusion is a decision, not an oversight: the live bot forecasts date questions on an
    # epoch-seconds axis, but the backtest, ablation and residual analysis stay date-free
    # until a Mantic date question has resolved under this code.
    assert not isinstance(typed_res, datetime)

    return GroundTruth(
        question_id=qid,
        question_type=question_type,
        resolution=typed_res,
        resolution_string=question.resolution_string or str(typed_res),
        community_prediction=community_prediction,
        actual_resolution_time=question.actual_resolution_time,
        question_text=question.question_text,
        page_url=question.page_url,
    )


def _extract_numeric_community_cdf(question: MetaculusQuestion) -> list[float] | None:
    """Extract numeric community CDF from api_json (same path as scoring_patches.py)."""
    api_json = getattr(question, "api_json", None)
    if not isinstance(api_json, dict):
        return None

    question_obj = api_json.get("question") if isinstance(api_json.get("question"), dict) else api_json
    if not isinstance(question_obj, dict):
        return None

    aggregations = question_obj.get("aggregations")
    if not isinstance(aggregations, dict):
        return None

    rw = aggregations.get("recency_weighted")
    rw_latest = rw.get("latest") if isinstance(rw, dict) else None
    if not isinstance(rw_latest, dict):
        return None

    fv = rw_latest.get("forecast_values")
    if isinstance(fv, list) and len(fv) >= 2:
        try:
            return [float(x) for x in fv]
        except (TypeError, ValueError):
            return None

    return None


def _renormalized_probs(raw_values: list[Any]) -> list[float] | None:
    """Coerce to floats and renormalize to sum 1, or None when a value isn't numeric.

    Metaculus publishes MC aggregations that occasionally sum slightly off 1; a
    non-numeric entry means this aggregation shape isn't usable and the caller should
    fall through to the next one.
    """
    try:
        probs = [float(value) for value in raw_values]
    except (TypeError, ValueError):
        return None
    total = sum(probs)
    if abs(total - 1.0) > 1e-6 and total > 0:
        probs = [p / total for p in probs]
    return probs


def _extract_mc_community_probs(question: MetaculusQuestion) -> list[float] | None:
    """Extract MC community probabilities from api_json (same path as scoring_patches.py)."""
    api_json = getattr(question, "api_json", None)
    if not isinstance(api_json, dict):
        return None

    question_obj = api_json.get("question") if isinstance(api_json.get("question"), dict) else api_json
    if not isinstance(question_obj, dict):
        return None

    options = getattr(question, "options", None)
    if options is None and isinstance(question_obj.get("options"), list):
        options = question_obj.get("options")

    aggregations = question_obj.get("aggregations")
    if not isinstance(aggregations, dict):
        return None

    rw = aggregations.get("recency_weighted")
    rw_latest = rw.get("latest") if isinstance(rw, dict) else None
    if not isinstance(rw_latest, dict):
        return None

    # Try forecast_values first
    fv = rw_latest.get("forecast_values")
    if isinstance(fv, list) and options and isinstance(options, list) and len(fv) == len(options):
        probs = _renormalized_probs(fv)
        if probs is not None:
            return probs

    # Try probability_yes_per_category
    pyc = rw_latest.get("probability_yes_per_category")
    if isinstance(pyc, dict) and options and isinstance(options, list):
        probs = _renormalized_probs([pyc.get(opt, 0.0) for opt in options])
        if probs is not None:
            return probs

    return None


def _prepare_question_for_backtest(question: MetaculusQuestion) -> MetaculusQuestion:
    """Deep copy a resolved question and sanitize it to look like an open question.

    Sets state=OPEN, clears the resolution fields, and blanks background_info to an empty string: the
    prompt templates interpolate it, so None would reach every forecaster as the word "None".
    """
    clean = copy.deepcopy(question)
    clean.state = QuestionState.OPEN
    clean.resolution_string = None
    clean.actual_resolution_time = None
    clean.background_info = ""
    return clean
