"""Gemini-only research for the probabilistic-tools ablation benchmark.

This module calls ``gemini_search_provider`` directly, runs a bounded second pass via
``run_gap_fill_pass``, and persists the concatenated blob in ``AblationCache``.

Concatenation format and gap-fill threshold mirror ``main.run_research`` so
downstream code paths see the same shape they'd see in production.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from forecasting_tools import MetaculusQuestion

from metaculus_bot import constants
from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.window_patch import patched_gap_fill_year_for_questions
from metaculus_bot.constants import GAP_FILL_MIN_RESEARCH_CHARS, GEMINI_SEARCH_DEFAULT_MODEL
from metaculus_bot.research import targeted
from metaculus_bot.research.gemini_search import gemini_search_provider
from metaculus_bot.research.targeted import run_gap_fill_pass

__all__ = ["run_gemini_only_research", "run_gemini_research_for_qids"]

logger: logging.Logger = logging.getLogger(__name__)

_GAP_FILL_HEADER = "\n\n---\n\n## Targeted Gap-Fill (second pass)\n\n"


@contextmanager
def _patched_gap_fill_max_gaps(value: int) -> Iterator[None]:
    """Override ``GAP_FILL_MAX_GAPS`` in both ``constants`` and ``research.targeted``.

    ``research.targeted`` does ``from metaculus_bot.constants import GAP_FILL_MAX_GAPS``
    at import time, binding the integer in its own namespace; patching only
    ``constants`` would leave the analyzer call seeing the old value. Patch both
    and restore both in ``finally`` so tests don't leak state across the suite.
    """
    original_constants = constants.GAP_FILL_MAX_GAPS
    original_tr = targeted.GAP_FILL_MAX_GAPS
    constants.GAP_FILL_MAX_GAPS = value
    targeted.GAP_FILL_MAX_GAPS = value
    try:
        yield
    finally:
        constants.GAP_FILL_MAX_GAPS = original_constants
        targeted.GAP_FILL_MAX_GAPS = original_tr


def _count_gap_sections(addendum: str) -> int:
    """Count ``### Gap N:`` section headers in the gap-fill addendum.

    ``run_gap_fill_pass`` emits one ``### Gap <idx>: <text>`` per resolved gap.
    Counting headers is the only way to recover the actual gap count from the
    string output without re-running the analyzer.
    """
    return sum(1 for line in addendum.splitlines() if line.startswith("### Gap "))


def _build_meta(
    *,
    first_pass: str,
    gap_fill: str,
    gap_fill_used: bool,
    gap_fill_enabled: bool,
    gap_fill_max_gaps: int,
    is_benchmarking: bool,
    gemini_model: str,
) -> dict:
    return {
        "gemini_search_used": True,
        "gap_fill_used": gap_fill_used,
        "gap_fill_enabled": gap_fill_enabled,
        "gap_count": _count_gap_sections(gap_fill) if gap_fill_used else 0,
        "first_pass_chars": len(first_pass),
        "gap_fill_chars": len(gap_fill),
        "researched_at": datetime.now(UTC).isoformat(),
        "gemini_model": gemini_model,
        "gap_fill_max_gaps": gap_fill_max_gaps,
        "is_benchmarking": is_benchmarking,
    }


async def run_gemini_only_research(
    question: MetaculusQuestion,
    cache: AblationCache,
    *,
    gap_fill_max_gaps: int = 3,
    is_benchmarking: bool = True,
    force: bool = False,
    gemini_model: str | None = None,
    enable_gap_fill: bool = True,
) -> tuple[str, dict]:
    """Run Gemini grounded search + bounded gap-fill, cached on disk.

    A cache hit (with ``force=False``) returns without API calls. A primary Gemini
    failure re-raises, so the caller decides whether to drop the qid. A failed
    gap-fill caches the first-pass blob alone with ``gap_fill_used=False``:
    ``run_gap_fill_pass`` absorbs its own provider and API failures and returns
    ``""`` (production semantics), so anything it does raise is a bug and propagates.
    ``gemini_model`` reaches the provider as ``model_slug``, which wins over the
    ``GEMINI_SEARCH_MODEL`` env var (the CLI flag is canonical), and ``enable_gap_fill=False``
    skips the second pass entirely; meta records ``gap_fill_enabled`` beside ``gap_fill_used``
    so cached blobs are self-describing. The module-global patches the gap-fill pass needs (the
    ``GAP_FILL_MAX_GAPS`` value and the analyzer-prompt year rewrite) belong to
    ``run_gemini_research_for_qids``, which holds them for the whole batch; ``gap_fill_max_gaps``
    here only records what the batch requested in the meta payload.
    """
    qid = question.id_of_question
    assert qid is not None, "MetaculusQuestion must have id_of_question set"

    if not force:
        cached = cache.read_research(qid)
        if cached is not None:
            logger.info(f"Cache HIT for qid {qid}")
            await asyncio.sleep(0)
            return cached

    logger.info(f"Cache MISS for qid {qid}, fetching...")

    effective_model = gemini_model or GEMINI_SEARCH_DEFAULT_MODEL

    provider = gemini_search_provider(model_slug=gemini_model, is_benchmarking=is_benchmarking)
    first_pass = await provider(question)

    gap_fill_blob = ""
    gap_fill_used = False

    if enable_gap_fill and len(first_pass.strip()) >= GAP_FILL_MIN_RESEARCH_CHARS:
        gap_fill_blob = await run_gap_fill_pass(
            question,
            first_pass,
            is_benchmarking=is_benchmarking,
        )
        gap_fill_used = bool(gap_fill_blob)
        logger.info(f"Gap-fill returned {len(gap_fill_blob)} chars for qid {qid}")

    blob = f"{first_pass}{_GAP_FILL_HEADER}{gap_fill_blob}" if gap_fill_blob else first_pass

    meta = _build_meta(
        first_pass=first_pass,
        gap_fill=gap_fill_blob,
        gap_fill_used=gap_fill_used,
        gap_fill_enabled=enable_gap_fill,
        gap_fill_max_gaps=gap_fill_max_gaps,
        is_benchmarking=is_benchmarking,
        gemini_model=effective_model,
    )

    cache.write_research(qid, blob, meta)
    return blob, meta


async def run_gemini_research_for_qids(
    questions: list[MetaculusQuestion],
    cache: AblationCache,
    *,
    gap_fill_max_gaps: int = 3,
    is_benchmarking: bool = True,
    force: bool = False,
    concurrency: int = 4,
    gemini_model: str | None = None,
    enable_gap_fill: bool = True,
) -> dict[int, tuple[str, dict] | None]:
    """Run ``run_gemini_only_research`` per question under a semaphore.

    Each question is its own failure boundary: one that raises caches nothing,
    is logged with its traceback and surfaces as ``None`` in the result dict,
    while the other questions of the paid run still complete. A cancelled child is
    the loop shutting down rather than a question failing, so it propagates.

    The gap-fill pass reads two module globals, ``GAP_FILL_MAX_GAPS`` and
    ``gap_fill_analyzer_prompt``; both are patched ONCE here around the whole gather,
    because a per-question patch of a module global is not concurrency-safe (see
    ``patched_gap_fill_year_for_questions``).
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(question: MetaculusQuestion) -> tuple[str, dict]:
        async with semaphore:
            return await run_gemini_only_research(
                question,
                cache,
                gap_fill_max_gaps=gap_fill_max_gaps,
                is_benchmarking=is_benchmarking,
                force=force,
                gemini_model=gemini_model,
                enable_gap_fill=enable_gap_fill,
            )

    with (
        patched_gap_fill_year_for_questions(questions),
        _patched_gap_fill_max_gaps(gap_fill_max_gaps),
    ):
        outcomes = await asyncio.gather(*(_one(q) for q in questions), return_exceptions=True)

    results: dict[int, tuple[str, dict] | None] = {}
    for question, outcome in zip(questions, outcomes, strict=True):
        qid = question.id_of_question
        assert qid is not None, "MetaculusQuestion must have id_of_question set"
        if isinstance(outcome, BaseException):
            if not isinstance(outcome, Exception):
                raise outcome
            logger.error(f"Research failed for qid {qid}", exc_info=outcome)
            results[qid] = None
        else:
            results[qid] = outcome
    return results
