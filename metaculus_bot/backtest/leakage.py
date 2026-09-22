"""Research pre-screening: run research, detect leakage, cache clean results."""

import asyncio
import logging
from typing import Any

from forecasting_tools import GeneralLlm, MetaculusQuestion

from metaculus_bot.backtest.scoring import GroundTruth
from metaculus_bot.constants import ASKNEWS_MAX_CONCURRENCY, LEAKAGE_DETECTOR_MODEL
from metaculus_bot.research.providers import choose_provider_with_name

logger: logging.Logger = logging.getLogger(__name__)


def _question_id(question: MetaculusQuestion) -> int:
    qid = question.id_of_question
    if qid is None:
        raise ValueError("MetaculusQuestion is missing id_of_question")
    return qid


async def screen_research_for_leakage(
    questions: list[MetaculusQuestion],
    ground_truths: dict[int, GroundTruth],
    concurrency: int = ASKNEWS_MAX_CONCURRENCY,
) -> tuple[list[MetaculusQuestion], dict[int, GroundTruth], dict[int, str]]:
    research_provider, _ = choose_provider_with_name(is_benchmarking=True)
    # temperature=None defers reasoning models to provider defaults; redundant on
    # ft 0.2.92 (GeneralLlm ctor default is already None). No top_p. 2026-09-22: effort
    # low->max and max_tokens=500 removed (operator: this backtest-only screen is not
    # time-sensitive, and a max_tokens cap crashes calls for no good reason since the
    # reasoning tokens count against it).
    detector_llm = GeneralLlm(model=LEAKAGE_DETECTOR_MODEL, temperature=None, reasoning={"effort": "max"})
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        _process_single_question(
            q,
            ground_truths[_question_id(q)],
            research_provider=research_provider,
            detector_llm=detector_llm,
            semaphore=semaphore,
        )
        for q in questions
    ]
    results = await asyncio.gather(*tasks)

    leaked_ids: set[int] = set()
    research_cache: dict[int, str] = {}

    for qid, research_text, is_leaked in results:
        if is_leaked:
            leaked_ids.add(qid)
        elif research_text is not None:
            research_cache[qid] = research_text

    clean_questions = [q for q in questions if q.id_of_question not in leaked_ids]
    clean_ground_truths = {qid: gt for qid, gt in ground_truths.items() if qid not in leaked_ids}

    total = len(questions)
    excluded = len(leaked_ids)
    leakage_rate_pct = (excluded / total * 100) if total > 0 else 0.0
    logger.info(f"{excluded}/{total} questions excluded due to research leakage ({leakage_rate_pct:.0f}% leakage rate)")

    if leakage_rate_pct > 50:
        logger.warning(
            f"High leakage rate: {leakage_rate_pct:.0f}% of questions had research leakage. "
            f"Research provider may be returning resolution information."
        )

    return clean_questions, clean_ground_truths, research_cache


async def _process_single_question(
    question: MetaculusQuestion,
    ground_truth: GroundTruth,
    *,
    research_provider: Any,
    detector_llm: GeneralLlm,
    semaphore: asyncio.Semaphore,
) -> tuple[int, str | None, bool]:
    qid = _question_id(question)
    try:
        async with semaphore:
            # ResearchCallable expects the MetaculusQuestion object, not its text —
            # providers do isinstance checks and attribute access (e.g. the ts-anchor
            # provider reads question.open_time / scheduled_resolution_time), which
            # silently no-op on a bare string.
            research_text = await research_provider(question)
    except Exception:  # noqa: BLE001  # soft-fail boundary: any research-provider failure must not drop the question from the backtest
        logger.warning(f"Research failed for Q{qid}, keeping question without cached research")
        return (qid, None, False)

    is_leaked = await _check_single_question_leakage(question, ground_truth, research_text, detector_llm)
    return (qid, research_text, is_leaked)


async def _check_single_question_leakage(
    question: MetaculusQuestion,
    ground_truth: GroundTruth,
    research_text: str,
    detector_llm: GeneralLlm,
) -> bool:
    prompt = (
        "You are checking whether a research report reveals the actual outcome of a forecasting question. "
        "The question has already resolved, and we need to know if the research contains this information.\n\n"
        f"Question: {question.question_text}\n"
        f"Resolution criteria: {question.resolution_criteria}\n"
        f"Actual resolution: {ground_truth.resolution_string}\n\n"
        "The research report is provided below within <research> tags. "
        "Does it contain information that clearly reveals or strongly implies "
        "the actual resolution? Answer YES or NO, then briefly explain.\n\n"
        f"<research>\n{research_text}\n</research>"
    )
    try:
        response = await detector_llm.invoke(prompt)
        return response.strip().upper().startswith("YES")
    except Exception:  # noqa: BLE001  # soft-fail boundary: a detector-LLM failure must not drop the question; keep it and let scoring see it
        logger.warning(f"Leakage check failed for Q{question.id_of_question}, conservatively keeping question")
        return False
