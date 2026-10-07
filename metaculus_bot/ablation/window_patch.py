"""Context managers that anchor prompt-injected dates to a question's mid-window.

For ablation backtests on resolved questions, the production prompts'
``_forecasting_window_str`` reveals the resolution status by computing
"days from now" against ``datetime.now()``. These helpers monkey-patch
the prompt builders and restore the originals on exit: the window patch
for the duration of one question's forecast, the gap-fill year patch for
a whole concurrent batch, routed per call.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot import prompts as prompts_module
from metaculus_bot.research import targeted

__all__ = [
    "compute_mid_window_today",
    "patched_gap_fill_year_for_question",
    "patched_gap_fill_year_for_questions",
    "patched_window_and_year_for_question",
    "patched_window_for_question",
]


def compute_mid_window_today(question: MetaculusQuestion | Any) -> datetime:
    """Return the mid-window datetime for ``question``.

    Mid-window = ``open_time + (scheduled_resolution_time - open_time) / 2``.
    Both timestamps are required; absence is a data bug, not a graceful path.
    """
    assert question.open_time is not None, "question.open_time is required"
    assert question.scheduled_resolution_time is not None, "question.scheduled_resolution_time is required"
    delta = question.scheduled_resolution_time - question.open_time
    return question.open_time + delta / 2


def _question_identity(question: Any) -> tuple[str, int]:
    """Hashable identity for question-routing in the patched function.

    Real questions populate ``id_of_question``; synthetic test fixtures
    sometimes leave it ``None`` and we fall back to ``id(question)`` so
    distinct objects don't collide.
    """
    qid = getattr(question, "id_of_question", None)
    if qid is None:
        return ("py_id", id(question))
    return ("metaculus_id", int(qid))


_window_patch_active: bool = False


@contextmanager
def patched_window_for_question(question: MetaculusQuestion | Any) -> Iterator[None]:
    """Monkey-patch ``_forecasting_window_str`` to anchor today on ``question``'s mid-window.

    Calls with a different question fall through to the original ``datetime.now()``
    implementation. Re-entrancy is unsupported and raises ``RuntimeError``.
    """
    global _window_patch_active
    if _window_patch_active:
        raise RuntimeError("patched_window_for_question is already active; nested patches unsupported")

    original = prompts_module._forecasting_window_str
    target_identity = _question_identity(question)
    mid_window = compute_mid_window_today(question)

    def _patched(q: Any) -> str:
        if _question_identity(q) != target_identity:
            return original(q)
        return prompts_module.forecasting_window_at(q, mid_window)

    prompts_module._forecasting_window_str = _patched
    _window_patch_active = True
    try:
        yield
    finally:
        prompts_module._forecasting_window_str = original
        _window_patch_active = False


def _replacement_years_by_question_text(questions: Sequence[MetaculusQuestion | Any]) -> dict[str, int]:
    """Map each question's text to the year its analyzer prompt should claim as "now".

    ``scheduled_resolution_time.year - 1`` cannot leak the resolution timing. Two questions
    sharing a text take the earlier year, which leaks neither one's.
    """
    years: dict[str, int] = {}
    for question in questions:
        assert question.scheduled_resolution_time is not None, "question.scheduled_resolution_time is required"
        year = question.scheduled_resolution_time.year - 1
        years[question.question_text] = min(year, years.get(question.question_text, year))
    return years


@contextmanager
def patched_gap_fill_year_for_questions(questions: Sequence[MetaculusQuestion | Any]) -> Iterator[None]:
    """Patch ``gap_fill_analyzer_prompt`` for a BATCH, neutralizing the ``{datetime.now(UTC).year}`` leak.

    The analyzer prompt interpolates the current year into a "stale info" rubric ("e.g., no
    2026 data on a near-term question"), which tells the forecaster the question has resolved;
    each question's prompt is rewritten to its own replacement year instead. ONE wrapper serves
    the whole batch, routed on the ``question_text`` argument, because a batch runs
    concurrently: a per-question patch captured the previous question's wrapper as its original,
    so the later rewrite found nothing left to match, and a non-LIFO exit left the globals
    stale. Either raise below lands in ``run_gap_fill_pass``'s soft-fail, which logs
    GAP_FILL_ANALYZER_FAILED and continues the question without gap-fill.

    ``research.targeted`` from-imports the prompt at module scope, so patching only ``prompts``
    would leave ``run_gap_fill_pass`` calling an un-intercepted copy; both bindings are patched
    and restored in ``finally`` (as ``_patched_gap_fill_max_gaps`` in ``ablation/research.py``).
    """
    replacement_years = _replacement_years_by_question_text(questions)
    original = prompts_module.gap_fill_analyzer_prompt
    original_targeted = targeted.gap_fill_analyzer_prompt

    def _patched(question_text: str, *args: Any, **kwargs: Any) -> str:
        replacement_year = replacement_years.get(question_text)
        if replacement_year is None:
            raise RuntimeError(f"gap-fill year patch has no question in this batch matching {question_text!r}")
        rendered = original(question_text, *args, **kwargs)
        # Rubric item 8 renders the UTC year, so that is the year to look for.
        pattern = rf"\bno {datetime.now(UTC).year} data\b"
        rendered, substitutions = re.subn(pattern, f"no {replacement_year} data", rendered)
        if substitutions == 0:
            # Rubric item 8 is unconditional, so no match means template drift; raising beats leaking.
            raise RuntimeError(f"gap-fill year leak not neutralized: {pattern!r} did not match the analyzer prompt")
        return rendered

    prompts_module.gap_fill_analyzer_prompt = _patched
    targeted.gap_fill_analyzer_prompt = _patched
    try:
        yield
    finally:
        prompts_module.gap_fill_analyzer_prompt = original
        targeted.gap_fill_analyzer_prompt = original_targeted


@contextmanager
def patched_gap_fill_year_for_question(question: MetaculusQuestion | Any) -> Iterator[None]:
    """Batch-of-one form of ``patched_gap_fill_year_for_questions``."""
    with patched_gap_fill_year_for_questions([question]):
        yield


@contextmanager
def patched_window_and_year_for_question(question: MetaculusQuestion | Any) -> Iterator[None]:
    """Apply both ``patched_window_for_question`` and ``patched_gap_fill_year_for_question``."""
    with patched_window_for_question(question), patched_gap_fill_year_for_question(question):
        yield
