"""Tests for the ablation-time mid-window monkey-patch context managers.

The bot's prompts include `_forecasting_window_str(question)`, which calls
`datetime.now()` to interpolate "Today: ..." into every forecasting prompt.
For ablation backtests on resolved 2026 questions, the negative "from now"
deltas tell the forecaster the question already resolved — a leakage vector.

This module patches `_forecasting_window_str` (and the
`{datetime.now(UTC).year}` token in `gap_fill_analyzer_prompt`) for the
duration of a context manager scoped to one specific question.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest
from forecasting_tools import NumericQuestion

from metaculus_bot import prompts as prompts_module
from metaculus_bot.ablation.window_patch import (
    compute_mid_window_today,
    patched_gap_fill_year_for_question,
    patched_gap_fill_year_for_questions,
    patched_window_and_year_for_question,
    patched_window_for_question,
)
from metaculus_bot.research import targeted


def _question(
    *,
    open_time: datetime = datetime(2026, 1, 1),
    scheduled_resolution_time: datetime = datetime(2026, 5, 1),
    id_of_question: int | None = 12345,
    question_text: str = "Q?",
) -> SimpleNamespace:
    """Minimal question stub with the attributes the two patchers read.

    ``question_text`` is what the gap-fill year patcher routes a prompt call on, so it must
    match the ``question_text`` the test then passes to ``gap_fill_analyzer_prompt``.
    """
    return SimpleNamespace(
        open_time=open_time,
        scheduled_resolution_time=scheduled_resolution_time,
        id_of_question=id_of_question,
        question_text=question_text,
    )


# compute_mid_window_today


class TestComputeMidWindowToday:
    def test_returns_midpoint_between_open_and_resolution(self) -> None:
        question = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
        )
        # Jan 1 → May 1 = 120 days; midpoint = Jan 1 + 60d = Mar 2.
        assert compute_mid_window_today(question) == datetime(2026, 3, 2)

    def test_asserts_on_missing_open_time(self) -> None:
        question = _question()
        question.open_time = None
        with pytest.raises(AssertionError):
            compute_mid_window_today(question)

    def test_asserts_on_missing_scheduled_resolution_time(self) -> None:
        question = _question()
        question.scheduled_resolution_time = None
        with pytest.raises(AssertionError):
            compute_mid_window_today(question)


# patched_window_for_question


class TestPatchedWindowForQuestion:
    def test_replaces_today_with_mid_window_date(self) -> None:
        question = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
        )
        with patched_window_for_question(question):
            output = prompts_module._forecasting_window_str(cast(NumericQuestion, question))

        # Mid-window of Jan 1 → May 1 is Mar 2.
        assert "Today: 2026-03-02" in output
        today_str = datetime.now(UTC).strftime("%Y-%m-%d")
        assert f"Today: {today_str}" not in output

    def test_restored_on_normal_exit(self) -> None:
        question = _question()
        before = prompts_module._forecasting_window_str
        with patched_window_for_question(question):
            assert prompts_module._forecasting_window_str is not before
        after = prompts_module._forecasting_window_str
        assert after is before

    def test_restored_on_exception(self) -> None:
        question = _question()
        before = prompts_module._forecasting_window_str

        with pytest.raises(RuntimeError, match="boom"), patched_window_for_question(question):
            raise RuntimeError("boom")

        assert prompts_module._forecasting_window_str is before

    def test_falls_through_for_other_question(self) -> None:
        question_a = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
            id_of_question=11,
        )
        question_b = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
            id_of_question=22,
        )

        with patched_window_for_question(question_a):
            output_b = prompts_module._forecasting_window_str(cast(NumericQuestion, question_b))

        # _forecasting_window_str renders "Today:" in UTC (tz-robust since the ft-0.2.92 prep).
        today_str = datetime.now(UTC).strftime("%Y-%m-%d")
        assert f"Today: {today_str}" in output_b

    def test_falls_through_for_other_question_when_id_is_none(self) -> None:
        """Synthetic test fixtures may have id_of_question=None;
        identity-by-id falls back to id() so distinct objects don't collide."""
        question_a = _question(id_of_question=None)
        question_b = _question(id_of_question=None)

        with patched_window_for_question(question_a):
            output_b = prompts_module._forecasting_window_str(cast(NumericQuestion, question_b))

        today_str = datetime.now(UTC).strftime("%Y-%m-%d")
        assert f"Today: {today_str}" in output_b

    def test_output_structure_matches_original(self) -> None:
        question = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
        )
        with patched_window_for_question(question):
            output = prompts_module._forecasting_window_str(cast(NumericQuestion, question))

        assert output.startswith("Today: ")
        assert "Question opened: 2026-01-01" in output
        assert "Scheduled to resolve: 2026-05-01" in output
        assert "Forecasting window: open date" in output
        assert prompts_module.EVENT_WINDOW_RULE in output
        assert "days ago" in output
        assert "days from now" in output

    def test_re_entrancy_raises(self) -> None:
        question = _question()
        with (
            patched_window_for_question(question),
            pytest.raises(RuntimeError, match="already active"),
            patched_window_for_question(question),
        ):
            pass

    def test_re_entrancy_does_not_corrupt_outer_restore(self) -> None:
        """If an inner enter raises RuntimeError, the outer cleanup must still run."""
        question = _question()
        before = prompts_module._forecasting_window_str

        with (
            patched_window_for_question(question),
            pytest.raises(RuntimeError),
            patched_window_for_question(question),
        ):
            pass

        assert prompts_module._forecasting_window_str is before


# patched_gap_fill_year_for_question


class TestPatchedGapFillYearForQuestion:
    def test_replaces_current_year_in_gap_fill_prompt(self) -> None:
        question = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
        )
        current_year = str(datetime.now(UTC).year)

        with patched_gap_fill_year_for_question(question):
            prompt = prompts_module.gap_fill_analyzer_prompt(
                question_text="Q?",
                resolution_criteria="rc",
                fine_print="fp",
                first_pass_research="fpr",
                is_benchmarking=False,
            )

        # The "no <YEAR> data" leak must NOT contain the actual current year.
        assert f"no {current_year} data" not in prompt
        # Should be replaced with year before the question's resolution year.
        assert "no 2025 data" in prompt

    def test_restored_on_normal_exit(self) -> None:
        question = _question()
        before = prompts_module.gap_fill_analyzer_prompt
        with patched_gap_fill_year_for_question(question):
            assert prompts_module.gap_fill_analyzer_prompt is not before
        assert prompts_module.gap_fill_analyzer_prompt is before

    def test_restored_on_exception(self) -> None:
        question = _question()
        before = prompts_module.gap_fill_analyzer_prompt
        with pytest.raises(RuntimeError, match="boom"), patched_gap_fill_year_for_question(question):
            raise RuntimeError("boom")
        assert prompts_module.gap_fill_analyzer_prompt is before

    def test_threads_through_to_real_prompt_function(self) -> None:
        """The wrapper must still produce a useful prompt — sanity-check
        that the question text and other args still appear."""
        question = _question(question_text="Will BTC hit 200k?")
        with patched_gap_fill_year_for_question(question):
            prompt = prompts_module.gap_fill_analyzer_prompt(
                question_text="Will BTC hit 200k?",
                resolution_criteria="rc",
                fine_print="fp",
                first_pass_research="fpr",
                is_benchmarking=False,
            )
        assert "Will BTC hit 200k?" in prompt

    def test_patched_function_visible_from_targeted_namespace(self) -> None:
        """``research.targeted`` from-imports the prompt at module scope, and
        ``run_gap_fill_pass`` calls that bare name — so the patch must rebind
        the ``targeted`` binding too, or the production call path renders the
        real current year (the bug that made this guard inert on every real
        ablation run)."""
        question = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
        )
        current_year = str(datetime.now(UTC).year)
        before_targeted = targeted.gap_fill_analyzer_prompt

        with patched_gap_fill_year_for_question(question):
            assert targeted.gap_fill_analyzer_prompt is prompts_module.gap_fill_analyzer_prompt
            prompt = targeted.gap_fill_analyzer_prompt(
                question_text="Q?",
                resolution_criteria="rc",
                fine_print="fp",
                first_pass_research="fpr",
                is_benchmarking=False,
            )

        assert f"no {current_year} data" not in prompt
        assert "no 2025 data" in prompt
        # The targeted-namespace binding is restored on exit as well.
        assert targeted.gap_fill_analyzer_prompt is before_targeted

    def test_raises_when_leak_phrase_is_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Zero substitutions means the prompt template and this guard drifted
        apart (rubric item 8 is unconditional) — that must raise, not silently
        leak the real current year into the analyzer prompt."""
        question = _question()
        monkeypatch.setattr(
            prompts_module,
            "gap_fill_analyzer_prompt",
            lambda *_args, **_kwargs: "a template with no stale-info rubric phrase",
        )

        with (
            patched_gap_fill_year_for_question(question),
            pytest.raises(RuntimeError, match="year leak not neutralized"),
        ):
            prompts_module.gap_fill_analyzer_prompt(
                question_text="Q?",
                resolution_criteria="rc",
                fine_print="fp",
                first_pass_research="fpr",
                is_benchmarking=False,
            )


# patched_gap_fill_year_for_questions


class TestPatchedGapFillYearForQuestions:
    """The batch form: one wrapper for the whole batch, routed by ``question_text``."""

    @staticmethod
    def _render(question_text: str, *, positional: bool = False) -> str:
        if positional:
            return prompts_module.gap_fill_analyzer_prompt(
                question_text,
                "rc",
                "fp",
                "fpr",
                is_benchmarking=False,
            )
        return prompts_module.gap_fill_analyzer_prompt(
            question_text=question_text,
            resolution_criteria="rc",
            fine_print="fp",
            first_pass_research="fpr",
            is_benchmarking=False,
        )

    def test_routes_each_question_to_its_own_year(self) -> None:
        resolving_2026 = _question(scheduled_resolution_time=datetime(2026, 5, 1), question_text="Q resolving 2026")
        resolving_2025 = _question(scheduled_resolution_time=datetime(2025, 5, 1), question_text="Q resolving 2025")

        with patched_gap_fill_year_for_questions([resolving_2026, resolving_2025]):
            prompt_2026 = self._render("Q resolving 2026")
            prompt_2025 = self._render("Q resolving 2025")

        assert "no 2025 data" in prompt_2026
        assert "no 2024 data" in prompt_2025

    def test_routes_a_positional_question_text(self) -> None:
        """``question_text`` is the prompt's first parameter, so a positional call must route too."""
        question = _question(scheduled_resolution_time=datetime(2026, 5, 1), question_text="Q positional")

        with patched_gap_fill_year_for_questions([question]):
            prompt = self._render("Q positional", positional=True)

        assert "no 2025 data" in prompt

    def test_raises_for_a_question_outside_the_batch(self) -> None:
        """A prompt call for a question the batch never registered is a harness bug, not a silent leak."""
        question = _question(question_text="Q in the batch")

        with (
            patched_gap_fill_year_for_questions([question]),
            pytest.raises(RuntimeError, match="Question outside the batch"),
        ):
            self._render("Question outside the batch")

    def test_shared_question_text_takes_the_earlier_year(self) -> None:
        """Duplicate texts collapse to the earlier year, which leaks neither question's timing."""
        earlier = _question(scheduled_resolution_time=datetime(2025, 5, 1), question_text="Same text")
        later = _question(scheduled_resolution_time=datetime(2026, 5, 1), question_text="Same text")

        with patched_gap_fill_year_for_questions([later, earlier]):
            prompt = self._render("Same text")

        assert "no 2024 data" in prompt
        assert "no 2025 data" not in prompt

    def test_one_wrapper_serves_both_module_bindings(self) -> None:
        question = _question()
        with patched_gap_fill_year_for_questions([question]):
            assert targeted.gap_fill_analyzer_prompt is prompts_module.gap_fill_analyzer_prompt

    def test_restores_both_bindings_on_exception(self) -> None:
        question = _question()
        before_prompts = prompts_module.gap_fill_analyzer_prompt
        before_targeted = targeted.gap_fill_analyzer_prompt

        with pytest.raises(RuntimeError, match="boom"), patched_gap_fill_year_for_questions([question]):
            raise RuntimeError("boom")

        assert prompts_module.gap_fill_analyzer_prompt is before_prompts
        assert targeted.gap_fill_analyzer_prompt is before_targeted


# patched_window_and_year_for_question


class TestPatchedWindowAndYearForQuestion:
    def test_applies_both_patches(self) -> None:
        question = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
        )
        current_year = str(datetime.now(UTC).year)

        with patched_window_and_year_for_question(question):
            window = prompts_module._forecasting_window_str(cast(NumericQuestion, question))
            gap_prompt = prompts_module.gap_fill_analyzer_prompt(
                question_text="Q?",
                resolution_criteria="rc",
                fine_print="fp",
                first_pass_research="fpr",
                is_benchmarking=False,
            )

        assert "Today: 2026-03-02" in window
        assert f"no {current_year} data" not in gap_prompt
        assert "no 2025 data" in gap_prompt

    def test_restores_both_on_exit(self) -> None:
        question = _question()
        before_window = prompts_module._forecasting_window_str
        before_gap = prompts_module.gap_fill_analyzer_prompt

        with patched_window_and_year_for_question(question):
            pass

        assert prompts_module._forecasting_window_str is before_window
        assert prompts_module.gap_fill_analyzer_prompt is before_gap

    def test_restores_both_on_exception(self) -> None:
        question = _question()
        before_window = prompts_module._forecasting_window_str
        before_gap = prompts_module.gap_fill_analyzer_prompt

        with pytest.raises(RuntimeError, match="boom"), patched_window_and_year_for_question(question):
            raise RuntimeError("boom")

        assert prompts_module._forecasting_window_str is before_window
        assert prompts_module.gap_fill_analyzer_prompt is before_gap
