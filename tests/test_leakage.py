"""Tests for the leakage detection module."""

import asyncio
from datetime import datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import MetaculusQuestion

from metaculus_bot.backtest.leakage import (
    _check_single_question_leakage,
    _process_single_question,
    screen_research_for_leakage,
)
from metaculus_bot.backtest.scoring import GroundTruth
from metaculus_bot.constants import LEAKAGE_DETECTOR_MODEL


def _as_questions(questions: list[MagicMock]) -> list[MetaculusQuestion]:
    return cast(list[MetaculusQuestion], questions)


def _make_question(qid: int, text: str = "Will X happen?") -> MagicMock:
    q = MagicMock()
    q.id_of_question = qid
    q.question_text = text
    q.resolution_criteria = "Resolves YES if X happens."
    return q


def _make_ground_truth(qid: int, resolution_string: str = "Yes") -> GroundTruth:
    return GroundTruth(
        question_id=qid,
        question_type="binary",
        resolution=True,
        resolution_string=resolution_string,
        community_prediction=0.6,
        actual_resolution_time=datetime(2025, 6, 1),
        question_text="Will X happen?",
    )


class TestCheckSingleQuestionLeakage:
    @pytest.mark.asyncio
    async def test_detects_leakage_when_llm_says_yes(self):
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = "YES - the research clearly states the outcome."

        question = _make_question(1)
        gt = _make_ground_truth(1)

        result = await _check_single_question_leakage(question, gt, "The question resolved YES.", detector_llm)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_leakage_when_llm_says_no(self):
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = "NO - the research does not reveal the outcome."

        question = _make_question(1)
        gt = _make_ground_truth(1)

        result = await _check_single_question_leakage(question, gt, "Some neutral research text.", detector_llm)
        assert result is False

    @pytest.mark.asyncio
    async def test_case_insensitive_yes_detection(self):
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = "  yes - it leaks."

        question = _make_question(1)
        gt = _make_ground_truth(1)

        result = await _check_single_question_leakage(question, gt, "research text", detector_llm)
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_llm_failure(self):
        detector_llm = AsyncMock()
        detector_llm.invoke.side_effect = RuntimeError("LLM call failed")

        question = _make_question(1)
        gt = _make_ground_truth(1)

        result = await _check_single_question_leakage(question, gt, "research text", detector_llm)
        assert result is False

    @pytest.mark.asyncio
    async def test_prompt_contains_question_and_ground_truth(self):
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = "NO"

        question = _make_question(1, text="Will the sun explode?")
        gt = _make_ground_truth(1, resolution_string="No")
        cast(Any, gt).resolution_criteria = "Resolves YES if sun explodes"

        await _check_single_question_leakage(question, gt, "solar research", detector_llm)

        prompt_sent = detector_llm.invoke.call_args[0][0]
        assert "Will the sun explode?" in prompt_sent
        assert "No" in prompt_sent
        assert "solar research" in prompt_sent


class TestProcessSingleQuestion:
    @pytest.mark.asyncio
    async def test_successful_research_no_leakage(self):
        research_provider = AsyncMock(return_value="Clean research text")
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = "NO - no leakage found."
        semaphore = asyncio.Semaphore(5)

        question = _make_question(42)
        gt = _make_ground_truth(42)

        qid, research_text, is_leaked = await _process_single_question(
            question, gt, research_provider=research_provider, detector_llm=detector_llm, semaphore=semaphore
        )

        assert qid == 42
        assert research_text == "Clean research text"
        assert is_leaked is False

    @pytest.mark.asyncio
    async def test_successful_research_with_leakage(self):
        research_provider = AsyncMock(return_value="The question resolved YES on June 1.")
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = "YES - clearly reveals the outcome."
        semaphore = asyncio.Semaphore(5)

        question = _make_question(42)
        gt = _make_ground_truth(42)

        qid, research_text, is_leaked = await _process_single_question(
            question, gt, research_provider=research_provider, detector_llm=detector_llm, semaphore=semaphore
        )

        assert qid == 42
        assert research_text == "The question resolved YES on June 1."
        assert is_leaked is True

    @pytest.mark.asyncio
    async def test_research_failure_keeps_question(self):
        research_provider = AsyncMock(side_effect=RuntimeError("API down"))
        detector_llm = AsyncMock()
        semaphore = asyncio.Semaphore(5)

        question = _make_question(42)
        gt = _make_ground_truth(42)

        qid, research_text, is_leaked = await _process_single_question(
            question, gt, research_provider=research_provider, detector_llm=detector_llm, semaphore=semaphore
        )

        assert qid == 42
        assert research_text is None
        assert is_leaked is False
        detector_llm.invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_provider_receives_question_object_not_text(self):
        # ResearchCallable expects the MetaculusQuestion object; providers do attribute
        # access (open_time, scheduled_resolution_time) that silently no-ops on a str.
        received: list[Any] = []

        async def recording_provider(arg: Any) -> str:  # test double; awaited by code under test
            received.append(arg)
            return "research"

        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = "NO"
        semaphore = asyncio.Semaphore(5)

        question = _make_question(42, text="Will X resolve?")
        gt = _make_ground_truth(42)

        await _process_single_question(
            question, gt, research_provider=recording_provider, detector_llm=detector_llm, semaphore=semaphore
        )

        assert received == [question]  # the object itself, not question.question_text
        assert received[0].question_text == "Will X resolve?"  # attribute access works on it

    @pytest.mark.asyncio
    async def test_respects_semaphore(self):
        call_order: list[str] = []

        async def slow_research(text: str) -> str:
            call_order.append("start")
            await asyncio.sleep(0.05)
            call_order.append("end")
            return "research"

        research_provider = slow_research
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = "NO"
        semaphore = asyncio.Semaphore(1)

        q1 = _make_question(1)
        q2 = _make_question(2)
        gt1 = _make_ground_truth(1)
        gt2 = _make_ground_truth(2)

        await asyncio.gather(
            _process_single_question(
                q1, gt1, research_provider=research_provider, detector_llm=detector_llm, semaphore=semaphore
            ),
            _process_single_question(
                q2, gt2, research_provider=research_provider, detector_llm=detector_llm, semaphore=semaphore
            ),
        )

        assert call_order == ["start", "end", "start", "end"]


class TestScreenResearchForLeakage:
    @pytest.mark.asyncio
    @patch("metaculus_bot.backtest.leakage.GeneralLlm")
    @patch("metaculus_bot.backtest.leakage.choose_provider_with_name")
    async def test_filters_leaked_questions(self, mock_choose_provider_with_name, mock_llm_class):
        mock_provider = AsyncMock(return_value="some research")
        mock_choose_provider_with_name.return_value = (mock_provider, "mock")

        mock_detector = AsyncMock()
        responses = {
            1: "NO - clean",
            2: "YES - leaks the answer",
            3: "NO - clean",
        }

        async def invoke_side_effect(prompt: str) -> str:  # test double; awaited by code under test
            for qid, resp in responses.items():
                if f"Question {qid}" in prompt or f"question_{qid}" in prompt:
                    return resp
            return "NO"

        mock_detector.invoke = invoke_side_effect
        mock_llm_class.return_value = mock_detector

        questions = [_make_question(1, f"Question {i}") for i in [1, 2, 3]]
        for i, q in enumerate(questions, 1):
            q.id_of_question = i
            q.question_text = f"Question {i}"

        ground_truths = {i: _make_ground_truth(i) for i in [1, 2, 3]}

        clean_questions, clean_gts, research_cache = await screen_research_for_leakage(
            _as_questions(questions), ground_truths, concurrency=5
        )

        assert len(clean_questions) == 2
        clean_ids = {q.id_of_question for q in clean_questions}
        assert 1 in clean_ids
        assert 3 in clean_ids
        assert 2 not in clean_ids

        assert 1 in clean_gts
        assert 3 in clean_gts
        assert 2 not in clean_gts

        assert 1 in research_cache
        assert 3 in research_cache
        assert 2 not in research_cache

    @pytest.mark.asyncio
    @patch("metaculus_bot.backtest.leakage.GeneralLlm")
    @patch("metaculus_bot.backtest.leakage.choose_provider_with_name")
    async def test_all_clean_questions_pass_through(self, mock_choose_provider_with_name, mock_llm_class):
        mock_provider = AsyncMock(return_value="research text")
        mock_choose_provider_with_name.return_value = (mock_provider, "mock")

        mock_detector = AsyncMock()
        mock_detector.invoke.return_value = "NO"
        mock_llm_class.return_value = mock_detector

        questions = [_make_question(i) for i in [1, 2, 3]]
        for i, q in enumerate(questions, 1):
            q.id_of_question = i

        ground_truths = {i: _make_ground_truth(i) for i in [1, 2, 3]}

        clean_questions, clean_gts, research_cache = await screen_research_for_leakage(
            _as_questions(questions), ground_truths, concurrency=5
        )

        assert len(clean_questions) == 3
        assert len(clean_gts) == 3
        assert len(research_cache) == 3

    @pytest.mark.asyncio
    @patch("metaculus_bot.backtest.leakage.GeneralLlm")
    @patch("metaculus_bot.backtest.leakage.choose_provider_with_name")
    async def test_research_failure_keeps_question_no_cache(self, mock_choose_provider_with_name, mock_llm_class):
        call_count = 0

        async def research_side_effect(text: str) -> str:  # test double; awaited by code under test
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("API error")
            return "research text"

        mock_choose_provider_with_name.return_value = (research_side_effect, "mock")

        mock_detector = AsyncMock()
        mock_detector.invoke.return_value = "NO"
        mock_llm_class.return_value = mock_detector

        questions = [_make_question(i) for i in [1, 2, 3]]
        for i, q in enumerate(questions, 1):
            q.id_of_question = i

        ground_truths = {i: _make_ground_truth(i) for i in [1, 2, 3]}

        clean_questions, clean_gts, research_cache = await screen_research_for_leakage(
            _as_questions(questions), ground_truths, concurrency=5
        )

        assert len(clean_questions) == 3
        assert len(clean_gts) == 3
        assert len(research_cache) == 2

    @pytest.mark.asyncio
    @patch("metaculus_bot.backtest.leakage.GeneralLlm")
    @patch("metaculus_bot.backtest.leakage.choose_provider_with_name")
    async def test_choose_provider_called_with_is_benchmarking(self, mock_choose_provider_with_name, mock_llm_class):
        mock_choose_provider_with_name.return_value = (AsyncMock(return_value="research"), "mock")
        mock_detector = AsyncMock()
        mock_detector.invoke.return_value = "NO"
        mock_llm_class.return_value = mock_detector

        questions = [_make_question(1)]
        questions[0].id_of_question = 1
        ground_truths = {1: _make_ground_truth(1)}

        await screen_research_for_leakage(_as_questions(questions), ground_truths)

        mock_choose_provider_with_name.assert_called_once_with(is_benchmarking=True)

    @pytest.mark.asyncio
    @patch("metaculus_bot.backtest.leakage.GeneralLlm")
    @patch("metaculus_bot.backtest.leakage.choose_provider_with_name")
    async def test_detector_llm_built_at_high_effort_with_no_max_tokens(
        self, mock_choose_provider_with_name, mock_llm_class
    ):
        """2026-09-22: the backtest-only leakage screen is not time-sensitive, so it runs at
        effort=high with no max_tokens cap (a cap crashes calls for no good reason since the
        reasoning tokens count against it)."""
        mock_choose_provider_with_name.return_value = (AsyncMock(return_value="research"), "mock")
        mock_detector = AsyncMock()
        mock_detector.invoke.return_value = "NO"
        mock_llm_class.return_value = mock_detector

        questions = [_make_question(1)]
        questions[0].id_of_question = 1
        ground_truths = {1: _make_ground_truth(1)}

        await screen_research_for_leakage(_as_questions(questions), ground_truths)

        mock_llm_class.assert_called_once_with(
            model=LEAKAGE_DETECTOR_MODEL, temperature=None, reasoning={"effort": "high"}
        )

    @pytest.mark.asyncio
    @patch("metaculus_bot.backtest.leakage.GeneralLlm")
    @patch("metaculus_bot.backtest.leakage.choose_provider_with_name")
    async def test_empty_questions_returns_empty(self, mock_choose_provider_with_name, mock_llm_class):
        mock_choose_provider_with_name.return_value = (AsyncMock(return_value="research"), "mock")
        mock_llm_class.return_value = AsyncMock()

        clean_questions, clean_gts, research_cache = await screen_research_for_leakage([], {})

        assert clean_questions == []
        assert clean_gts == {}
        assert research_cache == {}
