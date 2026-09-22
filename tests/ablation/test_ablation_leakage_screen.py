"""Tests for the ablation leakage screen.

Operates on the cached full research blob (first-pass + gap-fill addendum). Caches
the verdict per qid, and conservatively drops the question on detector failure
(unlike the production screen in metaculus_bot.backtest.leakage which keeps
the question).
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools.data_models.questions import MetaculusQuestion
from litellm.exceptions import APIConnectionError

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.leakage_screen import (
    DEFAULT_DETECTOR_MODEL,
    screen_batch,
    screen_research_blob,
)
from metaculus_bot.backtest.scoring import GroundTruth


def _transient_api_error(message: str) -> APIConnectionError:
    """Realistic transient detector failure — what an actual provider blip raises.

    The detector retry loop only catches the narrow transient set
    (``litellm.APIError`` + ``asyncio.TimeoutError`` + ``ValueError``); using
    a real ``litellm`` exception keeps these tests aligned with what the
    runtime actually sees, instead of papering over a buggy ``RuntimeError``.
    """
    return APIConnectionError(message=message, llm_provider="z-ai", model="glm-4.5-air:free")


def _blob_sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _make_question(qid: int, text: str = "Will X happen?") -> MetaculusQuestion:
    q = MagicMock()
    q.id_of_question = qid
    q.question_text = text
    q.resolution_criteria = "Resolves YES if X happens."
    return cast(MetaculusQuestion, q)


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


@pytest.fixture(autouse=True)
def _shrink_detector_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speed up the retry-sleep so tests stay fast.

    The production constant (~2s) is sized to give Z.AI a chance to recover
    from a brief outage; in tests we just need to verify the retry happens.
    """
    monkeypatch.setattr("metaculus_bot.ablation.leakage_screen._DETECTOR_RETRY_BACKOFF_SECONDS", 0.0)


# ---------------------------------------------------------------------------
# screen_research_blob
# ---------------------------------------------------------------------------


class TestScreenResearchBlob:
    def test_default_detector_model_is_free_glm_air(self) -> None:
        assert DEFAULT_DETECTOR_MODEL == "openrouter/z-ai/glm-4.5-air:free"

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_verdict(self, cache: AblationCache) -> None:
        cached_payload = {
            "is_leaked": False,
            "detector_response": "NO - clean.",
            "detector_model": "openrouter/z-ai/glm-4.5-air:free",
            "detector_failed": False,
            "screened_at": "2026-05-12T10:00:00",
        }
        cache.write_leakage_screen(qid=42, payload=cached_payload)

        detector_llm = AsyncMock()
        question = _make_question(42)
        gt = _make_ground_truth(42)

        verdict = await screen_research_blob(question, gt, "any research blob", cache, detector_llm=detector_llm)

        assert verdict["is_leaked"] is False
        assert verdict["detector_response"] == "NO - clean."
        assert verdict["detector_failed"] is False
        detector_llm.invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_true_bypasses_cache(self, cache: AblationCache) -> None:
        cached_payload = {
            "is_leaked": False,
            "detector_response": "NO - stale verdict.",
            "detector_model": "openrouter/z-ai/glm-4.5-air:free",
            "detector_failed": False,
            "screened_at": "2026-05-12T10:00:00",
        }
        cache.write_leakage_screen(qid=42, payload=cached_payload)

        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = '{"is_leaked": true, "explanation": "the article reveals the outcome"}'

        question = _make_question(42)
        gt = _make_ground_truth(42)

        verdict = await screen_research_blob(
            question, gt, "Some research that leaks.", cache, detector_llm=detector_llm, force=True
        )

        assert verdict["is_leaked"] is True
        assert "is_leaked" in verdict["detector_response"]
        detector_llm.invoke.assert_called_once()

        on_disk = cache.read_leakage_screen(qid=42)
        assert on_disk is not None
        assert on_disk["is_leaked"] is True

    @pytest.mark.asyncio
    async def test_clean_research_caches_not_leaked(self, cache: AblationCache) -> None:
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = '{"is_leaked": false, "explanation": "nothing in here"}'

        question = _make_question(42)
        gt = _make_ground_truth(42)

        verdict = await screen_research_blob(question, gt, "Some clean research.", cache, detector_llm=detector_llm)

        assert verdict["is_leaked"] is False
        assert verdict["detector_failed"] is False
        assert verdict["detector_response"] == '{"is_leaked": false, "explanation": "nothing in here"}'
        assert verdict["detector_model"] == DEFAULT_DETECTOR_MODEL

        on_disk = cache.read_leakage_screen(qid=42)
        assert on_disk is not None
        assert on_disk["is_leaked"] is False
        assert on_disk["detector_failed"] is False
        assert on_disk["cache_schema_version"] == 1

    @pytest.mark.asyncio
    async def test_leaked_research_caches_leaked(self, cache: AblationCache) -> None:
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = (
            '{"is_leaked": true, "explanation": "the article literally states X resolved YES"}'
        )

        question = _make_question(42)
        gt = _make_ground_truth(42)

        verdict = await screen_research_blob(
            question, gt, "The article: X resolved YES on June 1.", cache, detector_llm=detector_llm
        )

        assert verdict["is_leaked"] is True
        assert verdict["detector_failed"] is False
        assert "is_leaked" in verdict["detector_response"]

        on_disk = cache.read_leakage_screen(qid=42)
        assert on_disk is not None
        assert on_disk["is_leaked"] is True

    @pytest.mark.asyncio
    async def test_markdown_decorated_yes_is_correctly_parsed(self, cache: AblationCache) -> None:
        """Regression: Q43131 returned ``**Answer: YES**`` (markdown bold). The old
        prefix matcher (response.strip().upper().startswith("YES")) saw
        ``**ANSWER: YES**`` and returned False — leaving a leaky question undropped.

        With structured JSON output, markdown decorations inside the explanation
        field are inert; only ``is_leaked`` matters.
        """
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = (
            '{"is_leaked": true, "explanation": "**Answer: YES** the answer is revealed"}'
        )

        question = _make_question(42)
        gt = _make_ground_truth(42)

        verdict = await screen_research_blob(
            question, gt, "Research blob with implication leak.", cache, detector_llm=detector_llm
        )

        assert verdict["is_leaked"] is True
        assert verdict["detector_failed"] is False

    @pytest.mark.asyncio
    async def test_invalid_json_is_conservatively_dropped(self, cache: AblationCache) -> None:
        """If the LLM emits free-form prose with no JSON, conservatively drop."""
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = "Sorry, I can't parse this... YES it's leaked though"

        question = _make_question(42)
        gt = _make_ground_truth(42)

        verdict = await screen_research_blob(question, gt, "Research blob.", cache, detector_llm=detector_llm)

        assert verdict["is_leaked"] is True
        assert verdict["detector_failed"] is True

    @pytest.mark.asyncio
    async def test_json_with_extra_prose_is_extracted(self, cache: AblationCache) -> None:
        """If the LLM wraps the JSON in conversational prose, extract via regex fallback."""
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = (
            'Here\'s my analysis:\n{"is_leaked": false, "explanation": "background only"}'
        )

        question = _make_question(42)
        gt = _make_ground_truth(42)

        verdict = await screen_research_blob(question, gt, "Research blob.", cache, detector_llm=detector_llm)

        assert verdict["is_leaked"] is False
        assert verdict["detector_failed"] is False

    @pytest.mark.asyncio
    async def test_detector_failure_returns_is_leaked_true_conservatively(self, cache: AblationCache) -> None:
        detector_llm = AsyncMock()
        detector_llm.invoke.side_effect = _transient_api_error("network")

        question = _make_question(42)
        gt = _make_ground_truth(42)

        verdict = await screen_research_blob(question, gt, "Some research.", cache, detector_llm=detector_llm)

        assert verdict["is_leaked"] is True
        assert verdict["detector_failed"] is True
        assert verdict["detector_response"] == "<failed>"

        on_disk = cache.read_leakage_screen(qid=42)
        assert on_disk is not None
        assert on_disk["is_leaked"] is True
        assert on_disk["detector_failed"] is True

    def test_extract_is_leaked_accepts_string_true(self) -> None:
        """Some free-tier providers emit ``"true"``/``"false"`` strings inside JSON
        when the model is sloppy about quoting. The extractor must accept these
        case-insensitively rather than dropping the question to detector_failed."""
        from metaculus_bot.ablation.leakage_screen import _extract_is_leaked

        assert _extract_is_leaked('{"is_leaked": "true", "explanation": "x"}') is True

    def test_extract_is_leaked_accepts_string_false_case_insensitive(self) -> None:
        from metaculus_bot.ablation.leakage_screen import _extract_is_leaked

        assert _extract_is_leaked('{"is_leaked": "False", "explanation": "x"}') is False

    def test_extract_is_leaked_accepts_yes_no_strings(self) -> None:
        from metaculus_bot.ablation.leakage_screen import _extract_is_leaked

        assert _extract_is_leaked('{"is_leaked": "yes", "explanation": "x"}') is True
        assert _extract_is_leaked('{"is_leaked": "NO", "explanation": "x"}') is False

    def test_extract_is_leaked_rejects_garbage_string(self) -> None:
        """Strings that aren't a recognised true/false form raise so the caller
        falls through to detector_failed=True (conservative drop)."""
        from metaculus_bot.ablation.leakage_screen import _extract_is_leaked

        with pytest.raises(ValueError, match=r"'is_leaked' string must be true/false/yes/no"):
            _extract_is_leaked('{"is_leaked": "maybe", "explanation": "x"}')

    def test_extract_is_leaked_still_accepts_real_bools(self) -> None:
        """The bool-form path must still work — string acceptance is additive."""
        from metaculus_bot.ablation.leakage_screen import _extract_is_leaked

        assert _extract_is_leaked('{"is_leaked": true, "explanation": "x"}') is True
        assert _extract_is_leaked('{"is_leaked": false, "explanation": "x"}') is False

    @pytest.mark.asyncio
    async def test_transient_failure_then_success_retries_once(self, cache: AblationCache) -> None:
        """A single transient detector failure must be retried, not fall through
        to detector_failed=True.

        Z.AI's free tier occasionally has 5-10 minute outages; without an
        in-call retry, a brief blip drops a chunk of questions to
        is_leaked=True (conservative) when re-screen later would have
        succeeded. One retry on transient errors is cheap insurance —
        the detector is a free model.
        """
        detector_llm = AsyncMock()
        detector_llm.invoke = AsyncMock(
            side_effect=[
                _transient_api_error("transient blip"),
                '{"is_leaked": false, "explanation": "clean on retry"}',
            ]
        )

        question = _make_question(99)
        gt = _make_ground_truth(99)

        verdict = await screen_research_blob(question, gt, "research blob", cache, detector_llm=detector_llm)

        assert verdict["is_leaked"] is False
        assert verdict["detector_failed"] is False
        assert detector_llm.invoke.await_count == 2

    @pytest.mark.asyncio
    async def test_two_transient_failures_falls_through_to_detector_failed(self, cache: AblationCache) -> None:
        """If BOTH attempts fail, the detector is conservatively marked failed."""
        detector_llm = AsyncMock()
        detector_llm.invoke = AsyncMock(side_effect=_transient_api_error("persistent failure"))

        question = _make_question(98)
        gt = _make_ground_truth(98)

        verdict = await screen_research_blob(question, gt, "research blob", cache, detector_llm=detector_llm)

        assert verdict["is_leaked"] is True
        assert verdict["detector_failed"] is True
        assert detector_llm.invoke.await_count == 2

    @pytest.mark.asyncio
    async def test_real_bug_propagates_instead_of_being_swallowed(self, cache: AblationCache) -> None:
        """Fail-fast: a programming bug (KeyError, AttributeError, TypeError)
        inside ``_extract_is_leaked`` or the LLM wrapper must propagate so
        the operator sees the stack trace, not be silently bucketed as
        ``detector_failed=True``. The retry loop only catches the narrow
        transient set; everything else surfaces.
        """
        detector_llm = AsyncMock()
        detector_llm.invoke = AsyncMock(side_effect=KeyError("schema_field_typo"))

        question = _make_question(77)
        gt = _make_ground_truth(77)

        with pytest.raises(KeyError, match="schema_field_typo"):
            await screen_research_blob(question, gt, "research blob", cache, detector_llm=detector_llm)

        # Crucially: nothing was written to cache — an undiscovered bug doesn't
        # poison the verdict cache with detector_failed=True entries that future
        # runs would honor.
        assert cache.read_leakage_screen(qid=77) is None

    @pytest.mark.asyncio
    async def test_detector_failure_logs_exception(
        self, cache: AblationCache, caplog: pytest.LogCaptureFixture
    ) -> None:
        detector_llm = AsyncMock()
        detector_llm.invoke.side_effect = _transient_api_error("network blew up")

        question = _make_question(42)
        gt = _make_ground_truth(42)

        with caplog.at_level(logging.ERROR, logger="metaculus_bot.ablation.leakage_screen"):
            await screen_research_blob(question, gt, "Some research.", cache, detector_llm=detector_llm)

        exception_records = [r for r in caplog.records if r.exc_info is not None and r.levelno == logging.ERROR]
        assert exception_records, "expected logger.exception to record an exc_info ERROR-level entry"
        assert any("42" in r.getMessage() for r in exception_records)

    @pytest.mark.asyncio
    async def test_empty_research_blob_short_circuits(self, cache: AblationCache) -> None:
        detector_llm = AsyncMock()

        question = _make_question(42)
        gt = _make_ground_truth(42)

        verdict = await screen_research_blob(question, gt, "", cache, detector_llm=detector_llm)

        assert verdict["is_leaked"] is False
        assert verdict["detector_failed"] is False
        assert verdict["detector_response"] == "<empty research; nothing to leak>"
        detector_llm.invoke.assert_not_called()

        on_disk = cache.read_leakage_screen(qid=42)
        assert on_disk is not None
        assert on_disk["detector_response"] == "<empty research; nothing to leak>"

    @pytest.mark.asyncio
    async def test_diagnostics_only_blob_short_circuits(self, cache: AblationCache) -> None:
        """A fully-failed research run returns a diagnostics-ONLY blob (just the
        ``## Provider Diagnostics`` section: provider names/status/timing, nothing leakable).
        It must short-circuit the detector exactly like an empty string, never get screened."""
        from metaculus_bot.research.provider_diagnostics import ProviderResult, format_provider_diagnostics_block

        diagnostics_only = format_provider_diagnostics_block(
            [
                ProviderResult(name="asknews", status="inactive", chars=0, latency_ms=12, error_type="ForbiddenError"),
                ProviderResult(
                    name="native_search", status="errored", chars=0, latency_ms=8, error_type="RuntimeError"
                ),
            ]
        )
        assert "## Provider Diagnostics" in diagnostics_only  # precondition: real diagnostics block

        detector_llm = AsyncMock()
        question = _make_question(42)
        gt = _make_ground_truth(42)

        verdict = await screen_research_blob(question, gt, diagnostics_only, cache, detector_llm=detector_llm)

        assert verdict["is_leaked"] is False
        assert verdict["detector_failed"] is False
        detector_llm.invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_blob_with_real_research_plus_diagnostics_is_screened(self, cache: AblationCache) -> None:
        """A blob that carries genuine research (a provider section) ALONGSIDE the diagnostics
        block must still be screened — only diagnostics-only blobs short-circuit."""
        from metaculus_bot.research.provider_diagnostics import ProviderResult, format_provider_diagnostics_block

        diagnostics = format_provider_diagnostics_block(
            [ProviderResult(name="native_search", status="ok", chars=42, latency_ms=10)]
        )
        blob = f"## Web Research (Native Search)\nSome real research content.\n\n{diagnostics}"

        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = '{"is_leaked": false, "explanation": "no reveal"}'
        question = _make_question(42)
        gt = _make_ground_truth(42)

        verdict = await screen_research_blob(question, gt, blob, cache, detector_llm=detector_llm)

        detector_llm.invoke.assert_called_once()
        assert verdict["detector_failed"] is False

    @pytest.mark.asyncio
    async def test_records_screened_at_iso_datetime(self, cache: AblationCache) -> None:
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = '{"is_leaked": false, "explanation": "x"}'

        question = _make_question(42)
        gt = _make_ground_truth(42)

        verdict = await screen_research_blob(question, gt, "research", cache, detector_llm=detector_llm)

        # ISO 8601 datetime parses cleanly
        datetime.fromisoformat(verdict["screened_at"])


# ---------------------------------------------------------------------------
# screen_batch
# ---------------------------------------------------------------------------


class TestScreenBatch:
    @pytest.mark.asyncio
    @patch("metaculus_bot.ablation.leakage_screen.GeneralLlm")
    async def test_filters_leaked_questions(self, mock_llm_class, cache: AblationCache) -> None:
        responses = {
            1: '{"is_leaked": false, "explanation": "clean"}',
            2: '{"is_leaked": true, "explanation": "leaks"}',
            3: '{"is_leaked": false, "explanation": "clean"}',
        }

        async def invoke_side_effect(prompt: str) -> str:
            await asyncio.sleep(0)
            for qid, resp in responses.items():
                if f"Question {qid}" in prompt:
                    return resp
            return '{"is_leaked": false, "explanation": "default"}'

        mock_detector = AsyncMock()
        mock_detector.invoke = invoke_side_effect
        mock_llm_class.return_value = mock_detector

        questions = [_make_question(i, f"Question {i}") for i in [1, 2, 3]]
        ground_truths = {i: _make_ground_truth(i) for i in [1, 2, 3]}
        research_payloads = {i: f"Research for Q{i}" for i in [1, 2, 3]}

        clean_questions, clean_gts, all_verdicts = await screen_batch(
            questions, ground_truths, research_payloads, cache, concurrency=2
        )

        clean_ids = {q.id_of_question for q in clean_questions}
        assert clean_ids == {1, 3}
        assert set(clean_gts.keys()) == {1, 3}
        assert set(all_verdicts.keys()) == {1, 2, 3}
        assert all_verdicts[1]["is_leaked"] is False
        assert all_verdicts[2]["is_leaked"] is True
        assert all_verdicts[3]["is_leaked"] is False

    @pytest.mark.asyncio
    @patch("metaculus_bot.ablation.leakage_screen.GeneralLlm")
    async def test_drops_questions_with_failed_detector(self, mock_llm_class, cache: AblationCache) -> None:
        async def invoke_side_effect(prompt: str) -> str:
            await asyncio.sleep(0)
            if "Question 1" in prompt:
                raise _transient_api_error("transient failure")
            return '{"is_leaked": false, "explanation": "clean"}'

        mock_detector = AsyncMock()
        mock_detector.invoke = invoke_side_effect
        mock_llm_class.return_value = mock_detector

        questions = [_make_question(i, f"Question {i}") for i in [1, 2]]
        ground_truths = {i: _make_ground_truth(i) for i in [1, 2]}
        research_payloads = {i: f"Research for Q{i}" for i in [1, 2]}

        clean_questions, clean_gts, all_verdicts = await screen_batch(
            questions, ground_truths, research_payloads, cache, concurrency=2
        )

        clean_ids = {q.id_of_question for q in clean_questions}
        assert clean_ids == {2}
        assert set(clean_gts.keys()) == {2}
        assert all_verdicts[1]["is_leaked"] is True
        assert all_verdicts[1]["detector_failed"] is True
        assert all_verdicts[2]["is_leaked"] is False

    @pytest.mark.asyncio
    @patch("metaculus_bot.ablation.leakage_screen.GeneralLlm")
    async def test_warns_on_high_leakage_rate(
        self, mock_llm_class, cache: AblationCache, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def invoke_side_effect(prompt: str) -> str:
            await asyncio.sleep(0)
            if "Question 5" in prompt:
                return '{"is_leaked": false, "explanation": "clean"}'
            return '{"is_leaked": true, "explanation": "leaks"}'

        mock_detector = AsyncMock()
        mock_detector.invoke = invoke_side_effect
        mock_llm_class.return_value = mock_detector

        questions = [_make_question(i, f"Question {i}") for i in [1, 2, 3, 4, 5]]
        ground_truths = {i: _make_ground_truth(i) for i in [1, 2, 3, 4, 5]}
        research_payloads = {i: f"Research for Q{i}" for i in [1, 2, 3, 4, 5]}

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.ablation.leakage_screen"):
            await screen_batch(questions, ground_truths, research_payloads, cache, concurrency=5)

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("High leakage rate" in r.getMessage() for r in warning_records)

    @pytest.mark.asyncio
    @patch("metaculus_bot.ablation.leakage_screen.GeneralLlm")
    async def test_skips_qids_missing_research(
        self, mock_llm_class, cache: AblationCache, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_detector = AsyncMock()
        mock_detector.invoke = AsyncMock(return_value='{"is_leaked": false, "explanation": "x"}')
        mock_llm_class.return_value = mock_detector

        questions = [_make_question(i, f"Question {i}") for i in [1, 2, 3]]
        ground_truths = {i: _make_ground_truth(i) for i in [1, 2, 3]}
        # qid 2 is missing from research_payloads
        research_payloads = {1: "Research 1", 3: "Research 3"}

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.ablation.leakage_screen"):
            clean_questions, clean_gts, all_verdicts = await screen_batch(
                questions, ground_truths, research_payloads, cache, concurrency=5
            )

        # qid 2 not screened, no LLM call for it, not in all_verdicts, not in clean set
        assert mock_detector.invoke.await_count == 2
        assert set(all_verdicts.keys()) == {1, 3}
        clean_ids = {q.id_of_question for q in clean_questions}
        assert clean_ids == {1, 3}
        assert set(clean_gts.keys()) == {1, 3}

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "missing research" in r.getMessage().lower() or "no research" in r.getMessage().lower()
            for r in warning_records
        )

    @pytest.mark.asyncio
    @patch("metaculus_bot.ablation.leakage_screen.GeneralLlm")
    async def test_builds_detector_llm_once(self, mock_llm_class, cache: AblationCache) -> None:
        mock_detector = AsyncMock()
        mock_detector.invoke = AsyncMock(return_value='{"is_leaked": false, "explanation": "x"}')
        mock_llm_class.return_value = mock_detector

        questions = [_make_question(i, f"Question {i}") for i in [1, 2, 3, 4, 5]]
        ground_truths = {i: _make_ground_truth(i) for i in [1, 2, 3, 4, 5]}
        research_payloads = {i: f"Research for Q{i}" for i in [1, 2, 3, 4, 5]}

        await screen_batch(questions, ground_truths, research_payloads, cache, concurrency=2)

        assert mock_llm_class.call_count == 1
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["model"] == DEFAULT_DETECTOR_MODEL
        # temperature=None: reasoning models defer to provider defaults; None
        # keeps litellm from injecting temperature=0 (see _build_detector_llm).
        assert call_kwargs["temperature"] is None
        # max_tokens=32_000 is the combined reasoning+content budget for the
        # default reasoning model (glm-4.5-air:free). Production leakage.py used
        # to cap at 500 against gpt-5.6-luna, but 500 starves the reasoning budget
        # on long blobs and yields content=None; prod now runs uncapped at
        # effort=max against gpt-6-luna (2026-09-22). See _build_detector_llm docstring.
        assert call_kwargs["max_tokens"] == 32_000
        # response_format=json_object asks providers to honor JSON-mode. Replaces
        # the YES/NO-prefix parser, which mis-parsed markdown-decorated outputs
        # like ``**Answer: YES**`` (live observation: Q43131).
        assert call_kwargs["extra_body"] == {"response_format": {"type": "json_object"}}

    @pytest.mark.asyncio
    @patch("metaculus_bot.ablation.leakage_screen.GeneralLlm")
    async def test_respects_semaphore_concurrency(self, mock_llm_class, cache: AblationCache) -> None:
        in_flight = 0
        max_in_flight = 0

        async def invoke_side_effect(prompt: str) -> str:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return '{"is_leaked": false, "explanation": "x"}'

        mock_detector = AsyncMock()
        mock_detector.invoke = invoke_side_effect
        mock_llm_class.return_value = mock_detector

        questions = [_make_question(i, f"Question {i}") for i in range(1, 11)]
        ground_truths = {i: _make_ground_truth(i) for i in range(1, 11)}
        research_payloads = {i: f"Research {i}" for i in range(1, 11)}

        await screen_batch(questions, ground_truths, research_payloads, cache, concurrency=3)

        assert max_in_flight <= 3
        # Sanity: enough parallelism existed to actually exercise the semaphore.
        assert max_in_flight >= 2

    @pytest.mark.asyncio
    @patch("metaculus_bot.ablation.leakage_screen.GeneralLlm")
    async def test_clean_ground_truths_in_sync_with_clean_questions(self, mock_llm_class, cache: AblationCache) -> None:
        async def invoke_side_effect(prompt: str) -> str:
            await asyncio.sleep(0)
            if "Question 2" in prompt:
                return '{"is_leaked": true, "explanation": "leaks"}'
            return '{"is_leaked": false, "explanation": "clean"}'

        mock_detector = AsyncMock()
        mock_detector.invoke = invoke_side_effect
        mock_llm_class.return_value = mock_detector

        questions = [_make_question(i, f"Question {i}") for i in [1, 2, 3]]
        ground_truths = {i: _make_ground_truth(i) for i in [1, 2, 3]}
        research_payloads = {i: f"Research {i}" for i in [1, 2, 3]}

        clean_questions, clean_gts, _ = await screen_batch(
            questions, ground_truths, research_payloads, cache, concurrency=5
        )

        clean_q_ids = {q.id_of_question for q in clean_questions}
        assert clean_q_ids == set(clean_gts.keys())

    @pytest.mark.asyncio
    @patch("metaculus_bot.ablation.leakage_screen.GeneralLlm")
    async def test_force_true_propagates_to_per_question(self, mock_llm_class, cache: AblationCache) -> None:
        cached_payload = {
            "is_leaked": False,
            "detector_response": "NO - stale.",
            "detector_model": DEFAULT_DETECTOR_MODEL,
            "detector_failed": False,
            "screened_at": "2026-05-12T10:00:00",
        }
        cache.write_leakage_screen(qid=1, payload=cached_payload)

        mock_detector = AsyncMock()
        mock_detector.invoke = AsyncMock(
            return_value='{"is_leaked": true, "explanation": "leak detected on re-screen"}'
        )
        mock_llm_class.return_value = mock_detector

        questions = [_make_question(1, "Question 1")]
        ground_truths = {1: _make_ground_truth(1)}
        research_payloads = {1: "fresh research"}

        clean_questions, _, all_verdicts = await screen_batch(
            questions, ground_truths, research_payloads, cache, concurrency=1, force=True
        )

        assert mock_detector.invoke.await_count == 1
        assert all_verdicts[1]["is_leaked"] is True
        assert clean_questions == []


# ---------------------------------------------------------------------------
# C3 part 2: research_blob_sha on screen verdict
# ---------------------------------------------------------------------------


class TestResearchBlobShaInVerdict:
    @pytest.mark.asyncio
    async def test_screen_verdict_records_research_blob_sha(self, cache: AblationCache) -> None:
        """The verdict payload must record a hash of the screened blob so a
        downstream cache reader can detect a stale verdict against a fresher
        blob (e.g. after --force-stages prune).
        """
        detector_llm = AsyncMock()
        detector_llm.invoke.return_value = '{"is_leaked": false, "explanation": "clean"}'

        question = _make_question(42)
        gt = _make_ground_truth(42)
        research_blob = "the research blob being screened"

        verdict = await screen_research_blob(question, gt, research_blob, cache, detector_llm=detector_llm)

        assert "research_blob_sha" in verdict
        assert verdict["research_blob_sha"] == _blob_sha(research_blob)

        on_disk = cache.read_leakage_screen(qid=42)
        assert on_disk is not None
        assert on_disk["research_blob_sha"] == _blob_sha(research_blob)

    @pytest.mark.asyncio
    async def test_empty_blob_short_circuit_records_sha(self, cache: AblationCache) -> None:
        """Empty-blob short-circuit verdicts also carry the (empty-string) sha."""
        detector_llm = AsyncMock()

        question = _make_question(42)
        gt = _make_ground_truth(42)

        verdict = await screen_research_blob(question, gt, "", cache, detector_llm=detector_llm)

        assert verdict["research_blob_sha"] == _blob_sha("")
