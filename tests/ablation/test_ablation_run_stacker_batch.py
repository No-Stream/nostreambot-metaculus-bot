"""Batch fan-out, window-patch serialization and default stacker-LLM construction.

Split out of ``test_ablation_run_stacker.py``. Covers ``run_stacker_batch`` (per-qid
results, per-question failure isolation, LLM reuse), the ``_WINDOW_PATCH_LOCK`` that keeps
concurrent stacker calls from nesting the global window monkey-patch, and the
``stacker_llm=None`` path that builds the donated-key-first defaults. Factories come from
``tests/ablation_stacker_fakes.py``, fixtures from ``tests/ablation/conftest.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.ablation.run_stacker import (
    ARM_STACK,
    run_stacker_batch,
    run_stacker_for_arm,
)
from tests.ablation_stacker_fakes import _binary_payload, _make_binary_q, _run, _three_binary_forecasters

# ===========================================================================
# Window patch active during stacker call
# ===========================================================================


class TestWindowPatchActive:
    def test_window_patch_active_during_stacker_invocation(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        from metaculus_bot.ablation import window_patch as wp  # HARNESS-SCAN-EXEMPT-function-level-import

        observed_active: list[bool] = []

        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            observed_active.append(wp._window_patch_active)
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert observed_active == [True]
        # And after the call, no longer active
        assert wp._window_patch_active is False


# ===========================================================================
# run_stacker_batch
# ===========================================================================


class TestRunStackerBatch:
    def test_batch_returns_dict_keyed_by_qid(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        qid_to_data = {
            10: {
                "question": _make_binary_q(qid=10),
                "research": "R10",
                "forecaster_payloads": _three_binary_forecasters(),
            },
            20: {
                "question": _make_binary_q(qid=20),
                "research": "R20",
                "forecaster_payloads": _three_binary_forecasters(),
            },
        }

        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            results = _run(
                run_stacker_batch(
                    qid_to_data=qid_to_data,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert set(results.keys()) == {10, 20}
        assert all(r["success"] for r in results.values())

    def test_batch_per_question_failure_does_not_kill_batch(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        qid_to_data = {
            10: {
                "question": _make_binary_q(qid=10),
                "research": "R10",
                "forecaster_payloads": _three_binary_forecasters(),
            },
            20: {
                "question": _make_binary_q(qid=20),
                "research": "R20",
                "forecaster_payloads": {  # only 1 valid -> insufficient
                    model_slug_to_filename("openrouter/test/m1"): _binary_payload("openrouter/test/m1", 0.5),
                },
            },
            30: {
                "question": _make_binary_q(qid=30),
                "research": "R30",
                "forecaster_payloads": _three_binary_forecasters(),
            },
        }

        def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            results = _run(
                run_stacker_batch(
                    qid_to_data=qid_to_data,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert set(results.keys()) == {10, 20, 30}
        assert results[10]["success"] is True
        assert results[20]["success"] is False
        assert results[30]["success"] is True

    def test_batch_uses_passed_llms_directly(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Passing the same llm objects across multiple qids is the contract — they are reused."""
        qid_to_data = {
            10: {
                "question": _make_binary_q(qid=10),
                "research": "R10",
                "forecaster_payloads": _three_binary_forecasters(),
            },
            20: {
                "question": _make_binary_q(qid=20),
                "research": "R20",
                "forecaster_payloads": _three_binary_forecasters(),
            },
        }

        seen_stacker_llms: list[Any] = []

        def _fake_stacker(*args: Any, **_kwargs: Any) -> tuple[float, str]:
            seen_stacker_llms.append(args[0])
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            _run(
                run_stacker_batch(
                    qid_to_data=qid_to_data,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=stacker_llm,
                    fallback_stacker_llm=fallback_stacker_llm,
                    parser_llm=parser_llm,
                )
            )
        assert len(seen_stacker_llms) == 2
        assert all(s is stacker_llm for s in seen_stacker_llms)


# ===========================================================================
# Concurrent stacker calls — window-patch lock serializes patched section
# ===========================================================================


class TestConcurrentStackerLock:
    def test_concurrent_stacker_calls_serialized_under_lock(
        self,
        cache: AblationCache,
        stacker_llm: MagicMock,
        fallback_stacker_llm: MagicMock,
        parser_llm: MagicMock,
    ) -> None:
        """Two concurrent run_stacker_for_arm calls must NOT both be inside
        ``patched_window_for_question`` simultaneously.

        ``patched_window_for_question`` is a global monkey-patch that raises
        ``RuntimeError`` on nested entry. Without the module-level
        ``_WINDOW_PATCH_LOCK`` serializing patched-section entry, two
        concurrent stacker calls would race and the second to enter would
        crash. The lock keeps each call inside its own patched region.
        """
        from metaculus_bot.ablation import window_patch as wp  # HARNESS-SCAN-EXEMPT-function-level-import

        observed_active_during_call: list[bool] = []
        max_concurrent_in_patch = 0
        currently_in_patch = 0

        async def _fake_stacker(*_args: Any, **_kwargs: Any) -> tuple[float, str]:
            nonlocal max_concurrent_in_patch, currently_in_patch
            observed_active_during_call.append(wp._window_patch_active)
            currently_in_patch += 1
            max_concurrent_in_patch = max(max_concurrent_in_patch, currently_in_patch)
            # Yield so a second concurrent call gets a chance to interleave
            # if the lock isn't doing its job. The test only catches the bug
            # if there's a real opportunity for concurrent entry.
            await asyncio.sleep(0.01)
            currently_in_patch -= 1
            return 0.5, "meta"

        async def _drive() -> tuple[dict, dict]:
            with (
                patch(
                    "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                    return_value="",
                ),
                patch(
                    "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                    return_value="",
                ),
                patch(
                    "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                    new=AsyncMock(side_effect=_fake_stacker),
                ),
            ):
                return await asyncio.gather(
                    run_stacker_for_arm(
                        question=_make_binary_q(qid=101),
                        research_blob="R101",
                        forecaster_payloads=_three_binary_forecasters(),
                        arm=ARM_STACK,
                        cache=cache,
                        stacker_llm=stacker_llm,
                        fallback_stacker_llm=fallback_stacker_llm,
                        parser_llm=parser_llm,
                    ),
                    run_stacker_for_arm(
                        question=_make_binary_q(qid=102),
                        research_blob="R102",
                        forecaster_payloads=_three_binary_forecasters(),
                        arm=ARM_STACK,
                        cache=cache,
                        stacker_llm=stacker_llm,
                        fallback_stacker_llm=fallback_stacker_llm,
                        parser_llm=parser_llm,
                    ),
                )

        results = _run(_drive())
        assert all(r["success"] for r in results)
        # Patch was active during every stacker call (both succeeded)
        assert observed_active_during_call == [True, True]
        # The lock kept patched-section entry serialized — at most one stacker
        # call inside patched_window_for_question at a time.
        assert max_concurrent_in_patch == 1, (
            f"window patch lock failed to serialize concurrent calls "
            f"(max_concurrent_in_patch={max_concurrent_in_patch})"
        )
        # And after the calls, no longer active.
        assert wp._window_patch_active is False


# ===========================================================================
# Default stacker LLM construction — donated-key wrapper
# ===========================================================================


class TestDefaultStackerWiredViaDonatedKey:
    """When callers pass ``stacker_llm=None`` we construct claude-opus-4.5
    (primary) and gpt-6-sol (fallback) routed via ``build_llm_with_openrouter_fallback``
    so the Metaculus-donated key is tried before the operator's paid key.

    This mirrors production STACKER_LLM / STACKER_FALLBACK_LLM in
    ``llm_configs.py``. An earlier iteration tried an OpenAI model as primary,
    but the operator's local-`.env` donated key data-policy blocked it;
    Anthropic models work cleanly. Production with a different
    ``OAI_ANTH_OPENROUTER_KEY`` GitHub-secret value behaved differently.
    """

    def test_default_stacker_uses_opus_4_5_via_donated_key_wrapper(
        self, cache: AblationCache, parser_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from metaculus_bot.ablation.run_stacker import (  # HARNESS-SCAN-EXEMPT-function-level-import
            DEFAULT_STACKER_FALLBACK_MODEL,
            DEFAULT_STACKER_MODEL,
        )
        from metaculus_bot.fallback_openrouter import (
            FallbackOpenRouterLlm,  # HARNESS-SCAN-EXEMPT-function-level-import
        )

        # Both keys present + distinct → wrapper chooses FallbackOpenRouterLlm.
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "fake_donated")
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake_paid")

        # Pin the new defaults at the constant level — primary is opus-4.5,
        # fallback is gpt-6-sol (different provider for independent failure
        # mode; matches prod STACKER_FALLBACK_LLM post the 2026-09-22 migration).
        assert DEFAULT_STACKER_MODEL == "openrouter/anthropic/claude-opus-4.5"
        assert DEFAULT_STACKER_FALLBACK_MODEL == "openrouter/openai/gpt-6-sol"

        captured_llms: list[Any] = []

        def _fake_stacker(*args: Any, **_kwargs: Any) -> tuple[float, str]:
            captured_llms.append(args[0])
            return 0.5, "meta"

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            payload = _run(
                run_stacker_for_arm(
                    question=_make_binary_q(qid=1),
                    research_blob="R",
                    forecaster_payloads=_three_binary_forecasters(),
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=None,  # exercise default construction
                    fallback_stacker_llm=None,
                    parser_llm=parser_llm,
                )
            )
        assert payload["success"] is True
        assert len(captured_llms) == 1
        primary = captured_llms[0]
        assert isinstance(primary, FallbackOpenRouterLlm), (
            f"Default stacker should be FallbackOpenRouterLlm; got {type(primary).__name__}"
        )
        assert primary.model == "openrouter/anthropic/claude-opus-4.5"

    def test_default_stacker_in_batch_uses_opus_4_5_via_donated_key_wrapper(
        self, cache: AblationCache, parser_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from metaculus_bot.fallback_openrouter import (
            FallbackOpenRouterLlm,  # HARNESS-SCAN-EXEMPT-function-level-import
        )

        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "fake_donated")
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake_paid")

        captured_llms: list[Any] = []

        def _fake_stacker(*args: Any, **_kwargs: Any) -> tuple[float, str]:
            captured_llms.append(args[0])
            return 0.5, "meta"

        qid_to_data = {
            10: {
                "question": _make_binary_q(qid=10),
                "research": "R10",
                "forecaster_payloads": _three_binary_forecasters(),
            },
        }

        with (
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.run_tools_for_forecaster",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.tool_runner.build_cross_model_aggregation",
                return_value="",
            ),
            patch(
                "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
                new=AsyncMock(side_effect=_fake_stacker),
            ),
        ):
            results = _run(
                run_stacker_batch(
                    qid_to_data=qid_to_data,
                    arm=ARM_STACK,
                    cache=cache,
                    stacker_llm=None,  # exercise default construction
                    fallback_stacker_llm=None,
                    parser_llm=parser_llm,
                )
            )
        assert results[10]["success"] is True
        assert len(captured_llms) == 1
        primary = captured_llms[0]
        assert isinstance(primary, FallbackOpenRouterLlm)
        assert primary.model == "openrouter/anthropic/claude-opus-4.5"
