"""The per-role dollar ledger behind ``CREDIT_ROLE_SPEND`` (``metaculus_bot.credit_telemetry``).

OpenRouter's own per-call ``usage`` object reaches litellm's success callback tagged with the
``role`` / ``key_alias`` metadata the LLM builders stamp on every completion. These tests pin the
ledger arithmetic, the marker line's exact shape (the archive harvester keys on it), the token
fields that make prompt-cache hit rates and reasoning spend observable per role, and that a role
with no cost data renders ``usd=n/a`` rather than a fabricated zero. The balance snapshots, the
floor check and the drained-key probe live in tests/test_credit_telemetry.py.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from typing import Any

import litellm
import pytest
from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models import general_llm as ft_general_llm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.types.utils import ModelResponse, Usage

from metaculus_bot.check_openrouter_credits import KEY_SPECS
from metaculus_bot.constants import PROMPT_TOKENS_ALERT_THRESHOLD
from metaculus_bot.credit_telemetry import (
    DIRECT_KEY_ALIAS,
    DONATED_KEY_ALIAS,
    KEY_ALIAS_METADATA_KEY,
    PERSONAL_KEY_ALIAS,
    QUESTION_METADATA_KEY,
    ROLE_METADATA_KEY,
    UNKNOWN_KEY_ALIAS,
    UNTAGGED_ROLE,
    RoleSpendTracker,
    TokenCounts,
    drain_litellm_callbacks,
    install_role_spend_tracker,
    llm_call_metadata,
    log_role_spend,
    log_run_summary,
    plain_llm_key_alias,
    record_llm_call_spend,
    reset_role_spend,
    role_spend_rows,
    run_spend_summary,
)
from metaculus_bot.fallback_openrouter import FallbackOpenRouterLlm, build_llm_with_openrouter_fallback
from metaculus_bot.llm_configs import (
    DISAGREEMENT_ANALYZER_LLM,
    FORECASTER_LLMS,
    MARKET_QUERY_AUTHOR_LLM_CONFIG,
    MARKET_RANKER_LLM_CONFIG,
    PARSER_LLM,
    STACKER_FALLBACK_LLM,
    STACKER_LLM,
    SUMMARIZER_LLM,
    forecaster_role,
)
from scripts.telemetry.markers import MARKER_SPECS

NO_TOKENS_TAIL = " prompt_tokens=0 completion_tokens=0 cached_tokens=0 reasoning_tokens=0"
NO_MAX_PROMPT_TAIL = " max_prompt_tokens=0"
UNCOSTED_TAIL = NO_TOKENS_TAIL + " charged_usd=n/a byok_calls=0" + NO_MAX_PROMPT_TAIL


@pytest.fixture
def clean_role_ledger() -> Iterator[None]:
    """Empty ledger before AND after, and no ``RoleSpendTracker`` left in litellm's
    process-global callback lists: the tracker is installed once per process in prod, so a
    test that installs it must not leak it into the rest of the session."""
    reset_role_spend()
    yield
    for callback in list(litellm.callbacks):
        if isinstance(callback, RoleSpendTracker):
            litellm.logging_callback_manager.remove_callback_from_all_lists(callback)
    reset_role_spend()


def _role_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("CREDIT_ROLE_SPEND:")]


def _success_kwargs(metadata: dict[str, str] | None) -> dict[str, Any]:
    """The ``kwargs`` litellm hands a success callback: our tag rides
    ``litellm_params["metadata"]`` (verified against litellm 1.92's function_setup)."""
    return {"model": "openai/gpt-5.6-luna", "litellm_params": {"acompletion": True, "metadata": metadata}}


def _response_with_usage(prompt_tokens: int = 10, completion_tokens: int = 5, **usage_fields: Any) -> ModelResponse:
    """A litellm ModelResponse whose usage carries OpenRouter's accounting fields.

    ``litellm.Usage`` keeps every extra constructor kwarg as an attribute (``cost``,
    ``cost_details``, ``is_byok``) and wraps ``prompt_tokens_details`` /
    ``completion_tokens_details`` dicts, which is exactly how an OpenRouter body's ``usage``
    object reaches the callback in prod (``convert_dict_to_response`` builds ``Usage(**body["usage"])``).
    """
    return ModelResponse(
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            **usage_fields,
        )
    )


@pytest.mark.usefixtures("clean_role_ledger")
class TestRoleSpendLedger:
    def test_rows_sum_cost_and_byok_upstream_per_role_and_key(self, caplog) -> None:
        """Two donated-key forecaster calls (BYOK: a small OpenRouter fee in ``cost`` plus the
        provider charge in ``upstream_inference_cost``) and one personal-key call whose whole
        charge is ``cost``. Same role, different keys, so two rows."""
        record_llm_call_spend(
            "forecaster:openai", DONATED_KEY_ALIAS, cost_usd=0.001, byok_upstream_usd=0.12, is_byok=True
        )
        record_llm_call_spend(
            "forecaster:openai", DONATED_KEY_ALIAS, cost_usd=0.002, byok_upstream_usd=0.08, is_byok=True
        )
        record_llm_call_spend("forecaster:openai", PERSONAL_KEY_ALIAS, cost_usd=0.25, byok_upstream_usd=None)

        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_role_spend()

        assert _role_lines(caplog) == [
            "CREDIT_ROLE_SPEND: role=forecaster:openai key=personal usd=0.2500 calls=1 costed_calls=1 byok_usd=0.0000"
            + NO_TOKENS_TAIL
            + " charged_usd=0.2500 byok_calls=0"
            + NO_MAX_PROMPT_TAIL,
            "CREDIT_ROLE_SPEND: role=forecaster:openai key=donated usd=0.2030 calls=2 costed_calls=2 byok_usd=0.2000"
            + NO_TOKENS_TAIL
            + " charged_usd=0.2030 byok_calls=2"
            + NO_MAX_PROMPT_TAIL,
        ]

    def test_charged_usd_counts_the_upstream_cost_only_on_byok_calls(self, caplog) -> None:
        """The double count the 2026-09-09 cost pass found: off BYOK, OpenRouter echoes the upstream
        cost beside ``cost`` (the personal-key Google slot: 0.57 charged, 1.14 in ``usd``), and only
        ``cost`` hits the key. ``usd`` keeps its old meaning; ``charged_usd`` is the money."""
        record_llm_call_spend(
            "forecaster:google", PERSONAL_KEY_ALIAS, cost_usd=0.5716, byok_upstream_usd=0.5716, is_byok=False
        )
        record_llm_call_spend(
            "forecaster:openai", DONATED_KEY_ALIAS, cost_usd=0.0, byok_upstream_usd=1.1409, is_byok=True
        )
        record_llm_call_spend(
            "forecaster:openai", DONATED_KEY_ALIAS, cost_usd=None, byok_upstream_usd=None, is_byok=True
        )

        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_role_spend()

        assert _role_lines(caplog) == [
            "CREDIT_ROLE_SPEND: role=forecaster:google key=personal usd=1.1432 calls=1 costed_calls=1 byok_usd=0.5716"
            + NO_TOKENS_TAIL
            + " charged_usd=0.5716 byok_calls=0"
            + NO_MAX_PROMPT_TAIL,
            "CREDIT_ROLE_SPEND: role=forecaster:openai key=donated usd=1.1409 calls=2 costed_calls=1 byok_usd=1.1409"
            + NO_TOKENS_TAIL
            + " charged_usd=1.1409 byok_calls=2"
            + NO_MAX_PROMPT_TAIL,
        ]

    def test_token_counts_sum_per_row_including_uncosted_calls(self, caplog) -> None:
        """Tokens are independent of cost data: a call OpenRouter did not cost still spent tokens,
        so the token tail sums over every call while ``costed_calls`` keeps counting the dollars."""
        record_llm_call_spend(
            "gap_fill_v2_driver",
            DONATED_KEY_ALIAS,
            cost_usd=0.0,
            byok_upstream_usd=0.03,
            is_byok=True,
            tokens=TokenCounts(prompt=40_000, completion=900, cached=38_000, reasoning=700),
        )
        record_llm_call_spend(
            "gap_fill_v2_driver",
            DONATED_KEY_ALIAS,
            cost_usd=None,
            byok_upstream_usd=None,
            is_byok=True,
            tokens=TokenCounts(prompt=1_000, completion=100, cached=0, reasoning=50),
        )

        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_role_spend()

        assert _role_lines(caplog) == [
            "CREDIT_ROLE_SPEND: role=gap_fill_v2_driver key=donated usd=0.0300 calls=2 costed_calls=1 byok_usd=0.0300"
            " prompt_tokens=41000 completion_tokens=1000 cached_tokens=38000 reasoning_tokens=750"
            " charged_usd=0.0300 byok_calls=2 max_prompt_tokens=40000",
        ]

    def test_max_prompt_tokens_is_the_largest_single_prompt_not_the_sum(self) -> None:
        """The packet-size read: ``prompt_tokens`` sums a row's calls, so a 41k loop peak is invisible
        in a 300k total; ``max_prompt_tokens`` keeps the largest single prompt the row ever sent."""
        for prompt in (2_800, 41_176, 15_900):
            record_llm_call_spend(
                "gap_fill_v2_driver",
                DONATED_KEY_ALIAS,
                cost_usd=0.0,
                byok_upstream_usd=0.01,
                is_byok=True,
                tokens=TokenCounts(prompt=prompt, completion=100),
            )

        (row,) = role_spend_rows()
        assert row.tokens.prompt == 59_876
        assert row.max_prompt_tokens == 41_176

    def test_uncosted_calls_render_na_not_zero(self, caplog) -> None:
        """A completion that carried no usage.cost is still a call, but its dollars are UNKNOWN;
        rendering 0.0000 would read as "this role is free"."""
        record_llm_call_spend("perplexity_research", DIRECT_KEY_ALIAS, cost_usd=None, byok_upstream_usd=None)
        record_llm_call_spend("perplexity_research", DIRECT_KEY_ALIAS, cost_usd=None, byok_upstream_usd=None)

        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_role_spend()

        assert _role_lines(caplog) == [
            "CREDIT_ROLE_SPEND: role=perplexity_research key=direct usd=n/a calls=2 costed_calls=0 byok_usd=n/a"
            + UNCOSTED_TAIL,
        ]

    def test_mixed_costed_and_uncosted_reports_both_counts(self, caplog) -> None:
        """The sum covers only the costed calls, and ``costed_calls < calls`` says so."""
        record_llm_call_spend("parser", DONATED_KEY_ALIAS, cost_usd=0.01, byok_upstream_usd=None)
        record_llm_call_spend("parser", DONATED_KEY_ALIAS, cost_usd=None, byok_upstream_usd=None)

        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_role_spend()

        assert _role_lines(caplog) == [
            "CREDIT_ROLE_SPEND: role=parser key=donated usd=0.0100 calls=2 costed_calls=1 byok_usd=0.0000"
            + NO_TOKENS_TAIL
            + " charged_usd=0.0100 byok_calls=0"
            + NO_MAX_PROMPT_TAIL,
        ]

    def test_rows_sort_by_usd_descending_with_uncosted_last(self) -> None:
        record_llm_call_spend("parser", DONATED_KEY_ALIAS, cost_usd=0.01, byok_upstream_usd=None)
        record_llm_call_spend("untagged", UNKNOWN_KEY_ALIAS, cost_usd=None, byok_upstream_usd=None)
        record_llm_call_spend("forecaster:google", PERSONAL_KEY_ALIAS, cost_usd=0.30, byok_upstream_usd=None)
        record_llm_call_spend("summarizer", DONATED_KEY_ALIAS, cost_usd=0.0, byok_upstream_usd=0.05)

        assert [(row.role, row.key_alias) for row in role_spend_rows()] == [
            ("forecaster:google", PERSONAL_KEY_ALIAS),
            ("summarizer", DONATED_KEY_ALIAS),
            ("parser", DONATED_KEY_ALIAS),
            ("untagged", UNKNOWN_KEY_ALIAS),
        ]

    def test_empty_ledger_says_so_without_the_row_shape(self, caplog) -> None:
        """A run with zero completions must still leave a line (silence is indistinguishable
        from a run that died first), but not one the harvester could mistake for a row."""
        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_role_spend()

        (line,) = _role_lines(caplog)
        assert "role=" not in line
        assert "no successful LLM completions" in line

    def test_every_row_parses_under_the_registry_regex(self, caplog) -> None:
        """Seam pin: the emitted line is a data contract with scripts/telemetry/markers.py."""
        record_llm_call_spend(
            "forecaster:google",
            PERSONAL_KEY_ALIAS,
            cost_usd=0.57,
            byok_upstream_usd=0.57,
            tokens=TokenCounts(prompt=52_000, completion=6_000, cached=0, reasoning=5_200),
        )
        record_llm_call_spend("perplexity_research", DIRECT_KEY_ALIAS, cost_usd=None, byok_upstream_usd=None)
        spec = next(s for s in MARKER_SPECS if s.name == "credit_role_spend")

        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_role_spend()

        matches = [spec.regex.search(line) for line in _role_lines(caplog)]
        assert all(match is not None for match in matches)
        costed, uncosted = matches
        assert costed is not None
        assert uncosted is not None
        assert costed.group("cached_tokens") == "0"
        assert costed.group("reasoning_tokens") == "5200"
        assert (costed.group("usd"), costed.group("charged_usd"), costed.group("byok_calls")) == (
            "1.1400",
            "0.5700",
            "0",
        )
        assert costed.group("max_prompt_tokens") == "52000"
        assert uncosted.group("usd") == "n/a"
        assert uncosted.group("prompt_tokens") == "0"
        assert (uncosted.group("charged_usd"), uncosted.group("byok_calls")) == ("n/a", "0")
        assert uncosted.group("max_prompt_tokens") == "0"

    def test_key_aliases_are_the_credit_spend_key_names(self) -> None:
        """``CREDIT_ROLE_SPEND key=`` must join onto ``CREDIT_SPEND key=`` / ``CREDIT_BALANCE key=``,
        whose vocabulary is KEY_SPECS."""
        assert {DONATED_KEY_ALIAS, PERSONAL_KEY_ALIAS} == set(KEY_SPECS)
        assert DIRECT_KEY_ALIAS not in KEY_SPECS
        assert UNKNOWN_KEY_ALIAS not in KEY_SPECS


class TestLlmCallMetadata:
    def test_role_and_key_ride_the_two_metadata_keys(self) -> None:
        assert llm_call_metadata("stacker", DONATED_KEY_ALIAS) == {
            ROLE_METADATA_KEY: "stacker",
            KEY_ALIAS_METADATA_KEY: DONATED_KEY_ALIAS,
        }

    def test_missing_role_is_tagged_untagged_at_construction(self) -> None:
        """Construction, not the callback, owns the default: every metaculus_bot-built LLM
        carries an explicit role token, so an ``untagged`` row in a run log means a builder
        call site forgot its ``role=``."""
        assert llm_call_metadata(None, PERSONAL_KEY_ALIAS)[ROLE_METADATA_KEY] == UNTAGGED_ROLE

    def test_question_ref_rides_a_third_key_only_when_given(self) -> None:
        """The roster LLMs are built once per process and cannot know their question, so the key is
        absent by default; the v2 driver stamps it per call and PROMPT_SIZE_ALERT reads it back."""
        assert QUESTION_METADATA_KEY not in llm_call_metadata("stacker", DONATED_KEY_ALIAS)
        tagged = llm_call_metadata("gap_fill_v2_driver", DONATED_KEY_ALIAS, question_ref="https://x/questions/650/")
        assert tagged[QUESTION_METADATA_KEY] == "https://x/questions/650/"

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("openrouter/openai/gpt-5.6-sol", "forecaster:openai"),
            ("openrouter/anthropic/claude-opus-4.8", "forecaster:anthropic"),
            ("openrouter/google/gemini-3.1-pro-preview", "forecaster:google"),
        ],
    )
    def test_forecaster_role_is_the_vendor_slot(self, model: str, expected: str) -> None:
        """Latest-per-vendor roster: the slot outlives any one model, so the role does too."""
        assert forecaster_role(model) == expected

    def test_forecaster_role_rejects_a_non_openrouter_slug(self) -> None:
        with pytest.raises(ValueError, match="openrouter/<vendor>/<model>"):
            forecaster_role("perplexity/sonar")

    def test_plain_llm_key_alias(self) -> None:
        """A plain GeneralLlm with no api_key reads OPENROUTER_API_KEY from the environment for
        openrouter/ slugs; anything else bills a provider-direct key."""
        assert plain_llm_key_alias("openrouter/x-ai/grok-4.5") == PERSONAL_KEY_ALIAS
        assert plain_llm_key_alias("perplexity/sonar-reasoning") == DIRECT_KEY_ALIAS


@pytest.mark.usefixtures("clean_role_ledger")
class TestRoleSpendTracker:
    async def test_callback_reads_role_key_and_openrouter_usage_fields(self) -> None:
        """A BYOK call: OpenRouter's fee in ``cost``, the provider's charge in ``upstream_inference_cost``,
        ``is_byok`` true. Both payers were charged, so ``charged_usd`` is their sum."""
        tracker = RoleSpendTracker()
        response = _response_with_usage(cost=0.0015, is_byok=True, cost_details={"upstream_inference_cost": 0.31})

        await tracker.async_log_success_event(
            _success_kwargs(llm_call_metadata("stacker", DONATED_KEY_ALIAS)), response, None, None
        )

        (row,) = role_spend_rows()
        assert (row.role, row.key_alias, row.calls, row.costed_calls, row.byok_calls) == (
            "stacker",
            DONATED_KEY_ALIAS,
            1,
            1,
            1,
        )
        assert row.usd == pytest.approx(0.3115)
        assert row.byok_usd == pytest.approx(0.31)
        assert row.charged_usd == pytest.approx(0.3115)

    async def test_non_byok_call_that_echoes_upstream_cost_is_charged_once(self) -> None:
        """The production shape behind the double count (run 34091717001, the personal-key Google
        slot): ``cost`` and ``upstream_inference_cost`` both 0.1394, ``is_byok`` false, and the key's
        settled usage moved by 0.14. ``usd`` still shows the legacy sum so old rows stay comparable."""
        tracker = RoleSpendTracker()
        response = _response_with_usage(cost=0.1394, is_byok=False, cost_details={"upstream_inference_cost": 0.1394})

        await tracker.async_log_success_event(
            _success_kwargs(llm_call_metadata("forecaster:google", PERSONAL_KEY_ALIAS)), response, None, None
        )

        (row,) = role_spend_rows()
        assert row.usd == pytest.approx(0.2788)
        assert row.byok_usd == pytest.approx(0.1394)
        assert row.charged_usd == pytest.approx(0.1394)
        assert row.byok_calls == 0

    async def test_usage_without_is_byok_reads_as_not_byok(self) -> None:
        """A body that omits ``is_byok`` charges ``cost`` only; on a BYOK key that shows as
        ``charged_usd`` below ``byok_usd`` with ``byok_calls=0``, the visible signature of
        OpenRouter dropping the field, not a silent zero."""
        tracker = RoleSpendTracker()
        response = _response_with_usage(cost=0.0, cost_details={"upstream_inference_cost": 0.31})

        await tracker.async_log_success_event(
            _success_kwargs(llm_call_metadata("summarizer", DONATED_KEY_ALIAS)), response, None, None
        )

        (row,) = role_spend_rows()
        assert (row.usd, row.byok_usd, row.charged_usd, row.byok_calls) == (
            pytest.approx(0.31),
            pytest.approx(0.31),
            0.0,
            0,
        )

    async def test_callback_reads_cached_and_reasoning_tokens_off_the_usage_details(self) -> None:
        """The two OpenRouter detail objects, ``prompt_tokens_details.cached_tokens`` and
        ``completion_tokens_details.reasoning_tokens``, are what make a prompt-cache hit rate
        and the hidden reasoning spend observable per role."""
        tracker = RoleSpendTracker()
        response = _response_with_usage(
            prompt_tokens=41_176,
            completion_tokens=873,
            cost=0.0,
            cost_details={"upstream_inference_cost": 0.03},
            prompt_tokens_details={"cached_tokens": 39_936, "cache_write_tokens": 0, "audio_tokens": 0},
            completion_tokens_details={"reasoning_tokens": 640},
        )

        await tracker.async_log_success_event(
            _success_kwargs(llm_call_metadata("gap_fill_v2_driver", DONATED_KEY_ALIAS)), response, None, None
        )

        (row,) = role_spend_rows()
        assert row.tokens == TokenCounts(prompt=41_176, completion=873, cached=39_936, reasoning=640)

    async def test_usage_without_detail_objects_reads_zero_cached_and_reasoning(self) -> None:
        """A provider that reports no details (or null inside them) still yields the base counts;
        cached and reasoning read 0, not a crash and not None."""
        tracker = RoleSpendTracker()
        await tracker.async_log_success_event(
            _success_kwargs(llm_call_metadata("parser", PERSONAL_KEY_ALIAS)),
            _response_with_usage(
                prompt_tokens=12,
                completion_tokens=3,
                cost=0.01,
                prompt_tokens_details={"cached_tokens": None},
                completion_tokens_details={"reasoning_tokens": None},
            ),
            None,
            None,
        )

        (row,) = role_spend_rows()
        assert row.tokens == TokenCounts(prompt=12, completion=3, cached=0, reasoning=0)

    async def test_non_byok_usage_has_no_upstream_component(self) -> None:
        """OpenRouter sends upstream_inference_cost as null (or omits cost_details) off BYOK."""
        tracker = RoleSpendTracker()
        response = _response_with_usage(cost=0.02, cost_details={"upstream_inference_cost": None})

        await tracker.async_log_success_event(
            _success_kwargs(llm_call_metadata("parser", PERSONAL_KEY_ALIAS)), response, None, None
        )

        (row,) = role_spend_rows()
        assert (row.usd, row.byok_usd, row.charged_usd, row.costed_calls) == (
            pytest.approx(0.02),
            0.0,
            pytest.approx(0.02),
            1,
        )

    async def test_missing_metadata_files_under_untagged_and_unknown_key(self) -> None:
        """Any litellm completion the bot did not build (forecasting-tools' own helpers, an
        ablation harness) still counts, visibly, under the two sentinel labels."""
        tracker = RoleSpendTracker()
        await tracker.async_log_success_event(_success_kwargs(None), _response_with_usage(cost=0.5), None, None)

        (row,) = role_spend_rows()
        assert (row.role, row.key_alias, row.calls) == (UNTAGGED_ROLE, UNKNOWN_KEY_ALIAS, 1)

    async def test_usage_without_cost_is_a_call_but_not_a_costed_call(self) -> None:
        tracker = RoleSpendTracker()
        await tracker.async_log_success_event(
            _success_kwargs(llm_call_metadata("perplexity_research", DIRECT_KEY_ALIAS)),
            _response_with_usage(),
            None,
            None,
        )

        (row,) = role_spend_rows()
        assert (row.calls, row.costed_calls, row.usd) == (1, 0, None)
        assert row.tokens == TokenCounts(prompt=10, completion=5)

    async def test_response_without_usage_is_a_call_with_zero_tokens(self) -> None:
        tracker = RoleSpendTracker()
        await tracker.async_log_success_event(
            _success_kwargs(llm_call_metadata("parser", DONATED_KEY_ALIAS)), ModelResponse(), None, None
        )

        (row,) = role_spend_rows()
        assert (row.calls, row.costed_calls, row.usd, row.charged_usd, row.tokens) == (1, 0, None, None, TokenCounts())

    async def test_non_finite_cost_is_treated_as_unreported(self) -> None:
        """Same rule as the balance parser: NaN would poison every sum it touched."""
        tracker = RoleSpendTracker()
        await tracker.async_log_success_event(
            _success_kwargs(llm_call_metadata("parser", DONATED_KEY_ALIAS)),
            _response_with_usage(cost=float("nan")),
            None,
            None,
        )

        (row,) = role_spend_rows()
        assert (row.calls, row.costed_calls, row.usd) == (1, 0, None)

    def test_install_is_idempotent(self) -> None:
        install_role_spend_tracker()
        install_role_spend_tracker()
        assert sum(isinstance(cb, RoleSpendTracker) for cb in litellm.callbacks) == 1

    async def test_real_litellm_mock_path_delivers_the_role_tag_after_drain(self, monkeypatch) -> None:
        """End to end through forecasting-tools and REAL litellm (network short-circuited by
        ``mock_response``): the builder's ``role=`` reaches the ledger via
        ``metadata`` -> ``litellm_params`` -> the success callback -> the logging worker.

        The drain is load-bearing: litellm enqueues the callback from a ``create_task``,
        so without it the row is not there yet when the awaited call returns."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "sk-or-v1-DONATEDsecretAB12")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-PERSONALsecretCD34")
        install_role_spend_tracker()
        real_acompletion = litellm.acompletion

        async def mocked_acompletion(**kwargs: Any) -> Any:
            return await real_acompletion(**kwargs, mock_response="ok")

        monkeypatch.setattr(ft_general_llm, "acompletion", mocked_acompletion)
        llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5.6-luna", role="parser", allowed_tries=1)
        assert isinstance(llm, FallbackOpenRouterLlm)

        assert await llm.invoke("hi") == "ok"
        await drain_litellm_callbacks()

        (row,) = role_spend_rows()
        # The mock body carries token counts but no OpenRouter usage.cost: counted, dollars unknown, never a zero.
        assert (row.role, row.key_alias, row.calls, row.costed_calls, row.usd) == (
            "parser",
            DONATED_KEY_ALIAS,
            1,
            0,
            None,
        )
        assert row.tokens.prompt > 0

    async def test_drain_is_bounded_and_a_no_op_with_no_pending_callbacks(self) -> None:
        """cli.main's finally must never stall on telemetry; with nothing queued this returns
        immediately rather than waiting on a worker that never started."""
        await asyncio.wait_for(drain_litellm_callbacks(), timeout=1.0)

    async def test_drain_timeout_warns_and_returns_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A flush that never finishes must not turn a published run into a crashed one.

        The timeout is reachable without a bug on our side: litellm allows each queued
        callback 20s (``LOGGING_WORKER_MAX_TIME_PER_COROUTINE``), twice this drain's
        bound, and a worker loop that dies on a non-``CancelledError`` leaves
        ``queue.join()`` outstanding forever. The drain runs from
        ``cli._forecast_with_callback_drain``'s ``finally``, and nothing between there
        and process exit catches, so a raise here discarded a fully published run's
        reports and skipped ``log_report_summary`` plus the whole degradation/exit
        block (the q45085 failure shape), or demoted a real forecast error to
        ``__context__``. It warns and returns; the ledger may under-count.
        """

        async def never_finishes() -> None:
            await asyncio.Event().wait()

        monkeypatch.setattr(GLOBAL_LOGGING_WORKER, "flush", never_finishes)
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.credit_telemetry"):
            await asyncio.wait_for(drain_litellm_callbacks(timeout_s=0.01), timeout=5.0)

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(msg.startswith("LITELLM_CALLBACK_DRAIN_TIMEOUT:") for msg in warnings), warnings
        # Seam pin: the WARN has been a harvested marker since 2026-09-04; ``%.1f`` renders 0.01 as 0.0.
        spec = next(s for s in MARKER_SPECS if s.name == "litellm_callback_drain_timeout")
        match = spec.regex.search(caplog.text)
        assert match is not None, warnings
        assert match.group("timeout_s") == "0.0"


def _alert_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("PROMPT_SIZE_ALERT:")]


@pytest.mark.usefixtures("clean_role_ledger")
class TestPromptSizeAlert:
    """One WARNING per LLM call whose prompt exceeds ``PROMPT_TOKENS_ALERT_THRESHOLD``.

    The forecaster prompt measures about 17k tokens and the gap-fill v2 loop peaks near 41k
    (2026-09-09 cost pass), so a call over the 150k threshold is a packet that blew up, the
    operator's "500k is way too much and could degrade model performance" case. The call is
    already billed when the callback sees its usage, so this reads and never gates.
    """

    async def test_oversized_prompt_fires_with_role_question_and_threshold(self, caplog) -> None:
        tracker = RoleSpendTracker()
        metadata = llm_call_metadata(
            "gap_fill_v2_driver", DONATED_KEY_ALIAS, question_ref="https://www.metaculus.com/questions/38975/"
        )
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.credit_telemetry"):
            await tracker.async_log_success_event(
                _success_kwargs(metadata), _response_with_usage(prompt_tokens=160_000, cost=0.4), None, None
            )

        assert _alert_lines(caplog) == [
            "PROMPT_SIZE_ALERT: role=gap_fill_v2_driver question=https://www.metaculus.com/questions/38975/"
            f" prompt_tokens=160000 threshold={PROMPT_TOKENS_ALERT_THRESHOLD}"
        ]
        # The call still lands on the ledger; the alert is a companion line, not a substitute row.
        (row,) = role_spend_rows()
        assert (row.tokens.prompt, row.max_prompt_tokens) == (160_000, 160_000)

    async def test_normal_prompt_is_silent(self, caplog) -> None:
        tracker = RoleSpendTracker()
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.credit_telemetry"):
            await tracker.async_log_success_event(
                _success_kwargs(llm_call_metadata("forecaster:openai", DONATED_KEY_ALIAS)),
                _response_with_usage(prompt_tokens=17_000, cost=0.25),
                None,
                None,
            )
        assert _alert_lines(caplog) == []

    async def test_a_prompt_exactly_at_the_threshold_is_silent(self, caplog) -> None:
        tracker = RoleSpendTracker()
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.credit_telemetry"):
            await tracker.async_log_success_event(
                _success_kwargs(llm_call_metadata("parser", DONATED_KEY_ALIAS)),
                _response_with_usage(prompt_tokens=PROMPT_TOKENS_ALERT_THRESHOLD, cost=0.01),
                None,
                None,
            )
        assert _alert_lines(caplog) == []

    async def test_roster_call_without_a_question_reads_na_and_parses_under_the_registry_regex(self, caplog) -> None:
        """A roster LLM carries no question in its metadata; the line says so rather than dropping
        the field, and the registry regex still harvests it (with ``qid`` None)."""
        tracker = RoleSpendTracker()
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.credit_telemetry"):
            await tracker.async_log_success_event(
                _success_kwargs(llm_call_metadata("forecaster:anthropic", DONATED_KEY_ALIAS)),
                _response_with_usage(prompt_tokens=512_000, cost=1.5),
                None,
                None,
            )

        (line,) = _alert_lines(caplog)
        assert " question=n/a " in line
        spec = next(s for s in MARKER_SPECS if s.name == "prompt_size_alert")
        match = spec.regex.search(line)
        assert match is not None
        assert (match.group("role"), match.group("question")) == ("forecaster:anthropic", "n/a")
        assert (match.group("prompt_tokens"), match.group("threshold")) == (
            "512000",
            str(PROMPT_TOKENS_ALERT_THRESHOLD),
        )


def _summary_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("CREDIT_RUN_SUMMARY:")]


@pytest.mark.usefixtures("clean_role_ledger")
class TestRunSpendSummary:
    """One ``CREDIT_RUN_SUMMARY`` line per run: the ledger folded down to cost per question.

    The documented per-question figure was wrong by five-fold for two months because the only
    per-run instrument printed per role with no question denominator (2026-09-09 cost pass,
    section 6). This line puts the denominator beside the money on every run.
    """

    def _ledger(self) -> None:
        record_llm_call_spend(
            "forecaster:openai",
            DONATED_KEY_ALIAS,
            cost_usd=0.0,
            byok_upstream_usd=1.20,
            is_byok=True,
            tokens=TokenCounts(prompt=68_000, completion=9_000, cached=0, reasoning=8_000),
        )
        record_llm_call_spend(
            "forecaster:google",
            PERSONAL_KEY_ALIAS,
            cost_usd=0.55,
            byok_upstream_usd=0.55,
            is_byok=False,
            tokens=TokenCounts(prompt=70_000, completion=5_000, cached=0, reasoning=4_000),
        )
        for prompt in (3_000, 41_000):
            record_llm_call_spend(
                "gap_fill_v2_driver",
                DONATED_KEY_ALIAS,
                cost_usd=0.0,
                byok_upstream_usd=0.25,
                is_byok=True,
                tokens=TokenCounts(prompt=prompt, completion=800, cached=prompt - 1_000, reasoning=500),
            )
        record_llm_call_spend("perplexity_research", DIRECT_KEY_ALIAS, cost_usd=None, byok_upstream_usd=None)

    def test_summary_folds_the_ledger_down_to_dollars_per_question(self) -> None:
        self._ledger()

        summary = run_spend_summary(n_questions=4)

        assert summary.n_questions == 4
        assert summary.charged_usd == pytest.approx(1.20 + 0.55 + 0.50)
        assert summary.usd_per_question == pytest.approx(2.25 / 4)
        assert summary.donated_usd == pytest.approx(1.70)
        assert summary.personal_usd == pytest.approx(0.55)
        assert summary.prompt_tokens == 68_000 + 70_000 + 44_000
        assert summary.cached_tokens == 2_000 + 40_000
        assert summary.cached_share == pytest.approx(42_000 / 182_000)
        assert (summary.max_prompt_tokens, summary.max_prompt_role) == (70_000, "forecaster:google")

    def test_line_shape_is_the_registry_contract(self, caplog) -> None:
        self._ledger()
        spec = next(s for s in MARKER_SPECS if s.name == "credit_run_summary")

        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_run_summary(n_questions=4)

        (line,) = _summary_lines(caplog)
        assert line == (
            "CREDIT_RUN_SUMMARY: n_questions=4 charged_usd=2.2500 usd_per_question=0.5625"
            " donated_usd=1.7000 personal_usd=0.5500 prompt_tokens=182000 cached_tokens=42000 cached_share=0.2308"
            " max_prompt_tokens=70000 max_prompt_role=forecaster:google"
        )
        assert spec.regex.search(line) is not None

    def test_zero_questions_reports_the_money_with_no_fabricated_rate(self, caplog) -> None:
        """A run that spent and forecast nothing it could publish (or crashed before the reports came
        back) still reports what it booked; the per-question rate is unknown, not zero."""
        self._ledger()

        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_run_summary(n_questions=0)

        (line,) = _summary_lines(caplog)
        assert " n_questions=0 charged_usd=2.2500 usd_per_question=n/a " in line

    def test_a_key_with_no_rows_reads_zero_and_a_key_with_only_uncosted_rows_reads_na(self, caplog) -> None:
        """A Mantic run never touches the donated key, so its donated total is a true 0.0000; a key
        whose every call OpenRouter left uncosted has an UNKNOWN total and must not read as free."""
        record_llm_call_spend("forecaster:openai", PERSONAL_KEY_ALIAS, cost_usd=None, byok_upstream_usd=None)

        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_run_summary(n_questions=1)

        (line,) = _summary_lines(caplog)
        assert " charged_usd=n/a usd_per_question=n/a donated_usd=0.0000 personal_usd=n/a " in line

    def test_empty_ledger_still_emits_one_line(self, caplog) -> None:
        """Emitted on every path, like the alertable breakdown, so a run with no completions leaves a
        record the archive can count rather than an absence it cannot."""
        spec = next(s for s in MARKER_SPECS if s.name == "credit_run_summary")

        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_run_summary(n_questions=0)

        (line,) = _summary_lines(caplog)
        assert line == (
            "CREDIT_RUN_SUMMARY: n_questions=0 charged_usd=n/a usd_per_question=n/a donated_usd=0.0000"
            " personal_usd=0.0000 prompt_tokens=0 cached_tokens=0 cached_share=n/a max_prompt_tokens=0"
            " max_prompt_role=none"
        )
        match = spec.regex.search(line)
        assert match is not None
        assert match.group("max_prompt_role") == "none"


class TestProdLlmsAreRoleTagged:
    """Every LLM ``llm_configs`` builds for prod carries its role tag, and the roster slots
    derive theirs from the slug. Roster-agnostic on purpose: a swap must not be able to
    leave a slot booking as ``untagged``."""

    def test_roster_slots_are_tagged_by_vendor(self) -> None:
        assert FORECASTER_LLMS, "roster must be non-empty for this pin to mean anything"
        for llm in FORECASTER_LLMS:
            assert llm.litellm_kwargs["metadata"][ROLE_METADATA_KEY] == forecaster_role(llm.model)

    @pytest.mark.parametrize(
        ("llm", "role"),
        [
            (SUMMARIZER_LLM, "summarizer"),
            (PARSER_LLM, "parser"),
            (STACKER_LLM, "stacker"),
            (STACKER_FALLBACK_LLM, "stacker_fallback"),
            (DISAGREEMENT_ANALYZER_LLM, "crux_analyzer"),
        ],
    )
    def test_support_slots_carry_their_role(self, llm: GeneralLlm, role: str) -> None:
        assert llm.litellm_kwargs["metadata"][ROLE_METADATA_KEY] == role

    def test_market_stage_configs_carry_their_role(self) -> None:
        """Raw dicts fed to build_llm_with_openrouter_fallback(**config) at call time."""
        assert MARKET_RANKER_LLM_CONFIG["role"] == "market_ranker"
        assert MARKET_QUERY_AUTHOR_LLM_CONFIG["role"] == "market_query_author"
