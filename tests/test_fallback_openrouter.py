import logging
import time
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import GeneralLlm
from litellm.exceptions import APIError, RateLimitError, ServiceUnavailableError

from metaculus_bot import fallback_openrouter
from metaculus_bot.constants import CREDIT_ALERT_RESUME_DATE
from metaculus_bot.credit_telemetry import (
    DIRECT_KEY_ALIAS,
    DONATED_KEY_ALIAS,
    PERSONAL_KEY_ALIAS,
    UNTAGGED_ROLE,
    DonatedKeyState,
    llm_call_metadata,
    reset_donated_key_state_cache,
)
from metaculus_bot.fallback_openrouter import (
    DONATED_KEY_PROVIDERS,
    FallbackOpenRouterLlm,
    build_llm_with_openrouter_fallback,
    get_credit_key_fallback_count,
    get_donated_404_fallback_count,
    get_generic_key_fallback_count,
    is_credit_caused_error,
    is_suppressible_credit_error,
    reset_credit_key_fallback_count,
    reset_donated_404_fallback_count,
    reset_generic_key_fallback_count,
    should_retry_with_general_key,
    should_route_via_donated_key,
)

# The verbatim 2026-07-26 production 403 (HTTP 403 rather than the documented 402, the
# phrase "Key limit exceeded (total limit)", a "code":403 field the old negative rule
# matched on, and a key hash carrying none of 401/402/403/429). Shared from conftest so
# every suite replaying it asserts against the same bytes — the constant's own comment
# explains why that matters.
from tests.conftest import PRODUCTION_KEY_LIMIT_403


def _api_error(status: int, message: str) -> APIError:
    """A litellm ``APIError`` that REPORTS ``status``, the shape every OpenRouter failure
    arrives in. The reported int is what the classification reads; the message is the
    untrusted half (key hash, replayed prompt), so tests vary it freely."""
    return APIError(status_code=status, message=message, llm_provider="openrouter", model="openai/gpt-5.6-sol")


# A realistic OpenRouter moderation 403. The body carries `reasons` and
# `flagged_input` (up to ~100 chars of OUR OWN PROMPT replayed back), which is why
# a numeric substring match on the whole message is unsafe.
MODERATION_403 = (
    "litellm.APIError: APIError: OpenrouterException - "
    '{"error":{"message":"Your input was flagged for the following reasons: violence",'
    '"code":403,"metadata":{"reasons":["violence"],'
    '"flagged_input":"Will the conflict escalate before Aug 18 2026?",'
    '"provider_name":"OpenAI","model_slug":"openai/gpt-5.6-sol"}}}'
)

# The same moderation 403 whose replayed prompt text contains the token "402" — a
# dollar figure or bill number is entirely ordinary in a forecasting prompt. The
# old bare-"402" substring cue would read this as an empty wallet, bill the
# personal key for a call that will refuse again, and exempt a real moderation
# block from alerting.
MODERATION_403_ECHOING_402 = (
    "litellm.APIError: APIError: OpenrouterException - "
    '{"error":{"message":"Your input was flagged for the following reasons: violence",'
    '"code":403,"metadata":{"reasons":["violence"],'
    '"flagged_input":"Will the $402 billion defense package pass before Aug 18 2026?",'
    '"provider_name":"OpenAI","model_slug":"openai/gpt-5.6-sol"}}}'
)

# The footgun that dictates the cue wording. "limit exceeded" is a substring of
# this, so the short form would classify every rate-limit breach as credit-caused
# and silently exempt it from alerting for the whole suppression window.
FREE_MODEL_RATE_LIMIT_429 = (
    "litellm.RateLimitError: OpenrouterException - "
    '{"error":{"message":"Rate limit exceeded: free-models-per-day","code":429}}'
)


class TestPredicates:
    def test_should_route_via_donated_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # OpenAI / Anthropic always route via the donated key when one is configured.
        assert should_route_via_donated_key("openrouter/openai/gpt-5.1") is True
        assert should_route_via_donated_key("openrouter/anthropic/claude-sonnet-4") is True
        # Google is gated on GEMINI_USE_DONATED_OPENROUTER_KEY. With the toggle on,
        # flash models prefer the donated key — but gemini-3.1-pro is on the
        # DONATED_KEY_BLOCKED_GOOGLE_MODELS blocklist (free-tier BYOK → 429), so it
        # is pinned to the personal key even with the toggle ON.
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")
        assert should_route_via_donated_key("openrouter/google/gemini-3.5-flash") is True
        assert should_route_via_donated_key("openrouter/google/gemini-3.1-flash-lite") is True
        assert should_route_via_donated_key("openrouter/google/gemini-3-flash-preview") is True
        assert should_route_via_donated_key("openrouter/google/gemini-3.1-pro-preview") is False
        # Explicit toggle off: ALL Google calls go through the operator's personal key only.
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "false")
        assert should_route_via_donated_key("openrouter/google/gemini-3.1-pro-preview") is False
        assert should_route_via_donated_key("openrouter/google/gemini-3.5-flash") is False
        # Default with the env var unset is ON (donated key with personal fallback):
        # after Metaculus raised the Google rate limits (2026-06-16) the donated key
        # serves most Gemini. A flash model routes donated-first by default; the
        # gemini-3.1-pro slug stays blocklisted (pinned to personal) regardless.
        monkeypatch.delenv("GEMINI_USE_DONATED_OPENROUTER_KEY", raising=False)
        assert should_route_via_donated_key("openrouter/google/gemini-3.5-flash") is True
        assert should_route_via_donated_key("openrouter/google/gemini-3.1-pro-preview") is False
        # Providers NOT covered by the donated key.
        assert should_route_via_donated_key("openrouter/x-ai/grok-4.1-fast") is False
        assert should_route_via_donated_key("openrouter/qwen/qwen3-235b") is False
        # Non-OpenRouter slugs.
        assert should_route_via_donated_key("perplexity/sonar") is False
        # Defensive: bogus inputs return False rather than raising.
        assert should_route_via_donated_key("openrouter/") is False  # parts < 2
        assert should_route_via_donated_key("") is False

    def test_master_switch_off_pins_every_provider_to_personal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DONATED_OPENROUTER_KEY_ENABLED=false (a Mantic run) wins over every per-provider rule.

        Metaculus donated the key for its own tournaments, so a run for another platform
        must never route through it: OpenAI and Anthropic (unconditionally donated when
        the switch is on) and Google (even with its own toggle explicitly ON) all read
        False. This predicate is the single gate for the builder, ``api_key_utils`` and
        the gap-fill v2 transport, so False here is personal-only routing everywhere.
        """
        monkeypatch.setenv("DONATED_OPENROUTER_KEY_ENABLED", "false")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")
        assert should_route_via_donated_key("openrouter/openai/gpt-5.6-sol") is False
        assert should_route_via_donated_key("openrouter/anthropic/claude-opus-4.8") is False
        assert should_route_via_donated_key("openrouter/google/gemini-3.5-flash") is False

    def test_master_switch_unset_or_true_leaves_routing_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default ON: a Metaculus run that never sets the switch routes exactly as before."""
        monkeypatch.delenv("GEMINI_USE_DONATED_OPENROUTER_KEY", raising=False)
        for raw in (None, "true"):
            if raw is None:
                monkeypatch.delenv("DONATED_OPENROUTER_KEY_ENABLED", raising=False)
            else:
                monkeypatch.setenv("DONATED_OPENROUTER_KEY_ENABLED", raw)
            assert should_route_via_donated_key("openrouter/openai/gpt-5.6-sol") is True
            assert should_route_via_donated_key("openrouter/anthropic/claude-opus-4.8") is True
            assert should_route_via_donated_key("openrouter/google/gemini-3.5-flash") is True
            assert should_route_via_donated_key("openrouter/google/gemini-3.1-pro-preview") is False
            assert should_route_via_donated_key("openrouter/x-ai/grok-4.1-fast") is False

    def test_donated_key_providers_set(self) -> None:
        # Pin the membership so any drift surfaces in code review rather than
        # silently changing routing.
        assert frozenset({"openai", "anthropic", "google"}) == DONATED_KEY_PROVIDERS

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("HTTP 402 Payment Required", True),
            ("payment required", True),
            ("insufficient credit on key", True),
            ("401 Unauthorized", True),
            ("invalid API key", True),
            ("disabled api key", True),
            # Donated-key allowed-providers 404 quirk → should fall back.
            ("404 No allowed providers are available for the selected model.", True),
            ("no allowed providers", True),
            # Donated-key data-policy / guardrail 404 (added 2026-05-17 for OpenAI
            # native search migration) — donated key's data-collection guardrail
            # blocks OpenAI native search; the personal key has no such block.
            (
                "404 No endpoints available matching your guardrail restrictions and data policy. "
                "Configure: https://openrouter.ai/settings/privacy",
                True,
            ),
            ("matching your guardrail restrictions", True),
            ("data policy", True),
            # 429 rate-limit: textual detection falls back (BYOK quotas are per-key).
            ("429 Too Many Requests", True),
            ("Rate limit exceeded", True),
            # Belt-and-suspenders textual patterns for 429 edge cases.
            ('{"code":429, "message": "rate limited"}', True),
            ("rate-limited upstream by provider", True),
            # Negative: moderation / infrastructure errors do NOT fall back.
            ("403 Forbidden moderation", False),
            ("502 Bad Gateway", False),
            ("503 Service Unavailable", False),
        ],
    )
    def test_should_retry_with_general_key(self, message: str, expected: bool) -> None:
        assert should_retry_with_general_key(Exception(message)) is expected

    def test_litellm_rate_limit_error_triggers_fallback(self) -> None:
        """litellm.RateLimitError (typed 429) triggers fallback — BYOK quotas are independent."""
        exc = RateLimitError(
            message="Rate limit exceeded on openrouter",
            model="openrouter/google/gemini-3.1-pro-preview",
            llm_provider="openrouter",
        )
        assert should_retry_with_general_key(exc) is True

    def test_litellm_service_unavailable_does_not_trigger_fallback(self) -> None:
        """litellm.ServiceUnavailableError (503) does NOT trigger fallback — infrastructure issue."""
        exc = ServiceUnavailableError(
            message="503 Service Unavailable",
            model="openrouter/openai/gpt-5.1",
            llm_provider="openrouter",
        )
        assert should_retry_with_general_key(exc) is False


class TestFallbackOpenRouterLlm:
    @pytest.mark.asyncio
    async def test_primary_success_no_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.1",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        # Patch the internal primary call point to avoid network.
        monkeypatch.setattr(llm, "_invoke_once_using_primary", AsyncMock(return_value="answer"))

        out = await llm.invoke("hi")
        assert out == "answer"

    @pytest.mark.asyncio
    async def test_fallback_on_402(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/anthropic/claude-sonnet-4",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("HTTP 402 Payment Required: insufficient credit")),
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        out = await llm.invoke("hi")
        assert out == "ok"

    @pytest.mark.asyncio
    async def test_fallback_on_429_rate_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """429 on primary key falls back to secondary — BYOK quotas are independent."""
        llm = FallbackOpenRouterLlm(
            model="openrouter/google/gemini-3.1-pro-preview",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        exc = RateLimitError(
            message="Rate limit exceeded",
            model="openrouter/google/gemini-3.1-pro-preview",
            llm_provider="openrouter",
        )
        monkeypatch.setattr(llm, "_invoke_once_using_primary", AsyncMock(side_effect=exc))
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="fallback_ok"))

        out = await llm.invoke("hi")
        assert out == "fallback_ok"

    @pytest.mark.asyncio
    async def test_no_fallback_on_403(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.1",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("403 Forbidden moderation")),
        )
        # Stubbing the secondary is what makes "no fallback" assertable. Left
        # unpatched, a regression that DID fall back would hit the autouse
        # network-egress guard and raise its own RuntimeError, which the bare
        # `raises(Exception)` would happily accept — so the test would still pass
        # while the behavior it guards had inverted. The call-count assertion is
        # the real content; the raise is secondary.
        secondary = AsyncMock(return_value="fallback_ok")
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", secondary)

        with pytest.raises(Exception, match="403 Forbidden moderation"):
            await llm.invoke("hi")
        secondary.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_fallback_on_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """503 Service Unavailable re-raises without fallback — infrastructure issue, not key-scoped."""
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.1",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        exc = ServiceUnavailableError(
            message="503 Service Unavailable",
            model="openrouter/openai/gpt-5.1",
            llm_provider="openrouter",
        )
        monkeypatch.setattr(llm, "_invoke_once_using_primary", AsyncMock(side_effect=exc))

        with pytest.raises(ServiceUnavailableError):
            await llm.invoke("hi")

    @pytest.mark.asyncio
    async def test_no_secondary_key_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.1",
            primary_api_key="special",
            secondary_api_key=None,
            temperature=0,
        )

        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("401 Unauthorized")),
        )
        # A 401 IS a fallback-worthy cause, so what this test pins is that the
        # missing secondary key — not the error class — is what stops the retry.
        # Stubbing the secondary keeps the assertion honest: without it, a
        # regression that attempted the fallback anyway would raise the
        # network-egress guard's RuntimeError and still satisfy a bare
        # `raises(Exception)`. The original 401 must be what propagates.
        secondary = AsyncMock(return_value="fallback_ok")
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", secondary)

        with pytest.raises(Exception, match="401 Unauthorized"):
            await llm.invoke("hi")
        secondary.assert_not_called()


class TestKeyLimitExceeded403:
    """OpenRouter reports a per-key SPEND-CAP breach as HTTP 403 with the text
    "Key limit exceeded (total limit)" — not the 402 its own error docs document.

    litellm has no 403 branch for OpenRouter, so this always arrives as a bare
    ``APIError``, and the old negative rule (``"403" in msg`` → never fall back,
    written for content moderation) matched the ``"code":403`` field in the JSON
    body. Two of three forecasters, native search, the AskNews summarizer, the
    financial classifier, prediction-market keyword extraction, gap-fill v1 and
    gap-fill v2 all died on a run where the operator's funded personal key sat
    idle. These tests pin the classification in both directions: a spend-cap 403
    falls back, a moderation 403 still does not.
    """

    def test_production_message_falls_back_and_is_credit_classified(self) -> None:
        """The exact string from the 2026-07-26 run log. Both predicates must fire:
        the retry decision routes the call to the funded key, and the credit
        classification is what lets cli exempt the expected empty wallet from CI
        alerting. Neither reads the network — both are textual.
        """
        exc = Exception(PRODUCTION_KEY_LIMIT_403)
        assert should_retry_with_general_key(exc) is True
        assert is_credit_caused_error(exc) is True

    @pytest.mark.parametrize(
        "message",
        [
            "Key limit exceeded (total limit).",
            "key limit exceeded (daily limit)",
            'OpenrouterException - {"error":{"message":"Key limit exceeded (monthly limit).","code":403}}',
        ],
    )
    def test_key_limit_variants_all_classify_as_credit(self, message: str) -> None:
        """The cue is the phrase, not the parenthetical, so the daily and monthly
        cap variants are covered without enumerating them.
        """
        assert is_credit_caused_error(Exception(message)) is True
        assert should_retry_with_general_key(Exception(message)) is True

    def test_moderation_403_does_not_fall_back_and_is_not_credit(self) -> None:
        """Negative control. A moderation block is not key-scoped — the personal key
        would refuse the same prompt — so it must keep failing closed.
        """
        exc = Exception(MODERATION_403)
        assert should_retry_with_general_key(exc) is False
        assert is_credit_caused_error(exc) is False

    def test_moderation_403_echoing_402_in_prompt_is_still_moderation(self) -> None:
        """The nasty variant, and a latent defect this fix closes rather than adds.

        OpenRouter replays up to ~100 characters of our own prompt as
        ``flagged_input``, and forecasting prompts are full of dollar figures and
        bill numbers. A bare "402" substring match on the whole message therefore
        reads an ordinary moderation refusal as an empty wallet — which would both
        bill the personal key for a call that refuses again and exempt a real
        moderation block from alerting.
        """
        exc = Exception(MODERATION_403_ECHOING_402)
        assert "402" in str(exc)
        assert should_retry_with_general_key(exc) is False
        assert is_credit_caused_error(exc) is False

    def test_free_model_rate_limit_429_is_not_credit_classified(self) -> None:
        """The footgun regression test: this is why the cue is "key limit exceeded"
        and not the shorter "limit exceeded", which is a substring of
        "Rate limit exceeded: free-models-per-day". Shortening the cue would
        classify every 429 as credit-caused and silently exempt real rate-limit
        breakage from alerting for the whole suppression window.
        """
        exc = Exception(FREE_MODEL_RATE_LIMIT_429)
        assert is_credit_caused_error(exc) is False
        # It still falls back — a 429 is key-scoped (BYOK quotas are per-key) — it
        # just isn't exempt from alerting.
        assert should_retry_with_general_key(exc) is True

    def test_plain_402_still_falls_back_without_a_403_in_the_body(self) -> None:
        """The documented 402 path is untouched by the moderation hardening."""
        exc = Exception('{"error":{"message":"Insufficient credits","code":402}}')
        assert should_retry_with_general_key(exc) is True
        assert is_credit_caused_error(exc) is True


class TestCreditSuppressibility:
    """``is_suppressible_credit_error`` — the drained-vs-revoked discriminator.

    The operator's decision: a genuinely DRAINED donated key must not redden CI
    while they self-fund the season, but a key Metaculus revoked or re-capped to
    zero must stay red. Both produce identical "Key limit exceeded" text, so the
    text cue alone cannot make that call — we probe OpenRouter's free
    ``/auth/key`` endpoint and only DRAINED is exempt.

    Fallback ROUTING is deliberately not gated on the probe: a stale or cached
    balance read must never be able to strand the ensemble on a dry key.
    """

    @pytest.fixture(autouse=True)
    def _no_real_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default every test in this class to "probe was never able to run", so a
        test that forgets to pin a state can't reach the network.
        """
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        reset_donated_key_state_cache()

    def _pin_state(self, monkeypatch: pytest.MonkeyPatch, state: DonatedKeyState) -> MagicMock:
        probe = MagicMock(return_value=state)
        monkeypatch.setattr(fallback_openrouter, "classify_donated_key_state", probe)
        return probe

    def test_drained_key_is_suppressible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._pin_state(monkeypatch, DonatedKeyState.DRAINED)
        assert is_suppressible_credit_error(Exception(PRODUCTION_KEY_LIMIT_403)) is True

    @pytest.mark.parametrize(
        "state",
        [DonatedKeyState.REVOKED, DonatedKeyState.ZEROED, DonatedKeyState.FUNDED, DonatedKeyState.UNKNOWN],
    )
    def test_every_other_state_stays_alertable(self, monkeypatch: pytest.MonkeyPatch, state: DonatedKeyState) -> None:
        """The regression that matters: a revoked key, a key re-capped to zero, a
        key that still has money, and a probe that failed all keep CI red.
        """
        self._pin_state(monkeypatch, state)
        assert is_suppressible_credit_error(Exception(PRODUCTION_KEY_LIMIT_403)) is False

    def test_probe_failure_fails_safe_to_alertable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No donated key configured → the probe can't run → UNKNOWN → still red.
        A broken probe must never be able to silently green a run.
        """
        assert is_suppressible_credit_error(Exception(PRODUCTION_KEY_LIMIT_403)) is False

    def test_documented_402_needs_no_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 402 / insufficient-credit response is unambiguous: the wallet is empty.
        Only the ambiguous spend-cap 403 pays for a probe, so the long-standing 402
        path keeps working with the endpoint unreachable.
        """
        probe = self._pin_state(monkeypatch, DonatedKeyState.UNKNOWN)
        assert is_suppressible_credit_error(Exception("HTTP 402 Payment Required: insufficient credit")) is True
        probe.assert_not_called()

    @pytest.mark.parametrize("message", ["401 Unauthorized", "404 no allowed providers", MODERATION_403])
    def test_non_credit_causes_are_never_suppressible(self, monkeypatch: pytest.MonkeyPatch, message: str) -> None:
        probe = self._pin_state(monkeypatch, DonatedKeyState.DRAINED)
        assert is_suppressible_credit_error(Exception(message)) is False
        probe.assert_not_called()

    def test_reported_429_echoing_402_is_not_exempted_from_alerting(self) -> None:
        """A rate limit must stay alertable even when the body echoes "402".

        The mirror image of the terse-402 case, and the more dangerous direction. A
        reported 429 whose replayed prompt contains "402" and no moderation word
        routes correctly on the status — but while the classification was text-only
        it read the bare digits as an empty wallet, so a genuine rate-limit fallback
        was subtracted from ``alertable`` for the whole suppression window. Same
        failure the synthesis predicted for a too-short credit cue, reached through
        the numeric door instead.
        """
        body = (
            "OpenrouterException - "
            '{"error":{"message":"Rate limited","code":429,'
            '"metadata":{"input":"Will the $402 billion package pass before Aug 18 2026?"}}}'
        )
        exc = RateLimitError(message=body, model="openrouter/openai/gpt-5.6-sol", llm_provider="openrouter")
        assert "402" in str(exc)  # the poisoning cue is present
        assert should_retry_with_general_key(exc) is True  # 429 is key-scoped: still falls back
        assert is_credit_caused_error(exc) is False
        assert is_suppressible_credit_error(exc) is False

    def test_terse_reported_402_is_credit_classified_like_it_routes(self) -> None:
        """Routing and alerting must answer "was this about money?" identically.

        A reported 402 whose prose spells out neither a credit phrase nor the digits
        (litellm renders it as just ``litellm.APIError: wallet empty``) routes to the
        paid key on the reported status. While the classification stayed text-only it
        did NOT count as credit-caused, so the run reddened CI on precisely the
        expected empty wallet the suppression window exists for.
        """
        exc = _api_error(402, "wallet empty")
        assert "402" not in str(exc)  # the gap only exists because the digits are absent
        assert should_retry_with_general_key(exc) is True
        assert is_credit_caused_error(exc) is True
        assert is_suppressible_credit_error(exc) is True


class TestCreditCauseClassification:
    """``is_credit_caused_error`` is the ONE place the "empty wallet" text cues
    live: ``should_retry_with_general_key`` and the credit-subset counter both go
    through it, so the retry decision and the counter can't drift apart.

    Only these messages are exempt from alerting inside a suppression window (see
    ``constants.credit_alerts_active``, closed since 2026-09-03) — everything else
    keeps reddening CI whatever the window says.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "HTTP 402 Payment Required",
            "payment required",
            "insufficient credit on key",
            "out of credits",
            "insufficient funds",
            "402 Payment Required: insufficient credit",
        ],
    )
    def test_credit_messages_classified_as_credit_caused(self, message: str) -> None:
        assert is_credit_caused_error(Exception(message)) is True
        # Same messages must still trigger the fallback itself — the exemption
        # changes accounting, never routing.
        assert should_retry_with_general_key(Exception(message)) is True

    @pytest.mark.parametrize(
        "message",
        [
            "401 Unauthorized",
            "invalid API key",
            "disabled api key",
            "404 No allowed providers are available for the selected model.",
            "429 Too Many Requests",
            "Rate limit exceeded",
            "matching your guardrail restrictions",
            "data policy",
            "403 Forbidden moderation",
            "503 Service Unavailable",
        ],
    )
    def test_non_credit_messages_not_classified_as_credit_caused(self, message: str) -> None:
        """The regression that matters: real breakage must never be exempted."""
        assert is_credit_caused_error(Exception(message)) is False


class TestPaidFallbackWarningExitClause:
    """The paid-fallback WARNING must not promise a red CI run it won't deliver.

    ``credit_alerts_active`` is monkeypatched (rather than reading the wall clock)
    so both branches stay covered either side of the 2026-09-03 resume date. The
    ``suppressible`` flag is passed in by the caller, which has already paid for
    the donated-key probe — the note renders the decision, it doesn't re-derive it.
    """

    def test_suppressible_credit_cause_during_window_says_not_alertable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fallback_openrouter, "credit_alerts_active", lambda: False)
        note = fallback_openrouter._fallback_alert_note(suppressible=True)
        assert "NOT counted as alertable" in note
        assert CREDIT_ALERT_RESUME_DATE.isoformat() in note

    def test_credit_cause_after_resume_says_exit_non_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fallback_openrouter, "credit_alerts_active", lambda: True)
        note = fallback_openrouter._fallback_alert_note(suppressible=True)
        assert note == "Run will complete, then exit non-zero to alert."

    def test_non_credit_cause_always_says_exit_non_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even mid-suppression, a 401 still promises the red run it will cause."""
        monkeypatch.setattr(fallback_openrouter, "credit_alerts_active", lambda: False)
        note = fallback_openrouter._fallback_alert_note(suppressible=False)
        assert note == "Run will complete, then exit non-zero to alert."

    def test_revoked_donated_key_is_not_promised_as_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A credit-SHAPED failure on a revoked key is not suppressible, so the
        note must promise the red run it will actually cause. Before the
        discriminator, the note keyed on the text cue alone and would have claimed
        the run stays green.
        """
        monkeypatch.setattr(fallback_openrouter, "credit_alerts_active", lambda: False)
        note = fallback_openrouter._fallback_alert_note(suppressible=False)
        assert "NOT counted as alertable" not in note


class TestFallbackCounters:
    """Every donated->personal fallback must be counted + logged loudly so silent
    personal-key spend can't accumulate. ``_generic_key_fallback_count`` counts ALL
    fallback causes; ``_donated_404_fallback_count`` (allowed-providers 404) and
    ``_credit_key_fallback_count`` (402/insufficient-credit) are two disjoint
    subsets of that total. cli.py folds the generic counter into its non-zero-exit
    alert and subtracts the credit subset while credit alerting is suppressed.
    """

    def setup_method(self) -> None:
        reset_generic_key_fallback_count()
        reset_donated_404_fallback_count()
        reset_credit_key_fallback_count()

    def teardown_method(self) -> None:
        reset_generic_key_fallback_count()
        reset_donated_404_fallback_count()
        reset_credit_key_fallback_count()

    @pytest.mark.asyncio
    async def test_credit_fallback_bumps_credit_subset_and_still_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 402 fallback bumps the generic total AND the credit subset, and still
        logs the loud PAID PERSONAL-KEY FALLBACK warning. The suppression only
        removes the event from cli's alertable arithmetic — never from the logs.
        """
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.6-sol",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("HTTP 402 Payment Required: insufficient credit")),
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.fallback_openrouter"):
            assert await llm.invoke("hi") == "ok"

        assert get_generic_key_fallback_count() == 1
        assert get_credit_key_fallback_count() == 1
        assert get_donated_404_fallback_count() == 0
        assert any("PAID PERSONAL-KEY FALLBACK" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "message",
        [
            "401 Unauthorized: invalid api key",
            "429 Too Many Requests: rate limit exceeded",
        ],
    )
    async def test_non_credit_fallback_leaves_credit_subset_at_zero(
        self, monkeypatch: pytest.MonkeyPatch, message: str
    ) -> None:
        """401 / 429 bump only the generic total. If either leaked into the credit
        subset, cli would subtract it and a genuinely broken key would ship green.
        """
        llm = FallbackOpenRouterLlm(
            model="openrouter/anthropic/claude-opus-4.8",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(llm, "_invoke_once_using_primary", AsyncMock(side_effect=Exception(message)))
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        assert await llm.invoke("hi") == "ok"
        assert get_generic_key_fallback_count() == 1
        assert get_credit_key_fallback_count() == 0

    @pytest.mark.asyncio
    async def test_donated_404_leaves_credit_subset_at_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The two subsets are disjoint: a 404 belongs to the 404 subset only, so
        cli's single subtraction can't double-discount it.
        """
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.4",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("404 No allowed providers are available for the selected model.")),
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        assert await llm.invoke("hi") == "ok"
        assert get_generic_key_fallback_count() == 1
        assert get_donated_404_fallback_count() == 1
        assert get_credit_key_fallback_count() == 0

    @pytest.mark.asyncio
    async def test_generic_fallback_bumps_counter_and_logs_every_time(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 401/402 (non-404) fallback bumps ONLY the generic counter, not the 404 subset,
        and logs a WARNING on EVERY fallback (no once-per-instance suppression)."""
        llm = FallbackOpenRouterLlm(
            model="openrouter/anthropic/claude-opus-4.8",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("HTTP 402 Payment Required: insufficient credit")),
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.fallback_openrouter"):
            assert await llm.invoke("hi") == "ok"
            assert await llm.invoke("hi") == "ok"

        # Generic counter bumped on both fallbacks; 404 subset untouched.
        assert get_generic_key_fallback_count() == 2
        assert get_donated_404_fallback_count() == 0
        # Loud WARNING on every fallback, not just the first.
        paid_warnings = [r for r in caplog.records if "PAID PERSONAL-KEY FALLBACK" in r.getMessage()]
        assert len(paid_warnings) == 2

    @pytest.mark.asyncio
    async def test_key_limit_403_on_drained_key_bumps_credit_subset(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The production failure, end to end through the wrapper: the spend-cap 403
        now falls back to the personal key, and because the probe confirms the
        donated key is genuinely drained, the event lands in the suppressible credit
        subset so cli exits zero on it.
        """
        monkeypatch.setattr(
            fallback_openrouter, "classify_donated_key_state", MagicMock(return_value=DonatedKeyState.DRAINED)
        )
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.6-sol",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm, "_invoke_once_using_primary", AsyncMock(side_effect=Exception(PRODUCTION_KEY_LIMIT_403))
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.fallback_openrouter"):
            assert await llm.invoke("hi") == "ok"

        assert get_generic_key_fallback_count() == 1
        assert get_credit_key_fallback_count() == 1
        assert get_donated_404_fallback_count() == 0
        # Nothing is silenced by the suppression — the loud warning still fires.
        assert any("PAID PERSONAL-KEY FALLBACK" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_key_limit_403_on_revoked_key_stays_alertable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same error text, revoked key: still falls back (the run should publish if
        it can), but the credit subset stays empty so cli exits non-zero. This is
        the whole reason the discriminator exists — a text cue alone cannot tell
        these two runs apart.
        """
        monkeypatch.setattr(
            fallback_openrouter, "classify_donated_key_state", MagicMock(return_value=DonatedKeyState.REVOKED)
        )
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.6-sol",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm, "_invoke_once_using_primary", AsyncMock(side_effect=Exception(PRODUCTION_KEY_LIMIT_403))
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        assert await llm.invoke("hi") == "ok"
        assert get_generic_key_fallback_count() == 1
        assert get_credit_key_fallback_count() == 0

    @pytest.mark.asyncio
    async def test_moderation_403_still_raises_and_counts_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Negative control at the wrapper level: a moderation 403 must not reach the
        personal key and must not bump any counter.
        """
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.6-sol",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm, "_invoke_once_using_primary", AsyncMock(side_effect=Exception(MODERATION_403_ECHOING_402))
        )
        secondary = AsyncMock(return_value="should not be reached")
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", secondary)

        with pytest.raises(Exception, match="flagged"):
            await llm.invoke("hi")

        secondary.assert_not_awaited()
        assert get_generic_key_fallback_count() == 0
        assert get_credit_key_fallback_count() == 0

    @pytest.mark.asyncio
    async def test_donated_404_fallback_bumps_both_counters(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 404 'no allowed providers' fallback bumps BOTH the generic total and the 404 subset
        (a 404 fallback is still a personal-key fallback)."""
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.4",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("404 No allowed providers are available for the selected model.")),
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.fallback_openrouter"):
            assert await llm.invoke("hi") == "ok"

        assert get_generic_key_fallback_count() == 1
        assert get_donated_404_fallback_count() == 1
        assert any("no allowed providers" in r.getMessage() for r in caplog.records)


class TestBuilder:
    def test_builder_returns_wrapper_when_both_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5.1")
        assert isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_plain_when_only_general(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5.1")
        # Not wrapper, should be a GeneralLlm
        assert isinstance(llm, GeneralLlm)
        assert not isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_plain_for_non_donated_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Models from providers not in DONATED_KEY_PROVIDERS get a plain GeneralLlm
        (no donated-key wrapping). Grok via x-ai is the canonical example.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/x-ai/grok-4.1-fast")
        # Not wrapper, should be a GeneralLlm
        assert isinstance(llm, GeneralLlm)
        assert not isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_returns_wrapper_for_google_flash_when_donated_toggle_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flash Google models route via the donated wrapper when the toggle is ON.

        Originally added in task #12 (Google in DONATED_KEY_PROVIDERS). After the
        2026-06-16 rate-limit bump the donated key serves the flash models, so a
        flash slug returns a FallbackOpenRouterLlm (donated primary, personal
        fallback). gemini-3.1-pro is handled separately by the blocklist test.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")
        llm = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.5-flash")
        assert isinstance(llm, FallbackOpenRouterLlm)

    @pytest.mark.parametrize("model", ["gemini-3.1-pro-preview", "gemini-4-argon"])
    def test_builder_plain_for_google_pro_blocklisted_when_toggle_on(
        self, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        """Blocklisted Google models are pinned to the personal key even with
        the donated toggle ON — so the builder returns a plain GeneralLlm (no
        donated attempt, no 429, no fallback-counter bump), while a flash model in
        the same env returns a FallbackOpenRouterLlm.

        Temporary workaround; see DONATED_KEY_BLOCKED_GOOGLE_MODELS
        (``TODO(gemini-3.1-pro-donated)``) and FUTURE.md.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")
        assert should_route_via_donated_key(f"openrouter/google/{model}") is False
        pro = build_llm_with_openrouter_fallback(f"openrouter/google/{model}")
        assert isinstance(pro, GeneralLlm)
        assert not isinstance(pro, FallbackOpenRouterLlm)
        flash = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.5-flash")
        assert isinstance(flash, FallbackOpenRouterLlm)

    def test_builder_plain_for_google_when_donated_toggle_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the donated-key toggle off, ALL Google calls bypass the donated
        wrapper entirely — the resulting LLM is a plain GeneralLlm using the
        operator's general OpenRouter key.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "false")
        llm = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.5-flash")
        assert isinstance(llm, GeneralLlm)
        assert not isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_returns_wrapper_for_google_flash_when_toggle_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default (env var unset) is ON: flash Google calls prefer the donated wrapper.

        After Metaculus raised the Google rate limits (2026-06-16) the donated key
        serves most Gemini, so with two distinct keys configured a flash slug
        returns a FallbackOpenRouterLlm (donated primary, personal fallback).
        gemini-3.1-pro stays blocklisted (plain GeneralLlm) — see the dedicated
        blocklist test.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        monkeypatch.delenv("GEMINI_USE_DONATED_OPENROUTER_KEY", raising=False)
        llm = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.5-flash")
        assert isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_plain_personal_when_master_switch_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With DONATED_OPENROUTER_KEY_ENABLED=false and BOTH keys present, the builder returns a
        plain GeneralLlm billed to the personal key for every donated-eligible provider.

        This is the Mantic-run shape: the fallback wrapper (donated primary, personal secondary)
        must not be built, because its primary would spend Metaculus's key on another platform.
        The donated key never reaches the LLM's kwargs either; litellm reads OPENROUTER_API_KEY
        from the environment, which is what ``plain_llm_key_alias`` tags as ``personal``.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        monkeypatch.setenv("DONATED_OPENROUTER_KEY_ENABLED", "false")
        for model in (
            "openrouter/openai/gpt-5.6-sol",
            "openrouter/anthropic/claude-opus-4.8",
            "openrouter/google/gemini-3.5-flash",
        ):
            llm = build_llm_with_openrouter_fallback(model, role="parser")
            assert isinstance(llm, GeneralLlm)
            assert not isinstance(llm, FallbackOpenRouterLlm)
            assert llm.litellm_kwargs["metadata"] == llm_call_metadata("parser", PERSONAL_KEY_ALIAS)
            assert llm.litellm_kwargs.get("api_key") is None


class TestRoleTagging:
    """Every LLM the builder returns carries the ``CREDIT_ROLE_SPEND`` metadata tag, stamped
    with the key that instance actually bills — the ledger's join onto ``CREDIT_SPEND key=``
    depends on the alias being right per branch, not just present."""

    def test_wrapper_tags_primary_donated_and_secondary_personal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5.1", role="parser")
        assert isinstance(llm, FallbackOpenRouterLlm)
        assert llm.litellm_kwargs["metadata"] == llm_call_metadata("parser", DONATED_KEY_ALIAS)
        assert llm._secondary_llm is not None
        assert llm._secondary_llm.litellm_kwargs["metadata"] == llm_call_metadata("parser", PERSONAL_KEY_ALIAS)

    def test_single_key_branch_tags_whichever_key_it_uses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        personal_only = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5.1", role="summarizer")
        assert personal_only.litellm_kwargs["metadata"] == llm_call_metadata("summarizer", PERSONAL_KEY_ALIAS)

        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        donated_only = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5.1", role="summarizer")
        assert donated_only.litellm_kwargs["metadata"] == llm_call_metadata("summarizer", DONATED_KEY_ALIAS)

    def test_plain_branch_tags_personal_for_openrouter_and_direct_otherwise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        # Blocklisted google pro is pinned to the personal key with no donated attempt.
        pinned = build_llm_with_openrouter_fallback(
            "openrouter/google/gemini-3.1-pro-preview", role="forecaster:google"
        )
        assert not isinstance(pinned, FallbackOpenRouterLlm)
        assert pinned.litellm_kwargs["metadata"] == llm_call_metadata("forecaster:google", PERSONAL_KEY_ALIAS)
        # A non-OpenRouter slug bills its own provider's key: outside the ledger's remit,
        # but still counted under a label that says so.
        direct = build_llm_with_openrouter_fallback("perplexity/sonar-reasoning", role="perplexity_research")
        assert direct.litellm_kwargs["metadata"] == llm_call_metadata("perplexity_research", DIRECT_KEY_ALIAS)

    def test_missing_role_books_as_untagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5.1")
        assert llm.litellm_kwargs["metadata"]["role"] == UNTAGGED_ROLE


class TestDeprecationTripwire:
    """Post-submission deprecation tripwire (added 2026-05-17 after the
    2026-05-15 x-ai/grok-4.1-fast deprecation silently 404'd for ~2 days).
    """

    def test_records_deprecation_404(self) -> None:
        """A canonical OpenRouter deprecation 404 is recorded with model + msg."""
        from metaculus_bot.fallback_openrouter import (
            _DEPRECATION_ALERTS,
            _record_deprecation_if_matched,
            clear_deprecation_alerts,
        )

        clear_deprecation_alerts()
        msg = "Grok 4.1 Fast is deprecated. xAI recommends switching to Grok 4.3"
        matched = _record_deprecation_if_matched("x-ai/grok-4.1-fast", msg)
        assert matched is True
        assert [("x-ai/grok-4.1-fast", msg)] == _DEPRECATION_ALERTS
        clear_deprecation_alerts()

    def test_does_not_record_unrelated_error(self) -> None:
        """Plain 401/429/etc. don't match — false positives turn CI red without cause."""
        from metaculus_bot.fallback_openrouter import (
            _DEPRECATION_ALERTS,
            _record_deprecation_if_matched,
            clear_deprecation_alerts,
        )

        clear_deprecation_alerts()
        for msg in (
            "401 Unauthorized: invalid api key",
            "429 Too Many Requests: rate limit exceeded",
            "402 Payment Required: insufficient credit",
            "404 No allowed providers",
            "503 Service Unavailable",
        ):
            matched = _record_deprecation_if_matched("openrouter/openai/gpt-5.5", msg)
            assert matched is False, f"falsely matched: {msg}"
        assert _DEPRECATION_ALERTS == []
        clear_deprecation_alerts()

    def test_check_exits_on_populated_list(self) -> None:
        """Tripwire fires sys.exit(1) when at least one deprecation was seen."""
        from metaculus_bot.fallback_openrouter import (
            _record_deprecation_if_matched,
            check_deprecation_alerts_and_exit,
            clear_deprecation_alerts,
        )

        clear_deprecation_alerts()
        _record_deprecation_if_matched("x-ai/grok-4.1-fast", "Grok 4.1 Fast is deprecated.")
        with pytest.raises(SystemExit) as exc_info:
            check_deprecation_alerts_and_exit()
        assert exc_info.value.code == 1
        clear_deprecation_alerts()

    def test_check_returns_silently_on_empty(self) -> None:
        """No deprecation observed → tripwire is a no-op (run completes cleanly)."""
        from metaculus_bot.fallback_openrouter import (
            check_deprecation_alerts_and_exit,
            clear_deprecation_alerts,
        )

        clear_deprecation_alerts()
        # Must not raise SystemExit
        check_deprecation_alerts_and_exit()


class TestStatusCodeClassification:
    """Numeric routing reads the reported ``status_code``, not digits in the message.

    litellm formats the message as ``APIError: {provider} - {raw body}``. An OpenRouter
    body carries a 64-hex key hash with a small but non-negligible chance of containing
    one of 401/402/403/429/502/503 (derived exactly by
    ``test_key_hash_status_collision_is_small_but_nonnegligible``) and, on a moderation
    refusal, up to ~100 characters of OUR OWN PROMPT in ``flagged_input``. Grepping that
    text for status digits reads coincidences as statuses in both directions: a
    coincidental "429" makes a moderation 403 fall back and bill the personal key for a
    call that will refuse again, and a "402" in a forecasting prompt (a dollar figure, a
    bill number) reads as an empty wallet.

    Every litellm exception carries the real status as an int, so where one is reported
    it is the only numeric evidence consulted. Where none is (a plain ``Exception``),
    the substring cues stay live — see ``TestPredicates`` for those.
    """

    # The production 403 body as litellm renders it inside a typed exception: the same
    # bytes, minus the prefix litellm adds when stringifying. Derived rather than
    # re-pasted so exploring a new case in one place can't leave the other asserting the
    # old shape (neither ``_is_status`` nor ``_llm_status_code`` ever sees the prefix).
    _DRAINED = PRODUCTION_KEY_LIMIT_403.removeprefix("litellm.APIError: ")

    def test_spend_cap_403_still_falls_back(self) -> None:
        """The load-bearing ordering: a DRAINED-key 403 is credit-caused, so it falls back.

        The credit check runs upstream of the 403 veto; making the veto status-based must
        not let it overtake that.
        """
        assert should_retry_with_general_key(_api_error(403, self._DRAINED)) is True

    def test_moderation_403_does_not_fall_back(self) -> None:
        """A genuine moderation 403 still refuses to swap keys — both keys would refuse."""
        body = 'APIError: OpenrouterException - {"error":{"message":"Blocked by moderation","code":403}}'
        assert should_retry_with_general_key(_api_error(403, body)) is False

    def test_prompt_echo_402_does_not_read_as_credit(self) -> None:
        """A moderation 403 replaying a prompt containing "402" is not an empty wallet.

        Deliberately omits every moderation WORD cue, so the text-only veto cannot save
        this case — the reported 403 is the only thing that distinguishes it. Left
        unguarded it bills the personal key for a call that will refuse again AND, being
        classified credit-caused, exempts a real content block from CI alerting.
        """
        body = (
            "APIError: OpenrouterException - "
            '{"error":{"message":"Input blocked by provider policy",'
            '"metadata":{"reasons":["policy"],"flagged input":"...HR 402 authorizes $402 million..."},'
            '"code":403}}'
        )
        exc = _api_error(403, body)
        assert should_retry_with_general_key(exc) is False

    def test_key_hash_digits_do_not_trigger_fallback(self) -> None:
        """A 403 whose key hash happens to contain "429" must not read as a rate limit.

        A non-negligible fraction of key rotations produce a hash like this (see
        ``test_key_hash_status_collision_is_small_but_nonnegligible``). The current donated
        hash contains none of the six statuses, which is luck, not design.
        """
        body = (
            'APIError: OpenrouterException - {"error":{"message":"Blocked by content policy. '
            'Key: https://openrouter.ai/keys/a429b502c401d9f3","code":403}}'
        )
        assert should_retry_with_general_key(_api_error(403, body)) is False

    def test_status_detected_when_message_has_no_digits(self) -> None:
        """Detection no longer depends on the provider spelling the status in prose."""
        assert should_retry_with_general_key(_api_error(429, "upstream is busy right now")) is True
        assert should_retry_with_general_key(_api_error(401, "bad credentials")) is True
        assert should_retry_with_general_key(_api_error(402, "wallet empty")) is True

    def test_infrastructure_statuses_still_refuse_fallback(self) -> None:
        """5xx are infrastructure, not key-scoped — swapping keys cannot help."""
        assert should_retry_with_general_key(_api_error(502, "upstream died")) is False
        assert should_retry_with_general_key(_api_error(503, "upstream unavailable")) is False

    def test_route_scoped_404_text_cues_survive(self) -> None:
        """The 404 family is classified by TEXT, and must stay that way.

        "no allowed providers" and the guardrail / data-policy block are both 404s that
        DO warrant a key swap, while a plain 404 (missing model) does not — so the status
        alone cannot decide, and these cues must not be collapsed into a numeric rule.
        """
        allowed = "404 No allowed providers are available for the selected model."
        guardrail = "404 No endpoints available matching your guardrail restrictions and data policy."
        assert should_retry_with_general_key(_api_error(404, allowed)) is True
        assert should_retry_with_general_key(_api_error(404, guardrail)) is True
        assert should_retry_with_general_key(_api_error(404, "model not found")) is False

    def test_non_llm_exception_keeps_textual_classification(self) -> None:
        """A statusless exception is classified exactly as before (regression guard).

        ``requests``-style errors keep their status on ``.response``, and the suite
        asserts this predicate against plain ``Exception("401 Unauthorized")`` strings.
        Neither reports a ``status_code``, so both must keep reaching the text cues.
        """
        assert should_retry_with_general_key(Exception("401 Unauthorized")) is True
        assert should_retry_with_general_key(Exception("HTTP 402 Payment Required")) is True
        assert should_retry_with_general_key(Exception("429 Too Many Requests")) is True
        assert should_retry_with_general_key(Exception("403 Forbidden moderation")) is False
        assert should_retry_with_general_key(Exception("502 Bad Gateway")) is False


class TestCreditClassificationPrecedence:
    """Explicit credit wording > moderation wording > the reported status.

    The three signals disagree in real bodies, so their precedence is the whole design:

    * Explicit phrases outrank everything. A drained-key 403 must fall back, and its body
      can legitimately carry HTTP boilerplate like "Forbidden" — letting a generic
      moderation WORD veto "Key limit exceeded" would resurrect the exact bug this whole
      change exists to fix.
    * Moderation wording outranks the status. A refusal body replays up to ~100 chars of
      our own prompt, so it can contain any digits at all; where wording and status
      disagree, take the conservative branch and do not bill the paid key.
    * The status decides only what is left, replacing the bare-digit cue that
      ``_is_credit_message`` has to fall back on when no status was reported.
    """

    def test_moderation_403_echoing_prompt_402_is_not_credit(self) -> None:
        """403 whose ``flagged_input`` replays a prompt containing "402" → not credit.

        The forecasting prompt is full of dollar figures and bill numbers, so this is the
        realistic shape of the poisoning case.
        """
        body = (
            "APIError: OpenrouterException - "
            '{"error":{"message":"Blocked","metadata":{"reasons":["policy"],'
            '"flagged_input":"...HR 402 authorizes $402 million in FY2026..."},"code":403}}'
        )
        exc = _api_error(403, body)
        assert is_credit_caused_error(exc) is False
        assert should_retry_with_general_key(exc) is False

    def test_reported_402_outranks_moderation_wording(self) -> None:
        """A reported 402 is credit-caused even when the body carries a moderation word.

        402 IS "Payment Required" — the status has no other meaning, and OpenRouter
        reports refusals as 403. So a reported 402 is stronger evidence about money than
        an incidental English word in the body, and the failure asymmetry agrees: reading
        a real 402 as a refusal strands the ensemble on a dry key (the production bug),
        while reading a hypothetical 402-shaped refusal as credit costs one paid call
        that refuses again.

        The moderation veto still governs 403, which is the genuinely ambiguous status.
        """
        balance = "APIError: OpenrouterException - Forbidden: account balance too low"
        exc = _api_error(402, balance)
        assert is_credit_caused_error(exc) is True
        assert should_retry_with_general_key(exc) is True

        flagged = 'APIError: OpenrouterException - {"error":{"message":"flagged for moderation","code":402}}'
        assert is_credit_caused_error(_api_error(402, flagged)) is True

    def test_explicit_credit_wording_outranks_http_forbidden_boilerplate(self) -> None:
        """A drained-key 403 rendered as "403 Forbidden" still falls back.

        Regression guard on the precedence order: "forbidden" is a moderation CUE, so if
        the veto were allowed to outrank the explicit phrase, the production dry-key
        failure would stop falling back and the ensemble would strand on the dead key.
        """
        body = "APIError: OpenrouterException - 403 Forbidden: Key limit exceeded (total limit)."
        exc = _api_error(403, body)
        assert is_credit_caused_error(exc) is True
        assert should_retry_with_general_key(exc) is True


class TestFallbackAccountingConcurrency:
    """``record_donated_key_fallback`` must not stall the loop, and must not lose counts.

    The spend-cap 403 path reaches ``is_suppressible_credit_error``, which probes
    ``/auth/key`` over blocking httpx. Called straight from a coroutine that stalls EVERY
    concurrently in-flight forecaster and research task for up to
    ``DONATED_KEY_PROBE_TIMEOUT_S``, not just the call that hit the 403, eating into
    per-question soft deadlines.

    The fix threads the PROBE only. The counting has to stay on the event loop: the three
    module counters are mutated with ``+=``, which compiles to LOAD_GLOBAL / INPLACE_ADD /
    STORE_GLOBAL and is interruptible between bytecodes. Moving the whole function to a
    worker (``asyncio.to_thread(record_donated_key_fallback, ...)``) would let N
    forecasters failing on one dry key — the exact 2026-07-26 shape — race the increment,
    undercount ``_generic_key_fallback_count``, and take a degraded run GREEN. That is the
    failure this whole change exists to prevent, so it gets its own test.
    """

    @pytest.fixture(autouse=True)
    def _clean_counters(self):
        reset_generic_key_fallback_count()
        reset_credit_key_fallback_count()
        reset_donated_404_fallback_count()
        # The probe caches per process, so without this the test asserts against the
        # cached path and passes vacuously — the stall is only reachable on the FIRST
        # spend-cap 403 of a run.
        reset_donated_key_state_cache()
        yield
        reset_generic_key_fallback_count()
        reset_credit_key_fallback_count()
        reset_donated_404_fallback_count()
        reset_donated_key_state_cache()

    @pytest.mark.asyncio
    async def test_blocking_probe_does_not_stall_the_event_loop(self) -> None:
        """A slow probe must not prevent other tasks from making progress."""
        import asyncio
        import time

        ticks: list[int] = []

        async def _ticker() -> None:
            for _ in range(40):
                await asyncio.sleep(0.005)
                ticks.append(1)

        def _slow_probe() -> DonatedKeyState:
            time.sleep(0.2)
            return DonatedKeyState.DRAINED

        with patch.object(fallback_openrouter, "classify_donated_key_state", _slow_probe):
            ticker = asyncio.create_task(_ticker())
            await fallback_openrouter.record_donated_key_fallback(
                "openrouter/openai/gpt-5.6-sol", Exception(PRODUCTION_KEY_LIMIT_403)
            )
            progressed_during_probe = len(ticks)
            ticker.cancel()

        # Blocking the loop for 0.2s would leave the ticker at zero; threaded, it gets
        # ~40 chances. A low bar keeps this robust on a loaded machine.
        assert progressed_during_probe >= 2, f"event loop appears blocked ({progressed_during_probe=})"
        assert get_credit_key_fallback_count() == 1

    @pytest.mark.asyncio
    async def test_concurrent_fallbacks_lose_no_counts(self) -> None:
        """N forecasters failing on one dry key must count exactly N.

        Guards the trap in the fix itself: the accounting has to run on the event loop.
        """
        import asyncio

        def _drained() -> DonatedKeyState:
            return DonatedKeyState.DRAINED

        with patch.object(fallback_openrouter, "classify_donated_key_state", _drained):
            await asyncio.gather(
                *(
                    fallback_openrouter.record_donated_key_fallback(f"model-{i}", Exception(PRODUCTION_KEY_LIMIT_403))
                    for i in range(50)
                )
            )

        assert get_generic_key_fallback_count() == 50
        assert get_credit_key_fallback_count() == 50


class TestPromptEchoCreditPhrases:
    """A refusal body replays our own prompt, so ordinary credit ENGLISH can echo too.

    Closing the "402" digit echo was only half the hole. litellm formats the message as
    ``APIError: {provider} - {raw body}`` and an OpenRouter moderation 403 body carries
    ``flagged_input``: up to ~100 characters of the prompt we sent. A forecasting prompt
    can say "insufficient funds" or "payment required" for entirely ordinary reasons, and
    those are matched anywhere in the message — so a content block was billing the paid
    key AND being exempted from alerting as an expected empty wallet.

    The split is by SPECIFICITY, not by category. "Key limit exceeded" is OpenRouter's own
    spend-cap wording and will not show up in a question about an election; the other four
    phrases are ordinary English. So the spend-cap cue outranks the moderation veto and the
    generic phrases sit under it.
    """

    @staticmethod
    def _moderation_body(flagged_input: str) -> str:
        return (
            "litellm.APIError: APIError: OpenrouterException - "
            '{"error":{"message":"Your input was flagged for the following reasons: violence",'
            '"code":403,"metadata":{"reasons":["violence"],'
            f'"flagged_input":"{flagged_input}"}}}}'
        )

    @pytest.mark.parametrize(
        "flagged_input",
        [
            "Will Bank X be declared insolvent for insufficient funds after the bombing?",
            "Will the ransom demand state payment required by Friday?",
            "Will the treasury report insufficient credit in the reserve facility?",
            "Will the vault be out of credits before the siege ends?",
        ],
    )
    def test_moderation_403_echoing_credit_english_is_not_credit(self, flagged_input: str) -> None:
        """The reported 403 must beat an ordinary credit phrase replayed from our prompt."""
        exc = _api_error(403, self._moderation_body(flagged_input))
        assert should_retry_with_general_key(exc) is False
        assert is_credit_caused_error(exc) is False

    @pytest.mark.parametrize(
        "flagged_input",
        [
            "Will the Fed raise the rate limit on reserve accounts before Aug 18?",
            "Will the unauthorized withdrawal be reported before the audit?",
        ],
    )
    def test_moderation_403_echoing_key_scoped_english_does_not_fall_back(self, flagged_input: str) -> None:
        """Rate-limit and credential wording in a replayed prompt must not force a key swap.

        These stayed non-credit even before the fix, but they still flipped ROUTING to
        True — billing the personal key for a call the paid key will also refuse.
        """
        exc = _api_error(403, self._moderation_body(flagged_input))
        assert should_retry_with_general_key(exc) is False

    def test_statusless_moderation_echo_is_not_credit_or_suppressible(self) -> None:
        """The statusless spelling is the worst case, so it gets its own guard.

        With no ``status_code`` there is no status to fall back on, and
        ``is_suppressible_credit_error`` short-circuits to True before ever consulting the
        ``/auth/key`` probe — so a content block was exempted from alerting without the
        drained-vs-revoked discriminator ever running.
        """
        body = self._moderation_body("Will Bank X be declared insolvent for insufficient funds?")
        exc = Exception(body)
        assert should_retry_with_general_key(exc) is False
        assert is_credit_caused_error(exc) is False
        assert is_suppressible_credit_error(exc) is False

    def test_statusless_spend_cap_beats_forbidden_boilerplate(self) -> None:
        """REGRESSION GUARD: the spend-cap cue outranks the veto, statusless included.

        "forbidden" is a moderation cue AND generic HTTP boilerplate. Gating the spend-cap
        phrase behind the veto — the obvious way to fix the echo hole — would make a
        drained key rendered as "403 Forbidden: Key limit exceeded" stop falling back and
        strand the ensemble on the dead key. That is the production bug, re-created by the
        fix for a smaller one, and the statusless spelling is how the run log rendered it.
        """
        for body in (
            "litellm.APIError: 403 Forbidden: Key limit exceeded (total limit).",
            PRODUCTION_KEY_LIMIT_403,
        ):
            exc = Exception(body)
            assert should_retry_with_general_key(exc) is True, body
            assert is_credit_caused_error(exc) is True, body

    def test_route_scoped_block_still_falls_back_on_a_403(self) -> None:
        """A guardrail / allowed-providers block keeps its escape hatch on any status.

        Deciding 403 early must not swallow these: they are the one class of non-credit
        403 the personal key genuinely CAN route, and the donated-key data-policy block is
        why OpenAI native search works at all.
        """
        guardrail = (
            "No endpoints available matching your guardrail restrictions and data policy. "
            "Configure: https://openrouter.ai/settings/privacy"
        )
        assert should_retry_with_general_key(_api_error(403, guardrail)) is True
        assert should_retry_with_general_key(_api_error(403, "No allowed providers are available")) is True


class TestStatuslessPromptEcho:
    """The other prompt-echo carrier: an exception that reports no HTTP status at all.

    Closing the echo hole for bodies that REPORT a status left the statusless door open,
    and the realistic carrier is not a moderation refusal at all — it is
    forecasting-tools' empty-completion guard, which raises
    ``RuntimeError(f"LLM answer is an empty string. The model was {model} and the prompt
    was: {prompt}")`` with up to 2000 characters of the prompt verbatim
    (``general_llm.py``). A forecasting prompt says "402" about 13% of the time (measured
    over the 963-bundle research archive) and "insufficient funds" whenever the question
    is about a bank.

    With nothing after the echo marked as ours, a benign zero-output blip billed the
    personal key, was classified credit-caused, and — because the 402 family skips the
    ``/auth/key`` probe — was SUBTRACTED from cli's ``alertable``, taking a degraded run
    green without the drained-vs-revoked discriminator ever running.
    """

    @staticmethod
    def _zero_output(prompt: str) -> RuntimeError:
        """forecasting-tools' empty-completion RuntimeError, verbatim in shape."""
        return RuntimeError(
            f"LLM answer is an empty string. The model was openrouter/openai/gpt-5.6-terra and the prompt was: {prompt}"
        )

    @pytest.fixture(autouse=True)
    def _probe_would_be_observable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make any ``/auth/key`` probe fail loudly, so a case that reaches it can't pass.

        A configured donated key is load-bearing: without one the probe short-circuits to
        UNKNOWN before touching ``fetch_auth_key``, and every assertion below would hold
        for the wrong reason.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "sk-fake-donated")
        reset_donated_key_state_cache()
        monkeypatch.setattr(
            fallback_openrouter,
            "classify_donated_key_state",
            MagicMock(side_effect=AssertionError("the probe must not be reached for a prompt echo")),
        )

    @pytest.mark.parametrize(
        "prompt",
        [
            "Will revenue top $402M before Aug 18 2026?",
            "Will Bank X be declared insolvent for insufficient funds?",
            "Will the ransom demand state payment required by Friday?",
            "Will the vault be out of credits before the siege ends?",
        ],
    )
    def test_zero_output_echoing_credit_wording_is_not_credit(self, prompt: str) -> None:
        exc = self._zero_output(prompt)
        assert should_retry_with_general_key(exc) is False
        assert is_credit_caused_error(exc) is False
        assert is_suppressible_credit_error(exc) is False

    @pytest.mark.parametrize(
        "prompt",
        [
            "Will the EU adopt a stricter data policy in 2026?",
            "Will OpenAI ship stricter guardrails before Aug 18 2026?",
            "Will the unauthorized withdrawal be reported before the audit?",
        ],
    )
    def test_zero_output_echoing_route_or_credential_wording_does_not_fall_back(self, prompt: str) -> None:
        """These never reached the credit counter, but they did force a wasted paid call.

        A zero-output blip is not key-scoped — the personal key would return the same
        empty completion — so swapping keys buys nothing and bills the operator.
        """
        assert should_retry_with_general_key(self._zero_output(prompt)) is False

    def test_typed_403_echoing_guardrail_wording_does_not_fall_back(self) -> None:
        """``_ROUTE_SCOPED_TEXT_CUES`` had no moderation veto and matched at every status.

        So a moderation 403 whose ``flagged_input`` replays a question ABOUT guardrails
        read as the donated key's data-policy block and swapped to the paid key for a call
        it will refuse just the same.
        """
        body = (
            "litellm.APIError: APIError: OpenrouterException - "
            '{"error":{"message":"Your input was flagged for the following reasons: violence",'
            '"code":403,"metadata":{"reasons":["violence"],'
            '"flagged_input":"Will OpenAI ship stricter guardrails before Aug 18 2026?"}}}'
        )
        exc = _api_error(403, body)
        assert should_retry_with_general_key(exc) is False

    def test_zero_output_echoing_deprecated_records_no_deprecation_alert(self) -> None:
        """The echo also reached the deprecation tripwire, whose only consequence is exit 1.

        A question about a deprecated standard turned a benign empty completion into
        ``MODEL DEPRECATION DETECTED ... model=<unknown>`` and a red run.
        """
        fallback_openrouter.clear_deprecation_alerts()
        try:
            should_retry_with_general_key(self._zero_output("Will the deprecated standard be retired in 2026?"))
            assert fallback_openrouter._DEPRECATION_ALERTS == []
        finally:
            fallback_openrouter.clear_deprecation_alerts()

    def test_real_deprecation_still_records(self) -> None:
        """Positive control: provider wording sits BEFORE any echo marker, so it survives."""
        fallback_openrouter.clear_deprecation_alerts()
        try:
            msg = (
                "Grok 4.1 Fast is deprecated. xAI recommends switching to Grok 4.3. "
                "The model was x-ai/grok-4.1-fast and the prompt was: Will inflation fall?"
            )
            assert fallback_openrouter._record_deprecation_if_matched("x-ai/grok-4.1-fast", msg) is True
            assert len(fallback_openrouter._DEPRECATION_ALERTS) == 1
        finally:
            fallback_openrouter.clear_deprecation_alerts()

    def test_real_spend_cap_403_still_classifies_as_credit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Positive control, and the one that matters: truncation must not disarm the fix.

        OpenRouter's own spend-cap wording precedes any echo marker, so the drained-key
        403 keeps falling back and keeps being credit-classified.
        """
        monkeypatch.setattr(
            fallback_openrouter, "classify_donated_key_state", MagicMock(return_value=DonatedKeyState.DRAINED)
        )
        exc = Exception(PRODUCTION_KEY_LIMIT_403)
        assert should_retry_with_general_key(exc) is True
        assert is_credit_caused_error(exc) is True
        assert is_suppressible_credit_error(exc) is True


class TestProbeCannotOutlastItsBudget:
    """The probe's promised 5s bound has to be a bound on TOTAL time, not per-operation.

    ``DONATED_KEY_PROBE_TIMEOUT_S`` is forwarded to httpx as a bare float, which sets each
    operation's timeout independently and does not cap elapsed time — a server trickling
    bytes slower than the read timeout resets the clock on every chunk (measured: a
    ``timeout=1.0`` GET against a 0.5s-per-byte trickler took 10.2s). This lands on the
    RECOVERY path: ``record_donated_key_fallback`` is awaited before
    ``_invoke_once_using_secondary``, so probe latency delays the personal-key call even
    though routing was already decided textually. A degraded-but-not-dead OpenRouter
    control plane is exactly what co-occurs with a spend-cap 403, and the tight-budget
    callers are the ones this reaches: prediction-market keyword extraction has a 15s wall
    cap, the financial classifier 30s.
    """

    @pytest.fixture(autouse=True)
    def _clean_state(self, monkeypatch: pytest.MonkeyPatch):
        reset_donated_key_state_cache()
        reset_generic_key_fallback_count()
        reset_credit_key_fallback_count()
        monkeypatch.setattr(fallback_openrouter, "DONATED_KEY_PROBE_TIMEOUT_S", 0.2)
        yield
        reset_generic_key_fallback_count()
        reset_credit_key_fallback_count()
        reset_donated_key_state_cache()

    @pytest.mark.asyncio
    async def test_slow_probe_is_abandoned_at_the_cap_and_stays_alertable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A probe that outlasts the cap must not hold the fallback, and must not suppress.

        Timing out is an inconclusive answer, so it degrades exactly like UNKNOWN: counted,
        not subtracted from ``alertable``.
        """

        def never_answers() -> DonatedKeyState:
            time.sleep(5.0)  # far past the patched cap; the wait_for must not wait for it
            return DonatedKeyState.DRAINED

        monkeypatch.setattr(fallback_openrouter, "classify_donated_key_state", never_answers)

        started = time.monotonic()
        await fallback_openrouter.record_donated_key_fallback(
            "openrouter/openai/gpt-5.6-sol", Exception(PRODUCTION_KEY_LIMIT_403)
        )
        elapsed = time.monotonic() - started

        assert elapsed < 2.0, f"probe held the fallback for {elapsed:.1f}s despite a 0.2s cap"
        assert fallback_openrouter.get_generic_key_fallback_count() == 1
        assert fallback_openrouter.get_credit_key_fallback_count() == 0

    @pytest.mark.asyncio
    async def test_probe_answering_inside_the_cap_still_suppresses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Positive control: the cap must not disarm suppression for a probe that answers."""
        monkeypatch.setattr(
            fallback_openrouter, "classify_donated_key_state", MagicMock(return_value=DonatedKeyState.DRAINED)
        )
        await fallback_openrouter.record_donated_key_fallback(
            "openrouter/openai/gpt-5.6-sol", Exception(PRODUCTION_KEY_LIMIT_403)
        )
        assert fallback_openrouter.get_generic_key_fallback_count() == 1
        assert fallback_openrouter.get_credit_key_fallback_count() == 1


class TestProbeFailureCannotVetoRouting:
    """An unanswerable ``/auth/key`` probe must never abort the fallback it is annotating.

    The probe promises "never raises", and now catches broadly enough to keep that promise.
    ``record_donated_key_fallback`` guards its call anyway, because the promise is not
    something routing should have to trust: the probe runs BEFORE
    ``_invoke_once_using_secondary``, so an escape stranded the ensemble on the dry donated
    key with the funded personal key untried — the 2026-07-26 incident reached through the
    exception path rather than a stale balance read. The routing decision is already final
    and purely textual by the time this runs, so alerting bookkeeping must not gate it.

    Both cases patch ``classify_donated_key_state`` rather than ``fetch_auth_key``. That is
    the seam the guard defends, and it is env-independent: the probe short-circuits to
    UNKNOWN before touching ``fetch_auth_key`` when no donated key is configured, and CI
    injects no OpenRouter secrets, so a ``fetch_auth_key`` patch would make these assertions
    hold for the wrong reason there while passing locally off the repo's ``.env``.
    """

    _BOOMS: ClassVar[list[Exception]] = [RuntimeError("transport wedged"), FileNotFoundError("bad SSL_CERT_FILE")]

    @pytest.fixture(autouse=True)
    def _clean_counters(self, monkeypatch: pytest.MonkeyPatch):
        reset_donated_key_state_cache()
        reset_generic_key_fallback_count()
        reset_credit_key_fallback_count()
        yield
        reset_generic_key_fallback_count()
        reset_credit_key_fallback_count()
        reset_donated_key_state_cache()

    @pytest.mark.parametrize("boom", _BOOMS)
    @pytest.mark.asyncio
    async def test_probe_exception_leaves_the_event_counted_and_alertable(
        self, monkeypatch: pytest.MonkeyPatch, boom: BaseException
    ) -> None:
        monkeypatch.setattr(fallback_openrouter, "classify_donated_key_state", MagicMock(side_effect=boom))
        await fallback_openrouter.record_donated_key_fallback(
            "openrouter/openai/gpt-5.6-sol", Exception(PRODUCTION_KEY_LIMIT_403)
        )

        # Counted, so the personal-key spend is visible...
        assert fallback_openrouter.get_generic_key_fallback_count() == 1
        # ...and NOT subtracted from alertable, because the probe never answered.
        assert fallback_openrouter.get_credit_key_fallback_count() == 0

    @pytest.mark.parametrize("boom", _BOOMS)
    @pytest.mark.asyncio
    async def test_personal_key_call_still_happens_when_the_probe_raises(
        self, monkeypatch: pytest.MonkeyPatch, boom: BaseException
    ) -> None:
        """The headline behavior, end to end: the funded key is still tried.

        The counter assertions above call ``record_donated_key_fallback`` directly, so they
        cannot see whether the wrapper went on to invoke the secondary — which is the whole
        point of the guard, and the exact thing the incident lost.
        """
        monkeypatch.setattr(fallback_openrouter, "classify_donated_key_state", MagicMock(side_effect=boom))
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.6-sol",
            primary_api_key="donated",
            secondary_api_key="personal",
            temperature=0,
        )
        monkeypatch.setattr(
            llm, "_invoke_once_using_primary", AsyncMock(side_effect=Exception(PRODUCTION_KEY_LIMIT_403))
        )
        secondary = AsyncMock(return_value="fallback_ok")
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", secondary)

        assert await llm.invoke("hi") == "fallback_ok"
        secondary.assert_awaited_once()
        assert fallback_openrouter.get_generic_key_fallback_count() == 1
        assert fallback_openrouter.get_credit_key_fallback_count() == 0


class TestStatuslessDigitEchoIsNotAStatus:
    """A statusless failure must not read OUR OWN prompt's digits as an HTTP status.

    ``4a1dd1f`` closed this for the credit cues but left ``_is_status``'s digit fallback
    reading the untruncated message, so the same hole stayed open at every other status.
    forecasting-tools' empty-completion guard raises
    ``RuntimeError("LLM answer is an empty string. ... and the prompt was: <up to 2000 chars>")``,
    and a question about a bill numbered 429 or 401 then routed to the operator's paid key
    for a call that returns empty again. Frequency in the 989-bundle research archive:
    "429" in 10.2% of prompts, "401" in 13.8% — the same order as the 13.0% for "402" that
    justified the original truncation.
    """

    _ZERO_OUTPUT = "LLM answer is an empty string. The model was openrouter/openai/gpt-5.6-sol and the prompt was: {q}"

    @pytest.mark.parametrize(
        "question",
        [
            "Will S.429 (the Fentanyl Act) become law before 2027?",
            "Will H.R.401 pass the Senate before 2027?",
            "Will the 403 Forbidden error rate exceed 1% next quarter?",
            "Will the 502 bus route be extended before 2027?",
        ],
    )
    def test_prompt_echoed_status_digits_do_not_trigger_a_key_swap(self, question: str) -> None:
        exc = RuntimeError(self._ZERO_OUTPUT.format(q=question))
        assert should_retry_with_general_key(exc) is False, question
        assert is_credit_caused_error(exc) is False, question

    @pytest.mark.parametrize(
        ("message", "falls_back"),
        [
            ("401 unauthorized", True),
            ("429 too many requests", True),
            ("403 forbidden", False),
            ("502 bad gateway", False),
        ],
    )
    def test_plain_status_strings_survive_echo_stripping(self, message: str, falls_back: bool) -> None:
        """The carve-out this fix removed was unnecessary, and this pins why.

        ``_without_prompt_echo`` cuts only at an echo marker. These strings carry none, so
        they pass through byte-identical and the statusless callers that predate the
        prompt-echo work classify exactly as they always have — including the two that
        correctly do NOT swap keys: a bare 403 is a moderation refusal both keys would
        repeat, and a 502 is an upstream outage that is not key-scoped.
        """
        assert fallback_openrouter._without_prompt_echo(message) == message
        assert should_retry_with_general_key(Exception(message)) is falls_back, message
