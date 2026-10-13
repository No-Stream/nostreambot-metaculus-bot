import asyncio
import logging
import os
import sys
from typing import Any

import litellm.exceptions
from forecasting_tools import GeneralLlm

from metaculus_bot.constants import (
    CREDIT_ALERT_RESUME_DATE,
    OAI_ANTH_OPENROUTER_KEY_ENV,
    OPENROUTER_API_KEY_ENV,
    credit_alerts_active,
    donated_openrouter_key_enabled,
    gemini_use_donated_openrouter_key,
)
from metaculus_bot.credit_telemetry import (
    DONATED_KEY_ALIAS,
    DONATED_KEY_PROBE_TIMEOUT_S,
    PERSONAL_KEY_ALIAS,
    DonatedKeyState,
    classify_donated_key_state,
    llm_call_metadata,
    plain_llm_key_alias,
)
from metaculus_bot.llm_retry import llm_status_code

logger: logging.Logger = logging.getLogger(__name__)


# Where provider error text ends and our echoed prompt begins. See docs/operations.md "Prompt-echo truncation".
_PROMPT_ECHO_MARKERS: tuple[str, ...] = ("and the prompt was:", "flagged_input", "flagged input")


def _without_prompt_echo(lowercased_msg: str) -> str:
    """Drop the tail that follows the first prompt-echo marker.

    Everything past a marker is text WE sent, so it must not classify anything; the marker itself
    is kept, because ``flagged_input`` is also one of ``_MODERATION_CUES``. The two echo carriers,
    the measured prompt-digit rates, the incident behind the truncation and the status-code
    carve-out: docs/operations.md "Prompt-echo truncation".
    """
    cut = len(lowercased_msg)
    for marker in _PROMPT_ECHO_MARKERS:
        found_at = lowercased_msg.find(marker)
        if found_at != -1:
            cut = min(cut, found_at + len(marker))
    return lowercased_msg[:cut]


def _record_deprecation_if_matched(model: str, error_msg: str) -> bool:
    """Append to ``_DEPRECATION_ALERTS`` iff the error message looks like a model deprecation.

    Returns True iff matched (and recorded). Match is case-insensitive substring
    against ``_DEPRECATION_PATTERNS``. Designed to be called from any LLM call
    site that observes an exception — the ``FallbackOpenRouterLlm`` wrapper for
    donated-key models, ``_run_providers_parallel`` for plain-GeneralLlm research
    providers (Grok native search), etc. Idempotent within a single recording —
    every distinct error string adds an entry; cli.py only checks ``len > 0``.
    """
    msg_lower = _without_prompt_echo(error_msg.lower())
    if any(pattern in msg_lower for pattern in _DEPRECATION_PATTERNS):
        _DEPRECATION_ALERTS.append((model, error_msg))
        return True
    return False


def clear_deprecation_alerts() -> None:
    """Reset the alert list. Used by tests; not for production code."""
    _DEPRECATION_ALERTS.clear()


def has_deprecation_alerts() -> bool:
    """Whether any deprecation was recorded this run.

    Read by ``cli.py``'s end-of-run summary so a run that
    ``check_deprecation_alerts_and_exit`` will shortly turn red cannot first
    label itself "clean" (the list is module-private, so callers cannot inspect
    it directly).
    """
    return bool(_DEPRECATION_ALERTS)


def check_deprecation_alerts_and_exit() -> None:
    """Post-submission tripwire: log loudly and ``sys.exit(1)`` if any deprecation was seen.

    Called from ``cli.py`` AFTER ``forecast_on_tournament`` / ``forecast_questions``
    completes — so every publishable question is already on Metaculus regardless
    of exit status. Returns silently when the alert list is empty.
    """
    if not _DEPRECATION_ALERTS:
        return
    banner = "=" * 78
    logger.error(banner)
    logger.error("MODEL DEPRECATION DETECTED — %d alert(s) recorded this run", len(_DEPRECATION_ALERTS))
    logger.error("OpenRouter (or another provider) returned a deprecation-shaped error for one or more")
    logger.error("models the bot called. Submission completed via fallbacks, but the model lineup needs")
    logger.error("updating. See metaculus_bot/llm_configs.py and metaculus_bot/constants.py.")
    logger.error(banner)
    for model_slug, error_msg in _DEPRECATION_ALERTS:
        logger.error("  model=%s | error=%s", model_slug, error_msg)
    logger.error(banner)
    sys.exit(1)


# Post-submission tripwire: red CI, never a mid-run abort. See docs/operations.md "The model-deprecation tripwire".
_DEPRECATION_ALERTS: list[tuple[str, str]] = []

# Conservative on purpose: a false positive reddens CI. See docs/operations.md "The model-deprecation tripwire".
_DEPRECATION_PATTERNS: tuple[str, ...] = (
    "deprecated",
    "recommends switching to",
)


# Diagnostic subset of the fallback total, never added to alertable. See docs/operations.md "The fallback counters".
_donated_404_fallback_count: int = 0


def get_donated_404_fallback_count() -> int:
    """Read the module-level counter for donated-key 404 fallback events."""
    return _donated_404_fallback_count


def reset_donated_404_fallback_count() -> None:
    """Reset the counter to zero. Used by tests; not for production code."""
    global _donated_404_fallback_count  # noqa: PLW0603  # module-global run counter is the design (AGENTS.md)
    _donated_404_fallback_count = 0


# Every donated-to-personal fallback, all causes: the alerting input. See docs/operations.md "The fallback counters".
_generic_key_fallback_count: int = 0


def get_generic_key_fallback_count() -> int:
    """Read the module-level counter for donated->personal key fallback events (all causes)."""
    return _generic_key_fallback_count


def reset_generic_key_fallback_count() -> None:
    """Reset the counter to zero. Used by tests; not for production code."""
    global _generic_key_fallback_count  # noqa: PLW0603  # module-global run counter is the design (AGENTS.md)
    _generic_key_fallback_count = 0


# Credit-caused subset, subtracted only while alerting is suppressed. See docs/operations.md "The fallback counters".
_credit_key_fallback_count: int = 0


def get_credit_key_fallback_count() -> int:
    """Read the module-level counter for credit-caused donated->personal fallbacks."""
    return _credit_key_fallback_count


def reset_credit_key_fallback_count() -> None:
    """Reset the counter to zero. Used by tests; not for production code."""
    global _credit_key_fallback_count  # noqa: PLW0603  # module-global run counter is the design (AGENTS.md)
    _credit_key_fallback_count = 0


# Everything outside this set 404s on the donated key. See docs/operations.md "The donated-key provider set".
DONATED_KEY_PROVIDERS: frozenset[str] = frozenset({"openai", "anthropic", "google"})

# TODO(gemini-3.1-pro-donated): pinned to the personal key; see docs/operations.md "The Google blocklist".
DONATED_KEY_BLOCKED_GOOGLE_MODELS: frozenset[str] = frozenset({"gemini-3.1-pro", "gemini-4-argon"})


def should_route_via_donated_key(model: str) -> bool:
    """Whether ``model`` should prefer the Metaculus-donated key (with paid-key fallback).

    The ``DONATED_OPENROUTER_KEY_ENABLED`` master switch is read first, so a Mantic run routes
    every slug to the operator's personal key; past it, an ``openrouter/<provider>/<model>`` slug
    matches ``DONATED_KEY_PROVIDERS``, Google additionally needs
    ``GEMINI_USE_DONATED_OPENROUTER_KEY``, and ``DONATED_KEY_BLOCKED_GOOGLE_MODELS`` is pinned to
    the personal key. Why each rule exists, and why the switch is an environment variable rather
    than a runtime toggle: docs/operations.md "should_route_via_donated_key".
    """
    if not isinstance(model, str):  # a non-slug routes to the personal key rather than crashing the key decision
        return False
    if not donated_openrouter_key_enabled():
        return False
    if not model.startswith("openrouter/"):
        return False
    parts = model.split("/")
    if len(parts) < 2:
        return False
    provider = parts[1]
    if provider not in DONATED_KEY_PROVIDERS:
        return False
    if provider == "google":
        if not gemini_use_donated_openrouter_key():
            return False
        model_name = "/".join(parts[2:])
        if any(model_name.startswith(blocked) for blocked in DONATED_KEY_BLOCKED_GOOGLE_MODELS):
            return False
    return True


# A drained spend cap arrives as 403 with this phrase. See docs/operations.md "What a dry donated key actually returns".
KEY_LIMIT_EXCEEDED_CUE = "key limit exceeded"

# Ordinary English a prompt can carry, so these sit below the veto. See docs/operations.md "The credit arbiter".
_GENERIC_CREDIT_PHRASES: tuple[str, ...] = (
    "payment required",
    "insufficient credit",
    "out of credits",
    "insufficient funds",
)

# The body may replay our own prompt, so word cues only. See docs/operations.md "The credit arbiter".
_MODERATION_CUES: tuple[str, ...] = ("moderation", "forbidden", "flagged_input", "flagged for")


# English wording, so a statusless exception still classifies. See docs/operations.md "The text cue sets".
_RATE_LIMIT_TEXT_CUES: tuple[str, ...] = ("too many requests", "rate limit", "rate-limited upstream")
_BAD_CREDENTIAL_TEXT_CUES: tuple[str, ...] = ("unauthorized", "invalid api key", "disabled api key")

# Scoped to the key's routing, so the personal key can serve the call. See docs/operations.md "The text cue sets".
_ROUTE_SCOPED_TEXT_CUES: tuple[str, ...] = ("no allowed providers", "guardrail", "data policy")


def _is_status(reported_status: int | None, code: int, lowercased_msg: str) -> bool:
    """Whether the failure is HTTP ``code``, preferring the reported status over digits.

    A reported status is the only numeric evidence consulted, because an OpenRouter body carries a
    key hash and, on a moderation refusal, a replay of our own prompt. With no status reported the
    digit match falls back to the echo-stripped message. The collision arithmetic, the measured
    prompt-digit rates and the plain-``Exception`` pin: docs/operations.md "Status detection over
    message digits".
    """
    if reported_status is not None:
        return reported_status == code
    return str(code) in _without_prompt_echo(lowercased_msg)


def _is_credit_failure(reported_status: int | None, lowercased_msg: str) -> bool:
    """Whether this failure means "the key is out of money". The one credit arbiter.

    ``should_retry_with_general_key`` (routing) and ``is_credit_caused_error`` (alerting, through
    the credit-subset counter in ``record_donated_key_fallback``) both reach the answer here, so a
    cue edit cannot make them disagree: the spend-cap phrase wins outright, then a reported status
    decides alone, then moderation wording vetoes ordinary credit English. Why that order, and why
    nothing here reads a live balance: docs/operations.md "The credit arbiter".
    """
    provider_text = _without_prompt_echo(lowercased_msg)
    if KEY_LIMIT_EXCEEDED_CUE in provider_text:
        return True
    if reported_status is not None:
        return reported_status == 402
    if any(cue in provider_text for cue in _MODERATION_CUES):
        return False
    return "402" in provider_text or any(phrase in provider_text for phrase in _GENERIC_CREDIT_PHRASES)


def is_credit_caused_error(exc: Exception) -> bool:
    """Whether ``exc`` is a credit shortfall (402, spend-cap 403, insufficient credit).

    The public form of ``_is_credit_failure``, so the routing decision in
    ``should_retry_with_general_key`` and the alerting decision in
    ``is_suppressible_credit_error`` answer "was this about money?" the same way. They
    used to disagree: routing became status-aware while this stayed text-only, so a
    terse reported-402 (``APIError(status_code=402, message="wallet empty")``) fell
    back to the paid key without being credit-classified — reddening CI on exactly
    the expected empty wallet the suppression window exists for.
    """
    return _is_credit_failure(llm_status_code(exc), str(exc).lower())


def is_suppressible_credit_error(exc: Exception) -> bool:
    """Whether ``exc`` is the EXPECTED drained donated key, not some other breakage.

    Only this narrower class is exempt from CI alerting inside a suppression window
    (``constants.credit_alerts_active``). A revoked or re-capped key returns the same
    "Key limit exceeded" text, so the verdict comes from OpenRouter's free ``/auth/key`` probe
    rather than from the cue, and every inconclusive answer stays alertable. Why the 402 family
    skips the probe, and why routing never consults it: docs/operations.md "What a dry donated key
    actually returns".
    """
    if not is_credit_caused_error(exc):
        return False
    if KEY_LIMIT_EXCEEDED_CUE not in _without_prompt_echo(str(exc).lower()):
        return True
    return classify_donated_key_state() is DonatedKeyState.DRAINED


def should_retry_with_general_key(exc: Exception) -> bool:
    """
    Decide whether a failure likely indicates a key-scoped issue where falling back is appropriate.

    It falls back on 429, 401, 402, the spend-cap 403 and a route-scoped block the personal key can
    serve; it keeps the key on a moderation or permission 403, on a 502 or 503 outage and on a
    plain missing-model 404. Numeric detection reads the status the provider reported, never digits
    in the message. The full trigger list, the four orderings inside the function that are
    load-bearing, and what is out of scope: docs/operations.md "The fallback decision".
    """
    msg_raw = str(exc)
    # Records only; the slug-bearing call is in invoke(). See docs/operations.md "The model-deprecation tripwire".
    _record_deprecation_if_matched("<unknown>", msg_raw)

    # BYOK quotas are per-key, and litellm exhausted its own retries before raising.
    if isinstance(exc, litellm.exceptions.RateLimitError):
        return True

    msg = msg_raw.lower()
    # Authoritative for every numeric branch below; None means classify on text.
    status = llm_status_code(exc)
    # No word cue may read our own echoed prompt; the digit fallbacks still see the whole message.
    provider_text = _without_prompt_echo(msg)

    # The body is the least trustworthy input here, so only two shapes earn a key swap.
    if status == 403:
        return KEY_LIMIT_EXCEEDED_CUE in provider_text or any(cue in provider_text for cue in _ROUTE_SCOPED_TEXT_CUES)

    # Belt-and-suspenders for 429s litellm did not raise as the typed exception.
    if _is_status(status, 429, msg) or any(cue in provider_text for cue in _RATE_LIMIT_TEXT_CUES):
        return True

    # Positive signals: credentials/credits
    if _is_status(status, 401, msg) or any(cue in provider_text for cue in _BAD_CREDENTIAL_TEXT_CUES):
        return True
    if _is_credit_failure(status, msg):
        return True
    # Last positive signal; everything it does not match keeps the key, because a swap cannot help.
    return any(cue in provider_text for cue in _ROUTE_SCOPED_TEXT_CUES)


def _is_donated_404(exc: Exception) -> bool:
    """Whether this exception is the donated-key allowed-providers 404 specifically.

    Used to bump the alerting counter only on this fallback class — not for
    401/402 (those are credit/key issues, not the allowed-providers quirk).
    """
    return "no allowed providers" in str(exc).lower()


def _fallback_alert_note(*, suppressible: bool) -> str:
    """The "what happens to the exit code" clause of the paid-fallback WARNING.

    A suppressible credit fallback during the window does NOT redden CI (see
    ``constants.credit_alerts_active``), so saying it will would mislead whoever
    greps this line. Every other cause still exits non-zero.

    ``suppressible`` is the caller's already-computed
    ``is_suppressible_credit_error`` verdict — it has paid for the donated-key
    probe, and re-deriving it here would either duplicate that HTTP call or key the
    note on the text cue alone and promise a green run a revoked key won't deliver.
    """
    if suppressible and not credit_alerts_active():
        return (
            "Cause is a credit shortfall, so it is NOT counted as alertable until "
            f"{CREDIT_ALERT_RESUME_DATE.isoformat()} (operator is self-funding the season)."
        )
    return "Run will complete, then exit non-zero to alert."


async def record_donated_key_fallback(model: str, exc: Exception) -> None:
    """Count and log ONE donated -> personal-key fallback that is about to happen.

    The shared accounting seam for every donated-first call path, so a paid personal-key call can
    never happen unrecorded: one generic count, at most one subset count, and a loud warning. Call
    it only when the fallback will actually be attempted, because a rejected fallback bills
    nothing. Why it is async, why the accounting block below contains no await, and the path that
    used to bypass it: docs/operations.md "The shared accounting seam".
    """
    # Threaded and bounded: blocking httpx here would stall every in-flight task.
    try:
        suppressible = await asyncio.wait_for(
            asyncio.to_thread(is_suppressible_credit_error, exc), timeout=DONATED_KEY_PROBE_TIMEOUT_S
        )
    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except: alerting bookkeeping must not abort the fallback
        logger.exception(
            "DONATED_KEY_PROBE_FAILED: model=%s — the /auth/key probe raised or outlasted its %.1fs "
            "budget, so this fallback stays ALERTABLE. The personal-key call proceeds regardless; "
            "alerting bookkeeping must not gate recovery.",
            model,
            DONATED_KEY_PROBE_TIMEOUT_S,
        )
        suppressible = False

    # No await below: the increments are only bytecode-atomic, so a checkpoint would lose events.
    global _generic_key_fallback_count  # noqa: PLW0603  # no await below; increments stay bytecode-atomic
    _generic_key_fallback_count += 1
    if suppressible:
        global _credit_key_fallback_count  # noqa: PLW0603  # no await below; increments stay bytecode-atomic
        _credit_key_fallback_count += 1
    if _is_donated_404(exc):
        global _donated_404_fallback_count  # noqa: PLW0603  # no await below; increments stay bytecode-atomic
        _donated_404_fallback_count += 1
        logger.warning(
            "Donated OpenRouter key returned 404 'no allowed providers' for model=%s; "
            "falling back to general (paid personal) key. This means the donated key's "
            "server-side allowed-providers list does not cover this model's upstream "
            "provider. Run will complete, then exit non-zero to alert. error=%s: %s",
            model,
            type(exc).__name__,
            exc,
        )
    else:
        logger.warning(
            "PAID PERSONAL-KEY FALLBACK: donated OpenRouter key failed for model=%s, so this "
            "call billed to the personal OPENROUTER_API_KEY instead of the free donated key. "
            "%s error=%s: %s",
            model,
            _fallback_alert_note(suppressible=suppressible),
            type(exc).__name__,
            exc,
        )


class FallbackOpenRouterLlm(GeneralLlm):
    """A GeneralLlm wrapper that prefers a Metaculus-donated OpenRouter key, falling back to the
    operator's general key on credential/credit/allowed-providers errors. Used for models routed
    through providers covered by the donated key (see ``DONATED_KEY_PROVIDERS``).

    ``role`` tags every completion for the ``CREDIT_ROLE_SPEND`` ledger
    (``credit_telemetry.llm_call_metadata``); the primary is stamped ``donated`` and the
    secondary ``personal``, so a fallback's spend lands on the key that actually billed it.
    """

    def __init__(
        self,
        *,
        model: str,
        primary_api_key: str | None,
        secondary_api_key: str | None,
        role: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model, api_key=primary_api_key, metadata=llm_call_metadata(role, DONATED_KEY_ALIAS), **kwargs
        )
        self._secondary_llm: GeneralLlm | None = (
            GeneralLlm(
                model=model, api_key=secondary_api_key, metadata=llm_call_metadata(role, PERSONAL_KEY_ALIAS), **kwargs
            )
            if secondary_api_key
            else None
        )

    async def invoke(self, prompt: Any, system_prompt: str | None = None) -> str:  # type: ignore[override]
        """Try the donated key, then the personal key when the failure is key-scoped.

        ``system_prompt`` is threaded through both key paths, so it survives a fallback unchanged
        (the parameter was widened to match ft 0.2.92's ``GeneralLlm.invoke`` signature). The
        deprecation re-record and the ASYNC120 waiver: docs/operations.md "The wrapper and the
        builder".
        """
        try:
            return await self._invoke_once_using_primary(prompt, system_prompt)
        except Exception as e:
            # Re-record with the real slug; duplicates are harmless, cli.py only checks non-empty.
            _record_deprecation_if_matched(self.model, str(e))
            if self._secondary_llm is not None and should_retry_with_general_key(e):
                # ASYNC120: the caller asked for a fallback, so dropping the primary's error is correct.
                await record_donated_key_fallback(self.model, e)
                return await self._invoke_once_using_secondary(prompt, system_prompt)
            raise

    async def _invoke_once_using_primary(self, prompt: Any, system_prompt: str | None = None) -> str:
        return await super().invoke(prompt, system_prompt)

    async def _invoke_once_using_secondary(self, prompt: Any, system_prompt: str | None = None) -> str:
        if self._secondary_llm is None:
            raise RuntimeError("No secondary key configured for fallback")
        return await self._secondary_llm.invoke(prompt, system_prompt)


def build_llm_with_openrouter_fallback(model: str, *, role: str | None = None, **kwargs: Any) -> GeneralLlm:
    """
    Construct a GeneralLlm that automatically falls back from the Metaculus-donated OpenRouter
    key to the operator's general key for providers covered by the donated key (see
    ``DONATED_KEY_PROVIDERS``). For other models, returns a plain GeneralLlm.

    ``role`` names the spend line every completion of this LLM is booked under in the
    ``CREDIT_ROLE_SPEND`` ledger (``credit_telemetry.llm_call_metadata`` lists the roles in
    use). Pass it at every production call site; a missing role books as ``untagged``.
    """
    if should_route_via_donated_key(model):
        special_key = os.getenv(OAI_ANTH_OPENROUTER_KEY_ENV)
        general_key = os.getenv(OPENROUTER_API_KEY_ENV)

        # If both keys exist and are distinct, use the fallback wrapper
        if special_key and general_key and special_key != general_key:
            return FallbackOpenRouterLlm(
                model=model,
                primary_api_key=special_key,
                secondary_api_key=general_key,
                role=role,
                **kwargs,
            )

        # Else fall back to whichever key is available (no runtime fallback possible)
        api_key = special_key or general_key
        key_alias = DONATED_KEY_ALIAS if special_key else PERSONAL_KEY_ALIAS
        return GeneralLlm(model=model, api_key=api_key, metadata=llm_call_metadata(role, key_alias), **kwargs)

    # Every other slug: keyless, so litellm reads OPENROUTER_API_KEY from the environment.
    return GeneralLlm(model=model, metadata=llm_call_metadata(role, plain_llm_key_alias(model)), **kwargs)
