"""Tests for the elapsed-gated transient-retry helper (``metaculus_bot.llm_retry``).

Background: a production LLM call failed with an INSTANT (``time taken=0.001s``)
``litellm_exc.Timeout`` during a concurrent async burst. Because the call was
``allowed_tries=1`` (forecasting-tools' tenacity ``stop_after_attempt(1)`` ⇒ zero
retries), it lost all work with no recovery.

``invoke_with_transient_retry`` retries ONLY fast/transient failures and NEVER
retries slow ones. The elapsed-time gate is the load-bearing safety constraint:
retrying a real multi-minute timeout would be catastrophic, so a failure that
took longer than ``max_elapsed_s`` is never retried regardless of its type.

All tests are self-contained (no live API). Backoff-sequence tests patch
``asyncio.sleep`` so they run instantly. Awaitable factories are built with
``AsyncMock`` (matching ``tests/test_gap_fill_pass.py`` style) so the source has
no bare ``async def`` helpers for flake8-async to complain about.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import asknews_sdk.errors as asknews_errors
import asknews_sdk.response as asknews_response
import litellm.exceptions as litellm_exc
import pytest
import requests

from metaculus_bot import fetch_hardening
from metaculus_bot.constants import (
    CRUX_SOFT_DEADLINE,
    FORECASTER_SOFT_DEADLINE,
    KALSHI_CATALOGUE_WALL_TIMEOUT,
    MARKET_QUERY_AUTHOR_BACKOFFS,
    MARKET_QUERY_AUTHOR_WALL_TIMEOUT,
    MARKET_RANKER_BACKOFFS,
    MARKET_RANKER_WALL_TIMEOUT,
    METACULUS_CLOSE_WINDOW_SECONDS,
    PER_QUESTION_WALL_CLOCK_DEADLINE,
    PREDICTION_MARKET_TIMEOUT,
    PUBLISH_POST_RETRIES,
    PUBLISH_POST_TIMEOUT,
    STACKER_FALLBACK_SOFT_DEADLINE,
    STACKER_SOFT_DEADLINE,
    SUMMARIZER_WALL_TIMEOUT,
    WALL_CLOCK_STACKING_MIN_BUDGET,
)
from metaculus_bot.llm_configs import REASONING_MODEL_CONFIG, UTILITY_MODEL_CONFIG
from metaculus_bot.llm_retry import (
    DEFAULT_TRANSIENT_BACKOFFS,
    NON_RETRYABLE_HTTP_STATUS_CODES,
    TRANSIENT_RETRY_EXCEPTIONS,
    TRANSIENT_RETRY_MAX_ELAPSED_S,
    _is_transient_type,
    invoke_with_broad_retry,
    invoke_with_transient_retry,
    is_broadly_retryable,
    llm_error_fields,
    llm_status_code,
)
from metaculus_bot.research.market_retrieval import venues
from metaculus_bot.research.market_retrieval.generation import MANIFOLD_DETAIL_WALL_S
from metaculus_bot.research.market_retrieval.http import (
    HTTP_RETRY_BACKOFFS_SECS,
    HTTP_RETRY_JITTER_BOUNDS,
    PLATFORM_HTTP_TIMEOUT,
)
from metaculus_bot.research.prediction_market import SNAPSHOT_STAGE_BUDGET_S
from tests.conftest import PRODUCTION_KEY_LIMIT_403

# A wall_timeout comfortably larger than any simulated call duration in these
# tests, so the helper's own asyncio.wait_for never fires unless we want it to.
_BIG_WALL = 1000.0

# Captured BEFORE the autouse fixture patches the module-level asyncio.sleep.
# Tests that need a *real* wall-clock delay (to trip the wall_timeout guard or
# cross a real elapsed gate) use this rather than the patched no-op sleep.
_REAL_SLEEP = asyncio.sleep


def _timeout(msg: str = "Connection timed out.") -> litellm_exc.Timeout:
    return litellm_exc.Timeout(msg, model="m", llm_provider="openrouter")


def _fake_clock(start: float, after: float) -> Callable[[], float]:
    """Return a monotonic() stand-in: first reading ``start``, all later ``after``.

    Robust to however many times the helper (or asyncio internals) sample the
    clock — only the first call is the attempt-start, every subsequent reading
    reports the post-failure time.
    """
    state = {"n": 0}

    def _clock() -> float:
        value = start if state["n"] == 0 else after
        state["n"] += 1
        return value

    return _clock


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch asyncio.sleep inside llm_retry so backoffs don't slow the suite.

    Returned so individual tests can assert call args. The patch targets the
    name as imported in the module under test.
    """
    fake_sleep = AsyncMock()
    monkeypatch.setattr("metaculus_bot.llm_retry.asyncio.sleep", fake_sleep)
    return fake_sleep


@pytest.mark.asyncio
async def test_fast_timeout_retries_then_succeeds() -> None:
    """A fast (sub-elapsed-gate) litellm_exc.Timeout retries and the next call wins."""
    awaitable = AsyncMock(side_effect=[_timeout(), "ok"])

    out = await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="t")

    assert out == "ok"
    assert awaitable.await_count == 2


@pytest.mark.asyncio
async def test_success_on_first_try_calls_once_no_sleep(no_real_sleep: AsyncMock) -> None:
    """Happy path: one call, no backoff sleep."""
    awaitable = AsyncMock(return_value="ok")

    out = await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="t")

    assert out == "ok"
    assert awaitable.await_count == 1
    no_real_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_slow_transient_failure_not_retried() -> None:
    """A litellm_exc.Timeout that takes LONGER than max_elapsed_s is NOT retried.

    This is the load-bearing safety gate: a real multi-minute stall surfaces as a
    transient-typed exception but must never be retried. We simulate elapsed time
    via a patched time.monotonic so the test stays instant.
    """
    awaitable = AsyncMock(side_effect=_timeout("real stall"))
    # start reads 0.0; every later reading reports past the gate, so the measured
    # elapsed forces the slow-failure branch regardless of internal clock samples.
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with patch("metaculus_bot.llm_retry.time.monotonic", clock), pytest.raises(litellm_exc.Timeout):
        await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="t")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_asyncio_timeout_error_not_retried() -> None:
    """The wall-clock guard firing (asyncio.TimeoutError) is a SLOW failure — never retried.

    The autouse fixture patches the helper's backoff ``asyncio.sleep``, but the
    helper's ``asyncio.wait_for`` is untouched, so a tiny ``wall_timeout`` against
    a long-running awaitable produces a genuine ``asyncio.TimeoutError``. Because
    that signals the slow-failure path (elapsed ≈ wall_timeout), there must be no
    retry — only one attempt is started.
    """
    calls = {"n": 0}

    async def _hang() -> str:
        # _REAL_SLEEP, not the patched no-op, so the call genuinely outlives the
        # 0.01s wall_timeout and asyncio.wait_for raises TimeoutError.
        await _REAL_SLEEP(5)
        return "never"

    def factory() -> Awaitable[str]:
        calls["n"] += 1
        return _hang()

    with pytest.raises(asyncio.TimeoutError):
        await invoke_with_transient_retry(factory, wall_timeout=0.01, label="t")

    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_non_transient_exception_not_retried() -> None:
    """A non-transient error (AuthenticationError) is re-raised on the first attempt."""
    awaitable = AsyncMock(side_effect=litellm_exc.AuthenticationError("bad key", model="m", llm_provider="openrouter"))

    with pytest.raises(litellm_exc.AuthenticationError):
        await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="t")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_value_error_not_retried() -> None:
    """A plain ValueError is not in the transient set → not retried."""
    awaitable = AsyncMock(side_effect=ValueError("nope"))

    with pytest.raises(ValueError, match="nope"):
        await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="t")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_rate_limit_error_not_retried() -> None:
    """RateLimitError is deliberately excluded (handled by key-swap / AskNews backoff)."""
    awaitable = AsyncMock(side_effect=litellm_exc.RateLimitError("slow down", model="m", llm_provider="openrouter"))

    with pytest.raises(litellm_exc.RateLimitError):
        await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="t")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_exhausting_backoffs_reraises_last_and_calls_n_plus_one_times() -> None:
    """All attempts fail transiently → last exc re-raised, awaitable called len(backoffs)+1 times."""
    awaitable = AsyncMock(
        side_effect=litellm_exc.APIConnectionError(message="blip", model="m", llm_provider="openrouter")
    )

    with pytest.raises(litellm_exc.APIConnectionError):
        await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="t")

    assert awaitable.await_count == len(DEFAULT_TRANSIENT_BACKOFFS) + 1


@pytest.mark.asyncio
async def test_backoff_sequence_in_order(no_real_sleep: AsyncMock) -> None:
    """Backoff sleeps happen in order: 1.0, 10.0, 30.0 (the default sequence)."""
    awaitable = AsyncMock(
        side_effect=litellm_exc.InternalServerError(message="500", model="m", llm_provider="openrouter")
    )

    with pytest.raises(litellm_exc.InternalServerError):
        await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="t")

    actual = [c.args[0] for c in no_real_sleep.await_args_list]
    assert actual == list(DEFAULT_TRANSIENT_BACKOFFS)
    assert actual == [1.0, 10.0, 30.0]


@pytest.mark.asyncio
async def test_custom_backoffs_control_attempt_count(no_real_sleep: AsyncMock) -> None:
    """A custom backoffs tuple sets attempts = len(backoffs)+1 and its own sleep schedule."""
    awaitable = AsyncMock(
        side_effect=litellm_exc.ServiceUnavailableError(message="503", model="m", llm_provider="openrouter")
    )

    with pytest.raises(litellm_exc.ServiceUnavailableError):
        await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="t", backoffs=(0.5, 2.0))

    assert awaitable.await_count == 3  # len((0.5, 2.0)) + 1
    assert [c.args[0] for c in no_real_sleep.await_args_list] == [0.5, 2.0]


@pytest.mark.asyncio
async def test_elapsed_gate_blocks_retry_even_for_transient_type() -> None:
    """A transient-typed exc whose elapsed exceeds max_elapsed_s is NOT retried.

    Distinct from test_slow_transient_failure_not_retried in that it uses a real
    (short) custom max_elapsed_s and a genuine sleep to cross it, exercising the
    actual time.monotonic measurement rather than a patched clock.
    """

    async def _slow_then_fail() -> str:
        # _REAL_SLEEP so the measured elapsed actually crosses the 0.01s gate;
        # the patched no-op sleep would report elapsed≈0 and wrongly retry.
        await _REAL_SLEEP(0.05)
        raise _timeout("slow blip")

    awaitable = AsyncMock(side_effect=_slow_then_fail)

    with pytest.raises(litellm_exc.Timeout):
        await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="t", max_elapsed_s=0.01)

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_make_awaitable_called_fresh_each_attempt() -> None:
    """The factory must produce a FRESH awaitable per attempt (coroutines are single-await).

    If the helper reused a single coroutine, attempt 2 would raise RuntimeError
    ("cannot reuse already awaited coroutine") instead of retrying cleanly.
    """
    factory_calls = {"n": 0}
    awaitable = AsyncMock(side_effect=[_timeout("blip"), "ok"])

    def factory():
        factory_calls["n"] += 1
        return awaitable()

    out = await invoke_with_transient_retry(factory, wall_timeout=_BIG_WALL, label="t")

    assert out == "ok"
    assert factory_calls["n"] == 2
    assert awaitable.await_count == 2


@pytest.mark.asyncio
async def test_warning_logged_on_each_retry(caplog: pytest.LogCaptureFixture) -> None:
    """Each retry emits a WARNING carrying the label so retries are auditable in logs."""
    awaitable = AsyncMock(side_effect=_timeout("blip"))

    with caplog.at_level("WARNING", logger="metaculus_bot.llm_retry"), pytest.raises(litellm_exc.Timeout):
        await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="my_label")

    retry_warnings = [r for r in caplog.records if "my_label" in r.message]
    # One warning per retry = len(backoffs) (the final attempt does not warn).
    assert len(retry_warnings) == len(DEFAULT_TRANSIENT_BACKOFFS)


def test_elapsed_gate_default_is_30s() -> None:
    """Canary: the universal "no retry after 30s" deadline-safety rule (Round-2).

    The gate measures the failed attempt's own duration; a 5-min reasoning attempt
    that then times out must never spawn another call (it would miss the question
    submission deadline). 30s sits above any genuine transient blip and far below
    every real per-call timeout (120/300/360/420/480/500s).
    """
    assert TRANSIENT_RETRY_MAX_ELAPSED_S == 30.0


def test_wall_timeout_is_per_attempt_not_a_total_budget() -> None:
    """Pin the semantics every caller's budget arithmetic depends on.

    ``invoke_with_transient_retry`` starts a FRESH ``asyncio.wait_for(...,
    wall_timeout)`` inside the attempt loop, so ``wall_timeout`` bounds each attempt
    individually — the worst case for a call site is
    ``(len(backoffs) + 1) * wall_timeout + sum(backoffs)``, NOT ``wall_timeout``.
    Callers that size the value to fit inside an outer ``wait_for`` (see
    ``MARKET_QUERY_AUTHOR_WALL_TIMEOUT`` against ``PREDICTION_MARKET_TIMEOUT``) read it
    the other way round and silently overrun, so make the semantics explicit here
    rather than leaving them implied by the loop body.

    Two attempts each hitting the wall must therefore consume ~2x the wall, and the
    per-attempt ``asyncio.TimeoutError`` must not be retried (it is a SLOW failure).
    """
    wall = 0.05
    attempt_durations: list[float] = []

    async def _hang() -> str:
        started = time.monotonic()
        try:
            await _REAL_SLEEP(5)
        finally:
            attempt_durations.append(time.monotonic() - started)
        return "never"

    async def _run() -> None:
        # A predicate that accepts asyncio.TimeoutError plus a gate above the wall
        # forces the retry the production predicates deliberately refuse, which is
        # what exposes the per-attempt (rather than total) nature of the cap.
        with pytest.raises(asyncio.TimeoutError):
            await invoke_with_transient_retry(
                _hang,
                wall_timeout=wall,
                label="per_attempt_wall",
                backoffs=(0.0,),
                max_elapsed_s=_BIG_WALL,
                predicate=lambda _exc: True,
            )

    asyncio.run(_run())

    assert len(attempt_durations) == 2, "wall_timeout bounds each attempt, so both attempts must run"
    assert sum(attempt_durations) >= wall * 1.5, (
        f"two attempts must cost ~2x the wall, not one wall total: {attempt_durations=}"
    )


def _stage_worst(wall: float, backoffs: tuple[float, ...]) -> float:
    """One retried stage's worst-case wall clock.

    The wall is PER ATTEMPT (see the test above), so the ceiling is
    ``(len(backoffs) + 1) * wall + sum(backoffs)`` and NOT the wall itself. At a 15.0s keyword
    wall that product was 31.0s against a 30.0s ``PREDICTION_MARKET_TIMEOUT`` — the overrun the
    per-attempt cap existed to prevent stayed reachable without either attempt hitting its own
    wall, because ``sum(backoffs) < T`` and ``wall <= T`` both held true at 31.0s. Asserting
    the product rather than the two halves separately is the whole point of this file's copy of
    the arithmetic.
    """
    return (len(backoffs) + 1) * wall + sum(backoffs)


def test_market_retrieval_stage_budgets_fit_the_snapshot_timeout() -> None:
    """The ranked pipeline's SERIAL CHAIN of worst cases must fit the snapshot budget.

    Recomputed here from the raw constants rather than read off the module, so this is an
    independent derivation of the arithmetic and not a restatement of it: stages 1a (Kalshi
    catalogue, PredictIt dump) and 1b (query author) run concurrently, so the chain takes their
    max, then venue search, then the Manifold detail fan-out, then ranking, all serial. If the
    chain overran, the snapshot-level ``wait_for`` would cancel every stage and mis-attribute
    the loss to ``snapshot`` rather than to whichever stage actually ran long.

    The INEQUALITY is what is asserted, deliberately — not equality with a precomputed number,
    which would have to be edited every time a wall moves and would therefore stop being a
    check.
    """
    max_venue_attempts = max(
        venues.MANIFOLD_MAX_ATTEMPTS,
        venues.POLYMARKET_MAX_ATTEMPTS,
        venues.PREDICTIT_MAX_ATTEMPTS,
    )
    http_stage = (
        PLATFORM_HTTP_TIMEOUT * max_venue_attempts
        + sum(HTTP_RETRY_BACKOFFS_SECS[: max_venue_attempts - 1]) * HTTP_RETRY_JITTER_BOUNDS[1]
    )
    assert MANIFOLD_DETAIL_WALL_S == 10.0
    author = _stage_worst(MARKET_QUERY_AUTHOR_WALL_TIMEOUT, MARKET_QUERY_AUTHOR_BACKOFFS)
    ranker = _stage_worst(MARKET_RANKER_WALL_TIMEOUT, MARKET_RANKER_BACKOFFS)
    chain = max(author, KALSHI_CATALOGUE_WALL_TIMEOUT, http_stage) + http_stage + MANIFOLD_DETAIL_WALL_S + ranker

    assert chain <= PREDICTION_MARKET_TIMEOUT, (
        f"the pipeline's worst-case chain of {chain}s must fit inside the {PREDICTION_MARKET_TIMEOUT}s snapshot budget"
    )
    # The provider recomputes the same chain to WARN when an env override sits below it, so the
    # two derivations must agree — otherwise the warning fires (or stays silent) on a number
    # nothing else in the codebase believes.
    assert chain == SNAPSHOT_STAGE_BUDGET_S, (
        f"prediction_market.SNAPSHOT_STAGE_BUDGET_S ({SNAPSHOT_STAGE_BUDGET_S}s) disagrees with the "
        f"independently derived chain ({chain}s)"
    )


def test_the_ranker_gets_no_retry() -> None:
    """One attempt for the ranking call, by design rather than by omission.

    36 of 36 measured calls parsed on the first try, a retry on a ~36k-token prompt is expensive
    latency, and there is a good deterministic fallback sitting right there (the pool-order
    fail-open slate). The query author DOES get one retry, because its worst case is cheap and
    its output is additive.
    """
    assert MARKET_RANKER_BACKOFFS == ()
    assert len(MARKET_QUERY_AUTHOR_BACKOFFS) == 1


# ---------------------------------------------------------------------------
# Soft-deadline / model-timeout alignment
#
# Every soft deadline in constants.py is documented in terms of a timeout that
# lives in llm_configs.py ("slightly above the stacker LLM's own litellm
# timeout", "matches the summarizer's litellm per-request timeout", "the
# analyzer's own bound ... is looser than this"). Those relationships are what
# make each wrapper a backstop rather than the primary bound, and NOTHING
# enforced them: the values sit in two files that change for unrelated reasons,
# and the existing deadline tests all monkeypatch the constant to a fraction of a
# second to test the wrapper's BEHAVIOR, which passes at any value. So a
# llm_configs.py timeout bump silently inverts the intended ordering and the
# soft deadline starts firing first, converting a clean provider timeout into an
# opaque asyncio.TimeoutError. Pin the arithmetic the comments assert, in the
# same spirit as the keyword-budget test above.
# ---------------------------------------------------------------------------


def test_stacker_soft_deadline_sits_above_the_reasoning_model_timeout() -> None:
    """The stacker wrapper must be a backstop, not the primary bound.

    ``STACKER_SOFT_DEADLINE``'s comment says it is set "slightly above the stacker
    LLM's own litellm timeout ... so the model's timeout fires first with a clean
    exception when possible". Inverted, every stalled stacker surfaces as an
    ``asyncio.TimeoutError`` from the wrapper instead of the provider's own typed
    error, and the fallback chain loses the reason it fired.
    """
    model_timeout = REASONING_MODEL_CONFIG["timeout"]
    assert model_timeout < STACKER_SOFT_DEADLINE, (
        f"STACKER_SOFT_DEADLINE={STACKER_SOFT_DEADLINE} must exceed the stacker's litellm "
        f"timeout={model_timeout} so the model's own timeout fires first"
    )
    # "Slightly" above: a huge margin would mean a stuck call burns budget the
    # fallback stacker needs. The fallback's own deadline is the natural scale.
    assert STACKER_SOFT_DEADLINE - model_timeout <= STACKER_FALLBACK_SOFT_DEADLINE, (
        "the backstop margin should stay small relative to the fallback's budget"
    )


def test_stacker_fallback_deadline_is_tighter_than_the_primary() -> None:
    """The fallback fires late on the critical path, so it gets less time.

    Documented at ``STACKER_FALLBACK_SOFT_DEADLINE``. If this ever inverted, a
    failing primary followed by a slow fallback would eat the publish budget that
    ``WALL_CLOCK_STACKING_MIN_BUDGET`` reserves.
    """
    assert STACKER_FALLBACK_SOFT_DEADLINE < STACKER_SOFT_DEADLINE


def test_summarizer_wall_timeout_matches_the_utility_model_timeout() -> None:
    """``SUMMARIZER_WALL_TIMEOUT`` is documented as matching its litellm budget.

    The point of the equality is that the per-attempt cap and the underlying
    request budget expire together, so a breach means the request really is stuck
    rather than the wrapper being impatient.
    """
    assert UTILITY_MODEL_CONFIG["timeout"] == SUMMARIZER_WALL_TIMEOUT, (
        f"SUMMARIZER_WALL_TIMEOUT={SUMMARIZER_WALL_TIMEOUT} must equal the summarizer's "
        f"litellm timeout={UTILITY_MODEL_CONFIG['timeout']}"
    )


def test_crux_soft_deadline_is_tighter_than_the_analyzer_model_timeout() -> None:
    """The crux extractor is deliberately capped BELOW its model's own timeout.

    Unlike the stacker, here the wrapper is meant to fire first: the comment says
    the analyzer's own per-attempt bound "is looser than this, so without the
    wrapper a stalled call runs well past the crux's usefulness". A crux that
    arrives too late is worthless even if it arrives, so this one inverts the
    backstop relationship on purpose — pin it so the intent survives.
    """
    assert UTILITY_MODEL_CONFIG["timeout"] > CRUX_SOFT_DEADLINE, (
        f"CRUX_SOFT_DEADLINE={CRUX_SOFT_DEADLINE} must undercut the analyzer's litellm "
        f"timeout={UTILITY_MODEL_CONFIG['timeout']} so the wrapper bounds a stalled crux"
    )


def test_forecaster_soft_deadline_caps_the_model_retry_ladder() -> None:
    """A single stuck forecaster must not be able to hold a question.

    ``FORECASTER_SOFT_DEADLINE``'s comment describes the worst case it caps:
    the reasoning model's litellm timeout times its ``allowed_tries``. If a config
    bump ever made the ladder shorter than the deadline the wrapper would stop
    being the binding constraint and the drop-with-a-WARNING path would go dead.
    """
    ladder_worst_case = REASONING_MODEL_CONFIG["timeout"] * REASONING_MODEL_CONFIG["allowed_tries"]
    assert ladder_worst_case > FORECASTER_SOFT_DEADLINE, (
        f"FORECASTER_SOFT_DEADLINE={FORECASTER_SOFT_DEADLINE} must cap the model's own "
        f"{ladder_worst_case}s retry ladder, else the wrapper never binds"
    )


def test_per_question_deadline_reserves_exactly_the_publish_budget() -> None:
    """The close-window remainder is sized to the stacking/publish reserve.

    ``PER_QUESTION_WALL_CLOCK_DEADLINE`` is documented as sitting "just inside"
    the hourly close window, and the leftover is what pays for stacker-skip plus
    the two publish POSTs. The equality is deliberate, not incidental: reserving
    LESS than ``WALL_CLOCK_STACKING_MIN_BUDGET`` would let a question run past its
    own close window while still believing it had time to publish.
    """
    reserve = METACULUS_CLOSE_WINDOW_SECONDS - PER_QUESTION_WALL_CLOCK_DEADLINE
    assert reserve >= WALL_CLOCK_STACKING_MIN_BUDGET, (
        f"only {reserve}s of the {METACULUS_CLOSE_WINDOW_SECONDS}s close window is reserved, "
        f"less than the {WALL_CLOCK_STACKING_MIN_BUDGET}s publish budget"
    )


def test_stacking_min_budget_clears_both_publish_posts() -> None:
    """``WALL_CLOCK_STACKING_MIN_BUDGET`` must cover the publish worst case.

    Two hardened POSTs (the prediction and the comment), each up to
    ``PUBLISH_POST_TIMEOUT * (PUBLISH_POST_RETRIES + 1)``. Documented at the
    constant; unpinned until now, so a retry-count bump could have silently made
    the reserve too small to publish inside.
    """
    publish_worst_case = 2 * PUBLISH_POST_TIMEOUT * (PUBLISH_POST_RETRIES + 1)
    assert publish_worst_case < WALL_CLOCK_STACKING_MIN_BUDGET, (
        f"WALL_CLOCK_STACKING_MIN_BUDGET={WALL_CLOCK_STACKING_MIN_BUDGET} must exceed the "
        f"{publish_worst_case}s two-POST publish worst case"
    )


# Broad-predicate retry (Round-2 change 3): for the allowed_tries>=2 sites
# (forecasters, crux analyzer, AskNews summarizer) that must keep their valuable
# fast retries (empty-response, parser hiccup) but obey the same 30s gate.
@pytest.mark.asyncio
async def test_broad_retry_retries_generic_error_under_gate() -> None:
    """A generic ValueError (e.g. empty-model-response) under the gate is retried."""
    awaitable = AsyncMock(side_effect=[ValueError("empty response"), "ok"])

    out = await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="b")

    assert out == "ok"
    assert awaitable.await_count == 2


@pytest.mark.asyncio
async def test_broad_retry_does_not_retry_authentication_error() -> None:
    """AuthenticationError is clearly-permanent → never retried, even under the gate."""
    awaitable = AsyncMock(side_effect=litellm_exc.AuthenticationError("bad key", model="m", llm_provider="openrouter"))

    with pytest.raises(litellm_exc.AuthenticationError):
        await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_broad_retry_does_not_retry_bad_request_error() -> None:
    """BadRequestError (and its subclasses, e.g. ContextWindowExceeded) is permanent."""
    awaitable = AsyncMock(side_effect=litellm_exc.BadRequestError("bad", model="m", llm_provider="openrouter"))

    with pytest.raises(litellm_exc.BadRequestError):
        await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == 1


@pytest.mark.parametrize(
    "bug_exc",
    [TypeError("x"), AttributeError("x"), NameError("x"), KeyError("x"), IndexError("x"), ImportError("x")],
)
@pytest.mark.asyncio
async def test_broad_retry_does_not_retry_python_bug_types(bug_exc: Exception) -> None:
    """Python-internal bug types fail fast (CLAUDE.md §2): never retried, surface immediately.

    A TypeError/AttributeError/etc. signals a code defect, never a transient API
    condition; retrying it 3× just delays a clean traceback. Distinct from the
    empty-response RuntimeError, which IS broadly retryable (covered elsewhere).
    """
    awaitable = AsyncMock(side_effect=bug_exc)

    with pytest.raises(type(bug_exc)):
        await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_broad_retry_retries_empty_response_runtimeerror() -> None:
    """RuntimeError (forecasting-tools' empty-model-response signal) IS broadly retried.

    Guards against over-tightening the predicate: RuntimeError must stay retryable
    even though it is a builtin, because it's the canonical valuable in-invoke retry.
    """
    awaitable = AsyncMock(side_effect=[RuntimeError("LLM answer is an empty string"), "ok"])

    out = await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="b")

    assert out == "ok"
    assert awaitable.await_count == 2


@pytest.mark.asyncio
async def test_broad_retry_does_not_retry_slow_failure_of_any_type() -> None:
    """A generic error that takes LONGER than the gate is NOT retried (deadline safety).

    Even though ValueError would normally be broadly-retryable, a slow attempt
    (>30s) must never trigger another call.
    """
    awaitable = AsyncMock(side_effect=ValueError("slow generic failure"))
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with (
        patch("metaculus_bot.llm_retry.time.monotonic", clock),
        pytest.raises(ValueError, match="slow generic failure"),
    ):
        await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_broad_retry_does_not_retry_asyncio_timeout() -> None:
    """asyncio.TimeoutError from the wall guard is a slow failure — not retried.

    It is retryable-by-type (not in PERMANENT_NO_RETRY_EXCEPTIONS), but it only
    fires at the wall cap (>> 30s), so the elapsed gate blocks the retry. We model
    the wall-cap firing as an asyncio.TimeoutError raised after the gate elapses
    (deterministic clock), rather than racing a real tiny wait_for timeout.
    """
    awaitable = AsyncMock(side_effect=TimeoutError())
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with patch("metaculus_bot.llm_retry.time.monotonic", clock), pytest.raises(asyncio.TimeoutError):
        await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_broad_retry_shares_backoff_sequence(no_real_sleep: AsyncMock) -> None:
    """The broad path reuses the same loop/backoff logic (no duplicated machinery)."""
    awaitable = AsyncMock(side_effect=ValueError("always"))

    with pytest.raises(ValueError, match="always"):
        await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == len(DEFAULT_TRANSIENT_BACKOFFS) + 1
    assert [c.args[0] for c in no_real_sleep.await_args_list] == list(DEFAULT_TRANSIENT_BACKOFFS)


@pytest.mark.asyncio
async def test_predicate_param_overrides_transient_type_check() -> None:
    """Supplying predicate= on invoke_with_transient_retry replaces the type check.

    A ValueError is NOT in the transient type set, but a predicate that accepts it
    makes it retryable — proving predicate supersedes the default isinstance check
    while the shared gate/backoff/wall logic is reused.
    """
    awaitable = AsyncMock(side_effect=[ValueError("blip"), "ok"])

    out = await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="p", predicate=lambda _exc: True)

    assert out == "ok"
    assert awaitable.await_count == 2


# ---------------------------------------------------------------------------
# Zero-output slow-exemption (2026-07-25 fix).
#
# A forecaster (claude-opus-4.8) returned a whitespace-only HTTP 200 body from
# OpenRouter; litellm re-wrapped the JSONDecodeError as an APIError whose message
# starts "Unable to get json response". The failure arrived SLOWLY (>30s of
# reasoning at xhigh effort), so the universal elapsed gate blocked the only
# retry layer and the forecaster was dropped — the question published on 2 of 3
# models and CI went red, with NO retry WARNING in the log.
#
# The fix exempts ZERO-OUTPUT failures (empty/whitespace body → APIError "Unable
# to get json response"; empty completion → forecasting-tools RuntimeError "empty
# string") from the elapsed gate: a slow zero-output is re-rolled ONCE, no
# backoff. A genuine slow timeout / generic slow failure stays blocked, because
# re-rolling a call that already spent the wall budget is the deadline risk the
# gate exists to prevent. The exemption is ANDed with should_retry, so it only
# fires on the broad-predicate sites (forecasters/crux/summarizer); the
# transient-predicate sites (stacker, research) reject APIError/RuntimeError by
# type and are untouched.
# ---------------------------------------------------------------------------

# The exact production signature from the 2026-07-25 run: an HTTP 200 with a
# whitespace-only body, so litellm's raw_response.json() raised a JSONDecodeError
# ("Expecting value: line 269 column 1") that litellm re-wrapped as this APIError.
_PROD_WHITESPACE_BODY_MESSAGE = "Unable to get json response - Expecting value: line 269 column 1 (char 1474)"


def _api_error(message: str) -> litellm_exc.APIError:
    """Build a litellm APIError mirroring the OpenRouter empty-body failure shape."""
    return litellm_exc.APIError(
        status_code=500, message=message, llm_provider="openrouter", model="anthropic/claude-opus-4.8"
    )


@pytest.mark.asyncio
async def test_slow_zero_output_api_error_is_retried_and_succeeds() -> None:
    """The 2026-07-25 regression: a slow (>gate) whitespace-body APIError is re-rolled once.

    Exact production message + a body that failed to JSON-parse. The first monotonic
    reading is the attempt start (0.0); every later reading is past the gate, forcing
    the slow-failure branch on attempt 0. The re-roll then wins.
    """
    awaitable = AsyncMock(side_effect=[_api_error(_PROD_WHITESPACE_BODY_MESSAGE), "recovered"])
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with patch("metaculus_bot.llm_retry.time.monotonic", clock):
        out = await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="forecaster_binary")

    assert out == "recovered"
    assert awaitable.await_count == 2


@pytest.mark.asyncio
async def test_slow_empty_string_runtimeerror_is_retried() -> None:
    """forecasting-tools' empty-completion RuntimeError, arriving slowly, is re-rolled once.

    The body parsed fine but the completion string was empty; general_llm.py raises
    ``RuntimeError("LLM answer is an empty string ...")``. That is zero-output too.
    """
    err = RuntimeError("LLM answer is an empty string. The model was anthropic/claude-opus-4.8 and the prompt was: ...")
    awaitable = AsyncMock(side_effect=[err, "ok"])
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with patch("metaculus_bot.llm_retry.time.monotonic", clock):
        out = await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="forecaster_numeric")

    assert out == "ok"
    assert awaitable.await_count == 2


@pytest.mark.asyncio
async def test_slow_zero_output_emits_warning_with_label(caplog: pytest.LogCaptureFixture) -> None:
    """The slow zero-output re-roll logs a WARNING carrying the label (diagnosable next time).

    The 2026-07-25 run had NO retry WARNING because the gate silently blocked the
    retry; the exemption must make its re-roll visible in logs.
    """
    awaitable = AsyncMock(side_effect=[_api_error(_PROD_WHITESPACE_BODY_MESSAGE), "ok"])
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with (
        patch("metaculus_bot.llm_retry.time.monotonic", clock),
        caplog.at_level("WARNING", logger="metaculus_bot.llm_retry"),
    ):
        out = await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="forecaster_mc")

    assert out == "ok"
    warnings = [r for r in caplog.records if "forecaster_mc" in r.message]
    assert len(warnings) == 1
    assert "zero-output" in warnings[0].message


@pytest.mark.asyncio
async def test_slow_zero_output_reroll_capped_at_one_no_backoff(no_real_sleep: AsyncMock) -> None:
    """A persistently-failing slow zero-output is re-rolled EXACTLY once (no backoff ladder).

    The exemption bypasses the deadline gate, so it must stay bounded: one immediate
    re-roll, no backoff sleeps. Uses a real (tiny) sleep against a small gate so each
    attempt genuinely measures as slow, exercising the real time.monotonic path.
    """

    async def _slow_zero_output() -> str:
        await _REAL_SLEEP(0.05)  # > the 0.01s gate below → measured as slow
        raise _api_error(_PROD_WHITESPACE_BODY_MESSAGE)

    awaitable = AsyncMock(side_effect=_slow_zero_output)

    with pytest.raises(litellm_exc.APIError):
        await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="b", max_elapsed_s=0.01)

    assert awaitable.await_count == 2  # attempt 0 + exactly one exempted re-roll
    no_real_sleep.assert_not_awaited()  # the exemption path takes no backoff sleep


@pytest.mark.asyncio
async def test_fast_zero_output_api_error_retried_unchanged() -> None:
    """A FAST whitespace-body APIError retries via the normal ladder (behavior unchanged)."""
    awaitable = AsyncMock(side_effect=[_api_error(_PROD_WHITESPACE_BODY_MESSAGE), "ok"])

    out = await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="b")

    assert out == "ok"
    assert awaitable.await_count == 2


@pytest.mark.asyncio
async def test_slow_non_empty_runtimeerror_not_exempted() -> None:
    """A slow RuntimeError whose message is NOT the empty-completion signal stays blocked.

    Guards the zero-output predicate against over-matching by exception TYPE: only the
    "empty string" message shape is exempt, not every RuntimeError.
    """
    awaitable = AsyncMock(side_effect=RuntimeError("some unrelated runtime failure"))
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with patch("metaculus_bot.llm_retry.time.monotonic", clock), pytest.raises(RuntimeError):
        await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_slow_permanent_error_not_retried_even_slow() -> None:
    """A permanent error (BadRequestError) is never retried, even slow — the exemption can't reach it.

    The zero-output check is ANDed with should_retry, and the permanent types are not
    litellm.APIError / RuntimeError subclasses, so a permanent error can never sneak
    through the slow-exemption path. Content-policy blocks and bad requests must still
    fail fast (CLAUDE.md §2).
    """
    awaitable = AsyncMock(
        side_effect=litellm_exc.BadRequestError("Unable to get json response - malformed", model="m", llm_provider="o")
    )
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with patch("metaculus_bot.llm_retry.time.monotonic", clock), pytest.raises(litellm_exc.BadRequestError):
        await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_transient_predicate_does_not_exempt_zero_output() -> None:
    """The default (transient-type) predicate does NOT get the zero-output exemption.

    The stacker and research providers call invoke_with_transient_retry with the
    default predicate; a whitespace-body APIError is not a transient TYPE, so
    should_retry rejects it and the exemption never fires. This is what scopes the
    fix to the broad-predicate sites (forecasters/crux/summarizer) without any
    per-call-site branching — the stacker's tight 500s/300s budget is untouched.
    """
    awaitable = AsyncMock(side_effect=_api_error(_PROD_WHITESPACE_BODY_MESSAGE))
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with patch("metaculus_bot.llm_retry.time.monotonic", clock), pytest.raises(litellm_exc.APIError):
        await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="stacker_binary")

    assert awaitable.await_count == 1


# ---------------------------------------------------------------------------
# Deterministic-4xx non-retry (2026-07-26 fix).
#
# The donated OpenRouter key hit its spend cap and OpenRouter answered every call
# with HTTP 403 "Key limit exceeded (total limit)" in ~35ms. litellm has no 403
# branch in _map_openrouter_exception, so it arrives as a BARE APIError — the root
# of the litellm/openai tree, matching nothing in PERMANENT_NO_RETRY_EXCEPTIONS.
# (PermissionDeniedError is in that list and reads as though it covers 403, but it
# subclasses a different openai branch and litellm never raises it for OpenRouter,
# so it gave zero coverage.) The broad predicate therefore called it retryable, and
# because it failed in 0.035s the TRANSIENT_RETRY_MAX_ELAPSED_S gate — designed to block
# SLOW failures — waved it through: the AskNews summarizer burned ~42s on the full
# DEFAULT_TRANSIENT_BACKOFFS ladder and two forecasters each did a 3-attempt dance, all
# against a key that was dry the entire run.
#
# The fix reads the authoritative ``status_code`` int that every litellm exception
# carries and treats the deterministic client-error statuses as permanent. Substring
# matching numbers against the message is NOT viable here: litellm builds the message
# as f"APIError: {provider} - {body}", and the OpenRouter body embeds both a 64-hex
# key hash and, on a moderation refusal, up to ~100 chars of our own prompt in
# ``flagged_input`` — either can make a status-shaped number appear that was never a
# status. How likely the key hash alone is to do that is derived by
# ``test_key_hash_status_collision_is_small_but_nonnegligible`` below, which is the
# only place that arithmetic is written down.
# ---------------------------------------------------------------------------

# The status substrings a message-scanning classifier would look for. Not a code
# constant: the point is what a reader of the raw body might match on.
_STATUS_SUBSTRINGS: tuple[str, ...] = ("401", "402", "403", "429", "502", "503")
_KEY_HASH_LEN = 64
_HEX_ALPHABET = "0123456789abcdef"


def _p_hash_contains(patterns: tuple[str, ...], length: int = _KEY_HASH_LEN) -> float:
    """Exact P(a uniform random hex string of ``length`` contains any of ``patterns``).

    DP over "the last two characters seen", which is sufficient state because every
    pattern is three characters long. Exact rather than a Poisson/independence
    approximation, and it handles the overlaps between patterns ("402" and "403" share
    the prefix "40") that a naive 1-(1-p)^n would get wrong.
    """
    assert all(len(p) == 3 for p in patterns), "state is the last 2 chars, so patterns must be length 3"
    # Probability mass per suffix state; absorbed (matched) mass leaves the system.
    states: dict[str, float] = {"": 1.0}
    step = 1.0 / len(_HEX_ALPHABET)
    for _ in range(length):
        nxt: dict[str, float] = {}
        for suffix, mass in states.items():
            for char in _HEX_ALPHABET:
                window = suffix + char
                if window in patterns:
                    continue  # matched, so this path is absorbed
                nxt[window[-2:]] = nxt.get(window[-2:], 0.0) + mass * step
        states = nxt
    return 1.0 - sum(states.values())


def test_key_hash_status_collision_is_small_but_nonnegligible() -> None:
    """Why reading status digits out of the message body is unsafe, derived not asserted.

    Every classifier in this repo that could have matched status digits in prose instead
    reads ``llm_status_code``. The justification is that an OpenRouter body carries a
    64-hex key hash whose characters are status-shaped often enough to matter. This pins
    that claim as arithmetic so it can't drift into folklore, and deliberately asserts
    BANDS rather than exact figures — the design only depends on the order of magnitude.
    """
    single = _p_hash_contains(("403",))
    # Order 1%: far too likely to dismiss, given the cost is stranding the ensemble on a
    # dry key, and far too unlikely to be caught by eyeballing one live key hash.
    assert 0.01 < single < 0.02

    any_status = _p_hash_contains(_STATUS_SUBSTRINGS)
    # Order 10% across the six statuses a message scanner might match — roughly one key
    # rotation in twelve produces a hash that lies about its own status.
    assert 0.05 < any_status < 0.15

    # Six near-disjoint chances, so the union lands near (but below) six times one.
    assert 5.0 * single < any_status < 6.0 * single


# Verbatim from the 2026-07-26 06:45 UTC production run (run_log_facts.md): the donated
# key at $0.00 of its $850 cap. Note what it does NOT contain — "credit", "insufficient",
# "balance", or "402" — and that the only "403" is the JSON code field, which is exactly
# why classification must read status_code, not the text. Shared from conftest so this
# reasoning about the exact bytes stays true for every suite that replays it.
_PROD_KEY_LIMIT_MESSAGE = PRODUCTION_KEY_LIMIT_403


def _api_error_with_status(status: int, message: str = "boom") -> litellm_exc.APIError:
    """Build a bare litellm APIError carrying ``status``, the shape OpenRouter 403s take."""
    return litellm_exc.APIError(
        status_code=status, message=message, llm_provider="openrouter", model="openai/gpt-5.6-sol"
    )


@pytest.mark.parametrize("status", sorted(NON_RETRYABLE_HTTP_STATUS_CODES))
def test_broad_predicate_rejects_deterministic_client_statuses(status: int) -> None:
    """A bare APIError carrying a hard 4xx is permanent — re-rolling it cannot change it.

    402 and 403 are the ones that actually move: 400/401/404/422 already map to typed
    permanent litellm classes, so the status rule is close to a no-op for them.
    """
    assert is_broadly_retryable(_api_error_with_status(status)) is False


def test_broad_predicate_rejects_production_key_limit_403() -> None:
    """The verbatim 2026-07-26 dry-donated-key 403 is permanent, not a transient blip."""
    assert is_broadly_retryable(_api_error_with_status(403, _PROD_KEY_LIMIT_MESSAGE)) is False


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_broad_predicate_still_retries_server_and_rate_limit_statuses(status: int) -> None:
    """5xx, 429 and 408 stay retryable — those genuinely can succeed on a re-roll."""
    assert is_broadly_retryable(_api_error_with_status(status)) is True


def test_transient_types_are_never_reclassified_as_permanent() -> None:
    """Guard against generalizing the rule to "all 4xx are permanent".

    ``litellm.Timeout`` carries status 408 — a 4xx — and is the ORIGINAL reason this
    module exists (an instant 0.001s Timeout during a concurrent async burst that lost
    all work at allowed_tries=1). Widening NON_RETRYABLE_HTTP_STATUS_CODES to the whole
    4xx range would silently kill that, so pin every transient type as retryable.
    """
    transient = (
        _timeout(),
        litellm_exc.APIConnectionError(message="conn", model="m", llm_provider="openrouter"),
        litellm_exc.InternalServerError(message="500", model="m", llm_provider="openrouter"),
        litellm_exc.ServiceUnavailableError(message="503", model="m", llm_provider="openrouter"),
    )
    # Every configured transient type is represented above, so widening the type set
    # without extending this guard fails here rather than silently going uncovered.
    assert {type(exc) for exc in transient} == set(TRANSIENT_RETRY_EXCEPTIONS)
    for exc in transient:
        assert is_broadly_retryable(exc) is True, f"{type(exc).__name__} must stay retryable"
        assert _is_transient_type(exc) is True, f"{type(exc).__name__} must stay transient"


def test_broad_predicate_retries_zero_output_statuses() -> None:
    """Regression guard on the zero-output carve-out (2026-07-25 fix).

    The whitespace-body APIError takes its status from the raw response: 200 in the
    production case, 500 in the repo fixture. Neither may appear in the non-retryable
    set, or the carve-out that recovered a dropped forecaster silently dies.
    """
    assert 200 not in NON_RETRYABLE_HTTP_STATUS_CODES
    assert 500 not in NON_RETRYABLE_HTTP_STATUS_CODES
    assert is_broadly_retryable(_api_error_with_status(200, _PROD_WHITESPACE_BODY_MESSAGE)) is True
    assert is_broadly_retryable(_api_error(_PROD_WHITESPACE_BODY_MESSAGE)) is True


def test_broad_predicate_ignores_status_on_non_llm_exceptions() -> None:
    """The status rule is scoped to the litellm/openai exception tree.

    ``requests``-shaped HTTP errors carry their status on ``.response``, not on the
    exception, and this repo deliberately treats a CDN/WAF 403 on the Metaculus
    question-list endpoint as RETRYABLE (the 2026-05-19 incident — see
    ``fetch_hardening._is_retryable`` and its tests). "403 is permanent" is a claim
    about LLM provider calls only, so nothing outside that tree may be reclassified.
    """
    cdn_403 = requests.HTTPError("403 Client Error")
    cdn_403.response = MagicMock(spec=requests.Response)
    cdn_403.response.status_code = 403

    assert fetch_hardening._is_retryable(cdn_403) is True
    assert is_broadly_retryable(cdn_403) is True


def test_broad_predicate_falls_back_to_type_check_without_status() -> None:
    """A plain Exception whose text mentions 403 keeps its old classification.

    Nothing textual was added: the rule reads the status attribute or nothing at all.
    A statusless exception still flows through the type deny-list exactly as before.
    """
    assert is_broadly_retryable(Exception("403 Forbidden")) is True
    assert is_broadly_retryable(ValueError("402 payment required")) is True


@pytest.mark.asyncio
async def test_production_403_consumes_one_attempt_with_no_backoff(no_real_sleep: AsyncMock) -> None:
    """The wall-clock fix: the production 403 costs ONE attempt, not four.

    Before this change the summarizer ran attempts 1-4 and slept the full
    1s/10s/30s ladder (~42s of dead wall clock, log 06:46:01-06:46:43) against a key
    that could not succeed. Asserting attempt count + zero sleeps rather than real
    elapsed time keeps the test deterministic.
    """
    awaitable = AsyncMock(side_effect=_api_error_with_status(403, _PROD_KEY_LIMIT_MESSAGE))

    with pytest.raises(litellm_exc.APIError):
        await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="asknews_summarizer")

    assert awaitable.await_count == 1
    no_real_sleep.assert_not_awaited()


def test_llm_status_code_reads_the_reported_status() -> None:
    """The shared numeric primitive returns the provider's own status, nothing else."""
    assert llm_status_code(_api_error_with_status(403, _PROD_KEY_LIMIT_MESSAGE)) == 403
    assert llm_status_code(_api_error_with_status(402)) == 402


def test_llm_status_code_is_none_outside_the_llm_exception_tree() -> None:
    """Only ``openai.APIError`` descendants can report a status.

    Two families depend on this returning ``None`` so their callers stay on text cues:
    ``requests``-shaped errors (status lives on ``.response``, and a CDN/WAF 403 on the
    Metaculus question-list endpoint must stay RETRYABLE per the 2026-05-19 incident),
    and the AskNews SDK, whose ``asknews_sdk.errors`` classes carry a ``.code`` and do
    not subclass ``openai.APIError`` at all.
    """
    cdn_403 = requests.HTTPError("403 Client Error")
    cdn_403.response = MagicMock(spec=requests.Response)
    cdn_403.response.status_code = 403

    # The SDK stores the response verbatim without reading it, so a spec'd stand-in is
    # enough to build the real exception type — what matters here is its class, not its
    # payload. Built with the genuine SDK class rather than a look-alike so the test
    # keeps passing only while AskNews really does stay outside the openai tree.
    asknews_403 = asknews_errors.ForbiddenError(
        MagicMock(spec=asknews_response.APIResponse), "Your subscription is not currently active", 403011
    )

    assert llm_status_code(cdn_403) is None
    assert llm_status_code(Exception("403 Forbidden")) is None
    assert llm_status_code(asknews_403) is None


def test_llm_status_code_rejects_a_non_int_status() -> None:
    """A string status is not numeric evidence — the narrowing keeps it out.

    litellm normally sets an int, but the attribute is untyped and a provider-shaped
    string ("403") must not be compared against the int frozenset, or membership
    silently fails open on a status that IS a hard rejection.
    """
    exc = _api_error_with_status(403)
    # setattr, not a direct assignment: the whole point is a value the annotation
    # forbids, so both type checkers would flag the plain form.
    setattr(exc, "status_code", "403")  # noqa: B010

    assert llm_status_code(exc) is None


@pytest.mark.asyncio
async def test_slow_403_is_not_rescued_by_the_zero_output_exemption() -> None:
    """A hard 403 can never reach the zero-output re-roll, even worded like one.

    The exemption is ANDed with the retry predicate, so making the status permanent
    closes the path outright — a defense against a provider that returns an
    unparseable body *and* a client-error status.
    """
    awaitable = AsyncMock(side_effect=_api_error_with_status(403, _PROD_WHITESPACE_BODY_MESSAGE))
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with patch("metaculus_bot.llm_retry.time.monotonic", clock), pytest.raises(litellm_exc.APIError):
        await invoke_with_broad_retry(awaitable, wall_timeout=_BIG_WALL, label="forecaster_binary")

    assert awaitable.await_count == 1


# ---------------------------------------------------------------------------
# OpenRouter overload (2026-10-01). OpenRouter returned "Our servers are currently
# overloaded" as an error object in the response body carrying a 5xx code. litellm
# maps any OpenRouter status outside its short list (400/401/404/408/422/429/503/504)
# to a BARE APIError, which the type-only transient check never retried: native_search
# and two gap-fill v1 searches died in ~2s on the first attempt.
# ---------------------------------------------------------------------------

_PROD_OVERLOADED_MESSAGE = "OpenrouterException - Our servers are currently overloaded. Please try again later."


@pytest.mark.parametrize("status", [500, 502, 520, 529])
def test_transient_check_retries_a_bare_server_error_status(status: int) -> None:
    assert _is_transient_type(_api_error_with_status(status, _PROD_OVERLOADED_MESSAGE)) is True


@pytest.mark.parametrize("status", [200, 400, 402, 403, 404, 422, 429])
def test_transient_check_still_rejects_a_bare_non_server_status(status: int) -> None:
    """429 stays out on purpose: rate limits go through key-swap / AskNews backoff, as for RateLimitError."""
    assert _is_transient_type(_api_error_with_status(status, _PROD_OVERLOADED_MESSAGE)) is False


def test_transient_check_keeps_zero_output_bodies_out() -> None:
    """A whitespace-body APIError arrives at 500 too; the transient sites must keep rejecting it, or the
    slow zero-output re-roll would start firing on the stacker and research budgets."""
    assert _is_transient_type(_api_error(_PROD_WHITESPACE_BODY_MESSAGE)) is False


@pytest.mark.asyncio
async def test_fast_overloaded_error_is_retried_then_succeeds() -> None:
    awaitable = AsyncMock(side_effect=[_api_error_with_status(502, _PROD_OVERLOADED_MESSAGE), "recovered"])

    result = await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="native_search")

    assert result == "recovered"
    assert awaitable.await_count == 2


@pytest.mark.asyncio
async def test_slow_overloaded_error_is_not_retried() -> None:
    """The elapsed gate still governs: widening the predicate adds no slow-failure retries."""
    awaitable = AsyncMock(side_effect=_api_error_with_status(502, _PROD_OVERLOADED_MESSAGE))
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with patch("metaculus_bot.llm_retry.time.monotonic", clock), pytest.raises(litellm_exc.APIError):
        await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="native_search")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_retry_warning_names_the_error_type_and_status(caplog: pytest.LogCaptureFixture) -> None:
    awaitable = AsyncMock(side_effect=[_api_error_with_status(502, _PROD_OVERLOADED_MESSAGE), "ok"])

    with caplog.at_level("WARNING", logger="metaculus_bot.llm_retry"):
        await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="native_search")

    (warning,) = [r.message for r in caplog.records if r.message.startswith("LLM_RETRY[native_search]")]
    assert "error=APIError status=502" in warning


@pytest.mark.asyncio
async def test_retry_result_type_follows_the_awaitable() -> None:
    """Generic over the awaited type, so the agentic loop can retry a raw completion object."""
    completion = {"choices": []}
    awaitable = AsyncMock(side_effect=[_timeout(), completion])

    result = await invoke_with_transient_retry(awaitable, wall_timeout=_BIG_WALL, label="gap_fill_v2_driver")

    assert result is completion


def test_llm_error_fields_reads_type_and_status() -> None:
    assert llm_error_fields(_api_error_with_status(502)) == "error=APIError status=502"
    assert llm_error_fields(_timeout()) == "error=Timeout status=408"


def test_llm_error_fields_reports_none_outside_the_llm_tree() -> None:
    """``None`` is the registry's null sentinel, so the status field always parses."""
    assert llm_error_fields(ValueError("bad")) == "error=ValueError status=None"
