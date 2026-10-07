"""Wall-clock hardening for the synchronous Metaculus publish path.

forecasting-tools makes the publish POSTs against
``https://www.metaculus.com/api/`` via blocking ``requests.post`` calls (see
``forecasting_tools/helpers/metaculus_client.py``). The three prediction POSTs
all funnel through one shared private helper; the comment POST is standalone:

- ``post_binary_question_prediction`` / ``post_numeric_question_prediction`` /
  ``post_multiple_choice_question_prediction`` -> ``_post_question_prediction`` -> ``requests.post``
- ``post_question_comment`` -> ``requests.post``

If the Metaculus API hangs mid-tournament, those calls block the asyncio event
loop (they're invoked synchronously from inside the ``async def
publish_report_to_metaculus`` methods on each report type) and block every other
Q in the batch from publishing. ``apply_publish_hardening()`` monkey-patches the
shared prediction helper ``_post_question_prediction`` (which covers all three
prediction types) and ``post_question_comment`` at startup with two layers of
defense:

1. Request-side socket timeout (primary): for the duration of the wrapped call,
   ``requests.post`` on the metaculus_client module is patched to set
   ``timeout=PUBLISH_POST_TIMEOUT``. This makes the underlying socket actually
   close when the server stalls, so the worker thread terminates instead of
   leaking.

2. ``concurrent.futures.Future.result(timeout=...)`` cap (belt-and-suspenders):
   covers pathological cases where a request might somehow ignore the socket
   timeout (e.g. unbounded DNS resolution before connect). Note that
   ``Future.cancel()`` does NOT interrupt a running thread; without layer (1)
   the worker would keep running until socket close, risking duplicate publishes
   on retry. Because we patch the shared private helper and unwrap its upstream
   ``@retry_with_exponential_backoff`` (below), there is no inner retry left to
   catch the socket-close ``Timeout`` and sleep through it, so an abandoned
   worker dies when its socket closes at ``PUBLISH_POST_TIMEOUT`` rather than
   lingering for minutes. Layer (1) makes the abandoned-worker scenario
   unreachable.

Each wrapper retries once on timeout / connection error.

**Why patch the private helper, not the public prediction wrappers.** The three
public ``post_*_question_prediction`` methods are *undecorated* wrappers that do
synchronous input validation (bounds/monotonicity checks that raise
``ValueError``) and then delegate to the ``@retry_with_exponential_backoff()``-
decorated ``_post_question_prediction``. Patching the public wrappers (as an
earlier version did) left that inner upstream retry in place — a second retry
loop *beneath* ours. On a stall the outer ``Future.result`` cap fired at 20s but
the abandoned worker kept running for 2-4 min inside the inner retry (catch
socket-close ``Timeout`` -> ``sleep(min(delay*jitter, 75s))`` -> retry), which
(a) widened the window in which overlapping publishes leaked the process-global
patched ``requests.post`` (that per-call save/restore is gone — see
``_install_post_timeout_override``) and
(b) saturated the publish thread pool during a sustained API stall. Patching
``_post_question_prediction`` and unwrapping its decorator collapses the
prediction path to a single retry layer; the public wrappers keep their
validation on the caller thread and delegate into the hardened helper. (The one
non-idempotent publish, ``post_question_comment``, was already single-layer and
stays wrapped exactly as before.)

**0.2.92 architecture.** Publishing moved from the (now deprecated)
``MetaculusApi`` classmethod shim onto ``MetaculusClient``. ``ForecastBot``
publishes through an *instance* — ``report.publish_report_to_metaculus`` calls
``metaculus_client.post_*(...)`` on ``self.metaculus_client = MetaculusClient()``.
Patching the deprecated ``MetaculusApi`` shim would test green and silently
no-op in prod, so we patch ``MetaculusClient`` at the class level (plain instance
methods): instance method lookup resolves through the class, so a class-level
patch covers every instance the bot builds.

Two 0.2.92 behaviors shape the two layers:

- ``MetaculusClient`` now passes ``timeout=self.timeout`` (default 30s) on the
  POST, so a naive ``setdefault`` timeout injection would be a no-op. Layer (1)
  therefore **overrides** the timeout to the tighter ``PUBLISH_POST_TIMEOUT``
  (20s) for the duration of the wrapped call — this aligns the worker-side socket
  close with the caller-side ``Future.result`` cap (both 20s) so the worker
  thread dies at the cap instead of lingering to the upstream 30s. Overriding is
  safe here: the only caller is ``MetaculusClient`` itself (small publish
  payloads), and a tighter publish ceiling is exactly the intent.
- Both patched methods (``_post_question_prediction`` and
  ``post_question_comment``) are ``@retry_with_exponential_backoff()`` decorated
  upstream. We wrap **beneath** that decorator (via ``__wrapped__``) for both, so
  ours is the single retry layer on each path. Total wall-clock is bounded by the
  ``Future.result`` cap regardless, and duplicate-publish risk on retry is
  inherent to any retry-on-timeout (a re-POST is an idempotent overwrite for
  predictions).

We use ``concurrent.futures.ThreadPoolExecutor`` (rather than asyncio.to_thread)
because the patched callsite remains synchronous — calling code is
``metaculus_client.post_*(...)`` without await — so we can't return a coroutine.

The wrappers are attached as plain functions (instance methods) so both
class-level and instance-level calls preserve the original ``self``-first calling
convention (0.2.92 uses instance methods, not classmethods).

**Layer 3: get the whole publish off the event loop
(``apply_report_publish_offload``).** Layers 1 and 2 bound how long a POST can
take; neither stops it from freezing everything else. The caller-side
``future.result(timeout=...)`` is a *synchronous* block, and its only caller is
ft's ``async def publish_report_to_metaculus``, whose body issues both POSTs with
no await and no ``to_thread`` (verified on 0.2.92 for all three report types:
zero ``await`` expressions in those bodies). So the event loop is pinned for the
full publish — measured at 2.01s for a 2s stubbed publish, with an
``asyncio.sleep(0.01)`` heartbeat getting 4 ticks instead of ~200.

That matters because ft's ``forecast_questions`` runs every question of a batch
under one ``asyncio.gather``. While the loop is pinned, no sibling question's
tasks can run, but their wall-clock deadlines keep advancing
(``_forecaster_with_soft_deadline``, ``PER_QUESTION_WALL_CLOCK_DEADLINE``,
``GAP_FILL_V2_WALL_DEADLINE``, ``PREDICTION_MARKET_TIMEOUT``). A forecaster near
its soft deadline is then cancelled on time it never got to use and recorded as
``DROP_CAUSE_TIMEOUT_SOFT_DEADLINE`` — a misattributed drop that degrades the
ensemble, and with ``MIN_FORECASTERS_TO_PUBLISH=1`` can shrink it to one. Real
cost per question is two ``_sleep_between_requests`` calls (``time.sleep``, 3.5-4.5s
each) plus network, and up to ``PUBLISH_POST_TIMEOUT * (PUBLISH_POST_RETRIES+1)``
per POST if Metaculus stalls.

The fix wraps the *async* seam — ``publish_report_to_metaculus`` on each report
class — with ``await asyncio.to_thread(...)``, so both POSTs and the blocking
wait ride a worker thread and the loop stays free. Patching there rather than
inside the sync wrapper is what makes it possible at all: the sync wrapper cannot
return a coroutine (see above), whereas this method already is one.

**Layer 4: the pre-publish close-time gate.** That same async seam is the last
point that still knows WHICH question is being published, so it is where
``publish_gate.skip_publish_if_closed`` runs. A question that closed mid-run
cannot accept a forecast, and POSTing anyway earns a 405 that propagates out of
ft's per-question handler and (via ``cli.main``'s ``log_report_summary``) takes
the end-of-run alertable summary with it — the q45085 shape. See
``metaculus_bot/publish_gate.py`` for why the gate carries no safety margin and
why it skips the comment POST too.

Idempotent: calling ``apply_publish_hardening()`` more than once is a no-op
(checked via a sentinel attribute on ``MetaculusClient``). It applies all four
layers, so there is one entry point for publish hardening.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
import threading
from collections.abc import Callable
from typing import Any

import requests
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
from forecasting_tools.data_models.numeric_report import NumericReport
from forecasting_tools.helpers import metaculus_client as _ft_metaculus_client
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from metaculus_bot.constants import PUBLISH_POST_RETRIES, PUBLISH_POST_TIMEOUT, QUESTION_PLATFORM_HOSTS
from metaculus_bot.http_status import http_status_from_exception
from metaculus_bot.publish_gate import skip_publish_if_closed
from metaculus_bot.run_status import RUN_STATUS, safe_error_type_from_exception

assert PUBLISH_POST_RETRIES >= 0, "PUBLISH_POST_RETRIES must be non-negative"

logger = logging.getLogger(__name__)

_SENTINEL = "_publish_hardening_applied"

# Layer 3's seam: the async publish entry point each report class defines, and the
# report classes the bot actually publishes through (one per forecastable question
# type, via DataOrganizer.get_report_type_for_question_type — pinned in
# tests/test_publish_hardening_concurrency.py so a new question type can't quietly
# publish unoffloaded). ConditionalReport is excluded deliberately: the bot never
# builds one, and its publish just delegates to the yes/no reports, which are
# BinaryReports already covered here.
_PUBLISH_METHOD = "publish_report_to_metaculus"
_PATCHED_REPORT_TYPES: tuple[type[ForecastReport], ...] = (
    BinaryReport,
    NumericReport,
    MultipleChoiceReport,
)
# Per-function marker, not a class attribute: the report classes inherit from a common
# base, so a class-level sentinel on one would read as set on its siblings and skip
# their patch.
_REPORT_SENTINEL = "_publish_offload_applied"

# Method names to patch. Both are @retry_with_exponential_backoff()-decorated
# instance methods on MetaculusClient that each wrap a single requests.post; we
# install our wrapper *beneath* the upstream decorator (via __wrapped__) so ours
# is the single retry layer. ``_post_question_prediction`` is the shared private
# helper that all three public ``post_*_question_prediction`` wrappers delegate
# to, so patching it hardens every prediction type at once (the public wrappers
# keep their synchronous input validation on the caller thread). We deliberately
# do NOT patch the public prediction wrappers: they're undecorated, so wrapping
# them would leave the inner upstream retry on ``_post_question_prediction`` in
# place, stacking two retry loops on the prediction path (see module docstring).
_PATCHED_METHODS: tuple[str, ...] = (
    "_post_question_prediction",
    "post_question_comment",
)

# Single shared executor across both wrappers. Publish calls are infrequent
# and serialized within a single Q's publish_report_to_metaculus().
_executor: concurrent.futures.ThreadPoolExecutor | None = None

# Per-run count of publish attempts that exhausted the retry budget — the counter
# that makes a publish-ATTEMPT failure visible. ``questions_failed_to_publish``
# increments only under the min-forecasters floor ("too thin to attempt"), so
# before this counter a 405/500/exhausted-timeout out of the actual POST left
# every counter at zero (q45085, 2026-08-03: two failed attempts, counters all
# zero). Module-scoped like ``prediction_market._SOURCE_LOSSES`` because the
# wrapper has no handle back to the bot; ``forecast_questions`` resets it at run
# start. Telemetry only — the exception still propagates exactly as before.
_PUBLISH_ATTEMPT_FAILURES: int = 0

# Layer 3 runs each report's publish on its own worker thread, and ft's
# asyncio.gather can have several questions publishing at once, so two exhausted
# publishes really can reach the increment concurrently. ``+=`` on a module global
# is interruptible between bytecodes, so an unlocked pair could lose one — the
# undercount that would let a two-question publish outage read as one. Same
# reasoning as ``fallback_openrouter.record_donated_key_fallback``'s accounting
# block; the lock is held for one addition and never around I/O.
_COUNTER_LOCK = threading.Lock()

# 4xx statuses worth a second attempt. Every other 4xx is a permanent verdict on
# THIS request — a 405 "already closed to forecasting", a 401 on a bad token, a 400
# malformed payload — so retrying can only burn wall clock inside the per-question
# budget and delay every sibling question's publish. q45085 spent its second
# attempt 4s later on a 405 that could not have succeeded. 5xx and transport-level
# failures (socket timeout, connection reset) still retry exactly as before.
_RETRYABLE_4XX: frozenset[int] = frozenset({408, 429})

# The __cause__ walk + "Status code: NNN" fallback that recovers the status ft's
# message-only HTTPError re-raise hides now lives in metaculus_bot.http_status
# (shared with fetch_hardening, whose retry policy is deliberately different).


def _bump_publish_attempt_failure() -> None:
    global _PUBLISH_ATTEMPT_FAILURES
    with _COUNTER_LOCK:
        _PUBLISH_ATTEMPT_FAILURES += 1


def publish_attempt_failures() -> int:
    """Per-run count of retry-exhausted publish attempts (folded into alertable_count)."""
    return _PUBLISH_ATTEMPT_FAILURES


def reset_publish_attempt_failures() -> None:
    """Zero the counter at run start; without this it leaks across runs/tests sharing a process."""
    global _PUBLISH_ATTEMPT_FAILURES
    _PUBLISH_ATTEMPT_FAILURES = 0


def _is_retryable(exc: BaseException) -> bool:
    """False only for a 4xx that a second identical POST cannot fix.

    Unknown status (a bare connection error, a timeout) means retry: the whole
    point of the retry budget is transport flakiness, and refusing to retry
    something we couldn't classify would be a regression. Deliberately the
    INVERSE default of ``fetch_hardening``'s policy (which allow-lists retryable
    statuses): a lost publish forfeits the question, a lost fetch retries next
    run. The status read is shared (``metaculus_bot.http_status``); only the
    policies differ.
    """
    status = http_status_from_exception(exc)
    if status is None or not 400 <= status < 500:
        return True
    return status in _RETRYABLE_4XX


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="publish-hardening")
    return _executor


def _install_post_timeout_override(timeout_s: float) -> None:
    """Patch ``metaculus_client.requests.post`` ONCE to force ``timeout=timeout_s``.

    Overrides (not setdefault) the timeout: 0.2.92's ``MetaculusClient`` always
    passes ``timeout=self.timeout`` (30s), so a setdefault would never fire. We
    force our tighter publish ceiling so the socket closes in step with the
    caller-side ``Future.result`` cap.

    **Installed once and left in place**, matching ``fetch_hardening``'s GET twin
    (``_install_get_timeout_default``, which chose this shape for the same
    reason). This was previously a per-call context manager that saved and
    restored the module global, which is only correct under strict LIFO nesting —
    and the timeout-and-retry path violates that by construction: ``future.cancel()``
    returns False on a running future, so a timed-out orphan and its retry are
    both alive in the shared pool, both inside the context manager. Reproduced
    with two overlapping entries where the outer exits first: it restores the
    ORIGINAL post, so the inner's wrapper stays installed permanently, one layer
    deeper per occurrence, with no way back. The class-patch sentinel didn't help
    — it guards the one-time method patch, not this per-call one.

    **Scoped to the question-platform hosts** (``QUESTION_PLATFORM_HOSTS``: Metaculus and
    Mantic), which the per-call version got for free and a permanent install does
    not: ``metaculus_client.requests`` IS the global ``requests`` module (verified —
    ``mc.requests is requests``), so an unconditional forced timeout would also
    re-time every OTHER POST in the process. Real POST callers share it:
    ``exa_py.api``, litellm's Databricks path, huggingface_hub, streamlit. A 20s
    publish ceiling is right for a small publish payload and wrong for a long
    research call, and *lowering* someone else's timeout can only manufacture
    failures. So the host check is what makes the install-once shape safe, not
    decoration. Mantic is inside the scope because its publishes go through the
    same client: left unbounded, a stalled Mantic POST would be abandoned by the
    caller-side ``Future.result`` cap while its worker thread ran on, which is the
    duplicate-publish shape layer 1 exists to prevent.

    Idempotent by construction: the only caller is ``apply_publish_hardening``,
    which is itself sentinel-guarded, so no double-wrapping.
    """
    original_post = _ft_metaculus_client.requests.post

    @functools.wraps(original_post)
    def post_with_timeout(*args: Any, **kwargs: Any) -> Any:
        # ft always passes the URL positionally (metaculus_client.py's four publish
        # POSTs), but read the kwarg too so a future keyword call still scopes right.
        url = args[0] if args else kwargs.get("url", "")
        if isinstance(url, bytes):
            url = url.decode("utf-8", errors="replace")
        # Matched on the host substring alone, so a METACULUS_API_BASE_URL override with a
        # different path still hits and every platform the client publishes to is inside the scope.
        if isinstance(url, str) and any(host in url for host in QUESTION_PLATFORM_HOSTS):
            kwargs["timeout"] = timeout_s
        return original_post(*args, **kwargs)

    _ft_metaculus_client.requests.post = post_with_timeout


def _wrap_with_timeout_retry(method_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    """Return a sync wrapper that runs ``original`` on a worker thread with timeout + retry.

    ``original`` is the undecorated MetaculusClient instance method, so the
    wrapper is installed as a plain function (an instance method) and forwards
    ``self`` through ``*args`` transparently. The wrapper layers two timeout
    mechanisms:
    - Request-side: ``requests.post`` on metaculus_client is patched once at
      ``apply_publish_hardening`` time to force ``timeout=PUBLISH_POST_TIMEOUT``,
      bounding the underlying socket (see ``_install_post_timeout_override`` for
      why that install is global rather than per-call).
    - Caller-side: ``Future.result(timeout=...)`` provides a final ceiling.
    """

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        publish_kind = "comments" if "comment" in method_name.lower() else "forecasts"
        RUN_STATUS.record_publish_attempt(publish_kind)
        executor = _get_executor()
        attempts = PUBLISH_POST_RETRIES + 1  # read at call time so tests' monkeypatch is honored

        last_exc: BaseException = RuntimeError(f"PUBLISH_HARDENING: {method_name} loop exited without running")
        for attempt in range(1, attempts + 1):
            future = executor.submit(original, *args, **kwargs)
            try:
                result = future.result(timeout=PUBLISH_POST_TIMEOUT)
                RUN_STATUS.record_publish_success(publish_kind)
                return result
            except concurrent.futures.TimeoutError as exc:
                last_exc = exc
                future.cancel()
                logger.warning(
                    "PUBLISH_HARDENING: %s attempt %d/%d timed out after %ds",
                    method_name,
                    attempt,
                    attempts,
                    PUBLISH_POST_TIMEOUT,
                )
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "PUBLISH_HARDENING: %s attempt %d/%d failed (%s: %s)",
                    method_name,
                    attempt,
                    attempts,
                    type(exc).__name__,
                    exc,
                )
                if not _is_retryable(exc):
                    if attempt < attempts:
                        # Deliberately a second line rather than extra text on the one
                        # above: the ``publish_hardening`` MarkerSpec is anchored on the
                        # "attempt N/M failed (Type: msg)" shape, and appending to it
                        # would stop every publish failure from harvesting. This line
                        # carries no "attempt N/M" clause, so the spec ignores it. Only
                        # emitted when a retry was actually forgone — on the last
                        # attempt "not retrying" would be a tautology.
                        logger.warning(
                            "PUBLISH_HARDENING: %s not retrying status %s — a second identical POST cannot succeed",
                            method_name,
                            http_status_from_exception(exc),
                        )
                    break
        # The single terminal failure point on the publish path: every retry burned
        # and the question's POST never landed. Counted here (and only here) so the
        # end-of-run counters see it; the raise is unchanged.
        _bump_publish_attempt_failure()
        RUN_STATUS.record_publish_failure(
            publish_kind,
            error_type=safe_error_type_from_exception(last_exc),
            http_status=http_status_from_exception(last_exc),
        )
        raise last_exc

    return wrapper


def _wrap_publish_off_the_loop(original: Callable[..., Any]) -> Callable[..., Any]:
    """Return an async ``publish_report_to_metaculus`` that gates, then runs ``original`` off the loop.

    ``original`` is ft's own coroutine function, whose body issues both POSTs
    synchronously. ``asyncio.to_thread(asyncio.run, coro)`` drives that coroutine to
    completion on a worker thread, so the calling loop keeps servicing every other
    question in ft's ``asyncio.gather`` while the publish blocks.

    A fresh loop per publish is fine here precisely because the body never awaits
    anything: it touches no object bound to the caller's loop, so there is nothing to
    share across the boundary. That is checked by
    ``tests/test_ft_upgrade_seams.py`` — if a future ft version adds a real ``await``
    to these bodies, that pin fails and this wrapper needs revisiting.

    Layer 4 rides here too: this is the last seam that still knows which question is
    being published, so the close-time gate runs before the offload (on the loop
    thread, which is what keeps its counter lock-free). ft's own return value is
    ``None`` for all three report types, so a skip returns ``None`` and is
    indistinguishable to the caller from a completed publish — by design, since ft
    discards it.
    """

    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # args[0] is the report instance (these are plain instance methods). Read the
        # question off it with a default because the seam belongs to a third-party
        # class: if a future ft report stops carrying one, the gate must fail OPEN and
        # publish, never silently withhold.
        report = args[0] if args else None
        if skip_publish_if_closed(getattr(report, "question", None)):
            return None
        return await asyncio.to_thread(lambda: asyncio.run(original(*args, **kwargs)))

    return wrapper


def apply_report_publish_offload() -> None:
    """Move each report type's async publish onto a worker thread. Idempotent.

    Layer 3 (see module docstring): the timeout/retry layers bound how LONG a publish
    takes; this one stops it from freezing every sibling question while it runs.
    """
    for report_type in _PATCHED_REPORT_TYPES:
        raw = report_type.__dict__.get(_PUBLISH_METHOD)
        if raw is None:
            raise AttributeError(
                f"PUBLISH_HARDENING: {report_type.__name__} defines no {_PUBLISH_METHOD!r} to patch. "
                "The forecasting-tools publish seam moved or was renamed; repoint "
                "_PATCHED_REPORT_TYPES (see tests/test_ft_upgrade_seams.py)."
            )
        if getattr(raw, _REPORT_SENTINEL, False):
            continue
        wrapped = _wrap_publish_off_the_loop(raw)
        setattr(wrapped, _REPORT_SENTINEL, True)
        setattr(report_type, _PUBLISH_METHOD, wrapped)


def apply_publish_hardening() -> None:
    """Patch the publish path: forced socket timeout, timeout + retry, and loop offload. Idempotent."""
    if getattr(MetaculusClient, _SENTINEL, False):
        return

    # Layer 1: force the tighter publish timeout on metaculus_client's requests.post.
    # Installed once here rather than per-call — see _install_post_timeout_override.
    _install_post_timeout_override(PUBLISH_POST_TIMEOUT)

    # Layer 3: get the (synchronous-bodied) async publish off the event loop, so a
    # publish can't starve the sibling questions sharing ft's asyncio.gather.
    apply_report_publish_offload()

    for method_name in _PATCHED_METHODS:
        # These are plain instance methods on MetaculusClient (not classmethods),
        # so we install a plain function that Python binds as an instance method.
        # Wrap the *undecorated* original (``__wrapped__``) so we sit beneath the
        # upstream @retry_with_exponential_backoff on both patched methods rather
        # than stacking two retry loops. Fail fast (rather than silently skip) if
        # the seam moved: an ft rename would otherwise leave publishes unhardened.
        raw = MetaculusClient.__dict__.get(method_name)
        if raw is None:
            raise AttributeError(
                f"PUBLISH_HARDENING: MetaculusClient defines no {method_name!r} to patch. "
                "The forecasting-tools publish seam moved or was renamed; repoint "
                "_PATCHED_METHODS (see tests/test_ft_upgrade_seams.py::TestPublishHardeningWrapsRealPublishPath)."
            )
        original_func = getattr(raw, "__wrapped__", raw)
        wrapped = _wrap_with_timeout_retry(method_name, original_func)
        setattr(MetaculusClient, method_name, wrapped)

    setattr(MetaculusClient, _SENTINEL, True)
    logger.info(
        "Publish hardening applied: %d MetaculusClient.post_* methods wrapped with %ds timeout + %d retry",
        len(_PATCHED_METHODS),
        PUBLISH_POST_TIMEOUT,
        PUBLISH_POST_RETRIES,
    )
