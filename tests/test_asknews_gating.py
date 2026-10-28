import types
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest


def _make_q(text: str) -> MagicMock:
    """Build a minimal MetaculusQuestion-shaped mock for the new ResearchCallable
    contract. Tests only care about question_text on this path."""
    q = MagicMock()
    q.question_text = text
    return q


def _install_asknews_stub(monkeypatch: pytest.MonkeyPatch, on_search: Callable[[str], object]) -> None:
    """Install a minimal asknews_sdk stub that routes search calls to on_search.

    The stub emulates only what our provider needs: async context manager and
    a .news.search_news coroutine returning an object with .as_dicts.
    """

    module: Any = types.ModuleType("asknews_sdk")

    class _News:
        async def search_news(self, *, query, n_articles, return_type, strategy):  # type: ignore[no-untyped-def]
            import inspect

            result = on_search(strategy)
            if inspect.isawaitable(result):
                result = await result  # type: ignore[assignment]
            return result

    class AsyncAskNewsSDK:
        def __init__(self, *_, **__):  # type: ignore[no-untyped-def]
            self.news = _News()

        async def __aenter__(self):  # pragma: no cover - trivial
            return self

        async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
            return False

    module.AsyncAskNewsSDK = AsyncAskNewsSDK
    monkeypatch.setitem(__import__("sys").modules, "asknews_sdk", module)


@pytest.mark.asyncio
async def test_asknews_rate_gate_runs_before_both_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure AskNews is selected and globals are reset
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")
    monkeypatch.setenv("RESEARCH_PROVIDER", "asknews")

    # Capture rate-gate invocations and internal sleeps
    gate_calls: list[None] = []
    sleep_durations: list[float] = []

    async def fake_gate() -> None:
        gate_calls.append(None)

    async def fake_sleep(secs: float) -> None:
        sleep_durations.append(secs)

    # Install stubs
    _install_asknews_stub(
        monkeypatch,
        on_search=lambda strategy: types.SimpleNamespace(as_dicts=[]),
    )
    import metaculus_bot.research.providers as rp

    monkeypatch.setattr(rp, "_ASKNEWS_GLOBAL_SEMAPHORE", None, raising=False)
    monkeypatch.setattr(rp, "_ASKNEWS_LAST_CALL_TS", 0.0, raising=False)
    monkeypatch.setattr(rp, "_asknews_rate_gate", fake_gate, raising=True)
    monkeypatch.setattr(rp.asyncio, "sleep", fake_sleep, raising=True)

    provider, name = rp.choose_provider_with_name()
    assert name == "asknews"
    await provider(_make_q("Will X happen?"))

    # Two calls -> two gates; and we keep an explicit inter-call sleep
    assert len(gate_calls) == 2
    assert any(d > 0 for d in sleep_durations)


@pytest.mark.asyncio
async def test_global_semaphore_serializes_concurrent_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")

    in_flight = 0
    max_in_flight = 0

    async def _record(_strategy: str) -> object:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        # Yield to allow other tasks to run
        import asyncio as _a

        await _a.sleep(0)  # cooperative yield only
        in_flight -= 1
        return types.SimpleNamespace(as_dicts=[])

    _install_asknews_stub(monkeypatch, on_search=_record)
    import metaculus_bot.research.providers as rp

    # Force concurrency 1 and neutralize gating/sleeps to keep the test fast
    monkeypatch.setattr(rp, "ASKNEWS_MAX_CONCURRENCY", 1, raising=False)
    monkeypatch.setattr(rp, "_ASKNEWS_GLOBAL_SEMAPHORE", None, raising=False)

    async def noop_gate() -> None:  # pragma: no cover - trivial
        return None

    async def noop_sleep(_secs: float) -> None:  # pragma: no cover - trivial
        return None

    monkeypatch.setattr(rp, "_asknews_rate_gate", noop_gate, raising=True)
    monkeypatch.setattr(rp.asyncio, "sleep", noop_sleep, raising=True)

    provider, _ = rp.choose_provider_with_name()
    # Launch multiple requests concurrently; each request does 2 sequential calls
    await __import__("asyncio").gather(*(provider(_make_q(f"Q{i}")) for i in range(4)))

    # Entire attempt is under the global semaphore -> at most 1 in-flight search
    assert max_in_flight == 1


@pytest.mark.asyncio
async def test_rps_gate_sleeps_before_historical_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")
    monkeypatch.setenv("RESEARCH_PROVIDER", "asknews")

    # Return empty articles; only timing matters
    _install_asknews_stub(
        monkeypatch,
        on_search=lambda strategy: types.SimpleNamespace(as_dicts=[]),
    )
    import metaculus_bot.research.providers as rp

    # Use a low RPS so the second call must sleep
    monkeypatch.setattr(rp, "ASKNEWS_MAX_RPS", 0.5, raising=False)  # min interval = 2.0s
    monkeypatch.setattr(rp, "_ASKNEWS_GLOBAL_SEMAPHORE", None, raising=False)
    monkeypatch.setattr(rp, "_ASKNEWS_LAST_CALL_TS", 0.0, raising=False)

    # Controlled monotonic sequence: first gate no-sleep, second gate sleeps ~2s
    times = [100.0, 100.0, 102.0]

    def fake_monotonic() -> float:
        return times.pop(0) if times else 102.0

    sleep_calls: list[float] = []

    async def fake_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    monkeypatch.setattr(rp.time, "monotonic", fake_monotonic, raising=True)
    monkeypatch.setattr(rp.asyncio, "sleep", fake_sleep, raising=True)

    provider, _ = rp.choose_provider_with_name()
    await provider(_make_q("Test"))

    # We expect both the explicit inter-call sleep and ~2.0s RPS sleep
    assert any(s > 0 for s in sleep_calls)
    assert any(abs(s - 2.0) < 1e-6 for s in sleep_calls)


class TestRateGateLockLifecycle:
    """The RPS gate's lock is created lazily, matching the semaphore beside it.

    Both are process-wide asyncio primitives for the same provider, and
    ``get_asknews_semaphore`` already owned a lazy get-or-create; the lock did not.
    That asymmetry had no reason behind it, and the import-time shape is the one that
    can bind to a loop which later dies.

    Scope note, so a reader doesn't over-trust this class: the failure an import-time
    ``asyncio.Lock`` can cause — "is bound to a different event loop" after a lock is
    left HELD at loop close — is real and reproducible in isolation, but NOT reachable
    through this gate. Driving the real ``_asknews_rate_gate`` that way, a second
    ``asyncio.run`` succeeds, because the cancellation releases the lock before the
    loop closes and a released lock rebinds cleanly. So these tests pin the lifecycle
    that was made consistent, and deliberately do not claim to reproduce a bug.
    """

    def test_the_gate_works_across_two_event_loops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two real ``asyncio.run`` calls, with no reset of the lock between them.

        Passes both before and after the change — recorded as such rather than dressed
        up as a regression test. Its value is pinning that the lazy accessor did not
        BREAK cross-loop reuse, which a naive "recreate every call" implementation
        would have (each loop would get its own lock and the RPS gate would stop
        serializing).
        """
        import asyncio

        import metaculus_bot.research.providers as rp

        monkeypatch.setattr(rp, "_ASKNEWS_RATE_LOCK", None, raising=False)
        monkeypatch.setattr(rp, "_ASKNEWS_LAST_CALL_TS", 0.0, raising=False)
        monkeypatch.setattr(rp, "ASKNEWS_MAX_RPS", 1000.0, raising=False)

        async def _contend() -> None:
            await asyncio.gather(*(rp._asknews_rate_gate() for _ in range(4)))

        asyncio.run(_contend())
        first_lock = rp._ASKNEWS_RATE_LOCK
        assert first_lock is not None, "using the gate must have created the lock"

        asyncio.run(_contend())

        assert rp._ASKNEWS_RATE_LOCK is first_lock, (
            "the gate must keep ONE process-wide lock; a per-call lock would silently stop serializing AskNews calls"
        )

    def test_lock_is_not_created_at_import(self) -> None:
        """A fresh import must leave the lock unbound.

        The only test here that discriminates: reintroducing
        ``_ASKNEWS_RATE_LOCK = asyncio.Lock()`` at module scope fails this (verified by
        mutation) even though nothing contends.
        """
        import importlib

        import metaculus_bot.research.providers as rp

        reloaded = importlib.reload(rp)
        try:
            assert reloaded._ASKNEWS_RATE_LOCK is None, (
                "the RPS lock must be created lazily by _get_asknews_rate_lock, not at import"
            )
        finally:
            # Leave the module in its normal state for the rest of the session; other
            # tests monkeypatch attributes on this module object.
            importlib.reload(rp)
