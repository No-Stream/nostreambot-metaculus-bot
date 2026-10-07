import socket
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from curl_cffi.requests import AsyncSession as CurlAsyncSession
from curl_cffi.requests import Session as CurlSession
from forecasting_tools import BinaryQuestion, GeneralLlm, MultipleChoiceQuestion, NumericQuestion
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

# `playwright._impl._browser_type` is a PRIVATE module path, imported against an unpinned
# `playwright>=1.54.0`; if a version bump breaks this line, the private path moved, and the
# checkout is fine. See `_block_native_egress` for why the guard needs this class specifically.
from playwright._impl._browser_type import BrowserType as PlaywrightBrowserType

from metaculus_bot.publish_gate import reset_publish_skipped_closed
from metaculus_bot.publish_hardening import reset_publish_attempt_failures
from metaculus_bot.research import page_digest
from metaculus_bot.research.degradation_views import reset_run_degradation_counters
from metaculus_bot.research.fetch_ladder import run_cache
from scripts import gha_artifacts

_OPEN = datetime(2026, 1, 1)
_RESOLVE = datetime(2026, 5, 1)


@pytest.fixture(autouse=True)
def _clear_fetch_ladder_run_cache() -> Iterator[None]:
    """Give every test a fresh process-run cache while preserving reuse inside one test."""
    run_cache.clear()
    yield
    run_cache.clear()


@pytest.fixture(autouse=True)
def _stub_page_digest_provider(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise real digest fallback by default; digest tests override this provider boundary."""
    if request.node.get_closest_marker("live") is not None:
        return
    client = MagicMock(invoke=AsyncMock(side_effect=TimeoutError("offline extractor double")))
    monkeypatch.setattr(page_digest, "build_llm_with_openrouter_fallback", MagicMock(return_value=client))


# ---------------------------------------------------------------------------
# Network-egress guard (money-safety backstop)
# ---------------------------------------------------------------------------

# Hosts a socket may connect to without tripping the guard. Loopback only —
# everything else is a real host and must be stubbed by the test. AF_UNIX and
# socket.socketpair() are allowed unconditionally (they never carry an INET
# address); asyncio's self-pipe / event-loop internals rely on them, so blocking
# them would wedge the whole suite.
_ALLOWED_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"})  # noqa: S104  # allowlist of loopback hosts for the egress guard, not a bind address


def _host_from_address(address: Any) -> str | None:
    """Pull the host out of a connect() address (INET = (host, port), INET6 adds flow/scope)."""
    if isinstance(address, tuple) and address:
        return address[0]
    return None


def _address_is_blocked(family: int, address: Any) -> bool:
    """True iff this is an AF_INET/AF_INET6 connect to a non-loopback host."""
    if family not in (socket.AF_INET, socket.AF_INET6):
        # AF_UNIX and everything else (socketpair, unix domain sockets) is fine.
        return False
    host = _host_from_address(address)
    if host is None:
        return False
    return host not in _ALLOWED_HOSTS


def _egress_guard_exempt(request: pytest.FixtureRequest) -> bool:
    """True when the test carries one of the two markers that opt it out of every egress guard.

    Shared by the socket guard and the native-egress guard below so a test cannot be exempt from
    one transport and blocked on another; the markers' semantics are documented on
    :func:`_block_network_egress`.
    """
    return (
        request.node.get_closest_marker("allow_network") is not None
        or request.node.get_closest_marker("live") is not None
    )


@pytest.fixture(autouse=True)
def _block_network_egress(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Block all real network egress during the test suite (money-safety backstop).

    Monkeypatches ``socket.socket.connect`` / ``connect_ex`` to raise a clear
    ``RuntimeError`` for any AF_INET/AF_INET6 connect to a non-localhost host, so
    no test can silently reach a paid API. Modeled on how ``pytest-socket`` gates
    (localhost + AF_UNIX + socketpair allowed, real hosts blocked) without taking
    the dependency. It sees only Python's own sockets; the two transports in this
    venv that open theirs elsewhere (a Chromium subprocess, libcurl) are refused
    by :func:`_block_native_egress`.

    Two independent markers skip the guard:

    - ``@pytest.mark.allow_network`` — a normally-selected test that must reach a
      real host (belt-and-suspenders escape hatch).
    - ``@pytest.mark.live`` — the live suite (e.g. ``tests/test_smoke_real_llm.py``),
      which makes real API calls by design and carries no ``allow_network`` marker.

    Deselection and exemption are separate concerns. ``addopts = -m 'not live'``
    DESELECTS the live suite so a plain ``make test`` never runs it (and it never
    reaches this guard). ``make test_live`` (``pytest -m live``) re-selects it, at
    which point this exemption is what keeps the guard from blocking the real API
    calls those tests exist to make. So the ``live`` exemption is load-bearing for
    ``make test_live``, not merely belt-and-suspenders.

    See also ``metaculus_bot.ablation.offline_replay.no_network()`` — a scoped
    context manager (ablation replay only) that blocks ``socket.getaddrinfo`` at
    the DNS level; this autouse guard is complementary, blocking ``connect`` /
    ``connect_ex`` at the socket level so it also catches literal-IP connects that
    skip DNS resolution entirely. Different scopes on purpose; don't consolidate.
    """
    if _egress_guard_exempt(request):
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        if _address_is_blocked(self.family, address):
            raise RuntimeError(
                f"Network access blocked in tests: {address}. "
                "Stub the client or mark the test @pytest.mark.allow_network."
            )
        return real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        if _address_is_blocked(self.family, address):
            raise RuntimeError(
                f"Network access blocked in tests: {address}. "
                "Stub the client or mark the test @pytest.mark.allow_network."
            )
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


# ---------------------------------------------------------------------------
# Native-egress guard: the two transports the socket guard cannot see
# ---------------------------------------------------------------------------

_PLAYWRIGHT_BROWSER_ENTRY_POINTS: tuple[str, ...] = (
    "launch",
    "launch_persistent_context",
    "connect",
    "connect_over_cdp",
)


@pytest.fixture
def native_egress_attempts() -> list[str]:
    """The browser launches and libcurl requests :func:`_block_native_egress` refused during one test.

    Requested by name only by the tests that exercise the guard itself (``tests/test_egress_guards.py``):
    they assert the refusal was recorded, then clear it so the guard's teardown check does not fail
    them for the trip they deliberately caused. Every other test gets the list implicitly through
    the autouse guard and never touches it.
    """
    return []


@pytest.fixture(autouse=True)
def _block_native_egress(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, native_egress_attempts: list[str]
) -> Iterator[None]:
    """Refuse the two egress paths that never pass through ``socket.socket.connect`` (money-safety backstop).

    :func:`_block_network_egress` patches Python's socket, which every pure-Python client (aiohttp,
    httpx, requests, the OpenAI and Google SDKs) goes through. Two transports in this venv do not:

    - **Headless Chromium**, launched through Playwright by ``metaculus_bot/research/rendered_fetch.py``
      for both the Tier-1 resolution-source rung and gap-fill v2's ``fetch`` tool. The browser is a
      subprocess with its own network stack, so the socket patch is invisible to it. Verified
      2026-09-03: three resolution-source marker tests launched a real browser the moment the Tier-1
      rung landed, and connected to a real host from a unit test.
    - **libcurl**, entered from Python through ``curl_cffi``, driven by two production callers now:
      the TLS-impersonation transport in ``metaculus_bot/research/impersonated_fetch.py`` (the
      Tier-1 resolution-source rung and gap-fill v2's ``fetch`` tool both dial it, so this is the
      most important reason the guard must stay armed), and the client ``yfinance`` drives for
      every ticker fetch in ``metaculus_bot/research/financial_data.py`` and ``ts_fetch.py``. The C
      library opens its own sockets. ``curl_cffi`` is a runtime dependency (pyproject
      ``[project] dependencies``) since 2026-09-04, when the impersonation rung landed; the
      module-scope import at the top of this file is now hard for that reason rather than because
      it used to arrive only as yfinance's transitive dependency.

    Chokepoints, chosen so every caller trips them whichever public API it holds.
    ``playwright._impl._browser_type.BrowserType`` is the one class both ``playwright.async_api`` and
    ``playwright.sync_api`` delegate to, and its four entry points (``launch``,
    ``launch_persistent_context``, ``connect``, ``connect_over_cdp``) are the only ways to obtain a
    browser. ``curl_cffi.requests.Session.request`` / ``AsyncSession.request`` are where every verb
    helper lands (``get``, ``post``, ``stream``, the module-level ``curl_cffi.requests.get``), and
    ``yfinance`` reaches libcurl only through ``Session.get``.

    Each refusal raises a ``RuntimeError`` naming this fixture AND is recorded in
    ``native_egress_attempts``; the fixture fails the test at teardown if anything was recorded. The
    record is what makes a trip loud: ``render_page`` soft-fails any exception out of the browser into
    its ``None`` "declined" signal and ``_render_yfinance_block`` logs and returns ``""``, so without
    it a refused launch would be byte-identical to the renderer being absent and the leaking test
    would stay green.

    Deliberately not covered: Playwright's node driver, which ``async_playwright()`` spawns before any
    launch (a local pipe-connected process with no egress of its own), and curl_cffi's raw
    ``Curl.perform`` and websocket paths, which nothing in this venv's callers reaches. The same two
    markers exempt a test as for the socket guard.
    """
    if _egress_guard_exempt(request):
        yield
        return

    def _refuse(attempt: str) -> None:
        native_egress_attempts.append(attempt)
        raise RuntimeError(
            f"Native network egress blocked in tests by _block_native_egress: {attempt}. "
            "Stub the transport or mark the test @pytest.mark.allow_network."
        )

    def _browser_guard(entry_point: str):
        async def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            del self, args, kwargs
            _refuse(f"playwright BrowserType.{entry_point}")

        return guarded

    def guarded_curl_request(self: Any, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        del self, args, kwargs
        _refuse(f"curl_cffi {method} {url}")

    async def guarded_async_curl_request(self: Any, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        del self, args, kwargs
        _refuse(f"curl_cffi {method} {url}")

    for entry_point in _PLAYWRIGHT_BROWSER_ENTRY_POINTS:
        monkeypatch.setattr(PlaywrightBrowserType, entry_point, _browser_guard(entry_point))
    monkeypatch.setattr(CurlSession, "request", guarded_curl_request)
    monkeypatch.setattr(CurlAsyncSession, "request", guarded_async_curl_request)

    yield

    if native_egress_attempts:
        pytest.fail(
            "The test attempted native network egress, which _block_native_egress refused and the code "
            f"under test then swallowed: {native_egress_attempts}. Stub the transport."
        )


# ---------------------------------------------------------------------------
# Persisted-artifact-store guard (data-safety backstop)


@pytest.fixture(autouse=True)
def _redirect_artifact_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the persisted artifact store at a temp dir for every test.

    ``backtests/gha_artifact_store/`` is the durable local copy of GHA artifacts that
    expire at 90 days, so a test writing into it plants fake data in real research
    evidence. Not hypothetical: a run-log test that simply omitted ``store_dir`` persisted
    a fixture artifact named ``research-2`` there, and the next real offline harvest
    ingested its one-line log into the live telemetry archive as an ``unknown`` run.

    ``scripts.gha_artifacts._resolve_store_dir`` reads this module attribute at CALL time
    precisely so this redirect works; a signature default would be captured at import and
    leave the omission unprotected. Tests that pass ``store_dir`` explicitly are
    unaffected.
    """
    # Numbered mktemp scans every sibling on every test; let the store create a unique path only when used.
    store_dir = tmp_path_factory.getbasetemp() / "gha_artifact_store" / uuid4().hex
    monkeypatch.setattr(gha_artifacts, "DEFAULT_STORE_DIR", str(store_dir))


# Shared failure fixtures
# ---------------------------------------------------------------------------

# The verbatim OpenRouter response that cost the 2026-07-26 tournament run two of three
# forecasters and most of the research stack. Copied character-for-character from the run
# log (the donated key at $0.00 of its $850 cap): HTTP 403 rather than the 402 OpenRouter's
# docs promise, the phrase "Key limit exceeded (total limit)", and a ``"code":403`` field in
# the body — which is what the old negative rule matched on, vetoing the fallback and
# leaving the funded personal key untried.
#
# Shared rather than copied per test file because the exact bytes are the assertion
# substrate: tests reason about what this string does NOT contain ("credit",
# "insufficient", "balance", "402") and about which status digits the 64-hex key hash
# happens to carry (none of 401/402/403/429 — the only "403" is the JSON code field).
# Divergent copies would quietly invalidate that reasoning.
PRODUCTION_KEY_LIMIT_403 = (
    "litellm.APIError: APIError: OpenrouterException - "
    '{"error":{"message":"Key limit exceeded (total limit). Manage it using '
    "https://openrouter.ai/workspaces/default/keys/"
    '8f5af82f134c33c0dbada6e1ce93b780819cc08716001bef5ab4af81791702bd","code":403}}'
)


def gather_predictions_stub(result: tuple[Any, Any, Any]) -> AsyncMock:
    """An ``AsyncMock`` stand-in for ``TemplateForecaster._gather_predictions_with_wall_clock``.

    ``_research_and_make_predictions`` (``metaculus_bot/forecaster.py``) builds one
    coroutine per forecaster by CALLING ``_forecaster_with_soft_deadline``, then hands
    the whole list to ``_gather_predictions_with_wall_clock``, which owns them from
    that point on. Tests that stub the forecaster with an ``AsyncMock`` make each of
    those calls produce a real coroutine object, so a plain ``MagicMock`` stand-in for
    gather silently drops them: every one is later garbage-collected unawaited and
    emits ``RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never
    awaited``. Those warnings are attributed to whichever unrelated test happened to
    trigger the collection, which is why they were so hard to place.

    This closes the coroutines it receives (honoring gather's ownership contract)
    without running the stubs, then returns ``result`` — the ``(valid_predictions,
    errors, exception_group)`` triple the real function returns. The returned mock
    records its calls normally, so assertions on gather's args still work.
    """

    async def _close_tasks_and_return(tasks, *_args, **_kwargs):
        for task in tasks:
            task.close()
        return result

    return AsyncMock(side_effect=_close_tasks_and_return)


def make_mock_binary_question(qid: int = 1001) -> MagicMock:
    """Return a ``MagicMock(spec=BinaryQuestion)`` with standard fields populated."""
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = qid
    q.question_text = "Will it rain?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = f"https://example.com/q/{qid}"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    # Real MetaculusQuestion objects always carry api_json; the prompts read
    # api_json["question"] for the Mantic multi_resolution and grid clauses.
    q.api_json = {"question": {}}
    return q


def make_mock_mc_question(qid: int = 1002, options: list[str] | None = None) -> MagicMock:
    """Return a ``MagicMock(spec=MultipleChoiceQuestion)`` with configurable options."""
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.id_of_question = qid
    q.question_text = "Which color?"
    q.options = options if options is not None else ["Red", "Blue", "Green"]
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = f"https://example.com/q/{qid}"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    # Real MetaculusQuestion objects always carry api_json; the prompts read
    # api_json["question"] for the Mantic multi_resolution and grid clauses.
    q.api_json = {"question": {}}
    return q


@pytest.fixture
def mock_os_getenv():
    with patch("os.getenv") as mock_getenv:
        yield mock_getenv


def make_mock_numeric_question(
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 100.0,
    open_lower_bound: bool = False,
    open_upper_bound: bool = False,
    zero_point: float | None = None,
    cdf_size: int = 201,
    id_of_question: int = 42,
    page_url: str | None = None,
    question_text: str = "What will X be?",
    background_info: str = "bg",
    resolution_criteria: str = "rc",
    fine_print: str = "",
    unit_of_measure: str = "USD",
    nominal_lower_bound: float | None = None,
    nominal_upper_bound: float | None = None,
    id_of_post: int | None = None,
    with_open_resolve_times: bool = False,
) -> MagicMock:
    """Return a ``MagicMock(spec=NumericQuestion)`` with all common fields populated.

    Centralizes the small differences that used to live in ~8 inline helpers across
    the test suite. Field defaults match the most common shape (closed [0, 100] in
    USD with question id 42); per-test overrides land via keyword args.

    ``with_open_resolve_times=True`` populates ``open_time`` (now - 30d) and
    ``scheduled_resolution_time`` (now + 365d), required by helpers that call
    ``_forecasting_window_str``.
    """
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = id_of_question
    q.id_of_post = id_of_post if id_of_post is not None else id_of_question
    q.page_url = page_url if page_url is not None else f"https://example.com/q/{id_of_question}"
    q.question_text = question_text
    q.background_info = background_info
    q.resolution_criteria = resolution_criteria
    q.fine_print = fine_print
    q.unit_of_measure = unit_of_measure
    q.lower_bound = lower_bound
    q.upper_bound = upper_bound
    q.open_lower_bound = open_lower_bound
    q.open_upper_bound = open_upper_bound
    q.zero_point = zero_point
    q.cdf_size = cdf_size
    q.nominal_lower_bound = nominal_lower_bound
    q.nominal_upper_bound = nominal_upper_bound
    # No platform-specific flags: the prompts read Mantic's ``multi_resolution`` / ``precision``
    # off ``api_json["question"]``, and a MagicMock attribute chain is truthy.
    q.api_json = {"question": {}}
    if with_open_resolve_times:
        q.open_time = datetime.now() - timedelta(days=30)
        q.scheduled_resolution_time = datetime.now() + timedelta(days=365)
    return q


@pytest.fixture
def make_mock_numeric_q():
    """Pytest-fixture wrapper around ``make_mock_numeric_question``."""
    return make_mock_numeric_question


@pytest.fixture(autouse=True)
def _enable_per_type_stacking(monkeypatch):
    """Force the per-type stacking gates ON for the whole test suite.

    Production defaults all three ``<TYPE>_STACKING_ENABLED`` flags to DISABLED
    (the stacker only runs when a deploy explicitly opts in). Most stacking
    tests, however, exist to exercise the stacking MECHANISM (crux extraction,
    targeted search, aggregation, fallbacks, thresholds) — they assume the
    stacker is reachable. Setting the flags here keeps those tests faithful to
    their intent without each having to opt in.

    Tests that assert the production DEFAULT (off-when-unset) or a specific
    polarity override their flag in the test body via ``monkeypatch.delenv`` /
    ``monkeypatch.setenv``; that runs after this setup fixture, so the later
    value wins.
    """
    monkeypatch.setenv("BINARY_STACKING_ENABLED", "true")
    monkeypatch.setenv("MC_STACKING_ENABLED", "true")
    monkeypatch.setenv("NUMERIC_STACKING_ENABLED", "true")


@pytest.fixture(autouse=True)
def _clear_gemini_client_cache():
    """Clear the module-global genai.Client lru_cache between tests.

    The Gemini provider caches one client per API key via functools.lru_cache;
    without clearing, a test that mocks genai.Client will see a stale cached
    mock from an earlier test that used a different mock.

    Autouse-global because the gemini client cache is process-wide and can
    pollute even unrelated tests if any prior test loads the module (e.g. via
    a transitive import in main.py / research_providers.py). Scoping to
    gemini-named tests would miss those indirect-load cases. The clear is
    cheap (a single ``cache_clear`` on a 1-entry lru_cache) so the per-test
    cost is negligible — leaving the autouse global is the simpler,
    safer choice.
    """
    from metaculus_bot.research import gemini_search as gsp

    gsp._cached_client_for_key.cache_clear()
    yield
    gsp._cached_client_for_key.cache_clear()


@pytest.fixture(autouse=True)
def _stop_litellm_logging_worker() -> Iterator[None]:
    """Stop litellm's global logging worker on its own loop before the loop is abandoned.

    ``nest_asyncio`` (applied by ``forecasting_tools``) makes ``asyncio.run`` run on the current
    loop without cancelling leftover tasks, so a sync test driving the CLI strands the worker's
    ``_worker_loop`` task on an open loop. When that loop is closed later, the pending coroutine's
    garbage collection raises ``RuntimeError: Event loop is closed`` into an unrelated test.
    Async tests need nothing: pytest-asyncio's runner already cancelled the task (``done()``).
    ``_bound_loop`` is private; litellm's public ``stop()`` is a coroutine and needs that loop.
    """
    yield
    worker_task = GLOBAL_LOGGING_WORKER._worker_task
    worker_loop = GLOBAL_LOGGING_WORKER._bound_loop
    if worker_task is not None and not worker_task.done() and worker_loop is not None and not worker_loop.is_closed():
        worker_loop.run_until_complete(GLOBAL_LOGGING_WORKER.stop())


def _zero_alertable_counters() -> None:
    """Zero every module-global counter ``alertable_count`` sums, as run start does."""
    reset_run_degradation_counters()
    reset_publish_attempt_failures()
    reset_publish_skipped_closed()


@pytest.fixture(autouse=True)
def _isolate_alertable_counters() -> Iterator[None]:
    """Give every test the fresh-run counters ``forecast_questions`` grants a real run.

    The prediction-market, provider-health, publish-hardening and close-gate counters are
    module state (each soft-fails with no handle back to the bot), so a degradation one test
    records reddens a later test's fresh-bot ``alertable_count == 0`` whenever collection puts
    the two in that order. Two files leaked that way:
    ``tests/cli/test_cli_provider_degradation.py`` and ``tests/test_provider_flag_and_logging.py``.
    """
    _zero_alertable_counters()
    yield
    _zero_alertable_counters()


@pytest.fixture
def test_llms():
    """Shared LLM config with a mock default and real parser/researcher/summarizer."""
    from metaculus_bot.llm_configs import PARSER_LLM, RESEARCHER_LLM, SUMMARIZER_LLM

    return {
        "default": MagicMock(),
        "parser": PARSER_LLM,
        "researcher": RESEARCHER_LLM,
        "summarizer": SUMMARIZER_LLM,
    }


def _build_mock_question(
    *,
    question_id: int,
    question_text: str,
    resolution_criteria: str | None = None,
    fine_print: str | None = None,
) -> MagicMock:
    question = MagicMock()
    question.id_of_question = question_id
    question.question_text = question_text
    question.page_url = f"https://example.com/q/{question_id}"
    if resolution_criteria is not None:
        question.resolution_criteria = resolution_criteria
    if fine_print is not None:
        question.fine_print = fine_print
    return question


@pytest.fixture
def make_mock_question():
    """Factory for building mock MetaculusQuestion objects with configurable fields."""
    return _build_mock_question


# Shared TemplateForecaster mocks.
#
# Shared rather than per-file because the forecaster's own tests are split across
# three modules by responsibility (the bot itself, drop attribution, degradation
# counters) and all three construct the same one-forecaster bot. The open/resolve
# times are relative to now (not conftest's fixed _OPEN/_RESOLVE) because the
# prompt builders call _forecasting_window_str, which reads them as a live window.


def make_mock_general_llm(model: str = "mock_model") -> MagicMock:
    """A ``MagicMock(spec=GeneralLlm)`` whose ``invoke`` returns canned reasoning."""
    llm = MagicMock(spec=GeneralLlm)
    llm.model = model
    llm.invoke = AsyncMock(return_value="mock reasoning")
    return llm


@pytest.fixture
def mock_general_llm() -> MagicMock:
    return make_mock_general_llm()


@pytest.fixture
def mock_binary_question() -> MagicMock:
    question = MagicMock(spec=BinaryQuestion)
    question.page_url = "http://example.com/binary_question"
    question.question_text = "Binary Test Question"
    question.background_info = "Binary background info"
    question.resolution_criteria = "Binary resolution criteria"
    question.fine_print = "Binary fine print"
    question.unit_of_measure = "binary units"
    question.id_of_question = 456
    question.open_time = datetime.now() - timedelta(days=30)
    question.scheduled_resolution_time = datetime.now() + timedelta(days=365)
    # Real MetaculusQuestion objects always carry api_json; the prompts read
    # api_json["question"] for the Mantic multi_resolution and grid clauses.
    question.api_json = {"question": {}}
    return question
