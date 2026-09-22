"""Seam-pinning tests for the ``GeneralLlm`` -> ``litellm.acompletion`` kwarg funnel.

These lock down the behaviours the 0.2.54 -> 0.2.92 forecasting-tools upgrade is
known (or suspected) to silently break. Every test is GREEN on the currently
installed 0.2.54 and is written to keep the migration honest — if a seam here
goes red at 0.2.92, the migration touched something load-bearing.

Seams covered:

* **Kwarg funnel** (``TestProductionKwargShapesReachAcompletion``): our exact
  production kwarg shapes — ``reasoning={"effort": "xhigh"}``,
  ``extra_body={"verbosity": "high"}``, top-level ``verbosity``,
  ``plugins=[{"id": "web", ...}]`` + ``web_search_options``,
  ``response_format=<pydantic model>``, ``timeout`` — must reach
  ``acompletion`` byte-for-byte. ``GeneralLlm`` funnels everything via
  ``**self.litellm_kwargs`` today; if 0.2.92 adds param filtering or reshapes
  ``pass_through_unknown_kwargs`` these arrive mangled and the models silently
  run with the wrong reasoning effort / no web plugin / no structured output.

* **temperature default semantics** (``TestTemperatureNoneWorkaround``): pre-0.2.92
  the ctor default was ``temperature=0`` — so our reasoning configs pass
  ``temperature=None`` *explicitly* to keep a hard ``0`` from being injected
  (reasoning models degrade under an explicit sampling temperature). At 0.2.92 the
  ctor default flipped to ``None`` (revised here in W5), so that explicit ``None``
  is now redundant-but-harmless and is kept as a defensive pin against a future
  default flip. NOTE: ``temperature=None`` does NOT drop the key from the
  acompletion kwargs — the funnel forwards ``temperature=None`` and litellm (with
  ``drop_params``) strips the None value downstream. We pin the funnel behaviour we
  can observe: both the explicit ``None`` and the omitted arg reach the funnel as
  ``None`` at 0.2.92 (pre-0.2.92 the omitted arg injected ``0``).

* **litellm.drop_params global** (``TestDropParamsGlobal``): the agentic
  research loop calls raw ``acompletion`` and relies on ``litellm.drop_params``
  being ``True`` globally (``research/agentic/llm.py`` strips
  ``reasoning_effort`` for OpenRouter only because of it). ``GeneralLlm`` sets
  the global to ``True`` on its first real call — NOT at import — so this pins
  that a ``GeneralLlm`` invoke leaves the process-wide flag set.

* **FallbackOpenRouterLlm key swap** (``TestFallbackOpenRouterKeySwap``): a
  donated-key credential failure (401) must construct/route to a *second*
  ``GeneralLlm`` on the personal key and succeed; a 403 must NOT fall back. These
  drive the fake all the way down at the ``acompletion`` boundary (not at
  ``_invoke_once_using_primary``) so the real two-instance key funnel is
  exercised.

* **agentic tools-path executes under the installed litellm**
  (``TestAgenticToolsPathExecutesUnderLitellm``): the agentic gap-fill v2 loop is
  the ONLY caller that passes ``tools=`` to ``acompletion``. litellm >=1.92
  eagerly imports its proxy MCP-gateway handler — which needs ``fastapi``, a
  proxy-only extra we do NOT install — whenever ``tools=`` is present, BEFORE it
  checks whether MCP is actually in use. That import fired on v2's first call,
  crashed with ``ModuleNotFoundError: No module named 'fastapi'`` (wrapped as
  ``APIConnectionError``), and the loop soft-failed to "" — so v2 (an always-on
  prod research feature) was SILENTLY DEAD with nothing red in CI. The fix
  (``research/agentic/llm.py`` passes ``_skip_mcp_handler=True``) skips that
  import. This test drives the REAL ``litellm.acompletion`` through the REAL
  production wrapper ``build_default_llm_call`` with a non-empty ``tools_json``
  (offline via ``mock_response`` — the mock short-circuits the network but the
  MCP import still fires because it is gated on ``tools=`` first), so it CRASHES
  if the skip kwarg is ever dropped. Unlike the shape-pins above, this pins that
  the tools-path can EXECUTE at all under the installed litellm — the exact gap
  that let the fastapi defect ship.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import agents
import litellm
import pytest
from agents.tracing import get_trace_provider
from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models import general_llm as ft_general_llm
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices, Message, Usage

from metaculus_bot.credit_telemetry import ROLE_METADATA_KEY, reset_donated_key_state_cache
from metaculus_bot.fallback_openrouter import FallbackOpenRouterLlm
from metaculus_bot.llm_configs import FORECASTER_LLMS, STACKER_FALLBACK_LLM, STACKER_LLM
from metaculus_bot.research.agentic import llm as agentic_llm
from metaculus_bot.research.agentic.types import LoopConfig
from metaculus_bot.research.providers import build_native_search_llm
from metaculus_bot.structured_parse import PercentileListWrapper, _build_constrained_llm
from tests.conftest import PRODUCTION_KEY_LIMIT_403

# Our production reasoning-config shapes, pinned literally so a reader sees the
# exact values and the tests stay non-vacuous even if the roster rotates.
_PROD_REASONING_XHIGH: dict[str, str] = {"effort": "xhigh"}
_PROD_VERBOSITY_EXTRA_BODY: dict[str, str] = {"verbosity": "high"}


def _canned_response(text: str = "MODEL ANSWER") -> ModelResponse:
    """A minimally-valid litellm ``ModelResponse`` GeneralLlm will accept.

    GeneralLlm asserts the response is a ``ModelResponse`` with ``list[Choices]``
    choices and a ``Usage`` object, and raises on an empty-string answer, so all
    three fields must be populated with a non-empty message.
    """
    response = ModelResponse()
    response.choices = [Choices(message=Message(role="assistant", content=text), finish_reason="stop", index=0)]
    response.usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)  # type: ignore[attr-defined]
    return response


def _install_acompletion(
    monkeypatch: pytest.MonkeyPatch,
    *,
    script: list[Exception | str] | None = None,
    answer: str = "MODEL ANSWER",
) -> list[dict[str, Any]]:
    """Patch ``acompletion`` where general_llm.py looks it up; record every call's kwargs.

    ``from litellm import acompletion`` binds the symbol as a module global in
    ``forecasting_tools.ai_models.general_llm`` and it is called there as a bare
    ``acompletion(...)`` — so the correct patch point is that module attribute,
    not ``litellm.acompletion``.

    With ``script`` unset every call returns a canned success. With ``script``
    set, call ``i`` raises (if the element is an ``Exception``) or returns a
    canned response carrying that text (if it is a ``str``) — used to drive the
    donated-key-fails-then-personal-key-succeeds fallback path.
    """
    calls: list[dict[str, Any]] = []
    script_iter = iter(script) if script is not None else None

    async def fake_acompletion(**kwargs: Any) -> ModelResponse:
        calls.append(kwargs)
        if script_iter is not None:
            action = next(script_iter)
            if isinstance(action, Exception):
                raise action
            return _canned_response(action)
        return _canned_response(answer)

    monkeypatch.setattr(ft_general_llm, "acompletion", fake_acompletion)
    return calls


@pytest.fixture(autouse=True)
def _disable_agents_tracing() -> Any:
    """Silence the openai-agents trace exporter for the duration of each test.

    0.2.54's ``GeneralLlm`` wraps invokes in ``track_generation`` and stashes
    ``self.litellm_kwargs`` (which includes a pydantic ``response_format`` *class*)
    in the span. The background trace exporter then tries to ``json.dumps`` it and
    raises ``TypeError: Object of type ModelMetaclass is not JSON serializable``
    on a daemon thread — harmless to the assertion but noisy. Disable tracing and
    restore the prior state so we don't leak the flag into other test modules.
    """

    # No public getter for the current state; read the private flag defensively
    # (default False = the library default) so the restore never leaks True.
    previously_disabled = bool(getattr(get_trace_provider(), "_disabled", False))
    agents.set_tracing_disabled(True)
    yield
    agents.set_tracing_disabled(previously_disabled)


class TestProductionKwargShapesReachAcompletion:
    """Our exact production kwarg shapes must reach ``acompletion`` untouched."""

    async def test_full_reasoning_shape_survives_the_funnel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One instance carrying every production-shaped kwarg; each arrives intact."""
        calls = _install_acompletion(monkeypatch)
        response_format = PercentileListWrapper
        llm = GeneralLlm(
            model="openrouter/anthropic/claude-opus-4.8",
            temperature=None,
            timeout=480,
            allowed_tries=1,
            api_key="key",
            reasoning=_PROD_REASONING_XHIGH,
            extra_body=_PROD_VERBOSITY_EXTRA_BODY,
            verbosity="high",
            plugins=[{"id": "web", "max_results": 20, "engine": "native"}],
            web_search_options={"search_context_size": "high"},
            response_format=response_format,
        )

        out = await llm.invoke("forecast this")

        assert out == "MODEL ANSWER"
        assert len(calls) == 1
        sent = calls[0]
        # Every nested shape survives byte-for-byte (identity for the pydantic class).
        assert sent["reasoning"] == {"effort": "xhigh"}
        assert sent["extra_body"] == {"verbosity": "high"}
        assert sent["verbosity"] == "high"
        assert sent["plugins"] == [{"id": "web", "max_results": 20, "engine": "native"}]
        assert sent["web_search_options"] == {"search_context_size": "high"}
        assert sent["response_format"] is response_format
        assert sent["timeout"] == 480
        # temperature=None is forwarded (litellm drops the None downstream), not turned into 0.
        assert sent["temperature"] is None
        # model prefix stripping is a no-op for openrouter/* — the full slug reaches litellm.
        assert sent["model"] == "openrouter/anthropic/claude-opus-4.8"

    async def test_native_search_config_object_funnels_web_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The real ``build_native_search_llm`` object funnels the web-search plumbing."""
        calls = _install_acompletion(monkeypatch)
        # Explicit effort/verbosity overrides so the shapes are pinned regardless of env.
        llm = build_native_search_llm("openai/gpt-5.6-terra", reasoning_effort="low", verbosity="high")

        await llm.invoke("research this")

        assert len(calls) == 1
        sent = calls[0]
        assert sent["plugins"] == [{"id": "web", "max_results": 20, "engine": "native"}]
        assert sent["web_search_options"] == {"search_context_size": "high"}
        assert sent["verbosity"] == "high"  # top-level, NOT tucked in extra_body
        assert sent["reasoning"] == {"effort": "low"}
        assert sent["temperature"] is None

    async def test_constrained_parser_config_object_funnels_pydantic_response_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real structured-parse builder funnels a pydantic ``response_format`` class + extra_body."""
        calls = _install_acompletion(monkeypatch)
        llm = _build_constrained_llm(PercentileListWrapper, "openrouter/openai/gpt-5.6-luna")

        await llm.invoke("parse this")

        assert len(calls) == 1
        sent = calls[0]
        # A pydantic MODEL CLASS (not a dict) must arrive — this is what enables strict json_schema.
        assert sent["response_format"] is PercentileListWrapper
        assert isinstance(sent["response_format"], type)
        assert issubclass(sent["response_format"], PercentileListWrapper)
        assert sent["extra_body"] == {"provider": {"require_parameters": True}}
        assert sent["reasoning"] == {"effort": "low"}
        assert sent["temperature"] is None
        # role="parser" books the constrained primary on the same CREDIT_ROLE_SPEND line as
        # PARSER_LLM (both are one parsing job on the same tier). Dropped, it books as
        # ``untagged`` and the parser row silently vanishes from a run's ledger — asserted
        # here because every construction path in build_llm_with_openrouter_fallback stamps
        # the tag as litellm ``metadata``, so this is where a dropped kwarg is observable.
        assert sent["metadata"][ROLE_METADATA_KEY] == "parser"

    async def test_real_forecaster_roster_funnels_declared_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every real forecaster slot funnels its declared kwargs (and temperature=None) intact.

        Roster-agnostic: iterates the live ``FORECASTER_LLMS`` singletons and asserts
        the funnel is transparent for whatever is configured. Also confirms the
        current production reasoning/verbosity shapes are present on at least one slot
        so this stays a meaningful pin rather than a vacuous loop, and that no slot sends
        ``verbosity`` beside ``reasoning``: on Anthropic, OpenRouter maps both onto one
        effort knob and verbosity wins, which silently ran the Opus slot at high, not xhigh.
        """
        assert FORECASTER_LLMS, "roster must be non-empty for this pin to mean anything"
        saw_xhigh_reasoning = False

        for llm in FORECASTER_LLMS:
            declared = llm.litellm_kwargs
            calls = _install_acompletion(monkeypatch)
            await llm.invoke("forecast")
            assert len(calls) == 1
            sent = calls[0]

            # temperature=None must survive for every reasoning forecaster (no injected 0).
            assert sent["temperature"] is None
            assert sent["timeout"] == declared["timeout"]
            if "reasoning" in declared:
                assert sent["reasoning"] == declared["reasoning"]
                if declared["reasoning"] == _PROD_REASONING_XHIGH:
                    saw_xhigh_reasoning = True
            if "extra_body" in declared:
                assert sent["extra_body"] == declared["extra_body"]
            sends_verbosity = "verbosity" in sent or "verbosity" in (sent.get("extra_body") or {})
            assert not ("reasoning" in sent and sends_verbosity), f"{llm.model} sends verbosity beside reasoning"

        assert saw_xhigh_reasoning, "expected a forecaster with reasoning={'effort':'xhigh'} in the roster"


def test_no_llm_config_sends_verbosity_beside_reasoning() -> None:
    """OpenRouter maps both ``verbosity`` and ``reasoning.effort`` onto Anthropic's single
    ``output_config.effort`` and verbosity wins, so sending both silently overrides the declared effort."""
    for llm in [*FORECASTER_LLMS, STACKER_LLM, STACKER_FALLBACK_LLM]:
        declared = llm.litellm_kwargs
        sends_verbosity = "verbosity" in declared or "verbosity" in (declared.get("extra_body") or {})
        assert not ("reasoning" in declared and sends_verbosity), f"{llm.model} sends verbosity beside reasoning"


class TestTemperatureNoneWorkaround:
    """Temperature default semantics at 0.2.92.

    Decision (W5, 0.2.92 upgrade): the ``GeneralLlm`` ctor default flipped 0 -> None
    at 0.2.92 (verified: general_llm.py stores ``temperature`` unchanged, default
    ``None``). So omitting the arg now yields ``None`` just like passing it — the
    hard-0 injection these tests were the canary for is gone. Our reasoning configs
    still pass ``temperature=None`` explicitly; that is now redundant-but-harmless
    and is kept as a defensive pin against a future upstream default flip. These two
    tests pin the new invariant: both the explicit ``None`` and the omitted arg reach
    the funnel as ``None`` (litellm's drop_params strips it downstream), never a 0.
    """

    async def test_temperature_none_is_forwarded_not_coerced_to_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _install_acompletion(monkeypatch)
        llm = GeneralLlm(model="openrouter/openai/gpt-5.6-sol", temperature=None, timeout=480, allowed_tries=1)

        await llm.invoke("hi")

        assert len(calls) == 1
        # temperature=None reaches the funnel unchanged (the None, not a 0). litellm's
        # drop_params strips the None value before it hits the provider.
        assert calls[0]["temperature"] is None

    async def test_omitting_temperature_defaults_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """0.2.92 semantics: the ctor default is None, so omitting the arg yields None (no injected 0).

        Pre-0.2.92 this injected a hard 0, which is why our reasoning configs pass
        ``temperature=None`` explicitly. At 0.2.92 that explicit None is redundant
        with the new default — this test pins that omission no longer reintroduces a 0.
        """
        calls = _install_acompletion(monkeypatch)
        llm = GeneralLlm(model="openrouter/openai/gpt-5.6-sol", timeout=480, allowed_tries=1)

        await llm.invoke("hi")

        assert len(calls) == 1
        assert calls[0]["temperature"] is None


class TestDropParamsGlobal:
    """``litellm.drop_params`` must be globally True after a GeneralLlm call.

    The agentic loop (``research/agentic/llm.py``) calls raw ``acompletion`` and
    relies on this global being True to silently strip params litellm can't map for
    OpenRouter. ``GeneralLlm`` sets it inside ``_mockable_direct_call_to_model`` on
    every call — NOT at import — so we reset it to False, invoke, and assert it flipped.
    """

    async def test_general_llm_invoke_sets_drop_params_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Reset via monkeypatch (auto-restored) so this is deterministic regardless of
        # whatever earlier tests in the suite already invoked a GeneralLlm.
        monkeypatch.setattr(litellm, "drop_params", False)
        calls = _install_acompletion(monkeypatch)
        llm = GeneralLlm(model="openrouter/openai/gpt-5.6-sol", temperature=None, timeout=480, allowed_tries=1)

        assert litellm.drop_params is False  # ctor alone does not flip it
        await llm.invoke("hi")

        assert len(calls) == 1
        assert litellm.drop_params is True  # the invoke sets the process-wide global the agentic loop leans on


class TestFallbackOpenRouterKeySwap:
    """FallbackOpenRouterLlm's donated->personal key swap, exercised at the acompletion boundary."""

    async def test_401_falls_back_to_personal_key_via_second_general_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A donated-key 401 constructs/uses the personal-key GeneralLlm and succeeds."""
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.6-sol",
            primary_api_key="donated-key",
            secondary_api_key="personal-key",
            temperature=None,
            timeout=480,
            allowed_tries=1,
        )
        # The fallback secondary is a real, separate GeneralLlm carrying the personal key.
        assert isinstance(llm._secondary_llm, GeneralLlm)  # pinning the two-instance funnel
        assert llm._secondary_llm.litellm_kwargs["api_key"] == "personal-key"

        calls = _install_acompletion(
            monkeypatch,
            script=[Exception("401 Unauthorized: invalid api key"), "PERSONAL-KEY ANSWER"],
        )

        out = await llm.invoke("hi")

        assert out == "PERSONAL-KEY ANSWER"
        # Two calls: donated key raised 401, then the personal-key instance succeeded.
        assert [c["api_key"] for c in calls] == ["donated-key", "personal-key"]

    async def test_403_does_not_fall_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Negative control: a 403 (moderation/blocked) propagates — both keys would refuse."""
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.6-sol",
            primary_api_key="donated-key",
            secondary_api_key="personal-key",
            temperature=None,
            timeout=480,
            allowed_tries=1,
        )
        calls = _install_acompletion(
            monkeypatch,
            script=[Exception("403 Forbidden moderation")],
        )

        with pytest.raises(Exception, match="403 Forbidden moderation"):
            await llm.invoke("hi")

        # Only the primary was tried — no personal-key fallback on a 403.
        assert [c["api_key"] for c in calls] == ["donated-key"]

    async def test_429_does_fall_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 429 rate-limit DOES fall back — BYOK quotas are per-key, so the personal key may serve it.

        (This pins the real ``should_retry_with_general_key`` contract: 429 is a
        fall-back trigger, distinct from the 403/502/503 non-fallback cases.)
        """
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.6-sol",
            primary_api_key="donated-key",
            secondary_api_key="personal-key",
            temperature=None,
            timeout=480,
            allowed_tries=1,
        )
        calls = _install_acompletion(
            monkeypatch,
            script=[Exception("429 Too Many Requests"), "PERSONAL-KEY ANSWER"],
        )

        out = await llm.invoke("hi")

        assert out == "PERSONAL-KEY ANSWER"
        assert [c["api_key"] for c in calls] == ["donated-key", "personal-key"]

    async def test_key_limit_exceeded_403_falls_back_to_personal_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 2026-07-26 production failure, driven at the acompletion boundary.

        OpenRouter reports a drained per-key spend cap as HTTP 403 with
        "Key limit exceeded (total limit)" — not the 402 its docs promise — and
        litellm has no 403 branch for OpenRouter, so it arrives as a bare
        ``APIError``. The old rule vetoed any message containing "403", which the
        ``"code":403`` field in the body supplied, so the funded personal key was
        never tried and two of three forecasters died.

        The donated key env var is cleared so the wrapper's drained-vs-revoked probe
        classifies as UNKNOWN without touching the network. UNKNOWN is the pessimistic
        verdict (the run stays alertable), which makes this the strictest version of
        the assertion: routing to the personal key must not depend on the probe.
        """
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        reset_donated_key_state_cache()
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.6-sol",
            primary_api_key="donated-key",
            secondary_api_key="personal-key",
            temperature=None,
            timeout=480,
            allowed_tries=1,
        )
        calls = _install_acompletion(
            monkeypatch,
            script=[Exception(PRODUCTION_KEY_LIMIT_403), "PERSONAL-KEY ANSWER"],
        )

        out = await llm.invoke("hi")

        assert out == "PERSONAL-KEY ANSWER"
        assert [c["api_key"] for c in calls] == ["donated-key", "personal-key"]


def _forward_to_real_litellm_with_mock(mock_text: str) -> tuple[Callable[..., Any], list[dict[str, Any]]]:
    """Wrap the REAL ``litellm.acompletion``, injecting ``mock_response``.

    The whole point of this indirection: ``build_default_llm_call``'s
    ``_call_once`` builds the kwargs dict internally and calls
    ``acompletion(**kwargs)`` against the ``acompletion`` symbol bound in
    ``research/agentic/llm.py``. We monkeypatch that symbol to this wrapper, which
    forwards to the genuine ``litellm.acompletion`` with ``mock_response`` added.
    That keeps the REAL litellm call executing — so litellm's ``tools=``-gated
    eager import of its proxy MCP-gateway handler still fires (``mock_response``
    only short-circuits the network, AFTER that import). A plain ``AsyncMock``
    here would defeat the whole test by skipping the import entirely.

    Returns the wrapper plus a ``calls`` list recording each forwarded call's
    kwargs, so a test can assert what actually reached ``litellm.acompletion``
    (specifically that a non-empty ``tools=`` did — the seam only fires on that).
    """
    calls: list[dict[str, Any]] = []

    async def _wrapper(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return await litellm.acompletion(**kwargs, mock_response=mock_text)

    return _wrapper, calls


class TestAgenticToolsPathExecutesUnderLitellm:
    """The agentic tools-path must be able to EXECUTE under the installed litellm.

    See the module docstring: v2 is the only ``tools=`` caller, litellm >=1.92
    eagerly imports a fastapi-dependent MCP handler on ``tools=`` before checking
    if MCP is used, and that crashed v2 silently. This drives the real
    ``litellm.acompletion`` through the real ``build_default_llm_call`` wrapper
    with a real ``tools_json`` (offline via ``mock_response``), so it goes red the
    instant ``_skip_mcp_handler=True`` is dropped from ``llm.py``.
    """

    async def test_tools_path_survives_litellm_mcp_import_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # drop_params must be True for the wrapper's OpenRouter reasoning_effort
        # kwarg handling; a GeneralLlm invoke sets it globally in prod, but this
        # test never constructs one, so set it here (monkeypatch auto-restores).
        monkeypatch.setattr(litellm, "drop_params", True)
        # Personal-key-only routing keeps the wrapper on a single deterministic
        # acompletion call (no donated->personal fallback branch to reason about).
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-personal-key")
        # Forward to the REAL litellm so the tools=-gated MCP import actually runs.
        wrapper, forwarded_calls = _forward_to_real_litellm_with_mock("MOCKED")
        monkeypatch.setattr(agentic_llm, "acompletion", wrapper)

        call = agentic_llm.build_default_llm_call(LoopConfig(model="openai/gpt-5.6-terra", reasoning_effort="low"))
        # One plain function tool — exactly the shape the gap-fill loop passes.
        # Its presence is what trips litellm's eager MCP-handler import.
        tools_json = [
            {
                "type": "function",
                "function": {
                    "name": "fetch",
                    "description": "fetch a url",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            }
        ]

        # Without _skip_mcp_handler=True in llm.py this raises APIConnectionError
        # (wrapping ModuleNotFoundError: No module named 'fastapi'); with it, the
        # real litellm call reaches the mock and returns cleanly.
        result = await call([{"role": "user", "content": "hi"}], tools_json)

        assert result.choices[0].message.content == "MOCKED"
        # The MCP-import seam only fires when a non-empty tools= actually reaches
        # litellm.acompletion. Assert it did, so this test can't silently stop
        # exercising the fastapi gate (e.g. if the wrapper's kwarg plumbing drifts).
        assert forwarded_calls, "wrapper never forwarded a call to litellm.acompletion"
        forwarded_tools = forwarded_calls[-1].get("tools")
        assert isinstance(forwarded_tools, list), (
            f"tools= did not reach litellm.acompletion as a non-empty list: {forwarded_tools!r}"
        )
        assert forwarded_tools, f"tools= did not reach litellm.acompletion as a non-empty list: {forwarded_tools!r}"
