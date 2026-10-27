"""Tests for the free-model forecaster lineup used by the ablation runner.

Posture pinned by these tests:

* Lineup is 4 free models. ``gpt-oss-120b:free`` was removed (task #16,
  donated-key wrapper 404'd). ``glm-4.5-air:free`` was removed
  (Phase A.3 Package 3b, 2026-05-14, after qid 43171 hallucinated
  partial-week TSA data with sigma=13K vs ensemble sigma ~965K).
* The FREE forecaster + parser are constructed as plain ``GeneralLlm`` (no
  ``api_key`` set), so litellm reads ``OPENROUTER_API_KEY`` from env at
  invoke time — ``:free`` models bypass the donated key because their
  providers aren't in ``DONATED_KEY_PROVIDERS`` (donated key 404s).
* The PROD lineup routes through the donated-key wrapper: ablation IS Metaculus
  work, and its anthropic/openai models are donated-covered, so they bill to the
  donated key with personal-key fallback.
"""

from __future__ import annotations

from forecasting_tools import GeneralLlm


def test_free_forecaster_models_count_is_four() -> None:
    """4 forecasters after glm-4.5-air:free was removed (Phase A.3 Package 3b, 2026-05-14)."""
    from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

    assert len(FREE_FORECASTER_MODELS) == 4


def test_glm_dropped_from_lineup() -> None:
    """Regression: qid 43171 hallucinated TSA partial-week data; GLM dropped 2026-05-14.

    GLM-4.5-air emitted a "normal" distribution with sigma=13K vs ensemble median sigma ~965K
    (1.3% of ensemble sigma). Arm B's stacker over-weighted GLM and saturated the schema
    floor (-220 log score). Sign-off: user, post-Phase-A.2.
    """
    from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

    assert not any("glm-4.5-air" in m for m in FREE_FORECASTER_MODELS), (
        f"glm-4.5-air must not be in lineup; found {FREE_FORECASTER_MODELS}"
    )


def test_free_forecaster_models_all_have_free_suffix() -> None:
    from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

    assert all(model.endswith(":free") for model in FREE_FORECASTER_MODELS), FREE_FORECASTER_MODELS


def test_free_forecaster_models_all_openrouter_prefix() -> None:
    from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

    assert all(model.startswith("openrouter/") for model in FREE_FORECASTER_MODELS)


def test_build_free_forecaster_llms_returns_four_plain_general_llms() -> None:
    """Builder constructs plain GeneralLlm (no api_key, no FallbackOpenRouterLlm wrap)."""
    from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS, build_free_forecaster_llms
    from metaculus_bot.fallback_openrouter import FallbackOpenRouterLlm

    llms = build_free_forecaster_llms()
    assert len(llms) == 4
    for llm, expected_model in zip(llms, FREE_FORECASTER_MODELS, strict=False):
        assert isinstance(llm, GeneralLlm)
        assert not isinstance(llm, FallbackOpenRouterLlm), (
            f"{expected_model} must be plain GeneralLlm (no donated-key wrap)"
        )
        assert llm.model == expected_model


def test_build_free_parser_llm_uses_gemma_4_31b() -> None:
    """Parser is a Google-served free model (avoids the donated-key wrap entirely).

    History:
    - Originally gpt-oss-120b:free → OAI-prefixed → routed via donated-key wrap → 404 (task #16).
    - Then gemma-4-26b-a4b-it:free → over-applied "do not guess", emitted nulls (task #18 bake-off).
    - Now gemma-4-31b-it:free → 3/3 PASS in the bake-off, deterministic, returns 11 valid floats.

    Plain GeneralLlm (no donated-key wrap) so the operator's paid OPENROUTER_API_KEY
    drives auth at invoke time; ablation stays off the donated quota.
    """
    from metaculus_bot.ablation.forecaster_lineup import FREE_PARSER_MODEL, build_free_parser_llm
    from metaculus_bot.fallback_openrouter import FallbackOpenRouterLlm

    parser = build_free_parser_llm()
    assert isinstance(parser, GeneralLlm)
    assert not isinstance(parser, FallbackOpenRouterLlm)
    assert parser.model == FREE_PARSER_MODEL == "openrouter/google/gemma-4-31b-it:free"


def test_build_prod_forecaster_llms_routes_via_donated_wrapper(monkeypatch) -> None:
    """Prod ensemble bills to the donated key: each model is wrapped in
    FallbackOpenRouterLlm (donated primary -> personal fallback). Ablation IS
    Metaculus work, and these anthropic/openai models are donated-covered.

    Requires both keys set + distinct for the wrapper to engage (matches
    ``build_llm_with_openrouter_fallback``); in CI neither is set, so we inject
    distinct placeholders.
    """
    from metaculus_bot.ablation.forecaster_lineup import PROD_FORECASTER_MODELS, build_prod_forecaster_llms
    from metaculus_bot.fallback_openrouter import FallbackOpenRouterLlm

    monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
    monkeypatch.setenv("OPENROUTER_API_KEY", "general")

    llms = build_prod_forecaster_llms()
    assert len(llms) == len(PROD_FORECASTER_MODELS) == 3
    for llm, expected_model in zip(llms, PROD_FORECASTER_MODELS, strict=False):
        assert isinstance(llm, FallbackOpenRouterLlm), (
            f"{expected_model} must route via the donated-key wrapper (Metaculus work -> donated key)"
        )
        assert llm.model == expected_model
