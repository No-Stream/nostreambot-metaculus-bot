"""
Tests pinning the values of key forecasting clamp constants.

These pin the *specific bounds* separately from the clamp-logic tests, so a
future hyperparameter change is a single-point edit with a clear test to
update and reason about.
"""

import pytest

from metaculus_bot.constants import (
    BINARY_PROB_MAX,
    BINARY_PROB_MIN,
    BINARY_STACKING_ENABLED_ENV,
    DONATED_OPENROUTER_KEY_ENABLED_ENV,
    EXTREME_CALL_HIGH,
    EXTREME_CALL_LOW,
    GAP_FILL_V2_READER_MODEL,
    GEMINI_SEARCH_DEFAULT_MODEL,
    MANTIC_API_BASE_URL,
    MANTIC_BOT_USER_ID,
    MANTIC_HOST,
    MANTIC_SITE_URL,
    MC_PROB_MAX,
    MC_PROB_MIN,
    MC_STACKING_ENABLED_ENV,
    METACULUS_HOST,
    NATIVE_SEARCH_DEFAULT_MODEL,
    NATIVE_SEARCH_REASONING_EFFORT_DEFAULT,
    NATIVE_SEARCH_TIMEOUT,
    NATIVE_SEARCH_VERBOSITY_DEFAULT,
    NUMERIC_STACKING_ENABLED_ENV,
    QUESTION_PLATFORM_HOSTS,
    THIN_PUBLISH_BINARY_CEIL,
    THIN_PUBLISH_BINARY_FLOOR,
    donated_openrouter_key_enabled,
    env_flag_enabled,
)


class TestBinaryClampBounds:
    """Pin binary clamp bounds at [0.02, 0.98]."""

    def test_binary_prob_min_is_0_02(self):
        """BINARY_PROB_MIN == 0.02.

        Adopted from Preseen-Atlas (top bot on spring-AIB-2026), whose comments
        publish `submitted = 0.96 * model_estimate + 0.02` in every forecast —
        equivalent to clipping to [0.02, 0.98]. We're taking the clip (tail
        protection via log-loss reasoning) without the linear shrink.

        See: scratch_docs_and_planning/atlas_inspired_improvements.md (Workstream B).
        """
        assert BINARY_PROB_MIN == 0.02

    def test_binary_prob_max_is_0_98(self):
        """BINARY_PROB_MAX == 0.98.

        See `test_binary_prob_min_is_0_02` for rationale (Atlas-inspired
        clip-only adoption).
        """
        assert BINARY_PROB_MAX == 0.98


class TestMCClampBounds:
    """MC clamp bounds are [0.01, 0.99], aligned to ft 0.2.92's PredictedOptionList
    validator (which clamps every option into [0.01, 0.99] on construction). Matching
    those bounds makes the upstream validator a no-op on our output — see the constant's
    comment and clamp_and_renormalize_probs."""

    def test_mc_prob_min_is_0_01(self):
        assert MC_PROB_MIN == 0.01

    def test_mc_prob_max_is_0_99(self):
        assert MC_PROB_MAX == 0.99


class TestNativeSearchDefaults:
    """Pin OpenAI native-search defaults.

    Model + verbosity locked from ``scratch/native_search_bench_2026-05-17/comparison_v3.md``.
    Reasoning effort dropped medium→low on 2026-05-20 after an OpenRouter
    whitespace-stream incident consumed 8m37s on a single call; v3 bench
    measured low at ~50s vs medium at ~230s, so low gives ~4.5× more headroom
    under the wall-clock cap. Quality at low is not graded by the v3 bench
    (sanity-check only) — rerun the bench with a graded `low` arm if you
    want to revert.
    """

    def test_native_search_default_model_is_gpt_6_sol(self):
        """Locks the default OpenRouter model to ``openai/gpt-6-sol``
        (2026-07-17 sol→terra flip per the blind research-role audit,
        scratch/research_role_audit_2026-07-17/ — terra 1st, sol 2nd; then the
        2026-09-22 GPT-6 migration, where Terra has no GPT-6 successor so the
        role moved to Sol)."""
        assert NATIVE_SEARCH_DEFAULT_MODEL == "openai/gpt-6-sol"

    def test_native_search_reasoning_effort_default_is_low(self):
        """Low effort gives ~4.5× faster wall-clock vs medium on the v3 bench
        (~50s vs ~230s), keeping us well clear of NATIVE_SEARCH_WALL_TIMEOUT
        (420s) after the 2026-05-20 OpenRouter whitespace-stream incident.
        Override via NATIVE_SEARCH_REASONING_EFFORT env if a workflow needs
        medium quality back."""
        assert NATIVE_SEARCH_REASONING_EFFORT_DEFAULT == "low"

    def test_native_search_verbosity_default_is_low(self):
        """Low verbosity keeps the response tight without losing substance."""
        assert NATIVE_SEARCH_VERBOSITY_DEFAULT == "low"

    def test_native_search_timeout_is_360s(self):
        """360s cap leaves ~130s headroom on top of observed p99 (~230s)."""
        assert NATIVE_SEARCH_TIMEOUT == 360


class TestGeminiNativeSdkModelDefaults:
    """Pin both native google-genai model ids to gemini-3.8-flash.

    Verified live on that SDK 2026-09-03 by scripts/probes/gemini_verify.py: grounding
    chunks came back from the google_search tool, thinking_level was accepted, and
    url_context retrieved a robots-allowed host. The grounded-search provider and the
    gap-fill v2 reader deliberately run the SAME id so one verification covers both, and
    so the shared 5k/month grounded-prompt pool is drawn by one model.
    """

    def test_grounded_search_default_model_is_gemini_3_8_flash(self):
        assert GEMINI_SEARCH_DEFAULT_MODEL == "gemini-3.8-flash"

    def test_gap_fill_v2_reader_model_is_gemini_3_8_flash(self):
        assert GAP_FILL_V2_READER_MODEL == "gemini-3.8-flash"

    def test_both_native_surfaces_run_the_same_id(self):
        """Trivially true while they match, which is the point: it fails the moment one
        surface is flipped to a model the other has not been verified on."""
        assert GEMINI_SEARCH_DEFAULT_MODEL == GAP_FILL_V2_READER_MODEL


class TestEnvFlagEnabledDefaultKwarg:
    """Tests for the ``default`` keyword argument on ``env_flag_enabled``."""

    def test_unset_env_returns_default_true(self, monkeypatch):
        """When env var is unset, returns the provided default (True)."""
        monkeypatch.delenv("_TEST_FLAG_NONEXISTENT_XYZ", raising=False)
        assert env_flag_enabled("_TEST_FLAG_NONEXISTENT_XYZ", default=True) is True

    def test_unset_env_returns_default_false(self, monkeypatch):
        """When env var is unset and default=False, returns False."""
        monkeypatch.delenv("_TEST_FLAG_NONEXISTENT_XYZ", raising=False)
        assert env_flag_enabled("_TEST_FLAG_NONEXISTENT_XYZ", default=False) is False

    def test_unset_env_returns_false_when_no_default_specified(self, monkeypatch):
        """Backward compat: no default kwarg means default=False."""
        monkeypatch.delenv("_TEST_FLAG_NONEXISTENT_XYZ", raising=False)
        assert env_flag_enabled("_TEST_FLAG_NONEXISTENT_XYZ") is False

    def test_explicit_false_overrides_default_true(self, monkeypatch):
        """Explicit 'false' always returns False regardless of default."""
        monkeypatch.setenv("_TEST_FLAG_XYZ", "false")
        assert env_flag_enabled("_TEST_FLAG_XYZ", default=True) is False

    def test_explicit_true_overrides_default_false(self, monkeypatch):
        """Explicit 'true' always returns True regardless of default."""
        monkeypatch.setenv("_TEST_FLAG_XYZ", "true")
        assert env_flag_enabled("_TEST_FLAG_XYZ", default=False) is True

    def test_explicit_zero_overrides_default_true(self, monkeypatch):
        """'0' is falsy regardless of default."""
        monkeypatch.setenv("_TEST_FLAG_XYZ", "0")
        assert env_flag_enabled("_TEST_FLAG_XYZ", default=True) is False

    def test_explicit_one_overrides_default_false(self, monkeypatch):
        """'1' is truthy regardless of default."""
        monkeypatch.setenv("_TEST_FLAG_XYZ", "1")
        assert env_flag_enabled("_TEST_FLAG_XYZ", default=False) is True

    def test_empty_string_treated_as_unset(self, monkeypatch):
        """Empty string env var treated same as unset — returns default."""
        monkeypatch.setenv("_TEST_FLAG_XYZ", "")
        assert env_flag_enabled("_TEST_FLAG_XYZ", default=True) is True

    def test_unrecognized_value_returns_default(self, monkeypatch):
        """Garbage value falls through to default."""
        monkeypatch.setenv("_TEST_FLAG_XYZ", "maybe")
        assert env_flag_enabled("_TEST_FLAG_XYZ", default=True) is True


class TestPerTypeStackingEnvVarNames:
    """Pin the env var name constants for per-type stacking gates."""

    def test_binary_stacking_enabled_env_name(self):
        assert BINARY_STACKING_ENABLED_ENV == "BINARY_STACKING_ENABLED"

    def test_mc_stacking_enabled_env_name(self):
        assert MC_STACKING_ENABLED_ENV == "MC_STACKING_ENABLED"

    def test_numeric_stacking_enabled_env_name(self):
        assert NUMERIC_STACKING_ENABLED_ENV == "NUMERIC_STACKING_ENABLED"


class TestDonatedOpenRouterKeyMasterSwitch:
    """Pin the master switch for the Metaculus-donated OpenRouter key.

    Default ON keeps every Metaculus run unchanged; a Mantic run sets the env var false so
    the donated key (Metaculus's money, for Metaculus tournaments) is never spent on another
    platform. The env var NAME is a data contract with the Mantic workflow yaml and with
    cli's fail-shut startup assertion, so it is pinned as a literal.
    """

    def test_env_var_name(self):
        assert DONATED_OPENROUTER_KEY_ENABLED_ENV == "DONATED_OPENROUTER_KEY_ENABLED"

    def test_unset_defaults_to_enabled(self, monkeypatch):
        monkeypatch.delenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, raising=False)
        assert donated_openrouter_key_enabled() is True

    def test_empty_string_is_unset(self, monkeypatch):
        monkeypatch.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, "")
        assert donated_openrouter_key_enabled() is True

    @pytest.mark.parametrize("raw", ["false", "0", "no", "FALSE"])
    def test_false_y_values_disable(self, monkeypatch, raw):
        monkeypatch.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, raw)
        assert donated_openrouter_key_enabled() is False

    def test_explicit_true_enables(self, monkeypatch):
        monkeypatch.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, "true")
        assert donated_openrouter_key_enabled() is True


class TestThinPublishBinaryFloor:
    """The single-survivor publish floor is [0.05, 0.95], aliased to the EXTREME_CALL band.

    One definition of "extreme" serves both the telemetry that measures the exposure and
    the clamp that prices it, so the two cannot drift apart. Receipt for the values:
    scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md §2 (clamp-variant table).
    """

    def test_floor_is_0_05(self):
        assert THIN_PUBLISH_BINARY_FLOOR == 0.05

    def test_ceiling_is_0_95(self):
        assert THIN_PUBLISH_BINARY_CEIL == 0.95

    def test_floor_and_ceiling_alias_the_extreme_call_band(self):
        """Trivially true while the alias holds, which is the point: paired with the two
        literal pins above, it fails the moment somebody re-hardcodes either edge."""
        assert THIN_PUBLISH_BINARY_FLOOR == EXTREME_CALL_LOW
        assert THIN_PUBLISH_BINARY_CEIL == EXTREME_CALL_HIGH

    def test_floor_sits_strictly_inside_the_per_model_clamp(self):
        """A 0.03 member call passes the per-model [0.02, 0.98] clamp untouched, which is
        why the floor is a new mechanism rather than a retune of BINARY_PROB_MIN/MAX."""
        assert BINARY_PROB_MIN < THIN_PUBLISH_BINARY_FLOOR < THIN_PUBLISH_BINARY_CEIL < BINARY_PROB_MAX


class TestQuestionPlatformHosts:
    """Each platform host is spelled ONCE in constants.py; everything else derives from it.

    Literal pins rather than derived ones, deliberately: the derivation is what a re-point (a
    staging host, a domain change when Series 2 opens) relies on to move every consumer at once,
    and these literals are what make a wrong derivation loud rather than split-brain.
    """

    def test_metaculus_host(self):
        assert METACULUS_HOST == "metaculus.com"

    def test_mantic_host_and_the_urls_derived_from_it(self):
        assert MANTIC_HOST == "competitions.mantic.com"
        assert MANTIC_SITE_URL == "https://competitions.mantic.com"
        assert MANTIC_API_BASE_URL == "https://competitions.mantic.com/api"

    def test_question_platform_hosts_names_both_platforms(self):
        """The one list behind the publish-timeout scope, the self-reference refusal and the
        gap-fill v2 driver text; an added platform lands here and nowhere else."""
        assert QUESTION_PLATFORM_HOSTS == ("metaculus.com", "competitions.mantic.com")

    def test_the_bots_own_mantic_account_id(self):
        """``nostreambot-bot``, ``is_bot`` true, read off the public ``/api/users/81/`` on 2026-09-08."""
        assert MANTIC_BOT_USER_ID == 81
