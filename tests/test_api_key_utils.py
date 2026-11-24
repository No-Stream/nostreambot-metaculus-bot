from metaculus_bot.api_key_utils import get_openrouter_api_key


class TestApiKeyUtils:
    def test_openai_model_uses_special_key(self, monkeypatch):
        """OpenAI models should use OAI_ANTH_OPENROUTER_KEY when available."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("openrouter/openai/gpt-5")
        assert result == "special_key"

    def test_anthropic_model_uses_special_key(self, monkeypatch):
        """Anthropic models should use OAI_ANTH_OPENROUTER_KEY when available."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("openrouter/anthropic/claude-sonnet-4")
        assert result == "special_key"

    def test_google_flash_model_uses_special_key_when_donated_toggle_on(self, monkeypatch):
        """Flash Google models route via the donated key when GEMINI_USE_DONATED_OPENROUTER_KEY=true.

        Added to DONATED_KEY_PROVIDERS in task #12; later gated behind the
        donated-key toggle. gemini-3.1-pro is blocklisted (covered separately).
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")

        result = get_openrouter_api_key("openrouter/google/gemini-3.5-flash")
        assert result == "special_key"

    def test_google_pro_model_uses_general_key_even_when_toggle_on(self, monkeypatch):
        """gemini-3.1-pro is pinned to the personal key via DONATED_KEY_BLOCKED_GOOGLE_MODELS
        even with the donated toggle ON — no donated attempt, no 429, no fallback-
        counter bump. A flash model in the same env still uses the special key.

        Temporary workaround; see DONATED_KEY_BLOCKED_GOOGLE_MODELS
        (``TODO(gemini-3.1-pro-donated)``) and the retired design log (git history).
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")

        assert get_openrouter_api_key("openrouter/google/gemini-3.1-pro-preview") == "general_key"
        assert get_openrouter_api_key("openrouter/google/gemini-3.5-flash") == "special_key"

    def test_google_model_uses_general_key_when_donated_toggle_off(self, monkeypatch):
        """With the donated-key toggle off, Google calls use the general key."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "false")

        result = get_openrouter_api_key("openrouter/google/gemini-3.5-flash")
        assert result == "general_key"

    def test_google_flash_model_default_uses_special_key(self, monkeypatch):
        """Default (env var unset) is ON: flash Google models route via the donated (special) key.

        After Metaculus raised the Google rate limits (2026-06-16) the donated key
        serves most Gemini, so the default prefers it for flash models.
        gemini-3.1-pro stays blocklisted → general key (covered separately).
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")
        monkeypatch.delenv("GEMINI_USE_DONATED_OPENROUTER_KEY", raising=False)

        result = get_openrouter_api_key("openrouter/google/gemini-3.5-flash")
        assert result == "special_key"

    def test_non_donated_provider_uses_general_key(self, monkeypatch):
        """Providers NOT in DONATED_KEY_PROVIDERS (e.g. x-ai for Grok) use the general key."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("openrouter/x-ai/grok-4.1-fast")
        assert result == "general_key"

    def test_fallback_to_general_key_when_special_missing(self, monkeypatch):
        """Should fall back to general key when special key not available."""
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("openrouter/openai/gpt-5.1")
        assert result == "general_key"

    def test_non_openrouter_model_uses_general_key(self, monkeypatch):
        """Non-OpenRouter models should use general key."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("perplexity/sonar-reasoning-pro")
        assert result == "general_key"

    def test_returns_none_when_no_keys_available(self, monkeypatch):
        """Should return None when no API keys are available."""
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = get_openrouter_api_key("openrouter/openai/gpt-5.1")
        assert result is None

    def test_master_switch_off_returns_general_key_for_every_donated_provider(self, monkeypatch):
        """DONATED_OPENROUTER_KEY_ENABLED=false (a Mantic run): the donated key is never chosen,
        even with both keys set to distinct values and the Gemini toggle explicitly ON.

        Metaculus donated that key for its own tournaments; a run for another platform spends
        only the operator's personal key.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")
        monkeypatch.setenv("DONATED_OPENROUTER_KEY_ENABLED", "false")

        assert get_openrouter_api_key("openrouter/openai/gpt-5.6-sol") == "general_key"
        assert get_openrouter_api_key("openrouter/anthropic/claude-opus-4.8") == "general_key"
        assert get_openrouter_api_key("openrouter/google/gemini-3.5-flash") == "general_key"

    def test_master_switch_unset_keeps_special_key(self, monkeypatch):
        """Default ON: a Metaculus run that never sets the switch still prefers the donated key."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")
        monkeypatch.delenv("DONATED_OPENROUTER_KEY_ENABLED", raising=False)

        assert get_openrouter_api_key("openrouter/openai/gpt-5.6-sol") == "special_key"
