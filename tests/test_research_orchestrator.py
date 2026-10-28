"""Tests for the extracted ResearchOrchestrator.

Verifies that the orchestrator:
1. Delegates to providers correctly
2. Handles caching (benchmarking mode)
3. Runs gap-fill when enabled
4. Combines parallel provider results with headers
5. Falls back gracefully on provider failure
6. Exposes provider_failure_count for alerting
"""

import asyncio
import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import litellm.exceptions as litellm_exc
import pytest
from forecasting_tools import GeneralLlm

from metaculus_bot.constants import GAP_FILL_MIN_RESEARCH_CHARS
from metaculus_bot.prompts import SUMMARIZER_SOFT_FAIL_BANNER
from metaculus_bot.research.asknews_summarization import summarize_asknews
from metaculus_bot.research.gap_fill_stages import run_gap_fill_passes
from metaculus_bot.research.orchestrator import ResearchOrchestrator
from metaculus_bot.research.provider_diagnostics import (
    format_provider_diagnostics_block,
    pop_provider_detail,
    record_provider_detail,
)
from scripts.telemetry.markers import MARKER_SPECS


@pytest.fixture
def mock_llm() -> GeneralLlm:
    return GeneralLlm(model="test/model", temperature=0.0)


@pytest.fixture
def question() -> MagicMock:
    q = MagicMock()
    q.id_of_question = 42
    q.question_text = "Will X happen by 2027?"
    q.page_url = "https://metaculus.com/questions/42"
    q.resolution_criteria = "Resolves YES if X happens"
    q.fine_print = ""
    q.open_time = datetime(2026, 3, 15)
    return q


@pytest.fixture
def orchestrator(mock_llm: GeneralLlm) -> ResearchOrchestrator:
    return ResearchOrchestrator(
        default_llm=mock_llm,
        summarizer_llm=mock_llm,
    )


class TestRunResearch:
    @pytest.mark.asyncio
    async def test_calls_providers_and_returns_combined_result(self, orchestrator, question):
        provider_a = AsyncMock(return_value="Result from provider A")
        provider_b = AsyncMock(return_value="Result from provider B")

        with (
            patch.object(
                orchestrator,
                "_select_research_providers",
                return_value=[(provider_a, "asknews"), (provider_b, "native_search")],
            ),
            # AskNews is summarized inline; the briefing replaces the raw articles.
            patch.object(
                orchestrator._summarizer_llm,
                "invoke",
                new_callable=AsyncMock,
                return_value="AskNews briefing",
            ),
        ):
            result = await orchestrator.run_research(question)

        assert "AskNews briefing" in result  # AskNews summarized
        assert "Result from provider A" not in result  # raw articles replaced
        assert "Result from provider B" in result  # native search raw
        assert "## News Articles (AskNews)" in result
        assert "## Web Research (Native Search)" in result

    @pytest.mark.asyncio
    async def test_uses_cache_in_benchmarking_mode(self, mock_llm, question):
        cache = {42: "cached research"}
        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            is_benchmarking=True,
            research_cache=cache,
        )

        result = await orch.run_research(question)
        assert result == "cached research"

    @pytest.mark.asyncio
    async def test_stores_to_cache_in_benchmarking_mode(self, mock_llm, question):
        cache: dict[int, str] = {}
        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            is_benchmarking=True,
            research_cache=cache,
        )

        provider = AsyncMock(return_value="fresh research")

        with patch.object(orch, "_select_research_providers", return_value=[(provider, "custom")]):
            result = await orch.run_research(question)

        assert result  # non-empty
        assert cache[42]  # stored

    @pytest.mark.asyncio
    async def test_gap_fill_appended_when_enabled(self, orchestrator, question, monkeypatch):
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

        provider = AsyncMock(return_value="A" * 200)  # exceeds GAP_FILL_MIN_RESEARCH_CHARS

        with (
            patch.object(orchestrator, "_select_research_providers", return_value=[(provider, "custom")]),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass",
                new_callable=AsyncMock,
                return_value="gap fill addendum",
            ) as mock_gap_fill,
        ):
            result = await orchestrator.run_research(question)

        assert "gap fill addendum" in result
        assert "Targeted Gap-Fill" in result
        mock_gap_fill.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_provider_failure_count_incremented_on_non_timeout_failure(self, orchestrator, question):
        """The counter means "provider failures", NOT timeouts.

        The 2026-07-26 run reported ``research_provider_timeouts=1`` for a hard 403
        that failed in 137ms, while the provider-diagnostics line in the same log
        correctly said ``errored | APIError``. The generic failure branch never
        inspected the exception type, so the identifier was always a misnomer.
        """
        failing = AsyncMock(side_effect=RuntimeError('APIError: {"code":403} Key limit exceeded (total limit)'))
        working = AsyncMock(return_value="ok")

        with patch.object(
            orchestrator,
            "_select_research_providers",
            return_value=[(failing, "native_search"), (working, "custom")],
        ):
            result = await orchestrator.run_research(question)

        assert orchestrator.provider_failure_count == 1
        assert "ok" in result

    @pytest.mark.asyncio
    async def test_provider_failure_count_also_counts_genuine_timeouts(self, orchestrator, question):
        """A real timeout is a provider failure too — the rename widens the name, not the set."""
        failing = AsyncMock(side_effect=TimeoutError())
        working = AsyncMock(return_value="ok")

        with patch.object(
            orchestrator,
            "_select_research_providers",
            return_value=[(failing, "native_search"), (working, "custom")],
        ):
            await orchestrator.run_research(question)

        assert orchestrator.provider_failure_count == 1


class TestAskNewsSummarization:
    @pytest.mark.asyncio
    async def test_invokes_summarizer_llm(self, orchestrator, question):
        with patch.object(orchestrator._summarizer_llm, "invoke", new_callable=AsyncMock, return_value="summary"):
            result = await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert result == "summary"

    @pytest.mark.asyncio
    async def test_prompt_carries_open_date_and_window_stamping_rules(self, orchestrator, question):
        """Summarizer hardening (2026-07-08): the prompt must (a) state the
        question's open date, (b) demand precise dating with a targeted
        PRE-WINDOW flag only for facts that could otherwise be misread as
        already satisfying the criteria (no blanket IN-WINDOW stamps), and
        (c) carry the single-source rule."""
        with patch.object(
            orchestrator._summarizer_llm, "invoke", new_callable=AsyncMock, return_value="summary"
        ) as invoke:
            await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert invoke.await_args is not None
        prompt = invoke.await_args.args[0]
        # Collapse whitespace so assertions don't depend on where clean_indents wraps lines.
        collapsed = " ".join(prompt.split())
        # (a) Open date threaded from the question object (fixture: 2026-03-15).
        assert "2026-03-15" in prompt
        assert "opened on 2026-03-15" in collapsed
        # (b) Lighter window-stamping: precise dating, targeted PRE-WINDOW label
        #     only for facts that could look like they satisfy the criteria; no
        #     blanket IN-WINDOW stamps. Full tag on first occurrence, short
        #     "[PRE-WINDOW]" tag on repeats (display compression only).
        assert "Date every fact precisely" in collapsed
        assert "[PRE-WINDOW — occurred before question open, cannot itself satisfy the criteria]" in collapsed
        assert "FIRST time such a flag appears in the briefing, use the full tag" in collapsed
        assert 'for every subsequent occurrence use the short tag "[PRE-WINDOW]"' in collapsed
        assert "[IN-WINDOW]" not in collapsed
        assert "base-rate" in collapsed
        # (c) Single-source rule: label + carry hedges + no promotion to confirmed.
        assert "[SINGLE-SOURCE]" in collapsed
        assert "hedges" in collapsed
        assert "NEVER promote a single-source claim to a confirmed or factual statement" in collapsed
        # (d) No-forecast rule (bench: sol-low appended its own probability table,
        #     which would anchor the six downstream forecasters): evidence only.
        assert "NEVER include your own forecast, probability estimate, or probability distribution" in collapsed

    @pytest.mark.asyncio
    async def test_prompt_carries_the_mc_ballot(self, orchestrator, question):
        """The summarizer's relevance screen needs the candidate names: on q44952 (World
        Yo-Yo champion, MC) no research stage ever saw the ballot, and AskNews returned
        zero mentions of the eventual winner even though the options named him."""
        question.options = ["Mir Kim", "Hunter Feuerstein", "Other"]
        with patch.object(
            orchestrator._summarizer_llm, "invoke", new_callable=AsyncMock, return_value="summary"
        ) as invoke:
            await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert invoke.await_args is not None
        prompt = invoke.await_args.args[0]
        assert "Options (in resolution order): Mir Kim | Hunter Feuerstein | Other" in prompt

    @pytest.mark.asyncio
    async def test_missing_open_time_asserts(self, orchestrator, question):
        """Missing open_time is a data bug — fail loudly (matches the
        _forecasting_window_str contract), never silently skip stamping."""
        question.open_time = None
        with pytest.raises(AssertionError):
            await orchestrator._summarize_asknews(question, "raw asknews articles")

    @pytest.mark.asyncio
    async def test_empty_research_skips_summarizer(self, orchestrator, question):
        with patch.object(orchestrator._summarizer_llm, "invoke", new_callable=AsyncMock) as invoke:
            result = await orchestrator._summarize_asknews(question, "   ")

        assert result == "   "
        invoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_soft_falls_back_to_raw_on_transient_summarizer_error(self, orchestrator, question):
        """Transient LLM-provider errors (timeouts, API hiccups) soft-fail to the raw articles.

        Round-2: the summarizer invoke is wrapped in the broad 30s-gated retry, which
        retries this fast asyncio.TimeoutError the full backoff schedule before it
        surfaces and is caught by _SUMMARIZER_TRANSIENT_EXCEPTIONS → raw articles.
        asyncio.sleep is patched so the backoffs are instant.
        """
        with (
            patch("metaculus_bot.llm_retry.asyncio.sleep", new=AsyncMock()),
            patch.object(
                orchestrator._summarizer_llm,
                "invoke",
                new_callable=AsyncMock,
                side_effect=TimeoutError("summarizer timed out"),
            ),
        ):
            result = await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert "raw asknews articles" in result
        assert SUMMARIZER_SOFT_FAIL_BANNER in result

    @pytest.mark.asyncio
    async def test_non_transient_summarizer_error_propagates(self, orchestrator, question):
        """A genuine bug must ultimately crash, not silently degrade.

        Round-2: the summarizer invoke is now wrapped in the broad 30s-gated retry,
        which retries an AttributeError (not in PERMANENT_NO_RETRY_EXCEPTIONS) a few
        times before exhausting and propagating. The key contract is unchanged —
        the bug surfaces rather than being swallowed into a soft-fall-back to raw
        articles (which the _SUMMARIZER_TRANSIENT_EXCEPTIONS catch would NOT do for
        AttributeError anyway). asyncio.sleep is patched so the backoffs are instant.
        """
        with (
            patch("metaculus_bot.llm_retry.asyncio.sleep", new=AsyncMock()),
            patch.object(
                orchestrator._summarizer_llm,
                "invoke",
                new_callable=AsyncMock,
                side_effect=AttributeError("prompt builder bug"),
            ),
            pytest.raises(AttributeError),
        ):
            await orchestrator._summarize_asknews(question, "raw asknews articles")

    @pytest.mark.asyncio
    async def test_fast_blip_on_summarizer_invoke_retries_then_succeeds(self, orchestrator, question):
        """A fast litellm.Timeout on the summarizer invoke is retried; the next call wins.

        Locks the Round-2 broad-retry wiring: SUMMARIZER_LLM is allowed_tries=1, so
        this gated wrapper is its sole retry layer.
        """
        import litellm.exceptions as litellm_exc

        invoke = AsyncMock(side_effect=[litellm_exc.Timeout("blip", model="m", llm_provider="openrouter"), "summary"])
        with (
            patch("metaculus_bot.llm_retry.asyncio.sleep", new=AsyncMock()),
            patch.object(orchestrator._summarizer_llm, "invoke", new=invoke),
        ):
            result = await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert result == "summary"
        assert invoke.await_count == 2

    @pytest.mark.asyncio
    async def test_slow_summarizer_failure_not_retried_soft_falls_to_raw(self, orchestrator, question):
        """A summarizer failure past the 30s gate is NOT retried; it soft-falls to raw articles.

        A slow litellm.Timeout is a transient-type exception that _SUMMARIZER_TRANSIENT_
        EXCEPTIONS catches, so after the gate blocks the retry the orchestrator returns
        the raw articles (one invoke, no deadline-risking re-fire).
        """
        import litellm.exceptions as litellm_exc

        from metaculus_bot.llm_retry import TRANSIENT_RETRY_MAX_ELAPSED_S

        invoke = AsyncMock(side_effect=litellm_exc.Timeout("stall", model="m", llm_provider="openrouter"))
        clock = iter([0.0] + [TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0] * 20)

        with (
            patch("metaculus_bot.llm_retry.time.monotonic", lambda: next(clock)),
            patch.object(orchestrator._summarizer_llm, "invoke", new=invoke),
        ):
            result = await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert "raw asknews articles" in result
        assert SUMMARIZER_SOFT_FAIL_BANNER in result
        assert invoke.await_count == 1

    @pytest.mark.asyncio
    async def test_soft_fail_banners_the_bundle_and_counts(self, orchestrator, question):
        """A summarizer soft-fail must be VISIBLE to the forecaster and to CI.

        The 2026-07-26 run shipped 12 raw unsummarized AskNews articles with none of
        the 2026-07-18 audit rules applied (per-article relevance gate, recency-first
        ordering, supersession arithmetic, evidence-age opener, [PRE-WINDOW] labeling)
        and nothing anywhere said so: no counter, and AskNews still reported
        ``status=ok`` because provider status is computed from POST-summarizer text.
        """
        with (
            patch("metaculus_bot.llm_retry.asyncio.sleep", new=AsyncMock()),
            patch.object(
                orchestrator._summarizer_llm,
                "invoke",
                new_callable=AsyncMock,
                side_effect=TimeoutError("summarizer timed out"),
            ),
        ):
            result = await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert orchestrator.summarizer_failure_count == 1
        # Banner leads the section so it is read before the ungated articles.
        assert result.startswith(SUMMARIZER_SOFT_FAIL_BANNER)
        assert "raw asknews articles" in result
        # Not a markdown h1/h2: _demote_inner_headings would shift it and the
        # framework's section renormalization would mangle the provider header.
        assert not result.lstrip().startswith("#")

    @pytest.mark.asyncio
    async def test_blank_summary_also_banners_and_counts(self, orchestrator, question):
        """A blank summary degrades the briefing exactly as an exception does."""
        with patch.object(orchestrator._summarizer_llm, "invoke", new_callable=AsyncMock, return_value="   "):
            result = await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert orchestrator.summarizer_failure_count == 1
        assert SUMMARIZER_SOFT_FAIL_BANNER in result
        assert "raw asknews articles" in result

    @pytest.mark.asyncio
    async def test_successful_summary_has_no_banner_and_no_count(self, orchestrator, question):
        with patch.object(orchestrator._summarizer_llm, "invoke", new_callable=AsyncMock, return_value="briefing"):
            result = await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert result == "briefing"
        assert orchestrator.summarizer_failure_count == 0

    @pytest.mark.asyncio
    async def test_empty_research_does_not_count_as_summarizer_failure(self, orchestrator, question):
        """No articles to summarize is not a summarizer failure — nothing was lost."""
        with patch.object(orchestrator._summarizer_llm, "invoke", new_callable=AsyncMock):
            await orchestrator._summarize_asknews(question, "   ")

        assert orchestrator.summarizer_failure_count == 0

    @pytest.mark.asyncio
    async def test_soft_fail_surfaces_as_asknews_source_loss_in_diagnostics(self, mock_llm, question):
        """AskNews reads ``ok`` (post-summarizer text is non-empty), so the loss has to
        ride the per-source details seam or it stays invisible in comment + archive."""
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        asknews = AsyncMock(return_value="raw asknews articles")

        with (
            patch("metaculus_bot.llm_retry.asyncio.sleep", new=AsyncMock()),
            patch.object(
                orch._summarizer_llm,
                "invoke",
                new_callable=AsyncMock,
                side_effect=TimeoutError("summarizer timed out"),
            ),
        ):
            _, results, _ = await orch._run_providers_parallel(question, [(asknews, "asknews")])

        assert results[0].status == "ok"
        assert results[0].details["sources"]["summarizer"].startswith("error(")
        rendered = format_provider_diagnostics_block(results)
        assert "lost=summarizer:error(" in rendered

    @pytest.mark.asyncio
    async def test_only_asknews_is_summarized(self, orchestrator, question):
        """AskNews output is summarized; every other provider passes through raw."""
        asknews = AsyncMock(return_value="raw asknews articles")
        native = AsyncMock(return_value="native search prose")

        with (
            patch.object(
                orchestrator,
                "_select_research_providers",
                return_value=[(asknews, "asknews"), (native, "native_search")],
            ),
            patch.object(
                orchestrator._summarizer_llm,
                "invoke",
                new_callable=AsyncMock,
                return_value="ASKNEWS BRIEFING",
            ) as invoke,
        ):
            result = await orchestrator.run_research(question)

        # Summarizer ran exactly once, and on the AskNews payload specifically.
        invoke.assert_awaited_once()
        assert invoke.await_args is not None
        assert "raw asknews articles" in invoke.await_args.args[0]
        assert "ASKNEWS BRIEFING" in result
        assert "raw asknews articles" not in result
        # Native search prose is delivered verbatim.
        assert "native search prose" in result

    @pytest.mark.asyncio
    async def test_asknews_failure_keeps_other_research_without_fallback(self, mock_llm, question, monkeypatch):
        for key in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "OPENROUTER_API_KEY", "PERPLEXITY_API_KEY", "EXA_API_KEY"):
            monkeypatch.setenv(key, "fake-key")
        for key in (
            "NATIVE_SEARCH_ENABLED",
            "GEMINI_SEARCH_ENABLED",
            "PREDICTION_MARKETS_ENABLED",
            "GAP_FILL_ENABLED",
            "GAP_FILL_V2_ENABLED",
        ):
            monkeypatch.setenv(key, "true")
        for key in ("FINANCIAL_DATA_ENABLED", "TS_ANCHOR_ENABLED", "RESOLUTION_SOURCE_ENABLED"):
            monkeypatch.setenv(key, "false")

        captured: dict = {}
        orch = ResearchOrchestrator(
            default_llm=mock_llm, summarizer_llm=mock_llm, research_sink=lambda **kw: captured.update(kw)
        )
        asknews = AsyncMock(side_effect=RuntimeError("asknews down"))
        native = AsyncMock(return_value="native search prose " * 20)
        gemini = AsyncMock(return_value="gemini search prose")
        markets = AsyncMock(return_value="market research prose")
        with (
            patch("metaculus_bot.research.providers._asknews_provider", return_value=asknews),
            patch("metaculus_bot.research.orchestrator.native_search_provider", return_value=native),
            patch("metaculus_bot.research.gemini_search.gemini_search_provider", return_value=gemini),
            patch("metaculus_bot.research.prediction_market.prediction_market_provider", return_value=markets),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass",
                new_callable=AsyncMock,
                return_value="targeted findings",
            ) as targeted,
            patch(
                "metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2",
                new_callable=AsyncMock,
                return_value="## Agentic Research Findings\n\nagentic findings",
            ) as agentic,
            patch.object(mock_llm, "invoke", new_callable=AsyncMock) as summarize,
        ):
            text = await orch.run_research(question)

        assert captured["providers_attempted"] == ["asknews", "native_search", "gemini_search", "prediction_market"]
        assert captured["providers_succeeded"] == ["native_search", "gemini_search", "prediction_market"]
        assert {result["name"]: result["status"] for result in captured["provider_results"]} == {
            "asknews": "errored",
            "native_search": "ok",
            "gemini_search": "ok",
            "prediction_market": "ok",
        }
        for expected in (
            "native search prose",
            "gemini search prose",
            "market research prose",
            "targeted findings",
            "agentic findings",
        ):
            assert expected in text
        assert "News Articles (AskNews)" not in text
        assert orch.provider_failure_count == 1
        assert "asknews: errored" in captured["provider_diagnostics_block"]
        assert captured["asknews_raw"] == ""
        assert not hasattr(orch, "_attempt_research_fallback")
        asknews.assert_awaited_once()
        targeted.assert_awaited_once()
        agentic.assert_awaited_once()
        summarize.assert_not_awaited()


class TestProviderSelection:
    def test_returns_empty_stub_when_no_providers_configured(self, mock_llm, monkeypatch):
        monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
        monkeypatch.delenv("ASKNEWS_SECRET", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")
        monkeypatch.setenv("GEMINI_SEARCH_ENABLED", "false")
        monkeypatch.setenv("FINANCIAL_DATA_ENABLED", "false")
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "false")
        monkeypatch.delenv("RESOLUTION_SOURCE_ENABLED", raising=False)
        monkeypatch.delenv("RESEARCH_PROVIDER", raising=False)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
        )
        providers = orch._select_research_providers()
        assert len(providers) == 1
        assert providers[0][1] == "none"

    def test_includes_resolution_source_when_enabled(self, mock_llm, monkeypatch):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")
        monkeypatch.setenv("GEMINI_SEARCH_ENABLED", "false")
        monkeypatch.setenv("FINANCIAL_DATA_ENABLED", "false")
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "false")
        monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
        monkeypatch.delenv("ASKNEWS_SECRET", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("RESEARCH_PROVIDER", raising=False)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
        )
        providers = orch._select_research_providers()
        names = [n for _, n in providers]
        assert "resolution_source" in names

    @pytest.mark.parametrize("fast_path", [True, False])
    def test_resolution_source_learns_about_the_fast_path(self, mock_llm, monkeypatch, fast_path):
        """The provider's two EXPENSIVE rungs (the browser, the paid reader) decline on the fast
        path, and the factory call here is the only place the flag can reach it."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")
        monkeypatch.setenv("GEMINI_SEARCH_ENABLED", "false")
        monkeypatch.setenv("FINANCIAL_DATA_ENABLED", "false")
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "false")
        monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
        monkeypatch.delenv("ASKNEWS_SECRET", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("RESEARCH_PROVIDER", raising=False)
        factory = MagicMock(return_value=AsyncMock(return_value=""))

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        with patch("metaculus_bot.research.resolution_source.resolution_source_provider", factory):
            providers = orch._select_research_providers(fast_path=fast_path)

        assert "resolution_source" in [n for _, n in providers]
        factory.assert_called_once_with(is_benchmarking=False, fast_path=fast_path)

    def test_excludes_resolution_source_when_flag_unset(self, mock_llm, monkeypatch):
        monkeypatch.delenv("RESOLUTION_SOURCE_ENABLED", raising=False)
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")
        monkeypatch.setenv("GEMINI_SEARCH_ENABLED", "false")
        monkeypatch.setenv("FINANCIAL_DATA_ENABLED", "false")
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "false")
        monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
        monkeypatch.delenv("ASKNEWS_SECRET", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("RESEARCH_PROVIDER", raising=False)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
        )
        providers = orch._select_research_providers()
        names = [n for _, n in providers]
        assert "resolution_source" not in names

    def test_resolution_source_body_composes_with_header_and_demoted_headings(self):
        """Body written by the resolution_source provider composes with the
        orchestrator header machinery: h2 provider header sits above the body,
        and any in-body h1/h2 subheadings are demoted so the whole snapshot is
        a well-formed section within the combined research doc.
        """
        from metaculus_bot.research.orchestrator import _demote_inner_headings
        from metaculus_bot.research.section_format import provider_header

        body = (
            "Caveat: only reproduces text as fetched; forecaster must judge relevance.\n\n"
            "### https://example.com/data\n"
            "Some extracted content.\n"
        )
        # Simulate assemble_provider_sections' composition step (section_format.py).
        header = provider_header("resolution_source")
        composed = f"{header}\n{_demote_inner_headings(body)}"

        assert "## Resolution Source Snapshot" in composed
        # The h3 "### https://example.com/data" is untouched by the demoter
        # (which only shifts h1/h2), so it stays h3 under the h2 header. If
        # future author writes it as h1/h2 the demoter shifts it to h3/h4.
        assert composed.index("## Resolution Source Snapshot") < composed.index("### https://example.com/data")

    def test_includes_native_search_when_enabled(self, mock_llm, monkeypatch):
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "true")
        monkeypatch.setenv("NATIVE_SEARCH_MODEL", "openai/gpt-5.5")
        monkeypatch.setenv("FINANCIAL_DATA_ENABLED", "false")
        monkeypatch.setenv("GEMINI_SEARCH_ENABLED", "false")
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "false")
        monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
        monkeypatch.delenv("ASKNEWS_SECRET", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("RESEARCH_PROVIDER", raising=False)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
        )
        providers = orch._select_research_providers()
        names = [n for _, n in providers]
        assert "native_search" in names


class TestConcurrencyLimiter:
    @pytest.mark.asyncio
    async def test_limits_concurrent_research_calls(self, mock_llm, question):
        max_concurrent = 2
        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            max_concurrent_research=max_concurrent,
        )
        active = {"count": 0, "max_seen": 0}
        lock = asyncio.Lock()

        async def slow_provider(q):
            async with lock:
                active["count"] += 1
                active["max_seen"] = max(active["max_seen"], active["count"])
            await asyncio.sleep(0.05)
            async with lock:
                active["count"] -= 1
            return "done"

        with patch.object(orch, "_select_research_providers", return_value=[(slow_provider, "custom")]):
            results = await asyncio.gather(
                orch.run_research(question),
                orch.run_research(question),
                orch.run_research(question),
                orch.run_research(question),
            )

        assert all("done" in r for r in results)
        assert active["max_seen"] <= max_concurrent


class TestIntegrationWithTemplateForecaster:
    """Verify that TemplateForecaster delegates to ResearchOrchestrator correctly."""

    @pytest.mark.asyncio
    async def test_bot_run_research_delegates_to_orchestrator(self, question):
        from main import TemplateForecaster
        from metaculus_bot.aggregation_strategies import AggregationStrategy

        sentinel = GeneralLlm(model="test/sentinel", temperature=0.0)
        bot = TemplateForecaster(
            llms={"default": sentinel, "parser": sentinel, "researcher": sentinel, "summarizer": sentinel},
            aggregation_strategy=AggregationStrategy.MEAN,
        )

        mock_provider = AsyncMock(return_value="delegated research")

        with patch.object(bot._research, "_select_research_providers", return_value=[(mock_provider, "custom")]):
            result = await bot.run_research(question)

        assert "delegated research" in result

    @pytest.mark.asyncio
    async def test_bot_run_research_patchable_at_bot_level(self, question):
        """Existing tests patch bot.run_research directly; this must still work."""
        from main import TemplateForecaster
        from metaculus_bot.aggregation_strategies import AggregationStrategy

        sentinel = GeneralLlm(model="test/sentinel", temperature=0.0)
        bot = TemplateForecaster(
            llms={"default": sentinel, "parser": sentinel, "researcher": sentinel, "summarizer": sentinel},
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        bot.run_research = AsyncMock(return_value="patched result")  # ty: ignore[invalid-assignment]
        result = await bot.run_research(question)
        assert result == "patched result"

    def test_provider_failure_count_exposed_via_bot(self, question):
        """Bot exposes the provider-failure count from the orchestrator for alerting."""
        from main import TemplateForecaster
        from metaculus_bot.aggregation_strategies import AggregationStrategy

        sentinel = GeneralLlm(model="test/sentinel", temperature=0.0)
        bot = TemplateForecaster(
            llms={"default": sentinel, "parser": sentinel, "researcher": sentinel, "summarizer": sentinel},
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        bot._research.provider_failure_count = 5
        assert bot._research_provider_failure_count == 5

    @pytest.mark.asyncio
    async def test_summarizer_soft_fail_reaches_bot_alertable_count(self, question):
        """End-to-end wiring for the summarizer counter: a real soft-fail inside the
        orchestrator has to reach the number cli.py exits on."""
        from main import TemplateForecaster
        from metaculus_bot.aggregation_strategies import AggregationStrategy

        sentinel = GeneralLlm(model="test/sentinel", temperature=0.0)
        bot = TemplateForecaster(
            llms={"default": sentinel, "parser": sentinel, "researcher": sentinel, "summarizer": sentinel},
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        assert bot.alertable_count == 0

        with (
            patch("metaculus_bot.llm_retry.asyncio.sleep", new=AsyncMock()),
            patch.object(
                bot._research._summarizer_llm,
                "invoke",
                new_callable=AsyncMock,
                side_effect=TimeoutError("summarizer timed out"),
            ),
        ):
            await bot._research._summarize_asknews(question, "raw asknews articles")

        assert bot._summarizer_failure_count == 1
        assert bot.alertable_count == 1


class TestProviderDiagnosticsCapture:
    """Per-provider observability: _run_providers_parallel returns a ProviderResult list
    alongside the combined text, and run_research appends a diagnostics block."""

    @pytest.mark.asyncio
    async def test_status_ok_for_nonempty_provider(self, mock_llm, question):
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        provider = AsyncMock(return_value="some research prose")

        _, results, _ = await orch._run_providers_parallel(question, [(provider, "native_search")])

        assert len(results) == 1
        assert results[0].name == "native_search"
        assert results[0].status == "ok"
        assert results[0].chars == len("some research prose")
        assert isinstance(results[0].latency_ms, int)
        assert results[0].latency_ms >= 0
        assert results[0].error_type is None
        assert results[0].error_message is None
        assert results[0].details == {}

    @pytest.mark.asyncio
    async def test_provider_detail_drained_into_result(self, mock_llm, question):
        """A provider's recorded per-source detail is drained into its ProviderResult.

        This is the seam that makes partial degradation visible: the provider knows
        its per-source outcome but can only return a str, so it records to the
        (qid, provider) registry and _run_one pops it into details."""
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)

        async def provider(q):
            record_provider_detail(
                q.id_of_question,
                "prediction_market",
                {"sources": {"kalshi": "dropped(size_cap)", "polymarket": "ok(2)"}},
            )
            return "prediction market snapshot prose"

        _, results, _ = await orch._run_providers_parallel(question, [(provider, "prediction_market")])

        assert results[0].status == "ok"
        assert results[0].details == {"sources": {"kalshi": "dropped(size_cap)", "polymarket": "ok(2)"}}
        # Registry drained — nothing leaks into a subsequent same-key call.
        assert pop_provider_detail(question.id_of_question, "prediction_market") == {}

    @pytest.mark.asyncio
    async def test_registry_drained_even_when_provider_errors(self, mock_llm, question):
        """If a provider records detail then raises, _run_one still drains the registry
        (so it can't leak into a later call) — the errored result carries the error, not details."""
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)

        async def provider(q):
            record_provider_detail(q.id_of_question, "resolution_source", {"sources": {"a.gov": "blocked"}})
            raise RuntimeError("boom after recording")

        _, results, _ = await orch._run_providers_parallel(question, [(provider, "resolution_source")])

        assert results[0].status == "errored"
        assert pop_provider_detail(question.id_of_question, "resolution_source") == {}

    @pytest.mark.asyncio
    async def test_partial_loss_reaches_comment_block_not_research_text(self, mock_llm, question):
        """The partial-loss signal rides the stashed diagnostics block (comment path) only —
        it must never reach the forecaster-facing research text. The diagnostics seam that
        withholds the whole block from research holds for the per-source detail too."""
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)

        async def provider(q):
            record_provider_detail(
                q.id_of_question,
                "prediction_market",
                {"sources": {"kalshi": "dropped(size_cap)", "polymarket": "ok(1)"}},
            )
            return "prediction market snapshot prose"

        with patch.object(orch, "_select_research_providers", return_value=[(provider, "prediction_market")]):
            research = await orch.run_research(question)

        # Forecasters never see the degradation marker.
        assert "lost=" not in research
        assert "dropped(size_cap)" not in research
        # But it IS in the stashed comment block, visible at a glance.
        block = orch.pop_provider_diagnostics(question.id_of_question)
        assert "kalshi:dropped(size_cap)" in block
        assert "sources=1/2" in block

    @pytest.mark.asyncio
    async def test_status_empty_for_blank_provider(self, mock_llm, question):
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        provider = AsyncMock(return_value="   ")

        _, results, _ = await orch._run_providers_parallel(question, [(provider, "native_search")])

        assert results[0].status == "empty"
        assert results[0].chars == 0

    @pytest.mark.asyncio
    async def test_status_errored_for_raising_provider(self, mock_llm, question):
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        provider = AsyncMock(side_effect=RuntimeError("boom"))

        _, results, _ = await orch._run_providers_parallel(question, [(provider, "native_search")])

        assert results[0].status == "errored"
        assert results[0].chars == 0
        assert results[0].error_type == "RuntimeError"
        assert results[0].error_message is not None
        assert "boom" in results[0].error_message
        assert orch.provider_failure_count == 1

    @pytest.mark.asyncio
    async def test_raising_provider_emits_a_parseable_failure_marker(self, mock_llm, question, caplog):
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        overloaded = litellm_exc.APIError(
            status_code=502, message="overloaded", llm_provider="openrouter", model="openai/gpt-6.1-sol"
        )
        provider = AsyncMock(side_effect=overloaded)

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.orchestrator"):
            await orch._run_providers_parallel(question, [(provider, "native_search")])

        (line,) = [r.message for r in caplog.records if r.message.startswith("RESEARCH_PROVIDER_FAILED:")]
        spec = next(spec for spec in MARKER_SPECS if spec.name == "research_provider_failed")
        match = spec.regex.search(line)
        assert match is not None, line
        assert match["question"] == "42"
        assert match["provider"] == "native_search"
        assert match["error"] == "APIError"
        assert match["status"] == "502"

    @pytest.mark.asyncio
    async def test_status_inactive_for_asknews_subscription_error(self, mock_llm, question):
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)

        class ForbiddenError(Exception):
            pass

        provider = AsyncMock(side_effect=ForbiddenError("403011 - subscription is not currently active"))

        _, results, _ = await orch._run_providers_parallel(question, [(provider, "asknews")])

        assert results[0].status == "inactive"
        assert results[0].chars == 0
        assert results[0].error_type == "ForbiddenError"
        # Off-season is expected, not an alertable failure.
        assert orch.provider_failure_count == 0

    @pytest.mark.asyncio
    async def test_fallback_provider_is_none_on_the_normal_path(self, mock_llm, question):
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        provider = AsyncMock(return_value="native prose")

        _, results, _ = await orch._run_providers_parallel(question, [(provider, "native_search")])

        assert results[0].status == "ok"
        assert results[0].fallback_provider is None

    @pytest.mark.asyncio
    async def test_one_result_per_provider_mixed_statuses(self, mock_llm, question):
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        ok = AsyncMock(return_value="native prose")
        failing = AsyncMock(side_effect=RuntimeError("timeout"))

        _, results, _ = await orch._run_providers_parallel(
            question, [(ok, "native_search"), (failing, "gemini_search")]
        )

        by_name = {r.name: r for r in results}
        assert by_name["native_search"].status == "ok"
        assert by_name["gemini_search"].status == "errored"

    @pytest.mark.asyncio
    async def test_diagnostics_block_withheld_from_research_but_logged_and_poppable(self, mock_llm, question, caplog):
        """Diagnostics seam: run_research output (the forecaster-facing text) must NOT
        carry the block, but it must still be (a) logged at INFO and (b) stashed for the
        comment path via pop_provider_diagnostics."""

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        ok = AsyncMock(return_value="native prose")
        failing = AsyncMock(side_effect=RuntimeError("timeout"))

        with (
            patch.object(
                orch,
                "_select_research_providers",
                return_value=[(ok, "native_search"), (failing, "gemini_search")],
            ),
            caplog.at_level(logging.INFO),
        ):
            research = await orch.run_research(question)

        # Forecaster-facing research is clean.
        assert "## Provider Diagnostics" not in research
        assert "native prose" in research

        # Logged at INFO for run-log triage.
        diag_log = next(rec.message for rec in caplog.records if "Provider diagnostics for URL" in rec.message)
        assert "## Provider Diagnostics" in diag_log

        # Stashed for the comment path, keyed by qid.
        block = orch.pop_provider_diagnostics(question.id_of_question)
        assert "## Provider Diagnostics" in block
        diag_lines = [line for line in block.splitlines() if line.startswith("- ")]
        assert any(line.startswith("- native_search: ok |") for line in diag_lines)
        errored_line = next(line for line in diag_lines if line.startswith("- gemini_search: errored |"))
        assert "RuntimeError" in errored_line

        # Pop semantics: a second pop returns empty (stash must not grow across a batch).
        assert orch.pop_provider_diagnostics(question.id_of_question) == ""
        assert orch.pop_provider_diagnostics(None) == ""

    @pytest.mark.asyncio
    async def test_gap_fill_research_stays_clean_of_diagnostics(self, mock_llm, question, monkeypatch):
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        provider = AsyncMock(return_value="A" * 200)

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass",
                new_callable=AsyncMock,
                return_value="gap fill addendum",
            ),
        ):
            research = await orch.run_research(question)

        # Gap-fill addendum lands in the forecaster-facing text; diagnostics do not.
        assert "Targeted Gap-Fill" in research
        assert "## Provider Diagnostics" not in research
        assert "## Provider Diagnostics" in orch.pop_provider_diagnostics(question.id_of_question)

    @pytest.mark.asyncio
    async def test_sink_receives_attempted_and_succeeded_excluding_failures(self, mock_llm, question, monkeypatch):
        """End-to-end guard for the schema-v2 derivation feeding the research sink.

        ``providers_attempted`` is every selected provider in order; ``providers_succeeded``
        is ONLY the ok/fallback ones. This pins the exclusion of ``empty``, ``errored``, AND —
        most importantly — ``inactive`` (AskNews off-season), so a regression that derived
        ``providers_succeeded`` from the attempted list, or included ``inactive``, would fail here.
        The AskNews subscription error is classified as ``inactive``.
        """
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)

        captured: dict = {}

        def sink(**kwargs) -> None:
            captured.update(kwargs)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            research_sink=sink,
        )

        class ForbiddenError(Exception):
            pass

        ok = AsyncMock(return_value="real research prose")
        empty = AsyncMock(return_value="")
        errored = AsyncMock(side_effect=RuntimeError("provider blew up"))
        inactive = AsyncMock(side_effect=ForbiddenError("403011 - subscription is not currently active"))

        selected = [
            (ok, "native_search"),
            (empty, "gemini_search"),
            (errored, "financial_data"),
            (inactive, "asknews"),
        ]

        with patch.object(orch, "_select_research_providers", return_value=selected):
            await orch.run_research(question)

        # Sink fired (question fixture id_of_question=42 satisfies the qid gate).
        assert captured, "research_sink was never called"
        assert captured["providers_attempted"] == ["native_search", "gemini_search", "financial_data", "asknews"]
        # Only the ok provider succeeded — empty, errored, AND inactive are all excluded.
        assert captured["providers_succeeded"] == ["native_search"]
        assert "gemini_search" not in captured["providers_succeeded"]  # empty
        assert "financial_data" not in captured["providers_succeeded"]  # errored
        assert "asknews" not in captured["providers_succeeded"]  # inactive (off-season)

        # provider_results round-trips each per-provider status as a list of dicts.
        statuses = {r["name"]: r["status"] for r in captured["provider_results"]}
        assert statuses == {
            "native_search": "ok",
            "gemini_search": "empty",
            "financial_data": "errored",
            "asknews": "inactive",
        }

    @pytest.mark.asyncio
    async def test_sink_receives_raw_asknews_text_pre_summarization(self, mock_llm, question, monkeypatch):
        """2026-07-18 audit hygiene: the archive stores only the post-summarization
        briefing, so the sink must also receive the raw pre-summarization AskNews
        article text — otherwise FETCH-vs-SUMMARIZE attribution and summarizer
        replays require fresh paid pulls."""
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)

        captured: dict = {}

        def sink(**kwargs) -> None:
            captured.update(kwargs)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            research_sink=sink,
        )
        asknews = AsyncMock(return_value="raw asknews articles")
        native = AsyncMock(return_value="native search prose")

        with (
            patch.object(
                orch,
                "_select_research_providers",
                return_value=[(asknews, "asknews"), (native, "native_search")],
            ),
            patch.object(
                orch._summarizer_llm,
                "invoke",
                new_callable=AsyncMock,
                return_value="ASKNEWS BRIEFING",
            ),
        ):
            research = await orch.run_research(question)

        # Forecaster-facing text carries the briefing; the sink carries the raw.
        assert "ASKNEWS BRIEFING" in research
        assert "raw asknews articles" not in research
        assert captured["asknews_raw"] == "raw asknews articles"
        # The raw is a sibling field, never folded into research_text.
        assert "raw asknews articles" not in captured["research_text"]


class TestDemoteInnerHeadings:
    """Provider bodies must not carry headings at/above the h2 provider header.

    Otherwise the framework's report_sections_to_markdown raises ("First section
    must be at the highest heading level") and degrades to the [Hashtag] fallback.
    """

    def test_demotes_h1_and_h2_to_at_least_h3(self) -> None:
        from metaculus_bot.research.orchestrator import _demote_inner_headings

        body = "# Historical Context\nbody\n## Recent Developments\nmore\n### Already deep\nkept"
        out = _demote_inner_headings(body)
        assert "### Historical Context" in out  # h1 -> h3
        assert "#### Recent Developments" in out  # h2 -> h4
        assert "### Already deep" in out  # h3 untouched
        assert not out.startswith("# "), "must not start with h1"
        assert "\n# " not in out, "no h1 on any line"
        assert not out.startswith("## "), "must not start with h2"
        assert "\n## " not in out, "no h2 on any line"

    def test_inline_hash_not_treated_as_heading(self) -> None:
        from metaculus_bot.research.orchestrator import _demote_inner_headings

        body = "Issue #42 and C# are not headings\n# Real Heading"
        out = _demote_inner_headings(body)
        assert "Issue #42 and C# are not headings" in out
        assert "### Real Heading" in out

    def test_combined_research_first_section_is_minimum_level(self) -> None:
        """After normalization, every provider's h2 header outranks its body.

        The first section of the combined string must be the minimum heading
        level so the framework wouldn't raise on it.
        """
        import re

        from metaculus_bot.research.orchestrator import _demote_inner_headings

        # Simulate the assembly: h2 provider header + a body that had its own h1.
        provider_body = "# Historical Context\nstuff\n## Recent\nmore"
        combined = f"## News Articles (AskNews)\n{_demote_inner_headings(provider_body)}"

        heading_levels = [len(m.group(1)) for m in re.finditer(r"^(#+)\s", combined, re.MULTILINE)]
        assert heading_levels, "precondition: combined must contain headings"
        # The first heading (provider header, h2) must be the minimum.
        assert heading_levels[0] == min(heading_levels)
        assert heading_levels[0] == 2


class TestStagesReportTheirAccountingInsteadOfCounting:
    """The split stages RETURN what happened; the orchestrator owns the counters.

    This is the contract that replaced the mixin inheritance: ``summarize_asknews``
    and ``run_gap_fill_passes`` are callable with no orchestrator at all, and they
    touch no counter. If a future edit moves a ``+= 1`` back inside either module,
    the orchestrator's own bump makes it a DOUBLE count — which reads as two
    degradations where one happened, on exactly the counters that redden CI.
    """

    @pytest.mark.asyncio
    async def test_summarize_asknews_reports_the_soft_fail_reason_without_a_counter(self, mock_llm, question) -> None:
        with patch.object(mock_llm, "invoke", new_callable=AsyncMock, return_value="   "):
            text, reason = await summarize_asknews(question, "raw asknews articles", summarizer_llm=mock_llm)

        assert reason == "blank_output"
        assert text.startswith(SUMMARIZER_SOFT_FAIL_BANNER)

    @pytest.mark.asyncio
    async def test_a_clean_summary_reports_no_reason(self, mock_llm, question) -> None:
        with patch.object(mock_llm, "invoke", new_callable=AsyncMock, return_value="briefing"):
            text, reason = await summarize_asknews(question, "raw asknews articles", summarizer_llm=mock_llm)

        assert (text, reason) == ("briefing", None)

    @pytest.mark.asyncio
    async def test_run_gap_fill_passes_reports_a_budget_cut_without_a_counter(self, question, monkeypatch) -> None:
        """The fast path skips both passes; the outcome says so rather than bumping.

        ``budget_cut`` is one boolean for the whole question precisely so the
        orchestrator's per-question dedup can't be defeated by two cut stages.
        """
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

        outcome = await run_gap_fill_passes(
            question, "research prose", fast_path=True, is_benchmarking=False, time_budget=None
        )

        assert outcome.budget_cut is True
        assert outcome.research == "research prose"
        assert (outcome.v2_payload, outcome.v2_errors) == (None, 0)

    @pytest.mark.asyncio
    async def test_run_gap_fill_passes_reports_a_v2_crash_without_a_counter(self, question, monkeypatch) -> None:
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

        with patch(
            "metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2",
            new_callable=AsyncMock,
            side_effect=RuntimeError("v2 escaped its own soft-fail"),
        ):
            outcome = await run_gap_fill_passes(
                question, "research prose", fast_path=False, is_benchmarking=False, time_budget=None
            )

        assert outcome.v2_errors == 1
        assert outcome.budget_cut is False
        assert outcome.research == "research prose"

    @pytest.mark.asyncio
    async def test_run_gap_fill_passes_reports_a_v1_crash_without_a_counter(self, question, monkeypatch) -> None:
        """A v1 failure callback becomes stage accounting, just as v2's does."""
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.delenv("GAP_FILL_V2_ENABLED", raising=False)

        async def failing_v1(
            _question,
            _research,
            *,
            is_benchmarking: bool,
            on_error,
        ) -> str:
            assert is_benchmarking is False
            on_error(ValueError("invalid analyzer schema"))
            return ""

        with patch(
            "metaculus_bot.research.targeted.run_gap_fill_pass",
            new=failing_v1,
        ):
            outcome = await run_gap_fill_passes(
                question,
                "x" * GAP_FILL_MIN_RESEARCH_CHARS,
                fast_path=False,
                is_benchmarking=False,
                time_budget=None,
            )

        assert outcome.v1_errors == 1
        assert outcome.v2_errors == 0
        assert outcome.budget_cut is False
        assert outcome.research == "x" * GAP_FILL_MIN_RESEARCH_CHARS

    @pytest.mark.asyncio
    async def test_run_gap_fill_passes_counts_a_v1_escape_from_the_failure_seam(self, question, monkeypatch) -> None:
        """An unexpected v1 escape is counted at the stage guard, like v2's escape path."""
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.delenv("GAP_FILL_V2_ENABLED", raising=False)

        with patch(
            "metaculus_bot.research.targeted.run_gap_fill_pass",
            new_callable=AsyncMock,
            side_effect=RuntimeError("v1 escaped its own soft-fail"),
        ):
            outcome = await run_gap_fill_passes(
                question,
                "x" * GAP_FILL_MIN_RESEARCH_CHARS,
                fast_path=False,
                is_benchmarking=False,
                time_budget=None,
            )

        assert outcome.v1_errors == 1
        assert outcome.v2_errors == 0
        assert outcome.budget_cut is False
        assert outcome.research == "x" * GAP_FILL_MIN_RESEARCH_CHARS

    @pytest.mark.asyncio
    async def test_run_gap_fill_passes_does_not_report_a_legitimate_empty_v1_result(
        self, question, monkeypatch
    ) -> None:
        """An analyzer that answers with no gaps is an ordinary empty result."""
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.delenv("GAP_FILL_V2_ENABLED", raising=False)

        async def empty_v1(
            _question,
            _research,
            *,
            is_benchmarking: bool,
            on_error,
        ) -> str:
            assert is_benchmarking is False
            assert on_error is not None
            return ""

        with patch(
            "metaculus_bot.research.targeted.run_gap_fill_pass",
            new=empty_v1,
        ):
            outcome = await run_gap_fill_passes(
                question,
                "x" * GAP_FILL_MIN_RESEARCH_CHARS,
                fast_path=False,
                is_benchmarking=False,
                time_budget=None,
            )

        assert outcome.v1_errors == 0
        assert outcome.budget_cut is False
        assert outcome.research == "x" * GAP_FILL_MIN_RESEARCH_CHARS
