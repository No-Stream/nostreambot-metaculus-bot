"""Research orchestration extracted from TemplateForecaster.

Encapsulates provider selection, the primary provider's fallback ladder, caching, the
diagnostics seam, and the per-question time budget. The TemplateForecaster delegates
run_research to an instance of this class.

The stages that are self-contained live in sibling modules as plain functions, called
from here. The two that produce accounting RETURN it (a soft-fail reason, a
``GapFillOutcome``) and this class owns the counters, which is what keeps the counters
in one place while the stages stay callable without an orchestrator:

* ``provider_fanout`` — running the selected providers under the research-phase deadline.
* ``section_format`` — provider headers and heading levels in the assembled bundle.
* ``asknews_summarization`` — the AskNews-only summarizer pass (it's the one provider
  that returns raw article text rather than LLM-written prose; all others pass through).
* ``gap_fill_stages`` — the two optional gap-fill passes and their budget accounting.
* ``degradation_views`` — read-only views onto the research side's module counters,
  re-exposed below as orchestrator attributes for the consumers that read them there.
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict

from forecasting_tools import GeneralLlm, clean_indents
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.api_key_utils import get_openrouter_api_key
from metaculus_bot.constants import (
    DEFAULT_MAX_CONCURRENT_RESEARCH,
    EXA_API_KEY_ENV,
    FINANCIAL_DATA_ENABLED_ENV,
    GEMINI_SEARCH_ENABLED_ENV,
    GEMINI_SEARCH_MODEL_ENV,
    NATIVE_SEARCH_ENABLED_ENV,
    NATIVE_SEARCH_MODEL_ENV,
    OPENROUTER_API_KEY_ENV,
    PERPLEXITY_API_KEY_ENV,
    PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER,
    PREDICTION_MARKETS_ENABLED_ENV,
    RESOLUTION_SOURCE_ENABLED_ENV,
    TS_ANCHOR_ENABLED_ENV,
    env_flag_enabled,
)
from metaculus_bot.fallback_openrouter import _record_deprecation_if_matched
from metaculus_bot.llm_retry import llm_error_fields
from metaculus_bot.prompts import OUTSIDE_VENUE_MARKET_ODDS_POLICY
from metaculus_bot.research import degradation_views
from metaculus_bot.research.asknews_summarization import summarize_asknews
from metaculus_bot.research.gap_fill_stages import run_gap_fill_passes
from metaculus_bot.research.provider_diagnostics import (
    SUCCEEDED_STATUSES,
    ProviderResult,
    format_provider_diagnostics_block,
    pop_provider_detail,
    record_provider_detail,
)
from metaculus_bot.research.provider_fanout import _empty_provider, await_providers_within_deadline
from metaculus_bot.research.providers import (
    ResearchCallable,
    _invoke_exa_research,
    _invoke_perplexity_research,
    choose_provider_with_name,
    is_asknews_subscription_error,
    native_search_provider,
)
from metaculus_bot.research.section_format import _demote_inner_headings, assemble_provider_sections
from metaculus_bot.run_status import RUN_STATUS
from metaculus_bot.time_budget import QuestionTimeBudget

# Re-export for out-of-package callers. See docs/research.md "Orchestrator implementation notes".
__all__ = ["ResearchOrchestrator", "_demote_inner_headings"]

_PROVIDER_ERROR_MESSAGE_MAX_CHARS = 300

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """Manages research provider selection, parallel execution, caching, and gap-fill."""

    def __init__(
        self,
        *,
        default_llm: GeneralLlm,
        summarizer_llm: GeneralLlm,
        custom_provider: ResearchCallable | None = None,
        research_cache: dict[int, str] | None = None,
        is_benchmarking: bool = False,
        allow_research_fallback: bool = True,
        max_concurrent_research: int = DEFAULT_MAX_CONCURRENT_RESEARCH,
        research_sink: Callable[..., None] | None = None,
    ) -> None:
        self._default_llm = default_llm
        self._summarizer_llm = summarizer_llm
        self._custom_provider = custom_provider
        self._research_cache = research_cache
        self._is_benchmarking = is_benchmarking
        self._allow_research_fallback = allow_research_fallback
        self._concurrency_limiter = asyncio.Semaphore(max_concurrent_research)
        self._research_sink = research_sink
        # Withheld from the forecaster-facing text. See docs/research.md "Orchestrator implementation notes".
        self._comment_diagnostics: dict[int, str] = {}
        # Any exception, not just timeouts. See docs/research.md "Orchestrator implementation notes".
        self.provider_failure_count: int = 0
        # Alertable: ships raw articles as the briefing. See docs/research.md "Orchestrator implementation notes".
        self.summarizer_failure_count: int = 0
        # Genuine v1 failures only, and alertable. See docs/research.md "Orchestrator implementation notes".
        self.gap_fill_v1_error_count: int = 0
        # Genuine v2 crashes only, and alertable. See docs/research.md "Orchestrator implementation notes".
        self.gap_fill_v2_error_count: int = 0
        # Off-fast-path budget thinning, deduped. See docs/research.md "Orchestrator implementation notes".
        self.research_budget_cut_count: int = 0
        self._research_budget_cut_seen: set[object] = set()

    def _record_research_budget_cut(self, question: MetaculusQuestion, *, fast_path: bool) -> None:
        """Count one question's budget-driven research degradation, once, off the fast path."""
        if fast_path:
            return
        key: object = getattr(question, "id_of_question", None) or id(question)
        if key in self._research_budget_cut_seen:
            return
        self._research_budget_cut_seen.add(key)
        self.research_budget_cut_count += 1

    async def run_research(self, question: MetaculusQuestion, time_budget: QuestionTimeBudget | None = None) -> str:
        """Build one question's research bundle, inside its time budget if it has one.

        ``time_budget`` is None for every caller that isn't the per-question
        pipeline (tests, the research-only tooling), and then the phase runs
        unbounded exactly as it did before. When present it does two things: on a
        thin window (``fast_path``) the OPTIONAL stages are not run at all, and in
        every case the phase is bounded by its share of the remaining budget so it
        cannot spend the time the forecast fan-out and the prediction POST need.
        """
        cache_key, cached = self._lookup_research_cache(question)
        if cached is not None:
            logger.info(f"Using cached research for question {cache_key}")
            return cached

        async with self._concurrency_limiter:
            cache_key, cached = self._lookup_research_cache(question)
            if cached is not None:
                logger.info(f"Using cached research for question {cache_key} (double-check)")
                return cached

            fast_path = time_budget is not None and time_budget.fast_path
            providers = self._select_research_providers(fast_path=fast_path)
            provider_names = [name for _, name in providers]
            logger.info(f"Using research providers: {provider_names}")

            research, provider_results, asknews_raw = await self._run_providers_parallel(
                question, providers, time_budget=time_budget
            )
            for provider_result in provider_results:
                RUN_STATUS.record_provider_result(
                    name=provider_result.name,
                    status=provider_result.status,
                    error_type=provider_result.error_type,
                    details=provider_result.details,
                )
            if any(pr.status == "deadline" for pr in provider_results):
                # Off the fast path nothing else counts this. See docs/research.md "Orchestrator implementation notes".
                self._record_research_budget_cut(question, fast_path=fast_path)

            research, gap_fill_v2_payload = await self._run_gap_fill_passes(
                question, research, fast_path=fast_path, time_budget=time_budget
            )

            gap_fill_used = "## Targeted Gap-Fill (second pass)" in research

            # Deliberately kept out of the returned research. See docs/research.md "Orchestrator implementation notes".
            diagnostics_block = format_provider_diagnostics_block(provider_results)
            qid = getattr(question, "id_of_question", None)
            if diagnostics_block:
                logger.info(f"Provider diagnostics for URL {question.page_url}:\n{diagnostics_block}")
                if qid is not None:
                    self._comment_diagnostics[qid] = diagnostics_block

            self._store_research_cache(cache_key, research)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")

            if self._research_sink is not None and qid is not None:
                try:
                    # providers_used is legacy. See docs/research.md "Orchestrator implementation notes".
                    self._research_sink(
                        qid=qid,
                        post_id=getattr(question, "id_of_post", None),
                        page_url=question.page_url,
                        question_text=question.question_text,
                        research_text=research,
                        providers_used=provider_names,
                        gap_fill_used=gap_fill_used,
                        provider_results=[asdict(r) for r in provider_results],
                        providers_attempted=provider_names,
                        providers_succeeded=[r.name for r in provider_results if r.status in SUCCEEDED_STATUSES],
                        gap_fill_v2=gap_fill_v2_payload,
                        provider_diagnostics_block=diagnostics_block,
                        asknews_raw=asknews_raw,
                    )
                except (
                    Exception
                ):  # HARNESS-SCAN-EXEMPT-broad-except — archive write is best-effort; never blocks the forecast
                    logger.exception("Research sink failed for qid=%d; continuing", qid)

            return research

    def pop_provider_diagnostics(self, qid: int | None) -> str:
        """Return-and-clear the comment-bound provider-diagnostics block for a question.

        The other half of the diagnostics seam in ``run_research``: the block is
        withheld from the forecaster-facing research text, and the forecaster pops
        it here when assembling ``research_report`` (the published comment).
        Popping keeps the stash from growing across a batch. Returns ``""`` when
        no diagnostics were recorded for the qid.
        """
        if qid is None:
            return ""
        return self._comment_diagnostics.pop(qid, "")

    def _lookup_research_cache(self, question: MetaculusQuestion) -> tuple[int | None, str | None]:
        cache_key = getattr(question, "id_of_question", None)
        if not self._is_benchmarking or self._research_cache is None or cache_key is None:
            return cache_key, None
        return cache_key, self._research_cache.get(cache_key)

    def _store_research_cache(self, cache_key: int | None, research: str) -> None:
        if not self._is_benchmarking or self._research_cache is None or cache_key is None:
            return
        self._research_cache[cache_key] = research
        logger.info(f"Cached research for question {cache_key}")

    def _select_research_provider(self) -> tuple[ResearchCallable, str]:
        if self._custom_provider is not None:
            return self._custom_provider, "custom"

        provider, provider_name = choose_provider_with_name(
            self._default_llm,
            exa_callback=self._call_exa_smart_searcher,
            # Each rung gets the vendor its env var pays for. See docs/research.md "Orchestrator implementation notes".
            perplexity_callback=self._call_perplexity_direct,
            openrouter_callback=self._call_perplexity_openrouter,
            is_benchmarking=self._is_benchmarking,
        )
        return provider, provider_name

    def _select_research_providers(self, fast_path: bool = False) -> list[tuple[ResearchCallable, str]]:
        """Assemble the enabled providers for one question.

        ``fast_path`` is the time-budget thin-window mode: drop the two SLOW search
        providers (native_search, gemini_search) and keep everything else. The cheap
        hard-capped providers stay, because they run CONCURRENTLY with the primary and
        so cannot shorten the phase; ``resolution_source`` is handed the flag instead, so
        its two expensive escalation rungs decline while its cheap rungs run. Anything
        still straggling past the research window is cancelled by
        ``await_providers_within_deadline`` with its partial bundle kept. The measured
        provenance for what the fast path sheds is in docs/research.md "Orchestrator
        implementation notes".
        """
        providers: list[tuple[ResearchCallable, str]] = []

        primary, primary_name = self._select_research_provider()
        if primary_name != "none":
            providers.append((primary, primary_name))

        if not fast_path and env_flag_enabled(NATIVE_SEARCH_ENABLED_ENV):
            model = os.getenv(NATIVE_SEARCH_MODEL_ENV)
            providers.append(
                (
                    native_search_provider(model, is_benchmarking=self._is_benchmarking),
                    "native_search",
                )
            )

        if not fast_path and env_flag_enabled(GEMINI_SEARCH_ENABLED_ENV):
            from metaculus_bot.research.gemini_search import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # gated google-genai provider
                gemini_search_provider,
            )

            gemini_model = os.getenv(GEMINI_SEARCH_MODEL_ENV)
            providers.append(
                (
                    gemini_search_provider(gemini_model, is_benchmarking=self._is_benchmarking),
                    "gemini_search",
                )
            )

        if env_flag_enabled(FINANCIAL_DATA_ENABLED_ENV):
            from metaculus_bot.research.financial_data import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # gated pandas/yfinance/fredapi provider
                financial_data_provider,
            )

            providers.append((financial_data_provider(is_benchmarking=self._is_benchmarking), "financial_data"))

        if env_flag_enabled(TS_ANCHOR_ENABLED_ENV):
            from metaculus_bot.research.timeseries_anchor import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # gated numpy/pandas provider
                timeseries_anchor_provider,
            )

            providers.append((timeseries_anchor_provider(is_benchmarking=self._is_benchmarking), "timeseries_anchor"))

        if env_flag_enabled(PREDICTION_MARKETS_ENABLED_ENV):
            from metaculus_bot.research.prediction_market import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # gated rapidfuzz/aiohttp provider
                prediction_market_provider,
            )

            providers.append((prediction_market_provider(is_benchmarking=self._is_benchmarking), "prediction_market"))

        if env_flag_enabled(RESOLUTION_SOURCE_ENABLED_ENV):
            from metaculus_bot.research.resolution_source import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # gated aiohttp/trafilatura provider
                resolution_source_provider,
            )

            # Stays on the fast path; the flag makes only its two expensive rungs decline.
            providers.append(
                (
                    resolution_source_provider(is_benchmarking=self._is_benchmarking, fast_path=fast_path),
                    "resolution_source",
                )
            )

        if not providers:
            providers.append((_empty_provider, "none"))

        return providers

    def _failed_provider_result(self, name: str, exc: Exception, latency_ms: int, qid: int | None) -> ProviderResult:
        """Classify a provider that raised: ``inactive`` for expected off-season AskNews, else ``errored``.

        Only ``errored`` bumps ``provider_failure_count`` (which reddens CI) and
        feeds the deprecation matcher; an inactive subscription is a known
        off-season state, not degradation.
        """
        if name == "asknews" and is_asknews_subscription_error(exc):
            status = "inactive"
            logger.info(
                "Research provider %s inactive (expected off-season): %s: %s",
                name,
                type(exc).__name__,
                exc,
            )
        else:
            status = "errored"
            self.provider_failure_count += 1
            logger.warning(
                f"RESEARCH_PROVIDER_FAILED: question={qid} provider={name} {llm_error_fields(exc)} detail={exc}"
            )
            _record_deprecation_if_matched(f"<provider:{name}>", str(exc))
        return ProviderResult(
            name=name,
            status=status,
            chars=0,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
            error_message=str(exc)[:_PROVIDER_ERROR_MESSAGE_MAX_CHARS],
        )

    async def _run_providers_parallel(
        self,
        question: MetaculusQuestion,
        providers: list[tuple[ResearchCallable, str]],
        time_budget: QuestionTimeBudget | None = None,
    ) -> tuple[str, list[ProviderResult], str]:
        """Run the selected providers concurrently and assemble their sections.

        Returns the assembled bundle, one ``ProviderResult`` per provider, and the raw
        pre-summarization AskNews articles for the archive (``""`` when AskNews did not
        run, errored, or fell back to an already-prose provider). See docs/research.md
        "Orchestrator implementation notes" for why that raw text is captured.
        """
        asknews_raw_holder: dict[str, str] = {}

        async def _run_one(provider: ResearchCallable, name: str) -> tuple[str, ProviderResult]:
            started = time.monotonic()
            # Needed to drain the per-source registry. See docs/research.md "Orchestrator implementation notes".
            qid = getattr(question, "id_of_question", None)
            try:
                fallback_provider: str | None = None
                if name == "asknews" and self._allow_research_fallback:
                    raw, fallback_provider = await self._fetch_research_with_fallback(question, provider, name)
                else:
                    raw = await provider(question)
                used_fallback = fallback_provider is not None
                # AskNews alone returns raw markdown. See docs/research.md "Orchestrator implementation notes".
                if name == "asknews" and not used_fallback and raw and raw.strip():
                    # Empty raw must skip the summarizer. See docs/research.md "Orchestrator implementation notes".
                    asknews_raw_holder["text"] = raw
                    raw = await self._summarize_asknews(question, raw)
                latency_ms = int((time.monotonic() - started) * 1000)
                has_output = bool(raw and raw.strip())
                if not has_output:
                    status = "empty"
                elif used_fallback:
                    status = "fallback"
                else:
                    status = "ok"
                result = ProviderResult(
                    name=name,
                    status=status,
                    chars=len(raw) if has_output else 0,
                    latency_ms=latency_ms,
                    details=pop_provider_detail(qid, name),
                    fallback_provider=fallback_provider,
                )
                return (raw, result)
            except asyncio.CancelledError:
                # BaseException skips the drain below. See docs/research.md "Orchestrator implementation notes".
                pop_provider_detail(qid, name)
                raise
            except Exception as e:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except — converted to a ProviderResult(status=errored/inactive); one provider failing never kills the research phase
                # No stale entry may leak into a later call. See docs/research.md "Orchestrator implementation notes".
                pop_provider_detail(qid, name)
                latency_ms = int((time.monotonic() - started) * 1000)
                return ("", self._failed_provider_result(name, e, latency_ms, qid))

        results = await await_providers_within_deadline(providers, _run_one, time_budget)
        combined, provider_results = assemble_provider_sections(results)
        return combined, provider_results, asknews_raw_holder.get("text", "")

    async def _summarize_asknews(self, question: MetaculusQuestion, research: str) -> str:
        """Summarize raw AskNews articles, counting a soft-fail on the way through.

        ``summarize_asknews`` reports the degradation instead of counting it, so the
        ``summarizer_failure_count`` the end-of-run degradation line reads is bumped
        here. Kept as a method because it is the suite's class-level patch surface.
        """
        text, soft_fail_reason = await summarize_asknews(question, research, summarizer_llm=self._summarizer_llm)
        if soft_fail_reason is not None:
            self.summarizer_failure_count += 1
        return text

    async def _run_gap_fill_passes(
        self,
        question: MetaculusQuestion,
        research: str,
        *,
        fast_path: bool,
        time_budget: QuestionTimeBudget | None,
    ) -> tuple[str, dict | None]:
        """Run both gap-fill passes and absorb their accounting into this bot's counters.

        ``gap_fill_stages`` returns what happened rather than counting it, so this is
        the one place a v1 or v2 crash bumps its corresponding error counter and the one place a
        budget cut reaches ``_record_research_budget_cut`` (whose per-question dedup and
        fast-path suppression then apply once, however many stages were cut).
        """
        outcome = await run_gap_fill_passes(
            question,
            research,
            fast_path=fast_path,
            is_benchmarking=self._is_benchmarking,
            time_budget=time_budget,
        )
        self.gap_fill_v1_error_count += outcome.v1_errors
        self.gap_fill_v2_error_count += outcome.v2_errors
        if outcome.budget_cut:
            self._record_research_budget_cut(question, fast_path=fast_path)
        return outcome.research, outcome.v2_payload

    async def _fetch_research_with_fallback(
        self,
        question: MetaculusQuestion,
        provider: ResearchCallable,
        provider_name: str,
    ) -> tuple[str, str | None]:
        """Return ``(research_text, fallback_provider_name)``.

        ``fallback_provider_name`` is None on the normal path and otherwise names the
        vendor that actually answered ("openrouter" / "perplexity" / "exa"). The caller
        uses it for two things: to skip AskNews summarization on already-prose output, and
        to label the research section with the source that produced it.

        A vendor swap on the PRIMARY provider is real degradation, not a success, so a
        per-source loss token is recorded here for the diagnostics line and the schema-v2
        archive. See docs/research.md "Orchestrator implementation notes" for what the swap
        costs and why it is deliberately not a new alertable counter.
        """
        try:
            return (await provider(question), None)
        except Exception as exc:
            if self._allow_research_fallback and provider_name == "asknews":
                logger.warning(f"Primary research provider '{provider_name}' failed with {type(exc).__name__}: {exc}")
                fallback_text, fallback_name = await self._attempt_research_fallback(question.question_text)
                if fallback_text is not None:
                    record_provider_detail(
                        getattr(question, "id_of_question", None),
                        provider_name,
                        {
                            "sources": {
                                provider_name: f"error({type(exc).__name__})",
                                "fallback": f"ok({fallback_name})",
                            }
                        },
                    )
                    return (fallback_text, fallback_name)
            raise

    async def _attempt_research_fallback(self, question_text: str) -> tuple[str | None, str | None]:
        """Return ``(research_text, provider_name)``, or ``(None, None)`` if no rung answered.

        The provider name rides back so the caller can label the section with the vendor
        that actually answered; rendering it as AskNews mislabeled the source in the
        published comment and in the archive.
        """
        # Ordered by cost, not index quality. See docs/research.md "AskNews fallback (primary-only)".
        try:
            if os.getenv(OPENROUTER_API_KEY_ENV):
                logger.info("Falling back to openrouter/perplexity for research")
                return (await self._call_perplexity(question_text, use_open_router=True), "openrouter")
            if os.getenv(PERPLEXITY_API_KEY_ENV):
                logger.info("Falling back to Perplexity for research")
                return (await self._call_perplexity(question_text, use_open_router=False), "perplexity")
            if os.getenv(EXA_API_KEY_ENV):
                logger.info("Falling back to Exa search for research")
                return (await self._call_exa_smart_searcher(question_text), "exa")
        except Exception as fallback_exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except — best-effort fallback; logs and returns None so the primary error propagates
            logger.warning(f"Fallback research provider also failed: {type(fallback_exc).__name__}: {fallback_exc}")
        return (None, None)

    async def _call_perplexity(self, question: MetaculusQuestion | str, use_open_router: bool = True) -> str:
        question_text = question.question_text if isinstance(question, MetaculusQuestion) else question

        # Interpolated from the one definition in `prompts`. See docs/research.md "Orchestrator implementation notes".
        prediction_markets_instruction = (
            ""
            if self._is_benchmarking
            else (
                f"In addition to news, cover: {OUTSIDE_VENUE_MARKET_ODDS_POLICY} "
                "(If there are no relevant markets of that kind, simply skip reporting on this and "
                "DO NOT speculate what they would say.)"
            )
        )

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            {prediction_markets_instruction}
            You DO NOT produce forecasts yourself; you must provide ALL relevant data to the superforecaster so they can make an expert judgment.

            Question:
            {question_text}
            """
        )
        # Explicit credential routing: direct Perplexity passes None, OpenRouter resolves its key first.
        api_key = get_openrouter_api_key(PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER) if use_open_router else None
        return await _invoke_perplexity_research(
            prompt,
            use_open_router=use_open_router,
            api_key=api_key,
        )

    async def _call_perplexity_openrouter(self, question: MetaculusQuestion) -> str:
        return await self._call_perplexity(question, use_open_router=True)

    async def _call_perplexity_direct(self, question: MetaculusQuestion) -> str:
        return await self._call_perplexity(question, use_open_router=False)

    async def _call_exa_smart_searcher(self, question: MetaculusQuestion | str) -> str:
        question_text = question.question_text if isinstance(question, MetaculusQuestion) else question
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question_text}"
        )
        return await _invoke_exa_research(self._default_llm, prompt)

    # The attribute surface over ``degradation_views``. See docs/research.md "Orchestrator implementation notes".
    @property
    def prediction_market_degraded_count(self) -> int:
        return degradation_views.prediction_market_degraded_count()

    @property
    def prediction_market_source_loss_count(self) -> int:
        return degradation_views.prediction_market_source_loss_count()

    @property
    def provider_degradation_count(self) -> int:
        return degradation_views.provider_degradation_count()

    def log_provider_degradation_summary(self) -> None:
        degradation_views.log_provider_degradation_summary()

    def reset_run_degradation_counters(self) -> None:
        degradation_views.reset_run_degradation_counters()
