"""Research provider strategy abstraction.

`choose_provider_with_name` returns an async callable (and its name) that, given
a `MetaculusQuestion`, returns formatted research.  The selection is governed
by environment variables so the logic lives in one place instead of being in
`TemplateForecaster.run_research`.

Providers receive the full question (not just the text) so they can use
auxiliary fields like `id_of_question` for caching, `resolution_criteria` for
keyword extraction, and `scheduled_resolution_time` for backtest-leakage
defenses (see `prediction_market_provider.py`).
"""

import asyncio
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from forecasting_tools import GeneralLlm, SmartSearcher
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    ASKNEWS_BACKOFF_SECS,
    ASKNEWS_CLIENT_ID_ENV,
    ASKNEWS_MAX_CONCURRENCY,
    ASKNEWS_MAX_RPS,
    ASKNEWS_MAX_TRIES,
    ASKNEWS_SECRET_ENV,
    ASKNEWS_WALL_TIMEOUT,
    EXA_API_KEY_ENV,
    NATIVE_SEARCH_CONTEXT_SIZE,
    NATIVE_SEARCH_DEFAULT_MODEL,
    NATIVE_SEARCH_MAX_RESULTS,
    NATIVE_SEARCH_MAX_TOKENS,
    NATIVE_SEARCH_MODEL_ENV,
    NATIVE_SEARCH_REASONING_EFFORT_DEFAULT,
    NATIVE_SEARCH_REASONING_EFFORT_ENV,
    NATIVE_SEARCH_TIMEOUT,
    NATIVE_SEARCH_VERBOSITY_DEFAULT,
    NATIVE_SEARCH_VERBOSITY_ENV,
    OPENROUTER_API_KEY_ENV,
    PERPLEXITY_API_KEY_ENV,
    PERPLEXITY_RESEARCH_MODEL,
    PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER,
    PERPLEXITY_WALL_TIMEOUT,
    RESEARCH_PROVIDER_ENV,
)
from metaculus_bot.credit_telemetry import llm_call_metadata, plain_llm_key_alias
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.llm_retry import invoke_with_transient_retry
from metaculus_bot.prompts import OUTSIDE_VENUE_MARKET_ODDS_POLICY, web_research_prompt
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.raw_log import record_raw_research

ResearchCallable = Callable[[MetaculusQuestion], Awaitable[str]]
logger = logging.getLogger(__name__)


class _OmittedPerplexityApiKey:
    pass


_OMITTED_API_KEY = _OmittedPerplexityApiKey()


# ---------------------------------------------------------------------------
# Concrete provider helpers
# ---------------------------------------------------------------------------


_ASKNEWS_GLOBAL_SEMAPHORE: asyncio.Semaphore | None = None
_ASKNEWS_RATE_LOCK: asyncio.Lock | None = None
_ASKNEWS_LAST_CALL_TS: float = 0.0


def _get_asknews_rate_lock() -> asyncio.Lock:
    """Get-or-create the process-wide lock guarding the AskNews RPS gate.

    Lazy purely for consistency with ``get_asknews_semaphore`` below, which owns the
    identical lifecycle for the same provider; an import-time primitive is also the
    shape that can bind to a loop that later dies. A latent-shape cleanup rather than
    a fix for an observed failure, and deliberately unpaired with a staleness check:
    see docs/research.md "AskNews dual-phase search".
    """
    global _ASKNEWS_RATE_LOCK  # noqa: PLW0603  # sole lazy-init of the process-wide AskNews lock; see docstring
    if _ASKNEWS_RATE_LOCK is None:
        _ASKNEWS_RATE_LOCK = asyncio.Lock()
    return _ASKNEWS_RATE_LOCK


async def _asknews_rate_gate() -> None:
    global _ASKNEWS_LAST_CALL_TS  # noqa: PLW0603  # process-wide AskNews RPS clock, shared by the provider and agentic tools
    if ASKNEWS_MAX_RPS <= 0:
        return
    min_interval = 1.0 / ASKNEWS_MAX_RPS
    async with _get_asknews_rate_lock():
        now = time.monotonic()
        wait = _ASKNEWS_LAST_CALL_TS + min_interval - now
        if wait > 0:
            await asyncio.sleep(wait)
            now = time.monotonic()
        _ASKNEWS_LAST_CALL_TS = now


async def asknews_rate_gate() -> None:
    """Public seam for the process-wide AskNews RPS gate.

    Delegates at call time so tests that monkeypatch ``_asknews_rate_gate``
    keep intercepting calls routed through this public name (used by
    ``research.agentic.tools``).
    """
    await _asknews_rate_gate()


def get_asknews_semaphore() -> asyncio.Semaphore:
    """Get-or-create the single process-wide AskNews concurrency semaphore.

    Owns the only lazy-init of ``_ASKNEWS_GLOBAL_SEMAPHORE`` so every AskNews
    caller (the two-phase provider here and ``research.agentic.tools``)
    contends on the same throttle.
    """
    global _ASKNEWS_GLOBAL_SEMAPHORE  # noqa: PLW0603  # sole lazy-init of the process-wide AskNews semaphore; see docstring
    if _ASKNEWS_GLOBAL_SEMAPHORE is None:
        _ASKNEWS_GLOBAL_SEMAPHORE = asyncio.Semaphore(max(1, int(ASKNEWS_MAX_CONCURRENCY)))
    return _ASKNEWS_GLOBAL_SEMAPHORE


def is_asknews_subscription_error(exc: BaseException) -> bool:
    """True iff exc is AskNews's 403011 subscription-inactive signature.

    Narrow match (class name AND inner message) so a generic "403 Forbidden"
    from an unrelated provider isn't silenced. SDK raises ForbiddenError with
    code 403011 or "subscription is not currently active" when billing lapses.
    """
    msg = str(exc).lower()
    return "forbiddenerror" in type(exc).__name__.lower() and (
        "403011" in msg or "subscription is not currently active" in msg
    )


# Per-phase AskNews search parameters: (log label, SDK strategy, n_articles).
_ASKNEWS_PHASES: dict[str, tuple[str, str, int]] = {
    "hot": ("HOT", "latest news", 6),
    "historical": ("HIST", "news knowledge", 10),
}

# A fixed wait before each phase on top of the RPS gate, because the vendor 429s anyway.
_ASKNEWS_PHASE_WAIT_SEC = 10.1


def _is_asknews_retryable(err: Exception) -> bool:
    """True iff err is one of AskNews's known-transient rate or concurrency errors.

    Matched on the message rather than on a status code, unlike the LLM paths that read
    ``llm_status_code``, because the AskNews SDK raises its own error classes and never
    subclasses ``openai.APIError``: see docs/research.md "AskNews dual-phase search".
    """
    msg = str(err).lower()
    return ("429" in msg) or ("rate limit" in msg) or ("concurrency limit" in msg)


async def _asknews_phase(
    sdk: Any,
    question_text: str,
    *,
    phase: str,
    tries: int,
    backoff: float,
    qid: int | None,
) -> tuple[Any, int]:
    """Run one AskNews search phase with its own retry loop; returns ``(articles, attempts_used)``.

    ``attempts_used`` is what lets the HISTORICAL phase spend only the budget HOT
    left over, so a phase that burned retries can't double the wall-clock cost.
    Only transient rate/concurrency errors are retried; anything else raises
    immediately, and an exhausted ladder re-raises the last error.
    """
    label, strategy, n_articles = _ASKNEWS_PHASES[phase]
    last_exc: Exception | None = None
    attempt_used = 0
    for attempt in range(1, tries + 1):
        attempt_used = attempt
        try:
            if phase == "historical":
                logger.info(
                    f"AskNews {label} attempt {attempt}/{tries}: Passing rate gate before historical news call..."
                )
                await _asknews_rate_gate()
                logger.info(f"AskNews {label} attempt {attempt}/{tries}: Calling historical news...")
            else:
                logger.info(f"AskNews {label} attempt {attempt}/{tries}: Calling latest news...")
                await _asknews_rate_gate()
            response = await sdk.news.search_news(
                query=question_text,
                n_articles=n_articles,
                return_type="both",
                strategy=strategy,
            )
            articles = response.as_dicts
            record_raw_research(qid=qid, provider="asknews", phase=phase, payload=articles)
            return articles, attempt_used
        except Exception as e:  # HARNESS-SCAN-EXEMPT-broad-except  # retry-then-reraise: non-retryable and exhausted ladder both re-raise below
            last_exc = e
            if not _is_asknews_retryable(e):
                raise
            if attempt < tries:
                sleep_for = backoff * (10 + 3**attempt)
                await asyncio.sleep(sleep_for)
            else:
                assert last_exc is not None
                raise last_exc  # noqa: B904  # re-raises the exception being handled; `from` would self-reference
    raise AssertionError("unreachable: the retry ladder either returns or raises")


def _asknews_provider() -> ResearchCallable:
    get_asknews_semaphore()

    async def _fetch(question: MetaculusQuestion) -> str:
        """Fetch AskNews research for one question under a hard wall-clock cap.

        The inner retry ladder fails fast on non-retryable errors but leaves a genuine
        network hang (connect stall, DNS hang, an unclosed stream) unbounded, so the cap
        is what stops a stuck AskNews call holding the whole research phase hostage.
        """
        return await asyncio.wait_for(
            _fetch_impl(question.question_text, qid=getattr(question, "id_of_question", None)),
            timeout=ASKNEWS_WALL_TIMEOUT,
        )

    async def _fetch_impl(question_text: str, *, qid: int | None = None) -> str:
        assert _ASKNEWS_GLOBAL_SEMAPHORE is not None
        tries = max(1, int(ASKNEWS_MAX_TRIES))
        backoff = float(ASKNEWS_BACKOFF_SECS)

        async with _ASKNEWS_GLOBAL_SEMAPHORE:
            from asknews_sdk import (  # noqa: PLC0415  # late import: tests patch asknews_sdk.AsyncAskNewsSDK at source
                AsyncAskNewsSDK,
            )

            client_id = os.getenv(ASKNEWS_CLIENT_ID_ENV)
            secret = os.getenv(ASKNEWS_SECRET_ENV)
            if not client_id or not secret:
                raise ValueError("ASKNEWS_CLIENT_ID and ASKNEWS_SECRET environment variables must be set")

            logger.info(
                f"AskNews: Using custom integration, client_id={client_id[:8]}..."  # HARNESS-SCAN-EXEMPT-subsampling: a log-line preview, not a data reduction
            )

            async with AsyncAskNewsSDK(
                client_id=client_id,
                client_secret=secret,
                scopes={"news"},
            ) as sdk:
                logger.info(f"AskNews: Waiting {_ASKNEWS_PHASE_WAIT_SEC}s before hot news call...")
                await asyncio.sleep(_ASKNEWS_PHASE_WAIT_SEC)
                hot_articles, hot_attempt_used = await _asknews_phase(
                    sdk, question_text, phase="hot", tries=tries, backoff=backoff, qid=qid
                )
                assert hot_articles is not None

                logger.info(f"AskNews: Waiting {_ASKNEWS_PHASE_WAIT_SEC}s before historical news call...")
                await asyncio.sleep(_ASKNEWS_PHASE_WAIT_SEC)
                historical_articles, _ = await _asknews_phase(
                    sdk,
                    question_text,
                    phase="historical",
                    tries=max(1, tries - (hot_attempt_used - 1)),
                    backoff=backoff,
                    qid=qid,
                )
                assert historical_articles is not None

                logger.info(
                    f"AskNews: Got {len(hot_articles)} hot articles, {len(historical_articles)} historical articles"
                )

                formatted_articles = _format_asknews_dual_sections(
                    hot_articles=hot_articles,
                    historical_articles=historical_articles,
                )
                if not formatted_articles:
                    logger.warning(
                        f"ASKNEWS_NO_ARTICLES: question={qid} "
                        f"hot={len(hot_articles)} historical={len(historical_articles)}"
                    )
                    record_provider_detail(qid, "asknews", {"sources": {"articles": "empty(no_articles)"}})
                    return ""

                logger.info(
                    f"AskNews: Success, got {len(formatted_articles)} chars from {len(hot_articles)} hot + {len(historical_articles)} historical articles"
                )
                return formatted_articles

    return _fetch


def _format_single_article(article: Any) -> str:
    pub_date = article.pub_date.strftime("%B %d, %Y %I:%M %p")
    return (
        f"**{article.eng_title}**\n{article.summary}\n"
        f"Original language: {article.language}\n"
        f"Publish date: {pub_date}\n"
        f"Source:[{article.source_id}]({article.article_url})\n\n"
    )


def _format_asknews_dual_sections(
    hot_articles: list[Any],
    historical_articles: list[Any],
) -> str:
    """Format AskNews articles into two labeled sections: Historical Context and Recent Developments.

    Deduplicates within each list and cross-deduplicates (hot articles that duplicate historical
    URLs are removed), historical section first. Both phases empty returns ``""``, never a prose
    "no articles" sentence, and this stays pure: the WARN and the ``lost=articles:...`` token
    belong to the caller that owns the qid. Both rules and their receipts:
    docs/research.md "AskNews dual-phase search".
    """
    hist_deduped = _dedup_articles_by_url(historical_articles) if historical_articles else []
    hot_deduped = _dedup_articles_by_url(hot_articles) if hot_articles else []

    if hist_deduped:
        hist_urls = {_normalize_url_for_dedup(str(a.article_url)) for a in hist_deduped}
        hot_deduped = [a for a in hot_deduped if _normalize_url_for_dedup(str(a.article_url)) not in hist_urls]

    if not hist_deduped and not hot_deduped:
        return ""

    total_before = len(historical_articles) + len(hot_articles)
    total_after = len(hist_deduped) + len(hot_deduped)
    removed = total_before - total_after
    if removed > 0:
        logger.info(f"AskNews URL dedup: {total_before} -> {total_after} (removed {removed} duplicates)")

    formatted_articles = "Here are the relevant news articles:\n\n"

    if hist_deduped:
        sorted_hist = sorted(hist_deduped, key=lambda x: x.pub_date, reverse=True)
        formatted_articles += "## Historical Context & Background\n\n"
        for article in sorted_hist:
            formatted_articles += _format_single_article(article)

    if hot_deduped:
        sorted_hot = sorted(hot_deduped, key=lambda x: x.pub_date, reverse=True)
        formatted_articles += "\n## Recent Developments & Current News\n\n"
        for article in sorted_hot:
            formatted_articles += _format_single_article(article)

    return formatted_articles


async def _invoke_exa_research(
    default_llm: GeneralLlm,
    prompt: str,
    *,
    include_works_cited_list: bool | None = None,
    use_brackets_around_citations: bool | None = None,
) -> str:
    citation_kwargs: dict[str, bool] = {}
    if include_works_cited_list is not None:
        citation_kwargs["include_works_cited_list"] = include_works_cited_list
    if use_brackets_around_citations is not None:
        citation_kwargs["use_brackets_around_citations"] = use_brackets_around_citations
    searcher = SmartSearcher(
        model=default_llm,
        temperature=None,  # ignored on a preconfigured GeneralLlm; keeps litellm off the fallback str path
        num_searches_to_run=2,
        num_sites_per_search=10,
        **citation_kwargs,
    )
    return await searcher.invoke(prompt)


def _exa_provider(default_llm: GeneralLlm) -> ResearchCallable:
    async def _fetch(question: MetaculusQuestion) -> str:
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            " you a question they intend to forecast on. To be a great assistant, you generate"
            " a concise but detailed rundown of the most relevant news, including if the question"
            " would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question.question_text}"
        )
        # Pin the current citation shape against a future framework-default flip.
        return await _invoke_exa_research(
            default_llm,
            prompt,
            include_works_cited_list=False,
            use_brackets_around_citations=False,
        )

    return _fetch


async def _invoke_perplexity_research(
    prompt: str,
    *,
    use_open_router: bool,
    api_key: str | _OmittedPerplexityApiKey | None = _OMITTED_API_KEY,
) -> str:
    model_name = PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER if use_open_router else PERPLEXITY_RESEARCH_MODEL
    metadata = llm_call_metadata("perplexity_research", plain_llm_key_alias(model_name))
    model_kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": None,  # provider-default sampling, pinned against a future GeneralLlm default flip
        "allowed_tries": 1,  # the elapsed-gated wrapper below is the sole retry owner
        "metadata": metadata,
    }
    if api_key is not _OMITTED_API_KEY:
        model_kwargs["api_key"] = api_key
    model = GeneralLlm(**model_kwargs)
    return await invoke_with_transient_retry(
        lambda: model.invoke(prompt),
        wall_timeout=PERPLEXITY_WALL_TIMEOUT,
        label="perplexity_research",
    )


def _perplexity_provider(use_open_router: bool = False, is_benchmarking: bool = False) -> ResearchCallable:
    async def _fetch(question: MetaculusQuestion) -> str:
        """Run Perplexity research, dropping the market-odds ask when benchmarking (leakage).

        The market-odds ask is interpolated from ``OUTSIDE_VENUE_MARKET_ODDS_POLICY`` rather than
        restated, because a second copy drifted once: docs/research.md "Primary provider: a
        priority ladder".
        """
        prediction_markets_instruction = (
            "" if is_benchmarking else f"In addition to news, cover: {OUTSIDE_VENUE_MARKET_ODDS_POLICY}\n"
        )
        prompt = (
            "You are an assistant to a superforecaster.\n"
            "Generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.\n"
            f"{prediction_markets_instruction}"
            "Do not produce forecasts yourself. Provide data for the superforecaster.\n\n"
            f"Question:\n{question.question_text}"
        )
        # Omitting api_key preserves LiteLLM's provider-specific environment lookup.
        return await _invoke_perplexity_research(prompt, use_open_router=use_open_router)

    return _fetch


def build_native_search_llm(
    model_slug: str | None = None,
    *,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    search_context_size: str | None = None,
    role: str = "native_search",
) -> GeneralLlm:
    """Build a GeneralLlm configured for OpenAI native web search via OpenRouter.

    Shared by the native search research provider, the targeted research module, the gap-fill
    resolver and the resolver probe (``scripts/probes/gap_fill_resolver_probe.py``). ``role`` is
    the CREDIT_ROLE_SPEND line the completions book under, so the search spend lines stay
    separable in the run log.

    Reasoning effort and verbosity come from the NATIVE_SEARCH_REASONING_EFFORT /
    NATIVE_SEARCH_VERBOSITY env at call time (so workflow overrides take effect without
    re-importing) unless the caller passes an explicit override, which always wins; an empty
    string from either source drops the kwarg. ``search_context_size`` overrides the
    NATIVE_SEARCH_CONTEXT_SIZE constant the same way; there is no env read because production
    runs one size. Key routing and the ``allowed_tries=1`` rationale: docs/research.md
    "OpenAI native search".
    """
    base_model = model_slug or os.getenv(NATIVE_SEARCH_MODEL_ENV, NATIVE_SEARCH_DEFAULT_MODEL)
    model_with_search = f"openrouter/{base_model}"

    kwargs: dict = {
        "model": model_with_search,
        "role": role,
        "temperature": None,  # provider-default sampling, pinned against a future GeneralLlm default flip
        "max_tokens": NATIVE_SEARCH_MAX_TOKENS,
        "timeout": NATIVE_SEARCH_TIMEOUT,
        "allowed_tries": 1,  # the caller's wall is the budget; a same-call retry only multiplies a drip (2026-05-20)
        "plugins": [{"id": "web", "max_results": NATIVE_SEARCH_MAX_RESULTS, "engine": "native"}],
        "web_search_options": {"search_context_size": search_context_size or NATIVE_SEARCH_CONTEXT_SIZE},
    }

    effort = (
        reasoning_effort
        if reasoning_effort is not None
        else os.getenv(NATIVE_SEARCH_REASONING_EFFORT_ENV, NATIVE_SEARCH_REASONING_EFFORT_DEFAULT)
    )
    if effort:
        kwargs["reasoning"] = {"effort": effort}

    # Top-level, not inside extra_body: the canonical litellm / OpenRouter form for gpt-5 verbosity.
    verbosity_value = (
        verbosity if verbosity is not None else os.getenv(NATIVE_SEARCH_VERBOSITY_ENV, NATIVE_SEARCH_VERBOSITY_DEFAULT)
    )
    if verbosity_value:
        kwargs["verbosity"] = verbosity_value

    return build_llm_with_openrouter_fallback(**kwargs)


def _native_search_provider(
    model_slug: str | None = None,
    is_benchmarking: bool = False,
) -> ResearchCallable:
    """Research provider using models with native web search capability via OpenRouter :online suffix."""

    async def _fetch(question: MetaculusQuestion) -> str:
        from metaculus_bot.constants import (  # noqa: PLC0415  # late read: tests patch this constant on the constants module
            NATIVE_SEARCH_WALL_TIMEOUT,
        )

        llm = build_native_search_llm(model_slug)
        prompt = web_research_prompt(
            question.question_text,
            resolution_criteria=question.resolution_criteria or "",
            fine_print=question.fine_print or "",
            # The MC ballot (None on other types): a model can only search names it was shown (q44952).
            options=getattr(question, "options", None),
            is_benchmarking=is_benchmarking,
            citation_style="markdown",
        )
        logger.info(f"NativeSearch: Calling {llm.model} for research")
        # The wrapper owns the wall cap and recovers instant aiohttp blips without retrying a stall.
        result = await invoke_with_transient_retry(
            lambda: llm.invoke(prompt), wall_timeout=NATIVE_SEARCH_WALL_TIMEOUT, label="native_search"
        )
        logger.info(f"NativeSearch: Got {len(result)} chars from {llm.model}")
        record_raw_research(
            qid=getattr(question, "id_of_question", None),
            provider="native_search",
            payload=result,
        )
        # Only the forecaster-facing text is stripped; the raw log above keeps the untouched payload.
        return _strip_utm_source(result)

    return _fetch


# Public alias for the native search provider (used by tests and external callers)
native_search_provider = _native_search_provider


# ---------------------------------------------------------------------------
# Strategy selector
# ---------------------------------------------------------------------------


def _forced_provider_choice(
    forced_lc: str,
    *,
    default_llm: GeneralLlm | None,
    exa_callback: ResearchCallable | None,
    perplexity_callback: ResearchCallable | None,
    openrouter_callback: ResearchCallable | None,
    is_benchmarking: bool,
) -> tuple[ResearchCallable, str] | None:
    """Resolve an explicit ``RESEARCH_PROVIDER`` override, or None to fall through to auto."""
    if forced_lc == "asknews":
        # Fail fast if creds missing to make misconfig obvious
        if not (os.getenv(ASKNEWS_CLIENT_ID_ENV) and os.getenv(ASKNEWS_SECRET_ENV)):
            raise ValueError("RESEARCH_PROVIDER=asknews requires ASKNEWS_CLIENT_ID and ASKNEWS_SECRET to be set")
        return _asknews_provider(), "asknews"
    if forced_lc == "exa":
        if exa_callback is not None:
            return exa_callback, "exa"
        if default_llm is None:
            raise ValueError("RESEARCH_PROVIDER=exa requires default_llm or exa_callback to be provided")
        return _exa_provider(default_llm), "exa"
    if forced_lc == "perplexity":
        if perplexity_callback is not None:
            return perplexity_callback, "perplexity"
        return _perplexity_provider(use_open_router=False, is_benchmarking=is_benchmarking), "perplexity"
    if forced_lc == "openrouter":
        if openrouter_callback is not None:
            return openrouter_callback, "openrouter"
        return _perplexity_provider(use_open_router=True, is_benchmarking=is_benchmarking), "openrouter"
    # Any other value behaves as auto
    return None


def _auto_provider_choice(
    *,
    default_llm: GeneralLlm | None,
    exa_callback: ResearchCallable | None,
    perplexity_callback: ResearchCallable | None,
    openrouter_callback: ResearchCallable | None,
    is_benchmarking: bool,
) -> tuple[ResearchCallable, str]:
    """First provider whose credentials are present, in the documented priority order."""
    if os.getenv(ASKNEWS_CLIENT_ID_ENV) and os.getenv(ASKNEWS_SECRET_ENV):
        return _asknews_provider(), "asknews"

    if os.getenv(EXA_API_KEY_ENV):
        if exa_callback is not None:
            return exa_callback, "exa"
        if default_llm is None:
            raise ValueError("default_llm must be provided for Exa research provider")
        return _exa_provider(default_llm), "exa"

    if os.getenv(PERPLEXITY_API_KEY_ENV):
        if perplexity_callback is not None:
            return perplexity_callback, "perplexity"
        return _perplexity_provider(use_open_router=False, is_benchmarking=is_benchmarking), "perplexity"

    if os.getenv(OPENROUTER_API_KEY_ENV):
        if openrouter_callback is not None:
            return openrouter_callback, "openrouter"
        return _perplexity_provider(use_open_router=True, is_benchmarking=is_benchmarking), "openrouter"

    async def _empty(_: MetaculusQuestion) -> str:
        return ""

    return _empty, "none"


def choose_provider_with_name(
    default_llm: GeneralLlm | None = None,
    *,
    exa_callback: ResearchCallable | None = None,
    perplexity_callback: ResearchCallable | None = None,
    openrouter_callback: ResearchCallable | None = None,
    is_benchmarking: bool = False,
) -> tuple[ResearchCallable, str]:
    """Return a research coroutine and its provider name.

    Priority order replicates pre-refactor behaviour:
    1. AskNews (ASKNEWS_CLIENT_ID & ASKNEWS_SECRET)
    2. Exa.ai (EXA_API_KEY)
    3. Perplexity (PERPLEXITY_API_KEY)
    4. Perplexity via OpenRouter (OPENROUTER_API_KEY)
    5. Fallback stub that returns an empty string.

    ``RESEARCH_PROVIDER`` forces a specific provider; an unrecognized value falls
    through to the priority order above.
    """
    forced = os.getenv(RESEARCH_PROVIDER_ENV)
    if forced:
        choice = _forced_provider_choice(
            forced.strip().lower(),
            default_llm=default_llm,
            exa_callback=exa_callback,
            perplexity_callback=perplexity_callback,
            openrouter_callback=openrouter_callback,
            is_benchmarking=is_benchmarking,
        )
        if choice is not None:
            return choice

    return _auto_provider_choice(
        default_llm=default_llm,
        exa_callback=exa_callback,
        perplexity_callback=perplexity_callback,
        openrouter_callback=openrouter_callback,
        is_benchmarking=is_benchmarking,
    )


# ---------------------------------------------------------------------------
# URL normalization and dedup helpers (simple, robust, testable)
# ---------------------------------------------------------------------------


# Captures the leading separator and an optional trailing `&` so removal keeps the query well-formed.
_UTM_SOURCE_OPENAI_RE = re.compile(r"([?&])utm_source=openai\b(&)?")


def _strip_utm_source(text: str) -> str:
    """Drop ``utm_source=openai`` tracking params from URLs in native-search text.

    Handles the param as the sole query param (``?utm_source=openai`` -> ``), the
    first of several (``?utm_source=openai&a=b`` -> ``?a=b``), or a later one
    (``&utm_source=openai`` -> ``). Other query params are preserved. Operates on
    the free-text research blob (the native-search LLM emits the URLs inline).
    """

    def _repl(match: re.Match[str]) -> str:
        """Keep the leading separator only when another param follows (``?a&utm&b`` -> ``?a&b``)."""
        return match.group(1) if match.group(2) else ""

    return _UTM_SOURCE_OPENAI_RE.sub(_repl, text)


def _normalize_url_for_dedup(url: str) -> str:
    """Return a canonicalized URL for deduplication.

    - Lowercase scheme and netloc
    - Drop fragment
    - Remove common tracking params (utm_*, gclid, fbclid, igshid, ref, mc_cid, mc_eid)
    - Sort remaining query params, and strip a trailing slash inside a param value (b=2/ -> b=2)
    - Strip single trailing slash on path
    - Normalize mobile and AMP variants (m. subdomain, trailing /amp)
    """
    if not url:
        return url
    parts = urlsplit(url)
    scheme = (parts.scheme or "").lower()
    netloc = (parts.netloc or "").lower()

    if netloc.startswith("m."):
        netloc = netloc[2:]

    path = parts.path or ""
    if path.endswith("/amp"):
        path = path[:-4]
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    drop_keys = {"gclid", "fbclid", "igshid", "ref", "mc_cid", "mc_eid"}
    kept_params = []
    for k, raw_value in parse_qsl(parts.query, keep_blank_values=True):
        if k.startswith("utm_") or k in drop_keys:
            continue
        value = raw_value.rstrip("/") if isinstance(raw_value, str) and raw_value.endswith("/") else raw_value
        kept_params.append((k, value))
    kept_params.sort()
    query = urlencode(kept_params, doseq=True)

    fragment = ""

    return urlunsplit((scheme, netloc, path, query, fragment))


def _dedup_articles_by_url(articles: list[Any]) -> list[Any]:
    """Order-preserving deduplication of articles by normalized URL.

    Articles may be objects with attribute `article_url` or dicts with key `article_url`.
    Items without a URL are kept.
    """
    seen: set[str] = set()
    result: list[Any] = []
    for item in articles:
        url = item.get("article_url") if isinstance(item, dict) else getattr(item, "article_url", None)  # type: ignore[unreachable]

        if not url:
            result.append(item)
            continue

        norm = _normalize_url_for_dedup(str(url))
        if norm in seen:
            continue
        seen.add(norm)
        result.append(item)

    return result
