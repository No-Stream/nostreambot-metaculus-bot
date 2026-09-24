# Research subsystem

This is the reference for how the bot gathers evidence before any forecaster
LLM sees a question. If you want the forecasting/aggregation side, see the other
docs; this file covers only the research phase.

For a quick map of the code: orchestration lives in
`metaculus_bot/research/orchestrator.py`, provider selection in
`metaculus_bot/research/providers.py`, and each add-on provider is its own module
under `metaculus_bot/research/`.

## The shape of a research run

Every question goes through `ResearchOrchestrator.run_research`
(`research/orchestrator.py`). The flow is:

1. **Cache check.** In benchmarking runs the orchestrator caches research per
   question id so a replayed backtest doesn't re-pay for the same pull. Live runs
   don't cache (`_lookup_research_cache`, `research/orchestrator.py`).
2. **Provider selection.** `_select_research_providers`
   (`research/orchestrator.py`) picks exactly one **primary** provider, then
   appends every enabled **add-on** provider. Each add-on is behind its own env
   flag, so the set that actually runs depends on configuration.
3. **Parallel fan-out.** `_run_providers_parallel`
   (`research/orchestrator.py`) runs all selected providers concurrently via
   `asyncio.gather`. One provider failing never kills the phase; the failure is
   recorded as a per-provider result and the rest proceed. Global concurrency is
   bounded by a semaphore sized by `DEFAULT_MAX_CONCURRENT_RESEARCH`.
4. **Assembly.** Each provider's output is prefixed with a fixed `##` section
   header (`provider_header`, `research/section_format.py`) and the sections
   are joined with `---` rules. Any stray `#`/`##` heading inside a provider's
   body is demoted two levels so it never competes with the section headers.
5. **Gap-fill.** Two second-pass gap-fill passes (v1 and v2) run concurrently on
   the assembled bundle and append their own sections. See "Gap-fill" below.
6. **Diagnostics + persistence.** A provider-diagnostics block (which provider
   succeeded, char counts, latency) is logged and stashed for the published
   Metaculus comment, but is deliberately kept OUT of the text forecasters read.
   If a research sink is wired, the full record is written for backtest replay.

The single string returned from `run_research` is what every forecaster in the
ensemble reads verbatim.

`research/section_format.py` also owns `detect_providers` and `detect_gap_fill`,
which decode section headers for both the log and comment backfill scripts. The
scripts share detection rules while retaining their separate archive record builders
and provenance.

### Orchestrator implementation notes (`research/orchestrator.py`)

The prose below was carried as comment blocks and long docstrings inside
`metaculus_bot/research/orchestrator.py` until 2026-09-09, when the AST smell scanner's comment
rules were applied to that module. Each block left one line of why in the code plus a pointer to
this section, and the entries here are in file order, each naming the attribute or function it came
from. No executable code changed in that pass.

**The `__all__` re-export (module level).** `_demote_inner_headings` moved to
`research/section_format.py` but is still imported from this module path by callers outside the
package, and the re-export is what keeps that working. It also keeps the auto-formatter from
stripping an otherwise-unused import.

**`_comment_diagnostics` (in `__init__`).** Comment-bound provider-diagnostics blocks, keyed by
question id. `run_research` returns forecaster-clean text; `TemplateForecaster` pops the block via
`pop_provider_diagnostics` when assembling the published comment.

**`provider_failure_count`.** A per-run count of research-provider calls that FAILED, meaning any
exception rather than only timeouts: the generic failure branch in `_run_one` never inspects the
exception type. It excludes the expected off-season AskNews subscription error, which reports
`status="inactive"` and is not alertable.

**`summarizer_failure_count`.** A per-run count of AskNews summarizer soft-fails, either a transient
LLM error or blank output, each of which ships raw unscreened articles in place of the analyst
briefing. Alertable by operator decision 2026-07-26 on quality grounds: provider status is computed
from POST-summarizer text, so a permanently dead summarizer would otherwise degrade every briefing
while AskNews keeps reporting `status="ok"`.

**`gap_fill_v1_error_count`.** A v1 analyzer, schema or resolver failure is alertable even though
the stage returns the surviving first-pass research and publication continues. The v1 stage reports
one failure per pass through its callback, including partial schema drift, while a valid empty
analysis and legitimate triage drops stay at zero; an unexpected escape from the stage guard is
counted like the v2 escape path. It is surfaced as `gap_fill_v1_errors` in the forecaster snapshot,
the run summary and `alertable_count`, separately from budget cuts and v2 failures.

**`gap_fill_v2_error_count`.** Genuine gap-fill-v2 CRASHES only, not idle "driver found nothing"
runs and not deadline hits. It mirrors `provider_failure_count`: surfaced to the forecaster as
`_gap_fill_v2_error_count` and folded into `alertable_count`, so a dead v2 feature reddens CI. A
dead-on-arrival bug, the fastapi eager-import defect being the worked example, bumps this on EVERY
question and reddens CI immediately, while a one-off transient provider 500 bumps it once and gives
an accepted rare false alarm, because investigating that beats silently missing a dead feature.
`run_research` holds the three mutually exclusive bump points.

**`research_budget_cut_count`.** A per-QUESTION count of research thinned by the time budget OFF the
fast path: a provider cancelled at the research-phase deadline, or gap-fill cut or skipped for
budget, on a question whose window was wide enough that `fast_path` never fired. The fast path has
its own alertable counter (`_time_budget_fast_path_count`, forecaster-side); without this second one
the band just above the threshold, where the research window can still sit under research's
configured worst case, degraded silently and the end-of-run census read all-clear. It is
deduplicated per question through the `_research_budget_cut_seen` set, so a question losing a
provider AND both gap-fill passes counts once, and fast-path questions are excluded so nothing
double-charges.

**`run_research`: the deadline-cancelled provider.** A provider cancelled at the research-phase
deadline is budget-driven degradation, and off the fast path nothing else counts it, which is why
`_record_research_budget_cut` is called there.

**`run_research`: the diagnostics seam.** The provider-diagnostics block is deliberately NOT
appended to the returned research, because forecasters and the gap-fill v2 driver brief consume that
text verbatim and must never see it. It still reaches its three destinations: the INFO log line just
below its construction, the research archive via the sink's `provider_diagnostics_block` kwarg, and
the published comment, stashed per question id in `run_research` and popped by
`TemplateForecaster.pop_provider_diagnostics` at comment-build time.

**`run_research`: the sink's two provider lists.** `provider_results` is the authoritative
per-provider outcome; `providers_used` is kept only for legacy archive readers.

**`_select_research_provider`: the two Perplexity callbacks.** Each rung gets the vendor its env var
pays for. Binding the bare `_call_perplexity` here would hand priority 3 the method's
OpenRouter-first default, which is deliberate on the AskNews-fallback path (where
`_attempt_research_fallback` prefers the cheap route) and wrong here, because it collapses the
ladder's two Perplexity rungs into one and passes `api_key=None` whenever only `PERPLEXITY_API_KEY`
is set.

**`_select_research_providers`: what the fast path can and cannot shed.** The optional providers all
run CONCURRENTLY with the primary, whose own worst case (AskNews 300 s plus summarizer 300 s,
sequential inside one provider) is the phase's longest configured pole. Dropping the cheap
hard-capped providers (`resolution_source` 45 s, `prediction_market` 150 s, `ts_anchor` 20 s, the
financial classifier 30 s) therefore cannot shorten the phase and only discards the resolution
ground truth. What the fast path CAN shed is the measured tail: `native_search` is the phase's
slowest provider on 51.5% of questions and reached 292 s against the primary's 110 s measured worst
case (`scratch/residual_2026-08-24/time_budget_design.md`).

**`_select_research_providers`: `resolution_source` stays on the fast path.** It is cheap and hard
capped at 45 s, so it stays in; the flag is handed to it so its two EXPENSIVE ladder rungs, the
Chromium launch and the paid reader, decline instead, while its direct fetch and cheap rungs run.

**`_run_providers_parallel`: the raw AskNews capture.** `asknews_raw_holder` carries the raw
pre-summarization AskNews article text for the research archive, added as 2026-07-18 audit hygiene:
the archive otherwise stores only the post-summarization briefing, so FETCH-versus-SUMMARIZE
attribution and summarizer replays required fresh paid pulls. It stays empty when AskNews did not
run, errored, or fell back to an already-prose provider.

**`_run_one`: draining the provider-detail registry.** A multi-source provider records its
per-source outcome into the (question id, provider) registry during the call, so the question id is
read up front and the entry drained here. That is what makes partial upstream loss, Kalshi dropped
over the size cap being the worked example, ride into `ProviderResult.details` instead of vanishing
behind a healthy `ok`.

**`_run_one`: which providers are summarized.** AskNews returns raw article markdown with no LLM
prose, so it is summarized into an analyst briefing. Every other provider already emits LLM-written
prose (native search, Gemini, Perplexity, Exa) or deterministic tables (financial data, prediction
markets), so they pass through raw and take no lossy second-pass summarization. When AskNews fails
and the run falls back to Perplexity or Exa, that fallback is already prose, so summarization is
skipped there too.

**`_run_one`: empty raw skips the summarizer entirely.** There is nothing to brief from, and asking
anyway spends a call to get either a refusal or an invented briefing, since the summarizer prompt
has no no-data escape. AskNews has already recorded an `articles: empty(no_articles)` loss token, so
the `empty` status stays distinguishable from a skipped run.

**`_run_one`: the `CancelledError` branch.** A deadline-cancelled provider must drain its registry
entry too. `CancelledError` is a `BaseException` and would otherwise skip both drain paths, leaving
exactly the stale same-key entry the `except Exception` below it exists to prevent. It is re-raised
so the caller still records the cancellation as `status="deadline"`.

**`_run_one`: the `except Exception` drain.** Drain and discard any partial detail the provider
recorded before raising: an errored result carries the error, not source detail, and a stale entry
must not leak into a later same-key call.

**`_fetch_research_with_fallback`: a vendor swap is degradation.** A swap on the PRIMARY provider is
real degradation rather than a success: a different index, a different recency profile, and, because
the fallback is already prose, no AskNews summarizer pass, so the briefing loses the per-article
relevance gate, the `[PRE-WINDOW]` labeling and the recency reordering that the 2026-07-18 audit
made load-bearing. It used to bump no counter and record no detail, so it read as healthy; a
per-source loss token is now recorded so the diagnostics line and the schema-v2 archive carry it.
Deliberately NOT a new alertable counter: folding one into `alertable_count` changes what CI treats
as red, which is the operator's call.

**`_attempt_research_fallback`: the ordering and the header keys.** The cost-ordered fallback list
and why it diverges from the primary ladder are in "AskNews fallback (primary-only)" above. One
further detail: the names this function returns are the same keys `provider_header` maps, so the
section header follows the vendor that actually answered, automatically.

**`_call_perplexity`: the market-odds policy.** This prompt carries the same narrowed market-odds
policy as `web_research_prompt` and the direct-Perplexity provider, interpolated from the one
definition in `prompts` rather than restated, because this copy kept the retired blanket "briefly
research prediction markets" ask after that policy had been narrowed to the venues the live snapshot
cannot cover. The no-speculation tail is this prompt's own and stays: it is an anti-fabrication rule
about an empty result, not a second opinion on which venues to read.

**`_call_perplexity`: explicit credential routing.** Keep this call site's routing explicit. Direct
Perplexity passes `None`, while the OpenRouter route resolves its key before construction.

**The degradation-counter property surface.** The research side's degradation counters live in
`degradation_views`, along with their long "why is this alertable" rationales. The five one-line
members at the bottom of `ResearchOrchestrator` are the orchestrator-attribute surface that
`forecaster.py`, `cli.py` and `degradation_counters.py` read them through.

## Primary provider: a priority ladder

There is always exactly one primary provider, chosen by
`choose_provider_with_name` (`research/providers.py`). It walks a fixed
priority order and returns the first provider whose credentials are present:

1. **AskNews** if `ASKNEWS_CLIENT_ID` and `ASKNEWS_SECRET` are set. This is the
   production case.
2. **Exa.ai** (`SmartSearcher`) if `EXA_API_KEY` is set: a generic rundown
   (`_exa_provider`, `research/providers.py`).
3. **Perplexity direct** if `PERPLEXITY_API_KEY` is set. Model:
   `PERPLEXITY_RESEARCH_MODEL` (`constants.py`); the function is
   `_perplexity_provider` (`research/providers.py`), and its prompt explicitly
   asks for prediction-market consideration unless the run is benchmarking.
4. **Perplexity via OpenRouter** if `OPENROUTER_API_KEY` is set. Same function
   called with `use_open_router=True`, same model, prefixed for the OpenRouter
   route: `PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER`.
5. **Empty stub** if none of the above: research is just the add-on providers.

In production the AskNews credentials are present, so Exa and the two Perplexity
routes never run as the primary. They are fallbacks, not peers. To force a
specific primary regardless of credentials, set `RESEARCH_PROVIDER=<name>`
(`asknews` / `exa` / `perplexity` / `openrouter`); any other value behaves as
auto. Forcing `asknews` without the AskNews creds fails loudly rather than
silently picking a different provider.

Exa and Perplexity client construction and invocation live in
`research/providers.py` (`_invoke_exa_research` and
`_invoke_perplexity_research`). Both the standalone provider factories and the
orchestrator use these helpers. Each caller retains its existing prompt and
constructor options: the Exa factory explicitly disables citation formatting,
while the orchestrator uses SDK defaults; the Perplexity factory omits `api_key`,
while the orchestrator passes `None` for direct access or resolves the OpenRouter
key. These distinctions remain part of the call contract.

The Perplexity prompt interpolates `OUTSIDE_VENUE_MARKET_ODDS_POLICY` rather than restating the
market-odds ask, because a second copy of it drifted once. This provider is the primary whenever
the AskNews credentials are absent, so its copy is live policy, and it kept the retired blanket
"consider all relevant prediction markets" wording after the first-pass prompt had been narrowed
to the venues the live market snapshot does not cover. Both helpers also pass `temperature=None`
to pin provider-default sampling against a future `GeneralLlm` default flip, and the Perplexity
model is built at `allowed_tries=1` so the elapsed-gated `invoke_with_transient_retry` wrapper is
the only retry owner on that path. On the Exa side `temperature` is ignored outright when the
model is a preconfigured `GeneralLlm`; `None` is what keeps litellm from applying a sampling
param on the fallback string path.

### AskNews fallback (primary-only)

AskNews is the only primary that gets a runtime fallback. If the AskNews fetch
raises, `_fetch_research_with_fallback` (`research/orchestrator.py`) tries a
prose provider instead, in a **different** order than the primary ladder:
OpenRouter-Perplexity first (cheapest, prose-returning), then direct Perplexity,
then Exa last (its `SmartSearcher` spins up its own multi-search loop, the most
expensive path). The primary ladder orders by index quality; this fallback list
orders by cost, because it only ever fires after AskNews has already failed.
It selects the first fallback with credentials; failure of that selected provider
does not advance to another fallback.

One AskNews error is treated specially: a `403011` "subscription is not currently
active" signature (`is_asknews_subscription_error`, `research/providers.py`) is
logged as `inactive` (an expected off-season state), not `errored`, so it doesn't
inflate the timeout counter or look like a real failure in diagnostics.

### AskNews dual-phase search

AskNews is the one provider that returns raw article text rather than
LLM-written prose, so it has two distinct stages: fetch, then summarize.

**Fetch** (`_asknews_provider`, `research/providers.py`) runs two phases
against the AskNews SDK, asking HISTORICAL for a larger article budget than HOT:

- **Phase 1, HOT:** `strategy="latest news"`.
- **Phase 2, HISTORICAL:** `strategy="news knowledge"`.

Both phases share a retry budget (`ASKNEWS_MAX_TRIES`) that only retries on
known-transient rate/concurrency errors (429, "rate limit", "concurrency
limit"); anything else raises immediately. A process-wide semaphore
(`ASKNEWS_MAX_CONCURRENCY`) and an RPS gate (`ASKNEWS_MAX_RPS`) throttle calls,
plus a fixed wait before each phase (`_ASKNEWS_PHASE_WAIT_SEC`, module-level in
`research/providers.py`), because the API rate-limits aggressively even when we
stay under our own limits. All three
throttles take env overrides, and the shipped `.env.template` deliberately sets a
*lower* RPS than the `constants.py` default: the constant is the ceiling, not the
operating point, so the two disagreeing is expected rather than a drift bug. A
hard wall-clock timeout (`ASKNEWS_WALL_TIMEOUT`) backstops a network hang so a
stuck AskNews call can't hold the whole phase hostage.

Two details of that machinery are easy to break on a later edit. The retry predicate
(`_is_asknews_retryable`) matches on the error message rather than on a status code, unlike the
LLM paths that read `llm_status_code`, because the AskNews SDK raises its own
`asknews_sdk.errors` classes carrying a `.code` (429000, 429001, 403011) and never subclasses
`openai.APIError`, so a status-based primitive reads None here and would switch this retry off
silently. And both process-wide asyncio primitives, the concurrency semaphore
(`get_asknews_semaphore`) and the lock guarding the RPS clock (`_get_asknews_rate_lock`), are
created lazily on first use. The lock's laziness is a latent-shape cleanup rather than a fix for
an observed failure: an `asyncio.Lock` binds to the running loop when it first creates a future,
and a lock left held at loop close does wedge later loops with "is bound to a different event
loop", which reproduces in isolation where a cancelled holder leaves a waiter queued. Driving the
real `_asknews_rate_gate` the same way, though, a second `asyncio.run` succeeds, because the
cancellation releases the lock before the loop closes. It is deliberately not paired with a
staleness check: detecting a stale binding needs a private `_get_loop` probe that reads clean in
exactly the case that later fails, so the check would be reassuring rather than effective.

The two article lists are formatted into two labeled sections, "Historical
Context & Background" and "Recent Developments & Current News", with within-list
and cross-list URL deduplication (`_format_asknews_dual_sections`,
`research/providers.py`). Dedup normalizes URLs first (drops tracking params,
`m.` mobile subdomains, `/amp` suffixes, fragments) so the same story from two
feeds collapses to one entry.

**Both phases empty returns `""`, not a "No articles were found" sentence.** That
sentence defeated every downstream empty guard at once: the orchestrator's
`has_output` check saw chars>0 and reported `ok`, the summarizer (whose prompt has
no no-data escape) was handed the sentence as its article set, and the resulting
briefing rendered under the AskNews header as though it were research. The
formatter now logs `ASKNEWS_NO_ARTICLES`, records an `articles: empty(no_articles)`
source loss so the diagnostics line reads `empty | 0 chars | lost=articles:...`
rather than a bare `empty`, and the orchestrator skips the summarizer call
entirely. Gemini's cited-link floor is the same pattern one provider over.

`_format_asknews_dual_sections` stays pure and does none of that reporting itself. The
`ASKNEWS_NO_ARTICLES` WARN and the `lost=articles:...` registry token belong to
`_asknews_provider`, which owns the question id; a formatter writing the module-global
provider-detail registry raced `_degraded_to_raw_articles`' write for the same key only by
accident of ordering.

The raw pre-summarization article markdown is captured separately and archived
(the `asknews_raw` field) so a later audit can replay the summarizer or attribute
a bad briefing to fetch-vs-summarize without paying for a fresh AskNews pull.

### AskNews summarizer (the analyst briefing)

Raw AskNews articles are compressed into an analyst briefing by an LLM before any
forecaster sees them (`_summarize_asknews`, `research/orchestrator.py`). The
summarizer model is a low-effort utility slot defined in `llm_configs.py`
(`SUMMARIZER_LLM`); it runs at `allowed_tries=1` and is wrapped in an
elapsed-gated broad retry (`invoke_with_broad_retry`, whose gate is
`TRANSIENT_RETRY_MAX_ELAPSED_S` in `llm_retry.py`) with a wall cap
(`SUMMARIZER_WALL_TIMEOUT`). If the summarizer hits a transient LLM error or returns
blank, the orchestrator soft-fails to the **raw** articles rather than dropping the
news entirely. A non-transient error
(a prompt-construction bug, a refactor's `AttributeError`) is allowed to
propagate, because that's a real bug, not a degradation to tolerate.

The prompt (`asknews_summarizer_prompt`, `prompts.py`) tells the model to
produce a comprehensive briefing that extracts every decision-relevant fact,
number, quote, and expert opinion, dates each one, and separates facts from
opinion. It also shares the source-provenance / trust-ladder vocabulary with the
web-research prompt. Several rules are load-bearing for calibration:

- **Never paraphrase numbers.** Percentages, probabilities, dates, and counts are
  copied exactly.
- **Pre-window flagging.** Any event that happened before the question opened,
  and so can't itself satisfy the resolution criteria, is tagged `[PRE-WINDOW]`
  but kept as base-rate context.
- **Single-source labeling.** A claim resting on one outlet is tagged
  `[SINGLE-SOURCE]` with its original hedges preserved; it's never promoted to a
  confirmed fact.

The 2026-07-18 revision to this prompt added three things worth calling out:

- **Evidence-age lead.** The briefing must open by stating the date of the newest
  article that *directly* bears on resolution ("Newest directly-relevant article:
  2026-07-14"), or say explicitly when nothing directly reports on the resolution
  quantity and the section is only background.
- **Supersession.** When a newer article supersedes an older one on the same fact
  (a withdrawal, an updated count, a final decision), the briefing states which
  version governs today and compresses the stale version to one line, instead of
  giving obsolete detail equal weight. Deadline/window questions must quote the
  underlying dates and rules rather than assert a conclusion.
- **Relevance screen + proportionality.** Each article is screened for direct
  bearing on the resolution criteria; anything off-topic is dropped and listed on
  a single "Screened out as not decision-relevant" line. Briefing length must
  track surviving decision-relevant content: comprehensive when there's real
  material, short when few articles survive, never padded to look thorough.

## Add-on providers (parallel, each independently gated)

On top of the single primary, every enabled add-on provider runs in parallel.
Each is behind its own env flag and produces its own `##` section. In production
all of these are on.

### OpenAI native search: `NATIVE_SEARCH_ENABLED`

OpenAI web search via OpenRouter's native web plugin
(`_native_search_provider` / `build_native_search_llm`,
`research/providers.py`). The model is `NATIVE_SEARCH_DEFAULT_MODEL`, run at the
reasoning effort and verbosity in `NATIVE_SEARCH_REASONING_EFFORT_DEFAULT` /
`NATIVE_SEARCH_VERBOSITY_DEFAULT`, under a per-request timeout
(`NATIVE_SEARCH_TIMEOUT`) and a hard wall-clock cap
(`NATIVE_SEARCH_WALL_TIMEOUT`) set just above it. Model, effort, and verbosity are
overridable via `NATIVE_SEARCH_MODEL` / `NATIVE_SEARCH_REASONING_EFFORT` /
`NATIVE_SEARCH_VERBOSITY`. On the wire the effort goes out as
`reasoning={"effort": ...}` and the verbosity as a top-level `verbosity` kwarg, which is the
canonical litellm and OpenRouter form for gpt-5. An earlier version tucked the verbosity inside
`extra_body`, which worked because OpenRouter merges the body, but the top-level form matches the
docs and survives any future `extra_body` validation.

The web plugin itself runs at `NATIVE_SEARCH_CONTEXT_SIZE` (`constants.py`, currently "high"),
which is OpenAI's `search_context_size`: it sets how much retrieved page text comes back as input
tokens, and that text is billed. `build_native_search_llm` takes a per-call
`search_context_size` override the same way it takes `reasoning_effort` and `verbosity`, with no
env read behind it because production runs a single size. The one consumer of that override is
`scripts/probes/gap_fill_resolver_probe.py`, a paid, operator-gated probe that compares the
gap-fill resolver's answers across context sizes and models.

The forecaster-facing text goes through `_strip_utm_source`, which removes the
`?utm_source=openai` param OpenAI native search tags onto every citation URL. It is pure tracking
noise that would otherwise be fanned into every forecaster prompt and into the published comment;
the raw research log keeps the untouched payload, so archival fidelity is unaffected.

The model migrated on 2026-07-09 to `gpt-5.6-sol`, then on 2026-07-17 to
`gpt-5.6-terra` per the blind research-role audit
(`scratch/research_role_audit_2026-07-17/`: terra 1st, sol 2nd, luna 3rd; verdict
"MARGINAL EDGE", terra at −42% cost). Effort has been low since 2026-05-20 for
latency reasons; see `constants.py`.

The model is built at `allowed_tries=1` on purpose: an earlier incident
(2026-05-20) had OpenRouter drip whitespace keep-alive bytes for over eight
minutes before returning malformed JSON, and retrying that call just multiplies
the wait. `NATIVE_SEARCH_WALL_TIMEOUT` plus a single try is what bounds the worst case. That wall
cap is owned by `invoke_with_transient_retry`, which also recovers instant aiohttp blips (litellm
issue #14895) on this `allowed_tries=1` LLM without ever retrying a slow stall, which its elapsed
gate prevents. It routes through `build_llm_with_openrouter_fallback`, so it bills the
Metaculus-donated `OAI_ANTH_OPENROUTER_KEY` first and falls back to the personal
`OPENROUTER_API_KEY` on credential/credit errors. That fallback is
`FallbackOpenRouterLlm` (`metaculus_bot/fallback_openrouter.py`). The donated key
used to be blocked here by a data-policy restriction; that block has been
RESOLVED, verified 2026-06-25 by a live call returning 200 with grounding, so
native search now routes through and bills the donated key. The prompt is the
shared `web_research_prompt` with markdown citations.

### Gemini grounded search: `GEMINI_SEARCH_ENABLED` + `GOOGLE_API_KEY`

Real first-party Google Search grounding via the `google-genai` SDK
(`research/gemini_search.py`), NOT via OpenRouter. This adds a genuinely distinct
search index to the ensemble. Model and request timeout come from
`GEMINI_SEARCH_DEFAULT_MODEL` / `GEMINI_SEARCH_TIMEOUT` (`constants.py`). It
enables both the `google_search` tool and the `url_context` tool, so the model
can read specific URLs named in a question's fine print directly.

Output is now built from the model's own markdown links rather than Google's
grounding metadata. Google drops that metadata on a substantial share of
gemini-3.8-flash responses, including responses that did search. The 2026-09-22
probe showed the model wrote Google's search redirect links in 10/10 calls,
while metadata appeared in only 1/10; 135/136 unique links resolved to a real
page with one no-follow GET (HTTP 302 plus `Location`), and no non-redirect links
were written. Receipt: `scratch/gemini_grounding_2026-09-22/README.md` and
`scratch/gemini_grounding_2026-09-22/selfcite_raw/selfcite_*.json`.

### Self-cited search links

`_SEARCH_LINK_CITATION_CLAUSE` requires the model to copy each tool-returned
`vertexaisearch.cloud.google.com/grounding-api-redirect/...` URL exactly and
inline beside the factual claim. `research/search_redirects.py` resolves each
distinct redirect concurrently through the existing HTTP transport. A cited
redirect is verified only when it resolves to an absolute HTTP(S) target. A
non-redirect link is verified only when the response's `url_context` telemetry
records a successful read of that exact URL. The formatter rewrites verified
links to `<label> [N]`, shares a number when distinct redirect tokens resolve to
the same target, and rewrites every other link to `<label> [unverified link]`.
The `### Sources` block lists each numbered target as its resolved hostname and
URL, never the model's label or a raw Google redirect token.

The floor now asks whether the response cites at least one verified search hit.
Successful `url_context` reads that the response does not cite do not pass the
floor, and text with no verified cited links is suppressed with
`GEMINI_UNGROUNDED_SUPPRESSED`. This blocks the Q38195 shape: confident prose
with fake tier tags and no links. The provider still strips old numeric citation
indices and runs `_check_attributions` / `rewrite_unsupported_attributions`
against the verified source domains. Google's per-sentence `groundingSupports`
alignment is gone; self-attribution is checked only for whether a cited link is
a real search hit, using the same standard as the OpenRouter native-search
provider's markdown citations. The resolved-link counts are recorded by
`GEMINI_SELF_CITATION`.

**Attributions the response's verified source record cannot back.** Gemini also
writes self-invented source-tier tags (`[A: NASA]`, `[B: Reuters]`,
`[C: Time and Date]`), and across the 323 archived sections, 478 of the 681
outlet-named tier attributions (70%) name an outlet absent from that same
response's verified-domain list. q44953 claimed `[A: NASA]` for the eclipse path
over a source list of perlan.is / guidetoiceland.is / timeanddate.com; q45401
named 19 institutions (Bloomberg, FactSet, Goldman Sachs, Kalshi, AP, …) over a
single verified domain. The forecaster prompts instruct weighting by source tier,
so an unbacked tier tag is an authority claim we
manufactured. `_check_attributions` → `rewrite_unsupported_attributions`
(`research/gemini_attribution.py`, shipped 2026-09-01)
replaces each one with `[unverified attribution]` at format time, after link
rewriting and before the `### Sources` block is appended. The verified source
labels are the single evidence base for both the check and the rendered block,
so the two can never disagree about what our record says.
A supported outlet in the same bracket survives verbatim with its own separator:
`[A: FDA, B: Food Safety Magazine]` on a record holding fda.gov renders
`[A: FDA, unverified attribution]`. Several unsupported names in one bracket
collapse to a single marker. The tier grade goes with the outlet it was read off,
because the grade IS the claim. It never touches a word outside a bracket: the
FACT is not what is being disputed (an aggregator domain can carry another
outlet's copy), only the provenance claim, which is why the marker says
*unverified* and not *false*.

Tags that name a CLASS of source rather than an outlet (`[A: official]`,
`[A: peer-reviewed journal]`, `[C: prediction platform]`) are rewritten to the same
marker since 2026-09-24, and counted apart as `generic`. Until then they passed
through untouched, and the prompt's own examples (`"[A: official]"`,
`"[C: aggregator]"`) taught them: in the 2026-09-24 Q14333 smoke (run 36008672128)
10 of Gemini's 12 tags were class descriptions, and the 2026-09-22 probe's
old-prompt calls show 85% of tags in that style. A class tag lets the model claim
tier A with nothing to check it against, so it is read as not following the prompt,
which now asks for the outlet (`prompts._SOURCE_TIER_TAG_INSTRUCTION`). A name is
generic when every identity token is in `_DESCRIPTOR_TOKENS`, a vocabulary built
from the archive's and the probes' descriptive tags; a slash-joined tag keeps its
named half (`[A: official / GRG]` is checked as GRG). Replayed over the archive,
420 generic tags flip to rewritten and no named tag changes verdict
(`scratch/attribution_named_tags_2026-09-24/`). Once tags name the outlet, Gemini
tends to make the tag the link label (`[A: NOAA](url)`, sometimes inside one more
bracket pair): 4 of 5 responses in the 2026-09-24 named-tag probe, about 95 of 101
tags. Rendered as a plain label that loses its brackets (`A: NOAA [1]`) and escapes
the check, so the formatter first rewrites a tier-tag label to the double-bracket
form (`gemini_search._bracket_tier_tag_link_labels`), which renders `[A: NOAA] [1]`.
The probe's tags were then 101 named, 0 generic, and after both fixes about 97% of
the rescorable ones backed by their own source list. Matching of named tags is biased
hard toward KEEPING, because a false
strip discards real provenance while a false keep merely leaves one tag standing.
Any one of six rules credits a name: it concatenates into the domain
(`Golf Channel` / golfchannel.com); all of its identity tokens appear in the
domain (`The Guardian` / guardian.co.uk); the token sets intersect on a token
that is not a class word (`LSE Blogs` / lse.ac.uk, while a shared "research" alone
cannot credit `Research Institute of Foo` against demographic-research.org); a
domain core sits inside the name, the sub-brand shape (`Chosunbiz` / chosun.com);
a single-token name is a subsequence of the label (`WaPo` / washingtonpost.com:
single-token only, since a subsequence test over a multiword name credits almost
anything); or a domain core abbreviates the name (`Times of Central Asia` /
timesca.com). The domain cores are every label left of the public suffix
(`research/public_suffix.registrable_domain`), each read by its first alphanumeric
run, less stop and class words: `nhc` and `noaa` for nhc.noaa.gov, `colostate` for
tropical.colostate.edu, `grg` for grg-supercentenarians.org. Self-cited sources list
full hostnames, and until 2026-09-24 only the first label counted, which read
tropical.colostate.edu as `tropical` and stripped 9 correctly linked `Colorado
State University` tags in the named-tag probe. A response with no renderable
verified source label is skipped rather than blanket-marked (q44802): with no
evidence base, a rewrite would dress our own render failure as the model's
embellishment.
That skip is what makes the count's ABSENCE meaningful: on a schema-v2 record an
absent `unsupported_attributions` means the check had no evidence base or the
record predates the change, while a recorded 0 means it ran and found nothing.
The token is defined where the forecaster reads it: `prompts._SOURCE_PROVENANCE_LADDER`
carries one bullet saying the tag named no outlet, or one the pipeline could not
match against its own retrieval record, that the claim itself may still be correct, and that the
evidence is untiered rather than low-tier. Without that, the ladder tells the model
to weight by tier while a token it has never seen stands where the tier was.
Per-response counts ride
`GEMINI_UNSUPPORTED_ATTRIBUTION: question=... tagged=N unsupported=N groups=N
labels=N generic=N` (INFO; `generic` appended 2026-09-24, emitted only when `unsupported` > 0, harvested as
`gemini_unsupported_attribution`, and deliberately NOT alertable: the habit is the
model's, not a bot defect) and the provider-diagnostics
`unsupported_attributions` count (always, so a zero is a measurement); nothing
keys on either. `labels` rides the line because the same `unsupported` count reads
completely differently against it: q38195 named 21 outlets over ONE grounded
domain, aft.org. `groups` is the render footprint, which sits below `unsupported`
because of the collapse, and since 2026-09-24 also counts groups rewritten only for
a generic tag. There is no `rewritten` or `stripped` field, because rewritten items
are always `unsupported` plus `generic` and the check never removes a bracket
outright. The diagnostics line carries its
denominator, `tier_tags`, next to it, because the marker is gated on
`unsupported`: without the denominator a response that carried no outlet-named
tier tag at all and one whose every tag was backed both archive as
`unsupported_attributions=0`, so a model that quietly stopped tagging would read
as a model that tagged accurately. `tier_tags` counts outlet-named items only and
`generic_tier_tags` the class-description items, so the two together are every
checked tier item (both always recorded; the diagnostics line shows the nonzero
ones).

Measured over all 323 sections: 48 sections rewritten, 203 attributions kept, 478
marked, 0 idempotency failures, and 0 sections where any text outside a bracket
changed. The 70% headline reconciles with the audit's published 87% (276/318) via
86% (590/685), which is what the audit's own matching rule gives through this
harness's extraction (the residual gap is occurrence- versus distinct-name
counting and the three-source union versus artifact-only), and the six keep rules
then move 86% → 70%. All 11 fully-unsupported sections were read in context and
all 11 are true positives; one residual arguable case is test-pinned
(`NewsRadio WFLA` against a grounded iheart.com, 2 of 681: Google reported only
the parent domain, and no general rule recovers a subdomain the SDK never sent);
and the deliberate false-KEEP exposure is enumerated at 20 occurrences across 10
distinct names (2.9%), all short acronyms or shared tokens. Rules, counts, both
review sets and the similarity screen behind the false-strip review:
`scratch/next_season_bundle_2026-09/item4_attribution_check/VALIDATION.md`; the 87%
receipt is `scratch/residual_2026-08-31/gemini_search_audit/cutB_pattern.md` §3.2.

This provider uses the operator's personal `GOOGLE_API_KEY` (a paid-tier Google
AI Studio key). There is no Metaculus-donated key on the google-genai side: the
donated path only exists for OpenRouter-routed Gemini. If grounded search starts
soft-failing across a run, check the AI Studio prepaid-credit balance first
(exhaustion shows up as 429s, not surprise charges).

### Financial data: `FINANCIAL_DATA_ENABLED` (+ `FRED_API_KEY` for live FRED)

For questions about trackable financial/economic metrics
(`research/financial_data.py`). A cheap LLM classifier
(`FINANCIAL_CLASSIFIER_MODEL`, low effort) decides whether the question is
financial and which tickers / FRED series apply, and, critically, resolving
identifiers are *also* extracted deterministically from URLs in the resolution
criteria (`extract_financial_identifiers_from_criteria`). That extraction is the
load-bearing guarantee: even if the classifier misreads the question, the series
the question actually resolves on still fires. The two sets are merged
(extraction is additive), then fetched in parallel:

- **yfinance** for tickers: the latest price, dated ("Latest price: X (as of
  DATE)"), with " — today's bar, in progress" appended (live only) when the
  newest bar is today's, and a stale-latest warning when the newest bar is older
  than its own cadence explains (also logged as the `FINANCIAL_STALE_LATEST`
  telemetry marker); period returns via date-based at-or-before lookups, where a
  label whose match slips past the basis's grace discloses the actual span
  ("1d (actual 2d)") and a label with no observation at or before its target is
  omitted; an annualized volatility over the recent window
  (`FINANCIAL_YFINANCE_RECENT_DAYS` trailing observations); a 52-week high/low
  range (a row-count slice of roughly one year of bars on the series' own daily
  basis); recent closes; and (live only) fundamentals.
- **FRED** for economic series: latest value (dated), previous value, change from
  the previous observation (a row step, whatever the series' cadence), a
  date-based year-over-year change, recent observations.

Both yfinance paths (live and backtest) fetch by explicit calendar start date,
`as_of − FINANCIAL_YFINANCE_LOOKBACK_DAYS` (390 days; `as_of` defaults to now). A
bare `period="Nd"` is deliberately avoided: Yahoo's chart API reads that custom
range as N trading BARS for listed assets but ~N calendar DATES for 24/7 ones,
one integer under two unit systems. Under benchmarking every fetch is
additionally ceilinged to the question's `open_time`: yfinance sets an explicit
`end` at `as_of` and skips the leaky live `.info` call, and FRED routes through a
keyless point-in-time path (`ts_fetch`, ALFRED vintages) so revised macro series
return the vintage known at forecast time, not today's revisions. A
forecaster-invisible HTML-comment routing marker records which identifiers fired,
which came from extraction vs. the classifier, and any unrecognized
(fetched-anyway-but-flagged) IDs.

**Where the code lives.** `financial_data.py` itself keeps the classifier, the
identifier extraction and capping, the job fan-out and the yfinance block. The
2026-09-01 additions below pushed it past the file-size ceiling, so two feature
areas moved to siblings: `research/currency_pegs.py` (the `HARD_PEG_ANCHORS` table
plus `peg_for_ticker` / `peg_disclosure_lines`, stdlib-only so it can never cycle)
and `research/fred_rendering.py` (value/change formatting, `_render_fred_series`,
the first-release table, and BOTH FRED fetchers). The fetchers ride with the
renderers deliberately: `_fetch_fred_first_releases` reads
`Fred.earliest_realtime_start` / `latest_realtime_end` off the class, and its test
proves that by patching `Fred` where the client is constructed, so client
construction and the class-attribute reads have to stay in one patchable
namespace. **Every `Fred` / `fetch_series` patch target is therefore
`metaculus_bot.research.fred_rendering`, not `financial_data`**: fredapi's real
class carries the identical literals, so a patch at the wrong module stays green
while proving nothing. Tests split the same way: `tests/test_currency_pegs.py`,
`tests/test_fred_rendering.py`, with the shared yfinance mock and synthetic series
in `tests/financial_fakes.py`.

The rendered block is no longer just derived stats. Five additions landed
2026-09-01 out of the 2026-08-31 residual round (q44797, q44944):

- **Hard pegs arrive labeled.** `HARD_PEG_ANCHORS` is a static table of eleven
  currencies whose dollar cross is a fixed quote rather than a traded market: the
  Common Monetary Area trio against the rand, Denmark and the two CFA francs
  against the euro, Brunei against the Singapore dollar, and Hong Kong / UAE /
  Saudi / Qatar against the dollar itself (every rate and date verified against
  the issuing authority 2026-09-01). A pegged ticker's block carries a warning
  saying what is fixed and that day-to-day movement is mostly quote noise, then
  appends the liquid anchor cross's whole block below, labeled. Nothing is
  SUBSTITUTED (the question still resolves on the pegged pair, so its own quote
  stays on the page), and a dollar-pegged currency has no substitute cross, so the
  block says so instead of inventing one. Deliberately a static table, not a
  correlation detector: hard pegs are published policy and do not need inferring.
- **A variance-ratio noise flag, with the robust figure as the headline.**
  `variance_ratio` (`research/ts_estimators.py`) is an overlapping Lo-MacKinlay
  ratio on log returns over the provider's full held history (~265 daily bars, NOT
  the 30-row volatility window: the statistic is uninformative at n=30). Below
  `FINANCIAL_VARIANCE_RATIO_FLOOR` the block prints the flag, leads with
  `multi_period_annualized_vol_pct` measured on overlapping
  `FINANCIAL_VARIANCE_RATIO_LAG`-step returns, and labels the 30-row figure
  noise-suspect. **Promoting the long window alone does not reach the case:** both
  volatilities are computed from ONE-day returns and independent quote noise
  inflates both equally, so on q44797's series the long window moves 17.85% to
  15.2% where an honest estimate was 11-14%; the robust figure equals the one-day
  figure times √VR by construction, which is the same statistic's own remedy. Both
  estimators return None on a sample with no measurable return variation, since the
  ratio there is a quotient of floating-point rounding noise (it read 0.369 on an
  exact ramp, a confident noise flag manufactured out of mantissa bits). The
  screen, its `FINANCIAL_VARIANCE_RATIO_MIN_RETURNS` sample floor and the marker's
  one format string live in `research/noise_flag.py` (`screen_for_quote_noise` /
  `noise_flag_line`), shared by both surfaces, because the two copies of the vol
  estimator had already drifted once: the q44882 `sqrt(252)`-on-a-24/7-series
  defect was fixed in one copy weeks before the other. Only the forecaster-facing
  prose is local to each renderer, since the two say different things about what
  else in their section the noise affects.
- **The long-horizon volatility now prints beside the 30-row one on every block**,
  flag or no flag, labeled with its actual row count and step unit.
- **FRED levels render at full precision.** Five sites in `_render_fred_series`
  used `:.4g`, which turned a Case-Shiller print of 331.893 into "331.9" on a
  question whose displayed range was four index points wide, and the Fed balance
  sheet into "6.7e+06". They go through `format_decimal_value` /
  `format_decimal_change` now, which live in `research/number_format.py`
  (stdlib-only, so the FRED block and the inline-chart rung in
  `research/resolution_chart_data.py` can share one rule without dragging pandas or
  fredapi into the latter): fixed-point, up to six decimals, trailing zeros
  stripped, never scientific notation, which also cleans up float-subtraction
  noise, so 0.8729999999999905 renders "0.873". The time-series anchor's own
  formatter, `ts_render._fmt`, was swept the same way at the same time:
  fixed-point up to THREE decimals above 100 (three, not six, because it also
  renders the empirical P10/P50/P90 band, where six decimals on an estimate would
  be fabricated precision), `:.4g` below. Both providers append unconditionally, so
  before that sweep one bundle could state two different values for one
  observation in two adjacent sections.
- **First release vs current vintage.** For a revising FRED series the question
  actually resolves on (`is_resolving_source`, URL-extracted from the resolution
  criteria rather than merely named by the classifier, and not on
  `FRED_NON_REVISING_SERIES`), one extra free ALFRED call renders a table of the
  recent prints' initial releases, current values, revisions, and the observed
  revision direction. It carries the q44944 dossier's mandatory caveat: a
  revision-direction adjustment and a same-source leading indicator measure the
  same underlying data, so apply one, not both.

**A hallucinated FRED series no longer erases the block, and no longer hides as an
`empty`.** The reference tables carry no exchange-rate FRED series at all and only
three currency crosses, so on a question about any other currency the classifier
had nothing to route to and invented an id: q45363 (the Boliviano-USD rate) got
`DEXBOUS`, which does not exist on FRED, with no Yahoo cross named beside it, so
the forecasters got no level and no realized volatility on a currency question.
The verification pass measured that a member sized off the resolving series' own
30-print volatility would have scored +55.35 spot peer alone, better than every
member that ran. Three changes. (1) The classifier prompt now routes every
exchange rate to a Yahoo cross (`USD<ISO>=X` / `<ISO>USD=X`, the spelling matching
how the question quotes the rate) and forbids inventing a FRED id. That is the fix
at the cause, because the currency's ISO code is not recoverable downstream:
FRED's country codes are not ISO currency codes and the `BO` in `DEXBOUS` is a
country, so the classifier is the only step that can name the pair. (2) A series
FRED reports as nonexistent raises `UnknownFredSeries`
(`research/fred_rendering.py`, keyed on FRED's own
`400 "The series does not exist"` body, which fredapi surfaces as a `ValueError`),
so it reaches diagnostics
as `unknown_series` rather than the ambiguous `empty` that was q45363's only trace,
with one `FRED_UNKNOWN_SERIES: series_id=... proposed_by=classifier|resolution_url|gap_fill_driver`
WARN harvested as `fred_unknown_series`, non-alertable, since an invented id is
the classifier's habit rather than a bot crash, and `proposed_by` separates that
from a question whose own resolution criteria link a dead FRED page; `gap_fill_driver` identifies
an unknown id returned by the known-API gap-fill path. (3) When a
question's exchange-rate identifiers carry nothing, the section is ABSENT, no "we
looked and found nothing" line, for the same reason AskNews returns `""` rather
than its old `No articles were found` sentence: any non-empty return flips the
orchestrator's status from `empty` to `ok`, counts the provider in
`providers_succeeded`, and defeats every downstream empty guard at once, so prose
can never stand in for an absent section. What carries the signal instead is
`counts.fx_identifiers_empty`, the number of attempted identifiers whose name has
the shape of an exchange rate on either vendor (`DEX????` on FRED, `???=X` /
`??????=X` on Yahoo: shape predicates `is_fred_fx_series` / `is_yahoo_fx_ticker` /
`is_fx_identifier` in `research/fx_identifiers.py`) and whose `details["sources"]`
token is a loss under the canonical `is_lost_source`, so `empty`,
`unknown_series`, `error` and `skipped(no_fred_api_key)` all count. It is recorded
on every path, so a 0 means the check ran rather than never having run, and it is
independent of whether the section rendered: an FX identifier lost beside a ticker
that rendered fine is the same partial gap and reads the same way, beside the
`sources=<ok>/<total>` the diagnostics line already carries. Separately, a FRED
series skipped for a missing `FRED_API_KEY` now records `skipped(no_fred_api_key)`,
where it used to leave no source token at all and N unfetched series read as a
fully healthy line. The keyless benchmarking fetcher stays silent on all of this:
fredgraph cannot tell a bad id from a vintage predating the series.

Both volatility surfaces emit
`FINANCIAL_NOISE_FLAG: surface=financial_data|ts_anchor symbol=... vr_lag=... vr=...
floor=... short_vol=... long_vol=... robust_vol=...` at INFO, harvested as
`financial_noise_flag` and non-alertable: it describes the vendor's data, not a
bot defect. The sibling flag on the time-series-anchor surface
(`ts_render._realized_vol_lines`) runs the same screen with the same constants,
because the anchor routes to any Yahoo ticker a resolution URL cites and would
otherwise render an equally inflated figure with no disclosure; its prose also
states that the anchor's change BANDS are unaffected, since those are empirical
multi-observation quantiles over which the noise cancels. `long_vol` reads `None`
on the anchor surface, which computes no long-horizon window at all. `surface=` is
what tells that apart from a yfinance series too short to hold one.

### Prediction-market snapshot: `PREDICTION_MARKETS_ENABLED`

A crowd-forecast cross-check. `research/prediction_market.py` is the seam module;
the retrieval pipeline lives in `research/market_retrieval/`. This is **ranked
retrieval**, live since 2026-08-04 (`e75e708`), and it replaced a keyword/fuzzy
design that the 2026-08-03 bake-off measured at 0/17 on near-identical markets.
Generation is deliberately recall-maximal and ALL the judgment sits in one LLM
ranking call, because the bake-off measured that selection, not generation, is the
binding constraint: a perfect ranker over the pool that already exists reaches
14/16 questions while the same pool's deterministic top-4 reaches 5/16.

Four stages per question:

1. **Catalogue prefetch**, concurrent with a **query author**. Kalshi's complete
   open-events catalogue (~10k open events, streamed from `/events`) is paginated
   (`KALSHI_CATALOGUE_WALL_TIMEOUT`,
   `KALSHI_PREFETCH_MAX_PAGES`, `KALSHI_PAGE_SLEEP_S` between pages) and projected
   down as each page streams in, tiered so that only the fields read across an
   event's nested markets are kept past the first one. It is cached for
   `KALSHI_CACHE_TTL_S` only if it **completed**: a pull cut short by a 429, the wall
   or a runaway bound still serves the question that paid for it, but pinning that
   partial list would let one blip on the first question starve the whole run. Both
   catalogues are pulled SINGLE-FLIGHT, one pull per cache key at a time: the TTL
   check cannot see a pull that has started and not finished, so a run's concurrent
   questions used to open one whole pagination each against the same venue and be
   rate-limited for it. Callers arriving while a pull is in flight await that pull and
   share its outcome, a failure included, since re-asking a rate limiter from three
   more questions is a second violation rather than a retry. One lost pull therefore
   bumps `kalshi_catalogue_fetch_failures` once rather than once per waiting question.
   The catalogue is read from `api.elections.kalshi.com`, while Kalshi's current docs name
   `external-api.kalshi.com`. A read-only check on 2026-09-09 found no drift: both hosts
   returned byte-identical payloads for the same market and event endpoints and both
   answered the catalogue endpoint, so the code keeps the elections host. If that host is
   ever retired, the provider's Kalshi `error(...)` source token (an `http_` status or the
   connection error's class name) and the `kalshi_catalogue_fetch_failures` counter will
   show it on the first run.
   PredictIt's whole ~197-market dump is one GET, and all ~197 go into the pool
   UNFILTERED: its old fuzzy pre-filter ranked "Will the Pope visit Cuba" above the
   on-topic market. Neither venue needs a query, which is
   what makes the concurrency free. The query author is one LLM call
   (`MARKET_QUERY_AUTHOR_LLM_CONFIG`) emitting domain vocabulary the question's own
   tokens cannot reach; its output is ADDITIVE to a deterministic query set, so its
   failure costs no recall.
2. **Venue-native search** for Manifold and Polymarket, the two venues whose own
   index is the only way in, at width 60 each. Every deterministic query plus every
   query-author addition is issued unconditionally, in parallel after dedup, with
   per-query failure isolation, and every query is stripped of digit-bearing tokens
   first because Manifold's `term` is a strict conjunction that one date token
   zeroes. The enumerable venues score against the UN-stripped set, where a year is
   real signal against a catalogue of dated market titles. Manifold is asked for
   `contractType=ALL` rather than `BINARY`: multi-outcome markets are ~30% of its
   catalogue and were structurally unreachable before, and their price arrives from
   the stage-3 detail fan-out rather than the search listing, which carries no
   per-answer data at all.

   At PARSE time the author's own synonyms are filtered by a narrower rule than the
   blanket digit strip: a synonym is **dropped whole** (never trimmed to a remnant,
   which would reach the floorless fuzzy channel and score ~100 against every event
   whose rules mention one generic word) only when it carries a DATE-like token: a
   four-digit group in 1900-2099, a bare number in a synonym that names nothing else,
   or a 1-2 digit day beside a month word. Digits belonging to a name survive
   verbatim (`U-3`, `S&P 500`, `10-K`, `50bp`), because series-code vocabulary is
   most of what the author exists to contribute and the conjunction cliff is a
   property of the enumerable venues' call site, which still strips. The rule
   knowingly mistakes `Russell 2000` for a year: a bare in-range four-digit token is
   the measured hazard, and the question's own words reach the venues regardless.
3. **Pool assembly**, three channels unioned, with channel order as the ranking: a
   settlement-source join (Kalshi events whose `settlement_sources` domains match a
   publisher the question's own resolution URLs name, matched by registrable domain
   through the vendored public-suffix list in `research/public_suffix.py`, the recall
   channel a word-overlap scorer structurally cannot see), then the venue-index hits,
   then the enumerable universes ranked by a fuzzy scorer with NO floor (Kalshi to
   width 100). A bounded Manifold detail
   fan-out then fills in the
   `textDescription` rules text the search listing omits, via per-market detail
   GETs, and on a multi-outcome row its leading
   answers, the only price such a row has, since the search reports none. A row
   whose detail GET failed stays title-only rather than costing the snapshot.
4. **Ranking**: one call (`MARKET_RANKER_LLM_CONFIG`, luna at low effort, a
   ~36k-token prompt) over the whole ~380-440
   candidate pool, returning up to 8 rows in ranked order, each stamped with a
   relation tier and a one-phrase `why`. The tier vocabulary is exactly four words,
   in this order of strength: `same_quantity_same_date` >
   `same_quantity_other_cut` > `driver_or_consequence` > `weak`.
   Width is the model's choice in 0..8 (an empty array is a
   VALID answer, not a failure), and nothing downstream re-orders, re-scores or
   caps per venue. Exactly one deterministic pass runs after it, and it changes no row's
   POSITION: `cap_stale_top_tier` (`market_retrieval/ranking.py`) refuses
   `same_quantity_same_date` on a row whose close
   date precedes the QUESTION's own `open_time` by more than
   `MARKET_STALENESS_TIER_CAP_DAYS` (60, local to that module), capping the grade
   one rung to
   `same_quantity_other_cut` and writing a `MarketMatch.tier_cap_note` that states
   the demotion and
   its arithmetic (`demoted from same-date: closed 162d before the question opened`, a
   shape the rendered legend defines; it does not restate the withdrawn grade, which the
   note's presence already implies because only the top tier is ever capped: the old
   wording closed the cell with `(ranker said same_quantity_same_date)` inside a table
   whose preamble tells forecasters to anchor on a same-date market's price). It shares the
   `why` cell's `WHY_CHARS` budget with the ranker's phrase (note first, phrase
   truncated to the remainder) rather than riding on top
   of it, so a capped row costs the section zero characters; while it was exempt,
   three capped rows on a maxed slate crossed the section budget and no fixture set a
   note, so nothing could see it. The note is its own field on purpose:
   `relation_tier` must stay one of the four vocabulary words (`STRONG_TIERS`
   membership picks which preamble renders, and every tier-conditioned residual cut
   tests it by equality) and `relevance_label` must stay the ranker's verbatim
   phrase, or "what the model said about a row our arithmetic overruled" becomes
   unrecoverable from the archive. It keys on question OPEN time, not forecast time,
   so the same market grades the same however late in the window the bot runs. It is
   disclosure rather than a drop (the row keeps its
   rank, its price, its liquidity cells and its rules bullet), because a wrongly
   excluded market is evidence
   the forecaster never sees. It fires on nothing in the 102 archived snapshots, so
   read it as a
   guard on a claim a long-closed market cannot make rather than as a measured fix: only 9
   archived rows are graded `same_quantity_same_date` at all, and q45163's own offender was
   graded one tier below that. Whether to demote that tier too (the dossier's own
   recommendation, `driver_or_consequence`) is an open operator decision in FUTURE.md,
   because that tier changes how the forecaster prompt weights the row. An actual
   demotion logs `MARKET_TIER_CAPPED: question=... rows=... capped=venue@rank` at
   INFO, only on a real cap, harvested as `market_tier_capped`.
   Ordering WITHIN a tier is the ranker prompt's job for the
   same reason: its signals block carries a fourth bullet making `closes` a RECENCY
   tiebreaker within a tier, and that has to live in the prompt because the renderer
   shows the ranker's order verbatim and re-sorting downstream is a measured
   non-option (a previous re-ordering pass lost 43 of 58 wanted rows).
   Everything else falls open to the deterministic retrieval-order top 8, marked
   `[ranking unavailable — showing retrieval order]`: unreadable output; a non-empty
   ranking array from which NO usable row can be read (a renamed index key, every
   index out of range), which raises `RankingShapeRegression`; and equally a
   transient LLM error on the call itself
   (the retry wrapper catches the `openai.APIError` family, which is what every
   litellm transport exception subclasses, and returns an empty completion the
   parser then reports unusable). Both land on the fail-open slate rather than
   costing the whole snapshot.

   That fail-open is kept strictly distinct from a DELIBERATE zero-row ranking over
   a non-empty pool, which renders one sentence saying so, because every genuine
   failure path renders nothing and that is what keeps an outage distinguishable
   from a considered empty answer. `RankingShapeRegression` exists for exactly that
   boundary: the renamed-index case used to arrive as `ok(0)` and render the
   deliberate-empty sentence, a forecaster-facing affirmative claim ("prediction
   markets were retrieved and reviewed… none was judged to bear on it") on a path
   where our own prompt/parser contract had broken.

The render is that order verbatim: implied probability, total volume and open
interest (approximate USD on the real-money venues, play-money mana on Manifold:
the legend says which, since the two are not comparable), a liquidity/participation
`signal` label (thin / decent / deep for real-money venues, thin / decent / high by
bettor count for Manifold, `no-liquidity-data` for PredictIt), close date,
`open`/`RESOLVED` status, and the ranker's `relation` + `why`, followed by each
market's resolution rules.

A close date already in the past when the forecast was made carries a `(Nd ago)` suffix
(`_close_cell`, `rendering.py`), on parent rows and `↳` sub-rows alike,
dated against the snapshot's own `forecast_time` so a later replay of the archived payload
reproduces what the forecaster saw rather than re-aging every row against the replay's clock.
The suffix claims only that the DATE has passed, not that trading stopped, because Manifold's
close dates are soft and its rows can read `status=open` past them. That is exactly how a
five-month-dead market reached rank 0 on q45163 with nothing in the table saying so, and the
disclosure fires on 62 of the 711 archived rendered rows (7 of them still labelled
`status=open`) at a measured cost of 1,017 characters across 102 archived snapshots. The
legend carries the reading, plus one caveat: the column is the venue's TRADING close rather
than its settlement date, and on the Kalshi rows this bot has rendered the two sit a median
+317 days apart, so a forecaster told to verify each market's resolution date was checking
against a different number.

Two row shapes have **no single probability** and render `-` rather than a number
on the parent row: a Kalshi event that is a threshold FAMILY (86.5% of that
catalogue), where one strike's price under the event's own title would answer a
question the row never asked, and a Manifold multi-outcome market. Both keep their
liquidity figures, which is what keeps the row
worth its width, and on a Kalshi family those figures are the SUM over its live
strikes, each converted at its own price, rather than the first strike's alone.

**Since 2026-08-25 (`58175a7`) a multi-outcome family renders WHOLE**, and that is
`rendering.py`'s most load-bearing contract. A family is a distribution over its own
outcome space, its forecast content is the SHAPE of that distribution, and no
subset carries a shape, so truncating from the end was answering the wrong
question. Measured before the fix: 108 of 162 archived families (67%) were
truncated, and truncation correlated WITH relevance (81% of
`same_quantity_other_cut` families versus 50% of `weak` ones); on q45189 all three
forecasters read the one surviving bracket of a ten-bracket margin ladder as an
equality constraint and cut the resolving bucket below their own prior. Now the
leading outcomes get full `↳` sub-rows (`MAX_CHILD_ROWS_PER_SNAPSHOT`, cut 24 → 14
because the bound now sizes only the FULL sub-rows), and every remaining outcome is
NAMED with its own price in one `↳ [remaining N]` ladder row instead of being
dropped. Under character pressure that ladder collapses groups in increasing order
of forecast content: unquoted first (no price, nothing to say), then settled, then
open outcomes by an escalating floor (`LADDER_PRICE_FLOORS`, walked one stage at a
time up to `LADDER_MAX_STAGE`), and every collapsed group states its count and its
summed price, a counted set rather than a silent cut. Settled is deliberately not
collapsed before unquoted: a Manifold threshold ladder settles its crossed rungs to
exactly 1.0 while the market stays open (10 of 17 on that module's committed
fixture), so those titles are the floor the series has already passed, which is why
the group names its LAST member.

Which open outcomes count as "least informative" depends on the family's SHAPE, and
`LADDER_CUMULATIVE_PRICE_SUM` (1.2) is what tells the two apart. A mutually
exclusive partition's prices sum to ~1 by construction (q45189's ten margin
brackets sum to 0.965; a real one lands in roughly 0.95-1.05), while a threshold
ladder's nested prices are SURVIVAL probabilities and sum to roughly the rung count
times the average survival, a median 1.46 across the archived Kalshi families, and
25.4 on the 50-rung gold ladder. On a partition the informative outcomes are the
highest-priced ones, so the collapse is cheapest-first. On a cumulative ladder they
are the ones nearest the CROSSING, so the collapse ranks by distance from
certainty, `min(p, 1-p)`: a 0.99 "above $3251" rung on a gold ladder trading near
$4400 is a near-certainty carrying no forecast content at all, and price-ranking is
close to worst-possible there. Half the archived Kalshi families are cumulative
threshold ladders. When every stage is spent and the section is still over, a
per-family hard bound keeps the highest-priced terms that fit
(`LADDER_HARD_BOUND_STAGE`, the sentinel `99` so the marker's `max_stage=` reads
"fell off the end of the ladder" rather than as one more ordinary stage) and closes
with a counted, summed remainder.

Every summed price names the count it covers. `_open_price_total` sums only OPEN,
PRICED members (a settled rung's price is a realized outcome and an unquoted one
has none), so the per-family hard bound renders `+N more (K priced, X summed)`. A
bare `+160 more (78.50 summed)` read as 78.50 across 160 outcomes when it was 78.50
across 157, and on a settled ladder the same shape hid rungs realized at 1.00.

Two supporting rules make that render trustworthy. The venue parsers no longer sort
children: all three price-bearing venues return the venue's own catalogue order and
the renderer owns presentation, because a parser sorting its children was deciding
what survived a budget it could not see, and price-descending scrambles threshold
ladders. And each venue BLANKS its own manufactured ~0.50 default at parse time, so
a fabricated price reaches neither the ranker nor the render nor a disclosure figure
(192 of 1,839 archived ranked-era child outcomes were in that class). Kalshi blanks
on a book at least `KALSHI_NO_PRICE_SPREAD` wide: an empty book is
`0.0000`/`1.0000`, whose midpoint is a synthetic $0.50 nobody quoted, and the cell
then renders the raw range `0.00-1.00`, which cannot be read as a point
probability. Polymarket blanks on Gamma's `["0.5","0.5"]` placeholder when the leg
carries no volume and no open interest. Manifold blanks (`_priced_or_none`) an
answer sitting at its untouched 0.5 prior with zero volume, in the ranker's
candidate segment as well as in the children, where a defaulted price had been
distorting selection upstream of the render.

Two liquidity-label corrections came out of the same work, both in
`market_retrieval/types.py` so a sub-row is labelled by the SAME rule as its
parent. A Manifold child whose OWN volume is present and zero now reads `thin`
whatever its parent market's bettor pool: Manifold publishes no per-answer bettor
count, so a market with 150 bettors labelled every one of its untouched answers
`high`, including the ones `_priced_or_none` had just refused a price for (62 of 399
archived Manifold children, 15.5%, rendered decent/high on zero own volume). Absent
volume is not evidence of no trading, so the same `is not None` gate applies. And a
Kalshi contract count with no price to convert by (no book AND no last trade) leaves
`total_vol` unknown rather than stating `$0`, which read as a market nobody traded.

The forecaster prompts tell models to weight by both axes (the liquidity label and
the relation tier, whose shared constants live in `prompts.py`), to read a RESOLVED
price as a realized outcome rather than a forecast, to read a family of `↳` rows as
a DISTRIBUTION rather than an equality constraint on a tail, and to resolve a
relation-vs-liquidity conflict in favour of liquidity: a thin market's price is
noisy even when its relation is tight, so widen around it rather than transplant it.

The staleness disclosure is measured against the section's own character budget.
It cost 1,017 characters across 102 archived snapshots (median 0), but the
adversarial worst case is tighter: the maxed budget in `tests/` moved 10,600 →
11,050 for the disclosure and 11,050 → 11,150 for the demotion note's legend
sentence, against a `RESEARCH_SECTION_CHAR_LIMIT / 4` ceiling of 11,249. That
leaves 99 characters of structural headroom (164 from the measured worst case of
11,085), so the next change that widens this section has to cut prose rather than
spend slack.

`MarketSnapshot.forecast_time` is set in `_fetch_market_snapshot_impl` to
`as_of or datetime.now(UTC)`, and the staleness suffix reads it rather than the
renderer taking the clock: the render has to be reproducible from an archived
snapshot alone, and a replay months later would otherwise stamp staleness on rows
the forecaster never saw it on. Archived snapshots predating the field carry None
and render exactly as before, and a backtest with `as_of` supplied can fire no
disclosure at all, because pool assembly already dropped everything closing at or
before it.

**Per-question telemetry**, all three harvested into the telemetry archive
(`market_ranking` / `market_child_render` / `market_ranking_degraded` in
`scripts/telemetry/markers.py`), so pool indices, prompt sizes, child-render counts
and degradation causes survive the 90-day GHA log expiry:

- `MARKET_RANKING: question=... pool=N outcome=ranked|failopen|empty rows=K
  prompt_chars=M rendered=...`, where `rendered` is the per-row
  `venue:pool_index@rank` list.
- `MARKET_CHILD_RENDER: question=... families=... full_rows=... ladder_rows=...
  outcomes=... named=... collapsed=... withheld=... max_stage=...
  ladder_chars=...`. `named + collapsed == outcomes` is the completeness invariant,
  so a line where they disagree is a render bug rather than a tuning signal;
  `withheld=` turns the blanking rules' prod incidence into a query rather than a
  guess (the Kalshi no-price spread threshold is calibrated on eleven fixture
  strikes); and `max_stage` / `ladder_chars` say whether `LADDER_SECTION_MAX_CHARS`
  binds on real slates.
- `MARKET_RANKING_DEGRADED: question=... pool=N reason=shape_regression|unreadable
  detail=...`, which says WHICH failure produced a fail-open:
  `outcome=failopen` alone cannot, and `reason=shape_regression` is the one that
  means OUR contract broke.

This provider is **hard-disabled under benchmarking** (`is_benchmarking=True`
returns `""`), regardless of the env flag, and that guard is the ONLY leakage
defence: markets retain their last-trade price after resolution, so a market that
closes between an `as_of` instant and now leaks either way. The provider path
therefore passes `as_of=None`. The filter itself survives for explicit callers
(backtests, replay tooling) and runs INSIDE pool assembly, ahead of the per-venue
width slice, so a leaked market never reaches the ranker and an ineligible row
frees its slot for an eligible one instead of consuming it. As a post-hoc filter
over the already-truncated pool it deleted rows the width had spent its slots on
and zeroed the per-venue counts provider health reads. Prod ran with a derived
`as_of` until 2026-08-04, and it cost real recall: it dropped every market closing
before the question resolved (the "same quantity, adjacent month" class that
carries most of the evidential value),
and prod telemetry recorded 20 of 47 archived runs where Polymarket fetched
candidates and rendered nothing because of it.

The benchmarking guard is also why this provider's forecasting value can't be
measured by the standard backtest gate: it was validated with live
`test_bot.yaml` runs and opt-in live integration tests.

The research section header is `## Prediction Market Snapshot`, and the prompts
import it from `prompts.py` as `MARKET_SNAPSHOT_SECTION_HEADER` to decide whether to
render their market-reading rules at all.

### Resolution-source fetcher: `RESOLUTION_SOURCE_ENABLED`

Fetches the exact URL(s) a question cites as its grading source
(`research/resolution_source.py`), so forecasters read the ground truth the
question resolves against. The page fetch is Tier-1: plain HTTP with browser-like
headers; long successful HTML then uses the paid page-digest extractor described below.
One narrow Tier-2 hop sits beside it (the
embedded Datawrapper dataset, below). When the direct fetch cannot read a page an
escalation ladder runs (`_escalate_unresolved`), each rung self-bounded against the same
provider wall and each returning a result that went through the same classification path,
so a rescued page is indistinguishable downstream from a directly-fetched one; the `route`
on every result says which rung produced it. The four rungs added on 2026-09-03, the
TLS-impersonating retry added on 2026-09-04, and the one paid rung among them, are described
under "The escalation ladder" below.

`research/resolution_presentation.py` owns the pure rendering step: provenance
leads, unreadable-embed disclosures, page and dataset text budgets, route caveats,
and the final section. Fetching, rung counts, and registered telemetry emission
remain in `resolution_source`. Presentation tests patch the constants
on `resolution_presentation`, where the renderer reads them.

`research/resolution_datawrapper.py` owns Datawrapper response classification,
freshness checks, dataset text, chart selection, and result ordering. Its diagnostic
warnings retain their text and use that module's logger. Requests, host limits,
timeouts, cancellation, and registered result markers remain in `resolution_source`.

It deterministically extracts URLs from resolution criteria + fine print (markdown
links and bare URLs, order-preserving dedup, Metaculus markdown-escapes undone;
`research/resolution_url_scan.py`). A URL body is a run of atoms: a markdown escape, a
balanced `(...)` pair, a balanced `[...]` pair, or any plain character, so a Wikipedia-style
`_(rocket)` path and a Rails-style API query such as
`documents.json?conditions[agencies][]=nuclear-regulatory-commission` both survive whole while a
lone closer from the surrounding prose still ends the match. The bracket atom came from the
Mantic readiness review (2026-09-08): question writers there resolve a question to "the count
this API query returns" and cite the query, the old class cut it at the first bracket, and the
truncated Federal Register query answers HTTP 200 with the UNFILTERED count (10000 against a
correct 125), which this provider would then have served as the grading evidence. Six of 556
public Mantic posts carried such a URL. A backtick ends a URL for the same reason: ten cited
URLs across nine of the 556 posts were fenced in backticks (several of them API endpoints), and
the fenced form 404s where the clean form answers 200. It then
skip-filters URLs that add nothing or belong to another provider (self-references
to either question platform's own site, metaculus.com or `competitions.mantic.com`,
refused on every run mode because `QUESTION_PLATFORM_HOSTS` names both hosts unconditionally:
a question page carries no new information and, on Mantic, shows the other bots' forecasts
and comments, which must not leak into research;
FRED series owned by financial-data; Yahoo `/quote/` pages owned by
yfinance), and caps at `RESOLUTION_SOURCE_MAX_URLS` *after* the skip filter so
a run of leading self-refs doesn't starve the real sources. Fetches run in
parallel with one-request-per-host politeness (a `Semaphore(1)` per netloc, keyed
per redirect hop, and shared process-wide since 2026-09-03 in the loop-scoped
`http_fetch.host_semaphores` map, handed out by `http_fetch.semaphore_for_host`:
with a map per provider
call, six questions citing one host each held their own semaphore and hit it six times
at once). Content is extracted with trafilatura (HTML), or read raw (JSON / text /
CSV). A PDF is read locally with pypdf and rendered as a passage digest (below);
anything else is left unread as `unsupported_type`.

**Which extraction publishes** is a policy, not a flag (`_extract_page_text`,
calibrated 2026-09-03 on 118 re-fetched bodies with five extractor variants run on identical
bytes; receipt `scratch/fetch_ladder_2026-09-03/chrome_calibration.md`). Trafilatura's default
recall is the primary extraction. Its text is scored by line shape: `content_share` is the
share of extracted characters that sit in table rows (lines starting with `|`) or in lines of
at least `RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS` (60), and a text at or above
`RESOLUTION_SOURCE_CONTENT_SHARE_MIN` (0.38) that also clears the 400-char chrome floor
publishes. A text that clears the floor on short lines alone is chrome, and the same input is
re-extracted with `favor_precision=True`, which is the one trafilatura setting that prunes
navigation out of the backup tree its readability fallback swaps in; that text publishes only
if it clears the floor and the same metric. Otherwise the page is withheld as
`no_resolving_content` with reason `thin_page`, so the rendered rung still fires. That second
pass is unbudgeted CPU on a body already in hand, so it is skipped when the default pass leaves
less than `RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S` (5 s) of the provider wall: a 5 MiB
navigation-tree DOM measured 42 s in one pass, and an overrun there discards every page the
question already fetched. A skipped pass withholds the page exactly
as a failed one does, so the rendered rung still gets its turn and the counts read the same. The
two
cases that fixed the policy: congress.gov, where the default extraction replaces the 2,411-char
bill-status card ("Latest Action", "Passed House") with 54,393 chars of a member-name
dropdown and precision restores the card; and uk.finance.yahoo.com, whose 1,191-char direct
body is a menu plus one quote line while its render is the full 23,991-char price table that
a floor-only check never reached because the menu counted as success. On the labelled
corpus the policy publishes every content text (46 of 46), publishes 2 chrome texts against
11 for default-only and 4 for precision-only, and withholds no content. The margin is about
0.05 on each side: navigation-tree chrome tops out at 0.329 (kasa.go.kr's homepage, a menu
with a news ticker) and the thinnest labelled content is 0.431 (a wastewaterscan dashboard
of 79-char readings). What it gives up, deliberately: prose-shaped boilerplate (AP's
cookie-consent wall, clinicaltrials.gov's glossary) is sentences and passes any line-shape
metric, and kasa's ticker line is withheld with its menu. Precision alone shipped until
2026-09-03 and withheld readable pages (kasa.go.kr pruned to 78 chars, two tracxn funding
tables, manifold's market body); default alone shipped for one day on a character-count
measurement, which under a head-preserving 6,000-char cap is the wrong metric, and the
earlier claim that its biggest gainers had been read by hand and were all content was
wrong. The policy's decisions ride `details["counts"]` as `chrome_metric_withholds`,
`chrome_metric_withholds_rescued` and `precision_fallback_rescues`; no status or reason token
changed.

Two free rungs sit under the HTML path, both reached only when the page carried nothing
readable. A **meta-refresh hop** follows the redirect no HTTP status announces:
cdc.gov's surveillance URLs answer 200 with a ~300-byte stub whose only content is
`<meta http-equiv="refresh" content="0; url=...">`, which the manual redirect loop
cannot see (no 3xx, no `Location`), so the stub used to be classified a JS wall and the
resolving page never fetched. The target is returned as the next hop, so it re-enters
the same classification path and consumes one of the same `MAX_REDIRECTS` slots, and it
passes exactly the same three checks a `Location` header does, via the shared
`_vetted_hop_target` helper. An **ARIA-table rewrite** (`rewrite_aria_tables`,
`research/http_fetch.py`) runs
before every extraction: cdc.gov builds its outbreak stat blocks out of
`<div role="table">` / `role="row"` / `role="cell"`, which is valid accessible markup
and invisible to trafilatura's table handling: the cyclosporiasis block rendered as a
bare "17,180 / 2" with no labels and no hospitalization count at all, because 922 sat in
an unwrapped cell. Rewritten to real table tags, the same page extracts
`| Hospitalizations | 922 |`. A page with no ARIA role is handed to trafilatura as the
original bytes, so its extraction is unchanged.

A **cited PDF is now read** rather than dropped (`_resolution_pdf_outcome` is the
branch): `research/document_text.py` extracts the
text with pypdf and selects the passages most relevant to the question's title plus its
resolution criteria (BM25, deterministic, no model call), rendering a digest that states
how many pages were read and labels each passage with its page as `[p.N]`. The ranking
query is title plus resolution criteria and deliberately not fine print, which is mostly
procedural boilerplate about ambiguity and annulment and would dilute the term set.
Measured 2026-09-03,
that path pulled 833,450 chars out of a 6.7 MB 220-page document in 5.3 s with the wanted
passage in it, while the paid alternative returned nothing for the same file. A body the
server did not declare as a PDF is still sniffed by its `%PDF-` magic, since several
government hosts serve documents as `application/octet-stream`: a declared document gets
the larger `DOCUMENT_TEXT_PDF_MAX_BYTES` cap (the receipt file is over the 5 MiB response
cap), an undeclared one keeps the smaller one. Bytes we read and could not turn into text
get their own status, `unreadable_document`, with `status_reason` naming which of
`no_text_layer` / `encrypted` / `malformed` applies; only the first could ever be rescued
by a paid document read, which is why it is not folded into `unsupported_type`. A document
we read in full whose passage selection matches no query term is WITHHELD as
`no_resolving_content` / `no_matching_passage` rather than published: its block is the
header, the outline and one sentence saying nothing matched, and published as `success`
that was prose standing in for an absent section, indistinguishable in the run log from a
document that handed the forecasters the resolving paragraph. It is also the one
`no_resolving_content` the paid url_context rung is not allowed to re-read, since we
already hold the document's text.

**Tier 2: the embedded Datawrapper dataset.** The first (and so far only) Tier-2 hop
shipped 2026-08-25 (`5f27c46`, receipt qids 44858/44841) and is narrow by design.
When a fetched page's RAW HTML embeds a Datawrapper chart, the chart also serves its
live "Get the data" CSV, and poll trackers lock their resolving daily series inside
exactly those iframes, which trafilatura drops at every setting. The hop uses ONLY
the version-free `static.dwcdn.net/data/<chart_id>.csv` route, because the page HTML
pins a stale chart version whose `datawrapper.dwcdn.net/<id>/<version>/dataset.csv`
form keeps serving 5-14-month-old snapshots as HTTP 200 (the naive fix the 2026-08-24
verifications refuted). A `Last-Modified` freshness guard then withholds anything
outside the window under the `stale_data` status rather than serving stale data as live
(the Wayback rung below uses the same status for an over-age archived capture of a cited
page; the two are told apart by `chart_id`): older than
`RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS`, undatable, or
implausibly far in the FUTURE: a future date past a six-hour clock-skew tolerance
means a broken clock, not maximal freshness. The hop is bounded by
`RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS`,
`RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS`,
`RESOLUTION_SOURCE_DATAWRAPPER_HOP_WALL_MARGIN_S` and
`RESOLUTION_SOURCE_DATAWRAPPER_MIN_HOP_BUDGET_S`. A served dataset leads with a
`Dataset published <ts>` liveness stamp.

**The escalation ladder.** Four more rungs shipped on 2026-09-03 and a fifth on 2026-09-04,
tried cheapest first behind the free hops above. Each declines by returning nothing, in which
case the direct route's own status stands. The ladder order is the `FetchRoute` Literal's:
the impersonated retry first, then the derived feed and the browser, then the archive, then
the paid reader.

`route=impersonate` (`research/impersonated_fetch.py`, the transport; `_impersonate_rung` in
`research/resolution_source.py`) re-dials a page that answered our aiohttp client 403, once,
through libcurl presenting a real Chrome TLS ClientHello and HTTP/2 settings fingerprint
(`curl_cffi`, profile pinned by `IMPERSONATE_BROWSER_TARGET`), and the body re-enters the same
classification path a direct 200 gets: HTML through `_classify_html_body`, JSON and text through
the raw-body rule, a PDF through the local document read (which is why an impersonated PDF
reads `route=pdf_local`, the accounting a meta-refresh hop onto a PDF already produces). It is
the one rung that leaves aiohttp without leaving our address. Measured 2026-09-04 from a GitHub
Actions runner with `scripts/probes/fetch_diagnostic.py`: four of the four Akamai-fronted
federal URLs that refused the bot's own client (bls.gov twice, one a PDF, cdc.gov,
fsis.usda.gov) answered the impersonated GET 200, so that refusal is a fingerprint verdict and
recoverable client-side; the four hosts that refused both (Cloudflare, CloudFront and DataDome
fronts) are the egress-IP population and stay the Wayback and paid rungs' business. The trigger
is `blocked` with HTTP 403 and nothing else: 429 is a throttle a fingerprint change could make
worse, 406 is content negotiation the impersonation profile's own `Accept` headers would only
guess at, 401 is authentication, and the question-platform self-reference hop's `blocked` carries the
redirect's 301 or 302, which is what keeps a URL this module refused from being handed to a
second transport. It sits between the direct fetch and the archive so a live page beats a stale
capture, and before the paid reader so a rescue saves the read on that URL entirely; the
position among the earlier rungs is a reading choice, because the browser rungs never fire on
`blocked`. Free (no key, no model, no spend), floored at
`RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S` like the other one-GET rungs, never fast-path
gated, memoized per HOST for the run once a host answers the impersonated client with a block
status (`IMPERSONATE_BLOCK_STATUSES`: the three `blocked` rows of the status table, 403, 406 and
429, plus 401 and 503, which the fetch line still records as `error` because the status table is
a telemetry contract and the memo is a policy; the memo is process-global and shared with
gap-fill v2, which dials the same transport from both of its free ladders, the `fetch` tool's
and `read_document`'s local acquisition; the transport writes the host memo for the host that
ANSWERED, since the impersonated client follows redirects itself and the block can come from a
later hop, and a separate per-URL memo for the exact URL dialed so that chain is not walked
again, while the dialed host itself stays dialable because a host that merely redirected never
refused us), and behind a kill
switch that is ON by default in code, `RESOLUTION_SOURCE_IMPERSONATE_ENABLED`, unlike the paid
rung's default-off. The trigger set, the block-shaped set the memo keys on and the kill switch
all live on the transport (`IMPERSONATE_TRIGGER_STATUSES`, `IMPERSONATE_BLOCK_STATUSES`,
`impersonation_enabled`, `note_refusal_if_block_shaped`), read by both fetchers at call time so
neither can drift from the other. A rescue's fetch line reads `status=ok http=200
route=impersonate` because the bytes came with a 200; the refusal
lives on the escalation line's `from_status=blocked`, as a Wayback rescue already reports the
snapshot's own status. An impersonated 200 that still classifies as unreadable stamps its verdict
on the escalation line (`rung=impersonate outcome=js_wall`) and leaves `blocked` standing, so
the paid rung stays reachable for that URL. libcurl never touches aiohttp's connect-time
resolver, so the transport carries the SSRF invariants itself; see the SSRF paragraph below.

`route=derived_api` (`research/derived_api.py`) serves the JSON feed a page loads its own
figures from. A JavaScript dashboard's numbers arrive over XHR after the DOM is ready and sit
in the served HTML at no wait condition, so the endpoint is found by the browser rung recording
the page's own requests, then remembered per HOST for the rest of the run: a second cited URL on
that host costs one GET (floor `RESOLUTION_SOURCE_DERIVED_API_MIN_BUDGET_S`, the one-request
floor) instead of a second browser launch. Because a host's feed is usually parameterised, the
lead on each served block names the endpoint and says whether it was discovered on that page or
on another page of the same host, so a forecaster can check that the feed covers the quantity
asked about. A response the render recorded is harvestable only when it came from the page's own
publisher by registrable domain (`research/public_suffix.py`, a leaf that reads the vendored
`research/data/public_suffix_list.dat` and is shared with the market-retrieval settlement join)
or from an explicitly allow-listed CDN, so a stranger's JSON on a shared suffix is never read as
the cited page's content.

`route=rendered` re-reads the page out of a headless Chromium render
(`research/rendered_fetch.py`, the transport shared with gap-fill v2's fetch ladder and its
process-global two-launch cap). It triggers on `js_wall` and on the `thin_page` shape of
`no_resolving_content`, both pages that answered 200 with nothing readable, and deliberately not
on `embed_shell`, since `page.content()` returns the main frame's HTML and an Infogram or
Flourish iframe comes back as a bare tag. Its floor,
`RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S` (12 s), is far above the one-request rungs' because a
launch plus a DOM-ready navigation costs several seconds even on a page that renders cleanly, and
the launch slot is contended process-wide, so a question with no budget left would take a slot a
sibling question could still land a page with. That floor is the PRE-gate check, read before the
render queues on the per-host gate and the launch cap. The browser is handed `direct.url`, the URL
the direct fetch LANDED on once its own redirect hops were followed and re-guarded, rather than the
cited URL, so the DNS pin the transport sets covers the host that actually serves the content and a
page whose canonical form is one ordinary hop away (`example.com` to `www.example.com`) is not
refused for taking it; when the two differ, the landing is re-vetted with the same self-reference
and public-URL checks every derived hop owes. Both render memos key on the URL rendered, the HTML
classifier keys on the URL the browser's main frame LANDED on (`RenderedPage.document_url`, the
landing when a navigation committed, else the rendered URL), and the rung's attempt stays keyed on
the cited URL, which is what the escalation line names. One accounting consequence, since
2026-09-04: on a `route=rendered` rescue the `FetchResult.url`, which is the `url=` of the
`RESOLUTION_SOURCE_FETCH` line and the published `### <url>` heading, is that landing URL, while the
`RESOLUTION_SOURCE_ESCALATION` line's `url=` is the cited URL, so a per-URL join between the two
lines must key on the escalation line; before that boundary the two agreed. A rescued result keeps the DIRECT
fetch's `http_status`, which is the `http=` on the fetch line: Chromium reports no status on a salvaged DOM, and
the fact worth archiving is that the page answered 200 and carried nothing readable. The transport's own post-gate need is
higher: `RENDER_MIN_GOTO_MS` (5 s of navigation) plus `RENDER_POST_GOTO_TAIL_MS` (the 2 s settle
and the 5 s DOM-read bound, reserved so a goto that runs its budget out can still be salvaged)
plus `RENDER_EXIT_RESERVE_MS` (3 s, described below), 15 s in all, so a render admitted with 12 to
15 s of wall left declines at the gates with a `wall_budget` skip rather than launching. That band
is deliberate: raising the floor to 15 s would make the pre-gate check truthful at the cost of the
band's reach, and the call is the operator's (FUTURE.md item 5). The rendered DOM re-enters
`_classify_html_body`, so a rescued page gets the same chart read, ARIA rewrite, floors and
disclosure leads as a directly-fetched one, and can still be withheld. A transport that declines
(Playwright missing or broken, a host that will not pin to a public IP, or a browser error) is
recorded as a SKIP with the reason `renderer_unavailable` rather than as a fired rung, because
nothing was rendered and so nothing about the page changed. Five nearby cases are kept OUT of
that reason, so that neither a memo hit, a queue timeout nor a fact about the page can read as
the Chromium install having failed. A URL an earlier question already rendered to nothing this
run is `rendered_no_text` (the memo doing its job). A render that ran out of budget queued behind
the launch gates, which the transport signals with `RenderBudgetExpired`, is `wall_budget` (the
same reason the pre-gate floor check records). A browser that was answered a non-200 where the
direct GET got 200 is `render_non_200`: the DOM belongs to the host's interstitial or error page,
so it is not read as content, and the URL is not memoised because a 429 is retryable. A rendered
DOM over `RENDERED_DOM_MAX_CHARS`, which the transport signals with `RenderDomOverCeiling`, is
`render_dom_too_large`, declined before anything copies it. And a main frame that landed on a host
other than the pinned one, which the transport signals with `RenderOffHost`, is `render_off_host`:
the transport checks the landing before and after `page.content()`, so the DOM is refused unread
or discarded unpublished, nothing from that render reaches the classifier and the direct result
stands. The check fails shut, so Chromium's own error document after a failed navigation
(`chrome-error://chromewebdata/`, seen live after `net::ERR_UNSAFE_PORT`) is refused under the same
token, which makes the count an upper bound on hostile landings. It emits one WARNING under the marker
`RENDERED_FETCH_OFF_HOST` (below) and, like every skip, no `RESOLUTION_SOURCE_ESCALATION` line.
Two more bounds hold the rung inside the wall, and which one fired is what the skip reason says.
Inside the transport, `page.content()` is capped at `RENDER_DOM_READ_TIMEOUT_MS`: on a settled DOM
it is a sub-second round trip, and it runs long only when the page keeps navigating after the settle
(measured 2026-09-03 on ogimet.com, where the goto timed out at 33 s as designed and the
unbounded read then blocked for a further 40 s, so the render ran 76 s against the 45 s wall and
every page the question had already fetched was discarded). That cap raises the transport's own
`RenderTimeout` and is the ONLY thing recorded as `render_timeout`, because a page that keeps
navigating is a fact about the page. Around the transport, the rung holds the whole `render_page`
call to the remaining budget with `asyncio.wait_for`, so no Playwright call can overrun the wall
from inside it; that outer cut records `wall_budget`, the same reason the pre-gate floor check and
a render queued behind the launch gates record, because what fired it was the question's remaining
clock rather than the page: the render was still queued behind the per-host gate or the launch
cap, or the transport overran its exit reserve. A render cut off either way says nothing about
whether Chromium works, so neither trips the once-per-run "rung unavailable" warning, and the
direct result is what stands.

The outer `wait_for` bounds when the rung stops WAITING, not when the transport stops RUNNING:
`asyncio.wait_for` cancels the render and then awaits its unwinding teardown, so the exit has to
fit inside the rung's bound as well. `_rendered_rung` therefore hands the transport a deadline
`RENDER_EXIT_RESERVE_MS` before its own `wait_for` fires: 3 s, the shared 2 s teardown budget plus
1 s for the launch (which runs after the transport recomputes its navigation budget, so it is
not in that budget) and the driver stop. The transport sizes the goto off that deadline less the
post-goto tail and clamps the harvest drain to it, and the reserve never lengthens a wait: it can
only shorten the goto or make the render decline earlier at `RENDER_MIN_GOTO_MS`. The three
teardown steps (unroute, context close, browser close) share ONE `RENDER_TEARDOWN_TIMEOUT_MS`
budget, started lazily by the first step that asks rather than at launch, so whatever the browser
does the exit costs at most 2 s before the driver stop, and a step that outlives what is left of
the budget is abandoned to that stop. The driver stop at the end of the `async_playwright()`
block is the one step left unbounded, deliberately: it is what actually kills the Chromium
process, and abandoning it on a wedged browser would leak 100 to 300 MB past `RENDER_LAUNCH_CAP`.
It is the exit's one named residual; a healthy stop is milliseconds and fits in the reserve's
spare second, and a wedged one is unmeasured live. With the reserve in place the transport's
DOM-read bound lands before the rung's outer cut even in the salvage shape (the goto ran its
budget out), so the timed-out memo, written at the `RenderTimeout` raise site, lands too, and a
second question citing the same hostile page records `render_timeout` again instead of paying
for another render.

`route=wayback` (`research/wayback.py`) serves an archived capture. The archive earns a rung
because it is the one free route whose EGRESS IS NOT OURS: measured 2026-09-03, the same client
with the same headers gets 403 from a GitHub Actions runner and 200 from a residential address
on bls.gov, cdc.gov and fsis.usda.gov alike. It triggers on `blocked` / `error` / `not_found`
and never on `js_wall`, because the archive stores the unrendered shell. Three bounds: the
`RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S` floor, at most
`RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS` snapshots per QUESTION (every snapshot shares the one
`web.archive.org` host gate, so N cited URLs would otherwise queue into N sequential archive
fetches inside the provider wall), and the age bound
`RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS`, which matches the Datawrapper freshness bound and is
the same judgment rather than a measurement. A capture is admissible as primary grading evidence
only with its age stated, which is what the lead renders. Three outcomes, in this order, and the
order is the design. The wrapped inner URL is unwrapped and re-checked first, because
`is_metaculus_self_ref` keys on hostname and a `web.archive.org/web/.../metaculus.com/...` URL
sails past every self-reference filter in the pipeline, and a failed SSRF or self-ref re-check
refuses the rung outright. That check refuses both question platforms' own sites, metaculus.com and
`competitions.mantic.com`, on every run mode (a Metaculus tournament run refuses a cited Mantic page
too), while the rest of mantic.com (blog.mantic.com, say) stays fetchable as an outside source; the
function keeps its name and the `metaculus_self_ref` status token is unchanged, both being data
contracts. Then a capture the archive never served DECLINES, leaving the direct
status standing, because "no archived copy exists" is a different fact from a stale one and the
direct status says more about the source. Only a capture we did read and cannot date, or can date
and it is too old, is withheld as `stale_data`. A withhold does not end the ladder: the paid rung
below is still asked about the DIRECT outcome (a stale archive is still a page we could not read
fresh), and the withhold is what stands when that rung is off or declines. This rung is NOT
flag-gated, so from its merge a cited page's `status` can be a rung's verdict where it used to
be the direct outcome: `stale_data` where the direct fetch said `blocked` / `error` /
`not_found`, and (flag on) `ungrounded`, or `no_resolving_content` with reason `not_addressed`,
where it said `blocked` / `js_wall` / `error` / `no_resolving_content`. An era-bucketed
`blocked` or `error` rate read off `status` alone will
show a drop at that merge that is a bookkeeping change, not hosts refusing us less; take the
direct outcome from `from_status` on the `RESOLUTION_SOURCE_ESCALATION` line, or partition
`status` by `route`, where `direct` rows are unchanged.

`route=url_context` is the LAST rung and the only paid one: Gemini reads the page for us
(`research/url_context_reader.py`, the reader shared with gap-fill v2's `read_document`),
reaching hosts our own client cannot because Gemini dials from Google's address. It defaults off
in code behind `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` and is set to `true` in every bot workflow
yaml since 2026-09-04, so it is live in production; `docs/operations.md` covers what it costs and
on whose key. Every gate is checked in increasing cost order before a cent is spent: the trigger statuses
(`blocked`, `js_wall`, `error`, `no_resolving_content`, tested against the DIRECT outcome, so a
withheld Wayback capture on the way down does not close the rung), the flag, the question's
time-budget fast path (recorded as a `fast_path` skip, and placed after the flag rather than before
it so a flag-off run never reports spend avoided on a rung that could not have fired), the API key,
the `RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S` floor, then the per-host `Google-Extended` robots
pre-check (`research/robots_policy.py`, whose cache and `ROBOTS_FETCH_TIMEOUT_S` bound are shared
with v2's reader; worth a request of its own because a host disallowing that token refuses
Gemini's fetch server-side), and the budget floor a SECOND time. The pre-check is the one gate
that costs a request, and the paid read runs in a thread that `asyncio.wait_for` cannot cancel, so
the client-side ceiling sized off the remaining budget is the only bound that can stop it; re-reading
the budget after the pre-check is what keeps that ceiling honest, and a pre-check that ate the room
records a `wall_budget` skip rather than a paid call nothing reads. It has its
own retry count, `RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS`, deliberately lower than the v2
reader's, because a retry inside a wall shared with every other cited URL spends the budget the
pages already fetched need in order to render. A per-QUESTION paid-read cap,
`RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS`, bounds how many reads a single question can pay
for across its cited URLs (the analogue of the Wayback per-question cap and a distinct quantity
from the SDK retry count), claimed last, only for a read that has cleared every cheaper gate, and
a read the cap declines records a `url_context_cap` skip. Two outcomes inside the trigger statuses
are excluded by REASON (`_URL_CONTEXT_EXCLUDED_REASONS`): `no_matching_passage`, a document we
already hold in full, and `metaculus_self_ref`, a redirect we refused because it landed on the
question platform's own site. The rung is handed the CITED url, so without the second exclusion
Gemini followed the same redirect and read the page we had just refused, a paid read that returns
nothing new (and on Mantic the other bots' forecasts as grading evidence); it is the `ssrf_blocked`
bypass closed by reason, because the self-reference's status is `blocked` by contract. Two facts about the trigger population
belong with that cap. The `no_resolving_content` trigger now includes the extractor policy's
chrome-metric withholds (reason `thin_page`), so a page whose default extraction was navigation and
whose precision fallback failed the same line-shape metric is a paid-read candidate rather than a
page we gave up on. And because the cap is two reads with no ordering over the candidates, claimed
by whichever of the question's concurrently fetched URLs reaches the gate first, those candidates
can take both slots ahead of a `blocked` page, whose Google-egress advantage is the rung's whole
reason to exist. `docs/operations.md` prices both against the calibration census. Zero successful
retrievals DISCARDS the text under the new terminal status `ungrounded`, the same floor
`gemini_search` and v2's `read_document` apply: Gemini answers fluently out of parametric memory
when every
retrieval failed, and a fluent unsourced answer under the primary-grading-evidence caption is
the Q38195 failure with a forecaster-facing blast radius. A read whose answer opens with the
prompt's `NOT_ADDRESSED` sentinel (`research/url_context_reader.py`, the model's designed reply
when the retrieved page does not discuss the ask) is withheld as `no_resolving_content` /
`not_addressed` rather than rendered: the page WAS retrieved, so it is not `ungrounded`, but under
the url_context lead the non-answer was prose standing in for an absent section, the shape the
PDF digest closes with `no_matching_passage`. The read stays on the record as the rung's own
verdict, since it was paid for. A read that lands leads with a
mandatory disclosure saying why the route was taken and that the text is a model's reading
rather than a copy of the page. Its spend is visible as a third `GEMINI_USAGE` role,
`resolution_source`.

Every non-direct route present in the sections that will RENDER also contributes one
forecaster-facing caveat sentence, from `ROUTE_CAVEATS` in
`research/resolution_fetch_result.py`: where the bytes came from, and what the reader must not
conclude from having them. The mapping is keyed by route and iterated, so it is both the
vocabulary check and the render order, cheapest and most transparent first and model-mediated
last. `direct` is deliberately ABSENT from it rather than mapped to an empty string, which is
what keeps an all-direct question's section byte-identical to what it rendered before the ladder
existed (the overwhelming majority of questions, pinned by a test). Every non-direct token now
has a live producer: the `impersonate` sentence was reserved on 2026-09-03 for a rung that did
not yet exist and has rendered since 2026-09-04, and the completeness test that guarded the
reserved token still asserts every non-direct route carries a sentence, so a future rung cannot
render rescued content with no disclosure at all. An impersonated PDF renders the `pdf_local`
sentence rather than the `impersonate` one, because `route` is the last rung that fired and the
local read is what produced the text; the impersonation fact does not change how a forecaster
should weight the content, unlike a capture's age or a model's mediation.

Every rung is self-bounding on the Datawrapper hop's pattern (wall minus elapsed
minus `RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S`, skipped below its own floor), because
the provider's outer `asyncio.wait_for` discards
every page that already fetched when it fires, so an overrunning rung costs the whole
question's resolution evidence rather than just its own attempt. A rung that FIRED
records itself on the result (`route=` on the fetch marker, plus one
`RESOLUTION_SOURCE_ESCALATION` line); a rung that was SKIPPED is counted under
`details["counts"]` instead of logged, alongside the fired counts. A zero
renders nothing while still surviving into the archive, which is what makes "the rung existed
and never fired" distinguishable from "this record predates the rung". Seven of the keys count
rungs that FIRED: `meta_refresh_hops`, `impersonate_attempts`, `pdf_documents_read`,
`rendered_attempts`, `derived_api_reads`, `wayback_attempts` and `url_context_reads`. There is
deliberately no `impersonate_rescues`: rescues are read off `route=` on the fetch marker, which
already partitions the population by rung. Three count the extractor
policy's decisions rather than rungs. `chrome_metric_withholds` is a cited URL somewhere on whose
ladder the line-shape metric withheld an HTML extraction, because that extraction cleared the
chrome floor on navigation alone: the withhold flag is carried onto a later rung's rescue, so a
direct body the metric withheld still counts here when the rendered rung, a derived feed or the
paid reader goes on to serve the page, and it counts a chart-rescued page whose chart block
published without that text (on a page with no chart block its `reason` is the same `thin_page` an under-floor page carries, so
this count is what separates the two). `chrome_metric_withholds_rescued` is the subset a later rung
then served, that is a withheld extraction on a URL whose final result is `success` on a route
other than `direct`, which is what makes "the metric cost us the page" separable from "the metric
sent the page one rung further and the ladder delivered it". `precision_fallback_rescues` is a page
published from the `favor_precision` re-extraction after the default one failed that metric. The
rest count rungs that
were SKIPPED, one key per skip reason rather than everything folded into `rung_budget_skips`,
because each names a different binding constraint. `rung_budget_skips` is the question that ran
out of wall, summed over every rung; the same skips are broken out per rung as
`meta_refresh_budget_skips`, `impersonate_budget_skips`, `pdf_local_budget_skips`,
`derived_api_budget_skips`, `rendered_budget_skips`, `wayback_budget_skips` and
`url_context_budget_skips`, because the
aggregate cannot say WHICH rung the wall is binding on and "how often is the paid rung starved
by the pages before it" is the question the flag's rollout asks.
`pdf_contention_skips` is a document left unread while two others were parsing, so the two-slot
parse gate is what binds. `renderer_unavailable_skips` is a browser rung that never rendered,
most often because Chromium is missing on the runner (the install step is `continue-on-error` in
every workflow, so its absence is by design), and it is invisible in `rendered_attempts`. It
excludes a URL an earlier question rendered to nothing, which is its own
`rendered_no_text_skips`, and the four facts about the page that follow, each under its own key,
so neither a memo hit nor a hostile page can inflate the install-failed signal.
`render_timeout_skips` is a browser rung the transport's own DOM-read cap cut off: a page that
keeps navigating after the settle, which is a fact about the page rather than about the runner or
the question's clock, and also invisible in `rendered_attempts`. A render the rung's OUTER bound
cut off instead lands in `rendered_budget_skips` with the rest of the wall skips, because that cut
fires while the render is still queued behind the launch gates or once the transport has overrun
its exit reserve, neither of which is about the page. `render_non_200_skips` is a browser answered
a non-200 on a page the direct GET got 200 from, so the DOM is the host's error page and is not
read as content; what binds is the edge telling Chromium apart from our GET, the rate the
escalation ladder's own case rests on. `render_dom_too_large_skips` is a browser that rendered the
page to a DOM over `RENDERED_DOM_MAX_CHARS` and was declined unread; what binds is the size
ceiling, a fact about the page that used to be folded into `renderer_unavailable_skips`, where it
pointed triage at the Playwright install. `render_off_host_skips` is a browser whose main frame
landed on a host other than the pinned one, a server-side redirect hop the route handler never
sees, so the transport refused the DOM unread on its pre-read check, or discarded it unpublished
when the navigation committed during the read itself; it is the one count that says how often a
cited page sends the browser somewhere else, which is what prices the host-equality rule.
`wayback_cap_skips` is a question that spent its snapshot attempts on earlier cited URLs, so the
per-question cap is what binds; `url_context_cap_skips` is the paid rung's analogue, a question
that spent its per-question paid-read budget (`RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS`) on
earlier cited URLs, so the spend cap binds rather than the wall or the flag. `fast_path_skips`
is an expensive rung (the render, the paid read)
declined because the QUESTION's close-derived budget put it on the time-budget fast path, a fact
about the question's window rather than about the provider's own 45 s wall, which is what
`rung_budget_skips` counts. `url_context_robots_skips` is the free `Google-Extended`
pre-check earning its request: the host would have refused the read server-side, so that is
spend avoided rather than a page lost and it must not read as a failure. `url_context_no_api_key_skips`
is the paid rung enabled with no `GOOGLE_API_KEY` set: a misconfiguration rather than a tuning
signal, and counted precisely because without it "flag on, key missing" is byte-identical in the
archive to "flag off". Three keys belong to the impersonated retry. `impersonate_disabled_skips`
is the kill switch (`RESOLUTION_SOURCE_IMPERSONATE_ENABLED`) set to off, a configuration rather
than a tuning signal and counted for the same reason as the missing-key skip above.
`impersonate_unpinnable_skips` is a retry declined because a hop's host would not resolve to a
vetted public address to pin the connection to, on the first hop (nothing dialed; the direct fetch
resolved the same host moments earlier, so what binds is DNS disagreeing with itself, and a
nonzero count is a flake or a rebinding host rather than a refusal) or on a later redirect hop
(the earlier hops were dialed, and the target is one the direct fetch never resolved, so the
DNS-disagreement reading does not apply). `impersonate_host_refused_skips` is the per-run host
memo declining a cited URL on a host that already answered the impersonated client with a block
status this run, the memo doing its job rather than a failure, which is the same distinction
`rendered_no_text_skips` draws for the browser; the memo is process-global and shared with
gap-fill v2, so the earlier refusal may have been a v2 `fetch` or `read_document` of a URL no
question cited.

It is **SSRF-hardened** because these URLs are user-authored and fetches run from
CI: a preflight `is_public_http_url` check rejects private / loopback /
link-local / non-global IPs, userinfo tricks, and non-HTTP schemes, and a
connect-time `FilteringResolver` (the actual DNS-rebinding boundary, not the
preflight) re-checks every resolved IP. Redirects are followed manually under the
`MAX_REDIRECTS` hop cap (`research/http_fetch.py`, shared with the v2 agentic
tools), re-guarding each `Location`. That per-hop re-guard is the aiohttp path's;
the `route=rendered` rung guards its requests differently, because Playwright's
request interception never sees a server-side redirect hop. Since 2026-09-04 the
transport closes that channel for the main frame rather than accepting it: once the
navigation has settled it compares the LANDING HOST, the hostname `page.url` ended
up on, with the PINNED HOST that its `--host-resolver-rules` launch argument covers,
and refuses the DOM unread when the two differ. The same comparison runs again after
the DOM read, because a navigation can commit inside that one driver round trip, and a
DOM that fails it is discarded unpublished. It also registers a
`route_web_socket` handler that never connects the socket to a server, so a page
WebSocket, which HTTP interception cannot see at all, is blocked. What is left is a
cross-host SUBRESOURCE, whose host Chromium resolves with no pin of ours, and the
route-guard comment in `research/rendered_fetch.py` is the authority on
what that transport does and does not cover. The `route=impersonate` rung is the third
shape: libcurl never touches aiohttp's connect-time `FilteringResolver`, so the transport in
`research/impersonated_fetch.py` carries the invariants itself. Every hop is pre-resolved
through the repo's one vetting predicate (`rendered_fetch.resolve_pinned_host`, which
rejects the whole hostname if ANY resolved address is disallowed), pinned to that address
with `CURLOPT_RESOLVE` on a session built for that one hop, and checked after the fact
against the address libcurl reports it connected to, refusing the body unread on a
mismatch. No automatic redirects: every hop is re-guarded through `_hop_refusal`,
re-resolved and re-pinned under the shared `MAX_REDIRECTS` cap, the proxy environment is
disabled so a pin cannot be bypassed by an `HTTP_PROXY` libcurl would otherwise honour,
and the body is capped at the same decompressed byte counts as the direct fetch: the page cap
(`RESOLUTION_SOURCE_MAX_RESPONSE_BYTES`) for every body, with one re-dial under the document
cap (`DOCUMENT_TEXT_PDF_MAX_BYTES`) when a declared PDF aborts on the first, so a cited PDF
between the two is read on this rung as `_resolution_pdf_outcome` would have read it. Every
refusal raises, and nothing from a refused response is returned. Per-URL truncation appends a
`[truncated at N chars — full source at URL]` marker, the aggregate section-budget
trim routes through the same marker-emitting truncator (a bare slice could cut
mid-sentence and eat that marker, so an already-truncated page rendered as
complete), and the formatter appends `[N additional source(s) omitted — section
budget]` when later sections are dropped for length.

The per-URL `FetchStatus` distinguishes two kinds of non-success, and only one is a
seam. `blocked` / `js_wall` / `no_resolving_content` are pages we could not READ, and
they are what the escalation rungs above trigger on, as is the `no_text_layer`
half of `unreadable_document` (a scan, where a model really is the only route). The one
exception inside that family is `no_resolving_content`'s `no_matching_passage`: we read the
whole document, so there is no harder fetch to try and the paid rung skips it. `empty_body`
(a 200 whose body is empty or whitespace-only) and `unsupported_type` (including a body whose
declared charset decodes to mojibake) are bodies that carried no information, refusals
rather than seams, because there is nothing on the other side to fetch harder. Both
exist because `status="success"` has to mean CONTENT: as `success`, an empty body
rendered an empty section under the "primary grading evidence" caveat, suppressed the
all-failed "yielded no usable content" notice for every sibling URL, and reported `ok`
to provider diagnostics.

`ungrounded` is the twelfth and newest status, and the only one that cost money: the paid
url_context rung answered with zero successful retrievals, so what came back is recall rather
than a read of the page and it is discarded rather than rendered. It has its own token because
it says something no other status does, that the host answered a third-party fetcher's request
with nothing while refusing ours. On the reason side there are now two vocabularies rather than
one, split by what each qualifies. `FetchStatusReason` qualifies a result's STATUS:
`embed_shell` / `thin_page` / `no_matching_passage` / `not_addressed` under
`no_resolving_content`, `no_text_layer`
/ `encrypted` / `malformed` under `unreadable_document`, and `budget_skipped` / `parse_contention`
under the `unsupported_type` a held-but-unparsed document earns. `RungSkipReason` qualifies a rung
attempt that PRODUCED NO RESULT (`RungAttempt.skipped_reason`), which is why it has nowhere else to
record its reason. Most of those attempts never ran at all; four of them did run a browser and
came back with nothing usable, `render_timeout`, `render_non_200`, `render_dom_too_large` and
`render_off_host`, so "skip" here means "produced nothing" rather than "never started". The
vocabulary is `wall_budget`, `wayback_cap`, `url_context_cap`, `fast_path`, `no_api_key`,
`robots_disallowed`, `rendered_no_text`, `renderer_unavailable`, `render_timeout`, `render_non_200`,
`render_dom_too_large`, `render_off_host`, and `parse_contention` again (a held document declined
for want of a parse slot records the skip AND stamps the withheld result's `status_reason`, the one
token shared by both Literals). Both are a closed `Literal`, so a misspelt reason is a type error
rather than a permanently-zero count.
`renderer_unavailable` is the browser declining before it rendered anything (missing, broken,
unpinnable host); `render_timeout` is a render the transport's own DOM-read cap cut off because the
page kept navigating, and the two are kept apart because only the first says anything about
Chromium. A render the rung's own outer bound cut off is `wall_budget` instead, because that bound
fires while the render is still queued behind the launch gates or once the transport has overrun
its exit reserve; a browser answered a non-200 where the direct GET got 200 is `render_non_200`;
a rendered DOM over `RENDERED_DOM_MAX_CHARS` is `render_dom_too_large`; and a main frame that
landed on a host other than the pinned one is `render_off_host`, the server-side redirect hop the
route handler never sees, whose DOM was refused unread on the transport's pre-read check or
discarded unpublished when the navigation committed during the read itself.
`no_resolving_content` has four reasons (`embed_shell`, `thin_page`, `no_matching_passage` and
`not_addressed`): the third is the only one that is a document rather than a page, and the fourth
is the paid reader's, a page Gemini retrieved whose answer said it does not discuss the ask. All
of them live in `research/resolution_fetch_result.py` with the rest of the vocabulary.

`vacuous_body_status` (`research/resolution_fetch_result.py`) is the one place that
decision is made, on every raw-body branch, Tier-1 JSON/XML/text/CSV and the Tier-2
dataset alike. Three ways a 200 carries nothing. It could not be DECODED: the body is
decoded BOM-first, then by its declared charset, and an undecodable-character ratio
above `MAX_UNDECODABLE_CHAR_RATIO` is refused as `unsupported_type`, because mojibake
like `0�.�4�2�` type-checks as text and rendered as grading evidence. It is empty or
whitespace-only, which is `empty_body`. Or (datasets only) it is not row-shaped, so
nothing may claim it is the chart's live series. That third check is deliberately
ordered BEFORE the freshness verdict, so an empty CDN body cannot borrow
`stale_data`'s benign diagnostics token (a DATASET's `stale_data`, the one carrying a
`chart_id`, reports to diagnostics as the benign "guard working as designed", which would
hide a broken hop; a cited page's `stale_data` from the Wayback rung is a lost source and
keeps its loss token). Row shape is also
decided on the PRE-strip text, because `looks_like_csv_rows` rejects markup by its
leading `<` and stripping first would remove exactly the allow-listed fragment tags
(`<p>`, `<div>`) a CDN soft-404 opens with, letting an error page carry the
authoritative `Dataset published` lead if its prose holds a comma. Those allow-listed
HTML tags ARE stripped from raw CSV/text bodies before truncation, which is worth 58
rows versus 13 at the same character budget on a live-shaped poll table.

That notice says "yielded no usable content" rather than "was
unreachable" because two of the statuses it covers (`no_resolving_content` and
`empty_body`) are pages that answered HTTP 200 and carried nothing, and "the tracker
was down" is different evidence from "the tracker has no reading"; the per-domain status
token beside it says which happened.

`no_resolving_content` (2026-09-01) is the newest of those seams and covers the page
that answers 200
with nothing but chrome. The floor is what decides it: below
`RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS` (400 characters) of extracted text the
page is withheld under
this status, which costs nothing because everything archived below that floor is site
chrome and the shortest archived extraction that carries the resolving content is 401
chars. That calibration was re-checked against the same census when the gate was
generalised: of 68 cited successes, all 8 below 400 chars are chrome, and the
per-URL list is in the constant's own comment. Above the floor the text still has to be
content-shaped (the extractor policy above: table rows and long lines, not a menu tree),
and a page that is chrome at both extractor settings takes the same `thin_page` withhold.
A page that passes is rendered as is, and where a
third-party data embed hid figures from it one bracketed line says plainly that those
figures are not in the text.

`FetchResult.status_reason` records which shape of chrome it was. `embed_shell` means the RAW HTML
named an embed whose numbers are real but locked inside it: Infogram, Flourish or
Tableau, detected by `unreadable_data_embed_providers` because trafilatura emits no
iframe or embed-script URLs at any setting; Datawrapper is deliberately excluded from
that scan since the Tier-2 hop reaches it. `thin_page` means no such provider was named.
The status's third reason is not a shape of chrome at all: `no_matching_passage` is a cited
document we read in full that discusses nothing the question asks about, withheld under the
same status because the outcome for a forecaster is the same, a section with nothing in it to
grade against. Its fourth, `not_addressed`, is the paid url_context rung's equivalent: a page
Gemini retrieved whose answer opened with the prompt's `NOT_ADDRESSED` sentinel, withheld for
the same reason.
That distinction used to be a GATE rather than a label, and the gate was removed on
2026-09-02 because it was wrong: the
2026-09-01 residual round found five content-free `success` renders and not one of them
named a provider, among them q45088's 127-char single-page-app tab list and q45215's 385
chars of Kazakh region names, both published under the primary-grading-evidence caveat.
The embed half of the story still comes from qids 44554/44556, where a Senate-forecast
tracker returned HTTP 200, extracted 2.9k chars of background, and published with zero
polling numbers in it, byte-identical across three questions, with the resolving average
sitting in two Infogram iframes and nothing anywhere saying so. `js_wall` keeps its own,
much lower floor and is checked between the two verdicts, so generalising the chrome
floor did not absorb the JS-walled population. A page can also draw both verdicts at once
and should: Tier-1 `no_resolving_content` on the page next to a Tier-2 `success` on its
Datawrapper dataset is the correct reading of a tracker whose prose we cannot use and
whose series we can.

One more rung (2026-09-02) reads data out of the page we already hold, with no second
request and no
LLM call. `resolution_chart_data.render_inline_chart_data` scans the raw HTML for a
Highcharts config (a `data-chart="{…}"` attribute, or a `Highcharts.chart(…)` call whose
argument is strict JSON) and, after an HTML-entity unescape plus a plain `json.loads`,
renders each series' most recent points as a compact
labelled block that leads the page text. Nothing is summed, interpolated,
unit-converted or re-derived: the block states the values the page's own chart holds; a
declared `datetime` x axis renders as UTC dates rather than epoch millis; and a config
that
does not parse is skipped at DEBUG. It runs on every fetched HTML page rather than only
thin ones, because the record it exists for is q43949, whose resolving IOM page extracted
roughly 80k chars of incident rows and prose carrying none of the resolving figures while
its annual series sat in the attribute, reading 1,240 for 2026 in a Wayback snapshot 25
days
before a forecast that landed about 340 too high. A thin-only gate would have missed the
record the rung exists for. Because chart data counts as content,
it also rescues a page the chrome floor would otherwise withhold.

Every fetched URL (Tier-1 page and Tier-2 dataset hop alike) emits one
`RESOLUTION_SOURCE_FETCH: question=... url=... status=... http=... embeds=...
[reason=...] [route=...]` line, harvested as `resolution_source_fetch`. `reason` is
appended only where the status alone is ambiguous, so archived lines stay
byte-identical and its absence keeps meaning "no reason applies"; `route` names which
rung of the escalation ladder produced the outcome. It REPLACED the older free-text
`resolution_source fetched <netloc> (<status>)` lines rather than joining them, so each
fetch is logged exactly once, and it is what turns a cut like "cdc.gov is 0 successes
in 1,069 fetch records" into a query rather than a re-scrape of run logs
that expire from GHA at 90 days. Because that line carries only the FINAL outcome per
URL, each escalated rung additionally emits `RESOLUTION_SOURCE_ESCALATION` with the
status that triggered it, the rung tried, what came back, and the wall-clock the rung
cost, which is what makes "does this rung rescue anything, and is it worth its latency"
answerable. The two lines spell one state two ways: a fetch that worked is
`status=ok` on the fetch line (the shared `fetch_outcome_token`, whose `ok` is what the
diagnostics formatter reads as "this source contributed") and `outcome=success` on the
escalation line (the verbatim `FetchStatus`), so a query joining the two has to treat
`ok` and `success` as the same outcome. Both are data contracts, so the difference is
documented here rather than re-spelled on either side. See "Reading run logs" in
`docs/operations.md` for the field meanings.

The paid rung adds three greppable log lines, each a registered marker spec since 2026-09-04,
when `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` went on in every bot workflow (until that merge none
of them could fire in production, and a spec would only have added an always-empty archive
column): `RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP: url=... host=...` (an INFO, the free pre-check
avoiding a known-zero paid read; harvested as `resolution_source_urlcontext_robots_skip`),
`RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url=... statuses=...` (a WARN, a paid read
discarded for retrieving nothing; `statuses` is every reported `url_retrieval_status`, `none` when
the SDK attached no entry; harvested as `resolution_source_urlcontext_ungrounded_suppressed`) and
`RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED: url=... host=...` (a WARN, a paid read withheld
because its answer opened with the `NOT_ADDRESSED` sentinel, so the page was retrieved and has
nothing on the ask; harvested as `resolution_source_urlcontext_not_addressed`). Their spellings
are pinned twice, by the rung's own tests and by the spec tests in
`tests/test_telemetry_markers.py`, so the emitter and the archive cannot drift apart, and the
first two are parallel to their `AGENTIC_URLCONTEXT_ROBOTS_SKIP` /
`AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED` twins on the gap-fill v2 reader. No archived run from
before that merge carries any of the three, so an era-bucketed rate starts there.
`docs/operations.md` "Reading run logs" has the field meanings.

The shared browser transport adds one greppable line of its own, a registered marker spec since
2026-09-04: `RENDERED_FETCH_OFF_HOST: scope=<resolution_source|gap_fill_v2> pinned_host=<host>
landed_host=<host> same_publisher=<true|false>`, a WARNING harvested as `rendered_fetch_off_host`. It
fires wherever the render was asked for, so `scope` says which caller paid for the launch, and it
names hostnames only, never the landing URL, which can carry a session token. `same_publisher` is `true` when the landing host shares the pinned host's registrable domain (a benign client-side hop such as `example.com` to `www.example.com`, refused by strict hostname equality and priced by this value) and `false` otherwise, including every landing with no hostname, so a `false` record is the security signal and a `true` record prices the strictness. It is the ONLY per-event record of an off-host
landing, because a refused render is a skip and a skip emits no `RESOLUTION_SOURCE_ESCALATION` line;
the per-question rate lives in `render_off_host_skips` under `details["counts"]`.

Like prediction markets, it is **hard-disabled under benchmarking** (current page
content post-dates any backtest window), on the same leakage rationale. The section
header is `## Resolution Source Snapshot`, and `RESOLUTION_SOURCE_ENABLED` was flipped
on in the three prod yamls on 2026-07-10 after a live-output eyeball.

### Time-series anchor: `TS_ANCHOR_ENABLED` (chart side-channel `TS_ANCHOR_CHART_ENABLED`, off)

A deterministic empirical anchor for numeric questions whose resolution series is
a fetchable FRED/yfinance series (`research/timeseries_anchor.py`). No LLM, no
model selection: it renders the latest value, dated ("as of DATE", with an
in-progress marker when today's bar is still forming and a stale-latest warning (the
same `FINANCIAL_STALE_LATEST` marker as the financial-data provider) when the newest
observation is older than its cadence explains), a multi-resolution
history, a 52-week range, and a horizon-matched empirical band built only from
the series' own past. The Phase-A offline replay found CV-gated model picks beat naive
out-of-sample only 43% of the time, while the naive empirical band was sharper and
better tail-calibrated, so this ships the naive band on purpose.

It is the *first* backtest-safe research provider: instead of hard-disabling under
benchmarking, it pins `as_of` to the question's `open_time` and fetches
point-in-time up to that date (ALFRED vintages for revising macro series), so the
data known at forecast time is the answer without leaking the resolution. The
text anchor is on in production; the chart-image side-channel
(`TS_ANCHOR_CHART_ENABLED`) is a separate flag and is off.

### SEC EDGAR client (`research/sec_edgar.py`; known-API ladder rung)

A client for SEC EDGAR's public JSON APIs, built 2026-09-09 so that sec.gov stops being a
blocked host. The known-API rung translates registered EDGAR URLs and answers them before any
page fetch is attempted. It remains inert when `SEC_EDGAR_CONTACT_EMAIL` is unset, so local and
CI runs without the contact secret still fall through to the ordinary page ladder.

**Why an API client rather than another fetch rung.** The 2026-09-09 fetch-gap inventory counted
12 blocked events on sec.gov across 5 questions (`Archives/edgar/data/...` filing documents:
SpaceX's S-1 amendments, Uber's 10-K and the Delivery Hero 8-K, Oracle's 10-K, an ETF 10-Q), all
HTTP 403 to the browser-shaped fetch. That 403 is EDGAR's fair-access policy
(sec.gov/os/webmaster-faq, "Developers"): automated access must "declare your user agent in
request headers" in the form `Sample Company Name AdminContact@<sample company domain>.com`,
and "our current maximum access rate is 10 requests per second". A browser fingerprint is the
wrong answer to a policy that asks for a name and an address, and the same facts are served as
JSON on data.sec.gov, so the fix is a client that follows the policy.

**What it reaches.** Every function takes the session from `edgar_session()`:

| Function | Endpoint | What it answers |
|---|---|---|
| `company_submissions(session, cik_or_ticker)` | `data.sec.gov/submissions/CIK##########.json`, and `www.sec.gov/files/company_tickers.json` when given a ticker | A filer's identity and its recent filings, each with `form`, `filing_date`, `report_date`, `primary_document` and a `primary_document_url` |
| `company_facts(session, cik)` | `data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | Every XBRL fact the filer reported; `CompanyFacts.values(concept, unit, taxonomy=)` returns them dated, oldest first, with the form and filing they came from |
| `frame(session, concept, unit, period)` | `data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json` | One concept across every filer for a calendar period, `CY####`, `CY####Q#` or `CY####Q#I` |
| `full_text_search(session, query, date_from=, date_to=, forms=)` | `efts.sec.gov/LATEST/search-index?q=&dateRange=custom&startdt=&enddt=&forms=` | Filings since 2001 matching a phrase, filtered by root form and file date, each hit carrying a `document_url` |
| `filing_document(session, url)` | `www.sec.gov/Archives/edgar/data/{cik}/{accession}/{file}` | The raw bytes and content type of one filing document; EDGAR hosts only |
| `ticker_to_cik(session, ticker)`, `pad_cik(cik)`, `filing_document_url(cik, accession, file)` | | The lookups the above are built from |

The endpoint shapes were read from SEC's API page (sec.gov/search-filings/edgar-application-
programming-interfaces) and confirmed against live responses on 2026-09-09; the full-text search
parameters were confirmed from the server's echoed query (`terms` on `root_forms`, `range` on
`file_date`), since SEC publishes no reference for that endpoint.

**The fair-access rule as implemented.** The User-Agent is
`SEC_EDGAR_USER_AGENT_TEMPLATE` with the contact read from `SEC_EDGAR_CONTACT_EMAIL` when the
session opens; unset, `edgar_session()` raises `SecEdgarContactUnsetError` before any socket
opens, so an anonymous User-Agent is never sent (a guard fails shut). Request starts are spaced
to `SEC_EDGAR_MAX_REQUESTS_PER_SECOND` (8, under SEC's 10) by one process-wide, loop-scoped
spacer shared across all three EDGAR hosts and every concurrent question, and each request also
holds the shared per-host politeness semaphore from `http_fetch`. Bodies stream through
`read_body_capped` under `SEC_EDGAR_MAX_RESPONSE_BYTES` (16 MiB, because a large filer's
companyfacts is 7.9 MB and Oracle's inline-XBRL 10-K is 6.9 MB, both past the 5 MiB page cap).
No retries: a 403 is a policy verdict and a 429 is the ceiling itself, and either raises
`SecEdgarError` with the status and SEC's body snippet. Redirects are refused rather than
followed, because every URL the client dials is a documented endpoint or an Archives document and
a followed hop would escape the spacer, the host gate and the EDGAR host check while still carrying
the fair-access User-Agent. The transport is `http_fetch.build_session` with the 20 s
`RESOLUTION_SOURCE_HTTP_TIMEOUT`. One server behaviour the client works around: the full-text
search drops its `file_date` filter entirely when only one of `startdt` / `enddt` is sent (probed
2026-09-09), so a one-sided range is completed with 2001-01-01 or today before it travels.

**The operator's one manual step, before it is wired in:** set `SEC_EDGAR_CONTACT_EMAIL` in the
local `.env` (see `.env.template`) and as a GitHub Actions secret surfaced into every bot
workflow's environment. Without it the client raises on first use, which is the intended
behaviour and also why no workflow should reference the module until the secret exists.

### Page digest (`research/page_digest.py`; the `page_digest_extractor` role)

The shared ladder binds `policy.digest` to `digest_page` for cited HTML over
`RESOLUTION_SOURCE_PER_URL_MAX_CHARS` and for held flat text read through gap-fill v2's
`read_document`. Fresh and cached HTML use the same presentation path with the current query
and remaining wall. Ordinary gap-fill `fetch` remains paginated, and PDFs retain their
page-aware BM25 digest. The driver-facing local-read method stays `digest_local`.

**Why a model and not the BM25 digest.** A cited page over the per-URL cap was read from the top,
so its tail was unreachable, and the loop's deterministic BM25 digest reached the tail with a
lexical ranker the operator does not trust as the primary mechanism. The operator's decision
(2026-09-09, `scratch_docs_and_planning/fetch_ladder_unification_plan_2026-09-09.md`, "The page
digest, as agreed") was a cheap model reading the page, a literal grounding check as the
hallucination guard, and BM25 demoted to pre-filter and fallback.

**What one call does.** `digest_page(text, query, budget_seconds=...)` returns a `PageDigest`
(`passages`, `passages_returned`, `passages_grounded`, `fallback_used`, `method`). The page's
opening window (`DOCUMENT_DIGEST_WINDOW_CHARS`, cut on a word boundary) always leads the passages,
so a reader still sees what the page is. The BM25 ranking of the page (`document_text.select_passages`)
and the page's normalisation run first, in one `asyncio.to_thread` hop like the two existing callers
of that ranker, and the hop's elapsed time is charged against the caller's budget. A page over
`PAGE_DIGEST_PREFILTER_MAX_CHARS` reaches the model as its best BM25 windows for the query in page
order, abutting windows spliced back into one span and `[...]` marking only a real cut; a shorter
page goes to the model whole. One call at `PAGE_DIGEST_EXTRACTOR_MODEL` and
`PAGE_DIGEST_EXTRACTOR_EFFORT`, built through `build_llm_with_openrouter_fallback` like every
support role, with a strict `response_format` schema (`PageDigestPassages`, a list of strings) and
`provider.require_parameters` so OpenRouter rejects the request rather than dropping the schema,
asks for at most `DOCUMENT_DIGEST_TOP_K` verbatim passages that bear on the query, most relevant
first. The prompt is `PAGE_DIGEST_EXTRACTOR_PROMPT` in the module, and its load-bearing clauses
(verbatim, ranked, no paraphrase) are pinned in `tests/test_page_digest.py`. The query is the
caller's: the fetcher passes the question title plus resolution criteria, the loop passes the
driver's ask.

**The grounding check is literal.** A returned passage is accepted only when it is a substring of
the page text after whitespace and quote-glyph normalisation (every run of whitespace collapsed to
one space and every straight, curly or backtick quote deleted, on both sides; the glyph rule is the
one `agentic/provenance.py` learned when a curly quote retyped straight hid a verbatim copy). The
passage a reader sees is the model's own text, so a copied table keeps its rows. A passage that fails
is dropped and counted; a repeat of an accepted passage, or one already inside the opening window,
is dropped but still counts as grounded. `passages_returned` minus `passages_grounded` is therefore
the model's fabrication count for that page. At most `DOCUMENT_DIGEST_TOP_K` accepted passages are
served after the opening, the same bound as the BM25 path, so the two mechanisms hand a caller the
same order of text.

**The fallback is today's digest.** The first `DOCUMENT_DIGEST_TOP_K` windows of the BM25 ranking
already computed are served with `fallback_used=True` and `method="bm25"` when the call raises one
of the expected failures (the `asyncio.wait_for` timeout; any `openai.APIError`, which is the root
of every litellm provider and API error; a pydantic `ValidationError` on an off-schema answer; or
the bare `RuntimeError` forecasting-tools raises for an empty completion, recognised by
`llm_retry.is_zero_output_failure` and the repo's most-seen zero-output failure), when nothing the
model returned survives grounding, when the page is empty, or when the remaining wall is too short
to try. Each of those paths logs one `PAGE_DIGEST` line naming its reason, which is the only place a
skipped call is told from a failed one (the marker's three counters read the same). An answer whose
grounded passages all lie inside the opening window is a success, not a fallback: the opening alone
is served with `fallback_used=False`. Any other exception propagates. The call is bounded by
`min(PAGE_DIGEST_EXTRACTOR_TIMEOUT_S, budget_seconds - elapsed - PAGE_DIGEST_WALL_MARGIN_S)`,
where `elapsed` is the BM25 hop's own time, and is not attempted below
`PAGE_DIGEST_MIN_CALL_BUDGET_S`, so a caller hands over its remaining wall verbatim and the digest
never lets a fetch overrun it. There is no retry: the fallback is the retry, and a second paid
attempt cannot fit inside the wall. Receipts for every constant: docs/constants.md "Page digest".

**Billing and telemetry.** Every call is tagged `role=page_digest_extractor`, so it lands on the
`CREDIT_ROLE_SPEND` ledger beside the other support roles: on a Metaculus run the donated key with
the personal fallback, on a Mantic run the personal key only, by the same rule as every
`openrouter/openai/` slug (docs/operations.md "API keys"). The module emits no marker. The
`PageDigest` carries the three counters (`passages_returned`, `passages_grounded`,
`fallback_used`) for the callers to append to their `RESOLUTION_SOURCE_FETCH` line as optional tail
fields (docs/telemetry_markers.md "RESOLUTION_SOURCE_FETCH").

**One known seam.** forecasting-tools trips a bare `assert isinstance(answer, str)` when a
completion's content is `None` (the shape a reasoning model produces when it spends its whole
budget on reasoning tokens). That `AssertionError` is not caught here, because catching it would
also swallow the four unrelated invariant asserts in the same forecasting-tools method, so a
`None`-content completion propagates to the caller rather than degrading to BM25. If the ladder's
seat wants that shape degraded, the fix is a typed exception upstream or the seat's own boundary,
not a broader catch here.

### Known-API registry (`research/known_api/`; wired into the ladder and gap-fill)

A URL whose host the registry recognises is answered by that host's public API rather than
fetched as a web page: no LLM, no paid key, the same client code the research providers already
use. It was built to fill rung 0 of the shared fetch ladder (the `policy.known_api` seat) and to
give the gap-fill v2 driver three explicit tools for a date window the URL forms cannot express.
`known_api/wiring.py` binds the registry to the ladder's rung-0 callback, and
`agentic/tools.py` appends the three deterministic tools to the gap-fill tool list. The callback
is bounded by the current per-URL remaining wall, so a slow API URL can fall through independently
while sibling URLs keep their completed results.

**What translates to what.** `translate(url)` (`known_api/translate.py`) reads one URL into one
`KnownApiCall`, or returns `None` when the registry does not own the host (which leaves the later
ladder rungs to try the page). The shapes it covers, all drawn from the 2026-09-09 cost pass's
observed driver fetches:

| Host and shape | Call |
|---|---|
| `fred.stlouisfed.org/series/{id}`, `/data/{id}` | `fred_series(id)` |
| `fred.stlouisfed.org/graph/fredgraph.csv`/`.xls?id=A,B` with `cosd`/`coed` or `observation_start`/`observation_end` | `fred_series` per id, at most two, window from the params |
| `alfred.stlouisfed.org/series?seid=`, `alfredgraph.csv?id=&vintage_date=` | `fred_series(id, first_release=true)` |
| `api.stlouisfed.org/fred/series/observations?series_id=` | `fred_series(id)` |
| `finance.yahoo.com/quote/{sym}` (regional `uk.`/`ca.` hosts, URL-encoded symbols, optional `period1`/`period2`) | `yahoo_history(sym, start, end)` |
| `query1.finance.yahoo.com/v8/finance/chart/{sym}` | `yahoo_history(sym, start, end)` |
| `kalshi.com/markets/{ticker}`, `/markets/{series}/{slug}/{ticker}`, `/api/v#/markets/{ticker}` | `market_snapshot("kalshi", ticker)` |
| `www.sec.gov/Archives/edgar/data/...` | EDGAR `filing_document` |
| `www.sec.gov/cgi-bin/browse-edgar?CIK=...` | EDGAR `company_submissions` |

Yahoo help pages, Yahoo/other news articles, the Kalshi contract-terms PDF on S3, and the FRED
release calendar are not translatable and stay on the page ladder.

**The backends** (`known_api/backends.py`) each return a neutral `KnownApiResult`
(`known_api/result.py`: a status in the `ok`/`empty`/`not_found`/`error` family, the rendered
markdown, the canonical source URL, and its links) and never raise to the caller: an unknown id
is `not_found` with the provider's message, an empty window is `empty`, a transport or quota
failure is `error` naming the exception class.

- `fred_series` reads one series over a date window through the keyed API (`fred_rendering.Fred`)
  when `FRED_API_KEY` is set, with an initial-release comparison available on `first_release`
  while the displayed observations remain current-vintage, and through the keyless `fredgraph`
  CSV (`ts_fetch.fetch_series`) otherwise; free text runs `Fred.search`. Keyless reads are
  current-vintage only.
- `yahoo_history` reads one symbol's adjusted price history through `ts_fetch.fetch_series`
  (yfinance), column in Close/High/Low/Open.
- `market_snapshot` reads a venue plus market: Kalshi by ticker through the event endpoint falling
  to the market endpoint, Kalshi free text over a supplied catalogue with zero new requests,
  Polymarket and Manifold search, and PredictIt from a supplied cached dump; it renders through
  the prediction-market snapshot renderer. The gap-fill binding supplies neither optional cache,
  so free-text Kalshi and PredictIt reads decline unless a caller provides those resources.
- `edgar` reads a translated EDGAR URL, and declines (returns `None`, so the ladder falls through
  to the page fetch) when `SEC_EDGAR_CONTACT_EMAIL` is unset; a company page renders its filings
  table and an Archives document its extracted text.

**The bounds** (`known_api/backends.py` constants): one series or ticker per call; a windowed read
capped at 400 observations, newest kept, with a line saying so, and a 30-observation default;
15 s on each FRED or Yahoo HTTP request, with a 15 s async response bound around the blocking
operation; the async bound declines a slow result but cannot terminate its worker thread, while
the client-side timeout ensures a stalled socket eventually releases it. A keyed FRED read can
make several requests for metadata or the optional first-release comparison, so these are
per-request bounds rather than a guarantee that the complete enrichment finishes in 15 s. Five
market rows; at most four Kalshi detail GETs per question, through one `KalshiGetBudget` shared by
the explicit market tool and the rung-0 callback; `PLATFORM_HTTP_TIMEOUT` per venue call. The
explicit tools and rung-0 callback also share the gap-fill question's HTTP session. The ladder
applies the remaining question wall to each callback invocation, and a timeout declines that URL
so another cited URL can continue.

**The two adapters** (`known_api/adapters.py`) are the whole coupling to the two callers:
`to_tool_outcome` returns the gap-fill loop's `ToolOutcome` with method `known_api`, and
`to_fetch_result` returns the resolution-source fetcher's `FetchResult` with a success carrying
the rendered text, route `known_api`, and the backend's canonical source links. The route is
recorded as a fetched provenance method, so the normal ladder and agentic provenance paths retain
the API source explicitly.

**No drift with the extraction seams.** The financial-data provider's identifier extraction
(`financial_data.extract_financial_identifiers_from_criteria`) and the resolution-source
fetcher's Yahoo skip (`resolution_url_scan.is_yahoo_ticker_url`) both consume `known_api.parse`,
so the fetcher's skip, the provider's extraction and the rung-0 translation cannot disagree about
what counts as a FRED, Yahoo or Kalshi URL. The provider now reads ids out of the `fredgraph`
CSV/XLS and the `query1` chart endpoint, and the fetcher's skip accepts the regional Yahoo hosts.

## Gap-fill (two passes, both concurrent, both on in prod)

After the primary + add-on bundle is assembled, two independent gap-fill passes
run **concurrently** in one `asyncio.gather` inside `run_research`
(`research/orchestrator.py`), so the research-phase wall-clock is `max(v1, v2)`,
not the sum. Each runs inside its
own try/except so a defect in one can never zero the other's output. Both consume
the pre-gap-fill bundle, which means the v2 driver's brief does not see v1's
addendum; v2's section appends after v1's.

### v1: targeted gap-fill (`research/targeted.py` `run_gap_fill_pass`)

Three stages, gated by `GAP_FILL_ENABLED` (and skipped when the first-pass bundle is
shorter than `GAP_FILL_MIN_RESEARCH_CHARS`, or when the question's close-derived time
budget drops it: the fast path, or a research phase that ran out of budget):

1. A non-grounded OpenRouter analyzer LLM (`GAP_FILL_ANALYZER_MODEL`, low effort)
   reads the first-pass research and emits a JSON list of up to `GAP_FILL_MAX_GAPS`
   factual gaps, ranked by decision-relevance, so the trailing slot holds the least
   forecast-moving gap. Since 2026-09-09 each gap carries three grades the code reads:
   `answerable_now`, `already_in_first_pass` and `same_need_as` (the prompt's
   `GRADE EVERY GAP` clause, `docs/prompts.md` "Research-side prompt rules"). The parser
   keeps one slot per listed item, so a malformed item (not an object, or no gap text)
   becomes an empty slot rather than shifting every later `same_need_as` pointer.
2. `triage_gaps` drops the gaps whose grades fail, dedupes restatements, and caps the
   survivors at `GAP_FILL_MAX_GAPS`, all before any resolver call; the
   `GAP_FILL_V1_TRIAGE` marker records the counts. The rules, the decisions behind them
   and the receipts are in "v1 triage" below.
3. Each survivor is resolved by a parallel OpenAI native web search
   (`GAP_FILL_RESOLVER_MODEL` at `GAP_FILL_RESOLVER_REASONING_EFFORT`, via
   OpenRouter on the donated key), briefed with the gap, the suggested query, the
   question title, and since 2026-09-09 the resolution criteria and fine print, so a
   "which figure resolves this" gap is answered against the criteria rather than the
   title (receipt q44267, `docs/prompts.md` "Research-side prompt rules").
   Because the searches run in parallel, latency is the slowest call, not the sum.
   The addendum numbers the survivors `### Gap 1..K` in analyzer order, and that index
   is the index into the raw record's `gaps` and `results`; a dropped gap's analyzer
   position is in the record's `dropped` list.

The resolver migrated off direct-Google grounding on 2026-06-25, which is why
`GOOGLE_API_KEY` is no longer required for gap-fill, and its model went sol → terra on
2026-07-20: terra was preferred-or-within-noise in all three 2026-07 blind role audits
at ~40-50% lower cost, which matters here because these searches are the single biggest
research line item at ~44% of spend. The whole pass never raises
(it returns `""` on any error) and appends its results under
`## Targeted Gap-Fill (second pass)`.

### v1 triage: the grade filter (`research/targeted.py` `triage_gaps`)

The lean-out of gap-fill v1 shipped 2026-09-09 on an operator ruling: lean by GRADE, in code, never
by position, with no second LLM pass and the cap left at 4. The 2026-09-09 cost pass found that
about a third of v1's resolver calls, roughly $0.21 of its $0.71 a question, bought nothing on the
archive: future-dated asks were 18% of gaps and a third of gap-one slots, 47% of the forced
current-reading gaps came back with a reading the first pass already carried with its date, and one
question in three carried a paraphrase pair, usually gap 2 restating gap 1. A positional cap of 2
would have saved more ($0.30) but dropped the useful gap on 4 of 6 traced questions, because the
analyzer's ranking is unreliable (`scratch/cost_pass_2026-09-09/v1_gap_redundancy/REDUNDANCY.md`,
`scratch/cost_pass_2026-09-09/v1_vs_v2/TRACES_SYNTHESIS.md`).

The analyzer grades every gap on three fields, and `triage_gaps` applies them in this order, one
reason per gap so the marker's counts partition the list:

1. **Grades.** `answerable_now` false (the gap can only be answered by an observation not yet
   made or a result not yet published; the prompt carries no run date, so the grade is defined
   by whether the observation exists, not by a date) drops the gap as `not_answerable`;
   `already_in_first_pass` true (the first pass already states the value or fact with its date)
   drops it as `in_first_pass`. These are the two discipline rules the ANSWERABLE NOW block
   already stated, made checkable.
2. **Dedupe.** `same_need_as` is the 1-based position of an earlier gap in the same list that the
   same fact from the same source would answer, else null. It is followed to the NEED it names, the
   root of its pointer chain, not to the literal gap: a restatement of a need that a kept gap is
   searching, or that the first pass already answers, is dropped as `same_need`; a restatement of
   a need nobody covers, because its earlier phrasing was future-dated or ungraded, becomes the
   need's carrier and is kept, and every later restatement of that need, whether it points at the
   carrier or at the original phrasing, is dropped. So gap 3 restating gap 2 restating gap 1
   collapses onto gap 1; a future-dated gap 1 with two present-tense rewordings as gaps 2 and 3
   keeps gap 2 only; a first-pass-answered gap 1 and its rewording as gap 2 keeps neither, whatever
   gap 2's own `already_in_first_pass` says.
3. **Schema drift fails shut.** Only the two booleans are required. A gap missing either of them, or
   carrying a string where a boolean belongs, or pointing at itself, forward, at zero or with a
   non-integer, or sitting in an empty slot, is dropped as `schema`, never read as passing: a grade
   that defaulted to passing would spend exactly the money the grade exists to save. An ABSENT
   `same_need_as` key still reads as null for compatibility with archived outputs. New analyzer
   calls enforce all six fields with strict JSON Schema and `provider.require_parameters=true`;
   providers must support the schema rather than silently dropping it. A `same_need_as` the analyzer
   did type is still validated. An
   analyzer that ignores the schema wholesale shows up as `dropped_schema=listed` on every question,
   and the marker line is emitted at WARNING in exactly that case (the same level as
   `GAP_FILL_ANALYZER_FAILED`, because v1 has gone dark while the analyzer still bills), while a
   question whose gaps all fail on legitimate grades stays at INFO, because that is the filter
   working. Any schema drop reports a v1 failure, including partial schema drift. The forecast
   proceeds using successful research, then the run exits 1 through its degradation accounting.
   A valid empty list or gaps dropped solely on legitimate grades do not count as failures.
4. **The cap last.** `GAP_FILL_MAX_GAPS` applies to the survivors, so a dropped gap never displaces
   a kept one; a survivor past the cap is dropped as `over_cap`. The analyzer is still asked for at
   most that many, so this binds only when it over-lists.

Three records make the filter measurable after the GitHub Actions logs expire. The
`GAP_FILL_V1_TRIAGE` marker (`docs/telemetry_markers.md`) carries `listed`, `kept` and a count per
reason, and is emitted for every question whose analyzer answered, so `listed=0` means the analyzer
answered and produced no parsable gaps (an unparseable reply lands there too, traced only by the
`GapFill: could not parse analyzer JSON` warning in the run log) and a dead analyzer stays
`GAP_FILL_ANALYZER_FAILED` alone. One INFO line per dropped gap
(`GapFill: dropped gap #<position> reason=<reason> same_need_as=<pointer>: <text>`) makes a run log
readable by eye. And the raw research record (`provider="gap_fill"`) carries the dropped gaps under
`dropped`, each with its analyzer `position` and `reason`, beside the kept `gaps` and their
`results`; that list is the precision check on the analyzer's grading (are the dropped gaps really
junk?) for the next redundancy audit. Since 2026-09-09 that record is written whenever the analyzer
answered, including the zero-gap and all-dropped cases (where `gaps` is empty), so a presence or
provider-mix count off the raw archive must filter on a non-empty `gaps`, with 2026-09-09 as the
era boundary; before it, a question with no searched gap wrote no record. The saving itself is
confirmed only by a paid run, which is the operator's call.

### v1 implementation notes (`research/targeted.py`)

The prose below was carried as comment blocks inside `metaculus_bot/research/targeted.py` until
2026-09-09, when the AST smell scanner's comment rules were applied to that module. Each block left
one line of why in the code plus a pointer to this section, and the entries here are in file order,
each naming the function it came from. No executable code changed in that pass.

**`_GAP_FILL_SOFT_FAIL_EXCEPTIONS` (module level).** The tuple is `(Exception,)`, and the breadth is
deliberate. It names the exceptions that trigger a soft-fail, meaning a `""` return, out of
`run_gap_fill_pass`. Failures return the surviving research and notify `on_error` once per pass,
so publishing can finish before the run exits 1. Listing specific exception classes was tractable while both stages
ran on google-genai. After the 2026-05-20 migration of the analyzer to OpenRouter and litellm it is
not, because the analyzer can now raise `litellm.APIError`, `openai.AuthenticationError`,
`anthropic.RateLimitError` and more. Catching `Exception` is what matches the stated policy, and it
keeps the catch in one place. `CancelledError` still propagates, because it inherits from
`BaseException` rather than from `Exception`.

**`extract_disagreement_crux`: the retry around the analyzer call.** The call goes through
`invoke_with_broad_retry`, a broad retry gated at 30 seconds, because `DISAGREEMENT_ANALYZER_LLM` is
configured `allowed_tries=1` in `llm_configs.py`. The retry recovers a fast blip or an empty response
while obeying the universal "no retry after 30 s" deadline rule. Its `wall_timeout` mirrors the
`CRUX_SOFT_DEADLINE` `asyncio.wait_for` that the `forecaster.py` call site (around line 697) already
applies.

**`run_targeted_search`: the wall-clock backstop, and the crux preview in the log line.** The
backstop is now owned by `invoke_with_transient_retry`, and it shares `NATIVE_SEARCH_WALL_TIMEOUT`
with the `native_search` research provider because both call the same LLM configuration through
`build_native_search_llm`. The 2026-05-20 OpenRouter whitespace-drip incident defeated litellm's
per-HTTP-request timeout, so the wall timeout is the hard cap regardless of what upstream does. The
transient-retry wrapper additionally recovers from instant aiohttp blips (litellm issue #14895) on
this `allowed_tries=1` LLM, and its elapsed gate means it never retries a slow stall. The INFO line
above the call previews the crux with `crux[:100]`, which truncates a string for display and reduces
no data; that slice carries a `HARNESS-SCAN-EXEMPT-subsampling` pragma for the scanner, held on the
flagged line by a `# fmt: skip` (see the note at the end of this section).

**`_parse_gap_list`: choosing an extractor, and the raw preview in the warning.** Fenced blocks are
preferred, through the canonical extractor in `structured_output_schema`, and the fallback is a
string-literal-aware balanced-brace scan for unfenced payloads with trailing commentary. Both helpers
live in one module so the brace scanner is fixed in one place. Invalid JSON, an empty response, or a
missing/non-list `gaps` field raises `ValueError`; only an explicit empty list means no gaps. The parse-failure warning previews the
analyzer response with `raw[:200]`, another display truncation carrying the same pragma and the same
`# fmt: skip`.

**`_run_analyzer`: `temperature=None`.** Passing `None` defers reasoning models to the provider
defaults. It is redundant on forecasting-tools 0.2.92, where the `GeneralLlm` constructor default is
already `None`, and no `top_p` is set.

**`_run_analyzer`: passing the multiple-choice ballot.** `options` is the multiple-choice ballot and
is `None` on every other question type. A "no coverage of candidate X" gap is only findable when the
analyzer knows the candidates (receipt q44952).

**`_run_analyzer`: the wall-clock backstop.** Also owned by `invoke_with_transient_retry`, and given
slight headroom over the litellm per-request timeout, 135 seconds against 120, so the cleaner
per-request error fires first when it can. That mirrors `NATIVE_SEARCH_WALL_TIMEOUT` against
`NATIVE_SEARCH_TIMEOUT`. The wrapper recovers from instant aiohttp blips (litellm issue #14895) on
this `allowed_tries=1` LLM without retrying a slow stall, which the elapsed gate prevents.

**`_resolve_single_gap`: the wall-clock backstop.** The same `NATIVE_SEARCH_WALL_TIMEOUT` wall, shared
with the `native_search` provider and the targeted search because all three build the same
`build_native_search_llm` configuration. The wall timeout is the hard cap regardless of upstream
whitespace-drip behavior (the 2026-05-20 incident), and the transient-retry wrapper again recovers
instant aiohttp blips (litellm issue #14895) on an `allowed_tries=1` LLM without retrying a slow stall
(elapsed gate).

**`run_gap_fill_pass`: the `GAP_FILL_ANALYZER_FAILED` marker.** The warning is a greppable marker
rather than plain prose because the analyzer gates the entire pass: its death silently zeroes one of
the largest research spend lines and looks identical to a question that legitimately had no gaps.
Gap-fill is not one of the orchestrator's `_run_one` providers, so it has no `ProviderResult` and no
`lost=` token to render, and a `record_provider_detail` entry under a `gap_fill` key would never be
drained, which was verified, and would just accumulate in the registry. The run-log marker is the
seam that exists for it, and its spec lives in `scripts/telemetry/markers.py`.

**`run_gap_fill_pass`: the `await asyncio.sleep(0)` on the analyzer-failed path.** A soft-fail there
is already an async no-op, so the sleep gives the scheduler a checkpoint and satisfies flake8-async's
ASYNC910 on every path. The no-gaps and all-dropped paths need none: they fall through the
`asyncio.gather` over an empty task list.

**`run_gap_fill_pass`: `return_exceptions=True` on the gather.** It captures per-gap failures so one
SDK error cannot take the whole addendum down.

**`run_gap_fill_pass`: the raw-research record.** It stores the raw per-gap search results alongside
the gaps the resolver searched (`gaps`, aligned with `results` and with the addendum's `Gap N` index)
and, since 2026-09-09, the gaps triage dropped (`dropped`, each with its analyzer position and reason);
it is written whenever the analyzer answered, empty `gaps` included (see "v1 triage" for the
presence-count consequence), and exceptions serialize to their `str` through the logger's encoder.

**The two `# fmt: skip` directives.** Both log lines carry `# fmt: skip` ahead of the
`HARNESS-SCAN-EXEMPT-subsampling` pragma, and the directive is load-bearing rather than cosmetic. The
scanner honors a pragma only on the lines the flagged expression itself spans, and without the
directive Ruff would split the `logger` call across lines and move the trailing comment onto the
closing parenthesis, at which point the finding returns and the tag reads as stale. Do not drop
either directive without moving the pragma onto the line that holds the slice.

### v2: agentic gap-fill (`research/agentic_gap_fill.py` `run_gap_fill_v2`)

A bounded agentic tool loop, living in `metaculus_bot/research/agentic/` behind the
seam `research/agentic_gap_fill.py` `run_gap_fill_v2`, run by a driver LLM
(`GAP_FILL_V2_DRIVER_MODEL` at
`GAP_FILL_V2_DRIVER_EFFORT`, both picked by the 2026-07-17 blind driver eval:
`scratch/driver_replay_2026-07-17/blind_judge_report.md`), gated by
`GAP_FILL_V2_ENABLED`. It has been on in every bot workflow since 2026-07-21T17:07Z
(`b4e9df0`), with v1 left on alongside for an overlap window.

The driver is briefed with the forecaster prompt template, privately
dry-runs a forecast to find fill/verify targets, then iterates over
four tools (`research/agentic/tools.py`): `search_news` (AskNews, through the same
rate gate as the primary provider), `search_web` (Exa direct), `fetch` (an
auto-escalating ladder: plain → local PDF extraction → headless Chromium →
`read_document`), and `read_document` (acquisition-first: the free rungs, then
`GAP_FILL_V2_READER_MODEL` via Gemini url_context). Two of those rungs are transports shared
with the Tier-1 resolution-source ladder rather than copies of it: the Chromium render
(`research/rendered_fetch.py`, called by the shared `fetch_ladder.rungs._rendered_rung`) and the
url_context read
(`research/url_context_reader.py`, with the `Google-Extended` pre-check in
`research/robots_policy.py`, moved out of `research/agentic/` when the second caller arrived).
It runs under a wall deadline
(`GAP_FILL_V2_WALL_DEADLINE`) and a tool-call budget
(`GAP_FILL_V2_MAX_TOOL_CALLS`), producing anytime output, and soft-fails to `""` at
every boundary. It appends a detached
citation-only findings artifact under `## Agentic Research Findings`, leading with a
`### ⚠ Corrections to the briefing` priority block; a ghost forecast is logged for
telemetry only (the `GHOST_FORECAST` marker) and never published. Like the other
leakage-sensitive providers,
it is benchmarking-guarded off (`is_benchmarking=True` returns `""`). See
`docs/agentic_gap_fill.md` for the full
tool loop, escalation ladder, telemetry, and design rationale.

## Diagnostics and persistence

`run_research` returns forecaster-clean text. The **provider-diagnostics block**
(which provider succeeded, char counts, latency per provider) is computed
separately (`format_provider_diagnostics_block`) and deliberately kept out of the
returned research: forecasters and the v2 driver must never see it. It reaches
three places instead: an INFO log line, the research archive (as its own field),
and the published Metaculus comment (stashed per question id, popped by the
forecaster at comment-build time via `pop_provider_diagnostics`).

A provider's `details` dict carries two conventions, and they answer different
questions. `details["sources"]` is the per-source outcome map, rendered into the
`lost=` suffix. `details["counts"]` (`provider_diagnostics._counts_suffix`) is the
second: an ordered `{name: number}` map of provider-INTERNAL quantities that are
neither a source outcome nor a failure: Gemini's `tier_tags` / `generic_tier_tags` /
`unsupported_attributions`, financial-data's `fx_identifiers_empty`, and the
resolution-source rung counts. **A zero renders nothing**, so every healthy provider's
`## Provider Diagnostics` line stays byte-identical to what it was before the map
existed, while `asdict` keeps the zero in the schema-v2 archive, which is exactly what
makes "the check ran and found none" distinguishable from "the check never ran".

When a research sink is wired, each question's research is written for backtest
replay by `ResearchPersistenceWriter` (`research/persistence.py`, at
`RESEARCH_SCHEMA_VERSION`). The record carries the assembled `research_text`, the
per-provider
`provider_results` (the authoritative outcome list), derived
`providers_attempted` / `providers_succeeded`, the `gap_fill_used` flag, and,
when they exist, the v2 agentic trace (`gap_fill_v2`), the diagnostics block, and
the raw pre-summarization AskNews articles (`asknews_raw`). Records flush to a
timestamped JSONL file. `providers_used` is retained but legacy/ambiguous; prefer
`provider_results` for any analysis.

## Production configuration

All six bot workflows
(`.github/workflows/run_bot_on_{tournament,metaculus_cup,minibench,mantic}.yaml`,
`test_bot.yaml` and `test_bot_basic.yaml`) enable the full research stack:

| Flag | Provider |
|---|---|
| `NATIVE_SEARCH_ENABLED` | OpenAI native search |
| `GEMINI_SEARCH_ENABLED` | Gemini grounded search |
| `FINANCIAL_DATA_ENABLED` | yfinance + FRED |
| `PREDICTION_MARKETS_ENABLED` | prediction-market snapshot |
| `RESOLUTION_SOURCE_ENABLED` | resolution-source fetcher |
| `TS_ANCHOR_ENABLED` | time-series anchor (text; chart off) |
| `GAP_FILL_ENABLED` | v1 targeted gap-fill |
| `GAP_FILL_V2_ENABLED` | v2 agentic gap-fill |

So in production the active research stack is: AskNews (primary, summarized) +
OpenAI native search + Gemini grounded search + financial data (when classified
financial) + prediction-market snapshot + Tier-1 resolution-source fetcher +
time-series
anchor + both gap-fill passes. Env flags, models, and timeouts live in
`metaculus_bot/constants.py`; provider models route through the shared
donated-then-personal OpenRouter fallback (`fallback_openrouter.py`), except
Gemini grounded search, which uses the personal Google key directly. The Mantic workflow
pins `DONATED_OPENROUTER_KEY_ENABLED` to false, so there every OpenRouter call is on the
personal key (`docs/operations.md` "Mantic").

All of that is subject to the question's close-derived time budget: a question on the
fast path runs the primary plus the cheap hard-capped providers only, with the slow
optional search providers dropped and BOTH gap-fill passes skipped. The resolution-source
fetcher stays in but learns about the fast path too (`resolution_source_provider(...,
fast_path=)`), and its two EXPENSIVE escalation rungs, the Chromium render and the paid
`url_context` read, decline on it before any side effect, each recording a `fast_path` skip
(`counts["fast_path_skips"]`); the cheap rungs run as they do off it. See the pipeline's
time-budget step for how the budget is granted and what it cuts.

## Cost note

The research providers hit live, paid APIs (AskNews, Exa, Perplexity, OpenRouter
credits, Google grounding, FRED). Running the bot or a backtest spends real money
and, in live modes, publishes to the platform it forecasts (Metaculus, or
competitions.mantic.com in `--mode mantic`). Do not launch a paid run without the
operator's approval; see `AGENTS.md` "Cost discipline". The unit/integration test
suite is self-contained and hits no paid APIs.
