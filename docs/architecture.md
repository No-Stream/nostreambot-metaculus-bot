# Architecture overview

This is the map a new contributor should read first. It walks through what happens
end to end when the bot forecasts one question, names the entry points, and points
you at the deeper docs for each subsystem. Read this, then dive into the specific
page you need.

The bot is a fork of the Metaculus starter template built on the `forecasting-tools`
framework. The core idea: for each question, gather research from several providers,
run an ensemble of large language models to produce independent forecasts, then
combine them into one prediction and publish it, with a comment, on Metaculus or (in
`--mode mantic`) on Mantic's Crucible competition, a fork of the Metaculus platform.

## Entry points

Three files form the startup chain:

- `main.py`: a thin shim. It re-exports `TemplateForecaster` (for anything that
  imports it) and, when run directly, calls `cli.main()`.
- `metaculus_bot/cli.py`: the command-line entry point. It parses `--mode`
  (`tournament`, `minibench`, `metaculus_cup`, `quarterly_cup`, `mantic`,
  `test_questions`) and the optional `--only-posts` post-id filter that narrows a
  tournament-shaped mode to chosen questions (the one-question smoke run; see
  `docs/operations.md` "Mantic"), builds the LLM roster dict from `llm_configs.py`, constructs a
  `TemplateForecaster` with `aggregation_strategy=CONDITIONAL_STACKING`, and runs the
  mode-specific forecast loop. Before any fetch it runs the API identity preflight
  (`api_preflight.verify_api_identity`, against the Metaculus API by default and the
  Mantic API in `--mode mantic`; it raises `ApiIdentityError` when the host does not
  answer like the platform), and in mantic mode the personal-keys-only assertion, the
  swap to the Mantic platform client (`metaculus_bot/mantic.py`) and two authenticated
  GETs, neither retried: the tournament list, which logs the `MANTIC_TOURNAMENTS` discovery
  line, then the configured tournament's own route (`GET /api/projects/tournaments/<slug>/`)
  for the forecast-permission check, which fails shut unless that route answers with a
  `user_permission` that allows forecasting; see `docs/operations.md` "Mantic". It also wires credit telemetry and decides the process exit code:
  the run exits non-zero when any degradation counter fired (`alertable_count` on
  `TemplateForecaster` sums them: dropped forecasters, questions that failed to
  publish, stacker fallbacks, research-provider and summarizer failures, gap-fill
  v2 errors, and prediction-market degradation), when the Mantic client dropped a post it
  could not parse (`mantic.get_post_drop_count`), when the Mantic slug is past its end date
  (`_check_tournament_dates`; advisory for the Metaculus tournament), or the donated
  OpenRouter key dropped below the $100 early-warning floor (`OPENROUTER_CREDIT_FLOOR_USD`,
  sized so the reminder to ask Metaculus for a top-up arrives with runway left).
  Credit-caused alerts are live again as of 2026-09-03 and are suppressed only
  inside a dated window; see "The credit-alert suppression window" in
  `docs/operations.md`. See `main` in `cli.py`.
- `metaculus_bot/forecaster.py`: the bot itself. `TemplateForecaster` subclasses the
  framework's `ForecastBot` and owns the per-question pipeline. The method to read
  first is `_research_and_make_predictions`.

Publication happens inside the framework's forecast loop, not in `cli.py`. Every
question that clears the min-forecasters guard is already on the platform (Metaculus,
or Mantic in `--mode mantic`) by the time `cli.py` decides the exit code.

### CLI startup wiring (`_configure_process`, `main`)

`_configure_process` does the process-global setup, and it runs at the runtime entry
point rather than at module import so that test imports and library consumers do not
inherit these mutations. Four things happen in it, in order.

Logging first. The root logger is configured at INFO, LiteLLM's own logger is pinned to
WARNING with propagation off, `metaculus_bot.forecaster` runs at DEBUG so a run log
carries the full per-question trace, and `openai.agents` is pinned to ERROR because it
is noisy at INFO.

Then the two client hardening patches. `apply_publish_hardening()` wraps the publish
POSTs with a timeout and a retry, bounded tighter than the upstream default, because a
single hung POST would block the whole batch (`metaculus_bot/publish_hardening.py` holds
the rationale). `apply_fetch_hardening()` wraps the question-list GET with a bounded
retry, because one transient 403, 429 or 5xx would otherwise kill the whole run
(`metaculus_bot/fetch_hardening.py`).

Then `reset_post_drop_count()`. The Mantic parse-drop counter is process-global, since
the client has no link back to the bot, and it is read into the exit arithmetic at the
end of the run. It is reset at startup rather than in `forecast_questions` because the
fetch it counts happens before those resets run.

Last the identity preflight (`metaculus_bot/api_preflight.py`), which exists because of
the DNS-parking incident: one unauthenticated check before any mode sends its token, so the
token never reaches a hijacked host. A Mantic run never contacts metaculus.com, so it does
not depend on Metaculus DNS health, and `_assert_personal_keys_only()` runs before even that
check, because the platform that donated the key is not the one being forecast (see
`docs/operations.md` "Personal keys only, and the switch fails shut").

`main` then wires the run, and four of its decisions are worth stating.

Mode selection (`_question_source`) pins `skip_previously_forecasted_questions` on for
every tournament-shaped mode, so a re-run cannot re-spend on questions already forecast.
The Metaculus cup is a good way to read the bot's performance on regularly open
questions; `mantic` is the same tournament shape over the Mantic slug, forecast through
the `ManticClient` that `main` injects; and the evergreen `test_questions` set is a good
way to read performance on a single question.

The roster dict is annotated `dict[str, Any]` deliberately. Its `"forecasters"` slot
holds a `list[GeneralLlm]` while the helper slots hold single `GeneralLlm` values, and
the parent `ForecastBot.__init__` annotates `llms` as `dict[str, str | GeneralLlm]`,
which, being invariant, cannot express the list value. `prepare_llm_config` consumes the
`"forecasters"` list at runtime.

The Mantic client is built after `_configure_process`, so the fail-shut key check and the
identity preflight have both passed before the Mantic token is even read; `None` leaves
the framework on its default Metaculus client. The two authenticated GETs that follow are
described in `docs/operations.md` "Startup checks and robustness rules".

Research persistence flushes inside the forecast `finally`. Records accumulate in memory
for the whole run, so an exception escaping `asyncio.run` (an `OSError`, the
invalid-run-mode `ValueError`, a `KeyboardInterrupt`, the SIGTERM from the 300-minute
`timeout-minutes`) would otherwise discard every question's research, and a 40-question
run that died on the last question would archive nothing. The workflows' upload step is
`if: always()`, so a crashed run's partial batch still reaches the GitHub Actions
artifact.

What `main` does after forecasting, the end-of-run breakdown line and the ordered exit
paths, is in `docs/operations.md` "The end-of-run breakdown and the exit ladder".

#### The research-archive label (`persisted_tournament_id`, `persisted_platform`)

Both functions are pure and keyed on the run mode. The tournament label is not pinned to
`TOURNAMENT_ID` because `ResearchPersistenceWriter` stamps `tournament_id` on every
record and residual analysis buckets and joins on it. A cup run labelled with the BOT
tournament's slug files cup questions inside the tournament's config eras and inside the
supply probe's per-slug rows, which is a silent data-corruption bug rather than a
cosmetic one: the label is the only thing on the record that says which competition the
question came from, since `run_mode` distinguishes the pipeline and not the object.

`mantic` is labelled with the Mantic tournament slug, and `persisted_platform` stamps the
platform (`mantic` or `metaculus`) beside it; what that platform field can and cannot
protect against is in `docs/operations.md` "How the mode works".

`test_questions` deliberately keeps `TOURNAMENT_ID`. The evergreen example set belongs to
no tournament, so no label is right; `run_mode` is what separates those records, and
re-labelling them now would make the archive's existing test-run records incomparable
with future ones for no gain.

`persisted_tournament_id` raises on an unknown mode, for the same reason
`_question_source` does: a mode added to `RunMode` without a decision here should fail
loudly at startup rather than mislabel a whole run's archive.

## The per-question pipeline

Everything below runs once per question inside `_research_and_make_predictions`,
under a shared per-question wall-clock budget (`PER_QUESTION_WALL_CLOCK_DEADLINE`,
sized to finish just inside the 60-minute Metaculus close window). Research,
forecaster fan-out, aggregation, and publish all draw from that one budget.

```
                 one question (Metaculus or Mantic)
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  1. RESEARCH  (ResearchOrchestrator.run_research)│
        │  Providers fan out in parallel:                  │
        │    - AskNews (primary, summarized to a briefing) │
        │    - OpenAI native search                        │
        │    - Gemini grounded search                      │
        │    - financial data (yfinance / FRED)            │
        │    - prediction-market snapshot                  │
        │    - resolution-source fetcher                   │
        │  Each is independently env-gated.                │
        └────────────────────────────────────────────────┘
                                 │  research bundle
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  2. GAP-FILL  (two passes, run concurrently)     │
        │    v1: analyzer finds gaps → parallel searches   │
        │    v2: agentic tool loop (bounded, driver LLM)   │
        │  Each appends its own section; both soft-fail.   │
        └────────────────────────────────────────────────┘
                                 │  enriched bundle
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  3. FORECASTER FAN-OUT                           │
        │  N forecaster LLMs run in parallel, each capped  │
        │  by FORECASTER_SOFT_DEADLINE. Type-specific      │
        │  runner per question (binary/MC/numeric/date).   │
        └────────────────────────────────────────────────┘
                                 │  N reasoned predictions
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  4. MIN-FORECASTERS GUARD                        │
        │  Fewer than MIN_FORECASTERS_TO_PUBLISH valid     │
        │  → skip this question, keep the batch going.     │
        └────────────────────────────────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  5. AGGREGATION  (CONDITIONAL_STACKING)          │
        │  Measure spread across forecasters.              │
        │  Low spread OR stacking disabled → MEDIAN.       │
        │  High spread + stacking on → crux + targeted     │
        │  search + stacker LLM rewrite.                   │
        │  Per-bin members (Mantic small grids) → MEAN.    │
        │  Mantic: floor each open tail at ≥ 5% (last).    │
        └────────────────────────────────────────────────┘
                                 │  one aggregated prediction
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  6. PUBLISHED COMMENT                            │
        │  Per-model bullets + full research + diagnostics.│
        └────────────────────────────────────────────────┘
```

### Intake: `forecast_questions`

Every entry path (`forecast_on_tournament`, `forecast_question`) funnels through
`forecast_questions` (`forecaster.py`) before the per-question pipeline starts. When
`skip_previously_forecasted_questions` is on, which `cli.py` pins for every
tournament-shaped mode, the re-spend guard runs here, and since 2026-09-09 it fails shut.
The framework derives `already_forecasted` inside a blanket except that answers False, so a
payload with no readable `my_forecasts` field (a list GET without `with_cp=true`, a Mantic
read that lost its token, an API change) would read as never forecast, and an hourly run
would re-forecast and re-publish the whole tournament.
`_drop_questions_with_unreadable_forecast_history` drops such a question before any spend,
with one `SKIP_GUARD_UNREADABLE: question=... post_id=... platform=... reason=my_forecasts_missing`
WARNING per post plus a count line; a present field with an empty history stays eligible.
The marker is registered as `skip_guard_unreadable`, and what to do when it fires is in
`docs/operations.md` "Scheduling reliability".

Three more things happen in the same chokepoint, in order. First the unsupported-type filter:
the 0.2.92 tournament fetch can return a `ConditionalQuestion`, which `_make_prediction` has
no runner for, so it is dropped here with one loud WARNING naming the dropped types rather than
as a per-question exception inside the fan-out (`DiscreteQuestion` is a `NumericQuestion` and
stays; `DateQuestion` runs on its epoch-seconds axis). The `ConditionalQuestion` branch in
`_make_prediction` that raises `NotImplementedError` is only the backstop for a caller that
reaches it without passing through this filter. Then the survivors are sorted tightest close
first, with a stable sort so questions sharing a close time keep fetch order and a missing
`close_time` sorting last (no urgency); the order decides who wins the shared research semaphore
and the per-run cap. Then the cap: with more questions than `max_questions_per_run` the
latest-closing ones are dropped and named in a `QUESTION_CAP_FORFEIT` WARNING, a registered
marker, because an unharvestable forfeit is gone at the 90-day log expiry.

### 0. Close-derived time budget

The budget is granted at intake by `metaculus_bot/time_budget.py`, before any spend: `total_s = min(PER_QUESTION_WALL_CLOCK_DEADLINE, close_time − now − PUBLISH_RESERVE_SECONDS)`, so the static 3510 s deadline is now only the UPPER bound on a question's budget (non-publishing runs, the backtests and ablations, keep exactly the static budget; `close_aware` gates on `publish_reports_to_metaculus`). Three consequences: (a) **intake skip**: a question whose budget is non-positive, or close-limited below `TIME_BUDGET_MIN_VIABLE_S`, is skipped before any research or forecaster spend (counted under `publish_skipped_closed`: latency cost us the question, however early we noticed); (b) **fast path**: below `TIME_BUDGET_FAST_PATH_THRESHOLD` (= the full pipeline's configured worst case) the slow optional search providers and BOTH gap-fill passes are dropped, and the resolution-source fetcher's two expensive escalation rungs (the Chromium render, the paid `url_context` read) decline with a `fast_path` skip while its direct fetch and cheap rungs still run, counted by the alertable `time_budget_fast_path`; (c) **research-phase deadline**: the provider phase and each gap-fill pass are bounded by `RESEARCH_PHASE_BUDGET_SHARE` of the remaining budget, cancelling stragglers (`RESEARCH_PHASE_DEADLINE` WARN; off the fast path such cuts count under the alertable `research_budget_cuts`). Every question logs a `TIME_BUDGET` marker; the loud markers (`TIME_BUDGET_FAST_PATH`, `GAP_FILL_SKIPPED_FOR_BUDGET`, `GAP_FILL_V1/V2_CUT_FOR_BUDGET`) all have telemetry-archive specs.

The budget is granted at the top of `_research_and_make_predictions`, before research starts,
because research, fan-out, aggregation and publish all draw from the one budget and research
time alone could otherwise overshoot it. The intake skip fires there too: a question that is
arithmetically unpublishable (not even an instant forecast leaves room for the prediction POST)
raises before any research or forecaster spend, with a message that names the close time so the
run log says why rather than reporting a mysterious zero-forecaster question. That is the q45085
shape, fetched 22 seconds before its close, forecast at full 3/3 strength, then rejected 405
(section 7 below); raising at intake produces the same outcome the close gate would produce at
publish time, minus a full ensemble's worth of spend. It bumps the close gate's own counter,
`publish_skipped_closed`, already alertable, so "latency cost us this question" has one home
however early the loss was noticed; `questions_failed_to_publish` remains the min-forecasters
floor's counter alone.


### 1. Research fan-out

`run_research` (`forecaster.py`) delegates to a `ResearchOrchestrator`
(`research/orchestrator.py`). It picks one primary provider by priority (AskNews in
prod, then Exa, then Perplexity, then a stub) and runs several additional providers
alongside it in parallel, each behind its own env flag. AskNews returns raw article
text and gets summarized into an analyst briefing inline; the other providers write
their own prose and pass through as-is. See [research.md](research.md) for provider
selection, gating, and the shared-vs-personal API-key routing.

### 2. Gap-fill (two passes)

After the first-pass bundle is assembled, two gap-fill passes run concurrently in one
`asyncio.gather` inside `run_research` (`orchestrator.py`), so the research phase costs `max(v1, v2)`
in wall-clock, not the sum:

- **v1** (`research/targeted.py`): an analyzer LLM reads the bundle, names up to a few
  factual gaps, then resolves each with a parallel web search. Appends a
  `## Targeted Gap-Fill (second pass)` section.
- **v2** (`research/agentic/`): a bounded agentic tool loop. A driver LLM privately
  dry-runs the forecast to find what to verify, then iterates over search/fetch/read
  tools within a wall deadline and tool-call cap. Appends an
  `## Agentic Research Findings` section, led by a corrections block.

Both are wrapped so a failure in one never zeroes the other or kills the forecast.
Both are on in prod. See [agentic_gap_fill.md](agentic_gap_fill.md) for the v2 loop,
tools, and telemetry.

The orchestrator also builds a provider-diagnostics block that is deliberately
withheld from the forecaster-facing text (so it never pollutes prompts) but is
re-attached to the published comment later. This is the "diagnostics seam": the
orchestrator's `pop_provider_diagnostics`, which `_research_and_make_predictions` in
`forecaster.py` drains once the research phase is done. `run_research` returns
forecaster-clean text, so the block never reaches a forecaster prompt, the stacker or the
gap-fill v2 driver brief; it rides down to `route_after_forecasts`, which re-appends it to
the comment-bound `research_report` strings only.

### 3. Forecaster fan-out

Each forecaster LLM runs through `_forecaster_with_soft_deadline` (`forecaster.py`),
which caps a single model at `FORECASTER_SOFT_DEADLINE` so one stuck
model can't hold the whole question. `_make_prediction` dispatches to the
type-specific runner (`forecaster_runners.py`) for binary, multiple-choice, numeric or
date questions. The N coroutines are gathered under the shared wall-clock budget by
`_gather_predictions_with_wall_clock` (`forecaster.py`), which cancels any
forecaster still pending at the deadline and counts the drop.

**Drop attribution.** The coroutines are built from `self._forecaster_llms` in order, so a
task's index names its model; that ordering contract is what lets the per-model drop telemetry
(`FORECASTER_DROPS`) name the model a cancelled or raised task belonged to, and `unknown`
appears only if the two lists desync. A forecaster that finished by raising is a dropped
ensemble member and counts as degradation, so `cli.py`'s alertable exit fires; a soft-deadline
`TimeoutError` was already counted at its raise site in `_forecaster_with_soft_deadline`, so
the gather excludes it to avoid double counting. `_record_forecaster_drop` is the single write
path for both the attributed drops list and the legacy `_forecasters_dropped_count` scalar, so
the two can never drift.

**The chart image.** When `TS_ANCHOR_CHART_ENABLED` is on, `_pull_research_chart` pops the
time-series-anchor chart rendered for this question out of the provider's per-session cache
and the base forecasters attach it as a vision message. The stacker path never receives it, so
the image reaches base models only. Off, the prod default, the read short-circuits, so a stale
entry from an earlier flag-on run is never attached.

**Inside `_make_prediction`.** After the runner returns, the reasoning is stamped with a
`Model: <slug>` prefix, which `performance_analysis.parsing` reads for per-model attribution.
Then `run_tools_for_forecaster` (`tool_runner.py`) runs the deterministic probability-math
tools over the forecaster's structured block and appends a "Computed quantities" section; it
no-ops when `PROBABILISTIC_TOOLS_ENABLED` is off, as it is in prod, or when no block was
emitted, so callers do not gate (the dormant path's history is in
[roster_history.md](roster_history.md)). Extraction telemetry (`EXTRACTION_RUNG`) is emitted
inside the value-extraction ladder the runners call, not by `_make_prediction`. Each branch
returns a `ReasonedPrediction[T]` for its own `T` while the signature promises
`ReasonedPrediction[PredictionTypes]`, so the return carries a `type: ignore`, the same pattern
the framework uses.

**The date path.** A `DateQuestion` stays a `DateQuestion` end to end, so the framework builds
a `DateReport`, telemetry says `qtype=date` and persistence sees a date, while the numeric math
runs on an adapter: `numeric/date_axis.py` (`as_epoch_question`, `numeric_view`) views the
question as a `NumericQuestion` on the epoch-seconds axis, which is exactly how
forecasting-tools and the Metaculus backend represent a date question (`.timestamp()` on the
bounds, the same CDF validation rules). `_run_forecast_on_date` (`forecaster.py`) calls
`run_date_forecast` (`forecaster_runners.py`), a thin wrapper of the numeric runner: the
`date_prompt` (`prompts.py`) asks for ISO dates and names the bin granularity, `DateStructured`
(`structured_output_schema.py`) carries the declared percentiles as datetimes, the extraction
ladder converts them to epoch seconds (`parse_forecast_date`: UTC always, a date-only value is noon
UTC of that day so it lands inside that day's right-closed bin), and the same guarded PCHIP
build, CDF-space aggregation (`numeric_view` at every routing site) and publish path follow
with `is_date` set so the comment renders dates through the framework formatter. Nominal bounds
are read from the API's `scaling` block and never derived for a date question. The analysis
side stays date-free by decision, each at an explicit seam: the backtest
(`backtest/question_prep.py`), the ablation harness (`ablation/run_pdf.py`), the residual
dataset (`performance_analysis/collector.py`) and the ghost scorer (`scripts/score_ghosts.py`);
see `docs/performance_analysis.md` "Date questions are excluded from the dataset". Detail:
[numeric_pipeline.md](numeric_pipeline.md) and `docs/operations.md` "Date questions".

**The per-bin path.** A Mantic numeric or date question whose published bins are its outcome space
and number 31 or fewer (`elicit_per_bin`, `numeric/config.py`: `PMF_ELICITATION_PLATFORMS` is
Mantic-only and `PMF_ELICITATION_MAX_BINS` is 31) is elicited per bin rather than as percentiles.
`run_numeric_forecast` and `run_date_forecast` (`forecaster_runners.py`) branch into
`_run_pmf_forecast` on that gate: `pmf_prompt` (`prompts.py`) asks for one probability per bin
label (`numeric/pmf_grid.py`, plus `below_range` / `above_range` where a bound is open), the
`pmf` ladder in `value_extraction.py` reads the block, and `numeric/pmf_cdf.py` normalizes the
declaration, blends it to the server's per-cell floors and assembles the CDF, bypassing
`sanitize_percentiles`, the PCHIP repair tiers, the discrete vote and the unit-mismatch guard
(each bypass reasoned in [numeric_pipeline.md](numeric_pipeline.md) "Per-bin elicitation"). The
`MEMBER_FORECAST` line carries `elicitation=pmf` with the `N + 2` PMF as `raw` and `published`.
Per-bin members are aggregated by the pointwise MEAN of their CDFs, the linear opinion pool,
rather than the median (section 5 below); the 201-point Metaculus continuous path and every
Metaculus question are untouched because the gate is false for them by construction.

The ensemble is a handful of forecaster LLMs, one per vendor. The exact roster
rotates often, so **read `metaculus_bot/llm_configs.py` for the current list** rather
than trusting any names written here. Support models (summarizer, parser, stacker,
disagreement analyzer) live in the same file. The standing design rule, each support
model's role, and the dated history of every roster change with the merge that made it
live are in [roster_history.md](roster_history.md).

Each forecaster emits its answer inside a fenced ```json STRUCTURED FORECAST block,
which is parsed by a deterministic extraction ladder (`value_extraction.py`). Numeric
questions produce the canonical percentile set (`STANDARD_PERCENTILES` in
`numeric/config.py`), turned into a PCHIP CDF on the `PCHIP_CDF_POINTS` grid. See
[numeric_pipeline.md](numeric_pipeline.md) for the percentile-to-CDF machinery and its
bound/step constraints.

### 4. Min-forecasters guard

If fewer than `MIN_FORECASTERS_TO_PUBLISH` (`constants.py`) forecasters
returned a valid prediction, the ensemble is too degraded to trust. The question is
skipped and a counter bumps for end-of-run alerting, but the rest of the batch and all
other publications continue. The guard lives in `_research_and_make_predictions`
(`forecaster.py`); raising there skips this question alone, and `cli.py` wires the
counter into the exit status.

The threshold is the `min_forecasters_to_publish` constructor argument, defaulting to the
module constant for production. Tests and the benchmark harnesses override it, because a
2-model ensemble would otherwise always fail the guard, and a threshold above the roster
width logs a warning at construction since every question would then fail. The framework
has had its own success-rate gate since 0.2.92: `_handle_errors_in__run_individual_question`
raises when `len(predictions) < expected_total_predictions * required_successful_predictions`,
and `expected_total_predictions` equals the `predictions_per_research_report` that
`prepare_llm_config` sets to the roster width. At its 0.5 default that gate would reject a
single-survivor publish on any roster wider than two, contradicting
`MIN_FORECASTERS_TO_PUBLISH`, so the constructor pins `required_successful_predictions=0.0`
and this guard stays the sole arbiter of whether a degraded ensemble publishes.

When the threshold is 1, a lone survivor publishes: the median of one forecast is
that forecast. Because the spread metrics in `spread_metrics.py` require at least two
predictions and raise otherwise, `route_after_forecasts` (`stacking_route.py`)
short-circuits the n == 1 case before spread computation and stacking and hands the
single prediction straight to the aggregator. Exception-driven drops still bump the
degradation counters, so a run thinned to one model reddens CI rather than silently
withholding the question.

#### Survivor and extreme-call telemetry

Past the guard, every question logs `FORECASTERS_SURVIVED: question=... survived=n/N models=...`
at INFO, the positive counterpart to the per-run `FORECASTER_DROPS` marker and the only place a
run log states the survivor count. It is load-bearing because the floor is low: a degraded
publish exits zero, the failure-path "Only n/N forecasters succeeded" line never fires, and the
comment-side `FORECASTERS_USED` marker never reaches stdout, so without this line a thinned
ensemble reads identically to a full one. `models=` names the survivors (read off each
prediction's own `Model:` prefix, not the configured roster) so survivors can be diffed against
drops from the log alone. Harvested into the telemetry archive as `forecasters_survived`
(`scripts/telemetry/markers.py`). Before this line existed, an operator asking "did every
forecaster survive?" had to count `EXTRACTION_RUNG` lines and dedupe model slugs to infer it;
the line is emitted unconditionally so the healthy case is stated rather than implied by the
absence of a warning. The roster is deliberately not the source of the names: it lists the
CONFIGURED models and the survivors are a subset, so reporting it would relabel a degraded run
as full.

The survivor names are derived once, positionally, from each prediction's `Model:` prefix and
reused by the `EXTREME_CALL` block, so the two lines stay joinable on the model field; reading
the prefix twice would let a future change to one reading drift from the other. The list keeps
a per-prediction `None` (rendered `unknown` on the extreme-call line) while the survivor line
drops and sorts it.

Immediately after that line, a BINARY question also logs one
`EXTREME_CALL: question=... model=... p=... side=low|high lone=... survivors=...` INFO line per
surviving member whose probability sat at or past an edge of the extreme band
(`format_extreme_call_markers`, `metaculus_bot/extreme_call.py`; band `EXTREME_CALL_LOW` /
`EXTREME_CALL_HIGH` in `constants.py`, inclusive at both edges). It is pure measurement: the
module reads probabilities and returns strings, and nothing clamps or gates on it (the
thin-publish floor in section 5 aliases the same two constants but is a separate rule keyed on
the survivor count). A member inside the band leaves no line, so `FORECASTERS_SURVIVED` in the
same run log is the denominator for any rate. `lone=true` means no other survivor was extreme on
the same side. Binary only, and `lone` is vacuous at `survivors=1`, which is why the survivor
count rides the same line. Harvested as `extreme_call`. The measured lone-versus-accompanied hit
rates, and why these counts must never be pooled with the 2026-08-31 memo's, are in
`docs/performance_analysis.md` "Receipts behind the survivor-conditional markers".

It is emitted at this point in `_research_and_make_predictions` because it is the one place
where the surviving predictions and their own model prefixes are both in hand before anything
has aggregated them; downstream, `route_after_forecasts` collapses the set to a published value
and the per-member calls are recoverable only from parsed comments. The `cast(float, ...)` on
each member's value is exact rather than defensive: the `isinstance(question, BinaryQuestion)`
check is the same predicate `_make_prediction` dispatches on, so every question that reaches
the block was routed to `run_binary_forecast`, which returns `ReasonedPrediction[float]` (a
conditional question raises there and never yields a prediction; numeric and date members are
`NumericDistribution`s and never enter the branch). An `isinstance` filter over the values
would silently drop a member instead.


### 5. Aggregation: CONDITIONAL_STACKING

The default strategy is `CONDITIONAL_STACKING` (set in `cli.py`'s `main`). Conceptually:

- Compute the spread across the N forecasts (`spread_metrics.compute_spread`).
- **Low spread**: return the MEDIAN of the raw per-model predictions. The one exception is a
  question elicited per bin (the Mantic coarse grids of section 3): its members are combined
  by the pointwise MEAN of their CDFs (`_numeric_combine_strategy`, `aggregation_pipeline.py`),
  because the pointwise median of three sharp per-bin members is the middle member's CDF
  outright, the platform floor on the bins the other two believed, and the mean keeps every
  believed bin at least a third of its mass. Percentile members keep the MEDIAN on both
  platforms. On every numeric, discrete or date question the `NUMERIC_AGGREGATE` marker's
  `method=` records which rule ran: `mean` (pooled per-bin members), `median`, `stacked` or
  `single`; `unrecorded` means the aggregation never recorded a method and is a bug signal.
- **The Mantic tail floor, last of all**: whichever path produced the aggregate, a Mantic
  numeric, discrete or date distribution then passes through `floor_published_tails`
  (`numeric/out_of_range_floor.py`) in `TemplateForecaster._aggregate_predictions`
  (`forecaster.py`), the one seam every aggregation path (stacked, base-combine, median
  fallback, single survivor, simple) returns through, so the publish gate, the comment and the
  marker all read the floored CDF, and the published tails and the combine method are logged
  once there (`member_forecast.py`). It raises each OPEN tail to at least
  `MANTIC_OUT_OF_RANGE_TAIL_FLOOR` (`constants.py`, 0.05) as far as the other tail leaves room,
  never reduces a tail, and leaves closed bounds and every Metaculus aggregate untouched. Mantic
  scores an out-of-range resolution against a fixed 0.05 reference, so the structural 1% tail the
  percentile path publishes would score −80.5 there. The marker carries both the raw and the
  floored tails (`oor_*_raw`, `tail_floor`). Rule, cap arithmetic and receipts:
  [numeric_pipeline.md](numeric_pipeline.md) "Step 11: the Mantic out-of-range tail floor".
- **High spread**: extract the disagreement crux with the analyzer LLM (under
  `CRUX_SOFT_DEADLINE`), run a targeted search on it (OpenAI native search on the same
  `NATIVE_SEARCH_*` model, effort, verbosity and timeout settings the native-search provider
  uses), then hand the full base-model reasonings plus that research to a stacker LLM that
  rewrites the forecast (`stacking.run_stacking_binary` / `_mc` / `_numeric`). The fallback
  ladder is primary `STACKER_LLM` under `STACKER_SOFT_DEADLINE` → `STACKER_FALLBACK_LLM` under
  `STACKER_FALLBACK_SOFT_DEADLINE` → MEDIAN, driven by `stack_predictions`
  (`aggregation_pipeline.py`).

Spread thresholds live in `constants.py`, one per question type:
`CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD` (a probability range),
`CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD` (a max per-option spread), and
`CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD` (a normalized percentile spread).

**Stacking is disabled in production.** All six bot workflow YAMLs (`run_bot_on_tournament`,
`run_bot_on_minibench`, `run_bot_on_metaculus_cup`, `run_bot_on_mantic`, `test_bot`,
`test_bot_basic`) set `BINARY_STACKING_ENABLED`, `MC_STACKING_ENABLED`, and
`NUMERIC_STACKING_ENABLED` to `false`, so even when spread exceeds the threshold, the per-type gate in
`route_after_forecasts` (`stacking_route.py`) bypasses the stacker and forces the
MEDIAN path. In effect, **prod runs MEDIAN of the raw forecasts.** The stacker chain
stays fully wired and is exercised in
backtests and ablation runs. `AggregationPipeline` owns the aggregation configuration,
per-question metadata, and counters. Its explicit operations are `base_combine`,
`stack_predictions`, and `simple_combine`. The framework-required
`TemplateForecaster._aggregate_predictions` hook selects the appropriate operation;
internal callers state that choice directly. Routing and stacked-result finalization
live in `stacking_route.py`, which receives the pipeline rather than the whole bot.
The conditional-stacking path runs the combined result
through a Platt-calibration hook (`_apply_platt_calibration` in
`aggregation_pipeline.py`), but that hook is gated by `PLATT_CALIBRATION_ENABLED`,
which is unset in every workflow, so in prod `apply_platt_calibration`
(`post_processing.py`) is a passthrough.

Multiple research reports for one question share the pipeline's existing
question-id maps. These remain separate because sibling reports can leave a skip
reason that must survive a later failed stack attempt. Stacked-result finalization
consumes meta reasoning; comment construction consumes outcome and skip metadata
after the parent comment builder returns successfully. The expected-combine set
is consumed by the framework's final combine. Moving ownership does not change
those read, write, or consumption points.

`tests/test_aggregation_lifecycle_e2e.py` covers the real framework lifecycle with
failed sibling reports, raw versus pre-stacked singles, and stacker fallback.
`tests/test_aggregation_report_e2e.py` drives the public entrypoint through numeric
and MC report construction with two reports and four model results. These tests
replace research and model calls while retaining fan-out, routing, aggregation,
and comment construction. Focused pipeline tests also pin cancellation and
validation before state consumption. `tests/test_aggregation_failure_lifecycle.py`
checks real timeout expiry and the state retained when the parent comment builder
fails after aggregation. Both cases passed unchanged against the implementation
before the ownership refactor.

#### The thin-publish floor

One survivor-conditional rule sits on top: when exactly ONE forecaster survived a BINARY
question, the published probability is clamped into
`[THIN_PUBLISH_BINARY_FLOOR, THIN_PUBLISH_BINARY_CEIL]` (`constants.py`, 0.05/0.95, defined by
aliasing `EXTREME_CALL_LOW` / `EXTREME_CALL_HIGH` so the extreme band has one definition, and
narrower than the per-model `[BINARY_PROB_MIN, BINARY_PROB_MAX]` = [0.02, 0.98] clamp the member
already passed) by `apply_thin_publish_floor` in `AggregationPipeline.base_combine`. It is
triggered by the `single_forecaster` skip reason rather than the prediction count (a fired
stacker's lone output shares that branch and is never floored); the per-model summary bullet
keeps the raw value; an actual move logs
`THIN_PUBLISH_FLOOR: question=... raw=... clamped=... survivors=1` (harvested as
`thin_publish_floor`); and a multi-member median is never floored, because median-of-1 has no
variance reduction, which is the whole justification. The motivating miss and the sweep that
priced the rule: `docs/performance_analysis.md` "Receipts behind the survivor-conditional
markers".

#### An unmeasurable spread is its own case

An UNMEASURABLE spread (non-positive normalizing denominator) reports `inf` and logs `SPREAD_UNDEFINED`, and `route_after_forecasts` treats it as its own case: MEDIAN with skip reason `spread_undefined`, spending no crux extraction / targeted search / stacker call on a question where nothing was measured. It used to report `0.0`, which read as an affirmative "the models agree" and published the marker `spread_below_threshold`: a measurement failure disguised as agreement. Latent in prod (the per-type gates are off) but live in backtests and ablation.


### 6. Published comment

The framework assembles the comment: per-model forecast bullets (annotated with model
names so per-model attribution survives comment trimming), the full research bundle,
the targeted-research section if stacking fired, and the provider-diagnostics block
re-attached via the seam. These published comments are also the durable per-model
record the performance-analysis tooling later parses.

The `summary_report` handed to the framework is a one-line stub, not the research corpus.
The framework embeds `summary_report` under "### Research Summary" and `research_report`
under "# RESEARCH", so setting both to the research text duplicated it and bloated the comment
past the character limit. The "### Research Summary" heading is emitted regardless of body, so
the trim anchor and the parser markers survive.

#### Comment trimming

`metaculus_bot/comment/trimming.py` enforces the platform's character limits: `trim_section` for
one section, `trim_comment` for the assembled whole. The budgets come from `constants.py`
(`SUMMARY_SECTION_CHAR_LIMIT`, `RESEARCH_SECTION_CHAR_LIMIT`, `FORECASTS_SECTION_CHAR_LIMIT`,
`COMMENT_CHAR_LIMIT`), and `_section_budget` maps a section name to its own budget so the
parser-critical FORECASTS section is not starved by a uniform cap. The module-private
`_COMMENT_HEAD_BUDGET` (10,000 chars) is sized to seat the summary with a safety margin while
leaving most of the budget to the tail, which is where the `STACKED=<bool>` marker and as many R1
rationales as fit live.

Everything the trim protects is something `performance_analysis` later parses back out of the
published comment: each rationale's `Model: openrouter/<provider>/<name>` line (keyed on by
`_R1_MODEL_RE` and `_REASONING_MODEL_PREFIX_RE` in `performance_analysis/comment_sections.py`), the
fenced json STRUCTURED FORECAST block that closes each rationale, the `*Forecaster N*:` bullets in
the summary, the trailing `STACKED` marker, and the provider-diagnostics `lost=summarizer:error(...)`
token naming which source degraded. A naive header-and-tail trim ate Forecaster 1's `Model:` line in
29 of 29 measured July 2026 trims, which is why the FORECASTS path shrinks each rationale block from
within (`_trim_rationales_within_blocks`, `_trim_block`, `_trim_single_body`) instead of keeping one
header and one long tail.

The stacker-combined shape is the sharpest case. When stacking fires,
`combine_stacker_and_base_reasoning` folds the stacker's meta-analysis and every base model's
reasoning into a single `## R1: Forecaster 1 Reasoning` block, each part ending with its own json
forecast block. Trimming that as one body keeps only the last json block and orphans it from its
`Model:` line, at which point `performance_analysis.parsing._split_stacker_combined_body`
re-attributes those forecast values to the last SURVIVING base model, which is silent
misattribution rather than plain loss. `_trim_stacker_combined_block` therefore splits on the same
delimiter and `Model:` regex the parser uses, water-fills the budget across the stacker portion and
each base sub-block with `_allocate_block_budgets`, trims each from within, and re-emits the
delimiter verbatim so the parser's stacker-body detection still fires.

Two smaller pins. The summarizer soft-fail banner is kept ahead of the tail by
`_pin_summarizer_banner`, because a soft-fail is itself what pushes a research bundle over budget
(raw articles run longer than the briefing), so a plain header-and-tail trim would have dropped the
disclosure precisely on the questions where it fired. That banner is located by search rather than
assumed to lead the body, since the orchestrator prepends it to the AskNews provider's own text and
it therefore sits after that provider's `## News Articles (AskNews)` header. And the
provider-diagnostics block is appended to the research body, so keeping only the front would drop
it; being the one bounded, non-narrative part of that body, it is pinned after the kept front and
dropped only when it cannot fit at all.

#### Ensemble-size disclosure (`FORECASTERS_USED`)

The comment trailer carries `FORECASTERS_USED=<used>/<configured>`: forecasters that
CONTRIBUTED (equal to the number of per-model summary bullets) out of those CONFIGURED. It
makes a degraded publish self-describing in the durable comment record, so residual analysis
can tell a dropped model from a roster change (a missing bullet is otherwise ambiguous); the
cause of any drop stays in run-log telemetry.

The count is recorded at the fan-out, in `_research_and_make_predictions` right after the
min-forecasters guard, the one point every downstream branch shares, so the stacked,
single-forecaster and base-combine paths agree on "forecasters that fed the published value"
by construction. It cannot be recovered later: route finalization collapses `predictions` to a
single aggregate, so counting the published collection reports 1 no matter how many forecasters
contributed, and on a stacked publish it would read 1/N on a healthy N-model run. It is
accumulated rather than assigned so `research_reports_per_question > 1` keeps the count equal
to the number of per-model bullets across all report sections. `_create_unified_explanation`
drains it per question; the collection sum remains the fallback only for the delegated path
(no forecaster roster configured, so the parent implementation ran the fan-out). On that same
path the roster is empty, so the configured count falls back to
`predictions_per_research_report`, the width the parent fans out over; `llm_setup` keeps the
two equal whenever a roster is set, so they differ only on delegation, and the parent asserts
the value is positive, so the denominator can never be 0.

### 7. Publish, behind a close-time gate

The gate lives in `publish_gate.py`, wired as layer 4 of `publish_hardening.py`'s patch of ft's `publish_report_to_metaculus`. Immediately before the POSTs, the question's `close_time` is compared to now; if the window has passed, or the question's cached `state` is already CLOSED/RESOLVED, the whole publish is SKIPPED (prediction and comment together, since a comment for a forecast the platform never accepted would seed `performance_analysis` with a forecast that doesn't exist there). The skip emits one `PUBLISH_SKIPPED_CLOSED: question=... reason=... close_time=... now=... overdue_s=... state=...` WARN, bumps `publish_skipped_closed` on the degradation line, and counts as ALERTABLE, because a skip means latency cost us the question, which is exactly what should redden CI. The run continues with every other question. Deliberately **no safety margin**: ft's publish body sleeps 3.5-4.5s twice, so a question with seconds left can still 405 after passing the gate, but widening it would start skipping publishes that would have landed, and a forfeited question costs far more than a rejected POST. That residual 405 now costs ONE attempt, not two: `publish_hardening` no longer retries a 4xx outside {408, 429}, since a second identical POST cannot fix a 405/401/400. Shipped 2026-08-25 as the root-cause fix for q45085 (2026-08-03: forecast at full 3/3 strength, submitted 12:05 against a 12:00 close, `405 "already closed to forecasting"`, whose crash also took out that run's end-of-run alertable summary).

### Run-level counters and the end-of-run summary

One bot instance is one run, so the counters `cli.py` reads to decide the exit status are
per-run totals, zeroed in `_init_alerting_counters` at construction. Beside the dropped
forecaster scalar sits the attributed drops list (which model, which question, why), kept in
lockstep with it by `_record_forecaster_drop`. `_time_budget_fast_path_count` counts questions
whose close time was too near for the full pipeline's worst case, so the optional research
stages were dropped (section 0): a fast-path publish is a degraded publish, the forecasters saw
a thinner research bundle, and it reddens CI for the same reason a thinned ensemble does, as a
symptom of upstream latency, which is what the operator wants paged. The question still
publishes. `_contributing_forecasters` is the per-question contributor map behind
`FORECASTERS_USED` (section 6).

`forecast_questions` resets the per-run state the bot does not own on the same cadence: the
PCHIP build statistics, the research orchestrator's degradation counters, and the two
module-scoped counters, the publish-hardening wrapper's retry-exhaustion count and the close
gate's skip count, which live at module scope because those wrappers have no handle back to the
bot.

After the framework fan-out returns, the run ends with three lines. The degradation summary
(`Degradation counters: ...`, harvested as `degradation_counters`) is the line that decides CI
colour: any non-zero counter means something got dropped, the stacker fell back or a research
provider failed, all states where `cli.py` should exit non-zero so we get paged, but every
publishable question has already been published by then. `FORECASTER_DROPS` is the per-model
attribution for the dropped scalar (which model failed, how often and why, plus a
`SYSTEMATIC_FORECASTER_FAILURE` WARNING when one model failed across several questions this
run). `PROVIDER_DEGRADATION` is the provider counterpart (which venue or signal degraded, on
how many rows or questions, and what to do about it), emitted even at zero so "no provider
degraded" is a recorded fact rather than an absent line. The exit ladder those lines feed is in
`docs/operations.md` "The end-of-run breakdown and the exit ladder".


## Framework integration (`forecasting-tools`)

What the bot takes from the framework, and the one place it overrides it:

- `GeneralLlm` for model interfaces (a wrapper around litellm).
- `MetaculusApi` for platform integration.
- Question types: `BinaryQuestion`, `NumericQuestion`, `MultipleChoiceQuestion`, `DateQuestion`
  (forecast on its epoch-seconds view, `numeric/date_axis.py`; `ConditionalQuestion` stays
  unsupported).
- Prediction types: `ReasonedPrediction`, `BinaryPrediction`, and friends.
- Research helpers: `AskNewsSearcher`, `SmartSearcher`.
- Numeric: `NumericDistribution`, `Percentile`. We subclass `NumericDistribution` as
  `PchipNumericDistribution` (`numeric/pchip_processing.py`) to override `get_cdf()` (the
  method ft 0.2.92's publish and aggregate paths call, with `.cdf` a deprecated property that
  delegates to it) so it returns our pre-computed `PCHIP_CDF_POINTS`-point PCHIP CDF. The
  framework's own CDF builder is used only on the fallback path.

### Harness seams on the constructor

Three constructor details exist for the harnesses and the framework rather than for the
per-question pipeline. `self.name` is declared on the instance because the benchmark and
backtest harnesses tag each bot with a display name (`benchmark/bot_factory.py` sets
`bot.name = spec["name"]`, and `backtest.py` filters on it); declaring it makes the attribute
statically known instead of needing scattered type ignores. `metaculus_client` is the client
the framework uses for the tournament fetch and every publish POST: `None` keeps its default
Metaculus client, and mantic mode injects a `ManticClient` (see "Entry points" above).
`required_successful_predictions=0.0` disables the framework's own success-rate gate so the
min-forecasters guard is the sole arbiter of a degraded publish (section 4 above).

### The offline end-to-end test (`tests/test_offline_e2e_forecast.py`)

The breaking-dependency tripwire. The litellm 1.92 crash (`acompletion(tools=...)` eagerly
imports a proxy MCP handler needing `fastapi`, which we do not install) fired only when the real
call EXECUTED with `tools=`, so a plain `import litellm` smoke test would not have caught it, and
the agentic gap-fill v2 loop, the only `tools=` caller, soft-failed to `""` so nothing went red in
CI. The test therefore drives the REAL code paths of every external dependency on the forecast
critical path (research, forecaster fan-out, aggregation, gap-fill v1 and v2) and stubs ONLY the
outermost network boundary, the socket-opening client call. If a dep upgrade breaks an import, a
transform or a call path anywhere in that stack, this test goes red.

**The two seams.** The LLM seam is a routing wrapper rather than a global mock, because the
pipeline makes many heterogeneous LLM calls that each need a DIFFERENT valid canned response
(binary, numeric and MC forecaster blocks, summarizer prose, gap-fill v1 gap JSON, native-search
prose, the agentic v2 driver's tool calls, the parser salvage), and a single
`litellm.mock_response` cannot satisfy all of them. Both `acompletion` chokepoints are patched
with a router that inspects the outgoing `model` and `messages` (the system-prompt text identifies
the call type), picks the matching canned text and forwards to the REAL
`litellm.acompletion(**kwargs, mock_response=<routed>)`. Forwarding to real litellm is
load-bearing: it executes all real litellm import, transform and tools-path code, which is what
catches the fastapi class of bug, while short-circuiting only the network. The v2 driver path adds
`mock_tool_calls` so the loop gets a real tool-call-shaped response. The provider seam stubs each
research provider's external client (AskNews SDK, google-genai `Client`, aiohttp session, Exa
client) at its lowest boundary, so our formatting and parsing code runs for real but no socket
opens; the autouse network-egress guard in `tests/conftest.py` is the backstop, turning a missed
stub into a clear `RuntimeError` rather than real spend.

**Coverage.** Binary, numeric and MC each run the full `forecast_questions` to
`_research_and_make_predictions` pipeline offline and produce a published-shape `ForecastReport`.
Real code executed per question: the whole research fan-out (AskNews plus summarizer, native
search, Gemini grounded, the financial-data classifier, the prediction-market snapshot across four
platforms, the resolution-source fetch), gap-fill v1 (analyzer plus parallel resolvers), gap-fill
v2 (the agentic tool loop, driving REAL `litellm.acompletion` with `tools=`), the forecaster
fan-out through the value-extraction ladder at rung=block, and CDF/MC post-processing plus
aggregation. Stacking is prod-disabled (the three `*_STACKING_ENABLED` flags default off and are
not set here), so the median/skipped aggregation path runs, which is the production default.
Partially exercised: the stacker LLM itself (crux, targeted search, stacker) is not driven, because
prod runs with stacking disabled, and `tests/test_conditional_stacking.py` covers that mechanism;
rendered-fetch (headless Chromium) and `read_document` (Gemini `url_context`) inside the agentic
loop fire only if the driver requests them, and the scripted driver concludes without them, so
those rungs are unit-tested in `tests/test_agentic_tools.py` instead.

**Why each fixture is shaped the way it is.** The test keeps one line of why apiece; the receipts
are here.

- `_RESOLUTION_URL`: a fetchable URL in the resolution criteria is what exercises the
  resolution-source provider (extract, fetch, trafilatura extract). The example.com and
  example.gov names are RFC-2606 reserved, the aiohttp session is stubbed so no socket opens, and
  the provider's SSRF-preflight `getaddrinfo` is patched to a public IP.
- The canned forecaster and stacker blocks: kept byte-for-byte in sync with the STRUCTURED FORECAST
  schemas (`metaculus_bot/structured_output_schema.py`) so the binary, MC and numeric extractors
  all land rung=block. Mirrors `tests/pipeline_test_helpers.py`'s canned reasonings.
- `_CANNED_QUERY_AUTHOR`: the prediction-market query author wants
  `{"synonyms": [...], "framings": [...]}`. Anything else, prose or a shape `parse_query_author`
  rejects, makes the stage report a lost source, which bumps the market provider's source-loss
  counter and fails `_assert_pipeline_ran`'s `alertable_count == 0`.
- `_CANNED_GAP_ANALYZER`: graded to pass the gap-fill v1 triage (`answerable_now` true,
  `already_in_first_pass` false, `same_need_as` null) so the resolver path runs. Drop either boolean
  grade and `triage_gaps` discards the gap as `schema` before any resolver call.
- `_REQUIRED_OK_PROVIDERS`: verified empirically (all three question types, INFO logs) to be
  identical, with asknews, native_search, gemini_search and resolution_source all landing `ok`
  while financial_data legitimately returns `empty` on a non-financial question, so financial_data
  is not asserted `ok`. prediction_market does render rows off the stubbed off-topic payloads but
  is left out of the set anyway: whether it renders is the canned RANKING's call, not a statement
  about the provider's health, and pinning it would make the set assert a test fixture. A dep break
  that errors any required provider is swallowed into `status="errored"` by the orchestrator, so
  this set is the direct catch for the non-litellm dependency class (google-genai, asknews,
  aiohttp).
- The off-topic market payloads: EVERY venue must return a POPULATED payload, and each must carry
  the liquidity fields `provider_health` declares for it. An EMPTY catalogue from a SUCCESSFUL
  fetch is a degradation in its own right (a dead response parser, or a silently emptied index),
  and `provider_health`'s `catalogue_empty` signal alerts on it by design, so an empty stub both
  trips that alert and skips the pool assembly the suite exists to exercise; a venue-complete
  payload set is also what a healthy prod run looks like, so the pipeline under test is the one
  prod runs rather than a degraded corner of it. The liquidity fields are equally load-bearing:
  `market_field_contract` fires when a declared field is absent from 100% of a venue's POOL rows,
  which is every row a populated payload produces whether or not the ranker keeps any of them, and
  every real open Kalshi market carries `volume_fp` and `open_interest_fp` (1,504 of 1,504
  measured), so a stub without them describes a payload that does not exist.
- The Manifold search listing carries no description, which is why the enrichment fan-out exists,
  and the detail record is where the rules text comes from. Its two endpoints are routed in order,
  detail path first, because a substring test for "manifold" alone would serve the detail request
  the search listing's array and leave every candidate title-only.

**What `_assert_pipeline_ran` pins.**

- `alertable_count == 0`: the sum of every degradation counter (forecasters dropped, publish
  failures, stacker fallbacks, research-provider failures, gap-fill v1 and v2 errors). The orchestrator
  SWALLOWS provider exceptions into `status="errored"` plus a counter bump rather than re-raising,
  so a broken provider dep would otherwise pass silently. This is the tripwire for it.
- `EXTRACTION_RUNG rung=block`: the forecaster's canned block parsed at rung 1, so the
  value-extraction ladder ran for real over real model output.
- `GAP_FILL_V2 ... error=None`: the agentic v2 loop executed and did not crash, which directly
  asserts the fastapi class of bug is absent, since a dead-on-arrival import error would stamp
  `error=<repr>` on every question. `tool_calls` must be at least 1 as well, because a crash-free
  marker with `tool_calls=0` passes the `error=None` check even though the driver never issued a
  tool call, so a driver that stopped sending tools would read as a healthy run. The scripted
  router sends `set_research_plan` then `conclude`, so `tool_calls` is at least 2 here, and the
  `(?<!dup_)` in the regex keeps the match off the sibling `dup_tool_calls=` field.
- `GAP_FILL_V1_TRIAGE listed=1 kept=1` plus the resolved `### Gap 1` section in the research
  bundle: this drives real analyzer JSON through `_parse_gap_list`, `triage_gaps` and the resolver.
  A v1 failure would leave the first-pass bundle publishable but increment the separate
  `gap_fill_v1_errors` counter, so the `alertable_count == 0` assertion catches it.
- Provider diagnostics: each required provider reports `ok`, meaning its real formatting code
  produced non-empty text, and no provider reports `errored`, the direct catch for a swallowed
  provider-dep break that the orchestrator turns into `status="errored"` instead of re-raising.
- `FORECASTERS_SURVIVED survived=N/N` with a non-empty `models=`: the survivor count stated
  positively in the log. Before the marker existed, the only line naming a survivor count was on
  the failure path and fired only below `MIN_FORECASTERS_TO_PUBLISH`, which is 1, so a 1-of-3
  publish exited zero and read identically to a healthy 3-of-3, and the count reached only the
  published Metaculus comment, never stdout, so an operator reading a run log had to count
  `EXTRACTION_RUNG` lines and dedupe model slugs to infer it. `models=` must name the survivors so
  a reader can diff them against `FORECASTER_DROPS` without cross-referencing the comment.

## The shared fetch ladder

Two code paths fetch web pages: the resolution-source fetcher, which reads the URLs a question
names as its grading source, and the gap-fill v2 agentic loop, which reads URLs a driver model
picks itself. The ladder they run lives in `metaculus_bot/research/fetch_ladder/`, and each caller
is an adapter over it. Both callers use `fetch_url`; the package's modules depend on
each other in one direction:
`guard.py` (the SSRF preflight, the vetted DNS resolve, the aiohttp session, the per-host
politeness gate), then `digest.py` and `verdict.py` (what a caller makes of a body it read), then
`policy.py` and `context.py`, then `run_cache.py` (complete reusable reads), then
`classify.py` (the one classification path for a body), then
`direct_fetch.py` (the bounded redirect loop and the single hop), then `rungs.py` (the seven
escalation rungs), then `ladder.py` (the dispatcher
and the entry point). The status, reason, route and skip vocabularies stay in
`research/resolution_fetch_result.py`, where they were, because every string in that module is a
telemetry contract the archive matches on.

The entry point is one coroutine, `ladder.fetch_url(url, policy=..., ctx=...)`. It consults
`policy.known_api` first, since a public API that answers the URL exactly costs no page fetch at
all, and then runs the ladder. The policy is bound onto the context there rather than threaded
through the twenty rung functions that already take a context, so a rung reads `ctx.policy` and
no rung signature carries a second argument. The session and the per-host semaphore map ride the
context for a related reason: a session per URL would change the fetcher's connector limits, so a
caller that already holds one passes it down, and a caller that holds none gets one opened and
closed for that URL alone. The gap-fill adapter keeps its own host map separate from the Tier-1
caller because the browser rung can hold its gate through a long Chromium launch.

The process-run cache sits after known-API rung 0 and before caller verdict and presentation. It
holds at most 50 URL keys in LRU order. HTML entries retain the full extraction, chart and embed
facts, and links; raw-text entries retain the complete decoded text; PDF entries retain only a
reference to the parsed-document side cache. A hit therefore reapplies the current caller's body
route, verdict, query, disclosure, link collection and character cap. An incompatible body route
declines the entry and fetches normally. A direct read that the current caller rejects or judges
too thin enters that caller's ordinary escalation without repeating the direct request. Cached
rendered, derived-feed and Wayback reads are terminal products of their acquisition routes, while
the paid `url_context` model answer is never cached. Errors, empty reads and throttle interstitials
are also excluded so retries remain real requests. Redirects are indexed by both the requested and
final URL.

Cache presentation is bounded by the current context's remaining wall and runs in a worker thread,
which keeps a PDF's query-specific BM25 verdict off the event loop. Hits carry no old rung attempts
and do not change the current context's clock or per-question counters. `FetchResult.cache_hit` is
an additive, default-false serialized field used by the loop adapter to emit `method=cache`; the
cached object is a separate typed read artifact, never a `FetchResult`, raw PDF bytes or `PdfText`
payload.

### The policy knobs

`LadderPolicy` (`policy.py`) is frozen and holds only what the two callers genuinely differ on.
Everything they share stays a plain module constant read where it is used: the two byte caps
(5 MiB for a page, 40 MiB for a declared PDF), the per-hop HTTP timeout and its floor, every
per-rung wall floor, the chrome and JavaScript-wall thresholds, the robots pre-check, the
platform self-reference refusal, and the per-question Wayback and paid-read caps. That line is
drawn deliberately: a constant a test patches on the module that reads it goes inert the moment a
rung reads it off a frozen dataclass instead, and it goes inert silently.

The same rule decides when a declared knob starts being read: every knob below is read off the
policy today except the two seats whose sibling implementations land separately.

There are three gap-fill presets rather than one, because the loop runs the ladder three ways.
`GAP_FILL_FETCH_POLICY` is the `fetch` tool's, `GAP_FILL_DOCUMENT_POLICY` is `read_document`'s free
acquisition ladder (25 s, and no archive rung: it sits in front of the paid reader and an archived
copy is not what a document read was asked for), and `GAP_FILL_DIRECT_POLICY` is one direct fetch
with no rungs at all, for the robots.txt pre-check, whose whole point is the body a floor or a rung
would replace. The three differ only in `total_wall_s` and `rungs_enabled`.

| Knob | Resolution-source preset | Gap-fill preset |
|---|---|---|
| `verdict` | `RESOLUTION_SOURCE_VERDICT`: both content floors, the header decides the branch | `GAP_FILL_VERDICT`: any non-empty extraction is content, the bytes decide the branch |
| `rungs_enabled` | all five escalation rungs | `impersonate` / `rendered` / `wayback` for `fetch`, minus `wayback` for the document ladder |
| `total_wall_s`, `rung_wall_margin_s` | 45 s less a 2 s margin, the provider's own wall | 90 s for the `fetch` tool, 25 s for the document ladder, no margin |
| `per_url_max_chars` | 6,000 per URL, applied at presentation | None: the loop windows a page at presentation instead |
| `wayback_max_age_days` | 30 days, past which a capture is withheld | None: the capture date is surfaced and the driver judges |
| `wayback_extra_trigger_statuses` | empty | `unsupported_type`: the loop also substitutes for a body it could not read at all |
| `wayback_needs_host_refusal` | False | True: a `blocked` carrying no host status is a refusal WE made |
| `impersonate_dial_wall_s` | None: the rung's remaining budget, which is one question's worth | one plain hop's timeout, so a slow chain cannot spend a whole tool budget |
| `disclose_unreadable_embeds` | True: a page hiding figures in an embed leads with that note | False |
| `thin_content_escalation_chars` | None: escalate to the browser on status alone | 500: a success that short with no chart block escalates |
| `collect_links` | False | True: the driver is handed the page's outbound links |
| `render_memo_scope` | `resolution_source` | `gap_fill_v2` |
| `caller` | `resolution_source` | `gap_fill_v2` |
| `known_api` | the known-API registry, or None | the same registry |
| `digest` | `page_digest.digest_page`, or None for the free BM25 selection | the same |

The Wayback row is the one genuine coupling rather than a preference. The 30-day bound is
calibrated on a page a question cites as its grading source: a month-old capture of the page a
question grades on is still evidence about that page, and a URL a driver chose carries no such
guarantee, so the loop surfaces the age and lets the driver decide.

### What a verdict decides

The READ is shared and identical for both callers: the charset-honouring decode, the ARIA-table
rewrite, the calibrated two-pass trafilatura extraction with its line-shape metric, the inline
chart-data read, the outbound-link collection, the meta-refresh detection, the pypdf parse. What
the two callers differ in is the VERDICT on what came back, and that is one policy field
(`policy.verdict`, `verdict.py`) rather than a flag per difference. Three decisions ride it.

Which BRANCH a body takes. The fetcher routes on the Content-Type header, with the `%PDF-` sniff
inside the fallback document branch, so a body labelled `text/html` is read as HTML whatever its
bytes say. The driver routes on the bytes first, because a mislabeled document is common on the
hosts it reaches: PDF magic or a declared PDF goes to the document branch, while image magic or a
declared image is retained for gap-fill's local image viewer and is not read as text. Image
`fetch(image_url)` and `read_document(image_url, ask)` immediately deliver pixels through the same
local viewer path as `view_image`; none returns navigation instructions or uses the paid
document-reader fallback. An `<html` substring anywhere in the body routes to HTML even under an
odd content type. Both callers route `application/xml`, `text/xml` and structured `+xml` media types
to the raw-text reader, which preserves the XML tags.

Whether an HTML extraction counts as CONTENT. The fetcher publishes only text that clears the
400-character chrome floor and the line-shape metric (`looks_like_page_chrome`,
`_no_content_verdict`), because its section is captioned primary grading evidence; anything else is
`embed_shell` / `js_wall` / `thin_page` with no text. The driver is handed any non-empty
extraction, and an empty one is `js_wall`, which its adapter renders as this ladder's own `empty`.
Both honour `chrome_metric_withheld`, so the 0.38 content-share metric is not a divergence.

What a parsed DOCUMENT says. The fetcher runs the BM25 passage digest in the parse's own thread hop
and withholds a document that matched no query term (`no_matching_passage`); the driver takes the
whole joined text with its truncation note, holds the parse in `research/document_cache.py` for its
own later ask-directed digest, and never withholds on a non-matching query. The parse itself never
rides the `FetchResult`: `PdfText.pages` is the whole document (833,450 characters on the receipt
file) against a 200,000-character archive cap, so one such field would truncate a whole question's
archived payload to a preview.

### Local source and image bodies

The shared body classifier also recognizes bounded local source containers. ZIP signatures and
ZIP/Office MIME types route to `research/source_documents.py`; OLE workbook bytes route to the
legacy `.xls` reader, and valid UTF-8 JSON labelled `application/octet-stream` is accepted by the
ordinary text route. The ZIP reader supports text members (`.txt`, `.md`, `.json`, `.xml`, `.log`),
CSV/TSV, `.xlsx`/`.xlsm`, `.xls`, and `.docx`. It identifies OOXML workbook and Word packages
before treating a ZIP as a generic archive. It never extracts members to disk or recursively
opens nested archives. Legacy `.doc`, encrypted sources, unsupported archive members, malformed
containers, and anything over a parser limit are disclosed as unreadable or unsupported.

Parsing runs in a worker thread behind the shared two-slot document-parse semaphore. A cancelled
caller leaves its slot held until that thread ends, so repeated timeouts cannot pile up abandoned
parsers. The cache retains the complete bounded `ParsedSource`, not the selector result or digest;
its 64 MiB byte budget also counts retained image bodies. Cache hits apply the current caller's
selectors, query, and character budget again. For gap-fill
`fetch`, a ZIP first returns an exact-name member inventory, and a selected workbook member first
returns a sheet inventory. Both inventories are navigation only and do not count as fetched
evidence. The driver selects a member or sheet to read and paginate its labeled text. For
`read_document`, an unselected local source is searched across all readable sections; exact member
and sheet selectors narrow the digest before its query-based excerpts are chosen. Resolution-source
presentation applies its own query-based excerpt and 6,000-character cap.

Spreadsheet readers return saved cell values. `.xlsx` and `.xlsm` formulas are paired with their
cached results; the parser does not evaluate formulas and explicitly labels a missing cached value.
`.xls` uses the saved values exposed by `xlrd` and carries the same notice. Active `.xls` percent
format tokens display the stored value as a percentage; quoted or escaped literal percent signs
preserve the stored number and include a format notice. ZIP directory entries remain in the
inventory as unreadable because they have no file content. Word text follows document order through
body paragraphs and tables, including nested tables, then reads each distinct non-empty default,
first-page, and even-page header and footer. Labels identify the variant and section, such as
`first-page header section 1`. Images, OCR, embedded objects, tracked-change content, text boxes,
footnotes, and other unsupported Word parts are not rendered as text; the tool result carries an
omission notice. General recursive archives, legacy `.doc`, Office rendering, and formula evaluation
remain unsupported.

The existing response limit is 5 MiB. Local-source limits are 20 MiB expanded content, 128 ZIP
entries, 32 sheets, 250,000 cells, and 2 million extracted characters. The run cache shares a
64 MiB LRU budget across parsed local-source text and retained image bodies. A parser limit
refuses the source rather than returning a partial extraction. A local parser refusal is terminal
in gap-fill: it does not invoke the paid `read_document` reader. Unsupported archive members are
listed in the inventory and left unread.

The classifier also extracts a small inventory of likely useful HTML image URLs from `<figure>`
captions and descriptive `alt` text. Those leads carry untrusted page metadata only, not pixels or
evidence; a `fetch` result labels them “pixels not read.” `view_image(url, crop=None)` fetches the
selected image through the shared transport, then normalizes supported static PNG, JPEG, WebP,
BMP, or GIF bytes to a metadata-free PNG. Direct `fetch(image_url)` and
`read_document(image_url, ask)` calls deliver pixels immediately through that same view path; they
do not return navigation instructions or invoke the paid document reader. SVG and all animated
formats, including APNG, are refused. Source decoding is capped at 25 megapixels; delivery is
capped at a 2,048-pixel longest edge, 2 megapixels, and 2 MiB. Normalization runs off the event loop
behind the shared two-slot parser semaphore; cancellation keeps its slot occupied until the worker
ends. Crop coordinates are `[left, top, right, bottom]` in displayed pixel orientation after EXIF
transforms. The three tools share a budget of four distinct normalized PNG hashes per question,
including crops; duplicate pixels reuse an ID and do not consume another slot.

Image bytes stay outside the transcript. After each batch of tool replies, the loop appends
byte-free image references; the next ordinary request to the same driver materializes those
references as image input. The pixels therefore add input tokens to that driver request without a
separate image-reader call. An image finding must name an image ID delivered on a previous model
turn, its registered source or final URL, and the driver's visual observation. Delivery provenance
proves which pixels were shown; it does not verify the interpretation. Visual findings label
numeric readings as transcribed or estimated. Normalized PNGs are written as content-addressed
`research_outputs/media/<sha256>.png` sidecars. The byte-free `images` manifest has one entry per
PNG hash with an `asset_path`, representative metadata, exact-deduplicated source/crop observations,
and separate unions of image URLs and parent-page URLs. Both research harvest paths verify and copy
referenced sidecars into the canonical archive.

### The HTML extractor policy, and the one classification path

`classify._extract_page_text` is the publishable extraction of an HTML body, and every decision in
it has a receipt. ARIA-role tables are rewritten to real tables first, then trafilatura runs at
default recall as the primary extractor with `favor_precision=True` as the fallback, and both are
scored by `verdict.content_share`. Running BOTH settings is the 2026-09-03 calibration's verdict
(`scratch/fetch_ladder_2026-09-03/chrome_calibration.md`: 118 bodies, five extractor variants on
identical bytes, texts labelled by hand), because each setting alone loses pages the other reads.
Precision alone withholds readable pages (kasa.go.kr pruned to 78 characters, two tracxn funding
tables, manifold's market body). Default alone publishes chrome: on congress.gov it swaps the
2,411-character bill-status card for 54,393 characters of a member-name dropdown, because
trafilatura's readability fallback replaces the main extraction when readability's text is over
twice as long and only precision prunes that dropdown out of the backup tree first; menu trees
(abs.gov.au, kasa.go.kr) then clear the chrome floor as `success`. On the calibration corpus the
two-pass policy publishes every labelled content text (46 of 46, three of them the congress.gov
card) and blocks the navigation-tree chrome, at the cost of one extra pass on the pages that fail
the metric. What it gives up: prose-shaped boilerplate (a cookie-consent wall, a glossary) passes
any line-shape metric, and kasa.go.kr's news ticker is withheld with its menu. An extraction under
the chrome floor skips the metric, because precision only ever shortens.

The precision pass is the one skippable part, and it declines under
`RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S` of remaining wall. Nothing else budgets this work:
the rendered rung gives the browser its whole remaining budget and classifies the DOM afterwards,
and a 5 MiB navigation tree costs seconds per pass against the 2 s margin the rung leaves the outer
`wait_for`. A skipped pass takes the exit a FAILED pass takes, the default text withheld under the
metric, so the wall can only ever withhold a page here, never publish one the metric refused.

Trafilatura gets the ORIGINAL BYTES in two cases, and in both its extraction is byte-identical to
what it was before the ARIA rewrite existed: a page with no ARIA role at all, and a page our own
decode mangled. The second is the one that matters. `decode_text_body` honours a BOM and the HTTP
header's `charset`, but a page that declares its encoding only in a `<meta charset>` decodes as
UTF-8 and comes back as mojibake, while trafilatura reading the bytes would have found the meta
declaration. That is why the gate is `undecodable_ratio == 0.0` rather than
`MAX_UNDECODABLE_CHAR_RATIO`: the shared bound is the refuse-the-whole-body threshold and is far
too loose here, since a mostly-ASCII cp1252 page whose only non-UTF-8 bytes are accented characters
scores about 0.01 against a bound of 0.10, and under the looser gate it took the rewrite and
reached forecasters as `R<?>sum<?> ... Qu<?>bec` where the bytes path returns the accents.

`classify._classify_html_body` is the one classification path for an HTML body, whichever rung
obtained it and whichever caller asked. Three things about its ORDER are load-bearing. Both embed
scans (`extract_datawrapper_charts`, `unreadable_data_embed_providers`) run on the RAW decoded HTML
before and regardless of main-text extraction, because trafilatura drops iframes and embed scripts
at every setting; the page's main text is trafilatura's to decode, which is why no vacuity check
runs on that branch. The inline chart read runs on EVERY HTML page rather than only thin ones,
because question 43949's page extracted about 80,000 characters of prose with none of the resolving
figures in it, so a thin-only gate would miss the record the rung exists for; it runs in its own
thread hop for the same reason the extraction does (one regex sweep plus a `json.loads` per config
over a body up to the 5 MiB response cap, or a rendered DOM up to `RENDERED_DOM_MAX_CHARS`,
measured at 22 ms on the 1.1 MB question-43949 page). And a chart block therefore RESCUES a page
the chrome floor would otherwise withhold, including a JavaScript-walled one, where the config in
the raw HTML is precisely the data the wall was hiding; a body the line-shape metric withheld does
not ride along under that block, because the metric's verdict is that the text is chrome and
publishing it filled the per-URL cap with navigation.

`classify._raw_body_outcome` strips allow-listed markup on the text branches only: a CSV or
plain-text body carrying an `<a href=…>` per row spends the per-URL budget on tags, while a JSON
body's angle brackets sit inside string values that are the data. Both text types get the strip
because the labels are demonstrably unreliable here, Datawrapper's own versioned route serving CSV
as `application/octet-stream`.

`classify._network_failure_class` buckets a transport exception for the fetch marker. The specific
subclasses are tested FIRST because aiohttp's TLS and DNS connector errors both subclass
`ClientConnectorError`, so the general connection bucket would swallow them, and the whole point of
the field is to tell a host that refused our TLS from one our egress IP could not resolve.
`malformed_response` is a response aiohttp's parser refused before the body was ours: a
`Content-Encoding` it cannot decode (the trueup.io zstd failure that had the brotli and zstd
decoders added, 2026-09-03), a header past the session's size caps, a bad status line. The parser
raises those as `HttpProcessingError` and the client re-raises them as
`ClientResponseError(status=400)`, a SIBLING of `ClientPayloadError` under `ClientError` rather than
a subclass, so `decode` cannot claim them and `connection` used to; the two say different things,
`decode` being a body that arrived and could not be read.

### What a self-produced refusal carries

`http_status` on a `FetchResult` means "a host answered us this", and nothing else. A refusal the
ladder makes itself carries None: the SSRF rejection of a derived hop, and the platform
self-reference refusal, both of which used to carry the redirect's own 301 or 302
(`guard._vetted_hop_target`, which is why it takes no `http_status` argument at all). Two rungs key
on that field to decide whether a second transport may be handed the URL — the impersonated retry
on a host's 403, and the archive rung under `wayback_needs_host_refusal` — and handing either a URL
we refused ourselves is the bypass the guard exists to prevent.

### The thin-content escalation

`_escalate_unresolved` returns a success unchanged, with one exception: a caller with a
thin-content floor (`thin_content_escalation_chars`) can mark a success `escalate_rendered`, and
that success goes to the browser and nothing else. A chart block pins the flag off, because a
render replaces the client-side series with a DOM lacking it (question 43949). When the render
declines or reads nothing, the thin text stands, which is the whole point of escalating on a
success rather than on a failure; the archive and the paid reader are never reached, because those
answer a page we could not read AT ALL and this one we did. The fetcher's verdict never sets the
flag, so its dispatcher cannot take this branch.

### Why the rungs sit in this order

Cheapest first, and each rung declines by returning nothing, in which case the previous outcome
stands. `ladder._escalate_unresolved` is the order; `ladder._run_rung` is the bracket that closes
each rung's attempts on that rung's own wall and outcome rather than the ladder's.

The impersonated retry is first, and its position is a reading choice rather than a functional
one. The trigger sets are disjoint (`rungs._rendered_rung_applies` fires only on `js_wall` and the
`thin_page` shape of `no_resolving_content`, never on `blocked`), so this rung never contends for
the browser escalation gate. First matches the `FetchRoute` order, meets a reader with the cheap
free retry before the expensive ones, sits before the archive so a live page beats a stale
capture, and sits before the paid reader so a rescue saves the read on that URL entirely.

The derived feed and the browser run under one per-question per-host gate
(`QuestionRungBudget.browser_escalation_gate`). The derived-API rung exists so a host with several
cited URLs pays for one Chromium launch, but the provider fans one task out per cited URL, so
every same-host URL asked `endpoint_for` before any render had finished, got None, queued on the
per-host gate inside the render, and launched its own browser after the first had already recorded
the endpoint. Holding the gate across the pair means the second URL re-asks once the first's
escalation is over and takes the feed off an ordinary GET instead. Waiting there costs the second
URL nothing it did not already pay queueing inside the render, and the rungs behind it re-read
their wall budget after the wait.

The fast-path decline sits in the dispatcher rather than inside the rung. The rung's own gates all
cost something, a budget read, a memo lookup or a launch, and the question's time-budget fast path
is a fact the dispatcher already holds.

Wayback comes before the paid read. The two trigger sets are disjoint from the browser's by
construction (`rungs._WAYBACK_TRIGGER_STATUSES`), so the order between the archive and the paid
reader is again a reading choice: the free route whose egress is not ours first, then the one that
spends money.

The paid Gemini read is last, because it is the only rung that spends money and the only one whose
product is a model's answer rather than the host's bytes. It is off by default in code and on in
every bot workflow. It is asked about the DIRECT outcome, and an archive withhold does not stand
in its way: a capture too old to serve is still a page we could not read fresh, which is exactly
the population this rung exists for. The withhold stays the ladder's fallback, so with the flag
off, or the reader declining, a stale capture still reports `stale_data`. The paid rung's own
attempt closes on the direct status when it declines, like every other rung, because the archive's
verdict is not an outcome a model read can produce and `rung=url_context outcome=stale_data` on
an escalation line read as if it were.

### What the dispatcher returns, and what it carries forward

`_escalate_unresolved` returns the first rung's rescue, or the direct result unchanged when every
rung declines or fails. A rung that fired and produced nothing still leaves its attempt on the
context, which is what makes `route=rendered status=js_wall` readable in the archive as "we tried
the browser and this is still the answer", the same convention the meta-refresh hop follows. The
Wayback rung is the one rung whose non-rescue is a verdict rather than None (`stale_data`, a
capture we read and will not serve), and it is kept as the fallback rather than as an early return
so the paid rung below is still reachable for that page. The `session` argument is the aiohttp
session the rungs that issue an ordinary GET use; the browser rung and the impersonated retry
ignore it, because Chromium and libcurl each bring their own transport.

Two facts travel out of `_fetch_one` with the result. The rungs that live inside the direct fetch,
the meta-refresh hop and the local PDF read, are over by the time it returns, so its status is what
they left standing and their attempts close on it. And `chrome_metric_withheld` is a fact about
this URL's ladder rather than about one result, so it is carried onto whatever the ladder returns:
a rung's rescue has its own extraction, which the metric never withheld, so summing off final
results alone gave the `chrome_metric_withholds` counter no count at all for exactly the withholds
the ladder then paid off, which is the policy's whole point.

### The per-URL context and the per-question budget

`LadderContext` (`context.py`) is one per fetched URL, so its `rungs` list belongs to that URL and
can be stamped onto its result, while `query`, `started`, `now`, `shared`, `policy`, `session` and
`host_sems` are the same for every URL in one provider call. Every field has a default, so the
monkeypatched fetch surface can still be driven with three positional arguments, and a default
context is simply "no question text, clock starts now", which gives a direct fetch exactly the
behaviour it had before the ladder existed.

Four of those fields carry a decision. `query` is the question's title plus its resolution
criteria, and it decides which passages of a 220-page PDF a forecaster sees. `started` is the
provider's own wall-clock origin, so every rung bounds itself against the same 45 s the outer
`asyncio.wait_for` uses. `now` is the wall-clock counterpart the Wayback rung ages a capture
against: a monotonic origin cannot date anything, and taking the clock inside the rung would make
an archived snapshot's disclosure depend on when it happened to run rather than on the fetch it
belongs to. `fast_path` is the question's time-budget thin-window mode (`time_budget.py`), handed
down from the orchestrator through the provider factory; the two expensive rungs decline on it
before any side effect and record a `fast_path` skip, while the cheap rungs run as they do off it.
It only ever declines, so a question with no fast path is byte-identical to one before the gate
existed.

`QuestionRungBudget` is the per-question half, shared across a question's cited URLs so the capped
rungs cannot be paid for once per URL. Its two counts are the Wayback snapshot cap and its
analogue, the paid `url_context` read cap, which is what stops a question citing several dead
sources from paying per source inside one provider wall. It is separate from the context because
what is being bounded is per-question: every Wayback snapshot shares the netloc
`web.archive.org`, so the loop-wide per-host `Semaphore(1)` would turn N cited URLs into N
sequential archive fetches inside a wall that discards work already done when it fires.

`browser_escalation_gate` is the `Semaphore(1)` that serializes one question's
derived-feed-then-browser escalations on a host; why it is held across the pair is under "Why the
rungs sit in this order" above. It is per question rather than loop-wide on purpose: the
cross-question shape still serializes on the loop-wide host gate exactly as before, and a loop-wide
gate here would be one more unbounded process-global acquire in front of a wall that discards
finished work (`FUTURE.md` item 5, the operator's call).

Two module-level tables in `context.py` are derived from each other rather than spelled twice.
`_RUNG_WALL_SKIP_PHRASE` maps each route to the human phrase its wall-budget skip logs, so the one
message template in `claim_rung_budget` reads the same as the six hand-copied lines it replaced.
`_BUDGET_GATED_RUNGS` is every rung that can record a `wall_budget` skip, in ladder order, and
`_rung_counts` breaks the aggregate `rung_budget_skips` out per member from it. It is derived from
the phrase map because the two drifted in opposite directions: a rung phrased but not listed lost
its `<rung>_budget_skips` key from the archive silently, and a rung listed but not phrased raised
`KeyError` inside `claim_rung_budget`, which the provider's `gather(return_exceptions=False)` turns
into losing every page of the question. Dict insertion order is the ladder order, so the keys are
unchanged.

## Import conventions

Imports go at module top, and `forecaster.py` has none inside functions. A
function-scoped import needs one of exactly three real justifications, and its
`# noqa: PLC0415` comment must name which:

1. **Genuinely optional dependency**: matplotlib behind an `ImportError` guard
   (`research/timeseries_anchor.py`, `calibration/fit_platt_cli.py`). matplotlib is in
   the dev group and prod installs `uv sync --no-dev`, so it is the one package that is
   genuinely absent at runtime, and the `DEP004` entry in `pyproject.toml` is where that
   exemption is declared to deptry. `rapidfuzz`, `yfinance` and `asknews` are all declared
   runtime dependencies, so a function-scoped import never protected against their
   absence.
2. **Late binding for a patch surface**: a test patches the name on its SOURCE module
   and the consumer must resolve it at call time. Hoisting a `from x import y` here binds
   the unpatched object at import time and silently defeats the test; this repo has
   shipped that bug. Live cases: `numeric.pipeline.sanitize_percentiles` from
   `ablation/run_stacker.py`; `numeric.pchip_cdf.*` from `numeric/pchip_processing.py` and
   `scripts/score_ghosts.py`; `asknews_sdk.AsyncAskNewsSDK` and
   `constants.NATIVE_SEARCH_WALL_TIMEOUT` from `research/providers.py`;
   `constants.FETCH_GET_RETRIES` from `fetch_hardening.py`;
   `fallback_openrouter.build_llm_with_openrouter_fallback` from `research/targeted.py`;
   and `ablation/forecasters.py`'s deliberate self-import (tests rebind
   `run_forecasters_for_question` on the module).
3. **A real circular import**: verify it by hoisting and importing, do not assume.
   Prefer fixing the module layout over keeping the lazy import.

Cold start is not a justification on its own. `import metaculus_bot.forecaster` costs
about 4.2 s, nearly all of it `forecasting_tools` and `litellm`; the four hoists done in
2026-08 added about 124 ms (2.7%), of which 105 ms is yfinance arriving via
`timeseries_anchor` → `ts_fetch`. matplotlib does not come along, because
`timeseries_anchor` imports `ts_chart` inside its own render guard. Re-measure with
`python -X importtime` if you add a top-level import that pulls scipy-, matplotlib- or
browser-weight machinery onto this path; the bar for a lazy import is a couple of hundred
milliseconds, not ten.

"The formatter would strip it" is also not a justification. Ruff only strips an import
with no usage, so add the import and its usage in the SAME edit and it survives.

Whichever applies, keep the `# noqa: PLC0415` and state the reason inline. Never add a
`# noqa` or a `HARNESS-SCAN-EXEMPT` marker to silence a linter or the smell scanner without one
of these three justifications. Deleting a marker by fixing the import it excused is always
welcome.

## Where the pieces live

| Concern | Module |
|---|---|
| Startup / CLI | `main.py`, `metaculus_bot/cli.py` |
| API identity preflight | `metaculus_bot/api_preflight.py` (`verify_api_identity`, its Metaculus wrapper, `ApiIdentityError`) |
| Mantic platform client (Crucible, a Metaculus fork) | `metaculus_bot/mantic.py` |
| Which platform a question is on | `metaculus_bot/question_platform.py` (`question_platform(question)` reads the `page_url` host; the `PLATFORM_METACULUS` / `PLATFORM_MANTIC` tokens live in `constants.py`). The prompts read it for the platform-aware scoring sentence and the Mantic out-of-range base rate; the per-bin gate and the tail floor key on it |
| Close-derived time budget (intake skip, fast path, research-phase deadline) | `metaculus_bot/time_budget.py` |
| Publish hardening and close gate | `metaculus_bot/publish_hardening.py` (the forced POST timeout is scoped to `QUESTION_PLATFORM_HOSTS` from `constants.py`, so it covers both platforms), `publish_gate.py` |
| Per-question orchestration | `metaculus_bot/forecaster.py` |
| Post-fan-out aggregation routing | `metaculus_bot/stacking_route.py` |
| Drop attribution / degradation counters | `metaculus_bot/drop_telemetry.py`; `degradation_counters.py` formats immutable snapshots built by `forecaster.py` |
| Research fan-out | `metaculus_bot/research/orchestrator.py`, `research/providers.py` |
| Outbound fetch transports | `research/http_fetch.py` (plain HTTP, redirects, per-host gates), `research/fetch_ladder/guard.py` (the SSRF preflight, the vetted DNS resolve, the aiohttp session), `research/impersonated_fetch.py` (the `curl_cffi` TLS-impersonating retry of a 403, with its own DNS pin and per-hop re-guard), `research/rendered_fetch.py` (headless Chromium), `research/url_context_reader.py` (one paid Gemini `url_context` read), `research/robots_policy.py` (the `Google-Extended` pre-check in front of that read) |
| The shared fetch ladder and its `fetch_url` entry point | `research/fetch_ladder/` (`policy.py` the per-caller knobs, `verdict.py` what a caller makes of a body it read, `context.py` the per-URL and per-question bookkeeping, `run_cache.py` the complete process-run read artifacts, `classify.py` one body's classification, `direct_fetch.py` the redirect loop, `rungs.py` the seven rungs, `ladder.py` the dispatcher and the `fetch_url` entry point, `throttle.py` the shared interstitial detector, `digest.py` the digest seat, `guard.py` the outbound guard), `research/resolution_fetch_result.py` (the status, reason and route vocabularies), `research/document_cache.py` (the parses a run holds, which never ride a `FetchResult`), `research/derived_api.py`, `research/wayback.py` |
| Local ZIP, CSV/TSV, Excel, and Word parsing and presentation | `research/source_documents.py` (bounded parse), `research/source_presentation.py` (member/sheet selection, navigation inventories, and query digests), `fetch_ladder/run_cache.py` (complete parsed-source cache artifact) |
| Gap-fill image leads, normalized image views, and archive sidecars | `research/image_leads.py`, `research/image_assets.py`, `research/agentic/image_messages.py`, `research/image_persistence.py` |
| Resolution-source fetcher: its adapter over that ladder | `research/resolution_source.py` (URL selection, the Datawrapper second phase, the provider factory, the telemetry emitter and the rung counts) |
| Resolution-source text and section budgets | `research/resolution_presentation.py` |
| Datawrapper response classification, freshness and dataset ordering | `research/resolution_datawrapper.py`; requests and question budgets remain in `research/resolution_source.py` |
| Gap-fill v1 / v2 | `research/targeted.py`, `research/agentic/` |
| Forecaster runners | `metaculus_bot/forecaster_runners.py` |
| Value extraction | `metaculus_bot/value_extraction.py` |
| Numeric CDF | `metaculus_bot/numeric/` |
| Date question as a numeric question on the epoch-seconds axis | `metaculus_bot/numeric/date_axis.py` (`EpochDateQuestion`, `as_epoch_question`, `numeric_view`, `parse_forecast_date`, `format_epoch`, `question_json`) |
| Per-bin PMF elicitation on the small Mantic grids | `metaculus_bot/numeric/config.py` (`elicit_per_bin`, `PMF_ELICITATION_MAX_BINS`, `PMF_ELICITATION_PLATFORMS`), `numeric/pmf_grid.py` (bin labels), `numeric/pmf_cdf.py` (PMF to CDF) |
| Mantic out-of-range tail floor, the last touch on a published numeric or date CDF | `metaculus_bot/numeric/out_of_range_floor.py` (`floor_published_tails`), `MANTIC_OUT_OF_RANGE_TAIL_FLOOR` in `constants.py`, applied in `forecaster.py` `_aggregate_predictions` |
| Aggregation + stacking | `metaculus_bot/aggregation_pipeline.py`, `stacking.py` |
| Model roster (source of truth) | `metaculus_bot/llm_configs.py` |
| Prompts | `metaculus_bot/prompts.py` |
| Constants / thresholds / env flags | `metaculus_bot/constants.py` |

## Related docs

- [research.md](research.md): research providers, gating, API-key routing.
- [numeric_pipeline.md](numeric_pipeline.md): percentiles to PCHIP CDF, bounds, steps.
- [value_extraction.md](value_extraction.md): the extraction ladder and its fidelity rules.
- [prompts.md](prompts.md): every forecasting-prompt rule and why it is there.
- [agentic_gap_fill.md](agentic_gap_fill.md): the v2 agentic research loop.
- [roster_history.md](roster_history.md): the ensemble roster, its history, dormant paths.
- [performance_analysis.md](performance_analysis.md): residual-analysis conventions.
- [operations.md](operations.md): running the bot, workflows, cost discipline, credits.

## A note on cost

Any command that hits live LLM or research APIs spends real money and, in live modes,
publishes comments to Metaculus. Do not launch one without the operator's approval.
The free, self-contained paths (`make test`, `make lint`, `make format`,
`make check_credits`) are safe to run anytime. Details in
[operations.md](operations.md) and the repo's `AGENTS.md`.
