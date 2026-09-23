# Constants and their receipts

This file holds the reasoning behind the values in `metaculus_bot/constants.py`. That module is a
flat registry: one home for every operational knob, because splitting it scatters lookups. Each
constant there carries at most one comment line, the single most important why, ending with a
pointer to the entry here that holds the rest. Entries below appear in the same order as the file,
one `##` section per section banner in `constants.py`, and one `###` entry per constant or per
group of constants that shared a rationale. Headings are the exact constant names, so a pointer of
the form `Receipt: docs/constants.md "NAME"` is grep-findable; where an entry heads a group, the
pointer names whichever of its constants fits the 120-character comment line, and grepping that name
lands on the heading. A few pointers name a `##` section heading instead, for a rationale that
belongs to the whole section rather than to one value. Constants whose names already say everything
(`MANTIC_SITE_URL`, the `*_ENABLED_ENV` flag names) have no entry.

While `constants.py` ran past 1,000 lines it carried a `HARNESS-SCAN-EXEMPT-monolithic-file-loc`
pragma recording that the flat registry is the design rather than an accident. Moving this prose
out brought the file back under that threshold, so the pragma is gone; re-add it with the same
reason if the file crosses 1,000 lines again.

Two conventions the whole file follows. Wall-clock caps come in pairs: a per-request timeout the
HTTP or LLM client enforces, and a slightly larger `asyncio.wait_for` wall that backstops it, sized
so the cleaner per-request error fires first when it can. Dated levers (credit alerting, provider
degradation, the cup reminder) read the calendar at CALL time rather than at import, so a
long-lived process crosses the date without a redeploy and tests inject a fixed date.

## TOURNAMENT IDs, UPDATE THESE EACH QUARTER/SEASON

### TOURNAMENT_ID, TOURNAMENT_END_DATE, TOURNAMENT_HARD_STOP_WEEKS

The AI Forecasting Benchmark / FutureEval tournament, a bot-only competition. Update these when a
new season starts; the project index is at <https://www.metaculus.com/project/aib/>.

Read the project object before editing. `/api/projects/tournaments/<slug-or-id>/` is the route that
serves a project (a bare `/api/projects/<id>/` answers 404), and `TOURNAMENT_END_DATE` is that
object's `forecasting_end_date`, not its `close_date`. The two differ by two months for this
season, and `close_date` is the later date on which the project itself closes.

Fall 2026 was published as project 33121 on 2026-09-04. The object was read directly from
`/api/projects/tournaments/33121/` on 2026-09-06: `start_date` 2026-09-28, `forecasting_end_date`
2027-01-06, `close_date` 2027-03-05, `score_type` spot_peer_tournament, `bot_leaderboard_status`
bots_only. So `TOURNAMENT_ID` is `fall-futureeval-2026` (project id 33121) and
`TOURNAMENT_END_DATE` is 2027-01-06, API-verified 2026-09-06.

`TOURNAMENT_HARD_STOP_WEEKS` is two weeks of wiggle room past the end date before
`check_tournament_dates` stops warning and raises.

### METACULUS_CUP_ID

The Metaculus Cup, a human plus bot competition. This constant used to hold the undated
`metaculus-cup` slug and rely on Metaculus redirecting it to whichever cup was current. Metaculus
now rejects that slug: the posts list answers HTTP 400 for `tournaments=metaculus-cup`, verified
2026-09-03. So the constant carries the season's dated slug and has to be re-pointed at each new
cup. There is no auto-resolving spelling left.

Fall 2026, read straight off `/api/projects/tournaments/metaculus-cup-fall-2026/` on 2026-09-03:
project id 33108, name "Metaculus Cup Fall 2026", `start_date` 2026-08-28T12:00:00Z,
`forecasting_end_date` 2027-01-01T00:00:00Z, `close_date` 2027-01-04T00:00:00Z, `score_type`
peer_tournament, `visibility` unlisted, `bot_leaderboard_status` exclude_and_show (bots forecast
and are shown, but sit outside the human leaderboard; every recent cup season reads the same, so
this is the cup's normal setting), `questions_count` 0, meaning the cup was open but had published
nothing yet. Either the slug or the numeric id resolves on that route. The slug is kept because it
is what the supply probe's per-slug rows and the research archive's labels read.

A note for residual analysis: `score_type` peer_tournament means cup records carry a
coverage-scaled peer score and no spot peer, unlike the bot tournament's spot_peer_tournament.
`performance_analysis/platform_scores.py` already keeps the two in separate sort tiers, so do not
pool them on one score field.

### MANTIC_HOST, MANTIC_SITE_URL, MANTIC_API_BASE_URL, MANTIC_TOURNAMENT_ID, MANTIC_TOURNAMENT_END_DATE

The Mantic "Crucible" competition at competitions.mantic.com, a fork of the open-source Metaculus
platform with the same API shape. See `docs/operations.md` "Mantic" for the mode itself.

Read straight off `/api/projects/tournaments/preseason-2/` on 2026-09-08: project id 4,
`start_date` 2026-09-03T11:19:48Z, `forecasting_end_date` equal to `close_date`
2026-09-20T12:00:00Z, `score_type` spot_baseline_tournament, `bot_leaderboard_status` bots_only.
So `MANTIC_TOURNAMENT_END_DATE` is that `forecasting_end_date` on project 4, API-verified
2026-09-08.

No Series 2 project exists on that API yet. The valid slugs as of 2026-09-08 are preseason-2,
series-1 and practice-series-1, and an unknown slug answers HTTP 400. Re-point
`MANTIC_TOURNAMENT_ID` when Series 2 opens.

### MANTIC_FETCH_QUESTION_CEILING

The tournament fetch asks the framework for this many questions so that it walks offsets until an
empty page instead of trusting Mantic's `next` link, which is advertised past the last page: probed
2026-09-08, offset 600 of the 520-post Series 1 still carried one. This is a ceiling, not an
expectation, and the fetch passes `error_if_question_target_missed=False`. Five hundred is five
pages; the most Mantic has ever held open at once is three questions, and the Series 2 rules allow
batch releases.

### MANTIC_BOT_USER_ID

The bot's own account, `nostreambot-bot`, public at `/api/users/81/`. An unauthenticated read of
that user has no `my_forecasts`, which is why the id is pinned here rather than discovered.

### MANTIC_OUT_OF_RANGE_TAIL_FLOOR

The least mass a published Mantic numeric, discrete or date aggregate carries beyond each open
bound, enforced in `numeric/out_of_range_floor.py`. Metaculus aggregates are untouched.

Mantic scores an out-of-range resolution against a fixed 0.05 reference, `50 * ln(mass / 0.05)`.
Five percent there scores 0, while the structural 1% the pipeline publishes when every percentile
sits inside the range scores -80.5. In Series 1, half the date questions, a quarter of the discrete
ones and an eighth of the numeric ones resolved outside the displayed range.

Applying this floor to every Series 1 competitor's own 4,082 published distributions cost at most
1.9 points per question on average for any type, and gained up to 9.1 for thin-tailed bots.
Receipts: `scratch_docs_and_planning/mantic_adversarial_candidates_2026-09-08.md`, section 1.
Operator-approved 2026-09-08.

### METACULUS_HOST, QUESTION_PLATFORM_HOSTS

The question platforms the bot publishes to, whose own pages are self-references for research. One
tuple, because the two sets are the same two hosts today; if they ever diverge, split the tuple
rather than adding a flag.

Each consumer matches its own way. `publish_hardening` scopes the forced POST timeout by URL
substring, `research.resolution_url_scan.is_metaculus_self_ref` matches the hostname and its
subdomains (so the Metaculus apex covers `www.` and the API host), and the gap-fill v2 driver text
names both hosts from here.

For Mantic only the competition host is listed. `www.mantic.com` and `blog.mantic.com` are the
company's marketing site and blog, which publish forecasts and are a legitimate outside source.

### PLATFORM_METACULUS, PLATFORM_MANTIC

The `platform` vocabulary: which question platform a question's ids and `page_url` belong to. These
are data-contract tokens, since the research archive keys off the exact spelling, so add one and
never re-spell one. `question_platform.question_platform` reads one off a question's `page_url`
host, and the research persistence writer stamps one on every archived record.

### gemini_use_donated_openrouter_key

Whether OpenRouter Gemini calls route through the Metaculus-donated key. The default is True:
after Metaculus raised the Google rate limits on 2026-06-16, the donated OpenRouter key
(`OAI_ANTH_OPENROUTER_KEY`) serves most Gemini models. Both `gemini-3.5-flash` and
`gemini-3.1-flash-lite` succeed on the donated key, verified by live call in that session. Setting
the env var to a false-y value ("false", "0", "no") forces personal-key-only routing.

The known exception is `gemini-3.1-pro-preview`, our forecaster slot, which is pinned to the
personal key rather than merely falling back to it. It sits on the
`DONATED_KEY_BLOCKED_GOOGLE_MODELS` blocklist in `fallback_openrouter`, so
`should_route_via_donated_key` returns False for it even when this toggle is True: no donated
attempt, no 429, and no personal-key-fallback-counter bump, which would otherwise redden CI on
every question. That model 429s on the donated key because it routes through a free-tier Google AI
Studio BYOK key with no Pro free tier (quota 0). This is a temporary workaround pending the
Metaculus-side BYOK fix; see the `TODO(gemini-3.1-pro-donated)` tag on that constant.

The value is read at call time rather than at import, so a workflow env change takes effect without
a re-import. Scope: this toggle only affects OpenRouter routing in `fallback_openrouter`. The
google-genai grounded-search provider has no donated path at all and always reads the operator's
personal `GOOGLE_API_KEY`.

### donated_openrouter_key_enabled

Whether any OpenRouter call may route through the Metaculus-donated key. The default is True, so
Metaculus runs are unchanged. A Mantic run sets `DONATED_OPENROUTER_KEY_ENABLED=false`: Metaculus
donated that key for its own tournaments, so a run that forecasts for another platform spends only
the operator's personal keys. `should_route_via_donated_key` in `fallback_openrouter` consults this
before any provider match, so one false-y value covers every OpenRouter key choice in the process,
including the key-swap fallback and the credit telemetry's donated-key probe.

It has to be an environment variable set before the process starts rather than a CLI flag, because
the roster's module-level `GeneralLlm` objects in `llm_configs` freeze their `api_key` at import and
`main.py` imports them before `cli.main` runs. Mantic mode therefore fails shut at startup when
this still reads True, in `cli._assert_personal_keys_only`.

The value is read at call time rather than at import, so a workflow env change needs no re-import.

### check_tournament_dates

Both operands go through `_as_utc` so the comparison is timezone-aware on the same side of the
clock. Only the wall-clock reference moves, from local to UTC: the tournament close date is a
Metaculus (UTC) date and prod runs on UTC GitHub Actions runners, so this shifts nothing in prod
and at most a few hours of a staleness warning locally.

## Cup-season configuration reminder (dated, DISCHARGED for fall 2026, re-armable)

### FALL_CUP_SLUG, FALL_CUP_REMINDER_DATE, FALL_CUP_CONFIGURED

This was a deliberate time bomb. From `FALL_CUP_REMINDER_DATE` every run reddened until somebody
pointed `METACULUS_CUP_ID` at the fall cup's dated slug and enabled the "Forecast on Metaculus Cup"
workflow, because the undated `metaculus-cup` slug the constant used to hold had started answering
HTTP 400, so a cup run would simply have found no questions.

It was discharged on 2026-09-03. Metaculus granted $1,500 of API credits for the fall season,
`METACULUS_CUP_ID` now names project 33108 (metadata verified against the API in that entry above),
the cup workflow runs the same hourly split cron the tournament does, and `FALL_CUP_CONFIGURED` is
True. That makes `fall_cup_reminder_due()` False on every date, so the reminder cannot fire, and
the companion test in `tests/test_tournament_dates.py` became a pin that the cup stays configured.
The one remaining operator step is enabling the workflow on GitHub, where it is `disabled_manually`
and no file in this repo can change that; see `docs/operations.md`.

To re-arm for the next cup (spring 2027), re-date `FALL_CUP_REMINDER_DATE` to a couple of weeks
before that cup opens and set `FALL_CUP_CONFIGURED` back to False. The names still say FALL because
that is the season they were written for and `FALL_CUP_SLUG` is imported elsewhere, as the default
slug list in `scripts/supply_probe.py`; a re-arm should rename them to the new season and update
those imports with it. `FALL_CUP_SLUG` aliases `METACULUS_CUP_ID` so that the probe's slug list and
the reminder cannot drift.

### fall_cup_reminder_due, credit_alerts_active, provider_degradation_alerts_active

All three dated levers read the local calendar day deliberately. Each one gates operator-facing
behaviour (a loud reminder, a `sys.exit(1)`, an alertable degradation count) and the resume date is
read in the operator's own calendar day. Prod runs on UTC GitHub Actions runners, so local equals
UTC there, and `datetime.now(UTC).date()` would only move the boundary in a local dev shell. That
is what the `# noqa: DTZ011` on each of them records.

`today` defaults to the system clock read at call time rather than at import, so a long-lived
process crosses the date without a redeploy and tests can inject a fixed date instead of depending
on the wall clock.

## Research concurrency, comment limits and credential env names (no banner in the file)

### DEFAULT_MAX_CONCURRENT_RESEARCH

Concurrency ceiling for research providers such as AskNews and Exa. Deliberately conservative for
AskNews; adjust after observing rate limits.

### BENCHMARK_BATCH_SIZE

Default batch size for benchmarking runs, kept modest to balance concurrency against provider rate
limits.

### FORECASTS_SECTION_CHAR_LIMIT, RESEARCH_SECTION_CHAR_LIMIT, SUMMARY_SECTION_CHAR_LIMIT, COMMENT_CHAR_LIMIT

Metaculus comment safety limits. The published comment has three top-level sections
(`# SUMMARY`, `# RESEARCH`, `# FORECASTS`), each trimmed to its own budget before assembly, in
`comment.trimming.trim_section`.

FORECASTS, which holds the per-model rationales and the fenced JSON forecast blocks, gets the
largest share because it carries the per-model attribution the residual pipeline parses. RESEARCH is
a lossy fallback-archive re-print. SUMMARY holds the parser-critical bullets and is sized well above
any realistic bullet block so it never clips them.

The three caps sum below `COMMENT_CHAR_LIMIT`, and `trim_comment` shrinks RESEARCH first, never
bullets or rationales, if the assembled comment plus framework overhead still overflows.

### RESEARCH_PROVIDER_ENV

Optional environment variable that forces research provider selection. Accepted values, case
insensitive: "auto", "asknews", "exa", "perplexity", "openrouter".

### TEST_QUESTIONS_OVERRIDE_ENV

Optional override for the `--mode test_questions` question set. When set to a non-empty comma or
whitespace separated list of Metaculus question URLs, the test_questions path forecasts exactly
those instead of the hardcoded evergreen `EXAMPLE_QUESTIONS` list in `cli.py`. The
`test_bot_basic` workflow uses it to run a single question end to end; unset preserves full
test_bot behaviour.

### OPENROUTER_API_KEY_ENV and the other credential env-var names

Named constants, matching the existing `*_ENV` convention used for `GOOGLE_API_KEY_ENV` and
`FRED_API_KEY_ENV`, so the literal strings are not duplicated across `api_key_utils`,
`fallback_openrouter`, `research_providers` and `research_orchestrator`. That duplication is exactly
the typo risk the convention exists to prevent. Which of these are shared (donated) and which are
personal is in `docs/operations.md` "API keys and the shared-vs-personal key model".

`DONATED_OPENROUTER_KEY_ENABLED_ENV` is the master switch for the Metaculus-donated OpenRouter key.
Default on; a Mantic run sets it false and fails shut at startup if it is not, see
`donated_openrouter_key_enabled` above.

### ASKNEWS_MAX_CONCURRENCY, ASKNEWS_MAX_RPS, ASKNEWS_MAX_TRIES, ASKNEWS_BACKOFF_SECS

AskNews provider safety limits, global across all bots in the process. The defaults are
conservative for pro plans, which allow 1 RPS sustained, 5 RPS burst and 5 concurrency: the
sustained rate here is 0.8 RPS, well below the 1 RPS sustained limit.

### ASKNEWS_WALL_TIMEOUT

A hard wall-clock bound around the full AskNews provider: hot phase, historical phase, sleeps and
retries together. AskNews's internal retry loop fails fast on non-retryable errors, but a network
hang is otherwise unbounded, so this backstops that case and keeps a stuck AskNews call from
holding the whole research phase hostage.

Sizing: each phase (hot and historical) sleeps before its first call and applies backoff
`ASKNEWS_BACKOFF_SECS * (10 + 3**attempt)` on 429 or rate-limit retries, up to
`ASKNEWS_MAX_TRIES` attempts per phase, in `research/providers.py`. This wall sits above that whole
two-phase retry envelope plus API time, with headroom, while still bounding a genuine hang.

## OpenRouter credit telemetry

### OPENROUTER_CREDIT_FLOOR_USD

An early-warning floor for the donated key's remaining balance (`limit_remaining`). Below it,
`cli.main` logs a loud warning and exits non-zero after all forecasting and publishing complete. It
is a reminder to ask Metaculus for a top-up, not an abort, and not a claim that the key is empty.

Sizing: $100 is roughly 56 questions of runway. A published question costs $2.07 to $2.21 all in as
booked ($2.00 after the ledger's non-BYOK double count is removed), of which about $1.79 draws on the
donated credits (2026-09-09 cost pass; `make cost_report` re-measures it from the archive). The earlier
"250 questions at $0.38 to $0.41" sizing quoted an OpenRouter-only lower bound on one key. The lead
time is the point, since only Metaculus can refill this key and the operator cannot, so the warning
has to arrive while there is still time to ask. It was $1.00 until 2026-09-03, which fired only once
the key was already dry.

The floor is meaningless for the personal key, which has no `limit_remaining`, so it is only checked
against the donated key. See `metaculus_bot/credit_telemetry.py`.

### PROMPT_TOKENS_ALERT_THRESHOLD

The prompt size above which one LLM call logs a `PROMPT_SIZE_ALERT` WARNING from the role-ledger
callback (`credit_telemetry.py`, `RoleSpendTracker`). The 2026-09-09 cost pass measured the largest
prompts any role sends: the forecaster prompt at about 17k tokens and the gap-fill v2 loop peaking near
41k on its last research turn (`scratch/cost_pass_2026-09-09/v2_cost_anatomy.md`). 150k is well above
both, so a fire means a packet blew up (a runaway bundle, a tool-result loop) rather than normal
variance, and it sits below the 500k the operator named as the size that would degrade model
performance. It reads, never gates: the call is already billed when the callback sees its usage, so the
alert is a log line and enters no degradation counter. The per-role `max_prompt_tokens` field on
`CREDIT_ROLE_SPEND` is the same measurement summarised per run, so the threshold can be re-sized from
the archive.

### CREDIT_ALERT_RESUME_DATE

Dated suppression of the credit alerts, not of the logs. Before this date the two paths that turn a
credit shortfall into a non-zero exit, the floor breach in `cli.main` and the credit-caused
donated-to-personal fallbacks folded into `alertable`, do not redden CI. Every `CREDIT_*` log line,
including `CREDIT_FLOOR_BREACH`, keeps firing throughout: only the exit status and the alertable
arithmetic change.

History: alerting was suppressed from 2026-07-26, when the donated key drained and the operator
started self-funding the season, until 2026-09-03, when Metaculus granted $1,500 of credits (the key
read $1,449 remaining of a $2,300 limit) and the resume date was moved up from 2026-09-10 to that
day. Re-arm a window by pushing this date forward, either in the file or through the env override.

Non-credit fallback causes stay fully alertable: a 401 invalid or disabled key, a 404 with no
allowed providers, a 429 rate limit, and guardrail or data-policy refusals are each real breakage
rather than an expected empty wallet.

### PROVIDER_DEGRADATION_SUPPRESSED_UNTIL

Prediction-market venues, or prefetch catalogues, whose degradation is known and accepted, each with
a dated resume. Same contract as `CREDIT_ALERT_RESUME_DATE`: the finding is still logged in full,
still rides the `PROVIDER_DEGRADATION` marker, and still names its resume date in the end-of-run
summary. Only its contribution to `alertable` is dropped, and only until the date. Dated rather than
a bare boolean so a stale acceptance cannot outlive the season unnoticed, and per-venue rather than
global so accepting a dead Manifold does not blind the operator to a dead Kalshi.

It ships empty on purpose. Both degradations this machinery was built for, Kalshi's blank liquidity
labels and Manifold's zero contribution, were fixed in the same round, so suppressing either would
have hidden the fix's own verification. The mechanism exists to give the operator a documented,
dated lever instead of reaching for a code deletion when a venue is genuinely dead for good. A venue
with no entry is always alertable.

A suppressed run is loud rather than quiet on purpose. The `PROVIDER_DEGRADATION` marker shows the
arithmetic (`findings=1 alertable=0 suppressed=1`), names the resume date, and says the run stays
green, and the per-finding line still fires. A run reading `alertable=0` beside real degradation is
the shape that most needs a written record: the 2026-07-26 drained-donated-key run is the precedent
for how quietly such a run goes unrecorded.

## Forecasting clamps and numeric smoothing

### BINARY_PROB_MIN, BINARY_PROB_MAX

The binary prediction clamp. It mirrors Preseen-Atlas's clip-only tail protection: Atlas publishes
`0.96 * estimate + 0.02` and we adopt the clip portion only. See
`scratch_docs_and_planning/atlas_inspired_improvements.md` Workstream B.

### EXTREME_CALL_LOW, EXTREME_CALL_HIGH

The extreme band on a binary probability: a member call at or past either edge, inclusive at both
edges. Nothing here clamps or gates anything. The band only decides which per-member `EXTREME_CALL`
telemetry lines get logged, in `metaculus_bot/extreme_call.py`, so that the lone-versus-accompanied
extreme split is a query instead of a hand reconstruction from parsed comments every residual round.

The clamp a single-survivor binary publish goes through is defined as these two constants:
`THIN_PUBLISH_BINARY_FLOOR` and `THIN_PUBLISH_BINARY_CEIL` in the same section alias them rather
than restating the literals, so the telemetry that measures the exposure and the clamp that prices
it cannot drift apart. Retuning the band here retunes both, which is the intent. Retune them
together or not at all.

Evidence for the 0.05 and 0.95 edges: `scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md`
section 2, "The mechanism", which found 9 lone extreme binary calls, 4 of them right, at a mean
stated confidence of 0.972.

### THIN_PUBLISH_BINARY_FLOOR, THIN_PUBLISH_BINARY_CEIL

A floor on the published binary probability when exactly one forecaster survived, applied by
`apply_thin_publish_floor` in `post_processing.py` and wired into
`AggregationPipeline.base_combine` on the "single_forecaster" skip reason.

This is a mechanism rather than a fit. The median of an intact ensemble absorbs a member's extreme
tail call, and median-of-1 supplies no such variance reduction, so the range the published value may
occupy is narrowed in exactly that state to price the missing aggregation: [0.05, 0.95] sits
strictly inside the per-model clamp [`BINARY_PROB_MIN`, `BINARY_PROB_MAX`] = [0.02, 0.98] that the
member already passed. It fires only on a single-survivor publish, so a multi-member median
publishes as is even one below 0.05, and it never touches the per-model record: the survivor's
declared value stays on the comment's summary bullet and only the published aggregate moves.

The evidence, with its honest caveat: the whole measured benefit is one question. q44874 published
gemini's lone 0.03 on a YES resolution and took -105.27 spot peer; at [0.05, 0.95] it is +51.08 with
zero measured cost on the other three archived solo binaries (one win, three exact zeros, so n=4
with one non-zero row). The value 0.05 is informed by that question. Values of 0.07 and 0.10 buy
more on 44874 but start taxing 44870 and 44873, which were right. The downside is bounded:
publishing 5% where a correct sub-5% call would have scored costs about -3.11 spot peer per
instance, against a -105 tail.

Always-on and global variants were priced and rejected. Always-on never improves the published
pre-flip ensemble, and a global [0.05, 0.95] over 408 binaries is -52.02, with 50 losses to 1 win.
Do not widen the trigger. Receipt:
`scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md` section 2 (the clamp-variant table
and "A synthesis correction the individual cuts miss") and section 3, option "1=".

The edges reuse the extreme-band constants above by aliasing them, so there is one definition of
"extreme" serving both the telemetry and the clamp, and no pair of literals to fall out of step.
Retune `EXTREME_CALL_LOW` and `EXTREME_CALL_HIGH` to move both, or neither.

### MC_PROB_MIN, MC_PROB_MAX

The multiple-choice prediction clamp, aligned to forecasting-tools 0.2.92's `PredictedOptionList`
validator, which unconditionally clamps every option into [0.01, 0.99], renormalizes, and raises
`ValueError` when any option moves more than 0.05 from its input. Matching those bounds makes the
upstream validator a no-op on our already-clamped, sum-1 output, which eliminates publish-time
`ValueError` risk on many-option ballots: a dominant option plus several near-floor options is
exactly where the upstream renormalize-after-clamp fires the >0.05 raise. See
`clamp_and_renormalize_probs` and `clamp_and_renormalize_mc`, which clamp before every
`PredictedOptionList` construction.

### PMF_BELOW_RANGE_KEY, PMF_ABOVE_RANGE_KEY

Per-bin block keys, present only where that bound is open. They share `bin_probs` with the labels so
one object sums to 1.

### NUM_VALUE_EPSILON_MULT, NUM_SPREAD_DELTA_MULT, NUM_MIN_PROB_STEP, NUM_MAX_STEP, NUM_RAMP_K_FACTOR

Numeric CDF smoothing and spacing knobs. The pipeline that consumes them is documented in
`docs/numeric_pipeline.md`.

### DISCRETE_SNAP_MAX_INTEGERS, DISCRETE_SNAP_UNIFORM_MIX

Discrete integer CDF snapping, for "continuous" questions whose outcomes are integers.

## Post-hoc Platt calibration of the final published probability

### PLATT_CALIBRATION_ENABLED_ENV, PLATT_BINARY_MAX_ABS_DEVIATION, PLATT_MC_MAX_ABS_DEVIATION

Final-output logistic recalibration following Metaculus's notebook "Improving Forecaster Performance
via Automated Calibration Adjustment" (2026-05-01). The fitted parameters live in
`metaculus_bot/calibration/params.py` and are hand-edited after running the `fit_platt_cli`.

Both deviation caps are hard absolute caps applied after the smooth logistic transform. They cap how
far the calibration may move any single probability from the raw aggregation output. The operator's
stance is "tweak, don't massively deviate": the underlying fit can want a large move and the cap
prevents us from acting on it. Tune by hand after seeing the unconstrained fit.

The MC cap is tighter than the binary one because the per-option Platt is applied N times per
question and small per-option drift compounds after renormalization.

## Conditional Stacking Thresholds

### CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD

Binary: the probability range (max minus min) across per-model predictions. Chosen over log-odds
spread because log-odds spread saturates on clamped-extreme models that are often correct,
conflating "one model is sure" with "the ensemble is split".

### CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD

Multiple choice: the maximum per-option probability spread, that is max minus min across models for
the worst option.

### CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD

Numeric: the maximum percentile spread normalized by the question range, measured at the 10th, 50th
and 90th percentiles.

## Native Search Provider

### NATIVE_SEARCH_DEFAULT_MODEL

Default model for native search, without the `openrouter/` prefix. This is critical-path research,
and the constant covers both the always-on native-search provider that runs on every question and
the targeted search on the stacking path. Effort stays at the env default of low, see
`NATIVE_SEARCH_REASONING_EFFORT_DEFAULT` below.

Changed 2026-07-17 from sol to terra per the blind research-role audit in
`scratch/research_role_audit_2026-07-17/`: terra won the native-search role first, sol second, luna
third, with the verdict "MARGINAL EDGE". Changed again 2026-09-22, terra to `gpt-6-sol`: GPT-6 shipped
with no Terra successor, so every Terra role moved to Sol 6 at the same (low) effort.

### NATIVE_SEARCH_MAX_TOKENS

No temperature or top_p is set, because reasoning models defer to provider defaults and the LLM is
built with `temperature=None` so litellm omits the param; see `build_native_search_llm`.

### NATIVE_SEARCH_TIMEOUT

The litellm per-HTTP-request timeout. Raised 240 to 360 on 2026-05-17 alongside the gpt-5.5
medium-effort migration; see `comparison_v3.md`.

### NATIVE_SEARCH_WALL_TIMEOUT

A wall-clock backstop for the native-search provider. `NATIVE_SEARCH_TIMEOUT` above is the litellm
per-HTTP-request timeout, which resets across retries, so an un-pinned `allowed_tries` multiplies it
(`build_native_search_llm` pins `allowed_tries=1` for exactly that reason). It was also observed
defeated entirely on 2026-05-20 by an OpenRouter response that dripped about 700 lines of whitespace
keep-alive bytes over 8m37s before closing with malformed JSON.

`asyncio.wait_for` around `llm.invoke` gives a hard wall-clock cap regardless of what the underlying
HTTP layer does. It carries slight headroom over the request timeout so the cleaner per-request error
fires first when possible.

### NATIVE_SEARCH_REASONING_EFFORT_DEFAULT, NATIVE_SEARCH_REASONING_EFFORT_ENV, NATIVE_SEARCH_VERBOSITY_DEFAULT, NATIVE_SEARCH_VERBOSITY_ENV

Reasoning effort and verbosity for the OpenAI native-search call, overridable through the
`NATIVE_SEARCH_REASONING_EFFORT` and `NATIVE_SEARCH_VERBOSITY` env vars. An empty string disables
passing the kwarg.

Effort dropped from medium to low on 2026-05-20 after the OpenRouter whitespace-stream incident that
consumed 8m37s on a single call. The v3 bench (`comparison_v3.md`) measured effort=low at about 50s
against effort=medium at about 230s, so low gives roughly 4.5 times faster wall-clock and far more
headroom under `NATIVE_SEARCH_WALL_TIMEOUT` and `NATIVE_SEARCH_TIMEOUT`. The quality cost of low is
now absorbed by the model-tier upgrade above, a smarter model at lower effort. Override via
`NATIVE_SEARCH_REASONING_EFFORT` if a workflow needs medium back.

Note that this default applies only to the native-search provider. `DISAGREEMENT_ANALYZER_LLM` is
also at low, in `llm_configs.py`, while the forecaster slots set their own effort per instance in
`llm_configs.py`.

### NATIVE_SEARCH_MAX_RESULTS, NATIVE_SEARCH_CONTEXT_SIZE

Native search web options, passed to the OpenRouter plugins. Context size accepts "low", "medium" or
"high".

## Perplexity (fallback research provider; dormant while AskNews wins the ladder)

### PERPLEXITY_RESEARCH_MODEL, PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER

The model both Perplexity call sites use: the provider factory in `research/providers.py` and the
orchestrator's AskNews-failure fallback. A single constant because the two sites each carried their
own literal and silently drifted, with `providers.py` pinned to Perplexity's non-reasoning tier while
the orchestrator used the reasoning one. The direct-provider route takes the bare slug; the
OpenRouter route takes it prefixed, which is what `get_openrouter_api_key` keys its routing on.

### PERPLEXITY_WALL_TIMEOUT

A wall-clock cap for the two Perplexity call sites. Both previously had no wall bound at all: unlike
native search and resolution-source, neither was ever migrated to the gated retry wrapper, so a
stalled reasoning-tier call could run as long as litellm let it. Sized between
`GEMINI_SEARCH_TIMEOUT` and the research phase's own budget, generous enough for a reasoning search
and bounded enough that a stall cannot dominate the phase.

## Resolution-Source Fetcher (Tier 1)

The char caps in this section apply to raw fetched content only. The policy is that raw passthrough
is capped and LLM-emitted research is never truncated.

### RESOLUTION_SOURCE_HTTP_TIMEOUT

Per-request timeout. The probe basis is 0 to 2 seconds typical, with slack for slow government sites.

### RESOLUTION_SOURCE_WALL_TIMEOUT

A hard cap on the whole provider.

### RESOLUTION_SOURCE_MAX_URLS

Cited URLs fetched per question. The archive measured 58 URLs over 40 questions, about 1.45 on
average, so this bounds pathological multi-URL questions.

### RESOLUTION_SOURCE_MAX_RESPONSE_BYTES

Response byte cap. The CISA KEV JSON is about 1.5 MB, so 5 MiB is headroom.

### LOCAL_SOURCE_MAX_EXPANDED_BYTES, LOCAL_SOURCE_MAX_ENTRIES, LOCAL_SOURCE_MAX_SHEETS, LOCAL_SOURCE_MAX_CELLS, LOCAL_SOURCE_MAX_CHARS, LOCAL_SOURCE_CACHE_MAX_BYTES

Bounds for local ZIP, spreadsheet, and Word extraction in `research/source_documents.py` and
the shared ladder's process-run cache. The response still has the existing 5 MiB page-byte cap;
after that, a source may expand to at most 20 MiB, contain at most 128 archive members, 32
worksheets, 250,000 tabular cells, and 2 million extracted characters. Parsed
local-source text and retained image bodies share a 64 MiB cache budget and are
evicted in LRU order. Parsers refuse a source that crosses a limit rather than
returning a partial parse.

The parser reads ZIP members as streams and never extracts files to disk. It reads text-like
members, CSV/TSV, `.xlsx`/`.xlsm`, `.xls`, and `.docx`; nested archives and legacy `.doc` are not
recursively or locally read. See `docs/architecture.md` "Local source and image bodies".

### RESOLUTION_SOURCE_PER_URL_MAX_CHARS, RESOLUTION_SOURCE_TOTAL_MAX_CHARS

The per-URL cap sits at the elbow of the full-extraction distribution (p50 2.2k, p75 5.2k) and cut
truncation from 48% to 21% on the 2026-07-09 smoke run, at roughly 1.5k tokens per URL. The total
carries headroom so the per-URL cap binds: the maximum observed section was about 11.1k at 6k per
URL, and 18k is about 4.5k tokens worst case.

The 6,000-character per-URL budget includes provenance/chart/embed leads, section labels, digest
headers, and truncation markers. Those disclosures consume part of the allowance rather than being
appended beyond it, including for local-source excerpts and rescued reads.

### RESOLUTION_SOURCE_MIN_SECTION_CHARS

The smallest share of the total a success may render into. Under it the section is omitted and
counted in the "[N additional source(s) omitted]" line instead.

Below the truncation marker's own length, `_truncate_with_marker` degrades to a bare slice, so a
rescued section landing on a remainder shorter than its provenance lead rendered that lead cut
mid-word with no marker (`[Archived copy from the Wayback M`) while the route caveat above it
promised a complete disclosure. Sized above the longest lead, which is the `derived_api` other-page
lead at about 260 chars. Reachable on prod constants: 6000 plus 6000 plus 5900 direct pages ahead of
a rescued fourth.

### RESOLUTION_SOURCE_JS_WALL_MIN_CHARS

A 200-OK response with less than this much extracted text counts as a JS wall, per FINDINGS.

### RESOLUTION_SOURCE_GLOBAL_CONCURRENCY

The `TCPConnector` limit. Per-host politeness is serialized separately.

### RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS

An extraction at or above the JS-wall floor can still be pure page chrome: a tab list, a region
selector, a feedback-form blurb, an "about the data" note. Below this many extracted chars a 200-OK
page is withheld as `no_resolving_content` rather than rendered as grading evidence, whatever became
of the content. `status_reason` records whether a routeless data embed was named (`embed_shell`, for
Infogram, Flourish or Tableau, see `unreadable_data_embed_providers`) or not (`thin_page`). The floor
was gated on a named provider when it shipped for qids 44554 and 44556, which withheld one shape of
chrome and published the other: the 2026-09-01 round found five content-free `success` renders and
not one of them named a provider.

Calibrated on the 89 archived resolution_source records and re-checked for the ungated rule on
2026-09-02. Of 68 cited successes, 8 sit under 400 chars and all 8 are chrome: region selectors
(data.wastewaterscan.org, 127 chars, twice), Kazakh region names (election.gov.kz, 385), AP org
boilerplate (355), an ABS release-date list with no figure (344), a tracker's "about the data" note
(262), a feedback-form blurb (camara.leg.br, 157) and a clinicaltrials.gov data-element pointer
(111). The shortest archived extraction that actually carries the resolving content is 401 chars,
myfloridaelections.com's election-date table. So this is the observed elbow, and it stays
deliberately below it: a page above the floor keeps its text, plus the embed disclosure where one
applies, because withholding a terse but real data table costs more than leaving one shell visible.

Unmoved but re-read on 2026-09-03, when the extractor policy changed under it. That census was
measured on the precision extractor alone. The floor is now applied twice per page, in
`resolution_source._extract_page_text`. First to the default (recall) extraction, which only ever
lengthens a text relative to precision, so on that pass the floor binds on strictly fewer pages:
live over 149 archived HTML URLs, 10 crossed it upward and 9 of them carry the resolving content
(funding tables, two Yahoo history tables, a market's own resolution rules); the tenth is a JS flight
board whose column headers plus disclaimer total 644, that is one shell published where one was
withheld. Then, when the default text clears the floor on chrome alone (the line-shape metric below),
to the `favor_precision` re-extraction of the same bytes, which is the extractor the census was
fitted on, so a precision text under the floor is withheld as `thin_page` exactly as before. Left
where it is: the elbow it was fitted to is a property of what chrome weighs, not of the extractor,
and refitting it against a recall-era census is its own measurement.

### RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS, RESOLUTION_SOURCE_CONTENT_SHARE_MIN

The line-shape check on an extraction that clears the floor, in `resolution_source.content_share`. A
non-table line at least `CONTENT_LINE_MIN_CHARS` long counts as content and a shorter one as chrome.
Navigation-tree chrome tops out at a content share of 0.329 (the kasa homepage, ambiguous) and 0.239
(strictly labelled, the manifold sidebar), while the thinnest labelled content is 0.431 (the
wastewaterscan dashboard). Calibrated 2026-09-03 on 118 bodies; receipt
`scratch/fetch_ladder_2026-09-03/chrome_calibration.md`.

### RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S

The wall budget under which the `favor_precision` re-extraction is skipped and the default text is
withheld as it would be had that pass failed, in `resolution_source._extract_page_text`.

The pass is CPU work, runs after the body is already in hand (on the rendered rung, after the browser
has spent the whole remaining budget) and costs a little more than the default pass did on the same
bytes. On a synthetic div-soup dashboard DOM on the operator's laptop, 2026-09-04, the precision pass
took 1.5 s at 1 MiB, 3.2 s at 2 MiB and 9.7 s at 5 MiB; nested list menus run 5 to 10 times cheaper,
and the second review pass measured 42 s at 5.2 MiB on a heavier synthetic tree. That is against the
2 s margin the rung leaves the provider's wall.

Sized so a body around 1 to 2 MiB still gets its second pass on a slower GitHub runner. A bigger body
near the wall is withheld rather than published, and the real lever for those is
`RENDERED_DOM_MAX_CHARS` (see `FUTURE.md`). The budget is read against the wall remaining after the
default pass, so that pass's own cost counts.

## Inline chart configs (Highcharts), read straight out of the page we already hold

For qid 43949 the resolving IOM page fetched 200 and extracted about 80k chars of incident rows and
prose carrying none of the resolving figures, because the annual series lives in
`<div class="charts-highchart" data-chart="{...}">`. `research/resolution_chart_data.py` unescapes and
`json.loads` that config: zero LLM calls, no second request. Charts are read on every fetched HTML
page, not only thin ones, because that page's prose was far above the shell floor and a thin-only
gate would miss the record the rung exists for.

### RESOLUTION_SOURCE_CHART_MAX_CHARTS

The resolving chart is roughly first in document order (the IOM page carries 5), the same assumption
the Datawrapper hop makes.

### RESOLUTION_SOURCE_CHART_MAX_SERIES

IOM's widest chart is 3 series: Undetermined, Female, Male.

### RESOLUTION_SOURCE_CHART_MAX_POINTS

Points per series, kept from the end, because the resolving value is the newest one. Sixteen keeps a
full annual series intact (IOM's is 13 points, 2014 to 2026) while bounding its 149-point monthly
sibling to roughly the last year and a half.

### RESOLUTION_SOURCE_CHART_BLOCK_MAX_CHARS

A hard cap on the whole rendered block, budgeted out of the 6,000-char per-URL page cap rather than
added on top of it, so chart data can never evict more than a third of a cited page's text. Measured:
the IOM page's three readable charts render in about 700 chars together.

### RESOLUTION_SOURCE_CHART_MAX_CANDIDATES, RESOLUTION_SOURCE_CHART_MAX_CONFIG_CHARS

Configs examined per page before the scan stops, and the per-config char bound, which also bounds the
brace scan for the inline-script form. Both exist so that a page with hundreds of `data-chart`
attributes, or one unclosed brace, costs a fixed amount of work.

## Datawrapper second hop (Tier 2)

Poll-tracker pages lock their resolving daily series inside Datawrapper iframes that trafilatura
drops, seen on qids 44858 and 44841. The hop fetches the version-free live dataset at
`static.dwcdn.net/data/<id>.csv` for charts found in a fetched page's raw HTML.

### RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS

Datasets per question. The hero or resolving chart is almost always first in document order; the
Trump tracker carries 5 embeds.

### RESOLUTION_SOURCE_DATAWRAPPER_MIN_HOP_BUDGET_S

The hop runs as a second network phase after the Tier-1 page gather, inside the same 45 s provider
wall, and the datasets share one CDN host so the per-host politeness semaphore serializes them: worst
case `MAX_CHARTS` times the 20 s HTTP timeout is 60 s, past the wall. Bounding the hop at whatever
wall budget remains, and skipping it below this floor, is what keeps a slow CDN tail from cancelling
the whole provider and throwing away Tier-1 pages that already fetched. The floor admits at least one
typical dwcdn fetch: a poll CSV is tens of KB off a CDN, sub-second to about 2 s, the same probe basis
as the HTTP timeout's "0-2s typical". Below it the hop cannot land anything and the pages are worth
more than the attempt.

### RESOLUTION_SOURCE_DATAWRAPPER_HOP_WALL_MARGIN_S

The wall margin the hop leaves the outer `wait_for`, so the inner bound fires first and the provider
returns the pages instead of being cancelled mid-render.

### RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS

A per-dataset render cap, deliberately well under the 6,000-char page cap. A middle-truncated daily
series keeps about 12 rows at each end per 1,000 chars, so 3,000 still carries weeks of values around
today, and the formatter budgets datasets against their own allowance (`MAX_CHARTS` times this) so a
chart's data can never evict the cited page text the section exists to serve.

### RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS

A freshness bound on the dataset's `Last-Modified` against the fetch time. Live trackers republish at
least daily, and the stale-route failure class this guards against served snapshots 5 to 14 months
old as HTTP 200, per the 2026-08-24 verifications. Older or undatable data is withheld
(`stale_data`), never served as live.

### RESOLUTION_SOURCE_CLOCK_SKEW_TOLERANCE

How far ahead of our clock a timestamp a host or the archive gives us may sit before the freshness
guards treat it as unusable rather than as freshest-possible. One constant for both guards, the
Datawrapper dataset's `Last-Modified` and the Wayback rung's capture stamp, because it is one
judgment: tolerate ordinary CDN or host clock skew and nothing more, since past that a future date
means a broken clock or a misparse, and each stamp authorizes a lead that asserts a date to
forecasters. Unlike the two 30-day age bounds around it, which are per-artifact calls about how fast
the underlying data moves, this one is about our own clock.

## Local document text (PDFs read with pypdf, `research/document_text.py`)

Measured 2026-09-03: local pypdf pulled 833,450 chars out of a 6.7 MB 220-page PDF in 5.3 s and the
passage the research driver was looking for was in it, while the paid Gemini `url_context` read of
the same file returned nothing. So a PDF we already hold is extracted and passage-selected locally,
and a model call is spent only on a document we cannot read at all.

### DOCUMENT_TEXT_MAX_PAGES

About twice the 220-page document behind the measurement, so a normal government report reads whole
while a 4,000-page appendix dump stays bounded.

### DOCUMENT_TEXT_MAX_SECONDS

About four times the measured 5.3 s for 220 pages, and it matches `RESOLUTION_SOURCE_HTTP_TIMEOUT`,
so parsing a document costs no more of the research phase than fetching it did.

This is a between-pages checkpoint, not an elapsed bound: `_read_pages` tests it only after each page
returns, so a single page can overrun it and no page is ever interrupted mid-parse, because the
extraction runs in a thread that cannot be cancelled. What bounds one page is the
decoded-bytes-per-stream cap set at import in `research/document_text.py`, which turns a page whose
content stream decompresses past the cap into that page's `''` rather than an unbounded parse.

The clock also starts late. `extract_pdf_text` reads the declared page count and walks the whole
bookmark outline before `_read_pages` computes any deadline, and this budget covers neither: the
measured ceiling is about 16 s for a body built to maximize both, so the real worst case is that
prologue plus this budget plus one page. See `FUTURE.md`, "The PDF parse overruns `max_seconds`".

### DOCUMENT_TEXT_PDF_MAX_BYTES

About six times the measured 6.7 MB file. Above this the parse is not worth a research phase, and the
bytes are refused before pypdf allocates.

### DOCUMENT_DIGEST_TOP_K, DOCUMENT_DIGEST_WINDOW_CHARS

Passages per document, and the window each passage spans. Six passages times 600 chars is about 3.6k
chars, roughly 900 tokens, the same order as one cited page under
`RESOLUTION_SOURCE_PER_URL_MAX_CHARS`. The window is about one paragraph of a report, and is mirrored
as `document_text.DEFAULT_WINDOW_CHARS`, pinned equal by a test.

### URL_CONTEXT_SIZE_GATE_TOKENS

A document we already hold whose estimated token count (chars divided by 4) exceeds this is never
sent to a paid `url_context` read; the digest serves it instead. The nine archived documents above
this bound carried 67% of all reader tokens, and the 833k-char case above is the shape that spends
most: the paid read of it returned nothing, so the spend bought a null answer.

## Page digest (`research/page_digest.py`, the `page_digest_extractor` support role)

A cited HTML page over `RESOLUTION_SOURCE_PER_URL_MAX_CHARS` used to be read from the top, so its tail
was unreachable, and the gap-fill loop's BM25 digest reached the tail with a lexical ranker the operator
does not trust as the primary mechanism. The operator's decision of 2026-09-09 (recorded in
`scratch_docs_and_planning/fetch_ladder_unification_plan_2026-09-09.md`, "The page digest, as agreed")
was a cheap model reading the page and returning verbatim passages, a literal grounding check, and
BM25 kept as the pre-filter and the fallback. The fallback is the digest both callers shipped before,
which is what makes the change strictly safer on the fetch wall. The module is documented in
docs/research.md "Page digest".

### PAGE_DIGEST_EXTRACTOR_MODEL, PAGE_DIGEST_EXTRACTOR_EFFORT

`openrouter/openai/gpt-6-luna` (gpt-5.6-luna -> gpt-6-luna on 2026-09-22, the GPT-6 release; effort
stays medium pending a decision) at reasoning effort `medium`, the operator's choice on 2026-09-09:
"luna is dirt cheap and medium will still be fast enough". `google/gemini-3.8-flash` is the noted
alternative. The slug carries the `openrouter/` prefix because `build_llm_with_openrouter_fallback`
routes the donated-versus-personal key off that prefix, exactly as `FINANCIAL_CLASSIFIER_MODEL` does; a
bare `openai/` slug would dial OpenAI directly on a key this repo does not carry (the plan document
wrote the slug without the prefix, and `GAP_FILL_RESOLVER_MODEL` can omit it only because
`build_native_search_llm` adds it). The resolver probe of 2026-09-09
(`scripts/probes/gap_fill_resolver_probe.py`, three runs on questions 44267 and 45199) priced luna at a
fifth to a quarter of terra's cost per call at low effort (ratios 0.185 to 0.278) and at two-fifths to
a half at the configured medium effort (ratios 0.386 to 0.486, luna at medium against terra at low,
since terra was never probed at medium), with no failed answers in any of the 64 calls.

### PAGE_DIGEST_EXTRACTOR_TIMEOUT_S

The ceiling on the one paid call. The same probe's 16 luna calls at medium effort ran 13.3 to 76.1 s
of wall, median 22.6 s, on prompts of 20k to 84k tokens, and only 6 of the 16 finished inside 20 s;
every one of them also ran a web search, which the digest call does not. The pre-filter below cuts the
page to about 4k tokens before the model reads it, so a digest call should land well under this, but
that no-search latency is an extrapolation and not a measurement: `fallback_used` on the
`RESOLUTION_SOURCE_FETCH` marker is the live reading of how often the ceiling binds once the fetch
ladder wires the digest in. The constant stays at 20 s because the two errors are not symmetric: a
ceiling that binds costs one cheap luna prompt and serves the BM25 digest that already shipped, while a
30 s ceiling would hand up to 32 s of the fetcher's 45 s `RESOLUTION_SOURCE_WALL_TIMEOUT` to one page's
digest on an unmeasured hunch. The call is bounded by `min(PAGE_DIGEST_EXTRACTOR_TIMEOUT_S,
budget_seconds - elapsed - PAGE_DIGEST_WALL_MARGIN_S)`, where `elapsed` is the BM25 thread hop's own
time, so a caller with less wall left than this gets a shorter call and never a longer one.

2026-09-22: 20 -> 30 s (operator). By then the live record was 2 of 5 gpt-5.6-luna digest calls timing out
(2026-09-11), and the 45 s wall no longer looked like the real constraint: `resolution_source` runs concurrently
with AskNews (research-archive latency median 61 s, 10th percentile 44 s) and native search (median 69 s), so its
own wall rarely lengthens the research phase, and the `min(...)` above still clips the call to whatever that wall
leaves. The same day gpt-6-luna ran real digest calls in 1.4 to 4.8 s at low and medium effort, with no
fallbacks (on archived pages cut to 6,000 chars, so shorter than prod's pre-filtered prompt); the OpenRouter
`openai/fast` priority tier made no visible difference. Receipts: `scratch/model_migration_2026-09-22/`.

### PAGE_DIGEST_WALL_MARGIN_S

Left to the caller's outer `wait_for` so the digest returns first and the BM25 fallback, the
presentation cap and the marker line all fit after it. The same 2 s the escalation rungs reserve under
`RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S`; a separate constant because the digest is called from both the
fetcher and the gap-fill loop, whose walls differ.

### PAGE_DIGEST_MIN_CALL_BUDGET_S

No call is made with less than this left after the margin. The fastest luna call in the probe at any
effort was 7.1 s on a 13k-token prompt, so a call handed 3 s of budget would bill its prompt tokens
and time out every time. Below the floor the BM25 digest is served with no paid request, which is
today's behaviour. A call attempted at the floor may still time out; that costs one prompt's tokens
and nothing on the wall.

### PAGE_DIGEST_PREFILTER_MAX_CHARS

About 4k tokens at the chars-over-four estimator, a third of the smallest prompt the probe timed (13k
tokens) and a fifth of the smallest medium-effort one (20k). A page past it reaches the model as its
best BM25 windows for the query (`DOCUMENT_DIGEST_WINDOW_CHARS` each, so 26 of them) in page order,
abutting windows spliced back together and `[...]` marking only a real cut, and a page under it goes
whole. A long page on which no query token occurs at all sends its head instead, which is what a
reader saw before.
The fetch-gap inventory of 2026-09-09 counted 28% of the loop's plain reads hitting the 8,000-char
presentation window, and the median plain read was under 4,000 chars, so the pre-filter fires on a
minority of pages and the model reads most pages complete.

## Resolution-source escalation rungs (free ones: meta-refresh hop, local PDF read)

Every rung runs inside the unchanged 45 s provider wall, and the outer `asyncio.wait_for` discards
every page that already fetched when it fires, so each rung is self-bounding on the same pattern as
the Datawrapper hop: wall minus elapsed minus a margin, skipped below a floor, degrading to whatever
the direct route already got.

### RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S

The margin left to the outer `wait_for` so the rung returns first. The Datawrapper hop keeps its own
historically-named twin, `RESOLUTION_SOURCE_DATAWRAPPER_HOP_WALL_MARGIN_S`.

### RESOLUTION_SOURCE_META_REFRESH_MIN_BUDGET_S

The hop is one more page GET, so it takes the same "0 to 2 s typical" probe basis as the HTTP timeout.

### RESOLUTION_SOURCE_PDF_MIN_BUDGET_S

The floor for the local pypdf parse, which spends CPU rather than network. It doubles as the minimum
`max_seconds` handed to `extract_pdf_text`, where the budget is capped at `DOCUMENT_TEXT_MAX_SECONDS`
above it, and 3 s is about 60% of the measured 5.3 s for 220 pages, so a short document still reads
whole and a long one comes back partial but labelled rather than not at all.

## Resolution-source escalation rungs that need a browser

### RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S

The rendered rung launches headless Chromium in `research/rendered_fetch.py`, shared with the gap-fill
v2 fetch ladder and its process-global `Semaphore(2)` launch cap.

Two reasons its floor is far above the one-request rungs' 3 s. A launch plus a DOM-ready navigation
measured 3 to 8 s across the 2026-09-03 replay corpus even on pages that rendered cleanly, and the
launch slot is contended process-wide, so a question with no budget left would take a slot a sibling
question could still land a page with.

This is the pre-gate floor, read before the render queues on the per-host gate and the launch cap. The
transport's own post-gate need is higher: a 5 s navigation (`RENDER_MIN_GOTO_MS`), the 2 s settle plus
the 5 s DOM-read bound (`RENDER_POST_GOTO_TAIL_MS`, reserved so a goto that runs its budget out can
still be salvaged), and the 3 s exit reserve the rung subtracts from the deadline it hands over
(`RENDER_EXIT_RESERVE_MS`, the shared teardown bound plus a second for the launch and the driver
stop), which is 15 s in all. The floor deliberately sits below that: a render admitted with 12 to 15 s
declines at the gates with an honest `wall_budget` skip rather than launching, and raising the floor
to 15 s would make the pre-gate check truthful at the cost of that band's reach, which is the
operator's call. Left at 12 s pending it.

### RENDERED_DOM_MAX_CHARS

A ceiling on the rendered DOM, the browser rung's counterpart to
`RESOLUTION_SOURCE_MAX_RESPONSE_BYTES` and sized to it. `page.content()` is a string, so this is a
character count taken before anything copies the DOM: the Tier-1 caller encodes it, decodes it back,
ARIA-rewrites it and hands trafilatura a tree several times its size, all while the 100 to 300 MB
browser is still resident, so an 8 MB dashboard DOM was roughly 60 to 110 MB per in-flight
classification. A DOM over the ceiling is declined with the transport's own `RenderDomOverCeiling`,
surfacing as the rung's `render_dom_too_large` skip, kept apart from a missing browser; the harvested
JSON is declined with it.

The name carries no `RESOLUTION_SOURCE_` prefix, for the reason given under
`IMPERSONATE_BROWSER_TARGET` below.

### RESOLUTION_SOURCE_DERIVED_API_MIN_BUDGET_S

The derived-feed GET is one request against a JSON endpoint an earlier render on the same host already
found, so its floor is the meta-refresh hop's, on the same "0 to 2 s typical" probe basis as the HTTP
timeout, rather than the browser floor above. It has its own name because the two rungs are tuned
independently: this one gets cheaper as a run goes on, the browser never does.

## The impersonated retry (`research/impersonated_fetch.py`, shared with gap-fill v2)

Measured 2026-09-04 from a GitHub Actions runner with `scripts/probes/fetch_diagnostic.py`: four
Akamai-fronted federal hosts answered the bot's own aiohttp client 403 and the same GET through
curl_cffi with Chrome impersonation 200. So the refusal is a TLS and HTTP/2 fingerprint verdict and is
recoverable client-side. The rung is free: no key, no model call, no spend.

### RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S

The floor for the retry. It is one GET against a host that just answered us, so it takes the
meta-refresh hop's 3.0 s and the derived-feed GET's 3.0 s, on the same "0 to 2 s typical" probe basis
as the HTTP timeout. Deliberately not the browser rung's 12.0 s: no process is launched, no gate is
contended process-wide, and no model round trip happens.

### RESOLUTION_SOURCE_IMPERSONATE_ENABLED_ENV

A kill switch, on by default in code, unlike the paid rung's default-off, because the rung is free and
bounded by its 403-only trigger, the per-run host memo and the floor above. An explicit "false" in a
workflow yaml turns it off without a code change. Read with `default=True`.

### RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S

A floor under the per-hop timeout that the two SSRF-guarded fetchers derive from the remaining wall
budget, `fetch_ladder.direct_fetch._fetch_one_hop` and the impersonated transport alike. A hop reached with the
budget already spent still gets a token attempt rather than a guaranteed-expired one: nothing
downstream distinguishes "timed out at 0.0 s" from "timed out at 0.5 s", and a fast host answering in
200 ms is a page we would otherwise refuse for free. Small enough that the overshoot stays well inside
`RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S`.

### IMPERSONATE_BROWSER_TARGET

The concrete curl_cffi impersonation profile, pinned rather than the floating "chrome" alias. The alias
resolves to a specific Chrome release inside the library, chrome146 at curl_cffi 0.15.0, so a routine
curl_cffi bump would silently change the TLS and HTTP/2 fingerprint and the User-Agent the federal
hosts see, which can flip the rung's success rate with no code change and makes the 2026-09-04
measurement non-reproducible. Pinning makes a fingerprint change a reviewable diff. No
`RESOLUTION_SOURCE_` prefix because gap-fill v2 dials the same transport, following
`RENDERED_DOM_MAX_CHARS`' precedent.

## Wayback Machine snapshots

The archive is the one free route whose egress is not ours, which is the whole reason it earns a rung.
Measured 2026-09-03: identical client, identical headers, 403 from a GitHub Actions runner and 200 from
a residential address on the same three government hosts.

### RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS

An age bound. A snapshot is admissible as primary grading evidence only with its age disclosed
(operator decision, 2026-09-03) and only inside this bound. Thirty days matches the Datawrapper
freshness guard's, and it is the same judgment rather than a measurement: it was calibrated on
daily-republishing trackers, so a question resolving on a weekly series arguably wants tighter.
Deliberately its own constant, not an alias of the Datawrapper bound, because these are two
independent calls about two different artifacts and tying them would make one impossible to tune.

### RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S

The floor for the snapshot fetch, which is one GET plus one redirect hop. Above the one-request rungs'
3 s because the archive is measurably slower than an ordinary host: the verification probe's own
response carried `LoadShardBlock;dur=1048ms` in its server-timing header, and the redirect means two
round trips through that.

### RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS

Snapshot attempts per question. Every snapshot shares the netloc `web.archive.org`, so Tier-1's
per-host `Semaphore(1)` serializes them: N cited URLs would queue into N sequential archive fetches
behind one gate, inside a 45 s wall that discards every page already fetched when it fires. Two is the
documented trade, so a question whose first two cited sources are both dead gets both tried, and a
question citing five gets its budget protected.

## The one PAID rung: Gemini url_context, last and behind its own flag

### RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV

The rung reaches hosts our client cannot: prod run 33775800806 read bls.gov and sagaftra.org PDFs that
our own fetch 403'd, because Gemini dials from Google's address rather than ours. It is also the only
rung here that spends money and the only one that is model-mediated, since what comes back is an answer
about the page rather than the page, so it defaults off in code and is turned on explicitly per
workflow yaml. It has been on in every bot workflow since 2026-09-04, by the operator's decision.

### RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S

The floor for the read, well above the free rungs' because a `url_context` call is a model round trip
that also fetches: the v2 reader's measured budget is tens of seconds, and below 15 s of remaining wall
this cannot land an answer before the provider's outer `wait_for` discards every page the question
already fetched.

### RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS

One attempt, against gap-fill v2's two. The retry there exists because a 503 UNAVAILABLE returns in
milliseconds and leaves most of a 55 s budget for a second try. Inside a 45 s provider wall shared with
every other cited URL there is no such room, and a second attempt would spend the budget that renders
the pages already fetched.

### RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS

Paid reads per question, the analogue of the Wayback cap and a different quantity from the SDK retry
count above. `_ATTEMPTS` is how many billed requests one read may dispatch; this is how many
`url_context` reads a single question may pay for across its cited URLs. Without it a question citing
several dead sources pays once per source inside the provider wall, and two bounds how much a single
question can spend when the flag is on. Its own constant so the two paid knobs tune independently.

### RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS

How much of a withheld `url_context` reply reaches the run log. A read we paid for and then discarded is
otherwise unauditable: the `not_addressed` sentinel means the model says the page does not discuss the
ask, and nothing on the record distinguishes that from the model dutifully reading a bot-challenge page
it was served instead, which is the question the smoke run could not answer for sagaftra.org. A few
hundred characters is enough to tell those apart and short enough that the discarded answer cannot flood
a log or read as evidence.

## Gemini Search Provider (Google AI Studio direct SDK)

The provider uses the google-genai SDK with the `GoogleSearch` grounding tool for first-party Google
Search results, which is distinct from OpenRouter's Exa-backed `:online` plugin. It adds a genuinely new
search index to the ensemble.

### GOOGLE_API_KEY_ENV

`GOOGLE_API_KEY` is the operator's personal Google AI Studio key. In CI it is stored as
`secrets.GEMINI_API_KEY` and surfaced as `GOOGLE_API_KEY` for the google-genai SDK. The grounded-search
provider always reads this, because Google AI Studio offers no donated or shared-key path.

### GEMINI_USE_DONATED_OPENROUTER_KEY_ENV

A toggle for OpenRouter Gemini routing only. It controls whether models such as
`openrouter/google/gemini-3.1-pro-preview` flow through the Metaculus-donated OpenRouter key
(`OAI_ANTH_OPENROUTER_KEY`) with paid-key fallback, or skip the donated wrapper entirely and route
through the operator's personal `OPENROUTER_API_KEY`. It does not affect the google-genai grounded-search
provider, which always uses the personal `GOOGLE_API_KEY`.

Default on since 2026-06-16: after Metaculus raised the Google rate limits, the donated key serves most
Gemini models, including gemini-3.5-flash and gemini-3.1-flash-lite. The known exception is
gemini-3.1-pro-preview, which is pinned to the personal key via the `DONATED_KEY_BLOCKED_GOOGLE_MODELS`
blocklist, with no donated attempt and no 429, pending the Metaculus-side BYOK fix; see
`TODO(gemini-3.1-pro-donated)` in `fallback_openrouter`. The full reasoning is under
`gemini_use_donated_openrouter_key` above.

### GEMINI_SEARCH_DEFAULT_MODEL

The grounded-search model, verified live on the native google-genai SDK on 2026-09-03 with three calls
from `scripts/probes/gemini_verify.py`: the response reported `model_version` gemini-3.8-flash, the
`google_search` tool returned a web search query, `thinking_level` was accepted,
and `url_context` retrieved a robots-allowed host. Grounding still needs billing enabled on the Google AI
Studio project.

Price, read off ai.google.dev on 2026-09-03: $0.75 in and $3.75 out per 1M through 2026-12-31, then $1.50
and $7.50. That is against $0.50 and $3.00 for the gemini-3-flash-preview this replaced, which is now
labelled a legacy preview.

Override via the `GEMINI_SEARCH_MODEL` env var. The old note that a free-tier project can point that at
gemini-2.5-flash holds only with the `thinking_level` caveat below read first, since the 2.5 line takes a
`thinking_budget` and would reject the request as it is built today.

No temperature, top_p or max_tokens overrides are set: the google-genai SDK defaults apply. Gemini 3 Flash
is a thinking model, Google's defaults are tuned for it, and capping either caused silent truncations in
the past.

### GEMINI_SEARCH_TIMEOUT

Six minutes. Automatic Function Calling can chain up to 10 tool round trips internally (search, model, URL
fetch, model, and so on), each about 15 to 20 s. A full 10-round chain takes 150 to 200 s, so 180 s was too
tight and produced observed timeouts on legitimate deep-research calls; 360 s gives twice the headroom over
the worst-case AFC chain. Gap-fill runs overlap with forecaster LLM calls, so a higher timeout adds zero
wall-clock cost. The observed p99 of non-AFC calls is about 52 s.

### GEMINI_SEARCH_LINK_RESOLVE_TIMEOUT_S

Ten seconds is the per-call wall for resolving all cited Google search redirect links after the Gemini
response arrives. The resolver runs all unique links concurrently through the existing HTTP transport,
with a session total timeout. The provider uses `min(GEMINI_SEARCH_LINK_RESOLVE_TIMEOUT_S, remaining)`
where `remaining` is what is left of `GEMINI_SEARCH_TIMEOUT`; if no wall remains, it skips resolution
and treats every cited link as unverified.

### GEMINI_SEARCH_THINKING_LEVEL

The thinking level for the grounded-search call, set explicitly by operator decision on 2026-09-03 rather
than left at the model's default, which for gemini-3-flash-preview is high: 71% of the grounded-search
output tokens measured in the 2026-09 spend reconstruction were thinking tokens, and this is a retrieval
plus summarise task rather than a reasoning one.

Only the level is set. The no-max_tokens rule above still holds, because capping output on a thinking model
is what caused the silent truncations. Re-pointing `GEMINI_SEARCH_MODEL` at the 2.5 line therefore also
means editing the `gemini_thinking_config(...)` call in `research/gemini_search.py` to send a
`thinking_budget` (1 to 24,576 on 2.5 Flash) or no `thinking_config` at all; there is deliberately no
model-family gate in code.

### GEMINI_SEARCH_HTTP_TIMEOUT_MS, GEMINI_SEARCH_HTTP_ATTEMPTS

The client-side per-attempt HTTP timeout in milliseconds, and the attempt count including the first, for the
grounded-search client. The SDK retries nothing by default, so a fast transient (the 503 UNAVAILABLE that
killed two production calls) used to lose the whole provider, and one retry recovers it.

The worst-case arithmetic leaves the hard bound unchanged. The outer
`asyncio.wait_for(..., GEMINI_SEARCH_TIMEOUT)` in `research/gemini_search.py` still cancels the whole call
at 360 s, and it genuinely can, being an async coroutine unlike `read_document`'s thread. So the nominal
product of 350 plus at most 2 (one jittered retry sleep) plus 350 never elapses: a first attempt that hangs
eats the window and the outer `wait_for` fires exactly as it does today, while a retry only completes when
the first attempt failed fast, which is the recovery case this exists for.

Why the per-attempt cap is not sized so the product fits under 360 s: that would need at most 176 s per
attempt, and the `GEMINI_SEARCH_TIMEOUT` entry above records legitimate AFC chains at 150 to 200 s, with
180 s too tight and observed timeouts. Shrinking the per-attempt allowance would newly fail calls that
succeed today, which is the one thing a timeout change here must not do. 350 s instead sits just under the
outer deadline, so nothing a single attempt can do today is cut short.

### GAP_FILL_V2_READER_THINKING_LEVEL, GAP_FILL_V2_READER_HTTP_ATTEMPTS

The same two settings for gap-fill v2's `read_document` backend, whose thinking level is a tier lower by
operator decision on 2026-09-03: quoting a fetched document back is the least reasoning-heavy of the Gemini
calls.

The reader's per-attempt timeout is not a constant here because it is derived from the total in-thread
budget, as `_READ_DOCUMENT_HTTP_TIMEOUT_MS` in `research/agentic/tool_backends.py`, where the arithmetic
lives. That call runs under `asyncio.to_thread`, so its outer `wait_for` cannot cancel it and the retry has
to fit inside today's budget rather than beside it.

## Second-pass gap-fill

After first-pass research completes, a cheap analyzer identifies up to `GAP_FILL_MAX_GAPS` factual gaps and
each is resolved by a parallel OpenAI native web search, see `GAP_FILL_RESOLVER_MODEL` below. The whole pass
fails soft: the forecast proceeds with first-pass research alone if any stage errors out.

### GAP_FILL_ANALYZER_MODEL

Non-grounded gap-listing. It reads the first-pass research and emits a JSON list of up to
`GAP_FILL_MAX_GAPS` factual gaps under the tight `GAP_FILL_ANALYZER_WALL_TIMEOUT` cap, which soft-fails
silently on breach, so low effort is the latency-safe choice: the task is decomposition rather than deep
judgment. Grounded search resolution still uses google-genai directly via `gemini_search_provider`, because
that path needs the search index. Changed 2026-09-22, terra to `gpt-6-sol`: GPT-6 shipped with no Terra
successor, so every Terra role moved to Sol 6 at the same (low) effort.

### GAP_FILL_MAX_GAPS

Lowered from 5 to 4 on 2026-07-20. The transcript vibe-analysis (Fable) found no positional value cliff, so a
fixed cutoff is safe now that the analyzer prompt ranks gaps by decision-relevance: see
`gap_fill_analyzer_prompt`, where the fourth slot holds the least valuable of the kept gaps rather than a
random one. Three was still judged unsafe, but the fifth gap is empirically a completeness stretch, since only
about 27% of 221 archived bundles rendered a fifth gap and the observed fifth gaps were confirmatory, so 5 to
4 is near-zero risk. Do not go below 4.

Since 2026-09-09 the cap applies to the gaps that survive the grade triage (`research/targeted.py`
`triage_gaps`, `docs/research.md` "v1 triage"), not to the analyzer's raw list, so a gap dropped as
future-dated, already answered by the first pass, or a restatement never displaces a kept one. The analyzer
is still asked for at most this many, so the cap binds only when it over-lists; a survivor past it is counted
as `dropped_over_cap` on the `GAP_FILL_V1_TRIAGE` marker. The 2026-09-09 cost pass confirmed the "do not go
below 4" rule from the other side: a positional cap of 2 would have dropped the useful gap on 4 of 6 traced
questions, which is why the lean-out is by grade rather than by count.

### GAP_FILL_ANALYZER_TIMEOUT

The analyzer call is non-grounded, with no Google Search, and should return quickly. A tight timeout prevents
a single hung analyzer request from holding a research concurrency slot for the full grounded-search budget.

### GAP_FILL_ANALYZER_WALL_TIMEOUT

A wall-clock backstop for the analyzer call, with slight headroom over `GAP_FILL_ANALYZER_TIMEOUT` so the
cleaner per-request error from litellm fires first when possible (auth failure, model-not-found and the like).
Same pattern as `NATIVE_SEARCH_WALL_TIMEOUT` against `NATIVE_SEARCH_TIMEOUT`. Without the headroom,
`asyncio.wait_for` and the litellm request timeout fire at the exact same second and we lose the descriptive
error message.

### GAP_FILL_MIN_RESEARCH_CHARS

Skip gap-fill when the first-pass research blob has fewer than this many non-whitespace characters, which
likely means all providers soft-failed and gap-fill would just hallucinate gaps or burn quota.

### GAP_FILL_RESOLVER_MODEL, GAP_FILL_RESOLVER_REASONING_EFFORT

On 2026-06-25 the per-gap resolver migrated off direct-Google grounded Gemini (google-genai on the personal
`GOOGLE_API_KEY`) to OpenAI native web search via OpenRouter, which bills the Metaculus-donated key. The
resolver fanned out up to `GAP_FILL_MAX_GAPS` parallel grounded calls per question, a per-gap cost multiplier
on the personal Google bill and the dominant unwanted spend. The single first-pass grounded Gemini call stays
on google-genai: the operator is fine paying for one call per question, and it uses `url_context`, which
OpenRouter cannot replicate. There is no `openrouter/` prefix on the model id here because
`build_native_search_llm` adds it.

This is agentic single-gap web research whose source-trust judgment lands directly in every forecaster prompt.
The workers run in parallel under `NATIVE_SEARCH_WALL_TIMEOUT`, so latency is the slowest call rather than the
sum, and effort stays low.

Changed from sol to terra on 2026-07-20. Terra was preferred or within noise against sol across all three
2026-07 blind role audits at roughly 40 to 50% lower cost, and these searches are about 44% of research spend
(17 calls in the 2026-07-19 run), the single biggest research line item, so the cost cut is the dominant
consideration. The 2026-07-09 bench had sol-low matching terra-low coverage 24 of 25; the blind audits plus the
cost weight flip it. Changed again 2026-09-22, terra to `gpt-6-sol`: GPT-6 shipped with no Terra successor, so
every Terra role moved to Sol 6 at the same (low) effort.

## Agentic gap-fill v2 (bounded research loop)

Second-generation gap-fill: a bounded agentic loop in `metaculus_bot/research/agentic/` that dry-runs the
panel's own forecasting template to identify fill, verify and resolution targets, then pursues them with
search, fetch and read tools. It runs concurrently with v1 during the overlap window, when both flags are on,
and soft-fails to `""` like v1. See `scratch_docs_and_planning/agentic_gap_fill_v2_plan.md` and
`docs/agentic_gap_fill.md`.

### GAP_FILL_IMAGE_MAX_SOURCE_PIXELS, GAP_FILL_IMAGE_MAX_EDGE, GAP_FILL_IMAGE_MAX_PIXELS, GAP_FILL_IMAGE_MAX_BYTES

Bounds on raster images normalized for the same gap-fill driver: at most 25 megapixels decoded from the
source, then at most 2,048 pixels on the longest side, 2 megapixels total, and 2 MiB for the normalized PNG.
The download still uses the shared 5 MiB response cap. An explicit crop can reduce an image to fit the output
limits, provided its full source fits the download and decode limits. The normalizer never crops automatically
or enlarges a small image.

### GAP_FILL_IMAGE_MAX_VIEWS

At most four distinct normalized PNG hashes are delivered per question, counting crops as views. Repeated URLs
or crops that normalize to an already delivered image reuse the existing image ID and do not consume another
slot.

### GAP_FILL_IMAGE_MAX_LEADS, GAP_FILL_IMAGE_LEADS_MAX_CHARS, GAP_FILL_IMAGE_METADATA_MAX_CHARS

The HTML image-lead scanner keeps at most three candidate URLs per page. Each URL is capped at 1,000 characters;
each untrusted alt text or figure caption is capped at 160 characters. The leads are navigation metadata only;
they do not say the pixels were fetched or inspected.

### GAP_FILL_V2_TOOL_BUDGET_LINE_RESERVE_CHARS

Dispatch reserves up to 512 characters inside each tool reply for its remaining-budget line. It subtracts that
space before truncating the tool body and clips the line to the space actually available, so it remains inside
`LoopConfig.max_result_chars`.

### GAP_FILL_V2_DRIVER_MODEL, GAP_FILL_V2_DRIVER_EFFORT

Driver model and effort picked by the blind five-arm replay eval on 2026-07-17,
`scratch/driver_replay_2026-07-17/blind_judge_report.md`: terra-low ranked first (fetch-verified grounding,
best source mix, 30 s wall, $0.36 per question), terra-medium second; sol-low burned budget on near-duplicate
searches and came fifth; sonnet-5 cited unfetched URLs, which is disqualifying for a researcher. All candidates
were openai or anthropic, so the loop's litellm binding routes via the donated OpenRouter key. Changed
2026-09-22, terra to `gpt-6-sol`: GPT-6 shipped with no Terra successor, so every Terra role moved to Sol 6 at
the same (low) effort default.

### GAP_FILL_V2_READER_MODEL

The `read_document` backend model on the native google-genai path,
`research/agentic/tool_backends.py._run_document_read_sync`. Verified live on that SDK on 2026-09-03 with
`scripts/probes/gemini_verify.py`: `url_context` retrieval succeeded on a robots-allowed host and
`thinking_level` was accepted, so the old caution that this id was unverified on the native AI Studio API is
retired.

It is the same id the grounded-search provider runs, at $0.75 in and $3.75 out per 1M through 2026-12-31 then
$1.50 and $7.50 (ai.google.dev, read 2026-09-03), against $1.50 and $9.00 for the gemini-3.5-flash it replaces.

A wrong id still soft-fails `read_document` (model-not-found becomes an error outcome), silently disabling the
directed-reading rung. `url_context` is also robots-gated, which is a separate cause of the same symptom: the
same probe got `URL_RETRIEVAL_STATUS_ERROR` on a host whose robots.txt disallows Google-Extended, so a refusal
can be the host's policy rather than a bad id.

### GAP_FILL_V2_MAX_TOOL_CALLS

Parallel tool calls each count against the cap. Steps are where latency lives, so batching is encouraged rather
than rationed. Raised with the W2 ambition floor on 2026-07-21: v2 runs 41 to 60 s of the
`GAP_FILL_V2_WALL_DEADLINE` budget, so the headroom is free. The satisficing problem was ambition rather than
budget, and with the conclude-gate floor in place the extra slots let the driver dig deeper on the few
decision-relevant gaps instead of stopping early.

### GAP_FILL_V2_WALL_DEADLINE

A hard wall for the whole loop, inside v1's worst-case envelope (`GAP_FILL_ANALYZER_WALL_TIMEOUT` then the
resolver wave under `NATIVE_SEARCH_WALL_TIMEOUT`), so running v2 concurrently with v1 adds no research-phase
wall-clock. The loop is anytime: hitting the deadline emits banked findings, never `""`.

### GAP_FILL_V2_CONCLUDE_THRESHOLD

With fewer than this many seconds remaining, the harness rejects every tool except conclude, forcing the loop to
wrap up inside the wall deadline.

### GAP_FILL_V2_MIN_CONTENT_CHARS

Below this many extracted chars, the fetch ladder escalates plain HTTP to headless-Chromium rendering. It is the
JS-wall heuristic, consumed by `tools.py`.

### GAP_FILL_V2_MAX_GAPS

The maximum ranked gaps the driver's `set_research_plan` tool may register (W1). An independent knob from v1's
`GAP_FILL_MAX_GAPS`: v2 gaps are cheap, since there is no dedicated per-gap search call and the driver works one
shared tool budget, but a focused work-list still beats a sprawling one. The gap list is ranked by
decision-relevance, so the cap drops the least forecast-moving gaps.

## Financial Data Provider

### FINANCIAL_CLASSIFIER_MODEL, FINANCIAL_CLASSIFIER_TIMEOUT

Binary-ish routing classification, asking whether this is a financial or economic question, under a 30 s
timeout. The task is capability-saturated, so it rides the cheapest capable tier: mini to luna on 2026-08-03,
when luna's markdown made it cheaper than mini. luna -> GPT-6 luna on 2026-09-22 (the GPT-6 release), same tier
logic.

### FINANCIAL_YFINANCE_LOOKBACK_DAYS, FINANCIAL_YFINANCE_RECENT_DAYS

The calendar-day lookback behind every yfinance `history()` fetch. Both paths, live and backtest, fetch by
explicit start date equal to `as_of` minus this many days, end-inclusive, so the window holds LOOKBACK plus 1
calendar dates.

It is never spent as a bare `period="Nd"`. Yahoo's chart API reads that custom range as N trading bars for
listed assets but roughly N calendar dates for 24/7 ones, one integer under two unit systems, which is how the
listed-asset backtest margin was never sized at all.

Sized so the deepest consumers clear on both daily-bar bases, with real headroom. On the 365 basis (24/7
markets) the "1y" return needs an observation at least 366 days back and the 52-week slice wants 365 rows, so
391 dates leave about 25 Yahoo gap-days of tolerance; the old 372 left 6, and a persistent one-day BTC-USD hole
has been observed live. On the 252 basis (exchange-traded) the "1y" return reaches about 365 calendar days back
and the 52-week slice wants 252 bars, so 391 dates is about 265 bars at the worst observed NYSE density (253
bars per 373-date window, measured over three years of real SPY windows), about 12 bars of margin where the old
372 measured margin exactly zero.

### FINANCIAL_VARIANCE_RATIO_LAG, FINANCIAL_VARIANCE_RATIO_FLOOR, FINANCIAL_VARIANCE_RATIO_MIN_RETURNS

A variance-ratio screen for a vendor-noise-dominated daily series. On q44797, USD/SZL's 17.8% "volatility" was
79% quote noise on a pegged cross, and all six forecasters sized their intervals off it. VR(q) near 1 is a
random walk; well below 1 means each day's move is largely reversed the next, which is what a thin quote on a
fixed cross looks like and what cancels over multi-day windows.

The lag is 5, one trading week, because it is the horizon where the two cases separate: the 44797 verification
(section 11) measured VR(5) at 0.472 on the noisy series against 0.740 on the clean anchor, while at VR(10) the
clean series read 0.208, leaving no separation.

The floor of 0.6 sits between those two, and is calibrated against seeded fixtures in
`tests/test_timeseries_anchor_provider.py` (`TestVarianceRatio`) rather than against the receipt's own numbers,
which came from a differently-parameterised estimator.

The minimum of 120 returns is set because the null standard error of VR(5) is about `sqrt(4.8/n)`: about 0.20 at
n=120 and about 0.40 at n=30, so the 30-row vol window cannot carry this statistic. The provider's own
`FINANCIAL_YFINANCE_LOOKBACK_DAYS` window holds about 265 daily bars, so a normal fetch clears the floor with
room and a short or gappy one gets no flag at all.

### FINANCIAL_FRED_VINTAGE_PRINTS

How many recent FRED prints the first-release-versus-current-vintage table covers. Revising macro series resolve
on the first print (q44944 resolved on first-release Case-Shiller) while the levels rendered beside it are
today's revised vintage, so the gap between the two is a forecastable, signed quantity. Four prints is enough to
read a revision direction on a monthly series without turning the block into a table nobody reads.

### MAX_FINANCIAL_IDENTIFIERS

A cap on how many tickers plus FRED series one question may fetch. The identifier list is whatever an LLM
classifier named plus whatever URL extraction found, and it was previously unbounded. Each identifier gets its
own `asyncio.to_thread`, all of them landing in the process-wide default executor that every other blocking call
shares (`ts_fetch`, `resolution_source`, the agentic fetch ladder, the `/auth/key` probe). Tasks queued behind a
saturated pool burn their `wait_for` budget without executing, so an over-eager classification on one question
degrades unrelated providers on others. Twelve is well above any plausible real question, since the classifier
prompt asks for the resolving series rather than a sector sweep, while bounding the worst case.

## SEC EDGAR client (`research/sec_edgar.py`; standalone, not yet a ladder rung)

### SEC_EDGAR_CONTACT_EMAIL_ENV, SEC_EDGAR_USER_AGENT_TEMPLATE

SEC's fair-access policy (sec.gov/os/webmaster-faq, "Developers", read 2026-09-09) asks automated clients to
"declare your user agent in request headers" in the form `Sample Company Name AdminContact@<sample company
domain>.com`, and the 403s sec.gov returned to the bot's browser-shaped fetches (12 blocked events over 5
questions in the archived gap-fill v2 transcripts) are that policy, not an anti-bot wall. The template carries
the identity half; the contact half is read from the env var when a session opens, never committed, and an unset
value makes `edgar_session()` raise before any socket opens. Fail shut rather than fall back to an anonymous
User-Agent, because sending one is exactly what the policy forbids and what gets an address blocked.

### SEC_EDGAR_MAX_REQUESTS_PER_SECOND

The same FAQ page: "our current maximum access rate is 10 requests per second", monitored per source. The client
spaces request STARTS process-wide across all three EDGAR hosts (www, data, efts) through one loop-scoped spacer,
so concurrent questions share the budget. Eight rather than ten leaves margin for `asyncio.sleep` granularity and
for the odd request another path in the process might make to sec.gov; the ceiling is SEC's and the margin is
ours, so a future measurement showing headroom can raise this toward ten.

### SEC_EDGAR_MAX_RESPONSE_BYTES

Measured 2026-09-09 with a declared User-Agent: JPMorgan's `companyfacts` JSON is 7.9 MB decompressed, Oracle's
inline-XBRL 10-K primary document (`orcl-20260531.htm`, cited by question 45199) is 6.9 MB, Uber's 10-K 3.2 MB,
a `frames` response for `Revenues/USD/CY2025` 341 KB, `company_tickers.json` 797 KB. The 5 MiB page cap the
resolution-source fetcher uses (`RESOLUTION_SOURCE_MAX_RESPONSE_BYTES`) would therefore drop both the facts of a
large filer and the very filings a question cites, so this client has its own cap at 16 MiB, still streamed
through `read_body_capped` so peak memory during the read is bounded by it.

## Soft deadlines to keep batch wall-clock inside the tournament cron window

### FORECASTER_SOFT_DEADLINE

A per-forecaster outer deadline, wrapped via `asyncio.wait_for` around each `_make_prediction` call. A single
stuck forecaster used to be able to hold a question for `REASONING_MODEL_CONFIG`'s litellm timeout times its
`allowed_tries` (see `llm_configs.py`); this caps that worst case, at which point the forecaster is dropped with
a loud WARNING and the other models carry the ensemble.

### MIN_FORECASTERS_TO_PUBLISH

The minimum number of successful base forecasters required to publish a question. Below it the question is
skipped entirely rather than publishing a weak ensemble.

Lowered to 1 on 2026-07-20, having gone 3 to 2 to 1 over that day. A threshold equal to the roster width
tolerates zero drops, each step below it tolerates one more, and 1 accepts publishing on a single surviving
forecaster. The operator accepts a single-forecaster publish: median-of-1 is the forecast itself, and
exception-driven drops stay CI-visible, counted as degradation since 687e113, so a degraded run, even one thinned
to a lone model, still reddens CI rather than silently withholding the question.

Note that `forecaster.py` short-circuits the n==1 case before spread computation and stacking, because the
`spread_metrics` helpers require at least 2 predictions and raise otherwise; see the single-forecaster guard in
`_research_and_make_predictions`.

### METACULUS_CLOSE_WINDOW_SECONDS

The Metaculus close window each scheduled run has to finish inside. The prod crons fire hourly, so a question's
whole forecast-and-publish cycle must fit in one hour. It is named because two deadlines below are sized against
it and the arithmetic was previously a bare 3600 living only in prose.

### PER_QUESTION_WALL_CLOCK_DEADLINE

A per-question wall-clock cutoff, sized just inside `METACULUS_CLOSE_WINDOW_SECONDS`. At the deadline, in-flight
forecasters are cancelled, we base-combine whatever completed (at least `MIN_FORECASTERS_TO_PUBLISH`) and submit.
The remainder is exactly `WALL_CLOCK_STACKING_MIN_BUDGET`, which is not a coincidence: it reserves time for the
stacker skip plus publish, see `PUBLISH_POST_TIMEOUT` and `PUBLISH_POST_RETRIES`. `tests/test_llm_retry.py` pins
both relationships.

### WALL_CLOCK_STACKING_MIN_BUDGET

Below this remaining-budget threshold, skip stacking and force `fallback_median` aggregation. Sized to clear the
publish-hardening worst case, which is the prediction POST plus the comment POST, each up to
`PUBLISH_POST_TIMEOUT * (PUBLISH_POST_RETRIES + 1)`, plus headroom.

## Close-aware per-question time budget (`metaculus_bot/time_budget.py`)

`PER_QUESTION_WALL_CLOCK_DEADLINE` above is sized against the cron period, not against a question's own deadline,
so on its own it lets a question closing in 20 minutes spend 58.5. The constants in this section size the
close-derived budget that bounds it.

### PUBLISH_RESERVE_SECONDS

Time held back from the budget so the prediction POST can still land. forecasting-tools'
`_post_question_prediction` opens with one `_sleep_between_requests` of 3.5 to 4.5 s before the POST, and
`publish_hardening` bounds that POST at `PUBLISH_POST_TIMEOUT * (PUBLISH_POST_RETRIES + 1)`, which is 40 s. Sixty
leaves about 15 s of slack.

It is deliberately smaller than `WALL_CLOCK_STACKING_MIN_BUDGET`, which reserves for both POSTs: only the
prediction has to beat the close, and a comment posted a few seconds late is still accepted.

### TIME_BUDGET_FAST_PATH_THRESHOLD

Below this effective budget, drop the optional research stages, meaning every provider but the primary plus both
gap-fill passes, and publish on the fast path.

Sized at exactly the full pipeline's configured worst case: research 1155 s (the provider phase is 600, being
AskNews 300 plus summarizer 300 sequentially inside one provider, then gap-fill 555, being analyzer 135 plus
resolver wave 420), plus `FORECASTER_SOFT_DEADLINE` 600, plus the publish tail 60, which totals 1815 s. So the
rule reads: stop running the optional stages once the full pipeline's worst case no longer fits the window. The
value is that sum, so there is no band where the envelope does not fit but the fast path stays off.
`SUMMARIZER_WALL_TIMEOUT` is defined further down the file, which is why the sum is stated rather than spelled as
an expression.

The measured false-positive cost is zero: 0 of 99 published triple-era questions had less than 54 minutes of
headroom at run start (`scratch/residual_2026-08-24/time_budget_design.md`), and the optional stages are worth
84 s (gap-fill v2 p50) to 183 s (dropping the provider tail at its observed max).

### TIME_BUDGET_MIN_VIABLE_S

Below this budget the question is skipped at intake rather than run on the fast path. The minimum viable path is
the primary provider (measured worst about 110 s live; the research phase's half-share of a 300 s budget is
150 s) plus at least one reasoning forecaster (typical completions run 100 to 300 s against
`FORECASTER_SOFT_DEADLINE` 600, and the other half-share of 300 s fits only the fastest of them).

Below about 5 minutes even that path essentially never lands: the fan-out produces 0 valid forecasters and the
min-forecasters guard drops the question after spending. So the intake skip converts guaranteed-wasted spend into
an immediate forfeit with a log line naming the close time.

### RESEARCH_PHASE_BUDGET_SHARE

The fraction of the total budget granted to the research phase as one fixed window anchored at the budget's start
(`research_phase_deadline_s = total * share - elapsed`), enforced as a deadline on the parallel-provider phase
and on each gap-fill pass.

Fixed rather than a rolling share of remaining: research consults the deadline at two sequential points, and
re-taking 50% of remaining at each compounds to about 75% of the budget, leaving the fan-out under its own soft
deadline on the close-limited band this budget exists for. The fixed window guarantees forecast-and-publish the
complementary half, and a slow intake spends research's half rather than the forecast's.

At the static 3510 s budget the window is about 1755 s, well above research's 1155 s configured worst case, so it
never fires on a roomy question; at a close-limited 2400 s budget it splits 1200 s research and 1200 s
forecast-and-publish.

### PUBLISH_POST_TIMEOUT, PUBLISH_POST_RETRIES

The per-publish-POST timeout, covering `post_binary`, `post_numeric`, `post_mc` and `post_question_comment`. Stock
forecasting-tools uses synchronous `requests.post` with no timeout, so a hung server can block the whole batch
indefinitely. `publish_hardening.py` wraps each POST on a `concurrent.futures.ThreadPoolExecutor` with a
`Future.result(timeout=...)` cap and also monkey-patches `requests.post` on the forecasting-tools module to inject
a request-side socket timeout, so the underlying socket actually closes when the server stalls. It retries once on
a timeout or connection error.

### FETCH_GET_TIMEOUT, FETCH_GET_RETRIES, FETCH_GET_BACKOFF_BASE, FETCH_GET_BACKOFF_JITTER

Fetch hardening: retry and timeout for question-list GETs to the Metaculus API. Stock forecasting-tools issues
`requests.get` with no timeout and no retry, so a single transient 403, 429 or 5xx anywhere in the question
pagination kills the whole CI run. Observed 2026-05-19: a CDN or WAF-style 403, with a 33 s stall and a generic
"API only available to authenticated users" body, on a healthy key. `fetch_hardening.py` wraps
`_get_questions_from_api`, the single chokepoint for every question-list GET, with a request-side socket timeout
plus a bounded retry on retryable statuses and connection-level errors.

The backoff is sized for the realistic failure mode: a CDN or WAF edge-node overload typically clears in 10 to
60 s, not 1 to 3 s. The observed 2026-05-19 incident had a 33 s server-side stall before the 403, so a backoff in
the 10 to 25 s range gives the edge layer time to recover. The cost of waiting is about zero, since the tournament
fetch is on a 20-minute cron with roughly 40 minutes of total budget, while the cost of retrying too soon is
hitting the same wall and burning the run.

### STACKER_SOFT_DEADLINE, STACKER_FALLBACK_SOFT_DEADLINE

The stacker soft deadline, set slightly above the stacker LLM's own litellm timeout (`REASONING_MODEL_CONFIG` in
`llm_configs.py`) so the model's timeout fires first with a clean exception when possible. This `wait_for` is a
final belt-and-suspenders backstop for a wholly stuck call. The stacker is configured with `allowed_tries=1` in
`llm_configs.py`, so we get one try before falling back.

The fallback model's deadline is tighter because we are already running late on the critical path by the time the
fallback fires.

### CRUX_SOFT_DEADLINE

A per-question soft deadline for the disagreement-crux extractor (`DISAGREEMENT_ANALYZER_LLM` in
`llm_configs.py`). It caps the worst case on the conditional-stacking critical path: the analyzer's own bound is
`UTILITY_MODEL_CONFIG`'s litellm timeout per attempt, which is looser than this, so without the wrapper a stalled
call runs well past the crux's usefulness.

### SUMMARIZER_WALL_TIMEOUT

A wall-clock cap for the AskNews summarizer invoke. The summarizer is set `allowed_tries=1` in `llm_configs.py`
and wrapped in the broad elapsed-gated retry, which previously had no wall guard at all. It matches the
summarizer's litellm per-request timeout (`UTILITY_MODEL_CONFIG` in `llm_configs.py`) so the per-attempt cap
aligns with the underlying request budget. On breach the summarizer soft-fails to the raw AskNews articles rather
than hanging the question.

## Benchmark driver tuning

### TYPE_MIX

The distribution mix over question types, ordered as (binary, numeric, multiple_choice).

## BACKTEST SETTINGS

### LEAKAGE_DETECTOR_MODEL

A mechanical leakage screen over research text, backtest-only. The task is saturated, so luna is the cheapest
capable tier: mini to luna on 2026-08-03; luna -> GPT-6 luna on 2026-09-22. Same day, the detector's `max_tokens=500`
cap was removed and its effort raised to `high` (briefly `max` the same day): this backtest-only screen is not time-sensitive, and a max_tokens
cap crashes calls for no good reason since reasoning tokens count against it (see `metaculus_bot/backtest/leakage.py`).

## Per-type stacking gates

### BINARY_STACKING_ENABLED_ENV, MC_STACKING_ENABLED_ENV, NUMERIC_STACKING_ENABLED_ENV

Each question type has an independent enable and disable flag. All three default to disabled: see the gate in
`main.py`, which reads them via `env_flag_enabled(..., default=False)`. A deploy opts a type back into stacking by
setting `<TYPE>_STACKING_ENABLED=true` in its env.

Background: ablation showed the stacker hurts numeric CRPS (median beats stack, p=0.042), so the numeric disable
is evidence-backed. Binary was a tie (p=0.496), so binary and MC are off as a low-risk default, being a
tie-at-best plus compute, and unmeasured on the current stack. Revisit after a prod-ish ablation or marker-era
resolutions; see `scratch_docs_and_planning/prod_ish_ablation_plan.md`.

## Prediction-market provider (Workstream G)

### PREDICTION_MARKETS_ENABLED_ENV

Env-gated. Resolved markets on all three platforms retain their last-trade price after resolution, so without the
`as_of` filter in `fetch_market_snapshot`, pulling a market for a resolved Metaculus question leaks
post-resolution pricing into the rationale. That is why the provider is hard-disabled under
`is_benchmarking=True`.

It is on in all prod workflows as of commit 3c12dbe, and prod runs with `is_benchmarking=False`, so the guard does
not suppress it there. The benchmarking guard means the standard `make backtest_*` gate cannot measure its
forecasting value: it was validated via the manual `test_bot.yaml` prod-mode run plus opt-in live integration
tests instead. See `atlas_inspired_improvements.md` section G.

### PREDICTION_MARKET_TIMEOUT

An outer wall-clock timeout for the full prediction-market snapshot: both LLM stages, the catalogue pulls, the
venue fan-out and the Manifold detail enrichment. It runs inside `asyncio.gather` alongside the other research
providers, so raising it adds no wall-clock time to the research phase; for scale,
`NATIVE_SEARCH_WALL_TIMEOUT` is 420, Gemini 360 and AskNews 300.

150 is the ranked pipeline's 131.5 s worst case (the table under the next section) plus margin. It was 30.0 under
the keyword and fuzzy design, which a roughly 36k-token ranking call and a full catalogue pull do not fit inside.
`prediction_market.SNAPSHOT_STAGE_BUDGET_S` recomputes that worst case from these constants and WARNs loudly at
provider init when an env override sits below it, so a stale `PREDICTION_MARKET_TIMEOUT=30` in a `.env` cannot
masquerade as a generic snapshot timeout.

## Ranked market retrieval (the two LLM stages and the catalogue pull)

Wall caps and backoff ladders for the snapshot's two LLM stages, plus the bounds on the full Kalshi events
catalogue pull. Each stage's worst case is `(len(backoffs) + 1) * wall + sum(backoffs)`, and the serial chain of
those worst cases has to fit under `PREDICTION_MARKET_TIMEOUT`. Stages 1a and 1b run concurrently, so the chain
takes the max of them.

| stage                                   | cap                    | worst |
|-----------------------------------------|------------------------|-------|
| 1a Kalshi catalogue (wall)              | 40.0                   |  40   |
| 1b query author (concurrent with 1a)    | wall 20, backoffs (1,) |  41   |
| 1a PredictIt dump (concurrent)          | 10 x 2 + 0.5           |  20.5 |
| 2  venue search                         | 10 x 2 + 0.5           |  20.5 |
| 2.5 manifold detail fan-out (wall)      | 10.0                   |  10   |
| 4  ranking                              | wall 60, backoffs ()   |  60   |
| total                                   | max(41, 40, 20.5) + 20.5 + 10 + 60 | 131.5 |

### MARKET_QUERY_AUTHOR_WALL_TIMEOUT, MARKET_QUERY_AUTHOR_BACKOFFS, MARKET_RANKER_WALL_TIMEOUT, MARKET_RANKER_BACKOFFS

The ranker gets no retry: 36 of 36 measured calls parsed first try, a retry on a roughly 36k-token prompt is
expensive latency, and the deterministic pool-order fail-open slate is a good fallback sitting right there. Its
wall is 60 rather than the originally specced 45 because the prompt grew about 50%, with the full PredictIt
universe plus Manifold enrichment, and prefill scales with it.

### KALSHI_CATALOGUE_WALL_TIMEOUT, KALSHI_PAGE_SLEEP_S, KALSHI_PREFETCH_EVENT_LIMIT, KALSHI_PREFETCH_MAX_PAGES

The Kalshi catalogue pull gets a wall-clock budget for the whole paginated fetch, retries included, so pagination
can never push the snapshot past its own timeout. The per-page `aiohttp.ClientTimeout` sits under this wall, not
beside it.

`KALSHI_PAGE_SLEEP_S` at 0.25 is measured, on the value itself. Six full cold pulls through the real
`kalshi_prefetch_events` (2026-08-04 and 05, free unauthenticated GETs,
`scratch/market_port_2026-08-04/kalshi_page_sleep_probe.py`, receipts in that directory's `qa_artifacts/`) all
completed: 10,083 to 10,093 events over 51 pages every time, zero 429s, wall 16.90 to 25.37 s against this 40 s
cap. Compare the zero-sleep baseline it replaced, an HTTP 429 on 2 of 4 pulls with 8 to 18% of the catalogue lost,
and the 0.25 value is doing the work it was introduced for. That matters because a 429 is deliberately
non-retryable for this venue: the pull stops, reports incomplete, and bumps both degradation counters, so the
sleep value can redden CI on a condition it causes.

The wall decomposes exactly, which is what makes the headroom trustworthy rather than lucky: 50 sleeps times 0.25
is 12.50 s fixed, plus 4.29 to 12.77 s of actual fetch, with a residual of about 0.10 s. So the sleep is half to
three-quarters of the elapsed pull and the worst observed case used 63% of the wall. Only the run's first question
pays it, thanks to a 6-hour TTL cache, and it does not move `SNAPSHOT_STAGE_BUDGET_S`, since stage 1a already
takes the max of the author wall, this wall and the HTTP stage.

Headroom to raise it if a 429 ever returns: 0.3 projects to about 27.8 s and 0.4 to about 32.8 s at the worst
measured fetch, so 0.4 is the practical ceiling under this wall. Beyond it `attempt_budget`, which doubles as the
per-page HTTP timeout, starves the late pages and trades throttle-loss for wall-timeout-loss, bumping the same two
counters. Raising the wall alongside is the other lever.

One residual observation, benign but worth knowing before re-probing: back-to-back pulls 60 s apart got
monotonically slower (fetch 4.47, then 8.15, then 12.77 s), and the same cadence at 120 s gaps stayed flat (4.29,
4.41, 7.11 s). That looks like soft rate pressure that drains, never a 429, and prod pulls the catalogue once per
run, so it constrains probe design (space pulls out, or a probe's own cadence manufactures the slowdown it
reports) rather than the constant.

`KALSHI_PREFETCH_EVENT_LIMIT` is a runaway guard well above the roughly 10.1k live open events;
`KALSHI_PREFETCH_MAX_PAGES` is the real bound.

## Time-Series Anchor Provider (Phase B)

### TS_ANCHOR_ENABLED_ENV

Env-gated off by default. The provider renders a deterministic empirical-band anchor grounded in the resolution
series' own history, from FRED or yfinance, for numeric questions that route cleanly to a known series, via
resolution-criteria URLs or a curated title registry.

There is no statsforecast and no model selection: the Phase-A offline replay
(`scratch/ts_anchor_replay_2026-07-16/synthesis.md`) found CV-gated model picks beat naive out-of-sample only 43%
of the time, so the empirical h-step-change band is the render. Its value is grounding plus sharpening our
over-wide published low tail, where cov@10 was 0.02 against a 0.10 target.

It is backtest-safe, the first research provider that is: live uses `as_of=now`, while `is_benchmarking` uses
`question.open_time` so series data up to resolution is the answer, rather than `scheduled_resolution` minus a
buffer, with ALFRED vintages at `as_of` for revising series.

### TS_ANCHOR_CHART_ENABLED_ENV

A chart-image side channel. When on, and `TS_ANCHOR_ENABLED` is also on, the provider renders a PNG of the anchor
(the series plus its P10 to P90 band, sized in `research/ts_chart.py`) for single-level questions and stashes it
per qid; the forecaster passes it to each base model as a vision message.

Off everywhere until the text-versus-image A/B, see `FUTURE.md` "TS anchor chart image". Independent of
`TS_ANCHOR_ENABLED` so the text anchor can ship before the costlier, unvalidated image does. Note that matplotlib
is a dev-only dependency, so under the bot workflows' `--no-dev` install, flipping this on degrades to the
text-only anchor with one ERROR log.

### TS_ANCHOR_TIMEOUT, TS_ANCHOR_HTTP_TIMEOUT

A wall-clock cap on the whole provider, covering the fetch fan-out plus the render. Fetches run in
`asyncio.to_thread` under `asyncio.wait_for`, and a hung endpoint soft-fails to `""`. `TS_ANCHOR_HTTP_TIMEOUT` is
the per-request timeout for a single FRED, ALFRED or yfinance fetch.

### TS_ANCHOR_LOOKBACK_YEARS, TS_ANCHOR_SPREAD_LOOKBACK_YEARS

History lookback for both the displayed tables and the empirical change and window-max distributions. Spread legs
use the shorter window to exclude the 2020-04-20 negative WTI settlement, which breaks the strictly-positive
log-return construction.

### TS_ANCHOR_SECTION_MAX_CHARS

A char budget for the whole rendered section, self-budgeted like `resolution_source`. The per-leg render truncates
history tables so multi-leg spreads stay bounded.

### TS_ANCHOR_NATIVE_TABLE_ROWS, TS_ANCHOR_WEEKLY_TABLE_ROWS, TS_ANCHOR_MONTHLY_TABLE_ROWS

History-table lengths per resolution: the last N native-frequency observations, weekly down-sampled closes (about
3 months of trading weeks), and monthly down-sampled (about 2 years).

### TS_ANCHOR_OPEN_BOUND_SPAN_TOLERANCE

How far past an open displayed edge a rendered band may sit before the magnitude backstop treats it as a
wrong-quantity anchor, measured in multiples of the displayed span. An open edge means the outcome genuinely can
settle beyond it, so the constraint has to loosen there, but treating open as no constraint at all disarmed the
backstop entirely on the roughly 95% of numeric questions that carry two open bounds.

The measured window is wide, and both ends are pinned by tests so a future tweak has to confront the evidence. The
rule compares the nearest band edge, not the P50, against the range, which is what makes the window wide: every
band the anchor has actually published overlaps its own range, so it scores 0.00 spans outside and no tolerance
below 1.0 can suppress it. Meanwhile the wrong-quantity shapes this exists to catch, such as a percent-unit band
on a basis-point question, sit 0.63 to 0.73 outside, so anything at or above about 0.63 stops catching them, and a
1.0 tolerance catches nothing, which is why the value is well under it.

Closed edges get no tolerance at all: the outcome cannot settle past them, so the original zero-overlap rule
stands there unchanged and this whole knob is inert.

## Raw research-provider payload logging (durable GHA-artifact tape)

### RAW_RESEARCH_LOG_ENABLED_ENV

Independent from `PERSIST_RESEARCH_ENABLED_ENV`, which archives the post-summarizer research text keyed per
question. This captures each provider's raw return: AskNews article dicts per phase, native and gemini raw
responses plus grounding, prediction-market contracts, resolution-source per-URL fetches and gap-fill search
results, appended as JSONL to a `run_logs/` file so the raw evidence behind every forecast survives the 90-day
artifact window without depending on published comments.

Off by default in code, with the env unset, so tests and local runs never write; the four workflow yamls set it on.

### RAW_RESEARCH_LOG_DIR_ENV, RAW_RESEARCH_LOG_DIR_DEFAULT

The directory the raw-research JSONL is appended to. It defaults to `run_logs/`, which every workflow tees stdout
to and uploads wholesale as an artifact, so the raw log rides along with no upload-glob change. Overridable
through the `RAW_RESEARCH_LOG_DIR` env var, which tests point at a tmp dir.

### RAW_RESEARCH_MAX_PAYLOAD_CHARS

A per-record serialized-payload cap. A raw AskNews dual-phase pull or a grounded Gemini response can be large;
beyond this many chars the payload is replaced with a bounded truncation marker, carrying a preview and the
original length, so one giant pull cannot blow up the log file. GitHub Actions zips the artifact on upload, so
on-disk size is the only concern, and 200 KB per record is generous headroom for real payloads.
