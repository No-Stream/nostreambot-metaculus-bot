# Fall 2026 preregistration

We commit to the measurements below (each one a "read": an instrument, a cohort and a signal rule)
for the Fall 2026 FutureEval Bot Tournament (`fall-futureeval-2026`, project 33121; first question
close 2026-09-28, forecasting ends 2027-01-06). We wrote them down on 2026-09-09, before any fall
question had resolved, because the sample will be small. The commitment is gentle. Any read not
listed is exploratory: the round write-up that reports it labels it so, and on its own it changes no
setting and closes no open question.

Summer's era read compared two rosters; the fall read compares two pipelines on one roster. On 63
STRICT summer records the type-adjusted gap between the rosters was +10.71 spot peer with an interval
clearing zero. Matching the forecast horizon (the retired roster had been asked longer questions)
took it to +8.19 with an interval covering zero. No question forecast under the production
configuration has resolved yet, so the fall reads are its first test. Spot peer is the score the
tournament ranks on, in points. STRICT drops the three exclusion cohorts (known-bug, degraded-run and
partial-degraded records), imported from `metaculus_bot.performance_analysis.cohorts`. PIT is the
probability integral transform, the published distribution evaluated at the realised value.
Calibrated PITs are uniform, so their standard deviation reads 0.2887 and 80% land inside
[0.10, 0.90], which is central-80 coverage. An era is a stretch during which the configuration
didn't change, a sub-era is one inside the three-model era, and every boundary is a merge-to-`main`
committer timestamp in UTC.

## The configuration under test

The treated arm is the fall configuration. The roster (gpt-5.6-sol, claude-opus-4.8 and
gemini-3.1-pro-preview) has been frozen since 2026-07-21. Three September merges opened the
`fall_config` sub-era at 2026-09-05T01:59:24Z, effective 2026-09-07T05:52:20Z because nothing was
forecast between the first merge and the last. They rewrote the prompts, adding the soft-clock and
history-discharged rules (read (c)), and gave each question a time budget derived from its close
time. They also rebuilt the ladder of fetch attempts for a cited resolution source, up to a paid
Gemini `url_context` read (read (e)). One more merge, the cost pass, is planned before the first
close. It changes gap-fill, the second-pass research stage, which runs as two passes today: v1, the
older search fan-out, and v2, an agentic tool loop whose driver model also logs a private "ghost"
forecast for telemetry. The candidates, priced per question:

| candidate | change | flag or constant | $/question |
|---|---|---|---|
| today | none | | 2.21 |
| v1 off (the cost pass's pick) | gap-fill v1 off, v2 kept, ghost cache-aligned | `GAP_FILL_ENABLED=false`; `loop.py` | 1.37 |
| halve v1 | v1's fan-out 4 gaps to 2 | `GAP_FILL_MAX_GAPS` | 1.84 |
| smaller search context | native search context high to medium | `NATIVE_SEARCH_CONTEXT_SIZE` | unmeasured |

Whether v1 goes off depends on a free comparison, in progress on 2026-09-09, of v1's findings
against v2's on archived bundles where both ran. Whichever candidate ships, the rule is the same.
The era boundary is that merge's merge-to-main committer timestamp, the round's tagging pass records
it as a sub-era inside `fall_config`, and the primary arm is every fall record forecast at or after
it. The 6 questions forecast between 2026-09-07 and that merge (1 practice question, 5 Metaculus Cup
questions) get their own sub-era and their own row in every table. If no cost merge lands before
2026-09-28, the `fall_config` sub-era as it stands is the arm.

The comparison arm is the `ranked_markets` sub-era (2026-08-06T01:28:49Z to 2026-08-26T17:23:30Z),
the last summer configuration with resolved questions. It holds 29 of them, so the whole
three-model era (63 STRICT records) is the secondary comparison arm, reported beside it. This is an
observational before-and-after comparison; type mix and forecast horizon are the only controls.
Metaculus Cup records have no spot peer (the cup scores on coverage-scaled peer, score type
`peer_tournament`), so we report them on their own scale.

## The reads we commit to

Every read runs at each checkpoint, STRICT and unfiltered, with the excluded records reported
separately. A cluster is a set of questions whose resolutions share one driver (a jobs report, a
poll wave). The round's `cluster_structure.py` curates them, effective n is their count, and the
verdict reads the cluster-bootstrap interval with that file passed as `--clusters`.

| read | instrument | cohort | signal rule |
|---|---|---|---|
| (a) score gap | `performance_analysis.era_gap`, the STRICT "type-adjusted, horizon-matched" row | fall tournament, spot peer | `two_sided_watch`: concern below -5 with the interval excluding zero; favourable above zero with the interval excluding zero; else no measurable difference |
| (b) numeric width | `performance_analysis.width_monitor` era table | fall numeric family (numeric and discrete) | hold `TAIL_WIDEN_K_TAIL` = 1.0 unless both trigger halves fire (central-80 coverage at or above 0.88 and PIT std below 0.25) at n of 30 or more |
| (c) prompt rules | hand coding of the members' rationales with the two failure-shape definitions from the 2026-09-02 audit | fall binary and multiple choice, hits coded alongside misses | incidence and paired spot peer against the summer baselines; descriptive until 2 or more flagged records a side |
| (d) cost | `CREDIT_ROLE_SPEND`, the per-role spend record, summed per run | every fall run that forecast a question | per-run median at or under $1.50 as booked, over 10 or more questions |
| (e) fast path, ladder, forfeits | `TIME_BUDGET`, `TIME_BUDGET_FAST_PATH`, `WALLCLOCK_ABORT`, `RESOLUTION_SOURCE_ESCALATION`, the `make supply_probe` forfeit sweep | every fall question a scheduled run picked up | any `WALLCLOCK_ABORT` or forfeit on a picked-up question is a bug to trace; fires and rescues reported per rung |
| (f) gap-fill v1 off | bundle length from the research archive; `GHOST_PRE_JSON` against `GHOST_FORECAST_JSON` through `ghost_pre_post.py`; the `era_gap` per-type table | fall records with a scored ghost | descriptive only |

The score gap (a) is the primary comparison, treated minus comparison in spot-peer points. Its
estimator is the one `era_gap` prints on its watch row: each record's spot peer residualized on the
pooled per-type mean, with the comparison arm capped at the treated arm's longest submit-to-resolve
lag, both arms STRICT. The tagging pass writes the sub-era names into `triple_subera_fine`
(`config_era` holds only the coarse eras, and the module selects its arms on whichever field
`--era-field` names), so the invocation is `uv run python -m metaculus_bot.performance_analysis.era_gap
--dataset <round>/perf_all_tagged.json --era-field triple_subera_fine --treated-era <fall sub-era>
--comparison-era ranked_markets --strict --clusters <round>/cluster_structure.json`, with
`fall_config` as the fall sub-era until the cost pass merges. Run before the first treated question
has resolved, it exits with a message naming the empty arm and prints no report. A concern verdict commits us to a trace dossier
for every fall miss against the `ranked_markets` code and a merge-by-merge review of the September
changes. Any revert is an operator decision at a new era boundary. The per-type table stays
exploratory at every checkpoint: summer's gap rested on one cell of 12 discrete questions, and a fall
cell that size can't carry a claim.

The width read (b) is the shipped monitor's era table on the fall numeric-family records. Summer
closed at a PIT std of 0.2646 and central-80 coverage of 0.889, a lean toward too wide: the coverage
half of the trigger fired and the standard-deviation half didn't. The operator prefers erring
slightly wide, so the hold rule is asymmetric on purpose. When both halves fire, the response is the
symmetric prompt clause FUTURE.md's width watch specifies ("match width to reasoning; don't pad or
sharpen from disposition"), and `TAIL_WIDEN_K_TAIL` stays where it is. A too-narrow read (PIT std
above 0.2887 with coverage below 0.80) we report and re-read under the same n floor.

Nothing in the pipeline records that an ensemble member applied the soft-clock rule or the
history-discharged rule, so read (c) codes the members' rationales by hand, blind to outcome, with
the two shape definitions from `scratch/failure_mode_audit_2026-09-02/AUDIT_SYNTHESIS.md` (its lens A
and lens C). Coding the hits alongside the misses gives the read a control side. At the summer rates
below, 30 fall records hold about 2 flagged records, so incidence is readable at every checkpoint and
the paired score read is a season-end read at best. The history-discharged rule shipped with the
operator's final say pending; if they decline it, this read narrows to the soft-clock rule.

| shape the rule targets | summer incidence | flagged records score |
|---|---|---|
| announced-unscheduled: a member treats an announced target date as a binding clock | 6.4% of records, 8.3% of binaries | 18.7 points worse |
| history-repeats: a member centers on a cadence its own analysis says was discharged | 12.1% of rationales | about 7 points worse |

Cost per question (d) is the per-role spend record summed per run and divided by the questions that
run forecast, as OpenRouter's per-call accounting books it. The record double-counts the Google
forecaster slot, so the corrected figure (about $0.13 lower) prints beside the booked one until that
fix lands, and a breached ceiling sends the per-role table back to the cost pass. The fast path (e)
never fired in summer, because it needs a scheduled run to pick the question up within about 31
minutes of close: `TIME_BUDGET` is the uncensored denominator, its `fast_path` and `close_limited`
fields the numerators, and `WALLCLOCK_ABORT` the missed-deadline count. The fetch ladder reports
fires and rescues per rung from `RESOLUTION_SOURCE_ESCALATION`, with the paid `url_context` rung on
its own line (2 attempts, 0 rescues through 2026-09-09). Forfeits (questions on closed or resolved
posts the bot never forecast) come from the supply probe's sweep; summer had 7.

If gap-fill v1 turns off, read (f) reports three descriptive quantities, because the fall arm
differs from `ranked_markets` in prompts and research at once. Bundle length comes from the research
archive's fall records, against summer's 49,500 to 52,900 characters. The driver's pre-versus-post
research read scores `GHOST_PRE_JSON` against `GHOST_FORECAST_JSON` on fall pairs; summer's value
was +7.18 log points on the 36 pairs the research moved [+1.26, +13.99], and the ghost is built
without v1's section, so this measures v2 exactly as before. The clean v1 read is a paid replay with
v1's section stripped from the archived research, an operator decision.

## Checkpoints, signal and noise

| checkpoint | when | what gets a number | what gets a verdict word |
|---|---|---|---|
| first read | 15 fall tournament STRICT records resolved | every read, descriptive | none |
| second read | 30 | every read | (a) the watch verdict; (b) the trigger; (d) the ceiling |
| season end | after 2027-01-06, plus the closed-but-unresolved backlog | every read | (c), if it has 2 or more flagged records a side |

The floors are arithmetic. Summer's horizon-matched interval on 63 treated records against 95 was
about 24 points wide, and the treated arm dominates the variance, so 15 fall records give an
interval roughly twice as wide and 30 about 1.5 times. Only a gap of order 20 points could clear
zero at the second read, and effective n runs below the record count (summer's 63 STRICT records
were 58 clusters). A signal is a verdict word from the signal-rule column, at or past its n floor,
on the clustered interval. Everything else is noise until the next checkpoint, and a favourable (a)
verdict is reported as a score gap and nothing more. A miss (a wrong-sided consensus, a forfeit, a
`WALLCLOCK_ABORT`) gets a trace dossier per the residual-analysis playbook whatever these rules say.

Four things stay fixed through the season. The roster stays as it is, because a change would open a
new era and reset every read. `TAIL_WIDEN_K_TAIL` stays at 1.0 under the rule in read (b). Stacking
(a second LLM rewriting the ensemble's forecast), the probabilistic tools and a mean aggregate stay
off, since all three were benchmarked and rejected; the per-bin Mantic grids are the one standing
exception. A new prompt rule ships only on a cohort measurement.

## Amendments (2026-11-04, first checkpoint)

The roster did not stay frozen: four merges changed models or research roles after this document
(021e059 gpt-6-sol and opus-5.5, #73 gpt-6.1-sol, #77 gemini-4-argon, #79 Terra research roles). Each is a
labelled fine sub-era in `eras.py`, and the operator ruled that every read pools them as one fall arm, so
read (a) selects on the coarse field: `--era-field triple_subera --treated-era triple_fall_config
--comparison-era triple_ranked_market`. The secondary comparison against the whole three-model era is now
runnable because `era_gap` accepts several labels per arm. The operator also shipped one new prompt rule,
`_COUNT_PROCESS_RULE`, on mechanism reasoning rather than a cohort measurement, departing from the "a new
prompt rule ships only on a cohort measurement" clause above; the market snapshot gained a metadata column in
the same branch. Both change what forecasters see and take an era label at their merge. First-checkpoint
numbers are in `scratch/residual_2026-11-04/SYNTHESIS.md`.

## Links

- The summer round: `scratch/residual_2026-09-09/SYNTHESIS.md`, `ERA_MAP.md`, `era_boundaries.json`, `dim_numeric-width.md`, `cluster_structure.json`.
- The instruments: `metaculus_bot/performance_analysis/era_gap.py` with `docs/performance_analysis.md` "The era gap and the horizon confound"; `metaculus_bot/performance_analysis/width_monitor.py` with FUTURE.md "Width post-ship watch"; `scripts/telemetry/markers.py` and `docs/telemetry_markers.md`.
- The prompt rules: `docs/prompts.md` on `_SOFT_CLOCK_RULE` and `_HISTORY_DISCHARGED_RULE`; FUTURE.md watch item 5 under the next-season bundle.
- Cost: `scratch/cost_pass_2026-09-09/COST_PASS.md` sections 4 and 5, and `ghost_pre_post.py` beside it.
- Procedure: `scratch_docs_and_planning/residual_analysis_playbook.md`.
