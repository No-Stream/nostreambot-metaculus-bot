# Residual / performance analysis: methodology and conventions

This is the reference half of the bot's residual-analysis work: the conventions that
make a number trustworthy, and the receipts behind each one. It covers the round pull
and the `--prior` rescoring diff, what the archives hold and which of their fields are
historically unreadable, era bucketing and the merge-to-main dating rule, the standing
exclusion cohorts, the scoring conventions (spot peer, and `spot_peer_delta` for
counterfactuals), the clip-threshold sweep, per-model forecast recovery, how to read the
width monitor's era table, the PIT convention for out-of-range resolutions, starved outer
tails, and question-supply / forfeit accounting.

Two sibling documents own the other halves, and this doc cross-links rather than
restates them. `docs/operations.md` § "Performance analysis and the width monitor" is
the **runbook**: the exact commands, the `--exclude-qids` mechanics, and what each
report prints. `scratch_docs_and_planning/residual_analysis_playbook.md` owns the
per-round **procedure** (Pre-pull → Recon → Pull → automated dims → per-question trace
dossiers with adversarial verification → Synthesize). Read this doc for *why a number
means what it means*; read those two for *what to type* and *what order to do it in*.

The pull and round-builder APIs described here were checked against current code on
2026-09-20. Measured figures are dated and carry their receipt path, because most are
snapshots of one round's archive rather than repo constants.

**Routine refreshes use the committed CLI and library API below.** Do not write new scratch
scripts, copy prior-round drivers, or recreate standard dimensions for a routine refresh without
an agreed functionality change. Dated `scratch/residual_<date>/` directories hold round inputs
and outputs. Focused follow-up analyses may use scratch scripts; if a standard step is missing,
report the gap and agree on a maintained addition.

## The round pull, and why `--prior` is mandatory

`metaculus_bot/performance_analysis/` evaluates the live bot's calibration against
actual resolutions. Entry point:

```bash
uv run python -m metaculus_bot.performance_analysis --tournament <slug> --output <path>
```

The `--tournament` default (`DEFAULT_TOURNAMENT`, `performance_analysis/cli.py`) lags
the live season, so pass the current slug explicitly. The **pull is read-only and
free**: it hits only the Metaculus API (the tournament's resolved posts, paged off the
list endpoint under `with_cp=true` so each page already carries the token's own
`my_forecasts` and no per-post GET is issued, plus the bot's own comments, user id
275109, auth via `METACULUS_TOKEN`), makes no LLM or research calls and publishes
nothing, so it is **not subject to the repo's cost gate** (unlike `make backtest_*` and
live runs).

**`--output` is required for a live pull and saves on the `--cached` path too.** It has no
default: it used to be `scratch/performance_data.json`, which is gitignored and absent in a
fresh clone, and once the cached path started saving, a default would have made a read-only
report clobber an unrelated pull. Omitting `--output` under `--cached` is the read-only report.
Passing it under `--cached --prior` is how the rescore tags reach disk; `save_dataset` used to
sit inside the live-pull branch, so that combination tagged records in memory and wrote
nothing, and nine rounds carried a `diff_prior.py` to work around it.

**One page's network blip no longer abandons the sweep.** `_api_get` retries HTTP 429 and the
transient network failures (`requests.exceptions.Timeout`, which covers read and connect
timeouts, and `ConnectionError`) out of ONE shared budget of `MAX_RETRIES` attempts, with
`RETRY_BACKOFF_SECS * attempt` between them. Sharing the budget is what keeps the change safe
in a path whose overrun costs forecasts: the worst case per page is unchanged from the 429-only
retry it replaced, `MAX_RETRIES` reads at `REQUEST_TIMEOUT_SECS` plus the backoffs, 105 s at
the values in `collector.py`. Nothing else is caught, so a genuine failure still crashes with
its own traceback rather than arriving as a slow success, and an exhausted 429 raises an error
naming the rate limit instead of reporting the whole rate-limited run as one unlucky request.
A sustained outage still fails the pull; re-running it is free.

**Pass `--prior <previous round's dataset>` on every round pull.** Metaculus
re-resolves questions IN PLACE without moving any timestamp we store. It edited q44798
(Halo: Campaign Evolved Metascore) from 80 to 82 with `resolution_set_time` left at a
stamp that PRECEDES the pull which still read 80, flipping that record's spot peer from
+5.41 to −5.42 between two rounds while the earlier round's published tables went
silently stale.

`--prior` diffs the resolution VALUE and every `metaculus_scores` field against the
prior pull (`performance_analysis/rescore_diff.py`), tags each moved record
`platform_rescored=True` with `platform_rescored_fields` / `prior_resolution` /
`prior_metaculus_scores`, and emits one `PLATFORM_RESCORED` WARN per changed field plus
a printed summary. Three details:

- The tag reads as a **TERNARY**: None means "no prior record, never compared", and
  only False means "compared, nothing moved".
- The field is `platform_rescored_fields` and **not** `rescored_fields`, which a round
  script already spends on the bot-side scores `collector.rescore_records` healed.
  Two different facts.
- Records store `metadata.resolution_set_time`, useful for bounding when an edit
  happened but never as the detector.

## What `make sync_all` pulls

"Residual analysis" implies `make sync_all` first, always (read-only and free). It
pulls **everything** sync-shaped in one command, and that matters because GHA artifacts
expire at 90 days, so a single-source pull silently and permanently drops whatever it
did not fetch. Three archives:

- **The research archive** (`backtests/research_archive/latest/<qid>.json`):
  per-question post-summarizer research. Precedence rules below.
- **The run-log telemetry archive** (`backtests/telemetry_archive/`): the
  `EXTRACTION_RUNG` / `GAP_FILL_V2` / `GHOST_FORECAST` / `OPEN_BOUND_PILING` /
  `CREDIT_*` markers, plus the 2026-08-25 honesty set
  (`NUMERIC_DEGENERATE_DECLARATION`, `NUMERIC_AGGREGATE_GRID_MISMATCH`,
  `SPREAD_UNDEFINED`, `MARKET_RANKING_DEGRADED`, `CDF_MAXSTEP_CLIP`), the 2026-09-01
  bundle's set (`EXTREME_CALL`, `THIN_PUBLISH_FLOOR`, `RESOLUTION_SOURCE_FETCH`,
  `CREDIT_ROLE_SPEND`, `GEMINI_GROUNDING_DENSITY` (historical; retired 2026-09-22), `GEMINI_UNSUPPORTED_ATTRIBUTION`,
  `FINANCIAL_NOISE_FLAG`, `MARKET_TIER_CAPPED`, `FRED_UNKNOWN_SERIES`), the 2026-09-02
  additions (`AGENTIC_FETCH_THROTTLED`, `MEMBER_FORECAST`), plus `GEMINI_USAGE` (the
  google-genai token and grounded-query accounting for all three Gemini surfaces, which bill
  outside OpenRouter and so appear in no `CREDIT_*` marker; the `role` field partitions them
  into `grounded_search`, gap-fill v2's `read_document` and the resolution-source ladder's
  `resolution_source`),
  `RESOLUTION_SOURCE_ESCALATION` (one line per escalated fetch rung, with what
  triggered it and what it cost), `AGENTIC_FETCH_LOCAL_DOC` (one line per gap-fill v2
  document read served from the host's own bytes instead of the paid reader; it fires only
  where text was actually served, so its absence measures nothing),
  `AGENTIC_URLCONTEXT_ROBOTS_SKIP` (one line per paid read skipped because the host's
  robots.txt disallows `Google-Extended`) and the paid resolution-source rung's own three
  lines (`RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP`,
  `RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED`,
  `RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED`: a read the pre-check refused, a paid read that
  retrieved nothing, and a paid read that retrieved a page with nothing on the ask; registered
  with the 2026-09-04 flag flip, so no run from before that merge carries any).
  `scripts/telemetry/markers.py` is the registry.
- **The raw research-provider payload archive**
  (`backtests/research_archive/raw/<run_id>.jsonl`, one file per run): each provider's
  RAW return before formatting: AskNews article dicts per HOT/HISTORICAL phase,
  native-search and Gemini raw responses with grounding, prediction-market contracts,
  resolution-source per-URL fetches, gap-fill v1 search results. Written by
  `metaculus_bot.research.raw_log` when `RAW_RESEARCH_LOG_ENABLED` is set, so the raw
  evidence behind every forecast is auditable without depending on published comments.
  `financial_data` is deliberately not captured; its raw series live only inside
  `to_thread` workers.

`scripts/research_sync/` holds the launchd job, wired to `sync_all` for the same
reason. The telemetry archive also feeds `make score_ghosts` (the gap-fill v2
ghost-vs-published log-score gate, ~0 scoreable until v2-era questions resolve). Dated
round outputs land under `scratch/residual_<date>/` (gitignored).

Per-question tracing is a first-class phase, not optional. Operator directive
2026-08-24: it is often the most valuable part of a residual round, and every miss
dossier gets an adversarial verification pass (the 2026-08-02 round revised 6 of 6
verified dossiers). The playbook supersedes the older `residual_rerun_workflow.js`.

## The research archive's `latest/` records come from three writers, and the difference matters

**GHA run artifacts** are the source of truth for every question since 2026-05-29: the
exact research text the forecasters saw, plus `provider_results` / `gap_fill_v2` /
`asknews_raw` on schema-v2 records. Most older artifact records predate those fields,
so their absence means "old record", not "degraded run".

**Metaculus comments** are a lossy fallback for older questions only: middle-trimmed
(`RESEARCH_SECTION_CHAR_LIMIT` / `COMMENT_CHAR_LIMIT`), sections re-headed one level
deeper (so `^## ` presence probes read near-zero), and missing `resolution_criteria`.

Both comment readers, the residual pull in `collector.py` (`fetch_bot_comments`) and this
backfill's `fetch_all_comments`, list the author's comments TWICE, once plain and once with
`is_private=true`, and merge the two by comment id. The bot POSTs every comment private and
Metaculus flips older ones public server-side, so the plain listing serves only the flipped
ones: on 2026-09-09 it returned 1,054 public summer comments and none of the six private
comments from 2026-09-07, and a single-listing pull yields fall records with no comment text,
no `bot_comment_created_at` and no per-model parse.

A third writer, `scripts/backfill_research_from_logs.py`, parses run logs and keys its
`qid` on the **POST id** while every other writer keys on the **QUESTION id**. The two
share one integer space, so a single `by_qid/<N>.jsonl` can legitimately hold two
different questions' research.

Precedence for `latest/<qid>.json` is therefore `artifact` > `comment_backfill` >
`log_backfill` (`record_precedence_key` in `scripts/download_research.py`), then
newest-by-parsed-timestamp within a class. Log-backfill text is untrimmed but
post-keyed, and `latest/` is read question-id-first, so promoting it serves the wrong
question. Measured: it made `latest/43592` return question 43591's research verbatim.

A Mantic run (`--mode mantic`) writes to the same archive and adds a second integer space
on top. Its records carry `platform=mantic` and a `tournament_id` equal to
`MANTIC_TOURNAMENT_ID`, but filenames are not namespaced by platform and `build_archive`
groups on the bare `qid`, so a Mantic id and a Metaculus id that meet merge into one
`by_qid` / `latest` / manifest entry, and `platform` is the field to filter on inside the
group. The margin today is about 13,700 Mantic posts: open Mantic posts sit in
the 650s, and the next Metaculus id already in the archive above them is 14333, from the
evergreen `test_questions` set (`docs/operations.md` "How the mode works" has the
arithmetic; the revisit is logged in `FUTURE.md`). None of this reaches the residual
dataset. `collector.py` reads `BASE_URL = "https://www.metaculus.com/api"` with the
Metaculus bot user id, so a Mantic question cannot become a record, and the Mantic-side
accounting (forecast, forfeit, miss rate per release hour) lives in
`make supply_probe_mantic` (`docs/supply_probe.md` "The Mantic mode").

Reading rules that follow from this:

- Read the record's `source` field (`"artifact"` | `"comment_backfill"` |
  `"log_backfill"`, mirrored as `latest_source` in `manifest.json`). **Never infer the
  class from `run_id` alone**; log-backfill run_ids are plain GHA run ids,
  indistinguishable from artifacts.
- **Never pool the classes** for a presence, provider-mix, or length claim.
  `providers_used` on a comment record is reconstructed from trimmed text:
  `financial_data` reads 31 where the artifacts say 253.
- `latest_timestamp` is the winning record's timestamp, not a freshness signal.
- `scripts/research_sync/verify_completeness.py` gates the merge stage: a question
  holding an artifact record must be served by one in `latest/`.
- `make backtest_with_cache` logs the source split it replays, so pre- and
  post-2026-08-03 cached-backtest numbers are not comparable.

### The research record's fields

`ResearchPersistenceWriter.record` (`metaculus_bot/research/persistence.py`) writes one
JSONL record per question. Several of its fields exist to answer a question the obvious
field cannot, and the reasons are below.

**`qid` versus `post_id`.** `qid` is the Metaculus QUESTION id (`id_of_question`), and
the archive keys `latest/<qid>.json` on it. `post_id` is the separate POST id, the one
that appears in `page_url`, and it diverges from `qid` on newer posts. It is written as
an explicit field so residual analysis can join a research record to telemetry markers
keyed on the post id (`GAP_FILL_V2` and `GHOST_FORECAST`) and to the perf dataset
without re-parsing the page URL. The field is additive: it defaults to None, and older
readers plus the URL itself still carry the post id, so nothing breaks.

**`providers_used` is legacy and ambiguous.** In live-capture records it meant
"attempted"; in comment-backfill records it meant "succeeded-with-output". It is kept
only for back-compat with old archive readers. `provider_results` is the authoritative
per-provider outcome, with `providers_attempted` and `providers_succeeded` as the
unambiguous derived lists. Those three arguments default to None so older callers, and
the backfill paths, keep working.

**`gap_fill_v2`** carries the agentic loop's trace when the v2 loop ran, and the key is
written only in that case so records stay compact when the flag is off. It holds four
keys: `transcript`, `telemetry`, `ghost` (nullable), and, since 2026-09-09, `ghost_v1`,
the v1 ghost forecast, also nullable. `_stamp_v1_ghost` in
`metaculus_bot/research/gap_fill_stages.py` writes `ghost_v1`, and writes None when the
v1 ghost did not run or did not survive its budget.

**`provider_diagnostics_block`** is the rendered `## Provider Diagnostics` markdown.
Since the diagnostics seam landed (2026-07) it is no longer embedded in `research_text`,
because forecasters must not see it, so it is archived as its own field to keep records
self-contained for grep-based triage.

**`platform`** (`PLATFORM_METACULUS` or `PLATFORM_MANTIC`) says which question platform
`qid`, `post_id` and `page_url` belong to. Filenames stay un-namespaced and the archive
builder groups on the bare qid, so this field is what separates the two platforms'
records inside a group, and what a cross-platform analysis filters on. It does not stop
a Mantic id and a Metaculus id that meet from sharing one `by_qid` or `latest` entry.
The margin is the gap from the Mantic counter, whose open posts sit in the 650s, up to
14333, the next Metaculus key above it: the evergreen test-question set puts 578, 14333
and 20683 in the archive, and every other Metaculus key is 38,000 and up. The field is
additive with passthrough readers, and records older than the field are all Metaculus.

**`asknews_raw`** is the raw pre-summarization AskNews article markdown, added for
2026-07-18 audit hygiene. `research_text` carries only the summarizer's briefing, so
without this field a FETCH-versus-SUMMARIZE attribution read or a summarizer replay
needs a fresh paid AskNews pull. It is written only when AskNews actually ran and
returned articles, and stays empty on the fallback and prose paths. Like
`provider_diagnostics_block` it is additive with passthrough readers, so it needed no
schema-version bump.

## Two treatment tags read as TERNARY, and two archived fields are historically unreadable

`research_tags.gfv2_loop_ran` is None on any record whose WRITER could not carry the
`gap_fill_v2` payload; only a schema-v2 `artifact` record can, and that writer omits
the key when the loop did not run. So on a carryable record its absence is a
*measurement*, and everywhere else it is *silence*. Reading it as a bool put 880
archived can't-carry records into the untreated arm against 77 measured ones, which
would have poisoned the v2 treated/untreated calibration split outright.

The companion `gfv2_confidence` grades a False `gfv2_present` the way
`anchor_confidence` grades the anchor read, with five values
(`performance_analysis/research_tags.py`):

| value | meaning |
|---|---|
| `header` | the section header itself was found |
| `payload_ran_no_section` | the loop ran and contributed nothing (a soft-fail) |
| `payload_confirms_absent` | a carryable record with neither header nor payload |
| `ambiguous_trimmed_no_payload` | a trimmed comment record; the section may have been trimmed away |
| `absent_no_payload` | an untrimmed record from a writer that cannot carry the payload |

Four more rules the tagger in `performance_analysis/research_tags.py` runs on, none of
them obvious from the code:

- **Header greps are depth-agnostic** (`^#{1,4}`). An artifact record heads its sections
  at `## `, while a comment-backfill record re-heads everything one level deeper, so an
  exact-depth grep would read every backfilled record as untreated.
- **The section flag is the treatment marker, and `gfv2_loop_ran` is deliberately a
  different fact.** The v2 driver banks a transcript on the record even when it soft-fails
  and contributes no section to the bundle, so payload presence overstates treatment.
- **`anchor_confidence` grades a False anchor read** against the trim-immune
  `## Provider Diagnostics` block: `header` when the header itself was found,
  `diag_ok_header_missing` when the provider ran and produced a section the text no longer
  carries (trimming ate it, so treatment is genuinely unclear), `diag_confirms_absent`
  when the diagnostics line says `empty` / `errored` / `skipped` / `timeout`, and
  `ambiguous_trimmed_no_diag` on a trimmed record with no diagnostics line at all, since
  trimming keeps the header and the tail and can eat a leading section.
- **A question with no archive record gets None on every tag, never False.** Absence of
  evidence is not an untreated record.

The payload-era boundary that decides carryability is `B4E9DF0_MERGED_AT`
(2026-07-21T17:07:37Z), aliased in `research_tags.py` as `_GFV2_PAYLOAD_ERA_START`: schema
v2 landed 2026-06-28 in `1655c43`, three weeks before the `gap_fill_v2` write reached
`main`, so a schema-v2 artifact from that window is a can't-carry record too rather than a
confident False. A present payload proves a run whatever the writer class; only an ABSENT
one depends on carryability, and a null payload counts as absent, because a
key-present-but-empty read as a run is the collapse the original `bool()` made. A
`log_backfill` record never tags at all: it is post-id-keyed and identified only by
`page_url`, so on an id collision the URL check cannot rule out a foreign question.

Separately, `metadata.nr_forecasters` (the Metaculus CROWD size) reads **0 in all 2196
records pulled before 2026-08-25**, because the collector read it off the question
dict, where it does not exist; it lives on the POST. Nothing rewrites the archive, so
treat a 0 on an older record as UNKNOWN, never as an empty crowd. Fresh pulls carry
real counts (typically 100-250 on tournament questions) or None when the post omits the
field, and `audit.py` renders None as `n/a`.

## Era bucketing is mandatory for calibration claims

Any calibration, aggregation, or bias claim computed on pooled resolved data is suspect
until split by config/roster era (proxy: `source_tournament`, or
`bot_comment_created_at` versus config-flip dates).

Three separate conclusions have flipped under era-bucketing:

- the numeric "too wide" verdict (2026-06, computed on pre-flip-only data);
- the "current pipeline too narrow" verdict (2026-07, softened and then reversed as
  post-flip n grew);
- the YES-side overconfidence finding (2026-07-08), which turned out to be
  spring-2026-era-local: fall was well-calibrated, and a pooled fit would have
  degraded fall out-of-sample.

Bucket by **major** config/roster changes: model swaps, aggregation changes, widening
flips, research-stage changes. **NOT** by every git hash: a small prompt tweak does not
start a new era; a forecaster-roster or pipeline-behavior change does. The judgment
call is "would this change plausibly shift the forecast distribution?" If unsure, run
the analysis both ways.

Read the merge-date rule below before fixing any boundary. In particular the whole
july15 bundle (everything authored 2026-07-15 through 07-20) is a SINGLE boundary at
**2026-07-21T17:07:37Z (`b4e9df0`)**. It carried gap-fill v2 on, the native-search and
crux-analyzer sol→terra swaps, both same-day forecaster-roster changes (the fable-5 →
opus-4.7 forecaster plus opus-4.8 stacker swap, then the drop from 6 to the 3-member
latest-per-vendor triple gpt-5.6-sol / opus-4.8 / gemini-3.1-pro-preview), the sol
forecaster's xhigh→high effort drop, and `MIN_FORECASTERS_TO_PUBLISH` 3→1. None of them
is separately datable, and no shift across the boundary can be attributed to any one of
them.

**Fitted calibration layers (shrinks, clamps, haircuts) require a decisive
out-of-sample era test before shipping**: fit on eras 1..k-1, must improve era k, else
they are drift bombs.

## Era boundaries are merge-to-main timestamps, never authoring dates

Prod runs from `main`, so a config change is live only from the moment its merge commit
lands there. Get the boundary from the first-parent log of `main`
(`git log --first-parent --format='%H %cI %s' main`) and then read the merge commit's
committer date (`TZ=UTC git log -1 --date=iso-local --format='%h %cd' <merge-sha>`).

A branch can sit for days. Two dated examples:

- the july15 bundle was authored 2026-07-15..07-20 and landed
  **2026-07-21T17:07:37Z (`b4e9df0`)**;
- the `base_rate_anchor` / `criteria_clauses` telemetry was authored 2026-07-08
  (`30bca2f`) and landed **2026-07-11T16:37:17Z (`642b027`)**.

Keying on the authoring date files every run in the gap under the wrong config, and
this has already cost real analysis twice.

First, it manufactured a phantom one-record `ts_anchor` era in `width_monitor.py` out of
a question whose own comment names the retired six-model roster (`grok-4.5`, `gpt-5.5`,
`opus-4.6`), all dropped by the same merge that landed the anchor, so the combination
is impossible post-merge. That phantom is gone. **Today the `ts_anchor` row is absent
from the width-monitor table for a different and correct reason:** empty eras are
omitted, and no post-july15-bundle numeric has resolved yet. The two causes are
sequential, not competing: a phantom row that was wrong, then a legitimately empty row
that is omitted.

Second, it made the guard-telemetry presence check read an "intermittent emission" rate
instead of a clean 100%: **58%** on the receipt's per-slot cohort
(`scratch/residual_2026-08-02/dim_ghosts-and-guards.md:118`) and 78.9% on a per-comment
recount. The spread between those two figures is itself a second reason not to lean on
the authoring-date number. The mechanism is exact regardless of cohort: all 8 binary
comments in the authored-but-unmerged gap window carry no anchor, and the first one that
does is 2026-07-12, after the merge, exactly as "prod runs from main" predicts.

**Corollary: several authoring dates often collapse into one boundary.** Nothing on
`main` changed between the 2026-07-12 merge (`f084bf7`) and `b4e9df0` (a
`git diff --stat f084bf7 b4e9df0^1 -- metaculus_bot/ .github/workflows/` comes back
empty), so 2026-07-15 / 07-17 / 07-18 / 07-20 are **one** era boundary, not four. That
merge landed the TS anchor, gap-fill v2, the six-models-to-triple roster drop,
`MIN_FORECASTERS_TO_PUBLISH` 3→1 and the sol→terra role swaps together, which also means
no width or score shift across it can be attributed to any one of them. Treating those
dates as separable slices a period of constant prod config and reads noise as a config
effect. When a doc gives an authoring date, say so, and put the landing date next to it.

### One home: `performance_analysis/eras.py`

Every boundary instant and every per-record era tag lives in `performance_analysis/eras.py`,
and nothing else in the tree declares one. Import from there rather than retyping an
instant: the era map had been retyped by hand into each round's own copy for five rounds,
and one of those copies carried an authoring date for four months without a single test
noticing. Where a module wants its own name for a boundary it aliases the object, as
`width_monitor.py` does with `WIDENING_FLIP` and `TS_ANCHOR_ENABLE` and `research_tags.py`
does with `_GFV2_PAYLOAD_ERA_START`. See "Vocabulary that collides" at the end for those
aliases.

| Constant | Merge | Committer instant (UTC) | What landed |
|---|---|---|---|
| `WIDENING_FLIP_MERGED_AT` | `0e85e1b` | 2026-05-18T17:21:19Z | numeric `k_tail` 1.25 to 1.0, `SPAN_FLOOR_GAMMA` 1.0 to 0.0, binary clamp 0.01/0.99 to 0.02/0.98 |
| `B4E9DF0_MERGED_AT` | `b4e9df0` (PR #55) | 2026-07-21T17:07:37Z | the july15 bundle: 3-member roster, gap-fill v2, TS anchor, `MIN_FORECASTERS_TO_PUBLISH` 3 to 1 |
| `FT_0292_MERGED_AT` | `325b1b0` (PR #57) | 2026-07-24T19:16:26Z | forecasting-tools 0.2.92, MC option clamp 0.005/0.995 to 0.01/0.99 |
| `JULY25_MERGED_AT` | `73e4782` (PR #58) | 2026-07-26T04:38:40Z | the zero-output retry carve-out |
| `DRY_KEY_FIX_MERGED_AT` | `c3c91cb` (PR #59) | 2026-07-28T03:07:53Z | a drained donated key falls back to the personal key |
| `RANKED_MARKET_MERGED_AT` | `bfd5df2` (PR #61) | 2026-08-06T01:28:49Z | ranked prediction-market retrieval |
| `TIME_BUDGET_MERGED_AT` | `951f8e4` (PR #64) | 2026-08-26T17:23:30Z | the close-derived time budget |
| `LINTERS_MERGED_AT` | `eded193` (PR #65) | 2026-08-28T03:54:03Z | the lint campaign, which is provenance only and opens no sub-era |
| `FALL_CONFIG_MERGED_AT` | `8d5a082` (PR #66) | 2026-09-05T01:59:24Z | the next-season bundle: prompt de-bloat, the shared fetch ladder |
| `IMPERSONATE_RUNG_MERGED_AT` | `a9cbe03` (PR #67) | 2026-09-05T15:31:40Z | the TLS-impersonation fetch rung |
| `FALL_TARGET_MERGED_AT` | `660fd35` (PR #68) | 2026-09-07T05:52:20Z | the fall tournament target |

`GRID_SCALED_MAX_STEP_MERGED_AT` is an alias of `B4E9DF0_MERGED_AT`, because `9f1175c`
(grid-scaled max-step for discrete CDF resampling) rode that same merge.

**The widening flip is the worked example of the rule.** `b8d730f` authored it on
2026-05-12 and is not on `main`'s first-parent history at all: it reached `main` only inside
`0e85e1b`, six days later. Every round's era map until 2026-09-12 carried
`datetime(2026, 5, 12)`, the authoring date truncated to midnight. Nothing published moved,
because zero resolved records fall in the six-day window, which is exactly why it survived:
the defect was latent, and a backfill recovering May 12-18 records would have activated it.

**The tags.** Five pure functions in the same module read `bot_comment_created_at` (the
submission time, because a question was forecast under whichever config was live when the
bot published its comment) and return the fields a round's tagging pass writes:

| Field | Function | Values | Read by |
|---|---|---|---|
| `config_era` | `era_of` | `pre_flip`, `post_flip`, `triple_era`, `no_ts` | `era_gap` (its default `ERA_FIELD`), the clip sweep's era windows |
| `triple_subera` | `triple_subera_of` | `triple_pre_market`, `triple_ranked_market`, `triple_time_budget`, `triple_fall_config` | the coarse pooling, where every fall merge is one bucket |
| `triple_subera_fine` | `triple_subera_fine_of` | `pre_ft_unfreeze`, `ft_0292`, `july25`, `post_dry_key_fix`, `ranked_markets`, `time_budget`, `fall_config` | `era_gap --era-field triple_subera_fine`, which is how the fall read selects its arms |
| `ft_unfreeze_side` | `ft_unfreeze_side_of` | `pre_ft_0292`, `ft_0292` | the MC-clamp cuts |
| `post_linters_merge` | `post_linters_merge_of` | `True` / `False` | provenance only |

Every sub-era function returns `None` outside the triple era. These strings are a data
contract: a renamed value breaks a standing instrument silently, so add a value rather than
re-spelling one. A new sub-era is one appended row in `FINE_SUBERA_TABLE` (and a coarse row
only if it deserves its own pooled bucket), which is why the three September merges share
`triple_fall_config`: they landed inside three days, and separable rows would slice a period
of near-constant config.

`tests/test_performance_analysis_eras.py` pins each instant against `git log -1 --format=%cI`
on its merge sha and asserts that sha is on `main`'s first-parent history, so an assertion
cannot pass by feeding a constant back to itself. It also walks the package and fails on any
module that redeclares an instant rather than aliasing the object; that scan found a twelfth
copy the day it was written. Those assertions need the git objects, so CI's test job checks
out with `fetch-depth: 0` and the tests skip themselves on a shallow clone.

## The known-pipeline-bug cohort

**One canonical home: `KNOWN_BUG_QIDS` in `performance_analysis/cohorts.py`.** These are
questions whose published forecast came out of a since-fixed pipeline defect rather than
judgment, so pooling them into a calibration or miss-ranking row measures the retired
bug. It currently holds five ids:

- **43746 / 43747**: the pre-2026-07-07 open-bound arithmetic bug.
- **43913**, added 2026-08-25: the pre-`9f1175c` discrete max-step cap. All six
  forecasters stated 79.5-83% on the outcome that resolved; the published CDF carried
  20.00% with its first bin pinned at exactly 0.200000 on an 11-point grid. Receipts:
  `scratch/residual_2026-08-24/dossiers/43913_dossier.md`.
- **43147 / 41798**, added 2026-09-01: the same defect family on pre_flip discrete
  records: published mass at the resolving value pinned at exactly 0.200000 by the
  retired flat cap, while even the least concentrated member wanted 0.525 / 0.635 there.
  Peers −34.75 / −35.50. Identified by the shipped `max_step_clamp_screen`. Receipts:
  `scratch/residual_2026-08-31/dim_numeric-width.md`.

Import the constant instead of re-hardcoding the ids; every private copy in a round's
analysis scripts has drifted from it at least once. Nothing excludes the cohort by
default; a caller passes it explicitly (`--exclude-qids known_bug`) and the excluded
count is rendered per row, so an exclusion is a visible choice rather than a silent
filter. See `docs/operations.md` for the `--exclude-qids` mechanics.

## The degraded-run cohort (dry-donated-key incident, 2026-07-26 → 07-28)

The pre-fix dry-key window published eleven triple-era questions on a thinned ensemble.
Exclude them from headline aggregates and report them separately. Since 2026-08-31 they
live beside `KNOWN_BUG_QIDS` in `performance_analysis/cohorts.py`:

- `DEGRADED_RUN_QIDS`: full 1-of-3 publishes, gemini only (the personal-key-pinned
  slot), question ids **44870-44877**.
- `PARTIAL_DEGRADED_QIDS`: partial 2-of-3, **44841, 44856, 44912**.

Both are reachable as `from metaculus_bot.performance_analysis import DEGRADED_RUN_QIDS`
and both are wired into `--exclude-qids` under the shorthands `degraded_run` /
`partial_degraded`. Import them; three separate rounds hardcoded private copies before
the constants existed.

**These are QUESTION ids.** The same eight questions carry post ids **44721-44728**, and
minibench POST ids 44873-44877 land inside the question-id range, so a join that matches
"either id" admits five unrelated questions. Translate through
`performance_analysis/id_mapping`, never raw integers.

On the research side, 44841 / 44856 are degraded identically to the full cohort (native
search errored, both gap-fill passes dead), so a research-conditioned cut must exclude
both sets together even though the forecaster-count tagging separates them.

The first two resolved in 2026-08, both favorably: 44870 spot peer **+20.11**
(published on gemini alone; coverage-scaled peer +14.38), 44841 spot peer **+24.52**
(peer +21.54). That is a two-question favorable draw, not evidence that degraded
publishes are fine. Receipts: `scratch/residual_2026-08-24/degraded_cohort.json`.

## The tournament ranks on SPOT PEER

**Never rank or aggregate on the coverage-scaled `peer_score`.** Verified against the
live API on 2026-08-31: the project carries `score_type=spot_peer_tournament`, every
question's `default_score_type` is `spot_peer`, and `spot_scoring_time` equals
`actual_close_time` on all 158 posts pulled.

`peer_score` on the same record is `spot_peer_score × coverage`. Measured on the
2026-08-31 round's 30 new records, that identity reproduces the platform's own
`peer_score` to a **median residual of 0.69 points (max 13.05)**, and the residual is
crowd movement in the 1.5-3h window between our submit and the close rather than
anything the bot did. Those are that round's numbers, not a repo constant; re-derive
with `scratch/residual_2026-08-31/dossiers/44798_peer_vs_spot.py`.

Because the bot submits exactly once and never revises (forecast history length 1 on
157 of 158), its coverage is mostly a function of how early it submitted. So coverage
scaling FLATTERS misses and dulls hits: q44872 scored peer −15.0 against spot peer
−38.8.

`performance_analysis/platform_scores.py` is the one place that encodes the preference:

- use `spot_peer_score()` / `ranking_score()` rather than indexing `metaculus_scores`;
- report peer beside it as a labelled secondary;
- never sort a mixed set on whichever field happens to be present; `RankingScore.tier`
  keeps spot-scored and peer-only records in separate sort tiers.

Bot-side scores are a different quantity entirely and are unaffected: Brier and log
score in `performance_analysis/scoring.py`, `expected_baseline_score` in
`scoring_patches.py` (a log score against the community prediction, not a platform peer
score), and `backtest.py`'s own scoring, which never reads platform peer at all.

## Price a counterfactual with `spot_peer_delta`, never by hand

The halving gets applied twice or not at all. Metaculus computes spot peer as
`100·(N/(N−1))·ln(p/gmp)` and then HALVES it for a continuous question (numeric,
discrete, date). The crowd's geometric mean includes us, so a counterfactual that moves
only OUR mass on the resolving outcome is worth `100·ln(new/old)`, halved for
continuous, with no crowd term left. Read from Metaculus's `scoring/score_math.py` on
2026-09-02; fetched copy at
`scratch/residual_2026-09-01/dossiers/44798_verify_metaculus_score_math.py`.

Both conversions have already been got wrong, and both mistakes INFLATE the figure:

- `numeric_log_score` ALREADY carries the halving (it returns `50·ln(...)`), so a
  difference of two of its values is already in spot-peer points. The 2026-08-31 q45065
  cap-smear replay doubled exactly that and priced the near-miss counterfactual at up to
  +404 when the truth is +202.
- Thirteen 2026-09-01 dossier scripts quoted `binary_log_score` deltas (log base 2) as
  peer points, which OVER-states each one by 1/ln2 ≈ 1.44. Correcting an archived binary
  figure therefore means **multiplying** it by ln 2 ≈ 0.693, not dividing.

`spot_peer_delta` (`metaculus_bot/scoring_common.py`, re-exported from
`performance_analysis.scoring`) is the one implementation. It raises on an unrecognized
question type rather than silently taking the un-halved branch, and
`tests/test_peer_delta_convention.py` pins both conversions. Corrected q45065 figures
and the full per-script sweep: `scratch/residual_2026-09-01/DOSSIER_SYNTHESIS.md` §7.2.

## The era gap and the horizon confound

**Horizon matching is a standing part of the era read, and the roster watch is two-sided.**
Entry point `metaculus_bot/performance_analysis/era_gap.py`, read-only and offline:

```bash
uv run python -m metaculus_bot.performance_analysis.era_gap --dataset <round>/perf_all_tagged.json \
    --treated-era triple_era --comparison-era post_flip --strict \
    [--era-field config_era] [--clusters <round>/cluster_structure.json] [--output-json <path>]
```

Run it twice per round, once with `--strict` (the exclusion cohorts from `cohorts.py` dropped
from BOTH arms, the count shown in the `excl` column) and once unfiltered, the same pairing the
clip sweep uses. The dataset is the round's tagged pull: every record carries `config_era`,
written by the round's tagging pass off `bot_comment_created_at` against the merge-date era map,
and the treated and comparison eras are whatever values that field holds. The pass writes only
the coarse eras there; sub-eras go to their own fields (`triple_subera` holds
`triple_pre_market` / `triple_ranked_market`, `triple_subera_fine` holds `ranked_markets`,
`post_dry_key_fix`, `ft_0292` and so on), so a sub-era arm needs `--era-field <field>`, which
names the field both arms are selected on and is echoed in the report header. An arm with no
scoreable records, whether a value the field never carries or a sub-era none of whose questions
has resolved yet, is a clean non-zero exit rather than a bootstrap over an empty array: the
message names the arm and lists the values the field does hold, so a mistyped era name is
diagnosed on the spot, and the same condition raises `EmptyArmError` in the library. Every number
is spot peer through `platform_scores.spot_peer_score`. The report prints four blocks:

- **Arms.** n, effective n (distinct UTC resolution days), exclusions, unscoreable records
  (no spot peer, submit time or resolve time), spot mean and median, fraction negative, lag
  median and maximum, type mix and tournament mix, for the treated arm, the comparison arm and
  the comparison arm after the horizon cap.
- **Spot mean by within-arm lag quartile.** The confound made visible: a comparison arm whose
  mean falls across its quartiles was asked longer-horizon questions than the treated arm could
  have been.
- **Gap under each control**, treated minus comparison: unadjusted; type-adjusted (the
  pre-registered estimator of the 2026-09-01 and 2026-09-09 rounds); type-adjusted and
  horizon-matched (the watch row); type x lag-quintile adjusted, with and without the cap. Each
  carries a cluster-bootstrap 95% interval, `P(gap<0)`, the by-record interval and the verdict.
- **Per-type gap** against the horizon-matched comparison arm, on the types both arms carry.

**The estimator.** Each record's spot peer is residualized on the mean for its question type
over both arms pooled, and the arms are differenced. The pool is taken after the cohort
exclusions (bug records do not inform the nuisance means) and before the horizon cap, and the
lag-quintile cuts and cell means come from the same pool, so every row differs from the
type-adjusted row by exactly the control it names. Lag is `actual_resolve_time` minus
`bot_comment_created_at` in days: the forecast horizon, the thing the question asked of the
bot. It is deliberately not `resolution_set_time`, the batch date on which Metaculus set the
resolution, and calendar batching is not treated as clustering. The horizon match caps the
comparison arm at the treated arm's longest lag (inclusive) and leaves the treated arm alone;
the lag-quintile form is the reweighting counterpart, which attenuates a within-bin trend
rather than removing it (`tests/test_era_gap.py::TestHorizonConfound` pins both behaviours on
a synthetic arm whose score falls linearly with lag).

**Why horizon, the receipt.** On 2026-09-09 the retired six-model arm's spot mean fell
monotonically across its submit-to-resolve lag quartiles, **+18.33, +10.96, +9.57, +4.94**,
its longest lag was 113.1 days and its median 43.5, while the live three-model arm had never
been asked anything beyond 45.2 days (median 24.9). Type adjustment cannot see any of that:
the type-adjusted gap was +10.71 with a cluster interval of [+1.72, +19.69], and capping the
comparison arm at 45.2 days took it to +8.19 with [-3.21, +19.90]. A season's final wave is
always its long tail, so the comparison arm inherits the season's long-horizon questions the
treated arm was never asked, and the drift is one-directional (the roster that was around
longer looks worse the longer its questions took to resolve). This is the third conclusion in
this file's era-bucketing section that a control would have changed, and it is why the control
now runs every round rather than being re-derived by hand in a scratch script.

**Clusters.** By default one UTC day of `actual_resolve_time` is one cluster, on both arms, and
`eff n` counts them. The key is the event date, never `resolution_set_time`: Metaculus writes
resolutions in batches, but that is not what makes the day key coarse. On 2026-09-09 the 63-record
STRICT triple arm had 20 event days against 18 batch days and 58 curated clusters, and the 20
records resolving on 2026-08-01 were twenty unrelated "before August 1" questions (Putin's
approval, the hottest US state, a Bitcoin level, a Bank Rate decision), so the coarseness is
calendar-scheduled deadlines resolving many independent questions together. The clustered
interval is therefore conservative and the verdict reads it; the by-record interval is printed
beside it as the other bracket, and the report adds a **Brackets disagree on zero** line whenever
the two do not agree on whether zero is inside, because on those rows the verdict depends on the
convention. To let a round's curated clusters drive the interval, pass
`--clusters <round>/cluster_structure.json`: the file's `strong` clusters (one shared resolution
driver) collapse to one draw, `weak` members (correlated residuals, still separate draws) and
unlabelled records stay their own cluster, the report header names the file and how many records
of each arm it labelled, and `eff n` becomes the round's own effective n (58 and 178 on the
2026-09-09 file, which labels only the wave and the triple era, so 140 of the 181 comparison
records are unlabelled there). On that file the STRICT type-adjusted interval is [+0.03, +17.35]
against [-1.48, +17.13] on the day key, so the day key is what put zero inside on that row (the
report flags the disagreement), while the horizon-matched watch row reads no measurable difference
under both: +5.56 with [-5.56, +16.41] curated and [-7.59, +16.07] on the day key. The by-record
bracket is drawn from its own seeded stream, so it is identical under every cluster convention.

**The two-sided watch** (`era_gap.two_sided_watch`, read on the STRICT type-adjusted
horizon-matched row): a **concern** reopens only when the point estimate is below -5 spot-peer
points AND the 95% interval excludes zero; a **favourable** gap with an interval excluding zero
is reported, never flagged; anything else is **no measurable difference**. This replaces the
2026-09-01 ledger rule "reopen the underperformance flag if the estimate leaves
[-6.67, +13.90]", whose upper edge would have reopened an *under*performance concern on
evidence that the roster is better (operator ruling 2026-09-09).

**Reconciliation with the 2026-09-09 round**, whose primary estimator was ASYMMETRIC (treated
STRICT, comparison arm with its three flagged records kept) and pooled type means over all 256
summer records including the excluded ones. The module's symmetric `--strict` read on the same
dataset, 63 against 181 records:

| read | n | gap | 95% CI (clustered) | 95% CI (by record) | verdict |
|---|--:|--:|---|---|---|
| type-adjusted | 63 / 181 | +8.78 | [-1.56, +17.14] | [-0.09, +17.52] | no measurable difference |
| type-adjusted, horizon-matched (lag <= 45.2 d) | 63 / 95 | +5.56 | [-7.69, +16.06] | [-5.53, +16.59] | no measurable difference |
| type x lag-quintile adjusted | 63 / 181 | +8.15 | [-1.89, +16.43] | [-0.43, +16.65] | no measurable difference |
| type x lag-quintile adjusted, horizon-matched | 63 / 95 | +7.45 | [-5.63, +17.71] | [-3.37, +18.15] | no measurable difference |

The symmetric horizon-matched arm holds 95 rather than the round's 97 because the two
known-bug numerics (43746 and 43747, at -110 and -132) sit inside the 45-day window; that is
most of the move from +8.19 to +5.56. Composing the module's functions asymmetrically (treated
STRICT, comparison all 184) reproduces the round's rows within the pool convention: +10.79
against +10.71, horizon-matched +8.12 against +8.19, quintile-adjusted +10.42 against +10.21;
with the round's own 256-record pool the module's arms give exactly +10.71, +8.19 and +8.91 (the
round's "symmetric exclusions" row). The per-type horizon-matched rows match the round exactly
(discrete +27.41 on 12 a side, multiple choice -9.46 on 10 against 18). The unadjusted gap
against all 184 is +11.04 on both.

**Retarget the comparison each season.** The summer six-model arm is nearly exhausted; from
the next round the interesting cut is the fall configuration against the `ranked_markets`
sub-era (or the whole `triple_era`). The 2026-09-09 round's tagging pass (`era_tags.py` in the
round directory, a data table of `(tag, opens at)` rows keyed on merge-to-main committer
timestamps) writes `fall_config` into `triple_subera_fine` for every record submitted at or after
2026-09-05T01:59:24Z, the merge of PR #66. It is ONE bucket for the three September merges (PRs
#66, #67 and #68), because no question was forecast between the first and the last of them, so a
finer split could never hold data; a first cut that split the three would have tagged every fall
question `fall_target` and left the preregistered `fall_config` arm empty for good. The cost-pass
merge, once on `main`, is one appended row in that table and opens the next fine sub-era inside
the same coarse `triple_fall_config` bucket; the preregistration names which fine sub-era is the
primary arm. The preregistered read (`scratch_docs_and_planning/fall_2026_preregistration.md`) is

```bash
uv run python -m metaculus_bot.performance_analysis.era_gap --dataset <round>/perf_all_tagged.json \
    --era-field triple_subera_fine --treated-era fall_config --comparison-era ranked_markets \
    --strict --clusters <round>/cluster_structure.json
```

Run before the first `fall_config` question resolves (on 2026-09-09 six were forecast and none had
closed) it exits with `era arm 'fall_config' has no scoreable records; nothing to compare yet` plus
the field's value counts, and prints no report.

## Building a round's cluster structure

**The clustering rule is code; which questions share a driver is curated data.** Entry point
`metaculus_bot/performance_analysis/cluster_structure.py`, read-only and offline, writing the file
the section above consumes:

```bash
uv run python -m metaculus_bot.performance_analysis.cluster_structure \
    --dataset <round>/dim_category_slim.json --output-json <round>/cluster_structure.json \
    [--tables metaculus_bot/performance_analysis/cluster_tables.json]
```

Several questions in any wave are one real-world event or one dated statistical release seen from
different angles, so a record count overstates the evidence: on the 2026-09-09 round the wave's
largest August miss and its largest win were both readings of one cyclosporiasis outbreak. Each
question id gets exactly one cluster, at one of three strengths. **`strong`** means one driver
mechanically resolves every member (one Florida primary, one Employment Situation release, one
Metaculus Cup leaderboard), so the cluster collapses to one observation. **`weak`** means a shared
regime, so residuals correlate but the draws are separate, and it collapses only as a sensitivity;
report the strong-only effective n as primary and the strong-and-weak number as the conservative
bound. **`single`** means no sibling in the cohort, and every unclustered record becomes its own
`single_<qid>` cluster so nothing is silently unlabelled. The same resolution-set DATE, the same
CATEGORY and the same question TEMPLATE are none of them cluster bases: Metaculus writes
resolutions in calendar batches on unrelated quantities, which the emitted
`resolution_set_date_histogram` shows so nobody has to take it on faith, and the template families
(one question shape resolved off one kind of instrument) are reported for method correlation and
never collapsed.

**The curated tables are a tracked asset, `cluster_tables.json` beside the module.** They hold
which questions cluster, at what strength, with the basis text for each, plus the clusters retired
for id continuity, the template families, the links considered and rejected, the forecast-but-
unresolved questions that join on resolution, and the round's own caveats. That is a human's
per-round judgement rather than a rule, it grows every round, and it is edited in the JSON, never
promoted into Python. **Every id in it is a QUESTION id.** Post and question ids share one integer
namespace and the 2026-09-09 tables prove the collision is live: they carry question ids 44873 and
44874, which are also minibench POST ids, so a table matched against post ids admits unrelated
questions (`cohorts.py` carries the same warning for the exclusion cohorts). The asset needs its
own negation in `.gitignore`, because the blanket `*.json` ignore matches at any depth;
`tests/test_performance_analysis_cluster_structure.py` asserts `git ls-files` really carries it, so
the "exists locally, absent on a fresh clone" trap cannot come back.

**Three guards fail shut rather than mislabelling a round.** A cluster naming a question outside
the measured cohort raises: out-of-cohort siblings belong in the basis prose, named and not
counted, because a cluster that quietly reaches outside the cohort would collapse draws the round
never measured. A question claimed by two clusters raises. A retired cluster that still owns an
assigned member raises, which is how a cluster retired too early is caught; a retired id whose
members live on under new ids is the SPLIT case and is fine, as when `aug2026_asset_prices` became
the strong `brent_spot_aug2026` plus `aug2026_retail_fuel` plus `aug2026_rates_and_risk`.

**Cluster ids are assigned over the union cohort, but collapse is computed inside whichever slice
is being measured**, so a cluster straddling two eras collapses less inside a single-era arm. The
seven reported slices are the wave new since the prior round (all of it, and its post-flip and
triple-era halves), the whole triple era, and that era `clean` (degraded-run and known-bug records
dropped), `strict` (partial-degraded dropped too) and strict-and-new-only. Each slice name carries
its own record count, so `triple_strict_63` reads itself. The degraded and bug records are cluster
members and are dropped by the slice, so the strict effective n already reflects them.

**The rule reproduces the round it was extracted from.** Run over
`scratch/residual_2026-09-09/dim_category_slim.json` with the tracked tables it rebuilds that
round's `cluster_structure.json` field for field, all fourteen top-level keys including
`qid_to_cluster`, the per-cluster member blocks, `effective_n` and the prose, differing only in the
`source` path. `era_gap --clusters` on the rebuilt file prints a byte-identical report, watch row
included (+5.56 [-5.56, +16.41]).

## The clip-threshold sweep

**The clip floors are priced by a standing sweep, and a looser clip is censored, never
measured.** Entry point `metaculus_bot/performance_analysis/clip_threshold.py`:

```bash
uv run python -m metaculus_bot.performance_analysis.clip_threshold --cached <dataset> \
  --exclude-qids known_bug,degraded_run,partial_degraded
```

It reprices every resolved binary and MC publish under a grid of candidate floors `c`
(binary 0.005 to 0.10, MC 0.005 to 0.10; module constants `BINARY_FLOOR_GRID` /
`MC_FLOOR_GRID` in `clip_threshold_sweep.py`) and reports each in spot-peer points via
`spot_peer_delta`, floor-only / ceiling-only / symmetric.

**Windows.** The NESTED windows are `all` / `last_300` / `last_200` / `last_100` (MC
adds `last_50`, because its whole archive is under 100) / `last_90d` /
`current_clamp_regime` / `triple_era`. The DISJOINT config-era slices are
`era_pre_flip` / `era_post_flip`, with `triple_era` the third. The distinction is
load-bearing: nested windows re-count one set of records at different sizes and so
cannot disagree, while the era slices partition the dated records, so agreement between
them is real evidence.

**Censoring.** A candidate at least as tight as the clamp that was in force is exact
from the published value. A candidate LOOSER than it cannot be priced on a record that
sat at that clamp, because the per-member clamp erased the raw value. Those records are
counted and bounded, never estimated:

- `cen` keys on the published value.
- `cen_m` keys on a clamped MEMBER in a median position, and that is the rule that
  actually bounds what could have moved: an even roster averages two middle members, so
  a 0.02 member publishes 0.025.
- Bounds are labelled `at_floor` (nothing moved) and `at_c` (every censored value was at
  or below `c`), plus the identified bracket.

**The in-force clamp is looked up per record** from `bot_comment_created_at` against
`WIDENING_FLIP_MERGED_AT` (binary, `0e85e1b`, 2026-05-18T17:21:19Z) and
`FT_0292_MERGED_AT` (MC, `325b1b0`, 2026-07-24T19:16:26Z), both merge-to-main committer
dates, living beside `B4E9DF0_MERGED_AT` in `eras.py`, and `width_monitor.WIDENING_FLIP`
aliases the first.

**Each window carries an insurance view**: the break-even clipped-side rate, a Jeffreys
interval on the observed rate, the expected loss under the bot's OWN prices (spot peer is
proper, so a clip costs a calibrated forecaster that much regardless), and the best case.

**Selection discipline.** The report prints an out-of-bag value of the fitted argmax,
because the argmax is a choice over the grid and its own row's CI ignores that
selection. The out-of-sample rule is that a floor fitted on the records older than a
window ships only if it carries into the window, and a fit that moves nothing in its own
complement is flagged `moves nothing`; its carry of 0 is vacuous rather than a pass. A
row that moves no record renders `identity` rather than a CI.

**Result on 2026-09-02** (`scratch/residual_2026-09-01/clip_threshold/dim_clip-threshold.md`):

- The live clamp has bound NO binary publish since it went live: 70 strict post-flip
  binaries span 0.034 to 0.925.
- Raising the binary floor loses in every window and era. At c = 0.05 the pooled figure
  is **−217.48 over 447** records, of which −214.76 is the retired pre-flip regime and
  −2.72 the 70 live-regime records. 0 of 81 moved records resolved on the clipped side,
  against a break-even rate of 3.08%, and the properness cost alone is −91.19.
- An MC floor is a tax on every question: **−3.53 per question** at c = 0.05,
  era-stable.
- Loosening is bounded at **+10.91 over 447 binaries** and **+0.50 over 97 MC**, all
  pre-flip.
- Every out-of-sample fit is the do-nothing candidate.
- The only pro-tightening row in either cohort is the single-survivor degraded publish
  q44874, whose shape the thin publish floor prices at **+51.08 over the 4 genuine k=1
  publishes with zero cost to the other three**.

### The sweep model

`clip_threshold_sweep.py` holds the model and the math; rendering lives in
`clip_threshold_report`, so the dependency runs one way, CLI to report to sweep. Four facts
about its representation carry the rest of the module.

**Both question types live in one vector shape.** `ClipRecord.published` is the outcome-space
probability vector, `(p_no, p_yes)` for binary and the option vector for MC, which is what lets
the counterfactual, the replay and the censoring rules stay single-branch. The clamp semantics
still differ: binary clamps `p_yes` and takes the complement, MC clamps every option and
renormalises, and `apply_bounds` is the one place that branches on it. For the same reason
`clampable_indices` is `p_yes` alone on binary. `p_no` is its complement, so a floor on `p_yes`
is the same constraint as a ceiling on `p_no`, and counting both would report a publish sitting
at the ceiling as floor-censored.

**Tightening intersects with the clamp in force before it computes anything.** A candidate at
least as tight as the one that was live is fully determined by the published value, whether or
not that value was itself clamped, so `clip_delta` takes that intersection rather than pricing
the candidate on its own.

**The member censoring rule reads median POSITIONS, and that is what makes it exact.** A member
above the floor in a non-median position cannot move the median however low its raw value was,
which is why `member_censored` checks the positions the median actually reads rather than
counting any member at the floor. Under a mean aggregator every member moves the publish, so
every position counts. For MC the rule runs per option, against the members in THAT option's
median slot; it deliberately does not count a member floored on option j while sitting in
option k's median slot with a non-floored value. Renormalisation does couple the two, but the
round's refutation pass rejected "any member component at the floor" as overstating the bound,
and the coupled move is bounded by the floored mass a looser clip releases anyway.

**A floor can be infeasible on a ballot.** An MC floor `c` cannot be DELIVERED where a ballot
has more than `1 / c` options (eleven options each at least 0.10 already exceed 1), and the live
clamp then returns its sub-floor fallback. Such records are priced like any other but counted in
`infeasible_n`, so a cell labelled "floor 0.10" says on how many ballots that floor was not the
floor actually applied. `floor_infeasible` mirrors the degenerate test in
`clamp_and_renormalize_probs`; binary has one free value and is always feasible.

### Sweep constants and tolerances

These live in `clip_threshold_sweep.py` rather than in `constants.py` because they are analysis
parameters and nothing in the live pipeline reads them.

- **`BINARY_FLOOR_GRID` / `MC_FLOOR_GRID`** hold the candidate FLOORS, each implying the ceiling
  `1 - c`, and are module constants so that a round can widen the grid without touching logic.
  Every candidate must satisfy `0 < c < 0.5` for the clamp to be a clamp, which a module-level assert enforces: at `c >= 0.5` the bounds invert (`lo > hi`) and
  `apply_bounds` collapses every publish to `1 - c`.
- **`MC_UNSHIPPABLE_NOTE`.** forecasting-tools 0.2.92's `PredictedOptionList` validator clamps
  every option into [0.01, 0.99] on construction, so an MC floor below `MC_PROB_MIN` is not
  shippable today whatever this sweep says about it. Those rows are labelled (the `shippable`
  field) and reported rather than dropped.
- **`BINARY_CENSOR_ATOL` = 1e-9, `MC_CENSOR_ATOL` = 0.0015.** A record sits AT its in-force
  bound within this tolerance. Binary publishes are a median rounded to 3 dp, so a clamped one
  hits the floor exactly; MC options pass through a renormalisation that leaves 0.0101 or 0.011
  where 0.01 was clamped, and the MC tolerance is coarse enough to catch that drift and nothing
  wider.
- **`DELTA_ATOL` = 1e-9.** A record counts as MOVED when its spot-peer delta clears this. Binary
  deltas are exactly 0 when nothing moves, but MC vectors are renormalised, so an unaffected MC
  record's delta is float noise of order 1e-14 points. The same threshold guards the row's
  driver: `sweep_row` names a `top1_question_id` only on a row that moved something, because an
  MC row whose candidate is looser than the in-force clamp still carries about 1e-13 of
  renormalisation noise, and a share computed over that noise reads as a real concentration
  (0.07) and names a question the candidate never touched, on a row whose own `n_affected` is 0.
- **`REPLAY_DISAGREE_ATOL` = 0.005.** The published-vector counterfactual and the per-model
  replay count as disagreeing when the resolving mass differs by more than half a point of
  probability, the resolution at which a disagreement could plausibly have changed a published
  forecast. The same tolerance decides whether a replayed aggregate REPRODUCES the published
  vector, which is how `detect_aggregator` tells median from mean.
- **`ARGMAX_TIE_ATOL`**, equal to `DELTA_ATOL`, is how close two candidates must be in spot-peer
  points to tie for the argmax. Ties are the norm rather than an edge case: every candidate at
  or below a window's in-force floor scores exactly 0 when no publish in that window was
  clamped, so the winner is usually a plateau. That is why `argmax_rows` returns the whole tied
  set and `argmax_row` takes its smallest `c`, the least interventionist winner.
- **`_CLAMP_HISTORY`** holds the clamp in force, oldest regime first, as `(start_or_None, lo,
  hi)` rows. Every row is a LITERAL so that moving `BINARY_PROB_MIN` or `MC_PROB_MIN` cannot
  retroactively reprice the records published under the retired clamp; the two asserts beside it
  fail loudly and force an APPEND of a new regime rather than an edit to the last row. Binary
  [0.01, 0.99] predates the earliest archived record, which is why its first row has no start.
  An undatable record (`moment=None`) gets the WIDEST historical clamp, because a censoring
  claim needs to know which floor was live and the assumption that claims the least is the
  loosest one.
- **`LOW_PRICE_BINS` / `HIGH_PRICE_BINS`** are extreme-bin edges, `(label, lower, upper,
  p_midpoint)`. Low bins are half-open as `(lower, upper]` and high bins as `[lower, upper)`.
  The implied rate of the COUNTED event is the midpoint for a low bin and its complement for a
  high one, because a high bin counts NO resolutions.

## Receipts behind the survivor-conditional markers

`FORECASTERS_SURVIVED`, `EXTREME_CALL` and `THIN_PUBLISH_FLOOR` are described as mechanisms
in `docs/architecture.md` (section 4 "Survivor and extreme-call telemetry" and section 5
"The thin-publish floor"). The measurements that motivated them live here.

**`lone=true` is the cut worth having.** The 2026-08-31 gemini-slot review found lone
extremes (no other survivor extreme on the same side) right 4 of 9 times, against 21 of 23
for accompanied ones. **Do not pool `EXTREME_CALL` counts with the memo's own.** The memo's
scripts implement the looser "no other member extreme at all", which disagrees with the
marker on 4 of 570 archived extreme member-calls and reads pre_flip lone as 48 where the marker
reads 52 (post_flip and triple_era agree exactly). Two scope facts keep the numerator
honest: the marker is binary only (MC concentration is a different measurement and was not
adopted), and `lone` is vacuous at `survivors=1`, which is why the survivor count rides the
same line.

**The thin-publish floor was priced on one miss.** q44874 published a lone 0.03 on gemini
alone during the dry-donated-key window (the degraded-run cohort above) and scored −105.27
spot peer. Median-of-1 has no variance reduction, which is why the rule is keyed on the
survivor count and a multi-member median is never floored. The clip-threshold sweep above
prices the clamp at +51.08 over the four genuine k=1 publishes with zero cost to the other
three. Receipt: `scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md` §2.

## Record fields: what each collector record carries, and the traps behind it

`collector.py` `_process_single_question` builds one flat dictionary per question, and
`_comment_signals` supplies the fields that come off the bot's own published comment. Those
field names are a data contract with the research archive, so a field is added rather than
renamed or repurposed in place. What each one means, and the trap behind it where there is
one:

- `per_model_forecasts` carries one entry per ensemble member on a non-stacked comment. On a
  stacked comment it collapses to the stacker's single aggregate, so any median or spread
  computation has to read `per_base_model_forecasts` instead, which is what
  `stacker_detection.py` does on a stacked record.
- `per_base_model_forecasts` is what the stacker-combined round-one reasoning body still
  discloses, and it is empty on a non-stacked comment. Its shape follows the question type: a
  binary question gives `dict[str, str]`, for instance `{"gpt-5.5": "72.0%"}`; a
  multiple-choice question gives `dict[str, dict[str, float]]`, one option dictionary per base
  model; and a numeric or discrete question gives an empty dictionary, because those types
  carry their per-member detail in `per_model_numeric_percentiles`, whose parser already
  handles stacker-combined bodies.
- `per_model_numeric_percentiles` is `{model_name: [(percentile, value), ...]}` for numeric and
  discrete questions, and empty for binary and multiple choice.
- A non-empty per-option probability dictionary from `parse_per_model_mc_option_probs` IS the
  multiple-choice detector inside `_comment_signals`; there is no separate type check. The
  older single-string bullet parser returned only the top option line, which is why a
  multiple-choice question needs the full per-option vector and the option dictionaries win
  over the legacy parse.
- `_comment_signals` logs at DEBUG, never WARNING, when a comment marked stacked yields no
  per-model entries at all. A drifted producer-side delimiter is worth surfacing during triage,
  but the legitimate shapes look identical: a middle-trimmed comment, or a stacked binary or
  multiple-choice question with no percentile restatement.
- `was_stacked` is the legacy tri-state: True or False when the `STACKED=<bool>` comment marker
  is present, None on an older comment where stacking status cannot be determined at all. It is
  kept for backward compatibility. Prefer `stacker_outcome`.
- `stacker_outcome` is one of `primary`, `fallback_llm`, `fallback_median`, `fallback_mean`,
  `skipped` or `skipped_config_off`, and `stacker_outcome_source` records which of three rungs
  read it: `marker_outcome` (the `STACKER_OUTCOME=` marker), `marker_legacy` (the older
  `STACKED=` marker), `historical_body` (body-shape detection for a comment predating either
  marker) or `none`. The field exists because `was_stacked` collapses a median fallback and an
  outright skip into the same False or None, which is lossy for any stacking-treatment-effect
  cut. `skipped_config_off`, added 2026-07-19, separates a config-suppressed skip from a
  below-threshold one; comments earlier than that collapse both into `skipped`.
- `stacker_skip_reason` is `spread_below_threshold`, `config_off` or `single_forecaster`, from
  the additive `STACKER_SKIP_REASON` marker, and None whenever the stacker did not skip or the
  comment predates the marker. A bare `skipped` outcome cannot tell a below-threshold skip from
  the single-forecaster short circuit (q44870), which is what the field was added for.
- `forecasters_used` and `forecasters_configured` come from the `FORECASTERS_USED` marker: the
  number of forecasters that contributed to the published aggregate, which equals the per-model
  bullet count, and the roster size on that run. Both are None on a comment predating the
  marker. This is the BOT ensemble size and is not `metadata.nr_forecasters`, which is the
  Metaculus crowd count. `forecasters_used < forecasters_configured` marks a degraded publish, a
  model that dropped, rather than a roster change, which is what resolves the "fewer than N
  bullets" ambiguity named in `AGENTS.md`.
- `bot_comment_created_at` is the ISO-8601 timestamp on the bot's own comment, so a cohort cut
  can filter on SUBMIT date (the May-vintage stack, for instance) rather than the coarser
  `actual_resolve_time` stamp on the question.
- `metaculus_scores` is the platform's own `my_forecasts.score_data`: `spot_peer_score`,
  `peer_score` (both ascending, so negative is worse than the crowd), `spot_baseline_score`,
  `baseline_score`, `coverage`, `weighted_coverage` and `relative_legacy_score`. It is None on a
  record fetched before score data was captured, and populated on any fresh pull of a resolved
  question. Read spot peer rather than peer, and read it through
  `performance_analysis/platform_scores.py` rather than by indexing this dictionary, so the
  convention cannot drift per consumer. See "The tournament ranks on SPOT PEER" above.
- `metadata.nr_forecasters` is the Metaculus CROWD size, and it lives on the POST rather than on
  the question. Verified against archived post payloads: every post carries `nr_forecasters` and
  `forecasts_count`, and no question dictionary carries either. Reading it off the question with
  a 0 default is what made the field read 0 in every record pulled before 2026-08-25, and
  because a real 0 is not a missing key it also killed `audit.py`'s own `n/a` fallback. It is
  now None when the post genuinely omits the field, which lets a crowd-size cut drop those
  records instead of averaging a fabricated zero into them. Fresh pulls carry real counts,
  typically 100-250 on tournament questions. See "Two treatment tags read as TERNARY" above for
  the archive-side consequence: nothing rewrites the archive, so a 0 on an older record is
  unknown rather than a measurement.
- `metadata.resolution_set_time` is stored so a re-resolution has something timestamp-shaped on
  the record, and it is never the detector. Metaculus edited q44798 from 80 to 82 with this
  field left at `2026-08-31T21:38:45Z`, a stamp that PRECEDES the pull which still read 80. Diff
  the resolution VALUE instead (`performance_analysis/rescore_diff.py`); the field only helps
  once you already know an edit happened, for bounding when the original resolution was set. See
  "The round pull, and why `--prior` is mandatory" above.
- A resolved DATE question never becomes a record at all. The live bot forecasts date questions
  on the epoch-seconds axis of `numeric/date_axis.py`, so a resolved one does reach the
  collector, and `_process_single_question` skips it with a WARNING ahead of `parse_resolution`,
  which would otherwise file it under "Unknown question type" and make the exclusion look like a
  parser bug. See "Date questions are excluded from the dataset" below.

## Recovering per-model forecasts

The bot's published Metaculus comments are the durable per-model record: on non-stacked
questions the summary carries one `*Forecaster N (model)*: value` bullet per ensemble
member (post-clamp values). `performance_analysis/collector.py`
`build_performance_dataset` already parses these into `per_model_forecasts` /
`per_model_mc_option_probs` / `per_model_numeric_percentiles` /
`per_base_model_forecasts`. Consumers import the bullet regex and the `Model:`-prefix
attribution from `performance_analysis/parsing.py`, which re-exports the mechanics from
`performance_analysis/comment_sections.py`.

Gotchas:

- Comments longer than `COMMENT_CHAR_LIMIT` are middle-trimmed (`comment/trimming.py`);
  summary bullets survive, but rationale-body percentile detail may not.
- Stacked-era questions publish only the stacker's aggregate bullet; base values are
  recoverable only from self-declared rationale text (the `## Base Model Reasoning`
  sub-blocks).
- Soft-deadline drops mean some questions have fewer than N bullets.
- Old-era (May-June 2026) blocks carry retired tier-2 fields (`mixture_components`,
  `tails`, `distribution_family_hint`) that the strict `parse_structured_block` schemas
  reject wholesale. A tolerant raw-JSON fallback rung recovers the declared values from
  block-only rationales that would otherwise vanish: strict block → prose regex →
  tolerant salvage, added 2026-07-15, imported from `parsing.py` and implemented in
  `performance_analysis/declared_value_recovery.py`. That rung explains the false
  "gemini missed 5/45" screening artifact. The other historical offender, an edge-value
  `concentration: 0.0`, no longer needs the salvage: since 2026-09-02 the strict MC
  schema reads an unusable `concentration` / `other_mass` as absent instead of rejecting
  the block, because both fields were retired from the prompt and a dormant field must
  never cost a ballot.
- Roster drift makes era-conditioning mandatory; see the era-bucketing section above.

### Per-model cuts run on a filtered cohort; aggregates don't

When no `Model:` line identifies a bullet, the parser keys it by position instead
(`anonymous_model_key` → `Forecaster N`), and on a stacker-fired question that
positional bucket holds the stacker's aggregate. Pooling it across questions therefore
produces a stacker-vs-base-model mixture posing as one model. Measured: 50 such
forecasts in the 2026-04 data.

Every per-model cut in `analysis.py` (`per_model_binary_scores`,
`stacking_effectiveness`, `disagreement_predicts_error`) therefore goes through
`per_model_cohort`, which drops anonymous keys and drops records whose stacker is
*confirmed* fired, logging both counts at INFO under `PER_MODEL_COHORT`. Only the
confirmed verdict excludes: `likely_stacker` is a high-spread-plus-large-delta heuristic
that also matches an ordinary MEAN-era aggregate, so honoring it would delete the
high-disagreement questions those cuts exist to measure.

The audit's per-question rankings and the synthesis tally inherit the same guards via
`ranking_cohort.per_model_ranking_cohort`, which calls `per_model_cohort` rather than
restating it, so the rule cannot drift between the aggregate cuts and the dossiers.
Numeric rankings additionally drop declared percentile curves under
`MIN_SCOREABLE_ANCHORS` (9) distinct anchors, unless EVERY member on the record is
equally sparse, which is sparse-ERA output rather than a partial recovery and still
compares equals. Otherwise a sparse recovery gets PCHIP'd into a full CDF and
log-scored beside 11-anchor siblings, worth ~96 points either direction.

`max_step_clamp_screen` gates on that same shared floor (in its `_member_bin_masses`
helper), because the screen's verdict turns on the MINIMUM member bin mass, so one
sparse recovery can decide it. The floor lives in `parsing.py` precisely so those two
consumers cannot drift; `stacker_detection.py` and `audit.py` read it too.
**`declared_percentile_pit` deliberately does NOT gate on it**: it only linearly
interpolates the declared pairs in percentile space for a single quantile, where a
3-anchor curve is coarse but not a fabricated distribution, and gating there would
delete the uniformly-sparse-era records (fall-2025 comments declare 8-percentile sets)
whose PITs are valid. It does still exclude anonymous keys.

Aggregate and overall calibration paths are deliberately untouched by all of this; they
still count every record.

### The attribution parsers are guarded on two cohorts, and only one runs in CI

`tests/data/performance_comments_mini.jsonl` is a checked-in miniature: one real comment
per distinct SHAPE (attributable vs not, trimmed vs intact, with vs without the
`### Research Summary` boundary marker, named vs anonymized, all four question types),
redacted down to the structural skeleton the parsers key on. It is the deterministic CI
floor: `TestMiniFixtureAttribution` (`tests/test_performance_analysis_attribution.py`)
and `TestAgainstCheckedInMiniComments` (`tests/test_comment_trimming.py`) are not
skip-gated, so a parse or trim regression reddens every PR.

The broad sweep over `scratch/performance_data.json` (283 records, every era) still runs
locally and catches shapes the miniature has not been taught, but that file is gitignored
and rewritten by each collector run, so it can never be the only guard; a parse
regression hid behind exactly that gap until 2026-07-27.

Regenerate the miniature with `uv run python scripts/derive_mini_comment_fixture.py`
when a pull introduces a genuinely new shape. The derivation only admits a record whose
miniature parses IDENTICALLY to its full-size source, and the shape-coverage test fails
loudly if the set ever narrows.

The redaction keeps only what the parsers key on. The comments are real published
Metaculus text, so everything carrying no parser signal is elided: research prose,
per-model rationale prose, third-party news headlines, and the question title. What
survives is the structural skeleton, meaning section headers, `*Forecaster N*` bullets,
`Model:` lines, percentile, probability and multiple-choice option value lines, and fenced
JSON blocks verbatim. `parse_per_model_reasoning_text` is public but deliberately outside
the faithfulness filter, because the redaction exists to elide rationale prose, which is
exactly what that parser returns, so it necessarily diverges on every record. Its key set
still survives the shrink; only the bodies go.

`scripts/derive_mini_comment_fixture.py --emit-expectations` re-renders the test suite's
`_EXPECTED_PARSES_BY_POST` table from the checked-in fixture, so it needs no local pull.
The values come from running the real parsers over the fixture, which makes the table a
characterization of current behaviour rather than an independent specification. That is
the point, since hand-transcribing a dozen nested dicts is how a typo becomes an
"expected" value, and it is also the risk: regenerating after a parser change will happily
bless the change. Read the emitted diff and confirm each moved value is intended.

## An out-of-range resolution gives a SET-valued PIT reading, not a forced 1.0 / 0.0

Metaculus reports a resolution past the displayed range as the bare string
`above_upper_bound` / `below_lower_bound`, so the resolution VALUE is unknown and
`F(resolution)` is pinned only to an interval: `[cdf[-1], 1]` above the ceiling,
`[0, cdf[0]]` below the floor. The old forced convention counted q44842 as a coverage
miss even though that forecast deliberately put 13% of its mass above the displayed
ceiling and won spot peer +24.4.

The convention has exactly one home, `PitReading` and `out_of_range_pit_reading` in
`performance_analysis/analysis.py`, and the two conventions riding on it differ
deliberately. **Coverage** counts an interval as covered when it INTERSECTS the band, a
miss only when the whole interval lies outside. **Point** statistics (`pit_std`,
`mean_pit`, the histogram) EXCLUDE intervals and disclose the excluded count as
`n_oob_interval`, because imputing a midpoint would manufacture a reading nobody
measured. `docs/operations.md` describes what the width monitor prints, where the same
quantity appears as the `set-valued (pt n)` column.

When a bound is closed, or open with no out-of-range mass, the interval's endpoints
coincide and the reading degenerates to exactly the old value, so nothing moves on those
records. The measured effect on the archive is triple-era cov80 0.727 → 0.818, which at
n=11 is one record (q44842) going from miss to covered rather than a distributional
shift.

Two API notes for any analysis caller:

- `compute_pit_details` is now `compute_pit_reading` (`width_monitor.py`), renamed with
  no compatibility wrapper on purpose, so a stale caller fails with an ImportError
  rather than silently getting the retired convention.
- `EraWidthMetrics.pit_std` / `mean_pit` are `float | None` behind a
  `point_metrics_underpowered` gate and render as `n/a`, so JSON consumers must expect
  nulls.

## Reading the width monitor's era table

`width_monitor.py` reports how wide the bot's published numeric distributions are and
whether that width is calibrated, split by config era, every column read off the published
201-point CDF. It exists because the bot has oscillated between too wide and too narrow
across the two width-relevant merges:

- until **2026-05-18** the pipeline intentionally widened tails (`k_tail=1.25` in the
  tail-widening pass);
- on **2026-05-18** widening was turned off (`k_tail=1.0`, identity) after a calibration
  study found the widened tails too fat;
- on **2026-07-21** the july15 bundle landed, whose width-relevant piece is the
  time-series-anchor prompt clause. It pushes "sharpen, don't widen", because published
  low-tail coverage was about 0.03 against a 0.10 target, badly too wide, so the forward
  risk flips toward over-sharpening and this monitor is what closes the loop on that
  transition. The same merge dropped the forecaster roster from six models to the
  latest-per-vendor triple and lowered `MIN_FORECASTERS_TO_PUBLISH`, so a width shift
  across that boundary cannot be attributed to the anchor alone.

Nothing finer earns a bucket, per the era-bucketing rule above: a pipeline-behaviour change
starts an era, a git hash does not. Both boundaries are aliased in `width_monitor.py` from
the `*_MERGED_AT` constants in `eras.py`, as `WIDENING_FLIP` and `TS_ANCHOR_ENABLE`, so
that this table and the clip sweep's binary-clamp regime can never disagree. The dating
rule and the command that re-derives a boundary are in "Era boundaries are merge-to-main
timestamps, never authoring dates" above; the two spellings are in "Vocabulary that
collides" below.

### What each column means

- **central-80% coverage** (`cov80`) is the fraction of PIT in [0.10, 0.90], calibrated at
  0.80, and **central-50% coverage** (`cov50`) the fraction in [0.25, 0.75], calibrated at
  0.50. Both carry Beta-Binomial / Jeffreys-prior 95% CIs.
- **cov@10 / cov@50 / cov@90** are P(PIT <= 0.10), P(PIT <= 0.50) and P(PIT <= 0.90),
  calibrated at 0.10, 0.50 and 0.90. The outer two read low- and high-tail coverage; the
  middle one reads directional bias, how often the resolution landed below our median.
- **PIT std** is calibrated against the Uniform(0,1) standard deviation, 1/sqrt(12) or
  about 0.289. Smaller means the PITs are piled in the center, so the distributions are too
  WIDE; larger means piled at the extremes, so they are too NARROW.
- **median relative band width** is the median over questions of (P90 - P10) / |P50| read
  off the published CDF. It is the raw sharpness metric and depends on no resolution, so it
  answers "how wide are we in absolute terms" while the coverage columns answer "is that
  width calibrated".
- **band_miss** is the out-of-band rate, P(PIT < 0.10) + P(PIT > 0.90), which is exactly
  1 - raw cov80 and so carries nothing on its own. The lo/hi split is the point: it
  separates a band that is too TIGHT, both tails elevated, from one of roughly the right
  width that is MIS-CENTERED, with the misses piled in one tail. `cov80` cannot express
  that distinction, and the two call for opposite corrections. A set-valued reading misses
  a tail only when the WHOLE interval lies outside it, which keeps the identity
  band_miss == 1 - cov80 exact, since an interval that fails to intersect [0.10, 0.90] lies
  entirely on one side of it.

PIT itself is F_bot(resolution) on the canonical Metaculus value grid
(`build_cdf_value_grid`), and the two out-of-range cases differ by what the platform told
us. A string marker (`below_lower_bound` / `above_upper_bound`) gives no value, so the
reading is the interval our own published tail mass pins F to, and every coverage column
counts it on band intersection while PIT std and mean PIT exclude it. A numeric resolution
beyond the grid keeps a point PIT, scored off the members' declared-percentile curves
rather than the grid clamp. Both conventions are in "An out-of-range resolution gives a
SET-valued PIT reading, not a forced 1.0 / 0.0" above. The method mirrors
`scratch/calibration_audit_2026-07-16/mc_numeric_calibration.py`.

### Rows below ten PIT readings render their point metrics as `n/a`

`MIN_N_FOR_POINT_METRICS` is 10. Below that many readings a row's point metrics (cov@10,
cov@50, cov@90, PIT std, mean PIT and band_miss) are not estimates: their resolution is
1/n, coarser than the finest calibrated target they are compared against, cov@10 at 0.10,
so the value can only land on a grid whose spacing exceeds the quantity being measured. At
n=1 PIT std is exactly 0.0, which reads as "maximally too wide" while carrying no
information. Those cells render `n/a` in the markdown; the JSON keeps the raw values
alongside an `underpowered` flag, since a script can decide for itself but a reader cannot
un-see a number. `cov80` and `cov50` are exempt, because their CIs widen honestly at small
n, which is exactly the disclosure the point metrics lack. `pit_std` and `mean_pit` run on
the point-only denominator, so a row can clear the floor on readings and still fall under
it on point values; `point_metrics_underpowered` is that second gate.

### The clustered CI is real machinery and currently inert

The `cov80` and `cov50` CIs are computed at `n_eff`, the count of distinct `post_id`
values, rather than at the raw question count, so that a post carrying several correlated
sub-questions cannot narrow the CI as though they were independent: the collector expands a
`group_of_questions` post into one record per sub-question, and those share a series, a
window and a resolution source. Clustering is on `post_id` alone, the one grouping key
already on every record, and a record with no `post_id` counts as its own family through a
unique positional sentinel, so it is never merged with another such record. The point
estimate is untouched, still cov_k / n; only the CI width reflects `n_eff`, via
`jeffreys_ci(round(cov_k * n_eff / n), n_eff)`.

**The correction is inert on every dataset measured so far, and the table says so per
row.** Measured 2026-08-25 across all archived pulls (residual_2026-06-15 through
residual_2026-08-24, plus coherence_2026-07-15): every post carried exactly one resolved
record, so `n_eff == n` everywhere and the rendered CI is the naive one. The mechanism
stays because a group post resolving into the tournament is a matter of question supply
rather than of code, but nothing may claim the CIs were widened unless
`EraWidthMetrics.ci_clustered` says they were, which is what the `n_eff` cell's
`(widened)` / `(=n)` marker reports. An earlier version of this reasoning asserted that
about 62% of records share a post; no archived dataset supports that figure.

### What `max_step_clamp_screen` looks for, and why its cap is era-dependent

The screen answers one question: did a per-bin max-step cap, rather than the forecasters,
decide the published mass at the truth? On a coarse discrete grid the pre-`9f1175c` flat 0.2
cap can hold the realized bin far below what every member asked for. q43913 published 0.200
where the members' own curves wanted 0.575 to 0.823, worth spot peer −41.20 and
coverage-scaled peer −38.67. That is a pipeline defect posing as a forecast error, and it
manufactures apparent dissent: each member keeps its concentrated mass while the published
curve does not.

The cap is looked up per record against the submit timestamp: the flat 0.2 before
`GRID_SCALED_MAX_STEP_MERGED_AT`, the record's own `grid_step_constraints(len(cdf))` maximum
after it. Without that gate every post-fix coarse-grid discrete that legitimately holds a
0.2 bin false-positives. A missing or unparseable timestamp reads as pre-fix, since every
undated record in the archive predates the fix.

A record is *suspected* only when all three hold: the realized bin is cap-bound (within
`_CLAMP_CAP_ATOL` of the era-correct cap, or at least `_CLAMP_CAP_NEAR_FRAC` of it, because
the min-step, ramp and discrete-snap machinery shaves a saturated bin about 1% under the
cap), at least two attributed member curves exist, and the LEAST concentrated member wants
at least `member_margin` more mass on that bin. "Every member" is the point: a clamp
overrides the whole ensemble, unlike a median. A cap-bound bin is not automatically our
defect, because the cap is the platform's own per-bin rule (`0.2 * 200 / N`); pre-`d4ee57f`
records additionally carry the slack-proportional smear, while post-`d4ee57f` the excess is
packed into adjacent bins.

`stacking_effectiveness` in the same module is a COUNTERFACTUAL cohort cut, not a record of
what the pipeline did. It buckets binary questions by whether their measured spread would
have tripped the production stacking trigger (strictly greater than the threshold, matching
production) and reports mean Brier per bucket. Stored data cannot say whether a given
prediction was actually stacked, so the cut shows how the trigger metric correlates with
outcome difficulty and nothing more.

## A starved outer tail is a different defect from the max-step smear, and it is systematic

On an open bound the declared outer tail can end up routed past the displayed range
entirely, leaving every in-range bin above the members' declared p99 pinned at the
platform's per-bin minimum step. Every resolution in that band then earns the same floor
score (~−219 at any grid size), so it is a CLIFF at a fixed location rather than a band
of the wrong width, which is why widening does not fix it, and why shipping the
detector is not in tension with the standing `k_tail` hold.

`scan_outer_tails` (`performance_analysis/outer_tail.py`, printed by the width monitor's
CLI) triggers on the band's MEAN per-bin mass expressed as a multiple of the platform
minimum step (`STARVED_OUTER_TAIL_FLOOR_MULTIPLE = 2.0`). It deliberately does **not**
trigger on the plan's proposed `tail_mass = 1 − F(p99)`, which carries no signal at all:
with the canonical anchors that quantity is ≈0.01 on every record by construction, so
q45218 reads 0.0142 and would not fire at any threshold that does not fire on nearly
everything. The multiple is scale-free, which is what tells a 2-bin band holding 0.003
(real density, harmless) from a 27-bin band holding 0.004 (the cliff).

```bash
python -m metaculus_bot.performance_analysis.width_monitor --cached <path>
```

prints a per-question "Starved outer tails" section after the era table;
`--output-starved-json` writes every scanned side, and `--exclude-qids` cohorts apply.
`docs/operations.md` describes the per-row fields and the member census.

**The first result is itself the finding**: it fires on 68 of the 417 measurable
open-bound sides across 49 questions, 19 of them starved on both sides, with 44 sides
sitting essentially exactly at the pipeline's own applied floor. So read a fire as "this
question carries a cliff", not "something broke here".

That calibration (the ~1.1x applied floor, the 44 sides in [1.00, 1.25)) is a Metaculus
measurement: on a Metaculus aggregate the pipeline's structural out-of-range tail is 1% and
the in-range band above the declared p99 sits at the platform minimum step. A published
Mantic aggregate is shaped differently by construction, because `floor_published_tails`
(`numeric/out_of_range_floor.py`) raises each open tail to at least 5%, as far as the other
tail leaves room, and rescales the interior, so its `tail mass` would read 0.05 rather than 0.01 and its band multiples would
not match these figures. Today that is moot: no Mantic record enters the dataset (the
collector is Metaculus-only, see the archive section above). It is the first thing to
re-derive if one ever does.

There is deliberately NO publish-time `STARVED_OUTER_TAIL` WARN. The reason, and what a
no-plumbing alternative would have to measure instead, are in the code comment above
`STARVED_OUTER_TAIL_FLOOR_MULTIPLE` and tracked in `FUTURE.md`.

## Question-supply counts need post status `closed`

`scripts/supply_probe.py` (`make supply_probe`, read-only and free: the Metaculus posts
list plus post detail) is the tracked replacement for two rounds' worth of scratch probes
that each queried only `statuses=open` and `statuses=resolved`, and so missed the 178
summer-tournament posts sitting at `closed` (closed to forecasting, not yet resolved),
26 of them the frozen-triple checkpoint cohort.

It also reports the backlog of unresolved questions past their own
`scheduled_resolve_time`, which is what separates "Metaculus is late resolving" from
"our pull is missing questions". Resolution is read per QUESTION, not per post, since a
group post's members resolve on their own schedules.

Since 2026-09-02 it also sweeps **FORFEITS**, every question on a `closed` or `resolved`
post that the bot never forecast at all. A forfeited question never enters the performance
dataset and so is invisible to any sweep that starts from questions the bot intook, which
is why the sweep belongs in the weekly read rather than in a round's scratch scripts. The
mechanics, the per-post cost, the `unknown` state and the six triple-era forfeits the
2026-09-01 round found are in `docs/supply_probe.md` "The forfeit sweep". Default slugs
come from the repo's constants; see `docs/operations.md` "Season-start checklist".

## Date questions are excluded from the dataset

The live bot forecasts date questions from the merge of the Mantic Crucible bundle (authored
2026-09-08, live only once it lands on `main`). Mantic's Crucible made them first class, and the
Metaculus run modes forecast them too where they used to be skipped, so a resolved date question
now reaches the collector. It does not enter the dataset:
`collector.py` `_process_single_question` skips it with a WARNING naming the question and
the post (`Skipping Q<id> (post <id>): date question, excluded from residual analysis by
decision`), ahead of `parse_resolution`, which would otherwise have filed it under "Unknown
question type" and made the exclusion look like a parser bug. The backtest
(`backtest/question_prep.py`) and the ablation harness (`ablation/run_pdf.py`) carry the same
exclusion at their own seams, and `scripts/score_ghosts.py` counts date ghosts by type and says
in its report that none can be scored. The decision is recorded in `FUTURE.md` "Mantic
Crucible" and holds until a date question has resolved under the live date path. A round pull
over a tournament with date questions logs one such WARNING per resolved date question; their
count is the number of resolved questions the round is not reading.

## Vocabulary that collides

Four pairs of names describe one thing, or two different things under similar names.
Keeping them straight is the same discipline the era rule asks for.

**Era-boundary constants have two vocabularies for one instant.** The width monitor
names its boundaries `WIDENING_FLIP` and `TS_ANCHOR_ENABLE`; both are aliases defined in
`width_monitor.py` of `eras.py`'s `WIDENING_FLIP_MERGED_AT` and
`B4E9DF0_MERGED_AT`. So the width monitor's `TS_ANCHOR_ENABLE` and the clip sweep's
`B4E9DF0_MERGED_AT` are the same 2026-07-21T17:07:37Z instant under two names. Prefer
the `*_MERGED_AT` names in new code, and never introduce a third spelling.

**Two anchor-count thresholds mean different things.** `MIN_SCOREABLE_ANCHORS` (9,
`parsing.py`) is the distinct-anchor floor for per-model RANKING and the max-step clamp
screen; below it a curve gets PCHIP-rebuilt into a full CDF and log-scored, worth ~96
points either direction. The outer-tail scan's own rule is separate and far lower: it
drops a member curve carrying fewer than two distinct percentile labels, because a
single recovered pair interpolates to a constant PIT at every resolution. They are not
in conflict; they gate different computations at different costs.

**`n_oob_interval` and the rendered column `set-valued (pt n)` are the same quantity**:
the count of out-of-range-interval PIT readings excluded from the point statistics. The
field name is what a script reads; the column label is what the width monitor prints.

**The `ts_anchor` era's absence has two true causes, in sequence.** A phantom
one-record `ts_anchor` era once existed because the boundary was keyed on an authoring
date; its tell was that the record's own comment named the retired `grok-4.5` /
`gpt-5.5` / `opus-4.6` roster, dropped by the same merge that landed the anchor. That is
fixed. The row is absent *today* because empty eras are omitted and no post-july15
numeric has resolved. Neither statement is stale; read them in that order.

## The round dataset builder

Every round produces one `perf_all_tagged.json`, the dataset every downstream lane reads. The
rules for building it are the same round to round and live in two tracked modules:
`performance_analysis/round_dataset.py` (load, dedup, heal, tag, cohort) and
`performance_analysis/round_outputs.py` (the four output files and the console report).
The package CLI pulls and reports one Metaculus tournament slug per call. For the tagged
multi-source round dataset, callers provide a `RoundSpec` to the maintained library API below.
There is no integrated command that collects all round inputs and assembles the full round.
Do not create a per-round scratch driver or copy standard dimension scripts; `scratch/residual_<date>/`
is for round inputs and outputs. Use a scratch script only for a distinct follow-up analysis.

### Calling the round dataset API

This is a direct invocation/configuration example for the tracked API, not a request to save a
new round script.

```python
from pathlib import Path

from metaculus_bot.performance_analysis.round_dataset import RoundSpec, build_round_dataset
from metaculus_bot.performance_analysis.round_outputs import write_round_outputs

round_dir = Path("scratch/residual_<date>")
prior_dir = Path("scratch/residual_<prior-date>")
spec = RoundSpec(
    round_dir=round_dir,
    label=round_dir.name.removeprefix("residual_"),
    prior_dir=prior_dir,
    prior_label=prior_dir.name.removeprefix("residual_"),
    telemetry_dir=Path("backtests/telemetry_archive"),
    weighted_slug="<weighted-slug>",
    required_slugs=("<required-slug>",),
    optional_slugs=("<optional-slug>",),
    reused_slugs=("<reused-slug>",),
)
write_round_outputs(build_round_dataset(spec))
```

Stage the conventional inputs in the round directory before calling the API: `perf_<slug>.json`
for each pulled tournament, `question_weights.json` for leaderboard weights, and
`platform_rescored.json` for the pull's re-resolution diff. A required slug whose file is missing
logs a WARNING and contributes nothing; an optional slug whose file is missing is silent, which
is how a probed-but-empty successor tournament is meant to behave. `label` and `prior_label` are
the provenance strings a record carries, so they are the round names and not paths.

### What the spine does, in order

1. Loads the prior round's tagged file once, and reads three things off it: the baselines this
   round does not re-pull, the cross-round rescore provenance, and the "was this already here,
   was it already scored" snapshot behind `is_new_since_prior` and `newly_scored`.
2. Loads each fresh pull, strips every tag field it owns off a reused record, and dedups on
   `(question_id, post_id)` preferring the fresh record. A reused record of the weighted slug
   that survives dedup means the fresh pull lost a question, and that is logged as a warning.
3. Carries `rescored_fields` and `platform_rescored` forward BY KEY rather than on the reused
   record itself. A fresh record wins dedup over its reused twin, so a carry on the record would
   be thrown away and "Metaculus re-resolved this once" would become indistinguishable from
   "never rescored".
4. Heals the stored scores through `collector.rescore_records` and records the per-field deltas.
   A fresh record changing here is a red flag, because the pull was scored by the current
   collector; a reused record changing is the expected healing of a stale stored value.
5. Stamps the era tags from `eras.py` and the leaderboard `question_weight`, which is set only on
   the weighted slug's records and is `None` everywhere else.
6. Pins the degraded cohort by joining the telemetry archive's `forecaster_drops` markers to that
   run's per-question markers, then unions the result with `cohorts.py`'s canonical set, so a gap
   in the archive cannot silently un-tag a known-degraded question.
7. Flags novelty and the exclusion cohorts, then tags Metaculus-side score movement and
   cross-checks the local ternary against the pull-side diff's distribution.

### Two invariants

**Field insertion order is part of the output.** `perf_all_tagged.json` is compared byte for byte
between rounds, and Python preserves dict insertion order into JSON, so the tagging steps assign
in a fixed order: provenance, then carried provenance, then healing, then era tags and weight,
then novelty and cohorts, then the platform-rescore fields. Reordering the steps rewrites the
file without changing a single value.

**The tag vocabulary is a contract.** `era_gap.py` selects its arms with
`--era-field triple_subera_fine`, and `clip_threshold.py` slices on the `pre_flip` / `post_flip`
vocabulary. A renamed tag value breaks a standing instrument silently, which is why the tag
functions live in `eras.py` and are imported rather than rewritten.

### The reproduction receipt

The spine was extracted from `scratch/residual_2026-09-09/bucket_by_era.py`, the tenth copy.
Running the tracked code over that round's own inputs reproduces `perf_all_tagged.json` (905
records, 88 MB), `new_since_prior.json` (78 records) and `degraded_cohort.json` byte for byte,
and reproduces every block of `counts_by_era.json` except two. `generated` is a wall-clock stamp.
`boundaries_utc.flip` now reads the corrected merge instant 2026-05-18T17:21:19Z instead of the
2026-05-12 authoring date every scratch copy carried; no record was submitted inside that
six-day window, so no record's era, tag or score moves with it.

### Bot-side healing versus platform re-resolution: four field families, two owners

A round carries two independent "this number changed" stories and they must never share a name.

`rescored_fields` (with `rescored_fields_this_round` and `rescored_fields_prior_rounds`) is OURS:
the bot-side score fields `collector.rescore_records` recomputed from the record's own stored
inputs. Those scores are pure functions of inputs the record already carries, so a change means
our scorer changed, not the platform.

`platform_rescored` (with `platform_rescored_this_round`, `platform_rescored_prior_rounds` and
`platform_rescored_pull_tag`) is METACULUS's, detected by `rescore_diff.diff_platform_rescores`.
It exists because Metaculus can change a resolution after the fact without moving any timestamp
we store. On 2026-08-31 it resolved q44798 (post 44645, "Halo: Campaign Evolved Metascore") at
80, the PS5 hero card on Metacritic, and then within 26 hours edited it to 82, the Xbox card the
resolution criteria actually name. `resolution_set_time` still read 2026-08-31T21:38:45Z
afterwards, which PRECEDES the pull that read 80, so nothing timestamp-shaped could have flagged
the edit. That record's spot peer went from +5.41 to -5.42 between two consecutive rounds, and
every table the earlier round published about it was silently stale. The only reliable detector
is a value-level diff of the pull against its predecessor: `resolution_raw` and
`resolution_parsed` verbatim, plus every key of `metaculus_scores` on either side, so a field
Metaculus adds later is diffed with no edit to the code.

`rescore_diff`'s own tag is deliberately three-state, because "compared, nothing moved" and
"never compared" are different facts and the second is what a run with no `--prior` produces.
`None` means no prior record existed for that `(question_id, post_id)`, `False` means compared
and unchanged, `True` means at least one field moved. On a `True` record only,
`platform_rescored_fields` names the fields, `prior_resolution` carries the prior
`resolution_raw` (equal to the current one on a score-only re-score, which is how a reader tells
the two cases apart) and `prior_metaculus_scores` carries the prior score block. Those two prior
snapshots are attached to moved records only: on an unchanged record the current values ARE the
prior ones, and copying them everywhere would double the dataset's size to say nothing. The old
`resolution_parsed` value is not recoverable from the tag, so `render_rescore_summary` reports it
as `None` rather than guessing; `resolution_raw` moves with it in every real case and that row
carries the values. A summary over records where nothing was compared says exactly that and
claims nothing about staleness, because a dataset that never went through the diff and a prior
pull with no overlapping key produce identical tags.

`RESCORE_ATOL` is 1e-6, shared by `rescore_diff` and `collector.rescore_records` so both sides of
a round comparison use one threshold. The platform's scores round-trip through JSON exactly and
our scorer reproduces them to about 1e-14, while the gaps this exists to catch are whole points:
the known-stale q44798 gaps start at 0.6.

### The three drift transitions

`round_dataset.score_transition` classifies every prior-versus-now score pair the round compares,
and each entry of `counts_by_era.json`'s `bot_log_score_drift_detail` and `platform_drift_detail`
carries the answer in its own `transition` field beside the `prior` and `now` values:

| Transition | What it means |
|---|---|
| `moved` | Both rounds carried a value and they differ by more than `SCORE_ATOL`. |
| `disappeared` | The prior round carried a value and this round carries none. |
| `appeared` | The prior round carried none and this round carries a value. |

The two one-sided cases used to be skipped outright: the guard required both sides non-null, so a
score Metaculus withdrew read as no drift at all. That made the top-line
`metaculus_platform_score_drift_fields` report zero and suppressed its warning while the pull-side
diff, which has always treated a one-sided null as a change, reported the same record as rescored.
A withdrawn score is the anomalous direction and the one a stale published table most needs to
hear about, so it is now drift.

The transition is a label on values the entry already carried rather than a separate count block,
because the measured frequency says no reader will ever have a long list to skim. Over the nine
archived rounds that carry a tagged file, 587 fresh-pull records matched a prior counterpart and
produced zero `appeared`, zero `disappeared`, and exactly one `moved`: q44798 on both `peer_score`
and `spot_peer_score` in the 2026-09-01 round. `appeared` is close to structurally impossible on a
compared record, because the collector pulls only resolved questions (`resolution_raw` is non-null
on all 905 records of the 2026-09-09 round) and Metaculus scores at resolution, so a record that
survives to a second pull already carried its score block in the first. The only null platform
scores in the whole archive are ten `fall-aib-2025` records with no `metaculus_scores` block at
all, and that slug is reused rather than re-pulled, so they never enter the comparison.

### Why the two sides of a drift comparison are read differently

`_drift_against_prior` reads the prior side as `prior_view.get("peer_score")` and the current side
as `peer_score(record)`, and that asymmetry is correct rather than a bug to tidy. The two arguments
are different shapes. `record` is a live perf record, which carries its platform scores only nested
under `metaculus_scores`; no archived record has ever carried a top-level `peer_score`, in any of
the 6,353 records across the nine tagged rounds. `prior_view` is not a record at all: it is one
value of the `prior_snapshot` index, an eight-key flat view that deliberately hoists
`metaculus_scores.peer_score` and `.spot_peer_score` to top-level keys and keeps no nested block.
So on the view the flat read is the only one that works and an accessor returns `None`, while on the
record the reverse holds.

Making both sides symmetric breaks the report in whichever direction you pick. Routing both through
the accessors reads `None` for every prior value, which under the transition classifier turns every
single re-pulled record into an `appeared`: 358 spurious entries on the 2026-09-01 to 2026-09-09
comparison alone, 179 records times two fields, plus the alertable warning, in a round where
Metaculus re-scored nothing. Routing both through the flat read would break the current side the
same way. `PLATFORM_DRIFT_ACCESSORS` is the one table both ends use, so `prior_snapshot` hoists
exactly the fields the drift path later reads flat and the two cannot drift apart; the snapshot goes
through the accessors rather than indexing `metaculus_scores`, which keeps the spot-peer rule intact.
`TestPriorViewIsFlatByConstruction` pins the shape of both arguments so a symmetric-looking
"cleanup" fails a test that says why.

Worth noting what the transition classifier bought here. Under the old both-non-null guard this
same mistake would have been silent forever, because a `None` prior made the platform branch
unable to report anything at all. It now announces itself as hundreds of `appeared` entries and a
warning on the first real round, which is the failure mode a guard should have.

### The two rescore paths ask different questions, and should not be made to agree

`rescore_diff.diff_platform_rescores` runs on the PULL side and asks whether Metaculus touched
this question at all. It diffs this pull against the prior round's raw `perf_<slug>.json`, over
`resolution_raw`, `resolution_parsed` and every key of `metaculus_scores` on either side.

`round_dataset._drift_against_prior` runs on the DATASET side and asks whether the numbers a round
actually ranks and publishes on moved. It reads `peer_score` and `spot_peer_score` through
`platform_scores.py`'s accessors, against the prior round's `perf_all_tagged.json`, which is the
dataset every downstream lane read and therefore the baseline a stale table came from.

Different baselines and different breadth, so the counts are not expected to match, and neither
path should be widened to imitate the other. The dataset side must keep reading through the
accessors rather than indexing `metaculus_scores` directly, or the continuous-question halving
lands twice or not at all. The one thing that must agree is checked: `_log_pull_tag_agreement`
warns when the local ternary `platform_rescored_pull_tag` distribution departs from the pull's own
`tag_distribution`. Both sides do read the same quantity for the two shared fields, by different
route on each end, for the reason the previous section gives.
