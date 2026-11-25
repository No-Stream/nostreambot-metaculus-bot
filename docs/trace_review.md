# Trace review

The trace review tool lets a person read the bot's resolved forecasts one at a time, with every member's
value, rationale, research and telemetry on one page, and filter or sort the whole history to find the
forecasts worth reading. It has two parts. `make trace_index` builds an offline index (a small parquet
bundle) from the newest residual round, a fresh read-only pull of what has resolved since, the
telemetry archive and the research archive. `make review` opens a Streamlit app over that bundle on
localhost. Neither spends credits or publishes anything; the build's only network traffic is the
collector's paced GETs of resolved posts and the bot's own comments.

## Commands

| Command | What it does |
|---|---|
| `make trace_index` | Builds `backtests/trace_index/` from Metaculus, plus Mantic when `MANTIC_TOKEN` is set (a warning says when it is skipped). Reads `.env`. |
| `make sync_all` | Runs the same build as its last step, so the index stays current with every sync and needs no separate command. |
| `make trace_index ARGS="--platforms metaculus mantic"` | Names the platforms explicitly; a named platform without its token fails. |
| `make trace_index ARGS="--skip-fresh-pull"` | No network at all; the index stops at the round, and the app says so. |
| `make trace_index ARGS="--refresh"` | Ignores the cached pulls and pulls again. |
| `make review` | The app on `localhost`, with Streamlit's usage statistics off (it would otherwise report to its vendor). |

Other builder flags: `--perf-dataset PATH` picks a round file other than the newest
`scratch/residual_YYYY-MM-DD/perf_all_tagged.json`, and `--output-dir PATH` writes the bundle
elsewhere. The app reads `$TRACE_REVIEW_BUNDLE_DIR` when it is set, which is how the tests point it at a
fixture bundle. A full Metaculus build on 2026-11-19 took about two minutes, most of it the paced pull,
and peaked at 1.5 GB of memory.

## What the build pulls

The round file is the base: its rows keep the round's values and tags, so the index agrees with the
residual round it rests on. On top of it the builder pulls the current-season Metaculus tournaments
(`TOURNAMENT_ID`, `METACULUS_CUP_ID`) plus every slug the round itself pulled fresh, and Mantic's
configured tournament plus every slug a Mantic run has named in its `mantic_tournaments` marker.
Questions the round does not hold become `record_origin == "fresh_pull"` rows, tagged by
`round_dataset.tag_fresh_records`, the same tagging the round uses. A round row that the platform has
re-resolved or re-scored since keeps the round's values and gets `rescored_since_round`; reading the new
values is the next round's job. The comment listing is fetched once per platform, and only posts the
round does not already cover get their archived comment expanded, because each expansion is one paced
request.

Each pull is cached under `backtests/trace_index/pulls/<platform>/` with a small metadata file and reused
for six hours (`PULL_MAX_AGE_HOURS`) as long as it was made against the same round. A build writes its
five files into `.building/` and moves them into place only after all of them are written, so a build
that fails (a dropped connection, a duplicated key, a missing token) leaves the previous bundle as it
was. Asking for Mantic without `MANTIC_TOKEN` fails at once with a message naming the variable and the
`--platforms metaculus` way out: an unauthenticated Mantic listing has no `my_forecasts` block, so it
would silently hold none of the bot's forecasts.

## What is left out

Date questions, Mantic multi-resolution questions, annulled or ambiguous resolutions and `[PRACTICE]`
posts never become rows, the same as in the residual dataset. The manifest counts each kind among the
resolved questions the bot forecast in the tournaments this build pulled, and the list page shows those
counts. They do not cover tournaments that only the round file holds. Multi-resolution questions are
excluded because none has resolved yet and how one reports its resolution is unread
([performance_analysis.md](performance_analysis.md) "Mantic records").

## The bundle

| File | One row per | Holds |
|---|---|---|
| `forecasts.parquet` | resolved forecast | identity, timing, question and outcome, every platform score, ensemble and cohort tags, research tags, cost, trait roll-ups |
| `members.parquet` | forecast x member | the member's value, its own replayed score, raw and published values from `member_forecast`, extraction rung, rationale traits |
| `texts.parquet` | forecast | the comment, the research text, the gap-fill v2 record and provider results (read one forecast at a time) |
| `telemetry.parquet` | marker line | every archived line joined to a forecast, plus its run's run-level lines |
| `manifest.json` | build | build time, round, pull metadata, row counts, exclusions, trait prevalence by config era |

Column names and dtypes are declared once in `metaculus_bot/trace_review/schema.py`, and both the builder
and the app cast to it, so a drifted column fails at the boundary. The join key is
`forecast_key = "<platform>:<question id>"`, because post and question ids share one integer space and
Mantic's ids collide with both.

## Where each column comes from

Scores come from the round or pull record through `platform_scores.py`. The headline score is spot peer
on Metaculus (coverage-scaled peer only as a labelled fallback, `headline_score_field`) and spot
baseline on Mantic, and the app never pools the two platforms. `type_log_score` is the log score that
belongs to the question type. `prob_on_outcome` is the published probability on what happened (on a
numeric question only for an out-of-range resolution), and `confident_wrong` is true when that is at
most 0.15 on binary or multiple choice, or when a numeric resolution falls outside the published 5th to
95th percentile range. `crowd_size` is the post's `nr_forecasters`, null where a pre-2026-08-25 pull
stored 0 for "never read".

Telemetry joins honour each line's `qid_kind` through `QuestionIdMap.resolve_marker_record`, per
platform, and a run is a Mantic run exactly when its run id appears in `mantic_tournaments.jsonl`. When
a forecast has lines from several runs the latest run is used and the build logs a warning. Research
comes from `research_tags.load_verified_archive_record`, which rejects a file whose own ids or platform
contradict the forecast and never accepts a `log_backfill` record. Provider outcomes are filled only
from schema-2 `artifact` records; a `comment_backfill` record's provider list is reconstructed from
trimmed text and is left null.

Member rows come from the comment (`per_model_numeric_percentiles` on numeric and discrete, the base
models on binary and multiple choice) plus any member only the `member_forecast` marker saw, such as a
Mantic per-bin member. On a stacked record the comment's values are the stacker's, so those rows have
`role == "stacker"` and nothing on the record is rankable. `own_score` is the residual library's solo
replay (`member_replay.replay_solo_score`); a per-bin member's CDF is the running sum of its bins.
`rankable` also excludes anonymous keys and a numeric curve with too few anchors beside denser ones,
per `ranking_cohort.py`.

## How to read a null

A null always means "not recorded", never "no". Event markers (`extreme_call`, `ghost_forecast_json`)
are logged only when something happens, so their flag is true when the forecast's run logged the
marker, false only when that run is dated on or after the marker's first line anywhere in the archive,
and null otherwise. A forecast whose run predates the telemetry archive (2026-07-13) has null telemetry
columns; on the 2026-11-19 build that was 847 of 1,014 forecasts. In the app, a narrowed range filter
drops null rows and says how many, and every yes/no filter has a separate "Not recorded" choice.

## Rationale traits

Each member rationale gets counts and flags from `trace_review/traits.py`: percentages quoted, hedges
(also per 1,000 characters), base rates stated with a number, status-quo mentions, resolution-criteria
mentions, and flags for citing a market price, saying no market exists, citing the community forecast
and flagging resolution ambiguity. The patterns were checked against 32 labelled snippets of real
rationales (`tests/data/trace_review_rationale_snippets.jsonl`), which include the near misses that
broke simpler phrase matches: "No prediction market data found", a bare "Expert and market priors"
heading, "labor market" and "base rate" with no number. Known false positives remain and are rare: a
market name used as a litigant near a percentage, "claim as written", and a central bank's "base rate"
quoted as a percentage. The manifest records each trait's prevalence by config era, because prompt
changes move these shares.

## The app

The **Forecasts** page shows one line about the build, six presets (Worst misses, Confident & wrong,
Most disagreement, Ghost vs published, Since last round, Strict cohort; each says what it does on hover),
one summary block per platform, and a table of every filtered row. Each question in the table is a link
that opens **Forecast detail** in a new tab with the current filters (Streamlit opens table links in a new
tab and offers no same-tab option). The detail page has a header with each score named by platform and
kind, previous and next buttons that walk the filtered order, a chart of every member beside the
published forecast and the resolution (on numeric and discrete questions a density or per-value chart,
with the cumulative curve one click away; a member keeps its model family's colour across forecasts),
and tabs for rationales (side by side, or one model at a time at full width), research, gap-fill v2,
telemetry and the published comment. Rationales and the comment are model-written markdown, so the app
escapes `$` and `~` before rendering them (Streamlit would otherwise read dollar amounts as LaTeX and
"~350" as strikethrough), and each rationale's traits show as "label: value" badges. The **Models**
page shows, per platform, model and question type, the mean "score alone" (each member's own
forecast scored as if published) over rankable rows, and for numeric the same mean without scores at
the censoring floor, the mean difference from the published score, how often the model was the
farthest member, and the trait means. Every filter is bound to a query parameter, so a pasted URL
reproduces the view, and all three pages render the same sidebar so the filters survive a page switch.
None of the means is era-bucketed unless an era filter is set, and the captions say so.

## Code

`metaculus_bot/trace_review/`: `sources.py` (round discovery, pulls and cache, telemetry index,
exclusion counts), `build.py` (assembly, tables, manifest, staged write), `forecast_rows.py` and
`member_rows.py` (one function per column group), `traits.py`, `schema.py`, `config.py`, then the app:
`bundle.py` (cached readers), `filters.py` (one `FilterSpec` row per filter), `plots.py`, `views.py`,
`app.py` and three one-line page files in `app_pages/` (not `pages/`: with a folder of that name
Streamlit runs a page file directly when its URL is opened, skipping `app.py` and its navigation). The
live pipeline may not import the package (an import-linter contract). Tests are in
`tests/trace_review/`: a fixture workspace built through the real collector from
`tests/data/performance_comments_mini.jsonl`, the builder end to end, filters, plots, the markdown
escaping, and Streamlit `AppTest` smoke tests of every page.
