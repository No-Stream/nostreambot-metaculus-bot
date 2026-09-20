# Residual analysis playbook (standing methodology)

Codified 2026-08-24 from the practice that evolved across `scratch/residual_2026-05-29/` →
`06-15` → `06-26` → `07-08` → `07-18` → `08-02`. This supersedes
`residual_rerun_workflow.js` (the 2026-06 workflow script, kept for history) as the
methodology reference AGENTS.md points at. Everything here is free and offline: Metaculus
API reads, GitHub artifact reads, local files. Zero paid provider calls, zero publishing,
no commits from analysis agents.

Read this playbook and the current `docs/performance_analysis.md` before using old
`scratch/residual_*` material. The dated directories hold datasets, logs and reports; routine
pulls, round assembly and standard dimensions use committed CLIs and modules. Do not create a
scratch driver, copy a prior-round driver, or reimplement a standard dimension. A focused
follow-up investigation may use a scratch script. If routine functionality is missing, record
the gap and agree on a maintained addition first.

Outputs land in a fresh dated dir `scratch/residual_<date>/` (gitignored — grep prior
rounds with `rg --no-ignore`). Never modify a prior round's dir; it is the diff baseline.

## Phase 0 — Pre-pull (always)

`make sync_all` before anything reads the archives. GHA artifacts expire at 90 days; a
stale archive silently drops recent questions and receipts. Non-negotiable first step.

## Phase 1 — Recon

1. **Era map.** `git log --first-parent --format='%H %cI %s' main` since the last round's
   boundary. For each merge, diff `metaculus_bot/ .github/workflows/` and classify:
   forecast-distribution-shifting (new sub-era) vs neutral. **Era boundaries are
   merge-to-main committer timestamps, never authoring dates** — this has manufactured
   phantom eras twice (AGENTS.md era-bucketing rule). Verify the roster actually stayed
   frozen (`git log -p -- metaculus_bot/llm_configs.py`).
2. **Follow-up ledger.** Read the prior round's SYNTHESIS (watch-item ledger + recommended
   next steps) and FUTURE.md. For each item: shipped since (commit evidence)? date/n-gate
   due now? still open? The ledger tells the dimension and dossier agents what to check.

## Phase 2 — Pull and assemble

- For Metaculus, use the committed performance-analysis CLI once per active slug. Its default
  tournament can lag the live season, so pass the slug explicitly and pass the matching prior
  pull to detect platform re-resolutions:

  ```bash
  uv run python -m metaculus_bot.performance_analysis \
    --tournament <slug> \
    --output scratch/residual_<date>/perf_<slug>.json \
    --prior scratch/residual_<prior>/perf_<slug>.json
  ```

- Before using a new slug, run `make probe_slugs`. `make verify_pull` is an offline audit, but it
  requires a same-pull raw-post checkpoint beside the records file, named
  `perf_<slug>_checkpoint.json`; the performance-analysis CLI does not write that checkpoint.
  Its `collector.fetch_resolved_questions` API returns the raw posts, but capturing them from a
  separate call gives a separate snapshot. If exact checkpoint capture is needed for routine
  verification, report that gap and agree on maintained CLI support rather than writing a
  scratch capture script.

- The `RoundSpec` builder and output writer described in `docs/performance_analysis.md` consume
  staged per-slug `perf_<slug>.json` files, the prior tagged dataset, `question_weights.json`,
  `platform_rescored.json`, and the telemetry archive. The checkpoint is for `verify_pull`, not
  the builder. Invoke the maintained API directly; there is no integrated multi-source round
  CLI or round-owned driver script.

- Reuse verified-complete baselines from prior rounds (each round's README records which pulls
  are safe to reuse and why).
- Era-tag every record on `bot_comment_created_at` (submission time) against the era map.
  Tag exclusion cohorts by IMPORTING the constants, never by retyping the ids:
  `KNOWN_BUG_QIDS`, `DEGRADED_RUN_QIDS` (dry-key 1-of-3), `PARTIAL_DEGRADED_QIDS` (2-of-3)
  from `metaculus_bot.performance_analysis` — the same sets the `known_bug` /
  `degraded_run` / `partial_degraded` `--exclude-qids` shorthands expand to. Three rounds
  hardcoded private copies of the degraded ids before the constants existed, and the
  known-bug copies have drifted from the canonical set at least once. Excluded from
  headline aggregates, reported separately — never silently dropped.
- Diff vs prior round (`new_since_prior.json`): the new cohort is what the round is about.
- Diff the prior round's records at the VALUE level, not just by presence — pass
  `--prior <prior dataset>` to the `performance_analysis` CLI, or call
  `diff_platform_rescores(prior_records, new_records)`
  (`metaculus_bot.performance_analysis.rescore_diff`). Metaculus re-resolves in place with no
  timestamp moving: it edited q44798 from 80 to 82 and flipped that record's spot peer from
  +5.41 to −5.42 while `resolution_set_time` still read a stamp preceding the pull that saw
  80, so the 2026-08-31 round's tables for it went stale silently. Anything the diff tags
  `platform_rescored` must be re-read before it is quoted, and the prior round's document
  corrected. Read the tag as a ternary: None is "no prior record, never compared", only
  False is "compared, nothing moved".
- Spot-check re-pull stability (a handful of prior spot-peer scores must reproduce).

## Phase 3 — Automated dimensions (era-bucketed, parallel)

The standing set, each reconciling explicitly with the prior round's same-named dim doc
(agree / disagree / refine + why):

- numeric width + PIT (width_monitor eras, cov80/cov50/cov@10, PIT std, band-miss lo/hi)
- binary + MC calibration (Beta-Binomial CIs per bucket, slope/intercept by era)
- per-model (through `per_model_cohort`; **ex-floor means + declared-band miss rates** —
  the −219.97 log-score floor censoring once flipped a worst-member conclusion)
- aggregation faithfulness + stacker state confirmation
- bot health (extraction rungs, gap-fill v2 telemetry, credit, close-margin latency,
  provider diagnostics)
- market snapshot informativeness
- ghosts + guards (score_ghosts, structured-JSON presence)
- cross-tournament / category
- consensus-miss mode statistics (see Phase 4 rules)
- clip-threshold sweep (`performance_analysis.clip_threshold`; STRICT run `--exclude-qids
  known_bug,degraded_run,partial_degraded` plus one unfiltered): each candidate binary/MC floor in spot
  peer over `all` / `last_N` / `last_90d` / `current_clamp_regime` / `triple_era` and the disjoint `era_*`
  slices; a floor fitted on older records must carry into the window (a fit that moves nothing is
  vacuous, not a pass); a candidate looser than the in-force clamp is CENSORED, read `cen` / `cen_m` bounds
- era scoreboard and era gap (`performance_analysis.era_gap`; `--strict` plus one unfiltered, treated
  era = the live roster, comparison era = the one it replaced; sub-era arms via `--era-field
  triple_subera_fine`, the coarse `config_era` being the default): per-arm spot mean / median / frac neg with
  effective n by resolution day, the type-adjusted gap AND its horizon-matched form (comparison arm capped
  at the treated arm's longest submit-to-resolve lag, plus the lag-quintile-adjusted form; the per-arm lag
  quartile table is the receipt), each with a cluster-bootstrap interval and the by-record bracket; clusters
  are event days by default, so pass `--clusters <round>/cluster_structure.json` once the round's curated
  strong clusters exist and quote that interval as primary (the report says when the brackets disagree on
  zero). The standing two-sided watch reads the STRICT type-adjusted horizon-matched row: a concern reopens only
  below -5 with the interval excluding zero; a favourable gap is reported, never flagged. Never headline the
  type-adjusted gap without the horizon-matched one beside it (2026-09-09: +10.71 became +8.19), and never
  reopen an underperformance flag on a positive gap. `docs/performance_analysis.md` "The era gap and the
  horizon confound".

## Phase 4 — Per-question tracing (the centerpiece; often the highest-value output)

**Mandatory every round, resourced at least as generously as the automated dims.** The
operator's standing directive (2026-08-24): human-style question-by-question tracing is
frequently the most valuable part of a residual round — treat it as a first-class phase,
not an optional garnish.

1. **Rank.** Rolling-window miss ranking (SPOT peer score, all types) → `MISS_RANKING.md`.
   `audit.select_cohort` already ranks on spot and logs `PLATFORM_RANKING_SOURCE`; a WARN
   there means some record fell back to coverage-scaled peer.
2. **Select.** All material new misses, plus **good-call controls** (3–5). Controls are
   load-bearing, not decorative: the 2026-08-02 publish-vs-own-anchor metric only died
   because a hit-side baseline exposed it as outcome-tracking (63% miss vs 33% hit — and
   the worse-than-own-LR version inverted). Any miss-side process metric quoted without
   its hit-side rate is presumptively survivorship bias.
3. **Trace.** One dossier agent per question, full pipeline walk: research bundle
   (`research_archive/latest/` + the RAW per-provider payloads in
   `research_archive/raw/`), per-model forecasts and rationales, aggregation path,
   telemetry markers (`marker_records_for_question` via
   `performance_analysis.id_mapping` — post-id vs question-id spaces differ per marker
   and hand-rolled joins produce false matches), resolution mechanics, and score
   counterfactuals. Classify: research-miss / consensus-judgment-miss / weighting failure
   (corrective was computed or in the bundle and published past) / aggregation loss /
   pipeline bug / defensible-loss. Name the cheapest change that would have saved it.
   **A dossier's per-model ranking table can legitimately be EMPTY or short, with the
   reason printed beneath it — read the caveat lines rather than treating a missing table
   as a parse failure.** `ranking_cohort.per_model_ranking_cohort` drops the whole record
   when the stacker fired (every per-model slot then holds the stacker's aggregate), drops
   anonymous `Forecaster N` keys (positional buckets, not models — `Forecaster 1` was the
   third most frequent "best model" in a previous synthesis tally), and on numeric drops
   declared curves under 9 distinct anchors unless every member on the record is equally
   sparse. A row labelled sparse-era is comparable WITHIN its question and not across
   questions: don't quote its absolute log score beside a denser question's. Any round
   script that tallies `ranked[0]["model"]` / `ranked[-1]["model"]` inherits these
   exclusions automatically, so read a shrinking tally as the fix landing rather than as
   lost data.
4. **Adversarially verify every miss dossier.** In 2026-08-02, 6 of 6 verified dossiers
   came back REVISED — classifications held but load-bearing numbers and headline
   counterfactuals were corrected or refuted. The verifier re-derives key numbers from
   primary sources, checks counterfactual dates (was the proposed source even published
   before the run?), and recomputes score math. Unverified dossiers are quotable only as
   "plausible but unaudited".
5. **Cross-dossier statistics.** Consensus vs dissent (a material dissenter must be
   right-sided; magnitude above the ensemble noise floor), failure-class shares, and the
   dossier↔breadth-statistics reconciliation. Declared-percentile PIT is censored at the
   outermost label — unanimous censoring is itself maximal consensus signal.

## Phase 5 — Synthesis

Headline first (the era scoreboard and whatever the round's load-bearing question was),
then: failure-mode verdict, watch-item ledger with per-item verdicts
(fired / clear / still-unmeasurable + what changes), bot-health verdict, recommended next
steps **split free-offline vs paid-operator-decides** (nothing paid runs during a residual
round — surface command + rough cost), curiosities. Caveat n honestly; state effective n
after clustering (same-day resolutions share a world state).

## Standing gotchas (each has burned a round at least once)

- **The tournament ranks on SPOT PEER; `peer_score` is `spot_peer_score × coverage`.**
  Never rank, aggregate, or headline on the coverage-scaled figure — the bot submits once
  and never revises, so its coverage is mostly submission timing, and the scaling flatters
  misses (q44872: peer −15.0 vs spot peer −38.8). Read platform scores through
  `performance_analysis/platform_scores.py`, which prefers spot and keeps peer-only records
  in their own sort tier; report peer beside spot as a labelled secondary only.
- **Price a counterfactual with `spot_peer_delta`, never by hand.** Metaculus halves the peer
  score of a continuous question (numeric / discrete / date) and does not halve binary or
  multiple choice, so moving only our own mass on the resolving outcome is worth
  `100·ln(new/old)`, halved for continuous. Both conversions have already gone wrong in round
  scripts, and both mistakes inflate the figure: `numeric_log_score` already carries the
  halving (it returns `50·ln`), so doubling its difference to "convert to peer" over-prices
  by 2× (the 2026-08-31 q45065 replay: +404 where the truth is +202), while
  `binary_log_score` is log base 2, so quoting its difference as peer points over-prices by
  1/ln2 ≈ 1.44 (thirteen 2026-09-01 dossier scripts). Correcting one of those archived binary
  figures means multiplying it by ln 2 ≈ 0.693, not dividing. Import `spot_peer_delta` from
  `metaculus_bot.performance_analysis.scoring`.
- Merge dates, not authoring dates, for every era boundary.
- **Horizon before verdict.** A season's final wave is its long tail, so the retired roster's arm
  inherits long-horizon questions the live roster was never asked and its score falls with lag
  (2026-09-09 quartiles +18.33, +10.96, +9.57, +4.94). Type adjustment cannot see that; the era-gap
  module's horizon-matched row can. Lag is `actual_resolve_time` minus `bot_comment_created_at`, never
  `resolution_set_time`.
- `performance_analysis.id_mapping` for any marker↔record join; never "match either id".
- Never pool research-archive record classes (`artifact` / `comment_backfill` /
  `log_backfill`) for presence, provider-mix, or length claims; comment-backfill re-heads
  sections (`##`→`###`) and trimming eats leading sections — the trim-immune instrument is
  the provider-diagnostics block.
- `per_model_cohort` is a scoring filter; using it for aggregate faithfulness once
  manufactured 43 phantom drifters.
- Numeric PIT on `zero_point` questions needs the geometric value grid
  (`build_cdf_value_grid`), not linear interpolation.
- Numeric per-model means must be quoted ex-floor alongside raw.
- Miss-side process metrics require hit-side controls.
- Small-n honesty: quote cluster structure and effective n; two catastrophes can carry a
  cohort mean.
- The Metaculus comments API only returns our own comments; competitor analysis needs
  other channels.
