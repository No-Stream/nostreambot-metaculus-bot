# Loss-attribution report: build plan (2026-11-25)

Branch: `loss-attribution` (cut from main after PR #87 merged; first commit 9def5aa records the
`fall_four_member` era boundary, unpushed). Status: planned, not started. Build in a fresh session.

## Why

Our routine checks are calibration tables (does p match frequency per bucket). The operator wants the
complementary error analysis: where the bot loses points, and whether it loses them by being
miscalibrated or by failing to tell outcomes apart. The trigger is multiple choice: MC scored -4.0 spot
peer on 9 fall questions (13 now) while scoring +15.5 over 123 questions in all eras, a free refresh on
2026-11-25 (`scratch/residual_2026-11-25_floors/`) found no floor or calibration cause, and the fall MC
losses sit where the top option is at 0.35-0.65 (4 of 9 top picks right at 0.5-0.85, about -11 each).
Separately, the gap-fill driver's single-model "ghost" forecast beats the published MC forecast
(`make score_ghosts`: +4.05 mean LOG-score delta on 28 MC questions; binary -9.9 on 67; numeric +0.8 on
55), and the agentic loop moved the driver toward the truth on MC 14 of 18 times (+10.7). This report is a
maintained tool so these questions get the same answer every round instead of a new scratch script.

## Scope (agreed core, then options)

Core, agreed with the operator:
1. **Points-lost accounting.** Bin each resolved binary and MC question by the probability the bot gave
   the REALIZED outcome; per bin report n, sum and mean spot peer, and share of total negative spot peer.
   This is accounting (low p on the outcome is usually, though not always, negative, since spot peer is
   relative to the other forecasters), so it is always shown next to
   an outcome-blind table: the same quantities binned by the bot's top-option probability (MC) or
   max(p, 1-p) (binary). Label the two tables "where the points went" and "by our confidence".
   Numeric/discrete: bin by PIT (`width_monitor.compute_pit_reading`): [0, .025), [.025, .1), [.1, .9),
   [.9, .975), [.975, 1], plus beyond-bound resolutions as their own rows (open or closed side). This is the
   "placement in the distribution" view.
2. **Reliability / resolution / uncertainty split.** CORP decomposition (Dimitriadis, Gneiting & Jordan
   2021, PNAS: isotonic recalibration via PAV, giving MCB miscalibration, DSC discrimination, UNC
   uncertainty, with score = MCB - DSC + UNC) for the Brier score and the log score. Binary: one
   observation per question. MC: one observation per question using the TOP option (p_top, did it win),
   because options within a ballot are dependent; report the option-level pooled version only as a
   secondary line, labelled as such. Also emit the binned reliability table (absorbing
   `scratch/residual_2026-11-25_floors/floor_calibration.py`, which then retires).
3. **MC discrimination.** For the published forecast, each member model, and the ghost: top-1 accuracy,
   mean rank of the winning option normalised by option count, mean p on the winner, and spot peer if
   that forecaster's vector had been published (`spot_peer_delta` against the published vector's p on the
   realized option). Plus a symmetric two-way count: questions where some member ranked the winner top
   and the median did not, AND questions where the median ranked it top and that member did not (both
   directions on ALL questions, never misses only; see decision-11 below).

Options (build if cheap, else leave as backlog follow-ups):
4. **Disagreement-conditioned ghost comparison.** Bin questions by OUTCOME-BLIND distance between the
   ghost and the published forecast (binary |p - q|; MC total-variation distance), and per bin report the
   mean spot-peer change from publishing the ghost, with resolution-day cluster intervals. This is the
   clean test of the ghost lead. Applies to binary, MC, numeric (numeric via the ghost's declared
   percentiles through `member_replay`-style CDF building; skip numeric if it needs new CDF code).
5. **Slice ranking.** Mean spot peer by slice (type, source_tournament, option count, anchor_present,
   gfv2_loop_ran, was_stacked, forecasters_used) with cluster intervals, sorted, explicitly descriptive
   (no multiplicity-corrected claims), plus the worst-decile question list with URLs for trace review.

Out of scope: a market-price reference. No same-quantity market price is stored on round records or in
the market telemetry markers (only retrieval counts); the 2026-11-04 market dimension recovered exact-date
prices for 4 questions only. Recorded as a follow-up, not built.

## Prior decisions this touches (read .backlog/decisions/)

- **decision-31** killed a per-model MC audit in 2026-05 because "per-model MC does not survive in stored
  comments". New evidence: the current round dataset has complete per-member MC vectors on 130 of 130
  resolved MC records (`per_model_forecasts`, parsed by `parsing.parse_per_model_mc_option_probs`). Cite
  this in the PR; it is the revival condition.
- **decision-11** ("median drowns the correct dissenter", rejected for survivorship bias): section 3's
  member-vs-median count is symmetric and on every question, which is the fix for that bias.
- **decision-15** (per-model peer ranking as an action): per-member rows are diagnostic only; the report
  says so in its header and does not rank members as a recommendation.
- **decision-2** (MC 0-5% bucket null) and the clip sweep are not re-run here.
- Any change this report motivates (e.g. weighting the ghost on MC) still needs the out-of-sample era
  test from docs/performance_analysis.md before shipping.

## Data and helpers (verified by a read-only Codex pass, 2026-11-25)

Dataset: `scratch/residual_2026-11-25_floors/perf_all_tagged.json` (1,033 records; binary 568, MC 130,
numeric 242, discrete 93), built with the committed `RoundSpec` / `build_round_dataset` API
(`round_dataset.py`). Rebuild per round; the tool takes `--cached <path>` like clip_threshold.

Record fields: `type`; binary `our_prob_yes`; MC `options` + aligned `our_forecast_values`;
numeric/discrete CDF `our_forecast_values` with `scaling`, `open_lower_bound`, `open_upper_bound`;
outcome `resolution_parsed` (bool / option string / value); spot peer via
`platform_scores.spot_peer_score(record)` (never peer_score); eras `config_era`, `triple_subera`,
`triple_subera_fine`; `platform`; exclusion tags `known_bug`, `degraded_run`, `partial_degraded`
(cohorts are QUESTION ids, `cohorts.py`); `research_source_class` (artifact / comment_backfill / null;
do not pool classes where a section depends on research); resolve time
`metadata.actual_resolve_time` (cluster key = its UTC date, as in `era_gap._scored`).

Members: `per_model_forecasts` (binary as "68.0%" strings, parse with `_parse_probability`; MC as
option -> prob dicts aligned by option name). Filter with `analysis.per_model_cohort` (drops confirmed
stacker records and anonymous "Forecaster N" keys) and complete ballots only. A comment with fewer
member bullets than `forecasters_used` is ambiguous: never read a missing member as a decline. Bucket
member rows by era, since the roster changed (see eras.py).

Ghosts: loading and joining live in `scripts/score_ghosts.py` (`join_and_score`, marker loading via
`scripts.telemetry.archive.load_marker_records` from `backtests/telemetry_archive`). Import-linter
forbids `metaculus_bot -> scripts` (pyproject.toml, "metaculus_bot does not import operator tooling").
Decision for the build: move the ghost marker parsing and the join (ghost `qid` is the record's
`post_id`, not `question_id`) into `metaculus_bot/performance_analysis/ghosts.py`, and make
`scripts/score_ghosts.py` import from it, keeping its output byte-identical (pin with a test on a
fixture). If the telemetry archive loader itself lives in scripts/, pass the loaded marker records in
from the CLI layer instead of importing it. Run `make lint_imports`.

Units: price every alternative (member, ghost) in SPOT PEER with
`scoring_common.spot_peer_delta(old_prob=, new_prob=, question_type=)` on the realized outcome, as the
clip sweep does (`clip_threshold_sweep.py` 283-325 aligns member MC dicts to `options`). Do not mix with
score_ghosts' log-score deltas; when quoting them side by side, label units.

Intervals: resolution-day cluster bootstrap via `era_gap.ClusterSums.from_values(values, clusters)` and
`.resampled_means(rng, draws)` (fixed seed); Jeffreys via `analysis.jeffreys_ci(k, n)`. Minimum-n gate:
print a cell's interval only when n >= 10 and clusters >= 5; otherwise show n and the point value with
"(n too small)". Era-bucket every table: `all` plus `--era-field` (default `triple_subera`), with the
fall config (`triple_fall_config`) always shown. State sample sizes in the report header (today: 130 MC
all eras, 13 in the fall config, 28 MC with a ghost).

## Shape

- `metaculus_bot/performance_analysis/loss_attribution.py`: pure computation, one function per section,
  each taking records and returning dataclasses.
- `metaculus_bot/performance_analysis/loss_attribution_report.py`: markdown + JSON rendering.
- `metaculus_bot/performance_analysis/ghosts.py`: as above.
- CLI: `uv run python -m metaculus_bot.performance_analysis.loss_attribution --cached <dataset>
  --exclude-qids known_bug,degraded_run,partial_degraded [--era-field triple_subera] [--telemetry-archive
  backtests/telemetry_archive] --out <dir>`, writing `loss_attribution.md` and `.json`. Mirror
  clip_threshold's CLI conventions.
- Docs: a "Loss attribution" section in docs/performance_analysis.md (what each table answers, which
  table is accounting vs diagnostic, the min-n gate, the decisions above); add the dimension to the
  standard list in scratch_docs_and_planning/residual_analysis_playbook.md.
- Free: reads local files only. No paid calls.

## Tests (TDD; tests/test_loss_attribution*.py)

- CORP on synthetic data: a perfectly calibrated forecaster gives MCB ~ 0; a constant forecaster gives
  DSC = 0; MCB - DSC + UNC equals the mean score exactly (Brier and log).
- Points-lost bins sum to the total spot peer of included records; the outcome-blind table uses no outcome.
- MC winner rank on hand-built ballots, including ties (average rank) and options missing from a member
  dict (record excluded from that member's row, counted).
- Symmetric dissenter counts on a hand-built 4-member case.
- Ghost join keys on post_id; score_ghosts' report on a fixture is byte-identical before and after the move.
- PIT bins including beyond-bound resolutions on open and closed sides.
- Exclusion cohorts and era bucketing applied; min-n gate renders "(n too small)".
- CLI smoke on a small committed fixture dataset; a real-data smoke only if the scratch dataset exists
  (skip otherwise: CI has no scratch data).

## Acceptance

`make test`, `make lint`, `make typecheck` (0 errors), `make lint_imports` green; CI green on the branch.
Run the tool on `scratch/residual_2026-11-25_floors/perf_all_tagged.json` and put the headline findings
in the PR description, with sample sizes and intervals, and what (if anything) they suggest testing next.
Suggested routing: Terra (medium) for loss_attribution.py + tests, Luna (xhigh) for the ghosts.py move
and docs; Opus or the main session reviews the report's wording.

## Reminder for the operator

After this, discuss cross-pollination of forecasts (predict, cross-pollinate, update, with prompt
caching): backlog task-45 "Short update round with partial cross-pollination". The operator calls it high
priority; the task is currently marked Medium.
