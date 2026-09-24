# Design log

Intent and dead ends, not current state, which lives in `docs/`. Cited `scratch/...` receipts are the operator's local
artifacts and gitignored on purpose, so a citation attributes a claim rather than pointing at a file a reader can open,
and planning docs named here may have moved out of the public repo. Neither is a broken link to fix.

## Open, operator decides

- **Gap-fill v1 for the fall**: leaned (about $1.75 a question, over the $1.50 ceiling) or off (about $1.37), against a
  roughly $1 target. Its gate discharged 2026-09-09 MIXED over 99 both-on bundles: forecaster lens v1 59 / v2 32,
  resolution lens 28 / 24, the passes complementary, v1 building reference classes while v2 reads the resolving
  instrument. A grade-based lean shipped; a positional cap would drop the useful gap.
- **Repo-wide comment-extraction sweep as its own CR**: 5,238 smell-hook findings over 436 files, four files done.
  Recipe that worked: one agent per file group, comments and docstrings only, an AST-equality check against HEAD with
  docstrings stripped, scanner at zero per file, prose to the topical doc, then `make all`.
- **Widen the market staleness tier cap** past `same_quantity_same_date`? Two lines in `_tier_cap_note`, held back
  because a resolved market on an adjacent cut is legitimately informative. The first `MARKET_TIER_CAPPED` line in a
  prod log is the finding, and a continued zero is an answer too.
- **Revisit the binary clip floor's 0.01-to-0.02 half?** `clip_threshold.py` priced both floors: tightening loses in
  every window and era, loosening is censored, and the clamp has bound no binary publish since 2026-05-18.
- **The 45 s resolution-source wall discards pages that already succeeded**, and the process-wide host gate amplifies
  it. Operator SKIPPED: remedies are a partial harvest, a budget-bounded acquire, or instrumentation only, and merging
  Tier-1's host map with gap-fill v2's waits on one. Timing surface, so nothing lands casually.
- **AskNews DeepNews** as an optional heavy `search_news_deep` tool: blocked on checking its limits and pricing.
- **LOW: migrate Gemini grounded search to Google's Interactions API** (`client.aio.interactions.create`, with
  `url_citation` annotations on text), and consider the same migration for gap-fill v2 `read_document` and the
  resolution-source `url_context` rung. The 2026-09-22 probe annotated 7/8 Interactions calls, all of which resolved,
  while one call with four searches dropped annotations; the same-day `generate_content` control annotated 5/6, so
  Interactions is not demonstrably better yet. Google staff named Interactions as the likely fix (forum thread
  174074). It would remove roughly 3k output tokens of self-citation redirect URLs per call. Revisit if Google
  confirms a fix or deprecates `generate_content`; receipt: `scratch/gemini_grounding_2026-09-22/README.md`.

## Open, priced levers not yet built

- **HIGH PRIORITY: gap-fill v2 driver on gpt-6-luna at xhigh/max instead of gpt-6-sol at low.** The driver sends a
  top-level `reasoning_effort`, which litellm 1.98's OpenRouter transform rewrites `max` -> `xhigh`, so the arm below
  ran at xhigh; testing true `max` needs the `reasoning={"effort": ...}` form. One-question replay on 2026-09-22 (Q44229, `scratch/model_migration_2026-09-22/`): both finished inside the 540 s loop (luna 146 s, sol
  85 s, 31-32 tool calls), and a blind Opus judge called it a tie leaning luna, low confidence. Luna read 882k prompt
  tokens against sol's 711k at a twentieth of the input price, so the driver's spend drops by roughly an order of
  magnitude. Needs a multi-question replay before switching; Sol stays for the season start.
  Same-day native-search probe: luna at true `max` gave the brief a blind judge preferred, but took 278 s against
  sol-low's 30 s; luna at `high` or `xhigh` there is the untested middle.
- **Tail-consistency check on the numeric block**: when a rationale derives a sigma then declares a tighter left tail,
  widen it deterministically. +11.93 baseline points on the q44453 cohort, from arithmetic the models already did.
- **Gap-fill v2 office-holder precedent rule**: on "will X assume office", retrieve how the current holder got the seat.
  About +31 spot peer on q44210, roughly 80% of that question's recoverable loss, from one query.
- **Consensus-fragility audit**: MEDIAN never asks WHY the members agree, and the cheapest form is a
  `load_bearing_claims` prompt field, since the deterministic and auditor-LLM forms tax the common cheap path.
  Best-supported competitor lever, on weakened evidence: dissent-toward-truth 9% on misses against 21% on hits.
- **Anchor-date discipline**: make a member state the date of the anchor it used. Weakened but not retired now that
  rendered values carry dates; do not quote q44553's +58 as an expected value.
- **High versus xhigh reasoning effort** on the forecaster slots, both really xhigh only since 2026-09-22 (gpt-6-sol
  high -> xhigh; the Anthropic slot's declared xhigh had been overridden to high by `verbosity: "high"` until then, see
  docs/roster_history.md): paired A/B, $60 to $90.

## Open, recorded and not built

**Mantic, awaiting live data.** Fast-path alertability in mantic mode; median versus mean for PERCENTILE members under a
bin log score, needing about thirty resolved Mantic percentile questions; the free post-651 reads after 2026-09-20 that
settle the day-bin edge convention (noon mapping kept meanwhile) and the scoring coefficient; benchmarking the 5%
out-of-range tail floor on this bot's OWN archived forecasts once `oor_low` / `oor_high` accumulate; the first log-scaled
discrete grid, of which the corpus has none; namespacing the archive grouping key before the Mantic post counter nears
Metaculus id 14333, about 13,700 posts away, and not before.

**Structure, each its own PR, and re-measure before acting.** Splitting `resolution_source.py` is blocked by 22
monkeypatch names, since a patch left on the old module stays green while proving nothing; `forecaster.py`'s remaining
seam is the soft-deadline machinery and `prompts.py`'s is the shared continuous template both prompts import. Also open:
a public `metaculus_get` plus the public-then-private comment paging three call sites duplicate, the fetch primitives
`agentic/tools.py` reaches into `resolution_source.py` for, and `basedpyright` strict on the core pipeline.

**Research reach, priced and not started.** Provider-degradation alerting past the prediction-market provider, keeping
every denominator INSIDE one question because prod runs carry one or two, so a per-question flag must never fire; Tier-2
for the js_wall and blocked slice; the trafilatura extraction that drops MediaWiki collapsible boxes and can leave a
"Declined to endorse" list reading as the inverse of the truth; a `yoy_pct` TS derivation and the level siblings
`mom_pct` / `mom_diff` deliberately lack, since a wrong-quantity band is worse than none; screenshot-plus-vision, OCR
and a social-media reader, against 5 empty renders and zero image fetches in the archive; and the egress-IP half of the
403 problem, whose only fix is an egress that is not a GitHub runner, called too complicated on 2026-09-03.

**Watch items, no action while they hold.** The post-Option-B over-sharpening trigger has NOT fired at n=5 of about 15
and both halves fail in the too-NARROW direction, so `k_tail` = 1.0 holds. `width_monitor.py` still pools treated with
untreated because it reads none of `research_tags.py`'s fields. The starved-outer-tail detector shipped but the
publish-time WARN did not, since reading `declared_percentiles` at publish time would silently never fire on the
discrete cohort it exists for; 68 of 417 measurable open-bound sides fire, so read a fire as "this question carries a
cliff". MC top-band under-commitment replicates in all four eras and needs 57 questions for exact p<0.05 against 35.

## Rejected: do not re-propose without new evidence

**Calibration and post-hoc layers**, all dead. Calibration as a lever at all (operator, 2026-07-16: the slope FLIPS by
era, so any fit spans opposite-signed eras and power is about 0.25 even at N=400). With it: post-hoc isotonic
(2026-05-10, a small-N artifact), Platt, directional shrink, YES-side shrink (2026-07-08, era-local, and a two-era fit
degraded held-out fall; never touch the NO side), shrinkage toward 50% (2026-05-10, costs calibrated extremes), the
signed deadzone haircut toward 0.5 (2026-07-08, 0 of 24 cells help in both well-powered eras, though the toward-0.5
direction constraint survives as a design rule), fixed-direction and one-sided anti-overprediction shaves (the binary
slope flips sign across rounds). Also killed: "more NO right than YES right" as miscalibration (2026-07-16, a base-rate
artifact), the fall [0.70, 0.90) under-confidence signature (retired 2026-08-24, the false positive multiplicity
predicts), the MC [0-5%) bucket as a pricing gap (measured NULL 2026-08-24; about 454 MC questions would be needed
against 86, so the 0.01 floor stays), and the 3D calibration grid (2026-05, about 0.7 questions per cell).

**Guards, clamps and screens.** Anchor-guard as a clamp or gate (2026-07-08: models leave their own anchor on about 88%
of forecasts, which is the normal outside-to-inside update, and gate precision is at most 29%). The anchor-overshoot
self-consistency screen (measured and REJECTED 2026-08-31, re-priced 2026-09-02: blocks that LEFT their anchor scored
better, not worse; never propose it as a screen). The anchor-floor guard on cheap tails and the no-market-no-extremize
cap (2026-07-08, era sign-flips and top-5 concentration; their revival conditions can no longer accrue since
`base_rate_anchor` left the prompt on 2026-09-02, which also closes the telemetry-first guard revival program).
Ceiling-only clip tightening (one pre-flip record, and it is the hard-clip form of the 2026-07-08 kill). An
`excluded_bins` list on the percentile block, the alternative to per-bin PMF elicitation (2026-09-08: eligibility lives
only in the criteria prose, so nothing can know which bins to exclude).

**Aggregation and ensemble.** Geometric-mean-of-odds base-combine (RUN 2026-07-16, decisive null in every era: keep
MEDIAN). Mean over median, always-on stacking, the stacker as judge, and coherence weighting (2026-07-15): benchmarked
and rejected, with stacking prod-disabled on every type since 2026-05-29; per-bin Mantic members pooling by MEAN is
decided by construction, not a revisit. JS-divergence diversity as a roster-selection signal (BUILT 2026-07-16 and
rejected: under MEDIAN decorrelation is INVERSELY related to marginal contribution, since independence can be
incompetence and MEDIAN discards outliers rather than harvesting them; kept only for near-clone detection). "Median
drowns the correct dissenter" (2026-07-08, survivorship bias). Mixture-model parameterization for numerics (REMOVED
2026-07-08, percentiles plus PCHIP beat it in every benchmark; do not cite "zero prod fires" as the reason, which was
wrong). Parametric mean/sigma representation, blanket sigma-widening, and the open-tail spike grid-compliance trick
(2026-06-26, all conflict with verified data). Dropping gemini-3.1-pro on the binary axis (did not replicate, 2026-05).
Per-model peer ranking as an action (in-sample, multiplicity-exposed, epoch-confounded, and it INVERTS on numeric).
Domain-aware CDF narrowing at k about 1.3 (HOLD: the measurement predates the 2026-05-18 `k_tail` flip so it would
overcorrect, and its "exclude finance" advice is inverted on current data). Parked rather than killed, revive only on new
evidence: the spread-triggered second forecast round (LOW since 2026-08-25 because stacking is prod-disabled, NOT
because the gate is dead, since it fires on 15 of 30 triple questions), always-on crux extraction, the TS-anchor chart
image A/B, claude-fable-5 for the Anthropic slots (pulled 2026-07-20 for `content=None` refusals, a reliability rather
than a quality problem), and per-type weighting by historical performance. Trimmed mean (RUN 2026-09-15 on the five-
and six-member eras, n=233 numeric-family: +1.6 [+0.1, +3.3] log/question pooled, all of it pre-flip, post-flip −0.01;
degenerate at three members, so a null for the live roster). Pointwise mean at k≥5 is the one revivable item: the
2026-09-15 replay at n=262 found +2.0 [−0.0, +4.3] pre-flip (six members, half the sum in two questions), −0.15
post-flip and −0.43 on the triple, so revisit only if the roster grows back to five or more. The same run replicated
the quantile-averaging loss at −7.4 [−13.4, −2.4], a tail effect, and attributed the median's edge over its average
member to location consensus (+6.3 [+4.7, +7.9]) with the width effect a null (−0.5 [−2.3, +1.4]); the vendor-diversity
delta at k=3 is a null on every type. Receipts: `scratch/aggregation_bench_2026-09-15/` in the artifacts repo.

**Research architecture.** End-to-end per-forecaster agentic research: NO (2026-07-16, re-confirmed 2026-07-19: BTF-2's
most accurate forecast was a strong prompt on good SHARED research, already this architecture, so the lever is
shared-research quality rather than integration topology). Metaculus `similar-posts` as a provider (2026-07-11: the
community-prediction value is null for our account and it returns only OPEN questions). The Metaculus-wide
resolved-sibling base-rate lookup (2026-07-11: the API nulls resolution, criteria and CP on every post this bot did not
itself forecast, leaving a `forecaster_id` self-history lookup as the only readable form, LOW). The market-deference
time-to-close term (measured structurally dead: no match closes near the question, because we forecast near open).
Firecrawl and Olostep (rejected for the DIY fetch ladder, superseded by gap-fill v2). The `venue_no_contribution`
provider-health rule (DELETED 2026-08-04, unsound under ranked retrieval and it fired on 45 healthy runs; only a
cross-run form over the telemetry archive is worth building). Aggregator feeds (Metaforecast shut down, Adjacent News
redundant with Kalshi plus Polymarket, PredictIt politics-only, PMXT only if venue breadth binds). Gemini grounding
through OpenRouter (NOT supported as of 2026-05-17, since Gemini falls back to Exa and would swap grounded retrieval for
text search; a dated reading, re-verify before quoting). The Kalshi "fall back to `volume_24h`" one-liner (rejected
2026-08-03 and pinned as rejected in `tests/test_prediction_market_liquidity_contract.py`: that field is 0 on
long-horizon markets, so scoring off it labels a deep market thin). Probabilistic tooling for base forecasters (built,
prod-DISABLED, settled dead path).

**Process and analysis.** Time-of-tournament Brier rolling averages (roster-swap confounded), `nr_forecasters` difficulty
stratification (peer score already normalizes for difficulty) and a per-model MC audit (per-model MC does not survive in
stored comments): all killed 2026-05. "Community Brier minus our Brier" as edge over market (an affine shift of Brier on
a balanced panel, so zero edge). Prompt edits on N=1 observations, the private-preview leading indicator and the
Klimt-sale contradiction clause. Advisory, non-binding critic passes (2026-07-08: a competitor's critique diagnosed our
exact defect and the number never moved, so the per-forecaster critic survives in binding-output form only).
`check_cup_comments.py` as a promoted script (NOT promoted 2026-09-12: the plain author listing is blind to private
comments and the cup is exactly the tournament whose comments stay private, so its loudest output is the wrong call;
`make supply_probe` answers the question off `my_forecasts`).

**Mantic, accepted with no change (do not re-raise).** Loosening the binary 0.98 clamp; the multiple-choice 0.01 floor;
withdrawing a forecast; the early-close `min()`; publish thread-pool sizing; clock and timezone handling; upcoming,
group, conditional and notebook posts; 400/404/405/429 publish handling; resolution-time bin aggregation; the 70-minute
step cap; the high-cardinality multiple-choice ceiling; Mantic comment backfill. Five 2026-09 readiness findings were
declined on the merits: retrying the pre-spend tournament GET, a `MEMBER_FORECAST` line for a member whose CDF build
raises (percentile and per-bin both), the supply probe's preflight as a frozen-dataclass `partial`, and letting the
public snapshot override an empty `my_forecasts`. Each remedy was a broad except, a second marker call site, or a
`partial` that breaks the patch-where-used contract.

## Study-only, not planned

A frozen point-in-time retrieval corpus (RetroSearch-style; the reproducible variant is date-gated CommonCrawl News plus
LanceDB, and frozen-artifact replay through `--research-dir` is the near-term substitute); longitudinal trajectory
scoring; a ForecastBench submission, a useful leak-free scoreboard but not free to forecast; necessary-condition and
scenario decomposition, highest ceiling of the experiment bucket on the weakest evidence, against an operator prior that
prompting tricks mostly do not work; separate outside and inside view stages; an LLM self-evaluation pass;
embedding-nearest-neighbour market matching; deterministic coherence projection, algebra that rarely binds since these
questions are mostly standalone; a stacker hint naming reliable dissenters, blocked on records the prod-disabled stacker
cannot produce.
