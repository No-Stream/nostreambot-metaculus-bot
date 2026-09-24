# Forecaster prompts

Every rule the forecasting prompts carry, why it is there, and what it cost to put
it there. The prompts themselves live in `metaculus_bot/prompts.py`; this doc is the
narrative behind them, so read it before adding, removing or rewording a rule. The
standing policy in one paragraph: a rule stays only if the pipeline requires it, it
scaffolds the model's reasoning, or it corrects a measured failure with no shorter
form. Each surviving rule is stated once, with its one-clause reason attached, as a
named module constant or a named template step. Instructions that shape the
structure of the answer live in the numbered template, not in a trailing checklist.

Last verified against the code on 2026-09-09.

## Where the prompts live

- `_benchmarking_warning`, `_forecasting_window_str`, `web_research_prompt`.
- Base: `binary_prompt`, `multiple_choice_prompt`, `numeric_prompt`, `date_prompt`. Each base prompt embeds the STRUCTURED FORECAST JSON-block schema instruction and requires the fenced ```json block to be the **last** output, with no trailing prose value lines. The value extraction ladder relies on this: rung 1 parses the block deterministically, and the tail-scan repair rung reads only the rationale's trailing `_TAIL_SCAN_CHARS`.
- The numeric, date and per-bin prompts are ONE template, `_continuous_prompt`, parameterized along two axes by two frozen dataclasses. `_ContinuousAxis` (chosen by `_kind_axis` for all three prompts: `_numeric_axis` for a `NumericQuestion`, `_date_axis` for a `DateQuestion` viewed through `numeric.date_axis`, so the kind half of the template is resolved in one place) carries the four slots that differ between a QUANTITY and a DATE whatever the elicitation: the status-quo question, the reference-class rules, the tail scenarios and the forecastability bullet. `_Elicitation` (built by `_percentile_elicitation` for the thirteen-percentile ask and `_pmf_elicitation` for the per-bin ask, section "Per-bin elicitation" below) carries every line that names the elicited object: the preamble's calibration nouns, the market clause's anchor tail, the axis block (units and bounds, dates and bounds, or bins and bounds, with the bound messages rendered as dates rather than epoch floats and the bin-granularity sentence), the scoring paragraph, the multi-resolution rule, the Mantic out-of-range sentence, the step-3 timeline verb, the step-7 small-delta check and anchor-adherence clause, the two step-8 width and tails bullets, the `outcome_type` step (present only on a percentile quantity), the final-check lead, the consistency line and the STRUCTURED FORECAST example (ISO strings and `question_type: "date"` on a date question, `question_type: "pmf"` per bin). The percentile fills are today's text verbatim: `TestPercentileFillIsTodaysText` in `tests/prompts/test_pmf_prompt.py` pins the exact rendered line of every slot, line breaks included, because the Metaculus render of the numeric and date prompts is byte-identical to what it was before the split (proven by rendering all seven Metaculus prompts at the pre-split commit and on the tree and diffing) and may change only with the operator's say. The date parse note allows repeated dates but not for every percentile (the 0.01 and 0.99 values must differ), because on the 201-point date grid (post 500's shape) the sanitizer withholds a fully collapsed set as a unit mismatch, while on a coarse outcome-space grid the collapse is spread within its bin and publishes, and teaches the working form for a one-day concentration: increasing UTC timestamps inside that day, with the example day taken from the question's own displayed range. The template reads one handle, the `NumericQuestion` (a `DateQuestion` arrives as its epoch view, which carries `page_url`, `api_json` and every prose field verbatim). Every shared rule is stated once in the template, so a numeric rule cannot drift from its date twin. `date_prompt(question, research, lower_bound_message, upper_bound_message)` is the date entry point; it also carries `_SOFT_CLOCK_RULE`, because an announced target date in a "when will X" question is the announced-but-unbound shape that rule was measured on. `_HISTORY_DISCHARGED_RULE` was deliberately not added to it, pending the operator's say.
- Stacking: `stacking_binary_prompt`, `stacking_multiple_choice_prompt`, `stacking_numeric_prompt`. Same block-last schema instruction as the base prompts (the stacker output flows through the same ladder). The stacker prompts also include a "Cross-model aggregation (deterministic math)" block at the top when `build_cross_model_aggregation` returns markdown.
- Conditional-stacking support: `disagreement_crux_prompt`, `targeted_search_prompt`.
- Gap-fill: `gap_fill_analyzer_prompt`, `gap_fill_search_prompt`.

## The 2026-09 de-bloat, and the size accounting

The 2026-09 bundle first added nine prompt rules and then, on 2026-09-02, cut the three base prompts by 22-23% (audit: `scratch/prompt_bloat_audit_2026-09-02.md`; git receipts for every removal: `scratch/prompt_debloat_2026-09-02/receipts.md`; the operator's item-by-item decisions: `scratch_docs_and_planning/announced_unscheduled_fix_plan_2026-09-02.md` section 7), then added the two Phase 1 rules that plan's Items A and C call for (`_SOFT_CLOCK_RULE` and `_HISTORY_DISCHARGED_RULE`, below), which are what the whole plan is named for. All of it shifts the forecast distribution, which is why the whole bundle lands in ONE merge (era-bucketing, below). The standing rule for the prompts is now: a rule stays only if the pipeline requires it, it scaffolds the model's reasoning, or it corrects a measured failure with no shorter form; each surviving rule is stated ONCE, with its one-clause reason attached, as a named module constant or a named template step; instructions that shape the structure of the answer live in the numbered template, not in a trailing checklist. Sizes, re-measured 2026-09-02 against the true pre-bundle baseline (method: render each prompt on one fixture question whose research string carries `MARKET_SNAPSHOT_SECTION_HEADER`, so the market clause is present on both sides of the comparison, then collapse all whitespace and count characters; the baseline column loads `git show 7e7d449:metaculus_bot/prompts.py` as a module beside the current one, which is exact because that file's only intra-repo imports are `numeric/config.py`, `numeric/utils.py` and `time_utils.py` and none of the three has changed since). Before the cut, at 7e7d449: binary 19,396, MC 14,184, numeric 18,983, gap-fill analyzer 4,693. Straight after the cut, at 2f13d0a: 15,043 / 10,885 / 14,805 / 4,168, so the cut itself took 22-23% off each base prompt and 11% off the analyzer. With the two Item A and C rules added back on top and the analyzer trimmed to its target: binary 16,095, MC 11,895, numeric 14,805 (it carries neither new rule), analyzer 4,023, a net 17.0% / 16.1% / 22.0% / 14.3% off the baseline. The two new constants are 648 and 366 collapsed chars. **The sizes sit 6-12% above the plan's section 7 targets (binary about 14,300, MC about 10,700, numeric about 13,700, analyzer about 3,800) against its "within about 5% is fine" criterion, and that overshoot is a recorded WAIVER rather than an oversight**: binary +12.6%, MC +11.2%, numeric +8.1%, analyzer +5.9%. Three things account for it, none of them trimmable without giving something up the operator asked for. The section 7.2 KEEP decisions (the 5b "three valid moves" paragraph, the 0b worked-examples justification, the twice-stated pre-open rule) are operator calls, not slack. Principle 3 requires every surviving rule to carry its one-clause reason, which is length spent on purpose. And the plan's own arithmetic was light: Item A's wording measures 648 collapsed characters against the 450 its section 4 estimated, which is most of the binary and MC gap by itself. Closing the remaining gap means cutting prose, not rules, and the next change that widens these prompts should do exactly that. An earlier version of this paragraph quoted 18,640 / 13,820 / 18,446 as the "before" figures; those came from an intermediate post-Item-D commit rather than 7e7d449 and understated the baseline by roughly the Item D schema-field text.

## Platform-aware and Mantic clauses (2026-09-08)

The Mantic Phase 2 merge (`scratch_docs_and_planning/mantic_phase2_plan_2026-09-08.md`, receipts in `mantic_research_2026-09-08/`) added the first platform-aware rules. The platform is read once, off the question's `page_url` host (`question_platform.question_platform`, a leaf module), so the Metaculus prompts changed only where a Metaculus-specific claim was false everywhere. The platform-gated clauses (the out-of-range base rates and the scoring-grid clause, which keys on grid fields only Mantic payloads carry) plus the rules gated on the payload's `multi_resolution` flag appear in no stacking prompt, by the standing rule; the one consequence worth naming is that a stacker on a multi-resolution question never learns the multi-resolution rule, which matters only if numeric stacking is re-enabled. Two of the clauses below ARE shared with the stacking prompts, and deliberately: `_scoring_sentence` and `_CONTINUOUS_SCORING_RULE` replaced wording the stacking prompts already carried verbatim, so the stacker reads the same scoring facts as the forecasters it combines.

- **`_scoring_sentence(question)`**, rendered in all six prompts (binary, MC, numeric/date and the three stacking prompts), picks `_METACULUS_SCORING_SENTENCE` or `_MANTIC_SCORING_SENTENCE`. The Metaculus sentence names the spot peer log score, which the bot tournaments use (the Metaculus Cup is coverage-scaled peer scoring, so on the cup it is one word off, recorded in FUTURE.md). The Mantic sentence names Crucible's spot baseline log score, compared to a uniform distribution rather than to other forecasters, so nothing is gained by disagreeing with the obvious answer and nothing is lost by giving it. "Your Metaculus question is:" became "Your question is:" at the same time.
- **`_CONTINUOUS_SCORING_RULE`** replaces the Metaculus-specific scoring paragraph in the continuous template. The deleted text claimed a uniform 0.01 PDF floor on every density and that sharpness above about 35 stops paying; both are Metaculus facts, and on Mantic they told the model the out-of-range cliff was an order of magnitude shallower than it is. What survives is the proper-scoring sentence, true everywhere, plus one sentence that mass beyond an open bound is scored as its own outcome against a reference of a few percent, so starving it is heavily punished (Mantic scores that bucket against 5%: a 1% tail is -80.5 baseline points, verified on 146 of 146 resolved out-of-range questions).
- **`_MANTIC_OUT_OF_RANGE_RATE_QUANTITY` / `_MANTIC_OUT_OF_RANGE_RATE_DATE`**, appended after the bound messages on Mantic only. Series 1, annulled excluded: 51 of 274 quantitative questions (18.6%) resolved outside the displayed range, 35 of 141 discrete (24.8%) and 16 of 133 numeric (12.0%), and 101 of 188 date questions with an open upper bound (53.7%) resolved above it, against 2.2 to 2.6% in the Metaculus archive, so keeping every percentile inside asserts a 1% chance of an out-of-range outcome. The numeric count includes seven escapes the platform stored as raw values outside the range rather than as a bound token (posts 512, 460, 426, 396, 305 linear; 387, 200 log-scaled), which a filter on the tokens alone misses. The mechanical 5% tail floor that enforces the same lesson on the published Mantic aggregate was built 2026-09-08 (`MANTIC_OUT_OF_RANGE_TAIL_FLOOR`, docs/numeric_pipeline.md Step 11); the prompt clause still matters because it moves the members' own declarations, and the floor's `oor_low_raw` / `oor_high_raw` fields record what the members did without it.
- **`_MULTI_RESOLUTION_CONTINUOUS_RULE` / `_MULTI_RESOLUTION_MC_RULE` / `_MULTI_RESOLUTION_BINARY_RULE`**, interpolated by `_multi_resolution_clause(question, rule)` only when the question's own API JSON declares `multi_resolution` `is True` (an identity test, because a MagicMock stub is truthy). One distribution is scored against N resolution values and averaged, so the optimal submission is the mixture: continuous and date questions forecast each resolution instance and report the mixture's percentiles, MC the expected share of resolutions per option, binary the expected fraction of Yes. N is never interpolated; while a question is open it lives only in the criteria prose. Priced on the live post 650 at 38.7 points for a single-day distribution.
- **`_scoring_grid_clause(question)`**: when the platform declares the grid (`precision` on a quantity, `date_granularity` on a date) it names the bin count, taken from the typed `cdf_size - 1` so the prompt can only describe the grid the pipeline submits, says a percentile's value selects the bin it falls in, and that mass finer than a bin is wasted. On a linear grid the count comes with the bin width; on a log-scaled grid (`zero_point` set) Mantic defines `precision` as the ratio between adjacent boundaries, so the clause says the grid is log-spaced and renders no width. Paired with the numeric-side plateau cap (`numeric/config.grid_bin_width`) that stops a concentrated forecast on a 3 to 21-bin grid publishing flattened.
- **Series-variant reconcile bullet**, rewritten: the displayed range is weak evidence about WHICH series variant resolves and NO evidence about the magnitude of the outcome. The old premise, that the bounds were set by someone who could see the real series, is false on Mantic, where question writers are paid for bot disagreement; the correction it carried (do not read inside-the-range as confirming the headline series) survives in a form true on both platforms.
- **Stacking MC preamble** now says the probabilities must "sum to 1.0", matching the decimal schema the parser reads (the base MC prompt lost its integer-percent statements in the 2026-09 de-bloat, below).

Sizes were not re-measured for this merge; the clauses above are additive and platform-gated, so the Metaculus prompts grew by the scoring-sentence swap and the coarse-grid and multi-resolution clauses only where their gates fire. The next size pass should re-run the 2026-09-02 method below against `7e7d449` and record the numbers here.

### Per-bin elicitation (Wave C, 2026-09-09)

On a Mantic grid of `PMF_ELICITATION_MAX_BINS` bins or fewer (`numeric.config.elicit_per_bin`; the plan is `scratch_docs_and_planning/mantic_wave_c_plan_2026-09-08.md`, section 2.8) the bins are the outcome space, and thirteen percentile anchors cannot say "zero on this bin": on question 651 PCHIP spread a quarter of the mass onto the three weekend days of a twelve-trading-day window, -14.4 baseline points. `pmf_prompt(question, research, lower_bound_message, upper_bound_message)` therefore asks for one probability per named bin. It is the continuous template with the per-bin `_Elicitation`, so every shared rule (status quo, provenance ladder, null-result reading, count-in-period class, the soft-clock rule on a date grid, forecastability, anchor on your math, bait-and-switch) reaches it through the same constants, and the question-kind slots (`_ContinuousAxis`) are untouched. It takes the `DateQuestion` or its epoch view interchangeably, like `date_prompt`; the runner decides WHEN to elicit per bin, the prompt only renders the ask. The bin list is the question's own labelled grid (`numeric.pmf_grid.PmfGrid.keys`, section "Bin labels" of the plan), never the bound message: on a centre-style grid the last label need not equal the displayed maximum (post 650 declares `nominal_max` one step above its last bin centre). The bound messages arrive already worded for the reserved keys from `numeric.utils.pmf_bound_messages`. Nothing in the per-bin prompt says "percentile" (pinned, with and without a market snapshot in the research), it carries no `outcome_type`, and its final checks are step (9).

The per-bin rules, each a named constant with its reason attached and a presence pin in `test_pmf_prompt.py`:

- **`_PER_BIN_SCORING_RULE`** (the Scoring Rule section, in place of `_CONTINUOUS_SCORING_RULE`): the question is scored on the bin the outcome falls in (or the `below_range` / `above_range` key beyond an open bound, scored as its own outcome against a reference of a few percent), it is a proper scoring rule, and probability on a bin the resolution criteria exclude (a weekend on a trading-day question, a count the rules rule out) is simply lost, so give such a bin 0. Reason: the excluded-bin loss is the loss this elicitation exists to remove, and the model has to know it.
- **`_PER_BIN_OUTPUT_RULE`** (the last Bins & Bounds bullet, a format template): one probability for EVERY key listed in the schema, spelled exactly and in order, summing to 1.0; 0 for a bin that cannot occur; a bin left at 0 is lifted to the platform's per-bin minimum (interpolated as `grid_step_constraints(cdf_size)[0]`, about 0.00048 on a 21-bin grid) downstream and the rest scaled down to keep the total at 1.0, which is what `numeric.pmf_cdf._blend_to_cell_floors` does. Reason: the block rung maps keys onto the grid and fails on a missing key (truncation must fall through, not publish a partial declaration); the floor is named per grid so it cannot be read as the 5% aggregate tail floor, which the member is deliberately not told about; and the mechanism is stated as it is (until 2026-09-09 the sentence said the floor "is added to every bin", which contradicts the sum-to-1.0 clause beside it and invites the model to pre-subtract).
- **`_PMF_KEY_RULES`** (Bins & Bounds, one sentence per `BinLabelStyle`, the dict total so every per-bin question renders exactly one key sentence, after the grid clause that names the width it refers to): a centre-labelled key is the value at the centre of its bin and the bin covers half the stated width either side of that value (unconditional, because `_is_center_aligned` admits any step: 5 of the 46 coarse Mantic centre grids step by 5, 500, 0.25 or 0.1, where a key `5` read as "5 up to 10" puts a belief of 3 one bin off); a day or week key "names the whole day it covers" / "the seven days beginning on it"; and where the labels are intervals (a geometric grid, nominal bounds on the range edges as on post 560, or a date grid with edges at arbitrary times) **`_PMF_INTERVAL_KEY_RULE`**: each key `a to b` is right-closed and the first bin also owns its lower edge. The grid clause before it (`_pmf_grid_clause`) names the bin count and width from the grid itself, not from the platform's `precision` flag, so it renders on every per-bin question.
- **`_PMF_CONSISTENCY_LINE`** (final checks): which bin holds the status quo or trend value, how much probability did you give it, and is that sensible. Reason: the percentile line asks which percentile is the status quo.
- **`_PMF_QUANTITY_FINAL_CHECK` / `_PMF_DATE_FINAL_CHECK`** (the final-check lead): a misread unit puts probability on the wrong bins; does every bin you gave probability to fall on a day the criteria allow. Reason: the percentile leads talk about the values the model outputs, and per bin it outputs none.
- **`_MANTIC_OUT_OF_RANGE_RATE_DATE_PMF` / `_MANTIC_OUT_OF_RANGE_RATE_QUANTITY_PMF`**: the same measured Series 1 facts as the percentile twins above, ending in the per-bin action (most of your probability belongs on `above_range`; give each open bound's key your honest probability, a token probability there asserts a near-zero chance of an out-of-range outcome) instead of a percentile placement. Rendered through the same gate, `_mantic_out_of_range_clause(question, date_rate=..., quantity_rate=...)`, which each elicitation calls with its own pair: Mantic only, the date sentence only when the upper bound is open, the quantity sentence when either bound is open. Genuinely different instructions from their twins, which is why they are constants and not slot fills.
- **`_MULTI_RESOLUTION_PMF_RULE`**: `_MULTI_RESOLUTION_CONTINUOUS_RULE` and this one are the two fills of `_MULTI_RESOLUTION_CONTINUOUS_TEMPLATE`, whose slot is the report clause ("report the mixture's percentiles" / "report that mixture as your per-bin probabilities"), pinned to differ in nothing else.

Two step-8 bullets are ONE format constant each with the elicited object as the slot, per the state-once rule: **`_WIDTH_BULLET`** (percentile fill "your interval width" / "a narrow interval" / "a wide one" / "a wide interval", today's text; per-bin fill "the spread of your probability across the bins" / "a concentrated forecast" / "a spread-out one" / "a spread-out forecast") and **`_UNKNOWN_UNKNOWNS_BULLET`** (percentile fill "Keep your extreme tails (P1 and P99) wide enough"; per-bin fill "Keep enough probability on the outer bins (and on `below_range` / `above_range` where they exist)"). The other percentile-naming lines are plain `_Elicitation` fields rather than constants: the preamble's calibration nouns ("the width of your prediction interval" / "that width" against "how far your probability spreads across the bins" / "that spread"; the sentence itself stays in the template because it spans two template lines and the Metaculus render keeps those line breaks), the market clause's anchor tail ("your percentiles should center on it" / "your probability should center on it"), step 3's "shift percentiles" / "move probability between bins", step 7's small-delta check ("+/- 10 percent on key percentiles" / "moving ten points of probability to a neighbouring bin") and anchor adherence ("your percentiles should stay close to it" / "your probability should stay concentrated around it"). The plan's section 2.8 proposed rewording three of these shared lines neutrally on both platforms; that was NOT done, because it would have changed the Metaculus render, which is the operator's call. The per-bin STRUCTURED FORECAST example is `_pmf_schema_block(grid)`: the real keys in grid order with the even-split example probabilities `_build_example_probs` gives the MC prompt, four pairs per line, and it must parse through `parse_structured_payload(body, "pmf")` (`PmfStructured`) with keys equal to `grid.keys`. No per-bin constant appears in the three stacking prompts or in the percentile numeric and date prompts (pinned).

## What the base prompts carry, and where

- **Market clause** (`_strong_evidence_market_clause`, all three base prompts) renders ONLY when the research carries `MARKET_SNAPSHOT_SECTION_HEADER` (`## Prediction Market Snapshot`, defined in `prompts.py` and imported by `research/section_format.py` the way `TS_ANCHOR_SECTION_HEADER` already was), so a backtest, a flag-off run or a soft-failed provider no longer hands the forecaster 1.5k chars about a table it does not have. Prod-neutral: the provider emits the header whenever it rendered anything, including the deliberate-empty "no sufficiently relevant market" sentence. One consumer needs the header handed to it explicitly: the gap-fill v2 driver's template skeleton (`research/agentic/driver_prompt.py`) renders the base prompts with a placeholder research string, so that placeholder now carries `MARKET_SNAPSHOT_SECTION_HEADER` and the driver dry-runs against the same template the forecasters read. The rendered table's legend (`MARKET_SIGNAL_LEGEND`, `research/market_retrieval/rendering.py`) owns NOTATION (liquidity labels and `no-liquidity-data`, evidential order, the four relation tiers, RESOLVED, `↳` sub-rows, `[remaining N]`); the prompt owns POLICY through `_MARKET_READING_RULES`: extrapolate from an other-cut market rather than discount it vaguely; liquidity governs a relation-vs-liquidity conflict because a thin price is noisy however tight its relation; a multi-outcome ladder is a DISTRIBUTION and never an equality constraint on a tail (the last two both q45189). That constant replaced the 1,908-char `_MARKET_RELATION_WEIGHTING_SENTENCE`, which re-taught the legend's notation, and `_MARKET_LIQUIDITY_WEIGHTING_SENTENCE` was removed as a verbatim duplicate of the legend.
- **`_SOURCE_PROVENANCE_LADDER`** (all three) says the briefing's claims arrive tagged by source tier, names the tag shape (`[A: ...]` to `[D: ...]`; the FULL tier definitions live once, in the research-side `_SOURCE_TIER_TAG_INSTRUCTION`, and the ladder carries a one-clause gloss per tier, so a tier re-cut has to move both or the forecaster reads the old boundary), keeps the two usage clauses the tag instruction does not state (use a C-tier report's cited facts, not its framing; D-tier is suggestive only), the motivation, primary-record-override and implausibility bullets, and a two-line definition of `[unverified attribution]` (the tag named no outlet, or the pipeline could not match the one it named; the claim may still be correct; treat it as untiered, not low-tier). The four A-D definitions it used to restate IN FULL were cut back to that gloss because every artifact record since the tagging landed in prod (99 of 1,069 archive records) already carries the tags.
- **`_NULL_RESULT_READING`** (all three, after the Strong/Moderate/Weak rubric): a null search result licenses only "we could not find evidence of X", and the weight of an absence scales with how well-indexed the source is. Receipt q44799, where the gap-fill resolver reported "I found no authoritative public record" and four of six forecasters converted that into "the authorization is absent"; the two that discounted it scored best. Its third bullet (absence is weaker still where the actor already demonstrated the behavior) was removed: it carried no receipt of its own and pushed the wrong way on q43837. `gap_fill_analyzer_prompt` carries a separately-worded auditor version ("NULL RESULTS ARE SEARCH OUTCOMES"). The stacking prompts are excluded, test-enforced.
- **`_COUNT_IN_PERIOD_REFERENCE_CLASS`** (all three, reference-class step), verbatim as shipped: for a question asking how many events of a kind occur in a period, the admissible outside view is the pooled realized rate over the longest comparable history, and a schedule of currently known candidates updates that rate rather than replacing it. Receipt q44561, where all six members built a "no failure announced yet, so Poisson(1.0)" schedule model instead of the pooled FDIC bank-failure rate.
- **`_REMAINING_EXPOSURE_SENTENCE`** (binary and MC, outside-view step): rates apply to the exposure that REMAINS, applied from now until the deadline with the elapsed event-free part of the window treated as observed, because a rate spread over the whole window prices time that has already passed. In binary it opens the conditional-hazard bullet, which is the same rule specialised to recurring events; in MC it is one bullet on its own, since MC has no hazard bullet. Receipt q43837 (six members applied a monthly announcement rate over the full window with 16 days already elapsed event-free, then OR-ed it with a scheduled path the rate already contained). It replaced `_REMAINING_EXPOSURE_RULE`, whose first bullet restated the hazard bullet twenty lines below it and whose second restated the disjointness clause that lives in the binary union line ("union only over paths that cannot be the same event"), so one rule read three times. MC never computes a union and carries no disjointness text.
- **`_SOFT_CLOCK_RULE`** (binary and MC, the last bullets of the reference-class step, directly after `_COUNT_IN_PERIOD_REFERENCE_CLASS` and before the timeframe step): a target date the responsible actor has not bound itself to (no statute, no contract, no published schedule it has a measured record of meeting) is evidence that a target EXISTS, not that it will hold; price the probability that the target lands inside the question window as its own number, from that actor's record of slips and scrubs, and do not raise it on an announcement, plan, tracker page or partner page; where a binding clock exists, compute the date from it and say which clock. It carries its receipt in the text (forecasts averaged 44% on events that happened 8% of the time). This is the fix the 2026-09-02 plan is named for. The 2026-09-02 failure-mode audit (`scratch/failure_mode_audit_2026-09-02/AUDIT_SYNTHESIS.md`, lens A) found the shape on 52 of 815 STRICT records (6.4%; 8.3% of binaries; coder kappa 0.74): members decomposed P(target lands in window) times P(event given target) and set the first term near 1 because the target had been announced. On the 37 flagged binaries the bot published a mean 0.44 for events that happened 3 times (8%), 13 records above 0.5 resolved NO and none went the other way, and flagged records score 18.7 spot-peer points worse than unflagged ones (95% CI 5.9 to 33.4) and are wrong-sided 40% of the time against 18%. Soft targets WITHOUT the decomposition move score fine (+13.6) and deadline questions in general are calibrated, so the rule names the move rather than the question shape, and every vendor carried it (biased up 0.27 to 0.47 on the shape, within 0.06 of zero off it). Receipts are q43837 (a Fall tournament start read off the Summer close), q44424 (an announced summit that slipped twice) and q44557 (a "planned August" launch off a partner page and a Wikipedia infobox); the contrast is q45217, where a statutory clock existed, the members computed the date and scored +45. The "measured record of meeting" carve-out is load-bearing: on q42305 a weekly bulletin with a measured one-to-three-week publication lag was a binding clock in practice and a near-1 timing term was right. It SUPERSEDES the two 2026-09-02 rules Item B removed (`_REMAINING_EXPOSURE_RULE` and `_ANCHOR_CONSISTENCY_RULE`). There is deliberately no structured-block field for it (the schema audit rejected a `target_holds_probability` slot): the block is written after the forecast is fixed, so the number lives in the rationale.
- **`_HISTORY_DISCHARGED_RULE`** (binary and MC, directly after `_SOFT_CLOCK_RULE`; both are rules for reading a base rate you just computed): if your own analysis names a reason the historical cadence has been discharged (its driver was met, the deadline passed, the rule changed), that cadence is a bound on your estimate, not its center; state the post-change estimate and what it rests on. Receipt, from the same audit's lens C: 12.1% of coded rationales carried the pattern, flagged records score about 7 spot-peer points worse (95% CI 2.7 to 12.2), and the old cadence failed in 83% of fires and in 13 of 13 on the live triple (the rule text carries that count as its parenthetical, worded as the small audit it is rather than as a flat rate, because at the 17-18% held rate the older era bands show, 0 of 13 is an ordinary draw). Treat those numbers as UPPER BOUNDS: coder agreement was 0.59 and the label is partly hindsight-contaminated. It is conditional on the member's OWN written acknowledgment so it cannot fire where nothing changed, and it shipped only once `_ANCHOR_CONSISTENCY_RULE`'s "do not move off your number when history counsels caution" was gone, since the two pulled opposite ways. Shipped on the plan's recommendation (section 6) with the operator's final say pending, and the standing watch item at the top of FUTURE.md says so. To revert: delete the constant, its two interpolation sites (one in `binary_prompt`, one in `multiple_choice_prompt`), and only the history-discharged cases inside `TestSoftClockAndHistoryDischargedRules`. That class also covers the APPROVED `_SOFT_CLOCK_RULE`, and seven of its cases assert both rules jointly (placement order, the numeric and stacking scope guards, the no-block-field guard, mode-agnosticism), which is why the two rules do not get a class each.
- **One anchor-adherence rule**, the "Anchor on your math" bullet in the binary and MC final-rationale step, now with a size: a move of more than about 15 points off a computed number needs a named, specific piece of new evidence. On a multi-clause binary question step 5b states that the clause product is the number that check anchors to, because it is more specific than the step-2 base rate, and the three valid moves are the permitted ways to leave it. `_ANCHOR_CONSISTENCY_RULE` was removed: its first bullet was the fourth request to state the outside-view number, and its second ("do not move off your number on a general feeling that history counsels caution") priced at about zero on the whole archive and would suppress good toward-resolution moves. The size's receipt is q44557 (four of six wrote a 17-25% base rate and published 35-55% on soft schedule signals).
- **Binary and MC template, one statement per check.** The pre-open rule is stated TWICE on purpose, in the forecasting-window line and in the status-quo derivation's "name the specific POST-OPEN event", because this footgun has cost the bot badly; the third statement (0a's 447-char restatement with the 1945 example) was removed. The odds check and the small-delta check are one "Odds and delta check". The trailing "Brief checklist" is gone: four of its six items re-asked for outputs the template already produces, and the two with no template twin, the bait-and-switch check and the consistency line, are the template's final numbered step, "Final checks". MC lost all three of its integer-percent statements, which contradicted the decimal schema the parser reads: the two "sum to 100%" lines, and the every-option floor, which asked for "a probability (1-99%)" and "at least 1%" and now interpolates `MC_PROB_MIN` so it reads "at least 0.01" (the every-option requirement itself is extraction-critical and stays). The schema line's "sum to 1.0" stays. A percent-scale `option_probs` is hard-rejected by `_check_option_probs`, cannot be repaired because it is valid JSON, and MC has no strip-and-retry, so the whole ballot used to drop to the paid LLM salvage rung: receipt q44558, one of only two MC salvages in 87 archived extractions. The same edit landed on the stacking twin. The 5b "three valid moves" reconciliation paragraph and the 0b sentence explaining why the worked examples exist are KEPT by operator decision, each stating its reason (all hedging goes through the clauses so the criteria stay consumed as constraints).
- **Numeric template.** One anchor-to-latest statement (the step-0 status-quo derivation) and one trend statement (step-3 trend continuation) remain; the step-1 "centered near this value" push, the step-3 "status-quo outcome" line and the step-7 trajectory check were removed as restatements. Step 7 asks for the outside-view central estimate and range and what the evidence moved, replacing "My base rate was X% ... moving to Y%", a probability template on a distribution question. Step 8 is **"Forecastability and width"**: decide whether the quantity is forecastable from current information (an administered or slow-moving series) or close to a random walk (a traded price, a volatile count), and if near-unforecastable center on the current value with a width taken from the series' realized variability rather than expecting movement you cannot source; it absorbs the preamble's calibration paragraph (the even-handed wording the 2026-07 width audit settled on, no directional push toward wide or narrow) and the tails line, and it replaces the old step 9b, whose `FORECASTABILITY: HIGH/MEDIUM/LOW` output line nothing parsed. The preamble keeps a one-line pointer to it. Step 9 is a one-line pointer to the schema note that defines `outcome_type`. The step-8 lines that restated the schema notes and the bound messages (think in ranges, strictly increasing, no scientific notation, open vs closed) were removed; the Units & Bounds bullet and the interpolated bound messages are the one copy, and the units bullet, bait-and-switch check and percentile consistency line are the numeric "Final checks".
- **Gap-fill analyzer** (`gap_fill_analyzer_prompt`). "DO NOT invent gaps for completeness" stays with its reason (each gap is a paid search); the "0-2 real gaps; a few have 3-5" counts were removed because the analyzer fills every slot regardless (since 2026-07-17 it fills every slot on about half of questions and lists three or more gaps on 96%; an earlier "55-77%" figure here matched no cap-aware reading of the archive) and "3-5" was stale against `GAP_FILL_MAX_GAPS = 4`. ANSWERABLE NOW keeps its mandate (when a question resolves off a live data source, at least one gap asks what it reads NOW, because the current reading is the fact that most often decides these questions), the carve-out that a first pass already stating the reading WITH its as-of date counts as answered (re-ask only if undated or older than the source's update cadence), and the never-the-resolution-date rule, in 843 whitespace-collapsed chars instead of 1,164 (both measured on the block's own source lines); receipt is the Nebraska/Texas natural experiment (44554 miss vs 44556 control) plus 62 of 320 gap-fill-bearing records carrying a gap the resolver itself called unanswerable-yet. ORDER THE GAPS is two sentences (order is the ranking; no rank fields). `_LAST_REAL_USE_GAP_RULE` was removed and its content folded into gap type 6 as a candidate clause (how an institutional rule actually applied at its most recent real application, receipt q45215): a second "one gap MUST ask" mandate does not add spend, it displaces other gaps, and beside ANSWERABLE NOW it pre-committed half the slots. The gap-fill **v2 driver** (`research/agentic/driver_prompt.py`) keeps its matching present-tense bullet, untouched. `GRADE EVERY GAP` (2026-09-09) adds three grade fields to the schema and one clause defining them; it is described under "Research-side prompt rules" below.
- **Stacking prompts** are untouched by the cut except one wording slip: they said "Each base-model analysis above carries its final forecast" while the analyses are interpolated below that sentence, and now say "below".

## Constant-by-constant rationale (moved from the prompts.py comments, 2026-09-09)

The AST smell scanner's comment rules apply to `metaculus_bot/prompts.py` like every other module,
so on 2026-09-09 the receipt blocks that used to sit above each constant moved here. Each constant
in the file now carries at most one comment line, the single fact the code cannot say, and the rest
lives below under a heading spelled exactly as the constant or function, so a pointer of the form
`Receipt: docs/prompts.md "_SOFT_CLOCK_RULE"` is grep-findable. Entries appear in file order. No
prompt text changed in that pass: all 31 rendered prompts were compared byte for byte before and
after, and the module's abstract syntax tree with docstrings stripped is identical to the commit
before it.

One mechanical convention runs through the whole file and is the reason so many constants mention
indentation. Every forecaster prompt is one `clean_indents` f-string, and `clean_indents` dedents by
the smallest indent it sees, so an interpolated multi-line block has to be pre-indented past the
template's own baseline or it collapses the dedent for the entire prompt. The baselines differ:
the binary template sits at 12 spaces and the multiple-choice and continuous templates at 8, so a
block shared by all three is written at 15 spaces or more, which nests correctly under either. A
block interpolated INLINE into a bullet (rather than as its own line) must instead carry no leading
indent at all, and a block interpolated at column 0 into a single-block prose prompt would defeat
the dedent outright, which is why one policy sometimes exists in two renderings.

### `_PERCENTILE_LABEL_MIN_WIDTH`

The minimum width of a rendered percentile label is "0." plus two decimals, so P10 reads "0.10"
rather than "0.1". `_percentile_label` trims trailing zeros and then pads back to this width.

### `_STANDARD_PERCENTILES_DECIMAL_CSV`, `_LOWEST_PERCENTILE_LABEL`, `_HIGHEST_PERCENTILE_LABEL`

The canonical percentile set as the prompts enumerate it. All three are derived from
`numeric.config.STANDARD_PERCENTILES` so that a change to the set can never leave a prompt asking
forecasters for percentiles the pipeline rejects.

### `_EXAMPLE_PROB_DECIMALS`, `_EXAMPLE_PROB_FLOOR`, `_EXAMPLE_PROB_CEIL`, `_build_example_probs`

`_EXAMPLE_PROB_DECIMALS` is the number of decimal places used for the illustrative example
probabilities in `_option_probs_example`. The floor and ceiling are safety epsilons so no example
bucket lands at exactly 0.0 or 1.0, because the prompt tells the model to use values in the open
interval (0, 1). The clamp inside `_build_example_probs` is load-bearing at the extremes: for a very
large option count the even split rounds to 0.0, and for a single option it rounds to 1.0, so
without the clamp the example would contradict the instruction beside it.

### `_option_probs_example`

Both `multiple_choice_prompt` and `stacking_multiple_choice_prompt` need the same shape, the real
option names as JSON keys with illustrative decimal probabilities summing to about 1.0. A parser can
only bind an LLM's output to the allowed options when the schema example carries the exact option
strings; literal `Option_A` placeholders yield `<<NOT_FOUND>>` on strict parsers.

Two implementation facts sit behind the function's docstring. It uses `json.dumps` for both keys and
values so that an option name carrying a double quote, a backslash or a newline still produces a
syntactically valid JSON example; a naive f-string would emit invalid JSON and mislead the model
about the schema. And the caller wraps the returned fragment in an outer pair of braces, because the
template supplies them as `"option_probs": {{{example}}}`, so the function strips the braces that
`json.dumps` puts around its own object body before returning. It returns the empty string for an
empty options list, which lets the caller render an empty JSON object degenerately with no special
case.

### `_forecasting_window_str`

`MetaculusQuestion` types `open_time` and `scheduled_resolution_time` as `datetime | None`, but a
real API-fetched question always populates both. The two asserts are there to fail fast: a missing
timestamp means the upstream data is broken, and a loud error is what we want rather than a silent
fallback that corrupts forecasts.

Both sides are normalized to timezone-aware UTC before subtracting. forecasting-tools 0.2.92 makes
question datetimes timezone-aware, and a naive `datetime.now()` minus an aware value raises
`TypeError`. Passing UTC explicitly also fixes a latent skew from 0.2.54, where a naive local clock
was compared against naive UTC data; that was harmless only when the host itself ran UTC, as CI
does. The rendered dates are unchanged for the naive-UTC inputs the pipeline used to see.

### `_SOURCE_TIER_TAG_INSTRUCTION`

The source-tier vocabulary for the RESEARCH-side prompts, meaning the web-research prompt and the
AskNews summarizer. The full A to D tier definitions live here and only here. The forecaster
prompts' provenance ladder, `_SOURCE_PROVENANCE_LADDER`, names the tag shape, carries a one-clause
gloss per tier, and relies on the briefing arriving already tagged. Re-cutting a tier boundary means
editing both, otherwise the forecaster reads the old boundary while the research side tags by the
new one.

Without research-side tags a C-tier aggregator claim arrives in the briefing looking identical to a
B-tier wire fact, and the ladder has nothing left to weight. The instruction is deliberately short,
because research output is itself an input to further summarization, and it is written at zero
indent so the text survives `clean_indents` verbatim in every consumer, in contrast to the ladder's
pre-indent contract.

Merging this constant to main changed the live research-output format, so it was timed to ride the
gap-fill v2 config-era boundary (the july15 merge) rather than being merged or cherry-picked on its
own.

Since 2026-09-24 each tag must name the specific outlet or publisher (`"[A: BLS]"`,
`"[B: Reuters]"`, `"[D: Reddit]"`), a category is explicitly not a name, a D-tier tag names the
platform or account, and a claim is tagged only when the outlet can be named and the tier is clear.
The old examples were categories (`"[A: official]"`, `"[C: aggregator]"`) and taught category tags:
in the Q14333 smoke (run 36008672128) 10 of Gemini's 12 tags were class descriptions such as
`[A: peer-reviewed journal]`, and the 2026-09-22 probe's old-prompt control calls show the same
style (85% of tags), so it predates self-citation. A category cannot be checked against the
retrieval record, which let the model claim tier A while naming nothing; Gemini's output now has
such tags rewritten to `[unverified attribution]` (`research/gemini_attribution.py`). The rule is
shared, so GPT native search and the AskNews summarizer name outlets too, though only Gemini's tags
are checked. The A to D definitions did not change. The 2026-09-24 named-tag probe (5 Gemini
responses on Q14333 and Q45571, `scratch/attribution_named_tags_2026-09-24/`) found 101 named tags
and 0 class tags, against 85% class tags on the old prompt; the model often makes the tag the link
label, which the Gemini formatter normalizes (see `docs/research.md`). Pins: `TestSourceTierTagging` in
`tests/prompts/test_research_clauses.py` (named examples present, category examples absent).

### `OUTSIDE_VENUE_MARKET_ODDS_POLICY` and `_OUTSIDE_VENUE_MARKET_ODDS_BULLET`

The FOCUS AREAS market-odds bullet, narrowed away from the four venues the structured
prediction-market snapshot already covers live. Across 42 ranked-era bundles the old blanket bullet
("Prediction market odds and forecasts (if available)") produced exactly one content-redundant
retrieval plus three stale covered-venue prices that contradicted correct live snapshot rows, which
is the only measured harm mode, while every realized instance of decisive market evidence came from
OUTSIDE those four venues: Good Judgment Open on q44869, CME FedWatch on q45401, and the Metaculus
crowd on q20683. Hence narrowed rather than removed. The wording was confirmed verbatim by the
operator on 2026-09-01 and the receipts are in
`scratch/residual_2026-08-31/market_odds_coverage.md`.

The policy is split in two so that it has exactly ONE definition across prompts that format it
differently. `web_research_prompt` wants a FOCUS AREAS bullet, while the two Perplexity prompts are
unbulleted prose whose whole body is a single `clean_indents` block, where an interpolated line
starting at column 0 would defeat the dedent for the entire prompt. The private constant is the
policy plus its leading dash, so the operator-confirmed text stays byte-identical on the surface it
was confirmed against. Restating the policy per prompt is what let the two Perplexity call sites
keep the retired blanket "consider all relevant prediction markets" ask after this one was narrowed,
until a review caught them.

### `_SEARCH_LINK_CITATION_CLAUSE`

The citation instruction for the Gemini grounded-search provider. Google drops `groundingMetadata`
on a substantial share of Gemini 3.8 Flash calls, including calls where the model searched. The
2026-09-22 self-citation probe showed the model wrote Google's
`vertexaisearch.cloud.google.com/grounding-api-redirect/...` links in 10/10 calls, while grounding
metadata appeared in only 1/10. The formatter resolves those exact links itself, numbers distinct
resolved target URLs, and marks links that do not resolve as unverified. The same probe resolved
135/136 unique links with one no-follow GET (HTTP 302 plus `Location`), and no non-redirect links
were written. Receipt: `scratch/gemini_grounding_2026-09-22/README.md` and
`scratch/gemini_grounding_2026-09-22/selfcite_raw/selfcite_*.json`.

The wording requires the model to copy only tool-returned URLs in full and to keep the
`_SOURCE_TIER_TAG_INSTRUCTION` tags alongside each link. Numeric citation markers remain forbidden;
the formatter assigns its own `[N]` markers after link verification. The `markdown` branch remains
the native-search provider's model-authored citation style, so this Gemini-only clause does not
change it.

### `SUMMARIZER_SOFT_FAIL_BANNER`

Prepended to the AskNews section when the summarizer soft-fails, so the raw articles are never
mistaken for a screened analyst briefing. The AskNews audit made five properties of
`asknews_summarizer_prompt` load-bearing: a hard per-article relevance gate, recency-first ordering,
supersession arithmetic, an evidence-age opener, and proportional length. The raw path has NONE of
them, it leads with the Historical section, and it loses the `[PRE-WINDOW]` labeling that FUTURE.md
credits with saving multiple questions. Hence the banner's instruction to date facts and screen
articles by hand, which is the same vocabulary the summarizer prompt defines and is kept beside it
so the two cannot drift.

The banner is deliberately NOT a markdown heading. `_demote_inner_headings` in the orchestrator would
shift an h1 or h2, and the framework's section renormalization would then mangle the provider
header.

### `_SOURCE_PROVENANCE_LADDER`

The source-provenance and motivation trust ladder, shared verbatim across the three forecaster
prompts (binary, multiple choice, continuous). Reverse-engineering high-scoring competitor bots
showed that they rank factual claims by proximity to the primary record and adjust by source
motivation. It is interpolated in place of the old "Separate facts from opinions" bullet, which now
leads the block, so the swap is clean and simply appends the ladder.

The A to D tier DEFINITIONS are stated once, in the research-side `_SOURCE_TIER_TAG_INSTRUCTION`,
and the briefing arrives carrying the tags (every artifact record since the tagging landed in prod).
The ladder names the tag shape, keeps the two usage clauses the tag instruction does not carry, and
glosses each tier in one clause. It used to restate all four definitions in full, which re-taught
the model a vocabulary the text in front of it was already written in. The gloss is not the
definition, so a tier re-cut has to move both places.

Every line is pre-indented to 15 spaces or more so `clean_indents` preserves the nesting in all
three prompts despite their differing baselines, binary at 12 and multiple-choice and continuous at
8.

The `[unverified attribution]` bullet says the tag named no outlet, or named one the pipeline could
not match against its retrieval record (the first cause added 2026-09-24, when class-description
tags started being rewritten; see `_SOURCE_TIER_TAG_INSTRUCTION`). The rest of the bullet is
unchanged: the claim may still be right, and it reads as untiered rather than low-tier.

### `_NULL_RESULT_READING`

How to read a searched-and-found-nothing result, shared verbatim across the three forecaster
prompts. On qid 44799 the gap-fill resolver reported "I found no authoritative public record" and
four of six forecasters converted that into "the authorization is absent"; the two that discounted
it scored best in the ensemble. A third bullet, that absence is weaker still where the actor has
already demonstrated the behavior, was dropped: it carried no receipt of its own and pushed the
wrong way on qid 43837, where eleven prior tournaments had been announced, none was found, and the
answer was NO. Same pre-indent contract as `_SOURCE_PROVENANCE_LADDER`.

### `_COUNT_IN_PERIOD_REFERENCE_CLASS`

Which reference class is admissible for a "how many X in period P" question, shared verbatim by the
binary, multiple-choice and continuous prompts, because count questions arrive as all three types.
On qid 44561 all six members built a "no failure announced yet, so Poisson(1.0)" schedule model
instead of the pooled FDIC bank-failure rate, and published far too low. Same pre-indent contract as
`_NULL_RESULT_READING`.

### `_SOFT_CLOCK_RULE`

The soft-clock rule: a target date the responsible actor is not BOUND to is evidence that a target
exists, not that it will hold. The measured receipt, the audit that produced it and the load-bearing
carve-out are under "What the base prompts carry, and where" above; three facts belong here. One
number that section rounds: deadline questions in general are calibrated at 0.25 published against
0.25 realized, which is why the rule names the MOVE rather than the question shape. The move is the
decomposition of P(target lands in window) times P(event given target) with the first term set near 1
on the strength of an announcement, and the audit found it on "will X happen before D" questions
whose only route to X was an ANNOUNCED target date. The rule renders in the binary and multiple-choice prompts only,
because the continuous prompt anchors on a range rather than on a probability, and separately in the
date prompt through `_date_axis`. And it carries no field in the STRUCTURED FORECAST block, since the
number belongs in the rationale and the block is written after the forecast is fixed. Same
pre-indent contract as `_COUNT_IN_PERIOD_REFERENCE_CLASS`.

### `_HISTORY_DISCHARGED_RULE`

History repeats past an acknowledged regime change: a member writes down a historical cadence, names
in the SAME rationale a reason it has been discharged (its driver was met, the deadline passed, the
rule changed), and keeps the old cadence as its central estimate anyway. The audit numbers, the
upper-bound caveat and the revert recipe are under "What the base prompts carry, and where" above.
Two additions. The significance figure behind the model-facing parenthetical: against the 17 to 18%
held rate the three older era bands show, 0 of 13 is an ordinary draw at p = 0.09, which is why the
text names the count as a small audit of this bot's own past forecasts rather than stating a flat
rate, since a bare figure reads to a forecaster as near-certainty. And the mechanics: binary and
multiple choice only, same pre-indent contract as the rule above.

### `_REMAINING_EXPOSURE_SENTENCE`

Apply the rate to the exposure that is LEFT. The q43837 receipt and the constant it replaced are
under "What the base prompts carry, and where" above. The mechanical facts: it is ONE sentence,
interpolated INLINE with no pre-indent into the binary conditional-hazard bullet, which is the same
rule specialised to recurring events, and standing alone as one bullet in the multiple-choice
outside-view step, which has no hazard bullet. Binary and multiple choice only, because the
continuous prompt anchors on a range rather than on a rate. The two-bullet constant it replaced
restated the hazard bullet twenty lines below it and the union clause five lines above it, so the one
rule read three times and the model was told nothing new twice.

### `_BINARY_CONDITIONAL_HAZARD_BULLET`

The binary outside-view step's conditional-hazard bullet, which OPENS with
`_REMAINING_EXPOSURE_SENTENCE` because the hazard check is that same rule specialised to recurring
events. It is a named constant rather than an inline interpolation for a formatting reason:
`ruff format` split the mid-bullet replacement field across three lines at an indent the surrounding
prompt does not use, in the region of this file that gets edited most often. The rendered text is
unchanged either way.

### `_MARKET_READING_RULES`

The three READING rules for the rendered market table that its own legend does not carry. The
legend, `research/market_retrieval/rendering.py` `MARKET_SIGNAL_LEGEND`, printed beside the table,
owns NOTATION: the liquidity labels and `no-liquidity-data`, the evidential row order, the four
`relation` tiers, RESOLVED, the sub-row arrow, `[remaining N]`, the "(Nd ago)" age suffix and
"demoted from same-date:". Re-teaching any of that in the prompt gave the model two
partially-overlapping glossaries, since the legend had grown labels the prompt never mentioned, so
the prompt keeps only POLICY, meaning what to DO with a row the legend has already explained.

Receipts: rules 2 and 3 are both q45189, where all three forecasters imported a thin single-strike
price at full weight, then read one bracket of a ten-bracket Kalshi ladder as an equality constraint
on a tail and cut the resolving bucket below their own prior. The bot published 0.130 and scored
-26.77 spot peer. Rule 1 is the ranked-retrieval design intent: an other-cut market is the same
quantity, so it is something to extrapolate from rather than to haircut.

`same_quantity_other_cut` is verbatim from `research/market_retrieval/ranking.py` `TIERS`. Renaming
it there without renaming it here silently teaches forecasters a vocabulary the table no longer
uses. The constant ships in all three forecaster prompts, gated with the rest of the market clause
on the snapshot section being present.

### `MARKET_SNAPSHOT_SECTION_HEADER`

The header the prediction-market research provider emits. `research/section_format.py`
`PROVIDER_SECTION_HEADERS` imports it from `prompts.py`, the same way it imports
`TS_ANCHOR_SECTION_HEADER`. The three forecaster prompts gate the whole market clause on this
substring, so the policy appears only when a snapshot was actually rendered. The gate is
prod-neutral: the provider emits the header whenever it rendered anything, including the
deliberate-empty "no sufficiently relevant market" sentence, and omits it only when it returned the
empty string, which happens under benchmarking, with the flag off, or on a soft-fail. Those are
exactly the prompts where about 1.5k characters of market policy had nothing to bear on.

### `_strong_evidence_market_clause`

The shared "prediction markets are strong evidence" clause for the three forecaster prompts. It
returns the empty string unless the research carries `MARKET_SNAPSHOT_SECTION_HEADER`, because the
clause is about reading a table, so it renders only when the table does. That is the same substring
gate the continuous prompt's time-series-anchor clause uses.

There is one path where the gate and the table come apart. On the provider's DELIBERATE-empty answer
the header is present with a single sentence and no table, so the reading rules render while the
legend that defines their notation (the sub-row arrow, `[remaining N]`, the liquidity and relation
labels) does not. That is left as is on purpose: a second gating condition would silently drop the
whole market policy from every prompt that DOES have a table the moment its false negative fired,
which is a far worse failure than three rules naming notation an empty section never uses.

The framing is identical across binary, multiple choice and continuous; only a few type-specific
words differ, namely the signal noun, the anchor verb phrase, the extrapolation target and the
projection tail. Centralizing it keeps the strong-evidence framing AND the reading rules in sync
across all three prompts. It is spliced into each prompt's `clean_indents` f-string and the embedded
newlines are cosmetic, since `clean_indents` and the whitespace-collapsing tests both ignore them.

Why the strong push is earned, so that a future prompt audit does not re-litigate it: past misses
traced to forecasters ignoring prediction markets, and the evidence is that a liquid, closely
matched real-money market is hard to beat. Treat one like a stock-market price, and this bot is not
assumed good enough to beat the stock market. Forecaster judgment operates in the match and mismatch
discounting (resolution criteria, resolution date, liquidity), not in waving the market off. The
all-caps shouting was dropped on 2026-07-18 as decoration; the strong push stays.

### `TS_ANCHOR_SECTION_HEADER`

The header the `timeseries_anchor` research provider emits, listed in `research/section_format.py`
`PROVIDER_SECTION_HEADERS`. The continuous prompt gates its anchor clause on this substring so the
guidance only appears when an anchor section is actually present.

### `_RESOLUTION_METRIC_ECHO_HEADER` and `_resolution_metric_echo_bullets`

The resolution-metric echo, a PHASE 0 disambiguation step that fires when the resolution criteria
name an official statistical series. The qid 44211 miss, June 2026 CBP southwest-border encounters,
had all six forecasters price the USBP-apprehensions component of a series that resolves on the
total: the research carried the definitional wedge, the historical conversion and an explicit
provider warning, and every model still resolved the ambiguity the same wrong way. Naming the exact
series and enumerating its variants BEFORE forecasting is the checklist-shaped guard, option (a) in
`scratch/residual_2026-07-18/followups/border_generalizability.md`. It is inert on questions with no
named series, and it addresses a measured 3 to 5 of 30 worst-miss family. Its design sibling is the
window-anchor block, `_forecasting_window_str`. The bullets are pre-indented to 15 spaces so
`clean_indents` keeps them nested under the prompt-native step header in both the binary (baseline
12) and continuous (baseline 8) prompts, the same trick `_SOURCE_PROVENANCE_LADDER` uses.

The continuous branch's reconcile bullet reads the displayed range as WEAK evidence about WHICH
variant resolves and as NO evidence about the magnitude of the outcome. It used to say that the
bounds were set by someone who could see the real series, so a candidate far outside the range is
probably the wrong variant. That is true on Metaculus and false on Mantic, where question writers
are paid for bot disagreement and 24.8% of resolved discrete and 53.7% of date questions escaped
their range (the receipt is under `_MANTIC_OUT_OF_RANGE_RATE_DATE` below), so a forecaster that
extrapolated correctly was told by this prompt to pull its percentiles back inside, which is worth
roughly 195 baseline points between the two outcomes. The 44211 correction survives: inside the
range confirms nothing.

### `_METACULUS_SCORING_SENTENCE` and `_MANTIC_SCORING_SENTENCE`

The one sentence every forecaster prompt opens with about how it is scored, chosen by the platform
the question came from, which `question_platform` reads off `page_url`. Until 2026-09-08 all six
prompts named "the Metaculus peer score" or "Metaculus' log-score", which was imprecise on Metaculus
(the bot tournaments score SPOT peer) and false on Mantic, whose Crucible leaderboard is spot
BASELINE: the reference there is a uniform distribution rather than the other forecasters, and no
community prediction exists while a question is open. Both scores are strictly proper, so the honest
forecast is optimal on both. The Mantic wording says so outright, because "compared to your peers"
invites contrarian drift, which a proper score only punishes. The sentence is shared with the three
stacking prompts, which carried the same wording before the swap.

### `multiple_choice_prompt`

The STRUCTURED FORECAST block example carries the REAL option names as JSON keys, because a strict
parser can only map placeholder keys like `Option_A` back onto real options through prose lines, and
the prompts no longer emit those. Same reason as in `_option_probs_example` above and in
`stacking_multiple_choice_prompt` below.

### `_CONTINUOUS_SCORING_RULE`

The continuous scoring paragraph, shared by the numeric, date and stacking-numeric prompts. Until
2026-09-08 it described Metaculus' implementation, a uniform 0.01 PDF floor (so that excluding the
truth costs ln(0.01), meaning -4.6) and a sharpness cap near 35. That told the model the cliff below
an out-of-range outcome was an order of magnitude shallower than it is on Mantic, where the
out-of-range bucket is scored against a fixed 5% reference with no floor: 1% of mass there scores
-80.5 baseline points, 5% scores 0, and 50% scores +115, and 28% of Series 1 questions resolved out
of range. The proper-scoring sentence is true on both platforms and stays. The open-bound sentence
replaces "scored as a binary event", which said nothing about the reference the bucket is scored
against.

### `_PER_BIN_OUTPUT_RULE`

The interpolated `min_step` is THIS grid's floor, not the aggregate's 5% out-of-range tail floor,
which the member is deliberately not told about. The last sentence states what
`numeric.pmf_cdf._blend_to_cell_floors` does so the model does not pre-subtract it, and the block
extraction rung fails on a missing key, which is why the rule asks for every key by name. The full
entry is under "Per-bin elicitation" above.

### `_MANTIC_OUT_OF_RANGE_RATE_DATE`, `_MANTIC_OUT_OF_RANGE_RATE_QUANTITY` and their per-bin twins

Mantic's measured out-of-range base rates, rendered only on a Mantic question that has the relevant
open bound; `question_platform` reads the platform off `page_url`. Metaculus questions never render
either sentence, so Metaculus behaviour cannot move.

The receipt is the Series 1 corpus under `scratch_docs_and_planning/mantic_research_2026-09-08/`,
520 questions with annulled ones excluded: 101 of 188 date questions with an open upper bound
resolved ABOVE it (53.7%); 35 of 141 discrete (24.8%) and 16 of 133 numeric (12.0%) resolved outside
their range, 51 of 274 quantitative questions combined (18.6%), against 2.2 to 2.6% in this bot's
Metaculus archives. Seven of the numeric escapes are stored as raw values outside the range rather
than as a bound token, which is how a filter on the tokens alone undercounts them at 9: posts 512,
460, 426, 396 and 305 on linear grids, and 387 and 200 on log grids.

The pipeline fact both sentences end on is structural. When all 13 percentiles sit inside the range,
the published CDF puts exactly 1% beyond each open bound, which Mantic scores at -80.5 points when
the outcome lands there.

### `_MULTI_RESOLUTION_CONTINUOUS_TEMPLATE` and the four multi-resolution rules

Mantic Series 2 "one forecast, many resolutions", section 7 of the rules document, live on Preseason
2 post 650, eleven daily bitcoin closes. The single submitted distribution is scored against every
resolution value and the scores averaged, `sum_i (c_i / C) * k * ln(p_i)`, whose argmax is
`p_i = E[c_i] / C`. The optimum is therefore the expected EMPIRICAL distribution of the resolution
set, a mixture over the instances, not the predictive distribution of any one of them. Priced on
post 650 at a 77k spot price and 2.5% daily volatility, a day-one distribution loses 38.7 baseline
points to the mixture. For multiple choice and binary the same argument gives the expected FREQUENCY
over options and the expected fraction of Yes.

The rules are gated on the question's own `multi_resolution` field, which is absent on Metaculus.
The gate is an identity test against `True` because a test stub's chain of `MagicMock`s is truthy.
The resolution count is deliberately NOT interpolated: `resolutions` is null while the question is
open, and the count lives only in the criteria prose. Base prompts only, per the standing rule, so a
stacker on a multi-resolution question would never learn this, which is a FUTURE.md note against
re-enabling numeric stacking.

### `_scoring_grid_clause`

The scoring grid, named when the platform declares it. Mantic buckets a continuous CDF into
`inbound_outcome_count` bins and scores the bin the outcome falls in, so detail finer than a bin is
invisible to the score. On a coarse grid, and post 651 had 12 one-day bins while Series 2 makes
day and week granularity and power-of-ten step sizes the default, a model that does not know the
grid smears a confident view across neighbouring bins.

`precision` (quantitative) and `date_granularity` (date) exist only on Mantic payloads, so a
Metaculus question renders nothing here. The bin count is the typed `cdf_size - 1`, the number the
CDF builder keys off, so the prompt can only ever describe the grid the pipeline submits. Mantic's
OpenAPI defines `precision` as the additive bin width on a linear scale but as the RATIO between
adjacent boundaries on a logarithmic one, which is why a `zero_point` grid names its geometry and no
width.

### `_date_axis`

The soft-clock rule is the date question's natural home. A "when will X happen" question with an
announced target date is the announced-but-unbound shape the rule was measured on, where binary
forecasts averaged 44% on events that happened 8% of the time, and here the mass on the target date
IS the timing term the rule asks to price separately. That is why `_date_axis` appends
`_SOFT_CLOCK_RULE` to its reference-class rules while `_numeric_axis` does not.

### `_date_percentile_blocks`

The schema example spans the displayed range in the question's own rendering, so the model sees the
exact string form its grid expects, a calendar date on a day or week grid and a timestamp on a
legacy fine grid, rather than an illustrative value from another calendar. The one-day timestamp
example in the notes is the range's midpoint day for the same reason.

### `_PMF_KEY_RULES`

How a per-bin prompt describes its keys, one sentence per label style, with the dict total over
`BinLabelStyle` so that every per-bin question renders exactly one key sentence. The centre sentence
is unconditional because `_is_center_aligned` admits any step: 5 of the 46 coarse Mantic centre
grids step by 5, 500, 0.25 or 0.1, and a key `5` read as "5 up to 10" puts a belief of 3 one bin
off. The full entry is under "Per-bin elicitation" above.

### `_continuous_prompt`

The time-series-anchor guidance is surfaced only when an anchor section is actually in the research,
through the same cheap substring gate `_strong_evidence_market_clause` applies to the market clause,
so neither clause spends prompt budget on a table the forecaster does not have.

### `stacking_multiple_choice_prompt`

The STRUCTURED FORECAST example carries the real option names as JSON keys because the downstream
parser can only recognize the actual options, exactly as in `multiple_choice_prompt`.

## Test pins

Every surviving rule has a presence pin on its wording and every removed constant or phrase an absence pin (`tests/prompts/`, split by surface into `test_base_prompt_rules.py`, `test_structured_block.py`, `test_research_clauses.py`, and since 2026-09-08 `test_date_prompt.py` for the date axis, `test_platform_and_mantic_clauses.py` for the platform-aware and Mantic-gated clauses and `test_pmf_prompt.py` for the per-bin elicitation and the byte-level pin on the percentile fills, with the shared question stubs and prompt renderers in `tests/prompt_builders.py`; plus `tests/test_open_bound_guidance.py`); the market gate's header is pinned against `PROVIDER_SECTION_HEADERS` so the two cannot drift; and no base-prompt rule may appear in the three stacking prompts.

### What `tests/prompts/test_research_clauses.py` asserts, and why

The prose that used to sit in that file's comment blocks and in its longest class docstring moved
here on 2026-09-09, in the same pass that emptied the `prompts.py` comments. The assertions
themselves are unchanged.

`TestPredictionMarketFraming` guards the whole market clause. Three things it holds the prompts to,
each with the failure it prevents. The framing must be STRONG EVIDENCE to weight heavily rather than
the old "not beholden" footnote, with a precise conditional adjustment: anchor when the market's
resolution criteria AND resolution date match the question, discount proportionally to any specific
named mismatch, and extrapolate across a date-only mismatch instead of applying a vague haircut. The
clause must NOT carry a "you may deviate from a market" carve-out, because that sentence undercut
the strong-evidence framing; the general principle that a forecaster may supplement the research
with its own training knowledge is a SEPARATE, prompt-wide directive, asserted on its own by
`_assert_general_expertise_principle`. And the gate: the whole clause renders ONLY when the research
carries the rendered `## Prediction Market Snapshot` section, the same way the continuous prompt
gates its time-series-anchor clause. That gate is prod-neutral, since the provider emits the header
whenever it rendered anything, including the deliberate-empty "no sufficiently relevant market"
sentence, and
omits it only when it returned the empty string, under benchmarking, with the flag off, or on a
soft-fail. Those are exactly the prompts where the market policy had nothing to bear on, which also
makes the leakage story simpler than it was: a benchmarking prompt no longer carries three paragraphs
about markets it cannot see. The mode-dependent leakage guard on the RESEARCH side still lives on
`web_research_prompt`, in `test_market_ask_present_non_benchmarking_absent_benchmarking`.

Notation against policy is the other axis that class pins. The rendered table's own legend,
`MARKET_SIGNAL_LEGEND`, defines the relation tiers, the evidential order, RESOLVED, the sub-row
arrow and `[remaining N]`; the prompt keeps only the three READING rules the legend does not carry,
and the negative assertions list the exact phrases the pre-2026-09 clause carried that duplicated
the legend. The three rules, in the order `_assert_strong_evidence_framing` checks them: rule 1, an
other-cut market is the same quantity at another date, threshold or source, so it is something to
extrapolate from rather than to haircut; rule 2, which label wins when the relation and liquidity
axes disagree, where a tight relation on a THIN market is the shape that cost q45189 (all three
forecasters imported a thin single-strike price at full weight), so the rule carries its reason (a
thin price is noisy however tight its relation) and is directional, widen around the implied value
rather than transplant it; rule 3, a family of sub-rows is a distribution over the market's own
question, so reading one bracket as an equality constraint on a tail is a category error, which is
the other half of q45189, where all three members cut the resolving bucket below their own prior that
way. Every assertion in that class first collapses whitespace, so none of them depends on where
`clean_indents` happened to wrap a line; `TestSourceTierTagging` does the same.

Four smaller notes from the same file. The benchmarking carve-out on the gap-fill analyzer is
checked in both halves at once: each accepted substring carries the "DO NOT" directive AND its
attachment to the words "prediction market", so a prompt that kept one without the other fails. The
time-series-anchor
assertions check that the clause points at the section and says what the band IS, including the
independent-window caveat, without telling the model how to weigh it, which is the neutrality the
operator chose on 2026-07-18. `TestWebResearchPromptPrimarySources` asserts that at least 3 of its 4
example domains appear anywhere in the prompt, so the prompt's domain list can evolve without the
test breaking on a single-domain rename. And the summarizer's `[PRE-WINDOW]` pin covers both spellings, the full tag on
first occurrence and the short tag afterwards, because that compression is display-only and the
pre-window warning semantics must stay intact.

## Research-side prompt rules

- **`GRADE EVERY GAP` in `gap_fill_analyzer_prompt`** (2026-09-09). Three grade fields beside each gap in the schema (`answerable_now`, `already_in_first_pass`, `same_need_as`) make the prompt's own discipline rules checkable by code: `research/targeted.py` `triage_gaps` drops a failing gap before its resolver call, with the rules and their receipts in `docs/research.md` "v1 triage". The clause sits immediately before the schema, states the consequence once with its reason (code reads the grades and drops a failing gap before its search is paid for), defines each field in one sentence, fixes the position convention (1 = the first gap) and names the two commonest repeat shapes from the archive (a dashboard and its monthly summary; official and preliminary results, questions 45088 and 44880). Receipt: about a third of v1's resolver calls bought nothing on the archive, future-dated asks 18% of gaps, re-bought first-pass readings on 47% of the forced current-reading slots, a paraphrase pair on one question in three (`scratch/cost_pass_2026-09-09/v1_gap_redundancy/REDUNDANCY.md`); a positional cap was rejected because it dropped the useful gap on 4 of 6 traced questions. The ANSWERABLE NOW block keeps its mandate and carve-out untouched, so the two sentences it shares with the `answerable_now` and `already_in_first_pass` definitions (the never-the-resolution-date rule and the dated-reading carve-out) are a candidate prose cut for a later pass, not taken here. Size: the analyzer renders at 4,766 whitespace-collapsed characters on a minimal fixture against 4,018 at the previous commit (+748, +18.6%), the 2026-09-02 method from the size paragraph above on a minimal fixture rather than that paragraph's market-bearing one, which is why the before figure reads 4,018 rather than 4,023; the whole increase is the clause and the three schema lines. Pins: `tests/prompts/test_research_clauses.py` `TestGapFillAnalyzerGradeFields`.
- **Resolution criteria and fine print in `gap_fill_search_prompt`** (2026-09-09). The per-gap resolver reads the question's `resolution_criteria` and `fine_print` under a labelled block ("Resolution criteria (what the question actually resolves on):", then "Fine print:" when there is any) right after the question title and before the search instruction, because a gap is routinely "which of these figures resolves the question" and the model answering it had only the title to go on. Receipt q44267, the 2026-09-09 round's worst miss at -95.66 spot peer: the analyzer, which already saw both fields, posed the right gap (the headline count of People's Liberation Army sorties around Taiwan, or the subset that entered the air-defence identification zone), and the resolver ruled for the headline count on the strength of sister question 40685, whose title asks about sorties "operating around Taiwan"; all six forecasters ratified it, and the resolving value (26, the zone-entry subset) fell below every member's 2.5th percentile. The slot is two required parameters on the prompt and a caller that passes the question object (`research/targeted.py` `_resolve_single_gap`), with no flag and no branch beyond omitting an empty fine print; the v2 driver brief (`research/agentic/driver_prompt.py` `build_user_brief`) already carried both. Pins: `tests/prompts/test_research_clauses.py` `TestGapFillSearchPrompt` (the slot, its placement, the empty fine print, the `(none provided)` placeholder shared with the analyzer, and a regression pin built from 44267's real criteria text asserting the zone-entry wording is present) and `tests/test_gap_fill_pass.py` `test_resolver_prompt_carries_the_resolution_criteria_and_fine_print` for the wiring.
- **Two vintage / as-of bullets in `web_research_prompt`'s GUIDELINES**: carry the publication date of every dated or forward-looking claim, and for a schedule or plan state when and where it was announced rather than presenting an undated recollection as a current fact. Both consumers (native search and gemini) see them.
- **`_SEARCH_LINK_CITATION_CLAUSE`**: the Gemini branch of `citation_clause` requires exact markdown links copied from the search tool, including Google's redirect tokens, and keeps source-tier tags alongside them. The formatter resolves and numbers those links; unresolved links are marked `[unverified link]`. The `markdown` (native-search) branch is unchanged.
- **`OUTSIDE_VENUE_MARKET_ODDS_POLICY` / `_OUTSIDE_VENUE_MARKET_ODDS_BULLET`**: one policy, two renderings: the public constant is the sentence, the private one is that sentence as a FOCUS AREAS bullet. `web_research_prompt` interpolates the bullet; the two Perplexity prompts (`research/providers.py::_perplexity_provider`, `research/orchestrator.py::_call_perplexity`) interpolate the sentence, because their bodies are single `clean_indents` blocks where an interpolated column-0 line would defeat the dedent. All three carry it because Perplexity becomes the PRIMARY provider whenever AskNews credentials are absent, and the two Perplexity sites kept asking for "all relevant prediction markets" after the bullet was narrowed, until a review caught it. **Narrowed** (wording confirmed verbatim by the operator 2026-09-01) to market-implied or crowd odds from sources OTHER than the four venues the prediction-market provider already snapshots live, with an explicit instruction not to report Polymarket/Kalshi/Manifold/PredictIt prices out of search results, whose indexed copies are usually days stale. Narrowed rather than removed because every realized instance of decisive market evidence came from outside those four (Good Judgment Open on q44869, CME FedWatch on q45401, the Metaculus crowd on q20683) while the only measured harm mode was stale covered-venue prices contradicting correct live snapshot rows. Benchmarking still suppresses the bullet entirely; note the leakage guard was strengthened at the same time, because after the narrowing the old "the string `Prediction market` is absent" assertion passed for the wrong reason (it is absent in both modes now), so the test asserts no covered venue, Metaculus, or CME FedWatch appears anywhere in a benchmarking prompt.

Not narrowed, and deliberately: the Exa and Perplexity fallback providers carry their own market-seeking instructions (`research/providers.py`, `research/orchestrator.py`) that this narrowing does not touch. They never run in prod (AskNews is the primary), so the inconsistency is latent; those two strings are where the narrowing would go if it is ever wanted.
