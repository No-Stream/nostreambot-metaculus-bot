# Numeric forecasting and aggregation

This is the reference for how the bot turns a numeric question into a probability
distribution. It covers the full path: each forecaster declares the standard percentile
set and probabilities beyond open bounds, the bot extracts and cleans them, builds a
201-point CDF that
satisfies Metaculus' server-side constraints, aggregates across the ensemble in CDF
space, and decides whether the ensemble disagrees enough to trigger conditional
stacking. On a coarse Mantic grid whose bins are the outcome space, forecasters instead
declare one probability per bin and the CDF is built from that PMF ("Per-bin elicitation",
between Steps 4 and 5). It also documents the time-series anchor
research provider, which grounds numeric forecasts in the resolution series' own history.

Every model name, flag, and default here is verified against the code. Where a value
lives in a constants file this doc names the constant rather than restating its value,
so read the constant for the current magnitude.

## What a numeric question looks like

A Metaculus `NumericQuestion` has a lower and upper bound, and each bound is either
open or closed. A closed bound is a hard limit (the outcome cannot fall past it); an
open bound is just the edge of the displayed range, and the true outcome can land
beyond it. Some questions also carry a `zero_point` (log-scaled axis) and a `cdf_size`
that differs from 201 (a discrete question). All of these change how the CDF is built,
so they thread through every stage below.

## Step 1: forecasters declare the standard percentiles

Each forecaster is prompted to emit the standard percentile set for its distribution.
`STANDARD_PERCENTILES` in `numeric/config.py` owns the set, and
`EXPECTED_PERCENTILE_COUNT` is derived from it with `len()` so the count can never
disagree with the list, and `STANDARD_PERCENTILES_CSV` is the label string the prompts
interpolate. Never restate the percentiles anywhere else in code or prompts; derive them
from `numeric/config.py`. The listing below is illustrative, for readers of this doc
only, and `numeric/config.py` remains the authority. As shipped today the set is:

```
1, 2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5, 99
```

Stored internally as decimals in `[0, 1]` rather than as 1-100 labels. The tail anchors
P1 and P99 exist so a forecaster can express probability mass beyond an open bound: if
you believe the outcome very likely exceeds an open upper bound, you place P50 and above
outside the displayed range. The prompt (see `prompts.numeric_prompt` and the bound
helper text from `bound_messages` in `numeric/utils.py`) tells forecasters exactly this
and warns them not to pile percentiles against an open edge.

The forecaster writes its percentiles inside a fenced ```json STRUCTURED FORECAST
block, which the prompt requires to be the last thing in the response. For each open
side it also writes an explicit decimal probability: `prob_below_lower_bound` for the
lower side and `prob_above_upper_bound` for the upper side. Numeric, discrete and date
percentile prompts use these fields. Closed sides have no such field.
The percentiles remain unconditional quantiles of the full distribution, including
out-of-range outcomes. This makes the member's tail belief explicit instead of letting
the outermost percentile's level determine it when every declared value is in range.

## Step 2: value extraction ladder

`extract_numeric` in `value_extraction.py` pulls the declared percentiles out of the
model's free-text rationale. It runs a four-rung ladder, most deterministic first, and
stops at the first rung that produces a valid result:

1. **block**: parse the fenced ```json block directly (`json.loads` plus Pydantic
   validation against `NumericStructured`). When the block carries a
   `declared_percentiles` dict, it is lifted straight into `Percentile` objects
   (`_numeric_from_block` in `value_extraction.py`). This is the normal path.
2. **repair**: if the block was malformed, run deterministic JSON repair
   (`json_repair`), or scan the rationale's last `_TAIL_SCAN_CHARS` characters for
   balanced braces. Skipped when rung 1 already produced a schema-valid model (repairing
   valid JSON is a no-op).
3. **llm**: as a last resort, call the parser LLM (`parse_structured`) over the full
   rationale. Logged loudly as a salvage.
4. **raise**: if every rung fails, raise `ValueExtractionError`; the caller drops or
   soft-fails that forecaster.

Post-rung validation is strict regardless of which rung produced the value
(`_validate_numeric` in `value_extraction.py`): it requires every label in
`STANDARD_PERCENTILES` and returns exactly that set, never padded. That is a SHAPE
check, and shape alone is not enough: a rung decoding under a schema must emit numbers,
so the validation is also a FIDELITY check, and values must be finite and ordered the way
their labels claim. See
[value_extraction.md](value_extraction.md) for the fidelity rules on all three question
types and the incidents that motivated each one.

Every successful extraction logs one line:

```
EXTRACTION_RUNG: question=... model=... qtype=numeric rung=... block_present=...
```

Watch for `rung=llm` and `block_present=False` in prod logs. Those are the two signals
that a forecaster stopped emitting a clean block.

The same call also decides discrete vs. continuous. `run_numeric_forecast`
(`forecaster_runners.py`) reads `NumericStructured.outcome_type` from the block first,
and only falls back to a parser LLM call (`OutcomeTypeResult`) when the block does not
declare it. That vote is carried forward to the discrete-snap decision in Step 7.

The two optional open-side probabilities come from the same accepted block or repaired
candidate as the percentiles. Each must be a finite number in `[0, 1)`; a present null,
boolean or nonnumeric value is invalid. An absent field keeps the existing
percentile-derived tail on that side. An invalid field is ignored on its own side and
the member survives without a new parser call; if both stated probabilities are valid
but sum to at least one, both are ignored. LLM salvage recovers percentiles only, so
both tail fields are missing on that rung; it never reuses tails from a rejected
candidate. Their validation and use are recorded by `NUMERIC_STATED_TAIL`, separately
from the existing extraction and member markers.

## Step 3: sanitize_percentiles

`sanitize_percentiles` (`numeric/pipeline.py`) turns the raw declared percentiles into
a clean, strictly-increasing, in-bounds set. In order:

1. `filter_to_standard_percentiles`: keep only the standard labels, drop extras and
   duplicates.
2. `validate_percentile_count_and_values`: assert the count and label set match
   `EXPECTED_PERCENTILE_COUNT` / `STANDARD_PERCENTILES` exactly
   (`numeric/validation.py`).
3. `sort_by_percentile_level`: order by percentile level.
4. `_apply_jitter_and_clamp` (`numeric/pipeline.py`): detect count-like (integer-
   adjacent) clusters and spread them, jitter exact duplicates, clamp values just outside a
   closed bound back inside it (the tolerance and landing rules are below), then enforce
   strictly increasing values.
   **On the 201-point continuous grid a WHOLE-SET epsilon collapse is not spread.** A model
   declaring (near-)the same value at all 13 percentiles has declared no width, and spreading
   it manufactured a ±6-unit distribution, which was precisely what let it pass Step 8's
   span-ratio test, so the invented width was load-bearing for publishing a forecast the
   model never stated. There a point mass gets only the format minimum (the jitter /
   strict-ordering epsilon) and reaches the guard with its own honest span, which
   withholds the forecaster; only PARTIAL clusters are spread. **Where the published bins
   are the outcome space** (`grid_is_outcome_space`, below) the collapse IS spread, like any
   other plateau and under the same one-bin cap: "100% on 2026-09-16" is fully expressible
   on twelve one-day bins and the date prompt invites that shape, so the member publishes
   with its mass in that day (0.97 of it on post 651) instead of being dropped as a unit
   mismatch. The width the spread adds stays inside the bin the forecaster named, and a
   genuine scale error on such a grid sits several bins outside the bounds and still raises
   in the clamp. Each collapse logs `NUMERIC_DEGENERATE_DECLARATION: question=... model=...
   n_unique=... span=... value_eps=... spread_applied=true|false` (harvested as
   `numeric_degenerate_declaration`; `spread_applied` reports which branch it took), so the
   per-model incidence is a query rather than a guess.
5. `_maybe_widen_tails`: optional tail widening (Step 4).

The schema deliberately accepts non-decreasing values: tied values are valid concentrated
declarations and the sanitizer can separate partial clusters. The prompt still asks for
strictly increasing values, while the schema rejects decreases before
`sort_by_percentile_level` can order the declarations. A whole-set collapse remains valid
schema input; on the continuous grid sanitization does not invent width for it and the
unit-mismatch guard withholds the member. The archive audit found exact ties in 2 of 346
declarations and no whole-set collapses, supporting this distinction without adding a
separate distinct-value requirement.

The closed-bound clamp (`numeric/bounds_clamping.py`) is governed by two different numbers.
`calculate_bounds_buffer` is the accept-or-raise TOLERANCE: a declared value may sit outside
a closed bound by at most the larger of the range-based tolerance (1% of the range, a flat
1.0 once the range exceeds 100) and one grid bin (`grid_bin_width`), and is clamped in;
anything further out raises and the member is dropped as a scale error. The bin floor holds
on EVERY grid, the 201-point continuous one included, because a value within one bin of the
edge is indistinguishable from the edge once bucketed: on a Metaculus question with range
> 200 the drop-versus-clamp threshold is therefore `range / 200` rather than the old flat
1.0, which removes the discontinuity where 1.0 was 1% of a 100-wide range and 0.005% of a
20,000-wide one. On a date question the axis is epoch seconds, so the flat 1.0 was a
one-second tolerance and a date named one day outside a closed bound used to drop the
member. The tolerance changes only drop-versus-clamp, never where a clamped value lands:
that is `minimum_separation` (`numeric/config.py`), the same standoff the spreader and the
strict-ordering passes keep just inside a closed bound. Landing the value one tolerance
inside instead moved it a whole bin on a coarse grid, and the left-to-right strict-ordering
pass then dragged every percentile declared inside that first bin up behind it (post 651:
6.3 points of mass left the day the member declared; a [0, 20000] Metaculus question
published four percentiles at 100.0 when three were in range). The heavy-clamping WARNING
counts values at that same standoff, so an in-range forecast concentrated near a bound no
longer reads as clamped.

Wherever the published bins are the outcome space, the spreader keeps a plateau inside the
bin it names: the plateau's TOTAL spread is capped at one bin width, `grid_bin_width`
(`numeric/config.py`), so the per-position spread is the smaller of the count-like unit and
`bin_width / (plateau_size - 1)` (`_spread_cluster_values`, `numeric/cluster_processing.py`).
The grid points of a discrete question are its bin edges, so a plateau at integer k spread
past k +- 0.5 handed the mass the forecaster put on k to the neighbouring bins: on the
three-bin Mantic post 253 a declared 90% on the first bin published 0.558 (Mantic edge-case
review 2026-09, rank 7; the corpus reproductions are
`tests/test_numeric_discrete_grid_plateaus.py`). What a plateau pins is an interval
(`P90 = 0` and `P95 = 1` say `F(0.5)` lies in [0.90, 0.95)), and the published mass is
that interval's lower edge less the uniform mixture. A PARTIAL plateau at a bound gets
exactly the pre-2026-09 treatment on every grid: the symmetric spread (under the cap here),
the closed-edge clamp to the `minimum_separation` standoff, then the shift-up against the
preceding declared value and the compression against the following one. On an OPEN bound a
partial plateau may therefore straddle the bound, and one declared beyond an open bound keeps
its symmetric spread where its neighbours leave room; both are the behaviour benchmarked on
Metaculus, pinned in `tests/test_cluster_processing.py`.

The WHOLE-SET collapse on an outcome-space grid, all 13 declared values equal, is the one
shape placed differently (`_spread_whole_set_collapse`, `numeric/cluster_processing.py`),
decided on the declared value `v`, the median element of the sorted set, with exact
comparisons: an exactly-equal collapse compares exactly against the bounds, an epsilon-chain
of near-equal values (which `is_degenerate_cluster` also admits) is centred rather than placed
off its first value, and `np.mean` is never used, because its rounding read a collapse exactly
on an open bound of 1.3 as beyond it and published 7 of 13 values past the bound (codex
re-checks, 2026-09). With `lower <= v <=
upper`, equality included, the full 12-step span goes inside the range, translated as needed
with its spacing intact: the bound value buckets into the terminal bin, so a plateau ON the
bound (a forecaster writing the bound's own timestamp at every percentile, which the date
parse notes invite) keeps its full span in that bin. Spread symmetrically about the bound
instead, half of it crossed an OPEN edge and published `cdf[0] == 0.5` on post 651's grid, a
coin flip on "before the window" invented from a declaration that named nothing before it,
while the CLOSED-edge clamp folded it to half its width, a span ratio of 6e-6 that Step 8
withheld at its 1e-5 threshold, the very drop the outcome-space carve-out exists to remove
(codex second-opinion review, 2026-09). The plateau may start exactly on the bound: a
declared P1 equal to the lower bound already does, and it builds the same CDF as one at the
standoff (closed: cdf[0] = 0; open: the structural 0.01). With `v` strictly beyond an OPEN
bound the collapse is a real out-of-range declaration and the spread stays symmetric about
`v`. With `v` strictly beyond a CLOSED bound the plateau starts at `v`, neither pulled inside
nor reaching further out, because the clamp runs after the spreader and judges the outermost
value against its tolerance: within tolerance the member is folded onto the bound, beyond it
the member is still dropped as a scale error. A whole-set collapse has no neighbours, so the
shift-up and compression repairs never re-space it; the three rounds of translating partial
plateaus too, each of which opened a new hole at the repairs, are why the rule is this
narrow. The predicate is
`grid_is_outcome_space` (`numeric/config.py`): a natively discrete question
(`DiscreteQuestion`, which every Metaculus discrete and every Mantic quantitative question
parses as) or any non-201 grid, the same predicate on which Step 7's discrete snap skips. On
those shapes the spread is the final word; on the 201-point continuous grid of a
`NumericQuestion` the unit spread is pre-processing the vote-gated snap can re-concentrate,
so that grid is byte-identical. The ablation harness rehydrates questions as
`NumericQuestion`, which diverges from prod only on Mantic 200-bin discrete questions, absent
from the Metaculus archives it replays.

It also resolves the `zero_point` every CDF for the question is built with
(`resolve_zero_point` in `numeric/validation.py`): the question's own, or `None` when it
equals the lower bound, where the geometric axis divides by zero. A non-201 grid is not a
reason to drop the log scale. The platform declares geometric bins on a `zero_point`
question at any bin count and maps the submitted probabilities onto them positionally, so
the linear axis the bot forced on every non-201 grid until 2026-09 would have published
probabilities computed at linear positions against geometric bins (a `[1, 1e6]` question
forecast at 1,000 read back as a median near 1.2). Metaculus never reached that branch,
because its log-scaled questions are always 201 points; Mantic's fine grids do.

## Step 4: tail widening

`widen_declared_percentiles` (`numeric/tail_widening.py`) can fatten the tails by
scaling each percentile's distance from the median in a transformed space (bounded
logit for closed-closed questions, log transforms for one-sided questions, identity for
open-open). The stretch ramps from zero near the center to a maximum `k_tail` at the
deepest tails.

In production this is an identity pass: `TAIL_WIDEN_K_TAIL` and
`TAIL_WIDEN_SPAN_FLOOR_GAMMA` (`numeric/config.py`) both default to their no-op
settings, so nothing is widened and no span floor is enforced. That is why: an empirical
calibration on 43 resolved numerics found `k_tail=1.0` gave the best-calibrated tails,
and `k_tail=1.25` moved away from ideal in every segment (see
`scratch_docs_and_planning/tail_widening_empirical_calibration.md`). Both knobs are
still per-call configurable, and the function raises `ValueError` if asked to narrow
rather than widen (narrowing is not implemented) or on a negative `span_floor_gamma`.

## Per-bin elicitation

Steps 1 to 4 and the PCHIP build in Step 5 are the percentile path. On a coarse grid whose
bins ARE the outcome space, the bot instead asks each forecaster for one probability per bin
and builds the CDF from that PMF directly. Thirteen percentile anchors cannot say "zero on this
bin", and PCHIP spreads mass onto bins the criteria exclude: Mantic question 651 (twelve
one-day bins, 8 to 19 September 2026, a trading-day question) has three weekend bins that
cannot resolve, and a percentile declaration put about a quarter of the mass on them, `50 *
ln(0.75) = -14.4` baseline points for no information. On a count grid of 13 bins or fewer,
13 strictly increasing percentile values cannot even exist, so the model repeats integers and
the pipeline reads a degenerate declaration. Motivation, corpus counts and the rejected
alternative (an `excluded_bins` list on the percentile block): backlog decision-7 "excluded_bins list on the percentile block (Mantic Crucible)".

**The gate** is `elicit_per_bin` (`numeric/config.py`): the question's platform is in
`PMF_ELICITATION_PLATFORMS`, the grid is an outcome space (`grid_is_outcome_space`: a
`DiscreteQuestion`, or any non-201 grid) and the bin count `cdf_size - 1` is at most
`PMF_ELICITATION_MAX_BINS`. The platform set is Mantic-only for the first landing, because the
ask is unproven live and on Metaculus it would move about half of all discrete questions (141
of 300 sampled have 31 bins or fewer); adding `PLATFORM_METACULUS` to that one constant is the
whole switch. By construction the gate is false on the 201-point continuous grid and on a
200-bin Mantic discrete question, so both keep the percentile path byte-identical. A date
question is gated on its epoch adapter (`numeric/date_axis.py`), which carries `page_url`.

**The labels** (`numeric/pmf_grid.py`, `pmf_grid`) name each bin the way the platform displays
it: the ISO date of the first UTC day a bin covers on a `day` or `week` date grid, the bin centre
on a centre-aligned quantity grid (the discrete convention, `range_min = nominal_min - step /
2`; a count question's bins are `0`, `1`, ...), and a right-closed `a to b` interval otherwise.
The reserved keys `below_range` and `above_range` (`PMF_BELOW_RANGE_KEY` / `PMF_ABOVE_RANGE_KEY`
in `constants.py`) appear only where the bound is open. The edges are `build_cdf_value_grid` on
the question's own axis, which reproduces the platform's `continuous_range` exactly, so a label
names the same bin the platform scores. The prompt's bound messages come from
`pmf_bound_messages` (`numeric/utils.py`), the per-bin twin of `bound_messages`: same displayed
bounds and order, but an open bound is described as its reserved key ("the probability that the
outcome resolves above ...; scored as its own outcome") and a closed bound says there is no
such key, instead of the percentile wording.

**The build** is `build_pmf_distribution(declared, view, model_name=...)` in
`numeric/pmf_cdf.py`. Its input is the platform's own PMF shape, `[below, p_0, ..., p_{N-1},
above]` with `N + 2` entries and a closed tail's entry at 0.0 (the shape Mantic exposes for
every stored forecast and the `MEMBER_FORECAST` marker records as `raw`). In order:

1. **Refuse mass in a closed tail.** A positive `below` on a closed lower bound, or `above` on a
   closed upper bound, raises `ValueError`. The extraction ladder rejects a reserved key on a
   closed bound, so no runner input arrives this way, but the builder is a public boundary and a
   guard fails shut: the first design pinned `cdf[0] = 0.0` after the cumulative sum, which
   silently moved a closed tail's mass into the first in-range bin (the plan review reproduced
   0.297 landing in bin 1). A malformed vector (wrong length, non-finite, negative, zero sum)
   raises `ValueError` too.
2. **Normalise** to sum 1. The ladder bounds the declared sum to
   `pmf_prob_sum_tolerance(key_count)` of 1.0 (`structured_output_schema.py`: the larger of 0.02
   and 0.005 per key, so 0.06 on post 651's 12-key grid and 0.165 at the 33-key maximum), and the
   normalisation here is why only the shape is load-bearing.
3. **Blend toward the cell floors.** The server needs every in-range step to be at least
   `round(0.01 / N, 9)` and an open tail to be at least 0.001 (`grid_step_constraints`). Let `f`
   be the vector of cell floors, each non-zero floor raised by `PMF_FLOOR_MARGIN` (1e-9, so the
   server's 9-decimal PMF rounding can never land a cell one unit under its floor) and a closed
   tail's floor 0, and `t = f / sum(f)` the floor distribution. The declaration `p` becomes
   `(1 - alpha) p + alpha t` with the smallest `alpha` that lifts every deficient cell to its
   floor: `max((f_i - p_i) / (t_i - p_i))` over the cells with `p_i < f_i`. That weight exists
   and is at most `sum(f)` (about 0.012), because `t_i > f_i > p_i` on every deficient cell.
   It is `_blend_with_uniform`'s idea (Step 5) applied to the cell floors exactly instead of to
   a uniform mixture, so it moves the least mass possible: a bin the model set to 0 ends up at
   exactly `min_step + 1e-9`, a certain member keeps 0.988 to 0.992 on its bin (0.9908 on
   651's 12-bin closed grid), and a declaration that already meets every floor is returned
   unchanged, which also makes the blend idempotent.
4. **Assemble** `cdf[0] = p'[0]`, `cdf[k] = cdf[k-1] + p'[k]`, so `cdf[N] = 1 - p'[N+1]`; pin a
   closed bound to exactly 0.0 / 1.0 (step 1 guarantees only float residue moves).
5. **`safe_cdf_bounds`** with the grid's own limits: the one implementation of the open-bound
   pins, the max-step packing and its `CDF_MAXSTEP_CLIP` marker (Step 5). At 31 bins or fewer
   the cap is 1.0, so the packing never fires today; it is there so raising the threshold past
   40 bins needs no new code. Closed bounds are re-pinned afterwards.
6. **`validate_grid_cdf`**, a fail-SHUT replica of the server rules below: length `cdf_size`, no
   NaN, every 9-decimal-rounded step within `grid_step_constraints`, closed bounds exactly
   pinned, open bounds at least 0.001 inside. Raises `RuntimeError`; nothing is repaired here.
7. **Wrap** exactly as the discrete percentile build does (`create_pchip_numeric_distribution`,
   Step 6): `declared_percentiles` is the CDF labelled by the grid edges, `cdf_size` is copied,
   `is_date` comes from the epoch adapter, `zero_point` from `resolve_zero_point`. Nothing
   downstream (publish, comment, spread, telemetry, aggregation) sees a new shape.
   `published_pmf(prediction)` reads the `N + 2` PMF back out (`[cdf[0], diff(cdf)..., 1 -
   cdf[-1]]`) for the `MEMBER_FORECAST` line's `published` field.

**What the per-bin path bypasses, and why each bypass is right.** `sanitize_percentiles` (Step
3) requires exactly the thirteen standard percentiles and there are none; nothing it does has a
per-bin analogue worth building, the floor blend is the whole repair. The PCHIP repair tiers
and `validate_cdf_construction` go with it at no loss (the latter returns immediately for any
distribution carrying `_pchip_cdf_values`). The unit-mismatch guard (Step 8) catches values
declared in the wrong units; a per-bin declaration states no values, only mass on labelled
bins, so the failure mode cannot occur, and routing the grid-shaped `declared_percentiles`
into it would make the guard fail open (all three ratios pass trivially on grid input). The
residual risk on this path is label MISREADING, which the prompt's grid sentence addresses and
no numeric guard can. The discrete vote and snap (Step 7) are already skipped on every
outcome-space grid, and the open-bound piling diagnostic takes declared percentiles; on the
per-bin path piling on an open edge IS the `above_range` value, which `oor_high` records.

`tests/test_numeric_pmf_cdf.py` runs the server oracle (`assert_server_accepts_cdf`) on the
build for six declaration shapes (one-hot on the first, middle and last bin, uniform, a
half-half split, everything in an open tail) across 3, 4, 12, 21 and 31 bins by the three
bound shapes Mantic uses, then across every one of the 32 distinct coarse grids in the
recorded 2026-09-08 corpus (`tests/data/mantic_cdf_grids_2026_09_08.json`); it pins the
floor arithmetic, the closed-tail refusal, every `validate_grid_cdf` raise path, and that the
pointwise MEAN of three sharp members (Step 9) is itself server-legal on every oracle grid.

## Step 5: PCHIP 201-point CDF

`build_numeric_distribution` (`numeric/pipeline.py`) hands the sanitized percentiles
to `generate_pchip_cdf_with_smoothing`, which calls `generate_pchip_cdf`
(`numeric/pchip_cdf.py`). This produces the 201-point CDF the bot actually submits.

The construction:

- Build a value grid from lower to upper bound: linear normally, geometric when
  `zero_point` is set (`build_cdf_value_grid` in `numeric/pchip_cdf.py`, matches the
  Metaculus backend's non-linear spacing).
- Fit a monotone PCHIP interpolator through the declared percentiles (log-space for
  strictly-positive series), evaluate it on the grid, and clamp to `[0, 1]`.
- Blend in a uniform mixture so the minimum step is satisfied before any repair tier is
  reached. This is the primary min-step mechanism, inline in `generate_pchip_cdf`
  (`numeric/pchip_cdf.py`).
- Enforce the min-step and max-step constraints, then re-pin the bounds.

### Explicit probabilities at open bounds

When a member states a valid open-side probability, the PCHIP construction adds a bound
anchor: `(lower_bound, prob_below_lower_bound)` or
`(upper_bound, 1 - prob_above_upper_bound)`. PCHIP interpolates between this anchor and
the nearest retained percentile instead of holding the outermost percentile's level
flat all the way to the displayed edge. This applies on linear and log-scaled axes and
on the question's own grid. A side without a valid statement follows the existing
percentile path.

The stated value takes precedence over contradictory percentile anchors. For a lower
bound, the builder drops an in-range percentile whose level is below the stated lower
tail, or a percentile at or below the lower bound whose level exceeds that tail. For
an upper bound, it drops an in-range percentile whose level exceeds
`1 - prob_above_upper_bound`, or one at or above the upper bound whose level is below
that cumulative probability. Equal-level ambiguities and duplicate points at the bound
are reconciled during construction to keep a monotone set of distinct anchors. This
is a CDF-construction repair: the raw percentile declaration remains available in
`MEMBER_FORECAST`.

The server's probability and step constraints still apply. The requested tail is
clamped between the open-tail minimum and the mass left after reserving every interior
bin's minimum and any open opposite tail's minimum. If two stated tails together leave
too little interior mass, their mass above the open-tail minima is reduced in proportion.
If the other side is unstated, the builder adjusts that side's implied endpoint first
when needed to fit the stated value. The applied endpoint therefore remains legal even
when the request cannot coexist with the grid constraints.
`NUMERIC_STATED_TAIL: question=... model=... bound=... stated=... status=...
dropped_percentiles=... resulting_mass=...` records the side, member, stated probability,
number of discarded percentile anchors, and resulting tail mass. Its `status` is `used`,
`missing`, `invalid` or `clamped`, so absent and invalid statements remain distinguishable
from statements the builder used. The full field contract is in
[telemetry_markers.md](telemetry_markers.md).

These fields affect the per-member percentile CDF only. Ensemble aggregation still
combines CDF heights pointwise with its existing strategy (Step 9), and the Mantic
publish-time floor still runs afterwards (Step 11). Mantic per-bin elicitation keeps its
existing `below_range` / `above_range` PMF entries and does not use the new fields.

### Server-side constraints

Metaculus validates `continuous_cdf` submissions (Mantic runs a fork of the same backend
with the same constants). The server formulas below are the upstream contract; the
constants `NUM_MIN_PROB_STEP` and `NUM_MAX_STEP` in `constants.py` mirror them at the
standard 201-point grid and are the defaults of the 201-point builders. Every grid,
including the standard one, derives its live limits from `grid_step_constraints`, below;
`numeric/config.py` keeps `MAX_CDF_PROB_STEP` as the named 201-grid cap: the reference the
open-bound piling threshold is calibrated against (`numeric/diagnostics.py`) and the
pre-scaling cap the residual analysis reads back (`performance_analysis/analysis.py`).

- **Length** = `cdf_size`, whose standard-continuous default is `PCHIP_CDF_POINTS`. On
  the platform side that is the question's `inbound_outcome_count + 1`, default 201.
- **Min step** per bin `NUM_MIN_PROB_STEP`, so no flat segments are allowed. The
  server formula is `round(0.01 / N, 9)` where `N = cdf_size - 1`, so 5e-5 at the
  default length.
- **Max step** per bin `NUM_MAX_STEP`, a spikiness cap. Server formula
  `0.2 * 200 / N`, so 0.2 at the default length. The server compares its 9-decimal-rounded
  PMF against this UNROUNDED cap; `grid_step_constraints`, below, is what keeps a clipped bin
  from rounding above it.
- **Closed bounds** are pinned exactly: `cdf[0] == 0.0`, `cdf[-1] == 1.0`.
- **Open bounds**: `cdf[0] >= 0.001`, `cdf[-1] <= 0.999`.
- **Strictly increasing**, implied by min step > 0.
- **There is no flat 0.59 cap.** Mantic's OpenAPI schema text says no two adjacent CDF values
  may differ by more than 0.59, "the largest number obtainable via the sliders"; that
  describes the slider interface, not the validator. Of 4,318 stored competitor forecasts in
  the 2026-09-08 corpus, 34 carry a single in-range bin above 0.59 (the maximum 0.8952 on a
  15-bin grid, 0.8468 by the `Mantic` account itself) and none exceed `0.2 * 200 / N`, and the
  upstream serializer has no such constant. A certain per-bin forecast is legal on every grid.

The upstream source for all of this is the open-source Metaculus backend,
<https://github.com/Metaculus/metaculus>, where the validation lives in
`questions/serializers/common.py`. The API itself is documented at
<https://www.metaculus.com/api/> (Swagger UI).

`grid_step_constraints` (`numeric/config.py`) applies those formulas to any grid as the
9-decimal values that survive the server's rounding: the min step is `round(0.01 / N, 9)`,
the server's own rounded floor, and the max step is the largest 9-decimal value not
exceeding `0.2 * 200 / N`, clamped at `1.0` (a probability step larger than that is
vacuous). The max-step floor exists because `safe_cdf_bounds` clips over-cap bins to
exactly the max step: with the raw cap, a clipped bin on the live 451-point Mantic question
rounded up to 0.088888889 and the whole submission was rejected with HTTP 400 (found in
review 2026-09, before any live run). The floor is a no-op wherever the cap is 9-decimal
exact, which covers every grid of 41 points or fewer and the 51, 101, 201 and 2,001-point
grids. There is no floor on the min step: an earlier version floored it at the 201-grid
value, which was a no-op on every Metaculus grid but 2.25x stricter than the server at 450
bins and 10x at 2,000, forcing that much extra uniform mixture into fine-grid tails. Mantic
quantitative questions run up to 2,000 bins, so that floor was removed in 2026-09
(`tests/test_numeric_fine_grids.py` pins the formulas and the 451- and 2,001-point publish
path, broad and cap-binding). On the standard continuous grid the function returns exactly
`(NUM_MIN_PROB_STEP, NUM_MAX_STEP)`, so continuous questions are unaffected; a coarse
discrete grid relaxes the max step upward, which is what lets a small-count distribution
keep its mass concentrated on the low integers.

`safe_cdf_bounds` (`numeric/pchip_cdf.py`) enforces the max-step rule by
redistributing excess mass while preserving the total, then re-enforces min-step after
the pin-and-cummax pass. `enforce_min_steps` (`numeric/pchip_cdf.py`) does a
forward-then-backward sweep to guarantee every adjacent pair is at least one min-step
apart. `_apply_ramp_smoothing` (`pchip_processing.py`) is a final tilt that adds a
tiny linear ramp when the raw CDF still has a sub-min-step bin.

**Where the clipped excess goes is our choice, and it is nearest-first.** The max-step
branch fires on any question where a forecaster declares more single-bin mass than the
platform's `0.2 * 200 / N` cap can hold, so unlike the min-step tiers it is a live path.
`_pack_excess_nearest_first` clips each over-cap bin and pours its excess into the
adjacent bins with headroom, walking outward one ring at a time and splitting the
remainder across a ring in proportion to each side's room. The retired policy handed the
excess out in proportion to every bin's *slack*, which on a fine grid where the other
bins are near-empty is a near-uniform spread: q45065 (2026-08-01) had all three
forecasters declare ~0.72 on the count that resolved, and published 47% of its mass above
35 deaths against their own ~2%, with no log line anywhere at the time. The cap itself is
the platform's and is untouched (a 0.72 single-bin mass is simply not expressible on a
201-point grid), so the honest repair is the legal shape
closest to the declaration. Every clip emits a `CDF_MAXSTEP_CLIP` WARN naming the
forecaster, the mass displaced, and how far it travelled (`scripts/telemetry/markers.py`
harvests it); it is deliberately **not** alertable, since a spike above a platform cap is
a forecaster's declaration rather than a bot defect. Its `bins_displaced` and
`max_offset_bins` fields are what make the repair's own footprint on a published forecast
a query rather than a reconstruction. The policy change is pinned by one golden in
`tests/test_discrete_snap.py`: `conc21_closed` is the only golden with over-cap bins, so it
is the only one the max-step repair touches, and it was recaptured on 2026-08-31. The old
capture spread the three spikes' 0.399 of clipped mass as a flat 0.0235 per bin across all
20 bins, including the eight the input CDF puts at zero; the recapture keeps it in the
neighbours of the bins it was declared in.

A THIRD numeric failure shape sits beside the min-step and max-step repairs and is
measured only offline: a STARVED OUTER TAIL, where the declared tail routes past the
displayed range and leaves every in-range bin above the members' p99 pinned at the
platform minimum step, so any resolution in that band earns the same floor score. It is a
cliff at a fixed location rather than a mis-sized band, which is why widening does not fix
it, and it has no publish-time WARN on purpose. Detail:
[performance_analysis.md](performance_analysis.md) and
`metaculus_bot/performance_analysis/outer_tail.py`.

`safe_cdf_bounds` holds the only implementation of this packing policy and is the single
choke point every CDF build passes through: the per-model build (`generate_pchip_cdf`), the
per-model ramp pass, the per-bin build ("Per-bin elicitation", step 5), the ensemble CDF
(`_postprocess_ensemble_cdf`), the discrete integer snap, the forecasting-tools fallback
builder on every bound shape, and the offline pooling paths. The fallback builder's
closed/closed branch used to return upstream's CDF raw, with no step or endpoint enforcement
and no `CDF_MAXSTEP_CLIP` marker possible; a tight declaration on the modal Mantic date shape
produced a 0.297 bin against the 0.2 cap and an HTTP 400 on the lone-survivor path, and the
branch was removed on 2026-09-08 (the history is in the retired design log (git history) under "Sentinel-value sweep
leftovers", sixth item).

### Open vs. closed bounds: a one-sided constraint, not a box

This is the subtle part. Bound pinning is **one-sided per tail**, not a clamp on
out-of-bound mass:

- **Closed lower bound** → `cdf[0]` is pinned to exactly `0.0`.
- **Closed upper bound** → `cdf[-1]` is pinned to exactly `1.0`.
- **Open lower bound** → `cdf[0]` is floored at a *minimum* of `0.001`. This is a
  required minimum positive mass, not a cap. The CDF can start well above 0.001.
- **Open upper bound** → `cdf[-1]` is ceilinged at a *maximum* of `0.999`. Again a
  required headroom, not a floor on how much mass sits below the ceiling.

There is no fixed cap on out-of-bound mass. A distribution can legitimately place,
say, 78% of its mass below an open lower bound. An explicit open-side probability sets
that bound's CDF anchor; without one, declared percentile *values* beyond the displayed
range determine `F(bound)` by interpolation (values are not clamped on open bounds).
The only ceiling on out-of-bound mass comes from min/max-step feasibility (roughly 0.99,
since 200 bins each
need at least a min-step). The `_pin_endpoints` helper (`numeric/utils.py`) and the
validation in `_validate_pchip_cdf` (`pchip_processing.py`) both apply this one-sided
logic.

If PCHIP construction fails outright, `create_fallback_numeric_distribution`
(`pchip_processing.py`) delegates the CDF build to forecasting-tools and runs the result
through `safe_cdf_bounds` with the grid's own step limits, whatever the bound shape: the
native builder would anchor an open lower bound at 0% once the standard set includes P1,
which Metaculus rejects, and on closed bounds it can emit an over-cap bin (the 0.297 case
above). `safe_cdf_bounds` pins nothing on a closed bound and `enforce_min_steps` caps at
[0.0, 1.0], so closed endpoints survive it exactly.

### Piling on an open edge (`OPEN_BOUND_PILING`)

A sibling WARN, `OPEN_BOUND_PILING: question=... model=... bound=... bin_mass=...
declared_edge=... bound_value=...` (`numeric/diagnostics.py`, threshold
`OPEN_BOUND_PILING_THRESHOLD` in `numeric/config.py`), fires when a model piles at least that
fraction of mass on the terminal displayed bin of an *open*-bound numeric question without
declaring any percentile beyond the edge. That is the "crammed the open ceiling" failure mode,
fixed 2026-07-12 by rendering nominal/displayed bounds in the numeric prompts
(`nominal_bounds` in `numeric/utils.py`). It takes the pre-resample model-declared percentiles
explicitly, because the discrete resample overwrites `prediction.declared_percentiles` with a
grid pinned to the raw bounds, which would defeat the above-edge exemption. The threshold is
calibrated against the 201-grid per-bin cap of 0.2 and scales down with the grid's own cap on
finer grids (0.044 at 451 points, 0.01 at 2,001), because the max-step repair clips a crammed
terminal bin to that cap, which on a fine grid is below the fixed 0.10; the 201-point and every
coarser grid keep 0.10 exactly.

### The MIN-step repair-tier signals are dead code on real forecasts

The PCHIP build logs its repair tiers as prose lines rather than markers:
`PCHIP minimum step enforcement required` (carrying `violated_steps=<k>/<n>`) and
`PCHIP aggressive enforcement completed` in `pchip_cdf.py`, `CDF ramp smoothing` (carrying
`min_prob_delta_before` and `min_prob_delta_after`) in `pchip_processing.py`, and the
end-of-run `PCHIP Summary` line whose `required_aggressive_enforcement` and `failed_entirely`
counters total them. On real model output none of the repair lines fires, because the
uniform-mixture construction pre-enforces the min-step before any repair tier is reached: the
2026-07-15 audit scored five repair signals (aggressive enforcement, two bounds-clamp
corrections, the violated-step fraction and the ramp-smoothing delta) over 1182 archived
numeric forecasts and found every one at zero, and the one genuinely degenerate case raised
out of the build (the `failed_entirely` count) instead of being repaired. Keep the guards (they
defend against pathological inputs), but their absence in logs carries no information,
and they must not be used as model-quality features. Verified 2026-07-15; receipts in
`scratch/coherence_2026-07-15/synthesis.md`.

The rebuild trigger and the rebuild's own range check carry the same `_MIN_STEP_TOLERANCE`
(1e-10) as the post-check and the final assertion (`numeric/pchip_cdf.py`). A forecast that
puts essentially all of its mass beyond an open bound leaves the in-range CDF as the bare
min-step ramp, whose required range equals its available range to within float epsilon;
untoleranced, the trigger fired on a step 1e-18 short and the range check then refused a range
1e-16 short, so the member was dropped on every non-201 grid and the 201-point grid fell
through to a forecasting-tools fallback that failed on the same input (Mantic edge-case
review 2026-09, rank 3; `TestAllMassBeyondABoundStillBuilds` in
`tests/test_numeric_fine_grids.py` pins 15, 201, 451 and 2,001 points, both bounds). With the
tolerance the rebuild can only be reached by a genuine shortfall, where it raises.

Do NOT generalize that to the MAX-step repair, which is a live path; see the
nearest-first packing section below. That the 2026-07-15 audit did not see it is an
artifact of its old DEBUG level, not evidence it never fires.

## Step 6: PchipNumericDistribution

The result is wrapped in a `PchipNumericDistribution` (`pchip_processing.py`), a
subclass of forecasting-tools' `NumericDistribution` whose `get_cdf()` override returns
the pre-computed CDF on the canonical question grid instead of rebuilding it. `get_cdf()` is the real
override: the `.cdf` property is a deprecated shim that delegates to it, so overriding
`.cdf` alone would miss the publish and aggregate paths. The `_pchip_cdf_values` attribute
also acts as the marker that CDF validation should be skipped (the constraints were
already enforced) and that discrete snapping can read the CDF back out.

## Step 7: discrete integer snapping

Some questions are labeled continuous (`cdf_size=201`) but resolve on integers ("how
many X will happen?"). A smooth CDF wastes mass between integers. If a strict majority
of forecasters voted DISCRETE (`majority_votes_discrete`), `maybe_snap_to_integers`
(`post_processing.py`) snaps the ensemble distribution to a step function
concentrated on integer values.

`snap_cdf_to_integers` (`numeric/discrete_snap.py`) extracts an integer PMF by
half-integer interpolation, rebuilds a step CDF, mixes in a uniform component for
min-step compliance, then runs `safe_cdf_bounds`. It is skipped when the range holds
more than `DISCRETE_SNAP_MAX_INTEGERS` integers (`constants.py`), when there
are no integers in bounds, when bounds are non-finite, when the question is natively
discrete (a `DiscreteQuestion`: every Metaculus discrete question and every Mantic
quantitative question), or when the grid is not the 201-point one the snap's step limits
belong to. The integer count is checked arithmetically (`floor(hi) - ceil(lo) + 1`) before
the integer array is built: a wide-range question such as post 40165 (revenue, bounds 2e10
to 8e10) holds 6e10 integers, about 480 GB as int64, and materialising them first killed a
replay in 2026-09. The type is the signal, not `cdf_size`: a 200-bin Mantic discrete question has
`cdf_size == 201` (thirteen in the Series 1 corpus, steps 1.0, 1.005 and 251.25) and the old
`cdf_size != 201` guard let them through. A natively discrete grid is already its outcome
space, so a 0.1-step grid resolves in tenths and an integer vote is simply wrong there, while
on a 1.0-step integer-centred grid the snap is a no-op (Mantic edge-case review 2026-09,
rank 14). There is no grid-alignment guard because none is needed: the step for integer k
lands in the bin containing k, and `resolution_to_bucket_index` scores a resolution of k in
that same bin on any grid, so the snap and the scorer agree by construction; the skip is
semantic, not geometric. Metaculus continuous count questions on [0, 10] or [0, 50] at 201
points, the snap's designed target, are unaffected. Snapping is decided at the ensemble
level, after aggregation.

## Step 8: unit-mismatch guard

Before a per-model prediction is accepted, `detect_unit_mismatch`
(`numeric/validation.py`) checks whether the declared values look off by orders of
magnitude relative to the question range. It flags a mismatch when any of three ratios
falls below its threshold; each threshold is a keyword argument on
`detect_unit_mismatch`, so the signature is where the values live:

- span between lowest and highest declared value, over the range
  (`span_ratio_threshold`);
- minimum adjacent step, over the range (`min_step_ratio_threshold`);
- maximum absolute value, over the range (`max_magnitude_ratio_threshold`).

On a flagged mismatch, `run_numeric_forecast` (`forecaster_runners.py`) raises
`UnitMismatchError` and withholds that forecaster's prediction rather than submitting a
distribution in the wrong units. No network or community stats are needed; it is a pure
sanity check on the numbers.

The guard **fails SHUT**: it used to wrap its arithmetic in a try/except that returned
"no mismatch" on any internal error, which is byte-identical to a passing check, so a
crash inside the guard silently published the order-of-magnitude error it exists to
block. Errors now propagate. Related: on the 201-point continuous grid a point-mass
declaration reaches this guard with its real (zero) span rather than the cluster spreader's
invented one, which is why Step 3 does not spread whole-set collapses there; where the bins
are the outcome space Step 3 spreads the collapse inside its bin and the guard passes it.

## Step 9: ensemble aggregation in CDF space

`aggregate_numeric` (`numeric/utils.py`) combines the per-model distributions
**pointwise in CDF space**, not by averaging percentiles:

1. Read each model's CDF heights in ORDER and align them POSITIONALLY: grid index `i` is
   Metaculus bucket `i/(n-1)`, so index `i` means the same thing for every model. A CDF
   that arrives on a different-length grid is resampled in cdf-LOCATION space (never
   value space: the PCHIP grid and forecasting-tools' builder compute the same value axis
   by different formulas, equal in exact arithmetic but not in the last float bits, so
   only the bucket index is shared by construction) and logs
   `NUMERIC_AGGREGATE_GRID_MISMATCH`, which should read zero in prod.
2. Take the mean or median of the cumulative probabilities at each index.

   This replaced a group-by-VALUE aggregation (a pandas groupby on float-equal `value`).
   The PCHIP grid (`np.linspace`) and the ft-fallback grid (`min + span*i/(n-1)`) differ
   in float rounding, so a mixed-path ensemble medianed over a rotating SUBSET of its
   members at misaligned points (measured at 225 unique x-values from 3 models, 48 of
   them with fewer than 3 contributors), and nothing recorded the partial membership.
3. `_postprocess_ensemble_cdf` (`numeric/utils.py`) re-pins the endpoints (one-sided
   open/closed logic), enforces monotonicity, applies ramp smoothing if any bin is below
   min-step, and runs `safe_cdf_bounds` with the step limits of the grid the CDF is on
   (`grid_step_constraints`). Nothing is resampled at this stage: step 1 already put the
   ensemble on the question's `cdf_size` grid, so a discrete question's aggregate gets
   the coarse grid's limits by construction. The result is labelled with the question's
   own value axis (`build_cdf_value_grid` on the `zero_point` that `resolve_zero_point`
   picked, the same one every member was built with), so the aggregate's
   `declared_percentiles`, its `get_cdf()` and the members all agree on where each
   probability sits. Until 2026-09 the aggregate's `declared_percentiles` were labelled
   with a linear axis even on a `zero_point` question, so the comment and the spread
   metric read a different distribution from the one published.

Percentile-space averaging would blur multi-modal disagreement; CDF-space averaging
preserves it. In production the base-combine path uses **MEDIAN** of the raw per-model
CDFs (`base_combine` in `aggregation_pipeline.py`, because the default strategy is
`CONDITIONAL_STACKING` and stacking is disabled in prod). Backtests and the mean arm use
MEAN.

**Per-bin members are pooled by the MEAN, on every path.** A member elicited per bin ("Per-bin
elicitation", above) can be sharp, 0.99 on one bin, and the pointwise
median of three sharp CDFs that disagree is the middle member's CDF outright: the published
forecast then carries 0.99 on that member's bin and the platform floor on the bins the other
two believed, the cliff a log score in the resolved bin punishes hardest. The linear opinion
pool is the pointwise MEAN of the members' CDFs (the mean of CDFs at each grid index is the CDF
of the mixture PMF), which `aggregate_numeric(..., "mean")` already computes. On question 651's
12-bin grid, three members certain of bins 3, 5 and 7 publish 0.0008 / 0.9908 / 0.0008 under
the median and 0.3308 each under the pool: expected Series 1 score `50 * ln(p / (1/12))` of
-112 against +69 if the three bins are equally likely, and +6 against +92 when two members
agree. The pool never gives a bin less than a third of the mass any member put there.
`aggregation_pipeline._numeric_combine_strategy` returns MEAN when `elicit_per_bin` holds for
the question's numeric view and the configured strategy otherwise, so percentile members keep
the MEDIAN everywhere, on both platforms (the benchmarked median-over-mean decision in
backlog decision-9 "Mean over median, always-on stacking, stacker as judge, coherence weighting" stands for them). The `NUMERIC_AGGREGATE` marker's trailing `method=` records which rule ran: `mean` (pooled
per-bin members), `median`, `stacked` or `single`; `unrecorded` means the aggregation never
recorded a method and is a bug signal.

## Step 10: the numeric spread metric

`numeric_percentile_spread` (`spread_metrics.py`) is what decides whether the
ensemble disagrees enough to trigger conditional stacking. It reads each model's P10,
P50, and P90 values (by percentile label, so growing the standard set cannot silently
shift them), takes the max-minus-min spread at each of those three percentiles, and
normalizes:

- **Closed-bound questions** → divide by the question range (`upper - lower`).
- **Open-ended questions** → divide by the ensemble interquartile range (median P90
  minus median P10), since the range is unbounded.

The largest of the three normalized spreads is the reported value. If it exceeds
`CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD` (`constants.py`), the
aggregator extracts the disagreement crux, runs a targeted search, and invokes the
stacker LLM; otherwise it returns the MEDIAN. (Stacking is disabled in all six bot
workflows, so in prod this metric is computed but the stacker branch does not fire; the
chain stays live in backtests and ablation.)

For discrete questions, the per-model `declared_percentiles` have already been
overwritten with a resampled CDF grid whose labels are cumulative probabilities, so
`_key_percentile_values` (`spread_metrics.py`) reads P10/P50/P90 by interpolating the
empirical CDF instead of looking up label nodes.

## Step 11: the Mantic out-of-range tail floor

`floor_published_tails` (`numeric/out_of_range_floor.py`) is the last touch on a numeric,
discrete or date aggregate before it is published, applied in
`TemplateForecaster._aggregate_predictions` (`forecaster.py`), the one seam every aggregation
path (median, mean, base-combine, stacker, single survivor) returns through, so the publish
gate, the comment and the `NUMERIC_AGGREGATE` marker all read the floored distribution.

**The rule.** On a question whose platform is Mantic (`question_platform` reads it off
`page_url`; `PLATFORM_MANTIC` in `constants.py`), the published CDF carries at least
`MANTIC_OUT_OF_RANGE_TAIL_FLOOR` (`constants.py`, 0.05) beyond each OPEN bound, as far as the
other tail leaves room for: an open tail under the floor is raised to
`min(0.05, 1 − the other tail − N × min step)`, where `N × min step` (`round(0.01 / N, 9)` per
bin) is the least mass the server lets the interior keep, so beside a thin tail it goes to exactly
the floor (`cdf[0] = 0.05`, or `cdf[-1] = 0.95`), beside a fat one it rises less (behind a 98%
lower tail on the 201-point grid the upper tail goes from 0.1% to 1%), and an interior already at
its minimum moves nothing; an open tail already at or above the floor is left alone (a tail is
never reduced), and a closed bound stays at its exact `0.0` / `1.0`. Both sides may move at once.
Without the cap a both-open aggregate with 98% below the lower bound came back with
`cdf[-1] = 0.95 < cdf[0]` and the rebuild raised at the seam, forfeiting the question (codex
review, 2026-09-08). The interior is rescaled affinely between the new endpoints, which
keeps it monotone and can only shrink steps, so the platform's max step cannot be newly
violated; the bins that sat exactly on the min step (the uniform-mixture tails of a
concentrated PCHIP build) do land below it, so `enforce_min_steps` (`numeric/pchip_cdf.py`)
runs again with the new endpoints as its caps. A lift under `_LIFT_TOLERANCE` (1e-10, the
server's 10-decimal CDF rounding, so invisible to it) is not a move: an aggregate whose interior
already sits at the minimum arrives with its endpoints a few 1e-15 off it, and without the
tolerance the seam rebuilt the distribution for that noise and the marker's `tail_floor` read
the tail's own raw mass as the floor applied. When an endpoint moves, the distribution is
rebuilt through `create_pchip_numeric_distribution` on the question's numeric view (the epoch
adapter for a date question, so `is_date` and the date rendering survive) with the floored
heights as both its CDF and its `declared_percentiles`, on the value axis the aggregate already
carried. `tests/test_out_of_range_floor.py` checks the server's `continuous_cdf` rules on the
result for every distinct grid in the recorded 2026-09-08 Mantic corpus
(`tests/data/mantic_cdf_grids_2026_09_08.json`: 88 grids, 3 to 450 bins, every bound
combination), on fourteen aggregate shapes each: six ordinary ones and eight with 94%, 95%, 98%
or 99.9% of the mass already beyond one bound.

**Why.** Mantic scores an out-of-range resolution against a fixed 0.05 reference,
`50 * ln(mass / 0.05)`: 5% there scores 0 and the structural 1% from the legacy fallback when every
declared percentile sits inside the range scores -80.5. Mantic's writers set the ranges and are
paid for bot disagreement; in Series 1 half the date questions, a quarter of the discrete and an
eighth of the numeric ones resolved outside the displayed range. Applying the same floor to every
Series 1 competitor's own 4,082 published distributions cost at most 1.9 points per question on
average for any type and gained up to 9.1 for thin-tailed bots
(`scratch_docs_and_planning/mantic_adversarial_candidates_2026-09-08.md`, section 1).
Operator-approved 2026-09-08.

**The platform gate.** Metaculus gets its own, lower floor, `METACULUS_OUT_OF_RANGE_TAIL_FLOOR`
(2%, since 2026-11-24; reasoning and receipts in docs/constants.md), through the same code; an aggregate
that needs no lift comes back as the very same object. Per-member forecasts are never touched (their `MEMBER_FORECAST` tails keep measuring what the
models declared); the backtest, ablation and benchmark harnesses replay Metaculus questions, so
the gate covers them without a flag.

**Telemetry.** `NUMERIC_AGGREGATE` keeps `oor_low` / `oor_high` as the PUBLISHED tails and adds
three trailing fields: `oor_low_raw` / `oor_high_raw`, the aggregate's own tails before the
floor, and `tail_floor`, the level the moved tails were actually raised to: `0.050000` normally,
less where the other tail capped the raise, and `0.000000` when nothing moved (Metaculus, a
closed-bound question, tails already at or above the floor, an interior already at its minimum).
Raw against published is how the floor gets benchmarked on this bot's own forecasts once live
Mantic telemetry accumulates.

## The time-series anchor provider

`research/timeseries_anchor.py` is a research provider (not part of the CDF pipeline)
that grounds numeric forecasts whose resolution series is a fetchable FRED or yfinance
series. It renders a deterministic empirical anchor with no LLM: the latest value, a
multi-resolution history, a 52-week range, and a horizon-matched empirical band. Its
section header in the briefing is `## Time Series Anchor` (`TS_ANCHOR_SECTION_HEADER`,
rendered by `provider_header` in `research/section_format.py`). Gated by
`TS_ANCHOR_ENABLED` (`_select_research_providers` in `research/orchestrator.py`); on in
all six workflows.

### Routing (deterministic, no LLM)

`route_question` (`timeseries_anchor.py`) maps a question to a series two ways, URL
first:

1. **URL extraction** from resolution criteria and fine print: a cited FRED series or
   Yahoo ticker is the ground-truth resolving source and wins.
2. **A conservative curated keyword registry** (`_TEMPLATE_REGISTRY` in
   `timeseries_anchor.py`): 10-year Treasury, VIX, CPI, unemployment, nonfarm
   payrolls, S&P 500, gold, and so on. Deliberately small and unambiguous.

Anything ambiguous (more than one series that is not a two-ticker spread, or more than
one keyword match) returns `""` and logs. Two Yahoo tickers become a relative-return
spread block.

### The empirical band

The band is the naive, model-free choice, deliberately. The Phase-A offline replay
(`scratch/ts_anchor_replay_2026-07-16/synthesis.md`) found that CV-gated statistical
model picks beat the naive out-of-sample only 43% of the time, while the naive empirical
h-step-change band was both sharper and better tail-calibrated. So the provider just
computes empirical quantiles (P10 / P50 / P90) of every overlapping h-step change in the
series' own history and applies them to the latest value (`_empirical_change_band` in
`timeseries_anchor.py`). Log-multiplicative for strictly-positive series, additive
otherwise. "Highest / peak / maximum" questions use a forward-window-max band instead
(`_empirical_max_band`). The horizon `h` is matched to the question's actual
forecast window, converted to native series steps by frequency (`horizon_steps`).

Derived-quantity questions (month-over-month change, MoM % inflation, monthly averages)
fit the band on the derived series, not the raw level (`_apply_derivation` in
`timeseries_anchor.py`).

### Point-in-time leakage safety

This is the only backtest-safe research provider that runs during benchmarks. Others
hard-disable under `is_benchmarking`; this one instead pins `as_of` to
`question.open_time` in benchmarks (live: `datetime.now(UTC)`) and fetches the series
point-in-time up to that date, so data known at forecast time is fair game without
leaking the resolution (`timeseries_anchor_provider` in `timeseries_anchor.py`).

The fetch layer (`research/ts_fetch.py`) enforces the invariant. Revising macro series
(CPI, payrolls, GDP) would leak if fetched from today's FRED, because today's data
contains revised historical values not known at forecast time. So those go through
**ALFRED point-in-time vintages** instead of plain FRED CSV. `fetch_series`
(`ts_fetch.py`) defaults every FRED series to ALFRED vintages *except* a small
non-revising allowlist (`FRED_NON_REVISING_SERIES` in `ts_fetch.py`: market prices and
survey levels like DGS10, Brent, gasoline). That default is fail-safe: an over-inclusive
ALFRED guess costs nothing for a non-revising series, but a revising series routed to
plain FRED would silently leak. A belt-and-suspenders check, `_assert_no_leakage`
(`ts_fetch.py`), raises `LeakageError` if any observation postdates the ceiling.

### Text anchor on, chart off

The provider always returns a text section. It can also render a small chart image
(matplotlib ribbon), stashed in a per-session side-channel for the forecaster's vision
message, but only for plain single-level questions. The chart is gated separately by
`TS_ANCHOR_CHART_ENABLED` (`_maybe_stash_single_chart` in `timeseries_anchor.py`), which
is **off** in all six workflows while the text anchor is **on**. Chart render failures
are swallowed so a plotting hiccup never breaks the text section.

## Clamps on the other question types

The numeric CDF bounds above are one of three publish-value clamps. The binary and MC
clamps live here too so that all three have one home.

- **Binary**: `[BINARY_PROB_MIN, BINARY_PROB_MAX]` (`constants.py`). Applied per-model in `forecaster_runners.py` and on stacker output in `stacking.py`. Median/mean of already-clamped values stays in-bounds, so no post-aggregation clip needed.
- **MC**: `[MC_PROB_MIN, MC_PROB_MAX]` (`constants.py`), set to match ft 0.2.92's `PredictedOptionList` validator, which clamps every option on construction: matching bounds makes it a no-op and removes publish-time `ValueError` risk on many-option ballots. Drift-free clamp-then-renormalize via `clamp_and_renormalize_probs` (`mc_processing.py`), applied BEFORE every `PredictedOptionList` construction and re-applied idempotently by `clamp_and_renormalize_mc` (`numeric/utils.py`); the repair pass keeps floored options from dividing back below the floor.
