"""``pmf_prompt``: the continuous template elicited per bin on an enumerable Mantic grid.

On a coarse grid the bins are the outcome space, so each forecaster is asked for one probability
per named bin (plus ``below_range`` / ``above_range`` where a bound is open) instead of thirteen
percentiles. The prompt is the SAME continuous template the numeric and date prompts render, with
every line that differs between the two elicitations carried by an ``_Elicitation`` slot: the
percentile fill reproduces today's Metaculus text byte for byte (``TestPercentileFillIsTodaysText``
is the permanent pin on that), and the per-bin fill is percentile-free. Every per-bin rule is a
named constant with a presence pin here and an absence pin on the percentile and stacking prompts.
"""

from __future__ import annotations

import json

import pytest

from metaculus_bot.constants import PMF_ABOVE_RANGE_KEY, PMF_BELOW_RANGE_KEY
from metaculus_bot.numeric.config import grid_step_constraints
from metaculus_bot.numeric.date_axis import as_epoch_question
from metaculus_bot.numeric.pmf_grid import pmf_grid
from metaculus_bot.prompts import (
    _CONTINUOUS_SCORING_RULE,
    _MANTIC_OUT_OF_RANGE_RATE_DATE,
    _MANTIC_OUT_OF_RANGE_RATE_DATE_PMF,
    _MANTIC_OUT_OF_RANGE_RATE_QUANTITY,
    _MANTIC_OUT_OF_RANGE_RATE_QUANTITY_PMF,
    _MULTI_RESOLUTION_CONTINUOUS_RULE,
    _MULTI_RESOLUTION_PMF_RULE,
    _PER_BIN_OUTPUT_RULE,
    _PER_BIN_SCORING_RULE,
    _PMF_CONSISTENCY_LINE,
    _SOFT_CLOCK_RULE,
    numeric_prompt,
    pmf_prompt,
)
from metaculus_bot.structured_output_schema import PmfStructured, parse_structured_payload
from tests.pipeline_test_helpers import make_real_date_question
from tests.prompt_builders import (
    _RESEARCH_WITH_MARKETS,
    MANTIC_PAGE_URL,
    METACULUS_PAGE_URL,
    _date_prompt_text,
    _date_q,
    _extract_last_json_block,
    _flat,
    _numeric_prompt_text,
    _numeric_q,
    _pmf_prompt_text,
    _pmf_q,
    _stacked_prompt_texts,
)

# Series 1 post 560's shape: nominal bounds ON the range edges, both open, so the bins are labelled as intervals.
_INTERVAL_GRID = {
    "cdf_size": 31,
    "lower_bound": 100.0,
    "upper_bound": 6100.0,
    "nominal_lower": 100.0,
    "nominal_upper": 6100.0,
    "open_lower": True,
}
# Series 1 post 253's shape: counts 0, 1, 2 with both bounds closed, so no reserved key exists.
_CLOSED_COUNT_GRID = {
    "cdf_size": 4,
    "lower_bound": -0.5,
    "upper_bound": 2.5,
    "nominal_lower": 0.0,
    "nominal_upper": 2.0,
    "open_upper": False,
}


def _mantic_date_q(**overrides):
    """Post 651's twelve-day grid as a Mantic question (the recorded loader renders a Metaculus URL)."""
    q = _date_q()
    q.page_url = MANTIC_PAGE_URL
    for name, value in overrides.items():
        setattr(q, name, value)
    return q


def _per_bin_output_rule(cdf_size: int) -> str:
    min_step, _ = grid_step_constraints(cdf_size)
    return _PER_BIN_OUTPUT_RULE.format(min_step=f"{min_step:g}")


_PER_BIN_CONSTANTS = (
    _PER_BIN_SCORING_RULE,
    _PMF_CONSISTENCY_LINE,
    _MANTIC_OUT_OF_RANGE_RATE_DATE_PMF,
    _MANTIC_OUT_OF_RANGE_RATE_QUANTITY_PMF,
    _MULTI_RESOLUTION_PMF_RULE,
)


class TestPercentileFillIsTodaysText:
    """The operator constraint behind the ``_Elicitation`` split: the Metaculus-rendered numeric and
    date prompts are byte-identical to the text before per-bin elicitation existed. Each pin is the
    exact rendered line of one slot's percentile fill, line breaks included, so a reworded fill or a
    slot moved onto one line fails here before it reaches a Metaculus question."""

    @pytest.mark.parametrize(
        "build", [pytest.param(_numeric_prompt_text, id="numeric"), pytest.param(_date_prompt_text, id="date")]
    )
    def test_every_slots_percentile_fill_renders_todays_line(self, build) -> None:
        prompt = build()
        assert (
            "Accuracy **and** calibration\n"
            "(especially the width of your prediction interval) are critical; how to set that width\n"
            "is step (8) of the template below."
        ) in prompt
        assert "- Time to resolution; describe how halving or doubling the timeline might shift percentiles." in prompt
        assert "- Small delta check: would +/- 10 percent on key percentiles still fit the reasoning?" in prompt
        assert (
            "- Anchor on your math: if you derived a central estimate or range from data (extrapolation, historical "
            "trend, explicit formula), your percentiles should stay close to it. Adjust only with specific evidence, "
            "not vibe."
        ) in prompt
        assert (
            "- Match your interval width to what your reasoning actually supports, and do not pad or sharpen out of a "
            "generic disposition. Log score punishes a narrow interval that misses far more than a wide one that "
            "covers, but a wide interval on a predictable quantity also bleeds points."
        ) in prompt
        assert (
            "- Keep your extreme tails (P1 and P99) wide enough to cover unknown unknowns you can actually name — but "
            "not padded out of generic caution."
        ) in prompt
        assert (
            "- Consistency line: which percentile corresponds to the status quo or trend, and is that sensible?"
            in prompt
        )
        assert f"── Scoring Rule ──\n{_CONTINUOUS_SCORING_RULE}\n" in prompt

    def test_the_market_clause_keeps_its_percentile_anchor(self) -> None:
        prompt = numeric_prompt(
            _numeric_q(), research=_RESEARCH_WITH_MARKETS, lower_bound_message="lbm", upper_bound_message="ubm"
        )
        assert "your percentiles should center on it" in prompt

    def test_the_multi_resolution_rule_still_reports_the_mixtures_percentiles(self) -> None:
        assert _MULTI_RESOLUTION_CONTINUOUS_RULE.endswith(
            "then pool those forecasts into a single mixture and report the mixture's percentiles (a distribution "
            "fitted to one instance is scored as though it had excluded every other value in the set)."
        )


class TestPerBinRulesArePresent:
    """Every per-bin constant renders once, in its section, with its load-bearing clauses intact."""

    def test_the_scoring_rule_replaces_the_density_rule_in_the_scoring_section(self) -> None:
        prompt = _pmf_prompt_text()
        flat = _flat(prompt)
        assert flat.count(_flat(_PER_BIN_SCORING_RULE)) == 1
        assert "scored on the bin the outcome falls in" in flat
        assert "probability on a bin the resolution criteria exclude" in flat
        assert "is simply lost, so give such a bin 0" in flat
        assert "this is a proper scoring rule" in flat
        assert _flat(_CONTINUOUS_SCORING_RULE) not in flat
        assert "log density score" not in flat
        assert (
            prompt.index("── Scoring Rule")
            < prompt.index("scored on the bin the outcome falls in")
            < prompt.index("── Intelligence Briefing")
        )

    def test_the_output_rule_names_every_key_and_the_grids_own_floor(self) -> None:
        """The parser maps keys onto the grid and every key must be present; the interpolated floor is the
        server's per-bin minimum for THIS grid (0.01 / 21 on post 643's shape), so the sentence cannot be
        read as the 5% tail floor the published Mantic aggregate carries. The mechanism sentence says what
        ``numeric.pmf_cdf._blend_to_cell_floors`` does (a 0 bin is lifted to the floor, the rest scaled
        down); "added to every bin" would invite the model to pre-subtract the floor it is told not to."""
        prompt = _pmf_prompt_text()
        flat = _flat(prompt)
        assert flat.count(_flat(_per_bin_output_rule(22))) == 1
        assert "one probability for every key listed in the schema below" in flat
        assert "spelled exactly as listed and in that order" in flat
        assert "use 0 for a bin you are certain cannot occur" in flat
        assert "a bin you leave at 0 is lifted to the platform's per-bin minimum" in flat
        assert "about 0.00047619 on this grid" in flat
        assert "the rest scaled down to keep the total at 1.0" in flat
        assert "added to every bin" not in flat
        assert "0.05" not in flat[flat.index("bins & bounds") : flat.index("── scoring rule")]
        assert (
            prompt.index("── Bins & Bounds")
            < prompt.index("one probability for EVERY key")
            < prompt.index("── Scoring Rule")
        )

    def test_the_floor_follows_the_grid(self) -> None:
        flat = _flat(_pmf_prompt_text(_pmf_q(**_CLOSED_COUNT_GRID)))
        assert "about 0.00333333 on this grid" in flat

    def test_the_consistency_line_asks_which_bin_holds_the_status_quo(self) -> None:
        prompt = _pmf_prompt_text()
        flat = _flat(prompt)
        final_checks = flat[flat.index("final checks") :]
        assert _flat(_PMF_CONSISTENCY_LINE) in final_checks
        assert "which bin holds the status quo or trend value" in final_checks
        assert "how much probability did you give it" in final_checks
        assert "which percentile corresponds" not in flat

    def test_the_preamble_and_template_steps_carry_the_per_bin_fills(self) -> None:
        prompt = _pmf_prompt_text()
        flat = _flat(prompt)
        assert (
            "accuracy **and** calibration (especially how far your probability spreads across the bins) are critical; "
            "how to set that spread is step (8) of the template below"
        ) in flat
        assert "describe how halving or doubling the timeline might move probability between bins" in flat
        assert (
            "small delta check: would moving ten points of probability to a neighbouring bin still fit the reasoning?"
            in flat
        )
        assert (
            "your probability should stay concentrated around it. adjust only with specific evidence, not vibe" in flat
        )
        assert (
            "match the spread of your probability across the bins to what your reasoning actually supports, and do not "
            "pad or sharpen out of a generic disposition. log score punishes a concentrated forecast that misses far "
            "more than a spread-out one that covers, but a spread-out forecast on a predictable quantity also bleeds "
            "points"
        ) in flat
        assert (
            "keep enough probability on the outer bins (and on `below_range` / `above_range` where they exist) to "
            "cover unknown unknowns you can actually name — but not padded out of generic caution"
        ) in flat

    def test_the_market_clause_anchors_probability_not_percentiles(self) -> None:
        flat = _flat(_pmf_prompt_text(research=_RESEARCH_WITH_MARKETS))
        assert "your probability should center on it" in flat
        assert "percentile" not in flat

    def test_no_outcome_type_step_so_final_checks_are_step_nine(self) -> None:
        prompt = _pmf_prompt_text()
        assert "outcome_type" not in prompt
        assert "(9) Final checks" in prompt
        assert "(10)" not in prompt
        assert prompt.rstrip().endswith("Write nothing after it.")


class TestPerBinAbsences:
    """No percentile vocabulary reaches the per-bin prompt, and no per-bin constant reaches any other prompt."""

    @pytest.mark.parametrize("research", ["r", _RESEARCH_WITH_MARKETS])
    def test_the_per_bin_prompt_never_says_percentile(self, research: str) -> None:
        for q in (_pmf_q(), _mantic_date_q(), _pmf_q(**_INTERVAL_GRID)):
            flat = _flat(_pmf_prompt_text(q, research=research))
            assert "percentile" not in flat
            assert "declared_percentiles" not in flat
            assert "p50" not in flat
            assert "interval width" not in flat

    def test_the_per_bin_prompt_drops_the_percentile_unit_and_axis_sentences(self) -> None:
        flat = _flat(_pmf_prompt_text())
        assert "units & bounds" not in flat
        assert "dates & bounds" not in flat
        assert "base units for output values" not in flat
        assert "350b" not in flat
        assert "suggestive of units" not in flat
        assert "must be numeric values in the base unit" not in flat
        assert "a percentile's value selects the bin it falls in" not in flat

    def test_no_per_bin_constant_reaches_the_percentile_or_stacking_prompts(self) -> None:
        for prompt in (_numeric_prompt_text(), _date_prompt_text(), *_stacked_prompt_texts()):
            flat = _flat(prompt)
            for constant in _PER_BIN_CONSTANTS:
                assert _flat(constant) not in flat
            assert "bin_probs" not in flat
            assert "bins & bounds" not in flat
            assert "one probability for every key" not in flat
            assert '"question_type": "pmf"' not in flat


class TestBinsAndBoundsBlock:
    """The axis block names the grid from the labelled bins themselves, never from the bound message
    (on a centre-style grid the last label need not equal the displayed maximum, post 650), and
    describes the keys the way the model must read them, one key sentence per label style: the value
    at the centre of a bin on a centre-labelled grid, a UTC calendar day or week on a date grid, a
    right-closed interval where the labels are intervals (a numeric grid with edge-aligned nominal
    bounds, a log-spaced grid, a date grid with edges at arbitrary times)."""

    def test_a_count_grid_names_its_unit_width_and_bin_count(self) -> None:
        flat = _flat(_pmf_prompt_text())
        block = flat[flat.index("── bins & bounds ──") : flat.index("── scoring rule ──")]
        assert "base unit of the bin labels: releases" in block
        assert "scoring grid: 21 bins of width 1 releases" in block
        assert "every key is the value at the centre of its bin" in block
        assert "the bin covers half the stated width either side of that value" in block
        assert block.index("bins of width 1 releases") < block.index("every key is the value at the centre")
        assert "right-closed interval" not in block
        assert "utc calendar date" not in block

    def test_a_centre_grid_with_a_wide_step_says_what_a_key_covers(self) -> None:
        """Five of the 46 coarse Mantic centre grids step by something other than 1 (5, 500, 0.25, 0.1), where
        a key ``5`` read as "5 up to 10" puts a belief of 3 one bin off: the centre sentence is unconditional."""
        q = _pmf_q(cdf_size=11, lower_bound=-2.5, upper_bound=47.5, nominal_lower=0.0, nominal_upper=45.0)
        prompt = _pmf_prompt_text(q)
        flat = _flat(prompt)
        block = flat[flat.index("── bins & bounds ──") : flat.index("── scoring rule ──")]
        assert "scoring grid: 10 bins of width 5 releases" in block
        assert "every key is the value at the centre of its bin" in block
        assert list(json.loads(_extract_last_json_block(prompt))["bin_probs"])[:3] == ["0", "5", "10"]

    def test_a_log_spaced_grid_names_the_geometry_and_the_interval_convention(self) -> None:
        """A ``zero_point`` makes the bins geometric, so the additive width phrase would be false across the
        axis and the labels are intervals (``_label_bins`` never centre-labels a log grid)."""
        q = _pmf_q(
            cdf_size=11,
            lower_bound=1.0,
            upper_bound=1024.0,
            nominal_lower=1.0,
            nominal_upper=1024.0,
            zero_point=0.0,
        )
        flat = _flat(_pmf_prompt_text(q))
        block = flat[flat.index("── bins & bounds ──") : flat.index("── scoring rule ──")]
        assert "scoring grid: 10 bins, log-spaced" in block
        assert "bins of width" not in block
        assert "each key `a to b` is a right-closed interval" in block
        assert "every key is the value at the centre" not in block

    def test_a_day_grid_describes_the_keys_as_utc_calendar_days(self) -> None:
        flat = _flat(_pmf_prompt_text(_mantic_date_q()))
        block = flat[flat.index("── bins & bounds ──") : flat.index("── scoring rule ──")]
        assert "every key is a utc calendar date and names the whole day it covers" in block
        assert "scoring grid: 12 bins of one calendar day each" in block
        assert "base unit" not in block
        assert "right-closed interval" not in block

    def test_a_week_grid_says_a_key_names_seven_days(self) -> None:
        q = make_real_date_question(cdf_size=5, date_granularity="week", open_upper_bound=True)
        flat = _flat(_pmf_prompt_text(q))
        assert "every key is a utc calendar date and names the seven days beginning on it" in flat
        assert "scoring grid: 4 bins of one calendar week each" in flat

    def test_a_date_grid_with_arbitrary_edges_states_the_right_closed_convention(self) -> None:
        """The platform sends an empty granularity on 50 of 51 live date questions; on a coarse one the
        edges fall at arbitrary times, so each key is a timestamp interval and no day-or-week sentence fits."""
        q = make_real_date_question(cdf_size=6, date_granularity="", open_upper_bound=True)
        flat = _flat(_pmf_prompt_text(q))
        block = flat[flat.index("── bins & bounds ──") : flat.index("── scoring rule ──")]
        assert "each key `a to b` is a right-closed interval" in block
        assert "scoring grid: 5 bins." in block
        assert "calendar" not in block
        assert "base unit" not in block

    def test_an_interval_grid_states_the_right_closed_convention(self) -> None:
        prompt = _pmf_prompt_text(_pmf_q(**_INTERVAL_GRID))
        flat = _flat(prompt)
        block = flat[flat.index("── bins & bounds ──") : flat.index("── scoring rule ──")]
        assert (
            "each key `a to b` is a right-closed interval: it contains everything above a up to and including b, and "
            "the first bin also contains its own lower edge"
        ) in block
        assert "scoring grid: 30 bins of width 200 releases" in block
        assert '"100 to 300"' in prompt

    def test_the_bin_list_comes_from_the_grid_labels_not_the_displayed_maximum(self) -> None:
        """Post 650's off-by-one on a coarse count grid: the writer declared ``nominal_max`` one step above the
        last bin's centre, so a list rendered from the bound message would name a bin that does not exist."""
        q = _pmf_q(cdf_size=21, upper_bound=19.5, nominal_upper=20.0)
        parsed = json.loads(_extract_last_json_block(_pmf_prompt_text(q)))
        keys = list(parsed["bin_probs"])
        assert keys == [*(str(k) for k in range(20)), PMF_ABOVE_RANGE_KEY]
        assert "20" not in keys


class TestPerBinSchemaBlock:
    def test_the_example_keys_are_the_grids_keys_in_order_and_sum_to_one(self) -> None:
        q = _pmf_q()
        grid = pmf_grid(q)
        body = _extract_last_json_block(_pmf_prompt_text(q))
        parsed = json.loads(body)
        assert parsed["question_type"] == "pmf"
        assert set(parsed) == {"question_type", "bin_probs"}
        assert tuple(parsed["bin_probs"]) == grid.keys == (*(str(k) for k in range(21)), PMF_ABOVE_RANGE_KEY)
        probs = list(parsed["bin_probs"].values())
        assert sum(probs) == pytest.approx(1.0, abs=0.002)
        assert all(0.0 < p < 1.0 for p in probs)

    def test_reserved_keys_appear_exactly_where_a_bound_is_open(self) -> None:
        both_open = json.loads(_extract_last_json_block(_pmf_prompt_text(_pmf_q(**_INTERVAL_GRID))))["bin_probs"]
        assert next(iter(both_open)) == PMF_BELOW_RANGE_KEY
        assert list(both_open)[-1] == PMF_ABOVE_RANGE_KEY
        closed = json.loads(_extract_last_json_block(_pmf_prompt_text(_pmf_q(**_CLOSED_COUNT_GRID))))["bin_probs"]
        assert list(closed) == ["0", "1", "2"]
        date = json.loads(_extract_last_json_block(_pmf_prompt_text(_mantic_date_q())))["bin_probs"]
        assert (
            tuple(date)
            == pmf_grid(as_epoch_question(_date_q())).keys
            == tuple(f"2026-09-{d:02d}" for d in range(8, 20))
        )

    def test_the_schema_header_names_the_object_and_the_exact_spelling_once(self) -> None:
        flat = _flat(_pmf_prompt_text())
        schema = flat[flat.rfind("structured forecast") :]
        assert "`bin_probs` is required and must contain every key below, spelled exactly and in this order" in schema
        assert "the values shown are format placeholders, not a suggested distribution" in schema
        assert schema.count("must contain") == 1
        assert "outcome_type" not in schema
        assert "strictly increasing" not in schema

    def test_the_example_block_parses_through_the_block_rungs_own_schema(self) -> None:
        """``PmfStructured`` is the schema the ladder's first rung applies; the example the prompt teaches
        must pass it, exactly as the date example must pass ``DateStructured``."""
        body = _extract_last_json_block(_pmf_prompt_text())
        block = parse_structured_payload(body, "pmf")
        assert isinstance(block, PmfStructured)
        assert tuple(block.bin_probs) == pmf_grid(_pmf_q()).keys


class TestManticPerBinBaseRates:
    """The measured Series 1 out-of-range rates, worded for the reserved keys, through the SAME gate as the
    percentile twins: Mantic only, the date sentence only when the upper bound is open, the quantity
    sentence when either bound is open. The percentile twins never render here (they instruct a
    percentile placement), and neither PMF sentence renders on a Metaculus question or a closed grid."""

    def test_a_mantic_count_grid_open_above_carries_the_quantity_rate_after_the_bound_messages(self) -> None:
        prompt = _pmf_prompt_text()
        flat = _flat(prompt)
        assert flat.count(_flat(_MANTIC_OUT_OF_RANGE_RATE_QUANTITY_PMF)) == 1
        assert "one in five past quantitative questions resolved outside the displayed range" in flat
        assert "a token probability there asserts a near-zero chance of an out-of-range outcome" in flat
        assert _flat(_MANTIC_OUT_OF_RANGE_RATE_QUANTITY) not in flat
        assert _flat(_MANTIC_OUT_OF_RANGE_RATE_DATE_PMF) not in flat
        assert prompt.index("is the top of the displayed range") < prompt.index(
            "one in five past quantitative questions"
        )

    def test_a_mantic_grid_open_below_only_carries_the_quantity_rate(self) -> None:
        flat = _flat(_pmf_prompt_text(_pmf_q(open_lower=True, open_upper=False)))
        assert _flat(_MANTIC_OUT_OF_RANGE_RATE_QUANTITY_PMF) in flat

    def test_a_closed_mantic_count_grid_carries_nothing(self) -> None:
        flat = _flat(_pmf_prompt_text(_pmf_q(**_CLOSED_COUNT_GRID)))
        assert "on this platform" not in flat
        assert "out-of-range outcome" not in flat

    def test_a_mantic_date_grid_open_above_carries_the_date_rate(self) -> None:
        q = make_real_date_question(open_upper_bound=True)
        flat = _flat(_pmf_prompt_text(q))
        assert flat.count(_flat(_MANTIC_OUT_OF_RANGE_RATE_DATE_PMF)) == 1
        assert "about half of past date questions with an open upper bound resolved after it (101 of 188" in flat
        assert "most of your probability belongs on `above_range`" in flat
        assert _flat(_MANTIC_OUT_OF_RANGE_RATE_DATE) not in flat
        assert _flat(_MANTIC_OUT_OF_RANGE_RATE_QUANTITY_PMF) not in flat

    def test_a_closed_mantic_date_grid_carries_nothing(self) -> None:
        assert "on this platform" not in _flat(_pmf_prompt_text(_mantic_date_q()))

    def test_a_metaculus_grid_carries_neither_sentence(self) -> None:
        flat = _flat(_pmf_prompt_text(_pmf_q(page_url=METACULUS_PAGE_URL)))
        assert "on this platform" not in flat
        assert _flat(_MANTIC_OUT_OF_RANGE_RATE_QUANTITY_PMF) not in flat
        assert _flat(_MANTIC_OUT_OF_RANGE_RATE_DATE_PMF) not in flat


class TestMultiResolutionPerBin:
    def test_a_multi_resolution_grid_reports_the_mixture_bin_by_bin(self) -> None:
        q = _pmf_q()
        q.api_json = {"question": {"multi_resolution": True}}
        prompt = _pmf_prompt_text(q)
        flat = _flat(prompt)
        assert flat.count(_flat(_MULTI_RESOLUTION_PMF_RULE)) == 1
        assert "report that mixture as your per-bin probabilities" in flat
        assert _flat(_MULTI_RESOLUTION_CONTINUOUS_RULE) not in flat
        assert (
            prompt.index("── Scoring Rule")
            < prompt.index("scored against EVERY resolution value")
            < prompt.index("── Intelligence Briefing")
        )

    def test_the_two_fills_differ_only_in_how_the_mixture_is_reported(self) -> None:
        percentile = _MULTI_RESOLUTION_CONTINUOUS_RULE.replace("report the mixture's percentiles", "")
        per_bin = _MULTI_RESOLUTION_PMF_RULE.replace("report that mixture as your per-bin probabilities", "")
        assert percentile == per_bin

    def test_without_the_flag_nothing_renders(self) -> None:
        assert "scored against every resolution" not in _flat(_pmf_prompt_text())


class TestSharedRulesReachThePerBinPrompt:
    """Every rule the numeric prompt states through the kind slots or the shared template lands once."""

    _SHARED = (
        "status-quo derivation",
        "open and unresolved as of",
        "name the specific qualifying event",
        "no qualifying event has yet occurred inside the window",
        "resolution-metric echo (named-series questions only)",
        "proximity to the primary record",
        "[unverified attribution]",
        "read a null search result as a null search result",
        "how many events of a kind occur in a period",
        "forecastability and width",
        "anchor on your math",
        "small delta check",
        "bait-and-switch check",
        "this block is the only authoritative source of your forecast",
        "what value would it resolve at",
        "coherent pathway for unusually low results",
        "decide how forecastable this quantity is from current information on this horizon",
    )

    def test_every_shared_rule_lands_once_on_a_quantity_grid(self) -> None:
        per_bin = _flat(_pmf_prompt_text())
        numeric = _flat(_numeric_prompt_text())
        for phrase in self._SHARED:
            assert per_bin.count(phrase) == numeric.count(phrase) >= 1, phrase

    def test_the_date_kind_slots_and_the_soft_clock_rule_reach_a_date_grid(self) -> None:
        flat = _flat(_pmf_prompt_text(_mantic_date_q()))
        assert "on what date would it resolve?" in flat
        assert (
            "coherent pathway for an unusually late date, including, where the upper bound is open, a date beyond the displayed window"
            in flat
        )
        assert "decide how forecastable this date is from current information" in flat
        assert flat.count(_flat(_SOFT_CLOCK_RULE)) == 1
        assert "measured record of meeting" not in _flat(_pmf_prompt_text())

    def test_the_final_check_lead_is_worded_for_bins(self) -> None:
        quantity = _flat(_pmf_prompt_text())
        assert "units: the bin labels are in the base unit named above" in quantity[quantity.index("final checks") :]
        assert "what are the units of the output values" not in quantity
        date = _flat(_pmf_prompt_text(_mantic_date_q()))
        assert (
            "calendar check: does every bin you gave probability to fall on a day the resolution criteria allow"
            in date[date.index("final checks") :]
        )
        assert "does every date you output" not in date


class TestPmfPromptAcceptsEitherView:
    def test_the_date_question_and_its_epoch_view_render_identically(self) -> None:
        q = _mantic_date_q()
        assert pmf_prompt(q, "r", "l", "u") == pmf_prompt(as_epoch_question(q), "r", "l", "u")
