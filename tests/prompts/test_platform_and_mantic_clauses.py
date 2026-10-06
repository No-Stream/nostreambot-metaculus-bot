"""The platform-aware and Mantic-gated clauses added for the Mantic Phase 2 bundle (2026-09-08).

Each is a named constant in ``metaculus_bot/prompts.py`` with one interpolation site per prompt,
and each has a presence pin on the platform or question that renders it and an absence pin on the
one that must not. The Metaculus rendering of every prompt must be unchanged in substance, which
is what the absence pins on a Metaculus stub establish.
"""

from __future__ import annotations

import pytest

from metaculus_bot.prompts import (
    _CONTINUOUS_SCORING_RULE,
    _MANTIC_OUT_OF_RANGE_RATE_DATE,
    _MANTIC_OUT_OF_RANGE_RATE_QUANTITY,
    _MANTIC_SCORING_SENTENCE,
    _METACULUS_SCORING_SENTENCE,
    _MULTI_RESOLUTION_BINARY_RULE,
    _MULTI_RESOLUTION_CONTINUOUS_RULE,
    _MULTI_RESOLUTION_MC_RULE,
    binary_prompt,
    multiple_choice_prompt,
    numeric_prompt,
    stacking_binary_prompt,
    stacking_multiple_choice_prompt,
    stacking_numeric_prompt,
)
from tests.prompt_builders import (
    MANTIC_PAGE_URL,
    METACULUS_PAGE_URL,
    _binary_q,
    _date_prompt_text,
    _date_q,
    _flat,
    _mc_q,
    _numeric_prompt_text,
    _numeric_q,
    _open_upper_date_q,
    _pmf_q,
    _stacked_prompt_texts,
)


def _mantic_numeric_q(
    *,
    open_upper: bool = True,
    open_lower: bool = False,
    cdf_size: int = 201,
    zero_point: float | None = None,
    **question_json,
):
    """A Mantic numeric stub: the grid comes off the typed model (``cdf_size``, ``zero_point``), the
    per-question flags (``precision``, ``multi_resolution``) off ``api_json["question"]``."""
    q = _numeric_q()
    q.page_url = MANTIC_PAGE_URL
    q.open_upper_bound = open_upper
    q.open_lower_bound = open_lower
    q.cdf_size = cdf_size
    q.zero_point = zero_point
    q.api_json = {"question": question_json}
    return q


def _numeric_text(q) -> str:
    return numeric_prompt(q, research="r", lower_bound_message="lbm", upper_bound_message="ubm")


class TestPlatformScoringSentence:
    """B3. One scoring sentence per platform, chosen off ``page_url``, in all six forecaster prompts.

    The Metaculus sentence names the SPOT peer log score the bot tournaments use and says it is a
    proper score; the Mantic sentence names Crucible's spot baseline log score and its uniform
    reference, so the model is never told it is being compared to other forecasters on a platform
    where it is not. "Metaculus" appears in no Mantic prompt.
    """

    def test_metaculus_stubs_get_the_spot_peer_sentence_in_every_prompt(self) -> None:
        prompts = [
            binary_prompt(_binary_q(), research="r"),
            multiple_choice_prompt(_mc_q(), research="r"),
            _numeric_prompt_text(),
            *_stacked_prompt_texts(),
        ]
        for prompt in prompts:
            flat = _flat(prompt)
            assert _flat(_METACULUS_SCORING_SENTENCE) in flat
            assert "spot peer log score" in flat
            assert "crucible" not in flat
            assert "spot baseline" not in flat

    def test_mantic_stubs_get_the_spot_baseline_sentence_in_every_prompt(self) -> None:
        binary = _binary_q()
        binary.page_url = MANTIC_PAGE_URL
        mc = _mc_q()
        mc.page_url = MANTIC_PAGE_URL
        numeric = _mantic_numeric_q()
        prompts = [
            binary_prompt(binary, research="r"),
            multiple_choice_prompt(mc, research="r"),
            _numeric_text(numeric),
            _date_prompt_text(_open_upper_date_q()),
            stacking_binary_prompt(binary, research="r", base_predictions=["a1"]),
            stacking_multiple_choice_prompt(mc, research="r", base_predictions=["a1"]),
            stacking_numeric_prompt(
                numeric, research="r", base_predictions=["a1"], lower_bound_message="l", upper_bound_message="u"
            ),
        ]
        for prompt in prompts:
            flat = _flat(prompt)
            assert _flat(_MANTIC_SCORING_SENTENCE) in flat
            assert "compares you to a uniform distribution over the outcomes rather than to other forecasters" in flat
            assert "nothing is gained by disagreeing with the obvious answer and nothing is lost by giving it" in flat
            assert "metaculus" not in flat, "a Mantic prompt must not attribute its scoring to Metaculus"

    def test_the_old_platform_attributions_are_gone(self) -> None:
        """All six prompts: the MC base prompt is where "Metaculus' log-score" lived."""
        for prompt in (
            binary_prompt(_binary_q(), research="r"),
            multiple_choice_prompt(_mc_q(), research="r"),
            _numeric_prompt_text(),
            _date_prompt_text(),
            *_stacked_prompt_texts(),
        ):
            assert "the Metaculus peer score" not in prompt
            assert "Metaculus' log-score" not in prompt
            assert "Your Metaculus question is" not in prompt
            assert "Metaculus continuous questions" not in prompt

    def test_the_stacking_mc_preamble_states_the_sum_in_decimals(self) -> None:
        """The base MC prompt lost its percent-scale sum statements in the 2026-09-02 de-bloat
        (a percent-scale ``option_probs`` is hard-rejected); the stacking twin's preamble still
        said 100% and is brought in line while its scoring sentence is replaced."""
        flat = _flat(stacking_multiple_choice_prompt(_mc_q(), research="r", base_predictions=["a1"]))
        assert "sum to 1.0" in flat
        assert "sum to **100%**" not in flat
        assert "sum to 100%" not in flat


class TestContinuousScoringRule:
    """B2. The Scoring Rule paragraph in the numeric, date and stacking-numeric prompts.

    The Metaculus-specific sentences (a uniform 0.01 PDF floor, ln(0.01) = -4.6, a sharpness cap
    near 35) told the model the out-of-range cliff was an order of magnitude shallower than Mantic's
    5% reference makes it. The proper-scoring sentence stays; the open-bound sentence replaces the
    uninformative "scored as a binary event".
    """

    _RETIRED = (
        "uniform 0.01 floor",
        "ln(0.01)",
        "-4.605",
        "sharpness is capped",
        "0.01 ≤ f ≤ ~35",
        "spiky tricks",
        "scored as a binary event",
        "Metaculus continuous questions",
    )

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(_numeric_prompt_text, id="numeric"),
            pytest.param(_date_prompt_text, id="date"),
            pytest.param(lambda: _stacked_prompt_texts()[2], id="stacking_numeric"),
        ],
    )
    def test_scoring_rule_is_the_shared_constant_once(self, build) -> None:
        prompt = build()
        flat = _flat(prompt)
        assert flat.count(_flat(_CONTINUOUS_SCORING_RULE)) == 1
        assert "mass beyond an open bound is scored as its own outcome against a reference of a few percent" in flat
        assert "so starving it is heavily punished" in flat
        assert "this is a proper scoring rule" in flat
        for retired in self._RETIRED:
            assert retired not in prompt, f"retired scoring sentence {retired!r} is back"

    def test_the_constant_carries_no_em_dash(self) -> None:
        assert "—" not in _CONTINUOUS_SCORING_RULE


class TestSeriesVariantClause:
    """B4. The resolution-metric echo's reconciliation bullet keeps the 44211 correction ("inside the
    range confirms nothing") and drops the premise that the bounds were set by someone who could see
    the real series, which on Mantic told a correctly extrapolating forecaster it had the wrong variant."""

    def test_numeric_and_date_prompts_carry_the_rewritten_bullet(self) -> None:
        for prompt in (_numeric_prompt_text(), _date_prompt_text()):
            flat = _flat(prompt)
            assert "reconcile each candidate against the displayed range above" in flat
            assert "weak evidence about which series variant resolves" in flat
            assert "no evidence about the magnitude of the outcome" in flat
            assert "a candidate that falls outside an open bound may still be the right variant" in flat
            assert 'do not read "inside the range" as confirming the headline or component series' in flat
            assert "could see the real series" not in flat
            assert "probably the wrong variant" not in flat


class TestManticOutOfRangeBaseRate:
    """B1. Mantic's measured out-of-range base rate, appended after the bound messages.

    Rendered only on a Mantic question with the relevant open bound: the quantity sentence for a
    numeric question with either bound open, the date sentence for a date question open above.
    Metaculus questions never see either, so Metaculus behaviour cannot move.
    """

    def test_mantic_numeric_with_an_open_bound_carries_the_quantity_rate(self) -> None:
        """Series 1 corpus, annulled excluded, re-verified 2026-09-08: 51 of 274 quantitative questions
        resolved outside the range (35 of 141 discrete, 16 of 133 numeric, the numeric count including
        the seven escapes the platform stored as raw values rather than as a bound token)."""
        for kwargs in ({"open_upper": True, "open_lower": False}, {"open_upper": False, "open_lower": True}):
            prompt = _numeric_text(_mantic_numeric_q(**kwargs))
            flat = _flat(prompt)
            assert _flat(_MANTIC_OUT_OF_RANGE_RATE_QUANTITY) in flat
            assert "one in five past quantitative questions resolved outside the displayed range" in flat
            assert "one in four of the discrete ones, one in eight of the continuous ones" in flat
            assert "one in seven" not in flat
            assert "one in fifteen" not in flat
            assert "asserts a 1% chance of an out-of-range outcome" in flat
            # It follows the bound messages, which is where the model reads about open bounds.
            assert prompt.index("ubm") < prompt.index("one in five past quantitative questions")

    def test_mantic_numeric_with_both_bounds_closed_carries_nothing(self) -> None:
        flat = _flat(_numeric_text(_mantic_numeric_q(open_upper=False, open_lower=False)))
        assert "on this platform" not in flat
        assert "out-of-range outcome" not in flat

    def test_mantic_date_open_above_carries_the_date_rate(self) -> None:
        flat = _flat(_date_prompt_text(_open_upper_date_q()))
        assert _flat(_MANTIC_OUT_OF_RANGE_RATE_DATE) in flat
        assert "about half of past date questions with an open upper bound resolved after it (101 of 188" in flat
        assert "your p50 belongs above the upper bound" in flat
        assert _flat(_MANTIC_OUT_OF_RANGE_RATE_QUANTITY) not in flat

    def test_mantic_date_with_a_closed_upper_bound_carries_nothing(self) -> None:
        """Post 651 is closed on both sides: the half-resolve-above rate is about OPEN upper bounds."""
        q = _date_q()
        q.page_url = MANTIC_PAGE_URL
        flat = _flat(_date_prompt_text(q))
        assert "on this platform" not in flat

    def test_metaculus_questions_carry_neither_sentence(self) -> None:
        q = _numeric_q()
        q.open_upper_bound = True
        flat = _flat(_numeric_text(q))
        assert "on this platform" not in flat
        assert _flat(_MANTIC_OUT_OF_RANGE_RATE_QUANTITY) not in flat
        assert _flat(_MANTIC_OUT_OF_RANGE_RATE_DATE) not in flat
        # Post 651 as recorded carries a Metaculus-style page_url in this fixture's default rendering
        # only if the caller sets one; the recorded payload's own URL is the competition host.


class TestMultiResolutionClause:
    """B5. Gated on the question's own ``multi_resolution: true`` (identity, not truthiness), one
    type-aware sentence: mixture percentiles for continuous, expected share per option for MC,
    expected fraction of Yes for binary. Never interpolates the resolution count, which the API
    does not expose while the question is open. Base prompts only."""

    def test_binary_multi_resolution_asks_for_the_expected_fraction_of_yes(self) -> None:
        q = _binary_q()
        q.api_json = {"question": {"multi_resolution": True}}
        flat = _flat(binary_prompt(q, research="r"))
        assert _flat(_MULTI_RESOLUTION_BINARY_RULE) in flat
        assert "the fraction of those resolutions you expect to be yes" in flat

    def test_mc_multi_resolution_asks_for_the_expected_share_per_option(self) -> None:
        q = _mc_q()
        q.api_json = {"question": {"multi_resolution": True}}
        flat = _flat(multiple_choice_prompt(q, research="r"))
        assert _flat(_MULTI_RESOLUTION_MC_RULE) in flat
        assert "the share of those resolutions you expect to land on it" in flat

    def test_numeric_multi_resolution_asks_for_the_mixture(self) -> None:
        q = _numeric_q()
        q.api_json = {"question": {"multi_resolution": True}}
        prompt = _numeric_text(q)
        flat = _flat(prompt)
        assert _flat(_MULTI_RESOLUTION_CONTINUOUS_RULE) in flat
        assert "pool those forecasts into a single mixture and report the mixture's percentiles" in flat
        # It is a scoring rule, so it sits in the Scoring Rule section.
        assert (
            prompt.index("── Scoring Rule")
            < prompt.index("scored against EVERY resolution value")
            < prompt.index("── Intelligence Briefing")
        )

    def test_date_multi_resolution_asks_for_the_mixture(self) -> None:
        q = _open_upper_date_q()
        q.api_json["question"]["multi_resolution"] = True
        assert _flat(_MULTI_RESOLUTION_CONTINUOUS_RULE) in _flat(_date_prompt_text(q))

    def test_no_clause_interpolates_a_count(self) -> None:
        for rule in (_MULTI_RESOLUTION_BINARY_RULE, _MULTI_RESOLUTION_MC_RULE, _MULTI_RESOLUTION_CONTINUOUS_RULE):
            assert "{" not in rule
            assert not any(char.isdigit() for char in rule)

    @pytest.mark.parametrize("flag", [False, None, "true", 1])
    def test_anything_but_true_renders_nothing(self, flag) -> None:
        """Identity, not truthiness: a MagicMock stub's attribute chain is truthy, and so is the
        string "true"; only the platform's own boolean renders the clause."""
        q = _numeric_q()
        q.api_json = {"question": {"multi_resolution": flag}}
        assert "scored against every resolution" not in _flat(_numeric_text(q))

    def test_metaculus_stubs_and_stacking_prompts_carry_no_clause(self) -> None:
        for prompt in (
            binary_prompt(_binary_q(), research="r"),
            multiple_choice_prompt(_mc_q(), research="r"),
            _numeric_prompt_text(),
            _date_prompt_text(),
            *_stacked_prompt_texts(),
        ):
            assert "scored against every resolution" not in _flat(prompt)


class TestScoringGridClause:
    """B6. The bin count and width, templated on Mantic's ``precision`` (quantitative) or
    ``date_granularity`` (date); absent on Metaculus questions, whose payloads carry neither.

    The bin count is the typed model's ``cdf_size - 1``, the same number the CDF builder keys off,
    so the prompt can never describe a grid other than the one the pipeline submits."""

    def test_numeric_with_precision_names_the_grid(self) -> None:
        q = _mantic_numeric_q(precision=100.0, cdf_size=451)
        q.unit_of_measure = "$"
        flat = _flat(_numeric_text(q))
        assert "scoring grid: 450 bins of width 100 $" in flat
        assert "a percentile's value selects the bin it falls in, so detail finer than one bin is wasted" in flat

    def test_numeric_with_precision_on_a_log_grid_names_no_width(self) -> None:
        """Mantic's OpenAPI defines ``precision`` on a logarithmic grid as the RATIO between adjacent
        boundaries, so the additive width phrase would be false across the whole axis; the clause
        names the geometry instead and renders no number it cannot stand behind."""
        q = _mantic_numeric_q(precision=1.1, cdf_size=201, zero_point=0.0)
        flat = _flat(_numeric_text(q))
        assert "scoring grid: 200 bins, log-spaced" in flat
        assert "a percentile's value selects the bin it falls in, so detail finer than one bin is wasted" in flat
        assert "bins of width" not in flat

    def test_numeric_without_precision_names_no_grid(self) -> None:
        q = _mantic_numeric_q(cdf_size=451)
        assert "scoring grid" not in _flat(_numeric_text(q))
        assert "scoring grid" not in _flat(_numeric_prompt_text())

    def test_date_with_day_granularity_names_the_day_bins(self) -> None:
        flat = _flat(_date_prompt_text(_date_q()))
        assert "scoring grid: 12 bins of one calendar day each, in utc" in flat
        assert "a date with no time of day means that whole day" in flat
        assert "detail finer than one day is wasted" in flat

    def test_date_without_granularity_names_no_grid(self) -> None:
        assert "scoring grid" not in _flat(_date_prompt_text(_open_upper_date_q()))


class TestDiscreteBinEdgesClause:
    """A discrete question's prompt names its bin edges, on both platforms. Models wrote "0 to 1"
    to mean zero on an integer count grid whose zero bin is [-0.5, 0.5], so percentiles meant for
    zero landed in bin 1: on Mantic 708 members stated 45%, 46% and 30% at zero but their built
    distributions held 34%, 39% and 22%. Metaculus discrete questions got no grid information at
    all before this."""

    @pytest.mark.parametrize("page_url", [METACULUS_PAGE_URL, MANTIC_PAGE_URL])
    def test_integer_count_grid_names_half_integer_edges_and_lowest_bin_example(self, page_url: str) -> None:
        question = _pmf_q(
            cdf_size=36, lower_bound=-0.5, upper_bound=34.5, nominal_lower=0.0, nominal_upper=34.0, page_url=page_url
        )
        flat = " ".join(_numeric_text(question).split())
        assert "Bin edges: -0.5, 0.5, 1.5, …, 34.5." in flat
        assert "A bin's probability is the share of your distribution between its edges" in flat
        assert "a percentile just past a bin's upper edge already falls in the next bin" in flat
        assert (
            "with nothing below -0.5, giving the lowest bin 40% means every percentile up to the 40th lies between "
            "-0.5 and 0.5" in flat
        )

    def test_non_integer_width_grid_renders_its_own_edges(self) -> None:
        question = _pmf_q(cdf_size=11, lower_bound=0.0, upper_bound=50.0, nominal_lower=0.0, nominal_upper=50.0)
        flat = " ".join(_numeric_text(question).split())
        assert "Bin edges: 0, 5, 10, …, 50." in flat
        assert "lies between 0 and 5" in flat

    def test_open_lower_bound_example_is_conditional_on_no_lower_tail_mass(self) -> None:
        """On an open lower bound probability may sit below the grid, so the lowest-bin example must
        not tell the model every low percentile belongs inside the first bin unconditionally."""
        question = _pmf_q(
            cdf_size=36, lower_bound=-0.5, upper_bound=34.5, nominal_lower=0.0, nominal_upper=34.0, open_lower=True
        )
        flat = " ".join(_numeric_text(question).split())
        assert "with nothing below -0.5, giving the lowest bin 40%" in flat
        assert "must lie between" not in flat

    def test_continuous_numeric_question_has_no_bin_edges_clause(self) -> None:
        flat = " ".join(_numeric_text(_numeric_q()).split())
        assert "Bin edges:" not in flat

    def test_log_scaled_discrete_grid_has_no_bin_edges_clause(self) -> None:
        question = _pmf_q(
            cdf_size=11, lower_bound=1.0, upper_bound=1000.0, nominal_lower=1.0, nominal_upper=1000.0, zero_point=0.0
        )
        flat = " ".join(_numeric_text(question).split())
        assert "Bin edges:" not in flat

    def test_stacking_numeric_prompt_does_not_carry_the_clause(self) -> None:
        question = _pmf_q(cdf_size=36, lower_bound=-0.5, upper_bound=34.5, nominal_lower=0.0, nominal_upper=34.0)
        stacked = stacking_numeric_prompt(
            question, research="r", base_predictions=[], lower_bound_message="lbm", upper_bound_message="ubm"
        )
        assert "Bin edges:" not in stacked
