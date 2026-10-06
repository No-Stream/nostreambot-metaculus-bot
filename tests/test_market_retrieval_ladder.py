"""The ``↳ [remaining N]`` ladder row: the completeness invariant, the compaction stages, the bounds.

Split out of ``test_market_retrieval_rendering`` because it guards a different property. That file
pins the TABLE — its columns, its legend, its bullets, its character budgets. This one pins the one
claim the 2026-08-24 residual round's Track A exists to make:

    **A multi-outcome family's price distribution reaches the forecaster whole.**

Every remaining outcome is either named with its own price or inside a group that states its count and
its summed price. Nothing else is acceptable, and ``TestCompletenessInvariant`` asserts it as a
property over every family shape rather than as a string match on one.

Why it matters, in the shape that cost real score. q45189 asked which bracket of the vote share Randy
Fine would win the FL-06 Republican primary with. Exactly one market in existence priced that
quantity — Kalshi's ten-bracket margin ladder — and the render showed the forecasters ONE of its
brackets. All three read that single bracket as an equality constraint on a tail (``P(F>70%) = 0.585``
"exactly"), each then cut its own pre-market mass on the resolving bucket, and the published 0.130
scored -26.77 spot. Complete enumeration of the same ladder lands at 0.19-0.29 under the models' own
margin→share formula. A single bracket of a distribution is not a small amount of the distribution; it
is a different kind of object.
"""

from __future__ import annotations

import re

import pytest

from metaculus_bot.research.market_retrieval.ranking import RENDER_BUDGET
from metaculus_bot.research.market_retrieval.rendering import (
    CHILD_ROW_MARKER,
    CHILD_TITLE_MAX_CHARS,
    LADDER_CUMULATIVE_PRICE_SUM,
    LADDER_HARD_BOUND_STAGE,
    LADDER_MAX_STAGE,
    LADDER_MIN_ROW_CHARS,
    LADDER_PRICE_FLOORS,
    LADDER_ROW_MAX_CHARS,
    LADDER_SECTION_MAX_CHARS,
    MAX_CHILD_ROWS_PER_MARKET,
    render_snapshot,
    render_snapshot_with_stats,
)
from metaculus_bot.research.market_retrieval.types import MarketChild, MarketMatch, MarketSnapshot
from tests.test_market_retrieval_rendering import _row, _table_rows

# One ladder term: `<label> <price>`, `<label> <price> R` when settled, or `<label> -` when the outcome
# carries no price. Anchored, because it is applied to a single already-split part.
_TERM_RE = re.compile(r"^(?P<label>.+?) (?:(?P<price>\d\.\d\d)(?P<resolved> R)?|-)$")

_LADDER_PREFIX = "[remaining "


def _ladder_titles(rendered: str) -> list[str]:
    """Every ladder row's title, in slate order."""
    return [
        cells["title"]
        for cells in _table_rows(rendered)
        if cells["platform"] == CHILD_ROW_MARKER and cells["title"].startswith(_LADDER_PREFIX)
    ]


def _full_sub_rows(rendered: str) -> list[dict[str, str]]:
    return [
        cells
        for cells in _table_rows(rendered)
        if cells["platform"] == CHILD_ROW_MARKER and not cells["title"].startswith(_LADDER_PREFIX)
    ]


def _ladder_groups(title: str) -> list[str]:
    """The ``+N ...`` collapse groups in one ladder title."""
    return [part for part in title.split("] ", 1)[1].split(" / ") if part.startswith("+")]


def _ladder_terms(title: str) -> list[tuple[str, float | None, bool]]:
    """Each individually-named outcome in one ladder title, as ``(label, price, is_resolved)``.

    Parsed by splitting on the separator and matching an anchored per-term pattern, rather than scanning
    the whole title: a label can legitimately contain a number and a dash (``Randy Fine, 30-40%``), and
    an unanchored scan reads those as prices.
    """
    out: list[tuple[str, float | None, bool]] = []
    for part in title.split("] ", 1)[1].split(" / "):
        if part.startswith("+"):
            continue
        match = _TERM_RE.match(part)
        assert match is not None, f"unparseable ladder term: {part!r}"
        price = match["price"]
        out.append((match["label"], None if price is None else float(price), bool(match["resolved"])))
    return out


def _open_price_total(children: tuple[MarketChild, ...]) -> float:
    return sum(
        child.implied_prob_yes for child in children if child.implied_prob_yes is not None and not child.is_resolved
    )


# The family shapes the completeness invariant is asserted over. Each is a real archived shape or the
# pathological bound of one, and together they cover every branch the ladder can take: a partition
# whose leaders ARE its mass, a cumulative ladder whose leaders are near-certainties carrying no
# information, a family with no forecast content at all, a family with no prices at all, the
# duplicate-instance trap, and a family no price floor can compact.
_FAMILY_SHAPES: dict[str, tuple[MarketChild, ...]] = {
    "partition": tuple(
        MarketChild(title=f"bracket {index}", implied_prob_yes=price)
        for index, price in enumerate((0.42, 0.29, 0.15, 0.08, 0.04, 0.02))
    ),
    "cumulative_ladder": tuple(
        MarketChild(title=f"Above ${3000 + 40 * index}", implied_prob_yes=price)
        for index, price in enumerate((0.99, 0.99, 0.98, 0.96, 0.91, 0.80, 0.62, 0.46, 0.28, 0.13, 0.05, 0.01))
    ),
    "all_settled": tuple(
        MarketChild(title=f"Above {index}", implied_prob_yes=1.0, is_resolved=True) for index in range(14)
    ),
    "all_unquoted": tuple(MarketChild(title=f"Candidate {index}", price_withheld=True) for index in range(14)),
    "single_child": (MarketChild(title="only rung", implied_prob_yes=0.33),),
    "duplicate_instances": (MarketChild(title="same rung", implied_prob_yes=0.11),) * 14,
    "two_hundred_rungs": tuple(
        MarketChild(title=f"rung {index}", implied_prob_yes=0.90)
        for index in range(200)  # HARNESS-SCAN-EXEMPT-subsampling
    ),
}


class TestQ45189BracketLadder:
    """The failure this design exists to remove, on the real archived family.

    Ten Kalshi brackets, prices straight off ``backtests/research_archive/raw`` (run 31205419393,
    q45189), in the order that archive stored them. Their open prices sum to 0.965 — a
    mutually-exclusive partition — so the arithmetic below is a real completeness check rather than a
    formatting assertion.
    """

    # The archived children, verbatim. The low `Randy Fine` brackets are the decisive ones: they are
    # what P(Fine wins with <=60%) is built from, and they are exactly what the shipped render cut.
    _BRACKETS = (
        ("Dan Bilzerian, 0-5%", 0.005, 12.5122, 1870.45),
        ("Randy Fine, >=50%", 0.585, 1376.61615, 1403.54),
        ("Dan Bilzerian, 5-10%", 0.005, 6.489249999999999, 1082.22),
        ("Dan Bilzerian, >=10%", 0.005, 3.80625, 492.5),
        ("Randy Fine, 40-50%", 0.12, 87.5064, 422.86),
        ("Randy Fine, 30-40%", 0.08, 60.8208, 340.76),
        ("Randy Fine, 5-10%", 0.045, 0.045, 1.0),
        ("Randy Fine, 20-30%", 0.04, 0.0, 0.0),
        ("Randy Fine, 10-20%", 0.04, 0.0, 0.0),
        ("Randy Fine, 0-5%", 0.04, 0.0, 0.0),
    )
    _DECISIVE = ("Randy Fine, 30-40%", "Randy Fine, 20-30%", "Randy Fine, 10-20%", "Randy Fine, 0-5%")

    def _children(self) -> tuple[MarketChild, ...]:
        return tuple(
            MarketChild(title=title, implied_prob_yes=price, total_volume=volume, open_interest=open_interest)
            for title, price, volume, open_interest in self._BRACKETS
        )

    def _slate(self) -> list[MarketMatch]:
        """A full 8-row slate, which is what squeezes this family down to two full sub-rows.

        The single-family case is not the interesting one — it renders every bracket as a full row and
        always did. A real question's snapshot is eight ranked markets sharing one sub-row budget, and
        that is where the truncation bit: this family got two rows and a marker line.

        The seven companions are two-outcome markets, matching the archive rather than this family: a
        ladder-title total of 240 chars is the archived median, so an ordinary snapshot compacts nothing
        and the ladder names every outcome. ``test_a_slate_of_wide_families_counts_the_cheap_tail``
        covers the other end.
        """
        companions = [
            _row(
                title=f"companion {index}",
                tier="weak",
                prob=None,
                children=(
                    MarketChild(title="yes leg", implied_prob_yes=0.61),
                    MarketChild(title="no leg", implied_prob_yes=0.39),
                ),
            )
            for index in range(RENDER_BUDGET - 1)
        ]
        return [
            _row(
                title="FL-06 Republican primary margin of victory?",
                tier="same_quantity_other_cut",
                why="same race, margin rather than share",
                prob=None,
                children=self._children(),
            ),
            *companions,
        ]

    def test_every_bracket_reaches_the_forecaster(self) -> None:
        """The regression, stated as arithmetic: the individually-named open price equals the family's.

        Pre-fix this family rendered two brackets and a marker row saying 27% of the open prices were
        omitted — so a forecaster could put arithmetic on 0.705 of the ladder's 0.965 and the entire low
        end, which is what decides the resolving bucket, was gone.

        Summed over the CHILD prices rather than the rendered cells, deliberately: a cell rounds to two
        decimals, so `0.005` renders `0.01` and three of them would inflate a cell-based total by 0.015.
        Rounding is a display property; what this asserts is that no outcome went unnamed.
        """
        rendered = render_snapshot(MarketSnapshot(matches=self._slate()))

        named_titles = {cells["title"] for cells in _full_sub_rows(rendered)}
        named_titles |= {label for label, _price, _resolved in _ladder_terms(_ladder_titles(rendered)[0])}
        # A superset, because the slate's other seven markets contribute their own rows.
        assert {title for title, *_ in self._BRACKETS} <= named_titles
        named_total = sum(price for title, price, *_ in self._BRACKETS if title in named_titles)
        assert named_total == pytest.approx(_open_price_total(self._children()))
        assert named_total == pytest.approx(0.965), "a mutually-exclusive partition, so the sum is a check"

    def test_the_decisive_low_brackets_are_named_with_their_prices(self) -> None:
        """The four brackets that determine whether Fine wins with <=60%. Every one was cut pre-fix,
        and each carries a price a forecaster can put arithmetic on."""
        rendered = render_snapshot(MarketSnapshot(matches=self._slate()))

        title = _ladder_titles(rendered)[0]
        for label in self._DECISIVE:
            assert label in title, f"{label} must reach the forecaster"
        assert "Randy Fine, 30-40% 0.08" in title
        assert "Randy Fine, 0-5% 0.04" in title

    def test_nothing_is_described_as_omitted(self) -> None:
        """The marker row this replaces was honest and insufficient: no model on q45189 mentioned it,
        and it restored no price. Its vocabulary must not survive alongside the ladder, or a
        forecaster meets two different claims about the same table."""
        rendered = render_snapshot(MarketSnapshot(matches=self._slate()))

        assert "omitted" not in rendered
        assert "render budget" not in rendered

    def test_the_leading_brackets_still_keep_their_full_rows(self) -> None:
        """Completeness must not cost the detail: the two highest-priced brackets keep their own
        volume, open interest and liquidity label, which is what the full rows are for."""
        rendered = render_snapshot(MarketSnapshot(matches=self._slate()))

        leading = _full_sub_rows(rendered)[:2]
        assert [cells["title"] for cells in leading] == ["Randy Fine, >=50%", "Randy Fine, 40-50%"]
        assert [cells["prob"] for cells in leading] == ["0.58", "0.12"]
        assert leading[0]["total_vol"] == "1377"
        assert leading[0]["OI"] == "1404"

    def test_a_slate_of_wide_families_counts_the_cheap_tail_rather_than_dropping_it(self) -> None:
        """The other end of the budget: eight ten-bracket ladders on one snapshot do exceed the ladder
        section allowance, so compaction fires. What it may NOT do is lose an outcome — the three
        near-zero Bilzerian brackets become `+3 under 0.02 (0.01 summed)`, a counted set carrying its own
        weight, and every bracket a forecaster could put real arithmetic on is still named."""
        slate = [
            _row(title=f"margin family {index}", tier="same_quantity_other_cut", prob=None, children=self._children())
            for index in range(RENDER_BUDGET)
        ]

        rendered, stats = render_snapshot_with_stats(MarketSnapshot(matches=slate))

        title = _ladder_titles(rendered)[0]
        assert "+3 under 0.02 (0.01 summed)" in title
        for label in self._DECISIVE:
            assert label in title
        assert stats.named + stats.collapsed == stats.outcomes == 10 * RENDER_BUDGET
        assert stats.ladder_chars <= LADDER_SECTION_MAX_CHARS


class TestCompletenessInvariant:
    """Every open priced outcome is a full row, a named ladder term, or in a counted, summed group.

    Asserted as a PROPERTY across family shapes rather than as a string match on one, because the
    shapes differ in exactly the way that made a selection rule unanswerable: a partition's leaders are
    its mass, a cumulative ladder's leaders are near-certainties carrying no information, and an
    all-settled family has no forecast content at all.
    """

    @pytest.mark.parametrize("shape", sorted(_FAMILY_SHAPES))
    def test_every_outcome_is_named_or_counted(self, shape: str) -> None:
        """Named + collapsed always equals the family's outcome count. That is the invariant; a shape
        where they disagree has lost an outcome somewhere between the parser and the table."""
        children = _FAMILY_SHAPES[shape]
        matches = [_row(title=shape, prob=None, children=children)]

        _, stats = render_snapshot_with_stats(MarketSnapshot(matches=matches))

        assert stats.outcomes == len(children)
        assert stats.named + stats.collapsed == stats.outcomes

    @pytest.mark.parametrize("shape", sorted(_FAMILY_SHAPES))
    def test_a_collapsed_group_always_states_its_count(self, shape: str) -> None:
        """A group with no count is the silent cut this design removes, so every `+` group in every
        title has to carry one."""
        children = _FAMILY_SHAPES[shape]
        rendered = render_snapshot(MarketSnapshot(matches=[_row(title=shape, prob=None, children=children)]))

        for title in _ladder_titles(rendered):
            for group in _ladder_groups(title):
                assert re.match(r"\+\d+ ", group), group

    @pytest.mark.parametrize("shape", sorted(_FAMILY_SHAPES))
    def test_every_ladder_row_fits_its_own_cap(self, shape: str) -> None:
        children = _FAMILY_SHAPES[shape]
        rendered = render_snapshot(MarketSnapshot(matches=[_row(title=shape, prob=None, children=children)]))

        for title in _ladder_titles(rendered):
            assert len(title) <= LADDER_ROW_MAX_CHARS

    def test_a_tiny_price_is_still_named(self) -> None:
        """The MC-option-coverage shape: six archived (option, family) pairs had their ONLY outcome
        matching a named Metaculus option cut, because it was the cheapest rung in the family. A
        0.0015 bracket is the answer to "could this option happen at all", so a rule that drops it by
        price answers the wrong question."""
        children = (
            *(
                MarketChild(title=f"outcome {index}", implied_prob_yes=0.5)
                for index in range(MAX_CHILD_ROWS_PER_MARKET)
            ),
            MarketChild(title="the matching option", implied_prob_yes=0.0015),
        )

        rendered = render_snapshot(MarketSnapshot(matches=[_row(title="ballot", prob=None, children=children)]))

        assert "the matching option 0.00" in _ladder_titles(rendered)[0]

    def test_duplicate_instances_are_counted_separately(self) -> None:
        """All ladder bookkeeping is by POSITION, never by object identity. A family can legitimately
        hold two equal outcomes, and the committed `TestRenderBudget._maxed_rows` fixture holds ten
        references to ONE `MarketChild`; an `id()`-keyed set silently merges them. Not hypothetical —
        it is what made the design's costing pass report the maxed fixture 3,664 chars light."""
        repeated = (MarketChild(title="same rung", implied_prob_yes=0.11),) * 14

        rendered, stats = render_snapshot_with_stats(
            MarketSnapshot(matches=[_row(title="repeats", prob=None, children=repeated)])
        )

        assert stats.outcomes == 14
        assert stats.named + stats.collapsed == 14
        title = _ladder_titles(rendered)[0]
        assert title.startswith("[remaining 4] ")
        assert title.count("same rung 0.11") == 4


class TestCompactionStages:
    """Which group a ladder row gives up first, and what each collapsed group has to say."""

    def _forced(self, children: tuple[MarketChild, ...], *, stage: int) -> str:
        """The ladder title for a one-family slate, forced to ``stage`` via the module's own helper.

        Reaches for the private stage builder deliberately: the stage a real render picks is whatever
        the character budgets require, so pinning "what stage 2 says" through a full render would mean
        constructing a fixture that happens to land there — a fixture that stops testing the thing the
        moment a budget constant moves.
        """
        from metaculus_bot.research.market_retrieval.rendering import _ladder_at_stage

        return _ladder_at_stage(children, stage=stage).title

    def test_stage_zero_names_everything(self) -> None:
        children = (
            MarketChild(title="open", implied_prob_yes=0.4),
            MarketChild(title="settled", implied_prob_yes=1.0, is_resolved=True),
            MarketChild(title="unquoted"),
        )

        title = self._forced(children, stage=0)

        assert title == "[remaining 3] open 0.40 / settled 1.00 R / unquoted -"
        assert "+" not in title

    def test_unquoted_outcomes_collapse_first(self) -> None:
        """They carry no price, so there is nothing to say about them — the cheapest thing a row can
        give up."""
        children = (
            MarketChild(title="open", implied_prob_yes=0.4),
            MarketChild(title="unquoted a"),
            MarketChild(title="unquoted b"),
        )

        title = self._forced(children, stage=1)

        assert title == "[remaining 3] open 0.40 / +2 unquoted"

    def test_settled_outcomes_collapse_second_and_name_the_last_one(self) -> None:
        """The Manifold threshold-ladder shape: 10 rungs settled at 1.00 while the market stays open,
        so those titles are the floor the series has already crossed. At stage 0 every one is named; at
        stage 2 the group states its span and names its LAST member in family order, which on an
        ordered ladder is the crossing point. Collapsing them to a bare count would delete that."""
        settled = tuple(
            MarketChild(title=f"Over ${4.00 + 0.10 * index:.2f}", implied_prob_yes=1.0, is_resolved=True)
            for index in range(10)
        )
        open_rungs = tuple(
            MarketChild(title=f"Over ${4.60 + 0.05 * index:.2f}", implied_prob_yes=price)
            for index, price in enumerate((0.4992, 0.3723, 0.2765, 0.2290, 0.1774, 0.1505, 0.1212))
        )
        children = settled + open_rungs

        assert all(f"Over ${4.00 + 0.10 * index:.2f}" in self._forced(children, stage=0) for index in range(10))

        title = self._forced(children, stage=2)

        assert "+10 settled at 1.00, last Over $4.90" in title
        assert "Over $4.60 0.50" in title, "the open rungs are untouched by the settled collapse"

    def test_a_mixed_price_settled_group_reports_its_span(self) -> None:
        """A Polymarket event's legs settle to 1.00 and 0.00, so one price would misdescribe the
        group."""
        children = (
            MarketChild(title="won", implied_prob_yes=1.0, is_resolved=True),
            MarketChild(title="lost", implied_prob_yes=0.0, is_resolved=True),
        )

        assert "+2 settled at 0.00-1.00, last lost" in self._forced(children, stage=2)

    def test_the_settled_span_refuses_an_unquoted_price_rather_than_fabricating_zero(self) -> None:
        """`_quoted_price` raises when the ``implied_prob_yes is not None`` filter regresses.

        Its predecessor ``or 0.0`` was dead code today (the settled list is pre-filtered),
        but a regression in that filter would have rendered a fabricated 0.00 into the
        settled span — the withheld-price-as-real-zero shape 58175a7 fixed at the venue
        parsers. A loud failure beats a wrong span.
        """
        from metaculus_bot.research.market_retrieval.rendering import _quoted_price

        with pytest.raises(ValueError, match="unquoted outcome reached price arithmetic"):
            _quoted_price(MarketChild(title="unquoted", price_withheld=True))

    def test_open_outcomes_collapse_last_cheapest_first(self) -> None:
        """The floor escalates only after unquoted and settled are gone, and a collapsed open group
        states its summed price — the one figure a forecaster needs to know the tail's weight."""
        children = tuple(
            MarketChild(title=f"rung {index}", implied_prob_yes=price)
            for index, price in enumerate((0.60, 0.30, 0.04, 0.03, 0.02, 0.01))
        )

        assert "+" not in self._forced(children, stage=2), "floor 0.00 collapses nothing"

        # Stage 3 is the FIRST floor above zero, and the comparison is strict: 0.02 is not below 0.02.
        assert self._forced(children, stage=3).endswith("+1 under 0.02 (0.01 summed)")
        assert "rung 4 0.02" in self._forced(children, stage=3)
        # Two stages further and the floor reaches 0.10, taking the whole cheap tail with it.
        assert self._forced(children, stage=5).endswith("+4 under 0.10 (0.10 summed)")
        assert "rung 0 0.60 / rung 1 0.30" in self._forced(children, stage=5)

    def test_the_floors_climb_monotonically(self) -> None:
        """Each stage may only ask for MORE compaction than the last, or escalating could grow a
        title and the section loop would not terminate usefully."""
        assert list(LADDER_PRICE_FLOORS) == sorted(LADDER_PRICE_FLOORS)
        assert LADDER_PRICE_FLOORS[0] == 0.0, "stage 2 must collapse no open outcome"
        assert 2 + len(LADDER_PRICE_FLOORS) - 1 == LADDER_MAX_STAGE

    def test_a_cumulative_ladder_collapses_both_tails_and_keeps_its_crossing(self) -> None:
        """The one place family SHAPE has to be read, and the defect that made it necessary.

        A collapse ranked by price is right on a partition and close to worst-possible on a cumulative
        threshold ladder, which is the design's own §2b finding one level down. On the archive's real
        50-rung gold ladder a price-ranked collapse named eight rungs at 0.99 — `Above $3251.99`,
        `Above $3691.99`, ... — deep in-the-money near-certainties carrying no forecast content at all,
        and counted `Above $4411.99 0.46`, the crossing that answers the question, into `+39 more`.

        Ranking by distance from certainty fixes it without a per-shape rule at the call site: on a
        partition `min(p, 1-p)` reduces to the price (every outcome sits below 0.5 once several share a
        1.0 budget), and on a cumulative ladder it collapses BOTH tails and keeps the middle. The shape
        test is the family's own summed open price — a partition sums to ~1, a survival ladder to far
        more.
        """
        ladder = tuple(
            MarketChild(title=f"Above ${3200 + 40 * index}", implied_prob_yes=price)
            for index, price in enumerate((0.99, 0.99, 0.99, 0.92, 0.74, 0.46, 0.21, 0.06, 0.02, 0.01))
        )
        assert sum(child.implied_prob_yes or 0.0 for child in ladder) > LADDER_CUMULATIVE_PRICE_SUM

        title = self._forced(ladder, stage=4)  # floor 0.05

        assert "Above $3400 0.46" in title, "the crossing is exactly what must survive"
        assert "Above $3360 0.74" in title, "and the rungs bracketing it"
        assert "Above $3440 0.21" in title, "and the rungs bracketing it"
        assert "Above $3200 0.99" not in title, "a 0.99 rung on a live ladder says nothing"
        assert "Above $3560 0.01" not in title, "and neither does its opposite tail"
        assert "+5 off certainty by under 0.05 (3.00 summed)" in title

    def test_a_partition_collapses_by_price_as_before(self) -> None:
        """The regression guard on the other shape: a mutually-exclusive family's leaders ARE its mass,
        so the shape-aware rule must leave it behaving exactly as a price-ranked collapse did. q45189's
        own ten brackets sum to 0.965, well under the cumulative threshold."""
        brackets = tuple(
            MarketChild(title=f"bracket {index}", implied_prob_yes=price)
            for index, price in enumerate((0.585, 0.12, 0.08, 0.045, 0.04, 0.04, 0.04, 0.005, 0.005, 0.005))
        )
        assert sum(child.implied_prob_yes or 0.0 for child in brackets) < LADDER_CUMULATIVE_PRICE_SUM

        title = self._forced(brackets, stage=4)  # floor 0.05

        assert "bracket 0 0.58" in title, "the leader survives, unlike on a cumulative ladder"
        assert "bracket 1 0.12" in title
        assert "bracket 2 0.08" in title
        assert "+7 under 0.05 (0.18 summed)" in title
        assert "off certainty" not in title, "the label must say 'under' — this family is a partition"

    def test_a_collapsed_group_counts_only_real_open_prices(self) -> None:
        """The disclosure arithmetic. A blanked price contributes nothing and a settled one is not
        forecast content, so a group's summed figure counts neither — which is what makes blanking the
        venues' manufactured 0.50 defaults correct these numbers for free rather than by a second
        rule."""
        children = (
            MarketChild(title="lead", implied_prob_yes=0.90),
            MarketChild(title="cheap open", implied_prob_yes=0.01),
            MarketChild(title="blanked", price_withheld=True, quote_low=0.0, quote_high=1.0),
            MarketChild(title="settled", implied_prob_yes=1.0, is_resolved=True),
        )

        title = self._forced(children, stage=3)

        assert "+1 under 0.02 (0.01 summed)" in title, "0.50 of a blanked price must not be summed in"


class TestLadderSectionBudget:
    """The section-level allowance, and what happens when compaction runs out."""

    def _families(self, *, count: int, prices: tuple[float, ...]) -> list[MarketMatch]:
        return [
            _row(
                title=f"family {index}",
                prob=None,
                children=tuple(
                    MarketChild(title=f"f{index} rung {position:02d}", implied_prob_yes=price)
                    for position, price in enumerate(prices)
                ),
            )
            for index in range(count)
        ]

    def test_the_longest_title_is_compacted_first(self) -> None:
        """Largest-first buys the most characters per stage and is independent of slate order, so it is
        also what keeps the render deterministic."""
        wide = tuple(0.90 - 0.004 * index for index in range(10))
        narrow = (0.60, 0.30)
        matches = self._families(count=1, prices=narrow) + self._families(count=7, prices=wide)
        for index, match in enumerate(matches):
            match.market_title = f"family {index}"

        _, stats = render_snapshot_with_stats(MarketSnapshot(matches=matches))

        assert stats.ladder_chars <= LADDER_SECTION_MAX_CHARS

    def test_the_render_is_byte_identical_across_calls(self) -> None:
        """Two calls on the same snapshot must produce the same bytes: the compaction loop breaks its
        ties on index and the stage builders take no set-iteration order, so nothing here may depend on
        hash seeding."""
        matches = self._families(count=RENDER_BUDGET, prices=tuple(0.90 - 0.004 * index for index in range(10)))
        snapshot = MarketSnapshot(matches=matches)

        assert render_snapshot(snapshot) == render_snapshot(snapshot)

    def test_a_slate_of_near_certainties_still_fits_the_section_allowance(self) -> None:
        """The shape stage escalation could not fix under a price-ranked collapse.

        A cumulative ladder quoting 0.99 down its whole in-the-money half — the archive holds a real
        50-rung gold ladder like this — has nothing unquoted, nothing settled, and nothing below any
        price floor, so a price-ranked collapse is a no-op at every stage: eight of them reached
        `LADDER_MAX_STAGE` with 3,672 chars of titles against a 1,400 allowance and rendered 12,951 chars
        against a committed 10,600 budget. Ranking by distance from certainty instead compacts them at
        the first real floor, because a 0.99 rung is 0.01 off certainty and carries no forecast content.
        """
        matches = self._families(count=RENDER_BUDGET, prices=(0.99,) * 20)

        rendered, stats = render_snapshot_with_stats(MarketSnapshot(matches=matches))

        assert stats.ladder_chars <= LADDER_SECTION_MAX_CHARS
        assert stats.named + stats.collapsed == stats.outcomes
        assert stats.collapsed > 0, "the fixture must be wide enough to force compaction"
        titles = _ladder_titles(rendered)
        # Not every row compacts: the loop escalates the longest title first and stops the moment the
        # section fits, so the rows it never reached stay uncompacted. That is the point of largest-first.
        assert any(_ladder_groups(title) for title in titles)
        for title in titles:
            assert _ladder_terms(title), "a row must never name nothing"
            for group in _ladder_groups(title):
                assert re.fullmatch(r"\+\d+ off certainty by under \d\.\d\d \(\d+\.\d\d summed\)", group), group

    def test_the_hard_bound_keeps_the_most_informative_terms(self) -> None:
        """The last tier, on the one shape no floor reaches: 200 open rungs at exactly 0.50.

        Every one of them sits the maximum 0.50 from certainty, so no floor can collapse them and the
        stage-0 title runs past 11,000 chars. The row keeps what fits and closes with a counted, summed
        remainder — and it always names at least one, because a row that named nothing would be the bare
        count this design removes.
        """
        children = tuple(
            MarketChild(title=f"rung {index:03d}", implied_prob_yes=0.50)
            for index in range(200)  # HARNESS-SCAN-EXEMPT-subsampling
        )

        rendered, stats = render_snapshot_with_stats(
            MarketSnapshot(matches=[_row(title="pathological", prob=None, children=children)])
        )

        title = _ladder_titles(rendered)[0]
        assert len(title) <= LADDER_ROW_MAX_CHARS
        assert re.search(r"\+\d+ more \(\d+ priced, \d+\.\d\d summed\)$", title), title
        assert stats.named >= 1
        assert stats.named + stats.collapsed == stats.outcomes

    def test_no_floor_can_empty_a_ladder_row(self) -> None:
        """The guarantee that keeps a floor from producing a bare count.

        200 rungs at 0.90 all sit 0.10 off certainty, so the top 0.50 floor would otherwise swallow every
        one of them and leave `[remaining 190] +190 off certainty by under 0.50 (...)` — arithmetically
        honest and useless. The most informative live outcome survives every floor.
        """
        children = tuple(
            MarketChild(title=f"rung {index:03d}", implied_prob_yes=0.90)
            for index in range(200)  # HARNESS-SCAN-EXEMPT-subsampling
        )
        from metaculus_bot.research.market_retrieval.rendering import _ladder_at_stage

        for stage in range(LADDER_MAX_STAGE + 1):
            row = _ladder_at_stage(children, stage=stage)
            assert _ladder_terms(row.title), f"stage {stage} named nothing"
            assert row.named >= 1

    def test_an_uncompacted_slate_names_every_outcome(self) -> None:
        """The common case, and the point of setting the allowance where the archive's own titles sit:
        37 of 42 archived snapshots compact nothing at all."""
        matches = self._families(count=RENDER_BUDGET, prices=(0.55, 0.25, 0.20))

        _, stats = render_snapshot_with_stats(MarketSnapshot(matches=matches))

        assert stats.max_stage == 0
        assert stats.collapsed == 0
        assert stats.named == stats.outcomes

    def test_a_slate_no_stage_can_compact_falls_to_the_per_family_hard_bound(self) -> None:
        """The SECTION-level hard bound, which is the tier that makes the allowance a real ceiling.

        Stage escalation only helps when a family has something cheap to collapse, and a family priced
        at exactly 0.50 has nothing: every rung sits the maximum distance from certainty, so no floor
        reaches it and every stage is a no-op. Eight such families exhaust the escalation loop while
        the section is still over — the shape that rendered 12,951 chars against a committed 10,600
        budget before this tier existed. Each family then falls to `_ladder_hard_bound` at its equal
        share of the allowance.

        `max_stage` is the discriminator: the sentinel says a row fell off the end of the ladder rather
        than stopping at an ordinary stage, and a per-ROW hard bound cannot produce it here because
        every one of these titles fits `LADDER_ROW_MAX_CHARS` on its own.
        """
        matches = self._families(count=RENDER_BUDGET, prices=(0.50,) * 12)

        rendered, stats = render_snapshot_with_stats(MarketSnapshot(matches=matches))

        titles = _ladder_titles(rendered)
        assert len(titles) == RENDER_BUDGET
        assert max(len(title) for title in titles) <= LADDER_ROW_MAX_CHARS, (
            "the fixture must reach the SECTION tier, not the per-row one, or this proves nothing"
        )
        assert stats.max_stage == LADDER_HARD_BOUND_STAGE
        assert stats.ladder_chars <= LADDER_SECTION_MAX_CHARS
        assert stats.named + stats.collapsed == stats.outcomes
        for title in titles:
            assert len(title) <= max(LADDER_SECTION_MAX_CHARS // len(titles), LADDER_MIN_ROW_CHARS)
            assert _ladder_terms(title), "a hard-bounded row still names its most informative outcome"
            assert re.search(r"\+\d+ more \(\d+ priced, \d+\.\d\d summed\)$", title), title

    def test_an_unquoted_outcome_is_the_first_thing_a_hard_bound_drops(self) -> None:
        """A hard bound spends its characters on prices, so an outcome carrying none goes first.

        `_forecast_content` reports -1.0 for an unquoted outcome precisely so it sorts behind every
        priced one at every floor, and the hard bound is the one path that ranks the WHOLE remainder —
        including the unquoted rungs the stage builders segregate. Getting this backwards would spend a
        capped row naming outcomes with nothing to say while counting real prices into the remainder.
        """
        children = tuple(
            MarketChild(title=f"rung {index:03d}", implied_prob_yes=0.50)
            for index in range(200)  # HARNESS-SCAN-EXEMPT-subsampling
        ) + tuple(MarketChild(title=f"unquoted {index}", price_withheld=True) for index in range(3))

        rendered, stats = render_snapshot_with_stats(
            MarketSnapshot(matches=[_row(title="pathological", prob=None, children=children)])
        )

        title = _ladder_titles(rendered)[0]
        assert stats.max_stage == LADDER_HARD_BOUND_STAGE
        assert "unquoted" not in title, "an outcome with no price must not evict a priced one from the row"
        named = _ladder_terms(title)
        assert named
        assert all(price is not None for _, price, _ in named)
        # The remainder's summed price counts the dropped PRICED rungs only, so the three unquoted ones
        # are inside the count and contribute nothing to the sum. The row now STATES that denominator:
        # the count the sum covers used to be invisible, so `+160 more (78.50 summed)` read as 78.50
        # across 160 outcomes when it was 78.50 across 157 — and on a settled ladder the same shape hid
        # rungs realized at 1.00.
        dropped_priced = stats.collapsed - 3
        assert f"+{stats.collapsed} more ({dropped_priced} priced, {dropped_priced * 0.5:.2f} summed)" in title
        assert stats.named + stats.collapsed == stats.outcomes

    def test_a_mixed_open_settled_remainder_states_the_count_its_sum_covers(self) -> None:
        """The M7 shape verbatim: settled rungs inflate the remainder's COUNT but not its SUM.

        `_open_price_total` deliberately excludes settled members (a realized 1.00 is an outcome,
        not a forecast), and `_ladder_content_key` sorts settled behind every open rung, so all
        eight land in the dropped set. Before the denominator was stated, `+N more (S summed)`
        read as S spread across N — hiding that eight of the N were realized at 1.00, an 8.00
        difference between what the figure covered and what the count implied.
        """
        children = tuple(
            MarketChild(title=f"rung {index:03d}", implied_prob_yes=0.50)
            for index in range(200)  # HARNESS-SCAN-EXEMPT-subsampling
        ) + tuple(MarketChild(title=f"Above {index}", implied_prob_yes=1.0, is_resolved=True) for index in range(8))

        rendered, stats = render_snapshot_with_stats(
            MarketSnapshot(matches=[_row(title="pathological", prob=None, children=children)])
        )

        title = _ladder_titles(rendered)[0]
        assert stats.max_stage == LADDER_HARD_BOUND_STAGE
        dropped_priced = stats.collapsed - 8
        # The sum is dropped_priced * 0.50 exactly: had the settled 1.00s leaked in, it would
        # read 8.00 higher against the same count.
        assert f"+{stats.collapsed} more ({dropped_priced} priced, {dropped_priced * 0.5:.2f} summed)" in title
        assert stats.named + stats.collapsed == stats.outcomes


class TestQuoteRangeCell:
    """``LO-HI``: what a ``prob`` cell says when a venue's book implies no price."""

    def test_a_blanked_child_renders_its_book_as_a_range(self) -> None:
        """An empty Kalshi book is bid 0.0000 / ask 1.0000, whose midpoint is a synthetic $0.50 nobody
        quoted. The range says what was actually on the book, and cannot be read as a point
        probability — which the 0.50 it replaces very much could."""
        children = (
            MarketChild(title="quoted rung", implied_prob_yes=0.31),
            MarketChild(title="no book", quote_low=0.0, quote_high=1.0, price_withheld=True),
        )

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[_row(prob=None, children=children)])))

        assert [cell["prob"] for cell in cells[1:]] == ["0.31", "0.00-1.00"]

    def test_a_blanked_child_sorts_behind_every_priced_one(self) -> None:
        """A real price always outranks an absent one, so the full rows go to the outcomes that have
        one. Pre-fix the fabricated 0.50 sorted these to the FRONT of a price-descending render — 59
        archived children rendered that way, against 18 under the order before it."""
        children = (
            MarketChild(title="no book", quote_low=0.0, quote_high=1.0, price_withheld=True),
            MarketChild(title="cheap but real", implied_prob_yes=0.02),
        )

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[_row(prob=None, children=children)])))

        assert [cell["title"] for cell in cells[1:]] == ["cheap but real", "no book"]

    def test_a_blanked_child_with_no_book_renders_a_dash(self) -> None:
        """A venue that publishes no book has nothing to show, and inventing a range would be the same
        class of fabrication as the midpoint."""
        children = (
            MarketChild(title="priced", implied_prob_yes=0.4),
            MarketChild(title="untouched leg", price_withheld=True),
        )

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[_row(prob=None, children=children)])))

        assert cells[2]["prob"] == "-"

    def test_a_range_never_parses_as_a_probability(self) -> None:
        """The property, stated as a property. The legend tells a forecaster to read `LO-HI` as
        unpriced, and that only holds if the cell cannot also be read as this column's two-decimal
        price."""
        probability_shaped = re.compile(r"^-?\d+\.\d{2}$")
        children = (MarketChild(title="wide book", quote_low=0.3, quote_high=1.0, price_withheld=True),)

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[_row(prob=None, children=children)])))

        assert cells[1]["prob"] == "0.30-1.00"
        assert not probability_shaped.match(cells[1]["prob"])

    def test_a_blanked_outcome_reads_as_unpriced_in_the_ladder(self) -> None:
        """The same claim one row down: a ladder term for a blanked outcome carries a dash, so it joins
        the `+N unquoted` group under compaction rather than a summed price group."""
        children = (
            *(
                MarketChild(title=f"real {index}", implied_prob_yes=0.5 - 0.01 * index)
                for index in range(MAX_CHILD_ROWS_PER_MARKET)
            ),
            MarketChild(title="untouched leg", price_withheld=True),
        )

        rendered = render_snapshot(MarketSnapshot(matches=[_row(prob=None, children=children)]))

        assert "untouched leg -" in _ladder_titles(rendered)[0]

    def test_a_refused_family_price_renders_the_parents_own_book_as_a_range(self) -> None:
        """The PARENT row's half of the same claim, which is a different code path from a sub-row's.

        A single-strike Kalshi family quotes that one strike's price as the ROW's, on a row the ranker
        stamped with a relation tier — so an empty book there used to put a synthetic $0.50 in the cell
        a forecaster is told to anchor on. `_row_cells` passes the parent's own `bid`/`ask` through, and
        the parent and the sub-row must render a refusal identically: a reader cannot be asked to learn
        two cell shapes for one fact.
        """
        parent = _row(title="Kalshi single-strike family", prob=None)
        parent.bid, parent.ask = 0.0, 1.0
        parent.price_withheld = True
        child_side = _row(prob=None, children=(MarketChild(title="no book", quote_low=0.0, quote_high=1.0),))

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[parent, child_side])))

        assert cells[0]["prob"] == "0.00-1.00"
        assert cells[0]["prob"] == cells[2]["prob"], "the parent and the sub-row must say it the same way"

    def test_a_priced_row_leads_with_its_price_and_shows_its_book_beside_it(self) -> None:
        """The control: every Kalshi row carries a two-sided book, so a range that outranked a real
        price would replace every price in the table with a range. The price stays first; the book
        rides beside it so a wide one (a 24-61 book behind a 0.42 midpoint) is visible."""
        parent = _row(title="Kalshi binary", prob=0.68)
        parent.bid, parent.ask = 0.66, 0.70

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[parent])))

        assert cells[0]["prob"] == "0.68 (bid-ask 0.66-0.70)"

    def test_the_legend_names_both_new_cell_shapes(self) -> None:
        """A legend that omits a shape a cell can hold teaches forecasters to guess at it, and the
        guess this range replaces was "the market says 50/50"."""
        rendered = render_snapshot(MarketSnapshot(matches=[_row()]))

        assert "`PRICE (bid-ask LO-HI)`" in rendered
        assert "`LO-HI` alone means there is a book but no usable price, not 50/50" in rendered
        assert "[remaining N]" in rendered
        assert "counted set with its summed price" in rendered


class TestChildRenderStatsPayload:
    """``ChildRenderStats``: the ``MARKET_CHILD_RENDER`` marker's fields, and what they have to mean."""

    def test_an_empty_snapshot_reports_zeroes(self) -> None:
        text, stats = render_snapshot_with_stats(MarketSnapshot(matches=[]))

        assert text == ""
        assert (stats.families, stats.outcomes, stats.ladder_rows, stats.withheld) == (0, 0, 0, 0)

    def test_the_counts_match_the_rendered_table(self) -> None:
        """Counted off the render rather than asserted from the inputs, so the marker cannot drift away
        from what a forecaster read."""
        matches = [
            _row(
                title="ladder",
                prob=None,
                children=tuple(MarketChild(title=f"rung {index}", implied_prob_yes=0.5) for index in range(14)),
            ),
            _row(title="binary", prob=0.42),
        ]

        rendered, stats = render_snapshot_with_stats(MarketSnapshot(matches=matches))

        assert stats.families == 1
        assert stats.outcomes == 14
        assert stats.full_rows == len(_full_sub_rows(rendered))
        assert stats.ladder_rows == len(_ladder_titles(rendered))
        assert stats.ladder_chars == sum(len(title) for title in _ladder_titles(rendered))

    def test_withheld_counts_parents_and_children(self) -> None:
        """The field the Kalshi spread threshold is calibrated against, so it has to see both places a
        price can be refused: a single-strike FAMILY quoting its one strike, and a strike sub-row."""
        parent = _row(
            prob=None,
            children=(
                MarketChild(title="a", implied_prob_yes=0.4),
                MarketChild(title="b", price_withheld=True, quote_low=0.0, quote_high=1.0),
            ),
        )
        parent.price_withheld = True

        _, stats = render_snapshot_with_stats(MarketSnapshot(matches=[parent]))

        assert stats.withheld == 2

    def test_a_healthy_snapshot_reports_no_withheld_prices(self) -> None:
        _, stats = render_snapshot_with_stats(
            MarketSnapshot(matches=[_row(prob=None, children=(MarketChild(title="a", implied_prob_yes=0.4),))])
        )

        assert stats.withheld == 0

    def test_the_title_cap_bounds_a_ladder_term(self) -> None:
        """A term's label takes the same cap a full sub-row's does, so one pathological outcome label
        cannot blow the row."""
        children = tuple(
            MarketChild(title="L" * 500, implied_prob_yes=0.5 - 0.01 * index)
            for index in range(MAX_CHILD_ROWS_PER_MARKET + 2)
        )

        rendered = render_snapshot(MarketSnapshot(matches=[_row(prob=None, children=children)]))

        title = _ladder_titles(rendered)[0]
        assert f"{'L' * CHILD_TITLE_MAX_CHARS} 0.40" in title
        assert "L" * (CHILD_TITLE_MAX_CHARS + 1) not in title
