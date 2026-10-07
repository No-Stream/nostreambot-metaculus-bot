"""The seam's formatter, a thin delegate over the market-retrieval renderer.

What belongs here is only what the SEAM decides: that an empty snapshot renders nothing, that the
degraded-ranking marker is derived from the snapshot's own ranking token rather than passed in,
which preamble a rendered row's top tier selects, and the liquidity-label thresholds. The render
itself — columns, cell formatting, budgets — is tested in `tests/test_market_retrieval_rendering.py`.

The `market_row` helper these use lives in `tests/market_retrieval_fakes.py`.
"""

from __future__ import annotations

import logging

import pytest

from metaculus_bot.research.market_retrieval.ranking import DEGRADED_RANKING_MARKER
from metaculus_bot.research.market_retrieval.rendering import TABLE_COLUMNS
from metaculus_bot.research.market_retrieval.types import MarketChild, MarketMatch
from metaculus_bot.research.prediction_market import MarketSnapshot, _liquidity_label, format_snapshot_for_research
from tests.market_retrieval_fakes import market_row as _row


class TestFormatterDelegate:
    def test_an_empty_snapshot_renders_nothing(self):
        """The early return is what produces `status="empty"` downstream and the
        attempted-vs-succeeded distinction residual analysis reads off the archive. Under ranked
        selection a zero-row render is a legitimate outcome, so this is a hot path."""
        assert format_snapshot_for_research(MarketSnapshot(matches=[])) == ""

    def test_a_deliberate_zero_row_ranking_renders_the_no_relevant_market_line(self):
        """The q45200 shape: a healthy pool, a successful ranking call, an empty answer. Before
        the notice, the section vanished wholesale and read exactly like a provider outage while
        the forecaster prompt still shipped the market-weighting clauses. The notice quotes the
        pool size so the forecaster knows how much was reviewed."""
        rendered = format_snapshot_for_research(MarketSnapshot(matches=[], sources={"ranking": "ok(0)"}, pool_size=381))

        assert "No sufficiently relevant market among 381 candidates" in rendered
        assert "not a provider outage" in rendered

    @pytest.mark.parametrize(
        ("sources", "pool_size"),
        [
            ({"ranking": "error(RankingUnusable)"}, 381),  # ranker died — nothing was judged
            ({"ranking": "none"}, 0),  # empty pool — nothing to rank
            ({"snapshot": "error(timeout)"}, 0),  # whole-provider failure
            ({}, 381),  # no ranking token at all
        ],
    )
    def test_every_non_deliberate_zero_still_renders_nothing(self, sources: dict[str, str], pool_size: int):
        """The gate must stay narrow: only the ranker's own empty answer over a non-empty pool
        earns the notice. A failure path claiming 'none was judged to bear' would launder an
        outage into a considered judgment — strictly worse than the silent section it replaces."""
        assert format_snapshot_for_research(MarketSnapshot(matches=[], sources=sources, pool_size=pool_size)) == ""

    def test_the_degraded_marker_is_derived_from_the_snapshots_own_ranking_token(self):
        """Derived, not passed in, so the render is reproducible from an archived snapshot
        alone — which matters because `record_raw_research` archives the snapshot and a replay
        tool renders from it."""
        rows = [_row("A market", tier="")]

        healthy = format_snapshot_for_research(MarketSnapshot(matches=rows, sources={"ranking": "ok(1)"}))
        degraded = format_snapshot_for_research(MarketSnapshot(matches=rows, sources={"ranking": "error(x)"}))

        assert DEGRADED_RANKING_MARKER not in healthy
        assert degraded.startswith(DEGRADED_RANKING_MARKER)

    def test_a_missing_ranking_token_reads_as_degraded(self):
        """The conservative direction: an unlabelled snapshot may be in retrieval order, and
        claiming evidential order falsely is worse than a marker on a healthy table."""
        assert DEGRADED_RANKING_MARKER in format_snapshot_for_research(MarketSnapshot(matches=[_row("A market")]))

    def test_cells_are_addressable_by_header_name(self):
        """Read by HEADER NAME, never by index. The previous version of this test carried a
        comment enumerating cell positions, and every column addition silently shifted them."""
        snapshot = MarketSnapshot(matches=[_row("Will X win?", platform="predictit")], sources={"ranking": "ok(1)"})

        formatted = format_snapshot_for_research(snapshot)
        header = next(line for line in formatted.splitlines() if line.startswith("| platform |"))
        columns = [cell.strip() for cell in header.strip("|").split("|")]
        row = next(line for line in formatted.splitlines() if line.startswith("| predictit"))
        cells = dict(zip(columns, [cell.strip() for cell in row.strip("|").split("|")], strict=True))

        assert columns == list(TABLE_COLUMNS)
        assert cells["signal"] == "no-liquidity-data"
        assert cells["status"] == "open"
        assert cells["relation"] == "same_quantity_other_cut"
        assert cells["why"] == "same series, adjacent month"

    def test_the_preamble_follows_the_ranker_top_tier(self):
        """Strong when a rendered row measures the same quantity, neutral otherwise. The selector
        is the ranker's own grade, replacing the content-overlap + confidence bar it retired."""
        strong = format_snapshot_for_research(
            MarketSnapshot(matches=[_row("A", tier="same_quantity_same_date")], sources={"ranking": "ok(1)"})
        )
        neutral = format_snapshot_for_research(
            MarketSnapshot(matches=[_row("A", tier="weak")], sources={"ranking": "ok(1)"})
        )

        assert "evidence as strong as its liquidity (`signal`) allows" in strong.lower()
        assert "may all be off-topic" in neutral.lower()
        assert "as strong as its liquidity" not in neutral.lower()
        # The retired vocabulary must not come back: rows are chosen by a model reading each
        # market's rules, not by word overlap.
        for rendered in (strong, neutral):
            assert "fuzzy" not in rendered.lower()
            assert "verify-carefully" not in rendered
            assert "likely-relevant" not in rendered

    @pytest.mark.parametrize(
        ("total_volume", "expected"),
        [
            (999.0, "thin"),
            (1_000.0, "decent"),
            (9_999.0, "decent"),
            (10_000.0, "strong"),
            (99_999.0, "strong"),
            (100_000.0, "deep"),
        ],
    )
    def test_liquidity_label_real_money_thresholds(self, total_volume, expected):
        row = _row("x")
        row.total_volume = total_volume
        row.open_interest = None
        assert _liquidity_label(row) == expected

    @pytest.mark.parametrize(("num_bettors", "expected"), [(5, "thin"), (50, "decent"), (200, "high")])
    def test_liquidity_label_manifold_bettor_thresholds(self, num_bettors, expected):
        row = _row("x", platform="manifold")
        row.total_volume = None
        row.open_interest = None
        row.num_bettors = num_bettors
        assert _liquidity_label(row) == expected

    @pytest.mark.parametrize("num_bettors", [5, 50, 200])
    def test_a_manifold_price_with_zero_own_volume_is_thin_at_every_bettor_count(self, num_bettors):
        """Zero own volume beats the bettor pool. The pool is a MARKET-level figure — Manifold
        publishes no per-answer count, so every child inherits its parent's — and a market with 200
        bettors was labelling its untouched answers `high` when nobody had bet on them at all (62 of
        399 archived Manifold children, 15.5%, read decent/high with zero own volume, right beside
        the blank price `_priced_or_none` had just refused them)."""
        row = _row("x", platform="manifold")
        row.total_volume = 0.0
        row.open_interest = None
        row.num_bettors = num_bettors
        assert _liquidity_label(row) == "thin"

    def test_a_manifold_price_with_absent_volume_still_reads_the_bettor_pool(self):
        """The complement, and the reason the gate is `is not None` rather than falsy: a MISSING
        volume is not evidence of no trading, so a payload shape that omits the field must not have
        every one of its rows demoted to `thin`. Same rule `_priced_or_none` applies."""
        row = _row("x", platform="manifold")
        row.total_volume = None
        row.open_interest = None
        row.num_bettors = 200
        assert _liquidity_label(row) == "high"


class TestChildRenderMarker:
    """``MARKET_CHILD_RENDER``: emitted by the seam, beside its ``MARKET_RANKING`` sibling.

    The line is the post-ship instrument for the 2026-08-25 no-manufactured-price change: the Kalshi
    no-price spread threshold is calibrated on eleven fixture strikes, so ``withheld=`` is how its prod
    incidence becomes a query instead of a guess. It lives at the SEAM rather than in the renderer
    because the renderer has no qid and no logger of its own, and the marker has to be keyed to a
    question to be useful in the telemetry archive.
    """

    def _family(self, count: int) -> MarketMatch:
        row = _row("A strike family", platform="kalshi")
        row.implied_prob_yes = None
        row.children = tuple(
            MarketChild(title=f"rung {index}", implied_prob_yes=0.5 - 0.01 * index) for index in range(count)
        )
        return row

    def test_the_marker_is_emitted_for_a_rendered_snapshot(self, caplog):
        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.prediction_market"):
            format_snapshot_for_research(
                MarketSnapshot(matches=[self._family(14)], sources={"ranking": "ok(1)"}), qid=45189
            )

        lines = [record.message for record in caplog.records if record.message.startswith("MARKET_CHILD_RENDER:")]
        assert len(lines) == 1
        assert "question=45189" in lines[0]
        assert "families=1" in lines[0]
        assert "outcomes=14" in lines[0]
        assert "ladder_rows=1" in lines[0]

    def test_the_marker_reports_withheld_prices(self, caplog):
        """The field the whole marker exists for, on both places a price can be refused."""
        family = self._family(4)
        family.price_withheld = True
        family.children = (
            *family.children,
            MarketChild(title="no book", quote_low=0.0, quote_high=1.0, price_withheld=True),
        )

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.prediction_market"):
            format_snapshot_for_research(MarketSnapshot(matches=[family], sources={"ranking": "ok(1)"}), qid=1)

        line = next(r.message for r in caplog.records if r.message.startswith("MARKET_CHILD_RENDER:"))
        assert "withheld=2" in line

    def test_no_marker_when_nothing_renders(self, caplog):
        """A snapshot that renders no table has no child accounting to report, and a line of zeroes
        would dilute the field distributions the marker exists to measure."""
        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.prediction_market"):
            format_snapshot_for_research(MarketSnapshot(matches=[], sources={"ranking": "ok(0)"}, pool_size=381), qid=1)

        assert not [r for r in caplog.records if r.message.startswith("MARKET_CHILD_RENDER:")]

    def test_the_marker_has_no_spaces_inside_a_field(self, caplog):
        """The harvester splits on whitespace, so a value carrying a space would shift every later
        field into the wrong key."""
        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.prediction_market"):
            format_snapshot_for_research(MarketSnapshot(matches=[self._family(14)]), qid=None)

        line = next(r.message for r in caplog.records if r.message.startswith("MARKET_CHILD_RENDER:"))
        fields = line.removeprefix("MARKET_CHILD_RENDER: ").split(" ")
        assert all(field.count("=") == 1 for field in fields), fields
        assert [field.split("=")[0] for field in fields] == [
            "question",
            "families",
            "full_rows",
            "ladder_rows",
            "outcomes",
            "named",
            "collapsed",
            "withheld",
            "max_stage",
            "ladder_chars",
        ]
