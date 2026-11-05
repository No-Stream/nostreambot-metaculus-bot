"""``MarketMatch``'s construction shape and its archive contract.

``MarketMatch`` is ADDITIVE-ONLY, and three consumers are why: ``raw_log`` archives the whole
snapshot through ``dataclasses.asdict`` under an envelope whose ``schema_version`` is shared
across every provider (so a removed field changes the archive shape with no version to bump),
``provider_health`` reads the liquidity fields by ``getattr`` name, and several test sites
construct the row with twelve positional arguments. These tests pin all three: the positional
prefix, the defaults on every field added since, and that the whole row survives ``asdict``
into JSON.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime

from metaculus_bot.research.market_retrieval.types import (
    MarketChild,
    MarketMatch,
    MarketSnapshot,
    ScalarEstimate,
    SettlementSource,
)

_CLOSE = datetime(2026, 8, 31, tzinfo=UTC)


def _positional_row() -> MarketMatch:
    """The row exactly as the venue parsers and six test sites build it: twelve positional args."""
    return MarketMatch(
        "kalshi",
        "Australia unemployment rate above 4.2% in July",
        "https://kalshi.com/markets/kxauunemp",
        0.62,
        0.60,
        0.64,
        0.04,
        1_200.0,
        _CLOSE,
        False,
        1.0,
        "Settles on the ABS seasonally adjusted rate.",
    )


class TestMarketMatchShape:
    def test_the_twelve_positional_arguments_still_bind_the_same_fields(self):
        """Six test sites construct the row positionally, so the prefix order is a contract."""
        match = _positional_row()
        assert match.platform == "kalshi"
        assert match.market_title == "Australia unemployment rate above 4.2% in July"
        assert match.market_url == "https://kalshi.com/markets/kxauunemp"
        assert (match.implied_prob_yes, match.bid, match.ask, match.spread) == (0.62, 0.60, 0.64, 0.04)
        assert (match.volume_24h, match.close_time, match.is_resolved) == (1_200.0, _CLOSE, False)
        assert match.match_confidence == 1.0
        assert match.raw_rules == "Settles on the ABS seasonally adjusted rate."

    def test_every_field_added_since_defaults_to_absent(self):
        """A row built by a venue parser that knows nothing about ranking must still be valid;
        the ranker fills the relation fields later, and the archive must read an unranked row
        as unranked rather than as rank 0 of tier ''."""
        match = _positional_row()
        assert (match.total_volume, match.liquidity, match.open_interest, match.num_bettors) == (None,) * 4
        assert match.venue_market_id == ""
        assert match.relation_tier == ""
        assert match.relevance_label == ""
        assert match.rank is None
        assert match.retrieval_channel == ""
        assert match.sub_title == ""
        assert match.settlement_sources == ()
        assert match.top_answers == ()
        assert match.children == ()
        assert match.scalar_estimate is None
        assert match.price_withheld is False
        assert match.tier_cap_note == ""


class TestArchiveRoundTrip:
    """``raw_log`` serializes the snapshot with ``dataclasses.asdict`` then JSON. Anything on the
    row that ``asdict`` cannot walk, or that JSON cannot encode, breaks the raw-research archive
    for the whole provider rather than for one field."""

    def test_asdict_walks_nested_settlement_sources(self):
        match = _positional_row()
        match.venue_market_id = "KXAUUNEMP-26JUL"
        match.relation_tier = "same_quantity_same_date"
        match.relevance_label = "near-identical: same ABS series, same month"
        match.rank = 0
        match.retrieval_channel = "settlement_join"
        match.sub_title = "Above 4.2%"
        match.settlement_sources = (
            SettlementSource(name="Australian Bureau of Statistics", url="https://www.abs.gov.au/"),
        )

        as_dict = dataclasses.asdict(MarketSnapshot(matches=[match], sources={"kalshi": "ok(1)"}))
        row = as_dict["matches"][0]
        # `asdict` preserves the container type, so a tuple field stays a tuple; JSON encodes it
        # as an array either way (pinned in the round-trip test below).
        assert row["settlement_sources"] == (
            {"name": "Australian Bureau of Statistics", "url": "https://www.abs.gov.au/"},
        )
        assert row["rank"] == 0
        assert row["relation_tier"] == "same_quantity_same_date"
        assert as_dict["sources"] == {"kalshi": "ok(1)"}

    def test_the_archived_row_is_json_encodable_once_datetimes_are_stringified(self):
        match = _positional_row()
        match.settlement_sources = (SettlementSource(name="BLS", url="https://www.bls.gov/cpi/"),)
        encoded = json.loads(json.dumps(dataclasses.asdict(MarketSnapshot(matches=[match])), default=str))
        assert encoded["matches"][0]["close_time"].startswith("2026-08-31")
        assert encoded["matches"][0]["settlement_sources"] == [{"name": "BLS", "url": "https://www.bls.gov/cpi/"}]
        assert encoded["sources"] == {}

    def test_multi_outcome_answers_survive_asdict_and_json_as_pairs(self):
        """A multi-outcome row's answers are its ONLY price, so losing them in the archive would
        make every such row read as priceless in residual analysis. Nested plain tuples, so
        `asdict` leaves them alone and JSON writes them as arrays of two."""
        match = _positional_row()
        match.platform = "manifold"
        match.implied_prob_yes = None  # what Manifold publishes for a multi-outcome market
        match.top_answers = (("Over $4.60", 0.4992), ("Over $4.65", 0.3723))

        as_dict = dataclasses.asdict(MarketSnapshot(matches=[match]))
        assert as_dict["matches"][0]["top_answers"] == (("Over $4.60", 0.4992), ("Over $4.65", 0.3723))

        encoded = json.loads(json.dumps(as_dict, default=str))
        assert encoded["matches"][0]["top_answers"] == [["Over $4.60", 0.4992], ["Over $4.65", 0.3723]]
        assert encoded["matches"][0]["implied_prob_yes"] is None

    def test_multi_outcome_children_survive_asdict_and_json_as_objects(self):
        """The sub-rows are where a multi-outcome market's prices live, so losing them in the archive
        would make residual analysis read every strike family, event and ballot as priceless — the
        exact blind spot the rendering change exists to end. A nested dataclass rather than plain
        pairs, because a child carries ten fields; `asdict` walks it the way it walks
        `SettlementSource`.

        The last three (`quote_low`, `quote_high`, `price_withheld`) are the 2026-08-25
        no-manufactured-price change, and they are why the whole dict is asserted rather than a few
        keys: `price_withheld` is what separates "this repo refused the venue's fabricated 0.50" from
        "the venue quoted nothing", and an archive that lost it could not tell a future reader which
        prices a forecaster actually saw.
        """
        match = _positional_row()
        match.implied_prob_yes = None  # what a market with several outcomes publishes
        match.children = (
            MarketChild(title="Before Nov 1, 2026", implied_prob_yes=0.175, total_volume=45_000.0, close_time=_CLOSE),
            MarketChild(title="Before Jul 1, 2026", implied_prob_yes=0.031, is_resolved=True),
            MarketChild(title="Before Jan 1, 2027", quote_low=0.0, quote_high=1.0, price_withheld=True),
        )

        as_dict = dataclasses.asdict(MarketSnapshot(matches=[match]))
        assert as_dict["matches"][0]["children"][0]["title"] == "Before Nov 1, 2026"
        assert as_dict["matches"][0]["children"][0]["implied_prob_yes"] == 0.175

        encoded = json.loads(json.dumps(as_dict, default=str))
        assert encoded["matches"][0]["children"][1] == {
            "title": "Before Jul 1, 2026",
            "implied_prob_yes": 0.031,
            "total_volume": None,
            "open_interest": None,
            "num_bettors": None,
            "is_resolved": True,
            # `json.dumps(default=...)` is only consulted for values the encoder cannot handle, so a
            # `None` close date encodes as JSON null while a real one goes through `str`.
            "close_time": None,
            "quote_low": None,
            "quote_high": None,
            "price_withheld": False,
            "volume_24h": None,
            "liquidity": None,
            "currency": "",
            "num_traders": None,
            "last_trade_time": None,
            "last_update_time": None,
            "price_change_1d": None,
            "price_change_since_previous": None,
            "price_change_1w": None,
        }
        assert encoded["matches"][0]["children"][0]["close_time"].startswith("2026-08-31")
        withheld = encoded["matches"][0]["children"][2]
        assert (withheld["implied_prob_yes"], withheld["quote_low"], withheld["quote_high"]) == (None, 0.0, 1.0)
        assert withheld["price_withheld"] is True

    def test_a_withheld_parent_price_survives_asdict(self):
        """The PARENT's own flag, on the single-strike-family case. Without it an archived row reading
        `implied_prob_yes: null` with a two-sided `bid`/`ask` is indistinguishable from a venue that
        published no price — and that row is exactly the one the Kalshi spread threshold is calibrated
        against, so the archive has to be able to answer how often it fired."""
        match = _positional_row()
        match.implied_prob_yes = None
        match.bid, match.ask = 0.0, 1.0
        match.price_withheld = True

        row = json.loads(json.dumps(dataclasses.asdict(MarketSnapshot(matches=[match])), default=str))["matches"][0]

        assert row["price_withheld"] is True
        assert (row["bid"], row["ask"]) == (0.0, 1.0), "the book is the evidence for the refusal"

    def test_a_scalar_estimate_survives_asdict_and_json_as_an_object(self):
        """A scalar market's value is its ONLY price — `implied_prob_yes` is empty on such a row by
        construction — so losing this field in the archive would make residual analysis read every
        scalar market as priceless, the same blind spot the children fix closed. A nested dataclass
        like `SettlementSource`, so `asdict` walks it without help.

        The bounds ride along because they are what makes the value interpretable after the fact:
        121 against a maximum of 250 and 121 against a maximum of 130 are different signals, and the
        archive is the only place a later reader can recover which one it was.
        """
        match = _positional_row()
        match.platform = "manifold"
        match.implied_prob_yes = None  # what a scalar market's row carries instead of a price
        match.scalar_estimate = ScalarEstimate(value=120.96691732988944, minimum=0.0, maximum=250.0, is_log_scale=False)

        as_dict = dataclasses.asdict(MarketSnapshot(matches=[match]))
        assert as_dict["matches"][0]["scalar_estimate"] == {
            "value": 120.96691732988944,
            "minimum": 0.0,
            "maximum": 250.0,
            "is_log_scale": False,
        }

        encoded = json.loads(json.dumps(as_dict, default=str))
        assert encoded["matches"][0]["scalar_estimate"]["value"] == 120.96691732988944
        assert encoded["matches"][0]["implied_prob_yes"] is None

    def test_a_capped_tier_archives_both_the_capped_grade_and_what_the_ranker_said(self):
        """The staleness cap's whole record. `relation_tier` has to stay one of the four vocabulary
        words (`STRONG_TIERS` membership picks the rendered preamble and every tier-conditioned
        residual cut tests the cell by equality), so the demotion's reason lives in its own field —
        which also keeps `relevance_label` the model's verbatim phrase, and therefore keeps "what the
        ranker said about a row our arithmetic overruled" recoverable from the archive.

        The note no longer spells out the withdrawn grade (it read "(ranker said
        same_quantity_same_date)" until the wording was cut back for the rendered cell), and the
        grade is still recoverable from these three fields: only the top tier is ever capped, so a
        non-empty `tier_cap_note` beside a `same_quantity_other_cut` grade says the ranker said
        `same_quantity_same_date` and our arithmetic refused it."""
        match = _positional_row()
        match.relation_tier = "same_quantity_other_cut"
        match.relevance_label = "conditional country market"
        match.tier_cap_note = "demoted from same-date: closed 163d before the question opened"

        row = json.loads(json.dumps(dataclasses.asdict(MarketSnapshot(matches=[match])), default=str))["matches"][0]
        assert row["relation_tier"] == "same_quantity_other_cut"
        assert row["relevance_label"] == "conditional country market"
        assert row["tier_cap_note"] == "demoted from same-date: closed 163d before the question opened"

    def test_the_snapshots_forecast_time_survives_the_archive(self):
        """It is what the render dates its staleness disclosures against, so a replay that could not
        read it back would either lose the disclosures or re-age every row against its own clock."""
        encoded = json.loads(
            json.dumps(
                dataclasses.asdict(MarketSnapshot(matches=[_positional_row()], forecast_time=_CLOSE)), default=str
            )
        )
        assert encoded["forecast_time"].startswith("2026-08-31")
        assert dataclasses.asdict(MarketSnapshot())["forecast_time"] is None

    def test_a_row_never_archives_both_a_probability_and_a_scalar_value(self):
        """The invariant the venue parser enforces, asserted where the archive would expose a break.

        Two numbers in one row's price slot is exactly the ambiguity this type split exists to
        remove: a reader could not tell which one the forecaster saw. `parse_manifold_matches` fills
        one branch or the other, so a row carrying both would mean that branch had been flattened.
        """
        priced = _positional_row()
        scalar = _positional_row()
        scalar.implied_prob_yes = None
        scalar.scalar_estimate = ScalarEstimate(value=120.967, minimum=0.0, maximum=250.0)

        for match in (priced, scalar):
            row = dataclasses.asdict(MarketSnapshot(matches=[match]))["matches"][0]
            assert (row["implied_prob_yes"] is None) != (row["scalar_estimate"] is None)
