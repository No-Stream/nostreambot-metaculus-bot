"""Tests for the unsupported-attribution check on Gemini grounded-search output.

Every fixture below is a shape the 323-section archived corpus actually contains (the
tier-tag syntax census and the false-strip review are in
``scratch/next_season_bundle_2026-09/item4_attribution_check/VALIDATION.md``), except the
out-of-scope cases, which pin the boundary the check must not cross.
"""

from metaculus_bot.research.bracket_groups import (
    BRACKET_GROUP_RE,
    iter_group_items,
    join_group_items,
    rebuild_group,
)
from metaculus_bot.research.gemini_attribution import (
    UNVERIFIED_ATTRIBUTION_MARKER,
    rewrite_unsupported_attributions,
)


def _rewrite(text: str, labels: list[str]) -> str:
    return rewrite_unsupported_attributions(text, labels).text


class TestSupportedAttributionsAreKept:
    """Matching is loose in the KEEP direction: an outlet missing from the grounded
    domains does not prove the fact wrong, so a false strip (real provenance discarded)
    costs more than a false keep (a tag left standing). Each case below is one of the
    five support rules, and each rule closes a false-strip class the corpus contains.
    """

    def test_name_that_concatenates_into_the_domain(self) -> None:
        text = "Play resumed at 14:00 [C: Golf Channel]."
        assert _rewrite(text, ["golfchannel.com"]) == text

    def test_exact_outlet_name(self) -> None:
        text = "The ministry confirmed the recall [B: Reuters]."
        assert _rewrite(text, ["reuters.com", "usda.gov"]) == text

    def test_extra_words_in_the_domain(self) -> None:
        text = "Turnout fell again [B: The Guardian]."
        assert _rewrite(text, ["guardian.co.uk"]) == text

    def test_sub_brand_against_a_parent_domain(self) -> None:
        # q44855 / q45245 / q45082 shapes: the record names the parent domain and the
        # model names the section of the site it read.
        for name, label in (
            ("LSE Blogs", "lse.ac.uk"),
            ("Chosunbiz", "chosun.com"),
            ("iHeartRadio", "iheart.com"),
        ):
            text = f"Support narrowed to two points [C: {name}]."
            assert _rewrite(text, [label]) == text, name

    def test_abbreviation_of_a_grounded_outlet(self) -> None:
        # q45186 / q44954 / q44855 / q45082 shapes: the abbreviation IS the outlet whose
        # domain we grounded, so flagging it would discard real provenance.
        for name, label in (
            ("WaPo", "washingtonpost.com"),
            ("WashPost", "washingtonpost.com"),
            ("RCP", "realclearpolling.com"),
            ("GEF", "global-energy-flow.com"),
        ):
            text = f"The tally was revised upward [B: {name}]."
            assert _rewrite(text, [label]) == text, name

    def test_slash_joined_pair_where_one_half_is_grounded(self) -> None:
        # ``Reuters/AP`` on a record holding apnews.com is a wire story with a second
        # outlet mentioned, not a fabricated attribution.
        text = "Both agencies carried the statement [B: Reuters/AP]."
        assert _rewrite(text, ["apnews.com"]) == text

    def test_domain_that_abbreviates_the_outlet(self) -> None:
        # The abbreviation relation runs both ways. q44880 / q45172 shapes: the outlet's own
        # domain is its initials or a clipped form of its name.
        for name, label in (
            ("Times of Central Asia", "timesca.com"),
            ("Gateway Investment Advisers", "gia.com"),
        ):
            text = f"The bill passed its second reading [B: {name}]."
            assert _rewrite(text, [label]) == text, name

    def test_title_side_of_a_label_counts(self) -> None:
        # Every archived label is a bare domain, matching the formatter's source-domain labels.
        # The formatter previously rendered ``<title> — <domain>`` when a chunk carried both,
        # and a title routinely names
        # the outlet the domain hides.
        text = "Coverage began Tuesday [C: Golf Channel]."
        assert _rewrite(text, ["Golf Channel highlights — sports.example.com"]) == text


class TestUnsupportedAttributionsAreRewritten:
    def test_single_tag_becomes_the_marker(self) -> None:
        # cutB's worked example, q44953: ``[A: NASA]`` for the path of totality over a
        # source list that never mentions NASA.
        out = _rewrite(
            "Reykjavik lies in the path of totality [A: NASA].",
            ["perlan.is", "guidetoiceland.is", "timeanddate.com"],
        )
        assert out == f"Reykjavik lies in the path of totality [{UNVERIFIED_ATTRIBUTION_MARKER}]."

    def test_the_tier_grade_goes_with_the_outlet(self) -> None:
        # The grade is an authority claim read off the outlet, so it cannot outlive it —
        # the forecaster prompts weight by tier.
        assert "A:" not in _rewrite("Filings closed Friday [A: NASA].", ["perlan.is"])

    def test_several_unsupported_names_in_one_group_collapse_to_one_marker(self) -> None:
        # q38195 shape. A second "we could not verify this" says nothing the first did not.
        out = _rewrite("Efficiency peaks by 2029 [B: Forbes; C: MIT Sloan].", ["aft.org"])
        assert out == f"Efficiency peaks by 2029 [{UNVERIFIED_ATTRIBUTION_MARKER}]."

    def test_mixed_group_keeps_the_supported_outlet(self) -> None:
        # q44856 shape: ``[A: FDA, B: Food Safety Magazine]`` on a record holding fda.gov.
        out = _rewrite(
            "The agency named the supplier [A: FDA, B: Food Safety Magazine].",
            ["fda.gov", "contagionlive.com"],
        )
        assert out == f"The agency named the supplier [A: FDA, {UNVERIFIED_ATTRIBUTION_MARKER}]."

    def test_semicolon_separated_group_keeps_its_own_separator(self) -> None:
        # q38195 shape: two tier tags in one bracket, semicolon-delimited.
        out = _rewrite("The contract bans automation [A: ILA; C: Sea news].", ["ila.org"])
        assert out == f"The contract bans automation [A: ILA; {UNVERIFIED_ATTRIBUTION_MARKER}]."

    def test_continuation_name_under_one_tier_is_checked_too(self) -> None:
        # q44879 shape: ``[D: GrackerAI, siberX]`` names two outlets under one grade.
        out = _rewrite("Two vendors published advisories [D: GrackerAI, siberX].", ["blackhat.com"])
        assert out == f"Two vendors published advisories [{UNVERIFIED_ATTRIBUTION_MARKER}]."

    def test_marker_lands_where_the_first_unsupported_name_was(self) -> None:
        out = _rewrite("Prices held [B: Reuters, C: Economic Times].", ["economictimes.com"])
        assert out == f"Prices held [{UNVERIFIED_ATTRIBUTION_MARKER}, C: Economic Times]."


class TestTheLooseningRulesStayBounded:
    """The keep-biased rules must not credit an outlet the record genuinely does not name,
    or the check does nothing. Each case here is a near-miss the corpus contains, and each
    is a DIFFERENT outlet from the domain it nearly matches.
    """

    def test_a_sibling_tabloid_is_not_credited(self) -> None:
        # The domain-abbreviates-name rule is anchored at the name's first token, so
        # ``dailystar`` cannot be read as ``daily`` + a prefix of ``express``.
        assert "unverified attribution" in _rewrite("Readers were told [B: Daily Express].", ["dailystar.co.uk"])

    def test_a_shared_topic_word_is_not_credited(self) -> None:
        for name, label in (
            ("Barchart", "ycharts.com"),
            ("Helsinki Times", "taipeitimes.com"),
            ("World Record Academy", "guinnessworldrecords.com"),
            ("The Climate Brink", "climateimpactcompany.com"),
            ("Traders Union", "tradingeconomics.com"),
        ):
            out = _rewrite(f"The figure was published [C: {name}].", [label])
            assert "unverified attribution" in out, name

    def test_a_tld_is_not_an_identity(self) -> None:
        # ``.ai`` is a TLD; crediting it made three unrelated outlets read as grounded.
        assert "unverified attribution" in _rewrite("The tool shipped [B: Kie.ai].", ["orcarouter.ai"])

    def test_known_residual_a_station_under_a_parent_domain_is_still_marked(self) -> None:
        """The one arguable case in the corpus, pinned deliberately.

        Google reported the chunk's domain as ``iheart.com`` while the model named the
        station, ``NewsRadio WFLA`` — plausibly the same page, but nothing in what we hold
        says so, and no general rule recovers a subdomain the SDK did not give us (2
        occurrences in 681 archived attributions). If a future change closes this class,
        this test failing is the intended prompt to re-read the reasoning rather than a
        regression.
        """
        out = _rewrite("The show reported it [C: NewsRadio WFLA].", ["iheart.com"])
        assert "unverified attribution" in out


class TestWhatTheCheckLeavesAlone:
    def test_academic_categories_are_not_publisher_attributions(self) -> None:
        for category in ("academic", "peer-reviewed", "academic / peer-reviewed"):
            text = f"The paper models the age distribution [B: {category}]."
            result = rewrite_unsupported_attributions(text, ["demographic-research.org"])
            assert result.text == text
            assert (result.tagged, result.unsupported, result.groups_rewritten) == (0, 0, 0)

    def test_academic_category_does_not_exempt_an_unbacked_publisher(self) -> None:
        text = "The paper models lifespan [B: academic / peer-reviewed]. The report agrees [B: Reuters]."
        result = rewrite_unsupported_attributions(text, ["demographic-research.org"])
        assert result.text == (
            "The paper models lifespan [B: academic / peer-reviewed]. The report agrees [unverified attribution]."
        )
        assert (result.tagged, result.unsupported, result.groups_rewritten) == (1, 1, 1)

    def test_generic_tier_words(self) -> None:
        # ``official`` / ``aggregator`` name a class of source, not an outlet, so there is
        # nothing in the grounding record to check them against. 307 of the corpus's 790
        # tier items are one of these.
        for text in (
            "FSIS posted the notice [A: official].",
            "Coverage was syndicated [C: aggregator].",
            "Both carried it [A: official, C: aggregator].",
            "A newswire moved it [B: wire service].",
            "Posted to a feed [D: social].",
            "Filed via the desk [A: official/wire].",
        ):
            assert _rewrite(text, ["perlan.is"]) == text, text

    def test_a_generic_tag_beside_an_unsupported_outlet(self) -> None:
        # q44802/q44808 shape, the corpus's single commonest group: the generic half stays
        # verbatim and only the named half is marked.
        out = _rewrite("The notice was revised [A: official, B: Reuters].", ["usda.gov"])
        assert out == f"The notice was revised [A: official, {UNVERIFIED_ATTRIBUTION_MARKER}]."

    def test_our_own_spliced_citation_markers(self) -> None:
        text = "Alpha.[1] Beta.[12] Gamma.[1, 3]"
        assert _rewrite(text, ["perlan.is"]) == text

    def test_a_bare_tier_grade_with_no_outlet(self) -> None:
        # What the citation-index strip leaves of q44944's ``[1.1.8, 2.1.4: A]``.
        text = "HPI rose [A]."
        assert _rewrite(text, ["perlan.is"]) == text

    def test_bare_name_brackets_are_out_of_scope(self) -> None:
        # No tier grade anywhere in the group. This surface shares its syntax with
        # markdown link text and editorial insertions, where a rewrite would corrupt
        # content, and it is ~5% the size of the tier surface.
        for text in (
            "Coverage continued [Forbes].",
            "Two outlets ran it [Consumer Reports, NPR].",
            'He said "[the states] decide" on Tuesday.',
            "See [Reuters](https://example.com/wire) for the wire copy.",
        ):
            assert _rewrite(text, ["perlan.is"]) == text, text

    def test_prose_attributions_are_out_of_scope(self) -> None:
        # The measured surface is the tier tag; a prose extractor was not validated, so
        # per the plan it stays out.
        text = "According to Reuters the vote slipped, and per NASA the path holds."
        assert _rewrite(text, ["perlan.is"]) == text

    def test_the_sentence_a_tag_decorates_is_untouched(self) -> None:
        sentence = "Cloud cover blocks the view 76% of the time on August 12"
        out = _rewrite(f"{sentence} [C: Time and Date, D: Metaculus].", ["timeanddate.com"])
        assert out.startswith(f"{sentence} [C: Time and Date, ")

    def test_is_idempotent(self) -> None:
        text = "Reykjavik lies in the path of totality [A: NASA, B: Reuters]. Cloudy [C: Time and Date]."
        once = _rewrite(text, ["timeanddate.com"])
        assert _rewrite(once, ["timeanddate.com"]) == once

    def test_no_labels_leaves_every_tag_standing(self) -> None:
        # A response whose chunks carry no renderable label gives the check no evidence
        # base at all; rewriting off an empty record would make our own render failure
        # look like the model's embellishment. The provider skips the call, and the
        # function itself is safe if it ever gets one.
        text = "The notice was revised [A: official, B: Reuters]."
        assert _rewrite(text, []) == text


class TestCounts:
    def test_counts_report_names_flags_and_render_footprint(self) -> None:
        result = rewrite_unsupported_attributions(
            "Alpha [A: official, B: Reuters]. Beta [C: Time and Date]. Gamma [D: Metaculus, C: Newsweek].",
            ["timeanddate.com"],
        )
        # tagged excludes the generic ``official``; the three unsupported names are
        # Reuters, Metaculus and Newsweek, and they land in two bracket groups.
        assert (result.tagged, result.unsupported, result.groups_rewritten) == (4, 3, 2)

    def test_a_fully_supported_response_reports_zero(self) -> None:
        result = rewrite_unsupported_attributions("Cloudy [C: Time and Date].", ["timeanddate.com"])
        assert (result.tagged, result.unsupported, result.groups_rewritten) == (1, 0, 0)


class TestTheSharedBracketGrammar:
    """This check and ``gemini_search``'s citation-index strip run back to back over one
    response body and read it through one grammar (``research/bracket_groups.py``), so the
    grammar's own contract is pinned here rather than only through its two consumers —
    written twice, the two passes could come to disagree about the same string.
    """

    def test_items_come_through_raw_with_the_separator_that_introduced_them(self) -> None:
        # Raw, because the two passes disagree about what an item still SAYS: the strip
        # drops one left with no alphanumeric character, this check drops one that is empty
        # once stripped. Tidying inside the split would silently change both.
        assert list(iter_group_items("A: NASA, 1.1.2; C: Time and Date")) == [
            ("", "A: NASA"),
            (",", " 1.1.2"),
            (";", " C: Time and Date"),
        ]

    def test_the_first_surviving_item_drops_the_separator_that_introduced_it(self) -> None:
        # The item that now LEADS the group cannot keep a separator, or a group whose first
        # item was removed renders ``[, B: Reuters]``. Later items keep the model's own
        # ``,`` versus ``;``.
        assert join_group_items([(",", "B: Reuters"), (";", "C: MIT Sloan")]) == "B: Reuters; C: MIT Sloan"
        assert join_group_items([]) == ""

    def test_a_rewritten_group_keeps_the_space_the_pattern_consumed(self) -> None:
        # The shared pattern optionally eats the space in front of a group, so the one
        # consumer that can DELETE a group ("office [1.1.1]. He" -> "office. He") can take
        # the space with it. Every other rewrite has to put it back, which is what
        # ``rebuild_group`` is for.
        match = BRACKET_GROUP_RE.search("The agency named the supplier [A: FDA]")
        assert match is not None
        assert match.group(0) == " [A: FDA]"
        assert rebuild_group(match, "A: FDA, unverified attribution") == " [A: FDA, unverified attribution]"
