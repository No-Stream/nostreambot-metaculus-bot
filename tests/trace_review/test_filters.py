"""Filter masks, ordering and previous/next over the built fixture bundle."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from metaculus_bot.constants import PLATFORM_MANTIC, PLATFORM_METACULUS
from metaculus_bot.trace_review.filters import (
    FILTER_SPECS,
    PRESET_HELP,
    PRESETS,
    SHARE_EVERY,
    SHARE_NONE,
    SHARE_SOME,
    TRISTATE_NO,
    TRISTATE_NOT_RECORDED,
    TRISTATE_YES,
    FilterState,
    apply_filters,
    choice_mask,
    contains_mask,
    dates_mask,
    neighbours,
    range_mask,
    share_mask,
    text_mask,
    tristate_mask,
)
from tests.trace_review.workspace_fixtures import COLLIDING_BINARY, EARLY_BINARY, NEW_SINCE_ROUND, forecast_key

# About what a third of the main pane holds at 1280px wide with the sidebar open.
MAX_PRESET_LABEL_CHARS = 20


def _keys(forecasts: pd.DataFrame, mask: pd.Series) -> set[str]:
    return set(forecasts.loc[mask, "forecast_key"])


def _no_texts() -> pd.DataFrame:
    raise AssertionError("the texts table was read although no text search was set")


class TestMasks:
    def test_every_filter_names_a_forecast_column(self, bundle_forecasts):
        assert {spec.column for spec in FILTER_SPECS} <= set(bundle_forecasts.columns)
        assert len({spec.param for spec in FILTER_SPECS}) == len(FILTER_SPECS)

    def test_choice_and_contains(self, bundle_forecasts):
        assert choice_mask(bundle_forecasts["platform"], []).all()
        assert _keys(bundle_forecasts, choice_mask(bundle_forecasts["platform"], [PLATFORM_MANTIC])) == {
            key for key in bundle_forecasts["forecast_key"] if key.startswith(f"{PLATFORM_MANTIC}:")
        }
        with_model = _keys(bundle_forecasts, contains_mask(bundle_forecasts["member_models"], ["gpt-5.1"]))
        assert with_model == {forecast_key(PLATFORM_METACULUS, COLLIDING_BINARY)}

    def test_a_full_range_keeps_nulls_and_a_narrowed_one_drops_them(self, bundle_forecasts):
        crowd = bundle_forecasts["crowd_size"]
        extent = (float(crowd.min()), float(crowd.max()))
        assert crowd.isna().any()
        assert range_mask(crowd, extent, extent).all()
        narrowed = range_mask(crowd, (extent[0], extent[1] - 0.5) if extent[0] < extent[1] else (0.0, 39.0), extent)
        assert not narrowed[crowd.isna()].any()

    def test_tristate_separates_no_from_not_recorded(self, bundle_forecasts):
        extreme = bundle_forecasts["any_extreme_call"]
        yes, no, unknown = (
            tristate_mask(extreme, choice) for choice in (TRISTATE_YES, TRISTATE_NO, TRISTATE_NOT_RECORDED)
        )
        assert not (yes & no).any()
        assert not (no & unknown).any()
        assert (yes | no | unknown).all()
        assert unknown.sum() == extreme.isna().sum() > 0
        assert forecast_key(PLATFORM_METACULUS, EARLY_BINARY) in _keys(bundle_forecasts, unknown)

    def test_dates_text_and_trait_share(self, bundle_forecasts):
        submitted = bundle_forecasts["submitted_at"]
        extent = (submitted.min().date(), submitted.max().date())
        assert dates_mask(submitted, extent, extent).all()
        assert not dates_mask(submitted, (date(2030, 1, 1), date(2030, 2, 1)), extent).any()
        assert text_mask(bundle_forecasts["title"], "FIXTURE question 41577").sum() == 1
        counts, denominators = bundle_forecasts["n_market_price_cited"], bundle_forecasts["n_rationales"]
        some, every, none = (
            share_mask(counts, denominators, choice) for choice in (SHARE_SOME, SHARE_EVERY, SHARE_NONE)
        )
        assert not (some & none).any()
        assert (every <= some | (denominators == 0)).all()
        assert not (none & (denominators.fillna(0) == 0)).any()


class TestOrderAndPresets:
    def test_worst_misses_is_every_forecast_ascending_by_headline(self, bundle_forecasts):
        params = PRESETS["Worst misses"]
        state = FilterState(sort=params["sort"], descending=params["desc"] == "true")

        rows = apply_filters(bundle_forecasts, state, _no_texts).rows

        assert len(rows) == len(bundle_forecasts)
        scores = rows["headline_score"].dropna().tolist()
        assert scores == sorted(scores)
        missing = rows["headline_score"].isna().tolist()
        assert missing == sorted(missing)  # unscored rows last

    def test_since_the_last_round_is_the_metaculus_overlay(self, bundle_forecasts):
        params = PRESETS["Since last round"]
        state = FilterState(values={"origin": [params["origin"]], "plat": [params["plat"]]})

        rows = apply_filters(bundle_forecasts, state, _no_texts).rows

        assert rows["forecast_key"].tolist() == [forecast_key(PLATFORM_METACULUS, NEW_SINCE_ROUND)]

    def test_a_narrowed_range_reports_the_nulls_it_dropped(self, bundle_forecasts):
        crowd = bundle_forecasts["crowd_size"]
        extent = (float(crowd.min()), float(crowd.max()) + 1.0)
        state = FilterState(values={"crowd": (extent[0], extent[1] - 0.5)}, extents={"crowd": extent})

        result = apply_filters(bundle_forecasts, state, _no_texts)

        assert result.nulls_dropped == {"Crowd size": int(crowd.isna().sum())}

    def test_the_text_search_reads_the_texts_table(self, bundle_forecasts):
        texts = pd.DataFrame(
            {
                "forecast_key": bundle_forecasts["forecast_key"],
                "comment_text": None,
                "research_text": [
                    "CENTCOM releases" if key.startswith("mantic") else None for key in bundle_forecasts["forecast_key"]
                ],
            }
        )
        state = FilterState(values={"text": "centcom", "textin": "Research"})

        rows = apply_filters(bundle_forecasts, state, lambda: texts).rows

        assert rows["forecast_key"].str.startswith(PLATFORM_MANTIC).all()
        assert len(rows) == 1


def test_preset_labels_fit_three_to_a_row_and_each_has_hover_text() -> None:
    assert max(len(name) for name in PRESETS) <= MAX_PRESET_LABEL_CHARS
    assert set(PRESET_HELP) == set(PRESETS)


@pytest.mark.parametrize(
    ("key", "expected"),
    [("a", (None, "b", 1)), ("b", ("a", "c", 2)), ("c", ("b", None, 3)), ("z", (None, None, None))],
)
def test_neighbours(key: str, expected: tuple) -> None:
    assert neighbours(["a", "b", "c"], key) == expected
