"""Each figure draws every member beside the published forecast and marks the outcome."""

from __future__ import annotations

import pandas as pd
import pytest

from metaculus_bot.constants import PLATFORM_MANTIC, PLATFORM_METACULUS
from metaculus_bot.trace_review.plots import forecast_figure
from tests.trace_review.workspace_fixtures import (
    COLLIDING_BINARY,
    MANTIC_DISCRETE,
    MANTIC_PER_BIN_MASSES,
    NAMED_MC,
    NUMERIC_BACKFILLED,
    forecast_key,
)


def _figure(forecasts: pd.DataFrame, members: pd.DataFrame, key: str) -> tuple[dict, pd.DataFrame]:
    """The figure as plotly's plain dict (``data`` traces, ``layout``), and the member rows it was drawn from."""
    forecast = forecasts[forecasts["forecast_key"] == key].iloc[0]
    rows = members[members["forecast_key"] == key]
    return forecast_figure(forecast, rows).to_dict(), rows


@pytest.mark.parametrize("ids", [COLLIDING_BINARY, NAMED_MC, NUMERIC_BACKFILLED])
def test_one_trace_per_member_plus_the_published_forecast(bundle_forecasts, bundle_members, ids):
    figure, rows = _figure(bundle_forecasts, bundle_members, forecast_key(PLATFORM_METACULUS, ids))

    names = [trace["name"] for trace in figure["data"]]
    assert len(names) == len(rows) + 1
    assert set(rows["model"]) <= set(names)
    assert any(name.startswith("Published") for name in names)


def test_the_resolution_is_marked(bundle_forecasts, bundle_members):
    figure, _ = _figure(bundle_forecasts, bundle_members, forecast_key(PLATFORM_METACULUS, NUMERIC_BACKFILLED))
    assert any(annotation["text"].startswith("Resolved") for annotation in figure["layout"]["annotations"])


def test_an_out_of_range_resolution_is_an_annotation_and_a_per_bin_curve_ends_below_its_tail(
    bundle_forecasts, bundle_members
):
    figure, _ = _figure(bundle_forecasts, bundle_members, forecast_key(PLATFORM_MANTIC, MANTIC_DISCRETE))

    assert [annotation["text"] for annotation in figure["layout"]["annotations"]] == ["Resolved above the range"]
    member_trace = next(trace for trace in figure["data"] if trace["name"] == "gpt-6.1-sol")
    assert member_trace["y"][-1] == pytest.approx(1.0 - MANTIC_PER_BIN_MASSES[-1])
