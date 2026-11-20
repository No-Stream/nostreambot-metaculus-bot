"""Each figure draws every member beside the published forecast and marks the outcome."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from metaculus_bot.constants import PLATFORM_MANTIC, PLATFORM_METACULUS
from metaculus_bot.performance_analysis.scaling import cdf_and_grid
from metaculus_bot.trace_review.plots import density_curve, forecast_figure, member_colors, replay_record
from tests.trace_review.workspace_fixtures import (
    COLLIDING_BINARY,
    MANTIC_DISCRETE,
    MANTIC_PER_BIN_MASSES,
    NAMED_MC,
    NUMERIC_BACKFILLED,
    forecast_key,
)


def _figure(
    forecasts: pd.DataFrame, members: pd.DataFrame, key: str, cumulative: bool = False
) -> tuple[Any, pd.DataFrame]:
    """The figure (typed Any: plotly's generated classes defeat the type checker) and its member rows."""
    forecast = forecasts[forecasts["forecast_key"] == key].iloc[0]
    rows = members[members["forecast_key"] == key]
    return forecast_figure(forecast, rows, cumulative=cumulative), rows


def _annotations(figure: Any) -> list[str]:
    return [annotation.text for annotation in figure.layout.annotations]


def _vertical_lines(figure: Any) -> list[float]:
    return [shape.x0 for shape in figure.layout.shapes]


def _trace(figure: Any, name: str) -> Any:
    return next(trace for trace in figure.data if trace.name == name)


@pytest.mark.parametrize("ids", [NAMED_MC, NUMERIC_BACKFILLED])
@pytest.mark.parametrize("cumulative", [False, True])
def test_one_trace_per_member_plus_the_published_forecast(bundle_forecasts, bundle_members, ids, cumulative):
    figure, rows = _figure(bundle_forecasts, bundle_members, forecast_key(PLATFORM_METACULUS, ids), cumulative)

    names = [trace.name for trace in figure.data]
    assert len(names) == len(rows) + 1
    assert set(rows["model"]) <= set(names)
    assert names.count("Published") == 1


def test_binary_draws_a_dot_per_member_and_labelled_lines_for_published_and_resolution(
    bundle_forecasts, bundle_members
):
    key = forecast_key(PLATFORM_METACULUS, COLLIDING_BINARY)
    figure, rows = _figure(bundle_forecasts, bundle_members, key)
    forecast = bundle_forecasts.set_index("forecast_key").loc[key]

    assert [trace.name for trace in figure.data] == rows["model"].tolist()
    assert all(trace.text[0].endswith("%") for trace in figure.data)
    assert _annotations(figure) == [f"Published {forecast['published_prob_yes']:.1%}", "Resolved NO"]
    assert _vertical_lines(figure) == [pytest.approx(forecast["published_prob_yes"]), 0.0]


def test_a_member_keeps_its_family_colour_and_a_second_family_member_gets_another_shade():
    colors = member_colors(["gpt-5", "claude-opus-4.8", "o3", "Forecaster 1"])

    assert colors["gpt-5"] == member_colors(["gpt-5"])["gpt-5"]
    assert len(set(colors.values())) == len(colors)


def test_the_resolution_is_marked_on_both_views(bundle_forecasts, bundle_members):
    for cumulative in (False, True):
        figure, _ = _figure(
            bundle_forecasts, bundle_members, forecast_key(PLATFORM_METACULUS, NUMERIC_BACKFILLED), cumulative
        )
        assert any(text.startswith("Resolved") for text in _annotations(figure))


def test_an_out_of_range_resolution_is_marked_at_the_edge_and_a_per_bin_curve_ends_below_its_tail(
    bundle_forecasts, bundle_members
):
    figure, _ = _figure(bundle_forecasts, bundle_members, forecast_key(PLATFORM_MANTIC, MANTIC_DISCRETE), True)

    assert _annotations(figure) == ["Resolved above the range"]
    member_trace = _trace(figure, "gpt-6.1-sol")
    assert member_trace.y[-1] == pytest.approx(1.0 - MANTIC_PER_BIN_MASSES[-1])
    assert _vertical_lines(figure)[0] == pytest.approx(member_trace.x[-1])


def test_the_discrete_density_is_the_mass_in_each_value_and_sums_with_the_tails_to_one(
    bundle_forecasts, bundle_members
):
    key = forecast_key(PLATFORM_MANTIC, MANTIC_DISCRETE)
    forecast = bundle_forecasts.set_index("forecast_key").loc[key]
    grid_and_cdf = cdf_and_grid(replay_record(forecast))
    assert grid_and_cdf is not None
    cdf, _ = grid_and_cdf
    figure, _ = _figure(bundle_forecasts, bundle_members, key)

    member_trace = _trace(figure, "gpt-6.1-sol")
    assert list(member_trace.y) == pytest.approx(MANTIC_PER_BIN_MASSES[1:-1])
    published = _trace(figure, "Published")
    assert float(np.sum(published.y)) + cdf[0] + (1 - cdf[-1]) == pytest.approx(1.0)


def test_a_continuous_density_is_mass_over_bin_width():
    grid = np.array([0.0, 1.0, 3.0])
    centres, density = density_curve(np.array([0.0, 0.5, 1.0]), grid, per_value=False)

    assert centres.tolist() == [0.5, 2.0]
    assert density.tolist() == [0.5, 0.25]
    assert float(np.sum(density * np.diff(grid))) == pytest.approx(1.0)
