"""One plotly figure per forecast: every member beside the published forecast, and the outcome.

Binary: members as dots on a 0-1 axis, a line at the published probability. Multiple choice:
published bars per option with the members as dots. Numeric and discrete: each member's CDF rebuilt
by the residual library (``member_replay.member_cdf``) on the question's own grid, the published
CDF heavier, the resolved value as a line, or an edge annotation when it fell outside the range.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from metaculus_bot.performance_analysis.member_replay import member_cdf, member_scoring_inputs
from metaculus_bot.performance_analysis.scaling import NUMERIC_TYPES, cdf_and_grid
from metaculus_bot.trace_review.member_rows import KIND_PMF, ROLE_STACKER, pmf_cdf

PUBLISHED_COLOR = "#1f2933"
RESOLUTION_COLOR = "#c2410c"
MEMBER_MARKER_SIZE = 11
PUBLISHED_LINE_WIDTH = 4
MEMBER_LINE_WIDTH = 1.6
FIGURE_HEIGHT = 420
FONT_SIZE = 15
MARGIN = {"l": 10, "r": 10, "t": 40, "b": 10}


def replay_record(forecast: pd.Series) -> dict:
    """The fields of a performance record the replay helpers read, rebuilt from a forecast row."""
    resolution = forecast["resolution_value"]
    if pd.isna(resolution):
        resolution = forecast["resolution_out_of_range"]
    return {
        "type": forecast["question_type"],
        "scaling": json.loads(forecast["scaling_json"]) if pd.notna(forecast["scaling_json"]) else {},
        "our_forecast_values": forecast["published_values"],
        "open_lower_bound": bool(forecast["open_lower_bound"]),
        "open_upper_bound": bool(forecast["open_upper_bound"]),
        "resolution_parsed": None if pd.isna(resolution) else resolution,
        "options": forecast["options"],
    }


def _layout(figure: go.Figure, title: str, **axes) -> go.Figure:
    figure.update_layout(
        title=title,
        height=FIGURE_HEIGHT,
        font={"size": FONT_SIZE},
        margin=MARGIN,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.18},
        **axes,
    )
    return figure


def _member_label(member: pd.Series) -> str:
    return f"{member['model']} (stacker)" if member["role"] == ROLE_STACKER else str(member["model"])


def binary_figure(forecast: pd.Series, members: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for _, member in members.iterrows():
        if pd.notna(member["prob_yes"]):
            figure.add_trace(
                go.Scatter(
                    x=[member["prob_yes"]],
                    y=[_member_label(member)],
                    mode="markers",
                    marker={"size": MEMBER_MARKER_SIZE},
                    name=_member_label(member),
                )
            )
    published = forecast["published_prob_yes"]
    figure.add_trace(
        go.Scatter(
            x=[published, published],
            y=[members["model"].iloc[0], members["model"].iloc[-1]] if len(members) else [0, 1],
            mode="lines",
            line={"color": PUBLISHED_COLOR, "width": PUBLISHED_LINE_WIDTH},
            name=f"Published {published:.1%}",
        )
    )
    outcome = "YES" if forecast["resolved_yes"] else "NO"
    figure.add_vline(x=1.0 if forecast["resolved_yes"] else 0.0, line={"color": RESOLUTION_COLOR, "dash": "dot"})
    figure.add_annotation(
        x=1.0 if forecast["resolved_yes"] else 0.0, y=1.05, yref="paper", text=f"Resolved {outcome}", showarrow=False
    )
    return _layout(figure, "Probability of YES by member", xaxis={"range": [-0.02, 1.02], "tickformat": ".0%"})


def multiple_choice_figure(forecast: pd.Series, members: pd.DataFrame) -> go.Figure:
    options = list(forecast["options"])
    resolved = forecast["resolution_raw"]
    figure = go.Figure(
        go.Bar(
            x=options,
            y=list(forecast["published_values"]),
            name="Published",
            marker={"color": ["#475569" if option != resolved else RESOLUTION_COLOR for option in options]},
            opacity=0.45,
        )
    )
    for _, member in members.iterrows():
        if pd.notna(member["option_probs_json"]):
            probs = json.loads(member["option_probs_json"])
            figure.add_trace(
                go.Scatter(
                    x=options,
                    y=[probs.get(option) for option in options],
                    mode="markers",
                    marker={"size": MEMBER_MARKER_SIZE},
                    name=_member_label(member),
                )
            )
    return _layout(figure, f"Option probabilities (resolved: {resolved})", yaxis={"tickformat": ".0%"})


def _member_curve(member: pd.Series, record: dict) -> list[float] | None:
    if pd.isna(member["percentiles_json"]):
        return None
    value = json.loads(member["percentiles_json"])
    if member["value_kind"] == KIND_PMF:
        return pmf_cdf(value)
    inputs = member_scoring_inputs(record)
    if inputs is None:
        return None
    try:
        return member_cdf(value, inputs)
    except (ValueError, RuntimeError):
        return None


def _mark_resolution(figure: go.Figure, forecast: pd.Series, grid: np.ndarray) -> None:
    out_of_range = forecast["resolution_out_of_range"]
    if pd.notna(out_of_range):
        above = out_of_range == "above_upper_bound"
        figure.add_annotation(
            x=float(grid[-1] if above else grid[0]),
            y=0.5,
            text="Resolved above the range" if above else "Resolved below the range",
            showarrow=True,
            ax=-60 if above else 60,
            ay=0,
            font={"color": RESOLUTION_COLOR},
        )
    elif pd.notna(forecast["resolution_value"]):
        value = float(forecast["resolution_value"])
        figure.add_vline(x=value, line={"color": RESOLUTION_COLOR, "dash": "dot"})
        figure.add_annotation(x=value, y=1.05, yref="paper", text=f"Resolved {value:,.4g}", showarrow=False)


def numeric_figure(forecast: pd.Series, members: pd.DataFrame) -> go.Figure:
    record = replay_record(forecast)
    grid_and_cdf = cdf_and_grid(record)
    figure = go.Figure()
    if grid_and_cdf is None:
        return _layout(figure, "No grid recorded for this question")
    published, grid = grid_and_cdf
    for _, member in members.iterrows():
        curve = _member_curve(member, record)
        if curve is not None and len(curve) == len(grid):
            figure.add_trace(
                go.Scatter(x=grid, y=curve, mode="lines", line={"width": MEMBER_LINE_WIDTH}, name=_member_label(member))
            )
    figure.add_trace(
        go.Scatter(
            x=grid,
            y=published,
            mode="lines",
            line={"color": PUBLISHED_COLOR, "width": PUBLISHED_LINE_WIDTH},
            name="Published",
        )
    )
    _mark_resolution(figure, forecast, grid)
    return _layout(figure, "Cumulative probability by member", yaxis={"range": [0, 1], "tickformat": ".0%"})


def forecast_figure(forecast: pd.Series, members: pd.DataFrame) -> go.Figure:
    question_type = forecast["question_type"]
    if question_type == "binary":
        return binary_figure(forecast, members)
    if question_type == "multiple_choice":
        return multiple_choice_figure(forecast, members)
    if question_type in NUMERIC_TYPES:
        return numeric_figure(forecast, members)
    raise ValueError(f"no figure for question type {question_type!r}")
