"""One plotly figure per forecast: every member beside the published forecast, and the outcome.

Binary: one row per member with its probability as a labelled dot, and full-height lines at the
published probability and at the resolved side. Multiple choice: published bars per option with the
members as dots, the resolved option marked. Numeric and discrete: each member's CDF rebuilt by the
residual library (``member_replay.member_cdf``) on the question's own grid, shown as a density
(per-value probability on a discrete question) or as the CDF itself, the published curve dashed on
top, and the resolved value as a line (at the edge of the range when it fell outside).

A member's colour comes from its model family, so a vendor keeps its colour from forecast to forecast.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from metaculus_bot.performance_analysis.member_replay import member_cdf, member_scoring_inputs
from metaculus_bot.performance_analysis.scaling import NUMERIC_TYPES, cdf_and_grid
from metaculus_bot.trace_review.member_rows import KIND_PMF, ROLE_STACKER, pmf_cdf

# Published is drawn light on the dark theme and dark on the light one; every other colour reads on both.
PUBLISHED_COLOR_DARK_THEME = "#f1f5f9"
PUBLISHED_COLOR_LIGHT_THEME = "#111827"
RESOLUTION_COLOR = "#ec4899"
PUBLISHED_BAR_COLOR = "#64748b"
# (model-name pattern matched at the start, (first model's colour, a second model of the same family's colour)).
# Patterns rather than literal ids, since model ids live only in llm_configs.py and constants.py.
MODEL_FAMILY_COLORS: tuple[tuple[re.Pattern[str], tuple[str, str]], ...] = (
    (re.compile(r"gpt|o\d"), ("#10b981", "#6ee7b7")),
    (re.compile(r"claude"), ("#f97316", "#fdba74")),
    (re.compile(r"gemini"), ("#3b82f6", "#93c5fd")),
    (re.compile(r"grok"), ("#a855f7", "#d8b4fe")),
    (re.compile(r"qwen"), ("#eab308", "#fde047")),
    (re.compile(r"kimi"), ("#06b6d4", "#67e8f9")),
)
OTHER_MODEL_COLORS = ("#94a3b8", "#cbd5e1")
MEMBER_OPACITY = 0.7
MEMBER_MARKER_SIZE = 13
MEMBER_LINE_WIDTH = 2
PUBLISHED_LINE_WIDTH = 3
RESOLUTION_LINE_WIDTH = 2
FIGURE_HEIGHT = 440
BINARY_ROW_HEIGHT = 56
BINARY_BASE_HEIGHT = 150
OPTION_ROW_HEIGHT = 34
OPTION_BASE_HEIGHT = 170
FONT_SIZE = 14
MARGIN = {"l": 10, "r": 30, "t": 80, "b": 10}
BINARY_MARGIN_TOP = 100
# Second label row above the plot, so the resolution label never sits on the published label.
LABEL_ROW_SHIFT_PX = 22


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


def format_quantity(value: float) -> str:
    """A numeric value for display: thousands grouped, never scientific notation above 1,000."""
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"


def published_color(dark_theme: bool) -> str:
    return PUBLISHED_COLOR_DARK_THEME if dark_theme else PUBLISHED_COLOR_LIGHT_THEME


def member_colors(models: list[str]) -> dict[str, str]:
    """A colour per model: its family's colour, a lighter shade for a second model of that family."""
    colors: dict[str, str] = {}
    seen_per_family: dict[int, int] = {}
    for model in models:
        family = next(
            (index for index, (pattern, _) in enumerate(MODEL_FAMILY_COLORS) if pattern.match(model.lower())),
            len(MODEL_FAMILY_COLORS),
        )
        shades = MODEL_FAMILY_COLORS[family][1] if family < len(MODEL_FAMILY_COLORS) else OTHER_MODEL_COLORS
        nth = seen_per_family.get(family, 0)
        seen_per_family[family] = nth + 1
        colors[model] = shades[nth % len(shades)]
    return colors


def _layout(figure: go.Figure, title: str, **layout) -> go.Figure:
    figure.update_layout(
        title={"text": title, "y": 0.98, "yanchor": "top"},
        height=FIGURE_HEIGHT,
        font={"size": FONT_SIZE},
        margin=MARGIN,
        legend={"orientation": "h", "y": -0.15},
        hoverlabel={"namelength": -1},
    )
    figure.update_layout(**layout)
    return figure


def _member_label(member: pd.Series) -> str:
    return f"{member['model']} (stacker)" if member["role"] == ROLE_STACKER else str(member["model"])


def _label_line(figure: go.Figure, x: float, text: str, color: str, *, row: int, xanchor: str = "center") -> None:
    """A text label just above the plot area at ``x``; row 0 sits nearest the plot, row 1 above it."""
    figure.add_annotation(
        x=x,
        y=1,
        yref="paper",
        yanchor="bottom",
        yshift=4 + row * LABEL_ROW_SHIFT_PX,
        xanchor=xanchor,
        text=text,
        showarrow=False,
        font={"color": color},
    )


def binary_figure(forecast: pd.Series, members: pd.DataFrame, dark_theme: bool) -> go.Figure:
    valued = members[members["prob_yes"].notna()]
    colors = member_colors([_member_label(member) for _, member in valued.iterrows()])
    figure = go.Figure()
    for _, member in valued.iterrows():
        label = _member_label(member)
        figure.add_trace(
            go.Scatter(
                x=[member["prob_yes"]],
                y=[label],
                mode="markers+text",
                text=[f"{member['prob_yes']:.0%}"],
                textposition="top center",
                textfont={"color": colors[label]},
                cliponaxis=False,
                marker={"size": MEMBER_MARKER_SIZE, "color": colors[label]},
                name=label,
                hovertemplate=f"{label}: %{{x:.1%}}<extra></extra>",
            )
        )
    published = float(forecast["published_prob_yes"])
    color = published_color(dark_theme)
    figure.add_vline(x=published, line={"color": color, "width": PUBLISHED_LINE_WIDTH, "dash": "dash"})
    _label_line(figure, published, f"Published {published:.1%}", color, row=0)
    resolved_side = 1.0 if forecast["resolved_yes"] else 0.0
    outcome = "YES" if forecast["resolved_yes"] else "NO"
    figure.add_vline(x=resolved_side, line={"color": RESOLUTION_COLOR, "width": RESOLUTION_LINE_WIDTH})
    _label_line(
        figure,
        resolved_side,
        f"Resolved {outcome}",
        RESOLUTION_COLOR,
        row=1,
        xanchor="right" if forecast["resolved_yes"] else "left",
    )
    return _layout(
        figure,
        "Probability of YES by member",
        height=BINARY_BASE_HEIGHT + BINARY_ROW_HEIGHT * max(len(valued), 1),
        margin={**MARGIN, "t": BINARY_MARGIN_TOP},
        showlegend=False,
        xaxis={"range": [-0.03, 1.03], "tickformat": ".0%", "title": "Probability of YES"},
        yaxis={"autorange": "reversed"},
    )


def multiple_choice_figure(forecast: pd.Series, members: pd.DataFrame) -> go.Figure:
    options = list(forecast["options"])
    resolved = forecast["resolution_raw"]
    labels = [f"<b>{option} (resolved)</b>" if option == resolved else option for option in options]
    figure = go.Figure(
        go.Bar(
            x=list(forecast["published_values"]),
            y=labels,
            orientation="h",
            name="Published",
            marker={"color": [RESOLUTION_COLOR if option == resolved else PUBLISHED_BAR_COLOR for option in options]},
            opacity=0.55,
            hovertemplate="Published, %{y}: %{x:.1%}<extra></extra>",
        )
    )
    valued = members[members["option_probs_json"].notna()]
    colors = member_colors([_member_label(member) for _, member in valued.iterrows()])
    for _, member in valued.iterrows():
        label = _member_label(member)
        probs = json.loads(member["option_probs_json"])
        figure.add_trace(
            go.Scatter(
                x=[probs.get(option) for option in options],
                y=labels,
                mode="markers",
                marker={"size": MEMBER_MARKER_SIZE, "color": colors[label], "opacity": MEMBER_OPACITY + 0.15},
                name=label,
                hovertemplate=f"{label}, %{{y}}: %{{x:.1%}}<extra></extra>",
            )
        )
    return _layout(
        figure,
        f"Probability per option (resolved: {resolved})",
        height=OPTION_BASE_HEIGHT + OPTION_ROW_HEIGHT * len(options),
        margin={**MARGIN, "t": 60},
        xaxis={"tickformat": ".0%", "rangemode": "tozero", "title": "Probability"},
        yaxis={"autorange": "reversed"},
    )


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


def density_curve(cdf: np.ndarray, grid: np.ndarray, per_value: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(bin centres, heights)`` from a CDF on its grid: the mass in each bin on a discrete question
    (``per_value``), the mass divided by the bin width on a continuous one. Mass below or above the range
    is not in any bin."""
    mass = np.diff(np.asarray(cdf, dtype=float))
    widths = np.diff(grid)
    centres = (grid[:-1] + grid[1:]) / 2
    return centres, mass if per_value else mass / widths


def _mark_resolution(figure: go.Figure, forecast: pd.Series, grid: np.ndarray) -> None:
    out_of_range = forecast["resolution_out_of_range"]
    if pd.notna(out_of_range):
        above = out_of_range == "above_upper_bound"
        edge = float(grid[-1] if above else grid[0])
        text = "Resolved above the range" if above else "Resolved below the range"
        xanchor = "right" if above else "left"
    elif pd.notna(forecast["resolution_value"]):
        edge = float(forecast["resolution_value"])
        text, xanchor = f"Resolved {format_quantity(edge)}", "center"
    else:
        return
    figure.add_vline(x=edge, line={"color": RESOLUTION_COLOR, "width": RESOLUTION_LINE_WIDTH})
    _label_line(figure, edge, text, RESOLUTION_COLOR, row=0, xanchor=xanchor)


def numeric_figure(forecast: pd.Series, members: pd.DataFrame, cumulative: bool, dark_theme: bool) -> go.Figure:
    record = replay_record(forecast)
    grid_and_cdf = cdf_and_grid(record)
    figure = go.Figure()
    if grid_and_cdf is None:
        return _layout(figure, "No grid recorded for this question")
    published, grid = grid_and_cdf
    per_value = forecast["question_type"] == "discrete"

    def shown(cdf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (grid, cdf) if cumulative else density_curve(cdf, grid, per_value)

    y_format = ".1%" if cumulative or per_value else ".3g"
    curves = [(member, _member_curve(member, record)) for _, member in members.iterrows()]
    curves = [(member, curve) for member, curve in curves if curve is not None and len(curve) == len(grid)]
    colors = member_colors([_member_label(member) for member, _ in curves])
    for member, curve in curves:
        label = _member_label(member)
        x, y = shown(np.asarray(curve, dtype=float))
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={
                    "width": MEMBER_LINE_WIDTH,
                    "color": colors[label],
                    "shape": "hvh" if per_value and not cumulative else "linear",
                },
                opacity=MEMBER_OPACITY,
                name=label,
                hovertemplate=f"{label}<br>%{{x:,.4g}}: %{{y:{y_format}}}<extra></extra>",
            )
        )
    x, y = shown(published)
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line={
                "color": published_color(dark_theme),
                "width": PUBLISHED_LINE_WIDTH,
                "dash": "dash",
                "shape": "hvh" if per_value and not cumulative else "linear",
            },
            name="Published",
            hovertemplate=f"Published<br>%{{x:,.4g}}: %{{y:{y_format}}}<extra></extra>",
        )
    )
    _mark_resolution(figure, forecast, grid)
    if cumulative:
        title, yaxis = "Cumulative probability by member", {"range": [0, 1], "tickformat": ".0%"}
    elif per_value:
        title, yaxis = "Probability of each value by member", {"rangemode": "tozero", "tickformat": ".0%"}
    else:
        title, yaxis = (
            "Probability density by member",
            {"rangemode": "tozero", "tickformat": ".3~g", "title": "Density"},
        )
    return _layout(figure, title, yaxis=yaxis, hovermode="closest")


def forecast_figure(
    forecast: pd.Series, members: pd.DataFrame, *, cumulative: bool = False, dark_theme: bool = True
) -> go.Figure:
    question_type = forecast["question_type"]
    if question_type == "binary":
        return binary_figure(forecast, members, dark_theme)
    if question_type == "multiple_choice":
        return multiple_choice_figure(forecast, members)
    if question_type in NUMERIC_TYPES:
        return numeric_figure(forecast, members, cumulative, dark_theme)
    raise ValueError(f"no figure for question type {question_type!r}")
