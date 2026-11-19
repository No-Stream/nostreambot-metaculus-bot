"""The sidebar's filters: one FilterSpec row per filter, one mask and one render function per kind.

Masks are plain pandas so the tests run without Streamlit. Every widget is bound to its query
parameter (``bind=BIND_TO_URL``), so the URL holds the whole view and every page that calls
``render_sidebar`` sees the same filters. A null is never a "no": a narrowed range or a Yes/No
choice drops null rows, and the tristate's "Not recorded" selects exactly them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd
import streamlit as st

from metaculus_bot.trace_review.config import RECORD_ORIGIN_FRESH

KIND_CHOICE = "choice"
KIND_CONTAINS = "contains"
KIND_RANGE = "range"
KIND_DATES = "dates"
KIND_TRISTATE = "tristate"
KIND_TEXT = "text"
KIND_SHARE = "share"

TRISTATE_ANY = "Any"
TRISTATE_YES = "Yes"
TRISTATE_NO = "No"
TRISTATE_NOT_RECORDED = "Not recorded"
TRISTATE_OPTIONS: tuple[str, ...] = (TRISTATE_ANY, TRISTATE_YES, TRISTATE_NO, TRISTATE_NOT_RECORDED)
SHARE_ANY = "Any"
SHARE_SOME = "Some member"
SHARE_EVERY = "Every member"
SHARE_NONE = "No member"
SHARE_OPTIONS: tuple[str, ...] = (SHARE_ANY, SHARE_SOME, SHARE_EVERY, SHARE_NONE)
SHARE_DENOMINATOR_COLUMN = "n_rationales"

TEXT_SEARCH_PARAM = "text"
TEXT_SEARCH_IN_PARAM = "textin"
TEXT_IN_COMMENT = "Comment"
TEXT_IN_RESEARCH = "Research"
TEXT_IN_BOTH = "Both"
TEXT_IN_OPTIONS: tuple[str, ...] = (TEXT_IN_BOTH, TEXT_IN_COMMENT, TEXT_IN_RESEARCH)
TEXT_COLUMNS_BY_CHOICE: dict[str, tuple[str, ...]] = {
    TEXT_IN_BOTH: ("comment_text", "research_text"),
    TEXT_IN_COMMENT: ("comment_text",),
    TEXT_IN_RESEARCH: ("research_text",),
}

BIND_TO_URL: Literal["query-params"] = "query-params"
SORT_PARAM = "sort"
DESC_PARAM = "desc"
KEY_PARAM = "key"
DEFAULT_SORT = "submitted_at"
SORTABLE_COLUMNS: tuple[str, ...] = (
    "submitted_at",
    "actual_resolve_time",
    "headline_score",
    "type_log_score",
    "spot_peer_score",
    "spot_baseline_score",
    "prob_on_outcome",
    "member_spread",
    "ghost_score_delta",
    "ghost_score_delta_abs",
    "lead_days",
    "close_margin_frac",
    "research_chars",
    "usd_per_question",
    "crowd_size",
    "n_members",
    "rationale_chars_mean",
    "hedge_per_1k_mean",
)


@dataclass(frozen=True)
class FilterSpec:
    param: str
    column: str
    kind: str
    label: str
    group: str


def _spec(group: str, rows: list[tuple[str, str, str, str]]) -> list[FilterSpec]:
    return [FilterSpec(param, column, kind, label, group) for param, column, kind, label in rows]


FILTER_SPECS: tuple[FilterSpec, ...] = (
    *_spec(
        "Question",
        [
            ("plat", "platform", KIND_CHOICE, "Platform"),
            ("tour", "tournament", KIND_CHOICE, "Tournament"),
            ("type", "question_type", KIND_CHOICE, "Question type"),
            ("cat", "category", KIND_CHOICE, "Category"),
            ("q", "title", KIND_TEXT, "Title contains"),
            ("openlo", "open_lower_bound", KIND_TRISTATE, "Open lower bound"),
            ("openhi", "open_upper_bound", KIND_TRISTATE, "Open upper bound"),
            ("origin", "record_origin", KIND_CHOICE, "Record origin"),
            ("rescored", "rescored_since_round", KIND_TRISTATE, "Rescored since the round"),
        ],
    ),
    *_spec(
        "Time",
        [
            ("submitted", "submitted_at", KIND_DATES, "Submitted"),
            ("resolved", "actual_resolve_time", KIND_DATES, "Resolved"),
            ("lead", "lead_days", KIND_RANGE, "Days from submit to resolution"),
            ("margin", "close_margin_frac", KIND_RANGE, "Share of the window left at submit"),
        ],
    ),
    *_spec(
        "Outcome",
        [
            ("score", "headline_score", KIND_RANGE, "Headline score"),
            ("scorefield", "headline_score_field", KIND_CHOICE, "Headline score field"),
            ("tls", "type_log_score", KIND_RANGE, "Log score (type's own)"),
            ("yes", "resolved_yes", KIND_TRISTATE, "Resolved yes (binary)"),
            ("oor", "resolution_out_of_range", KIND_CHOICE, "Out-of-range resolution"),
            ("p", "published_prob_yes", KIND_RANGE, "Published probability (binary)"),
            ("pout", "prob_on_outcome", KIND_RANGE, "Probability on what happened"),
            ("cw", "confident_wrong", KIND_TRISTATE, "Confident and wrong"),
            ("crowd", "crowd_size", KIND_RANGE, "Crowd size"),
        ],
    ),
    *_spec(
        "Ensemble",
        [
            ("model", "member_models", KIND_CONTAINS, "Member model"),
            ("nmem", "n_members", KIND_RANGE, "Members"),
            ("spread", "member_spread", KIND_RANGE, "Member spread"),
            ("agg", "aggregate_method", KIND_CHOICE, "Aggregate method"),
            ("stacker", "stacker_verdict", KIND_CHOICE, "Stacker verdict"),
            ("drop", "any_drop", KIND_TRISTATE, "A forecaster dropped"),
            ("extreme", "any_extreme_call", KIND_TRISTATE, "Extreme call"),
            ("rung", "extraction_not_clean", KIND_TRISTATE, "Extraction off the block rung"),
        ],
    ),
    *_spec(
        "Research",
        [
            ("rclass", "research_source_class", KIND_CHOICE, "Research record class"),
            ("provok", "providers_succeeded", KIND_CONTAINS, "Provider succeeded"),
            ("provfail", "providers_failed", KIND_CONTAINS, "Provider failed"),
            ("market", "market_snapshot_present", KIND_TRISTATE, "Market snapshot"),
            ("gfv2", "gfv2_loop_ran", KIND_TRISTATE, "Gap-fill v2 loop ran"),
            ("anchor", "anchor_present", KIND_TRISTATE, "Time-series anchor"),
            ("fast", "fast_path", KIND_TRISTATE, "Fast path"),
            ("rchars", "research_chars", KIND_RANGE, "Research characters"),
            ("ghost", "ghost_present", KIND_TRISTATE, "Ghost forecast logged"),
            ("ghostd", "ghost_score_delta", KIND_RANGE, "Ghost minus published score"),
        ],
    ),
    *_spec(
        "Config and cohorts",
        [
            ("era", "config_era", KIND_CHOICE, "Config era"),
            ("sub", "triple_subera_fine", KIND_CHOICE, "Triple sub-era"),
            ("deg", "degraded_run", KIND_TRISTATE, "Degraded run"),
            ("pdeg", "partial_degraded", KIND_TRISTATE, "Partially degraded"),
            ("bug", "known_bug", KIND_TRISTATE, "Known bug"),
            ("cost", "usd_per_question", KIND_RANGE, "USD per question (run mean)"),
        ],
    ),
    *_spec(
        "Text traits",
        [
            ("mkt", "n_market_price_cited", KIND_SHARE, "Cites a market price"),
            ("nomkt", "n_market_absent_stated", KIND_SHARE, "Says no market exists"),
            ("crowdcite", "n_crowd_forecast_cited", KIND_SHARE, "Cites the crowd forecast"),
            ("ambig", "n_resolution_ambiguity_flagged", KIND_SHARE, "Flags resolution ambiguity"),
            ("rlen", "rationale_chars_mean", KIND_RANGE, "Rationale length (mean)"),
            ("pct", "percent_count_mean", KIND_RANGE, "Percentages per rationale"),
            ("hedge", "hedge_per_1k_mean", KIND_RANGE, "Hedges per 1,000 characters"),
            ("baserate", "base_rate_number_count_mean", KIND_RANGE, "Base rates with a number"),
            ("sq", "status_quo_count_mean", KIND_RANGE, "Status-quo mentions"),
        ],
    ),
)
FILTER_GROUPS: tuple[str, ...] = tuple(dict.fromkeys(spec.group for spec in FILTER_SPECS))

PRESETS: dict[str, dict[str, str]] = {
    "Worst misses": {SORT_PARAM: "headline_score", DESC_PARAM: "false"},
    "Confident and wrong": {"cw": TRISTATE_YES, SORT_PARAM: "headline_score", DESC_PARAM: "false"},
    "Biggest member disagreement": {SORT_PARAM: "member_spread", DESC_PARAM: "true"},
    "Ghost differs most from published": {
        "ghost": TRISTATE_YES,
        SORT_PARAM: "ghost_score_delta_abs",
        DESC_PARAM: "true",
    },
    "Since the last round": {"origin": RECORD_ORIGIN_FRESH, "plat": "metaculus"},
    "Strict cohort": {"deg": TRISTATE_NO, "pdeg": TRISTATE_NO, "bug": TRISTATE_NO},
}


@dataclass
class FilterState:
    """The value of every rendered filter, keyed by query parameter, plus the sort."""

    values: dict[str, object] = field(default_factory=dict)
    extents: dict[str, tuple] = field(default_factory=dict)
    sort: str = DEFAULT_SORT
    descending: bool = True


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def _all(frame_or_series: pd.DataFrame | pd.Series) -> pd.Series:
    return pd.Series(data=True, index=frame_or_series.index)


def choice_mask(values: pd.Series, selected: list) -> pd.Series:
    return _all(values) if not selected else values.isin(selected).fillna(value=False).astype(bool)


def contains_mask(values: pd.Series, selected: list) -> pd.Series:
    if not selected:
        return _all(values)
    wanted = set(selected)
    return values.map(lambda items: isinstance(items, list) and bool(wanted.intersection(items))).astype(bool)


def range_mask(values: pd.Series, chosen: tuple, extent: tuple) -> pd.Series:
    """Full extent filters nothing and keeps nulls; a narrowed range drops them."""
    if tuple(chosen) == tuple(extent):
        return _all(values)
    low, high = chosen
    return values.between(low, high).fillna(value=False).astype(bool)


def dates_mask(values: pd.Series, chosen: tuple, extent: tuple) -> pd.Series:
    if len(chosen) != 2 or tuple(chosen) == tuple(extent):
        return _all(values)
    start, end = chosen
    days = values.dt.date
    return (values.notna() & (days >= start) & (days <= end)).fillna(value=False).astype(bool)


def tristate_mask(values: pd.Series, choice: str) -> pd.Series:
    if choice == TRISTATE_ANY:
        return _all(values)
    if choice == TRISTATE_NOT_RECORDED:
        return values.isna()
    return (values == (choice == TRISTATE_YES)).fillna(value=False).astype(bool)


def text_mask(values: pd.Series, needle: str) -> pd.Series:
    if not needle.strip():
        return _all(values)
    return values.str.contains(needle.strip(), case=False, regex=False).fillna(value=False).astype(bool)


def share_mask(counts: pd.Series, denominators: pd.Series, choice: str) -> pd.Series:
    """How many member rationales show a trait: some, every one, or none. No rationale is not recorded."""
    if choice == SHARE_ANY:
        return _all(counts)
    recorded = denominators.fillna(0) > 0
    by_choice = {SHARE_SOME: counts > 0, SHARE_EVERY: counts == denominators, SHARE_NONE: counts == 0}
    return (recorded & by_choice[choice]).fillna(value=False).astype(bool)


def spec_mask(spec: FilterSpec, forecasts: pd.DataFrame, value: object, extent: tuple | None) -> pd.Series:
    column = forecasts[spec.column]
    if spec.kind == KIND_CHOICE:
        return choice_mask(column, list(value or []))  # type: ignore[arg-type]
    if spec.kind == KIND_CONTAINS:
        return contains_mask(column, list(value or []))  # type: ignore[arg-type]
    if spec.kind == KIND_RANGE:
        return range_mask(column, tuple(value), extent or ())  # type: ignore[arg-type]
    if spec.kind == KIND_DATES:
        return dates_mask(column, tuple(value), extent or ())  # type: ignore[arg-type]
    if spec.kind == KIND_TRISTATE:
        return tristate_mask(column, str(value))
    if spec.kind == KIND_SHARE:
        return share_mask(column, forecasts[SHARE_DENOMINATOR_COLUMN], str(value))
    return text_mask(column, str(value or ""))


def text_search_mask(
    forecasts: pd.DataFrame, needle: str, where: str, load_texts: Callable[[], pd.DataFrame]
) -> pd.Series:
    """Substring search in the comment and/or research text, which only the texts table holds."""
    if not needle.strip():
        return _all(forecasts)
    texts = load_texts().set_index("forecast_key")
    hit = pd.Series(data=False, index=texts.index)
    for column in TEXT_COLUMNS_BY_CHOICE[where]:
        hit |= text_mask(texts[column], needle)
    return forecasts["forecast_key"].map(hit).fillna(value=False).astype(bool)


@dataclass(frozen=True)
class FilterResult:
    rows: pd.DataFrame
    nulls_dropped: dict[str, int]


def apply_filters(forecasts: pd.DataFrame, state: FilterState, load_texts: Callable[[], pd.DataFrame]) -> FilterResult:
    """The filtered rows in the chosen order, and how many null rows each narrowed range dropped."""
    mask = _all(forecasts)
    nulls_dropped: dict[str, int] = {}
    for spec in FILTER_SPECS:
        if spec.param not in state.values:
            continue
        extent = state.extents.get(spec.param)
        spec_rows = spec_mask(spec, forecasts, state.values[spec.param], extent)
        if spec.kind == KIND_RANGE and not spec_rows.all():
            nulls_dropped[spec.label] = int((mask & forecasts[spec.column].isna()).sum())
        mask &= spec_rows
    needle = str(state.values.get(TEXT_SEARCH_PARAM) or "")
    mask &= text_search_mask(forecasts, needle, str(state.values.get(TEXT_SEARCH_IN_PARAM, TEXT_IN_BOTH)), load_texts)
    ordered = forecasts[mask].sort_values(
        [state.sort, "forecast_key"], ascending=[not state.descending, True], na_position="last", kind="stable"
    )
    return FilterResult(rows=ordered, nulls_dropped={label: n for label, n in nulls_dropped.items() if n})


def neighbours(keys: list[str], key: str) -> tuple[str | None, str | None, int | None]:
    """``(previous, next, 1-based position)`` of ``key`` in the filtered order; all None when it is not there."""
    if key not in keys:
        return None, None, None
    index = keys.index(key)
    previous = keys[index - 1] if index > 0 else None
    following = keys[index + 1] if index + 1 < len(keys) else None
    return previous, following, index + 1


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _distinct(values: pd.Series) -> list:
    return sorted(values.dropna().unique().tolist(), key=str)


def _distinct_items(values: pd.Series) -> list:
    return sorted({item for items in values.dropna() for item in items}, key=str)


def _numeric_extent(values: pd.Series) -> tuple[float, float] | None:
    present = values.dropna().astype(float)
    if present.empty or present.min() == present.max():
        return None
    return float(present.min()), float(present.max())


def _date_extent(values: pd.Series) -> tuple[date, date] | None:
    present = values.dropna()
    if present.empty:
        return None
    return present.min().date(), present.max().date()


def _render_spec(spec: FilterSpec, forecasts: pd.DataFrame, state: FilterState) -> None:
    """One widget, bound to its query parameter; a filter with nothing to choose between is not drawn."""
    column, key = forecasts[spec.column], spec.param
    if spec.kind in (KIND_CHOICE, KIND_CONTAINS):
        options = _distinct(column) if spec.kind == KIND_CHOICE else _distinct_items(column)
        if len(options) > 1:
            state.values[key] = st.multiselect(spec.label, options, key=key, bind=BIND_TO_URL)
    elif spec.kind == KIND_RANGE:
        extent = _numeric_extent(column)
        if extent is not None:
            state.extents[key] = extent
            state.values[key] = st.slider(spec.label, extent[0], extent[1], extent, key=key, bind=BIND_TO_URL)
    elif spec.kind == KIND_DATES:
        date_extent = _date_extent(column)
        if date_extent is not None:
            state.extents[key] = date_extent
            chosen = st.date_input(spec.label, date_extent, *date_extent, key=key, bind=BIND_TO_URL)
            state.values[key] = tuple(chosen) if isinstance(chosen, tuple) else (chosen,)
    elif spec.kind in (KIND_TRISTATE, KIND_SHARE):
        options = TRISTATE_OPTIONS if spec.kind == KIND_TRISTATE else SHARE_OPTIONS
        state.values[key] = st.selectbox(spec.label, options, key=key, bind=BIND_TO_URL)
    else:
        state.values[key] = st.text_input(spec.label, key=key, bind=BIND_TO_URL)


def render_sidebar(forecasts: pd.DataFrame) -> FilterState:
    """Every filter, grouped in expanders, plus the full-text search and the sort; identical on every page."""
    state = FilterState()
    with st.sidebar:
        st.header("Filters")
        for group in FILTER_GROUPS:
            with st.expander(group, expanded=group == FILTER_GROUPS[0]):
                for spec in (spec for spec in FILTER_SPECS if spec.group == group):
                    _render_spec(spec, forecasts, state)
        with st.expander("Text search"):
            state.values[TEXT_SEARCH_PARAM] = st.text_input("Text contains", key=TEXT_SEARCH_PARAM, bind=BIND_TO_URL)
            state.values[TEXT_SEARCH_IN_PARAM] = st.selectbox(
                "Search in", TEXT_IN_OPTIONS, key=TEXT_SEARCH_IN_PARAM, bind=BIND_TO_URL
            )
        st.header("Order")
        state.sort = st.selectbox("Sort by", SORTABLE_COLUMNS, key=SORT_PARAM, bind=BIND_TO_URL)
        state.descending = st.toggle("Descending", value=True, key=DESC_PARAM, bind=BIND_TO_URL)
    return state
