"""The three pages' bodies as functions; the page files only call them.

Every page renders the same sidebar (``filters.render_sidebar``), so the filters, held in the URL,
survive moving between pages. Platforms are never pooled: each summary block and each per-model
table is per platform, because Metaculus ranks on spot peer and Mantic on spot baseline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import streamlit as st

from metaculus_bot.constants import PLATFORM_MANTIC
from metaculus_bot.performance_analysis.parsing import parse_per_model_reasoning_text
from metaculus_bot.performance_analysis.scaling import NUMERIC_TYPES, cdf_and_grid
from metaculus_bot.trace_review import bundle
from metaculus_bot.trace_review.filters import (
    KEY_PARAM,
    PRESET_HELP,
    PRESETS,
    SORT_LABELS,
    SORT_PARAM,
    FilterResult,
    apply_filters,
    neighbours,
    render_sidebar,
)
from metaculus_bot.trace_review.plots import forecast_figure, format_quantity, replay_record
from metaculus_bot.trace_review.traits import TRAIT_KIND_FLAG, TRAIT_KIND_RATE, TRAITS

PAGE_FORECASTS = "app_pages/forecasts.py"
PAGE_DETAIL = "app_pages/detail.py"
PAGE_MODELS = "app_pages/models.py"
DETAIL_URL_PATH = "detail"
RATIONALE_COLUMNS = 3
PRESETS_PER_ROW = 3
# Tall enough to read a 10-15k-character rationale without nested scrolling on a 1000px-high window.
RATIONALE_HEIGHT_PX = 820
RATIONALE_VIEW_SIDE_BY_SIDE = "Side by side"
RATIONALE_VIEW_ONE = "One model at a time"
QUESTION_TYPE_SHORT = {"multiple_choice": "MC"}
QUESTION_TYPE_LABELS = {"multiple_choice": "multiple choice"}
RESOLUTION_LABELS = {
    "yes": "Yes",
    "no": "No",
    "below_lower_bound": "below the range",
    "above_upper_bound": "above the range",
}
EXTRACTION_RUNG_LABELS = {
    "block": "Value read from the model's structured forecast block.",
    "repair": "Value read after repairing a malformed structured forecast block.",
    "llm": "No usable structured block; a parser model extracted the value from the text.",
}
EXTRACTION_RUNG_UNRECORDED = "How the value was read was not recorded (no telemetry for this run)."
RESEARCH_SOURCE_LABELS = {
    "artifact": "saved research artifact",
    "comment_backfill": "copy recovered from the published comment",
    "log_backfill": "copy recovered from the run log",
}
NOT_RANKABLE_LABELS = {
    "anonymous_key": "the comment names no model",
    "stacked_record": "a stacker set the published value",
    "stacker_row": "this is the stacker",
    "sparse_anchors": "too few percentiles to rebuild its curve",
}
CHART_DENSITY = "Probability density"
CHART_DENSITY_DISCRETE = "Probability of each value"
CHART_CUMULATIVE = "Cumulative probability"
TRAIT_DIGITS = 1
TEXT_PANEL_HEIGHT_PX = 640
MEDIAN = 0.5
PLATFORM_LABELS = {"metaculus": "Metaculus", PLATFORM_MANTIC: "Mantic"}
SCORE_FIELD_LABELS = {
    "spot_peer_score": "spot peer",
    "peer_score": "peer (coverage-scaled fallback)",
    "spot_baseline_score": "spot baseline",
}
FLAG_COLUMNS: tuple[tuple[str, str], ...] = (
    ("any_drop", "drop"),
    ("any_extreme_call", "extreme"),
    ("fast_path", "fast path"),
    ("gfv2_loop_ran", "gap-fill v2"),
    ("anchor_present", "anchor"),
    ("degraded_run", "degraded"),
    ("partial_degraded", "partly degraded"),
    ("known_bug", "known bug"),
)
COHORT_COLUMNS: tuple[str, ...] = ("known_bug", "degraded_run", "partial_degraded")
TABLE_COLUMNS: dict[str, str] = {
    "title": "Question",
    "headline_score": "Score",
    "published": "Published",
    "resolution_raw": "Resolution",
    "question_type": "Type",
    "submitted": "Submitted",
    "tournament": "Tournament",
    "score_kind": "Score kind",
    "type_log_score": "Log score",
    "member_spread": "Spread",
    "forecasters": "Members",
    "aggregation": "Aggregation",
    "config_era": "Era",
    "flags": "Flags",
    "platform": "Platform",
    "url": "Link",
}
MISSING = "-"
# The question cell's label: the URL-encoded title after "#" (Streamlit decodes the capture group).
DETAIL_LINK_LABEL_RE = r"#(.*)$"
# A link cell cannot wrap or show a tooltip, so the question column is wide and can be widened further.
QUESTION_COLUMN_WIDTH_PX = 460
_HEADING_RE = re.compile(r"^(#{1,6}) ", re.MULTILINE)
# A fenced code block or an inline code span: left as written, since a backslash shows inside code.
_CODE_RE = re.compile(r"(```.*?(?:```|\Z)|`[^`\n]+`)", re.DOTALL)
# Unescaped $ (Streamlit reads $...$ as LaTeX) and ~ (GitHub markdown reads ~...~ as strikethrough).
_MARKDOWN_SPECIAL_RE = re.compile(r"(?<!\\)([$~])")
# Rationale and comment headings start at this level, so a model's "# PHASE 0" does not dwarf the page.
TOP_HEADING_LEVEL = 4
MAX_HEADING_LEVEL = 6


@dataclass(frozen=True)
class Loaded:
    directory: str
    built_at: str
    manifest: dict
    forecasts: pd.DataFrame
    members: pd.DataFrame


def load_bundle_or_stop() -> Loaded:
    """The bundle, or an error naming ``make trace_index`` and a stopped page."""
    directory = bundle.bundle_dir()
    try:
        manifest = bundle.load_manifest(directory)
    except bundle.BundleUnavailableError as error:
        st.error(str(error))
        st.stop()
    built_at = str(manifest["built_at"])
    return Loaded(
        directory=str(directory),
        built_at=built_at,
        manifest=manifest,
        forecasts=bundle.load_forecasts(str(directory), built_at),
        members=bundle.load_members(str(directory), built_at),
    )


def filtered(loaded: Loaded) -> FilterResult:
    state = render_sidebar(loaded.forecasts)
    return apply_filters(loaded.forecasts, state, lambda: bundle.load_search_texts(loaded.directory, loaded.built_at))


def _filters_only_params() -> dict[str, list[str]]:
    """Every filter parameter with all its values: a multiselect or a range repeats its parameter."""
    return {name: st.query_params.get_all(name) for name in st.query_params if name != KEY_PARAM}


# ---------------------------------------------------------------------------
# Display strings
# ---------------------------------------------------------------------------


def _number(value: object, digits: int = 1) -> str:
    return MISSING if value is None or pd.isna(value) else f"{float(value):+.{digits}f}"  # type: ignore[arg-type]


def readable_markdown(text: str) -> str:
    """Model-written markdown made safe to render: ``$`` and ``~`` escaped outside code, headings demoted.

    Rationales are full of dollar amounts and "~350" approximations, which Streamlit would otherwise
    render as LaTeX math (unwrappable, spaces dropped) and strikethrough.
    """
    levels = [len(match.group(1)) for match in _HEADING_RE.finditer(text)]
    shift = TOP_HEADING_LEVEL - min(levels) if levels else 0
    parts = _CODE_RE.split(text)
    for index in range(0, len(parts), 2):
        escaped = _MARKDOWN_SPECIAL_RE.sub(r"\\\1", parts[index])
        parts[index] = _HEADING_RE.sub(
            lambda match: "#" * min(len(match.group(1)) + shift, MAX_HEADING_LEVEL) + " ", escaped
        )
    return "".join(parts)


def published_median(forecast: pd.Series) -> float | None:
    grid_and_cdf = cdf_and_grid(replay_record(forecast))
    if grid_and_cdf is None:
        return None
    cdf, grid = grid_and_cdf
    return float(np.interp(MEDIAN, cdf, grid)) if cdf[0] <= MEDIAN <= cdf[-1] else None


def published_display(forecast: pd.Series) -> str:
    question_type = forecast["question_type"]
    if question_type == "binary":
        return f"{forecast['published_prob_yes']:.1%}"
    if question_type == "multiple_choice":
        values, options = list(forecast["published_values"]), list(forecast["options"])
        top = int(np.argmax(values))
        return f"{options[top]} ({values[top]:.0%})"
    median = published_median(forecast) if question_type in NUMERIC_TYPES else None
    return MISSING if median is None else f"median {format_quantity(median)}"


def score_label(forecast: pd.Series) -> str:
    """ "Metaculus spot peer +12.3": the score always carries its platform and its kind."""
    field = forecast["headline_score_field"]
    if pd.isna(field):
        return "no platform score"
    platform = PLATFORM_LABELS.get(forecast["platform"], forecast["platform"])
    return f"{platform} {SCORE_FIELD_LABELS.get(field, field)} {_number(forecast['headline_score'])}"


def _is_true(value: object) -> bool:
    return value is not pd.NA and value is not None and bool(value)


def flags_display(forecast: pd.Series) -> str:
    flags = [label for column, label in FLAG_COLUMNS if _is_true(forecast[column])]
    if forecast["record_origin"] == "fresh_pull":
        flags.append("since round")
    if _is_true(forecast["rescored_since_round"]):
        flags.append("rescored")
    return ", ".join(flags)


@st.cache_data(show_spinner=False)
def table_frame(directory: str, built_at: str) -> pd.DataFrame:
    """The list's display columns for every forecast, computed once per bundle."""
    forecasts = bundle.load_forecasts(directory, built_at).set_index("forecast_key", drop=False)
    display = pd.DataFrame(index=forecasts.index)
    display["submitted"] = forecasts["submitted_at"].dt.strftime("%Y-%m-%d")
    for column in ("title", "question_type", "platform", "tournament", "headline_score"):
        display[column] = forecasts[column]
    display["resolution_raw"] = [resolution_display(row) for _, row in forecasts.iterrows()]
    display["published"] = [published_display(row) for _, row in forecasts.iterrows()]
    display["score_kind"] = forecasts["headline_score_field"].map(SCORE_FIELD_LABELS)
    display["type_log_score"] = forecasts["type_log_score"]
    display["member_spread"] = forecasts["member_spread"]
    display["forecasters"] = forecasts["n_members"]
    display["aggregation"] = forecasts["aggregate_method"].fillna(forecasts["stacker_verdict"])
    display["config_era"] = forecasts["config_era"]
    display["flags"] = [flags_display(row) for _, row in forecasts.iterrows()]
    display["url"] = forecasts["url"]
    return display[list(TABLE_COLUMNS)]


# ---------------------------------------------------------------------------
# The forecast list
# ---------------------------------------------------------------------------


def _manifest_line(manifest: dict) -> None:
    excluded = manifest["excluded"]
    fresh = sum(n for key, n in manifest["rows_by_platform_origin"].items() if key.endswith(":fresh_pull"))
    st.caption(
        f"Built {manifest['built_at'][:16].replace('T', ' ')} UTC on round {manifest['round_label']}; "
        f"{fresh} rows from the fresh pull. Excluded from the tournaments pulled in this build: "
        f"{excluded['date_question']} date, {excluded['multi_resolution']} multi-resolution, "
        f"{excluded['annulled_or_ambiguous']} annulled or ambiguous, {excluded['practice_post']} practice. "
        f"Newest telemetry run: {(manifest['telemetry_newest_run_date'] or MISSING)[:10]}."
    )
    if manifest["fresh_pull_skipped"]:
        st.warning(f"The fresh pull was skipped: this index stops at round {manifest['round_label']}.")


def _presets() -> None:
    items = list(PRESETS.items())
    for start in range(0, len(items), PRESETS_PER_ROW):
        for column, (name, params) in zip(
            st.columns(PRESETS_PER_ROW), items[start : start + PRESETS_PER_ROW], strict=False
        ):
            column.page_link(PAGE_FORECASTS, label=name, query_params=params, help=PRESET_HELP[name], width="stretch")


def log_score_line(rows: pd.DataFrame) -> str:
    """ "binary 42.8 · discrete 35.3 · MC 29.8": the mean type log score per question type."""
    means = rows.groupby("question_type")["type_log_score"].mean().dropna()
    return (
        " · ".join(f"{QUESTION_TYPE_SHORT.get(str(kind), kind)} {mean:.1f}" for kind, mean in means.items()) or MISSING
    )


def platform_summary(rows: pd.DataFrame) -> dict[str, object]:
    """One platform's block: the mean headline over the rows that carry its main field."""
    main_field = rows["headline_score_field"].mode().iloc[0] if rows["headline_score_field"].notna().any() else None
    on_main = rows[rows["headline_score_field"] == main_field]
    return {
        "rows": len(rows),
        "field": SCORE_FIELD_LABELS.get(main_field, main_field) if main_field else MISSING,
        "mean_headline": float(on_main["headline_score"].mean()) if len(on_main) else None,
        "n_headline": len(on_main),
        "n_other_field": int(
            (rows["headline_score_field"].notna() & (rows["headline_score_field"] != main_field)).sum()
        ),
        "weight": float(rows["question_weight"].sum()) if rows["question_weight"].notna().any() else None,
        "log_score_by_type": log_score_line(rows),
    }


def _summary_strip(rows: pd.DataFrame) -> None:
    platforms = sorted(rows["platform"].unique())
    if not platforms:
        st.info("No forecast matches these filters.")
        return
    for column, platform in zip(st.columns(len(platforms)), platforms, strict=True):
        summary = platform_summary(rows[rows["platform"] == platform])
        with column.container(border=True):
            st.subheader(PLATFORM_LABELS.get(platform, platform))
            st.metric(f"Mean {summary['field']} over {summary['n_headline']} rows", _number(summary["mean_headline"]))
            st.caption(
                f"{summary['rows']} rows; {summary['n_other_field']} scored on another field, left out of the mean; "
                f"summed question weight {summary['weight'] if summary['weight'] is not None else MISSING}"
            )
            st.caption(f"Mean log score: {summary['log_score_by_type']}")
    cohorts = ", ".join(f"{int(rows[column].fillna(value=False).sum())} {column}" for column in COHORT_COLUMNS)
    st.caption(
        f"Means cover the filtered rows and are not era-bucketed unless an era filter is set. Included: {cohorts}."
    )
    with st.expander("By platform, config era and triple sub-era"):
        grouped = rows.groupby(["platform", "config_era", "triple_subera_fine"], dropna=False).agg(
            rows=("forecast_key", "size"),
            mean_headline=("headline_score", "mean"),
            mean_log_score=("type_log_score", "mean"),
        )
        st.dataframe(grouped.round(2))


def detail_links(keys: pd.Series, titles: pd.Series, filter_params: dict[str, list[str]]) -> pd.Series:
    """Detail-page URLs carrying the current filters; the title rides in the fragment for the cell's label."""
    query = urlencode(filter_params, doseq=True)
    prefix = f"{DETAIL_URL_PATH}?{query + '&' if query else ''}{KEY_PARAM}="
    return (
        prefix
        + keys.map(lambda key: quote(str(key), safe=":"))
        + "#"
        + titles.map(lambda title: quote(str(title), safe=""))
    )


def _forecast_table(loaded: Loaded, result: FilterResult) -> None:
    display = table_frame(loaded.directory, loaded.built_at).loc[result.rows["forecast_key"]]
    display = display.assign(
        title=detail_links(display.index.to_series(), display["title"], _filters_only_params()).to_numpy()
    )
    st.dataframe(
        display,
        hide_index=True,
        column_config={
            **{name: st.column_config.Column(label) for name, label in TABLE_COLUMNS.items()},
            "title": st.column_config.LinkColumn(
                "Question",
                display_text=DETAIL_LINK_LABEL_RE,
                width=QUESTION_COLUMN_WIDTH_PX,
                help="Opens the forecast in a new tab. Drag the column edge to widen it.",
            ),
            "headline_score": st.column_config.NumberColumn("Score", format="%+.1f"),
            "type_log_score": st.column_config.NumberColumn("Log score", format="%.1f"),
            "member_spread": st.column_config.NumberColumn("Spread", format="%.3f"),
            "url": st.column_config.LinkColumn("Link", display_text="open"),
        },
    )


def render_forecasts_page() -> None:
    loaded = load_bundle_or_stop()
    result = filtered(loaded)
    st.title("Forecasts")
    _manifest_line(loaded.manifest)
    _presets()
    if SORT_PARAM in st.query_params and st.query_params[SORT_PARAM] == SORT_LABELS["member_spread"]:
        st.caption(
            "Spread is a probability range on binary and multiple choice and a range-normalized value on numeric; pick one type to compare like with like."
        )
    _summary_strip(result.rows)
    for label, n in result.nulls_dropped.items():
        st.caption(f"{n} rows with no recorded {label.lower()} were dropped by that filter.")
    st.caption(f"{len(result.rows)} of {len(loaded.forecasts)} forecasts. Click a question to open it in a new tab.")
    _forecast_table(loaded, result)


# ---------------------------------------------------------------------------
# Forecast detail
# ---------------------------------------------------------------------------


def _navigation(keys: list[str], key: str) -> None:
    previous, following, position = neighbours(keys, key)
    with st.container(horizontal=True, vertical_alignment="center"):
        st.page_link(
            PAGE_FORECASTS, label="Back to the list", icon=":material/arrow_back:", query_params=_filters_only_params()
        )
        if st.button("Previous", disabled=previous is None) and previous:
            st.query_params[KEY_PARAM] = previous
            st.rerun()
        if st.button("Next", disabled=following is None) and following:
            st.query_params[KEY_PARAM] = following
            st.rerun()
        st.caption(f"{position} of {len(keys)} in the filtered list" if position else "Not in the filtered list")


def era_label(forecast: pd.Series) -> str:
    """The fine sub-era where there is one, since it is what tells two configs of one era apart."""
    fine = forecast["triple_subera_fine"]
    return f"era {forecast['config_era']} / {fine}" if pd.notna(fine) else f"era {forecast['config_era']}"


def resolution_display(forecast: pd.Series) -> str:
    raw = str(forecast["resolution_raw"])
    if forecast["question_type"] in NUMERIC_TYPES and pd.notna(forecast["resolution_value"]):
        return format_quantity(float(forecast["resolution_value"]))
    return RESOLUTION_LABELS.get(raw, raw)


def _header(forecast: pd.Series) -> None:
    st.title(forecast["title"])
    platform = PLATFORM_LABELS.get(forecast["platform"], forecast["platform"])
    question_type = QUESTION_TYPE_LABELS.get(forecast["question_type"], forecast["question_type"])
    st.markdown(
        f":blue-badge[{platform}] :gray-badge[{question_type}] :gray-badge[{forecast['tournament']}] "
        f":gray-badge[config {era_label(forecast)}] [open on {platform}]({forecast['url']})"
    )
    submitted = forecast["submitted_at"].strftime("%Y-%m-%d") if pd.notna(forecast["submitted_at"]) else MISSING
    resolved = (
        forecast["actual_resolve_time"].strftime("%Y-%m-%d") if pd.notna(forecast["actual_resolve_time"]) else MISSING
    )
    st.markdown(
        f"Submitted **{submitted}**, resolved **{resolved}** as **{resolution_display(forecast)}**. "
        f"Published **{published_display(forecast)}**. **{score_label(forecast)}**, log score {_number(forecast['type_log_score'])}."
    )
    flags = flags_display(forecast)
    if flags:
        st.caption(f"Flags: {flags}")


def _member_value(member: pd.Series) -> str:
    if pd.notna(member["prob_yes"]):
        return f"{member['prob_yes']:.1%}"
    if pd.notna(member["option_probs_json"]):
        return ", ".join(f"{option}: {p:.0%}" for option, p in json.loads(member["option_probs_json"]).items())
    if pd.notna(member["center_value"]):
        return f"median {format_quantity(float(member['center_value']))}"
    return MISSING


def _rationale_column(member: pd.Series, rationale: str | None) -> None:
    st.subheader(str(member["model"]))
    reason = (
        ""
        if member["rankable"]
        else f" (left out of model rankings: {NOT_RANKABLE_LABELS.get(member['not_rankable_reason'], member['not_rankable_reason'])})"
    )
    st.markdown(f"Forecast **{_member_value(member)}** · scored alone **{_number(member['own_score'])}**{reason}")
    rung = member["extraction_rung"]
    st.caption(EXTRACTION_RUNG_LABELS.get(rung, rung) if pd.notna(rung) else EXTRACTION_RUNG_UNRECORDED)
    if member["has_rationale"]:
        st.markdown(trait_badges(member))
    with st.container(height=RATIONALE_HEIGHT_PX):
        st.markdown(readable_markdown(rationale) if rationale else "_No rationale in the comment for this member._")


def _trait_value(trait_kind: str, value: object) -> str:
    if trait_kind == TRAIT_KIND_FLAG:
        return "yes" if _is_true(value) else "no"
    if trait_kind == TRAIT_KIND_RATE:
        return f"{float(value):.{TRAIT_DIGITS}f}"  # type: ignore[arg-type]
    return f"{int(value):,}"  # type: ignore[arg-type]


def trait_badges(member: pd.Series) -> str:
    """Every recorded trait as a "label: value" badge; a flag that is set is highlighted."""
    badges = []
    for trait in TRAITS:
        value = member[trait.name]
        if pd.isna(value):
            continue
        color = "violet" if trait.kind == TRAIT_KIND_FLAG and _is_true(value) else "gray"
        badges.append(f":{color}-badge[{trait.label}: {_trait_value(trait.kind, value)}]")
    return " ".join(badges)


def _rationales_tab(members: pd.DataFrame, comment_text: str | None, key: str) -> None:
    rationales = parse_per_model_reasoning_text(comment_text) if comment_text else {}
    models = members["model"].tolist()
    if not models:
        st.info("No member values were recovered for this forecast.")
        return
    view = st.radio("View", (RATIONALE_VIEW_SIDE_BY_SIDE, RATIONALE_VIEW_ONE), horizontal=True, key=f"view_{key}")
    st.caption(
        "Scored alone: the platform score this member's forecast would have earned had it been published on its own. "
        "The badges are counted from the rationale text by pattern matching, so treat them as rough."
    )
    if view == RATIONALE_VIEW_ONE:
        model = st.radio("Model", models, horizontal=True, key=f"model_{key}") or models[0]
        _rationale_column(members[members["model"] == model].iloc[0], rationales.get(model))
        return
    shown = models
    if len(models) > RATIONALE_COLUMNS:
        shown = st.multiselect("Members to show", models, default=models[:RATIONALE_COLUMNS], key=f"members_{key}")
    for start in range(0, len(shown), RATIONALE_COLUMNS):
        row_models = shown[start : start + RATIONALE_COLUMNS]
        for column, model in zip(st.columns(RATIONALE_COLUMNS), row_models, strict=False):
            with column:
                _rationale_column(members[members["model"] == model].iloc[0], rationales.get(model))


def split_at_shallowest_heading(text: str) -> list[tuple[str, str]]:
    """``[(heading, body)]`` split at the text's shallowest markdown heading level."""
    levels = [len(match.group(1)) for match in _HEADING_RE.finditer(text)]
    if not levels:
        return [("Research", text)]
    marker = "#" * min(levels) + " "
    parts = re.split(rf"^(?={re.escape(marker)})", text, flags=re.MULTILINE)
    sections = []
    for part in (p for p in parts if p.strip()):
        first_line, _, body = part.partition("\n")
        heading = first_line.removeprefix(marker).strip() if first_line.startswith(marker) else "Preamble"
        sections.append((heading, body if first_line.startswith(marker) else part))
    return sections


def _raw_payloads(path: str, question_id: int) -> list[dict]:
    lines = (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    return [line for line in lines if line.get("qid") == question_id]


def _research_tab(forecast: pd.Series, texts: dict[str, object]) -> None:
    source = forecast["research_source_class"]
    if pd.isna(source):
        st.info("No research record in the archive belongs to this forecast.")
        return
    trimmed = forecast["research_is_trimmed"]
    trimmed_text = "not recorded" if pd.isna(trimmed) else ("yes" if trimmed else "no")
    st.markdown(f"Source: **{RESEARCH_SOURCE_LABELS.get(source, source)}** · trimmed: **{trimmed_text}**")
    if source == "comment_backfill":
        st.caption(
            "This text is the lossy copy recovered from the published comment, not the research the forecasters saw."
        )
    if texts["provider_results_json"]:
        st.dataframe(
            pd.DataFrame(json.loads(str(texts["provider_results_json"]))), hide_index=True, placeholder=MISSING
        )
    for heading, body in split_at_shallowest_heading(str(texts["research_text"] or "")):
        with st.expander(heading):
            st.markdown(readable_markdown(body))
    raw_path = texts["raw_research_path"]
    if raw_path and st.toggle("Show raw provider payloads", key=f"raw_{forecast['forecast_key']}"):
        for payload in _raw_payloads(str(raw_path), int(forecast["question_id"])):
            with st.expander(
                f"{payload.get('provider')} ({payload.get('phase')}, {payload.get('payload_chars')} chars)"
            ):
                st.text(payload.get("payload"))


def _gap_fill_tab(forecast: pd.Series, texts: dict[str, object], telemetry: pd.DataFrame) -> None:
    st.caption(
        "The ghost forecast is the gap-fill v2 driver's own private forecast from its findings; "
        "it is logged for evaluation and never published."
    )
    if pd.notna(forecast["ghost_log_score"]):
        st.markdown(
            f"Ghost forecast log score **{_number(forecast['ghost_log_score'])}** against the published "
            f"{_number(forecast['type_log_score'])} (difference {_number(forecast['ghost_score_delta'])})."
        )
    else:
        st.markdown("No ghost forecast was scored for this question.")
    lines = telemetry[telemetry["marker"] == "gap_fill_v2"]
    if len(lines):
        with st.expander("The run's gap-fill v2 summary line (raw)"):
            st.json(json.loads(lines.iloc[-1]["payload_json"]))
    payload = json.loads(str(texts["gap_fill_v2_json"])) if texts["gap_fill_v2_json"] else None
    if payload is None:
        st.info("No gap-fill v2 record for this forecast.")
        return
    if st.toggle("Show the transcript", key=f"transcript_{forecast['forecast_key']}"):
        for index, message in enumerate(payload.get("transcript") or []):
            with st.expander(f"{index + 1}. {message.get('role', 'message')}"):
                st.text(str(message.get("content")))


def _telemetry_tab(telemetry: pd.DataFrame, key: str) -> None:
    if telemetry.empty:
        st.info("No telemetry for this forecast: its run predates the telemetry archive.")
        return
    markers = sorted(telemetry["marker"].unique())
    chosen = st.multiselect("Markers", markers, key=f"markers_{key}")
    shown = telemetry[telemetry["marker"].isin(chosen)] if chosen else telemetry
    st.dataframe(shown.drop(columns="forecast_key"), hide_index=True, placeholder=MISSING)


def _comment_tab(comment_text: str | None, key: str) -> None:
    if not comment_text:
        st.info("No published comment was found for this forecast.")
        return
    if st.toggle("Raw text", key=f"rawcomment_{key}"):
        st.code(comment_text, language=None, wrap_lines=True)
    else:
        with st.container(height=TEXT_PANEL_HEIGHT_PX):
            st.markdown(readable_markdown(comment_text))


def _forecast_chart(forecast: pd.Series, members: pd.DataFrame, key: str) -> None:
    cumulative = False
    if forecast["question_type"] in NUMERIC_TYPES:
        density = CHART_DENSITY_DISCRETE if forecast["question_type"] == "discrete" else CHART_DENSITY
        shown = st.radio("Chart", (density, CHART_CUMULATIVE), horizontal=True, key=f"chart_{key}")
        cumulative = shown == CHART_CUMULATIVE
    dark_theme = st.context.theme.type != "light"
    st.plotly_chart(
        forecast_figure(forecast, members, cumulative=cumulative, dark_theme=dark_theme), key=f"figure_{key}"
    )


def render_detail_page() -> None:
    loaded = load_bundle_or_stop()
    result = filtered(loaded)
    key = st.query_params.get(KEY_PARAM)
    forecasts = loaded.forecasts.set_index("forecast_key", drop=False)
    if key is None:
        st.info("Open a forecast from the list to read it here.")
        st.page_link(
            PAGE_FORECASTS, label="Go to the list", icon=":material/arrow_back:", query_params=_filters_only_params()
        )
        return
    if key not in forecasts.index:
        st.error(f"No forecast with key {key!r} in this index. Pick one from the list.")
        st.page_link(PAGE_FORECASTS, label="Back to the list", query_params=_filters_only_params())
        return
    forecast = forecasts[forecasts["forecast_key"] == key].iloc[0]
    members = loaded.members[loaded.members["forecast_key"] == key]
    _navigation(result.rows["forecast_key"].tolist(), key)
    _header(forecast)
    _forecast_chart(forecast, members, key)
    texts = bundle.load_texts_for(loaded.directory, loaded.built_at, key)
    telemetry = bundle.load_telemetry_for(loaded.directory, loaded.built_at, key)
    comment_text = str(texts["comment_text"]) if texts["comment_text"] else None
    rationales, research, gap_fill, telemetry_tab, comment = st.tabs(
        ["Rationales", "Research", "Gap-fill v2", "Telemetry", "Comment"]
    )
    with rationales:
        _rationales_tab(members, comment_text, key)
    with research:
        _research_tab(forecast, texts)
    with gap_fill:
        _gap_fill_tab(forecast, texts, telemetry)
    with telemetry_tab:
        _telemetry_tab(telemetry, key)
    with comment:
        _comment_tab(comment_text, key)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MEANED_MEMBER_TRAITS: tuple[str, ...] = ("rationale_chars", "percent_count", "hedge_per_1k", "base_rate_number_count")


def model_table(members: pd.DataFrame) -> pd.DataFrame:
    """Per platform, model and question type; every score mean is over rankable rows only."""
    rankable = members[members["rankable"].fillna(value=False)]
    group = ["platform", "model", "question_type"]
    table = members.groupby(group).agg(rows=("forecast_key", "size"), farthest_share=("is_farthest_member", "mean"))
    scored = rankable.groupby(group).agg(
        rankable=("forecast_key", "size"),
        mean_own_score=("own_score", "mean"),
        mean_minus_published=("score_minus_published", "mean"),
    )
    uncensored = rankable[~rankable["score_censored"].fillna(value=False)]
    ex_floor = uncensored.groupby(group)["own_score"].mean().rename("mean_own_score_ex_floor")
    traits = members[members["has_rationale"].fillna(value=False)].groupby(group)[list(MEANED_MEMBER_TRAITS)].mean()
    return table.join([scored, ex_floor, traits]).reset_index().round(3)


# Raw JSON columns are long and unreadable in a cell; the detail page shows what they hold.
MEMBER_ROW_HIDDEN_COLUMNS: tuple[str, ...] = (
    "option_probs_json",
    "percentiles_json",
    "raw_value_json",
    "published_value_json",
)


def render_models_page() -> None:
    loaded = load_bundle_or_stop()
    result = filtered(loaded)
    st.title("Models")
    st.caption(
        "Score means use only rankable rows: no stacker rows, no stacked records, no members the comment does not "
        "name, and no numeric curves with too few percentiles. They are not era-bucketed unless an era filter is set. "
        "Trait columns are means over the rationales. Hover a column header for what it measures."
    )
    members = loaded.members[loaded.members["forecast_key"].isin(result.rows["forecast_key"])]
    column_config = {
        "model": st.column_config.TextColumn("Model"),
        "question_type": st.column_config.TextColumn("Type"),
        "rows": st.column_config.NumberColumn("Rows"),
        "rankable": st.column_config.NumberColumn("Rankable rows", help="Rows that enter the score means"),
        "mean_own_score": st.column_config.NumberColumn(
            "Mean score alone", format="%+.1f", help="The score each forecast would have earned if published on its own"
        ),
        "mean_minus_published": st.column_config.NumberColumn(
            "Mean score alone minus published", format="%+.1f", help="Positive: the model alone beat the ensemble"
        ),
        "mean_own_score_ex_floor": st.column_config.NumberColumn(
            "Mean score alone, floor scores dropped",
            format="%+.1f",
            help="The same mean without numeric scores at the censoring floor",
        ),
        "farthest_share": st.column_config.NumberColumn(
            "Farthest member", format="percent", help="Share of forecasts where this model was farthest from the others"
        ),
        "rationale_chars": st.column_config.NumberColumn("Rationale length (characters)", format="%d"),
        "percent_count": st.column_config.NumberColumn("Percentages quoted", format="%.1f"),
        "hedge_per_1k": st.column_config.NumberColumn("Hedge words per 1,000 characters", format="%.2f"),
        "base_rate_number_count": st.column_config.NumberColumn("Base rates with a number", format="%.2f"),
    }
    table = model_table(members).sort_values(
        ["platform", "question_type", "rows", "model"], ascending=[True, True, False, True]
    )
    for platform in sorted(table["platform"].unique()):
        st.subheader(PLATFORM_LABELS.get(platform, platform))
        st.dataframe(
            table[table["platform"] == platform][list(column_config)],
            hide_index=True,
            column_config=column_config,
            placeholder=MISSING,
        )
    st.subheader("Member rows")
    st.caption("Every member row behind the tables above, with its stored columns.")
    shown = members.drop(columns=list(MEMBER_ROW_HIDDEN_COLUMNS)).assign(
        open=DETAIL_URL_PATH + "?" + KEY_PARAM + "=" + members["forecast_key"]
    )
    st.dataframe(
        shown[["open", *(column for column in shown.columns if column != "open")]],
        hide_index=True,
        placeholder=MISSING,
        column_config={"open": st.column_config.LinkColumn("Forecast", display_text="open", pinned=True)},
    )
