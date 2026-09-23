"""Marker-spec registry + pure parser for bot run-log telemetry.

Each :class:`MarkerSpec` pairs a marker name (its archive-file stem) with a regex whose named
groups are the marker's fields, read against the ACTUAL emitted format string, and a
``qid_kind`` (the Metaculus id space its ``question=`` ref lives in: ``post_id`` or
``question_id``, ``None`` for markers with no question ref). ``parse_log_text`` matches each
line against every spec's regex via ``re.search``, so a spec is agnostic to the log-line prefix,
and breaks on the first match, so marker tokens are mutually exclusive per line.

Full detail per marker (emitter file and function, field semantics, receipts, dates, incidents),
the marker index, the ``qid_kind`` id-space rule, and the HTML-comment markers' harvesting-gap
note live in docs/telemetry_markers.md; each :class:`MarkerSpec` below carries only a one-line
pointer into that file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Why: kept local, not a metaculus_bot import, so this stays pure-stdlib. Receipt: docs/telemetry_markers.md "qid_kind".
QID_KIND_POST_ID = "post_id"
QID_KIND_QUESTION_ID = "question_id"

# Why: free text, never numerically coerced. Receipt: docs/telemetry_markers.md "Parsing notes".
_RAW_FIELDS: frozenset[str] = frozenset({"question", "summary", "forecast_json", "detail"})

# Why: no-data sentinels, not measured values. Receipt: docs/telemetry_markers.md "Parsing notes".
_NONE_SENTINELS: frozenset[str] = frozenset({"none", "n/a", "null"})

_INT_RE = re.compile(r"[+-]?\d+")
_LINE_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})")
_QID_URL_RE = re.compile(r"/questions/(\d+)")
_BARE_INT_RE = re.compile(r"\d+")

# Why: matches one key=value token of a generic counter tail. Receipt: docs/telemetry_markers.md "Parsing notes".
_KV_PAIR_RE = re.compile(r"(\w+)=([^,\s]+)")


@dataclass(frozen=True)
class MarkerSpec:
    """One telemetry marker: its archive-file stem, the field regex, and its id space.

    ``qid_kind`` names which Metaculus id space the marker's ``question=`` ref lives
    in (``"post_id"`` or ``"question_id"``); ``None`` for markers with no question
    ref (the credit markers). It is stamped onto every harvested record so a
    residual join knows how to translate a query into the record's id space instead
    of guessing (see the module docstring + ``performance_analysis.id_mapping``).

    ``raw_fields`` names fields of THIS spec kept verbatim rather than coerced, on top of
    the global ``_RAW_FIELDS``. That set is keyed by field name alone, so adding a name
    there changes its meaning on every spec that uses it (``thin_publish_floor.raw`` is a
    float, ``member_forecast.raw`` a JSON literal); a per-spec set keeps the two apart.
    """

    name: str
    regex: re.Pattern[str]
    qid_kind: str | None = None
    raw_fields: frozenset[str] = frozenset()


def coerce_value(raw: str | None) -> object:
    """Coerce a captured field string to bool / None / int / float, else keep the string.

    ``"True"``/``"true"`` -> ``True``; ``"n/a"``/``"None"``/``"null"`` -> ``None``;
    integer-looking -> ``int``; float-looking -> ``float``; everything else (model
    names, ``bound=upper``, ``qtype=binary``, ...) stays a ``str``.
    """
    if raw is None:
        return None
    text = raw.strip()
    low = text.lower()
    if low in _NONE_SENTINELS:
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if _INT_RE.fullmatch(text):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def qid_from_ref(ref: str | None) -> int | None:
    """Extract an integer question id from a Metaculus URL or a bare id string."""
    if ref is None:
        return None
    text = str(ref).strip()
    if text.lower() in _NONE_SENTINELS:
        return None
    url_match = _QID_URL_RE.search(text)
    if url_match:
        return int(url_match.group(1))
    if _BARE_INT_RE.fullmatch(text):
        return int(text)
    return None


def _parse_line_ts(line: str) -> str | None:
    """Extract the ``%(asctime)s`` prefix as an ISO-8601 string, or None if absent."""
    match = _LINE_TS_RE.match(line.lstrip())
    if not match:
        return None
    date, clock, millis = match.groups()
    return f"{date}T{clock}.{millis}000"


# Why: question= leads on gap-fill v2/ghost, trails elsewhere. Receipt: docs/telemetry_markers.md "Parsing notes".
MARKER_SPECS: list[MarkerSpec] = [
    MarkerSpec(
        "extraction_rung",
        re.compile(
            r"EXTRACTION_RUNG:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)"
            r"\s+qtype=(?P<qtype>\S+)\s+rung=(?P<rung>\S+)\s+block_present=(?P<block_present>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # value_extraction.py emits question.id_of_question
    ),
    MarkerSpec(
        "block_fallback",
        # Why: reasons runs to end of line; its block:/repair:/llm: prefixes keep coerce_value from ever converting it.
        re.compile(
            r"BLOCK_FALLBACK:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)\s+qtype=(?P<qtype>\S+)"
            r"\s+skipped=(?P<skipped>\d+)\s+rung=(?P<rung>\S+)\s+reasons=(?P<reasons>.*)$"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # value_extraction.py emits question.id_of_question
    ),
    MarkerSpec(
        "gap_fill_v2",
        # Why: optional tail groups keep pre-branch logs harvestable. Receipt: docs/telemetry_markers.md "GAP_FILL_V2".
        re.compile(
            r"(?:question=(?P<question>\S+)\s+)?GAP_FILL_V2:\s*model=(?P<model>.+?)"
            r"\s+steps=(?P<steps>\S+)\s+tool_calls=(?P<tool_calls>\S+)\s+searches=(?P<searches>\S+)"
            r"\s+fetches=(?P<fetches>\S+)\s+rendered=(?P<rendered>\S+)\s+reads=(?P<reads>\S+)"
            r"\s+dup_tool_calls=(?P<dup_tool_calls>\S+)\s+deadline_hit=(?P<deadline_hit>\S+)"
            r"\s+concluded_early=(?P<concluded_early>\S+)\s+wall_s=(?P<wall_s>\S+)"
            r"\s+findings=(?P<findings>\S+)\s+pending_leads=(?P<pending_leads>\S+)"
            r"\s+lint_rejections=(?P<lint_rejections>\S+)"
            r"(?:\s+provenance_rejections=(?P<provenance_rejections>\S+)"
            r"\s+quote_mismatch_warnings=(?P<quote_mismatch_warnings>\S+)"
            r"\s+plan_gaps=(?P<plan_gaps>\S+)\s+plan_skipped=(?P<plan_skipped>\S+)"
            r"\s+conclude_gate_rejections=(?P<conclude_gate_rejections>\S+)"
            r"(?:\s+error=(?P<error>.*))?)?"
        ),
        qid_kind=QID_KIND_POST_ID,  # agentic_gap_fill.py emits question.page_url (post id)
    ),
    MarkerSpec(
        "ghost_pre",
        # Why: the colon keeps GHOST_PRE: off GHOST_PRE_JSON. Receipt: docs/telemetry_markers.md "GHOST_PRE".
        re.compile(
            r"(?:question=(?P<question>\S+)\s+)?GHOST_PRE:\s*gaps=(?P<gaps>\S+)"
            r"\s+sensitive_assumptions=(?P<sensitive_assumptions>\S+)"
        ),
        qid_kind=QID_KIND_POST_ID,  # agentic_gap_fill.py log_prefix = question.page_url (post id)
    ),
    MarkerSpec(
        "ghost_pre_json",
        re.compile(r"(?:question=(?P<question>\S+)\s+)?GHOST_PRE_JSON:\s*(?P<forecast_json>\{.*\})\s*$"),
        qid_kind=QID_KIND_POST_ID,  # agentic_gap_fill.py log_prefix = question.page_url (post id)
    ),
    MarkerSpec(
        "ghost_forecast",
        re.compile(
            r"(?:question=(?P<question>\S+)\s+)?GHOST_FORECAST:\s*qtype=(?P<qtype>\S+)\s+summary=(?P<summary>.*)$"
        ),
        qid_kind=QID_KIND_POST_ID,  # agentic_gap_fill.py log_prefix = question.page_url (post id)
    ),
    MarkerSpec(
        "ghost_forecast_json",
        # Why: a colon after GHOST_FORECAST avoids collision. Receipt: docs/telemetry_markers.md "GHOST_FORECAST_JSON".
        re.compile(r"(?:question=(?P<question>\S+)\s+)?GHOST_FORECAST_JSON:\s*(?P<forecast_json>\{.*\})\s*$"),
        qid_kind=QID_KIND_POST_ID,  # agentic_gap_fill.py log_prefix = question.page_url (post id)
    ),
    MarkerSpec(
        "ghost_forecast_v1",
        # Why: the plain ghost's shape with a _V1 token, so one scorer reads both. Receipt: docs/telemetry_markers.md "GHOST_FORECAST_V1".
        re.compile(
            r"(?:question=(?P<question>\S+)\s+)?GHOST_FORECAST_V1:\s*qtype=(?P<qtype>\S+)\s+summary=(?P<summary>.*)$"
        ),
        qid_kind=QID_KIND_POST_ID,  # agentic_gap_fill.py log_prefix = question.page_url (post id)
    ),
    MarkerSpec(
        "ghost_forecast_v1_json",
        re.compile(r"(?:question=(?P<question>\S+)\s+)?GHOST_FORECAST_V1_JSON:\s*(?P<forecast_json>\{.*\})\s*$"),
        qid_kind=QID_KIND_POST_ID,  # agentic_gap_fill.py log_prefix = question.page_url (post id)
    ),
    MarkerSpec(
        "agentic_fetch_throttled",
        # Why: phrase is last since it may contain spaces. Receipt: docs/telemetry_markers.md "AGENTIC_FETCH_THROTTLED".
        re.compile(
            r"AGENTIC_FETCH_THROTTLED:\s*url=(?P<url>\S+)\s+method=(?P<method>\S+)"
            r"\s+chars=(?P<chars>\S+)\s+phrase=(?P<phrase>.*)"
        ),
    ),
    MarkerSpec(
        "agentic_fetch_local_doc",
        # Why: method splits pdf_local from digest_local. Receipt: docs/telemetry_markers.md "AGENTIC_FETCH_LOCAL_DOC".
        re.compile(
            r"AGENTIC_FETCH_LOCAL_DOC:\s*url=(?P<url>\S+)\s+method=(?P<method>\S+)"
            r"\s+chars=(?P<chars>\S+)\s+pages=(?P<pages>\S+)\s+passages=(?P<passages>\S+)"
        ),
    ),
    MarkerSpec(
        "agentic_urlcontext_robots_skip",
        # Why: cached per host, not per url. Receipt: docs/telemetry_markers.md "AGENTIC_URLCONTEXT_ROBOTS_SKIP".
        re.compile(r"AGENTIC_URLCONTEXT_ROBOTS_SKIP:\s*url=(?P<url>\S+)\s+host=(?P<host>\S+)"),
    ),
    MarkerSpec(
        "open_bound_piling",
        re.compile(
            r"OPEN_BOUND_PILING:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)"
            r"\s+bound=(?P<bound>\S+)\s+bin_mass=(?P<bin_mass>\S+)"
            r"\s+declared_edge=(?P<declared_edge>\S+)\s+bound_value=(?P<bound_value>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # numeric/diagnostics.py emits question.id_of_question
    ),
    MarkerSpec(
        "close_margin",
        re.compile(
            r"CLOSE_MARGIN:\s*question=(?P<question>\S+)\s+close_time=(?P<close_time>\S+)"
            r"\s+submitted_at=(?P<submitted_at>\S+)\s+window_s=(?P<window_s>\S+)"
            r"\s+margin_s=(?P<margin_s>\S+)\s+margin_frac=(?P<margin_frac>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # close_margin.py emits question.id_of_question
    ),
    MarkerSpec(
        "market_ranking",
        # Why: rendered is a comma list; \S+ takes it whole. Receipt: docs/telemetry_markers.md "MARKET_RANKING".
        re.compile(
            r"MARKET_RANKING:\s*question=(?P<question>\S+)\s+pool=(?P<pool>\S+)"
            r"\s+outcome=(?P<outcome>\S+)\s+rows=(?P<rows>\S+)"
            r"\s+prompt_chars=(?P<prompt_chars>\S+)\s+rendered=(?P<rendered>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # prediction_market.py emits question.id_of_question
    ),
    MarkerSpec(
        "market_child_render",
        # Why: named+collapsed==outcomes; else a render bug. Receipt: docs/telemetry_markers.md "MARKET_CHILD_RENDER".
        re.compile(
            r"MARKET_CHILD_RENDER:\s*question=(?P<question>\S+)\s+families=(?P<families>\S+)"
            r"\s+full_rows=(?P<full_rows>\S+)\s+ladder_rows=(?P<ladder_rows>\S+)"
            r"\s+outcomes=(?P<outcomes>\S+)\s+named=(?P<named>\S+)"
            r"\s+collapsed=(?P<collapsed>\S+)\s+withheld=(?P<withheld>\S+)"
            r"\s+max_stage=(?P<max_stage>\S+)\s+ladder_chars=(?P<ladder_chars>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # prediction_market.py emits question.id_of_question
    ),
    MarkerSpec(
        "market_ranking_degraded",
        # Why: not end-anchored for a future detail= line. Receipt: docs/telemetry_markers.md "MARKET_RANKING_DEGRADED".
        re.compile(
            r"MARKET_RANKING_DEGRADED:\s*question=(?P<question>\S+)\s+pool=(?P<pool>\S+)"
            r"\s+reason=(?P<reason>\S+)(?:\s+detail=(?P<detail>.*))?"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # prediction_market.py emits question.id_of_question
    ),
    MarkerSpec(
        "market_tier_capped",
        # Why: nothing fired yet; a first hit is the finding. Receipt: docs/telemetry_markers.md "MARKET_TIER_CAPPED".
        re.compile(
            r"MARKET_TIER_CAPPED:\s*question=(?P<question>\S+)\s+rows=(?P<rows>\S+)"
            r"\s+capped=(?P<capped>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # prediction_market.py emits question.id_of_question
    ),
    MarkerSpec(
        "numeric_degenerate_declaration",
        # Why: spread_applied gates the spreader. Receipt: docs/telemetry_markers.md "NUMERIC_DEGENERATE_DECLARATION".
        re.compile(
            r"NUMERIC_DEGENERATE_DECLARATION:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)"
            r"\s+n_unique=(?P<n_unique>\S+)\s+span=(?P<span>\S+)\s+value_eps=(?P<value_eps>\S+)"
            r"\s+spread_applied=(?P<spread_applied>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # numeric/pipeline.py emits question.id_of_question
    ),
    MarkerSpec(
        "numeric_aggregate_grid_mismatch",
        # Why: any record means a length drifted. Receipt: docs/telemetry_markers.md "NUMERIC_AGGREGATE_GRID_MISMATCH".
        re.compile(
            r"NUMERIC_AGGREGATE_GRID_MISMATCH:\s*question=(?P<question>\S+)"
            r"\s+model_index=(?P<model_index>\S+)\s+got_points=(?P<got_points>\S+)"
            r"\s+expected_points=(?P<expected_points>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # numeric/utils.py emits question.id_of_question
    ),
    MarkerSpec(
        "cdf_maxstep_clip",
        # Why: the two fields make the packing policy auditable. Receipt: docs/telemetry_markers.md "CDF_MAXSTEP_CLIP".
        re.compile(
            r"CDF_MAXSTEP_CLIP:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)"
            r"\s+clipped_mass=(?P<clipped_mass>\S+)\s+over_cap_bins=(?P<over_cap_bins>\S+)"
            r"\s+bins_displaced=(?P<bins_displaced>\S+)\s+max_offset_bins=(?P<max_offset_bins>\S+)"
            r"\s+pre_max_step=(?P<pre_max_step>\S+)\s+max_step=(?P<max_step>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # numeric/pchip_cdf.py is handed question.id_of_question
    ),
    MarkerSpec(
        "numeric_pchip_fallback",
        # Why: question renders N/A when the id was absent. Receipt: docs/telemetry_markers.md "NUMERIC_PCHIP_FALLBACK".
        re.compile(
            r"Question (?P<question>\S+): PCHIP CDF construction failed "
            r"\((?P<error>.*)\), falling back to forecasting-tools default"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # numeric/diagnostics.py emits question.id_of_question
    ),
    MarkerSpec(
        "spread_undefined",
        # Why: returns inf; a failure never reads as agreement. Receipt: docs/telemetry_markers.md "SPREAD_UNDEFINED".
        re.compile(
            r"SPREAD_UNDEFINED:\s*question=(?P<question>\S+)\s+qtype=(?P<qtype>\S+)"
            r"\s+denominator=(?P<denominator>\S+)\s+models=(?P<models>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # spread_metrics.py emits question.id_of_question
    ),
    MarkerSpec(
        "ts_anchor_route",
        # Why: covers routing-ELIGIBLE questions only, not all. Receipt: docs/telemetry_markers.md "TS_ANCHOR_ROUTE".
        re.compile(
            r"TS_ANCHOR_ROUTE:\s*question=(?P<question>\S+)\s+decision=(?P<decision>\S+)"
            r"\s+series=(?P<series>\S+)\s+step=(?P<step>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # ts_routing.py emits question.id_of_question
    ),
    MarkerSpec(
        "financial_stale_latest",
        # Why: surface splits two same-estimator emitters. Receipt: docs/telemetry_markers.md "FINANCIAL_STALE_LATEST".
        re.compile(
            r"FINANCIAL_STALE_LATEST:\s*surface=(?P<surface>\S+)\s+symbol=(?P<symbol>\S+)"
            r"\s+age_d=(?P<age_d>\S+)\s+cadence=(?P<cadence>\S+)"
        ),
    ),
    MarkerSpec(
        "financial_noise_flag",
        # Why: every field required; one emitter, one shape. Receipt: docs/telemetry_markers.md "FINANCIAL_NOISE_FLAG".
        re.compile(
            r"FINANCIAL_NOISE_FLAG:\s*surface=(?P<surface>\S+)\s+symbol=(?P<symbol>\S+)"
            r"\s+vr_lag=(?P<vr_lag>\S+)\s+vr=(?P<vr>\S+)\s+floor=(?P<floor>\S+)"
            r"\s+short_vol=(?P<short_vol>\S+)\s+long_vol=(?P<long_vol>\S+)"
            r"\s+robust_vol=(?P<robust_vol>\S+)"
        ),
    ),
    MarkerSpec(
        "fred_unknown_series",
        # Why: splits an invented id from a dead FRED link. Receipt: docs/telemetry_markers.md "FRED_UNKNOWN_SERIES".
        re.compile(r"FRED_UNKNOWN_SERIES:\s*series_id=(?P<series_id>\S+)\s+proposed_by=(?P<proposed_by>\S+)"),
    ),
    MarkerSpec(
        "asknews_no_articles",
        re.compile(
            r"ASKNEWS_NO_ARTICLES:\s*question=(?P<question>\S+)\s+hot=(?P<hot>\d+)\s+historical=(?P<historical>\d+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # providers.py logs question.id_of_question
    ),
    MarkerSpec(
        "resolution_source_fetch",
        # Why: reason/route/exc/digest/caller are optional, tail-keyed. Receipt: docs/telemetry_markers.md "RESOLUTION_SOURCE_FETCH".
        re.compile(
            r"RESOLUTION_SOURCE_FETCH:\s*question=(?P<question>\S+)\s+url=(?P<url>\S+)"
            r"\s+status=(?P<status>\S+)\s+http=(?P<http>\S+)\s+embeds=(?P<embeds>\S+)"
            r"(?:\s+reason=(?P<reason>\S+))?(?:\s+route=(?P<route>\S+))?"
            r"(?:\s+failure_class=(?P<failure_class>\S+))?(?:\s+exc=(?P<exc>\S+))?(?:\s+server=(?P<server>\S+))?"
            r"(?:\s+passages_returned=(?P<passages_returned>\S+))?(?:\s+passages_grounded=(?P<passages_grounded>\S+))?"
            r"(?:\s+fallback_used=(?P<fallback_used>\S+))?"
            r"(?:\s+caller=(?P<caller>\S+))?"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # resolution_source.py emits question.id_of_question
    ),
    MarkerSpec(
        "resolution_source_escalation",
        # Why: wall_s is that rung's own cost, caller is optional and tail-keyed. Receipt: docs/telemetry_markers.md "RESOLUTION_SOURCE_ESCALATION".
        re.compile(
            r"RESOLUTION_SOURCE_ESCALATION:\s*question=(?P<question>\S+)\s+url=(?P<url>\S+)"
            r"\s+from_status=(?P<from_status>\S+)\s+rung=(?P<rung>\S+)\s+outcome=(?P<outcome>\S+)"
            r"\s+wall_s=(?P<wall_s>\S+)(?:\s+caller=(?P<caller>\S+))?"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # resolution_source.py emits question.id_of_question
    ),
    MarkerSpec(
        "resolution_source_urlcontext_robots_skip",
        # Why: a paid call NOT billed. Receipt: docs/telemetry_markers.md "RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP".
        re.compile(r"RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP:\s*url=(?P<url>\S+)\s+host=(?P<host>\S+)"),
    ),
    MarkerSpec(
        "resolution_source_urlcontext_ungrounded_suppressed",
        # Why: paid, unserved. Receipt: docs/telemetry_markers.md "RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED".
        re.compile(
            r"RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED:\s*url=(?P<url>\S+)\s+statuses=(?P<statuses>\S+)"
        ),
    ),
    MarkerSpec(
        "resolution_source_urlcontext_not_addressed",
        # Why: paid; a true negative. Receipt: docs/telemetry_markers.md "RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED".
        re.compile(r"RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED:\s*url=(?P<url>\S+)\s+host=(?P<host>\S+)"),
    ),
    MarkerSpec(
        "rendered_fetch_off_host",
        # Why: same_publisher compares domains, not hosts. Receipt: docs/telemetry_markers.md "RENDERED_FETCH_OFF_HOST".
        re.compile(
            r"RENDERED_FETCH_OFF_HOST:\s*scope=(?P<scope>\S+)\s+pinned_host=(?P<pinned_host>\S+)"
            r"\s+landed_host=(?P<landed_host>\S+)\s+same_publisher=(?P<same_publisher>true|false)"
        ),
    ),
    MarkerSpec(
        "forecaster_drops",
        # Why: detail is verbatim JSON so slash slugs survive. Receipt: docs/telemetry_markers.md "FORECASTER_DROPS".
        re.compile(
            r"FORECASTER_DROPS:\s*total=(?P<total>\S+)\s+systematic=(?P<systematic>\S+)\s+detail=(?P<detail>\{.*\})\s*$"
        ),
    ),
    MarkerSpec(
        "systematic_forecaster_failure",
        # Why: the regex ends at causes=. Receipt: docs/telemetry_markers.md "SYSTEMATIC_FORECASTER_FAILURE".
        re.compile(
            r"SYSTEMATIC_FORECASTER_FAILURE:\s*model=(?P<model>\S+)\s+dropped_on_questions=(?P<dropped_on_questions>\d+)"
            r"\s+qids=(?P<qids>\S+)\s+causes=(?P<causes>\S+)"
        ),
    ),
    MarkerSpec(
        "forecasters_survived",
        # Why: on stdout, unlike its twin FORECASTERS_USED. Receipt: docs/telemetry_markers.md "FORECASTERS_SURVIVED".
        re.compile(
            r"FORECASTERS_SURVIVED:\s*question=(?P<question>\S+)\s+survived=(?P<survived>\d+)/(?P<configured>\d+)"
            r"\s+models=(?P<models>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # forecaster.py emits question.id_of_question
    ),
    MarkerSpec(
        "extreme_call",
        # Why: lone, not p, is the finding: neighbours change it. Receipt: docs/telemetry_markers.md "EXTREME_CALL".
        re.compile(
            r"EXTREME_CALL:\s*question=(?P<question>\S+)\s+model=(?P<model>\S+)\s+p=(?P<p>\S+)"
            r"\s+side=(?P<side>\S+)\s+lone=(?P<lone>\S+)\s+survivors=(?P<survivors>\d+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # forecaster.py emits question.id_of_question
    ),
    MarkerSpec(
        "thin_publish_floor",
        # Why: count IS the floor's own prod incidence. Receipt: docs/telemetry_markers.md "THIN_PUBLISH_FLOOR".
        re.compile(
            r"THIN_PUBLISH_FLOOR:\s*question=(?P<question>\S+)\s+raw=(?P<raw>\S+)\s+clamped=(?P<clamped>\S+)"
            r"\s+survivors=(?P<survivors>\d+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # same id space as forecasters_survived / extreme_call
    ),
    MarkerSpec(
        "member_forecast",
        # Why: raw/published stay verbatim so lines coerce alike. Receipt: docs/telemetry_markers.md "MEMBER_FORECAST".
        re.compile(
            r"MEMBER_FORECAST:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)\s+role=(?P<role>\S+)"
            r"\s+qtype=(?P<qtype>\S+)\s+raw=(?P<raw>\S+)\s+published=(?P<published>\S+)"
            r"(?:\s+oor_low=(?P<oor_low>\S+)\s+oor_high=(?P<oor_high>\S+))?"
            r"(?:\s+elicitation=(?P<elicitation>\S+))?"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # every emitter passes question.id_of_question
        raw_fields=frozenset({"raw", "published"}),
    ),
    MarkerSpec(
        "numeric_aggregate",
        # Why: method outranks the strategy and STACKER_OUTCOME. Receipt: docs/telemetry_markers.md "NUMERIC_AGGREGATE".
        re.compile(
            r"NUMERIC_AGGREGATE:\s*question=(?P<question>\S+)\s+qtype=(?P<qtype>\S+)"
            r"\s+cdf_size=(?P<cdf_size>\d+)\s+oor_low=(?P<oor_low>\S+)\s+oor_high=(?P<oor_high>\S+)"
            r"(?:\s+oor_low_raw=(?P<oor_low_raw>\S+)\s+oor_high_raw=(?P<oor_high_raw>\S+)"
            r"\s+tail_floor=(?P<tail_floor>\S+))?"
            r"(?:\s+method=(?P<method>\S+))?"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # forecaster.py passes question.id_of_question
    ),
    MarkerSpec(
        "degradation_counters",
        # Why: kv_pairs parses generically, no regex change. Receipt: docs/telemetry_markers.md "DEGRADATION_COUNTERS".
        re.compile(r"Degradation counters:\s*(?P<kv_pairs>.*)$"),
    ),
    MarkerSpec(
        "provider_degradation",
        # Why: emitted at findings=0 too; zero is recorded. Receipt: docs/telemetry_markers.md "PROVIDER_DEGRADATION".
        re.compile(
            r"PROVIDER_DEGRADATION:\s*run=(?P<run>\S+)\s+findings=(?P<findings>\d+)"
            r"\s+alertable=(?P<alertable>\d+)\s+suppressed=(?P<suppressed>\d+)"
            r"(?:\s+venues_observed=(?P<venues_observed>\d+)"
            r"\s+catalogues_observed=(?P<catalogues_observed>\d+)"
            r"\s+pool_rows=(?P<pool_rows>\d+))?"
            r"\s+detail=(?P<detail>\[.*?\])"
        ),
    ),
    MarkerSpec(
        "publish_hardening",
        # Why: attempt N/M excludes other same-prefix lines. Receipt: docs/telemetry_markers.md "PUBLISH_HARDENING".
        re.compile(
            r"PUBLISH_HARDENING:\s*(?P<method>\S+)\s+attempt\s+(?P<attempt>\d+)/(?P<attempts>\d+)\s+"
            r"(?:timed out after (?P<timeout_s>\d+)s"
            r"|failed \((?P<error_type>[^:()]+):\s*(?P<error>.*)\))\s*$"
        ),
    ),
    MarkerSpec(
        "publish_skipped_closed",
        # Why: overdue_s is negative under state_closed. Receipt: docs/telemetry_markers.md "PUBLISH_SKIPPED_CLOSED".
        re.compile(
            r"PUBLISH_SKIPPED_CLOSED:\s*question=(?P<question>\S+)\s+reason=(?P<reason>\S+)"
            r"\s+close_time=(?P<close_time>\S+)\s+now=(?P<now>\S+)"
            r"\s+overdue_s=(?P<overdue_s>\S+)\s+state=(?P<state>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # publish_gate.py emits question.id_of_question
    ),
    MarkerSpec(
        "mantic_question",
        # Why: type is as it arrived, before the client's rewrite. Receipt: docs/telemetry_markers.md "MANTIC_QUESTION".
        re.compile(
            r"MANTIC_QUESTION:\s*post=(?P<post>\S+)\s+question=(?P<question>\S+)\s+type=(?P<type>\S+)"
            r"\s+cdf_size=(?P<cdf_size>\S+)\s+multi_resolution=(?P<multi_resolution>\S+)"
            r"\s+date_granularity=(?P<date_granularity>\S+)\s+precision=(?P<precision>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # mantic.py emits question.id_of_question as question=
    ),
    MarkerSpec(
        "mantic_post_dropped",
        # Why: a post census; it never became a question. Receipt: docs/telemetry_markers.md "MANTIC_POST_DROPPED".
        re.compile(r"MANTIC_POST_DROPPED:\s*post=(?P<post>\S+)\s+type=(?P<type>\S+)\s+error=(?P<error>\S+)"),
    ),
    MarkerSpec(
        "mantic_tournaments",
        # Why: new lists slugs not the configured one. Receipt: docs/telemetry_markers.md "MANTIC_TOURNAMENTS".
        re.compile(
            r"MANTIC_TOURNAMENTS:\s*ongoing=(?P<ongoing>\S+)\s+configured=(?P<configured>\S+)\s+new=(?P<new>\S+)"
        ),
    ),
    MarkerSpec(
        "time_budget",
        # Why: runs for every question; CLOSE_MARGIN is censored. Receipt: docs/telemetry_markers.md "TIME_BUDGET".
        re.compile(
            r"TIME_BUDGET:\s*question=(?P<question>\S+)\s+budget_s=(?P<budget_s>\S+)"
            r"\s+close_time=(?P<close_time>\S+)\s+close_limited=(?P<close_limited>\S+)"
            r"\s+fast_path=(?P<fast_path>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # time_budget.py emits question.id_of_question
    ),
    MarkerSpec(
        "time_budget_fast_path",
        # Why: close_time stops at the semicolon, not \S+. Receipt: docs/telemetry_markers.md "TIME_BUDGET_FAST_PATH".
        re.compile(
            r"TIME_BUDGET_FAST_PATH:\s*qid=(?P<question>\S+)\s+budget=(?P<budget_s>[\d.]+)s\s+close_time=(?P<close_time>[^;]+);"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # forecaster.py emits question.id_of_question
    ),
    MarkerSpec(
        "wallclock_abort",
        # Why: the qid= anchor keeps a prose line out. Receipt: docs/telemetry_markers.md "WALLCLOCK_ABORT".
        re.compile(
            r"WALLCLOCK_ABORT:\s*qid=(?P<question>\S+)\s+elapsed=(?P<elapsed_s>[\d.]+)s"
            r"\s+forecasters_completed=(?P<forecasters_completed>\d+)/(?P<forecasters_configured>\d+)"
            r"\s+cancelled=(?P<cancelled>\d+)\s+remaining_budget=(?P<remaining_budget_s>-?[\d.]+)s"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # forecaster.py emits question.id_of_question
    ),
    MarkerSpec(
        "research_phase_deadline",
        # Why: counts and names only; no question ref. Receipt: docs/telemetry_markers.md "RESEARCH_PHASE_DEADLINE".
        re.compile(
            r"RESEARCH_PHASE_DEADLINE:\s*cancelled (?P<cancelled>\d+)/(?P<total>\d+) providers"
            r" after (?P<deadline_s>[\d.]+)s \((?P<providers>[^)]*)\)"
        ),
    ),
    MarkerSpec(
        "gap_fill_skipped_for_budget",
        # Why: research_phase_remaining reads n/a. Receipt: docs/telemetry_markers.md "GAP_FILL_SKIPPED_FOR_BUDGET".
        re.compile(
            r"GAP_FILL_SKIPPED_FOR_BUDGET:\s*question=(?P<question>\S+)"
            r"\s+fast_path=(?P<fast_path>\S+)\s+research_phase_remaining=(?P<research_phase_remaining>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # orchestrator logs question.id_of_question
    ),
    MarkerSpec(
        "gap_fill_cut_for_budget",
        # Why: gap_fill_pass is V1 or V2. Receipt: docs/telemetry_markers.md "GAP_FILL_CUT_FOR_BUDGET".
        re.compile(r"GAP_FILL_(?P<gap_fill_pass>V1|V2)_CUT_FOR_BUDGET:\s*question=(?P<question>\S+);"),
        qid_kind=QID_KIND_QUESTION_ID,  # orchestrator logs question.id_of_question
    ),
    MarkerSpec(
        "paid_personal_key_fallback",
        # Why: names which model fell back, and why. Receipt: docs/telemetry_markers.md "PAID_PERSONAL_KEY_FALLBACK".
        re.compile(
            r"PAID PERSONAL-KEY FALLBACK:\s*donated OpenRouter key failed for model=(?P<model>\S+?),"
            r".*?error=(?P<error_type>[^:]+):\s*(?P<error>.*)$"
        ),
    ),
    MarkerSpec(
        "run_alertable_summary",
        # Why: fires on every path; "clean" is the token. Receipt: docs/telemetry_markers.md "RUN_ALERTABLE_SUMMARY".
        re.compile(
            r"Run completed (?:(?P<outcome>clean) )?with (?P<alertable>\S+) alertable degradation event\(s\)\s*"
            r"\(bot=(?P<bot>\S+?), personal_key_fallback=(?P<personal_key_fallback>\S+?) of which "
            r"donated_404=(?P<donated_404>\S+?), credit=(?P<credit>\S+?)"
            r"(?: with (?P<suppressed_credit>\S+?) credit event\(s\) suppressed until (?P<resume_date>\S+?))?"
            r"(?:, donated_key=(?P<donated_key>\S+?))?"
            # Why: mantic_post_drops renders only when non-zero; absent harvests as None.
            r"(?:, mantic_post_drops=(?P<mantic_post_drops>\S+?))?\);"
        ),
    ),
    MarkerSpec(
        "only_posts",
        # Why: says which question a one-question paid run spent on. Receipt: docs/telemetry_markers.md "ONLY_POSTS".
        re.compile(r"ONLY_POSTS:\s*requested=(?P<requested>\S+)\s+matched=(?P<matched>\S+)\s+dropped=(?P<dropped>\d+)"),
    ),
    MarkerSpec(
        "question_cap_forfeit",
        # Why: each row is a paid forecast never made. Receipt: docs/telemetry_markers.md "QUESTION_CAP_FORFEIT".
        re.compile(
            r"QUESTION_CAP_FORFEIT:\s*platform=(?P<platform>\S+)\s+cap=(?P<cap>\d+)\s+total=(?P<total>\d+)"
            r"\s+dropped=(?P<dropped>\d+)\s+posts=(?P<posts>\S+)"
        ),
    ),
    MarkerSpec(
        "skip_guard_unreadable",
        # Why: the re-spend guard failed shut on this post. Receipt: docs/telemetry_markers.md "SKIP_GUARD_UNREADABLE".
        re.compile(
            r"SKIP_GUARD_UNREADABLE:\s*question=(?P<question>\S+)\s+post_id=(?P<post_id>\d+)"
            r"\s+platform=(?P<platform>\S+)\s+reason=(?P<reason>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # forecaster.py emits question.id_of_question beside the post id
    ),
    MarkerSpec(
        "gemini_ungrounded_suppressed",
        # Why: the only signal of a suppression. Receipt: docs/telemetry_markers.md "GEMINI_UNGROUNDED_SUPPRESSED".
        re.compile(
            r"GEMINI_UNGROUNDED_SUPPRESSED:\s*question=(?P<question>\S+)\s+model=(?P<model>.+?)"
            r"\s+queries=(?P<queries>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # gemini_search.py passes question.id_of_question
    ),
    MarkerSpec(
        "gemini_self_citation",
        # Why: records the self-citation verification counts before the floor. Receipt: docs/telemetry_markers.md "GEMINI_SELF_CITATION".
        re.compile(
            r"GEMINI_SELF_CITATION:\s*question=(?P<question>\S+)\s+model=(?P<model>\S+)"
            r"\s+links=(?P<links>\S+)\s+unique=(?P<unique>\S+)\s+resolved=(?P<resolved>\S+)"
            r"\s+unverified=(?P<unverified>\S+)\s+sources=(?P<sources>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # gemini_search.py passes question.id_of_question
    ),
    MarkerSpec(
        "gemini_grounding_density",
        # Why: historical production telemetry remains parseable. Receipt: docs/telemetry_markers.md "GEMINI_GROUNDING_DENSITY".
        re.compile(
            r"GEMINI_GROUNDING_DENSITY:\s*question=(?P<question>\S+)\s+chunks=(?P<chunks>\S+)"
            r"\s+supports=(?P<supports>\S+)\s+chars=(?P<chars>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # gemini_search.py passes question.id_of_question
    ),
    MarkerSpec(
        "gemini_unsupported_attribution",
        # Why: emitted only when unsupported > 0. Receipt: docs/telemetry_markers.md "GEMINI_UNSUPPORTED_ATTRIBUTION".
        re.compile(
            r"GEMINI_UNSUPPORTED_ATTRIBUTION:\s*question=(?P<question>\S+)\s+tagged=(?P<tagged>\S+)"
            r"\s+unsupported=(?P<unsupported>\S+)\s+groups=(?P<groups>\S+)\s+labels=(?P<labels>\S+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # gemini_search.py passes question.id_of_question
    ),
    MarkerSpec(
        "gemini_usage",
        # Why: one spec for three surfaces; question= is optional. Receipt: docs/telemetry_markers.md "GEMINI_USAGE".
        re.compile(
            r"GEMINI_USAGE:\s*role=(?P<role>\S+)\s+model=(?P<model>\S+)"
            r"\s+prompt_tokens=(?P<prompt_tokens>\S+)"
            r"\s+tool_use_prompt_tokens=(?P<tool_use_prompt_tokens>\S+)"
            r"\s+candidates_tokens=(?P<candidates_tokens>\S+)"
            r"\s+thoughts_tokens=(?P<thoughts_tokens>\S+)\s+total_tokens=(?P<total_tokens>\S+)"
            r"\s+search_queries=(?P<search_queries>\S+)(?:\s+question=(?P<question>\S+))?"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # gemini_search.py passes question.id_of_question
    ),
    MarkerSpec(
        "agentic_document_ungrounded_suppressed",
        # Why: optional statuses= tail. Receipt: docs/telemetry_markers.md "AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED".
        re.compile(r"AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED:\s*url=(?P<url>\S+)(?:\s+statuses=(?P<statuses>\S+))?"),
    ),
    MarkerSpec(
        "gap_fill_analyzer_failed",
        # Why: gates the pass; the only durable signal. Receipt: docs/telemetry_markers.md "GAP_FILL_ANALYZER_FAILED".
        re.compile(
            r"GAP_FILL_ANALYZER_FAILED:\s*question=(?P<question>\S+)\s+error=(?P<error>\S+)"
            r"(?:\s+detail=(?P<detail>.*))?$"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # targeted.py passes question.id_of_question
    ),
    MarkerSpec(
        "gap_fill_v1_triage",
        # Why: emitted only when the analyzer answered; listed=0 is "no parsable gaps", not a dead analyzer. Receipt: docs/telemetry_markers.md "GAP_FILL_V1_TRIAGE".
        re.compile(
            r"GAP_FILL_V1_TRIAGE:\s*question=(?P<question>\S+)\s+listed=(?P<listed>\d+)\s+kept=(?P<kept>\d+)"
            r"\s+dropped_not_answerable=(?P<dropped_not_answerable>\d+)"
            r"\s+dropped_in_first_pass=(?P<dropped_in_first_pass>\d+)"
            r"\s+dropped_same_need=(?P<dropped_same_need>\d+)"
            r"\s+dropped_schema=(?P<dropped_schema>\d+)"
            r"\s+dropped_over_cap=(?P<dropped_over_cap>\d+)"
        ),
        qid_kind=QID_KIND_QUESTION_ID,  # targeted.py passes question.id_of_question
    ),
    MarkerSpec(
        "credit_balance",
        re.compile(
            r"CREDIT_BALANCE:\s*key=(?P<key>\S+)\s+phase=(?P<phase>\S+)"
            r"(?:\s+remaining=(?P<remaining>\S+)\s+usage=(?P<usage>\S+))?"
        ),
    ),
    MarkerSpec(
        "credit_spend",
        # Why: source= is optional so pre-2026-07-27 logs harvest. Receipt: docs/telemetry_markers.md "CREDIT_SPEND".
        re.compile(
            r"CREDIT_SPEND:\s*key=(?P<key>\S+)\s+run_delta_usd=(?P<run_delta_usd>\S+)\s+remaining=(?P<remaining>\S+)"
            r"(?:\s+source=(?P<source>\S+))?"
        ),
    ),
    MarkerSpec(
        "credit_role_spend",
        # Why: the three 2026-09-09 tails are optional so older rows parse. Receipt: docs/telemetry_markers.md "CREDIT_ROLE_SPEND".
        re.compile(
            r"CREDIT_ROLE_SPEND:\s*role=(?P<role>\S+)\s+key=(?P<key>\S+)\s+usd=(?P<usd>\S+)\s+calls=(?P<calls>\d+)"
            r"\s+costed_calls=(?P<costed_calls>\d+)\s+byok_usd=(?P<byok_usd>\S+)"
            r"(?:\s+prompt_tokens=(?P<prompt_tokens>\d+)\s+completion_tokens=(?P<completion_tokens>\d+)"
            r"\s+cached_tokens=(?P<cached_tokens>\d+)\s+reasoning_tokens=(?P<reasoning_tokens>\d+))?"
            r"(?:\s+charged_usd=(?P<charged_usd>\S+)\s+byok_calls=(?P<byok_calls>\d+))?"
            r"(?:\s+max_prompt_tokens=(?P<max_prompt_tokens>\d+))?"
        ),
    ),
    MarkerSpec(
        "credit_run_summary",
        # Why: one line per run on every path; n/a and none are sentinels. Receipt: docs/telemetry_markers.md "CREDIT_RUN_SUMMARY".
        re.compile(
            r"CREDIT_RUN_SUMMARY:\s*n_questions=(?P<n_questions>\d+)\s+charged_usd=(?P<charged_usd>\S+)"
            r"\s+usd_per_question=(?P<usd_per_question>\S+)\s+donated_usd=(?P<donated_usd>\S+)"
            r"\s+personal_usd=(?P<personal_usd>\S+)\s+prompt_tokens=(?P<prompt_tokens>\d+)"
            r"\s+cached_tokens=(?P<cached_tokens>\d+)\s+cached_share=(?P<cached_share>\S+)"
            r"\s+max_prompt_tokens=(?P<max_prompt_tokens>\d+)\s+max_prompt_role=(?P<max_prompt_role>\S+)"
        ),
    ),
    MarkerSpec(
        "prompt_size_alert",
        # Why: question= is n/a off the v2 driver, the one call site that stamps it. Receipt: docs/telemetry_markers.md "PROMPT_SIZE_ALERT".
        re.compile(
            r"PROMPT_SIZE_ALERT:\s*role=(?P<role>\S+)\s+question=(?P<question>\S+)"
            r"\s+prompt_tokens=(?P<prompt_tokens>\d+)\s+threshold=(?P<threshold>\d+)"
        ),
        qid_kind=QID_KIND_POST_ID,  # the v2 driver stamps question.page_url (post id); every other role reads n/a
    ),
    MarkerSpec(
        "credit_floor_breach",
        re.compile(r"CREDIT_FLOOR_BREACH:\s*key=(?P<key>\S+)\s+remaining=(?P<remaining>\S+)\s+floor=(?P<floor>\S+)"),
    ),
    MarkerSpec(
        "donated_key_state",
        # Why: the state= anchor keeps the prose "DONATED_KEY_STATE: /auth/key probe failed" line out.
        re.compile(r"DONATED_KEY_STATE:\s*state=(?P<state>\S+)"),
    ),
    MarkerSpec(
        "litellm_callback_drain_timeout",
        # Why: the "within <n>s" clause is pinned. Receipt: docs/telemetry_markers.md "LITELLM_CALLBACK_DRAIN_TIMEOUT".
        re.compile(r"LITELLM_CALLBACK_DRAIN_TIMEOUT:.*?within (?P<timeout_s>[\d.]+)s"),
    ),
    MarkerSpec(
        "stacker_outcome",
        re.compile(
            r"<!--\s*STACKER_OUTCOME="
            r"(?P<outcome>primary|fallback_llm|fallback_median|fallback_mean|skipped_config_off|skipped)"
            r"\s*-->",
            re.IGNORECASE,
        ),
    ),
    MarkerSpec(
        "stacker_skip_reason",
        # Why: splits what plain "skipped" conflates. Receipt: docs/telemetry_markers.md "STACKER_SKIP_REASON".
        re.compile(
            r"<!--\s*STACKER_SKIP_REASON=(?P<reason>spread_below_threshold|spread_undefined"
            r"|config_off|single_forecaster|wall_clock_budget)\s*-->",
            re.IGNORECASE,
        ),
    ),
    MarkerSpec("tools_used", re.compile(r"<!--\s*TOOLS_USED=(?P<value>true|false)\s*-->", re.IGNORECASE)),
    MarkerSpec(
        "forecasters_used",
        # Why: in the comment, not stdout; here for completeness. Receipt: docs/telemetry_markers.md "FORECASTERS_USED".
        re.compile(r"<!--\s*FORECASTERS_USED=(?P<used>\d+)/(?P<configured>\d+)\s*-->", re.IGNORECASE),
    ),
]


def _build_record(
    spec: MarkerSpec,
    match: re.Match[str],
    *,
    line: str,
    seq: int,
    meta: dict[str, str],
) -> dict:
    """Assemble one archive record from a regex match + run metadata."""
    record: dict = {
        "marker": spec.name,
        "run_id": meta["run_id"],
        "workflow": meta["workflow"],
        "artifact": meta["artifact"],
        "run_date": meta["run_date"],
        "log_file": meta["log_file"],
        "seq": seq,
        "line_ts": _parse_line_ts(line),
    }
    for field, raw in match.groupdict().items():
        if field == "kv_pairs":
            # Why: an unemitted key is absent, never zero. Receipt: docs/telemetry_markers.md "DEGRADATION_COUNTERS".
            for key, value in _KV_PAIR_RE.findall(raw or ""):
                record[key] = coerce_value(value)
            continue
        record[field] = raw if (field in _RAW_FIELDS or field in spec.raw_fields) else coerce_value(raw)
    if "question" in record:
        record["qid"] = qid_from_ref(record["question"])
        # Why: stamps the id space so a residual join never guesses. Receipt: docs/telemetry_markers.md "qid_kind".
        record["qid_kind"] = spec.qid_kind
    return record


def parse_log_text(
    text: str,
    *,
    run_id: str,
    workflow: str,
    artifact: str,
    run_date: str,
    log_file: str,
) -> dict[str, list[dict]]:
    """Parse all telemetry markers from one log-text blob into per-marker record lists.

    ``seq`` is a per-marker ordinal within this blob; because a run's logs are parsed
    in stable order, re-harvesting produces byte-identical records — which the archive
    merge relies on for idempotent replace-by-run (see :mod:`scripts.telemetry.archive`).
    """
    meta = {
        "run_id": run_id,
        "workflow": workflow,
        "artifact": artifact,
        "run_date": run_date,
        "log_file": log_file,
    }
    harvested: dict[str, list[dict]] = {spec.name: [] for spec in MARKER_SPECS}
    counters: dict[str, int] = {spec.name: 0 for spec in MARKER_SPECS}

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        for spec in MARKER_SPECS:
            match = spec.regex.search(line)
            if match:
                harvested[spec.name].append(_build_record(spec, match, line=line, seq=counters[spec.name], meta=meta))
                counters[spec.name] += 1
                break  # marker tokens are mutually exclusive — one marker per line
    return harvested
