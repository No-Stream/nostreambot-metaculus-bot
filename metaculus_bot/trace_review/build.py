"""The trace-review builder: round file + fresh pulls + telemetry + research archive -> parquet bundle.

``build_index`` writes the four tables and the manifest into ``<output>/.building/`` and moves them
into place only after all five are written, so a failed build leaves the previous bundle as it was.
Cached pulls that finished before a failure are kept, so a retry is cheaper.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from metaculus_bot.constants import MANTIC_TOURNAMENT_ID, PLATFORM_MANTIC, PLATFORM_METACULUS, TOURNAMENT_ID
from metaculus_bot.performance_analysis import collector
from metaculus_bot.performance_analysis.collector import MANTIC_PLATFORM, METACULUS_PLATFORM
from metaculus_bot.performance_analysis.id_mapping import QuestionIdMap, record_platform
from metaculus_bot.performance_analysis.rescore_diff import diff_platform_rescores
from metaculus_bot.performance_analysis.research_tags import load_verified_archive_record
from metaculus_bot.performance_analysis.round_dataset import (
    DEGRADED_FULL_QUESTION_IDS,
    DEGRADED_PARTIAL_QUESTION_IDS,
    KNOWN_BUG_QUESTION_IDS,
    load_fresh,
    record_key,
    tag_fresh_records,
)
from metaculus_bot.trace_review.config import (
    BUILDING_DIR_NAME,
    DEFAULT_BUNDLE_DIR,
    DEFAULT_RESEARCH_LATEST_DIR,
    DEFAULT_RESEARCH_RAW_DIR,
    DEFAULT_SCRATCH_DIR,
    DEFAULT_TELEMETRY_DIR,
    FORECASTS_FILE,
    FRESH_PROVENANCE_PREFIX,
    MANIFEST_FILE,
    MEMBERS_FILE,
    PARQUET_COMPRESSION,
    PULLS_DIR_NAME,
    RECORD_ORIGIN_FRESH,
    RECORD_ORIGIN_ROUND,
    TELEMETRY_FILE,
    TEXTS_FILE,
    TEXTS_ROW_GROUP_SIZE,
)
from metaculus_bot.trace_review.forecast_rows import ForecastInputs, forecast_row
from metaculus_bot.trace_review.forecast_rows import type_log_score as record_type_log_score
from metaculus_bot.trace_review.member_rows import member_rows
from metaculus_bot.trace_review.schema import (
    FORECAST_COLUMNS,
    MEMBER_COLUMNS,
    SCHEMA_VERSION,
    TELEMETRY_COLUMNS,
    TEXT_COLUMNS,
    frame_from_rows,
)
from metaculus_bot.trace_review.sources import (
    EXCLUSION_KINDS,
    PullTarget,
    TelemetryIndex,
    build_telemetry_index,
    ensure_pulls,
    forecast_key_of,
    mantic_slugs,
    metaculus_slugs,
    newest_round_dataset,
    round_label_of,
)
from metaculus_bot.trace_review.traits import TRAIT_KIND_FLAG, TRAITS

logger: logging.Logger = logging.getLogger(__name__)

NO_PRIOR_ROUND_LABEL = "none"
ALL_ERAS = "all"


@dataclass(frozen=True)
class BuildConfig:
    """Where the builder reads and writes, which platforms it pulls and whether it may use the cache."""

    output_dir: Path = DEFAULT_BUNDLE_DIR
    round_path: Path | None = None
    platforms: tuple[str, ...] = (PLATFORM_METACULUS, PLATFORM_MANTIC)
    scratch_dir: Path = DEFAULT_SCRATCH_DIR
    telemetry_dir: Path = DEFAULT_TELEMETRY_DIR
    research_latest_dir: Path = DEFAULT_RESEARCH_LATEST_DIR
    research_raw_dir: Path = DEFAULT_RESEARCH_RAW_DIR
    refresh: bool = False
    skip_fresh_pull: bool = False


@dataclass
class AssembledRecords:
    """The round's records plus the overlay, tagged, with what each pull reported."""

    records: list[dict]
    round_count: int
    pulls: dict[str, dict[str, dict]] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    manifest: dict


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _load_pulled(target: PullTarget, provenance: str) -> list[dict]:
    return [
        record
        for slug in target.slugs
        for record in load_fresh(slug, target.perf_path(slug), provenance, required=True)
    ]


def _weights(metas: dict[str, dict], slug: str) -> dict[int, float]:
    return {int(qid): float(weight) for qid, weight in (metas.get(slug) or {}).get("question_weights", {}).items()}


def _mark_rescored_round_rows(round_records: list[dict], fresh: list[dict]) -> None:
    """Flag the round rows Metaculus changed since the round; the row keeps the round's values.

    Diffed on copies: ``diff_platform_rescores`` writes four tag fields onto the records it is
    given, one of which shares a name with a round tag of a different type.
    """
    diff = diff_platform_rescores(round_records, copy.deepcopy(fresh))
    changed = {(change.question_id, change.post_id) for change in diff.changes}
    compared = {record_key(record) for record in fresh}
    for record in round_records:
        key = record_key(record)
        record["rescored_since_round"] = (key in changed) if key in compared else None
    logger.info(f"rescored since the round: {len(changed)} of {diff.compared} compared round row(s)")


def _metaculus_overlay(
    round_records: list[dict], *, label: str, config: BuildConfig, now: datetime
) -> tuple[list[dict], dict]:
    round_keys = {record_key(record) for record in round_records}
    target = PullTarget(
        platform=METACULUS_PLATFORM,
        slugs=metaculus_slugs(round_records),
        platform_dir=config.output_dir / PULLS_DIR_NAME / PLATFORM_METACULUS,
        round_label=label,
        round_keys=round_keys,
    )
    metas = ensure_pulls(target, research_latest_dir=config.research_latest_dir, refresh=config.refresh, now=now)
    fresh = _load_pulled(target, f"{FRESH_PROVENANCE_PREFIX}{now.date().isoformat()}")
    _mark_rescored_round_rows(round_records, fresh)
    overlay = [record for record in fresh if record_key(record) not in round_keys]
    tag_fresh_records(
        overlay,
        prior_records=round_records,
        prior_label=label,
        weights=_weights(metas, TOURNAMENT_ID),
        weighted_slug=TOURNAMENT_ID,
        telemetry_dir=config.telemetry_dir,
    )
    logger.info(f"metaculus: {len(overlay)} record(s) resolved since round {label}")
    return overlay, metas


def _mantic_records(*, label: str, config: BuildConfig, now: datetime) -> tuple[list[dict], dict]:
    target = PullTarget(
        platform=MANTIC_PLATFORM,
        slugs=mantic_slugs(config.telemetry_dir),
        platform_dir=config.output_dir / PULLS_DIR_NAME / PLATFORM_MANTIC,
        round_label=label,
    )
    metas = ensure_pulls(target, research_latest_dir=config.research_latest_dir, refresh=config.refresh, now=now)
    records = _load_pulled(target, f"{FRESH_PROVENANCE_PREFIX}{now.date().isoformat()}")
    cohort_ids = KNOWN_BUG_QUESTION_IDS | DEGRADED_FULL_QUESTION_IDS | DEGRADED_PARTIAL_QUESTION_IDS
    colliding = sorted(r["question_id"] for r in records if r["question_id"] in cohort_ids)
    if colliding:
        # The cohort constants are bare Metaculus question ids; tagging cannot tell the platforms apart.
        raise ValueError(f"Mantic question id(s) {colliding} collide with a Metaculus cohort constant")
    tag_fresh_records(
        records,
        prior_records=[],
        prior_label=NO_PRIOR_ROUND_LABEL,
        weights=_weights(metas, MANTIC_TOURNAMENT_ID),
        weighted_slug=MANTIC_TOURNAMENT_ID,
        telemetry_dir=config.telemetry_dir,
    )
    logger.info(f"mantic: {len(records)} resolved record(s)")
    return records, metas


def assemble_records(round_records: list[dict], *, label: str, config: BuildConfig, now: datetime) -> AssembledRecords:
    """The round's rows, then (unless skipped) the Metaculus overlay and the Mantic pull, all tagged."""
    for record in round_records:
        record["record_origin"] = RECORD_ORIGIN_ROUND
        record["rescored_since_round"] = None
    assembled = AssembledRecords(records=list(round_records), round_count=len(round_records))
    if config.skip_fresh_pull:
        logger.warning(f"fresh pull skipped: the index stops at round {label}")
        return assembled
    overlay: list[dict] = []
    if PLATFORM_METACULUS in config.platforms:
        metaculus, assembled.pulls[PLATFORM_METACULUS] = _metaculus_overlay(
            round_records, label=label, config=config, now=now
        )
        overlay.extend(metaculus)
    if PLATFORM_MANTIC in config.platforms:
        mantic, assembled.pulls[PLATFORM_MANTIC] = _mantic_records(label=label, config=config, now=now)
        overlay.extend(mantic)
    for record in overlay:
        record["record_origin"] = RECORD_ORIGIN_FRESH
        record["rescored_since_round"] = None
    assembled.records.extend(overlay)
    return assembled


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


@dataclass
class Tables:
    forecasts: pd.DataFrame
    members: pd.DataFrame
    texts: pd.DataFrame
    telemetry: pd.DataFrame
    telemetry_index: TelemetryIndex


def _text_row(key: str, record: dict, archive: dict | None, raw_dir: Path) -> dict[str, object]:
    archive = archive or {}
    raw_path = raw_dir / f"{archive.get('run_id')}.jsonl"
    return {
        "forecast_key": key,
        "comment_text": record.get("comment_text"),
        "research_text": archive.get("research_text"),
        "gap_fill_v2_json": json.dumps(archive["gap_fill_v2"]) if archive.get("gap_fill_v2") is not None else None,
        "provider_results_json": json.dumps(archive["provider_results"]) if archive.get("provider_results") else None,
        "provider_diagnostics_block": archive.get("provider_diagnostics_block"),
        "raw_research_path": str(raw_path) if archive.get("run_id") and raw_path.exists() else None,
    }


def _telemetry_rows(key: str, lines: Sequence[dict]) -> list[dict[str, object]]:
    return [
        {
            "forecast_key": key,
            "marker": line["marker"],
            "run_id": str(line["run_id"]),
            "run_date": line.get("run_date"),
            "seq": line.get("seq"),
            "line_ts": line.get("line_ts"),
            "model": line.get("model"),
            "payload_json": json.dumps(line),
        }
        for line in lines
    ]


def _check_unique_keys(records: Sequence[dict]) -> None:
    duplicates = [key for key, n in Counter(forecast_key_of(r) for r in records).items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate forecast_key(s) in the assembled records: {sorted(duplicates)}")


def build_tables(records: list[dict], config: BuildConfig) -> Tables:
    """Join every record to its telemetry run and research record and build the four tables."""
    _check_unique_keys(records)
    id_maps = {
        platform: QuestionIdMap.from_perf_records(r for r in records if record_platform(r) == platform)
        for platform in (PLATFORM_METACULUS, PLATFORM_MANTIC)
    }
    telemetry = build_telemetry_index(config.telemetry_dir, id_maps)
    forecasts, members, texts, telemetry_rows = [], [], [], []
    for record in records:
        key, platform = forecast_key_of(record), record_platform(record)
        run = telemetry.run_for(key)
        archive = load_verified_archive_record(record, config.research_latest_dir)
        rows = member_rows(
            record, key=key, platform=platform, lines=run.lines, type_log_score=record_type_log_score(record)
        )
        inputs = ForecastInputs(record, key, platform, run, archive, rows, telemetry.first_run_dates)
        forecasts.append(forecast_row(inputs))
        members.extend(rows)
        texts.append(_text_row(key, record, archive, config.research_raw_dir))
        telemetry_rows.extend(_telemetry_rows(key, [*telemetry.all_lines_for(key), *run.run_lines]))
    return Tables(
        forecasts=frame_from_rows(forecasts, FORECAST_COLUMNS),
        members=frame_from_rows(members, MEMBER_COLUMNS),
        texts=frame_from_rows(texts, TEXT_COLUMNS).sort_values("forecast_key", ignore_index=True),
        telemetry=frame_from_rows(telemetry_rows, TELEMETRY_COLUMNS),
        telemetry_index=telemetry,
    )


# ---------------------------------------------------------------------------
# Manifest and writing
# ---------------------------------------------------------------------------


def trait_prevalence(members: pd.DataFrame, forecasts: pd.DataFrame) -> list[dict[str, object]]:
    """Share of member rationales showing each trait (a flag set, or a count above zero), overall
    and by config era, since prompt changes move these shares."""
    with_rationale = members[members["has_rationale"].fillna(value=False) & (members["role"] == "member")]
    eras = with_rationale["forecast_key"].map(forecasts.set_index("forecast_key")["config_era"]).fillna("unknown")
    rows: list[dict[str, object]] = []
    for trait in TRAITS:
        shown = (
            with_rationale[trait.name].astype(bool) if trait.kind == TRAIT_KIND_FLAG else with_rationale[trait.name] > 0
        )
        for era, era_mask in [
            (ALL_ERAS, pd.Series(data=True, index=with_rationale.index)),
            *((e, eras == e) for e in sorted(eras.unique())),
        ]:
            n = int(era_mask.sum())
            rows.append(
                {
                    "trait": trait.name,
                    "config_era": era,
                    "n_rationales": n,
                    "share": float(shown[era_mask].mean()) if n else None,
                }
            )
    return rows


def _exclusions(pulls: dict[str, dict[str, dict]]) -> dict[str, int]:
    totals: Counter[str] = Counter(dict.fromkeys(EXCLUSION_KINDS, 0))
    for metas in pulls.values():
        for meta in metas.values():
            totals.update(meta["excluded"])
    return dict(totals)


def build_manifest(
    tables: Tables,
    assembled: AssembledRecords,
    *,
    round_path: Path,
    label: str,
    config: BuildConfig,
    built_at: datetime,
) -> dict:
    forecasts = tables.forecasts
    origin_counts = Counter((forecasts["platform"] + ":" + forecasts["record_origin"]).tolist())
    newest = tables.telemetry_index.newest_run_date
    return {
        "schema_version": SCHEMA_VERSION,
        "built_at": built_at.isoformat(),
        "round_path": str(round_path),
        "round_label": label,
        "round_records": assembled.round_count,
        "platforms": list(config.platforms),
        "fresh_pull_skipped": config.skip_fresh_pull,
        "pulls": assembled.pulls,
        "telemetry_newest_run_date": newest.isoformat() if newest else None,
        "row_counts": {
            "forecasts": len(forecasts),
            "members": len(tables.members),
            "texts": len(tables.texts),
            "telemetry": len(tables.telemetry),
        },
        "rows_by_platform_origin": dict(sorted(origin_counts.items())),
        "excluded": _exclusions(assembled.pulls),
        "no_comment": int((~forecasts["has_comment"]).sum()),
        "no_telemetry": int(forecasts["telemetry_run_id"].isna().sum()),
        "no_research": int(forecasts["research_source_class"].isna().sum()),
        "trait_prevalence": trait_prevalence(tables.members, forecasts),
    }


def write_bundle(tables: Tables, manifest: dict, output_dir: Path) -> None:
    """Write all five files into the staging directory, then move each into place."""
    staging = output_dir / BUILDING_DIR_NAME
    staging.mkdir(parents=True, exist_ok=True)
    tables.forecasts.to_parquet(staging / FORECASTS_FILE, compression=PARQUET_COMPRESSION, index=False)
    tables.members.to_parquet(staging / MEMBERS_FILE, compression=PARQUET_COMPRESSION, index=False)
    tables.texts.to_parquet(
        staging / TEXTS_FILE, compression=PARQUET_COMPRESSION, index=False, row_group_size=TEXTS_ROW_GROUP_SIZE
    )
    tables.telemetry.to_parquet(staging / TELEMETRY_FILE, compression=PARQUET_COMPRESSION, index=False)
    (staging / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    for name in (FORECASTS_FILE, MEMBERS_FILE, TEXTS_FILE, TELEMETRY_FILE, MANIFEST_FILE):
        os.replace(staging / name, output_dir / name)
    staging.rmdir()


def _log_report(manifest: dict) -> None:
    logger.info(f"TRACE_INDEX rows: {manifest['row_counts']}")
    logger.info(f"TRACE_INDEX rows by platform and origin: {manifest['rows_by_platform_origin']}")
    logger.info(f"TRACE_INDEX excluded from the pulled tournaments: {manifest['excluded']}")
    logger.info(
        f"TRACE_INDEX no comment: {manifest['no_comment']}, no telemetry: {manifest['no_telemetry']}, "
        f"no research record: {manifest['no_research']}; newest telemetry run {manifest['telemetry_newest_run_date']}"
    )
    for row in manifest["trait_prevalence"]:
        if row["config_era"] == ALL_ERAS:
            share = "not recorded" if row["share"] is None else f"{row['share']:.1%}"
            logger.info(f"TRACE_INDEX trait {row['trait']}: {share} of {row['n_rationales']} rationales")


def build_index(config: BuildConfig) -> BuildResult:
    """Build the bundle under ``config.output_dir`` and return its manifest."""
    built_at = datetime.now(UTC)
    round_path = config.round_path or newest_round_dataset(config.scratch_dir)
    label = round_label_of(round_path)
    logger.info(f"building the trace index on round {label} ({round_path})")
    round_records = collector.load_dataset(str(round_path))
    assembled = assemble_records(round_records, label=label, config=config, now=built_at)
    tables = build_tables(assembled.records, config)
    manifest = build_manifest(tables, assembled, round_path=round_path, label=label, config=config, built_at=built_at)
    # Reported before the swap, so a report that fails leaves the previous bundle in place.
    _log_report(manifest)
    write_bundle(tables, manifest, config.output_dir)
    return BuildResult(output_dir=config.output_dir, manifest=manifest)
