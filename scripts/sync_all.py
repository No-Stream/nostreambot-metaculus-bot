"""Single-pass driver for the three archive syncs (``make sync_all``).

The three standalone syncs each independently (1) enumerate EVERY repo artifact and
(2) download overlapping families into their own temp dir: ``research-*`` gets pulled
THREE times (research + telemetry + raw), ``logs-*`` twice, and the enumeration runs
3×. With ~100 live artifacts that's ~300 subprocess+unzip invocations.

This driver enumerates ONCE over the UNION family (``research-*`` + ``logs-*``),
downloads each unique artifact ONCE into the PERSISTED artifact store
(``backtests/gha_artifact_store/``, see ``scripts/gha_artifacts``), and runs all three
harvests over the same persisted run dirs before writing the three archives:

* research JSONL           -> backtests/research_archive/     (research-* dirs only)
* run-log telemetry markers -> backtests/telemetry_archive/
* raw research-provider logs -> backtests/research_archive/raw/

Then the three archive builds run exactly as their standalone counterparts do
(research: download records + backfill -> dedup -> build; telemetry + raw:
replace-by-run merge). Read-only + free (GitHub API only), no publishing.

Backfill from Metaculus comments (``backfill_research_from_comments.py``) is NOT run
here — it hits Metaculus, not GHA. The Makefile ``sync_all`` target runs it first so
its ``comments_backfill.jsonl`` is on disk when this driver's research build loads it.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from metaculus_bot.research.image_persistence import ingest_record_images
from scripts.download_raw_research import DEFAULT_ARCHIVE_DIR as RAW_ARCHIVE_DIR
from scripts.download_raw_research import harvest_raw_logs_from_dir
from scripts.download_raw_research import merge_and_write as merge_raw_research
from scripts.download_research import (
    DEFAULT_BACKFILL_DIR,
    DEFAULT_OUTPUT_DIR,
    RESEARCH_ARTIFACT_PREFIX,
    build_archive,
    deduplicate_records,
    guard_against_truncation,
    load_backfill,
    load_existing_by_qid,
    load_jsonl_records,
    research_jsonl_files,
)
from scripts.download_run_logs import DEFAULT_ARCHIVE_DIR as TELEMETRY_ARCHIVE_DIR
from scripts.download_run_logs import (
    RUN_LOG_ARTIFACT_PREFIXES,
    harvest_run_logs_from_dir,
    infer_workflow,
    resolve_workflow_map,
)
from scripts.download_run_logs import merge_and_write as merge_telemetry
from scripts.gha_artifacts import add_store_arguments, persisted_run_dirs, select_artifacts
from scripts.telemetry.archive import HarvestedRun

logger = logging.getLogger(__name__)

DEFAULT_REPO = "No-Stream/metaculus-bot"


@dataclass
class SyncSummary:
    """Rolled-up results of one single-pass sync, for the final report + tests.

    ``harvested`` counts run dirs READ from the store, which is not the number downloaded:
    the grab is skip-if-present, so a re-run harvests everything and downloads nothing (and
    ``--from-store`` downloads nothing by construction). ``ensure_store_current`` logs the
    download side.
    """

    total_artifacts: int
    harvested: int
    expired: list[dict]
    research_questions: int
    research_records: int
    telemetry_totals: dict[str, int]
    telemetry_runs: int
    raw_totals: dict[str, int] = field(default_factory=dict)


def run_sync(
    repo: str,
    since_days: int,
    *,
    research_dir: Path,
    backfill_dir: Path,
    telemetry_dir: Path,
    raw_dir: Path,
    store_dir: Path | str | None = None,
    from_store: bool = False,
) -> SyncSummary:
    """Persist the union family ONCE, then harvest + build all three archives from the store.

    ``from_store=True`` makes the whole sync offline: the selection comes from the
    persisted store's ``_meta.json``s instead of the artifacts endpoint and nothing is
    downloaded, so an ingest fix can be re-run over the same bytes for free.
    """
    selection = select_artifacts(
        repo,
        family_prefixes=RUN_LOG_ARTIFACT_PREFIXES,
        since_days=since_days,
        family_label="run-log",
        store_dir=store_dir,
        from_store=from_store,
    )

    workflow_map = resolve_workflow_map(repo, telemetry_dir, from_store=from_store)

    research_records: list[dict] = []
    telemetry_runs: list[HarvestedRun] = []
    raw_harvested: dict[str, list[dict]] = {}
    harvested_dirs = 0

    for run_id, art, run_dir in persisted_run_dirs(
        selection, repo, store_dir=store_dir, from_store=from_store, progress_noun="artifacts"
    ):
        harvested_dirs += 1
        name = art.get("name", "")

        # (a) Research JSONL — research-* dirs only. Every bot workflow (three prod plus
        # test_bot and test_bot_basic) uploads under that name as of 2026-08-03, when the
        # test pair moved off logs-* so their research gets archived too; the surviving
        # pre-rename logs-* artifacts carry only run_logs/, so they contribute telemetry
        # below but no research records.
        if name.startswith(RESEARCH_ARTIFACT_PREFIX):
            for jsonl_file in research_jsonl_files(run_dir):
                new_records = load_jsonl_records(jsonl_file)
                for record in new_records:
                    ingest_record_images(record, jsonl_file.parent, research_dir)
                research_records.extend(new_records)

        # (b) Run-log telemetry markers.
        harvested = harvest_run_logs_from_dir(
            run_dir,
            run_id=str(run_id),
            workflow=infer_workflow(name, run_id, workflow_map),
            artifact=name,
            run_date=art.get("created_at", ""),
        )
        if harvested is not None:
            telemetry_runs.append(harvested)

        # (c) Raw research-provider payload logs.
        for harvested_run_id, records in harvest_raw_logs_from_dir(run_dir).items():
            raw_harvested.setdefault(harvested_run_id, []).extend(records)

    research_questions, research_count = _build_research_archive(research_records, backfill_dir, research_dir)
    telemetry_totals = merge_telemetry(telemetry_dir, telemetry_runs)
    raw_totals = merge_raw_research(raw_dir, raw_harvested)

    return SyncSummary(
        total_artifacts=selection.total_artifacts,
        harvested=harvested_dirs,
        expired=selection.expired,
        research_questions=research_questions,
        research_records=research_count,
        telemetry_totals=telemetry_totals,
        telemetry_runs=len(telemetry_runs),
        raw_totals=raw_totals,
    )


def _build_research_archive(downloaded_records: list[dict], backfill_dir: Path, research_dir: Path) -> tuple[int, int]:
    """Merge downloaded research records + what's on disk + backfill, dedup, and build.

    Mirrors ``download_research.main``'s Phase 2/3 exactly (re-ingest the existing
    ``by_qid/`` records, load backfill, dedup by (qid, run_id), guard against
    truncation, build) so both the standalone script and this driver end up with an
    archive holding BOTH artifact and comment-backfill records. Re-ingesting the
    existing records is what keeps a download that came back short of last time from
    silently deleting artifacts: ``build_archive`` overwrites ``by_qid/`` wholesale and
    artifact records live nowhere else. Returns ``(distinct_questions, unique_records)``;
    leaves the archive untouched when there is nothing to build.
    """
    all_records = list(downloaded_records)
    all_records.extend(load_existing_by_qid(research_dir))
    all_records.extend(load_backfill(backfill_dir))
    if not all_records:
        logger.warning("Research: no records (no artifacts downloaded and no backfill). Archive not rebuilt.")
        return 0, 0

    deduped = deduplicate_records(all_records)
    logger.info(f"Research: {len(deduped)} unique records (from {len(all_records)} total)")
    guard_against_truncation(research_dir, deduped)
    build_archive(deduped, research_dir)
    questions = len({r["qid"] for r in deduped if r.get("qid") is not None})
    return questions, len(deduped)


def _expired_by_family(expired: list[dict]) -> tuple[int, int]:
    """Split the expired-artifact count into (research-*, logs-*) for the summary."""
    research = sum(1 for a in expired if str(a.get("name", "")).startswith(RESEARCH_ARTIFACT_PREFIX))
    return research, len(expired) - research


def _report(summary: SyncSummary, *, research_dir: Path, telemetry_dir: Path, raw_dir: Path) -> None:
    expired_research, expired_logs = _expired_by_family(summary.expired)
    logger.info("=" * 60)
    logger.info("sync_all single-pass harvest complete")
    logger.info(
        f"Enumerated {summary.total_artifacts} artifact(s); harvested {summary.harvested} run dir(s) from the "
        f"store (the grab itself reports its own downloaded/skipped counts); "
        f"{len(summary.expired)} expired/lost ({expired_research} research-*, {expired_logs} logs-*)"
    )
    logger.info(
        f"Research archive : {summary.research_questions} question(s), {summary.research_records} record(s) "
        f"-> {research_dir}"
    )
    logger.info(f"Telemetry archive: {summary.telemetry_runs} run(s) with logs -> {telemetry_dir}")
    for marker, count in sorted(summary.telemetry_totals.items()):
        logger.info(f"  {marker:32s} {count}")
    raw_records = sum(summary.raw_totals.values())
    logger.info(f"Raw-research     : {len(summary.raw_totals)} run(s), {raw_records} record(s) -> {raw_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-pass GHA sync: research + telemetry + raw-research archives in one download pass."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo (owner/name)")
    parser.add_argument(
        "--since-days",
        type=int,
        default=0,
        help="Optional post-filter: only artifacts created within N days (0 = every live artifact).",
    )
    parser.add_argument("--research-dir", default=DEFAULT_OUTPUT_DIR, help="Where to write the research archive")
    parser.add_argument("--backfill-dir", default=DEFAULT_BACKFILL_DIR, help="Where the research backfill JSONL lives")
    parser.add_argument("--telemetry-dir", default=TELEMETRY_ARCHIVE_DIR, help="Where to write the telemetry archive")
    parser.add_argument("--raw-dir", default=RAW_ARCHIVE_DIR, help="Where to write the raw-research archive")
    add_store_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    research_dir = Path(args.research_dir)
    telemetry_dir = Path(args.telemetry_dir)
    raw_dir = Path(args.raw_dir)
    summary = run_sync(
        repo=args.repo,
        since_days=args.since_days,
        research_dir=research_dir,
        backfill_dir=Path(args.backfill_dir),
        telemetry_dir=telemetry_dir,
        raw_dir=raw_dir,
        store_dir=args.store_dir,
        from_store=args.from_store,
    )
    _report(summary, research_dir=research_dir, telemetry_dir=telemetry_dir, raw_dir=raw_dir)


if __name__ == "__main__":
    main()
