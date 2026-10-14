"""Download GHA artifacts and archive the raw research-provider payload logs.

metaculus_bot.research.raw_log appends each provider's RAW return (AskNews article
dicts per phase, native/Gemini raw responses, prediction-market contracts,
resolution-source fetches, gap-fill search results) to ``run_logs/raw_research_<run_id>.jsonl``.
All five bot workflows upload ``run_logs/`` bundled inside ``research-<run_id>``. Like the
telemetry harvest, this still pulls BOTH artifact families, because the two test
workflows uploaded ``logs-<run_id>`` until 2026-08-03 and those artifacts remain on GHA
until their retention expires.

Archive layout under ``backtests/research_archive/raw/`` (gitignored via ``backtests/``):

    <run_id>.jsonl   # one file per bot run, mirroring that run's raw_research log

IDEMPOTENCY — REPLACE-BY-RUN: a run's uploaded log is immutable, so re-harvesting
overwrites ``<run_id>.jsonl`` byte-identically (records deduped within a run on
``(qid, provider, phase)``, keeping the latest ``fetched_at``). Runs absent from a
harvest (expired artifact) are left untouched. Re-running neither dupes nor loses data.

WHY THIS MUST RUN REGULARLY: GHA deletes artifacts after 90 days. This local archive
is the only durable copy of the raw evidence behind each forecast. Read-only + free
(GitHub API only); no paid LLM/research calls, no publishing. Wrapped by
``make sync_raw_research`` and folded into the single-pass ``make sync_all``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

from metaculus_bot.research.source_capture_persistence import ingest_source_capture_records

# Enumeration + download go through the shared core; the run-log family (research-* +
# logs-*) constant lives with the run-log harvester. Raw research rides in the same
# artifacts as the run logs.
from scripts.download_run_logs import RUN_LOG_ARTIFACT_PREFIXES
from scripts.gha_artifacts import add_store_arguments, persisted_run_dirs, select_artifacts
from scripts.telemetry.jsonl import load_jsonl_records

logger = logging.getLogger(__name__)

DEFAULT_REPO = "No-Stream/nostreambot-metaculus-bot"
DEFAULT_ARCHIVE_DIR = "backtests/research_archive/raw"

# raw_research_<run_id>.jsonl — the run_id is the GITHUB_RUN_ID stamped at write time
# (or "local"), which equals the artifact's originating workflow_run id.
RAW_LOG_FILENAME_RE = re.compile(r"^raw_research_(?P<run_id>.+)\.jsonl$")


def harvest_raw_logs_from_dir(run_dir: Path, *, source_archive_dir: Path | None = None) -> dict[str, list[dict]]:
    """Return ``{run_id: [records]}`` from every ``raw_research_*.jsonl`` under ``run_dir``.

    ``run_id`` comes from the filename (the records themselves don't carry it). A dir
    with no raw-research logs yields ``{}`` — expected for an artifact that predates
    the feature or ran with RAW_RESEARCH_LOG disabled.
    """
    out: dict[str, list[dict]] = {}
    for path in sorted(Path(run_dir).glob("**/raw_research_*.jsonl")):
        match = RAW_LOG_FILENAME_RE.match(path.name)
        if match is None:
            continue
        records = load_jsonl_records(path)
        if source_archive_dir is not None:
            ingest_source_capture_records(records, path.parent, source_archive_dir)
        out.setdefault(match.group("run_id"), []).extend(records)
    return out


def _dedup_run_records(records: list[dict]) -> list[dict]:
    """Dedup one run's records on ``(qid, provider, phase)``, keeping the latest fetched_at."""
    by_key: dict[tuple, dict] = {}
    for record in records:
        key = (record.get("qid"), record.get("provider"), record.get("phase"))
        existing = by_key.get(key)
        if existing is None or record.get("fetched_at", "") >= existing.get("fetched_at", ""):
            by_key[key] = record
    return sorted(
        by_key.values(),
        key=lambda r: (str(r.get("qid")), str(r.get("provider")), str(r.get("phase")), str(r.get("fetched_at"))),
    )


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    """Atomically write ``records`` to ``path`` as JSONL (temp sibling + ``os.replace``).

    This rewrites a run's ONLY durable raw-research copy in place. An in-place
    ``open(path, "w")`` truncates the file on open, so a mid-write crash would leave it
    truncated; instead we stream into a sibling temp file and atomically rename over the
    target, keeping the existing file intact until the full new content is on disk. The
    temp file shares ``path``'s directory so the rename is same-filesystem.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def merge_and_write(archive_dir: Path, harvested: dict[str, list[dict]]) -> dict[str, int]:
    """Write ``<archive_dir>/<run_id>.jsonl`` for each harvested run (replace-by-run).

    Returns ``{run_id: record_count}`` after dedup. Runs not in ``harvested`` are left
    on disk untouched.
    """
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    totals: dict[str, int] = {}
    for run_id, records in harvested.items():
        deduped = _dedup_run_records(records)
        _write_jsonl_atomic(archive_dir / f"{run_id}.jsonl", deduped)
        totals[run_id] = len(deduped)
    return totals


def download_and_harvest(
    repo: str,
    since_days: int,
    archive_dir: Path,
    *,
    store_dir: Path | str | None = None,
    from_store: bool = False,
) -> tuple[dict[str, int], int]:
    """Persist every live run-log artifact, harvest raw logs from the store, merge into the archive.

    Enumeration + persistence go through the shared core; this function contributes only
    the raw-research-specific harvest (``harvest_raw_logs_from_dir``) and merge. With
    ``from_store=True`` nothing is downloaded. Returns ``(per_run_totals, expired_count)``.
    """
    selection = select_artifacts(
        repo,
        family_prefixes=RUN_LOG_ARTIFACT_PREFIXES,
        since_days=since_days,
        family_label="run-log",
        store_dir=store_dir,
        from_store=from_store,
    )

    harvested: dict[str, list[dict]] = {}
    for _run_id, _art, run_dir in persisted_run_dirs(
        selection, repo, store_dir=store_dir, from_store=from_store, progress_noun="run-log artifacts"
    ):
        for harvested_run_id, records in harvest_raw_logs_from_dir(run_dir, source_archive_dir=archive_dir).items():
            harvested.setdefault(harvested_run_id, []).extend(records)

    totals = merge_and_write(archive_dir, harvested)
    return totals, len(selection.expired)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GHA artifacts and archive raw research-provider logs.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo (owner/name)")
    parser.add_argument(
        "--since-days",
        type=int,
        default=0,
        help="Optional post-filter: only artifacts created within N days (0 = every live artifact).",
    )
    parser.add_argument("--archive-dir", default=DEFAULT_ARCHIVE_DIR, help="Where to write the raw-research archive")
    add_store_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    totals, expired_count = download_and_harvest(
        repo=args.repo,
        since_days=args.since_days,
        archive_dir=Path(args.archive_dir),
        store_dir=args.store_dir,
        from_store=args.from_store,
    )
    total_records = sum(totals.values())
    logger.info("=" * 60)
    logger.info(
        f"Raw-research harvest complete: {len(totals)} run(s) with raw logs, "
        f"{total_records} records archived, {expired_count} expired/lost"
    )
    logger.info(f"Archive written to {args.archive_dir}")


if __name__ == "__main__":
    sys.exit(main())
