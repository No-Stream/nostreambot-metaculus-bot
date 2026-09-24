"""Download research artifacts from GHA and merge with backfill into a local archive.

Enumerates EVERY research artifact in the repo via GitHub's artifacts REST endpoint,
downloads the research JSONL each bot run uploads (artifact name `research-<run_id>`),
combines them with existing backfill data, and writes a queryable local archive:

  backtests/research_archive/
    latest/<qid>.json      # the AUTHORITATIVE research per question (see PRECEDENCE)
    by_qid/<qid>.jsonl     # all versions per question (best-first)
    manifest.json          # index: {qid: {latest_timestamp, latest_source, versions_count,
                           #              sources, providers}}

PRECEDENCE: WHICH RECORD WINS latest/<qid>.json
-----------------------------------------------
Source class first (`artifact` > `comment_backfill` > `log_backfill`), then
newest-by-parsed-timestamp within a class. Source class HAS to outrank the timestamp: a
question's comment is published minutes AFTER its research runs, so a plain timestamp sort
hands `latest/` to the lossy comment reconstruction on every question that has both — which
is what it did, on 255 of the 256 dual-sourced questions, leaving `latest/` at 25 artifact /
989 backfill instead of 269 / 737 / 8. `log_backfill` sorts LAST for a different reason
(its `qid` is a POST id, so it must not displace a question-keyed record); see
`record_precedence_key` and `record_source`.

WHY THIS MUST RUN REGULARLY
---------------------------
GHA uploads each run's `research_outputs/` artifact with `retention-days: 90` (see every
.github/workflows/*bot*.yaml). After 90 days the artifact is deleted from GitHub FOREVER,
so this local archive is the only durable copy. The puller is manual
(`make sync_research`); schedule it (see scripts/research_sync/) so artifacts are captured
well inside the 90-day window.

COVERAGE STRATEGY
-----------------
Enumeration + download run through the shared ``scripts.gha_artifacts`` core, which
lists every artifact via the AUTHORITATIVE, COMPLETE paginated REST endpoint (no
1000-result cap, unlike `gh run list`). Every bot run's artifact is named
`research-<run_id>` regardless of which workflow produced it — the three prod tournaments
and, since 2026-08-03, the two test workflows — so filtering to the `research-` prefix
captures everything.
Expired artifacts are unrecoverable; the core logs them loudly so the operator knows
exactly what (if anything) was lost.

`--since-days` is an OPTIONAL post-filter on each artifact's `created_at`. The DEFAULT
is no window (pull every live artifact), since the endpoint already returns everything.
"""

import argparse
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

# One page-url id parser for the whole repo: the archive reader validates identity with
# the same regex (QuestionIds.matches_archive_record), so a second copy here could drift
# into accepting records the reader rejects.
from metaculus_bot.performance_analysis.id_mapping import PAGE_URL_ID_PATTERN
from metaculus_bot.research.image_persistence import ingest_record_images

# Enumeration + persistence run through the shared core.
from scripts.gha_artifacts import (
    add_store_arguments,
    persisted_run_dirs,
    select_artifacts,
)
from scripts.telemetry.jsonl import load_jsonl_records

logger = logging.getLogger(__name__)

# Artifacts whose name starts with this prefix are bot research uploads. Every bot
# workflow uploads `research-<run_id>` (the test pair joined the prod three on 2026-08-03),
# so this single prefix captures all of them via the (workflow-agnostic) artifacts API.
RESEARCH_ARTIFACT_PREFIX = "research-"

# The raw research-provider payload log (metaculus_bot.research.raw_log) rides in
# run_logs/ INSIDE the same research-* artifact as research_outputs/. Its records
# (qid/provider/phase/payload) are a different shape from the per-question research
# records and would corrupt the archive's (qid, run_id) dedup, so the main-archive
# glob must skip these files. scripts/download_raw_research.py archives them separately.
RAW_RESEARCH_LOG_PREFIX = "raw_research_"

# Archive layout defaults, shared with scripts/sync_all.py so the single-pass driver
# writes the same locations the standalone script does.
DEFAULT_OUTPUT_DIR = "backtests/research_archive"
DEFAULT_BACKFILL_DIR = "backtests/research_archive/backfill"

# The THREE writers that feed the archive. Recorded on the winning record as `source` and
# in the manifest as `latest_source` so consumers stop re-deriving it from the run_id
# prefix by hand (which is how the third writer went unnoticed).
#
#   artifact         live in-run capture, uploaded as a GHA artifact. qid = QUESTION id.
#   comment_backfill scripts/backfill_research_from_comments.py, run_id `comment-<id>`.
#                    qid = QUESTION id; carries the post id as `on_post`.
#   log_backfill     scripts/backfill_research_from_logs.py, which parses a run LOG.
#                    qid = POST id (it is parsed out of the page URL), and the run_id is a
#                    plain GHA run id, so this writer is indistinguishable from `artifact`
#                    on run_id alone.
SOURCE_ARTIFACT = "artifact"
SOURCE_COMMENT_BACKFILL = "comment_backfill"
SOURCE_LOG_BACKFILL = "log_backfill"
COMMENT_BACKFILL_RUN_ID_PREFIX = "comment-"
LOG_BACKFILL_RUN_MODE = "backfill_from_logs"


def research_jsonl_files(run_dir: Path) -> list[Path]:
    """List the per-question research JSONL under ``run_dir``, excluding raw-research logs."""
    return [p for p in run_dir.glob("**/*.jsonl") if not p.name.startswith(RAW_RESEARCH_LOG_PREFIX)]


def load_backfill(backfill_dir: Path) -> list[dict]:
    """Load all JSONL records from the backfill directory."""
    if not backfill_dir.exists():
        logger.info(f"Backfill directory does not exist: {backfill_dir}")
        return []

    records = []
    for jsonl_file in sorted(backfill_dir.glob("*.jsonl")):
        file_records = load_jsonl_records(jsonl_file)
        records.extend(file_records)
        logger.debug(f"Loaded {len(file_records)} records from {jsonl_file.name}")

    logger.info(f"Loaded {len(records)} total backfill records from {backfill_dir}")
    return records


def deduplicate_records(records: list[dict]) -> list[dict]:
    """Deduplicate by (qid, run_id), keeping the ``record_precedence_key`` winner.

    Same ordering as the merge stage, deliberately: dedup used to compare timestamps as RAW
    STRINGS, which cannot agree with the parsed-datetime, source-class-first ordering
    ``record_precedence_key`` applies downstream. A (qid, run_id) collision between an
    artifact and a log-backfill record — the same run id is what makes them collide — was
    then settled here by lexicographic timestamp, so the artifact could be DISCARDED before
    precedence ever saw it, whichever order the two arrived in. Two stages ranking the same
    records two ways is the disagreement, not either ranking on its own.
    """
    by_key: dict[tuple[int, str], dict] = {}
    for record in records:
        qid = record.get("qid")
        run_id = record.get("run_id", "")
        if qid is None:
            continue
        key = (qid, str(run_id))
        existing = by_key.get(key)
        if existing is None or record_precedence_key(record) > record_precedence_key(existing):
            by_key[key] = record
    return list(by_key.values())


def download_research_artifacts(
    repo: str,
    since_days: int,
    *,
    store_dir: Path | str | None = None,
    from_store: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> list[dict]:
    """Persist every LIVE research artifact and return its JSONL records from the store.

    Delegates enumeration + persistence to the shared core (``select_artifacts`` +
    ``persisted_run_dirs``), then reads the per-question research JSONL from each
    PERSISTED run dir (excluding the raw-research logs that ride alongside). Referenced
    image sidecars are hash-verified and copied under ``output_dir/media/``. Logs how many
    artifacts were read, records added, and how many were EXPIRED/lost.

    `since_days <= 0` (the default) disables the window and pulls every live artifact.
    ``from_store=True`` reads only what is already persisted and makes no network call.
    """
    selection = select_artifacts(
        repo,
        family_prefixes=(RESEARCH_ARTIFACT_PREFIX,),
        since_days=since_days,
        family_label="research",
        store_dir=store_dir,
        from_store=from_store,
    )
    logger.info(f"Harvesting {len(selection.by_run)} research artifact(s)...")

    all_records: list[dict] = []
    downloaded = 0
    records_added = 0
    for _run_id, _art, run_dir in persisted_run_dirs(
        selection, repo, store_dir=store_dir, from_store=from_store, progress_noun="research artifacts"
    ):
        jsonl_files = research_jsonl_files(run_dir)
        if jsonl_files:
            downloaded += 1
            for jsonl_file in jsonl_files:
                new_records = load_jsonl_records(jsonl_file)
                for record in new_records:
                    ingest_record_images(record, jsonl_file.parent, Path(output_dir))
                records_added += len(new_records)
                all_records.extend(new_records)

    logger.info(
        f"Download phase complete: {downloaded}/{len(selection.by_run)} artifacts downloaded, "
        f"{records_added} records added ({len(selection.expired)} expired/lost)"
    )
    return all_records


def record_source(record: dict) -> str:
    """Classify a record by WRITER, not by content.

    An artifact record is the authoritative capture — the exact research text the
    forecasters saw, plus (on schema-v2 records) ``provider_results`` / ``gap_fill_v2`` /
    ``asknews_raw``. A comment-backfill record is a lossy reconstruction from published
    text (middle-trimmed past ``COMMENT_CHAR_LIMIT``, sections re-headed one level
    deeper, no ``resolution_criteria``) and exists only for questions predating the
    artifact upload step.

    The log-backfill writer needs TWO rungs because it deliberately mimics the
    live-capture shape: it stamps a plain GHA ``run_id`` and even ``run_mode="tournament"``.
    Newer records carry an honest ``run_mode``; the 11 already on disk are recognized by
    the two fields that writer leaves EMPTY and live capture never does (``tournament_id``
    is always the tournament slug there, and ``question_text`` is always populated).
    Telling it apart matters because its ``qid`` is a POST id, so it must not be mistaken
    for a question-keyed record — see ``record_precedence_key``.
    """
    run_id = str(record.get("run_id", ""))
    if run_id.startswith(COMMENT_BACKFILL_RUN_ID_PREFIX):
        return SOURCE_COMMENT_BACKFILL
    if record.get("run_mode") == LOG_BACKFILL_RUN_MODE:
        return SOURCE_LOG_BACKFILL
    if record.get("tournament_id") == "" and record.get("question_text") == "":
        return SOURCE_LOG_BACKFILL
    return SOURCE_ARTIFACT


def _parse_timestamp(value: object) -> datetime:
    """Parse an archive timestamp to an aware UTC datetime; unparseable sorts oldest.

    Records mix a bare ``...Z`` form (the 11 log-backfilled 2026-05 records) with
    ``...+00:00`` plus fractional seconds (the other 269), and comparing those two AS
    STRINGS is not order-preserving: at the same second they first differ at the
    fraction/suffix, where ``Z`` (0x5A) outranks ``.`` (0x2E), so the bare — earlier —
    timestamp sorts higher. A non-UTC offset would invert it by hours (``Z`` > ``-``).
    Nothing currently mis-orders because the format switch happens to align with the era
    boundary, which makes string sorting a trap rather than an active bug.
    """
    try:
        parsed = datetime.fromisoformat(str(value))  # 3.11+ parses the Z suffix natively
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=UTC)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


# Descending precedence for latest/<qid>.json. Artifact-always-wins over comment backfill
# is deliberate and has no "but the backfill is richer" escape hatch: the 58 backfill
# records measurably LONGER than their artifact are longer only because the comment carries
# the ## Provider Diagnostics block that the artifact deliberately withholds from
# research_text (forecasters must never see it) and stores in its own field. Preferring
# them would replay diagnostics into backtest prompts.
#
# log_backfill ranks LAST despite holding untrimmed text, because its qid is a POST id
# while every other writer's qid is a QUESTION id. `latest/<qid>.json` is read question-id
# first (QuestionIds.lookup_order), so letting a post-keyed record outrank a question-keyed
# one in the same group serves a DIFFERENT question's research: measured on this archive,
# ranking it with the artifacts made latest/43592 return question 43591's research and
# latest/43602 return question 43599's. Last place still lets it win when it is the only
# record for that key, which is how the post-id lookup rung finds it.
_SOURCE_RANK = {SOURCE_ARTIFACT: 2, SOURCE_COMMENT_BACKFILL: 1, SOURCE_LOG_BACKFILL: 0}


def record_precedence_key(record: dict) -> tuple[int, datetime]:
    """Sort key for choosing latest/<qid>.json: by source class, then newest within it."""
    return (_SOURCE_RANK[record_source(record)], _parse_timestamp(record.get("timestamp")))


def _post_id_from_own_page_url(record: dict) -> int | None:
    """The post id a record's OWN ``page_url`` embeds, or None.

    Only meaningful for the two writers whose ``page_url`` is the POST url — live capture
    and log backfill. A comment-backfill record's ``page_url`` is the QUESTION url (990 of
    990 on this archive), so reading a post id out of it would write the question id into
    ``post_id``; those records carry ``on_post`` already and never need this.
    """
    if record_source(record) == SOURCE_COMMENT_BACKFILL:
        return None
    match = PAGE_URL_ID_PATTERN.search(str(record.get("page_url", "")))
    return int(match.group(1)) if match is not None else None


def _merge_latest(qid_records: list[dict]) -> dict:
    """The precedence winner, stamped with its source and healed of a missing post_id.

    Copies rather than mutating so ``by_qid/`` keeps the writers' verbatim records —
    ``source`` is a ``latest/``-and-manifest field, never part of the stored history.

    ``post_id`` was added to the live-capture schema mid-era, so 229 of 280 artifact
    records predate it; without healing, promoting an artifact over a backfill would drop
    the only explicit post-id evidence on 255 questions and leave
    ``QuestionIds.matches_archive_record`` leaning on the page_url id alone.

    The heal reads the winner's OWN post url, never a sibling's ``on_post``, and that
    distinction is load-bearing. A ``by_qid`` group can mix id spaces — log-backfill keys
    on the post id, everyone else on the question id — so "sibling in the same group" is
    not evidence of "same question". Borrowing a sibling's ``on_post`` there writes the one
    post id that makes ``matches_archive_record`` accept the record, disarming the very
    check meant to catch a foreign record (measured: it forged a passing id on qids 43592,
    43602 and 43605). Self-healing cannot: verified on all 253 dual-source groups, an
    artifact's own page_url id equals its sibling's ``on_post``, so the heal is equally
    informative where the records agree and correctly REJECTABLE where they don't.
    """
    winner = dict(qid_records[0])
    winner["source"] = record_source(winner)
    if winner.get("post_id") is None:
        recovered = _post_id_from_own_page_url(winner)
        if recovered is not None:
            winner["post_id"] = recovered
    return winner


def load_existing_by_qid(output_dir: Path) -> list[dict]:
    """Load every record already in by_qid/ so a rebuild never DROPS artifact data.

    Artifact records exist ONLY in ``by_qid/`` — nothing writes them back to
    ``backfill/`` — and ``build_archive`` overwrites each ``by_qid/<qid>.jsonl``
    wholesale, so a rebuild that loads backfill alone silently deletes them. Measured on
    the live archive before this existed: a ``--skip-download`` rebuild took ``by_qid/``
    from 280 artifact records to 27, unrecoverably without a re-download. Dedup by
    (qid, run_id) makes re-ingesting these idempotent against a fresh download.
    """
    by_qid_dir = output_dir / "by_qid"
    if not by_qid_dir.exists():
        return []
    records: list[dict] = []
    for jsonl_file in sorted(by_qid_dir.glob("*.jsonl")):
        records.extend(load_jsonl_records(jsonl_file))
    logger.info(f"Loaded {len(records)} existing record(s) from {by_qid_dir}")
    return records


# The classes a rebuild must never shrink, because neither can be re-derived on demand:
# an artifact record is re-downloadable only until its GHA artifact hits 90-day retention,
# and a log_backfill record not even that — the 11 on disk were parsed from run logs whose
# artifacts expired long ago, so this archive is their only copy. comment_backfill is
# deliberately excluded: those are re-derivable from the Metaculus comments API at will, so
# losing one is a re-run, not a data-loss event.
_PROTECTED_SOURCES = frozenset({SOURCE_ARTIFACT, SOURCE_LOG_BACKFILL})


def count_protected_records(records: list[dict]) -> int:
    """How many of ``records`` belong to a class that cannot be re-fetched (see _PROTECTED_SOURCES)."""
    return sum(1 for record in records if record_source(record) in _PROTECTED_SOURCES)


def guard_against_truncation(output_dir: Path, rebuilt: list[dict]) -> None:
    """Refuse to rebuild when the new record set holds fewer un-refetchable records than disk.

    A TRIPWIRE ON THE RE-INGEST, not live protection against a short download — with
    ``load_existing_by_qid`` in both build paths, a partial download can no longer shrink
    anything, so on the happy path this always passes. What it catches is the code
    regression that would make it dangerous again: a caller that stops re-ingesting, or a
    ``by_qid/`` read that silently comes back empty. That is worth a guard because
    ``build_archive`` rewrites the whole archive from whatever list it is handed and these
    records have no other home — one bad rebuild is unrecoverable.

    Both sides run the SAME classifier, so the comparison stays like-for-like: reading one
    side as artifacts-only and the other as artifacts-plus-log-backfill would make every
    rebuild raise on the 11 log-backfill records.

    Hence also the independent disk read rather than the caller's ``load_existing_by_qid``
    result: comparing a list against its own superset could never catch the thing this
    exists to catch.
    """
    existing = count_protected_records(load_existing_by_qid(output_dir))
    new = count_protected_records(rebuilt)
    if new < existing:
        raise RuntimeError(
            f"Refusing to rebuild: would drop un-refetchable records {existing} -> {new} "
            f"(classes {sorted(_PROTECTED_SOURCES)}). A partial/failed download must not "
            "truncate the archive; re-run the sync."
        )
    logger.info(f"Truncation guard: un-refetchable records {existing} on disk -> {new} in the rebuild")


def build_archive(records: list[dict], output_dir: Path) -> None:
    """Write the merged archive: latest/, by_qid/, manifest.json.

    ``latest/<qid>.json`` holds the ``record_precedence_key`` winner (artifact over
    comment backfill, newest within a class) stamped with its ``source``; ``by_qid/``
    holds every version verbatim, best-first.
    """
    for record in records:
        # Rebuild-only runs still verify media preserved by an earlier artifact harvest.
        ingest_record_images(record, output_dir, output_dir)

    latest_dir = output_dir / "latest"
    by_qid_dir = output_dir / "by_qid"
    latest_dir.mkdir(parents=True, exist_ok=True)
    by_qid_dir.mkdir(parents=True, exist_ok=True)

    # Group by qid
    by_qid: dict[int, list[dict]] = {}
    for record in records:
        qid = record.get("qid")
        if qid is None:
            continue
        by_qid.setdefault(qid, []).append(record)

    # Sort each group best-first: artifact class outranks comment backfill, then newest.
    for qid in by_qid:
        by_qid[qid].sort(key=record_precedence_key, reverse=True)

    manifest: dict[str, dict] = {}

    for qid, qid_records in sorted(by_qid.items()):
        # Write latest/<qid>.json
        latest_record = _merge_latest(qid_records)
        latest_path = latest_dir / f"{qid}.json"
        with open(latest_path, "w") as f:
            json.dump(latest_record, f, indent=2)

        # Write by_qid/<qid>.jsonl (best-first, records verbatim as their writers left them)
        by_qid_path = by_qid_dir / f"{qid}.jsonl"
        with open(by_qid_path, "w") as f:
            for record in qid_records:
                f.write(json.dumps(record) + "\n")

        # Collect all providers seen across all versions
        all_providers: set[str] = set()
        for record in qid_records:
            all_providers.update(record.get("providers_used", []))

        manifest[str(qid)] = {
            "latest_timestamp": latest_record.get("timestamp", ""),
            "latest_source": latest_record["source"],
            "versions_count": len(qid_records),
            "sources": sorted({record_source(r) for r in qid_records}),
            "providers": sorted(all_providers),
        }

    # Write manifest.json
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    source_counts = Counter(entry["latest_source"] for entry in manifest.values())
    logger.info(
        f"Archive built: {len(by_qid)} questions, {len(records)} total records (latest/ sources: {dict(source_counts)})"
    )


def main():
    parser = argparse.ArgumentParser(description="Download research artifacts from GHA and merge with backfill.")
    parser.add_argument(
        "--since-days",
        type=int,
        default=0,
        help=(
            "Optional post-filter: only download artifacts created within this many days. "
            "Default 0 = no window (pull EVERY live artifact). The artifacts endpoint already "
            "returns everything inside the 90-day retention window with no cap, so a window is "
            "rarely needed — use it only to scope a targeted re-pull."
        ),
    )
    parser.add_argument("--repo", default="No-Stream/metaculus-bot", help="GitHub repo")
    parser.add_argument(
        "--backfill-dir",
        default=DEFAULT_BACKFILL_DIR,
        help="Where backfill JSONL lives",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the merged archive",
    )
    parser.add_argument(
        "--rebuild-only",
        "--skip-download",
        action="store_true",
        dest="rebuild_only",
        help=(
            "Skip the artifact harvest entirely and rebuild the archive from local ARCHIVE data "
            "only (the records already in by_qid/ plus the backfill dir). Artifact records are "
            "NOT discarded. To re-harvest the artifacts themselves off local disk, use "
            "--from-store instead: it re-reads the persisted JSONL and so recovers records a "
            "past ingest bug dropped, which a rebuild cannot. `--skip-download` is a deprecated "
            "alias for this flag."
        ),
    )
    add_store_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    output_dir = Path(args.output_dir)
    backfill_dir = Path(args.backfill_dir)

    all_records: list[dict] = []

    # --- Phase 1: Harvest artifacts (from GHA, or from the store with --from-store) ---
    if args.rebuild_only:
        # The two offline flags are NOT the same thing and asking for both is almost
        # certainly a mistake: --rebuild-only skips the artifact harvest entirely, so
        # pairing it with --from-store silently drops the store read the operator asked for.
        if args.from_store:
            logger.warning(
                "--rebuild-only overrides --from-store: the artifact harvest is skipped, so nothing is read "
                "from the store. Drop --rebuild-only to re-harvest the persisted artifacts."
            )
    else:
        all_records.extend(
            download_research_artifacts(
                repo=args.repo,
                since_days=args.since_days,
                store_dir=args.store_dir,
                from_store=args.from_store,
                output_dir=output_dir,
            )
        )

    # --- Phase 2: Load what the archive already holds, then the backfill ---
    # Existing records come first so a rebuild (or a download that returned less than
    # last time) re-ingests the on-disk artifacts instead of letting build_archive
    # overwrite by_qid/ with a backfill-only set.
    all_records.extend(load_existing_by_qid(output_dir))
    all_records.extend(load_backfill(backfill_dir))

    # --- Phase 3: Deduplicate and build archive ---
    if not all_records:
        logger.warning("No records found (no artifacts downloaded and no backfill data). Nothing to build.")
        return

    deduplicated = deduplicate_records(all_records)
    logger.info(f"After deduplication: {len(deduplicated)} unique records (from {len(all_records)} total)")

    guard_against_truncation(output_dir, deduplicated)
    build_archive(deduplicated, output_dir)
    logger.info(f"Archive written to {output_dir}")


if __name__ == "__main__":
    main()
