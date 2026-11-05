"""Shared GHA-artifact enumeration + download core for the archive-sync scripts.

``download_research`` / ``download_run_logs`` / ``download_raw_research`` all do the
same six things: (1) enumerate EVERY repo artifact via the paginated REST endpoint,
(2) filter to an artifact family by name prefix, (3) report expired (unrecoverable)
artifacts loudly, (4) window by ``--since-days``, (5) dedup by originating
workflow-run id, and (6) make each unique artifact's contents available on local disk.
This module is the single implementation of that skeleton.

Two consumers use it:

* the three standalone scripts, each over its own family (research-* for the
  research archive; research-* + logs-* for the run-log + raw-research archives);
* ``scripts/sync_all.py``, which enumerates + downloads ONCE over the UNION family
  and runs all three harvests over the shared run dirs — so a ``make sync_all``
  does one download per artifact instead of three (one per standalone script).

THE PERSISTED ARTIFACT STORE
----------------------------
Downloads land in ``backtests/gha_artifact_store/<artifact-name>/`` and STAY THERE.
This is the fix for the original design, where every artifact was extracted into one
``tempfile.TemporaryDirectory`` that the generator wiped on close: harvesting had to
happen inside the iteration, and any ingest bug downstream destroyed the payload
rather than leaving it re-parseable on disk. GHA deletes artifacts at 90 days
(``maximum_allowed_days: 90`` for this repo — the ceiling cannot be raised), so the
staging area is transient by construction and local disk is the source of truth from
the moment an artifact is grabbed.

Each store dir holds the artifact's extracted contents plus a ``_meta.json``
(``artifact_id`` / ``name`` / ``created_at`` / ``run_id``) so an artifact's age is
knowable without another API call.

IDEMPOTENCY — SKIP-IF-PRESENT. An uploaded artifact is immutable, so a re-grab of one
already in the store is a no-op: ``ensure_store_current`` re-downloads only what is
absent or INCOMPLETE (a dir with no ``_meta.json``, i.e. a previous run interrupted
mid-extraction). Deleting a store dir is therefore the way to force a re-fetch.

The download path is READ-ONLY and FREE — it hits only the GitHub API, no LLM /
research calls, no publishing. ``from_store=True`` skips the API entirely and harvests
what is already on disk, so re-parsing after an ingest fix costs nothing and needs no
network.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Where downloaded artifacts are persisted, one dir per artifact NAME (not per run: a
# single run can upload more than one artifact, e.g. the pre-2026-08-03 test workflows'
# research-* plus logs-*, and name-keying keeps them from colliding). Under
# ``backtests/``, which .gitignore excludes — the store is local-disk state, never
# committed. 864 dirs / 44 MB as of 2026-08-03 (measured ~18 MB/month, ~220 MB/year),
# so nothing here needs compression or pruning; permanence is the entire point.
DEFAULT_STORE_DIR = "backtests/gha_artifact_store"

# Written INSIDE each store dir. Its presence is the "extraction finished" marker that
# makes skip-if-present safe, so it is written before the dir is moved into place.
STORE_META_FILENAME = "_meta.json"

# subprocess.run timeouts (seconds). Bounding both keeps one slow/hung `gh` call from
# stalling a scheduled pull: a per-artifact download that times out is skipped (the
# other artifacts still process), and the artifacts enumeration can't block forever.
GH_API_TIMEOUT_S = 120
ARTIFACT_DOWNLOAD_TIMEOUT_S = 180
ARTIFACT_DOWNLOAD_WORKERS = 6


def verify_gh_cli() -> None:
    """Ensure gh CLI is installed and authenticated."""
    try:
        # `gh` is resolved off PATH on purpose: FileNotFoundError IS the "not installed"
        # branch below, which a shutil.which() preflight would only restate.
        subprocess.run(["gh", "--version"], capture_output=True, check=True)  # noqa: S607
    except FileNotFoundError:
        logger.error("gh CLI not found. Install from https://cli.github.com/")
        sys.exit(1)
    except subprocess.CalledProcessError:
        logger.error("gh CLI returned an error. Check authentication with 'gh auth status'.")
        sys.exit(1)

    # Same PATH lookup as above; fixed argv, no shell, nothing caller-supplied.
    result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, check=False)  # noqa: S607
    if result.returncode != 0:
        logger.error(f"gh CLI not authenticated: {result.stderr.strip()}")
        sys.exit(1)


def _parse_created_at(value: str) -> datetime | None:
    """Parse a GH `created_at` ISO-8601 timestamp (e.g. '2026-05-22T14:00:00Z')."""
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def list_research_artifacts(repo: str) -> list[dict]:
    """List EVERY artifact in the repo via the paginated artifacts REST endpoint.

    `gh api --paginate` follows the Link headers fully, so this returns all artifacts
    across all workflows with no 1000-result cap. We page in batches of 100 (the API
    max page size) and emit one JSON object per artifact via `--jq`, then parse the
    newline-delimited stream. Each object carries `id`, `name`, `created_at`,
    `expires_at`, `expired`, `size_in_bytes`, and the originating `workflow_run.id`.

    Filtering to a specific artifact family happens downstream (see ``split_by_family``)
    so callers can also see/report non-family and expired artifacts. Returns the raw
    artifact objects.
    """
    cmd = [
        "gh",
        "api",
        "--paginate",
        f"/repos/{repo}/actions/artifacts?per_page=100",
        "--jq",
        (".artifacts[] | {id, name, created_at, expires_at, expired, size_in_bytes, run_id: .workflow_run.id}"),
    ]
    try:
        # S603: fixed `gh api` argv, no shell. The only interpolation is ``repo``, an
        # operator-supplied ``owner/name`` that lands inside a URL path, not an argv slot.
        result = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=GH_API_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        logger.error(f"gh api artifacts listing timed out ({GH_API_TIMEOUT_S}s) for {repo}")
        sys.exit(1)
    if result.returncode != 0:
        logger.error(f"gh api artifacts listing failed for {repo}: {result.stderr.strip()}")
        sys.exit(1)

    artifacts: list[dict] = []
    for line_num, raw_line in enumerate(result.stdout.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            artifacts.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"Malformed artifact JSON at line {line_num}, skipping")
    return artifacts


def split_by_family(artifacts: list[dict], prefixes: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    """Split artifacts whose name starts with any of ``prefixes`` into ``(live, expired)``."""
    family = [a for a in artifacts if str(a.get("name", "")).startswith(prefixes)]
    live = [a for a in family if not a.get("expired")]
    expired = [a for a in family if a.get("expired")]
    return live, expired


def report_expired(expired: list[dict], family_label: str) -> None:
    """Log every expired artifact by name + created_at (a loud, operator-watched LOST line).

    Expired artifacts are past the 90-day retention and gone from GitHub forever, so this
    is the only signal that data was silently lost. The ``LOST: <name> (created_at=...)``
    format is load-bearing — operators grep for it.
    """
    if expired:
        logger.warning(
            f"{len(expired)} {family_label} artifact(s) are EXPIRED and UNRECOVERABLE (past 90-day retention):"
        )
        for art in sorted(expired, key=lambda a: a.get("created_at", "")):
            logger.warning(f"  LOST: {art.get('name')} (created_at={art.get('created_at')})")
    else:
        logger.info(f"No expired {family_label} artifacts — nothing lost to the 90-day window.")


def _dedup_by_run(live: list[dict]) -> dict[int, dict]:
    """Keep one artifact per originating workflow-run id (first-wins on the pagination stream)."""
    by_run: dict[int, dict] = {}
    for art in live:
        run_id = art.get("run_id")
        if run_id is None:
            logger.warning(f"Live artifact {art.get('name')} has no workflow_run id, skipping")
            continue
        by_run.setdefault(run_id, art)
    return by_run


@dataclass
class ArtifactSelection:
    """Result of ``select_run_artifacts``: the artifacts to download + the expired ones.

    ``by_run`` maps run_id -> artifact (deduped, live, windowed). ``expired`` is the
    family's expired artifacts, retained so the caller can report the count in its
    summary (they're already logged loudly by ``report_expired``).
    """

    by_run: dict[int, dict]
    expired: list[dict]
    total_artifacts: int


def _apply_window(live: list[dict], since_days: int) -> list[dict]:
    """Keep artifacts created within ``since_days``; ``since_days <= 0`` keeps everything."""
    if since_days <= 0:
        return live
    cutoff = datetime.now(UTC) - timedelta(days=since_days)
    before = len(live)
    windowed = [a for a in live if (_parse_created_at(a.get("created_at", "")) or cutoff) >= cutoff]
    logger.info(f"--since-days={since_days} post-filter: {len(windowed)}/{before} live artifacts within window")
    return windowed


def select_run_artifacts(
    repo: str, *, family_prefixes: tuple[str, ...], since_days: int, family_label: str
) -> ArtifactSelection:
    """Enumerate once, filter to the family, report expired, window by ``since_days``, dedup by run_id.

    ``since_days <= 0`` disables the window and selects every live artifact (the endpoint
    already returns everything inside the 90-day retention window with no cap).
    """
    verify_gh_cli()

    all_artifacts = list_research_artifacts(repo)
    live, expired = split_by_family(all_artifacts, family_prefixes)
    logger.info(
        f"Artifacts endpoint returned {len(all_artifacts)} total, "
        f"{len(live)} live + {len(expired)} expired {family_label} artifact(s)"
    )
    report_expired(expired, family_label)

    live = _apply_window(live, since_days)
    return ArtifactSelection(by_run=_dedup_by_run(live), expired=expired, total_artifacts=len(all_artifacts))


def _download_artifact_to(run_id: int, repo: str, artifact_name: str, dest_dir: Path) -> Path | None:
    """Download one artifact into ``dest_dir/<run_id>``; return the dir or None on failure.

    Resolves the artifact by its originating workflow-run id (carried on each artifact
    object as ``workflow_run.id``). A timeout or non-zero exit is logged and returns None
    so the download loop can skip this run and keep going — one hung `gh` never sinks the pull.
    """
    run_dir = dest_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["gh", "run", "download", str(run_id), "--repo", repo, "--name", artifact_name, "--dir", str(run_dir)]
    try:
        # S603: fixed `gh run download` argv, no shell. Every interpolated value comes from
        # the artifact objects GitHub itself returned (or an operator-supplied ``--repo``),
        # and each occupies its own argv slot, so nothing can inject a flag or a command.
        result = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=ARTIFACT_DOWNLOAD_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            f"Timed out ({ARTIFACT_DOWNLOAD_TIMEOUT_S}s) downloading {artifact_name} (run {run_id}); skipping"
        )
        return None
    if result.returncode != 0:
        logger.warning(f"Failed to download {artifact_name} (run {run_id}): {result.stderr.strip()}")
        return None
    return run_dir


def _resolve_store_dir(store_dir: Path | str | None) -> Path:
    """The store path to use, reading ``DEFAULT_STORE_DIR`` at CALL time when unset.

    Resolving late rather than binding the constant into each signature is what lets the
    test suite redirect the store wholesale (the autouse fixture in ``tests/conftest.py``):
    a signature default is captured at import, so a test that merely OMITTED ``store_dir``
    wrote into the operator's real 42 MB store — and one did, until its fixture log turned
    up in the live telemetry archive.
    """
    return Path(DEFAULT_STORE_DIR if store_dir is None else store_dir)


def store_run_dir(store_dir: Path | str | None, artifact_name: str) -> Path:
    """The store path an artifact's extracted contents live at (keyed by artifact NAME)."""
    return _resolve_store_dir(store_dir) / artifact_name


def read_store_meta(run_dir: Path) -> dict | None:
    """Read a store dir's ``_meta.json``, or None when it is absent/unreadable."""
    meta_path = Path(run_dir) / STORE_META_FILENAME
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning(f"Unreadable {STORE_META_FILENAME} in {run_dir}; treating the dir as not persisted")
        return None
    return meta if isinstance(meta, dict) else None


def is_persisted(store_dir: Path | str | None, artifact_name: str) -> bool:
    """Whether this artifact is already fully persisted (dir present AND meta written)."""
    return read_store_meta(store_run_dir(store_dir, artifact_name)) is not None


def _write_store_meta(run_dir: Path, art: dict) -> None:
    """Stamp the artifact's identity + upload time beside its contents.

    Written before the dir is moved into place, so a store dir either has meta and is
    complete or has none and gets re-downloaded. ``artifact_id`` / ``run_id`` are stored
    as strings to match the 859 dirs the first bulk grab wrote.
    """
    meta = {
        "artifact_id": str(art.get("id", "")),
        "name": str(art.get("name", "")),
        "created_at": str(art.get("created_at", "")),
        "run_id": str(art.get("run_id", "")),
    }
    (Path(run_dir) / STORE_META_FILENAME).write_text(json.dumps(meta))


def _swap_into_place(src: Path, dest: Path) -> Path:
    """Move a fully-extracted staging dir onto ``dest``, replacing whatever was there.

    ``dest`` is only ever occupied by an INCOMPLETE copy here: ``ensure_store_current``
    skips artifacts that are already persisted, so the only dir this clears is one a
    previous run left without ``_meta.json``. A download that failed never reaches this
    function at all (``persist_artifact`` returns early), which is what keeps a failed
    extraction from clobbering a good copy.
    """
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(src, dest)
    return dest


def persist_artifact(art: dict, repo: str, store_dir: Path | str | None = None) -> Path | None:
    """Download one artifact into the store, or None if the download failed.

    Extraction happens in a staging sibling and is moved into place only once complete,
    so an interrupted or failed `gh run download` leaves no half-populated store dir.
    """
    name = str(art.get("name", ""))
    run_id = art.get("run_id")
    if not name or run_id is None:
        logger.warning(f"Artifact {art.get('id')} has no name/run_id; cannot persist")
        return None

    store_dir = _resolve_store_dir(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    staging = store_dir / f".staging-{name}-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        downloaded = _download_artifact_to(int(run_id), repo, name, staging)
        if downloaded is None:
            return None
        _write_store_meta(downloaded, art)
        return _swap_into_place(downloaded, store_run_dir(store_dir, name))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@dataclass
class StoreSyncStats:
    """What ``ensure_store_current`` did: how much was already on disk vs. newly grabbed."""

    already_present: int = 0
    downloaded: int = 0
    failed: int = 0


def _ordered_artifacts(selection: ArtifactSelection) -> list[dict]:
    """The selection's artifacts oldest-upload-first (name breaks created_at ties).

    Deterministic order matters twice: the grab walks oldest-first so the artifacts
    closest to expiry are secured first, and the harvest replays in the same order every
    run, which keeps replace-by-run merges reproducible.
    """
    return sorted(selection.by_run.values(), key=lambda a: (str(a.get("created_at", "")), str(a.get("name", ""))))


def ensure_store_current(
    selection: ArtifactSelection, repo: str, *, store_dir: Path | str | None = None, progress_noun: str = "artifacts"
) -> StoreSyncStats:
    """Persist every selected artifact that isn't already in the store.

    Skip-if-present (uploaded artifacts are immutable) makes a re-run cheap: only new
    artifacts and dirs left incomplete by an earlier interruption are fetched. Missing
    artifacts are submitted oldest-first to a bounded worker pool; completion order may
    differ. A per-artifact download that fails is counted and skipped so one hung `gh`
    never aborts the pull. Progress prints every 25 successful downloads.
    """
    store_dir = _resolve_store_dir(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    ordered = _ordered_artifacts(selection)
    stats = StoreSyncStats()

    missing: list[dict] = []
    for art in ordered:
        if is_persisted(store_dir, str(art.get("name", ""))):
            stats.already_present += 1
            continue
        missing.append(art)

    with ThreadPoolExecutor(max_workers=ARTIFACT_DOWNLOAD_WORKERS) as executor:
        futures = [executor.submit(persist_artifact, art, repo, store_dir) for art in missing]
        for future in as_completed(futures):
            if future.result() is None:
                stats.failed += 1
                continue
            stats.downloaded += 1
            if stats.downloaded % 25 == 0:
                print(f"  persisted {stats.downloaded} new {progress_noun}", flush=True)

    logger.info(
        f"Artifact store {store_dir}: {len(ordered)} selected -> {stats.already_present} already persisted, "
        f"{stats.downloaded} newly downloaded, {stats.failed} failed"
    )
    return stats


def iter_store_run_dirs(
    selection: ArtifactSelection, store_dir: Path | str | None = None
) -> Iterator[tuple[int, dict, Path]]:
    """Yield ``(run_id, artifact, run_dir)`` from the STORE for each selected artifact.

    Makes no network call and no temp dir: the yielded ``run_dir`` is the persisted copy,
    so a consumer may re-read it after the iteration and a re-parse needs no re-download.
    A selected artifact missing from the store (its download failed) is skipped, and the
    count is reported so a short grab is visible.
    """
    store_dir = _resolve_store_dir(store_dir)
    missing: list[str] = []
    for art in _ordered_artifacts(selection):
        name = str(art.get("name", ""))
        if not is_persisted(store_dir, name):
            missing.append(name)
            continue
        yield int(art["run_id"]), art, store_run_dir(store_dir, name)
    if missing:
        # The count above is exact; the tail is named for grep-ability, not analysis.
        named = ", ".join(missing[:5])  # HARNESS-SCAN-EXEMPT-subsampling
        logger.warning(
            f"{len(missing)} selected artifact(s) are not in the store at {store_dir} and were NOT harvested "
            f"(first few: {named})"
        )


def store_artifacts(store_dir: Path | str | None = None) -> list[dict]:
    """Every persisted artifact as an artifact object, read from the store's ``_meta.json``s.

    The offline counterpart to ``list_research_artifacts``: same shape, no API call.
    Nothing in the store can be ``expired`` — that is the whole point of persisting it.
    """
    store_dir = _resolve_store_dir(store_dir)
    if not store_dir.exists():
        logger.warning(f"Artifact store {store_dir} does not exist yet; nothing to harvest offline")
        return []

    artifacts: list[dict] = []
    for run_dir in sorted(store_dir.iterdir()):
        # Leading-dot dirs are this module's own staging/scratch names, never artifacts.
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        meta = read_store_meta(run_dir)
        if meta is None:
            logger.warning(
                f"Store dir {run_dir.name} has no readable {STORE_META_FILENAME} (incomplete grab); "
                "skipping — a sync will re-download it"
            )
            continue
        run_id = str(meta.get("run_id", ""))
        if not run_id.isdigit():
            logger.warning(f"Store dir {run_dir.name} has a non-numeric run_id {run_id!r}; skipping")
            continue
        artifacts.append(
            {
                "id": meta.get("artifact_id"),
                "name": str(meta.get("name") or run_dir.name),
                "created_at": str(meta.get("created_at", "")),
                "expired": False,
                "run_id": int(run_id),
            }
        )
    return artifacts


def select_store_artifacts(
    store_dir: Path | str | None = None, *, family_prefixes: tuple[str, ...], since_days: int, family_label: str
) -> ArtifactSelection:
    """Build a selection from the PERSISTED store instead of the GitHub API (no network)."""
    all_artifacts = store_artifacts(store_dir)
    live, _expired = split_by_family(all_artifacts, family_prefixes)
    logger.info(
        f"Artifact store holds {len(all_artifacts)} persisted artifact(s), {len(live)} in the {family_label} family"
    )
    live = _apply_window(live, since_days)
    return ArtifactSelection(by_run=_dedup_by_run(live), expired=[], total_artifacts=len(all_artifacts))


def select_artifacts(
    repo: str,
    *,
    family_prefixes: tuple[str, ...],
    since_days: int,
    family_label: str,
    store_dir: Path | str | None = None,
    from_store: bool = False,
) -> ArtifactSelection:
    """Select artifacts to harvest, from GitHub (default) or from the persisted store."""
    if from_store:
        return select_store_artifacts(
            store_dir, family_prefixes=family_prefixes, since_days=since_days, family_label=family_label
        )
    return select_run_artifacts(repo, family_prefixes=family_prefixes, since_days=since_days, family_label=family_label)


def add_store_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the two store flags every sync script shares (``--store-dir`` / ``--from-store``)."""
    parser.add_argument(
        "--store-dir",
        default=None,
        help=f"Where downloaded artifacts are persisted, one dir per artifact name (default: {DEFAULT_STORE_DIR}).",
    )
    parser.add_argument(
        "--from-store",
        action="store_true",
        help=(
            "OFFLINE re-parse: harvest the artifacts already persisted in --store-dir and make "
            "no network call at all. Use after fixing an ingest bug — the bytes are already on "
            "disk, so re-parsing them is free and needs no re-download."
        ),
    )


def persisted_run_dirs(
    selection: ArtifactSelection,
    repo: str,
    *,
    store_dir: Path | str | None = None,
    from_store: bool = False,
    progress_noun: str = "artifacts",
) -> Iterator[tuple[int, dict, Path]]:
    """Ensure the selection is on disk, then yield each artifact's PERSISTED run dir.

    The download phase ("make the store current") and the harvest phase ("iterate the
    store") are separate on purpose: the harvest reads only local disk, so an ingest bug
    is re-runnable against the same bytes with ``from_store=True``. With
    ``from_store=True`` the download phase is skipped entirely and nothing touches the
    network.
    """
    if not from_store:
        ensure_store_current(selection, repo, store_dir=store_dir, progress_noun=progress_noun)
    return iter_store_run_dirs(selection, store_dir)
