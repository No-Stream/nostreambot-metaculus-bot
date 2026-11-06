"""Guards on the persisted artifact store (``scripts/gha_artifacts``).

The store exists because GHA artifacts are a 90-day staging area we cannot extend, and
because the original download path extracted every artifact into ONE
``tempfile.TemporaryDirectory`` that the generator wiped on close — so an ingest bug
downstream destroyed the payload instead of leaving it re-parseable. Each class here
locks one property that failure mode taught us to need:

1. a harvested run dir SURVIVES the iteration (the temp-dir defect);
2. the on-disk layout matches the 859 dirs already in the live store, since the layout
   is the contract between the bulk grab and every later sync;
3. a re-grab of an immutable artifact is a no-op, never a re-download or a duplicate;
4. a failed download cannot damage a copy already on disk;
5. the offline harvest touches no network at all, which is what makes re-parsing after
   an ingest fix free.
"""

from __future__ import annotations

import json
import subprocess
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

from scripts import gha_artifacts
from scripts.gha_artifacts import (
    STORE_META_FILENAME,
    ArtifactSelection,
    ensure_store_current,
    is_persisted,
    iter_store_run_dirs,
    persist_artifact,
    persisted_run_dirs,
    read_store_meta,
    select_store_artifacts,
    store_artifacts,
    store_run_dir,
)

REPO = "No-Stream/nostreambot-metaculus-bot"


@pytest.mark.parametrize("repetition", range(2))
def test_default_store_is_lazy_and_isolated_per_test(repetition: int, tmp_path_factory: pytest.TempPathFactory) -> None:
    store = Path(gha_artifacts.DEFAULT_STORE_DIR)
    assert store.is_relative_to(tmp_path_factory.getbasetemp())
    assert not store.exists()

    with mock.patch("scripts.gha_artifacts._download_artifact_to", side_effect=_fake_gh_download(str(repetition))):
        run_dir = persist_artifact(_artifact("research-100", 100, datetime.now(UTC)), REPO)

    assert run_dir is not None
    assert run_dir.is_relative_to(store)
    assert is_persisted(store, "research-100")


def _artifact(name: str, run_id: int, created: datetime) -> dict:
    """An artifact object shaped like ``list_research_artifacts``'s --jq projection."""
    return {
        "id": run_id * 10,
        "name": name,
        "created_at": created.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": (created + timedelta(days=90)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expired": False,
        "size_in_bytes": 1234,
        "run_id": run_id,
    }


def _selection(*artifacts: dict) -> ArtifactSelection:
    return ArtifactSelection(
        by_run={int(a["run_id"]): a for a in artifacts}, expired=[], total_artifacts=len(artifacts)
    )


def _fake_gh_download(payload: str = "research payload"):
    """A stand-in for ``gh run download``: writes ``dest_dir/<run_id>/research_<run_id>.jsonl``."""

    def download(run_id, repo, artifact_name, dest_dir):
        run_dir = Path(dest_dir) / str(run_id)
        (run_dir / "research_outputs").mkdir(parents=True, exist_ok=True)
        (run_dir / "research_outputs" / f"research_{run_id}.jsonl").write_text(
            json.dumps({"qid": run_id, "run_id": str(run_id), "text": payload}) + "\n"
        )
        return run_dir

    return download


def _persist_one(art: dict, store_dir: Path, *, payload: str = "research payload") -> Path | None:
    with mock.patch("scripts.gha_artifacts._download_artifact_to", side_effect=_fake_gh_download(payload)):
        return persist_artifact(art, REPO, store_dir)


class TestHarvestedRunDirsOutliveTheHarvest:
    """The defect that motivated the store: the payload must still be on disk afterwards.

    The old ``download_run_dirs`` yielded dirs inside a ``TemporaryDirectory``, so the
    docstring had to warn that consumers harvest INSIDE the iteration — and any parse bug
    lost the artifact permanently rather than leaving it re-readable.
    """

    def test_run_dir_still_readable_after_the_generator_is_exhausted(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        selection = _selection(_artifact("research-100", 100, datetime.now(UTC)))

        with mock.patch("scripts.gha_artifacts._download_artifact_to", side_effect=_fake_gh_download()):
            harvested = list(persisted_run_dirs(selection, REPO, store_dir=store))

        (_run_id, _art, run_dir) = harvested[0]
        assert run_dir.exists(), "the persisted run dir must survive the harvest"
        assert list(run_dir.glob("**/*.jsonl")), "and so must its payload"

    def test_the_dir_is_the_store_path_not_a_temp_copy(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        selection = _selection(_artifact("research-100", 100, datetime.now(UTC)))

        with mock.patch("scripts.gha_artifacts._download_artifact_to", side_effect=_fake_gh_download()):
            (_run_id, _art, run_dir) = next(iter(persisted_run_dirs(selection, REPO, store_dir=store)))

        assert run_dir == store_run_dir(store, "research-100")

    def test_no_staging_dirs_are_left_behind(self, tmp_path: Path) -> None:
        """Extraction happens in a ``.staging-*`` sibling that must not outlive the grab."""
        store = tmp_path / "store"
        _persist_one(_artifact("research-100", 100, datetime.now(UTC)), store)

        assert [p.name for p in store.iterdir()] == ["research-100"]


class TestStoreLayoutMatchesTheLiveStore:
    """``_meta.json``'s four fields, written as the existing 859 dirs write them.

    The bulk grab that populated the live store is gone; its layout is the contract, so a
    drift here (a renamed key, an int where a string was) would make ``store_artifacts``
    silently skip real data.
    """

    def test_meta_carries_the_four_fields(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        created = datetime(2026, 5, 29, 13, 31, 49, tzinfo=UTC)
        _persist_one(_artifact("research-26639832588", 26639832588, created), store)

        meta = read_store_meta(store_run_dir(store, "research-26639832588"))
        assert meta == {
            "artifact_id": str(26639832588 * 10),
            "name": "research-26639832588",
            "created_at": "2026-05-29T13:31:49Z",
            "run_id": "26639832588",
        }

    def test_meta_ids_are_strings_as_the_live_store_writes_them(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        _persist_one(_artifact("research-100", 100, datetime.now(UTC)), store)

        meta = read_store_meta(store_run_dir(store, "research-100"))
        assert meta is not None
        assert isinstance(meta["run_id"], str)
        assert isinstance(meta["artifact_id"], str)

    def test_a_dir_with_meta_reads_back_as_an_artifact_object(self, tmp_path: Path) -> None:
        """``store_artifacts`` is the offline stand-in for the artifacts endpoint."""
        store = tmp_path / "store"
        _persist_one(_artifact("research-100", 100, datetime(2026, 7, 1, tzinfo=UTC)), store)

        assert store_artifacts(store) == [
            {
                "id": "1000",
                "name": "research-100",
                "created_at": "2026-07-01T00:00:00Z",
                "expired": False,
                "run_id": 100,
            }
        ]

    def test_store_keys_on_artifact_name_so_one_run_can_hold_two_artifacts(self, tmp_path: Path) -> None:
        """A pre-rename test run uploaded both ``research-<id>`` and ``logs-<id>``."""
        store = tmp_path / "store"
        now = datetime.now(UTC)
        _persist_one(_artifact("research-100", 100, now), store)
        _persist_one(_artifact("logs-100", 100, now), store)

        assert sorted(p.name for p in store.iterdir()) == ["logs-100", "research-100"]


class TestSkipIfPresentIdempotency:
    """An uploaded artifact is immutable, so a re-grab must be a no-op — never a re-download."""

    def test_second_grab_downloads_nothing(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        selection = _selection(_artifact("research-100", 100, datetime.now(UTC)))

        with mock.patch("scripts.gha_artifacts._download_artifact_to", side_effect=_fake_gh_download()) as dl:
            first = ensure_store_current(selection, REPO, store_dir=store)
            second = ensure_store_current(selection, REPO, store_dir=store)

        assert (first.downloaded, first.already_present) == (1, 0)
        assert (second.downloaded, second.already_present) == (0, 1)
        assert dl.call_count == 1, "the artifact is immutable; the second pass must not refetch it"

    def test_re_grab_leaves_the_payload_byte_identical(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        art = _artifact("research-100", 100, datetime.now(UTC))
        _persist_one(art, store)
        before = (store_run_dir(store, "research-100") / "research_outputs" / "research_100.jsonl").read_text()

        # A second grab whose download would write DIFFERENT bytes must not run at all.
        with mock.patch("scripts.gha_artifacts._download_artifact_to", side_effect=_fake_gh_download("REPLACED")):
            ensure_store_current(_selection(art), REPO, store_dir=store)

        after = (store_run_dir(store, "research-100") / "research_outputs" / "research_100.jsonl").read_text()
        assert after == before

    def test_no_duplicate_dirs_accumulate(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        art = _artifact("research-100", 100, datetime.now(UTC))
        for _ in range(3):
            _persist_one(art, store)

        assert [p.name for p in store.iterdir()] == ["research-100"]

    def test_an_incomplete_dir_is_refetched(self, tmp_path: Path) -> None:
        """A dir without ``_meta.json`` is a grab interrupted mid-extraction, not a copy."""
        store = tmp_path / "store"
        interrupted = store_run_dir(store, "research-100")
        interrupted.mkdir(parents=True)
        (interrupted / "half_written.jsonl").write_text("{")
        assert not is_persisted(store, "research-100")

        with mock.patch("scripts.gha_artifacts._download_artifact_to", side_effect=_fake_gh_download()) as dl:
            stats = ensure_store_current(
                _selection(_artifact("research-100", 100, datetime.now(UTC))), REPO, store_dir=store
            )

        assert (dl.call_count, stats.downloaded) == (1, 1)
        assert is_persisted(store, "research-100")
        assert not (interrupted / "half_written.jsonl").exists(), "the incomplete copy is replaced, not merged"


class TestConcurrentStoreDownloads:
    def test_missing_artifacts_download_concurrently_and_stats_are_exact(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        now = datetime.now(UTC)
        present = _artifact("research-100", 100, now - timedelta(days=4))
        _persist_one(present, store)
        selection = _selection(
            present,
            _artifact("research-200", 200, now - timedelta(days=3)),
            _artifact("research-300", 300, now - timedelta(days=2)),
            _artifact("research-400", 400, now - timedelta(days=1)),
        )
        barrier = threading.Barrier(3)
        attempted: list[str] = []
        attempted_lock = threading.Lock()

        def persist(art: dict, repo: str, store_dir: Path | str | None = None) -> Path | None:
            name = str(art["name"])
            with attempted_lock:
                attempted.append(name)
            barrier.wait(timeout=5)
            return None if name == "research-300" else store_run_dir(store_dir, name)

        with mock.patch("scripts.gha_artifacts.persist_artifact", side_effect=persist):
            stats = ensure_store_current(selection, REPO, store_dir=store)

        assert stats.already_present == 1
        assert stats.downloaded == 2
        assert stats.failed == 1
        assert set(attempted) == {"research-200", "research-300", "research-400"}


class TestAFailedGrabDamagesNothing:
    """One hung `gh` must not sink the pull, and must not touch what is already on disk."""

    def test_failed_download_leaves_an_existing_copy_intact(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        art = _artifact("research-100", 100, datetime.now(UTC))
        _persist_one(art, store, payload="the good copy")

        # Called directly, bypassing skip-if-present, to prove the write path itself is safe.
        with mock.patch("scripts.gha_artifacts._download_artifact_to", return_value=None):
            assert persist_artifact(art, REPO, store) is None

        payload = (store_run_dir(store, "research-100") / "research_outputs" / "research_100.jsonl").read_text()
        assert "the good copy" in payload
        assert is_persisted(store, "research-100")

    def test_failed_download_persists_nothing_and_writes_no_meta(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        with mock.patch("scripts.gha_artifacts._download_artifact_to", return_value=None):
            assert persist_artifact(_artifact("research-100", 100, datetime.now(UTC)), REPO, store) is None

        assert not store_run_dir(store, "research-100").exists()
        assert not is_persisted(store, "research-100")

    def test_one_failure_does_not_abort_the_rest_of_the_grab(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        now = datetime.now(UTC)
        selection = _selection(
            _artifact("research-100", 100, now - timedelta(days=2)),
            _artifact("research-200", 200, now - timedelta(days=1)),
        )
        succeed = _fake_gh_download()

        def flaky(run_id, repo, artifact_name, dest_dir):
            return None if run_id == 100 else succeed(run_id, repo, artifact_name, dest_dir)

        with mock.patch("scripts.gha_artifacts._download_artifact_to", side_effect=flaky):
            stats = ensure_store_current(selection, REPO, store_dir=store)

        assert (stats.downloaded, stats.failed) == (1, 1)
        assert is_persisted(store, "research-200")

    def test_a_selected_artifact_missing_from_the_store_is_skipped_by_the_harvest(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        now = datetime.now(UTC)
        present = _artifact("research-200", 200, now)
        _persist_one(present, store)

        harvested = list(iter_store_run_dirs(_selection(_artifact("research-100", 100, now), present), store))

        assert [run_id for run_id, _art, _dir in harvested] == [200]

    def test_an_unreadable_meta_is_treated_as_not_persisted(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        _persist_one(_artifact("research-100", 100, datetime.now(UTC)), store)
        (store_run_dir(store, "research-100") / STORE_META_FILENAME).write_text("{ truncated")

        assert not is_persisted(store, "research-100")
        assert store_artifacts(store) == []


class TestOfflineSelectionTouchesNoNetwork:
    """``--from-store`` is the free re-parse path, so nothing in it may call out."""

    @pytest.fixture(autouse=True)
    def _ban_subprocess(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Any `gh` invocation from the shared core fails this class's tests loudly."""

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError(f"offline path invoked a subprocess: {args!r}")

        monkeypatch.setattr(subprocess, "run", forbidden)

    def test_selection_and_harvest_run_entirely_off_disk(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        now = datetime.now(UTC)
        # Pre-populating goes through the same mocked download seam every other test uses,
        # so even this setup proves the store path never shells out.
        _persist_one(_artifact("research-100", 100, now - timedelta(days=1)), store)
        _persist_one(_artifact("research-200", 200, now), store)

        selection = select_store_artifacts(store, family_prefixes=("research-",), since_days=0, family_label="research")
        harvested = list(persisted_run_dirs(selection, REPO, store_dir=store, from_store=True))

        assert [run_id for run_id, _art, _dir in harvested] == [100, 200]

    def test_family_prefixes_filter_the_store(self, tmp_path: Path) -> None:
        """A future artifact family in the store must not be handed to a research harvest."""
        store = tmp_path / "store"
        now = datetime.now(UTC)
        _persist_one(_artifact("research-100", 100, now), store)
        _persist_one(_artifact("benchmark-results-400", 400, now), store)

        selection = select_store_artifacts(store, family_prefixes=("research-",), since_days=0, family_label="research")

        assert set(selection.by_run) == {100}
        assert selection.total_artifacts == 2, "total counts the whole store, the selection only the family"

    def test_since_days_still_windows_an_offline_harvest(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        now = datetime.now(UTC)
        _persist_one(_artifact("research-old", 100, now - timedelta(days=40)), store)
        _persist_one(_artifact("research-new", 200, now - timedelta(days=2)), store)

        selection = select_store_artifacts(store, family_prefixes=("research-",), since_days=7, family_label="research")

        assert set(selection.by_run) == {200}

    def test_harvest_order_is_oldest_upload_first(self, tmp_path: Path) -> None:
        """Deterministic order keeps replace-by-run merges reproducible across runs."""
        store = tmp_path / "store"
        now = datetime.now(UTC)
        _persist_one(_artifact("research-newest", 300, now), store)
        _persist_one(_artifact("research-oldest", 100, now - timedelta(days=5)), store)
        _persist_one(_artifact("research-middle", 200, now - timedelta(days=2)), store)

        selection = select_store_artifacts(store, family_prefixes=("research-",), since_days=0, family_label="research")

        assert [rid for rid, _art, _dir in iter_store_run_dirs(selection, store)] == [100, 200, 300]

    def test_an_absent_store_yields_an_empty_selection_rather_than_raising(self, tmp_path: Path) -> None:
        selection = select_store_artifacts(
            tmp_path / "never_created", family_prefixes=("research-",), since_days=0, family_label="research"
        )

        assert selection.by_run == {}
        assert selection.expired == [], "nothing in a store can be expired — that is the point of persisting it"
