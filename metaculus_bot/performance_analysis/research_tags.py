"""Research-archive treatment tags for performance records.

Reads ``backtests/research_archive/latest/<qid>.json`` and derives per-question treatment
flags (did the forecasters' bundle carry a Time Series Anchor section, an Agentic Research
Findings section from gap-fill v2) so calibration cuts can split treated from untreated
records without re-deriving the tags out-of-band each round. The residual rounds'
``anchor_tags.json`` is the reference output.

Every detection rule is source-class-aware, because the three archive writers head their
sections at different depths and only one of them can carry the ``gap_fill_v2`` payload.
Both gfv2 tags and ``anchor_confidence`` therefore read as TERNARY: None means the record
could not have recorded the fact, which is not the same as a measured False. The rules,
the confidence tokens and the receipt behind each are in ``docs/performance_analysis.md``
"Two treatment tags read as TERNARY".
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from metaculus_bot.performance_analysis.eras import B4E9DF0_MERGED_AT
from metaculus_bot.performance_analysis.id_mapping import QuestionIds, record_platform

logger = logging.getLogger(__name__)

DEFAULT_RESEARCH_ARCHIVE_LATEST = Path("backtests/research_archive/latest")

ANCHOR_SECTION_TITLE = "Time Series Anchor"
GFV2_SECTION_TITLE = "Agentic Research Findings"

# Depth-agnostic, exact-title header matches (see module docstring).
_ANCHOR_HEADER_RE = re.compile(r"^#{1,4}\s*" + re.escape(ANCHOR_SECTION_TITLE) + r"\s*$", re.MULTILINE)
_GFV2_HEADER_RE = re.compile(r"^#{1,4}\s*" + re.escape(GFV2_SECTION_TITLE) + r"\s*$", re.MULTILINE)

# One Provider Diagnostics line, e.g. ``- timeseries_anchor: ok | 855 chars | 1171 ms``.
_DIAG_TS_ANCHOR_RE = re.compile(
    r"^-\s*timeseries_anchor:\s*(?P<status>ok|empty|errored|fallback|skipped|timeout)\s*\|\s*\d+\s*chars",
    re.MULTILINE,
)

_ABSENT_TAGS: dict[str, None] = {
    "anchor_present": None,
    "gfv2_present": None,
    "gfv2_loop_ran": None,
    "gfv2_confidence": None,
    "anchor_confidence": None,
    "research_source_class": None,
}

# The one writer class whose records can carry the ``gap_fill_v2`` payload (module docstring).
_GFV2_PAYLOAD_SCHEMA_VERSION = 2
_GFV2_PAYLOAD_SOURCE = "artifact"
# The payload write reached main in b4e9df0, three weeks after schema v2: an alias, never a copy.
_GFV2_PAYLOAD_ERA_START = B4E9DF0_MERGED_AT


def _writer_can_carry_gfv2_payload(record: dict) -> bool:
    """Whether this record's WRITER could have recorded a v2 loop, had one run.

    An undatable record (no parseable ``timestamp``) reads as can't-carry: the
    conservative side, since the alternative stamps "the only confident untreated
    read" on a record whose writer may physically lack the key.
    """
    schema_version = record.get("schema_version")
    if not isinstance(schema_version, int):
        return False
    if schema_version < _GFV2_PAYLOAD_SCHEMA_VERSION or record.get("source") != _GFV2_PAYLOAD_SOURCE:
        return False
    try:
        written_at = datetime.fromisoformat(str(record.get("timestamp")))
    except ValueError:
        return False
    if written_at.tzinfo is None:
        written_at = written_at.replace(tzinfo=UTC)
    return written_at >= _GFV2_PAYLOAD_ERA_START


def _timeseries_anchor_diag_status(record: dict) -> str | None:
    """The ``timeseries_anchor:`` status from the Provider Diagnostics block, if any.

    Prefers the record's dedicated ``provider_diagnostics_block`` field; older
    records only carry the block inline in ``research_text``, so fall back to
    slicing the text from the block's title onward.
    """
    block = record.get("provider_diagnostics_block") or ""
    if not block:
        text = record.get("research_text") or ""
        idx = text.find("Provider Diagnostics")
        block = text[idx:] if idx >= 0 else ""
    match = _DIAG_TS_ANCHOR_RE.search(block)
    return match.group("status") if match else None


def research_tags_for_record(record: dict) -> dict:
    """Derive the treatment tags from one research-archive record."""
    text = record.get("research_text") or ""
    anchor_present = bool(_ANCHOR_HEADER_RE.search(text))
    diag_status = _timeseries_anchor_diag_status(record)

    if anchor_present:
        confidence = "header"
    elif diag_status in ("ok", "fallback"):
        # The provider produced a section the text no longer carries: trimming ate the header.
        confidence = "diag_ok_header_missing"
    elif diag_status in ("empty", "errored", "skipped", "timeout"):
        confidence = "diag_confirms_absent"
    elif record.get("is_trimmed"):
        confidence = "ambiguous_trimmed_no_diag"
    else:
        confidence = "absent_no_diag"

    gfv2_present = bool(_GFV2_HEADER_RE.search(text))
    gfv2_loop_ran, gfv2_confidence = _gfv2_read(record, gfv2_present)

    return {
        "anchor_present": anchor_present,
        "gfv2_present": gfv2_present,
        "gfv2_loop_ran": gfv2_loop_ran,
        "gfv2_confidence": gfv2_confidence,
        "anchor_confidence": confidence,
        "research_source_class": record.get("source"),
    }


def _gfv2_read(record: dict, gfv2_present: bool) -> tuple[bool | None, str]:
    """``(gfv2_loop_ran, gfv2_confidence)`` for one archive record.

    Either the banked ``gap_fill_v2`` payload or the section header itself proves the loop
    ran, whatever the writer class; None means neither is present and the writer could not
    have carried the payload anyway, which is "unrecorded" rather than "did not run". The
    five confidence tokens are tabulated in ``docs/performance_analysis.md`` "Two treatment
    tags read as TERNARY".
    """
    if gfv2_present:
        return True, "header"
    if record.get("gap_fill_v2") is not None:
        return True, "payload_ran_no_section"
    # Only an ABSENT payload depends on carryability, and a null payload counts as absent.
    if not _writer_can_carry_gfv2_payload(record):
        return None, "ambiguous_trimmed_no_payload" if record.get("is_trimmed") else "absent_no_payload"
    return False, "payload_confirms_absent"


def _load_archive_record(qid: object, latest_dir: Path | str) -> dict | None:
    """Load ``latest/<qid>.json``, or None when absent/unreadable."""
    path = Path(latest_dir) / f"{qid}.json"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Unreadable research-archive record {path}: {exc}")
        return None
    return record if isinstance(record, dict) else None


def research_tags_for_qid(qid: object, latest_dir: Path | str = DEFAULT_RESEARCH_ARCHIVE_LATEST) -> dict:
    """Tags for one question id off ``latest/<qid>.json``; all-None when no record exists.

    A ``log_backfill`` record also reads all-None: that writer class keys on the
    POST id while every other writer keys on the QUESTION id, and the two share one
    integer space — so from a bare qid the record's identity cannot be verified,
    and stamping a possibly-foreign question's tags is worse than no tags. Callers
    holding a full performance record should use :func:`attach_research_tags`,
    which verifies identity via the record's own id pair.
    """
    if qid is None:
        return dict(_ABSENT_TAGS)
    record = _load_archive_record(qid, latest_dir)
    if record is None or record.get("source") == "log_backfill":
        return dict(_ABSENT_TAGS)
    return research_tags_for_record(record)


def load_verified_archive_record(record: dict, latest_dir: Path | str) -> dict | None:
    """The ``latest/`` research record that provably belongs to one performance record, or None.

    Lookup walks the record's own id pair (:meth:`QuestionIds.lookup_order`, question-id
    first) and accepts a file only when :meth:`QuestionIds.matches_archive_record` confirms it
    belongs to this question: ``latest/`` mixes two id spaces (log-backfill records key on the
    POST id, everything else on the QUESTION id, in one integer namespace), so a bare filename
    hit can serve a DIFFERENT question's research (the measured ``latest/43592`` collision). A
    ``log_backfill`` record is never accepted, because its only identity is the page URL, and
    neither is a record from another platform, since Mantic ids share that integer space too.
    """
    ids = QuestionIds.from_perf_record(record)
    platform = record_platform(record)
    for candidate_qid in ids.lookup_order():
        loaded = _load_archive_record(candidate_qid, latest_dir)
        if loaded is None or not ids.matches_archive_record(loaded):
            continue
        if loaded.get("source") == "log_backfill" or record_platform(loaded) != platform:
            continue
        return loaded
    return None


def attach_research_tags(records: list[dict], latest_dir: Path | str = DEFAULT_RESEARCH_ARCHIVE_LATEST) -> None:
    """Stamp the treatment tags onto each performance record in place.

    The archive record is found by :func:`load_verified_archive_record`. A record with no
    verified archive counterpart gets the all-None tags rather than False.
    """
    tagged = 0
    for record in records:
        loaded = load_verified_archive_record(record, latest_dir)
        tags = dict(_ABSENT_TAGS) if loaded is None else research_tags_for_record(loaded)
        record.update(tags)
        if tags["anchor_present"] is not None:
            tagged += 1
    logger.info(f"Research tags attached: {tagged}/{len(records)} records have an archive record")
