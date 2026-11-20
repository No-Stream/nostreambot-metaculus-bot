"""Bidirectional Metaculus post-id <-> question-id mapping for residual analysis.

Metaculus posts CONTAIN questions. A post has an ``id`` (the POST id, used in the
public URL ``/questions/<post_id>``) and its question has a SEPARATE ``id`` (the
QUESTION id). On older single-question posts the two coincide
(``post_id == question_id``) — which is why most questions show no divergence — but
on newer posts they differ (e.g. post 38880 wraps question 38195).

The bot's run-log telemetry markers are keyed INCONSISTENTLY across marker types:

* ``EXTRACTION_RUNG`` / ``OPEN_BOUND_PILING`` / ``CLOSE_MARGIN`` log
  ``question.id_of_question`` — the QUESTION id (38195 in the example above).
* ``GAP_FILL_V2`` / ``GHOST_FORECAST`` / ``GHOST_FORECAST_JSON`` log
  ``question.page_url`` — a POST id (38880).

A residual grep that filters on ONE id therefore silently drops the records keyed
on the OTHER. And because both ids live in the SAME integer namespace and collide
across questions (one question's post id can equal a *different* question's
question id), a blind "match either id" is unsafe — it admits cross-question false
matches. The two safe primitives are:

1. an EXPLICIT mapping, built at pull time from the performance dataset — the only
   source that carries BOTH ids per question side by side (the collector emits
   ``post_id`` and ``question_id`` on every record); and
2. SELF-DESCRIBING telemetry records: each marker record carries ``qid_kind``
   (``"post_id"`` | ``"question_id"``) naming which space its ``qid`` lives in
   (emitted by :func:`scripts.telemetry.markers._build_record`).

With both, a lookup TRANSLATES the query into the record's own id space instead of
guessing. This module is that mapping plus the telemetry-filter helper that
respects ``qid_kind``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from metaculus_bot.constants import PLATFORM_METACULUS

# The two id spaces, named. These are the canonical ``qid_kind`` values; the
# telemetry marker parser imports them so the emitted tag and the reader stay in
# lockstep (a typo on either side would silently re-open the split-brain).
QID_KIND_POST_ID = "post_id"
QID_KIND_QUESTION_ID = "question_id"

# The id embedded in a question-platform page URL (a POST id, except comment-backfill
# records which build the URL from the question id): Metaculus's question and community
# URL shapes, and Mantic's (competitions.mantic.com, a Metaculus-platform fork; the rest
# of mantic.com is not a question platform). Duplicated from
# ``scripts.backfill_research_from_logs.QID_PATTERN`` because scripts stay free of
# package imports (same convention as the telemetry markers' local qid_kind
# literals); ``tests/test_id_mapping.py`` pins the two patterns equal.
PAGE_URL_ID_PATTERN = re.compile(r"(?:metaculus\.com/(?:questions|c/[^/]+)|competitions\.mantic\.com/questions)/(\d+)")


def record_platform(record: dict) -> str:
    """The platform a performance or research-archive record belongs to.

    Both record kinds gained a ``platform`` field only when Mantic arrived; every record written
    before then is Metaculus (no older archive record has a Mantic page URL), so an absent or
    empty field reads as Metaculus.
    """
    return record.get("platform") or PLATFORM_METACULUS


def _coerce_int(value: object) -> int | None:
    """Coerce a perf-record / marker id to ``int``, or ``None`` if absent/unparseable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("+-").isdigit():
            return int(text)
        return None
    if isinstance(value, float):
        return int(value)
    return None


@dataclass(frozen=True)
class QuestionIds:
    """The (post_id, question_id) pair for one Metaculus question.

    Either id may be ``None`` (a record that recorded only one of them). Built
    from a performance-analysis record via :meth:`from_perf_record`, since that is
    the one place both ids are known authoritatively (straight from the API); or
    from a live ``MetaculusQuestion`` via :meth:`from_question`.
    """

    post_id: int | None
    question_id: int | None

    @classmethod
    def from_perf_record(cls, record: dict) -> QuestionIds:
        """Extract the id pair from a collector performance-dataset record."""
        return cls(
            post_id=_coerce_int(record.get("post_id")),
            question_id=_coerce_int(record.get("question_id")),
        )

    @classmethod
    def from_question(cls, question: object) -> QuestionIds:
        """Extract the id pair from a ``MetaculusQuestion`` (``id_of_post`` / ``id_of_question``)."""
        return cls(
            post_id=_coerce_int(getattr(question, "id_of_post", None)),
            question_id=_coerce_int(getattr(question, "id_of_question", None)),
        )

    def lookup_order(self) -> list[int]:
        """The ids to try when looking a question up in a store, question-id first.

        Question-id is preferred because the dominant research-archive writers (live
        persistence + comment backfill) key on it; post-id is the fallback that also
        finds log-backfill records (keyed on the post id parsed from the URL) and
        post-id-keyed telemetry. Order-preserving and de-duplicated, so a
        non-divergent question (``post_id == question_id``) yields a single id.
        """
        order: list[int] = []
        for i in (self.question_id, self.post_id):
            if i is not None and i not in order:
                order.append(i)
        return order

    def all_ids(self) -> set[int]:
        """The non-``None`` ids for this question (one or two integers)."""
        return {i for i in (self.post_id, self.question_id) if i is not None}

    def id_for_kind(self, kind: str | None) -> int | None:
        """This question's id in the named space (``"post_id"``/``"question_id"``)."""
        if kind == QID_KIND_POST_ID:
            return self.post_id
        if kind == QID_KIND_QUESTION_ID:
            return self.question_id
        return None

    def matches_archive_record(self, record: dict) -> bool:
        """Whether a research-archive record can belong to this question.

        Archive filenames carry no ``qid_kind`` — live persistence and comment
        backfill key ``latest/<id>.json`` on the QUESTION id while the log
        backfill keys on the POST id — so a filename hit alone is the blind
        "match either id" this module's docstring calls unsafe: question A's
        post id can equal a different question B's question id, and A's
        post-id-named file may then hold B's record. This validator rejects a
        record whose self-declared identity contradicts ``self``:

        * an explicit post-id field — ``post_id`` on schema-v2 live records,
          ``on_post`` on comment-backfill records — that differs from
          ``self.post_id``; a matching one is authoritative and accepts
          outright (a post id uniquely identifies the post); or
        * a ``page_url`` whose embedded id is not one of this question's ids.
          The URL id is ambiguous across writers (log backfill embeds the POST
          id, comment backfill the QUESTION id), hence the either-id check.

        Records with neither field usable pass, preserving legacy behavior for
        log-backfill records, whose only identity is the page URL.
        """
        record_post_id = _coerce_int(record.get("post_id"))
        if record_post_id is None:
            record_post_id = _coerce_int(record.get("on_post"))
        if record_post_id is not None and self.post_id is not None:
            return record_post_id == self.post_id
        page_url = record.get("page_url")
        if isinstance(page_url, str):
            match = PAGE_URL_ID_PATTERN.search(page_url)
            if match is not None and int(match.group(1)) not in self.all_ids():
                return False
        return True


class QuestionIdMap:
    """Bidirectional post-id <-> question-id index built from the perf dataset.

    Resolves a marker's ``(qid, qid_kind)`` back to the full :class:`QuestionIds`
    pair, so telemetry keyed in either space can be joined to the perf record (and
    thence to the OTHER id space). The perf dataset is the only authoritative
    both-id source, so that is what :meth:`from_perf_records` consumes.
    """

    def __init__(self) -> None:
        self._by_post: dict[int, QuestionIds] = {}
        self._by_question: dict[int, QuestionIds] = {}

    @classmethod
    def from_perf_records(cls, records: Iterable[dict]) -> QuestionIdMap:
        mapping = cls()
        for record in records:
            mapping.add(QuestionIds.from_perf_record(record))
        return mapping

    def add(self, ids: QuestionIds) -> None:
        if ids.post_id is not None:
            self._by_post[ids.post_id] = ids
        if ids.question_id is not None:
            self._by_question[ids.question_id] = ids

    def resolve(self, qid: int, kind: str | None = None) -> QuestionIds | None:
        """Resolve one id to its (post_id, question_id) pair.

        ``kind`` names the id space and gives a precise, collision-proof lookup.
        When ``kind`` is ``None`` (a legacy record with no ``qid_kind``), we try the
        question-id index first, then the post-id index — best-effort, and
        ambiguous only across the rare cross-question integer collision.
        """
        if kind == QID_KIND_POST_ID:
            return self._by_post.get(qid)
        if kind == QID_KIND_QUESTION_ID:
            return self._by_question.get(qid)
        return self._by_question.get(qid) or self._by_post.get(qid)

    def resolve_marker_record(self, record: dict) -> QuestionIds | None:
        """Resolve one telemetry marker record (``qid`` + ``qid_kind``) to its pair."""
        qid = _coerce_int(record.get("qid"))
        if qid is None:
            return None
        return self.resolve(qid, record.get("qid_kind"))


def marker_records_for_question(records: Iterable[dict], ids: QuestionIds) -> list[dict]:
    """Every telemetry marker record belonging to one question, across BOTH id spaces.

    Each record's ``qid_kind`` selects which id of ``ids`` its ``qid`` must equal,
    so an ``EXTRACTION_RUNG`` (question-id space) and a ``GAP_FILL_V2`` (post-id
    space) for the SAME divergent question are both collected — without the
    cross-question false matches a blind ``qid in {post_id, question_id}`` would
    admit.

    Legacy records with no ``qid_kind`` (archived before self-describing markers)
    fall back to matching against either id. That is best-effort, and safe in
    practice: re-syncing the telemetry archive from the immutable run logs
    re-parses every record and restores ``qid_kind``.
    """
    wanted = ids.all_ids()
    out: list[dict] = []
    for record in records:
        qid = _coerce_int(record.get("qid"))
        if qid is None:
            continue
        kind = record.get("qid_kind")
        if kind is None:
            if qid in wanted:
                out.append(record)
            continue
        if qid == ids.id_for_kind(kind):
            out.append(record)
    return out
