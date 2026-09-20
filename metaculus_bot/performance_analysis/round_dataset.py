"""The residual round's dataset builder: dedup, era tags, cohorts, provenance.

Every round pulls resolved questions, merges them with the baselines the previous round already
tagged, and writes one ``perf_all_tagged.json`` that every downstream lane reads. The rules for
doing that are the same round to round and live here; only the directories, the labels and the
tournament slugs change, and callers supply those in a :class:`RoundSpec`.

Field insertion order is load-bearing: ``perf_all_tagged.json`` is compared byte-for-byte between
rounds, so the tagging steps below assign in a fixed order. Invoke this maintained API directly
for routine refreshes; do not create a new scratch driver. The input contract and permitted
follow-up analyses are documented in ``docs/performance_analysis.md`` "The round dataset builder".
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from metaculus_bot.performance_analysis.cohorts import DEGRADED_RUN_QIDS, KNOWN_BUG_QIDS, PARTIAL_DEGRADED_QIDS
from metaculus_bot.performance_analysis.collector import rescore_records
from metaculus_bot.performance_analysis.eras import (
    B4E9DF0_MERGED_AT,
    WIDENING_FLIP_MERGED_AT,
    era_of,
    ft_unfreeze_side_of,
    post_linters_merge_of,
    triple_subera_fine_of,
    triple_subera_of,
)
from metaculus_bot.performance_analysis.id_mapping import QuestionIdMap, QuestionIds
from metaculus_bot.performance_analysis.platform_scores import peer_score, spot_peer_score
from metaculus_bot.performance_analysis.scaling import NUMERIC_TYPES
from metaculus_bot.time_utils import parse_iso_utc

logger: logging.Logger = logging.getLogger(__name__)

RecordKey = tuple[object, object]

# Stripped off a reused record before retagging, so no stale tag survives a rule change.
TAG_FIELDS: tuple[str, ...] = (
    "config_era",
    "triple_subera",
    "triple_subera_fine",
    "ft_unfreeze_side",
    "post_linters_merge",
    "source_tournament",
    "source_is_fresh_pull",
    "record_provenance",
    "degraded_run",
    "partial_degraded",
    "known_bug",
    "is_new_since_prior",
    "newly_scored",
    "rescored_fields",
    "rescored_fields_prior_rounds",
    "rescored_fields_this_round",
    "platform_rescored",
    "platform_rescored_prior_rounds",
    "platform_rescored_this_round",
    "platform_rescored_pull_tag",
    "platform_rescored_fields",
    "question_weight",
)

# Bot-side scores, recomputed from stored inputs every round by collector.rescore_records.
SCORE_FIELDS: tuple[str, ...] = ("brier_score", "log_score", "numeric_log_score", "mc_log_score")
# The subset compared against the prior round: a move here is our scorer's, never the platform's.
BOT_DRIFT_FIELDS: tuple[str, ...] = ("log_score", "mc_log_score", "numeric_log_score")
SCORE_ATOL = 1e-6

# The three ways a score can differ between rounds; a null on one side is a move, not a non-event.
DRIFT_APPEARED = "appeared"
DRIFT_DISAPPEARED = "disappeared"
DRIFT_MOVED = "moved"

# One table, so prior_snapshot hoists exactly the fields the drift path later reads flat.
PLATFORM_DRIFT_ACCESSORS: tuple[tuple[str, Callable[[dict], float | None]], ...] = (
    ("peer_score", peer_score),
    ("spot_peer_score", spot_peer_score),
)

# The cohort constants hold strings ('43746', ...); perf records carry an int question_id.
KNOWN_BUG_QUESTION_IDS: frozenset[int] = frozenset(int(qid) for qid in KNOWN_BUG_QIDS)
DEGRADED_FULL_QUESTION_IDS: frozenset[int] = frozenset(int(qid) for qid in DEGRADED_RUN_QIDS)
DEGRADED_PARTIAL_QUESTION_IDS: frozenset[int] = frozenset(int(qid) for qid in PARTIAL_DEGRADED_QIDS)

# Per-question markers a dropping run leaves: one question-keyed, one post-keyed, ids overlapping.
DEGRADED_JOIN_MARKERS: tuple[str, ...] = ("extraction_rung", "gap_fill_v2")


@dataclass(frozen=True)
class RoundSpec:
    """One round's state: where its files live, what to load, how to label provenance."""

    round_dir: Path
    label: str
    prior_dir: Path
    prior_label: str
    telemetry_dir: Path
    weighted_slug: str
    required_slugs: tuple[str, ...]
    optional_slugs: tuple[str, ...]
    reused_slugs: tuple[str, ...]

    @property
    def fresh_provenance(self) -> str:
        return f"fresh_{self.label}"

    @property
    def reused_provenance(self) -> str:
        return f"reused_{self.prior_label}_tagged"

    @property
    def prior_tagged_path(self) -> Path:
        return self.prior_dir / "perf_all_tagged.json"

    @property
    def weights_path(self) -> Path:
        return self.round_dir / "question_weights.json"

    @property
    def pull_rescore_path(self) -> Path:
        return self.round_dir / "platform_rescored.json"

    def perf_path(self, slug: str) -> Path:
        return self.round_dir / f"perf_{slug}.json"


@dataclass(frozen=True)
class DegradedCohort:
    """The dry-key degraded cohort as this round sees it, plus the telemetry audit trail."""

    runs: list[dict]
    telemetry_question_ids: set[int]
    union_question_ids: set[int]
    resolved_pairs: set[QuestionIds]


@dataclass(frozen=True)
class RoundDataset:
    """The tagged records plus everything the round's output files quote about how they got there."""

    spec: RoundSpec
    records: list[dict]
    healing_changes: list[dict]
    degraded: DegradedCohort
    prior_index: dict[RecordKey, dict]
    bot_score_drift: list[dict]
    platform_score_drift: list[dict]
    pull_payload: dict


def record_key(record: dict) -> RecordKey:
    """The dedup and cross-round join key. Both ids, because either alone collides."""
    return (record.get("question_id"), record.get("post_id"))


def sorted_question_ids(records: Iterable[dict]) -> list[int]:
    """The question ids of ``records``, sorted, for a log line or an audit list."""
    return sorted(question_id for record in records if (question_id := record.get("question_id")) is not None)


def is_scored(record: dict) -> bool:
    """A record is scored if it carries a non-null score for its own question type."""
    qtype = record.get("type")
    if qtype == "binary":
        return record.get("log_score") is not None
    if qtype in NUMERIC_TYPES:
        return record.get("numeric_log_score") is not None
    if qtype == "multiple_choice":
        return record.get("mc_log_score") is not None
    return False


def strip_tags(record: dict) -> dict:
    for tag_field in TAG_FIELDS:
        record.pop(tag_field, None)
    return record


def load_marker_records(telemetry_dir: Path, marker: str) -> list[dict]:
    """One marker's archived lines. Local reader because the package may not import ``scripts``."""
    path = telemetry_dir / f"{marker}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def degraded_cohort(id_map: QuestionIdMap, telemetry_dir: Path) -> DegradedCohort:
    """Questions touched by a triple-era run that dropped forecasters, telemetry joined to canon.

    Every ``forecaster_drops`` marker with ``total > 0`` at or after the triple-era boundary, joined
    to that run's own per-question markers and unioned with the canonical cohort, so an archive gap
    cannot silently un-tag a known-degraded question.
    """
    drops = [
        drop
        for drop in load_marker_records(telemetry_dir, "forecaster_drops")
        if (drop.get("total") or 0) > 0
        and (parse_iso_utc(drop.get("run_date")) or WIDENING_FLIP_MERGED_AT) >= B4E9DF0_MERGED_AT
    ]
    markers_by_run: dict[str, list[dict]] = defaultdict(list)
    for marker in DEGRADED_JOIN_MARKERS:
        for record in load_marker_records(telemetry_dir, marker):
            markers_by_run[record["run_id"]].append(record)

    resolved: set[QuestionIds] = set()
    marker_question_ids: set[int] = set()
    runs: list[dict] = []
    for drop in drops:
        run_id = drop["run_id"]
        markers = markers_by_run.get(run_id, [])
        question_ids = sorted({m["qid"] for m in markers if m.get("qid_kind") == "question_id"})
        post_ids = sorted({m["qid"] for m in markers if m.get("qid_kind") == "post_id"})
        marker_question_ids.update(question_ids)
        run_resolved: list[dict] = []
        for marker_record in markers:
            ids = id_map.resolve_marker_record(marker_record)
            if ids is not None:
                resolved.add(ids)
                run_resolved.append({"post_id": ids.post_id, "question_id": ids.question_id})
        runs.append(
            {
                "run_id": run_id,
                "run_date": drop.get("run_date"),
                "workflow": drop.get("workflow"),
                "drops_total": drop.get("total"),
                "drop_detail": drop.get("detail"),
                "marker_question_ids": question_ids,
                "marker_post_ids": post_ids,
                "resolved_against_perf_records": run_resolved,
            }
        )
    return DegradedCohort(
        runs=runs,
        telemetry_question_ids=marker_question_ids,
        union_question_ids=set(marker_question_ids) | set(DEGRADED_FULL_QUESTION_IDS),
        resolved_pairs=resolved,
    )


def load_fresh(slug: str, path: Path, provenance: str, *, required: bool) -> list[dict]:
    """One fresh per-tournament pull, provenance-stamped. A missing optional slug is not an error."""
    if not path.exists():
        if required:
            logger.warning(f"REQUIRED source missing, skipping: {path}")
        return []
    with open(path) as f:
        records = json.load(f)
    if not records:
        logger.info(f"  loaded    0 records from {slug} (empty file)")
        return []
    for record in records:
        strip_tags(record)
        record["source_tournament"] = slug
        record["source_is_fresh_pull"] = True
        record["record_provenance"] = provenance
    logger.info(f"  loaded {len(records):>4} records from {slug} [{provenance}]")
    return records


def load_reused(prior_records: list[dict], spec: RoundSpec) -> list[dict]:
    """The baselines this round does not re-pull, retagged from the prior round's tagged file."""
    out: list[dict] = []
    counts: Counter = Counter()
    for prior in prior_records:
        slug = prior.get("source_tournament")
        if slug not in spec.reused_slugs:
            continue
        record = strip_tags(dict(prior))
        record["source_tournament"] = slug
        record["source_is_fresh_pull"] = False
        record["record_provenance"] = spec.reused_provenance
        out.append(record)
        counts[slug] += 1
    for slug in spec.reused_slugs:
        logger.info(f"  loaded {counts[slug]:>4} records from {slug} [reused from {spec.prior_label} tagged file]")
    return out


def prior_provenance(prior_records: list[dict]) -> dict[RecordKey, dict[str, list[str]]]:
    """Prior rounds' rescore provenance, keyed rather than carried: a fresh record wins dedup."""
    return {
        record_key(record): {
            "rescored_fields": sorted(record.get("rescored_fields") or []),
            "platform_rescored": sorted(record.get("platform_rescored") or []),
        }
        for record in prior_records
    }


def prior_snapshot(prior_records: list[dict], prior_label: str) -> tuple[set[RecordKey], set[RecordKey], dict]:
    """(all prior keys, prior keys already scored, the prior per-record view) for the new-since cut."""
    keys = {record_key(record) for record in prior_records}
    scored = {record_key(record) for record in prior_records if is_scored(record)}
    index = {
        record_key(record): {
            "log_score": record.get("log_score"),
            "mc_log_score": record.get("mc_log_score"),
            "numeric_log_score": record.get("numeric_log_score"),
            # Hoisted flat here and read flat by the drift path; insertion order is output order.
            **{field: accessor(record) for field, accessor in PLATFORM_DRIFT_ACCESSORS},
            "our_prob_yes": record.get("our_prob_yes"),
            "config_era": record.get("config_era"),
            "triple_subera_fine": record.get("triple_subera_fine"),
        }
        for record in prior_records
    }
    logger.info(f"  prior round ({prior_label}): {len(prior_records)} records, {len(scored)} scored")
    return keys, scored, index


def dedup(records: list[dict]) -> list[dict]:
    """Dedup on (question_id, post_id), preferring a fresh-pull record over its reused twin."""
    by_key: dict[RecordKey, dict] = {}
    dropped = 0
    dropped_by_slug: Counter = Counter()
    for record in records:
        key = record_key(record)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = record
            continue
        dropped += 1
        dropped_by_slug[record.get("source_tournament")] += 1
        if record.get("source_is_fresh_pull") and not existing.get("source_is_fresh_pull"):
            by_key[key] = record
    if dropped:
        logger.info(f"  deduped {dropped} duplicate (question_id, post_id) records: {dict(dropped_by_slug)}")
    return list(by_key.values())


def score_transition(prior_value: float | None, now: float | None) -> str | None:
    """Which of the three ways a score pair differs between two reads, or None when it held.

    A score appearing or disappearing is a real change, so it is classified rather than skipped:
    ``docs/performance_analysis.md`` "The three drift transitions".
    """
    if prior_value is None and now is None:
        return None
    if prior_value is None:
        return DRIFT_APPEARED
    if now is None:
        return DRIFT_DISAPPEARED
    return DRIFT_MOVED if abs(float(prior_value) - float(now)) > SCORE_ATOL else None


def _score_deltas(before: dict[str, float | None], record: dict) -> dict[str, dict]:
    deltas: dict[str, dict] = {}
    for score_field in SCORE_FIELDS:
        old, new = before[score_field], record.get(score_field)
        if score_transition(old, new) is not None:
            deltas[score_field] = {"before": old, "after": new}
    return deltas


def heal_stored_scores(records: list[dict], fresh_provenance: str) -> list[dict]:
    """Recompute every score from stored inputs and return the per-record change log.

    A fresh record changing here is a red flag, because the pull was scored by the current
    collector; a reused record changing is the expected healing of a stale stored value.
    """
    before = {record_key(record): {f: record.get(f) for f in SCORE_FIELDS} for record in records}
    changed_n = rescore_records(records)
    changes: list[dict] = []
    for record in records:
        deltas = _score_deltas(before[record_key(record)], record)
        this_round = sorted(deltas)
        record["rescored_fields_this_round"] = this_round
        record["rescored_fields"] = sorted(set(record.get("rescored_fields_prior_rounds") or []) | set(this_round))
        if deltas:
            changes.append(
                {
                    "question_id": record.get("question_id"),
                    "post_id": record.get("post_id"),
                    "source_tournament": record.get("source_tournament"),
                    "provenance": record.get("record_provenance"),
                    "zero_point": (record.get("scaling") or {}).get("zero_point"),
                    "type": record.get("type"),
                    "deltas": deltas,
                }
            )
    fresh_changed = [change for change in changes if change["provenance"] == fresh_provenance]
    cumulative = [record for record in records if record["rescored_fields"]]
    logger.info(
        f"  cumulative rescored_fields (prior rounds' heals carried + this round's): "
        f"{len(cumulative)} record(s) -> {sorted_question_ids(cumulative)}"
    )
    logger.info(f"  rescore_records reports {changed_n} record(s) healed; per-field audit found {len(changes)}")
    logger.info(f"  reused records healed: {len(changes) - len(fresh_changed)}")
    if fresh_changed:
        logger.warning(
            f"{len(fresh_changed)} FRESH-pull record(s) changed under rescore; the fresh pull should "
            f"already carry current-scorer values, so investigate before trusting either: {fresh_changed}"
        )
    else:
        logger.info("  fresh-pull records healed: 0 (as expected, the fresh pull used the current scorer)")
    for change in changes:
        if change["provenance"] != fresh_provenance:
            detail = "; ".join(f"{f}: {d['before']} -> {d['after']}" for f, d in change["deltas"].items())
            logger.info(
                f"      healed [{change['source_tournament']}] qid={change['question_id']} "
                f"type={change['type']} zero_point={change['zero_point']}: {detail}"
            )
    return changes


def load_question_weights(path: Path) -> dict[int, float]:
    """Leaderboard question weights, keyed by QUESTION id; the leaderboard totals spot * weight."""
    with open(path) as f:
        payload = json.load(f)
    weights = {int(qid): float(weight) for qid, weight in payload["weights"].items()}
    recon = payload.get("reconciliation") or {}
    logger.info(
        f"  loaded {len(weights)} question weight(s); leaderboard reconciliation residual "
        f"{recon.get('residual')} on board {recon.get('leaderboard_board_id')}"
    )
    logger.info(f"  weight distribution over the resolved posts: {payload.get('weight_distribution')}")
    return weights


def load_pull_rescores(path: Path) -> tuple[set[RecordKey], dict[RecordKey, list[str]], dict]:
    """The pull side's platform re-resolution diff (``rescore_diff`` wrote it during the pull)."""
    with open(path) as f:
        payload = json.load(f)
    fields_by_key: dict[RecordKey, list[str]] = defaultdict(list)
    for change in payload.get("changes") or []:
        fields_by_key[(change.get("question_id"), change.get("post_id"))].append(change["field"])
    logger.info(
        f"  platform_rescored.json: compared={payload.get('compared')} "
        f"rescored_records={payload.get('rescored_records')} "
        f"unmatched={payload.get('unmatched_no_prior_counterpart')} "
        f"changed_fields={len(payload.get('changes') or [])} "
        f"tag_distribution={payload.get('tag_distribution')}"
    )
    return set(fields_by_key), {key: sorted(set(fields)) for key, fields in fields_by_key.items()}, payload


def _drift_against_prior(record: dict, prior_view: dict) -> tuple[list[dict], list[dict]]:
    """One re-pulled record's score moves, split into ours (bot-side) and Metaculus's.

    The two sides of each comparison are read DIFFERENTLY on purpose: ``docs/performance_analysis.md``
    "Why the two sides of a drift comparison are read differently".
    """
    key = record_key(record)
    bot_drift: list[dict] = []
    for score_field in BOT_DRIFT_FIELDS:
        old, new = prior_view.get(score_field), record.get(score_field)
        transition = score_transition(old, new)
        if transition is not None:
            bot_drift.append({"key": key, "field": score_field, "transition": transition, "prior": old, "now": new})
    platform_drift: list[dict] = []
    for score_field, accessor in PLATFORM_DRIFT_ACCESSORS:
        old, now = prior_view.get(score_field), accessor(record)
        transition = score_transition(old, now)
        if transition is not None:
            platform_drift.append(
                {"key": key, "field": score_field, "transition": transition, "prior": old, "now": now}
            )
    return bot_drift, platform_drift


def _transition_tally(drift: list[dict]) -> dict[str, int]:
    """How many of this round's drift entries appeared, disappeared and moved."""
    counts = Counter(entry["transition"] for entry in drift)
    return {state: counts[state] for state in (DRIFT_APPEARED, DRIFT_DISAPPEARED, DRIFT_MOVED) if counts[state]}


def _log_drift_entries(drift: list[dict]) -> None:
    for entry in drift:
        logger.info(f"      {entry['key']} {entry['field']} {entry['transition']}: {entry['prior']} -> {entry['now']}")


def tag_platform_rescores(
    records: list[dict],
    prior_index: dict[RecordKey, dict],
    spec: RoundSpec,
) -> tuple[list[dict], list[dict], dict]:
    """Tag Metaculus-side score movement, cross-checking the pull's diff against a local recompute."""
    pull_keys, pull_fields, pull_payload = load_pull_rescores(spec.pull_rescore_path)

    bot_drift: list[dict] = []
    platform_drift: list[dict] = []
    for record in records:
        key = record_key(record)
        prior = prior_index.get(key)
        moved: list[str] = []
        if prior is not None and record.get("source_is_fresh_pull"):
            record_bot_drift, record_platform_drift = _drift_against_prior(record, prior)
            bot_drift.extend(record_bot_drift)
            platform_drift.extend(record_platform_drift)
            moved = [entry["field"] for entry in record_platform_drift]
        # The pull-side diff also compares the resolution and every metaculus_scores key.
        moved = sorted(set(moved) | set(pull_fields.get(key, [])))
        record["platform_rescored_this_round"] = moved
        record["platform_rescored"] = sorted(set(record.get("platform_rescored_prior_rounds") or []) | set(moved))
        compared = record.get("source_is_fresh_pull") and prior is not None
        record["platform_rescored_pull_tag"] = (key in pull_keys) if compared else None

    _log_pull_tag_agreement(records, spec.weighted_slug, pull_payload)
    logger.info(f"  bot log-score drift: {len(bot_drift)} field(s) changed {_transition_tally(bot_drift)}")
    _log_drift_entries(bot_drift)
    logger.info(
        f"  metaculus platform-score drift (spot_peer + peer): {len(platform_drift)} field(s) changed "
        f"{_transition_tally(platform_drift)}"
    )
    _log_drift_entries(platform_drift)
    if platform_drift:
        logger.warning(
            f"a platform score {'/'.join(sorted({e['transition'] for e in platform_drift}))} on a re-pulled "
            "record: Metaculus re-scored, re-resolved or un-scored it, so any prior-round table quoting "
            "the old value is stale for that question."
        )
    cumulative = [record for record in records if record["platform_rescored"]]
    logger.info(
        f"  cumulative platform_rescored (prior rounds carried + this round): {len(cumulative)} record(s) -> "
        f"{sorted_question_ids(cumulative)}"
    )
    return bot_drift, platform_drift, pull_payload


def _log_pull_tag_agreement(records: list[dict], weighted_slug: str, pull_payload: dict) -> None:
    """The local ternary tag must reproduce the pull side's distribution exactly."""
    tag_counts = Counter(
        "none" if r["platform_rescored_pull_tag"] is None else ("true" if r["platform_rescored_pull_tag"] else "false")
        for r in records
        if r["source_tournament"] == weighted_slug
    )
    logger.info(f"  platform_rescored_pull_tag over the {weighted_slug} records: {dict(tag_counts)}")
    expected = pull_payload.get("tag_distribution") or {}
    states = ("true", "false", "none")
    if {k: tag_counts.get(k, 0) for k in states} != {k: expected.get(k, 0) for k in states}:
        logger.warning(f"ternary tag distribution disagrees with platform_rescored.json ({expected}), investigate")
    else:
        logger.info("  ternary tag distribution reproduces platform_rescored.json exactly")


def _load_sources(spec: RoundSpec, prior_records: list[dict]) -> list[dict]:
    logger.info("=== loading sources ===")
    records: list[dict] = []
    for slug in spec.required_slugs:
        records.extend(load_fresh(slug, spec.perf_path(slug), spec.fresh_provenance, required=True))
    for slug in spec.optional_slugs:
        records.extend(load_fresh(slug, spec.perf_path(slug), spec.fresh_provenance, required=False))
    records.extend(load_reused(prior_records, spec))
    logger.info(f"total records loaded (pre-dedup): {len(records)}")
    records = dedup(records)
    logger.info(f"total records after dedup: {len(records)}")

    survivors = [r for r in records if r["source_tournament"] == spec.weighted_slug and not r["source_is_fresh_pull"]]
    logger.info(f"  reused records that SURVIVED dedup (the fresh pull is missing them): {len(survivors)}")
    if survivors:
        logger.warning(f"the fresh {spec.weighted_slug} pull lost {len(survivors)} record(s), investigate")
    for record in survivors:
        logger.info(
            f"      qid={record.get('question_id')} pid={record.get('post_id')} :: {(record.get('title') or '')[:60]}"
        )

    carried = prior_provenance(prior_records)
    for record in records:
        provenance = carried.get(record_key(record)) or {}
        record["rescored_fields_prior_rounds"] = provenance.get("rescored_fields") or []
        record["platform_rescored_prior_rounds"] = provenance.get("platform_rescored") or []
    for provenance_field in ("rescored_fields_prior_rounds", "platform_rescored_prior_rounds"):
        with_provenance = [r for r in records if r[provenance_field]]
        logger.info(
            f"  carried prior-round {provenance_field.removesuffix('_prior_rounds')} provenance onto "
            f"{len(with_provenance)} record(s): {sorted_question_ids(with_provenance)}"
        )
    return records


def _tag_eras_and_weights(records: list[dict], weights: dict[int, float], weighted_slug: str) -> None:
    for record in records:
        record["config_era"] = era_of(record)
        record["triple_subera"] = triple_subera_of(record)
        record["triple_subera_fine"] = triple_subera_fine_of(record)
        record["ft_unfreeze_side"] = ft_unfreeze_side_of(record)
        record["post_linters_merge"] = post_linters_merge_of(record)
        question_id = record.get("question_id")
        record["question_weight"] = (
            weights.get(question_id)
            if record["source_tournament"] == weighted_slug and question_id is not None
            else None
        )
    logger.info(f"  tagged {len(records)} records on bot_comment_created_at")
    weighted = [r for r in records if r["question_weight"] is not None]
    downweighted = [r for r in weighted if r["question_weight"] < 1.0]
    logger.info(
        f"  question_weight stamped on {len(weighted)} {weighted_slug} record(s); {len(downweighted)} carry a "
        f"weight below 1 (distribution {dict(Counter(r['question_weight'] for r in downweighted).most_common())})"
    )
    unweighted = [r for r in records if r["source_tournament"] == weighted_slug and r["question_weight"] is None]
    if unweighted:
        logger.warning(
            f"{len(unweighted)} {weighted_slug} record(s) have NO weight: {[r['question_id'] for r in unweighted]}"
        )


def _tag_cohorts_and_novelty(
    records: list[dict],
    prior_all: set[RecordKey],
    prior_scored: set[RecordKey],
    degraded: DegradedCohort,
) -> None:
    for record in records:
        key = record_key(record)
        qid = record.get("question_id")
        record["is_new_since_prior"] = key not in prior_all
        record["newly_scored"] = is_scored(record) and key not in prior_scored
        in_triple = record["config_era"] == "triple_era"
        record["degraded_run"] = in_triple and (
            qid in degraded.union_question_ids or QuestionIds.from_perf_record(record) in degraded.resolved_pairs
        )
        record["partial_degraded"] = in_triple and qid in DEGRADED_PARTIAL_QUESTION_IDS
        record["known_bug"] = qid in KNOWN_BUG_QUESTION_IDS


def _log_degraded_cohort(degraded: DegradedCohort) -> None:
    logger.info("=== degraded (dry-key) cohort, pinned from telemetry + canonical list ===")
    for run in degraded.runs:
        logger.info(
            f"  {run['run_date']} {run['workflow']} run={run['run_id']} drops={run['drops_total']} "
            f"marker_question_ids={run['marker_question_ids']} marker_post_ids={run['marker_post_ids']}"
        )
    telemetry_ids = degraded.telemetry_question_ids
    logger.info(f"  telemetry-derived degraded question_ids: {sorted(telemetry_ids)}")
    logger.info(f"  canonical DEGRADED_RUN_QIDS: {sorted(DEGRADED_FULL_QUESTION_IDS)}")
    logger.info(f"  telemetry-only (not in the canonical list): {sorted(telemetry_ids - DEGRADED_FULL_QUESTION_IDS)}")
    logger.info(f"  canonical-only (telemetry no longer shows): {sorted(DEGRADED_FULL_QUESTION_IDS - telemetry_ids)}")
    logger.info(f"  -> union used for tagging: {sorted(degraded.union_question_ids)}")
    logger.info(f"  -> id-map resolved {len(degraded.resolved_pairs)} (question_id, post_id) pair(s)")


def build_round_dataset(spec: RoundSpec) -> RoundDataset:
    """Load, dedup, heal, tag and cohort-flag every record this round covers."""
    with open(spec.prior_tagged_path) as f:
        prior_records = json.load(f)

    records = _load_sources(spec, prior_records)

    logger.info("=== healing stored scores (collector.rescore_records) ===")
    healing_changes = heal_stored_scores(records, spec.fresh_provenance)

    logger.info("=== leaderboard question weights ===")
    weights = load_question_weights(spec.weights_path)

    logger.info("=== tagging ===")
    _tag_eras_and_weights(records, weights, spec.weighted_slug)

    # The perf dataset is the only authoritative both-id source, so the map is built after loading.
    degraded = degraded_cohort(QuestionIdMap.from_perf_records(records), spec.telemetry_dir)
    _log_degraded_cohort(degraded)

    logger.info("=== new-since-prior ===")
    prior_all, prior_scored, prior_index = prior_snapshot(prior_records, spec.prior_label)
    _tag_cohorts_and_novelty(records, prior_all, prior_scored, degraded)

    logger.info("=== platform re-resolution (the pull's diff, cross-checked here) ===")
    bot_drift, platform_drift, pull_payload = tag_platform_rescores(records, prior_index, spec)

    return RoundDataset(
        spec=spec,
        records=records,
        healing_changes=healing_changes,
        degraded=degraded,
        prior_index=prior_index,
        bot_score_drift=bot_drift,
        platform_score_drift=platform_drift,
        pull_payload=pull_payload,
    )
