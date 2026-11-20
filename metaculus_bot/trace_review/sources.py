"""Everything the trace-review builder reads: the round file, the fresh pulls and their cache, the
telemetry archive, and the exclusion counts a pull's raw posts imply.

No function here builds a bundle row. The pulls are the only network access the tool has: free,
read-only GETs through the collector's paced ``api_get`` (docs/trace_review.md "What the build pulls").
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from metaculus_bot import api_preflight
from metaculus_bot.constants import (
    MANTIC_TOURNAMENT_ID,
    METACULUS_CUP_ID,
    PLATFORM_MANTIC,
    PLATFORM_METACULUS,
    TOURNAMENT_ID,
)
from metaculus_bot.performance_analysis import collector
from metaculus_bot.performance_analysis.collector import CollectorPlatform, questions_on_post
from metaculus_bot.performance_analysis.id_mapping import QuestionIdMap, record_platform
from metaculus_bot.performance_analysis.parsing import SKIP_RESOLUTIONS
from metaculus_bot.performance_analysis.round_dataset import RecordKey, load_marker_records
from metaculus_bot.time_utils import parse_iso_utc
from metaculus_bot.trace_review.config import (
    MANTIC_RUN_MARKER,
    PULL_MAX_AGE_HOURS,
    ROUND_DATASET_NAME,
    ROUND_DIR_PREFIX,
    RUN_LEVEL_MARKERS,
    RUNS_INDEX_FILE,
)

logger: logging.Logger = logging.getLogger(__name__)

_ROUND_DIR_RE = re.compile(rf"^{re.escape(ROUND_DIR_PREFIX)}(\d{{4}}-\d{{2}}-\d{{2}})$")
PRACTICE_TITLE_PREFIX = "[PRACTICE]"

EXCLUDED_DATE = "date_question"
EXCLUDED_MULTI_RESOLUTION = "multi_resolution"
EXCLUDED_ANNULLED = "annulled_or_ambiguous"
EXCLUDED_PRACTICE = "practice_post"
EXCLUSION_KINDS: tuple[str, ...] = (EXCLUDED_DATE, EXCLUDED_MULTI_RESOLUTION, EXCLUDED_ANNULLED, EXCLUDED_PRACTICE)


class MissingPlatformTokenError(RuntimeError):
    """A requested platform's token is not set, so its pull would return nothing the bot forecast."""


def forecast_key(platform: str, question_id: object) -> str:
    """The bundle's join key: the question id is the platform's own primary key."""
    return f"{platform}:{question_id}"


def forecast_key_of(record: Mapping[str, object]) -> str:
    return forecast_key(record_platform(dict(record)), record["question_id"])


# ---------------------------------------------------------------------------
# The round
# ---------------------------------------------------------------------------


def newest_round_dataset(scratch_dir: Path) -> Path:
    """The ``perf_all_tagged.json`` of the newest ``residual_YYYY-MM-DD`` round, by the date in its name."""
    dated: list[tuple[date, Path]] = []
    for directory in scratch_dir.iterdir():
        match = _ROUND_DIR_RE.match(directory.name)
        if match is not None and (directory / ROUND_DATASET_NAME).exists():
            dated.append((date.fromisoformat(match.group(1)), directory / ROUND_DATASET_NAME))
    if not dated:
        raise FileNotFoundError(f"no {ROUND_DIR_PREFIX}YYYY-MM-DD/{ROUND_DATASET_NAME} under {scratch_dir}")
    return max(dated)[1]


def round_label_of(round_path: Path) -> str:
    return round_path.parent.name.removeprefix(ROUND_DIR_PREFIX)


def metaculus_slugs(round_records: Iterable[Mapping[str, object]]) -> list[str]:
    """The current-season slugs plus every slug the round pulled fresh, so late resolutions arrive."""
    fresh = [str(r["source_tournament"]) for r in round_records if r.get("source_is_fresh_pull")]
    return list(dict.fromkeys([TOURNAMENT_ID, METACULUS_CUP_ID, *fresh]))


def mantic_slugs(telemetry_dir: Path) -> list[str]:
    """The configured Mantic tournament plus every one a Mantic run has named in its preflight marker."""
    configured = [str(line["configured"]) for line in load_marker_records(telemetry_dir, MANTIC_RUN_MARKER)]
    return list(dict.fromkeys([MANTIC_TOURNAMENT_ID, *configured]))


# ---------------------------------------------------------------------------
# What a pull's raw posts say
# ---------------------------------------------------------------------------


def _bot_forecast(question: dict) -> bool:
    my_forecasts = question.get("my_forecasts")
    return isinstance(my_forecasts, dict) and bool(my_forecasts.get("latest"))


def _exclusion_kind(question: dict) -> str | None:
    """Why the collector drops this resolved, bot-forecast question, in the collector's own order."""
    if question.get("type") == "date":
        return EXCLUDED_DATE
    if question.get("multi_resolution") is True:
        return EXCLUDED_MULTI_RESOLUTION
    if str(question.get("resolution")) in SKIP_RESOLUTIONS:
        return EXCLUDED_ANNULLED
    return None


def exclusion_counts(posts: Iterable[dict]) -> dict[str, int]:
    """Resolved questions the bot forecast that never become records, by reason; practice posts by post."""
    counts: Counter[str] = Counter(dict.fromkeys(EXCLUSION_KINDS, 0))
    for post in posts:
        forecast = [q for q in questions_on_post(post) if _bot_forecast(q) and q.get("resolution") is not None]
        if str(post.get("title") or "").startswith(PRACTICE_TITLE_PREFIX):
            counts[EXCLUDED_PRACTICE] += bool(forecast)
            continue
        counts.update(kind for q in forecast if (kind := _exclusion_kind(q)) is not None)
    return dict(counts)


def question_weights(posts: Iterable[dict]) -> dict[int, float]:
    """``{question id: question_weight}`` as the posts list serves it (the leaderboard multiplier)."""
    weights: dict[int, float] = {}
    for post in posts:
        for question in questions_on_post(post):
            weight = question.get("question_weight")
            if weight is not None:
                weights[int(question["id"])] = float(weight)
    return weights


def posts_needing_comments(posts: Iterable[dict], round_keys: set[RecordKey]) -> set[int]:
    """Posts holding a question the round does not already cover; only these need their comment."""
    return {
        int(post["id"])
        for post in posts
        if any((q.get("id"), post["id"]) not in round_keys for q in questions_on_post(post))
    }


# ---------------------------------------------------------------------------
# Pulls and their cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PullTarget:
    """One platform's slugs to pull, where the cache lives and what the comment lookup may skip."""

    platform: CollectorPlatform
    slugs: Sequence[str]
    platform_dir: Path
    round_label: str
    round_keys: set[RecordKey] = field(default_factory=set)

    def perf_path(self, slug: str) -> Path:
        return self.platform_dir / f"perf_{slug}.json"

    def meta_path(self, slug: str) -> Path:
        return self.platform_dir / f"pull_{slug}.json"


def require_token(platform: CollectorPlatform) -> str:
    token = os.environ.get(platform.token_env)
    if not token:
        raise MissingPlatformTokenError(
            f"{platform.token_env} is not set, so a {platform.name} pull would see none of the bot's forecasts. "
            f'Add it to .env, or build without {platform.name}: make trace_index ARGS="--platforms '
            f'{PLATFORM_METACULUS if platform.name == PLATFORM_MANTIC else PLATFORM_MANTIC}"'
        )
    return token


def _cache_is_fresh(meta_path: Path, round_label: str, now: datetime) -> bool:
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    pulled_at = datetime.fromisoformat(meta["pulled_at"])
    return meta["round_label"] == round_label and now - pulled_at < timedelta(hours=PULL_MAX_AGE_HOURS)


def ensure_pulls(target: PullTarget, *, research_latest_dir: Path, refresh: bool, now: datetime) -> dict[str, dict]:
    """Pull every stale slug of one platform (comments listed once), and return each slug's metadata.

    A slug is reused from the cache when its pull is younger than ``PULL_MAX_AGE_HOURS`` and was
    made against the same round, since which comments were expanded depends on the round.
    """
    token = require_token(target.platform)
    stale = [
        slug for slug in target.slugs if refresh or not _cache_is_fresh(target.meta_path(slug), target.round_label, now)
    ]
    logger.info(f"{target.platform.name}: {len(target.slugs) - len(stale)} cached pull(s) reused, {len(stale)} to pull")
    if stale:
        api_preflight.verify_api_identity(target.platform.api_base_url)
        base_url = target.platform.api_base_url
        comments = collector.fetch_bot_comments(target.platform.bot_user_id, token, base_url=base_url)
        target.platform_dir.mkdir(parents=True, exist_ok=True)
        for slug in stale:
            posts = collector.fetch_resolved_questions(slug, token, base_url=base_url)
            records = collector.records_from_posts(
                posts,
                comments,
                token,
                platform=target.platform,
                research_archive_dir=research_latest_dir,
                comment_post_ids=posts_needing_comments(posts, target.round_keys),
            )
            collector.save_dataset(records, str(target.perf_path(slug)))
            meta = {
                "platform": target.platform.name,
                "slug": slug,
                "pulled_at": now.isoformat(),
                "round_label": target.round_label,
                "n_posts": len(posts),
                "n_records": len(records),
                "excluded": exclusion_counts(posts),
                "question_weights": {str(qid): weight for qid, weight in question_weights(posts).items()},
            }
            target.meta_path(slug).write_text(json.dumps(meta, indent=2))
    return {slug: json.loads(target.meta_path(slug).read_text()) for slug in target.slugs}


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunTelemetry:
    """One forecast's lines from its chosen run, and that run's run-level lines."""

    run_id: str | None
    run_date: datetime | None
    n_runs: int
    lines: list[dict]
    run_lines: list[dict]


NO_TELEMETRY = RunTelemetry(run_id=None, run_date=None, n_runs=0, lines=[], run_lines=[])


@dataclass(frozen=True)
class TelemetryIndex:
    """Archived marker lines joined to forecasts, plus when each marker first appears anywhere."""

    lines_by_forecast: dict[str, dict[str, list[dict]]]
    run_level_lines: dict[str, list[dict]]
    first_run_dates: dict[str, datetime]
    newest_run_date: datetime | None

    def run_for(self, key: str) -> RunTelemetry:
        """The forecast's latest run, the way the collector keeps the newest comment."""
        runs = self.lines_by_forecast.get(key)
        if not runs:
            return NO_TELEMETRY
        run_id = max(runs, key=lambda rid: _run_date(runs[rid][0]) or datetime.min.replace(tzinfo=UTC))
        if len(runs) > 1:
            logger.warning(f"{key} has telemetry from {len(runs)} runs; using the latest, {run_id}")
        return RunTelemetry(
            run_id=run_id,
            run_date=_run_date(runs[run_id][0]),
            n_runs=len(runs),
            lines=runs[run_id],
            run_lines=self.run_level_lines.get(run_id, []),
        )

    def all_lines_for(self, key: str) -> list[dict]:
        return [line for lines in self.lines_by_forecast.get(key, {}).values() for line in lines]


def _run_date(line: Mapping[str, object]) -> datetime | None:
    raw = line.get("run_date")
    return parse_iso_utc(raw if isinstance(raw, str) else None)


def _marker_names(telemetry_dir: Path) -> list[str]:
    return sorted(path.stem for path in telemetry_dir.glob("*.jsonl") if path.name != RUNS_INDEX_FILE)


def build_telemetry_index(telemetry_dir: Path, id_maps: Mapping[str, QuestionIdMap]) -> TelemetryIndex:
    """Join every per-question marker line to a forecast, telling the platforms apart by run id.

    ``workflow`` reads "unknown" on most archived lines, so the platform comes from the one
    marker a Mantic run always logs. Within a platform the line's own ``qid_kind`` picks the id
    space (``id_mapping.resolve_marker_record``); a line resolving to no forecast is dropped.
    """
    by_marker = {marker: load_marker_records(telemetry_dir, marker) for marker in _marker_names(telemetry_dir)}
    mantic_runs = {line["run_id"] for line in by_marker.get(MANTIC_RUN_MARKER, [])}
    lines_by_forecast: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    run_level: dict[str, list[dict]] = defaultdict(list)
    first_dates: dict[str, datetime] = {}
    newest: datetime | None = None
    for marker, lines in by_marker.items():
        dates = [d for d in (_run_date(line) for line in lines) if d is not None]
        if dates:
            first_dates[marker] = min(dates)
            newest = max(dates) if newest is None else max(newest, *dates)
        for line in lines:
            if marker in RUN_LEVEL_MARKERS and "qid" not in line:
                run_level[line["run_id"]].append(line)
                continue
            platform = PLATFORM_MANTIC if line["run_id"] in mantic_runs else PLATFORM_METACULUS
            ids = id_maps[platform].resolve_marker_record(line)
            if ids is not None:
                lines_by_forecast[forecast_key(platform, ids.question_id)][line["run_id"]].append(line)
    joined = sum(len(lines) for runs in lines_by_forecast.values() for lines in runs.values())
    logger.info(f"telemetry: {joined} per-question line(s) joined to {len(lines_by_forecast)} forecast(s)")
    return TelemetryIndex(
        lines_by_forecast={key: dict(runs) for key, runs in lines_by_forecast.items()},
        run_level_lines=dict(run_level),
        first_run_dates=first_dates,
        newest_run_date=newest,
    )
