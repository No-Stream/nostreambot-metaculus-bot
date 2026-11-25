"""Question-supply probe over a tournament's post statuses, INCLUDING ``closed``.

Per slug it reports posts and questions at each status, the backlog of unresolved questions
past their own ``scheduled_resolve_time``, the FORFEIT sweep (closed or resolved questions the
bot never forecast, with their windows) and the miss rate per UTC release hour. Two platforms:
Metaculus (default; ``METACULUS_TOKEN`` required, ``my_forecasts`` read off the list page and
fetched per post only where it is missing) and Mantic's Crucible (``--platform mantic``; public
reads, ``MANTIC_TOKEN`` optional, resolved questions classified from the public spot-time
snapshot). Every list GET carries ``with_cp=true``, which is what puts both of those on the list
page. The platform seams are the ``PlatformProbe`` table in ``scripts/supply_probe_platforms.py``.

Read-only and free: only the platform's posts list and post detail, no LLM, research or publish
call. Paging stops on the first short page and every request carries a bounded 429 retry. Why
it exists, the API facts it is built around and how to read the report: ``docs/supply_probe.md``.

Usage:
    uv run python scripts/supply_probe.py
    uv run python scripts/supply_probe.py --slugs fall-futureeval-2026 --statuses open closed
    uv run python scripts/supply_probe.py --no-forfeits          # counts only, no detail GETs
    uv run python scripts/supply_probe.py --platform mantic     # public reads; MANTIC_TOKEN optional
    make supply_probe
    make supply_probe ARGS="--slugs metaculus-cup-fall-2026 --output /tmp/supply.json"
    make supply_probe_mantic ARGS="--slugs series-2 --output scratch/mantic_supply.json"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from metaculus_bot.api_preflight import verify_api_identity
from metaculus_bot.config import load_environment
from metaculus_bot.constants import MANTIC_TOKEN_ENV, MANTIC_TOURNAMENT_ID, PLATFORM_METACULUS

# Shared with the scoring pull so the two count questions one way (docs/supply_probe.md "API facts").
from metaculus_bot.performance_analysis.collector import MIN_REQUEST_INTERVAL_SECS, questions_on_post
from metaculus_bot.time_utils import _as_utc, parse_iso_utc
from scripts.supply_probe_platforms import (
    DEFAULT_SLUGS,
    FORECAST_ABSENT,
    FORECAST_PRESENT,
    FORECAST_UNKNOWN,
    METACULUS_PROBE,
    PLATFORM_PROBES,
    POSTS_URL,
    PlatformProbe,
    bot_forecast_state,
)

logger = logging.getLogger(__name__)

# `closed` is the whole point of the utility; --statuses asks about any other status the API accepts.
DEFAULT_STATUSES: tuple[str, ...] = ("open", "closed", "resolved")

# The forfeit sweep's scope: an OPEN question the bot has not forecast yet is not a forfeit.
FORFEIT_STATUSES: tuple[str, ...] = ("closed", "resolved")

PAGE_SIZE = 100
MAX_PAGES = 40  # 4,000 posts per status — an order of magnitude above any slug we probe
REQUEST_SPACING_SECS = 1.0
# Shared with the scoring pull so the two read-only Metaculus walkers keep one politeness.
DETAIL_REQUEST_SPACING_SECS = MIN_REQUEST_INTERVAL_SECS
# At 2 s spacing this is an INFO line about every 50 s: enough to tell a slow sweep from a wedged one.
DETAIL_PROGRESS_EVERY = 25
REQUEST_TIMEOUT_SECS = 45
MAX_RETRIES = 6
RETRY_BACKOFF_SECS = 6.0
SECONDS_PER_DAY = 86_400.0
SECONDS_PER_HOUR = 3_600.0
MINUTES_PER_HOUR = 60.0
DEFAULT_MAX_BACKLOG_ROWS = 20
DEFAULT_MAX_FORFEIT_ROWS = 20


@dataclass(frozen=True)
class QuestionRow:
    """One forecastable question, tagged with the post status it was paged under."""

    question_id: int
    post_id: int | None
    post_status: str
    question_type: str | None
    title: str
    scheduled_resolve_time: str | None
    is_resolved: bool
    open_time: str | None = None
    close_time: str | None = None
    # FORECAST_PRESENT / FORECAST_ABSENT / FORECAST_UNKNOWN; UNKNOWN whenever no payload answered.
    forecast_state: str = FORECAST_UNKNOWN


@dataclass(frozen=True)
class StatusCount:
    status: str
    posts: int
    questions: int


@dataclass(frozen=True)
class BacklogRow:
    """An unresolved question already past its own scheduled resolve time."""

    question_id: int
    post_id: int | None
    post_status: str
    question_type: str | None
    title: str
    scheduled_resolve_time: str
    overdue_days: float


@dataclass(frozen=True)
class ForfeitRow:
    """A question whose forecasting window shut without the bot ever forecasting it."""

    question_id: int
    post_id: int | None
    post_status: str
    question_type: str | None
    title: str
    open_time: str | None
    close_time: str | None
    window_hours: float | None
    is_resolved: bool


@dataclass(frozen=True)
class ForecastStateCounts:
    """How the forfeit-eligible questions split on "did we forecast this".

    ``unknown`` is disclosed rather than folded into either arm: it means the payload never
    answered, which is not the same fact as a forfeit. A slug where ``with_forecast`` is 0
    while ``without_forecast`` is large is far more likely a non-bot ``METACULUS_TOKEN`` than
    a total forfeit, and the split is what makes that readable.
    """

    with_forecast: int = 0
    without_forecast: int = 0
    unknown: int = 0

    @property
    def total(self) -> int:
        return self.with_forecast + self.without_forecast + self.unknown


@dataclass(frozen=True)
class ReleaseHourRow:
    """The forfeit-eligible questions that opened in one UTC hour of day, and how they split.

    ``hour_utc`` is None for questions whose ``open_time`` did not parse. ``miss_rate`` is
    ``no_forecast / (forecast + no_forecast)``, the share of DECIDED questions we lost, and None
    when nothing in the bucket was decided.
    """

    hour_utc: int | None
    questions: int
    forecast: int
    no_forecast: int
    unknown: int
    miss_rate: float | None


@dataclass(frozen=True)
class WindowMinutes:
    """Distribution of the realized open-to-close window over the eligible questions that carry both ends."""

    shortest: float
    median: float
    longest: float
    questions: int


@dataclass(frozen=True)
class SlugSupply:
    """One slug's supply census, or the error that stopped it."""

    slug: str
    status_counts: tuple[StatusCount, ...] = ()
    total_posts: int = 0
    total_questions: int = 0
    backlog: tuple[BacklogRow, ...] = ()
    unresolved_without_schedule: int = 0
    resolved_within_unresolved_posts: int = 0
    forfeits: tuple[ForfeitRow, ...] = ()
    forecast_states: ForecastStateCounts = ForecastStateCounts()
    by_release_hour: tuple[ReleaseHourRow, ...] = ()
    window_minutes: WindowMinutes | None = None
    error: str | None = None

    @property
    def worst_overdue_days(self) -> float:
        """Overdue margin of the worst backlog question; 0.0 when nothing is overdue."""
        return self.backlog[0].overdue_days if self.backlog else 0.0


def _question_is_resolved(question: Mapping[str, Any], post_status: str) -> bool:
    """Whether THIS question has resolved, independent of its post's status.

    A group post's status is the post's: its members resolve on their own schedules, so
    reading post status alone counts already-resolved members of a ``closed`` group as
    backlog. ``resolution`` is compared against None rather than tested for truthiness —
    a count question can resolve to 0, and a truthiness test would file that as pending.
    """
    if post_status == "resolved":
        return True
    return question.get("actual_resolve_time") is not None or question.get("resolution") is not None


def _close_time(question: Mapping[str, Any]) -> str | None:
    """When forecasting actually shut, preferring the realized close over the scheduled one."""
    return question.get("actual_close_time") or question.get("scheduled_close_time")


def question_rows(
    posts_by_status: Mapping[str, Sequence[Mapping[str, Any]]], *, platform: PlatformProbe = METACULUS_PROBE
) -> list[QuestionRow]:
    """Flatten the per-status post pages into one row per (question, status) pairing, each
    carrying the platform's read of whether the bot forecast it."""
    rows: list[QuestionRow] = []
    for status, posts in posts_by_status.items():
        for post in posts:
            title = str(post.get("title") or "")
            for question in questions_on_post(post):
                rows.append(
                    QuestionRow(
                        question_id=question["id"],
                        post_id=post.get("id"),
                        post_status=status,
                        question_type=question.get("type"),
                        title=title,
                        scheduled_resolve_time=question.get("scheduled_resolve_time"),
                        is_resolved=_question_is_resolved(question, status),
                        open_time=question.get("open_time"),
                        close_time=_close_time(question),
                        forecast_state=platform.forecast_state(question),
                    )
                )
    return rows


def _first_per_question_id(rows: Iterable[QuestionRow]) -> list[QuestionRow]:
    """One row per question id, keeping the first — a question can page under two statuses."""
    seen: set[int] = set()
    unique: list[QuestionRow] = []
    for row in rows:
        if row.question_id in seen:
            continue
        seen.add(row.question_id)
        unique.append(row)
    return unique


def _backlog_rows(rows: Sequence[QuestionRow], now: datetime) -> tuple[tuple[BacklogRow, ...], int]:
    """Overdue rows (worst first) plus the count whose schedule was absent or unreadable.

    An unreadable schedule is DISCLOSED rather than imputed: a question with no resolve
    date is not evidence of an on-time question.
    """
    overdue: list[BacklogRow] = []
    without_schedule = 0
    for row in rows:
        scheduled = parse_iso_utc(row.scheduled_resolve_time)
        if scheduled is None:
            without_schedule += 1
            continue
        overdue_days = (now - scheduled).total_seconds() / SECONDS_PER_DAY
        if overdue_days <= 0:
            continue
        overdue.append(
            BacklogRow(
                question_id=row.question_id,
                post_id=row.post_id,
                post_status=row.post_status,
                question_type=row.question_type,
                title=row.title,
                scheduled_resolve_time=str(row.scheduled_resolve_time),
                overdue_days=overdue_days,
            )
        )
    return tuple(sorted(overdue, key=lambda r: -r.overdue_days)), without_schedule


def _window_hours(row: QuestionRow) -> float | None:
    """Length of the forecasting window in hours, or None when either end is unreadable."""
    opened, closed = parse_iso_utc(row.open_time), parse_iso_utc(row.close_time)
    if opened is None or closed is None:
        return None
    return (closed - opened).total_seconds() / SECONDS_PER_HOUR


def _forfeit_eligible(rows: Iterable[QuestionRow]) -> list[QuestionRow]:
    """The closed/resolved questions, one row each.

    Deduped because a post that resolves mid-probe pages under both ``closed`` and ``resolved``
    and would otherwise be counted (and listed) twice. Every copy of a post carries the same
    enrichment, so which one survives changes only the reported status.
    """
    return _first_per_question_id(row for row in rows if row.post_status in FORFEIT_STATUSES)


def _forfeit_rows(eligible: Sequence[QuestionRow]) -> tuple[tuple[ForfeitRow, ...], ForecastStateCounts]:
    """Forfeited questions (newest window first) plus the forecast-state split behind them.

    Newest first because a weekly read is about what we just lost; the window length rides
    each row instead of ordering it, since a short window and a stale one are different
    diagnoses and only one of them is urgent.
    """
    counts = ForecastStateCounts(
        with_forecast=sum(1 for row in eligible if row.forecast_state == FORECAST_PRESENT),
        without_forecast=sum(1 for row in eligible if row.forecast_state == FORECAST_ABSENT),
        unknown=sum(1 for row in eligible if row.forecast_state == FORECAST_UNKNOWN),
    )
    forfeits = [
        ForfeitRow(
            question_id=row.question_id,
            post_id=row.post_id,
            post_status=row.post_status,
            question_type=row.question_type,
            title=row.title,
            open_time=row.open_time,
            close_time=row.close_time,
            window_hours=_window_hours(row),
            is_resolved=row.is_resolved,
        )
        for row in eligible
        if row.forecast_state == FORECAST_ABSENT
    ]
    forfeits.sort(key=lambda row: (row.open_time or "", row.question_id), reverse=True)
    return tuple(forfeits), counts


def _miss_rate(forecast: int, no_forecast: int) -> float | None:
    """Share of DECIDED questions that were forfeited; None when nothing was decided."""
    decided = forecast + no_forecast
    return no_forecast / decided if decided else None


def _release_hour_row(hour_utc: int | None, rows: Sequence[QuestionRow]) -> ReleaseHourRow:
    forecast = sum(1 for row in rows if row.forecast_state == FORECAST_PRESENT)
    no_forecast = sum(1 for row in rows if row.forecast_state == FORECAST_ABSENT)
    return ReleaseHourRow(
        hour_utc=hour_utc,
        questions=len(rows),
        forecast=forecast,
        no_forecast=no_forecast,
        unknown=sum(1 for row in rows if row.forecast_state == FORECAST_UNKNOWN),
        miss_rate=_miss_rate(forecast, no_forecast),
    )


def _release_hour_rows(eligible: Sequence[QuestionRow]) -> tuple[ReleaseHourRow, ...]:
    """The eligible questions bucketed by the UTC hour of day they opened, ascending.

    The hour of ``open_time`` is the release hour: on Mantic's 60-minute windows it is the hour
    the bot had to land a run in, so the miss rate per bucket is the direct read of which cron
    slots GitHub delivers. An unreadable ``open_time`` gets its own None bucket, listed last.
    """
    by_hour: dict[int | None, list[QuestionRow]] = {}
    for row in eligible:
        opened = parse_iso_utc(row.open_time)
        by_hour.setdefault(None if opened is None else opened.hour, []).append(row)
    ordered = sorted(by_hour, key=lambda hour: (hour is None, hour or 0))
    return tuple(_release_hour_row(hour, by_hour[hour]) for hour in ordered)


def _window_minutes(eligible: Sequence[QuestionRow]) -> WindowMinutes | None:
    """Distribution of the realized open-to-close window, over the questions where both ends read."""
    windows = [hours * MINUTES_PER_HOUR for hours in map(_window_hours, eligible) if hours is not None]
    if not windows:
        return None
    return WindowMinutes(
        shortest=min(windows), median=statistics.median(windows), longest=max(windows), questions=len(windows)
    )


def summarize_slug_supply(
    slug: str,
    posts_by_status: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    now: datetime,
    platform: PlatformProbe = METACULUS_PROBE,
) -> SlugSupply:
    """Partition one slug's paged posts by status, then compute its backlog, forfeits and
    per-release-hour miss table.

    Pure: the caller supplies the pages and the clock. Per-status counts report what the
    API returned for that status; the totals count each post and question once, because a
    post that resolves mid-probe can be paged under both ``closed`` and ``resolved``.

    The forfeit sweep reads whatever the supplied payloads carry through the platform's
    ``forecast_state``. A list page fetched with the platform's ``list_params`` answers as it is;
    a Metaculus page read without ``with_cp=true`` carries no ``my_forecasts``, so every eligible
    question comes back ``unknown`` and the forfeit list is empty — call
    :func:`resolve_bot_forecasts` on the pages first (as :func:`probe_slugs` does) to fill the gaps.
    """
    rows = question_rows(posts_by_status, platform=platform)
    resolved_ids = {row.question_id for row in rows if row.is_resolved}
    unresolved = [row for row in _first_per_question_id(rows) if row.question_id not in resolved_ids]
    # A naive `now` from an analysis script must not make the tz-aware overdue subtraction raise.
    backlog, without_schedule = _backlog_rows(unresolved, _as_utc(now))
    eligible = _forfeit_eligible(rows)
    forfeits, forecast_states = _forfeit_rows(eligible)

    return SlugSupply(
        slug=slug,
        status_counts=tuple(
            StatusCount(
                status=status,
                posts=len(posts),
                questions=sum(len(questions_on_post(post)) for post in posts),
            )
            for status, posts in posts_by_status.items()
        ),
        total_posts=len({post.get("id") for posts in posts_by_status.values() for post in posts}),
        total_questions=len({row.question_id for row in rows}),
        backlog=backlog,
        unresolved_without_schedule=without_schedule,
        resolved_within_unresolved_posts=len(
            {row.question_id for row in rows if row.is_resolved and row.post_status != "resolved"}
        ),
        forfeits=forfeits,
        forecast_states=forecast_states,
        by_release_hour=_release_hour_rows(eligible),
        window_minutes=_window_minutes(eligible),
    )


def get_json(params: dict[str, str | int], token: str | None, *, url: str = POSTS_URL) -> dict:
    """GET a posts endpoint with a bounded, 429-aware retry.

    ``url`` defaults to the Metaculus posts LIST; the forfeit sweep passes a post's detail URL, the
    Mantic mode its own list URL and ``scripts/probe_slugs.py`` a project URL through the same retry,
    since every one of those endpoints shares the rate limiter that motivated it. A None ``token``
    sends no ``Authorization`` header (public Mantic read). Public because the slug probe reuses it.

    Local rather than reusing ``performance_analysis.collector``'s helper, which is scoped to the
    scoring pull (three retries, then a ``RuntimeError``): every caller here soft-fails per slug, so
    an exhausted retry has to arrive as a ``requests`` exception.

    The exhausted 429 breaks out and raises the descriptive error below. It used to fall
    through to ``raise_for_status`` on the last attempt, which made that raise unreachable
    and reported six rate-limited attempts as one unlucky request.
    """
    headers = {"Authorization": f"Token {token}"} if token else {}
    for attempt in range(MAX_RETRIES):
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECS)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        if attempt == MAX_RETRIES - 1:
            break
        wait = RETRY_BACKOFF_SECS * (attempt + 1)
        logger.warning(f"Rate limited (429); retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
        time.sleep(wait)
    raise requests.HTTPError(f"429 rate limit: retries exhausted after {MAX_RETRIES} attempts")


def fetch_posts_by_status(
    slug: str, statuses: Sequence[str], token: str | None, *, platform: PlatformProbe = METACULUS_PROBE
) -> dict[str, list[dict]]:
    """Page every requested status for one slug. Raises ``requests.RequestException``.

    Stops on the first short page: the scratch probes this replaces found the Metaculus
    tournament-filtered list serving no usable total, and Mantic advertises ``next`` past its
    last page with ``count`` null, so page length is the only end-of-results signal either
    platform gives. ``MAX_PAGES`` bounds the walk. The platform's ``list_params`` ride every page
    GET, token or not: ``with_cp=true`` is what puts the public snapshot on a Mantic page, and the
    tokenless mode classified nothing while it was sent only under a token.
    """
    posts_by_status: dict[str, list[dict]] = {}
    for status in statuses:
        posts: list[dict] = []
        for page in range(MAX_PAGES):
            params: dict[str, str | int] = {
                "tournaments": slug,
                "statuses": status,
                "limit": PAGE_SIZE,
                "offset": page * PAGE_SIZE,
                **platform.list_params,
            }
            data = get_json(params, token, url=platform.posts_url)
            results = data.get("results") or []
            posts.extend(results)
            if len(results) < PAGE_SIZE:
                break
            time.sleep(REQUEST_SPACING_SECS)
        else:
            logger.warning(f"{slug} statuses={status}: hit MAX_PAGES={MAX_PAGES}; counts are a lower bound")
        logger.info(f"{slug} statuses={status}: {len(posts)} posts")
        posts_by_status[status] = posts
    return posts_by_status


def _posts_needing_detail(posts_by_status: Mapping[str, Sequence[dict]]) -> dict[object, str]:
    """Post ids on a forfeit-eligible status whose questions do not answer ``my_forecasts``.

    Maps id -> the status it was first seen under, purely for the log line. A post whose
    questions all already carry a readable block costs no request. Keys are typed ``object``
    because they come straight off untyped JSON; the None ones are dropped here, which is
    what lets the caller use them as lookup keys without re-checking.
    """
    needed: dict[object, str] = {}
    for status in FORFEIT_STATUSES:
        for post in posts_by_status.get(status) or []:
            post_id = post.get("id")
            if post_id is None or post_id in needed:
                continue
            questions = questions_on_post(post)
            if questions and any(bot_forecast_state(q) == FORECAST_UNKNOWN for q in questions):
                needed[post_id] = status
    return needed


def resolve_bot_forecasts(posts_by_status: dict[str, list[dict]], token: str | None, *, slug: str | None = None) -> int:
    """Fill in ``my_forecasts`` on forfeit-eligible Metaculus posts, in place. Returns fetches issued.

    One detail GET per post that needs one; the fetched payload replaces that post under EVERY
    status it was paged under, so two copies of a post that resolved mid-probe cannot disagree.
    With ``with_cp=true`` on the list GET the page already answers for nearly every post, so this
    is the fallback for the ones left unknown. Metaculus-only (``POSTS_URL``): Mantic's list page
    always answers, so it is never called there.

    ``slug`` only labels the log lines. The sweep can spend minutes issuing spaced GETs, so it
    reports progress every ``DETAIL_PROGRESS_EVERY`` posts (a wedged run used to look exactly like
    a working one), with one DEBUG line per GET for a per-URL trace.

    A post whose detail GET fails stays as it was and reads ``unknown`` rather than forfeited: one
    unreachable post must not cost the slug its census. Raises nothing; every GET failing is a large ``unknown``.
    """
    needed = _posts_needing_detail(posts_by_status)
    if not needed:
        return 0
    total = len(needed)
    label = f"{slug} forfeit sweep" if slug else "forfeit sweep"
    logger.info(f"{label}: fetching my_forecasts detail for {total} post(s)")

    fetched: dict[object, dict] = {}
    for index, (post_id, status) in enumerate(needed.items()):
        logger.debug(f"{label}: detail GET post {post_id} ({status}), {index + 1}/{total}")
        try:
            fetched[post_id] = get_json({}, token, url=f"{POSTS_URL}{post_id}/")
        except requests.RequestException as exc:
            logger.warning(f"{label}: post {post_id} ({status}) detail fetch failed ({exc}); state stays unknown")
        done = index + 1
        if done % DETAIL_PROGRESS_EVERY == 0 and done < total:
            logger.info(f"{label}: {done}/{total} detail GETs done ({len(fetched)} answered)")
        if index < total - 1:
            time.sleep(DETAIL_REQUEST_SPACING_SECS)
    logger.info(f"{label}: {total}/{total} detail GETs done ({len(fetched)} answered)")

    for status, posts in posts_by_status.items():
        posts_by_status[status] = [fetched.get(post.get("id"), post) for post in posts]
    return len(fetched)


def probe_slugs(
    slugs: Sequence[str],
    statuses: Sequence[str],
    token: str | None,
    *,
    now: datetime,
    resolve_forfeits: bool = False,
    platform: PlatformProbe = METACULUS_PROBE,
) -> list[SlugSupply]:
    """Survey every slug, soft-failing per slug so one dead slug reports as an error row.

    Scoped to ``requests.RequestException`` (transport, HTTP status and JSON-decode failures of
    the call). A survey over several slugs expects some to be dead — the bare ``metaculus-cup``
    slug and an unknown Mantic slug both answer 400 — and aborting on the first would hide the
    live ones. Anything that is not a request failure is a contract break and crashes.

    ``resolve_forfeits`` costs one detail GET per closed/resolved post the list page did not
    already answer for, which with ``with_cp=true`` on every list GET is close to none. It
    defaults OFF so a caller that only wants counts pays nothing; the CLI turns it ON
    (``--no-forfeits`` to opt out), because a forfeit is what the weekly read exists to catch.
    Moot where the list page always answers (Mantic): no detail GET is issued there whatever the
    flag says.
    """
    supplies: list[SlugSupply] = []
    for slug in slugs:
        try:
            posts_by_status = fetch_posts_by_status(slug, statuses, token, platform=platform)
            if resolve_forfeits and platform.sweep_needs_detail_gets:
                resolve_bot_forecasts(posts_by_status, token, slug=slug)
        except requests.RequestException as exc:
            logger.warning(f"{slug}: supply probe failed ({exc})")
            supplies.append(SlugSupply(slug=slug, error=str(exc)))
            continue
        supplies.append(summarize_slug_supply(slug, posts_by_status, now=now, platform=platform))
    return supplies


def _render_backlog(supply: SlugSupply, max_rows: int) -> list[str]:
    lines: list[str] = []
    if not supply.backlog:
        lines.append("  Unresolved past scheduled_resolve_time: 0")
    else:
        lines.append(
            f"  Unresolved past scheduled_resolve_time: {len(supply.backlog)} "
            f"(worst {supply.worst_overdue_days:.1f} days overdue)"
        )
        lines.append(f"    {'qid':>8} {'post':>8} {'status':<9} {'overdue_d':>9}  {'scheduled':<17} title")
        for row in supply.backlog[:max_rows]:
            lines.append(
                f"    {row.question_id:>8} {row.post_id!s:>8} {row.post_status:<9} {row.overdue_days:>9.1f}  "
                f"{row.scheduled_resolve_time[:16]:<17} {row.title[:60]}"
            )
        hidden = max(0, len(supply.backlog) - max_rows)
        if hidden:
            lines.append(f"    +{hidden} more overdue (raise --max-backlog-rows to see them)")
    if supply.unresolved_without_schedule:
        lines.append(f"  Unresolved with no readable scheduled_resolve_time: {supply.unresolved_without_schedule}")
    if supply.resolved_within_unresolved_posts:
        lines.append(
            f"  Questions already resolved inside non-resolved posts: {supply.resolved_within_unresolved_posts}"
        )
    return lines


def _render_forfeits(supply: SlugSupply, max_rows: int, platform: PlatformProbe) -> list[str]:
    """The forfeit block: what the bot never forecast, and the state split behind the count."""
    states = supply.forecast_states
    if states.total == 0:
        return []
    lines = [
        f"  Closed/resolved questions never forecast by the bot: {len(supply.forfeits)} "
        f"(of {states.total}; forecast {states.with_forecast}, unknown {states.unknown})"
    ]
    if states.unknown == states.total:
        lines.append(f"    {platform.all_unknown_hint}")
        return lines
    if states.with_forecast == 0 and states.without_forecast:
        lines.append(
            f"    !!! no question on this slug carries a bot forecast. {platform.identity_hint} "
            "before reading these as forfeits."
        )
    if not supply.forfeits:
        return lines
    lines.append(f"    {'qid':>8} {'post':>8} {'status':<9} {'window_h':>8}  {'opened':<17} title")
    for row in supply.forfeits[:max_rows]:
        window = f"{row.window_hours:.1f}" if row.window_hours is not None else "n/a"
        opened = (row.open_time or "unknown")[:16]
        lines.append(
            f"    {row.question_id:>8} {row.post_id!s:>8} {row.post_status:<9} {window:>8}  "
            f"{opened:<17} {row.title[:60]}"
        )
    hidden = max(0, len(supply.forfeits) - max_rows)
    if hidden:
        lines.append(f"    +{hidden} more forfeited (raise --max-forfeit-rows to see them)")
    return lines


_RELEASE_HOUR_HEADER = (
    f"    {'hour_utc':>8} {'questions':>9} {'forecast':>8} {'no_forecast':>11} {'unknown':>7} {'miss_rate':>9}"
)


def _release_hour_line(row: ReleaseHourRow, label: str) -> str:
    miss_rate = f"{row.miss_rate:.1%}" if row.miss_rate is not None else "-"
    return f"    {label:>8} {row.questions:>9} {row.forecast:>8} {row.no_forecast:>11} {row.unknown:>7} {miss_rate:>9}"


def _render_release_hours(supply: SlugSupply) -> list[str]:
    """The cadence instrument: per UTC release hour, how many eligible questions we lost.

    The total row is built from ``forecast_states`` rather than by re-summing the rows: both
    cover the same eligible set, so the two blocks of the report cannot disagree.
    """
    if not supply.by_release_hour:
        return []
    states = supply.forecast_states
    total = ReleaseHourRow(
        hour_utc=None,
        questions=states.total,
        forecast=states.with_forecast,
        no_forecast=states.without_forecast,
        unknown=states.unknown,
        miss_rate=_miss_rate(states.with_forecast, states.without_forecast),
    )
    lines = [
        "  Miss rate by UTC release hour (closed/resolved questions; miss_rate = no_forecast / decided):",
        _RELEASE_HOUR_HEADER,
        *(
            _release_hour_line(row, "n/a" if row.hour_utc is None else str(row.hour_utc))
            for row in supply.by_release_hour
        ),
        _release_hour_line(total, "total"),
    ]
    window = supply.window_minutes
    if window is not None:
        lines.append(
            f"    window open-to-close, minutes: min {window.shortest:.0f} / median {window.median:.0f} / "
            f"max {window.longest:.0f} (over {window.questions} questions)"
        )
    return lines


def render_report(
    supplies: Sequence[SlugSupply],
    *,
    now: datetime,
    max_backlog_rows: int = DEFAULT_MAX_BACKLOG_ROWS,
    max_forfeit_rows: int = DEFAULT_MAX_FORFEIT_ROWS,
    platform: PlatformProbe = METACULUS_PROBE,
) -> str:
    """Render the survey as text. Pure — no clock read, no IO."""
    lines = [
        f"{platform.name.capitalize()} question-supply probe. "
        "Post status `closed` means closed to forecasting but NOT yet resolved.",
        platform.classification_note,
        f"as of {now.isoformat()}",
    ]
    for supply in supplies:
        lines.append("")
        lines.append(f"=== {supply.slug} ===")
        if supply.error is not None:
            lines.append(f"  ERROR: {supply.error} (no counts for this slug)")
            continue
        lines.append(f"  {'status':<12}{'posts':>8}{'questions':>11}")
        for count in supply.status_counts:
            lines.append(f"  {count.status:<12}{count.posts:>8}{count.questions:>11}")
        lines.append(f"  {'total':<12}{supply.total_posts:>8}{supply.total_questions:>11}")
        lines.extend(_render_backlog(supply, max_backlog_rows))
        lines.extend(_render_forfeits(supply, max_forfeit_rows, platform))
        lines.extend(_render_release_hours(supply))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Report question supply per tournament slug, counting `closed` posts, on Metaculus or Mantic."
    )
    parser.add_argument(
        "--platform",
        choices=sorted(PLATFORM_PROBES),
        default=PLATFORM_METACULUS,
        help=(
            "Question platform to probe (default: %(default)s). Mantic's read endpoints are public, so "
            f"{MANTIC_TOKEN_ENV} is optional there and only lets closed-but-unresolved questions classify."
        ),
    )
    parser.add_argument(
        "--slugs",
        nargs="+",
        default=None,
        help=(
            f"Tournament slugs to probe (default: the platform's season slugs, {' '.join(DEFAULT_SLUGS)} on "
            f"Metaculus and {MANTIC_TOURNAMENT_ID} on Mantic)"
        ),
    )
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=list(DEFAULT_STATUSES),
        help=f"Post statuses to count (default: {' '.join(DEFAULT_STATUSES)})",
    )
    parser.add_argument(
        "--max-backlog-rows",
        type=int,
        default=DEFAULT_MAX_BACKLOG_ROWS,
        help="Overdue questions listed per slug (default: %(default)s)",
    )
    parser.add_argument(
        "--max-forfeit-rows",
        type=int,
        default=DEFAULT_MAX_FORFEIT_ROWS,
        help="Never-forecast questions listed per slug (default: %(default)s)",
    )
    parser.add_argument(
        "--no-forfeits",
        dest="forfeits",
        action="store_false",
        help=(
            "Skip the Metaculus forfeit sweep's detail GETs (one read-only GET per closed/resolved post whose "
            "list page did not already carry my_forecasts), leaving those questions reported as unknown. "
            "Mantic classifies off the list page and never issues them."
        ),
    )
    parser.add_argument("--output", default=None, help="Optional path to dump the census as JSON.")
    args = parser.parse_args(argv)
    platform = PLATFORM_PROBES[args.platform]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    load_environment()
    token = os.environ.get(platform.token_env) or None
    if platform.token_required and not token:
        parser.error(f"{platform.token_env} is not set (put it in .env or the environment)")
    if token is None:
        logger.info(
            f"{platform.token_env} is not set: running public-only, so closed-but-unresolved questions read unknown"
        )

    # The host is vetted before any token goes out (DNS-parking incident; see metaculus_bot/api_preflight.py).
    verify_api_identity(platform.base_url)

    now = datetime.now(UTC)
    slugs = args.slugs or list(platform.default_slugs)
    supplies = probe_slugs(
        slugs, tuple(args.statuses), token, now=now, resolve_forfeits=args.forfeits, platform=platform
    )
    print(
        render_report(
            supplies,
            now=now,
            max_backlog_rows=args.max_backlog_rows,
            max_forfeit_rows=args.max_forfeit_rows,
            platform=platform,
        )
    )

    if args.output:
        payload = {
            "generated_at": now.isoformat(),
            "platform": platform.name,
            "bot_user_id": platform.bot_user_id,
            "slugs": [asdict(supply) for supply in supplies],
        }
        Path(args.output).write_text(json.dumps(payload, indent=2))
        logger.info(f"Wrote supply census to {args.output}")


if __name__ == "__main__":
    main()
