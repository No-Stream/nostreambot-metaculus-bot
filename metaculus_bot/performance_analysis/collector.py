"""Data collection from the Metaculus API.

Fetches resolved questions and bot comments, matches them, parses per-model
predictions, computes scores, and returns a structured dataset.
"""

import json
import logging
import os
import re
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from metaculus_bot.config import load_environment
from metaculus_bot.constants import (
    MANTIC_API_BASE_URL,
    MANTIC_BOT_USER_ID,
    MANTIC_TOKEN_ENV,
    METACULUS_TOKEN_ENV,
    PLATFORM_MANTIC,
    PLATFORM_METACULUS,
)
from metaculus_bot.performance_analysis.parsing import (
    parse_explanation_detail_status,
    parse_forecasters_used_marker,
    parse_inferred_stacker_outcome,
    parse_per_base_model_forecasts,
    parse_per_model_forecasts,
    parse_per_model_mc_option_probs,
    parse_per_model_numeric_percentiles,
    parse_resolution,
    parse_stacked_marker,
    parse_stacker_skip_reason_marker,
)
from metaculus_bot.performance_analysis.rescore_diff import RESCORE_ATOL, diff_platform_rescores
from metaculus_bot.performance_analysis.research_tags import DEFAULT_RESEARCH_ARCHIVE_LATEST, attach_research_tags
from metaculus_bot.performance_analysis.scaling import grid_zero_point
from metaculus_bot.performance_analysis.scoring import binary_log_score, brier_score, mc_log_score, numeric_log_score

logger: logging.Logger = logging.getLogger(__name__)

load_environment()

BASE_URL = "https://www.metaculus.com/api"
DEFAULT_BOT_USER_ID = 275109
DEFAULT_TOURNAMENT = "spring-aib-2026"


@dataclass(frozen=True, slots=True)
class CollectorPlatform:
    """One question platform the collector can pull: its API root, the bot's account, the token variable.

    Mantic's Crucible is a Metaculus fork with the same posts and comments API, so a Mantic pull runs
    the same functions against a different root (docs/performance_analysis.md "Mantic records").
    """

    name: str
    api_base_url: str
    bot_user_id: int
    token_env: str


METACULUS_PLATFORM = CollectorPlatform(PLATFORM_METACULUS, BASE_URL, DEFAULT_BOT_USER_ID, METACULUS_TOKEN_ENV)
MANTIC_PLATFORM = CollectorPlatform(PLATFORM_MANTIC, MANTIC_API_BASE_URL, MANTIC_BOT_USER_ID, MANTIC_TOKEN_ENV)
COLLECTOR_PLATFORMS: dict[str, CollectorPlatform] = {
    platform.name: platform for platform in (METACULUS_PLATFORM, MANTIC_PLATFORM)
}

# Minimum gap between any two API requests this process sends, to either platform. Cloudflare's per-IP
# limiter answers a burst with 429s that outlast the 2s Retry-After it sends: on 2026-11-19
# back-to-back comment expansions drew two 429s each (5s and 10s backoffs) before succeeding,
# while a comment listing paced at 2s drew none over 17 pages. Paced inside api_get so every
# caller (page loops, comment expansion, the backfill script) shares one budget.
MIN_REQUEST_INTERVAL_SECS: float = 2.0
MAX_RETRIES: int = 3
RETRY_BACKOFF_SECS: float = 5.0
REQUEST_TIMEOUT_SECS: float = 30.0
PAGE_SIZE: int = 100
_TRUNCATED_COMMENT_MARKER = "### WARNING: Content Truncated"
_TRUNCATED_COMMENT_STUB_RE = re.compile(
    r"(?s).*\n### WARNING: Content Truncated\nThis comment has been archived to save space\.\n"
    r"To retrieve full content, please visit \[https://www\.metaculus\.com/api/comments/(?P<comment_id>\d+)/\]"
    r"(?:\(https://www\.metaculus\.com/api/comments/(?P<link_comment_id>\d+)/\))?$"
)

# Retried; anything else propagates (docs/performance_analysis.md "The round pull").
_TRANSIENT_NETWORK_ERRORS = (requests.exceptions.Timeout, requests.exceptions.ConnectionError)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _make_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Token {token}"}


@dataclass
class _RequestPacer:
    """When this process last started an API request, so the next one can keep its distance."""

    last_started_at: float | None = None

    def wait_for_slot(self) -> None:
        """Sleep until ``MIN_REQUEST_INTERVAL_SECS`` has passed since the previous request started."""
        if self.last_started_at is not None:
            remaining = MIN_REQUEST_INTERVAL_SECS - (time.monotonic() - self.last_started_at)
            if remaining > 0:
                time.sleep(remaining)
        self.last_started_at = time.monotonic()


_REQUEST_PACER = _RequestPacer()


def api_get(path: str, token: str, params: dict | None = None, *, base_url: str = BASE_URL) -> dict:
    """GET one page, retrying a 429 or a transient network failure within one shared budget.

    Every call first waits for its slot under ``MIN_REQUEST_INTERVAL_SECS``; the retries inside
    one call need no extra wait because each backoff is already longer than the interval.

    Both causes draw on the same ``MAX_RETRIES`` attempts, so the worst-case wall clock is
    unchanged from the 429-only retry this replaced: ``MAX_RETRIES`` reads at
    ``REQUEST_TIMEOUT_SECS`` plus the backoffs between them. Exhaustion raises rather than
    returning a partial page, because a sweep that reports a slow success is worse than one
    that stops. Receipts: docs/performance_analysis.md "The round pull".
    """
    url = f"{base_url}{path}"
    headers = _make_headers(token)
    _REQUEST_PACER.wait_for_slot()
    for attempt in range(MAX_RETRIES):
        last_attempt = attempt == MAX_RETRIES - 1
        wait = RETRY_BACKOFF_SECS * (attempt + 1)
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECS)
        except _TRANSIENT_NETWORK_ERRORS as exc:
            if last_attempt:
                raise
            logger.warning(
                f"{type(exc).__name__} on {path}, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES}): {exc}"
            )
            time.sleep(wait)
            continue
        if resp.status_code == 429:
            if last_attempt:
                break
            logger.warning(f"Rate limited (429), retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    # Reached only by an exhausted 429, which raise_for_status reported as one unlucky request.
    raise requests.exceptions.HTTPError(f"429 rate limit on {path}: retries exhausted after {MAX_RETRIES} attempts")


# ---------------------------------------------------------------------------
# Fetch functions
# ---------------------------------------------------------------------------


def fetch_resolved_questions(tournament: str, token: str, *, base_url: str = BASE_URL) -> list[dict]:
    """Every resolved post in a tournament, as the posts list serves them.

    ``with_cp=true`` puts the token's own ``my_forecasts`` (forecast values and score data) on
    the list page next to the resolution, scaling and bounds the records read, for a single
    question and for a group post's members alike, so no per-post GET is needed.
    """
    posts: list[dict] = []
    offset = 0
    while True:
        data = api_get(
            "/posts/",
            token,
            params={
                "tournaments": tournament,
                "statuses": "resolved",
                "with_cp": "true",
                "limit": PAGE_SIZE,
                "offset": offset,
            },
            base_url=base_url,
        )
        results = data.get("results", [])
        posts.extend(results)
        logger.info(f"Fetched post listing page: {offset=}, got {len(results)} posts")
        if not results or data.get("next") is None:
            break
        offset += PAGE_SIZE
    logger.info(f"Found {len(posts)} resolved posts in tournament '{tournament}'")
    return posts


def _page_comments(token: str, params: dict[str, object], *, base_url: str = BASE_URL) -> list[dict]:
    """Page one ``/comments/`` listing to exhaustion, on top of the caller's params.

    Mantic sets ``next`` on every page, the last one included, so the empty page is what ends a
    Mantic listing (one extra GET); Metaculus ends on a null ``next``.
    """
    comments: list[dict] = []
    offset = 0
    while True:
        data = api_get("/comments/", token, params={**params, "limit": PAGE_SIZE, "offset": offset}, base_url=base_url)
        results = data.get("results", [])
        comments.extend(results)
        logger.info(f"Fetched comments page: {offset=}, got {len(results)} comments, {params=}")
        if not results or data.get("next") is None:
            break
        offset += PAGE_SIZE
    return comments


def fetch_bot_comments(author_id: int, token: str, *, base_url: str = BASE_URL) -> list[dict]:
    """Every comment by an author, public and private, deduplicated by comment id.

    The bot POSTs comments with ``is_private: true`` and Metaculus flips older ones public
    server-side, so the default author listing serves only the flipped ones and the recent
    private comments come back solely under ``is_private=true`` (verified live 2026-09-09,
    when the six fall comments were missing from a 1,054-comment public pull). Returns raw
    comment dicts, public listing order first.
    """
    public = _page_comments(token, {"author": author_id}, base_url=base_url)
    private = _page_comments(token, {"author": author_id, "is_private": "true"}, base_url=base_url)

    merged: dict[int, dict] = {}
    for comment in (*public, *private):
        merged[comment["id"]] = comment

    n_public, n_private, n_unique = len(public), len(private), len(merged)
    logger.info(f"Fetched author comments: {n_public=} {n_private=} {n_unique=}")
    return list(merged.values())


# ---------------------------------------------------------------------------
# Comment lookup
# ---------------------------------------------------------------------------


def expand_truncated_comment(comment: dict, token: str, *, base_url: str = BASE_URL) -> dict:
    """Replace a Metaculus archived comment stub with its per-comment API payload.

    The stub pattern names www.metaculus.com on purpose: Mantic served no stub in 218 comments,
    and a Mantic stub would fail the match and stop the pull with the ValueError below.
    """
    comment_text = comment.get("text") or comment.get("comment_text") or ""
    if _TRUNCATED_COMMENT_MARKER not in comment_text:
        return comment

    match = _TRUNCATED_COMMENT_STUB_RE.fullmatch(comment_text)
    comment_id = comment["id"]
    linked_comment_id = match["link_comment_id"] if match is not None else None
    if (
        match is None
        or int(match["comment_id"]) != comment_id
        or (linked_comment_id is not None and int(linked_comment_id) != comment_id)
    ):
        raise ValueError(f"comment {comment_id} has a malformed truncation stub")

    full_comment = api_get(f"/comments/{comment_id}/", token, base_url=base_url)
    full_text = full_comment.get("text")
    if full_comment.get("id") != comment_id:
        raise ValueError(f"full comment response for {comment_id} has a mismatched id")
    if not isinstance(full_text, str) or not full_text.strip():
        raise ValueError(f"full comment {comment_id} has no text")
    return {**comment, "text": full_text}


def _build_comment_lookup(
    comments: list[dict],
    token: str,
    *,
    base_url: str = BASE_URL,
    post_ids: Collection[int] | None = None,
) -> dict[int, dict]:
    """Build post_id -> comment mapping, expanding the most recent comment when archived.

    With ``post_ids`` only those posts are kept and expanded. The author listing covers every post
    the bot ever commented on, and each archived stub costs one paced GET, so a pull of one
    tournament should not expand stubs for the other thousand posts.
    """
    lookup: dict[int, dict] = {}
    for c in comments:
        post_id = c.get("on_post")
        if post_id is None or (post_ids is not None and post_id not in post_ids):
            continue
        existing = lookup.get(post_id)
        if existing is None or c["id"] > existing["id"]:
            lookup[post_id] = c
    return {post_id: expand_truncated_comment(comment, token, base_url=base_url) for post_id, comment in lookup.items()}


# ---------------------------------------------------------------------------
# Record processing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CommentSignals:
    """Everything the bot's own comment tells us about one post's forecast.

    Bundled so the post-level and question-level processors pass one value rather than
    a dozen parallel parameters; every field is None/empty when the post has no comment.
    """

    text: str | None
    comment_id: int | None
    created_at: str | None
    per_model: dict[str, str] | dict[str, dict[str, float]]
    per_model_numeric_percentiles: dict[str, list[tuple[float, float]]]
    was_stacked: bool | None
    stacker_outcome: str | None
    stacker_outcome_source: str
    stacker_skip_reason: str | None
    forecasters_used: tuple[int, int] | None


def _comment_signals(comment: dict | None, post_id: int) -> _CommentSignals:
    """Parse every per-comment marker and per-model recovery off one bot comment."""
    comment_text = comment.get("text") or comment.get("comment_text") if comment else None

    per_model_numeric_percentiles = parse_per_model_numeric_percentiles(comment_text) if comment_text else {}
    per_model_mc_option_probs = parse_per_model_mc_option_probs(comment_text) if comment_text else {}
    # A non-empty option dict is the multiple-choice detector (docs/performance_analysis.md "Record fields").
    per_model = per_model_mc_option_probs or (parse_per_model_forecasts(comment_text) if comment_text else {})
    was_stacked = parse_stacked_marker(comment_text) if comment_text else None
    # Tri-state, read off a three-rung marker fallback (docs/performance_analysis.md "Record fields").
    if comment_text:
        stacker_outcome, stacker_outcome_source = parse_inferred_stacker_outcome(comment_text)
    else:
        stacker_outcome, stacker_outcome_source = None, "none"

    # DEBUG, not WARNING: a drifted delimiter and the legitimate shapes look identical here.
    if was_stacked is True and not per_model_numeric_percentiles and not per_model:
        logger.debug(
            f"Stacked comment yielded no per-model entries: post_id={post_id}, "
            f"comment_len={len(comment_text) if comment_text else 0}"
        )

    return _CommentSignals(
        text=comment_text,
        comment_id=comment["id"] if comment else None,
        created_at=comment.get("created_at") if comment else None,
        per_model=per_model,
        per_model_numeric_percentiles=per_model_numeric_percentiles,
        was_stacked=was_stacked,
        stacker_outcome=stacker_outcome,
        stacker_outcome_source=stacker_outcome_source,
        # A bare "skipped" outcome cannot tell a below-threshold skip from the single-forecaster one.
        stacker_skip_reason=parse_stacker_skip_reason_marker(comment_text) if comment_text else None,
        # (n_used, n_configured): what tells a degraded publish from a genuine roster change.
        forecasters_used=parse_forecasters_used_marker(comment_text) if comment_text else None,
    )


def questions_on_post(post_data: Mapping[str, Any]) -> list[dict]:
    """The post's question dicts — a group's members, or its single question.

    Empty for a post carrying neither (tournaments hold notebook posts too). Public
    because every consumer of the Metaculus posts list needs this same unwrapping and
    `scripts/supply_probe.py` had grown its own copy; one shared reading keeps a probe's
    question counts comparable with the scoring pull's.
    """
    group = post_data.get("group_of_questions")
    if group is not None:
        return list(group.get("questions") or [])
    question = post_data.get("question")
    return [question] if isinstance(question, dict) else []


def _process_post(post_data: dict, comment_lookup: dict[int, dict]) -> list[dict]:
    """Process a single post into one or more question records."""
    post_id = post_data["id"]
    title = post_data.get("title", "")

    if title.startswith("[PRACTICE]"):
        title_preview = title[:60]  # HARNESS-SCAN-EXEMPT-subsampling  # log display truncation
        logger.info(f"  Skipping PRACTICE post {post_id}: {title_preview}")
        return []

    questions = questions_on_post(post_data)
    if not questions:
        logger.warning(f"  Post {post_id} has no question data")
        return []

    signals = _comment_signals(comment_lookup.get(post_id), post_id)

    records: list[dict] = []
    for q in questions:
        # A stacked comment discloses base values only in its R1 body (docs/performance_analysis.md "Record fields").
        q_type_for_base = q.get("type", "") if q else ""
        per_base_model_forecasts = parse_per_base_model_forecasts(signals.text, q_type_for_base) if signals.text else {}
        record = _process_single_question(
            post_id,
            title,
            q,
            comment_text=signals.text,
            comment_id=signals.comment_id,
            per_model=signals.per_model,
            per_model_numeric_percentiles=signals.per_model_numeric_percentiles,
            was_stacked=signals.was_stacked,
            post_data=post_data,
            comment_created_at=signals.created_at,
            stacker_outcome=signals.stacker_outcome,
            stacker_outcome_source=signals.stacker_outcome_source,
            stacker_skip_reason=signals.stacker_skip_reason,
            per_base_model_forecasts=per_base_model_forecasts,
            forecasters_used=signals.forecasters_used,
        )
        if record is not None:
            records.append(record)
    return records


def _our_forecast(q: dict, q_type: str) -> tuple[list[float] | None, float | None, dict | None]:
    """``(forecast_values, prob_yes, metaculus_scores)`` from the question's own record.

    ``forecast_values is None`` means the bot never forecast this question, which is the
    caller's signal to drop the record.
    """
    my_forecasts = q.get("my_forecasts")
    if not (my_forecasts and my_forecasts.get("latest")):
        return None, None, None

    forecast_values = my_forecasts["latest"].get("forecast_values")
    prob_yes = None
    if q_type == "binary" and forecast_values and len(forecast_values) >= 2:
        prob_yes = forecast_values[1]

    raw_sd = my_forecasts.get("score_data")
    metaculus_scores = (
        {
            "peer_score": raw_sd.get("peer_score"),
            "spot_peer_score": raw_sd.get("spot_peer_score"),
            "baseline_score": raw_sd.get("baseline_score"),
            "spot_baseline_score": raw_sd.get("spot_baseline_score"),
            "coverage": raw_sd.get("coverage"),
            "weighted_coverage": raw_sd.get("weighted_coverage"),
            "relative_legacy_score": raw_sd.get("relative_legacy_score"),
        }
        if raw_sd
        else None
    )
    return forecast_values, prob_yes, metaculus_scores


def _post_category(post_data: dict) -> str | None:
    """The post's first project category name, if it carries one."""
    category_list = post_data.get("projects", {}).get("category", [])
    if category_list and isinstance(category_list, list) and len(category_list) > 0:
        return category_list[0].get("name")
    return None


def _process_single_question(
    post_id: int,
    title: str,
    q: dict,
    *,
    comment_text: str | None,
    comment_id: int | None,
    per_model: dict[str, str] | dict[str, dict[str, float]],
    per_model_numeric_percentiles: dict[str, list[tuple[float, float]]],
    was_stacked: bool | None,
    post_data: dict,
    comment_created_at: str | None = None,
    stacker_outcome: str | None = None,
    stacker_outcome_source: str = "none",
    stacker_skip_reason: str | None = None,
    per_base_model_forecasts: dict[str, str | dict[str, float]] | None = None,
    forecasters_used: tuple[int, int] | None = None,
) -> dict | None:
    """Process a single question dict into a scored record."""
    question_id = q.get("id")
    q_type = q.get("type", "")
    if q_type == "date":
        # Excluded by decision, not a parse failure (docs/performance_analysis.md "Date questions").
        logger.warning(
            f"  Skipping Q{question_id} (post {post_id}): date question, excluded from residual analysis by decision"
        )
        return None
    if q.get("multi_resolution") is True:
        # Mantic only, and none has resolved yet: how one reports its resolution is unread, so
        # single-resolution scoring could be silently wrong (docs/performance_analysis.md "Mantic records").
        logger.warning(
            f"  Skipping Q{question_id} (post {post_id}): multi-resolution question, scoring not yet verified"
        )
        return None
    resolution_raw = q.get("resolution")

    if resolution_raw is None:
        logger.info(f"  Skipping Q{question_id} (no resolution)")
        return None

    resolution_parsed, should_skip = parse_resolution(str(resolution_raw), q_type)
    if should_skip:
        logger.info(f"  Skipping Q{question_id}: resolution={resolution_raw}")
        return None

    forecast_values, prob_yes, metaculus_scores = _our_forecast(q, q_type)
    if forecast_values is None:
        logger.info(f"  Skipping Q{question_id}: no forecast from us")
        return None

    scaling = q.get("scaling") or {}
    open_lower = q.get("open_lower_bound", False)
    open_upper = q.get("open_upper_bound", False)
    options = q.get("options")
    category = _post_category(post_data)
    q_title = q.get("title") or title

    # Field meanings, provenance and the traps behind them: docs/performance_analysis.md "Record fields".
    record = {
        "post_id": post_id,
        "question_id": question_id,
        "title": q_title,
        "type": q_type,
        "resolution_raw": str(resolution_raw),
        "resolution_parsed": resolution_parsed,
        "our_forecast_values": forecast_values,
        "our_prob_yes": prob_yes,
        "per_model_forecasts": per_model,
        # On a stacked record per_model_forecasts holds only the aggregate, so spread cuts read this.
        "per_base_model_forecasts": per_base_model_forecasts or {},
        # Numeric and discrete only; empty for binary and multiple choice.
        "per_model_numeric_percentiles": per_model_numeric_percentiles,
        # Legacy tri-state, kept for back-compat; prefer stacker_outcome below for new analyses.
        "was_stacked": was_stacked,
        # Separates a median fallback from a skip, which was_stacked collapses into one False.
        "stacker_outcome": stacker_outcome,
        "stacker_outcome_source": stacker_outcome_source,
        # None whenever the stacker did not skip, or the comment predates the marker.
        "stacker_skip_reason": stacker_skip_reason,
        # The BOT ensemble size, not metadata.nr_forecasters, which is the Metaculus CROWD count.
        "forecasters_used": forecasters_used[0] if forecasters_used is not None else None,
        "forecasters_configured": forecasters_used[1] if forecasters_used is not None else None,
        "scaling": scaling,
        "open_lower_bound": open_lower,
        "open_upper_bound": open_upper,
        "options": options,
        "comment_text": comment_text,
        # Explicitly unknown when the public comment's bounded fallback omitted rationale detail.
        "explanation_detail_status": parse_explanation_detail_status(comment_text),
        "comment_id": comment_id,
        # SUBMIT date, so a cohort cut need not key on the coarser actual_resolve_time.
        "bot_comment_created_at": comment_created_at,
        # Read spot peer, not peer, and read it through platform_scores.py rather than this dict.
        "metaculus_scores": metaculus_scores,
        "metadata": {
            # Lives on the POST, not the question; 0 on a pre-2026-08-25 pull means unknown, not empty.
            "nr_forecasters": post_data.get("nr_forecasters"),
            "open_time": q.get("open_time"),
            "actual_resolve_time": q.get("actual_resolve_time"),
            "scheduled_resolve_time": q.get("scheduled_resolve_time"),
            # Never the re-resolution detector: it can PRECEDE the pull that still read the old value.
            "resolution_set_time": q.get("resolution_set_time"),
            "category": category,
        },
        "brier_score": None,
        "log_score": None,
        "numeric_log_score": None,
        "mc_log_score": None,
    }

    _compute_scores(record)
    return record


def resolve_numeric_record_to_score_inputs(
    record: dict,
) -> tuple[float, float, float, float | None] | None:
    """Coerce a numeric/discrete record to (res_float, lower, upper, zero_point).

    Returns None when the record can't be scored (missing bounds,
    unrecognized resolution). ``above_upper_bound`` / ``below_lower_bound``
    are coerced to ``upper + 1.0`` / ``lower - 1.0`` to feed
    ``numeric_log_score``'s out-of-bounds branch. ``zero_point`` is
    interpreted via :func:`grid_zero_point` (a serialized ``0`` stays log when
    ``range_min`` is positive, and only genuinely-absent/non-positive-floor
    cases collapse to the linear ``None`` sentinel).

    Shared by ``_compute_scores`` (record-level scoring) and
    ``audit._rank_numeric`` (per-model scoring) so both paths follow the
    same coercion rules.
    """
    scaling = record.get("scaling") or {}
    lower_raw = scaling.get("range_min")
    upper_raw = scaling.get("range_max")
    if lower_raw is None or upper_raw is None:
        return None

    lower_bound = float(lower_raw)
    upper_bound = float(upper_raw)

    zero_point = grid_zero_point(scaling.get("zero_point"), lower_bound)

    resolution = record.get("resolution_parsed")
    if resolution == "above_upper_bound":
        res_float = upper_bound + 1.0
    elif resolution == "below_lower_bound":
        res_float = lower_bound - 1.0
    elif isinstance(resolution, (int, float)) and not isinstance(resolution, bool):
        res_float = float(resolution)
    else:
        return None

    return res_float, lower_bound, upper_bound, zero_point


def _compute_scores(record: dict) -> None:
    """Compute and set score fields on a record dict in place."""
    q_type = record["type"]
    resolution = record["resolution_parsed"]
    forecast_values = record["our_forecast_values"]

    if forecast_values is None:
        return

    if q_type == "binary" and isinstance(resolution, bool):
        prob_yes = record["our_prob_yes"]
        if prob_yes is not None:
            record["brier_score"] = brier_score(prob_yes, resolution)
            record["log_score"] = binary_log_score(prob_yes, resolution)

    elif q_type in ("numeric", "discrete"):
        score_inputs = resolve_numeric_record_to_score_inputs(record)
        if score_inputs is None:
            return
        res_float, lower_bound, upper_bound, zero_point = score_inputs

        try:
            record["numeric_log_score"] = numeric_log_score(
                forecast_values,
                res_float,
                lower_bound,
                upper_bound,
                open_lower_bound=record["open_lower_bound"],
                open_upper_bound=record["open_upper_bound"],
                zero_point=zero_point,
            )
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Failed numeric scoring for post {record.get('post_id')}: {e}")

    elif q_type == "multiple_choice" and isinstance(resolution, str):
        options = record.get("options") or []
        if resolution in options and forecast_values:
            try:
                correct_idx = options.index(resolution)
                record["mc_log_score"] = mc_log_score(forecast_values, correct_idx)
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed MC scoring for post {record.get('post_id')}: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def records_from_posts(
    posts: list[dict],
    comments: list[dict],
    token: str,
    *,
    platform: CollectorPlatform = METACULUS_PLATFORM,
    research_archive_dir: str | Path = DEFAULT_RESEARCH_ARCHIVE_LATEST,
    prior_records: Sequence[dict] | None = None,
    comment_post_ids: Collection[int] | None = None,
) -> list[dict]:
    """Turn fetched posts and the author's comments into scored, tagged records.

    The half of :func:`build_performance_dataset` after the two fetches, public so a caller that
    pulls several tournaments can fetch the comment listing once. ``comment_post_ids`` limits
    which posts get their comment looked up (and archived stubs expanded); None means every post
    in ``posts``. A post outside the set still becomes a record, with no comment fields.
    """
    lookup_post_ids = {post["id"] for post in posts} if comment_post_ids is None else set(comment_post_ids)
    comment_lookup = _build_comment_lookup(comments, token, base_url=platform.api_base_url, post_ids=lookup_post_ids)
    logger.info(f"Comments mapped to {len(comment_lookup)} unique posts")

    records = [record for post_data in posts for record in _process_post(post_data, comment_lookup)]
    for record in records:
        record["platform"] = platform.name

    attach_research_tags(records, research_archive_dir)
    if prior_records is not None:
        diff = diff_platform_rescores(prior_records, records)
        logger.info(f"Diffed against prior dataset: {diff.rescored} record(s) re-resolved or re-scored")

    logger.info(f"Collected {len(records)} question records")
    return records


def build_performance_dataset(
    tournament: str = DEFAULT_TOURNAMENT,
    token: str | None = None,
    author_id: int | None = None,
    research_archive_dir: str | Path = DEFAULT_RESEARCH_ARCHIVE_LATEST,
    *,
    prior_records: Sequence[dict] | None = None,
    platform: CollectorPlatform = METACULUS_PLATFORM,
) -> list[dict]:
    """Fetch questions and comments, match them, parse per-model predictions, compute scores.

    Returns the dataset as a list of record dicts. ``token`` defaults to the platform's token
    variable and ``author_id`` to the platform's bot account. Every record carries ``platform``
    and the six research-archive treatment tags read off ``research_archive_dir``
    (:mod:`~metaculus_bot.performance_analysis.research_tags`), and with ``prior_records`` (a
    previous round's dataset) it is also diffed against that pull to tag whatever the platform
    re-resolved or re-scored in place (:mod:`~metaculus_bot.performance_analysis.rescore_diff`).
    Both tag families read as TERNARIES, where None means "not measured" and only False means
    measured-and-absent: docs/performance_analysis.md, "Pass --prior on every round pull" and
    "Two treatment tags read as TERNARY".
    """
    if token is None:
        token = os.environ[platform.token_env]
    if author_id is None:
        author_id = platform.bot_user_id

    logger.info(f"Fetching resolved questions from {platform.name} tournament '{tournament}'...")
    posts = fetch_resolved_questions(tournament, token, base_url=platform.api_base_url)

    logger.info(f"Fetching all bot comments (author_id={author_id})...")
    all_comments = fetch_bot_comments(author_id, token, base_url=platform.api_base_url)
    logger.info(f"Fetched {len(all_comments)} total comments")
    return records_from_posts(
        posts,
        all_comments,
        token,
        platform=platform,
        research_archive_dir=research_archive_dir,
        prior_records=prior_records,
    )


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def save_dataset(data: list[dict], path: str) -> None:
    """Save dataset to a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved {len(data)} records to {path}")


_SCORE_FIELDS = ("brier_score", "log_score", "numeric_log_score", "mc_log_score")


def _rescorable(record: object) -> bool:
    """Whether a cached record carries every field ``_compute_scores`` reads.

    ``_compute_scores`` stays fail-fast for the fresh-build path (``_process_post``
    constructs every key); ``rescore_records`` takes arbitrary JSON off disk —
    hand-trimmed fixtures, older schemas — so the tolerance lives here. The check
    is branch-aware: a numeric record does not need ``our_prob_yes``.
    """
    if not isinstance(record, dict):
        return False
    if not all(k in record for k in ("type", "resolution_parsed", "our_forecast_values")):
        return False
    q_type = record.get("type")
    if q_type == "binary" and "our_prob_yes" not in record:
        return False
    return q_type not in ("numeric", "discrete") or all(k in record for k in ("open_lower_bound", "open_upper_bound"))


def rescore_records(records: list[dict]) -> int:
    """Recompute every record's score fields from its own stored inputs, in place.

    Scores are pure functions of fields the record already carries (type,
    resolution, forecast values, scaling, bounds) — but the score VALUES in a
    cached dataset are whatever the scorer computed when the file was written,
    so a scorer fix never reaches previously-saved JSON. That bit for a month:
    seven fall-2025 ``zero_point`` log-scaled records kept linear-bucket
    ``numeric_log_score`` values up to 358.8 off the platform after the
    zero-point coercion fix (``3c7a3e2``) had already corrected the live path,
    because the round datasets merge frozen per-tournament baselines.

    A field is only overwritten when recomputation yields a value, so healing
    can never DELETE a score whose inputs are no longer recomputable. Returns
    the number of records whose scores changed.
    """
    changed = 0
    for record in records:
        if not _rescorable(record):
            continue
        fresh = {**record, **dict.fromkeys(_SCORE_FIELDS)}
        _compute_scores(fresh)
        record_changed = False
        for field in _SCORE_FIELDS:
            new = fresh[field]
            old = record.get(field)
            if new is not None and (old is None or abs(new - old) > RESCORE_ATOL):
                record[field] = new
                record_changed = True
        changed += record_changed
    return changed


def load_dataset(path: str) -> list[dict]:
    """Load dataset from a JSON file, healing any stale stored scores (see rescore_records)."""
    with open(path) as f:
        data = json.load(f)
    changed = rescore_records(data)
    if changed:
        logger.info(f"Loaded {len(data)} records from {path}; rescored {changed} with stale stored scores")
    else:
        logger.info(f"Loaded {len(data)} records from {path}")
    return data
