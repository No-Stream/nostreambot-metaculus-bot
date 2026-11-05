"""Extract research text from Metaculus bot comments for backtest replay.

The bot posts a structured comment on each question it forecasts. This script
fetches all such comments, extracts the research section, and writes JSONL
compatible with the research archive system.
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import requests

from metaculus_bot.api_preflight import verify_metaculus_api_identity
from metaculus_bot.performance_analysis.collector import expand_truncated_comment
from metaculus_bot.research.section_format import detect_gap_fill, detect_providers

logger = logging.getLogger(__name__)

BOT_USER_ID = 275109
BASE_URL = "https://www.metaculus.com/api"
PAGE_SIZE = 100
FETCH_DELAY = 2.0

RESEARCH_START_MARKER = "# RESEARCH"
FORECASTS_MARKER = "# FORECASTS"


def get_token() -> str:
    """Load METACULUS_TOKEN from environment or .env file."""
    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("METACULUS_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not token:
        logger.error("METACULUS_TOKEN not found in environment or .env file")
        sys.exit(1)
    return token


def api_get(endpoint: str, token: str, params: dict | None = None) -> dict:
    """Authenticated GET to the Metaculus API."""
    url = f"{BASE_URL}{endpoint}"
    headers = {"Authorization": f"Token {token}"}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _page_comments(token: str, params: dict[str, object]) -> list[dict]:
    """Page one ``/comments/`` listing to exhaustion, on top of the caller's params."""
    comments: list[dict] = []
    offset = 0
    while True:
        data = api_get("/comments/", token, params={**params, "limit": PAGE_SIZE, "offset": offset})
        results = data.get("results", [])
        comments.extend(results)
        logger.info(f"Fetched {len(comments)} comments so far ({offset=}, {params=})")
        if not results or data.get("next") is None:
            break
        offset += PAGE_SIZE
        time.sleep(FETCH_DELAY)
    return comments


def fetch_all_comments(token: str) -> list[dict]:
    """Every bot comment, public and private, deduplicated by comment id.

    The bot POSTs comments private and Metaculus flips older ones public server-side, so the
    default author listing serves only the flipped ones (docs/performance_analysis.md, "The
    research archive's latest/ records come from three writers").
    """
    public = _page_comments(token, {"author": BOT_USER_ID})
    private = _page_comments(token, {"author": BOT_USER_ID, "is_private": "true"})

    merged: dict[int, dict] = {}
    for comment in (*public, *private):
        merged[comment["id"]] = comment

    n_public, n_private, n_unique = len(public), len(private), len(merged)
    logger.info(f"Fetched bot comments: {n_public=} {n_private=} {n_unique=}")
    return [expand_truncated_comment(comment, token) for comment in merged.values()]


def get_question_id_for_post(post_id: int, token: str) -> int | None:
    """Resolve a post_id to a question_id via the posts API.

    Soft-fails per post: one unreachable/404/malformed post must not abandon the rest of
    the backfill. Scoped to ``requests.RequestException``, which covers every transport,
    HTTP-status and JSON-decode failure of the call — a shape violation in a 200 body
    (say ``question`` arriving as a list) is a contract break and still crashes.
    """
    try:
        data = api_get(f"/posts/{post_id}/", token)
        question = data.get("question")
        if question and isinstance(question, dict):
            return question.get("id")
        if isinstance(question, int):
            return question
        return None
    except requests.RequestException as e:
        logger.warning(f"Failed to resolve post {post_id} to question: {e}")
        return None


def extract_research_text(comment_text: str) -> str | None:
    """Extract the research section from a bot comment.

    Returns the text between '# RESEARCH' and '# FORECASTS' markers,
    stripping the '# RESEARCH' and '## Report 1 Research' header lines.
    Returns None if no research section found or if it's empty.
    """
    research_start = comment_text.find(RESEARCH_START_MARKER)
    if research_start == -1:
        return None

    # Start after the "# RESEARCH" line
    content_start = research_start + len(RESEARCH_START_MARKER)

    forecasts_start = comment_text.find(FORECASTS_MARKER, content_start)
    if forecasts_start == -1:
        research_block = comment_text[content_start:]
    else:
        research_block = comment_text[content_start:forecasts_start]

    # Strip leading header lines: blank lines and "## Report 1 Research"
    lines = research_block.split("\n")
    content_start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in ("", "## Report 1 Research"):
            content_start_idx = i + 1
        else:
            break

    research_text = "\n".join(lines[content_start_idx:]).strip()
    return research_text if research_text else None


def detect_trimmed(research_text: str) -> bool:
    """Check if research was trimmed due to length."""
    lower = research_text.lower()
    return "[... trimmed for length]" in lower or "[...trimmed" in lower


def build_record(comment: dict, question_id: int | None) -> dict | None:
    """Build a JSONL record from a comment dict. Returns None if no research found."""
    comment_text = comment.get("text") or comment.get("comment_text") or ""
    if not comment_text:
        return None

    research_text = extract_research_text(comment_text)
    if not research_text:
        return None

    post_id = comment.get("on_post")
    comment_id = comment.get("id")

    return {
        "schema_version": 1,
        "qid": question_id,
        "page_url": f"https://www.metaculus.com/questions/{question_id}/" if question_id else "",
        "question_text": "",
        "research_text": research_text,
        "providers_used": detect_providers(research_text),
        "run_mode": "tournament",
        "tournament_id": "",
        "timestamp": comment.get("created_at", ""),
        "run_id": f"comment-{comment_id}",
        "research_chars": len(research_text),
        "gap_fill_used": detect_gap_fill(research_text),
        "on_post": post_id,
        "is_trimmed": detect_trimmed(research_text),
    }


def load_existing_post_ids(output_file: Path) -> set[int]:
    """Post ids already written to the backfill file, so a re-run appends only new ones."""
    existing_post_ids: set[int] = set()
    if not output_file.exists():
        return existing_post_ids
    with open(output_file) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if "on_post" in record:
                    existing_post_ids.add(record["on_post"])
    return existing_post_ids


@dataclass
class BackfillTally:
    """What one pass over the fetched comments produced, for the closing summary line."""

    records_written: int = 0
    trimmed: int = 0
    no_research: int = 0


def append_backfill_records(
    comments: list[dict],
    token: str,
    out_handle: TextIO,
    existing_post_ids: set[int],
    *,
    dry_run: bool,
) -> BackfillTally:
    """Write a JSONL record for every not-yet-captured comment carrying a research section.

    ``existing_post_ids`` is read AND extended, so a duplicate post inside one fetch is
    skipped the same way a re-run skips what a previous run wrote. The post_id ->
    question_id lookup is cached and rate-gated (one sleep per uncached post).
    """
    post_to_qid: dict[int, int | None] = {}
    tally = BackfillTally()

    for idx, comment in enumerate(comments, 1):
        post_id: int | None = comment.get("on_post")
        if post_id is None or post_id in existing_post_ids:
            continue

        if post_id not in post_to_qid:
            post_to_qid[post_id] = get_question_id_for_post(post_id, token)
            time.sleep(FETCH_DELAY)

        record = build_record(comment, question_id=post_to_qid[post_id])
        if record is None:
            tally.no_research += 1
            continue

        if record["is_trimmed"]:
            tally.trimmed += 1

        if not dry_run:
            out_handle.write(json.dumps(record) + "\n")
            existing_post_ids.add(post_id)
        tally.records_written += 1

        if idx % 50 == 0:
            logger.info(f"Processed {idx}/{len(comments)} comments, {tally.records_written} records written")

    return tally


def main():
    parser = argparse.ArgumentParser(description="Backfill research archive from Metaculus bot comments.")
    parser.add_argument("--output-dir", default="backtests/research_archive/backfill", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Fetch comments but don't write")
    parser.add_argument("--limit", type=int, default=None, help="Max comments to process (for testing)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    token = get_token()

    # Confirm the host is the real API before the token goes out (metaculus_bot/api_preflight.py).
    verify_metaculus_api_identity()

    logger.info("Fetching bot comments from Metaculus API...")
    comments = fetch_all_comments(token)
    logger.info(f"Total comments fetched: {len(comments)}")

    if args.limit:
        comments = comments[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "comments_backfill.jsonl"

    existing_post_ids = load_existing_post_ids(output_file)
    with open(output_file, "a") as f:
        tally = append_backfill_records(comments, token, f, existing_post_ids, dry_run=args.dry_run)

    logger.info(
        f"Done. {tally.records_written} new records, {tally.trimmed} trimmed, "
        f"{tally.no_research} without research section"
    )


if __name__ == "__main__":
    main()
