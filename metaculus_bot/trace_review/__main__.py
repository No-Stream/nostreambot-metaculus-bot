"""Build the trace-review bundle: ``python -m metaculus_bot.trace_review`` (``make trace_index``).

Free and read-only: the only network access is the collector's paced GETs of resolved posts and
the bot's own comments. Flags: docs/trace_review.md "Building the index".
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from metaculus_bot.constants import MANTIC_TOKEN_ENV, PLATFORM_MANTIC, PLATFORM_METACULUS
from metaculus_bot.trace_review.build import BuildConfig, build_index
from metaculus_bot.trace_review.config import DEFAULT_BUNDLE_DIR

logger: logging.Logger = logging.getLogger(__name__)


def platforms_with_tokens() -> tuple[str, ...]:
    """Metaculus always (its pull fails fast without a token); Mantic only when its token is set.

    This default is what lets ``make sync_all`` rebuild the index unattended before the operator
    has added MANTIC_TOKEN. Naming ``--platforms mantic`` explicitly still fails without it.
    """
    if os.environ.get(MANTIC_TOKEN_ENV):
        return (PLATFORM_METACULUS, PLATFORM_MANTIC)
    logger.warning("%s is not set, so this build skips the fresh Mantic pull", MANTIC_TOKEN_ENV)
    return (PLATFORM_METACULUS,)


def parse_args(argv: list[str] | None = None) -> BuildConfig:
    parser = argparse.ArgumentParser(description="Build the offline trace-review index (free, read-only)")
    parser.add_argument("--perf-dataset", type=Path, default=None, help="Round file (default: the newest round's)")
    parser.add_argument(
        "--platforms",
        nargs="+",
        choices=(PLATFORM_METACULUS, PLATFORM_MANTIC),
        default=None,
        help="Platforms to pull fresh (default: Metaculus, plus Mantic when MANTIC_TOKEN is set)",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore cached pulls and pull again")
    parser.add_argument(
        "--skip-fresh-pull", action="store_true", help="No network at all: the index stops at the round"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BUNDLE_DIR, help="Bundle directory")
    args = parser.parse_args(argv)
    return BuildConfig(
        output_dir=args.output_dir,
        round_path=args.perf_dataset,
        platforms=tuple(args.platforms) if args.platforms else platforms_with_tokens(),
        refresh=args.refresh,
        skip_fresh_pull=args.skip_fresh_pull,
    )


# Live-pipeline markers the member replay re-emits once per curve; offline they are noise.
REPLAY_NOISE_LOGGERS: tuple[str, ...] = ("metaculus_bot.numeric.pchip_cdf",)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
    load_dotenv()
    for name in REPLAY_NOISE_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    build_index(parse_args(argv))


if __name__ == "__main__":
    main()
