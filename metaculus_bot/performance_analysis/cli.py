"""CLI entry point for performance analysis.

Usage:
    python -m metaculus_bot.performance_analysis [--tournament SLUG] [--output PATH] [--cached PATH]
                                                 [--prior PATH] [--platform {metaculus,mantic}]

``--output`` saves on both paths and has no default: docs/performance_analysis.md
"The round pull, and why ``--prior`` is mandatory".
"""

import argparse
import logging
import sys

from metaculus_bot.api_preflight import verify_api_identity, verify_metaculus_api_identity
from metaculus_bot.constants import PLATFORM_MANTIC, PLATFORM_METACULUS
from metaculus_bot.performance_analysis.analysis import generate_report
from metaculus_bot.performance_analysis.collector import (
    COLLECTOR_PLATFORMS,
    build_performance_dataset,
    load_dataset,
    save_dataset,
)
from metaculus_bot.performance_analysis.rescore_diff import diff_platform_rescores, render_rescore_summary

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_TOURNAMENT = "spring-aib-2026"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Metaculus bot performance analysis")
    parser.add_argument("--tournament", default=DEFAULT_TOURNAMENT, help="Tournament slug (default: %(default)s)")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Where to save the dataset. Saves on both the live-pull and the --cached path, and is "
            "REQUIRED for a live pull. Omit it on --cached for a read-only report."
        ),
    )
    parser.add_argument("--cached", default=None, help="Load from cached JSON instead of fetching from API")
    parser.add_argument(
        "--prior",
        default=None,
        help=(
            "A previous round's dataset JSON. Every question present in both is diffed on its "
            "resolution and its Metaculus scores, and anything Metaculus changed in place is "
            "tagged platform_rescored and logged as a PLATFORM_RESCORED warning."
        ),
    )
    parser.add_argument(
        "--platform",
        choices=sorted(COLLECTOR_PLATFORMS),
        default=PLATFORM_METACULUS,
        help="Which platform the live pull reads, with its own token and bot account (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.cached is None and args.output is None:
        parser.error("--output is required for a live pull, which would otherwise discard everything it fetched")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)

    prior = load_dataset(args.prior) if args.prior else None

    if args.cached:
        logger.info(f"Loading cached dataset from {args.cached}")
        data = load_dataset(args.cached)
        # Catches a re-resolution after the fact, with no second API pull.
        if prior is not None:
            diff_platform_rescores(prior, data)
    else:
        # The live pull sends the platform's token, so confirm the host first (see api_preflight.py).
        platform = COLLECTOR_PLATFORMS[args.platform]
        if platform.name == PLATFORM_MANTIC:
            verify_api_identity(platform.api_base_url)
        else:
            verify_metaculus_api_identity()
        data = build_performance_dataset(tournament=args.tournament, prior_records=prior, platform=platform)

    # Both paths save: the cached path tags records in place and used to drop them on exit.
    if args.output:
        save_dataset(data, args.output)

    if prior is not None:
        print("\n".join(render_rescore_summary(data)))

    report = generate_report(data)
    print(report)


if __name__ == "__main__":
    main()
