#!/usr/bin/env python3
"""Developer wrapper for `st-name-ranking db origins classify`.

Use the CLI command for durable maintenance workflows. This script remains for
local runtime estimates and preset wrappers under `scripts/`.
"""

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path


def _ensure_src_path() -> None:
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify unclassified name origins")
    parser.add_argument("--limit", type=int, help="Maximum number of names to classify")
    parser.add_argument("--batch-size", type=int, default=100, help="Names to process per batch")
    parser.add_argument("--show-stats", action="store_true", help="Print stats before and after classification")
    parser.add_argument("--stats-only", action="store_true", help="Print stats without running classification")
    parser.add_argument(
        "--estimate-seconds-per-name",
        type=float,
        help="Print an estimated runtime using this per-name rate",
    )
    parser.add_argument("--title", default="Name Origin Classification", help="Heading to print before running")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _ensure_src_path()

    from st_name_ranking.commands.cli import _print_origin_classification_stats, _run_origin_classification
    from st_name_ranking.persistence.database import get_stats

    print(f"=== {args.title} ===")

    stats_before = get_stats()
    unclassified = (
        stats_before["unclassified_names"] if isinstance(stats_before, dict) else stats_before.unclassified_names
    )
    if args.show_stats or args.stats_only:
        _print_origin_classification_stats("Before classification:")

    if args.stats_only:
        return 0

    if unclassified == 0:
        print("No unclassified names. Exiting.")
        return 0

    run_limit = min(args.limit, unclassified) if args.limit is not None else unclassified
    if args.estimate_seconds_per_name is not None:
        estimated_seconds = run_limit * args.estimate_seconds_per_name
        print(f"Estimated time: {estimated_seconds:.0f}s ({estimated_seconds / 60:.1f} minutes)")

    print("Starting classification...")
    start = time.time()
    try:
        classified = _run_origin_classification(
            limit=args.limit,
            batch_size=args.batch_size,
            show_stats=False,
        )
    except KeyboardInterrupt:
        print("\nClassification interrupted by user.")
        return 1
    except (RuntimeError, ValueError) as e:
        print(f"\nClassification failed: {e}")
        return 1

    elapsed = time.time() - start
    print(f"\nClassification completed in {elapsed:.1f}s.")
    print(f"Classified {classified} names.")
    if elapsed > 0:
        print(f"Rate: {classified / elapsed:.1f} names/second")

    if args.show_stats:
        _print_origin_classification_stats("After classification:")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
