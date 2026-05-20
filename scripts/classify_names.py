#!/usr/bin/env python3
"""Parameterized driver for origin classification batches."""

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _ensure_src_path() -> None:
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))


def _print_stats(label: str) -> None:
    from st_name_ranking.database import get_stats

    stats = get_stats()
    total = _stat(stats, "total_names")
    classified = _stat(stats, "classified_names")
    unclassified = _stat(stats, "unclassified_names")
    origin_distribution = _stat(stats, "origin_distribution")

    print(label)
    print(f"  Total names: {total}")
    if total > 0:
        print(f"  Classified: {classified} ({classified / total * 100:.1f}%)")
    else:
        print("  Classified: 0 (0.0%)")
    print(f"  Unclassified: {unclassified}")

    if origin_distribution:
        print("  Origin distribution:")
        for region, count in origin_distribution.items():
            percentage = count / total * 100 if total > 0 else 0.0
            print(f"    {region}: {count} ({percentage:.1f}%)")


def _stat(stats: object, name: str) -> Any:
    if isinstance(stats, dict):
        return stats[name]
    return getattr(stats, name)


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

    from st_name_ranking.classify_origins import classify_all_names
    from st_name_ranking.database import get_stats

    print(f"=== {args.title} ===")

    stats_before = get_stats()
    unclassified = _stat(stats_before, "unclassified_names")
    if args.show_stats or args.stats_only:
        _print_stats("Before classification:")

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
        classified = classify_all_names(limit=args.limit, batch_size=args.batch_size)
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
        _print_stats("\nAfter classification:")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
