#!/usr/bin/env python3
"""Classify all remaining unclassified names."""

import logging
import sys
from pathlib import Path


def main():
    # Add src to Python path for imports
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))

    from st_name_ranking.classify_origins import classify_all_names
    from st_name_ranking.database import get_stats

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=== Full Name Origin Classification ===")
    stats_before = get_stats()
    print("Before classification:")
    print(f"  Total names: {stats_before['total_names']}")
    print(
        f"  Classified: {stats_before['classified_names']} "
        f"({stats_before['classified_names'] / stats_before['total_names'] * 100:.1f}%)",
    )
    print(f"  Unclassified: {stats_before['unclassified_names']}")

    if stats_before["unclassified_names"] == 0:
        print("No unclassified names. Exiting.")
        return 0

    # Estimate time based on previous performance (12ms per name)
    estimated_seconds = stats_before["unclassified_names"] * 0.012
    print(
        f"\nEstimated time: {estimated_seconds:.0f}s ({estimated_seconds / 60:.1f} minutes)",
    )
    print("Starting classification...")

    try:
        classified = classify_all_names(limit=None, batch_size=200)
        print(f"\n✅ Classification completed. Classified {classified} names.")
    except KeyboardInterrupt:
        print("\n⚠️ Classification interrupted by user.")
        sys.exit(1)
    except (RuntimeError, ValueError) as e:
        print(f"\n❌ Classification failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    stats_after = get_stats()
    print("\nAfter classification:")
    print(f"  Total names: {stats_after['total_names']}")
    print(
        f"  Classified: {stats_after['classified_names']} "
        f"({stats_after['classified_names'] / stats_after['total_names'] * 100:.1f}%)",
    )
    print(f"  Unclassified: {stats_after['unclassified_names']}")

    # Show origin distribution
    print("\nOrigin distribution:")
    for region, count in stats_after["origin_distribution"].items():
        print(
            f"  {region}: {count} ({count / stats_after['total_names'] * 100:.1f}%)",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
