#!/usr/bin/env python3
import logging
import sys
import time
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from st_name_ranking.classify_origins import classify_all_names  # noqa: E402
from st_name_ranking.database import get_stats  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    stats = get_stats()
    unclassified = stats["unclassified_names"]
    if unclassified == 0:
        print("No unclassified names.")
        return 0

    print(f"Unclassified names: {unclassified}")
    print(
        f"Estimated time: {unclassified * 0.0065:.0f}s ({unclassified * 0.0065 / 60:.1f} minutes)",
    )
    print("Starting classification...")

    start = time.time()
    try:
        classified = classify_all_names(limit=None, batch_size=200)
        elapsed = time.time() - start
        print(f"\n✅ Classification completed in {elapsed:.1f}s.")
        print(f"Classified {classified} names.")
        print(f"Rate: {classified / elapsed:.1f} names/second")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Final stats
    stats = get_stats()
    print("\nFinal stats:")
    print(f"Total names: {stats['total_names']}")
    print(
        f"Classified: {stats['classified_names']} ({stats['classified_names'] / stats['total_names'] * 100:.1f}%)",
    )
    print(f"Unclassified: {stats['unclassified_names']}")
    print("\nOrigin distribution:")
    for region, count in stats["origin_distribution"].items():
        print(
            f"  {region}: {count} ({count / stats['total_names'] * 100:.1f}%)",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
