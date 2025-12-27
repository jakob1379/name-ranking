#!/usr/bin/env python3
"""
Initialize the SQLite database for name ranking application.

This script:
1. Creates the database schema (if not exists)
2. Syncs names from godkendtefornavne submodule
3. Optionally runs initial origin classification

Usage:
    python init_database.py [--classify]
"""

import argparse
import sys
from pathlib import Path

from st_name_ranking.database import (
    get_stats,
    init_database,
    sync_names_with_submodule,
)


def main():
    parser = argparse.ArgumentParser(
        description="Initialize name ranking database"
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Run initial origin classification (requires ethnidata)",
    )

    args = parser.parse_args()

    print("Initializing database...")
    init_database()
    print("✓ Database schema created")

    print("Syncing names from submodule...")
    try:
        inserted = sync_names_with_submodule()
        print(f"✓ Synced {inserted} new names from submodule")
    except Exception as e:
        print(f"✗ Failed to sync names: {e}")
        sys.exit(1)



    if args.classify:
        print("Running initial origin classification...")
        try:
            from classify_origins import classify_all_names

            classified = classify_all_names()
            print(f"✓ Classified {classified} names")
        except ImportError:
            print(
                "✗ ethnidata not installed. Install with: pip install ethnidata"
            )
            print("  Or run later: python classify_origins.py")
        except Exception as e:
            print(f"✗ Classification failed: {e}")

    # Show statistics
    stats = get_stats()
    print("\nDatabase Statistics:")
    print(f"  Total names: {stats['total_names']}")
    print(
        f"  Classified names: {stats['classified_names']} "
        f"({stats['classified_names'] / stats['total_names'] * 100:.1f}%)"
    )
    print(f"  Rated names: {stats['rated_names']}")

    print("\nOrigin Distribution:")
    for region, count in stats["origin_distribution"].items():
        percentage = count / stats["total_names"] * 100
        print(f"  {region}: {count} ({percentage:.1f}%)")

    print("\n✅ Database initialization complete!")
    print(f"Database file: {Path('names.db').absolute()}")


if __name__ == "__main__":
    main()
