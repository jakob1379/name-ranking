#!/usr/bin/env python3
"""
Initialize the SQLite database for name ranking application.

This script:
1. Creates the database schema (if not exists)
2. Syncs names from godkendtefornavne submodule
3. Migrates ratings from ratings.json (if exists)
4. Optionally runs initial origin classification

Usage:
    python init_database.py [--classify] [--ratings-path PATH]
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from database import (
    init_database,
    sync_names_with_submodule,
    migrate_ratings_from_json,
    get_stats,
)


def main():
    parser = argparse.ArgumentParser(description="Initialize name ranking database")
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Run initial origin classification (requires name2nat)"
    )
    parser.add_argument(
        "--ratings-path",
        type=Path,
        default=Path("../sort-names/ratings.json"),
        help="Path to ratings.json file (default: ../sort-names/ratings.json)"
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
    
    print("Migrating ratings from JSON...")
    migrated = migrate_ratings_from_json(args.ratings_path)
    print(f"✓ Migrated {migrated} ratings from {args.ratings_path}")
    
    if args.classify:
        print("Running initial origin classification...")
        try:
            from classify_origins import classify_all_names
            classified = classify_all_names()
            print(f"✓ Classified {classified} names")
        except ImportError:
            print("✗ name2nat not installed. Install with: pip install name2nat")
            print("  Or run later: python classify_origins.py")
        except Exception as e:
            print(f"✗ Classification failed: {e}")
    
    # Show statistics
    stats = get_stats()
    print("\nDatabase Statistics:")
    print(f"  Total names: {stats['total_names']}")
    print(f"  Classified names: {stats['classified_names']} ({stats['classified_names']/stats['total_names']*100:.1f}%)")
    print(f"  Rated names: {stats['rated_names']}")
    
    print("\nOrigin Distribution:")
    for region, count in stats['origin_distribution'].items():
        percentage = count / stats['total_names'] * 100
        print(f"  {region}: {count} ({percentage:.1f}%)")
    
    print("\n✅ Database initialization complete!")
    print(f"Database file: {Path('names.db').absolute()}")


if __name__ == "__main__":
    main()
