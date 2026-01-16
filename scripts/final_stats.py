#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from st_name_ranking.database import get_stats  # noqa: E402

stats = get_stats()
print("=== Final Classification Stats ===")
print(f"Total names: {stats['total_names']}")
print(
    f"Classified: {stats['classified_names']} ({stats['classified_names'] / stats['total_names'] * 100:.1f}%)",
)
print(f"Unclassified: {stats['unclassified_names']}")
print("\nOrigin distribution:")
for region, count in stats["origin_distribution"].items():
    print(f"  {region}: {count} ({count / stats['total_names'] * 100:.1f}%)")

# Check Nordic classification rate
nordic = stats["origin_distribution"].get("Nordic", 0)
print(
    f"\nNordic classification: {nordic} names ({nordic / stats['total_names'] * 100:.1f}%)",
)
print("\nNote: Nordic detection may need improvement.")
