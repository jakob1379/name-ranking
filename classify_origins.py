#!/usr/bin/env python3
"""
Classify name origins using name2nat package.

This script:
1. Gets unclassified names from database
2. Uses name2nat to predict nationality
3. Maps nationality to region using region_mapping table
4. Updates database with region and confidence
5. Handles batch processing with progress reporting
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from database import get_unclassified_names, update_name_origin, get_connection


def get_region_for_nationality(nationality: str) -> Tuple[str, float]:
    """
    Map nationality to region using database mapping.
    Returns (region, confidence_adjustment).
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT region FROM region_mapping WHERE nationality = ?",
            (nationality,)
        )
        row = cursor.fetchone()
        if row:
            return row[0], 1.0  # Full confidence for exact match
        
        # Try partial matching (e.g., "American" matches "United States")
        cursor = conn.execute(
            "SELECT region FROM region_mapping WHERE ? LIKE '%' || nationality || '%' OR nationality LIKE '%' || ? || '%'",
            (nationality, nationality)
        )
        row = cursor.fetchone()
        if row:
            return row[0], 0.8  # Reduced confidence for partial match
        
        # Default to International
        return "International", 0.5


def classify_name(name: str) -> Optional[Tuple[str, float]]:
    """
    Classify a single name using name2nat.
    Returns (region, confidence) or None if classification failed.
    """
    try:
        from name2nat import Name2nat
        
        # Initialize classifier (lazy load)
        if not hasattr(classify_name, "_classifier"):
            classify_name._classifier = Name2nat()
        
        classifier = classify_name._classifier
        
        # Predict nationality
        # name2nat expects a list of names, returns list of (name, [(nationality, prob), ...])
        results = classifier([name], top_n=1)
        if not results:
            return None
        
        # Extract top prediction
        name_result = results[0]
        if not name_result[1]:  # No predictions
            return None
        
        nationality, confidence = name_result[1][0]
        
        # Map nationality to region
        region, region_confidence = get_region_for_nationality(nationality)
        
        # Combine confidences
        combined_confidence = confidence * region_confidence
        
        return region, combined_confidence
        
    except ImportError:
        print("Error: name2nat not installed. Install with: pip install name2nat")
        raise
    except Exception as e:
        print(f"Error classifying name '{name}': {e}")
        return None


def classify_batch(names_batch: list, batch_size: int = 100) -> int:
    """
    Classify a batch of names.
    Returns number of successfully classified names.
    """
    classified_count = 0
    
    for i, name_data in enumerate(names_batch):
        name_id = name_data["id"]
        name = name_data["name"]
        
        # Classify
        result = classify_name(name)
        if result:
            region, confidence = result
            update_name_origin(name_id, region, confidence)
            classified_count += 1
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(names_batch)} names...")
    
    return classified_count


def classify_all_names(limit: Optional[int] = None, batch_size: int = 100) -> int:
    """
    Classify all unclassified names.
    Returns total number of names classified.
    """
    print("Fetching unclassified names...")
    unclassified = get_unclassified_names(limit)
    
    if not unclassified:
        print("No unclassified names found.")
        return 0
    
    total = len(unclassified)
    print(f"Found {total} unclassified names.")
    
    if limit and limit < total:
        print(f"Limiting to {limit} names.")
        unclassified = unclassified[:limit]
        total = len(unclassified)
    
    print("Starting classification...")
    print("This may take a while for large datasets.")
    print("Progress: (each dot = 10 names)")
    
    start_time = time.time()
    classified = 0
    
    # Process in batches
    for i in range(0, total, batch_size):
        batch = unclassified[i:i + batch_size]
        batch_classified = classify_batch(batch, batch_size)
        classified += batch_classified
        
        elapsed = time.time() - start_time
        rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
        remaining = total - (i + len(batch))
        eta = remaining / rate if rate > 0 else 0
        
        print(f"  Batch {i//batch_size + 1}: Classified {batch_classified}/{len(batch)}")
        print(f"  Total: {classified}/{total} ({classified/total*100:.1f}%)")
        print(f"  Rate: {rate:.1f} names/sec, ETA: {eta:.0f} seconds")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Classification complete!")
    print(f"   Classified {classified} of {total} names")
    print(f"   Time: {elapsed:.1f} seconds ({elapsed/total:.2f} sec/name)")
    
    return classified


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify name origins")
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of names to classify (for testing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show classification statistics only"
    )
    
    args = parser.parse_args()
    
    if args.stats:
        from database import get_stats
        stats = get_stats()
        total = stats["total_names"]
        classified = stats["classified_names"]
        print(f"Classification Statistics:")
        print(f"  Total names: {total}")
        print(f"  Classified: {classified} ({classified/total*100:.1f}%)")
        print(f"  Unclassified: {total - classified}")
        return
    
    try:
        classify_all_names(args.limit, args.batch_size)
    except ImportError:
        print("\n❌ name2nat is not installed.")
        print("Install it with: pip install name2nat")
        print("Or add to pyproject.toml dependencies:")
        print('  dependencies = [\n    ...\n    "name2nat>=0.0.0",\n    ...\n  ]')
        sys.exit(1)


if __name__ == "__main__":
    main()
