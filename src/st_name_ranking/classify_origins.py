#!/usr/bin/env python3
"""Classify name origins using ethnidata package.

This script:
1. Gets unclassified names from database
2. Uses ethnidata to predict nationality
3. Maps nationality to region using region_mapping table
4. Updates database with region and confidence
5. Handles batch processing with progress reporting
"""

import argparse
import logging
import sys
import time
from typing import Any, NamedTuple


class ClassificationResult(NamedTuple):
    """Origin classification result."""

    region: str
    confidence: float


from st_name_ranking.database import (
    get_connection,
    get_names_with_origins,
    get_stats,
    get_unclassified_names,
    update_name_origin,
)
from st_name_ranking.origin_classifier import (
    get_classifier as get_origin_classifier,
)

logger = logging.getLogger(__name__)

# Minimum confidence threshold for classification results
MIN_CONFIDENCE_THRESHOLD = 0.1

# Module-level cache for ethnidata classifier
_classifier_cache: Any | None = None

# Module-level cache for reference names
_reference_names_cache: dict[str, tuple[str, float, str, str]] | None = None


# Type alias for the classifier function
Classifier = "Callable[[str], ClassificationResult | None]"


def get_classifier() -> Classifier:
    """Get or create the ethnidata classifier (lazy load).

    Returns:
        Function that takes a name and returns (region, confidence) or None.

    Raises:
        ImportError: If ethnidata package is not installed.
    """
    global _classifier_cache

    if _classifier_cache is not None:
        return _classifier_cache

    try:
        from ethnidata import EthniData  # noqa: PLC0415
    except ImportError:
        _msg = "ethnidata not installed. Install with: pip install ethnidata"
        raise ImportError(_msg)

    _classifier_cache = EthniData()
    logger.debug("Initialized ethnidata classifier")
    return _classifier_cache


def get_region_for_nationality(nationality: str) -> tuple[str, float]:
    """Map nationality to region using database mapping.
    Returns (region, confidence_adjustment).
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT region FROM region_mapping WHERE nationality = ?",
            (nationality,),
        )
        row = cursor.fetchone()
        if row:
            return row[0], 1.0  # Full confidence for exact match

        # Try partial matching (e.g., "American" matches "United States")
        cursor = conn.execute(
            "SELECT region FROM region_mapping "
            "WHERE ? LIKE '%' || nationality || '%' "
            "OR nationality LIKE '%' || ? || '%'",
            (nationality, nationality),
        )
        row = cursor.fetchone()
        if row:
            return row[0], 0.8  # Reduced confidence for partial match

        # Default to International
        return "International", 0.5


def classify_name(name: str) -> tuple[str, float] | None:
    """Classify a single name using hierarchical classifier.
    Returns (region, confidence) or None if classification failed.
    """
    try:
        # Get reference names from already classified names in database
        reference_names = _get_reference_names()
        classifier = get_origin_classifier(reference_names)

        region, confidence = classifier.classify(name)

        if confidence < MIN_CONFIDENCE_THRESHOLD:  # Fallback if classifier returns minimal confidence
            return None

        logger.debug(
            "Classified %s -> %s (confidence: %.2f)",
            name,
            region,
            confidence,
        )
        return region, confidence

    except (ImportError, AttributeError, ValueError, RuntimeError) as e:
        logger.warning("Error classifying name '%s': %s", name, e)
        return None


def _get_reference_names() -> dict[str, tuple[str, float, str, str]]:
    """Get dictionary of known name -> (region, confidence, phonetic_primary, phonetic_secondary) from database.

    Cached for performance using module-level cache variable.

    Returns:
        Dictionary mapping name to (region, confidence, phonetic_primary, phonetic_secondary).
    """
    global _reference_names_cache

    if _reference_names_cache is not None:
        return _reference_names_cache

    try:
        _reference_names_cache = get_names_with_origins(
            confidence_threshold=0.5,
        )
        logger.debug(
            "Loaded %d reference names",
            len(_reference_names_cache),
        )
    except sqlite3.Error as e:
        logger.warning("Failed to load reference names: %s", e)
        _reference_names_cache = {}

    return _reference_names_cache


def classify_batch(names_batch: list, batch_size: int = 100) -> int:
    """Classify a batch of names using ethnidata.
    Returns number of successfully classified names.
    """
    if not names_batch:
        return 0

    logger.debug("Classifying batch of %d names", len(names_batch))

    # Process each name individually (ethnidata doesn't support batch)
    classified_count = 0

    for i, name_data in enumerate(names_batch):
        name_id = name_data.id
        name = name_data.name

        result = classify_name(name)
        if result:
            region, confidence = result
            update_name_origin(name_id, region, confidence)
            classified_count += 1

        if (i + 1) % 10 == 0:
            logger.debug("  Processed %d/%d names", i + 1, len(names_batch))

    logger.info("Batch classified %d/%d names", classified_count, len(names_batch))
    return classified_count


def _classify_individually(names_batch: list) -> int:
    """Fallback: classify names individually (used when batch processing fails)."""
    classified_count = 0

    for i, name_data in enumerate(names_batch):
        name_id = name_data.id
        name = name_data.name

        result = classify_name(name)
        if result:
            region, confidence = result
            update_name_origin(name_id, region, confidence)
            classified_count += 1

        if (i + 1) % 10 == 0:
            logger.debug(
                "  Individually processed %d/%d names",
                i + 1,
                len(names_batch),
            )

    logger.info(
        "Individually classified %d/%d names",
        classified_count,
        len(names_batch),
    )
    return classified_count


def classify_all_names(
    limit: int | None = None,
    batch_size: int = 100,
) -> int:
    """Classify all unclassified names.
    Returns total number of names classified.
    """
    logger.info("Fetching unclassified names...")
    unclassified = get_unclassified_names(limit)

    if not unclassified:
        logger.info("No unclassified names found.")
        return 0

    total = len(unclassified)
    logger.info("Found %d unclassified names.", total)

    if limit and limit < total:
        logger.info("Limiting to %d names.", limit)
        unclassified = unclassified[:limit]
        total = len(unclassified)

    logger.info("Starting classification...")
    logger.info("This may take a while for large datasets.")
    logger.debug("Progress: (each dot = 10 names)")

    start_time = time.time()
    classified = 0

    # Process in batches
    for i in range(0, total, batch_size):
        batch = unclassified[i : i + batch_size]
        logger.debug(
            "Processing batch %d (%d names)",
            i // batch_size + 1,
            len(batch),
        )

        batch_classified = classify_batch(batch, batch_size)
        classified += batch_classified

        elapsed = time.time() - start_time
        rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
        remaining = total - (i + len(batch))
        eta = remaining / rate if rate > 0 else 0

        logger.info(
            "Batch %d: Classified %d/%d",
            i // batch_size + 1,
            batch_classified,
            len(batch),
        )
        logger.info(
            "Total: %d/%d (%.1f%%)",
            classified,
            total,
            classified / total * 100,
        )
        logger.debug("Rate: %.1f names/sec, ETA: %.0f seconds", rate, eta)

    elapsed = time.time() - start_time
    logger.info("✅ Classification complete!")
    logger.info("   Classified %d of %d names", classified, total)
    logger.info(
        "   Time: %.1f seconds (%.2f sec/name)",
        elapsed,
        elapsed / total,
    )

    return classified


def main() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Classify name origins")
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of names to classify (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show classification statistics only",
    )

    args = parser.parse_args()

    if args.stats:
        stats = get_stats()
        total = stats["total_names"]
        classified = stats["classified_names"]
        print("Classification Statistics:")
        print(f"  Total names: {total}")
        print(f"  Classified: {classified} ({classified / total * 100:.1f}%)")
        print(f"  Unclassified: {total - classified}")
        return

    try:
        classify_all_names(args.limit, args.batch_size)
    except ImportError:
        print("\n❌ ethnidata is not installed.")
        print("Install it with: pip install ethnidata")
        print("Or add to pyproject.toml dependencies:")
        print(
            '  dependencies = [\n    ...\n    "ethnidata>=4.1.1",\n    ...\n  ]',
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
