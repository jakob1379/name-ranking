#!/usr/bin/env python3
"""Batch orchestration for hierarchical name-origin classification.

The canonical maintenance entrypoint is ``st-name-ranking db origins classify``.
"""

import logging
import sqlite3
import time

from st_name_ranking.classification.origin_classifier import (
    OriginResult,
    get_or_create_classifier,
    reset_classifier_cache,
)
from st_name_ranking.persistence.database import (
    get_db_path,
    get_names_with_origins,
    get_unclassified_names,
    update_name_origin,
)
from st_name_ranking.types import UnclassifiedName

logger = logging.getLogger(__name__)

# Minimum confidence threshold for classification results
MIN_CONFIDENCE_THRESHOLD = 0.1


def classify_name(name: str) -> OriginResult | None:
    """Classify a single name, returning None when classification is unavailable."""
    reference_names = _get_reference_names()
    classifier = get_or_create_classifier(reference_names)

    region, confidence = classifier.classify(name)
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        return None

    logger.debug(
        "Classified %s -> %s (confidence: %.2f)",
        name,
        region,
        confidence,
    )
    return OriginResult(region, confidence)


def _get_reference_names() -> dict[str, tuple[str, float, str, str]]:
    """Load known names used as origin-classification references."""
    current_db_path = str(get_db_path())
    cache = getattr(_get_reference_names, "_cache", None)
    cache_db_path = getattr(_get_reference_names, "_cache_db_path", None)
    if cache is not None and cache_db_path == current_db_path:
        return cache

    try:
        reference_names = get_names_with_origins(
            confidence_threshold=0.5,
        )
    except sqlite3.Error as e:
        msg = "Failed to load origin-classification reference names from database"
        raise RuntimeError(msg) from e

    _get_reference_names._cache = reference_names
    _get_reference_names._cache_db_path = current_db_path
    logger.debug(
        "Loaded %d reference names",
        len(_get_reference_names._cache),
    )
    return _get_reference_names._cache


def reset_reference_cache() -> None:
    """Clear cached reference data and classifier instances."""
    if hasattr(_get_reference_names, "_cache"):
        delattr(_get_reference_names, "_cache")
    if hasattr(_get_reference_names, "_cache_db_path"):
        delattr(_get_reference_names, "_cache_db_path")
    reset_classifier_cache()


def classify_batch(names_batch: list[UnclassifiedName], batch_size: int = 100) -> int:
    """Classify a batch and return the number of stored results."""
    if not names_batch:
        return 0

    logger.debug("Classifying batch of %d names", len(names_batch))

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


def classify_all_names(
    limit: int | None = None,
    batch_size: int = 100,
    progress_callback: "Callable[[int, int], None] | None" = None,
) -> int:
    """Classify unclassified names and return the number of stored results."""
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
    processed = 0

    for i in range(0, total, batch_size):
        batch = unclassified[i : i + batch_size]
        logger.debug(
            "Processing batch %d (%d names)",
            i // batch_size + 1,
            len(batch),
        )

        batch_classified = classify_batch(batch, batch_size)
        classified += batch_classified
        processed += len(batch)
        if batch_classified:
            reset_reference_cache()

        if progress_callback:
            progress_callback(processed, total)

        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = total - processed
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
