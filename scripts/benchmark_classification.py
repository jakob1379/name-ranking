#!/usr/bin/env python3
"""
Benchmark origin classification performance with phonetic caching.
"""

import logging
import random
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING)


def benchmark_reference_names_load() -> tuple[
    dict[str, tuple[str, float, str, str]],
    float,
]:
    """Benchmark loading reference names with phonetic codes."""
    start = time.perf_counter()
    reference_names = get_names_with_origins(confidence_threshold=0.5)
    elapsed = time.perf_counter() - start
    return reference_names, elapsed


def benchmark_classification_batch(
    names: list[str],
    reference_names: dict[str, tuple[str, float, str, str]],
    batch_size: int = 100,
) -> tuple[int, float]:
    """Benchmark classification of a batch of names."""
    classifier = OriginClassifier(reference_names)

    total_names = len(names)
    start = time.perf_counter()

    # Classify in chunks
    for i in range(0, total_names, batch_size):
        batch = names[i : i + batch_size]
        classifier.classify_batch(batch)
        # Optional: update database (skip for benchmark)
        # for name, (region, confidence) in zip(batch, results):
        #     pass

    elapsed = time.perf_counter() - start
    return total_names, elapsed


def main():
    # Add src to Python path for imports
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    from st_name_ranking.database import (
        get_unclassified_names,
        init_database,
    )

    print("=== Origin Classification Performance Benchmark ===\n")

    # Initialize database
    print("1. Initializing database...")
    init_database()

    # Load reference names
    print("2. Loading reference names with phonetic codes...")
    reference_names, load_time = benchmark_reference_names_load()
    print(
        f"   Loaded {len(reference_names)} reference names in {load_time:.3f}s",
    )
    print(
        f"   Average load time per reference: {load_time / max(1, len(reference_names)):.6f}s",
    )

    # Get unclassified names for testing
    print("3. Fetching unclassified names...")
    unclassified = get_unclassified_names(limit=2000)  # Limit for benchmark
    print(f"   Found {len(unclassified)} unclassified names (limited to 2000)")

    if len(unclassified) < 100:
        print("   Not enough unclassified names for benchmark.")
        return

    # Select random subset for benchmark
    sample_size = min(1000, len(unclassified))
    random.seed(42)
    sampled = random.sample(unclassified, sample_size)
    sampled_names = [item["name"] for item in sampled]

    print(f"4. Benchmarking classification of {sample_size} names...")
    total_names, elapsed = benchmark_classification_batch(
        sampled_names,
        reference_names,
        batch_size=100,
    )

    print("\n=== Results ===")
    print(f"Total names classified: {total_names}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Time per name: {elapsed / total_names:.5f}s")
    print(f"Names per second: {total_names / elapsed:.1f}")

    # Estimate full classification of remaining ~33k names
    remaining = 33366  # from earlier stats
    estimated_time = (elapsed / total_names) * remaining
    print("\n=== Projection for full classification ===")
    print(f"Remaining unclassified names: {remaining}")
    print(
        f"Estimated time: {estimated_time:.1f}s ({estimated_time / 60:.1f} minutes)",
    )

    # Performance target check
    target_per_name = 0.005  # 5ms per name
    actual_per_name = elapsed / total_names
    if actual_per_name <= target_per_name:
        print(
            f"✅ Performance target met: {actual_per_name:.5f}s/name ≤ {target_per_name}s/name",
        )
    else:
        print(
            f"⚠️ Performance target missed: {actual_per_name:.5f}s/name > {target_per_name}s/name",
        )


if __name__ == "__main__":
    main()
