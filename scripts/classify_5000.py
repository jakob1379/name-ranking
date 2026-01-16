#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from st_name_ranking.classify_origins import classify_all_names  # noqa: E402
from st_name_ranking.database import get_stats  # noqa: E402

stats_before = get_stats()
print(
    f"Before: {stats_before['classified_names']} classified, {stats_before['unclassified_names']} unclassified",
)

classified = classify_all_names(limit=5000, batch_size=200)

stats_after = get_stats()
print(
    f"After: {stats_after['classified_names']} classified, {stats_after['unclassified_names']} unclassified",
)
print(f"Classified {classified} names in this run.")
