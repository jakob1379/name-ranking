#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from st_name_ranking.classify_origins import classify_all_names  # noqa: E402

if __name__ == "__main__":
    print("Testing classification of 10 names...")
    classified = classify_all_names(limit=10, batch_size=10)
    print(f"Classified {classified} names.")
