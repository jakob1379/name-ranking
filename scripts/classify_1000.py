#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def main():
    # Add src to Python path for imports
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))

    from st_name_ranking.classify_origins import classify_all_names

    print("Classifying 1000 names...")
    classified = classify_all_names(limit=1000, batch_size=100)
    print(f"Classified {classified} names.")


if __name__ == "__main__":
    main()
