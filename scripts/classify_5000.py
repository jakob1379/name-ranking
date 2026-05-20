#!/usr/bin/env python3
"""Compatibility wrapper for classifying 5000 names."""

from classify_names import main

if __name__ == "__main__":
    raise SystemExit(
        main(["--limit", "5000", "--batch-size", "200", "--show-stats", "--title", "Classify 5000 Names"]),
    )
