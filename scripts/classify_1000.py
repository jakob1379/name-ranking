#!/usr/bin/env python3
"""Compatibility wrapper for classifying 1000 names."""

from classify_names import main

if __name__ == "__main__":
    raise SystemExit(main(["--limit", "1000", "--batch-size", "100", "--title", "Classify 1000 Names"]))
