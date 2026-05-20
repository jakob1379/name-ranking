#!/usr/bin/env python3
"""Compatibility wrapper for classifying all currently unclassified names."""

from classify_names import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--batch-size",
                "200",
                "--show-stats",
                "--estimate-seconds-per-name",
                "0.0065",
                "--title",
                "Classify Remaining Names",
            ],
        ),
    )
