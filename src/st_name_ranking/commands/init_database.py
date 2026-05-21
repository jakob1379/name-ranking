#!/usr/bin/env python3
"""Deprecated adapter for the canonical ``st-name-ranking db init`` command."""

import argparse
import sys
from collections.abc import Sequence

from st_name_ranking.commands.cli import init as init_database_command


def main(argv: Sequence[str] | None = None) -> None:
    """Delegate legacy ``python init_database.py`` usage to ``db init``.

    Remove this adapter with the deprecated top-level compatibility surface in 0.3.0.
    """
    parser = argparse.ArgumentParser(
        description="Deprecated adapter for 'st-name-ranking db init'",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Classify unclassified origins after initialization",
    )
    args = parser.parse_args(argv)

    init_database_command(classify=args.classify)


if __name__ == "__main__":
    main(sys.argv[1:])
