"""Compatibility alias for :mod:`st_name_ranking.commands.cli`."""

import sys

from st_name_ranking.commands import cli as _cli

sys.modules[__name__] = _cli
