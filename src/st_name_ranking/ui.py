"""Compatibility alias for :mod:`st_name_ranking.interface.ui`."""

import sys

from st_name_ranking.interface import ui as _ui

sys.modules[__name__] = _ui
