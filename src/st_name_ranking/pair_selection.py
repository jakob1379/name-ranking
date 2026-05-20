"""Compatibility alias for :mod:`st_name_ranking.active_learning.selection`."""

from __future__ import annotations

import sys

from st_name_ranking.active_learning import selection as _selection

sys.modules[__name__] = _selection
