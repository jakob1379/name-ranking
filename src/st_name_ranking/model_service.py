"""Compatibility alias for :mod:`st_name_ranking.active_learning.lazy_updates`."""

from __future__ import annotations

import sys

from st_name_ranking.active_learning import lazy_updates as _lazy_updates

sys.modules[__name__] = _lazy_updates
