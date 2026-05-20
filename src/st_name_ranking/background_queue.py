"""Compatibility alias for :mod:`st_name_ranking.active_learning.queue`."""

from __future__ import annotations

import sys

from st_name_ranking.active_learning import queue as _queue

sys.modules[__name__] = _queue
