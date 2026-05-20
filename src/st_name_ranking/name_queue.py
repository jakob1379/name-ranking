"""Compatibility alias for :mod:`st_name_ranking.learning.name_queue`."""

import sys

from st_name_ranking.learning import name_queue as _name_queue

sys.modules[__name__] = _name_queue
