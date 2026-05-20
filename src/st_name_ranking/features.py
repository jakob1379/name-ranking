"""Compatibility alias for :mod:`st_name_ranking.learning.features`."""

import sys

from st_name_ranking.learning import features as _features

sys.modules[__name__] = _features
