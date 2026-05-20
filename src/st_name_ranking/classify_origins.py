"""Compatibility alias for :mod:`st_name_ranking.classification.classify_origins`."""

import sys

from st_name_ranking.classification import classify_origins as _classify_origins

sys.modules[__name__] = _classify_origins
