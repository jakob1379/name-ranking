"""Compatibility alias for :mod:`st_name_ranking.persistence.feature_cache`."""

import sys

from st_name_ranking.persistence import feature_cache as _feature_cache

sys.modules[__name__] = _feature_cache
