"""Compatibility alias for :mod:`st_name_ranking.persistence.feature_store`."""

import sys

from st_name_ranking.persistence import feature_store as _feature_store

sys.modules[__name__] = _feature_store
