"""Compatibility alias for :mod:`st_name_ranking.persistence.sync_store`."""

import sys

from st_name_ranking.persistence import sync_store as _sync_store

sys.modules[__name__] = _sync_store
