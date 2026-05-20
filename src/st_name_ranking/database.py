"""Compatibility alias for :mod:`st_name_ranking.persistence.database`."""

import sys

from st_name_ranking.persistence import database as _database

sys.modules[__name__] = _database
