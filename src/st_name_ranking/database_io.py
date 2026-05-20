"""Compatibility alias for :mod:`st_name_ranking.persistence.database_io`."""

import sys

from st_name_ranking.persistence import database_io as _database_io

sys.modules[__name__] = _database_io
