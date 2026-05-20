"""Compatibility alias for :mod:`st_name_ranking.persistence.data_loader`."""

import sys

from st_name_ranking.persistence import data_loader as _data_loader

sys.modules[__name__] = _data_loader
