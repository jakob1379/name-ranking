"""Compatibility alias for :mod:`st_name_ranking.learning.model`."""

import sys

from st_name_ranking.learning import model as _model

sys.modules[__name__] = _model
