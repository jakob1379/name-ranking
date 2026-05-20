"""Compatibility alias for :mod:`st_name_ranking.classification.origin_classifier`."""

import sys

from st_name_ranking.classification import origin_classifier as _origin_classifier

sys.modules[__name__] = _origin_classifier
