"""Compatibility alias for :mod:`st_name_ranking.interface.app_actions`."""

import sys

from st_name_ranking.interface import app_actions as _app_actions

sys.modules[__name__] = _app_actions
