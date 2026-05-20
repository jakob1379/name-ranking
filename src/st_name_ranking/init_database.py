"""Compatibility alias for :mod:`st_name_ranking.commands.init_database`."""

import sys

from st_name_ranking.commands import init_database as _init_database

sys.modules[__name__] = _init_database

if __name__ == "__main__":
    _init_database.main()
