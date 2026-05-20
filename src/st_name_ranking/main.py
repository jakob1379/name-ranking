"""Compatibility alias for :mod:`st_name_ranking.interface.main`."""

import sys

from st_name_ranking.interface import main as _main

sys.modules[__name__] = _main

if __name__ == "__main__":
    _main.main()
