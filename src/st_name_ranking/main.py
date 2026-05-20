"""Deprecated alias for :mod:`st_name_ranking.interface.main`."""

from st_name_ranking._compat import install_deprecated_module_alias

if __name__ == "__main__":
    from st_name_ranking.interface.main import main

    main()
else:
    install_deprecated_module_alias(__name__, "st_name_ranking.interface.main", remove_in="0.3.0")
