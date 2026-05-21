"""Compatibility tests for the legacy init_database module."""

from unittest.mock import patch

import pytest

from st_name_ranking import init_database


def test_main_delegates_to_canonical_db_init_without_classification():
    """Legacy module should not duplicate the canonical db init workflow."""
    with patch("st_name_ranking.commands.init_database.init_database_command") as init_command:
        init_database.main([])

    init_command.assert_called_once_with(classify=False)


def test_main_delegates_to_canonical_db_init_with_classification():
    """The legacy --classify flag should map to the canonical command option."""
    with patch("st_name_ranking.commands.init_database.init_database_command") as init_command:
        init_database.main(["--classify"])

    init_command.assert_called_once_with(classify=True)


def test_main_rejects_unknown_legacy_arguments():
    """The adapter should fail fast for arguments the canonical command does not expose."""
    with pytest.raises(SystemExit) as exc_info:
        init_database.main(["--unknown-option"])

    assert exc_info.value.code == 2
