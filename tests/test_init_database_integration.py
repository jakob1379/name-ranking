"""Integration tests for init_database module."""

import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from st_name_ranking import init_database


class TestInitDatabaseIntegration:
    """Integration tests for init_database module."""

    def test_main_basic(self):
        """Test main function without classification."""
        # Mock database operations
        mock_stats = {
            "total_names": 100,
            "classified_names": 50,
            "rated_names": 20,
            "origin_distribution": {"Nordic": 30, "European": 20},
        }

        with (
            patch("st_name_ranking.init_database.init_database") as mock_init,
            patch("st_name_ranking.init_database.sync_names_with_submodule") as mock_sync,
            patch("st_name_ranking.init_database.get_stats", return_value=mock_stats),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            mock_sync.return_value = 10

            # Call main with no arguments
            sys.argv = ["init_database.py"]
            init_database.main()

            # Verify database was initialized
            mock_init.assert_called_once()
            mock_sync.assert_called_once()

            # Check output contains expected messages
            output = mock_stdout.getvalue()
            assert "Initializing database..." in output
            assert "Database schema created" in output
            assert "Synced 10 new names from submodule" in output
            assert "Database Statistics:" in output
            assert "Total names: 100" in output

    def test_main_with_classification_success(self):
        """Test main function with successful classification."""
        mock_stats = {
            "total_names": 100,
            "classified_names": 50,
            "rated_names": 20,
            "origin_distribution": {"Nordic": 30, "European": 20},
        }

        # Mock classify_all_names to return a count
        mock_classify_all = MagicMock(return_value=25)

        with (
            patch("st_name_ranking.init_database.init_database"),
            patch("st_name_ranking.init_database.sync_names_with_submodule") as mock_sync,
            patch("st_name_ranking.init_database.get_stats", return_value=mock_stats),
            patch("st_name_ranking.classify_origins.classify_all_names", mock_classify_all),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            mock_sync.return_value = 10

            # Call main with --classify argument
            sys.argv = ["init_database.py", "--classify"]
            init_database.main()

            # Verify classification was attempted
            mock_classify_all.assert_called_once()

            # Check output contains classification messages
            output = mock_stdout.getvalue()
            assert "Running initial origin classification..." in output
            assert "Classified 25 names" in output

    def test_main_with_classification_import_error(self):
        """Test main function when ethnidata is not installed."""
        mock_stats = {
            "total_names": 100,
            "classified_names": 50,
            "rated_names": 20,
            "origin_distribution": {"Nordic": 30, "European": 20},
        }

        # Mock ImportError when importing classify_origins
        with (
            patch("st_name_ranking.init_database.init_database"),
            patch("st_name_ranking.init_database.sync_names_with_submodule") as mock_sync,
            patch("st_name_ranking.init_database.get_stats", return_value=mock_stats),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            # Mock classify_all_names to raise ImportError (ethnidata not installed)
            import_error = ImportError("ethnidata not installed. Install with: pip install ethnidata")
            with patch("st_name_ranking.classify_origins.classify_all_names", side_effect=import_error):
                mock_sync.return_value = 10

                sys.argv = ["init_database.py", "--classify"]
                init_database.main()

                # Check error message is shown
                output = mock_stdout.getvalue()
                assert "ethnidata not installed" in output
                assert "Install with: pip install ethnidata" in output

    def test_main_sync_error(self):
        """Test main function when name sync fails."""
        with (
            patch("st_name_ranking.init_database.init_database"),
            patch("st_name_ranking.init_database.sync_names_with_submodule") as mock_sync,
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
            patch("sys.stderr", new_callable=StringIO),
        ):
            mock_sync.side_effect = Exception("Submodule not found")

            sys.argv = ["init_database.py"]

            # Should exit with code 1
            with pytest.raises(SystemExit) as exc_info:
                init_database.main()

            assert exc_info.value.code == 1
            output = mock_stdout.getvalue()
            assert "Failed to sync names" in output

    def test_main_classification_error(self):
        """Test main function when classification fails with non-import error."""
        mock_stats = {
            "total_names": 100,
            "classified_names": 50,
            "rated_names": 20,
            "origin_distribution": {"Nordic": 30, "European": 20},
        }

        # Mock classify_all_names to raise an exception
        mock_classify_all = MagicMock(side_effect=Exception("Classification API error"))

        with (
            patch("st_name_ranking.init_database.init_database"),
            patch("st_name_ranking.init_database.sync_names_with_submodule") as mock_sync,
            patch("st_name_ranking.init_database.get_stats", return_value=mock_stats),
            patch("st_name_ranking.classify_origins.classify_all_names", mock_classify_all),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            mock_sync.return_value = 10

            sys.argv = ["init_database.py", "--classify"]
            init_database.main()

            # Should show error but continue
            output = mock_stdout.getvalue()
            assert "Classification failed" in output
            # Should still show statistics
            assert "Database Statistics:" in output
