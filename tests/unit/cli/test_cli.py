"""CLI integration tests for the Name Ranking application."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from st_name_ranking.commands.cli import app
from st_name_ranking.persistence.feature_store import FeatureCacheRebuildResult
from st_name_ranking.types import DatabaseStats


@pytest.fixture
def cli_runner():
    """Fixture for CLI testing."""
    return CliRunner()


@pytest.fixture(autouse=True)
def mock_feature_cache_rebuild(monkeypatch):
    """Keep CLI command tests focused on command wiring, not feature extraction."""
    monkeypatch.setattr(
        "st_name_ranking.commands.cli.rebuild_feature_cache",
        lambda **_: FeatureCacheRebuildResult(
            version="test",
            feature_names=["length"],
            feature_set_id=1,
            processed=0,
        ),
    )


@pytest.fixture
def temp_db_path():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Clean up
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def mock_db_path(temp_db_path):
    """Mock the database path in st_name_ranking.persistence.database."""
    from st_name_ranking.persistence import database

    original_path = database.get_db_path()

    database.set_db_path(temp_db_path)
    database.init_database()

    yield temp_db_path

    database.set_db_path(original_path)


@pytest.fixture
def mock_submodule_path(tmp_path):
    """Create a mock submodule directory structure."""
    submodule_path = tmp_path / "godkendtefornavne"
    submodule_path.mkdir()

    # Create mock JSON file for sync_names_with_submodule
    json_file = submodule_path / "allenavne.json"
    json_data = [
        {"name": "Anna", "gender": "F", "name_type": "first"},
        {"name": "Peter", "gender": "M", "name_type": "first"},
        {"name": "Maria", "gender": "F", "name_type": "first"},
    ]
    json_file.write_text(json.dumps(json_data, indent=2))

    return submodule_path


def test_cli_help(cli_runner):
    """Test that the CLI help works."""
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Name Ranking Database Management CLI" in result.output
    assert "db" in result.output

    db_help = cli_runner.invoke(app, ["db", "--help"])
    assert db_help.exit_code == 0
    assert "init" in db_help.output
    assert "stats" in db_help.output


def test_cli_init_basic(mock_db_path, cli_runner):
    """Test basic database initialization."""
    # Mock the database initialization to avoid submodule dependency
    with (
        patch("st_name_ranking.commands.cli.init_database") as mock_init,
        patch("st_name_ranking.commands.cli.sync_names_with_submodule") as mock_sync,
        patch("st_name_ranking.commands.cli.get_stats") as mock_stats,
    ):
        mock_init.return_value = None
        mock_sync.return_value = 0  # No new names
        # Mock stats to avoid database queries
        mock_stats.return_value = DatabaseStats(
            total_names=100,
            classified_names=20,
            unclassified_names=80,
            rated_names=100,
            origin_distribution={"International": 80, "European": 20},
        )

        result = cli_runner.invoke(app, ["db", "init"])
        assert result.exit_code == 0
        assert "Initializing Name Ranking Database" in result.output
        assert "Database schema created" in result.output
        assert "Synced 0 new names from submodule" in result.output

        # Check that functions were called
        mock_init.assert_called_once()
        mock_sync.assert_called_once()
        mock_stats.assert_called_once()


def test_cli_init_with_classify(mock_db_path, cli_runner):
    """Test database initialization with classification."""
    with (
        patch("st_name_ranking.commands.cli.init_database") as mock_init,
        patch("st_name_ranking.commands.cli.sync_names_with_submodule") as mock_sync,
        patch("st_name_ranking.commands.cli.classify_all_names") as mock_classify,
        patch("st_name_ranking.commands.cli.get_stats") as mock_stats,
    ):
        mock_init.return_value = None
        mock_sync.return_value = 0  # No new names
        mock_classify.return_value = 3  # 3 names classified
        # Mock stats to avoid database queries
        mock_stats.return_value = DatabaseStats(
            total_names=100,
            classified_names=20,
            unclassified_names=80,
            rated_names=100,
            origin_distribution={"International": 80, "European": 20},
        )

        result = cli_runner.invoke(app, ["db", "init", "--classify"])
        assert result.exit_code == 0
        assert "Database schema created" in result.output
        assert "Synced 0 new names from submodule" in result.output
        assert "Processing Data Enrichment" in result.output
        assert "Classified 3 names" in result.output

        # Check that functions were called
        mock_init.assert_called_once()
        mock_sync.assert_called_once()
        mock_classify.assert_called_once()
        mock_stats.assert_called_once()


def test_cli_classify(mock_db_path, cli_runner):
    """Test classify command."""
    # First initialize the database and insert some names
    with (
        patch("st_name_ranking.commands.cli.init_database") as mock_init,
        patch(
            "st_name_ranking.commands.cli.sync_names_with_submodule",
        ) as mock_sync_init,
        patch("st_name_ranking.commands.cli.get_stats") as mock_stats_init,
    ):
        mock_init.return_value = None
        mock_sync_init.return_value = 0

        mock_stats_init.return_value = DatabaseStats(
            total_names=100,
            classified_names=20,
            unclassified_names=80,
            rated_names=100,
            origin_distribution={"International": 80, "European": 20},
        )

        # Initialize
        init_result = cli_runner.invoke(app, ["db", "init"])
        assert init_result.exit_code == 0

    # Test classify
    with patch("st_name_ranking.commands.cli.classify_all_names") as mock_classify:
        mock_classify.return_value = 10  # 10 names classified

        result = cli_runner.invoke(app, ["db", "origins", "classify", "--limit", "50"])
        assert result.exit_code == 0
        assert "Processing Data Enrichment" in result.output
        assert "Classified 10 names" in result.output

        # Check that classify was called with limit
        mock_classify.assert_called_once_with(50, 100)


def test_cli_classify_with_batch_size(mock_db_path, cli_runner):
    """Test classify command with custom batch size."""
    # First initialize the database
    with (
        patch("st_name_ranking.commands.cli.init_database") as mock_init,
        patch(
            "st_name_ranking.commands.cli.sync_names_with_submodule",
        ) as mock_sync_init,
        patch("st_name_ranking.commands.cli.get_stats") as mock_stats_init,
    ):
        mock_init.return_value = None
        mock_sync_init.return_value = 0

        mock_stats_init.return_value = DatabaseStats(
            total_names=100,
            classified_names=20,
            unclassified_names=80,
            rated_names=100,
            origin_distribution={"International": 80, "European": 20},
        )
        init_result = cli_runner.invoke(app, ["db", "init"])
        assert init_result.exit_code == 0

    # Test classify with batch size
    with patch("st_name_ranking.commands.cli.classify_all_names") as mock_classify:
        mock_classify.return_value = 5  # 5 names classified

        result = cli_runner.invoke(app, ["db", "origins", "classify", "--batch-size", "20"])
        assert result.exit_code == 0
        assert "Processing Data Enrichment" in result.output
        assert "Classified 5 names" in result.output

        # Check that classify was called with batch size
        mock_classify.assert_called_once_with(None, 20)


def test_cli_stats(mock_db_path, cli_runner):
    """Test stats command."""
    # First initialize the database
    with (
        patch("st_name_ranking.commands.cli.init_database") as mock_init,
        patch(
            "st_name_ranking.commands.cli.sync_names_with_submodule",
        ) as mock_sync_init,
        patch("st_name_ranking.commands.cli.get_stats") as mock_stats_init,
    ):
        mock_init.return_value = None
        mock_sync_init.return_value = 0

        mock_stats_init.return_value = DatabaseStats(
            total_names=100,
            classified_names=20,
            unclassified_names=80,
            rated_names=100,
            origin_distribution={"International": 80, "European": 20},
        )
        init_result = cli_runner.invoke(app, ["db", "init"])
        assert init_result.exit_code == 0

    # Test stats
    with patch("st_name_ranking.commands.cli.get_stats") as mock_stats:
        mock_stats.return_value = DatabaseStats(
            total_names=100,
            classified_names=50,
            unclassified_names=50,
            rated_names=75,
            origin_distribution={"International": 80, "European": 20},
        )

        result = cli_runner.invoke(app, ["db", "stats"])
        assert result.exit_code == 0
        assert "Database Statistics" in result.output
        assert "100" in result.output  # total names
        assert "75" in result.output  # total ratings
        assert "50" in result.output  # classified names

        # Check that stats was called
        mock_stats.assert_called_once()


def test_cli_init_integration(initialized_db, mock_submodule_path, cli_runner):
    """Test CLI init command with real database and mocked submodule."""

    # Mock stats to avoid division by zero
    mock_stats = DatabaseStats(
        total_names=100,
        classified_names=20,
        unclassified_names=80,
        rated_names=75,
        origin_distribution={"International": 80, "European": 20},
    )

    # Ensure the database path is set to our temporary path (already done by initialized_db)
    # Patch sync_names_with_submodule to use our mock submodule path
    with (
        patch("st_name_ranking.commands.cli.sync_names_with_submodule") as mock_sync,
        patch("st_name_ranking.commands.cli.classify_all_names") as mock_classify,
        patch("st_name_ranking.commands.cli.get_stats", return_value=mock_stats) as mock_get_stats,
    ):
        mock_sync.return_value = 3  # Simulate 3 names synced
        mock_classify.return_value = 0  # No classification

        result = cli_runner.invoke(app, ["db", "init"])
        assert result.exit_code == 0
        assert "Initializing Name Ranking Database" in result.output
        assert "Database schema created" in result.output
        assert "Synced 3 new names from submodule" in result.output

        # Verify sync was called with default path
        mock_sync.assert_called_once()
        mock_get_stats.assert_called_once()

    # Additional test: init with --classify flag
    with (
        patch("st_name_ranking.commands.cli.sync_names_with_submodule") as mock_sync,
        patch("st_name_ranking.commands.cli.classify_all_names") as mock_classify,
        patch("st_name_ranking.commands.cli.get_stats", return_value=mock_stats) as mock_get_stats,
    ):
        mock_sync.return_value = 3
        mock_classify.return_value = 2
        result = cli_runner.invoke(app, ["db", "init", "--classify"])
        assert result.exit_code == 0
        assert "Classified 2 names" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
