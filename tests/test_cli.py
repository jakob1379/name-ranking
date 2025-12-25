"""
CLI integration tests for the Name Ranking application.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from st_name_ranking.cli import app


@pytest.fixture
def cli_runner():
    """Fixture for CLI testing."""
    return CliRunner()


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
def temp_ratings_path():
    """Create a temporary ratings.json file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        ratings_path = Path(f.name)
        # Write sample ratings
        sample_ratings = {
            "Anna": {"rating": 1500, "matches": 10},
            "Peter": {"rating": 1400, "matches": 8},
        }
        f.write(json.dumps(sample_ratings).encode())
    yield ratings_path
    # Clean up
    if ratings_path.exists():
        ratings_path.unlink()


@pytest.fixture
def mock_db_path(temp_db_path):
    """Mock the database path in st_name_ranking.database."""
    from st_name_ranking import database
    original_path = database.DB_PATH
    
    # Patch the DB_PATH
    database.DB_PATH = temp_db_path
    # Reset initialization flag
    database._initialized = False
    
    yield temp_db_path
    
    # Restore original path
    database.DB_PATH = original_path
    database._initialized = False


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
    assert "Commands:" in result.output
    assert "init" in result.output
    assert "sync" in result.output
    assert "migrate" in result.output
    assert "classify" in result.output
    assert "stats" in result.output


def test_cli_init_basic(mock_db_path, cli_runner):
    """Test basic database initialization."""
    with patch('st_name_ranking.database.get_submodule_path') as mock_get_path:
        # Mock submodule path to avoid actual submodule dependency
        mock_get_path.return_value = Path("/mock/path")
        
        result = cli_runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "Initializing name ranking database" in result.output
        assert "Database initialized successfully" in result.output
        
        # Check that database file was created
        assert mock_db_path.exists()
        
        # Check that tables were created
        import sqlite3
        conn = sqlite3.connect(mock_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        assert 'names' in tables
        assert 'ratings' in tables
        assert 'region_mapping' in tables


def test_cli_init_with_classify(mock_db_path, cli_runner):
    """Test database initialization with classification."""
    with patch('st_name_ranking.database.get_submodule_path') as mock_get_path, \
         patch('st_name_ranking.cli.classify_all_names') as mock_classify:
        
        mock_get_path.return_value = Path("/mock/path")
        mock_classify.return_value = (3, 0)  # 3 names classified, 0 errors
        
        result = cli_runner.invoke(app, ["init", "--classify"])
        assert result.exit_code == 0
        assert "Database initialized successfully" in result.output
        assert "Running initial origin classification" in result.output
        assert "Classification complete" in result.output
        
        # Check that classify was called
        mock_classify.assert_called_once()


def test_cli_init_with_custom_ratings(mock_db_path, temp_ratings_path, cli_runner):
    """Test database initialization with custom ratings path."""
    with patch('st_name_ranking.database.get_submodule_path') as mock_get_path, \
         patch('st_name_ranking.database.migrate_ratings_from_json') as mock_migrate:
        
        mock_get_path.return_value = Path("/mock/path")
        mock_migrate.return_value = 2  # 2 ratings migrated
        
        result = cli_runner.invoke(app, [
            "init", 
            "--ratings-path", str(temp_ratings_path)
        ])
        assert result.exit_code == 0
        assert "Database initialized successfully" in result.output
        assert "Migrating ratings" in result.output
        
        # Check that migrate was called with correct path
        mock_migrate.assert_called_once_with(temp_ratings_path)


def test_cli_sync(mock_db_path, mock_submodule_path, cli_runner):
    """Test sync command."""
    # First initialize the database
    with patch('st_name_ranking.database.get_submodule_path') as mock_get_path:
        mock_get_path.return_value = mock_submodule_path
        
        # Initialize
        init_result = cli_runner.invoke(app, ["init"])
        assert init_result.exit_code == 0
    
    # Now test sync
    with patch('st_name_ranking.database.get_submodule_path') as mock_get_path, \
         patch('st_name_ranking.database.sync_names_with_submodule') as mock_sync:
        
        mock_get_path.return_value = mock_submodule_path
        mock_sync.return_value = (2, 1)  # 2 new names, 1 existing
        
        result = cli_runner.invoke(app, ["sync"])
        assert result.exit_code == 0
        assert "Syncing Names from Submodule" in result.output
        assert "2 new names added" in result.output
        assert "1 existing names skipped" in result.output
        
        # Check that sync was called
        mock_sync.assert_called_once()


def test_cli_migrate(mock_db_path, temp_ratings_path, cli_runner):
    """Test migrate command."""
    # First initialize the database
    with patch('st_name_ranking.database.get_submodule_path') as mock_get_path:
        mock_get_path.return_value = Path("/mock/path")
        init_result = cli_runner.invoke(app, ["init"])
        assert init_result.exit_code == 0
    
    # Test migrate
    with patch('st_name_ranking.database.migrate_ratings_from_json') as mock_migrate:
        mock_migrate.return_value = 3  # 3 ratings migrated
        
        result = cli_runner.invoke(app, ["migrate", "--ratings-path", str(temp_ratings_path)])
        assert result.exit_code == 0
        assert "Migrating Ratings from JSON" in result.output
        assert "3 ratings migrated" in result.output
        
        # Check that migrate was called with correct path
        mock_migrate.assert_called_once_with(temp_ratings_path)


def test_cli_classify(mock_db_path, cli_runner):
    """Test classify command."""
    # First initialize the database and insert some names
    with patch('st_name_ranking.database.get_submodule_path') as mock_get_path, \
         patch('st_name_ranking.database.insert_names_via_sql') as mock_insert:
        
        mock_get_path.return_value = Path("/mock/path")
        
        # Initialize
        init_result = cli_runner.invoke(app, ["init"])
        assert init_result.exit_code == 0
    
    # Test classify
    with patch('st_name_ranking.cli.classify_all_names') as mock_classify:
        mock_classify.return_value = (10, 0)  # 10 names classified, 0 errors
        
        result = cli_runner.invoke(app, ["classify", "--limit", "50"])
        assert result.exit_code == 0
        assert "Classifying Name Origins" in result.output
        assert "10 names classified" in result.output
        assert "0 classification errors" in result.output
        
        # Check that classify was called with limit
        mock_classify.assert_called_once_with(limit=50, batch_size=100)


def test_cli_classify_with_batch_size(mock_db_path, cli_runner):
    """Test classify command with custom batch size."""
    # First initialize the database
    with patch('st_name_ranking.database.get_submodule_path') as mock_get_path:
        mock_get_path.return_value = Path("/mock/path")
        init_result = cli_runner.invoke(app, ["init"])
        assert init_result.exit_code == 0
    
    # Test classify with batch size
    with patch('st_name_ranking.cli.classify_all_names') as mock_classify:
        mock_classify.return_value = (5, 1)  # 5 names classified, 1 error
        
        result = cli_runner.invoke(app, ["classify", "--batch-size", "20"])
        assert result.exit_code == 0
        assert "Classifying Name Origins" in result.output
        assert "5 names classified" in result.output
        assert "1 classification errors" in result.output
        
        # Check that classify was called with batch size
        mock_classify.assert_called_once_with(limit=None, batch_size=20)


def test_cli_stats(mock_db_path, cli_runner):
    """Test stats command."""
    # First initialize the database
    with patch('st_name_ranking.database.get_submodule_path') as mock_get_path:
        mock_get_path.return_value = Path("/mock/path")
        init_result = cli_runner.invoke(app, ["init"])
        assert init_result.exit_code == 0
    
    # Test stats
    with patch('st_name_ranking.database.get_stats') as mock_stats:
        mock_stats.return_value = {
            'total_names': 100,
            'total_ratings': 75,
            'classified_names': 50,
            'unclassified_names': 50,
            'avg_rating': 1500.5,
            'highest_rating': 1800.0,
            'lowest_rating': 1200.0,
            'total_matches': 1000,
        }
        
        result = cli_runner.invoke(app, ["stats"])
        assert result.exit_code == 0
        assert "Database Statistics" in result.output
        assert "100" in result.output  # total names
        assert "75" in result.output   # total ratings
        assert "50" in result.output   # classified names
        
        # Check that stats was called
        mock_stats.assert_called_once()


def test_cli_init_with_invalid_ratings_path(mock_db_path, cli_runner):
    """Test initialization with invalid ratings path."""
    with patch('st_name_ranking.database.get_submodule_path') as mock_get_path:
        mock_get_path.return_value = Path("/mock/path")
        
        result = cli_runner.invoke(app, ["init", "--ratings-path", "/nonexistent/path"])
        # Should still succeed (ratings migration is optional)
        assert result.exit_code == 0
        assert "Database initialized successfully" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
