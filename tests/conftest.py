"""
Pytest fixtures for st_name_ranking tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


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
def initialized_db(mock_db_path):
    """Initialize a fresh database with schema."""
    from st_name_ranking.database import get_connection, init_database

    init_database()

    # Verify tables exist
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "names" in tables
        assert "ratings" in tables
        assert "region_mapping" in tables

    return mock_db_path


@pytest.fixture
def sample_names_data():
    """Sample names data for testing."""
    return [
        {"name": "Anna", "gender": "F", "name_type": "first"},
        {"name": "Peter", "gender": "M", "name_type": "first"},
        {"name": "Maria", "gender": "F", "name_type": "first"},
        {"name": "Jens", "gender": "M", "name_type": "first"},
    ]


@pytest.fixture
def sample_ratings_data():
    """Sample ratings data for testing."""
    return [
        {
            "name": "Anna",
            "rating": 1500,
            "matches": 10,
            "wins": 5,
            "losses": 3,
            "draws": 2,
        },
        {
            "name": "Peter",
            "rating": 1400,
            "matches": 8,
            "wins": 4,
            "losses": 3,
            "draws": 1,
        },
    ]


@pytest.fixture
def mock_submodule_path(tmp_path):
    """Create a mock submodule directory structure."""
    submodule_path = tmp_path / "godkendtefornavne"
    submodule_path.mkdir()

    # Create mock CSV files (for other uses)
    boys_file = submodule_path / "drengenavne.csv"
    girls_file = submodule_path / "pigenavne.csv"

    # Write simple CSV content
    boys_file.write_text("name\nPeter\nJens\nLars\n")
    girls_file.write_text("name\nAnna\nMaria\nIda\n")

    # Create mock JSON file for sync_names_with_submodule
    json_file = submodule_path / "allenavne.json"
    json_data = [
        {"name": "Anna", "gender": "F", "name_type": "first"},
        {"name": "Peter", "gender": "M", "name_type": "first"},
        {"name": "Maria", "gender": "F", "name_type": "first"},
    ]
    import json

    json_file.write_text(json.dumps(json_data, indent=2))

    return submodule_path


@pytest.fixture
def mock_classifier():
    """Mock the ethnidata classifier to avoid PyTorch issues."""
    with patch("st_name_ranking.classify_origins.get_classifier") as mock_get:
        mock_classifier = mock_get.return_value
        mock_classifier.predict_nationality.return_value = {
            "country_name": "Denmark",
            "confidence": 0.85,
            "country": "DK",
            "region": "Europe",
        }
        yield mock_classifier


@pytest.fixture(autouse=True)
def cleanup_db_state():
    """Clean up database state before each test."""
    from st_name_ranking import database

    # Reset initialization flag before each test
    database._initialized = False
    yield
    # Reset after test
    database._initialized = False
