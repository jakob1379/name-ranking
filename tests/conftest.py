"""Pytest fixtures for st_name_ranking tests."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    original_path = database.get_db_path()

    database.set_db_path(temp_db_path)

    yield temp_db_path

    database.set_db_path(original_path)


@pytest.fixture
def initialized_db(mock_db_path):
    """Initialize a fresh database with schema."""
    from st_name_ranking.active_learning import selection
    from st_name_ranking.database import get_connection, init_database

    # Clear active-learning caches to prevent stale data between tests.
    selection.reset_active_learning_state()

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
    """Mock the ethnidata classifier to avoid missing database file."""
    from st_name_ranking import origin_classifier

    # Clear singleton cache to ensure fresh classifier
    origin_classifier.reset_classifier_cache()

    # The classifier expects a callable that returns (region, confidence) tuple
    mock_instance = MagicMock(return_value=("European", 0.85))
    with patch(
        "st_name_ranking.origin_classifier._create_ethnidata_classifier",
        return_value=mock_instance,
    ):
        yield mock_instance


# -----------------------------------------------------------------------------

# Integration test configuration
# -----------------------------------------------------------------------------


def pytest_addoption(parser):
    """Add command-line options for integration tests."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires running application)",
    )
    parser.addoption(
        "--run-playwright",
        action="store_true",
        default=False,
        help="Run Playwright tests (requires browser installation)",
    )
    parser.addoption(
        "--app-url",
        default="http://localhost:8501",
        help="URL of running Streamlit application for integration tests",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires running application)",
    )
    config.addinivalue_line(
        "markers",
        "playwright: mark test as Playwright test (requires browser)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration and Playwright tests unless explicitly requested."""
    skip_integration = pytest.mark.skip(reason="Need --run-integration option to run")
    skip_playwright = pytest.mark.skip(reason="Need --run-playwright option to run")

    for item in items:
        # Skip integration tests unless --run-integration is set
        if "integration" in item.keywords and not config.getoption("--run-integration"):
            item.add_marker(skip_integration)

        # Skip playwright tests unless --run-playwright is set
        if "playwright" in item.keywords and not config.getoption("--run-playwright"):
            item.add_marker(skip_playwright)


@pytest.fixture
def app_url(request):
    """Get the application URL from command line option."""
    return request.config.getoption("--app-url")
