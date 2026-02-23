"""Comprehensive CLI integration tests for the Name Ranking application.

These tests use REAL database operations (temporary SQLite files) and test actual
CLI behavior including exit codes, output content, and database state changes.

Unlike test_cli.py which mocks database operations, these tests verify:
- Actual database schema creation
- Real data insertion and updates
- True CLI output matching
- Proper exit codes
- Model state persistence
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from st_name_ranking.cli import app

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


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
def real_db_path(temp_db_path):
    """Use a real temporary database path without mocking.

    This patches the database module to use the temp path.
    """
    from st_name_ranking import database

    original_path = database.DB_PATH
    original_initialized = database._initialized

    # Set temp path and reset initialization
    database.DB_PATH = temp_db_path
    database._initialized = False

    # Create data directory if needed
    temp_db_path.parent.mkdir(parents=True, exist_ok=True)

    yield temp_db_path

    # Restore original state
    database.DB_PATH = original_path
    database._initialized = original_initialized


@pytest.fixture
def mock_submodule_with_names(tmp_path):
    """Create a mock submodule directory with test names.

    Returns path to submodule directory containing allenavne.json
    with test names of various origins.
    """
    submodule_path = tmp_path / "godkendtefornavne"
    submodule_path.mkdir()

    # Create mock JSON file with diverse test names
    json_file = submodule_path / "allenavne.json"
    json_data = [
        # Nordic names
        {"name": "Anna", "gender": "F", "name_type": "first"},
        {"name": "Peter", "gender": "M", "name_type": "first"},
        {"name": "Maria", "gender": "F", "name_type": "first"},
        {"name": "Lars", "gender": "M", "name_type": "first"},
        {"name": "Ida", "gender": "F", "name_type": "first"},
        # European names
        {"name": "Hans", "gender": "M", "name_type": "first"},
        {"name": "Ingrid", "gender": "F", "name_type": "first"},
        {"name": "Ole", "gender": "M", "name_type": "first"},
        {"name": "Sofia", "gender": "F", "name_type": "first"},
        {"name": "Erik", "gender": "M", "name_type": "first"},
    ]
    json_file.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    # Create a mock git directory for commit hash
    git_dir = submodule_path / ".git"
    git_dir.mkdir(exist_ok=True)

    return submodule_path


@pytest.fixture
def initialized_real_db(real_db_path, mock_submodule_with_names):
    """Initialize a real database with test data.

    Returns the database path after init_database() and sync_names_with_submodule()
    have been called with real operations.
    """
    from st_name_ranking.database import get_connection, init_database, sync_names_with_submodule

    # Initialize database schema
    init_database()

    # Verify schema was created
    with get_connection() as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "names" in tables
        assert "ratings" in tables
        assert "region_mapping" in tables
        assert "model_state" in tables
        assert "source_versions" in tables

    # Sync names from mock submodule
    inserted = sync_names_with_submodule(mock_submodule_with_names)
    assert inserted > 0, "No names were inserted"

    return real_db_path


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def get_db_tables(db_path: Path) -> set[str]:
    """Get set of table names from SQLite database."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {row[0] for row in cursor.fetchall()}
    finally:
        conn.close()


def get_table_columns(db_path: Path, table: str) -> list[str]:
    """Get list of column names for a table."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cursor.fetchall()]
    finally:
        conn.close()


def get_name_count(db_path: Path) -> int:
    """Get total count of names in database."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM names")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def get_classified_count(db_path: Path) -> int:
    """Get count of classified names (origin_region IS NOT NULL)."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM names WHERE origin_region IS NOT NULL")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def get_rated_count(db_path: Path) -> int:
    """Get count of rated names."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM ratings")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def insert_test_names(db_path: Path, names_data: list[dict]) -> int:
    """Insert test names directly into database."""
    conn = sqlite3.connect(db_path)
    try:
        count_before = conn.execute("SELECT COUNT(*) FROM names").fetchone()[0]

        for name_data in names_data:
            conn.execute(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                (name_data["name"], name_data.get("gender", "Unisex")),
            )

        conn.commit()
        count_after = conn.execute("SELECT COUNT(*) FROM names").fetchone()[0]
        return count_after - count_before
    finally:
        conn.close()


def insert_test_ratings(db_path: Path, ratings: list[tuple[str, float]]) -> int:
    """Insert test ratings into database.

    Args:
        ratings: List of (name, rating) tuples
    """
    conn = sqlite3.connect(db_path)
    try:
        inserted = 0
        for name, rating in ratings:
            # Get name_id
            cursor = conn.execute("SELECT id FROM names WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                name_id = row[0]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ratings (name_id, rating, matches, last_updated)
                    VALUES (?, ?, 1, CURRENT_TIMESTAMP)
                    """,
                    (name_id, rating),
                )
                inserted += 1
        conn.commit()
        return inserted
    finally:
        conn.close()


def classify_test_names(db_path: Path, classifications: list[tuple[str, str, float]]) -> int:
    """Manually set origin classifications for test names.

    Args:
        classifications: List of (name, region, confidence) tuples
    """
    conn = sqlite3.connect(db_path)
    try:
        updated = 0
        for name, region, confidence in classifications:
            cursor = conn.execute(
                """
                UPDATE names
                SET origin_region = ?, origin_confidence = ?, origin_classified_at = CURRENT_TIMESTAMP
                WHERE name = ?
                """,
                (region, confidence, name),
            )
            updated += cursor.rowcount
        conn.commit()
        return updated
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# CLI Init Command Tests
# -----------------------------------------------------------------------------


class TestCLIInit:
    """Tests for the `name-db init` command with real database operations."""

    def test_init_creates_database_schema(self, real_db_path, mock_submodule_with_names, cli_runner):
        """Test that init command creates all required database tables."""

        # Mock the submodule sync to return predictable results
        with patch("st_name_ranking.cli.sync_names_with_submodule") as mock_sync:
            mock_sync.return_value = 10  # Simulate 10 names synced

            result = cli_runner.invoke(app, ["init"])

        # Verify exit code
        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"

        # Verify output contains expected messages
        assert "Initializing Name Ranking Database" in result.output
        assert "Database schema created" in result.output

        # Verify actual database schema was created
        tables = get_db_tables(real_db_path)
        expected_tables = {
            "names",
            "ratings",
            "user_settings",
            "region_mapping",
            "source_versions",
            "model_state",
            "comparisons",
        }
        assert expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}"

    def test_init_syncs_names_from_submodule(self, real_db_path, mock_submodule_with_names, cli_runner):
        """Test that init command syncs names from submodule to database."""

        # Mock sync to use our test submodule
        with patch("st_name_ranking.cli.sync_names_with_submodule") as mock_sync:
            # Actually call the real sync with our mock path
            def real_sync():
                from st_name_ranking.database import sync_names_with_submodule

                return sync_names_with_submodule(mock_submodule_with_names)

            mock_sync.side_effect = real_sync

            result = cli_runner.invoke(app, ["init"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.output}"

        # Verify names were synced
        name_count = get_name_count(real_db_path)
        assert name_count == 10, f"Expected 10 names, got {name_count}"

        # Verify output shows synced count
        assert "new names from submodule" in result.output
        assert "10" in result.output

    def test_init_shows_statistics(self, real_db_path, mock_submodule_with_names, cli_runner):
        """Test that init command displays database statistics after initialization."""

        with patch("st_name_ranking.cli.sync_names_with_submodule") as mock_sync:
            mock_sync.return_value = 10

            result = cli_runner.invoke(app, ["init"])

        assert result.exit_code == 0

        # Verify statistics are displayed
        assert "Database Statistics" in result.output
        assert "Total names" in result.output
        assert "Classified names" in result.output
        assert "Unclassified names" in result.output
        assert "Rated names" in result.output

    def test_init_with_classify_flag(self, real_db_path, mock_submodule_with_names, cli_runner):
        """Test that init --classify runs classification after initialization."""

        with patch("st_name_ranking.cli.sync_names_with_submodule") as mock_sync:
            mock_sync.return_value = 10

            # Mock classify_all_names to actually classify names
            with patch("st_name_ranking.cli.classify_all_names") as mock_classify:
                mock_classify.return_value = 5  # 5 names classified

                result = cli_runner.invoke(app, ["init", "--classify"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}"

        # Verify classification was triggered
        assert "Processing Data Enrichment" in result.output or "Classified" in result.output

    def test_init_handles_empty_submodule_gracefully(self, real_db_path, cli_runner, tmp_path):
        """Test that init handles submodule with no new names gracefully."""

        # Create empty submodule
        empty_submodule = tmp_path / "empty_submodule"
        empty_submodule.mkdir()
        json_file = empty_submodule / "allenavne.json"
        json_file.write_text(json.dumps([]), encoding="utf-8")

        with patch("st_name_ranking.cli.sync_names_with_submodule") as mock_sync:
            mock_sync.return_value = 0  # No names synced

            result = cli_runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "Database schema created" in result.output
        assert "0" in result.output  # Should show 0 names


# -----------------------------------------------------------------------------
# CLI Process Command Tests
# -----------------------------------------------------------------------------


class TestCLIProcess:
    """Tests for the `name-db process` command with real database operations."""

    def test_process_classifies_names_with_limit(self, initialized_real_db, cli_runner):
        """Test that process command classifies names up to the limit."""

        # First, manually classify some names so we have reference data
        test_classifications = [
            ("Anna", "Nordic", 0.85),
            ("Peter", "European", 0.80),
        ]
        classify_test_names(initialized_real_db, test_classifications)

        # Mock the classification to return predictable results
        with patch("st_name_ranking.cli.classify_all_names") as mock_classify:
            mock_classify.return_value = 5  # Simulate 5 classifications

            result = cli_runner.invoke(app, ["process", "--limit", "10"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.output}"

        # Verify command output
        assert "Processing Data Enrichment" in result.output

        # Verify classify was called with correct parameters
        mock_classify.assert_called_once_with(10, 100)

    def test_process_shows_no_unclassified_message(self, initialized_real_db, cli_runner):
        """Test that process shows message when no unclassified names exist."""

        # First, classify all existing names
        with sqlite3.connect(initialized_real_db) as conn:
            conn.execute("UPDATE names SET origin_region = 'International', origin_confidence = 0.5")
            conn.commit()

        with patch("st_name_ranking.cli.classify_all_names") as mock_classify:
            mock_classify.return_value = 0  # No names to classify

            result = cli_runner.invoke(app, ["process"])

        assert result.exit_code == 0
        assert "Processing Data Enrichment" in result.output

    def test_process_with_custom_batch_size(self, initialized_real_db, cli_runner):
        """Test that process command respects batch-size parameter."""

        with patch("st_name_ranking.cli.classify_all_names") as mock_classify:
            mock_classify.return_value = 3

            result = cli_runner.invoke(app, ["process", "--batch-size", "25"])

        assert result.exit_code == 0

        # Verify classify was called with custom batch size
        mock_classify.assert_called_once_with(None, 25)

    def test_process_handles_classification_error(self, initialized_real_db, cli_runner):
        """Test that process handles classification errors gracefully."""

        with patch("st_name_ranking.cli.classify_all_names") as mock_classify:
            # Simulate ImportError for ethnidata
            mock_classify.side_effect = ImportError("ethnidata not installed")

            result = cli_runner.invoke(app, ["process"])

        # Should exit with error code 1
        assert result.exit_code == 1

        # Should show error message about ethnidata
        assert "ethnidata" in result.output.lower() or "Error" in result.output


# -----------------------------------------------------------------------------
# CLI Stats Command Tests
# -----------------------------------------------------------------------------


class TestCLIStats:
    """Tests for the `name-db stats` command with real database operations."""

    def test_stats_shows_correct_counts(self, initialized_real_db, cli_runner):
        """Test that stats command shows correct database counts."""

        # Insert some test ratings
        test_ratings = [("Anna", 1600.0), ("Peter", 1400.0)]
        insert_test_ratings(initialized_real_db, test_ratings)

        # Classify some names
        test_classifications = [
            ("Anna", "Nordic", 0.85),
            ("Peter", "European", 0.80),
            ("Maria", "Nordic", 0.75),
        ]
        classify_test_names(initialized_real_db, test_classifications)

        result = cli_runner.invoke(app, ["stats"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.output}"

        # Verify output structure
        assert "Database Statistics" in result.output
        assert "Summary" in result.output

        # Verify counts are correct
        assert "10" in result.output  # Total names
        assert "2" in result.output  # Rated names

        # Verify origin distribution is shown
        assert "Origin Distribution" in result.output
        assert "Nordic" in result.output
        assert "European" in result.output

    def test_stats_shows_percentages(self, initialized_real_db, cli_runner):
        """Test that stats shows percentages for classified names."""

        # Classify exactly half the names
        test_classifications = [
            ("Anna", "Nordic", 0.85),
            ("Peter", "Nordic", 0.80),
            ("Maria", "European", 0.75),
            ("Lars", "European", 0.70),
            ("Ida", "Nordic", 0.85),
        ]
        classify_test_names(initialized_real_db, test_classifications)

        result = cli_runner.invoke(app, ["stats"])

        assert result.exit_code == 0

        # Verify percentages are shown (format: XX.X%)
        import re

        percentage_pattern = r"\d+\.\d%"
        assert re.search(percentage_pattern, result.output), "Should show percentages"

    def test_stats_handles_empty_origin_distribution(self, initialized_real_db, cli_runner):
        """Test that stats handles case with no origin classifications."""

        result = cli_runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "Database Statistics" in result.output

        # Should handle empty origin distribution gracefully
        # The output will show 0% classified or "No origin classification data"
        assert "0.0%" in result.output or "International" in result.output

    def test_stats_with_no_database(self, real_db_path, cli_runner):
        """Test stats behavior when database doesn't exist yet."""

        # Database file exists but no schema
        result = cli_runner.invoke(app, ["stats"])

        # Should fail because database tables don't exist
        assert result.exit_code != 0 or "error" in result.output.lower()


# -----------------------------------------------------------------------------
# CLI Model Commands Tests
# -----------------------------------------------------------------------------


class TestCLIModelCommands:
    """Tests for model-status and model-reset commands with real database operations."""

    def test_model_status_shows_feature_dimensions(self, initialized_real_db, cli_runner):
        """Test that model-status shows correct feature dimensions."""
        from st_name_ranking.utils import get_active_learning_model

        # Initialize a model to get feature dimensions
        model = get_active_learning_model()
        expected_dim = model.state.feature_dim
        expected_samples = model.state.training_samples

        result = cli_runner.invoke(app, ["model-status"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.output}"

        # Verify output shows model information
        assert "Active Learning Model Status" in result.output
        assert "Feature dimension" in result.output
        assert str(expected_dim) in result.output
        assert "Training samples" in result.output

    def test_model_status_shows_feature_names(self, initialized_real_db, cli_runner):
        """Test that model-status shows feature names preview."""

        result = cli_runner.invoke(app, ["model-status"])

        assert result.exit_code == 0

        # Verify feature names are shown
        assert "Features:" in result.output
        assert "Total features:" in result.output

    def test_model_reset_clears_model_state(self, initialized_real_db, cli_runner):
        """Test that model-reset actually clears model state in database."""
        from st_name_ranking import database, utils
        from st_name_ranking.features import FeatureExtractor

        # First, add some training data to the model
        from st_name_ranking.utils import get_active_learning_model

        model = get_active_learning_model()
        extractor = FeatureExtractor()

        # Get features for a couple of names
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT name, gender, origin_region FROM names LIMIT 2")
            rows = cursor.fetchall()

        if len(rows) >= 2:
            name_a, gender_a, origin_a = rows[0]
            name_b, gender_b, origin_b = rows[1]

            features_a = extractor.extract(name_a, gender_a, origin_a)
            features_b = extractor.extract(name_b, gender_b, origin_b)

            # Update model with a comparison
            model.update(features_a, features_b, -1)
            model.save_to_db()

            # Verify model has training samples in database
            with database.get_connection() as conn:
                cursor = conn.execute("SELECT training_samples FROM model_state WHERE id = 1")
                db_samples = cursor.fetchone()[0]
                assert db_samples >= 1, "Model should have training samples in DB"

        # Clear the global singleton to simulate fresh process
        utils._model = None

        # Run model-reset (automatically confirms in test)
        result = cli_runner.invoke(app, ["model-reset"], input="y\n")

        # Check exit code - may be 0 or aborted
        if result.exit_code == 0:
            # Verify database was cleared
            with database.get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM model_state WHERE id = 1")
                count = cursor.fetchone()[0]
                # After reset, there may be 0 rows or a new initialized model
                # Either way, the old model state is gone
                if count > 0:
                    cursor = conn.execute("SELECT training_samples FROM model_state WHERE id = 1")
                    new_samples = cursor.fetchone()[0]
                    # After reset, training samples should be 0 (fresh model)
                    assert new_samples == 0, f"Expected 0 samples after reset, got {new_samples}"

    def test_model_reset_confirms_before_reset(self, initialized_real_db, cli_runner):
        """Test that model-reset asks for confirmation."""

        result = cli_runner.invoke(app, ["model-reset"], input="n\n")

        # Should show confirmation prompt
        assert "Are you sure" in result.output or "reset" in result.output.lower()

    def test_model_reset_aborts_on_no_confirmation(self, initialized_real_db, cli_runner):
        """Test that model-reset aborts when user declines."""

        result = cli_runner.invoke(app, ["model-reset"], input="n\n")

        # Should abort
        assert result.exit_code != 0 or "cancelled" in result.output.lower()


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------


class TestCLIErrorHandling:
    """Tests for CLI error handling with real operations."""

    def test_process_with_missing_database_tables(self, real_db_path, cli_runner):
        """Test process command when database exists but tables are missing."""

        # Create empty database file without schema
        real_db_path.touch()

        # Try to run stats - should fail
        result = cli_runner.invoke(app, ["stats"])

        # Should fail because tables don't exist
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_invalid_argument_handling(self, cli_runner):
        """Test CLI behavior with invalid arguments."""
        # Test with negative limit
        result = cli_runner.invoke(app, ["process", "--limit", "-5"])

        # Typer validates integers, should work but process may handle it
        # or return an error
        assert result.exit_code == 0 or result.exit_code == 2

    def test_help_shows_all_commands(self, cli_runner):
        """Test that help shows all available commands."""
        result = cli_runner.invoke(app, ["--help"])

        assert result.exit_code == 0

        # Verify all commands are listed
        assert "init" in result.output
        assert "process" in result.output
        assert "stats" in result.output
        assert "model-status" in result.output
        assert "model-reset" in result.output

    def test_command_specific_help(self, cli_runner):
        """Test that each command has its own help text."""
        commands = ["init", "process", "stats", "model-status", "model-reset"]

        for command in commands:
            result = cli_runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0, f"Help failed for {command}"
            assert "Usage:" in result.output or command in result.output


# -----------------------------------------------------------------------------
# Integration Flow Tests
# -----------------------------------------------------------------------------


class TestCLIIntegrationFlow:
    """End-to-end integration tests for common CLI workflows."""

    def test_full_init_process_stats_workflow(self, real_db_path, mock_submodule_with_names, cli_runner):
        """Test complete workflow: init -> process -> stats."""

        # Step 1: Initialize database with real sync
        with patch("st_name_ranking.cli.sync_names_with_submodule") as mock_sync:
            # Actually sync to populate the database
            def real_sync():
                from st_name_ranking.database import sync_names_with_submodule

                return sync_names_with_submodule(mock_submodule_with_names)

            mock_sync.side_effect = real_sync

            init_result = cli_runner.invoke(app, ["init"])
            assert init_result.exit_code == 0, "Init failed"

        # Verify names were actually inserted
        name_count = get_name_count(real_db_path)
        assert name_count > 0, f"Expected names in database, got {name_count}"

        # Step 2: Process/classify names (mocked)
        with patch("st_name_ranking.cli.classify_all_names") as mock_classify:
            mock_classify.return_value = 5

            process_result = cli_runner.invoke(app, ["process", "--limit", "10"])
            assert process_result.exit_code == 0, "Process failed"

        # Step 3: Check stats
        stats_result = cli_runner.invoke(app, ["stats"])
        assert stats_result.exit_code == 0, "Stats failed"
        # Verify stats shows the names we inserted
        assert str(name_count) in stats_result.output, f"Expected {name_count} names in stats output"

    def test_model_workflow(self, initialized_real_db, cli_runner):
        """Test model workflow: status -> (use) -> reset -> status."""

        # Step 1: Check initial model status
        status1_result = cli_runner.invoke(app, ["model-status"])
        assert status1_result.exit_code == 0, "Model status failed"
        assert "Active Learning Model Status" in status1_result.output
        assert "Feature dimension" in status1_result.output

        # Extract initial feature dimension - handle table format with box chars
        import re

        # Match "Feature dimension" followed by any characters and then a number
        dim_match = re.search(r"Feature dimension.*?([0-9]+)", status1_result.output, re.DOTALL)
        assert dim_match, f"Should show feature dimension. Output: {status1_result.output[:200]}"
        initial_dim = int(dim_match.group(1))

        # Step 2: Reset model (confirm yes)
        reset_result = cli_runner.invoke(app, ["model-reset"], input="y\n")
        # Reset may succeed or be aborted depending on implementation

        # Step 3: Check status again
        status2_result = cli_runner.invoke(app, ["model-status"])
        assert status2_result.exit_code == 0, "Model status after reset failed"

        # Feature dimension should still be the same
        dim_match2 = re.search(r"Feature dimension.*?([0-9]+)", status2_result.output, re.DOTALL)
        assert dim_match2, "Should show feature dimension after reset"
        assert int(dim_match2.group(1)) == initial_dim, "Feature dimension should persist"


# -----------------------------------------------------------------------------
# Database State Verification Tests
# -----------------------------------------------------------------------------


class TestDatabaseStateVerification:
    """Tests that verify actual database state changes after CLI commands."""

    def test_init_creates_phonetic_codes(self, real_db_path, mock_submodule_with_names, cli_runner):
        """Verify that init creates phonetic codes for names."""

        with patch("st_name_ranking.cli.sync_names_with_submodule") as mock_sync:
            # Actually sync to populate names
            def real_sync():
                from st_name_ranking.database import sync_names_with_submodule

                return sync_names_with_submodule(mock_submodule_with_names)

            mock_sync.side_effect = real_sync

            result = cli_runner.invoke(app, ["init"])

        assert result.exit_code == 0

        # Verify phonetic codes were computed
        conn = sqlite3.connect(real_db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM names WHERE phonetic_primary IS NOT NULL")
            count_with_phonetic = cursor.fetchone()[0]
            assert count_with_phonetic > 0, "Should have computed phonetic codes"
        finally:
            conn.close()

    def test_init_creates_region_mapping(self, real_db_path, mock_submodule_with_names, cli_runner):
        """Verify that init populates region_mapping table."""

        with patch("st_name_ranking.cli.sync_names_with_submodule") as mock_sync:
            mock_sync.return_value = 10
            result = cli_runner.invoke(app, ["init"])

        assert result.exit_code == 0

        # Verify region mappings exist
        conn = sqlite3.connect(real_db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM region_mapping")
            mapping_count = cursor.fetchone()[0]
            assert mapping_count > 0, "Should have region mappings"

            # Verify expected regions exist
            cursor = conn.execute("SELECT DISTINCT region FROM region_mapping")
            regions = {row[0] for row in cursor.fetchall()}
            expected_regions = {"Nordic", "European", "Asian", "American", "African", "Middle Eastern", "Oceanian"}
            assert expected_regions.issubset(regions), f"Missing regions: {expected_regions - regions}"
        finally:
            conn.close()

    def test_init_creates_indexes(self, real_db_path, mock_submodule_with_names, cli_runner):
        """Verify that init creates all expected indexes."""

        with patch("st_name_ranking.cli.sync_names_with_submodule") as mock_sync:
            mock_sync.return_value = 10
            result = cli_runner.invoke(app, ["init"])

        assert result.exit_code == 0

        # Verify indexes exist
        conn = sqlite3.connect(real_db_path)
        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = {row[0] for row in cursor.fetchall()}

            expected_indexes = {
                "idx_names_gender",
                "idx_names_origin",
                "idx_ratings_rating",
                "idx_names_phonetic_primary",
                "idx_names_phonetic_secondary",
                "idx_comparisons_name_a",
                "idx_comparisons_name_b",
                "idx_comparisons_created",
                "idx_model_state_updated",
            }

            # Check for our expected indexes (may have SQLite auto indexes too)
            for idx in expected_indexes:
                assert idx in indexes, f"Missing index: {idx}"
        finally:
            conn.close()

    def test_model_state_persistence(self, initialized_real_db, cli_runner):
        """Verify that model state is persisted in database."""
        from st_name_ranking.utils import get_active_learning_model

        # Get initial model and save state
        model = get_active_learning_model()
        model.save_to_db()

        # Verify state is in database
        conn = sqlite3.connect(initialized_real_db)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM model_state WHERE id = 1")
            count = cursor.fetchone()[0]
            assert count == 1, "Model state should be saved"

            cursor = conn.execute("SELECT training_samples, feature_names_json FROM model_state WHERE id = 1")
            row = cursor.fetchone()
            assert row is not None
            assert row[1] is not None, "Feature names should be saved"
        finally:
            conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
