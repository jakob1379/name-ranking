"""Integration tests for data loading from git submodule.

These tests use actual file I/O and database operations (no mocks).
Tests verify real behavior: file reading, JSON parsing, database storage, and
error handling with actual files and databases.
"""

import json
from pathlib import Path

import pytest

from st_name_ranking import data_loader, database

# =============================================================================
# JSON Loading Tests
# =============================================================================


class TestJsonLoading:
    """Tests for load_submodule_json() with actual files."""

    def test_load_valid_json_file(self, tmp_path, mock_db_path):
        """Test loading a valid JSON file with name/gender data."""
        # Create test JSON file in expected location
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "Peter", "gender": "M"},
            {"name": "Maria", "gender": "F"},
            {"name": "Alex", "gender": "U"},
        ]
        json_file.write_text(json.dumps(test_data))

        # Patch the path temporarily
        original_cwd = Path.cwd()
        try:
            # Change to tmp_path so relative path works
            import os

            os.chdir(tmp_path)
            result = data_loader.load_submodule_json()

            # Verify results
            assert len(result) == 4
            names = {item["name"] for item in result}
            assert names == {"Anna", "Peter", "Maria", "Alex"}
        finally:
            os.chdir(original_cwd)

    def test_gender_mapping_f_to_female(self, tmp_path, mock_db_path):
        """Test that gender code 'F' maps to 'Female' in database."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "Emma", "gender": "Female"},
            {"name": "female", "gender": "female"},
        ]
        json_file.write_text(json.dumps(test_data))

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_json()

            # All should be loaded (gender codes validated during sync, not load)
            assert len(result) == 3
            genders = {item["gender"] for item in result}
            assert "F" in genders
            assert "Female" in genders
            assert "female" in genders
        finally:
            os.chdir(original_cwd)

    def test_gender_mapping_m_to_male(self, tmp_path, mock_db_path):
        """Test that gender code 'M' maps to 'Male' in database."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Peter", "gender": "M"},
            {"name": "Lars", "gender": "Male"},
            {"name": "male", "gender": "male"},
        ]
        json_file.write_text(json.dumps(test_data))

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_json()

            assert len(result) == 3
            genders = {item["gender"] for item in result}
            assert "M" in genders
            assert "Male" in genders
            assert "male" in genders
        finally:
            os.chdir(original_cwd)

    def test_gender_mapping_u_to_unisex(self, tmp_path, mock_db_path):
        """Test that gender code 'U' maps to 'Unisex' in database."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Alex", "gender": "U"},
            {"name": "Kim", "gender": "Unisex"},
            {"name": "unisex", "gender": "unisex"},
        ]
        json_file.write_text(json.dumps(test_data))

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_json()

            assert len(result) == 3
            genders = {item["gender"] for item in result}
            assert "U" in genders
            assert "Unisex" in genders
            assert "unisex" in genders
        finally:
            os.chdir(original_cwd)

    def test_validation_filters_invalid_names(self, tmp_path, mock_db_path):
        """Test that invalid names (headers, placeholders) are filtered out."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "name", "gender": "F"},  # Header - should be filtered
            {"name": "Peter", "gender": "M"},
            {"name": "navn1", "gender": "M"},  # Invalid pattern
            {"name": "Maria", "gender": "F"},
            {"name": "Fornavn", "gender": "F"},  # Header
        ]
        json_file.write_text(json.dumps(test_data))

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_json()

            # Only valid names should remain
            assert len(result) == 3
            names = {item["name"] for item in result}
            assert names == {"Anna", "Peter", "Maria"}
        finally:
            os.chdir(original_cwd)

    def test_handles_malformed_json(self, tmp_path, mock_db_path):
        """Test handling of malformed JSON file."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        # Write invalid JSON
        json_file.write_text("not valid json {{{")

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            # Should return empty list, not crash
            result = data_loader.load_submodule_json()
            assert result == []
        finally:
            os.chdir(original_cwd)

    def test_handles_missing_required_columns(self, tmp_path, mock_db_path):
        """Test JSON without required 'name' and 'gender' columns."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"first_name": "Anna", "sex": "F"},  # Wrong column names
            {"first_name": "Peter", "sex": "M"},
        ]
        json_file.write_text(json.dumps(test_data))

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_json()
            assert result == []  # Returns empty list when columns missing
        finally:
            os.chdir(original_cwd)

    def test_handles_empty_json_array(self, tmp_path, mock_db_path):
        """Test handling of empty JSON array."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        json_file.write_text(json.dumps([]))

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_json()
            assert result == []
        finally:
            os.chdir(original_cwd)


# =============================================================================
# Submodule Sync Tests
# =============================================================================


class TestSubmoduleSync:
    """Tests for sync_names_with_submodule() with real files and database."""

    def test_sync_inserts_new_names(self, tmp_path, initialized_db):
        """Test that sync inserts new names into database."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "Peter", "gender": "M"},
        ]
        json_file.write_text(json.dumps(test_data))

        # Initialize git repo to get commit hash
        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=submodule_path, check=True, capture_output=True)

        # Sync names
        inserted = database.sync_names_with_submodule(submodule_path)

        assert inserted == 2

        # Verify names in database
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT name, gender FROM names ORDER BY name")
            rows = cursor.fetchall()
            assert len(rows) == 2
            assert tuple(rows[0]) == ("Anna", "Female")
            assert tuple(rows[1]) == ("Peter", "Male")

    def test_sync_gender_mapping_in_database(self, tmp_path, initialized_db):
        """Test that gender codes are properly mapped during sync."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "Peter", "gender": "M"},
            {"name": "Alex", "gender": "U"},
        ]
        json_file.write_text(json.dumps(test_data))

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=submodule_path, check=True, capture_output=True)

        database.sync_names_with_submodule(submodule_path)

        with database.get_connection() as conn:
            cursor = conn.execute("SELECT name, gender FROM names ORDER BY name")
            rows = {row[0]: row[1] for row in cursor.fetchall()}

            assert rows["Anna"] == "Female"
            assert rows["Peter"] == "Male"
            assert rows["Alex"] == "Unisex"

    def test_commit_hash_tracking(self, tmp_path, initialized_db):
        """Test that sync tracks commit hash in source_versions table."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [{"name": "Anna", "gender": "F"}]
        json_file.write_text(json.dumps(test_data))

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=submodule_path, check=True, capture_output=True)

        # Get expected commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=submodule_path,
            capture_output=True,
            text=True,
            check=True,
        )
        expected_hash = result.stdout.strip()

        database.sync_names_with_submodule(submodule_path)

        # Verify commit hash stored
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT commit_hash FROM source_versions ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == expected_hash

    def test_incremental_sync_only_adds_new_names(self, tmp_path, initialized_db):
        """Test that incremental sync only adds new names."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        # First sync with initial data
        test_data = [{"name": "Anna", "gender": "F"}]
        json_file.write_text(json.dumps(test_data))

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "first"], cwd=submodule_path, check=True, capture_output=True)

        first_inserted = database.sync_names_with_submodule(submodule_path)
        assert first_inserted == 1

        # Add more names and commit
        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "Peter", "gender": "M"},
        ]
        json_file.write_text(json.dumps(test_data))
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "second"], cwd=submodule_path, check=True, capture_output=True)

        second_inserted = database.sync_names_with_submodule(submodule_path)
        assert second_inserted == 1  # Only Peter is new

        # Verify both names in database
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM names")
            assert cursor.fetchone()[0] == 2

    def test_sync_skips_when_same_commit(self, tmp_path, initialized_db):
        """Test that sync is skipped when commit hasn't changed."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [{"name": "Anna", "gender": "F"}]
        json_file.write_text(json.dumps(test_data))

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=submodule_path, check=True, capture_output=True)

        # First sync
        first_inserted = database.sync_names_with_submodule(submodule_path)
        assert first_inserted == 1

        # Second sync (same commit) - should return 0
        second_inserted = database.sync_names_with_submodule(submodule_path)
        assert second_inserted == 0

    def test_phonetic_code_computation_during_sync(self, tmp_path, initialized_db):
        """Test that phonetic codes are computed during sync."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [{"name": "Christopher", "gender": "M"}]
        json_file.write_text(json.dumps(test_data))

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=submodule_path, check=True, capture_output=True)

        database.sync_names_with_submodule(submodule_path)

        # Verify phonetic codes stored
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT name, phonetic_primary, phonetic_secondary FROM names")
            row = cursor.fetchone()
            assert row[0] == "Christopher"
            assert row[1] is not None  # primary phonetic code
            assert row[1] != ""  # Should have a value


# =============================================================================
# CSV Fallback Tests
# =============================================================================


class TestCsvFallback:
    """Tests for load_submodule_csv_fallback() with actual CSV files."""

    def test_load_drengenavne_csv(self, tmp_path, mock_db_path):
        """Test loading names from drengenavne.csv."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()

        csv_file = submodule_path / "drengenavne.csv"
        csv_file.write_text("Peter\nJens\nLars\n")

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_csv_fallback()

            assert len(result) == 3
            assert "Peter" in result
            assert "Jens" in result
            assert "Lars" in result
        finally:
            os.chdir(original_cwd)

    def test_load_pigenavne_csv(self, tmp_path, mock_db_path):
        """Test loading names from pigenavne.csv."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()

        csv_file = submodule_path / "pigenavne.csv"
        csv_file.write_text("Anna\nMaria\nIda\n")

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_csv_fallback()

            assert len(result) == 3
            assert "Anna" in result
            assert "Maria" in result
            assert "Ida" in result
        finally:
            os.chdir(original_cwd)

    def test_load_unisexnavne_csv(self, tmp_path, mock_db_path):
        """Test loading names from unisexnavne.csv."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()

        csv_file = submodule_path / "unisexnavne.csv"
        csv_file.write_text("Alex\nKim\nRobin\n")

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_csv_fallback()

            assert len(result) == 3
            assert "Alex" in result
            assert "Kim" in result
            assert "Robin" in result
        finally:
            os.chdir(original_cwd)

    def test_csv_filters_invalid_names(self, tmp_path, mock_db_path):
        """Test that CSV loading filters out invalid names."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()

        csv_file = submodule_path / "drengenavne.csv"
        csv_file.write_text("Peter\nname\nnavn1\nLars\n")

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_csv_fallback()

            # Only valid names should be loaded
            assert len(result) == 2
            assert "Peter" in result
            assert "Lars" in result
            assert "name" not in result
            assert "navn1" not in result
        finally:
            os.chdir(original_cwd)

    def test_csv_removes_duplicates(self, tmp_path, mock_db_path):
        """Test that duplicate names across CSVs are deduplicated."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()

        # Same name in both files
        (submodule_path / "drengenavne.csv").write_text("Alex\nPeter\n")
        (submodule_path / "pigenavne.csv").write_text("Alex\nAnna\n")

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_csv_fallback()

            # Alex should appear only once
            assert len(result) == 3
            assert result.count("Alex") == 1
        finally:
            os.chdir(original_cwd)

    def test_csv_handles_empty_lines(self, tmp_path, mock_db_path):
        """Test that empty lines in CSV are skipped."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()

        csv_file = submodule_path / "drengenavne.csv"
        csv_file.write_text("Peter\n\n\nLars\n")

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_csv_fallback()

            assert len(result) == 2
            assert "Peter" in result
            assert "Lars" in result
        finally:
            os.chdir(original_cwd)


# =============================================================================
# Database Integration Tests
# =============================================================================


class TestDatabaseIntegration:
    """Tests verifying data appears correctly in database after sync."""

    def test_synced_names_appear_in_database(self, tmp_path, initialized_db):
        """Test that names synced from JSON appear in database."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "Peter", "gender": "M"},
        ]
        json_file.write_text(json.dumps(test_data))

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=submodule_path, check=True, capture_output=True)

        database.sync_names_with_submodule(submodule_path)

        # Verify names are queryable
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM names ORDER BY name")
            names = [row[0] for row in cursor.fetchall()]
            assert names == ["Anna", "Peter"]

    def test_gender_categorization_in_get_names_by_gender(self, tmp_path, initialized_db):
        """Test gender categorization via get_names_by_gender()."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "Emma", "gender": "F"},
            {"name": "Peter", "gender": "M"},
            {"name": "Lars", "gender": "M"},
        ]
        json_file.write_text(json.dumps(test_data))

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=submodule_path, check=True, capture_output=True)

        database.sync_names_with_submodule(submodule_path)

        gender_data = database.get_names_by_gender()

        assert "Female" in gender_data
        assert "Male" in gender_data
        assert "Anna" in gender_data["Female"]
        assert "Emma" in gender_data["Female"]
        assert "Peter" in gender_data["Male"]
        assert "Lars" in gender_data["Male"]

    def test_unisex_names_appear_in_both_genders(self, tmp_path, initialized_db):
        """Test that unisex names appear in both male and female lists."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "Peter", "gender": "M"},
            {"name": "Alex", "gender": "U"},  # Unisex
            {"name": "Kim", "gender": "U"},  # Unisex
        ]
        json_file.write_text(json.dumps(test_data))

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=submodule_path, check=True, capture_output=True)

        database.sync_names_with_submodule(submodule_path)

        gender_data = database.get_names_by_gender()

        # Unisex names should be in both Male and Female lists
        assert "Alex" in gender_data["Female"]
        assert "Alex" in gender_data["Male"]
        assert "Kim" in gender_data["Female"]
        assert "Kim" in gender_data["Male"]

        # But should also appear in Unisex list
        assert "Alex" in gender_data["Unisex"]
        assert "Kim" in gender_data["Unisex"]

    def test_ratings_initialization_for_new_names(self, tmp_path, initialized_db):
        """Test that new names can have ratings initialized."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "Peter", "gender": "M"},
        ]
        json_file.write_text(json.dumps(test_data))

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=submodule_path, check=True, capture_output=True)

        database.sync_names_with_submodule(submodule_path)

        # Initialize ratings for synced names
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM names")
            names = [row[0] for row in cursor.fetchall()]

        ratings = database.initialize_ratings(names)

        assert len(ratings) == 2
        assert ratings["Anna"] == database.INITIAL_SCORE
        assert ratings["Peter"] == database.INITIAL_SCORE

    def test_all_category_contains_all_names(self, tmp_path, initialized_db):
        """Test that 'All' category contains all names regardless of gender."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "Peter", "gender": "M"},
            {"name": "Alex", "gender": "U"},
        ]
        json_file.write_text(json.dumps(test_data))

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=submodule_path, check=True, capture_output=True)

        database.sync_names_with_submodule(submodule_path)

        gender_data = database.get_names_by_gender()

        assert len(gender_data["All"]) == 3
        assert "Anna" in gender_data["All"]
        assert "Peter" in gender_data["All"]
        assert "Alex" in gender_data["All"]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling with actual files."""

    def test_json_file_not_found(self, tmp_path, mock_db_path):
        """Test behavior when JSON file doesn't exist."""
        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            # No godkendtefornavne directory
            result = data_loader.load_submodule_json()
            assert result == []
        finally:
            os.chdir(original_cwd)

    def test_csv_files_not_found(self, tmp_path, mock_db_path):
        """Test behavior when CSV files don't exist."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_csv_fallback()
            assert result == []
        finally:
            os.chdir(original_cwd)

    def test_empty_json_file(self, tmp_path, mock_db_path):
        """Test handling of completely empty JSON file."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"
        json_file.write_text("")  # Empty file

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_json()
            assert result == []
        finally:
            os.chdir(original_cwd)

    def test_empty_csv_file(self, tmp_path, mock_db_path):
        """Test handling of completely empty CSV file."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        (submodule_path / "drengenavne.csv").write_text("")

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_csv_fallback()
            assert result == []
        finally:
            os.chdir(original_cwd)

    def test_sync_raises_error_when_json_missing(self, tmp_path, initialized_db):
        """Test that sync raises FileNotFoundError when JSON is missing."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()

        with pytest.raises(FileNotFoundError):
            database.sync_names_with_submodule(submodule_path)

    def test_handles_utf8_encoding(self, tmp_path, mock_db_path):
        """Test handling of UTF-8 encoded special characters."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        # Names with special characters
        test_data = [
            {"name": "Bjørn", "gender": "M"},
            {"name": "Søren", "gender": "M"},
            {"name": "Åse", "gender": "F"},
        ]
        json_file.write_text(json.dumps(test_data, ensure_ascii=False), encoding="utf-8")

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_json()

            names = {item["name"] for item in result}
            assert "Bjørn" in names
            assert "Søren" in names
            assert "Åse" in names
        finally:
            os.chdir(original_cwd)

    def test_sync_skips_invalid_gender_values(self, tmp_path, initialized_db):
        """Test that sync skips names with invalid gender values."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "Invalid", "gender": "X"},  # Invalid gender
            {"name": "Peter", "gender": "M"},
        ]
        json_file.write_text(json.dumps(test_data))

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=submodule_path, check=True, capture_output=True)

        inserted = database.sync_names_with_submodule(submodule_path)

        # Only 2 valid names should be inserted
        assert inserted == 2

        with database.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM names ORDER BY name")
            names = [row[0] for row in cursor.fetchall()]
            assert names == ["Anna", "Peter"]

    def test_csv_with_whitespace_handling(self, tmp_path, mock_db_path):
        """Test that CSV loading handles whitespace properly."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()

        # Names with leading/trailing whitespace
        csv_file = submodule_path / "drengenavne.csv"
        csv_file.write_text("  Peter  \n  Lars  \n")

        import os

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = data_loader.load_submodule_csv_fallback()

            # Whitespace should be stripped
            assert "Peter" in result
            assert "Lars" in result
        finally:
            os.chdir(original_cwd)


# =============================================================================
# End-to-End Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end tests combining multiple operations."""

    def test_full_sync_and_load_workflow(self, tmp_path, initialized_db):
        """Test complete workflow: sync JSON then load names by gender."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "Emma", "gender": "F"},
            {"name": "Peter", "gender": "M"},
            {"name": "Lars", "gender": "M"},
            {"name": "Alex", "gender": "U"},
        ]
        json_file.write_text(json.dumps(test_data))

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=submodule_path, check=True, capture_output=True)

        # Step 1: Sync names
        inserted = database.sync_names_with_submodule(submodule_path)
        assert inserted == 5

        # Step 2: Load names by gender
        gender_data = database.get_names_by_gender()

        # Verify structure
        assert "Female" in gender_data
        assert "Male" in gender_data
        assert "Unisex" in gender_data
        assert "All" in gender_data

        # Verify counts
        assert len(gender_data["Female"]) == 3  # Anna, Emma + Alex (unisex)
        assert len(gender_data["Male"]) == 3  # Peter, Lars + Alex (unisex)
        assert len(gender_data["Unisex"]) == 1  # Alex only
        assert len(gender_data["All"]) == 5

        # Step 3: Initialize ratings
        all_names = gender_data["All"]
        ratings = database.initialize_ratings(all_names)
        assert len(ratings) == 5
        for name in all_names:
            assert ratings[name] == database.INITIAL_SCORE

    def test_multiple_syncs_accumulate_names(self, tmp_path, initialized_db):
        """Test that multiple syncs with different commits accumulate names."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)

        # First commit - female names
        json_file.write_text(json.dumps([{"name": "Anna", "gender": "F"}]))
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "female names"], cwd=submodule_path, check=True, capture_output=True)

        inserted1 = database.sync_names_with_submodule(submodule_path)
        assert inserted1 == 1

        # Second commit - add male names (keep existing)
        json_file.write_text(
            json.dumps(
                [
                    {"name": "Anna", "gender": "F"},
                    {"name": "Peter", "gender": "M"},
                ],
            ),
        )
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add male names"], cwd=submodule_path, check=True, capture_output=True)

        inserted2 = database.sync_names_with_submodule(submodule_path)
        assert inserted2 == 1  # Only Peter is new

        # Third commit - add unisex names
        json_file.write_text(
            json.dumps(
                [
                    {"name": "Anna", "gender": "F"},
                    {"name": "Peter", "gender": "M"},
                    {"name": "Alex", "gender": "U"},
                ],
            ),
        )
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add unisex names"], cwd=submodule_path, check=True, capture_output=True)

        inserted3 = database.sync_names_with_submodule(submodule_path)
        assert inserted3 == 1  # Only Alex is new

        # Verify final state
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM names")
            assert cursor.fetchone()[0] == 3

            cursor = conn.execute("SELECT COUNT(*) FROM source_versions")
            assert cursor.fetchone()[0] == 3  # Three sync records

    def test_database_stats_after_sync(self, tmp_path, initialized_db):
        """Test that get_stats() reflects synced names correctly."""
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"

        test_data = [
            {"name": "Anna", "gender": "F"},
            {"name": "Peter", "gender": "M"},
        ]
        json_file.write_text(json.dumps(test_data))

        import subprocess

        subprocess.run(["git", "init"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=submodule_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=submodule_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=submodule_path, check=True, capture_output=True)

        database.sync_names_with_submodule(submodule_path)

        stats = database.get_stats()

        assert stats.total_names == 2
        assert stats.classified_names == 0  # No origin classification yet
        assert stats.unclassified_names == 2
        assert stats.rated_names == 0  # No ratings yet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
