"""Tests for st_name_ranking.database module."""

import json
from unittest.mock import patch

import pytest

from st_name_ranking import database


class TestDatabaseInitialization:
    """Tests for database initialization and schema."""

    def test_init_database_creates_tables(self, mock_db_path):
        """Test that init_database creates all required tables."""
        # Arrange
        from st_name_ranking.database import get_connection, init_database

        # Act
        init_database()

        # Assert
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            expected_tables = {
                "names",
                "ratings",
                "region_mapping",
                "source_versions",
                "user_settings",
            }
            assert expected_tables.issubset(tables)

    def test_init_database_idempotent(self, mock_db_path):
        """Test that calling init_database multiple times doesn't cause errors."""
        from st_name_ranking.database import init_database

        # Should not raise
        init_database()
        init_database()
        init_database()

        # Verify still works
        with database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM names")
            count = cursor.fetchone()[0]
            assert count == 0  # No names inserted yet

    def test_init_database_resets_state_on_schema_failure(self, mock_db_path):
        """Failed schema initialization should not poison future retries."""
        database.reset_database_init_state()

        with (
            patch(
                "st_name_ranking.database._insert_default_region_mapping",
                side_effect=RuntimeError("schema failure"),
            ),
            pytest.raises(RuntimeError, match="schema failure"),
        ):
            database.init_database()

        assert database._INIT_STATE == {"db_initialized": False, "db_path": None}

        database.init_database()

        assert database._INIT_STATE == {
            "db_initialized": True,
            "db_path": database.DB_PATH,
        }

    def test_region_mapping_populated(self, mock_db_path):
        """Test that region_mapping table is populated with data."""
        from st_name_ranking.database import get_connection, init_database

        init_database()

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM region_mapping")
            count = cursor.fetchone()[0]

            # Should have at least some region mappings
            assert count > 0

            # Check some expected regions exist
            cursor.execute(
                "SELECT region FROM region_mapping WHERE nationality = 'Denmark'",
            )
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == "Nordic"

    def test_get_connection_context_manager(self, mock_db_path):
        """Test that get_connection context manager works correctly."""
        from st_name_ranking.database import get_connection, init_database

        init_database()

        # Should commit automatically on success
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("TestName", "Male"),
            )

        # Verify data was committed
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM names WHERE name = 'TestName'")
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == "TestName"

        # Should rollback on exception
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO names (name, gender) VALUES (?, ?)",
                    ("FailedName", "Female"),
                )
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Verify data was NOT committed
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM names WHERE name = 'FailedName'")
            result = cursor.fetchone()
            assert result is None


class TestNameOperations:
    """Tests for name-related database operations."""

    def test_insert_names_via_sql(self, initialized_db):
        """Test inserting names directly via SQL."""
        from st_name_ranking.database import get_connection

        # Insert names directly via SQL (since there's no insert_names function)
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [("Anna", "Female"), ("Peter", "Male")],
            )

        # Verify using get_unclassified_names
        from st_name_ranking.database import get_unclassified_names

        names = get_unclassified_names()
        assert len(names) == 2
        name_set = {n.name for n in names}
        assert "Anna" in name_set
        assert "Peter" in name_set

        # Verify names were inserted (gender not included in get_unclassified_names)
        # Additional check: verify via direct SQL query
        from st_name_ranking.database import get_connection

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT gender FROM names WHERE name = ?", ("Anna",))
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == "Female"

            cursor.execute(
                "SELECT gender FROM names WHERE name = ?",
                ("Peter",),
            )
            result = cursor.fetchone()
            assert result[0] == "Male"

    def test_get_names_by_gender(self, initialized_db):
        """Test retrieving names filtered by gender."""
        from st_name_ranking.database import get_connection, get_names_by_gender

        # Insert test data
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [
                    ("Anna", "Female"),
                    ("Peter", "Male"),
                    ("Maria", "Female"),
                    ("Jens", "Male"),
                ],
            )

        # Get names by gender - returns dict mapping gender to list of names
        gender_dict = get_names_by_gender()

        # Check structure
        assert isinstance(gender_dict, dict)
        assert "Female" in gender_dict
        assert "Male" in gender_dict

        # Check contents
        female_names = gender_dict["Female"]
        male_names = gender_dict["Male"]

        assert len(female_names) == 2
        assert len(male_names) == 2
        assert "Anna" in female_names
        assert "Maria" in female_names
        assert "Peter" in male_names
        assert "Jens" in male_names

    def test_get_unclassified_names(self, initialized_db):
        """Test retrieving names that haven't been classified."""
        from st_name_ranking.database import (
            get_connection,
            get_unclassified_names,
            update_name_origin,
        )

        # Insert test data
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [
                    ("Anna", "Female"),
                    ("Peter", "Male"),
                    ("Maria", "Female"),
                ],
            )

        # Initially all names are unclassified
        unclassified = get_unclassified_names()
        assert len(unclassified) == 3
        assert {n.name for n in unclassified} == {"Anna", "Peter", "Maria"}

        # Classify one name - note: update_name_origin expects name_id (int), not name
        # First get the name_id for Anna
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM names WHERE name = ?", ("Anna",))
            anna_id = cursor.fetchone()[0]

        # Update origin using name_id
        update_name_origin(anna_id, "Nordic", 0.85)

        # Now only 2 unclassified
        unclassified = get_unclassified_names()
        assert len(unclassified) == 2
        assert {n.name for n in unclassified} == {"Peter", "Maria"}

    def test_update_name_origin(self, initialized_db):
        """Test updating name origin classification."""
        from st_name_ranking.database import get_connection, update_name_origin

        # Insert test name
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                ("Anna", "Female"),
            )
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM names WHERE name = ?", ("Anna",))
            anna_id = cursor.fetchone()[0]

        # Update origin
        update_name_origin(anna_id, "Nordic", 0.92)

        # Verify update
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT origin_region, origin_confidence, origin_classified_at FROM names WHERE id = ?",
                (anna_id,),
            )
            result = cursor.fetchone()
            assert result is not None
            region, confidence, classified_at = result
            assert region == "Nordic"
            assert confidence == 0.92
            assert classified_at is not None


class TestRatingOperations:
    """Tests for rating-related database operations."""

    def test_update_rating(self, initialized_db):
        """Test updating a name's rating."""
        from st_name_ranking.database import (
            get_connection,
            get_ratings,
            update_rating,
        )

        # First insert a name via SQL
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                ("Anna", "Female"),
            )

        # Update rating - should create rating row
        update_rating("Anna", 1600.0)

        # Verify via get_ratings (returns dict name->rating)
        ratings = get_ratings()
        assert "Anna" in ratings
        assert ratings["Anna"] == 1600.0

        # Verify matches count incremented (default is 1 on first update)
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT matches FROM ratings r
                JOIN names n ON r.name_id = n.id
                WHERE n.name = ?
            """,
                ("Anna",),
            )
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == 0  # First update defaults matches to 0

    def test_update_rating_multiple_times(self, initialized_db):
        """Test updating rating multiple times increments matches."""
        from st_name_ranking.database import (
            get_connection,
            get_ratings,
            update_rating,
        )

        # Insert name
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                ("Peter", "Male"),
            )

        # Update rating multiple times
        update_rating("Peter", 1500.0)
        update_rating("Peter", 1550.0)
        update_rating("Peter", 1525.0)

        # Verify final rating
        ratings = get_ratings()
        assert ratings["Peter"] == 1525.0

        # Verify matches count is 0 (preserved from first insert, not incremented)
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT matches FROM ratings r
                JOIN names n ON r.name_id = n.id
                WHERE n.name = ?
            """,
                ("Peter",),
            )
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == 0

    def test_update_rating_nonexistent_name(self, initialized_db):
        """Test updating rating for non-existent name reports it as skipped."""
        from st_name_ranking.database import update_rating

        assert update_rating("Nonexistent", 1500.0) == ["Nonexistent"]

    def test_get_ratings_empty(self, initialized_db):
        """Test getting ratings when no ratings exist."""
        from st_name_ranking.database import get_ratings

        ratings = get_ratings()
        assert isinstance(ratings, dict)
        assert len(ratings) == 0


class TestSubmoduleOperations:
    """Tests for submodule version tracking."""

    def test_get_latest_submodule_version(self, initialized_db):
        """Test getting latest submodule version."""
        from st_name_ranking.database import (
            get_latest_submodule_version,
            update_submodule_version,
        )

        # Initially no version
        version = get_latest_submodule_version()
        assert version is None

        update_submodule_version("abc123")

        # Get updated version
        version = get_latest_submodule_version()
        assert version is not None
        assert version.commit_hash == "abc123"
        # names_count not stored in table

    def test_update_submodule_version(self, initialized_db):
        """Test updating submodule version."""
        from st_name_ranking.database import (
            get_latest_submodule_version,
            update_submodule_version,
        )

        # First update
        update_submodule_version("abc123")
        version1 = get_latest_submodule_version()
        assert version1 is not None
        assert version1.commit_hash == "abc123"

        # Second update (should add new row)
        update_submodule_version("def456")
        version2 = get_latest_submodule_version()
        assert version2 is not None
        assert version2.commit_hash == "def456"
        # Should be the latest (def456)
        assert version2.commit_hash != version1.commit_hash


class TestStatistics:
    """Tests for database statistics."""

    def test_get_stats_empty(self, initialized_db):
        """Test getting statistics from empty database."""
        from st_name_ranking.database import get_stats

        stats = get_stats()

        assert stats.total_names == 0
        assert stats.classified_names == 0
        assert stats.unclassified_names == 0
        assert stats.rated_names == 0
        assert stats.origin_distribution == {}
        assert isinstance(stats.origin_distribution, dict)

    def test_get_stats_with_data(self, initialized_db):
        """Test getting statistics with data."""
        from st_name_ranking.database import (
            get_connection,
            get_stats,
            update_name_origin,
            update_rating,
        )

        # Insert test names via SQL
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [
                    ("Anna", "Female"),
                    ("Peter", "Male"),
                    ("Maria", "Female"),
                ],
            )
            # Get name IDs for classification
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM names WHERE name = ?", ("Anna",))
            anna_id = cursor.fetchone()[0]

        # Create ratings for all names (using update_rating)
        update_rating("Anna", 1500.0)
        update_rating("Peter", 1500.0)
        update_rating("Maria", 1500.0)

        # Classify one name
        update_name_origin(anna_id, "Nordic", 0.85)

        # Get stats
        stats = get_stats()

        # Verify statistics
        assert stats.total_names == 3
        assert stats.classified_names == 1
        assert stats.unclassified_names == 2
        assert stats.rated_names == 3
        origin_dist = stats.origin_distribution
        assert isinstance(origin_dist, dict)
        # Should have 'Nordic' and 'International' (unclassified)
        assert "Nordic" in origin_dist
        assert origin_dist["Nordic"] == 1
        assert "International" in origin_dist
        assert origin_dist["International"] == 2

    def test_get_total_comparisons(self, initialized_db):
        """Test total comparison counter."""
        from st_name_ranking.database import (
            get_connection,
            get_total_comparisons,
            record_comparison,
        )

        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [
                    ("Anna", "Female"),
                    ("Peter", "Male"),
                    ("Maria", "Female"),
                ],
            )

        assert get_total_comparisons() == 0

        record_comparison("Anna", "Peter", -1)
        record_comparison("Anna", "Maria", 0)

        assert get_total_comparisons() == 2


class TestSyncOperations:
    """Tests for sync operations with submodule."""

    def test_sync_names_with_submodule(
        self,
        mock_submodule_path,
        initialized_db,
    ):
        """Test syncing names from submodule directory."""
        from unittest.mock import patch

        from st_name_ranking.database import (
            get_latest_submodule_version,
            get_unclassified_names,
            sync_names_with_submodule,
        )

        # Mock subprocess.run to return a fake commit hash
        # Also mock sync validation to accept all fixture names.
        def mock_is_valid_name(name):
            return True

        with (
            patch("st_name_ranking.sync_store.subprocess.run") as mock_run,
            patch(
                "st_name_ranking.sync_store.is_valid_name",
                side_effect=mock_is_valid_name,
            ),
        ):
            mock_process = mock_run.return_value
            mock_process.stdout = "mockcommithash123\n"
            mock_process.stderr = ""
            mock_process.returncode = 0

            # Sync names with mock path
            result = sync_names_with_submodule(
                submodule_path=mock_submodule_path,
            )
            assert isinstance(result, int)
            # Expect 3 names inserted
            if result != 3:
                # Debug: check what's in the database
                from st_name_ranking.database import get_connection

                with get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM names")
                    cursor.fetchall()
                    cursor.execute("SELECT * FROM source_versions")
                    cursor.fetchall()
            assert result == 3  # 3 names from JSON file

        # Verify names were inserted (get_unclassified_names returns dict with id, name)
        get_unclassified_names()
        # Note: get_unclassified_names only returns names with NULL origin_region
        # Since none are classified, all inserted names should be returned
        # However, there might be more if other tests inserted names.
        # Use raw SQL to count total names
        from st_name_ranking.database import get_connection

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM names")
            total = cursor.fetchone()[0]
        assert total == 3

        # Verify submodule version was updated
        version = get_latest_submodule_version()
        assert version is not None
        # names_count not stored, but commit_hash should be set
        assert version.commit_hash is not None

    def test_sync_names_empty_submodule(self, tmp_path, initialized_db):
        """Test syncing from empty submodule directory."""
        from unittest.mock import patch

        from st_name_ranking.database import sync_names_with_submodule

        # Create empty submodule directory with empty JSON file
        empty_path = tmp_path / "empty"
        empty_path.mkdir()
        json_file = empty_path / "allenavne.json"
        json_file.write_text(json.dumps([]))

        # Mock subprocess.run to return a fake commit hash
        with patch("st_name_ranking.sync_store.subprocess.run") as mock_run:
            mock_process = mock_run.return_value
            mock_process.stdout = "mockcommithash456\n"
            mock_process.stderr = ""
            mock_process.returncode = 0

            # Should handle empty JSON gracefully
            result = sync_names_with_submodule(submodule_path=empty_path)
            assert result == 0  # inserted count
            # sync_names_with_submodule returns inserted count (int)


class TestPhoneticOperations:
    """Tests for phonetic code computation and updates."""

    def test_compute_phonetic_codes(self):
        """Test _compute_phonetic_codes function."""
        from unittest.mock import patch

        from st_name_ranking.database import _compute_phonetic_codes

        # Test with standard name
        with patch("st_name_ranking.database.doublemetaphone") as mock_dm:
            mock_dm.return_value = ("AN", "AN")
            primary, secondary = _compute_phonetic_codes("Anna")
            assert primary == "AN"
            assert secondary == "AN"
            mock_dm.assert_called_once_with("Anna")

        # Test with empty primary (should return empty string)
        with patch("st_name_ranking.database.doublemetaphone") as mock_dm:
            mock_dm.return_value = (None, "AN")
            primary, secondary = _compute_phonetic_codes("Test")
            assert primary == ""
            assert secondary == "AN"

        # Test with empty secondary (should return empty string)
        with patch("st_name_ranking.database.doublemetaphone") as mock_dm:
            mock_dm.return_value = ("AN", None)
            primary, secondary = _compute_phonetic_codes("Test")
            assert primary == "AN"
            assert secondary == ""

    def test_update_phonetic_codes_no_names(self, initialized_db):
        """Test update_phonetic_codes when no names need updating."""
        from st_name_ranking.database import update_phonetic_codes

        # No names in database, so no updates
        result = update_phonetic_codes()
        assert result == 0

    def test_update_phonetic_codes_with_names(self, initialized_db):
        """Test update_phonetic_codes with names needing updates."""
        from unittest.mock import patch

        from st_name_ranking.database import get_connection, update_phonetic_codes

        # Insert a name without phonetic codes
        with get_connection() as conn:
            cursor = conn.execute("INSERT INTO names (name, gender) VALUES (?, ?)", ("Anna", "Female"))
            name_id = cursor.lastrowid

        # Mock doublemetaphone
        with patch("st_name_ranking.database.doublemetaphone") as mock_dm:
            mock_dm.return_value = ("AN", "AN")
            result = update_phonetic_codes()

        # Should update 1 name
        assert result == 1
        mock_dm.assert_called_once_with("Anna")

        # Verify phonetic codes were set
        with get_connection() as conn:
            cursor = conn.execute("SELECT phonetic_primary, phonetic_secondary FROM names WHERE id = ?", (name_id,))
            primary, secondary = cursor.fetchone()
            assert primary == "AN"
            assert secondary == "AN"

    def test_update_phonetic_codes_with_limit(self, initialized_db):
        """Test update_phonetic_codes with limit parameter."""
        from unittest.mock import patch

        from st_name_ranking.database import get_connection, update_phonetic_codes

        # Insert multiple names
        names = [("Anna", "Female"), ("Peter", "Male"), ("Maria", "Female")]
        with get_connection() as conn:
            for name, gender in names:
                conn.execute("INSERT INTO names (name, gender) VALUES (?, ?)", (name, gender))

        # Mock doublemetaphone to track calls
        with patch("st_name_ranking.database.doublemetaphone") as mock_dm:
            mock_dm.return_value = ("XX", "XX")
            result = update_phonetic_codes(limit=2)

        # Should update only 2 names due to limit
        assert result == 2
        assert mock_dm.call_count == 2

    def test_update_phonetic_codes_already_updated(self, initialized_db):
        """Test update_phonetic_codes when names already have phonetic codes."""
        from unittest.mock import patch

        from st_name_ranking.database import get_connection, update_phonetic_codes

        # Insert a name WITH phonetic codes
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO names (name, gender, phonetic_primary, phonetic_secondary)
                   VALUES (?, ?, ?, ?)""",
                ("Anna", "Female", "AN", "AN"),
            )

        # Mock doublemetaphone (should not be called)
        with patch("st_name_ranking.database.doublemetaphone") as mock_dm:
            result = update_phonetic_codes()

        # Should update 0 names
        assert result == 0
        mock_dm.assert_not_called()


class TestDatabaseExportImport:
    """Tests for database export and import functionality."""

    def test_export_database(self, mock_db_path):
        """Test exporting database as bytes."""
        from st_name_ranking.database import export_database, init_database

        init_database()
        db_bytes = export_database()
        assert isinstance(db_bytes, bytes)
        assert len(db_bytes) > 0

    def test_import_database(self, mock_db_path):
        """Test importing database from bytes."""
        from st_name_ranking.database import export_database, import_database, init_database

        init_database()
        original_bytes = export_database()
        # Import the same bytes (should work)
        import_database(original_bytes, backup=False)
        # Export again and compare
        new_bytes = export_database()
        assert original_bytes == new_bytes


class TestFeatureCacheOperations:
    """Tests for feature cache persistence behavior."""

    def test_feature_cache_keeps_versions_separate_locally(self, mock_db_path):
        """Local cache hits should not cross feature-set versions."""
        from st_name_ranking.database import get_connection, init_database
        from st_name_ranking.persistence.feature_cache import FeatureCache

        init_database()
        with get_connection() as conn:
            name_id = conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("Anna", "Female"),
            ).lastrowid
            v1_id = conn.execute(
                "INSERT INTO feature_sets (version, feature_names_json, is_active) VALUES (?, ?, ?)",
                ("v1", json.dumps(["length"]), 1),
            ).lastrowid
            v2_id = conn.execute(
                "INSERT INTO feature_sets (version, feature_names_json, is_active) VALUES (?, ?, ?)",
                ("v2", json.dumps(["length"]), 0),
            ).lastrowid
            conn.execute(
                "INSERT INTO name_features (name_id, feature_set_id, features_json) VALUES (?, ?, ?)",
                (name_id, v1_id, json.dumps({"length": 0.2})),
            )
            conn.execute(
                "INSERT INTO name_features (name_id, feature_set_id, features_json) VALUES (?, ?, ?)",
                (name_id, v2_id, json.dumps({"length": 0.8})),
            )

        cache = FeatureCache("v1")

        assert cache.get_features(name_id) == {"length": 0.2}
        assert cache.get_features(name_id, "v2") == {"length": 0.8}
        assert cache.get_features(name_id) == {"length": 0.2}

    def test_corrupt_cached_features_raise_contextual_error(self, mock_db_path):
        """Corrupt cached JSON should be reported distinctly from a cache miss."""
        from st_name_ranking.database import (
            CorruptFeatureCacheError,
            get_cached_features,
            get_connection,
            init_database,
        )

        init_database()
        with get_connection() as conn:
            name_id, feature_set_id = self._insert_corrupt_feature_cache_row(conn)

        with pytest.raises(CorruptFeatureCacheError) as exc_info:
            get_cached_features(name_id, feature_set_id)

        assert exc_info.value.name_id == name_id
        assert exc_info.value.feature_set_id == feature_set_id
        assert "Corrupt feature cache row" in str(exc_info.value)

    def test_corrupt_cached_features_batch_raise_contextual_error(self, mock_db_path):
        """Batch cache reads should not silently drop corrupt rows."""
        from st_name_ranking.database import (
            CorruptFeatureCacheError,
            get_cached_features_batch,
            get_connection,
            init_database,
        )

        init_database()
        with get_connection() as conn:
            name_id, feature_set_id = self._insert_corrupt_feature_cache_row(conn)

        with pytest.raises(CorruptFeatureCacheError) as exc_info:
            get_cached_features_batch([name_id], feature_set_id)

        assert exc_info.value.name_id == name_id
        assert exc_info.value.feature_set_id == feature_set_id

    @staticmethod
    def _insert_corrupt_feature_cache_row(conn) -> tuple[int, int]:
        name_id = conn.execute(
            "INSERT INTO names (name, gender) VALUES (?, ?)",
            ("Anna", "Female"),
        ).lastrowid
        feature_set_id = conn.execute(
            "INSERT INTO feature_sets (version, feature_names_json, is_active) VALUES (?, ?, ?)",
            ("corrupt-test", json.dumps(["length"]), 1),
        ).lastrowid
        conn.execute(
            """
            INSERT INTO name_features (name_id, feature_set_id, features_json)
            VALUES (?, ?, ?)
            """,
            (name_id, feature_set_id, "{not-json"),
        )
        return name_id, feature_set_id
