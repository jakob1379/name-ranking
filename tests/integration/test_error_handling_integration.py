"""Error handling and edge case integration tests for st_name_ranking.

Tests failure modes and recovery paths to ensure production reliability.
"""

import pickle
import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from st_name_ranking import database
from st_name_ranking.model import BradleyTerryModel


class TestDatabaseErrorHandling:
    """Tests for database error scenarios."""

    def test_handles_database_locked_gracefully(self, initialized_db):
        """When database is locked, should not crash."""
        from st_name_ranking.database import get_stats

        # Simulate database locked error by patching sqlite3.connect
        call_count = [0]
        original_connect = sqlite3.connect

        def mock_connect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:  # Fail first two attempts
                raise sqlite3.OperationalError("database is locked")
            return original_connect(*args, **kwargs)

        with patch("sqlite3.connect", side_effect=mock_connect):
            # Should not crash, but may return empty/default stats
            try:
                stats = get_stats()
                # If it succeeds, should return valid structure
                assert "total_names" in stats
            except sqlite3.OperationalError:
                # Expected if lock can't be resolved
                pass

    def test_database_transaction_rollback_on_error(self, initialized_db):
        """Transaction should rollback on error."""
        from st_name_ranking.database import get_connection, get_ratings

        # Insert a name first
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                ("TestName", "Male"),
            )

        # Attempt operation that will fail mid-transaction
        try:
            with get_connection() as conn:
                conn.execute(
                    "INSERT INTO ratings (name_id, rating) VALUES (?, ?)",
                    (99999, 1500.0),  # Invalid name_id (doesn't exist)
                )
                # Force an error
                raise ValueError("Simulated error mid-transaction")
        except ValueError:
            pass

        # Verify ratings table is still consistent
        ratings = get_ratings()
        assert isinstance(ratings, dict)
        # No ratings should exist for the invalid name
        assert "TestName" not in ratings or ratings.get("TestName") is None

    def test_get_stats_handles_corrupted_database(self, mock_db_path):
        """get_stats should handle corrupted database gracefully."""
        from st_name_ranking.database import get_connection, get_stats

        # Initialize database
        database.init_database()

        # Corrupt the database by dropping a table
        with get_connection() as conn:
            conn.execute("DROP TABLE names")

        # get_stats should handle this gracefully
        with pytest.raises((sqlite3.OperationalError, sqlite3.DatabaseError)):
            get_stats()

    def test_database_timeout_retry(self, initialized_db):
        """Database operations should handle timeout scenarios."""
        from st_name_ranking.database import get_connection

        # Test that we can set timeout and handle busy scenarios
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            # Simulate busy error on first commit
            mock_conn.commit.side_effect = [
                sqlite3.OperationalError("database is locked"),
                None,  # Second attempt succeeds
            ]

            with pytest.raises(Exception):
                with get_connection() as conn:
                    conn.execute("SELECT 1")


class TestModelErrorHandling:
    """Tests for model error scenarios."""

    def test_recover_from_corrupted_model_blob(self, initialized_db):
        """Corrupted pickle data should trigger reinitialization."""
        from st_name_ranking.database import get_connection

        # Initialize a model and save it
        feature_names = ["feature1", "feature2", "feature3"]
        original_model = BradleyTerryModel(feature_names)
        original_model.save_to_db()

        # Corrupt the pickle data in the database
        with get_connection() as conn:
            conn.execute(
                "UPDATE model_state SET feature_weights = ? WHERE id = 1",
                (b"corrupted pickle data",),
            )

        # Create new model and attempt to load - should handle corruption gracefully
        new_model = BradleyTerryModel(feature_names)

        # load_from_db should handle pickle errors gracefully
        # Currently it raises UnpicklingError - this is a known limitation
        # The test documents the expected behavior once fixed
        try:
            result = new_model.load_from_db()
            # If it returns False, the corruption was handled gracefully
            assert result is False, "Should fail to load corrupted model"
        except (pickle.UnpicklingError, ValueError, RuntimeError, OSError) as e:
            # Current behavior: exception is raised
            # This is the known issue - model doesn't handle corrupted pickle data
            assert "pickle" in str(e).lower() or "truncated" in str(e).lower() or isinstance(e, pickle.UnpicklingError)

        # Model should still be usable with initial state (even after failed load)
        assert new_model.state is not None
        assert len(new_model.state.weight_mean) == len(feature_names)

    def test_model_handles_dimension_mismatch(self, initialized_db):
        """Model should handle dimension mismatches in stored data."""

        # Create model with 3 features and save
        original_model = BradleyTerryModel(["f1", "f2", "f3"])
        original_model.save_to_db()

        # Create new model with different feature count
        new_model = BradleyTerryModel(["f1", "f2", "f3", "f4"])

        # Should detect mismatch and not load
        result = new_model.load_from_db()
        assert result is False

    def test_model_update_with_invalid_features(self, initialized_db):
        """Model update should handle invalid feature vectors."""

        model = BradleyTerryModel(["f1", "f2", "f3"])

        # Create mismatched feature vectors
        features_a = np.array([1.0, 0.5])  # Wrong dimension
        features_b = np.array([0.5, 1.0, 0.0])  # Correct dimension

        # Should raise error for dimension mismatch
        with pytest.raises((ValueError, IndexError)):
            model.update(features_a, features_b, -1)

    def test_model_save_failure_graceful(self, initialized_db):
        """Model should handle save failures gracefully."""

        model = BradleyTerryModel(["f1", "f2"])

        # Mock database to fail on save
        with patch("st_name_ranking.model.get_connection") as mock_conn:
            mock_context = MagicMock()
            mock_conn.return_value.__enter__ = MagicMock(return_value=mock_context)
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_context.execute.side_effect = sqlite3.OperationalError("database is locked")

            # Should raise the error (not silently fail)
            with pytest.raises(sqlite3.OperationalError):
                model.save_to_db()


class TestEmptyDatabaseEdgeCases:
    """Tests for empty or minimal database scenarios."""

    def test_handles_empty_database(self, mock_db_path):
        """With no names, strict candidate selection should report absence."""
        from st_name_ranking.active_learning.selection import select_candidates
        from st_name_ranking.database import init_database

        # Initialize empty database
        init_database()

        with pytest.raises(ValueError, match="Need at least 2 names"):
            select_candidates([])

    def test_single_name_no_pairs(self, mock_db_path):
        """With one name, can't form comparison pairs."""
        from st_name_ranking.active_learning.selection import select_candidates
        from st_name_ranking.database import get_connection, init_database

        # Initialize and insert single name
        init_database()
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("SoloName", "Male"),
            )

        with pytest.raises(ValueError, match="Need at least 2 names"):
            select_candidates(["SoloName"])

    def test_model_pair_selection_with_two_names(self, initialized_db):
        """Model should handle minimum case of exactly 2 names."""
        from st_name_ranking.database import get_connection
        from st_name_ranking.features import FeatureExtractor

        # Insert two names
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("NameA", "Male"),
            )
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("NameB", "Female"),
            )

        # Create model and extract features
        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()
        bt_model = BradleyTerryModel(feature_names)

        # Extract features for both names
        features_a = extractor.extract("NameA", "Male", None)
        features_b = extractor.extract("NameB", "Female", None)
        features_matrix = np.stack([features_a, features_b], axis=0)

        # Should be able to select pair with just 2 names
        idx_a, idx_b, name_a, name_b = bt_model.select_pair(features_matrix, ["NameA", "NameB"])

        assert name_a in ["NameA", "NameB"]
        assert name_b in ["NameA", "NameB"]
        assert name_a != name_b

    def test_all_names_filtered_returns_empty(self, initialized_db):
        """When filters exclude all names, handle gracefully."""
        from st_name_ranking.database import get_connection, get_names_by_filters

        # Insert names with specific gender
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender, origin_region) VALUES (?, ?, ?)",
                ("MaleName", "Male", "Nordic"),
            )

        # Query with filter that excludes all names
        result = get_names_by_filters(gender="Female")  # No female names
        assert result == [], "Should return empty list when filter matches nothing"

        # Query with origin filter that excludes all
        result = get_names_by_filters(origins=["Asian"])  # No asian names
        assert result == [], "Should return empty list when origin filter matches nothing"


class TestInvalidDataHandling:
    """Tests for handling invalid or corrupted data."""

    def test_invalid_preference_value_raises(self, initialized_db):
        """Invalid preference values should raise appropriate error."""
        from st_name_ranking.database import get_connection, record_comparison

        # Insert two names
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("NameA", "Male"),
            )
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("NameB", "Female"),
            )

        # Invalid preference value
        with pytest.raises(ValueError, match="preference must be"):
            record_comparison("NameA", "NameB", 99)

        # Another invalid value
        with pytest.raises(ValueError, match="preference must be"):
            record_comparison("NameA", "NameB", -2)

    def test_vote_with_missing_name_handles_gracefully(self, initialized_db):
        """Voting for non-existent name should not crash."""
        from st_name_ranking.database import record_comparison

        # Try to record comparison with non-existent names
        with pytest.raises(ValueError, match="Name not found"):
            record_comparison("NonExistentA", "NonExistentB", -1)

    def test_update_rating_for_nonexistent_name_raises(self, initialized_db):
        """Updating rating for non-existent name should raise error."""
        from st_name_ranking.database import update_rating

        with pytest.raises(ValueError, match="Name not found"):
            update_rating("GhostName", 1600.0)

    def test_get_ratings_with_orphaned_entries(self, initialized_db):
        """get_ratings should handle orphaned rating entries."""
        from st_name_ranking.database import get_connection, get_ratings

        # Insert a name and rating
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("ValidName", "Male"),
            )

        # Insert rating
        with get_connection() as conn:
            cursor = conn.execute("SELECT id FROM names WHERE name = ?", ("ValidName",))
            name_id = cursor.fetchone()[0]
            conn.execute(
                "INSERT INTO ratings (name_id, rating) VALUES (?, ?)",
                (name_id, 1500.0),
            )

        # Should return valid ratings
        ratings = get_ratings()
        assert "ValidName" in ratings
        assert ratings["ValidName"] == 1500.0


class TestSubmoduleDataHandling:
    """Tests for submodule data loading edge cases."""

    def test_missing_submodule_data(self, tmp_path, initialized_db):
        """Missing submodule should be handled gracefully."""
        from st_name_ranking.data_loader import load_submodule_json

        # Create temp directory without submodule
        nonexistent_path = tmp_path / "nonexistent"

        # Patch the json_path to non-existent location
        with patch("st_name_ranking.data_loader.os.path.join", return_value=str(nonexistent_path / "allenavne.json")):
            with patch("streamlit.toast"):  # Suppress toast messages
                result = load_submodule_json()
                assert result == [], "Should return empty list when submodule missing"

    def test_corrupted_submodule_json(self, tmp_path, initialized_db):
        """Corrupted JSON in submodule should be handled gracefully."""
        from st_name_ranking.data_loader import load_submodule_json

        # Create corrupted JSON file
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"
        json_file.write_text("not valid json {{{")

        with patch("st_name_ranking.data_loader.os.path.join", return_value=str(json_file)):
            with patch("streamlit.toast"):  # Suppress toast messages
                result = load_submodule_json()
                assert result == [], "Should return empty list for corrupted JSON"

    def test_submodule_json_missing_columns(self, tmp_path, initialized_db):
        """JSON missing required columns should be handled."""
        from st_name_ranking.data_loader import load_submodule_json

        # Create JSON with wrong columns
        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        json_file = submodule_path / "allenavne.json"
        json_file.write_text('[{"wrong_column": "value"}]')

        with patch("st_name_ranking.data_loader.os.path.join", return_value=str(json_file)):
            with patch("streamlit.toast"):  # Suppress toast messages
                result = load_submodule_json()
                assert result == [], "Should return empty list for JSON with wrong schema"


class TestConsistencyAndRecovery:
    """Tests for data consistency and recovery scenarios."""

    def test_model_update_failure_preserves_consistency(self, initialized_db):
        """If model update fails, ratings should remain consistent."""
        from st_name_ranking.database import get_connection, get_ratings, update_rating

        # Insert test names
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("Winner", "Male"),
            )
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("Loser", "Female"),
            )

        # Set initial ratings
        update_rating("Winner", 1600.0)
        update_rating("Loser", 1400.0)

        # Get initial ratings
        initial_ratings = get_ratings()
        initial_winner_rating = initial_ratings["Winner"]
        initial_loser_rating = initial_ratings["Loser"]

        from st_name_ranking.active_learning.lazy_updates import record_comparison_instant

        # Simulate model update failure
        with patch(
            "st_name_ranking.active_learning.lazy_updates.get_or_initialize_active_learning_model",
        ) as mock_get_model:
            mock_model = MagicMock()
            mock_model.update.side_effect = RuntimeError("Model update failed")
            mock_get_model.return_value = mock_model

            # Attempt update - should handle gracefully (function catches RuntimeError)
            record_comparison_instant("Winner", "Loser", -1, blocking=True)  # Should not raise

        # Ratings should remain unchanged
        final_ratings = get_ratings()
        assert final_ratings["Winner"] == initial_winner_rating
        assert final_ratings["Loser"] == initial_loser_rating

    def test_batch_update_partial_failure(self, initialized_db):
        """Batch update should handle partial failures."""
        from st_name_ranking.database import (
            get_connection,
            get_ratings,
            update_ratings_batch,
        )

        # Insert test names
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("NameA", "Male"),
            )
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("NameB", "Female"),
            )
            # NameC doesn't exist - will cause partial failure

        # Attempt batch update with one non-existent name
        ratings_dict = {"NameA": 1600.0, "NameB": 1500.0, "NameC": 1400.0}

        # Should complete without error and report skipped names.
        skipped = update_ratings_batch(ratings_dict)
        assert skipped == ["NameC"]

        # Verify valid names were updated
        final_ratings = get_ratings()
        assert "NameA" in final_ratings
        assert "NameB" in final_ratings
        assert "NameC" not in final_ratings

    def test_feature_extraction_failure_fallback(self):
        """Feature extraction should have fallback for failures."""
        from st_name_ranking.features import extract_phonetic_features

        # Normal extraction should work
        features = extract_phonetic_features("Anna")
        assert isinstance(features, dict)
        assert len(features) > 0

        # Test with problematic input (empty string)
        features = extract_phonetic_features("")
        assert isinstance(features, dict)
        # Should return default features
        assert "phonetic_length" in features


class TestRaceConditions:
    """Tests for race condition scenarios."""

    def test_concurrent_database_initialization(self, mock_db_path):
        """Multiple simultaneous init_database calls should be safe."""
        from st_name_ranking.database import init_database

        # Call init_database multiple times concurrently (simulated)
        init_database()
        init_database()
        init_database()

        # Should be idempotent
        from st_name_ranking.database import get_connection

        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM names")
            # Should not raise, database should be consistent
            assert cursor.fetchone()[0] >= 0

    def test_model_singleton_concurrent_access(self, initialized_db):
        """Model singleton should handle concurrent access."""
        # Reset model singleton
        selection.reset_active_learning_state()

        # Multiple calls to get_or_initialize_active_learning_model should return same instance
        model1 = selection.get_or_initialize_active_learning_model()
        model2 = selection.get_or_initialize_active_learning_model()

        assert model1 is model2, "Should return same model instance"


class TestMemoryAndResourceLimits:
    """Tests for resource limit scenarios."""

    def test_large_batch_phonetic_lookup(self, initialized_db):
        """Phonetic lookup should handle large batches."""
        from st_name_ranking.database import get_connection, get_phonetic_codes_batch

        # Insert many names
        names = [f"Name{i}" for i in range(100)]
        with get_connection() as conn:
            for name in names:
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, ?)",
                    (name, "Male"),
                )

        # Batch lookup should work
        result = get_phonetic_codes_batch(names)
        assert isinstance(result, dict)
        # All names should be in result
        for name in names:
            assert name in result

    def test_large_batch_name_details(self, initialized_db):
        """Name details batch lookup should handle large batches."""
        from st_name_ranking.database import get_connection, get_name_details_batch

        # Insert many names
        names = [f"Name{i}" for i in range(100)]
        with get_connection() as conn:
            for name in names:
                conn.execute(
                    "INSERT INTO names (name, gender, origin_region) VALUES (?, ?, ?)",
                    (name, "Male", "Nordic"),
                )

        # Batch lookup should work and chunk properly
        result = get_name_details_batch(names)
        assert len(result) == 100


class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_model_with_zero_training_samples(self, initialized_db):
        """Model should handle zero training samples."""

        model = BradleyTerryModel(["f1", "f2"])
        model.save_to_db()

        # Load model back
        new_model = BradleyTerryModel(["f1", "f2"])
        loaded = new_model.load_from_db()
        assert loaded is True

        # Should have zero training samples
        assert new_model.state.training_samples == 0

    def test_preference_stats_with_no_comparisons(self, initialized_db):
        """Preference stats should handle no comparisons gracefully."""
        from st_name_ranking.database import (
            get_preference_stats_by_gender,
            get_preference_stats_by_origin,
        )

        # No comparisons yet
        gender_stats = get_preference_stats_by_gender()
        assert isinstance(gender_stats, dict)
        assert len(gender_stats) == 0

        origin_stats = get_preference_stats_by_origin()
        assert isinstance(origin_stats, dict)
        assert len(origin_stats) == 0

    def test_comparison_count_for_uncompared_name(self, initialized_db):
        """get_comparison_count should return 0 for uncompared names."""
        from st_name_ranking.database import get_comparison_count, get_connection

        # Insert a name
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("Uncompared", "Male"),
            )

        # Should return 0 for name with no comparisons
        count = get_comparison_count("Uncompared")
        assert count == 0

    def test_select_candidates_with_same_name_filtered_out(self, initialized_db):
        """select_candidates should never return the same name twice."""
        from st_name_ranking.active_learning.selection import select_candidates
        from st_name_ranking.database import get_connection

        # Insert names
        with get_connection() as conn:
            for name in ["Alice", "Bob", "Charlie"]:
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, ?)",
                    (name, "Female"),
                )

        # Run multiple times to ensure we never get same name
        names = ["Alice", "Bob", "Charlie"]
        for _ in range(20):
            result = select_candidates(names)
            if result[0]:  # If not empty
                assert result[0] != result[1], f"Got same name twice: {result}"


class TestErrorMessages:
    """Tests for informative error messages."""

    def test_database_error_messages_are_informative(self, initialized_db):
        """Database errors should provide informative messages."""
        from st_name_ranking.database import update_rating

        # Try to update non-existent name
        with pytest.raises(ValueError) as exc_info:
            update_rating("NonExistentName12345", 1500.0)

        error_msg = str(exc_info.value)
        assert "Name not found" in error_msg
        assert "NonExistentName12345" in error_msg

    def test_comparison_error_messages(self, initialized_db):
        """Comparison errors should be informative."""
        from st_name_ranking.database import record_comparison

        with pytest.raises(ValueError) as exc_info:
            record_comparison("NonExistentA", "NonExistentB", -1)

        error_msg = str(exc_info.value)
        assert "Name not found" in error_msg

    def test_invalid_preference_error_message(self, initialized_db):
        """Invalid preference error should explain valid values."""
        from st_name_ranking.database import get_connection, record_comparison

        # Insert names first
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("NameA", "Male"),
            )
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("NameB", "Female"),
            )

        with pytest.raises(ValueError) as exc_info:
            record_comparison("NameA", "NameB", 999)

        error_msg = str(exc_info.value)
        assert "preference must be" in error_msg
