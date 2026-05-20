"""Integration tests for the core user workflow of the Name Ranking application.

Tests the complete round-trip: Feature extraction → Model → Database
"""

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from st_name_ranking import database
from st_name_ranking.database import (
    get_connection,
    init_database,
    record_comparison,
    update_rating,
    update_ratings_batch,
    update_ratings_batch_values,
)
from st_name_ranking.features import FeatureExtractor
from st_name_ranking.model import BradleyTerryModel


@pytest.fixture
def temp_db():
    """Create a temporary database file and patch the active database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    original_path = database.get_db_path()
    database.set_db_path(db_path)

    yield db_path

    # Cleanup
    database.set_db_path(original_path)
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def fresh_db(temp_db):
    """Initialize a fresh database with schema."""
    init_database()
    return temp_db


@pytest.fixture
def sample_names():
    """Sample names with gender and origin for testing."""
    return [
        ("Emma", "Female", "Nordic"),
        ("Liam", "Male", "European"),
        ("Sofia", "Female", "European"),
        ("Noah", "Male", "American"),
        ("Olivia", "Female", "International"),
        ("Lucas", "Male", "Nordic"),
    ]


@pytest.fixture
def populated_db(fresh_db, sample_names):
    """Database populated with sample names and ratings."""
    with get_connection() as conn:
        # Insert names
        for name, gender, origin in sample_names:
            conn.execute(
                """
                INSERT OR IGNORE INTO names (name, gender, origin_region)
                VALUES (?, ?, ?)
                """,
                (name, gender, origin),
            )

        # Initialize ratings for all names
        cursor = conn.execute("SELECT name FROM names")
        names = [row[0] for row in cursor.fetchall()]
        for name in names:
            name_id = conn.execute("SELECT id FROM names WHERE name = ?", (name,)).fetchone()[0]
            conn.execute(
                """
                INSERT OR IGNORE INTO ratings (name_id, rating, matches)
                VALUES (?, 1500.0, 0)
                """,
                (name_id,),
            )

    return fresh_db


class TestFullPreferenceWorkflow:
    """Test the complete preference recording workflow."""

    def test_preference_workflow_round_trip(self, populated_db):
        """Test full workflow: extract features → model update → database records."""
        # Step 1: Extract features for two names
        extractor = FeatureExtractor()

        # Get name details from database
        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT name, gender, origin_region FROM names WHERE name IN (?, ?)",
                ("Emma", "Liam"),
            )
            rows = cursor.fetchall()

        assert len(rows) == 2, "Both names should exist in database"

        name_data = {row[0]: (row[1], row[2]) for row in rows}
        emma_gender, emma_origin = name_data["Emma"]
        liam_gender, liam_origin = name_data["Liam"]

        # Extract features
        emma_features = extractor.extract("Emma", emma_gender, emma_origin)
        liam_features = extractor.extract("Liam", liam_gender, liam_origin)

        assert emma_features.shape == liam_features.shape, "Feature dimensions should match"
        assert emma_features.shape[0] > 0, "Features should not be empty"

        # Step 2: Initialize BradleyTerryModel
        feature_names = extractor.get_feature_names()
        model = BradleyTerryModel(feature_names)

        # Verify initial state
        assert model.state.training_samples == 0
        assert np.allclose(model.state.weight_mean, 0.0)

        # Step 3: Record user preference (Emma > Liam, preference=-1)
        model.update(emma_features, liam_features, preference=-1)

        # Step 4: Verify model weights updated
        assert model.state.training_samples == 1
        assert not np.allclose(model.state.weight_mean, 0.0), "Weights should have changed"

        # Step 5: Compute ratings and update database
        emma_utility = model.get_utility(emma_features.reshape(1, -1))[0]
        liam_utility = model.get_utility(liam_features.reshape(1, -1))[0]

        emma_rating = 1500 + emma_utility * 500
        liam_rating = 1500 + liam_utility * 500

        update_rating("Emma", emma_rating)
        update_rating("Liam", liam_rating)

        # Step 6: Record comparison in database
        record_comparison("Emma", "Liam", preference=-1)

        # Step 7: Verify database state
        with get_connection() as conn:
            # Check ratings updated
            cursor = conn.execute(
                """
                SELECT n.name, r.rating
                FROM names n
                JOIN ratings r ON n.id = r.name_id
                WHERE n.name IN ('Emma', 'Liam')
                """,
            )
            ratings = {row[0]: row[1] for row in cursor.fetchall()}

            assert "Emma" in ratings
            assert "Liam" in ratings
            assert ratings["Emma"] == pytest.approx(emma_rating, rel=1e-5)
            assert ratings["Liam"] == pytest.approx(liam_rating, rel=1e-5)

            # Check comparison recorded
            cursor = conn.execute(
                """
                SELECT c.preference, n1.name as name_a, n2.name as name_b
                FROM comparisons c
                JOIN names n1 ON c.name_a_id = n1.id
                JOIN names n2 ON c.name_b_id = n2.id
                WHERE n1.name = 'Emma' AND n2.name = 'Liam'
                """,
            )
            comparison = cursor.fetchone()
            assert comparison is not None, "Comparison should be recorded"
            assert comparison[0] == -1, "Preference should be -1 (Emma preferred)"

    def test_draw_workflow(self, populated_db):
        """Test workflow with draw preference (preference=0)."""
        extractor = FeatureExtractor()

        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT name, gender, origin_region FROM names WHERE name IN (?, ?)",
                ("Sofia", "Olivia"),
            )
            rows = cursor.fetchall()

        name_data = {row[0]: (row[1], row[2]) for row in rows}

        sofia_features = extractor.extract("Sofia", *name_data["Sofia"])
        olivia_features = extractor.extract("Olivia", *name_data["Olivia"])

        feature_names = extractor.get_feature_names()
        model = BradleyTerryModel(feature_names)

        # Record draw
        model.update(sofia_features, olivia_features, preference=0)

        assert model.state.training_samples == 1

        # Record in database
        record_comparison("Sofia", "Olivia", preference=0)

        # Verify
        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT preference FROM comparisons c "
                "JOIN names n1 ON c.name_a_id = n1.id "
                "JOIN names n2 ON c.name_b_id = n2.id "
                "WHERE n1.name = 'Sofia' AND n2.name = 'Olivia'",
            )
            result = cursor.fetchone()
            assert result[0] == 0

    def test_both_disliked_workflow(self, populated_db):
        """Test workflow with both disliked (preference=2)."""
        extractor = FeatureExtractor()

        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT name, gender, origin_region FROM names WHERE name IN (?, ?)",
                ("Noah", "Lucas"),
            )
            rows = cursor.fetchall()

        name_data = {row[0]: (row[1], row[2]) for row in rows}

        noah_features = extractor.extract("Noah", *name_data["Noah"])
        lucas_features = extractor.extract("Lucas", *name_data["Lucas"])

        feature_names = extractor.get_feature_names()
        model = BradleyTerryModel(feature_names)

        # Record both disliked - this creates 2 training samples
        model.update_both_disliked(noah_features, lucas_features)

        assert model.state.training_samples == 2, "Both disliked creates 2 comparisons"

        # Record in database
        record_comparison("Noah", "Lucas", preference=2)

        # Verify
        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT preference FROM comparisons c "
                "JOIN names n1 ON c.name_a_id = n1.id "
                "JOIN names n2 ON c.name_b_id = n2.id "
                "WHERE n1.name = 'Noah' AND n2.name = 'Lucas'",
            )
            result = cursor.fetchone()
            assert result[0] == 2


class TestModelPersistenceRoundTrip:
    """Test model save/load round-trip through database."""

    def test_model_save_and_load_weights_identical(self, populated_db):
        """Verify model weights are identical after save/load cycle."""
        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()

        # Create and train model
        model1 = BradleyTerryModel(feature_names)

        # Add some training data
        with get_connection() as conn:
            cursor = conn.execute("SELECT name, gender, origin_region FROM names LIMIT 4")
            rows = cursor.fetchall()

        names_data = [(row[0], row[1], row[2]) for row in rows]

        # Create several comparisons
        comparisons = [
            (names_data[0], names_data[1], -1),  # First preferred
            (names_data[2], names_data[3], 1),  # Second preferred
            (names_data[0], names_data[2], 0),  # Draw
        ]

        for (name_a, gender_a, origin_a), (name_b, gender_b, origin_b), pref in comparisons:
            feat_a = extractor.extract(name_a, gender_a, origin_a)
            feat_b = extractor.extract(name_b, gender_b, origin_b)
            model1.update(feat_a, feat_b, pref)

        # Get predictions before save
        test_name = names_data[0]
        test_features = extractor.extract(*test_name)
        utility_before = model1.get_utility(test_features.reshape(1, -1))[0]

        # Save weights for comparison
        weights_before = model1.state.weight_mean.copy()
        cov_before = model1.state.weight_cov.copy()
        training_samples_before = model1.state.training_samples

        # Step 1: Save to database
        model1.save_to_db()

        # Verify saved to database
        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT feature_weights, uncertainty_matrix, training_samples FROM model_state WHERE id = 1",
            )
            row = cursor.fetchone()
            assert row is not None, "Model state should be saved"

        # Step 2: Load in new model instance
        model2 = BradleyTerryModel(feature_names)
        loaded = model2.load_from_db()

        assert loaded is True, "Model should load successfully"

        # Step 3: Verify weights are identical
        assert np.allclose(model2.state.weight_mean, weights_before), "Weights should be identical after load"
        assert np.allclose(model2.state.weight_cov, cov_before), "Covariance should be identical after load"
        assert model2.state.training_samples == training_samples_before, "Training samples should match"

        # Step 4: Verify predictions are consistent
        utility_after = model2.get_utility(test_features.reshape(1, -1))[0]
        assert utility_before == pytest.approx(utility_after, rel=1e-10), "Predictions should be identical"

    def test_model_load_returns_false_when_no_state(self, fresh_db):
        """Test that load_from_db returns False when no model state exists."""
        extractor = FeatureExtractor()
        model = BradleyTerryModel(extractor.get_feature_names())

        loaded = model.load_from_db()
        assert loaded is False, "Should return False when no model state exists"

    def test_multiple_saves_overwrite(self, populated_db):
        """Test that multiple saves properly overwrite previous state."""
        import time

        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()
        model = BradleyTerryModel(feature_names)

        # First save (initial state)
        model.save_to_db()

        # Small delay to ensure timestamp difference
        time.sleep(0.1)

        # Train model
        with get_connection() as conn:
            cursor = conn.execute("SELECT name, gender, origin_region FROM names LIMIT 2")
            rows = cursor.fetchall()

        feat_a = extractor.extract(rows[0][0], rows[0][1], rows[0][2])
        feat_b = extractor.extract(rows[1][0], rows[1][1], rows[1][2])
        model.update(feat_a, feat_b, -1)

        # Second save (trained state)
        model.save_to_db()

        # Verify state was updated by checking training_samples
        with get_connection() as conn:
            cursor = conn.execute("SELECT training_samples, last_updated FROM model_state WHERE id = 1")
            row = cursor.fetchone()
            assert row[0] == 1, "Training samples should be 1"
            timestamp2 = row[1]
            assert timestamp2 is not None

        # Load and verify
        model2 = BradleyTerryModel(feature_names)
        model2.load_from_db()
        assert model2.state.training_samples == 1

        # Verify only one row exists
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM model_state")
            count = cursor.fetchone()[0]
            assert count == 1, "Should only have one model state row"


class TestFeatureExtractionIntegration:
    """Test feature extraction integration with database and model."""

    def test_feature_vector_dimensions_match_model(self, populated_db):
        """Verify feature vector dimensions match model expectations."""
        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()

        model = BradleyTerryModel(feature_names)

        # Extract features for a name from database
        with get_connection() as conn:
            cursor = conn.execute("SELECT name, gender, origin_region FROM names LIMIT 1")
            name, gender, origin = cursor.fetchone()

        features = extractor.extract(name, gender, origin)

        # Verify dimensions match
        assert features.shape[0] == model.d, (
            f"Feature dimension {features.shape[0]} should match model dimension {model.d}"
        )
        assert features.shape[0] == len(feature_names), "Feature count should match feature names count"

    def test_features_feed_correctly_into_model_update(self, populated_db):
        """Test that extracted features work correctly with model.update()."""
        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()
        model = BradleyTerryModel(feature_names)

        # Get two names with complete data
        with get_connection() as conn:
            cursor = conn.execute("SELECT name, gender, origin_region FROM names WHERE gender IS NOT NULL LIMIT 2")
            rows = cursor.fetchall()

        assert len(rows) == 2, "Need two names with gender data"

        # Extract features
        features_list = []
        for name, gender, origin in rows:
            features = extractor.extract(name, gender, origin)
            features_list.append(features)

        # Should be able to update model without error
        model.update(features_list[0], features_list[1], -1)

        # Verify model was updated
        assert model.state.training_samples == 1

    def test_batch_feature_extraction_consistency(self, populated_db):
        """Test batch extraction produces same results as individual extraction."""
        extractor = FeatureExtractor()

        # Get names from database
        with get_connection() as conn:
            cursor = conn.execute("SELECT name, gender, origin_region FROM names LIMIT 5")
            rows = cursor.fetchall()

        names = [row[0] for row in rows]
        genders = [row[1] for row in rows]
        origins = [row[2] for row in rows]

        # Extract individually
        individual_features = [
            extractor.extract(name, gender, origin) for name, gender, origin in zip(names, genders, origins)
        ]

        # Extract in batch
        batch_features = extractor.batch_extract(names, genders, origins)

        # Verify consistency
        assert len(individual_features) == batch_features.shape[0]
        for i, individual in enumerate(individual_features):
            assert np.allclose(individual, batch_features[i]), f"Batch feature {i} should match individual extraction"

    def test_feature_names_consistency(self, populated_db):
        """Test that feature names are consistent across extractions."""
        extractor = FeatureExtractor()

        # Get names with different characteristics
        with get_connection() as conn:
            cursor = conn.execute("SELECT name, gender, origin_region FROM names LIMIT 3")
            rows = cursor.fetchall()

        # Extract features for each
        feature_names_list = []
        for name, gender, origin in rows:
            _, feature_names = extractor.extract(name, gender, origin), extractor.get_feature_names()
            feature_names_list.append(feature_names)

        # All should have same feature names
        for i in range(1, len(feature_names_list)):
            assert feature_names_list[0] == feature_names_list[i], (
                "Feature names should be consistent across extractions"
            )


class TestBatchOperations:
    """Test batch operations with database."""

    def test_batch_rating_updates_consistency(self, fresh_db):
        """Test that batch rating updates maintain database consistency."""
        # Insert test names
        test_names = [f"TestName{i}" for i in range(50)]

        with get_connection() as conn:
            for name in test_names:
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, 'Unisex')",
                    (name,),
                )

        # Create ratings dict
        ratings_dict = {name: 1500.0 + i * 10 for i, name in enumerate(test_names)}

        # Update in batch
        update_ratings_batch_values(ratings_dict)

        # Verify all ratings were updated
        with get_connection() as conn:
            cursor = conn.execute("SELECT n.name, r.rating FROM names n JOIN ratings r ON n.id = r.name_id")
            db_ratings = {row[0]: row[1] for row in cursor.fetchall()}

        for name, expected_rating in ratings_dict.items():
            assert name in db_ratings, f"Rating for {name} should exist"
            assert db_ratings[name] == pytest.approx(expected_rating, rel=1e-5)

    def test_batch_operations_atomic_rollback_on_error(self, fresh_db):
        """Test that batch operations are atomic (all succeed or all fail)."""
        # Insert test names
        test_names = [f"TestName{i}" for i in range(10)]

        with get_connection() as conn:
            for name in test_names:
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, 'Unisex')",
                    (name,),
                )

        # Create ratings dict with one invalid name
        ratings_dict = dict.fromkeys(test_names, 1600.0)
        ratings_dict["NonExistentName"] = 1700.0  # This name doesn't exist

        # This should complete without error and report skipped names.
        skipped = update_ratings_batch_values(ratings_dict)
        assert skipped == ["NonExistentName"]

        # Verify valid names were still updated
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM ratings")
            count = cursor.fetchone()[0]
            assert count == 10, "Should have 10 ratings (valid names only)"

    def test_large_batch_100_plus_names(self, fresh_db):
        """Test rating updates with 100+ names."""
        # Insert 120 test names
        test_names = [f"BatchName{i:03d}" for i in range(120)]

        with get_connection() as conn:
            for idx, name in enumerate(test_names):
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, ?)",
                    (name, "Unisex" if idx % 2 == 0 else "Male"),
                )

        # Create ratings for all names
        ratings_dict = {name: 1400.0 + (i % 200) for i, name in enumerate(test_names)}

        # Update in batch
        update_ratings_batch(ratings_dict)

        # Verify all ratings
        with get_connection() as conn:
            cursor = conn.execute("SELECT n.name, r.rating, r.matches FROM names n JOIN ratings r ON n.id = r.name_id")
            rows = cursor.fetchall()

        db_ratings = {row[0]: (row[1], row[2]) for row in rows}

        assert len(db_ratings) == 120, "All 120 ratings should be in database"

        for name, expected_rating in ratings_dict.items():
            assert name in db_ratings
            assert db_ratings[name][0] == pytest.approx(expected_rating, rel=1e-5)
            assert db_ratings[name][1] == 1, "Match count should be incremented to 1"

    def test_batch_update_preserves_existing_match_counts(self, fresh_db):
        """Test that batch updates preserve existing match counts appropriately."""
        # Insert test names
        test_names = [f"MatchTest{i}" for i in range(5)]

        with get_connection() as conn:
            for name in test_names:
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, 'Unisex')",
                    (name,),
                )
                name_id = conn.execute("SELECT id FROM names WHERE name = ?", (name,)).fetchone()[0]
                # Pre-populate with match count
                conn.execute(
                    "INSERT INTO ratings (name_id, rating, matches) VALUES (?, 1500, 5)",
                    (name_id,),
                )

        # Update ratings using batch (should increment matches)
        ratings_dict = dict.fromkeys(test_names, 1600.0)
        update_ratings_batch(ratings_dict)

        # Verify match counts incremented
        with get_connection() as conn:
            for name in test_names:
                cursor = conn.execute(
                    """
                    SELECT r.matches FROM ratings r
                    JOIN names n ON r.name_id = n.id
                    WHERE n.name = ?
                    """,
                    (name,),
                )
                matches = cursor.fetchone()[0]
                assert matches == 6, f"Match count for {name} should be 6 (was 5, +1)"

    def test_batch_values_update_preserves_match_counts(self, fresh_db):
        """Test that batch_values update does NOT increment match counts."""
        # Insert test names
        test_names = [f"ValueTest{i}" for i in range(5)]

        with get_connection() as conn:
            for name in test_names:
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, 'Unisex')",
                    (name,),
                )
                name_id = conn.execute("SELECT id FROM names WHERE name = ?", (name,)).fetchone()[0]
                conn.execute(
                    "INSERT INTO ratings (name_id, rating, matches) VALUES (?, 1500, 5)",
                    (name_id,),
                )

        # Update ratings using batch_values (should NOT increment matches)
        ratings_dict = dict.fromkeys(test_names, 1600.0)
        update_ratings_batch_values(ratings_dict)

        # Verify match counts unchanged
        with get_connection() as conn:
            for name in test_names:
                cursor = conn.execute(
                    """
                    SELECT r.matches FROM ratings r
                    JOIN names n ON r.name_id = n.id
                    WHERE n.name = ?
                    """,
                    (name,),
                )
                matches = cursor.fetchone()[0]
                assert matches == 5, f"Match count for {name} should still be 5"


class TestIntegrationEdgeCases:
    """Test edge cases and error handling in the core workflow."""

    def test_model_update_with_invalid_preference(self, populated_db):
        """Test that invalid preference values raise errors."""
        with get_connection() as conn:
            cursor = conn.execute("SELECT name, gender, origin_region FROM names LIMIT 2")
            rows = cursor.fetchall()

        # Invalid preference should raise error in record_comparison
        with pytest.raises(ValueError):
            record_comparison(rows[0][0], rows[1][0], preference=99)

    def test_preference_recorded_for_nonexistent_name(self, fresh_db):
        """Test error when recording preference for non-existent names."""
        with pytest.raises(ValueError) as exc_info:
            record_comparison("NonExistent", "AlsoNonExistent", preference=-1)

        assert "Name not found" in str(exc_info.value)

    def test_feature_extractor_cache_works_correctly(self, populated_db):
        """Test that feature extractor caching works as expected."""
        extractor = FeatureExtractor()

        with get_connection() as conn:
            cursor = conn.execute("SELECT name, gender, origin_region FROM names LIMIT 1")
            name, gender, origin = cursor.fetchone()

        # First extraction
        features1 = extractor.extract(name, gender, origin)

        # Second extraction (should use cache)
        features2 = extractor.extract(name, gender, origin)

        # Should be identical object or at least equal
        assert np.allclose(features1, features2)

    def test_model_with_no_training_data_can_save_load(self, fresh_db):
        """Test that untrained model can be saved and loaded."""
        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()

        # Create model without training
        model1 = BradleyTerryModel(feature_names)

        # Save
        model1.save_to_db()

        # Load in new instance
        model2 = BradleyTerryModel(feature_names)
        loaded = model2.load_from_db()

        assert loaded is True
        assert model2.state.training_samples == 0
        assert np.allclose(model2.state.weight_mean, 0.0)

    def test_database_transaction_rollback(self, fresh_db):
        """Test that database transactions properly rollback on error."""
        # Insert a name
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, 'Unisex')",
                ("TransactionTest",),
            )

        # Try an operation that will fail partway through
        # The get_connection context manager should rollback on exception
        try:
            with get_connection() as conn:
                # This will succeed
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, 'Male')",
                    ("ValidName",),
                )
                # This will fail (duplicate name)
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, 'Female')",
                    ("TransactionTest",),  # Duplicate
                )
        except sqlite3.IntegrityError:
            pass  # Expected

        # Verify the valid insert was rolled back
        with get_connection() as conn:
            cursor = conn.execute("SELECT name FROM names WHERE name = 'ValidName'")
            result = cursor.fetchone()
            assert result is None, "Transaction should have been rolled back"


class TestEndToEndWorkflow:
    """End-to-end tests simulating real user interactions."""

    def test_multiple_comparisons_accumulate(self, populated_db):
        """Test that multiple user comparisons accumulate correctly."""
        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()
        model = BradleyTerryModel(feature_names)

        # Get several names
        with get_connection() as conn:
            cursor = conn.execute("SELECT name, gender, origin_region FROM names LIMIT 4")
            rows = cursor.fetchall()

        names_data = [(row[0], row[1], row[2]) for row in rows]

        # Simulate 5 comparisons
        comparisons = [
            (0, 1, -1),  # names[0] > names[1]
            (2, 3, 1),  # names[3] > names[2]
            (0, 2, -1),  # names[0] > names[2]
            (1, 3, 0),  # names[1] = names[3] (draw)
            (0, 3, -1),  # names[0] > names[3]
        ]

        for i, j, pref in comparisons:
            name_a, gender_a, origin_a = names_data[i]
            name_b, gender_b, origin_b = names_data[j]

            feat_a = extractor.extract(name_a, gender_a, origin_a)
            feat_b = extractor.extract(name_b, gender_b, origin_b)

            model.update(feat_a, feat_b, pref)
            record_comparison(name_a, name_b, pref)

        # Verify model state
        assert model.state.training_samples == 5

        # Verify database state
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            count = cursor.fetchone()[0]
            assert count == 5, "Should have 5 comparisons recorded"

        # Save and reload model
        model.save_to_db()

        model2 = BradleyTerryModel(feature_names)
        model2.load_from_db()

        assert model2.state.training_samples == 5

    def test_rating_changes_after_multiple_updates(self, populated_db):
        """Test that ratings evolve correctly after multiple updates."""
        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()
        model = BradleyTerryModel(feature_names)

        with get_connection() as conn:
            cursor = conn.execute("SELECT name, gender, origin_region FROM names LIMIT 3")
            rows = cursor.fetchall()

        names = [row[0] for row in rows]
        name_data = {row[0]: (row[1], row[2]) for row in rows}

        # Get initial ratings
        initial_ratings = {}
        for name in names:
            features = extractor.extract(name, *name_data[name])
            utility = model.get_utility(features.reshape(1, -1))[0]
            initial_ratings[name] = 1500 + utility * 500

        # All initial ratings should be 1500 (untrained model)
        for rating in initial_ratings.values():
            assert rating == pytest.approx(1500.0)

        # Train model with comparisons
        for i in range(len(names) - 1):
            feat_a = extractor.extract(names[i], *name_data[names[i]])
            feat_b = extractor.extract(names[i + 1], *name_data[names[i + 1]])
            model.update(feat_a, feat_b, -1)  # First name always wins

        # Get updated ratings
        final_ratings = {}
        for name in names:
            features = extractor.extract(name, *name_data[name])
            utility = model.get_utility(features.reshape(1, -1))[0]
            final_ratings[name] = 1500 + utility * 500

        # Ratings should have diverged
        assert final_ratings[names[0]] > final_ratings[names[-1]], (
            "First name should have higher rating than last after training"
        )

        # Save ratings to database
        for name, rating in final_ratings.items():
            update_rating(name, rating)

        # Verify ratings in database
        with get_connection() as conn:
            for name, expected_rating in final_ratings.items():
                cursor = conn.execute(
                    """
                    SELECT r.rating FROM ratings r
                    JOIN names n ON r.name_id = n.id
                    WHERE n.name = ?
                    """,
                    (name,),
                )
                db_rating = cursor.fetchone()[0]
                assert db_rating == pytest.approx(expected_rating, rel=1e-5)
