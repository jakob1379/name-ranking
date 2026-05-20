"""Integration tests for Bradley-Terry model with database."""

import numpy as np


class TestBradleyTerryModelIntegration:
    """Integration tests for BradleyTerryModel with database."""

    def test_initialize_model_if_needed(self, initialized_db):
        """Test model initialization and database persistence."""
        from st_name_ranking.database import get_connection
        from st_name_ranking.model import initialize_model_if_needed

        # Define feature names (must match what features module uses)
        # We'll use a simple known set
        feature_names = [
            "phonetic_pos_0",
            "phonetic_pos_1",
            "phonetic_pos_2",
            "phonetic_pos_3",
            "phonetic_length",
            "contains_vowels",
            "has_secondary",
            "name_length",
            "name_length_squared",
            "syllable_count",
            "syllable_density",
            "vowel_ratio",
            "consonant_ratio",
            "first_letter",
            "last_letter",
            "contains_danish",
            "gender_male",
            "gender_female",
            "gender_unisex",
            "origin_nordic",
            "origin_european",
            "origin_asian",
            "origin_african",
            "origin_middle_eastern",
            "origin_international",
        ]

        # First call should create new model and save to database
        model1 = initialize_model_if_needed(feature_names)
        assert model1 is not None
        assert model1.feature_names == feature_names
        assert model1.state.training_samples == 0

        # Verify model was saved to database
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM model_state WHERE id = 1")
            count = cursor.fetchone()[0]
            assert count == 1

        # Second call should load existing model
        model2 = initialize_model_if_needed(feature_names)
        assert model2 is not None
        assert model2.feature_names == feature_names
        assert model2.state.training_samples == 0
        # Weight means should be equal (within tolerance)
        np.testing.assert_array_almost_equal(
            model1.state.weight_mean,
            model2.state.weight_mean,
        )
        np.testing.assert_array_almost_equal(
            model1.state.weight_cov,
            model2.state.weight_cov,
        )

    def test_model_update_and_save(self, initialized_db):
        """Test model update with comparisons and database persistence."""
        from st_name_ranking.database import get_connection
        from st_name_ranking.features import FeatureExtractor
        from st_name_ranking.model import BradleyTerryModel

        # Create feature extractor to get consistent feature names
        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()

        # Create model
        model = BradleyTerryModel(feature_names)
        assert model.state.training_samples == 0

        # Insert test names into database
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender, origin_region) VALUES (?, ?, ?)",
                [
                    ("Anna", "Female", "Nordic"),
                    ("Peter", "Male", "European"),
                    ("Alex", "Unisex", "Asian"),
                ],
            )

        # Extract features for the names
        features_anna = extractor.extract("Anna", "Female", "Nordic")
        features_peter = extractor.extract("Peter", "Male", "European")

        # Update model with a comparison (Anna preferred over Peter)
        model.update(features_anna, features_peter, preference=-1)
        assert model.state.training_samples == 1

        # Save model to database
        model.save_to_db()

        # Verify model state saved
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT training_samples FROM model_state WHERE id = 1")
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == 1

        # Create a new model instance and load from database
        model2 = BradleyTerryModel(feature_names)
        loaded = model2.load_from_db()
        assert loaded is True
        assert model2.state.training_samples == 1
        np.testing.assert_array_almost_equal(
            model.state.weight_mean,
            model2.state.weight_mean,
        )
        np.testing.assert_array_almost_equal(
            model.state.weight_cov,
            model2.state.weight_cov,
        )

    def test_model_select_pair(self, initialized_db):
        """Test pair selection with Thompson sampling."""
        from st_name_ranking.features import FeatureExtractor
        from st_name_ranking.model import BradleyTerryModel

        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()

        # Create model
        model = BradleyTerryModel(feature_names)

        # Insert test names
        from st_name_ranking.database import get_connection

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

        # Get names list
        names = ["Anna", "Peter", "Maria", "Jens"]
        # Extract features
        features = np.stack(
            [extractor.extract(name, "Female" if name in ["Anna", "Maria"] else "Male", None) for name in names],
        )

        # Select pair - should return valid indices and names
        idx_a, idx_b, name_a, name_b = model.select_pair(features, names)
        assert idx_a != idx_b
        assert 0 <= idx_a < len(names)
        assert 0 <= idx_b < len(names)
        assert name_a == names[idx_a]
        assert name_b == names[idx_b]

        # After some updates, selection may change
        # Update model with a comparison
        model.update(features[0], features[1], preference=-1)
        # Select again (should still work)
        idx_a2, idx_b2, name_a2, name_b2 = model.select_pair(features, names)
        assert idx_a2 != idx_b2

    def test_model_select_top_k_pairs(self, initialized_db):
        """Test batch pair selection."""
        from st_name_ranking.features import FeatureExtractor
        from st_name_ranking.model import BradleyTerryModel

        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()
        model = BradleyTerryModel(feature_names)

        # Insert test names
        from st_name_ranking.database import get_connection

        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [
                    ("Anna", "Female"),
                    ("Peter", "Male"),
                    ("Maria", "Female"),
                    ("Jens", "Male"),
                    ("Lars", "Male"),
                ],
            )

        names = ["Anna", "Peter", "Maria", "Jens", "Lars"]
        features = np.stack(
            [extractor.extract(name, "Female" if name in ["Anna", "Maria"] else "Male", None) for name in names],
        )

        # Select top 3 pairs
        pairs = model.select_top_k_pairs(features, names, k=3)
        assert len(pairs) == 3
        for idx_a, idx_b, name_a, name_b in pairs:
            assert idx_a != idx_b
            assert 0 <= idx_a < len(names)
            assert 0 <= idx_b < len(names)
            assert name_a == names[idx_a]
            assert name_b == names[idx_b]

        # Pairs should be unique (by normalized indices)
        pair_set = set()
        for idx_a, idx_b, _, _ in pairs:
            pair = (min(idx_a, idx_b), max(idx_a, idx_b))
            assert pair not in pair_set
            pair_set.add(pair)

        # Test with k larger than possible unique pairs (should still work)
        # With 5 names, max unique pairs = 10
        pairs_large = model.select_top_k_pairs(features, names, k=10)
        assert len(pairs_large) == 10
        # All pairs should be unique
        pair_set_large = set()
        for idx_a, idx_b, _, _ in pairs_large:
            pair = (min(idx_a, idx_b), max(idx_a, idx_b))
            assert pair not in pair_set_large
            pair_set_large.add(pair)

    def test_model_update_batch(self, initialized_db):
        """Test batch update with multiple comparisons."""
        from st_name_ranking.features import FeatureExtractor
        from st_name_ranking.model import BradleyTerryModel

        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()
        model = BradleyTerryModel(feature_names)

        # Insert test names
        from st_name_ranking.database import get_connection

        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [
                    ("Anna", "Female"),
                    ("Peter", "Male"),
                    ("Maria", "Female"),
                ],
            )

        names = ["Anna", "Peter", "Maria"]
        features = np.stack(
            [extractor.extract(name, "Female" if name in ["Anna", "Maria"] else "Male", None) for name in names],
        )

        # Create batch of comparisons
        comparisons = [
            (features[0], features[1], -1),  # Anna preferred over Peter
            (features[1], features[2], 1),  # Maria preferred over Peter (preference=1 means name_b wins)
            (features[0], features[2], 0),  # Anna vs Maria draw
        ]

        initial_samples = model.state.training_samples
        model.update_batch(comparisons)
        assert model.state.training_samples == initial_samples + len(comparisons)

        # Verify model changed (weights not zero)
        assert not np.allclose(model.state.weight_mean, 0.0)

        # Save and load to verify persistence
        model.save_to_db()
        model2 = BradleyTerryModel(feature_names)
        loaded = model2.load_from_db()
        assert loaded is True
        assert model2.state.training_samples == model.state.training_samples
        np.testing.assert_array_almost_equal(
            model.state.weight_mean,
            model2.state.weight_mean,
        )

    def test_model_update_both_disliked(self, initialized_db):
        """Test update_both_disliked adds two comparisons."""
        from st_name_ranking.features import FeatureExtractor
        from st_name_ranking.model import BradleyTerryModel

        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()
        model = BradleyTerryModel(feature_names)

        # Insert test names
        from st_name_ranking.database import get_connection

        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [
                    ("Anna", "Female"),
                    ("Peter", "Male"),
                ],
            )

        # Extract features
        features_anna = extractor.extract("Anna", "Female", None)
        features_peter = extractor.extract("Peter", "Male", None)

        initial_samples = model.state.training_samples
        model.update_both_disliked(features_anna, features_peter)
        assert model.state.training_samples == initial_samples + 2

        # Verify weights changed
        assert not np.allclose(model.state.weight_mean, 0.0)
