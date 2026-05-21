"""Integration tests for phonetic similarity and active learning integration.

Tests phonetic clustering and active learning pair selection to ensure
diversity in candidate selection.
"""

import time
from unittest.mock import patch

import numpy as np

from st_name_ranking.active_learning.selection import (
    get_or_create_feature_extractor,
    select_candidate_batch,
    select_candidates,
)
from st_name_ranking.learning.model import (
    BradleyTerryModel,
    _get_phonetic_codes_cached,
    _group_names_by_phonetic,
    _select_cross_cluster_pairs,
)
from st_name_ranking.persistence import database
from st_name_ranking.phonetic_similarity import (
    batch_compute_phonetic_codes,
)
from st_name_ranking.types import NamePair


class TestPhoneticGrouping:
    """Tests for phonetic clustering functionality."""

    def test_phonetic_grouping_creates_clusters(self, initialized_db):
        """
        Names with same phonetic code should be grouped together.
        Names with different codes should be in different clusters.
        """
        # Insert names with known phonetic properties
        # Anna and Hannah both have primary code 'AN'
        # Peter has code 'PTR', Jens has 'JNS'
        names_data = [
            ("Anna", "Female", "AN", ""),
            ("Hannah", "Female", "AN", ""),
            ("Peter", "Male", "PTR", ""),
            ("Jens", "Male", "JNS", ""),
        ]

        with database.get_connection() as conn:
            for name, gender, primary, secondary in names_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO names
                    (name, gender, phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, gender, primary, secondary),
                )

        # Test grouping
        names = ["Anna", "Hannah", "Peter", "Jens"]
        clusters = _group_names_by_phonetic(names)

        # Verify clusters exist
        assert "AN" in clusters
        assert "PTR" in clusters
        assert "JNS" in clusters

        # Verify Anna and Hannah are in same cluster
        an_cluster = clusters["AN"]
        assert len(an_cluster) == 2
        name_indices = {names[i] for i in an_cluster}
        assert name_indices == {"Anna", "Hannah"}

        # Verify Peter and Jens are in different clusters
        assert len(clusters["PTR"]) == 1
        assert len(clusters["JNS"]) == 1
        assert names[clusters["PTR"][0]] == "Peter"
        assert names[clusters["JNS"][0]] == "Jens"

    def test_phonetic_grouping_with_danish_names(self, initialized_db):
        """Test phonetic grouping with Danish-specific names."""
        # Names with Danish characters
        names_data = [
            ("Søren", "Male", "SRN", ""),
            ("Møller", "Male", "MLR", ""),
            ("Jørgen", "Male", "JRN", ""),
            ("Kærlig", "Female", "KRL", ""),
            ("Åse", "Female", "AS", ""),
        ]

        with database.get_connection() as conn:
            for name, gender, primary, secondary in names_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO names
                    (name, gender, phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, gender, primary, secondary),
                )

        names = ["Søren", "Møller", "Jørgen", "Kærlig", "Åse"]
        clusters = _group_names_by_phonetic(names)

        # Each should be in its own cluster (different phonetic codes)
        assert len(clusters) == 5
        for name in names:
            # Find which cluster this name is in
            found = False
            for cluster_names in clusters.values():
                if any(names[i] == name for i in cluster_names):
                    found = True
                    break
            assert found, f"Name {name} not found in any cluster"


class TestCrossClusterPairSelection:
    """Tests for cross-cluster pair selection."""

    def test_cross_cluster_pairs_are_different_phonetic(self, initialized_db):
        """
        Selected cross-cluster pairs should have different phonetic codes.
        """
        # Setup: Create clusters with different phonetic codes
        names_data = [
            ("Anna", "Female", "AN", ""),
            ("Anne", "Female", "AN", ""),
            ("Peter", "Male", "PTR", ""),
            ("Petra", "Female", "PTR", ""),
            ("Jens", "Male", "JNS", ""),
            ("Jonas", "Male", "JNS", ""),
        ]

        with database.get_connection() as conn:
            for name, gender, primary, secondary in names_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO names
                    (name, gender, phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, gender, primary, secondary),
                )

        names = ["Anna", "Anne", "Peter", "Petra", "Jens", "Jonas"]
        clusters = _group_names_by_phonetic(names)

        # Select cross-cluster pairs
        rng = np.random.default_rng(42)
        pairs = _select_cross_cluster_pairs(clusters, n_pairs=10, rng=rng)

        # Verify pairs have different phonetic codes
        phonetic_map = {
            "Anna": "AN",
            "Anne": "AN",
            "Peter": "PTR",
            "Petra": "PTR",
            "Jens": "JNS",
            "Jonas": "JNS",
        }

        for idx_a, idx_b in pairs:
            name_a = names[idx_a]
            name_b = names[idx_b]
            code_a = phonetic_map[name_a]
            code_b = phonetic_map[name_b]
            assert code_a != code_b, f"Pair ({name_a}, {name_b}) has same phonetic code: {code_a}"

    def test_cross_cluster_pairs_unique(self, initialized_db):
        """Cross-cluster pairs should be unique (no duplicates)."""
        names_data = [
            ("Anna", "Female", "AN", ""),
            ("Hannah", "Female", "AN", ""),
            ("Peter", "Male", "PTR", ""),
            ("Jens", "Male", "JNS", ""),
        ]

        with database.get_connection() as conn:
            for name, gender, primary, secondary in names_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO names
                    (name, gender, phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, gender, primary, secondary),
                )

        names = ["Anna", "Hannah", "Peter", "Jens"]
        clusters = _group_names_by_phonetic(names)

        rng = np.random.default_rng(42)
        pairs = _select_cross_cluster_pairs(clusters, n_pairs=20, rng=rng)

        # Check for duplicates (normalized order)
        pair_set = set()
        for idx_a, idx_b in pairs:
            # Already normalized in function, but verify
            assert idx_a < idx_b, "Pair indices should be normalized"
            pair = (idx_a, idx_b)
            assert pair not in pair_set, f"Duplicate pair found: {pair}"
            pair_set.add(pair)


class TestActiveLearningPhoneticDiversity:
    """Tests for active learning with phonetic diversity."""

    def test_active_learning_prefers_phonetic_diversity(self, initialized_db):
        """
        Given names in different phonetic clusters,
        selection should prefer cross-cluster pairs.
        """
        # Insert names across multiple phonetic clusters
        names_data = [
            ("Anna", "Female", "AN", ""),
            ("Anne", "Female", "AN", ""),
            ("Annie", "Female", "AN", ""),
            ("Peter", "Male", "PTR", ""),
            ("Petra", "Female", "PTR", ""),
            ("Jens", "Male", "JNS", ""),
            ("Jonas", "Male", "JNS", ""),
            ("Lars", "Male", "LRS", ""),
        ]

        with database.get_connection() as conn:
            for name, gender, primary, secondary in names_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO names
                    (name, gender, phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, gender, primary, secondary),
                )

        names = ["Anna", "Anne", "Annie", "Peter", "Petra", "Jens", "Jonas", "Lars"]

        # Get features using the actual feature extractor
        extractor = get_or_create_feature_extractor()
        feature_names = extractor.get_feature_names()

        # Create model with actual feature names
        model = BradleyTerryModel(feature_names, prior_variance=1.0)
        model.rng = np.random.default_rng(42)

        # Get features
        features = extractor.batch_extract(
            names,
            ["Female", "Female", "Female", "Male", "Female", "Male", "Male", "Male"],
            [None] * len(names),
        )

        # Select multiple pairs and verify diversity
        cross_cluster_count = 0
        total_pairs = 20

        phonetic_map = {
            "Anna": "AN",
            "Anne": "AN",
            "Annie": "AN",
            "Peter": "PTR",
            "Petra": "PTR",
            "Jens": "JNS",
            "Jonas": "JNS",
            "Lars": "LRS",
        }

        for _ in range(total_pairs):
            idx_a, idx_b, name_a, name_b = model.select_pair(features, names)
            if phonetic_map[name_a] != phonetic_map[name_b]:
                cross_cluster_count += 1

        # With high probability, most pairs should be cross-cluster
        # Since there are many cross-cluster options available
        assert cross_cluster_count >= total_pairs * 0.7, (
            f"Expected mostly cross-cluster pairs, got {cross_cluster_count}/{total_pairs}"
        )

    def test_fallback_when_all_names_same_phonetic(self, initialized_db):
        """
        When all names share phonetic code,
        should fallback to random selection.
        """
        # Insert names all with same phonetic code
        names_data = [
            ("Anna", "Female", "AN", ""),
            ("Anne", "Female", "AN", ""),
            ("Annie", "Female", "AN", ""),
            ("Ann", "Female", "AN", ""),
        ]

        with database.get_connection() as conn:
            for name, gender, primary, secondary in names_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO names
                    (name, gender, phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, gender, primary, secondary),
                )

        names = ["Anna", "Anne", "Annie", "Ann"]
        clusters = _group_names_by_phonetic(names)

        # Should only have one cluster
        assert len(clusters) == 1
        assert "AN" in clusters

        # Try to select cross-cluster pairs - should return empty
        rng = np.random.default_rng(42)
        pairs = _select_cross_cluster_pairs(clusters, n_pairs=10, rng=rng)
        assert len(pairs) == 0, "Should return empty when only one cluster"

        # But model.select_pair should still work (fallback)
        extractor = get_or_create_feature_extractor()
        feature_names = extractor.get_feature_names()
        model = BradleyTerryModel(feature_names, prior_variance=1.0)
        model.rng = np.random.default_rng(42)

        features = extractor.batch_extract(
            names,
            ["Female"] * len(names),
            [None] * len(names),
        )

        # Should successfully select a pair (fallback to random)
        idx_a, idx_b, name_a, name_b = model.select_pair(features, names)
        assert name_a in names
        assert name_b in names
        assert name_a != name_b


class TestThompsonSampling:
    """Tests for Thompson sampling integration."""

    def test_thompson_sampling_selects_uncertain_pairs(self, initialized_db):
        """
        With high uncertainty, should select pairs with utility difference near 0.5.
        """
        # Insert diverse names
        names_data = [
            ("Anna", "Female", "AN", ""),
            ("Peter", "Male", "PTR", ""),
            ("Jens", "Male", "JNS", ""),
            ("Lars", "Male", "LRS", ""),
        ]

        with database.get_connection() as conn:
            for name, gender, primary, secondary in names_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO names
                    (name, gender, phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, gender, primary, secondary),
                )

        names = ["Anna", "Peter", "Jens", "Lars"]

        # Get features first to determine feature dimension
        extractor = get_or_create_feature_extractor()
        feature_names = extractor.get_feature_names()

        # Create model with high uncertainty (prior only, no training)
        model = BradleyTerryModel(feature_names, prior_variance=1.0)
        model.rng = np.random.default_rng(42)

        # Get features
        features = extractor.batch_extract(
            names,
            ["Female", "Male", "Male", "Male"],
            [None] * len(names),
        )

        # Sample multiple times and check that different pairs are selected
        # (Thompson sampling should explore)
        selected_pairs = set()
        for _ in range(50):
            idx_a, idx_b, name_a, name_b = model.select_pair(features, names)
            pair = tuple(sorted([name_a, name_b]))
            selected_pairs.add(pair)

        # With high uncertainty, should see variety in selected pairs
        assert len(selected_pairs) >= 2, (
            f"Thompson sampling should explore, got only {len(selected_pairs)} unique pairs"
        )


class TestBatchSelection:
    """Tests for batch candidate selection."""

    def test_batch_selection_returns_unique_pairs(self, initialized_db):
        """
        Batch selection should return unique pairs (no duplicates).
        """
        # Insert names with diverse phonetic codes
        names_data = [
            ("Anna", "Female", "AN", ""),
            ("Anne", "Female", "AN", ""),
            ("Peter", "Male", "PTR", ""),
            ("Petra", "Female", "PTR", ""),
            ("Jens", "Male", "JNS", ""),
            ("Jonas", "Male", "JNS", ""),
        ]

        with database.get_connection() as conn:
            for name, gender, primary, secondary in names_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO names
                    (name, gender, phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, gender, primary, secondary),
                )

        names = ["Anna", "Anne", "Peter", "Petra", "Jens", "Jonas"]

        # Use the utils function for batch selection
        # Mock the model to avoid database dependencies in feature extraction
        with patch(
            "st_name_ranking.active_learning.selection.get_or_initialize_active_learning_model",
        ) as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.select_top_k_pairs.return_value = [
                NamePair(idx_a=0, idx_b=2, name_a="Anna", name_b="Peter"),
                NamePair(idx_a=1, idx_b=4, name_a="Anne", name_b="Jens"),
                NamePair(idx_a=3, idx_b=5, name_a="Petra", name_b="Jonas"),
            ]

            pairs = select_candidate_batch(names, batch_size=3)

        # Verify pairs are unique
        pair_set = set()
        for name_a, name_b in pairs:
            pair = tuple(sorted([name_a, name_b]))
            assert pair not in pair_set, f"Duplicate pair found: {pair}"
            pair_set.add(pair)

    def test_batch_selection_respects_batch_size(self, initialized_db):
        """Batch selection should return exactly batch_size pairs."""
        names_data = [
            ("Anna", "Female", "AN", ""),
            ("Peter", "Male", "PTR", ""),
            ("Jens", "Male", "JNS", ""),
            ("Lars", "Male", "LRS", ""),
            ("Ole", "Male", "AL", ""),
            ("Erik", "Male", "ARK", ""),
        ]

        with database.get_connection() as conn:
            for name, gender, primary, secondary in names_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO names
                    (name, gender, phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, gender, primary, secondary),
                )

        names = ["Anna", "Peter", "Jens", "Lars", "Ole", "Erik"]

        # Test with different batch sizes
        for batch_size in [1, 2, 3]:
            with patch(
                "st_name_ranking.active_learning.selection.get_or_initialize_active_learning_model",
            ) as mock_get_model:
                mock_model = mock_get_model.return_value
                # Return batch_size unique pairs
                mock_pairs = [
                    NamePair(idx_a=i, idx_b=i + batch_size, name_a=names[i], name_b=names[i + batch_size])
                    for i in range(batch_size)
                ]
                mock_model.select_top_k_pairs.return_value = mock_pairs

                pairs = select_candidate_batch(names, batch_size=batch_size)
                assert len(pairs) == batch_size, f"Expected {batch_size} pairs, got {len(pairs)}"


class TestPhoneticFeatures:
    """Tests for phonetic feature extraction."""

    def test_phonetic_features_extracted_correctly(self, initialized_db):
        """
        Phonetic features should be part of feature vector.
        """
        from st_name_ranking.learning.features import extract_phonetic_features

        # Test feature extraction
        features = extract_phonetic_features("Anna")

        # Should have phonetic position features
        assert "phonetic_pos_0" in features
        assert "phonetic_pos_1" in features
        assert "phonetic_pos_2" in features
        assert "phonetic_pos_3" in features

        # Should have length and vowel features
        assert "phonetic_length" in features
        assert "contains_vowels" in features
        assert "has_secondary" in features

        # Values should be normalized to [0, 1]
        for key, value in features.items():
            assert 0.0 <= value <= 1.0, f"Feature {key} = {value} not in [0, 1]"

    def test_phonetic_features_different_for_different_names(self, initialized_db):
        """Different names should have different phonetic features."""
        from st_name_ranking.learning.features import extract_phonetic_features

        # Names with different phonetic codes
        features_anna = extract_phonetic_features("Anna")
        features_peter = extract_phonetic_features("Peter")
        features_jens = extract_phonetic_features("Jens")

        # Position features should differ
        assert features_anna["phonetic_pos_0"] != features_peter["phonetic_pos_0"], (
            "Anna and Peter should have different first phonetic char"
        )

        # All three should have different features
        anna_vec = [features_anna[f"phonetic_pos_{i}"] for i in range(4)]
        peter_vec = [features_peter[f"phonetic_pos_{i}"] for i in range(4)]
        jens_vec = [features_jens[f"phonetic_pos_{i}"] for i in range(4)]

        assert anna_vec != peter_vec, "Anna and Peter phonetic vectors should differ"
        assert peter_vec != jens_vec, "Peter and Jens phonetic vectors should differ"


class TestPhoneticCache:
    """Tests for phonetic code caching behavior."""

    def test_phonetic_cache_improves_performance(self, initialized_db):
        """
        Repeated phonetic lookups should use cache.
        """
        # Insert test names
        names_data = [
            ("Anna", "Female", "AN", ""),
            ("Peter", "Male", "PTR", ""),
            ("Jens", "Male", "JNS", ""),
        ]

        with database.get_connection() as conn:
            for name, gender, primary, secondary in names_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO names
                    (name, gender, phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, gender, primary, secondary),
                )

        names = ["Anna", "Peter", "Jens"]

        # Clear any existing cache
        _get_phonetic_codes_cached.cache_clear()

        # First call - should hit database
        start_time = time.perf_counter()
        result1 = _get_phonetic_codes_cached(tuple(names))
        first_call_time = time.perf_counter() - start_time

        # Second call - should hit cache
        start_time = time.perf_counter()
        result2 = _get_phonetic_codes_cached(tuple(names))
        second_call_time = time.perf_counter() - start_time

        # Results should be identical
        assert result1 == result2

        # Cache info should show one miss and one hit
        cache_info = _get_phonetic_codes_cached.cache_info()
        assert cache_info.hits >= 1, "Expected at least one cache hit"
        assert cache_info.misses >= 1, "Expected at least one cache miss"

        # Second call should be faster (or at least not significantly slower)
        # This is a loose assertion since timing can vary
        assert second_call_time < first_call_time * 2, (
            f"Cached call ({second_call_time:.6f}s) should not be much slower than first call ({first_call_time:.6f}s)"
        )

    def test_batch_phonetic_computation(self, initialized_db):
        """Test batch computation of phonetic codes."""
        names = ["Anna", "Peter", "Jens", "Smith", "Johnson"]

        # Compute codes in batch
        codes = batch_compute_phonetic_codes(names)

        # Should have result for each name
        assert len(codes) == len(names)
        for name in names:
            assert name in codes
            primary, secondary = codes[name]
            assert isinstance(primary, str)
            assert primary != ""  # Should have a primary code
            assert isinstance(secondary, str)

    def test_phonetic_similarity_function(self):
        """Test phonetic similarity computation."""
        from st_name_ranking.phonetic_similarity import phonetic_similarity

        # Same name should have similarity 1.0
        assert phonetic_similarity("Anna", "Anna") == 1.0

        # Known similar names (Smith/Smyth often have same code)
        score = phonetic_similarity("Smith", "Smyth")
        assert score in [0.0, 0.5, 1.0], "Similarity should be 0, 0.5, or 1.0"

        # Different names likely have 0.0
        score = phonetic_similarity("Anna", "Xylophone")
        assert score == 0.0 or score == 0.5


class TestUtilsIntegration:
    """Integration tests for utility functions."""

    def test_select_candidates_with_diverse_phonetics(self, initialized_db):
        """
        Test that select_candidates produces valid results with diverse phonetics.
        """
        # Insert diverse names
        names_data = [
            ("Anna", "Female", "AN", ""),
            ("Peter", "Male", "PTR", ""),
            ("Jens", "Male", "JNS", ""),
            ("Lars", "Male", "LRS", ""),
            ("Ole", "Male", "AL", ""),
        ]

        with database.get_connection() as conn:
            for name, gender, primary, secondary in names_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO names
                    (name, gender, phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, gender, primary, secondary),
                )

        names = ["Anna", "Peter", "Jens", "Lars", "Ole"]

        # Get features for these names
        extractor = get_or_create_feature_extractor()
        features = extractor.batch_extract(
            names,
            ["Female", "Male", "Male", "Male", "Male"],
            [None] * len(names),
        )

        # Select candidates
        name_a, name_b = select_candidates(names, features)

        # Should return valid names
        assert name_a in names
        assert name_b in names
        assert name_a != name_b

    def test_select_candidates_fallback(self, initialized_db):
        """Test fallback behavior when model fails."""
        names_data = [
            ("Anna", "Female", "AN", ""),
            ("Peter", "Male", "PTR", ""),
        ]

        with database.get_connection() as conn:
            for name, gender, primary, secondary in names_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO names
                    (name, gender, phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, gender, primary, secondary),
                )

        names = ["Anna", "Peter"]

        # Force model to fail by mocking
        with patch(
            "st_name_ranking.active_learning.selection.get_or_initialize_active_learning_model",
        ) as mock_get_model:
            mock_get_model.side_effect = RuntimeError("Model failure")

            # Should fallback to basic selection
            name_a, name_b = select_candidates(names)

            assert name_a in names
            assert name_b in names
            assert name_a != name_b


class TestModelStatePersistence:
    """Tests for model state persistence with phonetic features."""

    def test_model_save_and_load_preserves_state(self, initialized_db):
        """Test that model state can be saved and loaded."""
        feature_names = ["phonetic_pos_0", "phonetic_pos_1", "phonetic_pos_2"]

        # Create and train model
        model1 = BradleyTerryModel(feature_names, prior_variance=1.0)

        # Simulate some training
        features_a = np.array([0.3, 0.5, 0.7])
        features_b = np.array([0.2, 0.4, 0.6])
        model1.update(features_a, features_b, -1)  # A preferred

        # Save to database
        model1.save_to_db()

        # Create new model and load
        model2 = BradleyTerryModel(feature_names)
        loaded = model2.load_from_db()

        assert loaded, "Model should load successfully"

        # Verify state is preserved
        np.testing.assert_array_almost_equal(
            model1.state.weight_mean,
            model2.state.weight_mean,
        )
        np.testing.assert_array_almost_equal(
            model1.state.weight_cov,
            model2.state.weight_cov,
        )
        assert model1.state.training_samples == model2.state.training_samples


class TestPhoneticCodesBatchFunction:
    """Tests for get_phonetic_codes_batch function."""

    def test_get_phonetic_codes_batch(self, initialized_db):
        """Test batch retrieval of phonetic codes from database."""
        names_data = [
            ("Anna", "Female", "AN", "AN"),
            ("Peter", "Male", "PTR", ""),
            ("Jens", "Male", "JNS", ""),
        ]

        with database.get_connection() as conn:
            for name, gender, primary, secondary in names_data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO names
                    (name, gender, phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, gender, primary, secondary),
                )

        # Get phonetic codes in batch
        result = database.get_phonetic_codes_batch(["Anna", "Peter", "Jens"])

        assert len(result) == 3
        assert result["Anna"] == ("AN", "AN")
        assert result["Peter"] == ("PTR", "")
        assert result["Jens"] == ("JNS", "")

    def test_get_phonetic_codes_batch_missing_names(self, initialized_db):
        """Test batch retrieval with some missing names."""
        # Insert only one name
        with database.get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO names
                (name, gender, phonetic_primary, phonetic_secondary)
                VALUES (?, ?, ?, ?)
                """,
                ("Anna", "Female", "AN", ""),
            )

        # Request multiple names, only one exists
        result = database.get_phonetic_codes_batch(["Anna", "NonExistent", "AlsoMissing"])

        # Should only return the existing name
        assert len(result) == 1
        assert "Anna" in result
        assert "NonExistent" not in result
        assert "AlsoMissing" not in result
