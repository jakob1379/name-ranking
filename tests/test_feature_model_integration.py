"""Integration tests for Feature Extraction + Model Updates.

Tests the integration between:
- features.py: FeatureExtractor class
- model.py: BradleyTerryModel class
- utils.py: get_name_features(), get_names_features()

Critical paths tested:
1. Feature extraction produces valid input for model
2. Batch feature extraction consistency
3. Feature caching behavior
4. Missing name handling (graceful degradation)
5. Feature dimension consistency across calls
"""

import numpy as np
import pytest

from st_name_ranking.database import get_connection
from st_name_ranking.features import FeatureExtractor, extract_all_features, extract_suffix_features
from st_name_ranking.model import BradleyTerryModel
from st_name_ranking.utils import get_name_features, get_names_features

# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def test_names_with_metadata(initialized_db):
    """Insert test names with known gender/origin into database."""
    test_names = [
        ("Anna", "Female", "Nordic"),
        ("Peter", "Male", "European"),
        ("Maria", "Female", "European"),
        ("Jens", "Male", "Nordic"),
        ("Søren", "Male", "Nordic"),  # Danish letters
        ("Zhang", "Male", "Asian"),
        ("Amara", "Female", "African"),
        ("Youssef", "Male", "Middle Eastern"),
        ("Emma", "Female", None),  # No origin
        ("X Æ A-12", "Unisex", "International"),  # Edge case name
    ]

    with get_connection() as conn:
        for name, gender, origin in test_names:
            conn.execute(
                """
                INSERT OR REPLACE INTO names
                (name, gender, origin_region, phonetic_primary, phonetic_secondary)
                VALUES (?, ?, ?, 'TEST', 'TEST')
                """,
                (name, gender, origin),
            )

    return [n[0] for n in test_names]


@pytest.fixture
def feature_extractor():
    """Create a fresh FeatureExtractor instance."""
    return FeatureExtractor()


@pytest.fixture
def model(feature_extractor):
    """Create a BradleyTerryModel with correct feature dimension."""
    feature_names = feature_extractor.get_feature_names()
    return BradleyTerryModel(feature_names)


# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------


def test_extracted_features_work_with_model_update(initialized_db, test_names_with_metadata):
    """Extract features for two names and update model successfully.

    Verifies the end-to-end flow:
    1. Extract features from database
    2. Create model with correct dimensions
    3. Update model with comparison
    4. Verify model state updated
    """
    # Get two test names
    name_a = test_names_with_metadata[0]  # Anna
    name_b = test_names_with_metadata[1]  # Peter

    # Extract features using utils (queries database)
    features_a = get_name_features(name_a)
    features_b = get_name_features(name_b)

    # Verify we got numpy arrays
    assert isinstance(features_a, np.ndarray)
    assert isinstance(features_b, np.ndarray)
    assert features_a.ndim == 1
    assert features_b.ndim == 1

    # Create model with matching feature dimension
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()
    model = BradleyTerryModel(feature_names)

    # Verify dimensions match
    assert features_a.shape[0] == model.d
    assert features_b.shape[0] == model.d

    # Store initial training samples
    initial_samples = model.state.training_samples

    # Update model with preference (name_a preferred over name_b)
    model.update(features_a, features_b, -1)

    # Verify model was updated
    assert model.state.training_samples == initial_samples + 1
    assert model.state.weight_mean is not None
    assert model.state.weight_cov is not None

    # Verify we can get utility predictions
    utility_a = model.get_utility(features_a.reshape(1, -1))
    utility_b = model.get_utility(features_b.reshape(1, -1))
    assert isinstance(utility_a, np.ndarray)
    assert isinstance(utility_b, np.ndarray)


def test_batch_and_single_extraction_equivalent(initialized_db, test_names_with_metadata):
    """Batch extraction should match individual extractions.

    Verifies that:
    - batch_extract() returns same results as multiple extract() calls
    - get_names_features() matches multiple get_name_features() calls
    """
    # Select subset of names for testing
    test_subset = test_names_with_metadata[:4]

    # Method 1: Individual extraction via utils
    individual_features = [get_name_features(name) for name in test_subset]
    individual_matrix = np.stack(individual_features, axis=0)

    # Method 2: Batch extraction via utils
    batch_matrix = get_names_features(test_subset)

    # Method 3: Direct FeatureExtractor batch_extract
    extractor = FeatureExtractor()
    with get_connection() as conn:
        details = []
        for name in test_subset:
            cursor = conn.execute(
                "SELECT gender, origin_region FROM names WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            details.append(row or (None, None))

    genders = [d[0] for d in details]
    origins = [d[1] for d in details]
    direct_batch = extractor.batch_extract(test_subset, genders, origins)

    # All methods should produce equivalent results
    np.testing.assert_array_almost_equal(individual_matrix, batch_matrix, decimal=10)
    np.testing.assert_array_almost_equal(individual_matrix, direct_batch, decimal=10)

    # Verify shapes
    expected_shape = (len(test_subset), len(extractor.get_feature_names()))
    assert individual_matrix.shape == expected_shape
    assert batch_matrix.shape == expected_shape
    assert direct_batch.shape == expected_shape


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"genders": ["Female"]}, "genders"),
        ({"origin_regions": ["Nordic"]}, "origin_regions"),
        ({"name_ids": [1]}, "name_ids"),
    ],
)
def test_batch_extract_rejects_metadata_length_mismatch(feature_extractor, kwargs, field):
    """Batch metadata must align with every requested name."""
    with pytest.raises(ValueError, match=field):
        feature_extractor.batch_extract(["Anna", "Peter"], **kwargs)


def test_suffix_features_empty_name_returns_zero_features():
    """Suffix features should handle empty names like other feature groups."""
    features = extract_suffix_features("")

    assert features
    assert all(value == 0.0 for value in features.values())


def test_feature_dimensions_consistent(initialized_db, test_names_with_metadata):
    """All feature vectors should have same dimension.

    Tests that:
    - Different names produce same feature dimension
    - Feature dimension matches model expectation
    - Consistent across single and batch extraction
    """
    extractor = FeatureExtractor()
    expected_dim = len(extractor.get_feature_names())

    # Test various name types
    test_cases = [
        ("Anna", "Female", "Nordic"),
        ("Peter", "Male", "European"),
        ("Zhang", "Male", "Asian"),
        ("X", "Female", None),  # Very short name
        ("VeryLongNameIndeed", "Unisex", "International"),  # Long name
    ]

    for name, gender, origin in test_cases:
        # Single extraction
        features_single = extractor.extract(name, gender, origin)
        assert features_single.shape[0] == expected_dim, f"Dimension mismatch for {name}"

        # Via utils (queries DB)
        if name in test_names_with_metadata:
            features_util = get_name_features(name)
            assert features_util.shape[0] == expected_dim, f"Utils dimension mismatch for {name}"

    # Batch extraction should maintain same dimension
    names = [n[0] for n in test_cases]
    genders = [n[1] for n in test_cases]
    origins = [n[2] for n in test_cases]
    batch_features = extractor.batch_extract(names, genders, origins)

    assert batch_features.shape == (len(test_cases), expected_dim)

    # Verify model compatibility
    model = BradleyTerryModel(extractor.get_feature_names())
    assert model.d == expected_dim

    # Model should accept features of this dimension
    model.update(batch_features[0], batch_features[1], -1)


def test_missing_name_returns_default_features(initialized_db):
    """Non-existent names should return default/empty features.

    Verifies graceful degradation when name is not in database.
    """
    # Ensure database is initialized but empty of our test name
    nonexistent_name = "NonExistentNameXYZ123"

    # Verify name doesn't exist
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM names WHERE name = ?",
            (nonexistent_name,),
        )
        assert cursor.fetchone()[0] == 0

    # Should still return features (with None gender/origin)
    features = get_name_features(nonexistent_name)

    # Should be a valid numpy array
    assert isinstance(features, np.ndarray)
    assert features.ndim == 1

    # Extractor should handle None values gracefully
    extractor = FeatureExtractor()
    features_direct = extractor.extract(nonexistent_name, None, None)

    assert isinstance(features_direct, np.ndarray)
    assert features_direct.shape[0] == len(extractor.get_feature_names())

    # Features should be valid (not NaN or Inf)
    assert np.all(np.isfinite(features_direct))

    # Default gender encoding should be equal distribution
    feature_dict, feature_names = extract_all_features(nonexistent_name, None, None)
    assert feature_dict["gender_male"] == 0.333
    assert feature_dict["gender_female"] == 0.333
    assert feature_dict["gender_unisex"] == 0.334

    # Default origin should be International
    assert feature_dict["origin_international"] == 1.0


def test_feature_caching_works_correctly(initialized_db, test_names_with_metadata):
    """Repeated extractions should use cache.

    Verifies:
    - Same inputs return identical arrays (same object from cache)
    - Cache is used for repeated calls
    - Different inputs produce different cache entries
    """
    extractor = FeatureExtractor()
    name = test_names_with_metadata[0]  # Anna
    gender = "Female"
    origin = "Nordic"

    # Clear any existing in-memory vector cache
    extractor._local_cache.clear()

    # First extraction - should compute
    features1 = extractor.extract(name, gender, origin)
    cache_key = (name, gender, origin)
    assert cache_key in extractor._local_cache

    # Second extraction - should use cache
    features2 = extractor.extract(name, gender, origin)

    # Should be identical object (from cache)
    assert features1 is features2

    # Different gender should be different cache entry
    features_male = extractor.extract(name, "Male", origin)
    assert features_male is not features1

    # Different origin should be different cache entry
    features_european = extractor.extract(name, gender, "European")
    assert features_european is not features1

    # Different name should be different cache entry
    features_other = extractor.extract("Peter", gender, origin)
    assert features_other is not features1

    # Batch extraction should use cache internally (extract returns cached object)
    # but np.stack creates a new array, so we verify values match, not identity
    batch_result = extractor.batch_extract([name], [gender], [origin])
    # First element should have same values as cached features
    np.testing.assert_array_equal(batch_result[0], features1)


def test_gender_origin_encoded_in_features(initialized_db, test_names_with_metadata):
    """Verify gender and origin are properly encoded in feature vectors.

    Tests that:
    - Different genders produce different feature values
    - Different origins produce different feature values
    - Encodings are one-hot like (0 or 1 for known values)
    """
    extractor = FeatureExtractor()
    name = "TestName"

    # Test gender encoding
    gender_tests = [
        ("Male", {"gender_male": 1.0, "gender_female": 0.0, "gender_unisex": 0.0}),
        ("Female", {"gender_male": 0.0, "gender_female": 1.0, "gender_unisex": 0.0}),
        ("Unisex", {"gender_male": 0.0, "gender_female": 0.0, "gender_unisex": 1.0}),
        (None, {"gender_male": 0.333, "gender_female": 0.333, "gender_unisex": 0.334}),
    ]

    for gender, expected in gender_tests:
        features_dict, feature_names = extract_all_features(name, gender, "Nordic")
        for key, expected_val in expected.items():
            assert key in features_dict, f"Missing feature: {key}"
            assert features_dict[key] == pytest.approx(expected_val, abs=0.001), (
                f"Gender {gender}: {key} should be {expected_val}, got {features_dict[key]}"
            )

    # Test origin encoding
    origin_tests = [
        "Nordic",
        "European",
        "Asian",
        "African",
        "Middle Eastern",
        "International",
    ]

    for origin in origin_tests:
        features_dict, _ = extract_all_features(name, "Male", origin)

        # Check that the correct origin feature is 1.0
        expected_key = f"origin_{origin.lower().replace(' ', '_')}"
        assert expected_key in features_dict, f"Missing origin feature: {expected_key}"
        assert features_dict[expected_key] == 1.0, f"Origin {origin}: {expected_key} should be 1.0"

        # Check that other origin features are 0.0
        for key, val in features_dict.items():
            if key.startswith("origin_") and key != expected_key:
                assert val == 0.0, f"Origin {origin}: {key} should be 0.0, got {val}"

    # Test default (unknown) origin defaults to International
    features_dict, _ = extract_all_features(name, "Male", None)
    assert features_dict["origin_international"] == 1.0


def test_batch_extraction_handles_partial_missing(initialized_db):
    """Batch extraction handles mix of existing and missing names gracefully."""
    # Insert only some names
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO names (name, gender, origin_region) VALUES (?, ?, ?)",
            ("ExistingName", "Female", "Nordic"),
        )

    # Mix of existing and non-existing
    names = ["ExistingName", "MissingName1", "MissingName2"]

    # Should not raise error
    features = get_names_features(names)

    # Should return features for all names
    assert features.shape[0] == 3

    # All should have same dimension
    extractor = FeatureExtractor()
    expected_dim = len(extractor.get_feature_names())
    assert features.shape[1] == expected_dim

    # All should be valid (finite) values
    assert np.all(np.isfinite(features))


def test_model_update_batch_with_extracted_features(initialized_db, test_names_with_metadata):
    """Model batch update works with batch-extracted features.

    Tests the full pipeline: batch feature extraction → model batch update
    """
    # Get features for multiple names
    names = test_names_with_metadata[:4]
    features = get_names_features(names)

    # Create model
    extractor = FeatureExtractor()
    model = BradleyTerryModel(extractor.get_feature_names())

    # Create batch comparisons
    comparisons = [
        (features[0], features[1], -1),  # Anna beats Peter
        (features[2], features[3], 1),  # Jens beats Maria
        (features[0], features[2], 0),  # Anna draws with Maria
    ]

    initial_samples = model.state.training_samples

    # Batch update
    model.update_batch(comparisons)

    # Verify all comparisons were processed
    assert model.state.training_samples == initial_samples + 3

    # Verify model can still make predictions
    utilities = model.get_utility(features)
    assert utilities.shape[0] == len(names)
    assert np.all(np.isfinite(utilities))


def test_phonetic_features_consistency(initialized_db):
    """Phonetic features are consistent for similar names."""
    extractor = FeatureExtractor()

    # Names with similar phonetic patterns
    similar_names = ["Anna", "Ana", "Anne", "Ann"]

    features_list = []
    for name in similar_names:
        feat_dict, _ = extract_all_features(name, "Female", "European")
        # Extract just phonetic features
        phonetic_feats = {k: v for k, v in feat_dict.items() if k.startswith("phonetic_")}
        features_list.append(phonetic_feats)

    # All should have same phonetic feature keys
    key_sets = [set(f.keys()) for f in features_list]
    assert all(keys == key_sets[0] for keys in key_sets), "Phonetic features inconsistent"

    # Similar names should have some similar phonetic feature values
    # (at least more similar than completely different names)
    different_name = "Zxqwty"
    diff_dict, _ = extract_all_features(different_name, "Female", "European")
    diff_phonetic = {k: v for k, v in diff_dict.items() if k.startswith("phonetic_")}

    # Similar names should be more similar to each other than to different name
    anna_vec = np.array(list(features_list[0].values()))
    ana_vec = np.array(list(features_list[1].values()))
    diff_vec = np.array(list(diff_phonetic.values()))

    sim_anna_ana = np.linalg.norm(anna_vec - ana_vec)
    sim_anna_diff = np.linalg.norm(anna_vec - diff_vec)

    # Anna-Ana should be more similar than Anna-Zxqwty
    assert sim_anna_ana < sim_anna_diff


def test_linguistic_features_edge_cases(initialized_db):
    """Linguistic features handle edge cases properly."""
    edge_cases = [
        ("", "Female", "European"),  # Empty string
        ("A", "Male", "Nordic"),  # Single character
        ("ÆØÅ", "Unisex", "Nordic"),  # Danish letters only
        ("A" * 50, "Female", "International"),  # Very long name
    ]

    extractor = FeatureExtractor()

    for name, gender, origin in edge_cases:
        # Should not raise exception
        features_dict, feature_names = extract_all_features(name, gender, origin)

        # All values should be finite
        vector = extractor.extract(name, gender, origin)
        assert np.all(np.isfinite(vector)), f"Non-finite values for name: '{name}'"

        # Check specific linguistic features
        assert "name_length" in features_dict
        assert "vowel_ratio" in features_dict
        assert "consonant_ratio" in features_dict

        # Ratios should be in valid range [0, 1]
        assert 0 <= features_dict["vowel_ratio"] <= 1
        assert 0 <= features_dict["consonant_ratio"] <= 1


def test_model_save_load_with_extracted_features(initialized_db, test_names_with_metadata):
    """Model can be saved and loaded, maintaining feature compatibility.

    Verifies the database integration path for model persistence.
    """
    # Extract features and train model
    name_a = test_names_with_metadata[0]
    name_b = test_names_with_metadata[1]

    features_a = get_name_features(name_a)
    features_b = get_name_features(name_b)

    extractor = FeatureExtractor()
    model = BradleyTerryModel(extractor.get_feature_names())

    # Train with some updates
    for _ in range(5):
        model.update(features_a, features_b, -1)

    original_mean = model.state.weight_mean.copy()
    original_cov = model.state.weight_cov.copy()
    original_samples = model.state.training_samples

    # Save to database
    model.save_to_db()

    # Load into new model instance
    loaded_model = BradleyTerryModel(extractor.get_feature_names())
    success = loaded_model.load_from_db()

    assert success, "Model should load successfully from database"

    # Verify state matches
    np.testing.assert_array_almost_equal(loaded_model.state.weight_mean, original_mean)
    np.testing.assert_array_almost_equal(loaded_model.state.weight_cov, original_cov)
    assert loaded_model.state.training_samples == original_samples
    assert loaded_model.feature_names == extractor.get_feature_names()

    # Loaded model should work with same features
    utility = loaded_model.get_utility(features_a.reshape(1, -1))
    assert np.isfinite(utility[0])


def test_feature_extractor_get_feature_names_consistency(initialized_db):
    """Feature names are consistent and ordered."""
    extractor = FeatureExtractor()

    # Get feature names multiple times
    names1 = extractor.get_feature_names()
    names2 = extractor.get_feature_names()

    # Should be identical
    assert names1 == names2

    # Should be sorted
    assert names1 == sorted(names1)

    # Should include expected feature categories
    feature_categories = {
        "phonetic": any("phonetic_" in n for n in names1),
        "gender": any("gender_" in n for n in names1),
        "origin": any("origin_" in n for n in names1),
        "linguistic": any(n in names1 for n in ["name_length", "vowel_ratio"]),
    }

    assert all(feature_categories.values()), f"Missing categories: {feature_categories}"
