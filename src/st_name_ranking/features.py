"""Feature extraction for names.

Extracts phonetic, linguistic, and metadata features for preference learning.
All features are normalized to [0, 1] range for model compatibility.
"""

import logging
from collections.abc import Sequence

import numpy as np
import pyphen
from metaphone import doublemetaphone

logger = logging.getLogger(__name__)

# Global instances for performance
_pyphen = pyphen.Pyphen(lang="da")  # Danish hyphenation for syllable counting


def extract_phonetic_features(name: str) -> dict[str, float]:
    """Extract phonetic features using Double Metaphone encoding.
    Returns a dictionary of phonetic features.
    """
    try:
        # Get Double Metaphone encoding (primary and secondary)
        primary, secondary = doublemetaphone(name)

        # Handle empty secondary
        primary = primary or ""
        secondary = secondary or ""

        # For simplicity, use first 4 characters of primary encoding, padded
        encoding_str = primary[:4].ljust(4, "_")

        # Create one-hot like features for common phonetic patterns
        features = {}

        # Position-specific phonetic features
        for i, char in enumerate(encoding_str):
            features[f"phonetic_pos_{i}"] = ord(char) / 255.0  # Normalized ASCII

        # Length of encoding
        features["phonetic_length"] = len(primary) / 10.0

        # Contains vowel-like sounds in primary encoding
        vowels = {"A", "E", "I", "O", "U"}
        encoding_chars = set(primary)
        features["contains_vowels"] = 1.0 if vowels.intersection(encoding_chars) else 0.0

        # Whether secondary encoding exists
        features["has_secondary"] = 1.0 if secondary else 0.0

        return features
    except (AttributeError, ValueError, TypeError) as e:
        logger.warning("Failed to extract phonetic features for '%s': %s", name, e)
        # Return empty features
        return {f"phonetic_pos_{i}": 0.0 for i in range(4)} | {
            "phonetic_length": 0.0,
            "contains_vowels": 0.0,
            "has_secondary": 0.0,
        }


def extract_linguistic_features(name: str) -> dict[str, float]:
    """Extract linguistic features: length, syllables, vowel ratio, etc."""
    name_lower = name.lower()

    # Basic length features
    features = {
        "name_length": len(name) / 20.0,  # Normalize, assuming max 20 chars
        "name_length_squared": (len(name) ** 2) / 400.0,
    }

    # Syllable count (approximate for Danish)
    try:
        hyphenated = _pyphen.inserted(name_lower)
        syllable_count = hyphenated.count("-") + 1
        features["syllable_count"] = syllable_count / 6.0  # Normalize
        features["syllable_density"] = syllable_count / max(len(name), 1)
    except (AttributeError, ValueError):
        # Fallback: rough estimate based on vowels
        vowel_count = sum(1 for c in name_lower if c in "aeiouyæøå")
        syllable_count = max(1, vowel_count // 2)
        features["syllable_count"] = syllable_count / 6.0
        features["syllable_density"] = syllable_count / max(len(name), 1)

    # Vowel/consonant ratios
    vowels = sum(1 for c in name_lower if c in "aeiouyæøå")
    consonants = len(name) - vowels
    features["vowel_ratio"] = vowels / max(len(name), 1)
    features["consonant_ratio"] = consonants / max(len(name), 1)

    # Position features
    if len(name) >= 1:
        features["first_letter"] = ord(name_lower[0]) / 255.0
        features["last_letter"] = ord(name_lower[-1]) / 255.0
    else:
        features["first_letter"] = 0.0
        features["last_letter"] = 0.0

    # Contains specific letters common in Danish
    danish_letters = {"æ", "ø", "å"}
    features["contains_danish"] = 1.0 if any(c in danish_letters for c in name_lower) else 0.0

    return features


def extract_gender_features(gender: str | None) -> dict[str, float]:
    """One-hot encoding for gender."""
    if gender == "Male":
        return {"gender_male": 1.0, "gender_female": 0.0, "gender_unisex": 0.0}
    if gender == "Female":
        return {"gender_male": 0.0, "gender_female": 1.0, "gender_unisex": 0.0}
    if gender == "Unisex":
        return {"gender_male": 0.0, "gender_female": 0.0, "gender_unisex": 1.0}
    # Unknown/All - equal distribution
    return {
        "gender_male": 0.333,
        "gender_female": 0.333,
        "gender_unisex": 0.334,
    }


def extract_origin_features(origin_region: str | None) -> dict[str, float]:
    """One-hot encoding for origin region."""
    regions = [
        "Nordic",
        "European",
        "Asian",
        "African",
        "Middle Eastern",
        "International",
    ]
    features = {f"origin_{region.lower().replace(' ', '_')}": 0.0 for region in regions}

    if origin_region and origin_region in regions:
        features[f"origin_{origin_region.lower().replace(' ', '_')}"] = 1.0
    else:
        # Default to International
        features["origin_international"] = 1.0

    return features


def extract_all_features(
    name: str,
    gender: str | None = None,
    origin_region: str | None = None,
    *,
    include_phonetic: bool = True,
    include_linguistic: bool = True,
    include_metadata: bool = True,
) -> tuple[dict[str, float], list[str]]:
    """Extract all features for a name.

    Returns:
        Tuple of (feature_dict, feature_names) where feature_dict maps
        feature names to values, and feature_names is the ordered list
        of feature names for consistent vectorization.

    """
    features = {}

    if include_phonetic:
        features.update(extract_phonetic_features(name))

    if include_linguistic:
        features.update(extract_linguistic_features(name))

    if include_metadata:
        features.update(extract_gender_features(gender))
        features.update(extract_origin_features(origin_region))

    # Get ordered feature names (sorted for consistency)
    feature_names = sorted(features.keys())

    return features, feature_names


def features_to_vector(
    features: dict[str, float],
    feature_names: list[str],
) -> np.ndarray:
    """Convert feature dictionary to numpy vector in consistent order."""
    return np.array(
        [features.get(name, 0.0) for name in feature_names],
        dtype=np.float32,
    )


class FeatureExtractor:
    """Cached feature extractor for batch processing."""

    def __init__(self) -> None:
        self._feature_names = None
        self._feature_cache = {}

    def get_feature_names(self) -> list[str]:
        """Get the ordered list of feature names.
        Computed once on first call.
        """
        if self._feature_names is None:
            # Extract features for a dummy name to get feature names
            dummy_features, feature_names = extract_all_features("Test")
            self._feature_names = feature_names
        return self._feature_names

    def extract(
        self,
        name: str,
        gender: str | None = None,
        origin_region: str | None = None,
    ) -> np.ndarray:
        """Extract feature vector for a name, with caching."""
        cache_key = (name, gender, origin_region)

        if cache_key not in self._feature_cache:
            features, _ = extract_all_features(name, gender, origin_region)
            vector = features_to_vector(features, self.get_feature_names())
            self._feature_cache[cache_key] = vector

        return self._feature_cache[cache_key]

    def batch_extract(
        self,
        names: list[str],
        genders: Sequence[str | None] | None = None,
        origin_regions: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Extract feature vectors for multiple names.
        Returns 2D array of shape (n_names, n_features).
        """
        if genders is None:
            genders = [None] * len(names)  # type: ignore[assignment]
        if origin_regions is None:
            origin_regions = [None] * len(names)  # type: ignore[assignment]

        # Type narrowing: after above assignments, neither is None
        assert genders is not None
        assert origin_regions is not None

        vectors = []
        for name, gender, origin in zip(names, genders, origin_regions):
            vectors.append(self.extract(name, gender, origin))

        return np.stack(vectors, axis=0)
