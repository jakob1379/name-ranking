"""Feature extraction for names.

Extracts phonetic, linguistic, and metadata features for preference learning.
All features are normalized to [0, 1] range for model compatibility.
"""

import logging
import re
from collections.abc import Sequence

import numpy as np
import pyphen
from metaphone import doublemetaphone

from st_name_ranking.persistence.feature_cache import FeatureCache

logger = logging.getLogger(__name__)

_pyphen = pyphen.Pyphen(lang="da")  # Danish hyphenation for syllable counting


def _metadata_sequence[T](
    names: Sequence[str],
    values: Sequence[T | None] | None,
    label: str,
) -> Sequence[T | None]:
    """Return metadata values with the same length as names."""
    if values is None:
        return [None] * len(names)
    if len(values) != len(names):
        msg = f"{label} length ({len(values)}) must match names length ({len(names)})"
        raise ValueError(msg)
    return values


# Sonority scale (higher = more sonorous) for phonetic pleasantness analysis
SONORITY = {
    # Vowels (highest sonority)
    "a": 16,
    "e": 16,
    "i": 16,
    "o": 16,
    "u": 16,
    "y": 16,
    "æ": 16,
    "ø": 16,
    "å": 16,
    # Glides
    "j": 14,
    "w": 14,
    # Liquids
    "l": 12,
    "r": 12,
    # Nasals
    "m": 10,
    "n": 10,
    "ŋ": 10,
    # Fricatives
    "f": 8,
    "v": 8,
    "s": 8,
    "h": 8,
    "ð": 6,
    # Plosives (lowest sonority)
    "p": 2,
    "b": 2,
    "t": 2,
    "d": 2,
    "k": 2,
    "g": 2,
}

# Soft sounds (liquids, nasals, glides, voiced fricatives)
SOFT_SOUNDS = {"l", "r", "m", "n", "v", "w", "j", "y", "ð"}

# Hard sounds (plosives, voiceless fricatives)
HARD_SOUNDS = {"p", "t", "k", "b", "d", "g", "f", "s", "h", "c"}

# Roundness mapping for Bouba-Kiki effect (positive = round/soft, negative = sharp)
ROUNDNESS = {
    "m": 2,
    "n": 1,
    "l": 1,
    "r": 1,
    "u": 2,
    "o": 2,
    "å": 2,
    "ø": 1,
    "æ": 1,
    "y": 1,
    "a": 1,
    "k": -2,
    "t": -2,
    "p": -1,
    "i": -2,
    "e": -1,
    "s": -1,
}

# Danish vowels including æ, ø, å (for cluster detection and CV patterns)
VOWELS = set("aeiouyæøå")
# Regex pattern for consonant clusters (2+ consecutive consonants)
CLUSTER_PATTERN = re.compile(r"[^aeiouyæøå]{2,}")

# Gender-associated suffix patterns from cross-cultural research
FEMININE_SUFFIXES = {
    "a": 0.91,
    "ia": 0.96,
    "na": 0.96,
    "la": 0.93,
    "ta": 0.93,
    "ja": 0.93,
    "ra": 0.92,
    "ka": 0.92,
    "ea": 0.95,
    "ie": 0.86,
    "ette": 1.0,
    "ine": 0.85,
    "elle": 0.88,
}

MASCULINE_SUFFIXES = {
    "o": 0.76,
    "os": 0.92,
    "us": 0.96,
    "as": 0.91,
    "is": 0.80,
    "er": 0.75,
    "or": 0.78,
    "ar": 0.70,
    "ck": 0.91,
    "sh": 0.91,
    "rt": 0.90,
    "ik": 0.90,
    "ef": 0.94,
    "ib": 0.92,
}

FEMININE_PREFIXES = [
    "ann",
    "lil",
    "kat",
    "ros",
    "mai",
    "mel",
    "ell",
    "may",
    "nat",
    "ana",
    "ali",
    "ani",
]

MASCULINE_PREFIXES = [
    "abd",
    "wil",
    "mat",
    "moh",
    "mar",
    "ale",
]

DANISH_VOWELS = set("aeiouyæøå")
SUFFIX_FEATURE_KEYS = [
    "ends_with_a",
    "ends_with_o",
    "ends_with_us",
    "ends_with_ie",
    "ends_with_ette",
    "ends_with_elle",
    "ends_with_ck",
    "ends_with_sh",
    "suffix_feminine_score",
    "suffix_masculine_score",
    "prefix_feminine",
    "prefix_masculine",
    "ends_open_syllable",
    "coda_weight",
    "closed_heavy",
]


def extract_suffix_features(name: str) -> dict[str, float]:
    """Extract suffix and prefix-based gender cue features.

    Returns features based on cross-cultural research showing strong gender
    associations with certain name endings and beginnings.
    """
    if not name:
        return dict.fromkeys(SUFFIX_FEATURE_KEYS, 0.0)

    name_lower = name.lower()
    features: dict[str, float] = {}

    features["ends_with_a"] = 1.0 if name_lower.endswith("a") else 0.0
    features["ends_with_o"] = 1.0 if name_lower.endswith("o") else 0.0
    features["ends_with_us"] = 1.0 if name_lower.endswith("us") else 0.0
    features["ends_with_ie"] = 1.0 if name_lower.endswith("ie") else 0.0
    features["ends_with_ette"] = 1.0 if name_lower.endswith("ette") else 0.0
    features["ends_with_elle"] = 1.0 if name_lower.endswith("elle") else 0.0
    features["ends_with_ck"] = 1.0 if name_lower.endswith("ck") else 0.0
    features["ends_with_sh"] = 1.0 if name_lower.endswith("sh") else 0.0

    fem_scores = [score for suffix, score in FEMININE_SUFFIXES.items() if name_lower.endswith(suffix)]
    features["suffix_feminine_score"] = max(fem_scores) if fem_scores else 0.0

    masc_scores = [score for suffix, score in MASCULINE_SUFFIXES.items() if name_lower.endswith(suffix)]
    features["suffix_masculine_score"] = max(masc_scores) if masc_scores else 0.0

    features["prefix_feminine"] = 1.0 if any(name_lower.startswith(prefix) for prefix in FEMININE_PREFIXES) else 0.0
    features["prefix_masculine"] = 1.0 if any(name_lower.startswith(prefix) for prefix in MASCULINE_PREFIXES) else 0.0

    trailing_consonants = 0
    for char in reversed(name_lower):
        if char in DANISH_VOWELS:
            break
        trailing_consonants += 1

    features["ends_open_syllable"] = 1.0 if name_lower and name_lower[-1] in DANISH_VOWELS else 0.0
    features["coda_weight"] = min(trailing_consonants / 3.0, 1.0)
    features["closed_heavy"] = 1.0 if trailing_consonants >= 2 else 0.0

    return features


def extract_phonetic_features(name: str) -> dict[str, float]:
    """Extract phonetic features using Double Metaphone encoding."""
    try:
        primary, secondary = doublemetaphone(name)
    except (AttributeError, ValueError, TypeError) as e:
        logger.warning("Failed to extract phonetic features for '%s': %s", name, e)
        return {f"phonetic_pos_{i}": 0.0 for i in range(4)} | {
            "phonetic_length": 0.0,
            "contains_vowels": 0.0,
            "has_secondary": 0.0,
        }
    else:
        primary = primary or ""
        secondary = secondary or ""

        encoding_str = primary[:4].ljust(4, "_")

        features = {}

        for i, char in enumerate(encoding_str):
            features[f"phonetic_pos_{i}"] = ord(char) / 255.0

        features["phonetic_length"] = len(primary) / 10.0

        vowels = {"A", "E", "I", "O", "U"}
        encoding_chars = set(primary)
        features["contains_vowels"] = 1.0 if vowels.intersection(encoding_chars) else 0.0

        features["has_secondary"] = 1.0 if secondary else 0.0

        return features


def extract_linguistic_features(name: str) -> dict[str, float]:
    """Extract linguistic features: length, syllables, vowel ratio, etc."""
    name_lower = name.lower()

    features = {
        "name_length": len(name),
        "name_length_squared": len(name) ** 2,
    }

    # Syllable count (approximate for Danish)
    try:
        hyphenated = _pyphen.inserted(name_lower)
        syllable_count = hyphenated.count("-") + 1
    except (AttributeError, ValueError):
        # Fallback: rough estimate based on vowels
        vowel_count = sum(1 for c in name_lower if c in "aeiouyæøå")
        syllable_count = max(1, vowel_count // 2)

    features["syllable_count"] = syllable_count / 6.0
    features["syllable_density"] = syllable_count / max(len(name), 1)

    vowels = sum(1 for c in name_lower if c in "aeiouyæøå")
    consonants = len(name) - vowels
    features["vowel_ratio"] = vowels / max(len(name), 1)
    features["consonant_ratio"] = consonants / max(len(name), 1)

    first_letter = ord(name_lower[0]) / 255.0 if name else 0.0
    last_letter = ord(name_lower[-1]) / 255.0 if name else 0.0
    features["first_letter"] = first_letter
    features["last_letter"] = last_letter

    danish_letters = {"æ", "ø", "å"}
    features["contains_danish"] = 1.0 if any(c in danish_letters for c in name_lower) else 0.0

    return features


def extract_sonority_features(name: str) -> dict[str, float]:
    """Extract sonority profile and sound pleasantness features.

    Sonority hierarchy: vowels > glides > liquids > nasals > fricatives > plosives
    Names with natural sonority profiles are perceived as more pleasant.

    """
    name_lower = name.lower()
    length = max(len(name), 1)

    sonority_values = []
    for char in name_lower:
        if char in SONORITY:
            sonority_values.append(SONORITY[char])

    if not sonority_values:
        return {
            "mean_sonority": 0.0,
            "sonority_variance": 0.0,
            "sonority_range": 0.0,
            "softness_ratio": 0.0,
            "hardness_ratio": 0.0,
            "roundness_score": 0.0,
            "liquid_density": 0.0,
            "nasal_ending": 0.0,
            "stop_ending": 0.0,
        }

    mean_sonority = sum(sonority_values) / len(sonority_values)
    sonority_variance = sum((x - mean_sonority) ** 2 for x in sonority_values) / len(sonority_values)
    sonority_range = max(sonority_values) - min(sonority_values)

    soft_count = sum(1 for c in name_lower if c in SOFT_SOUNDS)
    hard_count = sum(1 for c in name_lower if c in HARD_SOUNDS)
    soft_total = max(soft_count + hard_count, 1)
    softness_ratio = soft_count / soft_total
    hardness_ratio = hard_count / soft_total

    roundness_sum = sum(ROUNDNESS.get(c, 0) for c in name_lower)
    roundness_score = roundness_sum / length if length > 0 else 0.0

    liquid_count = sum(1 for c in name_lower if c in {"l", "r"})
    liquid_density = liquid_count / length

    nasal_ending = 1.0 if name_lower and name_lower[-1] in {"m", "n"} else 0.0

    stop_ending = 1.0 if name_lower and name_lower[-1] in {"p", "t", "k", "b", "d", "g"} else 0.0

    # Normalize mean_sonority to [0, 1] range (max sonority is 16, min is 2)
    mean_sonority_normalized = (mean_sonority - 2.0) / 14.0
    sonority_variance_normalized = sonority_variance / 196.0
    sonority_range_normalized = sonority_range / 14.0

    return {
        "mean_sonority": mean_sonority_normalized,
        "sonority_variance": sonority_variance_normalized,
        "sonority_range": sonority_range_normalized,
        "softness_ratio": softness_ratio,
        "hardness_ratio": hardness_ratio,
        "roundness_score": roundness_score,
        "liquid_density": liquid_density,
        "nasal_ending": nasal_ending,
        "stop_ending": stop_ending,
    }


def name_to_cv_pattern(name: str) -> str:
    """Convert a name to a consonant/vowel pattern, e.g. Emma -> VCCV."""
    return "".join("V" if c in VOWELS else "C" for c in name.lower())


def extract_cluster_features(name: str) -> dict[str, float]:
    """Extract consonant cluster complexity features.

    Consonant clusters affect pronounceability and aesthetic perception.
    All features return float values normalized to [0, 1] range.

    """
    if not name:
        return {
            "consonant_clusters": 0.0,
            "max_cluster_len": 0.0,
            "has_complex_cluster": 0.0,
            "initial_cluster": 0.0,
            "final_cluster": 0.0,
            "medial_cluster_density": 0.0,
            "cv_alternation_ratio": 0.0,
            "starts_consonant": 0.0,
            "ends_vowel": 0.0,
            "has_cvc_pattern": 0.0,
            "has_vcv_pattern": 0.0,
        }

    name_lower = name.lower()
    name_len = len(name)

    clusters = CLUSTER_PATTERN.findall(name_lower)
    cluster_count = len(clusters)

    features: dict[str, float] = {
        "consonant_clusters": min(cluster_count / max(name_len, 1), 1.0),
    }

    max_cluster = max((len(c) for c in clusters), default=0)
    features["max_cluster_len"] = min(max_cluster / 4.0, 1.0)

    features["has_complex_cluster"] = 1.0 if any(len(c) > 2 for c in clusters) else 0.0

    features["initial_cluster"] = (
        1.0 if (name_len >= 2 and name_lower[0] not in VOWELS and name_lower[1] not in VOWELS) else 0.0
    )

    features["final_cluster"] = (
        1.0 if (name_len >= 2 and name_lower[-1] not in VOWELS and name_lower[-2] not in VOWELS) else 0.0
    )

    # Internal clusters exclude initial and final if they exist
    internal_clusters = cluster_count
    if features["initial_cluster"] > 0.5:
        internal_clusters -= 1
    if features["final_cluster"] > 0.5:
        internal_clusters -= 1
    features["medial_cluster_density"] = min(max(internal_clusters, 0) / max(name_len, 1), 1.0)

    cv_pattern = name_to_cv_pattern(name)

    transitions = sum(1 for i in range(len(cv_pattern) - 1) if cv_pattern[i] != cv_pattern[i + 1])
    features["cv_alternation_ratio"] = transitions / max(name_len - 1, 1) if name_len > 1 else 0.0

    features["starts_consonant"] = 1.0 if name_lower[0] not in VOWELS else 0.0

    features["ends_vowel"] = 1.0 if name_lower[-1] in VOWELS else 0.0

    features["has_cvc_pattern"] = 1.0 if "CVC" in cv_pattern else 0.0

    features["has_vcv_pattern"] = 1.0 if "VCV" in cv_pattern else 0.0

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
    """Extract enabled feature groups and return values plus a stable feature order."""
    features = {}

    if include_phonetic:
        features.update(extract_phonetic_features(name))

    if include_linguistic:
        features.update(extract_linguistic_features(name))
        features.update(extract_sonority_features(name))
        features.update(extract_suffix_features(name))
        features.update(extract_cluster_features(name))

    if include_metadata:
        features.update(extract_gender_features(gender))
        features.update(extract_origin_features(origin_region))

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
    """Cached feature extractor with database-backed persistence."""

    def __init__(
        self,
        feature_set_version: str = "v1",
        feature_cache: FeatureCache | None = None,
    ) -> None:
        """Initialize the extractor with an optional persistent cache."""
        self._feature_names: list[str] | None = None
        self._feature_set_version = feature_set_version
        self._feature_cache = feature_cache
        self._local_cache: dict[tuple, np.ndarray] = {}

    def _get_cache(self) -> FeatureCache:
        """Lazy initialize feature cache."""
        if self._feature_cache is None:
            self._feature_cache = FeatureCache(
                self._feature_set_version,
                self.get_feature_names(),
            )
        return self._feature_cache

    def get_feature_names(self) -> list[str]:
        """Get the ordered feature names, computed once from the active feature schema."""
        if self._feature_names is None:
            _, feature_names = extract_all_features("Test")
            self._feature_names = feature_names
        return self._feature_names

    def extract(
        self,
        name: str,
        gender: str | None = None,
        origin_region: str | None = None,
        name_id: int | None = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Extract one feature vector, using local and persistent caches when available."""
        cache_key = (name, gender, origin_region)

        if cache_key in self._local_cache:
            return self._local_cache[cache_key]

        if use_cache and name_id is not None:
            try:
                cached_features = self._get_cache().get_features(name_id)
                if cached_features is not None:
                    feature_names = self.get_feature_names()
                    vector = features_to_vector(cached_features, feature_names)
                    self._local_cache[cache_key] = vector
                    return vector
            except ValueError:
                # Feature set doesn't exist yet, will compute and cache
                pass

        features, _ = extract_all_features(name, gender, origin_region)
        vector = features_to_vector(features, self.get_feature_names())

        self._local_cache[cache_key] = vector

        if use_cache and name_id is not None:
            try:
                self._get_cache().set_features(name_id, features_dict=features)
            except Exception as e:
                logger.warning("Failed to cache features for %s: %s", name, e)

        return vector

    def batch_extract(
        self,
        names: list[str],
        genders: Sequence[str | None] | None = None,
        origin_regions: Sequence[str | None] | None = None,
        name_ids: Sequence[int | None] | None = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Extract feature vectors as an array of shape (n_names, n_features)."""
        genders = _metadata_sequence(names, genders, "genders")
        origin_regions = _metadata_sequence(names, origin_regions, "origin_regions")
        name_ids = _metadata_sequence(names, name_ids, "name_ids")

        vectors = []
        for name, gender, origin, name_id in zip(names, genders, origin_regions, name_ids, strict=True):
            vectors.append(self.extract(name, gender, origin, name_id, use_cache))

        return np.stack(vectors, axis=0)

    @property
    def feature_set_version(self) -> str:
        """Get the feature set version."""
        return self._feature_set_version

    @property
    def feature_cache(self) -> FeatureCache | None:
        """Get the feature cache instance."""
        return self._feature_cache
