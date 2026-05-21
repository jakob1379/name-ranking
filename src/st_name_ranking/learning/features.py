"""Feature extraction for names.

Extracts phonetic, linguistic, and metadata features for preference learning.
All features are normalized to [0, 1] range for model compatibility.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

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
SUFFIX_FLAG_ENDINGS = {
    "ends_with_a": "a",
    "ends_with_o": "o",
    "ends_with_us": "us",
    "ends_with_ie": "ie",
    "ends_with_ette": "ette",
    "ends_with_elle": "elle",
    "ends_with_ck": "ck",
    "ends_with_sh": "sh",
}
CLUSTER_FEATURE_KEYS = (
    "consonant_clusters",
    "max_cluster_len",
    "has_complex_cluster",
    "initial_cluster",
    "final_cluster",
    "medial_cluster_density",
    "cv_alternation_ratio",
    "starts_consonant",
    "ends_vowel",
    "has_cvc_pattern",
    "has_vcv_pattern",
)


@dataclass(frozen=True)
class FeatureGroupOptions:
    """Feature groups included in one extraction pass."""

    include_phonetic: bool = True
    include_linguistic: bool = True
    include_metadata: bool = True


@dataclass(frozen=True)
class FeatureCacheOptions:
    """Persistent-cache controls for extracting one name."""

    name_id: int | None = None
    use_cache: bool = True


@dataclass(frozen=True)
class FeatureBatchContext:
    """Persistent-cache metadata for extracting a batch of names."""

    name_ids: Sequence[int | None] | None = None
    use_cache: bool = True


def extract_suffix_features(name: str) -> dict[str, float]:
    """Extract suffix and prefix-based gender cue features.

    Returns features based on cross-cultural research showing strong gender
    associations with certain name endings and beginnings.
    """
    if not name:
        return dict.fromkeys(SUFFIX_FEATURE_KEYS, 0.0)

    name_lower = name.lower()
    features = {key: float(name_lower.endswith(suffix)) for key, suffix in SUFFIX_FLAG_ENDINGS.items()}
    features["suffix_feminine_score"] = _best_suffix_score(name_lower, FEMININE_SUFFIXES)
    features["suffix_masculine_score"] = _best_suffix_score(name_lower, MASCULINE_SUFFIXES)
    features["prefix_feminine"] = float(any(name_lower.startswith(prefix) for prefix in FEMININE_PREFIXES))
    features["prefix_masculine"] = float(any(name_lower.startswith(prefix) for prefix in MASCULINE_PREFIXES))

    trailing_consonants = 0
    for char in reversed(name_lower):
        if char in DANISH_VOWELS:
            break
        trailing_consonants += 1

    features["ends_open_syllable"] = 1.0 if name_lower and name_lower[-1] in DANISH_VOWELS else 0.0
    features["coda_weight"] = min(trailing_consonants / 3.0, 1.0)
    features["closed_heavy"] = 1.0 if trailing_consonants >= 2 else 0.0

    return features


def _best_suffix_score(name_lower: str, scores: dict[str, float]) -> float:
    """Return the strongest matching suffix score."""
    matches = (score for suffix, score in scores.items() if name_lower.endswith(suffix))
    return max(matches, default=0.0)


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
    roundness_score = roundness_sum / length

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


def _empty_cluster_features() -> dict[str, float]:
    """Return the zero vector for consonant-cluster features."""
    return dict.fromkeys(CLUSTER_FEATURE_KEYS, 0.0)


def _edge_cluster_features(name_lower: str) -> tuple[float, float]:
    """Return initial and final cluster flags."""
    if len(name_lower) < 2:
        return 0.0, 0.0
    starts_with_cluster = name_lower[0] not in VOWELS and name_lower[1] not in VOWELS
    ends_with_cluster = name_lower[-1] not in VOWELS and name_lower[-2] not in VOWELS
    return float(starts_with_cluster), float(ends_with_cluster)


def _cv_transition_ratio(cv_pattern: str) -> float:
    """Return the ratio of consonant/vowel transitions in a CV pattern."""
    if len(cv_pattern) <= 1:
        return 0.0
    transitions = sum(left != right for left, right in pairwise(cv_pattern))
    return transitions / (len(cv_pattern) - 1)


def extract_cluster_features(name: str) -> dict[str, float]:
    """Extract consonant cluster complexity features.

    Consonant clusters affect pronounceability and aesthetic perception.
    All features return float values normalized to [0, 1] range.

    """
    if not name:
        return _empty_cluster_features()

    name_lower = name.lower()
    name_len = len(name)

    clusters = CLUSTER_PATTERN.findall(name_lower)
    cluster_count = len(clusters)
    max_cluster = max((len(c) for c in clusters), default=0)
    initial_cluster, final_cluster = _edge_cluster_features(name_lower)

    internal_clusters = cluster_count - int(initial_cluster) - int(final_cluster)
    medial_cluster_density = min(max(internal_clusters, 0) / name_len, 1.0)

    cv_pattern = name_to_cv_pattern(name)
    return {
        "consonant_clusters": min(cluster_count / name_len, 1.0),
        "max_cluster_len": min(max_cluster / 4.0, 1.0),
        "has_complex_cluster": float(any(len(cluster) > 2 for cluster in clusters)),
        "initial_cluster": initial_cluster,
        "final_cluster": final_cluster,
        "medial_cluster_density": medial_cluster_density,
        "cv_alternation_ratio": _cv_transition_ratio(cv_pattern),
        "starts_consonant": float(name_lower[0] not in VOWELS),
        "ends_vowel": float(name_lower[-1] in VOWELS),
        "has_cvc_pattern": float("CVC" in cv_pattern),
        "has_vcv_pattern": float("VCV" in cv_pattern),
    }


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
    options: FeatureGroupOptions | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Extract enabled feature groups and return values plus a stable feature order."""
    options = options or FeatureGroupOptions()
    features = {}

    if options.include_phonetic:
        features.update(extract_phonetic_features(name))

    if options.include_linguistic:
        features.update(extract_linguistic_features(name))
        features.update(extract_sonority_features(name))
        features.update(extract_suffix_features(name))
        features.update(extract_cluster_features(name))

    if options.include_metadata:
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

    def _load_cached_vector(self, name: str, name_id: int) -> np.ndarray | None:
        """Return a cached feature vector, or None when cache state is unavailable."""
        try:
            cached_features = self._get_cache().get_features(name_id)
        except ValueError as exc:
            logger.debug("Feature cache unavailable for %s: %s", name, exc)
            return None

        if cached_features is None:
            return None

        feature_names = self.get_feature_names()
        return features_to_vector(cached_features, feature_names)

    def _store_cached_features(self, name: str, name_id: int, features: dict[str, float]) -> None:
        """Persist computed features when the cache is usable."""
        try:
            self._get_cache().set_features(name_id, features_dict=features)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Disabling feature cache after write failure for %s: %s", name, exc)
            self._feature_cache = None

    def extract(
        self,
        name: str,
        gender: str | None = None,
        origin_region: str | None = None,
        cache_options: FeatureCacheOptions | None = None,
    ) -> np.ndarray:
        """Extract one feature vector, using local and persistent caches when available."""
        cache_options = cache_options or FeatureCacheOptions()
        cache_key = (name, gender, origin_region)

        if cache_key in self._local_cache:
            return self._local_cache[cache_key]

        if cache_options.use_cache and cache_options.name_id is not None:
            cached_vector = self._load_cached_vector(name, cache_options.name_id)
            if cached_vector is not None:
                self._local_cache[cache_key] = cached_vector
                return cached_vector

        features, _ = extract_all_features(name, gender, origin_region)
        vector = features_to_vector(features, self.get_feature_names())

        self._local_cache[cache_key] = vector

        if cache_options.use_cache and cache_options.name_id is not None:
            self._store_cached_features(name, cache_options.name_id, features)

        return vector

    def batch_extract(
        self,
        names: list[str],
        genders: Sequence[str | None] | None = None,
        origin_regions: Sequence[str | None] | None = None,
        *,
        context: FeatureBatchContext | None = None,
    ) -> np.ndarray:
        """Extract feature vectors as an array of shape (n_names, n_features)."""
        context = context or FeatureBatchContext()
        genders = _metadata_sequence(names, genders, "genders")
        origin_regions = _metadata_sequence(names, origin_regions, "origin_regions")
        name_ids = _metadata_sequence(names, context.name_ids, "name_ids")

        vectors = []
        for name, gender, origin, name_id in zip(names, genders, origin_regions, name_ids, strict=True):
            cache_options = FeatureCacheOptions(
                name_id=name_id,
                use_cache=context.use_cache,
            )
            vectors.append(self.extract(name, gender, origin, cache_options))

        return np.stack(vectors, axis=0)

    @property
    def feature_set_version(self) -> str:
        """Get the feature set version."""
        return self._feature_set_version

    @property
    def feature_cache(self) -> FeatureCache | None:
        """Get the feature cache instance."""
        return self._feature_cache
