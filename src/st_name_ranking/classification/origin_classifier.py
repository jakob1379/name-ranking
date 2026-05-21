#!/usr/bin/env python3
"""Origin classification for names using a chain of classifiers.

This module provides a hierarchical classification strategy:
1. Rule‑based Nordic detection (fast, high‑confidence for Danish names)
2. Phonetic similarity matching using Double Metaphone
3. ethnicolr Wikipedia model (if installed)
4. ethnidata API (if installed)

The goal is to maximize classification coverage while maintaining reasonable
accuracy, prioritizing speed for batch processing of ~50k names.
"""

import logging
import re
import unicodedata
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from metaphone import doublemetaphone

from st_name_ranking.persistence.database import get_connection


class OriginResult(NamedTuple):
    """Name origin classification result."""

    region: str
    confidence: float


logger = logging.getLogger(__name__)
ReferenceNameRecord = tuple[str, float, str, str]
ReferenceNames = Mapping[str, ReferenceNameRecord]
FrozenReferenceNames = tuple[tuple[str, ReferenceNameRecord], ...]

# Classification thresholds
MIN_ETHNIDATA_CONFIDENCE = 0.3
HIGH_CONFIDENCE_THRESHOLD = 0.6
MEDIUM_CONFIDENCE_THRESHOLD = 0.5
LOW_CONFIDENCE_THRESHOLD = 0.4

# Phonetic similarity threshold
MIN_PHONETIC_SIMILARITY = 0.6


def _get_region_for_nationality(nationality: str) -> tuple[str, float]:
    """Map nationality to region using database mapping.
    Returns (region, confidence_adjustment).
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT region FROM region_mapping WHERE nationality = ?",
            (nationality,),
        )
        row = cursor.fetchone()
        if row:
            return row[0], 1.0  # Full confidence for exact match

        # Try partial matching (e.g., "American" matches "United States")
        cursor = conn.execute(
            "SELECT region FROM region_mapping "
            "WHERE ? LIKE '%' || nationality || '%' "
            "OR nationality LIKE '%' || ? || '%'",
            (nationality, nationality),
        )
        row = cursor.fetchone()
        if row:
            return row[0], 0.8  # Reduced confidence for partial match

        # Default to International
        return "International", 0.5


# Type alias for classifier that returns (region, confidence) or None
ClassifierFunc = Callable[[str], OriginResult | None]


def _create_ethnidata_classifier() -> ClassifierFunc | None:
    """Create an ethnidata classifier callable that returns (region, confidence).
    Returns the classifier instance, or None if ethnidata is unavailable.
    """
    try:
        from ethnidata import EthniData  # noqa: PLC0415

        ethnidata = EthniData()
    except ImportError:
        logger.debug("ethnidata not installed")
        return None
    except (OSError, FileNotFoundError) as e:
        logger.warning("ethnidata data files missing or broken: %s", e)
        return None

    def classify_with_ethnidata(name: str) -> tuple[str, float] | None:
        """Classify a single name using ethnidata."""
        try:
            # Get nationality prediction
            prediction = ethnidata.predict_nationality(name)
            if not prediction:
                return None
            # Get country name and confidence
            country_name = prediction.get("country_name")
            if not country_name:
                return None
            confidence = prediction.get("confidence", 0.7)
            # Map country name to region using our mapping
            region, confidence_adjust = _get_region_for_nationality(
                country_name,
            )
            # Adjust confidence
            confidence = confidence * confidence_adjust
        except (ImportError, AttributeError, ValueError) as e:
            logger.debug("ethnidata classification failed for '%s': %s", name, e)
            return None
        else:
            return region, confidence

    return classify_with_ethnidata


# Nordic character patterns
NORDIC_CHARS = {"æ", "ø", "å", "Æ", "Ø", "Å"}

# Common Danish/Nordic name suffixes (with approximate confidence weights)
NORDIC_SUFFIXES = [
    ("sen", 0.9),  # Very common Danish patronymic
    ("datter", 0.95),  # Danish patronymic (female)
    ("gaard", 0.8),  # Farm/estate
    ("berg", 0.7),  # Mountain/hill
    ("holm", 0.8),  # Island
    ("lund", 0.8),  # Grove
    ("by", 0.6),  # Town/village
    ("rup", 0.9),  # Danish place name ending
    ("toft", 0.8),  # Homestead
    ("skov", 0.8),  # Forest
    ("høj", 0.9),  # Hill (contains ø)
    ("sted", 0.7),  # Place
    ("borg", 0.7),  # Castle/fort
    ("havn", 0.8),  # Harbor
    ("fjord", 0.8),  # Fjord
    ("dal", 0.7),  # Valley
    ("vang", 0.8),  # Field/meadow
    ("sø", 0.9),  # Lake (contains ø)
    ("ås", 0.9),  # Ridge (contains å)
    ("ström", 0.6),  # Stream (Swedish/Norwegian)
    ("quist", 0.6),  # Twig (Swedish)
    ("blad", 0.6),  # Leaf (Scandinavian)
]

# Strongly Nordic given‑name endings (conservative list to avoid false positives)
NORDIC_GIVEN_NAME_ENDINGS = [
    "bjørn",
    "fred",
    "vig",
    "mar",
    "tor",
    "olf",
    "ulf",
    "stein",
    "hard",
]

# Phonetic codes strongly associated with Nordic names
# These are Double Metaphone primary codes for known Nordic names
NORDIC_PHONETIC_CODES = {
    # Strongly Nordic consonant clusters (Danish, Norwegian, Swedish)
    "HJ",  # Hjalmar, Hjorth
    "KJ",  # Kjeld, Kjartan
    "TJ",  # Tjalfe, Tjørn
    "SV",  # Sven, Svend
    "ST",  # Sten, Stig (common but not exclusive)
    "BJ",  # Bjørn, Bjarne
    "NJ",  # Njord, Njal
    "RJ",  # Rjukan (Norwegian place)
    "MJ",  # Mjölnir (Norse mythology)
    # Note: We exclude generic codes like E, A, SK, BR, KR, FR, TR
    # Also exclude ambiguous codes like LJ (Slavic), VJ (Slavic)
}


def normalize_name(name: str) -> str:
    """Normalize name for classification:
    - Convert to NFKD Unicode normalization
    - Lowercase
    - Remove extra whitespace
    """
    name = unicodedata.normalize("NFKD", name).strip().lower()
    return re.sub(r"\s+", " ", name)


def rule_based_nordic_detection(name: str) -> tuple[str | None, float]:
    """Rule‑based detection of Nordic (Danish/Norwegian/Swedish) names.
    Returns (region, confidence) or (None, 0.0) if not detected.

    Confidence levels:
    - 0.95: Contains Nordic characters (æ, ø, å)
    - 0.85–0.90: Common Nordic suffix with high weight
    - 0.70–0.80: Moderate confidence patterns
    - 0.60: Weak evidence
    """
    normalized = normalize_name(name)

    # Check for Nordic characters (strongest signal)
    if any(char in NORDIC_CHARS for char in name):
        logger.debug("Name '%s' contains Nordic characters → Nordic", name)
        return "Nordic", 0.95

    # Check suffixes
    for suffix, weight in NORDIC_SUFFIXES:
        if normalized.endswith(suffix):
            confidence = weight * 0.9  # Slightly discount suffix-only evidence
            logger.debug(
                "Name '%s' ends with '%s' → Nordic (confidence: %.2f)",
                name,
                suffix,
                confidence,
            )
            return "Nordic", confidence

    # Check for common Danish given name endings (for first names)
    for ending in NORDIC_GIVEN_NAME_ENDINGS:
        if normalized.endswith(ending) and len(normalized) >= len(ending) + 2:
            # Only if the ending is not too short relative to name length
            return "Nordic", 0.65

    # Phonetic pattern check using Double Metaphone
    primary, secondary = doublemetaphone(name)
    if primary in NORDIC_PHONETIC_CODES or secondary in NORDIC_PHONETIC_CODES:
        logger.debug(
            "Name '%s' has Nordic phonetic code %s/%s → Nordic",
            name,
            primary,
            secondary,
        )
        return "Nordic", 0.75

    # No Nordic patterns detected
    return None, 0.0


def phonetic_similarity_classification(
    name: str,
    reference_names: ReferenceNames,
) -> tuple[str | None, float]:
    """Classify name by phonetic similarity to known reference names.

    Args:
        name: Name to classify
        reference_names: Dict mapping known names to (region, confidence, phonetic_primary, phonetic_secondary)

    Returns:
        (region, confidence) based on best phonetic match, or (None, 0.0)

    """
    primary, secondary = doublemetaphone(name)

    best_region = None
    best_confidence = 0.0
    best_score = 0.0

    for ref_region, ref_conf, ref_primary, ref_secondary in reference_names.values():
        # Compute phonetic similarity score using precomputed codes
        score = 0.0
        if primary == ref_primary:
            score = 1.0
        elif primary == ref_secondary or secondary == ref_primary:
            score = 0.8
        elif secondary == ref_secondary:
            score = 0.7
        elif primary and ref_primary and primary[0] == ref_primary[0]:
            # Same first character of primary code
            score = 0.5

        if score > best_score:
            best_score = score
            # Combine match score with reference confidence
            best_confidence = score * ref_conf * 0.9  # Discount for phonetic similarity
            best_region = ref_region

    if best_region and best_confidence > MIN_PHONETIC_SIMILARITY:
        logger.debug(
            "Name '%s' phonetically matches %s (score: %.2f, confidence: %.2f)",
            name,
            best_region,
            best_score,
            best_confidence,
        )
        return best_region, best_confidence

    return None, 0.0


ETHNICOLR_CATEGORY_COLS = [
    "Asian,GreaterEastAsian,EastAsian",
    "Asian,GreaterEastAsian,Japanese",
    "Asian,IndianSubContinent",
    "GreaterAfrican,Africans",
    "GreaterAfrican,Muslim",
    "GreaterEuropean,British",
    "GreaterEuropean,EastEuropean",
    "GreaterEuropean,Jewish",
    "GreaterEuropean,WestEuropean,French",
    "GreaterEuropean,WestEuropean,Germanic",
    "GreaterEuropean,WestEuropean,Hispanic",
    "GreaterEuropean,WestEuropean,Italian",
    "GreaterEuropean,WestEuropean,Nordic",
]

ETHNICOLR_REGION_MAPPING = {
    "GreaterEuropean,WestEuropean,Nordic": ("Nordic", 0.9),
    "GreaterEuropean,WestEuropean,Germanic": ("European", 0.8),
    "GreaterEuropean,WestEuropean,French": ("European", 0.8),
    "GreaterEuropean,WestEuropean,Italian": ("European", 0.8),
    "GreaterEuropean,WestEuropean,Hispanic": ("European", 0.7),
    "GreaterEuropean,British": ("European", 0.8),
    "GreaterEuropean,EastEuropean": ("European", 0.8),
    "GreaterEuropean,Jewish": ("International", 0.5),
    "Asian,GreaterEastAsian,EastAsian": ("Asian", 0.9),
    "Asian,GreaterEastAsian,Japanese": ("Asian", 0.9),
    "Asian,IndianSubContinent": ("Asian", 0.9),
    "GreaterAfrican,Africans": ("African", 0.9),
    "GreaterAfrican,Muslim": ("Middle Eastern", 0.8),
}


def _map_ethnicolr_to_region(row: dict[str, Any]) -> tuple[str | None, float]:
    """Map ethnicolr's 13 categories to our 8 regions."""
    best_cat = None
    best_prob = 0.0

    for cat in ETHNICOLR_CATEGORY_COLS:
        prob = row.get(f"{cat}_mean", 0.0)
        if prob > best_prob:
            best_prob = prob
            best_cat = cat

    if not best_cat:
        return None, 0.0

    region, multiplier = ETHNICOLR_REGION_MAPPING.get(best_cat, (None, 0.5))
    confidence = best_prob * multiplier
    return region, confidence


class EthnicolrClassifier:
    """Thin wrapper around ethnicolr predictions."""

    def __init__(self, predictor: Callable[..., Any]) -> None:
        self._predictor = predictor

    def classify_batch(
        self,
        names: list[str],
    ) -> list[tuple[str | None, float]]:
        """Classify batch of names using ethnicolr's Wikipedia model."""
        if not names:
            return []

        import pandas as pd  # noqa: PLC0415

        df = pd.DataFrame({"last": names})

        try:
            result_df = self._predictor(df, "last", conf_int=1.0)

            classifications = []
            for _, row in result_df.iterrows():
                region, confidence = _map_ethnicolr_to_region(row)
                classifications.append((region, confidence))
        except (ImportError, AttributeError, ValueError, RuntimeError) as e:
            logger.warning("ethnicolr classification failed: %s", e)
            return [(None, 0.0)] * len(names)
        else:
            return classifications


def get_ethnicolr_classifier() -> EthnicolrClassifier | None:
    """Lazy loader for ethnicolr classifier."""
    try:
        from ethnicolr import pred_wiki_ln  # noqa: PLC0415
    except ImportError:
        logger.debug("ethnicolr not installed")
        return None
    else:
        return EthnicolrClassifier(pred_wiki_ln)


class OriginClassifier:
    """Hierarchical origin classifier with chain‑of‑responsibility pattern.

    Classifiers are tried in order until one returns a classification with
    confidence above its threshold. The chain is:

    1. Rule‑based Nordic detection (threshold: 0.6)
    2. Phonetic similarity (threshold: 0.5, requires reference names)
    3. ethnicolr Wikipedia model (threshold: 0.4, if installed)
    4. ethnidata API (threshold: 0.3, if installed)

    If no classifier produces a result above threshold, returns "International"
    with low confidence (0.1).
    """

    def __init__(
        self,
        reference_names: ReferenceNames | None = None,
        ethnidata_classifier: ClassifierFunc | None = None,
        *,
        use_ethnidata: bool = True,
    ) -> None:
        """Args:
        reference_names: Dict of known name -> (region, confidence, phonetic_primary, phonetic_secondary) for
                        phonetic similarity classification.
        ethnidata_classifier: Pre‑initialized ethnidata classifier instance.
                              If None, will be lazy‑loaded when needed and enabled.
        use_ethnidata: Whether to lazy-load ethnidata fallback when no classifier is provided.

        """
        self.reference_names = dict(_freeze_reference_names(reference_names))
        self.ethnicolr = get_ethnicolr_classifier()
        self.ethnidata = ethnidata_classifier
        self._use_ethnidata = use_ethnidata or ethnidata_classifier is not None

    def _classify_with_phonetic(self, name: str) -> tuple[str | None, float]:
        """Classify using phonetic similarity if reference data exists."""
        if not self.reference_names:
            return None, 0.0

        return phonetic_similarity_classification(name, self.reference_names)

    def _classify_with_ethnicolr(self, name: str) -> tuple[str | None, float]:
        """Classify using ethnicolr model if available."""
        if not self.ethnicolr:
            return None, 0.0

        try:
            results = self.ethnicolr.classify_batch([name])
            if results:
                return results[0]
        except (ImportError, AttributeError, ValueError, IndexError) as e:
            logger.debug(
                "ethnicolr classification failed for '%s': %s",
                name,
                e,
            )

        return None, 0.0

    def _classify_with_ethnidata(self, name: str) -> tuple[str | None, float]:
        """Classify using ethnidata model if available."""
        if not self._use_ethnidata:
            return None, 0.0

        if self.ethnidata is None:
            self.ethnidata = _create_ethnidata_classifier()
            if self.ethnidata is None:
                self._use_ethnidata = False
                return None, 0.0

        try:
            result = self.ethnidata(name)
            if result and isinstance(result, tuple):
                return result
        except (ImportError, AttributeError, ValueError) as e:
            logger.debug(
                "ethnidata classification failed for '%s': %s",
                name,
                e,
            )

        return None, 0.0

    def classify(
        self,
        name: str,
    ) -> tuple[str, float]:
        """Classify a single name.

        Returns:
            Tuple of (region, confidence).
            Confidence is in range [0.1, 1.0].

        """
        # 1. Rule‑based Nordic detection
        region, confidence = rule_based_nordic_detection(name)
        if region and confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return region, confidence

        # 2. Phonetic similarity (if we have reference names)
        region, confidence = self._classify_with_phonetic(name)
        if region and confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
            return region, confidence

        # 3. ethnicolr (if installed)
        region, confidence = self._classify_with_ethnicolr(name)
        if region and confidence >= LOW_CONFIDENCE_THRESHOLD:
            return region, confidence

        # 4. ethnidata (if installed)
        region, confidence = self._classify_with_ethnidata(name)
        if region and confidence >= MIN_ETHNIDATA_CONFIDENCE:
            return region, confidence

        # 5. Fallback: International with low confidence
        return "International", 0.1

    def classify_batch(
        self,
        names: list[str],
        _genders: list[str | None] | None = None,
    ) -> list[tuple[str, float]]:
        """Classify a batch of names efficiently.

        Args:
            names: List of names to classify
            _genders: Optional list of genders (same length as names)

        Returns:
            List of (region, confidence) tuples

        """
        results = []

        # Process in chunks to balance memory and speed
        chunk_size = 100
        for i in range(0, len(names), chunk_size):
            chunk_names = names[i : i + chunk_size]

            chunk_results = []
            for name in chunk_names:
                region, confidence = self.classify(name)
                chunk_results.append((region, confidence))

            results.extend(chunk_results)

            logger.debug("Processed %d/%d names", i + len(chunk_names), len(names))

        return results


ReferenceCacheKey = tuple[str, FrozenReferenceNames]

_CLASSIFIER_CACHE: dict[ReferenceCacheKey, OriginClassifier] = {}


def _freeze_reference_names(reference_names: ReferenceNames | None) -> FrozenReferenceNames:
    """Return stable immutable reference-name content for cache keys and classifier state."""
    if not reference_names:
        return ()
    return tuple(sorted(reference_names.items()))


def _reference_cache_key(
    reference_names: ReferenceNames | None,
) -> ReferenceCacheKey:
    """Return the classifier-cache key for a reference-name set."""
    return ("reference_names", _freeze_reference_names(reference_names))


def reset_classifier_cache() -> None:
    """Clear cached classifier instances."""
    _CLASSIFIER_CACHE.clear()


def get_or_create_classifier(
    reference_names: ReferenceNames | None = None,
) -> OriginClassifier:
    """Return the cached classifier for a reference-name set, creating it when needed."""
    cache_key = _reference_cache_key(reference_names)
    if cache_key not in _CLASSIFIER_CACHE:
        _CLASSIFIER_CACHE[cache_key] = OriginClassifier(reference_names)
    return _CLASSIFIER_CACHE[cache_key]


def _run_demo() -> None:
    """Print sample origin classifications for manual inspection."""
    classifier = OriginClassifier()

    test_names = [
        "Jørgen",
        "Andersen",
        "Hansen",
        "Bjørk",
        "Sørensen",
        "Muhammed",
        "Zhang",
        "Wei",
        "Smith",
        "García",
    ]

    print("Testing origin classification:")
    for name in test_names:
        region, confidence = classifier.classify(name)
        print(f"  {name:15} → {region:15} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    _run_demo()
