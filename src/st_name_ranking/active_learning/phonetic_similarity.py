"""Phonetic similarity functions using Double Metaphone algorithm."""

import logging

from metaphone import doublemetaphone

logger = logging.getLogger(__name__)


def compute_phonetic_codes(name: str) -> tuple[str, str]:
    """Compute Double Metaphone codes for a name.
    Returns (primary_code, secondary_code).
    Secondary code is an empty string when not applicable.
    """
    primary, secondary = doublemetaphone(name)
    return primary or "", secondary or ""


def phonetic_similarity(name1: str, name2: str) -> float:
    """Compute a 0.0, 0.5, or 1.0 phonetic similarity score."""
    primary1, secondary1 = compute_phonetic_codes(name1)
    primary2, secondary2 = compute_phonetic_codes(name2)

    if primary1 == primary2:
        return 1.0
    if secondary1 and secondary1 == primary2:
        return 0.5
    if secondary2 and secondary2 == primary1:
        return 0.5
    if secondary1 and secondary2 and secondary1 == secondary2:
        return 0.5
    return 0.0


def batch_compute_phonetic_codes(
    names: list[str],
) -> dict[str, tuple[str, str]]:
    """Compute phonetic codes for multiple names efficiently.
    Returns dict mapping name -> (primary, secondary).
    """
    logger.debug("Computing phonetic codes for %d names", len(names))
    result = {}
    for name in names:
        result[name] = compute_phonetic_codes(name)
    return result


def phonetic_similarity_batch(
    target_codes: tuple[str, str],
    name_codes_dict: dict[str, tuple[str, str]],
) -> dict[str, float]:
    """Compute phonetic similarity between target codes and many names."""
    target_primary, target_secondary = target_codes
    similarities = {}

    for name, (primary, secondary) in name_codes_dict.items():
        if target_primary == primary:
            similarities[name] = 1.0
        elif (
            (target_secondary and target_secondary == primary)
            or (secondary and target_primary == secondary)
            or (target_secondary and secondary and target_secondary == secondary)
        ):
            similarities[name] = 0.5
        else:
            similarities[name] = 0.0

    return similarities


def get_phonetic_neighbors(
    target: str,
    names: list[str],
    threshold: float = 0.5,
    limit: int = 10,
) -> list[tuple[str, float]]:
    """Find names phonetically similar to target.
    Returns list of (name, similarity_score) sorted by similarity descending.
    """
    target_codes = compute_phonetic_codes(target)
    name_codes = batch_compute_phonetic_codes(names)
    similarities = phonetic_similarity_batch(target_codes, name_codes)

    filtered = [(name, score) for name, score in similarities.items() if score >= threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)

    return filtered[:limit]


# For integration with existing similarity module
