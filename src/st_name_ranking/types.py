"""Type definitions for st_name_ranking using NamedTuples.

This module provides structured type definitions to replace raw dicts and tuples
throughout the codebase, improving type safety and code clarity.
"""

from typing import NamedTuple


class NameRecord(NamedTuple):
    """Represents a name record from the database.

    Attributes:
        id: Database ID of the name
        name: The actual name string
        gender: Gender classification (Male, Female, Unisex, or None)
        origin_region: Geographic origin region or None
        origin_confidence: Confidence score for origin classification (0-1)
    """

    id: int
    name: str
    gender: str | None
    origin_region: str | None = None
    origin_confidence: float | None = None


class UnclassifiedName(NamedTuple):
    """Represents an unclassified name needing origin classification.

    Attributes:
        id: Database ID of the name
        name: The actual name string
    """

    id: int
    name: str


class DatabaseStats(NamedTuple):
    """Database statistics summary.

    Attributes:
        total_names: Total number of names in database
        classified_names: Number of names with origin classification
        unclassified_names: Number of names without origin classification
        rated_names: Number of names with ratings
        origin_distribution: Dict mapping region -> count
    """

    total_names: int
    classified_names: int
    unclassified_names: int
    rated_names: int
    origin_distribution: dict[str, int]


class PreferenceStats(NamedTuple):
    """Preference statistics for a group (gender, origin, or phonetic).

    Attributes:
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws
        total: Total number of comparisons
    """

    wins: int
    losses: int
    draws: int
    total: int


class NameDetails(NamedTuple):
    """Name details tuple for batch lookups.

    Attributes:
        gender: Gender classification or None
        origin_region: Origin region or None
    """

    gender: str | None
    origin_region: str | None


class PhoneticCodes(NamedTuple):
    """Phonetic encoding for a name.

    Attributes:
        primary: Primary Double Metaphone code
        secondary: Secondary Double Metaphone code
    """

    primary: str
    secondary: str


class SimilarityScore(NamedTuple):
    """Similarity score result.

    Attributes:
        name: The candidate name
        score: Similarity score (0-1 for vector, 0-100 for string)
    """

    name: str
    score: float


class NamePair(NamedTuple):
    """A pair of names for comparison.

    Attributes:
        idx_a: Index of first name in the list
        idx_b: Index of second name in the list
        name_a: First name string
        name_b: Second name string
    """

    idx_a: int
    idx_b: int
    name_a: str
    name_b: str


class RatingEntry(NamedTuple):
    """Single rating entry.

    Attributes:
        name: The name being rated
        rating: The preference rating score
    """

    name: str
    rating: float


class NameGender(NamedTuple):
    """Name with gender from JSON data.

    Attributes:
        name: The name string
        gender: Gender classification
    """

    name: str
    gender: str


class GenderedNames(NamedTuple):
    """Names organized by gender category.

    Attributes:
        male: List of male names
        female: List of female names
        unisex: List of unisex names
        all: List of all names
    """

    male: list[str]
    female: list[str]
    unisex: list[str]
    all: list[str]


class SourceVersion(NamedTuple):
    """Submodule version tracking.

    Attributes:
        commit_hash: Git commit hash of synced submodule
    """

    commit_hash: str


class ComparisonRecord(NamedTuple):
    """A pairwise comparison record.

    Attributes:
        name_a: First name in comparison
        name_b: Second name in comparison
        preference: Preference value (-1=a wins, 0=draw, 1=b wins, 2=both disliked)
    """

    name_a: str
    name_b: str
    preference: int
