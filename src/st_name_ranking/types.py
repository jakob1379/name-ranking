"""Type definitions for st_name_ranking using NamedTuples.

This module provides structured type definitions to replace raw dicts and tuples
throughout the codebase, improving type safety and code clarity.
"""

from typing import NamedTuple


class UnclassifiedName(NamedTuple):
    """Unclassified name needing origin classification."""

    id: int
    name: str


class DatabaseStats(NamedTuple):
    """Database statistics summary."""

    total_names: int
    classified_names: int
    unclassified_names: int
    rated_names: int
    origin_distribution: dict[str, int]


class PreferenceStats(NamedTuple):
    """Preference statistics for a group."""

    wins: int
    losses: int
    draws: int
    total: int


class NameDetails(NamedTuple):
    """Name details for batch lookups."""

    gender: str | None
    origin_region: str | None


class PhoneticCodes(NamedTuple):
    """Primary and secondary Double Metaphone codes."""

    primary: str
    secondary: str


class SimilarityScore(NamedTuple):
    """Similarity score result."""

    name: str
    score: float


class NamePair(NamedTuple):
    """A pair of names for comparison."""

    idx_a: int
    idx_b: int
    name_a: str
    name_b: str


class SourceVersion(NamedTuple):
    """Synced source-data submodule version."""

    commit_hash: str
