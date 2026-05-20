"""Database-backed feature cache for extracted name features."""

from __future__ import annotations

from typing import TYPE_CHECKING

from st_name_ranking.persistence import database

if TYPE_CHECKING:
    from st_name_ranking.types import FeatureValues


class FeatureCache:
    """Cache pre-computed name features with compute-once semantics."""

    def __init__(self, feature_set_version: str, feature_names: list[str] | None = None) -> None:
        """Initialize a cache for a specific feature schema version."""
        self._version = feature_set_version
        self._feature_set_id: int | None = None
        self._feature_names = feature_names
        self._local_cache: dict[int, FeatureValues] = {}

    def _get_feature_set_id(self) -> int:
        """Lazy-load or create the backing feature-set row."""
        if self._feature_set_id is None:
            if self._feature_names is None:
                feature_set = database.get_feature_set_by_version(self._version)
                if feature_set is None:
                    msg = f"Feature set version '{self._version}' not found"
                    raise ValueError(msg)
                self._feature_set_id = feature_set["id"]
                self._feature_names = feature_set["feature_names"]
            else:
                self._feature_set_id = database.get_or_create_feature_set(self._version, self._feature_names)
        return self._feature_set_id

    def get_features(self, name_id: int, feature_set_version: str | None = None) -> FeatureValues | None:
        """Get cached features or return None if not computed."""
        if name_id in self._local_cache:
            return self._local_cache[name_id]

        version = feature_set_version or self._version
        if version != self._version:
            feature_set = database.get_feature_set_by_version(version)
            if feature_set is None:
                return None
            features = database.get_cached_features(name_id, feature_set["id"])
        else:
            features = database.get_cached_features(name_id, self._get_feature_set_id())

        if features is not None:
            self._local_cache[name_id] = features

        return features

    def set_features(
        self,
        name_id: int,
        *,
        features_dict: FeatureValues,
        feature_set_version: str | None = None,
    ) -> None:
        """Cache computed features for one name."""
        version = feature_set_version or self._version

        if version != self._version:
            feature_set = database.get_feature_set_by_version(version)
            if feature_set is None:
                msg = f"Feature set version '{version}' not found"
                raise ValueError(msg)
            feature_set_id = feature_set["id"]
        else:
            feature_set_id = self._get_feature_set_id()

        self._local_cache[name_id] = features_dict
        database.set_cached_features(name_id, feature_set_id, features_dict)

    def set_features_batch(self, features_data: list[tuple[int, FeatureValues]]) -> int:
        """Cache computed features for multiple names."""
        if not features_data:
            return 0

        for name_id, features in features_data:
            self._local_cache[name_id] = features

        feature_set_id = self._get_feature_set_id()
        db_data = [(name_id, feature_set_id, features) for name_id, features in features_data]
        return database.set_cached_features_batch(db_data)

    @property
    def feature_names(self) -> list[str]:
        """Get the feature names for this feature set."""
        if self._feature_names is None:
            _ = self._get_feature_set_id()
        return self._feature_names or []

    @property
    def version(self) -> str:
        """Get the feature-set version."""
        return self._version
