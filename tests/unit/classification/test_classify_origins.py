"""Tests for st_name_ranking.classification.classify_origins module."""

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from st_name_ranking.classification import classify_origins, origin_classifier
from st_name_ranking.persistence import database


class TestOriginClassifierFactory:
    """Tests for the canonical origin_classifier factory used by classify_origins."""

    def setup_method(self):
        """Clear classifier cache before each test."""
        origin_classifier.reset_classifier_cache()

    def test_get_or_create_classifier_caches_hierarchical_classifier(self):
        """Test successful hierarchical classifier caching."""
        reference_names: dict[str, tuple[str, float, str, str]] = {}
        with patch("st_name_ranking.classification.origin_classifier.OriginClassifier") as mock_cls:
            classifier = origin_classifier.get_or_create_classifier(reference_names=reference_names)
            assert classifier == mock_cls.return_value

            # Second call should return cached classifier
            classifier2 = origin_classifier.get_or_create_classifier(reference_names=reference_names)
            assert classifier2 == classifier

            mock_cls.assert_called_once_with(reference_names)

    def test_get_or_create_classifier_uses_distinct_reference_set_cache_keys(self):
        """Different reference-name dictionaries should not share stale classifier state."""
        first_reference_set = {"Anna": ("Nordic", 0.9, "AN", "")}
        second_reference_set = {"Maria": ("European", 0.8, "MR", "")}

        with patch("st_name_ranking.classification.origin_classifier.OriginClassifier") as mock_cls:
            mock_cls.side_effect = [object(), object()]
            first_classifier = origin_classifier.get_or_create_classifier(first_reference_set)
            second_classifier = origin_classifier.get_or_create_classifier(second_reference_set)

        assert first_classifier is not second_classifier
        assert mock_cls.call_count == 2
        mock_cls.assert_any_call(first_reference_set)
        mock_cls.assert_any_call(second_reference_set)

    def test_get_or_create_classifier_reuses_cache_for_equivalent_reference_data(self):
        """Equivalent reference data should share a classifier regardless of dict identity."""
        first_reference_set = {"Anna": ("Nordic", 0.9, "AN", "")}
        second_reference_set = {"Anna": ("Nordic", 0.9, "AN", "")}

        with patch("st_name_ranking.classification.origin_classifier.OriginClassifier") as mock_cls:
            first_classifier = origin_classifier.get_or_create_classifier(first_reference_set)
            second_classifier = origin_classifier.get_or_create_classifier(second_reference_set)

        assert first_classifier is second_classifier
        mock_cls.assert_called_once_with(first_reference_set)

    def test_get_or_create_classifier_cache_key_tracks_in_place_reference_mutation(self):
        """Mutating reference content should not reuse a stale classifier for the same dict."""
        reference_set = {"Anna": ("Nordic", 0.9, "AN", "")}

        with patch("st_name_ranking.classification.origin_classifier.OriginClassifier") as mock_cls:
            mock_cls.side_effect = [object(), object()]
            first_classifier = origin_classifier.get_or_create_classifier(reference_set)
            reference_set["Anna"] = ("European", 0.8, "MR", "")
            second_classifier = origin_classifier.get_or_create_classifier(reference_set)

        assert first_classifier is not second_classifier
        assert mock_cls.call_count == 2

    def test_origin_classifier_freezes_reference_names(self):
        """A classifier should not observe later mutations to caller-owned reference data."""
        reference_set = {"Anna": ("Nordic", 0.9, "AN", "")}

        with patch("st_name_ranking.classification.origin_classifier.get_ethnicolr_classifier", return_value=None):
            classifier = origin_classifier.OriginClassifier(reference_set, use_ethnidata=False)

        reference_set["Anna"] = ("European", 0.8, "MR", "")
        reference_set["Maria"] = ("European", 0.8, "MR", "")

        assert classifier.reference_names == {"Anna": ("Nordic", 0.9, "AN", "")}

    def test_get_or_create_classifier_uses_origin_classifier_boundary(self):
        """Test factory construction errors surface from origin_classifier directly."""
        with patch("st_name_ranking.classification.origin_classifier.OriginClassifier", side_effect=ImportError):
            with pytest.raises(ImportError):
                origin_classifier.get_or_create_classifier(reference_names={})


class TestGetRegionForNationality:
    """Tests for nationality-to-region mapping."""

    def test_region_found(self, initialized_db):
        """Test when nationality mapping exists in database."""
        # Use a country not in default mapping
        from st_name_ranking.persistence.database import get_connection

        with get_connection() as conn:
            # First delete if exists (to avoid primary key conflict)
            conn.execute(
                "DELETE FROM region_mapping WHERE nationality = ?",
                ("TestCountry",),
            )
            conn.execute(
                "INSERT INTO region_mapping (nationality, region) VALUES (?, ?)",
                ("TestCountry", "TestRegion"),
            )

        region, confidence = origin_classifier._get_region_for_nationality(
            "TestCountry",
        )
        assert region == "TestRegion"
        assert confidence == 1.0  # Default confidence adjustment

    def test_region_not_found(self, initialized_db):
        """Test when nationality mapping does not exist."""
        region, confidence = origin_classifier._get_region_for_nationality("XX")
        assert region == "International"
        assert confidence == 0.5  # Penalty for unknown region

    def test_region_with_confidence(self, initialized_db):
        """Test region mapping with confidence adjustment."""
        from st_name_ranking.persistence.database import get_connection

        with get_connection() as conn:
            # Use different country to avoid default mapping conflict
            conn.execute(
                "DELETE FROM region_mapping WHERE nationality = ?",
                ("TestCountry2",),
            )
            conn.execute(
                "INSERT INTO region_mapping (nationality, region) VALUES (?, ?)",
                ("TestCountry2", "TestRegion2"),
            )

        region, confidence = origin_classifier._get_region_for_nationality(
            "TestCountry2",
        )
        assert region == "TestRegion2"
        assert confidence == 1.0  # Exact match


class TestClassifyName:
    """Tests for classify_name function."""

    def test_classify_name_success(self, mock_classifier, initialized_db):
        """Test successful name classification."""
        # The mock classifier returns (region, confidence) tuple directly
        # Configure it to return a specific region
        mock_classifier.return_value = ("TestRegion", 0.85)

        result = classify_origins.classify_name("Anna")

        assert result == ("TestRegion", 0.85)
        mock_classifier.assert_called_once_with("Anna")

    def test_classify_name_unknown_region(
        self,
        mock_classifier,
        initialized_db,
    ):
        """Test classification with unknown nationality."""
        # Mock returns low confidence -> triggers fallback
        mock_classifier.return_value = ("Unknown", 0.05)

        result = classify_origins.classify_name("UnknownName")

        # Falls back to International due to low confidence
        assert result == ("International", 0.1)

    def test_classify_name_import_error(self, initialized_db):
        """Test when ethnidata is not installed."""
        # Need to patch the specific module where _create_ethnidata_classifier is used
        with patch(
            "st_name_ranking.classification.origin_classifier._create_ethnidata_classifier",
            return_value=None,
        ):
            # When ethnidata fails, classifier falls back to International
            result = classify_origins.classify_name("Anna")
            assert result == ("International", 0.1)

    def test_classify_name_exception(self, mock_classifier, initialized_db):
        """Test handling of classifier exceptions."""
        mock_classifier.side_effect = RuntimeError("Classifier error")

        with pytest.raises(RuntimeError, match="Classifier error"):
            classify_origins.classify_name("Anna")

    def test_reference_name_database_failure_is_not_cached_empty(self, initialized_db):
        """Reference lookup failures should remain distinguishable from no reference data."""
        classify_origins.reset_reference_cache()
        with patch(
            "st_name_ranking.classification.classify_origins.get_names_with_origins",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            with pytest.raises(RuntimeError, match="Failed to load origin-classification reference names"):
                classify_origins._get_reference_names()

        assert classify_origins._REFERENCE_NAMES_CACHE is None
        assert classify_origins._REFERENCE_NAMES_CACHE_DB_PATH is None

    def test_reference_name_cache_reloads_after_db_path_change(self, tmp_path):
        """Reference-name cache entries should be scoped to the active database."""
        classify_origins.reset_reference_cache()
        original_path = database.get_db_path()

        with patch(
            "st_name_ranking.classification.classify_origins.get_names_with_origins",
            side_effect=[
                {"Anna": ("Nordic", 0.9, "AN", "")},
                {"Maria": ("European", 0.8, "MR", "")},
            ],
        ) as get_names_with_origins:
            try:
                database.set_db_path(tmp_path / "first.db")
                first = classify_origins._get_reference_names()
                first_again = classify_origins._get_reference_names()

                database.set_db_path(tmp_path / "second.db")
                second = classify_origins._get_reference_names()
            finally:
                database.set_db_path(original_path)
                classify_origins.reset_reference_cache()

        assert first is first_again
        assert first == {"Anna": ("Nordic", 0.9, "AN", "")}
        assert second == {"Maria": ("European", 0.8, "MR", "")}
        assert get_names_with_origins.call_count == 2


class TestClassifyAllNames:
    """Tests for classify_all_names function."""

    @patch("st_name_ranking.classification.classify_origins.classify_batch")
    @patch("st_name_ranking.classification.classify_origins.get_unclassified_names")
    @patch("st_name_ranking.classification.classify_origins.update_name_origin")
    @patch(
        "st_name_ranking.classification.classify_origins.st",
        new_callable=MagicMock,
        create=True,
    )
    def test_classify_all_names_success(
        self,
        mock_st,
        mock_update,
        mock_get_unclassified,
        mock_classify_batch,
    ):
        """Test successful classification of all unclassified names."""
        mock_get_unclassified.return_value = [
            {"id": 1, "name": "Anna"},
            {"id": 2, "name": "Peter"},
            {"id": 3, "name": "Maria"},
        ]
        mock_classify_batch.return_value = 2  # 2 names classified (limit=2)

        result = classify_origins.classify_all_names(limit=2)

        # Should only process up to limit
        mock_get_unclassified.assert_called_once_with(2)
        # Should call classify_batch with the 2 limited names
        mock_classify_batch.assert_called_once()
        call_args = mock_classify_batch.call_args
        # First arg should be list of names (with id, name dicts)
        batch_arg = call_args[0][0]
        assert len(batch_arg) == 2  # Limited to 2 names
        assert batch_arg[0] == {"id": 1, "name": "Anna"}
        assert batch_arg[1] == {"id": 2, "name": "Peter"}
        # Check batch_size default (not provided, uses default)
        assert call_args[1] == {}

        # Should update database for each name (mocked inside classify_batch)
        # classify_batch is mocked, so update_name_origin not called directly
        mock_update.assert_not_called()

        # Note: classify_all_names uses logging, not Streamlit progress/toast
        # Original test expected progress/toast but they're not implemented

        assert result == 2  # mock returns 2

    @patch("st_name_ranking.classification.classify_origins.get_unclassified_names")
    @patch(
        "st_name_ranking.classification.classify_origins.st",
        new_callable=MagicMock,
        create=True,
    )
    def test_classify_all_names_no_unclassified(
        self,
        mock_st,
        mock_get_unclassified,
    ):
        """Test when no unclassified names exist."""
        mock_get_unclassified.return_value = []

        result = classify_origins.classify_all_names()

        assert result == 0
        # Note: classify_all_names uses logging, not Streamlit toast
        # Original test expected toast but it's not implemented

    @patch("st_name_ranking.classification.classify_origins.get_unclassified_names")
    @patch(
        "st_name_ranking.classification.classify_origins.st",
        new_callable=MagicMock,
        create=True,
    )
    def test_classify_all_names_import_error(
        self,
        mock_st,
        mock_get_unclassified,
    ):
        """Test when ethnidata is not installed."""
        mock_get_unclassified.return_value = [
            SimpleNamespace(id=1, name="Anna"),
            SimpleNamespace(id=2, name="Peter"),
        ]

        with patch("st_name_ranking.classification.classify_origins.get_or_create_classifier", side_effect=ImportError):
            with pytest.raises(ImportError):
                classify_origins.classify_all_names()

            # Note: classify_all_names uses logging, not Streamlit toast
            # Original test expected toast but it's not implemented

    @patch("st_name_ranking.classification.classify_origins.get_unclassified_names")
    def test_classify_all_names_reference_database_failure_aborts(
        self,
        mock_get_unclassified,
        initialized_db,
    ):
        """Database failures while loading reference names should abort the run."""
        classify_origins.reset_reference_cache()
        mock_get_unclassified.return_value = [SimpleNamespace(id=1, name="Anna")]

        with patch(
            "st_name_ranking.classification.classify_origins.get_names_with_origins",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            with pytest.raises(RuntimeError, match="Failed to load origin-classification reference names"):
                classify_origins.classify_all_names()

        assert classify_origins._REFERENCE_NAMES_CACHE is None
        assert classify_origins._REFERENCE_NAMES_CACHE_DB_PATH is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
