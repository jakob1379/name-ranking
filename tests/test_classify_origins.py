"""Tests for st_name_ranking.classify_origins module."""

from unittest.mock import MagicMock, patch

import pytest

from st_name_ranking import classify_origins


class TestGetClassifier:
    """Tests for get_classifier function."""

    def setup_method(self):
        """Clear classifier cache before each test."""
        if hasattr(classify_origins.get_classifier, "_classifier"):
            delattr(classify_origins.get_classifier, "_classifier")

    def test_get_classifier_success(self):
        """Test successful classifier loading."""
        # Mock ethnidata import
        mock_n2n = MagicMock()
        with patch("ethnidata.EthniData", return_value=mock_n2n):
            classifier = classify_origins.get_classifier()
            assert classifier == mock_n2n

            # Second call should return cached classifier
            with patch("ethnidata.EthniData") as mock_ctor:
                classifier2 = classify_origins.get_classifier()
                assert classifier2 == mock_n2n

                mock_ctor.assert_not_called()

    def test_get_classifier_import_error(self):
        """Test when ethnidata is not installed."""
        with patch("ethnidata.EthniData", side_effect=ImportError), pytest.raises(ImportError):
            classify_origins.get_classifier()


class TestGetRegionForNationality:
    """Tests for get_region_for_nationality function."""

    def test_region_found(self, initialized_db):
        """Test when nationality mapping exists in database."""
        # Use a country not in default mapping
        from st_name_ranking.database import get_connection

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

        region, confidence = classify_origins.get_region_for_nationality(
            "TestCountry",
        )
        assert region == "TestRegion"
        assert confidence == 1.0  # Default confidence adjustment

    def test_region_not_found(self, initialized_db):
        """Test when nationality mapping does not exist."""
        region, confidence = classify_origins.get_region_for_nationality("XX")
        assert region == "International"
        assert confidence == 0.5  # Penalty for unknown region

    def test_region_with_confidence(self, initialized_db):
        """Test region mapping with confidence adjustment."""
        from st_name_ranking.database import get_connection

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

        region, confidence = classify_origins.get_region_for_nationality(
            "TestCountry2",
        )
        assert region == "TestRegion2"
        assert confidence == 1.0  # Exact match


class TestClassifyName:
    """Tests for classify_name function."""

    def test_classify_name_success(self, mock_classifier, initialized_db):
        """Test successful name classification."""
        # Setup region mapping
        from st_name_ranking.database import get_connection

        with get_connection() as conn:
            # Use unique country to avoid default mapping conflict
            conn.execute(
                "DELETE FROM region_mapping WHERE nationality = ?",
                ("TestCountry3",),
            )
            conn.execute(
                "INSERT INTO region_mapping (nationality, region) VALUES (?, ?)",
                ("TestCountry3", "TestRegion3"),
            )

        # Mock classifier returns TestCountry3 with high confidence
        mock_classifier.predict_nationality.return_value = {
            "country_name": "TestCountry3",
            "confidence": 0.85,
            "country": "TC",
            "region": "Test",
        }

        result = classify_origins.classify_name("Anna")

        assert result == ("TestRegion3", 0.85)
        mock_classifier.predict_nationality.assert_called_once_with("Anna")

    def test_classify_name_unknown_region(
        self,
        mock_classifier,
        initialized_db,
    ):
        """Test classification with unknown nationality."""
        mock_classifier.predict_nationality.return_value = {
            "country_name": "XX",
            "confidence": 0.75,
            "country": "XX",
            "region": "Unknown",
        }

        result = classify_origins.classify_name("UnknownName")

        assert result == ("International", 0.375)  # 0.75 * 0.5 penalty

    def test_classify_name_import_error(self, initialized_db):
        """Test when ethnidata is not installed."""
        with patch("ethnidata.EthniData", side_effect=ImportError):
            # Should return None because classification fails
            result = classify_origins.classify_name("Anna")
            assert result is None

    def test_classify_name_exception(self, mock_classifier, initialized_db):
        """Test handling of classifier exceptions."""
        mock_classifier.predict_nationality.side_effect = Exception(
            "Classifier error",
        )

        result = classify_origins.classify_name("Anna")
        # Should fall back to International with low confidence
        assert result == ("International", 0.1)


class TestClassifyAllNames:
    """Tests for classify_all_names function."""

    @patch("st_name_ranking.classify_origins.classify_batch")
    @patch("st_name_ranking.classify_origins.get_unclassified_names")
    @patch("st_name_ranking.classify_origins.update_name_origin")
    @patch(
        "st_name_ranking.classify_origins.st",
        new_callable=MagicMock,
        create=True,
    )
    @pytest.mark.skip(reason="UI integration optional")
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

        # Should show progress
        assert mock_st.progress.called
        assert mock_st.toast.called

        assert result == 2  # mock returns 2

    @patch("st_name_ranking.classify_origins.get_unclassified_names")
    @patch(
        "st_name_ranking.classify_origins.st",
        new_callable=MagicMock,
        create=True,
    )
    @pytest.mark.skip(reason="UI integration optional")
    def test_classify_all_names_no_unclassified(
        self,
        mock_st,
        mock_get_unclassified,
    ):
        """Test when no unclassified names exist."""
        mock_get_unclassified.return_value = []

        result = classify_origins.classify_all_names()

        assert result == 0
        mock_st.toast.assert_called_with(
            "No unclassified names found",
            icon="ℹ️",
        )

    @patch("st_name_ranking.classify_origins.get_unclassified_names")
    @patch(
        "st_name_ranking.classify_origins.st",
        new_callable=MagicMock,
        create=True,
    )
    @pytest.mark.skip(reason="UI integration optional")
    def test_classify_all_names_import_error(
        self,
        mock_st,
        mock_get_unclassified,
    ):
        """Test when ethnidata is not installed."""
        mock_get_unclassified.return_value = [
            {"id": 1, "name": "Anna"},
            {"id": 2, "name": "Peter"},
        ]

        with patch(
            "st_name_ranking.classify_origins.get_classifier",
            side_effect=ImportError,
        ):
            result = classify_origins.classify_all_names()

            assert result == 0
            mock_st.toast.assert_called_with(
                "ethnidata not installed. Install with: pip install ethnidata",
                icon="❌",
                duration="long",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
