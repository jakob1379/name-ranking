"""Integration tests for origin_classifier module."""

from unittest.mock import MagicMock, patch

import pytest

from st_name_ranking import origin_classifier


class TestOriginClassifierIntegration:
    """Integration tests for origin classifier."""

    @pytest.mark.skip(reason="Complex mocking broken - needs proper mock configuration")
    def test_get_classifier_with_mocked_dependencies(self, initialized_db):
        """Test getting classifier with mocked external dependencies."""
        # Mock ethnidata and ethnicolr imports
        mock_ethnidata = MagicMock()
        mock_ethnidata_classifier = MagicMock()
        mock_ethnidata_classifier.classify.return_value = ("Nordic", 0.8)
        mock_ethnidata.EthniData.return_value = mock_ethnidata_classifier

        mock_ethnicolr = MagicMock()
        mock_ethnicolr.pred_wiki_ln.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "ethnidata": mock_ethnidata,
                "ethnicolr": mock_ethnicolr,
            },
        ):
            classifier = origin_classifier.get_classifier(reference_names={})
            assert classifier is not None

            # Test classification with a name
            result = classifier.classify("TestName")
            assert result is not None
            region, confidence = result
            assert isinstance(region, str)
            assert isinstance(confidence, float)

    def test_rule_based_nordic_detection(self):
        """Test rule-based Nordic detection."""
        # Danish name with Nordic character
        region, confidence = origin_classifier.rule_based_nordic_detection("Bjørn")
        assert region == "Nordic"
        assert confidence > 0.5

        # Non-Nordic name
        region, confidence = origin_classifier.rule_based_nordic_detection("Xavier")
        assert region is None
        assert confidence == 0.0

    def test_rule_based_nordic_detection_comprehensive(self, monkeypatch):
        """Test additional branches of rule-based Nordic detection."""
        # Suffix detection
        region, confidence = origin_classifier.rule_based_nordic_detection("Hansen")
        assert region == "Nordic"
        assert confidence == pytest.approx(0.9 * 0.9)  # weight * discount

        # Given name ending detection
        region, confidence = origin_classifier.rule_based_nordic_detection("Alfred")
        assert region == "Nordic"
        assert confidence == pytest.approx(0.65)

        # No Nordic patterns (before mock)
        region, confidence = origin_classifier.rule_based_nordic_detection("Zhang")
        assert region is None
        assert confidence == 0.0

        # Phonetic pattern detection via mock
        monkeypatch.setattr("st_name_ranking.origin_classifier.doublemetaphone", lambda _: ("HJ", ""))
        region, confidence = origin_classifier.rule_based_nordic_detection("Test")
        assert region == "Nordic"
        assert confidence == pytest.approx(0.75)

    def test_phonetic_similarity_classification(self, initialized_db):
        """Test phonetic similarity classification with reference names."""
        from metaphone import doublemetaphone

        from st_name_ranking.database import get_connection

        ref_name = "ReferenceName"
        region = "Nordic"
        confidence = 0.9
        primary, secondary = doublemetaphone(ref_name)

        # Insert a reference name with known region
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO names (name, gender, origin_region, origin_confidence) VALUES (?, ?, ?, ?)",
                (ref_name, "Unisex", region, confidence),
            )

        # Build reference names dictionary with correct phonetic codes
        reference_names = {ref_name: (region, confidence, primary, secondary)}

        # Test phonetic similarity classification with slightly misspelled name
        result = origin_classifier.phonetic_similarity_classification(
            "RefrenceName",  # Slightly misspelled
            reference_names,
        )
        # Should match due to phonetic similarity
        assert result is not None
        region_result, confidence_result = result
        assert region_result == region
        assert confidence_result > 0.5

    @pytest.mark.skip(reason="Complex mocking of doublemetaphone affects both test and reference names")
    def test_phonetic_similarity_classification_secondary(self, monkeypatch):
        """Test phonetic similarity classification secondary matches and no-match fallback."""
        # Reference name with known phonetic codes
        ref_region = "Nordic"
        ref_conf = 0.9
        ref_primary = "ABCD"
        ref_secondary = "EFGH"
        reference_names = {"RefName": (ref_region, ref_conf, ref_primary, ref_secondary)}

        # Mock doublemetaphone to return specific primary/secondary for input name
        monkeypatch.setattr(
            "st_name_ranking.origin_classifier.doublemetaphone",
            lambda _: ("EFGH", "IJKL"),  # primary matches ref_secondary
        )
        region, confidence = origin_classifier.phonetic_similarity_classification("TestName", reference_names)
        assert region == ref_region
        assert confidence == pytest.approx(0.8 * ref_conf * 0.9)

        # Test secondary matches ref_primary
        monkeypatch.setattr(
            "st_name_ranking.origin_classifier.doublemetaphone",
            lambda _: ("WXYZ", "ABCD"),  # secondary matches ref_primary
        )
        region, confidence = origin_classifier.phonetic_similarity_classification("TestName2", reference_names)
        assert region == ref_region
        assert confidence == pytest.approx(0.8 * ref_conf * 0.9)

        # Test secondary matches ref_secondary
        monkeypatch.setattr(
            "st_name_ranking.origin_classifier.doublemetaphone",
            lambda _: ("WXYZ", "EFGH"),  # secondary matches ref_secondary
        )
        region, confidence = origin_classifier.phonetic_similarity_classification("TestName3", reference_names)
        assert region == ref_region
        assert confidence == pytest.approx(0.7 * ref_conf * 0.9)

        # Test first character match
        monkeypatch.setattr(
            "st_name_ranking.origin_classifier.doublemetaphone",
            lambda _: ("AXXX", "YYYY"),  # primary[0] matches ref_primary[0]
        )
        region, confidence = origin_classifier.phonetic_similarity_classification("TestName4", reference_names)
        assert region == ref_region
        assert confidence == pytest.approx(0.5 * ref_conf * 0.9)

        # Test no match (score 0)
        monkeypatch.setattr("st_name_ranking.origin_classifier.doublemetaphone", lambda _: ("ZZZZ", "WWWW"))
        region, confidence = origin_classifier.phonetic_similarity_classification("TestName5", reference_names)
        assert region is None
        assert confidence == 0.0

    def test_get_classifier_without_ethnidata(self, initialized_db):
        """Test classifier when ethnidata is not installed."""
        from unittest.mock import patch

        # Patch ethnidata classifier to simulate missing package
        with patch("st_name_ranking.origin_classifier._create_ethnidata_classifier", return_value=None):
            classifier = origin_classifier.get_classifier(reference_names={})
            # Should still return a classifier (using rule-based and phonetic)
            assert classifier is not None

            # Classify a name - should fall back to International with low confidence
            result = classifier.classify("TestName")
            # Should return tuple (region, confidence)
            assert isinstance(result, tuple)
            assert len(result) == 2
            region, confidence = result
            assert region == "International"
            assert confidence == 0.1

    def test_get_region_for_nationality(self, initialized_db):
        """Test mapping nationality to region with exact, partial, and default matches."""
        from st_name_ranking.database import get_connection

        # Insert a custom mapping for partial match testing
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO region_mapping (region, nationality) VALUES (?, ?)",
                ("European", "Testland"),
            )

        # Exact match
        region, confidence = origin_classifier._get_region_for_nationality("Testland")
        assert region == "European"
        assert confidence == 1.0

        # Partial match: nationality contains stored substring
        region, confidence = origin_classifier._get_region_for_nationality("Greater Testland")
        assert region == "European"
        assert confidence == 0.8

        # Partial match: stored nationality contains substring
        region, confidence = origin_classifier._get_region_for_nationality("Test")
        assert region == "European"
        assert confidence == 0.8

        # Default fallback for unknown nationality
        region, confidence = origin_classifier._get_region_for_nationality("UnknownCountry")
        assert region == "International"
        assert confidence == 0.5
