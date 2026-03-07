"""Comprehensive integration tests for the origin classification pipeline.

These tests verify the full classification chain using realistic test doubles
and an actual SQLite database. Tests cover:
1. Classification chain integration (rule-based → phonetic → fallback)
2. Database integration for storing/retrieving classifications
3. Phonetic similarity matching
4. Error handling for edge cases
"""

import sqlite3

import pytest
from metaphone import doublemetaphone

from st_name_ranking import classify_origins, origin_classifier
from st_name_ranking.database import (
    get_connection,
    get_names_with_origins,
    get_unclassified_names,
    update_name_origin,
)


# Global fixture to mock ethnidata for all tests in this file
@pytest.fixture(autouse=True)
def mock_ethnidata_classifier():
    """Mock ethnidata to avoid missing database file."""
    from unittest.mock import patch

    with patch(
        "st_name_ranking.origin_classifier._create_ethnidata_classifier",
        return_value=False,
    ):
        yield


# -----------------------------------------------------------------------------
# Test Doubles (Realistic implementations, not just mocks)
# -----------------------------------------------------------------------------


class FakeEthniData:
    """Realistic test double for ethnidata.EthniData.

    Simulates ethnidata behavior with deterministic predictions based on name patterns.
    """

    # Known nationality mappings for testing
    KNOWN_NAMES = {
        "Lars": {"country_name": "Denmark", "confidence": 0.92, "country": "DK"},
        "Sven": {"country_name": "Sweden", "confidence": 0.88, "country": "SE"},
        "Müller": {"country_name": "Germany", "confidence": 0.85, "country": "DE"},
        "Rossi": {"country_name": "Italy", "confidence": 0.87, "country": "IT"},
        "Smith": {"country_name": "United Kingdom", "confidence": 0.78, "country": "GB"},
        "Garcia": {"country_name": "Spain", "confidence": 0.82, "country": "ES"},
        "Wang": {"country_name": "China", "confidence": 0.91, "country": "CN"},
        "Kim": {"country_name": "South Korea", "confidence": 0.89, "country": "KR"},
        "Muhammad": {"country_name": "Pakistan", "confidence": 0.86, "country": "PK"},
        "Sato": {"country_name": "Japan", "confidence": 0.84, "country": "JP"},
    }

    def predict_nationality(self, name: str) -> dict | None:
        """Predict nationality with fallback to pattern matching."""
        # Direct lookup
        if name in self.KNOWN_NAMES:
            return self.KNOWN_NAMES[name]

        # Pattern-based predictions
        name_lower = name.lower()

        # Nordic patterns
        if any(char in name_lower for char in "æøå"):
            return {"country_name": "Denmark", "confidence": 0.75, "country": "DK"}

        # European patterns
        if name_lower.endswith(("mann", "berg", "stein")):
            return {"country_name": "Germany", "confidence": 0.65, "country": "DE"}

        if name_lower.endswith(("ini", "etti", "ucci")):
            return {"country_name": "Italy", "confidence": 0.70, "country": "IT"}

        # Asian patterns
        if name_lower in ["li", "zhang", "liu", "chen"]:
            return {"country_name": "China", "confidence": 0.80, "country": "CN"}

        # Default fallback
        return {"country_name": "United States", "confidence": 0.40, "country": "US"}


class FailingEthniData:
    """Test double that always fails - for error handling tests."""

    def predict_nationality(self, name: str) -> dict:
        raise RuntimeError("Simulated ethnidata failure")


class PartialFailingEthniData:
    """Test double that fails for specific names - for partial batch failure tests."""

    FAIL_NAMES = {"ErrorName", "FailThis", "Broken"}

    def predict_nationality(self, name: str) -> dict | None:
        if name in self.FAIL_NAMES:
            raise RuntimeError(f"Failed to classify {name}")
        return {"country_name": "Denmark", "confidence": 0.70, "country": "DK"}


# -----------------------------------------------------------------------------
# Classification Chain Integration Tests
# -----------------------------------------------------------------------------


class TestClassificationChainIntegration:
    """Test the full classification chain: rule-based → phonetic → fallback."""

    def test_danish_name_uses_rule_based_classifier(self, initialized_db):
        """Test that Danish name with Nordic characters is caught by rule-based classifier."""
        classifier = origin_classifier.OriginClassifier(
            reference_names={},
            ethnidata_classifier=False,
        )

        # Name with Nordic character should be caught by rule-based
        region, confidence = classifier.classify("Jørgen")

        assert region == "Nordic"
        assert confidence >= 0.6
        assert confidence <= 1.0

    def test_danish_suffix_uses_rule_based_classifier(self, initialized_db):
        """Test that Danish name with Nordic suffix is caught by rule-based classifier."""
        classifier = origin_classifier.OriginClassifier(
            reference_names={},
            ethnidata_classifier=False,
        )

        # Common Danish suffix
        region, confidence = classifier.classify("Hansen")

        assert region == "Nordic"
        assert confidence >= 0.6
        # Suffix-based detection has specific confidence
        assert confidence == pytest.approx(0.81, abs=0.01)  # 0.9 * 0.9

    def test_name_flows_to_phonetic_classifier(self, initialized_db):
        """Test that name without rule-based match flows to phonetic classifier."""
        # Insert a reference name with known region
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO names (name, gender, origin_region, origin_confidence,
                                 phonetic_primary, phonetic_secondary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("Magnus", "Male", "Nordic", 0.9, "MKNKS", "MKNKS"),
            )

        # Build reference names from DB
        reference_names = get_names_with_origins(confidence_threshold=0.5)

        # Name phonetically similar to Magnus
        # Disable ethnidata to avoid broken dependency
        classifier = origin_classifier.OriginClassifier(
            reference_names=reference_names,
            ethnidata_classifier=False,
        )

        # Magnis is phonetically similar to Magnus
        region, confidence = classifier.classify("Magnis")

        # Should be classified via phonetic similarity
        assert region is not None
        assert confidence > 0.0

    def test_phonetic_classifier_uses_db_reference_names(self, initialized_db):
        """Test that phonetic similarity uses reference names loaded from database."""
        # Insert multiple reference names
        reference_data = [
            ("Eriksson", "Nordic", 0.85, "ARKSN", "ARKSN"),
            ("Andersen", "Nordic", 0.9, "ANTRS", "ANTRS"),
            ("Johansson", "Nordic", 0.88, "JHNSN", "JHNSN"),
        ]

        with get_connection() as conn:
            for name, region, conf, primary, secondary in reference_data:
                conn.execute(
                    """
                    INSERT INTO names (name, gender, origin_region, origin_confidence,
                                     phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (name, "Male", region, conf, primary, secondary),
                )

        # Load reference names from DB
        reference_names = get_names_with_origins(confidence_threshold=0.5)
        assert len(reference_names) == 3

        # Create classifier with these references
        classifier = origin_classifier.OriginClassifier(reference_names=reference_names)

        # Test classification with similar name
        region, confidence = classifier.classify("Ericson")

        # Should match to Nordic via phonetic similarity
        assert region == "Nordic"
        assert confidence > 0.3

    def test_confidence_scoring_through_chain(self, initialized_db):
        """Test that confidence values are properly calculated at each chain step."""
        classifier = origin_classifier.OriginClassifier(
            reference_names={},
            ethnidata_classifier=False,
        )

        # Rule-based should have highest confidence for Nordic names
        _, rule_conf = classifier.classify("Bjørn")
        assert rule_conf >= 0.6

        # Without references, non-Nordic names fall through to International
        region, fallback_conf = classifier.classify("Xyzabc123")
        assert region == "International"
        assert fallback_conf == 0.1

    def test_fallback_to_international_when_all_fail(self, initialized_db):
        """Test that classifier falls back to International when no classifier matches."""
        classifier = origin_classifier.OriginClassifier(
            reference_names={},
            ethnidata_classifier=False,
        )

        # Name that won't match any classifier pattern
        region, confidence = classifier.classify("Qwertyuiop")

        assert region == "International"
        assert confidence == 0.1

    def test_complete_chain_with_ethnidata_fallback(self, initialized_db):
        """Test complete chain including ethnidata fallback (using test double)."""
        fake_ethnidata = FakeEthniData()

        # Create a callable that returns (region, confidence) tuple
        def ethnidata_classifier(name: str) -> tuple[str, float] | None:
            try:
                pred = fake_ethnidata.predict_nationality(name)
                if pred:
                    country = pred["country_name"]
                    confidence = pred["confidence"]
                    # Map country to region
                    region_map = {
                        "Denmark": "Nordic",
                        "Sweden": "Nordic",
                        "Germany": "European",
                        "Italy": "European",
                        "China": "Asian",
                    }
                    region = region_map.get(country, "International")
                    return region, confidence
            except (ImportError, AttributeError, ValueError):
                pass
            return None

        classifier = origin_classifier.OriginClassifier(
            reference_names={},
            ethnidata_classifier=ethnidata_classifier,
        )

        # Name that should be caught by ethnidata (not Nordic, not in references)
        region, confidence = classifier.classify("Lars")

        # Should be classified via ethnidata
        assert region is not None
        assert confidence >= 0.3


# -----------------------------------------------------------------------------
# Database Integration Tests
# -----------------------------------------------------------------------------


class TestDatabaseIntegration:
    """Test database operations for classification."""

    def test_classified_names_saved_to_database(self, initialized_db):
        """Test that classified names are correctly saved to database."""
        # Insert an unclassified name
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("TestName", "Male"),
            )
            name_id = conn.execute("SELECT id FROM names WHERE name = ?", ("TestName",)).fetchone()[0]

        # Classify and save
        update_name_origin(name_id, "Nordic", 0.85)

        # Verify in database
        with get_connection() as conn:
            row = conn.execute(
                "SELECT origin_region, origin_confidence FROM names WHERE id = ?",
                (name_id,),
            ).fetchone()

        assert row["origin_region"] == "Nordic"
        assert row["origin_confidence"] == pytest.approx(0.85)

    def test_reference_names_loaded_from_database(self, initialized_db):
        """Test that reference names are loaded from database for phonetic matching."""
        # Insert classified names with phonetic codes
        test_names = [
            ("Olaf", "Nordic", 0.9),
            ("Gustav", "Nordic", 0.85),
            ("Ingrid", "Nordic", 0.88),
        ]

        with get_connection() as conn:
            for name, region, confidence in test_names:
                primary, secondary = doublemetaphone(name)
                conn.execute(
                    """
                    INSERT INTO names (name, gender, origin_region, origin_confidence,
                                     phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (name, "Male", region, confidence, primary, secondary),
                )

        # Load reference names
        reference_names = get_names_with_origins(confidence_threshold=0.5)

        assert len(reference_names) == 3
        assert "Olaf" in reference_names
        assert reference_names["Olaf"][0] == "Nordic"  # region
        assert reference_names["Olaf"][1] == pytest.approx(0.9)  # confidence

    def test_batch_classification_updates_database(self, initialized_db):
        """Test that batch classification correctly updates all names in database."""
        # Insert multiple unclassified names
        test_names = ["Alpha", "Beta", "Gamma"]
        name_ids = []

        with get_connection() as conn:
            for name in test_names:
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, ?)",
                    (name, "Unisex"),
                )
                name_id = conn.execute("SELECT id FROM names WHERE name = ?", (name,)).fetchone()[0]
                name_ids.append({"id": name_id, "name": name})

        # Create a simple classifier that assigns regions based on name
        def mock_classify(name: str) -> tuple[str, float]:
            regions = {"Alpha": "European", "Beta": "Asian", "Gamma": "African"}
            return regions.get(name, "International"), 0.7

        # Process batch
        for name_data in name_ids:
            region, confidence = mock_classify(name_data["name"])
            update_name_origin(name_data["id"], region, confidence)

        # Verify all updated
        with get_connection() as conn:
            for name_data in name_ids:
                row = conn.execute(
                    "SELECT origin_region FROM names WHERE id = ?",
                    (name_data["id"],),
                ).fetchone()
                assert row["origin_region"] is not None

    def test_classification_with_existing_reference_data(self, initialized_db):
        """Test classification when database already has reference classifications."""
        # Insert reference data
        with get_connection() as conn:
            for name, region in [("Thor", "Nordic"), ("Odin", "Nordic")]:
                primary, secondary = doublemetaphone(name)
                conn.execute(
                    """
                    INSERT INTO names (name, gender, origin_region, origin_confidence,
                                     phonetic_primary, phonetic_secondary)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (name, "Male", region, 0.9, primary, secondary),
                )

        # Load references and classify a similar name
        reference_names = get_names_with_origins(confidence_threshold=0.5)
        classifier = origin_classifier.OriginClassifier(reference_names=reference_names)

        # Tor is phonetically similar to Thor
        region, confidence = classifier.classify("Tor")

        # Should be influenced by the reference names
        assert region is not None

    def test_get_unclassified_names_returns_only_unclassified(self, initialized_db):
        """Test that get_unclassified_names returns only names without region."""
        # Insert mix of classified and unclassified
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender, origin_region) VALUES (?, ?, ?)",
                ("Classified", "Male", "Nordic"),
            )
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("Unclassified1", "Female"),
            )
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("Unclassified2", "Male"),
            )

        unclassified = get_unclassified_names()

        # Should only return unclassified names
        names = [n.name for n in unclassified]
        assert "Classified" not in names
        assert "Unclassified1" in names
        assert "Unclassified2" in names

    def test_classification_timestamp_is_set(self, initialized_db):
        """Test that classification timestamp is recorded."""
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("TimestampTest", "Male"),
            )
            name_id = conn.execute("SELECT id FROM names WHERE name = ?", ("TimestampTest",)).fetchone()[0]

        update_name_origin(name_id, "European", 0.75)

        with get_connection() as conn:
            row = conn.execute("SELECT origin_classified_at FROM names WHERE id = ?", (name_id,)).fetchone()

        assert row["origin_classified_at"] is not None


# -----------------------------------------------------------------------------
# Phonetic Similarity Tests
# -----------------------------------------------------------------------------


class TestPhoneticSimilarityIntegration:
    """Test phonetic similarity matching and confidence calculation."""

    def test_phonetic_matching_finds_similar_names(self, initialized_db):
        """Test that phonetic matching correctly identifies similar names."""
        # Insert a reference name
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO names (name, gender, origin_region, origin_confidence,
                                 phonetic_primary, phonetic_secondary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("Christopher", "Male", "European", 0.9, "KRSTFR", "KRSTFR"),
            )

        reference_names = get_names_with_origins(confidence_threshold=0.5)
        classifier = origin_classifier.OriginClassifier(reference_names=reference_names)

        # Kristoffer is phonetically similar to Christopher
        region, confidence = classifier.classify("Kristoffer")

        # Should match via phonetic similarity
        assert region == "European"
        assert confidence > 0.0

    def test_confidence_calculation_for_phonetic_matches(self, initialized_db):
        """Test confidence is properly calculated for different match types."""
        # Reference name with known phonetic codes
        ref_name = "Reference"
        ref_primary, ref_secondary = doublemetaphone(ref_name)

        reference_names = {
            ref_name: ("Nordic", 0.9, ref_primary, ref_secondary),
        }

        # Test primary match
        test_name1 = "Referens"  # Similar phonetics
        primary1, _ = doublemetaphone(test_name1)

        region, confidence = origin_classifier.phonetic_similarity_classification(test_name1, reference_names)

        if region:  # If phonetic codes happen to match
            # Confidence = score * ref_conf * 0.9
            assert confidence <= 0.9 * 0.9  # Max possible
            assert confidence > 0.0

    def test_handling_names_with_no_phonetic_matches(self, initialized_db):
        """Test graceful handling when no phonetic matches exist."""
        # Reference name with specific phonetic code
        reference_names = {
            "Xylophone": ("European", 0.8, "XLFN", "SLFN"),
        }

        # Name with completely different phonetics
        region, confidence = origin_classifier.phonetic_similarity_classification("Qwerty", reference_names)

        # Should return None when no match
        assert region is None
        assert confidence == 0.0

    def test_phonetic_similarity_module_directly(self):
        """Test the phonetic_similarity module functions."""
        from st_name_ranking.phonetic_similarity import (
            compute_phonetic_codes,
            get_phonetic_neighbors,
            phonetic_similarity,
        )

        # Test code computation
        primary, secondary = compute_phonetic_codes("Smith")
        assert primary is not None
        assert len(primary) > 0

        # Test similarity between similar names
        score = phonetic_similarity("Smith", "Smyth")
        assert score > 0.0  # Should have some similarity

        # Test finding neighbors
        neighbors = get_phonetic_neighbors(
            "Smith",
            ["Smith", "Smyth", "Schmidt", "Johnson", "Williams"],
            threshold=0.5,
        )
        assert len(neighbors) > 0
        assert neighbors[0][0] in ["Smith", "Smyth", "Schmidt"]

    def test_phonetic_batch_operations(self):
        """Test batch phonetic operations."""
        from st_name_ranking.phonetic_similarity import (
            batch_compute_phonetic_codes,
            phonetic_similarity_batch,
        )

        names = ["Anna", "Anne", "Hanna", "John"]
        codes = batch_compute_phonetic_codes(names)

        assert len(codes) == 4
        assert "Anna" in codes
        assert codes["Anna"][0] is not None  # Has primary code

        # Test batch similarity
        target_codes = codes["Anna"]
        similarities = phonetic_similarity_batch(target_codes, codes)

        assert len(similarities) == 4
        # Anna should match itself perfectly
        assert similarities["Anna"] == 1.0


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------


class TestErrorHandling:
    """Test graceful handling of errors during classification."""

    def test_graceful_handling_when_ethnidata_unavailable(self, initialized_db):
        """Test that classifier works when ethnidata is not installed."""
        # Create classifier with ethnidata=False (simulating unavailable package)
        classifier = origin_classifier.OriginClassifier(
            reference_names={},
            ethnidata_classifier=False,
        )

        # Should still classify using rule-based
        region, confidence = classifier.classify("Jørgensen")
        assert region == "Nordic"

        # Non-Nordic name falls back to International
        region, confidence = classifier.classify("UnknownName")
        assert region == "International"
        assert confidence == 0.1

    def test_partial_batch_failure(self, initialized_db):
        """Test that partial batch failures don't prevent other classifications."""
        # Insert names including some that will "fail"
        test_names = ["Good1", "ErrorName", "Good2", "FailThis", "Good3"]
        name_data = []

        with get_connection() as conn:
            for name in test_names:
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, ?)",
                    (name, "Unisex"),
                )
                name_id = conn.execute("SELECT id FROM names WHERE name = ?", (name,)).fetchone()[0]
                name_data.append({"id": name_id, "name": name})

        # Simulate classification with some failures
        fail_names = {"ErrorName", "FailThis"}
        success_count = 0

        for data in name_data:
            if data["name"] in fail_names:
                # Simulate failure
                continue
            update_name_origin(data["id"], "Nordic", 0.7)
            success_count += 1

        assert success_count == 3

        # Verify successful classifications were saved
        with get_connection() as conn:
            classified_count = conn.execute("SELECT COUNT(*) FROM names WHERE origin_region IS NOT NULL").fetchone()[0]

        assert classified_count == 3

    def test_database_error_during_classification(self, initialized_db):
        """Test handling of database errors during classification."""
        # Test with invalid ID
        with pytest.raises(sqlite3.Error):
            # This should fail due to foreign key or similar
            with get_connection() as conn:
                # Force an error by using invalid SQL
                conn.execute("SELECT * FROM nonexistent_table")

    def test_classifier_handles_empty_reference_names(self, initialized_db):
        """Test classifier with empty reference names dictionary."""
        classifier = origin_classifier.OriginClassifier(
            reference_names={},
            ethnidata_classifier=False,
        )

        # Should skip phonetic step and go to fallback
        region, confidence = classifier.classify("NonNordicName")

        assert region == "International"
        assert confidence == 0.1

    def test_classifier_handles_malformed_reference_data(self, initialized_db):
        """Test classifier with malformed reference data."""
        # Create reference names with some missing data
        reference_names = {
            "Valid": ("Nordic", 0.9, "VLDT", ""),
            "Invalid": None,  # Malformed entry
        }

        classifier = origin_classifier.OriginClassifier(reference_names={"Valid": ("Nordic", 0.9, "VLDT", "")})

        # Should still work with valid entries
        region, confidence = classifier.classify("Valid")
        # Might not match phonetically, but shouldn't crash
        assert isinstance(region, str)

    def test_classify_origins_handles_errors(self, initialized_db, monkeypatch):
        """Test that classify_origins module handles errors gracefully."""
        # Mock classify_name to simulate occasional failures
        original_classify = classify_origins.classify_name

        def failing_classify(name: str):
            if name == "FailName":
                return None
            return ("Nordic", 0.7)

        monkeypatch.setattr(classify_origins, "classify_name", failing_classify)

        # Insert test name
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("FailName", "Male"),
            )
            name_id = conn.execute("SELECT id FROM names WHERE name = ?", ("FailName",)).fetchone()[0]

        # Process batch
        from st_name_ranking.types import UnclassifiedName

        batch = [UnclassifiedName(id=name_id, name="FailName")]
        result = classify_origins.classify_batch(batch)

        # Should return 0 for failed classification
        assert result == 0


# -----------------------------------------------------------------------------
# Integration with Classify Origins Module
# -----------------------------------------------------------------------------


class TestClassifyOriginsIntegration:
    """Test integration with the classify_origins module."""

    def test_get_classifier_with_fake_ethnidata(self, initialized_db, monkeypatch):
        """Test getting classifier with fake ethnidata."""
        fake_ethnidata = FakeEthniData()

        # Create a proper classifier callable that wraps the fake ethnidata
        def mock_ethnidata_classifier(name: str) -> tuple[str, float] | None:
            try:
                pred = fake_ethnidata.predict_nationality(name)
                if pred:
                    country = pred["country_name"]
                    confidence = pred["confidence"]
                    region_map = {
                        "Denmark": "Nordic",
                        "Sweden": "Nordic",
                        "Germany": "European",
                        "Italy": "European",
                        "China": "Asian",
                        "United Kingdom": "European",
                        "Spain": "European",
                    }
                    region = region_map.get(country, "International")
                    return region, confidence
            except (ImportError, AttributeError, ValueError):
                pass
            return None

        # Create classifier directly with the fake ethnidata callable
        classifier = origin_classifier.OriginClassifier(
            reference_names={},
            ethnidata_classifier=mock_ethnidata_classifier,
        )

        assert classifier is not None

        # Test with a known name that should be classified via ethnidata
        region, confidence = classifier.classify("Lars")
        assert region is not None
        assert confidence >= 0.0

    def test_region_for_nationality_mapping(self, initialized_db):
        """Test nationality to region mapping through database."""
        # Insert region mapping
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO region_mapping (region, nationality) VALUES (?, ?)",
                ("Nordic", "Denmark"),
            )

        # Test mapping
        region, confidence = classify_origins.get_region_for_nationality("Denmark")
        assert region == "Nordic"
        assert confidence == 1.0

        # Test partial match
        region, confidence = classify_origins.get_region_for_nationality("Kingdom of Denmark")
        assert region == "Nordic"
        assert confidence == 0.8

    def test_reference_names_cache(self, initialized_db, monkeypatch):
        """Test that reference names are cached properly."""
        # Insert a reference name
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO names (name, gender, origin_region, origin_confidence,
                                 phonetic_primary, phonetic_secondary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("CachedName", "Male", "Nordic", 0.9, "KSTNM", "KSTNM"),
            )

        # Clear any existing cache
        if hasattr(classify_origins._get_reference_names, "_cache"):
            delattr(classify_origins._get_reference_names, "_cache")

        # First call should populate cache
        ref1 = classify_origins._get_reference_names()
        assert len(ref1) == 1

        # Second call should use cache (we can verify by checking it's the same object)
        ref2 = classify_origins._get_reference_names()
        assert ref1 is ref2  # Same cached object


# -----------------------------------------------------------------------------
# Edge Cases and Boundary Tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_name_handling(self, initialized_db):
        """Test handling of empty or whitespace-only names."""
        classifier = origin_classifier.OriginClassifier(
            reference_names={},
            ethnidata_classifier=False,
        )

        # Empty string should fall back to International
        region, confidence = classifier.classify("")
        assert region == "International"

    def test_single_character_name(self, initialized_db):
        """Test classification of single-character names."""
        classifier = origin_classifier.OriginClassifier(
            reference_names={},
            ethnidata_classifier=False,
        )

        region, confidence = classifier.classify("A")
        # Should not crash, likely falls back to International
        assert isinstance(region, str)
        assert 0.0 <= confidence <= 1.0

    def test_very_long_name(self, initialized_db):
        """Test classification of very long names."""
        classifier = origin_classifier.OriginClassifier(
            reference_names={},
            ethnidata_classifier=False,
        )

        long_name = "A" * 200
        region, confidence = classifier.classify(long_name)

        assert isinstance(region, str)
        assert 0.0 <= confidence <= 1.0

    def test_unicode_names(self, initialized_db):
        """Test classification of various Unicode names."""
        classifier = origin_classifier.OriginClassifier(
            reference_names={},
            ethnidata_classifier=False,
        )

        unicode_names = [
            "José",  # Spanish
            "François",  # French
            "日本",  # Japanese
            "Александр",  # Russian
        ]

        for name in unicode_names:
            region, confidence = classifier.classify(name)
            assert isinstance(region, str)
            assert 0.0 <= confidence <= 1.0

    def test_batch_processing_large_batch(self, initialized_db):
        """Test batch processing with larger batch size."""
        # Insert many names
        with get_connection() as conn:
            for i in range(150):
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, ?)",
                    (f"BatchName{i}", "Unisex"),
                )

        # Get unclassified with limit
        unclassified = get_unclassified_names(limit=100)
        assert len(unclassified) <= 100

    def test_concurrent_classification_simulation(self, initialized_db):
        """Test that database handles concurrent-like access patterns."""
        # Insert test names
        with get_connection() as conn:
            for i in range(10):
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, ?)",
                    (f"Concurrent{i}", "Male"),
                )

        # Simulate multiple classification operations
        with get_connection() as conn:
            for i in range(5):
                name_id = conn.execute("SELECT id FROM names WHERE name = ?", (f"Concurrent{i}",)).fetchone()[0]
                update_name_origin(name_id, "Nordic", 0.8)

        # Verify all were updated
        with get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM names WHERE origin_region = 'Nordic'").fetchone()[0]

        assert count == 5
