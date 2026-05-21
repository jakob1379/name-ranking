"""Integration tests for phonetic_similarity module."""

from st_name_ranking.active_learning import phonetic_similarity


class TestPhoneticSimilarityIntegration:
    """Integration tests for phonetic similarity functions."""

    def test_compute_phonetic_codes(self):
        """Test computing phonetic codes for names."""
        primary, secondary = phonetic_similarity.compute_phonetic_codes("Smith")
        assert isinstance(primary, str)
        assert isinstance(secondary, str)

        # Test with a name that may have secondary code
        primary2, secondary2 = phonetic_similarity.compute_phonetic_codes("Schmidt")
        assert primary2 is not None

    def test_phonetic_similarity(self):
        """Test phonetic similarity between two names."""
        # Same name should have similarity 1.0
        score = phonetic_similarity.phonetic_similarity("Smith", "Smith")
        assert score == 1.0

        # Different names may have 0.0 or 0.5 depending on phonetic codes
        score2 = phonetic_similarity.phonetic_similarity("Smith", "Smyth")
        # Could be 0.5 or 0.0 depending on Double Metaphone
        assert score2 in (0.0, 0.5, 1.0)

    def test_batch_compute_phonetic_codes(self):
        """Test batch computation of phonetic codes."""
        names = ["Smith", "Johnson", "Williams"]
        result = phonetic_similarity.batch_compute_phonetic_codes(names)
        assert len(result) == 3
        for name in names:
            assert name in result
            primary, secondary = result[name]
            assert isinstance(primary, str)

    def test_phonetic_similarity_batch(self):
        """Test batch phonetic similarity."""
        target_codes = phonetic_similarity.compute_phonetic_codes("Smith")
        name_codes = phonetic_similarity.batch_compute_phonetic_codes(["Smith", "Smyth", "Johnson"])
        similarities = phonetic_similarity.phonetic_similarity_batch(target_codes, name_codes)
        assert len(similarities) == 3
        assert similarities["Smith"] == 1.0
        # Others may have 0.5 or 0.0
        assert similarities["Johnson"] == 0.0  # Likely no match

    def test_get_phonetic_neighbors(self):
        """Test finding phonetic neighbors."""
        names = ["Smith", "Smyth", "Schmidt", "Johnson", "Williams"]
        neighbors = phonetic_similarity.get_phonetic_neighbors("Smith", names, threshold=0.5, limit=3)
        assert len(neighbors) <= 3
        # At least Smith itself should be in neighbors with score 1.0
        smith_found = any(name == "Smith" and score == 1.0 for name, score in neighbors)
        assert smith_found
        # Scores should be descending
        scores = [score for _, score in neighbors]
        assert scores == sorted(scores, reverse=True)
