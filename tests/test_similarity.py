"""
Tests for st_name_ranking.similarity module.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from st_name_ranking import similarity


class TestGetStringSimilarityScores:
    """Tests for get_string_similarity_scores function."""

    @patch("st_name_ranking.similarity.process")
    def test_basic_similarity(self, mock_process):
        """Test basic string similarity."""
        target = "Anna"
        candidates = ["Anna", "Anne", "Annie", "Peter"]

        # Mock rapidfuzz.process.extract
        mock_process.extract.return_value = [
            ("Anna", 100.0, 0),
            ("Anne", 85.0, 1),
            ("Annie", 75.0, 2),
        ]

        results = similarity.get_string_similarity_scores(
            target, candidates, limit=3
        )

        # Verify results
        assert results == [("Anna", 100.0), ("Anne", 85.0), ("Annie", 75.0)]
        mock_process.extract.assert_called_once_with(
            target, candidates, scorer=similarity.fuzz.ratio, limit=3
        )

    def test_empty_candidates(self):
        """Test with empty candidates list."""
        results = similarity.get_string_similarity_scores("Anna", [])
        assert results == []

    @patch("st_name_ranking.similarity.process")
    def test_limit(self, mock_process):
        """Test limit parameter."""
        target = "Anna"
        candidates = ["Anna", "Anne", "Annie", "Peter", "Paul"]

        mock_process.extract.return_value = [
            ("Anna", 100.0, 0),
            ("Anne", 85.0, 1),
        ]

        results = similarity.get_string_similarity_scores(
            target, candidates, limit=2
        )
        assert len(results) == 2
        mock_process.extract.assert_called_once_with(
            target, candidates, scorer=similarity.fuzz.ratio, limit=2
        )

    @patch("st_name_ranking.similarity.process")
    def test_logging(self, mock_process, caplog):
        """Test that logging occurs."""
        import logging

        caplog.set_level(logging.DEBUG)

        target = "Anna"
        candidates = ["Anna", "Anne"]
        mock_process.extract.return_value = [("Anna", 100.0, 0)]

        similarity.get_string_similarity_scores(target, candidates)

        # Check debug log
        assert "String similarity search" in caplog.text
        assert "target='Anna'" in caplog.text
        assert "candidates=2" in caplog.text


class TestLoadEmbeddingModel:
    """Tests for load_embedding_model function."""

    @patch("st_name_ranking.similarity.SentenceTransformer")
    def test_load_model(self, mock_sentence_transformer):
        """Test loading the embedding model."""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        model = similarity.load_embedding_model()

        assert model == mock_model
        mock_sentence_transformer.assert_called_once_with(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )


class TestGetVectorSimilarityScores:
    """Tests for get_vector_similarity_scores function."""

    def test_empty_candidates(self):
        """Test with empty candidates list."""
        mock_model = MagicMock()
        results = similarity.get_vector_similarity_scores(
            mock_model, "Anna", []
        )
        assert results == []
        # Model should not be called
        assert not mock_model.encode.called

    @patch("st_name_ranking.similarity.np")
    def test_vector_similarity(self, mock_np):
        """Test vector similarity calculation."""
        mock_model = MagicMock()

        # Mock embeddings
        target_embedding = np.array([[0.1, 0.2, 0.3]])
        candidate_embeddings = np.array(
            [
                [0.1, 0.2, 0.3],  # Same as target -> high similarity
                [0.4, 0.5, 0.6],  # Different -> lower similarity
                [-0.1, -0.2, -0.3],  # Opposite direction -> negative similarity
            ]
        )

        mock_model.encode.side_effect = [target_embedding, candidate_embeddings]

        # Mock numpy operations
        # dot returns a mock array with flatten method
        mock_dot_result = MagicMock()
        mock_flatten_result = np.array([1.0, 0.32, -1.0])  # shape (3,)
        mock_dot_result.flatten.return_value = mock_flatten_result
        mock_np.dot.return_value = mock_dot_result
        # argsort returns indices in ascending order: -1.0 (idx 2), 0.32 (idx 1), 1.0 (idx 0)
        mock_np.argsort.return_value = np.array([2, 1, 0])

        target = "Anna"
        candidates = ["Anna", "Anne", "Annie"]

        results = similarity.get_vector_similarity_scores(
            mock_model, target, candidates, limit=2
        )

        # Verify model.encode calls
        assert mock_model.encode.call_count == 2
        mock_model.encode.assert_any_call([target])
        mock_model.encode.assert_any_call(candidates)

        # Verify numpy operations were called
        assert mock_np.dot.called
        assert mock_np.argsort.called

        # Verify results (should be top 2 based on mock scores)
        assert len(results) == 2
        assert results[0] == ("Anna", 1.0)
        assert results[1] == ("Anne", 0.32)

    @patch("st_name_ranking.similarity.np")
    def test_limit_exceeds_candidates(self, mock_np):
        """Test when limit exceeds number of candidates."""
        mock_model = MagicMock()
        target_embedding = np.array([[0.1, 0.2, 0.3]])
        candidate_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        mock_model.encode.side_effect = [target_embedding, candidate_embeddings]
        # dot returns a mock array with flatten method
        mock_dot_result = MagicMock()
        mock_flatten_result = np.array([1.0, 0.32])
        mock_dot_result.flatten.return_value = mock_flatten_result
        mock_np.dot.return_value = mock_dot_result
        # argsort ascending: indices [1, 0] because 0.32 < 1.0
        mock_np.argsort.return_value = np.array([1, 0])

        target = "Anna"
        candidates = ["Anna", "Anne"]

        # Limit larger than candidates
        results = similarity.get_vector_similarity_scores(
            mock_model, target, candidates, limit=10
        )

        assert len(results) == 2
        assert results[0] == ("Anna", 1.0)
        assert results[1] == ("Anne", 0.32)

    @patch("st_name_ranking.similarity.np")
    def test_logging(self, mock_np, caplog):
        """Test that logging occurs."""
        import logging

        caplog.set_level(logging.DEBUG)

        mock_model = MagicMock()
        target_embedding = np.array([[0.1]])
        candidate_embeddings = np.array([[0.1]])

        mock_model.encode.side_effect = [target_embedding, candidate_embeddings]
        mock_np.dot.return_value = np.array([1.0])
        mock_np.argsort.return_value = np.array([0])
        mock_np.flatten.return_value = np.array([1.0])

        similarity.get_vector_similarity_scores(mock_model, "Anna", ["Anna"])

        # Check debug log
        assert "Vector similarity search" in caplog.text
        assert "target='Anna'" in caplog.text
        assert "candidates=1" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
