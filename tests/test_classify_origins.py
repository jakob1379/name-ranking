"""
Tests for st_name_ranking.classify_origins module.
"""
from unittest.mock import patch, MagicMock

import pytest

from st_name_ranking import classify_origins, database


class TestGetClassifier:
    """Tests for get_classifier function."""
    
    def test_get_classifier_success(self):
        """Test successful classifier loading."""
        # Mock name2nat import
        mock_n2n = MagicMock()
        with patch('st_name_ranking.classify_origins.Name2nat', return_value=mock_n2n):
            classifier = classify_origins.get_classifier()
            assert classifier == mock_n2n
            
            # Second call should return cached classifier
            with patch('st_name_ranking.classify_origins.Name2nat') as mock_ctor:
                classifier2 = classify_origins.get_classifier()
                assert classifier2 == mock_n2n
                mock_ctor.assert_not_called()
    
    def test_get_classifier_import_error(self):
        """Test when name2nat is not installed."""
        with patch('st_name_ranking.classify_origins.Name2nat', side_effect=ImportError):
            with pytest.raises(ImportError, match="name2nat not installed"):
                classify_origins.get_classifier()


class TestGetRegionForNationality:
    """Tests for get_region_for_nationality function."""
    
    def test_region_found(self, initialized_db):
        """Test when nationality mapping exists in database."""
        # First ensure mapping exists
        from st_name_ranking.database import get_connection
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO region_mapping (nationality, region) VALUES (?, ?)",
                ("DK", "Scandinavia")
            )
        
        region, confidence = classify_origins.get_region_for_nationality("DK")
        assert region == "Scandinavia"
        assert confidence == 1.0  # Default confidence adjustment
    
    def test_region_not_found(self, initialized_db):
        """Test when nationality mapping does not exist."""
        region, confidence = classify_origins.get_region_for_nationality("XX")
        assert region == "Unknown"
        assert confidence == 0.5  # Penalty for unknown region
    
    def test_region_with_confidence(self, initialized_db):
        """Test region mapping with confidence adjustment."""
        from st_name_ranking.database import get_connection
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO region_mapping (nationality, region, confidence_adjustment) VALUES (?, ?, ?)",
                ("SE", "Scandinavia", 0.8)
            )
        
        region, confidence = classify_origins.get_region_for_nationality("SE")
        assert region == "Scandinavia"
        assert confidence == 0.8


class TestClassifyName:
    """Tests for classify_name function."""
    
    def test_classify_name_success(self, mock_classifier, initialized_db):
        """Test successful name classification."""
        # Setup region mapping
        from st_name_ranking.database import get_connection
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO region_mapping (nationality, region) VALUES (?, ?)",
                ("DK", "Scandinavia")
            )
        
        # Mock classifier returns Denmark with high confidence
        mock_classifier.predict.return_value = ("DK", 0.85)
        
        result = classify_origins.classify_name("Anna")
        
        assert result == ("Scandinavia", 0.85)
        mock_classifier.predict.assert_called_once_with(["Anna"])
    
    def test_classify_name_unknown_region(self, mock_classifier, initialized_db):
        """Test classification with unknown nationality."""
        mock_classifier.predict.return_value = ("XX", 0.75)
        
        result = classify_origins.classify_name("UnknownName")
        
        assert result == ("Unknown", 0.375)  # 0.75 * 0.5 penalty
    
    def test_classify_name_import_error(self):
        """Test when name2nat is not installed."""
        with patch('st_name_ranking.classify_origins.get_classifier', side_effect=ImportError):
            result = classify_origins.classify_name("Anna")
            assert result == ("Unknown", 0.0)
    
    def test_classify_name_exception(self, mock_classifier):
        """Test handling of classifier exceptions."""
        mock_classifier.predict.side_effect = Exception("Classifier error")
        
        result = classify_origins.classify_name("Anna")
        assert result == ("Unknown", 0.0)


class TestClassifyNames:
    """Tests for classify_names function (batch classification)."""
    
    def test_classify_names_batch(self, mock_classifier, initialized_db):
        """Test batch classification of multiple names."""
        # Setup region mapping
        from st_name_ranking.database import get_connection
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO region_mapping (nationality, region) VALUES (?, ?)",
                [("DK", "Scandinavia"), ("SE", "Scandinavia")]
            )
        
        # Mock classifier predictions
        mock_classifier.predict.side_effect = [
            ("DK", 0.85),  # Anna
            ("SE", 0.78),  # Peter
            ("DK", 0.92),  # Maria
        ]
        
        names = ["Anna", "Peter", "Maria"]
        results = classify_origins.classify_names(names)
        
        assert len(results) == 3
        assert results[0] == ("Scandinavia", 0.85)
        assert results[1] == ("Scandinavia", 0.78)
        assert results[2] == ("Scandinavia", 0.92)
        mock_classifier.predict.assert_called_once_with(names)
    
    def test_classify_names_empty(self, mock_classifier):
        """Test batch classification with empty list."""
        results = classify_origins.classify_names([])
        assert results == []
        mock_classifier.predict.assert_not_called()
    
    def test_classify_names_partial_failure(self, mock_classifier, initialized_db):
        """Test when some classifications fail."""
        from st_name_ranking.database import get_connection
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO region_mapping (nationality, region) VALUES (?, ?)",
                ("DK", "Scandinavia")
            )
        
        # First classification succeeds, second fails
        mock_classifier.predict.side_effect = [
            ("DK", 0.85),
            Exception("Classifier error"),
            ("DK", 0.92),
        ]
        
        names = ["Anna", "ErrorName", "Maria"]
        results = classify_origins.classify_names(names)
        
        # Should return results for successful classifications
        assert len(results) == 2
        assert results[0] == ("Scandinavia", 0.85)
        assert results[1] == ("Scandinavia", 0.92)


class TestClassifyAllNames:
    """Tests for classify_all_names function."""
    
    @patch('st_name_ranking.classify_origins.classify_names')
    @patch('st_name_ranking.classify_origins.get_unclassified_names')
    @patch('st_name_ranking.classify_origins.update_name_origin')
    @patch('st_name_ranking.classify_origins.st', new_callable=MagicMock)
    def test_classify_all_names_success(
        self, mock_st, mock_update, mock_get_unclassified, mock_classify_names
    ):
        """Test successful classification of all unclassified names."""
        mock_get_unclassified.return_value = ["Anna", "Peter", "Maria"]
        mock_classify_names.return_value = [
            ("Scandinavia", 0.85),
            ("Scandinavia", 0.78),
            ("Unknown", 0.5),
        ]
        
        result = classify_origins.classify_all_names(limit=2)
        
        # Should only process up to limit
        mock_get_unclassified.assert_called_once_with(limit=2)
        mock_classify_names.assert_called_once_with(["Anna", "Peter", "Maria"])
        
        # Should update database for each name
        assert mock_update.call_count == 3
        mock_update.assert_any_call("Anna", "Scandinavia", 0.85)
        mock_update.assert_any_call("Peter", "Scandinavia", 0.78)
        mock_update.assert_any_call("Maria", "Unknown", 0.5)
        
        # Should show progress
        assert mock_st.progress.called
        assert mock_st.toast.called
        
        assert result == 3
    
    @patch('st_name_ranking.classify_origins.get_unclassified_names')
    @patch('st_name_ranking.classify_origins.st', new_callable=MagicMock)
    def test_classify_all_names_no_unclassified(
        self, mock_st, mock_get_unclassified
    ):
        """Test when no unclassified names exist."""
        mock_get_unclassified.return_value = []
        
        result = classify_origins.classify_all_names()
        
        assert result == 0
        mock_st.toast.assert_called_with(
            "No unclassified names found",
            icon="ℹ️",
        )
    
    @patch('st_name_ranking.classify_origins.get_unclassified_names')
    @patch('st_name_ranking.classify_origins.st', new_callable=MagicMock)
    def test_classify_all_names_import_error(
        self, mock_st, mock_get_unclassified
    ):
        """Test when name2nat is not installed."""
        mock_get_unclassified.return_value = ["Anna", "Peter"]
        
        with patch('st_name_ranking.classify_origins.get_classifier', side_effect=ImportError):
            result = classify_origins.classify_all_names()
            
            assert result == 0
            mock_st.toast.assert_called_with(
                "name2nat not installed. Install with: pip install name2nat",
                icon="❌",
                duration="long",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
