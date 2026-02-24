"""Integration tests for classify_origins module."""

from unittest.mock import MagicMock, patch

from st_name_ranking import classify_origins


class TestClassifyAllNamesIntegration:
    """Integration tests for classify_all_names function."""

    def test_classify_all_names_with_mocked_classifier(self, initialized_db):
        """Test classifying names with mocked ethnidata classifier."""
        from st_name_ranking.database import get_connection

        # Ensure test name exists in database, unclassified
        with get_connection() as conn:
            # Delete if exists to start fresh
            conn.execute("DELETE FROM names WHERE name = ?", ("TestName",))
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("TestName", "Unisex"),
            )
            cursor = conn.execute("SELECT id FROM names WHERE name = ?", ("TestName",))
            row = cursor.fetchone()
            assert row is not None
            name_id = row[0]

        # Mock the origin classifier to return a region
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = ("Nordic", 0.9)

        with patch("st_name_ranking.classify_origins.get_origin_classifier") as mock_get:
            mock_get.return_value = mock_classifier

            # Call classify_all_names with limit=1
            result = classify_origins.classify_all_names(limit=1)

            # Should have classified 1 name
            assert result == 1

            # Verify region was updated in database
            with get_connection() as conn:
                cursor = conn.execute(
                    "SELECT origin_region, origin_confidence FROM names WHERE id = ?",
                    (name_id,),
                )
                row = cursor.fetchone()
                assert row is not None
                assert row[0] == "Nordic"
                assert row[1] == 0.9

    def test_classify_all_names_no_unclassified(self, initialized_db):
        """Test when there are no unclassified names."""
        # Ensure all names are already classified (or no names)
        result = classify_origins.classify_all_names()
        assert result == 0

    def test_classify_all_names_import_error(self, initialized_db):
        """Test when ethnidata is not installed."""
        # Mock ImportError when importing ethnidata
        with patch("ethnidata.EthniData", side_effect=ImportError):
            # Should still work (ethnidata is optional)
            result = classify_origins.classify_all_names(limit=1)
            # May classify 0 names because ethnidata not available, but other classifiers may work
            # For now, just ensure no exception
            assert result >= 0
