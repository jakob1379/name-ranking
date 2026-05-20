"""Tests for st_name_ranking.data_loader module."""

import sqlite3
from unittest.mock import patch

import pytest

from st_name_ranking import data_loader


class TestIsValidName:
    """Tests for is_valid_name function."""

    def test_valid_names(self):
        """Test that valid names return True."""
        valid_names = ["Anna", "Peter", "Maria", "Jens", "Lars", "Ida", "Emma"]
        for name in valid_names:
            assert data_loader.is_valid_name(name) is True

    def test_invalid_empty_or_none(self):
        """Test empty or None names return False."""
        assert data_loader.is_valid_name("") is False
        assert data_loader.is_valid_name(None) is False  # type: ignore[arg-type]
        assert data_loader.is_valid_name(" ") is False
        assert data_loader.is_valid_name("  ") is False

    def test_invalid_headers(self):
        """Test header/placeholder names return False."""
        invalid_names = [
            "name",
            "Name",
            "NAME",
            "navn",
            "Navn",
            "NAVN",
            "fornavn",
            "Fornavn",
            "FORNAVN",
            "firstname",
            "FirstName",
            "FIRSTNAME",
            "køn",
            "KØN",
            "Køn",
            "gender",
            "Gender",
            "GENDER",
            "kjønn",
            "Kjønn",
            "KJØNN",
        ]
        for name in invalid_names:
            assert data_loader.is_valid_name(name) is False

    def test_invalid_patterns(self):
        """Test pattern matches like 'name1', 'navn2', etc."""
        invalid_patterns = [
            "name1",
            "name 1",
            "name123",
            "navn1",
            "navn 2",
            "navn456",
            "fornavn1",
            "fornavn 3",
            "fornavn789",
        ]
        for name in invalid_patterns:
            assert data_loader.is_valid_name(name) is False

    def test_too_short(self):
        """Test names with less than 2 characters."""
        assert data_loader.is_valid_name("A") is False
        assert data_loader.is_valid_name("") is False
        assert data_loader.is_valid_name(" ") is False
        assert data_loader.is_valid_name("a") is False

    def test_valid_with_special_characters(self):
        """Test valid names with special characters (should be valid)."""
        # Assuming names with hyphens, spaces are valid after stripping
        assert data_loader.is_valid_name("Anne-Marie") is True
        assert data_loader.is_valid_name("Hans Peter") is True
        assert data_loader.is_valid_name("Bjørn") is True  # Norwegian letter


class TestStripNameNotes:
    """Tests for strip_name_notes function."""

    def test_strips_variant_notes(self):
        """Names with notes should keep only the base name."""
        raw = "Matteos - variant af godkendt fornavn"
        assert data_loader.strip_name_notes(raw) == "Matteos"

    def test_keeps_plain_name(self):
        """Plain names should be unchanged except trimming."""
        assert data_loader.strip_name_notes("  Anna  ") == "Anna"


class TestLoadRatings:
    """Tests for load_ratings function."""

    def test_load_ratings_empty_database(self, initialized_db):
        """Test loading ratings from empty database."""
        ratings = data_loader.load_ratings()
        assert ratings is not None
        assert isinstance(ratings, dict)
        assert len(ratings) == 0

    def test_load_ratings_with_data(self, initialized_db):
        """Test loading ratings when database has ratings."""
        from st_name_ranking.database import get_connection, update_rating

        # Insert a name and rating
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                ("Anna", "Female"),
            )

        update_rating("Anna", 1600.0)

        # Load ratings
        ratings = data_loader.load_ratings()
        assert ratings is not None
        assert "Anna" in ratings
        assert ratings["Anna"] == 1600.0

    @patch("st_name_ranking.data_loader.database.init_database")
    def test_load_ratings_database_error(self, mock_init):
        """Test load_ratings reports database errors distinctly."""
        mock_init.side_effect = sqlite3.Error("Database error")
        with pytest.raises(data_loader.DatabaseLoadError, match="Could not load ratings"):
            data_loader.load_ratings()


class TestSaveRatings:
    """Tests for save_ratings function."""

    def test_save_ratings_new_name(self, initialized_db):
        """Test saving ratings for a new name."""
        from st_name_ranking.database import get_connection

        # First ensure name exists
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                ("Peter", "Male"),
            )

        # Save rating
        data_loader.save_ratings({"Peter": 1550.0})

        # Verify rating saved
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT r.rating FROM ratings r
                JOIN names n ON r.name_id = n.id
                WHERE n.name = ?
            """,
                ("Peter",),
            )
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == 1550.0

    def test_save_ratings_multiple_names(self, initialized_db):
        """Test saving ratings for multiple names."""
        from st_name_ranking.database import get_connection

        # Insert names
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [("Anna", "Female"), ("Peter", "Male")],
            )

        # Save ratings
        data_loader.save_ratings({"Anna": 1700.0, "Peter": 1550.0})

        # Verify ratings
        ratings = data_loader.load_ratings()
        assert ratings is not None
        assert ratings["Anna"] == 1700.0
        assert ratings["Peter"] == 1550.0


class TestInitializeOrLoadRatings:
    """Tests for initialize_or_load_ratings function."""

    def test_initialize_new_names(self, initialized_db):
        """Test initializing ratings for new names."""
        names = ["Anna", "Peter", "Maria"]

        ratings = data_loader.initialize_or_load_ratings(names)

        assert len(ratings) == 3
        for name in names:
            assert name in ratings
            assert ratings[name] == 1500.0  # Default preference score

    def test_load_existing_ratings(self, initialized_db):
        """Test loading existing ratings without reinitializing."""
        from st_name_ranking.database import get_connection, update_rating

        # Insert names and set ratings
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [("Anna", "Female"), ("Peter", "Male")],
            )

        update_rating("Anna", 1700.0)
        update_rating("Peter", 1550.0)

        # Initialize or load should return existing ratings
        names = ["Anna", "Peter", "Maria"]  # Maria not in DB yet
        ratings = data_loader.initialize_or_load_ratings(names)

        assert ratings["Anna"] == 1700.0
        assert ratings["Peter"] == 1550.0
        assert ratings["Maria"] == 1500.0  # New name gets default


# TODO: Fix load_submodule_json tests - mocking polars DataFrame is complex
# class TestLoadSubmoduleJson:
#     """Tests for load_submodule_json function."""
#     pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
