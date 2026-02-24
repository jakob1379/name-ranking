"""Integration tests for data_loader module."""

from unittest.mock import MagicMock, patch

from st_name_ranking import data_loader


class TestDataLoaderIntegration:
    """Integration tests for data_loader module."""

    def test_load_ratings_from_database(self, initialized_db):
        """Test loading ratings from database."""
        from st_name_ranking.database import get_connection

        # Insert test names and ratings into database
        with get_connection() as conn:
            # Insert names
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [("Anna", "Female"), ("Peter", "Male"), ("Maria", "Female")],
            )
            # Insert ratings
            conn.executemany(
                """INSERT OR REPLACE INTO ratings (name_id, rating)
                   SELECT id, ? FROM names WHERE name = ?""",
                [(1600.0, "Anna"), (1550.0, "Peter"), (1500.0, "Maria")],
            )

        # Load ratings
        ratings = data_loader.load_ratings()

        # Verify ratings loaded correctly
        assert ratings is not None
        assert len(ratings) == 3
        assert ratings["Anna"] == 1600.0
        assert ratings["Peter"] == 1550.0
        assert ratings["Maria"] == 1500.0

    def test_save_ratings_to_database(self, initialized_db):
        """Test saving ratings to database."""
        from st_name_ranking.database import get_connection

        # Insert test names first
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [("Anna", "Female"), ("Peter", "Male")],
            )

        # Save ratings
        ratings = {"Anna": 1700.0, "Peter": 1650.0}
        result = data_loader.save_ratings(ratings)

        # Should return True on success
        assert result is True

        # Verify ratings were saved
        with get_connection() as conn:
            cursor = conn.execute(
                """SELECT n.name, r.rating FROM ratings r
                   JOIN names n ON r.name_id = n.id
                   ORDER BY n.name""",
            )
            saved_ratings = {row[0]: row[1] for row in cursor.fetchall()}
            assert saved_ratings == ratings

    def test_initialize_or_load_ratings(self, initialized_db):
        """Test initialize_or_load_ratings function."""
        from st_name_ranking.database import get_connection

        # Insert some names with existing ratings
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [("Anna", "Female"), ("Peter", "Male"), ("Maria", "Female")],
            )
            # Set rating for Anna only
            conn.execute(
                """INSERT OR REPLACE INTO ratings (name_id, rating)
                   SELECT id, 1600.0 FROM names WHERE name = ?""",
                ("Anna",),
            )

        # Test with names list (Anna has rating, Peter has rating, Maria no rating, NewName not in db)
        names = ["Anna", "Peter", "Maria", "NewName"]
        ratings = data_loader.initialize_or_load_ratings(names)

        # Should have ratings for all names
        assert len(ratings) == 4
        # Anna should keep existing rating
        assert ratings["Anna"] == 1600.0
        # Peter and Maria should get default 1500.0 (since Peter has no rating in db)
        assert ratings["Peter"] == 1500.0
        assert ratings["Maria"] == 1500.0
        # NewName should get default 1500.0
        assert ratings["NewName"] == 1500.0

    def test_load_names_by_gender_with_submodule(self, initialized_db):
        """Test loading names by gender from database after inserting names."""
        from st_name_ranking.database import get_connection

        # Insert test names into database
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [
                    ("Anna", "Female"),
                    ("Peter", "Male"),
                    ("Maria", "Female"),
                    ("Alex", "Unisex"),
                ],
            )

        # Test loading with sync_with_submodule=False (use existing database)
        gender_data = data_loader.load_names_by_gender(sync_with_submodule=False)

        # Should return gender-organized lists
        assert gender_data is not None
        assert "All" in gender_data
        assert "Female" in gender_data
        assert "Male" in gender_data
        assert "Unisex" in gender_data

        # Check counts
        assert len(gender_data["All"]) == 4
        assert len(gender_data["Female"]) == 3  # Anna, Maria, Alex (unisex)
        assert len(gender_data["Male"]) == 2  # Peter, Alex (unisex)
        assert len(gender_data["Unisex"]) == 1  # Alex only

        # Verify names are in correct groups
        assert "Anna" in gender_data["Female"]
        assert "Peter" in gender_data["Male"]
        assert "Maria" in gender_data["Female"]
        assert "Alex" in gender_data["Unisex"]
        # Unisex name appears in both Male and Female
        assert "Alex" in gender_data["Female"]
        assert "Alex" in gender_data["Male"]

    def test_load_names_by_gender_sync_with_submodule(self, initialized_db, tmp_path):
        """Test loading names by gender with sync to submodule."""
        from st_name_ranking.database import get_connection

        # Insert test names into database (simulate sync result)
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [
                    ("NewName1", "Female"),
                    ("NewName2", "Male"),
                ],
            )

        with patch("st_name_ranking.database.sync_names_with_submodule") as mock_sync:
            mock_sync.return_value = 2  # Simulate 2 names synced

            gender_data = data_loader.load_names_by_gender(sync_with_submodule=True)

            # Should have called sync (with default path)
            mock_sync.assert_called_once_with()

            # Should return gender data with our inserted names
            assert gender_data is not None
            assert "All" in gender_data
            assert "Female" in gender_data
            assert "Male" in gender_data
            assert "Unisex" in gender_data
            assert len(gender_data["All"]) == 2
            assert len(gender_data["Female"]) == 1
            assert len(gender_data["Male"]) == 1
            assert len(gender_data["Unisex"]) == 0
            # Verify names are present
            assert "NewName1" in gender_data["Female"]
            assert "NewName2" in gender_data["Male"]

    def test_is_valid_name(self):
        """Test name validation function."""
        # Valid names
        assert data_loader.is_valid_name("Anna") is True
        assert data_loader.is_valid_name("Peter") is True
        assert data_loader.is_valid_name("Maria") is True

        # Invalid names (too short, empty, None)
        assert data_loader.is_valid_name("") is False
        assert data_loader.is_valid_name("A") is False
        assert data_loader.is_valid_name("Ab") is True
        # Note: is_valid_name expects str, not None

        # Names with special characters (should be valid after cleaning)
        assert data_loader.is_valid_name("Anna-Marie") is True
        assert data_loader.is_valid_name("Jørgen") is True

    def test_load_submodule_json_not_found(self, tmp_path):
        """Test loading submodule JSON when file doesn't exist."""
        import os

        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        # No JSON file created

        mock_st = MagicMock()
        mock_st.toast = MagicMock()

        # Save real os.path.join before patching
        real_join = os.path.join

        with (
            patch("st_name_ranking.data_loader.st", mock_st),
            patch("st_name_ranking.data_loader.os.path.join") as mock_join,
        ):
            # Make os.path.join return our temp file path
            def join_side_effect(*args):
                if args[0] == "godkendtefornavne":
                    return str(submodule_path / args[1])
                return real_join(*args)

            mock_join.side_effect = join_side_effect

            result = data_loader.load_submodule_json()
            # Should return empty list when file not found
            assert result == []

    def test_load_submodule_csv_fallback(self, tmp_path):
        """Test CSV fallback loading."""
        import os

        submodule_path = tmp_path / "godkendtefornavne"
        submodule_path.mkdir()
        # Create the three expected CSV files with names
        drengenavne_content = """Peter
Hans
Jens"""
        pigenavne_content = """Anna
Maria
Ida"""
        unisexnavne_content = """Alex
Kim
Robin"""

        (submodule_path / "drengenavne.csv").write_text(drengenavne_content)
        (submodule_path / "pigenavne.csv").write_text(pigenavne_content)
        (submodule_path / "unisexnavne.csv").write_text(unisexnavne_content)

        mock_st = MagicMock()
        mock_st.toast = MagicMock()

        # Save real os.path.join before patching
        real_join = os.path.join

        with (
            patch("st_name_ranking.data_loader.st", mock_st),
            patch("st_name_ranking.data_loader.os.path.join") as mock_join,
        ):
            # Make os.path.join return our temp file path
            def join_side_effect(*args):
                if args[0] == "godkendtefornavne":
                    return str(submodule_path / args[1])
                return real_join(*args)

            mock_join.side_effect = join_side_effect

            result = data_loader.load_submodule_csv_fallback()
            # Should parse CSV and return list of names
            assert len(result) == 9
            assert "Peter" in result
            assert "Hans" in result
            assert "Jens" in result
            assert "Anna" in result
            assert "Maria" in result
            assert "Ida" in result
            assert "Alex" in result
            assert "Kim" in result
            assert "Robin" in result
