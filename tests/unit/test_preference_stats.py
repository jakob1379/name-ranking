"""Integration tests for preference statistics aggregation functions."""


class TestPreferenceStatistics:
    """Tests for preference statistics aggregation functions."""

    def test_preference_stats_empty(self, initialized_db):
        """Test preference statistics when no comparisons exist."""
        from st_name_ranking.persistence.database import (
            get_preference_stats_by_gender,
            get_preference_stats_by_origin,
            get_preference_stats_by_phonetic,
        )

        # All functions should return empty dicts
        gender_stats = get_preference_stats_by_gender()
        origin_stats = get_preference_stats_by_origin()
        phonetic_stats = get_preference_stats_by_phonetic()

        assert gender_stats == {}
        assert origin_stats == {}
        assert phonetic_stats == {}

    def test_preference_stats_by_gender(self, initialized_db):
        """Test preference statistics grouped by gender."""
        from st_name_ranking.persistence.database import (
            get_connection,
            get_preference_stats_by_gender,
            get_preference_stats_by_origin,
            get_preference_stats_by_phonetic,
        )

        # Insert test names with different genders
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [
                    ("Anna", "Female"),
                    ("Peter", "Male"),
                    ("Alex", "Unisex"),
                    ("Unknown", None),  # Null gender
                ],
            )

            # Get name IDs
            cursor = conn.cursor()
            name_ids = {}
            for name in ["Anna", "Peter", "Alex", "Unknown"]:
                cursor.execute("SELECT id FROM names WHERE name = ?", (name,))
                name_ids[name] = cursor.fetchone()[0]

            # Insert comparisons with different preferences
            # Comparison 1: Anna (Female) vs Peter (Male), preference = -1 (Anna preferred)
            conn.execute(
                "INSERT INTO comparisons (name_a_id, name_b_id, preference) VALUES (?, ?, ?)",
                (name_ids["Anna"], name_ids["Peter"], -1),
            )
            # Comparison 2: Alex (Unisex) vs Unknown (null), preference = 0 (draw)
            conn.execute(
                "INSERT INTO comparisons (name_a_id, name_b_id, preference) VALUES (?, ?, ?)",
                (name_ids["Alex"], name_ids["Unknown"], 0),
            )
            # Comparison 3: Peter (Male) vs Anna (Female), preference = 1 (Peter preferred)
            conn.execute(
                "INSERT INTO comparisons (name_a_id, name_b_id, preference) VALUES (?, ?, ?)",
                (name_ids["Peter"], name_ids["Anna"], 1),
            )

        # Get statistics
        gender_stats = get_preference_stats_by_gender()

        # Verify expected counts
        # Female: Anna appears in 2 comparisons:
        #   - Comparison 1: Anna wins (preference -1)
        #   - Comparison 3: Anna wins (preference 1) - Anna is name_b, preference 1 means name_b wins
        # So Female: wins=2, losses=0, draws=0, total=2
        assert "Female" in gender_stats
        assert gender_stats["Female"].wins == 2
        assert gender_stats["Female"].losses == 0
        assert gender_stats["Female"].draws == 0
        assert gender_stats["Female"].total == 2

        # Male: Peter appears in 2 comparisons:
        #   - Comparison 1: Peter loses (preference -1)
        #   - Comparison 3: Peter loses (preference 1) - Peter is name_a, preference 1 means name_a loses
        # So Male: wins=0, losses=2, draws=0, total=2
        assert "Male" in gender_stats
        assert gender_stats["Male"].wins == 0
        assert gender_stats["Male"].losses == 2
        assert gender_stats["Male"].draws == 0
        assert gender_stats["Male"].total == 2

        # Unisex: Alex appears in 1 comparison (draw)
        #   - Comparison 2: draw (preference 0)
        # So Unisex: wins=0, losses=0, draws=1, total=1
        assert "Unisex" in gender_stats
        assert gender_stats["Unisex"].wins == 0
        assert gender_stats["Unisex"].losses == 0
        assert gender_stats["Unisex"].draws == 1
        assert gender_stats["Unisex"].total == 1

        # Unknown (null gender) should be grouped as 'Unknown' per COALESCE
        # Unknown appears in 1 comparison (draw)
        assert "Unknown" in gender_stats
        assert gender_stats["Unknown"].wins == 0
        assert gender_stats["Unknown"].losses == 0
        assert gender_stats["Unknown"].draws == 1
        assert gender_stats["Unknown"].total == 1

        # Origin stats should have only 'International' (null origin)
        origin_stats = get_preference_stats_by_origin()
        assert "International" in origin_stats
        assert origin_stats["International"].wins == 2
        assert origin_stats["International"].losses == 2
        assert origin_stats["International"].draws == 2
        assert origin_stats["International"].total == 6

        # Phonetic stats should have only 'Unknown' (null phonetic codes)
        phonetic_stats = get_preference_stats_by_phonetic()
        assert "Unknown" in phonetic_stats
        assert phonetic_stats["Unknown"].wins == 2
        assert phonetic_stats["Unknown"].losses == 2
        assert phonetic_stats["Unknown"].draws == 2
        assert phonetic_stats["Unknown"].total == 6

    def test_preference_stats_by_origin(self, initialized_db):
        """Test preference statistics grouped by origin region."""
        from st_name_ranking.persistence.database import (
            get_connection,
            get_preference_stats_by_origin,
        )

        # Insert test names with different origin regions
        with get_connection() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO names
                   (name, gender, origin_region) VALUES (?, ?, ?)""",
                [
                    ("Anna", "Female", "Nordic"),
                    ("Peter", "Male", "European"),
                    ("Alex", "Unisex", "Asian"),
                    ("Unknown", None, None),  # Null origin
                ],
            )

            # Get name IDs
            cursor = conn.cursor()
            name_ids = {}
            for name in ["Anna", "Peter", "Alex", "Unknown"]:
                cursor.execute("SELECT id FROM names WHERE name = ?", (name,))
                name_ids[name] = cursor.fetchone()[0]

            # Insert comparisons
            # Anna (Nordic) vs Peter (European), preference = -1 (Anna preferred)
            conn.execute(
                "INSERT INTO comparisons (name_a_id, name_b_id, preference) VALUES (?, ?, ?)",
                (name_ids["Anna"], name_ids["Peter"], -1),
            )
            # Alex (Asian) vs Unknown (null), preference = 1 (Unknown preferred)
            conn.execute(
                "INSERT INTO comparisons (name_a_id, name_b_id, preference) VALUES (?, ?, ?)",
                (name_ids["Alex"], name_ids["Unknown"], 1),
            )

        # Get statistics
        origin_stats = get_preference_stats_by_origin()

        # Verify expected counts
        # Nordic: Anna appears in 1 comparison, wins (preference -1)
        assert "Nordic" in origin_stats
        assert origin_stats["Nordic"].wins == 1
        assert origin_stats["Nordic"].losses == 0
        assert origin_stats["Nordic"].draws == 0
        assert origin_stats["Nordic"].total == 1

        # European: Peter appears in 1 comparison, loses (preference -1)
        assert "European" in origin_stats
        assert origin_stats["European"].wins == 0
        assert origin_stats["European"].losses == 1
        assert origin_stats["European"].draws == 0
        assert origin_stats["European"].total == 1

        # Asian: Alex appears in 1 comparison, loses (preference 1)
        assert "Asian" in origin_stats
        assert origin_stats["Asian"].wins == 0
        assert origin_stats["Asian"].losses == 1
        assert origin_stats["Asian"].draws == 0
        assert origin_stats["Asian"].total == 1

        # International (null origin): Unknown appears in 1 comparison, wins (preference 1)
        assert "International" in origin_stats
        assert origin_stats["International"].wins == 1
        assert origin_stats["International"].losses == 0
        assert origin_stats["International"].draws == 0
        assert origin_stats["International"].total == 1

    def test_preference_stats_by_phonetic(self, initialized_db):
        """Test preference statistics grouped by phonetic code."""
        from st_name_ranking.persistence.database import (
            get_connection,
            get_preference_stats_by_phonetic,
        )

        # Insert test names with different phonetic codes
        with get_connection() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO names
                   (name, gender, phonetic_primary) VALUES (?, ?, ?)""",
                [
                    ("Anna", "Female", "AN"),
                    ("Peter", "Male", "PTR"),
                    ("Alex", "Unisex", "ALKS"),
                    ("Unknown", None, None),  # Null phonetic
                ],
            )

            # Get name IDs
            cursor = conn.cursor()
            name_ids = {}
            for name in ["Anna", "Peter", "Alex", "Unknown"]:
                cursor.execute("SELECT id FROM names WHERE name = ?", (name,))
                name_ids[name] = cursor.fetchone()[0]

            # Insert comparisons
            # Anna (AN) vs Peter (PTR), preference = -1 (Anna preferred)
            conn.execute(
                "INSERT INTO comparisons (name_a_id, name_b_id, preference) VALUES (?, ?, ?)",
                (name_ids["Anna"], name_ids["Peter"], -1),
            )
            # Alex (ALKS) vs Unknown (null), preference = 0 (draw)
            conn.execute(
                "INSERT INTO comparisons (name_a_id, name_b_id, preference) VALUES (?, ?, ?)",
                (name_ids["Alex"], name_ids["Unknown"], 0),
            )

        # Get statistics
        phonetic_stats = get_preference_stats_by_phonetic()

        # Verify expected counts
        # AN: Anna appears in 1 comparison, wins (preference -1)
        assert "AN" in phonetic_stats
        assert phonetic_stats["AN"].wins == 1
        assert phonetic_stats["AN"].losses == 0
        assert phonetic_stats["AN"].draws == 0
        assert phonetic_stats["AN"].total == 1

        # PTR: Peter appears in 1 comparison, loses (preference -1)
        assert "PTR" in phonetic_stats
        assert phonetic_stats["PTR"].wins == 0
        assert phonetic_stats["PTR"].losses == 1
        assert phonetic_stats["PTR"].draws == 0
        assert phonetic_stats["PTR"].total == 1

        # ALKS: Alex appears in 1 comparison, draw (preference 0)
        assert "ALKS" in phonetic_stats
        assert phonetic_stats["ALKS"].wins == 0
        assert phonetic_stats["ALKS"].losses == 0
        assert phonetic_stats["ALKS"].draws == 1
        assert phonetic_stats["ALKS"].total == 1

        # Unknown (null phonetic): Unknown appears in 1 comparison, draw (preference 0)
        assert "Unknown" in phonetic_stats
        assert phonetic_stats["Unknown"].wins == 0
        assert phonetic_stats["Unknown"].losses == 0
        assert phonetic_stats["Unknown"].draws == 1
        assert phonetic_stats["Unknown"].total == 1

    def test_preference_stats_multiple_comparisons_same_name(self, initialized_db):
        """Test preference statistics when a name appears in multiple comparisons."""
        from st_name_ranking.persistence.database import (
            get_connection,
            get_preference_stats_by_gender,
        )

        # Insert test names
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                [
                    ("Anna", "Female"),
                    ("Peter", "Male"),
                    ("Maria", "Female"),
                ],
            )

            # Get name IDs
            cursor = conn.cursor()
            name_ids = {}
            for name in ["Anna", "Peter", "Maria"]:
                cursor.execute("SELECT id FROM names WHERE name = ?", (name,))
                name_ids[name] = cursor.fetchone()[0]

            # Insert multiple comparisons involving Anna
            # Anna vs Peter: Anna wins (-1)
            conn.execute(
                "INSERT INTO comparisons (name_a_id, name_b_id, preference) VALUES (?, ?, ?)",
                (name_ids["Anna"], name_ids["Peter"], -1),
            )
            # Anna vs Maria: Anna loses (1) [Maria preferred]
            conn.execute(
                "INSERT INTO comparisons (name_a_id, name_b_id, preference) VALUES (?, ?, ?)",
                (name_ids["Anna"], name_ids["Maria"], 1),
            )
            # Peter vs Maria: draw (0)
            conn.execute(
                "INSERT INTO comparisons (name_a_id, name_b_id, preference) VALUES (?, ?, ?)",
                (name_ids["Peter"], name_ids["Maria"], 0),
            )

        # Get statistics
        gender_stats = get_preference_stats_by_gender()

        # Female: Anna appears in 2 comparisons (win, loss), Maria appears in 2 comparisons (win, draw)
        # Total female outcomes: wins=2 (Anna win + Maria win), losses=1 (Anna loss), draws=1 (Maria draw)
        assert "Female" in gender_stats
        assert gender_stats["Female"].wins == 2
        assert gender_stats["Female"].losses == 1
        assert gender_stats["Female"].draws == 1
        assert gender_stats["Female"].total == 4

        # Male: Peter appears in 2 comparisons (loss, draw)
        # Male outcomes: wins=0, losses=1 (Peter loss), draws=1 (Peter draw)
        assert "Male" in gender_stats
        assert gender_stats["Male"].wins == 0
        assert gender_stats["Male"].losses == 1
        assert gender_stats["Male"].draws == 1
        assert gender_stats["Male"].total == 2
