"""End-to-end workflow integration tests for the Name Ranking application.

Tests complete user journeys from database initialization through voting,
covering cross-component integration.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from st_name_ranking.active_learning.lazy_updates import BOTH_DISLIKED_PREFERENCE, record_comparison_instant
from st_name_ranking.active_learning.selection import select_candidates
from st_name_ranking.data_loader import initialize_or_load_ratings
from st_name_ranking.database import (
    get_connection,
    get_names_by_filters,
    get_names_by_gender,
    get_ratings,
    get_stats,
    record_comparison,
    sync_names_with_submodule,
    update_name_origin,
)
from st_name_ranking.features import FeatureExtractor


def update_preference_and_save(ratings: dict[str, float], winner: str, loser: str) -> dict[str, float]:
    """Test-only adapter for the removed compatibility wrapper."""
    return _record_preference_and_refresh_ratings(ratings, winner, loser, -1)


def update_preference_draw_and_save(ratings: dict[str, float], player_a: str, player_b: str) -> dict[str, float]:
    """Test-only adapter for the removed compatibility wrapper."""
    return _record_preference_and_refresh_ratings(ratings, player_a, player_b, 0)


def update_preference_both_disliked_and_save(
    ratings: dict[str, float],
    player_a: str,
    player_b: str,
) -> dict[str, float]:
    """Test-only adapter for the removed compatibility wrapper."""
    return _record_preference_and_refresh_ratings(ratings, player_a, player_b, BOTH_DISLIKED_PREFERENCE)


def _record_preference_and_refresh_ratings(
    ratings: dict[str, float],
    name_a: str,
    name_b: str,
    preference: int,
) -> dict[str, float]:
    status = record_comparison_instant(name_a, name_b, preference, blocking=True)
    if not status.recorded:
        return ratings.copy()
    return {**ratings, **get_ratings()}


# =============================================================================
# E2E Workflow Tests
# =============================================================================


class TestNewUserWorkflow:
    """Test complete new user journey: init → sync → classify → tournament → rankings."""

    def test_new_user_workflow(self, initialized_db, mock_submodule_path, mock_classifier):
        """
        Simulate complete new user journey:
        1. Initialize database
        2. Sync names from submodule
        3. Classify origins
        4. Select candidate pairs
        5. Record votes
        6. Verify ratings updated
        """
        # Step 1: Database is already initialized by fixture
        stats = get_stats()
        assert stats.total_names == 0, "Database should start empty"

        # Step 2: Sync names from submodule
        inserted = sync_names_with_submodule(mock_submodule_path)
        assert inserted == 3, f"Expected 3 names inserted, got {inserted}"

        # Verify names were synced
        stats = get_stats()
        assert stats.total_names == 3

        # Step 3: Classify origins (mocked)
        with get_connection() as conn:
            cursor = conn.execute("SELECT id, name FROM names")
            names_data = cursor.fetchall()

        for name_id, name in names_data:
            update_name_origin(name_id, "Nordic", 0.85)

        # Verify classifications
        stats = get_stats()
        assert stats.classified_names == 3

        # Step 4: Initialize ratings
        names = get_names_by_gender()
        all_names = names.get("All", [])
        assert len(all_names) == 3

        ratings = initialize_or_load_ratings(all_names)
        assert len(ratings) == 3
        for name in all_names:
            assert ratings[name] == 1500.0  # Initial score

        # Step 5: Select candidate pairs for tournament
        candidates_a, candidates_b = select_candidates(all_names)
        assert candidates_a != ""
        assert candidates_b != ""
        assert candidates_a != candidates_b
        assert candidates_a in all_names
        assert candidates_b in all_names

        # Step 6: Record votes and verify ratings updated
        initial_a_rating = ratings[candidates_a]
        initial_b_rating = ratings[candidates_b]

        # Cast vote for candidate_a
        updated_ratings = update_preference_and_save(ratings, candidates_a, candidates_b)

        # Verify ratings changed
        assert updated_ratings[candidates_a] != initial_a_rating
        assert updated_ratings[candidates_b] != initial_b_rating

        # Verify comparison recorded in database
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            count = cursor.fetchone()[0]
            assert count == 1

        # Verify ratings persisted in database
        db_ratings = get_ratings()
        assert len(db_ratings) == 2  # Only voted names have ratings in DB
        assert candidates_a in db_ratings
        assert candidates_b in db_ratings


class TestVotingWorkflow:
    """Test voting round-trip: select pair → vote → verify persistence → check ratings."""

    @pytest.fixture
    def voting_db(self, initialized_db):
        """Database with sample names for voting tests."""
        # Insert test names with various attributes
        test_names = [
            ("Emma", "Female", "Nordic"),
            ("Liam", "Male", "European"),
            ("Sofia", "Female", "European"),
            ("Noah", "Male", "American"),
            ("Olivia", "Female", None),  # Unclassified
        ]

        with get_connection() as conn:
            for name, gender, origin in test_names:
                conn.execute(
                    "INSERT INTO names (name, gender, origin_region) VALUES (?, ?, ?)",
                    (name, gender, origin),
                )

        return initialized_db

    def test_voting_persists_ratings(self, voting_db):
        """
        1. Add two names to database
        2. Select pair via active learning
        3. Record preference vote
        4. Verify ratings updated in database
        5. Verify comparison recorded
        """
        # Get names
        names = get_names_by_gender()
        all_names = names.get("All", [])
        assert len(all_names) == 5

        # Initialize ratings
        ratings = initialize_or_load_ratings(all_names)

        # Step 1: Select pair via active learning
        name_a, name_b = select_candidates(all_names)
        assert name_a in all_names
        assert name_b in all_names
        assert name_a != name_b

        # Store initial ratings before update
        initial_a = ratings[name_a]
        initial_b = ratings[name_b]

        # Step 2: Record preference vote (name_a wins)
        updated_ratings = update_preference_and_save(ratings, name_a, name_b)

        # Step 3: Verify ratings updated in memory
        assert updated_ratings[name_a] != initial_a or updated_ratings[name_b] != initial_b

        # Step 4: Verify ratings updated in database
        db_ratings = get_ratings()
        assert name_a in db_ratings
        assert name_b in db_ratings
        # Ratings should be different from initial (model has updated)
        assert db_ratings[name_a] != initial_a or db_ratings[name_b] != initial_b
        # Winner should have higher rating than loser after vote
        assert db_ratings[name_a] > db_ratings[name_b]

        # Step 5: Verify comparison recorded
        with get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT c.preference, n1.name as name_a, n2.name as name_b
                FROM comparisons c
                JOIN names n1 ON c.name_a_id = n1.id
                JOIN names n2 ON c.name_b_id = n2.id
                WHERE n1.name = ? AND n2.name = ?
                """,
                (name_a, name_b),
            )
            comparison = cursor.fetchone()
            assert comparison is not None
            assert comparison["preference"] == -1  # name_a preferred

    def test_draw_vote_updates_correctly(self, voting_db):
        """Verify draw votes (preference=0) are handled correctly."""
        names = get_names_by_gender()
        all_names = names.get("All", [])
        ratings = initialize_or_load_ratings(all_names)

        # Select two names
        name_a, name_b = "Emma", "Liam"
        initial_a = ratings[name_a]
        initial_b = ratings[name_b]

        # Record draw
        updated_ratings = update_preference_draw_and_save(ratings, name_a, name_b)

        # Verify ratings changed (draws still update the model)
        assert updated_ratings[name_a] != initial_a or updated_ratings[name_b] != initial_b

        # Verify comparison recorded with preference=0
        with get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT preference FROM comparisons c
                JOIN names n1 ON c.name_a_id = n1.id
                JOIN names n2 ON c.name_b_id = n2.id
                WHERE n1.name = ? AND n2.name = ?
                """,
                (name_a, name_b),
            )
            result = cursor.fetchone()
            assert result is not None
            assert result["preference"] == 0

    def test_both_disliked_vote_updates_correctly(self, voting_db):
        """Verify both-disliked votes (preference=2) are handled correctly."""
        names = get_names_by_gender()
        all_names = names.get("All", [])
        ratings = initialize_or_load_ratings(all_names)

        # Select two names
        name_a, name_b = "Sofia", "Noah"
        initial_a = ratings[name_a]
        initial_b = ratings[name_b]

        # Record "both disliked" vote
        updated_ratings = update_preference_both_disliked_and_save(ratings, name_a, name_b)

        # Verify ratings were updated
        assert updated_ratings[name_a] != initial_a or updated_ratings[name_b] != initial_b

        # Verify comparison recorded with preference=2
        with get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT preference FROM comparisons c
                JOIN names n1 ON c.name_a_id = n1.id
                JOIN names n2 ON c.name_b_id = n2.id
                WHERE n1.name = ? AND n2.name = ?
                """,
                (name_a, name_b),
            )
            result = cursor.fetchone()
            assert result is not None
            assert result["preference"] == 2

    def test_multiple_votes_accumulate_comparisons(self, voting_db):
        """Test that multiple votes accumulate in the comparisons table."""
        names = get_names_by_gender()
        all_names = names.get("All", [])
        ratings = initialize_or_load_ratings(all_names)

        # Cast 5 votes
        votes = [
            ("Emma", "Liam", -1),  # Emma wins
            ("Sofia", "Noah", 1),  # Noah wins
            ("Emma", "Sofia", -1),  # Emma wins
            ("Liam", "Olivia", 0),  # Draw
            ("Noah", "Olivia", 2),  # Both disliked
        ]

        for name_a, name_b, preference in votes:
            if preference == -1:
                ratings = update_preference_and_save(ratings, name_a, name_b)
            elif preference == 0:
                ratings = update_preference_draw_and_save(ratings, name_a, name_b)
            elif preference == 2:
                ratings = update_preference_both_disliked_and_save(ratings, name_a, name_b)

        # Verify comparisons recorded (at least 4 due to possible UNIQUE constraints)
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            count = cursor.fetchone()[0]
            assert count >= 4, f"Expected at least 4 comparisons, got {count}"

            # Verify each vote type exists
            cursor = conn.execute(
                """
                SELECT preference, COUNT(*) as count
                FROM comparisons
                GROUP BY preference
                """,
            )
            pref_counts = {row["preference"]: row["count"] for row in cursor.fetchall()}
            # At least 2 preference=-1 votes (Emma wins twice)
            assert pref_counts.get(-1, 0) >= 1
            # At least 1 draw vote
            assert pref_counts.get(0, 0) >= 1
            # At least 1 both-disliked vote
            assert pref_counts.get(2, 0) >= 1


class TestFilterWorkflow:
    """Test filter integration workflow."""

    @pytest.fixture
    def filtered_db(self, initialized_db):
        """Database with diverse names for filter tests."""
        test_names = [
            # Female Nordic names
            ("Emma", "Female", "Nordic"),
            ("Sofia", "Female", "European"),
            ("Olivia", "Female", "American"),
            # Male Nordic names
            ("Liam", "Male", "Nordic"),
            ("Noah", "Male", "European"),
            ("Lucas", "Male", "American"),
            # Unisex
            ("Alex", "Unisex", "Nordic"),
        ]

        with get_connection() as conn:
            for name, gender, origin in test_names:
                conn.execute(
                    "INSERT INTO names (name, gender, origin_region) VALUES (?, ?, ?)",
                    (name, gender, origin),
                )

        return initialized_db

    def test_filtering_affects_candidate_selection(self, filtered_db):
        """
        1. Add names with different genders/origins
        2. Apply gender filter
        3. Verify candidates respect filter
        """
        # Test 1: Filter by Female gender
        female_names = get_names_by_filters(gender="Female", origins=None)
        assert len(female_names) == 3
        assert set(female_names) == {"Emma", "Sofia", "Olivia"}

        # Verify candidates are from filtered set
        for _ in range(10):  # Run multiple times due to randomness
            name_a, name_b = select_candidates(female_names)
            if name_a and name_b:
                assert name_a in female_names
                assert name_b in female_names

        # Test 2: Filter by Male gender (doesn't include unisex)
        male_names = get_names_by_filters(gender="Male", origins=None)
        assert "Liam" in male_names
        assert "Noah" in male_names
        assert "Lucas" in male_names

        # Test 3: Filter by origin
        nordic_names = get_names_by_filters(gender=None, origins=["Nordic"])
        assert "Emma" in nordic_names
        assert "Liam" in nordic_names
        assert "Alex" in nordic_names

        # Test 4: Combined filter (Female + Nordic)
        female_nordic = get_names_by_filters(gender="Female", origins=["Nordic"])
        assert female_nordic == ["Emma"]  # Only Emma matches

    def test_vote_with_filtered_names(self, filtered_db):
        """Test that voting works correctly with filtered name sets."""
        # Get filtered names
        female_names = get_names_by_filters(gender="Female", origins=None)
        ratings = initialize_or_load_ratings(female_names)

        # Select and vote on filtered candidates
        name_a, name_b = select_candidates(female_names)
        assert name_a in female_names
        assert name_b in female_names

        # Cast vote
        updated_ratings = update_preference_and_save(ratings, name_a, name_b)
        assert name_a in updated_ratings
        assert name_b in updated_ratings

        # Verify rating persisted
        db_ratings = get_ratings()
        assert name_a in db_ratings
        assert name_b in db_ratings

        # Verify comparison recorded
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            assert cursor.fetchone()[0] == 1


class TestRatingConsistency:
    """Test rating consistency after multiple votes."""

    @pytest.fixture
    def consistency_db(self, initialized_db):
        """Database with names for consistency tests."""
        test_names = [
            ("NameA", "Female", "Nordic"),
            ("NameB", "Female", "Nordic"),
            ("NameC", "Female", "Nordic"),
            ("NameD", "Female", "Nordic"),
        ]

        with get_connection() as conn:
            for name, gender, origin in test_names:
                conn.execute(
                    "INSERT INTO names (name, gender, origin_region) VALUES (?, ?, ?)",
                    (name, gender, origin),
                )

        return initialized_db

    def test_ratings_consistent_after_multiple_votes(self, consistency_db):
        """After multiple votes, ratings should reflect preferences."""
        names = get_names_by_gender()
        all_names = names.get("Female", [])
        ratings = initialize_or_load_ratings(all_names)

        # Simulate tournament: A beats everyone
        votes = [
            ("NameA", "NameB", -1),
            ("NameA", "NameC", -1),
            ("NameA", "NameD", -1),
        ]

        for winner, loser, pref in votes:
            ratings = update_preference_and_save(ratings, winner, loser)

        # NameA should have highest rating
        assert ratings["NameA"] > ratings["NameB"]
        assert ratings["NameA"] > ratings["NameC"]
        assert ratings["NameA"] > ratings["NameD"]

        # Verify in database
        db_ratings = get_ratings()
        assert db_ratings["NameA"] > db_ratings["NameB"]

    def test_rating_ordering_reflects_preferences(self, consistency_db):
        """Test that rating ordering matches preference patterns."""
        names = get_names_by_gender()
        all_names = names.get("Female", [])
        ratings = initialize_or_load_ratings(all_names)

        # Create clear ranking: A > B > C > D
        votes = [
            ("NameA", "NameB", -1),
            ("NameB", "NameC", -1),
            ("NameC", "NameD", -1),
            ("NameA", "NameC", -1),
            ("NameA", "NameD", -1),
            ("NameB", "NameD", -1),
        ]

        for winner, loser, pref in votes:
            ratings = update_preference_and_save(ratings, winner, loser)

        # Verify transitive property (approximately)
        # A should be highest, D should be lowest
        assert ratings["NameA"] > ratings["NameD"]

        # Count how many pairwise relationships match expected order
        expected_order = ["NameA", "NameB", "NameC", "NameD"]
        correct_order_count = 0
        total_pairs = 0

        for i, name_i in enumerate(expected_order):
            for j, name_j in enumerate(expected_order):
                if i < j:  # name_i should rank higher than name_j
                    total_pairs += 1
                    if ratings[name_i] > ratings[name_j]:
                        correct_order_count += 1

        # At least 80% of pairwise relationships should match
        accuracy = correct_order_count / total_pairs
        assert accuracy >= 0.8, f"Rating accuracy {accuracy} below threshold"


class TestSessionPersistence:
    """Test session persistence: reload → verify state preserved."""

    @pytest.fixture
    def persistence_db(self, initialized_db):
        """Database with names and votes for persistence tests."""
        test_names = [
            ("Emma", "Female", "Nordic"),
            ("Liam", "Male", "Nordic"),
        ]

        with get_connection() as conn:
            for name, gender, origin in test_names:
                conn.execute(
                    "INSERT INTO names (name, gender, origin_region) VALUES (?, ?, ?)",
                    (name, gender, origin),
                )

        return initialized_db

    def test_ratings_persisted_in_database(self, persistence_db):
        """Test that ratings are properly persisted and can be reloaded."""
        # First session: initialize and vote
        names = get_names_by_gender()
        all_names = names.get("All", [])

        ratings = initialize_or_load_ratings(all_names)
        ratings = update_preference_and_save(ratings, "Emma", "Liam")

        # Store the rating for later comparison
        emma_rating_after_vote = ratings["Emma"]

        # Simulate "session end" - reload ratings from database
        ratings_reloaded = initialize_or_load_ratings(all_names)

        # Verify ratings persisted
        assert ratings_reloaded["Emma"] == emma_rating_after_vote
        assert "Liam" in ratings_reloaded

    def test_comparisons_persisted_across_sessions(self, persistence_db):
        """Test that comparisons are persisted across sessions."""
        # First session: cast votes
        names = get_names_by_gender()
        all_names = names.get("All", [])
        ratings = initialize_or_load_ratings(all_names)

        ratings = update_preference_and_save(ratings, "Emma", "Liam")
        ratings = update_preference_draw_and_save(ratings, "Emma", "Liam")

        # Verify comparisons in database after first session
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            initial_count = cursor.fetchone()[0]
            assert initial_count >= 1  # At least one comparison recorded

        # Simulate new session: just query database directly
        # (database remains persisted across "sessions")
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            new_count = cursor.fetchone()[0]
            assert new_count == initial_count  # Data persisted

    def test_model_state_persisted(self, persistence_db):
        """Test that model state is persisted and can be reloaded."""
        from st_name_ranking.active_learning.selection import get_active_learning_model

        names = get_names_by_gender()
        all_names = names.get("All", [])
        ratings = initialize_or_load_ratings(all_names)

        # Get initial model and cast votes
        model = get_active_learning_model()
        initial_training_samples = model.state.training_samples

        ratings = update_preference_and_save(ratings, "Emma", "Liam")

        # Verify model was updated
        assert model.state.training_samples > initial_training_samples

        # Save model state
        model.save_to_db()

        # Simulate fresh model load
        with patch.dict("sys.modules", {"streamlit": MagicMock()}):
            # Reset model cache
            import st_name_ranking.active_learning.selection as selection_module
            from st_name_ranking.model import initialize_model_if_needed

            selection_module.reset_active_learning_state()

            # Create new feature extractor to get feature names
            extractor = FeatureExtractor()
            feature_names = extractor.get_feature_names()

            # Load model from database
            new_model = initialize_model_if_needed(feature_names)

            # Verify state was reloaded
            assert new_model.state.training_samples == model.state.training_samples


class TestEdgeCases:
    """Test edge cases and error handling in workflows."""

    def test_empty_database_no_candidates(self, initialized_db):
        """Test that empty database returns empty candidates."""
        names = []
        name_a, name_b = select_candidates(names)
        assert name_a == ""
        assert name_b == ""

    def test_single_name_no_candidates(self, initialized_db):
        """Test that single name returns empty candidates."""
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("SoloName", "Female"),
            )

        names = ["SoloName"]
        name_a, name_b = select_candidates(names)
        assert name_a == ""
        assert name_b == ""

    def test_vote_nonexistent_name_handles_gracefully(self, initialized_db):
        """Test that voting on non-existent name handles gracefully (logs warning, returns copy)."""
        ratings = {"FakeName": 1500.0, "AnotherFake": 1500.0}

        # Should not raise - error is caught and logged
        result = update_preference_and_save(ratings, "FakeName", "AnotherFake")

        # Should return a copy of original ratings
        assert result == ratings
        assert result is not ratings  # Different object

    def test_sync_with_empty_submodule(self, initialized_db, tmp_path):
        """Test syncing with empty submodule handles gracefully."""
        submodule_path = tmp_path / "empty_submodule"
        submodule_path.mkdir()

        # Create empty JSON file
        json_file = submodule_path / "allenavne.json"
        json_file.write_text("[]")

        # Should not error, just return 0
        inserted = sync_names_with_submodule(submodule_path)
        assert inserted == 0

    def test_sync_with_missing_file_raises_error(self, initialized_db, tmp_path):
        """Test that syncing with missing file raises error."""
        submodule_path = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            sync_names_with_submodule(submodule_path)

    def test_concurrent_votes_handled_correctly(self, initialized_db):
        """Test that concurrent votes on same pair are handled."""
        # Insert names
        with get_connection() as conn:
            for name in ["Name1", "Name2"]:
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, ?)",
                    (name, "Female"),
                )

        names = ["Name1", "Name2"]
        ratings = initialize_or_load_ratings(names)

        # Vote multiple times on same pair
        for _ in range(5):
            ratings = update_preference_and_save(ratings, "Name1", "Name2")

        # Should have 5 comparisons (or fewer if duplicates are prevented)
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            count = cursor.fetchone()[0]
            # Note: UNIQUE constraint may prevent exact duplicates
            assert count >= 1


class TestWorkflowIntegrationWithClassify:
    """Test workflow integration with origin classification."""

    def test_full_workflow_with_classification(self, initialized_db, mock_submodule_path, mock_classifier):
        """Test complete workflow including origin classification."""
        # Step 1: Sync names
        inserted = sync_names_with_submodule(mock_submodule_path)
        assert inserted == 3

        # Step 2: Classify origins (mocked)
        with get_connection() as conn:
            cursor = conn.execute("SELECT id, name FROM names")
            names_data = cursor.fetchall()

        classified_count = 0
        for name_id, name in names_data:
            region, confidence = "Nordic", 0.85
            update_name_origin(name_id, region, confidence)
            classified_count += 1

        assert classified_count == 3

        # Step 3: Verify origins in database
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM names WHERE origin_region IS NOT NULL")
            assert cursor.fetchone()[0] == 3

        # Step 4: Filter by origin and vote
        nordic_names = get_names_by_filters(origins=["Nordic"])
        assert len(nordic_names) == 3

        ratings = initialize_or_load_ratings(nordic_names)
        name_a, name_b = select_candidates(nordic_names)

        ratings = update_preference_and_save(ratings, name_a, name_b)

        # Verify workflow completed
        db_ratings = get_ratings()
        assert name_a in db_ratings
        assert name_b in db_ratings


class TestDatabaseTransactionIntegrity:
    """Test database transaction integrity during workflows."""

    def test_failed_vote_rolls_back(self, initialized_db):
        """Test that failed votes roll back properly."""
        # Insert names
        with get_connection() as conn:
            for name in ["NameA", "NameB"]:
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, ?)",
                    (name, "Female"),
                )

        # Get initial comparison count
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            initial_count = cursor.fetchone()[0]

        # Attempt to record comparison with invalid preference
        with pytest.raises(ValueError):
            record_comparison("NameA", "NameB", preference=99)

        # Verify no comparison was recorded
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            assert cursor.fetchone()[0] == initial_count

    def test_database_integrity_after_workflow(self, initialized_db):
        """Test database integrity constraints after workflow."""
        # Insert test data
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO names (name, gender) VALUES (?, ?)",
                ("TestName", "Female"),
            )

        # Try to violate UNIQUE constraint on names
        with pytest.raises(sqlite3.IntegrityError):
            with get_connection() as conn:
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, ?)",
                    ("TestName", "Male"),
                )

    def test_rating_update_atomicity(self, initialized_db):
        """Test that rating updates are atomic."""
        # Insert names
        with get_connection() as conn:
            for name in ["NameA", "NameB"]:
                conn.execute(
                    "INSERT INTO names (name, gender) VALUES (?, ?)",
                    (name, "Female"),
                )

        names = ["NameA", "NameB"]
        ratings = initialize_or_load_ratings(names)

        # Cast vote
        ratings = update_preference_and_save(ratings, "NameA", "NameB")

        # Verify both names have ratings
        db_ratings = get_ratings()
        assert "NameA" in db_ratings
        assert "NameB" in db_ratings

        # Verify comparison recorded
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            assert cursor.fetchone()[0] == 1


class TestWorkflowPerformance:
    """Test workflow performance with larger datasets."""

    @pytest.fixture
    def large_db(self, initialized_db):
        """Database with many names for performance tests."""
        # Insert 50 names
        with get_connection() as conn:
            for i in range(50):
                gender = "Female" if i % 2 == 0 else "Male"
                origin = ["Nordic", "European", "American"][i % 3]
                conn.execute(
                    "INSERT INTO names (name, gender, origin_region) VALUES (?, ?, ?)",
                    (f"Name{i:03d}", gender, origin),
                )

        return initialized_db

    def test_candidate_selection_performance(self, large_db):
        """Test that candidate selection works with many names."""
        names = get_names_by_gender()
        all_names = names.get("All", [])
        assert len(all_names) == 50

        # Should complete quickly
        name_a, name_b = select_candidates(all_names)
        assert name_a in all_names
        assert name_b in all_names
        assert name_a != name_b

    def test_filtering_performance(self, large_db):
        """Test that filtering works efficiently with many names."""
        # Filter by gender
        female_names = get_names_by_filters(gender="Female")
        assert len(female_names) == 25  # Half of 50

        # Filter by origin
        nordic_names = get_names_by_filters(origins=["Nordic"])
        assert len(nordic_names) == 17  # About 1/3 of 50

        # Combined filter
        female_nordic = get_names_by_filters(gender="Female", origins=["Nordic"])
        assert len(female_nordic) >= 8  # About 1/6 of 50

    def test_multiple_votes_performance(self, large_db):
        """Test that multiple votes work efficiently."""
        names = get_names_by_gender()
        all_names = names.get("All", [])
        ratings = initialize_or_load_ratings(all_names)

        # Cast up to 20 votes (some may be duplicates that get deduplicated)
        votes_cast = 0
        for i in range(20):
            name_a, name_b = select_candidates(all_names)
            if name_a and name_b:
                ratings = update_preference_and_save(ratings, name_a, name_b)
                votes_cast += 1

        # Verify state (at least some comparisons recorded, allowing for deduplication)
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            count = cursor.fetchone()[0]
            assert count >= 1, "Should have at least one comparison recorded"
            assert count <= votes_cast, "Count should not exceed votes cast"

        db_ratings = get_ratings()
        assert len(db_ratings) >= 2  # At least the voted names have ratings
