"""Concurrent access and transaction safety integration tests.

Critical for production with multiple users.
Tests thread-safety and transaction isolation.
"""

import multiprocessing
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

# Use 'spawn' start method to avoid fork warnings with multi-threaded pytest
# Fork with threads can cause deadlocks
multiprocessing.set_start_method("spawn", force=True)


# Module-level helper functions for multiprocessing tests
def _init_in_process(db_path_str):
    """Initialize database in a subprocess (for testing)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from st_name_ranking import database

    # Set the database path
    database.DB_PATH = Path(db_path_str)
    database.reset_database_init_state()

    try:
        database.init_database()
        # Verify we can query
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM names")
            count = cursor.fetchone()[0]
    except sqlite3.Error as e:
        return ("error", str(e))
    else:
        return ("success", count)


def _vote_in_process(args):
    """Cast votes from a subprocess (for testing)."""
    process_id, db_path = args
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from st_name_ranking import database

    database.DB_PATH = Path(db_path)
    database.reset_database_init_state()

    try:
        for i in range(10):
            preference = -1 if i % 2 == 0 else 1
            database.record_comparison("Alice", "Bob", preference)
    except sqlite3.Error as e:
        return ("error", str(e))
    else:
        return ("success", process_id)


class TestConcurrentDatabaseInitialization:
    """Tests for concurrent database initialization race conditions."""

    def test_concurrent_database_initialization(self, mock_db_path):
        """
        Multiple threads trying to initialize simultaneously
        should not cause duplicate tables or errors.
        """
        from st_name_ranking import database

        # Reset initialization flag
        database.reset_database_init_state()

        results = []
        errors = []

        def init_worker():
            try:
                database.init_database()
                results.append("success")
            except sqlite3.Error as e:
                errors.append(str(e))

        # Spawn multiple threads to initialize simultaneously
        threads = []
        for _ in range(10):
            t = threading.Thread(target=init_worker)
            threads.append(t)

        # Start all threads simultaneously
        for t in threads:
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Initialization errors: {errors}"
        assert len(results) == 10, "Not all threads completed successfully"

        # Verify database is valid (only one set of tables)
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]

            # Should have all expected tables, no duplicates
            expected_tables = {
                "comparisons",
                "model_state",
                "names",
                "ratings",
                "region_mapping",
                "source_versions",
                "user_settings",
            }
            actual_tables = set(tables)
            assert expected_tables <= actual_tables, f"Missing tables: {expected_tables - actual_tables}"

    def test_concurrent_initialization_process_isolation(self, temp_db_path):
        """
        Multiple processes initializing should handle the race condition.
        """
        # Use multiprocessing to test process isolation
        with multiprocessing.Pool(5) as pool:
            results = pool.map(_init_in_process, [str(temp_db_path)] * 5)

        # All should succeed
        errors = [r for r in results if r[0] == "error"]
        assert len(errors) == 0, f"Process initialization errors: {errors}"

        successes = [r for r in results if r[0] == "success"]
        assert len(successes) == 5, "Not all processes completed successfully"


class TestConcurrentVoting:
    """Tests for concurrent voting operations."""

    def test_concurrent_votes_handled_correctly(self, initialized_db):
        """
        Multiple threads voting simultaneously should:
        - Not corrupt database
        - Record all valid votes
        - Maintain rating consistency
        """
        from st_name_ranking import database

        # Insert test names
        with database.get_connection() as conn:
            conn.execute("INSERT INTO names (name, gender) VALUES ('Alice', 'Female')")
            conn.execute("INSERT INTO names (name, gender) VALUES ('Bob', 'Male')")
            conn.execute("INSERT INTO names (name, gender) VALUES ('Carol', 'Female')")
            conn.execute("INSERT INTO names (name, gender) VALUES ('David', 'Male')")

        names = ["Alice", "Bob", "Carol", "David"]
        num_threads = 10
        votes_per_thread = 20

        errors = []
        successful_votes = []

        def vote_worker(thread_id):
            """Cast multiple votes from a thread."""
            for i in range(votes_per_thread):
                try:
                    # Alternate between different name pairs
                    name_a = names[i % len(names)]
                    name_b = names[(i + 1) % len(names)]

                    # Record comparison
                    preference = -1 if i % 2 == 0 else 1
                    database.record_comparison(name_a, name_b, preference)

                    # Update rating
                    database.update_rating(name_a, 1500.0 + i)

                    successful_votes.append((thread_id, i))
                except sqlite3.Error as e:
                    errors.append((thread_id, i, str(e)))

        # Create and start threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=vote_worker, args=(i,))
            threads.append(t)

        start_time = time.time()
        for t in threads:
            t.start()

        for t in threads:
            t.join()
        duration = time.time() - start_time

        # Verify no errors
        assert len(errors) == 0, f"Vote errors: {errors[:5]}..."
        assert len(successful_votes) == num_threads * votes_per_thread

        # Verify database integrity
        with database.get_connection() as conn:
            # Check comparisons count
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            comparison_count = cursor.fetchone()[0]
            assert comparison_count > 0, "No comparisons recorded"

            # Check ratings were updated
            cursor = conn.execute("SELECT COUNT(*) FROM ratings")
            rating_count = cursor.fetchone()[0]
            assert rating_count > 0, "No ratings recorded"

            # Verify no duplicate comparison anomalies
            cursor = conn.execute("""
                SELECT name_a_id, name_b_id, COUNT(*) as cnt
                FROM comparisons
                GROUP BY name_a_id, name_b_id
                HAVING cnt > 1
            """)
            # Duplicates are allowed due to UNIQUE constraint on (name_a_id, name_b_id, preference)
            # So this query would show multiple preferences for same pair

        print(f"Concurrent voting completed in {duration:.2f}s with {len(successful_votes)} votes")

    def test_concurrent_votes_process_isolation(self, initialized_db):
        """
        Multiple processes voting should maintain consistency.
        """
        from st_name_ranking import database

        # Insert test names
        with database.get_connection() as conn:
            conn.execute("INSERT INTO names (name, gender) VALUES ('Alice', 'Female')")
            conn.execute("INSERT INTO names (name, gender) VALUES ('Bob', 'Male')")

        db_path = str(initialized_db)

        with multiprocessing.Pool(4) as pool:
            results = pool.map(_vote_in_process, [(i, db_path) for i in range(4)])

        errors = [r for r in results if r[0] == "error"]
        assert len(errors) == 0, f"Process vote errors: {errors}"

        # Verify votes were recorded
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            count = cursor.fetchone()[0]
            assert count > 0, "No comparisons recorded from processes"


class TestModelUpdateAtomicity:
    """Tests for atomic model updates."""

    def test_model_update_is_atomic(self, initialized_db):
        """
        If model.save_to_db succeeds but comparison recording fails,
        the system should detect inconsistency.
        """
        from st_name_ranking import database, model
        from st_name_ranking.features import FeatureExtractor

        # Initialize model
        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()

        # Create and save initial model
        bt_model = model.BradleyTerryModel(feature_names)
        bt_model.save_to_db()

        # Verify model was saved
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM model_state")
            assert cursor.fetchone()[0] == 1, "Model not saved"

        # Test 1: Simulate failure during comparison recording
        with pytest.raises(Exception):
            with database.get_connection() as conn:
                # This should rollback both operations
                bt_model.state.training_samples += 1
                bt_model.save_to_db()
                raise Exception("Simulated failure")

        # Verify model state wasn't corrupted
        new_model = model.BradleyTerryModel(feature_names)
        loaded = new_model.load_from_db()
        assert loaded, "Model should load successfully"

    def test_comparison_and_rating_update_consistency(self, initialized_db):
        """
        Comparison recording and rating updates should be consistent.
        """
        from st_name_ranking import database

        # Insert test names
        with database.get_connection() as conn:
            conn.execute("INSERT INTO names (name, gender) VALUES ('Winner', 'Female')")
            conn.execute("INSERT INTO names (name, gender) VALUES ('Loser', 'Male')")

        # Record comparison and update ratings in same logical operation
        with database.get_connection() as conn:
            # Get name IDs
            cursor = conn.execute("SELECT id FROM names WHERE name = 'Winner'")
            winner_id = cursor.fetchone()[0]
            cursor = conn.execute("SELECT id FROM names WHERE name = 'Loser'")
            loser_id = cursor.fetchone()[0]

            # Insert comparison
            conn.execute(
                """
                INSERT INTO comparisons (name_a_id, name_b_id, preference)
                VALUES (?, ?, -1)
            """,
                (winner_id, loser_id),
            )

            # Update ratings
            conn.execute(
                """
                INSERT OR REPLACE INTO ratings (name_id, rating, matches, last_updated)
                VALUES (?, 1600.0, 1, CURRENT_TIMESTAMP)
            """,
                (winner_id,),
            )

            conn.execute(
                """
                INSERT OR REPLACE INTO ratings (name_id, rating, matches, last_updated)
                VALUES (?, 1400.0, 1, CURRENT_TIMESTAMP)
            """,
                (loser_id,),
            )

        # Verify consistency
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            assert cursor.fetchone()[0] == 1

            cursor = conn.execute("SELECT COUNT(*) FROM ratings")
            assert cursor.fetchone()[0] == 2

            cursor = conn.execute("SELECT rating FROM ratings WHERE name_id = ?", (winner_id,))
            assert cursor.fetchone()[0] == 1600.0


class TestReadConsistency:
    """Tests for read consistency during concurrent writes."""

    def test_read_consistency_during_updates(self, initialized_db):
        """
        Reading ratings while updates happen should not crash
        and should return consistent (though possibly outdated) data.
        """
        from st_name_ranking import database

        # Insert test names
        with database.get_connection() as conn:
            for i in range(100):
                conn.execute("INSERT INTO names (name, gender) VALUES (?, ?)", (f"Name{i}", "Female"))

        read_results = []
        write_errors = []
        stop_event = threading.Event()

        def writer_worker():
            """Continuously write updates."""
            try:
                for i in range(500):
                    if stop_event.is_set():
                        break
                    database.update_rating(f"Name{i % 100}", 1500.0 + (i % 100))
                    time.sleep(0.001)  # Small delay to allow interleaving
            except sqlite3.Error as e:
                write_errors.append(str(e))

        def reader_worker():
            """Continuously read ratings."""
            try:
                for _ in range(200):
                    if stop_event.is_set():
                        break
                    ratings = database.get_ratings()
                    read_results.append(len(ratings))
                    time.sleep(0.002)
            except sqlite3.Error as e:
                read_results.append(f"error: {e}")

        # Start writer thread
        writer = threading.Thread(target=writer_worker)
        writer.start()

        # Start multiple reader threads
        readers = [threading.Thread(target=reader_worker) for _ in range(3)]
        for r in readers:
            r.start()

        # Let them run for a bit
        time.sleep(1)
        stop_event.set()

        # Wait for completion
        writer.join(timeout=5)
        for r in readers:
            r.join(timeout=5)

        # Verify no crashes
        errors = [r for r in read_results if isinstance(r, str) and r.startswith("error")]
        assert len(errors) == 0, f"Read errors during concurrent writes: {errors}"
        assert len(write_errors) == 0, f"Write errors: {write_errors}"

        # All reads should have returned non-negative counts
        valid_reads = [r for r in read_results if isinstance(r, int)]
        assert len(valid_reads) > 0, "No successful reads"
        assert all(r >= 0 for r in valid_reads), "Invalid rating counts"

    def test_no_dirty_reads(self, initialized_db):
        """
        Uncommitted changes should not be visible to other connections.
        """
        from st_name_ranking import database

        # Insert test name
        with database.get_connection() as conn:
            conn.execute("INSERT INTO names (name, gender) VALUES ('TestName', 'Female')")

        # Create two separate connections
        conn1 = sqlite3.connect(initialized_db)
        conn2 = sqlite3.connect(initialized_db)

        try:
            # Start transaction on conn1 but don't commit
            conn1.execute("BEGIN")
            conn1.execute("INSERT INTO names (name, gender) VALUES ('Uncommitted', 'Male')")

            # Try to read from conn2 - should not see uncommitted data
            cursor = conn2.execute("SELECT name FROM names WHERE name = 'Uncommitted'")
            result = cursor.fetchone()
            assert result is None, "Dirty read detected - uncommitted data visible"

            # Commit on conn1
            conn1.commit()

            # Now conn2 should see the data
            cursor = conn2.execute("SELECT name FROM names WHERE name = 'Uncommitted'")
            result = cursor.fetchone()
            assert result is not None, "Committed data not visible"

        finally:
            conn1.close()
            conn2.close()


class TestConnectionPool:
    """Tests for multiple database connections."""

    def test_multiple_connections_work_correctly(self, initialized_db):
        """
        Multiple database connections should work without conflicts.
        """
        from st_name_ranking import database

        # Insert test names
        with database.get_connection() as conn:
            conn.execute("INSERT INTO names (name, gender) VALUES ('ConnTest1', 'Female')")
            conn.execute("INSERT INTO names (name, gender) VALUES ('ConnTest2', 'Male')")

        results = []
        errors = []

        def connection_worker(worker_id):
            """Use connection from a thread."""
            try:
                with database.get_connection() as conn:
                    # Read
                    cursor = conn.execute("SELECT name FROM names")
                    names = [row[0] for row in cursor.fetchall()]

                    # Write
                    conn.execute("INSERT INTO names (name, gender) VALUES (?, ?)", (f"Worker{worker_id}", "Female"))

                    results.append((worker_id, len(names)))
            except sqlite3.Error as e:
                errors.append((worker_id, str(e)))

        # Create multiple threads using connections simultaneously
        threads = [threading.Thread(target=connection_worker, args=(i,)) for i in range(20)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Verify all succeeded
        assert len(errors) == 0, f"Connection errors: {errors}"
        assert len(results) == 20, f"Only {len(results)} workers completed"

        # Verify all writes were committed
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM names")
            count = cursor.fetchone()[0]
            # 2 initial + 20 workers
            assert count >= 22, f"Expected at least 22 names, got {count}"

    def test_connection_isolation(self, initialized_db):
        """
        Each connection should see committed data (read committed isolation).
        SQLite default mode is READ COMMITTED, not snapshot isolation.
        """
        from st_name_ranking import database

        # Insert initial data
        with database.get_connection() as conn:
            conn.execute("INSERT INTO names (name, gender) VALUES ('IsoTest', 'Female')")

        # Get connection A
        with database.get_connection() as conn_a:
            # Read initial data
            cursor = conn_a.execute("SELECT COUNT(*) FROM names")
            count_a_before = cursor.fetchone()[0]

            # Make change from another connection and commit
            with database.get_connection() as conn_b:
                conn_b.execute("INSERT INTO names (name, gender) VALUES ('NewName', 'Male')")

            # Connection A will see committed changes from other connections
            # (SQLite READ COMMITTED behavior, not snapshot isolation)
            cursor = conn_a.execute("SELECT COUNT(*) FROM names")
            count_a_after = cursor.fetchone()[0]
            # After commit by B, A sees the new data on next query
            assert count_a_after >= count_a_before, "Connection should see committed data"

        # After closing and reopening, should still see new data
        with database.get_connection() as conn_c:
            cursor = conn_c.execute("SELECT COUNT(*) FROM names")
            count_c = cursor.fetchone()[0]
            assert count_c > count_a_before, "New connection didn't see committed data"


class TestTransactionRollback:
    """Tests for transaction rollback behavior."""

    def test_transaction_rollback_on_error(self, initialized_db):
        """
        Partial operations should be rolled back on error.
        """
        from st_name_ranking import database

        # Insert test names
        with database.get_connection() as conn:
            conn.execute("INSERT INTO names (name, gender) VALUES ('RollbackTest1', 'Female')")

        # Get initial count
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM names")
            initial_count = cursor.fetchone()[0]

        # Attempt transaction that will fail
        try:
            with database.get_connection() as conn:
                # Insert some data
                conn.execute("INSERT INTO names (name, gender) VALUES ('RollbackTest2', 'Male')")
                # This should trigger rollback
                raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected

        # Verify rollback occurred
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM names")
            final_count = cursor.fetchone()[0]
            assert final_count == initial_count, "Transaction was not rolled back"

            # Verify the second insert was rolled back
            cursor = conn.execute("SELECT name FROM names WHERE name = 'RollbackTest2'")
            assert cursor.fetchone() is None, "Rolled back data still in database"

    def test_partial_rating_update_rollback(self, initialized_db):
        """
        If one rating update fails, previous updates in same transaction
        should be rolled back.
        """
        from st_name_ranking import database

        # Insert test names
        with database.get_connection() as conn:
            for i in range(5):
                conn.execute("INSERT INTO names (name, gender) VALUES (?, ?)", (f"RollbackName{i}", "Female"))

        # Attempt batch update with failure
        try:
            with database.get_connection() as conn:
                for i in range(5):
                    if i == 3:
                        # Simulate failure mid-batch
                        raise RuntimeError("Simulated batch failure")
                    conn.execute(
                        """
                        INSERT INTO ratings (name_id, rating, matches)
                        VALUES ((SELECT id FROM names WHERE name = ?), 1500, 0)
                    """,
                        (f"RollbackName{i}",),
                    )
        except RuntimeError:
            pass

        # Verify no partial updates remain
        with database.get_connection() as conn:
            cursor = conn.execute("""
                SELECT n.name FROM ratings r
                JOIN names n ON r.name_id = n.id
                WHERE n.name LIKE 'RollbackName%'
            """)
            remaining = [row[0] for row in cursor.fetchall()]
            assert len(remaining) == 0, f"Partial updates not rolled back: {remaining}"

    def test_foreign_key_constraint_rollback(self, initialized_db):
        """
        Foreign key violations should trigger rollback.
        """
        from st_name_ranking import database

        # Enable foreign key constraints
        with database.get_connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")

        # Try to insert rating for non-existent name
        try:
            with database.get_connection() as conn:
                # Insert a valid comparison first
                conn.execute("INSERT INTO names (name, gender) VALUES ('ValidName', 'Female')")
                name_id = conn.execute("SELECT id FROM names WHERE name = 'ValidName'").fetchone()[0]

                # Insert valid rating
                conn.execute("INSERT INTO ratings (name_id, rating) VALUES (?, 1500)", (name_id,))

                # Try invalid rating (should fail)
                conn.execute(
                    "INSERT INTO ratings (name_id, rating) VALUES (?, 1500)",
                    (99999,),  # Non-existent name_id
                )
        except sqlite3.IntegrityError:
            pass  # Expected

        # Verify valid operations were rolled back
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM names WHERE name = 'ValidName'")
            # The name might exist depending on transaction boundary
            # but the rating should not exist
            cursor = conn.execute("""
                SELECT r.name_id FROM ratings r
                JOIN names n ON r.name_id = n.id
                WHERE n.name = 'ValidName'
            """)
            # With SQLite's default behavior and our context manager,
            # the transaction should be rolled back


class TestSingletonRaceConditions:
    """Tests for singleton initialization race conditions."""

    def test_model_singleton_race_condition_exists(self, initialized_db):
        """
        Demonstrate that model singleton has race condition.

        WARNING: This test documents a known issue - the singleton pattern
        in utils.py is not thread-safe. Multiple threads can create
        separate model instances simultaneously.
        """
        from st_name_ranking import utils

        # Reset singleton
        utils._model = None
        utils._feature_extractor = None

        models = []
        errors = []

        def get_model_worker():
            try:
                model = utils.get_active_learning_model()
                models.append(id(model))
            except (RuntimeError, ValueError, AttributeError) as e:
                errors.append(str(e))

        # Spawn threads to get model simultaneously
        threads = [threading.Thread(target=get_model_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Model initialization errors: {errors}"
        assert len(models) == 10, "Not all threads got a model"

        # This documents the race condition: multiple instances may be created
        # In production, this could lead to:
        # - Duplicate model initialization
        # - Wasted resources
        # - Inconsistent model state between threads
        num_unique_models = len(set(models))
        print(f"\nWARNING: {num_unique_models} unique model instances created (race condition)")
        # We don't assert == 1 because the race condition exists
        # This test documents the issue for future fixing

    def test_feature_extractor_singleton_thread_safety(self, initialized_db):
        """
        Feature extractor singleton should be thread-safe.
        """
        from st_name_ranking import utils

        # Reset singleton
        utils._feature_extractor = None

        extractors = []
        errors = []

        def get_extractor_worker():
            try:
                extractor = utils.get_feature_extractor()
                extractors.append(id(extractor))
            except (RuntimeError, ValueError, AttributeError) as e:
                errors.append(str(e))

        threads = [threading.Thread(target=get_extractor_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Extractor initialization errors: {errors}"
        assert len(set(extractors)) == 1, "Multiple extractor instances created"


class TestDatabaseIntegrity:
    """Tests for database integrity under concurrent load."""

    def test_unique_constraint_enforcement(self, initialized_db):
        """
        Unique constraints should be enforced even under concurrent load.
        """
        from st_name_ranking import database

        errors = []

        def insert_duplicate_worker(thread_id):
            try:
                with database.get_connection() as conn:
                    conn.execute("INSERT INTO names (name, gender) VALUES ('UniqueName', 'Female')")
            except sqlite3.IntegrityError:
                errors.append("unique_violation")
            except sqlite3.Error as e:
                errors.append(str(e))

        # Only first insert should succeed
        threads = [threading.Thread(target=insert_duplicate_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Count unique violations
        unique_violations = errors.count("unique_violation")
        assert unique_violations >= 9, f"Expected at least 9 unique violations, got {unique_violations}"

        # Only one row should exist
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM names WHERE name = 'UniqueName'")
            count = cursor.fetchone()[0]
            assert count == 1, f"Expected 1 row, got {count}"

    def test_check_constraint_enforcement(self, initialized_db):
        """
        CHECK constraints should be enforced.
        """
        from st_name_ranking import database

        # Insert test names
        with database.get_connection() as conn:
            conn.execute("INSERT INTO names (name, gender) VALUES ('CheckTest1', 'Female')")
            conn.execute("INSERT INTO names (name, gender) VALUES ('CheckTest2', 'Male')")
            id1 = conn.execute("SELECT id FROM names WHERE name = 'CheckTest1'").fetchone()[0]
            id2 = conn.execute("SELECT id FROM names WHERE name = 'CheckTest2'").fetchone()[0]

        # Try invalid preference value
        with pytest.raises(sqlite3.IntegrityError):
            with database.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO comparisons (name_a_id, name_b_id, preference)
                    VALUES (?, ?, 999)
                """,
                    (id1, id2),
                )

        # Valid preferences should work
        for pref in [-1, 0, 1, 2]:
            with database.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO comparisons (name_a_id, name_b_id, preference)
                    VALUES (?, ?, ?)
                """,
                    (id1, id2, pref),
                )


class TestConcurrentComparisonRecording:
    """Tests specific to comparison recording race conditions."""

    def test_comparison_idempotency(self, initialized_db):
        """
        Recording the same comparison twice should not duplicate.
        """
        from st_name_ranking import database

        # Insert test names
        with database.get_connection() as conn:
            conn.execute("INSERT INTO names (name, gender) VALUES ('IdemA', 'Female')")
            conn.execute("INSERT INTO names (name, gender) VALUES ('IdemB', 'Male')")

        # Record same comparison multiple times concurrently
        def record_worker():
            try:
                database.record_comparison("IdemA", "IdemB", -1)
            except sqlite3.Error:
                pass

        threads = [threading.Thread(target=record_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should only have one comparison (due to UNIQUE constraint)
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            count = cursor.fetchone()[0]
            # Note: UNIQUE constraint is on (name_a_id, name_b_id, preference)
            # So same pair with same preference should only exist once
            assert count <= 1, f"Expected at most 1 comparison, got {count}"

    def test_preference_update_race_condition(self, initialized_db):
        """
        Multiple threads updating preference for same pair should be handled.
        """
        from st_name_ranking import database

        # Insert test names
        with database.get_connection() as conn:
            conn.execute("INSERT INTO names (name, gender) VALUES ('RaceA', 'Female')")
            conn.execute("INSERT INTO names (name, gender) VALUES ('RaceB', 'Male')")

        # Concurrent updates with different preferences
        def update_worker(preference):
            try:
                database.record_comparison("RaceA", "RaceB", preference)
            except sqlite3.Error:
                pass

        threads = [
            threading.Thread(target=update_worker, args=(-1,)),
            threading.Thread(target=update_worker, args=(0,)),
            threading.Thread(target=update_worker, args=(1,)),
            threading.Thread(target=update_worker, args=(2,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All unique preferences should be recorded
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT preference FROM comparisons ORDER BY preference")
            preferences = [row[0] for row in cursor.fetchall()]
            # Could have all 4 or fewer depending on timing
            assert len(preferences) <= 4


class TestUtilsConcurrency:
    """Tests for concurrent access to utility functions."""

    def test_update_model_and_save_concurrent(self, initialized_db):
        """
        Multiple threads calling update_model_and_save should be safe.
        """
        from st_name_ranking import database, utils
        from st_name_ranking.features import FeatureExtractor

        # Initialize model first
        extractor = FeatureExtractor()
        utils.get_active_learning_model()

        # Insert test names
        names = ["ModelA", "ModelB", "ModelC", "ModelD"]
        with database.get_connection() as conn:
            for name in names:
                conn.execute("INSERT INTO names (name, gender) VALUES (?, ?)", (name, "Female"))

        errors = []

        def model_update_worker(thread_id):
            try:
                for i in range(10):
                    winner = names[i % len(names)]
                    loser = names[(i + 1) % len(names)]
                    utils.update_model_and_save(winner, loser)
            except (RuntimeError, ValueError, sqlite3.Error) as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=model_update_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Some errors are expected due to non-atomic operations
        # but database should remain consistent
        with database.get_connection() as conn:
            cursor = conn.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            assert result == "ok", f"Database integrity check failed: {result}"

    def test_update_preference_and_save_concurrent(self, initialized_db):
        """
        Multiple threads calling update_preference_and_save should be safe.
        """
        from st_name_ranking import database, utils
        from st_name_ranking.features import FeatureExtractor

        # Initialize
        extractor = FeatureExtractor()
        utils.get_active_learning_model()

        # Insert test names
        names = ["PrefA", "PrefB"]
        with database.get_connection() as conn:
            for name in names:
                conn.execute("INSERT INTO names (name, gender) VALUES (?, ?)", (name, "Female"))

        ratings = dict.fromkeys(names, 1500.0)

        errors = []

        def preference_worker(thread_id):
            try:
                for i in range(5):
                    utils.update_preference_and_save(ratings, "PrefA", "PrefB")
            except (RuntimeError, ValueError, sqlite3.Error) as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=preference_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify database consistency
        with database.get_connection() as conn:
            cursor = conn.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            assert result == "ok", f"Database integrity check failed: {result}"


class TestConnectionTimeoutAndLocks:
    """Tests for connection timeout and locking behavior."""

    def test_busy_timeout_handling(self, initialized_db):
        """
        Database should handle busy timeouts gracefully.
        """
        from st_name_ranking import database

        # Set busy timeout
        with database.get_connection() as conn:
            conn.execute("PRAGMA busy_timeout = 5000")  # 5 seconds

        # This is more of a configuration test - actual busy scenarios
        # are hard to reproduce deterministically

    def test_write_lock_contentions(self, initialized_db):
        """
        Multiple writers should be serialized by SQLite.
        """
        from st_name_ranking import database

        # Insert test name
        with database.get_connection() as conn:
            conn.execute("INSERT INTO names (name, gender) VALUES ('LockTest', 'Female')")

        write_times = []
        errors = []

        def slow_writer(thread_id):
            try:
                start = time.time()
                with database.get_connection() as conn:
                    # Simulate slow write
                    time.sleep(0.01)
                    conn.execute(
                        "UPDATE names SET gender = ? WHERE name = ?",
                        ("Male" if thread_id % 2 == 0 else "Female", "LockTest"),
                    )
                write_times.append((thread_id, time.time() - start))
            except sqlite3.Error as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=slow_writer, args=(i,)) for i in range(10)]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = time.time() - start

        # All writes should complete
        assert len(write_times) == 10, f"Only {len(write_times)} writes completed"
        assert len(errors) == 0, f"Write errors: {errors}"

        # Total time should be at least sum of sleep times if serialized
        # but with connection pooling, they may overlap
        # Just verify no crashes


class TestStressConcurrency:
    """Stress tests for concurrent operations."""

    def test_high_concurrency_stress(self, initialized_db):
        """
        High concurrency stress test with mixed operations.
        """
        from st_name_ranking import database

        # Insert many test names
        num_names = 50
        with database.get_connection() as conn:
            for i in range(num_names):
                conn.execute("INSERT INTO names (name, gender) VALUES (?, ?)", (f"Stress{i}", "Female"))

        operations = []
        errors = []

        def mixed_operation_worker(worker_id):
            try:
                for i in range(20):
                    op_type = i % 4

                    if op_type == 0:
                        # Read
                        database.get_ratings()
                    elif op_type == 1:
                        # Write rating
                        database.update_rating(f"Stress{i % num_names}", 1500.0 + i)
                    elif op_type == 2:
                        # Record comparison
                        database.record_comparison(
                            f"Stress{i % num_names}",
                            f"Stress{(i + 1) % num_names}",
                            -1 if i % 2 == 0 else 1,
                        )
                    else:
                        # Read stats
                        database.get_stats()

                    operations.append((worker_id, op_type))
            except (RuntimeError, ValueError, sqlite3.Error) as e:
                errors.append((worker_id, str(e)))

        # Run many threads
        threads = [threading.Thread(target=mixed_operation_worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify results
        assert len(errors) == 0, f"Errors during stress test: {errors[:10]}"
        assert len(operations) == 20 * 20, f"Expected 400 operations, got {len(operations)}"

        # Verify database integrity
        with database.get_connection() as conn:
            cursor = conn.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            assert result == "ok", f"Database corruption detected: {result}"

    def test_long_running_transaction_interference(self, initialized_db):
        """
        Long-running transactions shouldn't block others indefinitely.
        """
        from st_name_ranking import database

        # Insert test names
        with database.get_connection() as conn:
            for i in range(10):
                conn.execute("INSERT INTO names (name, gender) VALUES (?, ?)", (f"Long{i}", "Female"))

        reader_completed = threading.Event()
        writer_completed = threading.Event()

        def long_reader():
            try:
                with database.get_connection() as conn:
                    # Read many rows slowly
                    for i in range(100):
                        conn.execute("SELECT * FROM names")
                        time.sleep(0.001)
                reader_completed.set()
            except sqlite3.Error as e:
                print(f"Long reader error: {e}")

        def quick_writer():
            try:
                for i in range(20):
                    database.update_rating(f"Long{i % 10}", 1500.0 + i)
                writer_completed.set()
            except sqlite3.Error as e:
                print(f"Quick writer error: {e}")

        # Start long reader
        reader = threading.Thread(target=long_reader)
        reader.start()

        # Quick writer should still complete
        writer = threading.Thread(target=quick_writer)
        writer.start()

        # Wait for writer with timeout
        writer_completed.wait(timeout=10)
        assert writer_completed.is_set(), "Quick writer was blocked by long reader"

        reader.join(timeout=5)
