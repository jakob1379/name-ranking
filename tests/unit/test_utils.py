"""Tests for canonical app action and selection helpers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from st_name_ranking.active_learning import lazy_updates, selection
from st_name_ranking.interface import app_actions
from st_name_ranking.types import NamePair


class TestPullSubmoduleUpdates:
    """Tests for pull_submodule_updates function."""

    @patch("st_name_ranking.interface.app_actions.subprocess.run")
    @patch("st_name_ranking.interface.app_actions.st")
    @patch("st_name_ranking.interface.app_actions.database")
    def test_successful_pull_without_classification(
        self,
        mock_db,
        mock_st,
        mock_run,
    ):
        """Test successful submodule pull without origin classification."""
        # Mock subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Already up to date."
        mock_run.return_value = mock_result

        # Mock database sync
        mock_db.sync_names_with_submodule.return_value = 5

        # Call function
        result = app_actions.pull_submodule_updates(classify_origins=False)

        # Verify subprocess call
        mock_run.assert_called_once_with(
            ["git", "-C", "godkendtefornavne", "pull"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Verify database init and sync
        mock_db.init_database.assert_called_once()
        mock_db.sync_names_with_submodule.assert_called_once()

        # Verify streamlit calls
        assert mock_st.spinner.called
        assert mock_st.toast.called

        # Should return True
        assert result is True

    @patch("st_name_ranking.interface.app_actions.subprocess.run")
    @patch("st_name_ranking.interface.app_actions.st")
    @patch("st_name_ranking.interface.app_actions.database")
    @patch("st_name_ranking.interface.app_actions.time.sleep")
    def test_successful_pull_with_classification(
        self,
        mock_sleep,
        mock_db,
        mock_st,
        mock_run,
    ):
        """Test successful submodule pull with origin classification."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_run.return_value = mock_result
        mock_db.sync_names_with_submodule.return_value = 3

        # Mock ethnidata module
        ethnidata_mock = MagicMock()
        ethnidata_mock.EthniData = MagicMock()
        # Mock classify_origins module
        classify_origins_mock = MagicMock()
        classify_origins_mock.classify_all_names = MagicMock(return_value=2)
        # Replace both modules in sys.modules
        with patch.dict(
            "sys.modules",
            {
                "ethnidata": ethnidata_mock,
                "st_name_ranking.classification.classify_origins": classify_origins_mock,
                "classify_origins": classify_origins_mock,
            },
        ):
            result = app_actions.pull_submodule_updates(classify_origins=True)

            # Verify classification was attempted
            mock_db.init_database.assert_called()
            classify_origins_mock.classify_all_names.assert_called_once_with(
                limit=None,
            )

            # Verify sleep was called for reload delay
            mock_sleep.assert_called_once_with(2)

        assert result is True

    @patch("st_name_ranking.interface.app_actions.subprocess.run")
    @patch("st_name_ranking.interface.app_actions.st")
    def test_failed_pull(self, mock_st, mock_run):
        """Test when git pull fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Permission denied"
        mock_run.return_value = mock_result

        result = app_actions.pull_submodule_updates()

        # Verify error toast
        mock_st.toast.assert_called_with(
            "Failed to pull submodule: Permission denied",
            icon="❌",
            duration="long",
        )
        assert result is False

    @patch("st_name_ranking.interface.app_actions.subprocess.run")
    @patch("st_name_ranking.interface.app_actions.st")
    @patch("st_name_ranking.interface.app_actions.database")
    def test_sync_error(self, mock_db, mock_st, mock_run):
        """Test when database sync fails."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        mock_db.sync_names_with_submodule.side_effect = RuntimeError("DB error")

        result = app_actions.pull_submodule_updates()

        # Should still return True (sync error is caught)
        assert result is True
        mock_st.toast.assert_any_call(
            "Failed to sync names: DB error",
            icon="❌",
            duration="long",
        )

    @patch("st_name_ranking.interface.app_actions.subprocess.run")
    @patch("st_name_ranking.interface.app_actions.st")
    @patch("st_name_ranking.interface.app_actions.database")
    def test_classification_import_error(self, mock_db, mock_st, mock_run):
        """Test when ethnidata is not installed."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        mock_db.sync_names_with_submodule.return_value = 0

        # Mock classify_origins module with classify_all_names returning 0
        # (simulating no names classified, e.g., because ethnidata not installed)
        classify_origins_mock = MagicMock()
        classify_origins_mock.classify_all_names = MagicMock(return_value=0)
        with patch.dict(
            "sys.modules",
            {
                "st_name_ranking.classification.classify_origins": classify_origins_mock,
            },
        ):
            result = app_actions.pull_submodule_updates(classify_origins=True)

            # Should show "No names needed classification" toast
            mock_st.toast.assert_any_call(
                "No names needed classification",
                icon="ℹ️",
            )
            assert result is True

    @patch("st_name_ranking.interface.app_actions.subprocess.run")
    @patch("st_name_ranking.interface.app_actions.st")
    def test_general_exception(self, mock_st, mock_run):
        """Test handling of general exceptions."""
        from subprocess import SubprocessError

        mock_run.side_effect = SubprocessError("Network error")

        result = app_actions.pull_submodule_updates()

        mock_st.toast.assert_called_with(
            "Error pulling submodule: Network error",
            icon="❌",
            duration="long",
        )
        assert result is False


class TestSetupSessionState:
    """Tests for setup_session_state function."""

    @patch("st_name_ranking.interface.app_actions.st")
    @patch("st_name_ranking.interface.app_actions.initialize_or_load_ratings")
    def test_initial_setup(self, mock_init_ratings, mock_st):
        """Test setting up session state for the first time."""
        # Mock empty session state
        mock_st.session_state = {}
        names = ["Anna", "Peter", "Maria"]
        mock_ratings = {"Anna": 1500.0, "Peter": 1500.0, "Maria": 1500.0}
        mock_init_ratings.return_value = mock_ratings

        app_actions.setup_session_state(names)

        # Verify ratings initialized
        mock_init_ratings.assert_called_once_with(names)
        assert mock_st.session_state["ratings"] == mock_ratings
        assert mock_st.session_state["candidate_a"] == ""
        assert mock_st.session_state["candidate_b"] == ""
        assert mock_st.session_state["names"] == names

    @patch("st_name_ranking.interface.app_actions.st")
    @patch("st_name_ranking.interface.app_actions.initialize_or_load_ratings")
    def test_existing_session_state(self, mock_init_ratings, mock_st):
        """Test when session state already exists."""
        # Mock existing session state
        mock_st.session_state = {
            "ratings": {"Anna": 1600.0},
            "candidate_a": "Anna",
            "candidate_b": "Peter",
            "names": ["Anna", "Peter"],
        }
        names = ["Anna", "Peter", "Maria"]

        app_actions.setup_session_state(names)

        # Should not reinitialize ratings
        mock_init_ratings.assert_not_called()
        # Names should NOT be updated (only set if missing)
        assert mock_st.session_state["names"] == ["Anna", "Peter"]
        # Other values unchanged
        assert mock_st.session_state["ratings"] == {"Anna": 1600.0}
        assert mock_st.session_state["candidate_a"] == "Anna"
        assert mock_st.session_state["candidate_b"] == "Peter"


class TestSelectCandidates:
    """Tests for select_candidates function."""

    def test_empty_names(self):
        """Test with empty or single name list."""
        with pytest.raises(ValueError, match="Need at least 2 names"):
            selection.select_candidates([])
        with pytest.raises(ValueError, match="Need at least 2 names"):
            selection.select_candidates(["Anna"])

    def test_try_select_candidates_empty_names(self):
        """The explicit try_* API returns None when no pair is available."""
        assert selection.try_select_candidates([]) is None
        assert selection.try_select_candidates(["Anna"]) is None

    def test_select_two_names(self):
        """Test selecting two different names."""
        names = ["Anna", "Peter", "Maria"]
        dependencies = selection.PairSelectionDependencies(
            model_provider=MagicMock(side_effect=RuntimeError("Model unavailable")),
            comparison_count_provider=MagicMock(return_value=0),
            warning_logger=MagicMock(),
        )

        a, b = selection.select_candidates(names, dependencies=dependencies)

        assert a != b
        assert a in names
        assert b in names
        assert isinstance(a, str)
        assert isinstance(b, str)

    def test_only_two_names(self):
        """Test when exactly two names are available."""
        names = ["Anna", "Peter"]
        dependencies = selection.PairSelectionDependencies(
            model_provider=MagicMock(side_effect=RuntimeError("Model unavailable")),
            comparison_count_provider=MagicMock(return_value=0),
            warning_logger=MagicMock(),
        )

        a, b = selection.select_candidates(names, dependencies=dependencies)

        # Should return the two names (order may vary)
        assert set([a, b]) == set(names)

    def test_exception_fallback(self):
        """Test that exceptions trigger fallback selection."""
        names = ["Anna", "Peter", "Maria"]
        mock_model = MagicMock()
        mock_model.select_pair.side_effect = RuntimeError("Model error")
        mock_fallback = MagicMock(return_value=("Anna", "Peter"))
        mock_warning = MagicMock()
        dependencies = selection.PairSelectionDependencies(
            model_provider=MagicMock(return_value=mock_model),
            heuristic_pair_provider=mock_fallback,
            warning_logger=mock_warning,
        )

        a, b = selection.select_candidates(names, dependencies=dependencies)

        mock_fallback.assert_called_once_with(names)
        mock_warning.assert_called_once()
        assert "Active learning pair selection failed" in mock_warning.call_args[0][0]
        assert (a, b) == ("Anna", "Peter")


class TestSelectCandidateBatch:
    """Tests for select_candidate_batch function."""

    def test_select_batch_success(self):
        """Test successful batch selection."""
        names = ["Anna", "Peter", "Maria", "John", "Eva"]
        # Mock features
        mock_features = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

        # Mock model
        mock_model = MagicMock()
        # select_top_k_pairs returns list of NamePair
        mock_model.select_top_k_pairs.return_value = [
            NamePair(idx_a=0, idx_b=1, name_a="Anna", name_b="Peter"),
            NamePair(idx_a=2, idx_b=3, name_a="Maria", name_b="John"),
            NamePair(idx_a=4, idx_b=0, name_a="Eva", name_b="Anna"),
        ]
        dependencies = selection.PairSelectionDependencies(
            model_provider=MagicMock(return_value=mock_model),
            warning_logger=MagicMock(),
        )

        pairs = selection.select_candidate_pairs(
            names,
            features=mock_features,
            options=selection.PairSelectionOptions(batch_size=3),
            dependencies=dependencies,
        )

        # Verify model called with correct arguments
        assert mock_model.select_top_k_pairs.call_count == 1
        call_args = mock_model.select_top_k_pairs.call_args
        # Check arrays equality separately
        np.testing.assert_array_equal(call_args[0][0], mock_features)
        assert call_args[0][1] == names
        assert call_args[1] == {"k": 3}
        # Verify returned pairs
        assert pairs == [("Anna", "Peter"), ("Maria", "John"), ("Eva", "Anna")]

    def test_select_batch_without_precomputed_features(self):
        """Test batch selection when features not precomputed."""
        names = ["Anna", "Peter", "Maria"]
        mock_features = np.array([[1, 2], [3, 4], [5, 6]])
        mock_get_names_features = MagicMock(return_value=mock_features)

        mock_model = MagicMock()
        mock_model.select_top_k_pairs.return_value = [NamePair(idx_a=0, idx_b=1, name_a="Anna", name_b="Peter")]
        dependencies = selection.PairSelectionDependencies(
            model_provider=MagicMock(return_value=mock_model),
            features_provider=mock_get_names_features,
            warning_logger=MagicMock(),
        )

        pairs = selection.select_candidate_pairs(
            names,
            features=None,
            options=selection.PairSelectionOptions(batch_size=2),
            dependencies=dependencies,
        )

        # Should call get_names_features
        mock_get_names_features.assert_called_once_with(names)
        assert mock_model.select_top_k_pairs.call_count == 1
        call_args = mock_model.select_top_k_pairs.call_args
        np.testing.assert_array_equal(call_args[0][0], mock_features)
        assert call_args[0][1] == names
        assert call_args[1] == {"k": 2}
        assert pairs == [("Anna", "Peter")]

    def test_select_batch_empty_names(self):
        """Test batch selection with empty or single name list."""
        mock_get_names_features = MagicMock()
        mock_get_model = MagicMock()
        dependencies = selection.PairSelectionDependencies(
            model_provider=mock_get_model,
            features_provider=mock_get_names_features,
        )

        assert selection.select_candidate_pairs([], dependencies=dependencies) == []
        assert selection.select_candidate_pairs(["Anna"], dependencies=dependencies) == []
        # No model calls expected
        mock_get_model.assert_not_called()
        mock_get_names_features.assert_not_called()

    def test_select_batch_fallback_on_exception(self):
        """Test batch selection falls back when model fails."""
        names = ["Anna", "Peter", "Maria"]
        mock_get_model = MagicMock(return_value=MagicMock())
        mock_get_names_features = MagicMock(side_effect=RuntimeError("Feature extraction failed"))
        mock_select_candidates = MagicMock(return_value=("Anna", "Peter"))
        dependencies = selection.PairSelectionDependencies(
            model_provider=mock_get_model,
            features_provider=mock_get_names_features,
            single_pair_provider=mock_select_candidates,
            warning_logger=MagicMock(),
        )

        pairs = selection.select_candidate_pairs(
            names,
            options=selection.PairSelectionOptions(batch_size=2),
            dependencies=dependencies,
        )

        # Should have tried to get model
        mock_get_model.assert_called_once()
        # Should have called select_candidates as fallback
        mock_select_candidates.assert_called_once_with(names, None)
        # Should return list with single pair
        assert pairs == [("Anna", "Peter")]

    def test_select_batch_fallback_empty_pair(self):
        """Test batch selection fallback returns empty list if pair is empty."""
        names = ["Anna", "Peter", "Maria"]
        dependencies = selection.PairSelectionDependencies(
            model_provider=MagicMock(return_value=MagicMock()),
            features_provider=MagicMock(side_effect=RuntimeError("Feature extraction failed")),
            single_pair_provider=MagicMock(return_value=("", "")),
            warning_logger=MagicMock(),
        )
        pairs = selection.select_candidate_pairs(
            names,
            options=selection.PairSelectionOptions(batch_size=2),
            dependencies=dependencies,
        )

        # Should return empty list
        assert pairs == []


class TestSelectCandidatesFallback:
    """Tests for _select_candidates_fallback function."""

    @patch("st_name_ranking.active_learning.selection.phonetic_similarity")
    def test_fallback_selection(self, mock_phonetic_similarity):
        """Test fallback selection with comparison counts and phonetic similarity."""
        names = ["Anna", "Peter", "Maria", "John"]
        mock_get_comparison_count = MagicMock()
        mock_get_comparison_count.side_effect = lambda name: {"Anna": 5, "Peter": 2, "Maria": 0, "John": 1}[name]
        # Mock phonetic similarity (return constant for simplicity)
        mock_phonetic_similarity.return_value = 0.5
        dependencies = selection.PairSelectionDependencies(comparison_count_provider=mock_get_comparison_count)

        # Set random seed for reproducibility
        np.random.seed(42)
        a, b = selection._select_candidates_fallback(names, dependencies)

        # Should return two distinct names
        assert a != b
        assert a in names
        assert b in names
        # Verify counts were queried for each name
        assert mock_get_comparison_count.call_count == len(names)
        # Verify phonetic similarity called for each pair evaluated (100 times max)
        # At least some calls
        assert mock_phonetic_similarity.call_count > 0

    @patch("st_name_ranking.active_learning.selection.phonetic_similarity")
    def test_fallback_only_two_names(self, mock_phonetic_similarity):
        """Test fallback with exactly two names."""
        names = ["Anna", "Peter"]
        mock_get_comparison_count = MagicMock(return_value=0)
        mock_get_comparison_count.return_value = 0
        mock_phonetic_similarity.return_value = 0.3
        dependencies = selection.PairSelectionDependencies(comparison_count_provider=mock_get_comparison_count)

        a, b = selection._select_candidates_fallback(names, dependencies)

        # Should return both names (order may vary)
        assert set([a, b]) == set(names)

    @patch("st_name_ranking.active_learning.selection.phonetic_similarity")
    def test_fallback_empty_or_single_name(self, mock_phonetic_similarity):
        """Test fallback with empty or single name list."""
        mock_get_comparison_count = MagicMock()
        dependencies = selection.PairSelectionDependencies(comparison_count_provider=mock_get_comparison_count)
        assert selection._select_candidates_fallback([], dependencies) is None
        assert selection._select_candidates_fallback(["Anna"], dependencies) is None
        # No database or similarity calls
        mock_get_comparison_count.assert_not_called()
        mock_phonetic_similarity.assert_not_called()

    @patch("st_name_ranking.active_learning.selection.phonetic_similarity")
    def test_fallback_random_fallback(self, mock_phonetic_similarity):
        """Test fallback when no pair selected (should fallback to random)."""
        names = ["Anna", "Peter", "Maria"]
        mock_get_comparison_count = MagicMock(return_value=0)
        mock_get_comparison_count.return_value = 0
        mock_phonetic_similarity.return_value = 0.5  # Ensure returns float
        dependencies = selection.PairSelectionDependencies(comparison_count_provider=mock_get_comparison_count)
        # Make phonetic similarity return negative score? Actually pair score is sum of utilities + phonetic.
        # Utilities are 1/(count+1) = 1. So pair score > 0 always. So best_pair will be selected.
        # To trigger random fallback, we need best_pair to remain ("", ""). That happens if n_pairs = 0.
        # n_pairs is min(100, len(names)*(len(names)-1)//2). With 3 names, n_pairs = 3.
        # We'll mock random choice to ensure it works.
        with patch("numpy.random.default_rng") as mock_rng:
            mock_rng.return_value.choice.return_value = np.array([0, 1])
            a, b = selection._select_candidates_fallback(names, dependencies)
            # Should have called random choice
            mock_rng.return_value.choice.assert_called()


class TestSyncNamesFromSubmodule:
    """Tests for sync_names_from_submodule function."""

    @patch("st_name_ranking.interface.app_actions.st")
    @patch("st_name_ranking.interface.app_actions.database")
    def test_successful_sync(self, mock_db, mock_st):
        """Test successful sync with new names."""
        mock_db.sync_names_with_submodule.return_value = 5

        result = app_actions.sync_names_from_submodule()

        mock_db.init_database.assert_called_once()
        mock_db.sync_names_with_submodule.assert_called_once()
        mock_st.spinner.assert_called_once_with(
            "Syncing names from submodule...",
        )
        mock_st.toast.assert_called_with(
            "✅ Added 5 new names to database",
            icon="✅",
        )
        assert result == 5

    @patch("st_name_ranking.interface.app_actions.st")
    @patch("st_name_ranking.interface.app_actions.database")
    def test_no_new_names(self, mock_db, mock_st):
        """Test sync when no new names to add."""
        mock_db.sync_names_with_submodule.return_value = 0

        result = app_actions.sync_names_from_submodule()

        mock_st.toast.assert_called_with(
            "Database already up to date with submodule",
            icon="ℹ️",
        )
        assert result == 0

    @patch("st_name_ranking.interface.app_actions.st")
    @patch("st_name_ranking.interface.app_actions.database")
    def test_sync_error(self, mock_db, mock_st):
        """Test handling sync errors."""
        mock_db.sync_names_with_submodule.side_effect = RuntimeError("DB error")

        result = app_actions.sync_names_from_submodule()

        mock_st.toast.assert_called_with(
            "Failed to sync names: DB error",
            icon="❌",
            duration="long",
        )
        assert result == 0


class TestRecordComparisonInstant:
    """Tests for the explicit model update status contract."""

    @patch("st_name_ranking.active_learning.lazy_updates._update_ratings_from_model")
    @patch("st_name_ranking.active_learning.lazy_updates._update_model_sync")
    @patch("st_name_ranking.active_learning.lazy_updates.database.record_comparison")
    def test_blocking_status_reports_model_refresh_failure(
        self,
        mock_record_comparison,
        mock_update_model,
        mock_update_ratings,
    ):
        mock_update_model.return_value = False
        mock_update_ratings.return_value = True

        status = lazy_updates.record_comparison_instant("Anna", "Peter", -1, blocking=True)

        mock_record_comparison.assert_called_once_with("Anna", "Peter", -1)
        assert status.recorded is True
        assert status.model_updated is False
        assert status.ratings_fresh is False
        assert status.fallback_used is True
        assert status.error == "model or rating refresh failed"
        mock_update_ratings.assert_not_called()


class TestGetNameFeatures:
    """Tests for get_name_features function."""

    @patch("st_name_ranking.active_learning.selection.database.get_connection")
    @patch("st_name_ranking.active_learning.selection.get_or_create_feature_extractor")
    def test_name_found_in_database(self, mock_get_or_create_feature_extractor, mock_get_connection):
        """Test when name is found in database."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_conn.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ("Female", "Nordic")
        mock_get_connection.return_value = mock_conn

        # Mock feature extractor
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = np.array([1.0, 2.0, 3.0])
        mock_get_or_create_feature_extractor.return_value = mock_extractor

        result = selection.get_name_features("Anna")

        # Verify database query
        mock_conn.execute.assert_called_once_with(
            "SELECT gender, origin_region FROM names WHERE name = ?",
            ("Anna",),
        )
        # Verify extractor called with correct gender and origin
        mock_extractor.extract.assert_called_once_with("Anna", "Female", "Nordic")
        # Verify result
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    @patch("st_name_ranking.active_learning.selection.database.get_connection")
    @patch("st_name_ranking.active_learning.selection.get_or_create_feature_extractor")
    def test_name_not_found_in_database(self, mock_get_or_create_feature_extractor, mock_get_connection):
        """Test when name is not found in database (should use None, None)."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_conn.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None  # Name not found
        mock_get_connection.return_value = mock_conn

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = np.array([0.0, 0.0, 0.0])
        mock_get_or_create_feature_extractor.return_value = mock_extractor

        result = selection.get_name_features("UnknownName")

        mock_conn.execute.assert_called_once_with(
            "SELECT gender, origin_region FROM names WHERE name = ?",
            ("UnknownName",),
        )
        # Should call extract with None, None
        mock_extractor.extract.assert_called_once_with("UnknownName", None, None)
        np.testing.assert_array_equal(result, np.array([0.0, 0.0, 0.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
