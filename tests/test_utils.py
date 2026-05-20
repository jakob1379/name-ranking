"""Tests for st_name_ranking.utils module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from st_name_ranking import utils
from st_name_ranking.types import NamePair


class TestPullSubmoduleUpdates:
    """Tests for pull_submodule_updates function."""

    @patch("st_name_ranking.app_actions.subprocess.run")
    @patch("st_name_ranking.app_actions.st")
    @patch("st_name_ranking.app_actions.database")
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
        result = utils.pull_submodule_updates(classify_origins=False)

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

    @patch("st_name_ranking.app_actions.subprocess.run")
    @patch("st_name_ranking.app_actions.st")
    @patch("st_name_ranking.app_actions.database")
    @patch("st_name_ranking.app_actions.time.sleep")
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
                "st_name_ranking.classify_origins": classify_origins_mock,
                "classify_origins": classify_origins_mock,
            },
        ):
            result = utils.pull_submodule_updates(classify_origins=True)

            # Verify classification was attempted
            mock_db.init_database.assert_called()
            classify_origins_mock.classify_all_names.assert_called_once_with(
                limit=None,
            )

            # Verify sleep was called for reload delay
            mock_sleep.assert_called_once_with(2)

        assert result is True

    @patch("st_name_ranking.app_actions.subprocess.run")
    @patch("st_name_ranking.app_actions.st")
    def test_failed_pull(self, mock_st, mock_run):
        """Test when git pull fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Permission denied"
        mock_run.return_value = mock_result

        result = utils.pull_submodule_updates()

        # Verify error toast
        mock_st.toast.assert_called_with(
            "Failed to pull submodule: Permission denied",
            icon="❌",
            duration="long",
        )
        assert result is False

    @patch("st_name_ranking.app_actions.subprocess.run")
    @patch("st_name_ranking.app_actions.st")
    @patch("st_name_ranking.app_actions.database")
    def test_sync_error(self, mock_db, mock_st, mock_run):
        """Test when database sync fails."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        mock_db.sync_names_with_submodule.side_effect = RuntimeError("DB error")

        result = utils.pull_submodule_updates()

        # Should still return True (sync error is caught)
        assert result is True
        mock_st.toast.assert_any_call(
            "Failed to sync names: DB error",
            icon="❌",
            duration="long",
        )

    @patch("st_name_ranking.app_actions.subprocess.run")
    @patch("st_name_ranking.app_actions.st")
    @patch("st_name_ranking.app_actions.database")
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
                "st_name_ranking.classify_origins": classify_origins_mock,
            },
        ):
            result = utils.pull_submodule_updates(classify_origins=True)

            # Should show "No names needed classification" toast
            mock_st.toast.assert_any_call(
                "No names needed classification",
                icon="ℹ️",
            )
            assert result is True

    @patch("st_name_ranking.app_actions.subprocess.run")
    @patch("st_name_ranking.app_actions.st")
    def test_general_exception(self, mock_st, mock_run):
        """Test handling of general exceptions."""
        from subprocess import SubprocessError

        mock_run.side_effect = SubprocessError("Network error")

        result = utils.pull_submodule_updates()

        mock_st.toast.assert_called_with(
            "Error pulling submodule: Network error",
            icon="❌",
            duration="long",
        )
        assert result is False


class TestSetupSessionState:
    """Tests for setup_session_state function."""

    @patch("st_name_ranking.app_actions.st")
    @patch("st_name_ranking.app_actions.initialize_or_load_ratings")
    def test_initial_setup(self, mock_init_ratings, mock_st):
        """Test setting up session state for the first time."""
        # Mock empty session state
        mock_st.session_state = {}
        names = ["Anna", "Peter", "Maria"]
        mock_ratings = {"Anna": 1500.0, "Peter": 1500.0, "Maria": 1500.0}
        mock_init_ratings.return_value = mock_ratings

        utils.setup_session_state(names)

        # Verify ratings initialized
        mock_init_ratings.assert_called_once_with(names)
        assert mock_st.session_state["ratings"] == mock_ratings
        assert mock_st.session_state["candidate_a"] == ""
        assert mock_st.session_state["candidate_b"] == ""
        assert mock_st.session_state["names"] == names

    @patch("st_name_ranking.app_actions.st")
    @patch("st_name_ranking.app_actions.initialize_or_load_ratings")
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

        utils.setup_session_state(names)

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
        assert utils.select_candidates([]) == ("", "")
        assert utils.select_candidates(["Anna"]) == ("", "")

    @patch(
        "st_name_ranking.utils.database.get_comparison_count",
        return_value=0,
    )
    def test_select_two_names(self, mock_get_comparison_count):
        """Test selecting two different names."""
        names = ["Anna", "Peter", "Maria"]
        np.random.seed(42)  # For reproducibility

        a, b = utils.select_candidates(names)

        assert a != b
        assert a in names
        assert b in names
        assert isinstance(a, str)
        assert isinstance(b, str)

    @patch(
        "st_name_ranking.utils.database.get_comparison_count",
        return_value=0,
    )
    def test_only_two_names(self, mock_get_comparison_count):
        """Test when exactly two names are available."""
        names = ["Anna", "Peter"]
        a, b = utils.select_candidates(names)

        # Should return the two names (order may vary)
        assert set([a, b]) == set(names)

    @patch("st_name_ranking.utils.logger")
    @patch("st_name_ranking.utils._select_candidates_fallback")
    @patch("st_name_ranking.utils.get_active_learning_model")
    def test_exception_fallback(self, mock_get_model, mock_fallback, mock_logger):
        """Test that exceptions trigger fallback selection."""
        names = ["Anna", "Peter", "Maria"]
        # Mock model to raise exception
        mock_model = MagicMock()
        mock_model.select_pair.side_effect = RuntimeError("Model error")
        mock_get_model.return_value = mock_model
        # Mock fallback to return a pair
        mock_fallback.return_value = ("Anna", "Peter")

        a, b = utils.select_candidates(names)

        # Verify fallback was called
        mock_fallback.assert_called_once()
        # Verify warning logged
        mock_logger.warning.assert_called_once()
        assert "Active learning pair selection failed" in mock_logger.warning.call_args[0][0]
        # Should return fallback result
        assert (a, b) == ("Anna", "Peter")


class TestSelectCandidateBatch:
    """Tests for select_candidate_batch function."""

    @patch("st_name_ranking.utils.get_active_learning_model")
    @patch("st_name_ranking.utils.get_names_features")
    def test_select_batch_success(self, mock_get_names_features, mock_get_model):
        """Test successful batch selection."""
        names = ["Anna", "Peter", "Maria", "John", "Eva"]
        # Mock features
        mock_features = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        mock_get_names_features.return_value = mock_features

        # Mock model
        mock_model = MagicMock()
        # select_top_k_pairs returns list of NamePair
        mock_model.select_top_k_pairs.return_value = [
            NamePair(idx_a=0, idx_b=1, name_a="Anna", name_b="Peter"),
            NamePair(idx_a=2, idx_b=3, name_a="Maria", name_b="John"),
            NamePair(idx_a=4, idx_b=0, name_a="Eva", name_b="Anna"),
        ]
        mock_get_model.return_value = mock_model

        pairs = utils.select_candidate_batch(names, features=mock_features, batch_size=3)

        # Verify model called with correct arguments
        assert mock_model.select_top_k_pairs.call_count == 1
        call_args = mock_model.select_top_k_pairs.call_args
        # Check arrays equality separately
        np.testing.assert_array_equal(call_args[0][0], mock_features)
        assert call_args[0][1] == names
        assert call_args[1] == {"k": 3}
        # Verify returned pairs
        assert pairs == [("Anna", "Peter"), ("Maria", "John"), ("Eva", "Anna")]

    @patch("st_name_ranking.utils.get_active_learning_model")
    @patch("st_name_ranking.utils.get_names_features")
    def test_select_batch_without_precomputed_features(self, mock_get_names_features, mock_get_model):
        """Test batch selection when features not precomputed."""
        names = ["Anna", "Peter", "Maria"]
        mock_features = np.array([[1, 2], [3, 4], [5, 6]])
        mock_get_names_features.return_value = mock_features

        mock_model = MagicMock()
        mock_model.select_top_k_pairs.return_value = [NamePair(idx_a=0, idx_b=1, name_a="Anna", name_b="Peter")]
        mock_get_model.return_value = mock_model

        pairs = utils.select_candidate_batch(names, features=None, batch_size=2)

        # Should call get_names_features
        mock_get_names_features.assert_called_once_with(names)
        assert mock_model.select_top_k_pairs.call_count == 1
        call_args = mock_model.select_top_k_pairs.call_args
        np.testing.assert_array_equal(call_args[0][0], mock_features)
        assert call_args[0][1] == names
        assert call_args[1] == {"k": 2}
        assert pairs == [("Anna", "Peter")]

    @patch("st_name_ranking.utils.get_active_learning_model")
    @patch("st_name_ranking.utils.get_names_features")
    def test_select_batch_empty_names(self, mock_get_names_features, mock_get_model):
        """Test batch selection with empty or single name list."""
        assert utils.select_candidate_batch([]) == []
        assert utils.select_candidate_batch(["Anna"]) == []
        # No model calls expected
        mock_get_model.assert_not_called()
        mock_get_names_features.assert_not_called()

    @patch("st_name_ranking.utils.get_active_learning_model")
    @patch("st_name_ranking.utils.get_names_features")
    def test_select_batch_fallback_on_exception(self, mock_get_names_features, mock_get_model):
        """Test batch selection falls back when model fails."""
        names = ["Anna", "Peter", "Maria"]
        mock_get_names_features.side_effect = RuntimeError("Feature extraction failed")
        # Mock fallback select_candidates to return a pair
        with patch("st_name_ranking.utils.select_candidates") as mock_select_candidates:
            mock_select_candidates.return_value = ("Anna", "Peter")
            pairs = utils.select_candidate_batch(names, batch_size=2)

        # Should have tried to get model
        mock_get_model.assert_called_once()
        # Should have called select_candidates as fallback
        mock_select_candidates.assert_called_once_with(names, None)
        # Should return list with single pair
        assert pairs == [("Anna", "Peter")]

    @patch("st_name_ranking.utils.get_active_learning_model")
    @patch("st_name_ranking.utils.get_names_features")
    def test_select_batch_fallback_empty_pair(self, mock_get_names_features, mock_get_model):
        """Test batch selection fallback returns empty list if pair is empty."""
        names = ["Anna", "Peter", "Maria"]
        mock_get_names_features.side_effect = RuntimeError("Feature extraction failed")
        with patch("st_name_ranking.utils.select_candidates") as mock_select_candidates:
            mock_select_candidates.return_value = ("", "")  # empty pair
            pairs = utils.select_candidate_batch(names, batch_size=2)

        # Should return empty list
        assert pairs == []


class TestSelectCandidatesFallback:
    """Tests for _select_candidates_fallback function."""

    @patch("st_name_ranking.utils.database.get_comparison_count")
    @patch("st_name_ranking.utils.phonetic_similarity")
    def test_fallback_selection(self, mock_phonetic_similarity, mock_get_comparison_count):
        """Test fallback selection with comparison counts and phonetic similarity."""
        names = ["Anna", "Peter", "Maria", "John"]
        # Mock comparison counts
        mock_get_comparison_count.side_effect = lambda name: {"Anna": 5, "Peter": 2, "Maria": 0, "John": 1}[name]
        # Mock phonetic similarity (return constant for simplicity)
        mock_phonetic_similarity.return_value = 0.5

        # Set random seed for reproducibility
        np.random.seed(42)
        a, b = utils._select_candidates_fallback(names)

        # Should return two distinct names
        assert a != b
        assert a in names
        assert b in names
        # Verify counts were queried for each name
        assert mock_get_comparison_count.call_count == len(names)
        # Verify phonetic similarity called for each pair evaluated (100 times max)
        # At least some calls
        assert mock_phonetic_similarity.call_count > 0

    @patch("st_name_ranking.utils.database.get_comparison_count")
    @patch("st_name_ranking.utils.phonetic_similarity")
    def test_fallback_only_two_names(self, mock_phonetic_similarity, mock_get_comparison_count):
        """Test fallback with exactly two names."""
        names = ["Anna", "Peter"]
        mock_get_comparison_count.return_value = 0
        mock_phonetic_similarity.return_value = 0.3

        a, b = utils._select_candidates_fallback(names)

        # Should return both names (order may vary)
        assert set([a, b]) == set(names)

    @patch("st_name_ranking.utils.database.get_comparison_count")
    @patch("st_name_ranking.utils.phonetic_similarity")
    def test_fallback_empty_or_single_name(self, mock_phonetic_similarity, mock_get_comparison_count):
        """Test fallback with empty or single name list."""
        assert utils._select_candidates_fallback([]) == ("", "")
        assert utils._select_candidates_fallback(["Anna"]) == ("", "")
        # No database or similarity calls
        mock_get_comparison_count.assert_not_called()
        mock_phonetic_similarity.assert_not_called()

    @patch("st_name_ranking.utils.database.get_comparison_count")
    @patch("st_name_ranking.utils.phonetic_similarity")
    def test_fallback_random_fallback(self, mock_phonetic_similarity, mock_get_comparison_count):
        """Test fallback when no pair selected (should fallback to random)."""
        names = ["Anna", "Peter", "Maria"]
        mock_get_comparison_count.return_value = 0
        mock_phonetic_similarity.return_value = 0.5  # Ensure returns float
        # Make phonetic similarity return negative score? Actually pair score is sum of utilities + phonetic.
        # Utilities are 1/(count+1) = 1. So pair score > 0 always. So best_pair will be selected.
        # To trigger random fallback, we need best_pair to remain ("", ""). That happens if n_pairs = 0.
        # n_pairs is min(100, len(names)*(len(names)-1)//2). With 3 names, n_pairs = 3.
        # We'll mock random choice to ensure it works.
        with patch("numpy.random.default_rng") as mock_rng:
            mock_rng.return_value.choice.return_value = np.array([0, 1])
            a, b = utils._select_candidates_fallback(names)
            # Should have called random choice
            mock_rng.return_value.choice.assert_called()


class TestSyncNamesFromSubmodule:
    """Tests for sync_names_from_submodule function."""

    @patch("st_name_ranking.app_actions.st")
    @patch("st_name_ranking.app_actions.database")
    def test_successful_sync(self, mock_db, mock_st):
        """Test successful sync with new names."""
        mock_db.sync_names_with_submodule.return_value = 5

        result = utils.sync_names_from_submodule()

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

    @patch("st_name_ranking.app_actions.st")
    @patch("st_name_ranking.app_actions.database")
    def test_no_new_names(self, mock_db, mock_st):
        """Test sync when no new names to add."""
        mock_db.sync_names_with_submodule.return_value = 0

        result = utils.sync_names_from_submodule()

        mock_st.toast.assert_called_with(
            "Database already up to date with submodule",
            icon="ℹ️",
        )
        assert result == 0

    @patch("st_name_ranking.app_actions.st")
    @patch("st_name_ranking.app_actions.database")
    def test_sync_error(self, mock_db, mock_st):
        """Test handling sync errors."""
        mock_db.sync_names_with_submodule.side_effect = RuntimeError("DB error")

        result = utils.sync_names_from_submodule()

        mock_st.toast.assert_called_with(
            "Failed to sync names: DB error",
            icon="❌",
            duration="long",
        )
        assert result == 0


class TestUpdatePreferenceAndSave:
    """Tests for update_preference_and_save function (uses Bradley-Terry model)."""

    @patch("st_name_ranking.model_service.record_comparison_instant")
    def test_nonblocking_update_records_preference(self, mock_record_comparison):
        """Default path records the preference and returns an unchanged ratings copy."""
        ratings = {"Anna": 1500.0, "Peter": 1500.0}

        result = utils.update_preference_and_save(ratings, "Anna", "Peter")

        mock_record_comparison.assert_called_once_with("Anna", "Peter", -1, blocking=False)
        assert result == ratings
        assert result is not ratings

    @patch("st_name_ranking.model_service.database.update_rating")
    @patch("st_name_ranking.model_service._compute_rating_for_name")
    @patch("st_name_ranking.model_service.record_comparison_instant")
    def test_blocking_update_refreshes_ratings(
        self,
        mock_record_comparison,
        mock_compute_rating,
        mock_update_rating,
    ):
        ratings = {"Anna": 1500.0, "Peter": 1500.0}
        mock_compute_rating.side_effect = [1516.0, 1484.0]

        result = utils.update_preference_and_save(ratings, "Anna", "Peter", blocking=True)

        mock_record_comparison.assert_called_once_with("Anna", "Peter", -1, blocking=True)
        mock_compute_rating.assert_any_call("Anna")
        mock_compute_rating.assert_any_call("Peter")
        mock_update_rating.assert_any_call("Anna", 1516.0)
        mock_update_rating.assert_any_call("Peter", 1484.0)
        assert result == {"Anna": 1516.0, "Peter": 1484.0}

    @patch("st_name_ranking.model_service.database.update_rating")
    @patch("st_name_ranking.model_service._compute_rating_for_name")
    @patch("st_name_ranking.model_service.record_comparison_instant")
    def test_blocking_update_compute_failure(
        self,
        mock_record_comparison,
        mock_compute_rating,
        mock_update_rating,
    ):
        ratings = {"Anna": 1500.0, "Peter": 1500.0}
        mock_compute_rating.side_effect = RuntimeError("Computation error")

        result = utils.update_preference_and_save(ratings, "Anna", "Peter", blocking=True)

        mock_record_comparison.assert_called_once_with("Anna", "Peter", -1, blocking=True)
        assert result == ratings
        assert result is not ratings
        mock_update_rating.assert_not_called()


class TestRecordComparisonInstant:
    """Tests for the explicit model update status contract."""

    @patch("st_name_ranking.model_service._update_ratings_from_model")
    @patch("st_name_ranking.model_service._update_model_sync")
    @patch("st_name_ranking.model_service.database.record_comparison")
    def test_blocking_status_reports_model_refresh_failure(
        self,
        mock_record_comparison,
        mock_update_model,
        mock_update_ratings,
    ):
        mock_update_model.return_value = False
        mock_update_ratings.return_value = True

        status = utils.record_comparison_instant("Anna", "Peter", -1, blocking=True)

        mock_record_comparison.assert_called_once_with("Anna", "Peter", -1)
        assert status.recorded is True
        assert status.model_updated is False
        assert status.ratings_fresh is True
        assert status.fallback_used is True
        assert status.error == "model or rating refresh failed"


class TestUpdatePreferenceDrawAndSave:
    """Tests for update_preference_draw_and_save function (uses Bradley-Terry model)."""

    @patch("st_name_ranking.model_service.database.update_rating")
    @patch("st_name_ranking.model_service._compute_rating_for_name")
    @patch("st_name_ranking.model_service.record_comparison_instant")
    def test_successful_update(
        self,
        mock_record_comparison,
        mock_compute_rating,
        mock_update_rating,
    ):
        """Test successful draw update and save."""
        ratings = {"Anna": 1500.0, "Peter": 1500.0}
        # Mock computed ratings (draw may change both ratings slightly)
        mock_compute_rating.side_effect = [1492.0, 1508.0]  # player_a, player_b

        result = utils.update_preference_draw_and_save(ratings, "Anna", "Peter", blocking=True)

        mock_record_comparison.assert_called_once_with("Anna", "Peter", 0, blocking=True)
        # Check compute rating called for both names
        assert mock_compute_rating.call_count == 2
        mock_compute_rating.assert_any_call("Anna")
        mock_compute_rating.assert_any_call("Peter")
        # Check database updates
        assert mock_update_rating.call_count == 2
        mock_update_rating.assert_any_call("Anna", 1492.0)
        mock_update_rating.assert_any_call("Peter", 1508.0)
        # Check returned ratings
        assert result == {"Anna": 1492.0, "Peter": 1508.0}

    @patch("st_name_ranking.model_service.database.update_rating")
    @patch("st_name_ranking.model_service._compute_rating_for_name")
    @patch("st_name_ranking.model_service.record_comparison_instant")
    def test_save_failure(
        self,
        mock_record_comparison,
        mock_compute_rating,
        mock_update_rating,
    ):
        """Test when ratings computation fails."""
        ratings = {"Anna": 1500.0, "Peter": 1500.0}
        mock_compute_rating.side_effect = RuntimeError("Computation error")

        result = utils.update_preference_draw_and_save(ratings, "Anna", "Peter", blocking=True)

        assert result == ratings
        assert result is not ratings
        mock_record_comparison.assert_called_once_with("Anna", "Peter", 0, blocking=True)
        # compute_rating was called (and failed)
        assert mock_compute_rating.call_count == 1
        # database.update_rating should NOT be called
        assert mock_update_rating.call_count == 0


class TestUpdatePreferenceDownAndSave:
    """Tests for update_preference_down_and_save function (uses Bradley-Terry model)."""

    @patch("st_name_ranking.model_service.database.update_rating")
    @patch("st_name_ranking.model_service._compute_rating_for_name")
    @patch("st_name_ranking.model_service.record_comparison_instant")
    def test_successful_update(
        self,
        mock_record_comparison,
        mock_compute_rating,
        mock_update_rating,
    ):
        """Test successful down update and save."""
        ratings = {"Anna": 1500.0, "Peter": 1500.0}
        # Mock computed ratings (both may decrease)
        mock_compute_rating.side_effect = [1480.0, 1470.0]  # player_a, player_b

        result = utils.update_preference_down_and_save(ratings, "Anna", "Peter", blocking=True)

        mock_record_comparison.assert_called_once_with("Anna", "Peter", 2, blocking=True)
        # Check compute rating called for both names
        assert mock_compute_rating.call_count == 2
        mock_compute_rating.assert_any_call("Anna")
        mock_compute_rating.assert_any_call("Peter")
        # Check database updates
        assert mock_update_rating.call_count == 2
        mock_update_rating.assert_any_call("Anna", 1480.0)
        mock_update_rating.assert_any_call("Peter", 1470.0)
        # Check returned ratings
        assert result == {"Anna": 1480.0, "Peter": 1470.0}

    @patch("st_name_ranking.model_service.database.update_rating")
    @patch("st_name_ranking.model_service._compute_rating_for_name")
    @patch("st_name_ranking.model_service.record_comparison_instant")
    def test_save_failure(
        self,
        mock_record_comparison,
        mock_compute_rating,
        mock_update_rating,
    ):
        """Test when ratings computation fails."""
        ratings = {"Anna": 1500.0, "Peter": 1500.0}
        mock_compute_rating.side_effect = RuntimeError("Computation error")

        result = utils.update_preference_down_and_save(ratings, "Anna", "Peter", blocking=True)

        assert result == ratings
        assert result is not ratings
        mock_record_comparison.assert_called_once_with("Anna", "Peter", 2, blocking=True)
        # compute_rating was called (and failed)
        assert mock_compute_rating.call_count == 1
        # database.update_rating should NOT be called
        assert mock_update_rating.call_count == 0


class TestGetNameFeatures:
    """Tests for get_name_features function."""

    @patch("st_name_ranking.pair_selection.database.get_connection")
    @patch("st_name_ranking.pair_selection.get_feature_extractor")
    def test_name_found_in_database(self, mock_get_feature_extractor, mock_get_connection):
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
        mock_get_feature_extractor.return_value = mock_extractor

        result = utils.get_name_features("Anna")

        # Verify database query
        mock_conn.execute.assert_called_once_with(
            "SELECT gender, origin_region FROM names WHERE name = ?",
            ("Anna",),
        )
        # Verify extractor called with correct gender and origin
        mock_extractor.extract.assert_called_once_with("Anna", "Female", "Nordic")
        # Verify result
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    @patch("st_name_ranking.pair_selection.database.get_connection")
    @patch("st_name_ranking.pair_selection.get_feature_extractor")
    def test_name_not_found_in_database(self, mock_get_feature_extractor, mock_get_connection):
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
        mock_get_feature_extractor.return_value = mock_extractor

        result = utils.get_name_features("UnknownName")

        mock_conn.execute.assert_called_once_with(
            "SELECT gender, origin_region FROM names WHERE name = ?",
            ("UnknownName",),
        )
        # Should call extract with None, None
        mock_extractor.extract.assert_called_once_with("UnknownName", None, None)
        np.testing.assert_array_equal(result, np.array([0.0, 0.0, 0.0]))


class TestComputeRatingForName:
    """Tests for _compute_rating_for_name function."""

    @patch("st_name_ranking.model_service.get_active_learning_model")
    @patch("st_name_ranking.model_service.get_name_features")
    def test_compute_rating(self, mock_get_name_features, mock_get_model):
        """Test rating computation."""
        # Mock features and model
        mock_features = np.array([1.0, 2.0])
        mock_get_name_features.return_value = mock_features

        mock_model = MagicMock()
        mock_model.get_utility.return_value = np.array([0.32])  # utility
        mock_get_model.return_value = mock_model

        rating = utils._compute_rating_for_name("Anna")

        # Verify calls
        mock_get_name_features.assert_called_once_with("Anna")
        # Check get_utility called with correct arguments (array comparison)
        assert mock_model.get_utility.call_count == 1
        call_args = mock_model.get_utility.call_args
        np.testing.assert_array_equal(call_args[0][0], mock_features.reshape(1, -1))
        # rating = 1500 + utility * 500 = 1500 + 0.32*500 = 1660.0
        assert rating == 1660.0


class TestUpdateRatingsFromModel:
    """Tests for _update_ratings_from_model function."""

    @patch("st_name_ranking.model_service.database.update_ratings_batch_values")
    @patch("st_name_ranking.model_service.get_names_features")
    @patch("st_name_ranking.model_service.database.get_connection")
    @patch("st_name_ranking.model_service.get_active_learning_model")
    def test_successful_update(
        self,
        mock_get_model,
        mock_get_connection,
        mock_get_names_features,
        mock_update_ratings_batch_values,
    ):
        """Test successful update of ratings from model."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_conn.execute.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [("Anna",), ("Peter",), ("Maria",)]
        mock_get_connection.return_value = mock_conn

        # Mock model
        mock_model = MagicMock()
        mock_model.get_utility.return_value = np.array([0.2, -0.1, 0.3])  # utilities
        mock_get_model.return_value = mock_model

        # Mock features
        mock_features = np.array([[1, 2], [3, 4], [5, 6]])
        mock_get_names_features.return_value = mock_features

        # Call function
        utils._update_ratings_from_model()

        # Verify database query
        mock_conn.execute.assert_called_once_with("SELECT name FROM names")
        # Verify features extracted for all names
        mock_get_names_features.assert_called_once_with(["Anna", "Peter", "Maria"])
        # Verify model utilities computed
        mock_model.get_utility.assert_called_once_with(mock_features)
        # Verify batch update
        expected_ratings = {
            "Anna": 1500 + 0.2 * 500,  # 1600.0
            "Peter": 1500 + (-0.1) * 500,  # 1450.0
            "Maria": 1500 + 0.3 * 500,  # 1650.0
        }
        mock_update_ratings_batch_values.assert_called_once_with(expected_ratings)

    @patch("st_name_ranking.model_service.database.update_ratings_batch_values")
    @patch("st_name_ranking.model_service.get_names_features")
    @patch("st_name_ranking.model_service.database.get_connection")
    @patch("st_name_ranking.model_service.get_active_learning_model")
    def test_empty_names(
        self,
        mock_get_model,
        mock_get_connection,
        mock_get_names_features,
        mock_update_ratings_batch_values,
    ):
        """Test when no names in database."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_conn.execute.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []  # empty
        mock_get_connection.return_value = mock_conn

        utils._update_ratings_from_model()

        # Should return early after model initialization
        mock_get_model.assert_called_once()  # Model is initialized before checking names
        mock_get_names_features.assert_not_called()
        mock_update_ratings_batch_values.assert_not_called()

    @patch("st_name_ranking.model_service.database.update_ratings_batch_values")
    @patch("st_name_ranking.model_service.get_names_features")
    @patch("st_name_ranking.model_service.database.get_connection")
    @patch("st_name_ranking.model_service.get_active_learning_model")
    def test_exception_handling(
        self,
        mock_get_model,
        mock_get_connection,
        mock_get_names_features,
        mock_update_ratings_batch_values,
    ):
        """Test exception is caught and logged (no crash)."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_conn.execute.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [("Anna",)]
        mock_get_connection.return_value = mock_conn

        # Simulate error in feature extraction
        mock_get_names_features.side_effect = RuntimeError("Feature error")

        # Should not raise exception
        utils._update_ratings_from_model()

        # Verify no batch update attempted
        mock_update_ratings_batch_values.assert_not_called()


class TestModelUpdateExceptionHandling:
    """Tests for exception handling in model update functions."""

    @patch("st_name_ranking.model_service.logger")
    @patch("st_name_ranking.model_service.get_name_features")
    @patch("st_name_ranking.model_service.get_active_learning_model")
    def test_update_model_and_save_exception(self, mock_get_model, mock_get_name_features, mock_logger):
        """Test exception handling in update_model_and_save."""
        # Mock model that raises exception on update
        mock_model = MagicMock()
        mock_model.update.side_effect = RuntimeError("Update failed")
        mock_get_model.return_value = mock_model

        # Mock features
        mock_get_name_features.side_effect = [
            np.array([1, 2]),  # winner features
            np.array([3, 4]),  # loser features
        ]

        # Should not raise exception
        utils.update_model_and_save("winner", "loser")

        # Verify model.update called
        mock_model.update.assert_called_once()
        # Verify error logged (logger.exception is used)
        mock_logger.exception.assert_called_once()
        assert "Failed to update model" in mock_logger.exception.call_args[0][0]

    @patch("st_name_ranking.model_service.logger")
    @patch("st_name_ranking.model_service.get_name_features")
    @patch("st_name_ranking.model_service.get_active_learning_model")
    def test_update_model_draw_and_save_exception(self, mock_get_model, mock_get_name_features, mock_logger):
        """Test exception handling in update_model_draw_and_save."""
        mock_model = MagicMock()
        mock_model.update.side_effect = RuntimeError("Draw update failed")
        mock_get_model.return_value = mock_model

        mock_get_name_features.side_effect = [
            np.array([1, 2]),
            np.array([3, 4]),
        ]

        utils.update_model_draw_and_save("player_a", "player_b")

        mock_model.update.assert_called_once()
        mock_logger.exception.assert_called_once()
        assert "Failed to update model for draw" in mock_logger.exception.call_args[0][0]

    @patch("st_name_ranking.model_service.logger")
    @patch("st_name_ranking.model_service.get_name_features")
    @patch("st_name_ranking.model_service.get_active_learning_model")
    def test_update_model_down_and_save_exception(self, mock_get_model, mock_get_name_features, mock_logger):
        """Test exception handling in update_model_down_and_save."""
        mock_model = MagicMock()
        mock_model.update_both_disliked.side_effect = RuntimeError("Down update failed")
        mock_get_model.return_value = mock_model

        mock_get_name_features.side_effect = [
            np.array([1, 2]),
            np.array([3, 4]),
        ]

        utils.update_model_down_and_save("player_a", "player_b")

        mock_model.update_both_disliked.assert_called_once()
        mock_logger.exception.assert_called_once()
        assert "Failed to update model for both disliked" in mock_logger.exception.call_args[0][0]

    @patch("st_name_ranking.model_service.logger")
    @patch("st_name_ranking.model_service.get_name_features")
    @patch("st_name_ranking.model_service.get_active_learning_model")
    def test_update_model_and_save_success(self, mock_get_model, mock_get_name_features, mock_logger):
        """Test successful update_model_and_save."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_get_name_features.side_effect = [
            np.array([1, 2]),
            np.array([3, 4]),
        ]

        utils.update_model_and_save("winner", "loser")

        # Verify model.update called with correct arguments
        mock_model.update.assert_called_once()
        # Check that features_a and features_b were passed
        call_args = mock_model.update.call_args
        np.testing.assert_array_equal(call_args[0][0], np.array([1, 2]))
        np.testing.assert_array_equal(call_args[0][1], np.array([3, 4]))
        assert call_args[0][2] == -1  # preference for winner
        # Verify save_to_db called
        mock_model.save_to_db.assert_called_once()
        # No error logged
        mock_logger.error.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
