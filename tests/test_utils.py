"""
Tests for st_name_ranking.utils module.
"""
import time
from unittest.mock import patch, MagicMock, call

import numpy as np
import pytest
import streamlit as st

from st_name_ranking import utils, database
from st_name_ranking.data_loader import initialize_or_load_ratings, save_ratings
from st_name_ranking.elo import update_elo, update_elo_draw


class TestPullSubmoduleUpdates:
    """Tests for pull_submodule_updates function."""
    
    @patch('subprocess.run')
    @patch('st_name_ranking.utils.st')
    @patch('st_name_ranking.utils.database')
    def test_successful_pull_without_classification(self, mock_db, mock_st, mock_run):
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
    
    @patch('subprocess.run')
    @patch('st_name_ranking.utils.st')
    @patch('st_name_ranking.utils.database')
    @patch('time.sleep')
    def test_successful_pull_with_classification(self, mock_sleep, mock_db, mock_st, mock_run):
        """Test successful submodule pull with origin classification."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_run.return_value = mock_result
        mock_db.sync_names_with_submodule.return_value = 3
        
        # Mock classify_origins import and function
        ethnidata_mock = MagicMock()
        ethnidata_mock.EthniData = MagicMock()
        with patch.dict('sys.modules', {'ethnidata': ethnidata_mock}):
            with patch('st_name_ranking.classify_origins.classify_all_names') as mock_classify:
                mock_classify.return_value = 2
                
                result = utils.pull_submodule_updates(classify_origins=True)
                
                # Verify classification was attempted
                mock_db.init_database.assert_called()
                mock_classify.assert_called_once_with(limit=None)
                
                # Verify sleep was called for reload delay
                mock_sleep.assert_called_once_with(2)
            
            assert result is True
    
    @patch('subprocess.run')
    @patch('st_name_ranking.utils.st')
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
    
    @patch('subprocess.run')
    @patch('st_name_ranking.utils.st')
    @patch('st_name_ranking.utils.database')
    def test_sync_error(self, mock_db, mock_st, mock_run):
        """Test when database sync fails."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        mock_db.sync_names_with_submodule.side_effect = Exception("DB error")
        
        result = utils.pull_submodule_updates()
        
        # Should still return True (sync error is caught)
        assert result is True
        mock_st.toast.assert_any_call(
            "Failed to sync names: DB error",
            icon="❌",
            duration="long",
        )
    
    @patch('subprocess.run')
    @patch('st_name_ranking.utils.st')
    @patch('st_name_ranking.utils.database')
    def test_classification_import_error(self, mock_db, mock_st, mock_run):
        """Test when name2nat is not installed."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        mock_db.sync_names_with_submodule.return_value = 0
        
        # Simulate ImportError when importing classify_origins
        with patch('st_name_ranking.classify_origins', side_effect=ImportError, create=True):
            result = utils.pull_submodule_updates(classify_origins=True)
            
            # Should show warning toast
            mock_st.toast.assert_any_call(
                "name2nat not installed. Run: pip install name2nat",
                icon="⚠️",
            )
            assert result is True
    
    @patch('subprocess.run')
    @patch('st_name_ranking.utils.st')
    def test_general_exception(self, mock_st, mock_run):
        """Test handling of general exceptions."""
        mock_run.side_effect = Exception("Network error")
        
        result = utils.pull_submodule_updates()
        
        mock_st.toast.assert_called_with(
            "Error pulling submodule: Network error",
            icon="❌",
            duration="long",
        )
        assert result is False


class TestSetupSessionState:
    """Tests for setup_session_state function."""
    
    @patch('st_name_ranking.utils.st')
    @patch('st_name_ranking.utils.initialize_or_load_ratings')
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
    
    @patch('st_name_ranking.utils.st')
    @patch('st_name_ranking.utils.initialize_or_load_ratings')
    def test_existing_session_state(self, mock_init_ratings, mock_st):
        """Test when session state already exists."""
        # Mock existing session state
        mock_st.session_state = {
            "ratings": {"Anna": 1600.0},
            "candidate_a": "Anna",
            "candidate_b": "Peter",
            "names": ["Anna", "Peter"]
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
    
    def test_select_two_names(self):
        """Test selecting two different names."""
        names = ["Anna", "Peter", "Maria"]
        np.random.seed(42)  # For reproducibility
        
        a, b = utils.select_candidates(names)
        
        assert a != b
        assert a in names
        assert b in names
        assert isinstance(a, str)
        assert isinstance(b, str)
    
    def test_only_two_names(self):
        """Test when exactly two names are available."""
        names = ["Anna", "Peter"]
        a, b = utils.select_candidates(names)
        
        # Should return the two names (order may vary)
        assert set([a, b]) == set(names)


class TestSyncNamesFromSubmodule:
    """Tests for sync_names_from_submodule function."""
    
    @patch('st_name_ranking.utils.st')
    @patch('st_name_ranking.utils.database')
    def test_successful_sync(self, mock_db, mock_st):
        """Test successful sync with new names."""
        mock_db.sync_names_with_submodule.return_value = 5
        
        result = utils.sync_names_from_submodule()
        
        mock_db.init_database.assert_called_once()
        mock_db.sync_names_with_submodule.assert_called_once()
        mock_st.spinner.assert_called_once_with("Syncing names from submodule...")
        mock_st.toast.assert_called_with(
            "✅ Added 5 new names to database",
            icon="✅",
        )
        assert result == 5
    
    @patch('st_name_ranking.utils.st')
    @patch('st_name_ranking.utils.database')
    def test_no_new_names(self, mock_db, mock_st):
        """Test sync when no new names to add."""
        mock_db.sync_names_with_submodule.return_value = 0
        
        result = utils.sync_names_from_submodule()
        
        mock_st.toast.assert_called_with(
            "Database already up to date with submodule",
            icon="ℹ️",
        )
        assert result == 0
    
    @patch('st_name_ranking.utils.st')
    @patch('st_name_ranking.utils.database')
    def test_sync_error(self, mock_db, mock_st):
        """Test handling sync errors."""
        mock_db.sync_names_with_submodule.side_effect = Exception("DB error")
        
        result = utils.sync_names_from_submodule()
        
        mock_st.toast.assert_called_with(
            "Failed to sync names: DB error",
            icon="❌",
            duration="long",
        )
        assert result == 0


class TestUpdateEloAndSave:
    """Tests for update_elo_and_save function."""
    
    @patch('st_name_ranking.utils.save_ratings')
    @patch('st_name_ranking.utils.update_elo')
    @patch('st_name_ranking.utils.st')
    def test_successful_update(self, mock_st, mock_update_elo, mock_save_ratings):
        """Test successful Elo update and save."""
        ratings = {"Anna": 1500.0, "Peter": 1500.0}
        updated_ratings = {"Anna": 1516.0, "Peter": 1484.0}
        mock_update_elo.return_value = updated_ratings
        
        result = utils.update_elo_and_save(ratings, "Anna", "Peter")
        
        mock_update_elo.assert_called_once_with(ratings, "Anna", "Peter", 32.0)
        mock_save_ratings.assert_called_once_with(updated_ratings, names_to_update=["Anna", "Peter"])
        assert result == updated_ratings
        # No error toast
        assert not mock_st.toast.called
    
    @patch('st_name_ranking.utils.save_ratings')
    @patch('st_name_ranking.utils.update_elo')
    @patch('st_name_ranking.utils.st')
    def test_save_failure(self, mock_st, mock_update_elo, mock_save_ratings):
        """Test when save fails (should not break)."""
        ratings = {"Anna": 1500.0, "Peter": 1500.0}
        updated_ratings = {"Anna": 1516.0, "Peter": 1484.0}
        mock_update_elo.return_value = updated_ratings
        mock_save_ratings.side_effect = Exception("Save error")
        
        result = utils.update_elo_and_save(ratings, "Anna", "Peter")
        
        # Should still return updated ratings
        assert result == updated_ratings
        # Should show warning toast
        mock_st.toast.assert_called_with(
            "Failed to save ratings: Save error",
            icon="⚠️",
        )


class TestUpdateEloDrawAndSave:
    """Tests for update_elo_draw_and_save function."""
    
    @patch('st_name_ranking.utils.save_ratings')
    @patch('st_name_ranking.utils.update_elo_draw')
    @patch('st_name_ranking.utils.st')
    def test_successful_update(self, mock_st, mock_update_elo_draw, mock_save_ratings):
        """Test successful draw update and save."""
        ratings = {"Anna": 1500.0, "Peter": 1500.0}
        updated_ratings = {"Anna": 1500.0, "Peter": 1500.0}
        mock_update_elo_draw.return_value = updated_ratings
        
        result = utils.update_elo_draw_and_save(ratings, "Anna", "Peter")
        
        mock_update_elo_draw.assert_called_once_with(ratings, "Anna", "Peter", 32.0)
        mock_save_ratings.assert_called_once_with(updated_ratings, names_to_update=["Anna", "Peter"])
        assert result == updated_ratings
    
    @patch('st_name_ranking.utils.save_ratings')
    @patch('st_name_ranking.utils.update_elo_draw')
    @patch('st_name_ranking.utils.st')
    def test_save_failure(self, mock_st, mock_update_elo_draw, mock_save_ratings):
        """Test when save fails."""
        ratings = {"Anna": 1500.0, "Peter": 1500.0}
        updated_ratings = {"Anna": 1500.0, "Peter": 1500.0}
        mock_update_elo_draw.return_value = updated_ratings
        mock_save_ratings.side_effect = Exception("Save error")
        
        result = utils.update_elo_draw_and_save(ratings, "Anna", "Peter")
        
        assert result == updated_ratings
        mock_st.toast.assert_called_with(
            "Failed to save ratings: Save error",
            icon="⚠️",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
