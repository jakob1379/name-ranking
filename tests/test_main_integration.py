"""Integration tests for main.py Streamlit application."""

import json
from unittest.mock import MagicMock, patch

from st_name_ranking import main as main_module


class MockSessionState(dict):
    """Mock Streamlit session_state that supports both dict and attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as err:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from err

    def __setattr__(self, key, value):
        self[key] = value

    def __contains__(self, key):
        # Support 'key in session_state'
        return super().__contains__(key)


class TestMainIntegration:
    """Integration tests for main module."""

    def test_main_no_names_loaded(self):
        """Test main function when no names are loaded."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.set_page_config.return_value = None
        mock_st.title.return_value = None
        mock_st.sidebar = MagicMock()
        mock_st.session_state = MockSessionState()
        mock_st.rerun = MagicMock()
        mock_st.spinner = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()
        mock_st.toast = MagicMock()
        mock_st.error = MagicMock()
        mock_st.info = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.checkbox = MagicMock(return_value=False)
        mock_st.success = MagicMock()
        mock_st.columns = MagicMock(side_effect=lambda n: [MagicMock() for _ in range(n)])
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.button = MagicMock(return_value=False)
        mock_st.pills = MagicMock(return_value="All")
        mock_st.multiselect = MagicMock(return_value=[])

        # Mock database operations
        mock_stats = {
            "total_names": 0,
            "classified_names": 0,
            "rated_names": 0,
            "origin_distribution": {},
        }

        with (
            patch("st_name_ranking.main.st", mock_st),
            patch("st_name_ranking.main.database") as mock_db,
            patch("st_name_ranking.main.load_names_by_gender") as mock_load_names,
            patch("st_name_ranking.main.setup_session_state"),
        ):
            # Setup mocks
            mock_db.init_database.return_value = None
            mock_db.get_stats.return_value = mock_stats
            mock_db.get_all_origin_regions.return_value = ["Nordic", "European"]
            mock_db.load_user_setting.return_value = "[]"
            mock_db.save_user_setting.return_value = None

            # Simulate no names loaded
            mock_load_names.return_value = None

            # Call main function
            main_module.main()

            # Verify error message shown
            mock_st.error.assert_called_with("No names loaded in the database.")
            mock_st.info.assert_called()

    def test_main_names_loaded_with_filters(self):
        """Test main function when names are loaded and filters applied."""
        mock_st = MagicMock()
        mock_st.set_page_config.return_value = None
        mock_st.title.return_value = None
        mock_st.sidebar = MagicMock()
        mock_st.session_state = MockSessionState(
            {
                "all_names_data": {
                    "All": ["Anna", "Peter", "Maria"],
                    "Male": ["Peter"],
                    "Female": ["Anna", "Maria"],
                    "Unisex": [],
                },
                "all_names": ["Anna", "Peter", "Maria"],
                "gender_filter": "All",
                "origin_filter": [],
                "names": ["Anna", "Peter", "Maria"],
                "ratings": {"Anna": 1500.0, "Peter": 1500.0, "Maria": 1500.0},
            },
        )
        mock_st.rerun = MagicMock()
        mock_st.spinner = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()
        mock_st.toast = MagicMock()
        mock_st.pills = MagicMock(return_value="All")
        mock_st.multiselect = MagicMock(return_value=[])
        mock_st.button = MagicMock(return_value=False)
        mock_st.columns = MagicMock(side_effect=lambda n: [MagicMock() for _ in range(n)])
        mock_st.caption = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        # Mock tab context managers
        tab1 = MagicMock()
        tab2 = MagicMock()
        tab3 = MagicMock()
        tab1.__enter__ = MagicMock()
        tab1.__exit__ = MagicMock()
        tab2.__enter__ = MagicMock()
        tab2.__exit__ = MagicMock()
        tab3.__enter__ = MagicMock()
        tab3.__exit__ = MagicMock()
        mock_st.tabs.return_value = [tab1, tab2, tab3]

        with (
            patch("st_name_ranking.main.st", mock_st),
            patch("st_name_ranking.main.database") as mock_db,
            patch("st_name_ranking.main.load_names_by_gender") as mock_load_names,
            patch("st_name_ranking.main.setup_session_state"),
            patch("st_name_ranking.main.save_ratings"),
            patch("st_name_ranking.main.initialize_ratings"),
            patch("st_name_ranking.main.render_binary_filter") as mock_render_binary_filter,
            patch("st_name_ranking.main.render_tournament") as mock_render_tournament,
            patch("st_name_ranking.main.render_similarity") as mock_render_similarity,
            patch("st_name_ranking.main.sync_names_from_submodule"),
        ):
            # Setup database mocks
            mock_db.init_database.return_value = None
            mock_db.get_stats.return_value = {
                "total_names": 3,
                "classified_names": 2,
                "rated_names": 1,
                "origin_distribution": {"Nordic": 2, "European": 1},
            }
            mock_db.get_all_origin_regions.return_value = ["Nordic", "European"]
            mock_db.load_user_setting.return_value = "[]"
            mock_db.save_user_setting.return_value = None
            mock_db.get_names_by_filters.return_value = ["Anna", "Peter", "Maria"]

            # Mock load_names_by_gender to return data (but session state already has it)
            mock_load_names.return_value = mock_st.session_state["all_names_data"]

            # Call main function
            main_module.main()

            # Verify tabs were NOT created (button-based tab system)
            mock_st.tabs.assert_not_called()

            # Verify only active tab's render function was called
            mock_render_binary_filter.assert_called_with(["Anna", "Peter", "Maria"])
            mock_render_tournament.assert_not_called()
            mock_render_similarity.assert_not_called()

    def test_main_sync_names_button(self):
        """Test sync names button functionality."""
        mock_st = MagicMock()
        mock_st.set_page_config.return_value = None
        mock_st.title.return_value = None
        mock_st.sidebar = MagicMock()
        mock_st.session_state = MockSessionState()
        mock_st.rerun = MagicMock()
        mock_st.spinner = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()
        mock_st.toast = MagicMock()
        mock_st.error = MagicMock()
        mock_st.info = MagicMock()
        mock_st.checkbox = MagicMock(return_value=False)
        mock_st.pills = MagicMock(return_value="All")
        mock_st.multiselect = MagicMock(return_value=[])
        mock_st.caption = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()

        # Mock button to return True for sync button (first call)
        button_calls = [True, False, False, False]  # First button is sync
        mock_st.button = MagicMock(side_effect=button_calls)
        mock_st.columns = MagicMock(side_effect=lambda n: [MagicMock() for _ in range(n)])

        with (
            patch("st_name_ranking.main.st", mock_st),
            patch("st_name_ranking.main.database") as mock_db,
            patch("st_name_ranking.main.load_names_by_gender"),
            patch("st_name_ranking.main.setup_session_state"),
            patch("st_name_ranking.main.sync_names_from_submodule") as mock_sync,
            patch("random.choice") as mock_random_choice,
        ):
            mock_random_choice.return_value = "All"
            mock_db.init_database.return_value = None
            mock_db.get_stats.return_value = {
                "total_names": 0,
                "classified_names": 0,
                "rated_names": 0,
                "origin_distribution": {},
            }
            mock_db.get_all_origin_regions.return_value = ["Nordic", "European"]
            mock_db.load_user_setting.return_value = "[]"

            # Mock sync to return inserted count
            mock_sync.return_value = 5

            # Call main function
            main_module.main()

            # Verify sync was called
            mock_sync.assert_called_once()
            # Should trigger rerun (may be called multiple times due to session state updates)
            assert mock_st.rerun.call_count >= 1

    def test_main_save_ratings_button(self):
        """Test save ratings button functionality."""
        mock_st = MagicMock()
        mock_st.set_page_config.return_value = None
        mock_st.title.return_value = None
        mock_st.sidebar = MagicMock()
        mock_st.session_state = MockSessionState(
            {
                "all_names_data": {"All": ["Anna", "Peter"]},
                "all_names": ["Anna", "Peter"],
                "names": ["Anna", "Peter"],
                "ratings": {"Anna": 1600.0, "Peter": 1400.0},
            },
        )
        mock_st.rerun = MagicMock()
        mock_st.spinner = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()
        mock_st.toast = MagicMock()
        mock_st.error = MagicMock()
        mock_st.info = MagicMock()
        mock_st.success = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.checkbox = MagicMock(return_value=False)
        mock_st.pills = MagicMock(return_value="All")
        mock_st.multiselect = MagicMock(return_value=[])
        mock_st.button = MagicMock(
            side_effect=[False, True] + [False] * 10,
        )  # Second button is save ratings, others False
        mock_st.columns = MagicMock(side_effect=lambda n: [MagicMock() for _ in range(n)])
        mock_st.caption = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.write = MagicMock()
        mock_st.download_button = MagicMock()
        mock_st.file_uploader = MagicMock(return_value=None)
        mock_st.dialog = MagicMock()
        mock_st.dialog.return_value.__enter__ = MagicMock()
        mock_st.dialog.return_value.__exit__ = MagicMock()
        # Mock tabs with context managers
        tab1 = MagicMock()
        tab2 = MagicMock()
        tab3 = MagicMock()
        tab1.__enter__ = MagicMock()
        tab1.__exit__ = MagicMock()
        tab2.__enter__ = MagicMock()
        tab2.__exit__ = MagicMock()
        tab3.__enter__ = MagicMock()
        tab3.__exit__ = MagicMock()
        mock_st.tabs = MagicMock(return_value=[tab1, tab2, tab3])

        with (
            patch("st_name_ranking.main.st", mock_st),
            patch("st_name_ranking.main.database") as mock_db,
            patch("st_name_ranking.main.load_names_by_gender"),
            patch("st_name_ranking.main.setup_session_state"),
            patch("st_name_ranking.main.save_ratings") as mock_save_ratings,
        ):
            mock_db.init_database.return_value = None
            mock_db.get_stats.return_value = {
                "total_names": 2,
                "classified_names": 2,
                "rated_names": 2,
                "origin_distribution": {"Nordic": 2},
            }
            mock_db.get_all_origin_regions.return_value = ["Nordic"]
            mock_db.load_user_setting.return_value = "[]"
            mock_db.get_names_by_filters.return_value = ["Anna", "Peter"]

            # Mock save_ratings to return True
            mock_save_ratings.return_value = True

            # Call main function (simplified - we're mainly checking button logic)
            # We'll mock the tabs and rendering to avoid complexity
            main_module.main()

            # Verify save_ratings was called with correct ratings
            mock_save_ratings.assert_called_with({"Anna": 1600.0, "Peter": 1400.0})
            mock_st.toast.assert_any_call("Ratings saved!", icon="✅")

    def test_main_origin_filter_persistence(self):
        """Test origin filter persistence in database."""
        mock_st = MagicMock()
        mock_st.set_page_config.return_value = None
        mock_st.title.return_value = None
        mock_st.sidebar = MagicMock()
        mock_st.session_state = MockSessionState(
            {
                "all_names_data": {"All": ["Anna", "Peter"]},
                "all_names": ["Anna", "Peter"],
                "gender_filter": "All",
            },
        )
        mock_st.rerun = MagicMock()
        mock_st.spinner = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()
        mock_st.toast = MagicMock()
        mock_st.error = MagicMock()
        mock_st.info = MagicMock()
        mock_st.success = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.checkbox = MagicMock(return_value=False)
        mock_st.pills = MagicMock(return_value="All")

        # Simulate user selecting Nordic origin
        mock_st.multiselect = MagicMock(return_value=["Nordic"])
        mock_st.button = MagicMock(return_value=False)
        mock_st.columns = MagicMock(side_effect=lambda n: [MagicMock() for _ in range(n)])
        mock_st.caption = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.write = MagicMock()
        mock_st.download_button = MagicMock()
        mock_st.file_uploader = MagicMock(return_value=None)
        mock_st.dialog = MagicMock()
        mock_st.dialog.return_value.__enter__ = MagicMock()
        mock_st.dialog.return_value.__exit__ = MagicMock()
        # Mock tabs with context managers
        tab1 = MagicMock()
        tab2 = MagicMock()
        tab3 = MagicMock()
        tab1.__enter__ = MagicMock()
        tab1.__exit__ = MagicMock()
        tab2.__enter__ = MagicMock()
        tab2.__exit__ = MagicMock()
        tab3.__enter__ = MagicMock()
        tab3.__exit__ = MagicMock()
        mock_st.tabs = MagicMock(return_value=[tab1, tab2, tab3])

        with (
            patch("st_name_ranking.main.st", mock_st),
            patch("st_name_ranking.main.database") as mock_db,
            patch("st_name_ranking.main.load_names_by_gender"),
            patch("st_name_ranking.main.setup_session_state"),
        ):
            mock_db.init_database.return_value = None
            mock_db.get_stats.return_value = {
                "total_names": 2,
                "classified_names": 2,
                "rated_names": 2,
                "origin_distribution": {"Nordic": 2},
            }
            mock_db.get_all_origin_regions.return_value = ["Nordic", "European"]

            # First call: load saved setting (empty array)
            # Second call: after user changes selection
            load_setting_calls = ["[]", "[]"]
            mock_db.load_user_setting.side_effect = lambda _, default: (
                load_setting_calls.pop(0) if load_setting_calls else default
            )

            save_called_with = []

            def save_side_effect(key, value):
                save_called_with.append((key, value))

            mock_db.save_user_setting.side_effect = save_side_effect

            mock_db.get_names_by_filters.return_value = ["Anna", "Peter"]

            # Mock tabs and rendering
            main_module.main()

            # Verify save was called with selected origins
            assert len(save_called_with) > 0
            key, value = save_called_with[0]
            assert key == "selected_origins"
            # Value should be JSON array containing "Nordic"
            saved_data = json.loads(value)
            assert "Nordic" in saved_data
            # Should trigger rerun
            mock_st.rerun.assert_called()
