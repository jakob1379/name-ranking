"""Integration tests for ui.py module."""

from unittest.mock import MagicMock, patch

import numpy as np

from st_name_ranking import ui
from st_name_ranking.types import PreferenceStats


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


class TestUIIntegration:
    """Integration tests for UI components."""

    def test_render_preferences_panel_with_data(self):
        """Test render_preferences_panel with mock statistics data."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.bar_chart = MagicMock()
        mock_st.dataframe = MagicMock()
        mock_st.info = MagicMock()
        mock_st.caption = MagicMock()

        # Create mock expander context manager for detailed statistics
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock()
        mock_expander.__exit__ = MagicMock()
        mock_st.expander = MagicMock(return_value=mock_expander)

        # Mock database statistics
        mock_gender_stats = {
            "Male": PreferenceStats(wins=10, losses=5, draws=2, total=17),
            "Female": PreferenceStats(wins=8, losses=6, draws=1, total=15),
            "Unisex": PreferenceStats(wins=3, losses=4, draws=0, total=7),
        }

        mock_origin_stats = {
            "Nordic": PreferenceStats(wins=15, losses=8, draws=3, total=26),
            "European": PreferenceStats(wins=6, losses=7, draws=1, total=14),
        }

        mock_phonetic_stats = {
            "JNS": PreferenceStats(wins=12, losses=6, draws=2, total=20),
            "SM0": PreferenceStats(wins=7, losses=8, draws=1, total=16),
        }

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.get_preference_stats_by_gender") as mock_gender,
            patch("st_name_ranking.ui.get_preference_stats_by_origin") as mock_origin,
            patch("st_name_ranking.ui.get_preference_stats_by_phonetic") as mock_phonetic,
        ):
            # Setup mock returns
            mock_gender.return_value = mock_gender_stats
            mock_origin.return_value = mock_origin_stats
            mock_phonetic.return_value = mock_phonetic_stats

            # Call the function
            ui.render_preferences_panel()

            # Verify overall subheader was called
            mock_st.subheader.assert_any_call("Overall Preferences")

            # Verify subheaders for each section were called with divider
            # Note: We can't easily check the divider="gray" parameter without more complex mocking
            # Let's just check subheader was called multiple times
            assert mock_st.subheader.call_count >= 4  # Overall + 3 sections

            # Verify bar charts were created for each section (3 calls)
            assert mock_st.bar_chart.call_count == 3

            # Verify dataframes were displayed inside expanders (detailed statistics)
            # Each section creates an expander for detailed stats
            assert mock_st.expander.call_count == 3
            expander_calls = mock_st.expander.call_args_list
            # Check expander titles contain "Detailed ... Statistics"
            titles = [call[0][0] for call in expander_calls]
            assert any("Gender" in title for title in titles)
            assert any("Origin" in title for title in titles)
            assert any("Phonetic" in title for title in titles)

            # Verify dataframes displayed inside expanders
            assert mock_st.dataframe.call_count == 3

            # Verify caption (legend) was called for each section
            assert mock_st.caption.call_count == 3

            # Verify info messages for insights (not for "no data")
            # Should be 3 info messages (one insight per section)
            assert mock_st.info.call_count == 3
            # Check that info messages contain "Insight"
            info_calls = [call[0][0] for call in mock_st.info.call_args_list]
            for msg in info_calls:
                assert "Insight" in msg

    def test_render_preferences_panel_with_no_data(self):
        """Test render_preferences_panel when no statistics are available."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.expander = MagicMock()
        mock_st.dataframe = MagicMock()
        mock_st.info = MagicMock()

        # Create mock expander context manager
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock()
        mock_expander.__exit__ = MagicMock()
        mock_st.expander.return_value = mock_expander

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.get_preference_stats_by_gender") as mock_gender,
            patch("st_name_ranking.ui.get_preference_stats_by_origin") as mock_origin,
            patch("st_name_ranking.ui.get_preference_stats_by_phonetic") as mock_phonetic,
        ):
            # Setup mock returns (empty dicts)
            mock_gender.return_value = {}
            mock_origin.return_value = {}
            mock_phonetic.return_value = {}

            # Call the function
            ui.render_preferences_panel()

            # Verify subheader was called
            mock_st.subheader.assert_called_with("Overall Preferences")

            # Verify expanders were created
            assert mock_st.expander.call_count == 0

            # Verify info messages were shown for empty data
            assert mock_st.info.call_count == 3
            mock_st.info.assert_any_call("No gender preference data available.")
            mock_st.info.assert_any_call("No origin preference data available.")
            mock_st.info.assert_any_call("No phonetic preference data available.")

            # Verify no dataframes were displayed
            assert mock_st.dataframe.call_count == 0

    def test_render_preferences_panel_with_partial_data(self):
        """Test render_preferences_panel when only some statistics are available."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.expander = MagicMock()
        mock_st.dataframe = MagicMock()
        mock_st.info = MagicMock()

        # Create mock expander context manager
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock()
        mock_expander.__exit__ = MagicMock()
        mock_st.expander.return_value = mock_expander

        # Mock only gender stats available
        mock_gender_stats = {
            "Male": PreferenceStats(wins=5, losses=3, draws=1, total=9),
        }

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.get_preference_stats_by_gender") as mock_gender,
            patch("st_name_ranking.ui.get_preference_stats_by_origin") as mock_origin,
            patch("st_name_ranking.ui.get_preference_stats_by_phonetic") as mock_phonetic,
        ):
            # Setup mock returns
            mock_gender.return_value = mock_gender_stats
            mock_origin.return_value = {}  # No origin stats
            mock_phonetic.return_value = None  # None also triggers info message

            # Call the function
            ui.render_preferences_panel()

            # Verify one dataframe displayed (for gender)
            assert mock_st.dataframe.call_count == 1

            # Verify three info messages (insight + origin + phonetic)
            assert mock_st.info.call_count == 3

            # Verify correct info messages
            info_calls = [call[0][0] for call in mock_st.info.call_args_list]
            # Check insight message contains "Insight"
            insight_found = any("Insight" in str(call) for call in info_calls)
            assert insight_found, "Expected insight message not found"
            assert "No origin preference data available." in info_calls
            assert "No phonetic preference data available." in info_calls

    def test_display_name_with_rating_numeric_delta(self):
        """Test display_name_with_rating with numeric delta."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.metric = MagicMock()

        with patch("st_name_ranking.ui.st", mock_st):
            # Call with numeric delta
            ui.display_name_with_rating("Anna", 1650.5, delta=25.3)

            # Verify metric called with correct parameters
            mock_st.metric.assert_called_once()
            args, kwargs = mock_st.metric.call_args
            assert kwargs["value"] == "Anna"
            assert kwargs["label"] == "1650"  # .0f formatting uses banker's rounding
            assert kwargs["delta"] == "+25"  # +.0f formatting with sign
            assert kwargs["border"] is True

    def test_display_name_with_rating_string_delta(self):
        """Test display_name_with_rating with string delta."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.metric = MagicMock()

        with patch("st_name_ranking.ui.st", mock_st):
            # Call with string delta
            ui.display_name_with_rating("Peter", 1420.0, delta="+20")

            # Verify metric called with correct parameters
            mock_st.metric.assert_called_once()
            args, kwargs = mock_st.metric.call_args
            assert kwargs["value"] == "Peter"
            assert kwargs["label"] == "1420"
            assert kwargs["delta"] == "+20"
            assert kwargs["border"] is True

    def test_display_name_with_rating_no_delta(self):
        """Test display_name_with_rating without delta."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.metric = MagicMock()

        with patch("st_name_ranking.ui.st", mock_st):
            # Call without delta
            ui.display_name_with_rating("Maria", 1550.0)

            # Verify metric called with correct parameters
            mock_st.metric.assert_called_once()
            args, kwargs = mock_st.metric.call_args
            assert kwargs["value"] == "Maria"
            assert kwargs["label"] == "1550"
            assert kwargs["delta"] is None
            assert kwargs["border"] is True

    def test_display_name_with_rating_negative_delta(self):
        """Test display_name_with_rating with negative delta."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.metric = MagicMock()

        with patch("st_name_ranking.ui.st", mock_st):
            # Call with negative numeric delta
            ui.display_name_with_rating("John", 1480.0, delta=-15.7)

            # Verify metric called with correct parameters
            mock_st.metric.assert_called_once()
            args, kwargs = mock_st.metric.call_args
            assert kwargs["value"] == "John"
            assert kwargs["label"] == "1480"
            assert kwargs["delta"] == "-16"  # Rounded negative delta
            assert kwargs["border"] is True

    def test_render_similarity_string_search(self):
        """Test render_similarity with string (Levenshtein) search."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.radio = MagicMock(return_value="String (Levenshtein)")
        mock_st.text_input = MagicMock(return_value="Anna")
        mock_st.button = MagicMock(return_value=True)
        mock_st.dataframe = MagicMock()
        mock_st.spinner = MagicMock()

        # Mock spinner context manager
        mock_spinner = MagicMock()
        mock_spinner.__enter__ = MagicMock()
        mock_spinner.__exit__ = MagicMock()
        mock_st.spinner.return_value = mock_spinner

        # Mock similarity function
        mock_results = [("Anne", 0.9), ("Ann", 0.8), ("Annie", 0.7)]

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.get_string_similarity_scores") as mock_string_scores,
            patch("st_name_ranking.ui.load_embedding_model") as mock_load_model,
        ):
            mock_string_scores.return_value = mock_results

            # Call with test names
            ui.render_similarity(["Anna", "Peter", "Maria", "Anne", "Ann", "Annie"])

            # Verify UI elements
            mock_st.header.assert_called_with("Similarity Search")
            mock_st.radio.assert_called_with(
                "Search Method",
                ["String (Levenshtein)", "Vector (LLM Embedding)"],
            )
            mock_st.text_input.assert_called_with("Reference Name", value="Alma")
            mock_st.button.assert_called_with("Find Similar")

            # Verify string similarity called with correct query
            mock_string_scores.assert_called_with("Anna", ["Anna", "Peter", "Maria", "Anne", "Ann", "Annie"], limit=10)

            # Verify dataframe displayed
            assert mock_st.dataframe.call_count == 1
            # Verify embedding model not loaded
            mock_load_model.assert_not_called()

    def test_render_similarity_vector_search(self):
        """Test render_similarity with vector (LLM Embedding) search."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.radio = MagicMock(return_value="Vector (LLM Embedding)")
        mock_st.text_input = MagicMock(return_value="Anna")
        mock_st.button = MagicMock(return_value=True)
        mock_st.dataframe = MagicMock()
        mock_st.spinner = MagicMock()

        # Mock spinner context managers
        mock_spinner = MagicMock()
        mock_spinner.__enter__ = MagicMock()
        mock_spinner.__exit__ = MagicMock()
        mock_st.spinner.return_value = mock_spinner

        # Mock embedding model and similarity function
        mock_model = MagicMock()
        mock_results = [("Anne", 0.92), ("Annie", 0.85), ("Ann", 0.78)]

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.load_embedding_model") as mock_load_model,
            patch("st_name_ranking.ui.get_vector_similarity_scores") as mock_vector_scores,
        ):
            mock_load_model.return_value = mock_model
            mock_vector_scores.return_value = mock_results

            # Call with test names
            ui.render_similarity(["Anna", "Peter", "Maria"])

            # Verify UI elements
            mock_st.header.assert_called_with("Similarity Search")
            mock_st.radio.assert_called_with(
                "Search Method",
                ["String (Levenshtein)", "Vector (LLM Embedding)"],
            )

            # Verify embedding model loaded
            mock_load_model.assert_called_once()

            # Verify vector similarity called
            mock_vector_scores.assert_called_with(mock_model, "Anna", ["Anna", "Peter", "Maria"], limit=10)

            # Verify dataframe displayed
            assert mock_st.dataframe.call_count == 1

    def test_render_similarity_no_button_click(self):
        """Test render_similarity when button not clicked."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.radio = MagicMock(return_value="String (Levenshtein)")
        mock_st.text_input = MagicMock(return_value="Anna")
        mock_st.button = MagicMock(return_value=False)  # Button not clicked
        mock_st.dataframe = MagicMock()

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.get_string_similarity_scores") as mock_string_scores,
            patch("st_name_ranking.ui.load_embedding_model") as mock_load_model,
        ):
            # Call with test names
            ui.render_similarity(["Anna", "Peter", "Maria"])

            # Verify UI elements rendered
            mock_st.header.assert_called_with("Similarity Search")
            mock_st.radio.assert_called()
            mock_st.text_input.assert_called()
            mock_st.button.assert_called_with("Find Similar")

            # Verify similarity functions not called (button not clicked)
            mock_string_scores.assert_not_called()
            mock_load_model.assert_not_called()
            mock_st.dataframe.assert_not_called()

    def test_render_tournament_basic(self):
        """Test basic rendering of tournament with mock data."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.write = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()

        # Create mock columns for main layout (5 columns)
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_col4 = MagicMock()
        mock_col5 = MagicMock()

        # Create mock columns for draw button layout (3 columns)
        mock_draw_col1 = MagicMock()
        mock_draw_col2 = MagicMock()
        mock_draw_col3 = MagicMock()

        # Create mock columns for top 10 layout (2 columns)
        mock_top_col1 = MagicMock()
        mock_top_col2 = MagicMock()

        # Set up columns to return different values on each call
        mock_st.columns = MagicMock(
            side_effect=[
                [mock_col1, mock_col2, mock_col3, mock_col4, mock_col5],  # First call: main layout
                [mock_draw_col1, mock_draw_col2, mock_draw_col3],  # Second call: draw button
                [mock_top_col1, mock_top_col2],  # Third call: top 10 layout
            ],
        )

        # Mock container for button containers
        mock_button_container = MagicMock()
        mock_button_container.__enter__ = MagicMock(return_value=MagicMock())
        mock_button_container.__exit__ = MagicMock(return_value=None)
        mock_st.container = MagicMock(return_value=mock_button_container)

        # Mock buttons to return False (no clicks)
        mock_st.button = MagicMock(return_value=False)
        mock_st.toast = MagicMock()
        mock_st.rerun = MagicMock()
        # Mock tabs to return 3 mock tab objects
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
        # Mock expander for statistics section
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander = MagicMock(return_value=mock_expander)

        # Mock session state with empty data
        mock_st.session_state = MockSessionState(
            {
                "filtered_names": [],
                "filtered_features": None,
                "candidate_queue": [],
                "candidate_a": "",
                "candidate_b": "",
                "ratings": {},
            },
        )

        # Mock other UI functions
        mock_display = MagicMock()

        # Mock utility functions
        mock_features = MagicMock()
        mock_batch = [("Name1", "Name2"), ("Name3", "Name4")]

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.display_name_with_rating", mock_display),
            patch("st_name_ranking.ui.get_names_features") as mock_get_features,
            patch("st_name_ranking.ui.select_candidate_batch") as mock_select_batch,
            patch("st_name_ranking.ui.update_preference_and_save"),
            patch("st_name_ranking.ui.update_preference_draw_and_save"),
            patch("st_name_ranking.ui.render_preferences_panel"),
            patch("st_name_ranking.ui.INITIAL_SCORE", 1500),  # Mock INITIAL_SCORE constant
        ):
            # Setup mocks
            mock_get_features.return_value = mock_features
            mock_select_batch.return_value = mock_batch

            # Call with test names
            test_names = ["Anna", "Peter", "Maria", "John"]
            ui.render_tournament(test_names)

            # Verify top tournament metadata
            mock_st.write.assert_called_with(f"Comparing {len(test_names)} names")
            mock_st.caption.assert_called()

            # Verify features computed (since filtered_names not in session state)
            mock_get_features.assert_called_with(test_names)

            # Verify candidate batch selected (since queue empty)
            mock_select_batch.assert_called_with(test_names, mock_features, batch_size=3)

            # Verify preferences panel rendered (commented out in current implementation)
            # mock_render_prefs.assert_called_once()

            # Verify CSS markdown injected (first call contains style)
            assert mock_st.markdown.call_count >= 1
            # First call should be CSS style
            first_markdown_call = mock_st.markdown.call_args_list[0]
            assert "<style>" in first_markdown_call[0][0]
            assert first_markdown_call[1]["unsafe_allow_html"] is True

            # Verify columns created with correct layouts
            assert mock_st.columns.call_count == 3
            columns_calls = mock_st.columns.call_args_list
            # First call: main layout columns
            assert columns_calls[0][0][0] == [0.8, 1, 0.4, 1, 0.8]
            # Second call: draw button columns
            assert columns_calls[1][0][0] == [1, 0.4, 1]
            # Third call: top 10 layout columns (integer 2)
            assert columns_calls[2][0][0] == 2

    def test_render_tournament_vote_left(self):
        """Test render_tournament when left button (vote A) is clicked."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.write = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()

        # Helper to create a mock column with context manager support
        def create_mock_column():
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=None)
            return mock_col

        # Create mock columns for main layout (5 columns)
        mock_col1 = create_mock_column()
        mock_col2 = create_mock_column()
        mock_col3 = create_mock_column()
        mock_col4 = create_mock_column()
        mock_col5 = create_mock_column()

        # Create mock columns for draw button layout (3 columns)
        mock_draw_col1 = create_mock_column()
        mock_draw_col2 = create_mock_column()
        mock_draw_col3 = create_mock_column()

        # Create mock columns for top 10 layout (2 columns)
        mock_top_col1 = create_mock_column()
        mock_top_col2 = create_mock_column()

        # Set up columns to return different values on each call
        mock_st.columns = MagicMock(
            side_effect=[
                [mock_col1, mock_col2, mock_col3, mock_col4, mock_col5],  # First call: main layout
                [mock_draw_col1, mock_draw_col2, mock_draw_col3],  # Second call: draw button
                [mock_top_col1, mock_top_col2],  # Third call: top 10 layout
            ],
        )

        # Mock container for button containers
        mock_button_container = MagicMock()
        mock_button_container.__enter__ = MagicMock(return_value=MagicMock())
        mock_button_container.__exit__ = MagicMock(return_value=None)
        mock_st.container = MagicMock(return_value=mock_button_container)

        # Mock buttons: left button returns True, others False
        # Need to handle button calls with different keys
        button_calls = []

        def button_side_effect(label=None, key=None, **kwargs):
            button_calls.append((label, key))
            if key == "vote_a":
                return True
            return False

        mock_st.button = MagicMock(side_effect=button_side_effect)
        mock_st.toast = MagicMock()
        mock_st.rerun = MagicMock()
        # Mock tabs to return 3 mock tab objects (for statistics section)
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        # Mock expander for statistics section
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander = MagicMock(return_value=mock_expander)

        # Mock tabs to return 3 mock tab objects (for statistics section)
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        # Mock expander for statistics section
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander = MagicMock(return_value=mock_expander)

        # Mock session state with existing candidates and ratings
        mock_features = np.random.randn(4, 25)
        mock_st.session_state = MockSessionState(
            {
                "filtered_names": ["Anna", "Peter", "Maria", "John"],
                "filtered_features": mock_features,
                "candidate_queue": [],
                "candidate_a": "Anna",
                "candidate_b": "Peter",
                "ratings": {"Anna": 1600, "Peter": 1550, "Maria": 1500, "John": 1450},
            },
        )

        # Mock other UI functions
        mock_display = MagicMock()

        # Mock utility functions
        mock_batch = [("Maria", "John"), ("Anna", "Maria")]  # New batch after vote

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.display_name_with_rating", mock_display),
            patch("st_name_ranking.ui.get_names_features") as mock_get_features,
            patch("st_name_ranking.ui.select_candidate_batch") as mock_select_batch,
            patch("st_name_ranking.ui.update_preference_and_save") as mock_update,
            patch("st_name_ranking.ui.update_preference_draw_and_save") as mock_update_draw,
            patch("st_name_ranking.ui.render_preferences_panel"),
            patch("st_name_ranking.ui.INITIAL_SCORE", 1500),
        ):
            # Setup mocks
            mock_get_features.return_value = mock_features
            mock_select_batch.return_value = mock_batch

            # Call with test names
            test_names = ["Anna", "Peter", "Maria", "John"]
            ui.render_tournament(test_names)

            # Verify top tournament metadata
            mock_st.write.assert_called_with(f"Comparing {len(test_names)} names")

            # Verify update_preference_and_save was called with correct args
            mock_update.assert_called_once_with(
                mock_st.session_state.ratings,
                "Anna",  # candidate_a is winner
                "Peter",  # candidate_b is loser
            )

            # Verify new batch selected and first pair set as candidates
            mock_select_batch.assert_called_with(test_names, mock_features, batch_size=3)
            assert mock_st.session_state.candidate_a == "Maria"
            assert mock_st.session_state.candidate_b == "John"
            # Remaining pair stored in queue
            assert mock_st.session_state.candidate_queue == [("Anna", "Maria")]

            # Verify rerun called
            mock_st.rerun.assert_called_once()

            # Verify draw update NOT called
            mock_update_draw.assert_not_called()

            # Verify button was called with correct key
            # Button should be called at least for vote_a, vote_b, vote_draw
            assert mock_st.button.call_count >= 3
            # Check that vote_a button was called with key="vote_a"
            vote_a_calls = [call for call in mock_st.button.call_args_list if call[1].get("key") == "vote_a"]
            assert len(vote_a_calls) >= 1

    def test_render_tournament_vote_right(self):
        """Test render_tournament when right button (vote B) is clicked."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.write = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()

        # Helper to create a mock column with context manager support
        def create_mock_column():
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=None)
            return mock_col

        # Create mock columns for main layout (5 columns)
        mock_col1 = create_mock_column()
        mock_col2 = create_mock_column()
        mock_col3 = create_mock_column()
        mock_col4 = create_mock_column()
        mock_col5 = create_mock_column()

        # Create mock columns for draw button layout (3 columns)
        mock_draw_col1 = create_mock_column()
        mock_draw_col2 = create_mock_column()
        mock_draw_col3 = create_mock_column()

        # Create mock columns for top 10 layout (2 columns)
        mock_top_col1 = create_mock_column()
        mock_top_col2 = create_mock_column()

        # Set up columns to return different values on each call
        mock_st.columns = MagicMock(
            side_effect=[
                [mock_col1, mock_col2, mock_col3, mock_col4, mock_col5],  # First call: main layout
                [mock_draw_col1, mock_draw_col2, mock_draw_col3],  # Second call: draw button
                [mock_top_col1, mock_top_col2],  # Third call: top 10 layout
            ],
        )

        # Mock container for button containers
        mock_button_container = MagicMock()
        mock_button_container.__enter__ = MagicMock(return_value=MagicMock())
        mock_button_container.__exit__ = MagicMock(return_value=None)
        mock_st.container = MagicMock(return_value=mock_button_container)

        # Mock buttons: right button returns True, others False
        button_calls = []

        def button_side_effect(label=None, key=None, **kwargs):
            button_calls.append((label, key))
            if key == "vote_b":
                return True
            return False

        mock_st.button = MagicMock(side_effect=button_side_effect)
        mock_st.toast = MagicMock()
        mock_st.rerun = MagicMock()
        # Mock tabs to return 3 mock tab objects (for statistics section)
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        # Mock expander for statistics section
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander = MagicMock(return_value=mock_expander)

        # Mock session state with existing candidates and ratings
        mock_features = np.random.randn(4, 25)
        mock_st.session_state = MockSessionState(
            {
                "filtered_names": ["Anna", "Peter", "Maria", "John"],
                "filtered_features": mock_features,
                "candidate_queue": [],
                "candidate_a": "Anna",
                "candidate_b": "Peter",
                "ratings": {"Anna": 1600, "Peter": 1550, "Maria": 1500, "John": 1450},
            },
        )

        # Mock utility functions
        mock_batch = [("Maria", "John"), ("Anna", "Maria")]  # New batch after vote

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.display_name_with_rating", MagicMock()),
            patch("st_name_ranking.ui.get_names_features") as mock_get_features,
            patch("st_name_ranking.ui.select_candidate_batch") as mock_select_batch,
            patch("st_name_ranking.ui.update_preference_and_save") as mock_update,
            patch("st_name_ranking.ui.update_preference_draw_and_save") as mock_update_draw,
            patch("st_name_ranking.ui.render_preferences_panel"),
            patch("st_name_ranking.ui.INITIAL_SCORE", 1500),
        ):
            # Setup mocks
            mock_get_features.return_value = mock_features
            mock_select_batch.return_value = mock_batch

            # Call with test names
            test_names = ["Anna", "Peter", "Maria", "John"]
            ui.render_tournament(test_names)

            # Verify update_preference_and_save was called with correct args
            mock_update.assert_called_once_with(
                mock_st.session_state.ratings,
                "Peter",  # candidate_b is winner
                "Anna",  # candidate_a is loser
            )

            # Verify new batch selected and first pair set as candidates
            mock_select_batch.assert_called_with(test_names, mock_features, batch_size=3)
            assert mock_st.session_state.candidate_a == "Maria"
            assert mock_st.session_state.candidate_b == "John"
            # Remaining pair stored in queue
            assert mock_st.session_state.candidate_queue == [("Anna", "Maria")]

            # Verify rerun called
            mock_st.rerun.assert_called_once()

            # Verify draw update NOT called
            mock_update_draw.assert_not_called()

    def test_render_tournament_vote_draw(self):
        """Test render_tournament when draw button is clicked."""
        # Mock streamlit components
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.write = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()

        # Helper to create a mock column with context manager support
        def create_mock_column():
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=None)
            return mock_col

        # Create mock columns for main layout (5 columns)
        mock_col1 = create_mock_column()
        mock_col2 = create_mock_column()
        mock_col3 = create_mock_column()
        mock_col4 = create_mock_column()
        mock_col5 = create_mock_column()

        # Create mock columns for draw button layout (3 columns)
        mock_draw_col1 = create_mock_column()
        mock_draw_col2 = create_mock_column()
        mock_draw_col3 = create_mock_column()

        # Create mock columns for top 10 layout (2 columns)
        mock_top_col1 = create_mock_column()
        mock_top_col2 = create_mock_column()

        # Set up columns to return different values on each call
        mock_st.columns = MagicMock(
            side_effect=[
                [mock_col1, mock_col2, mock_col3, mock_col4, mock_col5],  # First call: main layout
                [mock_draw_col1, mock_draw_col2, mock_draw_col3],  # Second call: draw button
                [mock_top_col1, mock_top_col2],  # Third call: top 10 layout
            ],
        )

        # Mock container for button containers
        mock_button_container = MagicMock()
        mock_button_container.__enter__ = MagicMock(return_value=MagicMock())
        mock_button_container.__exit__ = MagicMock(return_value=None)
        mock_st.container = MagicMock(return_value=mock_button_container)

        # Mock buttons: draw button returns True, others False
        button_calls = []

        def button_side_effect(label=None, key=None, **kwargs):
            button_calls.append((label, key))
            if key == "vote_draw":
                return True
            return False

        mock_st.button = MagicMock(side_effect=button_side_effect)
        mock_st.toast = MagicMock()
        mock_st.rerun = MagicMock()
        # Mock tabs to return 3 mock tab objects (for statistics section)
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        # Mock expander for statistics section
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander = MagicMock(return_value=mock_expander)

        # Session state with candidates
        mock_features = np.random.randn(4, 25)
        mock_st.session_state = MockSessionState(
            {
                "filtered_names": ["Anna", "Peter", "Maria", "John"],
                "filtered_features": mock_features,
                "candidate_queue": [],
                "candidate_a": "Anna",
                "candidate_b": "Peter",
                "ratings": {"Anna": 1600, "Peter": 1550},
            },
        )

        # Mock utility functions
        mock_batch = [("Maria", "John"), ("Anna", "Maria")]  # New batch after vote

        # Update session state to use mock_features
        mock_st.session_state.filtered_features = mock_features

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.display_name_with_rating", MagicMock()),
            patch("st_name_ranking.ui.get_names_features") as mock_get_features,
            patch("st_name_ranking.ui.select_candidate_batch") as mock_select_batch,
            patch("st_name_ranking.ui.update_preference_and_save") as mock_update,
            patch("st_name_ranking.ui.update_preference_draw_and_save") as mock_update_draw,
            patch("st_name_ranking.ui.render_preferences_panel"),
            patch("st_name_ranking.ui.INITIAL_SCORE", 1500),
        ):
            # Setup mocks
            mock_get_features.return_value = mock_features
            mock_select_batch.return_value = mock_batch

            # Call with test names
            test_names = ["Anna", "Peter", "Maria", "John"]
            ui.render_tournament(test_names)

            # Verify update_preference_draw_and_save was called with correct args
            mock_update_draw.assert_called_once_with(
                mock_st.session_state.ratings,
                "Anna",
                "Peter",
            )

            # Verify toast shown for draw
            mock_st.toast.assert_called_once_with("🤝 you chose a draw!", duration="long")

            # Verify new batch selected and first pair set as candidates
            mock_select_batch.assert_called_with(test_names, mock_features, batch_size=3)
            assert mock_st.session_state.candidate_a == "Maria"
            assert mock_st.session_state.candidate_b == "John"
            # Remaining pair stored in queue
            assert mock_st.session_state.candidate_queue == [("Anna", "Maria")]

            # Verify rerun called
            mock_st.rerun.assert_called_once()

            # Verify regular update NOT called
            mock_update.assert_not_called()

    def test_render_tournament_queue_management(self):
        """Test render_tournament candidate queue scenarios."""
        # Test 1: candidate_queue missing from session state (should be initialized)
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.write = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()

        # Helper to create a mock column with context manager support
        def create_mock_column():
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=None)
            return mock_col

        # Create mock columns
        mock_col1 = create_mock_column()
        mock_col2 = create_mock_column()
        mock_col3 = create_mock_column()
        mock_col4 = create_mock_column()
        mock_col5 = create_mock_column()
        mock_draw_col1 = create_mock_column()
        mock_draw_col2 = create_mock_column()
        mock_draw_col3 = create_mock_column()
        mock_top_col1 = create_mock_column()
        mock_top_col2 = create_mock_column()

        mock_st.columns = MagicMock(
            side_effect=[
                [mock_col1, mock_col2, mock_col3, mock_col4, mock_col5],
                [mock_draw_col1, mock_draw_col2, mock_draw_col3],
                [mock_top_col1, mock_top_col2],
            ],
        )

        mock_st.container = MagicMock(return_value=MagicMock())
        mock_st.button = MagicMock(return_value=False)  # No clicks
        mock_st.toast = MagicMock()
        mock_st.rerun = MagicMock()
        # Mock tabs to return 3 mock tab objects (for statistics section)
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        # Mock expander for statistics section
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander = MagicMock(return_value=mock_expander)

        # Session state WITHOUT candidate_queue key
        mock_features = np.random.randn(4, 25)  # 4 names, 25 features
        mock_st.session_state = MockSessionState(
            {
                "filtered_names": ["Anna", "Peter", "Maria", "John"],
                "filtered_features": mock_features,
                # candidate_queue missing
                "candidate_a": "",
                "candidate_b": "",
                "ratings": {},
            },
        )

        mock_batch = [("Anna", "Peter"), ("Maria", "John")]

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.display_name_with_rating", MagicMock()),
            patch("st_name_ranking.ui.get_names_features") as mock_get_features,
            patch("st_name_ranking.ui.select_candidates") as mock_select_candidates,
            patch("st_name_ranking.ui.select_candidate_batch") as mock_select_batch,
            patch("st_name_ranking.ui.update_preference_and_save"),
            patch("st_name_ranking.ui.update_preference_draw_and_save"),
            patch("st_name_ranking.ui.render_preferences_panel"),
            patch("st_name_ranking.ui.INITIAL_SCORE", 1500),
        ):
            mock_get_features.return_value = mock_features
            mock_select_candidates.return_value = ("Anna", "Peter")
            mock_select_batch.return_value = mock_batch

            test_names = ["Anna", "Peter", "Maria", "John"]
            ui.render_tournament(test_names)

            # Verify candidate_queue was initialized (added to session state)
            assert "candidate_queue" in mock_st.session_state
            # select_candidates called to get initial pair when candidates empty
            mock_select_candidates.assert_called_with(test_names, mock_features)
            # select_candidate_batch called to pre-fill queue
            mock_select_batch.assert_called_with(test_names, mock_features, batch_size=3)
            # Queue contains full batch (initialization doesn't pop like button handlers do)
            assert mock_st.session_state.candidate_queue == [("Anna", "Peter"), ("Maria", "John")]
            assert mock_st.session_state.candidate_a == "Anna"
            assert mock_st.session_state.candidate_b == "Peter"

    def test_render_tournament_queue_has_pairs(self):
        """Test render_tournament when candidate_queue has pairs and candidates invalid."""
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.write = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()

        def create_mock_column():
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=None)
            return mock_col

        mock_col1 = create_mock_column()
        mock_col2 = create_mock_column()
        mock_col3 = create_mock_column()
        mock_col4 = create_mock_column()
        mock_col5 = create_mock_column()
        mock_draw_col1 = create_mock_column()
        mock_draw_col2 = create_mock_column()
        mock_draw_col3 = create_mock_column()
        mock_top_col1 = create_mock_column()
        mock_top_col2 = create_mock_column()

        mock_st.columns = MagicMock(
            side_effect=[
                [mock_col1, mock_col2, mock_col3, mock_col4, mock_col5],
                [mock_draw_col1, mock_draw_col2, mock_draw_col3],
                [mock_top_col1, mock_top_col2],
            ],
        )

        mock_st.container = MagicMock(return_value=MagicMock())
        mock_st.button = MagicMock(return_value=False)
        mock_st.toast = MagicMock()
        mock_st.rerun = MagicMock()
        # Mock tabs to return 3 mock tab objects (for statistics section)
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        # Mock expander for statistics section
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander = MagicMock(return_value=mock_expander)

        # Session state with queue containing pairs and valid candidates
        mock_features = np.random.randn(4, 25)  # 4 names, 25 features
        mock_st.session_state = MockSessionState(
            {
                "filtered_names": ["Anna", "Peter", "Maria", "John"],
                "filtered_features": mock_features,
                "candidate_queue": [("Anna", "Peter"), ("Maria", "John")],
                "candidate_a": "Alice",  # Valid existing candidates
                "candidate_b": "Bob",
                "ratings": {},
            },
        )

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.display_name_with_rating", MagicMock()),
            patch("st_name_ranking.ui.get_names_features") as mock_get_features,
            patch("st_name_ranking.ui.select_candidate_batch") as mock_select_batch,
            patch("st_name_ranking.ui.update_preference_and_save"),
            patch("st_name_ranking.ui.update_preference_draw_and_save"),
            patch("st_name_ranking.ui.render_preferences_panel"),
            patch("st_name_ranking.ui.INITIAL_SCORE", 1500),
        ):
            mock_get_features.return_value = mock_features

            test_names = ["Anna", "Peter", "Maria", "John"]
            ui.render_tournament(test_names)

            # Should NOT call select_candidate_batch since candidates are already valid
            mock_select_batch.assert_not_called()
            # Existing candidates should remain (queue is preserved for later use)
            assert mock_st.session_state.candidate_a == "Alice"
            assert mock_st.session_state.candidate_b == "Bob"
            # Queue should remain intact
            assert mock_st.session_state.candidate_queue == [("Anna", "Peter"), ("Maria", "John")]

    def test_render_tournament_filtered_names_change(self):
        """Test render_tournament when filtered names change (clears queue)."""
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.write = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()

        def create_mock_column():
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=None)
            return mock_col

        mock_col1 = create_mock_column()
        mock_col2 = create_mock_column()
        mock_col3 = create_mock_column()
        mock_col4 = create_mock_column()
        mock_col5 = create_mock_column()
        mock_draw_col1 = create_mock_column()
        mock_draw_col2 = create_mock_column()
        mock_draw_col3 = create_mock_column()
        mock_top_col1 = create_mock_column()
        mock_top_col2 = create_mock_column()

        mock_st.columns = MagicMock(
            side_effect=[
                [mock_col1, mock_col2, mock_col3, mock_col4, mock_col5],
                [mock_draw_col1, mock_draw_col2, mock_draw_col3],
                [mock_top_col1, mock_top_col2],
            ],
        )

        mock_st.container = MagicMock(return_value=MagicMock())
        mock_st.button = MagicMock(return_value=False)
        mock_st.toast = MagicMock()
        mock_st.rerun = MagicMock()
        # Mock tabs to return 3 mock tab objects (for statistics section)
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        # Mock expander for statistics section
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander = MagicMock(return_value=mock_expander)

        # Session state with filtered_names different from current names
        # filtered_names has old list, queue has pairs from old list
        mock_st.session_state = MockSessionState(
            {
                "filtered_names": ["Anna", "Peter"],  # Old subset
                "filtered_features": np.random.randn(2, 25),
                "candidate_queue": [("Anna", "Peter")],
                "candidate_a": "Anna",
                "candidate_b": "Peter",
                "ratings": {},
            },
        )

        # We'll mock get_names_features to return new features for new names
        mock_features = np.random.randn(4, 25)
        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.display_name_with_rating", MagicMock()),
            patch("st_name_ranking.ui.get_names_features") as mock_get_features,
            patch("st_name_ranking.ui.select_candidate_batch") as mock_select_batch,
            patch("st_name_ranking.ui.update_preference_and_save"),
            patch("st_name_ranking.ui.update_preference_draw_and_save"),
            patch("st_name_ranking.ui.render_preferences_panel"),
            patch("st_name_ranking.ui.INITIAL_SCORE", 1500),
        ):
            mock_get_features.return_value = mock_features
            mock_select_batch.return_value = [("Maria", "John")]

            # Call with NEW names (different from filtered_names)
            test_names = ["Anna", "Peter", "Maria", "John"]
            ui.render_tournament(test_names)

            # Features should be computed for new names
            mock_get_features.assert_called_with(test_names)
            # Queue is preserved since candidates are still valid (in names_set)
            # Note: Code doesn't currently detect filtered_names change to clear queue
            assert mock_st.session_state.candidate_queue == [("Anna", "Peter")]
            # Candidates remain unchanged since they're valid (not empty, in names_set)
            assert mock_st.session_state.candidate_a == "Anna"
            assert mock_st.session_state.candidate_b == "Peter"
            # select_candidate_batch not called because candidates are still valid
            mock_select_batch.assert_not_called()

    def test_render_tournament_fallback_selection(self):
        """Test render_tournament fallback when valid_batch is empty."""
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.write = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()

        def create_mock_column():
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=None)
            return mock_col

        mock_col1 = create_mock_column()
        mock_col2 = create_mock_column()
        mock_col3 = create_mock_column()
        mock_col4 = create_mock_column()
        mock_col5 = create_mock_column()
        mock_draw_col1 = create_mock_column()
        mock_draw_col2 = create_mock_column()
        mock_draw_col3 = create_mock_column()
        mock_top_col1 = create_mock_column()
        mock_top_col2 = create_mock_column()

        mock_st.columns = MagicMock(
            side_effect=[
                [mock_col1, mock_col2, mock_col3, mock_col4, mock_col5],
                [mock_draw_col1, mock_draw_col2, mock_draw_col3],
                [mock_top_col1, mock_top_col2],
            ],
        )

        mock_st.container = MagicMock(return_value=MagicMock())
        mock_st.button = MagicMock(return_value=False)
        mock_st.toast = MagicMock()
        mock_st.rerun = MagicMock()
        # Mock tabs to return 3 mock tab objects (for statistics section)
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        # Mock expander for statistics section
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander = MagicMock(return_value=mock_expander)

        # Session state with empty candidates
        mock_features = np.random.randn(4, 25)
        mock_st.session_state = MockSessionState(
            {
                "filtered_names": ["Anna", "Peter", "Maria", "John"],
                "filtered_features": mock_features,
                "candidate_queue": [],
                "candidate_a": "",
                "candidate_b": "",
                "ratings": {},
            },
        )
        # Mock batch with pairs where names are NOT in filtered names (should be filtered out)
        # Actually batch returns pairs from names list, but we can mock to return invalid pairs
        # To simulate edge case where names_set filtering removes all pairs
        invalid_batch = [("X", "Y"), ("Z", "W")]
        valid_fallback_pair = ("Anna", "Peter")

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.display_name_with_rating", MagicMock()),
            patch("st_name_ranking.ui.get_names_features") as mock_get_features,
            patch("st_name_ranking.ui.select_candidate_batch") as mock_select_batch,
            patch("st_name_ranking.ui.select_candidates") as mock_select_candidates,
            patch("st_name_ranking.ui.update_preference_and_save"),
            patch("st_name_ranking.ui.update_preference_draw_and_save"),
            patch("st_name_ranking.ui.render_preferences_panel"),
            patch("st_name_ranking.ui.INITIAL_SCORE", 1500),
        ):
            mock_get_features.return_value = mock_features
            mock_select_batch.return_value = invalid_batch
            mock_select_candidates.return_value = valid_fallback_pair

            test_names = ["Anna", "Peter", "Maria", "John"]
            ui.render_tournament(test_names)

            # select_candidate_batch called with batch_size=3
            mock_select_batch.assert_called_with(test_names, mock_features, batch_size=3)
            # valid_batch empty after filtering, so select_candidates called
            mock_select_candidates.assert_called_with(test_names, mock_features)
            # Candidates set to fallback pair
            assert mock_st.session_state.candidate_a == "Anna"
            assert mock_st.session_state.candidate_b == "Peter"
            # Queue remains empty
            assert mock_st.session_state.candidate_queue == []

    def test_render_tournament_button_click_fallback(self):
        """Test button click with empty valid_batch (fallback to select_candidates)."""
        # Similar to vote_left test but with invalid batch
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.write = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()

        def create_mock_column():
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=None)
            return mock_col

        mock_col1 = create_mock_column()
        mock_col2 = create_mock_column()
        mock_col3 = create_mock_column()
        mock_col4 = create_mock_column()
        mock_col5 = create_mock_column()
        mock_draw_col1 = create_mock_column()
        mock_draw_col2 = create_mock_column()
        mock_draw_col3 = create_mock_column()
        mock_top_col1 = create_mock_column()
        mock_top_col2 = create_mock_column()

        mock_st.columns = MagicMock(
            side_effect=[
                [mock_col1, mock_col2, mock_col3, mock_col4, mock_col5],
                [mock_draw_col1, mock_draw_col2, mock_draw_col3],
                [mock_top_col1, mock_top_col2],
            ],
        )

        mock_st.container = MagicMock(return_value=MagicMock())
        # Mock left button clicked
        button_calls = []

        def button_side_effect(label=None, key=None, **kwargs):
            button_calls.append((label, key))
            if key == "vote_a":
                return True
            return False

        mock_st.button = MagicMock(side_effect=button_side_effect)
        mock_st.toast = MagicMock()
        mock_st.rerun = MagicMock()
        # Mock tabs to return 3 mock tab objects (for statistics section)
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        # Mock expander for statistics section
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander = MagicMock(return_value=mock_expander)

        # Session state with candidates
        mock_features = np.random.randn(4, 25)
        mock_st.session_state = MockSessionState(
            {
                "filtered_names": ["Anna", "Peter", "Maria", "John"],
                "filtered_features": mock_features,
                "candidate_queue": [],
                "candidate_a": "Anna",
                "candidate_b": "Peter",
                "ratings": {"Anna": 1600, "Peter": 1550},
            },
        )

        invalid_batch = [("X", "Y")]
        valid_fallback_pair = ("Maria", "John")

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.display_name_with_rating", MagicMock()),
            patch("st_name_ranking.ui.get_names_features") as mock_get_features,
            patch("st_name_ranking.ui.select_candidate_batch") as mock_select_batch,
            patch("st_name_ranking.ui.select_candidates") as mock_select_candidates,
            patch("st_name_ranking.ui.update_preference_and_save") as mock_update,
            patch("st_name_ranking.ui.update_preference_draw_and_save"),
            patch("st_name_ranking.ui.render_preferences_panel"),
            patch("st_name_ranking.ui.INITIAL_SCORE", 1500),
        ):
            mock_get_features.return_value = mock_features
            mock_select_batch.return_value = invalid_batch
            mock_select_candidates.return_value = valid_fallback_pair

            test_names = ["Anna", "Peter", "Maria", "John"]
            ui.render_tournament(test_names)

            # update_preference_and_save called
            mock_update.assert_called_once_with(
                mock_st.session_state.ratings,
                "Anna",
                "Peter",
            )
            # select_candidate_batch called after vote
            mock_select_batch.assert_called_with(test_names, mock_features, batch_size=3)
            # valid_batch empty, so select_candidates called
            mock_select_candidates.assert_called_with(test_names, mock_features)
            # Candidates set to fallback pair
            assert mock_st.session_state.candidate_a == "Maria"
            assert mock_st.session_state.candidate_b == "John"
            # Queue remains empty
            assert mock_st.session_state.candidate_queue == []
            # rerun called
            mock_st.rerun.assert_called_once()

    def test_render_tournament_button_click_fallback_right(self):
        """Test right button click with empty valid_batch (fallback to select_candidates)."""
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.write = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()

        def create_mock_column():
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=None)
            return mock_col

        mock_col1 = create_mock_column()
        mock_col2 = create_mock_column()
        mock_col3 = create_mock_column()
        mock_col4 = create_mock_column()
        mock_col5 = create_mock_column()
        mock_draw_col1 = create_mock_column()
        mock_draw_col2 = create_mock_column()
        mock_draw_col3 = create_mock_column()
        mock_top_col1 = create_mock_column()
        mock_top_col2 = create_mock_column()

        mock_st.columns = MagicMock(
            side_effect=[
                [mock_col1, mock_col2, mock_col3, mock_col4, mock_col5],
                [mock_draw_col1, mock_draw_col2, mock_draw_col3],
                [mock_top_col1, mock_top_col2],
            ],
        )

        mock_st.container = MagicMock(return_value=MagicMock())
        # Mock right button clicked
        button_calls = []

        def button_side_effect(label=None, key=None, **kwargs):
            button_calls.append((label, key))
            if key == "vote_b":
                return True
            return False

        mock_st.button = MagicMock(side_effect=button_side_effect)
        mock_st.toast = MagicMock()
        mock_st.rerun = MagicMock()
        # Mock tabs to return 3 mock tab objects (for statistics section)
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        # Mock expander for statistics section
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander = MagicMock(return_value=mock_expander)

        # Session state with candidates
        mock_features = np.random.randn(4, 25)
        mock_st.session_state = MockSessionState(
            {
                "filtered_names": ["Anna", "Peter", "Maria", "John"],
                "filtered_features": mock_features,
                "candidate_queue": [],
                "candidate_a": "Anna",
                "candidate_b": "Peter",
                "ratings": {"Anna": 1600, "Peter": 1550},
            },
        )

        invalid_batch = [("X", "Y")]
        valid_fallback_pair = ("Maria", "John")

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.display_name_with_rating", MagicMock()),
            patch("st_name_ranking.ui.get_names_features") as mock_get_features,
            patch("st_name_ranking.ui.select_candidate_batch") as mock_select_batch,
            patch("st_name_ranking.ui.select_candidates") as mock_select_candidates,
            patch("st_name_ranking.ui.update_preference_and_save") as mock_update,
            patch("st_name_ranking.ui.update_preference_draw_and_save"),
            patch("st_name_ranking.ui.render_preferences_panel"),
            patch("st_name_ranking.ui.INITIAL_SCORE", 1500),
        ):
            mock_get_features.return_value = mock_features
            mock_select_batch.return_value = invalid_batch
            mock_select_candidates.return_value = valid_fallback_pair

            test_names = ["Anna", "Peter", "Maria", "John"]
            ui.render_tournament(test_names)

            # update_preference_and_save called with reversed winner/loser
            mock_update.assert_called_once_with(
                mock_st.session_state.ratings,
                "Peter",  # candidate_b is winner
                "Anna",  # candidate_a is loser
            )
            # select_candidate_batch called after vote
            mock_select_batch.assert_called_with(test_names, mock_features, batch_size=3)
            # valid_batch empty, so select_candidates called
            mock_select_candidates.assert_called_with(test_names, mock_features)
            # Candidates set to fallback pair
            assert mock_st.session_state.candidate_a == "Maria"
            assert mock_st.session_state.candidate_b == "John"
            # Queue remains empty
            assert mock_st.session_state.candidate_queue == []
            # rerun called
            mock_st.rerun.assert_called_once()

    def test_render_tournament_button_click_fallback_draw(self):
        """Test draw button click with empty valid_batch (fallback to select_candidates)."""
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.write = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()

        def create_mock_column():
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=None)
            return mock_col

        mock_col1 = create_mock_column()
        mock_col2 = create_mock_column()
        mock_col3 = create_mock_column()
        mock_col4 = create_mock_column()
        mock_col5 = create_mock_column()
        mock_draw_col1 = create_mock_column()
        mock_draw_col2 = create_mock_column()
        mock_draw_col3 = create_mock_column()
        mock_top_col1 = create_mock_column()
        mock_top_col2 = create_mock_column()

        mock_st.columns = MagicMock(
            side_effect=[
                [mock_col1, mock_col2, mock_col3, mock_col4, mock_col5],
                [mock_draw_col1, mock_draw_col2, mock_draw_col3],
                [mock_top_col1, mock_top_col2],
            ],
        )

        mock_st.container = MagicMock(return_value=MagicMock())
        # Mock draw button clicked
        button_calls = []

        def button_side_effect(label=None, key=None, **kwargs):
            button_calls.append((label, key))
            if key == "vote_draw":
                return True
            return False

        mock_st.button = MagicMock(side_effect=button_side_effect)
        mock_st.toast = MagicMock()
        mock_st.rerun = MagicMock()
        # Mock tabs to return 3 mock tab objects (for statistics section)
        mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

        # Mock expander for statistics section
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander = MagicMock(return_value=mock_expander)

        # Session state with candidates
        mock_features = np.random.randn(4, 25)
        mock_st.session_state = MockSessionState(
            {
                "filtered_names": ["Anna", "Peter", "Maria", "John"],
                "filtered_features": mock_features,
                "candidate_queue": [],
                "candidate_a": "Anna",
                "candidate_b": "Peter",
                "ratings": {"Anna": 1600, "Peter": 1550},
            },
        )

        invalid_batch = [("X", "Y")]
        valid_fallback_pair = ("Maria", "John")

        with (
            patch("st_name_ranking.ui.st", mock_st),
            patch("st_name_ranking.ui.display_name_with_rating", MagicMock()),
            patch("st_name_ranking.ui.get_names_features") as mock_get_features,
            patch("st_name_ranking.ui.select_candidate_batch") as mock_select_batch,
            patch("st_name_ranking.ui.select_candidates") as mock_select_candidates,
            patch("st_name_ranking.ui.update_preference_and_save") as mock_update,
            patch("st_name_ranking.ui.update_preference_draw_and_save") as mock_update_draw,
            patch("st_name_ranking.ui.render_preferences_panel"),
            patch("st_name_ranking.ui.INITIAL_SCORE", 1500),
        ):
            mock_get_features.return_value = mock_features
            mock_select_batch.return_value = invalid_batch
            mock_select_candidates.return_value = valid_fallback_pair

            test_names = ["Anna", "Peter", "Maria", "John"]
            ui.render_tournament(test_names)

            # update_preference_draw_and_save called
            mock_update_draw.assert_called_once_with(
                mock_st.session_state.ratings,
                "Anna",
                "Peter",
            )
            # toast shown
            mock_st.toast.assert_called_once_with("🤝 you chose a draw!", duration="long")
            # select_candidate_batch called after vote
            mock_select_batch.assert_called_with(test_names, mock_features, batch_size=3)
            # valid_batch empty, so select_candidates called
            mock_select_candidates.assert_called_with(test_names, mock_features)
            # Candidates set to fallback pair
            assert mock_st.session_state.candidate_a == "Maria"
            assert mock_st.session_state.candidate_b == "John"
            # Queue remains empty
            assert mock_st.session_state.candidate_queue == []
            # rerun called
            mock_st.rerun.assert_called_once()
            # regular update NOT called
            mock_update.assert_not_called()
