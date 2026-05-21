"""Integration tests for ui.py module."""

from unittest.mock import MagicMock, patch

import numpy as np

from st_name_ranking.active_learning.lazy_updates import ModelUpdateStatus
from st_name_ranking.interface import rankings_ui, similarity_ui, tournament_ui
from st_name_ranking.tournament_orchestration import TournamentRound, VoteResult
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


def _mock_column():
    column = MagicMock()
    column.__enter__ = MagicMock(return_value=column)
    column.__exit__ = MagicMock(return_value=None)
    return column


def _mock_container():
    container = MagicMock()
    container.__enter__ = MagicMock(return_value=container)
    container.__exit__ = MagicMock(return_value=None)
    return container


def _mock_tournament_streamlit(*, clicked_key: str | None):
    mock_st = MagicMock()
    mock_st.header = MagicMock()
    mock_st.write = MagicMock()
    mock_st.caption = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.info = MagicMock()
    mock_st.toast = MagicMock()
    mock_st.session_state = MockSessionState(
        {
            "candidate_a": "",
            "candidate_b": "",
            "ratings": {"Anna": 1600, "Bo": 1550, "Maria": 1500, "Peter": 1450},
        },
    )

    placeholder = MagicMock()
    placeholder.container = MagicMock(return_value=_mock_container())
    mock_st.empty = MagicMock(side_effect=[placeholder, MagicMock()])

    def columns_side_effect(spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [_mock_column() for _ in range(count)]

    def button_side_effect(*_args, key=None, **_kwargs):
        return key == clicked_key

    mock_st.columns = MagicMock(side_effect=columns_side_effect)
    mock_st.button = MagicMock(side_effect=button_side_effect)
    return mock_st


def _render_tournament_body(names: list[str]) -> None:
    tournament_ui.render_tournament.__wrapped__(names)


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
            patch("st_name_ranking.interface.rankings_ui.st", mock_st),
            patch("st_name_ranking.interface.rankings_ui.get_preference_stats_by_gender") as mock_gender,
            patch("st_name_ranking.interface.rankings_ui.get_preference_stats_by_origin") as mock_origin,
            patch("st_name_ranking.interface.rankings_ui.get_preference_stats_by_phonetic") as mock_phonetic,
        ):
            # Setup mock returns
            mock_gender.return_value = mock_gender_stats
            mock_origin.return_value = mock_origin_stats
            mock_phonetic.return_value = mock_phonetic_stats

            # Call the function
            rankings_ui.render_preferences_panel()

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
            patch("st_name_ranking.interface.rankings_ui.st", mock_st),
            patch("st_name_ranking.interface.rankings_ui.get_preference_stats_by_gender") as mock_gender,
            patch("st_name_ranking.interface.rankings_ui.get_preference_stats_by_origin") as mock_origin,
            patch("st_name_ranking.interface.rankings_ui.get_preference_stats_by_phonetic") as mock_phonetic,
        ):
            # Setup mock returns (empty dicts)
            mock_gender.return_value = {}
            mock_origin.return_value = {}
            mock_phonetic.return_value = {}

            # Call the function
            rankings_ui.render_preferences_panel()

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
            patch("st_name_ranking.interface.rankings_ui.st", mock_st),
            patch("st_name_ranking.interface.rankings_ui.get_preference_stats_by_gender") as mock_gender,
            patch("st_name_ranking.interface.rankings_ui.get_preference_stats_by_origin") as mock_origin,
            patch("st_name_ranking.interface.rankings_ui.get_preference_stats_by_phonetic") as mock_phonetic,
        ):
            # Setup mock returns
            mock_gender.return_value = mock_gender_stats
            mock_origin.return_value = {}  # No origin stats
            mock_phonetic.return_value = None  # None also triggers info message

            # Call the function
            rankings_ui.render_preferences_panel()

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

        with patch("st_name_ranking.interface.tournament_ui.st", mock_st):
            # Call with numeric delta
            tournament_ui.display_name_with_rating("Anna", 1650.5, delta=25.3)

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

        with patch("st_name_ranking.interface.tournament_ui.st", mock_st):
            # Call with string delta
            tournament_ui.display_name_with_rating("Peter", 1420.0, delta="+20")

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

        with patch("st_name_ranking.interface.tournament_ui.st", mock_st):
            # Call without delta
            tournament_ui.display_name_with_rating("Maria", 1550.0)

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

        with patch("st_name_ranking.interface.tournament_ui.st", mock_st):
            # Call with negative numeric delta
            tournament_ui.display_name_with_rating("John", 1480.0, delta=-15.7)

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
            patch("st_name_ranking.interface.similarity_ui.st", mock_st),
            patch("st_name_ranking.interface.similarity_ui.get_string_similarity_scores") as mock_string_scores,
            patch("st_name_ranking.interface.similarity_ui.load_embedding_model") as mock_load_model,
        ):
            mock_string_scores.return_value = mock_results

            # Call with test names
            similarity_ui.render_similarity(["Anna", "Peter", "Maria", "Anne", "Ann", "Annie"])

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
            patch("st_name_ranking.interface.similarity_ui.st", mock_st),
            patch("st_name_ranking.interface.similarity_ui.load_embedding_model") as mock_load_model,
            patch("st_name_ranking.interface.similarity_ui.get_vector_similarity_scores") as mock_vector_scores,
        ):
            mock_load_model.return_value = mock_model
            mock_vector_scores.return_value = mock_results

            # Call with test names
            similarity_ui.render_similarity(["Anna", "Peter", "Maria"])

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
            patch("st_name_ranking.interface.similarity_ui.st", mock_st),
            patch("st_name_ranking.interface.similarity_ui.get_string_similarity_scores") as mock_string_scores,
            patch("st_name_ranking.interface.similarity_ui.load_embedding_model") as mock_load_model,
        ):
            # Call with test names
            similarity_ui.render_similarity(["Anna", "Peter", "Maria"])

            # Verify UI elements rendered
            mock_st.header.assert_called_with("Similarity Search")
            mock_st.radio.assert_called()
            mock_st.text_input.assert_called()
            mock_st.button.assert_called_with("Find Similar")

            # Verify similarity functions not called (button not clicked)
            mock_string_scores.assert_not_called()
            mock_load_model.assert_not_called()
            mock_st.dataframe.assert_not_called()

    def test_render_tournament_prepares_round_at_orchestration_boundary(self):
        """Tournament rendering gets its pair from the orchestration boundary."""
        mock_st = _mock_tournament_streamlit(clicked_key=None)
        manager = MagicMock(target_size=5)
        round_state = TournamentRound(
            manager=manager,
            candidate_a="Anna",
            candidate_b="Bo",
            queue_stats={"queue_size": 2, "target_size": 5},
        )
        names = ["Anna", "Bo", "Maria", "Peter"]

        with (
            patch("st_name_ranking.interface.tournament_ui.st", mock_st),
            patch("st_name_ranking.interface.tournament_session.st", mock_st),
            patch("st_name_ranking.interface.tournament_ui.display_name_with_rating") as mock_display,
            patch(
                "st_name_ranking.interface.tournament_ui.get_or_start_tournament_queue",
                return_value=manager,
            ) as mock_get_manager,
            patch(
                "st_name_ranking.interface.tournament_ui.get_current_pair",
                return_value=("Anna", "Bo"),
            ) as mock_get_pair,
            patch(
                "st_name_ranking.interface.tournament_ui.get_queue_manager_stats",
                return_value={"queue_size": 2, "target_size": 5},
            ) as mock_stats,
            patch(
                "st_name_ranking.interface.tournament_ui.prepare_tournament_round",
                return_value=round_state,
            ) as mock_prepare,
            patch("st_name_ranking.interface.tournament_ui.record_tournament_vote") as mock_record_vote,
        ):
            _render_tournament_body(names)

        mock_get_manager.assert_called_once_with(names, len(names))
        mock_get_pair.assert_called_once_with(names)
        mock_stats.assert_called_once_with()
        mock_prepare.assert_called_once_with(names, manager, ("Anna", "Bo"), {"queue_size": 2, "target_size": 5})
        mock_record_vote.assert_not_called()
        assert mock_st.session_state.candidate_a == "Anna"
        assert mock_st.session_state.candidate_b == "Bo"
        displayed_names = [call.args[0] for call in mock_display.call_args_list]
        assert displayed_names == ["Anna", "Bo"]

    def test_render_tournament_vote_uses_orchestration_result(self):
        """A button vote records through orchestration and displays the returned pair."""
        mock_st = _mock_tournament_streamlit(clicked_key="vote_a")
        manager = MagicMock(target_size=5)
        names = ["Anna", "Bo", "Maria", "Peter"]
        round_state = TournamentRound(
            manager=manager,
            candidate_a="Anna",
            candidate_b="Bo",
            queue_stats={"queue_size": 1, "target_size": 5},
        )
        vote_result = VoteResult(
            previous_pair=("Anna", "Bo"),
            next_pair=("Maria", "Peter"),
            pair_source="queue",
            update_status=ModelUpdateStatus(recorded=True, model_updated=False, ratings_fresh=True),
        )

        with (
            patch("st_name_ranking.interface.tournament_ui.st", mock_st),
            patch("st_name_ranking.interface.tournament_session.st", mock_st),
            patch("st_name_ranking.interface.tournament_ui.display_name_with_rating") as mock_display,
            patch("st_name_ranking.interface.tournament_ui.get_or_start_tournament_queue", return_value=manager),
            patch("st_name_ranking.interface.tournament_ui.get_current_pair", return_value=("Anna", "Bo")),
            patch(
                "st_name_ranking.interface.tournament_ui.get_queue_manager_stats",
                return_value={"queue_size": 1, "target_size": 5},
            ),
            patch("st_name_ranking.interface.tournament_ui.prepare_tournament_round", return_value=round_state),
            patch(
                "st_name_ranking.interface.tournament_ui.record_tournament_vote",
                return_value=vote_result,
            ) as mock_record_vote,
        ):
            _render_tournament_body(names)

        mock_record_vote.assert_called_once_with(names, manager, "Anna", "Bo", -1)
        assert mock_st.session_state.candidate_a == "Maria"
        assert mock_st.session_state.candidate_b == "Peter"
        displayed_names = [call.args[0] for call in mock_display.call_args_list]
        assert displayed_names == ["Maria", "Peter", "Maria", "Peter"]

    def test_render_tournament_failed_vote_keeps_pair_and_toasts(self):
        """A failed vote result leaves the current pair visible instead of advancing."""
        mock_st = _mock_tournament_streamlit(clicked_key="vote_b")
        manager = MagicMock(target_size=5)
        names = ["Anna", "Bo", "Maria", "Peter"]
        round_state = TournamentRound(
            manager=manager,
            candidate_a="Anna",
            candidate_b="Bo",
            queue_stats={"queue_size": 1, "target_size": 5},
        )
        vote_result = VoteResult(
            previous_pair=("Anna", "Bo"),
            next_pair=("Anna", "Bo"),
            pair_source="unchanged",
            update_status=ModelUpdateStatus(
                recorded=False,
                model_updated=False,
                ratings_fresh=False,
                fallback_used=True,
                error="database locked",
            ),
        )

        with (
            patch("st_name_ranking.interface.tournament_ui.st", mock_st),
            patch("st_name_ranking.interface.tournament_session.st", mock_st),
            patch("st_name_ranking.interface.tournament_ui.display_name_with_rating") as mock_display,
            patch("st_name_ranking.interface.tournament_ui.get_or_start_tournament_queue", return_value=manager),
            patch("st_name_ranking.interface.tournament_ui.get_current_pair", return_value=("Anna", "Bo")),
            patch(
                "st_name_ranking.interface.tournament_ui.get_queue_manager_stats",
                return_value={"queue_size": 1, "target_size": 5},
            ),
            patch("st_name_ranking.interface.tournament_ui.prepare_tournament_round", return_value=round_state),
            patch(
                "st_name_ranking.interface.tournament_ui.record_tournament_vote",
                return_value=vote_result,
            ) as mock_record_vote,
        ):
            _render_tournament_body(names)

        mock_record_vote.assert_called_once_with(names, manager, "Anna", "Bo", 1)
        assert mock_st.session_state.candidate_a == "Anna"
        assert mock_st.session_state.candidate_b == "Bo"
        mock_st.toast.assert_called_once_with("Vote was not saved: database locked", icon="❌", duration="long")
        displayed_names = [call.args[0] for call in mock_display.call_args_list]
        assert displayed_names == ["Anna", "Bo", "Anna", "Bo"]

    def test_render_rankings_skips_landscape_with_small_sample(self):
        """Test rankings keeps table behavior and skips landscape for small samples."""
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.write = MagicMock()
        mock_st.dataframe = MagicMock()
        mock_st.info = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.cache_data = lambda **_: lambda func: func

        def create_tab():
            tab = MagicMock()
            tab.__enter__ = MagicMock(return_value=tab)
            tab.__exit__ = MagicMock(return_value=None)
            return tab

        mock_st.tabs = MagicMock(return_value=[create_tab(), create_tab(), create_tab()])
        mock_st.session_state = MockSessionState(
            {
                "ratings": {f"Name{i}": 1500 + i for i in range(12)},
                "all_names_data": {"Male": [], "Female": []},
            },
        )

        names = [f"Name{i}" for i in range(12)]
        with patch("st_name_ranking.interface.rankings_ui.st", mock_st):
            rankings_ui.render_rankings(names)

        assert mock_st.dataframe.call_count >= 1
        mock_st.info.assert_any_call("Preference landscape appears after at least 25 rated names.")

    def test_render_rankings_landscape_handles_model_data(self):
        """Test rankings landscape renders without raising on valid model outputs."""
        mock_st = MagicMock()
        mock_st.header = MagicMock()
        mock_st.write = MagicMock()
        mock_st.dataframe = MagicMock()
        mock_st.info = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.selectbox = MagicMock(return_value="All")
        mock_st.slider = MagicMock(return_value=42)
        mock_st.altair_chart = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.status = MagicMock()
        mock_st.cache_data = lambda **_: lambda func: func

        def create_tab():
            tab = MagicMock()
            tab.__enter__ = MagicMock(return_value=tab)
            tab.__exit__ = MagicMock(return_value=None)
            return tab

        mock_st.tabs = MagicMock(return_value=[create_tab(), create_tab(), create_tab()])
        names = [f"Name{i}" for i in range(30)]
        mock_st.session_state = MockSessionState(
            {
                "ratings": {name: 1500 + i for i, name in enumerate(names)},
                "all_names_data": {"Male": [], "Female": []},
            },
        )

        mock_model = MagicMock()
        mock_model.feature_names = [f"f{i}" for i in range(6)]
        mock_model.state.weight_mean = np.array([0.8, -0.5, 0.3, 0.1, -0.2, 0.05])
        mock_model.state.weight_cov = np.eye(6)
        features = np.random.randn(30, 6)

        with (
            patch("st_name_ranking.interface.rankings_ui.st", mock_st),
            patch(
                "st_name_ranking.interface.rankings_ui.get_or_initialize_active_learning_model",
                return_value=mock_model,
            ),
            patch("st_name_ranking.interface.rankings_ui.get_names_features", return_value=features),
        ):
            rankings_ui.render_rankings(names)

        assert mock_st.dataframe.call_count >= 3
        assert mock_st.selectbox.call_count == 0
        mock_st.status.assert_called_once()
