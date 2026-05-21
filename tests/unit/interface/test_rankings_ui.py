"""Unit tests for rankings UI render helpers."""

from unittest.mock import MagicMock

from st_name_ranking.interface import rankings_ui
from st_name_ranking.types import PreferenceStats


def _mock_streamlit() -> MagicMock:
    mock_st = MagicMock()
    mock_st.column_config.TextColumn = MagicMock(return_value="text-column")
    mock_st.column_config.NumberColumn = MagicMock(return_value="number-column")
    mock_st.expander.return_value.__enter__.return_value = None
    mock_st.expander.return_value.__exit__.return_value = None
    return mock_st


def test_render_preferences_panel_uses_section_configuration(monkeypatch):
    mock_st = _mock_streamlit()
    monkeypatch.setattr(rankings_ui, "st", mock_st)
    monkeypatch.setattr(
        rankings_ui,
        "get_preference_stats_by_gender",
        lambda: {"Female": PreferenceStats(wins=3, losses=1, draws=1, total=5)},
    )
    monkeypatch.setattr(rankings_ui, "get_preference_stats_by_origin", dict)
    monkeypatch.setattr(rankings_ui, "get_preference_stats_by_phonetic", lambda: None)

    rankings_ui.render_preferences_panel()

    mock_st.subheader.assert_any_call("Overall Preferences")
    mock_st.subheader.assert_any_call("Gender Preferences", divider="gray")
    mock_st.bar_chart.assert_called_once()
    mock_st.expander.assert_called_once_with("Detailed Gender Preferences Statistics", expanded=False)
    mock_st.dataframe.assert_called_once()
    mock_st.caption.assert_called_once()

    info_messages = [call.args[0] for call in mock_st.info.call_args_list]
    assert any("Insight" in message for message in info_messages)
    assert "No origin preference data available." in info_messages
    assert "No phonetic preference data available." in info_messages
