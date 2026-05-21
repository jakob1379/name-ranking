"""Tournament UI lifecycle guards."""

from unittest.mock import Mock

from st_name_ranking.active_learning.lazy_updates import ModelUpdateStatus
from st_name_ranking.interface import ui
from st_name_ranking.tournament_orchestration import VoteResult


class SessionState(dict):
    """Small Streamlit session_state stand-in for UI helper tests."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def test_render_tournament_empty_names_stops_before_queue(monkeypatch):
    prepare_round = Mock(side_effect=AssertionError("queue should not be prepared"))
    info = Mock()
    monkeypatch.setattr(ui, "prepare_tournament_round", prepare_round)
    monkeypatch.setattr(ui.st, "info", info)

    ui.render_tournament.__wrapped__([])

    prepare_round.assert_not_called()
    info.assert_called_once_with("No names to compare. Please select at least two names.")


def test_render_tournament_single_name_stops_before_queue(monkeypatch):
    prepare_round = Mock(side_effect=AssertionError("queue should not be prepared"))
    info = Mock()
    monkeypatch.setattr(ui, "prepare_tournament_round", prepare_round)
    monkeypatch.setattr(ui.st, "info", info)

    ui.render_tournament.__wrapped__(["Anna"])

    prepare_round.assert_not_called()
    info.assert_called_once_with("Only one name ('Anna') selected. Please select at least two names to compare.")


def test_record_vote_keeps_current_pair_when_vote_was_not_saved(monkeypatch):
    session_state = SessionState(
        candidate_a="Anna",
        candidate_b="Bo",
        ratings={"Anna": 1500.0, "Bo": 1500.0},
    )
    toast = Mock()
    record_vote = Mock(
        return_value=VoteResult(
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
        ),
    )
    monkeypatch.setattr(ui.st, "session_state", session_state)
    monkeypatch.setattr(ui.st, "toast", toast)
    monkeypatch.setattr(ui, "record_tournament_vote", record_vote)

    next_pair = ui._record_vote_and_get_next_pair(["Anna", "Bo", "Maria"], Mock(), -1)

    assert next_pair == ("Anna", "Bo")
    assert session_state.candidate_a == "Anna"
    assert session_state.candidate_b == "Bo"
    toast.assert_called_once_with("Vote was not saved: database locked", icon="❌", duration="long")
