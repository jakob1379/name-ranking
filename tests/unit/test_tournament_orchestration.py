"""Tournament orchestration vote-flow tests."""

from unittest.mock import Mock

from st_name_ranking import tournament_orchestration
from st_name_ranking.active_learning.lazy_updates import ModelUpdateStatus


def test_record_tournament_vote_advances_after_recorded_vote(monkeypatch):
    manager = Mock()
    manager.get_pair.return_value = ("Maria", "Peter")
    monkeypatch.setattr(
        tournament_orchestration,
        "record_comparison_instant",
        Mock(return_value=ModelUpdateStatus(recorded=True, model_updated=None, ratings_fresh=None)),
    )

    result = tournament_orchestration.record_tournament_vote(
        ["Anna", "Bo", "Maria", "Peter"],
        manager,
        "Anna",
        "Bo",
        -1,
    )

    assert result.previous_pair == ("Anna", "Bo")
    assert result.next_pair == ("Maria", "Peter")
    assert result.pair_source == "queue"
    assert result.update_status.recorded is True
    manager.get_pair.assert_called_once_with()


def test_record_tournament_vote_keeps_pair_when_vote_was_not_recorded(monkeypatch):
    manager = Mock()
    manager.get_pair.side_effect = AssertionError("next pair should not be selected")
    monkeypatch.setattr(
        tournament_orchestration,
        "record_comparison_instant",
        Mock(
            return_value=ModelUpdateStatus(
                recorded=False,
                model_updated=False,
                ratings_fresh=False,
                fallback_used=True,
                error="database locked",
            ),
        ),
    )

    result = tournament_orchestration.record_tournament_vote(
        ["Anna", "Bo", "Maria", "Peter"],
        manager,
        "Anna",
        "Bo",
        -1,
    )

    assert result.previous_pair == ("Anna", "Bo")
    assert result.next_pair == ("Anna", "Bo")
    assert result.pair_source == "unchanged"
    assert result.update_status.recorded is False
    assert result.update_status.error == "database locked"
    manager.get_pair.assert_not_called()
