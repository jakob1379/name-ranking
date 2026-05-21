"""Tournament orchestration vote-flow tests."""

from unittest.mock import Mock

from st_name_ranking import tournament_orchestration
from st_name_ranking.active_learning.lazy_updates import ModelUpdateStatus


def test_prepare_tournament_round_reuses_current_pair_without_touching_queue():
    manager = Mock()
    manager.get_pair.side_effect = AssertionError("existing pair should be reused")
    queue_stats = {"queue_size": 3, "target_size": 10}

    result = tournament_orchestration.prepare_tournament_round(
        ["Anna", "Bo", "Maria"],
        manager,
        ("Anna", "Bo"),
        queue_stats,
    )

    assert result.manager is manager
    assert result.candidate_a == "Anna"
    assert result.candidate_b == "Bo"
    assert result.queue_stats == queue_stats
    manager.get_pair.assert_not_called()


def test_prepare_tournament_round_uses_queue_pair_when_no_current_pair():
    manager = Mock()
    manager.get_pair.return_value = ("Maria", "Peter")
    queue_stats = {"queue_size": 1, "target_size": 10}

    result = tournament_orchestration.prepare_tournament_round(
        ["Anna", "Bo", "Maria", "Peter"],
        manager,
        None,
        queue_stats,
    )

    assert result.candidate_a == "Maria"
    assert result.candidate_b == "Peter"
    assert result.queue_stats == queue_stats
    manager.get_pair.assert_called_once_with()


def test_prepare_tournament_round_falls_back_to_random_pair_when_queue_empty(monkeypatch):
    manager = Mock()
    manager.get_pair.return_value = None
    select_random_pair = Mock(return_value=("Anna", "Peter"))
    monkeypatch.setattr(tournament_orchestration, "select_random_pair", select_random_pair)

    result = tournament_orchestration.prepare_tournament_round(
        ["Anna", "Bo", "Maria", "Peter"],
        manager,
        None,
        None,
    )

    assert result.candidate_a == "Anna"
    assert result.candidate_b == "Peter"
    assert result.queue_stats is None
    manager.get_pair.assert_called_once_with()
    select_random_pair.assert_called_once_with(["Anna", "Bo", "Maria", "Peter"])


def test_prepare_tournament_round_requires_at_least_two_names():
    manager = Mock()

    try:
        tournament_orchestration.prepare_tournament_round(["Anna"], manager, None, None)
    except ValueError as err:
        assert "Need at least 2 names" in str(err)
    else:
        raise AssertionError("Expected ValueError for one-name tournament")


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


def test_record_tournament_vote_falls_back_to_random_pair_after_recorded_vote(monkeypatch):
    manager = Mock()
    manager.get_pair.return_value = None
    select_random_pair = Mock(return_value=("Maria", "Peter"))
    monkeypatch.setattr(tournament_orchestration, "select_random_pair", select_random_pair)
    monkeypatch.setattr(
        tournament_orchestration,
        "record_comparison_instant",
        Mock(return_value=ModelUpdateStatus(recorded=True, model_updated=False, ratings_fresh=True)),
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
    assert result.pair_source == "random"
    assert result.update_status.recorded is True
    manager.get_pair.assert_called_once_with()
    select_random_pair.assert_called_once_with(["Anna", "Bo", "Maria", "Peter"])


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
