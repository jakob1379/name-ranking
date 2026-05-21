"""Tests for active-learning queue manager lifecycle."""

from unittest.mock import Mock, patch

from st_name_ranking.active_learning import queue
from st_name_ranking.interface import tournament_session


def test_queue_manager_refill_adds_model_selected_pairs(monkeypatch):
    manager = queue.QueueManager(
        ["Anna", "Bo", "Clara", "Dana"],
        target_size=3,
        refill_threshold=1,
        sample_size=4,
    )
    select_batch = Mock(
        return_value=[
            ("Anna", "Bo"),
            ("Bo", "Anna"),
            ("Clara", "Dana"),
        ],
    )
    monkeypatch.setattr(queue, "select_candidate_pairs", select_batch)

    manager._refill_queue()

    assert list(manager.queue) == [("Anna", "Bo"), ("Clara", "Dana")]
    stats = manager.get_stats()
    assert stats["refill_count"] == 1
    assert stats["last_refill_added"] == 2
    assert stats["thread_alive"] is False
    select_batch.assert_called_once()
    options = select_batch.call_args.kwargs["options"]
    assert options.batch_size == 3
    assert options.sample_size == 4
    assert options.min_training_samples == queue.MIN_TRAINING_SAMPLES
    assert options.fallback == "random"


def test_queue_manager_refill_ignores_existing_pairs_in_reverse_order(monkeypatch):
    manager = queue.QueueManager(
        ["Anna", "Bo", "Clara"],
        target_size=3,
        refill_threshold=1,
        sample_size=3,
    )
    manager.queue.append(("Anna", "Bo"))
    monkeypatch.setattr(
        queue,
        "select_candidate_pairs",
        Mock(return_value=[("Bo", "Anna"), ("Anna", "Clara")]),
    )

    manager._refill_queue()

    assert list(manager.queue) == [("Anna", "Bo"), ("Anna", "Clara")]
    assert manager.get_stats()["last_refill_added"] == 1


def test_queue_manager_refill_normalizes_reversed_existing_queue_pairs(monkeypatch):
    manager = queue.QueueManager(
        ["Anna", "Bo", "Clara"],
        target_size=3,
        refill_threshold=1,
        sample_size=3,
    )
    manager.queue.append(("Bo", "Anna"))
    monkeypatch.setattr(
        queue,
        "select_candidate_pairs",
        Mock(return_value=[("Anna", "Bo"), ("Bo", "Clara")]),
    )

    manager._refill_queue()

    assert list(manager.queue) == [("Bo", "Anna"), ("Bo", "Clara")]
    assert manager.get_stats()["last_refill_added"] == 1


def test_queue_manager_refill_without_pairs_leaves_stats_unchanged(monkeypatch):
    manager = queue.QueueManager(
        ["Anna", "Bo"],
        target_size=2,
        refill_threshold=1,
        sample_size=2,
    )
    monkeypatch.setattr(queue, "select_candidate_pairs", Mock(return_value=[]))

    manager._refill_queue()

    assert manager.pop_next_pair() is None
    stats = manager.get_stats()
    assert stats["refill_count"] == 0
    assert stats["last_refill_added"] == 0


def test_queue_manager_pop_next_pair_pops_pairs_in_order(monkeypatch):
    manager = queue.QueueManager(
        ["Anna", "Bo", "Clara"],
        target_size=2,
        refill_threshold=1,
        sample_size=3,
    )
    monkeypatch.setattr(
        queue,
        "select_candidate_pairs",
        Mock(return_value=[("Anna", "Bo"), ("Bo", "Clara")]),
    )
    manager._refill_queue()

    assert manager.pop_next_pair() == ("Anna", "Bo")
    assert manager.pop_next_pair() == ("Bo", "Clara")
    assert manager.pop_next_pair() is None


def test_queue_manager_recreated_when_middle_names_change(monkeypatch):
    """Queue reuse should compare the full names tuple, not just list edges."""
    session_state: dict[str, object] = {}
    monkeypatch.setattr(tournament_session.st, "session_state", session_state)

    with (
        patch.object(queue.QueueManager, "start", autospec=True),
        patch.object(queue.QueueManager, "stop", autospec=True) as stop,
    ):
        first = tournament_session.get_or_start_queue_manager(
            ["Anna", "Bo", "Clara"],
            target_size=2,
            sample_size=10,
        )
        second = tournament_session.get_or_start_queue_manager(
            ["Anna", "Dana", "Clara"],
            target_size=2,
            sample_size=10,
        )

    assert second is not first
    stop.assert_called_once_with(first)


def test_queue_manager_reused_for_exact_same_names(monkeypatch):
    """Identical queue settings and names should reuse the existing manager."""
    session_state: dict[str, object] = {}
    monkeypatch.setattr(tournament_session.st, "session_state", session_state)

    with (
        patch.object(queue.QueueManager, "start", autospec=True),
        patch.object(queue.QueueManager, "stop", autospec=True) as stop,
    ):
        first = tournament_session.get_or_start_queue_manager(
            ["Anna", "Bo", "Clara"],
            target_size=2,
            sample_size=10,
        )
        second = tournament_session.get_or_start_queue_manager(
            ["Anna", "Bo", "Clara"],
            target_size=2,
            sample_size=10,
        )

    assert second is first
    stop.assert_not_called()


def test_queue_manager_recreated_when_target_size_changes(monkeypatch):
    session_state: dict[str, object] = {}
    monkeypatch.setattr(tournament_session.st, "session_state", session_state)

    with (
        patch.object(queue.QueueManager, "start", autospec=True),
        patch.object(queue.QueueManager, "stop", autospec=True) as stop,
    ):
        first = tournament_session.get_or_start_queue_manager(
            ["Anna", "Bo", "Clara"],
            target_size=2,
            sample_size=10,
        )
        second = tournament_session.get_or_start_queue_manager(
            ["Anna", "Bo", "Clara"],
            target_size=3,
            sample_size=10,
        )

    assert second is not first
    stop.assert_called_once_with(first)


def test_queue_manager_stats_reports_session_manager(monkeypatch):
    manager = queue.QueueManager(["Anna", "Bo"], target_size=2, sample_size=2)
    monkeypatch.setattr(
        tournament_session.st,
        "session_state",
        {tournament_session.QUEUE_MANAGER_KEY: manager},
    )

    stats = tournament_session.get_queue_manager_stats()

    assert stats is not None
    assert stats["queue_size"] == 0
    assert stats["num_names"] == 2
    assert stats["thread_alive"] is False
