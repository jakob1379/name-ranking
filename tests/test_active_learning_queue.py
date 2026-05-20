"""Tests for active-learning queue manager lifecycle."""

from unittest.mock import patch

from st_name_ranking.active_learning import queue


def test_queue_manager_recreated_when_middle_names_change(monkeypatch):
    """Queue reuse should compare the full names tuple, not just list edges."""
    session_state: dict[str, object] = {}
    monkeypatch.setattr(queue.st, "session_state", session_state)

    with (
        patch.object(queue.QueueManager, "start", autospec=True),
        patch.object(queue.QueueManager, "stop", autospec=True) as stop,
    ):
        first = queue.get_or_start_queue_manager(
            ["Anna", "Bo", "Clara"],
            target_size=2,
            sample_size=10,
        )
        second = queue.get_or_start_queue_manager(
            ["Anna", "Dana", "Clara"],
            target_size=2,
            sample_size=10,
        )

    assert second is not first
    stop.assert_called_once_with(first)


def test_queue_manager_reused_for_exact_same_names(monkeypatch):
    """Identical queue settings and names should reuse the existing manager."""
    session_state: dict[str, object] = {}
    monkeypatch.setattr(queue.st, "session_state", session_state)

    with (
        patch.object(queue.QueueManager, "start", autospec=True),
        patch.object(queue.QueueManager, "stop", autospec=True) as stop,
    ):
        first = queue.get_or_start_queue_manager(
            ["Anna", "Bo", "Clara"],
            target_size=2,
            sample_size=10,
        )
        second = queue.get_or_start_queue_manager(
            ["Anna", "Bo", "Clara"],
            target_size=2,
            sample_size=10,
        )

    assert second is first
    stop.assert_not_called()
