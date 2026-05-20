"""Unit tests for lazy active-learning model updates."""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import Mock

import numpy as np

from st_name_ranking.active_learning import lazy_updates


class ImmediateExecutor:
    def __init__(self):
        self.submissions = []

    def submit(self, func, *args):
        self.submissions.append((func, args))
        future = Future()
        future.set_result(func(*args))
        return future


def test_record_comparison_returns_fallback_status_when_database_write_fails(monkeypatch):
    monkeypatch.setattr(
        lazy_updates.database,
        "record_comparison",
        Mock(side_effect=RuntimeError("database locked")),
    )
    monkeypatch.setattr(
        lazy_updates,
        "get_thread_executor",
        Mock(side_effect=AssertionError("executor should not start after failed write")),
    )

    status = lazy_updates.record_comparison_instant("Anna", "Peter", -1, blocking=True)

    assert not status.recorded
    assert not status.model_updated
    assert not status.ratings_fresh
    assert status.fallback_used
    assert status.error == "database locked"


def test_record_comparison_blocking_reports_completed_model_and_rating_updates(monkeypatch):
    executor = ImmediateExecutor()
    record_comparison = Mock()

    monkeypatch.setattr(lazy_updates.database, "record_comparison", record_comparison)
    monkeypatch.setattr(lazy_updates, "get_thread_executor", Mock(return_value=executor))
    monkeypatch.setattr(lazy_updates, "_update_model_sync", Mock(return_value=True))
    monkeypatch.setattr(lazy_updates, "_update_ratings_from_model", Mock(return_value=True))

    status = lazy_updates.record_comparison_instant("Anna", "Peter", -1, blocking=True)

    record_comparison.assert_called_once_with("Anna", "Peter", -1)
    assert len(executor.submissions) == 0
    assert status.recorded
    assert status.model_updated
    assert status.ratings_fresh
    assert not status.fallback_used
    assert status.error is None


def test_record_comparison_blocking_marks_fallback_when_model_update_fails(monkeypatch):
    executor = ImmediateExecutor()

    monkeypatch.setattr(lazy_updates.database, "record_comparison", Mock())
    monkeypatch.setattr(lazy_updates, "get_thread_executor", Mock(return_value=executor))
    monkeypatch.setattr(lazy_updates, "_update_model_sync", Mock(return_value=False))
    update_ratings = Mock(return_value=True)
    monkeypatch.setattr(lazy_updates, "_update_ratings_from_model", update_ratings)

    status = lazy_updates.record_comparison_instant("Anna", "Peter", -1, blocking=True)

    assert status.recorded
    assert not status.model_updated
    assert not status.ratings_fresh
    assert status.fallback_used
    assert status.error == "model or rating refresh failed"
    update_ratings.assert_not_called()


def test_record_comparison_nonblocking_returns_pending_status_after_scheduling(monkeypatch):
    executor = ImmediateExecutor()

    monkeypatch.setattr(lazy_updates.database, "record_comparison", Mock())
    monkeypatch.setattr(lazy_updates, "get_thread_executor", Mock(return_value=executor))
    monkeypatch.setattr(lazy_updates, "_update_model_sync", Mock(return_value=True))
    monkeypatch.setattr(lazy_updates, "_update_ratings_from_model", Mock(return_value=True))

    status = lazy_updates.record_comparison_instant("Anna", "Peter", -1, blocking=False)

    assert len(executor.submissions) == 1
    assert executor.submissions[0] == (lazy_updates._update_model_then_refresh_ratings, ("Anna", "Peter", -1))
    assert status.recorded
    assert status.model_updated is None
    assert status.ratings_fresh is None
    assert not status.fallback_used


def test_update_model_then_refresh_ratings_runs_in_order(monkeypatch):
    calls = []

    def update_model(*_args):
        calls.append("model")
        return True

    def update_ratings():
        calls.append("ratings")
        return True

    monkeypatch.setattr(lazy_updates, "_update_model_sync", update_model)
    monkeypatch.setattr(lazy_updates, "_update_ratings_from_model", update_ratings)

    assert lazy_updates._update_model_then_refresh_ratings("Anna", "Peter", -1) == (True, True)
    assert calls == ["model", "ratings"]


def test_update_model_then_refresh_ratings_skips_ratings_after_model_failure(monkeypatch):
    update_ratings = Mock(return_value=True)

    monkeypatch.setattr(lazy_updates, "_update_model_sync", Mock(return_value=False))
    monkeypatch.setattr(lazy_updates, "_update_ratings_from_model", update_ratings)

    assert lazy_updates._update_model_then_refresh_ratings("Anna", "Peter", -1) == (False, False)
    update_ratings.assert_not_called()


def test_update_model_sync_uses_both_disliked_update(monkeypatch):
    model = Mock()
    features_a = np.array([1.0, 0.0])
    features_b = np.array([0.0, 1.0])

    monkeypatch.setattr(lazy_updates, "get_active_learning_model", Mock(return_value=model))
    monkeypatch.setattr(lazy_updates, "get_name_features", Mock(side_effect=[features_a, features_b]))

    updated = lazy_updates._update_model_sync(
        "Anna",
        "Peter",
        lazy_updates.BOTH_DISLIKED_PREFERENCE,
    )

    assert updated
    model.update_both_disliked.assert_called_once()
    assert model.update_both_disliked.call_args.args == (features_a, features_b)
    model.update.assert_not_called()
    model.save_to_db.assert_called_once_with()


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeConnection:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query):
        assert query == "SELECT name FROM names"
        return FakeCursor(self._rows)


def test_update_ratings_from_model_writes_batch_ratings(monkeypatch):
    model = Mock()
    model.get_utility.return_value = np.array([0.2, -0.1])
    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    update_batch = Mock()

    monkeypatch.setattr(lazy_updates, "get_active_learning_model", Mock(return_value=model))
    monkeypatch.setattr(
        lazy_updates.database,
        "get_connection",
        Mock(return_value=FakeConnection([("Anna",), ("Peter",)])),
    )
    monkeypatch.setattr(lazy_updates, "get_names_features", Mock(return_value=features))
    monkeypatch.setattr(lazy_updates.database, "update_ratings_batch_values", update_batch)

    assert lazy_updates._update_ratings_from_model()

    model.get_utility.assert_called_once_with(features)
    update_batch.assert_called_once_with({"Anna": 1600.0, "Peter": 1450.0})
