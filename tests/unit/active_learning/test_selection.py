"""Unit tests for active-learning pair selection policies."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from st_name_ranking.active_learning import selection
from st_name_ranking.active_learning.selection import (
    PairSelectionDependencies,
    PairSelectionOptions,
    select_candidate_pairs,
)


def test_select_candidate_pairs_uses_random_fallback_before_min_training_samples(monkeypatch):
    model = Mock()
    model.state = SimpleNamespace(training_samples=0)
    dependencies = PairSelectionDependencies(
        model_provider=Mock(return_value=model),
        features_provider=Mock(side_effect=AssertionError("model features should not be loaded during fallback")),
        warning_logger=Mock(),
    )
    monkeypatch.setattr(
        "st_name_ranking.active_learning.selection.select_random_batch",
        Mock(return_value=[("Anna", "Peter")]),
    )

    pairs = select_candidate_pairs(
        ["Anna", "Peter", "Maria"],
        options=PairSelectionOptions(
            batch_size=2,
            min_training_samples=3,
            fallback="random",
        ),
        dependencies=dependencies,
    )

    assert pairs == [("Anna", "Peter")]
    model.select_top_k_pairs.assert_not_called()


def test_select_candidate_pairs_uses_model_after_min_training_samples():
    model = Mock()
    model.state = SimpleNamespace(training_samples=3)
    model_pair = SimpleNamespace(name_a="Anna", name_b="Maria")
    model.select_pair.return_value = model_pair
    features = np.array([[1.0], [2.0], [3.0]])
    dependencies = PairSelectionDependencies(
        model_provider=Mock(return_value=model),
        features_provider=Mock(return_value=features),
        warning_logger=Mock(),
    )

    pairs = select_candidate_pairs(
        ["Anna", "Peter", "Maria"],
        options=PairSelectionOptions(min_training_samples=3, fallback="random"),
        dependencies=dependencies,
    )

    assert pairs == [("Anna", "Maria")]
    model.select_pair.assert_called_once_with(features, ["Anna", "Peter", "Maria"])


def test_active_learning_singletons_initialize_once_under_concurrency(monkeypatch):
    selection.reset_active_learning_state()
    extractor_instances = []
    model_instances = []

    class FakeFeatureExtractor:
        def get_feature_names(self):
            return ["length"]

    def new_feature_extractor():
        time.sleep(0.01)
        extractor = FakeFeatureExtractor()
        extractor_instances.append(extractor)
        return extractor

    def initialize_model(feature_names):
        time.sleep(0.01)
        model = SimpleNamespace(feature_names=feature_names)
        model_instances.append(model)
        return model

    monkeypatch.setattr(selection, "FeatureExtractor", new_feature_extractor)
    monkeypatch.setattr(selection, "initialize_model_if_needed", initialize_model)
    barrier = threading.Barrier(10)
    models = []
    errors = []

    def get_model_worker():
        try:
            barrier.wait()
            models.append(selection.get_or_initialize_active_learning_model())
        except Exception as exc:
            errors.append(exc)

    try:
        threads = [threading.Thread(target=get_model_worker) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert len({id(model) for model in models}) == 1
        assert len(extractor_instances) == 1
        assert len(model_instances) == 1
    finally:
        selection.reset_active_learning_state()


def test_reset_active_learning_state_clears_model_and_extractor_caches():
    selection._ACTIVE_LEARNING_MODEL_CACHE = object()
    selection._ACTIVE_LEARNING_MODEL_CACHE_DB_PATH = "model.db"
    selection._FEATURE_EXTRACTOR_CACHE = object()
    selection._FEATURE_EXTRACTOR_CACHE_DB_PATH = "extractor.db"

    selection.reset_active_learning_state()

    assert selection._ACTIVE_LEARNING_MODEL_CACHE is None
    assert selection._ACTIVE_LEARNING_MODEL_CACHE_DB_PATH is None
    assert selection._FEATURE_EXTRACTOR_CACHE is None
    assert selection._FEATURE_EXTRACTOR_CACHE_DB_PATH is None


def test_active_learning_singletons_reinitialize_after_db_path_change(monkeypatch, tmp_path):
    selection.reset_active_learning_state()
    original_path = selection.database.get_db_path()
    extractor_instances = []
    model_instances = []

    class FakeFeatureExtractor:
        def get_feature_names(self):
            return ["length"]

    def new_feature_extractor():
        extractor = FakeFeatureExtractor()
        extractor_instances.append(extractor)
        return extractor

    def initialize_model(feature_names):
        model = SimpleNamespace(feature_names=feature_names)
        model_instances.append(model)
        return model

    monkeypatch.setattr(selection, "FeatureExtractor", new_feature_extractor)
    monkeypatch.setattr(selection, "initialize_model_if_needed", initialize_model)

    try:
        selection.database.set_db_path(tmp_path / "first.db")
        first_model = selection.get_or_initialize_active_learning_model()
        first_extractor = selection.get_or_create_feature_extractor()

        selection.database.set_db_path(tmp_path / "second.db")
        second_model = selection.get_or_initialize_active_learning_model()
        second_extractor = selection.get_or_create_feature_extractor()
    finally:
        selection.database.set_db_path(original_path)
        selection.reset_active_learning_state()

    assert first_model is not second_model
    assert first_extractor is not second_extractor
    assert len(model_instances) == 2
    assert len(extractor_instances) == 2
