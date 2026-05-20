"""Unit tests for active-learning pair selection policies."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

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
