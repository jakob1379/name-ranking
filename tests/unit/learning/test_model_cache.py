"""Unit tests for model-level cache keys."""

import numpy as np

from st_name_ranking.learning import model
from st_name_ranking.learning.model import BradleyTerryModel, CandidatePairScores
from st_name_ranking.types import PhoneticCodes


def test_phonetic_codes_cache_is_scoped_to_database_path(monkeypatch, tmp_path):
    """Identical name batches from different databases should not share cached rows."""
    calls = []

    def get_phonetic_codes_batch(names):
        calls.append(list(names))
        return {names[0]: PhoneticCodes(primary=f"call-{len(calls)}", secondary="")}

    monkeypatch.setattr(model, "get_phonetic_codes_batch", get_phonetic_codes_batch)
    model._get_phonetic_codes_cached.cache_clear()
    first_db = str(tmp_path / "first.db")
    second_db = str(tmp_path / "second.db")

    try:
        first = model._get_phonetic_codes_cached(first_db, ("Anna",))
        first_again = model._get_phonetic_codes_cached(first_db, ("Anna",))
        second = model._get_phonetic_codes_cached(second_db, ("Anna",))
    finally:
        model._get_phonetic_codes_cached.cache_clear()

    assert first is first_again
    assert first["Anna"].primary == "call-1"
    assert second["Anna"].primary == "call-2"
    assert calls == [["Anna"], ["Anna"]]


def test_update_batch_updates_training_samples_and_weights():
    bt_model = BradleyTerryModel(["length", "vowels"])
    initial_weights = bt_model.state.weight_mean.copy()

    bt_model.update_batch(
        [
            (
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                -1,
            ),
        ],
    )

    assert bt_model.state.training_samples == 1
    assert not np.allclose(bt_model.state.weight_mean, initial_weights)


def test_select_top_k_pairs_fills_missing_pairs(monkeypatch):
    bt_model = BradleyTerryModel(["length", "vowels"])
    names = ["Anna", "Peter", "Maria"]
    candidates = CandidatePairScores(
        idx_a=np.array([0]),
        idx_b=np.array([1]),
        score=np.array([1.0]),
    )
    monkeypatch.setattr(bt_model, "_score_candidate_pairs", lambda _features, _names: candidates)

    pairs = bt_model.select_top_k_pairs(np.zeros((3, 2)), names, k=3)

    assert [(pair.idx_a, pair.idx_b) for pair in pairs] == [(0, 1), (0, 2), (1, 2)]
