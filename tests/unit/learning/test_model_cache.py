"""Unit tests for model-level cache keys."""

from st_name_ranking.learning import model
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
