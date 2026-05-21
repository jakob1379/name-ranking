"""Tests for queued name display data."""

from st_name_ranking.learning.name_queue import compute_name_data


def test_compute_name_data_returns_fixed_item_shape():
    item = compute_name_data("Anna", 3, {"Anna": True})

    assert item == {
        "name": "Anna",
        "index": 3,
        "status": True,
        "status_text": "Included",
        "border_color": "#4CAF50",
        "bg_color": "#E8F5E9",
    }


def test_compute_name_data_marks_missing_inclusion_as_undecided():
    item = compute_name_data("Bo", 4, {})

    assert item["status"] is None
    assert item["status_text"] == "Not decided"
