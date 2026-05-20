"""Tournament UI lifecycle guards."""

from unittest.mock import Mock

from st_name_ranking import ui


def test_render_tournament_empty_names_stops_before_queue(monkeypatch):
    prepare_round = Mock(side_effect=AssertionError("queue should not be prepared"))
    info = Mock()
    monkeypatch.setattr(ui, "prepare_tournament_round", prepare_round)
    monkeypatch.setattr(ui.st, "info", info)

    ui.render_tournament.__wrapped__([])

    prepare_round.assert_not_called()
    info.assert_called_once_with("No names to compare. Please select at least two names.")


def test_render_tournament_single_name_stops_before_queue(monkeypatch):
    prepare_round = Mock(side_effect=AssertionError("queue should not be prepared"))
    info = Mock()
    monkeypatch.setattr(ui, "prepare_tournament_round", prepare_round)
    monkeypatch.setattr(ui.st, "info", info)

    ui.render_tournament.__wrapped__(["Anna"])

    prepare_round.assert_not_called()
    info.assert_called_once_with("Only one name ('Anna') selected. Please select at least two names to compare.")
