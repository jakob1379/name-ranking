"""Tests for shared Streamlit UI support helpers."""

from st_name_ranking.interface import ui_support


def test_render_timer_start_uses_perf_counter(monkeypatch):
    monkeypatch.setattr(ui_support.time, "perf_counter", lambda: 12.5)

    timer = ui_support.RenderTimer.start("Filter")

    assert timer.label == "Filter"
    assert timer.start_time == 12.5


def test_render_timer_log_reports_elapsed_milliseconds(monkeypatch, caplog):
    readings = iter([1.0, 1.25])
    monkeypatch.setattr(ui_support.time, "perf_counter", lambda: next(readings))
    timer = ui_support.RenderTimer.start("Tournament")

    with caplog.at_level("DEBUG", logger=ui_support.logger.name):
        timer.log("loaded")

    assert "Tournament [loaded]: 250.00ms" in caplog.text


def test_ui_support_thresholds_are_grouped_for_renderers():
    assert ui_support.MIN_NAMES_FOR_COMPARISON == 2
    assert ui_support.MIN_NAMES_FOR_LANDSCAPE > ui_support.MIN_NAMES_FOR_COMPARISON
    assert ui_support.FAST_REFILL_THRESHOLD_MS < ui_support.MODERATE_REFILL_THRESHOLD_MS
    assert ui_support.FILTER_SAVE_INTERVAL > 0
    assert ui_support.MAX_EXCLUDED_NAMES_DISPLAY > ui_support.FILTER_SAVE_INTERVAL
