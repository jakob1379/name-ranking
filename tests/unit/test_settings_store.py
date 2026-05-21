"""Tests for user-setting persistence."""

from st_name_ranking.persistence.settings_store import load_user_setting, save_user_setting


def test_settings_store_loads_default_and_round_trips_values(initialized_db):
    assert load_user_setting("theme", default="system") == "system"

    save_user_setting("theme", "dark")
    assert load_user_setting("theme", default="system") == "dark"

    save_user_setting("theme", "light")
    assert load_user_setting("theme", default="system") == "light"
