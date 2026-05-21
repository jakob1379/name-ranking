"""Focused tests for Streamlit main-flow decisions."""

from unittest.mock import Mock

import pytest

from st_name_ranking.interface import main as main_module


class SessionState(dict):
    """Small Streamlit session_state stand-in for smoke-level tests."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


@pytest.mark.parametrize(
    ("active_tab", "expected_renderer", "expected_names"),
    [
        ("Name Filter", "binary_filter", ["Anna", "Peter", "Maria"]),
        ("Tournament", "tournament", ["Anna", "Peter"]),
        ("Rankings", "rankings", ["Anna", "Peter"]),
        ("Similarity Search", "similarity", ["Anna", "Peter"]),
    ],
)
def test_resolve_active_tab_render_selects_renderer_and_dataset(
    active_tab,
    expected_renderer,
    expected_names,
):
    filtered_names = ["Anna", "Peter", "Maria"]
    included_names = ["Anna", "Peter"]

    result = main_module.resolve_active_tab_render(
        active_tab,
        filtered_names,
        included_names,
    )

    assert result.renderer == expected_renderer
    assert result.names == expected_names


def test_resolve_active_tab_render_falls_back_to_similarity_for_unknown_tab():
    result = main_module.resolve_active_tab_render(
        "Legacy tab",
        ["Anna", "Peter", "Maria"],
        ["Anna", "Peter"],
    )

    assert result.renderer == "similarity"
    assert result.names == ["Anna", "Peter"]


def test_main_delegates_loaded_name_flow(monkeypatch):
    st = Mock()
    st.session_state = SessionState(
        {
            "all_names_data": {"All": ["Anna", "Peter", "Maria"]},
            "all_names": ["Anna", "Peter", "Maria"],
            "active_tab": "Name Filter",
        },
    )
    rendered_tab = Mock()

    monkeypatch.setattr(main_module, "st", st)
    monkeypatch.setattr(main_module, "render_sidebar", Mock())
    monkeypatch.setattr(
        main_module,
        "resolve_filtered_names",
        Mock(return_value=["Anna", "Peter", "Maria"]),
    )
    monkeypatch.setattr(main_module, "_get_included_names", Mock(return_value=["Anna", "Peter"]))
    monkeypatch.setattr(main_module, "resolve_tournament_sample_size", Mock())
    monkeypatch.setattr(main_module, "_render_active_tab", rendered_tab)

    main_module.main()

    rendered_tab.assert_called_once_with(["Anna", "Peter", "Maria"], ["Anna", "Peter"])
    assert st.session_state.tournament_filtered_count == 2
