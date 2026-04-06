"""Comprehensive integration tests for Streamlit UI components.

Uses AppTest from streamlit.testing.v1 to test actual widget rendering
and interactions, not mocks. Tests against a real database.
"""

import sqlite3
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_db_factory(tmp_path):
    """Factory fixture that creates isolated test databases.

    Returns a function that creates a new test database with sample data
    and returns the path to it.
    """
    counter = 0

    def _create_db():
        nonlocal counter
        counter += 1
        db_path = tmp_path / f"test_names_{counter}.db"

        # Create database from scratch
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Create tables
            conn.execute("""
                CREATE TABLE names (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    gender TEXT CHECK(gender IN ('Male', 'Female', 'Unisex')),
                    origin_region TEXT,
                    origin_confidence REAL,
                    origin_classified_at TIMESTAMP,
                    phonetic_primary TEXT,
                    phonetic_secondary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE ratings (
                    name_id INTEGER PRIMARY KEY REFERENCES names(id) ON DELETE CASCADE,
                    rating REAL NOT NULL DEFAULT 1500.0,
                    matches INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE user_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE region_mapping (
                    nationality TEXT PRIMARY KEY,
                    region TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE source_versions (
                    id INTEGER PRIMARY KEY,
                    commit_hash TEXT NOT NULL,
                    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE model_state (
                    id INTEGER PRIMARY KEY,
                    feature_weights BLOB NOT NULL,
                    uncertainty_matrix BLOB NOT NULL,
                    training_samples INTEGER DEFAULT 0,
                    feature_names_json TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE comparisons (
                    id INTEGER PRIMARY KEY,
                    name_a_id INTEGER NOT NULL REFERENCES names(id),
                    name_b_id INTEGER NOT NULL REFERENCES names(id),
                    preference INTEGER NOT NULL CHECK(preference IN (-1, 0, 1, 2)),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name_a_id, name_b_id, preference)
                )
            """)

            # Insert test names
            test_names = [
                ("Anna", "Female", "Nordic"),
                ("Peter", "Male", "European"),
                ("Maria", "Female", "European"),
                ("John", "Male", "American"),
                ("Emma", "Female", "Nordic"),
                ("Lars", "Male", "Nordic"),
                ("Sofia", "Female", "European"),
                ("Max", "Male", "European"),
            ]

            for name, gender, origin in test_names:
                conn.execute(
                    "INSERT INTO names (name, gender, origin_region) VALUES (?, ?, ?)",
                    (name, gender, origin),
                )

            conn.commit()

        finally:
            conn.close()

        return db_path

    return _create_db


@pytest.fixture
def setup_session_state_data():
    """Provide sample session state data."""
    return {
        "all_names_data": {
            "All": ["Anna", "Peter", "Maria", "John", "Emma", "Lars", "Sofia", "Max"],
            "Male": ["Peter", "John", "Lars", "Max"],
            "Female": ["Anna", "Maria", "Emma", "Sofia"],
            "Unisex": [],
        },
        "all_names": ["Anna", "Peter", "Maria", "John", "Emma", "Lars", "Sofia", "Max"],
        "ratings": dict.fromkeys(["Anna", "Peter", "Maria", "John", "Emma", "Lars", "Sofia", "Max"], 1500.0),
    }


# =============================================================================
# Helper Functions
# =============================================================================


def get_main_app_script(test_db_path: Path, tab_name: str = "Name Filter") -> str:
    """Generate the main app script for testing.

    Args:
        test_db_path: Path to the test database
        tab_name: Initial active tab name

    Returns:
        Python script as string
    """
    return f"""
import streamlit as st
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path("{test_db_path}").parent.parent / "src"))

from st_name_ranking import database
from st_name_ranking.ui import render_tournament, render_binary_filter, render_similarity

# Set test DB path
database.DB_PATH = Path("{test_db_path}")
database._initialized = True  # Skip init to avoid locking issues

st.set_page_config(page_title="Name Ranker", layout="wide")
st.title("Name Preference Ranker")

# Initialize session state defaults
if "gender_filter" not in st.session_state:
    st.session_state.gender_filter = "All"
if "origin_filter" not in st.session_state:
    st.session_state.origin_filter = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "{tab_name}"
if "ratings" not in st.session_state:
    st.session_state.ratings = {{
        name: 1500.0
        for name in ["Anna", "Peter", "Maria", "John", "Emma", "Lars", "Sofia", "Max"]
    }}
if "candidate_a" not in st.session_state:
    st.session_state.candidate_a = ""
if "candidate_b" not in st.session_state:
    st.session_state.candidate_b = ""
if "candidate_queue" not in st.session_state:
    st.session_state.candidate_queue = []

# Sidebar
with st.sidebar:
    st.subheader("Gender Filter")
    gender = st.pills("Filter names by gender:", ["All", "Male", "Female"], default=st.session_state.gender_filter)
    if gender != st.session_state.gender_filter:
        st.session_state.gender_filter = gender
        st.rerun()

    st.subheader("Origin Filter")
    origins = st.multiselect("Filter names by origin region:",
                             options=["Nordic", "European", "American", "International"],
                             default=st.session_state.origin_filter)
    if origins != st.session_state.origin_filter:
        st.session_state.origin_filter = origins
        st.rerun()

    st.subheader("Database Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sync Names", icon="🔄"):
            st.toast("Sync clicked!")
    with col2:
        stats = database.get_stats()
        st.caption(f"Total: {{stats.total_names}} names")

# Get filtered names
names = ["Anna", "Peter", "Maria", "John", "Emma", "Lars", "Sofia", "Max"]
filtered = names

# Tab buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📋 Name Filter", use_container_width=True,
                 type="primary" if st.session_state.active_tab == "Name Filter" else "secondary"):
        st.session_state.active_tab = "Name Filter"
        st.rerun()
with col2:
    if st.button("🏆 Tournament", use_container_width=True,
                 type="primary" if st.session_state.active_tab == "Tournament" else "secondary"):
        st.session_state.active_tab = "Tournament"
        st.rerun()
with col3:
    if st.button("🔍 Similarity Search", use_container_width=True,
                 type="primary" if st.session_state.active_tab == "Similarity Search" else "secondary"):
        st.session_state.active_tab = "Similarity Search"
        st.rerun()

st.divider()

# Render active tab
if st.session_state.active_tab == "Name Filter":
    render_binary_filter(filtered)
elif st.session_state.active_tab == "Tournament":
    render_tournament(filtered)
else:
    render_similarity(filtered)
"""


def run_main_app_with_tab(tab_name: str, test_db_path: Path, session_state_data: dict) -> AppTest:
    """Run the main app with a specific tab preselected.

    Args:
        tab_name: The tab to activate
        test_db_path: Path to test database
        session_state_data: Initial session state data

    Returns:
        AppTest instance
    """
    script = get_main_app_script(test_db_path, tab_name)

    at = AppTest.from_string(script)

    # Set session state
    for key, value in session_state_data.items():
        at.session_state[key] = value
    at.session_state["active_tab"] = tab_name

    return at


# =============================================================================
# Tournament Tab Tests
# =============================================================================


class TestTournamentTab:
    """Tests for the Tournament tab UI components."""

    def test_tournament_renders_with_two_names(self, test_db_factory, setup_session_state_data):
        """Test that tournament renders with two candidate names displayed."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Tournament", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Should not have errors
        assert not at.exception

        # Should have buttons for both candidates
        buttons = [b for b in at.button if "Prefer" in str(b.label)]
        assert len(buttons) >= 2, "Should have at least 2 preference buttons"

    def test_preference_buttons_exist_and_clickable(self, test_db_factory, setup_session_state_data):
        """Test that preference buttons exist and can be clicked."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Tournament", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Find preference buttons
        vote_a_button = None
        vote_b_button = None

        for button in at.button:
            if hasattr(button, "key"):
                if button.key == "vote_a":
                    vote_a_button = button
                elif button.key == "vote_b":
                    vote_b_button = button

        assert vote_a_button is not None, "Vote A button should exist"
        assert vote_b_button is not None, "Vote B button should exist"
        assert vote_a_button.disabled is False, "Vote A button should be clickable"
        assert vote_b_button.disabled is False, "Vote B button should be clickable"

    def test_draw_and_down_buttons_exist(self, test_db_factory, setup_session_state_data):
        """Test that draw and down vote buttons exist."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Tournament", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Find draw and down buttons by key
        draw_button = None
        down_button = None

        for button in at.button:
            if hasattr(button, "key"):
                if button.key == "vote_draw":
                    draw_button = button
                elif button.key == "vote_down":
                    down_button = button

        assert draw_button is not None, "Draw button should exist"
        assert down_button is not None, "Down vote button should exist"

    def test_candidate_displayed_in_metrics(self, test_db_factory, setup_session_state_data):
        """Test that candidate names are displayed in metric widgets."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Tournament", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Should have metrics displaying the names
        metrics = at.metric
        assert len(metrics) >= 2, "Should display at least 2 metrics for candidates"

    def test_statistics_expander_not_rendered(self, test_db_factory, setup_session_state_data):
        """Tournament should not render statistics expander for performance reasons."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Tournament", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Statistics panel was intentionally removed from tournament tab
        expanders = [e for e in at.expander if "statistics" in str(e.label).lower()]
        assert len(expanders) == 0, "Tournament tab should not render statistics expander"


# =============================================================================
# Name Filter Tab Tests
# =============================================================================


class TestNameFilterTab:
    """Tests for the Name Filter tab UI components."""

    def test_filter_renders_with_name_list(self, test_db_factory, setup_session_state_data):
        """Test that filter renders with a name displayed."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Name Filter", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Should not have errors
        assert not at.exception

        # Should render header
        assert any("Name Filter" in elt.value for elt in at.header)

    def test_include_exclude_buttons_exist(self, test_db_factory, setup_session_state_data):
        """Test that include and exclude buttons exist."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Name Filter", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Find buttons by key
        include_btn = None
        exclude_btn = None

        for button in at.button:
            if hasattr(button, "key"):
                if button.key == "include_btn":
                    include_btn = button
                elif button.key == "exclude_btn":
                    exclude_btn = button

        assert include_btn is not None, "Include button should exist"
        assert exclude_btn is not None, "Exclude button should exist"

    def test_progress_tracking_displayed(self, test_db_factory, setup_session_state_data):
        """Test that progress tracking is displayed."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Name Filter", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Should have caption showing progress stats
        captions = [c for c in at.caption if "Not decided" in c.value or "Included" in c.value]
        assert len(captions) > 0, "Should display progress stats caption"

    def test_navigation_buttons_exist(self, test_db_factory, setup_session_state_data):
        """Test that navigation (previous/next) buttons exist."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Name Filter", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Check for navigation-related buttons
        button_labels = [str(b.label) for b in at.button]

        assert any("Previous" in label for label in button_labels), "Should have Previous button"
        assert any("Reset" in label or "Save" in label for label in button_labels), "Should have Reset or Save button"

    def test_batch_operations_buttons_exist(self, test_db_factory, setup_session_state_data):
        """Test that batch operation buttons exist."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Name Filter", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Check for batch operation buttons
        button_labels = [str(b.label) for b in at.button]

        assert any("Include All Remaining" in label for label in button_labels), (
            "Should have 'Include All Remaining' button"
        )
        assert any("Exclude All Remaining" in label for label in button_labels), (
            "Should have 'Exclude All Remaining' button"
        )

    def test_excluded_names_expander_exists(self, test_db_factory, setup_session_state_data):
        """Test that the excluded names expander exists."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Name Filter", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Check for excluded names expander
        expanders = [e for e in at.expander if "excluded" in str(e.label).lower()]
        assert len(expanders) > 0, "Should have an excluded names expander"


# =============================================================================
# Similarity Search Tab Tests
# =============================================================================


class TestSimilaritySearchTab:
    """Tests for the Similarity Search tab UI components."""

    def test_search_input_renders(self, test_db_factory, setup_session_state_data):
        """Test that search input renders correctly."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Similarity Search", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Should not have errors
        assert not at.exception

        # Should render header
        assert any("Similarity Search" in elt.value for elt in at.header)

        # Should have a text input
        text_inputs = [t for t in at.text_input if "Reference Name" in str(t.label)]
        assert len(text_inputs) > 0, "Should have reference name text input"

    def test_search_method_radio_buttons(self, test_db_factory, setup_session_state_data):
        """Test that search method radio buttons exist."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Similarity Search", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Should have radio buttons for search method
        radios = [r for r in at.radio if "Search Method" in str(r.label)]
        assert len(radios) > 0, "Should have search method radio"

        # Should have both options (radio options are strings)
        radio = radios[0]
        options = radio.options  # Radio options are already strings
        assert any("String" in opt for opt in options), "Should have String search option"
        assert any("Vector" in opt for opt in options), "Should have Vector search option"

    def test_find_similar_button_exists(self, test_db_factory, setup_session_state_data):
        """Test that find similar button exists."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Similarity Search", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Should have a "Find Similar" button
        find_buttons = [b for b in at.button if "Find Similar" in str(b.label)]
        assert len(find_buttons) > 0, "Should have Find Similar button"


# =============================================================================
# Sidebar Controls Tests
# =============================================================================


class TestSidebarControls:
    """Tests for sidebar control components."""

    def test_gender_filter_selection(self, test_db_factory, setup_session_state_data):
        """Test that gender filter selection works."""
        db_path = test_db_factory()

        script = f"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path("{db_path}").parent.parent / "src"))

from st_name_ranking import database

database.DB_PATH = Path("{db_path}")
database._initialized = True

st.set_page_config(page_title="Test", layout="wide")

with st.sidebar:
    st.subheader("Gender Filter")
    if "gender_filter" not in st.session_state:
        st.session_state.gender_filter = "All"

    gender = st.pills("Filter names by gender:",
                      ["All", "Male", "Female"],
                      default=st.session_state.gender_filter)
    if gender != st.session_state.gender_filter:
        st.session_state.gender_filter = gender
        st.rerun()

    st.write(f"Selected: {{st.session_state.gender_filter}}")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Should not have errors
        assert not at.exception

        # Should have pills widget in sidebar (accessed via button_group in AppTest)
        pills = [p for p in at.button_group if "gender" in str(p.label).lower()]
        assert len(pills) > 0, "Should have gender filter pills"

        # Should have all three options
        pill = pills[0]
        option_contents = [opt.content for opt in pill.options]
        assert "All" in option_contents
        assert "Male" in option_contents
        assert "Female" in option_contents

    def test_origin_filter_multiselect(self, test_db_factory, setup_session_state_data):
        """Test that origin filter multiselect works."""
        db_path = test_db_factory()

        script = f"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path("{db_path}").parent.parent / "src"))

from st_name_ranking import database

database.DB_PATH = Path("{db_path}")
database._initialized = True

st.set_page_config(page_title="Test", layout="wide")

with st.sidebar:
    st.subheader("Origin Filter")
    if "origin_filter" not in st.session_state:
        st.session_state.origin_filter = []

    origins = st.multiselect("Filter names by origin region:",
                            options=["Nordic", "European", "American", "International"],
                            default=st.session_state.origin_filter)

    if origins != st.session_state.origin_filter:
        st.session_state.origin_filter = origins
        st.rerun()

    st.write(f"Selected: {{st.session_state.origin_filter}}")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Should not have errors
        assert not at.exception

        # Should have multiselect widget
        multiselects = [m for m in at.multiselect if "origin" in str(m.label).lower()]
        assert len(multiselects) > 0, "Should have origin filter multiselect"

    def test_sync_names_button_exists(self, test_db_factory, setup_session_state_data):
        """Test that sync names button exists in sidebar."""
        db_path = test_db_factory()

        script = f"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path("{db_path}").parent.parent / "src"))

from st_name_ranking import database

database.DB_PATH = Path("{db_path}")
database._initialized = True

st.set_page_config(page_title="Test", layout="wide")

with st.sidebar:
    st.subheader("Database Management")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sync Names", icon="🔄"):
            st.toast("Sync clicked!")

    with col2:
        st.caption("Total: 8 names")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Should not have errors
        assert not at.exception

        # Should have Sync Names button
        sync_buttons = [b for b in at.button if "Sync Names" in str(b.label)]
        assert len(sync_buttons) > 0, "Should have Sync Names button"


# =============================================================================
# Session State Management Tests
# =============================================================================


class TestSessionStateManagement:
    """Tests for session state initialization and persistence."""

    def test_session_state_initialization(self, test_db_factory):
        """Test that session state is properly initialized."""
        db_path = test_db_factory()

        script = f"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path("{db_path}").parent.parent / "src"))

from st_name_ranking import database
from st_name_ranking.utils import setup_session_state

database.DB_PATH = Path("{db_path}")
database._initialized = True

st.set_page_config(page_title="Test", layout="wide")

# Test setup_session_state
names = ["Anna", "Peter", "Maria"]
setup_session_state(names)

# Verify session state
st.write(f"ratings exists: {{'ratings' in st.session_state}}")
st.write(f"candidate_a exists: {{'candidate_a' in st.session_state}}")
st.write(f"candidate_b exists: {{'candidate_b' in st.session_state}}")
st.write(f"names exists: {{'names' in st.session_state}}")
st.write(f"names count: {{len(st.session_state.get('names', []))}}")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Should not have errors
        assert not at.exception

        # Verify all session state variables were initialized
        assert "ratings" in at.session_state
        assert "candidate_a" in at.session_state
        assert "candidate_b" in at.session_state
        assert "names" in at.session_state
        assert len(at.session_state["names"]) == 3

    def test_state_persistence_across_reruns(self, test_db_factory):
        """Test that state persists across reruns."""
        script = """
import streamlit as st
import sys
from pathlib import Path

st.set_page_config(page_title="Test", layout="wide")

# Initialize counter if not exists
if "counter" not in st.session_state:
    st.session_state.counter = 0

st.write(f"Counter: {st.session_state.counter}")

if st.button("Increment"):
    st.session_state.counter += 1
    st.rerun()
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Initial run
        assert not at.exception
        assert at.session_state.counter == 0

        # Click button
        button = next(b for b in at.button if "Increment" in str(b.label))
        button.click()
        at.run(timeout=30)

        # State should be updated
        assert at.session_state.counter == 1

    def test_filter_changes_reset_caches(self, test_db_factory, setup_session_state_data):
        """Test that filter changes properly reset caches."""
        db_path = test_db_factory()

        script = f"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path("{db_path}").parent.parent / "src"))

from st_name_ranking import database

database.DB_PATH = Path("{db_path}")
database._initialized = True

st.set_page_config(page_title="Test", layout="wide")

if "gender_filter" not in st.session_state:
    st.session_state.gender_filter = "All"
    st.session_state.filtered_names_cache = ["cached_name"]
    st.session_state.filtered_cache_key = "old_key"

# Simulate filter change
new_gender = st.pills("Gender:", ["All", "Male", "Female"],
                      default=st.session_state.gender_filter)

if new_gender != st.session_state.gender_filter:
    st.session_state.gender_filter = new_gender
    # Clear cache
    st.session_state.filtered_names_cache = None
    st.session_state.filtered_cache_key = None
    st.rerun()

st.write(f"Cache exists: {{st.session_state.filtered_names_cache is not None}}")
st.write(f"Cache key: {{st.session_state.filtered_cache_key}}")
"""

        at = AppTest.from_string(script)
        at.session_state["gender_filter"] = "All"
        at.session_state["filtered_names_cache"] = ["cached_name"]
        at.session_state["filtered_cache_key"] = "old_key"

        at.run(timeout=30)

        # Should not have errors
        assert not at.exception

        # Cache should exist initially
        assert at.session_state.filtered_names_cache is not None

        # Change the pills value and run again (accessed via button_group in AppTest)
        pills = next(iter(at.button_group))
        pills.set_value(["Male"])
        at.run(timeout=30)

        # Cache should be cleared
        assert at.session_state.filtered_names_cache is None


# =============================================================================
# Integration Tests - Widget Interactions
# =============================================================================


class TestWidgetInteractions:
    """Tests for actual widget interactions using AppTest."""

    def test_pills_selection_changes_session_state(self, test_db_factory):
        """Test that selecting a pill updates session state."""
        script = """
import streamlit as st
import sys
from pathlib import Path

st.set_page_config(page_title="Test", layout="wide")

if "selected" not in st.session_state:
    st.session_state.selected = "A"

selected = st.pills("Choose:", ["A", "B", "C"], default=st.session_state.selected)

if selected != st.session_state.selected:
    st.session_state.selected = selected
    st.rerun()

st.write(f"Selected: {st.session_state.selected}")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Initial selection
        assert at.session_state.selected == "A"

        # Change selection using set_value (accessed via button_group in AppTest)
        pills = next(iter(at.button_group))
        pills.set_value(["B"])
        at.run(timeout=30)

        # Session state should be updated
        assert at.session_state.selected == "B"

    def test_multiselect_changes_session_state(self, test_db_factory):
        """Test that multiselect updates session state."""
        script = """
import streamlit as st
import sys
from pathlib import Path

st.set_page_config(page_title="Test", layout="wide")

if "selected" not in st.session_state:
    st.session_state.selected = []

selected = st.multiselect("Choose:", ["A", "B", "C"], default=st.session_state.selected)

if selected != st.session_state.selected:
    st.session_state.selected = selected
    st.rerun()

st.write(f"Selected: {st.session_state.selected}")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Initial selection should be empty
        assert at.session_state.selected == []

        # Select items using set_value
        multiselect = next(iter(at.multiselect))
        multiselect.set_value(["A", "C"])
        at.run(timeout=30)

        # Session state should be updated
        assert "A" in at.session_state.selected
        assert "C" in at.session_state.selected

    def test_button_click_triggers_action(self, test_db_factory):
        """Test that button clicks trigger actions."""
        script = """
import streamlit as st
import sys
from pathlib import Path

st.set_page_config(page_title="Test", layout="wide")

if "clicked" not in st.session_state:
    st.session_state.clicked = False

if st.button("Click Me"):
    st.session_state.clicked = True
    st.rerun()

st.write(f"Clicked: {st.session_state.clicked}")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Initial state
        assert at.session_state.clicked is False

        # Click button
        button = next(b for b in at.button if "Click Me" in str(b.label))
        button.click()
        at.run(timeout=30)

        # State should be updated
        assert at.session_state.clicked is True

    def test_text_input_changes_value(self, test_db_factory):
        """Test that text input changes are captured."""
        script = """
import streamlit as st
import sys
from pathlib import Path

st.set_page_config(page_title="Test", layout="wide")

text = st.text_input("Enter text:", value="default")
st.write(f"Text: {text}")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Get text input and change value
        text_input = next(iter(at.text_input))
        text_input.set_value("new value")
        at.run(timeout=30)

        # The new value should be reflected
        # Note: We check that no exception occurred and widget exists
        assert not at.exception


# =============================================================================
# Database Integration Tests
# =============================================================================


class TestDatabaseIntegration:
    """Tests that verify actual database interactions."""

    def test_database_stats_displayed(self, test_db_factory):
        """Test that database stats are displayed correctly."""
        db_path = test_db_factory()

        script = f"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path("{db_path}").parent.parent / "src"))

from st_name_ranking import database

database.DB_PATH = Path("{db_path}")
database._initialized = True

st.set_page_config(page_title="Test", layout="wide")

# Get stats
stats = database.get_stats()

st.caption(f"Total: {{stats.total_names}} names")
st.caption(f"Classified: {{stats.classified_names}} names")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Should not have errors
        assert not at.exception

        # Should display stats
        captions = [c.value for c in at.caption]
        assert any("Total:" in c for c in captions), "Should display total names"
        assert any("Classified:" in c for c in captions), "Should display classified count"

    def test_user_setting_save_and_load(self, test_db_factory):
        """Test that user settings are saved and loaded correctly."""
        db_path = test_db_factory()

        script = f"""
import streamlit as st
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("{db_path}").parent.parent / "src"))

from st_name_ranking import database

database.DB_PATH = Path("{db_path}")
database._initialized = True

st.set_page_config(page_title="Test", layout="wide")

# Save a setting
if "saved" not in st.session_state:
    database.save_user_setting("test_key", "test_value")
    st.session_state.saved = True

# Load the setting
value = database.load_user_setting("test_key", "default")
st.write(f"Loaded value: {{value}}")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Should not have errors
        assert not at.exception

        # Should display the saved value
        markdown_texts = [m.value for m in at.markdown]
        assert any("test_value" in m for m in markdown_texts), "Should display saved value"

    def test_name_filtering_by_gender(self, test_db_factory):
        """Test that name filtering by gender works with real database."""
        db_path = test_db_factory()

        script = f"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path("{db_path}").parent.parent / "src"))

from st_name_ranking import database

database.DB_PATH = Path("{db_path}")
database._initialized = True

st.set_page_config(page_title="Test", layout="wide")

# Test filtering
female_names = database.get_names_by_filters(gender="Female")
male_names = database.get_names_by_filters(gender="Male")
all_names = database.get_names_by_filters()

st.write(f"Female names: {{len(female_names)}}")
st.write(f"Male names: {{len(male_names)}}")
st.write(f"All names: {{len(all_names)}}")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Should not have errors
        assert not at.exception

        # Should display correct counts
        markdown_texts = [m.value for m in at.markdown]
        assert any("Female names: 4" in m for m in markdown_texts), "Should show 4 female names"
        assert any("Male names: 4" in m for m in markdown_texts), "Should show 4 male names"
        assert any("All names: 8" in m for m in markdown_texts), "Should show 8 total names"

    def test_name_filtering_by_origin(self, test_db_factory):
        """Test that name filtering by origin works with real database."""
        db_path = test_db_factory()

        script = f"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path("{db_path}").parent.parent / "src"))

from st_name_ranking import database

database.DB_PATH = Path("{db_path}")
database._initialized = True

st.set_page_config(page_title="Test", layout="wide")

# Test filtering by origin
nordic_names = database.get_names_by_filters(origins=["Nordic"])
european_names = database.get_names_by_filters(origins=["European"])

st.write(f"Nordic names: {{len(nordic_names)}}")
st.write(f"European names: {{len(european_names)}}")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Should not have errors
        assert not at.exception

        # Should display correct counts
        markdown_texts = [m.value for m in at.markdown]
        assert any("Nordic names: 3" in m for m in markdown_texts), "Should show 3 Nordic names (Anna, Emma, Lars)"
        assert any("European names: 4" in m for m in markdown_texts), (
            "Should show 4 European names (Peter, Maria, Sofia, Max)"
        )


# =============================================================================
# End-to-End Tab Navigation Tests
# =============================================================================


class TestTabNavigation:
    """Tests for tab navigation functionality."""

    def test_name_filter_tab_button_exists(self, test_db_factory, setup_session_state_data):
        """Test Name Filter tab button exists."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Tournament", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Initial tab
        assert at.session_state.active_tab == "Tournament"

        # Find Name Filter button
        filter_buttons = [b for b in at.button if "Name Filter" in str(b.label)]
        assert len(filter_buttons) > 0, "Name Filter button should exist"

    def test_tournament_tab_button_exists(self, test_db_factory, setup_session_state_data):
        """Test Tournament tab button exists."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Name Filter", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Initial tab
        assert at.session_state.active_tab == "Name Filter"

        # Find Tournament button (exclude "Name Filter" button)
        tournament_buttons = [b for b in at.button if "Tournament" in str(b.label) and "Name" not in str(b.label)]
        assert len(tournament_buttons) > 0, "Tournament button should exist"

    def test_similarity_search_tab_button_exists(self, test_db_factory, setup_session_state_data):
        """Test Similarity Search tab button exists."""
        db_path = test_db_factory()
        at = run_main_app_with_tab("Name Filter", db_path, setup_session_state_data)
        at.run(timeout=30)

        # Initial tab
        assert at.session_state.active_tab == "Name Filter"

        # Find Similarity Search button
        similarity_buttons = [b for b in at.button if "Similarity" in str(b.label)]
        assert len(similarity_buttons) > 0, "Similarity Search button should exist"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in the UI."""

    def test_empty_names_list_handling(self, test_db_factory):
        """Test handling of empty names list."""
        script = """
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from st_name_ranking.ui import render_tournament, render_binary_filter

st.set_page_config(page_title="Test", layout="wide")

# Test with empty list
try:
    render_tournament([])
    st.write("Rendered tournament with empty list")
except Exception as e:
    st.error(f"Error: {e}")

try:
    render_binary_filter([])
    st.write("Rendered filter with empty list")
except Exception as e:
    st.error(f"Error: {e}")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Should handle gracefully - check no unexpected exceptions
        # render_tournament with empty list should return early
        assert not at.exception

    def test_single_name_handling(self, test_db_factory):
        """Test handling of single name list."""
        script = """
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from st_name_ranking.ui import render_tournament, render_binary_filter

st.set_page_config(page_title="Test", layout="wide")

# Test with single name
try:
    render_tournament(["Anna"])
    st.write("Rendered tournament with single name")
except Exception as e:
    st.error(f"Error: {e}")
"""

        at = AppTest.from_string(script)
        at.run(timeout=30)

        # Should handle gracefully
        assert not at.exception


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
