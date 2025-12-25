"""
Playwright UI integration tests for the Name Ranking application.
"""
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import requests
from playwright.sync_api import Page, expect


def is_streamlit_running(port: int = 8501) -> bool:
    """Check if Streamlit is running on the specified port."""
    try:
        response = requests.get(f"http://localhost:{port}/healthz", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


class StreamlitServer:
    """Manage a Streamlit server process for testing."""
    
    def __init__(self, app_file: Path, port: int = 8501, host: str = "localhost"):
        self.app_file = app_file
        self.port = port
        self.host = host
        self.process = None
        self.url = f"http://{host}:{port}"
    
    def start(self, timeout: int = 30):
        """Start the Streamlit server."""
        if is_streamlit_running(self.port):
            raise RuntimeError(f"Port {self.port} is already in use")
        
        # Create a temporary directory for Streamlit config
        temp_dir = tempfile.mkdtemp()
        env = os.environ.copy()
        env["STREAMLIT_SERVER_PORT"] = str(self.port)
        env["STREAMLIT_SERVER_ADDRESS"] = self.host
        env["STREAMLIT_SERVER_HEADLESS"] = "true"
        env["STREAMLIT_BROWSER_SERVER_ADDRESS"] = self.host
        
        # Start Streamlit process
        cmd = [
            "streamlit", "run", str(self.app_file),
            "--server.port", str(self.port),
            "--server.address", self.host,
            "--server.headless", "true",
            "--browser.serverAddress", self.host,
            "--logger.level", "error",
            "--theme.base", "light",
            "--client.toolbarMode", "minimal",
        ]
        
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Wait for server to start
        start_time = time.time()
        while time.time() - start_time < timeout:
            if is_streamlit_running(self.port):
                return
            time.sleep(0.5)
        
        # If we get here, server didn't start
        self.stop()
        raise TimeoutError(f"Streamlit server failed to start on port {self.port}")
    
    def stop(self):
        """Stop the Streamlit server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


@pytest.fixture(scope="session")
def streamlit_server():
    """Fixture to start and stop a Streamlit server for testing."""
    app_file = Path(__file__).parent.parent / "st_name_ranking" / "main.py"
    with StreamlitServer(app_file, port=8502) as server:
        yield server


@pytest.fixture
def streamlit_page(streamlit_server, page: Page):
    """Fixture to navigate to the Streamlit app."""
    page.goto(streamlit_server.url)
    yield page


def test_app_loads(streamlit_page: Page):
    """Test that the app loads and shows the main title."""
    # Check the main title
    expect(streamlit_page).to_have_title("Name Ranker")
    
    # Check for the main heading
    heading = streamlit_page.locator("h1")
    expect(heading).to_have_text("Name Preference Ranker")
    
    # Check that tabs are present
    tabs = streamlit_page.locator('[data-testid="stTab"]')
    expect(tabs).to_have_count(2)
    
    # Check tab names
    tab_labels = streamlit_page.locator('[data-testid="stTabButton"]')
    expect(tab_labels).to_have_count(2)
    expect(tab_labels.nth(0)).to_have_text("Tournament")
    expect(tab_labels.nth(1)).to_have_text("Similarity Search")


def test_tournament_tab_basic_interaction(streamlit_page: Page):
    """Test basic interaction with the Tournament tab."""
    # Click on Tournament tab (should be active by default)
    tournament_tab = streamlit_page.locator('[data-testid="stTabButton"]').nth(0)
    tournament_tab.click()
    
    # Check for name comparison elements
    # Should have two name cards
    name_cards = streamlit_page.locator('.stButton')  # This is approximate
    expect(name_cards).to_have_count_at_least(2)
    
    # Check for voting buttons
    # Looking for left/right vote buttons
    vote_buttons = streamlit_page.get_by_role("button").filter(has_text="Vote")
    expect(vote_buttons).to_have_count_at_least(1)
    
    # Check for draw button
    draw_button = streamlit_page.get_by_role("button").filter(has_text="Draw")
    expect(draw_button).to_be_visible()
    
    # Check for top rankings section
    rankings_header = streamlit_page.get_by_text("Top Rankings")
    expect(rankings_header).to_be_visible()


def test_similarity_tab_basic_interaction(streamlit_page: Page):
    """Test basic interaction with the Similarity Search tab."""
    # Click on Similarity Search tab
    similarity_tab = streamlit_page.locator('[data-testid="stTabButton"]').nth(1)
    similarity_tab.click()
    
    # Check for search input
    search_input = streamlit_page.get_by_label("Reference name")
    expect(search_input).to_be_visible()
    
    # Check for search button
    search_button = streamlit_page.get_by_role("button").filter(has_text="Search")
    expect(search_button).to_be_visible()
    
    # Check for similarity type selection
    similarity_type = streamlit_page.get_by_text("Similarity type")
    expect(similarity_type).to_be_visible()


def test_sidebar_controls(streamlit_page: Page):
    """Test that sidebar controls are present."""
    # Check for gender filter
    gender_filter = streamlit_page.get_by_text("Gender filter")
    expect(gender_filter).to_be_visible()
    
    # Check for origin filter
    origin_filter = streamlit_page.get_by_text("Origin region filter")
    expect(origin_filter).to_be_visible()
    
    # Check for submodule management buttons
    sync_buttons = streamlit_page.get_by_role("button").filter(has_text="Sync")
    expect(sync_buttons).to_have_count_at_least(1)
    
    # Check for ratings management
    ratings_section = streamlit_page.get_by_text("Ratings Management")
    expect(ratings_section).to_be_visible()


def test_vote_interaction(streamlit_page: Page):
    """Test voting interaction (simulate clicking vote buttons)."""
    # Make sure we're on Tournament tab
    tournament_tab = streamlit_page.locator('[data-testid="stTabButton"]').nth(0)
    tournament_tab.click()
    
    # Get the current names being displayed
    # This is tricky without specific selectors - we'll look for h2 or large text
    name_elements = streamlit_page.locator("h2, h3").filter(has_text=/.+/)
    
    # At least two names should be displayed
    expect(name_elements).to_have_count_at_least(2)
    
    # Try to click vote left button (if we can identify it)
    # We'll look for buttons with "←" or "Vote Left"
    vote_left_buttons = streamlit_page.get_by_role("button").filter(
        has_text=re.compile(r"(←|Vote.*Left|Left.*Vote)", re.IGNORECASE)
    )
    
    if vote_left_buttons.count() > 0:
        # Click the first vote left button
        vote_left_buttons.first.click()
        # Check for some feedback (toast message or rating update)
        # This is basic - just verifying the click doesn't crash
        time.sleep(0.5)
    
    # Try to click vote right button
    vote_right_buttons = streamlit_page.get_by_role("button").filter(
        has_text=re.compile(r"(→|Vote.*Right|Right.*Vote)", re.IGNORECASE)
    )
    
    if vote_right_buttons.count() > 0:
        vote_right_buttons.first.click()
        time.sleep(0.5)
    
    # Try to click draw button
    draw_button = streamlit_page.get_by_role("button").filter(has_text="Draw")
    if draw_button.count() > 0:
        draw_button.first.click()
        time.sleep(0.5)


def test_search_interaction(streamlit_page: Page):
    """Test similarity search interaction."""
    # Go to Similarity Search tab
    similarity_tab = streamlit_page.locator('[data-testid="stTabButton"]').nth(1)
    similarity_tab.click()
    
    # Enter a search query
    search_input = streamlit_page.get_by_label("Reference name")
    search_input.fill("Anna")
    
    # Click search button
    search_button = streamlit_page.get_by_role("button").filter(has_text="Search")
    search_button.click()
    
    # Check for results (might take a moment)
    time.sleep(1)
    
    # Look for results table or list
    results = streamlit_page.locator('[data-testid="stTable"], table')
    if results.count() > 0:
        expect(results).to_be_visible()


if __name__ == "__main__":
    # For manual testing
    pytest.main([__file__, "-v"])
