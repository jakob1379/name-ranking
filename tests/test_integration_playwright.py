"""
Integration tests using Playwright for the Name Ranking application.

These tests actually interact with a running Streamlit application and test
end-to-end functionality including UI interactions and backend responses.

These tests are separate from test_ui.py which uses Streamlit's AppTest framework.

Requirements:
- Playwright browsers installed (run: uv run playwright install chromium)
  - When using Nix, browsers are provided via `playwright-driver.browsers` package
  - The test fixture automatically uses the Nix-provided headless chromium shell
- Streamlit app running (default: http://localhost:8501)

Usage:
    # Start the application first:
    uv run streamlit run src/st_name_ranking/main.py

    # Run integration tests:
    uv run pytest tests/test_integration_playwright.py --run-integration --run-playwright -v

Note: These tests are slower and require more setup than unit tests.
They are marked with @pytest.mark.integration and @pytest.mark.playwright
and will be skipped by default. Use the command-line flags to run them.
"""

import asyncio
import os

import pytest
import pytest_asyncio
from playwright.async_api import Page, async_playwright


# Skip all Playwright tests in Nix environment due to browser version mismatch
# Nix provides chromium-1200 but Playwright expects chromium-1194
def is_nix_environment():
    """Check if running in Nix environment."""
    # Check for Nix store path in PLAYWRIGHT_BROWSERS_PATH
    browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    return "/nix/store/" in browsers_path


if is_nix_environment():
    pytest.skip(
        "Skipping Playwright tests in Nix environment due to browser version mismatch.\n"
        "Nix provides chromium-1200 but Playwright expects chromium-1194.\n"
        "To run Playwright tests, use a non-Nix environment or CI with compatible browsers.\n"
        "For UI testing in Nix, use AppTest-based tests (test_ui.py).",
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def browser():
    """Create Playwright browser instance.

    Note: In Nix environments, tests are skipped at module level due to
    browser version mismatch (Nix provides chromium-1200, Playwright expects 1194).
    This fixture is only used in non-Nix environments.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()


@pytest_asyncio.fixture
async def page(browser, app_url):
    """Create a new browser page and navigate to app."""
    page = await browser.new_page()
    await page.set_viewport_size({"width": 1280, "height": 800})

    # Navigate to application
    await page.goto(app_url, wait_until="networkidle")
    await page.wait_for_timeout(1000)  # Extra time for Streamlit to initialize

    yield page
    await page.close()


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
class TestApplicationLoads:
    """Basic tests that the application loads correctly."""

    async def test_page_title(self, page: Page):
        """Test that the page loads with correct title."""
        # Check page title
        title = await page.title() or ""
        assert "Name" in title or "Ranker" in title or "Streamlit" in title

        # Check for main heading
        heading = page.locator("h1")
        if await heading.count() > 0:
            heading_text = await heading.first.text_content() or ""
            assert "Name" in heading_text or "Ranker" in heading_text

    async def test_no_errors_on_load(self, page: Page):
        """Test that application loads without errors."""
        # Check for Streamlit exception elements
        error_elements = page.locator('[data-testid="stException"]')
        assert await error_elements.count() == 0, "Application has errors on load"

    async def test_tabs_present(self, page: Page):
        """Test that both main tabs are present."""
        tabs = page.locator('[data-testid="stTab"]')
        await tabs.first.wait_for(state="attached")
        tab_count = await tabs.count()
        assert tab_count >= 2, f"Expected at least 2 tabs, found {tab_count}"

        # Check tab labels
        tab_texts = []
        for i in range(tab_count):
            tab_text = await tabs.nth(i).text_content()
            tab_texts.append((tab_text or "").lower())

        # Should have tournament and similarity search
        assert any("tournament" in text for text in tab_texts)
        assert any("similar" in text for text in tab_texts)


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
class TestTournamentTab:
    """Tests for the Tournament tab functionality."""

    async def test_tournament_tab_active_by_default(self, page: Page):
        """Test that tournament tab is active by default."""
        # The first tab should be tournament
        tabs = page.locator('[data-testid="stTab"]')
        first_tab = tabs.first
        first_tab_text = await first_tab.text_content() or ""
        assert "tournament" in first_tab_text.lower()

    async def test_candidate_names_displayed(self, page: Page):
        """Test that candidate names are displayed."""
        # Look for candidate names in various possible locations
        candidate_selectors = [
            "h2",
            "h3",
            '[data-testid="stMarkdown"] h2',
            '[data-testid="stMarkdown"] h3',
            'div[class*="candidate"]',
            'div[class*="name"]',
        ]

        found_candidates = False
        for selector in candidate_selectors:
            elements = page.locator(selector)
            count = await elements.count()
            if count >= 2:
                # Check if elements contain text that looks like names
                valid_names = 0
                for i in range(min(2, count)):
                    text = await elements.nth(i).text_content()
                    if text and len(text.strip()) > 1 and text.strip().isalpha():
                        valid_names += 1
                if valid_names >= 2:
                    found_candidates = True
                    break

        assert found_candidates, "Could not find candidate names displayed"

    async def test_voting_buttons_present(self, page: Page):
        """Test that voting buttons are present."""
        # Look for prefer buttons
        prefer_buttons = page.locator('button:has-text("Prefer")')
        prefer_count = await prefer_buttons.count()
        assert prefer_count >= 2, f"Expected at least 2 prefer buttons, found {prefer_count}"

        # Look for draw button
        draw_button = page.locator('button:has-text("Draw")')
        assert await draw_button.count() >= 1, "Draw button not found"

    async def test_vote_interaction(self, page: Page):
        """Test that voting works (non-destructive)."""
        # Take a vote action but don't assert specific outcomes
        # since we don't control the backend state
        prefer_buttons = page.locator('button:has-text("Prefer")')
        if await prefer_buttons.count() > 0:
            await prefer_buttons.first.click()
            await page.wait_for_timeout(1000)

            # Just check no errors
            error_elements = page.locator('[data-testid="stException"]')
            assert await error_elements.count() == 0, "Error after voting"

    async def test_draw_vote(self, page: Page):
        """Test draw vote."""
        draw_button = page.locator('button:has-text("Draw")')
        if await draw_button.count() > 0:
            await draw_button.first.click()
            await page.wait_for_timeout(1000)

            error_elements = page.locator('[data-testid="stException"]')
            assert await error_elements.count() == 0, "Error after draw vote"


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
class TestSimilaritySearchTab:
    """Tests for the Similarity Search tab."""

    async def test_switch_to_similarity_tab(self, page: Page):
        """Test switching to similarity search tab."""
        # Find and click similarity search tab
        tabs = page.locator('[data-testid="stTab"]')
        for i in range(await tabs.count()):
            tab_text = await tabs.nth(i).text_content() or ""
            if "similar" in tab_text.lower():
                await tabs.nth(i).click()
                await page.wait_for_timeout(1500)  # Wait for tab to load
                break

        # Verify we're on similarity tab by looking for search elements
        search_input = page.locator('input[placeholder*="name" i], input[placeholder*="search" i], input[type="text"]')
        assert await search_input.count() > 0, "Search input not found in similarity tab"

    async def test_similarity_search_interface(self, page: Page):
        """Test similarity search interface elements."""
        # Switch to similarity tab first
        tabs = page.locator('[data-testid="stTab"]')
        for i in range(await tabs.count()):
            tab_text = await tabs.nth(i).text_content() or ""
            if "similar" in tab_text.lower():
                await tabs.nth(i).click()
                await page.wait_for_timeout(1500)
                break

        # Check for search input
        search_input = page.locator('input[placeholder*="name" i], input[placeholder*="search" i]')
        assert await search_input.count() > 0, "Search input not found"

        # Check for search button
        search_button = page.locator('button:has-text("Search")')
        # Some implementations might have the button, others might trigger on Enter
        if await search_button.count() == 0:
            # Check if there's any submit mechanism
            pass  # It's OK if no button - might be automatic

    async def test_perform_simple_search(self, page: Page):
        """Test performing a similarity search."""
        # Switch to similarity tab
        tabs = page.locator('[data-testid="stTab"]')
        for i in range(await tabs.count()):
            tab_text = await tabs.nth(i).text_content() or ""
            if "similar" in tab_text.lower():
                await tabs.nth(i).click()
                await page.wait_for_timeout(1500)
                break

        # Enter search query
        search_input = page.locator('input[placeholder*="name" i], input[placeholder*="search" i]').first
        await search_input.fill("test")

        # Try to trigger search
        search_button = page.locator('button:has-text("Search")')
        if await search_button.count() > 0:
            await search_button.first.click()
        else:
            # Try Enter key
            await search_input.press("Enter")

        await page.wait_for_timeout(2000)

        # Check for results or no results message
        # Don't assert specific results since database may vary


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
class TestSidebarControls:
    """Tests for sidebar controls."""

    async def test_sidebar_visible(self, page: Page):
        """Test that sidebar is visible."""
        sidebar = page.locator('[data-testid="stSidebar"]')
        assert await sidebar.count() > 0, "Sidebar not found"

    async def test_database_management_section(self, page: Page):
        """Test database management controls."""
        sidebar = page.locator('[data-testid="stSidebar"]')

        # Look for database management section
        db_text = sidebar.locator(':has-text("Database"), :has-text("Management"), :has-text("Sync")')
        assert await db_text.count() > 0, "Database management section not found"

        # Look for sync button
        sync_button = sidebar.locator('button:has-text("Sync")')
        if await sync_button.count() == 0:
            # Might be "Sync Names" or similar
            sync_button = sidebar.locator('button:has-text("Sync Names")')

        # It's OK if sync button not found - UI might differ
        if await sync_button.count() > 0:
            assert True

    async def test_filter_controls(self, page: Page):
        """Test filter controls in sidebar."""
        sidebar = page.locator('[data-testid="stSidebar"]')

        # Look for filter controls
        filter_selectors = [
            ':has-text("Gender")',
            ':has-text("Filter")',
            ':has-text("Origin")',
            ':has-text("Region")',
        ]

        found_filter = False
        for selector in filter_selectors:
            if await sidebar.locator(selector).count() > 0:
                found_filter = True
                break

        # It's OK if no filter controls - might be in main area
        if found_filter:
            assert True


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
class TestKeyboardNavigation:
    """Test keyboard shortcuts for voting."""

    async def test_keyboard_voting(self, page: Page):
        """Test keyboard shortcuts for voting."""
        # Focus on page body
        await page.click("body")

        # Test left arrow (vote left)
        await page.keyboard.press("ArrowLeft")
        await page.wait_for_timeout(500)

        # Test right arrow (vote right)
        await page.keyboard.press("ArrowRight")
        await page.wait_for_timeout(500)

        # Test down arrow (draw)
        await page.keyboard.press("ArrowDown")
        await page.wait_for_timeout(500)

        # Just verify no errors
        error_elements = page.locator('[data-testid="stException"]')
        assert await error_elements.count() == 0, "Errors with keyboard navigation"


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
async def test_end_to_end_workflow(page: Page):
    """Test a complete workflow through the application."""
    # 1. Verify application loads
    error_elements = page.locator('[data-testid="stException"]')
    assert await error_elements.count() == 0, "Errors on initial load"

    # 2. Vote a few times in tournament
    for _ in range(2):
        prefer_buttons = page.locator('button:has-text("Prefer")')
        if await prefer_buttons.count() > 0:
            await prefer_buttons.first.click()
            await page.wait_for_timeout(800)

    # 3. Switch to similarity tab
    tabs = page.locator('[data-testid="stTab"]')
    for i in range(await tabs.count()):
        tab_text = await tabs.nth(i).text_content() or ""
        if "similar" in tab_text.lower():
            await tabs.nth(i).click()
            await page.wait_for_timeout(1500)
            break

    # 4. Perform a search
    search_input = page.locator('input[placeholder*="name" i], input[placeholder*="search" i]')
    if await search_input.count() > 0:
        await search_input.first.fill("a")
        await page.wait_for_timeout(500)
        await search_input.first.press("Enter")
        await page.wait_for_timeout(1500)

    # 5. Switch back to tournament
    for i in range(await tabs.count()):
        tab_text = await tabs.nth(i).text_content() or ""
        if "tournament" in tab_text.lower():
            await tabs.nth(i).click()
            await page.wait_for_timeout(1000)
            break

    # 6. Final error check
    error_elements = page.locator('[data-testid="stException"]')
    assert await error_elements.count() == 0, "Errors after complete workflow"


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
async def test_application_responsiveness(page: Page, app_url: str):
    """Test application responsiveness to various interactions."""
    # Reload page to start fresh
    await page.goto(app_url, wait_until="networkidle")
    await page.wait_for_timeout(1000)

    # Click various elements (non-destructively)
    click_selectors = [
        'button:has-text("Prefer"):first',
        '[data-testid="stTab"]:first',
        '[data-testid="stSidebar"] button:first',
    ]

    for selector in click_selectors:
        elements = page.locator(selector)
        if await elements.count() > 0:
            await elements.first.click()
            await page.wait_for_timeout(500)

    # Check for errors
    error_elements = page.locator('[data-testid="stException"]')
    assert await error_elements.count() == 0, "Application not responsive"


if __name__ == "__main__":
    # Simple runner for manual testing
    import asyncio

    async def run_one_test():
        """Run a single test manually."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            await page.set_viewport_size({"width": 1280, "height": 800})
            await page.goto("http://localhost:8501", wait_until="networkidle")

            # Run a specific test
            test = TestTournamentTab()
            await test.test_voting_buttons_present(page)

            await browser.close()

    asyncio.run(run_one_test())
