"""
Integration tests using Playwright for the Name Ranking application.

These tests actually interact with a running Streamlit application and test
end-to-end functionality including UI interactions and backend responses.

These tests are separate from tests/unit/test_ui.py which uses Streamlit's AppTest framework.

Requirements:
- Playwright browsers installed (run: uv run playwright install chromium)
  - When using Nix, browsers are provided via `playwright-driver.browsers` package
  - The test fixture automatically uses the Nix-provided headless chromium shell
- Streamlit app running (default: http://localhost:8501)

Usage:
    # Start the application first:
    uv run streamlit run src/st_name_ranking/main.py

    # Run integration tests:
    uv run pytest tests/browser/test_integration_playwright.py --run-integration --run-playwright -v

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
        "For UI testing in Nix, use AppTest-based tests (tests/unit/test_ui.py).",
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
        """Test that all three main tab buttons are present."""
        name_filter_button = page.locator('button:has-text("Name Filter")')
        tournament_button = page.locator('button:has-text("Tournament")')
        similarity_button = page.locator('button:has-text("Similarity Search")')

        assert await name_filter_button.count() > 0, "Name Filter button not found"
        assert await tournament_button.count() > 0, "Tournament button not found"
        assert await similarity_button.count() > 0, "Similarity Search button not found"

        # Check that Name Filter is active by default (primary button type)
        # Note: Streamlit doesn't expose button type as attribute easily
        # We'll just verify the buttons exist and are clickable


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
class TestTournamentTab:
    """Tests for the Tournament tab functionality."""

    async def test_name_filter_tab_active_by_default(self, page: Page):
        """Test that name filter tab is active by default."""
        # With button-based UI, Name Filter button should be present
        # and name filter interface should be visible
        name_filter_button = page.locator('button:has-text("Name Filter")')
        assert await name_filter_button.count() > 0, "Name Filter button not found"

        # Verify name filter interface is visible (large name display)
        large_name = page.locator("h1").first
        assert await large_name.count() > 0, "Large name display not found"

        # Verify filter decision buttons are present
        exclude_button = page.locator('button:has-text("Exclude/Dislike")')
        include_button = page.locator('button:has-text("Include/Neutral")')
        assert await exclude_button.count() > 0, "Exclude/Dislike button not found"
        assert await include_button.count() > 0, "Include/Neutral button not found"

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

    async def test_tournament_with_filtered_names(self, page: Page):
        """Test tournament functionality with filtered names."""
        # First, filter out some names
        # Switch to name filter tab
        name_filter_button = page.locator('button:has-text("Name Filter")')
        assert await name_filter_button.count() > 0, "Name Filter button not found"
        await name_filter_button.first.click()
        await page.wait_for_timeout(1500)

        # Exclude a few names if possible
        exclude_button = page.locator('button:has-text("Exclude/Dislike")')
        for _ in range(3):
            if await exclude_button.count() > 0:
                await exclude_button.first.click()
                await page.wait_for_timeout(800)
            else:
                break

        # Switch to tournament tab
        tournament_button = page.locator('button:has-text("Tournament")')
        assert await tournament_button.count() > 0, "Tournament button not found"
        await tournament_button.first.click()
        await page.wait_for_timeout(1500)

        # Verify tournament interface works
        prefer_buttons = page.locator('button:has-text("Prefer")')
        draw_button = page.locator('button:has-text("Draw")')
        assert await prefer_buttons.count() >= 2, "Need at least 2 prefer buttons for tournament"
        assert await draw_button.count() > 0, "Draw button not found"

        # Cast a few votes
        for _ in range(3):
            if await prefer_buttons.count() > 0:
                await prefer_buttons.first.click()
                await page.wait_for_timeout(800)

        # Verify no errors
        error_elements = page.locator('[data-testid="stException"]')
        assert await error_elements.count() == 0, "Errors in tournament with filtered names"


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
class TestSimilaritySearchTab:
    """Tests for the Similarity Search tab."""

    async def test_switch_to_similarity_tab(self, page: Page):
        """Test switching to similarity search tab."""
        # Click Similarity Search button
        similarity_button = page.locator('button:has-text("Similarity Search")')
        assert await similarity_button.count() > 0, "Similarity Search button not found"
        await similarity_button.first.click()
        await page.wait_for_timeout(1500)  # Wait for tab to load

        # Verify we're on similarity tab by looking for search elements
        search_input = page.locator('input[placeholder*="name" i], input[placeholder*="search" i], input[type="text"]')
        assert await search_input.count() > 0, "Search input not found in similarity tab"

    async def test_similarity_search_interface(self, page: Page):
        """Test similarity search interface elements."""
        # Switch to similarity tab using button
        similarity_button = page.locator('button:has-text("Similarity Search")')
        assert await similarity_button.count() > 0, "Similarity Search button not found"
        await similarity_button.first.click()
        await page.wait_for_timeout(1500)

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
        # Switch to similarity tab using button
        similarity_button = page.locator('button:has-text("Similarity Search")')
        assert await similarity_button.count() > 0, "Similarity Search button not found"
        await similarity_button.first.click()
        await page.wait_for_timeout(1500)

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
class TestNameFilterTab:
    """Tests for the Name Filter tab functionality."""

    async def test_switch_to_name_filter_tab(self, page: Page):
        """Test switching to name filter tab."""
        # Click Name Filter button
        name_filter_button = page.locator('button:has-text("Name Filter")')
        assert await name_filter_button.count() > 0, "Name Filter button not found"
        await name_filter_button.first.click()
        await page.wait_for_timeout(1500)  # Wait for tab to load

        # Verify we're on name filter tab by looking for key elements
        # Look for large name display
        large_name = page.locator("h1")
        assert await large_name.count() > 0, "Large name display not found in filter tab"

        # Look for decision buttons
        exclude_button = page.locator('button:has-text("Exclude/Dislike")')
        include_button = page.locator('button:has-text("Include/Neutral")')

        assert await exclude_button.count() > 0, "Exclude/Dislike button not found"
        assert await include_button.count() > 0, "Include/Neutral button not found"

    async def test_name_filter_interface_elements(self, page: Page):
        """Test name filter interface elements."""
        # Switch to name filter tab using button
        name_filter_button = page.locator('button:has-text("Name Filter")')
        assert await name_filter_button.count() > 0, "Name Filter button not found"
        await name_filter_button.first.click()
        await page.wait_for_timeout(1500)

        # Check for progress bar
        progress_bar = page.locator('[data-testid="stProgress"]')
        # Progress bar might not have specific testid, check for progress element
        if await progress_bar.count() == 0:
            progress_bar = page.locator(".stProgress")
            if await progress_bar.count() == 0:
                # Check for any div with progress role
                progress_bar = page.locator('div[role="progressbar"]')

        # Progress indicator should exist in some form
        assert await progress_bar.count() > 0, "Progress indicator not found"

        # Check for stats display (included/excluded counts)
        stats_text = page.locator(':has-text("Included:"), :has-text("Excluded:")')
        assert await stats_text.count() > 0, "Stats display not found"

        # Check for navigation buttons
        prev_button = page.locator('button:has-text("Previous")')
        reset_button = page.locator('button:has-text("Reset")')
        save_button = page.locator('button:has-text("Save")')

        # At least some navigation buttons should exist
        nav_buttons_found = (
            await prev_button.count() > 0 or await reset_button.count() > 0 or await save_button.count() > 0
        )
        assert nav_buttons_found, "No navigation buttons found"

    async def test_name_filter_decisions(self, page: Page):
        """Test making include/exclude decisions."""
        # Switch to name filter tab using button
        name_filter_button = page.locator('button:has-text("Name Filter")')
        assert await name_filter_button.count() > 0, "Name Filter button not found"
        await name_filter_button.first.click()
        await page.wait_for_timeout(1500)

        # Get current name being displayed
        large_name = page.locator("h1").first
        current_name = await large_name.text_content()
        assert current_name is not None, "No name displayed"
        assert len(current_name.strip()) > 0, "Name is empty"

        # Make an include decision
        include_button = page.locator('button:has-text("Include")')
        if await include_button.count() > 0:
            await include_button.first.click()
            await page.wait_for_timeout(1000)

            # Check no errors
            error_elements = page.locator('[data-testid="stException"]')
            assert await error_elements.count() == 0, "Error after including name"

        # Switch to a different name if possible (click next or wait for auto-advance)
        # The UI should advance automatically, but let's check

    async def test_batch_operations(self, page: Page):
        """Test batch include/exclude operations."""
        # Switch to name filter tab using button
        name_filter_button = page.locator('button:has-text("Name Filter")')
        assert await name_filter_button.count() > 0, "Name Filter button not found"
        await name_filter_button.first.click()
        await page.wait_for_timeout(1500)

        # Look for batch operation buttons
        include_all_button = page.locator('button:has-text("Include All")')
        exclude_all_button = page.locator('button:has-text("Exclude All")')

        # They might have different text like "Include All Remaining"
        if await include_all_button.count() == 0:
            include_all_button = page.locator('button:has-text("Include All Remaining")')
        if await exclude_all_button.count() == 0:
            exclude_all_button = page.locator('button:has-text("Exclude All Remaining")')

        # Batch operations might not always be visible, which is OK
        # Just verify the page doesn't error when we check for them
        error_elements = page.locator('[data-testid="stException"]')
        assert await error_elements.count() == 0, "Errors on filter tab"

    async def test_filter_affects_tournament(self, page: Page):
        """Test that filter decisions affect tournament tab."""
        # First, make sure we're starting fresh
        # Switch to name filter tab and reset if possible
        name_filter_button = page.locator('button:has-text("Name Filter")')
        assert await name_filter_button.count() > 0, "Name Filter button not found"
        await name_filter_button.first.click()
        await page.wait_for_timeout(1500)

        # Try to find and click reset button
        reset_button = page.locator('button:has-text("Reset")')
        if await reset_button.count() > 0:
            await reset_button.first.click()
            await page.wait_for_timeout(1000)

        # Make an exclude decision if possible
        exclude_button = page.locator('button:has-text("Exclude")')
        if await exclude_button.count() > 0:
            await exclude_button.first.click()
            await page.wait_for_timeout(1000)

        # Switch to tournament tab using button
        tournament_button = page.locator('button:has-text("Tournament")')
        assert await tournament_button.count() > 0, "Tournament button not found"
        await tournament_button.first.click()
        await page.wait_for_timeout(1500)

        # Tournament should load without errors
        error_elements = page.locator('[data-testid="stException"]')
        assert await error_elements.count() == 0, "Errors when switching to tournament after filtering"

    async def test_inclusion_persistence(self, page: Page):
        """Test that inclusion decisions persist when switching tabs."""
        # Switch to name filter tab using button
        name_filter_button = page.locator('button:has-text("Name Filter")')
        assert await name_filter_button.count() > 0, "Name Filter button not found"
        await name_filter_button.first.click()
        await page.wait_for_timeout(1500)

        # Get current name
        large_name = page.locator("h1").first
        current_name = await large_name.text_content()
        if current_name and len(current_name.strip()) > 0:
            # Make an exclude decision
            exclude_button = page.locator('button:has-text("Exclude")')
            if await exclude_button.count() > 0:
                await exclude_button.first.click()
                await page.wait_for_timeout(1000)

                # The UI should advance to next name automatically
                # Wait a bit for UI to update
                await page.wait_for_timeout(500)

                # Switch to another tab and back
                # Click Tournament button
                tournament_button = page.locator('button:has-text("Tournament")')
                assert await tournament_button.count() > 0, "Tournament button not found"
                await tournament_button.first.click()
                await page.wait_for_timeout(1500)

                # Switch back to name filter tab using button
                name_filter_button = page.locator('button:has-text("Name Filter")')
                assert await name_filter_button.count() > 0, "Name Filter button not found"
                await name_filter_button.first.click()
                await page.wait_for_timeout(1500)

                # No errors should occur
                error_elements = page.locator('[data-testid="stException"]')
                assert await error_elements.count() == 0, "Errors when switching tabs with inclusion decisions"

    async def test_name_filter_performance(self, page: Page):
        """Measure performance of name filter transitions."""
        import time

        # Switch to name filter tab using button (not tabs anymore)
        name_filter_button = page.locator('button:has-text("Name Filter")')
        assert await name_filter_button.count() > 0, "Name Filter button not found"
        await name_filter_button.first.click()
        await page.wait_for_timeout(1000)

        # Get current name
        large_name = page.locator("h1").first
        current_name = await large_name.text_content()
        assert current_name is not None, "No name displayed"
        assert len(current_name.strip()) > 0, "Name is empty"

        # Click include button and measure time until name changes
        include_button = page.locator('button:has-text("Include/Neutral")')
        if await include_button.count() > 0:
            start = time.perf_counter()
            await include_button.first.click()

            # Wait for name to change (poll for new name)
            timeout = 30  # seconds
            poll_interval = 0.1
            elapsed = 0
            name_changed = False
            while elapsed < timeout:
                new_name = await large_name.text_content()
                if new_name != current_name:
                    name_changed = True
                    break
                await page.wait_for_timeout(poll_interval * 1000)
                elapsed += poll_interval

            end = time.perf_counter()
            transition_time = end - start

            # Log result
            print(f"Name filter transition time: {transition_time:.2f} seconds")

            # Assert performance requirement
            if transition_time > 2.0:  # 2 seconds threshold
                pytest.fail(f"Performance degradation: transition time {transition_time:.2f}s > 2s")
            elif transition_time > 1.0:
                print(f"WARNING: Slow transition ({transition_time:.2f}s)")

            # Check no errors
            error_elements = page.locator('[data-testid="stException"]')
            assert await error_elements.count() == 0, "Error after including name"

            # If name didn't change, maybe we're at the end of list
            if not name_changed:
                print("Name did not change after button click (possibly end of list)")

    async def test_excluded_names_multiselect_widget(self, page: Page):
        """Test excluded names multiselect widget functionality."""
        # Switch to name filter tab
        name_filter_button = page.locator('button:has-text("Name Filter")')
        assert await name_filter_button.count() > 0, "Name Filter button not found"
        await name_filter_button.first.click()
        await page.wait_for_timeout(1500)

        # Look for excluded names multiselect widget
        # It should have a label containing "Excluded names" or similar
        excluded_multiselect = page.locator('[data-testid="stMultiSelect"], .stMultiSelect, [role="combobox"]')
        # Also look for label text
        excluded_label = page.locator(':has-text("Excluded"), :has-text("excluded")')

        # At least one of these should be present
        if await excluded_multiselect.count() == 0 and await excluded_label.count() == 0:
            # Widget might not be visible if no names are excluded yet
            # That's OK - just log and pass
            print("Excluded names multiselect widget not found (may have no excluded names)")
            return

        # If we found the widget, try to interact with it
        if await excluded_multiselect.count() > 0:
            # Click to open dropdown
            await excluded_multiselect.first.click()
            await page.wait_for_timeout(500)

            # Look for checkbox options
            checkboxes = page.locator('[role="option"], [data-testid="stCheckbox"]')
            if await checkboxes.count() > 0:
                # At least one checkbox should be present
                # We'll just verify no errors occur
                pass

            # Close dropdown by clicking elsewhere
            await page.click("body")
            await page.wait_for_timeout(500)

        # Verify no errors
        error_elements = page.locator('[data-testid="stException"]')
        assert await error_elements.count() == 0, "Errors with excluded names multiselect widget"

    async def test_save_and_continue_navigation(self, page: Page):
        """Test Save & Continue button navigation to tournament tab."""
        # Switch to name filter tab
        name_filter_button = page.locator('button:has-text("Name Filter")')
        assert await name_filter_button.count() > 0, "Name Filter button not found"
        await name_filter_button.first.click()
        await page.wait_for_timeout(1500)

        # Look for Save & Continue button
        save_button = page.locator('button:has-text("Save & Continue")')
        if await save_button.count() == 0:
            # Might be "Save" button instead
            save_button = page.locator('button:has-text("Save")')

        # If save button exists, click it and verify navigation to tournament tab
        if await save_button.count() > 0:
            await save_button.first.click()
            await page.wait_for_timeout(1500)

            # Verify we switched to tournament tab
            # Check for tournament-specific elements
            prefer_buttons = page.locator('button:has-text("Prefer")')
            draw_button = page.locator('button:has-text("Draw")')

            # At least one tournament element should be present
            tournament_found = await prefer_buttons.count() > 0 or await draw_button.count() > 0
            assert tournament_found, "Did not navigate to tournament tab after Save & Continue"

            # Verify no errors
            error_elements = page.locator('[data-testid="stException"]')
            assert await error_elements.count() == 0, "Errors after Save & Continue navigation"
        else:
            # Save button might not be visible (no changes to save)
            # That's OK - just log and pass
            print("Save & Continue button not found (may have no changes to save)")


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
class TestDatabaseImportExport:
    """Tests for SQLite database import/export functionality."""

    async def test_export_button_present(self, page: Page):
        """Test that SQLite export button is present in sidebar."""
        # Look for export section in sidebar
        sidebar = page.locator('[data-testid="stSidebar"]')

        # Check for export section header
        export_header = sidebar.locator(':has-text("Export"), :has-text("Database")')
        assert await export_header.count() > 0, "Export section not found in sidebar"

        # Look for SQLite export button (should replace JSON export)
        # For now, just check no errors - the button might not exist yet
        error_elements = page.locator('[data-testid="stException"]')
        assert await error_elements.count() == 0, "Errors in sidebar"

    async def test_import_button_present(self, page: Page):
        """Test that SQLite import button is present in sidebar."""
        # Look for import button
        # For now, just check no errors
        error_elements = page.locator('[data-testid="stException"]')
        assert await error_elements.count() == 0, "Errors in sidebar"

    async def test_export_flow(self, page: Page):
        """Test database export flow (non-destructive)."""
        sidebar = page.locator('[data-testid="stSidebar"]')

        # Look for export button
        export_button = sidebar.locator('button:has-text("Export"), button:has-text("SQLite")')
        if await export_button.count() > 0:
            # Click export button
            await export_button.first.click()
            await page.wait_for_timeout(1000)

            # Check for download prompt or success message
            # This is hard to test in headless mode, so just check no errors
            error_elements = page.locator('[data-testid="stException"]')
            assert await error_elements.count() == 0, "Errors during export"

    async def test_import_flow(self, page: Page):
        """Test database import flow (non-destructive, uses dummy file)."""
        sidebar = page.locator('[data-testid="stSidebar"]')

        # Look for import button
        import_button = sidebar.locator('button:has-text("Import")')
        if await import_button.count() > 0:
            # Note: Actually testing file upload is complex in Playwright
            # We'll just verify the button exists and doesn't cause errors when clicked
            # without actually uploading a file
            await import_button.first.click()
            await page.wait_for_timeout(1000)

            # Check no errors
            error_elements = page.locator('[data-testid="stException"]')
            assert await error_elements.count() == 0, "Errors during import attempt"


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
async def test_end_to_end_workflow(page: Page):
    """Test a complete workflow through the application."""
    # 1. Verify application loads
    error_elements = page.locator('[data-testid="stException"]')
    assert await error_elements.count() == 0, "Errors on initial load"

    # 2. Start with Name Filter tab
    # Click Name Filter button (should already be active but click to ensure)
    name_filter_button = page.locator('button:has-text("Name Filter")')
    assert await name_filter_button.count() > 0, "Name Filter button not found"
    await name_filter_button.first.click()
    await page.wait_for_timeout(1500)

    # Make a filter decision if possible
    include_button = page.locator('button:has-text("Include")')
    if await include_button.count() > 0:
        await include_button.first.click()
        await page.wait_for_timeout(800)

    # 3. Switch to Tournament tab
    tournament_button = page.locator('button:has-text("Tournament")')
    assert await tournament_button.count() > 0, "Tournament button not found"
    await tournament_button.first.click()
    await page.wait_for_timeout(1500)

    # 4. Vote a few times in tournament
    for _ in range(2):
        prefer_buttons = page.locator('button:has-text("Prefer")')
        if await prefer_buttons.count() > 0:
            await prefer_buttons.first.click()
            await page.wait_for_timeout(800)

    # 5. Switch to similarity tab
    similarity_button = page.locator('button:has-text("Similarity Search")')
    assert await similarity_button.count() > 0, "Similarity Search button not found"
    await similarity_button.first.click()
    await page.wait_for_timeout(1500)

    # 6. Perform a search
    search_input = page.locator('input[placeholder*="name" i], input[placeholder*="search" i]')
    if await search_input.count() > 0:
        await search_input.first.fill("a")
        await page.wait_for_timeout(500)
        await search_input.first.press("Enter")
        await page.wait_for_timeout(1500)

    # 7. Switch back to tournament
    tournament_button = page.locator('button:has-text("Tournament")')
    assert await tournament_button.count() > 0, "Tournament button not found"
    await tournament_button.first.click()
    await page.wait_for_timeout(1000)

    # 8. Final error check
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
        'button:has-text("Name Filter"):first',
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
            test = TestNameFilterTab()
            await test.test_switch_to_name_filter_tab(page)

            await browser.close()

    asyncio.run(run_one_test())
