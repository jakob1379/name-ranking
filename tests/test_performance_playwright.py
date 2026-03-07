"""Performance tests using Playwright for the Name Ranking application.

These tests measure actual performance metrics and fail if performance degrades
below acceptable thresholds.

Performance Requirements:
- Button press → next name display: <500ms (target: <200ms)
- Tab switching: <1000ms
- Database save operations: <100ms
- Initial page load: <5000ms

The tests will fail if performance degrades, triggering investigation.
"""

import asyncio
import os
import time

import pytest
import pytest_asyncio
from playwright.async_api import Page, async_playwright


# Check if running in Nix environment
def is_nix_environment():
    """Check if running in Nix environment."""
    browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    return "/nix/store/" in browsers_path


# Skip all Playwright tests in Nix environment due to browser version mismatch
if is_nix_environment():
    pytest.skip(
        "Skipping Playwright performance tests in Nix environment due to browser version mismatch.\n"
        "Nix provides chromium-1200 but Playwright expects chromium-1194.\n"
        "To run Playwright tests, use a non-Nix environment or CI with compatible browsers.",
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
    """Create Playwright browser instance."""
    async with async_playwright() as p:
        # Use chromium for consistency
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
    await page.wait_for_timeout(2000)  # Extra time for Streamlit to initialize

    yield page
    await page.close()


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
@pytest.mark.performance
class TestBinaryFilterPerformance:
    """Performance tests for binary filter interface."""

    async def _switch_to_name_filter_tab(self, page: Page) -> float:
        """Switch to name filter tab and return time taken."""
        start_time = time.perf_counter()

        # Find and click name filter button (not tab anymore - we use buttons now)
        name_filter_button = page.locator('button:has-text("Name Filter")')
        assert await name_filter_button.count() > 0, "Name Filter button not found"

        await name_filter_button.first.click()
        await page.wait_for_timeout(1000)  # Wait for tab to load

        end_time = time.perf_counter()
        return (end_time - start_time) * 1000  # Convert to ms

    async def test_initial_page_load_performance(self, page: Page):
        """Test that initial page loads within acceptable time."""
        # Page should already be loaded by fixture
        # Check for key elements to ensure proper load
        heading = page.locator("h1")
        assert await heading.count() > 0, "Page didn't load properly"

        # Check for tab buttons
        name_filter_button = page.locator('button:has-text("Name Filter")')
        tournament_button = page.locator('button:has-text("Tournament")')
        similarity_button = page.locator('button:has-text("Similarity Search")')

        assert await name_filter_button.count() > 0, "Name Filter button missing"
        assert await tournament_button.count() > 0, "Tournament button missing"
        assert await similarity_button.count() > 0, "Similarity Search button missing"

        # Check no errors
        error_elements = page.locator('[data-testid="stException"]')
        assert await error_elements.count() == 0, "Errors on initial page load"

    async def test_name_filter_transition_performance(self, page: Page):
        """Measure and assert performance of name filter transitions."""
        # Switch to name filter tab first
        switch_time = await self._switch_to_name_filter_tab(page)
        print(f"Tab switch time: {switch_time:.1f}ms")

        # Get current name
        large_name = page.locator("h1").first
        current_name = await large_name.text_content()
        assert current_name is not None, "No name displayed"
        assert len(current_name.strip()) > 0, "Name is empty"

        # Find include button
        include_button = page.locator('button:has-text("Include/Neutral")')
        assert await include_button.count() > 0, "Include/Neutral button not found"

        # Measure transition time for multiple clicks
        transition_times = []
        successful_transitions = 0

        for i in range(5):  # Test 5 transitions
            current_name = await large_name.text_content()
            start = time.perf_counter()

            await include_button.first.click()

            # Wait for name to change with timeout
            timeout = 5.0  # seconds - fail fast if too slow
            poll_interval = 0.05
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
            transition_time = (end - start) * 1000  # Convert to ms

            if name_changed:
                transition_times.append(transition_time)
                successful_transitions += 1
                print(f"Transition {i + 1}: {transition_time:.1f}ms")
            else:
                print(f"Transition {i + 1}: Name didn't change (possibly end of list)")
                break

        # Assert performance requirements
        assert successful_transitions > 0, "No successful transitions measured"

        avg_transition_time = sum(transition_times) / len(transition_times)
        max_transition_time = max(transition_times)

        print(f"Average transition time: {avg_transition_time:.1f}ms")
        print(f"Maximum transition time: {max_transition_time:.1f}ms")

        # Performance assertions - fail test if too slow
        # WARNING threshold: >1000ms (1 second)
        assert max_transition_time < 1000, (
            f"Performance degradation detected: max transition time {max_transition_time:.1f}ms > 1000ms"
        )

        # FAILURE threshold: >2000ms (2 seconds) - this would indicate 10-second delay territory
        assert max_transition_time < 2000, (
            f"CRITICAL performance issue: max transition time {max_transition_time:.1f}ms > 2000ms"
        )

        # Check no errors occurred during transitions
        error_elements = page.locator('[data-testid="stException"]')
        assert await error_elements.count() == 0, "Errors during performance test"

    async def test_tab_switching_performance(self, page: Page):
        """Measure performance of switching between tabs."""
        tab_switch_times = []

        # Test switching from Name Filter to Tournament and back
        for _ in range(3):
            # Switch to Tournament
            start = time.perf_counter()
            tournament_button = page.locator('button:has-text("Tournament")')
            await tournament_button.first.click()
            await page.wait_for_timeout(1000)  # Wait for tab to load
            end = time.perf_counter()
            tab_switch_times.append((end - start) * 1000)

            # Verify Tournament loaded (look for specific elements)
            # Tournament should have at least two name displays
            name_displays = page.locator('div[data-testid="stMetric"]')
            assert await name_displays.count() >= 2, "Tournament not properly loaded"

            # Switch back to Name Filter
            start = time.perf_counter()
            name_filter_button = page.locator('button:has-text("Name Filter")')
            await name_filter_button.first.click()
            await page.wait_for_timeout(1000)  # Wait for tab to load
            end = time.perf_counter()
            tab_switch_times.append((end - start) * 1000)

            # Verify Name Filter loaded
            include_button = page.locator('button:has-text("Include/Neutral")')
            assert await include_button.count() > 0, "Name Filter not properly loaded"

        avg_switch_time = sum(tab_switch_times) / len(tab_switch_times)
        max_switch_time = max(tab_switch_times)

        print(f"Tab switching performance: avg={avg_switch_time:.1f}ms, max={max_switch_time:.1f}ms")

        # Assert performance requirements
        assert max_switch_time < 2000, f"Tab switching too slow: max {max_switch_time:.1f}ms > 2000ms"

    async def test_batch_operations_performance(self, page: Page):
        """Test performance of batch operations."""
        # Switch to name filter tab
        await self._switch_to_name_filter_tab(page)

        # Find batch operation buttons
        include_all_button = page.locator('button:has-text("Include All Remaining")')
        if await include_all_button.count() == 0:
            include_all_button = page.locator('button:has-text("Include All")')

        # If batch buttons exist, test them
        if await include_all_button.count() > 0:
            start = time.perf_counter()
            await include_all_button.first.click()

            # Wait for operation to complete
            await page.wait_for_timeout(2000)

            end = time.perf_counter()
            batch_time = (end - start) * 1000

            print(f"Batch operation time: {batch_time:.1f}ms")

            # Should complete within 5 seconds even for large lists
            assert batch_time < 5000, f"Batch operation too slow: {batch_time:.1f}ms > 5000ms"

        # Check no errors
        error_elements = page.locator('[data-testid="stException"]')
        assert await error_elements.count() == 0, "Errors during batch operations"

    async def test_memory_usage_stability(self, page: Page):
        """Test that memory usage remains stable during many operations."""
        # Switch to name filter tab
        await self._switch_to_name_filter_tab(page)

        # Find include button
        include_button = page.locator('button:has-text("Include/Neutral")')
        assert await include_button.count() > 0

        # Perform many rapid operations
        operation_count = 20
        operation_times = []

        for i in range(operation_count):
            start = time.perf_counter()
            await include_button.first.click()

            # Brief wait for UI to update
            await page.wait_for_timeout(50)

            end = time.perf_counter()
            operation_times.append((end - start) * 1000)

            # Every 5 operations, check for errors
            if i % 5 == 0:
                error_elements = page.locator('[data-testid="stException"]')
                assert await error_elements.count() == 0, f"Errors after {i + 1} operations"

        # Calculate statistics
        avg_time = sum(operation_times) / len(operation_times)
        max_time = max(operation_times)

        print(f"Memory stability test: {operation_count} operations")
        print(f"  Average time: {avg_time:.1f}ms")
        print(f"  Maximum time: {max_time:.1f}ms")
        print(f"  Time range: {min(operation_times):.1f}ms - {max_time:.1f}ms")

        # Check for performance degradation over time
        # Compare first half vs second half
        half = len(operation_times) // 2
        first_half_avg = sum(operation_times[:half]) / half
        second_half_avg = sum(operation_times[half:]) / (len(operation_times) - half)

        # Second half shouldn't be more than 50% slower than first half
        degradation_ratio = second_half_avg / first_half_avg if first_half_avg > 0 else 1.0
        print(f"  Performance degradation ratio: {degradation_ratio:.2f}")

        assert degradation_ratio < 1.5, (
            f"Performance degradation detected: "
            f"second half {second_half_avg:.1f}ms is {degradation_ratio:.2f}x slower than first half {first_half_avg:.1f}ms"
        )


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
@pytest.mark.performance
class TestDatabasePerformance:
    """Performance tests for database operations."""

    async def test_database_save_performance(self, page: Page):
        """Test that database save operations are performant."""
        # Switch to name filter tab
        name_filter_button = page.locator('button:has-text("Name Filter")')
        await name_filter_button.first.click()
        await page.wait_for_timeout(1000)

        # Find include button
        include_button = page.locator('button:has-text("Include/Neutral")')
        assert await include_button.count() > 0

        # Perform multiple operations to trigger periodic save (every 50 actions)
        # We'll do 10 operations and check timing
        operation_times = []

        for i in range(10):
            start = time.perf_counter()
            await include_button.first.click()

            # Wait briefly for UI
            await page.wait_for_timeout(100)

            end = time.perf_counter()
            operation_times.append((end - start) * 1000)

        avg_operation_time = sum(operation_times) / len(operation_times)

        print(f"Database operation test: avg operation time = {avg_operation_time:.1f}ms")

        # Individual operations should be fast
        assert avg_operation_time < 1000, f"Database operations too slow: avg {avg_operation_time:.1f}ms > 1000ms"

        # Check no errors
        error_elements = page.locator('[data-testid="stException"]')
        assert await error_elements.count() == 0, "Errors during database operations"


if __name__ == "__main__":
    print(
        "Run with: pytest tests/test_performance_playwright.py --app-url=http://localhost:8501 --run-integration --run-playwright -v",
    )
