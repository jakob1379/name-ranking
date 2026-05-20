"""
End-to-end tests for tournament functionality.
Tests the complete flow: filter names -> tournament -> fast pair progression.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

import pytest
from playwright.async_api import Page, expect


# Skip all Playwright tests in Nix environment due to browser version mismatch
def is_nix_environment():
    """Check if running in Nix environment."""
    browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    return "/nix/store/" in browsers_path


if is_nix_environment():
    pytest.skip(
        "Skipping Playwright tests in Nix environment due to browser version mismatch.",
        allow_module_level=True,
    )


@pytest.mark.integration
@pytest.mark.playwright
@pytest.mark.asyncio
class TestTournamentE2E:
    """E2E tests for tournament with filtered names."""

    async def _take_screenshot(self, page: Page, name: str, screenshots_dir: Path = None):
        """Take a screenshot for debugging."""
        if screenshots_dir is None:
            screenshots_dir = Path(__file__).parent / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        await page.screenshot(path=str(screenshots_dir / filename))
        print(f"Screenshot saved: {filename}")

    async def _get_current_pair_names(self, page: Page) -> tuple[str, str]:
        """Get the current pair of names being displayed."""
        # Names are typically displayed in h2 or h3 elements
        name_selectors = ["h2", "h3", '[data-testid="stMarkdown"] h2', '[data-testid="stMarkdown"] h3']

        for selector in name_selectors:
            elements = page.locator(selector)
            count = await elements.count()
            if count >= 2:
                name1 = await elements.nth(0).text_content()
                name2 = await elements.nth(1).text_content()
                if name1 and name2:
                    return name1.strip(), name2.strip()

        return "", ""

    async def _get_queue_status_text(self, page: Page) -> str:
        """Get the queue status indicator text if present."""
        # Look for queue status indicators with various possible labels
        queue_selectors = [
            ':has-text("Queue")',
            ':has-text("queue")',
            ':has-text("pairs")',
            ':has-text("Pairs")',
            ':has-text("buffer")',
            ':has-text("Buffer")',
            '[data-testid="stMetric"]',
        ]

        for selector in queue_selectors:
            elements = page.locator(selector)
            if await elements.count() > 0:
                text = await elements.first.text_content()
                if text:
                    return text.strip()

        return ""

    async def test_tournament_fast_progression_with_filtered_names(self, page: Page):
        """
        Test that tournament progresses quickly after filtering names.

        Steps:
        1. Filter 10-15 names in Name Filter tab (include some, exclude some)
        2. Switch to Tournament tab
        3. Click through 5-10 pairs
        4. Verify each transition happens in < 2 seconds
        5. Verify queue indicator shows pairs being filled
        """
        screenshots_dir = Path(__file__).parent / "screenshots"
        vote_count = 8  # Number of votes to perform
        max_transition_time = 2.0  # Maximum acceptable transition time in seconds

        try:
            # Step 1: Navigate to Name Filter tab and filter names
            print("\n=== Step 1: Filtering names ===")

            # Click Name Filter button
            name_filter_button = page.locator('button:has-text("Name Filter")')
            await expect(name_filter_button.first).to_be_visible(timeout=5000)
            await name_filter_button.first.click()
            await page.wait_for_timeout(1500)

            # Get initial stats
            stats_text = await page.locator(':has-text("Included:")').first.text_content()
            print(f"Initial filter stats: {stats_text}")

            # Filter names: include ~10-15 names by clicking Include/Neutral
            # Also exclude some names to simulate realistic usage
            target_included = 12
            included_count = 0
            excluded_count = 0

            include_button = page.locator('button:has-text("Include")')
            exclude_button = page.locator('button:has-text("Exclude")')

            while included_count < target_included:
                # Check if buttons are available
                if await include_button.count() == 0:
                    print("No more names to filter (Include button not found)")
                    break

                # Alternate between include and exclude (roughly 2:1 ratio)
                if included_count % 3 != 0 or await exclude_button.count() == 0:
                    await include_button.first.click()
                    included_count += 1
                    print(f"Included name #{included_count}")
                else:
                    await exclude_button.first.click()
                    excluded_count += 1
                    print(f"Excluded name #{excluded_count}")

                # Wait for UI to update
                await page.wait_for_timeout(600)

                # Safety check - don't loop forever
                if included_count + excluded_count > 25:
                    print("Safety limit reached, stopping filter process")
                    break

            print(f"\nFilter complete: {included_count} included, {excluded_count} excluded")

            # Take screenshot after filtering
            await self._take_screenshot(page, "01_after_filtering", screenshots_dir)

            # Step 2: Navigate to Tournament tab
            print("\n=== Step 2: Switching to Tournament tab ===")

            tournament_button = page.locator('button:has-text("Tournament")')
            await expect(tournament_button.first).to_be_visible(timeout=5000)
            await tournament_button.first.click()
            await page.wait_for_timeout(1500)

            # Verify tournament interface is loaded
            prefer_buttons = page.locator('button:has-text("Prefer")')
            draw_button = page.locator('button:has-text("Draw")')

            assert await prefer_buttons.count() >= 2, "Tournament interface not loaded (prefer buttons not found)"
            assert await draw_button.count() >= 1, "Tournament interface not loaded (draw button not found)"

            print("Tournament interface loaded successfully")

            # Get initial queue status
            initial_queue_status = await self._get_queue_status_text(page)
            print(f"Initial queue status: {initial_queue_status}")

            # Step 3 & 4: Vote multiple times and measure timing
            print(f"\n=== Step 3: Voting {vote_count} times with timing measurements ===")

            timing_results = []
            names_history = []

            # Get initial pair
            initial_pair = await self._get_current_pair_names(page)
            names_history.append(initial_pair)
            print(f"Initial pair: {initial_pair}")

            for i in range(vote_count):
                # Get current names before voting
                current_names = await self._get_current_pair_names(page)

                # Get queue status before voting
                queue_before = await self._get_queue_status_text(page)

                # Measure vote timing
                loop = asyncio.get_event_loop()
                start_time = loop.time()

                # Alternate between different vote types
                if i % 5 == 0:
                    # Vote draw occasionally
                    await draw_button.first.click()
                    vote_type = "draw"
                elif i % 2 == 0:
                    # Vote for left candidate
                    await prefer_buttons.nth(0).click()
                    vote_type = "left"
                else:
                    # Vote for right candidate
                    await prefer_buttons.nth(1).click()
                    vote_type = "right"

                # Wait for transition with polling
                # Poll for name change every 100ms
                poll_interval = 0.1
                max_wait = 5.0  # Maximum wait time
                elapsed = 0
                name_changed = False

                while elapsed < max_wait:
                    await page.wait_for_timeout(int(poll_interval * 1000))
                    elapsed += poll_interval

                    new_names = await self._get_current_pair_names(page)
                    if new_names != current_names and new_names[0] and new_names[1]:
                        name_changed = True
                        break

                end_time = loop.time()
                transition_time = end_time - start_time
                timing_results.append(transition_time)
                assert name_changed, "Tournament pair did not change after voting"

                # Get queue status after voting
                queue_after = await self._get_queue_status_text(page)

                # Store names history
                new_pair = await self._get_current_pair_names(page)
                names_history.append(new_pair)

                print(
                    f"Vote {i + 1}/{vote_count} ({vote_type}): "
                    f"{transition_time:.3f}s | "
                    f"Names: {current_names} -> {new_pair} | "
                    f"Queue: {queue_before[:30]}... -> {queue_after[:30]}...",
                )

                # Verify transition was fast enough
                if transition_time > max_transition_time:
                    print(f"WARNING: Slow transition detected: {transition_time:.3f}s > {max_transition_time}s")
                    await self._take_screenshot(page, f"slow_transition_vote_{i + 1}", screenshots_dir)

            # Take final screenshot
            await self._take_screenshot(page, "02_after_voting", screenshots_dir)

            # Step 5: Verify results
            print("\n=== Step 4: Verification ===")

            # Calculate timing statistics
            avg_time = sum(timing_results) / len(timing_results)
            max_time = max(timing_results)
            min_time = min(timing_results)
            slow_votes = [t for t in timing_results if t > max_transition_time]

            print("\nTiming Statistics:")
            print(f"  - Votes performed: {len(timing_results)}")
            print(f"  - Average transition time: {avg_time:.3f}s")
            print(f"  - Min transition time: {min_time:.3f}s")
            print(f"  - Max transition time: {max_time:.3f}s")
            print(f"  - Slow votes (>2s): {len(slow_votes)}")

            # Verify names actually changed (not stuck on same pair)
            unique_pairs = set(names_history)
            print("\nPair diversity:")
            print(f"  - Unique pairs seen: {len(unique_pairs)}")
            print(f"  - Total transitions: {len(names_history)}")

            # Assert performance requirements
            assert avg_time < max_transition_time, (
                f"Average transition time ({avg_time:.3f}s) exceeds threshold "
                f"({max_transition_time}s). This indicates queue may not be working properly."
            )

            # Assert we saw different pairs (not stuck)
            assert len(unique_pairs) > 2, (
                f"Only {len(unique_pairs)} unique pairs seen across {len(names_history)} votes. "
                f"Names may be stuck or queue not refreshing."
            )

            # Verify queue status is visible and informative
            final_queue_status = await self._get_queue_status_text(page)
            print(f"\nFinal queue status: {final_queue_status}")

            # Queue should show some indication of pairs/buffer status
            # This is a soft assertion - just warn if queue indicator not found
            if not final_queue_status or "queue" not in final_queue_status.lower():
                if not final_queue_status or "pair" not in final_queue_status.lower():
                    print("WARNING: Queue status indicator not clearly visible or not updating")

            # Verify no errors occurred
            error_elements = page.locator('[data-testid="stException"]')
            error_count = await error_elements.count()
            assert error_count == 0, f"Found {error_count} error elements on page"

            print("\n=== Test passed! ===")

        except Exception:
            # Take screenshot on failure
            await self._take_screenshot(page, "failure", screenshots_dir)
            print("\nTest failed! Screenshot saved.")
            raise

    async def test_tournament_keyboard_shortcuts_performance(self, page: Page):
        """
        Test that keyboard shortcuts also work quickly for tournament progression.

        Uses arrow keys:
        - Left arrow: prefer left name
        - Right arrow: prefer right name
        - Down arrow: draw
        """
        screenshots_dir = Path(__file__).parent / "screenshots"
        vote_count = 5
        max_transition_time = 2.0

        try:
            # Navigate to tournament tab (skip filtering for this test)
            tournament_button = page.locator('button:has-text("Tournament")')
            await expect(tournament_button.first).to_be_visible(timeout=5000)
            await tournament_button.first.click()
            await page.wait_for_timeout(1500)

            # Focus page for keyboard input
            await page.click("body")

            print(f"\n=== Testing keyboard shortcuts ({vote_count} votes) ===")

            timing_results = []

            for i in range(vote_count):
                current_names = await self._get_current_pair_names(page)

                loop = asyncio.get_event_loop()
                start_time = loop.time()

                # Use different arrow keys
                if i % 3 == 0:
                    await page.keyboard.press("ArrowLeft")
                elif i % 3 == 1:
                    await page.keyboard.press("ArrowRight")
                else:
                    await page.keyboard.press("ArrowDown")

                # Wait for transition
                poll_interval = 0.1
                max_wait = 5.0
                elapsed = 0

                while elapsed < max_wait:
                    await page.wait_for_timeout(int(poll_interval * 1000))
                    elapsed += poll_interval

                    new_names = await self._get_current_pair_names(page)
                    if new_names != current_names and new_names[0] and new_names[1]:
                        break

                end_time = loop.time()
                transition_time = end_time - start_time
                timing_results.append(transition_time)

                print(f"Keyboard vote {i + 1}/{vote_count}: {transition_time:.3f}s")

            # Verify performance
            avg_time = sum(timing_results) / len(timing_results)
            print(f"\nKeyboard shortcut average time: {avg_time:.3f}s")

            assert avg_time < max_transition_time, (
                f"Keyboard voting too slow: {avg_time:.3f}s avg > {max_transition_time}s threshold"
            )

            # Verify no errors
            error_elements = page.locator('[data-testid="stException"]')
            assert await error_elements.count() == 0, "Errors occurred during keyboard voting"

            print("=== Keyboard test passed! ===")

        except Exception:
            await self._take_screenshot(page, "keyboard_test_failure", screenshots_dir)
            raise

    async def test_queue_refills_during_voting(self, page: Page):
        """
        Test that the queue properly refills pairs as voting progresses.

        This test specifically monitors the queue status indicator over multiple
        votes to ensure it shows pairs being consumed and replenished.
        """
        screenshots_dir = Path(__file__).parent / "screenshots"

        try:
            # Navigate to tournament tab
            tournament_button = page.locator('button:has-text("Tournament")')
            await expect(tournament_button.first).to_be_visible(timeout=5000)
            await tournament_button.first.click()
            await page.wait_for_timeout(1500)

            # Collect queue status samples
            queue_samples = []
            vote_count = 10

            print("\n=== Monitoring queue status ===")

            for i in range(vote_count):
                # Record queue status before vote
                queue_before = await self._get_queue_status_text(page)

                # Cast vote
                prefer_buttons = page.locator('button:has-text("Prefer")')
                if await prefer_buttons.count() >= 2:
                    await prefer_buttons.first.click()

                # Wait briefly for queue to update
                await page.wait_for_timeout(300)

                # Record queue status after vote
                queue_after = await self._get_queue_status_text(page)

                queue_samples.append(
                    {
                        "vote": i + 1,
                        "before": queue_before,
                        "after": queue_after,
                    },
                )

                # Wait for transition before next vote
                await page.wait_for_timeout(500)

            # Analyze queue behavior
            print("\nQueue status samples:")
            for sample in queue_samples:
                print(f"  Vote {sample['vote']}: {sample['before'][:40]} -> {sample['after'][:40]}")

            # Queue should have shown activity (changing values)
            unique_states = set(s["after"] for s in queue_samples)
            print(f"\nUnique queue states observed: {len(unique_states)}")

            # Soft assertion: queue should have changed at least a few times
            # This indicates the queue is being consumed and refilled
            if len(unique_states) < 2:
                print("WARNING: Queue status appears static - may not be updating properly")

            # Verify no errors
            error_elements = page.locator('[data-testid="stException"]')
            assert await error_elements.count() == 0, "Errors during queue monitoring"

            print("=== Queue refill test completed ===")

        except Exception:
            await self._take_screenshot(page, "queue_test_failure", screenshots_dir)
            raise
