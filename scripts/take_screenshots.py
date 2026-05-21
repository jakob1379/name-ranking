#!/usr/bin/env python3
"""
Screenshot utility for Name Ranking application.

This script helps capture screenshots of the application for documentation.
It requires the Streamlit app to be running locally.

Usage:
    uv run python scripts/take_screenshots.py [--output-dir DIR]

Requirements:
    playwright (install with: uv run playwright install)
"""

import argparse
import asyncio
from pathlib import Path

DEFAULT_APP_HOST = "localhost"
DEFAULT_APP_PORT = 8501
DEFAULT_APP_URL = f"http://{DEFAULT_APP_HOST}:{DEFAULT_APP_PORT}"

try:
    from playwright.async_api import async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


async def capture_screenshots(output_dir: Path, url: str = DEFAULT_APP_URL) -> None:
    """Capture screenshots of the running application."""
    if not PLAYWRIGHT_AVAILABLE:
        msg = "playwright is not available; install browser binaries with `uv run playwright install chromium`"
        raise RuntimeError(msg)

    output_dir.mkdir(exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})

        print(f"Navigating to {url}...")
        await page.goto(url, wait_until="networkidle")

        # Wait for application to load
        await page.wait_for_timeout(2000)

        # Capture different views
        screenshots = {
            "01_home.png": "Main tournament view",
            "02_similarity.png": "Similarity search tab",
            "03_sidebar.png": "Sidebar controls",
        }

        for filename, description in screenshots.items():
            print(f"Capturing {description}...")
            # For different tabs, we need to click navigation
            if "similarity" in filename:
                # Click the similarity search tab
                await page.click('button[data-testid="stTab"][aria-label="Similarity Search"]', timeout=5000)
                await page.wait_for_timeout(1000)
            elif "sidebar" in filename:
                # Ensure sidebar is open (it usually is)
                await page.wait_for_timeout(500)

            await page.screenshot(path=output_dir / filename, full_page=True)

        await browser.close()

    print(f"\nScreenshots saved to {output_dir}/")
    print("Suggested usage in documentation:")
    print("  ![Main Tournament View](screenshots/01_home.png)")
    print("  ![Similarity Search](screenshots/02_similarity.png)")
    print("  ![Sidebar Controls](screenshots/03_sidebar.png)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture application screenshots")
    parser.add_argument(
        "--output-dir",
        default="screenshots",
        help="Output directory for screenshots (default: screenshots/)",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_APP_URL,
        help=f"URL of running Streamlit app (default: {DEFAULT_APP_URL})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not PLAYWRIGHT_AVAILABLE:
        print("=" * 60)
        print("Playwright not available. To install:")
        print("  1. Install playwright package (already in dev dependencies)")
        print("  2. Install browser binaries:")
        print("     uv run playwright install chromium")
        print()
        print("Alternatively, take screenshots manually:")
        print("  1. Start the application:")
        print("     uv run streamlit run src/st_name_ranking/main.py")
        print(f"  2. Open {DEFAULT_APP_URL} in your browser")
        print("  3. Capture screenshots of:")
        print("     - Main tournament view")
        print("     - Similarity search tab")
        print("     - Sidebar controls")
        print("=" * 60)
        return

    asyncio.run(capture_screenshots(output_dir, str(args.url)))


if __name__ == "__main__":
    main()
