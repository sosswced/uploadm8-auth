"""Optional full-page screenshots for visual review."""

from __future__ import annotations

import urllib.parse
from pathlib import Path

import pytest
from playwright.sync_api import Page

from tests.test_full_app_flow import _on_login_page

SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"


@pytest.mark.e2e
def test_screenshot_dashboard(logged_in_page: Page, base_url: str, api_base: str):
    page = logged_in_page
    if _on_login_page(page):
        pytest.skip("Not authenticated")
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    q = urllib.parse.quote(api_base, safe="")
    page.goto(f"{base_url}/dashboard.html?api={q}", wait_until="networkidle", timeout=30000)
    page.wait_for_timeout(1500)
    out = SCREENSHOT_DIR / "dashboard.png"
    page.screenshot(path=str(out), full_page=True)
    assert out.exists()
