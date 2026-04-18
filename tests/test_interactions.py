"""Buttons, toggles, filters — non-destructive checks."""

from __future__ import annotations

import urllib.parse

import pytest
from playwright.sync_api import Page, expect

from tests.e2e_resilience import goto_with_backoff
from tests.test_full_app_flow import _on_login_page


@pytest.mark.e2e
def test_theme_toggle_present_on_dashboard(page: Page, base_url: str, api_base: str):
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, f"{base_url}/dashboard.html?api={q}", timeout=20000)
    page.wait_for_timeout(1200)
    if _on_login_page(page):
        pytest.skip("Not authenticated")
    btn = page.locator(".theme-toggle:visible, #themeToggle:visible, button[onclick*='toggleTheme']:visible").first
    expect(btn).to_be_visible(timeout=10000)


@pytest.mark.e2e
def test_menu_toggle_present_mobile_viewport(page: Page, base_url: str, api_base: str):
    page.set_viewport_size({"width": 390, "height": 844})
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, f"{base_url}/dashboard.html?api={q}", timeout=20000)
    page.wait_for_timeout(1000)
    if _on_login_page(page):
        pytest.skip("Not authenticated")
    menu = page.locator("#menuToggle")
    expect(menu).to_be_visible(timeout=8000)
