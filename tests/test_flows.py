"""Page loads, sidebar presence, and basic navigation (E2E)."""

from __future__ import annotations

import urllib.parse

import pytest
from playwright.sync_api import Page, expect

from tests.e2e_resilience import goto_with_backoff
from tests.test_full_app_flow import _on_login_page, _skip_if_login


@pytest.mark.e2e
def test_public_index_loads(page: Page, base_url: str, api_base: str):
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, f"{base_url}/index.html?api={q}", timeout=20000)
    expect(page.locator("body")).to_contain_text("Upload", ignore_case=True)


@pytest.mark.e2e
def test_login_page_loads(page: Page, base_url: str, api_base: str):
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, f"{base_url}/login.html?api={q}", timeout=20000)
    expect(page.locator("#loginForm")).to_be_visible()


@pytest.mark.e2e
def test_dashboard_sidebar_after_auth(page: Page, base_url: str, api_base: str):
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, f"{base_url}/dashboard.html?api={q}", timeout=20000)
    page.wait_for_timeout(1500)
    if _on_login_page(page):
        pytest.skip("Not authenticated")
    _skip_if_login(page)
    # Injected sidebar contains nav links
    expect(page.locator("#sidebar")).to_be_visible(timeout=10000)
    expect(page.locator(".sidebar-nav")).to_be_visible(timeout=10000)


@pytest.mark.e2e
def test_nav_to_upload_via_goto(page: Page, base_url: str, api_base: str):
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, f"{base_url}/upload.html?api={q}", timeout=25000)
    page.wait_for_timeout(1200)
    if _on_login_page(page):
        pytest.skip("Not authenticated")
    expect(page.locator("body")).to_contain_text("Upload", ignore_case=True)
