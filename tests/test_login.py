"""Login flow E2E."""

from __future__ import annotations

import os
import re
import urllib.parse

import pytest
from playwright.sync_api import Page, expect

from tests.e2e_resilience import goto_with_backoff, is_rate_limited


@pytest.mark.e2e
def test_login_success_redirects_to_dashboard(page: Page, base_url: str, api_base: str):
    email = (os.environ.get("PLAYWRIGHT_TEST_EMAIL") or os.environ.get("PLAYWRIGHT_ADMIN_EMAIL") or "").strip()
    password = (os.environ.get("PLAYWRIGHT_TEST_PASSWORD") or os.environ.get("PLAYWRIGHT_ADMIN_PASSWORD") or "").strip()
    if not email or not password:
        pytest.skip("PLAYWRIGHT_TEST_EMAIL/PASSWORD or PLAYWRIGHT_ADMIN_* not set")

    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, f"{base_url}/login.html?api={q}", timeout=20000)
    page.fill("#email", email)
    page.fill("#password", password)
    page.click("#submitBtn")
    page.wait_for_url("**/dashboard.html", timeout=45000)
    expect(page).to_have_url(re.compile(r"dashboard", re.I))


@pytest.mark.e2e
def test_login_page_invalid_shows_error(page: Page, base_url: str, api_base: str):
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, f"{base_url}/login.html?api={q}", timeout=20000)
    page.fill("#email", "not-a-real-user-xyz@example.com")
    page.fill("#password", "wrong-password-12345")
    page.click("#submitBtn")
    page.wait_for_timeout(1500)
    if is_rate_limited(page):
        page.wait_for_timeout(3000)
        page.click("#submitBtn")
    err = page.locator("#alertError.show, #alertError, .alert-error, .error-message").first
    expect(err).to_be_visible(timeout=10000)
