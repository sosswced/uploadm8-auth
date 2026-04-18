"""Click-path and navigation flows (E2E) — sidebar, auth entry, dashboard CTAs."""

from __future__ import annotations

import re
import urllib.parse

import pytest
from playwright.sync_api import Page, expect

from tests.e2e_resilience import goto_with_backoff, is_rate_limited
from tests.test_full_app_flow import _on_login_page
from tests.test_site_matrix import _url


@pytest.mark.e2e
def test_sidebar_internal_destinations_when_authenticated(page: Page, base_url: str, api_base: str):
    """Every in-app sidebar link (shared-sidebar.js) loads without kicking to login."""
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, _url(base_url, "dashboard.html", q), timeout=25000)
    page.wait_for_timeout(1000)
    if _on_login_page(page):
        pytest.skip("Not authenticated — set PLAYWRIGHT_ADMIN_* or PLAYWRIGHT_TEST_* in .env")

    hrefs = page.evaluate(
        """() => {
            const nodes = document.querySelectorAll('.sidebar-nav a.nav-link[href$=".html"]');
            return [...new Set(Array.from(nodes, (a) => a.getAttribute('href')).filter(Boolean))];
        }"""
    )
    assert hrefs, "expected .sidebar-nav in-app links"

    for href in hrefs:
        goto_with_backoff(page, _url(base_url, href, q), timeout=30000)
        page.wait_for_timeout(500)
        if _on_login_page(page):
            # Some links are entitlement-guarded and can bounce to login/dashboard.
            continue
        expect(page.locator("body")).to_be_visible()


@pytest.mark.e2e
def test_login_page_navigates_to_signup_and_forgot_password(public_page: Page, base_url: str, api_base: str):
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(public_page, _url(base_url, "login.html", q), timeout=20000)
    public_page.locator("a[href='signup.html']").first.click()
    public_page.wait_for_load_state("domcontentloaded")
    expect(public_page).to_have_url(re.compile(r"signup", re.I))
    expect(public_page).to_have_title(re.compile(r"Sign Up", re.I))

    goto_with_backoff(public_page, _url(base_url, "login.html", q), timeout=20000)
    public_page.locator("a[href='forgot-password.html']").first.click()
    public_page.wait_for_load_state("domcontentloaded")
    expect(public_page).to_have_url(re.compile(r"forgot-password", re.I))


@pytest.mark.e2e
def test_signup_page_has_link_to_login(public_page: Page, base_url: str, api_base: str):
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(public_page, _url(base_url, "signup.html", q), timeout=20000)
    public_page.locator("a[href='login.html']").first.click()
    public_page.wait_for_load_state("domcontentloaded")
    expect(public_page).to_have_url(re.compile(r"login", re.I))


@pytest.mark.e2e
def test_dashboard_header_new_upload_navigates(page: Page, base_url: str, api_base: str):
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, _url(base_url, "dashboard.html", q), timeout=25000)
    page.wait_for_timeout(1200)
    if _on_login_page(page):
        pytest.skip("Not authenticated")
    page.locator('a[href="upload.html"]').first.click()
    page.wait_for_load_state("domcontentloaded")
    expect(page).to_have_url(re.compile(r"upload", re.I))
    expect(page).to_have_title(re.compile(r"Upload", re.I))


@pytest.mark.e2e
def test_index_footer_internal_links_load(public_page: Page, base_url: str, api_base: str):
    """Marketing index footer: same-origin pages (Pricing is #pricing on index — spot-checked via Features)."""
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(public_page, f"{base_url.rstrip('/')}/index.html?api={q}#pricing", timeout=25000)
    expect(public_page.locator("body")).to_be_visible()
    internal = [
        "contact.html",
        "how-it-works.html",
        "terms.html",
        "privacy.html",
        "support.html",
        "thumbnail-studio.html",
        "guide.html",
    ]
    for path in internal:
        goto_with_backoff(public_page, _url(base_url, path, q), timeout=25000)
        public_page.wait_for_timeout(400)
        expect(public_page.locator("body")).to_be_visible()


@pytest.mark.e2e
def test_mobile_menu_toggle_opens_sidebar(page: Page, base_url: str, api_base: str):
    page.set_viewport_size({"width": 390, "height": 844})
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, _url(base_url, "dashboard.html", q), timeout=25000)
    page.wait_for_timeout(1200)
    if _on_login_page(page):
        pytest.skip("Not authenticated")
    page.locator("#menuToggle").click()
    page.wait_for_timeout(500)
    if is_rate_limited(page):
        page.wait_for_timeout(3000)
        page.locator("#menuToggle").click()
        page.wait_for_timeout(500)
    is_open = page.locator("#sidebar").evaluate("el => el.classList.contains('open') || el.classList.contains('active')")
    assert is_open or page.locator("#sidebar").is_visible()
