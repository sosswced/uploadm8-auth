"""
Admin / master_admin UI E2E: every admin route, scroll, and safe interactions.

Credentials: set PLAYWRIGHT_ADMIN_EMAIL + PLAYWRIGHT_ADMIN_PASSWORD (or PLAYWRIGHT_TEST_*)
for an account with admin or master_admin role. Never commit passwords.

Run: PLAYWRIGHT_RUN_ADMIN_E2E=1 python run_tests.py admin
"""

from __future__ import annotations

import os
import re
import time
import urllib.parse

import pytest
from playwright.sync_api import Page, expect

from tests.test_full_app_flow import _on_login_page

pytestmark = [pytest.mark.e2e, pytest.mark.admin]


def _api_q() -> str:
    return urllib.parse.quote(
        os.environ.get("PLAYWRIGHT_API_BASE", "http://127.0.0.1:8000"),
        safe="",
    )


def _goto(page: Page, base_url: str, path: str) -> None:
    q = _api_q()
    p = path.lstrip("/")
    page.goto(f"{base_url.rstrip('/')}/{p}?api={q}", wait_until="domcontentloaded", timeout=45000)


def _skip_if_login(page: Page) -> None:
    page.wait_for_timeout(600)
    if _on_login_page(page):
        pytest.skip("Not logged in — set PLAYWRIGHT_ADMIN_EMAIL/PASSWORD (or TEST_*) in .env")


def _skip_if_access_denied(page: Page) -> None:
    """Admin pages use #accessDenied or access-denied class when role is insufficient."""
    page.wait_for_timeout(800)
    for sel in ("#accessDenied", ".access-denied"):
        loc = page.locator(sel).first
        if loc.count() > 0:
            try:
                if loc.is_visible():
                    pytest.skip("Access Denied — use an admin or master_admin account in PLAYWRIGHT_ADMIN_*")
            except Exception:
                pass
    html = page.content().lower()
    if "access denied" in html and ("permission" in html or "don't have" in html):
        pytest.skip("Access Denied — use an admin or master_admin account")


def _wait_any_visible(page: Page, selectors: list[str], timeout_s: float = 60.0) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for sel in selectors:
            loc = page.locator(sel).first
            if loc.count() > 0:
                try:
                    if loc.is_visible(timeout=500):
                        return sel
                except Exception:
                    pass
        page.wait_for_timeout(400)
    pytest.fail(f"Expected one of admin shells to appear: {selectors}")


def _scroll(page: Page) -> None:
    for _ in range(10):
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(100)
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(200)


# (path, substring in <title>, css selectors for main admin shell when authorized)
ADMIN_PAGES: list[tuple[str, str, list[str]]] = [
    ("admin.html", "Admin Dashboard", ["#adminContent"]),
    ("admin-kpi.html", "Admin KPIs", ["#adminContent"]),
    ("admin-marketing.html", "Marketing Ops", ["#content"]),
    ("admin-calculator.html", "Business Calculator", ["#adminContent"]),
    ("admin-wallet.html", "Wallet Manager", ["#mainContent"]),
    ("admin-data-integrity.html", "Data Integrity", ["#content"]),
    ("account-management.html", "Account Management", ["#adminContent"]),
]


@pytest.mark.parametrize("path,title_part,content_selectors", ADMIN_PAGES)
def test_admin_page_loads_title_and_main_shell(
    page: Page, base_url: str, path: str, title_part: str, content_selectors: list[str]
) -> None:
    _goto(page, base_url, path)
    _skip_if_login(page)
    expect(page).to_have_title(re.compile(re.escape(title_part), re.I))
    _skip_if_access_denied(page)
    _wait_any_visible(page, content_selectors, timeout_s=90.0)
    _scroll(page)


def test_admin_users_redirects_to_account_management(page: Page, base_url: str) -> None:
    _goto(page, base_url, "admin-users.html")
    _skip_if_login(page)
    page.wait_for_url(re.compile(r"account-management", re.I), timeout=20000)


def test_admin_dashboard_nav_and_action_cards(page: Page, base_url: str) -> None:
    _goto(page, base_url, "admin.html")
    _skip_if_login(page)
    _skip_if_access_denied(page)
    _wait_any_visible(page, ["#adminContent"], timeout_s=90.0)

    n_nav = page.locator(".admin-nav a.admin-nav-btn").count()
    for i in range(n_nav):
        _goto(page, base_url, "admin.html")
        _wait_any_visible(page, ["#adminContent"], timeout_s=60.0)
        link = page.locator(".admin-nav a.admin-nav-btn").nth(i)
        href = (link.get_attribute("href") or "").strip()
        if not href.endswith(".html"):
            continue
        link.click()
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(600)
        _skip_if_access_denied(page)
        expect(page.locator("body")).to_be_visible()

    # In-app action cards only (skip Stripe onclick div)
    for _ in range(5):
        _goto(page, base_url, "admin.html")
        _wait_any_visible(page, ["#adminContent"], timeout_s=60.0)
        ac = page.locator("a.action-card[href$='.html']")
        if ac.count() == 0:
            break
        ac.first.click()
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(600)
        _skip_if_access_denied(page)
        expect(page.locator("body")).to_be_visible()


def test_admin_dashboard_announcement_modal_open_close(page: Page, base_url: str) -> None:
    _goto(page, base_url, "admin.html")
    _skip_if_login(page)
    _skip_if_access_denied(page)
    _wait_any_visible(page, ["#adminContent"], timeout_s=90.0)

    page.locator(".action-card.announce").first.click()
    expect(page.locator("#announcementModal.active")).to_be_visible(timeout=10000)
    page.locator("#announcementModal .modal-close").first.click()
    page.wait_for_timeout(400)


def test_admin_kpi_refresh_and_time_range(page: Page, base_url: str) -> None:
    _goto(page, base_url, "admin-kpi.html")
    _skip_if_login(page)
    _skip_if_access_denied(page)
    _wait_any_visible(page, ["#adminContent"], timeout_s=90.0)

    page.locator("#kpiRefreshBtn").click()
    page.wait_for_timeout(1500)
    page.select_option("#timeRange", index=min(2, page.locator("#timeRange option").count() - 1))
    page.wait_for_timeout(500)
    _scroll(page)


def test_admin_marketing_refresh_search_scroll(page: Page, base_url: str) -> None:
    _goto(page, base_url, "admin-marketing.html")
    _skip_if_login(page)
    _skip_if_access_denied(page)
    _wait_any_visible(page, ["#content"], timeout_s=90.0)

    page.locator("#refreshBtn").click()
    page.wait_for_timeout(1200)
    page.locator("#searchInput").fill("test")
    page.wait_for_timeout(400)
    _scroll(page)


def test_admin_wallet_search_focus_and_tabs(page: Page, base_url: str) -> None:
    _goto(page, base_url, "admin-wallet.html")
    _skip_if_login(page)
    _skip_if_access_denied(page)
    _wait_any_visible(page, ["#mainContent"], timeout_s=90.0)

    page.locator("#userSearchInput").click()
    page.locator("#userSearchInput").fill("a")
    page.wait_for_timeout(400)
    for wid in ("walletBtnPut", "walletBtnAic"):
        if page.locator(f"#{wid}").count():
            page.locator(f"#{wid}").click()
            page.wait_for_timeout(200)
    _scroll(page)


def test_admin_data_integrity_filters_visible(page: Page, base_url: str) -> None:
    _goto(page, base_url, "admin-data-integrity.html?severity=ERROR&hours=72")
    _skip_if_login(page)
    _skip_if_access_denied(page)
    _wait_any_visible(page, ["#content"], timeout_s=90.0)
    _scroll(page)


def test_account_management_tabs_users_audit_analytics(page: Page, base_url: str) -> None:
    _goto(page, base_url, "account-management.html")
    _skip_if_login(page)
    _skip_if_access_denied(page)
    _wait_any_visible(page, ["#adminContent"], timeout_s=90.0)

    for name in ("users", "audit", "analytics"):
        btn = page.locator(f'.tab-btn[data-tab="{name}"]')
        if btn.count():
            btn.click()
            page.wait_for_timeout(800)
            expect(page.locator("body")).to_be_visible()
    _scroll(page)


def test_admin_calculator_interaction_safe(page: Page, base_url: str) -> None:
    _goto(page, base_url, "admin-calculator.html")
    _skip_if_login(page)
    _skip_if_access_denied(page)
    _wait_any_visible(page, ["#adminContent"], timeout_s=90.0)
    _scroll(page)
