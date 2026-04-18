"""
Smoke-test every HTML route: title + body visible + no hard navigation failure.

* public — must load without auth (marketing, legal, auth entry pages).
* shell — app pages with sidebar; skips if session is not logged in.
* admin — admin UIs; skips if not logged in; non-admins may see Access Denied (still a pass).
* admin-users.html — legacy path only; tested via redirect to account-management.html (see dedicated test).
"""

from __future__ import annotations

import re
import urllib.parse

import pytest
from playwright.sync_api import Page, expect

from tests.e2e_resilience import goto_with_backoff
from tests.test_full_app_flow import _on_login_page


def _url(base_url: str, path: str, api_q: str) -> str:
    path = path.lstrip("/")
    return f"{base_url.rstrip('/')}/{path}?api={api_q}"


def _title_re(fragment: str) -> re.Pattern[str]:
    return re.compile(re.escape(fragment), re.I)


# (path, substring that must appear in document title, category)
PUBLIC_PAGES: list[tuple[str, str]] = [
    ("index.html", "UploadM8"),
    ("about.html", "About"),
    ("blog.html", "Blog"),
    ("contact.html", "Contact"),
    ("how-it-works.html", "How It Works"),
    ("walkthrough.html", "How It Works"),
    ("privacy.html", "Privacy"),
    ("terms.html", "Terms"),
    ("cookies.html", "Cookie"),
    ("security.html", "Security"),
    ("dmca.html", "DMCA"),
    ("subprocessors.html", "Subprocessors"),
    ("refunds.html", "Refund"),
    ("support.html", "Support"),
    ("data-deletion.html", "Data Deletion"),
    ("login.html", "Sign In"),
    ("signup.html", "Sign Up"),
    ("forgot-password.html", "Reset Password"),
    ("reset-password.html", "Set New Password"),
    ("verify-email.html", "Verify Email"),
    ("check-email.html", "Check Your Email"),
    ("confirm-email.html", "Confirming Email"),
    ("unsubscribe.html", "Email Preferences"),
]

SHELL_PAGES: list[tuple[str, str]] = [
    ("dashboard.html", "Dashboard"),
    ("upload.html", "Upload"),
    ("thumbnail-studio.html", "Thumbnail Studio"),
    ("queue.html", "Queue"),
    ("scheduled.html", "Scheduled"),
    ("platforms.html", "Connected Accounts"),
    ("groups.html", "Account Groups"),
    ("analytics.html", "Analytics"),
    ("kpi.html", "Upload KPIs"),
    ("settings.html", "Settings"),
    ("guide.html", "Feature Guide"),
    ("color-preferences.html", "Color Preferences"),
    ("billing.html", "Billing"),
    ("billing/success.html", "Payment Confirmed"),
    ("success.html", "Activating"),
    ("account-management.html", "Account Management"),
    ("report-bug.html", "Report a bug"),
]

ADMIN_PAGES: list[tuple[str, str]] = [
    ("admin.html", "Admin Dashboard"),
    ("admin-kpi.html", "Admin KPIs"),
    ("admin-marketing.html", "Marketing Ops"),
    ("admin-calculator.html", "Business Calculator"),
    ("admin-wallet.html", "Wallet Manager"),
    ("admin-data-integrity.html", "Data Integrity"),
    ("admin-incidents.html", "Ops incidents"),
]


@pytest.mark.e2e
@pytest.mark.parametrize("path,title_part", PUBLIC_PAGES)
def test_public_page_loads_and_title(public_page: Page, base_url: str, api_base: str, path: str, title_part: str):
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(public_page, _url(base_url, path, q), timeout=30000)
    expect(public_page).to_have_title(_title_re(title_part))
    expect(public_page.locator("body")).to_be_visible()
    assert "not found" not in public_page.title().lower()


@pytest.mark.e2e
@pytest.mark.parametrize("path,title_part", SHELL_PAGES)
def test_shell_page_loads_when_authenticated(page: Page, base_url: str, api_base: str, path: str, title_part: str):
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, _url(base_url, path, q), timeout=30000)
    page.wait_for_timeout(600)
    if _on_login_page(page):
        pytest.skip("Not authenticated — set PLAYWRIGHT_ADMIN_* or PLAYWRIGHT_TEST_* in .env")
    expect(page).to_have_title(_title_re(title_part))
    expect(page.locator("body")).to_be_visible()


@pytest.mark.e2e
@pytest.mark.parametrize("path,title_part", ADMIN_PAGES)
def test_admin_page_loads_when_authenticated(page: Page, base_url: str, api_base: str, path: str, title_part: str):
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, _url(base_url, path, q), timeout=30000)
    page.wait_for_timeout(1200)
    if _on_login_page(page):
        pytest.skip("Not authenticated — set PLAYWRIGHT_ADMIN_* or PLAYWRIGHT_TEST_* in .env")
    expect(page).to_have_title(_title_re(title_part))
    expect(page.locator("body")).to_be_visible()


@pytest.mark.e2e
def test_admin_users_legacy_redirects_to_account_management(page: Page, base_url: str, api_base: str) -> None:
    """admin-users.html is a bookmark-compat redirect, not a standalone admin surface."""
    q = urllib.parse.quote(api_base, safe="")
    goto_with_backoff(page, _url(base_url, "admin-users.html", q), timeout=30000)
    page.wait_for_timeout(400)
    if _on_login_page(page):
        pytest.skip("Not authenticated — set PLAYWRIGHT_ADMIN_* or PLAYWRIGHT_TEST_* in .env")
    page.wait_for_url(re.compile(r"account-management", re.I), timeout=20000)
    expect(page.locator("body")).to_be_visible()
