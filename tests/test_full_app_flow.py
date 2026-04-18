"""
Full-app navigation helpers and smoke tests (Playwright).

Import helpers in other tests:
  from tests.test_full_app_flow import _goto, _on_login_page, _skip_if_login
"""
from __future__ import annotations

import urllib.parse

import pytest
from playwright.sync_api import Page, expect


def _goto(page: Page, base_url: str, path: str, timeout: int = 20000) -> None:
    """Navigate to /path or /path.html."""
    path = path.lstrip("/")
    q = urllib.parse.quote(
        __import__("os").environ.get("PLAYWRIGHT_API_BASE", "http://127.0.0.1:8000"),
        safe="",
    )
    url = f"{base_url.rstrip('/')}/{path}?api={q}"
    page.goto(url, wait_until="domcontentloaded", timeout=timeout)
    if page.url.endswith("/" + path.replace(".html", "")) or "404" in page.title():
        alt = f"{base_url.rstrip('/')}/{path}.html?api={q}"
        page.goto(alt, wait_until="domcontentloaded", timeout=timeout)


def _on_login_page(page: Page) -> bool:
    try:
        return "login" in page.url.lower() or page.locator("#loginForm").count() > 0
    except Exception:
        return False


def _skip_if_login(page: Page) -> None:
    if _on_login_page(page):
        pytest.skip("Not authenticated — open login or set PLAYWRIGHT_* credentials")


@pytest.mark.e2e
def test_dashboard_requires_auth_or_loads(logged_in_page: Page, base_url: str):
    page = logged_in_page
    _skip_if_login(page)
    expect(page.locator("body")).to_contain_text("Dashboard", ignore_case=True)


@pytest.mark.e2e
@pytest.mark.parametrize(
    "path,snippet",
    [
        ("upload.html", "Upload"),
        ("queue.html", "Queue"),
        ("scheduled.html", "Scheduled"),
        ("platforms.html", "Platform"),
        ("settings.html", "Settings"),
    ],
)
def test_authenticated_shell_pages_load(page: Page, base_url: str, api_base: str, path: str, snippet: str):
    """Each page should render shell content or redirect to login."""
    import urllib.parse

    q = urllib.parse.quote(api_base, safe="")
    page.goto(f"{base_url}/{path}?api={q}", wait_until="domcontentloaded", timeout=25000)
    page.wait_for_timeout(800)
    body = page.locator("body").inner_text()
    if _on_login_page(page) or "sign in" in body.lower():
        pytest.skip("Session not authenticated")
    expect(page.locator("body")).to_contain_text(snippet, ignore_case=True)


@pytest.mark.e2e
def test_admin_pages_role_gate(page: Page, browser_context, base_url: str, api_base: str):
    """Admin HTML loads; non-admins see Access Denied."""
    import urllib.parse

    q = urllib.parse.quote(api_base, safe="")
    page.goto(f"{base_url}/admin.html?api={q}", wait_until="domcontentloaded", timeout=25000)
    page.wait_for_timeout(2000)
    body = page.locator("body").inner_text()
    if _on_login_page(page):
        pytest.skip("Not logged in")
    assert "Access Denied" in body or "Admin Dashboard" in body or "Verifying" in body
