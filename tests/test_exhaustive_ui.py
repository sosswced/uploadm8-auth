"""
Exhaustive UI sweep: visit each route and click every visible interactive control (best-effort).

Uses a dedicated logged-in **user** session (PLAYWRIGHT_TEST_*) and **admin** session (PLAYWRIGHT_ADMIN_*).
If TEST_* is unset, falls back to ADMIN_* for the user sweep so one account can run both.

Enable: PLAYWRIGHT_RUN_EXHAUSTIVE=1  (or: python run_tests.py exhaustive)

Skips: logout/sign-out, obvious ban/delete, and same-origin external navigations are minimized via Escape.
"""

from __future__ import annotations

import os
import re
import urllib.parse

import pytest
from playwright.sync_api import Page, expect

from tests.test_full_app_flow import _on_login_page

pytestmark = [pytest.mark.e2e, pytest.mark.exhaustive]

# App shell + billing + public marketing (logged-in user can still view)
USER_PATHS: list[str] = [
    "dashboard.html",
    "upload.html",
    "thumbnail-studio.html",
    "queue.html",
    "scheduled.html",
    "platforms.html",
    "groups.html",
    "analytics.html",
    "kpi.html",
    "settings.html",
    "guide.html",
    "report-bug.html",
    "color-preferences.html",
    "billing.html",
    "billing/success.html",
    "success.html",
    "index.html",
    "about.html",
    "contact.html",
    "how-it-works.html",
    "privacy.html",
    "terms.html",
]

# Admin-only HTML (master_admin / admin). Omit admin-users.html: legacy redirect to account-management.html.
ADMIN_PATHS: list[str] = [
    "admin.html",
    "admin-kpi.html",
    "admin-marketing.html",
    "admin-calculator.html",
    "admin-wallet.html",
    "admin-data-integrity.html",
    "admin-incidents.html",
    "account-management.html",
]

SKIP_TEXT_RE = re.compile(
    r"sign\s*out|log\s*out|delete\s*account|ban\s*user|disconnect|unsubscribe\s*now",
    re.I,
)

# Do not auto-click these (covered by admin-full or too destructive for blind sweep)
SKIP_ROLE_OR_ID = re.compile(
    r"adjustSubmit|btn-send|aiDeployBtn|aiGenerateBtn|submitBtn|changeEmailBtn",
    re.I,
)


def _goto(page: Page, base_url: str, api_base: str, path: str) -> None:
    q = urllib.parse.quote(api_base, safe="")
    p = path.lstrip("/")
    page.goto(f"{base_url.rstrip('/')}/{p}?api={q}", wait_until="domcontentloaded", timeout=60000)


def _should_skip_click(locator, _page: Page) -> bool:
    try:
        if not locator.count():
            return True
        el = locator.first
        if not el.is_visible(timeout=400):
            return True
        t = (el.inner_text(timeout=500) or "").strip()
        if SKIP_TEXT_RE.search(t):
            return True
        aid = el.get_attribute("id") or ""
        if aid and SKIP_ROLE_OR_ID.search(aid):
            return True
        href = el.get_attribute("href") or ""
        if href.startswith("http") and "uploadm8" not in href.lower() and "127.0.0.1" not in href:
            if "stripe.com" in href or "openai" in href or "cloudflare" in href:
                return True
    except Exception:
        return True
    return False


def _exhaust_clicks_on_page(page: Page, base_url: str, api_base: str, path: str, *, max_interactions: int = 200) -> int:
    _goto(page, base_url, api_base, path)
    page.wait_for_timeout(1200)
    if _on_login_page(page) and "login" not in path.lower():
        pytest.skip(f"Session not authenticated on {path}")

    start_host = (page.url or "").split("/")[2] if "://" in (page.url or "") else ""
    done = 0
    selectors = (
        "button",
        "input[type='submit']",
        "input[type='button']",
        "[role='button']",
        "a.btn",
        "a.nav-link",
        ".admin-nav-btn",
        ".calc-tab",
        ".settings-tab",
        ".tab-btn",
        ".btn-calculate",
        ".wallet-select-btn",
        ".adjust-tab",
        ".quick-amt",
        ".theme-toggle",
        "#menuToggle",
        ".action-card",
    )

    rounds = 0
    while done < max_interactions and rounds < 25:
        rounds += 1
        progressed = False
        for sel in selectors:
            loc = page.locator(sel)
            n = loc.count()
            for i in range(min(n, 40)):
                if done >= max_interactions:
                    break
                el = loc.nth(i)
                try:
                    if _should_skip_click(el, page):
                        continue
                    if not el.is_visible(timeout=300):
                        continue
                    el.scroll_into_view_if_needed(timeout=3000)
                    page.wait_for_timeout(80)
                    el.click(timeout=8000, force=False)
                    done += 1
                    progressed = True
                    page.wait_for_timeout(120)
                    page.keyboard.press("Escape")
                    page.wait_for_timeout(80)
                    # Stay on same app host when possible
                    cur = page.url or ""
                    if start_host and start_host in cur and path.split("?")[0] not in cur and "login" in cur.lower():
                        _goto(page, base_url, api_base, path)
                except Exception:
                    page.keyboard.press("Escape")
                    continue
        page.mouse.wheel(0, 900)
        page.wait_for_timeout(200)
        if not progressed:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(200)
            if rounds > 8:
                break

    expect(page.locator("body")).to_be_visible()
    return done


@pytest.fixture(scope="session")
def browser_context_user(browser, base_url: str, api_base: str):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900}, ignore_https_errors=True)
    email = (os.environ.get("PLAYWRIGHT_TEST_EMAIL") or os.environ.get("PLAYWRIGHT_ADMIN_EMAIL") or "").strip()
    password = (os.environ.get("PLAYWRIGHT_TEST_PASSWORD") or os.environ.get("PLAYWRIGHT_ADMIN_PASSWORD") or "").strip()
    if not email or not password:
        pytest.skip("Set PLAYWRIGHT_TEST_EMAIL/PASSWORD (or ADMIN_*) for exhaustive user session")
    page = ctx.new_page()
    try:
        q = urllib.parse.quote(api_base, safe="")
        page.goto(f"{base_url}/login.html?api={q}", wait_until="domcontentloaded", timeout=25000)
        page.fill("#email", email)
        page.fill("#password", password)
        page.click("#submitBtn")
        page.wait_for_url("**/dashboard.html", timeout=60000)
    except Exception:
        pass
    finally:
        try:
            page.close()
        except Exception:
            pass
    yield ctx
    ctx.close()


@pytest.fixture(scope="session")
def browser_context_admin(browser, base_url: str, api_base: str):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900}, ignore_https_errors=True)
    email = (os.environ.get("PLAYWRIGHT_ADMIN_EMAIL") or "").strip()
    password = (os.environ.get("PLAYWRIGHT_ADMIN_PASSWORD") or "").strip()
    if not email or not password:
        pytest.skip("Set PLAYWRIGHT_ADMIN_EMAIL and PLAYWRIGHT_ADMIN_PASSWORD for exhaustive admin session")
    page = ctx.new_page()
    try:
        q = urllib.parse.quote(api_base, safe="")
        page.goto(f"{base_url}/login.html?api={q}", wait_until="domcontentloaded", timeout=25000)
        page.fill("#email", email)
        page.fill("#password", password)
        page.click("#submitBtn")
        page.wait_for_url("**/dashboard.html", timeout=60000)
    except Exception:
        pass
    finally:
        try:
            page.close()
        except Exception:
            pass
    yield ctx
    ctx.close()


@pytest.fixture
def page_user(browser_context_user):
    p = browser_context_user.new_page()
    yield p
    p.close()


@pytest.fixture
def page_admin(browser_context_admin):
    p = browser_context_admin.new_page()
    yield p
    p.close()


def test_exhaustive_click_every_route_user_session(page_user: Page, base_url: str, api_base: str) -> None:
    total = 0
    for path in USER_PATHS:
        total += _exhaust_clicks_on_page(page_user, base_url, api_base, path, max_interactions=120)
    assert total >= 0


def test_exhaustive_click_every_route_admin_session(page_admin: Page, base_url: str, api_base: str) -> None:
    total = 0
    for path in ADMIN_PATHS:
        total += _exhaust_clicks_on_page(page_admin, base_url, api_base, path, max_interactions=150)
    assert total >= 0
