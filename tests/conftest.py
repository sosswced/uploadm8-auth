"""
Playwright + pytest fixtures for UploadM8 E2E tests.

The `frontend/` directory lives in this repo. The API (`app.py`) is a slim FastAPI app;
static HTML is often served separately (e.g. another port or CDN). Default E2E URLs
assume API on port 8000; set PLAYWRIGHT_BASE_URL if the UI is elsewhere.

Optional: serve `frontend/` separately (e.g. python -m http.server 8080) and set
PLAYWRIGHT_BASE_URL=http://127.0.0.1:8080 (PLAYWRIGHT_API_BASE stays 8000).

Requires (for authenticated tests):
  - PLAYWRIGHT_TEST_EMAIL / PLAYWRIGHT_TEST_PASSWORD, or PLAYWRIGHT_ADMIN_*
"""
from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path

import pytest


def _playwright_server_reachable(base_url: str) -> bool:
    base = (base_url or "").rstrip("/")
    if not base:
        return False
    for path in ("/health", "/login.html"):
        try:
            urllib.request.urlopen(f"{base}{path}", timeout=2.5)
            return True
        except Exception:
            continue
    return False


def pytest_collection_modifyitems(config, items) -> None:
    """Skip @pytest.mark.e2e tests when no server is listening (local pytest without uvicorn)."""
    if os.environ.get("PLAYWRIGHT_REQUIRE_SERVER", "").strip().lower() in ("1", "true", "yes"):
        return
    base = os.environ.get("PLAYWRIGHT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    if _playwright_server_reachable(base):
        return
    skip_unavailable = pytest.mark.skip(
        reason=(
            f"No HTTP server at {base} (tried /health and /login.html). "
            f"Start: python -m uvicorn app:app --host 127.0.0.1 --port 8000  "
            f"or set PLAYWRIGHT_REQUIRE_SERVER=1 to fail instead of skip."
        )
    )
    for item in items:
        if item.get_closest_marker("e2e"):
            item.add_marker(skip_unavailable)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Heavy 'live' E2E journeys run only when PLAYWRIGHT_RUN_LIVE=1 (or via run_tests.py live)."""
    if item.get_closest_marker("live"):
        if os.environ.get("PLAYWRIGHT_RUN_LIVE", "").strip().lower() not in ("1", "true", "yes"):
            pytest.skip(
                "Live journey tests: set PLAYWRIGHT_RUN_LIVE=1 or run: python run_tests.py live"
            )
    if item.get_closest_marker("admin"):
        if os.environ.get("PLAYWRIGHT_RUN_ADMIN_E2E", "").strip().lower() not in ("1", "true", "yes"):
            pytest.skip(
                "Admin E2E tests: set PLAYWRIGHT_RUN_ADMIN_E2E=1 or run: python run_tests.py admin"
            )
    if item.get_closest_marker("exhaustive"):
        if os.environ.get("PLAYWRIGHT_RUN_EXHAUSTIVE", "").strip().lower() not in ("1", "true", "yes"):
            pytest.skip(
                "Exhaustive UI sweep: set PLAYWRIGHT_RUN_EXHAUSTIVE=1 or run: python run_tests.py exhaustive"
            )

# Project root (parent of tests/)
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.environ.get("PLAYWRIGHT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


@pytest.fixture(scope="session")
def api_base() -> str:
    return os.environ.get("PLAYWRIGHT_API_BASE", "http://127.0.0.1:8000").rstrip("/")


@pytest.fixture(scope="session")
def playwright_instance():
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    yield pw
    pw.stop()


@pytest.fixture(scope="session")
def browser(playwright_instance):
    headless = os.environ.get("PLAYWRIGHT_HEADLESS", "true").lower() in ("true", "1", "yes")
    browser = playwright_instance.chromium.launch(headless=headless)
    yield browser
    browser.close()


@pytest.fixture(scope="session")
def browser_context(browser, base_url: str, api_base: str):
    """
    Single browser context; logs in once if credentials are set and login succeeds.
    """
    context = browser.new_context(
        viewport={"width": 1280, "height": 900},
        ignore_https_errors=True,
    )
    # When running admin E2E, prefer PLAYWRIGHT_ADMIN_* so TEST_* can stay a non-admin user.
    _admin_run = os.environ.get("PLAYWRIGHT_RUN_ADMIN_E2E", "").strip().lower() in ("1", "true", "yes")
    _admin_full = os.environ.get("PLAYWRIGHT_ADMIN_E2E_FULL", "").strip().lower() in ("1", "true", "yes")
    if _admin_run or _admin_full:
        email = (os.environ.get("PLAYWRIGHT_ADMIN_EMAIL") or os.environ.get("PLAYWRIGHT_TEST_EMAIL") or "").strip()
        password = (os.environ.get("PLAYWRIGHT_ADMIN_PASSWORD") or os.environ.get("PLAYWRIGHT_TEST_PASSWORD") or "").strip()
    else:
        email = (os.environ.get("PLAYWRIGHT_TEST_EMAIL") or os.environ.get("PLAYWRIGHT_ADMIN_EMAIL") or "").strip()
        password = (os.environ.get("PLAYWRIGHT_TEST_PASSWORD") or os.environ.get("PLAYWRIGHT_ADMIN_PASSWORD") or "").strip()

    if email and password:
        page = context.new_page()
        try:
            q = urllib.parse.quote(api_base, safe="")
            page.goto(f"{base_url}/login.html?api={q}", wait_until="domcontentloaded", timeout=20000)
            page.fill("#email", email)
            page.fill("#password", password)
            page.click("#submitBtn")
            page.wait_for_url("**/dashboard.html", timeout=45000)
        except Exception:
            # Leave context unauthenticated; tests may skip
            pass
        finally:
            try:
                page.close()
            except Exception:
                pass

    yield context
    context.close()


@pytest.fixture
def page(browser_context):
    p = browser_context.new_page()
    yield p
    p.close()


@pytest.fixture
def public_page(browser):
    """
    Fresh unauthenticated page for public/auth routes.
    Avoids session redirects from the shared logged-in browser_context.
    """
    context = browser.new_context(
        viewport={"width": 1280, "height": 900},
        ignore_https_errors=True,
    )
    p = context.new_page()
    yield p
    p.close()
    context.close()


@pytest.fixture
def logged_in_page(browser_context, base_url: str, api_base: str):
    """Page that should already be authenticated if session login worked."""
    p = browser_context.new_page()
    q = urllib.parse.quote(api_base, safe="")
    p.goto(f"{base_url}/dashboard.html?api={q}", wait_until="domcontentloaded", timeout=20000)
    yield p
    p.close()


@pytest.fixture
def auth_storage_state(browser_context):
    """Returns True if context appears logged in (dashboard not login)."""
    return browser_context
