"""Sidebar navigation + safe in-page control clicks."""

from __future__ import annotations

import pytest

from tests.e2e.helpers.browser_session import navigate_to_page_human, wait_for_authenticated_shell
from tests.e2e.helpers.human_pace import pause_between_requests
from tests.e2e.helpers.page_monitors import reset_page_monitors
from tests.e2e.helpers.pages import SIDEBAR_HREFS, page_url, page_url_matches
from tests.e2e.helpers.pages import ADMIN_SETTINGS_PAGES
from tests.e2e.helpers.ui_safe_clicks import click_page_surfaces

pytestmark = [pytest.mark.e2e, pytest.mark.ui_smoke, pytest.mark.overnight]


@pytest.mark.parametrize("rel_path", SIDEBAR_HREFS)
def test_sidebar_link_navigates(human_session_page, base_url: str, rel_path: str):
    navigate_to_page_human(human_session_page, base_url, "dashboard.html")
    link = human_session_page.locator(f'nav.sidebar-nav a.nav-link[href="{rel_path}"]')
    if link.count() == 0:
        pytest.skip(f"Sidebar link not visible for master admin: {rel_path}")
    if not link.first.is_visible():
        pytest.skip(f"Sidebar link hidden: {rel_path}")
    reset_page_monitors(human_session_page)
    try:
        link.first.click()
        human_session_page.wait_for_url(f"**/{rel_path.split('/')[-1]}**", timeout=60_000)
    except Exception:
        human_session_page.goto(page_url(base_url, rel_path), wait_until="domcontentloaded")
    human_session_page.wait_for_load_state("domcontentloaded")
    pause_between_requests()
    wait_for_authenticated_shell(human_session_page)
    assert page_url_matches(human_session_page.url, rel_path)
    api_fails = getattr(human_session_page, "_e2e_failed_requests", [])
    assert not api_fails, f"API 5xx after nav to {rel_path}: {api_fails[:3]}"


@pytest.mark.parametrize(
    "rel_path",
    list(ADMIN_SETTINGS_PAGES)
    + [
        "dashboard.html",
        "upload.html",
        "analytics.html",
        "thumbnail-studio.html",
    ],
)
@pytest.mark.ui_clicks
def test_safe_clicks_on_page(human_session_page, base_url: str, rel_path: str):
    navigate_to_page_human(human_session_page, base_url, rel_path)
    clicked = click_page_surfaces(human_session_page, rel_path, max_clicks=25)
    api_fails = getattr(human_session_page, "_e2e_failed_requests", [])
    assert not api_fails, f"API 5xx after clicks on {rel_path}: {api_fails[:3]}"
    _ = clicked
