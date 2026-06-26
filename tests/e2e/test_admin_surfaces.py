"""Admin settings, user management tabs, and billing admin navigation."""

from __future__ import annotations

import pytest

from tests.e2e.helpers.browser_session import navigate_to_page_human, wait_for_authenticated_shell
from tests.e2e.helpers.pages import ADMIN_SETTINGS_PAGES
from tests.e2e.helpers.ui_safe_clicks import click_admin_settings_surfaces, click_page_surfaces

pytestmark = [pytest.mark.e2e, pytest.mark.ui_smoke, pytest.mark.overnight, pytest.mark.ui_clicks]


@pytest.mark.parametrize("rel_path", ADMIN_SETTINGS_PAGES)
def test_admin_settings_surfaces(human_session_page, base_url: str, rel_path: str):
    navigate_to_page_human(human_session_page, base_url, rel_path)
    wait_for_authenticated_shell(human_session_page)
    assert "login.html" not in human_session_page.url, f"Session lost on {rel_path}"

    clicked = click_page_surfaces(human_session_page, rel_path, max_clicks=22)
    api_fails = getattr(human_session_page, "_e2e_failed_requests", [])
    assert not api_fails, f"API 5xx on {rel_path}: {api_fails[:3]}"
    assert clicked, f"Expected at least one safe click on {rel_path}"


@pytest.mark.parametrize(
    "rel_path,tab_selector",
    [
        ("settings.html", "a.settings-tab"),
        ("account-management.html", ".tabs-container .tab-btn"),
        ("admin.html", "a.um8-billing-tab"),
        ("admin-billing-catalog.html", "a.um8-billing-tab"),
    ],
)
def test_admin_primary_tabs(human_session_page, base_url: str, rel_path: str, tab_selector: str):
    navigate_to_page_human(human_session_page, base_url, rel_path)
    wait_for_authenticated_shell(human_session_page)
    tabs = human_session_page.locator(tab_selector)
    assert tabs.count() > 0, f"No tabs matching {tab_selector} on {rel_path}"
    clicked = click_admin_settings_surfaces(human_session_page, max_clicks=12)
    assert clicked, f"Tab/toolbar sweep produced no clicks on {rel_path}"
