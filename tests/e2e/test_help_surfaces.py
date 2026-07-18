"""Smoke checks for Feature Guide, Settings help tabs, and bug report form."""

from __future__ import annotations

import pytest

from tests.e2e.helpers.browser_session import navigate_to_page_human, wait_for_authenticated_shell
from tests.e2e.helpers.pages import page_url

pytestmark = [pytest.mark.e2e, pytest.mark.ui_smoke]


def test_guide_public_and_playbook_hash(public_page, base_url: str):
    url = page_url(base_url, "guide.html#feat-settings-playbook")
    resp = public_page.goto(url, wait_until="domcontentloaded")
    assert resp is not None and resp.status < 500
    assert public_page.locator("#feat-settings-playbook").count() > 0
    assert "login.html" not in public_page.url


def test_report_bug_form_markup(public_page, base_url: str):
    url = page_url(base_url, "report-bug.html?upload_id=test-upload-id")
    public_page.goto(url, wait_until="domcontentloaded")
    assert "login.html" not in public_page.url
    assert public_page.locator("#bugForm, #bugMessage").count() >= 1
    assert public_page.locator("#bugAuthGate").count() >= 1


def test_settings_token_balances_tab(human_session_page, base_url: str):
    navigate_to_page_human(human_session_page, base_url, "settings.html#token-balances")
    wait_for_authenticated_shell(human_session_page)
    assert human_session_page.locator("#token-balances-panel").count() > 0
    assert human_session_page.locator("#settings-tab-token-balances.active").count() >= 1
    # Deep-link must load ledger (not leave an empty tbody from the early-hash race).
    human_session_page.wait_for_function(
        """() => {
            const tb = document.getElementById('tbLedgerTbody');
            if (!tb) return false;
            return tb.children.length > 0;
        }""",
        timeout=20000,
    )
    assert human_session_page.locator("#tbLedgerTbody tr").count() >= 1


def test_settings_guide_handbook_button(human_session_page, base_url: str):
    navigate_to_page_human(human_session_page, base_url, "settings.html#guide")
    wait_for_authenticated_shell(human_session_page)
    btn = human_session_page.locator("#settingsGuideHandbookBtn")
    assert btn.count() >= 1
    btn.click()
    iframe = human_session_page.locator("#settingsGuideIframe")
    human_session_page.wait_for_function(
        """() => {
            const el = document.getElementById('settingsGuideIframe');
            return !!(el && el.src && el.src.indexOf('feat-settings-playbook') !== -1);
        }""",
        timeout=10000,
    )
    assert "feat-settings-playbook" in (iframe.get_attribute("src") or "")
