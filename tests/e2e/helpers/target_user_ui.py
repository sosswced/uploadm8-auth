"""Playwright flows for admin features on the canonical target user."""

from __future__ import annotations

from typing import Any

from playwright.sync_api import Page, expect

from tests.e2e.helpers.browser_session import (
    ensure_authed,
    human_login_via_form,
    navigate_to_page_human,
    wait_for_authenticated_shell,
)
from tests.e2e.helpers.config import e2e_target_user_id, e2e_target_user_name, e2e_target_user_search_terms
from tests.e2e.helpers.human_pace import click_delay_ms, pause_between_requests
from tests.e2e.helpers.ui_safe_clicks import click_admin_settings_surfaces


def _search_term() -> str:
    terms = e2e_target_user_search_terms()
    return terms[0] if terms else e2e_target_user_name()


def _ensure_session_fresh(page: Page, base_url: str) -> None:
    """Re-auth after a long page tour so account-mgmt / wallet are not on a stale tab."""
    ensure_authed(page, base_url)
    if "login.html" in page.url:
        human_login_via_form(page, base_url)
    navigate_to_page_human(page, base_url, "dashboard.html")
    wait_for_authenticated_shell(page)
    pause_between_requests()


def _wait_users_table_ready(page: Page) -> None:
    page.wait_for_function(
        """() => {
            const body = document.getElementById('usersTableBody');
            if (!body) return false;
            const text = body.textContent || '';
            if (text.includes('Loading users')) return false;
            const users = window.AccountMgmt && Array.isArray(window.AccountMgmt.users)
                ? window.AccountMgmt.users.length : 0;
            return users > 0 || body.querySelectorAll('tr td').length > 0;
        }""",
        timeout=120_000,
    )


def exercise_account_management_target(page: Page, base_url: str) -> dict[str, Any]:
    """Users list → search → edit modal → audit/analytics tabs (read-only)."""
    uid = e2e_target_user_id()
    name = e2e_target_user_name()
    out: dict[str, Any] = {"user_id": uid, "steps": []}

    navigate_to_page_human(page, base_url, "account-management.html")
    wait_for_authenticated_shell(page)
    _wait_users_table_ready(page)
    refresh = page.locator("#refreshBtn")
    if refresh.count():
        refresh.click()
        page.wait_for_timeout(800)

    search = page.locator("#acctMgmtSearchInput")
    name_key = name.split()[0] if name else ""
    matched_term: str | None = None
    row = page.locator(f"#usersTableBody tr:has-text('{name_key}')").first
    for term in e2e_target_user_search_terms():
        search.fill("")
        page.wait_for_timeout(200)
        search.fill(term)
        page.wait_for_timeout(500)
        candidate = page.locator(f"#usersTableBody tr:has-text('{name_key}')").first
        if candidate.count() > 0:
            try:
                candidate.wait_for(state="visible", timeout=5_000)
                row = candidate
                matched_term = term
                break
            except Exception:
                continue
    expect(row).to_be_visible(timeout=60_000)
    out["steps"].append(f"search={matched_term or _search_term()!r}")
    if row.count() == 0:
        row = page.locator(f"#usersTableBody tr:has-text('{uid[:8]}')").first
    expect(row).to_be_visible(timeout=30_000)
    row.locator('button[title="Edit"]').click()
    page.wait_for_timeout(click_delay_ms())

    modal = page.locator("#editModal.active")
    expect(modal).to_be_visible(timeout=15_000)
    modal_text = modal.inner_text()
    assert uid[:8] in modal_text or name.split()[0] in modal_text, "Edit modal missing target user"
    out["steps"].append("edit_modal_open")

    page.locator('[data-um8-fn="um8AccountMgmtCloseModal"]').first.click()
    page.wait_for_timeout(click_delay_ms())

    for tab_name, tab_sel in (
        ("audit", 'button.tab-btn[data-tab="audit"]'),
        ("analytics", 'button.tab-btn[data-tab="analytics"]'),
    ):
        page.locator(tab_sel).click()
        page.wait_for_timeout(click_delay_ms())
        out["steps"].append(f"tab_{tab_name}")
        if tab_name == "audit":
            page.locator("#auditRefreshBtn").click()
            page.wait_for_timeout(1000)
            out["steps"].append("audit_refresh")

    page.locator('button.tab-btn[data-tab="users"]').click()
    page.wait_for_timeout(click_delay_ms())
    click_admin_settings_surfaces(page, max_clicks=8)
    out["steps"].append("users_tab_toolbar")
    return out


def _wait_wallet_page_ready(page: Page) -> None:
    page.wait_for_function(
        "() => document.getElementById('mainContent') && getComputedStyle(document.getElementById('mainContent')).display !== 'none'",
        timeout=90_000,
    )


def _wallet_search_and_select(page: Page) -> None:
    inp = page.locator("#userSearchInput")
    inp.fill(_search_term())
    result = page.locator(".search-result-item").first
    expect(result).to_be_visible(timeout=30_000)
    result.click()
    page.wait_for_timeout(click_delay_ms())


def exercise_admin_wallet_target(page: Page, base_url: str) -> dict[str, Any]:
    """Wallet admin: search target user → load PUT/AIC balances + ledger."""
    uid = e2e_target_user_id()
    name = e2e_target_user_name()
    out: dict[str, Any] = {"user_id": uid, "steps": []}

    navigate_to_page_human(page, base_url, "admin-wallet.html")
    wait_for_authenticated_shell(page)
    _wait_wallet_page_ready(page)

    _wallet_search_and_select(page)
    out["steps"].append("wallet_search")

    expect(page.locator("#selectedUserCard")).to_be_visible(timeout=15_000)
    sel_text = page.locator("#selectedUserCard").inner_text()
    assert name.split()[0] in sel_text or uid[:8] in sel_text, "Wallet selection mismatch"
    out["steps"].append("wallet_selected")

    expect(page.locator("#putStatVal")).not_to_have_text("—", timeout=30_000)
    expect(page.locator("#aicStatVal")).not_to_have_text("—", timeout=30_000)
    out["steps"].append("wallet_balances_loaded")

    page.locator("#walletBtnPut").click()
    page.wait_for_timeout(300)
    page.locator("#walletBtnAic").click()
    page.wait_for_timeout(300)
    out["steps"].append("wallet_put_aic_toggle")

    for tab_href in (
        "admin-billing-catalog.html",
        "admin-stripe-catalog.html",
        "admin-billing-weights.html",
    ):
        link = page.locator(f'a.um8-billing-tab[href="{tab_href}"]')
        if link.count() and link.first.is_visible():
            link.first.click()
            page.wait_for_load_state("domcontentloaded")
            pause_between_requests()
            wait_for_authenticated_shell(page)
            click_admin_settings_surfaces(page, max_clicks=6)
            out["steps"].append(f"billing_tab_{tab_href}")
            navigate_to_page_human(page, base_url, "admin-wallet.html")
            wait_for_authenticated_shell(page)
            _wait_wallet_page_ready(page)
            _wallet_search_and_select(page)

    return out


def exercise_target_user_admin_ui(page: Page, base_url: str) -> dict[str, Any]:
    """Full read-only admin UI pass for the target user."""
    _ensure_session_fresh(page, base_url)
    report: dict[str, Any] = {"account_management": None, "wallet": None, "ok": False}
    report["account_management"] = exercise_account_management_target(page, base_url)
    report["wallet"] = exercise_admin_wallet_target(page, base_url)
    report["ok"] = True
    return report
