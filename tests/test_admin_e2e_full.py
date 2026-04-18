"""
Full admin integration E2E: real API side effects (announcements, wallet adjust, marketing AI, calculator).

Requires master_admin or admin with API access. Set in .env (never commit secrets):

  PLAYWRIGHT_ADMIN_EMAIL=...
  PLAYWRIGHT_ADMIN_PASSWORD=...

Enable this suite only when you accept production cost / data:

  PLAYWRIGHT_ADMIN_E2E_FULL=1

Run: python run_tests.py admin-full
"""

from __future__ import annotations

import os
import re
import time
import urllib.parse

import pytest
from playwright.sync_api import Page, expect

from tests.test_full_app_flow import _on_login_page
from tests.test_admin_e2e import (  # noqa: I001
    _goto,
    _skip_if_access_denied,
    _skip_if_login,
    _wait_any_visible,
)

pytestmark = [pytest.mark.e2e, pytest.mark.admin, pytest.mark.admin_full]


def _admin_email() -> str:
    return (
        os.environ.get("PLAYWRIGHT_ADMIN_EMAIL")
        or os.environ.get("PLAYWRIGHT_TEST_EMAIL")
        or ""
    ).strip()


def _full_enabled() -> bool:
    return os.environ.get("PLAYWRIGHT_ADMIN_E2E_FULL", "").strip().lower() in ("1", "true", "yes")


@pytest.fixture(autouse=True)
def _require_full_flag():
    if not _full_enabled():
        pytest.skip("Set PLAYWRIGHT_ADMIN_E2E_FULL=1 to run destructive admin integration tests")


def test_full_send_announcement_email_to_self(page: Page, base_url: str) -> None:
    """Specific audience = logged-in admin only; email channel only."""
    email = _admin_email()
    if not email or "@" not in email:
        pytest.skip("Set PLAYWRIGHT_ADMIN_EMAIL to your admin account email")

    _goto(page, base_url, "admin.html")
    _skip_if_login(page)
    _skip_if_access_denied(page)
    _wait_any_visible(page, ["#adminContent"], timeout_s=120.0)
    page.wait_for_timeout(3000)

    page.locator(".action-card.announce").first.click()
    expect(page.locator("#announcementModal.active")).to_be_visible(timeout=15000)

    page.locator('#announcementModal input[name="audience"][value="specific"]').click()
    expect(page.locator("#userSearchContainer.active")).to_be_visible(timeout=5000)

    modal = page.locator("#announcementModal")
    modal.locator("#userSearchInput").fill(email)
    page.wait_for_timeout(1200)
    first = modal.locator("#userSearchResults .user-search-result").first
    expect(first).to_be_visible(timeout=15000)
    first.click()

    ts = str(int(time.time()))
    modal.locator("#announcementTitle").fill(f"[E2E] Automated test {ts}")
    modal.locator("#announcementBody").fill(
        "Playwright admin-full suite. Safe to ignore. Do not reply."
    )
    modal.locator("#channelEmail").check()
    modal.locator("#channelDiscordCommunity").uncheck()
    modal.locator("#channelDiscordWebhooks").uncheck()

    with page.expect_response(
        lambda r: r.request.method == "POST" and "/api/admin/announcements" in r.url,
        timeout=120000,
    ) as resp_info:
        modal.locator("button.btn-send").click()
    resp = resp_info.value
    assert resp.ok, f"announcement HTTP {resp.status}"
    page.wait_for_timeout(2000)
    expect(page.locator("#announcementModal.active")).not_to_be_visible(timeout=120000)


def test_full_admin_kpi_refresh_edit_mode(page: Page, base_url: str) -> None:
    _goto(page, base_url, "admin-kpi.html")
    _skip_if_login(page)
    _skip_if_access_denied(page)
    _wait_any_visible(page, ["#adminContent"], timeout_s=90.0)

    page.locator("#kpiRefreshBtn").click()
    page.wait_for_timeout(2500)
    page.locator("#editModeBtn").click()
    page.wait_for_timeout(400)
    page.locator("#editModeBtn").click()
    page.wait_for_timeout(400)
    page.locator("#timeRange").select_option(index=0)
    page.wait_for_timeout(500)
    _scroll(page)


def test_full_calculator_presets_and_run(page: Page, base_url: str) -> None:
    _goto(page, base_url, "admin-calculator.html")
    _skip_if_login(page)
    _skip_if_access_denied(page)
    _wait_any_visible(page, ["#adminContent"], timeout_s=90.0)

    page.locator("button:has-text(\"Load Live Data\")").click()
    page.wait_for_timeout(4000)
    page.locator("#preset-growth").click()
    page.wait_for_timeout(500)

    page.locator("button.btn-calculate").filter(has_text=re.compile(r"Calculate", re.I)).first.click()
    page.wait_for_timeout(2000)

    for tab in ("Dashboard", "Capacity", "Health Score", "Enterprise Quote"):
        btn = page.locator(".calc-tab").filter(has_text=re.compile(re.escape(tab), re.I))
        if btn.count():
            btn.first.click()
            page.wait_for_timeout(400)
    page.locator(".calc-tab").filter(has_text=re.compile(r"Inputs", re.I)).first.click()
    page.wait_for_timeout(300)
    _scroll(page)


def test_full_marketing_ai_generate_and_deploy(page: Page, base_url: str) -> None:
    _goto(page, base_url, "admin-marketing.html")
    _skip_if_login(page)
    _skip_if_access_denied(page)
    _wait_any_visible(page, ["#content"], timeout_s=90.0)

    page.locator("#refreshBtn").click()
    page.wait_for_timeout(2000)

    page.locator("#aiGenerateBtn").click()
    expect(page.locator("#aiStatus")).to_contain_text(
        re.compile(r"Generated|deterministic|failed|Generation", re.I),
        timeout=180000,
    )
    out = page.locator("#aiOutput")
    assert len(out.inner_text().strip()) > 10

    page.locator("#aiApplyBtn").click()
    page.wait_for_timeout(800)

    page.locator("#aiDeployBtn").click()
    expect(page.locator("#aiStatus")).to_contain_text(
        re.compile(r"deploy|blocked|guardrail|failed", re.I),
        timeout=180000,
    )


def test_full_wallet_adjust_add_put_one(page: Page, base_url: str) -> None:
    email = _admin_email()
    if not email:
        pytest.skip("PLAYWRIGHT_ADMIN_EMAIL not set")

    _goto(page, base_url, "admin-wallet.html")
    _skip_if_login(page)
    _skip_if_access_denied(page)
    _wait_any_visible(page, ["#mainContent"], timeout_s=90.0)

    page.locator("#userSearchInput").fill(email)
    page.wait_for_timeout(1200)
    expect(page.locator(".search-result-item").first).to_be_visible(timeout=20000)
    page.locator(".search-result-item").first.click()
    page.wait_for_timeout(4000)

    expect(page.locator("#selectedUserCard")).to_be_visible(timeout=15000)
    page.locator("#walletBtnPut").click()
    page.locator("#modeAdd").click()
    page.locator("#adjustAmount").fill("1")
    page.locator(".reason-shortcut[data-reason='QA / testing reset']").click()

    with page.expect_response(
        lambda r: r.request.method == "POST" and "/wallet/adjust" in r.url,
        timeout=60000,
    ) as resp_info:
        page.locator("#adjustSubmitBtn").click()
    assert resp_info.value.ok, f"wallet adjust {resp_info.value.status}"

    expect(page.locator(".adjust-result.ok")).to_be_visible(timeout=30000)


def _scroll(page: Page) -> None:
    for _ in range(8):
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(80)
