"""Conservative in-page click sweeps (tabs, toggles) — skips destructive actions."""

from __future__ import annotations

import re

import os

from playwright.sync_api import Page

DANGEROUS_RE = re.compile(
    r"(logout|log\s*out|sign\s*out|sign\s*in\s*as|switch\s*user|different\s*user|"
    r"delete|remove|destroy|ban|unban|wipe|reset\s*password|update\s*email|save\s*changes|"
    r"cancel\s*subscription|charge|refund|transfer|purge|revoke|disconnect\s*all|"
    r"delete\s*account|submit\s*ticket|approve\s*campaign|reject\s*campaign|"
    r"deploy\s*ai|save\s*campaign|run\s*execution|generate\s*ai\s*plan|promotion\s*eval)",
    re.I,
)

_ADMIN_DESTRUCTIVE_IDS = frozenset(
    {
        "saveUserBtn",
        "adminResetPwBtn",
        "adminChangeEmailBtn",
        "adminVerifyEmailBtn",
        "aiGenerateBtn",
        "aiDeployBtn",
        "aiApplyBtn",
        "cmpSaveBtn",
        "cmpExecTickBtn",
        "mlPromoEvalBtn",
    }
)

# Settings tabs, billing admin nav, account-mgmt tabs, safe admin toolbars.
ADMIN_SURFACE_SELECTORS = (
    "a.settings-tab",
    ".settings-tabs [role=tab]",
    "a.um8-billing-tab",
    ".tabs-container .tab-btn",
    "a.action-card:not([data-um8-open-blank])",
    "button.toolbar-btn",
    "#refreshBtn",
    "#auditRefreshBtn",
    "#touchpointRefreshBtn",
    "#mlPriorsRefreshBtn",
    "#mlLeaderboardBtn",
    "#mlReportsBundleBtn",
    "#truthToggleRawBtn",
    "button.btn-mini",
    ".audit-page-btn",
    "#cmpBroadAudienceBtn",
    "#cmpPreviewBtn",
)

# Session-ending shell controls only — sidebar nav links and other buttons stay clickable.
_SHELL_AUTH_FN = frozenset({"logout", "um8SwitchUser", "um8Logout", "signOut"})
_SHELL_AUTH_IDS = frozenset({"sidebarLogoutBtn", "sidebarSwitchUserBtn"})

SAFE_ROLE_SELECTORS = (
    "nav.sidebar-nav a.nav-link",
    "#menuToggle",
    "[role=tab]",
    "[role=tablist] button",
    "button[data-tab]",
    "button[data-bs-toggle=tab]",
    ".tab-btn",
    ".nav-tabs button",
    ".nav-tabs a",
    "details summary",
    "button[aria-expanded]",
    ".accordion-header button",
    ".filter-chip",
    ".btn-ghost:not([type=submit])",
    "button.btn-secondary:not([type=submit])",
    ".pill-row button",
    ".segmented-control button",
    "[data-um8-fn]:not([data-um8-fn=logout]):not([data-um8-fn=um8SwitchUser])",
)


def _is_safe_label(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 80:
        return False
    return not DANGEROUS_RE.search(t)


def _is_destructive_admin_control(el) -> bool:
    try:
        el_id = (el.get_attribute("id") or "").strip()
        if el_id in _ADMIN_DESTRUCTIVE_IDS:
            return True
        if el.get_attribute("data-um8-open-blank") == "1":
            return True
        if el.evaluate(
            """(node) => !!(node && node.closest && node.closest('.toggle-switch, .toggle-item input'))"""
        ):
            return True
        return False
    except Exception:
        return True


def _is_session_ending_control(el) -> bool:
    """Only Sign out / Switch user — not other sidebar or page buttons."""
    try:
        fn = (el.get_attribute("data-um8-fn") or "").strip()
        if fn in _SHELL_AUTH_FN:
            return True
        el_id = (el.get_attribute("id") or "").strip()
        if el_id in _SHELL_AUTH_IDS:
            return True
        return False
    except Exception:
        return True


def collect_safe_click_targets(page: Page, *, max_per_page: int = 40) -> list[str]:
    """Return human-readable labels for elements we will try to click."""
    labels: list[str] = []
    for sel in SAFE_ROLE_SELECTORS:
        for el in page.locator(sel).all():
            if len(labels) >= max_per_page:
                return labels
            try:
                if not el.is_visible():
                    continue
                if _is_session_ending_control(el):
                    continue
                text = (el.inner_text(timeout=500) or el.get_attribute("aria-label") or "").strip()
                if not _is_safe_label(text):
                    continue
                key = f"{sel}|{text}"
                if key not in labels:
                    labels.append(key)
            except Exception:
                continue
    return labels


def _click_selector_group(
    page: Page,
    sel: str,
    *,
    max_clicks: int,
    clicked: list[str],
) -> None:
    loc = page.locator(sel)
    count = min(loc.count(), max_clicks - len(clicked))
    for i in range(count):
        if len(clicked) >= max_clicks:
            return
        el = loc.nth(i)
        try:
            if not el.is_visible():
                continue
            if _is_session_ending_control(el) or _is_destructive_admin_control(el):
                continue
            text = (el.inner_text(timeout=500) or el.get_attribute("aria-label") or "").strip()
            if not _is_safe_label(text):
                continue
            el.click(timeout=5000)
            delay = int(os.environ.get("E2E_CLICK_DELAY_MS", "280"))
            page.wait_for_timeout(delay)
            clicked.append(text or sel)
        except Exception:
            continue


def click_admin_settings_surfaces(page: Page, *, max_clicks: int = 20) -> list[str]:
    """Exercise admin/settings/user-mgmt tabs and safe toolbar controls (no logout/ban/deploy)."""
    clicked: list[str] = []
    for sel in ADMIN_SURFACE_SELECTORS:
        _click_selector_group(page, sel, max_clicks=max_clicks, clicked=clicked)
        if len(clicked) >= max_clicks:
            break
    return clicked


def click_safe_targets(page: Page, *, max_per_page: int = 40) -> list[str]:
    """Click safe UI controls; return list of clicked labels (for reporting)."""
    clicked: list[str] = []
    for sel in SAFE_ROLE_SELECTORS:
        _click_selector_group(page, sel, max_clicks=max_per_page, clicked=clicked)
        if len(clicked) >= max_per_page:
            break
    return clicked


def click_page_surfaces(page: Page, rel_path: str, *, max_clicks: int = 25) -> list[str]:
    """Sidebar + in-page controls; admin/settings pages also get tab + toolbar coverage."""
    from tests.e2e.helpers.pages import ADMIN_SETTINGS_PAGES

    clicked = click_safe_targets(page, max_per_page=max_clicks)
    leaf = rel_path.split("/")[-1]
    if leaf in ADMIN_SETTINGS_PAGES and len(clicked) < max_clicks:
        extra = click_admin_settings_surfaces(page, max_clicks=max(12, max_clicks - len(clicked)))
        clicked.extend(extra)
    return clicked
