"""Playwright console/network monitors for E2E page loads."""

from __future__ import annotations


def reset_page_monitors(page) -> None:
    """Drop accumulated console/network noise between navigations."""
    errs = getattr(page, "_e2e_console_errors", None)
    if isinstance(errs, list):
        errs.clear()
    fails = getattr(page, "_e2e_failed_requests", None)
    if isinstance(fails, list):
        fails.clear()


def settle_page_monitors(page, *, wait_ms: int = 2000) -> None:
    """Clear stale errors, wait for in-flight fetches, then measure only new noise."""
    reset_page_monitors(page)
    page.wait_for_timeout(wait_ms)
