"""Load every static page — authenticated shell + public marketing."""

from __future__ import annotations

import pytest

from tests.e2e.helpers.browser_session import goto_page_resilient, navigate_to_page_human
from tests.e2e.helpers.human_pace import pause_between_requests
from tests.e2e.helpers.pages import AUTHENTICATED_PAGES, PUBLIC_PAGES, NO_APP_SHELL_PAGES, page_url

pytestmark = [pytest.mark.e2e, pytest.mark.ui_smoke, pytest.mark.overnight]


@pytest.mark.parametrize("rel_path", PUBLIC_PAGES)
def test_public_page_loads(public_page, base_url: str, rel_path: str):
    url = page_url(base_url, rel_path)
    goto_page_resilient(public_page, url)
    assert "UploadM8" in (public_page.title() or "")


@pytest.mark.parametrize("rel_path", AUTHENTICATED_PAGES)
def test_authenticated_page_loads(human_session_page, base_url: str, rel_path: str):
    navigate_to_page_human(human_session_page, base_url, rel_path)
    pause_between_requests()
    if rel_path in NO_APP_SHELL_PAGES:
        assert "login.html" not in human_session_page.url
        assert rel_path.split("/")[-1] in human_session_page.url or "UploadM8" in (human_session_page.title() or "")
        return
    errors = getattr(human_session_page, "_e2e_console_errors", [])
    api_fails = getattr(human_session_page, "_e2e_failed_requests", [])
    assert not api_fails, f"API 5xx on {rel_path}: {api_fails[:5]}"
    hard = [e for e in errors if "uncaught" in e.lower() or "typeerror" in e.lower()]
    assert not hard, f"Console errors on {rel_path}: {hard[:3]}"
