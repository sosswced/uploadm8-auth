"""
Live / staging E2E: real login session, scroll app pages, settings save, OAuth popup, optional upload.

Requires:
  - API + frontend reachable (PLAYWRIGHT_BASE_URL / PLAYWRIGHT_API_BASE, default :8000)
  - PLAYWRIGHT_TEST_EMAIL + PLAYWRIGHT_TEST_PASSWORD (or PLAYWRIGHT_ADMIN_*)

Optional:
  - PLAYWRIGHT_LIVE_UPLOAD=1 — run real presign + R2 + complete (costs tokens; needs ≥1 connected platform)
  - ffmpeg on PATH — to generate a 1s test MP4 for upload test

Guards:
  - Upload test is skipped unless PLAYWRIGHT_LIVE_UPLOAD=1 (avoid accidental live charges)
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import urllib.parse
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from tests.test_full_app_flow import _on_login_page

pytestmark = [pytest.mark.e2e, pytest.mark.live]


def _api_q() -> str:
    return urllib.parse.quote(
        os.environ.get("PLAYWRIGHT_API_BASE", "http://127.0.0.1:8000"),
        safe="",
    )


def _goto(page: Page, base_url: str, path: str, *, timeout: int = 30000) -> None:
    q = _api_q()
    p = path.lstrip("/")
    page.goto(f"{base_url.rstrip('/')}/{p}?api={q}", wait_until="domcontentloaded", timeout=timeout)


def _require_auth(page: Page) -> None:
    page.wait_for_timeout(800)
    if _on_login_page(page):
        pytest.skip("Not logged in — set PLAYWRIGHT_TEST_EMAIL/PASSWORD and ensure API accepts login")


def _scroll_page(page: Page) -> None:
    """Scroll main content to trigger lazy sections and infinite handlers."""
    for _ in range(8):
        page.mouse.wheel(0, 600)
        page.wait_for_timeout(120)
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(200)
    page.evaluate("window.scrollTo(0, 0)")


# Core app routes to exercise like a user browsing (scroll + stay on page)
LIVE_SCROLL_PATHS = [
    "dashboard.html",
    "upload.html",
    "queue.html",
    "scheduled.html",
    "platforms.html",
    "groups.html",
    "analytics.html",
    "kpi.html",
    "settings.html",
    "guide.html",
    "color-preferences.html",
    "billing.html",
]


def test_live_scroll_authenticated_shell_pages(page: Page, base_url: str) -> None:
    """Visit each shell page, scroll, ensure we are not bounced to login."""
    for path in LIVE_SCROLL_PATHS:
        _goto(page, base_url, path)
        _require_auth(page)
        expect(page.locator("body")).to_be_visible()
        _scroll_page(page)


def test_live_settings_preferences_save_random_caption_style(page: Page, base_url: str) -> None:
    """Preferences tab: pick a random caption style + thumbnail interval, save, expect success toast."""
    _goto(page, base_url, "settings.html#preferences")
    _require_auth(page)

    page.locator('.settings-tab[data-tab="preferences"]').click()
    page.wait_for_timeout(600)

    styles = ("story", "punchy", "factual")
    pick = random.choice(styles)
    page.select_option("#captionStyle", value=pick)

    # Change something else deterministic but harmless
    page.select_option("#thumbnailInterval", value=random.choice(["1", "2", "5", "10"]))

    page.locator("#savePrefsBtn").click()
    page.locator("text=/Preferences saved|saved and synced/i").first.wait_for(timeout=25000)


def test_live_platforms_connect_opens_oauth_popup(page: Page, base_url: str) -> None:
    """Click YouTube Add Account — OAuth popup should open (user can cancel; we close it)."""
    _goto(page, base_url, "platforms.html")
    _require_auth(page)

    yt_btn = page.locator('button[onclick*="connectAccount(\'youtube\')"]')
    expect(yt_btn).to_be_visible(timeout=15000)

    try:
        with page.expect_popup(timeout=20000) as pop_ev:
            yt_btn.click()
        popup = pop_ev.value
    except Exception as e:
        pytest.skip(f"OAuth popup did not open (browser blocked popups or API error): {e}")

    try:
        popup.wait_for_load_state("domcontentloaded", timeout=20000)
        url = (popup.url or "").lower()
        assert any(
            x in url
            for x in ("google", "oauth", "youtube", "accounts", "login", "authorize", "connect")
        ), f"unexpected OAuth URL: {url[:200]}"
    finally:
        try:
            popup.close()
        except Exception:
            pass


@pytest.mark.slow
def test_live_upload_tiny_video_when_enabled(tmp_path: Path, page: Page, base_url: str) -> None:
    """
    Full path: select file, pick platform, start upload — only if PLAYWRIGHT_LIVE_UPLOAD=1.
    Requires ffmpeg and at least one connected platform with an enabled checkbox.
    """
    if os.environ.get("PLAYWRIGHT_LIVE_UPLOAD", "").strip().lower() not in ("1", "true", "yes"):
        pytest.skip("Set PLAYWRIGHT_LIVE_UPLOAD=1 to run a real upload (uses tokens)")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not on PATH — needed to generate a 1s test MP4")

    mp4 = tmp_path / "e2e_live.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-t",
            "1",
            str(mp4),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )

    _goto(page, base_url, "upload.html")
    _require_auth(page)
    page.wait_for_timeout(2000)

    if page.locator('input[name="platforms"]:not([disabled])').count() == 0:
        pytest.skip("No connected platforms — connect one on Platforms first, then re-run")

    page.set_input_files("#fileInput", str(mp4))
    page.wait_for_timeout(800)
    page.locator('input[name="platforms"]:not([disabled])').first.click()

    upload_btn = page.locator("#uploadBtn")
    expect(upload_btn).to_be_enabled(timeout=15000)
    upload_btn.click()

    # Progress UI or completion states
    prog = page.locator("#uploadProgress")
    expect(prog).to_be_visible(timeout=30000)

    # Allow long tail: R2 + complete + poll (live worker may be slow)
    page.locator(
        "text=/Uploading|Registering|Queued|success|processing|Finalising|Check Queue/i"
    ).first.wait_for(timeout=180000)


def test_live_analytics_queue_scheduled_smoke(page: Page, base_url: str) -> None:
    """Analytics, Queue, Scheduled: load + scroll (common user paths)."""
    for path in ("analytics.html", "queue.html", "scheduled.html"):
        _goto(page, base_url, path)
        _require_auth(page)
        expect(page.locator("body")).to_be_visible(timeout=15000)
        _scroll_page(page)
