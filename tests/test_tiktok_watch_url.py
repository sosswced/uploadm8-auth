"""TikTok www.tiktok.com/video/{id} (no @handle) 404s — never emit that path."""

from __future__ import annotations

from pathlib import Path

from services.tiktok_api import rewrite_tiktok_watch_url, tiktok_watch_url
from stages.context import PlatformResult
from stages.notify_stage import _fallback_post_url, _normalize_post_url

ROOT = Path(__file__).resolve().parent.parent
SYNTH_VID = "7123456789012345678"


def test_tiktok_watch_url_requires_handle():
    assert tiktok_watch_url(SYNTH_VID, "creator") == (
        f"https://www.tiktok.com/@creator/video/{SYNTH_VID}"
    )
    assert tiktok_watch_url(SYNTH_VID, "@creator") == (
        f"https://www.tiktok.com/@creator/video/{SYNTH_VID}"
    )
    assert tiktok_watch_url(SYNTH_VID, "") == ""
    assert tiktok_watch_url(SYNTH_VID, None) == ""
    assert tiktok_watch_url("", "creator") == ""


def test_rewrite_rebuilds_handleless_www_path():
    bare = f"https://www.tiktok.com/video/{SYNTH_VID}"
    assert rewrite_tiktok_watch_url(bare, username="creator") == (
        f"https://www.tiktok.com/@creator/video/{SYNTH_VID}"
    )
    assert rewrite_tiktok_watch_url(bare, username="") == ""
    assert rewrite_tiktok_watch_url(
        f"https://www.tiktok.com/video/{SYNTH_VID}?lang=en",
        username="creator",
    ) == f"https://www.tiktok.com/@creator/video/{SYNTH_VID}"


def test_rewrite_keeps_handle_and_short_links():
    good = f"https://www.tiktok.com/@creator/video/{SYNTH_VID}?lang=en"
    assert rewrite_tiktok_watch_url(good, username="") == good
    short = "https://vm.tiktok.com/ZMabcdefg/"
    assert rewrite_tiktok_watch_url(short, username="creator") == short


def test_rewrite_builds_from_id_and_handle_when_url_empty():
    assert rewrite_tiktok_watch_url(
        "", video_id=SYNTH_VID, username="creator"
    ) == f"https://www.tiktok.com/@creator/video/{SYNTH_VID}"


def test_notify_normalizes_handleless_tiktok_url():
    result = PlatformResult(
        platform="tiktok",
        success=True,
        platform_video_id=SYNTH_VID,
        platform_url=f"https://www.tiktok.com/video/{SYNTH_VID}",
        account_username="creator",
    )
    assert _normalize_post_url(result) == (
        f"https://www.tiktok.com/@creator/video/{SYNTH_VID}"
    )


def test_notify_drops_handleless_tiktok_url_without_username():
    result = PlatformResult(
        platform="tiktok",
        success=True,
        platform_video_id=SYNTH_VID,
        platform_url=f"https://www.tiktok.com/video/{SYNTH_VID}",
        account_username="",
    )
    assert _normalize_post_url(result) is None
    assert _fallback_post_url(result) is None


def test_frontend_does_not_emit_handleless_tiktok_watch_url():
    utils = (ROOT / "frontend" / "js" / "upload-utils.js").read_text(encoding="utf-8")
    analytics = (ROOT / "frontend" / "analytics.html").read_text(encoding="utf-8")
    assert "return /^\\/@[^/]+\\/video\\//i.test(path);" in utils
    assert "www.tiktok.com/video/' + encodeURIComponent(vid)" not in utils
    assert "www.tiktok.com/video/' + encodeURIComponent(vid)" not in analytics
    verify_src = (ROOT / "stages" / "verify_stage.py").read_text(encoding="utf-8")
    assert "tiktok_watch_url" in verify_src
    assert 'f"https://www.tiktok.com/video/{' not in verify_src
