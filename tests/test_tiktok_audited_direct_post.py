"""TikTok Direct Post — audit-approved public publish (no privacy clamp)."""

from __future__ import annotations

from pathlib import Path

from core.config import OAUTH_CONFIG
from services.tiktok_api import (
    tiktok_app_audited,
    tiktok_direct_post_status,
    tiktok_force_private_unaudited,
    tiktok_unaudited_mode,
)
from stages.publish_stage import _tiktok_private_only_api_error


def test_direct_post_public_enabled():
    assert tiktok_app_audited() is True
    assert tiktok_unaudited_mode() is False
    assert tiktok_force_private_unaudited() is False
    status = tiktok_direct_post_status()
    assert status["api"] == "content_posting_direct_post"
    assert status["source"] == "FILE_UPLOAD"
    assert status["public_publish_enabled"] is True
    assert status["privacy_clamped_to_self_only"] is False
    assert status["unaudited_mode"] is False


def test_stale_env_cannot_disable_direct_post(monkeypatch):
    monkeypatch.setenv("TIKTOK_APP_AUDITED", "0")
    monkeypatch.setenv("TIKTOK_FORCE_PRIVATE_UNAUDITED", "1")
    assert tiktok_app_audited() is True
    assert tiktok_force_private_unaudited() is False
    assert tiktok_direct_post_status()["public_publish_enabled"] is True


def test_oauth_scope_matches_portal():
    scope = OAUTH_CONFIG["tiktok"]["scope"]
    for part in (
        "user.info.basic",
        "user.info.stats",
        "video.publish",
        "video.upload",
        "video.list",
    ):
        assert part in scope
    # Not on portal yet — requesting it causes invalid_scope on authorize.
    assert "user.info.profile" not in scope


def test_private_only_api_error_detection():
    assert _tiktok_private_only_api_error(
        '{"error":{"code":"unaudited_client_can_only_post_to_private_accounts"}}'
    )
    assert not _tiktok_private_only_api_error('{"error":{"code":"ok"}}')


def test_frontend_wires_direct_post_banner():
    html = Path("frontend/upload.html").read_text(encoding="utf-8")
    js = Path("frontend/js/tiktok-export.js").read_text(encoding="utf-8")
    copy = Path("frontend/js/tiktok-ux-copy.js").read_text(encoding="utf-8")
    assert "tiktokAuditedNotice" in html
    assert "tt-audited-banner" in html
    assert "tiktokAuditedNotice" in js
    assert "auditedBannerHtml" in copy
    assert "syncDirectPostBanner" in js
    assert "App audit in progress" not in copy
