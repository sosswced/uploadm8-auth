"""TikTok Direct Post audited vs unaudited privacy clamp."""

from __future__ import annotations

from pathlib import Path

from services.tiktok_api import (
    tiktok_app_audited,
    tiktok_direct_post_status,
    tiktok_force_private_unaudited,
    tiktok_unaudited_mode,
)
from stages.publish_stage import (
    _tiktok_force_private_unaudited_enabled,
    _tiktok_unaudited_private_only_error,
)


def test_audited_enables_public_direct_post(monkeypatch):
    monkeypatch.setenv("TIKTOK_APP_AUDITED", "1")
    monkeypatch.delenv("TIKTOK_FORCE_PRIVATE_UNAUDITED", raising=False)
    assert tiktok_app_audited() is True
    assert tiktok_unaudited_mode() is False
    assert tiktok_force_private_unaudited() is False
    status = tiktok_direct_post_status()
    assert status["api"] == "content_posting_direct_post"
    assert status["source"] == "FILE_UPLOAD"
    assert status["public_publish_enabled"] is True
    assert status["privacy_clamped_to_self_only"] is False
    assert _tiktok_force_private_unaudited_enabled() is False


def test_app_audited_zero_does_not_clamp_publish(monkeypatch):
    """UI may show unaudited, but publish only clamps on FORCE_PRIVATE."""
    monkeypatch.setenv("TIKTOK_APP_AUDITED", "0")
    monkeypatch.delenv("TIKTOK_FORCE_PRIVATE_UNAUDITED", raising=False)
    assert tiktok_app_audited() is False
    assert tiktok_unaudited_mode() is True
    assert tiktok_force_private_unaudited() is False
    assert tiktok_direct_post_status()["public_publish_enabled"] is True
    assert _tiktok_force_private_unaudited_enabled() is False


def test_unset_env_defaults_to_audited(monkeypatch):
    monkeypatch.delenv("TIKTOK_APP_AUDITED", raising=False)
    monkeypatch.delenv("TIKTOK_FORCE_PRIVATE_UNAUDITED", raising=False)
    assert tiktok_app_audited() is True
    assert tiktok_unaudited_mode() is False
    assert tiktok_direct_post_status()["public_publish_enabled"] is True


def test_force_private_overrides_audited(monkeypatch):
    monkeypatch.setenv("TIKTOK_APP_AUDITED", "1")
    monkeypatch.setenv("TIKTOK_FORCE_PRIVATE_UNAUDITED", "1")
    assert tiktok_app_audited() is True
    assert tiktok_force_private_unaudited() is True
    assert tiktok_direct_post_status()["public_publish_enabled"] is False


def test_unaudited_error_detection():
    assert _tiktok_unaudited_private_only_error(
        '{"error":{"code":"unaudited_client_can_only_post_to_private_accounts"}}'
    )
    assert not _tiktok_unaudited_private_only_error('{"error":{"code":"ok"}}')


def test_frontend_wires_audited_banner():
    html = Path("frontend/upload.html").read_text(encoding="utf-8")
    js = Path("frontend/js/tiktok-export.js").read_text(encoding="utf-8")
    copy = Path("frontend/js/tiktok-ux-copy.js").read_text(encoding="utf-8")
    assert "tiktokAuditedNotice" in html
    assert "tt-audited-banner" in html
    assert "tiktokAuditedNotice" in js
    assert "auditedBannerHtml" in copy
    assert "TIKTOK_APP_AUDITED=1" in Path(".env.example").read_text(encoding="utf-8")
