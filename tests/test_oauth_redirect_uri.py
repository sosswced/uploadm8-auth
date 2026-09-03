"""OAuth redirect URI uses OAUTH_PUBLIC_BASE_URL when set."""

from __future__ import annotations

import core.config as config
from core.oauth import get_oauth_redirect_uri


def test_redirect_uses_base_url_by_default(monkeypatch):
    monkeypatch.setattr(config, "BASE_URL", "https://auth.uploadm8.com")
    monkeypatch.setattr(config, "OAUTH_PUBLIC_BASE_URL", None)
    assert get_oauth_redirect_uri("tiktok") == "https://auth.uploadm8.com/api/oauth/tiktok/callback"


def test_redirect_prefers_oauth_public_base(monkeypatch):
    monkeypatch.setattr(config, "BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setattr(config, "OAUTH_PUBLIC_BASE_URL", "https://abc.ngrok-free.app")
    assert (
        get_oauth_redirect_uri("youtube")
        == "https://abc.ngrok-free.app/api/oauth/youtube/callback"
    )
