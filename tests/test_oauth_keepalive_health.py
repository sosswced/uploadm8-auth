"""OAuth keepalive rotation + connected-account health."""

from __future__ import annotations

from services.platform_accounts import account_status, serialize_platform_account
from services.platform_oauth_refresh import _refresh_hard_failed


def test_refresh_hard_failed_missing_refresh_token():
    before = {"access_token": "a1"}
    after = {"access_token": "a1"}
    assert _refresh_hard_failed("youtube", before, after) is True
    assert _refresh_hard_failed("tiktok", before, after) is True


def test_refresh_hard_failed_false_when_access_rotated():
    before = {"access_token": "old", "refresh_token": "rt"}
    after = {"access_token": "new", "refresh_token": "rt", "expires_at": "2026-08-07T12:00:00+00:00"}
    assert _refresh_hard_failed("youtube", before, after) is False


def test_account_status_oauth_health_needs_reconnection():
    assert (
        account_status("tok-1", {}, oauth_health="needs_reconnection")
        == "needs_reconnection"
    )
    assert account_status("tok-1", {}, oauth_health="ok") == "active"
    assert account_status("tok-1", {"tok-1": "TOKEN_EXPIRED"}) == "needs_reconnection"


def test_serialize_exposes_oauth_health():
    row = {
        "id": "tok-1",
        "account_id": "ext-1",
        "account_name": "Demo",
        "account_username": "demo",
        "account_avatar": None,
        "is_primary": True,
        "created_at": None,
        "last_oauth_reconnect_at": None,
        "last_used_at": None,
        "oauth_health": "needs_reconnection",
    }
    out = serialize_platform_account(row, presign=False)
    assert out["status"] == "needs_reconnection"
    assert out["oauth_health"] == "needs_reconnection"
