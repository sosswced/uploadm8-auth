"""Meta OAuth scopes and publish permission gates after App Review approval."""

from __future__ import annotations

from services.meta_oauth import (
    facebook_page_access_token_after_refresh,
    meta_app_credentials,
    meta_facebook_oauth_scope,
    meta_instagram_oauth_scope,
    meta_oauth_auth_type,
    pick_managed_page,
    require_facebook_publish,
    require_instagram_publish,
)


def test_facebook_full_scope_uses_pages_manage_posts_not_publish_video(monkeypatch):
    monkeypatch.setenv("META_OAUTH_MODE", "full")
    monkeypatch.delenv("META_FACEBOOK_OAUTH_SCOPE", raising=False)
    scope = meta_facebook_oauth_scope()
    assert "pages_manage_posts" in scope
    assert "read_insights" in scope
    assert "publish_video" not in scope.split(",")


def test_instagram_full_scope_includes_publish_insights_and_basic_deps(monkeypatch):
    monkeypatch.setenv("META_OAUTH_MODE", "full")
    monkeypatch.delenv("META_INSTAGRAM_OAUTH_SCOPE", raising=False)
    scope = meta_instagram_oauth_scope()
    for perm in (
        "instagram_basic",
        "instagram_content_publish",
        "instagram_manage_insights",
        "pages_show_list",
        "pages_read_engagement",
        "pages_read_user_content",
        "business_management",
    ):
        assert perm in scope
    assert "publish_video" not in scope


def test_meta_oauth_auth_type_rerequests_new_permissions():
    auth_type = meta_oauth_auth_type()
    assert "rerequest" in auth_type.split(",")
    assert "reauthenticate" in auth_type.split(",")


def test_facebook_credentials_prefer_facebook_client_id(monkeypatch):
    monkeypatch.setenv("FACEBOOK_CLIENT_ID", "fb-app")
    monkeypatch.setenv("FACEBOOK_CLIENT_SECRET", "fb-secret")
    monkeypatch.setenv("META_APP_ID", "meta-app")
    monkeypatch.setenv("META_APP_SECRET", "meta-secret")
    app_id, app_secret = meta_app_credentials("facebook")
    assert app_id == "fb-app"
    assert app_secret == "fb-secret"
    ig_id, ig_secret = meta_app_credentials("instagram")
    assert ig_id == "meta-app"
    assert ig_secret == "meta-secret"


def test_pick_managed_page_matches_id_beyond_first():
    pages = [
        {"id": "111", "access_token": "t1"},
        {"id": "222", "access_token": "t2"},
    ]
    assert pick_managed_page(pages, page_id="222")["access_token"] == "t2"
    assert pick_managed_page(pages, page_id="999") is None
    assert pick_managed_page(pages)["id"] == "111"


def test_facebook_refresh_keeps_page_token_when_accounts_empty():
    kept = facebook_page_access_token_after_refresh(
        stored_access_token="page-token",
        stored_page_id="222",
        matched_page=None,
        new_user_token="user-llt",
    )
    assert kept == "page-token"


def test_facebook_refresh_uses_matched_page_token():
    kept = facebook_page_access_token_after_refresh(
        stored_access_token="old-page-token",
        stored_page_id="222",
        matched_page={"id": "222", "access_token": "fresh-page-token"},
        new_user_token="user-llt",
    )
    assert kept == "fresh-page-token"


def test_oauth_callback_only_stamps_non_expiring_when_llt_ok():
    import inspect

    from routers import oauth as oauth_mod

    src = inspect.getsource(oauth_mod.oauth_callback)
    assert "meta_llt_ok" in src
    assert "non_expiring=True" in src
    assert "storing short-lived page token" in src


def test_require_facebook_publish_accepts_pages_manage_posts():
    token = {
        "meta_permissions": [
            {"permission": "pages_manage_posts", "status": "granted"},
            {"permission": "publish_video", "status": "declined"},
        ]
    }
    assert require_facebook_publish(token) is None


def test_require_facebook_publish_rejects_live_only_publish_video():
    """publish_video alone must not authorize Page VOD/Reels."""
    token = {
        "meta_permissions": [
            {"permission": "pages_manage_posts", "status": "declined"},
            {"permission": "publish_video", "status": "granted"},
        ]
    }
    msg = require_facebook_publish(token)
    assert msg
    assert "pages_manage_posts" in msg


def test_require_facebook_publish_blocks_when_page_publish_declined():
    token = {
        "meta_permissions": [
            {"permission": "pages_manage_posts", "status": "declined"},
        ]
    }
    msg = require_facebook_publish(token)
    assert msg
    assert "pages_manage_posts" in msg


def test_require_facebook_publish_unknown_snapshot_allows_attempt():
    """Legacy tokens without a permission snapshot may still call Graph."""
    assert require_facebook_publish({}) is None
    assert require_facebook_publish({"meta_permissions": []}) is None


def test_require_instagram_publish_asks_reconnect_not_app_review():
    token = {
        "meta_permissions": [
            {"permission": "instagram_content_publish", "status": "declined"},
        ]
    }
    msg = require_instagram_publish(token)
    assert msg
    assert "Reconnect" in msg
    assert "App Review" not in msg
