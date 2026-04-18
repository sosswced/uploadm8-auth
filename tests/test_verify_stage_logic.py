from stages.verify_stage import (
    _interpret_facebook_verify_payload,
    _interpret_instagram_verify_payload,
)
import asyncio
from stages import verify_stage as vs


def test_interpret_instagram_confirmed_finished():
    status, url = _interpret_instagram_verify_payload(
        {
            "id": "123",
            "status_code": "FINISHED",
            "permalink": "https://www.instagram.com/reel/abc123/",
            "media_type": "REELS",
        },
        "123",
    )
    assert status == "confirmed"
    assert url == "https://www.instagram.com/reel/abc123/"


def test_interpret_instagram_pending_processing():
    status, url = _interpret_instagram_verify_payload(
        {"id": "123", "status_code": "IN_PROGRESS"},
        "123",
    )
    assert status == "pending"
    assert url is None


def test_interpret_facebook_confirmed_without_status_uses_id():
    status, url = _interpret_facebook_verify_payload(
        {"id": "999"},
        "999",
    )
    assert status == "confirmed"
    assert url == "https://www.facebook.com/watch/?v=999"


def test_interpret_facebook_rejected_error_status():
    status, url = _interpret_facebook_verify_payload(
        {"id": "999", "status": {"video_status": "error"}},
        "999",
    )
    assert status == "rejected"
    assert url is None


def test_verify_single_attempt_routes_instagram(monkeypatch):
    calls = {}

    async def _fake_load(_pool, _user_id, db_key):
        calls["db_key"] = db_key
        return {"access_token": "token"}

    async def _fake_verify(media_id, _token):
        calls["media_id"] = media_id
        return "confirmed", "https://www.instagram.com/reel/abc123/"

    async def _fake_update(_pool, attempt_id, verify_status, platform_url=None):
        calls["update"] = (attempt_id, verify_status, platform_url)

    monkeypatch.setattr(vs.db_stage, "load_platform_token", _fake_load)
    monkeypatch.setattr(vs, "verify_instagram", _fake_verify)
    monkeypatch.setattr(vs.db_stage, "update_publish_attempt_verified", _fake_update)

    attempt = {
        "id": "attempt-1",
        "platform": "instagram",
        "platform_post_id": "ig-media-1",
        "user_id": "user-1",
    }
    asyncio.run(vs.verify_single_attempt(None, attempt))

    assert calls["db_key"] == "instagram"
    assert calls["media_id"] == "ig-media-1"
    assert calls["update"] == ("attempt-1", "confirmed", "https://www.instagram.com/reel/abc123/")


def test_verify_single_attempt_routes_facebook(monkeypatch):
    calls = {}

    async def _fake_load(_pool, _user_id, db_key):
        calls["db_key"] = db_key
        return {"access_token": "token"}

    async def _fake_verify(video_id, _token):
        calls["video_id"] = video_id
        return "pending", None

    async def _fake_update(_pool, attempt_id, verify_status, platform_url=None):
        calls["update"] = (attempt_id, verify_status, platform_url)

    monkeypatch.setattr(vs.db_stage, "load_platform_token", _fake_load)
    monkeypatch.setattr(vs, "verify_facebook", _fake_verify)
    monkeypatch.setattr(vs.db_stage, "update_publish_attempt_verified", _fake_update)

    attempt = {
        "id": "attempt-2",
        "platform": "facebook",
        "platform_post_id": "fb-video-1",
        "user_id": "user-2",
    }
    asyncio.run(vs.verify_single_attempt(None, attempt))

    assert calls["db_key"] == "facebook"
    assert calls["video_id"] == "fb-video-1"
    assert calls["update"] == ("attempt-2", "pending", None)
