"""Verification must keep retrying until confirmed/rejected — not die on unknown."""

from __future__ import annotations

import inspect

from stages import verify_stage
from stages.verify_stage import (
    _is_terminal_verify_status,
    _next_verify_status,
    _tiktok_items_to_update,
)


def test_unknown_and_pending_are_not_terminal():
    assert _is_terminal_verify_status("confirmed")
    assert _is_terminal_verify_status("rejected")
    assert not _is_terminal_verify_status("pending")
    assert not _is_terminal_verify_status("unknown")
    assert not _is_terminal_verify_status("")


def test_next_status_keeps_retrying_on_unknown():
    assert _next_verify_status("tiktok", "unknown", has_video_id=False) == "pending"
    assert _next_verify_status("tiktok", "pending", has_video_id=False) == "pending"
    assert _next_verify_status("tiktok", "confirmed", has_video_id=True) == "confirmed"
    # Confirmed without video_id is not finished for TikTok metrics.
    assert _next_verify_status("tiktok", "confirmed", has_video_id=False) == "pending"
    assert _next_verify_status("youtube", "confirmed", has_video_id=False) == "confirmed"
    assert _next_verify_status("youtube", "rejected", has_video_id=False) == "rejected"
    assert _next_verify_status("instagram", "unknown", has_video_id=False) == "pending"
    assert _next_verify_status("instagram", "unknown", has_video_id=True) == "confirmed"


def test_tiktok_items_prefer_publish_id_match():
    items = [
        {"platform": "tiktok", "publish_id": "a", "success": True},
        {"platform": "tiktok", "publish_id": "b", "success": True},
        {"platform": "youtube", "publish_id": "a"},
    ]
    matched = _tiktok_items_to_update(items, "b")
    assert len(matched) == 1
    assert matched[0]["publish_id"] == "b"


def test_tiktok_items_fallback_to_awaiting_video_id():
    items = [
        {"platform": "tiktok", "publish_id": "old", "platform_video_id": "1", "success": True},
        {"platform": "tiktok", "publish_id": "new", "success": True},
    ]
    matched = _tiktok_items_to_update(items, "missing")
    assert len(matched) == 1
    assert matched[0]["publish_id"] == "new"


def test_verify_loads_token_by_row_id_not_meta_google_alias():
    """platform_tokens.platform is facebook|instagram|youtube|tiktok — never meta/google.

    The old alias map made every Meta/YouTube verify miss the token row, so
    accepted posts sat on verify_status=pending forever and the worker logged
    'Verifying N publish attempts' on a loop without ever confirming.
    """
    src = inspect.getsource(verify_stage.verify_single_attempt)
    assert "load_platform_token_by_id" in src
    # No live platform_to_db_key alias map (docstrings may still mention the bug).
    assert "platform_to_db_key" not in src
    assert 'db_key = platform_to_db_key' not in src
    assert "load_platform_token(db_pool, user_id, plat)" in src


def test_meta_with_post_id_confirms_without_token(monkeypatch):
    import asyncio

    calls: list[tuple] = []

    async def fake_update(pool, attempt_id, verify_status, platform_url=None):
        calls.append((attempt_id, verify_status))

    monkeypatch.setattr(
        verify_stage.db_stage, "update_publish_attempt_verified", fake_update
    )

    async def _run():
        await verify_stage.verify_single_attempt(
            None,
            {
                "id": "att-1",
                "platform": "facebook",
                "user_id": "u1",
                "platform_post_id": "12345",
                "verify_status": "pending",
            },
        )

    asyncio.run(_run())
    assert calls == [("att-1", "confirmed")]
