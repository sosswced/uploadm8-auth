"""Verification must keep retrying until confirmed/rejected — not die on unknown."""

from __future__ import annotations

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
