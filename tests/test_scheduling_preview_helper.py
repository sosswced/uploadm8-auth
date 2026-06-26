"""Tests for scheduling preview response helper."""

from __future__ import annotations

from datetime import datetime, timezone

from services.scheduling_preview import preview_response_payload, smart_schedule_explanation


def test_preview_response_payload_shape():
    smart = {
        "tiktok": datetime(2026, 6, 15, 19, 0, tzinfo=timezone.utc),
        "youtube": datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc),
    }
    sm = {k: v.isoformat() for k, v in smart.items()}
    out = preview_response_payload(
        smart,
        sm,
        seed="test-seed",
        smart_schedule_days=14,
        user_timezone="America/Chicago",
    )
    assert out["seed"] == "test-seed"
    assert out["smart_schedule"] == sm
    assert out["schedule"] == sm
    assert "tiktok" in out["explanation"]
    assert out["scheduled_time"] == min(sm.values())


def test_smart_schedule_explanation_has_reason():
    smart = {"tiktok": datetime(2026, 6, 15, 19, 0, tzinfo=timezone.utc)}
    exp = smart_schedule_explanation(smart, user_timezone="UTC")
    assert "reason" in exp["tiktok"]
