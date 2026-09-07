"""Per-tier schedule horizon: ladder, labels, and presign enforcement."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from stages.entitlements import (
    TIER_CONFIG,
    get_entitlements_for_tier,
    resolve_schedule_horizon_days,
    scheduling_window_display_label,
    tier_cfg_to_api_dict,
)
from services.upload.schedule_guard import validate_presign_schedule


PUBLIC_LADDER = ("free", "creator_lite", "creator_pro", "studio", "agency")


def test_schedule_horizon_ladder_strictly_increases_across_public_tiers():
    days = [TIER_CONFIG[t]["schedule_horizon_days"] for t in PUBLIC_LADDER]
    assert days == sorted(days)
    assert days[0] == 14
    assert days[-1] == 5000
    # Creator Lite must not sit below Free (the old hours inversion).
    assert days[1] > days[0]


def test_lookahead_hours_matches_horizon_days():
    for slug, cfg in TIER_CONFIG.items():
        days = cfg["schedule_horizon_days"]
        assert cfg["lookahead_hours"] == days * 24, slug


def test_entitlements_expose_horizon_and_label():
    ent = get_entitlements_for_tier("studio")
    assert ent.schedule_horizon_days == 365
    assert ent.lookahead_hours == 8760
    api = tier_cfg_to_api_dict("studio", TIER_CONFIG["studio"])
    assert api["schedule_horizon_days"] == 365
    assert api["scheduling_window_label"] == "1 year"


def test_scheduling_window_labels_are_day_based():
    assert scheduling_window_display_label(schedule_horizon_days=14) == "14 days"
    assert scheduling_window_display_label(schedule_horizon_days=365) == "1 year"
    assert scheduling_window_display_label(schedule_horizon_days=5000) == "up to 5000 days"
    # Legacy hours-only rows still resolve.
    assert resolve_schedule_horizon_days(lookahead_hours=720) == 30


def test_validate_presign_rejects_smart_window_past_horizon():
    data = SimpleNamespace(
        schedule_mode="smart",
        platforms=["tiktok"],
        smart_schedule_days=90,
        scheduled_time=None,
    )
    with pytest.raises(HTTPException) as exc:
        validate_presign_schedule(data, schedule_horizon_days=14)
    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "schedule_horizon_exceeded"
    assert exc.value.detail["schedule_horizon_days"] == 14


def test_validate_presign_allows_smart_window_inside_horizon():
    data = SimpleNamespace(
        schedule_mode="smart",
        platforms=["tiktok"],
        smart_schedule_days=14,
        scheduled_time=None,
    )
    validate_presign_schedule(data, schedule_horizon_days=14)


def test_validate_presign_rejects_manual_date_past_horizon():
    far = datetime.now(timezone.utc) + timedelta(days=60)
    data = SimpleNamespace(
        schedule_mode="scheduled",
        platforms=["youtube"],
        scheduled_time=far.isoformat(),
        smart_schedule_days=14,
    )
    with pytest.raises(HTTPException) as exc:
        validate_presign_schedule(data, schedule_horizon_days=14)
    assert exc.value.detail["code"] == "schedule_horizon_exceeded"


def test_validate_presign_allows_manual_date_inside_horizon():
    soon = datetime.now(timezone.utc) + timedelta(days=3)
    data = SimpleNamespace(
        schedule_mode="scheduled",
        platforms=["youtube"],
        scheduled_time=soon.isoformat(),
        smart_schedule_days=14,
    )
    validate_presign_schedule(data, schedule_horizon_days=14)
