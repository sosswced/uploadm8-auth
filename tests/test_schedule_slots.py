"""Tests for shared schedule slot helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from services.schedule_slots import (
    all_slot_datetimes,
    compute_scheduled_stats,
    day_occupancy_from_today,
    day_offsets_from_today,
    rows_for_blocked_days,
)


def _utc(y, m, d, h=12):
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def test_all_slot_datetimes_smart_multi_day():
    row = {
        "schedule_mode": "smart",
        "scheduled_time": _utc(2026, 6, 15, 19),
        "schedule_metadata": {
            "tiktok": "2026-06-15T19:00:00+00:00",
            "youtube": "2026-06-19T14:00:00+00:00",
        },
    }
    slots = all_slot_datetimes(row)
    assert len(slots) == 2
    assert slots[0].day == 15
    assert slots[1].day == 19


def test_compute_scheduled_stats_counts_friday_for_smart():
    now = _utc(2026, 6, 19, 10)
    rows = [
        {
            "schedule_mode": "smart",
            "scheduled_time": _utc(2026, 6, 15, 19),
            "schedule_metadata": {
                "tiktok": "2026-06-15T19:00:00+00:00",
                "youtube": "2026-06-19T14:00:00+00:00",
            },
        }
    ]
    stats = compute_scheduled_stats(rows, now=now)
    assert stats["pending"] == 1
    assert stats["today"] == 1
    assert stats["week"] == 1


def test_day_offsets_blocks_friday():
    today = _utc(2026, 6, 10).date()
    row = {
        "schedule_mode": "smart",
        "schedule_metadata": {"youtube": "2026-06-13T14:00:00+00:00"},
        "scheduled_time": _utc(2026, 6, 13, 14),
    }
    offsets = day_offsets_from_today(row, today=today, num_days=7)
    assert 3 in offsets
    occ = day_occupancy_from_today(row, today=today, num_days=7)
    assert occ.get(3) == 1


def test_day_occupancy_counts_spill_through_expand_horizon():
    """Spill day 20 is invisible at num_days=14 but visible when horizon=28."""
    today = _utc(2026, 6, 10).date()
    row = {
        "schedule_mode": "smart",
        "schedule_metadata": {
            "tiktok": "2026-06-30T12:00:00+00:00",  # offset 20
        },
        "scheduled_time": _utc(2026, 6, 30, 12),
    }
    assert day_occupancy_from_today(row, today=today, num_days=14) == {}
    occ = day_occupancy_from_today(row, today=today, num_days=14, horizon=28)
    assert occ.get(20) == 1
    assert 20 in day_offsets_from_today(row, today=today, num_days=14, horizon=28)


def test_day_occupancy_counts_multiple_platform_slots_same_day():
    today = _utc(2026, 6, 10).date()
    row = {
        "schedule_mode": "smart",
        "schedule_metadata": {
            "tiktok": "2026-06-12T12:00:00+00:00",
            "youtube": "2026-06-12T18:00:00+00:00",
        },
        "scheduled_time": _utc(2026, 6, 12, 12),
    }
    occ = day_occupancy_from_today(row, today=today, num_days=7)
    assert occ.get(2) == 2


def test_get_existing_scheduled_days_uses_friday_metadata():
    """Occupancy offsets match schedule_metadata, not just scheduled_time."""
    import asyncio

    from core.scheduling import get_existing_scheduled_days

    class _FakeConn:
        async def fetch(self, *_args, **_kwargs):
            return [
                {
                    "scheduled_time": _utc(2026, 6, 11, 12),
                    "schedule_mode": "smart",
                    "schedule_metadata": {
                        "tiktok": "2026-06-11T12:00:00+00:00",
                        "youtube": "2026-06-13T14:00:00+00:00",
                    },
                }
            ]

    import core.scheduling as sched

    fixed_now = _utc(2026, 6, 10, 12)
    old_now = sched._now_utc
    sched._now_utc = lambda: fixed_now
    try:
        used = asyncio.run(get_existing_scheduled_days(_FakeConn(), "user-1", 7))
    finally:
        sched._now_utc = old_now

    assert used.get(1) == 1
    assert used.get(3) == 1
