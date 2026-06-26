"""Tests for smart scheduling algorithm (jitter, TZ, deterministic seed)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.scheduling import (
    _JITTER_MAX_SECONDS,
    _apply_subsecond_jitter,
    _rng_from_seed,
    calculate_smart_schedule,
    utc_weights_as_local,
)


def test_jitter_stays_within_thirty_minutes():
    rng = _rng_from_seed("jitter-test")
    now = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)
    anchor = datetime(2026, 6, 12, 19, 0, tzinfo=timezone.utc)
    for _ in range(50):
        out = _apply_subsecond_jitter(anchor, now, rng=rng)
        delta = abs((out - anchor).total_seconds())
        assert delta <= _JITTER_MAX_SECONDS


def test_deterministic_seed_same_schedule():
    platforms = ["tiktok", "youtube"]
    a = calculate_smart_schedule(
        platforms,
        num_days=7,
        user_timezone="America/Chicago",
        random_seed="upload-abc-123",
    )
    b = calculate_smart_schedule(
        platforms,
        num_days=7,
        user_timezone="America/Chicago",
        random_seed="upload-abc-123",
    )
    assert a.keys() == b.keys()
    for plat in platforms:
        assert a[plat] == b[plat]


def test_different_seeds_produce_different_schedules():
    platforms = ["tiktok", "youtube", "instagram"]
    a = calculate_smart_schedule(platforms, num_days=14, random_seed="seed-a")
    b = calculate_smart_schedule(platforms, num_days=14, random_seed="seed-b")
    assert a != b


def test_utc_weights_as_local_shifts_hours():
    utc_w = [0.0] * 24
    utc_w[19] = 1.0
    ref = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)
    local_w = utc_weights_as_local(utc_w, __import__("zoneinfo").ZoneInfo("America/New_York"), ref)
    peak = max(range(24), key=lambda h: local_w[h])
    assert peak != 19
