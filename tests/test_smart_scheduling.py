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


def test_static_priors_use_local_hot_windows():
    """Priors come from hot windows — mass in frames, not a single hardcoded hour."""
    from core.scheduling import (
        STATIC_PRIOR_RESEARCH_VERSION,
        hour_in_hot_windows,
        platform_hot_windows,
        static_hour_prior_24,
    )

    assert STATIC_PRIOR_RESEARCH_VERSION.startswith("2026")
    wins = platform_hot_windows("tiktok")
    assert any(w["label"] == "afternoon" for w in wins)
    assert any(w["label"] == "evening" for w in wins)

    tt = static_hour_prior_24("tiktok")
    in_frame = sum(tt[h] for h in range(24) if hour_in_hot_windows(h, "tiktok"))
    out_frame = sum(tt[h] for h in range(24) if not hour_in_hot_windows(h, "tiktok"))
    assert in_frame > out_frame

    fb = static_hour_prior_24("facebook")
    morning = sum(fb[h] for h in range(9, 13))
    assert morning > sum(fb[h] for h in range(0, 6))


def test_smart_schedule_samples_around_hot_windows():
    """Repeated static draws should mostly land inside platform hot frames."""
    from zoneinfo import ZoneInfo

    import core.scheduling as sched

    now = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)
    old_now = sched._now_utc
    sched._now_utc = lambda: now
    try:
        hits = 0
        for i in range(40):
            result = calculate_smart_schedule(
                ["tiktok"],
                num_days=14,
                user_timezone="America/Chicago",
                random_seed=f"window-sample-{i}",
            )
            local = result["tiktok"].astimezone(ZoneInfo("America/Chicago"))
            if sched.hour_in_hot_windows(local.hour, "tiktok"):
                hits += 1
        assert hits >= 28
    finally:
        sched._now_utc = old_now


def test_calculate_smart_schedule_honors_local_hour_weights():
    """When weights are local, a delta-only peak at hour 15 schedules ~15:00 local."""
    from zoneinfo import ZoneInfo

    import core.scheduling as sched

    now = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)
    old_now = sched._now_utc
    sched._now_utc = lambda: now
    try:
        weights = [0.0] * 24
        weights[15] = 1.0
        result = calculate_smart_schedule(
            ["tiktok"],
            num_days=7,
            user_timezone="America/Chicago",
            hour_weights_by_platform={"tiktok": weights},
            hour_weights_are_local=True,
            window_bias_strength=0.0,
            random_seed="local-peak-15",
        )
        local = result["tiktok"].astimezone(ZoneInfo("America/Chicago"))
        assert local.hour == 15
    finally:
        sched._now_utc = old_now
