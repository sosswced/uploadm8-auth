"""Hardening for Publish Now / Schedule / Smart Schedule testing reliability."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.models import UploadInit
from core.scheduling import (
    _pick_day_offset,
    _rng_from_seed,
    calculate_smart_schedule,
    clamp_smart_schedule_days,
)
from datetime import datetime, timezone


def test_clamp_smart_schedule_days():
    assert clamp_smart_schedule_days(0) == 14
    assert clamp_smart_schedule_days(-5) == 14
    assert clamp_smart_schedule_days(None) == 14
    assert clamp_smart_schedule_days("NaN") == 14
    assert clamp_smart_schedule_days(7) == 7
    assert clamp_smart_schedule_days(14) == 14
    assert clamp_smart_schedule_days(365) == 365
    assert clamp_smart_schedule_days(9999) == 730
    assert clamp_smart_schedule_days("30") == 30


def test_upload_init_rejects_zero_smart_days():
    with pytest.raises(ValidationError):
        UploadInit(
            filename="a.mp4",
            file_size=10,
            content_type="video/mp4",
            platforms=["tiktok"],
            schedule_mode="smart",
            smart_schedule_days=0,
        )


def test_upload_init_accepts_presets():
    for days in (2, 3, 5, 7, 14, 30, 50, 90, 180, 365):
        u = UploadInit(
            filename="a.mp4",
            file_size=10,
            content_type="video/mp4",
            platforms=["youtube", "tiktok"],
            schedule_mode="smart",
            smart_schedule_days=days,
        )
        assert u.smart_schedule_days == days


def test_pick_day_offset_never_crashes_on_zero_or_exhausted():
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    rng = _rng_from_seed("exhaust")
    # Zero days clamps internally
    off = _pick_day_offset(now, "tiktok", 0, set(), None, rng)
    assert isinstance(off, int) and off >= 1

    # Fully occupied short window packs inside 1..7 (never past window)
    blocked = {d: 1 for d in range(1, 8)}
    used = set()
    off2 = _pick_day_offset(now, "youtube", 7, used, blocked, rng)
    assert 1 <= off2 <= 7


def test_pick_day_offset_packs_when_window_full():
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    rng = _rng_from_seed("blocked-last")
    # Entire window occupied — still returns an offset inside 1..7
    blocked = {d: 2 for d in range(1, 8)}
    off = _pick_day_offset(now, "tiktok", 7, set(), blocked, rng)
    assert 1 <= off <= 7


def test_calculate_smart_schedule_clamps_and_spreads_platforms():
    sched = calculate_smart_schedule(
        ["tiktok", "youtube", "instagram"],
        num_days=0,  # clamped to 14
        user_timezone="America/Chicago",
        random_seed="test-smart-zero",
    )
    assert set(sched) == {"tiktok", "youtube", "instagram"}
    days = {(dt.astimezone(timezone.utc).date()) for dt in sched.values()}
    # Prefer different calendar days across platforms when window allows
    assert len(days) >= 2


def test_dense_batch_packs_inside_window_without_spill():
    """When primary window is full, later slots pack inside num_days (no spill)."""
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    import core.scheduling as sched

    old_now = sched._now_utc
    sched._now_utc = lambda: now
    try:
        occupancy: dict[int, int] = {d: 1 for d in range(1, 15)}  # 14d full
        result = calculate_smart_schedule(
            ["tiktok", "youtube"],
            num_days=14,
            user_timezone="UTC",
            day_occupancy=occupancy,
            random_seed="pack-batch",
        )
        for dt in result.values():
            offset = (dt.astimezone(timezone.utc).date() - now.date()).days
            assert 1 <= offset <= 14
        assert set(result) == {"tiktok", "youtube"}
    finally:
        sched._now_utc = old_now


def test_slots_never_exceed_num_days_even_with_many_platforms():
    """More platforms than days still stay inside the window."""
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    import core.scheduling as sched

    old_now = sched._now_utc
    sched._now_utc = lambda: now
    try:
        result = calculate_smart_schedule(
            ["tiktok", "youtube", "instagram", "facebook"],
            num_days=2,
            user_timezone="UTC",
            random_seed="tight-window",
        )
        assert len(result) == 4
        for dt in result.values():
            offset = (dt.astimezone(timezone.utc).date() - now.date()).days
            assert 1 <= offset <= 2
    finally:
        sched._now_utc = old_now


def test_eleven_video_batch_packs_into_two_to_five_day_windows():
    """Dense QA batch: 11 combos × 4 platforms over 2–5 days must stay in-window.

    Mirrors serialized smart-presign occupancy (upload.html packs siblings
    sequentially) so short-window logic tests do not spill past num_days.
    """
    from services.scheduling_preview import occupancy_from_schedule

    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    import core.scheduling as sched

    plats = ["tiktok", "youtube", "instagram", "facebook"]
    old_now = sched._now_utc
    sched._now_utc = lambda: now
    try:
        for num_days in (2, 3, 5):
            extra: dict[int, int] = {}
            total = 0
            for i in range(11):
                smart = calculate_smart_schedule(
                    plats,
                    num_days=num_days,
                    user_timezone="America/Chicago",
                    day_occupancy=dict(extra),
                    random_seed=f"qa-batch:slot-{i}",
                )
                assert set(smart.keys()) == set(plats)
                for dt in smart.values():
                    offset = (dt.astimezone(timezone.utc).date() - now.date()).days
                    assert 1 <= offset <= num_days, (num_days, i, offset, dt)
                    total += 1
                for offset, count in occupancy_from_schedule(smart, now=now).items():
                    extra[offset] = extra.get(offset, 0) + count
            assert total == 11 * 4
            assert all(1 <= d <= num_days for d in extra)
    finally:
        sched._now_utc = old_now


def test_twenty_video_batch_packs_over_three_day_window():
    """Dynamic N-over-W: 20 videos × 4 platforms over ~72h (3 days) stays in-window."""
    from services.scheduling_preview import occupancy_from_schedule

    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    import core.scheduling as sched

    plats = ["tiktok", "youtube", "instagram", "facebook"]
    num_days = 3
    old_now = sched._now_utc
    sched._now_utc = lambda: now
    try:
        extra: dict[int, int] = {}
        total = 0
        for i in range(20):
            smart = calculate_smart_schedule(
                plats,
                num_days=num_days,
                user_timezone="America/Chicago",
                day_occupancy=dict(extra),
                random_seed=f"dense-20:slot-{i}",
            )
            assert len(smart) == 4
            for dt in smart.values():
                offset = (dt.astimezone(timezone.utc).date() - now.date()).days
                assert 1 <= offset <= num_days, (i, offset, dt)
                total += 1
            for offset, count in occupancy_from_schedule(smart, now=now).items():
                extra[offset] = extra.get(offset, 0) + count
        assert total == 20 * 4
        assert max(extra.values()) >= 2  # must pack (80 slots / 3 days)
    finally:
        sched._now_utc = old_now


def test_starter_queue_depth_allows_dense_twenty_file_batch():
    """Starter queue must fit a 20-file dense smart batch when otherwise empty."""
    from stages.entitlements import TIER_CONFIG

    assert int(TIER_CONFIG["free"]["queue_depth"]) >= 20


def test_paid_queue_depth_fits_long_horizon_batches():
    """500 over 30d and 5000 over 50d must not be blocked by Pro/Studio queue caps."""
    from stages.entitlements import TIER_CONFIG

    assert int(TIER_CONFIG["creator_pro"]["queue_depth"]) >= 500
    assert int(TIER_CONFIG["studio"]["queue_depth"]) >= 5000
    assert int(TIER_CONFIG["agency"]["queue_depth"]) >= 5000


def test_five_hundred_videos_pack_over_thirty_and_fifty_days():
    """Long-horizon: 500 videos × 4 platforms over 30d and 50d stay in-window."""
    from services.scheduling_preview import occupancy_from_schedule

    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    import core.scheduling as sched

    plats = ["tiktok", "youtube", "instagram", "facebook"]
    old_now = sched._now_utc
    sched._now_utc = lambda: now
    try:
        for num_days in (30, 50):
            extra: dict[int, int] = {}
            total = 0
            for i in range(500):
                smart = calculate_smart_schedule(
                    plats,
                    num_days=num_days,
                    user_timezone="America/Chicago",
                    day_occupancy=dict(extra),
                    random_seed=f"long-500:slot-{i}",
                )
                assert len(smart) == 4
                for dt in smart.values():
                    offset = (dt.astimezone(timezone.utc).date() - now.date()).days
                    assert 1 <= offset <= num_days, (num_days, i, offset)
                    total += 1
                for offset, count in occupancy_from_schedule(smart, now=now).items():
                    extra[offset] = extra.get(offset, 0) + count
            assert total == 500 * 4
            assert extra
            assert all(1 <= d <= num_days for d in extra)
            assert max(extra.values()) >= 1
    finally:
        sched._now_utc = old_now


def test_five_thousand_videos_pack_over_fifty_days():
    """Agency-scale: 5000 videos × 4 platforms over 50 days stay in-window."""
    from services.scheduling_preview import occupancy_from_schedule

    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    import core.scheduling as sched

    plats = ["tiktok", "youtube", "instagram", "facebook"]
    num_days = 50
    old_now = sched._now_utc
    sched._now_utc = lambda: now
    try:
        extra: dict[int, int] = {}
        total = 0
        for i in range(5000):
            smart = calculate_smart_schedule(
                plats,
                num_days=num_days,
                user_timezone="UTC",
                day_occupancy=dict(extra),
                random_seed=f"scale-5k:slot-{i}",
            )
            assert len(smart) == 4
            for dt in smart.values():
                offset = (dt.astimezone(timezone.utc).date() - now.date()).days
                assert 1 <= offset <= num_days, (i, offset)
                total += 1
            for offset, count in occupancy_from_schedule(smart, now=now).items():
                extra[offset] = extra.get(offset, 0) + count
        assert total == 5000 * 4
        # 20k slots / 50 days => packing pressure
        assert max(extra.values()) >= (20000 // 50)
    finally:
        sched._now_utc = old_now


def test_process_before_publish_for_deferred_modes():
    """Documented contract: trim/burn/studio run in process; deferred only publishes."""
    import inspect
    import pathlib

    from services import deferred_publish_schedule as dps
    import worker

    worker_src = pathlib.Path(worker.__file__).read_text(encoding="utf-8")
    assert "apply_youtube_copyright_shorts_after_audio" in worker_src
    assert "ready_to_publish" in worker_src
    assert "platforms_due_for_publish" in inspect.getsource(dps)
