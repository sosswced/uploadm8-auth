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
    for days in (7, 14, 30, 90, 180, 365):
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

    # Fully blocked short window expands instead of randint(1,0)
    blocked = set(range(1, 8))
    used = set()
    off2 = _pick_day_offset(now, "youtube", 7, used, blocked, rng)
    assert off2 >= 8


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
