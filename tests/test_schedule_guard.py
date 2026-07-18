"""Schedule guard and auto-retry policy unit tests."""

from datetime import datetime, timezone

from services.retry_policy import (
    auto_retry_backoff_minutes,
    bump_auto_retry_metadata,
    get_auto_retry_count,
    should_auto_retry_upload,
)
from services.upload.schedule_guard import (
    ERROR_SCHEDULE_INCOMPLETE,
    UPLOAD_ERROR_MESSAGES,
    upload_has_schedule,
    validate_presign_schedule,
)


class _PresignData:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_upload_has_schedule_immediate_always_true():
    assert upload_has_schedule({"schedule_mode": "immediate"}) is True


def test_upload_has_schedule_requires_time_for_scheduled():
    assert upload_has_schedule({"schedule_mode": "scheduled", "scheduled_time": None}) is False
    assert upload_has_schedule(
        {"schedule_mode": "scheduled", "scheduled_time": datetime.now(timezone.utc)}
    ) is True


def test_upload_has_schedule_smart_requires_complete_metadata():
    now = datetime.now(timezone.utc)
    # Top-level time alone is not enough for multi-platform smart.
    assert upload_has_schedule(
        {
            "schedule_mode": "smart",
            "scheduled_time": now,
            "platforms": ["tiktok", "youtube"],
            "schedule_metadata": {"tiktok": now.isoformat()},
        }
    ) is False
    assert upload_has_schedule(
        {
            "schedule_mode": "smart",
            "scheduled_time": now,
            "platforms": ["tiktok", "youtube"],
            "schedule_metadata": {
                "tiktok": now.isoformat(),
                "youtube": now.isoformat(),
            },
        }
    ) is True
    assert upload_has_schedule(
        {
            "schedule_mode": "smart",
            "scheduled_time": None,
            "platforms": ["youtube"],
            "schedule_metadata": {"youtube": now.isoformat()},
        }
    ) is False


def test_validate_presign_schedule_rejects_scheduled_without_time():
    import pytest
    from fastapi import HTTPException

    data = _PresignData(schedule_mode="scheduled", platforms=["youtube"], scheduled_time=None)
    with pytest.raises(HTTPException) as exc:
        validate_presign_schedule(data)
    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "schedule_required"


def test_validate_presign_schedule_rejects_smart_without_platforms():
    import pytest
    from fastapi import HTTPException

    data = _PresignData(schedule_mode="smart", platforms=[])
    with pytest.raises(HTTPException) as exc:
        validate_presign_schedule(data)
    assert exc.value.status_code == 400


def test_schedule_incomplete_has_user_message():
    assert ERROR_SCHEDULE_INCOMPLETE in UPLOAD_ERROR_MESSAGES


def test_auto_retry_backoff_escalates():
    assert auto_retry_backoff_minutes(0) == 2
    assert auto_retry_backoff_minutes(1) == 5
    assert auto_retry_backoff_minutes(9) == 15


def test_should_auto_retry_transient_immediate_failed():
    row = {
        "status": "failed",
        "schedule_mode": "immediate",
        "error_code": "INTERNAL",
        "output_artifacts": {},
    }
    assert should_auto_retry_upload(row) is True


def test_should_not_auto_retry_hard_block():
    row = {
        "status": "failed",
        "schedule_mode": "immediate",
        "error_code": "PLATFORM_AUTH_FAILED",
        "output_artifacts": {},
    }
    assert should_auto_retry_upload(row) is False


def test_repair_scheduled_without_time_uses_smart_path():
    """scheduled mode without time should be repairable via smart generation (not hard fail)."""
    from services.upload.schedule_guard import repair_upload_schedule
    import asyncio

    class FakeConn:
        def __init__(self):
            self.updates = []

        async def execute(self, sql, *args):
            self.updates.append((sql, args))

    row = {
        "id": "u-1",
        "user_id": "user-1",
        "schedule_mode": "scheduled",
        "platforms": ["youtube"],
        "scheduled_time": None,
        "schedule_metadata": None,
    }

    async def _run():
        conn = FakeConn()
        from unittest.mock import AsyncMock, patch

        fake_smart = {"youtube": datetime(2026, 6, 15, 18, 0, tzinfo=timezone.utc)}
        with patch(
            "services.upload.schedule_guard.build_smart_schedule_for_upload",
            new=AsyncMock(return_value=fake_smart),
        ):
            return await repair_upload_schedule(conn, row)

    ok, sched, meta = asyncio.run(_run())
    assert ok is True
    assert sched is not None
    assert meta and "youtube" in meta


def test_bump_auto_retry_metadata_increments():
    arts = bump_auto_retry_metadata({"retry": {"count": 1}}, error_code="INTERNAL")
    assert get_auto_retry_count(arts) == 1
    arts2 = bump_auto_retry_metadata(arts, error_code="TIMEOUT")
    assert get_auto_retry_count(arts2) == 2


def test_normalize_platform_list_and_missing_slots():
    from services.upload.schedule_guard import (
        normalize_platform_list,
        smart_metadata_missing_platforms,
    )

    assert normalize_platform_list(["TikTok", "youtube", "TIKTOK", " facebook "]) == [
        "tiktok",
        "youtube",
        "facebook",
    ]
    missing = smart_metadata_missing_platforms(
        ["tiktok", "youtube", "instagram", "facebook"],
        {"tiktok": "2026-06-15T19:00:00+00:00"},
    )
    assert missing == ["youtube", "instagram", "facebook"]


def test_repair_fills_incomplete_smart_metadata():
    """Smart row with scheduled_time but missing platform slots must be repaired."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from services.upload.schedule_guard import repair_upload_schedule

    class FakeConn:
        def __init__(self):
            self.updates = []

        async def execute(self, sql, *args):
            self.updates.append((sql, args))

    row = {
        "id": "u-partial-meta",
        "user_id": "user-1",
        "schedule_mode": "smart",
        "platforms": ["tiktok", "youtube", "instagram", "facebook"],
        "scheduled_time": datetime(2026, 6, 15, 19, 0, tzinfo=timezone.utc),
        "schedule_metadata": {"tiktok": "2026-06-15T19:00:00+00:00"},
    }

    async def _run():
        conn = FakeConn()
        fake_smart = {
            "youtube": datetime(2026, 6, 16, 14, 0, tzinfo=timezone.utc),
            "instagram": datetime(2026, 6, 17, 18, 0, tzinfo=timezone.utc),
            "facebook": datetime(2026, 6, 18, 13, 0, tzinfo=timezone.utc),
        }
        with patch(
            "services.upload.schedule_guard.build_smart_schedule_for_upload",
            new=AsyncMock(return_value=fake_smart),
        ):
            return await repair_upload_schedule(conn, row), conn

    (ok, sched, meta), conn = asyncio.run(_run())
    assert ok is True
    assert sched is not None
    assert set(meta.keys()) == {"tiktok", "youtube", "instagram", "facebook"}
    assert "tiktok" in meta
    assert conn.updates


def test_calculate_smart_schedule_covers_all_four_platforms():
    from core.scheduling import calculate_smart_schedule

    plats = ["tiktok", "YouTube", "instagram", "FACEBOOK"]
    schedule = calculate_smart_schedule(plats, num_days=14, user_timezone="America/Chicago", random_seed="test-all")
    assert set(schedule.keys()) == {"tiktok", "youtube", "instagram", "facebook"}
    assert all(isinstance(v, datetime) for v in schedule.values())
    # Distinct day slots preferred (used_days) — times should differ across platforms
    assert len({v.isoformat() for v in schedule.values()}) >= 3


def test_repair_complete_metadata_null_scheduled_time_derives_anchor():
    """Complete smart metadata + null scheduled_time must not rebuild all slots."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from services.upload.schedule_guard import repair_upload_schedule

    class FakeConn:
        def __init__(self):
            self.updates = []

        async def execute(self, sql, *args):
            self.updates.append((sql, args))

    past = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    future = datetime(2026, 8, 1, 18, 0, tzinfo=timezone.utc)
    row = {
        "id": "u-anchor",
        "user_id": "user-1",
        "schedule_mode": "smart",
        "platforms": ["tiktok", "youtube"],
        "scheduled_time": None,
        "schedule_metadata": {
            "tiktok": past.isoformat(),
            "youtube": future.isoformat(),
        },
        "platform_results": [
            {"platform": "tiktok", "success": True},
            {"platform": "youtube", "success": False},
        ],
    }

    async def _run():
        conn = FakeConn()
        with patch(
            "services.upload.schedule_guard.build_smart_schedule_for_upload",
            new=AsyncMock(side_effect=AssertionError("must not rebuild")),
        ):
            return await repair_upload_schedule(conn, row), conn

    (ok, sched, meta), conn = asyncio.run(_run())
    assert ok is True
    assert sched == future
    assert set(meta.keys()) == {"tiktok", "youtube"}
    assert conn.updates


def test_repair_does_not_rewind_scheduled_time_to_published_platform():
    """Partial publish: anchor must be min of pending slots, not published past slots."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from services.upload.schedule_guard import repair_upload_schedule

    class FakeConn:
        def __init__(self):
            self.updates = []

        async def execute(self, sql, *args):
            self.updates.append((sql, args))

    past = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    future_yt = datetime(2026, 8, 2, 14, 0, tzinfo=timezone.utc)
    future_ig = datetime(2026, 8, 3, 16, 0, tzinfo=timezone.utc)
    row = {
        "id": "u-partial-pub",
        "user_id": "user-1",
        "schedule_mode": "smart",
        "platforms": ["tiktok", "youtube", "instagram"],
        "scheduled_time": past,
        "schedule_metadata": {"tiktok": past.isoformat()},
        "platform_results": [
            {"platform": "tiktok", "success": True},
        ],
    }

    async def _run():
        conn = FakeConn()
        fake_smart = {"youtube": future_yt, "instagram": future_ig}
        with patch(
            "services.upload.schedule_guard.build_smart_schedule_for_upload",
            new=AsyncMock(return_value=fake_smart),
        ):
            return await repair_upload_schedule(conn, row), conn

    (ok, sched, meta), conn = asyncio.run(_run())
    assert ok is True
    assert sched == future_yt
    assert sched > past
    assert set(meta.keys()) == {"tiktok", "youtube", "instagram"}
    assert conn.updates


def test_anchor_platforms_exclude_successful():
    from services.upload.schedule_guard import (
        anchor_platforms_for_scheduled_time,
        successful_platforms_from_results,
    )

    pr = [
        {"platform": "TikTok", "success": True},
        {"platform": "youtube", "status": "published"},
        {"platform": "instagram", "success": False},
    ]
    assert successful_platforms_from_results(pr) == {"tiktok", "youtube"}
    assert anchor_platforms_for_scheduled_time(
        ["tiktok", "youtube", "instagram"], pr
    ) == ["instagram"]


def test_upload_has_schedule_false_for_partial_smart_meta():
    from services.upload.schedule_guard import upload_has_schedule

    assert upload_has_schedule(
        {
            "schedule_mode": "smart",
            "platforms": ["tiktok", "youtube"],
            "scheduled_time": datetime(2026, 6, 15, 19, 0, tzinfo=timezone.utc),
            "schedule_metadata": {"tiktok": "2026-06-15T19:00:00+00:00"},
        }
    ) is False
    assert upload_has_schedule(
        {
            "schedule_mode": "smart",
            "platforms": ["tiktok", "youtube"],
            "scheduled_time": datetime(2026, 6, 15, 19, 0, tzinfo=timezone.utc),
            "schedule_metadata": {
                "tiktok": "2026-06-15T19:00:00+00:00",
                "youtube": "2026-06-16T14:00:00+00:00",
            },
        }
    ) is True


def test_bootstrap_repairs_partial_smart_metadata():
    """Bootstrap must repair smart rows with partial metadata, not only null meta."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from services.upload.schedule_guard import bootstrap_repair_user_schedules

    class FakeConn:
        def __init__(self):
            self.updates = []

        async def fetch(self, sql, *args):
            return [
                {
                    "id": "u-partial-boot",
                    "user_id": "user-1",
                    "schedule_mode": "smart",
                    "platforms": ["tiktok", "youtube", "instagram", "facebook"],
                    "scheduled_time": datetime(2026, 6, 15, 19, 0, tzinfo=timezone.utc),
                    "schedule_metadata": {"tiktok": "2026-06-15T19:00:00+00:00"},
                    "status": "pending",
                }
            ]

        async def execute(self, sql, *args):
            self.updates.append((sql, args))

    class FakePool:
        def acquire(self):
            conn = FakeConn()
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=conn)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

    async def _run():
        pool = FakePool()
        fake_smart = {
            "youtube": datetime(2026, 6, 16, 14, 0, tzinfo=timezone.utc),
            "instagram": datetime(2026, 6, 17, 18, 0, tzinfo=timezone.utc),
            "facebook": datetime(2026, 6, 18, 20, 0, tzinfo=timezone.utc),
        }
        with patch(
            "services.upload.schedule_guard.build_smart_schedule_for_upload",
            new=AsyncMock(return_value=fake_smart),
        ), patch(
            "services.upload.schedule_guard.loud_upload_schedule_failure",
            new=AsyncMock(),
        ):
            return await bootstrap_repair_user_schedules(pool, "user-1", limit=10)

    summary = asyncio.run(_run())
    assert summary["repaired"] == 1
    assert summary["failed"] == 0
