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
