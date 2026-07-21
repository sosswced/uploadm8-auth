"""POST /api/scheduled/{id}/randomize-schedule unit tests."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException


class _FakeConn:
    def __init__(self, row):
        self.row = row
        self.executes = []

    async def fetchval(self, *_a, **_k):
        return 1

    async def fetchrow(self, sql, *args):
        return self.row

    async def execute(self, sql, *args):
        self.executes.append((sql, args))


class _FakePool:
    """Matches acquire_db: await pool.acquire(timeout=...) + release."""

    def __init__(self, conn):
        self._conn = conn

    async def acquire(self, timeout=None):
        return self._conn

    async def release(self, _conn):
        return None


def _user():
    return {"id": str(uuid4()), "billing_user_id": None}


def test_randomize_schedule_one_time_ready_to_publish():
    from routers.scheduled import randomize_schedule
    import core.state

    upload_id = uuid4()
    user = _user()
    row = {
        "id": str(upload_id),
        "status": "ready_to_publish",
        "schedule_mode": "scheduled",
        "platforms": ["tiktok", "youtube"],
        "schedule_metadata": None,
        "scheduled_time": datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        "user_id": user["id"],
    }
    conn = _FakeConn(row)
    pool = _FakePool(conn)
    slot_a = datetime(2026, 7, 22, 18, 15, tzinfo=timezone.utc)
    slot_b = datetime(2026, 7, 25, 20, 0, tzinfo=timezone.utc)

    async def _run():
        with patch.object(core.state, "db_pool", pool), patch(
            "services.upload.schedule_guard.build_smart_schedule_for_upload",
            new=AsyncMock(return_value={"tiktok": slot_a, "youtube": slot_b}),
        ):
            return await randomize_schedule(upload_id, user)

    res = asyncio.run(_run())
    assert res["status"] == "updated"
    assert res["mode"] == "scheduled"
    assert res["scheduled_time"] == slot_a.isoformat()
    assert res["smart_schedule"] is None
    assert any("scheduled_time" in (sql or "").lower() for sql, _ in conn.executes)


def test_randomize_schedule_smart_mode_applies_slots():
    from routers.scheduled import randomize_schedule
    import core.state

    upload_id = uuid4()
    user = _user()
    row = {
        "id": str(upload_id),
        "status": "ready_to_publish",
        "schedule_mode": "smart",
        "platforms": ["tiktok"],
        "schedule_metadata": {"tiktok": "2026-01-01T12:00:00+00:00"},
        "scheduled_time": datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        "user_id": user["id"],
    }
    conn = _FakeConn(row)
    pool = _FakePool(conn)
    slot = datetime(2026, 8, 1, 16, 30, tzinfo=timezone.utc)

    async def _run():
        with patch.object(core.state, "db_pool", pool), patch(
            "services.upload.schedule_guard.build_smart_schedule_for_upload",
            new=AsyncMock(return_value={"tiktok": slot}),
        ), patch(
            "routers.scheduled._apply_smart_schedule",
            new=AsyncMock(),
        ) as apply_mock:
            out = await randomize_schedule(upload_id, user)
            assert apply_mock.await_count == 1
            return out

    res = asyncio.run(_run())
    assert res["mode"] == "smart"
    assert res["smart_schedule"]["tiktok"]
    assert res["scheduled_time"] == slot.isoformat()


def test_randomize_schedule_rejects_completed():
    from routers.scheduled import randomize_schedule
    import core.state

    upload_id = uuid4()
    user = _user()
    row = {
        "id": str(upload_id),
        "status": "completed",
        "schedule_mode": "scheduled",
        "platforms": ["tiktok"],
        "schedule_metadata": None,
        "scheduled_time": datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        "user_id": user["id"],
    }
    conn = _FakeConn(row)
    pool = _FakePool(conn)

    async def _run():
        with patch.object(core.state, "db_pool", pool):
            return await randomize_schedule(upload_id, user)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_run())
    assert exc.value.status_code == 400
