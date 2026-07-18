"""Unit tests for stuck upload fail-fast recovery."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

from services.upload import stuck_recovery as sr
from services.upload.schedule_guard import (
    ERROR_ABANDONED_PENDING,
    ERROR_PUBLISH_SLOT_MISSING,
    ERROR_STUCK_READY_TO_PUBLISH,
)


def _utc(y, m, d, h=12, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


class _FakeConn:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    async def fetch(self, sql: str, *args):
        return self.rows

    async def fetchval(self, sql: str, *args):
        return args[0] if args else True

    async def execute(self, sql: str, *args):
        return "OK"


def test_fail_deferred_batch_without_slot_when_no_next(monkeypatch):
    conn = _FakeConn([])
    mark = AsyncMock()
    loud = AsyncMock()
    repair = AsyncMock(return_value=(False, None, None))
    monkeypatch.setattr(sr, "loud_upload_schedule_failure", loud)
    monkeypatch.setattr(sr, "mark_schedule_incomplete_failed", mark)
    monkeypatch.setattr(sr, "repair_upload_schedule", repair)

    snap = {
        "schedule_mode": "smart",
        "platforms": ["tiktok", "youtube"],
        "schedule_metadata": {},
        "scheduled_time": None,
    }
    results = [{"platform": "tiktok", "success": True}]

    async def _run():
        return await sr.fail_deferred_batch_without_slot(
            conn, "u1", "user1", snap, results, db_pool=None
        )

    assert asyncio.run(_run()) is True
    assert mark.await_args.kwargs.get("error_code") == ERROR_PUBLISH_SLOT_MISSING
    repair.assert_awaited()


def test_fail_deferred_batch_repairs_then_skips_fail(monkeypatch):
    conn = _FakeConn([])
    mark = AsyncMock()
    loud = AsyncMock()
    nxt = _utc(2026, 6, 17, 14)
    repair = AsyncMock(
        return_value=(
            True,
            nxt,
            {"tiktok": "2026-06-15T19:00:00+00:00", "youtube": "2026-06-17T14:00:00+00:00"},
        )
    )
    monkeypatch.setattr(sr, "loud_upload_schedule_failure", loud)
    monkeypatch.setattr(sr, "mark_schedule_incomplete_failed", mark)
    monkeypatch.setattr(sr, "repair_upload_schedule", repair)

    snap = {
        "schedule_mode": "smart",
        "platforms": ["tiktok", "youtube"],
        "schedule_metadata": {"tiktok": "2026-06-15T19:00:00+00:00"},
        "scheduled_time": _utc(2026, 6, 15, 19),
    }
    results = [{"platform": "tiktok", "success": True}]

    async def _run():
        return await sr.fail_deferred_batch_without_slot(
            conn, "u1", "user1", snap, results, db_pool=None
        )

    assert asyncio.run(_run()) is False
    mark.assert_not_awaited()


def test_fail_deferred_batch_skips_when_next_slot_exists(monkeypatch):
    conn = _FakeConn([])
    mark = AsyncMock()
    monkeypatch.setattr(sr, "mark_schedule_incomplete_failed", mark)

    snap = {
        "schedule_mode": "smart",
        "platforms": ["tiktok", "youtube"],
        "schedule_metadata": {
            "tiktok": "2026-06-15T19:00:00+00:00",
            "youtube": "2026-06-17T14:00:00+00:00",
        },
        "scheduled_time": _utc(2026, 6, 15, 19),
    }
    results = [{"platform": "tiktok", "success": True}]

    async def _run():
        return await sr.fail_deferred_batch_without_slot(
            conn, "u1", "user1", snap, results, db_pool=None
        )

    assert asyncio.run(_run()) is False
    mark.assert_not_awaited()


def test_recover_stuck_ready_fails_null_schedule(monkeypatch):
    past = _utc(2026, 1, 1)
    rows = [
        {
            "id": "u-ready-1",
            "user_id": "user-1",
            "schedule_mode": "smart",
            "platforms": ["tiktok"],
            "schedule_metadata": None,
            "scheduled_time": None,
            "platform_results": None,
            "target_accounts": None,
            "processed_assets": None,
            "updated_at": past,
            "created_at": past,
        }
    ]
    conn = _FakeConn(rows)
    repair = AsyncMock(return_value=(False, None, None))
    mark = AsyncMock()
    loud = AsyncMock()
    monkeypatch.setattr(sr, "repair_upload_schedule", repair)
    monkeypatch.setattr(sr, "mark_schedule_incomplete_failed", mark)
    monkeypatch.setattr(sr, "loud_upload_schedule_failure", loud)

    async def _run():
        return await sr.recover_stuck_ready_to_publish(conn, None, dispatch_publish=None, limit=5)

    stats = asyncio.run(_run())
    assert stats["failed"] == 1
    assert mark.await_args.kwargs.get("error_code") == ERROR_PUBLISH_SLOT_MISSING


def test_recover_stuck_ready_immediate_null_schedule_does_not_fail(monkeypatch):
    """Immediate uploads often have null scheduled_time — must not PUBLISH_SLOT_MISSING."""
    recent = datetime.now(timezone.utc) - timedelta(minutes=2)
    rows = [
        {
            "id": "u-ready-imm",
            "user_id": "user-1",
            "schedule_mode": "immediate",
            "platforms": ["tiktok", "youtube"],
            "schedule_metadata": None,
            "scheduled_time": None,
            "platform_results": None,
            "target_accounts": None,
            "processed_assets": {"tiktok": "x", "youtube": "y"},
            "updated_at": recent,
            "created_at": recent,
        }
    ]
    conn = _FakeConn(rows)
    mark = AsyncMock()
    loud = AsyncMock()
    monkeypatch.setattr(sr, "mark_schedule_incomplete_failed", mark)
    monkeypatch.setattr(sr, "loud_upload_schedule_failure", loud)

    async def _run():
        return await sr.recover_stuck_ready_to_publish(
            conn,
            None,
            dispatch_publish=AsyncMock(),
            redispatch_minutes=30,
            fail_minutes=120,
            limit=5,
        )

    stats = asyncio.run(_run())
    assert stats["failed"] == 0
    mark.assert_not_awaited()
    # Fresh immediate row: still inside redispatch window → skipped
    assert stats["skipped"] >= 1


def test_recover_stuck_ready_repairs_incomplete_metadata(monkeypatch):
    past = _utc(2026, 1, 1)
    rows = [
        {
            "id": "u-ready-meta",
            "user_id": "user-1",
            "schedule_mode": "smart",
            "platforms": ["tiktok", "youtube", "instagram", "facebook"],
            "schedule_metadata": {"tiktok": "2026-01-01T12:00:00+00:00"},
            "scheduled_time": past,
            "platform_results": [{"platform": "tiktok", "success": True}],
            "target_accounts": None,
            "processed_assets": None,
            "updated_at": past,
            "created_at": past,
        }
    ]
    conn = _FakeConn(rows)
    nxt = datetime.now(timezone.utc) + timedelta(hours=2)
    repair = AsyncMock(
        return_value=(
            True,
            nxt,
            {
                "tiktok": past.isoformat(),
                "youtube": nxt.isoformat(),
                "instagram": nxt.isoformat(),
                "facebook": nxt.isoformat(),
            },
        )
    )
    mark = AsyncMock()
    loud = AsyncMock()
    monkeypatch.setattr(sr, "repair_upload_schedule", repair)
    monkeypatch.setattr(sr, "mark_schedule_incomplete_failed", mark)
    monkeypatch.setattr(sr, "loud_upload_schedule_failure", loud)

    async def _run():
        return await sr.recover_stuck_ready_to_publish(
            conn, None, dispatch_publish=None, limit=5
        )

    stats = asyncio.run(_run())
    assert stats["failed"] == 0
    assert stats["repaired"] >= 1
    mark.assert_not_awaited()
    repair.assert_awaited()


def test_recover_stuck_ready_fails_after_fail_window(monkeypatch):
    now = datetime.now(timezone.utc)
    due = now - timedelta(minutes=150)
    rows = [
        {
            "id": "u-ready-2",
            "user_id": "user-1",
            "schedule_mode": "scheduled",
            "platforms": ["youtube"],
            "schedule_metadata": None,
            "scheduled_time": due,
            "platform_results": None,
            "target_accounts": None,
            "processed_assets": {"youtube": "x"},
            "updated_at": now - timedelta(minutes=60),
            "created_at": due,
        }
    ]
    conn = _FakeConn(rows)
    mark = AsyncMock()
    loud = AsyncMock()
    monkeypatch.setattr(sr, "mark_schedule_incomplete_failed", mark)
    monkeypatch.setattr(sr, "loud_upload_schedule_failure", loud)

    async def _run():
        return await sr.recover_stuck_ready_to_publish(
            conn,
            None,
            dispatch_publish=AsyncMock(),
            redispatch_minutes=30,
            fail_minutes=120,
            limit=5,
        )

    stats = asyncio.run(_run())
    assert stats["failed"] == 1
    assert mark.await_args.kwargs.get("error_code") == ERROR_STUCK_READY_TO_PUBLISH


def test_recover_abandoned_pending(monkeypatch):
    past = datetime.now(timezone.utc) - timedelta(hours=30)
    rows = [
        {
            "id": "u-pend-1",
            "user_id": "user-1",
            "schedule_mode": "immediate",
            "error_code": None,
            "created_at": past,
        }
    ]
    conn = _FakeConn(rows)
    mark = AsyncMock()
    loud = AsyncMock()
    monkeypatch.setattr(sr, "mark_schedule_incomplete_failed", mark)
    monkeypatch.setattr(sr, "loud_upload_schedule_failure", loud)

    async def _run():
        return await sr.recover_abandoned_pending(conn, None, hours=24, limit=5)

    assert asyncio.run(_run()) == 1
    assert mark.await_args.kwargs.get("error_code") == ERROR_ABANDONED_PENDING
