"""HTTP contract tests for upload lifecycle routes (presign, complete, cancel)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import core.state
from app import app
from core.deps import get_current_user

UPLOAD_ID = "0ad8c0a8-a1d1-49dd-841b-1d85630929d6"
USER_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

FAKE_USER = {
    "id": USER_ID,
    "subscription_tier": "creator_pro",
    "billing_user_id": USER_ID,
    "email": "contract@test.uploadm8.com",
}

PRESIGN_BODY = {
    "filename": "20250224_0073_CAM_EVNT.MP4",
    "file_size": 48_000_000,
    "content_type": "video/mp4",
    "platforms": ["tiktok"],
    "has_telemetry": True,
    "schedule_mode": "immediate",
}


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *_args):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


class FakeConn:
    def __init__(self, *, fetchrow=None, execute=None):
        self.fetchrow = AsyncMock(return_value=fetchrow)
        self.execute = AsyncMock(side_effect=execute)


@pytest.fixture(scope="module")
def upload_client():
    async def _fake_user():
        return FAKE_USER

    app.dependency_overrides[get_current_user] = _fake_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_current_user, None)


def test_presign_rejects_unsupported_content_type(upload_client: TestClient):
    core.state.db_pool = FakePool(FakeConn())
    body = {**PRESIGN_BODY, "content_type": "application/pdf"}
    r = upload_client.post("/api/uploads/presign", json=body)
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["detail"]


def test_presign_returns_presigned_url(upload_client: TestClient):
    core.state.db_pool = FakePool(FakeConn())
    pres = {
        "upload_id": UPLOAD_ID,
        "r2_key": f"uploads/{USER_ID}/{UPLOAD_ID}/video.mp4",
        "put_cost": 10,
        "aic_cost": 5,
        "billing_breakdown": {"put": 10},
        "user_prefs": {"auto_thumbnails": True, "auto_captions": False, "ai_hashtags_enabled": True},
        "smart_schedule": None,
        "telemetry_r2_key": f"uploads/{USER_ID}/{UPLOAD_ID}/telemetry.map",
    }
    with patch(
        "routers.uploads_lifecycle.presign_create_upload",
        new=AsyncMock(return_value=pres),
    ), patch(
        "routers.uploads_lifecycle.generate_presigned_upload_url",
        side_effect=lambda key, _ct: f"https://r2.test/{key}?sig=1",
    ), patch(
        "routers.uploads_lifecycle.log_system_event",
        new=AsyncMock(),
    ):
        r = upload_client.post("/api/uploads/presign", json=PRESIGN_BODY)

    assert r.status_code == 200
    data = r.json()
    assert data["upload_id"] == UPLOAD_ID
    assert "presigned_url" in data
    assert data["telemetry_presigned_url"]
    assert data["preferences_applied"]["auto_thumbnails"] is True


def test_complete_enqueues_immediate_upload(upload_client: TestClient):
    core.state.db_pool = FakePool(FakeConn())
    upload_row = {
        "id": UPLOAD_ID,
        "platforms": ["tiktok"],
        "schedule_mode": "immediate",
        "scheduled_time": None,
        "schedule_metadata": None,
    }
    tx = {
        "new_status": "queued",
        "schedule_mode": "immediate",
        "upload": upload_row,
        "user_prefs": {"auto_thumbnails": True},
        "already_completed": False,
    }
    with patch(
        "routers.uploads_lifecycle.complete_upload_transaction",
        new=AsyncMock(return_value=tx),
    ), patch(
        "routers.uploads_lifecycle.enqueue_job",
        new=AsyncMock(return_value=True),
    ), patch(
        "routers.uploads_lifecycle.inline_rescue_if_stuck",
        new=AsyncMock(),
    ), patch(
        "routers.uploads_lifecycle.log_system_event",
        new=AsyncMock(),
    ):
        r = upload_client.post(f"/api/uploads/{UPLOAD_ID}/complete", json={})

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    assert body["upload_id"] == UPLOAD_ID
    assert body["schedule_mode"] == "immediate"


def test_complete_idempotent_when_already_queued(upload_client: TestClient):
    core.state.db_pool = FakePool(FakeConn())
    upload_row = {
        "id": UPLOAD_ID,
        "platforms": ["tiktok"],
        "schedule_mode": "immediate",
        "scheduled_time": None,
        "schedule_metadata": None,
    }
    tx = {
        "new_status": "queued",
        "schedule_mode": "immediate",
        "upload": upload_row,
        "user_prefs": {},
        "already_completed": True,
    }
    with patch(
        "routers.uploads_lifecycle.complete_upload_transaction",
        new=AsyncMock(return_value=tx),
    ), patch(
        "routers.uploads_lifecycle.enqueue_job",
        new=AsyncMock(),
    ) as mock_enqueue:
        r = upload_client.post(f"/api/uploads/{UPLOAD_ID}/complete", json={})

    assert r.status_code == 200
    assert r.json()["already_completed"] is True
    mock_enqueue.assert_not_called()


def test_cancel_pending_upload_refunds_and_cancels(upload_client: TestClient):
    conn = FakeConn(
        fetchrow={
            "put_reserved": 12,
            "aic_reserved": 3,
            "status": "pending",
            "r2_key": "uploads/u1/video.mp4",
            "telemetry_r2_key": None,
            "processed_r2_key": None,
            "thumbnail_r2_key": None,
        }
    )
    core.state.db_pool = FakePool(conn)
    with patch(
        "routers.uploads_lifecycle.refund_tokens",
        new=AsyncMock(),
    ) as mock_refund, patch(
        "routers.uploads_lifecycle.clear_cancel_signal",
        new=AsyncMock(),
    ), patch(
        "routers.uploads_lifecycle.log_system_event",
        new=AsyncMock(),
    ):
        r = upload_client.post(f"/api/uploads/{UPLOAD_ID}/cancel")

    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    mock_refund.assert_awaited_once()


def test_cancel_rejects_terminal_upload(upload_client: TestClient):
    conn = FakeConn(
        fetchrow={
            "put_reserved": 0,
            "aic_reserved": 0,
            "status": "completed",
            "r2_key": "uploads/u1/video.mp4",
            "telemetry_r2_key": None,
            "processed_r2_key": None,
            "thumbnail_r2_key": None,
        }
    )
    core.state.db_pool = FakePool(conn)
    r = upload_client.post(f"/api/uploads/{UPLOAD_ID}/cancel")
    assert r.status_code == 400
    assert "Cannot cancel" in r.json()["detail"]


def test_presign_then_complete_lifecycle_chain(upload_client: TestClient):
    """Presign → complete for immediate upload enqueues worker job once."""
    core.state.db_pool = FakePool(FakeConn())
    pres = {
        "upload_id": UPLOAD_ID,
        "r2_key": f"uploads/{USER_ID}/{UPLOAD_ID}/video.mp4",
        "put_cost": 8,
        "aic_cost": 2,
        "billing_breakdown": {},
        "user_prefs": {"auto_thumbnails": True, "auto_captions": True, "ai_hashtags_enabled": False},
        "smart_schedule": None,
        "telemetry_r2_key": f"uploads/{USER_ID}/{UPLOAD_ID}/telemetry.map",
    }
    upload_row = {
        "id": UPLOAD_ID,
        "platforms": ["tiktok"],
        "schedule_mode": "immediate",
        "scheduled_time": datetime.now(timezone.utc),
        "schedule_metadata": None,
    }
    tx = {
        "new_status": "queued",
        "schedule_mode": "immediate",
        "upload": upload_row,
        "user_prefs": pres["user_prefs"],
        "already_completed": False,
    }

    with patch(
        "routers.uploads_lifecycle.presign_create_upload",
        new=AsyncMock(return_value=pres),
    ), patch(
        "routers.uploads_lifecycle.generate_presigned_upload_url",
        side_effect=lambda key, _ct: f"https://r2.test/{key}?sig=1",
    ), patch(
        "routers.uploads_lifecycle.complete_upload_transaction",
        new=AsyncMock(return_value=tx),
    ), patch(
        "routers.uploads_lifecycle.enqueue_job",
        new=AsyncMock(return_value=True),
    ) as mock_enqueue, patch(
        "routers.uploads_lifecycle.inline_rescue_if_stuck",
        new=AsyncMock(),
    ), patch(
        "routers.uploads_lifecycle.log_system_event",
        new=AsyncMock(),
    ):
        presign = upload_client.post("/api/uploads/presign", json=PRESIGN_BODY)
        assert presign.status_code == 200
        uid = presign.json()["upload_id"]
        complete = upload_client.post(f"/api/uploads/{uid}/complete", json={"title": "Contract test"})

    assert complete.status_code == 200
    assert complete.json()["status"] == "queued"
    mock_enqueue.assert_awaited_once()
    job = mock_enqueue.await_args.args[0]
    assert job["upload_id"] == UPLOAD_ID
    assert job["user_id"] == USER_ID
