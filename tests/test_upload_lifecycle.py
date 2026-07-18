"""Upload lifecycle contract helpers (unit-level, no DB)."""

import asyncio
from unittest.mock import AsyncMock, patch

from services.upload.status import (
    CANCELLABLE_STATUSES,
    COMPLETE_IDEMPOTENT_STATUSES,
    is_requeueable_upload,
)
from services.upload.schedule_guard import bootstrap_repair_user_schedules
from services.upload_funnel import (
    emit_funnel_terminal_if_needed,
    emit_upload_funnel_event,
    get_upload_funnel_events,
)


class _FakeCtx:
    def __init__(self, state: str, *, emitted: bool = False):
        self.state = state
        self.error_code = None
        self._funnel_terminal_emitted = emitted


def test_complete_idempotent_statuses_include_pipeline_states():
    for st in ("queued", "staged", "processing", "ready_to_publish", "completed", "partial"):
        assert st in COMPLETE_IDEMPOTENT_STATUSES


def test_complete_idempotent_excludes_pending_and_failed():
    assert "pending" not in COMPLETE_IDEMPOTENT_STATUSES
    assert "failed" not in COMPLETE_IDEMPOTENT_STATUSES
    assert "cancelled" not in COMPLETE_IDEMPOTENT_STATUSES


def test_funnel_terminal_emit_once():
    uid = "test-upload-funnel-terminal"
    ctx = _FakeCtx("failed")
    emit_funnel_terminal_if_needed(uid, ctx)
    emit_funnel_terminal_if_needed(uid, ctx)
    events = get_upload_funnel_events(uid)
    terminal = [e for e in events if str(e.get("event", "")).startswith("terminal_")]
    assert len(terminal) == 1
    assert terminal[0]["event"] == "terminal_failed"


def test_funnel_terminal_skips_processing():
    uid = "test-upload-funnel-processing"
    ctx = _FakeCtx("processing")
    emit_funnel_terminal_if_needed(uid, ctx)
    assert get_upload_funnel_events(uid) == []


def test_upload_funnel_emit_and_read():
    uid = "test-upload-funnel-001"
    emit_upload_funnel_event(uid, "presign_ok", {"put_cost": 10})
    emit_upload_funnel_event(uid, "r2_complete", {})
    events = get_upload_funnel_events(uid)
    assert len(events) >= 2
    assert events[-1]["event"] == "r2_complete"
    assert events[0]["event"] == "presign_ok"


def test_is_requeueable_upload_policy():
    assert is_requeueable_upload("pending", None) is True
    assert is_requeueable_upload("pending", "ENQUEUE_FAILED") is True
    assert is_requeueable_upload("pending", "PUBLISH_SLOT_MISSING") is True
    assert is_requeueable_upload("pending", "STUCK_PENDING") is True
    assert is_requeueable_upload("pending", "PLATFORM_AUTH_FAILED") is False
    assert is_requeueable_upload("queued", "ENQUEUE_FAILED") is False
    assert is_requeueable_upload("failed", "ENQUEUE_FAILED") is False


def test_is_retryable_upload_policy():
    from services.upload.status import is_retryable_upload

    assert is_retryable_upload("failed") is True
    assert is_retryable_upload("cancelled") is True
    assert is_retryable_upload("partial", has_failed_platform=True) is True
    assert is_retryable_upload("partial", has_failed_platform=False) is False
    assert is_retryable_upload("pending") is False
    assert is_retryable_upload("processing", stale_processing=True) is True
    assert is_retryable_upload("processing", stale_processing=False) is False
    # Hard-blocks must match classify_retry_error (API returns 409).
    assert is_retryable_upload("failed", error_code="PLATFORM_AUTH_FAILED") is False
    assert is_retryable_upload("failed", error_code="INSUFFICIENT_TOKENS") is False
    assert is_retryable_upload("failed", error_code="SOURCE_NOT_IN_R2") is False
    assert is_retryable_upload("failed", error_code="INTERNAL") is True
    assert is_retryable_upload(
        "partial", error_code="PLATFORM_AUTH_FAILED", has_failed_platform=True
    ) is False


def test_cancellable_statuses_match_cancel_policy():
    assert CANCELLABLE_STATUSES == ("pending", "scheduled", "queued")
    for st in ("staged", "ready_to_publish", "processing", "failed", "completed"):
        assert st not in CANCELLABLE_STATUSES


def test_bootstrap_schedule_repair_soft_fail_sets_error_not_failed():
    """Bootstrap repair must not flip uploads to failed — only annotate errors."""
    fake_conn_executed: list[tuple] = []

    class FakeConn:
        async def fetch(self, *_a, **_k):
            return [
                {
                    "id": "u-soft-1",
                    "user_id": "user-1",
                    "schedule_mode": "smart",
                    "platforms": ["youtube"],
                    "schedule_metadata": None,
                    "scheduled_time": None,
                    "status": "pending",
                }
            ]

        async def execute(self, sql, *args):
            fake_conn_executed.append((sql, args))

    class Ctx:
        async def __aenter__(self):
            return FakeConn()

        async def __aexit__(self, *_):
            return False

    class FakePool:
        def acquire(self):
            return Ctx()

    async def _run():
        with patch(
            "services.upload.schedule_guard.repair_upload_schedule",
            new=AsyncMock(return_value=(False, None, None)),
        ), patch(
            "services.upload.schedule_guard.loud_upload_schedule_failure",
            new=AsyncMock(),
        ):
            return await bootstrap_repair_user_schedules(FakePool(), "user-1", limit=5)

    summary = asyncio.run(_run())
    assert summary == {"repaired": 0, "failed": 1}
    sql_blob = " ".join(sql for sql, _ in fake_conn_executed).lower()
    assert "status = 'failed'" not in sql_blob
    assert "error_code" in sql_blob
