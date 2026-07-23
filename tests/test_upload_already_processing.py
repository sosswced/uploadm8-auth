"""Worker must not skip jobs that only have status=processing (pre-enqueue)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from worker import upload_row_indicates_active_pipeline


def test_status_processing_without_stage_is_not_active():
    assert (
        upload_row_indicates_active_pipeline(
            status="processing",
            processing_stage=None,
            updated_at=datetime.now(timezone.utc),
        )
        is False
    )


def test_status_processing_with_stage_is_active():
    assert (
        upload_row_indicates_active_pipeline(
            status="processing",
            processing_stage="transcode",
            updated_at=datetime.now(timezone.utc),
        )
        is True
    )


def test_stale_active_stage_is_reclaimable():
    old = datetime.now(timezone.utc) - timedelta(minutes=45)
    assert (
        upload_row_indicates_active_pipeline(
            status="processing",
            processing_stage="transcode",
            updated_at=old,
            stale_after_minutes=20,
        )
        is False
    )


def test_claimed_stage_is_active():
    assert (
        upload_row_indicates_active_pipeline(
            status="processing",
            processing_stage="claimed",
            updated_at=datetime.now(timezone.utc),
        )
        is True
    )


def test_long_transcode_still_active_under_default_stale_window(monkeypatch):
    monkeypatch.setenv("ACTIVE_PIPELINE_STALE_MINUTES", "90")
    mid = datetime.now(timezone.utc) - timedelta(minutes=45)
    assert (
        upload_row_indicates_active_pipeline(
            status="processing",
            processing_stage="transcode",
            updated_at=mid,
        )
        is True
    )
