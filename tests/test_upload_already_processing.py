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


def test_age_window_elapsed_is_reclaimable_even_with_stage():
    """ACTIVE_PIPELINE age expiry still clears the age latch (heartbeat is additive)."""
    old = datetime.now(timezone.utc) - timedelta(minutes=100)
    assert (
        upload_row_indicates_active_pipeline(
            status="processing",
            processing_stage="transcode",
            updated_at=old,
            stale_after_minutes=90,
        )
        is False
    )


def test_heartbeat_owner_concept_independent_of_age_latch():
    """Document contract: live owners must skip even when age latch is false.

    ``_upload_already_processing`` checks owning_worker_id before age; this
    unit locks the age helper so long FFmpeg cannot look reclaimable by age alone.
    """
    from services.upload.orphan_processing import owning_worker_id, pipeline_row_looks_active

    assert pipeline_row_looks_active(status="processing", processing_stage="transcode")
    workers = [
        {
            "worker_id": "w1",
            "status": "alive",
            "active_process_jobs": [{"upload_id": "u-long", "stage": "transcode"}],
            "active_publish_jobs": [],
        }
    ]
    assert owning_worker_id("u-long", workers) == "w1"
    # Age latch would say reclaimable after 100m — ownership still protects.
    old = datetime.now(timezone.utc) - timedelta(minutes=100)
    assert (
        upload_row_indicates_active_pipeline(
            status="processing",
            processing_stage="transcode",
            updated_at=old,
            stale_after_minutes=90,
        )
        is False
    )
