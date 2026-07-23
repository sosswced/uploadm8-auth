"""Heartbeat-orphan processing reclaim helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from services.upload.orphan_processing import (
    classify_orphan_reclaim,
    has_publishable_processed_assets,
    is_heartbeat_orphan_processing,
    orphan_processing_grace_seconds,
    owning_worker_id,
    pipeline_row_looks_active,
    seconds_since_update,
    worker_owns_upload,
)


def _worker(wid: str, status: str, *upload_ids: str) -> dict:
    return {
        "worker_id": wid,
        "status": status,
        "active_process_jobs": [{"upload_id": u, "stage": "transcode"} for u in upload_ids],
        "active_publish_jobs": [],
    }


def test_worker_owns_upload_from_active_jobs():
    w = _worker("w1", "alive", "u-1")
    assert worker_owns_upload(w, "u-1") is True
    assert worker_owns_upload(w, "u-2") is False


def test_owning_worker_id_prefers_alive_or_stale():
    workers = [
        _worker("dead-1", "dead", "u-1"),
        _worker("live-1", "alive", "u-1"),
    ]
    assert owning_worker_id("u-1", workers) == "live-1"
    assert owning_worker_id("u-9", workers) is None


def test_stale_worker_still_counts_as_owner():
    workers = [_worker("stale-1", "stale", "u-1")]
    assert owning_worker_id("u-1", workers, include_stale=True) == "stale-1"
    assert owning_worker_id("u-1", workers, include_stale=False) is None


def test_orphan_when_unowned_past_grace(monkeypatch):
    monkeypatch.setenv("ORPHAN_PROCESSING_GRACE_SEC", "90")
    assert orphan_processing_grace_seconds() == 90
    old = datetime.now(timezone.utc) - timedelta(seconds=120)
    assert (
        is_heartbeat_orphan_processing(
            status="processing",
            processing_stage="transcode",
            updated_at=old,
            upload_id="u-1",
            workers=[_worker("w1", "alive", "other")],
            grace_sec=90,
        )
        is True
    )


def test_not_orphan_within_grace_even_if_unowned(monkeypatch):
    monkeypatch.setenv("ORPHAN_PROCESSING_GRACE_SEC", "90")
    recent = datetime.now(timezone.utc) - timedelta(seconds=30)
    assert (
        is_heartbeat_orphan_processing(
            status="processing",
            processing_stage="transcode",
            updated_at=recent,
            upload_id="u-1",
            workers=[_worker("w1", "alive")],
            grace_sec=90,
        )
        is False
    )


def test_not_orphan_when_alive_owner():
    recent = datetime.now(timezone.utc) - timedelta(minutes=5)
    assert (
        is_heartbeat_orphan_processing(
            status="processing",
            processing_stage="transcode",
            updated_at=recent,
            upload_id="u-1",
            workers=[_worker("w1", "alive", "u-1")],
            grace_sec=90,
        )
        is False
    )


def test_empty_stage_is_not_orphan():
    old = datetime.now(timezone.utc) - timedelta(minutes=10)
    assert (
        is_heartbeat_orphan_processing(
            status="processing",
            processing_stage="",
            updated_at=old,
            upload_id="u-1",
            workers=[],
            grace_sec=90,
        )
        is False
    )


def test_seconds_since_update():
    past = datetime.now(timezone.utc) - timedelta(seconds=42)
    age = seconds_since_update(past)
    assert age is not None
    assert 40 <= age <= 45


def test_pipeline_row_looks_active():
    assert pipeline_row_looks_active(status="processing", processing_stage="transcode")
    assert not pipeline_row_looks_active(status="processing", processing_stage="")
    assert not pipeline_row_looks_active(status="queued", processing_stage="transcode")


def test_has_publishable_processed_assets():
    assert has_publishable_processed_assets({"tiktok": "processed/x/tiktok.mp4"})
    assert not has_publishable_processed_assets({"thumb_tiktok": "t.jpg", "default": "d.mp4"})
    assert not has_publishable_processed_assets({})
    assert not has_publishable_processed_assets(None)


def test_classify_orphan_reclaim_publish_vs_process():
    assert (
        classify_orphan_reclaim({"processed_assets": {"youtube": "processed/y.mp4"}})
        == "publish"
    )
    assert classify_orphan_reclaim({"processed_assets": {}}) == "process"
    assert classify_orphan_reclaim(None) == "process"


def test_empty_fleet_past_grace_is_orphan():
    old = datetime.now(timezone.utc) - timedelta(seconds=120)
    assert (
        is_heartbeat_orphan_processing(
            status="processing",
            processing_stage="transcode",
            updated_at=old,
            upload_id="u-1",
            workers=[],
            grace_sec=90,
        )
        is True
    )
