"""Detect and reclaim processing uploads that no live worker owns.

After a Render recycle / OOM mid-FFmpeg the DB row stays ``status=processing``
with a stage set. Stream reclaim ACKs the Redis message on skip, so without
heartbeat-aware reclaim users wait for ``STALE_PROCESSING_MINUTES``.

Contract:
- Alive/stale heartbeats that list the upload in ``active_process_jobs`` /
  ``active_publish_jobs`` own it — never reclaim while owned.
- Unowned rows past a short grace are orphans.
- Orphans with publishable ``processed_assets`` → ``ready_to_publish`` (never
  re-encode). Mid-pipeline orphans → ``queued`` + checkpoint resume.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger("uploadm8-worker")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default)) or default))
    except (TypeError, ValueError):
        return default


def orphan_processing_grace_seconds() -> int:
    """Seconds after last ``updated_at`` before an unowned processing row is reclaimable.

    Covers claim→first-heartbeat lag and brief GC pauses. Default 90s (alive=30,
    stale<120, heartbeat every 10s).
    """
    return _env_int("ORPHAN_PROCESSING_GRACE_SEC", 90, minimum=15)


def orphan_recovery_interval_seconds() -> int:
    """How often the worker scans for heartbeat-unowned processing rows."""
    return _env_int("ORPHAN_PROCESSING_RECOVERY_INTERVAL_SEC", 45, minimum=15)


def _aware_utc(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    return None


def _job_upload_ids(jobs: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(jobs, list):
        return out
    for item in jobs:
        if isinstance(item, dict):
            uid = str(item.get("upload_id") or "").strip()
            if uid:
                out.append(uid)
        elif item is not None:
            uid = str(item).strip()
            if uid:
                out.append(uid)
    return out


def worker_owns_upload(worker: dict, upload_id: str) -> bool:
    uid = str(upload_id or "").strip()
    if not uid:
        return False
    for jkey in ("active_process_jobs", "active_publish_jobs"):
        if uid in _job_upload_ids(worker.get(jkey)):
            return True
    return False


def owning_worker_id(
    upload_id: str,
    workers: Sequence[dict],
    *,
    include_stale: bool = True,
) -> Optional[str]:
    """Return worker_id if an alive (or stale) heartbeat lists this upload."""
    uid = str(upload_id or "").strip()
    if not uid:
        return None
    allowed = {"alive", "stale"} if include_stale else {"alive"}
    for w in workers or []:
        st = str(w.get("status") or "").lower()
        if st not in allowed:
            continue
        if worker_owns_upload(w, uid):
            wid = str(w.get("worker_id") or "").strip()
            return wid or "unknown"
    return None


def seconds_since_update(updated_at: Any, *, now: Optional[datetime] = None) -> Optional[float]:
    ts = _aware_utc(updated_at)
    if ts is None:
        return None
    clock = now or datetime.now(timezone.utc)
    return max(0.0, (clock - ts).total_seconds())


def has_publishable_processed_assets(processed_assets: Any) -> bool:
    """True when encoded platform videos exist (publish-phase, do not re-FFmpeg)."""
    try:
        from core.helpers import coerce_processed_assets_map

        assets = coerce_processed_assets_map(processed_assets)
    except Exception:
        assets = processed_assets if isinstance(processed_assets, dict) else {}
        if isinstance(processed_assets, str):
            import json

            try:
                assets = json.loads(processed_assets) if processed_assets.strip() else {}
            except Exception:
                assets = {}
    if not isinstance(assets, dict) or not assets:
        return False
    return any(
        k and not str(k).startswith("thumb_") and str(k) != "default" and assets.get(k)
        for k in assets
    )


def is_heartbeat_orphan_processing(
    *,
    status: Any,
    processing_stage: Any,
    updated_at: Any,
    upload_id: str,
    workers: Sequence[dict],
    grace_sec: Optional[int] = None,
    now: Optional[datetime] = None,
) -> bool:
    """True when DB says in-pipeline but no alive/stale worker owns the upload.

    Empty ``processing_stage`` is not an orphan (pre-claim /complete state) —
    those are claimable via ``mark_processing_started`` already.
    """
    if str(status or "").strip().lower() != "processing":
        return False
    if not str(processing_stage or "").strip():
        return False
    if owning_worker_id(upload_id, workers, include_stale=True):
        return False
    grace = orphan_processing_grace_seconds() if grace_sec is None else max(0, int(grace_sec))
    age = seconds_since_update(updated_at, now=now)
    if age is None:
        # No clock → only reclaim when fleet is visible and clearly not owning.
        return bool(workers)
    return age >= float(grace)


def pipeline_row_looks_active(
    *,
    status: Any,
    processing_stage: Any,
) -> bool:
    """Stage-entered processing (independent of age / heartbeat)."""
    if str(status or "").strip().lower() != "processing":
        return False
    return bool(str(processing_stage or "").strip())


async def fetch_fleet_workers(db_pool) -> List[dict]:
    if not db_pool:
        return []
    try:
        from services.worker_fleet_snapshot import fetch_worker_heartbeat_rows

        return await fetch_worker_heartbeat_rows(db_pool)
    except Exception as e:
        logger.debug("orphan_processing fleet fetch skipped: %s", e)
        return []


async def reset_orphan_processing_to_queued(db_pool, upload_id: str) -> bool:
    """Flip unowned ``processing`` → ``queued`` so ``mark_processing_started`` can claim.

    Keeps ``pipeline_resume`` / checkpoint artifacts intact.
    """
    if not db_pool or not upload_id:
        return False
    async with db_pool.acquire() as conn:
        tag = await conn.execute(
            """
            UPDATE uploads
               SET status = 'queued',
                   processing_started_at = NULL,
                   error_code = NULL,
                   error_detail = NULL,
                   updated_at = NOW()
             WHERE id = $1::uuid
               AND status = 'processing'
               AND COALESCE(NULLIF(BTRIM(processing_stage), ''), '') <> ''
            """,
            upload_id,
        )
    return str(tag or "") != "UPDATE 0"


async def reset_orphan_processing_to_ready_to_publish(db_pool, upload_id: str) -> bool:
    """Publish-phase orphan: flip to ``ready_to_publish`` (assets already on R2)."""
    if not db_pool or not upload_id:
        return False
    async with db_pool.acquire() as conn:
        tag = await conn.execute(
            """
            UPDATE uploads
               SET status = 'ready_to_publish',
                   processing_started_at = NULL,
                   ready_to_publish_at = COALESCE(ready_to_publish_at, NOW()),
                   error_code = NULL,
                   error_detail = NULL,
                   updated_at = NOW()
             WHERE id = $1::uuid
               AND status = 'processing'
            """,
            upload_id,
        )
    return str(tag or "") != "UPDATE 0"


async def list_orphan_processing_upload_ids(
    db_pool,
    *,
    workers: Optional[Sequence[dict]] = None,
    limit: int = 20,
    grace_sec: Optional[int] = None,
) -> List[dict]:
    """DB rows that look mid-pipeline but are heartbeat-unowned past grace."""
    if not db_pool:
        return []
    fleet = list(workers) if workers is not None else await fetch_fleet_workers(db_pool)
    grace = orphan_processing_grace_seconds() if grace_sec is None else max(0, int(grace_sec))
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, status, processing_stage, updated_at,
                   processing_started_at, processed_assets, schedule_mode
              FROM uploads
             WHERE status = 'processing'
               AND COALESCE(NULLIF(BTRIM(processing_stage), ''), '') <> ''
               AND COALESCE(updated_at, processing_started_at, created_at)
                   < NOW() - ($1::int * INTERVAL '1 second')
             ORDER BY COALESCE(updated_at, created_at) ASC
             LIMIT $2
            """,
            grace,
            max(1, int(limit)),
        )
    out: List[dict] = []
    for row in rows:
        d = dict(row)
        uid = str(d.get("id") or "")
        if is_heartbeat_orphan_processing(
            status=d.get("status"),
            processing_stage=d.get("processing_stage"),
            updated_at=d.get("updated_at"),
            upload_id=uid,
            workers=fleet,
            grace_sec=grace,
        ):
            out.append(d)
    return out


def classify_orphan_reclaim(upload_row: Optional[dict]) -> str:
    """Return ``publish`` | ``process`` for an orphan processing row."""
    if not upload_row:
        return "process"
    if has_publishable_processed_assets(upload_row.get("processed_assets")):
        return "publish"
    return "process"


async def still_orphan_for_reclaim(
    db_pool,
    upload_id: str,
    *,
    workers: Optional[Sequence[dict]] = None,
    grace_sec: Optional[int] = None,
) -> Tuple[bool, Optional[dict]]:
    """Re-fetch row + fleet; True only if still an unowned orphan past grace."""
    if not db_pool or not upload_id:
        return False, None
    fleet = list(workers) if workers is not None else await fetch_fleet_workers(db_pool)
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, status, processing_stage, updated_at,
                   processing_started_at, processed_assets, schedule_mode
              FROM uploads
             WHERE id = $1::uuid
            """,
            upload_id,
        )
    if not row:
        return False, None
    d = dict(row)
    ok = is_heartbeat_orphan_processing(
        status=d.get("status"),
        processing_stage=d.get("processing_stage"),
        updated_at=d.get("updated_at"),
        upload_id=str(d.get("id") or ""),
        workers=fleet,
        grace_sec=grace_sec,
    )
    return ok, d if ok else None
