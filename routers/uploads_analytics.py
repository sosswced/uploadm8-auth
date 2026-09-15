"""Upload analytics and thumbnail sync HTTP routes (thin).

Business logic lives in services.upload_analytics_sync (engagement writers,
TikTok hydrate/backfill, Meta/TikTok/YouTube fetch orchestration).
"""

import asyncio
import logging

import asyncpg
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

import core.state
from core.deps import get_current_user
from services.platform_posted_thumbnails import (
    background_sync_posted_thumbnails,
    upload_ids_needing_posted_thumbnail_sync,
)
from services.upload_analytics_sync import (
    _background_sync_uploads_analytics,
    _sync_upload_analytics_core,
)
from services.uploads_handlers import poll_upload_thumbnails_payload
from services.workspace import resolve_billing_user_id

logger = logging.getLogger("uploadm8-api")

router = APIRouter(prefix="/api/uploads", tags=["uploads"])


async def _background_sync_uploads_thumbnails(user_id: str, upload_ids: list[str]) -> None:
    await background_sync_posted_thumbnails(core.state.db_pool, user_id, upload_ids)


@router.post("/sync-analytics/all")
async def sync_all_upload_analytics(
    background_tasks: BackgroundTasks,
    max_uploads: int = Query(800, ge=1, le=2000),
    async_mode: bool = Query(True),
    user: dict = Depends(get_current_user),
):
    """Batch engagement sync for many completed uploads."""
    uid = resolve_billing_user_id(user)
    try:
        async with core.state.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id FROM uploads
                WHERE user_id = $1::uuid
                  AND status = ANY($2::varchar[])
                  AND platform_results IS NOT NULL
                  AND platform_results::text NOT IN ('null', '[]', '{}')
                ORDER BY analytics_synced_at ASC NULLS FIRST, created_at DESC
                LIMIT $3
                """,
                uid,
                ["completed", "succeeded", "partial"],
                max_uploads,
            )
    except (TimeoutError, OSError, asyncio.TimeoutError, asyncpg.PostgresConnectionError) as e:
        logger.warning("sync-analytics/all: database unavailable user=%s: %s", uid[:8], e)
        raise HTTPException(503, "Analytics sync temporarily unavailable — try again shortly") from e
    ids = [str(r["id"]) for r in rows]

    if async_mode:
        background_tasks.add_task(_background_sync_uploads_analytics, uid, ids)
        return {"ok": True, "queued": len(ids), "async_mode": True}

    synced = 0
    for up_id in ids:
        try:
            await _sync_upload_analytics_core(user, up_id)
            synced += 1
        except HTTPException:
            pass
        await asyncio.sleep(0.25)
    try:
        from services.ml_scoring_job import maybe_recompute_quality_after_analytics_sync

        await maybe_recompute_quality_after_analytics_sync(core.state.db_pool, uid)
    except Exception as e:
        logger.warning("sync-analytics/all sync quality recompute user=%s: %s", str(uid)[:8], e)
    return {"ok": True, "candidates": len(ids), "synced": synced, "async_mode": False}


@router.post("/sync-thumbnails/all")
async def sync_all_upload_thumbnails(
    background_tasks: BackgroundTasks,
    max_uploads: int = Query(120, ge=1, le=400),
    async_mode: bool = Query(True),
    user: dict = Depends(get_current_user),
):
    """Queue live platform cover fetch for completed uploads."""
    uid = resolve_billing_user_id(user)
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, status, platform_results, thumbnail_r2_key, output_artifacts, platforms
            FROM uploads
            WHERE user_id = $1::uuid
              AND status = ANY($2::varchar[])
              AND platform_results IS NOT NULL
              AND platform_results::text NOT IN ('null', '[]', '{}')
            ORDER BY updated_at DESC
            LIMIT $3
            """,
            uid,
            ["completed", "succeeded", "partial"],
            max_uploads,
        )
    ids = [str(r["id"]) for r in rows if upload_ids_needing_posted_thumbnail_sync(dict(r))]

    if async_mode:
        if ids:
            background_tasks.add_task(_background_sync_uploads_thumbnails, uid, ids)
        return {"ok": True, "queued": len(ids), "async_mode": True}

    await background_sync_posted_thumbnails(core.state.db_pool, uid, ids)
    return {"ok": True, "synced": len(ids), "async_mode": False}


@router.get("/thumbnails/poll")
async def poll_upload_thumbnails(
    ids: str = Query(..., description="Comma-separated upload UUIDs (max 40)"),
    user: dict = Depends(get_current_user),
):
    """Fast DB-only read of thumbnail URLs for visible rows."""
    raw = [x.strip() for x in (ids or "").split(",") if x.strip()]
    upload_ids = raw[:40]
    if not upload_ids:
        return {"thumbnails": {}}
    payload = await poll_upload_thumbnails_payload(
        core.state.db_pool, resolve_billing_user_id(user), upload_ids
    )
    return {"thumbnails": payload}


@router.post("/{upload_id}/sync-analytics")
async def sync_upload_analytics(upload_id: str, user: dict = Depends(get_current_user)):
    """Fetch latest engagement stats for a single completed upload from platform APIs."""
    return await _sync_upload_analytics_core(user, upload_id)
