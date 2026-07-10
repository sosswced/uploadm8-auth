"""Upload read/update routes: list, detail, thumbnails, queue stats."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, Response

import core.state
from core.audit import log_system_event
from core.config import R2_BUCKET_NAME
from core.deps import get_current_user, get_current_user_readonly, get_verified_user_id
from core.models import UploadUpdate
from core.r2 import (
    _normalize_r2_key,
    generate_presigned_upload_url,
    get_s3_client,
    r2_object_exists,
)
from core.wallet import refund_tokens
from services.uploads_api import update_upload_metadata
from services.thumbnail_regenerate import regenerate_upload_thumbnail, should_skip_regenerate
from services.uploads_handlers import (
    collect_thumbnail_repair_ids,
    fetch_upload_detail,
    fetch_upload_queue_stats,
    fetch_user_uploads_list,
    posted_thumbnail_fallback_for_upload,
    repair_upload_thumbnails_batch,
    stream_upload_thumbnail_bytes,
)

logger = logging.getLogger("uploadm8-api")

router = APIRouter(prefix="/api/uploads", tags=["uploads"])


def _schedule_thumbnail_repair(background_tasks: BackgroundTasks, user_id: str, payload: Any) -> None:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("uploads"), list):
        items = payload["uploads"]
    else:
        items = []
    repair_ids = collect_thumbnail_repair_ids(items)
    if repair_ids and core.state.db_pool:
        background_tasks.add_task(
            repair_upload_thumbnails_batch,
            core.state.db_pool,
            user_id,
            repair_ids,
        )


@router.get("")
async def get_uploads(
    background_tasks: BackgroundTasks,
    status: Optional[str] = None,
    view: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    trill_only: bool = False,
    meta: bool = False,
    sort: Optional[str] = Query(None, pattern="^(created_at|views|engagement)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    days: Optional[int] = Query(None, ge=1, le=3650),
    slim: bool = False,
    user: dict = Depends(get_current_user_readonly),
):
    """Upload queue list for current user."""
    uid = str(user.get("billing_user_id") or user["id"])
    since = None
    if days is not None:
        since = datetime.now(timezone.utc) - timedelta(days=days)
    effective_limit = min(limit, 200) if sort in ("views", "engagement") else limit
    payload = await fetch_user_uploads_list(
        core.state.db_pool,
        uid,
        status=status,
        view=view,
        limit=effective_limit,
        offset=offset,
        trill_only=trill_only,
        meta=meta,
        workspace_id=(user.get("workspace") or {}).get("id"),
        sort=sort,
        order=order,
        since=since,
        slim=slim,
    )
    if not slim:
        _schedule_thumbnail_repair(background_tasks, uid, payload)
    return payload


@router.get("/queue-stats")
async def get_uploads_queue_stats(user: dict = Depends(get_current_user)):
    """Queue summary counts for queue.html and dashboard.html."""
    return await fetch_upload_queue_stats(core.state.db_pool, str(user["id"]))


@router.get("/{upload_id}/thumbnail")
async def get_upload_thumbnail(
    upload_id: str,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user_readonly),
):
    """Stream the upload card thumbnail from R2."""
    uid = str(user.get("billing_user_id") or user["id"])
    pool = core.state.db_pool
    try:
        raw, content_type, etag_key = await stream_upload_thumbnail_bytes(pool, uid, upload_id)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        fallback = None
        if pool:
            fallback = await posted_thumbnail_fallback_for_upload(pool, uid, upload_id)
            background_tasks.add_task(
                repair_upload_thumbnails_batch,
                pool,
                uid,
                [upload_id],
            )
        if fallback:
            return RedirectResponse(url=fallback, status_code=302)
        raise
    return Response(
        content=raw,
        media_type=content_type,
        headers={
            "Cache-Control": "public, max-age=3600",
            "ETag": f'"{etag_key}"',
        },
    )


@router.post("/{upload_id}/generate-thumbnail")
async def generate_thumbnail_for_upload(
    upload_id: str,
    force: bool = Query(False, description="Regenerate even when a thumbnail already exists"),
    user: dict = Depends(get_current_user),
):
    """Backfill / regenerate the thumbnail for an existing upload."""
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT u.id, u.r2_key, u.processed_r2_key, u.thumbnail_r2_key, u.status, u.platforms,
                   u.title, u.caption, u.ai_title, u.ai_caption, u.ai_generated_title, u.ai_generated_caption,
                   u.user_preferences, u.output_artifacts, u.platform_results,
                   usr.subscription_tier AS ent_subscription_tier,
                   usr.role AS ent_role,
                   usr.flex_enabled AS ent_flex_enabled
            FROM uploads u
            JOIN users usr ON usr.id = u.user_id
            WHERE u.id = $1 AND u.user_id = $2
            """,
            upload_id,
            user["id"],
        )
    if not row:
        raise HTTPException(404, "Upload not found")

    if should_skip_regenerate(thumbnail_r2_key=row.get("thumbnail_r2_key"), force=force):
        tk = row.get("thumbnail_r2_key")
        if tk and await asyncio.to_thread(r2_object_exists, str(tk)):
            try:
                s3 = get_s3_client()
                url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": R2_BUCKET_NAME, "Key": _normalize_r2_key(row["thumbnail_r2_key"])},
                    ExpiresIn=3600,
                )
                return {"thumbnail_url": url, "r2_key": row["thumbnail_r2_key"], "generated": False}
            except Exception:
                pass

    upload_dict = {
        "id": row["id"],
        "r2_key": row["r2_key"],
        "processed_r2_key": row["processed_r2_key"],
        "thumbnail_r2_key": row["thumbnail_r2_key"],
        "status": row["status"],
        "platforms": row["platforms"],
        "title": row["title"],
        "caption": row.get("caption"),
        "ai_title": row["ai_title"],
        "ai_caption": row.get("ai_caption"),
        "ai_generated_title": row["ai_generated_title"],
        "ai_generated_caption": row.get("ai_generated_caption"),
        "user_preferences": row["user_preferences"],
        "output_artifacts": row.get("output_artifacts"),
        "platform_results": row.get("platform_results"),
    }
    user_dict = {
        "subscription_tier": row["ent_subscription_tier"],
        "role": row["ent_role"],
        "flex_enabled": row["ent_flex_enabled"],
    }
    try:
        out = await regenerate_upload_thumbnail(
            db_pool=core.state.db_pool,
            upload_id=upload_id,
            user_id=str(user["id"]),
            upload_row=upload_dict,
            user_row=user_dict,
            force=force,
        )
        return out
    except ValueError as e:
        code = str(e)
        if code == "no_video_key":
            raise HTTPException(400, "No video file key found for this upload") from e
        if code == "video_not_in_storage":
            raise HTTPException(
                404,
                "Video file is not in storage. It may have been deleted or the upload never completed.",
            ) from e
        if code == "ffmpeg_failed":
            raise HTTPException(500, "Thumbnail extraction produced no output") from e
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Thumbnail generation failed: {e}") from e


@router.post("/{upload_id}/thumbnail-presign")
async def presign_thumbnail_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """Get a presigned URL for uploading a custom thumbnail."""
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, status FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id,
            user["id"],
        )
    if not row:
        raise HTTPException(404, "Upload not found")
    editable = ("pending", "scheduled", "queued", "staged", "ready_to_publish")
    if row["status"] not in editable:
        raise HTTPException(400, "Cannot change thumbnail after upload is processing or published")

    thumb_r2_key = f"thumbnails/{user['id']}/{upload_id}/custom.jpg"
    presigned_url = generate_presigned_upload_url(thumb_r2_key, "image/jpeg")
    return {"presigned_url": presigned_url, "r2_key": thumb_r2_key}


@router.get("/{upload_id}")
async def get_upload_details(upload_id: str, user_id: str = Depends(get_verified_user_id)):
    """Upload detail for current user."""
    return await fetch_upload_detail(core.state.db_pool, upload_id, user_id)


@router.post("/{upload_id}/ask")
async def ask_upload(
    upload_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """
    Answer a question about this upload using hydration evidence only.

    Body: ``{"question": "..."}``. Citations are ``evidence_ids`` from persisted
    artifacts (place, shot list, claims, hydration). No ungrounded free-form invent.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    question = str(body.get("question") or body.get("q") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    if len(question) > 2000:
        raise HTTPException(400, "question too long")

    from services.upload_qa import ask_upload_question

    uid = str(user.get("billing_user_id") or user["id"])
    result = await ask_upload_question(core.state.db_pool, upload_id, uid, question)
    if result.get("status") == "not_found":
        raise HTTPException(404, "Upload not found")
    return result


@router.patch("/{upload_id}")
async def update_upload(
    upload_id: str,
    update_data: UploadUpdate,
    user: dict = Depends(get_current_user),
):
    """Update an upload's metadata: title, caption, hashtags, scheduled_time, smart_schedule."""
    async with core.state.db_pool.acquire() as conn:
        await update_upload_metadata(conn, upload_id, user["id"], update_data)
    return {"status": "updated", "id": upload_id}


@router.delete("/{upload_id}")
async def delete_upload(upload_id: str, request: Request, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT put_reserved, aic_reserved, status, title, platforms FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id,
            user["id"],
        )
        if not upload:
            raise HTTPException(404, "Upload not found")
        if upload["status"] in ("pending", "queued"):
            await refund_tokens(conn, user["id"], upload["put_reserved"], upload["aic_reserved"], upload_id)
        await conn.execute("DELETE FROM uploads WHERE id = $1", upload_id)
        await log_system_event(
            conn,
            user_id=str(user["id"]),
            action="UPLOAD_DELETED",
            event_category="UPLOAD",
            resource_type="upload",
            resource_id=upload_id,
            details={
                "title": upload["title"],
                "status_at_delete": upload["status"],
                "platforms": list(upload["platforms"] or []),
            },
            request=request,
            severity="WARNING",
        )
    return {"status": "deleted"}
