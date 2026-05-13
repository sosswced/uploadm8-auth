"""
UploadM8 Upload routes -- extracted from app.py.

Handles upload lifecycle: presign, complete, cancel, retry, list, update,
thumbnail generation, analytics sync, and single-upload detail.
"""

import json
import logging
import os
import uuid
from typing import Dict, List, Optional, Tuple

import asyncio

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

import core.state
from core.audit import log_system_event
from core.auth import decrypt_blob
from core.cancel_signal import clear_cancel_signal, signal_cancel
from core.config import R2_BUCKET_NAME
from core.deps import get_current_user, get_verified_user_id
from core.helpers import _safe_json, get_plan
from core.models import CompleteUploadBody, UploadInit, UploadUpdate
from core.queue import enqueue_job
from core.r2 import (
    _delete_r2_objects,
    _normalize_r2_key,
    generate_presigned_download_url,
    generate_presigned_upload_url,
    get_s3_client,
    r2_object_exists,
)
from services.smart_schedule_insights import calculate_smart_schedule_data_driven
from core.wallet import partial_refund_tokens, refund_tokens
from routers.preferences import get_user_prefs_for_upload
from stages.entitlements import get_entitlements_for_tier
from services.platform_oauth_refresh import refresh_decrypted_token_for_row
from services.sync_analytics_helpers import resolve_token_candidates_for_platform_result
from services.uploads_api import update_upload_metadata
from services.thumbnail_regenerate import (
    regenerate_upload_thumbnail,
    should_skip_regenerate,
)
from services.uploads_handlers import (
    ALLOWED_VIDEO_TYPES,
    complete_upload_transaction,
    compute_smart_schedule_display,
    fetch_upload_detail,
    fetch_upload_queue_stats,
    fetch_user_uploads_list,
    presign_create_upload,
)
from services.retry_policy import (
    MAX_USER_RETRIES_DEFAULT,
    RETRY_IDEMPOTENCY_TTL_SEC,
    bump_retry_metadata,
    classify_retry_error,
    get_retry_count,
    split_platform_results,
)

logger = logging.getLogger("uploadm8-api")

router = APIRouter(prefix="/api/uploads", tags=["uploads"])


# ============================================================
# Routes
# ============================================================

_INLINE_RESCUE_ENABLED = (os.environ.get("UPLOAD_INLINE_RESCUE_ENABLED", "true").lower() in ("1", "true", "yes", "on"))
_INLINE_RESCUE_DELAY_SEC = max(30, int(os.environ.get("UPLOAD_INLINE_RESCUE_DELAY_SEC", "150") or 150))


async def _inline_rescue_if_stuck(upload_id: str, user_id: str, user_prefs: Dict[str, object], ent) -> None:
    """
    Safety net for local/single-process setups:
    if a just-completed immediate upload remains at stage `upload` for too long,
    run the processing pipeline inline from API so thumbnails/persona are still produced.
    """
    if not _INLINE_RESCUE_ENABLED:
        return
    if core.state.db_pool is None:
        return
    await asyncio.sleep(float(_INLINE_RESCUE_DELAY_SEC))
    try:
        async with core.state.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, status, processing_stage, processing_progress, updated_at,
                       processing_started_at, thumbnail_r2_key, output_artifacts
                FROM uploads
                WHERE id = $1::uuid AND user_id = $2::uuid
                """,
                upload_id,
                user_id,
            )
        if not row:
            return
        status = str(row.get("status") or "").lower()
        stage = str(row.get("processing_stage") or "").lower()
        progress = int(row.get("processing_progress") or 0)
        thumb_key = str(row.get("thumbnail_r2_key") or "").strip()
        arts = _safe_json(row.get("output_artifacts"), {}) or {}
        # Bail if pipeline moved forward or produced artifacts.
        if thumb_key:
            return
        if isinstance(arts, dict) and (
            arts.get("thumbnail_trace")
            or arts.get("pikzels_prompt_by_platform")
            or arts.get("hydration_payload")
        ):
            return
        if status not in ("queued", "processing", "staged", "pending"):
            return
        if stage not in ("", "upload"):
            return
        if progress not in (0, 87):
            return

        logger.warning(
            "[%s] inline-rescue: stuck at stage=%s progress=%s status=%s after %ss — running pipeline inline",
            upload_id, stage or "-", progress, status, _INLINE_RESCUE_DELAY_SEC,
        )
        import worker as worker_runtime

        # Reuse API pools so inline run has DB/Redis handles.
        worker_runtime.db_pool = core.state.db_pool
        worker_runtime.redis_client = core.state.redis_client

        job_data = {
            "upload_id": upload_id,
            "user_id": user_id,
            "preferences": user_prefs or {},
            "priority_class": getattr(ent, "priority_class", "normal"),
            "plan_features": {
                "ai": getattr(ent, "can_ai", True),
                "priority": getattr(ent, "can_priority", False),
                "watermark": getattr(ent, "can_watermark", True),
                "ai_depth": getattr(ent, "ai_depth", "standard"),
                "caption_frames": getattr(ent, "max_caption_frames", 6),
            },
        }
        ok = await worker_runtime.run_processing_pipeline(job_data)
        logger.warning("[%s] inline-rescue pipeline finished ok=%s", upload_id, bool(ok))
    except Exception as exc:
        logger.exception("[%s] inline-rescue failed: %s", upload_id, exc)

@router.post("/presign")
async def presign_upload(data: UploadInit, request: Request, user: dict = Depends(get_current_user)):
    """Create upload with user preferences applied"""
    if core.state.db_pool is None:
        raise HTTPException(503, detail="Database unavailable")

    if data.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(400, detail=f"Unsupported file type: {data.content_type}")

    async with core.state.db_pool.acquire() as conn:
        pres = await presign_create_upload(conn, data, user)
    upload_id = pres["upload_id"]
    r2_key = pres["r2_key"]
    put_cost = pres["put_cost"]
    aic_cost = pres["aic_cost"]
    user_prefs = pres["user_prefs"]
    smart_schedule = pres["smart_schedule"]
    telemetry_r2_key = pres.get("telemetry_r2_key")

    try:
        presigned_url = generate_presigned_upload_url(r2_key, data.content_type)
        result = {
            "upload_id": upload_id,
            "presigned_url": presigned_url,
            "r2_key": r2_key,
            "put_cost": put_cost,
            "aic_cost": aic_cost,
            "schedule_mode": data.schedule_mode,
            "target_accounts": data.target_accounts or [],
            "preferences_applied": {
                "auto_captions": bool(user_prefs.get("auto_captions")),
                "auto_thumbnails": bool(user_prefs.get("auto_thumbnails")),
                "ai_hashtags": bool(user_prefs.get("ai_hashtags_enabled")),
            },
        }

        if smart_schedule:
            result["smart_schedule"] = {
                p: (v.isoformat() if hasattr(v, "isoformat") else str(v)) for p, v in smart_schedule.items()
            }

        if telemetry_r2_key:
            result["telemetry_presigned_url"] = generate_presigned_upload_url(
                telemetry_r2_key,
                "application/octet-stream",
            )
            result["telemetry_r2_key"] = telemetry_r2_key
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(
            "presign URL generation failed upload_id=%s user=%s: %s",
            upload_id,
            user.get("id"),
            e,
            exc_info=True,
        )
        try:
            async with core.state.db_pool.acquire() as conn:
                await refund_tokens(conn, str(user["id"]), put_cost, aic_cost, upload_id)
                await conn.execute("DELETE FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user["id"])
        except Exception as re:
            logger.error("presign rollback failed upload_id=%s: %s", upload_id, re, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "storage_presign_failed",
                "message": "Could not generate upload URL. Check R2 credentials (R2_ACCOUNT_ID, keys, bucket) on the API server.",
            },
        ) from e

    # Fire-and-forget audit -- does not affect upload flow
    async with core.state.db_pool.acquire() as _ac:
        await log_system_event(_ac, user_id=str(user["id"]), action="UPLOAD_INITIATED",
                               event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                               details={"filename": data.filename, "platforms": list(data.platforms or []),
                                        "schedule_mode": data.schedule_mode, "put_cost": put_cost,
                                        "aic_cost": aic_cost, "file_size": data.file_size},
                               request=request)

    return result


@router.post("/smart-schedule/preview")
async def preview_smart_schedule(platforms: List[str] = Query(...), days: int = Query(7), user: dict = Depends(get_current_user)):
    """Preview smart schedule times using fleet + your historical engagement (UTC)."""
    if not platforms:
        raise HTTPException(400, "At least one platform required")

    async with core.state.db_pool.acquire() as conn:
        schedule = await calculate_smart_schedule_data_driven(
            conn, str(user["id"]), platforms, num_days=days, blocked_day_offsets=None
        )

    return {
        "schedule": {p: dt.isoformat() for p, dt in schedule.items()},
        "explanation": {
            p: {
                "date": dt.strftime("%A, %B %d"),
                "time": dt.strftime("%I:%M %p"),
                "reason": (
                    f"Data-informed UTC slot for {p.title()} "
                    "(upload engagement by hour + momentum, blended with defaults)"
                ),
            }
            for p, dt in schedule.items()
        },
    }


@router.post("/{upload_id}/complete")
async def complete_upload(
    upload_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """
    Complete upload and either enqueue immediately (immediate mode) or stage
    for deferred processing (scheduled / smart mode).

    IMMEDIATE  -> status=queued, pushed to Redis -> worker fires NOW
    SCHEDULED  -> status=staged, NOT pushed to Redis -> scheduler fires at scheduled_time - processing_window
    SMART      -> status=staged, NOT pushed to Redis -> scheduler fires at first scheduled_time - processing_window

    Request body may include title, caption, hashtags from the upload page (manual metadata for single-file uploads).
    These override presign defaults and are stored before enqueue.
    """
    body = {}
    try:
        raw = await request.body()
        if raw:
            body = json.loads(raw) or {}
    except Exception:
        pass

    async with core.state.db_pool.acquire() as conn:
        tx = await complete_upload_transaction(conn, upload_id, str(user["id"]), body)

    new_status = tx["new_status"]
    schedule_mode = tx["schedule_mode"]
    upload = tx["upload"]
    user_prefs = tx["user_prefs"]

    # Resolve full entitlements -- drives queue routing, AI depth, priority class
    ent = get_entitlements_for_tier(user.get("subscription_tier", "free"))

    if schedule_mode not in ("scheduled", "smart"):
        job_data = {
            "upload_id": upload_id,
            "user_id": str(user["id"]),
            "preferences": user_prefs,
            "plan_features": {
                "ai":           ent.can_ai,
                "priority":     ent.can_priority,
                "watermark":    ent.can_watermark,
                "ai_depth":     ent.ai_depth,
                "caption_frames": ent.max_caption_frames,
            },
            "priority_class": ent.priority_class,
        }
        enqueued = await enqueue_job(job_data, lane="process", priority_class=ent.priority_class)
        if not enqueued:
            async with core.state.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE uploads
                    SET status = 'pending',
                        error_code = 'ENQUEUE_FAILED',
                        error_detail = 'Processing queue unavailable. Retry shortly or use Reprepare.',
                        updated_at = NOW()
                    WHERE id = $1
                    """,
                    upload_id,
                )
            logger.error("[%s] enqueue_job failed after complete — reverted to pending", upload_id)
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "queue_unavailable",
                    "message": "Upload saved but could not reach the processing queue. Try again in a moment.",
                },
            )
        # Safety-net for local/single-process runs where worker loop might be down.
        background_tasks.add_task(
            _inline_rescue_if_stuck,
            upload_id,
            str(user["id"]),
            user_prefs or {},
            ent,
        )

    smart_schedule_display = compute_smart_schedule_display(schedule_mode, upload.get("schedule_metadata"))

    # Audit: upload submitted to pipeline
    await log_system_event(user_id=str(user["id"]), action="UPLOAD_SUBMITTED",
                           event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                           details={"schedule_mode": schedule_mode, "new_status": new_status,
                                    "platforms": list(upload.get("platforms") or [])},
                           request=request)

    return {
        "status": new_status,
        "upload_id": upload_id,
        "schedule_mode": schedule_mode,
        "scheduled_time": upload["scheduled_time"].isoformat() if upload.get("scheduled_time") else None,
        "smart_schedule": smart_schedule_display,
        "processing_features": {
            # `plan` is not defined in this scope -- use ent (resolved above)
            "auto_captions":  bool(user_prefs.get("auto_captions"))        if ent.can_ai else False,
            "auto_thumbnails": bool(user_prefs.get("auto_thumbnails"))     if ent.can_ai else False,
            "ai_hashtags":    bool(user_prefs.get("ai_hashtags_enabled"))  if ent.can_ai else False,
        }
    }


@router.post("/{upload_id}/reprepare")
async def reprepare_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Generate a fresh presigned R2 URL for an upload stuck in pending state.
    Used when the browser refreshed mid-transfer before /complete was called.
    The DB record exists -- we just issue new PUT URLs so the client can retry.
    """
    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT id, r2_key, filename, status, telemetry_r2_key FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
        if not upload:
            raise HTTPException(404, "Upload not found")
        if upload["status"] not in ("pending",):
            raise HTTPException(400, f"Upload is not resumable (status: {upload['status']}). Use /retry for failed uploads.")

    r2_key = upload["r2_key"]
    filename = upload["filename"] or ""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    ct_map = {"mp4": "video/mp4", "mov": "video/quicktime", "avi": "video/x-msvideo", "webm": "video/webm"}
    content_type = ct_map.get(ext, "video/mp4")

    result = {
        "upload_id": upload_id,
        "presigned_url": generate_presigned_upload_url(r2_key, content_type),
        "r2_key": r2_key,
        "filename": filename,
        "status": upload["status"],
    }
    if upload["telemetry_r2_key"]:
        result["telemetry_presigned_url"] = generate_presigned_upload_url(
            upload["telemetry_r2_key"], "application/octet-stream"
        )
        result["telemetry_r2_key"] = upload["telemetry_r2_key"]

    return result


@router.post("/{upload_id}/cancel")
async def cancel_upload(upload_id: str, request: Request, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT put_reserved, aic_reserved, status, r2_key, telemetry_r2_key, processed_r2_key, thumbnail_r2_key FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
        if not upload: raise HTTPException(404, "Upload not found")
        if upload["status"] in ("completed", "succeeded", "cancelled", "failed"):
            raise HTTPException(400, "Cannot cancel this upload")

        current_status = upload["status"]

        if current_status == "processing":
            # Durable signal (worker reads this at every stage boundary)
            await conn.execute(
                "UPDATE uploads SET cancel_requested = TRUE, updated_at = NOW() WHERE id = $1",
                upload_id
            )
            # Fast signal so long-running stages (transcode, publish) can abort
            # within seconds instead of waiting for the next stage boundary.
            await signal_cancel(core.state.redis_client, upload_id)
            await log_system_event(conn, user_id=str(user["id"]), action="UPLOAD_CANCEL_REQUESTED",
                                   event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                                   details={"status_at_cancel": current_status}, request=request)
            return {"status": "cancel_requested", "message": "Cancel signal sent -- job will stop at next checkpoint"}
        else:
            # Pre-processing cancel (pending/queued): tokens were never debited,
            # nothing is in flight, so we can finalize immediately.
            await conn.execute(
                "UPDATE uploads SET cancel_requested = TRUE, status = 'cancelled', updated_at = NOW() WHERE id = $1",
                upload_id
            )
            await refund_tokens(conn, user["id"], upload["put_reserved"], upload["aic_reserved"], upload_id)
            # Drop the Redis flag in case a previous cancel attempt set one.
            await clear_cancel_signal(core.state.redis_client, upload_id)
            await log_system_event(conn, user_id=str(user["id"]), action="UPLOAD_CANCELLED",
                                   event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                                   details={"status_at_cancel": current_status}, request=request, severity="WARNING")
            # Remove derived/processed assets from R2. Keep the user's original
            # source (r2_key + telemetry_r2_key) so they can retry without
            # re-uploading the file. The source is GC'd when the upload row
            # itself is deleted (or by the periodic cancelled-uploads sweeper).
            r2_keys = [k for k in (
                upload.get("processed_r2_key"),
                upload.get("thumbnail_r2_key"),
            ) if k]
            if r2_keys:
                await _delete_r2_objects(r2_keys)
            return {"status": "cancelled"}


@router.get("")
async def get_uploads(
    status: Optional[str] = None,
    view: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    trill_only: bool = False,
    meta: bool = False,
    user: dict = Depends(get_current_user),
):
    """
    Upload queue list for current user.

    Filter by status (exact) or view (semantic group):
      view=pending   -> status IN (pending, staged, queued, scheduled, ready_to_publish) -- waiting uploads incl. smart/scheduled
      view=processing -> status = processing
      view=completed -> status IN (completed, succeeded, partial)
      view=failed   -> status = failed
      view=staged   -> same as pending (alias for Staged/Pending filter)
      view=smart_schedule -> schedule_mode='smart' AND status IN pending group

    Contract (frontend-safe):
      - status_label: human-readable label for display (fixes ? succeeded, ? staged)
      - thumbnail_url, platform_results, hashtags, etc.
    """
    return await fetch_user_uploads_list(
        core.state.db_pool,
        str(user["id"]),
        status=status,
        view=view,
        limit=limit,
        offset=offset,
        trill_only=trill_only,
        meta=meta,
    )


@router.get("/queue-stats")
async def get_uploads_queue_stats(user: dict = Depends(get_current_user)):
    """
    Queue summary counts for queue.html and dashboard.html.
    Use these counts for Pending, Processing, Completed, Failed cards.
    Pending includes staged, queued, scheduled, ready_to_publish (smart + scheduled).
    """
    return await fetch_upload_queue_stats(core.state.db_pool, str(user["id"]))


@router.post("/{upload_id}/generate-thumbnail")
async def generate_thumbnail_for_upload(
    upload_id: str,
    force: bool = Query(False, description="Regenerate even when a thumbnail already exists"),
    user: dict = Depends(get_current_user),
):
    """
    Backfill / regenerate the thumbnail for an existing upload.

    Uses the processed video when available, extracts a base frame with FFmpeg, then
    runs the same styled stack as the worker (Pikzels v2 when configured, else PIL
    template) for accounts with custom styled thumbnails. Tier without styled thumbs
    gets FFmpeg-only JPEG.

    Query ``force=1`` to replace an existing thumbnail (default: return current URL only).
    """
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT u.id, u.r2_key, u.processed_r2_key, u.thumbnail_r2_key, u.status, u.platforms,
                   u.title, u.ai_title, u.ai_generated_title, u.user_preferences,
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
        "ai_title": row["ai_title"],
        "ai_generated_title": row["ai_generated_title"],
        "user_preferences": row["user_preferences"],
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
    """
    Get a presigned URL for uploading a custom thumbnail.
    After uploading, call PATCH /api/uploads/{upload_id} with thumbnail_r2_key in the body (if supported).
    """
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, status FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
    if not row:
        raise HTTPException(404, "Upload not found")
    editable = ("pending", "scheduled", "queued", "staged", "ready_to_publish")
    if row["status"] not in editable:
        raise HTTPException(400, "Cannot change thumbnail after upload is processing or published")

    thumb_r2_key = f"thumbnails/{user['id']}/{upload_id}/custom.jpg"
    presigned_url = generate_presigned_upload_url(thumb_r2_key, "image/jpeg")
    return {"presigned_url": presigned_url, "r2_key": thumb_r2_key}


async def _fetch_platform_video_engagement(
    client: httpx.AsyncClient,
    plat: str,
    video_id: str,
    pr: dict,
    access_token: str,
) -> Optional[Dict[str, int]]:
    """
    Call the platform metrics API for one video/reel/post. Returns a stats dict or None
    if the request failed or returned no usable payload (caller may try another token).
    """
    if not access_token:
        return None
    try:
        if plat == "tiktok" and video_id:
            resp = await client.post(
                "https://open.tiktokapis.com/v2/video/query/",
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                params={"fields": "id,view_count,like_count,comment_count,share_count"},
                json={"filters": {"video_ids": [str(video_id)]}},
            )
            if resp.status_code != 200:
                return None
            vids = resp.json().get("data", {}).get("videos", []) or []
            if not vids:
                return None
            v = vids[0]
            return {
                "views": int(v.get("view_count") or 0),
                "likes": int(v.get("like_count") or 0),
                "comments": int(v.get("comment_count") or 0),
                "shares": int(v.get("share_count") or 0),
            }

        if plat == "youtube" and video_id:
            resp = await client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={"part": "statistics", "id": str(video_id)},
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code != 200:
                return None
            items = resp.json().get("items", []) or []
            if not items:
                return None
            st = items[0].get("statistics", {})
            return {
                "views": int(st.get("viewCount") or 0),
                "likes": int(st.get("likeCount") or 0),
                "comments": int(st.get("commentCount") or 0),
                "shares": 0,
            }

        if plat == "instagram" and video_id:
            media_id = pr.get("platform_video_id") or pr.get("media_id") or video_id
            resp = await client.get(
                f"https://graph.facebook.com/v21.0/{media_id}/insights",
                params={
                    "access_token": access_token,
                    "metric": "views,plays,likes,comments,saved,shares,reach",
                },
            )
            if resp.status_code != 200:
                return None
            s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
            ig_views = ig_plays = 0
            for m in resp.json().get("data", []) or []:
                name = m.get("name", "")
                vals = m.get("values", [])
                val = int(vals[-1].get("value", 0) if vals else m.get("value", 0) or 0)
                if name == "views":
                    ig_views = val
                elif name == "plays":
                    ig_plays = val
                elif name == "likes":
                    s["likes"] += val
                elif name == "comments":
                    s["comments"] += val
                elif name == "shares":
                    s["shares"] += val
            s["views"] = ig_views or ig_plays
            return s

        if plat == "facebook" and video_id:
            resp = await client.get(
                f"https://graph.facebook.com/v21.0/{video_id}",
                params={
                    "access_token": access_token,
                    "fields": "insights.metric(total_video_views,total_video_reactions_by_type_total,total_video_comments,total_video_shares)",
                },
            )
            if resp.status_code != 200:
                return None
            s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
            for m in resp.json().get("insights", {}).get("data", []) or []:
                name = m.get("name", "")
                vals = m.get("values", [{}])
                val = vals[-1].get("value", 0) if vals else 0
                if isinstance(val, dict):
                    val = sum(val.values())
                val = int(val or 0)
                if name == "total_video_views":
                    s["views"] += val
                elif name == "total_video_reactions_by_type_total":
                    s["likes"] += val
                elif name == "total_video_comments":
                    s["comments"] += val
                elif name == "total_video_shares":
                    s["shares"] += val
            return s
    except Exception as e:
        logger.warning("sync-analytics fetch %s/%s: %s", plat, video_id, e)
        return None
    return None


def _merge_stats_into_platform_result(pr: dict, s: Dict[str, int]) -> None:
    """Write rollup fields used by the dashboard normalizeUpload() aggregation."""
    pr["views"] = s["views"]
    pr["view_count"] = s["views"]
    pr["likes"] = s["likes"]
    pr["like_count"] = s["likes"]
    pr["comments"] = s["comments"]
    pr["comment_count"] = s["comments"]
    pr["shares"] = s["shares"]
    pr["share_count"] = s["shares"]


def _plat_token_resolution_maps(
    token_rows: list,
    token_map_by_id: Dict[str, dict],
    token_map_by_platform: Dict[str, dict],
) -> Tuple[Dict[Tuple[str, str], dict], Dict[Tuple[str, str], Tuple[str, dict]], Dict[str, List[Tuple[str, dict]]]]:
    """Build (platform, account_id) maps and per-platform token lists from refreshed tokens."""
    token_map_by_plat_account: Dict[Tuple[str, str], dict] = {}
    plat_account_row_map: Dict[Tuple[str, str], Tuple[str, dict]] = {}
    platform_token_rows: Dict[str, List[Tuple[str, dict]]] = {}
    for tr in token_rows:
        tid = str(tr["id"])
        dec = token_map_by_id.get(tid)
        if not dec:
            continue
        plat = str(tr.get("platform") or "").lower()
        aid = tr.get("account_id")
        if aid is not None and str(aid).strip() != "":
            a = str(aid).strip()
            token_map_by_plat_account[(plat, a)] = dec
            plat_account_row_map[(plat, a)] = (tid, dec)
        platform_token_rows.setdefault(plat, []).append((tid, dec))
    return token_map_by_plat_account, plat_account_row_map, platform_token_rows


async def _sync_upload_analytics_core(user: dict, upload_id: str) -> dict:
    """
    Shared implementation for per-upload analytics sync.
    Raises HTTPException(404) if the upload does not exist for this user.
    """
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, platforms, platform_results, status FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"],
        )
    if not row:
        raise HTTPException(404, "Upload not found")

    if row["status"] not in ("completed", "succeeded", "partial"):
        return {"synced": False, "reason": "not_completed", "views": 0, "likes": 0, "comments": 0, "shares": 0}

    # Parse platform_results to get per-platform video IDs
    raw_pr = _safe_json(row["platform_results"], [])
    pr_list = []
    if isinstance(raw_pr, list):
        pr_list = [x for x in raw_pr if isinstance(x, dict)]
    elif isinstance(raw_pr, dict):
        pr_list = [{"platform": k, **v} if isinstance(v, dict) else {"platform": k} for k, v in raw_pr.items()]
    # Alias canonical worker fields so lookups below find them
    for pr in pr_list:
        if pr.get("platform_video_id") and not pr.get("video_id"):
            pr["video_id"] = pr["platform_video_id"]
        if pr.get("platform_url") and not pr.get("url"):
            pr["url"] = pr["platform_url"]

    # Get tokens for all connected platforms (include id for multi-account lookup)
    async with core.state.db_pool.acquire() as conn:
        token_rows = await conn.fetch(
            "SELECT id, platform, token_blob, account_id FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
            user["id"],
        )

    token_map_by_id = {}
    token_map_by_platform = {}
    uid = str(user["id"])
    for tr in token_rows:
        try:
            dec = decrypt_blob(tr["token_blob"])
            if dec:
                if tr["platform"] == "instagram" and not dec.get("ig_user_id") and tr["account_id"]:
                    dec["ig_user_id"] = str(tr["account_id"])
                if tr["platform"] == "facebook" and not dec.get("page_id") and tr["account_id"]:
                    dec["page_id"] = str(tr["account_id"])
                token_id = str(tr["id"])
                dec = await refresh_decrypted_token_for_row(
                    tr["platform"],
                    dec,
                    db_pool=core.state.db_pool,
                    user_id=uid,
                    token_row_id=token_id,
                )
                token_map_by_id[token_id] = dec
                plat_norm = str(tr.get("platform") or "").lower()
                if plat_norm:
                    token_map_by_platform[plat_norm] = dec
        except Exception:
            pass

    token_map_by_plat_account, plat_account_row_map, platform_token_rows = _plat_token_resolution_maps(
        list(token_rows), token_map_by_id, token_map_by_platform
    )

    total_views = total_likes = total_comments = total_shares = 0
    platform_stats: Dict[str, Dict[str, int]] = {}
    rows_with_video_id = 0
    fetched_any = False

    async with httpx.AsyncClient(timeout=20) as client:
        for pr in pr_list:
            plat = str(pr.get("platform") or "").lower()
            video_id = (
                pr.get("platform_video_id")
                or pr.get("video_id")
                or pr.get("videoId")
                or pr.get("id")
                or pr.get("media_id")
                or pr.get("post_id")
                or pr.get("share_id")
            )
            if not video_id:
                continue
            rows_with_video_id += 1

            candidates = resolve_token_candidates_for_platform_result(
                pr,
                token_map_by_id,
                token_map_by_plat_account,
                token_map_by_platform,
                plat_account_row_map=plat_account_row_map,
                platform_token_rows=platform_token_rows,
            )
            if not candidates:
                continue

            s: Optional[Dict[str, int]] = None
            for tok in candidates:
                at = (tok or {}).get("access_token", "")
                s = await _fetch_platform_video_engagement(client, plat, str(video_id), pr, at)
                if s is not None:
                    break

            if not s:
                continue

            _merge_stats_into_platform_result(pr, s)
            fetched_any = True
            total_views += s["views"]
            total_likes += s["likes"]
            total_comments += s["comments"]
            total_shares += s["shares"]
            prev = platform_stats.get(plat)
            if prev:
                platform_stats[plat] = {
                    "views": prev["views"] + s["views"],
                    "likes": prev["likes"] + s["likes"],
                    "comments": prev["comments"] + s["comments"],
                    "shares": prev["shares"] + s["shares"],
                }
            else:
                platform_stats[plat] = dict(s)

    async with core.state.db_pool.acquire() as conn:
        if pr_list:
            pr_json = json.dumps(pr_list)
            await conn.execute(
                """UPDATE uploads SET views=$1, likes=$2, comments=$3, shares=$4,
                       platform_results = $7::jsonb,
                       analytics_synced_at=NOW(), updated_at=NOW()
                   WHERE id=$5 AND user_id=$6""",
                total_views,
                total_likes,
                total_comments,
                total_shares,
                upload_id,
                user["id"],
                pr_json,
            )
        else:
            await conn.execute(
                """UPDATE uploads SET views=$1, likes=$2, comments=$3, shares=$4,
                       analytics_synced_at=NOW(), updated_at=NOW()
                   WHERE id=$5 AND user_id=$6""",
                total_views,
                total_likes,
                total_comments,
                total_shares,
                upload_id,
                user["id"],
            )

    if not rows_with_video_id:
        return {
            "synced": False,
            "reason": "no_platform_video_ids",
            "views": total_views,
            "likes": total_likes,
            "comments": total_comments,
            "shares": total_shares,
            "platform_stats": platform_stats,
        }
    if not fetched_any:
        return {
            "synced": False,
            "reason": "no_tokens_or_metrics",
            "message": "No working OAuth token matched this upload, or platforms returned no data.",
            "views": total_views,
            "likes": total_likes,
            "comments": total_comments,
            "shares": total_shares,
            "platform_stats": platform_stats,
        }

    return {
        "synced": True,
        "views": total_views,
        "likes": total_likes,
        "comments": total_comments,
        "shares": total_shares,
        "platform_stats": platform_stats,
    }


async def _background_sync_uploads_analytics(user_id: str, upload_ids: list[str]) -> None:
    for up_id in upload_ids:
        try:
            await _sync_upload_analytics_core({"id": user_id}, up_id)
        except HTTPException:
            pass
        except Exception as e:
            logger.warning("sync-analytics/all upload=%s: %s", up_id, e)
        await asyncio.sleep(0.35)


@router.post("/sync-analytics/all")
async def sync_all_upload_analytics(
    background_tasks: BackgroundTasks,
    max_uploads: int = Query(800, ge=1, le=2000),
    async_mode: bool = Query(True),
    user: dict = Depends(get_current_user),
):
    """
    Batch engagement sync for many completed uploads (dashboard / queue auto-sync).
    async_mode=true queues work and returns immediately.
    """
    uid = str(user["id"])
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
    return {"ok": True, "candidates": len(ids), "synced": synced, "async_mode": False}


@router.post("/{upload_id}/sync-analytics")
async def sync_upload_analytics(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Fetch latest engagement stats for a single completed upload from platform APIs.
    Uses the video IDs stored in platform_results to query per-video metrics.
    Updates the uploads table (views, likes, comments, shares) and returns fresh data.
    """
    return await _sync_upload_analytics_core(user, upload_id)


@router.delete("/{upload_id}")
async def delete_upload(upload_id: str, request: Request, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT put_reserved, aic_reserved, status, title, platforms FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user["id"])
        if not upload: raise HTTPException(404, "Upload not found")
        if upload["status"] in ("pending", "queued"):
            await refund_tokens(conn, user["id"], upload["put_reserved"], upload["aic_reserved"], upload_id)
        await conn.execute("DELETE FROM uploads WHERE id = $1", upload_id)
        await log_system_event(conn, user_id=str(user["id"]), action="UPLOAD_DELETED",
                               event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                               details={"title": upload["title"], "status_at_delete": upload["status"],
                                        "platforms": list(upload["platforms"] or [])},
                               request=request, severity="WARNING")
    return {"status": "deleted"}


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


_RETRYABLE_STATUSES = ("failed", "cancelled", "partial")


@router.post("/{upload_id}/retry")
async def retry_upload(
    upload_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Re-queue a failed / cancelled / partial upload for processing.

    Behavior:
      * Idempotent: a second click within ``RETRY_IDEMPOTENCY_TTL_SEC`` no-ops
        with HTTP 200 ``{"status": "already_queued"}`` instead of double-enqueueing.
      * Soft cap: blocks at HTTP 429 once ``MAX_USER_RETRIES_DEFAULT`` is reached.
      * Pre-flight gate: deterministic errors (token-empty, OAuth revoked, plan
        block) fail fast at HTTP 409 with a hint instead of burning a retry.
      * Partial-aware: for ``partial`` status, only re-publishes the platforms
        that previously failed; succeeded entries are preserved.
      * Audited: writes ``UPLOAD_RETRIED`` to the system event log with mode +
        retry_count + prior error_code.
    """
    user_id_str = str(user["id"])

    # ── Idempotency lock (Redis SETNX) ────────────────────────────────────
    # Held for a few seconds so a double-click / concurrent tab doesn't enqueue
    # two jobs for the same upload. Failing soft (no Redis) is fine — we just
    # lose dedupe protection, not correctness.
    redis = core.state.redis_client
    lock_key = f"upload_retry_lock:{upload_id}"
    if redis is not None:
        try:
            acquired = await redis.set(lock_key, user_id_str, nx=True, ex=RETRY_IDEMPOTENCY_TTL_SEC)
        except Exception as e:
            logger.warning(f"retry idempotency lock unavailable for {upload_id}: {e}")
            acquired = True
        if not acquired:
            return {"status": "already_queued", "upload_id": upload_id}

    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
        if not upload:
            raise HTTPException(404, "Upload not found")

        current_status = (upload["status"] or "").lower()
        if current_status not in _RETRYABLE_STATUSES:
            raise HTTPException(
                400,
                f"Upload status '{current_status}' is not retryable. "
                f"Allowed: {', '.join(_RETRYABLE_STATUSES)}.",
            )

        # ── Pre-flight: block deterministic re-failures with a clear hint ──
        decision = classify_retry_error(upload.get("error_code"))
        if not decision.allowed:
            raise HTTPException(
                decision.http_status,
                detail={
                    "code": decision.code,
                    "message": decision.message,
                    "hint": decision.hint,
                    "error_code": upload.get("error_code"),
                },
            )

        # ── Soft retry cap ────────────────────────────────────────────────
        existing_artifacts = upload.get("output_artifacts") or {}
        if isinstance(existing_artifacts, str):
            try:
                existing_artifacts = json.loads(existing_artifacts) or {}
            except Exception:
                existing_artifacts = {}
        prior_count = get_retry_count(existing_artifacts)
        if prior_count >= MAX_USER_RETRIES_DEFAULT:
            raise HTTPException(
                429,
                detail={
                    "code": "retry_cap_reached",
                    "message": f"This upload has been retried {prior_count} times. "
                               "Edit the upload, fix the underlying issue, or contact support.",
                    "retry_count": prior_count,
                    "max_retries": MAX_USER_RETRIES_DEFAULT,
                },
            )

        # ── Partial: figure out which platforms to retry ──────────────────
        prior_platform_results = upload.get("platform_results")
        if isinstance(prior_platform_results, str):
            try:
                prior_platform_results = json.loads(prior_platform_results)
            except Exception:
                prior_platform_results = []
        succeeded_entries, failed_platforms = split_platform_results(prior_platform_results)

        retry_mode = "full"
        retry_subset: Optional[List[str]] = None
        seeded_results: Optional[List[Dict]] = None

        if current_status == "partial":
            if not failed_platforms:
                # Nothing actually failed — nothing to retry. Tell the client
                # plainly instead of silently re-running everything.
                raise HTTPException(
                    400,
                    detail={
                        "code": "no_failed_platforms",
                        "message": "This upload is marked partial but every platform "
                                   "result is successful. Nothing to retry.",
                    },
                )
            retry_mode = "partial"
            retry_subset = failed_platforms
            seeded_results = succeeded_entries  # worker will pre-seed ctx.platform_results

        # ── Reset transient processing state ──────────────────────────────
        # Keep engagement + cost fields intact. For partial retries we KEEP
        # platform_results so successful posts stay visible in the UI; the
        # worker merges new attempts on top.
        new_artifacts = bump_retry_metadata(
            existing_artifacts,
            actor_user_id=user_id_str,
            prior_error_code=upload.get("error_code"),
            mode=retry_mode,
            retry_platforms=retry_subset,
        )

        # Reset transient processing state. We deliberately keep ``platform_results``
        # intact so partial-success entries stay visible while the retry runs;
        # the worker pre-seeds them into ctx.platform_results and the final
        # write merges new attempts on top.
        await conn.execute(
            """
            UPDATE uploads
            SET status = 'queued',
                error_code = NULL,
                error_detail = NULL,
                processing_started_at = NULL,
                processing_finished_at = NULL,
                completed_at = NULL,
                cancel_requested = FALSE,
                output_artifacts = $3::jsonb,
                updated_at = NOW()
            WHERE id = $1 AND user_id = $2
            """,
            upload_id, user["id"], json.dumps(new_artifacts),
        )

        user_prefs = await get_user_prefs_for_upload(conn, user["id"])

        await log_system_event(
            conn,
            user_id=user_id_str,
            action="UPLOAD_RETRIED",
            event_category="UPLOAD",
            resource_type="upload",
            resource_id=upload_id,
            details={
                "mode": retry_mode,
                "retry_count": prior_count + 1,
                "max_retries": MAX_USER_RETRIES_DEFAULT,
                "prior_error_code": upload.get("error_code"),
                "prior_status": current_status,
                "platforms_retried": retry_subset or list(upload.get("platforms") or []),
            },
            request=request,
        )

    # Drop any leftover cancel flag from a previous run so the new worker
    # job isn't immediately aborted by a stale Redis cancel signal.
    await clear_cancel_signal(core.state.redis_client, upload_id)

    # Resolve full entitlements -- drives queue routing, AI depth, priority class
    ent = get_entitlements_for_tier(user.get("subscription_tier", "free"))

    job_data: Dict = {
        "upload_id": upload_id,
        "user_id": user_id_str,
        "preferences": user_prefs,
        "plan_features": {
            "ai":             ent.can_ai,
            "priority":       ent.can_priority,
            "watermark":      ent.can_watermark,
            "ai_depth":       ent.ai_depth,
            "caption_frames": ent.max_caption_frames,
        },
        "priority_class": ent.priority_class,
        "action": "retry",
        "retry_mode": retry_mode,
        "retry_count": prior_count + 1,
    }
    if retry_subset:
        job_data["retry_platforms_subset"] = retry_subset
    if seeded_results:
        job_data["prior_platform_results"] = seeded_results

    enqueued = await enqueue_job(job_data, lane="process", priority_class=ent.priority_class)
    if not enqueued:
        async with core.state.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE uploads
                SET status = 'failed',
                    error_code = 'ENQUEUE_FAILED',
                    error_detail = 'Processing queue unavailable. Click Retry again in a moment.',
                    updated_at = NOW()
                WHERE id = $1 AND user_id = $2
                """,
                upload_id,
                user_id_str,
            )
        logger.error("[%s] enqueue_job failed after retry — marked failed ENQUEUE_FAILED", upload_id)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "queue_unavailable",
                "message": "Could not reach the processing queue. Try again shortly.",
            },
        )
    return {
        "status": "requeued",
        "upload_id": upload_id,
        "mode": retry_mode,
        "retry_count": prior_count + 1,
        "max_retries": MAX_USER_RETRIES_DEFAULT,
        "platforms": retry_subset or list(upload.get("platforms") or []),
    }


@router.get("/{upload_id}")
async def get_upload_details(upload_id: str, user_id: str = Depends(get_verified_user_id)):
    """
    Upload detail for current user.

    Uses ``get_verified_user_id`` (JWT only, no DB) plus a single connection in
    ``fetch_upload_detail`` for user gates + upload + recognition — avoids a second
    pool checkout and duplicate connection teardown (Sentry: consecutive queries /
    ``pg_advisory_unlock_all`` on ``/api/uploads/{id}``).

    Contract (frontend-safe):
      - thumbnail_url: presigned R2 URL (if thumbnail_r2_key exists)
      - platform_results: always list
      - hashtags: always list[str]
      - title/caption: falls back to AI values when empty
      - ai_title/ai_caption/ai_hashtags always present
      - duration_seconds computed from processing timestamps when available
    """
    return await fetch_upload_detail(core.state.db_pool, upload_id, user_id)
