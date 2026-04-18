"""
UploadM8 Upload routes -- extracted from app.py.

Handles upload lifecycle: presign, complete, cancel, retry, list, update,
thumbnail generation, analytics sync, and single-upload detail.
"""

import json
import logging
import pathlib
import uuid
from typing import List, Optional

import asyncio

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

import core.state
from core.audit import log_system_event
from core.auth import decrypt_blob
from core.config import R2_BUCKET_NAME
from core.deps import get_current_user
from core.helpers import _safe_json, get_plan
from core.models import CompleteUploadBody, UploadInit, UploadUpdate
from core.queue import enqueue_job
from core.r2 import (
    _delete_r2_objects,
    _normalize_r2_key,
    generate_presigned_download_url,
    generate_presigned_upload_url,
    get_s3_client,
)
from services.smart_schedule_insights import calculate_smart_schedule_data_driven
from core.wallet import partial_refund_tokens, refund_tokens
from routers.preferences import get_user_prefs_for_upload
from stages.entitlements import get_entitlements_for_tier
from services.platform_oauth_refresh import refresh_decrypted_token_for_row
from services.uploads_api import update_upload_metadata
from services.uploads_handlers import (
    ALLOWED_VIDEO_TYPES,
    complete_upload_transaction,
    compute_smart_schedule_display,
    fetch_upload_detail,
    fetch_upload_queue_stats,
    fetch_user_uploads_list,
    presign_create_upload,
)

logger = logging.getLogger("uploadm8-api")

router = APIRouter(prefix="/api/uploads", tags=["uploads"])


# ============================================================
# Routes
# ============================================================

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

        if getattr(data, "has_telemetry", False):
            telem_key = f"uploads/{user['id']}/{upload_id}/telemetry.map"
            result["telemetry_presigned_url"] = generate_presigned_upload_url(telem_key, "application/octet-stream")
            result["telemetry_r2_key"] = telem_key
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
async def complete_upload(upload_id: str, request: Request, user: dict = Depends(get_current_user)):
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
        await enqueue_job(job_data, lane="process", priority_class=ent.priority_class)

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
            await conn.execute(
                "UPDATE uploads SET cancel_requested = TRUE, updated_at = NOW() WHERE id = $1",
                upload_id
            )
            await log_system_event(conn, user_id=str(user["id"]), action="UPLOAD_CANCEL_REQUESTED",
                                   event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                                   details={"status_at_cancel": current_status}, request=request)
            return {"status": "cancel_requested", "message": "Cancel signal sent -- job will stop at next checkpoint"}
        else:
            await conn.execute(
                "UPDATE uploads SET cancel_requested = TRUE, status = 'cancelled', updated_at = NOW() WHERE id = $1",
                upload_id
            )
            await refund_tokens(conn, user["id"], upload["put_reserved"], upload["aic_reserved"], upload_id)
            await log_system_event(conn, user_id=str(user["id"]), action="UPLOAD_CANCELLED",
                                   event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                                   details={"status_at_cancel": current_status}, request=request, severity="WARNING")
            # Remove video and related assets from R2 so they don't persist
            r2_keys = [k for k in (
                upload.get("r2_key"),
                upload.get("telemetry_r2_key"),
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
async def generate_thumbnail_for_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Backfill / regenerate the thumbnail for an existing upload.

    Workflow:
      1. Fetch the video from R2 to a temp file
      2. Run FFmpeg to extract a frame at 30% into the video
      3. Upload the JPEG to R2 at thumbnails/{user_id}/{upload_id}/thumbnail.jpg
      4. Update thumbnail_r2_key in the uploads row
      5. Return a fresh presigned URL

    This fixes the gap where uploads processed before the worker fix
    have thumbnail_r2_key = NULL in the database.
    """
    import tempfile, subprocess
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, r2_key, thumbnail_r2_key, status FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
    if not row:
        raise HTTPException(404, "Upload not found")

    # If thumbnail already exists, just return the presigned URL
    if row.get("thumbnail_r2_key"):
        try:
            s3 = get_s3_client()
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": _normalize_r2_key(row["thumbnail_r2_key"])},
                ExpiresIn=3600,
            )
            return {"thumbnail_url": url, "r2_key": row["thumbnail_r2_key"], "generated": False}
        except Exception:
            pass  # fall through and regenerate

    r2_key = row.get("r2_key")
    if not r2_key:
        raise HTTPException(400, "No video file key found for this upload")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        video_path = tmp_path / "video.mp4"
        thumb_path = tmp_path / "thumbnail.jpg"

        # 1. Download video from R2
        try:
            s3 = get_s3_client()
            s3.download_file(R2_BUCKET_NAME, _normalize_r2_key(r2_key), str(video_path))
        except Exception as e:
            raise HTTPException(500, f"Could not download video from storage: {e}")

        # 2. Get duration then extract frame at 30%
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
                capture_output=True, text=True, timeout=30
            )
            duration = 10.0
            if probe.returncode == 0:
                import json as _json
                for stream in _json.loads(probe.stdout).get("streams", []):
                    if stream.get("codec_type") == "video":
                        duration = float(stream.get("duration", 10) or 10)
                        break
            offset = max(0.5, duration * 0.30)
        except Exception:
            offset = 5.0

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{offset:.3f}",
                    "-i", str(video_path),
                    "-vframes", "1",
                    "-q:v", "2",
                    "-vf", "scale=1080:-2",
                    str(thumb_path),
                ],
                capture_output=True, timeout=60
            )
            if result.returncode != 0 or not thumb_path.exists():
                # Fallback: try at 1 second
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", "1", "-i", str(video_path),
                     "-vframes", "1", "-q:v", "2", "-vf", "scale=1080:-2", str(thumb_path)],
                    capture_output=True, timeout=30
                )
        except Exception as e:
            raise HTTPException(500, f"FFmpeg thumbnail extraction failed: {e}")

        if not thumb_path.exists():
            raise HTTPException(500, "Thumbnail extraction produced no output")

        # 3. Upload to R2
        thumb_r2_key = f"thumbnails/{user['id']}/{upload_id}/thumbnail.jpg"
        try:
            s3.upload_file(
                str(thumb_path), R2_BUCKET_NAME, thumb_r2_key,
                ExtraArgs={"ContentType": "image/jpeg"}
            )
        except Exception as e:
            raise HTTPException(500, f"Failed to upload thumbnail to storage: {e}")

        # 4. Update DB
        async with core.state.db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE uploads SET thumbnail_r2_key = $1, updated_at = NOW() WHERE id = $2",
                thumb_r2_key, upload_id
            )

        # 5. Return presigned URL
        try:
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": thumb_r2_key},
                ExpiresIn=3600,
            )
        except Exception:
            url = None

        return {
            "thumbnail_url": url,
            "r2_key": thumb_r2_key,
            "generated": True,
            "offset_seconds": offset,
        }


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
                token_map_by_platform[tr["platform"]] = dec
        except Exception:
            pass

    total_views = total_likes = total_comments = total_shares = 0
    platform_stats = {}

    async with httpx.AsyncClient(timeout=20) as client:
        for pr in pr_list:
            plat = str(pr.get("platform") or "").lower()
            # platform_video_id is the canonical field written by db.py/mark_processing_completed
            # video_id / media_id / share_id etc. are legacy / webhook-written variants
            video_id = (
                pr.get("platform_video_id")  # canonical (worker pipeline)
                or pr.get("video_id") or pr.get("videoId") or pr.get("id")
                or pr.get("media_id") or pr.get("post_id") or pr.get("share_id")
            )
            if not video_id:
                continue
            # Do not require success==True: partial uploads and some legacy rows omit it; if we
            # have a platform video id, the metrics APIs are the source of truth.

            # Multi-account: token_row_id is platform_tokens.id (UUID). account_id is the
            # platform's own account id — never use it as a key into token_map_by_id.
            tid = pr.get("token_row_id")
            tok = token_map_by_id.get(str(tid), {}) if tid else {}
            if not tok:
                tok = token_map_by_platform.get(plat, {})
            access_token = tok.get("access_token", "")
            if not access_token:
                continue

            try:
                if plat == "tiktok" and video_id:
                    resp = await client.post(
                        "https://open.tiktokapis.com/v2/video/query/",
                        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                        params={"fields": "id,view_count,like_count,comment_count,share_count"},
                        json={"filters": {"video_ids": [str(video_id)]}},
                    )
                    if resp.status_code == 200:
                        vids = resp.json().get("data", {}).get("videos", []) or []
                        if vids:
                            v = vids[0]
                            s = {"views": int(v.get("view_count") or 0), "likes": int(v.get("like_count") or 0),
                                 "comments": int(v.get("comment_count") or 0), "shares": int(v.get("share_count") or 0)}
                            platform_stats["tiktok"] = s
                            total_views    += s["views"];    total_likes   += s["likes"]
                            total_comments += s["comments"]; total_shares  += s["shares"]

                elif plat == "youtube" and video_id:
                    resp = await client.get(
                        "https://www.googleapis.com/youtube/v3/videos",
                        params={"part": "statistics", "id": str(video_id)},
                        headers={"Authorization": f"Bearer {access_token}"},
                    )
                    if resp.status_code == 200:
                        items = resp.json().get("items", []) or []
                        if items:
                            st = items[0].get("statistics", {})
                            s = {"views": int(st.get("viewCount") or 0), "likes": int(st.get("likeCount") or 0),
                                 "comments": int(st.get("commentCount") or 0), "shares": 0}
                            platform_stats["youtube"] = s
                            total_views    += s["views"];    total_likes   += s["likes"]
                            total_comments += s["comments"]

                elif plat == "instagram" and video_id:
                    # Instagram Insights API requires numeric media_id (not shortcode)
                    media_id = pr.get("platform_video_id") or pr.get("media_id") or video_id
                    resp = await client.get(
                        f"https://graph.facebook.com/v21.0/{media_id}/insights",
                        params={"access_token": access_token,
                                "metric": "views,plays,likes,comments,saved,shares,reach"},
                    )
                    if resp.status_code == 200:
                        s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
                        ig_views = ig_plays = 0
                        for m in resp.json().get("data", []) or []:
                            name = m.get("name", "")
                            vals = m.get("values", [])
                            val  = int(vals[-1].get("value", 0) if vals else m.get("value", 0) or 0)
                            if name == "views":       ig_views     = val
                            elif name == "plays":     ig_plays     = val  # deprecated fallback
                            elif name == "likes":     s["likes"]   += val
                            elif name == "comments":  s["comments"] += val
                            elif name == "shares":    s["shares"]  += val
                        s["views"] = ig_views or ig_plays  # prefer views over deprecated plays
                        platform_stats["instagram"] = s
                        total_views    += s["views"];    total_likes   += s["likes"]
                        total_comments += s["comments"]; total_shares  += s["shares"]

                elif plat == "facebook" and video_id:
                    page_id = tok.get("page_id", "")
                    resp = await client.get(
                        f"https://graph.facebook.com/v21.0/{video_id}",
                        params={"access_token": access_token,
                                "fields": "insights.metric(total_video_views,total_video_reactions_by_type_total,total_video_comments,total_video_shares)"},
                    )
                    if resp.status_code == 200:
                        s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
                        for m in resp.json().get("insights", {}).get("data", []) or []:
                            name = m.get("name", "")
                            vals = m.get("values", [{}])
                            val  = vals[-1].get("value", 0) if vals else 0
                            if isinstance(val, dict): val = sum(val.values())
                            val = int(val or 0)
                            if name == "total_video_views":                      s["views"]    += val
                            elif name == "total_video_reactions_by_type_total":  s["likes"]    += val
                            elif name == "total_video_comments":                  s["comments"] += val
                            elif name == "total_video_shares":                    s["shares"]   += val
                        platform_stats["facebook"] = s
                        total_views    += s["views"];    total_likes   += s["likes"]
                        total_comments += s["comments"]; total_shares  += s["shares"]

            except Exception as e:
                logger.warning(f"sync-analytics error for {plat}/{video_id}: {e}")
                continue

    # Persist to DB
    async with core.state.db_pool.acquire() as conn:
        await conn.execute(
            """UPDATE uploads SET views=$1, likes=$2, comments=$3, shares=$4,
                   analytics_synced_at=NOW(), updated_at=NOW()
               WHERE id=$5 AND user_id=$6""",
            total_views, total_likes, total_comments, total_shares,
            upload_id, user["id"],
        )

    return {
        "synced": True,
        "views": total_views, "likes": total_likes,
        "comments": total_comments, "shares": total_shares,
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


@router.post("/{upload_id}/retry")
async def retry_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """Reset a failed/cancelled upload and re-queue it for processing."""
    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
        if not upload:
            raise HTTPException(404, "Upload not found")

        # Only allow retry for terminal states
        if upload["status"] not in ("failed", "cancelled"):
            raise HTTPException(400, "Only failed or cancelled uploads can be retried")

        # Reset processing state (keep engagement + cost fields intact)
        await conn.execute(
            """
            UPDATE uploads
            SET status = 'pending',
                error_code = NULL,
                error_detail = NULL,
                processing_started_at = NULL,
                processing_finished_at = NULL,
                completed_at = NULL,
                cancel_requested = FALSE,
                updated_at = NOW()
            WHERE id = $1 AND user_id = $2
            """,
            upload_id, user["id"]
        )

        # Pull latest preferences (and respect plan entitlements)
        user_prefs = await get_user_prefs_for_upload(conn, user["id"])
        plan = get_plan(user.get("subscription_tier", "free"))

    job_data = {
        "job_id": str(uuid.uuid4()),
        "upload_id": upload_id,
        "user_id": str(user["id"]),
        "preferences": user_prefs,
        "plan_features": {
            "ai": plan.get("ai", False),
            "priority": plan.get("priority", False),
            "watermark": plan.get("watermark", True),
        },
        "action": "retry",
    }

    await enqueue_job(job_data, priority=plan.get("priority", False))
    return {"status": "requeued", "upload_id": upload_id}


@router.get("/{upload_id}")
async def get_upload_details(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Upload detail for current user.

    Contract (frontend-safe):
      - thumbnail_url: presigned R2 URL (if thumbnail_r2_key exists)
      - platform_results: always list
      - hashtags: always list[str]
      - title/caption: falls back to AI values when empty
      - ai_title/ai_caption/ai_hashtags always present
      - duration_seconds computed from processing timestamps when available
    """
    return await fetch_upload_detail(core.state.db_pool, upload_id, str(user["id"]))
