"""Upload lifecycle routes: presign, complete, cancel, retry."""

import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from typing import Dict, List, Optional

import core.state
from core.audit import log_system_event
from core.cancel_signal import clear_cancel_signal, signal_cancel
from core.db_pool import acquire_db
from core.deps import get_current_user
from core.queue import enqueue_job
from core.r2 import _delete_r2_objects, generate_presigned_upload_url
from core.wallet import refund_tokens
from core.models import UploadInit
from routers.preferences import get_user_prefs_for_upload
from services.upload.r2_storage_guard import (
    ERROR_SOURCE_NOT_IN_R2,
    ERROR_STORAGE_CHECK_UNAVAILABLE,
    SOURCE_NOT_IN_R2_MESSAGE,
    upload_source_definitely_missing_in_r2,
    upload_source_head_status,
    upload_source_present_in_r2,
)
from services.upload.inline_rescue import inline_rescue_if_stuck
from services.uploads_handlers import (
    ALLOWED_VIDEO_TYPES,
    complete_upload_transaction,
    compute_smart_schedule_display,
    estimate_upload_costs,
    presign_create_upload,
)
from services.retry_policy import (
    MAX_USER_RETRIES_DEFAULT,
    RETRY_IDEMPOTENCY_TTL_SEC,
    bump_retry_metadata,
    classify_retry_error,
    get_retry_count,
    split_platform_results,
    upload_is_stale_processing,
)
from services.upload_funnel import emit_upload_funnel_event
from services.upload.status import CANCELLABLE_STATUSES
from stages.entitlements import get_entitlements_for_tier

logger = logging.getLogger("uploadm8-api")

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

_RETRYABLE_STATUSES = ("failed", "cancelled", "partial")


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
    billing_breakdown = pres.get("billing_breakdown")
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
            "billing_breakdown": billing_breakdown,
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

        try:
            from services.content_success_model import score_presign_init

            result["content_hotness_hint"] = score_presign_init(
                platforms=list(data.platforms or []),
                user_prefs=user_prefs,
                schedule_mode=data.schedule_mode,
            )
        except Exception:
            pass
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

    async with core.state.db_pool.acquire() as _ac:
        await log_system_event(
            _ac,
            user_id=str(user["id"]),
            action="UPLOAD_INITIATED",
            event_category="UPLOAD",
            resource_type="upload",
            resource_id=upload_id,
            details={
                "filename": data.filename,
                "platforms": list(data.platforms or []),
                "schedule_mode": data.schedule_mode,
                "put_cost": put_cost,
                "aic_cost": aic_cost,
                "file_size": data.file_size,
            },
            request=request,
        )

    emit_upload_funnel_event(upload_id, "presign_ok", {"put_cost": put_cost, "aic_cost": aic_cost})
    return result


@router.post("/estimate")
async def estimate_upload(data: UploadInit, user: dict = Depends(get_current_user)):
    """Preview PUT/AIC costs and queue depth without reserving tokens or creating a row."""
    if core.state.db_pool is None:
        raise HTTPException(503, detail="Database unavailable")
    if data.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(400, detail=f"Unsupported file type: {data.content_type}")
    async with core.state.db_pool.acquire() as conn:
        return await estimate_upload_costs(conn, data, user)


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
    already_completed = bool(tx.get("already_completed"))

    ent = get_entitlements_for_tier(user.get("subscription_tier", "free"))

    emit_upload_funnel_event(upload_id, "r2_complete", {"schedule_mode": schedule_mode, "already_completed": already_completed})

    if already_completed:
        smart_schedule_display = compute_smart_schedule_display(schedule_mode, upload.get("schedule_metadata"))
        return {
            "status": new_status,
            "upload_id": upload_id,
            "schedule_mode": schedule_mode,
            "already_completed": True,
            "scheduled_time": upload["scheduled_time"].isoformat() if upload.get("scheduled_time") else None,
            "smart_schedule": smart_schedule_display,
            "processing_features": {
                "auto_captions": bool(user_prefs.get("auto_captions")) if ent.can_ai else False,
                "auto_thumbnails": bool(user_prefs.get("auto_thumbnails")) if ent.can_ai else False,
                "ai_hashtags": bool(user_prefs.get("ai_hashtags_enabled")) if ent.can_ai else False,
            },
        }

    if schedule_mode not in ("scheduled", "smart"):
        job_data = {
            "upload_id": upload_id,
            "user_id": str(user["id"]),
            "preferences": user_prefs,
            "plan_features": {
                "ai": ent.can_ai,
                "priority": ent.can_priority,
                "watermark": ent.can_watermark,
                "ai_depth": ent.ai_depth,
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
        background_tasks.add_task(
            inline_rescue_if_stuck,
            upload_id,
            str(user["id"]),
            user_prefs or {},
            ent,
        )

    smart_schedule_display = compute_smart_schedule_display(schedule_mode, upload.get("schedule_metadata"))

    await log_system_event(
        user_id=str(user["id"]),
        action="UPLOAD_SUBMITTED",
        event_category="UPLOAD",
        resource_type="upload",
        resource_id=upload_id,
        details={
            "schedule_mode": schedule_mode,
            "new_status": new_status,
            "platforms": list(upload.get("platforms") or []),
        },
        request=request,
    )

    return {
        "status": new_status,
        "upload_id": upload_id,
        "schedule_mode": schedule_mode,
        "scheduled_time": upload["scheduled_time"].isoformat() if upload.get("scheduled_time") else None,
        "smart_schedule": smart_schedule_display,
        "processing_features": {
            "auto_captions": bool(user_prefs.get("auto_captions")) if ent.can_ai else False,
            "auto_thumbnails": bool(user_prefs.get("auto_thumbnails")) if ent.can_ai else False,
            "ai_hashtags": bool(user_prefs.get("ai_hashtags_enabled")) if ent.can_ai else False,
        },
    }


@router.post("/{upload_id}/requeue")
async def requeue_upload(
    upload_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """
    Re-submit a pending upload after ENQUEUE_FAILED or similar transient errors.

    For smart/scheduled rows, rebuilds missing publish slots first (same repair
    path as Retry) so SCHEDULE_INCOMPLETE / PUBLISH_SLOT_MISSING recoveries work.
    Clears error fields and re-runs the /complete transition (enqueue or stage).
    """
    from services.upload.schedule_guard import repair_upload_schedule
    from services.upload.status import is_requeueable_upload

    async with acquire_db(core.state.db_pool) as conn:
        upload = await conn.fetchrow(
            """
            SELECT id, user_id, status, r2_key, error_code, schedule_mode,
                   platforms, schedule_metadata, scheduled_time, platform_results
            FROM uploads WHERE id = $1 AND user_id = $2
            """,
            upload_id,
            user["id"],
        )
        if not upload:
            raise HTTPException(404, "Upload not found")
        if not is_requeueable_upload(str(upload["status"] or ""), upload.get("error_code")):
            raise HTTPException(
                400,
                detail={
                    "code": "not_requeueable",
                    "message": "Only pending uploads with a requeueable error can be re-queued. Use Retry for failed uploads.",
                },
            )
        head_status = upload_source_head_status(upload)
        if head_status == "missing":
            raise HTTPException(
                409,
                detail={
                    "code": ERROR_SOURCE_NOT_IN_R2,
                    "message": SOURCE_NOT_IN_R2_MESSAGE,
                    "hint": "Re-upload the video file, then re-queue.",
                },
            )
        if head_status == "unknown":
            raise HTTPException(
                503,
                detail={
                    "code": ERROR_STORAGE_CHECK_UNAVAILABLE,
                    "message": "Storage check temporarily unavailable. Try again in a moment.",
                },
            )

        mode = str(upload.get("schedule_mode") or "").strip().lower()
        if mode in ("smart", "scheduled"):
            ok, _, _ = await repair_upload_schedule(conn, dict(upload))
            if not ok:
                raise HTTPException(
                    400,
                    detail={
                        "code": "schedule_repair_failed",
                        "message": (
                            "Could not rebuild publish slots for this upload. "
                            "Edit the schedule on Scheduled, then re-queue."
                        ),
                    },
                )

        await conn.execute(
            """
            UPDATE uploads
            SET error_code = NULL,
                error_detail = NULL,
                updated_at = NOW()
            WHERE id = $1 AND status = 'pending'
            """,
            upload_id,
        )

    logger.warning("[%s] user requeue — re-running complete", upload_id)
    return await complete_upload(upload_id, request, background_tasks, user)


@router.post("/{upload_id}/reprepare")
async def reprepare_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Generate a fresh presigned R2 URL for an upload stuck in pending state.
    """
    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT id, r2_key, filename, status, telemetry_r2_key FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id,
            user["id"],
        )
        if not upload:
            raise HTTPException(404, "Upload not found")
        if upload["status"] not in ("pending",):
            raise HTTPException(
                400, f"Upload is not resumable (status: {upload['status']}). Use /retry for failed uploads."
            )

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
            upload_id,
            user["id"],
        )
        if not upload:
            raise HTTPException(404, "Upload not found")
        current_status = (upload["status"] or "").lower()
        if current_status in ("completed", "succeeded", "cancelled", "failed"):
            raise HTTPException(400, "Cannot cancel this upload")

        if current_status not in CANCELLABLE_STATUSES and current_status != "processing":
            raise HTTPException(
                400,
                detail={
                    "code": "not_cancellable",
                    "message": (
                        "This upload is already being prepared for publish and cannot be cancelled. "
                        "Edit metadata from Queue or contact support if you need help."
                    ),
                },
            )

        if current_status == "processing":
            await conn.execute(
                "UPDATE uploads SET cancel_requested = TRUE, updated_at = NOW() WHERE id = $1",
                upload_id,
            )
            await signal_cancel(core.state.redis_client, upload_id)
            await log_system_event(
                conn,
                user_id=str(user["id"]),
                action="UPLOAD_CANCEL_REQUESTED",
                event_category="UPLOAD",
                resource_type="upload",
                resource_id=upload_id,
                details={"status_at_cancel": current_status},
                request=request,
            )
            return {"status": "cancel_requested", "message": "Cancel signal sent -- job will stop at next checkpoint"}

        await conn.execute(
            "UPDATE uploads SET cancel_requested = TRUE, status = 'cancelled', updated_at = NOW() WHERE id = $1",
            upload_id,
        )
        await refund_tokens(conn, user["id"], upload["put_reserved"], upload["aic_reserved"], upload_id)
        await clear_cancel_signal(core.state.redis_client, upload_id)
        await log_system_event(
            conn,
            user_id=str(user["id"]),
            action="UPLOAD_CANCELLED",
            event_category="UPLOAD",
            resource_type="upload",
            resource_id=upload_id,
            details={"status_at_cancel": current_status},
            request=request,
            severity="WARNING",
        )
        r2_keys = [
            k
            for k in (
                upload.get("processed_r2_key"),
                upload.get("thumbnail_r2_key"),
            )
            if k
        ]
        if r2_keys:
            await _delete_r2_objects(r2_keys)
        return {"status": "cancelled"}


@router.post("/{upload_id}/retry")
async def retry_upload(
    upload_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Re-queue a failed / cancelled / partial upload for processing."""
    user_id_str = str(user["id"])

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
            upload_id,
            user["id"],
        )
        if not upload:
            raise HTTPException(404, "Upload not found")

        current_status = (upload["status"] or "").lower()
        stale_processing_retry = current_status == "processing" and upload_is_stale_processing(upload)
        if current_status not in _RETRYABLE_STATUSES and not stale_processing_retry:
            raise HTTPException(
                400,
                f"Upload status '{current_status}' is not retryable. "
                f"Allowed: {', '.join(_RETRYABLE_STATUSES)}"
                + (" or stale processing (20+ min without progress)." if current_status == "processing" else "."),
            )

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

        from core.helpers import coerce_output_artifacts_dict

        # Legacy rows may store a JSON array; dict(list) 500s (UPLOADM8-7W).
        existing_artifacts = coerce_output_artifacts_dict(upload.get("output_artifacts"))
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

        prior_platform_results = upload.get("platform_results")
        if isinstance(prior_platform_results, str):
            try:
                prior_platform_results = json.loads(prior_platform_results)
            except Exception:
                prior_platform_results = []
        succeeded_entries, failed_platforms = split_platform_results(prior_platform_results)

        # Smart / scheduled: rebuild missing per-platform slots before re-queue
        # (PUBLISH_SLOT_MISSING / SCHEDULE_INCOMPLETE recoveries).
        mode = str(upload.get("schedule_mode") or "").strip().lower()
        if mode in ("smart", "scheduled"):
            from services.upload.schedule_guard import repair_upload_schedule

            ok, _, _ = await repair_upload_schedule(conn, dict(upload))
            if not ok:
                raise HTTPException(
                    400,
                    detail={
                        "code": "schedule_repair_failed",
                        "message": (
                            "Could not rebuild publish slots for this upload. "
                            "Edit the schedule on Scheduled, then retry."
                        ),
                    },
                )

        retry_mode = "full"
        retry_subset: Optional[List[str]] = None
        seeded_results: Optional[List[Dict]] = None

        if current_status == "partial":
            if not failed_platforms:
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
            seeded_results = succeeded_entries

        new_artifacts = bump_retry_metadata(
            existing_artifacts,
            actor_user_id=user_id_str,
            prior_error_code=upload.get("error_code"),
            mode=retry_mode,
            retry_platforms=retry_subset,
        )

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
            upload_id,
            user["id"],
            json.dumps(new_artifacts),
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

    await clear_cancel_signal(core.state.redis_client, upload_id)

    ent = get_entitlements_for_tier(user.get("subscription_tier", "free"))

    job_data: Dict = {
        "upload_id": upload_id,
        "user_id": user_id_str,
        "preferences": user_prefs,
        "plan_features": {
            "ai": ent.can_ai,
            "priority": ent.can_priority,
            "watermark": ent.can_watermark,
            "ai_depth": ent.ai_depth,
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
