"""
UploadM8 Scheduled-uploads routes — extracted from app.py.
"""

from core.db_pool import acquire_db
import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from core.config import R2_BUCKET_NAME
import core.state
from core.deps import get_current_user, get_current_user_readonly
from core.wallet import refund_tokens
from core.helpers import _load_uploads_columns, _now_utc, _pick_cols, _safe_col
from core.sql_allowlist import UPLOADS_METADATA_PATCH_COLUMNS, assert_set_fragments_columns
from core.r2 import get_s3_client, _normalize_r2_key, resolve_stored_account_avatar_url
from core.models import SmartScheduleOnlyUpdate
from services.upload.list_detail import _upload_error_message
from services.retry_policy import upload_is_overdue_ready_to_publish
from services.upload.status import (
    CANCELLABLE_STATUSES,
    SCHEDULE_ATTENTION_ERROR_CODES,
    is_requeueable_upload,
    is_retryable_upload,
)
from services.uploads_handlers import SCHEDULED_PIPELINE_STATUSES, scheduled_in_clause
from services.shell_bootstrap import _fetch_platforms_bundle, _ok, run_schedule_repair_background
from services.upload.schedule_guard import schedule_slot_iso
from stages.entitlements import entitlements_to_dict, get_entitlements_from_user

logger = logging.getLogger(__name__)

_CANCELLABLE_SET = frozenset(CANCELLABLE_STATUSES)


def _is_cancellable(status: str) -> bool:
    return (status or "").lower() in _CANCELLABLE_SET


def _processing_window_minutes() -> int:
    import os

    try:
        return max(1, int(os.environ.get("PROCESSING_WINDOW_MINUTES", "15")))
    except (TypeError, ValueError):
        return 15


def _presign_thumbnail_r2_key(thumb_key: Any) -> Optional[str]:
    if not thumb_key or not str(thumb_key).strip():
        return None
    try:
        s3 = get_s3_client()
        return s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": R2_BUCKET_NAME,
                "Key": _normalize_r2_key(thumb_key),
            },
            ExpiresIn=3600,
        )
    except Exception:
        return None


def _smart_schedule_from_upload_row(upload: dict) -> Optional[dict]:
    """Read smart_schedule dict from a list/detail row (not PATCH validation)."""
    try:
        sm = upload.get("schedule_metadata")
        if sm and upload.get("schedule_mode") == "smart":
            return sm if isinstance(sm, dict) else json.loads(sm)
    except Exception:
        pass
    return None


async def _scheduled_stats_row(conn: Any, uid: str) -> dict[str, int]:
    """Pending / today / week — counts any smart slot in the window, not just scheduled_time."""
    from services.schedule_slots import compute_scheduled_stats

    ph, statuses = scheduled_in_clause(2)
    rows = await conn.fetch(
        f"""
        SELECT scheduled_time, schedule_mode, schedule_metadata
        FROM uploads
        WHERE user_id = $1
          AND status IN ({ph})
        """,
        uid,
        *statuses,
    )
    payload = []
    for row in rows:
        sm = row.get("schedule_metadata")
        if sm and isinstance(sm, str):
            try:
                sm = json.loads(sm)
            except Exception:
                sm = None
        payload.append(
            {
                "scheduled_time": row.get("scheduled_time"),
                "schedule_mode": row.get("schedule_mode"),
                "schedule_metadata": sm,
            }
        )
    return compute_scheduled_stats(payload, now=_now_utc())


async def _scheduled_upload_list_for_user(
    pool: Any,
    uid: str,
    *,
    presign_thumbnails: bool = False,
    presign_account_avatars: bool = False,
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        cols = await _load_uploads_columns(pool)
        select_cols = _pick_cols(_SCHEDULED_LIST_COLS, cols)
        if not select_cols:
            select_cols = [
                "id",
                "filename",
                "title",
                "scheduled_time",
                "platforms",
                "status",
                "created_at",
                "schedule_mode",
            ]
        col_sql = ", ".join(select_cols)
        ph, statuses = scheduled_in_clause(2)
        # Also surface recent terminal schedule failures so Retry is reachable
        # from scheduled.html (not only Queue).
        attn_codes = sorted(SCHEDULE_ATTENTION_ERROR_CODES)
        attn_ph = ", ".join(
            f"${i}" for i in range(2 + len(statuses), 2 + len(statuses) + len(attn_codes))
        )
        uploads = await conn.fetch(
            f"""
            SELECT {col_sql}
            FROM uploads
            WHERE user_id = $1
              AND (
                status IN ({ph})
                OR (
                  status = 'failed'
                  AND LOWER(COALESCE(schedule_mode, '')) IN ('smart', 'scheduled')
                  AND UPPER(COALESCE(error_code, '')) IN ({attn_ph})
                  AND created_at > NOW() - INTERVAL '14 days'
                )
              )
            ORDER BY
              CASE WHEN status = 'failed' THEN 0 ELSE 1 END,
              scheduled_time ASC NULLS LAST,
              created_at ASC
            """,
            uid,
            *statuses,
            *attn_codes,
        )

        all_token_ids: set[str] = set()
        normalized_by_row: list[List[str]] = []
        for upload in uploads:
            tids = _normalize_target_account_uuids(upload.get("target_accounts"))
            normalized_by_row.append(tids)
            all_token_ids.update(tids)

        token_rows_by_id: dict = {}
        if all_token_ids:
            tok_rows = await conn.fetch(
                """SELECT id, platform, account_name, account_username, account_avatar
                   FROM platform_tokens
                   WHERE user_id = $1 AND id = ANY($2::uuid[]) AND revoked_at IS NULL""",
                uid,
                list(all_token_ids),
            )
            for r in tok_rows:
                token_rows_by_id[str(r["id"])] = r

        result: list[dict[str, Any]] = []
        for upload, target_ids in zip(uploads, normalized_by_row):
            has_thumbnail = bool(upload.get("thumbnail_r2_key") and str(upload.get("thumbnail_r2_key")).strip())
            thumbnail_url = (
                _presign_thumbnail_r2_key(upload["thumbnail_r2_key"])
                if presign_thumbnails and has_thumbnail
                else None
            )
            smart_schedule = _smart_schedule_from_upload_row(upload)
            platform_results = upload.get("platform_results")
            if isinstance(platform_results, str):
                try:
                    platform_results = json.loads(platform_results)
                except Exception:
                    platform_results = None

            target_account_details = []
            for tid in target_ids:
                r = token_rows_by_id.get(tid)
                if not r:
                    continue
                target_account_details.append(
                    {
                        "id": str(r["id"]),
                        "platform": r["platform"],
                        "name": r["account_name"] or "",
                        "username": r["account_username"] or "",
                        "avatar": resolve_stored_account_avatar_url(
                            r["account_avatar"],
                            presign=presign_account_avatars,
                        )
                        or "",
                    }
                )

            result.append(
                {
                    "id": str(upload["id"]),
                    "filename": upload.get("filename") or "",
                    "title": upload.get("title") or "Untitled",
                    "scheduled_time": upload["scheduled_time"].isoformat()
                    if upload.get("scheduled_time")
                    else None,
                    "timezone": str(upload.get("timezone") or "UTC"),
                    "platforms": list(upload["platforms"]) if upload.get("platforms") else [],
                    "target_accounts": target_ids,
                    "target_account_details": target_account_details,
                    "thumbnail": thumbnail_url,
                    "has_thumbnail": has_thumbnail,
                    "caption": upload.get("caption"),
                    "status": upload["status"],
                    "schedule_mode": upload.get("schedule_mode") or "scheduled",
                    "smart_schedule": smart_schedule,
                    "platform_results": platform_results,
                    "is_editable": upload["status"] in SCHEDULED_PIPELINE_STATUSES,
                    "is_cancellable": _is_cancellable(upload["status"]),
                    "is_requeueable": is_requeueable_upload(
                        str(upload.get("status") or ""), upload.get("error_code")
                    ),
                    "is_retryable": is_retryable_upload(
                        str(upload.get("status") or ""),
                        error_code=upload.get("error_code"),
                        has_failed_platform=any(
                            str((r or {}).get("status") or "").lower()
                            in ("failed", "error", "cancelled", "canceled")
                            for r in (platform_results or [])
                            if isinstance(r, dict)
                        ),
                        overdue_ready=upload_is_overdue_ready_to_publish(dict(upload)),
                    ),
                    "is_overdue": upload_is_overdue_ready_to_publish(dict(upload)),
                    "error_code": upload.get("error_code"),
                    "error": _upload_error_message(dict(upload)),
                    "created_at": upload["created_at"].isoformat()
                    if upload.get("created_at")
                    else None,
                    "updated_at": upload["updated_at"].isoformat()
                    if upload.get("updated_at")
                    else None,
                }
            )
    return result

_SCHEDULED_LIST_COLS = [
    "id",
    "filename",
    "title",
    "scheduled_time",
    "platforms",
    "target_accounts",
    "thumbnail_r2_key",
    "caption",
    "status",
    "created_at",
    "updated_at",
    "timezone",
    "schedule_mode",
    "schedule_metadata",
    "platform_results",
    "error_code",
    "error_detail",
]

_SCHEDULED_DETAIL_COLS = [
    "id",
    "title",
    "scheduled_time",
    "platforms",
    "timezone",
    "thumbnail_r2_key",
    "caption",
    "hashtags",
    "privacy",
    "status",
    "created_at",
    "updated_at",
    "schedule_mode",
    "schedule_metadata",
    "platform_results",
    "error_code",
    "error_detail",
]


from services.scheduled_target_accounts import normalize_target_account_uuids as _normalize_target_account_uuids

router = APIRouter(prefix="/api/scheduled", tags=["scheduled"])


# ------------------------------------------------------------------
# GET /api/scheduled  — removed; use /bootstrap or /list
# ------------------------------------------------------------------
@router.get("")
async def get_scheduled_removed():
    raise HTTPException(
        410,
        detail={
            "code": "endpoint_removed",
            "message": "GET /api/scheduled was removed. Use GET /api/scheduled/bootstrap or GET /api/scheduled/list.",
            "successor": "/api/scheduled/bootstrap",
        },
    )


# ------------------------------------------------------------------
# GET /api/scheduled/stats
# ------------------------------------------------------------------
@router.get("/stats")
async def get_scheduled_stats(user: dict = Depends(get_current_user_readonly)):
    """Get scheduled upload statistics for the current user"""
    uid = user["id"]
    try:
        async with acquire_db(core.state.db_pool) as conn:
            return await _scheduled_stats_row(conn, uid)
    except Exception:
        logger.exception("get_scheduled_stats failed user_id=%s", uid)
        raise


# ------------------------------------------------------------------
# GET /api/scheduled/bootstrap — stats + list + platforms (one round-trip)
# ------------------------------------------------------------------
@router.get("/bootstrap")
async def get_scheduled_bootstrap(
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user_readonly),
):
    """
    First paint for scheduled.html: stats, upload list (no R2 presign), and
    platform accounts (redirect avatars, not presigned).

    Schedule repair runs in the background so first paint is not blocked.
    """
    uid = str(user["id"])
    pool = core.state.db_pool
    if pool is None:
        raise HTTPException(503, "Database unavailable")
    plan = entitlements_to_dict(get_entitlements_from_user(user))

    async def _stats():
        async with acquire_db(pool) as conn:
            return await _scheduled_stats_row(conn, uid)

    try:
        stats, uploads, platforms = await asyncio.gather(
            _stats(),
            _scheduled_upload_list_for_user(
                pool,
                uid,
                presign_thumbnails=False,
                presign_account_avatars=False,
            ),
            _fetch_platforms_bundle(pool, uid, plan, include_auth_errors=False),
            return_exceptions=True,
        )
    except Exception:
        logger.exception("get_scheduled_bootstrap failed user_id=%s", uid)
        raise

    background_tasks.add_task(run_schedule_repair_background, pool, uid)

    stats = _ok(stats)
    uploads = _ok(uploads)
    platforms = _ok(platforms)
    if stats is None and uploads is None:
        logger.warning("scheduled bootstrap degraded (stats+uploads unavailable) user_id=%s", uid)
        return {
            "stats": {},
            "uploads": [],
            "platforms": platforms if platforms is not None else [],
            "processing_window_minutes": _processing_window_minutes(),
            "schedule_repair": None,
        }

    return {
        "stats": stats or {},
        "uploads": uploads if uploads is not None else [],
        "platforms": platforms,
        "processing_window_minutes": _processing_window_minutes(),
        "schedule_repair": None,
    }


# ------------------------------------------------------------------
# GET /api/scheduled/list  — paginated list
# ------------------------------------------------------------------
@router.get("/list")
async def get_scheduled_list(
    user: dict = Depends(get_current_user_readonly),
    presign_thumbnails: bool = Query(
        False,
        description="Presign R2 thumbnail URLs (slow for large lists; use bootstrap + thumbnails/poll instead)",
    ),
):
    """Get list of all scheduled uploads for the current user"""
    uid = user["id"]
    try:
        return await _scheduled_upload_list_for_user(
            core.state.db_pool,
            uid,
            presign_thumbnails=presign_thumbnails,
            presign_account_avatars=presign_thumbnails,
        )
    except Exception:
        logger.exception("get_scheduled_list failed user_id=%s", uid)
        raise


def _infer_smart_spread_days(schedule_metadata: Any, default: int = 14) -> int:
    """Guess spread window from existing smart slots (for recalculate)."""
    from services.deferred_publish_schedule import parse_schedule_metadata

    slots = parse_schedule_metadata(schedule_metadata)
    if not slots:
        return default
    today = _now_utc().date()
    max_off = 0
    for dt in slots.values():
        max_off = max(max_off, (dt.date() - today).days)
    return max(7, min(730, max(default, max_off + 1)))


# ------------------------------------------------------------------
# POST /api/scheduled/{upload_id}/recalculate-smart
# ------------------------------------------------------------------
@router.post("/{upload_id:uuid}/recalculate-smart")
async def recalculate_smart_schedule(upload_id: UUID, user: dict = Depends(get_current_user)):
    """Recompute per-platform smart times (respects blocked days, excludes this upload)."""
    uid_str = str(upload_id)
    bill_id = str(user.get("billing_user_id") or user["id"])
    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            """
            SELECT id, status, schedule_mode, platforms, schedule_metadata, user_id
            FROM uploads
            WHERE id = $1 AND user_id = $2
            """,
            uid_str,
            user["id"],
        )
        if not upload:
            raise HTTPException(404, "Scheduled upload not found")
        if (upload.get("schedule_mode") or "").lower() != "smart":
            raise HTTPException(400, "Recalculate is only available for smart-scheduled uploads")
        if upload["status"] not in SCHEDULED_PIPELINE_STATUSES:
            raise HTTPException(400, "Cannot recalculate schedule for an upload that is no longer editable")

        platforms = list(upload["platforms"] or [])
        if not platforms:
            raise HTTPException(400, "Upload has no platforms")

        num_days = _infer_smart_spread_days(upload.get("schedule_metadata"))
        from services.upload.schedule_guard import build_smart_schedule_for_upload

        smart = await build_smart_schedule_for_upload(
            conn,
            bill_id,
            platforms,
            num_days=num_days,
            exclude_upload_id=uid_str,
            random_seed=str(uuid.uuid4()),
        )
        if not smart:
            raise HTTPException(
                500,
                detail={
                    "code": "schedule_generation_failed",
                    "message": "Could not generate new smart schedule times.",
                },
            )
        sm = {p: schedule_slot_iso(dt) for p, dt in smart.items()}
        await _apply_smart_schedule(conn, uid_str, user["id"], sm)

    scheduled_min = min(smart.values()).isoformat() if smart else None
    return {
        "status": "updated",
        "id": uid_str,
        "smart_schedule": sm,
        "scheduled_time": scheduled_min,
    }


# ------------------------------------------------------------------
# GET /api/scheduled/{upload_id}  — single upload detail
# ------------------------------------------------------------------
@router.get("/{upload_id:uuid}")
async def get_scheduled_upload(upload_id: UUID, user: dict = Depends(get_current_user)):
    """Get details of a specific scheduled upload (for edit form)"""
    uid_str = str(upload_id)
    async with core.state.db_pool.acquire() as conn:
        cols = await _load_uploads_columns(core.state.db_pool)
        select_cols = _pick_cols(_SCHEDULED_DETAIL_COLS, cols)
        if not select_cols:
            select_cols = ["id", "title", "scheduled_time", "platforms", "status", "created_at", "schedule_mode"]
        col_sql = ", ".join(select_cols)
        upload = await conn.fetchrow(
            f"""
            SELECT {col_sql}
            FROM uploads
            WHERE id = $1 AND user_id = $2
            """,
            uid_str,
            user["id"],
        )

        if not upload:
            raise HTTPException(404, "Scheduled upload not found")

        thumbnail_url = None
        tkey = upload.get("thumbnail_r2_key")
        if tkey:
            try:
                s3 = get_s3_client()
                thumbnail_url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": R2_BUCKET_NAME, "Key": _normalize_r2_key(tkey)},
                    ExpiresIn=3600,
                )
            except Exception:
                pass

        sm = upload.get("schedule_metadata")
        if sm and isinstance(sm, str):
            try:
                sm = json.loads(sm)
            except Exception:
                sm = None
        elif not isinstance(sm, dict):
            sm = None

        platform_results = upload.get("platform_results")
        if isinstance(platform_results, str):
            try:
                platform_results = json.loads(platform_results)
            except Exception:
                platform_results = None

    return {
        "id": str(upload["id"]),
        "title": upload.get("title") or "Untitled",
        "scheduled_time": upload["scheduled_time"].isoformat() if upload.get("scheduled_time") else None,
        "timezone": str(upload.get("timezone") or "UTC"),
        "platforms": list(upload["platforms"]) if upload.get("platforms") else [],
        "thumbnail": thumbnail_url,
        "caption": upload.get("caption"),
        "hashtags": list(upload["hashtags"]) if upload.get("hashtags") else [],
        "privacy": upload.get("privacy"),
        "status": upload["status"],
        "schedule_mode": upload.get("schedule_mode") or "scheduled",
        "schedule_metadata": sm,
        "smart_schedule": sm,  # alias for scheduled.html saveScheduledUpload()
        "platform_results": platform_results,
        "is_editable": upload.get("status") in SCHEDULED_PIPELINE_STATUSES,
        "is_cancellable": _is_cancellable(upload.get("status") or ""),
        "is_requeueable": is_requeueable_upload(
            str(upload.get("status") or ""), upload.get("error_code")
        ),
        "is_retryable": is_retryable_upload(
            str(upload.get("status") or ""),
            error_code=upload.get("error_code"),
            has_failed_platform=any(
                str((r or {}).get("status") or "").lower()
                in ("failed", "error", "cancelled", "canceled")
                for r in (platform_results or [])
                if isinstance(r, dict)
            ),
            overdue_ready=upload_is_overdue_ready_to_publish(dict(upload)),
        ),
        "is_overdue": upload_is_overdue_ready_to_publish(dict(upload)),
        "error_code": upload.get("error_code"),
        "error": _upload_error_message(dict(upload)),
        "created_at": upload["created_at"].isoformat() if upload.get("created_at") else None,
        "updated_at": upload["updated_at"].isoformat() if upload.get("updated_at") else None,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _parse_smart_schedule(sm: dict, upload_platforms: list) -> tuple:
    """
    Parse smart_schedule (platform -> ISO datetime string).
    Returns (schedule_metadata_json, scheduled_time_dt).
    Same logic as create flow: validate platforms, parse ISO, set scheduled_time = min.
    """
    if not isinstance(sm, dict):
        raise HTTPException(400, "smart_schedule must be a dict of platform -> ISO datetime string")
    if not sm:
        raise HTTPException(400, "smart_schedule requires per-platform times (non-empty object)")
    platforms = list(upload_platforms or [])
    for k in sm:
        if k not in platforms:
            raise HTTPException(400, f"smart_schedule platform '{k}' not in upload platforms")
    dts = []
    for v in sm.values():
        if not v:
            continue
        s = str(v).replace("Z", "+00:00").replace("z", "+00:00")
        try:
            dts.append(datetime.fromisoformat(s))
        except ValueError:
            raise HTTPException(400, "smart_schedule values must be valid ISO datetime strings")
    metadata = {k: v for k, v in sm.items() if v}
    scheduled_dt = min(dts) if dts else None
    return metadata, scheduled_dt


async def _apply_smart_schedule(conn, upload_id: str, user_id: str, sm: dict) -> None:
    """Apply smart_schedule to upload (same logic as create)."""
    upload = await conn.fetchrow(
        "SELECT id, status, platforms FROM uploads WHERE id = $1 AND user_id = $2",
        upload_id, user_id
    )
    if not upload:
        raise HTTPException(404, "Upload not found")
    if upload["status"] not in SCHEDULED_PIPELINE_STATUSES:
        raise HTTPException(400, "Cannot edit upload that is already processing or published")

    metadata, scheduled_dt = _parse_smart_schedule(sm, upload["platforms"])
    await conn.execute("""
        UPDATE uploads SET
            schedule_metadata = $1::jsonb,
            scheduled_time = $2,
            schedule_mode = 'smart',
            updated_at = NOW()
        WHERE id = $3 AND user_id = $4
    """, json.dumps(metadata), scheduled_dt, upload_id, user_id)


async def _update_upload_metadata(conn, upload_id: str, user_id: str, update_data) -> None:
    """PATCH /api/uploads - title, caption, hashtags, scheduled_time, smart_schedule."""
    upload = await conn.fetchrow(
        "SELECT id, status, platforms FROM uploads WHERE id = $1 AND user_id = $2",
        upload_id, user_id
    )
    if not upload:
        raise HTTPException(404, "Upload not found")
    if upload["status"] not in SCHEDULED_PIPELINE_STATUSES:
        raise HTTPException(400, "Cannot edit upload that is already processing or published")

    _SCHED_COLS = UPLOADS_METADATA_PATCH_COLUMNS
    updates = []
    params: list = [upload_id, user_id]
    param_count = 2

    if update_data.title is not None:
        param_count += 1
        updates.append(f"{_safe_col('title', _SCHED_COLS)} = ${param_count}")
        params.append(update_data.title)

    if update_data.caption is not None:
        param_count += 1
        updates.append(f"{_safe_col('caption', _SCHED_COLS)} = ${param_count}")
        params.append(update_data.caption)

    if update_data.hashtags is not None:
        param_count += 1
        updates.append(f"{_safe_col('hashtags', _SCHED_COLS)} = ${param_count}")
        params.append(update_data.hashtags)

    if update_data.scheduled_time is not None:
        param_count += 1
        updates.append(f"{_safe_col('scheduled_time', _SCHED_COLS)} = ${param_count}")
        params.append(update_data.scheduled_time)

    if update_data.smart_schedule is not None:
        metadata, scheduled_dt = _parse_smart_schedule(update_data.smart_schedule, upload["platforms"])
        param_count += 1
        updates.append(f"{_safe_col('schedule_metadata', _SCHED_COLS)} = ${param_count}::jsonb")
        params.append(json.dumps(metadata))
        if scheduled_dt is not None:
            param_count += 1
            updates.append(f"{_safe_col('scheduled_time', _SCHED_COLS)} = ${param_count}")
            params.append(scheduled_dt)
        param_count += 1
        updates.append(f"{_safe_col('schedule_mode', _SCHED_COLS)} = ${param_count}")
        params.append("smart")

    if not updates:
        raise HTTPException(400, "No updates provided")

    param_count += 1
    updates.append(f"{_safe_col('updated_at', _SCHED_COLS)} = ${param_count}")
    params.append(_now_utc())

    assert_set_fragments_columns(updates, UPLOADS_METADATA_PATCH_COLUMNS)
    await conn.execute(f"UPDATE uploads SET {', '.join(updates)} WHERE id = $1 AND user_id = $2", *params)


# ------------------------------------------------------------------
# PATCH /api/scheduled/{upload_id}  — update smart schedule
# ------------------------------------------------------------------
@router.patch("/{upload_id:uuid}")
async def update_scheduled_upload(
    upload_id: UUID,
    update_data: SmartScheduleOnlyUpdate,
    user: dict = Depends(get_current_user),
):
    """Update a scheduled upload's smart_schedule only (platform -> ISO datetime string)."""
    uid_str = str(upload_id)
    async with core.state.db_pool.acquire() as conn:
        await _apply_smart_schedule(conn, uid_str, user["id"], update_data.smart_schedule)
    return {"status": "updated", "id": uid_str}


# ------------------------------------------------------------------
# DELETE /api/scheduled/{upload_id}  — cancel a scheduled upload
# ------------------------------------------------------------------
@router.delete("/{upload_id:uuid}")
async def cancel_scheduled_upload(upload_id: UUID, user: dict = Depends(get_current_user)):
    """Cancel/delete a scheduled upload"""
    uid_str = str(upload_id)
    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow("""
            SELECT id, put_reserved, aic_reserved, status
            FROM uploads
            WHERE id = $1 AND user_id = $2
        """, uid_str, user["id"])

        if not upload:
            raise HTTPException(404, "Scheduled upload not found")

        # NOTE: cancel is intentionally narrower than SCHEDULED_PIPELINE_STATUSES.
        # 'staged' and 'ready_to_publish' rows are already claimed by the worker;
        # deleting them mid-publish risks double-spending refunded tokens. Users
        # see those rows in the scheduled list (is_editable=True for metadata
        # edits) but cannot fully cancel — this is by design.
        if upload["status"] not in _CANCELLABLE_SET:
            raise HTTPException(
                400,
                detail={
                    "code": "not_cancellable",
                    "message": "Cannot cancel upload that is already processing or completed",
                },
            )

        # Refund reserved tokens if any
        if upload["put_reserved"] > 0 or upload["aic_reserved"] > 0:
            await refund_tokens(
                conn,
                user["id"],
                upload["put_reserved"],
                upload["aic_reserved"],
                uid_str,
            )

        # Delete the upload
        await conn.execute("DELETE FROM uploads WHERE id = $1", uid_str)

    return {"status": "cancelled", "id": uid_str}
