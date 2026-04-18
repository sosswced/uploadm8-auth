"""
UploadM8 Scheduled-uploads routes — extracted from app.py.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from core.config import R2_BUCKET_NAME
import core.state
from core.deps import get_current_user
from core.wallet import refund_tokens
from core.helpers import _load_uploads_columns, _now_utc, _pick_cols, _safe_col
from core.sql_allowlist import UPLOADS_METADATA_PATCH_COLUMNS, assert_set_fragments_columns
from core.r2 import get_s3_client, _normalize_r2_key, resolve_stored_account_avatar_url
from core.models import SmartScheduleOnlyUpdate

logger = logging.getLogger(__name__)

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
    "timezone",
    "schedule_mode",
    "schedule_metadata",
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
    "schedule_mode",
    "schedule_metadata",
]


def _normalize_target_account_uuids(raw: Any) -> List[str]:
    """
    uploads.target_accounts is TEXT[]; values must be platform_tokens.id UUID strings.
    Invalid entries (or a mistaken string stored as JSON) must not reach ANY($1::uuid[]).
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                raw = parsed
            else:
                return []
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for x in raw:
        if x is None:
            continue
        s = str(x).strip()
        if not s or s in seen:
            continue
        try:
            out.append(str(uuid.UUID(s)))
            seen.add(s)
        except (ValueError, AttributeError, TypeError):
            continue
    return out

router = APIRouter(prefix="/api/scheduled", tags=["scheduled"])


# ------------------------------------------------------------------
# GET /api/scheduled  — list scheduled uploads
# ------------------------------------------------------------------
@router.get("")
async def get_scheduled(user: dict = Depends(get_current_user)):
    """Get scheduled uploads (scheduled + smart modes, all pending statuses)"""
    async with core.state.db_pool.acquire() as conn:
        uploads = await conn.fetch("""
            SELECT * FROM uploads
            WHERE user_id = $1
              AND status IN ('pending', 'queued', 'scheduled', 'staged', 'ready_to_publish')
            ORDER BY scheduled_time ASC NULLS LAST, created_at ASC
        """, user["id"])
    return [{"id": str(u["id"]), "filename": u["filename"], "platforms": u["platforms"], "status": u["status"], "title": u["title"], "scheduled_time": u["scheduled_time"].isoformat() if u["scheduled_time"] else None, "created_at": u["created_at"].isoformat() if u["created_at"] else None, "schedule_mode": u["schedule_mode"]} for u in uploads]


# ------------------------------------------------------------------
# GET /api/scheduled/stats
# ------------------------------------------------------------------
@router.get("/stats")
async def get_scheduled_stats(user: dict = Depends(get_current_user)):
    """Get scheduled upload statistics for the current user"""
    uid = user["id"]
    try:
        async with core.state.db_pool.acquire() as conn:
            now = _now_utc()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = today_start + timedelta(days=1)
            week_end = now + timedelta(days=7)

            # Count pending uploads (includes staged, ready_to_publish)
            pending_count = await conn.fetchval("""
                SELECT COUNT(*) FROM uploads
                WHERE user_id = $1
                AND scheduled_time IS NOT NULL
                AND scheduled_time > $2
                AND status IN ('pending', 'scheduled', 'queued', 'staged', 'ready_to_publish')
            """, uid, now)

            # Count uploads today
            today_count = await conn.fetchval("""
                SELECT COUNT(*) FROM uploads
                WHERE user_id = $1
                AND scheduled_time >= $2
                AND scheduled_time < $3
                AND status IN ('pending', 'scheduled', 'queued', 'staged', 'ready_to_publish')
            """, uid, today_start, today_end)

            # Count uploads in the next 7 days (rolling window from now)
            week_count = await conn.fetchval("""
                SELECT COUNT(*) FROM uploads
                WHERE user_id = $1
                AND scheduled_time >= $2
                AND scheduled_time < $3
                AND status IN ('pending', 'scheduled', 'queued', 'staged', 'ready_to_publish')
            """, uid, now, week_end)

        return {
            "pending": int(pending_count or 0),
            "today": int(today_count or 0),
            "week": int(week_count or 0),
        }
    except Exception:
        logger.exception("get_scheduled_stats failed user_id=%s", uid)
        raise


# ------------------------------------------------------------------
# GET /api/scheduled/list  — paginated list
# ------------------------------------------------------------------
@router.get("/list")
async def get_scheduled_list(user: dict = Depends(get_current_user)):
    """Get list of all scheduled uploads for the current user"""
    uid = user["id"]
    try:
        async with core.state.db_pool.acquire() as conn:
            cols = await _load_uploads_columns(core.state.db_pool)
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
            uploads = await conn.fetch(
                f"""
                SELECT {col_sql}
                FROM uploads
                WHERE user_id = $1
                AND status IN ('pending', 'scheduled', 'queued', 'staged', 'ready_to_publish')
                ORDER BY scheduled_time ASC NULLS LAST, created_at ASC
                """,
                uid,
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

            result = []
            for upload, target_ids in zip(uploads, normalized_by_row):
                thumbnail_url = None
                if upload.get("thumbnail_r2_key"):
                    try:
                        s3 = get_s3_client()
                        thumbnail_url = s3.generate_presigned_url(
                            "get_object",
                            Params={
                                "Bucket": R2_BUCKET_NAME,
                                "Key": _normalize_r2_key(upload["thumbnail_r2_key"]),
                            },
                            ExpiresIn=3600,
                        )
                    except Exception:
                        pass

                smart_schedule = None
                try:
                    sm = upload.get("schedule_metadata")
                    if sm and upload.get("schedule_mode") == "smart":
                        smart_schedule = sm if isinstance(sm, dict) else json.loads(sm)
                except Exception:
                    pass

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
                            "avatar": resolve_stored_account_avatar_url(r["account_avatar"]) or "",
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
                        "caption": upload.get("caption"),
                        "status": upload["status"],
                        "schedule_mode": upload.get("schedule_mode") or "scheduled",
                        "smart_schedule": smart_schedule,
                        "is_editable": upload["status"]
                        in ("pending", "staged", "queued", "scheduled", "ready_to_publish"),
                        "created_at": upload["created_at"].isoformat()
                        if upload.get("created_at")
                        else None,
                    }
                )

        return result
    except Exception:
        logger.exception("get_scheduled_list failed user_id=%s", uid)
        raise


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
        "is_editable": upload.get("status") in ("pending", "staged", "queued", "scheduled", "ready_to_publish"),
        "created_at": upload["created_at"].isoformat() if upload.get("created_at") else None,
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
    editable = ("pending", "scheduled", "queued", "staged", "ready_to_publish")
    if upload["status"] not in editable:
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
    editable = ("pending", "scheduled", "queued", "staged", "ready_to_publish")
    if upload["status"] not in editable:
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

        if upload["status"] not in ['pending', 'scheduled', 'queued']:
            raise HTTPException(400, "Cannot cancel upload that is already processing or completed")

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
