"""Upload complete transaction (staged / queued transition)."""

from __future__ import annotations

import re
from typing import Any, List

from fastapi import HTTPException

from core.helpers import _safe_col, sanitize_hashtag_body
from core.sql_allowlist import UPLOADS_COMPLETE_BODY_COLUMNS, assert_set_fragments_columns
from routers.preferences import get_user_prefs_for_upload

from services.upload.schedule_guard import (
    ERROR_SCHEDULE_INCOMPLETE,
    UPLOAD_ERROR_MESSAGES,
    loud_upload_schedule_failure,
    mark_schedule_incomplete_failed,
    repair_upload_schedule,
    upload_has_schedule,
)
from services.upload.status import COMPLETE_IDEMPOTENT_STATUSES
from services.upload.tiktok import validate_upload_row_tiktok_settings


async def complete_upload_transaction(conn, upload_id: str, user_id: str, body: dict) -> dict:
    """
    Apply optional title/caption/hashtags, set staged or queued.

    Presign owns platforms, schedule, TikTok settings, and billing reservation.
    Complete owns optional title/caption/hashtags patch only.

    Returns:
    new_status, schedule_mode, upload (dict snapshot), user_prefs,
    already_completed (bool) when status was already in-flight or terminal.
    """
    upload = await conn.fetchrow(
        "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
        upload_id,
        user_id,
    )
    if not upload:
        raise HTTPException(404, "Upload not found")

    user_prefs = await get_user_prefs_for_upload(conn, user_id)

    current_status = (upload.get("status") or "").lower()
    if current_status in ("failed", "cancelled"):
        raise HTTPException(
            409,
            detail={
                "code": "not_completable",
                "message": f"Upload status '{current_status}' cannot be completed. Use Retry or Reprepare.",
                "hint": "Retry from Queue for failed uploads, or Reprepare if the file never reached storage.",
            },
        )
    if current_status in COMPLETE_IDEMPOTENT_STATUSES:
        upload_dict = dict(upload)
        return {
            "new_status": current_status,
            "schedule_mode": upload.get("schedule_mode") or "immediate",
            "upload": upload_dict,
            "user_prefs": user_prefs,
            "already_completed": True,
        }

    schedule_mode = upload.get("schedule_mode") or "immediate"

    _COMPLETE_COLS = UPLOADS_COMPLETE_BODY_COLUMNS
    updates: List[str] = []
    params: List[Any] = []
    idx = 1
    if body.get("title") is not None:
        updates.append(f"{_safe_col('title', _COMPLETE_COLS)} = ${idx}")
        params.append(str(body["title"])[:512])
        idx += 1
    if body.get("caption") is not None:
        updates.append(f"{_safe_col('caption', _COMPLETE_COLS)} = ${idx}")
        params.append(str(body["caption"])[:10000])
        idx += 1
    if body.get("hashtags") is not None:
        raw_tags = body["hashtags"]
        if isinstance(raw_tags, str):
            raw_tags = [t.strip() for t in re.split(r"[\s,]+", str(raw_tags)) if t.strip()]
        tags: List[str] = []
        for t in (raw_tags if isinstance(raw_tags, (list, tuple)) else []):
            tag_body = sanitize_hashtag_body(str(t))
            if tag_body:
                tags.append(f"#{tag_body}")
        blocked = set(
            str(x).strip().lstrip("#").lower()
            for x in (user_prefs.get("blocked_hashtags") or user_prefs.get("blockedHashtags") or [])
        )
        tags = [t for t in tags if t and t.lstrip("#").lower() not in blocked]
        tags = list(dict.fromkeys(tags))[: int(user_prefs.get("max_hashtags", 30))]
        updates.append(f"{_safe_col('hashtags', _COMPLETE_COLS)} = ${idx}")
        params.append(tags)
        idx += 1

    if any(
        k in body
        for k in ("vehicle_make_id", "vehicleMakeId", "vehicle_model_id", "vehicleModelId")
    ):
        vm_raw = body.get("vehicle_make_id", body.get("vehicleMakeId"))
        vmd_raw = body.get("vehicle_model_id", body.get("vehicleModelId"))
        try:
            vm_id = int(vm_raw) if vm_raw is not None and str(vm_raw).strip() != "" else None
        except (TypeError, ValueError):
            raise HTTPException(400, "Invalid vehicle_make_id") from None
        try:
            vmd_id = int(vmd_raw) if vmd_raw is not None and str(vmd_raw).strip() != "" else None
        except (TypeError, ValueError):
            raise HTTPException(400, "Invalid vehicle_model_id") from None
        if vm_id is not None and vmd_id is not None:
            ok = await conn.fetchrow(
                "SELECT 1 FROM vehicle_models WHERE id = $1 AND make_id = $2",
                vmd_id,
                vm_id,
            )
            if not ok:
                raise HTTPException(400, "Invalid vehicle model for selected make")
        elif vmd_id is not None and vm_id is None:
            raise HTTPException(400, "vehicle_make_id required when vehicle_model_id is set")
        updates.append(f"{_safe_col('vehicle_make_id', _COMPLETE_COLS)} = ${idx}")
        params.append(vm_id)
        idx += 1
        updates.append(f"{_safe_col('vehicle_model_id', _COMPLETE_COLS)} = ${idx}")
        params.append(vmd_id)
        idx += 1

    if updates:
        assert_set_fragments_columns(updates, UPLOADS_COMPLETE_BODY_COLUMNS)
        params.append(upload_id)
        await conn.execute(
            f"UPDATE uploads SET {', '.join(updates)}, updated_at = NOW() WHERE id = ${idx}",
            *params,
        )
        upload = await conn.fetchrow(
            "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id,
            user_id,
        )
        if not upload:
            raise HTTPException(404, "Upload not found")
        upload = dict(upload)

    await validate_upload_row_tiktok_settings(conn, dict(upload))

    if schedule_mode in ("scheduled", "smart"):
        upload_dict = dict(upload)
        if not upload_has_schedule(upload_dict):
            ok, _, _ = await repair_upload_schedule(conn, upload_dict)
            if not ok:
                detail = UPLOAD_ERROR_MESSAGES.get(
                    ERROR_SCHEDULE_INCOMPLETE,
                    "Schedule could not be generated for this upload.",
                )
                await mark_schedule_incomplete_failed(
                    conn,
                    upload_id,
                    detail=detail,
                )
                try:
                    import core.state as _state

                    if _state.db_pool:
                        await loud_upload_schedule_failure(
                            upload_id,
                            user_id,
                            reason=detail,
                            schedule_mode=schedule_mode,
                            db_pool=_state.db_pool,
                        )
                except Exception:
                    pass
                raise HTTPException(
                    409,
                    detail={
                        "code": ERROR_SCHEDULE_INCOMPLETE,
                        "message": detail,
                        "hint": "Pick a schedule time and try again, or contact support if this persists.",
                    },
                )
            upload = await conn.fetchrow(
                "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
                upload_id,
                user_id,
            )
            if not upload:
                raise HTTPException(404, "Upload not found")

        new_status = "staged"
        await conn.execute(
            "UPDATE uploads SET status = 'staged', updated_at = NOW() WHERE id = $1",
            upload_id,
        )
    else:
        new_status = "queued"
        await conn.execute(
            "UPDATE uploads SET status = 'queued', updated_at = NOW() WHERE id = $1",
            upload_id,
        )

    return {
        "new_status": new_status,
        "schedule_mode": schedule_mode,
        "upload": dict(upload),
        "user_prefs": user_prefs,
        "already_completed": False,
    }
