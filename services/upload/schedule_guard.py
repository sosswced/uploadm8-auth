"""Guarantee scheduled/smart uploads always have concrete schedule times."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from core.scheduling import calculate_smart_schedule, get_existing_scheduled_days
from services.smart_schedule_insights import calculate_smart_schedule_data_driven
from services.upload.r2_storage_guard import ERROR_SOURCE_NOT_IN_R2, SOURCE_NOT_IN_R2_MESSAGE

logger = logging.getLogger("uploadm8-api")

ERROR_SCHEDULE_INCOMPLETE = "SCHEDULE_INCOMPLETE"
ERROR_SCHEDULE_NO_PLATFORMS = "SCHEDULE_NO_PLATFORMS"
ERROR_PUBLISH_SLOT_MISSING = "PUBLISH_SLOT_MISSING"
ERROR_STUCK_READY_TO_PUBLISH = "STUCK_READY_TO_PUBLISH"
ERROR_STUCK_PENDING = "STUCK_PENDING"
ERROR_ABANDONED_PENDING = "ABANDONED_PENDING"

UPLOAD_ERROR_MESSAGES: Dict[str, str] = {
    ERROR_SCHEDULE_INCOMPLETE: (
        "This upload has no publish time. Edit the schedule or retry — our team was alerted."
    ),
    ERROR_SCHEDULE_NO_PLATFORMS: "Select at least one platform before scheduling.",
    ERROR_PUBLISH_SLOT_MISSING: (
        "This upload has no remaining publish slot. It was marked failed so it cannot sit forever — "
        "retry or reschedule from Queue."
    ),
    ERROR_STUCK_READY_TO_PUBLISH: (
        "Publishing stalled past the safety window. Marked failed so it cannot hang — "
        "use Retry from Queue if the video is still needed."
    ),
    ERROR_STUCK_PENDING: (
        "This upload never finished registration (no publish schedule). Marked failed — "
        "upload again or use Re-queue if the file is still in storage."
    ),
    ERROR_ABANDONED_PENDING: (
        "Upload was left incomplete (file never finished registering). Marked failed."
    ),
    ERROR_SOURCE_NOT_IN_R2: SOURCE_NOT_IN_R2_MESSAGE,
    "ENQUEUE_FAILED": (
        "Upload saved but the processing queue was unavailable. We will retry automatically."
    ),
    "QUEUE_UNAVAILABLE": (
        "The processing queue was temporarily unavailable. Use Re-queue or wait for automatic retry."
    ),
    "STALE_PROCESSING": (
        "Processing stalled and was stopped. Use Retry from Queue if you still need this upload."
    ),
}


def schedule_slot_iso(v: Any) -> str:
    if v is None:
        return ""
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return str(v)


async def _user_timezone(conn: Any, user_id: str) -> str:
    row = await conn.fetchrow("SELECT timezone FROM users WHERE id = $1", user_id)
    tz = (row.get("timezone") if row else None) or "America/Chicago"
    return str(tz).strip() or "America/Chicago"


async def build_smart_schedule_for_upload(
    conn: Any,
    user_id: str,
    platforms: List[str],
    *,
    num_days: int = 14,
    exclude_upload_id: Optional[str] = None,
    random_seed: Optional[str] = None,
    user_timezone: Optional[str] = None,
) -> Dict[str, datetime]:
    """Return per-platform UTC slots; static priors when data-driven signals are empty."""
    if not platforms:
        return {}

    tz = user_timezone or await _user_timezone(conn, user_id)
    blocked = await get_existing_scheduled_days(
        conn, user_id, num_days, exclude_upload_id=exclude_upload_id
    )
    schedule = await calculate_smart_schedule_data_driven(
        conn,
        user_id,
        platforms,
        num_days=num_days,
        blocked_day_offsets=blocked or None,
        user_timezone=tz,
        random_seed=random_seed,
    )
    if schedule:
        return schedule

    logger.warning(
        "smart_schedule data-driven empty for user=%s platforms=%s — using static priors",
        user_id,
        platforms,
    )
    return calculate_smart_schedule(
        platforms,
        num_days=num_days,
        user_timezone=tz,
        blocked_day_offsets=blocked or None,
        random_seed=random_seed,
    )


def validate_presign_schedule(data: Any) -> None:
    """Reject presign when schedule_mode cannot produce concrete times."""
    mode = (getattr(data, "schedule_mode", None) or "immediate").strip().lower()
    platforms = list(getattr(data, "platforms", None) or [])

    if mode == "scheduled":
        if not getattr(data, "scheduled_time", None):
            logger.warning(
                "presign rejected: scheduled mode without scheduled_time platforms=%s",
                platforms,
            )
            raise HTTPException(
                400,
                detail={
                    "code": "schedule_required",
                    "message": "Pick a date and time for scheduled uploads.",
                    "hint": "Set scheduled_time in the presign request or switch to Upload Now.",
                },
            )
    elif mode == "smart":
        if not platforms:
            logger.warning("presign rejected: smart schedule without platforms")
            raise HTTPException(
                400,
                detail={
                    "code": ERROR_SCHEDULE_NO_PLATFORMS,
                    "message": UPLOAD_ERROR_MESSAGES[ERROR_SCHEDULE_NO_PLATFORMS],
                },
            )


def upload_has_schedule(upload_row: dict) -> bool:
    mode = (upload_row.get("schedule_mode") or "immediate").strip().lower()
    if mode not in ("scheduled", "smart"):
        return True
    return bool(upload_row.get("scheduled_time"))


# Pipeline statuses that may await a schedule before publish.
_AWAITING_SCHEDULE_STATUSES = (
    "pending",
    "staged",
    "queued",
    "scheduled",
    "ready_to_publish",
)


async def repair_upload_schedule(
    conn: Any,
    upload_row: dict,
    *,
    num_days: int = 14,
) -> Tuple[bool, Optional[datetime], Optional[dict]]:
    """
    Fill missing scheduled_time / schedule_metadata for staged smart/scheduled rows.

    Returns (ok, scheduled_time, schedule_metadata dict or None).
    """
    upload_id = str(upload_row.get("id") or "")
    user_id = str(upload_row.get("user_id") or "")
    mode = (upload_row.get("schedule_mode") or "immediate").strip().lower()
    platforms = list(upload_row.get("platforms") or [])

    if mode not in ("scheduled", "smart"):
        return True, upload_row.get("scheduled_time"), upload_row.get("schedule_metadata")

    scheduled_time = upload_row.get("scheduled_time")
    meta_raw = upload_row.get("schedule_metadata")
    schedule_metadata = meta_raw if isinstance(meta_raw, dict) else None
    if meta_raw and not schedule_metadata:
        try:
            parsed = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
            schedule_metadata = parsed if isinstance(parsed, dict) else None
        except Exception:
            schedule_metadata = None

    if scheduled_time is not None:
        if mode != "smart" or schedule_metadata:
            return True, scheduled_time, schedule_metadata

    if mode in ("smart", "scheduled"):
        if not platforms:
            logger.error("[%s] schedule repair failed: no platforms", upload_id)
            return False, None, None
        smart = await build_smart_schedule_for_upload(
            conn,
            user_id,
            platforms,
            num_days=num_days,
            exclude_upload_id=upload_id,
            random_seed=upload_id,
        )
        if not smart:
            logger.error("[%s] schedule repair failed: smart calc returned empty", upload_id)
            return False, None, None
        schedule_metadata = {p: schedule_slot_iso(dt) for p, dt in smart.items()}
        scheduled_time = min(smart.values())
    else:
        return False, None, None

    await conn.execute(
        """
        UPDATE uploads
        SET scheduled_time = $2,
            schedule_metadata = $3::jsonb,
            updated_at = NOW()
        WHERE id = $1
        """,
        upload_id,
        scheduled_time,
        json.dumps(schedule_metadata, default=str) if schedule_metadata else None,
    )
    logger.warning(
        "[%s] schedule repaired mode=%s scheduled_time=%s",
        upload_id,
        mode,
        scheduled_time,
    )
    return True, scheduled_time, schedule_metadata


async def mark_schedule_incomplete_failed(
    conn: Any,
    upload_id: str,
    *,
    detail: str,
    error_code: str = ERROR_SCHEDULE_INCOMPLETE,
) -> None:
    code = (error_code or ERROR_SCHEDULE_INCOMPLETE).strip().upper() or ERROR_SCHEDULE_INCOMPLETE
    await conn.execute(
        """
        UPDATE uploads
        SET status = 'failed',
            error_code = $2,
            error_detail = $3,
            updated_at = NOW()
        WHERE id = $1
          AND status NOT IN ('failed', 'cancelled', 'completed', 'succeeded', 'partial')
        """,
        upload_id,
        code,
        detail[:4000],
    )


async def bootstrap_repair_user_schedules(
    pool: Any,
    user_id: str,
    *,
    limit: int = 30,
) -> dict:
    """
    On queue/scheduled bootstrap: assign times to pending/awaiting rows missing them.

    Returns summary ``{repaired, failed}`` for logging.
    """
    repaired = 0
    failed = 0
    if pool is None:
        return {"repaired": 0, "failed": 0}

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, schedule_mode, platforms, schedule_metadata, scheduled_time, status
            FROM uploads
            WHERE user_id = $1
              AND schedule_mode IN ('scheduled', 'smart')
              AND status = ANY($2::text[])
              AND (
                    scheduled_time IS NULL
                    OR (schedule_mode = 'smart' AND (schedule_metadata IS NULL OR schedule_metadata = 'null'::jsonb))
                  )
            ORDER BY created_at ASC
            LIMIT $3
            """,
            user_id,
            list(_AWAITING_SCHEDULE_STATUSES),
            limit,
        )
        for row in rows:
            upload_id = str(row["id"])
            ok, _, _ = await repair_upload_schedule(conn, dict(row))
            if ok:
                repaired += 1
                logger.warning("[%s] bootstrap schedule repair OK (status=%s)", upload_id, row.get("status"))
            else:
                failed += 1
                detail = UPLOAD_ERROR_MESSAGES.get(
                    ERROR_SCHEDULE_INCOMPLETE,
                    "Schedule could not be generated for this upload.",
                )
                # Bootstrap repair must not flip rows to failed — worker/complete paths own that.
                await conn.execute(
                    """
                    UPDATE uploads
                    SET error_code = COALESCE(error_code, $2),
                        error_detail = COALESCE(error_detail, $3),
                        updated_at = NOW()
                    WHERE id = $1 AND status NOT IN ('failed', 'cancelled')
                    """,
                    upload_id,
                    ERROR_SCHEDULE_INCOMPLETE,
                    detail[:4000],
                )
                await loud_upload_schedule_failure(
                    upload_id,
                    user_id,
                    reason=detail,
                    schedule_mode=str(row.get("schedule_mode") or ""),
                    db_pool=pool,
                )

    if repaired or failed:
        logger.warning(
            "bootstrap schedule repair user=%s repaired=%s failed=%s",
            user_id,
            repaired,
            failed,
        )
    return {"repaired": repaired, "failed": failed}


async def loud_upload_schedule_failure(
    upload_id: str,
    user_id: str,
    *,
    reason: str,
    schedule_mode: str,
    db_pool: Any = None,
) -> None:
    """ERROR log + admin incident for schedule pipeline breaks."""
    logger.error(
        "[%s] UPLOAD_SCHEDULE_FAILURE user=%s mode=%s reason=%s",
        upload_id,
        user_id,
        schedule_mode,
        reason,
    )
    if not db_pool:
        return
    try:
        from stages.notify_stage import notify_admin_error

        await notify_admin_error(
            "upload_schedule_failure",
            {
                "upload_id": upload_id,
                "user_id": user_id,
                "schedule_mode": schedule_mode,
                "reason": reason,
            },
            db_pool,
        )
    except Exception:
        logger.exception("[%s] failed to notify admin for schedule failure", upload_id)
