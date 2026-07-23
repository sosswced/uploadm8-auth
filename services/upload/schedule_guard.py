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
        "use Retry from Queue or Scheduled (slots are rebuilt on retry)."
    ),
    ERROR_STUCK_READY_TO_PUBLISH: (
        "Publishing stalled past the safety window. Marked failed so it cannot hang — "
        "use Retry from Queue or Scheduled if the video is still needed."
    ),
    ERROR_STUCK_PENDING: (
        "This upload never finished registration (no publish schedule). Marked failed — "
        "upload again, or Retry / Re-queue if the file is still in storage."
    ),
    ERROR_ABANDONED_PENDING: (
        "Upload was left incomplete (file never finished registering). Marked failed — "
        "upload again, or Retry if the file is still in storage."
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


def normalize_platform_list(platforms: Any) -> List[str]:
    """Lowercase unique platforms preserving first-seen order."""
    out: List[str] = []
    seen: set[str] = set()
    for raw in platforms or []:
        p = str(raw or "").strip().lower()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def smart_metadata_missing_platforms(
    platforms: Any,
    schedule_metadata: Any,
) -> List[str]:
    """Platforms that lack a resolvable datetime in ``schedule_metadata``."""
    from services.deferred_publish_schedule import parse_schedule_metadata

    plats = normalize_platform_list(platforms)
    meta = parse_schedule_metadata(schedule_metadata)
    return [p for p in plats if p not in meta]


_SUCCESS_PLATFORM_STATUSES = frozenset(
    {"published", "succeeded", "success", "completed", "partial"}
)


def successful_platforms_from_results(platform_results: Any) -> set[str]:
    """Lowercase platforms already marked successful in ``platform_results``."""
    from services.retry_policy import split_platform_results

    succeeded, _ = split_platform_results(
        platform_results if isinstance(platform_results, list) else None
    )
    out: set[str] = set()
    for entry in succeeded:
        name = str(entry.get("platform") or "").strip().lower()
        if name:
            out.add(name)
    # Also accept status-shaped rows when ``success`` was omitted.
    raw = platform_results
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = None
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            if entry.get("success") is True:
                continue  # already counted
            st = str(entry.get("status") or "").strip().lower()
            if st in _SUCCESS_PLATFORM_STATUSES:
                name = str(entry.get("platform") or "").strip().lower()
                if name:
                    out.add(name)
    return out


def anchor_platforms_for_scheduled_time(
    platforms: List[str],
    platform_results: Any,
) -> List[str]:
    """Platforms that should drive top-level ``scheduled_time`` (exclude published)."""
    done = successful_platforms_from_results(platform_results)
    pending = [p for p in platforms if p not in done]
    return pending if pending else list(platforms)


def min_slot_among(
    slots: Dict[str, datetime],
    platforms: List[str],
) -> Optional[datetime]:
    times = [slots[p] for p in platforms if p in slots]
    return min(times) if times else None


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
    """Return per-platform UTC slots (lowercase keys) for every requested platform."""
    plats = normalize_platform_list(platforms)
    if not plats:
        return {}

    from core.scheduling import clamp_smart_schedule_days

    num_days = clamp_smart_schedule_days(num_days)
    tz = user_timezone or await _user_timezone(conn, user_id)
    blocked = await get_existing_scheduled_days(
        conn, user_id, num_days, exclude_upload_id=exclude_upload_id
    )
    schedule = await calculate_smart_schedule_data_driven(
        conn,
        user_id,
        plats,
        num_days=num_days,
        blocked_day_offsets=blocked or None,
        user_timezone=tz,
        random_seed=random_seed,
    )
    if not schedule:
        logger.warning(
            "smart_schedule data-driven empty for user=%s platforms=%s — using static priors",
            user_id,
            plats,
        )
        schedule = calculate_smart_schedule(
            plats,
            num_days=num_days,
            user_timezone=tz,
            blocked_day_offsets=blocked or None,
            random_seed=random_seed,
        )

    # Guarantee every platform has a slot (unknown platforms fall back to tiktok priors).
    out: Dict[str, datetime] = {}
    for p in plats:
        slot = schedule.get(p) if schedule else None
        if slot is None and schedule:
            # Case-insensitive lookup if builder returned mixed keys
            for k, v in schedule.items():
                if str(k).strip().lower() == p:
                    slot = v
                    break
        if slot is not None:
            out[p] = slot
    if len(out) < len(plats):
        missing = [p for p in plats if p not in out]
        logger.warning(
            "smart_schedule incomplete user=%s missing=%s — filling with static priors",
            user_id,
            missing,
        )
        filler = calculate_smart_schedule(
            missing,
            num_days=num_days,
            user_timezone=tz,
            blocked_day_offsets=blocked or None,
            random_seed=f"{random_seed or user_id}:fill",
        )
        for p in missing:
            if p in filler:
                out[p] = filler[p]
    return out


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
        from core.scheduling import clamp_smart_schedule_days

        raw_days = getattr(data, "smart_schedule_days", 14)
        try:
            days_i = int(raw_days)
        except (TypeError, ValueError):
            days_i = -1
        if days_i < 1 or days_i > 730:
            # Pydantic also enforces ge/le; this catches raw dict / repair callers.
            clamped = clamp_smart_schedule_days(raw_days)
            try:
                setattr(data, "smart_schedule_days", clamped)
            except Exception:
                pass
    elif mode not in ("immediate", "scheduled", "smart"):
        raise HTTPException(
            400,
            detail={
                "code": "invalid_schedule_mode",
                "message": "schedule_mode must be immediate, scheduled, or smart.",
            },
        )


def upload_has_schedule(upload_row: dict) -> bool:
    """True when the row is ready to complete without schedule repair.

    Smart mode requires a top-level ``scheduled_time`` **and** a resolvable slot
    for every platform in ``schedule_metadata``. Partial metadata must return
    False so ``/complete`` runs ``repair_upload_schedule`` before staging.
    """
    mode = (upload_row.get("schedule_mode") or "immediate").strip().lower()
    if mode not in ("scheduled", "smart"):
        return True
    if not upload_row.get("scheduled_time"):
        return False
    if mode == "smart":
        missing = smart_metadata_missing_platforms(
            upload_row.get("platforms"),
            upload_row.get("schedule_metadata"),
        )
        if missing:
            return False
    return True


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
    Fill missing ``scheduled_time`` / incomplete ``schedule_metadata`` for smart/scheduled rows.

    Smart mode requires a resolvable slot for **every** platform. Partial metadata
    (e.g. only tiktok after a multi-platform presign) is rebuilt for missing platforms
    and merged — existing slots are preserved.

    Returns (ok, scheduled_time, schedule_metadata dict or None).
    """
    from services.deferred_publish_schedule import parse_schedule_metadata

    upload_id = str(upload_row.get("id") or "")
    user_id = str(upload_row.get("user_id") or "")
    mode = (upload_row.get("schedule_mode") or "immediate").strip().lower()
    platforms = normalize_platform_list(upload_row.get("platforms") or [])

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

    existing = parse_schedule_metadata(schedule_metadata) if schedule_metadata else {}
    missing = [p for p in platforms if p not in existing] if mode == "smart" else []
    anchor_plats = anchor_platforms_for_scheduled_time(
        platforms, upload_row.get("platform_results")
    )

    # Scheduled mode: only need a top-level scheduled_time.
    if mode == "scheduled" and scheduled_time is not None:
        return True, scheduled_time, schedule_metadata

    # Smart mode: complete metadata — do not rebuild slots. If scheduled_time is
    # null, derive the anchor from pending (non-published) platform slots only.
    if mode == "smart" and existing and not missing:
        normalized = {p: schedule_slot_iso(existing[p]) for p in platforms if p in existing}
        if scheduled_time is not None:
            return True, scheduled_time, normalized or schedule_metadata
        derived = min_slot_among(existing, anchor_plats)
        if derived is None:
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
            derived,
            json.dumps(normalized, default=str) if normalized else None,
        )
        logger.warning(
            "[%s] schedule anchor derived from metadata scheduled_time=%s platforms=%s",
            upload_id,
            derived,
            anchor_plats,
        )
        return True, derived, normalized or schedule_metadata

    if not platforms:
        logger.error("[%s] schedule repair failed: no platforms", upload_id)
        return False, None, None

    # Rebuild missing (or all) platform slots.
    need = missing if (mode == "smart" and existing and missing) else platforms
    smart = await build_smart_schedule_for_upload(
        conn,
        user_id,
        need,
        num_days=num_days,
        exclude_upload_id=upload_id,
        random_seed=upload_id,
    )
    if not smart:
        logger.error("[%s] schedule repair failed: smart calc returned empty", upload_id)
        return False, None, None

    merged: Dict[str, datetime] = dict(existing)
    for p, dt in smart.items():
        pl = str(p).strip().lower()
        if pl:
            merged[pl] = dt

    # Ensure every requested platform is present after merge.
    still_missing = [p for p in platforms if p not in merged]
    if still_missing:
        logger.error(
            "[%s] schedule repair failed: still missing platforms=%s",
            upload_id,
            still_missing,
        )
        return False, None, None

    schedule_metadata = {p: schedule_slot_iso(merged[p]) for p in platforms}
    # Never rewind the row anchor to an already-published platform's past slot.
    scheduled_time = min_slot_among(merged, anchor_plats) or min(merged[p] for p in platforms)

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
        "[%s] schedule repaired mode=%s scheduled_time=%s platforms=%s anchor=%s",
        upload_id,
        mode,
        scheduled_time,
        platforms,
        anchor_plats,
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

    Includes smart rows with **partial** ``schedule_metadata`` (some platforms
    missing slots), not only null metadata.

    Returns summary ``{repaired, failed}`` for logging.
    """
    repaired = 0
    failed = 0
    if pool is None:
        return {"repaired": 0, "failed": 0}

    async with pool.acquire() as conn:
        # Over-fetch smart/scheduled awaiting rows; filter with upload_has_schedule
        # so partial metadata (e.g. only tiktok) is repaired, not only NULL meta.
        rows = await conn.fetch(
            """
            SELECT id, user_id, schedule_mode, platforms, schedule_metadata, scheduled_time, status
            FROM uploads
            WHERE user_id = $1
              AND schedule_mode IN ('scheduled', 'smart')
              AND status = ANY($2::text[])
            ORDER BY created_at ASC
            LIMIT $3
            """,
            user_id,
            list(_AWAITING_SCHEDULE_STATUSES),
            max(limit * 3, 60),
        )
        checked = 0
        for row in rows:
            if checked >= limit:
                break
            row_d = dict(row)
            if upload_has_schedule(row_d):
                continue
            checked += 1
            upload_id = str(row["id"])
            ok, _, _ = await repair_upload_schedule(conn, row_d)
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
