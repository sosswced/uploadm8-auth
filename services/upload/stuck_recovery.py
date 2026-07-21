"""Worker-side recovery for uploads stuck in pending/staged/ready_to_publish states.

Fail-fast contract:
- Missing schedule that cannot be repaired → ``failed`` (not soft-annotate forever).
- ``ready_to_publish`` past due with no recoverable slot → ``failed``.
- ``ready_to_publish`` overdue but still due → re-dispatch publish.
- Abandoned ``pending`` (never completed) past age → ``failed``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from services.deferred_publish_schedule import (
    next_due_scheduled_time,
    platforms_due_for_publish,
    still_has_pending_publish_slots,
)
from services.retry_policy import (
    MAX_AUTO_RETRIES_DEFAULT,
    auto_retry_backoff_minutes,
    bump_auto_retry_metadata,
    get_auto_retry_count,
    should_auto_retry_upload,
)
from services.upload.r2_storage_guard import (
    ERROR_SOURCE_NOT_IN_R2,
    SOURCE_NOT_IN_R2_MESSAGE,
    mark_source_not_in_r2_failed,
    upload_source_head_status,
)
from services.upload.schedule_guard import (
    ERROR_ABANDONED_PENDING,
    ERROR_PUBLISH_SLOT_MISSING,
    ERROR_SCHEDULE_INCOMPLETE,
    ERROR_STUCK_PENDING,
    ERROR_STUCK_READY_TO_PUBLISH,
    UPLOAD_ERROR_MESSAGES,
    loud_upload_schedule_failure,
    mark_schedule_incomplete_failed,
    repair_upload_schedule,
    upload_has_schedule,
)

logger = logging.getLogger("uploadm8-worker")

EnqueueFn = Callable[[dict], Awaitable[bool]]
BuildPayloadFn = Callable[..., Awaitable[Optional[dict]]]
DeferredPublishFn = Callable[[str, str], Awaitable[Any]]


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


# How long ready_to_publish may sit past its due slot before we re-dispatch.
STUCK_READY_REDISPATCH_MINUTES = _env_int("STUCK_READY_REDISPATCH_MINUTES", 30)
# After this many minutes past due with no successful publish progress → fail.
STUCK_READY_FAIL_MINUTES = _env_int("STUCK_READY_FAIL_MINUTES", 120)
# Pending scheduled/smart without schedule after this age → fail.
STUCK_PENDING_FAIL_HOURS = _env_int("STUCK_PENDING_FAIL_HOURS", 6)
# Immediate/any pending never completed after this age → fail (abandoned upload).
ABANDONED_PENDING_HOURS = _env_int("ABANDONED_PENDING_HOURS", 24)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


async def recover_staged_without_schedule(
    conn: Any,
    db_pool: Any,
    *,
    limit: int = 20,
) -> int:
    """Repair or fail staged/pending scheduled uploads missing schedule completeness.

    Smart mode: also recovers rows with ``scheduled_time`` set but incomplete
    ``schedule_metadata`` (missing platform slots).
    """
    rows = await conn.fetch(
        """
        SELECT id, user_id, schedule_mode, platforms, schedule_metadata, scheduled_time, status
        FROM uploads
        WHERE status IN ('staged', 'pending')
          AND COALESCE(schedule_mode, 'immediate') IN ('scheduled', 'smart')
        ORDER BY updated_at ASC
        LIMIT $1
        """,
        max(limit * 3, 40),
    )
    handled = 0
    for row in rows:
        if handled >= limit:
            break
        row_d = dict(row)
        if upload_has_schedule(row_d):
            continue
        upload_id = str(row["id"])
        user_id = str(row["user_id"])
        status = str(row.get("status") or "")
        ok, _, _ = await repair_upload_schedule(conn, row_d)
        if ok:
            logger.warning(
                "[%s] stuck recovery: repaired missing schedule on %s upload",
                upload_id,
                status,
            )
            handled += 1
            continue
        # Pending soft-annotate only briefly; fail once past age so they don't sit forever.
        if status == "pending":
            age_hours = await conn.fetchval(
                """
                SELECT EXTRACT(EPOCH FROM (NOW() - COALESCE(created_at, updated_at))) / 3600.0
                FROM uploads WHERE id = $1
                """,
                row["id"],
            )
            if age_hours is not None and float(age_hours) < float(STUCK_PENDING_FAIL_HOURS):
                detail = UPLOAD_ERROR_MESSAGES.get(
                    ERROR_SCHEDULE_INCOMPLETE,
                    "Schedule could not be generated for this upload.",
                )
                await conn.execute(
                    """
                    UPDATE uploads
                    SET error_code = COALESCE(error_code, $2),
                        error_detail = COALESCE(error_detail, $3),
                        updated_at = NOW()
                    WHERE id = $1 AND status = 'pending'
                    """,
                    upload_id,
                    ERROR_SCHEDULE_INCOMPLETE,
                    detail[:4000],
                )
                handled += 1
                continue
            fail_code = ERROR_STUCK_PENDING
        else:
            fail_code = ERROR_SCHEDULE_INCOMPLETE

        detail = UPLOAD_ERROR_MESSAGES.get(
            fail_code,
            UPLOAD_ERROR_MESSAGES.get(
                ERROR_SCHEDULE_INCOMPLETE,
                "Schedule could not be generated for this upload.",
            ),
        )
        await mark_schedule_incomplete_failed(conn, upload_id, detail=detail, error_code=fail_code)
        await loud_upload_schedule_failure(
            upload_id,
            user_id,
            reason=detail,
            schedule_mode=str(row.get("schedule_mode") or ""),
            db_pool=db_pool,
        )
        logger.error(
            "[%s] stuck recovery: marked failed %s (no schedule time, was %s)",
            upload_id,
            fail_code,
            status,
        )
        handled += 1
    return handled


async def recover_enqueue_failed_pending(
    conn: Any,
    *,
    build_payload: BuildPayloadFn,
    enqueue: EnqueueFn,
    limit: int = 10,
) -> int:
    """Re-enqueue immediate uploads stuck in pending after ENQUEUE_FAILED."""
    rows = await conn.fetch(
        """
        SELECT id, user_id
        FROM uploads
        WHERE status = 'pending'
          AND error_code = 'ENQUEUE_FAILED'
          AND updated_at < NOW() - INTERVAL '2 minutes'
        ORDER BY updated_at ASC
        LIMIT $1
        """,
        limit,
    )
    recovered = 0
    for row in rows:
        upload_id = str(row["id"])
        user_id = str(row["user_id"])
        payload = await build_payload(
            upload_id,
            user_id,
            deferred=False,
            job_id=f"enqueue-recover-{upload_id}",
        )
        if not payload or not await enqueue(payload):
            logger.error("[%s] enqueue recovery failed — still pending ENQUEUE_FAILED", upload_id)
            continue
        await conn.execute(
            """
            UPDATE uploads
            SET status = 'queued',
                error_code = NULL,
                error_detail = NULL,
                updated_at = NOW()
            WHERE id = $1 AND status = 'pending'
            """,
            upload_id,
        )
        logger.warning("[%s] enqueue recovery: pending → queued after ENQUEUE_FAILED", upload_id)
        recovered += 1
    return recovered


async def recover_auto_retry_failed_immediate(
    conn: Any,
    *,
    build_payload: BuildPayloadFn,
    enqueue: EnqueueFn,
    limit: int = 8,
    max_retries: int = MAX_AUTO_RETRIES_DEFAULT,
) -> int:
    """Re-queue failed uploads with auto-retryable errors (exponential backoff).

    Immediate mode runs a full process job. Scheduled/smart modes repair publish
    slots first, then enqueue a deferred process job so publish waits for slots.
    """
    rows = await conn.fetch(
        """
        SELECT id, user_id, error_code, output_artifacts, schedule_mode, status,
               updated_at, platforms, schedule_metadata, scheduled_time, timezone,
               r2_key
        FROM uploads
        WHERE status = 'failed'
          AND COALESCE(schedule_mode, 'immediate') IN ('immediate', 'scheduled', 'smart')
          AND error_code IS NOT NULL
        ORDER BY updated_at ASC
        LIMIT $1
        """,
        limit * 3,
    )
    recovered = 0
    for row in rows:
        if recovered >= limit:
            break
        row_d = dict(row)
        if not should_auto_retry_upload(row_d, max_retries=max_retries):
            continue
        # Only fail hard when we know the key and R2 says missing (not when key
        # is absent from the row shape — that would poison every auto-retry).
        if row_d.get("r2_key") and upload_source_head_status(row_d) == "missing":
            await mark_source_not_in_r2_failed(
                conn,
                str(row["id"]),
                detail=SOURCE_NOT_IN_R2_MESSAGE,
            )
            logger.error(
                "[%s] auto-retry skipped: source missing in R2 — marked %s",
                row["id"],
                ERROR_SOURCE_NOT_IN_R2,
            )
            continue
        arts = row.get("output_artifacts")
        if isinstance(arts, str):
            try:
                arts = json.loads(arts)
            except Exception:
                arts = {}
        prior = get_auto_retry_count(arts if isinstance(arts, dict) else None)
        backoff = auto_retry_backoff_minutes(prior)
        stale_enough = await conn.fetchval(
            """
            SELECT (updated_at < NOW() - ($2::int * INTERVAL '1 minute'))::int
            FROM uploads WHERE id = $1
            """,
            row["id"],
            backoff,
        )
        if not stale_enough:
            continue

        upload_id = str(row["id"])
        user_id = str(row["user_id"])
        mode = str(row.get("schedule_mode") or "immediate").strip().lower()
        deferred = mode in ("scheduled", "smart")
        if deferred:
            from services.upload.schedule_guard import repair_upload_schedule

            ok, _, _ = await repair_upload_schedule(conn, row_d)
            if not ok:
                logger.warning(
                    "[%s] auto-retry skipped: schedule repair failed (mode=%s)",
                    upload_id,
                    mode,
                )
                continue

        payload = await build_payload(
            upload_id,
            user_id,
            deferred=deferred,
            job_id=f"auto-retry-{upload_id}-{prior + 1}",
        )
        if not payload or not await enqueue(payload):
            logger.error("[%s] auto-retry enqueue failed (attempt %s)", upload_id, prior + 1)
            continue

        new_arts = bump_auto_retry_metadata(
            arts if isinstance(arts, dict) else None,
            error_code=row.get("error_code"),
        )
        await conn.execute(
            """
            UPDATE uploads
            SET status = 'queued',
                error_code = NULL,
                error_detail = NULL,
                processing_started_at = NULL,
                output_artifacts = $2::jsonb,
                updated_at = NOW()
            WHERE id = $1 AND status = 'failed'
            """,
            upload_id,
            json.dumps(new_arts, default=str),
        )
        logger.warning(
            "[%s] auto-retry: failed → queued (attempt %s/%s, prior_error=%s, deferred=%s)",
            upload_id,
            prior + 1,
            max_retries,
            row.get("error_code"),
            deferred,
        )
        recovered += 1
    return recovered


async def recover_stuck_staged_past_window(
    conn: Any,
    *,
    build_payload: BuildPayloadFn,
    enqueue: EnqueueFn,
    stale_minutes: int = 90,
    limit: int = 10,
) -> int:
    """Re-dispatch staged uploads whose scheduled_time passed long ago."""
    rows = await conn.fetch(
        """
        SELECT id, user_id
        FROM uploads
        WHERE status = 'staged'
          AND scheduled_time IS NOT NULL
          AND scheduled_time < NOW() - ($1::int * INTERVAL '1 minute')
          AND updated_at < NOW() - INTERVAL '30 minutes'
        ORDER BY scheduled_time ASC
        LIMIT $2
        """,
        stale_minutes,
        limit,
    )
    recovered = 0
    for row in rows:
        upload_id = str(row["id"])
        user_id = str(row["user_id"])
        claimed = await conn.fetchval(
            """
            UPDATE uploads
            SET status = 'queued', updated_at = NOW()
            WHERE id = $1 AND status = 'staged'
            RETURNING id
            """,
            row["id"],
        )
        if not claimed:
            continue
        payload = await build_payload(
            upload_id,
            user_id,
            deferred=True,
            job_id=f"staged-recover-{upload_id}",
        )
        if payload and await enqueue(payload):
            logger.warning(
                "[%s] stuck staged recovery: re-enqueued deferred process (scheduled_time overdue)",
                upload_id,
            )
            recovered += 1
        else:
            await conn.execute(
                """
                UPDATE uploads
                SET status = 'staged', updated_at = NOW()
                WHERE id = $1 AND status = 'queued'
                """,
                row["id"],
            )
            logger.error("[%s] stuck staged recovery: enqueue failed — reverted to staged", upload_id)
    return recovered


async def recover_abandoned_pending(
    conn: Any,
    db_pool: Any,
    *,
    hours: Optional[int] = None,
    limit: int = 25,
) -> int:
    """
    Fail pending uploads that never completed registration (no /complete).

    Skips ENQUEUE_FAILED (handled by recover_enqueue_failed_pending).
    """
    age_h = hours if hours is not None else ABANDONED_PENDING_HOURS
    rows = await conn.fetch(
        """
        SELECT id, user_id, schedule_mode, error_code, created_at
        FROM uploads
        WHERE status = 'pending'
          AND COALESCE(error_code, '') NOT IN ('ENQUEUE_FAILED', 'QUEUE_UNAVAILABLE')
          AND created_at < NOW() - ($1::int * INTERVAL '1 hour')
        ORDER BY created_at ASC
        LIMIT $2
        """,
        age_h,
        limit,
    )
    failed = 0
    for row in rows:
        upload_id = str(row["id"])
        user_id = str(row["user_id"])
        mode = str(row.get("schedule_mode") or "immediate")
        detail = UPLOAD_ERROR_MESSAGES.get(
            ERROR_ABANDONED_PENDING,
            "Upload was left incomplete.",
        )
        await mark_schedule_incomplete_failed(
            conn,
            upload_id,
            detail=detail,
            error_code=ERROR_ABANDONED_PENDING,
        )
        await loud_upload_schedule_failure(
            upload_id,
            user_id,
            reason=detail,
            schedule_mode=mode,
            db_pool=db_pool,
        )
        logger.warning("[%s] stuck recovery: abandoned pending → failed %s", upload_id, ERROR_ABANDONED_PENDING)
        failed += 1
    return failed


async def recover_stuck_ready_to_publish(
    conn: Any,
    db_pool: Any,
    *,
    dispatch_publish: Optional[DeferredPublishFn] = None,
    redispatch_minutes: Optional[int] = None,
    fail_minutes: Optional[int] = None,
    limit: int = 20,
) -> dict[str, int]:
    """
    Unstick ``ready_to_publish`` rows:

    1. Null ``scheduled_time`` → repair or fail ``PUBLISH_SLOT_MISSING``.
    2. Past due + unhandled platforms with no resolvable next slot → fail.
    3. Past due + platforms due → re-dispatch deferred publish (or fail after fail_minutes).
    """
    redispatch_m = redispatch_minutes if redispatch_minutes is not None else STUCK_READY_REDISPATCH_MINUTES
    fail_m = fail_minutes if fail_minutes is not None else STUCK_READY_FAIL_MINUTES
    now = _now_utc()

    rows = await conn.fetch(
        """
        SELECT id, user_id, schedule_mode, platforms, schedule_metadata, scheduled_time,
               platform_results, target_accounts, processed_assets, updated_at, created_at
        FROM uploads
        WHERE status = 'ready_to_publish'
        ORDER BY COALESCE(scheduled_time, updated_at) ASC
        LIMIT $1
        """,
        limit,
    )

    stats = {"repaired": 0, "redispatched": 0, "failed": 0, "skipped": 0}
    for row in rows:
        upload_id = str(row["id"])
        user_id = str(row["user_id"])
        upload = dict(row)
        mode = str(upload.get("schedule_mode") or "scheduled").strip().lower()

        # --- Missing scheduled_time ---
        if upload.get("scheduled_time") is None:
            if mode in ("scheduled", "smart"):
                ok, _, _ = await repair_upload_schedule(conn, upload)
                if ok:
                    stats["repaired"] += 1
                    logger.warning("[%s] ready_to_publish: repaired missing scheduled_time", upload_id)
                    continue
                detail = UPLOAD_ERROR_MESSAGES.get(
                    ERROR_PUBLISH_SLOT_MISSING,
                    "No publish slot available.",
                )
                await mark_schedule_incomplete_failed(
                    conn,
                    upload_id,
                    detail=detail,
                    error_code=ERROR_PUBLISH_SLOT_MISSING,
                )
                await loud_upload_schedule_failure(
                    upload_id,
                    user_id,
                    reason=detail,
                    schedule_mode=mode,
                    db_pool=db_pool,
                )
                stats["failed"] += 1
                logger.warning(
                    "[%s] ready_to_publish → failed %s (null scheduled_time)",
                    upload_id,
                    ERROR_PUBLISH_SLOT_MISSING,
                )
                continue
            # Immediate (and similar) rows often have null scheduled_time by design.
            # Treat as due-now and fall through to idle/redispatch logic — never fail
            # solely because scheduled_time is null.
            if mode != "immediate":
                detail = UPLOAD_ERROR_MESSAGES.get(
                    ERROR_PUBLISH_SLOT_MISSING,
                    "No publish slot available.",
                )
                await mark_schedule_incomplete_failed(
                    conn,
                    upload_id,
                    detail=detail,
                    error_code=ERROR_PUBLISH_SLOT_MISSING,
                )
                await loud_upload_schedule_failure(
                    upload_id,
                    user_id,
                    reason=detail,
                    schedule_mode=mode,
                    db_pool=db_pool,
                )
                stats["failed"] += 1
                logger.warning(
                    "[%s] ready_to_publish → failed %s (null scheduled_time mode=%s)",
                    upload_id,
                    ERROR_PUBLISH_SLOT_MISSING,
                    mode,
                )
                continue

        st = upload.get("scheduled_time")
        if hasattr(st, "tzinfo") and st is not None and st.tzinfo is None:
            st = st.replace(tzinfo=timezone.utc)

        due = platforms_due_for_publish(upload, now)
        next_anchor = next_due_scheduled_time(upload, upload.get("platform_results"))
        has_pending = still_has_pending_publish_slots(upload, upload.get("platform_results"))

        # Smart with remaining targets but no resolvable slot → repair once, then fail
        if mode == "smart" and has_pending and next_anchor is None and not due:
            ok, repaired_st, repaired_meta = await repair_upload_schedule(conn, upload)
            if ok and repaired_st is not None:
                upload["scheduled_time"] = repaired_st
                if repaired_meta is not None:
                    upload["schedule_metadata"] = repaired_meta
                next_anchor = next_due_scheduled_time(upload, upload.get("platform_results"))
                due = platforms_due_for_publish(upload, now)
                has_pending = still_has_pending_publish_slots(
                    upload, upload.get("platform_results")
                )
                if next_anchor is not None or due:
                    stats["repaired"] += 1
                    st = repaired_st or next_anchor or upload.get("scheduled_time")
                    logger.warning(
                        "[%s] ready_to_publish: repaired missing slots → next=%s",
                        upload_id,
                        next_anchor,
                    )
                    # Fall through to overdue / redispatch logic with repaired row.
                else:
                    ok = False
            if not ok or (has_pending and next_anchor is None and not due):
                detail = UPLOAD_ERROR_MESSAGES.get(
                    ERROR_PUBLISH_SLOT_MISSING,
                    "No remaining publish slot.",
                )
                await mark_schedule_incomplete_failed(
                    conn,
                    upload_id,
                    detail=detail,
                    error_code=ERROR_PUBLISH_SLOT_MISSING,
                )
                await loud_upload_schedule_failure(
                    upload_id,
                    user_id,
                    reason=detail,
                    schedule_mode=mode,
                    db_pool=db_pool,
                )
                stats["failed"] += 1
                logger.warning(
                    "[%s] ready_to_publish → failed %s (no next slot)",
                    upload_id,
                    ERROR_PUBLISH_SLOT_MISSING,
                )
                continue

        if mode == "smart":
            overdue = bool(due)
            anchor_for_age = next_anchor or st
        elif mode == "immediate" and st is None:
            # Immediate publish: null scheduled_time means due now; age off updated_at.
            overdue = True
            anchor_for_age = upload.get("updated_at") or upload.get("created_at") or now
        else:
            overdue = st is not None and st <= now
            anchor_for_age = st

        if not overdue:
            if mode == "smart" and next_anchor is not None and st is not None and next_anchor != st:
                await conn.execute(
                    """
                    UPDATE uploads
                    SET scheduled_time = $2, updated_at = NOW()
                    WHERE id = $1 AND status = 'ready_to_publish'
                    """,
                    upload_id,
                    next_anchor,
                )
                stats["repaired"] += 1
            else:
                stats["skipped"] += 1
            continue

        age_minutes = 0.0
        if anchor_for_age is not None:
            if getattr(anchor_for_age, "tzinfo", None) is None:
                anchor_for_age = anchor_for_age.replace(tzinfo=timezone.utc)
            age_minutes = max(0.0, (now - anchor_for_age).total_seconds() / 60.0)

        updated_at = upload.get("updated_at")
        idle_minutes = 0.0
        if updated_at is not None:
            if getattr(updated_at, "tzinfo", None) is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            idle_minutes = max(0.0, (now - updated_at).total_seconds() / 60.0)

        # Prefer idle for redispatch cadence — fixed scheduled_time ages forever past
        # the window and would otherwise re-fire every recovery tick.
        if idle_minutes < float(redispatch_m) and age_minutes < float(fail_m):
            stats["skipped"] += 1
            continue

        if age_minutes >= float(fail_m) and idle_minutes >= min(30.0, float(fail_m) / 2):
            detail = UPLOAD_ERROR_MESSAGES.get(
                ERROR_STUCK_READY_TO_PUBLISH,
                "Publishing stalled.",
            )
            await mark_schedule_incomplete_failed(
                conn,
                upload_id,
                detail=detail,
                error_code=ERROR_STUCK_READY_TO_PUBLISH,
            )
            await loud_upload_schedule_failure(
                upload_id,
                user_id,
                reason=detail,
                schedule_mode=mode,
                db_pool=db_pool,
            )
            stats["failed"] += 1
            logger.warning(
                "[%s] ready_to_publish → failed %s (overdue %.0fm)",
                upload_id,
                ERROR_STUCK_READY_TO_PUBLISH,
                age_minutes,
            )
            continue

        # Never redispatch when nothing is due — deferred publish would no-op,
        # reset updated_at, and burn R2/CPU on a ~30m loop (Render OOM pattern).
        if not due:
            if idle_minutes >= float(fail_m):
                detail = UPLOAD_ERROR_MESSAGES.get(
                    ERROR_PUBLISH_SLOT_MISSING,
                    "No remaining publish slot.",
                )
                await mark_schedule_incomplete_failed(
                    conn,
                    upload_id,
                    detail=detail,
                    error_code=ERROR_PUBLISH_SLOT_MISSING,
                )
                await loud_upload_schedule_failure(
                    upload_id,
                    user_id,
                    reason=detail,
                    schedule_mode=mode,
                    db_pool=db_pool,
                )
                stats["failed"] += 1
                logger.warning(
                    "[%s] ready_to_publish → failed %s (overdue, nothing due)",
                    upload_id,
                    ERROR_PUBLISH_SLOT_MISSING,
                )
            else:
                stats["skipped"] += 1
            continue

        if dispatch_publish is None:
            stats["skipped"] += 1
            continue

        claimed = await conn.fetchval(
            """
            UPDATE uploads
            SET status = 'processing',
                processing_started_at = COALESCE(processing_started_at, NOW()),
                updated_at = NOW()
            WHERE id = $1 AND status = 'ready_to_publish'
            RETURNING id
            """,
            row["id"],
        )
        if not claimed:
            stats["skipped"] += 1
            continue

        try:
            await dispatch_publish(upload_id, user_id)
            stats["redispatched"] += 1
            logger.warning(
                "[%s] ready_to_publish recovery: re-dispatched deferred publish (overdue %.0fm)",
                upload_id,
                age_minutes,
            )
        except Exception as e:
            await conn.execute(
                """
                UPDATE uploads
                SET status = 'ready_to_publish', updated_at = NOW()
                WHERE id = $1 AND status = 'processing'
                """,
                row["id"],
            )
            logger.error(
                "[%s] ready_to_publish recovery: dispatch failed (%s) — reverted",
                upload_id,
                e,
            )
            stats["skipped"] += 1

    return stats


async def fail_deferred_batch_without_slot(
    conn: Any,
    upload_id: str,
    user_id: str,
    upload_snap: dict[str, Any],
    platform_results: Any,
    *,
    db_pool: Any = None,
) -> bool:
    """
    When a partial smart publish has remaining targets but no next slot,
    attempt schedule repair once, then mark failed if still unresolvable.

    Returns True if the row was failed.
    """
    if not still_has_pending_publish_slots(upload_snap, platform_results):
        return False
    if next_due_scheduled_time(upload_snap, platform_results) is not None:
        return False

    # Repair-before-fail: rebuild missing per-platform slots, then re-check.
    snap = dict(upload_snap)
    snap["id"] = upload_id
    snap["user_id"] = user_id
    ok, repaired_st, repaired_meta = await repair_upload_schedule(conn, snap)
    if ok:
        if repaired_st is not None:
            snap["scheduled_time"] = repaired_st
        if repaired_meta is not None:
            snap["schedule_metadata"] = repaired_meta
        nxt = next_due_scheduled_time(snap, platform_results)
        if nxt is not None:
            await conn.execute(
                """
                UPDATE uploads
                SET status = 'ready_to_publish',
                    scheduled_time = $2,
                    schedule_metadata = COALESCE($3::jsonb, schedule_metadata),
                    updated_at = NOW()
                WHERE id = $1
                  AND status NOT IN ('failed', 'cancelled', 'completed', 'succeeded')
                """,
                upload_id,
                nxt,
                json.dumps(repaired_meta, default=str) if isinstance(repaired_meta, dict) else None,
            )
            logger.warning(
                "[%s] partial smart publish: repaired slots → ready_to_publish next=%s",
                upload_id,
                nxt,
            )
            return False

    detail = UPLOAD_ERROR_MESSAGES.get(
        ERROR_PUBLISH_SLOT_MISSING,
        "No remaining publish slot after partial publish.",
    )
    await mark_schedule_incomplete_failed(
        conn,
        upload_id,
        detail=detail,
        error_code=ERROR_PUBLISH_SLOT_MISSING,
    )
    await loud_upload_schedule_failure(
        upload_id,
        user_id,
        reason=detail,
        schedule_mode=str(upload_snap.get("schedule_mode") or "smart"),
        db_pool=db_pool,
    )
    logger.warning(
        "[%s] partial smart publish → failed %s (no next slot)",
        upload_id,
        ERROR_PUBLISH_SLOT_MISSING,
    )
    return True
