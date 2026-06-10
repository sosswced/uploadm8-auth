"""Worker-side recovery for uploads stuck in pending/staged/failed states."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Awaitable, Optional

from services.retry_policy import (
    MAX_AUTO_RETRIES_DEFAULT,
    auto_retry_backoff_minutes,
    bump_auto_retry_metadata,
    get_auto_retry_count,
    should_auto_retry_upload,
)
from services.upload.schedule_guard import (
    ERROR_SCHEDULE_INCOMPLETE,
    UPLOAD_ERROR_MESSAGES,
    loud_upload_schedule_failure,
    mark_schedule_incomplete_failed,
    repair_upload_schedule,
)

logger = logging.getLogger("uploadm8-worker")

EnqueueFn = Callable[[dict], Awaitable[bool]]
BuildPayloadFn = Callable[..., Awaitable[Optional[dict]]]


async def recover_staged_without_schedule(
    conn: Any,
    db_pool: Any,
    *,
    limit: int = 20,
) -> int:
    """Repair or fail staged uploads missing scheduled_time (would never dispatch)."""
    rows = await conn.fetch(
        """
        SELECT id, user_id, schedule_mode, platforms, schedule_metadata, scheduled_time
        FROM uploads
        WHERE status = 'staged'
          AND scheduled_time IS NULL
        ORDER BY updated_at ASC
        LIMIT $1
        """,
        limit,
    )
    handled = 0
    for row in rows:
        upload_id = str(row["id"])
        user_id = str(row["user_id"])
        ok, _, _ = await repair_upload_schedule(conn, dict(row))
        if ok:
            logger.warning("[%s] stuck recovery: repaired missing schedule on staged upload", upload_id)
            handled += 1
            continue
        detail = UPLOAD_ERROR_MESSAGES.get(
            ERROR_SCHEDULE_INCOMPLETE,
            "Schedule could not be generated for this upload.",
        )
        await mark_schedule_incomplete_failed(conn, upload_id, detail=detail)
        await loud_upload_schedule_failure(
            upload_id,
            user_id,
            reason=detail,
            schedule_mode=str(row.get("schedule_mode") or ""),
            db_pool=db_pool,
        )
        logger.error("[%s] stuck recovery: marked failed %s (no schedule time)", upload_id, ERROR_SCHEDULE_INCOMPLETE)
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
    """Re-queue failed immediate uploads with transient errors (exponential backoff)."""
    rows = await conn.fetch(
        """
        SELECT id, user_id, error_code, output_artifacts, schedule_mode, status, updated_at
        FROM uploads
        WHERE status = 'failed'
          AND COALESCE(schedule_mode, 'immediate') = 'immediate'
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
        if not should_auto_retry_upload(dict(row), max_retries=max_retries):
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
        payload = await build_payload(
            upload_id,
            user_id,
            deferred=False,
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
            "[%s] auto-retry: failed → queued (attempt %s/%s, prior_error=%s)",
            upload_id,
            prior + 1,
            max_retries,
            row.get("error_code"),
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
