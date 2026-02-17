"""
UploadM8 Worker Database Functions
===================================
Database helpers used by the worker pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import asyncpg

from .context import JobContext

logger = logging.getLogger("uploadm8-worker")


# ============================================================
# Load Functions
# ============================================================
async def load_upload_record(pool: asyncpg.Pool, upload_id: str) -> Optional[dict]:
    """Load an upload record by ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM uploads WHERE id = $1", upload_id)
        return dict(row) if row else None


async def load_user(pool: asyncpg.Pool, user_id: str) -> Optional[dict]:
    """Load a user record by ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        return dict(row) if row else None


async def load_user_settings(pool: asyncpg.Pool, user_id: str) -> dict:
    """Load user settings, returning defaults if none exist."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM user_settings WHERE user_id = $1", user_id)
        if row:
            return dict(row)

        # Try user_preferences table as fallback
        row = await conn.fetchrow("SELECT * FROM user_preferences WHERE user_id = $1", user_id)
        if row:
            return dict(row)

    return {}


async def load_user_entitlement_overrides(pool: asyncpg.Pool, user_id: str) -> Optional[dict]:
    """
    Load per-user entitlement overrides set by admin.
    Returns None if no overrides table exists or no overrides set.
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM entitlement_overrides WHERE user_id = $1", user_id
            )
            return dict(row) if row else None
    except asyncpg.exceptions.UndefinedTableError:
        # Table doesn't exist yet — no overrides
        return None
    except Exception as e:
        logger.debug(f"Entitlement overrides lookup failed (non-fatal): {e}")
        return None


# ============================================================
# Status Updates
# ============================================================
async def mark_processing_started(pool: asyncpg.Pool, ctx: JobContext):
    """Mark upload as processing."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE uploads
            SET status = 'processing',
                processing_started_at = $2,
                updated_at = NOW()
            WHERE id = $1
            """,
            ctx.upload_id,
            ctx.started_at or datetime.now(timezone.utc),
        )


async def mark_processing_completed(pool: asyncpg.Pool, ctx: JobContext):
    """Mark upload as completed (success or failed based on ctx.state)."""
    platform_results_json = None
    if ctx.platform_results:
        platform_results_json = json.dumps(
            [
                {
                    "platform": r.platform,
                    "success": r.success,
                    "platform_video_id": r.platform_video_id,
                    "platform_url": r.platform_url,
                    "publish_id": r.publish_id,
                    "error_code": r.error_code,
                    "error_message": r.error_message,
                    "verify_status": r.verify_status,
                    "http_status": r.http_status,
                    "views": r.views,
                    "likes": r.likes,
                }
                for r in ctx.platform_results
            ]
        )

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE uploads
            SET status = $2,
                processing_finished_at = $3,
                completed_at = CASE WHEN $2 = 'succeeded' THEN $3 ELSE completed_at END,
                error_code = $4,
                error_detail = $5,
                platform_results = $6::jsonb,
                compute_seconds = $7,
                updated_at = NOW()
            WHERE id = $1
            """,
            ctx.upload_id,
            ctx.state,
            ctx.finished_at or datetime.now(timezone.utc),
            ctx.error_code,
            ctx.error_message,
            platform_results_json,
            ctx.compute_seconds,
        )


async def mark_processing_failed(
    pool: asyncpg.Pool, ctx: JobContext, error_code: str, error_detail: str
):
    """Mark upload as failed with error info."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE uploads
            SET status = 'failed',
                error_code = $2,
                error_detail = $3,
                processing_finished_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
            """,
            ctx.upload_id,
            error_code,
            error_detail,
        )


async def mark_cancelled(pool: asyncpg.Pool, upload_id: str):
    """Mark upload as cancelled."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE uploads
            SET status = 'cancelled',
                processing_finished_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
            """,
            upload_id,
        )


async def check_cancel_requested(pool: asyncpg.Pool, upload_id: str) -> bool:
    """Check if cancellation was requested for this upload."""
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            "SELECT cancel_requested FROM uploads WHERE id = $1", upload_id
        )
        return bool(val)


# ============================================================
# Metadata & Counts
# ============================================================
async def save_generated_metadata(pool: asyncpg.Pool, ctx: JobContext):
    """Save AI-generated metadata (title, caption, hashtags) back to the upload."""
    updates = []
    params = [ctx.upload_id]
    idx = 1

    if ctx.ai_title:
        idx += 1
        updates.append(f"ai_generated_title = ${idx}")
        params.append(ctx.ai_title)

    if ctx.ai_caption:
        idx += 1
        updates.append(f"ai_generated_caption = ${idx}")
        params.append(ctx.ai_caption)

    if ctx.ai_hashtags:
        idx += 1
        updates.append(f"ai_generated_hashtags = ${idx}")
        params.append(ctx.ai_hashtags)

    if not updates:
        return

    async with pool.acquire() as conn:
        try:
            await conn.execute(
                f"UPDATE uploads SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1",
                *params,
            )
        except asyncpg.exceptions.UndefinedColumnError as e:
            logger.warning(f"save_generated_metadata skipped (column missing): {e}")


async def increment_upload_count(pool: asyncpg.Pool, user_id: str):
    """Increment the user's completed upload count (last_active_at update)."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET last_active_at = NOW(), updated_at = NOW() WHERE id = $1",
            user_id,
        )
