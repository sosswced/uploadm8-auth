"""
UploadM8 Worker Database Functions
===================================
Database helpers used by the worker pipeline stages.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import asyncpg

from .context import JobContext

logger = logging.getLogger("uploadm8-worker")


# ============================================================
# Load Functions (used by worker.py run_pipeline)
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
    """Load user settings/preferences, returning defaults if none exist."""
    async with pool.acquire() as conn:
        # Try user_settings first
        try:
            row = await conn.fetchrow("SELECT * FROM user_settings WHERE user_id = $1", user_id)
            if row:
                return dict(row)
        except asyncpg.exceptions.UndefinedTableError:
            pass

        # Try user_preferences as fallback
        try:
            row = await conn.fetchrow("SELECT * FROM user_preferences WHERE user_id = $1", user_id)
            if row:
                return dict(row)
        except asyncpg.exceptions.UndefinedTableError:
            pass

    return {}


async def load_user_entitlement_overrides(pool: asyncpg.Pool, user_id: str) -> Optional[dict]:
    """Load per-user entitlement overrides set by admin."""
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM entitlement_overrides WHERE user_id = $1", user_id
            )
            return dict(row) if row else None
    except asyncpg.exceptions.UndefinedTableError:
        return None
    except Exception as e:
        logger.debug(f"Entitlement overrides lookup failed (non-fatal): {e}")
        return None


# ============================================================
# Status Updates (used by worker.py)
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
    """Mark upload as completed."""
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
    try:
        async with pool.acquire() as conn:
            val = await conn.fetchval(
                "SELECT cancel_requested FROM uploads WHERE id = $1", upload_id
            )
            return bool(val)
    except Exception:
        return False


# ============================================================
# Metadata & Counts (used by worker.py)
# ============================================================

async def save_generated_metadata(pool: asyncpg.Pool, ctx: JobContext):
    """Save AI-generated metadata back to the upload."""
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
    """Increment user's completed upload count."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET last_active_at = NOW(), updated_at = NOW() WHERE id = $1",
            user_id,
        )


# ============================================================
# Platform Tokens (used by publish_stage)
# ============================================================

async def load_platform_token(pool: asyncpg.Pool, user_id: str, platform: str) -> Optional[dict]:
    """Load a stored platform OAuth token for a user."""
    try:
        async with pool.acquire() as conn:
            # Try platform_tokens table
            try:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM platform_tokens
                    WHERE user_id = $1 AND platform = $2
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    user_id,
                    platform,
                )
                if row:
                    row_dict = dict(row)
                    # Token data may be in 'token_data' JSONB column or 'encrypted_token'
                    token_data = row_dict.get("token_data")
                    if token_data:
                        if isinstance(token_data, str):
                            return json.loads(token_data)
                        return dict(token_data) if hasattr(token_data, "keys") else token_data
                    # May be encrypted blob
                    encrypted = row_dict.get("encrypted_token")
                    if encrypted:
                        if isinstance(encrypted, str):
                            return json.loads(encrypted)
                        return dict(encrypted) if hasattr(encrypted, "keys") else encrypted
                    return row_dict
            except asyncpg.exceptions.UndefinedTableError:
                pass

            # Fallback: connected_accounts table
            try:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM connected_accounts
                    WHERE user_id = $1 AND platform = $2
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    user_id,
                    platform,
                )
                if row:
                    row_dict = dict(row)
                    token_data = row_dict.get("token_data") or row_dict.get("encrypted_token")
                    if token_data:
                        if isinstance(token_data, str):
                            return json.loads(token_data)
                        return dict(token_data) if hasattr(token_data, "keys") else token_data
                    return row_dict
            except asyncpg.exceptions.UndefinedTableError:
                pass

        return None
    except Exception as e:
        logger.error(f"Failed to load platform token for {user_id}/{platform}: {e}")
        return None


# ============================================================
# Publish Attempts / Ledger (used by publish_stage + verify_stage)
# ============================================================

async def insert_publish_attempt(
    pool: asyncpg.Pool,
    upload_id: str,
    user_id: str,
    platform: str,
) -> Optional[str]:
    """Insert a publish attempt row and return its ID."""
    attempt_id = str(uuid.uuid4())
    try:
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO publish_attempts (id, upload_id, user_id, platform, status, created_at)
                    VALUES ($1, $2, $3, $4, 'pending', NOW())
                    """,
                    attempt_id,
                    upload_id,
                    user_id,
                    platform,
                )
                return attempt_id
            except asyncpg.exceptions.UndefinedTableError:
                logger.debug("publish_attempts table not found, skipping ledger")
                return None
    except Exception as e:
        logger.warning(f"insert_publish_attempt failed: {e}")
        return None


async def update_publish_attempt_success(
    pool: asyncpg.Pool,
    attempt_id: str,
    platform_post_id: Optional[str] = None,
    platform_url: Optional[str] = None,
    http_status: Optional[int] = None,
    response_payload: Optional[dict] = None,
    publish_id: Optional[str] = None,
):
    """Mark a publish attempt as accepted (Step A success)."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE publish_attempts
                SET status = 'accepted',
                    platform_post_id = $2,
                    platform_url = $3,
                    http_status = $4,
                    response_payload = $5::jsonb,
                    publish_id = $6,
                    updated_at = NOW()
                WHERE id = $1
                """,
                attempt_id,
                platform_post_id,
                platform_url,
                http_status,
                json.dumps(response_payload) if response_payload else None,
                publish_id,
            )
    except Exception as e:
        logger.warning(f"update_publish_attempt_success failed: {e}")


async def update_publish_attempt_failed(
    pool: asyncpg.Pool,
    attempt_id: str,
    error_code: str = "UNKNOWN",
    error_message: str = "",
    http_status: Optional[int] = None,
    response_payload: Optional[dict] = None,
):
    """Mark a publish attempt as failed."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE publish_attempts
                SET status = 'failed',
                    error_code = $2,
                    error_message = $3,
                    http_status = $4,
                    response_payload = $5::jsonb,
                    updated_at = NOW()
                WHERE id = $1
                """,
                attempt_id,
                error_code,
                error_message,
                http_status,
                json.dumps(response_payload) if response_payload else None,
            )
    except Exception as e:
        logger.warning(f"update_publish_attempt_failed failed: {e}")


async def update_publish_attempt_verified(
    pool: asyncpg.Pool,
    attempt_id: str,
    verify_status: str,
    platform_url: Optional[str] = None,
):
    """Update verification status of a publish attempt (Step B)."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE publish_attempts
                SET verify_status = $2,
                    platform_url = COALESCE($3, platform_url),
                    verified_at = NOW(),
                    updated_at = NOW()
                WHERE id = $1
                """,
                attempt_id,
                verify_status,
                platform_url,
            )
    except Exception as e:
        logger.warning(f"update_publish_attempt_verified failed: {e}")


async def load_pending_verifications(pool: asyncpg.Pool, limit: int = 50) -> List[dict]:
    """Load publish attempts that need verification polling."""
    try:
        async with pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    """
                    SELECT pa.*, u.id AS uid
                    FROM publish_attempts pa
                    JOIN uploads u ON u.id = pa.upload_id
                    WHERE pa.status = 'accepted'
                      AND (pa.verify_status IS NULL OR pa.verify_status = 'pending')
                      AND pa.created_at > NOW() - INTERVAL '24 hours'
                    ORDER BY pa.created_at ASC
                    LIMIT $1
                    """,
                    limit,
                )
                return [dict(r) for r in rows]
            except asyncpg.exceptions.UndefinedTableError:
                return []
    except Exception as e:
        logger.debug(f"load_pending_verifications: {e}")
        return []


# ============================================================
# Admin Notifications (used by notify_stage)
# ============================================================

async def load_admin_notification_webhook(pool: asyncpg.Pool) -> Optional[str]:
    """Load admin Discord webhook URL from settings."""
    try:
        async with pool.acquire() as conn:
            try:
                val = await conn.fetchval(
                    "SELECT value FROM admin_settings WHERE key = 'discord_webhook_url'"
                )
                return str(val) if val else None
            except asyncpg.exceptions.UndefinedTableError:
                return None
    except Exception:
        return None


# ============================================================
# Per-Platform Asset Persistence
# ============================================================

async def save_processed_assets(pool: asyncpg.Pool, upload_id: str, assets: Dict[str, str]):
    """Save per-platform R2 keys to the uploads table.

    assets = {"tiktok": "processed/.../tiktok.mp4", "youtube": "processed/.../youtube.mp4", ...}

    Tries processed_assets JSONB column first.
    Falls back to storing in output_artifacts if column doesn't exist.
    """
    assets_json = json.dumps(assets)

    async with pool.acquire() as conn:
        # Try the dedicated column first
        try:
            await conn.execute(
                """
                UPDATE uploads
                SET processed_assets = $2::jsonb,
                    updated_at = NOW()
                WHERE id = $1
                """,
                upload_id,
                assets_json,
            )
            logger.info(f"Saved processed_assets for upload {upload_id}: {list(assets.keys())}")
            return
        except asyncpg.exceptions.UndefinedColumnError:
            pass  # Column doesn't exist yet, fall back

        # Fallback: store in existing JSONB column if available
        try:
            await conn.execute(
                """
                UPDATE uploads
                SET output_artifacts = COALESCE(output_artifacts, '{}'::jsonb) || jsonb_build_object('processed_assets', $2::jsonb),
                    updated_at = NOW()
                WHERE id = $1
                """,
                upload_id,
                assets_json,
            )
            logger.info(f"Saved processed_assets (fallback) for upload {upload_id}")
        except Exception as e:
            logger.warning(f"Could not save processed_assets: {e}")


async def load_processed_assets(pool: asyncpg.Pool, upload_id: str) -> Dict[str, str]:
    """Load per-platform R2 keys for an upload.

    Returns dict like {"tiktok": "processed/.../tiktok.mp4", ...}
    """
    async with pool.acquire() as conn:
        # Try dedicated column
        try:
            val = await conn.fetchval(
                "SELECT processed_assets FROM uploads WHERE id = $1",
                upload_id,
            )
            if val:
                return json.loads(val) if isinstance(val, str) else dict(val)
        except asyncpg.exceptions.UndefinedColumnError:
            pass

        # Fallback: check output_artifacts
        try:
            val = await conn.fetchval(
                "SELECT output_artifacts->'processed_assets' FROM uploads WHERE id = $1",
                upload_id,
            )
            if val:
                return json.loads(val) if isinstance(val, str) else dict(val)
        except Exception:
            pass

    return {}
