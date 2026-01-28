"""
UploadM8 Database Stage
=======================
Database operations for the worker pipeline.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import asyncpg

from .context import JobContext

logger = logging.getLogger("uploadm8-worker")


async def load_upload_record(pool: asyncpg.Pool, upload_id: str) -> Optional[dict]:
    """Load upload record from database."""
    if not pool:
        return None
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id, user_id, r2_key, telemetry_r2_key, processed_r2_key,
                   filename, file_size, platforms, title, caption, privacy,
                   status, scheduled_time, schedule_mode, cancel_requested,
                   created_at, updated_at
            FROM uploads WHERE id = $1
        """, upload_id)
        
        if not row:
            return None
        
        return dict(row)


async def load_user(pool: asyncpg.Pool, user_id: str) -> Optional[dict]:
    """Load user record from database."""
    if not pool:
        return None
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id, email, name, role, subscription_tier, upload_quota,
                   uploads_this_month, unlimited_uploads, subscription_status,
                   current_period_end, created_at
            FROM users WHERE id::text = $1
        """, user_id)
        
        if not row:
            return None
        
        return dict(row)


async def load_user_settings(pool: asyncpg.Pool, user_id: str) -> dict:
    """Load user settings from database."""
    if not pool:
        return {}
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT discord_webhook, telemetry_enabled, hud_enabled, hud_position,
                   speeding_mph, euphoria_mph, hud_speed_unit, hud_color,
                   hud_font_family, hud_font_size, selected_page_id,
                   ffmpeg_screenshot_interval, auto_generate_thumbnails,
                   auto_generate_captions, auto_generate_hashtags,
                   default_hashtag_count, always_use_hashtags
            FROM user_settings WHERE user_id::text = $1
        """, user_id)
        
        if not row:
            return {
                "telemetry_enabled": True,
                "hud_enabled": True,
                "hud_position": "bottom-left",
                "speeding_mph": 80,
                "euphoria_mph": 100,
                "hud_speed_unit": "mph",
                "hud_color": "#FFFFFF",
                "hud_font_family": "Arial",
                "hud_font_size": 24,
                "ffmpeg_screenshot_interval": 5,
                "auto_generate_thumbnails": True,
                "auto_generate_captions": True,
                "auto_generate_hashtags": True,
                "default_hashtag_count": 5,
                "always_use_hashtags": False,
            }
        
        return dict(row)


async def load_user_entitlement_overrides(pool: asyncpg.Pool, user_id: str) -> dict:
    """Load any custom entitlement overrides for user."""
    if not pool:
        return {}
    
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT entitlement_key, entitlement_value, value_type
            FROM user_entitlements WHERE user_id::text = $1
        """, user_id)
        
        overrides = {}
        for row in rows:
            key = row["entitlement_key"]
            value = row["entitlement_value"]
            vtype = row.get("value_type", "string")
            
            if vtype == "bool":
                overrides[key] = value.lower() in ("true", "1", "yes")
            elif vtype == "int":
                try:
                    overrides[key] = int(value)
                except:
                    pass
            else:
                overrides[key] = value
        
        return overrides


async def load_platform_token(pool: asyncpg.Pool, user_id: str, platform: str) -> Optional[str]:
    """Load encrypted platform token from database."""
    if not pool:
        return None
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT token_blob FROM platform_tokens
            WHERE user_id::text = $1 AND platform = $2
        """, user_id, platform)
        
        if not row:
            return None
        
        return row["token_blob"]


async def load_all_platform_tokens(pool: asyncpg.Pool, user_id: str) -> Dict[str, Any]:
    """Load all platform tokens for user."""
    if not pool:
        return {}
    
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT platform, token_blob, account_name, account_username, account_avatar
            FROM platform_tokens WHERE user_id::text = $1
        """, user_id)
        
        return {
            row["platform"]: {
                "token_blob": row["token_blob"],
                "account_name": row.get("account_name"),
                "account_username": row.get("account_username"),
                "account_avatar": row.get("account_avatar"),
            }
            for row in rows
        }


async def mark_processing_started(pool: asyncpg.Pool, ctx: JobContext):
    """Mark upload as processing started."""
    if not pool:
        return
    
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE uploads SET 
                status = 'processing',
                processing_started_at = $2,
                updated_at = NOW()
            WHERE id = $1
        """, ctx.upload_id, ctx.started_at or datetime.now(timezone.utc))


async def mark_processing_completed(pool: asyncpg.Pool, ctx: JobContext):
    """Mark upload as completed with results."""
    if not pool:
        return
    
    status = "completed" if ctx.is_success() else "partial" if ctx.is_partial_success() else "failed"
    
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE uploads SET 
                status = $2,
                processed_r2_key = $3,
                processing_finished_at = $4,
                completed_at = CASE WHEN $2 = 'completed' THEN NOW() ELSE NULL END,
                error_code = $5,
                error_detail = $6,
                platform_results = $7,
                updated_at = NOW()
            WHERE id = $1
        """, 
            ctx.upload_id,
            status,
            ctx.processed_r2_key,
            ctx.finished_at or datetime.now(timezone.utc),
            ctx.error_code,
            ctx.error_message,
            json.dumps([r.__dict__ for r in ctx.platform_results])
        )


async def mark_processing_failed(pool: asyncpg.Pool, ctx: JobContext, error_code: str, error_message: str):
    """Mark upload as failed."""
    if not pool:
        return
    
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE uploads SET 
                status = 'failed',
                processing_finished_at = NOW(),
                error_code = $2,
                error_detail = $3,
                updated_at = NOW()
            WHERE id = $1
        """, ctx.upload_id, error_code, error_message)


async def mark_cancelled(pool: asyncpg.Pool, upload_id: str):
    """Mark upload as cancelled."""
    if not pool:
        return
    
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE uploads SET 
                status = 'cancelled',
                processing_finished_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
        """, upload_id)


async def check_cancel_requested(pool: asyncpg.Pool, upload_id: str) -> bool:
    """Check if cancel has been requested for this upload."""
    if not pool:
        return False
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT cancel_requested FROM uploads WHERE id = $1",
            upload_id
        )
        return row["cancel_requested"] if row else False


async def increment_upload_count(pool: asyncpg.Pool, user_id: str):
    """Increment user's monthly upload count."""
    if not pool:
        return
    
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE users SET 
                uploads_this_month = uploads_this_month + 1,
                last_active_at = NOW(),
                updated_at = NOW()
            WHERE id::text = $1
        """, user_id)


async def save_job_state(pool: asyncpg.Pool, ctx: JobContext):
    """Save job state for resumability."""
    if not pool:
        return
    
    state_json = json.dumps({
        "job_id": ctx.job_id,
        "state": ctx.state,
        "stage": ctx.stage,
        "attempt_count": ctx.attempt_count,
        "output_artifacts": ctx.output_artifacts,
        "errors": ctx.errors,
    })
    
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO job_state (upload_id, state_json, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (upload_id) DO UPDATE SET
                state_json = $2, updated_at = NOW()
        """, ctx.upload_id, state_json)


async def load_job_state(pool: asyncpg.Pool, upload_id: str) -> Optional[dict]:
    """Load saved job state for resumption."""
    if not pool:
        return None
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT state_json FROM job_state WHERE upload_id = $1",
            upload_id
        )
        
        if row and row["state_json"]:
            return json.loads(row["state_json"])
        return None


async def track_openai_cost(pool: asyncpg.Pool, user_id: str, operation: str, tokens: int, cost_usd: float):
    """Track OpenAI API usage and cost."""
    if not pool:
        return
    
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO cost_tracking (user_id, category, operation, tokens, cost_usd, created_at)
            VALUES ($1::uuid, 'openai', $2, $3, $4, NOW())
        """, user_id, operation, tokens, cost_usd)


async def track_storage_usage(pool: asyncpg.Pool, user_id: str, bytes_used: int, operation: str):
    """Track R2 storage usage."""
    if not pool:
        return
    
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO storage_tracking (user_id, bytes_used, operation, created_at)
            VALUES ($1::uuid, $2, $3, NOW())
        """, user_id, bytes_used, operation)
