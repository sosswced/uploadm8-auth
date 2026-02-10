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
                   filename, file_size, platforms, title, caption, hashtags, privacy,
                   generated_title, generated_caption, generated_hashtags, platform_hashtags,
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
    """
    Load user settings and preferences from database.
    
    This combines data from both user_settings and user_preferences tables
    to provide a complete settings dictionary for the worker.
    """
    if not pool:
        return {}
    
    async with pool.acquire() as conn:
        # Load from user_settings (legacy HUD/telemetry settings)
        settings_row = await conn.fetchrow("""
            SELECT discord_webhook, telemetry_enabled, hud_enabled, hud_position,
                   speeding_mph, euphoria_mph, hud_speed_unit, hud_color,
                   hud_font_family, hud_font_size, selected_page_id
            FROM user_settings WHERE user_id::text = $1
        """, user_id)
        
        # Load from user_preferences (caption/hashtag/AI settings)
        prefs_row = await conn.fetchrow("""
            SELECT auto_captions, auto_thumbnails, thumbnail_interval,
                   ai_hashtags_enabled, ai_hashtag_count, ai_hashtag_style,
                   hashtag_position, max_hashtags,
                   always_hashtags, blocked_hashtags, platform_hashtags,
                   trill_enabled, trill_min_score, trill_hud_enabled,
                   trill_ai_enhance, trill_openai_model
            FROM user_preferences WHERE user_id::text = $1
        """, user_id)
        
        # Build combined settings dict
        settings = {}
        
        # Add user_settings data (HUD/telemetry)
        if settings_row:
            settings.update({
                "discord_webhook": settings_row.get("discord_webhook"),
                "telemetry_enabled": settings_row.get("telemetry_enabled", True),
                "hud_enabled": settings_row.get("hud_enabled", True),
                "hud_position": settings_row.get("hud_position", "bottom-left"),
                "speeding_mph": settings_row.get("speeding_mph", 80),
                "euphoria_mph": settings_row.get("euphoria_mph", 100),
                "hud_speed_unit": settings_row.get("hud_speed_unit", "mph"),
                "hud_color": settings_row.get("hud_color", "#FFFFFF"),
                "hud_font_family": settings_row.get("hud_font_family", "Arial"),
                "hud_font_size": settings_row.get("hud_font_size", 24),
                "selected_page_id": settings_row.get("selected_page_id"),
            })
        else:
            # Defaults for user_settings
            settings.update({
                "telemetry_enabled": True,
                "hud_enabled": True,
                "hud_position": "bottom-left",
                "speeding_mph": 80,
                "euphoria_mph": 100,
                "hud_speed_unit": "mph",
                "hud_color": "#FFFFFF",
                "hud_font_family": "Arial",
                "hud_font_size": 24,
            })
        
        # Add user_preferences data (AI/captions/hashtags)
        if prefs_row:
            # Parse JSONB fields
            always_hashtags = prefs_row.get("always_hashtags", [])
            blocked_hashtags = prefs_row.get("blocked_hashtags", [])
            platform_hashtags = prefs_row.get("platform_hashtags", {})
            
            # Handle JSONB parsing if needed (some drivers return as strings)
            if isinstance(always_hashtags, str):
                try:
                    always_hashtags = json.loads(always_hashtags)
                except:
                    always_hashtags = []
            
            if isinstance(blocked_hashtags, str):
                try:
                    blocked_hashtags = json.loads(blocked_hashtags)
                except:
                    blocked_hashtags = []
            
            if isinstance(platform_hashtags, str):
                try:
                    platform_hashtags = json.loads(platform_hashtags)
                except:
                    platform_hashtags = {}
            
            settings.update({
                "auto_captions": prefs_row.get("auto_captions", False),
                "auto_thumbnails": prefs_row.get("auto_thumbnails", False),
                "thumbnail_interval": prefs_row.get("thumbnail_interval", 5),
                "ai_hashtags_enabled": prefs_row.get("ai_hashtags_enabled", False),
                "ai_hashtag_count": prefs_row.get("ai_hashtag_count", 5),
                "ai_hashtag_style": prefs_row.get("ai_hashtag_style", "mixed"),
                "hashtag_position": prefs_row.get("hashtag_position", "end"),
                "max_hashtags": prefs_row.get("max_hashtags", 15),
                "always_hashtags": always_hashtags or [],
                "blocked_hashtags": blocked_hashtags or [],
                "platform_hashtags": platform_hashtags or {},
                "trill_enabled": prefs_row.get("trill_enabled", False),
                "trill_min_score": prefs_row.get("trill_min_score", 60),
                "trill_hud_enabled": prefs_row.get("trill_hud_enabled", False),
                "trill_ai_enhance": prefs_row.get("trill_ai_enhance", False),
                "trill_openai_model": prefs_row.get("trill_openai_model", "gpt-4o-mini"),
            })
        else:
            # Defaults for user_preferences
            settings.update({
                "auto_captions": False,
                "auto_thumbnails": False,
                "thumbnail_interval": 5,
                "ai_hashtags_enabled": False,
                "ai_hashtag_count": 5,
                "ai_hashtag_style": "mixed",
                "hashtag_position": "end",
                "max_hashtags": 15,
                "always_hashtags": [],
                "blocked_hashtags": [],
                "platform_hashtags": {},
                "trill_enabled": False,
                "trill_min_score": 60,
                "trill_hud_enabled": False,
                "trill_ai_enhance": False,
                "trill_openai_model": "gpt-4o-mini",
            })
        
        # Add legacy aliases for backward compatibility
        settings["auto_generate_captions"] = settings["auto_captions"]
        settings["auto_generate_hashtags"] = settings["ai_hashtags_enabled"]
        settings["auto_generate_thumbnails"] = settings["auto_thumbnails"]
        settings["ffmpeg_screenshot_interval"] = settings["thumbnail_interval"]
        settings["default_hashtag_count"] = settings["ai_hashtag_count"]
        
        return settings


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


async def save_generated_metadata(pool: asyncpg.Pool, ctx: JobContext):
    """Persist AI-generated metadata and per-platform hashtags.

    Writes:
    - generated_title/caption/hashtags
    - platform_hashtags jsonb map

    Also backfills title/caption/hashtags only if the user did not provide overrides.
    """
    if not pool:
        return

    gen_title = (ctx.ai_title or "").strip() if getattr(ctx, "ai_title", None) else None
    gen_caption = (ctx.ai_caption or "").strip() if getattr(ctx, "ai_caption", None) else None
    gen_hashtags = getattr(ctx, "ai_hashtags", None) or None

    platform_map = getattr(ctx, "platform_hashtags_map", None) or {}
    final_hashtags = getattr(ctx, "final_hashtags", None) or (gen_hashtags or getattr(ctx, "hashtags", None) or [])

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE uploads
            SET
                generated_title = COALESCE($2, generated_title),
                generated_caption = COALESCE($3, generated_caption),
                generated_hashtags = COALESCE($4::text[], generated_hashtags),
                platform_hashtags = CASE
                    WHEN $5::jsonb IS NOT NULL THEN $5::jsonb
                    ELSE platform_hashtags
                END,
                metadata_generated_at = NOW(),

                -- Only fill live fields if user did not provide them
                title = CASE
                    WHEN (title IS NULL OR btrim(title) = '') AND $2 IS NOT NULL THEN $2
                    ELSE title
                END,
                caption = CASE
                    WHEN (caption IS NULL OR btrim(caption) = '') AND $3 IS NOT NULL THEN $3
                    ELSE caption
                END,
                hashtags = CASE
                    WHEN (hashtags IS NULL OR array_length(hashtags,1) IS NULL OR array_length(hashtags,1) = 0)
                         AND $6::text[] IS NOT NULL
                    THEN $6::text[]
                    ELSE hashtags
                END,

                updated_at = NOW()
            WHERE id = $1
            """,
            ctx.upload_id,
            gen_title,
            gen_caption,
            gen_hashtags,
            __import__('json').dumps(platform_map) if platform_map is not None else None,
            final_hashtags if final_hashtags else None,
        )
