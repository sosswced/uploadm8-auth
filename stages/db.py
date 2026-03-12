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
    """
    Load user settings/preferences, returning defaults if none exist.

    CRITICAL: The frontend saves ALL hashtag preferences — alwaysHashtags,
    blockedHashtags, platformHashtags, aiHashtagCount — to the users.preferences
    JSONB column via /api/me/preferences.  We MUST read that column here or
    zero hashtag enforcement will ever apply.

    Query order:
      1. user_settings table   (worker-side settings, snake_case keys)
      2. user_preferences table (fallback, legacy)
      3. users.preferences JSONB (frontend-saved settings, camelCase keys)
         → merged on top so frontend values always win for hashtag fields
    """
    result: dict = {}

    async with pool.acquire() as conn:
        # ── 1. user_settings table ────────────────────────────────────────
        try:
            row = await conn.fetchrow(
                "SELECT * FROM user_settings WHERE user_id = $1", user_id
            )
            if row:
                result = dict(row)
        except asyncpg.exceptions.UndefinedTableError:
            pass

        # ── 2. user_preferences fallback ─────────────────────────────────
        if not result:
            try:
                row = await conn.fetchrow(
                    "SELECT * FROM user_preferences WHERE user_id = $1", user_id
                )
                if row:
                    result = dict(row)
            except asyncpg.exceptions.UndefinedTableError:
                pass

        # ── 3. users.preferences JSONB — the source of truth for hashtag prefs ──
        # The settings page saves everything here via /api/me/preferences.
        # We merge these on top so frontend values always override stale rows.
        try:
            prefs_val = await conn.fetchval(
                "SELECT preferences FROM users WHERE id = $1", user_id
            )
            if prefs_val:
                if isinstance(prefs_val, str):
                    prefs = json.loads(prefs_val)
                elif hasattr(prefs_val, "keys"):
                    prefs = dict(prefs_val)
                else:
                    prefs = {}

                if isinstance(prefs, dict) and prefs:
                    # Normalise: store both camelCase and snake_case versions of
                    # every hashtag/caption field so the pipeline can use either.
                    FIELD_MAP = {
                        "alwaysHashtags":   "always_hashtags",
                        "blockedHashtags":  "blocked_hashtags",
                        "platformHashtags": "platform_hashtags",
                        "aiHashtagCount":   "ai_hashtag_count",
                        "aiHashtagsEnabled":"ai_hashtags_enabled",
                        "aiHashtagStyle":   "ai_hashtag_style",
                        "autoCaptions":     "auto_captions",
                        "captionStyle":     "caption_style",
                        "captionTone":      "caption_tone",
                        "captionVoice":     "caption_voice",
                        "captionFrameCount":"caption_frame_count",
                        "maxHashtags":      "max_hashtags",
                        "trillOpenaiModel": "openai_model",
                    }
                    for camel, snake in FIELD_MAP.items():
                        val = prefs.get(camel)
                        if val is None:
                            val = prefs.get(snake)
                        if val is not None:
                            # Frontend values override worker-side settings rows
                            result[camel] = val
                            result[snake] = val

                    # Merge any remaining keys that weren't in our map
                    for k, v in prefs.items():
                        if k not in result:
                            result[k] = v

        except Exception as e:
            logger.debug(f"Could not read users.preferences for {user_id} (non-fatal): {e}")

    return result


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
                completed_at = CASE WHEN $2 IN ('succeeded','partial') THEN $3 ELSE completed_at END,
                error_code = $4,
                error_detail = $5,
                platform_results = $6::jsonb,
                compute_seconds = $7,
                thumbnail_r2_key = COALESCE($8, thumbnail_r2_key),
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
            ctx.thumbnail_r2_key or None,
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
    """Mark upload as cancelled and refund any reserved tokens back to the user's wallet."""
    async with pool.acquire() as conn:
        # Fetch reserved token amounts so we can refund them
        row = await conn.fetchrow(
            "SELECT user_id, put_reserved, aic_reserved FROM uploads WHERE id = $1",
            upload_id,
        )
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
        # Refund reserved tokens back to wallet (cancel_requested path: pending cancel was
        # set by the API but token refund was skipped because job was mid-processing)
        if row and (row["put_reserved"] or row["aic_reserved"]):
            await conn.execute(
                """
                UPDATE wallets
                SET put_reserved = GREATEST(0, put_reserved - $1),
                    aic_reserved = GREATEST(0, aic_reserved - $2)
                WHERE user_id = $3
                """,
                row["put_reserved"] or 0,
                row["aic_reserved"] or 0,
                row["user_id"],
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


async def update_stage_progress(pool: asyncpg.Pool, upload_id: str, stage: str, progress: int):
    """Write the current pipeline stage and % progress to the DB so the queue screen
    and upload page timer can both display live status.
    Writes to processing_stage + processing_progress — the same columns the
    single-upload endpoint (/api/uploads/{id}) already reads and returns as
    processingStage / processingProgress to upload.html's polling loop.
    """
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE uploads
                SET processing_stage    = $2,
                    processing_progress = $3,
                    updated_at          = NOW()
                WHERE id = $1
                """,
                upload_id,
                stage,
                progress,
            )
    except Exception:
        # Non-fatal — if columns don't exist yet neither screen will show progress,
        # but the pipeline won't crash
        pass


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


# ============================================================
# Caption memory (few-shot retrieval for caption_stage)
# ============================================================

async def fetch_caption_memory_examples(
    pool: asyncpg.Pool,
    user_id: str,
    category: str,
    limit: int = 3,
) -> List[dict]:
    """
    Load recent caption examples for this user + category for prompt injection.
    Table may not exist yet — returns [] on missing table.
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT ai_title, ai_caption, ai_hashtags, caption_voice, caption_tone, caption_style
                  FROM upload_caption_memory
                 WHERE user_id = $1 AND category = $2
                 ORDER BY created_at DESC
                 LIMIT $3
                """,
                user_id,
                (category or "general").lower(),
                max(1, min(limit, 8)),
            )
        return [dict(r) for r in rows] if rows else []
    except asyncpg.exceptions.UndefinedTableError:
        return []
    except Exception as e:
        logger.debug(f"fetch_caption_memory_examples: {e}")
        return []


async def insert_caption_memory(
    pool: asyncpg.Pool,
    user_id: str,
    upload_id: str,
    category: str,
    platforms: List[str],
    ai_title: Optional[str],
    ai_caption: Optional[str],
    ai_hashtags: Optional[List[str]],
    caption_voice: str = "",
    caption_tone: str = "",
    caption_style: str = "",
    source: str = "auto",
) -> None:
    """Persist one row for future few-shot retrieval. Non-fatal on missing table."""
    if not (ai_title or ai_caption or ai_hashtags):
        return
    try:
        import json as _json
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO upload_caption_memory (
                    user_id, upload_id, category, platforms,
                    ai_title, ai_caption, ai_hashtags,
                    caption_voice, caption_tone, caption_style, source
                ) VALUES (
                    $1, $2::uuid, $3, $4::jsonb,
                    $5, $6, $7::jsonb,
                    $8, $9, $10, $11
                )
                """,
                user_id,
                upload_id,
                (category or "general").lower(),
                _json.dumps(platforms or []),
                ai_title,
                ai_caption,
                _json.dumps(ai_hashtags or []),
                caption_voice or None,
                caption_tone or None,
                caption_style or None,
                source,
            )
    except asyncpg.exceptions.UndefinedTableError:
        pass
    except Exception as e:
        logger.debug(f"insert_caption_memory: {e}")


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
                    token_data = row_dict.get("token_blob") or row_dict.get("token_data")
                    if token_data:
                        if isinstance(token_data, str):
                            parsed = json.loads(token_data)
                            # Double-encoded: json.loads returned a string instead of dict
                            # e.g. token_data = '"{\access_token\:\...\}"'
                            if isinstance(parsed, str):
                                try:
                                    parsed = json.loads(parsed)
                                except Exception:
                                    pass
                        else:
                            parsed = dict(token_data) if hasattr(token_data, "keys") else token_data
                        # Inject platform-specific IDs from the DB row's account_id column
                        # into the token blob if they're missing. This fixes old tokens stored
                        # before the OAuth callback was updated to include these fields.
                        account_id_col = row_dict.get("account_id")
                        if account_id_col and isinstance(parsed, dict):
                            # Instagram: account_id column stores the Instagram Business Account ID
                            # which is exactly what ig_user_id needs to be.
                            # Facebook: account_id stores the Facebook *user* ID (not Page ID),
                            # so we cannot inject it as page_id — FB connections need a reconnect.
                            if platform == "instagram" and not parsed.get("ig_user_id"):
                                parsed["ig_user_id"] = str(account_id_col)
                                logger.debug(f"Injected ig_user_id={account_id_col} from account_id column")
                        return parsed
                    # May be encrypted blob
                    encrypted = row_dict.get("encrypted_token")
                    if encrypted:
                        if isinstance(encrypted, str):
                            parsed = json.loads(encrypted)
                        else:
                            parsed = dict(encrypted) if hasattr(encrypted, "keys") else encrypted
                        if account_id_col and isinstance(parsed, dict):
                            if platform == "instagram" and not parsed.get("ig_user_id"):
                                parsed["ig_user_id"] = str(account_id_col)
                        return parsed
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
                    account_id_col = row_dict.get("account_id")
                    token_data = row_dict.get("token_blob") or row_dict.get("token_data") or row_dict.get("encrypted_token")
                    if token_data:
                        if isinstance(token_data, str):
                            parsed = json.loads(token_data)
                            # Double-encoded guard
                            if isinstance(parsed, str):
                                try:
                                    parsed = json.loads(parsed)
                                except Exception:
                                    pass
                        else:
                            parsed = dict(token_data) if hasattr(token_data, "keys") else token_data
                        if account_id_col and isinstance(parsed, dict):
                            if platform == "instagram" and not parsed.get("ig_user_id"):
                                parsed["ig_user_id"] = str(account_id_col)
                                logger.debug(f"Injected ig_user_id={account_id_col} from connected_accounts.account_id")
                        return parsed
                    return row_dict
            except asyncpg.exceptions.UndefinedTableError:
                pass

        return None
    except Exception as e:
        logger.error(f"Failed to load platform token for {user_id}/{platform}: {e}")
        return None


async def load_platform_token_by_id(pool: asyncpg.Pool, token_id: str) -> Optional[dict]:
    """Load a stored platform OAuth token by its platform_tokens.id (UUID).

    Used for multi-account publishing where the caller knows exactly which
    connected account to publish to.  Returns the parsed token blob with
    'platform' and 'account_name' injected for caller convenience.
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM platform_tokens WHERE id = $1 AND revoked_at IS NULL",
                token_id,
            )
            if not row:
                return None
            row_dict = dict(row)
            token_data = row_dict.get("token_blob") or row_dict.get("token_data")
            if not token_data:
                return None
            if isinstance(token_data, str):
                parsed = json.loads(token_data)
                if isinstance(parsed, str):
                    try:
                        parsed = json.loads(parsed)
                    except Exception:
                        pass
            else:
                parsed = dict(token_data) if hasattr(token_data, "keys") else token_data
            if isinstance(parsed, dict):
                parsed["_platform"] = row_dict.get("platform", "")
                parsed["_account_name"] = row_dict.get("account_name", "")
                parsed["_token_id"] = str(row_dict.get("id", ""))
                account_id_col = row_dict.get("account_id")
                if account_id_col:
                    if row_dict.get("platform") == "instagram" and not parsed.get("ig_user_id"):
                        parsed["ig_user_id"] = str(account_id_col)
            return parsed
    except Exception as e:
        logger.error(f"Failed to load platform token by id {token_id}: {e}")
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


async def save_refreshed_token(
    pool: asyncpg.Pool,
    user_id: str,
    platform: str,
    access_token: str,
    refresh_token: Optional[str] = None,
    open_id: Optional[str] = None,
) -> None:
    """
    Persist a refreshed platform OAuth token back to the database.

    CRITICAL FIX: token_blob is stored as TEXT (json.dumps of encrypted blob).
    Previous version used JSONB || operator which does STRING CONCATENATION on
    TEXT columns — producing '{"kid":"v1"...}{"access_token":"new"}' garbage.

    Correct approach:
      1. Read current token_blob TEXT from DB
      2. Parse it: json.loads -> encrypted dict {kid, nonce, ciphertext}
      3. Decrypt -> plaintext token dict
      4. Merge new fields into plaintext dict
      5. Re-encrypt -> new encrypted blob
      6. Write back as json.dumps(new_encrypted_blob)  — clean TEXT overwrite

    This ensures the stored value is always a single valid JSON string.
    """
    if not pool or not access_token:
        return

    try:
        # Import encrypt/decrypt from publish_stage (same module that reads tokens)
        # We do a lazy import to avoid circular dependency
        from .publish_stage import decrypt_token_blob, init_enc_keys

        # Also need encrypt_blob — import from app context via env-based reimplementation
        import os, base64, secrets as _secrets
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _AESGCM

        TOKEN_ENC_KEYS_RAW = os.environ.get("TOKEN_ENC_KEYS", "")
        _enc_keys: Dict[str, bytes] = {}
        _current_kid = "v1"

        if TOKEN_ENC_KEYS_RAW:
            clean = TOKEN_ENC_KEYS_RAW.strip().strip('"').replace("\n", "")
            for part in [p.strip() for p in clean.split(",") if p.strip()]:
                if ":" not in part:
                    continue
                kid, b64key = part.split(":", 1)
                try:
                    raw = base64.b64decode(b64key.strip())
                    if len(raw) == 32:
                        _enc_keys[kid.strip()] = raw
                        _current_kid = kid.strip()
                except Exception:
                    pass

        def _encrypt(data: dict) -> str:
            """Re-encrypt a token dict and return json.dumps of the encrypted blob."""
            if not _enc_keys:
                return json.dumps(data)  # no keys — store plaintext (dev mode)
            key = _enc_keys[_current_kid]
            aesgcm = _AESGCM(key)
            nonce = _secrets.token_bytes(12)
            ct = aesgcm.encrypt(nonce, json.dumps(data).encode(), None)
            blob = {
                "kid": _current_kid,
                "nonce": base64.b64encode(nonce).decode(),
                "ciphertext": base64.b64encode(ct).decode(),
            }
            return json.dumps(blob)

        async with pool.acquire() as conn:
            # ── Try platform_tokens ───────────────────────────────────────
            for table in ("platform_tokens", "connected_accounts"):
                try:
                    row = await conn.fetchrow(
                        f"SELECT id, token_blob FROM {table} "
                        f"WHERE user_id = $1 AND platform = $2 "
                        f"ORDER BY updated_at DESC LIMIT 1",
                        user_id, platform,
                    )
                except asyncpg.exceptions.UndefinedTableError:
                    continue
                except Exception:
                    continue

                if not row:
                    continue

                row_id = row["id"]
                raw_blob = row["token_blob"]

                # Parse the current stored blob (TEXT or JSONB both handled)
                current_encrypted: Optional[dict] = None
                try:
                    if isinstance(raw_blob, str):
                        current_encrypted = json.loads(raw_blob)
                        # Handle double-encoded
                        if isinstance(current_encrypted, str):
                            current_encrypted = json.loads(current_encrypted)
                    elif isinstance(raw_blob, dict):
                        current_encrypted = dict(raw_blob)
                    else:
                        current_encrypted = dict(raw_blob) if raw_blob else None
                except Exception:
                    current_encrypted = None

                # Decrypt to get plaintext token fields
                current_plain: dict = {}
                if current_encrypted:
                    try:
                        init_enc_keys()
                        current_plain = decrypt_token_blob(current_encrypted) or {}
                    except Exception:
                        # If decrypt fails, start fresh with just the new token
                        current_plain = {}

                # Merge new values into plaintext
                current_plain["access_token"] = access_token
                if refresh_token:
                    current_plain["refresh_token"] = refresh_token
                if open_id:
                    current_plain["open_id"] = open_id

                # Re-encrypt and write back as clean TEXT
                new_blob_str = _encrypt(current_plain)

                try:
                    await conn.execute(
                        f"UPDATE {table} SET token_blob = $1, updated_at = NOW() "
                        f"WHERE id = $2",
                        new_blob_str,
                        row_id,
                    )
                    logger.info(f"save_refreshed_token: persisted {platform} token for user={user_id} table={table}")
                    return
                except Exception as write_err:
                    logger.warning(f"save_refreshed_token: write failed on {table}: {write_err}")
                    continue

            logger.warning(f"save_refreshed_token: no row found for {platform} user={user_id}")

    except Exception as e:
        logger.warning(f"save_refreshed_token failed (non-fatal) for {platform} user={user_id}: {e}")
