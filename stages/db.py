"""
UploadM8 Worker Database Functions
===================================
Database helpers used by the worker pipeline stages.
"""

from __future__ import annotations

import json
import logging
import math
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

    def _has_content(v, is_platform_map: bool = False) -> bool:
        if v is None:
            return False
        if is_platform_map and isinstance(v, dict):
            return any(
                (isinstance(x, list) and len(x) > 0)
                or (isinstance(x, str) and x.strip())
                for x in (v.values() or [])
            )
        if isinstance(v, (list, tuple)):
            return len(v) > 0
        if isinstance(v, dict):
            return len(v) > 0
        return bool(v)

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

        # ── 2. user_preferences (fallback or merge) ─────────────────────────
        try:
            prefs_row = await conn.fetchrow(
                "SELECT * FROM user_preferences WHERE user_id = $1", user_id
            )
            if prefs_row:
                prefs_d = dict(prefs_row)
                if not result:
                    result = prefs_d
                else:
                    # Merge preference columns so POST /api/settings/preferences is respected
                    for k in ("styled_thumbnails", "auto_thumbnails", "auto_captions", "thumbnail_interval",
                              "default_privacy", "ai_hashtags_enabled", "ai_hashtag_count", "always_hashtags",
                              "blocked_hashtags", "platform_hashtags", "email_notifications", "discord_webhook",
                              "use_audio_context"):
                        if k in prefs_d and prefs_d[k] is not None:
                            result[k] = prefs_d[k]
                    # Ensure camelCase aliases for hashtag fields (context.get_effective_hashtags checks both)
                    for snake, camel in [("platform_hashtags", "platformHashtags"), ("always_hashtags", "alwaysHashtags"), ("blocked_hashtags", "blockedHashtags")]:
                        if result.get(snake) is not None and result.get(camel) is None:
                            result[camel] = result[snake]
                    result.setdefault("styled_thumbnails", True)
                    result.setdefault("styledThumbnails", result.get("styled_thumbnails", True))
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
                        "autoThumbnails":   "auto_thumbnails",
                        "styledThumbnails": "styled_thumbnails",
                        "captionStyle":     "caption_style",
                        "captionTone":      "caption_tone",
                        "captionVoice":     "caption_voice",
                        "captionFrameCount":"caption_frame_count",
                        "maxHashtags":      "max_hashtags",
                        "trillOpenaiModel": "openai_model",
                        "useAudioContext":  "use_audio_context",
                        "audioTranscription": "audio_transcription",
                    }
                    for camel, snake in FIELD_MAP.items():
                        val = prefs.get(camel)
                        if val is None:
                            val = prefs.get(snake)
                        if val is not None:
                            # Do not let empty hashtag prefs from users.preferences
                            # wipe non-empty platform/always/blocked tags from user_preferences.
                            if snake in ("always_hashtags", "blocked_hashtags") and not _has_content(val):
                                continue
                            if snake == "platform_hashtags" and not _has_content(val, is_platform_map=True):
                                continue
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
                    "token_row_id": getattr(r, "token_row_id", None),
                    "account_id": getattr(r, "account_id", None),
                    "account_name": getattr(r, "account_name", None),
                    "account_username": getattr(r, "account_username", None),
                    "account_avatar": getattr(r, "account_avatar", None),
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
        # ML-ready feature persistence (non-fatal): keeps rich context per upload so
        # offline scorers can learn what creative strategies perform over time.
        try:
            await conn.execute(
                """
                INSERT INTO upload_feature_events
                    (user_id, upload_id, category, audio_context, vision_context,
                     video_understanding, thumbnail_brief, output_artifacts,
                     ai_title, ai_caption, ai_hashtags)
                VALUES
                    ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb, $9, $10, $11::jsonb)
                """,
                str(ctx.user_id),
                str(ctx.upload_id),
                str(ctx.get_canonical_category() or ""),
                json.dumps(getattr(ctx, "audio_context", None) or {}),
                json.dumps(getattr(ctx, "vision_context", None) or {}),
                json.dumps(getattr(ctx, "video_understanding", None) or {}),
                json.dumps(getattr(ctx, "thumbnail_brief", None) or {}),
                json.dumps(getattr(ctx, "output_artifacts", None) or {}),
                getattr(ctx, "ai_title", None),
                getattr(ctx, "ai_caption", None),
                json.dumps(getattr(ctx, "ai_hashtags", None) or []),
            )
        except Exception as e:
            logger.debug(f"upload_feature_events insert skipped: {e}")


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
    predicted_quality_score: Optional[float] = None,
    strategy_json: Optional[Dict[str, Any]] = None,
    platform_winners: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist one row for future few-shot retrieval. Non-fatal on missing table."""
    if not (ai_title or ai_caption or ai_hashtags):
        return
    try:
        import json as _json
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO upload_caption_memory (
                        user_id, upload_id, category, platforms,
                        ai_title, ai_caption, ai_hashtags,
                        caption_voice, caption_tone, caption_style, source,
                        predicted_quality_score, strategy_json, platform_winners
                    ) VALUES (
                        $1, $2::uuid, $3, $4::jsonb,
                        $5, $6, $7::jsonb,
                        $8, $9, $10, $11,
                        $12, $13::jsonb, $14::jsonb
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
                    float(predicted_quality_score) if predicted_quality_score is not None else None,
                    _json.dumps(strategy_json or {}),
                    _json.dumps(platform_winners or {}),
                )
            except asyncpg.exceptions.UndefinedColumnError:
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


async def fetch_recent_thumbnail_style_signatures(
    pool: asyncpg.Pool,
    user_id: str,
    platform: str,
    limit: int = 24,
) -> List[str]:
    """
    Return recent style signatures for anti-repeat rendering memory.
    Non-fatal if table is absent.
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT style_signature
                  FROM upload_thumbnail_style_memory
                 WHERE user_id = $1
                   AND platform = $2
                 ORDER BY created_at DESC
                 LIMIT $3
                """,
                str(user_id),
                str(platform or "").lower(),
                max(1, min(limit, 100)),
            )
        out: List[str] = []
        for r in rows or []:
            s = str(r.get("style_signature") or "").strip()
            if s:
                out.append(s)
        return out
    except asyncpg.exceptions.UndefinedTableError:
        return []
    except Exception as e:
        logger.debug(f"fetch_recent_thumbnail_style_signatures: {e}")
        return []


async def fetch_recent_thumbnail_style_history(
    pool: asyncpg.Pool,
    user_id: str,
    platform: str,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """
    Return recent style history rows for diversity policy (signature + pack + score).
    Non-fatal when table is missing.
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT style_signature, style_pack, score, created_at
                  FROM upload_thumbnail_style_memory
                 WHERE user_id = $1
                   AND platform = $2
                 ORDER BY created_at DESC
                 LIMIT $3
                """,
                str(user_id),
                str(platform or "").lower(),
                max(1, min(limit, 120)),
            )
        out: List[Dict[str, Any]] = []
        for r in rows or []:
            out.append(
                {
                    "signature": str(r.get("style_signature") or "").strip(),
                    "style_pack": str(r.get("style_pack") or "").strip().lower(),
                    "score": float(r.get("score") or 0.0),
                    "created_at": str(r.get("created_at") or ""),
                }
            )
        return out
    except asyncpg.exceptions.UndefinedTableError:
        return []
    except Exception as e:
        logger.debug(f"fetch_recent_thumbnail_style_history: {e}")
        return []


async def insert_thumbnail_style_signature(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    upload_id: str,
    platform: str,
    style_signature: str,
    style_pack: str = "",
    score: float = 0.0,
) -> None:
    """
    Persist a style signature for anti-repeat memory.
    Ignores duplicates per (upload_id, platform, style_signature).
    """
    sig = (style_signature or "").strip()
    if not sig:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO upload_thumbnail_style_memory (
                    user_id, upload_id, platform, style_signature, style_pack, score
                ) VALUES (
                    $1, $2::uuid, $3, $4, $5, $6
                )
                ON CONFLICT (upload_id, platform, style_signature) DO NOTHING
                """,
                str(user_id),
                str(upload_id),
                str(platform or "").lower(),
                sig,
                (style_pack or None),
                float(score or 0.0),
            )
    except asyncpg.exceptions.UndefinedTableError:
        pass
    except Exception as e:
        logger.debug(f"insert_thumbnail_style_signature: {e}")


def _m8_engagement_raw(views: int, likes: int, comments: int, shares: int) -> float:
    """Weighted engagement for ranking priors (same spirit as KPI dashboards)."""
    return float(
        max(0, views)
        + 2 * max(0, likes)
        + 3 * max(0, comments)
        + 2 * max(0, shares)
    )


def _m8_normalize_prior(avg_raw: float) -> float:
    """Map rolling average raw engagement to a soft prior in [-5, 10] for M8 ranking."""
    if avg_raw <= 0:
        return 0.0
    # Log-scaled: small accounts get small nudge; viral history gets a larger (capped) bump
    x = math.log10(avg_raw + 1.0)
    prior = (x - 1.5) * 4.0
    return max(-5.0, min(10.0, prior))


async def fetch_m8_historical_signals(
    pool: asyncpg.Pool,
    user_id: str,
    category: str,
    platforms: List[str],
    *,
    prior_row_limit: int = 40,
    pattern_limit: int = 6,
) -> Dict[str, Any]:
    """
    Analytics-backed priors + caption pattern memory for M8 v2.

    Reads from `upload_caption_memory` JOIN `uploads` (views/likes/comments/shares
    filled by the analytics sync loop). Per-platform rolling engagement and top
    caption snippets for prompt injection.

    Returns a dict suitable for m8_engine.rank_and_select (platform keys) plus
    __pattern_corpus__ / __strategy_priors__ / __meta__ for prompts.
    """
    out: Dict[str, Any] = {
        "__pattern_corpus__": [],
        "__strategy_priors__": {"top": [], "lookback_days": 180},
        "__meta__": {"source": "db", "ok": False},
    }
    plats = [str(p).lower() for p in (platforms or []) if p]
    if not plats or not user_id:
        return out

    cat = (category or "general").lower()

    try:
        async with pool.acquire() as conn:
            for plat in plats:
                plat_json = json.dumps([plat])
                try:
                    rows = await conn.fetch(
                        """
                        SELECT COALESCE(u.views, 0)::bigint AS views,
                               COALESCE(u.likes, 0)::bigint AS likes,
                               COALESCE(u.comments, 0)::bigint AS comments,
                               COALESCE(u.shares, 0)::bigint AS shares
                          FROM upload_caption_memory ucm
                          INNER JOIN uploads u ON u.id = ucm.upload_id
                         WHERE ucm.user_id = $1
                           AND ucm.category = $2
                           AND ucm.platforms @> $3::jsonb
                           AND u.created_at > NOW() - INTERVAL '180 days'
                         ORDER BY u.created_at DESC
                         LIMIT $4
                        """,
                        user_id,
                        cat,
                        plat_json,
                        max(5, min(prior_row_limit, 80)),
                    )
                except asyncpg.exceptions.UndefinedColumnError:
                    rows = await conn.fetch(
                        """
                        SELECT COALESCE(u.views, 0)::bigint AS views,
                               COALESCE(u.likes, 0)::bigint AS likes,
                               COALESCE(u.comments, 0)::bigint AS comments,
                               COALESCE(u.shares, 0)::bigint AS shares
                          FROM upload_caption_memory ucm
                          INNER JOIN uploads u ON u.id = ucm.upload_id
                         WHERE ucm.user_id = $1
                           AND ucm.category = $2
                           AND ucm.platforms @> $3::jsonb
                         ORDER BY u.created_at DESC
                         LIMIT $4
                        """,
                        user_id,
                        cat,
                        plat_json,
                        max(5, min(prior_row_limit, 80)),
                    )

                if not rows:
                    continue
                raws = [_m8_engagement_raw(int(r["views"]), int(r["likes"]), int(r["comments"]), int(r["shares"])) for r in rows]
                avg_raw = sum(raws) / len(raws)
                out[plat] = {
                    "engagement_prior": round(_m8_normalize_prior(avg_raw), 4),
                    "sample_n": len(rows),
                    "avg_engagement_raw": round(avg_raw, 2),
                }

            # Pattern memory: best past captions for this category + platform (rolling pool)
            patterns: List[Dict[str, Any]] = []
            for plat in plats:
                plat_json = json.dumps([plat])
                try:
                    prow = await conn.fetch(
                        """
                        SELECT ucm.ai_caption,
                               COALESCE(ucm.caption_style, '') AS caption_style,
                               COALESCE(ucm.caption_tone, '') AS caption_tone,
                               (COALESCE(u.views, 0) + 2 * COALESCE(u.likes, 0)
                                + 3 * COALESCE(u.comments, 0) + 2 * COALESCE(u.shares, 0))::double precision AS eng_raw
                          FROM upload_caption_memory ucm
                          INNER JOIN uploads u ON u.id = ucm.upload_id
                         WHERE ucm.user_id = $1
                           AND ucm.category = $2
                           AND ucm.platforms @> $3::jsonb
                           AND LENGTH(TRIM(COALESCE(ucm.ai_caption, ''))) > 24
                           AND u.created_at > NOW() - INTERVAL '180 days'
                         ORDER BY eng_raw DESC, u.created_at DESC
                         LIMIT $4
                        """,
                        user_id,
                        cat,
                        plat_json,
                        max(1, min(pattern_limit, 12)),
                    )
                except asyncpg.exceptions.UndefinedColumnError:
                    prow = await conn.fetch(
                        """
                        SELECT ucm.ai_caption,
                               COALESCE(ucm.caption_style, '') AS caption_style,
                               COALESCE(ucm.caption_tone, '') AS caption_tone,
                               (COALESCE(u.views, 0) + 2 * COALESCE(u.likes, 0)
                                + 3 * COALESCE(u.comments, 0) + 2 * COALESCE(u.shares, 0))::double precision AS eng_raw
                          FROM upload_caption_memory ucm
                          INNER JOIN uploads u ON u.id = ucm.upload_id
                         WHERE ucm.user_id = $1
                           AND ucm.category = $2
                           AND ucm.platforms @> $3::jsonb
                           AND LENGTH(TRIM(COALESCE(ucm.ai_caption, ''))) > 24
                         ORDER BY eng_raw DESC, u.created_at DESC
                         LIMIT $4
                        """,
                        user_id,
                        cat,
                        plat_json,
                        max(1, min(pattern_limit, 12)),
                    )
                for r in prow or []:
                    cap = (r.get("ai_caption") or "").strip()
                    if len(cap) < 24:
                        continue
                    patterns.append(
                        {
                            "platform": plat,
                            "snippet": cap[:420],
                            "caption_style": (r.get("caption_style") or "").strip(),
                            "caption_tone": (r.get("caption_tone") or "").strip(),
                            "engagement_raw": float(r.get("eng_raw") or 0),
                        }
                    )

            # De-dup near-identical snippets while preserving order
            seen: set = set()
            uniq: List[Dict[str, Any]] = []
            for p in patterns:
                key = (p["platform"], p["snippet"][:80])
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(p)
            out["__pattern_corpus__"] = uniq[: max(3, pattern_limit)]

            # First-class ML priors from upload_quality_scores_daily.
            # We pull a user-level top strategy table for prompt biasing.
            try:
                strat_rows = await conn.fetch(
                    """
                    SELECT
                        strategy_key,
                        SUM(samples)::bigint AS samples,
                        CASE
                          WHEN SUM(samples) > 0
                          THEN SUM(mean_engagement * samples)::double precision / NULLIF(SUM(samples)::double precision, 0)
                          ELSE 0.0
                        END AS weighted_mean_engagement,
                        MAX(ci95_high)::double precision AS max_ci95_high,
                        COUNT(DISTINCT day)::int AS days_with_data
                    FROM upload_quality_scores_daily
                    WHERE user_id = $1::uuid
                      AND day >= (CURRENT_DATE - (180::int || ' days')::interval)::date
                      AND (
                            platform = 'all'
                            OR platform = ANY($2::text[])
                      )
                    GROUP BY strategy_key
                    ORDER BY weighted_mean_engagement DESC, samples DESC
                    LIMIT 10
                    """,
                    user_id,
                    ["all"] + plats,
                )
                out["__strategy_priors__"] = {
                    "lookback_days": 180,
                    "top": [
                        {
                            "strategy_key": str(r.get("strategy_key") or "default"),
                            "samples": int(r.get("samples") or 0),
                            "weighted_mean_engagement": float(r.get("weighted_mean_engagement") or 0),
                            "max_ci95_high": float(r.get("max_ci95_high") or 0),
                            "days_with_data": int(r.get("days_with_data") or 0),
                        }
                        for r in (strat_rows or [])
                    ],
                }
            except Exception as strat_err:
                logger.debug(f"fetch_m8_historical_signals strategy priors skipped: {strat_err}")

            out["__meta__"] = {"source": "upload_caption_memory+uploads", "ok": True}
    except asyncpg.exceptions.UndefinedTableError:
        return {"__pattern_corpus__": [], "__meta__": {"source": "missing_table", "ok": False}}
    except Exception as e:
        logger.debug(f"fetch_m8_historical_signals: {e}")
        return {"__pattern_corpus__": [], "__meta__": {"source": "error", "ok": False, "err": str(e)[:120]}}

    return out


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
                        if isinstance(parsed, dict):
                            parsed["_account_name"] = row_dict.get("account_name", "")
                            parsed["_account_username"] = row_dict.get("account_username", "") or ""
                            parsed["_account_avatar"] = row_dict.get("account_avatar", "") or ""
                        return parsed
                    # May be encrypted blob
                    encrypted = row_dict.get("encrypted_token")
                    if encrypted:
                        if isinstance(encrypted, str):
                            parsed = json.loads(encrypted)
                        else:
                            parsed = dict(encrypted) if hasattr(encrypted, "keys") else encrypted
                        account_id_col = row_dict.get("account_id")
                        if account_id_col and isinstance(parsed, dict):
                            if platform == "instagram" and not parsed.get("ig_user_id"):
                                parsed["ig_user_id"] = str(account_id_col)
                        if isinstance(parsed, dict):
                            parsed["_account_name"] = row_dict.get("account_name", "")
                            parsed["_account_username"] = row_dict.get("account_username", "") or ""
                            parsed["_account_avatar"] = row_dict.get("account_avatar", "") or ""
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
                        if isinstance(parsed, dict):
                            parsed["_account_name"] = row_dict.get("account_name", "")
                            parsed["_account_username"] = row_dict.get("account_username", "") or ""
                            parsed["_account_avatar"] = row_dict.get("account_avatar", "") or ""
                        return parsed
                    return row_dict
            except asyncpg.exceptions.UndefinedTableError:
                pass

        return None
    except Exception as e:
        logger.error(f"Failed to load platform token for {user_id}/{platform}: {e}")
        return None


async def load_platform_token_with_identity(
    pool: asyncpg.Pool,
    user_id: str,
    platform: str,
    token_row_id: Optional[str] = None,
) -> tuple:
    """
    Load a platform token AND its account identity fields.

    If token_row_id is provided (a specific platform_tokens.id UUID),
    load that exact row — used when target_accounts specifies which account to publish to.

    Returns (token_blob_dict, identity_dict) or (None, None).
    identity_dict = {token_row_id, account_id, account_username, account_name, account_avatar}
    """
    try:
        async with pool.acquire() as conn:
            if token_row_id:
                row = await conn.fetchrow(
                    """SELECT * FROM platform_tokens
                       WHERE id = $1 AND user_id = $2 AND revoked_at IS NULL""",
                    token_row_id, user_id,
                )
            else:
                row = await conn.fetchrow(
                    """SELECT * FROM platform_tokens
                       WHERE user_id = $1 AND platform = $2 AND revoked_at IS NULL
                       ORDER BY is_primary DESC NULLS LAST, updated_at DESC
                       LIMIT 1""",
                    user_id, platform,
                )

            if not row:
                return None, None

            row_dict = dict(row)
            identity = {
                "token_row_id":     str(row_dict["id"]),
                "account_id":       row_dict.get("account_id") or "",
                "account_username": row_dict.get("account_username") or "",
                "account_name":     row_dict.get("account_name") or "",
                "account_avatar":   row_dict.get("account_avatar") or "",
            }

            token_data = row_dict.get("token_blob") or row_dict.get("token_data")
            if token_data:
                if isinstance(token_data, str):
                    parsed = json.loads(token_data)
                    if isinstance(parsed, str):
                        try:
                            parsed = json.loads(parsed)
                        except Exception:
                            pass
                else:
                    parsed = dict(token_data) if hasattr(token_data, "keys") else token_data

                account_id_col = row_dict.get("account_id")
                if account_id_col and isinstance(parsed, dict):
                    if platform == "instagram" and not parsed.get("ig_user_id"):
                        parsed["ig_user_id"] = str(account_id_col)

                if isinstance(parsed, dict):
                    parsed["_account_name"] = row_dict.get("account_name", "")
                    parsed["_account_username"] = row_dict.get("account_username", "") or ""
                    parsed["_account_avatar"] = row_dict.get("account_avatar", "") or ""
                return parsed, identity

            return None, None

    except Exception as e:
        logger.warning(f"load_platform_token_with_identity failed: {e}")
        return None, None


async def load_all_platform_token_ids(pool: asyncpg.Pool, user_id: str, platform: str) -> list[str]:
    """Return all platform_tokens.id for the user's connected accounts on this platform.
    Used for multi-account publishing: when no target_accounts specified, publish to ALL.
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id FROM platform_tokens
                WHERE user_id = $1 AND platform = $2 AND revoked_at IS NULL
                ORDER BY updated_at DESC
                """,
                user_id,
                platform,
            )
            return [str(r["id"]) for r in rows]
    except Exception as e:
        logger.error(f"Failed to load platform token ids for {user_id}/{platform}: {e}")
        return []


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
                parsed["_account_username"] = row_dict.get("account_username", "") or ""
                parsed["_account_avatar"] = row_dict.get("account_avatar", "") or ""
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

async def write_system_event_log(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    event_category: str,
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[dict] = None,
    severity: str = "INFO",
    outcome: str = "SUCCESS",
):
    """Write a user-facing audit event to system_event_log (non-fatal best effort)."""
    try:
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO system_event_log
                        (user_id, event_category, action, resource_type, resource_id,
                         details, severity, outcome, created_at)
                    VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, $7, $8, NOW())
                    """,
                    user_id,
                    event_category,
                    action,
                    resource_type,
                    resource_id,
                    json.dumps(details or {}),
                    severity,
                    outcome,
                )
            except asyncpg.exceptions.UndefinedTableError:
                # Older DBs may not have migration 700 yet.
                return
    except Exception as e:
        logger.debug(f"write_system_event_log failed (non-fatal): {e}")


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
    """Load admin Discord webhook URL from admin_settings.settings_json (notifications.admin_webhook_url)."""
    try:
        async with pool.acquire() as conn:
            val = await conn.fetchval(
                "SELECT settings_json->'notifications'->>'admin_webhook_url' FROM admin_settings WHERE id = 1"
            )
            return str(val).strip() if val else None
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


async def update_platform_token_identity_from_upload(
    pool: asyncpg.Pool,
    token_row_id: str,
    user_id: str,
    account_username: Optional[str] = None,
    account_name: Optional[str] = None,
    account_id: Optional[str] = None,
) -> None:
    """
    Backfill platform_tokens with account identity when we get it from post-upload
    response (e.g. YouTube channelTitle). Only updates fields that are currently empty.
    """
    if not pool or not token_row_id or not user_id:
        return
    updates = []
    params: list = [user_id, token_row_id]
    idx = 3
    if account_username and (account_username or "").strip():
        updates.append(f"account_username = COALESCE(NULLIF(TRIM(account_username), ''), ${idx})")
        params.append((account_username or "").strip())
        idx += 1
    if account_name and (account_name or "").strip():
        updates.append(f"account_name = COALESCE(NULLIF(TRIM(account_name), ''), ${idx})")
        params.append((account_name or "").strip())
        idx += 1
    if account_id and (account_id or "").strip():
        updates.append(f"account_id = COALESCE(NULLIF(TRIM(account_id), ''), ${idx})")
        params.append((account_id or "").strip())
        idx += 1
    if not updates:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"""UPDATE platform_tokens SET {', '.join(updates)}, updated_at = NOW()
                    WHERE id = $2 AND user_id = $1 AND revoked_at IS NULL""",
                params[0],
                params[1],
                *params[2:],
            )
            logger.debug(f"Updated platform_tokens identity for token_row_id={token_row_id[:8]}")
    except Exception as e:
        logger.warning(f"update_platform_token_identity_from_upload failed: {e}")


async def save_refreshed_token(
    pool: asyncpg.Pool,
    user_id: str,
    platform: str,
    access_token: str,
    refresh_token: Optional[str] = None,
    open_id: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
    token_row_id: Optional[str] = None,
    token_data: Optional[Dict[str, Any]] = None,
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
            # Fast path: caller already knows the concrete platform_tokens row and
            # has the decrypted token payload. This avoids an extra SELECT per
            # refresh call in loops that already fetched all token rows.
            if token_row_id and token_data is not None:
                plain = dict(token_data or {})
                plain["access_token"] = access_token
                if refresh_token:
                    plain["refresh_token"] = refresh_token
                if open_id:
                    plain["open_id"] = open_id
                if extra_fields:
                    for k, v in extra_fields.items():
                        if v is not None:
                            plain[k] = v

                new_blob_str = _encrypt(plain)
                cmd = await conn.execute(
                    """
                    UPDATE platform_tokens
                       SET token_blob = $1,
                           updated_at = NOW()
                     WHERE id = $2
                       AND user_id = $3
                       AND revoked_at IS NULL
                    """,
                    new_blob_str,
                    token_row_id,
                    user_id,
                )
                if str(cmd).upper().startswith("UPDATE 1"):
                    logger.info(
                        f"save_refreshed_token: persisted {platform} token for user={user_id} "
                        f"token_row_id={str(token_row_id)[:8]}"
                    )
                    return
                logger.warning(
                    f"save_refreshed_token: token_row_id not found for {platform} "
                    f"user={user_id} token_row_id={token_row_id}"
                )

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
                if extra_fields:
                    for k, v in extra_fields.items():
                        if v is not None:
                            current_plain[k] = v

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
