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

from core.helpers import (
    _safe_col,
    coerce_jsonb_dict,
    coerce_jsonb_list,
    merge_platform_hashtag_overlay,
)
from core.sql_allowlist import (
    OAUTH_TOKEN_STORAGE_TABLES,
    OAUTH_TOKEN_STORAGE_TABLES_ORDERED,
    UPLOADS_AI_GENERATED_METADATA_COLUMNS,
    assert_relation_name,
    assert_set_fragments_columns,
)

from .context import JobContext, is_placeholder_upload_caption, is_placeholder_upload_title

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

        # ── 2. user_preferences (fallback or merge) ─────────────────────────
        try:
            prefs_row = await conn.fetchrow(
                "SELECT * FROM user_preferences WHERE user_id = $1", user_id
            )
            if prefs_row:
                prefs_d = dict(prefs_row)
                # Defensive: even with the JSONB codec registered, legacy rows
                # may already be double-encoded on disk (the bad
                # `'"["tester","qwe"]"'` payload). Coerce hashtag columns here
                # so the worker never feeds a string into get_effective_hashtags.
                if "always_hashtags" in prefs_d:
                    prefs_d["always_hashtags"] = coerce_jsonb_list(prefs_d["always_hashtags"])
                if "blocked_hashtags" in prefs_d:
                    prefs_d["blocked_hashtags"] = coerce_jsonb_list(prefs_d["blocked_hashtags"])
                if "platform_hashtags" in prefs_d:
                    prefs_d["platform_hashtags"] = coerce_jsonb_dict(
                        prefs_d["platform_hashtags"], default={}
                    )
                if not result:
                    result = {
                        k: v
                        for k, v in prefs_d.items()
                        if k not in {"user_id", "created_at", "updated_at", "id"}
                    }
                else:
                    # Merge all preference columns from user_preferences (authoritative for upload prefs).
                    _skip_prefs_merge = frozenset(
                        {"user_id", "created_at", "updated_at", "id"}
                    )
                    for k, v in prefs_d.items():
                        if k in _skip_prefs_merge or v is None:
                            continue
                        result[k] = v
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
                # `coerce_jsonb_dict` peels both single- and double-JSON-encoded
                # strings so worker reads survive the historical write bug where
                # `users.preferences` was passed through `json.dumps()` on top of
                # the asyncpg JSONB codec.
                if isinstance(prefs_val, str):
                    prefs = coerce_jsonb_dict(prefs_val, default={})
                elif hasattr(prefs_val, "keys"):
                    prefs = dict(prefs_val)
                else:
                    prefs = {}

                if isinstance(prefs, dict) and prefs:
                    # Same defensive coercion for hashtag fields nested inside
                    # users.preferences — they are the source of truth that the
                    # frontend writes to, but a partial string value here would
                    # silently override the (correct) user_preferences row.
                    for camel, snake in (
                        ("alwaysHashtags", "always_hashtags"),
                        ("blockedHashtags", "blocked_hashtags"),
                    ):
                        if camel in prefs:
                            prefs[camel] = coerce_jsonb_list(prefs[camel])
                        if snake in prefs:
                            prefs[snake] = coerce_jsonb_list(prefs[snake])
                    for camel, snake in (
                        ("platformHashtags", "platform_hashtags"),
                    ):
                        if camel in prefs:
                            prefs[camel] = coerce_jsonb_dict(prefs[camel], default={})
                        if snake in prefs:
                            prefs[snake] = coerce_jsonb_dict(prefs[snake], default={})
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
                        "thumbnailInterval": "thumbnail_interval",
                        "captionStyle":     "caption_style",
                        "captionTone":      "caption_tone",
                        "captionVoice":     "caption_voice",
                        "captionFrameCount":"caption_frame_count",
                        "maxHashtags":      "max_hashtags",
                        "hashtagPosition":  "hashtag_position",
                        "trillOpenaiModel": "openai_model",
                        "useAudioContext":  "use_audio_context",
                        "audioTranscription": "audio_transcription",
                        "thumbnailStudioEnabled": "thumbnail_studio_enabled",
                        "thumbnailStudioEngineEnabled": "thumbnail_studio_engine_enabled",
                        "thumbnailPikzelsEnabled": "thumbnail_pikzels_enabled",
                        "thumbnailPersonaEnabled": "thumbnail_persona_enabled",
                        "thumbnailDefaultPersonaId": "thumbnail_default_persona_id",
                        "thumbnailPersonaStrength": "thumbnail_persona_strength",
                        "thumbnailPikzelsPersonaId": "thumbnail_pikzels_persona_id",
                        "thumbnailStyleEnabled": "thumbnail_style_enabled",
                        "thumbnailPikzelsStyleEnabled": "thumbnail_pikzels_style_enabled",
                        "thumbnailPikzelsStyleId": "thumbnail_pikzels_style_id",
                        "thumbnailStylePikzelsId": "thumbnail_style_pikzels_id",
                        "thumbnailStyle": "thumbnail_style",
                        "thumbnailStylePrompt": "thumbnail_style_prompt",
                        "thumbnailStudioStrict": "thumbnail_studio_strict",
                        "thumbnailPikzelsStrict": "thumbnail_pikzels_strict",
                        "thumbnailSelectionMode": "thumbnail_selection_mode",
                        "thumbnailRenderPipeline": "thumbnail_render_pipeline",
                        "aiServiceTelemetry": "ai_service_telemetry",
                        "aiServiceDashcamOSD": "ai_service_dashcam_osd",
                        "aiServiceAudioSignals": "ai_service_audio_signals",
                        "aiServiceMusicDetection": "ai_service_music_detection",
                        "aiServiceAudioSummary": "ai_service_audio_summary",
                        "aiServiceEmotionSignals": "ai_service_emotion_signals",
                        "aiServiceCaptionWriter": "ai_service_caption_writer",
                        "aiServiceThumbnailDesigner": "ai_service_thumbnail_designer",
                        "aiServiceSpeechToText": "ai_service_speech_to_text",
                        "aiServiceSceneUnderstanding": "ai_service_scene_understanding",
                        # Settings UI saves webhook under discordWebhook in users.preferences;
                        # user_settings row may still carry discord_webhook=NULL, which would
                        # block the generic "if k not in result" merge from copying camelCase.
                        "discordWebhook": "discord_webhook",
                    }
                    for camel, snake in FIELD_MAP.items():
                        val = prefs.get(camel)
                        if val is None:
                            val = prefs.get(snake)
                        if val is None:
                            continue
                        # platformHashtags: merge, never replace a rich row with bare {}.
                        if camel == "platformHashtags":
                            base_ph = result.get("platformHashtags") or result.get(
                                "platform_hashtags"
                            )
                            merged_ph = merge_platform_hashtag_overlay(base_ph, val)
                            result["platformHashtags"] = merged_ph
                            result["platform_hashtags"] = merged_ph
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

        # ── 4. user_color_preferences — platform badge / accent colors ────────
        try:
            color_row = await conn.fetchrow(
                """
                SELECT tiktok_color, youtube_color, instagram_color,
                       facebook_color, accent_color
                FROM user_color_preferences
                WHERE user_id = $1
                """,
                user_id,
            )
            if color_row:
                from services.platform_colors import normalize_platform_colors

                colors = normalize_platform_colors(dict(color_row))
                result["platform_colors"] = colors
                result["platformColors"] = colors
        except asyncpg.exceptions.UndefinedTableError:
            pass
        except Exception as e:
            logger.debug("Could not read user_color_preferences for %s (non-fatal): %s", user_id, e)

    try:
        from core.upload_baseline_defaults import apply_upload_baseline_defaults

        apply_upload_baseline_defaults(result)
    except Exception as e:
        logger.debug("upload baseline defaults skipped for %s: %s", user_id, e)

    return result


async def merge_pikzels_thumbnail_persona_id(
    pool: asyncpg.Pool, user_id: str, settings: Dict[str, Any]
) -> None:
    """
    Inject ``thumbnail_pikzels_persona_id`` / ``thumbnailPikzelsPersonaId`` from
    ``creator_personas.profile_json`` when the user picked a default persona UUID.
    Pikzels /v2/thumbnail/* expects the **Pikzonality** id, not our internal persona row id.
    """
    pid = settings.get("thumbnail_default_persona_id") or settings.get("thumbnailDefaultPersonaId")
    if not pid or not str(pid).strip():
        settings.pop("thumbnail_pikzels_persona_id", None)
        settings.pop("thumbnailPikzelsPersonaId", None)
        settings.pop("thumbnail_persona_display_name", None)
        settings.pop("thumbnailPersonaDisplayName", None)
        return
    uid = str(user_id).strip()
    if not uid:
        return
    try:
        persona_uuid = uuid.UUID(str(pid).strip())
    except (ValueError, TypeError, AttributeError):
        return
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    cp.name        AS persona_name,
                    cp.profile_json,
                    (
                        SELECT pua.pikzels_pikzonality_id::text
                        FROM pikzels_user_assets pua
                        WHERE pua.user_id = cp.user_id
                          AND pua.local_persona_id = cp.id
                          AND pua.kind = 'persona'
                          AND pua.status = 'linked'
                        ORDER BY pua.updated_at DESC NULLS LAST, pua.created_at DESC
                        LIMIT 1
                    ) AS linked_pikzels_id
                FROM creator_personas cp
                WHERE cp.id = $1::uuid AND cp.user_id = $2::uuid
                """,
                persona_uuid,
                uid,
            )
    except asyncpg.exceptions.UndefinedTableError:
        return
    except Exception as e:
        logger.debug("merge_pikzels_thumbnail_persona_id: %s", e)
        return
    if not row:
        return
    pkz = str(row["linked_pikzels_id"] or "").strip()
    prof = row["profile_json"]
    if isinstance(prof, str):
        try:
            prof = json.loads(prof)
        except Exception:
            prof = {}
    if not isinstance(prof, dict):
        prof = {}
    if not pkz:
        pkz = str(prof.get("pikzels_pikzonality_id") or "").strip()
    if pkz:
        settings["thumbnail_pikzels_persona_id"] = pkz
        settings["thumbnailPikzelsPersonaId"] = pkz
    else:
        settings.pop("thumbnail_pikzels_persona_id", None)
        settings.pop("thumbnailPikzelsPersonaId", None)

    # Persist the persona's display name into settings so downstream stages
    # (notably ``services.hydration_enforcer._fallback_anchor_from_ctx``) can
    # inject the creator's brand into the published copy when the evidence
    # pool is empty. Without this the fallback reads as a bare filename which
    # honors the user's "my persona is set" intent only weakly.
    persona_name = str(row["persona_name"] or "").strip()
    if persona_name:
        settings["thumbnail_persona_display_name"] = persona_name
        settings["thumbnailPersonaDisplayName"] = persona_name
    else:
        settings.pop("thumbnail_persona_display_name", None)
        settings.pop("thumbnailPersonaDisplayName", None)


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
    """Mark upload as cancelled.

    NOTE: Token refunds are NOT done here. The worker's ``CancelRequested``
    handler calls :func:`worker._release_tokens`, which is the single owner
    of all wallet/ledger writes for cancellations and failures. Touching the
    wallet here too would either double-decrement ``put_reserved`` (relying
    on ``GREATEST(0, …)`` to mask the bug) or skip the ledger entry entirely.
    """
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

    _ai_cols = UPLOADS_AI_GENERATED_METADATA_COLUMNS
    if ctx.ai_title:
        idx += 1
        updates.append(f"{_safe_col('ai_generated_title', _ai_cols)} = ${idx}")
        params.append(ctx.ai_title)

    if ctx.ai_caption:
        idx += 1
        updates.append(f"{_safe_col('ai_generated_caption', _ai_cols)} = ${idx}")
        params.append(ctx.ai_caption)

    if ctx.ai_hashtags:
        idx += 1
        updates.append(f"{_safe_col('ai_generated_hashtags', _ai_cols)} = ${idx}")
        params.append(ctx.ai_hashtags)

    artifact_patch: Dict[str, Any] = {}
    if getattr(ctx, "m8_platform_titles", None):
        artifact_patch["m8_platform_titles"] = ctx.m8_platform_titles
        ctx.output_artifacts["m8_platform_titles"] = json.dumps(ctx.m8_platform_titles)
    if getattr(ctx, "m8_platform_captions", None):
        artifact_patch["m8_platform_captions"] = ctx.m8_platform_captions
        ctx.output_artifacts["m8_platform_captions"] = json.dumps(ctx.m8_platform_captions)
    if getattr(ctx, "m8_platform_hashtags", None):
        artifact_patch["m8_platform_hashtags"] = ctx.m8_platform_hashtags
        ctx.output_artifacts["m8_platform_hashtags"] = json.dumps(ctx.m8_platform_hashtags)

    # Diagnostic artifacts that other stages stash on ``ctx.output_artifacts`` but
    # never persist on their own. Without these the operator cannot answer
    # "why is hydration not rewriting" or "did Pikzels apply the persona" after
    # the fact. They are small (sub-KB) JSON blobs so always carrying them on
    # the same UPDATE is essentially free.
    _diag_keys = (
        "hydration_report",
        "hydration_payload",
        "studio_render_report",
        "thumbnail_brief_json",
        "thumbnail_render_method",
        "thumbnail_selection_method",
        "thumbnail_category",
        "thumbnail_trace",
        "platform_thumbnail_map",
        "platform_thumbnail_r2_keys",
        "thumbnail_r2_candidates",
        "pikzels_prompt_by_platform",
        "provider_error_trace",
        "ai_pipeline_trace_v1",
    )
    in_mem = getattr(ctx, "output_artifacts", None) or {}
    if isinstance(in_mem, dict):
        for k in _diag_keys:
            v = in_mem.get(k)
            if v is None:
                continue
            artifact_patch[k] = v

    if artifact_patch:
        idx += 1
        updates.append(f"output_artifacts = COALESCE(output_artifacts, '{{}}'::jsonb) || ${idx}::jsonb")
        params.append(json.dumps(artifact_patch, default=str))

    if not updates:
        return

    metadata_updates = [u for u in updates if not u.startswith("output_artifacts")]
    if metadata_updates:
        assert_set_fragments_columns(metadata_updates, UPLOADS_AI_GENERATED_METADATA_COLUMNS)

    async with pool.acquire() as conn:
        try:
            await conn.execute(
                f"UPDATE uploads SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1",
                *params,
            )
            if ctx.ai_title and is_placeholder_upload_title(ctx.title or "", ctx.filename or ""):
                await conn.execute(
                    """
                    UPDATE uploads
                       SET title = $2,
                           updated_at = NOW()
                     WHERE id = $1
                       AND (
                            title IS NULL
                         OR btrim(title) = ''
                         OR lower(btrim(title)) IN (
                              'video', 'my video', 'new video', 'untitled',
                              'untitled video', 'upload', 'uploadm8 video',
                              'open road', 'open road adventure',
                              'road trip', 'road trip vibes'
                            )
                         OR btrim(title) = $3
                       )
                    """,
                    ctx.upload_id,
                    ctx.ai_title,
                    (ctx.title or "").strip(),
                )
                ctx.title = ctx.ai_title
            if ctx.ai_caption and is_placeholder_upload_caption(ctx.caption or ""):
                await conn.execute(
                    """
                    UPDATE uploads
                       SET caption = $2,
                           updated_at = NOW()
                     WHERE id = $1
                       AND (
                            caption IS NULL
                         OR btrim(caption) = ''
                         OR lower(btrim(caption)) IN (
                              'video', 'my video', 'new video', 'untitled',
                              'untitled video', 'upload', 'uploadm8 video',
                              'check this out', 'watch this', 'new upload',
                              'new clip', 'road trip vibes', 'travel vibes',
                              'scenic drive', 'open road vibes'
                            )
                         OR btrim(caption) = $3
                       )
                    """,
                    ctx.upload_id,
                    ctx.ai_caption,
                    (ctx.caption or "").strip(),
                )
                ctx.caption = ctx.ai_caption
        except asyncpg.exceptions.UndefinedColumnError as e:
            if artifact_patch and metadata_updates:
                try:
                    meta_params = [ctx.upload_id]
                    meta_updates: List[str] = []
                    for val, col in (
                        (ctx.ai_title, "ai_generated_title"),
                        (ctx.ai_caption, "ai_generated_caption"),
                        (ctx.ai_hashtags, "ai_generated_hashtags"),
                    ):
                        if val:
                            meta_params.append(val)
                            meta_updates.append(
                                f"{_safe_col(col, _ai_cols)} = ${len(meta_params)}"
                            )
                    if meta_updates:
                        await conn.execute(
                            f"UPDATE uploads SET {', '.join(meta_updates)}, updated_at = NOW() WHERE id = $1",
                            *meta_params,
                        )
                        logger.warning(
                            "save_generated_metadata stored AI columns but skipped output_artifacts (column missing): %s",
                            e,
                        )
                        return
                except asyncpg.exceptions.UndefinedColumnError:
                    pass
            logger.warning(f"save_generated_metadata skipped (column missing): {e}")


async def save_trill_metadata(pool: asyncpg.Pool, ctx: JobContext) -> None:
    """Persist Trill/telemetry evidence so completed uploads visibly prove drive analysis ran."""
    trill = getattr(ctx, "trill_score", None) or getattr(ctx, "trill", None)
    telemetry = getattr(ctx, "telemetry_data", None) or getattr(ctx, "telemetry", None)
    if not trill and not telemetry:
        return

    try:
        from stages.pipeline_checkpoint import telemetry_to_dict, trill_to_dict
    except Exception:  # pragma: no cover - defensive import guard
        telemetry_to_dict = lambda t: None  # type: ignore
        trill_to_dict = lambda t: None  # type: ignore

    score = None
    bucket = None
    if trill:
        try:
            score = float(getattr(trill, "score", 0) or 0)
        except (TypeError, ValueError):
            score = None
        bucket = getattr(trill, "bucket", None)

    patch = {
        "trill": trill_to_dict(trill),
        "telemetry": telemetry_to_dict(telemetry),
    }

    async with pool.acquire() as conn:
        if getattr(ctx, "vehicle_make_id", None) or getattr(ctx, "vehicle_model_id", None):
            try:
                from services.vehicle_catalog import fetch_vehicle_labels

                lab = await fetch_vehicle_labels(
                    conn, ctx.vehicle_make_id, ctx.vehicle_model_id
                )
                patch["vehicle"] = {
                    "make_id": ctx.vehicle_make_id,
                    "model_id": ctx.vehicle_model_id,
                    "make_name": getattr(ctx, "vehicle_make_name", None) or lab.get("make_name"),
                    "model_name": getattr(ctx, "vehicle_model_name", None) or lab.get("model_name"),
                }
            except Exception:
                patch["vehicle"] = {
                    "make_id": getattr(ctx, "vehicle_make_id", None),
                    "model_id": getattr(ctx, "vehicle_model_id", None),
                }
        patch = {k: v for k, v in patch.items() if v is not None}
        try:
            await conn.execute(
                """
                UPDATE uploads
                SET trill_score = COALESCE($2::numeric, trill_score),
                    speed_bucket = COALESCE($3, speed_bucket),
                    trill_metadata = COALESCE(trill_metadata, '{}'::jsonb) || $4::jsonb,
                    output_artifacts = COALESCE(output_artifacts, '{}'::jsonb) || $5::jsonb,
                    updated_at = NOW()
                WHERE id = $1
                """,
                ctx.upload_id,
                score,
                bucket,
                json.dumps(patch),
                json.dumps({"trill": patch.get("trill")}),
            )
        except asyncpg.exceptions.UndefinedColumnError:
            # Older DBs may not have the Trill columns yet; still keep a debug
            # breadcrumb in output_artifacts when that column exists.
            try:
                await conn.execute(
                    """
                    UPDATE uploads
                    SET output_artifacts = COALESCE(output_artifacts, '{}'::jsonb) || $2::jsonb,
                        updated_at = NOW()
                    WHERE id = $1
                    """,
                    ctx.upload_id,
                    json.dumps({"trill": patch.get("trill")}),
                )
            except Exception:
                logger.debug("save_trill_metadata fallback skipped", exc_info=True)
        except Exception as e:
            logger.debug("save_trill_metadata skipped: %s", e)


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


async def merge_job_output_artifacts_strings(pool: asyncpg.Pool, upload_id: str, artifacts: Dict[str, str]) -> None:
    """Merge string-keyed worker artifacts into uploads.output_artifacts (jsonb)."""
    if not artifacts:
        return
    try:
        from stages import pipeline_checkpoint as _pchk

        await _pchk.merge_output_artifacts_patch(pool, upload_id, dict(artifacts))
    except Exception as e:
        logger.debug(f"merge_job_output_artifacts_strings: {e}")


async def fetch_m8_historical_signals(
    pool: asyncpg.Pool,
    user_id: str,
    category: str,
    platforms: List[str],
) -> Dict[str, Any]:
    """
    Pattern snippets (caption memory) + engagement priors (upload_quality_scores_daily)
    for M8 prompt conditioning.
    """
    pattern_corpus: List[Dict[str, Any]] = []
    pri_top: List[Dict[str, Any]] = []
    plats = [str(p).strip().lower() for p in (platforms or []) if str(p).strip()]
    if not plats:
        plats = ["tiktok"]
    lookback_days = 120

    try:
        async with pool.acquire() as conn:
            try:
                # Phase #8: filter pattern_corpus by hydration_score so the
                # M8 prompt's "imitate past uploads" memory only references
                # captions that were grounded in real evidence. Joins are
                # LEFT so older uploads without a recognition summary still
                # contribute (with a 0 score, last in the ranking) to keep
                # the corpus warm for new users.
                mem_rows = await conn.fetch(
                    """
                    SELECT m.platforms,
                           m.ai_caption,
                           m.caption_style,
                           m.caption_tone,
                           m.caption_voice,
                           COALESCE(rs.hydration_score, 0) AS hydration_score
                      FROM upload_caption_memory m
                      LEFT JOIN upload_recognition_summary rs
                        ON rs.upload_id = m.upload_id
                     WHERE m.user_id = $1::uuid AND m.category = $2
                     ORDER BY COALESCE(rs.hydration_score, 0) DESC, m.created_at DESC
                     LIMIT 12
                    """,
                    user_id,
                    (category or "general").lower(),
                )
                # If we have any hydrated rows (>=0.4), drop unhydrated ones
                # entirely — the LLM should not learn from generic past copy.
                # If everything is unhydrated (cold start), keep top 8 anyway
                # so we don't ship an empty corpus.
                rows_l = list(mem_rows or [])
                hydrated = [r for r in rows_l if float(r["hydration_score"] or 0) >= 0.4]
                pool_rows = hydrated if len(hydrated) >= 4 else rows_l
                for r in pool_rows[:8]:
                    plat = ""
                    rawp = r["platforms"]
                    if isinstance(rawp, str):
                        try:
                            rawp = json.loads(rawp)
                        except Exception:
                            rawp = []
                    if isinstance(rawp, list) and rawp:
                        plat = str(rawp[0]).lower().strip()
                    snip = (r["ai_caption"] or "").strip().replace("\n", " ")
                    if len(snip) < 12:
                        continue
                    pattern_corpus.append(
                        {
                            "platform": plat or "unknown",
                            "snippet": snip[:400],
                            "caption_style": (r["caption_style"] or "") or "",
                            "caption_tone": (r["caption_tone"] or "") or "",
                            "caption_voice": (r["caption_voice"] or "") or "",
                            "hydration_score": float(r["hydration_score"] or 0),
                        }
                    )
            except asyncpg.exceptions.UndefinedTableError:
                pass
            except Exception as e:
                logger.debug(f"fetch_m8_historical_signals memory: {e}")

            try:
                pri_rows = await conn.fetch(
                    """
                    WITH agg AS (
                        SELECT strategy_key,
                               SUM(samples)::bigint AS samples,
                               CASE WHEN SUM(samples) > 0 THEN
                                 SUM(mean_engagement * samples::double precision) / SUM(samples::double precision)
                               ELSE 0.0 END AS weighted_mean_engagement,
                               MAX(ci95_high)::double precision AS max_ci95_high
                          FROM upload_quality_scores_daily
                         WHERE user_id = $1::uuid
                           AND day >= (CURRENT_DATE - $3::int)
                           AND (platform = 'all' OR platform = ANY($2::text[]))
                         GROUP BY strategy_key
                        HAVING SUM(samples) >= 2
                    )
                    SELECT * FROM agg
                    ORDER BY weighted_mean_engagement DESC NULLS LAST
                    LIMIT 8
                    """,
                    user_id,
                    plats,
                    lookback_days,
                )
                for r in pri_rows or []:
                    pri_top.append(
                        {
                            "strategy_key": str(r["strategy_key"] or ""),
                            "samples": int(r["samples"] or 0),
                            "weighted_mean_engagement": float(r["weighted_mean_engagement"] or 0.0),
                            "max_ci95_high": float(r["max_ci95_high"] or 0.0),
                        }
                    )
            except asyncpg.exceptions.UndefinedTableError:
                pass
            except Exception as e:
                logger.debug(f"fetch_m8_historical_signals priors: {e}")

    except Exception as e:
        logger.debug(f"fetch_m8_historical_signals: {e}")

    visual_recall: Dict[str, List[str]] = {}
    try:
        from services.visual_entity_memory import fetch_user_entity_recall

        visual_recall = await fetch_user_entity_recall(
            pool,
            user_id=user_id,
            category=(category or "general").lower(),
            limit_per_bucket=8,
        )
    except Exception as e:
        logger.debug(f"fetch_m8_historical_signals visual_recall: {e}")

    return {
        "__pattern_corpus__": pattern_corpus,
        "__strategy_priors__": {"top": pri_top, "lookback_days": lookback_days},
        "__visual_entity_recall__": visual_recall,
        "__meta__": {
            "ok": True,
            "pattern_n": len(pattern_corpus),
            "prior_n": len(pri_top),
            "visual_recall_buckets": sum(1 for v in (visual_recall or {}).values() if v),
        },
    }


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


async def touch_platform_token_last_used(pool: asyncpg.Pool, token_row_id: str) -> None:
    """Record that a platform_tokens row was used for publishing."""
    if not token_row_id:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE platform_tokens
                SET last_used_at = NOW(), updated_at = NOW()
                WHERE id = $1 AND revoked_at IS NULL
                """,
                str(token_row_id),
            )
    except Exception as e:
        logger.debug("touch_platform_token_last_used failed for %s: %s", token_row_id, e)


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
    """Load admin Discord webhook URL from admin_settings.settings_json (notifications.admin_webhook_url)."""
    try:
        async with pool.acquire() as conn:
            val = await conn.fetchval(
                "SELECT settings_json->'notifications'->>'admin_webhook_url' FROM admin_settings WHERE id = 1"
            )
            return str(val).strip() if val else None
    except Exception:
        return None


# ── Free-tier video watermark (drawtext) ───────────────────────────────────
DEFAULT_WATERMARK_BURN_TEXT = "Upload M8"


def sanitize_watermark_burn_text(raw: Any) -> str:
    """Single-line text safe for FFmpeg drawtext; used by admin API + worker."""
    s = str(raw or "").strip()
    if not s:
        return DEFAULT_WATERMARK_BURN_TEXT
    s = " ".join(s.split())
    if len(s) > 80:
        s = s[:80].rstrip()
    return s or DEFAULT_WATERMARK_BURN_TEXT


async def load_watermark_burn_text(pool: asyncpg.Pool) -> str:
    """Load master-admin watermark string from admin_settings (row id=1)."""
    try:
        async with pool.acquire() as conn:
            val = await conn.fetchval(
                "SELECT settings_json->>'watermark_burn_text' FROM admin_settings WHERE id = 1"
            )
        return sanitize_watermark_burn_text(val)
    except Exception:
        return DEFAULT_WATERMARK_BURN_TEXT


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
    token_row_id: Optional[str] = None,
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
        # Same encryption contract as the API: require TOKEN_ENC_KEYS; never persist plaintext.
        from core.auth import decrypt_blob, encrypt_blob, init_enc_keys

        init_enc_keys()

        async with pool.acquire() as conn:
            # ── Try platform_tokens ───────────────────────────────────────
            for raw_table in OAUTH_TOKEN_STORAGE_TABLES_ORDERED:
                table = assert_relation_name(raw_table, OAUTH_TOKEN_STORAGE_TABLES)
                try:
                    if token_row_id:
                        row = await conn.fetchrow(
                            f"SELECT id, token_blob FROM {table} "
                            f"WHERE id = $1::uuid AND user_id = $2::uuid AND platform = $3",
                            token_row_id,
                            user_id,
                            platform,
                        )
                    else:
                        row = await conn.fetchrow(
                            f"SELECT id, token_blob FROM {table} "
                            f"WHERE user_id = $1 AND platform = $2 "
                            f"ORDER BY updated_at DESC LIMIT 1",
                            user_id,
                            platform,
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
                        current_plain = decrypt_blob(current_encrypted) or {}
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
                new_blob_str = json.dumps(encrypt_blob(current_plain))

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

            if token_row_id:
                logger.warning(
                    f"save_refreshed_token: no row id={token_row_id} for {platform} user={user_id}"
                )
            else:
                logger.warning(f"save_refreshed_token: no row found for {platform} user={user_id}")

    except Exception as e:
        logger.warning(f"save_refreshed_token failed (non-fatal) for {platform} user={user_id}: {e}")
