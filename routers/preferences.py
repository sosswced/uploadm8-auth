"""
UploadM8 User Preferences routes -- extracted from app.py.

Handles color preferences and content/upload preferences (hashtags, captions,
thumbnails, Trill settings, etc.).
"""

import json
import logging
import os
from fastapi import APIRouter, Body, Depends, HTTPException, Query

import core.state
from core.deps import get_current_user, get_current_user_readonly
from core.helpers import (
    _now_utc,
    _safe_json,
    _safe_col,
    coerce_jsonb_dict,
    coerce_jsonb_list,
    merge_platform_hashtag_overlay,
)
from core.sql_allowlist import USER_COLOR_PREFERENCES_UPDATE_COLUMNS, assert_set_fragments_columns
from core.models import ColorPreferencesUpdate, UserPreferencesUpdate
from services.user_preferences_persist import save_user_content_preferences
from services.thumbnail_personas_list import list_thumbnail_studio_personas
from stages.entitlements import get_entitlements_for_tier
from services.ml_hub_config import get_ml_hub_urls, ml_hub_huggingface_dict
from core.upload_baseline_defaults import UPLOAD_PREF_STRIP_KEYS, apply_upload_baseline_defaults
from stages.tiktok_cover_burn import default_tiktok_burn_styled_cover_pref

logger = logging.getLogger(__name__)

router = APIRouter(tags=["preferences"])

# Audit / metadata columns on user_preferences that asyncpg returns as native
# datetime/UUID types. Workers do not need them in the job payload snapshot.
_UPLOAD_PREF_STRIP_KEYS = UPLOAD_PREF_STRIP_KEYS


# ============================================================
# User Color Preferences
# ============================================================
@router.get("/api/colors")
async def get_color_preferences(user: dict = Depends(get_current_user)):
    """Get user's custom color preferences for platforms"""
    async with core.state.db_pool.acquire() as conn:
        prefs = await conn.fetchrow("""
            SELECT
                tiktok_color, youtube_color, instagram_color,
                facebook_color, accent_color
            FROM user_color_preferences
            WHERE user_id = $1
        """, user["id"])

        if not prefs:
            # Return defaults
            return {
                "tiktok_color": "#000000",
                "youtube_color": "#FF0000",
                "instagram_color": "#E4405F",
                "facebook_color": "#1877F2",
                "accent_color": "#3B82F6"
            }

    return {
        "tiktok_color": prefs["tiktok_color"],
        "youtube_color": prefs["youtube_color"],
        "instagram_color": prefs["instagram_color"],
        "facebook_color": prefs["facebook_color"],
        "accent_color": prefs["accent_color"]
    }

@router.put("/api/colors")
async def update_color_preferences(
    colors: ColorPreferencesUpdate,
    user: dict = Depends(get_current_user)
):
    """Update user's custom color preferences"""
    async with core.state.db_pool.acquire() as conn:
        # Check if preferences exist
        exists = await conn.fetchval(
            "SELECT 1 FROM user_color_preferences WHERE user_id = $1",
            user["id"]
        )

        if not exists:
            # Create default preferences
            await conn.execute("""
                INSERT INTO user_color_preferences (user_id)
                VALUES ($1)
            """, user["id"])

        # Build update query
        _COLOR_COLS = USER_COLOR_PREFERENCES_UPDATE_COLUMNS
        updates = []
        params = [user["id"]]
        param_count = 1

        if colors.tiktok_color is not None:
            param_count += 1
            updates.append(f"{_safe_col('tiktok_color', _COLOR_COLS)} = ${param_count}")
            params.append(colors.tiktok_color)

        if colors.youtube_color is not None:
            param_count += 1
            updates.append(f"{_safe_col('youtube_color', _COLOR_COLS)} = ${param_count}")
            params.append(colors.youtube_color)

        if colors.instagram_color is not None:
            param_count += 1
            updates.append(f"{_safe_col('instagram_color', _COLOR_COLS)} = ${param_count}")
            params.append(colors.instagram_color)

        if colors.facebook_color is not None:
            param_count += 1
            updates.append(f"{_safe_col('facebook_color', _COLOR_COLS)} = ${param_count}")
            params.append(colors.facebook_color)

        if colors.accent_color is not None:
            param_count += 1
            updates.append(f"{_safe_col('accent_color', _COLOR_COLS)} = ${param_count}")
            params.append(colors.accent_color)

        if not updates:
            raise HTTPException(400, "No color updates provided")

        # Always update updated_at
        param_count += 1
        updates.append(f"{_safe_col('updated_at', _COLOR_COLS)} = ${param_count}")
        params.append(_now_utc())

        assert_set_fragments_columns(updates, USER_COLOR_PREFERENCES_UPDATE_COLUMNS)

        query = f"""
            UPDATE user_color_preferences
            SET {', '.join(updates)}
            WHERE user_id = $1
        """

        await conn.execute(query, *params)

    return {"status": "updated"}


# ============================================================
# User Content Preferences — helpers
# ============================================================

def _parse_users_preferences(raw) -> dict:
    """Parse users.preferences JSONB into a dict."""
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            return json.loads(raw) or {}
        except Exception:
            return {}
    return dict(raw) if hasattr(raw, "keys") else {}


def _overlay_users_prefs_on_result(result: dict, up: dict) -> None:
    """Overlay users.preferences onto result. users.preferences wins for all fields."""
    if not isinstance(up, dict) or not up:
        return
    # Hashtag fields
    for camel, snake, key in [
        ("alwaysHashtags", "always_hashtags", "always_hashtags"),
        ("blockedHashtags", "blocked_hashtags", "blocked_hashtags"),
        ("platformHashtags", "platform_hashtags", "platform_hashtags"),
    ]:
        v = up[camel] if camel in up else up.get(snake) if snake in up else None
        if v is None:
            continue
        # Coerce JSON-string payloads back to real Python types. users.preferences
        # is a JSONB blob; nested fields could be saved as strings by old client
        # code paths, and `get_effective_hashtags` expects list/dict.
        if key == "platform_hashtags":
            v = coerce_jsonb_dict(v, default={})
            # Do not wipe rich user_preferences.platform_hashtags with a bare {}.
            if isinstance(v, dict):
                base = result.get("platformHashtags") or result.get("platform_hashtags") or {}
                merged = merge_platform_hashtag_overlay(base, v)
                result["platformHashtags"] = result["platform_hashtags"] = merged
                continue
        else:
            v = coerce_jsonb_list(v)
        result[camel] = result[key] = v
    # Scalar prefs from users.preferences
    for camel, snake in [
        ("maxHashtags", "max_hashtags"), ("aiHashtagCount", "ai_hashtag_count"),
        ("aiHashtagsEnabled", "ai_hashtags_enabled"), ("captionStyle", "caption_style"),
        ("captionTone", "caption_tone"), ("captionVoice", "caption_voice"),
        ("captionFrameCount", "caption_frame_count"), ("aiHashtagStyle", "ai_hashtag_style"),
        ("hashtagPosition", "hashtag_position"), ("autoCaptions", "auto_captions"),
        ("autoThumbnails", "auto_thumbnails"), ("styledThumbnails", "styled_thumbnails"),
        ("defaultPrivacy", "default_privacy"), ("thumbnailInterval", "thumbnail_interval"),
    ]:
        v = up.get(camel) if up.get(camel) is not None else up.get(snake)
        if v is not None:
            result[camel] = result[snake] = v

    _overlay_upload_ai_audio_studio_prefs(result, up)


def _hydrate_snake_camel_mirror(result: dict) -> None:
    """Ensure camelCase + snake_case aliases exist for billing/worker merge paths."""
    pairs = [
        ("auto_captions", "autoCaptions"),
        ("auto_thumbnails", "autoThumbnails"),
        ("styled_thumbnails", "styledThumbnails"),
        ("thumbnail_interval", "thumbnailInterval"),
        ("default_privacy", "defaultPrivacy"),
        ("ai_hashtags_enabled", "aiHashtagsEnabled"),
        ("ai_hashtag_count", "aiHashtagCount"),
        ("ai_hashtag_style", "aiHashtagStyle"),
        ("hashtag_position", "hashtagPosition"),
        ("max_hashtags", "maxHashtags"),
        ("email_notifications", "emailNotifications"),
        ("discord_webhook", "discordWebhook"),
        ("trill_enabled", "trillEnabled"),
        ("trill_min_score", "trillMinScore"),
        ("trill_ai_enhance", "trillAiEnhance"),
        ("trill_openai_model", "trillOpenaiModel"),
        ("use_audio_context", "useAudioContext"),
        ("youtube_shorts_copyright_trim", "youtubeShortsCopyrightTrim"),
        ("audio_transcription", "audioTranscription"),
        ("ai_service_telemetry", "aiServiceTelemetry"),
        ("ai_service_dashcam_osd", "aiServiceDashcamOSD"),
        ("ai_service_audio_signals", "aiServiceAudioSignals"),
        ("ai_service_music_detection", "aiServiceMusicDetection"),
        ("ai_service_audio_summary", "aiServiceAudioSummary"),
        ("ai_service_emotion_signals", "aiServiceEmotionSignals"),
        ("ai_service_caption_writer", "aiServiceCaptionWriter"),
        ("ai_service_thumbnail_designer", "aiServiceThumbnailDesigner"),
        ("ai_service_speech_to_text", "aiServiceSpeechToText"),
        ("ai_service_scene_understanding", "aiServiceSceneUnderstanding"),
        ("thumbnail_studio_enabled", "thumbnailStudioEnabled"),
        ("thumbnail_studio_engine_enabled", "thumbnailStudioEngineEnabled"),
        ("thumbnail_persona_enabled", "thumbnailPersonaEnabled"),
    ]
    for snake, camel in pairs:
        if snake in result and result[snake] is not None:
            result.setdefault(camel, result[snake])
        elif camel in result and result[camel] is not None:
            result.setdefault(snake, result[camel])


def _overlay_upload_ai_audio_studio_prefs(result: dict, up: dict) -> None:
    """Merge audio, per-service AI toggles, and thumbnail-studio keys from users.preferences."""
    if not isinstance(up, dict) or not up:
        return

    def _pair_bool(camel: str, snake: str) -> None:
        v = up.get(camel) if up.get(camel) is not None else up.get(snake)
        if v is None:
            return
        if isinstance(v, str):
            b = v.lower() not in ("false", "0", "no", "off", "")
        else:
            b = bool(v)
        result[camel] = result[snake] = b

    for camel, snake in [
        ("useAudioContext", "use_audio_context"),
        ("youtubeShortsCopyrightTrim", "youtube_shorts_copyright_trim"),
        ("audioTranscription", "audio_transcription"),
        ("aiServiceTelemetry", "ai_service_telemetry"),
        ("aiServiceDashcamOSD", "ai_service_dashcam_osd"),
        ("aiServiceAudioSignals", "ai_service_audio_signals"),
        ("aiServiceMusicDetection", "ai_service_music_detection"),
        ("aiServiceAudioSummary", "ai_service_audio_summary"),
        ("aiServiceEmotionSignals", "ai_service_emotion_signals"),
        ("aiServiceCaptionWriter", "ai_service_caption_writer"),
        ("aiServiceThumbnailDesigner", "ai_service_thumbnail_designer"),
        ("aiServiceSpeechToText", "ai_service_speech_to_text"),
        ("aiServiceSceneUnderstanding", "ai_service_scene_understanding"),
        ("thumbnailStudioEnabled", "thumbnail_studio_enabled"),
        ("thumbnailStudioEngineEnabled", "thumbnail_studio_engine_enabled"),
        ("thumbnailPersonaEnabled", "thumbnail_persona_enabled"),
    ]:
        _pair_bool(camel, snake)

    for camel, snake in [
        ("thumbnailDefaultPersonaId", "thumbnail_default_persona_id"),
        ("thumbnailPersonaStrength", "thumbnail_persona_strength"),
    ]:
        v = up.get(camel) if up.get(camel) is not None else up.get(snake)
        if v is not None:
            result[camel] = result[snake] = v

    def _pair_choice(
        camel: str, snake: str, default: str, allowed: frozenset
    ) -> None:
        v = up.get(camel) if up.get(camel) is not None else up.get(snake)
        if v is None:
            return
        s = str(v).strip().lower()
        if s not in allowed:
            s = default
        result[camel] = result[snake] = s

    _pair_choice(
        "thumbnailSelectionMode",
        "thumbnail_selection_mode",
        "ai",
        frozenset(("ai", "sharpness")),
    )
    _pair_choice(
        "thumbnailRenderPipeline",
        "thumbnail_render_pipeline",
        "auto",
        frozenset(("auto", "studio_renderer", "ai_edit", "template", "none")),
    )

    # Legacy users.preferences key (thumbnailPikzelsEnabled) → engine toggle
    if (
        up.get("thumbnailStudioEngineEnabled") is None
        and up.get("thumbnail_studio_engine_enabled") is None
    ):
        leg = up.get("thumbnailPikzelsEnabled")
        if leg is None:
            leg = up.get("thumbnail_pikzels_enabled")
        if leg is not None:
            if isinstance(leg, str):
                eb = leg.lower() not in ("false", "0", "no", "off", "")
            else:
                eb = bool(leg)
            result["thumbnailStudioEngineEnabled"] = result["thumbnail_studio_engine_enabled"] = eb


async def get_user_prefs_for_upload(conn, user_id: int) -> dict:
    """Helper to fetch user preferences for upload processing.
    Merges user_preferences table + users.preferences + user_settings so:
    - PUT /api/me/preferences (users.preferences) wins for hashtag/caption fields
    - POST /api/settings/preferences (user_preferences) provides defaults
    """
    result = {}
    # Read from user_preferences table
    prefs_row = await conn.fetchrow(
        "SELECT * FROM user_preferences WHERE user_id = $1",
        user_id
    )
    if prefs_row:
        pr = dict(prefs_row)
        for _k in _UPLOAD_PREF_STRIP_KEYS:
            pr.pop(_k, None)
        result.update(pr)
        # Defensive: JSONB columns can come back as JSON-encoded strings if a
        # previous write double-encoded the value or if the connection pool is
        # missing the JSONB codec. Coerce them to real Python types here so the
        # malformed `'["tester","qwe"]'` string never reaches the worker.
        result["always_hashtags"] = coerce_jsonb_list(result.get("always_hashtags"))
        result["blocked_hashtags"] = coerce_jsonb_list(result.get("blocked_hashtags"))
        result["platform_hashtags"] = coerce_jsonb_dict(
            result.get("platform_hashtags"),
            default={"tiktok": [], "youtube": [], "instagram": [], "facebook": []},
        ) or {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}
        styled = bool(result.get("styled_thumbnails", True))
        result["styled_thumbnails"] = styled
        result["styledThumbnails"] = styled
        _hydrate_snake_camel_mirror(result)

    # Overlay users.preferences (PUT /api/me/preferences writes here) -- full overlay
    users_prefs_row = await conn.fetchrow("SELECT preferences FROM users WHERE id = $1", user_id)
    up = _parse_users_preferences(users_prefs_row["preferences"] if users_prefs_row else None)
    _overlay_users_prefs_on_result(result, up)

    if result:
        _hydrate_snake_camel_mirror(result)
        apply_upload_baseline_defaults(result)
        return result

    # Fallback: Try legacy JSONB locations
    # Try user_settings.preferences_json first
    prefs_row = await conn.fetchrow(
        "SELECT preferences_json FROM user_settings WHERE user_id = $1",
        user_id
    )

    if prefs_row and prefs_row["preferences_json"]:
        # `coerce_jsonb_dict` peels both single- and double-encoded JSON strings.
        prefs = coerce_jsonb_dict(prefs_row["preferences_json"], default={})
    else:
        # Try users.preferences (oldest fallback)
        prefs_row = await conn.fetchrow(
            "SELECT preferences FROM users WHERE id = $1",
            user_id
        )
        if prefs_row and prefs_row["preferences"]:
            prefs = coerce_jsonb_dict(prefs_row["preferences"], default={})
        else:
            prefs = {}

    # Return preferences with defaults (convert camelCase to snake_case for internal use)
    styled = prefs.get("styledThumbnails", prefs.get("styled_thumbnails", True))
    out = {
        "auto_captions": prefs.get("autoCaptions", False),
        "auto_thumbnails": prefs.get("autoThumbnails", prefs.get("auto_thumbnails", True)),
        "styled_thumbnails": styled,
        "styledThumbnails": styled,
        "thumbnail_interval": prefs.get("thumbnailInterval", 5),
        "default_privacy": prefs.get("defaultPrivacy", "public"),
        "ai_hashtags_enabled": prefs.get("aiHashtagsEnabled", False),
        "ai_hashtag_count": prefs.get("aiHashtagCount", 5),
        "ai_hashtag_style": prefs.get("aiHashtagStyle", "mixed"),
        "hashtag_position": prefs.get("hashtagPosition", "end"),
        "max_hashtags": prefs.get("maxHashtags", 30),
        # `coerce_jsonb_*` peels JSON-string payloads. Critical for legacy
        # rows where alwaysHashtags / platformHashtags were stored as nested
        # JSON-encoded strings inside the preferences blob.
        "always_hashtags": coerce_jsonb_list(prefs.get("alwaysHashtags") or prefs.get("always_hashtags")),
        "blocked_hashtags": coerce_jsonb_list(prefs.get("blockedHashtags") or prefs.get("blocked_hashtags")),
        "platform_hashtags": coerce_jsonb_dict(
            prefs.get("platformHashtags") or prefs.get("platform_hashtags"),
            default={"tiktok": [], "youtube": [], "instagram": [], "facebook": []},
        ) or {"tiktok": [], "youtube": [], "instagram": [], "facebook": []},
        "email_notifications": prefs.get("emailNotifications", True),
        "discord_webhook": prefs.get("discordWebhook", None)
    }
    if isinstance(prefs, dict):
        _overlay_users_prefs_on_result(out, prefs)
    _hydrate_snake_camel_mirror(out)
    apply_upload_baseline_defaults(out)
    return out


# ============================================================
# GET /api/settings/preferences
# ============================================================
@router.get("/api/settings/preferences")
async def get_user_preferences(
    include_personas: bool = Query(False, description="Include thumbnail studio personas (slow; settings only)"),
    user: dict = Depends(get_current_user_readonly),
):
    """GET user content preferences - used by settings page AND upload workflow"""
    try:
        async with core.state.db_pool.acquire() as conn:
            try:
                prefs = await conn.fetchrow(
                    "SELECT * FROM user_preferences WHERE user_id = $1",
                    user["id"]
                )
            except Exception:
                prefs = None  # fall through to INSERT-on-demand

            if not prefs:
                await conn.execute("INSERT INTO user_preferences (user_id) VALUES ($1)", user["id"])
                prefs = await conn.fetchrow(
                    "SELECT * FROM user_preferences WHERE user_id = $1",
                    user["id"]
                )

            d = dict(prefs) if prefs else {}

            # Ensure arrays are properly formatted as lists
            always_tags = d.get("always_hashtags")
            blocked_tags = d.get("blocked_hashtags")
            platform_tags = d.get("platform_hashtags")

            # Defensive parse: JSONB might come back as a single- or double-encoded
            # string when a write path used `json.dumps()` on top of the asyncpg
            # JSONB codec. The helpers peel both layers so the malformed
            # `'["tester","qwe"]'` string is recovered as a real Python list.
            always_tags = coerce_jsonb_list(always_tags)
            blocked_tags = coerce_jsonb_list(blocked_tags)
            platform_tags = coerce_jsonb_dict(
                platform_tags,
                default={"tiktok": [], "youtube": [], "instagram": [], "facebook": []},
            )

            out = {
                "autoCaptions": d.get("auto_captions", False),
                "autoThumbnails": (
                    True if d.get("auto_thumbnails") is None else bool(d.get("auto_thumbnails"))
                ),
                "thumbnailInterval": str(d.get("thumbnail_interval", 5)),
                "defaultPrivacy": d.get("default_privacy", "public"),
                "aiHashtagsEnabled": d.get("ai_hashtags_enabled", False),
                "aiHashtagCount": str(d.get("ai_hashtag_count", 5)),
                "aiHashtagStyle": d.get("ai_hashtag_style", "mixed"),
                "hashtagPosition": d.get("hashtag_position", "end"),
                "maxHashtags": str(d.get("max_hashtags", 15)),
                "alwaysHashtags": always_tags or [],
                "blockedHashtags": blocked_tags or [],
                "platformHashtags": platform_tags or {"tiktok": [], "youtube": [], "instagram": [], "facebook": []},
                "emailNotifications": d.get("email_notifications", True),
                "discordWebhook": d.get("discord_webhook"),
                "trillEnabled": (
                    True if d.get("trill_enabled") is None else bool(d.get("trill_enabled"))
                ),
                "trillMinScore": (
                    60 if d.get("trill_min_score") is None else int(d.get("trill_min_score") or 60)
                ),
                "trillAiEnhance": (
                    True if d.get("trill_ai_enhance") is None else bool(d.get("trill_ai_enhance"))
                ),
                "trillOpenaiModel": d.get("trill_openai_model", "gpt-4o-mini"),
                "trillLeaderboardOptIn": bool(d.get("trill_leaderboard_opt_in", False)),
                "trillMapSharingOptIn": bool(d.get("trill_map_sharing_opt_in", False)),
                "styledThumbnails": d.get("styled_thumbnails", True),
                "useAudioContext": bool(d.get("use_audio_context", True)),
                "audioTranscription": bool(d.get("audio_transcription", True)),
                "authSecurityAlerts": bool(d.get("auth_security_alerts", True)),
                "digestEmails": bool(d.get("digest_emails", True)),
                "scheduledAlertEmails": bool(d.get("scheduled_alert_emails", True)),
                "aiServiceTelemetry": bool(d.get("ai_service_telemetry", True)),
                "aiServiceDashcamOSD": bool(d.get("ai_service_dashcam_osd", True)),
                "aiServiceAudioSignals": bool(d.get("ai_service_audio_signals", True)),
                "aiServiceMusicDetection": bool(d.get("ai_service_music_detection", True)),
                "aiServiceAudioSummary": bool(d.get("ai_service_audio_summary", True)),
                "aiServiceEmotionSignals": bool(d.get("ai_service_emotion_signals", False)),
                "aiServiceCaptionWriter": bool(d.get("ai_service_caption_writer", True)),
                "aiServiceThumbnailDesigner": bool(d.get("ai_service_thumbnail_designer", True)),
                "aiServiceSpeechToText": bool(d.get("ai_service_speech_to_text", True)),
                "aiServiceSceneUnderstanding": bool(d.get("ai_service_scene_understanding", True)),
                "aiServiceFrameInspector": bool(d.get("ai_service_frame_inspector", True)),
                "aiServiceVideoAnalyzer": bool(d.get("ai_service_video_analyzer", True)),
            }
            # Overlay users.preferences -- source of truth for hashtags + caption (PUT /api/me/preferences)
            users_prefs = None
            try:
                users_prefs = await conn.fetchval("SELECT preferences FROM users WHERE id = $1", user["id"])
            except Exception as col_err:
                logger.debug("users.preferences SELECT failed: %s", col_err)
            up = _parse_users_preferences(users_prefs) if users_prefs else {}
            if up:
                # Use key presence — never `a or b` here: [] and {} are falsy and would
                # drop the overlay value and fall back to defaults, wiping user_preferences.
                if "alwaysHashtags" in up or "always_hashtags" in up:
                    v = up["alwaysHashtags"] if "alwaysHashtags" in up else up["always_hashtags"]
                    out["alwaysHashtags"] = v if isinstance(v, list) else []
                if "blockedHashtags" in up or "blocked_hashtags" in up:
                    v = up["blockedHashtags"] if "blockedHashtags" in up else up["blocked_hashtags"]
                    out["blockedHashtags"] = v if isinstance(v, list) else []
                if "platformHashtags" in up or "platform_hashtags" in up:
                    v = up["platformHashtags"] if "platformHashtags" in up else up["platform_hashtags"]
                    if not isinstance(v, dict):
                        v = {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}
                    base_ph = out.get("platformHashtags") or {}
                    out["platformHashtags"] = merge_platform_hashtag_overlay(base_ph, v)
                if up.get("maxHashtags") is not None or up.get("max_hashtags") is not None:
                    out["maxHashtags"] = str(up.get("maxHashtags") or up.get("max_hashtags") or 15)
                if up.get("aiHashtagCount") is not None or up.get("ai_hashtag_count") is not None:
                    out["aiHashtagCount"] = str(up.get("aiHashtagCount") or up.get("ai_hashtag_count") or 5)
                out["captionStyle"] = up.get("captionStyle") or up.get("caption_style") or "story"
                out["captionTone"] = up.get("captionTone") or up.get("caption_tone") or "authentic"
                out["captionVoice"] = up.get("captionVoice") or up.get("caption_voice") or "default"
                out["captionFrameCount"] = up.get("captionFrameCount") or up.get("caption_frame_count") or 6
                _overlay_upload_ai_audio_studio_prefs(out, up)
            else:
                out.setdefault("captionStyle", "story")
                out.setdefault("captionTone", "authentic")
                out.setdefault("captionVoice", "default")
                out.setdefault("captionFrameCount", 6)
                out.setdefault("thumbnailSelectionMode", "sharpness")
                out.setdefault("thumbnail_selection_mode", "sharpness")
                out.setdefault("thumbnailRenderPipeline", "auto")
                out.setdefault("thumbnail_render_pipeline", "auto")
            apply_upload_baseline_defaults(out)
            tier_row = await conn.fetchrow(
                "SELECT subscription_tier FROM users WHERE id = $1", user["id"]
            )
            tier_slug = str((tier_row or {}).get("subscription_tier") or "free")
            ent = get_entitlements_for_tier(tier_slug)
            pref_explicit = (
                d.get("tiktok_burn_styled_cover") is not None
                or up.get("tiktokBurnStyledCover") is not None
                or up.get("tiktok_burn_styled_cover") is not None
            )
            if not pref_explicit:
                burn_default = default_tiktok_burn_styled_cover_pref(ent)
                out["tiktokBurnStyledCover"] = burn_default
                out["tiktok_burn_styled_cover"] = burn_default
            out["tiktokBurnStyledCoverTierDefault"] = default_tiktok_burn_styled_cover_pref(ent)
            out["tiktokBurnStyledCoverAvailable"] = tier_slug != "free"
            out["tiktokBurnStyledCoverFast"] = tier_slug not in ("free",)
            if include_personas:
                try:
                    plist = await list_thumbnail_studio_personas(conn, user["id"])
                    out["thumbnail_personas"] = out["thumbnailPersonas"] = plist
                except Exception:
                    out["thumbnail_personas"] = out["thumbnailPersonas"] = []
            else:
                out["thumbnail_personas"] = out["thumbnailPersonas"] = []
            return out
    except Exception as e:
        logger.exception("get_user_preferences failed: %s", e)
        # Return defaults so settings page loads; avoid 500 when DB schema mismatch or migration not run
        return apply_upload_baseline_defaults({
            "autoCaptions": False, "autoThumbnails": True, "thumbnailInterval": "5",
            "defaultPrivacy": "public", "aiHashtagsEnabled": False, "aiHashtagCount": "5",
            "aiHashtagStyle": "mixed", "hashtagPosition": "end", "maxHashtags": "15",
            "alwaysHashtags": [], "blockedHashtags": [],
            "platformHashtags": {"tiktok": [], "youtube": [], "instagram": [], "facebook": []},
            "emailNotifications": True, "discordWebhook": None,
            "trillEnabled": True, "trillMinScore": 60,
            "trillAiEnhance": True, "trillOpenaiModel": "gpt-4o-mini",
            "trillLeaderboardOptIn": False, "trillMapSharingOptIn": False,
            "styledThumbnails": True,
            "captionStyle": "story", "captionTone": "authentic", "captionVoice": "default", "captionFrameCount": 6,
            "thumbnailSelectionMode": "sharpness", "thumbnail_selection_mode": "sharpness",
            "thumbnailRenderPipeline": "auto", "thumbnail_render_pipeline": "auto",
            "aiServiceDashcamOSD": True, "ai_service_dashcam_osd": True,
            "thumbnail_personas": [], "thumbnailPersonas": [],
        })


# ============================================================
# POST /api/settings/preferences
# ============================================================
@router.post("/api/settings/preferences")
async def save_user_preferences(
    payload: dict = Body(...),
    user: dict = Depends(get_current_user),
):
    """
    SAVE user content preferences.

    Persists the full settings payload (including audio + AI service toggles) to
    ``user_preferences``, syncs Trill/discord into ``user_settings``, and merges
    caption fields into ``users.preferences`` for the worker.
    """
    async with core.state.db_pool.acquire() as conn:
        return await save_user_content_preferences(conn, user, payload)


# ============================================================
# PUT /api/settings/preferences (backward-compat alias)
# ============================================================
@router.put("/api/settings/preferences")
async def save_user_preferences_put(
    prefs: UserPreferencesUpdate,
    user: dict = Depends(get_current_user)
):
    """Backward-compatible alias for clients that still call PUT"""
    return await save_user_preferences(prefs.model_dump(by_alias=True), user)


# ============================================================
# GET /api/settings/channel-catalog
# ============================================================
@router.get("/api/settings/channel-catalog")
async def get_channel_visual_catalog(user: dict = Depends(get_current_user)):
    """
    What UploadM8 has learned from this creator's uploads (Google Vision + VI buckets).
    Powers Settings → "What UploadM8 knows about my channel" and HF ML export.
    """
    from services.thumbnail_niches import normalize_niche
    from services.visual_entity_memory import fetch_channel_catalog_detail

    category = "general"
    try:
        async with core.state.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT preferences FROM users WHERE id = $1::uuid",
                user["id"],
            )
            if row:
                prefs = coerce_jsonb_dict(row.get("preferences") or {})
                nested = prefs.get("thumbnailDefaultStrategy") or prefs.get("thumbnail_default_strategy")
                if isinstance(nested, dict) and nested.get("audience_niche"):
                    category = normalize_niche(str(nested["audience_niche"]))
    except Exception as e:
        logger.debug("channel-catalog niche lookup: %s", e)

    catalog = await fetch_channel_catalog_detail(
        core.state.db_pool,
        user_id=str(user["id"]),
        category=category,
        limit_per_bucket=14,
    )
    out: dict = {"catalog": catalog}
    role = str(user.get("role") or "").strip().lower()
    if role in ("admin", "master_admin"):
        hub_urls = get_ml_hub_urls()
        hf = ml_hub_huggingface_dict()
        out["ml_hub"] = {
            "dataset_repo": hub_urls.get("dataset_repo"),
            "dataset_url": hub_urls.get("dataset_url"),
            "trackio_space_url": hub_urls.get("trackio_space_url"),
            "hf_sync_enabled": os.environ.get("UM8_HF_SYNC_VISUAL_ENTITIES", "").strip().lower()
            in ("1", "true", "yes"),
            "docs": {
                "datasets": hf.get("datasets_hub"),
                "trainer": hf.get("trl_docs"),
                "jobs": hf.get("hub_docs_jobs"),
                "evaluation": hf.get("evaluation_doc"),
            },
        }
    return out
