"""
UploadM8 User Preferences routes -- extracted from app.py.

Handles color preferences and content/upload preferences (hashtags, captions,
thumbnails, Trill settings, etc.).
"""

import json
import logging
import pathlib
from fastapi import APIRouter, Body, Depends, HTTPException

import core.state
from core.deps import get_current_user
from core.helpers import _now_utc, _safe_json, _safe_col, merge_platform_hashtag_overlay
from core.sql_allowlist import USER_COLOR_PREFERENCES_UPDATE_COLUMNS, assert_set_fragments_columns
from core.models import ColorPreferencesUpdate, UserPreferencesUpdate
from services.user_preferences_persist import save_user_content_preferences

logger = logging.getLogger(__name__)

router = APIRouter(tags=["preferences"])


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
                "accent_color": "#F97316"
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

_DEBUG_LOG_PATH = pathlib.Path(__file__).resolve().parent.parent / "debug-0d13f7.log"

def _dbg_write(msg: str, data: dict = None, hid: str = "H3"):
    try:
        import json as _j
        line = _j.dumps({"sessionId":"0d13f7","hypothesisId":hid,"location":"app.py:get_user_preferences","message":msg,"data":data or {},"timestamp":__import__("time").time()*1000}) + "\n"
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


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
        # Do not wipe rich user_preferences.platform_hashtags with a bare {} from
        # users.preferences (empty dict is falsy in some merges and was stored by mistake).
        if key == "platform_hashtags" and isinstance(v, dict):
            base = result.get("platformHashtags") or result.get("platform_hashtags") or {}
            merged = merge_platform_hashtag_overlay(base, v)
            result["platformHashtags"] = result["platform_hashtags"] = merged
            continue
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
        ("trill_hud_enabled", "trillHudEnabled"),
        ("trill_ai_enhance", "trillAiEnhance"),
        ("trill_openai_model", "trillOpenaiModel"),
        ("use_audio_context", "useAudioContext"),
        ("audio_transcription", "audioTranscription"),
        ("ai_service_telemetry", "aiServiceTelemetry"),
        ("ai_service_audio_signals", "aiServiceAudioSignals"),
        ("ai_service_music_detection", "aiServiceMusicDetection"),
        ("ai_service_audio_summary", "aiServiceAudioSummary"),
        ("ai_service_emotion_signals", "aiServiceEmotionSignals"),
        ("ai_service_caption_writer", "aiServiceCaptionWriter"),
        ("ai_service_thumbnail_designer", "aiServiceThumbnailDesigner"),
        ("ai_service_frame_inspector", "aiServiceFrameInspector"),
        ("ai_service_speech_to_text", "aiServiceSpeechToText"),
        ("ai_service_video_analyzer", "aiServiceVideoAnalyzer"),
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
        ("audioTranscription", "audio_transcription"),
        ("aiServiceTelemetry", "ai_service_telemetry"),
        ("aiServiceAudioSignals", "ai_service_audio_signals"),
        ("aiServiceMusicDetection", "ai_service_music_detection"),
        ("aiServiceAudioSummary", "ai_service_audio_summary"),
        ("aiServiceEmotionSignals", "ai_service_emotion_signals"),
        ("aiServiceCaptionWriter", "ai_service_caption_writer"),
        ("aiServiceThumbnailDesigner", "ai_service_thumbnail_designer"),
        ("aiServiceFrameInspector", "ai_service_frame_inspector"),
        ("aiServiceSpeechToText", "ai_service_speech_to_text"),
        ("aiServiceVideoAnalyzer", "ai_service_video_analyzer"),
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
    - user_settings provides hud_enabled for Trill HUD cost calculation
    """
    result = {}
    # Read from user_preferences table
    prefs_row = await conn.fetchrow(
        "SELECT * FROM user_preferences WHERE user_id = $1",
        user_id
    )
    if prefs_row:
        pr = dict(prefs_row)
        for _k in ("created_at", "updated_at", "user_id"):
            pr.pop(_k, None)
        result.update(pr)
        result["always_hashtags"] = result.get("always_hashtags") or []
        result["blocked_hashtags"] = result.get("blocked_hashtags") or []
        result["platform_hashtags"] = result.get("platform_hashtags") or {
            "tiktok": [],
            "youtube": [],
            "instagram": [],
            "facebook": [],
        }
        styled = bool(result.get("styled_thumbnails", True))
        result["styled_thumbnails"] = styled
        result["styledThumbnails"] = styled
        _hydrate_snake_camel_mirror(result)

    # Overlay users.preferences (PUT /api/me/preferences writes here) -- full overlay
    users_prefs_row = await conn.fetchrow("SELECT preferences FROM users WHERE id = $1", user_id)
    up = _parse_users_preferences(users_prefs_row["preferences"] if users_prefs_row else None)
    _overlay_users_prefs_on_result(result, up)

    # Overlay user_settings for hud_enabled (Trill HUD; PUT /api/settings writes here)
    try:
        us_row = await conn.fetchrow("SELECT hud_enabled FROM user_settings WHERE user_id = $1", user_id)
        if us_row and us_row.get("hud_enabled") is not None:
            result["hud_enabled"] = bool(us_row["hud_enabled"])
    except Exception:
        pass

    if result:
        _hydrate_snake_camel_mirror(result)
        return result

    # Fallback: Try legacy JSONB locations
    # Try user_settings.preferences_json first
    prefs_row = await conn.fetchrow(
        "SELECT preferences_json FROM user_settings WHERE user_id = $1",
        user_id
    )

    if prefs_row and prefs_row["preferences_json"]:
        prefs_data = prefs_row["preferences_json"]
        if isinstance(prefs_data, str):
            prefs = json.loads(prefs_data)
        else:
            prefs = prefs_data
    else:
        # Try users.preferences (oldest fallback)
        prefs_row = await conn.fetchrow(
            "SELECT preferences FROM users WHERE id = $1",
            user_id
        )
        if prefs_row and prefs_row["preferences"]:
            prefs_data = prefs_row["preferences"]
            if isinstance(prefs_data, str):
                prefs = json.loads(prefs_data)
            else:
                prefs = prefs_data
        else:
            prefs = {}

    # Return preferences with defaults (convert camelCase to snake_case for internal use)
    styled = prefs.get("styledThumbnails", prefs.get("styled_thumbnails", True))
    out = {
        "auto_captions": prefs.get("autoCaptions", False),
        "auto_thumbnails": prefs.get("autoThumbnails", False),
        "styled_thumbnails": styled,
        "styledThumbnails": styled,
        "thumbnail_interval": prefs.get("thumbnailInterval", 5),
        "default_privacy": prefs.get("defaultPrivacy", "public"),
        "ai_hashtags_enabled": prefs.get("aiHashtagsEnabled", False),
        "ai_hashtag_count": prefs.get("aiHashtagCount", 5),
        "ai_hashtag_style": prefs.get("aiHashtagStyle", "mixed"),
        "hashtag_position": prefs.get("hashtagPosition", "end"),
        "max_hashtags": prefs.get("maxHashtags", 30),
        "always_hashtags": prefs.get("alwaysHashtags", []),
        "blocked_hashtags": prefs.get("blockedHashtags", []),
        "platform_hashtags": prefs.get("platformHashtags", {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}),
        "email_notifications": prefs.get("emailNotifications", True),
        "discord_webhook": prefs.get("discordWebhook", None)
    }
    # Add hud_enabled from user_settings for fallback path
    try:
        us_row = await conn.fetchrow("SELECT hud_enabled FROM user_settings WHERE user_id = $1", user_id)
        if us_row and us_row.get("hud_enabled") is not None:
            out["hud_enabled"] = bool(us_row["hud_enabled"])
    except Exception:
        pass
    if isinstance(prefs, dict):
        _overlay_users_prefs_on_result(out, prefs)
    _hydrate_snake_camel_mirror(out)
    return out


# ============================================================
# GET /api/settings/preferences
# ============================================================
@router.get("/api/settings/preferences")
async def get_user_preferences(user: dict = Depends(get_current_user)):
    """GET user content preferences - used by settings page AND upload workflow"""
    # #region agent log
    _dbg_write("get_user_preferences ENTRY", {"user_id": str(user.get("id"))})
    # #endregion
    try:
        async with core.state.db_pool.acquire() as conn:
            # #region agent log
            _dbg_write("Before SELECT user_preferences")
            # #endregion
            try:
                prefs = await conn.fetchrow(
                    "SELECT * FROM user_preferences WHERE user_id = $1",
                    user["id"]
                )
            except Exception as e1:
                _dbg_write("user_preferences SELECT failed", {"error": str(e1), "type": type(e1).__name__})
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

            # DEBUG: Log what we loaded from database
            logger.info(f"Loading preferences for user {user['id']}")
            logger.info(f"always_hashtags from DB: {always_tags} (type: {type(always_tags)})")
            logger.info(f"blocked_hashtags from DB: {blocked_tags} (type: {type(blocked_tags)})")

            # Parse JSON strings if needed (JSONB might come back as strings)
            if isinstance(always_tags, str):
                try:
                    always_tags = json.loads(always_tags)
                except Exception:
                    always_tags = []
            if isinstance(blocked_tags, str):
                try:
                    blocked_tags = json.loads(blocked_tags)
                except Exception:
                    blocked_tags = []
            if isinstance(platform_tags, str):
                try:
                    platform_tags = json.loads(platform_tags)
                except Exception:
                    platform_tags = {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}

            out = {
                "autoCaptions": d.get("auto_captions", False),
                "autoThumbnails": d.get("auto_thumbnails", False),
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
                "trillEnabled": bool(d.get("trill_enabled", False)),
                "trillMinScore": int(d.get("trill_min_score", 0) or 0),
                "trillHudEnabled": bool(d.get("trill_hud_enabled", False)),
                "trillAiEnhance": bool(d.get("trill_ai_enhance", False)),
                "trillOpenaiModel": d.get("trill_openai_model", "gpt-4o-mini"),
                "styledThumbnails": d.get("styled_thumbnails", True),
                "useAudioContext": bool(d.get("use_audio_context", True)),
                "audioTranscription": bool(d.get("audio_transcription", True)),
                "authSecurityAlerts": bool(d.get("auth_security_alerts", True)),
                "digestEmails": bool(d.get("digest_emails", True)),
                "scheduledAlertEmails": bool(d.get("scheduled_alert_emails", True)),
                "aiServiceTelemetry": bool(d.get("ai_service_telemetry", True)),
                "aiServiceAudioSignals": bool(d.get("ai_service_audio_signals", True)),
                "aiServiceMusicDetection": bool(d.get("ai_service_music_detection", True)),
                "aiServiceAudioSummary": bool(d.get("ai_service_audio_summary", True)),
                "aiServiceEmotionSignals": bool(d.get("ai_service_emotion_signals", False)),
                "aiServiceCaptionWriter": bool(d.get("ai_service_caption_writer", True)),
                "aiServiceThumbnailDesigner": bool(d.get("ai_service_thumbnail_designer", True)),
                "aiServiceFrameInspector": bool(d.get("ai_service_frame_inspector", True)),
                "aiServiceSpeechToText": bool(d.get("ai_service_speech_to_text", True)),
                "aiServiceVideoAnalyzer": bool(d.get("ai_service_video_analyzer", True)),
                "aiServiceSceneUnderstanding": bool(d.get("ai_service_scene_understanding", True)),
            }
            # #region agent log
            _dbg_write("Before SELECT users.preferences")
            # #endregion
            # Overlay users.preferences -- source of truth for hashtags + caption (PUT /api/me/preferences)
            users_prefs = None
            try:
                users_prefs = await conn.fetchval("SELECT preferences FROM users WHERE id = $1", user["id"])
            except Exception as col_err:
                _dbg_write("users.preferences SELECT failed (column may not exist)", {"error": str(col_err)})
            # #region agent log
            _dbg_write("After SELECT users.preferences", {"has_prefs": users_prefs is not None})
            # #endregion
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
                out.setdefault("thumbnailSelectionMode", "ai")
                out.setdefault("thumbnail_selection_mode", "ai")
                out.setdefault("thumbnailRenderPipeline", "auto")
                out.setdefault("thumbnail_render_pipeline", "auto")
            return out
    except Exception as e:
        # #region agent log
        _dbg_write("get_user_preferences EXCEPTION", {"error": str(e), "type": type(e).__name__})
        # #endregion
        logger.exception("get_user_preferences failed: %s", e)
        # Return defaults so settings page loads; avoid 500 when DB schema mismatch or migration not run
        return {
            "autoCaptions": False, "autoThumbnails": False, "thumbnailInterval": "5",
            "defaultPrivacy": "public", "aiHashtagsEnabled": False, "aiHashtagCount": "5",
            "aiHashtagStyle": "mixed", "hashtagPosition": "end", "maxHashtags": "15",
            "alwaysHashtags": [], "blockedHashtags": [],
            "platformHashtags": {"tiktok": [], "youtube": [], "instagram": [], "facebook": []},
            "emailNotifications": True, "discordWebhook": None,
            "trillEnabled": False, "trillMinScore": 60, "trillHudEnabled": False,
            "trillAiEnhance": True, "trillOpenaiModel": "gpt-4o-mini", "styledThumbnails": True,
            "captionStyle": "story", "captionTone": "authentic", "captionVoice": "default", "captionFrameCount": 6,
            "thumbnailSelectionMode": "ai", "thumbnail_selection_mode": "ai",
            "thumbnailRenderPipeline": "auto", "thumbnail_render_pipeline": "auto",
        }


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
