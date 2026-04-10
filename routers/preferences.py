"""
UploadM8 User Preferences routes -- extracted from app.py.

Handles color preferences and content/upload preferences (hashtags, captions,
thumbnails, Trill settings, etc.).
"""

import json
import logging
import pathlib
import traceback

from fastapi import APIRouter, Body, Depends, HTTPException

import core.state
from core.deps import get_current_user
from core.helpers import _now_utc, _safe_json, _safe_col
from core.models import ColorPreferencesUpdate, UserPreferencesUpdate
from stages.entitlements import get_entitlements_for_tier

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
        _COLOR_COLS = frozenset({"tiktok_color", "youtube_color", "instagram_color", "facebook_color", "accent_color", "updated_at"})
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
        v = up.get(camel) if up.get(camel) is not None else up.get(snake)
        if v is not None:
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
        styled = prefs_row.get("styled_thumbnails", True)
        result = {
            "auto_captions": prefs_row["auto_captions"],
            "auto_thumbnails": prefs_row["auto_thumbnails"],
            "styled_thumbnails": styled,
            "styledThumbnails": styled,
            "thumbnail_interval": prefs_row["thumbnail_interval"],
            "default_privacy": prefs_row["default_privacy"],
            "ai_hashtags_enabled": prefs_row["ai_hashtags_enabled"],
            "ai_hashtag_count": prefs_row["ai_hashtag_count"],
            "ai_hashtag_style": prefs_row["ai_hashtag_style"],
            "hashtag_position": prefs_row["hashtag_position"],
            "max_hashtags": prefs_row["max_hashtags"],
            "always_hashtags": prefs_row["always_hashtags"] or [],
            "blocked_hashtags": prefs_row["blocked_hashtags"] or [],
            "platform_hashtags": prefs_row["platform_hashtags"] or {"tiktok": [], "youtube": [], "instagram": [], "facebook": []},
            "email_notifications": prefs_row["email_notifications"],
            "discord_webhook": prefs_row["discord_webhook"]
        }

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
                if up.get("alwaysHashtags") is not None or up.get("always_hashtags") is not None:
                    v = up.get("alwaysHashtags") or up.get("always_hashtags")
                    out["alwaysHashtags"] = v if isinstance(v, list) else []
                if up.get("blockedHashtags") is not None or up.get("blocked_hashtags") is not None:
                    v = up.get("blockedHashtags") or up.get("blocked_hashtags")
                    out["blockedHashtags"] = v if isinstance(v, list) else []
                if up.get("platformHashtags") is not None or up.get("platform_hashtags") is not None:
                    v = up.get("platformHashtags") or up.get("platform_hashtags")
                    out["platformHashtags"] = v if isinstance(v, dict) else {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}
                if up.get("maxHashtags") is not None or up.get("max_hashtags") is not None:
                    out["maxHashtags"] = str(up.get("maxHashtags") or up.get("max_hashtags") or 15)
                if up.get("aiHashtagCount") is not None or up.get("ai_hashtag_count") is not None:
                    out["aiHashtagCount"] = str(up.get("aiHashtagCount") or up.get("ai_hashtag_count") or 5)
                out["captionStyle"] = up.get("captionStyle") or up.get("caption_style") or "story"
                out["captionTone"] = up.get("captionTone") or up.get("caption_tone") or "authentic"
                out["captionVoice"] = up.get("captionVoice") or up.get("caption_voice") or "default"
                out["captionFrameCount"] = up.get("captionFrameCount") or up.get("caption_frame_count") or 6
            else:
                out.setdefault("captionStyle", "story")
                out.setdefault("captionTone", "authentic")
                out.setdefault("captionVoice", "default")
                out.setdefault("captionFrameCount", 6)
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

    Contract:
    - Frontend sends camelCase keys.
    - DB stores snake_case columns in user_preferences.
    - JSON columns are stored as jsonb (always_hashtags, blocked_hashtags, platform_hashtags).
    """
    # #region agent log
    try:
        import json as _dj, time as _dt, pathlib as _dp
        _dp.Path("debug-2656c2.log").open("a").write(_dj.dumps({"sessionId":"2656c2","hypothesisId":"SAVE-ENTRY","location":"app.py:save_user_preferences","message":"save_user_preferences called","data":{"user_id":str(user.get("id")),"payload_keys":list(payload.keys())[:30]},"timestamp":int(_dt.time()*1000)})+"\n")
    except Exception: pass
    # #endregion

    CAMEL_TO_SNAKE = {
        "autoCaptions": "auto_captions",
        "autoThumbnails": "auto_thumbnails",
        "thumbnailInterval": "thumbnail_interval",
        "defaultPrivacy": "default_privacy",
        "aiHashtagsEnabled": "ai_hashtags_enabled",
        "aiHashtagCount": "ai_hashtag_count",
        "aiHashtagStyle": "ai_hashtag_style",
        "hashtagPosition": "hashtag_position",
        "maxHashtags": "max_hashtags",
        "alwaysHashtags": "always_hashtags",
        "blockedHashtags": "blocked_hashtags",
        "platformHashtags": "platform_hashtags",
        "emailNotifications": "email_notifications",
        "discordWebhook": "discord_webhook",
        "trillEnabled": "trill_enabled",
        "trillMinScore": "trill_min_score",
        "trillHudEnabled": "trill_hud_enabled",
        "trillAiEnhance": "trill_ai_enhance",
        "trillOpenaiModel": "trill_openai_model",
        "styledThumbnails": "styled_thumbnails",
        "captionStyle": "caption_style",
        "captionTone": "caption_tone",
        "captionVoice": "caption_voice",
        "captionFrameCount": "caption_frame_count",
    }

    def normalize_prefs_payload(p: dict) -> dict:
        out: dict = {}
        for k, v in (p or {}).items():
            out[CAMEL_TO_SNAKE.get(k, k)] = v
        return out

    p = normalize_prefs_payload(payload)

    # defaults / coercions (frontend may send strings from text inputs)
    def _coerce_hashtag_list(v):
        """Normalize hashtag input to flat list of strings"""
        if v is None:
            return []
        if isinstance(v, list):
            # Simple flatten - just convert each item to string and filter
            result = []
            for item in v:
                if isinstance(item, str) and item and not item.startswith('[') and not item.startswith('"'):
                    # Only add if it's a simple string, not JSON garbage
                    clean = item.strip().lower().replace('#', '')
                    if clean and len(clean) < 50:  # Reasonable hashtag length
                        result.append(clean)
            return result
        if isinstance(v, str):
            # Simple comma-separated string
            s = v.strip()
            if not s or s.startswith('[') or s.startswith('"'):
                # Looks like JSON garbage, ignore it
                return []
            # Normal comma-separated
            parts = [p.strip().lower().replace('#', '') for p in s.split(',')]
            return [p for p in parts if p and len(p) < 50]
        return []

    def _coerce_platform_map(v):
        default_map = {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}
        if v is None:
            return default_map
        if isinstance(v, dict):
            # Ensure each platform value is a list of strings
            out = {}
            for k, val in v.items():
                out[str(k)] = _coerce_hashtag_list(val)
            # Ensure all expected keys exist
            for k in default_map.keys():
                out.setdefault(k, [])
            return out
        if isinstance(v, str):
            # Try JSON string first, else treat as a global hashtag list applied to all
            s = v.strip()
            if not s:
                return default_map
            try:
                obj = json.loads(s)
                return _coerce_platform_map(obj)
            except Exception:
                lst = _coerce_hashtag_list(s)
                return {k: lst[:] for k in default_map.keys()}
        return default_map

    always = _coerce_hashtag_list(p.get("always_hashtags"))
    blocked = _coerce_hashtag_list(p.get("blocked_hashtags"))
    platform = _coerce_platform_map(p.get("platform_hashtags"))

    # DEBUG: Log what we're about to save
    logger.info(f"Saving preferences for user {user['id']}")
    logger.info(f"always_hashtags: {always} (type: {type(always)})")
    logger.info(f"blocked_hashtags: {blocked} (type: {type(blocked)})")
    logger.info(f"platform_hashtags: {platform} (type: {type(platform)})")

    # core scalar coercions
    auto_captions = bool(p.get("auto_captions", False))
    auto_thumbnails = bool(p.get("auto_thumbnails", False))
    styled_thumbnails = bool(p.get("styled_thumbnails", True))

    try:
        thumbnail_interval = int(p.get("thumbnail_interval", 5))
    except Exception:
        thumbnail_interval = 5

    default_privacy = str(p.get("default_privacy", "public") or "public").lower()
    if default_privacy not in ("public", "unlisted", "private"):
        default_privacy = "public"

    ai_hashtags_enabled = bool(p.get("ai_hashtags_enabled", False))

    try:
        ai_hashtag_count = int(p.get("ai_hashtag_count", 5))
    except Exception:
        ai_hashtag_count = 5

    ai_hashtag_style = str(p.get("ai_hashtag_style", "mixed") or "mixed").lower()
    if ai_hashtag_style not in ("trending", "niche", "mixed"):
        ai_hashtag_style = "mixed"

    hashtag_position = str(p.get("hashtag_position", "end") or "end").lower()
    if hashtag_position not in ("start", "end"):
        hashtag_position = "end"

    try:
        max_hashtags = int(p.get("max_hashtags", 15))
    except Exception:
        max_hashtags = 15

    email_notifications = bool(p.get("email_notifications", True))
    discord_webhook = p.get("discord_webhook")

    # Trill fields (user_preferences)
    trill_enabled = bool(p.get("trill_enabled", False))
    try:
        trill_min_score = int(p.get("trill_min_score", 60))
        trill_min_score = max(0, min(100, trill_min_score))
    except (TypeError, ValueError):
        trill_min_score = 60
    trill_hud_enabled = bool(p.get("trill_hud_enabled", False))
    trill_ai_enhance = bool(p.get("trill_ai_enhance", True))
    trill_openai_model = str(p.get("trill_openai_model", "gpt-4o-mini") or "gpt-4o-mini")[:50]

    try:
      async with core.state.db_pool.acquire() as conn:
        # ensure row exists
        await conn.execute(
            "INSERT INTO user_preferences (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
            user["id"],
        )

        # #region agent log
        try:
            import json as _dj2, time as _dt2
            open("debug-2656c2.log","a").write(_dj2.dumps({"sessionId":"2656c2","hypothesisId":"SAVE-PRE-UPDATE","location":"app.py:save_prefs_update","message":"about to UPDATE user_preferences","data":{"user_id":str(user.get("id")),"auto_captions":auto_captions,"styled_thumbnails":styled_thumbnails,"trill_enabled":trill_enabled,"always_len":len(always),"blocked_len":len(blocked),"platform_keys":list(platform.keys())},"timestamp":int(_dt2.time()*1000)})+"\n")
        except Exception: pass
        # #endregion

        await conn.execute(
            """
            UPDATE user_preferences SET
                auto_captions = $1,
                auto_thumbnails = $2,
                styled_thumbnails = $3,
                thumbnail_interval = $4,
                default_privacy = $5,
                ai_hashtags_enabled = $6,
                ai_hashtag_count = $7,
                ai_hashtag_style = $8,
                hashtag_position = $9,
                max_hashtags = $10,
                always_hashtags = $11::jsonb,
                blocked_hashtags = $12::jsonb,
                platform_hashtags = $13::jsonb,
                email_notifications = $14,
                discord_webhook = $15,
                trill_enabled = $16,
                trill_min_score = $17,
                trill_hud_enabled = $18,
                trill_ai_enhance = $19,
                trill_openai_model = $20,
                updated_at = NOW()
            WHERE user_id = $21
            """,
            auto_captions,
            auto_thumbnails,
            styled_thumbnails,
            thumbnail_interval,
            default_privacy,
            ai_hashtags_enabled,
            ai_hashtag_count,
            ai_hashtag_style,
            hashtag_position,
            max_hashtags,
            json.dumps(always),
            json.dumps(blocked),
            json.dumps(platform),
            email_notifications,
            discord_webhook,
            trill_enabled,
            trill_min_score,
            trill_hud_enabled,
            trill_ai_enhance,
            trill_openai_model,
            user["id"],
        )

        # #region agent log
        try:
            open("debug-2656c2.log","a").write(_dj2.dumps({"sessionId":"2656c2","hypothesisId":"SAVE-POST-UPDATE","location":"app.py:save_prefs_update","message":"UPDATE user_preferences succeeded","timestamp":int(_dt2.time()*1000)})+"\n")
        except Exception: pass
        # #endregion

        # Sync discord_webhook to user_settings so worker (load_user_settings) gets it
        await conn.execute(
            """
            INSERT INTO user_settings (user_id, discord_webhook) VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET discord_webhook = $2, updated_at = NOW()
            """,
            user["id"],
            discord_webhook,
        )

        # Sync caption fields to users.preferences (worker caption_stage reads from there)
        caption_keys = ("captionStyle", "captionTone", "captionVoice", "captionFrameCount")
        caption_snake = ("caption_style", "caption_tone", "caption_voice", "caption_frame_count")
        if any(k in p or sk in p for k, sk in zip(caption_keys, caption_snake)):
            try:
                _CAPTION_STYLES = ("story", "punchy", "factual")
                _CAPTION_TONES = ("hype", "calm", "cinematic", "authentic")
                _CAPTION_VOICES = ("default", "mentor", "hypebeast", "best_friend", "teacher", "cinematic_narrator")
                users_prefs_row = await conn.fetchval("SELECT preferences FROM users WHERE id = $1", user["id"])
                users_prefs = {}
                if users_prefs_row:
                    users_prefs = json.loads(users_prefs_row) if isinstance(users_prefs_row, str) else (users_prefs_row or {})
                if not isinstance(users_prefs, dict):
                    users_prefs = {}
                if "captionStyle" in p or "caption_style" in p:
                    v = str(p.get("captionStyle") or p.get("caption_style") or "story").strip().lower()
                    users_prefs["captionStyle"] = users_prefs["caption_style"] = v if v in _CAPTION_STYLES else "story"
                if "captionTone" in p or "caption_tone" in p:
                    v = str(p.get("captionTone") or p.get("caption_tone") or "authentic").strip().lower()
                    users_prefs["captionTone"] = users_prefs["caption_tone"] = v if v in _CAPTION_TONES else "authentic"
                if "captionVoice" in p or "caption_voice" in p:
                    v = str(p.get("captionVoice") or p.get("caption_voice") or "default").strip().lower()
                    users_prefs["captionVoice"] = users_prefs["caption_voice"] = v if v in _CAPTION_VOICES else "default"
                if "captionFrameCount" in p or "caption_frame_count" in p:
                    try:
                        ent = get_entitlements_for_tier(user.get("subscription_tier", "free"))
                        max_frames = ent.max_caption_frames or 20
                        v = int(p.get("captionFrameCount") or p.get("caption_frame_count") or 6)
                        v = max(2, min(v, max_frames))
                        users_prefs["captionFrameCount"] = users_prefs["caption_frame_count"] = v
                    except (TypeError, ValueError):
                        users_prefs["captionFrameCount"] = users_prefs["caption_frame_count"] = 6
                await conn.execute(
                    "UPDATE users SET preferences = $1, updated_at = NOW() WHERE id = $2",
                    json.dumps(users_prefs),
                    user["id"],
                )
            except Exception as _cap_err:
                logger.warning(f"Caption sync to users.preferences failed (column may not exist): {_cap_err}")

        # immediate read-after-write to validate persistence (helps front-end debugging)
        row = await conn.fetchrow(
            "SELECT updated_at FROM user_preferences WHERE user_id = $1",
            user["id"],
        )

        # #region agent log
        try:
            open("debug-2656c2.log","a").write(_dj2.dumps({"sessionId":"2656c2","hypothesisId":"SAVE-COMPLETE","location":"app.py:save_prefs","message":"save complete","data":{"updated_at":str(row["updated_at"]) if row else None},"timestamp":int(_dt2.time()*1000)})+"\n")
        except Exception: pass
        # #endregion

      return {"ok": True, "updatedAt": (row["updated_at"].isoformat() if row and row.get("updated_at") else None)}
    except Exception as _save_err:
        logger.error(f"save_user_preferences UPDATE failed: {type(_save_err).__name__}: {_save_err}\n{traceback.format_exc()[-800:]}")
        # Retry without trill columns (handles pre-migration schemas)
        try:
            async with core.state.db_pool.acquire() as conn:
                await conn.execute(
                    """UPDATE user_preferences SET
                        auto_captions=$1, auto_thumbnails=$2, thumbnail_interval=$3,
                        default_privacy=$4, ai_hashtags_enabled=$5, ai_hashtag_count=$6,
                        ai_hashtag_style=$7, hashtag_position=$8, max_hashtags=$9,
                        always_hashtags=$10::jsonb, blocked_hashtags=$11::jsonb,
                        platform_hashtags=$12::jsonb, email_notifications=$13,
                        discord_webhook=$14, updated_at=NOW()
                       WHERE user_id=$15""",
                    auto_captions, auto_thumbnails, thumbnail_interval, default_privacy,
                    ai_hashtags_enabled, ai_hashtag_count, ai_hashtag_style, hashtag_position,
                    max_hashtags, json.dumps(always), json.dumps(blocked), json.dumps(platform),
                    email_notifications, discord_webhook, user["id"],
                )
                row = await conn.fetchrow("SELECT updated_at FROM user_preferences WHERE user_id=$1", user["id"])
            return {"ok": True, "updatedAt": (row["updated_at"].isoformat() if row and row.get("updated_at") else None)}
        except Exception as _retry_err:
            logger.error(f"save_user_preferences retry also failed: {_retry_err}")
            raise HTTPException(500, f"Could not save preferences: {type(_retry_err).__name__}")


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
