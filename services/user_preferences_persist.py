"""Persist POST/PUT /api/settings/preferences — extracted from app.py for testability and slimmer routes."""
from __future__ import annotations

import json
import logging
import re
import traceback
from typing import Any, Mapping

from fastapi import HTTPException

from stages.entitlements import get_entitlements_for_tier

from core.upload_preference_dependencies import normalize_preferences_dict, normalize_upload_preferences_snake

logger = logging.getLogger(__name__)

_CAMEL_TO_SNAKE: dict[str, str] = {
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
    "authSecurityAlerts": "auth_security_alerts",
    "digestEmails": "digest_emails",
    "scheduledAlertEmails": "scheduled_alert_emails",
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
    "thumbnailSelectionMode": "thumbnail_selection_mode",
    "thumbnailRenderPipeline": "thumbnail_render_pipeline",
    "useAudioContext": "use_audio_context",
    "audioTranscription": "audio_transcription",
    "aiServiceTelemetry": "ai_service_telemetry",
    "aiServiceAudioSignals": "ai_service_audio_signals",
    "aiServiceMusicDetection": "ai_service_music_detection",
    "aiServiceAudioSummary": "ai_service_audio_summary",
    "aiServiceEmotionSignals": "ai_service_emotion_signals",
    "aiServiceCaptionWriter": "ai_service_caption_writer",
    "aiServiceThumbnailDesigner": "ai_service_thumbnail_designer",
    "aiServiceFrameInspector": "ai_service_frame_inspector",
    "aiServiceSpeechToText": "ai_service_speech_to_text",
    "aiServiceVideoAnalyzer": "ai_service_video_analyzer",
    "aiServiceSceneUnderstanding": "ai_service_scene_understanding",
    "authSecurityAlerts": "auth_security_alerts",
    "digestEmails": "digest_emails",
    "scheduledAlertEmails": "scheduled_alert_emails",
}


def _normalize_prefs_payload(p: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (p or {}).items():
        out[_CAMEL_TO_SNAKE.get(k, k)] = v
    return out


def _coerce_hashtag_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        result: list[str] = []
        for item in v:
            if item is None:
                continue
            sitem = str(item).strip() if not isinstance(item, str) else item.strip()
            if not sitem or sitem.startswith("[") or sitem.startswith('"'):
                continue
            clean = re.sub(r"[^a-z0-9_]", "", sitem.lower().replace("#", ""))
            if clean and len(clean) < 50:
                result.append(clean)
        return result
    if isinstance(v, str):
        s = v.strip()
        if not s or s.startswith("[") or s.startswith('"'):
            return []
        parts = [re.sub(r"[^a-z0-9_]", "", p.strip().lower().replace("#", "")) for p in s.split(",")]
        return [p for p in parts if p and len(p) < 50]
    return []


def _coerce_platform_map(v: Any) -> dict[str, list[str]]:
    default_map = {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}
    _aliases = {"google": "youtube", "ig": "instagram", "fb": "facebook"}
    if v is None:
        return default_map
    if isinstance(v, dict):
        out: dict[str, list[str]] = {}
        for k, val in v.items():
            nk = str(k).strip().lower()
            nk = _aliases.get(nk, nk)
            out[nk] = _coerce_hashtag_list(val)
        for k in default_map:
            out.setdefault(k, [])
        return out
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return default_map
        try:
            obj = json.loads(s)
            return _coerce_platform_map(obj)
        except Exception:
            lst = _coerce_hashtag_list(s)
            return {k: lst[:] for k in default_map}
    return default_map


async def save_user_content_preferences(conn, user: dict[str, Any], payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Full UPDATE user_preferences + user_settings sync + optional users.preferences caption merge.
    Returns {"ok": True, "updatedAt": iso | None}.
    """
    uid = user["id"]
    raw_in = dict(payload or {})
    existing_row = await conn.fetchrow("SELECT * FROM user_preferences WHERE user_id = $1", uid)
    if existing_row:
        ex = dict(existing_row)
        keys_sent = set(raw_in.keys())
        for camel, snake in _CAMEL_TO_SNAKE.items():
            if camel in keys_sent or snake in keys_sent:
                continue
            if snake in ex:
                raw_in[camel] = ex[snake]
        for jsnake, jcamel in (
            ("always_hashtags", "alwaysHashtags"),
            ("blocked_hashtags", "blockedHashtags"),
            ("platform_hashtags", "platformHashtags"),
        ):
            if jcamel in keys_sent or jsnake in keys_sent:
                continue
            if jsnake in ex:
                raw_in[jsnake] = ex[jsnake]

    p = _normalize_prefs_payload(raw_in)
    p = normalize_upload_preferences_snake(p)

    always = _coerce_hashtag_list(p.get("always_hashtags"))
    blocked = _coerce_hashtag_list(p.get("blocked_hashtags"))
    platform = _coerce_platform_map(p.get("platform_hashtags"))

    log = logger
    log.info("Saving preferences for user %s", uid)
    log.info("always_hashtags: %s", always)
    log.info("blocked_hashtags: %s", blocked)
    log.info("platform_hashtags: %s", platform)

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
    # UI: lowercase | capitalized | camelcase | mixed — legacy DB rows may use trending/niche.
    if ai_hashtag_style not in (
        "lowercase",
        "capitalized",
        "camelcase",
        "mixed",
        "trending",
        "niche",
    ):
        ai_hashtag_style = "mixed"

    hashtag_position = str(p.get("hashtag_position", "end") or "end").lower()
    # UI includes "First comment (IG only)" (comment) and legacy "caption" — publish may still treat
    # comment/caption like "end" until a first-comment API path exists.
    if hashtag_position not in ("start", "end", "comment", "caption"):
        hashtag_position = "end"

    try:
        max_hashtags = int(p.get("max_hashtags", 15))
    except Exception:
        max_hashtags = 15

    email_notifications = bool(p.get("email_notifications", True))
    auth_security_alerts = bool(p.get("auth_security_alerts", True))
    digest_emails = bool(p.get("digest_emails", True))
    scheduled_alert_emails = bool(p.get("scheduled_alert_emails", True))
    discord_webhook = p.get("discord_webhook")

    trill_enabled = bool(p.get("trill_enabled", False))
    try:
        trill_min_score = int(p.get("trill_min_score", 60))
        trill_min_score = max(0, min(100, trill_min_score))
    except (TypeError, ValueError):
        trill_min_score = 60
    trill_hud_enabled = bool(p.get("trill_hud_enabled", False))
    trill_ai_enhance = bool(p.get("trill_ai_enhance", True))
    trill_openai_model = str(p.get("trill_openai_model", "gpt-4o-mini") or "gpt-4o-mini")[:50]
    use_audio_context = bool(p.get("useAudioContext", p.get("use_audio_context", True)))
    audio_transcription = bool(p.get("audioTranscription", p.get("audio_transcription", True)))
    ai_service_telemetry = bool(p.get("aiServiceTelemetry", p.get("ai_service_telemetry", True)))
    ai_service_audio_signals = bool(p.get("aiServiceAudioSignals", p.get("ai_service_audio_signals", True)))
    ai_service_music_detection = bool(p.get("aiServiceMusicDetection", p.get("ai_service_music_detection", True)))
    ai_service_audio_summary = bool(p.get("aiServiceAudioSummary", p.get("ai_service_audio_summary", True)))
    ai_service_emotion_signals = bool(p.get("aiServiceEmotionSignals", p.get("ai_service_emotion_signals", False)))
    ai_service_caption_writer = bool(p.get("aiServiceCaptionWriter", p.get("ai_service_caption_writer", True)))
    ai_service_thumbnail_designer = bool(
        p.get("aiServiceThumbnailDesigner", p.get("ai_service_thumbnail_designer", True))
    )
    ai_service_frame_inspector = bool(p.get("aiServiceFrameInspector", p.get("ai_service_frame_inspector", True)))
    ai_service_speech_to_text = bool(p.get("aiServiceSpeechToText", p.get("ai_service_speech_to_text", True)))
    ai_service_video_analyzer = bool(p.get("aiServiceVideoAnalyzer", p.get("ai_service_video_analyzer", True)))
    ai_service_scene_understanding = bool(
        p.get("aiServiceSceneUnderstanding", p.get("ai_service_scene_understanding", True))
    )

    try:
        await conn.execute(
            "INSERT INTO user_preferences (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
            uid,
        )

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
                auth_security_alerts = $15,
                digest_emails = $16,
                scheduled_alert_emails = $17,
                discord_webhook = $18,
                trill_enabled = $19,
                trill_min_score = $20,
                trill_hud_enabled = $21,
                trill_ai_enhance = $22,
                trill_openai_model = $23,
                use_audio_context = $24,
                audio_transcription = $25,
                ai_service_telemetry = $26,
                ai_service_audio_signals = $27,
                ai_service_music_detection = $28,
                ai_service_audio_summary = $29,
                ai_service_emotion_signals = $30,
                ai_service_caption_writer = $31,
                ai_service_thumbnail_designer = $32,
                ai_service_frame_inspector = $33,
                ai_service_speech_to_text = $34,
                ai_service_video_analyzer = $35,
                ai_service_scene_understanding = $36,
                updated_at = NOW()
            WHERE user_id = $37
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
            auth_security_alerts,
            digest_emails,
            scheduled_alert_emails,
            discord_webhook,
            trill_enabled,
            trill_min_score,
            trill_hud_enabled,
            trill_ai_enhance,
            trill_openai_model,
            use_audio_context,
            audio_transcription,
            ai_service_telemetry,
            ai_service_audio_signals,
            ai_service_music_detection,
            ai_service_audio_summary,
            ai_service_emotion_signals,
            ai_service_caption_writer,
            ai_service_thumbnail_designer,
            ai_service_frame_inspector,
            ai_service_speech_to_text,
            ai_service_video_analyzer,
            ai_service_scene_understanding,
            uid,
        )

        await conn.execute(
            """
            INSERT INTO user_settings (user_id, discord_webhook, telemetry_enabled) VALUES ($1, $2, $3)
            ON CONFLICT (user_id) DO UPDATE SET
                discord_webhook = $2,
                telemetry_enabled = $3,
                updated_at = NOW()
            """,
            uid,
            discord_webhook,
            trill_enabled,
        )

        caption_keys = ("captionStyle", "captionTone", "captionVoice", "captionFrameCount")
        caption_snake = ("caption_style", "caption_tone", "caption_voice", "caption_frame_count")
        thumb_keys = ("thumbnailSelectionMode", "thumbnailRenderPipeline")
        thumb_snake = ("thumbnail_selection_mode", "thumbnail_render_pipeline")
        studio_keys = (
            "thumbnailStudioEnabled",
            "thumbnail_studio_enabled",
            "thumbnailStudioEngineEnabled",
            "thumbnail_studio_engine_enabled",
            "thumbnailPersonaEnabled",
            "thumbnail_persona_enabled",
            "thumbnailDefaultPersonaId",
            "thumbnail_default_persona_id",
            "thumbnailPersonaStrength",
            "thumbnail_persona_strength",
        )
        if any(k in p or sk in p for k, sk in zip(caption_keys, caption_snake)) or any(
            k in p or sk in p for k, sk in zip(thumb_keys, thumb_snake)
        ) or any(k in p for k in studio_keys):
            try:
                _CAPTION_STYLES = ("story", "punchy", "factual")
                _CAPTION_TONES = ("hype", "calm", "cinematic", "authentic")
                _CAPTION_VOICES = (
                    "default",
                    "mentor",
                    "hypebeast",
                    "best_friend",
                    "teacher",
                    "cinematic_narrator",
                )
                users_prefs_row = await conn.fetchval("SELECT preferences FROM users WHERE id = $1", uid)
                users_prefs: dict[str, Any] = {}
                if users_prefs_row:
                    users_prefs = (
                        json.loads(users_prefs_row) if isinstance(users_prefs_row, str) else (users_prefs_row or {})
                    )
                if not isinstance(users_prefs, dict):
                    users_prefs = {}
                if "captionStyle" in p or "caption_style" in p:
                    v = str(p.get("captionStyle") or p.get("caption_style") or "story").strip().lower()
                    users_prefs["captionStyle"] = users_prefs["caption_style"] = (
                        v if v in _CAPTION_STYLES else "story"
                    )
                if "captionTone" in p or "caption_tone" in p:
                    v = str(p.get("captionTone") or p.get("caption_tone") or "authentic").strip().lower()
                    users_prefs["captionTone"] = users_prefs["caption_tone"] = (
                        v if v in _CAPTION_TONES else "authentic"
                    )
                if "captionVoice" in p or "caption_voice" in p:
                    v = str(p.get("captionVoice") or p.get("caption_voice") or "default").strip().lower()
                    users_prefs["captionVoice"] = users_prefs["caption_voice"] = (
                        v if v in _CAPTION_VOICES else "default"
                    )
                if "captionFrameCount" in p or "caption_frame_count" in p:
                    try:
                        ent = get_entitlements_for_tier(user.get("subscription_tier", "free"))
                        max_frames = ent.max_caption_frames or 20
                        v = int(p.get("captionFrameCount") or p.get("caption_frame_count") or 6)
                        v = max(2, min(v, max_frames))
                        users_prefs["captionFrameCount"] = users_prefs["caption_frame_count"] = v
                    except (TypeError, ValueError):
                        users_prefs["captionFrameCount"] = users_prefs["caption_frame_count"] = 6
                if "thumbnailSelectionMode" in p or "thumbnail_selection_mode" in p:
                    v = str(
                        p.get("thumbnailSelectionMode") or p.get("thumbnail_selection_mode") or "ai"
                    ).strip().lower()
                    if v not in ("ai", "sharpness"):
                        v = "ai"
                    users_prefs["thumbnailSelectionMode"] = users_prefs["thumbnail_selection_mode"] = v
                if "thumbnailRenderPipeline" in p or "thumbnail_render_pipeline" in p:
                    v = str(
                        p.get("thumbnailRenderPipeline") or p.get("thumbnail_render_pipeline") or "auto"
                    ).strip().lower()
                    allowed = frozenset(("auto", "studio_renderer", "ai_edit", "template", "none"))
                    if v not in allowed:
                        v = "auto"
                    users_prefs["thumbnailRenderPipeline"] = users_prefs["thumbnail_render_pipeline"] = v

                def _pick_bool(camel: str, snake: str) -> bool | None:
                    if camel in p:
                        return bool(p[camel])
                    if snake in p:
                        return bool(p[snake])
                    return None

                v_studio = _pick_bool("thumbnailStudioEnabled", "thumbnail_studio_enabled")
                if v_studio is not None:
                    users_prefs["thumbnailStudioEnabled"] = users_prefs["thumbnail_studio_enabled"] = v_studio
                v_engine = _pick_bool("thumbnailStudioEngineEnabled", "thumbnail_studio_engine_enabled")
                if v_engine is not None:
                    users_prefs["thumbnailStudioEngineEnabled"] = users_prefs["thumbnail_studio_engine_enabled"] = (
                        v_engine
                    )
                v_persona = _pick_bool("thumbnailPersonaEnabled", "thumbnail_persona_enabled")
                if v_persona is not None:
                    users_prefs["thumbnailPersonaEnabled"] = users_prefs["thumbnail_persona_enabled"] = v_persona
                if "thumbnailDefaultPersonaId" in p or "thumbnail_default_persona_id" in p:
                    pid = p.get("thumbnailDefaultPersonaId")
                    if pid is None:
                        pid = p.get("thumbnail_default_persona_id")
                    pid_s = str(pid or "").strip() or None
                    users_prefs["thumbnailDefaultPersonaId"] = users_prefs["thumbnail_default_persona_id"] = pid_s
                if "thumbnailPersonaStrength" in p or "thumbnail_persona_strength" in p:
                    try:
                        ps = int(p.get("thumbnailPersonaStrength") or p.get("thumbnail_persona_strength") or 70)
                        ps = max(0, min(100, ps))
                    except (TypeError, ValueError):
                        ps = 70
                    users_prefs["thumbnailPersonaStrength"] = users_prefs["thumbnail_persona_strength"] = ps

                normalize_preferences_dict(users_prefs)
                await conn.execute(
                    "UPDATE users SET preferences = $1, updated_at = NOW() WHERE id = $2",
                    json.dumps(users_prefs),
                    uid,
                )
            except Exception as _cap_err:
                log.warning("Caption sync to users.preferences failed (column may not exist): %s", _cap_err)

        row = await conn.fetchrow("SELECT updated_at FROM user_preferences WHERE user_id = $1", uid)

        return {
            "ok": True,
            "updatedAt": (row["updated_at"].isoformat() if row and row.get("updated_at") else None),
        }
    except Exception as _save_err:
        log.error(
            "save_user_preferences UPDATE failed: %s: %s\n%s",
            type(_save_err).__name__,
            _save_err,
            traceback.format_exc()[-800:],
        )
        try:
            await conn.execute(
                """UPDATE user_preferences SET
                    auto_captions=$1, auto_thumbnails=$2, thumbnail_interval=$3,
                    default_privacy=$4, ai_hashtags_enabled=$5, ai_hashtag_count=$6,
                    ai_hashtag_style=$7, hashtag_position=$8, max_hashtags=$9,
                    always_hashtags=$10::jsonb, blocked_hashtags=$11::jsonb,
                    platform_hashtags=$12::jsonb, email_notifications=$13,
                    auth_security_alerts=$14, digest_emails=$15, scheduled_alert_emails=$16,
                    discord_webhook=$17, updated_at=NOW()
                   WHERE user_id=$18""",
                auto_captions,
                auto_thumbnails,
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
                auth_security_alerts,
                digest_emails,
                scheduled_alert_emails,
                discord_webhook,
                uid,
            )
            row = await conn.fetchrow("SELECT updated_at FROM user_preferences WHERE user_id=$1", uid)
            return {
                "ok": True,
                "updatedAt": (row["updated_at"].isoformat() if row and row.get("updated_at") else None),
            }
        except Exception as _retry_err:
            log.error("save_user_preferences retry also failed: %s", _retry_err)
            raise HTTPException(500, f"Could not save preferences: {type(_retry_err).__name__}") from _retry_err


async def save_user_content_preferences_with_pool(pool, user: dict[str, Any], payload: Mapping[str, Any] | None) -> dict[str, Any]:
    async with pool.acquire() as conn:
        return await save_user_content_preferences(conn, user, payload)
