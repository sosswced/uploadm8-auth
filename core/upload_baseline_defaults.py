"""
Upload / worker preference baselines — single source of truth.

Ensures users with no ``user_preferences`` row (or sparse NULL columns) still
have every key the pipeline reads, and that Redis job payloads are JSON-safe
(asyncpg returns UUID/datetime objects on table rows).

Used by:
  - ``stages.db.load_user_settings``
  - ``routers.preferences.get_user_prefs_for_upload`` / GET settings
  - ``worker._build_process_job_payload`` (sanitize before enqueue)
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, FrozenSet, Mapping, Optional

# Columns returned from DB that must never enter Redis job JSON.
UPLOAD_PREF_STRIP_KEYS: FrozenSet[str] = frozenset(
    {
        "id",
        "user_id",
        "created_at",
        "updated_at",
        "trill_public_name_reviewed_at",
        "trill_public_name_reviewed_by",
        "trill_welcome_modal_seen_at",
    }
)

ENTRY_TIER_SLUGS: FrozenSet[str] = frozenset({"free"})

# Universal floor: publish path succeeds without visiting Settings.
UNIVERSAL_UPLOAD_BASELINE: Dict[str, Any] = {
    # Captions / hashtags
    "auto_captions": False,
    "autoCaptions": False,
    "auto_thumbnails": True,
    "autoThumbnails": True,
    "styled_thumbnails": True,
    "styledThumbnails": True,
    "thumbnail_interval": 5,
    "thumbnailInterval": "5",
    "default_privacy": "public",
    "defaultPrivacy": "public",
    "ai_hashtags_enabled": False,
    "aiHashtagsEnabled": False,
    "ai_hashtag_count": 5,
    "aiHashtagCount": "5",
    "ai_hashtag_style": "mixed",
    "aiHashtagStyle": "mixed",
    "hashtag_position": "end",
    "hashtagPosition": "end",
    "max_hashtags": 15,
    "maxHashtags": "15",
    "always_hashtags": [],
    "alwaysHashtags": [],
    "blocked_hashtags": [],
    "blockedHashtags": [],
    "platform_hashtags": {
        "tiktok": [],
        "youtube": [],
        "instagram": [],
        "facebook": [],
    },
    "platformHashtags": {
        "tiktok": [],
        "youtube": [],
        "instagram": [],
        "facebook": [],
    },
    # Caption style (M8 when user enables captions later)
    "caption_style": "story",
    "captionStyle": "story",
    "caption_tone": "authentic",
    "captionTone": "authentic",
    "caption_voice": "default",
    "captionVoice": "default",
    "caption_frame_count": 3,
    "captionFrameCount": 3,
    # Thumbnail studio
    "thumbnail_selection_mode": "sharpness",
    "thumbnailSelectionMode": "sharpness",
    "thumbnail_render_pipeline": "auto",
    "thumbnailRenderPipeline": "auto",
    "thumbnail_studio_enabled": True,
    "thumbnailStudioEnabled": True,
    "thumbnail_studio_engine_enabled": True,
    "thumbnailStudioEngineEnabled": True,
    "thumbnail_persona_enabled": False,
    "thumbnailPersonaEnabled": False,
    # Audio
    "use_audio_context": True,
    "useAudioContext": True,
    "audio_transcription": True,
    "audioTranscription": True,
    # Trill / drive
    "trill_enabled": True,
    "trillEnabled": True,
    "trill_min_score": 60,
    "trillMinScore": 60,
    "trill_ai_enhance": True,
    "trillAiEnhance": True,
    "trill_openai_model": "gpt-4o-mini",
    "trillOpenaiModel": "gpt-4o-mini",
    # Per-service AI toggles (Settings page mirrors these keys)
    "ai_service_telemetry": True,
    "aiServiceTelemetry": True,
    "ai_service_dashcam_osd": True,
    "aiServiceDashcamOSD": True,
    "ai_service_audio_signals": True,
    "aiServiceAudioSignals": True,
    "ai_service_music_detection": True,
    "aiServiceMusicDetection": True,
    "ai_service_audio_summary": True,
    "aiServiceAudioSummary": True,
    "ai_service_emotion_signals": False,
    "aiServiceEmotionSignals": False,
    "ai_service_caption_writer": True,
    "aiServiceCaptionWriter": True,
    "ai_service_thumbnail_designer": True,
    "aiServiceThumbnailDesigner": True,
    "ai_service_speech_to_text": True,
    "aiServiceSpeechToText": True,
    "ai_service_scene_understanding": True,
    "aiServiceSceneUnderstanding": True,
    "ai_service_frame_inspector": True,
    "aiServiceFrameInspector": True,
    "ai_service_video_analyzer": True,
    "aiServiceVideoAnalyzer": True,
}

# Free / entry: lighter defaults when keys are unset (faster, fewer API calls).
FREE_TIER_PROCESSING_DEFAULTS: Dict[str, Any] = {
    "auto_captions": False,
    "autoCaptions": False,
    "auto_thumbnails": True,
    "autoThumbnails": True,
    "styled_thumbnails": False,
    "styledThumbnails": False,
    "use_audio_context": False,
    "useAudioContext": False,
    "audio_transcription": False,
    "audioTranscription": False,
    "thumbnail_studio_enabled": False,
    "thumbnailStudioEnabled": False,
    "thumbnail_studio_engine_enabled": False,
    "thumbnailStudioEngineEnabled": False,
    "thumbnail_render_pipeline": "none",
    "thumbnailRenderPipeline": "none",
    "ai_service_telemetry": False,
    "aiServiceTelemetry": False,
    "ai_service_dashcam_osd": False,
    "aiServiceDashcamOSD": False,
    "ai_service_audio_signals": False,
    "aiServiceAudioSignals": False,
    "ai_service_music_detection": False,
    "aiServiceMusicDetection": False,
    "ai_service_audio_summary": False,
    "aiServiceAudioSummary": False,
    "ai_service_caption_writer": False,
    "aiServiceCaptionWriter": False,
    "ai_service_thumbnail_designer": False,
    "aiServiceThumbnailDesigner": False,
    "ai_service_speech_to_text": False,
    "aiServiceSpeechToText": False,
    "ai_service_scene_understanding": False,
    "aiServiceSceneUnderstanding": False,
    "ai_service_frame_inspector": False,
    "aiServiceFrameInspector": False,
    "ai_service_video_analyzer": False,
    "aiServiceVideoAnalyzer": False,
    "tiktok_burn_styled_cover": False,
    "tiktokBurnStyledCover": False,
}


def _fill_missing(settings: Dict[str, Any], defaults: Mapping[str, Any]) -> None:
    for key, val in defaults.items():
        if key not in settings or settings[key] is None:
            settings[key] = val


def apply_upload_baseline_defaults(
    settings: Optional[Dict[str, Any]],
    *,
    tier: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge universal (and optional tier) defaults into *settings* in place."""
    out: Dict[str, Any] = settings if settings is not None else {}
    slug = str(tier or "").strip().lower()
    # Free/entry first so universal baseline does not re-enable heavy AI toggles.
    if slug in ENTRY_TIER_SLUGS:
        _fill_missing(out, FREE_TIER_PROCESSING_DEFAULTS)
    _fill_missing(out, UNIVERSAL_UPLOAD_BASELINE)
    return out


def apply_free_tier_processing_defaults(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Worker hook: conservative defaults for Starter tier when prefs are sparse."""
    return apply_upload_baseline_defaults(settings, tier="free")


def json_safe_value(value: Any) -> Any:
    """Recursively coerce values for ``json.dumps`` / Redis job payloads."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def serialize_job_payload(payload: Mapping[str, Any]) -> str:
    """
    JSON-encode a Redis process/publish lane job body.

    asyncpg rows can leave ``uuid.UUID`` / ``datetime`` in preferences; the stdlib
    encoder rejects those unless every nested value is coerced first.
    """
    return json.dumps(json_safe_value(dict(payload)))


def sanitize_settings_for_job_payload(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Strip DB metadata and return JSON-serializable preferences for Redis enqueue.

    Fixes stale-recovery failures: ``load_user_settings`` can include ``user_id``
    as a native UUID from ``user_settings`` / ``user_preferences`` rows.
    """
    if not settings:
        return {}
    cleaned: Dict[str, Any] = {}
    for key, val in settings.items():
        if key in UPLOAD_PREF_STRIP_KEYS:
            continue
        cleaned[key] = json_safe_value(val)
    return cleaned
