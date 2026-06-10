"""Upload init preference merging and snake/camelCase normalization."""

from __future__ import annotations

from typing import Any, Dict

_UPLOAD_PREF_MIRROR_PAIRS = [
    ("thumbnail_studio_engine_enabled", "thumbnailStudioEngineEnabled"),
    ("thumbnail_pikzels_enabled", "thumbnailPikzelsEnabled"),
    ("thumbnail_persona_enabled", "thumbnailPersonaEnabled"),
    ("thumbnail_default_persona_id", "thumbnailDefaultPersonaId"),
    ("thumbnail_persona_strength", "thumbnailPersonaStrength"),
    ("tiktok_post_settings", "tiktokPostSettings"),
    ("blocked_hashtags", "blockedHashtags"),
    ("default_privacy", "defaultPrivacy"),
    ("ai_hashtags_enabled", "aiHashtagsEnabled"),
    ("max_hashtags", "maxHashtags"),
    ("default_vehicle_make_id", "defaultVehicleMakeId"),
    ("default_vehicle_model_id", "defaultVehicleModelId"),
]


def normalize_user_prefs_snapshot(user_prefs: Dict[str, Any]) -> None:
    """Ensure snake_case and camelCase aliases exist on an upload prefs snapshot."""
    for snake, camel in _UPLOAD_PREF_MIRROR_PAIRS:
        if snake in user_prefs and user_prefs[snake] is not None:
            user_prefs.setdefault(camel, user_prefs[snake])
        elif camel in user_prefs and user_prefs[camel] is not None:
            user_prefs.setdefault(snake, user_prefs[camel])


def merge_upload_init_thumbnail_preferences(user_prefs: Dict[str, Any], data: Any) -> None:
    """Overlay presign-body thumbnail toggles onto the snapshot stored on ``uploads.user_preferences``."""
    use_eng = getattr(data, "thumbnail_use_studio_engine", None)
    if use_eng is not None:
        v = bool(use_eng)
        user_prefs["thumbnail_studio_engine_enabled"] = v
        user_prefs["thumbnailStudioEngineEnabled"] = v
        # Worker thumbnail stage treats Pikzels v2 as the studio engine; keep legacy
        # keys aligned when the uploader opts into Aurora / studio for this job.
        if v:
            user_prefs["thumbnail_pikzels_enabled"] = True
            user_prefs["thumbnailPikzelsEnabled"] = True
    use_pkz = getattr(data, "thumbnail_use_pikzels", None)
    if use_pkz is not None:
        v = bool(use_pkz)
        user_prefs["thumbnail_pikzels_enabled"] = v
        user_prefs["thumbnailPikzelsEnabled"] = v
    elif use_eng is None:
        # Presign default: when the server has Pikzels configured, opt uploads into
        # studio unless the client explicitly disabled engine/pikzels on the body.
        try:
            from stages.pikzels_api import studio_renderer_enabled

            if studio_renderer_enabled():
                user_prefs["thumbnail_pikzels_enabled"] = True
                user_prefs["thumbnailPikzelsEnabled"] = True
                user_prefs.setdefault("thumbnail_studio_engine_enabled", True)
                user_prefs.setdefault("thumbnailStudioEngineEnabled", True)
        except Exception:
            pass

    use_per = getattr(data, "thumbnail_use_persona", None)
    if use_per is True:
        user_prefs["thumbnail_persona_enabled"] = True
        user_prefs["thumbnailPersonaEnabled"] = True
    elif use_per is False:
        user_prefs["thumbnail_persona_enabled"] = False
        user_prefs["thumbnailPersonaEnabled"] = False
        user_prefs.pop("thumbnail_default_persona_id", None)
        user_prefs.pop("thumbnailDefaultPersonaId", None)

    pid = getattr(data, "thumbnail_persona_id", None)
    if pid and str(pid).strip():
        s = str(pid).strip()
        user_prefs["thumbnail_default_persona_id"] = s
        user_prefs["thumbnailDefaultPersonaId"] = s
        user_prefs["thumbnail_persona_enabled"] = True
        user_prefs["thumbnailPersonaEnabled"] = True

    pst = getattr(data, "thumbnail_persona_strength", None)
    if pst is not None:
        try:
            v = max(0, min(100, int(pst)))
        except (TypeError, ValueError):
            v = 70
        user_prefs["thumbnail_persona_strength"] = v
        user_prefs["thumbnailPersonaStrength"] = v


def merge_upload_init_tiktok_post_settings(user_prefs: Dict[str, Any], data: Any) -> None:
    """Persist TikTok export UI choices on the upload row for publish_stage."""
    raw = getattr(data, "tiktok_post_settings", None)
    if raw is None:
        raw = getattr(data, "tiktokPostSettings", None)
    if not isinstance(raw, dict) or not raw:
        return
    user_prefs["tiktok_post_settings"] = raw
    user_prefs["tiktokPostSettings"] = raw
