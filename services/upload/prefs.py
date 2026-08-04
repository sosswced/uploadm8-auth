"""Upload init preference merging and snake/camelCase normalization."""

from __future__ import annotations

from typing import Any, Dict

_UPLOAD_PREF_MIRROR_PAIRS = [
    ("thumbnail_studio_engine_enabled", "thumbnailStudioEngineEnabled"),
    ("thumbnail_pikzels_enabled", "thumbnailPikzelsEnabled"),
    ("thumbnail_persona_enabled", "thumbnailPersonaEnabled"),
    ("thumbnail_default_persona_id", "thumbnailDefaultPersonaId"),
    ("thumbnail_persona_strength", "thumbnailPersonaStrength"),
    ("thumbnail_apply_mode", "thumbnailApplyMode"),
    ("thumbnail_ref_persona_mode", "thumbnailRefPersonaMode"),
    ("thumbnail_source_job_id", "thumbnailSourceJobId"),
    ("thumbnail_source_variant_id", "thumbnailSourceVariantId"),
    ("thumbnail_studio_strict", "thumbnailStudioStrict"),
    ("tiktok_post_settings", "tiktokPostSettings"),
    ("blocked_hashtags", "blockedHashtags"),
    ("default_privacy", "defaultPrivacy"),
    ("ai_hashtags_enabled", "aiHashtagsEnabled"),
    ("max_hashtags", "maxHashtags"),
    ("default_vehicle_make_id", "defaultVehicleMakeId"),
    ("default_vehicle_model_id", "defaultVehicleModelId"),
    ("youtube_shorts_copyright_trim", "youtubeShortsCopyrightTrim"),
    ("use_audio_context", "useAudioContext"),
    ("ai_service_music_detection", "aiServiceMusicDetection"),
    ("randomize_caption_creative", "randomizeCaptionCreative"),
    ("caption_creative_pick_mode", "captionCreativePickMode"),
    ("caption_creative_combo_index", "captionCreativeComboIndex"),
    ("caption_creative_vary_style", "captionCreativeVaryStyle"),
    ("caption_creative_vary_tone", "captionCreativeVaryTone"),
    ("caption_creative_vary_voice", "captionCreativeVaryVoice"),
    ("caption_creative_shuffle_seed", "captionCreativeShuffleSeed"),
    ("caption_creative_deck_cursor", "captionCreativeDeckCursor"),
    ("caption_creative_deck_fingerprint", "captionCreativeDeckFingerprint"),
    ("caption_style", "captionStyle"),
    ("caption_tone", "captionTone"),
    ("caption_voice", "captionVoice"),
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
    engine_explicitly_off = use_eng is False
    if use_eng is not None:
        v = bool(use_eng)
        user_prefs["thumbnail_studio_engine_enabled"] = v
        user_prefs["thumbnailStudioEngineEnabled"] = v
        # Worker thumbnail stage treats Pikzels v2 as the studio engine; keep legacy
        # keys aligned when the uploader opts into Aurora / studio for this job.
        if v:
            user_prefs["thumbnail_pikzels_enabled"] = True
            user_prefs["thumbnailPikzelsEnabled"] = True
            # Prefer fail/retry over silent template when the user paid for Pikzels
            # on this batch (account Settings can still force strict off).
            if "thumbnail_studio_strict" not in user_prefs and "thumbnailStudioStrict" not in user_prefs:
                user_prefs["thumbnail_studio_strict"] = True
                user_prefs["thumbnailStudioStrict"] = True
        else:
            # Explicit engine-off on this upload must also disable Pikzels for the job.
            user_prefs["thumbnail_pikzels_enabled"] = False
            user_prefs["thumbnailPikzelsEnabled"] = False
    use_pkz = getattr(data, "thumbnail_use_pikzels", None)
    # Engine-off wins: do not let a later pikzels=true (or server default) re-enable.
    if use_pkz is not None and not engine_explicitly_off:
        v = bool(use_pkz)
        user_prefs["thumbnail_pikzels_enabled"] = v
        user_prefs["thumbnailPikzelsEnabled"] = v
    # Do NOT auto-enable Pikzels just because PIKZELS_API_KEY is set when the
    # client omitted the engine flag — that billed every API upload. Explicit
    # thumbnail_use_studio_engine / thumbnail_use_pikzels (or saved Settings)
    # must opt in. TUP once-per-setup gate relies on the client sending false.

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
        user_prefs["thumbnail_persona_required"] = True
        user_prefs["thumbnailPersonaRequired"] = True

    pst = getattr(data, "thumbnail_persona_strength", None)
    if pst is not None:
        try:
            v = max(0, min(100, int(pst)))
        except (TypeError, ValueError):
            v = 70
        user_prefs["thumbnail_persona_strength"] = v
        user_prefs["thumbnailPersonaStrength"] = v

    from services.thumbnail_apply_mode import (
        bind_source_ids_into_prefs,
        normalize_apply_mode,
        normalize_ref_persona_mode,
    )

    apply_mode = getattr(data, "thumbnail_apply_mode", None)
    if apply_mode is None:
        apply_mode = getattr(data, "thumbnailApplyMode", None)
    if apply_mode is not None and str(apply_mode).strip():
        mode = normalize_apply_mode(apply_mode)
        user_prefs["thumbnail_apply_mode"] = mode
        user_prefs["thumbnailApplyMode"] = mode

    rpm = getattr(data, "thumbnail_ref_persona_mode", None)
    if rpm is None:
        rpm = getattr(data, "thumbnailRefPersonaMode", None)
    if rpm is not None and str(rpm).strip():
        ref_mode = normalize_ref_persona_mode(rpm)
        user_prefs["thumbnail_ref_persona_mode"] = ref_mode
        user_prefs["thumbnailRefPersonaMode"] = ref_mode

    job_id = getattr(data, "thumbnail_source_job_id", None) or getattr(data, "thumbnailSourceJobId", None)
    var_id = getattr(data, "thumbnail_source_variant_id", None) or getattr(
        data, "thumbnailSourceVariantId", None
    )
    if job_id or var_id:
        bind_source_ids_into_prefs(user_prefs, job_id=job_id, variant_id=var_id)

    strict = getattr(data, "thumbnail_studio_strict", None)
    if strict is None:
        strict = getattr(data, "thumbnailStudioStrict", None)
    if strict is not None:
        user_prefs["thumbnail_studio_strict"] = bool(strict)
        user_prefs["thumbnailStudioStrict"] = bool(strict)


def merge_upload_init_caption_creative(user_prefs: Dict[str, Any], data: Any) -> None:
    """Overlay per-upload caption style/tone/voice randomize, locks, and fixed values.

    For cycle/shuffle mode, allocates the next shuffled-deck combo on the upload
    prefs (not file order). Account deck cursor/seed are advanced in-place so the
    caller can persist them back to ``users.preferences``.
    """
    from core.caption_creative import (
        allocate_shuffled_cycle_draw,
        normalize_caption_creative_pick_mode,
        normalize_caption_style,
        normalize_caption_tone,
        normalize_caption_voice,
    )

    mode_raw = getattr(data, "caption_creative_pick_mode", None)
    if mode_raw is None:
        mode_raw = getattr(data, "captionCreativePickMode", None)
    rand_raw = getattr(data, "randomize_caption_creative", None)
    if rand_raw is None:
        rand_raw = getattr(data, "randomizeCaptionCreative", None)

    if mode_raw is not None and str(mode_raw).strip() != "":
        mode = normalize_caption_creative_pick_mode(mode_raw)
    elif rand_raw is not None:
        mode = normalize_caption_creative_pick_mode(rand_raw)
    else:
        mode = None

    if mode is not None:
        user_prefs["caption_creative_pick_mode"] = mode
        user_prefs["captionCreativePickMode"] = mode
        on = mode != "off"
        user_prefs["randomize_caption_creative"] = on
        user_prefs["randomizeCaptionCreative"] = on

    def _merge_vary(attr_snake: str, attr_camel: str, out_snake: str, out_camel: str) -> None:
        raw = getattr(data, attr_snake, None)
        if raw is None:
            raw = getattr(data, attr_camel, None)
        if raw is None:
            return
        b = bool(raw)
        user_prefs[out_snake] = b
        user_prefs[out_camel] = b

    _merge_vary(
        "caption_creative_vary_style",
        "captionCreativeVaryStyle",
        "caption_creative_vary_style",
        "captionCreativeVaryStyle",
    )
    _merge_vary(
        "caption_creative_vary_tone",
        "captionCreativeVaryTone",
        "caption_creative_vary_tone",
        "captionCreativeVaryTone",
    )
    _merge_vary(
        "caption_creative_vary_voice",
        "captionCreativeVaryVoice",
        "caption_creative_vary_voice",
        "captionCreativeVaryVoice",
    )

    # Locked-axis fixed values (and optional batch overrides of Settings defaults).
    style_raw = getattr(data, "caption_style", None)
    if style_raw is None:
        style_raw = getattr(data, "captionStyle", None)
    if style_raw is not None and str(style_raw).strip() != "":
        s = normalize_caption_style(style_raw)
        user_prefs["caption_style"] = s
        user_prefs["captionStyle"] = s

    tone_raw = getattr(data, "caption_tone", None)
    if tone_raw is None:
        tone_raw = getattr(data, "captionTone", None)
    if tone_raw is not None and str(tone_raw).strip() != "":
        t = normalize_caption_tone(tone_raw)
        user_prefs["caption_tone"] = t
        user_prefs["captionTone"] = t

    voice_raw = getattr(data, "caption_voice", None)
    if voice_raw is None:
        voice_raw = getattr(data, "captionVoice", None)
    if voice_raw is not None and str(voice_raw).strip() != "":
        v = normalize_caption_voice(voice_raw)
        user_prefs["caption_voice"] = v
        user_prefs["captionVoice"] = v

    effective = normalize_caption_creative_pick_mode(
        user_prefs.get("captionCreativePickMode") or user_prefs.get("caption_creative_pick_mode")
    )
    if effective == "cycle":
        # Ignore any client-sent combo index — deck draw must not depend on file order.
        user_prefs.pop("caption_creative_combo_index", None)
        user_prefs.pop("captionCreativeComboIndex", None)
        allocate_shuffled_cycle_draw(user_prefs)


async def persist_caption_creative_deck_state(conn: Any, user_id: Any, user_prefs: Dict[str, Any]) -> None:
    """Write shuffled-deck cursor/seed back to ``users.preferences`` after a cycle draw."""
    import json

    from core.caption_creative import normalize_caption_creative_pick_mode

    mode = normalize_caption_creative_pick_mode(
        user_prefs.get("captionCreativePickMode") or user_prefs.get("caption_creative_pick_mode")
    )
    if mode != "cycle":
        return
    seed = user_prefs.get("captionCreativeShuffleSeed") or user_prefs.get("caption_creative_shuffle_seed")
    cursor = user_prefs.get("captionCreativeDeckCursor")
    if cursor is None:
        cursor = user_prefs.get("caption_creative_deck_cursor")
    fingerprint = user_prefs.get("captionCreativeDeckFingerprint") or user_prefs.get(
        "caption_creative_deck_fingerprint"
    )
    if seed is None and cursor is None:
        return
    try:
        raw = await conn.fetchval("SELECT preferences FROM users WHERE id = $1", user_id)
    except Exception:
        return
    prefs: Dict[str, Any] = {}
    if raw:
        if isinstance(raw, str):
            try:
                prefs = json.loads(raw) or {}
            except Exception:
                prefs = {}
        elif isinstance(raw, dict):
            prefs = dict(raw)
    if not isinstance(prefs, dict):
        prefs = {}
    if seed is not None:
        prefs["captionCreativeShuffleSeed"] = prefs["caption_creative_shuffle_seed"] = seed
    if cursor is not None:
        prefs["captionCreativeDeckCursor"] = prefs["caption_creative_deck_cursor"] = int(cursor)
    if fingerprint is not None:
        prefs["captionCreativeDeckFingerprint"] = prefs["caption_creative_deck_fingerprint"] = fingerprint
    try:
        await conn.execute(
            "UPDATE users SET preferences = $1::jsonb, updated_at = NOW() WHERE id = $2",
            json.dumps(prefs, default=str),
            user_id,
        )
    except Exception:
        return


def merge_upload_init_tiktok_post_settings(user_prefs: Dict[str, Any], data: Any) -> None:
    """Persist TikTok export UI choices on the upload row for publish_stage."""
    raw = getattr(data, "tiktok_post_settings", None)
    if raw is None:
        raw = getattr(data, "tiktokPostSettings", None)
    if not isinstance(raw, dict) or not raw:
        return
    user_prefs["tiktok_post_settings"] = raw
    user_prefs["tiktokPostSettings"] = raw
