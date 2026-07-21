"""
Per-service AIC (AI credit) weights for upload billing.

Maps each major backend AI/ML stage to a relative weight. Duration-sensitive
services are multiplied by duration_multiplier(); fixed-cost services are not.

Tune SERVICE_WEIGHTS to match vendor cost + margin — numbers are arbitrary units
that fold into integer AIC debited at presign (same integer the wallet uses).

Aligned with worker pipeline stages (see worker.py header).
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# Relative AIC units per service — anchored so 1 AIC ≈ $0.01 intended retail value.
# Target: ~12–15 AIC for default light 60s job; ~22–25 AIC for full-smart 60s
# (desired all-in AI COGS ~$0.15–$0.25). Tune live via admin Debit weights page.
# Audit bands (Jul 2026 recalibration from Render/GCP/Pikzels/Upstash statements):
#   4–5   minute-metered cloud APIs (twelvelabs, video_intelligence) — opt-in
#   3–5   LLM / heavy gen (caption_llm, thumbnail_ai)
#   1–2   vision / light audio / search
#   1     local CPU / telemetry (negligible API $)
# Studio/Pikzels ops use separate debit tables in services/thumbnail_studio.py
# (2.5× vendor $), not these upload-pipeline weights alone.
SERVICE_WEIGHTS: Dict[str, int] = {
    "twelvelabs": 4,  # Full-video index + Pegasus — opt-in; duration-scaled
    "video_intelligence": 3,  # GCP Video Intelligence — opt-in; duration-scaled
    "caption_llm": 4,  # GPT caption / hashtags / title
    "audio_whisper": 0,  # OpenAI Whisper — AIC-exempt (accuracy ladder); still duration-aware for ops
    "thumbnail_ai": 3,  # Thumbnail gen stack (+12% per extra thumb in compute)
    "vision_google": 2,  # Cloud Vision labels on caption/thumb frames
    "dashcam_osd": 1,  # Vision OCR sweep — opt-in
    "audio_gpt_classify": 1,  # GPT-4o-mini audio summary — fixed
    "audio_acr": 1,  # ACRCloud fingerprint (when configured)
    "thumbnail_recreate_ai": 90,  # Aligns with Pikzels image→thumb ~$0.36 × 2.5
    "persona_consistency": 95,  # Aligns with Pikzels create-persona ~$0.38 × 2.5
    "thumbnail_ctr_ranker": 8,  # Score thumb ~$0.03 × ~2.5
    "marketing_image": 2,  # Product card + optional Pikzels overlay
    "thumbnail_competitor_gap": 8,  # Competitor gap analysis mode
    "audio_yamnet": 1,  # Local AudioSet / YAMNet — CPU only
    "telemetry_trill": 1,  # .map parse + Nominatim — negligible API $
    "trend_intel": 1,  # SerpAPI / YouTube Data — one light search per caption job
}

# Vendor cost scales with clip/audio length — apply duration_multiplier().
DURATION_SCALED: FrozenSet[str] = frozenset(
    {
        "audio_whisper",
        "video_intelligence",
        "twelvelabs",
    }
)

# Services that may appear in enabled_services but must not invent AIC when alone.
# Weight 0 is the billing signal; this set documents product intent for UI/copy.
AIC_BILLING_EXEMPT: FrozenSet[str] = frozenset({"audio_whisper"})

SERVICE_PREF_KEYS: Dict[str, str] = {
    "telemetry_trill": "aiServiceTelemetry",
    "audio_yamnet": "aiServiceAudioSignals",
    "audio_acr": "aiServiceMusicDetection",
    "audio_gpt_classify": "aiServiceAudioSummary",
    "caption_llm": "aiServiceCaptionWriter",
    "thumbnail_ai": "aiServiceThumbnailDesigner",
    "vision_google": "aiServiceFrameInspector",
    "audio_whisper": "aiServiceSpeechToText",
    "video_intelligence": "aiServiceVideoAnalyzer",
    "twelvelabs": "aiServiceSceneUnderstanding",
    "dashcam_osd": "aiServiceDashcamOSD",
}

# ``user_preferences`` column names persisted by save_user_content_preferences.
AI_SERVICE_DB_FIELD_TO_ID: Dict[str, str] = {
    "ai_service_telemetry": "telemetry_trill",
    "ai_service_dashcam_osd": "dashcam_osd",
    "ai_service_audio_signals": "audio_yamnet",
    "ai_service_music_detection": "audio_acr",
    "ai_service_audio_summary": "audio_gpt_classify",
    "ai_service_caption_writer": "caption_llm",
    "ai_service_thumbnail_designer": "thumbnail_ai",
    "ai_service_speech_to_text": "audio_whisper",
    "ai_service_scene_understanding": "twelvelabs",
    "ai_service_frame_inspector": "vision_google",
    "ai_service_video_analyzer": "video_intelligence",
}

PREF_KEY_TO_SERVICE_ID: Dict[str, str] = {v: k for k, v in SERVICE_PREF_KEYS.items()}


def clamp_ai_service_db_fields(
    allowed_services: Optional[FrozenSet[str]],
    fields: Dict[str, bool],
) -> Dict[str, bool]:
    """Force AI service toggles off when the user's tier does not allow the service."""
    if allowed_services is None:
        return dict(fields)
    out = dict(fields)
    for db_field, svc_id in AI_SERVICE_DB_FIELD_TO_ID.items():
        if db_field in out and svc_id not in allowed_services:
            out[db_field] = False
    return out


def clamp_ai_service_pref_payload(
    allowed_services: Optional[FrozenSet[str]],
    prefs: Dict[str, Any],
) -> Dict[str, Any]:
    """Clamp camelCase/snakeCase AI service keys in a preferences dict."""
    if allowed_services is None:
        return prefs
    out = dict(prefs)
    for pref_key, svc_id in PREF_KEY_TO_SERVICE_ID.items():
        snake = pref_key[0].lower() + "".join(
            ("_" + c.lower()) if c.isupper() else c for c in pref_key[1:]
        )
        for key in (pref_key, snake):
            if key in out and svc_id not in allowed_services:
                out[key] = False
    return out

# User-facing, provider-agnostic labels/help text.
SERVICE_PUBLIC_META: Dict[str, Dict[str, str]] = {
    "telemetry_trill": {
        "label": "Telemetry Insights",
        "description": "Uses route/speed map data to improve captions and context when telemetry is uploaded.",
    },
    "audio_yamnet": {
        "label": "Audio Event Detection",
        "description": "Detects sounds like speech, music, engines, crowd noise, and ambient events.",
    },
    "audio_acr": {
        "label": "Music Detection",
        "description": "Identifies likely music tracks to improve metadata and copyright awareness.",
    },
    "thumbnail_recreate_ai": {
        "label": "Thumbnail Recreate",
        "description": "Analyzes reference thumbnails and generates style-matched variants.",
    },
    "persona_consistency": {
        "label": "Persona Consistency",
        "description": "Applies creator persona profile for stable face, lighting, and expression continuity.",
    },
    "thumbnail_ctr_ranker": {
        "label": "Thumbnail CTR Ranker",
        "description": "Scores candidate variants for mobile readability and likely click performance.",
    },
    "thumbnail_competitor_gap": {
        "label": "Competitor Gap Analysis",
        "description": "Benchmarks visual strategy against niche competitors and suggests stronger hooks.",
    },
    "audio_gpt_classify": {
        "label": "Audio Content Summary",
        "description": "Summarizes audio themes, pacing, and category signals for better outputs.",
    },
    "caption_llm": {
        "label": "Caption and Hashtag Writer",
        "description": "Generates titles, captions, and hashtag sets from video + context.",
    },
    "thumbnail_ai": {
        "label": "Thumbnail Designer",
        "description": "Creates and styles thumbnail options for higher click-through.",
    },
    "vision_google": {
        "label": "Frame Inspector",
        "description": "Reads key visual details (faces/text/objects) from representative frames.",
    },
    "dashcam_osd": {
        "label": "Dashcam OSD Reader",
        "description": "Reads on-screen date, time, GPS, speed, and driver name already in your clip (OCR only—no new overlay burned in).",
    },
    "audio_whisper": {
        "label": "Speech-to-Text Transcript",
        "description": (
            "Transcribes speech for accurate context and stronger caption quality. "
            "Included at no extra AIC — enable freely for talking-head and voiceover clips."
        ),
    },
    "video_intelligence": {
        "label": "Video Analyzer",
        "description": "Analyzes full video structure and actions to improve semantic understanding.",
    },
    "twelvelabs": {
        "label": "Scene Understanding",
        "description": "Deep understanding of scenes and narrative flow across the whole clip.",
    },
    "trend_intel": {
        "label": "Search Title Trends",
        "description": "Pulls a short sample of recent YouTube-style titles for the niche so captions align with real search language.",
    },
    "marketing_image": {
        "label": "Marketing Product Card",
        "description": "Generates per-user product card art and optional headline overlay for marketing surfaces.",
    },
}


def merge_service_weights_from_db(db_map: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """
    Code defaults overlaid with DB rows from ``billing_service_weights``.
    Unknown keys in db_map are ignored; missing DB rows use ``SERVICE_WEIGHTS``.
    """
    out = dict(SERVICE_WEIGHTS)
    if not db_map:
        return out
    for k, v in db_map.items():
        sk = str(k or "").strip()
        if sk not in SERVICE_WEIGHTS:
            continue
        try:
            out[sk] = max(0, min(5000, int(v)))
        except (TypeError, ValueError):
            continue
    return out


def _aic_raw_parts(
    enabled: Set[str],
    duration_seconds: float,
    max_caption_frames: int,
    num_thumbnails: int,
    weights: Dict[str, int],
) -> Dict[str, float]:
    """Non-negative float contribution per service (caption_llm includes frame surcharge)."""
    dm = duration_multiplier(duration_seconds)
    parts: Dict[str, float] = {}
    for svc in enabled:
        w = float(weights.get(svc, 0))
        if w <= 0:
            continue
        if svc == "thumbnail_ai":
            n = max(1, int(num_thumbnails))
            w *= 1.0 + 0.12 * float(max(0, n - 1))
        if svc in DURATION_SCALED:
            parts[svc] = w * dm
        else:
            parts[svc] = w
    if "caption_llm" in enabled:
        parts["caption_llm"] = parts.get("caption_llm", 0.0) + float(_caption_frame_surcharge(max_caption_frames))
    return parts


def _allocate_aic_integers(parts: Dict[str, float], target: int) -> Dict[str, int]:
    """Largest-remainder allocation so values sum exactly to ``target``."""
    if target <= 0 or not parts:
        return {}
    wsum = sum(parts.values())
    if wsum <= 0:
        return {k: 0 for k in parts}
    pref = {k: (target * parts[k] / wsum) for k in parts}
    floors = {k: int(math.floor(pref[k])) for k in parts}
    rem = target - sum(floors.values())
    fracs = sorted(((pref[k] - floors[k], k) for k in parts), reverse=True)
    for i in range(max(0, rem)):
        floors[fracs[i % len(fracs)][1]] += 1
    return floors


def build_billing_breakdown(
    *,
    put_breakdown: Dict[str, Any],
    aic_total: int,
    aic_by_service: Dict[str, int],
    duration_seconds_estimated: float,
    duration_multiplier_val: float,
    enabled_services: List[str],
) -> Dict[str, Any]:
    """Canonical JSON for ``uploads.billing_breakdown`` and ``token_ledger.meta``."""
    return {
        "schema_version": 1,
        "put": dict(put_breakdown),
        "aic": {
            "total": int(aic_total),
            "duration_seconds_estimated": round(float(duration_seconds_estimated), 2),
            "duration_multiplier": round(float(duration_multiplier_val), 4),
            "enabled_services": list(enabled_services),
            "by_service": dict(sorted(aic_by_service.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
    }


def service_catalog(weights: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """Provider-agnostic service catalog for UI/settings screens."""
    wsrc = weights if weights is not None else SERVICE_WEIGHTS
    rows: List[Dict[str, Any]] = []
    for service_id, weight in sorted(wsrc.items(), key=lambda kv: kv[1], reverse=True):
        if service_id not in SERVICE_WEIGHTS:
            continue
        meta = SERVICE_PUBLIC_META.get(service_id, {})
        rows.append(
            {
                "id": service_id,
                "pref_key": SERVICE_PREF_KEYS.get(service_id),
                "label": meta.get("label", service_id),
                "description": meta.get("description", ""),
                "weight": int(weight),
                "duration_scaled": bool(service_id in DURATION_SCALED),
            }
        )
    return rows


def _pref_true(prefs: Dict[str, Any], key: str, default: bool = True) -> bool:
    """Read camelCase/snake_case variants from settings payloads."""
    if not key:
        return default
    snake = key[0].lower()
    for ch in key[1:]:
        snake += ("_" + ch.lower()) if ch.isupper() else ch
    variants = [key, snake, snake.replace("_o_s_d", "_osd")]
    raw = default
    for variant in variants:
        if variant in prefs:
            raw = prefs[variant]
            break
    if isinstance(raw, str):
        return raw.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(raw)


def user_pref_ai_service_enabled(
    prefs: Dict[str, Any],
    logical_service_id: str,
    *,
    default: bool = True,
    allowed_services: Optional[Set[str]] = None,
) -> bool:
    """
    Worker/runtime gate aligned with billing SERVICE_PREF_KEYS.

    logical_service_id matches SERVICE_WEIGHTS / SERVICE_PREF_KEYS keys, e.g.
    ``telemetry_trill``, ``caption_llm``, ``thumbnail_ai``, ``audio_whisper``,
    ``vision_google`` (Frame Inspector in UI).
    """
    if allowed_services is not None and logical_service_id not in allowed_services:
        return False
    pref_key = SERVICE_PREF_KEYS.get(logical_service_id)
    if not pref_key:
        if allowed_services is not None:
            return logical_service_id in allowed_services
        return default
    return _pref_true(prefs, pref_key, default)


def _env_bool(key: str, default: str = "false") -> bool:
    return os.environ.get(key, default).lower() == "true"


def billing_env_from_os() -> Dict[str, bool]:
    """Same toggles the worker uses — keep defaults in sync with stage modules."""
    return {
        "AUDIO_STAGE_ENABLED": _env_bool("AUDIO_STAGE_ENABLED", "true"),
        "YAMNET_ENABLED": _env_bool("YAMNET_ENABLED", "true"),
        "VISION_STAGE_ENABLED": _env_bool("VISION_STAGE_ENABLED", "true"),
        "ACRCLOUD_CONFIGURED": bool(
            (os.environ.get("ACRCLOUD_ACCESS_KEY") or os.environ.get("ACR_ACCESS_KEY") or "").strip()
            and (
                os.environ.get("ACRCLOUD_ACCESS_SECRET")
                or os.environ.get("ACR_ACCESS_SECRET")
                or os.environ.get("ACRCLOUD_SECRET_KEY")
                or ""
            ).strip()
            and (os.environ.get("ACRCLOUD_HOST") or os.environ.get("ACR_HOST") or "").strip()
        ),
        "TREND_INTEL_AVAILABLE": bool(
            ((os.environ.get("SERPAPI_API_KEY") or "").strip() or (os.environ.get("YOUTUBE_DATA_API_KEY") or "").strip())
            and (os.environ.get("TREND_INTEL_DISABLED") or "").strip().lower()
            not in ("1", "true", "yes", "on")
        ),
    }


def estimate_duration_seconds(
    file_size: Optional[int],
    duration_hint: Optional[float],
) -> float:
    """
    Best-effort duration before the file is probed on the worker.
    Prefer client-reported duration; else rough estimate from size; else 2 min default.
    """
    if duration_hint is not None and duration_hint > 0:
        return float(min(max(duration_hint, 5.0), 3600.0))
    if file_size and int(file_size) > 0:
        # ~2.8 Mbps average effective rate → bytes/sec upper bound for length guess
        est = int(file_size) / (350 * 1024)
        return float(max(20.0, min(est, 600.0)))
    return 120.0


def duration_multiplier(duration_seconds: float) -> float:
    """
    Shorter videos use less AIC on duration-scaled services.
    ~1.0 at 1 min, ~1.35 at 3 min, ~1.75 at 10 min (Whisper / VI / Twelve Labs scale with length).
    """
    minutes = max(0.25, float(duration_seconds) / 60.0)
    raw = 0.58 + 0.42 * math.pow(minutes, 0.55)
    return float(max(0.78, min(2.1, raw)))


def _caption_frame_surcharge(caption_frames: int) -> int:
    """Extra AIC for analyzing many frames (matches legacy tiers in entitlements)."""
    cf = int(caption_frames)
    if cf <= 6:
        return 0
    if cf <= 12:
        return 1
    if cf <= 24:
        return 2
    return 3


def resolve_enabled_ai_services(
    *,
    can_ai: bool,
    user_prefs: Dict[str, Any],
    use_ai_request: bool,
    has_telemetry: bool,
    env: Dict[str, bool],
    tier_allowed: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Decide which logical services participate in this upload for billing.

    use_ai_request: client checked "use AI" for this upload — counts as wanting
    caption + thumbnail stack unless user_prefs explicitly contradict (we still
    respect auto_* flags for audio).
    """
    out: Set[str] = set()
    if not can_ai:
        return out

    want_caption = bool(
        user_prefs.get("auto_captions")
        or user_prefs.get("autoCaptions")
        or user_prefs.get("ai_hashtags_enabled")
        or user_prefs.get("aiHashtagsEnabled")
        or use_ai_request
    )
    want_thumb = bool(
        user_prefs.get("auto_thumbnails")
        or user_prefs.get("autoThumbnails")
        or use_ai_request
    )
    use_audio = bool(user_prefs.get("use_audio_context", user_prefs.get("useAudioContext", True)))
    transcribe = bool(user_prefs.get("audio_transcription", user_prefs.get("audioTranscription", True)))

    if has_telemetry and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("telemetry_trill", ""), True):
        out.add("telemetry_trill")

    audio_on = env.get("AUDIO_STAGE_ENABLED", True) and use_audio
    if audio_on:
        if _pref_true(user_prefs, SERVICE_PREF_KEYS.get("audio_gpt_classify", ""), True):
            out.add("audio_gpt_classify")
        if env.get("YAMNET_ENABLED", True) and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("audio_yamnet", ""), True):
            out.add("audio_yamnet")
        if transcribe and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("audio_whisper", ""), False):
            out.add("audio_whisper")
        if env.get("ACRCLOUD_CONFIGURED") and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("audio_acr", ""), True):
            out.add("audio_acr")

    # Frame inspector (Vision) — on by default for caption/thumb quality; user can disable.
    vision_on = (
        env.get("VISION_STAGE_ENABLED", True)
        and (want_caption or want_thumb)
        and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("vision_google", ""), True)
    )
    if vision_on:
        out.add("vision_google")

    # Twelve Labs + Video Intelligence are opt-in (heavy $/min). Defaults off in baseline.
    if want_caption and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("twelvelabs", ""), False):
        out.add("twelvelabs")

    if (want_caption or want_thumb) and _pref_true(
        user_prefs, SERVICE_PREF_KEYS.get("video_intelligence", ""), False
    ):
        out.add("video_intelligence")

    if want_thumb and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("thumbnail_ai", ""), True):
        out.add("thumbnail_ai")

    if want_caption and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("caption_llm", ""), True):
        out.add("caption_llm")
        # SerpAPI / YouTube title sample for M8 — same user gate as caption writer.
        if env.get("TREND_INTEL_AVAILABLE", False):
            out.add("trend_intel")

    # Dashcam OSD OCR — opt-in; bill only when Vision path is on and pref allows.
    if vision_on and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("dashcam_osd", ""), False):
        out.add("dashcam_osd")

    if tier_allowed is not None:
        out = {svc for svc in out if svc in tier_allowed}
    return out


def effective_num_thumbnails(
    ent_max_thumbnails: int,
    user_prefs: Dict[str, Any],
    use_ai_request: bool,
    thumbnail_count: Optional[int],
) -> int:
    """PUT + thumbnail_ai billing: prefer explicit count, else prefs / use_ai."""
    cap = max(1, int(ent_max_thumbnails or 1))
    if thumbnail_count is not None:
        return max(1, min(int(thumbnail_count), cap))
    if use_ai_request or user_prefs.get("auto_thumbnails") or user_prefs.get("autoThumbnails"):
        return cap
    return 1


def _fallback_billable_part(enabled: Set[str], weights: Dict[str, int]) -> Dict[str, float]:
    """When all raw parts are zero, charge 1 AIC on a positive-weight service only."""
    billable = [
        s
        for s in sorted(enabled)
        if float(weights.get(s, 0)) > 0 and s not in AIC_BILLING_EXEMPT
    ]
    if not billable:
        return {}
    return {billable[0]: 1.0}


def compute_aic_service_charge(
    *,
    enabled: Set[str],
    duration_seconds: float,
    max_caption_frames: int,
    num_thumbnails: int,
    weights: Dict[str, int],
) -> int:
    """Return integer AIC for the enabled service set (ceil of summed float parts)."""
    if not enabled:
        return 0
    parts = _aic_raw_parts(
        enabled, duration_seconds, max_caption_frames, num_thumbnails, weights
    )
    if enabled and sum(parts.values()) <= 0:
        parts = _fallback_billable_part(enabled, weights)
        if not parts:
            return 0
    total = sum(parts.values())
    return max(1, int(math.ceil(total))) if total > 0 else 0


def compute_aic_breakdown(
    *,
    can_ai: bool,
    user_prefs: Dict[str, Any],
    use_ai_request: bool,
    has_telemetry: bool,
    duration_seconds: float = 0.0,
    file_size: Optional[int] = None,
    duration_hint: Optional[float] = None,
    max_caption_frames: int,
    num_thumbnails: int,
    env: Optional[Dict[str, bool]] = None,
    service_weights_map: Optional[Dict[str, Any]] = None,
    tier_allowed: Optional[Set[str]] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Returns (aic_integer, debug_dict) for API responses and logging.
    """
    env = env or billing_env_from_os()
    merged_w = merge_service_weights_from_db(service_weights_map)
    if not can_ai:
        return 0, {
            "enabled_services": [],
            "duration_seconds_estimated": 0.0,
            "duration_multiplier": 1.0,
            "num_thumbnails": num_thumbnails,
            "service_weights": {},
            "aic_by_service": {},
            "reason": "plan_cannot_ai",
        }

    hint = duration_hint if duration_hint is not None else (duration_seconds if duration_seconds > 0 else None)
    dur = estimate_duration_seconds(file_size, hint)
    dm = duration_multiplier(dur)

    enabled = resolve_enabled_ai_services(
        can_ai=True,
        user_prefs=user_prefs,
        use_ai_request=use_ai_request,
        has_telemetry=has_telemetry,
        env=env,
        tier_allowed=tier_allowed,
    )
    if not enabled:
        return 0, {
            "enabled_services": [],
            "duration_seconds_estimated": round(dur, 2),
            "duration_multiplier": round(dm, 4),
            "num_thumbnails": num_thumbnails,
            "service_weights": {},
            "aic_by_service": {},
            "reason": "no_ai_services_selected",
        }

    parts = _aic_raw_parts(enabled, dur, max_caption_frames, num_thumbnails, merged_w)
    if enabled and sum(parts.values()) <= 0:
        parts = _fallback_billable_part(enabled, merged_w)
    aic = compute_aic_service_charge(
        enabled=enabled,
        duration_seconds=dur,
        max_caption_frames=max_caption_frames,
        num_thumbnails=num_thumbnails,
        weights=merged_w,
    )
    by_svc = _allocate_aic_integers(parts, aic) if parts else {}

    debug = {
        "enabled_services": sorted(enabled),
        "duration_seconds_estimated": round(dur, 2),
        "duration_multiplier": round(dm, 4),
        "num_thumbnails": num_thumbnails,
        "service_weights": {k: int(merged_w.get(k, 0)) for k in sorted(enabled) if k in merged_w},
        "aic_by_service": dict(sorted(by_svc.items(), key=lambda kv: (-kv[1], kv[0]))),
        "service_catalog": service_catalog(merged_w),
    }
    return aic, debug


def compute_presign_put_aic_costs(
    ent: Any,
    *,
    num_publish_targets: int,
    file_size: Optional[int],
    duration_hint: Optional[float],
    has_telemetry: bool,
    use_ai_checkbox: bool,
    user_prefs: Dict[str, Any],
    num_thumbnails_override: Optional[int] = None,
    service_weights_map: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    PUT + AIC reservation for upload presign / wallet (aligned with pipeline prefs).

    Returns ``(put_total, aic_total, billing_breakdown)`` for DB + ledger meta.

    """
    from stages.entitlements import PRIORITY_QUEUE_CLASSES, compute_put_breakdown

    cap = max(1, int(getattr(ent, "max_thumbnails", 1) or 1))
    if num_thumbnails_override is not None:
        num_thumbs = max(1, min(int(num_thumbnails_override), cap))
    else:
        num_thumbs = effective_num_thumbnails(
            getattr(ent, "max_thumbnails", 1) or 1,
            user_prefs,
            use_ai_checkbox,
            None,
        )

    is_priority = getattr(ent, "priority_class", "") in PRIORITY_QUEUE_CLASSES
    put_break = compute_put_breakdown(
        num_platforms=num_publish_targets,
        is_priority=is_priority,
        num_thumbnails=num_thumbs,
    )
    put = int(put_break["total"])

    if not getattr(ent, "can_ai", False):
        bd = build_billing_breakdown(
            put_breakdown=put_break,
            aic_total=0,
            aic_by_service={},
            duration_seconds_estimated=0.0,
            duration_multiplier_val=1.0,
            enabled_services=[],
        )
        return put, 0, bd

    try:
        emax = int(getattr(ent, "max_caption_frames", 6) or 6)
    except (TypeError, ValueError):
        emax = 6
    ufc = user_prefs.get("captionFrameCount") or user_prefs.get("caption_frame_count") or emax
    try:
        user_fc_int = int(ufc)
    except (TypeError, ValueError):
        user_fc_int = emax
    eff_caption_frames = max(2, min(user_fc_int, emax, 12))

    tier_allowed = getattr(ent, "allowed_ai_services", None)
    if tier_allowed is not None and not isinstance(tier_allowed, set):
        tier_allowed = set(tier_allowed)

    aic, dbg = compute_aic_breakdown(
        can_ai=True,
        user_prefs=user_prefs,
        use_ai_request=use_ai_checkbox,
        has_telemetry=has_telemetry,
        file_size=file_size,
        duration_hint=duration_hint,
        max_caption_frames=eff_caption_frames,
        num_thumbnails=num_thumbs,
        env=billing_env_from_os(),
        service_weights_map=service_weights_map,
        tier_allowed=tier_allowed,
    )
    hint = duration_hint if duration_hint is not None else None
    dur = estimate_duration_seconds(file_size, hint)
    dm = float(dbg.get("duration_multiplier") or duration_multiplier(dur))
    by_svc = dict(dbg.get("aic_by_service") or {})
    bd = build_billing_breakdown(
        put_breakdown=put_break,
        aic_total=int(aic),
        aic_by_service=by_svc,
        duration_seconds_estimated=float(dbg.get("duration_seconds_estimated") or dur),
        duration_multiplier_val=dm,
        enabled_services=list(dbg.get("enabled_services") or []),
    )
    return put, int(aic), bd
