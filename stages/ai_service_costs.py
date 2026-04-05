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
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple, List

# Relative AIC units per service — ordered by typical vendor $ to recoup (higher = costlier).
# Full-clip / minute-metered APIs sit at the top; local or near-free at the bottom.
SERVICE_WEIGHTS: Dict[str, int] = {
    "twelvelabs": 28,  # Full-video index + Pegasus — usually highest $/min
    "video_intelligence": 24,  # GCP Video Intelligence — billed per minute of video
    "caption_llm": 22,  # GPT-4o caption / hashtags / title — large token spend
    "audio_whisper": 18,  # OpenAI Whisper — $/minute; also in DURATION_SCALED
    "thumbnail_ai": 16,  # Playwright / rembg / gen stack (+12% per extra thumb in compute)
    "vision_google": 12,  # Cloud Vision — per-image, multi-feature on one frame
    "audio_hume": 10,  # Hume voice emotion API
    "audio_gpt_classify": 7,  # GPT-4o-mini — small prompt; fixed cost (not duration-scaled)
    "audio_acr": 5,  # ACRCloud fingerprint (when configured)
    "thumbnail_recreate_ai": 14,  # URL-to-thumbnail recreate + prompt synthesis
    "persona_consistency": 10,  # Persona profile consistency checks
    "thumbnail_ctr_ranker": 6,  # Pre-publish scoring + suggestions
    "thumbnail_competitor_gap": 4,  # Competitor gap analysis mode
    "audio_yamnet": 2,  # Local AudioSet / YAMNet — CPU only
    "telemetry_trill": 1,  # .map parse + Nominatim geocode — negligible API $
}

# Vendor cost scales with clip/audio length — apply duration_multiplier().
DURATION_SCALED: FrozenSet[str] = frozenset(
    {
        "audio_whisper",
        "audio_hume",
        "video_intelligence",
        "twelvelabs",
    }
)

# Per-service user setting keys (frontend/API-facing; defaults = enabled).
SERVICE_PREF_KEYS: Dict[str, str] = {
    "telemetry_trill": "aiServiceTelemetry",
    "audio_yamnet": "aiServiceAudioSignals",
    "audio_acr": "aiServiceMusicDetection",
    "audio_gpt_classify": "aiServiceAudioSummary",
    "audio_hume": "aiServiceEmotionSignals",
    "caption_llm": "aiServiceCaptionWriter",
    "thumbnail_ai": "aiServiceThumbnailDesigner",
    "vision_google": "aiServiceFrameInspector",
    "audio_whisper": "aiServiceSpeechToText",
    "video_intelligence": "aiServiceVideoAnalyzer",
    "twelvelabs": "aiServiceSceneUnderstanding",
}

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
    "audio_hume": {
        "label": "Emotion Signals",
        "description": "Estimates emotional tone from voice to refine caption tone and hooks.",
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
    "audio_whisper": {
        "label": "Speech-to-Text Transcript",
        "description": "Transcribes speech for accurate context and stronger caption quality.",
    },
    "video_intelligence": {
        "label": "Video Analyzer",
        "description": "Analyzes full video structure and actions to improve semantic understanding.",
    },
    "twelvelabs": {
        "label": "Scene Understanding",
        "description": "Deep understanding of scenes and narrative flow across the whole clip.",
    },
}


def service_catalog() -> List[Dict[str, Any]]:
    """Provider-agnostic service catalog for UI/settings screens."""
    rows: List[Dict[str, Any]] = []
    for service_id, weight in sorted(SERVICE_WEIGHTS.items(), key=lambda kv: kv[1], reverse=True):
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
    raw = prefs.get(key, prefs.get(snake, default))
    return bool(raw)


def _env_bool(key: str, default: str = "false") -> bool:
    return os.environ.get(key, default).lower() == "true"


def billing_env_from_os() -> Dict[str, bool]:
    """Same toggles the worker uses — keep defaults in sync with stage modules."""
    return {
        "AUDIO_STAGE_ENABLED": _env_bool("AUDIO_STAGE_ENABLED", "true"),
        "YAMNET_ENABLED": _env_bool("YAMNET_ENABLED", "true"),
        "HUME_ENABLED": _env_bool("HUME_ENABLED", "true"),
        "VISION_STAGE_ENABLED": _env_bool("VISION_STAGE_ENABLED", "true"),
        "TWELVELABS_ENABLED": _env_bool("TWELVELABS_ENABLED", "false"),
        "VIDEO_INTELLIGENCE_ENABLED": _env_bool("VIDEO_INTELLIGENCE_ENABLED", "false"),
        "ACRCLOUD_CONFIGURED": bool(
            (os.environ.get("ACRCLOUD_ACCESS_KEY") or os.environ.get("ACR_ACCESS_KEY") or "").strip()
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
        or user_prefs.get("ai_hashtags_enabled")
        or use_ai_request
    )
    want_thumb = bool(user_prefs.get("auto_thumbnails") or use_ai_request)
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
        if env.get("HUME_ENABLED", True) and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("audio_hume", ""), True):
            out.add("audio_hume")
        if transcribe and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("audio_whisper", ""), True):
            out.add("audio_whisper")
        if env.get("ACRCLOUD_CONFIGURED") and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("audio_acr", ""), True):
            out.add("audio_acr")

    vision_on = env.get("VISION_STAGE_ENABLED", True) and (want_caption or want_thumb)
    if vision_on and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("vision_google", ""), True):
        out.add("vision_google")

    if env.get("TWELVELABS_ENABLED") and want_caption and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("twelvelabs", ""), True):
        out.add("twelvelabs")

    if env.get("VIDEO_INTELLIGENCE_ENABLED") and (want_caption or want_thumb) and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("video_intelligence", ""), True):
        out.add("video_intelligence")

    if want_thumb and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("thumbnail_ai", ""), True):
        out.add("thumbnail_ai")

    if want_caption and _pref_true(user_prefs, SERVICE_PREF_KEYS.get("caption_llm", ""), True):
        out.add("caption_llm")

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
    if use_ai_request or user_prefs.get("auto_thumbnails"):
        return cap
    return 1


def compute_aic_service_charge(
    *,
    enabled: Set[str],
    duration_seconds: float,
    max_caption_frames: int,
    num_thumbnails: int,
) -> int:
    """Return integer AIC for the enabled service set."""
    if not enabled:
        return 0

    dm = duration_multiplier(duration_seconds)
    total = 0.0

    for svc in enabled:
        w = float(SERVICE_WEIGHTS.get(svc, 0))
        if w <= 0:
            continue
        if svc == "thumbnail_ai":
            n = max(1, int(num_thumbnails))
            # Scale sublinearly with extra frames/styles — +12% per thumb beyond first
            w *= 1.0 + 0.12 * float(max(0, n - 1))
        if svc in DURATION_SCALED:
            total += w * dm
        else:
            total += w

    if "caption_llm" in enabled:
        total += float(_caption_frame_surcharge(max_caption_frames))

    return max(1, int(math.ceil(total)))


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
) -> Tuple[int, Dict[str, Any]]:
    """
    Returns (aic_integer, debug_dict) for API responses and logging.
    """
    env = env or billing_env_from_os()
    if not can_ai:
        return 0, {
            "enabled_services": [],
            "duration_seconds_estimated": 0.0,
            "duration_multiplier": 1.0,
            "num_thumbnails": num_thumbnails,
            "service_weights": {},
            "reason": "plan_cannot_ai",
        }

    hint = duration_hint if duration_hint is not None else (duration_seconds if duration_seconds > 0 else None)
    dur = estimate_duration_seconds(file_size, hint)

    enabled = resolve_enabled_ai_services(
        can_ai=True,
        user_prefs=user_prefs,
        use_ai_request=use_ai_request,
        has_telemetry=has_telemetry,
        env=env,
    )
    if not enabled:
        return 0, {
            "enabled_services": [],
            "duration_seconds_estimated": round(dur, 2),
            "duration_multiplier": round(duration_multiplier(dur), 4),
            "num_thumbnails": num_thumbnails,
            "service_weights": {},
            "reason": "no_ai_services_selected",
        }

    aic = compute_aic_service_charge(
        enabled=enabled,
        duration_seconds=dur,
        max_caption_frames=max_caption_frames,
        num_thumbnails=num_thumbnails,
    )

    debug = {
        "enabled_services": sorted(enabled),
        "duration_seconds_estimated": round(dur, 2),
        "duration_multiplier": round(duration_multiplier(dur), 4),
        "num_thumbnails": num_thumbnails,
        "service_weights": {k: SERVICE_WEIGHTS[k] for k in sorted(enabled) if k in SERVICE_WEIGHTS},
        "service_catalog": service_catalog(),
    }
    return aic, debug
