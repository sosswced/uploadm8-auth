"""
Upload preference combinatorics — tier baselines, dependency graph, exhaustive generators.

Used by ``tests/test_settings_combinatorics.py`` and ``tests/test_settings_playwright.py``
to verify every boolean toggle combination normalizes, serializes, and respects
parent→child preference rules without breaking the upload pipeline.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, Optional, Tuple

from core.upload_baseline_defaults import (
    apply_upload_baseline_defaults,
    sanitize_settings_for_job_payload,
    serialize_job_payload,
)
from core.upload_preference_dependencies import normalize_preferences_dict
from stages.ai_service_costs import billing_env_from_os, resolve_enabled_ai_services
from stages.entitlements import PUBLIC_TIER_SLUGS, get_entitlements_for_tier
from stages.tiktok_cover_burn import tiktok_burn_enabled

# ---------------------------------------------------------------------------
# Tier defaults (effective after apply_upload_baseline_defaults on empty prefs)
# ---------------------------------------------------------------------------
PUBLIC_UPLOAD_TIERS: Tuple[str, ...] = tuple(PUBLIC_TIER_SLUGS)

TIER_BASELINE_SUMMARY: Dict[str, str] = {
    "free": (
        "Starter: autoThumbnails on, autoCaptions off, styled thumbs off, Thumbnail Studio off, "
        "all AI services off, TikTok burn off, audio context off. Watermark enforced at worker."
    ),
    "creator_lite": (
        "Creator Lite: universal baseline (light AI on by default), styled thumbs on, "
        "TikTok burn opt-in only, watermark off."
    ),
    "creator_pro": (
        "Creator Pro: full universal baseline, styled AI thumbs, TikTok burn default on when unset."
    ),
    "studio": "Studio: Pro-level AI defaults + higher caps (team seats, queue depth).",
    "agency": "Agency: max caps + white-label + flex.",
}

# Every user-facing boolean that affects upload processing (camelCase keys).
BOOLEAN_PREF_KEYS: Tuple[str, ...] = (
    # Upload / captions / thumbs
    "autoCaptions",
    "autoThumbnails",
    "styledThumbnails",
    "thumbnailStudioEnabled",
    "thumbnailStudioEngineEnabled",
    "thumbnailPersonaEnabled",
    "tiktokBurnStyledCover",
    # Audio
    "useAudioContext",
    "audioTranscription",
    # Trill / drive
    "trillEnabled",
    "trillAiEnhance",
    # Hashtags
    "aiHashtagsEnabled",
    # Per-service AI (Settings UI + baseline)
    "aiServiceTelemetry",
    "aiServiceDashcamOSD",
    "aiServiceAudioSignals",
    "aiServiceMusicDetection",
    "aiServiceAudioSummary",
    "aiServiceCaptionWriter",
    "aiServiceThumbnailDesigner",
    "aiServiceSpeechToText",
    "aiServiceSceneUnderstanding",
    "aiServiceFrameInspector",
    "aiServiceVideoAnalyzer",
)

# Parent → children enforced by normalize_preferences_dict when parent explicitly off.
PREFERENCE_DEPENDENCY_GRAPH: Dict[str, Tuple[str, ...]] = {
    "useAudioContext": (
        "audioTranscription",
        "aiServiceSpeechToText",
        "aiServiceAudioSignals",
        "aiServiceMusicDetection",
        "aiServiceAudioSummary",
    ),
    "audioTranscription": ("aiServiceSpeechToText",),
    "autoThumbnails": (
        "styledThumbnails",
        "thumbnailStudioEnabled",
        "thumbnailStudioEngineEnabled",
        "thumbnailPersonaEnabled",
    ),
    "thumbnailStudioEnabled": (
        "thumbnailStudioEngineEnabled",
        "thumbnailPersonaEnabled",
    ),
    "trillEnabled": ("aiServiceTelemetry",),
    "aiServiceCaptionWriter": (
        "aiServiceFrameInspector",
        "aiServiceSceneUnderstanding",
        "aiServiceVideoAnalyzer",
    ),
}

_SNAKE_MAP: Dict[str, str] = {
    "autoCaptions": "auto_captions",
    "autoThumbnails": "auto_thumbnails",
    "styledThumbnails": "styled_thumbnails",
    "thumbnailStudioEnabled": "thumbnail_studio_enabled",
    "thumbnailStudioEngineEnabled": "thumbnail_studio_engine_enabled",
    "thumbnailPersonaEnabled": "thumbnail_persona_enabled",
    "tiktokBurnStyledCover": "tiktok_burn_styled_cover",
    "useAudioContext": "use_audio_context",
    "audioTranscription": "audio_transcription",
    "trillEnabled": "trill_enabled",
    "trillAiEnhance": "trill_ai_enhance",
    "aiHashtagsEnabled": "ai_hashtags_enabled",
    "aiServiceTelemetry": "ai_service_telemetry",
    "aiServiceDashcamOSD": "ai_service_dashcam_osd",
    "aiServiceAudioSignals": "ai_service_audio_signals",
    "aiServiceMusicDetection": "ai_service_music_detection",
    "aiServiceAudioSummary": "ai_service_audio_summary",
    "aiServiceCaptionWriter": "ai_service_caption_writer",
    "aiServiceThumbnailDesigner": "ai_service_thumbnail_designer",
    "aiServiceSpeechToText": "ai_service_speech_to_text",
    "aiServiceSceneUnderstanding": "ai_service_scene_understanding",
    "aiServiceFrameInspector": "ai_service_frame_inspector",
    "aiServiceVideoAnalyzer": "ai_service_video_analyzer",
}


def boolean_combination_count() -> int:
    return 2 ** len(BOOLEAN_PREF_KEYS)


def _pref_truthy(d: Mapping[str, Any], camel: str) -> bool:
    snake = _SNAKE_MAP.get(camel, camel)
    for k in (camel, snake):
        if k in d and d[k] is not None:
            raw = d[k]
            if isinstance(raw, str):
                return raw.strip().lower() not in ("false", "0", "no", "off", "")
            return bool(raw)
    return False


def _explicitly_disabled(d: Mapping[str, Any], camel: str) -> bool:
    snake = _SNAKE_MAP.get(camel, camel)
    for k in (camel, snake):
        if k in d and d[k] is not None:
            return not _pref_truthy(d, camel)
    return False


def combo_dict_from_bits(bits: Tuple[bool, ...]) -> Dict[str, bool]:
    if len(bits) != len(BOOLEAN_PREF_KEYS):
        raise ValueError(f"expected {len(BOOLEAN_PREF_KEYS)} bits, got {len(bits)}")
    return {key: bit for key, bit in zip(BOOLEAN_PREF_KEYS, bits)}


def iter_all_boolean_combinations() -> Iterator[Dict[str, bool]]:
    """Yield all 2^N explicit boolean preference payloads (every toggle on/off)."""
    for bits in itertools.product((False, True), repeat=len(BOOLEAN_PREF_KEYS)):
        yield combo_dict_from_bits(bits)


def combo_index(bits: Tuple[bool, ...]) -> int:
    """Stable index 0 .. 2^N-1 for grepping / Playwright stratification."""
    idx = 0
    for i, b in enumerate(bits):
        if b:
            idx |= 1 << i
    return idx


def bits_from_index(index: int) -> Tuple[bool, ...]:
    n = len(BOOLEAN_PREF_KEYS)
    return tuple(bool((index >> i) & 1) for i in range(n))


def combo_dict_from_index(index: int) -> Dict[str, bool]:
    return combo_dict_from_bits(bits_from_index(index))


def get_tier_effective_baseline(tier: str) -> Dict[str, Any]:
    """Defaults applied when user has never opened Settings (empty prefs row)."""
    out: Dict[str, Any] = {}
    apply_upload_baseline_defaults(out, tier=tier)
    return out


def prepare_upload_preferences(
    raw_overrides: Optional[Mapping[str, Any]],
    *,
    tier: str,
) -> Dict[str, Any]:
    """
    Mirror worker enqueue path: normalize → tier baseline fill → JSON-safe sanitize.
    """
    d: Dict[str, Any] = dict(raw_overrides or {})
    normalize_preferences_dict(d)
    apply_upload_baseline_defaults(d, tier=tier)
    return sanitize_settings_for_job_payload(d)


def assert_dependency_invariants(prefs: Mapping[str, Any]) -> None:
    """After normalize, children must be off when parent was explicitly disabled."""
    for parent, children in PREFERENCE_DEPENDENCY_GRAPH.items():
        if not _explicitly_disabled(prefs, parent):
            continue
        for child in children:
            if _pref_truthy(prefs, child):
                raise AssertionError(
                    f"{child} must be off when {parent} is explicitly disabled "
                    f"(got {prefs.get(child)!r})"
                )

    # Emotion integration removed — always forced off.
    if _pref_truthy(prefs, "aiServiceEmotionSignals") or prefs.get("ai_service_emotion_signals"):
        raise AssertionError("aiServiceEmotionSignals must remain off")


def assert_tier_runtime_gates(prefs: Mapping[str, Any], tier: str) -> None:
    """Pipeline gates must not throw; tier policy must hold for hard gates."""
    ent = get_entitlements_for_tier(tier)
    env = billing_env_from_os()

    resolve_enabled_ai_services(
        can_ai=ent.can_ai,
        user_prefs=dict(prefs),
        use_ai_request=False,
        has_telemetry=False,
        env=env,
        tier_allowed=ent.allowed_ai_services,
    )

    if tier == "free":
        assert tiktok_burn_enabled(dict(prefs), ent) is False


def validate_settings_combination(
    raw_overrides: Mapping[str, Any],
    *,
    tier: str,
) -> Dict[str, Any]:
    """
    Full validation for one (tier, toggle vector) pair.

    Returns job-ready preferences dict. Raises AssertionError on invariant break.
    """
    prepared = prepare_upload_preferences(raw_overrides, tier=tier)
    assert_dependency_invariants(prepared)
    assert_tier_runtime_gates(prepared, tier)

    payload = {
        "upload_id": "00000000-0000-0000-0000-000000000001",
        "user_id": "00000000-0000-0000-0000-000000000002",
        "job_id": "combo-test",
        "preferences": prepared,
    }
    json.loads(serialize_job_payload(payload))
    return prepared


@dataclass(frozen=True)
class ComboSample:
    index: int
    tier: str
    overrides: Dict[str, bool]


def stratified_playwright_samples(
    *,
    max_per_tier: Optional[int] = None,
    indices: Optional[FrozenSet[int]] = None,
) -> List[ComboSample]:
    """
    Subset for Playwright/API smoke: every tier × (all combos or explicit indices).

    Default max_per_tier=None runs full cross-product (slow — use SETTINGS_PW_MAX env).
    """
    total = boolean_combination_count()
    if indices is not None:
        idx_list = sorted(indices)
    elif max_per_tier is not None and max_per_tier < total:
        step = max(1, total // max_per_tier)
        idx_list = list(range(0, total, step))[:max_per_tier]
    else:
        idx_list = list(range(total))

    out: List[ComboSample] = []
    for tier in PUBLIC_UPLOAD_TIERS:
        for idx in idx_list:
            out.append(
                ComboSample(
                    index=idx,
                    tier=tier,
                    overrides=combo_dict_from_index(idx),
                )
            )
    return out


def combo_signature(combo: Mapping[str, bool]) -> str:
    """Greppable one-line signature: ``autoCaptions=0|autoThumbnails=1|...``."""
    return "|".join(f"{k}={1 if combo.get(k) else 0}" for k in BOOLEAN_PREF_KEYS)
