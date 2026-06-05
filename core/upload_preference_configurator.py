"""
Upload preference configurator — packages, toggle impact preview, tier save clamping.

Powers Settings UI confirm/deny modals and server-side save enforcement.
Starter users may opt in to Audio / Caption / Drive packages; styled stack and
TikTok burn remain upgrade-gated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

from core.upload_preference_dependencies import normalize_preferences_dict
from stages.entitlements import normalize_tier
from stages.tiktok_cover_burn import default_tiktok_burn_styled_cover_pref, tiktok_burn_enabled

# Human labels for Settings modal (camelCase keys = HTML element ids).
PREF_LABELS: Dict[str, str] = {
    "autoCaptions": "Auto-generate captions",
    "autoThumbnails": "Auto-generate thumbnails",
    "styledThumbnails": "Styled thumbnails",
    "thumbnailStudioEnabled": "Thumbnail Studio flow",
    "thumbnailStudioEngineEnabled": "AuroraRender engine",
    "thumbnailPersonaEnabled": "Persona consistency",
    "tiktokBurnStyledCover": "TikTok styled cover burn-in",
    "useAudioContext": "Enable audio context",
    "audioTranscription": "Speech-to-text transcript",
    "trillEnabled": "Drive analysis (Trill)",
    "trillAiEnhance": "Trill AI content enhancement",
    "aiHashtagsEnabled": "AI-generated hashtags",
    "aiServiceTelemetry": "Telemetry insights (.map file)",
    "aiServiceDashcamOSD": "Dashcam OSD reader",
    "aiServiceAudioSignals": "Audio event detection",
    "aiServiceMusicDetection": "Music detection",
    "aiServiceAudioSummary": "Audio content summary",
    "aiServiceCaptionWriter": "Caption & hashtag writer",
    "aiServiceThumbnailDesigner": "Thumbnail designer",
    "aiServiceSpeechToText": "Speech-to-text (Whisper)",
    "aiServiceSceneUnderstanding": "Scene understanding (Twelve Labs)",
    "aiServiceFrameInspector": "Frame inspector (Vision)",
    "aiServiceVideoAnalyzer": "Video analyzer (Video Intelligence)",
    "defaultPrivacy": "Default privacy",
}

# Feature packages (configurator groups).
PREF_PACKAGES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "base_publish",
        "label": "Base publish",
        "description": "Privacy, connected platforms, and watermark on Starter — always included.",
        "master": None,
        "members": ("defaultPrivacy",),
        "starter_opt_in": True,
        "always_on": True,
    },
    {
        "id": "audio_intelligence",
        "label": "Audio intelligence",
        "description": "Transcribe speech and analyze audio for captions and hashtags.",
        "master": "useAudioContext",
        "members": (
            "useAudioContext",
            "audioTranscription",
            "aiServiceSpeechToText",
            "aiServiceAudioSignals",
            "aiServiceMusicDetection",
            "aiServiceAudioSummary",
        ),
        "starter_opt_in": True,
    },
    {
        "id": "caption_ai",
        "label": "Caption AI",
        "description": "AI titles, captions, scene context, and vision for posts.",
        "master": "autoCaptions",
        "members": (
            "autoCaptions",
            "aiServiceCaptionWriter",
            "aiServiceSceneUnderstanding",
            "aiServiceFrameInspector",
            "aiServiceVideoAnalyzer",
        ),
        "starter_opt_in": True,
    },
    {
        "id": "thumbnail_stack",
        "label": "Styled thumbnail stack",
        "description": "Styled overlays, Thumbnail Studio, AuroraRender, and vision for frames.",
        "master": "autoThumbnails",
        "members": (
            "autoThumbnails",
            "styledThumbnails",
            "thumbnailStudioEnabled",
            "thumbnailStudioEngineEnabled",
            "aiServiceThumbnailDesigner",
            "aiServiceFrameInspector",
            "aiServiceVideoAnalyzer",
        ),
        "starter_opt_in": False,
    },
    {
        "id": "thumbnail_persona",
        "label": "Thumbnail persona",
        "description": "Apply a saved creator persona during thumbnail rendering.",
        "master": "thumbnailPersonaEnabled",
        "members": ("thumbnailPersonaEnabled", "thumbnailStudioEnabled", "autoThumbnails"),
        "starter_opt_in": False,
    },
    {
        "id": "drive_trill",
        "label": "Drive / Trill",
        "description": "Analyze .map telemetry when uploaded; optional Trill AI enhancement.",
        "master": "trillEnabled",
        "members": ("trillEnabled", "aiServiceTelemetry", "trillAiEnhance"),
        "starter_opt_in": True,
    },
    {
        "id": "tiktok_cover_burn",
        "label": "TikTok styled cover burn-in",
        "description": "Composite styled 9:16 thumb into the TikTok MP4 at cover time (Pro+).",
        "master": "tiktokBurnStyledCover",
        "members": ("tiktokBurnStyledCover",),
        "starter_opt_in": False,
    },
)

# Parent → children (disable cascade when parent turned off).
DEPENDENCY_GRAPH: Dict[str, Tuple[str, ...]] = {
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
        "aiServiceThumbnailDesigner",
    ),
    "styledThumbnails": ("aiServiceThumbnailDesigner",),
    "thumbnailStudioEnabled": (
        "thumbnailStudioEngineEnabled",
        "thumbnailPersonaEnabled",
    ),
    "trillEnabled": ("aiServiceTelemetry", "trillAiEnhance"),
    "aiServiceCaptionWriter": ("aiServiceSceneUnderstanding",),
    "autoCaptions": (
        "aiServiceCaptionWriter",
        "aiServiceFrameInspector",
        "aiServiceVideoAnalyzer",
    ),
}

# Enabling a child implies these parents/peers should also turn on.
ENABLE_REQUIRES: Dict[str, Tuple[str, ...]] = {
    "audioTranscription": ("useAudioContext",),
    "aiServiceSpeechToText": ("useAudioContext", "audioTranscription"),
    "aiServiceAudioSignals": ("useAudioContext",),
    "aiServiceMusicDetection": ("useAudioContext",),
    "aiServiceAudioSummary": ("useAudioContext",),
    "aiServiceCaptionWriter": ("autoCaptions",),
    "aiServiceSceneUnderstanding": ("autoCaptions", "aiServiceCaptionWriter"),
    "aiServiceFrameInspector": ("autoCaptions",),
    "aiServiceVideoAnalyzer": ("autoCaptions",),
    "aiServiceThumbnailDesigner": ("autoThumbnails", "styledThumbnails"),
    "styledThumbnails": ("autoThumbnails",),
    "thumbnailStudioEnabled": ("autoThumbnails", "styledThumbnails"),
    "thumbnailStudioEngineEnabled": ("autoThumbnails", "thumbnailStudioEnabled"),
    "thumbnailPersonaEnabled": ("autoThumbnails", "thumbnailStudioEnabled"),
    "aiServiceTelemetry": ("trillEnabled",),
    "trillAiEnhance": ("trillEnabled",),
    "tiktokBurnStyledCover": (
        "autoThumbnails",
        "styledThumbnails",
        "thumbnailStudioEnabled",
    ),
}

# Cross-package: enabling master X also proposes these (from other packages).
CROSS_PACKAGE_ENABLE: Dict[str, Tuple[str, ...]] = {
    "autoCaptions": (
        "useAudioContext",
        "audioTranscription",
        "aiServiceSpeechToText",
        "aiServiceAudioSignals",
        "aiServiceMusicDetection",
        "aiServiceAudioSummary",
    ),
    "autoThumbnails": (
        "aiServiceFrameInspector",
        "aiServiceVideoAnalyzer",
    ),
}

# Starter may opt in (explicit ON is persisted when dependencies are satisfied).
FREE_TIER_OPT_IN_KEYS: FrozenSet[str] = frozenset(
    {
        "autoCaptions",
        "useAudioContext",
        "audioTranscription",
        "aiServiceSpeechToText",
        "aiServiceAudioSignals",
        "aiServiceMusicDetection",
        "aiServiceAudioSummary",
        "aiServiceCaptionWriter",
        "aiServiceSceneUnderstanding",
        "aiServiceFrameInspector",
        "aiServiceVideoAnalyzer",
        "trillEnabled",
        "autoThumbnails",
    }
)

# Starter cannot persist these as ON (upgrade or runtime-only).
FREE_TIER_CANNOT_ENABLE: FrozenSet[str] = frozenset(
    {
        "styledThumbnails",
        "thumbnailStudioEnabled",
        "thumbnailStudioEngineEnabled",
        "thumbnailPersonaEnabled",
        "tiktokBurnStyledCover",
        "aiServiceTelemetry",
        "aiServiceDashcamOSD",
        "aiServiceThumbnailDesigner",
        "trillAiEnhance",
    }
)

_PACKAGE_BY_MASTER: Dict[str, Dict[str, Any]] = {
    str(p["master"]): p for p in PREF_PACKAGES if p.get("master")
}

_PACKAGE_BY_ID: Dict[str, Dict[str, Any]] = {str(p["id"]): p for p in PREF_PACKAGES}


def pref_label(key: str) -> str:
    return PREF_LABELS.get(key, key)


def configurator_meta_for_tier(tier: str) -> Dict[str, Any]:
    """JSON-safe package metadata for Settings UI."""
    slug = normalize_tier(tier)
    ent_stub = type(
        "E",
        (),
        {"tier": slug, "can_ai_thumbnail_styling": slug not in ("free", "creator_lite")},
    )()
    packages = []
    for p in PREF_PACKAGES:
        always_on = bool(p.get("always_on"))
        master = p.get("master")
        blocked = (
            slug == "free"
            and not p.get("starter_opt_in", True)
            and not always_on
        )
        packages.append(
            {
                "id": p["id"],
                "label": p["label"],
                "description": p.get("description", ""),
                "master": master,
                "members": list(p["members"]),
                "starterOptIn": bool(p.get("starter_opt_in")),
                "alwaysOn": always_on,
                "tierBlocked": blocked,
            }
        )
    return {
        "packages": packages,
        "labels": dict(PREF_LABELS),
        "dependencyGraph": {k: list(v) for k, v in DEPENDENCY_GRAPH.items()},
        "crossPackageEnable": {k: list(v) for k, v in CROSS_PACKAGE_ENABLE.items()},
        "freeTierOptInKeys": sorted(FREE_TIER_OPT_IN_KEYS),
        "freeTierCannotEnable": sorted(FREE_TIER_CANNOT_ENABLE),
        "tiktokBurnTierDefault": default_tiktok_burn_styled_cover_pref(ent_stub),
        "telemetryRequiresMap": True,
    }


def _truthy(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        return val.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(val)


def _prefs_bool(prefs: Mapping[str, Any], key: str) -> bool:
    import re

    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
    for k in (key, snake):
        if k in prefs and prefs[k] is not None:
            return _truthy(prefs[k])
    return False


def _collect_disable_cascade(key: str, seen: Optional[Set[str]] = None) -> Set[str]:
    seen = seen or set()
    out: Set[str] = set()
    for child in DEPENDENCY_GRAPH.get(key, ()):
        if child not in seen:
            seen.add(child)
            out.add(child)
            out |= _collect_disable_cascade(child, seen)
    return out


def _collect_enable_requires(key: str, seen: Optional[Set[str]] = None) -> Set[str]:
    seen = seen or set()
    out: Set[str] = set()
    for req in ENABLE_REQUIRES.get(key, ()):
        if req not in seen:
            seen.add(req)
            out.add(req)
            out |= _collect_enable_requires(req, seen)
    for req in CROSS_PACKAGE_ENABLE.get(key, ()):
        if req not in seen:
            seen.add(req)
            out.add(req)
            out |= _collect_enable_requires(req, seen)
    return out


def _tier_block_message(key: str, tier: str) -> Optional[str]:
    slug = normalize_tier(tier)
    if slug != "free":
        if key == "tiktokBurnStyledCover":
            ent = type(
                "E",
                (),
                {
                    "tier": slug,
                    "can_ai_thumbnail_styling": slug
                    in ("creator_pro", "studio", "agency", "friends_family", "lifetime", "master_admin"),
                },
            )()
            if not default_tiktok_burn_styled_cover_pref(ent):
                return (
                    "TikTok styled cover burn-in requires Creator Pro or higher "
                    "(Lite: opt in in Settings)."
                )
        return None
    if key in FREE_TIER_CANNOT_ENABLE:
        pkg = _PACKAGE_BY_MASTER.get(key)
        name = pkg["label"] if pkg else pref_label(key)
        return (
            f"{name} is not included on Starter. Upgrade to Creator Lite or skip "
            f"for faster base uploads."
        )
    return None


def _preview_notes(key: str, new_value: bool, prefs: Mapping[str, Any], tier: str) -> Optional[str]:
    slug = normalize_tier(tier)
    if new_value and key == "tiktokBurnStyledCover":
        parts = ["Requires styled thumbnails (Creator Lite+)."]
        if slug == "free":
            parts.append("Upgrade to Creator Pro for default burn-in.")
        if not _prefs_bool(prefs, "styledThumbnails"):
            parts.append("We'll turn on Auto thumbnails and Styled thumbnails.")
        return " ".join(parts)
    if new_value and key == "trillEnabled":
        return (
            "Telemetry insights run only when you upload a .map file with the video. "
            "Trill AI enhance uses extra AI credits."
        )
    if not new_value and key == "useAudioContext" and _prefs_bool(prefs, "autoCaptions"):
        return "Captions may be lower quality without audio context."
    if not new_value and key == "autoCaptions":
        return "Vision and scene services may stay on if thumbnails still need them."
    return None


@dataclass
class PrefChangePreview:
    key: str
    new_value: bool
    enable: List[str] = field(default_factory=list)
    disable: List[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: Optional[str] = None
    package_id: Optional[str] = None
    package_label: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "newValue": self.new_value,
            "enable": [{"key": k, "label": pref_label(k)} for k in self.enable],
            "disable": [{"key": k, "label": pref_label(k)} for k in self.disable],
            "blocked": self.blocked,
            "blockReason": self.block_reason,
            "packageId": self.package_id,
            "packageLabel": self.package_label,
            "note": self.note,
        }


def preview_pref_change(
    key: str,
    new_value: bool,
    prefs: Mapping[str, Any],
    *,
    tier: str = "free",
) -> PrefChangePreview:
    """Compute enable/disable impact for a single toggle change."""
    key = str(key or "").strip()
    if not key:
        return PrefChangePreview(key="", new_value=new_value, blocked=True, block_reason="Missing preference key.")

    pkg = _PACKAGE_BY_MASTER.get(key)
    preview = PrefChangePreview(
        key=key,
        new_value=bool(new_value),
        package_id=pkg["id"] if pkg else None,
        package_label=pkg["label"] if pkg else None,
    )

    if new_value:
        block = _tier_block_message(key, tier)
        if block:
            preview.blocked = True
            preview.block_reason = block
            return preview
        to_enable = _collect_enable_requires(key)
        if pkg and key == pkg["master"]:
            to_enable |= set(pkg["members"])
        preview.enable = sorted(k for k in to_enable if k != key and not _prefs_bool(prefs, k))
        preview.note = _preview_notes(key, True, prefs, tier)
    else:
        to_disable = _collect_disable_cascade(key)
        if pkg and key == pkg["master"]:
            to_disable |= {m for m in pkg["members"] if m != key}
        preview.disable = sorted(k for k in to_disable if _prefs_bool(prefs, k))
        preview.note = _preview_notes(key, False, prefs, tier)

    return preview


def apply_pref_change(
    prefs: Dict[str, Any],
    key: str,
    new_value: bool,
    *,
    tier: str = "free",
) -> Dict[str, Any]:
    """Apply a confirmed toggle change plus cascades."""
    preview = preview_pref_change(key, new_value, prefs, tier=tier)
    if preview.blocked:
        raise ValueError(preview.block_reason or "Change blocked for tier.")
    out = dict(prefs)
    out[key] = bool(new_value)
    for k in preview.enable:
        out[k] = True
    for k in preview.disable:
        out[k] = False
    normalize_preferences_dict(out)
    return out


def clamp_prefs_for_tier(prefs: Dict[str, Any], *, tier: str) -> Dict[str, Any]:
    """Force-off toggles Starter cannot persist; normalize dependencies."""
    slug = normalize_tier(tier)
    out = dict(prefs)
    if slug == "free":
        for k in FREE_TIER_CANNOT_ENABLE:
            out[k] = False
    ent = type(
        "E",
        (),
        {
            "tier": slug,
            "can_ai_thumbnail_styling": slug
            in ("creator_pro", "studio", "agency", "friends_family", "lifetime", "master_admin"),
        },
    )()
    if not tiktok_burn_enabled(out, ent):
        out["tiktokBurnStyledCover"] = False
        out["tiktok_burn_styled_cover"] = False
    normalize_preferences_dict(out)
    return out


def packages_for_api() -> List[Dict[str, Any]]:
    return [dict(p) for p in PREF_PACKAGES]
