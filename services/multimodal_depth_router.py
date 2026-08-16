"""
Multimodal depth router — cheap clip classification + when to force deep scene.

Runs after Vision (and preferably after VI) so generic Vision labels can invert
the Twelve Labs ``TWELVELABS_SKIP_WHEN_VI_RICH`` cost gate. Also classifies a
coarse clip kind from filename / duration / labels for future stage budgets.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from core.vision_labels import vision_labels_are_weak

# Kill-switch: MULTIMODAL_DEPTH_FORCE_TL=0 disables force-TL behavior.
_FORCE_TL_ENABLED = (os.environ.get("MULTIMODAL_DEPTH_FORCE_TL", "true") or "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

_CLIP_KINDS = (
    "dashcam",
    "concert",
    "museum",
    "wearable",
    "vlog",
    "product",
    "gameplay",
    "music",
    "sports",
    "silent_scenic",
    "general",
)


def _env_bool(key: str, default: bool = True) -> bool:
    raw = (os.environ.get(key) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def classify_clip_kind(ctx: Any) -> str:
    """Cheap genre from filename, category, duration, and Vision labels.

    Dashcam-first. Concert / museum / wearable / vlog-from-speech come next so
    glasses POV and indoor culture clips are not routed as silent scenic.
    """
    raw_name = str(getattr(ctx, "filename", "") or "")
    fname = raw_name.upper()
    cat = str(getattr(ctx, "thumbnail_category", None) or "").strip().lower()
    labels = []
    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict):
        labels = [str(x).lower() for x in (vc.get("label_names") or []) if str(x).strip()]
    blob = " ".join(labels)
    ocr = str((vc.get("ocr_text") if isinstance(vc, dict) else "") or "").lower()

    from core.wearable_source import is_meta_glasses_filename

    meta_glasses = is_meta_glasses_filename(raw_name)
    dashcam_name = any(tok in fname for tok in ("DASH", "M8_", "ESCORT", "DRIVECAM", "BLACKVUE"))
    if dashcam_name or (cat in ("dashcam", "automotive") and not meta_glasses):
        return "dashcam"

    concert_labels = any(m in blob for m in ("concert", "stage", "crowd", "festival"))
    ac = getattr(ctx, "audio_context", None) or {}
    acr = False
    if isinstance(ac, dict):
        acr = bool(ac.get("music_title") or ac.get("music_artist") or ac.get("acr_match"))
    if concert_labels or (acr and any(m in blob for m in ("concert", "stage", "festival", "crowd"))):
        return "concert"

    if any(tok in ocr for tok in ("museum", "cathedral", "temple", "exhibit")) or cat in (
        "museum",
        "art",
    ):
        return "museum"

    driving = False
    try:
        from core.driving_evidence import has_driving_evidence

        driving = bool(has_driving_evidence(ctx))
    except Exception:
        driving = False
    wearable_labels = any(
        m in blob
        for m in (
            "backpack",
            "trail",
            "harbor",
            "harbour",
            "church",
            "plaza",
            "fountain",
            "glasses",
        )
    )
    # Meta export names are wearable before vlog-from-speech (talking walk stays POV).
    if not driving and (meta_glasses or cat in ("travel",) or wearable_labels):
        return "wearable"

    transcript = ""
    if isinstance(ac, dict):
        transcript = str(ac.get("transcript") or "").strip()
    if not transcript:
        transcript = str(getattr(ctx, "ai_transcript", None) or "").strip()
    has_faces = bool(vc.get("has_faces")) if isinstance(vc, dict) else False
    if any(tok in fname for tok in ("VLOG", "TALKING", "PODCAST")) or has_faces or len(transcript) >= 80:
        return "vlog"

    if any(tok in fname for tok in ("GAME", "GAMEPLAY", "TWITCH", "FORTNITE", "VALORANT")) or any(
        m in blob for m in ("video game", "screenshot", "controller")
    ):
        return "gameplay"
    if cat in ("music",) or any(m in blob for m in ("musical instrument", "microphone")):
        return "music"
    if cat in ("sports", "fitness") or any(
        m in blob for m in ("stadium", "soccer", "basketball", "baseball", "football", "jersey")
    ):
        return "sports"
    if cat in ("product", "business") or any(m in blob for m in ("product", "packaging", "cosmetics")):
        return "product"

    dur = float(getattr(ctx, "duration_seconds", None) or getattr(ctx, "duration", None) or 0)
    speech_like = len(transcript) >= 40
    if not speech_like and dur >= 20 and vision_labels_are_weak(
        labels,
        landmark_names=(vc.get("landmark_names") if isinstance(vc, dict) else None),
        logo_names=(vc.get("logo_names") if isinstance(vc, dict) else None),
        ocr_text=str((vc.get("ocr_text") if isinstance(vc, dict) else "") or ""),
    ):
        return "silent_scenic"
    return "general"


def _vision_weak_from_ctx(ctx: Any) -> bool:
    vc = getattr(ctx, "vision_context", None) or {}
    if not isinstance(vc, dict) or not vc:
        # No Vision yet — do not force from empty; cheap pre-route may still force by kind.
        return False
    return vision_labels_are_weak(
        vc.get("label_names") or [],
        landmark_names=vc.get("landmark_names") or [],
        logo_names=vc.get("logo_names") or [],
        ocr_text=str(vc.get("ocr_text") or ""),
    )


def _has_strong_place_or_speech(ctx: Any) -> bool:
    tel = getattr(ctx, "telemetry_data", None) or getattr(ctx, "telemetry", None)
    if tel is not None:
        if getattr(tel, "location_city", None) or getattr(tel, "gazetteer_place_name", None):
            return True
        pts = getattr(tel, "points", None) or []
        if pts:
            return True
    osd = getattr(ctx, "dashcam_osd_context", None) or {}
    if isinstance(osd, dict) and (osd.get("gps_path") or osd.get("max_speed_mph")):
        return True
    tr = (getattr(ctx, "ai_transcript", None) or "") or ""
    if len(str(tr).strip()) >= 80:
        return True
    pe = getattr(ctx, "place_evidence", None) or {}
    if isinstance(pe, dict) and (
        pe.get("landmarks") or pe.get("places") or pe.get("beaches") or pe.get("monuments")
    ):
        return True
    return False


def route_multimodal_depth(ctx: Any) -> Dict[str, Any]:
    """
    Decide whether to force Twelve Labs (and related depth hints).

    Returns dict with keys: clip_kind, force_twelvelabs, vision_weak, reason, reasons[].
    """
    kind = classify_clip_kind(ctx)
    vision_weak = _vision_weak_from_ctx(ctx)
    reasons: List[str] = [f"clip_kind={kind}"]
    force = False

    if not _FORCE_TL_ENABLED or not _env_bool("MULTIMODAL_DEPTH_FORCE_TL", True):
        return {
            "clip_kind": kind,
            "force_twelvelabs": False,
            "vision_weak": vision_weak,
            "reason": "depth_force_disabled",
            "reasons": reasons + ["MULTIMODAL_DEPTH_FORCE_TL=off"],
        }

    us = getattr(ctx, "user_settings", None) or {}
    if bool(us.get("force_twelvelabs") or us.get("forceTwelveLabs")):
        force = True
        reasons.append("user_forceTwelveLabs")

    # Generic Vision without place/speech → need narrative model.
    if vision_weak and not _has_strong_place_or_speech(ctx):
        force = True
        reasons.append("vision_labels_weak")

    # Non-dashcam niches often under-served when VI object count looks "rich".
    if kind in (
        "vlog",
        "product",
        "gameplay",
        "music",
        "sports",
        "silent_scenic",
        "concert",
        "museum",
        "wearable",
    ) and vision_weak:
        force = True
        reasons.append(f"niche_needs_depth:{kind}")

    reason = ";".join(reasons)
    return {
        "clip_kind": kind if kind in _CLIP_KINDS else "general",
        "force_twelvelabs": bool(force),
        "vision_weak": bool(vision_weak),
        "reason": reason,
        "reasons": reasons,
    }


def apply_depth_route_to_ctx(ctx: Any, route: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist route on ctx + optionally set forceTwelveLabs on user_settings."""
    route = route or route_multimodal_depth(ctx)
    if not isinstance(getattr(ctx, "output_artifacts", None), dict):
        ctx.output_artifacts = {}
    ctx.output_artifacts["multimodal_depth_route_v1"] = dict(route)
    ctx.output_artifacts["clip_kind"] = str(route.get("clip_kind") or "general")
    setattr(ctx, "multimodal_depth_route", dict(route))

    if route.get("force_twelvelabs"):
        us = dict(getattr(ctx, "user_settings", None) or {})
        us["forceTwelveLabs"] = True
        us["force_twelvelabs"] = True
        ctx.user_settings = us
    return route


__all__ = [
    "classify_clip_kind",
    "route_multimodal_depth",
    "apply_depth_route_to_ctx",
]
