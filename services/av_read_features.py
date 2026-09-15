"""
Shared AV-read distill feature columns and ctx → one-row builder for infer.

Must stay aligned with scripts/train_av_read_distill.py NUM/CAT features.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

NUM_FEATURES: List[str] = [
    "transcript_chars",
    "segment_count",
    "vision_label_count",
    "vision_ocr_chars",
    "evidence_lane_count",
    "keyframe_count",
    "grounding_score",
]
CAT_FEATURES: List[str] = ["clip_kind", "identity_confidence", "tl_status", "fusion_source"]


def _as_dict(val: Any) -> Dict[str, Any]:
    if isinstance(val, dict):
        return val
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def feature_row_from_ctx(ctx: Any) -> Dict[str, Any]:
    """Build one training-shaped row from live JobContext (pre- or post-TL)."""
    arts = getattr(ctx, "output_artifacts", None) or {}
    if not isinstance(arts, dict):
        arts = {}
    ac = getattr(ctx, "audio_context", None) or {}
    if not isinstance(ac, dict):
        ac = {}
    vc = getattr(ctx, "vision_context", None) or {}
    if not isinstance(vc, dict):
        vc = {}
    route = _as_dict(arts.get("multimodal_depth_route_v1"))
    pack = _as_dict(arts.get("av_training_pack_v1"))
    identity = _as_dict(arts.get("content_identity_v1"))
    gs = _as_dict(arts.get("grounding_score_v1"))

    transcript = (
        str(getattr(ctx, "ai_transcript", None) or "")
        or str(ac.get("transcript") or "")
    ).strip()
    segs = ac.get("transcript_segments") or []
    labels = vc.get("label_names") or vc.get("labels") or []
    ocr = str(vc.get("ocr_text") or "")
    lanes = 0
    if labels or ocr:
        lanes += 1
    if transcript:
        lanes += 1
    if arts.get("shot_list_v1"):
        lanes += 1
    if getattr(ctx, "video_understanding", None):
        lanes += 1

    try:
        grounding_score = float(gs.get("grounding_score")) if gs.get("grounding_score") is not None else None
    except (TypeError, ValueError):
        grounding_score = None

    clip_kind = str(
        route.get("clip_kind") or arts.get("clip_kind") or pack.get("clip_kind") or "general"
    )
    tl_status = str(pack.get("tl_status") or "unknown")
    # Before TL runs, mark unknown; after skip/fail stages may leave empty VU.
    vu = getattr(ctx, "video_understanding", None) or {}
    if isinstance(vu, dict) and (vu.get("scene_description") or vu.get("source") == "twelvelabs"):
        tl_status = "ok" if tl_status == "unknown" else tl_status
    fusion_source = str(pack.get("fusion_source") or "none")
    if isinstance(vu, dict) and str(vu.get("source") or "").startswith("fusion"):
        fusion_source = str(vu.get("source") or "fusion")

    fusion_text = str(
        (vu.get("scene_description") if isinstance(vu, dict) else "")
        or arts.get("fusion_summary")
        or ""
    )[:2000]

    return {
        "transcript_chars": len(transcript),
        "segment_count": len(segs) if isinstance(segs, list) else 0,
        "vision_label_count": len(labels) if isinstance(labels, list) else 0,
        "vision_ocr_chars": len(ocr),
        "evidence_lane_count": int(lanes),
        "keyframe_count": int(pack.get("keyframe_count") or 0),
        "grounding_score": grounding_score if grounding_score is not None else 0.0,
        "clip_kind": clip_kind or "general",
        "identity_confidence": str(identity.get("confidence") or "unknown"),
        "tl_status": tl_status or "unknown",
        "fusion_source": fusion_source or "none",
        "fusion_text": fusion_text,
    }


__all__ = ["NUM_FEATURES", "CAT_FEATURES", "feature_row_from_ctx"]
