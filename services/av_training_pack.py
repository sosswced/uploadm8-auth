"""
AV training pack v1 — compact supervised export of the accuracy ladder.

Builds ``av_training_pack_v1`` from hydration + identity + transcript/vision
for opt-in R2 retention. Sense+align training only; identity and M8 grounding
remain the product trust layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("uploadm8-worker")

AV_TRAINING_PACK_VERSION = 1
AV_TRAINING_PACK_ARTIFACT = "av_training_pack_v1"
MAX_KEYFRAMES = 8
MAX_TRANSCRIPT_SEGMENTS = 120


def hash_upload_id(upload_id: Any) -> str:
    raw = str(upload_id or "").strip()
    if not raw:
        return ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def pack_r2_prefix(user_id: Any, upload_id: Any) -> str:
    return f"ml-packs/{user_id}/{upload_id}"


def pack_json_r2_key(user_id: Any, upload_id: Any) -> str:
    return f"{pack_r2_prefix(user_id, upload_id)}/pack.json"


def keyframe_r2_key(user_id: Any, upload_id: Any, index: int) -> str:
    return f"{pack_r2_prefix(user_id, upload_id)}/kf_{int(index):02d}.jpg"


def _as_dict(val: Any) -> Dict[str, Any]:
    if isinstance(val, dict):
        return val
    if isinstance(val, str) and val.strip().startswith("{"):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _clip_kind(ctx: Any) -> str:
    arts = getattr(ctx, "output_artifacts", None) or {}
    if isinstance(arts, dict):
        ck = arts.get("clip_kind")
        if ck:
            return str(ck)
        route = arts.get("multimodal_depth_route_v1")
        if isinstance(route, dict) and route.get("clip_kind"):
            return str(route["clip_kind"])
    route = getattr(ctx, "multimodal_depth_route", None)
    if isinstance(route, dict) and route.get("clip_kind"):
        return str(route["clip_kind"])
    try:
        from services.multimodal_depth_router import classify_clip_kind

        return classify_clip_kind(ctx)
    except Exception:
        return "general"


def _tl_status_and_fusion_source(ctx: Any) -> Tuple[str, str]:
    """Derive tl_status and fusion_source for Layer B backup cohort labels."""
    vu = getattr(ctx, "video_understanding", None) or {}
    if not isinstance(vu, dict):
        vu = {}
    arts = getattr(ctx, "output_artifacts", None) or {}
    if not isinstance(arts, dict):
        arts = {}
    src = str(vu.get("source") or "").strip().lower()
    scene = str(vu.get("scene_description") or vu.get("summary") or "").strip()
    fusion_art = _as_dict(arts.get("scene_fusion"))

    if "twelve" in src or src in ("twelvelabs", "pegasus", "tl"):
        return "ok", "twelvelabs"
    if "enrich" in src or fusion_art.get("enriched"):
        return ("skipped" if not scene else "skipped"), "fusion_enrich"
    if "fusion" in src or "scene_fusion" in src or fusion_art.get("applied"):
        return "skipped", "scene_fusion"
    if scene and src:
        # Unknown narrative source that still produced prose
        if "fail" in src or "error" in src:
            return "failed", "scene_fusion" if fusion_art else "none"
        return "ok", src[:40]
    if scene:
        return "skipped", "scene_fusion" if fusion_art else "none"

    us = getattr(ctx, "user_settings", None) or {}
    prefs = getattr(ctx, "user_prefs", None) or us
    from stages.ai_service_costs import user_pref_ai_service_enabled

    if not user_pref_ai_service_enabled(prefs if isinstance(prefs, dict) else {}, "twelvelabs", default=False):
        return "disabled", "none" if not scene else "scene_fusion"
    return "failed" if not scene else "skipped", "none" if not scene else "scene_fusion"


def _teacher_flags(ctx: Any) -> Dict[str, Any]:
    arts = getattr(ctx, "output_artifacts", None) or {}
    route: Dict[str, Any] = {}
    if isinstance(arts, dict):
        route = _as_dict(arts.get("multimodal_depth_route_v1"))
    if not route:
        route = dict(getattr(ctx, "multimodal_depth_route", None) or {})

    vc = getattr(ctx, "vision_context", None) or {}
    ac = getattr(ctx, "audio_context", None) or {}
    vu = getattr(ctx, "video_understanding", None) or {}
    vi = getattr(ctx, "video_intelligence_context", None) or {}

    lanes: List[str] = []
    if isinstance(ac, dict) and (ac.get("transcript") or getattr(ctx, "ai_transcript", None)):
        lanes.append("transcript")
    if isinstance(vc, dict) and (vc.get("label_names") or vc.get("ocr_text")):
        lanes.append("vision")
    if isinstance(vi, dict) and (vi.get("shot_changes") or vi.get("segment_labels") or vi.get("labels")):
        lanes.append("video_intelligence")
    if isinstance(vu, dict) and (vu.get("scene_description") or vu.get("summary")):
        lanes.append("scene")
    if isinstance(arts, dict) and arts.get("shot_list_v1"):
        lanes.append("shot_list")
    if isinstance(arts, dict) and arts.get("content_identity_v1"):
        lanes.append("identity")
    if isinstance(arts, dict) and arts.get("hydration_payload"):
        lanes.append("hydration")

    force_tl = bool(route.get("force_twelvelabs"))
    vision_weak = bool(route.get("vision_weak"))
    if not route:
        us = getattr(ctx, "user_settings", None) or {}
        force_tl = bool(us.get("forceTwelveLabs") or us.get("force_twelvelabs"))
        try:
            from core.vision_labels import vision_labels_are_weak

            vision_weak = bool(vision_labels_are_weak(ctx))
        except Exception:
            vision_weak = False

    tl_status, fusion_source = _tl_status_and_fusion_source(ctx)
    needs_deep = bool(
        force_tl
        or vision_weak
        or ("scene" not in lanes and "vision" in lanes)
        or tl_status in ("failed",)
    )

    return {
        "lanes_on": lanes,
        "vision_weak": vision_weak,
        "force_tl": force_tl,
        "needs_deep_teacher": needs_deep,
        "tl_status": tl_status,
        "fusion_source": fusion_source,
    }


def _transcript_segments(ctx: Any) -> List[Dict[str, Any]]:
    ac = getattr(ctx, "audio_context", None) or {}
    segs = []
    if isinstance(ac, dict):
        segs = ac.get("transcript_segments") or []
    out: List[Dict[str, Any]] = []
    for s in (segs or [])[:MAX_TRANSCRIPT_SEGMENTS]:
        if not isinstance(s, dict):
            continue
        text = str(s.get("text") or "").strip()
        if not text:
            continue
        out.append({"start": s.get("start"), "end": s.get("end"), "text": text[:500]})
    return out


def _vision_slice(ctx: Any) -> Dict[str, Any]:
    vc = getattr(ctx, "vision_context", None) or {}
    if not isinstance(vc, dict):
        return {"labels": [], "ocr": "", "landmarks": [], "logos": []}
    labels = [str(x) for x in (vc.get("label_names") or vc.get("labels") or []) if str(x).strip()][:40]
    landmarks = [str(x) for x in (vc.get("landmarks") or []) if str(x).strip()][:20]
    logos = [str(x) for x in (vc.get("logos") or []) if str(x).strip()][:20]
    ocr = str(vc.get("ocr_text") or "")[:2000]
    return {"labels": labels, "ocr": ocr, "landmarks": landmarks, "logos": logos}


def _shot_list_summary(ctx: Any) -> Dict[str, Any]:
    arts = getattr(ctx, "output_artifacts", None) or {}
    raw = arts.get("shot_list_v1") if isinstance(arts, dict) else None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = None
    if not isinstance(raw, (list, dict)):
        return {"count": 0, "shots": []}
    shots = raw if isinstance(raw, list) else (raw.get("shots") or raw.get("items") or [])
    if not isinstance(shots, list):
        return {"count": 0, "shots": []}
    slim = []
    for s in shots[:40]:
        if isinstance(s, dict):
            slim.append(
                {
                    "start": s.get("start") or s.get("t0") or s.get("start_s"),
                    "end": s.get("end") or s.get("t1") or s.get("end_s"),
                    "label": str(s.get("label") or s.get("description") or "")[:120],
                }
            )
    return {"count": len(shots), "shots": slim}


def _identity_slice(ctx: Any) -> Dict[str, Any]:
    arts = getattr(ctx, "output_artifacts", None) or {}
    ident = {}
    if isinstance(arts, dict):
        ident = _as_dict(arts.get("content_identity_v1"))
    if not ident:
        try:
            from core.content_identity import get_content_identity

            ident = get_content_identity(ctx) or {}
        except Exception:
            ident = {}
    if not isinstance(ident, dict):
        return {}
    heroes = []
    for h in (ident.get("hero_facts") or [])[:8]:
        if not isinstance(h, dict):
            continue
        heroes.append(
            {
                "text": str(h.get("text") or "")[:200],
                "class": str(h.get("class") or ""),
                "score": h.get("score"),
                "providers": list(h.get("providers") or [])[:8],
            }
        )
    domains = []
    for d in (ident.get("domain_tags") or [])[:5]:
        if isinstance(d, dict):
            domains.append({"tag": str(d.get("tag") or ""), "confidence": d.get("confidence")})
        elif d:
            domains.append({"tag": str(d), "confidence": None})
    return {
        "subject": str(ident.get("subject") or "")[:200],
        "activity": str(ident.get("activity") or "")[:200],
        "setting": str(ident.get("setting") or "")[:200],
        "hero_facts": heroes,
        "domain_tags": domains,
        "confidence": ident.get("confidence"),
        "providers_seen": list(ident.get("providers_seen") or [])[:20],
        "novel_content": bool(ident.get("novel_content")),
    }


def _hydration_brief(ctx: Any) -> Dict[str, Any]:
    arts = getattr(ctx, "output_artifacts", None) or {}
    hp = {}
    if isinstance(arts, dict):
        hp = _as_dict(arts.get("hydration_payload"))
    if not hp:
        hp = _as_dict(getattr(ctx, "hydration_payload", None))
    if not hp:
        return {}
    ev = hp.get("evidence") if isinstance(hp.get("evidence"), dict) else {}
    return {
        "category": hp.get("category"),
        "anchor_phrase": str(hp.get("anchor_phrase") or "")[:200],
        "fusion_summary": str(hp.get("fusion_summary") or "")[:500],
        "hydration_story": str(hp.get("hydration_story") or "")[:800],
        "signal_hashtags": list(hp.get("signal_hashtags") or [])[:20],
        "speech_phrase": str((ev.get("speech") or {}).get("phrase") or "")[:200]
        if isinstance(ev.get("speech"), dict)
        else "",
    }


def _fusion_summary(ctx: Any) -> str:
    vu = getattr(ctx, "video_understanding", None) or {}
    if isinstance(vu, dict):
        for key in ("scene_description", "summary", "title_suggestion"):
            t = str(vu.get(key) or "").strip()
            if t:
                return t[:1200]
    arts = getattr(ctx, "output_artifacts", None) or {}
    if isinstance(arts, dict):
        sf = _as_dict(arts.get("scene_fusion"))
        t = str(sf.get("scene_description") or "").strip()
        if t:
            return t[:1200]
    return ""


def _grounding_slice(ctx: Any) -> Dict[str, Any]:
    arts = getattr(ctx, "output_artifacts", None) or {}
    gs = {}
    if isinstance(arts, dict):
        gs = _as_dict(arts.get("grounding_score_v1"))
    score = gs.get("grounding_score")
    try:
        score_f = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_f = None
    band = None
    if score_f is not None:
        if score_f < 0.35:
            band = "low"
        elif score_f < 0.65:
            band = "mid"
        else:
            band = "high"
    return {
        "grounding_score": score_f,
        "band": band,
        "clue_hits": gs.get("clue_hits"),
        "clue_count": gs.get("clue_count"),
    }


def collect_local_keyframe_paths(ctx: Any, *, limit: int = MAX_KEYFRAMES) -> List[str]:
    paths: List[str] = []
    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict):
        for p in vc.get("vision_multi_frame_paths") or []:
            if p and os.path.isfile(str(p)):
                paths.append(str(p))
    for attr in ("thumbnail_path", "thumbnail_local_path"):
        p = getattr(ctx, attr, None)
        if p and os.path.isfile(str(p)):
            paths.append(str(p))
    arts = getattr(ctx, "output_artifacts", None) or {}
    if isinstance(arts, dict):
        for key in ("thumbnail_candidate_paths", "frame_paths"):
            raw = arts.get(key)
            if isinstance(raw, list):
                for p in raw:
                    if p and os.path.isfile(str(p)):
                        paths.append(str(p))
    seen = set()
    out: List[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= limit:
            break
    return out


def build_av_training_pack(
    ctx: Any,
    *,
    keyframe_r2_keys: Optional[Sequence[str]] = None,
    pack_r2_key: Optional[str] = None,
    consent: bool = True,
) -> Dict[str, Any]:
    ac = getattr(ctx, "audio_context", None) or {}
    transcript = ""
    if isinstance(ac, dict):
        transcript = str(ac.get("transcript") or "").strip()
    if not transcript:
        transcript = str(getattr(ctx, "ai_transcript", None) or "").strip()
    teacher = _teacher_flags(ctx)
    duration = getattr(ctx, "duration_seconds", None)
    if duration is None:
        duration = getattr(ctx, "duration", None)
    try:
        duration_f = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_f = None

    return {
        "v": AV_TRAINING_PACK_VERSION,
        "upload_id_hash": hash_upload_id(getattr(ctx, "upload_id", None)),
        "duration_s": duration_f,
        "clip_kind": _clip_kind(ctx),
        "transcript_segments": _transcript_segments(ctx),
        "transcript_chars": len(transcript),
        "vision": _vision_slice(ctx),
        "shot_list": _shot_list_summary(ctx),
        "fusion_summary": _fusion_summary(ctx),
        "content_identity_v1": _identity_slice(ctx),
        "hydration": _hydration_brief(ctx),
        "grounding_score_v1": _grounding_slice(ctx),
        "teacher_flags": teacher,
        "keyframe_r2_keys": list(keyframe_r2_keys or []),
        "pack_r2_key": pack_r2_key or "",
        "consent": bool(consent),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def artifact_meta_from_pack(pack: Dict[str, Any]) -> Dict[str, Any]:
    identity = pack.get("content_identity_v1") if isinstance(pack.get("content_identity_v1"), dict) else {}
    heroes = identity.get("hero_facts") or []
    top_class = ""
    if heroes and isinstance(heroes[0], dict):
        top_class = str(heroes[0].get("class") or "")
    gs = pack.get("grounding_score_v1") if isinstance(pack.get("grounding_score_v1"), dict) else {}
    flags = pack.get("teacher_flags") if isinstance(pack.get("teacher_flags"), dict) else {}
    return {
        "v": pack.get("v", AV_TRAINING_PACK_VERSION),
        "pack_r2_key": pack.get("pack_r2_key") or "",
        "keyframe_r2_keys": list(pack.get("keyframe_r2_keys") or []),
        "keyframe_count": len(pack.get("keyframe_r2_keys") or []),
        "consent": bool(pack.get("consent")),
        "created_at": pack.get("created_at"),
        "clip_kind": pack.get("clip_kind"),
        "upload_id_hash": pack.get("upload_id_hash"),
        "transcript_chars": pack.get("transcript_chars"),
        "identity_hero_fact_class": top_class,
        "identity_confidence": identity.get("confidence"),
        "grounding_score": gs.get("grounding_score"),
        "grounding_band": gs.get("band"),
        "needs_deep_teacher": bool(flags.get("needs_deep_teacher")),
        "tl_status": flags.get("tl_status"),
        "fusion_source": flags.get("fusion_source"),
        "lanes_on": list(flags.get("lanes_on") or []),
    }


def persist_av_training_pack_to_r2(
    pack: Dict[str, Any],
    *,
    user_id: Any,
    upload_id: Any,
    local_keyframe_paths: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from core.r2 import put_object_bytes

    uid = str(user_id)
    oid = str(upload_id)
    kf_keys: List[str] = []
    for i, path in enumerate(list(local_keyframe_paths or [])[:MAX_KEYFRAMES]):
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as f:
                body = f.read()
            if not body:
                continue
            key = keyframe_r2_key(uid, oid, i)
            ctype = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
            put_object_bytes(key, body, ctype)
            kf_keys.append(key)
        except Exception as e:
            logger.debug("av pack keyframe upload skipped (%s): %s", path, e)

    pack = dict(pack)
    pack["keyframe_r2_keys"] = kf_keys
    pack_key = pack_json_r2_key(uid, oid)
    pack["pack_r2_key"] = pack_key
    body = json.dumps(pack, ensure_ascii=False, default=str).encode("utf-8")
    put_object_bytes(pack_key, body, "application/json")
    meta = artifact_meta_from_pack(pack)
    meta["pack_bytes"] = len(body)
    return pack, meta


__all__ = [
    "AV_TRAINING_PACK_VERSION",
    "AV_TRAINING_PACK_ARTIFACT",
    "MAX_KEYFRAMES",
    "hash_upload_id",
    "pack_r2_prefix",
    "pack_json_r2_key",
    "keyframe_r2_key",
    "build_av_training_pack",
    "artifact_meta_from_pack",
    "collect_local_keyframe_paths",
    "persist_av_training_pack_to_r2",
]
