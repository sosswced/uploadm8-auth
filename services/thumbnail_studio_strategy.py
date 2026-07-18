"""Shared readers/writers for Thumbnail Studio default strategy on users.preferences."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


def truthy_meta(value: Any) -> bool:
    """Loose truthiness for Studio feedback metadata flags (make_default, etc.)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "default")
    return False


def thumbnail_strategy_from_variant(
    *,
    job_row: Any,
    variant_id: str,
    variant_json: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist a selected Studio variation as upload-time prompt strategy, not fixed copy."""
    job = dict(job_row)
    v = dict(variant_json or {})
    from services.thumbnail_niches import normalize_niche

    job_yt_url = str(job.get("youtube_url") or "")[:500]
    job_yt_vid = str(job.get("youtube_video_id") or "")[:32]
    preview_r2 = str(v.get("preview_r2_key") or "").strip()[:500]
    job_id = str(job.get("id") or "")
    vid = str(variant_id or "")
    strategy = {
        "version": 2,
        "source": "thumbnail_studio_selected_variant",
        "job_id": job_id,
        "variant_id": vid,
        "source_job_id": job_id,
        "source_variant_id": vid,
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "format_key": str(v.get("format_key") or "")[:80],
        "layout_name": str(v.get("name") or "")[:120],
        "layout_pattern": str(v.get("layout_pattern") or "")[:240],
        "audience_niche": normalize_niche(str(job.get("niche") or ""))[:120],
        "reference_youtube_url": job_yt_url,
        "youtube_url": job_yt_url,
        "reference_youtube_video_id": job_yt_vid,
        "youtube_video_id": job_yt_vid,
        "reference_topic": str(job.get("topic") or "")[:200],
        "reference_strength": int(job.get("closeness") or 55),
        "competitor_gap_mode": bool(job.get("competitor_gap_mode")),
        "persona_id": str(job.get("persona_id") or "")[:80],
        "hydration_focus": str(v.get("hydration_focus") or "")[:120],
        "selected_headline_style": str(v.get("headline") or "")[:120],
        "text_position": str(v.get("text_position") or "")[:40],
        "contrast_profile": str(v.get("contrast_profile") or "")[:40],
        "emotion": str(v.get("emotion") or "")[:60],
        "face_scale": float(v.get("face_scale") or 0) if v.get("face_scale") is not None else 0,
        "preview_r2_key": preview_r2,
        "apply_mode": "cover_direct" if preview_r2 else "strategy_only",
        "platforms": {
            "youtube": {"apply_mode": "cover_direct" if preview_r2 else "strategy_only"},
            "tiktok": {"apply_mode": "letterbox" if preview_r2 else "fresh_generate"},
            "instagram": {"apply_mode": "letterbox" if preview_r2 else "fresh_generate"},
        },
    }
    return {k: val for k, val in strategy.items() if val not in ("", None)}


def read_thumbnail_studio_default_strategy(prefs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Resolve the Studio "make default" strategy object from preference blobs.

    Canonical keys (written by Thumbnail Studio feedback):
      thumbnailStudioDefaultStrategy / thumbnail_studio_default_strategy
    Legacy aliases (older readers / niche merge):
      thumbnailDefaultStrategy / thumbnail_default_strategy
    """
    if not isinstance(prefs, Mapping):
        return {}
    for key in (
        "thumbnailStudioDefaultStrategy",
        "thumbnail_studio_default_strategy",
        "thumbnailDefaultStrategy",
        "thumbnail_default_strategy",
    ):
        raw = prefs.get(key)
        if isinstance(raw, dict) and raw:
            return dict(raw)
    return {}


def strategy_audience_niche(prefs: Optional[Mapping[str, Any]], default: str = "general") -> str:
    nested = read_thumbnail_studio_default_strategy(prefs)
    niche = nested.get("audience_niche") or nested.get("niche")
    if niche is None and isinstance(prefs, Mapping):
        niche = prefs.get("audience_niche") or prefs.get("audienceNiche")
    s = str(niche or default).strip().lower() or default
    return s


def strategy_summary_line(prefs: Optional[Mapping[str, Any]]) -> Optional[str]:
    """One-line human summary for Upload/Settings UI."""
    strat = read_thumbnail_studio_default_strategy(prefs)
    if not strat:
        return None
    parts = []
    layout = (
        strat.get("layout_name")
        or strat.get("layout_pattern")
        or strat.get("format_key")
        or strat.get("layout")
    )
    if layout:
        parts.append(f"layout {str(layout).replace('_', ' ')}")
    niche = strat.get("audience_niche") or strat.get("niche")
    if niche:
        parts.append(f"niche {niche}")
    yt = strat.get("reference_youtube_url") or strat.get("youtube_url")
    if yt:
        parts.append("YouTube ref set")
    persona = strat.get("persona_id") or strat.get("persona_name")
    if persona:
        parts.append("persona linked")
    preview = str(strat.get("preview_r2_key") or strat.get("previewR2Key") or "").strip()
    mode = str(strat.get("apply_mode") or strat.get("applyMode") or "").strip().lower()
    if preview and mode == "cover_direct":
        parts.append("Studio winner image → YT/FB covers")
    elif preview:
        parts.append("Studio winner as Pikzels support image")
    if not parts:
        return "Studio default strategy saved"
    return " · ".join(parts)
