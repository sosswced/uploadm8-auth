"""Build Thumbnail Studio–style hydration context from an ``uploads`` row (jsonb)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from services.thumbnail_studio import normalize_hydration_context


def _json_obj(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _first_text(*values: Any, limit: int = 420) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = " ".join(str(x).strip() for x in value if str(x).strip())
        text = " ".join(str(value).strip().split())
        if text:
            return text[:limit]
    return ""


def hydration_context_from_upload_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same evidence bundle as ``GET /api/thumbnail-studio/hydration-context`` candidates,
    for a single upload row (must include output_artifacts, trill_metadata when available).
    """
    artifacts = _json_obj(row.get("output_artifacts"))
    trill_meta = _json_obj(row.get("trill_metadata"))
    report = _json_obj(artifacts.get("hydration_report"))
    evidence = _json_obj(report.get("evidence"))
    geo = _json_obj(evidence.get("geo"))
    music = _json_obj(evidence.get("music"))
    transcript = _json_obj(evidence.get("transcript"))
    telemetry = _json_obj(trill_meta.get("telemetry"))

    platform_captions = _json_obj(artifacts.get("m8_platform_captions"))
    story = _first_text(
        artifacts.get("hydration_story"),
        report.get("hydration_story"),
        report.get("story"),
        limit=620,
    )
    caption = _first_text(
        story,
        row.get("caption"),
        row.get("ai_caption"),
        platform_captions.get("youtube"),
        platform_captions.get("instagram"),
        platform_captions.get("tiktok"),
        transcript.get("phrase"),
        report.get("anchor"),
    )

    geo_bits = [
        telemetry.get("location_display"),
        geo.get("road") or telemetry.get("location_road"),
        geo.get("city") or telemetry.get("location_city"),
        geo.get("state") or telemetry.get("location_state"),
        geo.get("gazetteer_place") or telemetry.get("gazetteer_place_name"),
        geo.get("protected_area") or telemetry.get("padus_unit_name"),
    ]
    geo_text = " · ".join(dict.fromkeys(str(x).strip() for x in geo_bits if str(x or "").strip()))[:220]

    lat = telemetry.get("mid_lat") or telemetry.get("start_lat") or ""
    lon = telemetry.get("mid_lon") or telemetry.get("start_lon") or ""
    context = normalize_hydration_context(
        {
            "story": story,
            "caption": caption,
            "geo": geo_text,
            "latitude": lat,
            "longitude": lon,
            "artist": music.get("artist") or "",
            "track": music.get("title") or "",
        }
    )
    sources: List[str] = []
    if caption:
        sources.append("caption/transcript")
    if geo_text or lat or lon:
        sources.append("telemetry/geo")
    if music.get("artist") or music.get("title"):
        sources.append("audio recognition")
    context["_sources"] = sources
    context["_upload_id"] = str(row.get("id") or "")
    context["_upload_title"] = _first_text(row.get("title"), row.get("ai_title"), row.get("filename"), limit=160)
    created = row.get("created_at")
    context["_created_at"] = created.isoformat() if hasattr(created, "isoformat") else str(created or "")
    return context
