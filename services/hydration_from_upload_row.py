"""Build Thumbnail Studio–style hydration context from an ``uploads`` row (jsonb)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

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


def _parse_stored_hydration_payload(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Return canonical ``hydration_payload`` v2 from ``output_artifacts`` when present."""
    raw = artifacts.get("hydration_payload")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def hydration_payload_from_upload_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonical v2 hydration blob for Pikzels/M8 — prefers persisted ``hydration_payload``,
    else synthesizes from ``hydration_report`` + telemetry on the upload row.
    """
    artifacts = _json_obj(row.get("output_artifacts"))
    stored = _parse_stored_hydration_payload(artifacts)
    if stored.get("evidence") and int(stored.get("v") or 0) >= 2:
        return stored

    trill_meta = _json_obj(row.get("trill_metadata"))
    report = _json_obj(artifacts.get("hydration_report"))
    evidence = _json_obj(report.get("evidence"))
    geo = _json_obj(evidence.get("geo"))
    osd = _json_obj(evidence.get("osd"))
    music = _json_obj(evidence.get("music"))
    speech = _json_obj(evidence.get("speech"))
    vision = _json_obj(evidence.get("vision"))
    trill = _json_obj(evidence.get("trill"))
    telemetry = _json_obj(trill_meta.get("telemetry"))

    if not osd and telemetry:
        for key in ("max_speed_mph", "avg_speed_mph"):
            if telemetry.get(key) is not None:
                osd[key] = telemetry.get(key)

    story = _first_text(
        artifacts.get("hydration_story"),
        report.get("hydration_story"),
        report.get("story"),
        limit=700,
    )
    fusion = _first_text(
        artifacts.get("fusion_summary"),
        report.get("fusion_summary"),
        story,
        limit=1200,
    )
    sigs = report.get("signal_hashtags") or artifacts.get("signal_hashtags") or []
    if not isinstance(sigs, list):
        sigs = []

    return {
        "v": 2,
        "category": str(report.get("category") or artifacts.get("hydration_category") or "general"),
        "anchor_phrase": _first_text(report.get("anchor"), artifacts.get("anchor_phrase"), limit=220),
        "evidence": {
            "geo": geo,
            "osd": osd,
            "music": music,
            "speech": speech,
            "vision": vision if vision else {"labels": [], "ocr": "", "landmarks": [], "logos": []},
            "trill": trill,
        },
        "signal_hashtags": [str(x) for x in sigs if str(x).strip()][:24],
        "fusion_summary": fusion,
        "hydration_story": story,
        "trace_id": str(artifacts.get("hydration_trace_id") or report.get("trace_id") or ""),
        "category_source": "upload_row_synth",
    }


def flat_context_to_hydration_payload(ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a minimal canonical hydration_payload from Thumbnail Studio flat context
    (caption, geo, lat/lon, artist, speed, osd, trill fields).
    """
    if not isinstance(ctx, dict) or not ctx:
        return {}
    clean = normalize_hydration_context(ctx)
    if not clean:
        return {}

    geo: Dict[str, Any] = {}
    if clean.get("geo"):
        geo["display"] = clean["geo"]
    if clean.get("latitude") and clean.get("longitude"):
        try:
            geo["lat"] = float(str(clean["latitude"]).strip())
            geo["lon"] = float(str(clean["longitude"]).strip())
        except (TypeError, ValueError):
            pass

    osd: Dict[str, Any] = {}
    if clean.get("speed_mph"):
        try:
            osd["max_speed_mph"] = float(str(clean["speed_mph"]).strip())
        except (TypeError, ValueError):
            osd["max_speed_mph"] = clean["speed_mph"]
    if clean.get("osd_driver"):
        osd["driver_name"] = clean["osd_driver"]
    if clean.get("osd_recording_start"):
        osd["first_seen"] = clean["osd_recording_start"]

    music: Dict[str, Any] = {}
    if clean.get("artist"):
        music["artist"] = clean["artist"]
    if clean.get("track"):
        music["title"] = clean["track"]

    speech: Dict[str, Any] = {}
    if clean.get("speech"):
        speech["phrase"] = clean["speech"]

    trill: Dict[str, Any] = {}
    if clean.get("trill_bucket"):
        trill["bucket"] = clean["trill_bucket"]
    if clean.get("trill_score"):
        try:
            trill["score"] = float(str(clean["trill_score"]).strip())
        except (TypeError, ValueError):
            pass

    vision: Dict[str, Any] = {}
    if clean.get("vision_ocr"):
        vision["ocr"] = clean["vision_ocr"]
    if clean.get("vision_labels"):
        vision["labels"] = [
            x.strip() for x in str(clean["vision_labels"]).split(",") if x.strip()
        ][:12]

    story = clean.get("story") or clean.get("caption") or ""
    return {
        "v": 2,
        "category": "general",
        "anchor_phrase": "",
        "evidence": {
            "geo": geo,
            "osd": osd,
            "music": music,
            "speech": speech,
            "vision": vision or {"labels": [], "ocr": "", "landmarks": [], "logos": []},
            "trill": trill,
        },
        "signal_hashtags": [],
        "fusion_summary": story[:1200],
        "hydration_story": story[:700],
        "trace_id": "",
        "category_source": "flat_context",
    }


def hydration_context_from_upload_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same evidence bundle as ``GET /api/thumbnail-studio/hydration-context`` candidates,
    for a single upload row (must include output_artifacts, trill_metadata when available).
    """
    artifacts = _json_obj(row.get("output_artifacts"))
    trill_meta = _json_obj(row.get("trill_metadata"))
    report = _json_obj(artifacts.get("hydration_report"))
    evidence = _json_obj(report.get("evidence"))
    hp = hydration_payload_from_upload_row(row)
    ev = hp.get("evidence") if isinstance(hp.get("evidence"), dict) else {}
    geo = _json_obj(ev.get("geo"))
    osd = _json_obj(ev.get("osd"))
    music = _json_obj(ev.get("music"))
    speech = _json_obj(ev.get("speech"))
    trill = _json_obj(ev.get("trill"))
    vision = _json_obj(ev.get("vision"))
    transcript = _json_obj(evidence.get("transcript"))
    telemetry = _json_obj(trill_meta.get("telemetry"))

    platform_captions = _json_obj(artifacts.get("m8_platform_captions"))
    story = _first_text(
        hp.get("hydration_story"),
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

    lat = geo.get("lat") or telemetry.get("mid_lat") or telemetry.get("start_lat") or ""
    lon = geo.get("lon") or telemetry.get("mid_lon") or telemetry.get("start_lon") or ""
    speed_mph = osd.get("max_speed_mph") or telemetry.get("max_speed_mph") or ""
    context = normalize_hydration_context(
        {
            "story": story,
            "caption": caption,
            "geo": geo_text,
            "latitude": lat,
            "longitude": lon,
            "artist": music.get("artist") or "",
            "track": music.get("title") or "",
            "speed_mph": speed_mph,
            "osd_driver": osd.get("driver_name") or "",
            "osd_recording_start": osd.get("first_seen") or "",
            "speech": speech.get("phrase") or transcript.get("phrase") or "",
            "trill_bucket": trill.get("bucket") or "",
            "trill_score": trill.get("score") or "",
            "vision_ocr": str(vision.get("ocr") or "")[:180],
            "vision_labels": ", ".join(
                str(x) for x in (vision.get("labels") or [])[:8] if str(x).strip()
            ),
        }
    )
    sources: List[str] = []
    if caption:
        sources.append("caption/transcript")
    if geo_text or lat or lon:
        sources.append("telemetry/geo")
    if speed_mph or osd.get("driver_name"):
        sources.append("dashcam OSD")
    if music.get("artist") or music.get("title"):
        sources.append("audio recognition")
    if trill.get("bucket"):
        sources.append("Trill score")
    context["_sources"] = sources
    context["_upload_id"] = str(row.get("id") or "")
    context["_upload_title"] = _first_text(row.get("title"), row.get("ai_title"), row.get("filename"), limit=160)
    created = row.get("created_at")
    context["_created_at"] = created.isoformat() if hasattr(created, "isoformat") else str(created or "")
    return context
