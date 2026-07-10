"""
Place & entity evidence when .map / Trill telemetry is missing.

Pulls location and hard strings from every available lane:

* Google Vision landmarks (name + lat/lon → Nominatim reverse-geocode)
* Vision / VI OCR (beaches, monuments, stadiums, license plates, team names)
* Whisper transcript named entities (places / organizations)
* Dashcam OSD GPS (already handled elsewhere — we only fill gaps)

Persists ``ctx.place_evidence`` and ``output_artifacts.place_evidence_v1``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("uploadm8-worker")

# US-style plate: 1–3 letters + 1–4 digits, or digit-letter mixes common on dashcams.
_LICENSE_PLATE_RE = re.compile(
    r"\b(?=[A-Z0-9]*[A-Z][A-Z0-9]*\d|\d[A-Z0-9]*[A-Z])[A-Z0-9]{5,8}\b"
)

_BEACH_RE = re.compile(
    r"\b([A-Z][A-Za-z'&.\-]+(?:\s+[A-Z][A-Za-z'&.\-]+){0,3}\s+)?Beach\b",
    re.I,
)
_MONUMENT_RE = re.compile(
    r"\b([A-Z][A-Za-z'&.\-]+(?:\s+[A-Z][A-Za-z'&.\-]+){0,4}\s+)?"
    r"(Monument|Memorial|National\s+Park|State\s+Park|Historic\s+Site|Museum|Cathedral|Temple)\b",
    re.I,
)
_STADIUM_RE = re.compile(
    r"\b([A-Z][A-Za-z'&.\-]+(?:\s+[A-Z][A-Za-z'&.\-]+){0,3}\s+)?"
    r"(Stadium|Arena|Ballpark|Coliseum|Fieldhouse|Speedway)\b",
    re.I,
)

# Common pro / college team tokens (OCR / logos / transcript). Extend carefully.
_TEAM_TOKENS = frozenset(
    {
        "lakers",
        "celtics",
        "warriors",
        "bulls",
        "knicks",
        "nets",
        "heat",
        "spurs",
        "mavs",
        "mavericks",
        "cowboys",
        "patriots",
        "packers",
        "chiefs",
        "eagles",
        "giants",
        "jets",
        "bears",
        "steelers",
        "49ers",
        "niners",
        "yankees",
        "dodgers",
        "cubs",
        "mets",
        "red sox",
        "astros",
        "rangers",
        "bruins",
        "maple leafs",
        "canadiens",
        "blackhawks",
        "manchester united",
        "liverpool",
        "arsenal",
        "chelsea",
        "real madrid",
        "barcelona",
        "yankees",
        "seahawks",
        "ravens",
        "bills",
        "dolphins",
        "vikings",
        "saints",
        "falcons",
        "panthers",
        "cardinals",
        "raiders",
        "chargers",
        "broncos",
        "colts",
        "texans",
        "jaguars",
        "titans",
        "bengals",
        "browns",
        "lions",
        "commanders",
    }
)


def _uniq(items: List[str], *, limit: int = 12) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for raw in items:
        s = re.sub(r"\s+", " ", str(raw or "").strip())
        if not s or len(s) < 2:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s[:120])
        if len(out) >= limit:
            break
    return out


def _ocr_blobs(ctx: Any) -> str:
    chunks: List[str] = []
    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict):
        chunks.append(str(vc.get("ocr_text") or ""))
    vi = getattr(ctx, "video_intelligence", None) or {}
    if isinstance(vi, dict):
        for row in (vi.get("on_screen_text") or vi.get("text_detections") or [])[:40]:
            if isinstance(row, dict):
                chunks.append(str(row.get("text") or row.get("description") or ""))
            else:
                chunks.append(str(row))
    pe_prev = getattr(ctx, "place_evidence", None) or {}
    if isinstance(pe_prev, dict):
        chunks.append(" ".join(str(x) for x in (pe_prev.get("ocr_snippets") or [])[:8]))
    return "\n".join(chunks)


def _extract_license_plates(text: str) -> List[str]:
    if not text:
        return []
    # Prefer uppercase runs from OCR.
    up = text.upper()
    found = [m.group(0) for m in _LICENSE_PLATE_RE.finditer(up)]
    # Filter highway-like tokens (I-15, US101 already handled elsewhere).
    out: List[str] = []
    for p in found:
        if p.startswith("I") and p[1:].isdigit():
            continue
        if p in ("ERROR", "CAMERA", "ESCORT", "SPEED", "MPH"):
            continue
        out.append(p)
    return _uniq(out, limit=6)


def _extract_regex_places(text: str, pattern: re.Pattern[str]) -> List[str]:
    if not text:
        return []
    return _uniq([m.group(0).strip() for m in pattern.finditer(text)], limit=8)


def _extract_teams(text: str, logos: List[str]) -> List[str]:
    blob = f"{text} {' '.join(logos)}".lower()
    hits: List[str] = []
    for team in _TEAM_TOKENS:
        if team in blob:
            hits.append(team.title() if " " not in team else " ".join(w.capitalize() for w in team.split()))
    return _uniq(hits, limit=8)


def _landmark_entries(ctx: Any) -> List[Dict[str, Any]]:
    vc = getattr(ctx, "vision_context", None) or {}
    if not isinstance(vc, dict):
        return []
    raw = vc.get("landmarks") or []
    out: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                name = str(row).strip()
                if name:
                    out.append({"name": name})
                continue
            name = str(row.get("description") or row.get("name") or "").strip()
            if not name:
                continue
            entry: Dict[str, Any] = {"name": name, "score": row.get("score")}
            if row.get("lat") is not None and row.get("lon") is not None:
                try:
                    entry["lat"] = float(row["lat"])
                    entry["lon"] = float(row["lon"])
                except (TypeError, ValueError):
                    pass
            out.append(entry)
    # Fallback names-only list
    if not out:
        for name in vc.get("landmark_names") or []:
            n = str(name).strip()
            if n:
                out.append({"name": n})
    return out[:12]


def _telemetry_has_place(ctx: Any) -> bool:
    tel = getattr(ctx, "telemetry_data", None) or getattr(ctx, "telemetry", None)
    if tel is None:
        return False
    if getattr(tel, "location_city", None) or getattr(tel, "gazetteer_place_name", None):
        return True
    if getattr(tel, "location_display", None):
        return True
    pts = getattr(tel, "points", None) or []
    return bool(pts)


def _first_landmark_coords(landmarks: List[Dict[str, Any]]) -> Optional[Tuple[float, float, str]]:
    for lm in landmarks:
        lat, lon = lm.get("lat"), lm.get("lon")
        if lat is None or lon is None:
            continue
        try:
            la, lo = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        if -90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0 and not (la == 0.0 and lo == 0.0):
            return la, lo, str(lm.get("name") or "")
    return None


def extract_place_evidence(ctx: Any) -> Dict[str, Any]:
    """Synchronous extraction of place/entity evidence into a report dict."""
    landmarks = _landmark_entries(ctx)
    ocr = _ocr_blobs(ctx)
    logos: List[str] = []
    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict):
        logos = [str(x).strip() for x in (vc.get("logo_names") or []) if str(x).strip()]

    transcript_places: List[str] = []
    transcript_orgs: List[str] = []
    ac = getattr(ctx, "audio_context", None) or {}
    if isinstance(ac, dict):
        structured = ac.get("transcript_structured") or {}
        if isinstance(structured, dict):
            ne = structured.get("named_entities") or {}
            if isinstance(ne, dict):
                transcript_places = [str(x).strip() for x in (ne.get("places") or []) if str(x).strip()]
                transcript_orgs = [str(x).strip() for x in (ne.get("organizations") or []) if str(x).strip()]

    beaches = _extract_regex_places(ocr, _BEACH_RE)
    monuments = _extract_regex_places(ocr, _MONUMENT_RE)
    stadiums = _extract_regex_places(ocr, _STADIUM_RE)
    plates = _extract_license_plates(ocr)
    teams = _extract_teams(ocr + " " + " ".join(transcript_orgs), logos)

    # Landmark names that look like beaches/monuments
    for lm in landmarks:
        name = str(lm.get("name") or "")
        low = name.lower()
        if "beach" in low:
            beaches.append(name)
        if any(tok in low for tok in ("monument", "memorial", "park", "museum", "cathedral")):
            monuments.append(name)
        if any(tok in low for tok in ("stadium", "arena", "ballpark")):
            stadiums.append(name)

    places = _uniq(
        [str(lm.get("name") or "") for lm in landmarks]
        + transcript_places
        + beaches
        + monuments
        + stadiums,
        limit=16,
    )

    sources: List[str] = []
    if landmarks:
        sources.append("vision_landmark")
    if beaches or monuments or stadiums or plates:
        sources.append("ocr")
    if transcript_places:
        sources.append("transcript")
    if logos:
        sources.append("vision_logo")
    if _telemetry_has_place(ctx):
        sources.append("telemetry_or_osd")

    report: Dict[str, Any] = {
        "version": 1,
        "sources": sources,
        "landmarks": [{"name": lm.get("name"), "lat": lm.get("lat"), "lon": lm.get("lon")} for lm in landmarks],
        "places": places,
        "beaches": _uniq(beaches, limit=8),
        "monuments": _uniq(monuments, limit=8),
        "stadiums": _uniq(stadiums, limit=8),
        "license_plates": plates,
        "sports_teams": teams,
        "logos": logos[:8],
        "transcript_places": _uniq(transcript_places, limit=8),
        "has_map_telemetry": _telemetry_has_place(ctx),
        "geocode_from_landmark": None,
    }
    return report


async def backfill_place_from_vision(ctx: Any) -> Dict[str, Any]:
    """
    Extract place evidence and, when no .map/OSD place exists, reverse-geocode
    the best Vision landmark lat/lon onto ``ctx.telemetry_data``.
    """
    report = extract_place_evidence(ctx)
    setattr(ctx, "place_evidence", report)
    if not isinstance(getattr(ctx, "output_artifacts", None), dict):
        ctx.output_artifacts = {}
    ctx.output_artifacts["place_evidence_v1"] = report

    if report.get("has_map_telemetry"):
        return report

    coords = _first_landmark_coords(report.get("landmarks") or [])
    if not coords:
        return report

    lat, lon, landmark_name = coords
    try:
        from stages.context import TelemetryData
        from stages.telemetry_stage import (
            apply_padus_gazetteer_to_telemetry,
            reverse_geocode_details,
        )
    except Exception as e:
        logger.debug("[place_evidence] import for geocode failed: %s", e)
        return report

    tel = getattr(ctx, "telemetry_data", None) or getattr(ctx, "telemetry", None)
    if tel is None:
        tel = TelemetryData()
        ctx.telemetry_data = tel

    # Seed midpoint from landmark so PADUS/gazetteer can run.
    if getattr(tel, "mid_lat", None) is None:
        tel.mid_lat = lat
        tel.mid_lon = lon
    if getattr(tel, "start_lat", None) is None:
        tel.start_lat = lat
        tel.start_lon = lon

    try:
        details = await reverse_geocode_details(float(lat), float(lon))
    except Exception as e:
        logger.warning("[place_evidence] Nominatim failed for landmark %s: %s", landmark_name, e)
        details = None

    geo_note: Dict[str, Any] = {
        "landmark": landmark_name,
        "lat": lat,
        "lon": lon,
        "ok": bool(details),
    }
    if details:
        if not getattr(tel, "location_display", None):
            tel.location_display = details.get("location_display")
        if not getattr(tel, "location_city", None):
            tel.location_city = details.get("location_city")
        if not getattr(tel, "location_state", None):
            tel.location_state = details.get("location_state")
        if not getattr(tel, "location_country", None):
            tel.location_country = details.get("location_country")
        if not getattr(tel, "location_road", None):
            tel.location_road = details.get("location_road")
        geo_note["location_display"] = tel.location_display
        try:
            await apply_padus_gazetteer_to_telemetry(tel)
        except Exception as e:
            logger.debug("[place_evidence] PADUS/gazetteer skipped: %s", e)
        logger.info(
            "[place_evidence] geocoded Vision landmark %r (%.5f, %.5f) → %s",
            landmark_name,
            lat,
            lon,
            tel.location_display,
        )

    report["geocode_from_landmark"] = geo_note
    report["has_map_telemetry"] = _telemetry_has_place(ctx)
    if "vision_landmark" not in report["sources"]:
        report["sources"].append("vision_landmark")
    if geo_note.get("ok") and "nominatim" not in report["sources"]:
        report["sources"].append("nominatim")
    ctx.place_evidence = report
    ctx.output_artifacts["place_evidence_v1"] = report
    return report


def merge_place_evidence_into_pool(pool: Any, place_evidence: Optional[Dict[str, Any]]) -> None:
    """Attach extracted entities onto an EvidencePool (hydration_enforcer)."""
    if not place_evidence or not isinstance(place_evidence, dict):
        return
    # Prefer Vision landmark names already on pool; extend from report.
    for name in place_evidence.get("places") or []:
        n = str(name).strip()
        if n and n not in (pool.vision_landmarks or []):
            pool.vision_landmarks.append(n)
            if len(pool.vision_landmarks) >= 8:
                break

    # City from geocode if pool empty
    geo = place_evidence.get("geocode_from_landmark") or {}
    display = str(geo.get("location_display") or "")
    if display and not pool.city:
        # "City, ST, Country" style — take first segment as city hint
        pool.city = display.split(",")[0].strip()[:80] or pool.city

    setattr(pool, "place_beaches", list(place_evidence.get("beaches") or [])[:6])
    setattr(pool, "place_monuments", list(place_evidence.get("monuments") or [])[:6])
    setattr(pool, "place_stadiums", list(place_evidence.get("stadiums") or [])[:6])
    setattr(pool, "license_plates", list(place_evidence.get("license_plates") or [])[:6])
    setattr(pool, "sports_teams", list(place_evidence.get("sports_teams") or [])[:6])
    setattr(pool, "place_sources", list(place_evidence.get("sources") or []))


__all__ = [
    "extract_place_evidence",
    "backfill_place_from_vision",
    "merge_place_evidence_into_pool",
]
