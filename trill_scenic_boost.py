"""
Weighted scenic / route-significance boost for Trill scores.

Factors: state-line crossings, welcome signs, national parks & monuments,
major landmarks, oceans/lakes/rivers, scenic highways, mountain terrain.
Applied after Vision + Video Intelligence + telemetry geo enrichment.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from core.vision_entities import collect_visual_entities
from stages.context import JobContext, TrillScore

logger = logging.getLogger("uploadm8-worker.trill_scenic")

_SCENIC_MAX_BOOST = float(os.environ.get("TRILL_SCENIC_MAX_BOOST", "28") or "28")
_SCENIC_MAX_BOOST = max(0.0, min(_SCENIC_MAX_BOOST, 40.0))

# Per-factor caps (weighted scale; sum clamped by _SCENIC_MAX_BOOST).
_WEIGHTS: Dict[str, float] = {
    "state_crossing": float(os.environ.get("TRILL_SCENIC_WEIGHT_STATE_CROSSING", "8") or 8),
    "welcome_sign": float(os.environ.get("TRILL_SCENIC_WEIGHT_WELCOME_SIGN", "5") or 5),
    "national_park": float(os.environ.get("TRILL_SCENIC_WEIGHT_NATIONAL_PARK", "10") or 10),
    "major_landmark": float(os.environ.get("TRILL_SCENIC_WEIGHT_MAJOR_LANDMARK", "12") or 12),
    "water_body": float(os.environ.get("TRILL_SCENIC_WEIGHT_WATER", "8") or 8),
    "coastal": float(os.environ.get("TRILL_SCENIC_WEIGHT_COASTAL", "6") or 6),
    "scenic_highway": float(os.environ.get("TRILL_SCENIC_WEIGHT_HIGHWAY", "5") or 5),
    "mountain_terrain": float(os.environ.get("TRILL_SCENIC_WEIGHT_MOUNTAIN", "6") or 6),
}

_WELCOME_STATE_RE = re.compile(
    r"welcome\s+to\s+([A-Za-z][A-Za-z\s]{2,24}?)(?:\s*!|,|\s*$|\s+population)",
    re.IGNORECASE,
)
_WATER_RE = re.compile(
    r"\b("
    r"ocean|sea|pacific|atlantic|gulf|lake|river|bay|estuary|harbor|harbour|"
    r"beach|coast|coastline|shore|shoreline|waterfall|reservoir|creek|sound|"
    r"strait|channel|lagoon|inlet|marina|pier"
    r")\b",
    re.IGNORECASE,
)
_MOUNTAIN_RE = re.compile(
    r"\b("
    r"mountain|mount|peak|summit|canyon|gorge|mesa|butte|cliff|ridge|valley|"
    r"alps|sierra|rockies|appalachian|volcano|crater|national\s+forest"
    r")\b",
    re.IGNORECASE,
)
_PARK_RE = re.compile(
    r"\b("
    r"national\s+park|national\s+monument|national\s+forest|wilderness|"
    r"state\s+park|memorial|preserve|recreation\s+area|scenic\s+byway"
    r")\b",
    re.IGNORECASE,
)
_HIGHWAY_RE = re.compile(
    r"\b("
    r"interstate|i-?\s?\d{1,3}\b|us\s?(?:route|hwy|highway)\s?\d+|"
    r"route\s?66|historic\s+route|scenic\s+byway|parkway|turnpike"
    r")\b",
    re.IGNORECASE,
)
_MAJOR_LANDMARKS = (
    "grand canyon",
    "yosemite",
    "yellowstone",
    "zion",
    "monument valley",
    "mount rushmore",
    "golden gate",
    "hoover dam",
    "arches",
    "bryce canyon",
    "death valley",
    "niagara",
    "everglades",
    "glacier national",
    "rocky mountain national",
    "great smoky",
    "big sur",
    "redwood",
    "sequoia",
)

_US_STATE_NAMES: Dict[str, str] = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "district of columbia": "DC",
}


def _enabled() -> bool:
    return os.environ.get("TRILL_SCENIC_BOOST_ENABLED", "true").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _bucket_from_score(score: int) -> str:
    if score >= 90:
        return "gloryBoy"
    if score >= 80:
        return "euphoric"
    if score >= 60:
        return "sendIt"
    if score >= 40:
        return "spirited"
    return "chill"


def _normalize_state_token(raw: str) -> Optional[str]:
    t = re.sub(r"\s+", " ", str(raw or "").strip().lower())
    if not t:
        return None
    if len(t) == 2 and t.isalpha():
        return t.upper()
    if t in _US_STATE_NAMES:
        return _US_STATE_NAMES[t]
    for name, abbr in _US_STATE_NAMES.items():
        if name in t or t in name:
            return abbr
    return None


def _states_from_text(blob: str) -> Set[str]:
    out: Set[str] = set()
    if not blob:
        return out
    for m in _WELCOME_STATE_RE.finditer(blob):
        abbr = _normalize_state_token(m.group(1))
        if abbr:
            out.add(abbr)
    lower = blob.lower()
    for name, abbr in _US_STATE_NAMES.items():
        if re.search(rf"\b{re.escape(name)}\b", lower):
            out.add(abbr)
    return out


def _ocr_and_vi_text(ctx: JobContext) -> str:
    chunks: List[str] = []
    vc = ctx.vision_context if isinstance(ctx.vision_context, dict) else {}
    if vc.get("ocr_text"):
        chunks.append(str(vc["ocr_text"]))
    for src in (ctx.video_intelligence, ctx.video_intelligence_context):
        if not isinstance(src, dict):
            continue
        for row in src.get("on_screen_text") or src.get("text_detections") or []:
            if isinstance(row, dict):
                t = str(row.get("text") or "").strip()
            else:
                t = str(row).strip()
            if t:
                chunks.append(t)
    return " | ".join(chunks)[:8000]


def _label_blob(ctx: JobContext) -> str:
    parts: List[str] = []
    vc = ctx.vision_context if isinstance(ctx.vision_context, dict) else {}
    parts.extend(str(x) for x in (vc.get("label_names") or []))
    for src in (ctx.video_intelligence, ctx.video_intelligence_context):
        if not isinstance(src, dict):
            continue
        for key in ("top_labels", "segment_labels", "shot_labels"):
            for item in src.get(key) or []:
                if isinstance(item, dict):
                    parts.append(str(item.get("description") or ""))
                else:
                    parts.append(str(item))
    return " ".join(parts).lower()


def compute_scenic_breakdown(ctx: JobContext) -> Tuple[float, Dict[str, float], List[str]]:
    """Return (total_boost, breakdown_dict, human_factor_labels)."""
    breakdown: Dict[str, float] = {}
    factors: List[str] = []
    tel = ctx.telemetry_data or ctx.telemetry

    text_blob = _ocr_and_vi_text(ctx)
    labels = _label_blob(ctx)
    combined = f"{text_blob} {labels}".lower()

    states: Set[str] = set()
    if tel:
        for raw in (
            getattr(tel, "location_state", None),
            getattr(tel, "gazetteer_state_usps", None),
            getattr(tel, "location_display", None),
            getattr(tel, "location_start_display", None),
        ):
            abbr = _normalize_state_token(str(raw or ""))
            if abbr:
                states.add(abbr)
    states |= _states_from_text(text_blob)
    if len(states) >= 2:
        breakdown["state_crossing"] = _WEIGHTS["state_crossing"]
        factors.append(f"state crossing ({', '.join(sorted(states)[:4])})")
    elif _WELCOME_STATE_RE.search(text_blob):
        breakdown["welcome_sign"] = _WEIGHTS["welcome_sign"]
        factors.append("welcome sign")

    padus_name = ""
    if tel:
        padus_name = str(getattr(tel, "padus_unit_name", None) or "")
        if getattr(tel, "near_padus", False) and not padus_name:
            padus_name = "protected area"
    if padus_name and (_PARK_RE.search(padus_name) or "national" in padus_name.lower()):
        breakdown["national_park"] = _WEIGHTS["national_park"]
        factors.append(f"near {padus_name[:60]}")

    vc = ctx.vision_context if isinstance(ctx.vision_context, dict) else {}
    landmarks = [str(x) for x in (vc.get("landmark_names") or [])]
    entity_bundle = collect_visual_entities(
        vision_context=vc,
        video_intelligence=ctx.video_intelligence if isinstance(ctx.video_intelligence, dict) else None,
        video_intelligence_context=ctx.video_intelligence_context
        if isinstance(ctx.video_intelligence_context, dict)
        else None,
        category=str(getattr(ctx, "thumbnail_category", None) or "general"),
        filename=str(getattr(ctx, "filename", "") or ""),
    )
    landmarks.extend(entity_bundle.landmarks)
    landmark_hit = ""
    for lm in landmarks:
        low = lm.lower()
        if any(m in low for m in _MAJOR_LANDMARKS):
            landmark_hit = lm
            break
    if landmark_hit:
        breakdown["major_landmark"] = _WEIGHTS["major_landmark"]
        factors.append(f"landmark {landmark_hit[:50]}")
    elif landmarks:
        breakdown["major_landmark"] = min(_WEIGHTS["major_landmark"], 8.0)
        factors.append(f"landmark {landmarks[0][:50]}")

    if _WATER_RE.search(combined):
        breakdown["water_body"] = _WEIGHTS["water_body"]
        factors.append("water feature")
    if re.search(r"\b(coast|coastal|ocean|beach|shore|pacific|atlantic|gulf)\b", combined):
        breakdown["coastal"] = _WEIGHTS["coastal"]
        factors.append("coastal/ocean-side")

    road = ""
    if tel:
        road = str(getattr(tel, "location_road", None) or "")
    if road and _HIGHWAY_RE.search(road):
        breakdown["scenic_highway"] = _WEIGHTS["scenic_highway"]
        factors.append(f"highway {road[:40]}")
    elif _HIGHWAY_RE.search(text_blob):
        breakdown["scenic_highway"] = min(_WEIGHTS["scenic_highway"], 4.0)
        factors.append("highway signage")

    if _MOUNTAIN_RE.search(combined) or _PARK_RE.search(combined):
        breakdown["mountain_terrain"] = _WEIGHTS["mountain_terrain"]
        factors.append("mountain/canyon terrain")

    total = min(_SCENIC_MAX_BOOST, sum(breakdown.values()))
    return total, breakdown, factors


def apply_scenic_trill_boost(ctx: JobContext) -> Optional[TrillScore]:
    """Add weighted scenic boost to existing Trill score on ctx."""
    if not _enabled():
        return None
    trill = ctx.trill_score or ctx.trill
    if not trill or getattr(trill, "score", None) is None:
        return None

    boost, breakdown, factors = compute_scenic_breakdown(ctx)
    if boost <= 0:
        return trill

    try:
        base = int(getattr(trill, "base_score", 0) or 0)
    except (TypeError, ValueError):
        base = 0
    if base <= 0:
        base = int(trill.score)
        trill.base_score = base

    new_score = min(100, int(round(base + boost)))
    trill.scenic_boost = round(boost, 2)
    trill.scenic_breakdown = breakdown
    trill.scenic_factors = factors
    trill.score = new_score
    trill.bucket = _bucket_from_score(new_score)

    try:
        from stages.telemetry_stage import get_trill_modifiers

        tel = ctx.telemetry_data or ctx.telemetry
        max_mph = float(getattr(tel, "max_speed_mph", 0) or 0) if tel else 0.0
        modifier, hashtags = get_trill_modifiers(new_score, max_mph, trill.bucket)
        trill.title_modifier = modifier
        trill.hashtags = hashtags
    except Exception:
        pass

    ctx.trill_score = trill
    ctx.trill = trill
    logger.info(
        "[%s] Trill scenic boost +%.1f → %s (%s): %s",
        getattr(ctx, "upload_id", "?"),
        boost,
        new_score,
        trill.bucket,
        "; ".join(factors[:6]),
    )
    return trill
