"""
Extract specific visual entities from Vision + Video Intelligence for hydration stories.

Google label detection alone often returns coarse tags (car, land vehicle, windshield).
This module prefers logos, web entities, landmarks, localized objects, and filtered
segment labels so captions/thumbnails can reference brands, models, flora, and signage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.vision_labels import (
    filter_vision_labels_for_context,
    filter_vision_labels_for_hashtags,
    is_generic_vision_label,
    vision_label_slug,
)

# VI / Vision object descriptions that are not useful in user-facing copy.
_GENERIC_OBJECT_SLUGS = frozenset(
    {
        "vehicle",
        "car",
        "automobile",
        "motorvehicle",
        "landvehicle",
        "transport",
        "transportation",
        "wheel",
        "tire",
        "tyre",
        "person",
        "people",
        "human",
        "object",
        "thing",
        "entity",
    }
)

# Label hints for bucketing non-generic VI/Vision labels.
_VEHICLE_RE = re.compile(
    r"\b("
    r"sports?\s*car|sedan|coupe|hatchback|suv|pickup|truck|van|bus|motorcycle|"
    r"bicycle|scooter|semi|tractor|rv|camper|drift|racing|supercar|hypercar|"
    r"honda|toyota|ford|chevrolet|chevy|bmw|mercedes|audi|nissan|mazda|subaru|"
    r"jeep|dodge|ram|tesla|porsche|lamborghini|ferrari|corvette|mustang|camaro|"
    r"civic|accord|wrangler|f-150|model\s*[sxy3]|type\s*r"
    r")\b",
    re.I,
)
_FLORA_RE = re.compile(
    r"\b("
    r"flower|flowers|floral|bloom|blooms|rose|roses|sunflower|tulip|daisy|"
    r"lily|orchid|lavender|cherry\s*blossom|tree|palm|cactus|succulent|garden|"
    r"meadow|wildflower"
    r")\b",
    re.I,
)
_SIGNAGE_RE = re.compile(
    r"\b("
    r"stop\s*sign|yield|speed\s*limit|highway\s*sign|road\s*sign|street\s*sign|"
    r"exit\s*sign|billboard|traffic\s*light|signal|interstate|route\s*\d+|"
    r"us\s*\d+|i-?\d{1,3}|sr-?\d+|mph|km/h"
    r")\b",
    re.I,
)
_RESTAURANT_RE = re.compile(
    r"\b("
    r"restaurant|cafe|coffee|diner|grill|kitchen|bar\s*&\s*grill|"
    r"mcdonald|mcdonalds|burger\s*king|wendy|taco\s*bell|chipotle|"
    r"starbucks|dunkin|in-?n-?out|chick-?fil-?a|subway|kfc|pizza\s*hut|"
    r"domino|five\s*guys|shake\s*shack|whataburger|jack\s*in\s*the\s*box"
    r")\b",
    re.I,
)


@dataclass
class VisualEntityBundle:
    brands: List[str] = field(default_factory=list)
    vehicles: List[str] = field(default_factory=list)
    flora: List[str] = field(default_factory=list)
    signage: List[str] = field(default_factory=list)
    restaurants: List[str] = field(default_factory=list)
    landmarks: List[str] = field(default_factory=list)
    scene_labels: List[str] = field(default_factory=list)
    web_entities: List[str] = field(default_factory=list)


def _clean_phrase(raw: Any, *, max_len: int = 80) -> str:
    text = re.sub(r"\s*\([0-9.]+\)\s*$", "", str(raw or "").strip())
    text = re.sub(r"\s+", " ", text)
    return text[:max_len] if text else ""


def _append_unique(bucket: List[str], phrase: str, *, limit: int = 8) -> None:
    p = _clean_phrase(phrase)
    if not p or len(p) < 2:
        return
    key = p.lower()
    if any(x.lower() == key for x in bucket):
        return
    if len(bucket) >= limit:
        return
    bucket.append(p)


def _is_generic_object_label(raw: Any) -> bool:
    slug = vision_label_slug(raw)
    if not slug:
        return True
    if slug in _GENERIC_OBJECT_SLUGS:
        return True
    return is_generic_vision_label(raw, min_specific_len=3)


def _classify_scene_label(label: str, bundle: VisualEntityBundle) -> None:
    if _is_generic_object_label(label):
        return
    if _VEHICLE_RE.search(label):
        _append_unique(bundle.vehicles, label)
        return
    if _FLORA_RE.search(label):
        _append_unique(bundle.flora, label)
        return
    if _SIGNAGE_RE.search(label):
        _append_unique(bundle.signage, label)
        return
    if _RESTAURANT_RE.search(label):
        _append_unique(bundle.restaurants, label)
        return
    _append_unique(bundle.scene_labels, label)


def _rows_from_vi_logos(rows: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for row in rows or []:
        if isinstance(row, dict):
            desc = row.get("description") or row.get("entity") or ""
        else:
            desc = row
        p = _clean_phrase(desc)
        if p:
            out.append(p)
    return out


def _rows_from_vi_objects(rows: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for row in rows or []:
        if isinstance(row, dict):
            desc = row.get("description") or ""
        else:
            desc = row
        p = _clean_phrase(desc)
        if p and not _is_generic_object_label(p):
            out.append(p)
    return out


def collect_visual_entities(
    *,
    vision_context: Optional[Dict[str, Any]] = None,
    video_intelligence: Optional[Dict[str, Any]] = None,
    video_intelligence_context: Optional[Dict[str, Any]] = None,
    category: str = "general",
    filename: str = "",
) -> VisualEntityBundle:
    """Merge Vision + VI signals into specific entity buckets."""
    bundle = VisualEntityBundle()
    vc = vision_context if isinstance(vision_context, dict) else {}
    vi = video_intelligence if isinstance(video_intelligence, dict) else {}
    vic = video_intelligence_context if isinstance(video_intelligence_context, dict) else {}

    # Vision logos / landmarks / web / localized objects
    def _ingest_brand(name: str) -> None:
        if _RESTAURANT_RE.search(name):
            _append_unique(bundle.restaurants, name)
        else:
            _append_unique(bundle.brands, name)

    for name in vc.get("logo_names") or []:
        _ingest_brand(str(name))
    for row in vc.get("logos") or []:
        if isinstance(row, dict) and row.get("description"):
            _ingest_brand(str(row["description"]))

    for name in vc.get("landmark_names") or []:
        _append_unique(bundle.landmarks, str(name))
    for row in vc.get("landmarks") or []:
        if isinstance(row, dict) and row.get("description"):
            _append_unique(bundle.landmarks, str(row["description"]))

    for row in vc.get("web_entities") or []:
        if isinstance(row, dict):
            desc = row.get("description") or ""
            score = float(row.get("score") or 0.0)
            if score >= 0.45 and desc:
                _append_unique(bundle.web_entities, str(desc))
                if _VEHICLE_RE.search(desc):
                    _append_unique(bundle.vehicles, desc)
                elif _RESTAURANT_RE.search(desc):
                    _append_unique(bundle.restaurants, desc)

    for guess in vc.get("web_best_guess") or []:
        g = _clean_phrase(guess)
        if g and not is_generic_vision_label(g):
            _append_unique(bundle.web_entities, g)

    for row in vc.get("localized_objects") or []:
        if isinstance(row, dict):
            name = row.get("name") or row.get("description") or ""
            if name and not _is_generic_object_label(name):
                _classify_scene_label(str(name), bundle)

    label_pool: List[str] = []
    label_pool.extend(vc.get("label_names") or [])
    for row in vc.get("labels") or []:
        if isinstance(row, dict) and row.get("description"):
            label_pool.append(str(row["description"]))
    label_pool = filter_vision_labels_for_context(
        label_pool,
        category=category,
        filename=filename,
        min_specific_len=4,
    )
    for lbl in label_pool:
        _classify_scene_label(lbl, bundle)

    # Video Intelligence logos / objects / labels (both ctx slots)
    for src in (vi, vic):
        if not src:
            continue
        for brand in _rows_from_vi_logos(src.get("logos") or []):
            _ingest_brand(brand)
        for obj in _rows_from_vi_objects(src.get("object_tracks") or []):
            _classify_scene_label(obj, bundle)
        for key in ("top_labels", "shot_labels", "segment_labels"):
            raw = src.get(key) or []
            if not isinstance(raw, list):
                continue
            for item in raw:
                if isinstance(item, dict):
                    desc = item.get("description") or ""
                else:
                    desc = str(item)
                desc = _clean_phrase(desc)
                if desc and not _is_generic_object_label(desc):
                    _classify_scene_label(desc, bundle)

    # OCR: highways, speed HUD, business names on signs
    ocr_parts: List[str] = []
    if vc.get("ocr_text"):
        ocr_parts.append(str(vc["ocr_text"]))
    for src in (vi, vic):
        for row in (src.get("on_screen_text") or src.get("text_detections") or []):
            if isinstance(row, dict):
                t = str(row.get("text") or "").strip()
            else:
                t = str(row).strip()
            if t:
                ocr_parts.append(t)
    ocr_blob = " | ".join(ocr_parts)[:2000]
    if ocr_blob:
        for m in re.finditer(r"\b(I-?\s?\d{1,3}|US\s?\d{1,3}|SR-?\s?\d{1,4}|Route\s+\d+)\b", ocr_blob, re.I):
            _append_unique(bundle.signage, m.group(1).replace(" ", ""))
        for m in re.finditer(r"\b(\d{1,3})\s*MPH\b", ocr_blob, re.I):
            _append_unique(bundle.signage, f"{m.group(1)} MPH HUD")
        if _RESTAURANT_RE.search(ocr_blob):
            for m in _RESTAURANT_RE.finditer(ocr_blob):
                _append_unique(bundle.restaurants, m.group(0))

    return bundle


def visual_entity_story_clauses(bundle: VisualEntityBundle) -> List[str]:
    """Short factual clauses for ``build_hydration_story_text``."""
    clauses: List[str] = []
    if bundle.brands:
        clauses.append("Brands/logos: " + ", ".join(bundle.brands[:6]) + ".")
    if bundle.vehicles:
        clauses.append("Vehicles/models: " + ", ".join(bundle.vehicles[:6]) + ".")
    if bundle.restaurants:
        clauses.append("Food/retail seen: " + ", ".join(bundle.restaurants[:5]) + ".")
    if bundle.flora:
        clauses.append("Flora/scenery: " + ", ".join(bundle.flora[:5]) + ".")
    if bundle.landmarks:
        clauses.append("Landmarks/places: " + ", ".join(bundle.landmarks[:4]) + ".")
    if bundle.signage:
        clauses.append("Signage/HUD: " + ", ".join(bundle.signage[:6]) + ".")
    if bundle.web_entities and not bundle.brands and not bundle.vehicles:
        clauses.append("Web-matched entities: " + ", ".join(bundle.web_entities[:5]) + ".")
    if bundle.scene_labels:
        clauses.append("Scene specifics: " + ", ".join(bundle.scene_labels[:6]) + ".")
    return clauses


def build_scene_hook_line(
    *,
    place: str = "",
    max_speed_mph: float = 0.0,
    music_artist: str = "",
    music_title: str = "",
    bundle: Optional[VisualEntityBundle] = None,
    max_chars: int = 160,
) -> str:
    """
    One-line creative hook from fused signals, e.g.
    "racing near Dayton in a Honda Civic vibe to Chief Keef".
    """
    parts: List[str] = []
    if max_speed_mph >= 45:
        parts.append("fast run")
    elif max_speed_mph >= 25:
        parts.append("cruise")

    veh = ""
    if bundle:
        if bundle.vehicles:
            veh = bundle.vehicles[0]
        elif bundle.brands:
            veh = bundle.brands[0]
    if veh:
        parts.append(f"in a {veh}")

    place = _clean_phrase(place, max_len=48)
    if place:
        parts.append(f"near {place}")

    music_bits = " ".join(x for x in (music_artist, music_title) if x).strip()
    if music_bits:
        parts.append(f"vibing to {music_bits}")

    if not parts:
        return ""
    line = " ".join(parts)
    line = line[0].upper() + line[1:] if line else line
    if len(line) > max_chars:
        line = line[: max_chars - 1].rstrip() + "…"
    return line
