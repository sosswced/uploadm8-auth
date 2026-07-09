"""
Grade-A visual recognition rollup from Google Cloud Vision + Video Intelligence.

Produces a structured catalog (vehicles, brands, food, colors, products, text, …)
and a narrative summary for captions, M8, hydration, and thumbnails — not only
dashcam/Trill uploads.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.vision_labels import is_redundant_vision_label, resolve_ambient_profiles
from core.visual_entity_taxonomy import (
    YEAR_MAKE_MODEL_RE,
    classify_phrase,
    empty_catalog,
    narrative_bucket_labels,
    niche_bucket_order,
)
from services.thumbnail_niches import normalize_niche

logger = logging.getLogger("uploadm8.google_visual_recognition")

_WEB_MIN = float(os.environ.get("VISION_WEB_ENTITY_MIN_SCORE", "0.35") or "0.35")
_OBJECT_MIN = float(os.environ.get("VISION_LOCALIZED_OBJECT_MIN_SCORE", "0.45") or "0.45")


def _append_unique(bucket: List[Dict[str, Any]], phrase: str, *, source: str, score: float = 0.0, limit: int = 24) -> None:
    p = re.sub(r"\s+", " ", str(phrase or "").strip())
    if not p or len(p) < 2:
        return
    key = p.lower()
    if any(x.get("name", "").lower() == key for x in bucket):
        return
    if len(bucket) >= limit:
        return
    bucket.append({"name": p[:120], "source": source, "score": round(float(score or 0), 3)})


def _append_str(bucket: List[str], phrase: str, *, limit: int = 32) -> None:
    p = re.sub(r"\s+", " ", str(phrase or "").strip())
    if not p:
        return
    if any(x.lower() == p.lower() for x in bucket):
        return
    if len(bucket) >= limit:
        return
    bucket.append(p[:120])


def _rgb_color_name(r: int, g: int, b: int) -> str:
    """Simple dominant-color label from RGB."""
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r < 40 and g < 40 and b < 40:
        return "black"
    if r > g and r > b:
        return "red" if r - max(g, b) > 40 else "pink"
    if g > r and g > b:
        return "green"
    if b > r and b > g:
        return "blue"
    if r > 180 and g > 120 and b < 80:
        return "orange"
    if r > 180 and g > 180 and b < 100:
        return "yellow"
    if abs(r - g) < 25 and abs(g - b) < 25:
        return "silver" if min(r, g, b) > 120 else "gray"
    return f"rgb({r},{g},{b})"


def _ingest_scored_rows(
    catalog: Dict[str, List[Dict[str, Any]]],
    rows: Iterable[Any],
    *,
    source: str,
    name_keys: Sequence[str] = ("description", "name", "text", "label"),
) -> None:
    for row in rows or []:
        if isinstance(row, dict):
            name = ""
            for k in name_keys:
                if row.get(k):
                    name = str(row[k])
                    break
            score = float(row.get("score") or row.get("confidence") or 0)
        else:
            name = str(row)
            score = 0.0
        name = re.sub(r"\s*\([0-9.]+\)\s*$", "", name).strip()
        if not name:
            continue
        classify_phrase(name, catalog=catalog, source=source, score=score)


def build_recognition_catalog(
    *,
    vision_context: Optional[Dict[str, Any]] = None,
    video_intelligence: Optional[Dict[str, Any]] = None,
    video_intelligence_context: Optional[Dict[str, Any]] = None,
    category: str = "general",
    filename: str = "",
) -> Dict[str, Any]:
    """Structured entity buckets from merged Vision + VI."""
    catalog = empty_catalog()
    vc = vision_context if isinstance(vision_context, dict) else {}
    label_names = list(vc.get("label_names") or [])
    ambient = resolve_ambient_profiles(
        category=category,
        filename=filename,
        vision_label_names=label_names,
    )

    def _skip_ambient(name: str) -> bool:
        return is_redundant_vision_label(name, ambient_profiles=ambient)

    _ingest_scored_rows(catalog, vc.get("logos") or [], source="vision_logo")
    for name in vc.get("logo_names") or []:
        _append_unique(catalog["brands"], str(name), source="vision_logo", score=0.9)

    _ingest_scored_rows(catalog, vc.get("landmarks") or [], source="vision_landmark")
    for name in vc.get("landmark_names") or []:
        _append_unique(catalog["places"], str(name), source="vision_landmark", score=0.85)

    for row in vc.get("web_entities") or []:
        if isinstance(row, dict):
            desc = str(row.get("description") or "").strip()
            sc = float(row.get("score") or 0)
        else:
            desc = str(row).strip()
            sc = 0.0
        if desc and sc >= _WEB_MIN:
            _append_unique(catalog["web_matches"], desc, source="vision_web", score=sc)
            classify_phrase(desc, catalog=catalog, source="vision_web", score=sc)
    for guess in vc.get("web_best_guess") or []:
        g = str(guess).strip()
        if g:
            _append_unique(catalog["web_matches"], g, source="vision_web_guess", score=0.7)
            classify_phrase(g, catalog=catalog, source="vision_web_guess", score=0.7)

    _ingest_scored_rows(
        catalog,
        [o for o in (vc.get("localized_objects") or []) if not _skip_ambient(
            str((o.get("name") if isinstance(o, dict) else o) or "")
        )],
        source="vision_object",
        name_keys=("name", "description"),
    )
    _ingest_scored_rows(
        catalog,
        [lbl for lbl in (vc.get("labels") or []) if not _skip_ambient(
            str((lbl.get("description") if isinstance(lbl, dict) else lbl) or "")
        )],
        source="vision_label",
    )
    for name in vc.get("label_names") or []:
        if _skip_ambient(str(name)):
            continue
        classify_phrase(str(name), catalog=catalog, source="vision_label", score=0.6)

    for prop in vc.get("dominant_colors") or []:
        if isinstance(prop, dict) and prop.get("name"):
            _append_unique(
                catalog["colors"],
                str(prop["name"]),
                source="vision_color",
                score=float(prop.get("score") or 0),
            )

    ocr = str(vc.get("ocr_text") or "").strip()
    if ocr:
        for block in re.split(r"[\n|]+", ocr):
            line = block.strip()
            if len(line) >= 2:
                _append_unique(catalog["text_on_screen"], line[:100], source="vision_ocr", score=0.8)
                classify_phrase(line, catalog=catalog, source="vision_ocr", score=0.8)
        for m in YEAR_MAKE_MODEL_RE.finditer(ocr):
            yr, make, trim = m.group(1), m.group(2), m.group(3) or ""
            veh = f"{yr} {make} {trim}".strip()
            _append_unique(catalog["vehicles"], veh, source="vision_ocr_vehicle", score=0.95)

    for src_name, vi in (
        ("video_intelligence", video_intelligence),
        ("video_intelligence_context", video_intelligence_context),
    ):
        if not isinstance(vi, dict) or vi.get("error"):
            continue
        _ingest_scored_rows(catalog, vi.get("logos") or [], source=f"{src_name}_logo")
        _ingest_scored_rows(catalog, vi.get("object_tracks") or [], source=f"{src_name}_object")
        for key in ("segment_labels", "shot_labels", "top_labels"):
            raw = vi.get(key) or []
            parsed: List[Dict[str, Any]] = []
            for item in raw:
                if isinstance(item, dict):
                    parsed.append(item)
                else:
                    parsed.append({"description": str(item)})
            _ingest_scored_rows(catalog, parsed, source=f"{src_name}_{key}")
        for row in vi.get("on_screen_text") or vi.get("text_detections") or []:
            if isinstance(row, dict):
                t = str(row.get("text") or "").strip()
                sc = float(row.get("confidence") or 0)
            else:
                t = str(row).strip()
                sc = 0.0
            if t:
                _append_unique(catalog["text_on_screen"], t[:100], source=f"{src_name}_text", score=sc)
                classify_phrase(t, catalog=catalog, source=f"{src_name}_text", score=sc)
        if vi.get("person_segments"):
            catalog["people"].append(
                {"name": f"{len(vi['person_segments'])} person segment(s)", "source": src_name, "score": 0.7}
            )

    # Flat deduped lists for easy prompt injection
    flat: Dict[str, List[str]] = {}
    for key, rows in catalog.items():
        flat[key] = [r["name"] for r in rows][:24]

    return {
        "catalog": catalog,
        "flat": flat,
        "entity_count": len(catalog["all_entities"]),
        "ambient_profiles": sorted(ambient),
    }


def _resolve_content_niche(ctx: Any) -> str:
    """Best available niche/category for bucket prioritization."""
    cat = getattr(ctx, "thumbnail_category", None)
    if cat:
        return normalize_niche(str(cat))
    hp = getattr(ctx, "hydration_payload", None) or {}
    if isinstance(hp, dict) and hp.get("category"):
        return normalize_niche(str(hp["category"]))
    us = getattr(ctx, "user_settings", None) or {}
    if isinstance(us, dict):
        from services.thumbnail_studio_strategy import read_thumbnail_studio_default_strategy

        strat = read_thumbnail_studio_default_strategy(us)
        if strat.get("audience_niche"):
            return normalize_niche(str(strat["audience_niche"]))
    return "general"


def build_recognition_narrative(
    catalog_bundle: Dict[str, Any],
    *,
    niche: str = "general",
    max_chars: int = 2200,
) -> str:
    """Human-readable dense description of everything recognized."""
    if not isinstance(catalog_bundle, dict):
        return ""
    flat = catalog_bundle.get("flat") or {}
    if not isinstance(flat, dict):
        return ""

    ambient = tuple(catalog_bundle.get("ambient_profiles") or ())

    labels = narrative_bucket_labels()
    order = niche_bucket_order(niche)
    seen_buckets: set = set()
    parts: List[str] = []
    for bucket in order + list(labels.keys()):
        if bucket in seen_buckets:
            continue
        seen_buckets.add(bucket)
        title = labels.get(bucket)
        if not title:
            continue
        items = flat.get(bucket) or []
        if bucket == "objects":
            items = [
                o for o in items[:14]
                if not is_redundant_vision_label(o, ambient_profiles=ambient or None, min_specific_len=4)
            ]
        if items:
            cap = 12 if bucket in ("food", "plants", "objects") else 10
            parts.append(f"{title}: " + ", ".join(items[:cap]) + ".")
    if flat.get("text_on_screen") and "text_on_screen" not in {b for b in order}:
        parts.append("On-screen text: " + "; ".join(flat["text_on_screen"][:6]) + ".")
    web = catalog_bundle.get("catalog", {}).get("web_matches") or []
    web_names = [w["name"] for w in web if isinstance(w, dict)][:6]
    if web_names:
        parts.append("Web-matched: " + ", ".join(web_names) + ".")

    if not parts:
        all_ent = flat.get("all_entities") or []
        if all_ent:
            parts.append("Detected: " + ", ".join(all_ent[:20]) + ".")
        else:
            return ""

    text = "Google visual recognition — " + " ".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def attach_visual_recognition(ctx: Any) -> Dict[str, Any]:
    """
    Build catalog + narrative on ``ctx`` from vision_context and VI payloads.
    Safe to call after Vision only or after Vision + VI (merges both).
    """
    vc = getattr(ctx, "vision_context", None) or {}
    vi = getattr(ctx, "video_intelligence", None) or {}
    vic = getattr(ctx, "video_intelligence_context", None) or {}

    niche = _resolve_content_niche(ctx)
    bundle = build_recognition_catalog(
        vision_context=vc if isinstance(vc, dict) else None,
        video_intelligence=vi if isinstance(vi, dict) else None,
        video_intelligence_context=vic if isinstance(vic, dict) else None,
        category=niche,
        filename=str(getattr(ctx, "filename", "") or ""),
    )
    bundle["niche"] = niche
    narrative = build_recognition_narrative(bundle, niche=niche)

    if isinstance(vc, dict):
        vc = dict(vc)
        vc["recognition_catalog"] = bundle.get("catalog")
        vc["recognition_flat"] = bundle.get("flat")
        vc["recognition_summary"] = narrative
        vc["recognition_entity_count"] = bundle.get("entity_count", 0)
        ctx.vision_context = vc

    setattr(ctx, "visual_recognition", bundle)
    if narrative and isinstance(getattr(ctx, "output_artifacts", None), dict):
        ctx.output_artifacts["visual_recognition_summary"] = narrative[:4000]
        try:
            import json

            ctx.output_artifacts["visual_recognition_flat"] = json.dumps(
                bundle.get("flat") or {}, default=str
            )[:12000]
        except Exception:
            pass

    flat = bundle.get("flat") or {}
    logger.info(
        "[%s] visual recognition niche=%s entities=%s — vehicles=%s food=%s plants=%s outdoors=%s",
        getattr(ctx, "upload_id", "?"),
        niche,
        bundle.get("entity_count", 0),
        len(flat.get("vehicles") or []),
        len(flat.get("food") or []),
        len(flat.get("plants") or []),
        len(flat.get("outdoors") or []),
    )
    return bundle
