"""
Synthetic scene understanding when Twelve Labs is missing.

Fuses Google Video Intelligence, Cloud Vision OCR (welcome/highway signs),
dashcam OSD GPS/speed, ACR music, and Whisper into ``ctx.video_understanding``
so every AI upload still has a publishable scene narrative 24/7.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from stages.context import JobContext, build_hydration_story_text, build_video_story_timeline

logger = logging.getLogger("uploadm8-worker")

_WELCOME_PATTERNS = (
    re.compile(
        r"\b(?:welcome\s+to|entering|now\s+entering)\s+([A-Z][A-Za-z0-9' .\-]{2,40})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:city\s+of|town\s+of)\s+([A-Z][A-Za-z0-9' .\-]{2,40})",
        re.IGNORECASE,
    ),
)

_PLACE_NOISE = frozenset(
    {
        "california",
        "oregon",
        "washington",
        "nevada",
        "arizona",
        "texas",
        "usa",
        "united states",
        "america",
        "freeway",
        "highway",
        "interstate",
        "the",
        "city",
        "town",
    }
)


def extract_place_signs(*blobs: str) -> List[str]:
    """Pull ``Welcome to X`` / ``Entering X`` place names from OCR / VI text."""
    out: List[str] = []
    seen: set = set()
    for blob in blobs:
        text = str(blob or "").strip()
        if not text:
            continue
        for pat in _WELCOME_PATTERNS:
            for m in pat.finditer(text):
                place = _clean_place_sign(m.group(1))
                if not place:
                    continue
                key = place.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(place)
                if len(out) >= 4:
                    return out
    return out


def _clean_place_sign(raw: str) -> str:
    place = re.sub(r"\s+", " ", str(raw or "").strip(" .,;:|-"))
    # Truncate at common OCR junk after the place name.
    place = re.split(
        r"\b(?:exit|next|miles?|km|population|elev|city\s+limit|town\s+limit)\b",
        place,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" .,;:|-")
    if len(place) < 3 or len(place) > 40:
        return ""
    if place.lower() in _PLACE_NOISE:
        return ""
    # Title-case when OCR is ALL CAPS or mostly uppercase.
    letters = [c for c in place if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) >= 0.8:
        place = place.title()
    return place


def _ocr_blobs_from_ctx(ctx: JobContext) -> List[str]:
    blobs: List[str] = []
    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict):
        ocr = str(vc.get("ocr_text") or "").strip()
        if ocr:
            blobs.append(ocr)
    for vi_attr in ("video_intelligence", "video_intelligence_context"):
        vi = getattr(ctx, vi_attr, None) or {}
        if not isinstance(vi, dict):
            continue
        for row in vi.get("on_screen_text") or []:
            if isinstance(row, dict):
                t = str(row.get("text") or "").strip()
            else:
                t = str(row or "").strip()
            if t:
                blobs.append(t)
        summary = str(vi.get("summary_text") or "").strip()
        if summary:
            blobs.append(summary)
    return blobs


def collect_place_signs(ctx: JobContext) -> List[str]:
    """Welcome/Entering signs from all OCR sources — computed once per upload.

    Called from scene graph, timeline, hydration enforcer, and fusion; the
    OCR scan is pure so cache the result on ctx after the first pass.
    """
    cached = getattr(ctx, "_place_signs_cache", None)
    if isinstance(cached, list):
        return list(cached)
    signs = extract_place_signs(*_ocr_blobs_from_ctx(ctx))
    try:
        setattr(ctx, "_place_signs_cache", list(signs))
    except Exception:
        pass
    return signs


def has_scene_understanding(ctx: JobContext) -> bool:
    vu = getattr(ctx, "video_understanding", None) or {}
    if not isinstance(vu, dict):
        return False
    return bool(str(vu.get("scene_description") or vu.get("description") or "").strip())


def _speed_mph(ctx: JobContext) -> float:
    """Canonical consensus peak (single source of truth for fusion copy).

    Builds fresh (no artifact caching): fusion can run before all speed
    sources are extracted on some recovery paths, and caching a premature
    peak here would poison every later consumer.
    """
    try:
        from core.speed_consensus import build_speed_consensus

        return float(build_speed_consensus(ctx).get("peak_mph") or 0.0)
    except Exception:
        pass
    tel = getattr(ctx, "telemetry", None) or getattr(ctx, "telemetry_data", None)
    osd = getattr(ctx, "dashcam_osd_context", None) or {}
    tel_max = osd_max = 0.0
    if tel is not None:
        try:
            tel_max = float(getattr(tel, "max_speed_mph", 0) or 0)
        except (TypeError, ValueError):
            tel_max = 0.0
    if isinstance(osd, dict):
        try:
            osd_max = float(osd.get("max_speed_mph") or 0)
        except (TypeError, ValueError):
            osd_max = 0.0
    return max(tel_max, osd_max)


def _place_bits(ctx: JobContext, place_signs: List[str]) -> Tuple[str, str]:
    """Return (display_place, start_display)."""
    tel = getattr(ctx, "telemetry", None) or getattr(ctx, "telemetry_data", None)
    start = ""
    if tel is not None:
        start = str(getattr(tel, "location_start_display", None) or "").strip()
        for attr in (
            "location_display",
            "gazetteer_place_name",
            "location_city",
            "location_road",
            "padus_unit_name",
        ):
            v = str(getattr(tel, attr, None) or "").strip()
            if v:
                return v, start
    if place_signs:
        return place_signs[0], start
    return "", start


def _music_bits(ctx: JobContext) -> Tuple[str, str]:
    ac = getattr(ctx, "audio_context", None) or {}
    if not isinstance(ac, dict) or not ac.get("music_detected"):
        return "", ""
    return (
        str(ac.get("music_artist") or "").strip(),
        str(ac.get("music_title") or "").strip(),
    )


def _timeline_hook_lines(ctx: JobContext, *, limit: int = 4) -> List[str]:
    events = build_video_story_timeline(ctx, max_events=40) or []
    prefer = (
        "telemetry_speed",
        "osd_speed",
        "welcome_sign",
        "geo_place",
        "geo_road",
        "music",
        "landmark",
        "on_screen_text",
        "transcript",
        "object",
        "vi_label",
        "yamnet",
    )
    picked: List[str] = []
    seen: set = set()
    by_kind: Dict[str, List[str]] = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        kind = str(ev.get("kind") or "").lower()
        text = str(ev.get("text") or "").strip()
        if not text or len(text) < 6:
            continue
        by_kind.setdefault(kind, []).append(text)
    for kind in prefer:
        for text in by_kind.get(kind) or []:
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append(text)
            if len(picked) >= limit:
                return picked
    return picked


def build_fusion_scene(ctx: JobContext) -> Dict[str, Any]:
    """Deterministic scene_description + title_suggestion from fused providers."""
    place_signs = collect_place_signs(ctx)
    place, start_display = _place_bits(ctx, place_signs)
    speed = _speed_mph(ctx)
    artist, track = _music_bits(ctx)

    story = (build_hydration_story_text(ctx, max_chars=900) or "").strip()
    beats = _timeline_hook_lines(ctx, limit=4)

    from core.vision_entities import build_scene_hook_line, collect_visual_entities

    bundle = collect_visual_entities(
        vision_context=getattr(ctx, "vision_context", None) or {},
        video_intelligence=getattr(ctx, "video_intelligence", None) or {},
        video_intelligence_context=getattr(ctx, "video_intelligence_context", None) or {},
        category=str(getattr(ctx, "thumbnail_category", None) or "general"),
        filename=str(getattr(ctx, "filename", None) or ""),
    )
    hook = build_scene_hook_line(
        place=place or (place_signs[0] if place_signs else ""),
        max_speed_mph=speed,
        music_artist=artist,
        music_title=track,
        bundle=bundle,
        max_chars=140,
    )

    sentences: List[str] = []
    if hook:
        sentences.append(hook.rstrip(".") + ".")
    if story:
        # Prefer a compact factual paragraph; avoid duplicating the hook.
        story_use = story
        if hook and hook.lower().rstrip(".") in story.lower():
            story_use = story
        elif hook and len(story) > 40:
            sentences.append(story[:520].rstrip(".") + ".")
            story_use = ""
        if story_use and story_use not in " ".join(sentences):
            sentences.append(story_use[:520].rstrip(".") + ".")
    elif beats:
        sentences.append("Timeline: " + "; ".join(beats[:3]).rstrip(".") + ".")

    if place_signs and not any(p.lower() in " ".join(sentences).lower() for p in place_signs):
        sentences.append(f"Roadside signage: Welcome to {place_signs[0]}.")
    if start_display and start_display.lower() not in " ".join(sentences).lower():
        if place and start_display.lower() != place.lower():
            sentences.append(f"Run starts near {start_display}.")

    scene = " ".join(s for s in sentences if s).strip()
    if not scene:
        # Absolute floor — still better than empty for hydration.
        bits = []
        if speed >= 5:
            bits.append(f"{int(round(speed))} MPH")
        if place:
            bits.append(f"near {place}")
        if artist or track:
            bits.append(f"with {artist or track}")
        scene = ("Dashcam clip " + ", ".join(bits)).strip() + "." if bits else ""

    # Title suggestion always uses trusted peak (never a stale TL/hook MPH).
    title_parts: List[str] = []
    if speed >= 5:
        title_parts.append(f"{int(round(speed))} MPH")
    title_place = place or (place_signs[0] if place_signs else "") or start_display
    if title_place:
        title_parts.append(title_place)
    if artist:
        title_parts.append(artist)
    elif track:
        title_parts.append(track)
    if len(title_parts) >= 2:
        title_suggestion = " · ".join(title_parts[:3])
    elif title_parts:
        title_suggestion = title_parts[0]
    else:
        title_suggestion = (hook[:90] if hook else "")

    return {
        "scene_description": scene[:1200],
        "title_suggestion": title_suggestion[:100],
        "source": "fusion",
        "place_signs": place_signs,
        "start_display": start_display or None,
        "providers": {
            "hydration_story": bool(story),
            "timeline_beats": len(beats),
            "speed_mph": round(speed, 1) if speed else 0,
            "place": place or None,
            "music": bool(artist or track),
            "welcome_signs": len(place_signs),
        },
    }


def apply_scene_fusion(ctx: JobContext, *, force: bool = False) -> Dict[str, Any]:
    """Fill ``ctx.video_understanding`` when Twelve Labs left it empty.

    Returns the fusion report (empty dict when skipped because TL already won).
    Always speed-scrubs VU against consensus so TL prose cannot leak raw MPH.
    """
    from core.speed_consensus import ensure_video_understanding_speed_scrubbed

    vu = getattr(ctx, "video_understanding", None)
    if not isinstance(vu, dict):
        vu = {}
        ctx.video_understanding = vu

    existing = str(vu.get("scene_description") or vu.get("description") or "").strip()
    if existing and not force:
        # Mark TL-sourced payloads when source missing.
        if not vu.get("source"):
            vu["source"] = "twelve_labs"
        ensure_video_understanding_speed_scrubbed(ctx)
        return {"skipped": True, "reason": "scene_already_present", "source": vu.get("source")}

    fused = build_fusion_scene(ctx)
    if not fused.get("scene_description"):
        logger.info(
            "[scene_fusion] no evidence for upload %s — left video_understanding empty",
            getattr(ctx, "upload_id", "?"),
        )
        ensure_video_understanding_speed_scrubbed(ctx)
        return {"skipped": True, "reason": "no_evidence"}

    vu["scene_description"] = fused["scene_description"]
    if fused.get("title_suggestion") and not str(vu.get("title_suggestion") or "").strip():
        vu["title_suggestion"] = fused["title_suggestion"]
    vu["source"] = "fusion"
    vu["fusion"] = {
        "place_signs": fused.get("place_signs") or [],
        "start_display": fused.get("start_display"),
        "providers": fused.get("providers") or {},
    }
    ctx.video_understanding = vu
    ensure_video_understanding_speed_scrubbed(ctx)

    arts = getattr(ctx, "output_artifacts", None)
    if isinstance(arts, dict):
        arts["scene_fusion"] = {
            "source": "fusion",
            "scene_chars": len(str(vu.get("scene_description") or "")),
            "title_suggestion": vu.get("title_suggestion") or "",
            "providers": fused.get("providers") or {},
            "place_signs": fused.get("place_signs") or [],
        }

    logger.info(
        "[scene_fusion] filled scene for upload %s (%d chars, signs=%s)",
        getattr(ctx, "upload_id", "?"),
        len(str(vu.get("scene_description") or "")),
        fused.get("place_signs") or [],
    )
    return fused
