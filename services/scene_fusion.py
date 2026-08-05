"""
Synthetic scene understanding when Twelve Labs is missing.

Fuses Google Video Intelligence, Cloud Vision OCR (welcome/highway signs),
dashcam OSD GPS/speed, ACR music, and Whisper into ``ctx.video_understanding``
so every AI upload still has a publishable scene narrative 24/7.

When the deterministic fusion prose is thin (<~80 chars) and OpenAI is
available, ``enrich_thin_fusion_scene`` optionally expands it with one
fail-soft LLM call over the multimodal digest (same scrub / publishable-speed
rules as Twelve Labs).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from stages.context import JobContext, build_hydration_story_text, build_video_story_timeline

logger = logging.getLogger("uploadm8-worker")

FUSION_THIN_CHARS = int(os.environ.get("SCENE_FUSION_THIN_CHARS", "80") or 80)
FUSION_ENRICH_MODEL = os.environ.get("OPENAI_SCENE_FUSION_MODEL") or os.environ.get(
    "OPENAI_IDENTITY_MODEL", "gpt-4o-mini"
)
FUSION_ENRICH_TIMEOUT_SEC = float(os.environ.get("SCENE_FUSION_ENRICH_TIMEOUT_SEC", "12") or 12)
FUSION_ENRICH_MAX_TOKENS = 280

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
    """Publishable consensus peak for fusion titles/prose (high confidence only).

    Builds fresh (no artifact caching): fusion can run before all speed
    sources are extracted on some recovery paths. Medium/low HUD-only peaks
    are omitted so fusion cannot mint uncorroborated MPH into VU.
    """
    try:
        from core.speed_consensus import build_speed_consensus

        cons = build_speed_consensus(ctx)
        if str(cons.get("confidence") or "") == "high":
            return float(cons.get("peak_mph") or 0.0)
        return 0.0
    except Exception:
        pass
    # Fail closed without a consensus helper — never invent from raw OSD alone.
    tel = getattr(ctx, "telemetry", None) or getattr(ctx, "telemetry_data", None)
    if tel is not None:
        try:
            tel_max = float(getattr(tel, "max_speed_mph", 0) or 0)
            if tel_max >= 5:
                return tel_max
        except (TypeError, ValueError):
            pass
    return 0.0


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

    custom_queries = _build_native_custom_queries(
        ctx,
        place=place,
        place_signs=place_signs,
        artist=artist,
        track=track,
        speed=speed,
        beats=beats,
    )

    return {
        "scene_description": scene[:1200],
        "description": scene[:1200],  # TL-parity alias
        "title_suggestion": title_suggestion[:100],
        "source": "fusion",
        "custom_queries": custom_queries,
        "place_signs": place_signs,
        "start_display": start_display or None,
        "providers": {
            "hydration_story": bool(story),
            "timeline_beats": len(beats),
            "speed_mph": round(speed, 1) if speed else 0,
            "place": place or None,
            "music": bool(artist or track),
            "welcome_signs": len(place_signs),
            "custom_query_keys": list(custom_queries.keys()),
        },
    }


def _build_native_custom_queries(
    ctx: JobContext,
    *,
    place: str,
    place_signs: List[str],
    artist: str,
    track: str,
    speed: float,
    beats: List[str],
) -> Dict[str, str]:
    """TL-shaped custom_queries from native signals (no Twelve Labs index)."""
    out: Dict[str, str] = {}
    vc = getattr(ctx, "vision_context", None) or {}
    logos: List[str] = []
    if isinstance(vc, dict):
        for lg in (vc.get("logo_names") or vc.get("logos") or [])[:6]:
            if isinstance(lg, dict):
                t = str(lg.get("description") or lg.get("name") or "").strip()
            else:
                t = str(lg or "").strip()
            if t:
                logos.append(t)
    for vi_attr in ("video_intelligence", "video_intelligence_context"):
        vi = getattr(ctx, vi_attr, None) or {}
        if not isinstance(vi, dict):
            continue
        for lg in (vi.get("logos") or [])[:4]:
            if isinstance(lg, dict):
                t = str(lg.get("description") or "").strip()
            else:
                t = str(lg or "").strip()
            if t and t not in logos:
                logos.append(t)
    if logos:
        out["brands_visible"] = ", ".join(logos[:4])
    loc_bits = [b for b in (place, *(place_signs or [])) if b]
    if loc_bits:
        out["location_clue"] = ", ".join(list(dict.fromkeys(loc_bits))[:3])
    if artist or track:
        out["music_id"] = " — ".join(p for p in (artist, track) if p)
    # Publishable speed only (high confidence) in native queries.
    try:
        from core.speed_consensus import publishable_peak_mph

        pub = float(publishable_peak_mph(ctx) or 0)
    except Exception:
        pub = 0.0
    if pub >= 5:
        out["peak_speed"] = f"{int(round(pub))} MPH"
    tx = (getattr(ctx, "ai_transcript", None) or "").strip()
    if not tx:
        ac = getattr(ctx, "audio_context", None) or {}
        if isinstance(ac, dict):
            tx = str(ac.get("transcript") or "").strip()
    if tx:
        # First usable phrase, not full dump.
        phrase = re.split(r"[.!?]\s+", tx, maxsplit=1)[0].strip()[:120]
        if len(phrase) >= 8:
            out["speech_hook"] = phrase
    if beats and "timeline_hook" not in out:
        out["timeline_hook"] = "; ".join(beats[:2])[:160]
    return out


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

    scene_txt = str(fused.get("scene_description") or "")
    vu["scene_description"] = scene_txt
    vu["description"] = scene_txt  # TL-parity alias for all consumers
    if fused.get("title_suggestion") and not str(vu.get("title_suggestion") or "").strip():
        vu["title_suggestion"] = fused["title_suggestion"]
    cq = fused.get("custom_queries") if isinstance(fused.get("custom_queries"), dict) else {}
    if cq:
        # Merge — never clobber richer TL custom_queries if somehow present.
        existing_cq = vu.get("custom_queries") if isinstance(vu.get("custom_queries"), dict) else {}
        merged_cq = dict(cq)
        merged_cq.update({k: v for k, v in existing_cq.items() if v})
        vu["custom_queries"] = merged_cq
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
            "custom_queries": list((vu.get("custom_queries") or {}).keys()),
        }

    logger.info(
        "[scene_fusion] filled scene for upload %s (%d chars, signs=%s)",
        getattr(ctx, "upload_id", "?"),
        len(str(vu.get("scene_description") or "")),
        fused.get("place_signs") or [],
    )
    return fused


def fusion_scene_is_thin(ctx: JobContext, *, min_chars: int = 0) -> bool:
    """True when VU prose is missing or shorter than the enrich threshold."""
    threshold = min_chars or FUSION_THIN_CHARS
    vu = getattr(ctx, "video_understanding", None) or {}
    if not isinstance(vu, dict):
        return True
    scene = str(vu.get("scene_description") or vu.get("description") or "").strip()
    return len(scene) < max(20, int(threshold))


def _fusion_speed_contract(ctx: JobContext) -> str:
    try:
        from core.speed_consensus import get_speed_consensus, publishable_peak_mph

        pub = float(publishable_peak_mph(ctx) or 0)
        cons = get_speed_consensus(ctx)
        conf = str(cons.get("confidence") or "none")
    except Exception:
        pub, conf = 0.0, "none"
    if pub >= 5 and conf == "high":
        return (
            f"SPEED CONTRACT: the only publishable speed is {int(round(pub))} MPH "
            "(verified). Never invent other MPH/KMH numbers. Never treat lat/lon "
            "degrees, headings, or bare integers as speed."
        )
    return (
        "SPEED CONTRACT: there is NO verified speed. Never state any MPH/KMH number. "
        "Never treat GPS coordinates, degree headings, or bare integers as speed."
    )


def _parse_enrich_response(raw: str) -> Optional[Dict[str, str]]:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    scene = str(data.get("scene_description") or data.get("description") or "").strip()
    title = str(data.get("title_suggestion") or "").strip()
    if len(scene) < 40:
        return None
    return {
        "scene_description": scene[:1200],
        "title_suggestion": title[:100],
    }


async def enrich_thin_fusion_scene(ctx: JobContext) -> Dict[str, Any]:
    """Fail-soft OpenAI enrich when fusion (or empty TL) left thin VU prose.

    Never blocks the pipeline. Requires OPENAI_API_KEY. Skips when prose is
    already long enough or when the digest has no usable evidence.
    """
    from core.speed_consensus import ensure_video_understanding_speed_scrubbed

    report: Dict[str, Any] = {"attempted": False, "enriched": False}
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        report["reason"] = "no_openai_key"
        return report
    if os.environ.get("SCENE_FUSION_ENRICH", "1").strip().lower() in ("0", "false", "no", "off"):
        report["reason"] = "disabled"
        return report

    vu = getattr(ctx, "video_understanding", None)
    if not isinstance(vu, dict):
        vu = {}
        ctx.video_understanding = vu
    src = str(vu.get("source") or "")
    # Never overwrite a rich Twelve Labs narrative.
    if src == "twelve_labs" and not fusion_scene_is_thin(ctx):
        report["reason"] = "twelve_labs_present"
        return report
    if not fusion_scene_is_thin(ctx):
        report["reason"] = "scene_already_rich"
        return report

    try:
        from stages.context import build_multimodal_scene_digest

        digest = (build_multimodal_scene_digest(ctx, max_chars=3500) or "").strip()
    except Exception as e:
        report["reason"] = f"digest_failed:{e}"
        return report
    if len(digest) < 40:
        report["reason"] = "digest_too_thin"
        return report

    floor = str(vu.get("scene_description") or vu.get("description") or "").strip()
    prompt = f"""You expand a thin machine-fused scene summary into a short factual video description.

EVIDENCE DIGEST (only source of facts — do not invent):
{digest[:3200]}

EXISTING FLOOR (keep every concrete fact it already has):
{floor or "(empty)"}

{_fusion_speed_contract(ctx)}

Return STRICT JSON:
{{
  "scene_description": "2-4 factual sentences, 120-400 chars, grounded only in the digest",
  "title_suggestion": "≤10 words, concrete, no hype filler"
}}

Rules:
- Weave place, music, speech, and verified speed when present.
- No checklist stubs like "110 MPH, Road Name" alone.
- No lat/lon coordinates, degree headings, or bare integers as "speed".
- No emojis. No invented brands, people, or places."""

    report["attempted"] = True
    payload = {
        "model": FUSION_ENRICH_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": FUSION_ENRICH_MAX_TOKENS,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    try:
        from stages.outbound_rl import outbound_slot

        async with outbound_slot("openai"):
            async with httpx.AsyncClient(timeout=FUSION_ENRICH_TIMEOUT_SEC) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        if resp.status_code != 200:
            body = (resp.text or "")[:240]
            report["reason"] = f"http_{resp.status_code}"
            logger.warning(
                "[scene_fusion] enrich HTTP %s upload=%s: %s",
                resp.status_code,
                getattr(ctx, "upload_id", "?"),
                body,
            )
            return report
        content = (
            (resp.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
        )
        parsed = _parse_enrich_response(content)
        if not parsed:
            report["reason"] = "unparseable"
            return report
    except Exception as e:
        report["reason"] = f"error:{type(e).__name__}"
        logger.warning(
            "[scene_fusion] enrich failed (non-fatal) upload=%s: %s",
            getattr(ctx, "upload_id", "?"),
            e,
        )
        return report

    scene = parsed["scene_description"]
    # Keep floor facts if the model somehow dropped them.
    if floor and len(scene) < len(floor):
        report["reason"] = "enrich_shorter_than_floor"
        return report

    vu["scene_description"] = scene
    vu["description"] = scene
    if parsed.get("title_suggestion"):
        vu["title_suggestion"] = parsed["title_suggestion"]
    vu["source"] = "fusion_llm" if src in ("", "fusion") else src
    fusion_meta = vu.get("fusion") if isinstance(vu.get("fusion"), dict) else {}
    fusion_meta = dict(fusion_meta)
    fusion_meta["enriched"] = True
    fusion_meta["enrich_model"] = FUSION_ENRICH_MODEL
    vu["fusion"] = fusion_meta
    ctx.video_understanding = vu
    ensure_video_understanding_speed_scrubbed(ctx)

    arts = getattr(ctx, "output_artifacts", None)
    if isinstance(arts, dict):
        prev = arts.get("scene_fusion") if isinstance(arts.get("scene_fusion"), dict) else {}
        arts["scene_fusion"] = {
            **prev,
            "source": vu.get("source"),
            "enriched": True,
            "scene_chars": len(str(vu.get("scene_description") or "")),
            "title_suggestion": vu.get("title_suggestion") or "",
        }

    report["enriched"] = True
    report["scene_chars"] = len(str(vu.get("scene_description") or ""))
    logger.info(
        "[scene_fusion] enriched thin scene upload=%s chars=%s",
        getattr(ctx, "upload_id", "?"),
        report["scene_chars"],
    )
    return report
