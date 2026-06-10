"""
M8_ENGINE — UploadM8 multimodal caption brain (M8_ENGINE AI lineage)
====================================================================
Part of the **M8_ENGINE** family: this module is the multimodal publishing core.
**M8_ENGINE AI** (slug ``M8_ENGINE_AI``) names the unified machine-learning + AI layer
(quality priors, coach, growth intel); see `stages/m8_engine_brand.py`.

Builds a unified scene graph from audio + vision + telemetry + Twelve Labs,
then generates **five caption/title variants per target platform**, ranks them
with accuracy-first heuristics (and optional hooks for live platform stats),
and writes per-platform winners onto JobContext for publish_stage.

Version: see M8_ENGINE_VERSION. Designed to evolve as more uploads + analytics land.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncpg
import httpx

from core.helpers import strip_stray_hashtag_json_blob
from core.vision_labels import (
    is_generic_vision_label,
    penalize_generic_vision_hashtags,
    vision_labels_for_m8_scene_graph,
)

from .context import (
    JobContext,
    build_fusion_caption_rules,
    build_fusion_summary_text,
    build_hydration_story_text,
    build_multimodal_scene_digest,
    compute_route_spatial_summary,
    extract_landmark_hints,
)
from .m8_engine_brand import M8_ENGINE_AI_DISPLAY, M8_ENGINE_AI_SLUG, M8_ENGINE_SLUG
from .outbound_rl import outbound_slot
from services.hydration_payload import merge_m8_must_use_tokens, m8_hydration_contract_block
from services.pipeline_ai_trace import record_ai_pipeline_trace

logger = logging.getLogger("uploadm8-worker.m8")

M8_ENGINE_VERSION = "1.3.0"

# When uploads reach caption before platforms are persisted, or transcode was skipped with
# an empty ``platforms`` row, M8 must still rank + write per-platform captions. Matches
# ``JobContext`` narrative defaults (see multimodal_digest / hydration helpers).
M8_DEFAULT_PLATFORMS: Tuple[str, ...] = ("youtube", "instagram", "facebook", "tiktok")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def _effective_m8_platforms(ctx: JobContext) -> List[str]:
    """Normalize target platforms for scene graph build + selection writes."""
    out = [str(p).lower().strip() for p in (ctx.platforms or []) if str(p).strip()]
    if out:
        return out
    logger.warning(
        "M8: ctx.platforms is empty — using default %s so ranking/apply_selection still run.",
        list(M8_DEFAULT_PLATFORMS),
    )
    return list(M8_DEFAULT_PLATFORMS)


def _trace_m8(ctx: JobContext, upload_id: str, event: str, payload: Dict[str, Any]) -> None:
    record_ai_pipeline_trace(ctx, upload_id, f"m8.{event}", payload, log=logger)


def m8_evidence_matrix_enabled(user_settings: Optional[Dict[str, Any]]) -> bool:
    """
    Extra JSON block ``caption_evidence_matrix`` in the same multimodal M8 call.

    Enable: M8_CAPTION_STYLE_MATRIX=true | 1 | yes, or user ``multiStyleCaptions`` / ``multi_style_captions`` true.
    Disable: M8_CAPTION_STYLE_MATRIX=false | 0 | no, or when unset and no user flag (saves completion tokens).
    """
    raw = (os.environ.get("M8_CAPTION_STYLE_MATRIX") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    us = user_settings or {}
    if us.get("multiStyleCaptions") is not None:
        return bool(us.get("multiStyleCaptions"))
    if us.get("multi_style_captions") is not None:
        return bool(us.get("multi_style_captions"))
    return False


def _evidence_matrix_cell_specs(style_ui: str, tone_ui: str, voice_ui: str) -> List[Tuple[str, str, str]]:
    """
    Deduped list of (caption_style, caption_tone, caption_voice) triples:
    - Part A: story/punchy/factual x each voice with user's tone held constant.
    - Part B: authentic/hype/calm/cinematic with user's style+voice (adds tones not covered as constant).
    """
    styles = ("story", "punchy", "factual")
    voices = ("default", "mentor", "hypebeast", "best_friend", "teacher", "cinematic_narrator")
    tones = ("authentic", "hype", "calm", "cinematic")
    style_ui = (style_ui or "story").lower()
    tone_ui = (tone_ui or "authentic").lower()
    voice_ui = (voice_ui or "default").lower()
    seen: set[Tuple[str, str, str]] = set()
    out: List[Tuple[str, str, str]] = []
    for s in styles:
        for v in voices:
            key = (s, tone_ui, v)
            if key not in seen:
                seen.add(key)
                out.append(key)
    for t in tones:
        key = (style_ui, t, voice_ui)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _sanitize_evidence_matrix(raw: Any, expected_max: int) -> Optional[Dict[str, Any]]:
    """Normalize model matrix output; TikTok-oriented short captions."""
    if not isinstance(raw, dict):
        return None
    cells = raw.get("cells")
    if not isinstance(cells, list):
        return None
    clean: List[Dict[str, Any]] = []
    for item in cells[: max(1, int(expected_max) + 6)]:
        if not isinstance(item, dict):
            continue
        s = str(item.get("caption_style") or "").lower().strip()
        t = str(item.get("caption_tone") or "").lower().strip()
        v = str(item.get("caption_voice") or "").lower().strip()
        cap_raw = item.get("tiktok_caption")
        if cap_raw is None:
            cap_raw = item.get("caption")
        cap = strip_stray_hashtag_json_blob(str(cap_raw or "").strip())[:520]
        tags = item.get("hashtags") or []
        tl = [str(x).strip().lstrip("#") for x in (tags if isinstance(tags, list) else []) if str(x).strip()][:12]
        if not cap:
            continue
        clean.append(
            {
                "caption_style": s,
                "caption_tone": t,
                "caption_voice": v,
                "tiktok_caption": cap,
                "hashtags": tl,
            }
        )
    if not clean:
        return None
    return {"cells": clean, "format": "tiktok_micro", "version": M8_ENGINE_VERSION}

# Generic / “AI slop” phrases to penalize in variants (light-touch; expand over time).
# Expanded after seeing real-world model output like "Cruise under vast skies!
# Endless horizons await." which contains zero scene-graph evidence.
_GENERIC_PATTERNS = [
    r"\bjoin me\b",
    r"\blet's dive\b",
    r"\bunlock(ed)?\b",
    r"\byou won't believe\b",
    r"\bsecret\b",
    r"\bcontent creator\b",
    r"\bas an ai\b",
    r"\bembrace the chaos\b",
    r"\bhidden gem\b",
    r"\bexciting moments?\b",
    r"\bunbelievable moments?\b",
    r"\bwatch the road transform\b",
    r"\bin this (?:raw )?authentic moment\b",
    r"\bchannel(?:ing)? (?:my|your) emotions?\b",
    r"\bvast skies\b",
    r"\bendless horizons?\b",
    r"\bopen road\b",
    r"\b(?:adventure|journey) awaits?\b",
    r"\bbreath(?:e|taking) (?:in )?(?:the )?freedom\b",
    r"\bcruise (?:under|through|along)\b",
    r"\bbuckle up\b",
    r"\bridin'? dirty\b",
    r"\bvibes? only\b",
    r"\bgood vibes\b",
    r"\bscenic (?:vibes?|drive|views?)\b",
    r"\b(?:travel|highway|cloud) (?:vibes?|watching)\b",
    r"\bnature(?:'s)? (?:beauty|symphony)\b",
    r"\b(?:explore|discover) more\b",
]

def build_scene_graph(ctx: JobContext, category: str) -> Dict[str, Any]:
    """Single structured snapshot used by M8 prompts and ranking."""
    target_platforms = _effective_m8_platforms(ctx)
    ac = ctx.audio_context or {}
    vc = ctx.vision_context or {}
    vu = ctx.video_understanding or {}
    tel = ctx.telemetry or ctx.telemetry_data

    geo: Dict[str, Any] = {}
    if tel:
        n_pts = len(getattr(tel, "points", None) or [])
        geo = {
            "display": getattr(tel, "location_display", None),
            "start_display": getattr(tel, "location_start_display", None),
            "road": getattr(tel, "location_road", None),
            "city": getattr(tel, "location_city", None),
            "state": getattr(tel, "location_state", None),
            "country": getattr(tel, "location_country", None),
            "mid_lat": getattr(tel, "mid_lat", None),
            "mid_lon": getattr(tel, "mid_lon", None),
            "start_lat": getattr(tel, "start_lat", None),
            "start_lon": getattr(tel, "start_lon", None),
            "max_speed_mph": getattr(tel, "max_speed_mph", None),
            "avg_speed_mph": getattr(tel, "avg_speed_mph", None),
            "total_distance_miles": getattr(tel, "total_distance_miles", None),
            "duration_seconds": getattr(tel, "duration_seconds", None),
            "max_altitude_ft": getattr(tel, "max_altitude_ft", None),
            "map_point_count": n_pts or None,
            "speeding_seconds": getattr(tel, "speeding_seconds", None),
            "euphoria_seconds": getattr(tel, "euphoria_seconds", None),
        }
        rb, rp = compute_route_spatial_summary(getattr(tel, "points", None) or [], max_polyline_points=36)
        if rb:
            geo["route_bbox"] = rb
        if rp:
            geo["route_polyline_sample"] = rp
        gp = getattr(tel, "gazetteer_place_name", None)
        if gp:
            geo["gazetteer_place"] = str(gp).strip()
        gus = getattr(tel, "gazetteer_state_usps", None)
        if gus:
            geo["gazetteer_state_usps"] = str(gus).strip()
        if getattr(tel, "near_padus", False):
            geo["near_protected_land"] = True
        pun = getattr(tel, "padus_unit_name", None)
        if pun:
            geo["protected_area_name"] = str(pun).strip()

    tr = ctx.trill or ctx.trill_score
    trill_d: Dict[str, Any] = {}
    if tr:
        trill_d = {
            "score": getattr(tr, "score", None),
            "bucket": getattr(tr, "bucket", None),
            "title_modifier": getattr(tr, "title_modifier", None),
            "hashtags": list(getattr(tr, "hashtags", None) or [])[:12],
        }

    osd_ctx = ctx.dashcam_osd_context or {}
    osd_d: Dict[str, Any] = {}
    if isinstance(osd_ctx, dict) and osd_ctx and not osd_ctx.get("skipped"):
        fs = osd_ctx.get("first_seen") or {}
        ls = osd_ctx.get("last_seen") or {}
        osd_d = {
            "engine": osd_ctx.get("engine"),
            "frames_sampled": osd_ctx.get("frames_sampled"),
            "frames_with_signal": osd_ctx.get("frames_with_signal"),
            "coverage_pct": osd_ctx.get("coverage_pct"),
            "max_speed_mph": osd_ctx.get("max_speed_mph"),
            "avg_speed_mph": osd_ctx.get("avg_speed_mph"),
            "speed_unit_detected": osd_ctx.get("speed_unit_detected"),
            "driver_name": osd_ctx.get("driver_name"),
            "telemetry_backfilled": bool(osd_ctx.get("telemetry_backfilled")),
            "gps_fix_count": len(osd_ctx.get("gps_path") or []),
            "first_seen": {
                "date": fs.get("date"),
                "time": fs.get("time"),
                "lat": fs.get("lat"),
                "lon": fs.get("lon"),
                "speed_mph": fs.get("speed_mph"),
            },
            "last_seen": {
                "date": ls.get("date"),
                "time": ls.get("time"),
                "lat": ls.get("lat"),
                "lon": ls.get("lon"),
                "speed_mph": ls.get("speed_mph"),
            },
        }

    labels_full = vision_labels_for_m8_scene_graph(vc.get("label_names") or [], limit=24)
    ocr_full = (vc.get("ocr_text") or "").strip()
    landmark_names = [str(x).strip() for x in (vc.get("landmark_names") or []) if str(x).strip()]
    logo_names = [str(x).strip() for x in (vc.get("logo_names") or []) if str(x).strip()]
    hume_d = (ac.get("hume_emotions") or {}) if isinstance(ac.get("hume_emotions"), dict) else {}

    trend_block: Dict[str, Any] = {}
    ti_ctx = getattr(ctx, "trend_intel_context", None)
    if isinstance(ti_ctx, dict) and (ti_ctx.get("summary") or ti_ctx.get("rows")):
        rows_clean: List[Dict[str, Any]] = []
        for row in (ti_ctx.get("rows") or [])[:10]:
            if not isinstance(row, dict):
                continue
            rows_clean.append(
                {
                    "title": str(row.get("title") or "")[:200],
                    "channel": str(row.get("channel") or "")[:120],
                }
            )
        trend_block = {
            "query": ti_ctx.get("query"),
            "source": ti_ctx.get("source"),
            "summary": str(ti_ctx.get("summary") or "")[:1200],
            "rows": rows_clean,
        }

    hp_snap = getattr(ctx, "hydration_payload", None)
    fusion_summary_sg = build_fusion_summary_text(ctx)[:4000]
    hydration_story_sg = build_hydration_story_text(ctx)[:1200]
    if isinstance(hp_snap, dict):
        fs_h = str(hp_snap.get("fusion_summary") or "").strip()
        hs_h = str(hp_snap.get("hydration_story") or "").strip()
        if fs_h:
            fusion_summary_sg = fs_h[:4000]
        if hs_h:
            hydration_story_sg = hs_h[:1200]

    out_graph: Dict[str, Any] = {
        "engine_version": M8_ENGINE_VERSION,
        "m8_engine": {
            "family_slug": M8_ENGINE_SLUG,
            "ai_slug": M8_ENGINE_AI_SLUG,
            "ai_display": M8_ENGINE_AI_DISPLAY,
            "mlai_slug": M8_ENGINE_AI_SLUG,
            "mlai_display": M8_ENGINE_AI_DISPLAY,
        },
        "upload_id": ctx.upload_id,
        "filename": ctx.filename,
        "category": category,
        "platforms": target_platforms,
        "transcript": {
            "text": (ctx.ai_transcript or ac.get("transcript") or "")[:12000],
            "role": ac.get("transcript_role") or "",
            "language": ac.get("language") or ac.get("transcript_language") or "",
            "duration_seconds": ac.get("transcript_duration") or 0.0,
            "structured": ac.get("transcript_structured") or {},
            "segments": ac.get("transcript_segments") or [],
        },
        "music": {
            "detected": bool(ac.get("music_detected")),
            "title": ac.get("music_title") or "",
            "artist": ac.get("music_artist") or "",
            "genre": ac.get("music_genre") or "",
            "copyright_risk": bool(ac.get("copyright_risk")),
        },
        "audio_environment": {
            "sound_profile": ac.get("sound_profile") or "",
            "top_sound_class": ac.get("top_sound_class") or "",
            "yamnet_events": list(ac.get("yamnet_events") or []),
            "content_signals": list(ac.get("content_signals") or [])[:40],
            "hume_dominant_emotion": hume_d.get("dominant_emotion") or "",
        },
        "fusion_narrative": (ac.get("fusion_narrative") or "")[:2000],
        "fusion_summary": fusion_summary_sg,
        "hydration_story": hydration_story_sg,
        "multimodal_digest": build_multimodal_scene_digest(ctx, max_chars=10000),
        "vision": {
            "labels": labels_full,
            "label_count": len(labels_full),
            "ocr": ocr_full[:8000],
            "face_count": vc.get("face_count"),
            "has_faces": vc.get("has_faces"),
            "expressive_faces": bool(vc.get("expressive")),
            "landmarks": landmark_names[:12],
            "logos": logo_names[:12],
            "landmark_hints": extract_landmark_hints(labels_full, ocr_full, landmark_names),
            "web_entities": [
                (w.get("description") if isinstance(w, dict) else str(w))
                for w in (vc.get("web_entities") or [])[:16]
            ],
            "localized_objects": [
                (o.get("name") if isinstance(o, dict) else str(o))
                for o in (vc.get("localized_objects") or [])[:16]
            ],
            "dominant_colors": [
                (c.get("name") if isinstance(c, dict) else str(c))
                for c in (vc.get("dominant_colors") or [])[:8]
            ],
            "recognition_summary": str(vc.get("recognition_summary") or "")[:2500],
            "recognition_flat": (
                vc.get("recognition_flat")
                if isinstance(vc.get("recognition_flat"), dict)
                else (getattr(ctx, "visual_recognition", {}) or {}).get("flat") or {}
            ),
        },
        "video_understanding": {
            "scene": (vu.get("scene_description") or vu.get("description") or "")[:8000],
            "title_suggestion": (vu.get("title_suggestion") or "")[:200],
        },
        "geo": geo,
        "trill": trill_d,
        "dashcam_osd": osd_d,
    }

    # Video Intelligence structured tracks (object/text/person/logo) — keep
    # the heavy raw payload separate but expose the trimmed view here so
    # build_must_use_shortlist + the M8 prompt + the hydration enforcer can
    # all consume it from one place.
    vi = getattr(ctx, "video_intelligence", None) or {}
    if not vi:
        vic = getattr(ctx, "video_intelligence_context", None) or {}
        if isinstance(vic, dict) and not vic.get("error"):
            vi = {
                "top_labels": vic.get("top_labels") or [],
                "segment_labels": vic.get("segment_labels") or [],
                "shot_labels": vic.get("shot_labels") or [],
                "object_tracks": vic.get("object_tracks") or [],
                "on_screen_text": vic.get("on_screen_text") or [],
                "person_segments": vic.get("person_segments") or [],
                "logos": vic.get("logos") or [],
                "summary_text": vic.get("summary_text") or "",
            }
    if isinstance(vi, dict) and (
        vi.get("top_labels")
        or vi.get("segment_labels")
        or vi.get("shot_labels")
        or vi.get("object_tracks")
        or vi.get("logos")
        or vi.get("on_screen_text")
        or vi.get("person_segments")
        or vi.get("summary_text")
    ):
        out_graph["video_intelligence"] = {
            "top_labels": (vi.get("top_labels") or [])[:12],
            "segment_labels": (vi.get("segment_labels") or [])[:8],
            "shot_labels": (vi.get("shot_labels") or [])[:8],
            "object_tracks": (vi.get("object_tracks") or [])[:8],
            "on_screen_text": (vi.get("on_screen_text") or [])[:8],
            "person_segments": (vi.get("person_segments") or [])[:6],
            "logos": (vi.get("logos") or [])[:6],
            "summary": (vi.get("summary_text") or "")[:600],
        }

    if trend_block:
        out_graph["trend_intel"] = trend_block

    # Inject the ordered evidence timeline (built in worker.py after the OSD
    # stage) so the LLM gets per-second beats instead of one flattened
    # paragraph. Cap to ~32 events to keep the prompt compact.
    try:
        timeline_events: List[Dict[str, Any]] = []
        arts_raw = getattr(ctx, "output_artifacts", None) or {}
        ts_raw = arts_raw.get("timeline_story") if isinstance(arts_raw, dict) else None
        if isinstance(ts_raw, str) and ts_raw.strip():
            parsed_ts = json.loads(ts_raw)
            if isinstance(parsed_ts, list):
                timeline_events = parsed_ts
        if not timeline_events:
            try:
                from stages.context import build_video_story_timeline as _bvst
                timeline_events = _bvst(ctx, max_events=64) or []
            except Exception:
                timeline_events = []
        if timeline_events:
            out_graph["timeline"] = list(timeline_events)[:32]
            scene_story_artifact = arts_raw.get("scene_story") if isinstance(arts_raw, dict) else None
            if isinstance(scene_story_artifact, str) and scene_story_artifact.strip():
                out_graph["scene_story"] = scene_story_artifact[:1600]
    except Exception:
        pass

    return out_graph


def _platform_constraints(platform: str) -> str:
    p = (platform or "").lower()
    if p == "tiktok":
        return (
            "TikTok: caption ONLY (no separate title field). "
            "First line must hook hard in the first 3–5 words. "
            "Prefer 80–220 chars for the main caption body before hashtags. "
            "Hashtags: separate words without # in JSON; we add # at publish."
        )
    if p == "youtube":
        return (
            "YouTube Shorts / long-form: require a strong TITLE (max 100 chars) "
            "and a DESCRIPTION that works as the caption/body (100–500 chars). "
            "Title must not be clickbait lies; must match visible content."
        )
    if p in ("instagram", "facebook"):
        return (
            f"{p.title()}: caption-first. Optional very short headline phrase in title field "
            "or empty title. 100–350 chars main caption. Engagement-friendly but truthful."
        )
    return "Match the platform's native style; stay accurate."


def _persona_system_prompt(persona: str) -> str:
    p = (persona or "").lower().strip()
    if p == "creator_coach":
        return "Voice: creator coach. Teach clearly, avoid hype inflation, use practical verbs."
    if p == "hype_friend":
        return "Voice: hype friend. Fast, punchy lines; still truthful and specific."
    if p == "expert_analyst":
        return "Voice: expert analyst. Precise, evidence-led, strong nouns over slang."
    return "Voice: storyteller. Human, cinematic, and scene-anchored."


def _style_prompt(style: str, tone: str) -> str:
    s = (style or "").lower().strip()
    t = (tone or "").lower().strip()
    return (
        f"Style policy: {s}. Tone policy: {t}. "
        "Use one clear opening hook, concrete evidence, and one soft CTA."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Elite, UI-keyed creative directives for the M8 engine.
#
# WHY: the settings page exposes 3 caption styles, 4 tones, and 6 voices/personas.
# Previously M8 collapsed all of them into one generic sentence (`_style_prompt`)
# and a 3-way persona (`_persona_system_prompt`), so distinct UI selections produced
# near-identical copy. These libraries are keyed on the RAW UI values so every
# selection injects a structurally different blueprint into the prompt — making
# output visibly change with each choice while staying evidence-grounded.
#
# Contract for every directive: topic-agnostic delivery only. Vocabulary, stakes,
# and specifics MUST come from the Scene Graph evidence, never invented.
# ─────────────────────────────────────────────────────────────────────────────

# Caption STYLE = the structural architecture of the line (how it is built).
M8_STYLE_DIRECTIVES: Dict[str, Dict[str, str]] = {
    "story": {
        "label": "STORY — narrative arc",
        "blueprint": (
            "Build a 3-beat micro-arc straight from scene_graph.timeline: "
            "(1) SETUP — open on the earliest grounded beat (place / first on-screen text / opening label); "
            "(2) TURN — pivot on the peak beat (top MPH, climax object, landmark, or loudest audio cue); "
            "(3) PAYOFF — close on a late beat that resolves or reframes. "
            "Caption length 150–280 characters. Connective momentum between beats; no bullet fragments. "
            "Each variant must enter and exit the arc at a different beat so the 5 variants feel like 5 retellings, "
            "not one caption reworded."
        ),
    },
    "punchy": {
        "label": "PUNCHY — hook in first 3 words",
        "blueprint": (
            "Front-load the single most arresting CONCRETE fact in the first 3 words "
            "(a number, a place, a named object, a speed). One or two short lines, telegraphic rhythm, "
            "cut every connective and hedge ('just', 'really', 'kind of'). Under 120 characters. "
            "No narrative ramp — impact then stop. Across the 5 variants, rotate WHICH evidence token leads "
            "(speed → place → object → audio → trill) so no two hooks open on the same word class."
        ),
    },
    "factual": {
        "label": "FACTUAL — lead with the strongest stat",
        "blueprint": (
            "Lead with the single most impressive VERIFIABLE data point in the evidence — peak MPH, a count, "
            "a spec, a precise place/road, a date, an artist/title. Data-forward, zero fluff, no adjectives that "
            "evidence does not support. 100–200 characters. State the metric, then one tight line of grounded context. "
            "Across the 5 variants, lead with a different verified figure/spec each time; never repeat the same "
            "headline metric as variant 1."
        ),
    },
}

# Caption TONE = the emotional register / energy the copy is delivered in.
M8_TONE_DIRECTIVES: Dict[str, Dict[str, str]] = {
    "authentic": {
        "label": "AUTHENTIC — real talk, first-person, no fluff",
        "register": (
            "Human and direct; first-person or close second-person; plain words over marketing speak. "
            "Sound like a real person who was actually there. One honest observation beats a manufactured hook. "
            "Ban influencer filler ('okay guys', 'here's the thing', 'let me tell you'). No exclamation inflation."
        ),
    },
    "hype": {
        "label": "HYPE — high energy, power words, stop-the-scroll",
        "register": (
            "High momentum and conviction: strong verbs, tight clauses, forward pull, occasional emphatic word — "
            "still believable. Scale the intensity to the actual subject (a quiet craft gets urgent clarity, not "
            "party-bro shouting). Every spike of energy must trace to something literally on screen or in the audio. "
            "Never invent stakes the footage does not earn."
        ),
    },
    "cinematic": {
        "label": "CINEMATIC — poetic, atmospheric, film-trailer feel",
        "register": (
            "Scene-led, sensory language: light, shadow, motion, scale, texture — only what the frames support. "
            "Present tense where it heightens immediacy; trailer-like rhythm without melodrama or clichés that could "
            "apply to any clip. Every image must tether to a visible detail or a spoken line. Restraint over purple prose."
        ),
    },
    "calm": {
        "label": "CALM — measured, confident, let the footage speak",
        "register": (
            "Measured, breathable pacing; let concrete details carry the weight. Understatement over exclamation; "
            "cool, trustworthy register. No urgency theatrics. Confidence shown through specificity, not volume."
        ),
    },
}

# Caption VOICE / PERSONA = who is speaking (diction, point of view, sign-off).
M8_VOICE_DIRECTIVES: Dict[str, Dict[str, str]] = {
    "default": {
        "label": "DEFAULT — balanced, platform-friendly creator",
        "persona": (
            "Balanced creator voice: clear hook, specific middle, satisfying close. Confident but not performative. "
            "Match slang and terminology to what the content actually is (chef terms for food, dev terms for code, "
            "driver terms for a drive). Neutral, broadly likeable point of view."
        ),
    },
    "mentor": {
        "label": "MENTOR — wise, educational, authority",
        "persona": (
            "Experienced guide: 'you'-oriented, encouraging, zero condescension. Imply expertise through precise "
            "specifics, never a credentials flex. When the clip teaches or demonstrates anything, land one usable "
            "takeaway. Calm authority — the voice of someone who has done this many times."
        ),
    },
    "hypebeast": {
        "label": "HYPEBEAST — all-caps energy, slang, viral",
        "persona": (
            "Peak short-form energy: clipped sentences, rhythm, street/viral cadence, sparing ALL-CAPS on the one "
            "word that matters. Slang only when it fits the subject and platform — never empty viral filler "
            "('this is insane', 'no way'). All the hype must trace to a real on-screen or audio moment."
        ),
    },
    "best_friend": {
        "label": "BEST FRIEND — casual, real, relatable",
        "persona": (
            "Warm, unfiltered peer texting you about something cool: conversational fragments OK, light self-aware "
            "humor when the content allows, relatable aside. Never mean-spirited or faux-chaos. Reads like a friend, "
            "not a brand. Second-person ('you') and shared-moment framing welcome."
        ),
    },
    "teacher": {
        "label": "TEACHER — clear, informative, structured",
        "persona": (
            "Educator clarity: one central idea, a logical mini-arc, minimal jargon unless the visuals clearly expect "
            "it. If the clip is not instructional, still be precise — teach what happened or what to notice in the "
            "footage, not an unrelated life lesson. Structure and signposting over flourish."
        ),
    },
    "cinematic_narrator": {
        "label": "CINEMATIC — film narrator, epic, atmospheric",
        "persona": (
            "Third-person / omniscient trailer narrator: declarative, image-stacking, slightly elevated register. "
            "Anchored to real events in the clip — no epic narration of nothing happening. Reserve the biggest "
            "flourish for the genuine peak in the footage. Think voiceover, not influencer."
        ),
    },
}

# Strategy-slug → UI-voice fallback, so policy/strategy persona overrides still
# resolve to one of the rich UI voice directives instead of going generic.
_M8_PERSONA_SLUG_TO_VOICE_UI: Dict[str, str] = {
    "storyteller": "cinematic_narrator",
    "creator_coach": "mentor",
    "hype_friend": "hypebeast",
    "expert_analyst": "teacher",
}


def _m8_style_directive(style_ui: str) -> Dict[str, str]:
    return M8_STYLE_DIRECTIVES.get((style_ui or "").lower().strip(), M8_STYLE_DIRECTIVES["story"])


def _m8_tone_directive(tone_ui: str) -> Dict[str, str]:
    return M8_TONE_DIRECTIVES.get((tone_ui or "").lower().strip(), M8_TONE_DIRECTIVES["authentic"])


def _m8_voice_directive(voice_ui: str) -> Dict[str, str]:
    v = (voice_ui or "").lower().strip().replace("-", "_")
    if v in M8_VOICE_DIRECTIVES:
        return M8_VOICE_DIRECTIVES[v]
    # voice_ui may actually be a collapsed strategy persona slug — remap it.
    mapped = _M8_PERSONA_SLUG_TO_VOICE_UI.get(v)
    if mapped and mapped in M8_VOICE_DIRECTIVES:
        return M8_VOICE_DIRECTIVES[mapped]
    return M8_VOICE_DIRECTIVES["default"]


def _m8_creative_directive_block(style_ui: str, tone_ui: str, voice_ui: str) -> str:
    """Compose the three UI-keyed directives into one elite, self-differentiating
    creative brief. This is the primary driver of how copy reads, so that every
    distinct combination of (style, tone, voice) produces visibly different output.
    """
    style = _m8_style_directive(style_ui)
    tone = _m8_tone_directive(tone_ui)
    voice = _m8_voice_directive(voice_ui)
    return f"""━━ CREATOR VOICE OPERATING SYSTEM (user-selected — this is the PRIMARY driver of how copy reads) ━━
These three knobs are SEPARATE and compose together. They are delivery only — every fact, name, number, and
piece of vocabulary still comes from the Scene Graph evidence, never invented.

CAPTION STYLE → {style['label']}
  {style['blueprint']}

CAPTION TONE → {tone['label']}
  {tone['register']}

CAPTION VOICE / PERSONA → {voice['label']}
  {voice['persona']}

COMPOSITION RULES (non-negotiable):
- STYLE controls the structure/architecture, TONE controls the emotional register, VOICE controls who is speaking
  (diction + point of view). Apply all three at once; do not let one silently override another.
- If this exact (style, tone, voice) were changed by even one knob, the resulting copy MUST read noticeably
  different — different opening rhythm, sentence length, and word choice — not the same caption reskinned.
- Do NOT fall back to a neutral house voice. The selected voice's diction and point of view must be audible
  in every variant.
- Stay evidence-grounded: the persona is HOW it is said; the Scene Graph is WHAT is said.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def _task_prompt(generate_title: bool, generate_caption: bool, generate_hashtags: bool) -> str:
    return (
        f"Task switches: title={'on' if generate_title else 'off'}, "
        f"caption={'on' if generate_caption else 'off'}, hashtags={'on' if generate_hashtags else 'off'}."
    )


def _platform_prompt(platform: str, target: Dict[str, Any]) -> str:
    c = target.get("constraints") or {}
    banned = ", ".join(str(x) for x in (c.get("banned_phrases") or [])[:10])
    return (
        f"{_platform_constraints(platform)} "
        f"Persona={target.get('persona', 'storyteller')}; risk={target.get('risk_level', 'safe')}; "
        f"caption_len={c.get('caption_length_min', 80)}-{c.get('caption_length_max', 300)}; "
        "emoji_policy=none (no Unicode emojis or emoticons in title or caption); "
        f"hook_formula={c.get('hook_formula', 'scene_hook')}; "
        f"avoid_phrases=[{banned}]."
    )


def _user_brand_directive(ctx: JobContext) -> str:
    """Surface the creator's saved persona/brand display name to M8 so the
    generated copy carries their channel identity instead of generic AI clichés.

    Sourced from ``user_settings.thumbnail_persona_display_name`` which is
    populated by ``stages.db.merge_pikzels_thumbnail_persona_id`` from the
    ``creator_personas.name`` column. Returns an empty string when no persona
    is saved so we don't pollute the prompt with placeholder noise.
    """
    us = getattr(ctx, "user_settings", None) or {}
    if not isinstance(us, dict):
        return ""
    name = (
        us.get("thumbnail_persona_display_name")
        or us.get("thumbnailPersonaDisplayName")
        or us.get("creator_brand_name")
        or us.get("channel_display_name")
        or ""
    )
    name = str(name or "").strip()[:40]
    if not name:
        return ""
    return (
        f"USER BRAND/CREATOR PERSONA: '{name}'. "
        "Treat this as the creator's channel/brand identity. You MAY weave it "
        "naturally into ONE title or caption variant when it fits (e.g. as a "
        "sign-off, byline, or series prefix), but never spam it across every "
        "variant, never invent backstory about it, and never claim it is a "
        "real-world brand. The hashtags array MAY include the brand slug once. "
        "If the persona name does not fit naturally with the scene evidence, "
        "leave it out — evidence-grounding always wins."
    )


def _build_m8_prompt(
    ctx: JobContext,
    scene_graph: Dict[str, Any],
    category: str,
    caption_style: str,
    caption_tone: str,
    hashtag_style: str,
    hashtag_count: int,
    generate_title: bool,
    generate_caption: bool,
    generate_hashtags: bool,
    historical: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None,
    *,
    include_evidence_matrix: bool = False,
    caption_voice_ui: str = "default",
    extra_strategy_block: str = "",
) -> str:
    fusion = ""
    try:
        fusion = build_fusion_caption_rules(ctx) or ""
    except (AttributeError, TypeError, KeyError, ValueError) as e:
        logger.debug("m8_engine: build_fusion_caption_rules skipped: %s", e)

    platforms = [p for p in scene_graph.get("platforms") or [] if p]
    if not platforms:
        platforms = list(M8_DEFAULT_PLATFORMS)

    # Strategy outputs are built in content_strategy.build_content_strategy:
    # user account caption settings seed the master; JSON policy rules may override
    # per platform when match keys hit. Fallback keeps raw UI style/tone if outputs missing.
    strategy_outputs = (strategy or {}).get("outputs") or {}
    platform_targets = strategy_outputs.get("platform_targets") or {}
    base_persona = str(strategy_outputs.get("voice_persona") or "storyteller")
    base_style = str(strategy_outputs.get("caption_style") or caption_style)
    base_tone = str(strategy_outputs.get("tone") or caption_tone)

    plat_blocks = []
    for pl in platforms:
        targ = platform_targets.get(pl) or {}
        plat_blocks.append(f"- {pl.upper()}: {_platform_prompt(pl, targ)}")

    hashtag_rule = ""
    if generate_hashtags and hashtag_count > 0:
        hashtag_rule = (
            f"Include exactly {hashtag_count} hashtags per variant as JSON array of words "
            f"WITHOUT '#'. Style: {hashtag_style}. "
            "Each tag must be a concrete niche/topic/search term (artist fragment, hobby, vehicle, place type). "
            "When scene_graph.geo.gazetteer_place, geo.protected_area_name, or geo.near_protected_land are set, "
            "include at least one discovery tag tied to that real place or protected-land context (no false claims). "
            "When scene_graph.transcript.text, music.*, or audio_environment (e.g. yamnet_events) are populated, "
            "derive several tags from those signals (spoken topics, song title/artist fragments, ambient sound cues) "
            "so discovery matches what viewers hear — not only generic visuals. "
            "NEVER use meta filler: cinematic, caption, viral, content, video, reels, trending, photography, "
            "youtube, tiktok, instagram, follow, like, subscribe."
        )
    else:
        hashtag_rule = "Use empty [] for hashtags in every variant."

    title_rule = (
        "Include a non-empty title for YouTube when generate_title is true. "
        "Titles: specific to scene graph evidence; conversational capitalization; "
        "no emojis; avoid generic AI/clickbait openers (POV:, Wait until, This is why, You need to see). "
        "For TikTok use null title. For IG/FB title may be null or a 2–5 word headline."
        if generate_title
        else "Set title to null for all platforms that do not need titles."
    )

    title_evidence_contract = """
TITLE EVIDENCE BUILD CONTRACT (HARD — REJECTION RULES APPLY):
1. Build titles ONLY from these scene_graph fields (and their direct synonyms):
     - geo.road, geo.city, geo.state, geo.gazetteer_place, geo.protected_area_name
     - geo.max_speed_mph (formatted as "<N> MPH"), telemetry/osd peak speeds
     - trill.bucket (e.g. "Cruise", "Active", "Spirited", "Aggressive", "Reckless")
     - dominant tokens from vision.labels / video_intelligence.object_tracks /
       video_intelligence.on_screen_text / vision.landmarks / vision.logos
     - music.artist, music.title, music.genre — ARTIST or TITLE words ONLY,
       NEVER paraphrased lyrics or transcript phrases.
     - dashcam_osd.driver_name and dashcam_osd.first_seen.date when present.
     - scene_graph.timeline beats (place / on_screen_text / landmark fragments).
2. TITLES MUST NOT CONTAIN:
     - Any 4-word window that appears verbatim (case-insensitive) in
       transcript.text or transcript.segments[*].text.
     - Paraphrased lyric fragments. If music.copyright_risk is true or any
       transcript segment looks like a song lyric, treat the whole transcript as
       OFF-LIMITS for the title.
     - Profanity (bitch, shit, fuck, ass, damn, hell, slur tokens, etc.) — even
       when present in the source clip.
     - Generic clickbait openers: "POV:", "Wait until", "You won't believe",
       "This is why", "Watch this", "Insane", "Crazy" (used alone), "OMG".
3. PLATFORM TITLE SHAPES:
     - YouTube: 50–70 chars, Capitalized Headline Style, keyword-front-loaded
       (place / road / speed / Trill bucket leading); end with concrete noun
       (e.g. "I-5", "Yellowstone", "Mustang GT", "Cruise Run").
     - TikTok: title = null (caption-led platform).
     - Instagram / Facebook: 25–40 char punchy headline-style; nouns over verbs.
4. PER-PLATFORM VARIANCE:
     - YouTube title and Instagram/Facebook title MUST share at most 30% token
       overlap. If you start from the same evidence cluster, swap the leading
       cluster (geo → speed → trill → visual-object → music) for the smaller
       platform. The ranker WILL penalize duplicate titles.
5. EVIDENCE FALLBACK:
     - If none of the allowed fields are populated, return null for title on
       every platform. NEVER invent a place, speed, song, or driver name.
"""

    caption_rule = (
        "Write captions only when generate_caption is true; otherwise use empty string."
        if not generate_caption
        else (
            "Write vivid, specific captions grounded in Scene Graph evidence. "
            "When transcript.text is present, reflect real speech: topics, jokes, questions, or emotional beats "
            "(paraphrase; do not invent lines). When music.title / music.artist / music.genre are set, weave in "
            "factual listening context without claiming false ownership when music.copyright_risk is true "
            "or transcript indicates third-party lyrics. "
            "When audio_environment (sound_profile, yamnet_events, top_sound_class) is populated, mention "
            "concrete sounds (crowd, engine, rain, studio, conversation hubbub, etc.) where it strengthens the hook."
        )
    )

    sg = json.dumps(scene_graph, indent=2)[:24000]

    must_use_base = build_must_use_shortlist(scene_graph)
    must_use = merge_m8_must_use_tokens(must_use_base, ctx, max_tokens=12)
    must_use_block = ""
    cluster_block = ""
    if must_use:
        bullets = "\n".join(f"  - {tok}" for tok in must_use)
        must_use_block = (
            "\nMUST_USE EVIDENCE SHORTLIST (these tokens are FACTS from this clip — "
            "every winning caption AND title MUST contain at least 2 of these verbatim):\n"
            f"{bullets}\n"
            "Variants that fail to use ≥2 of these tokens will be REJECTED by the ranker "
            "with a -200 score and CANNOT win, regardless of how good the prose sounds.\n"
        )
        evidence_clusters: List[str] = []
        if any(("MPH" in t) or ("mph" in t) for t in must_use):
            evidence_clusters.append("SPEED-anchored")
        sg_geo = scene_graph.get("geo") or {}
        if sg_geo.get("road") or sg_geo.get("city") or sg_geo.get("gazetteer_place") or sg_geo.get("protected_area_name"):
            evidence_clusters.append("GEO-anchored")
        sg_music = scene_graph.get("music") or {}
        if sg_music.get("artist") or sg_music.get("title"):
            evidence_clusters.append("MUSIC-anchored")
        sg_tx = scene_graph.get("transcript") or {}
        if sg_tx.get("text") or (isinstance(sg_tx.get("structured"), dict) and sg_tx.get("structured")):
            evidence_clusters.append("SPEECH-anchored")
        sg_vi = scene_graph.get("video_intelligence") or {}
        if sg_vi.get("object_tracks") or sg_vi.get("on_screen_text") or sg_vi.get("logos"):
            evidence_clusters.append("VISUAL-OBJECT-anchored")
        sg_trill = scene_graph.get("trill") or {}
        if sg_trill.get("bucket"):
            evidence_clusters.append("TRILL-anchored")
        if evidence_clusters:
            unique_clusters = list(dict.fromkeys(evidence_clusters))[:5]
            picks = ", ".join(unique_clusters)
            cluster_block = (
                "\nEVIDENCE-CLUSTER VARIANT TARGETS: Diversify the 5 variants so they each lead with a "
                f"different anchor cluster when possible — available clusters here: {picks}. "
                "Each variant should foreground a different cluster as its primary hook. "
                "Variant 1 = strongest cluster, Variant 5 = secondary cluster, etc. This produces "
                "variety without sacrificing factual grounding.\n"
            )

    visual_memory_block = ""
    hist = historical or {}
    recall = hist.get("__visual_entity_recall__") or {}
    if isinstance(recall, dict) and any(recall.values()):
        try:
            from services.visual_entity_memory import format_recall_for_prompt

            vmem = format_recall_for_prompt(recall)
            if vmem:
                visual_memory_block = f"\n{vmem}\n"
        except Exception:
            pass

    pattern_block = ""
    pats = hist.get("__pattern_corpus__") or []
    if isinstance(pats, list) and pats:
        lines: List[str] = []
        for i, row in enumerate(pats[:8], 1):
            if not isinstance(row, dict):
                continue
            pl = str(row.get("platform") or "").lower()
            snip = str(row.get("snippet") or "").strip().replace("\n", " ")[:400]
            if len(snip) < 20:
                continue
            st = str(row.get("caption_style") or "").strip()
            tn = str(row.get("caption_tone") or "").strip()
            meta = f" [{st}/{tn}]" if st or tn else ""
            lines.append(f'{i}. ({pl}){meta}: "{snip}"')
        if lines:
            pattern_block = (
                "\nPATTERN MEMORY (your past uploads in this category - imitate opening rhythm, line breaks, and specificity; "
                "do NOT copy phrases verbatim; stay truthful to the current Scene Graph):\n"
                + "\n".join(lines)
                + "\n"
            )

    strategy_priors_block = ""
    pri = hist.get("__strategy_priors__") or {}
    pri_top = pri.get("top") if isinstance(pri, dict) else []
    if isinstance(pri_top, list) and pri_top:
        s_lines: List[str] = []
        for i, row in enumerate(pri_top[:5], 1):
            if not isinstance(row, dict):
                continue
            key = str(row.get("strategy_key") or "default")
            wm = float(row.get("weighted_mean_engagement") or 0.0)
            ci = float(row.get("max_ci95_high") or 0.0)
            n = int(row.get("samples") or 0)
            s_lines.append(f"{i}. key={key} | mean={wm:.2f}% | ci95_high={ci:.2f}% | n={n}")
        if s_lines:
            strategy_priors_block = (
                "\nML STRATEGY PRIORS (from upload_quality_scores_daily; use as directional bias, not hard rules):\n"
                + "\n".join(s_lines)
                + "\n"
                + "Bias guidance: favor hooks/angles that resemble higher-ranked keys, but always stay grounded in current Scene Graph evidence."
                + "\n"
            )

    matrix_section = ""
    matrix_close = ""
    if include_evidence_matrix:
        specs = _evidence_matrix_cell_specs(caption_style, caption_tone, caption_voice_ui)
        order_h = "; ".join(f"{a}|{b}|{c}" for a, b, c in specs)
        n_cells = len(specs)
        matrix_section = f"""
CAPTION EVIDENCE MATRIX (TikTok-style micro-captions, SAME frames + scene graph):
- Add top-level JSON key caption_evidence_matrix with object {{ "cells": [ ... ] }} (replace the placeholder empty array with {n_cells} filled objects).
- cells MUST have EXACTLY {n_cells} objects in THIS order (caption_style|caption_tone|caption_voice):
  {order_h}
- Each cell: caption_style, caption_tone, caption_voice (match row), tiktok_caption (45-320 chars),
  hashtags (0-6 strings without #) grounded in the same evidence as the main task.
- Block A: styles story,punchy,factual x each UI voice with tone FIXED to "{caption_tone}".
- Block B: tones authentic,hype,calm,cinematic with style FIXED to "{caption_style}" and voice FIXED to "{caption_voice_ui}".
- REGURGITATE at least one concrete token from vision, OCR, geo, transcript, VI timeline, or telemetry per cell when present.
- No duplicate captions; vary hook while staying truthful.
"""
        matrix_close = ',\n  "caption_evidence_matrix": {\n    "cells": []\n  }'

    hydration_contract_block = m8_hydration_contract_block(ctx, generate_hashtags=generate_hashtags)

    return f"""You are {M8_ENGINE_AI_DISPLAY} - UploadM8's M8_ENGINE multimodal publishing brain
(version {M8_ENGINE_VERSION}). You combine evidence from the scene graph with M8_ENGINE AI priors when provided.

Your job: produce HIGH-SPECIFICITY, NON-GENERIC titles/captions that clearly come from THIS footage,
not from a template. Avoid vague influencer filler. No false claims.

MULTIMODAL MANDATE: The Scene Graph may be long on purpose. Integrate ALL relevant layers -
transcript, music ID + genre, background/YAMNet sound, fusion_narrative/content_signals when present,
emotional_tone from audio context when present, GPS/lat-lon + place names, telemetry + Trill,
dashcam_osd HUD facts (speed/date/GPS/driver) when present,
Google Video Intelligence (full-clip labels + segment timeline when present),
merged multi-frame Vision (union labels + OCR from several moments), faces, landmark hints, and video understanding.
Use scene_graph.hydration_story as the short factual scene paragraph: it is the preferred backbone for chaining
time, place, visual subjects, music, speech, and analysis into one grounded story beat.
Use scene_graph.timeline (when present) as the ORDERED list of beats in this video — each entry is
{{t_seconds, kind, text}}. Treat it as the source of truth for sequence: hook from an early beat (low t_seconds),
land the climax from a peak beat (e.g. osd_speed / object / on_screen_text / landmark), and tie back to a
late beat for the closer. Captions and titles must reference at least one beat token verbatim (place name,
MPH number, on-screen text fragment, landmark, or driver) when the timeline contains them.
When trend_intel.summary / trend_intel.rows are present, treat them as **recent search-title shape** for the niche only —
borrow pacing and topic nouns, never copy titles or pretend this video matches an unrelated viral clip.
Do NOT write from transcript alone when other fields are populated. Prefer nouns from labels + geo + OCR over generic hype.
If dashcam_osd.gps_fix_count or geo fields are populated, each title/caption pair must include at least one
real route/HUD/place anchor: speed in MPH, road/city/state/protected-area/gazetteer place, Trill bucket, or driver name.

CAPTION + HASHTAG AUDIO DISCIPLINE: If transcript.text is non-empty, the caption hook should acknowledge what is
actually said (themes, names, punchlines) without fabricated quotes. If music.* identifies a track, captions and
hashtags may reference artist/title/genre tokens for discovery while respecting third-party / copyright semantics in
ANTI-GENERIC RULES. If audio_environment or fusion_narrative describe ambience, fold 1–2 audible cues into prose
and hashtag tokens so posts match how the clip sounds, not only how it looks.

SCENE GRAPH (evidence - do not invent beyond it; you may interpret visually grounded details):
{sg}

CATEGORY: {category}
USER ACCOUNT SETTINGS (UI): caption_style={caption_style} caption_tone={caption_tone} caption_voice={caption_voice_ui}
EFFECTIVE STRATEGY (policy/ML context — per-platform nuance only, NOT a license to flatten the user's voice): style={base_style} tone={base_tone} persona={base_persona}
{_user_brand_directive(ctx)}
{_m8_creative_directive_block(caption_style, caption_tone, caption_voice_ui)}
{_task_prompt(generate_title, generate_caption, generate_hashtags)}

{fusion}
{title_evidence_contract}
{must_use_block}
{hydration_contract_block}
{cluster_block}
{pattern_block}
{visual_memory_block}
{strategy_priors_block}{extra_strategy_block}
PLATFORM RULES:
{chr(10).join(plat_blocks)}
{matrix_section}
TASK:
For EACH platform listed in scene_graph.platforms, output EXACTLY 5 variants ranked as "variant_index" 1..5.
Each variant must feel meaningfully different (hook style, angle, emotion), not minor word swaps.

Fields per variant:
- title: string or null (YouTube needs title; TikTok null)
- caption: string (required for platforms that use captions)
- hashtags: array of strings without # (or empty array)

{title_rule}
{caption_rule}
{hashtag_rule}

ANTI-GENERIC RULES:
- Do not claim the creator wrote/performed an original song if transcript_role is third_party_lyrics or music is identified as a known track unless Scene Graph explicitly says otherwise.
- Do not write in first person as the recording artist (no "I channel", "my vocals", "my track") when lyrics are third-party unless the visuals show a clear performance.
- Prefer concrete nouns from labels/OCR/geo/telemetry over abstract hype.
- When scene_graph.vision.recognition_summary or recognition_flat is present, treat it as
  the primary inventory of visible entities (vehicles with year/make, brands, food,
  colors, products) and cite specific items from it in titles/captions/hashtags.
- HASHTAGS: Prefer scene_graph.timeline beats, video_intelligence.on_screen_text, geo,
  landmarks, logos, music artist/title, and VI object tracks. Do NOT use coarse Vision
  detector labels (color, windshield, boat, motorvehicle, lensflare, driving) unless they
  also appear in must_use tokens or on-screen text.
- Hashtags must read like real community search terms, not platform metadata.
- Captions are prose-only: never paste JSON hashtag arrays or quoted hash-plus-bracket fragments into caption; use the hashtags array only.
- No Unicode emojis or emoticons in any title or caption field.

Return ONLY valid JSON (no markdown) in this exact shape:
{{
  "m8_version": "{M8_ENGINE_VERSION}",
  "platforms": {{
    "tiktok": {{
      "variants": [
        {{"variant_index": 1, "title": null, "caption": "...", "hashtags": ["a","b"]}},
        {{"variant_index": 2, "title": null, "caption": "...", "hashtags": []}},
        {{"variant_index": 3, "title": null, "caption": "...", "hashtags": []}},
        {{"variant_index": 4, "title": null, "caption": "...", "hashtags": []}},
        {{"variant_index": 5, "title": null, "caption": "...", "hashtags": []}}
      ]
    }}
  }}{matrix_close}
}}

Include only keys for platforms in scene_graph.platforms (lowercase).
"""


def _penalize_generic(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    pen = 0.0
    for pat in _GENERIC_PATTERNS:
        if re.search(pat, t, re.I):
            pen += 8.0
    return pen


def _specificity_bonus(caption: str, title: str, scene_graph: Dict[str, Any]) -> float:
    """Reward overlap with labels/OCR/geo/fusion tokens."""
    blob = f"{caption} {title}".lower()
    toks: List[str] = []
    vis = scene_graph.get("vision") or {}
    labels = vis.get("labels") or []
    for x in labels[:20]:
        s = str(x).lower().strip()
        if len(s) > 2:
            toks.append(s)
    for key in ("landmarks", "logos"):
        for x in (vis.get(key) or [])[:8]:
            s = str(x).lower().strip()
            if len(s) > 2:
                toks.append(s)
    ocr = (scene_graph.get("vision", {}) or {}).get("ocr") or ""
    for part in re.split(r"\W+", ocr.lower()):
        if len(part) > 3:
            toks.append(part)
    for g in ("display", "start_display", "road", "city", "state", "gazetteer_place", "protected_area_name"):
        v = (scene_graph.get("geo") or {}).get(g)
        if v:
            toks.append(str(v).lower())
    osd = scene_graph.get("dashcam_osd") or {}
    if osd:
        for key in ("driver_name", "speed_unit_detected", "engine"):
            v = osd.get(key)
            if v:
                toks.append(str(v).lower())
        for key in ("max_speed_mph", "avg_speed_mph", "gps_fix_count"):
            v = osd.get(key)
            if v not in (None, ""):
                toks.append(str(int(float(v))) if str(v).replace(".", "", 1).isdigit() else str(v).lower())
    tr = scene_graph.get("trill") or {}
    if tr:
        for key in ("bucket", "title_modifier"):
            v = tr.get(key)
            if v:
                toks.append(str(v).lower())
        for x in tr.get("hashtags") or []:
            s = str(x).lower().strip().lstrip("#")
            if len(s) > 2:
                toks.append(s)
    mus = scene_graph.get("music") or {}
    if mus.get("detected"):
        for key in ("artist", "title", "genre"):
            v = mus.get(key)
            if v:
                toks.append(str(v).lower())
    ti = scene_graph.get("trend_intel") or {}
    for row in (ti.get("rows") or [])[:6]:
        if not isinstance(row, dict):
            continue
        tit = str(row.get("title") or "").lower()
        for part in re.split(r"\W+", tit):
            if len(part) > 3:
                toks.append(part)
    bonus = 0.0
    for tok in toks:
        if tok and tok in blob:
            bonus += 3.0
    return min(bonus, 25.0)


def _missing_primary_hydration(text: str, scene_graph: Dict[str, Any]) -> bool:
    """True when rich route/HUD/audio evidence exists but copy does not use any of it."""
    blob = (text or "").lower()
    if not blob:
        return True
    anchors: List[str] = []
    geo = scene_graph.get("geo") or {}
    for key in ("road", "city", "state", "gazetteer_place", "protected_area_name"):
        v = geo.get(key)
        if v:
            anchors.append(str(v))
    osd = scene_graph.get("dashcam_osd") or {}
    if osd.get("max_speed_mph"):
        try:
            anchors.append(str(int(round(float(osd.get("max_speed_mph"))))))
        except (TypeError, ValueError):
            pass
    tr = scene_graph.get("trill") or {}
    if tr.get("bucket"):
        anchors.append(str(tr.get("bucket")))
    mus = scene_graph.get("music") or {}
    for key in ("artist", "title", "genre"):
        if mus.get(key):
            anchors.append(str(mus.get(key)))
    if not anchors:
        return False
    return not any(a.lower().strip().lstrip("#") and a.lower().strip().lstrip("#") in blob for a in anchors)


def _primary_hydration_penalty(caption: str, title: str, scene_graph: Dict[str, Any]) -> float:
    if _missing_primary_hydration(f"{caption} {title}", scene_graph):
        return 18.0
    return 0.0


def build_must_use_shortlist(scene_graph: Dict[str, Any], *, max_tokens: int = 12) -> List[str]:
    """Concrete tokens the model MUST use ≥2 of in any winning caption/title.

    Drawn from the same evidence pool the hydration enforcer uses, but emitted
    as **prompt-friendly verbatim phrases** (mixed case, with units) so the
    model can copy them straight into prose. Order matters — earlier entries
    are higher-value anchors and are surfaced first to the LLM.
    """
    out: List[str] = []
    seen: set = set()

    def _push(token: Any) -> None:
        s = str(token or "").strip()
        if not s:
            return
        key = s.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(s)

    # 1. Speed (HUD / .map) — highest-info short anchor
    osd = scene_graph.get("dashcam_osd") or {}
    geo = scene_graph.get("geo") or {}
    max_mph = osd.get("max_speed_mph") or geo.get("max_speed_mph")
    if max_mph:
        try:
            _push(f"{int(round(float(max_mph)))} MPH")
        except (TypeError, ValueError):
            pass

    # 2. Place (road, gazetteer, city/state, protected area)
    if geo.get("road"):
        _push(str(geo.get("road")))
    if geo.get("gazetteer_place"):
        gz = str(geo.get("gazetteer_place"))
        usps = geo.get("gazetteer_state_usps") or geo.get("state")
        if usps:
            _push(f"{gz}, {usps}")
        else:
            _push(gz)
    elif geo.get("city"):
        city = str(geo.get("city"))
        if geo.get("state"):
            _push(f"{city}, {geo.get('state')}")
        else:
            _push(city)
    if geo.get("protected_area_name"):
        _push(str(geo.get("protected_area_name")))

    # 3. Music (ACR catalogue)
    music = scene_graph.get("music") or {}
    if music.get("artist") and music.get("title"):
        _push(f"{music.get('artist')} — {music.get('title')}")
    elif music.get("artist"):
        _push(str(music.get("artist")))
    elif music.get("title"):
        _push(str(music.get("title")))

    # 4. Trill bucket (driving energy)
    trill = scene_graph.get("trill") or {}
    if trill.get("bucket"):
        _push(f"Trill {trill.get('bucket')}")

    # 5. Vision landmarks + logos (specific named entities)
    vision = scene_graph.get("vision") or {}
    for lm in (vision.get("landmarks") or [])[:3]:
        _push(str(lm))
    for lg in (vision.get("logos") or [])[:3]:
        _push(str(lg))

    # 6. Video Intelligence object/person/text/logo tracks (added by phase 3)
    vi = scene_graph.get("video_intelligence") or {}
    for label in (vi.get("top_labels") or [])[:4]:
        desc = str(label).split(" (", 1)[0].strip()
        if desc and not is_generic_vision_label(desc):
            _push(desc)
    for row in (vi.get("segment_labels") or [])[:4]:
        if isinstance(row, dict) and row.get("description"):
            _push(str(row.get("description")))
        elif isinstance(row, str):
            _push(row)
    for track in (vi.get("object_tracks") or [])[:4]:
        if isinstance(track, dict) and track.get("description"):
            _push(str(track.get("description")))
    for txt in (vi.get("on_screen_text") or [])[:3]:
        if isinstance(txt, dict) and txt.get("text"):
            _push(str(txt.get("text"))[:60])
        elif isinstance(txt, str):
            _push(txt[:60])
    for logo in (vi.get("logos") or [])[:3]:
        if isinstance(logo, dict) and logo.get("description"):
            _push(str(logo.get("description")))

    # 6b. Google visual recognition buckets (vehicles, food, plants, …)
    rec_flat = vision.get("recognition_flat") or {}
    if isinstance(rec_flat, dict):
        cat = str(scene_graph.get("category") or "general").lower()
        try:
            from core.visual_entity_taxonomy import niche_bucket_order

            bucket_order = niche_bucket_order(cat)
        except Exception:
            bucket_order = ["vehicles", "food", "plants", "brands", "objects", "outdoors", "sports"]
        for bucket in bucket_order:
            for name in (rec_flat.get(bucket) or [])[:3]:
                if name and not is_generic_vision_label(str(name), min_specific_len=3):
                    _push(str(name))

    # 7. OSD driver name (HUD)
    if osd.get("driver_name"):
        _push(str(osd.get("driver_name")))

    # 8. Whisper structured topics / entities (added by phase 2)
    transcript = scene_graph.get("transcript") or {}
    structured = transcript.get("structured") if isinstance(transcript, dict) else {}
    if isinstance(structured, dict):
        ents = structured.get("named_entities") or {}
        if isinstance(ents, dict):
            for kind in ("places", "people", "products"):
                for name in (ents.get(kind) or [])[:2]:
                    _push(str(name))
        for topic in (structured.get("topics") or [])[:3]:
            _push(str(topic))
        if structured.get("key_phrase"):
            _push(str(structured.get("key_phrase"))[:80])

    return out[:max_tokens]


def _evidence_coverage(text: str, must_use: List[str]) -> int:
    """Count how many distinct must_use tokens appear in the text (case-insensitive)."""
    if not must_use or not text:
        return 0
    blob = text.lower()
    hits = 0
    for tok in must_use:
        if not tok:
            continue
        # Use first 3 words as a fuzzier match for long tokens like
        # "The Eagles — Hotel California" so word-order variations still count.
        head = " ".join(str(tok).lower().split()[:3])
        if head and head in blob:
            hits += 1
            continue
        if str(tok).lower() in blob:
            hits += 1
    return hits


def _evidence_coverage_score(
    caption: str, title: str, must_use: List[str], *, min_required: int = 2
) -> float:
    """Asymmetric: missing all evidence = -200 (hard reject), partial = recovery curve.

    This is the offense that makes the hydration enforcer rarely fire. A
    template-style variant ("Cruise under vast skies") will score 0 hits and
    receive -200.0, putting it well below any evidence-bearing variant — the
    LLM cannot win with category-seed copy when concrete evidence exists.
    """
    if not must_use:
        return 0.0
    blob = f"{caption} {title}"
    hits = _evidence_coverage(blob, must_use)
    if hits == 0:
        return -200.0
    if hits < min_required:
        return -50.0
    # Diminishing returns above min_required so we don't reward over-stuffing.
    return min(40.0, 18.0 + (hits - min_required) * 6.0)


def _best_hydration_anchor(scene_graph: Dict[str, Any]) -> str:
    geo = scene_graph.get("geo") or {}
    osd = scene_graph.get("dashcam_osd") or {}
    parts: List[str] = []
    if osd.get("max_speed_mph"):
        try:
            parts.append(f"{int(round(float(osd.get('max_speed_mph'))))} MPH")
        except (TypeError, ValueError):
            pass
    if geo.get("road"):
        parts.append(str(geo.get("road")))
    if geo.get("gazetteer_place"):
        parts.append(str(geo.get("gazetteer_place")))
    elif geo.get("city"):
        place = str(geo.get("city"))
        if geo.get("state"):
            place = f"{place}, {geo.get('state')}"
        parts.append(place)
    elif geo.get("protected_area_name"):
        parts.append(str(geo.get("protected_area_name")))
    tr = scene_graph.get("trill") or {}
    if tr.get("bucket"):
        parts.append(f"Trill {tr.get('bucket')}")
    mus = scene_graph.get("music") or {}
    if mus.get("artist") and mus.get("title"):
        parts.append(f"{mus.get('artist')} - {mus.get('title')}")
    return " near ".join(p for p in parts[:3] if p)


def _hydrate_caption_with_anchor(caption: str, scene_graph: Dict[str, Any]) -> str:
    anchor = _best_hydration_anchor(scene_graph)
    if not anchor:
        return caption
    caption = (caption or "").strip()
    if not caption:
        return anchor
    if anchor.lower() in caption.lower():
        return caption
    return f"{anchor}: {caption}"[:520]


def _hydrate_title_with_anchor(title: str, scene_graph: Dict[str, Any]) -> str:
    anchor = _best_hydration_anchor(scene_graph)
    title = (title or "").strip()
    if not anchor:
        return title
    if title and not _missing_primary_hydration(title, scene_graph):
        return title[:100]
    return anchor[:100]


def _attribution_penalty(caption: str, title: str, scene_graph: Dict[str, Any]) -> float:
    """Penalize claiming original music when lyrics are third-party."""
    role = (scene_graph.get("transcript") or {}).get("role") or ""
    music_on = bool((scene_graph.get("music") or {}).get("detected"))
    blob = f"{caption} {title}".lower()
    penalty = 0.0
    if role in ("third_party_lyrics", "third_party_music") or music_on:
        bad = [
            "i wrote",
            "my song",
            "my track",
            "new music",
            "original song",
            "just dropped my",
        ]
        for b in bad:
            if b in blob:
                penalty += 20.0
    return penalty


def _length_score(platform: str, caption: str, title: Optional[str]) -> float:
    p = (platform or "").lower()
    c = (caption or "").strip()
    t = (title or "").strip()
    score = 0.0
    if p == "tiktok":
        ln = len(c)
        if 60 <= ln <= 280:
            score += 12.0
        elif ln < 40:
            score -= 5.0
    if p == "youtube":
        if 20 <= len(t) <= 100:
            score += 10.0
        ln = len(c)
        if 80 <= ln <= 900:
            score += 8.0
    if p in ("instagram", "facebook"):
        ln = len(c)
        if 80 <= ln <= 400:
            score += 10.0
    return score


def _hook_strength_score(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return -8.0
    first = t[:75].lower()
    score = 0.0
    if "?" in first:
        score += 2.0
    if any(k in first for k in ("when", "why", "how", "this", "pov", "today")):
        score += 3.0
    if len(first.split()) <= 16:
        score += 2.0
    return score


def _quality_gate_penalty(platform: str, title: str, caption: str) -> float:
    blob = f"{title} {caption}".strip().lower()
    if not blob:
        return 20.0
    penalty = 0.0
    weak = ("watch this", "road vibes", "goes insane", "must watch")
    for w in weak:
        if w in blob:
            penalty += 14.0
    if platform == "youtube" and len((title or "").strip()) < 12:
        penalty += 6.0
    if len((caption or "").strip()) < 35:
        penalty += 8.0
    return penalty


def _preflight_artifact_checks(
    platform: str,
    candidate: Dict[str, Any],
    scene_graph: Dict[str, Any],
    strategy_target: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    title = str(candidate.get("title") or "").strip()
    caption = str(candidate.get("caption") or "").strip()
    tags = candidate.get("hashtags") or []
    weak_words = ("watch this", "road vibes", "goes insane", "must watch")
    title_fail = any(w in title.lower() for w in weak_words) or (platform == "youtube" and len(title) < 12)
    caption_fail = (len(caption) < 35) or any(w in caption.lower() for w in weak_words)
    # Tone mismatch heuristic: calm tone shouldn't use high-hype spam wording.
    tone = str((strategy_target or {}).get("tone") or "").lower()
    if tone in ("calm", "professional") and any(k in caption.lower() for k in ("insane", "crazy", "omg", "wtf")):
        caption_fail = True
    # Basic hashtag quality checks: duplicates/spam/meta tags.
    seen = set()
    hash_fail = False
    for t in tags if isinstance(tags, list) else []:
        slug = str(t).strip().lstrip("#").lower()
        if not slug:
            continue
        if slug in seen:
            hash_fail = True
            break
        seen.add(slug)
        if slug in {"viral", "trending", "follow", "like", "subscribe"}:
            hash_fail = True
            break
        if is_generic_vision_label(slug):
            hash_fail = True
            break
    # Persona consistency sanity check
    persona = str((strategy_target or {}).get("persona") or "").lower()
    persona_fail = False
    if persona == "expert_analyst" and any(x in caption.lower() for x in ("bro", "frfr", "no cap")):
        persona_fail = True
    return {
        "title_ok": not title_fail,
        "caption_ok": not caption_fail,
        "hashtags_ok": not hash_fail,
        "persona_ok": not persona_fail,
    }


def _repair_artifacts_selective(
    platform: str,
    winner: Dict[str, Any],
    ranked: List[Dict[str, Any]],
    scene_graph: Dict[str, Any],
    strategy_target: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    checks = _preflight_artifact_checks(platform, winner, scene_graph, strategy_target)
    if winner and _missing_primary_hydration(
        f"{winner.get('caption') or ''} {winner.get('title') or ''}",
        scene_graph,
    ):
        repaired = dict(winner or {})
        repaired["caption"] = _hydrate_caption_with_anchor(str(repaired.get("caption") or ""), scene_graph)
        if platform == "youtube":
            repaired["title"] = _hydrate_title_with_anchor(str(repaired.get("title") or ""), scene_graph)
        return repaired, _preflight_artifact_checks(platform, repaired, scene_graph, strategy_target)
    if all(checks.values()):
        return winner, checks

    repaired = dict(winner or {})
    runner = ranked[1] if len(ranked) > 1 else {}
    if not checks.get("title_ok", True):
        alt_t = str(runner.get("title") or "").strip()
        repaired["title"] = alt_t if alt_t else str(repaired.get("title") or "").replace("WATCH THIS", "").strip()
    if not checks.get("caption_ok", True):
        alt_c = str(runner.get("caption") or "").strip()
        if alt_c:
            repaired["caption"] = alt_c
        else:
            c = str(repaired.get("caption") or "").strip()
            repaired["caption"] = ("POV: " + c) if c and not c.lower().startswith("pov") else c
    if not checks.get("hashtags_ok", True):
        tags = repaired.get("hashtags") or []
        cleaned: List[str] = []
        seen = set()
        for t in tags if isinstance(tags, list) else []:
            slug = str(t).strip().lstrip("#").lower()
            if not slug or slug in seen or slug in {"viral", "trending", "follow", "like", "subscribe"}:
                continue
            seen.add(slug)
            cleaned.append(slug)
        repaired["hashtags"] = cleaned[:12]
    if not checks.get("persona_ok", True):
        cap = str(repaired.get("caption") or "")
        cap = re.sub(r"\b(bro|frfr|no cap)\b", "", cap, flags=re.IGNORECASE).strip()
        repaired["caption"] = cap

    if _missing_primary_hydration(
        f"{repaired.get('caption') or ''} {repaired.get('title') or ''}",
        scene_graph,
    ):
        repaired["caption"] = _hydrate_caption_with_anchor(str(repaired.get("caption") or ""), scene_graph)
        if platform == "youtube":
            repaired["title"] = _hydrate_title_with_anchor(str(repaired.get("title") or ""), scene_graph)

    post = _preflight_artifact_checks(platform, repaired, scene_graph, strategy_target)
    return repaired, post


def score_variant(
    platform: str,
    variant: Dict[str, Any],
    scene_graph: Dict[str, Any],
    historical_signals: Optional[Dict[str, Any]] = None,
    must_use: Optional[List[str]] = None,
) -> Tuple[float, str]:
    """
    Return (score, rationale). Higher is better.
    historical_signals: optional per-platform priors from DB/analytics (future).
    must_use: shortlist of evidence tokens that the variant MUST cover ≥2 of.
              Passing must_use=[] disables the hard-fail floor (used when no
              evidence is available at all — defense via hydration_enforcer
              still runs in those cases).
    """
    caption = str(variant.get("caption") or "")
    title = str(variant.get("title") or "") if variant.get("title") is not None else ""

    base = 50.0
    base += _length_score(platform, caption, title)
    base += _hook_strength_score(caption or title)
    base += _specificity_bonus(caption, title, scene_graph)
    base -= _penalize_generic(caption + " " + title)
    base -= _attribution_penalty(caption, title, scene_graph)
    base -= _quality_gate_penalty(platform, title, caption)
    base -= _primary_hydration_penalty(caption, title, scene_graph)
    tags = variant.get("hashtags") or []
    if isinstance(tags, list):
        base -= penalize_generic_vision_hashtags(tags)

    # Hard evidence-coverage floor: any candidate that uses ZERO tokens from
    # the must_use shortlist when evidence exists scores -200. This is the
    # mechanism that makes "Cruise under vast skies"-style template copy
    # literally unable to win — no length/hook bonus can recover from -200.
    if must_use:
        cov = _evidence_coverage_score(caption, title, must_use)
        base += cov
        coverage_note = f" (evidence_coverage={cov:+.0f})"
    else:
        coverage_note = ""

    hist_note = ""
    if historical_signals and platform in historical_signals:
        h = historical_signals[platform]
        eng = float(h.get("engagement_prior", 0) or 0)
        base += min(10.0, max(-5.0, eng))
        hist_note = " (+historical prior)"

    rationale = (
        f"length/style fit + specificity{coverage_note} "
        f"- generic/attribution penalties{hist_note}"
    )
    return base, rationale


def rank_and_select(
    parsed: Dict[str, Any],
    scene_graph: Dict[str, Any],
    historical_signals: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None,
    *,
    ctx: Optional[JobContext] = None,
) -> Dict[str, Any]:
    """
    Pick best variant per platform. Adds ranking_debug with scores.

    Computes must_use shortlist once per call so every variant on every
    platform is ranked against the same evidence anchors. The shortlist
    becomes part of the returned debug payload so downstream tools (and
    the hydration enforcer) can audit which evidence was available.
    """
    out_plat: Dict[str, Any] = {}
    platforms_block = (parsed.get("platforms") or {}) if isinstance(parsed, dict) else {}
    must_use_base = build_must_use_shortlist(scene_graph)
    must_use = merge_m8_must_use_tokens(must_use_base, ctx, max_tokens=12)

    for pl in scene_graph.get("platforms") or []:
        pl = str(pl).lower()
        strategy_target = ((strategy or {}).get("outputs") or {}).get("platform_targets", {}).get(pl, {})
        pdata = platforms_block.get(pl) or {}
        variants = pdata.get("variants") or []
        ranked: List[Dict[str, Any]] = []
        best: Optional[Tuple[float, Dict[str, Any]]] = None
        for v in variants[:12]:
            if not isinstance(v, dict):
                continue
            sc, why = score_variant(pl, v, scene_graph, historical_signals, must_use=must_use)
            item = {
                "variant_index": v.get("variant_index"),
                "title": v.get("title"),
                "caption": v.get("caption"),
                "hashtags": v.get("hashtags") or [],
                "score": round(sc, 4),
                "rationale": why,
            }
            ranked.append(item)
            if best is None or sc > best[0]:
                best = (sc, item)
        ranked.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
        winner = None
        runner_up = None
        title_validation_meta: Dict[str, Any] = {"rejected_titles": []}
        if ranked:
            winner = ranked[0]
            if len(ranked) > 1:
                runner_up = ranked[1]
            # Quality gate: if top result still weak, use best non-weak candidate.
            for cand in ranked:
                if float(cand.get("score") or 0) >= 45.0:
                    winner = cand
                    break

            # ── Title hard-ban filter ────────────────────────────────────
            # Walk candidates in score order; first one whose title passes
            # _validate_title wins. If every candidate fails AND we still need
            # a title for this platform, fall back to a deterministic
            # evidence-only title built from allowed scene_graph fields.
            need_title = pl in ("youtube", "instagram", "facebook")
            if need_title:
                for cand in ranked:
                    cand_title = str(cand.get("title") or "").strip()
                    ok_t, reason_t = _validate_title(cand_title, scene_graph, platform=pl)
                    if ok_t:
                        winner = cand
                        break
                    title_validation_meta["rejected_titles"].append({
                        "variant_index": cand.get("variant_index"),
                        "title": cand_title[:120],
                        "reason": reason_t,
                    })
                else:
                    # All variants failed — rewrite winner.title with a
                    # deterministic evidence-only title.
                    fallback = _deterministic_evidence_title(scene_graph, platform=pl)
                    if winner is None:
                        winner = ranked[0]
                    if fallback:
                        winner = dict(winner)
                        winner["title"] = fallback
                        title_validation_meta["evidence_fallback_used"] = True
                    else:
                        # No allowed evidence — return null title rather than
                        # ship a transcript/lyric quote.
                        winner = dict(winner)
                        winner["title"] = None
                        title_validation_meta["evidence_fallback_used"] = False
                        title_validation_meta["title_set_to_null"] = True
            elif pl == "tiktok":
                # Caption-led platform — null out any title to enforce contract.
                if winner is not None:
                    winner = dict(winner)
                    winner["title"] = None

        preflight_meta: Dict[str, bool] = {}
        if winner:
            winner, preflight_meta = _repair_artifacts_selective(
                pl, winner, ranked, scene_graph, strategy_target
            )
        out_plat[pl] = {
            "variants_ranked": ranked,
            "winner": winner,
            "runner_up": runner_up,
            "preflight": preflight_meta,
            "title_validation": title_validation_meta,
        }

    return {
        "m8_version": parsed.get("m8_version") or M8_ENGINE_VERSION,
        "platforms": out_plat,
        "must_use": must_use,
    }


# ---------------------------------------------------------------------------
# Title hard-ban validator + deterministic evidence-only fallback
#
# Implements the TITLE EVIDENCE BUILD CONTRACT — keep transcript / lyrics /
# profanity OUT of every winning title across every platform. Used by
# rank_and_select to filter LLM variants, and by apply_selection_to_context
# to force per-platform variance.
# ---------------------------------------------------------------------------

# Small denylist — extend cautiously. Matched as whole-word case-insensitive.
_TITLE_PROFANITY = frozenset({
    "bitch", "bitches", "shit", "shits", "fuck", "fucking", "fucked", "fucker",
    "ass", "asses", "asshole", "damn", "dammit", "hell", "piss", "pussy",
    "cunt", "dick", "cock", "bastard", "slut", "whore", "nigga", "nigger",
    "retard", "retarded", "faggot",
})

# Generic clickbait openers (and short stop-phrases) that smell like AI filler.
_TITLE_CLICKBAIT_RE = re.compile(
    r"^(pov:|wait until|you won't believe|this is why|watch this|"
    r"omg|crazy[!.]*$|insane[!.]*$|you need to see|here'?s why)",
    re.IGNORECASE,
)

_TITLE_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _title_tokens(text: str) -> List[str]:
    return [t.lower() for t in _TITLE_TOKEN_RE.findall(str(text or ""))]


def _validate_title(
    candidate: str,
    scene_graph: Dict[str, Any],
    *,
    platform: str = "",
) -> Tuple[bool, str]:
    """Return (is_valid, reason). Empty/None titles are considered valid (=null)."""
    title = (candidate or "").strip()
    if not title:
        return True, "empty_ok"

    title_lc = title.lower()
    if _TITLE_CLICKBAIT_RE.search(title_lc):
        return False, "clickbait_opener"

    title_toks = _title_tokens(title)
    for w in title_toks:
        if w in _TITLE_PROFANITY:
            return False, f"profanity:{w}"

    transcript_data = scene_graph.get("transcript") or {}
    transcript_text = str(transcript_data.get("text") or "")
    if transcript_text and title_toks:
        title_lower_str = " " + " ".join(title_toks) + " "
        ttoks = _title_tokens(transcript_text)
        # Any 4-gram from the transcript appearing verbatim in the title fails.
        for i in range(0, max(0, len(ttoks) - 3)):
            gram = " " + " ".join(ttoks[i : i + 4]) + " "
            if gram in title_lower_str:
                return False, "transcript_overlap_4gram"
        # Direct >=3-word phrase from any transcript segment also fails.
        for seg in (transcript_data.get("segments") or [])[:60]:
            if not isinstance(seg, dict):
                continue
            seg_toks = _title_tokens(seg.get("text") or "")
            if len(seg_toks) < 3:
                continue
            for i in range(0, len(seg_toks) - 2):
                gram = " " + " ".join(seg_toks[i : i + 3]) + " "
                if gram in title_lower_str:
                    return False, "transcript_segment_3gram"

    # Platform-specific length sanity (HARD floors/ceilings — the contract
    # already asks for shape but the LLM sometimes ignores it).
    plat = (platform or "").lower()
    L = len(title)
    if plat == "youtube" and (L < 20 or L > 100):
        return False, f"youtube_length:{L}"
    if plat in ("instagram", "facebook") and L > 100:
        return False, f"{plat}_length:{L}"

    return True, "ok"


def _deterministic_evidence_title(
    scene_graph: Dict[str, Any],
    *,
    platform: str = "",
    preferred_cluster: str = "auto",
) -> Optional[str]:
    """Build a safe, evidence-only title using only allowed scene_graph fields.

    Returns ``None`` if no allowed evidence is available. The output is shaped
    per-platform: YouTube 50-70 chars headline; IG/FB 25-40 char short.
    """
    geo = scene_graph.get("geo") or {}
    trill = scene_graph.get("trill") or {}
    osd = scene_graph.get("dashcam_osd") or {}
    music = scene_graph.get("music") or {}
    vi = scene_graph.get("video_intelligence") or {}
    vision = scene_graph.get("vision") or {}

    place_token = ""
    for k in ("gazetteer_place", "protected_area_name", "city", "state", "road"):
        v = geo.get(k)
        if isinstance(v, str) and v.strip():
            place_token = v.strip()
            break

    speed_token = ""
    try:
        mph_val = geo.get("max_speed_mph")
        if mph_val is None:
            mph_val = osd.get("max_speed_mph")
        if mph_val is not None:
            mph = int(round(float(mph_val)))
            if mph >= 5:
                speed_token = f"{mph} MPH"
    except (TypeError, ValueError):
        pass

    bucket_token = ""
    bv = trill.get("bucket")
    if isinstance(bv, str) and bv.strip():
        bucket_token = bv.strip().title()

    visual_token = ""
    obj_tracks = vi.get("object_tracks") or []
    if isinstance(obj_tracks, list):
        for ot in obj_tracks[:3]:
            if not isinstance(ot, dict):
                continue
            d = str(ot.get("description") or "").strip()
            if d:
                visual_token = d.title()
                break
    if not visual_token:
        landmarks = vision.get("landmarks") or []
        if landmarks:
            visual_token = str(landmarks[0]).strip().title()

    music_token = ""
    artist = str(music.get("artist") or "").strip()
    track = str(music.get("title") or "").strip()
    if artist and track:
        music_token = f"{artist} — {track}"
    elif artist or track:
        music_token = artist or track

    cluster_order: List[str]
    pref = (preferred_cluster or "").lower()
    if pref in ("geo", "speed", "trill", "visual", "music"):
        rest = [c for c in ("geo", "speed", "trill", "visual", "music") if c != pref]
        cluster_order = [pref] + rest
    else:
        cluster_order = ["geo", "speed", "trill", "visual", "music"]

    cluster_token: Dict[str, str] = {
        "geo": place_token,
        "speed": speed_token,
        "trill": bucket_token,
        "visual": visual_token,
        "music": music_token,
    }

    parts: List[str] = []
    for cl in cluster_order:
        tok = cluster_token.get(cl) or ""
        if tok and tok not in parts:
            parts.append(tok)
        if len(parts) >= 3:
            break

    if not parts:
        return None

    plat = (platform or "").lower()
    if plat == "youtube":
        joined = " · ".join(parts[:3])
        if speed_token and speed_token not in joined and len(joined) < 60:
            joined = f"{joined} at {speed_token}"
        if len(joined) > 70:
            joined = joined[:67].rstrip(" ,-—·") + "..."
        if len(joined) < 20 and bucket_token and bucket_token not in joined:
            joined = f"{joined} — {bucket_token} Run"
        return joined[:100]
    if plat in ("instagram", "facebook"):
        short = parts[0]
        if len(parts) > 1 and len(short) + 3 + len(parts[1]) <= 38:
            short = f"{short} · {parts[1]}"
        return short[:40]
    return " · ".join(parts[:2])[:80]


def _platform_title_from_caption(platform: str, caption: str) -> Optional[str]:
    p = (platform or "").lower()
    c = (caption or "").strip()
    if not c:
        return None
    first = c.split("\n", 1)[0].strip()
    if p == "youtube":
        if len(first) > 90:
            first = first[:90].rstrip(" .,!?:;") + "..."
        return first
    return None


def _ensure_platform_completeness(
    ranked: Dict[str, Any],
    scene_graph: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Guarantee each requested platform has at least one winner variant.
    If a platform is missing from model output, synthesize from strongest existing winner.
    """
    if not isinstance(ranked, dict):
        return ranked
    pblock = ranked.get("platforms") or {}
    if not isinstance(pblock, dict):
        return ranked
    required = [str(p).lower() for p in (scene_graph.get("platforms") or []) if p]
    if not required:
        return ranked

    donor: Dict[str, Any] = {}
    donor_score = float("-inf")
    for pl, block in pblock.items():
        w = (block or {}).get("winner") or {}
        if not isinstance(w, dict):
            continue
        sc = float(w.get("score") or 0.0)
        if sc > donor_score:
            donor = dict(w)
            donor_score = sc
    if not donor:
        return ranked

    for pl in required:
        block = pblock.get(pl) or {}
        winner = (block or {}).get("winner")
        if winner:
            continue
        caption = str(donor.get("caption") or "").strip()
        title = donor.get("title")
        if not title:
            title = _platform_title_from_caption(pl, caption)
        hashtags = donor.get("hashtags") or []
        synth = {
            "variant_index": donor.get("variant_index") or 999,
            "title": title if title else (None if pl == "tiktok" else ""),
            "caption": caption,
            "hashtags": list(hashtags) if isinstance(hashtags, list) else [],
            "score": round(max(45.0, donor_score - 3.0), 4),
            "rationale": "platform-completion fallback synthesized from best available winner",
            "synthetic": True,
        }
        pblock[pl] = {
            "variants_ranked": [synth],
            "winner": synth,
            "runner_up": None,
            "preflight": {"title_ok": True, "caption_ok": bool(caption), "hashtags_ok": True, "persona_ok": True},
        }
    ranked["platforms"] = pblock
    return ranked


async def _call_openai_m8_json(
    *,
    frames: List[Union[Path, str]],
    prompt: str,
    model: str,
    max_completion_tokens: int = 3500,
    http_timeout_sec: float = 120.0,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Vision + JSON response."""
    if not OPENAI_API_KEY:
        logger.warning(
            "M8 Engine: OPENAI_API_KEY unset — skipping multimodal caption call "
            "(set key on workers; captions fall back to legacy path if enabled)."
        )
        return {}, {"prompt": 0, "completion": 0}

    safe_prompt = prompt.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    content: List[Dict[str, Any]] = [{"type": "text", "text": safe_prompt}]
    for i, frame in enumerate(frames[:6]):
        try:
            with open(frame, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            detail = "high" if i < 2 else "low"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
                }
            )
        except (OSError, PermissionError, TypeError, ValueError) as e:
            logger.warning("M8: could not attach frame %s: %s", frame, e)

    tokens = {"prompt": 0, "completion": 0}
    mc = max(800, min(int(max_completion_tokens or 3500), 16000))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": mc,
        "temperature": 0.55,
        "response_format": {"type": "json_object"},
    }
    resp: Optional[httpx.Response] = None
    try:
        for attempt in range(2):
            async with outbound_slot("openai"):
                async with httpx.AsyncClient(timeout=max(60.0, float(http_timeout_sec or 120.0))) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
            if resp.status_code == 200:
                break
            # Rare intermittent upstream parse failures: retry once with same sanitized payload.
            body = (resp.text or "")[:500]
            bad_json_body = resp.status_code == 400 and "parse the JSON body" in body.lower()
            if bad_json_body and attempt == 0:
                logger.warning("M8 Engine got transient OpenAI 400 parse-body error, retrying once")
                await asyncio.sleep(0.8)
                continue
            break
        if resp is None:
            return {}, tokens
        if resp.status_code != 200:
            logger.error(f"M8 Engine HTTP {resp.status_code}: {resp.text[:400]}")
            return {}, tokens

        data = resp.json()
        usage = data.get("usage") or {}
        tokens = {
            "prompt": int(usage.get("prompt_tokens") or 0),
            "completion": int(usage.get("completion_tokens") or 0),
        }
        raw = data["choices"][0]["message"]["content"]
        if "```" in raw:
            raw = raw.split("```", 1)[-1].split("```", 1)[0]
        try:
            parsed = json.loads(raw.strip())
        except json.JSONDecodeError as e:
            logger.warning("M8 Engine JSON parse failed: %s", e)
            return {}, tokens
        return parsed, tokens
    except asyncio.CancelledError:
        raise
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as e:
        logger.warning("M8 Engine OpenAI request failed: %s", e)
        return {}, tokens


def apply_selection_to_context(
    ctx: JobContext,
    selection: Dict[str, Any],
    *,
    generate_hashtags: bool,
    hashtag_count: int,
    blocked_tags: List[str],
    always_tags: List[str],
    base_tags: List[str],
    category: str = "general",
    strategy: Optional[Dict[str, Any]] = None,
) -> None:
    """Write per-platform fields + legacy ai_* defaults for dashboard/preview."""
    from .caption_stage import _finalise_hashtags, strip_meta_hashtags  # late import avoids circular init

    platforms = _effective_m8_platforms(ctx)
    pdata = (selection.get("platforms") or {}) if isinstance(selection, dict) else {}

    ctx.m8_engine_output = selection
    ctx.m8_platform_titles = {}
    ctx.m8_platform_captions = {}
    ctx.m8_platform_hashtags = {}

    mat = selection.get("caption_evidence_matrix") if isinstance(selection, dict) else None
    if isinstance(mat, dict) and mat.get("cells"):
        ctx.m8_caption_evidence_matrix = mat
    else:
        ctx.m8_caption_evidence_matrix = {}

    for pl in platforms:
        block = pdata.get(pl) or {}
        w = (block.get("winner") or {}) if isinstance(block, dict) else {}
        if not w:
            continue
        ptarget = ((strategy or {}).get("outputs") or {}).get("platform_targets", {}).get(pl, {})
        pconstraints = ptarget.get("constraints") or {}
        policy_hashtag_count = int(pconstraints.get("hashtag_count") or hashtag_count or 0)
        policy_hashtag_count = max(0, min(policy_hashtag_count, hashtag_count or policy_hashtag_count))
        t = w.get("title")
        if t is not None and str(t).strip():
            ctx.m8_platform_titles[pl] = str(t).strip()[:120]
        c = w.get("caption")
        if c is not None and str(c).strip():
            ctx.m8_platform_captions[pl] = strip_stray_hashtag_json_blob(str(c).strip())[:2200]
        raw_tags = w.get("hashtags") or []
        if generate_hashtags and raw_tags:
            ac = getattr(ctx, "audio_context", None) or {}
            extras = list(ac.get("suggested_keywords") or [])[:16]
            # Retrieval-like constrained pool: category seeds + audio keywords + existing tags.
            retrieval_pool = list(base_tags or []) + list(always_tags or []) + extras
            retrieval_slugs = {
                str(x).strip().lstrip("#").lower().replace(" ", "")
                for x in retrieval_pool
                if str(x).strip()
            }
            picked_raw = [str(x).strip() for x in (raw_tags if isinstance(raw_tags, list) else []) if str(x).strip()]
            constrained_raw: List[str] = []
            unknown_added = 0
            for tg in picked_raw:
                slug = tg.lstrip("#").lower().replace(" ", "")
                if slug in retrieval_slugs:
                    constrained_raw.append(tg)
                    continue
                if unknown_added < 1:
                    constrained_raw.append(tg)
                    unknown_added += 1
            cleaned = strip_meta_hashtags(
                constrained_raw,
                policy_hashtag_count or hashtag_count,
                category=category,
                extra_seeds=extras,
            )
            fin = _finalise_hashtags(
                ai_tags=cleaned,
                base_tags=list(base_tags or []) + list(always_tags or []),
                blocked=blocked_tags,
                max_total=policy_hashtag_count or hashtag_count,
            )
            ctx.m8_platform_hashtags[pl] = fin

    # Platform backfill: if one platform comes back empty from model JSON, reuse
    # strongest available winner so publish stage still has coherent metadata.
    fallback_caption = ""
    fallback_title = ""
    fallback_tags: List[str] = []
    for pref in ("youtube", "tiktok", "instagram", "facebook"):
        if not fallback_caption and (ctx.m8_platform_captions or {}).get(pref):
            fallback_caption = str(ctx.m8_platform_captions[pref]).strip()
        if not fallback_title and (ctx.m8_platform_titles or {}).get(pref):
            fallback_title = str(ctx.m8_platform_titles[pref]).strip()
        if not fallback_tags and (ctx.m8_platform_hashtags or {}).get(pref):
            fallback_tags = list(ctx.m8_platform_hashtags[pref] or [])
    for pl in platforms:
        if fallback_caption and pl not in ctx.m8_platform_captions:
            ctx.m8_platform_captions[pl] = strip_stray_hashtag_json_blob(fallback_caption)[:2200]
        if fallback_title and pl not in ctx.m8_platform_titles and pl == "youtube":
            ctx.m8_platform_titles[pl] = fallback_title[:120]
        if fallback_tags and pl not in ctx.m8_platform_hashtags:
            ctx.m8_platform_hashtags[pl] = list(fallback_tags[: max(1, hashtag_count)])

    # ── Per-platform title variance enforcement ─────────────────────────
    # If two platform titles share more than 70% token overlap, rebuild the
    # lower-priority one from a different evidence cluster. Priority order:
    # youtube > instagram > facebook (tiktok title is always null by contract).
    scene_for_variance = getattr(ctx, "m8_scene_graph", None) or selection.get("scene_graph") or {}
    priority = ["youtube", "instagram", "facebook"]
    used_clusters: List[str] = []
    cluster_cycle = ["geo", "speed", "trill", "visual", "music"]
    for i, pl_hi in enumerate(priority):
        title_hi = (ctx.m8_platform_titles or {}).get(pl_hi)
        if not title_hi:
            continue
        toks_hi = set(_title_tokens(title_hi))
        if not toks_hi:
            continue
        for pl_lo in priority[i + 1:]:
            title_lo = (ctx.m8_platform_titles or {}).get(pl_lo)
            if not title_lo:
                continue
            toks_lo = set(_title_tokens(title_lo))
            if not toks_lo:
                continue
            overlap = len(toks_hi & toks_lo) / max(1, len(toks_hi | toks_lo))
            if overlap > 0.70:
                next_cluster = next(
                    (c for c in cluster_cycle if c not in used_clusters), cluster_cycle[0]
                )
                rebuilt = _deterministic_evidence_title(
                    scene_for_variance, platform=pl_lo, preferred_cluster=next_cluster
                )
                if rebuilt and rebuilt.lower() != str(title_lo).lower():
                    ok_t, _ = _validate_title(rebuilt, scene_for_variance, platform=pl_lo)
                    if ok_t:
                        ctx.m8_platform_titles[pl_lo] = rebuilt[:120]
                        used_clusters.append(next_cluster)

    # Legacy single fields — pick defaults for UI / non-platform consumers
    if "youtube" in ctx.m8_platform_titles:
        ctx.ai_title = ctx.m8_platform_titles["youtube"]
    elif ctx.m8_platform_titles:
        ctx.ai_title = next(iter(ctx.m8_platform_titles.values()))

    for pref in ("tiktok", "instagram", "facebook", "youtube"):
        if pref in ctx.m8_platform_captions:
            ctx.ai_caption = strip_stray_hashtag_json_blob(str(ctx.m8_platform_captions[pref]).strip())
            break
    if ctx.m8_platform_hashtags:
        # Prefer tiktok tags as global AI list if present
        if "tiktok" in ctx.m8_platform_hashtags:
            ctx.ai_hashtags = ctx.m8_platform_hashtags["tiktok"]
        else:
            ctx.ai_hashtags = next(iter(ctx.m8_platform_hashtags.values()))


async def run_m8_caption_engine(
    ctx: JobContext,
    *,
    frames: List[Union[Path, str]],
    category: str,
    caption_style: str,
    caption_tone: str,
    caption_voice: str,
    hashtag_style: str,
    hashtag_count: int,
    generate_title: bool,
    generate_caption: bool,
    generate_hashtags: bool,
    model: str,
    blocked_tags: List[str],
    always_tags: List[str],
    base_tags: List[str],
    db_pool: Any = None,
    strategy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full M8 caption pass: scene graph → OpenAI JSON → rank → ctx fields.
    Returns metadata dict for logging/cost.
    """
    scene = build_scene_graph(ctx, category)
    ctx.m8_scene_graph = scene
    _trace_m8(ctx, str(ctx.upload_id), "scene_graph", {
        "category": category,
        "platforms": scene.get("platforms") or [],
        "vision_labels": len(((scene.get("vision") or {}).get("labels") or [])),
        "ocr_chars": len(str((scene.get("vision") or {}).get("ocr") or "")),
        "transcript_chars": len(str((scene.get("transcript") or {}).get("text") or "")),
        "vi_objects": len(((scene.get("video_intelligence") or {}).get("object_tracks") or [])),
    })

    historical: Dict[str, Any] = {}
    if db_pool and ctx.user_id:
        try:
            historical = await fetch_historical_signals(
                db_pool,
                str(ctx.user_id),
                category,
                [str(p).lower() for p in (ctx.platforms or [])],
            )
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError, TypeError, ValueError) as e:
            logger.debug("M8 historical signals skipped: %s", e)

    include_mat = m8_evidence_matrix_enabled(ctx.user_settings or {})
    extra_strategy_block = ""
    if db_pool and ctx.user_id:
        try:
            from stages.m8_strategy_context import build_m8_strategy_context

            extra_strategy_block = await build_m8_strategy_context(
                db_pool, str(ctx.user_id), ctx
            )
        except Exception as e:
            logger.debug("M8 extra strategy context skipped: %s", e)
    prompt = _build_m8_prompt(
        ctx,
        scene,
        category,
        caption_style,
        caption_tone,
        hashtag_style,
        hashtag_count,
        generate_title,
        generate_caption,
        generate_hashtags,
        historical=historical,
        strategy=strategy,
        include_evidence_matrix=include_mat,
        caption_voice_ui=str(caption_voice or "default").lower(),
        extra_strategy_block=extra_strategy_block,
    )
    _trace_m8(ctx, str(ctx.upload_id), "prompt", {
        "model": model,
        "prompt_chars": len(prompt),
        "prompt_preview": prompt,
        "frame_count": len(frames),
        "include_matrix": bool(include_mat),
    })

    n_matrix = len(_evidence_matrix_cell_specs(caption_style, caption_tone, caption_voice))
    max_compl = 7800 if include_mat else 3500
    http_to = 200.0 if include_mat else 120.0
    parsed, tokens = await _call_openai_m8_json(
        frames=frames,
        prompt=prompt,
        model=model,
        max_completion_tokens=max_compl,
        http_timeout_sec=http_to,
    )
    if not parsed:
        _trace_m8(ctx, str(ctx.upload_id), "result", {
            "ok": False,
            "error": "empty_or_failed",
            "tokens": tokens,
        })
        return {"ok": False, "tokens": tokens, "error": "empty_or_failed"}

    matrix_san: Optional[Dict[str, Any]] = None
    if include_mat:
        matrix_san = _sanitize_evidence_matrix(
            parsed.get("caption_evidence_matrix") if isinstance(parsed, dict) else None,
            n_matrix,
        )
        if not matrix_san:
            logger.warning(
                "M8 caption_evidence_matrix missing or empty (expected ~%s cells); continuing with platforms only",
                n_matrix,
            )

    ranked = rank_and_select(parsed, scene, historical, strategy=strategy, ctx=ctx)
    ranked = _ensure_platform_completeness(ranked, scene)
    ranked["scene_graph"] = scene
    ranked["historical_signals"] = historical
    ranked["strategy_priors"] = (historical or {}).get("__strategy_priors__", {})
    if matrix_san:
        ranked["caption_evidence_matrix"] = matrix_san

    apply_selection_to_context(
        ctx,
        ranked,
        generate_hashtags=generate_hashtags,
        hashtag_count=hashtag_count,
        blocked_tags=blocked_tags,
        always_tags=always_tags,
        base_tags=base_tags,
        category=category,
        strategy=strategy,
    )

    ctx.m8_engine_meta = {
        "version": M8_ENGINE_VERSION,
        "family_slug": M8_ENGINE_SLUG,
        "ai_slug": M8_ENGINE_AI_SLUG,
        "ai_display": M8_ENGINE_AI_DISPLAY,
        "mlai_slug": M8_ENGINE_AI_SLUG,
        "mlai_display": M8_ENGINE_AI_DISPLAY,
        "model": model,
        "tokens": tokens,
        "strategy_version": ((strategy or {}).get("version") or ""),
        "ml_strategy_priors": (historical or {}).get("__strategy_priors__", {}),
        "caption_evidence_matrix": bool(matrix_san),
        "caption_evidence_matrix_cells": len((matrix_san or {}).get("cells") or []) if matrix_san else 0,
    }

    try:
        ctx.output_artifacts["m8_engine_json"] = json.dumps(ranked)[:490_000]
    except (TypeError, ValueError, OverflowError) as e:
        logger.debug("m8_engine: could not serialize m8_engine_json to artifacts: %s", e)

    if generate_caption and not (ctx.m8_platform_captions or {}).keys():
        _trace_m8(ctx, str(ctx.upload_id), "result", {
            "ok": False,
            "error": "m8_no_captions",
            "tokens": tokens,
            "ranked_platforms": list((ranked.get("platforms") or {}).keys()) if isinstance(ranked, dict) else [],
        })
        return {"ok": False, "tokens": tokens, "error": "m8_no_captions", "selection": ranked}

    _trace_m8(ctx, str(ctx.upload_id), "result", {
        "ok": True,
        "tokens": tokens,
        "ranked_platforms": list((ranked.get("platforms") or {}).keys()) if isinstance(ranked, dict) else [],
        "must_use_count": len(ranked.get("must_use") or []) if isinstance(ranked, dict) else 0,
    })
    return {"ok": True, "tokens": tokens, "selection": ranked}


async def fetch_historical_signals(
    db_pool: Any,
    user_id: str,
    category: str,
    platforms: List[str],
) -> Dict[str, Any]:
    """
    Per-platform engagement priors (rolling, from uploads.views/likes/... synced by worker)
    plus caption pattern snippets from upload_caption_memory for prompt conditioning.
    """
    try:
        from .db import fetch_m8_historical_signals

        return await fetch_m8_historical_signals(db_pool, user_id, category, platforms)
    except (ImportError, asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError, TypeError, ValueError) as e:
        logger.debug("fetch_historical_signals: %s", e)
        return {"__pattern_corpus__": [], "__meta__": {"ok": False}}
