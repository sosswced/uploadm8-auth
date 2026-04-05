"""
M8 Engine — UploadM8 multimodal caption brain
=============================================
Builds a unified scene graph from audio + vision + telemetry + Twelve Labs,
then generates **five caption/title variants per target platform**, ranks them
with accuracy-first heuristics (and optional future hooks for live platform stats),
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

from .context import (
    JobContext,
    build_fusion_caption_rules,
    build_fusion_summary_text,
    build_multimodal_scene_digest,
    extract_landmark_hints,
)
from .outbound_rl import outbound_slot

logger = logging.getLogger("uploadm8-worker.m8")

M8_ENGINE_VERSION = "1.2.0"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Generic / “AI slop” phrases to penalize in variants (light-touch; expand over time)
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
    r"\bin this (?:raw )?authentic moment\b",
    r"\bchannel(?:ing)? (?:my|your) emotions?\b",
]

def build_scene_graph(ctx: JobContext, category: str) -> Dict[str, Any]:
    """Single structured snapshot used by M8 prompts and ranking."""
    ac = ctx.audio_context or {}
    vc = ctx.vision_context or {}
    vu = ctx.video_understanding or {}
    tel = ctx.telemetry or ctx.telemetry_data

    geo: Dict[str, Any] = {}
    if tel:
        geo = {
            "display": getattr(tel, "location_display", None),
            "road": getattr(tel, "location_road", None),
            "city": getattr(tel, "location_city", None),
            "state": getattr(tel, "location_state", None),
            "country": getattr(tel, "location_country", None),
            "mid_lat": getattr(tel, "mid_lat", None),
            "mid_lon": getattr(tel, "mid_lon", None),
            "max_speed_mph": getattr(tel, "max_speed_mph", None),
            "avg_speed_mph": getattr(tel, "avg_speed_mph", None),
            "total_distance_miles": getattr(tel, "total_distance_miles", None),
        }

    tr = ctx.trill or ctx.trill_score
    trill_d: Dict[str, Any] = {}
    if tr:
        trill_d = {
            "score": getattr(tr, "score", None),
            "bucket": getattr(tr, "bucket", None),
        }

    labels_full = list(vc.get("label_names") or [])
    ocr_full = (vc.get("ocr_text") or "").strip()
    hume_d = (ac.get("hume_emotions") or {}) if isinstance(ac.get("hume_emotions"), dict) else {}

    return {
        "engine_version": M8_ENGINE_VERSION,
        "upload_id": ctx.upload_id,
        "filename": ctx.filename,
        "category": category,
        "platforms": [str(p).lower() for p in (ctx.platforms or [])],
        "transcript": {
            "text": (ctx.ai_transcript or ac.get("transcript") or "")[:12000],
            "role": ac.get("transcript_role") or "",
            "language": ac.get("language") or "",
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
        "fusion_summary": build_fusion_summary_text(ctx)[:4000],
        "multimodal_digest": build_multimodal_scene_digest(ctx, max_chars=10000),
        "vision": {
            "labels": labels_full,
            "label_count": len(labels_full),
            "ocr": ocr_full[:8000],
            "face_count": vc.get("face_count"),
            "has_faces": vc.get("has_faces"),
            "expressive_faces": bool(vc.get("expressive")),
            "landmark_hints": extract_landmark_hints(labels_full, ocr_full),
        },
        "video_understanding": {
            "scene": (vu.get("scene_description") or vu.get("description") or "")[:8000],
            "title_suggestion": (vu.get("title_suggestion") or "")[:200],
        },
        "geo": geo,
        "trill": trill_d,
    }


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
        f"emoji_density={c.get('emoji_density', 'low')}; "
        f"hook_formula={c.get('hook_formula', 'scene_hook')}; "
        f"avoid_phrases=[{banned}]."
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
) -> str:
    fusion = ""
    try:
        fusion = build_fusion_caption_rules(ctx) or ""
    except (AttributeError, TypeError, KeyError, ValueError) as e:
        logger.debug("m8_engine: build_fusion_caption_rules skipped: %s", e)

    platforms = [p for p in scene_graph.get("platforms") or [] if p]
    if not platforms:
        platforms = ["tiktok"]

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
            "NEVER use meta filler: cinematic, caption, viral, content, video, reels, trending, photography, "
            "youtube, tiktok, instagram, follow, like, subscribe."
        )
    else:
        hashtag_rule = "Use empty [] for hashtags in every variant."

    title_rule = (
        "Include a non-empty title for YouTube when generate_title is true. "
        "For TikTok use null title. For IG/FB title may be null or a 2–5 word headline."
        if generate_title
        else "Set title to null for all platforms that do not need titles."
    )

    caption_rule = (
        "Write captions only when generate_caption is true; otherwise use empty string."
        if not generate_caption
        else "Write vivid, specific captions grounded in Scene Graph evidence."
    )

    sg = json.dumps(scene_graph, indent=2)[:24000]

    pattern_block = ""
    hist = historical or {}
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
                "\nPATTERN MEMORY (your past uploads in this category — imitate opening rhythm, line breaks, and specificity; "
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

    return f"""You are M8 Engine — UploadM8's multimodal publishing brain (version {M8_ENGINE_VERSION}).

Your job: produce HIGH-SPECIFICITY, NON-GENERIC titles/captions that clearly come from THIS footage,
not from a template. Avoid vague influencer filler. No false claims.

MULTIMODAL MANDATE: The Scene Graph may be long on purpose. Integrate ALL relevant layers —
transcript, music ID + genre, background/YAMNet sound, Hume emotion (if present), GPS/lat-lon + place names,
telemetry, full vision labels (objects/vehicles/scene), faces, OCR text, landmark hints, and video understanding.
Do NOT write from transcript alone when other fields are populated. Prefer nouns from labels + geo + OCR over generic hype.

SCENE GRAPH (evidence — do not invent beyond it; you may interpret visually grounded details):
{sg}

CATEGORY: {category}
STYLE: {caption_style}  TONE: {caption_tone}
STRATEGY MASTER: style={base_style} tone={base_tone} persona={base_persona}
{_persona_system_prompt(base_persona)}
{_style_prompt(base_style, base_tone)}
{_task_prompt(generate_title, generate_caption, generate_hashtags)}

{fusion}
{pattern_block}
{strategy_priors_block}
PLATFORM RULES:
{chr(10).join(plat_blocks)}

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
- Hashtags must read like real community search terms, not platform metadata.

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
  }}
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
    labels = scene_graph.get("vision", {}).get("labels") or []
    for x in labels[:20]:
        s = str(x).lower().strip()
        if len(s) > 2:
            toks.append(s)
    ocr = (scene_graph.get("vision", {}) or {}).get("ocr") or ""
    for part in re.split(r"\W+", ocr.lower()):
        if len(part) > 3:
            toks.append(part)
    for g in ["location", "road", "display"]:
        v = (scene_graph.get("geo") or {}).get(g)
        if v:
            toks.append(str(v).lower())
    bonus = 0.0
    for tok in toks:
        if tok and tok in blob:
            bonus += 3.0
    return min(bonus, 25.0)


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

    post = _preflight_artifact_checks(platform, repaired, scene_graph, strategy_target)
    return repaired, post


def score_variant(
    platform: str,
    variant: Dict[str, Any],
    scene_graph: Dict[str, Any],
    historical_signals: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    """
    Return (score, rationale). Higher is better.
    historical_signals: optional per-platform priors from DB/analytics (future).
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

    # Future: weight by historical engagement priors
    hist_note = ""
    if historical_signals and platform in historical_signals:
        h = historical_signals[platform]
        # Soft nudge only — keep accuracy-first
        eng = float(h.get("engagement_prior", 0) or 0)
        base += min(10.0, max(-5.0, eng))
        hist_note = " (+historical prior)"

    rationale = (
        f"length/style fit + specificity bonus - generic/attribution penalties{hist_note}"
    )
    return base, rationale


def rank_and_select(
    parsed: Dict[str, Any],
    scene_graph: Dict[str, Any],
    historical_signals: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Pick best variant per platform. Adds ranking_debug with scores.
    """
    out_plat: Dict[str, Any] = {}
    platforms_block = (parsed.get("platforms") or {}) if isinstance(parsed, dict) else {}

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
            sc, why = score_variant(pl, v, scene_graph, historical_signals)
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
        if ranked:
            winner = ranked[0]
            if len(ranked) > 1:
                runner_up = ranked[1]
            # Quality gate: if top result still weak, use best non-weak candidate.
            for cand in ranked:
                if float(cand.get("score") or 0) >= 45.0:
                    winner = cand
                    break
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
        }

    return {
        "m8_version": parsed.get("m8_version") or M8_ENGINE_VERSION,
        "platforms": out_plat,
    }


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
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Vision + JSON response."""
    if not OPENAI_API_KEY:
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
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 3500,
        "temperature": 0.55,
        "response_format": {"type": "json_object"},
    }
    resp: Optional[httpx.Response] = None
    try:
        for attempt in range(2):
            async with outbound_slot("openai"):
                async with httpx.AsyncClient(timeout=120) as client:
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

    platforms = [str(p).lower() for p in (ctx.platforms or [])]
    pdata = (selection.get("platforms") or {}) if isinstance(selection, dict) else {}

    ctx.m8_engine_output = selection
    ctx.m8_platform_titles = {}
    ctx.m8_platform_captions = {}
    ctx.m8_platform_hashtags = {}

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
            ctx.m8_platform_captions[pl] = str(c).strip()[:2200]
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
            ctx.m8_platform_captions[pl] = fallback_caption[:2200]
        if fallback_title and pl not in ctx.m8_platform_titles and pl == "youtube":
            ctx.m8_platform_titles[pl] = fallback_title[:120]
        if fallback_tags and pl not in ctx.m8_platform_hashtags:
            ctx.m8_platform_hashtags[pl] = list(fallback_tags[: max(1, hashtag_count)])

    # Legacy single fields — pick defaults for UI / non-platform consumers
    if "youtube" in ctx.m8_platform_titles:
        ctx.ai_title = ctx.m8_platform_titles["youtube"]
    elif ctx.m8_platform_titles:
        ctx.ai_title = next(iter(ctx.m8_platform_titles.values()))

    for pref in ("tiktok", "instagram", "facebook", "youtube"):
        if pref in ctx.m8_platform_captions:
            ctx.ai_caption = ctx.m8_platform_captions[pref]
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
    )

    parsed, tokens = await _call_openai_m8_json(frames=frames, prompt=prompt, model=model)
    if not parsed:
        return {"ok": False, "tokens": tokens, "error": "empty_or_failed"}

    ranked = rank_and_select(parsed, scene, historical, strategy=strategy)
    ranked = _ensure_platform_completeness(ranked, scene)
    ranked["scene_graph"] = scene
    ranked["historical_signals"] = historical
    ranked["strategy_priors"] = (historical or {}).get("__strategy_priors__", {})

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
        "model": model,
        "tokens": tokens,
        "strategy_version": ((strategy or {}).get("version") or ""),
        "ml_strategy_priors": (historical or {}).get("__strategy_priors__", {}),
    }

    try:
        ctx.output_artifacts["m8_engine_json"] = json.dumps(ranked)[:490_000]
    except (TypeError, ValueError, OverflowError) as e:
        logger.debug("m8_engine: could not serialize m8_engine_json to artifacts: %s", e)

    if generate_caption and not (ctx.m8_platform_captions or {}).keys():
        return {"ok": False, "tokens": tokens, "error": "m8_no_captions", "selection": ranked}

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
