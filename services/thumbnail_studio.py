from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger("uploadm8.thumbnail_studio")
from urllib.parse import parse_qs, urlparse

import httpx


_PROTECTED_MARKS = {
    "nfl",
    "nba",
    "disney",
    "marvel",
    "pixar",
    "coca cola",
    "coca-cola",
    "nike",
    "adidas",
    "apple",
    "netflix",
}


def extract_youtube_video_id(url: str) -> str:
    text = (url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except ValueError as e:
        _log.debug("extract_youtube_video_id urlparse: %s", e)
        return ""

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").strip("/")
    if "youtu.be" in host and path:
        return path.split("/")[0]

    if "youtube.com" in host:
        if path == "watch":
            return (parse_qs(parsed.query).get("v") or [""])[0]
        if path.startswith("shorts/"):
            return path.split("/", 1)[1].split("/")[0]
        if path.startswith("embed/"):
            return path.split("/", 1)[1].split("/")[0]
    return ""


async def fetch_youtube_title(url: str) -> str:
    if not (url or "").strip():
        return ""
    endpoint = "https://www.youtube.com/oembed"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(endpoint, params={"url": url, "format": "json"})
            if r.status_code != 200:
                return ""
            data = r.json() if r.content else {}
            return str(data.get("title") or "").strip()[:220]
    except Exception as e:
        _log.debug("fetch_youtube_title failed url=%s: %s", url[:80], e)
        return ""


def estimate_studio_cost(
    *,
    variant_count: int,
    has_persona: bool,
    competitor_gap_mode: bool,
    has_channel_memory: bool,
) -> Tuple[int, int, Dict[str, Any]]:
    n = max(4, min(8, int(variant_count or 4)))
    put = 4 + n
    aic = 10 + (n * 2)
    if has_persona:
        aic += 5
    if competitor_gap_mode:
        aic += 4
    if has_channel_memory:
        aic += 2
    breakdown = {
        "variant_count": n,
        "components": {
            "base_put": 4,
            "variant_put": n,
            "base_aic": 10,
            "variant_aic": n * 2,
            "persona_aic": 5 if has_persona else 0,
            "competitor_gap_aic": 4 if competitor_gap_mode else 0,
            "channel_memory_aic": 2 if has_channel_memory else 0,
        },
    }
    return int(put), int(aic), breakdown


def estimate_pikzels_v2_call_cost(op: str) -> Tuple[int, int, Dict[str, Any]]:
    """
    Per-call token debit for proxied Pikzels v2 operations (PUT + AIC).
    Tuned to be lighter than full Thumbnail Studio recreate jobs.
    """
    o = (op or "").strip().lower()
    table: Dict[str, Tuple[int, int]] = {
        "prompt": (1, 4),
        "recreate": (1, 5),
        "edit": (1, 5),
        "one_click_fix": (1, 5),
        "faceswap": (1, 6),
        "score": (0, 2),
        "titles": (0, 3),
        "persona": (1, 8),
        "style": (1, 8),
    }
    put, aic = table.get(o, (1, 4))
    return int(put), int(aic), {"pikzels_v2_op": o, "put": put, "aic": aic}


def detect_safety_flags(text: str) -> List[str]:
    t = (text or "").lower()
    flags: List[str] = []
    for mark in _PROTECTED_MARKS:
        if mark in t:
            flags.append(f"possible_protected_mark:{mark}")
    celeb_hits = re.findall(r"\b(mrbeast|kim kardashian|elon musk|drake|taylor swift)\b", t, re.I)
    for h in celeb_hits:
        flags.append(f"celebrity_reference:{str(h).lower()}")
    return flags


def pattern_profile(seed_text: str) -> Dict[str, Any]:
    h = hashlib.sha1((seed_text or "thumbnail").encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    emotions = ["shock", "curiosity", "confidence", "urgency", "achievement"]
    text_pos = ["top", "center", "bottom"]
    contrast = ["high", "medium", "very_high"]
    return {
        "face_scale": round(0.32 + (x % 33) / 100.0, 2),
        "text_position": text_pos[(x // 7) % len(text_pos)],
        "contrast_profile": contrast[(x // 13) % len(contrast)],
        "emotion_bias": emotions[(x // 19) % len(emotions)],
    }


def format_library_rows() -> List[Dict[str, Any]]:
    return [
        {"key": "gaming_shock_face", "niche": "gaming", "name": "Shock Face + Big Win", "pattern": "big face, 2-4 word hook, neon edge", "social_proof": "High CTR pattern"},
        {"key": "finance_split", "niche": "finance", "name": "Before/After Split", "pattern": "split chart, money callout, clean text", "social_proof": "Used in 12k thumbnails"},
        {"key": "education_arrow", "niche": "education", "name": "Arrow To Insight", "pattern": "diagram + red arrow + 3-word promise", "social_proof": "High CTR pattern"},
        {"key": "automotive_speed", "niche": "automotive", "name": "Motion + Speed Tag", "pattern": "moving car, speed badge, punch headline", "social_proof": "Used in 8k thumbnails"},
        {"key": "lifestyle_glow", "niche": "lifestyle", "name": "Glow Portrait", "pattern": "subject closeup, glow edge, concise text", "social_proof": "High CTR pattern"},
    ]


def build_variant_suggestions(variant: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    face_scale = float(variant.get("face_scale") or 0.0)
    words = int(variant.get("headline_words") or 0)
    contrast = str(variant.get("contrast_profile") or "")
    if face_scale < 0.35:
        out.append("Face too small for mobile; crop tighter.")
    if words > 5:
        out.append("Text too long; keep hook to 3-4 words.")
    if contrast not in ("high", "very_high"):
        out.append("Low subject/background separation; increase contrast.")
    if not out:
        out.append("Strong baseline; run A/B against a stronger emotion cue.")
    return out


def generate_recreate_variants(
    *,
    youtube_title: str,
    topic: str,
    niche: str,
    closeness: int,
    variant_count: int,
    persona_name: str = "",
    competitor_gap_mode: bool = False,
    channel_memory_hint: str = "",
) -> List[Dict[str, Any]]:
    n = max(4, min(8, int(variant_count or 4)))
    closeness = max(0, min(100, int(closeness or 50)))
    base = (topic or youtube_title or "Untitled Concept").strip()
    niche_clean = (niche or "general").strip().lower()
    voice = (persona_name or "default").strip()

    rows: List[Dict[str, Any]] = []
    for i in range(n):
        seed = f"{base}|{niche_clean}|{closeness}|{i}|{voice}|{channel_memory_hint}"
        profile = pattern_profile(seed)
        vibe = "safe clone" if closeness >= 70 else "fresh remix"
        headline_words = 3 + (i % 3)
        headline = f"{base[:42]}".upper().split()
        headline = " ".join(headline[:headline_words]) or "WATCH THIS"
        ctr_score = 62.0 + (i * 4.1) + (0.08 * closeness)
        if competitor_gap_mode:
            ctr_score += 3.4
        if profile["contrast_profile"] == "very_high":
            ctr_score += 2.1
        variant = {
            "index": i + 1,
            "name": f"{niche_clean.title()} Variant {i + 1}",
            "headline": headline,
            "subhead": f"{vibe} for {niche_clean}",
            "persona": voice or None,
            "render_prompt": (
                f"Create a {niche_clean} thumbnail with {profile['emotion_bias']} emotion, "
                f"{profile['text_position']} text placement, {profile['contrast_profile']} contrast, "
                f"and approximately {profile['face_scale']:.2f} face scale."
            ),
            "ctr_score": round(min(98.0, ctr_score), 2),
            "face_scale": profile["face_scale"],
            "text_position": profile["text_position"],
            "contrast_profile": profile["contrast_profile"],
            "emotion": profile["emotion_bias"],
            "headline_words": headline_words,
            "watermark_preview": True,
            "safety_flags": detect_safety_flags(f"{headline} {base}"),
        }
        variant["suggestions"] = build_variant_suggestions(variant)
        rows.append(variant)
    rows.sort(key=lambda r: float(r.get("ctr_score") or 0), reverse=True)
    return rows
