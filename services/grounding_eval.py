"""
Deterministic grounding score — caption/title vs evidence pool.

Used after hydration_enforcer so admin AI trace and eval harness can measure
factual overlap without an LLM judge.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set


def _norm_tokens(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower()) if t}


def _pool_clue_strings(pool: Any) -> List[str]:
    clues: List[str] = []

    def add(v: Any) -> None:
        s = str(v or "").strip()
        if s and len(s) >= 2:
            clues.append(s)

    for attr in (
        "city",
        "state",
        "road",
        "gazetteer_place",
        "protected_area",
        "driver_name",
        "music_artist",
        "music_title",
        "transcript_phrase",
        "video_understanding_phrase",
        "video_summary_phrase",
    ):
        add(getattr(pool, attr, None))

    for attr in (
        "vision_landmarks",
        "vision_logos",
        "vision_highways",
        "vision_ocr_tokens",
        "transcript_nouns",
        "transcript_topics",
        "video_labels",
    ):
        for item in getattr(pool, attr, None) or []:
            add(item)

    for attr in ("place_beaches", "place_monuments", "place_stadiums", "sports_teams", "license_plates"):
        for item in getattr(pool, attr, None) or []:
            add(item)

    ents = getattr(pool, "transcript_entities", None) or {}
    if isinstance(ents, dict):
        for vals in ents.values():
            for item in vals or []:
                add(item)

    mph = float(getattr(pool, "max_speed_mph", 0) or 0)
    if mph >= 5:
        add(f"{int(round(mph))} mph")
        add(f"{int(round(mph))}mph")

    return clues


def _text_from_ctx(ctx: Any) -> str:
    chunks: List[str] = []
    for attr in ("ai_title", "ai_caption"):
        chunks.append(str(getattr(ctx, attr, "") or ""))
    mt = getattr(ctx, "m8_platform_titles", None) or {}
    mc = getattr(ctx, "m8_platform_captions", None) or {}
    if isinstance(mt, dict):
        chunks.extend(str(v) for v in mt.values())
    if isinstance(mc, dict):
        chunks.extend(str(v) for v in mc.values())
    tags = getattr(ctx, "ai_hashtags", None) or []
    if isinstance(tags, list):
        chunks.extend(str(t) for t in tags)
    return " ".join(chunks)


def compute_grounding_score(
    *,
    text: str,
    pool: Any,
    evidence_present: bool = True,
) -> Dict[str, Any]:
    """
    Return grounding metrics.

    ``grounding_score`` is 0..1 = clue_hits / max(clue_count, 1) when evidence
    exists; 0.0 when evidence exists but zero hits; 1.0 when no evidence (N/A
    pass — nothing to ground against).
    """
    clues = _pool_clue_strings(pool)
    blob = (text or "").lower()
    hits: List[str] = []
    misses: List[str] = []
    for c in clues:
        c_low = c.lower()
        # Require either full phrase or all significant tokens present.
        tokens = _norm_tokens(c_low)
        if len(c_low) >= 4 and c_low in blob:
            hits.append(c)
        elif tokens and tokens.issubset(_norm_tokens(blob)):
            hits.append(c)
        else:
            misses.append(c)

    clue_count = len(clues)
    hit_count = len(hits)
    if not evidence_present or clue_count == 0:
        score = 1.0 if not evidence_present else 0.0
        status = "no_evidence" if not evidence_present else "empty_clues"
    else:
        score = round(hit_count / float(clue_count), 4)
        status = "scored"

    lane_flags = {
        "geo": bool(getattr(pool, "city", None) or getattr(pool, "gazetteer_place", None)),
        "landmarks": bool(getattr(pool, "vision_landmarks", None)),
        "transcript": bool(getattr(pool, "transcript_phrase", None) or getattr(pool, "transcript_nouns", None)),
        "logos": bool(getattr(pool, "vision_logos", None)),
        "ocr": bool(getattr(pool, "vision_ocr_tokens", None) or getattr(pool, "license_plates", None)),
        "teams": bool(getattr(pool, "sports_teams", None)),
        "scene": bool(
            getattr(pool, "video_understanding_phrase", None) or getattr(pool, "video_summary_phrase", None)
        ),
    }
    evidence_lane_count = sum(1 for v in lane_flags.values() if v)

    return {
        "version": 1,
        "status": status,
        "grounding_score": float(score),
        "clue_count": clue_count,
        "hit_count": hit_count,
        "hits": hits[:16],
        "misses": misses[:16],
        "evidence_lane_count": evidence_lane_count,
        "lanes": lane_flags,
    }


def score_ctx_grounding(ctx: Any, pool: Any, *, evidence_present: bool) -> Dict[str, Any]:
    return compute_grounding_score(
        text=_text_from_ctx(ctx),
        pool=pool,
        evidence_present=evidence_present,
    )


__all__ = ["compute_grounding_score", "score_ctx_grounding"]
