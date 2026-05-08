"""
Trend intelligence for thumbnail/caption briefs — optional SerpAPI + YouTube Data API.

Fetches lightweight signals about what titles/visual framing are common in a niche
so the thumbnail brief can align (or deliberately contrast) with category trends.

Env:
  TREND_INTEL_DISABLED      — set true to skip all trend calls (ops kill-switch)
  SERPAPI_API_KEY           — SerpAPI (engine=youtube)
  YOUTUBE_DATA_API_KEY      — YouTube Data API v3 search.list (public search)
  TREND_INTEL_MAX_RESULTS   (default 8)

When any key is configured, the caption worker fetches a short title sample so M8
can align hooks with real search language (never copied verbatim).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx

from .context import JobContext
from .outbound_rl import outbound_slot

logger = logging.getLogger("uploadm8-worker.trend_intel")

SERPAPI_API_KEY = (os.environ.get("SERPAPI_API_KEY") or "").strip()
YOUTUBE_DATA_API_KEY = (os.environ.get("YOUTUBE_DATA_API_KEY") or "").strip()
TREND_INTEL_MAX = max(3, min(25, int(os.environ.get("TREND_INTEL_MAX_RESULTS", "8") or 8)))

_QUERY_STOPWORDS = {
    "video", "short", "shorts", "viral", "content", "clip", "upload",
    "scene", "view", "person", "people", "object", "outdoor", "indoor",
    "vehicle", "road", "sky", "cloud", "tree", "plant", "hand", "face",
    "tutorial", "review", "vlog", "makeup", "beauty", "education",
}

_GENERIC_TITLE_HINTS = (
    re.compile(r"\b(?:untitled|new upload|my video|video clip|quick clip|highlights?)\b", re.I),
    re.compile(r"\b(?:exciting moments?|amazing moments?|viral moments?|cool video)\b", re.I),
)


def trend_intel_runtime_available() -> bool:
    """True when keys exist and the worker is not globally opted out."""
    if (os.environ.get("TREND_INTEL_DISABLED") or "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return bool(SERPAPI_API_KEY or YOUTUBE_DATA_API_KEY)


def _trend_query_suffix(category_slug: str) -> str:
    """YouTube search tail — category-aware, not dashcam-only.

    Geo + telemetry often means automotive / road content, but many uploads
    (makeup, grading, reviews, vlogs) have no GPS at all; when they do, we
    still must not force "dashcam" into SerpAPI queries for non-automotive
    categories.
    """
    low = (category_slug or "").lower().replace("_", " ").strip()
    automotiveish = any(
        x in low
        for x in ("automotive", "dash", "dashcam", "driving", "car", "road", "highway", "telemetry")
    )
    if automotiveish:
        return "dashcam shorts"
    # Short vertical niches — "shorts" keeps results in Shorts-friendly title space.
    return "shorts"


def _looks_like_weak_title(title_hint: str) -> bool:
    t = (title_hint or "").strip()
    if len(t) < 18:
        return True
    return any(p.search(t) for p in _GENERIC_TITLE_HINTS)


def _clean_query_term(raw: Any, *, max_words: int = 4) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    s = re.sub(r"[_#]+", " ", s)
    s = re.sub(r"[^A-Za-z0-9&' -]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    words = s.split()[:max_words]
    if not words:
        return ""
    low = " ".join(words).lower()
    if low in _QUERY_STOPWORDS:
        return ""
    if len(low) < 3:
        return ""
    return " ".join(words)


def _query_evidence_terms(ctx: Optional[JobContext], *, category: str, limit: int = 4) -> List[str]:
    """Specific non-geo terms for trend search when title is weak.

    Priority:
      1. Video Intelligence objects/logos/full-video OCR.
      2. Whisper structured topics/entities/key phrase.
      3. Vision landmarks/logos/specific labels/OCR.
      4. ACR music artist/title/genre.

    This is what makes SerpAPI useful for makeup tutorials, color grading,
    reviews, and vlogs that have no .map / OSD / geo: the search query becomes
    "soft glam eyeshadow beauty shorts" instead of just "beauty shorts viral".
    """
    if ctx is None:
        return []

    category_words = {
        w.lower()
        for w in re.findall(r"[A-Za-z0-9]+", (category or "").replace("_", " "))
        if len(w) > 2
    }
    out: List[str] = []
    seen: set[str] = set()

    def push(raw: Any, *, max_words: int = 4) -> None:
        if len(out) >= limit:
            return
        term = _clean_query_term(raw, max_words=max_words)
        if not term:
            return
        low = term.lower()
        # Avoid bloating query with exact category duplicates.
        if low in category_words:
            return
        if low in seen:
            return
        seen.add(low)
        out.append(term)

    vi = getattr(ctx, "video_intelligence", None) or getattr(ctx, "video_intelligence_context", None) or {}
    if isinstance(vi, dict):
        for row in (vi.get("object_tracks") or [])[:6]:
            if isinstance(row, dict):
                push(row.get("description"))
        for row in (vi.get("logos") or [])[:4]:
            if isinstance(row, dict):
                push(row.get("description"), max_words=3)
        for row in (vi.get("on_screen_text") or vi.get("text_detections") or [])[:4]:
            if isinstance(row, dict):
                push(row.get("text"), max_words=5)

    ac = getattr(ctx, "audio_context", None) or {}
    structured = ac.get("transcript_structured") if isinstance(ac, dict) else {}
    if isinstance(structured, dict):
        for topic in (structured.get("topics") or [])[:4]:
            push(topic, max_words=4)
        entities = structured.get("named_entities") or {}
        if isinstance(entities, dict):
            for kind in ("products", "organizations", "places", "people"):
                for ent in (entities.get(kind) or [])[:3]:
                    push(ent, max_words=4)
        push(structured.get("key_phrase"), max_words=5)

    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict):
        for key in ("landmark_names", "logo_names"):
            for val in (vc.get(key) or [])[:4]:
                push(val, max_words=4)
        for label in (vc.get("label_names") or [])[:8]:
            push(label, max_words=3)
        ocr = str(vc.get("ocr_text") or "")
        for noun in re.findall(r"\b[A-Z][A-Za-z0-9&' -]{3,32}\b", ocr)[:4]:
            push(noun, max_words=4)

    if isinstance(ac, dict):
        # ACRCloud/music context: useful for dance, review, vlog, and creator clips.
        if ac.get("music_artist") and ac.get("music_title"):
            push(f"{ac.get('music_artist')} {ac.get('music_title')}", max_words=5)
        else:
            push(ac.get("music_artist"), max_words=4)
            push(ac.get("music_title"), max_words=4)
        push(ac.get("music_genre"), max_words=3)

    return out[:limit]


def _category_query(category: str, title_hint: str, ctx: Optional[JobContext] = None) -> str:
    c = (category or "general").replace("_", " ").strip()
    c_slug = (category or "general").strip()
    t = (title_hint or "").strip()
    suffix = _trend_query_suffix(c_slug)
    geo_bits: List[str] = []
    if ctx is not None:
        try:
            tel = ctx.telemetry or ctx.telemetry_data
            if tel is not None:
                city = getattr(tel, "location_city", None) or ""
                state = getattr(tel, "location_state", None) or ""
                road = getattr(tel, "location_road", None) or ""
                gaz = getattr(tel, "gazetteer_place_name", None) or ""
                for part in (city, state, road, gaz):
                    s = str(part).strip()
                    if s and s not in geo_bits:
                        geo_bits.append(s)
        except (AttributeError, TypeError, ValueError):
            pass
    geo_prefix = " ".join(geo_bits[:3]).strip()
    evidence_prefix = ""
    if _looks_like_weak_title(t):
        evidence_prefix = " ".join(_query_evidence_terms(ctx, category=c, limit=4)).strip()
    if t and len(t) > 3 and not _looks_like_weak_title(t):
        q = f"{geo_prefix} {t[:60]} {c} {suffix}".strip() if geo_prefix else f"{t[:60]} {c} {suffix}"
        return q[:120]
    if evidence_prefix:
        lead = " ".join(x for x in (geo_prefix, evidence_prefix) if x).strip()
        return f"{lead} {c} {suffix}".strip()[:120]
    if geo_prefix:
        return f"{geo_prefix} {c} {suffix}".strip()[:120]
    return f"{c} {suffix} viral"[:120]


async def _serp_youtube_titles(query: str) -> List[Dict[str, Any]]:
    if not SERPAPI_API_KEY:
        return []
    try:
        async with outbound_slot("serpapi"):
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(
                    "https://serpapi.com/search.json",
                    params={
                        "engine": "youtube",
                        "search_query": query,
                        "api_key": SERPAPI_API_KEY,
                    },
                )
        if r.status_code != 200:
            logger.warning("[trend_intel] SerpAPI HTTP %s", r.status_code)
            return []
        data = r.json()
        out: List[Dict[str, Any]] = []
        for row in (data.get("video_results") or [])[:TREND_INTEL_MAX]:
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "title": (row.get("title") or "")[:200],
                    "channel": (row.get("channel") or {}).get("name", "")
                    if isinstance(row.get("channel"), dict)
                    else "",
                    "snippet": (row.get("snippet") or "")[:300],
                }
            )
        return out
    except (
        httpx.RequestError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as e:
        logger.warning("[trend_intel] SerpAPI error: %s", e)
        return []


async def _yt_search_titles(query: str) -> List[Dict[str, Any]]:
    if not YOUTUBE_DATA_API_KEY:
        return []
    try:
        async with outbound_slot("youtube_data"):
            async with httpx.AsyncClient(timeout=25.0) as client:
                r = await client.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params={
                        "part": "snippet",
                        "q": query,
                        "type": "video",
                        "maxResults": TREND_INTEL_MAX,
                        "key": YOUTUBE_DATA_API_KEY,
                    },
                )
        if r.status_code != 200:
            logger.warning("[trend_intel] YouTube search HTTP %s", r.status_code)
            return []
        data = r.json()
        out: List[Dict[str, Any]] = []
        for item in (data.get("items") or []):
            sn = (item or {}).get("snippet") or {}
            if not isinstance(sn, dict):
                continue
            out.append(
                {
                    "title": (sn.get("title") or "")[:200],
                    "channel": (sn.get("channelTitle") or "")[:120],
                    "snippet": "",
                }
            )
        return out
    except (
        httpx.RequestError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as e:
        logger.warning("[trend_intel] YouTube Data API error: %s", e)
        return []


def _summarize(rows: List[Dict[str, Any]], category: str) -> str:
    if not rows:
        return ""
    titles = [r.get("title") or "" for r in rows if r.get("title")]
    if not titles:
        return ""
    # Short heuristic line for the brief (no heavy NLP dependency)
    joined = "; ".join(titles[:6])
    return (
        f"Recent {category} search titles (trend sample): {joined[:900]}. "
        "Use for tone/shape only — do not copy verbatim; stay accurate to this video."
    )


async def fetch_trend_intel(ctx: JobContext, *, category: Optional[str] = None) -> None:
    """
    Populate ctx.trend_intel_context with {summary, sources, rows} when keys exist.
    Safe no-op when disabled or on failure.

    Gated by ``TREND_INTEL_DISABLED`` env and by the caller (caption stage checks
    ``caption_llm`` / billing so we do not spend Serp quota when captions are off).
    """
    if not trend_intel_runtime_available():
        logger.debug("[trend_intel] skipped — disabled or no API keys")
        return

    cat = category or getattr(ctx, "thumbnail_category", None) or ctx.get_canonical_category() or "general"
    q = _category_query(str(cat), ctx.get_effective_title() or "", ctx)
    query_evidence_terms = _query_evidence_terms(ctx, category=str(cat), limit=4)

    rows: List[Dict[str, Any]] = []
    source = ""
    if SERPAPI_API_KEY:
        rows = await _serp_youtube_titles(q)
        source = "serpapi_youtube"
    if not rows and YOUTUBE_DATA_API_KEY:
        rows = await _yt_search_titles(q)
        source = "youtube_data_api"

    summary = _summarize(rows, str(cat))
    ctx.trend_intel_context = {
        "query": q,
        "query_evidence_terms": query_evidence_terms,
        "category": str(cat),
        "source": source or "none",
        "summary": summary,
        "rows": rows[:TREND_INTEL_MAX],
    }
    if summary:
        logger.info("[trend_intel] ok source=%s rows=%d", source, len(rows))
