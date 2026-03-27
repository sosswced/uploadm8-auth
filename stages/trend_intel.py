"""
Trend intelligence for thumbnail/caption briefs — optional SerpAPI + YouTube Data API.

Fetches lightweight signals about what titles/visual framing are common in a niche
so the thumbnail brief can align (or deliberately contrast) with category trends.

Env:
  TREND_INTEL_ENABLED       (default false)
  SERPAPI_API_KEY           — SerpAPI (engine=youtube or google)
  YOUTUBE_DATA_API_KEY      — YouTube Data API v3 search.list (public search)
  TREND_INTEL_MAX_RESULTS   (default 8)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from .context import JobContext
from .outbound_rl import outbound_slot

logger = logging.getLogger("uploadm8-worker.trend_intel")

TREND_INTEL_ENABLED = os.environ.get("TREND_INTEL_ENABLED", "false").lower() == "true"
SERPAPI_API_KEY = (os.environ.get("SERPAPI_API_KEY") or "").strip()
YOUTUBE_DATA_API_KEY = (os.environ.get("YOUTUBE_DATA_API_KEY") or "").strip()
TREND_INTEL_MAX = max(3, min(25, int(os.environ.get("TREND_INTEL_MAX_RESULTS", "8") or 8)))


def _category_query(category: str, title_hint: str) -> str:
    c = (category or "general").replace("_", " ").strip()
    t = (title_hint or "").strip()
    base = f"{c} shorts viral"
    if t and len(t) > 3:
        return f"{t[:60]} {c}"
    return base


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
    except Exception as e:
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
    except Exception as e:
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


async def fetch_trend_intel(ctx: JobContext) -> None:
    """
    Populate ctx.trend_intel_context with {summary, sources, rows} when enabled + keys.
    Safe no-op when disabled or on failure.
    """
    if not TREND_INTEL_ENABLED:
        return
    if not SERPAPI_API_KEY and not YOUTUBE_DATA_API_KEY:
        logger.debug("[trend_intel] skipped — no SERPAPI_API_KEY or YOUTUBE_DATA_API_KEY")
        return

    cat = getattr(ctx, "thumbnail_category", None) or ctx.get_canonical_category() or "general"
    q = _category_query(str(cat), ctx.get_effective_title() or "")

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
        "category": str(cat),
        "source": source or "none",
        "summary": summary,
        "rows": rows[:TREND_INTEL_MAX],
    }
    if summary:
        logger.info("[trend_intel] ok source=%s rows=%d", source, len(rows))
