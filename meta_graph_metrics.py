"""
Meta Graph API helpers when advanced permissions are missing or limited.

- Facebook: ``pages_read_engagement`` often allows reading the Page ``feed`` and
  summing engagement on Reel posts even when ``/{page-id}/videos`` fails (that
  edge commonly expects ``pages_read_user_content`` for listing).
- Instagram: ``/{ig-user-id}/media`` requires ``instagram_basic``; without it we
  return a degraded live payload so account rollups still include other platforms.
- Per-upload: when Insights fail, fall back to media/video object fields that work
  with Page token + basic product access.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import httpx

GRAPH = "https://graph.facebook.com/v21.0"


def _ig_views_from_media(md: Dict[str, Any]) -> int:
    mtype = str(md.get("media_type") or "").upper()
    if mtype in ("VIDEO", "REELS"):
        return int(md.get("video_view_count") or md.get("play_count") or 0)
    return 0


async def instagram_per_media_engagement_fallback(
    client: httpx.AsyncClient,
    access_token: str,
    media_id: str,
) -> Optional[Dict[str, Any]]:
    """
    When ``/{media_id}/insights`` is unavailable, use object fields.
    Returns metrics dict for rollup or None.
    """
    if not access_token or not media_id:
        return None
    media_resp = await client.get(
        f"{GRAPH}/{media_id}",
        params={
            "access_token": access_token,
            "fields": "media_type,video_view_count,play_count,like_count,comments_count",
        },
    )
    if media_resp.status_code != 200:
        return None
    md = media_resp.json() or {}
    mtype = str(md.get("media_type") or "").upper()
    if mtype not in ("VIDEO", "REELS"):
        return None
    return {
        "views": _ig_views_from_media(md),
        "likes": int(md.get("like_count") or 0),
        "comments": int(md.get("comments_count") or 0),
        "shares": 0,
    }


async def facebook_per_video_engagement_fallback(
    client: httpx.AsyncClient,
    access_token: str,
    video_id: str,
) -> Optional[Dict[str, Any]]:
    """When video insights are unavailable, use native video fields."""
    if not access_token or not video_id:
        return None
    fb_basic = await client.get(
        f"{GRAPH}/{video_id}",
        params={
            "access_token": access_token,
            "fields": "views,reactions.summary(true),comments.summary(true),shares",
        },
    )
    if fb_basic.status_code != 200:
        return None
    bd = fb_basic.json() or {}
    return {
        "views": int(bd.get("views") or 0),
        "likes": int(((bd.get("reactions") or {}).get("summary") or {}).get("total_count") or 0),
        "comments": int(((bd.get("comments") or {}).get("summary") or {}).get("total_count") or 0),
        "shares": int(((bd.get("shares") or {}).get("count")) or 0),
    }


def instagram_account_degraded_live(
    *,
    http_status: int,
    ig_user_id: str,
) -> Dict[str, Any]:
    """Account-level metrics when IG media cannot be listed (e.g. missing instagram_basic)."""
    return {
        "status": "live",
        "analytics_source": "insufficient_scope",
        "analytics_note": (
            "Instagram media list is not available with the current token "
            f"(HTTP {http_status}). Approve instagram_basic (and insights scopes as needed) "
            "and reconnect, or use META_OAUTH_MODE=full."
        ),
        "views": 0,
        "likes": 0,
        "comments": 0,
        "saves": 0,
        "reach": 0,
        "shares": 0,
        "video_count": 0,
        "ig_user_id": ig_user_id,
    }


async def facebook_page_feed_reel_engagement_rollups(
    client: httpx.AsyncClient,
    access_token: str,
    page_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Aggregate reactions, comments, and shares from recent Page posts whose
    permalinks look like Reels. Video view counts are not available on this path
    without ``read_insights`` — views stay 0.
    """
    if not access_token or not page_id:
        return None

    feed = await client.get(
        f"{GRAPH}/{page_id}/feed",
        params={
            "access_token": access_token,
            "limit": 35,
            "fields": "id,permalink_url,created_time,reactions.summary(true),comments.summary(true),shares",
        },
    )
    posts: List[Dict[str, Any]] = []
    if feed.status_code == 200:
        posts = feed.json().get("data", []) or []

    if not posts:
        pub = await client.get(
            f"{GRAPH}/{page_id}/published_posts",
            params={
                "access_token": access_token,
                "limit": 35,
                "fields": "id,permalink_url,created_time,reactions.summary(true),comments.summary(true),shares",
            },
        )
        if pub.status_code == 200:
            posts = pub.json().get("data", []) or []

    def _is_reel(p: Dict[str, Any]) -> bool:
        u = str(p.get("permalink_url") or "").lower()
        return "/reel/" in u or "facebook.com/reel" in u

    reel_posts = [p for p in posts if _is_reel(p)]
    if not reel_posts:
        return None

    total_reactions = total_comments = total_shares = 0
    for p in reel_posts[:15]:
        total_reactions += int(((p.get("reactions") or {}).get("summary") or {}).get("total_count") or 0)
        total_comments += int(((p.get("comments") or {}).get("summary") or {}).get("total_count") or 0)
        sh = p.get("shares")
        if isinstance(sh, dict):
            total_shares += int(sh.get("count") or 0)

    followers = 0
    try:
        pg = await client.get(
            f"{GRAPH}/{page_id}",
            params={"access_token": access_token, "fields": "followers_count,fan_count"},
        )
        if pg.status_code == 200:
            pg_data = pg.json()
            followers = int(pg_data.get("followers_count") or pg_data.get("fan_count") or 0)
    except Exception:
        pass

    return {
        "status": "live",
        "analytics_source": "page_feed_pages_read_engagement",
        "analytics_note": (
            "Totals use Page feed engagement for Reels (pages_read_engagement). "
            "Video view counts require read_insights or listing via pages_read_user_content."
        ),
        "views": 0,
        "reactions": total_reactions,
        "comments": total_comments,
        "shares": total_shares,
        "followers": followers,
        "video_count": len(reel_posts),
    }
