"""
Meta Graph API helpers when advanced permissions are missing or limited.

- Facebook: ``pages_read_engagement`` often allows reading the Page ``feed`` and
  summing engagement on Reel posts even when ``/{page-id}/videos`` fails (that
  edge commonly expects ``pages_read_user_content`` for listing).
- Instagram: ``/{ig-user-id}/media`` requires ``instagram_basic``; without it we
  return a degraded live payload so account rollups still include other platforms.
- Per-upload: when Insights fail *or* return likes with views still 0, fall back to
  media/video object fields that work with Page token + basic product access.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Fallback if meta_oauth import is unavailable; prefer META_GRAPH_API_VERSION.
GRAPH = "https://graph.facebook.com/v21.0"

_METRIC_KEYS = ("views", "likes", "comments", "shares", "saved", "reach")
_IG_SHORTCODE_RE = re.compile(
    r"(?:instagram\.com)/(?:reel|p|tv)/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_IG_SHORTCODE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def extract_instagram_shortcode(*candidates: Any) -> Optional[str]:
    """Pull reel/p shortcode from an explicit field or Instagram permalink."""
    for raw in candidates:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        if re.fullmatch(r"[A-Za-z0-9_-]{5,64}", s) and "://" not in s and "/" not in s:
            return s
        m = _IG_SHORTCODE_RE.search(s)
        if m:
            return m.group(1)
    return None


def instagram_shortcode_to_ig_pk(shortcode: str) -> Optional[str]:
    """Decode Instagram web shortcode to the numeric ``ig_id`` (pk) Graph returns on media."""
    sc = str(shortcode or "").strip()
    if not sc or any(c not in _IG_SHORTCODE_ALPHABET for c in sc):
        return None
    n = 0
    for c in sc:
        n = n * 64 + _IG_SHORTCODE_ALPHABET.index(c)
    return str(n) if n > 0 else None


async def resolve_instagram_media_id_by_shortcode(
    client: httpx.AsyncClient,
    access_token: str,
    ig_user_id: str,
    shortcode: str,
    *,
    max_pages: int = 12,
    caption_hint: Optional[str] = None,
) -> Optional[str]:
    """
    Map a public reel shortcode to the Graph media id via ``/{ig-user-id}/media``.

    Matches ``shortcode``, permalink path, or ``ig_id`` (shortcode pk). Optional
    ``caption_hint`` is a last-resort title/caption substring match.
    """
    sc = str(shortcode or "").strip()
    ig = str(ig_user_id or "").strip()
    at = str(access_token or "").strip()
    if not ig or not at:
        return None
    if not sc and not (caption_hint or "").strip():
        return None
    pk = instagram_shortcode_to_ig_pk(sc) if sc else None
    hint = str(caption_hint or "").strip()
    hint_token = ""
    if hint:
        if re.search(r"Represent", hint, re.I):
            hint_token = "Represent"
        else:
            stop = {
                "celebrating",
                "through",
                "before",
                "instagram",
                "shorts",
                "video",
                "watch",
            }
            words = re.findall(r"[A-Za-zÀ-ÿ0-9']{5,}", hint)
            for w in words:
                if w.lower() not in stop:
                    hint_token = w
                    break
            if not hint_token and words:
                hint_token = words[0]
    root = _graph_root()
    after: Optional[str] = None
    caption_fallback: Optional[str] = None
    from services.meta_oauth import meta_graph_slot

    for _ in range(max(1, int(max_pages))):
        params: Dict[str, Any] = {
            "access_token": at,
            "fields": "id,ig_id,permalink,shortcode,caption",
            "limit": 50,
        }
        if after:
            params["after"] = after
        try:
            async with meta_graph_slot():
                resp = await client.get(f"{root}/{ig}/media", params=params)
        except Exception as e:
            logger.warning("IG media list for shortcode resolve failed: %s", e)
            return None
        if resp.status_code == 403 and "request limit" in (resp.text or "").lower():
            logger.warning("IG media list rate-limited during shortcode resolve")
            return None
        if resp.status_code != 200:
            logger.warning(
                "IG media list HTTP %s for shortcode resolve: %s",
                resp.status_code,
                (getattr(resp, "text", None) or "")[:200],
            )
            return None
        payload = resp.json() or {}
        for m in payload.get("data") or []:
            mid = str(m.get("id") or "").strip()
            if not mid:
                continue
            if sc and str(m.get("shortcode") or "").strip() == sc:
                return mid
            perm = str(m.get("permalink") or "")
            if sc and sc in perm:
                return mid
            if pk and str(m.get("ig_id") or "").strip() == pk:
                return mid
            if hint_token and caption_fallback is None:
                cap = str(m.get("caption") or "")
                if hint_token.lower() in cap.lower():
                    caption_fallback = mid
        after = ((payload.get("paging") or {}).get("cursors") or {}).get("after")
        if not after:
            break
    return caption_fallback


async def reconcile_instagram_graph_media_id(
    client: httpx.AsyncClient,
    access_token: str,
    ig_user_id: str,
    *,
    media_id: Optional[str],
    shortcode: Optional[str],
    creation_id: Optional[str] = None,
    caption_hint: Optional[str] = None,
) -> Optional[str]:
    """
    Prefer a listable Graph media id after publish.

    Publish occasionally returns a non-listable id (container-family). Re-resolve via
    shortcode / ig_id / caption against ``/{ig-user-id}/media``.
    """
    published = str(media_id or "").strip() or None
    creation = str(creation_id or "").strip() or None
    sc = extract_instagram_shortcode(shortcode) if shortcode else None
    needs = False
    if not published:
        needs = True
    elif creation and published == creation:
        needs = True
    elif published.startswith("179") and (not creation or creation.startswith("179")):
        # Heuristic: listable IG media ids we see in production are typically 181…;
        # 179… often matches container/creation ids.
        needs = True
    if not needs and published:
        # Still verify object is readable; if not, resolve.
        try:
            from services.meta_oauth import meta_graph_slot

            async with meta_graph_slot():
                probe = await client.get(
                    f"{_graph_root()}/{published}",
                    params={"access_token": access_token, "fields": "id,shortcode"},
                )
            if probe.status_code != 200:
                needs = True
        except Exception:
            needs = True
    if not needs:
        return published
    resolved = await resolve_instagram_media_id_by_shortcode(
        client,
        access_token,
        ig_user_id,
        sc or "",
        caption_hint=caption_hint,
    )
    return resolved or published


def _graph_root() -> str:
    try:
        from services.meta_oauth import META_GRAPH_API_VERSION

        return f"https://graph.facebook.com/{META_GRAPH_API_VERSION}"
    except Exception:
        return GRAPH


def _insight_int(raw: Any) -> int:
    if isinstance(raw, dict):
        return int(sum(int(v or 0) for v in raw.values() if not isinstance(v, dict)))
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _ig_views_from_media(md: Dict[str, Any]) -> int:
    mtype = str(md.get("media_type") or "").upper()
    if mtype in ("VIDEO", "REELS"):
        return max(
            int(md.get("video_view_count") or 0),
            int(md.get("play_count") or 0),
        )
    return 0


def _merge_metric_dicts(a: Dict[str, int], b: Optional[Dict[str, Any]]) -> Dict[str, int]:
    out = {k: int(a.get(k) or 0) for k in _METRIC_KEYS}
    if not b:
        return out
    for k in _METRIC_KEYS:
        out[k] = max(out[k], int(b.get(k) or 0))
    return out


def _parse_ig_insights_metrics(payload: Dict[str, Any]) -> Dict[str, int]:
    s = {k: 0 for k in _METRIC_KEYS}
    for m in payload.get("data", []) or []:
        name = str(m.get("name") or "")
        vals = m.get("values", [])
        val = _insight_int(vals[-1].get("value", 0) if vals else m.get("value", 0))
        if name in ("views", "total_views", "crossposted_views", "plays"):
            s["views"] = max(s["views"], val)
        elif name == "likes":
            s["likes"] += val
        elif name == "comments":
            s["comments"] += val
        elif name == "shares":
            s["shares"] += val
        elif name == "saved":
            s["saved"] += val
        elif name == "reach":
            s["reach"] = max(s["reach"], val)
    return s


def _parse_fb_video_insights(payload: Dict[str, Any]) -> Dict[str, int]:
    s = {k: 0 for k in _METRIC_KEYS}
    rows = payload.get("data") or []
    if not rows and isinstance(payload.get("insights"), dict):
        rows = (payload.get("insights") or {}).get("data") or []
    for m in rows:
        name = str(m.get("name") or "")
        vals = m.get("values", [{}])
        val = _insight_int(vals[-1].get("value", 0) if vals else m.get("value", 0))
        if name in (
            "fb_reels_total_plays",
            "blue_reels_play_count",
            "total_video_views",
        ):
            s["views"] = max(s["views"], val)
        elif name in ("total_video_reactions_by_type_total", "total_video_reactions"):
            s["likes"] += val
        elif name == "total_video_comments":
            s["comments"] += val
        elif name == "total_video_shares":
            s["shares"] += val
    return s


def _ig_platform_url(md: Dict[str, Any]) -> Optional[str]:
    perm = str(md.get("permalink") or "").strip() or None
    if perm:
        return perm
    sc = str(md.get("shortcode") or "").strip()
    if sc:
        return f"https://www.instagram.com/reel/{sc}/"
    return None


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
        f"{_graph_root()}/{media_id}",
        params={
            "access_token": access_token,
            "fields": (
                "media_type,video_view_count,play_count,like_count,"
                "comments_count,permalink,shortcode"
            ),
        },
    )
    if media_resp.status_code != 200:
        return None
    md = media_resp.json() or {}
    mtype = str(md.get("media_type") or "").upper()
    if mtype not in ("VIDEO", "REELS"):
        return None
    out = {
        "views": _ig_views_from_media(md),
        "likes": int(md.get("like_count") or 0),
        "comments": int(md.get("comments_count") or 0),
        "shares": 0,
        "saved": 0,
        "reach": 0,
    }
    url = _ig_platform_url(md)
    if url:
        out["platform_url"] = url
        if md.get("shortcode"):
            out["shortcode"] = str(md.get("shortcode")).strip()
    return out


async def facebook_per_video_engagement_fallback(
    client: httpx.AsyncClient,
    access_token: str,
    video_id: str,
) -> Optional[Dict[str, Any]]:
    """When video insights are unavailable, use native video fields."""
    if not access_token or not video_id:
        return None
    fb_basic = await client.get(
        f"{_graph_root()}/{video_id}",
        params={
            "access_token": access_token,
            # video_views matches Analytics account-poll fallback; views is an alternate object field.
            "fields": (
                "video_views,views,reactions.summary(true),comments.summary(true),"
                "shares,permalink_url"
            ),
        },
    )
    if fb_basic.status_code != 200:
        return None
    bd = fb_basic.json() or {}
    out = {
        "views": max(int(bd.get("video_views") or 0), int(bd.get("views") or 0)),
        "likes": int(((bd.get("reactions") or {}).get("summary") or {}).get("total_count") or 0),
        "comments": int(((bd.get("comments") or {}).get("summary") or {}).get("total_count") or 0),
        "shares": int(((bd.get("shares") or {}).get("count")) or 0),
        "saved": 0,
        "reach": 0,
    }
    perm = str(bd.get("permalink_url") or "").strip()
    if perm:
        out["platform_url"] = perm
    return out


async def fetch_instagram_media_engagement(
    client: httpx.AsyncClient,
    access_token: str,
    media_id: str,
) -> Optional[Dict[str, int]]:
    """
    Approved IG path: insights (``views`` then ``total_views``) + object-field merge.

    Do not bundle deprecated ``plays`` with ``views`` — a 400 drops the payload.
    Insights often return likes while viewership is on ``total_views`` or object fields.
    Also returns ``saved`` / ``reach`` for account-poll totals (not unified views).
    """
    result = await fetch_instagram_media_engagement_result(
        client, access_token, media_id
    )
    return result.get("metrics")


async def fetch_instagram_media_engagement_result(
    client: httpx.AsyncClient,
    access_token: str,
    media_id: str,
) -> Dict[str, Any]:
    """
    Like ``fetch_instagram_media_engagement`` but returns
    ``{"metrics": dict|None, "reason": "ok"|"empty"|"rate_limited"|"not_found"|"error"}``.
    """
    if not access_token or not media_id:
        return {"metrics": None, "reason": "empty"}
    from services.meta_oauth import meta_graph_slot

    s = {k: 0 for k in _METRIC_KEYS}
    platform_url = None
    shortcode = None
    root = _graph_root()
    reason = "empty"
    # Compatible groups only — ``plays`` is deprecated and 400s when mixed with ``views``.
    for metric_str in (
        "views,reach,saved,shares,comments,likes",
        "total_views",
    ):
        try:
            async with meta_graph_slot():
                resp = await client.get(
                    f"{root}/{media_id}/insights",
                    params={"access_token": access_token, "metric": metric_str},
                )
        except Exception as e:
            logger.warning("IG insights request failed media=%s: %s", str(media_id)[:16], e)
            reason = "error"
            continue
        body = (getattr(resp, "text", None) or "")[:320]
        if resp.status_code == 403 and (
            '"code":4' in body or "request limit" in body.lower()
        ):
            logger.warning(
                "IG insights rate-limited media=%s: %s",
                str(media_id)[:16],
                body[:180],
            )
            return {"metrics": None, "reason": "rate_limited"}
        if resp.status_code == 400 and (
            "does not exist" in body.lower()
            or "unsupported get request" in body.lower()
        ):
            logger.warning(
                "IG insights media missing media=%s: %s",
                str(media_id)[:16],
                body[:180],
            )
            # Object gone — skip second metric group + avoid extra Graph noise.
            reason = "not_found"
            break
        if resp.status_code != 200:
            logger.warning(
                "IG insights HTTP %s media=%s: %s",
                resp.status_code,
                str(media_id)[:16],
                body[:240],
            )
            # nonexisting field (insights) → fall through to object fields
            continue
        parsed = _parse_ig_insights_metrics(resp.json() or {})
        s = _merge_metric_dicts(s, parsed)
        reason = "ok"
        if s["views"] > 0:
            break

    if reason != "rate_limited":
        try:
            async with meta_graph_slot():
                fb = await instagram_per_media_engagement_fallback(
                    client, access_token, str(media_id)
                )
        except Exception:
            fb = None
        if fb is None and reason == "not_found":
            return {"metrics": None, "reason": "not_found"}
        s = _merge_metric_dicts(s, fb)
        if fb:
            platform_url = fb.get("platform_url") or platform_url
            shortcode = fb.get("shortcode") or shortcode
            reason = "ok"

    if not (s["views"] or s["likes"] or s["comments"] or s["shares"] or s["saved"]):
        return {"metrics": None, "reason": reason if reason != "ok" else "empty"}
    if platform_url:
        s["platform_url"] = platform_url
    if shortcode:
        s["shortcode"] = shortcode
    return {"metrics": s, "reason": "ok"}


async def fetch_facebook_video_engagement(
    client: httpx.AsyncClient,
    access_token: str,
    video_id: str,
) -> Optional[Dict[str, int]]:
    """
    Approved FB path: Reels ``/video_insights`` plays, then embedded
    ``total_video_views``, then object ``video_views``.
    Unified views = max(reel plays, 3-second views, object views).
    """
    if not access_token or not video_id:
        return None
    from services.meta_oauth import meta_graph_slot

    s = {k: 0 for k in _METRIC_KEYS}
    platform_url = None
    root = _graph_root()

    try:
        async with meta_graph_slot():
            reel_resp = await client.get(
                f"{root}/{video_id}/video_insights",
                params={
                    "access_token": access_token,
                    "metric": "blue_reels_play_count,fb_reels_total_plays,total_video_views",
                    "period": "lifetime",
                },
            )
        if reel_resp.status_code == 200:
            s = _merge_metric_dicts(s, _parse_fb_video_insights(reel_resp.json() or {}))
        else:
            logger.warning(
                "FB video_insights HTTP %s video=%s: %s",
                reel_resp.status_code,
                str(video_id)[:24],
                (getattr(reel_resp, "text", None) or "")[:240],
            )
    except Exception as e:
        logger.warning("FB video_insights request failed video=%s: %s", str(video_id)[:24], e)

    # Embedded insights.metric on the video object fails for many Reels
    # ("nonexisting field insights"). Only try when /video_insights left views empty.
    if s["views"] <= 0:
        try:
            async with meta_graph_slot():
                resp = await client.get(
                    f"{root}/{video_id}",
                    params={
                        "access_token": access_token,
                        "fields": (
                            "insights.metric(total_video_views,total_video_reactions_by_type_total,"
                            "total_video_comments,total_video_shares)"
                        ),
                    },
                )
            if resp.status_code == 200:
                s = _merge_metric_dicts(s, _parse_fb_video_insights(resp.json() or {}))
            elif resp.status_code != 400:
                logger.warning(
                    "FB embedded insights HTTP %s video=%s: %s",
                    resp.status_code,
                    str(video_id)[:24],
                    (getattr(resp, "text", None) or "")[:240],
                )
        except Exception as e:
            logger.warning("FB embedded insights failed video=%s: %s", str(video_id)[:24], e)

    try:
        async with meta_graph_slot():
            fb = await facebook_per_video_engagement_fallback(
                client, access_token, str(video_id)
            )
    except Exception:
        fb = None
    s = _merge_metric_dicts(s, fb)
    if fb and fb.get("platform_url"):
        platform_url = fb["platform_url"]

    if not (s["views"] or s["likes"] or s["comments"] or s["shares"]):
        return None
    if platform_url:
        s["platform_url"] = platform_url
    return s


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
            f"(HTTP {http_status}). Reconnect Instagram in Connected Accounts "
            "so Meta can grant instagram_basic and insights permissions."
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
        f"{_graph_root()}/{page_id}/feed",
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
            f"{_graph_root()}/{page_id}/published_posts",
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
            f"{_graph_root()}/{page_id}",
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
