"""
Unified Content Catalog Sync Service
=====================================
Discovers every video on every connected account across TikTok, YouTube,
Instagram and Facebook and stores them as canonical rows in
`platform_content_items`.

Key design rules
----------------
* One row per (user_id, platform, account_id, platform_video_id) — enforced by DB UNIQUE.
* TikTok: no short/long or duration-based ``content_kind``; list pagination covers every video the API returns.
* source = 'external'  → found by catalog scan only
* source = 'uploadm8'  → published through the UploadM8 pipeline
* source = 'linked'    → was 'external', later matched to an UploadM8 upload
* Pagination cursors are persisted in `platform_content_sync_state` so each
  cron run only fetches *new* pages (incremental).
* Hard limits per run prevent runaway API quota usage.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import httpx

from stages.publish_stage import decrypt_token

from services.catalog_identity import dump_facebook_dual_cursor, parse_facebook_dual_cursor
from services.canonical_engagement import ROLLUP_VERSION as CANONICAL_ENGAGEMENT_ROLLUP_VERSION
from services import metric_definitions as metric_definitions_svc

logger = logging.getLogger("uploadm8.catalog_sync")

# YouTube Shorts: no native API flag; hashtag + duration cap (Shorts can be up to ~3 min).
_YOUTUBE_SHORTS_MAX_SEC = 180

# ── Tuning constants ────────────────────────────────────────────────────────
_MAX_PAGES_PER_TOKEN = 10       # max API pages to fetch per token per run (non-YouTube)
# YouTube: uploads playlist can exceed 500 videos; env override for large channels.
_YOUTUBE_CATALOG_MAX_PAGES = max(1, min(50, int(os.environ.get("YOUTUBE_CATALOG_MAX_PAGES", "25") or 25)))
_PAGE_SIZE_TIKTOK    = 20       # TikTok max_count
_PAGE_SIZE_YOUTUBE   = 50       # YouTube playlistItems maxResults
_PAGE_SIZE_META      = 25       # Instagram / Facebook page size
_SEMAPHORE_LIMIT     = 6        # max concurrent token fetches
_RETRY_BACKOFF_BASE  = 0.5      # seconds


def _int(v: Any) -> int:
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _merge_catalog_videos_by_id(a: List[Dict], b: List[Dict]) -> List[Dict]:
    """Stable merge; first list wins metrics for duplicate platform_video_id."""
    seen: set = set()
    out: List[Dict] = []
    for block in (a, b):
        for v in block:
            pid = str(v.get("platform_video_id") or "")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            out.append(v)
    return out


def _youtube_content_kind_and_rule(
    vsnip: Dict[str, Any],
    snip: Dict[str, Any],
    dur_sec: Optional[int],
) -> Tuple[str, str]:
    """
    Classify YouTube uploads as short vs long. Shorts feed uses #shorts and/or short duration;
    API does not expose an official is_shorts flag.
    """
    title = (vsnip.get("title") or snip.get("title") or "")
    desc = (vsnip.get("description") or "")[:8000]
    blob = f"{title}\n{desc}".lower().replace(" ", "").replace("\n", "")
    if "#shorts" in blob:
        return "short", "hashtag"
    if dur_sec is not None and dur_sec <= _YOUTUBE_SHORTS_MAX_SEC:
        return "short", "duration_cap"
    if dur_sec is not None:
        return "long", "duration"
    return "unknown", "unknown"


# ── Low-level retry helper ──────────────────────────────────────────────────
async def _with_retry(coro_factory, retries: int = 3):
    err = None
    for attempt in range(retries):
        try:
            return await coro_factory()
        except Exception as e:
            err = e
            if attempt < retries - 1:
                await asyncio.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.3))
    raise err or RuntimeError("retry_failed")


# ── DB helpers ──────────────────────────────────────────────────────────────
async def _upsert_content_item(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    platform_token_id: str,
    platform: str,
    account_id: str,
    platform_video_id: str,
    source: str = "external",
    content_kind: Optional[str] = None,
    title: Optional[str] = None,
    published_at: Optional[datetime] = None,
    thumbnail_url: Optional[str] = None,
    platform_url: Optional[str] = None,
    duration_seconds: Optional[int] = None,
    views: int = 0,
    likes: int = 0,
    comments: int = 0,
    shares: int = 0,
    visibility: Optional[str] = None,
    presence: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> str:
    """Upsert one video row; returns the row id."""
    row = await conn.fetchrow(
        """
        INSERT INTO platform_content_items
            (user_id, platform_token_id, platform, account_id, platform_video_id,
             source, content_kind, title, published_at, thumbnail_url, platform_url,
             duration_seconds, views, likes, comments, shares,
             visibility, presence,
             metrics_synced_at, extra, updated_at)
        VALUES
            ($1,$2,$3,$4,$5, $6,$7,$8,$9,$10,$11, $12,$13,$14,$15,$16, $17,$18, NOW(),$19,NOW())
        ON CONFLICT (user_id, platform, account_id, platform_video_id) DO UPDATE SET
            platform_token_id  = EXCLUDED.platform_token_id,
            content_kind       = COALESCE(EXCLUDED.content_kind, platform_content_items.content_kind),
            title              = COALESCE(EXCLUDED.title, platform_content_items.title),
            published_at       = COALESCE(EXCLUDED.published_at, platform_content_items.published_at),
            thumbnail_url      = COALESCE(EXCLUDED.thumbnail_url, platform_content_items.thumbnail_url),
            platform_url       = COALESCE(EXCLUDED.platform_url, platform_content_items.platform_url),
            duration_seconds   = COALESCE(EXCLUDED.duration_seconds, platform_content_items.duration_seconds),
            views              = GREATEST(EXCLUDED.views, platform_content_items.views),
            likes              = GREATEST(EXCLUDED.likes, platform_content_items.likes),
            comments           = GREATEST(EXCLUDED.comments, platform_content_items.comments),
            shares             = GREATEST(EXCLUDED.shares, platform_content_items.shares),
            visibility         = COALESCE(EXCLUDED.visibility, platform_content_items.visibility),
            presence           = COALESCE(EXCLUDED.presence, platform_content_items.presence),
            metrics_synced_at  = NOW(),
            extra              = platform_content_items.extra || EXCLUDED.extra,
            updated_at         = NOW()
        RETURNING id
        """,
        user_id, platform_token_id, platform, account_id, platform_video_id,
        source, content_kind, title, published_at, thumbnail_url, platform_url,
        duration_seconds, views, likes, comments, shares,
        visibility, presence,
        json.dumps(extra or {}),
    )
    return str(row["id"]) if row else ""


async def _update_sync_state(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    platform_token_id: str,
    platform: str,
    account_id: str,
    status: str,
    next_cursor: Optional[str],
    total_discovered: int,
    total_linked: int,
    error_detail: Optional[str] = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO platform_content_sync_state
            (user_id, platform_token_id, platform, account_id,
             last_synced_at, next_cursor, total_discovered, total_linked,
             status, error_detail, updated_at)
        VALUES ($1,$2,$3,$4, NOW(),$5,$6,$7, $8,$9,NOW())
        ON CONFLICT (user_id, platform_token_id) DO UPDATE SET
            last_synced_at   = NOW(),
            next_cursor      = EXCLUDED.next_cursor,
            total_discovered = EXCLUDED.total_discovered,
            total_linked     = EXCLUDED.total_linked,
            status           = EXCLUDED.status,
            error_detail     = EXCLUDED.error_detail,
            updated_at       = NOW()
        """,
        user_id, platform_token_id, platform, account_id,
        next_cursor, total_discovered, total_linked,
        status, error_detail,
    )


async def _get_sync_state(
    conn: asyncpg.Connection,
    user_id: str,
    platform_token_id: str,
) -> Optional[Dict]:
    row = await conn.fetchrow(
        "SELECT * FROM platform_content_sync_state WHERE user_id=$1 AND platform_token_id=$2",
        user_id, platform_token_id,
    )
    return dict(row) if row else None


# ── Per-platform catalog list fetchers ─────────────────────────────────────

async def _list_tiktok_videos(
    access_token: str,
    cursor: Optional[int] = None,
    page_size: int = _PAGE_SIZE_TIKTOK,
) -> Tuple[List[Dict], Optional[int], bool]:
    """
    Returns (videos, next_cursor, has_more).
    Every video returned by the TikTok list API is mapped the same way (no short/long or
    duration-based classification). Pagination walks the full catalog.
    Each video dict has: platform_video_id, title, views, likes, comments, shares,
    thumbnail_url, platform_url, duration_seconds, published_at, content_kind (unset).
    """
    from services.tiktok_api import tiktok_video_list_url, tiktok_envelope_error

    body: Dict[str, Any] = {"max_count": page_size}
    if cursor:
        body["cursor"] = cursor

    async def _call():
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                tiktok_video_list_url(),
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                json=body,
            )
            resp.raise_for_status()
            return resp.json()

    data = await _with_retry(_call)
    env_err = tiktok_envelope_error(data)
    if env_err:
        logger.warning(f"[catalog-sync] TikTok envelope error: {env_err}")
        return [], None, False

    d = data.get("data") or {}
    raw_videos = d.get("videos") or []
    has_more = bool(d.get("has_more", False))
    next_cursor = d.get("cursor")  # integer

    videos = []
    for v in raw_videos:
        vid_id = str(v.get("id") or "")
        if not vid_id:
            continue
        dur = _int(v.get("duration"))
        ct = v.get("create_time")
        pub = datetime.fromtimestamp(int(ct), tz=timezone.utc) if ct else None
        videos.append({
            "platform_video_id": vid_id,
            "title": (v.get("title") or v.get("video_description") or "")[:500],
            "views": _int(v.get("view_count")),
            "likes": _int(v.get("like_count")),
            "comments": _int(v.get("comment_count")),
            "shares": _int(v.get("share_count")),
            "thumbnail_url": v.get("cover_image_url"),
            "platform_url": v.get("share_url"),
            "duration_seconds": dur or None,
            "published_at": pub,
            "content_kind": None,
        })

    return videos, (int(next_cursor) if next_cursor else None), has_more


def _youtube_channel_id_for_catalog(token: Dict[str, Any], account_id: str) -> Optional[str]:
    """
    Prefer the channel tied to this connection (UC… id on platform_tokens / token blob).
    channels.list?mine=true only returns the default channel for the Google account — wrong when
    the user connected a brand / second channel stored as account_id.
    """
    for k in ("channel_id", "youtube_channel_id"):
        v = (token or {}).get(k)
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("UC") and len(s) >= 22:
                return s
    aid = (account_id or "").strip()
    if aid.startswith("UC") and len(aid) >= 22:
        return aid
    return None


async def _list_youtube_videos(
    access_token: str,
    page_token: Optional[str] = None,
    page_size: int = _PAGE_SIZE_YOUTUBE,
    channel_id: Optional[str] = None,
) -> Tuple[List[Dict], Optional[str], bool]:
    """
    Returns (videos, next_page_token, has_more).
    Uses uploads playlist → batched video statistics.
    """
    BASE = "https://www.googleapis.com/youtube/v3"
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient(timeout=25) as client:
        # 1. Resolve uploads playlist: prefer explicit channel id (matches Studio for that channel).
        ch_params: Dict[str, Any] = {"part": "contentDetails"}
        if channel_id:
            ch_params["id"] = channel_id
        else:
            ch_params["mine"] = "true"
        ch_resp = await client.get(f"{BASE}/channels", params=ch_params, headers=headers)
        if ch_resp.status_code != 200:
            logger.warning(f"[catalog-sync] YouTube channels HTTP {ch_resp.status_code}")
            return [], None, False
        ch_items = ch_resp.json().get("items") or []
        if not ch_items and channel_id:
            ch_resp = await client.get(
                f"{BASE}/channels",
                params={"part": "contentDetails", "mine": "true"},
                headers=headers,
            )
            if ch_resp.status_code == 200:
                ch_items = ch_resp.json().get("items") or []
        if not ch_items:
            return [], None, False
        uploads_playlist = (
            ch_items[0].get("contentDetails", {})
            .get("relatedPlaylists", {})
            .get("uploads", "")
        )
        if not uploads_playlist:
            return [], None, False

        # 2. Page through playlist items
        pi_params: Dict[str, Any] = {
            "part": "snippet,contentDetails",
            "playlistId": uploads_playlist,
            "maxResults": page_size,
        }
        if page_token:
            pi_params["pageToken"] = page_token

        pi_resp = await client.get(f"{BASE}/playlistItems", params=pi_params, headers=headers)
        if pi_resp.status_code != 200:
            logger.warning(f"[catalog-sync] YouTube playlistItems HTTP {pi_resp.status_code}")
            return [], None, False

        pi_body = pi_resp.json()
        pi_items = pi_body.get("items") or []
        next_pt = pi_body.get("nextPageToken")

        video_ids = [
            i["contentDetails"]["videoId"]
            for i in pi_items
            if i.get("contentDetails", {}).get("videoId")
        ]
        if not video_ids:
            return [], next_pt, bool(next_pt)

        # 3. Batch fetch statistics + snippet + status (privacy)
        vid_resp = await client.get(
            f"{BASE}/videos",
            params={"part": "statistics,snippet,contentDetails,status", "id": ",".join(video_ids)},
            headers=headers,
        )
        vid_items = vid_resp.json().get("items") or [] if vid_resp.status_code == 200 else []
        stat_map = {v["id"]: v for v in vid_items}

        videos = []
        for pi in pi_items:
            vid_id = (pi.get("contentDetails") or {}).get("videoId")
            if not vid_id:
                continue
            snip = (pi.get("snippet") or {})
            vdata = stat_map.get(vid_id, {})
            stats = vdata.get("statistics") or {}
            vsnip = vdata.get("snippet") or {}
            cd = vdata.get("contentDetails") or {}
            st = vdata.get("status") or {}
            yt_vis = str(st.get("privacyStatus") or "").lower() or None

            # ISO8601 duration → seconds (basic)
            dur_str = cd.get("duration") or ""
            dur_sec = _yt_iso_duration_to_seconds(dur_str)
            kind, kind_rule = _youtube_content_kind_and_rule(vsnip, snip, dur_sec)

            pub_str = vsnip.get("publishedAt") or snip.get("publishedAt")
            pub = None
            if pub_str:
                try:
                    pub = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                except Exception as e:
                    logger.debug("[catalog-sync] YouTube publishedAt parse failed %r: %s", pub_str[:40], e)

            thumb = (
                (vsnip.get("thumbnails") or snip.get("thumbnails") or {})
                .get("medium", {})
                .get("url")
            )
            extra: Dict[str, Any] = {
                "yt_kind_rule": kind_rule,
                "youtube_shorts_url": f"https://www.youtube.com/shorts/{vid_id}",
            }
            videos.append({
                "platform_video_id": vid_id,
                "title": (vsnip.get("title") or snip.get("title") or "")[:500],
                "views": _int(stats.get("viewCount")),
                "likes": _int(stats.get("likeCount")),
                "comments": _int(stats.get("commentCount")),
                "shares": 0,
                "thumbnail_url": thumb,
                "platform_url": (
                    f"https://www.youtube.com/shorts/{vid_id}"
                    if kind == "short"
                    else f"https://www.youtube.com/watch?v={vid_id}"
                ),
                "duration_seconds": dur_sec,
                "published_at": pub,
                "content_kind": kind,
                "visibility": yt_vis,
                "presence": "ok",
                "extra": extra,
            })

        return videos, next_pt, bool(next_pt)


def _yt_iso_duration_to_seconds(iso: str) -> Optional[int]:
    """Parse PT#H#M#S → total seconds."""
    if not iso or not iso.startswith("PT"):
        return None
    import re
    h = int((re.search(r"(\d+)H", iso) or [None, 0])[1] or 0)
    m = int((re.search(r"(\d+)M", iso) or [None, 0])[1] or 0)
    s = int((re.search(r"(\d+)S", iso) or [None, 0])[1] or 0)
    total = h * 3600 + m * 60 + s
    return total if total > 0 else None


async def _list_instagram_media(
    access_token: str,
    ig_user_id: str,
    after_cursor: Optional[str] = None,
    page_size: int = _PAGE_SIZE_META,
) -> Tuple[List[Dict], Optional[str], bool]:
    """Instagram Graph API: list user media (Reels + videos)."""
    fields = "id,caption,media_type,thumbnail_url,permalink,timestamp,like_count,comments_count"
    params: Dict[str, Any] = {
        "fields": fields,
        "limit": page_size,
        "access_token": access_token,
    }
    if after_cursor:
        params["after"] = after_cursor

    async def _call():
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"https://graph.facebook.com/v19.0/{ig_user_id}/media",
                params=params,
            )
            resp.raise_for_status()
            return resp.json()

    try:
        data = await _with_retry(_call)
    except Exception as e:
        logger.warning(f"[catalog-sync] Instagram media error: {e}")
        return [], None, False

    raw = data.get("data") or []
    paging = data.get("paging") or {}
    next_cursor = (paging.get("cursors") or {}).get("after") if paging.get("next") else None

    videos = []
    for m in raw:
        mtype = (m.get("media_type") or "").upper()
        if mtype not in ("VIDEO", "REELS"):
            continue  # skip images / carousels without video
        mid = str(m.get("id") or "")
        if not mid:
            continue
        kind = "reel" if mtype == "REELS" else "short"
        ts = m.get("timestamp")
        pub = None
        if ts:
            try:
                pub = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception as e:
                logger.debug("[catalog-sync] Instagram timestamp parse failed %r: %s", str(ts)[:40], e)
        videos.append({
            "platform_video_id": mid,
            "title": (m.get("caption") or "")[:500],
            "views": 0,  # IG doesn't expose video_views in basic media endpoint
            "likes": _int(m.get("like_count")),
            "comments": _int(m.get("comments_count")),
            "shares": 0,
            "thumbnail_url": m.get("thumbnail_url"),
            "platform_url": m.get("permalink"),
            "duration_seconds": None,
            "published_at": pub,
            "content_kind": kind,
        })

    return videos, next_cursor, bool(next_cursor)


async def _list_facebook_videos(
    access_token: str,
    page_id: str,
    after_cursor: Optional[str] = None,
    page_size: int = _PAGE_SIZE_META,
) -> Tuple[List[Dict], Optional[str], bool]:
    """Facebook Graph API: list page videos."""
    fields = "id,title,description,length,created_time,permalink_url,picture,likes.summary(true),comments.summary(true),shares"
    params: Dict[str, Any] = {
        "fields": fields,
        "limit": page_size,
        "access_token": access_token,
    }
    if after_cursor:
        params["after"] = after_cursor

    async def _call():
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"https://graph.facebook.com/v19.0/{page_id}/videos",
                params=params,
            )
            resp.raise_for_status()
            return resp.json()

    try:
        data = await _with_retry(_call)
    except Exception as e:
        logger.warning(f"[catalog-sync] Facebook videos error: {e}")
        return [], None, False

    raw = data.get("data") or []
    paging = data.get("paging") or {}
    next_cursor = (paging.get("cursors") or {}).get("after") if paging.get("next") else None

    videos = []
    for v in raw:
        vid_id = str(v.get("id") or "")
        if not vid_id:
            continue
        dur = v.get("length")
        kind = "short" if dur and float(dur) <= 60 else "long"
        ts = v.get("created_time")
        pub = None
        if ts:
            try:
                pub = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception as e:
                logger.debug("[catalog-sync] Facebook created_time parse failed %r: %s", str(ts)[:40], e)
        videos.append({
            "platform_video_id": vid_id,
            "title": (v.get("title") or v.get("description") or "")[:500],
            "views": 0,
            "likes": _int((v.get("likes") or {}).get("summary", {}).get("total_count")),
            "comments": _int((v.get("comments") or {}).get("summary", {}).get("total_count")),
            "shares": _int((v.get("shares") or {}).get("count")),
            "thumbnail_url": v.get("picture"),
            "platform_url": v.get("permalink_url"),
            "duration_seconds": int(float(dur)) if dur else None,
            "published_at": pub,
            "content_kind": kind,
        })

    return videos, next_cursor, bool(next_cursor)


async def _list_facebook_reels(
    access_token: str,
    page_id: str,
    after_cursor: Optional[str] = None,
    page_size: int = _PAGE_SIZE_META,
) -> Tuple[List[Dict], Optional[str], bool]:
    """
    Facebook Graph API: list page Reels (GET). Merged with /videos by id in the sync loop.
    Requires appropriate Page permissions; failures are non-fatal (empty list).
    """
    fields = "id,title,description,length,created_time,permalink_url,picture,likes.summary(true),comments.summary(true),shares"
    params: Dict[str, Any] = {
        "fields": fields,
        "limit": page_size,
        "access_token": access_token,
    }
    if after_cursor:
        params["after"] = after_cursor

    async def _call():
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"https://graph.facebook.com/v19.0/{page_id}/video_reels",
                params=params,
            )
            if resp.status_code != 200:
                logger.debug(
                    "[catalog-sync] Facebook video_reels HTTP %s: %s",
                    resp.status_code,
                    (resp.text or "")[:200],
                )
                return None
            return resp.json()

    try:
        data = await _with_retry(_call)
    except Exception as e:
        logger.debug("[catalog-sync] Facebook video_reels error: %s", e)
        return [], None, False

    if data is None:
        return [], None, False

    raw = data.get("data") or []
    paging = data.get("paging") or {}
    next_cursor = (paging.get("cursors") or {}).get("after") if paging.get("next") else None

    videos = []
    for v in raw:
        vid_id = str(v.get("id") or "")
        if not vid_id:
            continue
        dur = v.get("length")
        ts = v.get("created_time")
        pub = None
        if ts:
            try:
                pub = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception as e:
                logger.debug("[catalog-sync] Facebook Reel created_time parse failed %r: %s", str(ts)[:40], e)
        videos.append({
            "platform_video_id": vid_id,
            "title": (v.get("title") or v.get("description") or "")[:500],
            "views": 0,
            "likes": _int((v.get("likes") or {}).get("summary", {}).get("total_count")),
            "comments": _int((v.get("comments") or {}).get("summary", {}).get("total_count")),
            "shares": _int((v.get("shares") or {}).get("count")),
            "thumbnail_url": v.get("picture"),
            "platform_url": v.get("permalink_url"),
            "duration_seconds": int(float(dur)) if dur else None,
            "published_at": pub,
            "content_kind": "reel",
            "extra": {"fb_surface": "video_reels"},
        })

    return videos, next_cursor, bool(next_cursor)


# ── Token normalisation (reuse pattern from platform_metrics_job) ──────────
def _normalize_account_id(platform: str, row: Dict, token: Dict) -> str:
    raw = row.get("account_id")
    if raw:
        return str(raw)
    if platform == "instagram":
        return str(token.get("ig_user_id") or token.get("instagram_user_id") or "")
    if platform == "facebook":
        return str(token.get("page_id") or token.get("facebook_page_id") or token.get("fb_page_id") or "")
    if platform == "youtube":
        return str(token.get("channel_id") or "")
    if platform == "tiktok":
        return str(token.get("open_id") or "")
    return ""


async def _mark_youtube_pci_not_in_catalog(
    conn: asyncpg.Connection,
    user_id: str,
    platform_token_id: str,
    account_id: str,
    seen_ids: List[str],
) -> int:
    """
    After a full YouTube uploads-playlist walk (natural end, not max_pages cutoff),
    mark PCI rows for this token that are no longer returned by the channel API.
    """
    if not seen_ids:
        return 0
    r = await conn.execute(
        """
        UPDATE platform_content_items
           SET presence = 'not_in_channel_catalog',
               updated_at = NOW()
         WHERE user_id = $1::uuid
           AND platform = 'youtube'
           AND platform_token_id = $2::uuid
           AND account_id = $3
           AND NOT (platform_video_id = ANY($4::text[]))
        """,
        user_id,
        platform_token_id,
        account_id,
        seen_ids,
    )
    try:
        return int(str(r).split()[-1])
    except Exception:
        return 0


# ── Core per-token sync ─────────────────────────────────────────────────────
async def sync_catalog_for_token(
    pool: asyncpg.Pool,
    user_id: str,
    token_row: Dict,
    token: Dict,
) -> Dict[str, int]:
    """
    Discover all videos for one connected account token and upsert them into
    `platform_content_items`. Respects cursor from previous run for incremental sync.

    Returns {"discovered": N, "upserted": N, "errors": N}.
    """
    platform = str(token_row.get("platform") or "").lower()
    token_row_id = str(token_row.get("id") or "")
    account_id = _normalize_account_id(platform, token_row, token)
    access_token = token.get("access_token") or ""

    if platform not in ("tiktok", "youtube", "instagram", "facebook"):
        return {"discovered": 0, "upserted": 0, "errors": 0}

    async with pool.acquire() as conn:
        state = await _get_sync_state(conn, user_id, token_row_id)
        await _update_sync_state(
            conn,
            user_id=user_id, platform_token_id=token_row_id, platform=platform,
            account_id=account_id, status="syncing",
            next_cursor=state.get("next_cursor") if state else None,
            total_discovered=state.get("total_discovered", 0) if state else 0,
            total_linked=state.get("total_linked", 0) if state else 0,
        )

    prior_cursor = state.get("next_cursor") if state else None
    prior_discovered = state.get("total_discovered", 0) if state else 0

    discovered = 0
    upserted = 0
    errors = 0
    current_cursor: Any = prior_cursor

    # TikTok cursor is an integer; others are strings
    if platform == "tiktok" and current_cursor:
        try:
            current_cursor = int(current_cursor)
        except Exception:
            current_cursor = None

    max_pages = _MAX_PAGES_PER_TOKEN
    if platform == "youtube":
        max_pages = _YOUTUBE_CATALOG_MAX_PAGES
    yt_ch: Optional[str] = None
    if platform == "youtube":
        yt_ch = _youtube_channel_id_for_catalog(token, account_id)

    seen_youtube_ids: set = set()
    exhausted_playlist = False

    for _page in range(max_pages):
        try:
            if platform == "tiktok":
                videos, next_cur, has_more = await _list_tiktok_videos(
                    access_token, cursor=current_cursor)
            elif platform == "youtube":
                videos, next_cur, has_more = await _list_youtube_videos(
                    access_token, page_token=current_cursor, channel_id=yt_ch)
            elif platform == "instagram":
                ig_id = token.get("ig_user_id") or token.get("instagram_user_id") or account_id
                videos, next_cur, has_more = await _list_instagram_media(
                    access_token, ig_id, after_cursor=current_cursor)
            elif platform == "facebook":
                page_id = token.get("page_id") or token.get("facebook_page_id") or token.get("fb_page_id") or account_id
                cur_v, cur_r = parse_facebook_dual_cursor(
                    str(current_cursor) if current_cursor is not None else None
                )
                v_block, next_v, more_v = await _list_facebook_videos(
                    access_token, page_id, after_cursor=cur_v)
                r_block, next_r, more_r = await _list_facebook_reels(
                    access_token, page_id, after_cursor=cur_r)
                videos = _merge_catalog_videos_by_id(v_block, r_block)
                next_cur = dump_facebook_dual_cursor(next_v, next_r)
                has_more = bool(more_v or more_r)
            else:
                break
        except Exception as e:
            logger.warning(f"[catalog-sync] {platform}/{token_row_id} page error: {e}")
            errors += 1
            break

        discovered += len(videos)

        if platform == "youtube":
            for v in videos:
                _pid = str(v.get("platform_video_id") or "").strip()
                if _pid:
                    seen_youtube_ids.add(_pid)

        async with pool.acquire() as conn:
            for v in videos:
                try:
                    await _upsert_content_item(
                        conn,
                        user_id=user_id,
                        platform_token_id=token_row_id,
                        platform=platform,
                        account_id=account_id,
                        source="external",
                        **v,
                    )
                    upserted += 1
                except Exception as e:
                    logger.debug(f"[catalog-sync] upsert error {platform} {v.get('platform_video_id')}: {e}")
                    errors += 1

        if platform == "facebook":
            current_cursor = next_cur
        else:
            current_cursor = str(next_cur) if next_cur is not None else None
        if not has_more or not current_cursor:
            current_cursor = None
            exhausted_playlist = True
            break

    # Persist final state
    async with pool.acquire() as conn:
        if platform == "youtube" and exhausted_playlist and seen_youtube_ids:
            n_gone = await _mark_youtube_pci_not_in_catalog(
                conn, user_id, token_row_id, account_id, list(seen_youtube_ids)
            )
            if n_gone:
                logger.info(
                    "[catalog-sync] YouTube PCI reconcile: %s rows marked not_in_channel_catalog (token=%s…)",
                    n_gone,
                    token_row_id[:8],
                )
        # Run linker for newly synced items
        linked = await _link_uploads_for_user_token(conn, user_id, platform, account_id)
        # For TikTok: backfill platform_video_id on uploads that never got one.
        # TikTok's Content Posting API only returns a publish_id at upload time;
        # the real video_id is only available later.  Now that we have the full
        # catalog, we can match catalog rows to uploads by title so sync-analytics
        # can query fresh metrics going forward.
        if platform == "tiktok":
            await _backfill_tiktok_video_ids(conn, user_id, account_id)
        await _update_sync_state(
            conn,
            user_id=user_id, platform_token_id=token_row_id, platform=platform,
            account_id=account_id, status="done" if errors == 0 else "error",
            next_cursor=current_cursor,
            total_discovered=prior_discovered + discovered,
            total_linked=linked,
            error_detail=f"{errors} errors" if errors else None,
        )

    logger.info(f"[catalog-sync] {platform} user={user_id[:8]} discovered={discovered} upserted={upserted} linked={linked}")
    return {"discovered": discovered, "upserted": upserted, "errors": errors}


# ── TikTok video-id backfill ─────────────────────────────────────────────────
async def _backfill_tiktok_video_ids(
    conn: asyncpg.Connection,
    user_id: str,
    account_id: str,
) -> int:
    """
    After a TikTok catalog sync: find succeeded uploads that never received a
    ``platform_video_id`` in their ``platform_results`` (the Content Posting API
    only returns a ``publish_id`` at upload time) and match them to catalog rows
    by normalised title.  When a high-confidence match is found:
    * ``platform_results[].platform_video_id`` is patched on the upload row.
    * ``platform_content_items.upload_id`` is set on the catalog row.
    * ``platform_content_items.source`` is promoted to ``'linked'``.

    Returns the number of uploads updated.
    """
    import re

    def _norm(s: str) -> str:
        s = str(s or "").lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    # Catalog rows with a title for this user/account
    cat_rows = await conn.fetch(
        """
        SELECT id, platform_video_id, title, published_at, views, likes, comments, shares
          FROM platform_content_items
         WHERE user_id = $1::uuid AND platform = 'tiktok' AND account_id = $2
           AND platform_video_id IS NOT NULL AND platform_video_id != ''
           AND title IS NOT NULL AND title != ''
        """,
        user_id, account_id,
    )
    if not cat_rows:
        return 0

    cat_by_norm: Dict[str, dict] = {}
    for c in cat_rows:
        key = _norm(c["title"])
        if key and key not in cat_by_norm:
            cat_by_norm[key] = dict(c)

    # Succeeded TikTok uploads that still have no platform_video_id
    upload_rows = await conn.fetch(
        """
        SELECT id, title, caption, platform_results, created_at
          FROM uploads
         WHERE user_id = $1::uuid
           AND status IN ('succeeded', 'completed', 'partial')
           AND 'tiktok' = ANY(platforms)
        """,
        user_id,
    )

    updated = 0
    for row in upload_rows:
        pr = row["platform_results"]
        if isinstance(pr, str):
            try:
                pr = json.loads(pr)
            except Exception:
                pr = []
        if not isinstance(pr, list):
            pr = []

        # Check if TikTok entry already has a valid platform_video_id
        already_has_id = False
        tiktok_idx = -1
        for i, p in enumerate(pr):
            if not isinstance(p, dict):
                continue
            if str(p.get("platform") or "").lower() != "tiktok":
                continue
            vid = str(p.get("platform_video_id") or p.get("video_id") or "").strip()
            if vid and vid != "null":
                already_has_id = True
                break
            tiktok_idx = i

        if already_has_id:
            continue

        # Try to find a matching catalog row by title
        upload_title_norm = _norm(row.get("title") or "")
        upload_caption_norm = _norm(row.get("caption") or "")

        best_cat: Optional[dict] = None
        for key, cat in cat_by_norm.items():
            if upload_title_norm and upload_title_norm == key:
                best_cat = cat
                break
            if upload_title_norm and upload_title_norm in key:
                best_cat = cat
                break
            if upload_caption_norm and len(upload_caption_norm) >= 8 and upload_caption_norm in key:
                best_cat = cat
                break

        if not best_cat:
            continue

        found_vid = str(best_cat["platform_video_id"])
        uname_row = next(
            (str(p.get("account_username") or "").strip().lstrip("@")
             for p in pr if isinstance(p, dict) and str(p.get("platform") or "").lower() == "tiktok"),
            "",
        )
        tiktok_url = (
            f"https://www.tiktok.com/@{uname_row}/video/{found_vid}"
            if uname_row else
            f"https://www.tiktok.com/video/{found_vid}"
        )

        # Patch the platform_results entry
        if tiktok_idx >= 0:
            pr[tiktok_idx]["platform_video_id"] = found_vid
            pr[tiktok_idx]["video_id"] = found_vid
            pr[tiktok_idx]["platform_url"] = tiktok_url
            pr[tiktok_idx]["url"] = tiktok_url
        else:
            pr.append({
                "platform": "tiktok",
                "account_id": account_id,
                "platform_video_id": found_vid,
                "video_id": found_vid,
                "platform_url": tiktok_url,
                "success": True,
            })

        try:
            await conn.execute(
                """
                UPDATE uploads
                   SET platform_results = $1::jsonb, updated_at = NOW()
                 WHERE id = $2::uuid
                """,
                json.dumps(pr), str(row["id"]),
            )
            # Also promote catalog row source and link upload_id
            await conn.execute(
                """
                UPDATE platform_content_items
                   SET upload_id  = $1::uuid,
                       source     = CASE WHEN source = 'external' THEN 'linked' ELSE source END,
                       updated_at = NOW()
                 WHERE user_id = $2::uuid AND platform = 'tiktok'
                   AND account_id = $3 AND platform_video_id = $4
                """,
                str(row["id"]), user_id, account_id, found_vid,
            )
            updated += 1
            logger.info(
                "[catalog-sync] TikTok backfill: upload=%s matched video_id=%s",
                str(row["id"])[:8], found_vid,
            )
        except Exception as e:
            logger.debug("[catalog-sync] TikTok backfill update error: %s", e)

    return updated


# ── Upload → Catalog linker ─────────────────────────────────────────────────
async def _link_uploads_for_user_token(
    conn: asyncpg.Connection,
    user_id: str,
    platform: str,
    account_id: str,
) -> int:
    """
    Scan completed uploads for `user_id` + `platform`, extract platform_video_ids
    from `platform_results`, and set `upload_id` on matching catalog rows.
    Also upserts catalog rows for uploads that haven't been synced yet (source='uploadm8').
    Returns count of rows linked.
    """
    rows = await conn.fetch(
        """
        SELECT id, platform_results, title, thumbnail_r2_key, created_at
        FROM uploads
        WHERE user_id = $1
          AND status IN ('completed', 'succeeded', 'partial')
          AND $2 = ANY(platforms)
        """,
        user_id, platform,
    )

    linked = 0
    for row in rows:
        upload_id = str(row["id"])
        pr = row["platform_results"]
        if isinstance(pr, str):
            try:
                pr = json.loads(pr)
            except Exception:
                pr = []
        if not isinstance(pr, list):
            if isinstance(pr, dict):
                pr = [{"platform": k, **v} if isinstance(v, dict) else {"platform": k} for k, v in pr.items()]
            else:
                pr = []

        for p in pr:
            if not isinstance(p, dict):
                continue
            p_plat = str(p.get("platform") or "").lower()
            if p_plat != platform:
                continue

            vid_id = (
                p.get("platform_video_id")
                or p.get("video_id")
                or p.get("tiktok_video_id")
                or p.get("youtube_video_id")
            )
            if not vid_id:
                continue
            vid_id = str(vid_id)

            # Try to link existing external row
            result = await conn.execute(
                """
                UPDATE platform_content_items
                SET upload_id  = $1,
                    source     = 'linked',
                    updated_at = NOW()
                WHERE user_id = $2
                  AND platform = $3
                  AND account_id = $4
                  AND platform_video_id = $5
                  AND (upload_id IS NULL OR upload_id != $1)
                """,
                upload_id, user_id, platform, account_id, vid_id,
            )
            if result and result != "UPDATE 0":
                linked += 1
                continue

            # If no external row exists yet, insert as 'uploadm8' sourced
            pub_at = row["created_at"]
            try:
                await conn.execute(
                    """
                    INSERT INTO platform_content_items
                        (user_id, platform, account_id, platform_video_id,
                         upload_id, source, published_at, updated_at)
                    VALUES ($1,$2,$3,$4, $5,'uploadm8',$6,NOW())
                    ON CONFLICT (user_id, platform, account_id, platform_video_id) DO UPDATE SET
                        upload_id  = EXCLUDED.upload_id,
                        source     = CASE
                            WHEN platform_content_items.source = 'external' THEN 'linked'
                            ELSE 'uploadm8' END,
                        updated_at = NOW()
                    """,
                    user_id, platform, account_id, vid_id,
                    upload_id, pub_at,
                )
                linked += 1
            except Exception as e:
                logger.debug(f"[catalog-sync] upload insert error: {e}")

    return linked


# ── Orchestrator for a single user ─────────────────────────────────────────
async def sync_catalog_for_user(
    pool: asyncpg.Pool,
    user_id: str,
    force_full: bool = False,
) -> Dict[str, Any]:
    """
    Run catalog sync for every connected token for user_id.
    If force_full=True, clears pagination cursors so a full re-scan runs.
    """
    uid = str(user_id)

    async with pool.acquire() as conn:
        token_rows = await conn.fetch(
            """
            SELECT id, platform, token_blob, account_id
            FROM platform_tokens
            WHERE user_id = $1 AND revoked_at IS NULL
            """,
            uid,
        )
        if force_full:
            await conn.execute(
                "UPDATE platform_content_sync_state SET next_cursor = NULL WHERE user_id = $1",
                uid,
            )

    totals: Dict[str, int] = {"discovered": 0, "upserted": 0, "errors": 0, "tokens": 0}
    sem = asyncio.Semaphore(_SEMAPHORE_LIMIT)

    # Refresh tokens before syncing (reuse existing infrastructure)
    refreshed_jobs: List[Tuple[Dict, Dict]] = []
    for row in token_rows:
        plat = str(row["platform"] or "").lower()
        blob = row["token_blob"]
        if not blob or plat not in ("tiktok", "youtube", "instagram", "facebook"):
            continue
        try:
            token = decrypt_token(blob)
        except Exception:
            continue
        if not token:
            continue
        if plat == "instagram" and not token.get("ig_user_id") and row["account_id"]:
            token["ig_user_id"] = str(row["account_id"])
        if plat == "facebook" and not token.get("page_id") and row["account_id"]:
            token["page_id"] = str(row["account_id"])

        # Try token refresh
        try:
            from stages.publish_stage import _refresh_tiktok_token, _refresh_youtube_token, _refresh_meta_token
            row_pk = str(row.get("id") or "")
            if plat == "tiktok":
                token = await _refresh_tiktok_token(
                    token, db_pool=pool, user_id=uid, token_row_id=row_pk or None
                )
            elif plat == "youtube":
                token = await _refresh_youtube_token(
                    token, db_pool=pool, user_id=uid, token_row_id=row_pk or None
                )
            elif plat in ("instagram", "facebook"):
                token = await _refresh_meta_token(
                    token, platform=plat, db_pool=pool, user_id=uid, token_row_id=row_pk or None
                )
        except Exception as e:
            logger.debug(f"[catalog-sync] token refresh {plat}: {e}")

        refreshed_jobs.append((dict(row), token))

    async def _run_one(row_dict: Dict, token: Dict):
        async with sem:
            try:
                result = await sync_catalog_for_token(pool, uid, row_dict, token)
                return result
            except Exception as e:
                logger.warning(f"[catalog-sync] token {row_dict.get('id')}: {e}")
                return {"discovered": 0, "upserted": 0, "errors": 1}

    results = await asyncio.gather(*[_run_one(r, t) for r, t in refreshed_jobs])

    for r in results:
        totals["discovered"] += r.get("discovered", 0)
        totals["upserted"] += r.get("upserted", 0)
        totals["errors"] += r.get("errors", 0)
        totals["tokens"] += 1

    return totals


# ── Aggregate stats query ───────────────────────────────────────────────────
def _parse_period_to_sql_interval(period: Optional[str]) -> Optional[str]:
    """
    Convert a user-facing period string to a Postgres INTERVAL expression.

    Accepted formats (case-insensitive):
        • "7d"  / "7 days"  / "30"  (bare number → days)
        • "7h"  / "7 hours"
        • "7m"  / "7 minutes"
        • "all" / ""  / None  → no time filter

    Returns a Postgres-safe interval string like '7 days' / '7 hours',
    or None for "all time".
    """
    import re as _re
    if not period or str(period).lower() in ("all", "0", ""):
        return None
    p = str(period).strip().lower()
    # bare integer → days
    if _re.fullmatch(r"\d+", p):
        return f"{int(p)} days"
    # e.g. "7d", "7 days", "30days"
    m = _re.fullmatch(r"(\d+(?:\.\d+)?)\s*d(?:ays?)?", p)
    if m:
        v = float(m.group(1))
        return f"{v} days" if v != int(v) else f"{int(v)} days"
    # e.g. "7h", "24hours", "7 hours"
    m = _re.fullmatch(r"(\d+(?:\.\d+)?)\s*h(?:ours?)?", p)
    if m:
        v = float(m.group(1))
        return f"{v} hours" if v != int(v) else f"{int(v)} hours"
    # e.g. "30m", "90 minutes"
    m = _re.fullmatch(r"(\d+(?:\.\d+)?)\s*m(?:in(?:utes?)?)?", p)
    if m:
        v = float(m.group(1))
        return f"{v} minutes" if v != int(v) else f"{int(v)} minutes"
    # Unrecognised — treat as "30 days" safe fallback
    return "30 days"


async def get_catalog_aggregate(
    pool: asyncpg.Pool,
    user_id: str,
    days: Optional[int] = None,
    period: Optional[str] = None,
    platform: Optional[str] = None,
    source: Optional[str] = None,
    account_id: Optional[str] = None,
    window_start: Optional[datetime] = None,
    window_end_exclusive: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Return aggregated views/likes/comments/shares + per-platform breakdown
    and per-source breakdown from `platform_content_items`.

    Time filtering (first match wins), on
    ``COALESCE(pci.published_at, u.completed_at, u.created_at)`` (joins ``uploads``):

      • Custom UTC window: half-open [start, end) on that effective timestamp.
      • Rolling ``period`` / ``days``: last N hours/days/minutes on that timestamp.
      • Metrics: ``GREATEST`` catalog row vs linked ``uploads`` row when the upload
        targets a single platform (or ``platforms`` is unset), matching
        ``GET /api/catalog/content``.
    """
    uid = str(user_id)
    conditions = ["pci.user_id = $1"]
    params: List[Any] = [uid]

    explicit_window = window_start is not None or window_end_exclusive is not None
    interval: Optional[str] = None
    eff_ts = "COALESCE(pci.published_at, u.completed_at, u.created_at)"

    if explicit_window:
        if window_start is not None:
            conditions.append(f"{eff_ts} >= ${len(params) + 1}")
            params.append(window_start)
        if window_end_exclusive is not None:
            conditions.append(f"{eff_ts} < ${len(params) + 1}")
            params.append(window_end_exclusive)
        conditions.append(f"{eff_ts} IS NOT NULL")
    else:
        # Resolve interval: period string wins over bare days int
        if period:
            interval = _parse_period_to_sql_interval(period)
        elif days and days > 0:
            interval = f"{days} days"
        else:
            interval = None

        if interval:
            conditions.append(f"{eff_ts} >= NOW() - INTERVAL '{interval}'")
            conditions.append(f"{eff_ts} IS NOT NULL")

    if platform:
        conditions.append(f"pci.platform = ${len(params)+1}")
        params.append(platform.lower())
    if source and str(source).lower() != "all":
        conditions.append(f"pci.source = ${len(params)+1}")
        params.append(source.lower())
    if account_id:
        aid = str(account_id).strip()
        if aid:
            conditions.append(f"pci.account_id = ${len(params)+1}")
            params.append(aid)

    where = " AND ".join(conditions)

    _merge_views = """CASE
        WHEN u.id IS NULL THEN COALESCE(pci.views,0)::bigint
        WHEN u.platforms IS NULL OR COALESCE(cardinality(u.platforms), 0) = 0
          OR (cardinality(u.platforms) = 1 AND lower(trim(u.platforms[1])) = lower(trim(pci.platform::text)))
        THEN GREATEST(COALESCE(pci.views,0), COALESCE(u.views,0))::bigint
        ELSE COALESCE(pci.views,0)::bigint
    END"""
    _merge_likes = """CASE
        WHEN u.id IS NULL THEN COALESCE(pci.likes,0)::bigint
        WHEN u.platforms IS NULL OR COALESCE(cardinality(u.platforms), 0) = 0
          OR (cardinality(u.platforms) = 1 AND lower(trim(u.platforms[1])) = lower(trim(pci.platform::text)))
        THEN GREATEST(COALESCE(pci.likes,0), COALESCE(u.likes,0))::bigint
        ELSE COALESCE(pci.likes,0)::bigint
    END"""
    _merge_comments = """CASE
        WHEN u.id IS NULL THEN COALESCE(pci.comments,0)::bigint
        WHEN u.platforms IS NULL OR COALESCE(cardinality(u.platforms), 0) = 0
          OR (cardinality(u.platforms) = 1 AND lower(trim(u.platforms[1])) = lower(trim(pci.platform::text)))
        THEN GREATEST(COALESCE(pci.comments,0), COALESCE(u.comments,0))::bigint
        ELSE COALESCE(pci.comments,0)::bigint
    END"""
    _merge_shares = """CASE
        WHEN u.id IS NULL THEN COALESCE(pci.shares,0)::bigint
        WHEN u.platforms IS NULL OR COALESCE(cardinality(u.platforms), 0) = 0
          OR (cardinality(u.platforms) = 1 AND lower(trim(u.platforms[1])) = lower(trim(pci.platform::text)))
        THEN GREATEST(COALESCE(pci.shares,0), COALESCE(u.shares,0))::bigint
        ELSE COALESCE(pci.shares,0)::bigint
    END"""
    _from = """
        platform_content_items pci
        LEFT JOIN uploads u ON u.id = pci.upload_id AND u.user_id = pci.user_id
    """

    total_sql = f"""
            SELECT
                COUNT(*) as total_videos,
                COALESCE(SUM({_merge_views}), 0)    as views,
                COALESCE(SUM({_merge_likes}), 0)    as likes,
                COALESCE(SUM({_merge_comments}), 0) as comments,
                COALESCE(SUM({_merge_shares}), 0)   as shares
            FROM {_from}
            WHERE {where}
            """
    platform_sql = f"""
            SELECT
                pci.platform as platform,
                COUNT(*) as video_count,
                COALESCE(SUM({_merge_views}), 0)    as views,
                COALESCE(SUM({_merge_likes}), 0)    as likes,
                COALESCE(SUM({_merge_comments}), 0) as comments,
                COALESCE(SUM({_merge_shares}), 0)   as shares
            FROM {_from}
            WHERE {where}
            GROUP BY pci.platform
            ORDER BY views DESC
            """
    source_sql = f"""
            SELECT
                pci.source as source,
                COUNT(*) as video_count,
                COALESCE(SUM({_merge_views}), 0) as views
            FROM {_from}
            WHERE {where}
            GROUP BY pci.source
            """
    sync_sql = """
            SELECT platform, status, last_synced_at, total_discovered, total_linked
            FROM platform_content_sync_state
            WHERE user_id = $1
            ORDER BY last_synced_at DESC NULLS LAST
            """

    q_params = tuple(params)

    async def _load_total():
        async with pool.acquire() as conn:
            return await conn.fetchrow(total_sql, *q_params)

    async def _load_platform():
        async with pool.acquire() as conn:
            return await conn.fetch(platform_sql, *q_params)

    async def _load_source():
        async with pool.acquire() as conn:
            return await conn.fetch(source_sql, *q_params)

    async def _load_sync():
        async with pool.acquire() as conn:
            return await conn.fetch(sync_sql, uid)

    total_row, platform_rows, source_rows, sync_rows = await asyncio.gather(
        _load_total(), _load_platform(), _load_source(), _load_sync()
    )

    total_views = int(total_row["views"] or 0)
    total_likes = int(total_row["likes"] or 0)
    total_comments = int(total_row["comments"] or 0)
    total_shares = int(total_row["shares"] or 0)
    total_eng = (
        round((total_likes + total_comments + total_shares) / total_views * 100, 2)
        if total_views > 0 else 0.0
    )

    by_platform = {}
    for r in platform_rows:
        v = int(r["views"] or 0)
        l = int(r["likes"] or 0)
        c = int(r["comments"] or 0)
        s = int(r["shares"] or 0)
        by_platform[r["platform"]] = {
            "video_count": int(r["video_count"]),
            "views": v, "likes": l, "comments": c, "shares": s,
            "engagement_rate": round((l + c + s) / v * 100, 2) if v > 0 else 0.0,
        }

    by_source = {r["source"]: {"video_count": int(r["video_count"]), "views": int(r["views"] or 0)} for r in source_rows}

    sync_status = [
        {
            "platform": r["platform"],
            "status": r["status"],
            "last_synced_at": r["last_synced_at"].isoformat() if r["last_synced_at"] else None,
            "total_discovered": r["total_discovered"],
            "total_linked": r["total_linked"],
        }
        for r in sync_rows
    ]

    if explicit_window:
        period_out = "explicit_utc"
    elif period:
        period_out = str(period).strip().lower()
    elif days and days > 0:
        period_out = f"{int(days)}d"
    else:
        period_out = "all"

    out: Dict[str, Any] = {
        "total_videos": int(total_row["total_videos"] or 0),
        "views": total_views,
        "likes": total_likes,
        "comments": total_comments,
        "shares": total_shares,
        "engagement_rate": total_eng,
        "by_platform": by_platform,
        "by_source": by_source,
        "sync_status": sync_status,
        "period": period_out,
        "window_start_utc": window_start.isoformat() if window_start else None,
        "window_end_exclusive_utc": window_end_exclusive.isoformat() if window_end_exclusive else None,
        "generated_at": _now().isoformat(),
    }
    out["kpi_sources"] = {
        "canonical_engagement_rollup_version": CANONICAL_ENGAGEMENT_ROLLUP_VERSION,
        "vs_canonical_headline": (
            "GET /api/analytics uses deduped canonical engagement (pci + successful platform_results). "
            "This endpoint uses catalog SQL only — see metric_definitions.catalog_aggregate_engagement "
            "and metric_definitions.engagement_crosswalk."
        ),
        "live_aggregate": (
            "Not included here. GET /api/analytics.live_aggregate is platform_metrics_cache — not summed into headline."
        ),
        "time_basis": (
            "Filter and bucket rows by COALESCE(pci.published_at, u.completed_at, u.created_at) "
            "(half-open when explicit window; rolling NOW() - interval when using period/days)."
        ),
        "period_label": period_out,
    }
    out["metric_definitions"] = metric_definitions_svc.for_catalog_aggregate()
    return out


# ── All-users background sweep (called from worker cron) ───────────────────
async def refresh_catalog_for_all_users(pool: asyncpg.Pool) -> int:
    """Incremental catalog refresh for every user with an active token."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT user_id FROM platform_tokens WHERE revoked_at IS NULL"
        )
    uids = [str(r["user_id"]) for r in rows]
    sem = asyncio.Semaphore(4)

    async def _run(uid: str):
        async with sem:
            try:
                await sync_catalog_for_user(pool, uid)
                await asyncio.sleep(0.2)
                return True
            except Exception as e:
                logger.warning(f"[catalog-sync] all-users uid={uid[:8]}: {e}")
                return False

    results = await asyncio.gather(*[_run(u) for u in uids])
    return sum(1 for x in results if x)
