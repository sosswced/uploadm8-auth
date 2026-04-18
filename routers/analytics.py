"""
UploadM8 Analytics routes — extracted from app.py.
"""

import io
import csv
import json
import re
import time as _time
import asyncio as _asyncio
import logging
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, Response

import core.state
from core.deps import get_current_user, get_current_user_readonly
from core.auth import decrypt_blob
from core.helpers import _now_utc, _safe_json, get_plan
from services.growth_intelligence import fetch_user_pikzels_studio_usage, parse_range_since_until

logger = logging.getLogger("uploadm8-api")

router = APIRouter(tags=["analytics"])

# ------------------------------------------------------------
# Time range parsing (supports presets + custom 'Nd')
# ------------------------------------------------------------
_RANGE_PRESETS_MINUTES = {
    "24h": 24 * 60,
    "7d": 7 * 24 * 60,
    "30d": 30 * 24 * 60,
    "90d": 90 * 24 * 60,
    "6m": 180 * 24 * 60,
    "1y": 365 * 24 * 60,
}

def _range_to_minutes(range_str: str | None, default_minutes: int) -> int:
    r = (range_str or "").strip()
    if not r:
        return default_minutes
    if r in _RANGE_PRESETS_MINUTES:
        return _RANGE_PRESETS_MINUTES[r]
    m = re.fullmatch(r"(\d{1,4})d", r)
    if m:
        days = int(m.group(1))
        # Guardrails: 1 day .. 10 years
        days = max(1, min(days, 3650))
        return days * 24 * 60
    return default_minutes

# ============================================================
# In-memory cache per user  {user_id_str: {"fetched_at": float, "data": dict}}
# ============================================================
_platform_metrics_cache: dict = {}
_PLATFORM_CACHE_TTL = 3 * 60 * 60  # 3 hours


# ============================================================
# Platform metric fetchers
# ============================================================

async def _fetch_tiktok_metrics(access_token: str) -> dict:
    """TikTok Content API — video list totals + follower stats (requires video.list + user.info.stats)."""
    if not access_token:
        return {"status": "not_connected"}

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            # 1) Video list (requires video.list)
            resp = await client.post(
                "https://open.tiktokapis.com/v2/video/list/",
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                # fields is REQUIRED — without it TikTok returns videos with every stat = 0
                json={
                    "max_count": 20,
                    "fields": ["id","title","view_count","like_count","comment_count","share_count","duration","create_time"],
                },
            )
            if resp.status_code != 200:
                logger.warning(f"TikTok video list HTTP {resp.status_code}: {resp.text[:200]}")
                return {"status": "error", "error": f"video_list_http_{resp.status_code}"}

            videos = (resp.json().get("data", {}) or {}).get("videos", []) or []

            def _i(v):
                try:
                    return int(v or 0)
                except Exception:
                    return 0

            views    = sum(_i(v.get("view_count"))    for v in videos)
            likes    = sum(_i(v.get("like_count"))    for v in videos)
            comments = sum(_i(v.get("comment_count")) for v in videos)
            shares   = sum(_i(v.get("share_count"))   for v in videos)

            durs      = [_i(v.get("duration")) for v in videos if v.get("duration") is not None]
            avg_watch = round(sum(durs) / len(durs), 1) if durs else None

            # 2) User info stats (requires user.info.stats scope)
            followers = following = total_likes = video_count = None
            ui = await client.get(
                "https://open.tiktokapis.com/v2/user/info/",
                params={"fields": "follower_count,following_count,likes_count,video_count"},
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if ui.status_code == 200:
                user_obj    = ((ui.json().get("data", {}) or {}).get("user", {}) or {})
                followers   = user_obj.get("follower_count")
                following   = user_obj.get("following_count")
                total_likes = user_obj.get("likes_count")
                video_count = user_obj.get("video_count")
            else:
                logger.warning(f"TikTok user.info.stats HTTP {ui.status_code}: {ui.text[:200]}")

            return {
                "status": "live",
                "analytics_source": "video.list+user.info.stats",
                "followers":   _i(followers)   if followers   is not None else None,
                "following":   _i(following)   if following   is not None else None,
                "total_likes": _i(total_likes) if total_likes is not None else None,
                "video_count": _i(video_count) if video_count is not None else len(videos),
                "views":    views,
                "likes":    likes,
                "comments": comments,
                "shares":   shares,
                "avg_watch_seconds": avg_watch,
            }

    except Exception as e:
        logger.error(f"TikTok metrics error: {e}")
        return {"status": "error", "error": str(e)}


async def _fetch_youtube_metrics(access_token: str) -> dict:
    """YouTube Data API v3 + (optional) YouTube Analytics API."""
    if not access_token:
        return {"status": "not_connected"}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            ch = await client.get(
                "https://www.googleapis.com/youtube/v3/channels",
                params={"part": "statistics", "mine": "true"},
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if ch.status_code != 200:
                return {"status": "error", "error": f"HTTP {ch.status_code}"}

            items = ch.json().get("items", []) or []
            stats = items[0].get("statistics", {}) if items else {}

            views = likes = comments = shares = 0
            avg_watch = minutes_watched = None
            analytics_source = "channel_stats_fallback"
            try:
                today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                thirty = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
                an = await client.get(
                    "https://youtubeanalytics.googleapis.com/v2/reports",
                    params={
                        "ids":        "channel==MINE",
                        "startDate":  thirty,
                        "endDate":    today,
                        # dimensions=day locks column order: [0]=day [1]=views [2]=likes
                        # [3]=comments [4]=shares [5]=avgDuration [6]=minutesWatched
                        # Without it r[0] returns a date string, not view count
                        "dimensions": "day",
                        "metrics":    "views,likes,comments,shares,averageViewDuration,estimatedMinutesWatched",
                        "sort":       "day",
                    },
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if an.status_code == 200:
                    rows = an.json().get("rows", []) or []
                    if rows:
                        views           = sum(int(r[1] or 0) for r in rows)
                        likes           = sum(int(r[2] or 0) for r in rows)
                        comments        = sum(int(r[3] or 0) for r in rows)
                        shares          = sum(int(r[4] or 0) for r in rows)
                        dur_vals        = [float(r[5]) for r in rows if r[5]]
                        avg_watch       = round(sum(dur_vals) / len(dur_vals), 1) if dur_vals else None
                        minutes_watched = int(sum(float(r[6] or 0) for r in rows))
                        analytics_source = "yt-analytics"
                    else:
                        views    = int(stats.get("viewCount",    0))
                        likes    = int(stats.get("likeCount",    0)) if "likeCount"    in stats else 0
                        comments = int(stats.get("commentCount", 0)) if "commentCount" in stats else 0
                elif an.status_code == 403:
                    logger.warning("YouTube Analytics 403 — yt-analytics.readonly missing from token; user must reconnect")
                    views    = int(stats.get("viewCount",    0))
                    likes    = int(stats.get("likeCount",    0)) if "likeCount"    in stats else 0
                    comments = int(stats.get("commentCount", 0)) if "commentCount" in stats else 0
                else:
                    views    = int(stats.get("viewCount",    0))
                    likes    = int(stats.get("likeCount",    0)) if "likeCount"    in stats else 0
                    comments = int(stats.get("commentCount", 0)) if "commentCount" in stats else 0
            except Exception as ae:
                logger.warning(f"YouTube Analytics error (non-fatal): {ae}")
                views    = int(stats.get("viewCount",    0))
                likes    = int(stats.get("likeCount",    0)) if "likeCount"    in stats else 0
                comments = int(stats.get("commentCount", 0)) if "commentCount" in stats else 0

            return {
                "status": "live",
                "analytics_source": analytics_source,
                "views":           views,
                "likes":           likes,
                "comments":        comments,
                "shares":          shares,
                "subscribers":     int(stats.get("subscriberCount", 0)) if "subscriberCount" in stats else 0,
                "avg_watch_seconds": avg_watch,
                "minutes_watched": minutes_watched,
                "video_count":     int(stats.get("videoCount", 0)) if "videoCount" in stats else 0,
            }
    except Exception as e:
        logger.error(f"YouTube metrics error: {e}")
        return {"status": "error", "error": str(e)}


async def _fetch_instagram_metrics(access_token: str, ig_user_id: str) -> dict:
    """
    Instagram Graph API — Reels insights.
    Attempt 1: instagram_manage_insights (plays, reach, saved, shares, likes, comments)
    Attempt 2: basic instagram_basic fields (like_count, comments_count) — no advanced scope needed.
    """
    if not access_token or not ig_user_id:
        return {"status": "not_connected"}
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            media = await client.get(
                f"https://graph.facebook.com/v21.0/{ig_user_id}/media",
                params={"access_token": access_token, "fields": "id,media_type,timestamp", "limit": 20},
            )
            if media.status_code != 200:
                return {"status": "error", "error": f"HTTP {media.status_code}"}

            items = media.json().get("data", []) or []
            if not items:
                return {"status": "live", "views": 0, "likes": 0, "comments": 0,
                        "saves": 0, "reach": 0, "shares": 0, "video_count": 0,
                        "analytics_source": None}

            total_views = total_likes = total_comments = 0
            total_saves = total_reach = total_shares = 0
            analytics_source = None
            used_fallback = False

            for item in items[:10]:
                media_type = (item.get("media_type") or "IMAGE").upper()
                # "plays" on IMAGE/CAROUSEL silently kills the insights call
                if media_type in ("VIDEO", "REELS"):
                    metric_str = "plays,reach,saved,shares,comments,likes"
                    view_key   = "plays"
                else:
                    metric_str = "impressions,reach,saved,shares,comments,likes"
                    view_key   = "impressions"

                # ── Attempt 1: instagram_manage_insights ──────────────────────
                ins = await client.get(
                    f"https://graph.facebook.com/v21.0/{item['id']}/insights",
                    params={"access_token": access_token, "metric": metric_str},
                )
                if ins.status_code == 200:
                    analytics_source = "instagram_manage_insights"
                    for m in ins.json().get("data", []) or []:
                        name = m.get("name", "")
                        vals = m.get("values", [])
                        val  = vals[-1].get("value", 0) if vals else m.get("value", 0)
                        if isinstance(val, dict):
                            val = sum(val.values())
                        val = int(val or 0)
                        if name == view_key:       total_views    += val
                        elif name == "likes":      total_likes    += val
                        elif name == "comments":   total_comments += val
                        elif name == "saved":      total_saves    += val
                        elif name == "reach":      total_reach    += val
                        elif name == "shares":     total_shares   += val
                else:
                    # ── Attempt 2: basic media fields (instagram_basic only) ───
                    # like_count and comments_count are always available.
                    # views/plays unavailable without manage_insights.
                    fallback = await client.get(
                        f"https://graph.facebook.com/v21.0/{item['id']}",
                        params={"access_token": access_token,
                                "fields": "like_count,comments_count"},
                    )
                    if fallback.status_code == 200:
                        fb = fallback.json()
                        total_likes    += int(fb.get("like_count")     or 0)
                        total_comments += int(fb.get("comments_count") or 0)
                        used_fallback   = True

            if analytics_source is None and used_fallback:
                analytics_source = "instagram_basic_fallback"

            return {
                "status":           "live",
                "analytics_source": analytics_source,
                "views":            total_views,
                "likes":            total_likes,
                "comments":         total_comments,
                "saves":            total_saves,
                "reach":            total_reach,
                "shares":           total_shares,
                "video_count":      len(items),
            }
    except Exception as e:
        logger.error(f"Instagram metrics error: {e}")
        return {"status": "error", "error": str(e)}


async def _fetch_facebook_metrics(access_token: str, page_id: str) -> dict:
    """
    Facebook Graph API — Page video insights.
    Attempt 1: read_insights scope (total_video_views, reactions, comments, shares)
    Attempt 2: basic video fields (video_views, reactions.summary, comments.summary, shares)
               — available with just a page access token, no advanced scope needed.
    """
    if not access_token or not page_id:
        return {"status": "not_connected"}
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            vids = await client.get(
                f"https://graph.facebook.com/v21.0/{page_id}/videos",
                params={"access_token": access_token, "fields": "id,created_time", "limit": 15},
            )
            if vids.status_code != 200:
                return {"status": "error", "error": f"HTTP {vids.status_code}"}

            videos = vids.json().get("data", []) or []
            if not videos:
                followers = 0
                try:
                    pg = await client.get(
                        f"https://graph.facebook.com/v21.0/{page_id}",
                        params={"access_token": access_token, "fields": "followers_count,fan_count"},
                    )
                    if pg.status_code == 200:
                        pg_data = pg.json()
                        followers = pg_data.get("followers_count") or pg_data.get("fan_count") or 0
                except Exception:
                    pass
                return {"status": "live", "views": 0, "reactions": 0, "comments": 0,
                        "shares": 0, "followers": followers, "video_count": 0,
                        "analytics_source": None}

            total_views = total_reactions = total_comments = total_shares = 0
            analytics_source = None

            for vid in videos[:10]:
                try:
                    got_vid_stats = False

                    # ── Attempt 1: read_insights scope ────────────────────────
                    ins = await client.get(
                        f"https://graph.facebook.com/v21.0/{vid['id']}",
                        params={
                            "access_token": access_token,
                            "fields": "insights.metric(total_video_views,total_video_reactions_by_type_total,total_video_shares,total_video_comments)",
                        },
                    )
                    if ins.status_code == 200:
                        insights_data = ins.json().get("insights", {}).get("data", []) or []
                        if insights_data:
                            analytics_source = "read_insights+pages_read_engagement"
                            for m in insights_data:
                                name = m.get("name", "")
                                vals = m.get("values", [{}])
                                val  = vals[-1].get("value", 0) if vals else 0
                                if isinstance(val, dict):
                                    val = sum(val.values())
                                val = int(val or 0)
                                if   name == "total_video_views":                    total_views     += val
                                elif name == "total_video_reactions_by_type_total":  total_reactions += val
                                elif name == "total_video_shares":                   total_shares    += val
                                elif name == "total_video_comments":                 total_comments  += val
                            got_vid_stats = True

                    # ── Attempt 2: basic video fields (no read_insights needed) ─
                    if not got_vid_stats:
                        fallback = await client.get(
                            f"https://graph.facebook.com/v21.0/{vid['id']}",
                            params={
                                "access_token": access_token,
                                "fields": "video_views,reactions.summary(true),comments.summary(true),shares",
                            },
                        )
                        if fallback.status_code == 200:
                            fb = fallback.json()
                            total_views     += int(fb.get("video_views") or 0)
                            total_reactions += int((fb.get("reactions") or {}).get("summary", {}).get("total_count") or 0)
                            total_comments  += int((fb.get("comments")  or {}).get("summary", {}).get("total_count") or 0)
                            total_shares    += int((fb.get("shares")    or {}).get("count") or 0)
                            if analytics_source is None:
                                analytics_source = "basic_video_fields_fallback"

                except Exception as ve:
                    logger.warning(f"Facebook video insight error for {vid.get('id')} (skipping): {ve}")
                    continue

            followers = 0
            try:
                pg = await client.get(
                    f"https://graph.facebook.com/v21.0/{page_id}",
                    params={"access_token": access_token, "fields": "followers_count,fan_count"},
                )
                if pg.status_code == 200:
                    pg_data   = pg.json()
                    followers = pg_data.get("followers_count") or pg_data.get("fan_count") or 0
            except Exception as fe:
                logger.warning(f"Facebook followers fetch error (non-fatal): {fe}")

            return {
                "status":           "live",
                "analytics_source": analytics_source,
                "views":            total_views,
                "reactions":        total_reactions,
                "comments":         total_comments,
                "shares":           total_shares,
                "followers":        followers,
                "video_count":      len(videos),
            }
    except Exception as e:
        logger.error(f"Facebook metrics error: {e}")
        return {"status": "error", "error": str(e)}


# ============================================================
# DB cache helpers for platform metrics
# ============================================================

async def _platform_metrics_db_cache_get(conn, user_id: str) -> Optional[dict]:
    try:
        row = await conn.fetchrow("SELECT fetched_at, data FROM platform_metrics_cache WHERE user_id = $1", user_id)
        if not row:
            return None
        data = row["data"]
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                data = None
        if not isinstance(data, dict):
            return None
        out = dict(data)
        out["cached"] = True
        out["cache_source"] = "db"
        out["fetched_at"] = row["fetched_at"].isoformat() if row["fetched_at"] else out.get("fetched_at")
        return out
    except Exception:
        return None


async def _platform_metrics_db_cache_set(conn, user_id: str, output: dict) -> None:
    try:
        await conn.execute(
            """
            INSERT INTO platform_metrics_cache (user_id, fetched_at, data)
            VALUES ($1, NOW(), $2::jsonb)
            ON CONFLICT (user_id) DO UPDATE
            SET fetched_at = EXCLUDED.fetched_at,
                data = EXCLUDED.data
            """,
            user_id,
            json.dumps(output),
        )
    except Exception:
        pass


# ============================================================
# Route handlers
# ============================================================

@router.get("/api/analytics")
async def get_analytics(range: str = "30d", user: dict = Depends(get_current_user_readonly)):
    # 'all' maps to 1y (largest supported window). 'partial' = at least one platform succeeded.
    minutes = {"30m": 30, "1h": 60, "6h": 360, "12h": 720, "1d": 1440,
               "7d": 10080, "30d": 43200, "6m": 262800, "1y": 525600, "all": 525600}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    uid = user["id"]
    pool = core.state.db_pool

    # Single connection: previously asyncio.gather ran four acquire()s in parallel per request,
    # which could exhaust the small asyncpg pool (max 10) alongside parallel /api/dashboard/stats
    # and /api/wallet — leaving all three fetches pending in the browser.
    async with pool.acquire() as conn:
        try:
            stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total,
                   SUM(CASE WHEN status IN ('completed','succeeded','partial') THEN 1 ELSE 0 END)::int AS completed,
                   COALESCE(SUM(views), 0)::bigint AS views,
                   COALESCE(SUM(likes), 0)::bigint AS likes,
                   COALESCE(SUM(put_spent), 0)::int AS put_used,
                   COALESCE(SUM(aic_spent), 0)::int AS aic_used
            FROM uploads WHERE user_id = $1 AND created_at >= $2
            """, uid, since)
        except Exception as e:
            if e.__class__.__name__ != "UndefinedColumnError":
                raise
            stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total,
                   SUM(CASE WHEN status IN ('completed','succeeded','partial') THEN 1 ELSE 0 END)::int AS completed,
                   0::bigint AS views, 0::bigint AS likes,
                   0::int AS put_used, 0::int AS aic_used
            FROM uploads WHERE user_id = $1 AND created_at >= $2
            """, uid, since)

        daily = await conn.fetch(
            "SELECT DATE(created_at) AS date, COUNT(*)::int AS uploads "
            "FROM uploads WHERE user_id = $1 AND created_at >= $2 "
            "GROUP BY DATE(created_at) ORDER BY date",
            uid, since,
        )

        platforms = await conn.fetch(
            "SELECT unnest(platforms) AS platform, COUNT(*)::int AS count "
            "FROM uploads WHERE user_id = $1 AND created_at >= $2 "
            "AND status IN ('completed','succeeded','partial') "
            "GROUP BY platform",
            uid, since,
        )

        trill_stats = None
        try:
            trill_data = await conn.fetchrow("""
                SELECT
                    COUNT(*)::int AS trill_uploads,
                    COALESCE(AVG(trill_score), 0)::decimal AS avg_score,
                    COALESCE(MAX(trill_score), 0)::decimal AS max_score,
                    COALESCE(MAX(max_speed_mph), 0)::decimal AS max_speed_mph,
                    COALESCE(SUM(distance_miles), 0)::decimal AS total_distance_miles
                FROM uploads
                WHERE user_id = $1
                AND created_at >= $2
                AND trill_score IS NOT NULL
            """, uid, since)

            if trill_data and trill_data["trill_uploads"] > 0:
                speed_buckets = await conn.fetch("""
                    SELECT speed_bucket, COUNT(*)::int AS count
                    FROM uploads
                    WHERE user_id = $1
                    AND created_at >= $2
                    AND speed_bucket IS NOT NULL
                    GROUP BY speed_bucket
                """, uid, since)

                bucket_counts = {
                    "gloryBoy": 0,
                    "euphoric": 0,
                    "sendIt": 0,
                    "spirited": 0,
                    "chill": 0,
                }

                for bucket in speed_buckets:
                    if bucket["speed_bucket"] in bucket_counts:
                        bucket_counts[bucket["speed_bucket"]] = bucket["count"]

                trill_stats = {
                    "trill_uploads": trill_data["trill_uploads"],
                    "avg_score": float(trill_data["avg_score"]),
                    "max_score": float(trill_data["max_score"]),
                    "max_speed_mph": float(trill_data["max_speed_mph"]),
                    "total_distance_miles": float(trill_data["total_distance_miles"]),
                    "speed_buckets": bucket_counts,
                }
        except Exception as e:
            logger.warning("Trill stats unavailable: %s", e)
            trill_stats = None

    result = {
        "total_uploads": stats["total"] if stats else 0,
        "completed": stats["completed"] if stats else 0,
        "views": stats["views"] if stats else 0,
        "likes": stats["likes"] if stats else 0,
        "put_used": stats["put_used"] if stats else 0,
        "aic_used": stats["aic_used"] if stats else 0,
        "daily": [{"date": str(d["date"]), "uploads": d["uploads"]} for d in daily],
        "platforms": {p["platform"]: p["count"] for p in platforms}
    }

    if trill_stats:
        result["trill"] = trill_stats

    return result


@router.get("/api/analytics/pikzels-v2-usage")
async def analytics_pikzels_v2_usage(range: str = Query("30d"), user: dict = Depends(get_current_user)):
    """Thumbnail Studio (Pikzels API v2) action counts for the signed-in user (mirrors admin KPI shape)."""
    since, until = parse_range_since_until(range)
    try:
        async with core.state.db_pool.acquire() as conn:
            data = await fetch_user_pikzels_studio_usage(conn, str(user["id"]), since, until)
        return {"range": range, **data}
    except Exception as e:
        logger.warning("analytics pikzels-v2-usage: %s", e)
        return {"range": range, "total_calls": 0, "by_operation": []}


@router.get("/api/analytics/upload-counts-by-token")
async def analytics_upload_counts_by_token(user: dict = Depends(get_current_user)):
    """
    Completed uploads explicitly targeted at platform_tokens rows, keyed for CRM deep-dive:
    { by_platform: { tiktok: { "<token_uuid>": n, ... }, ... } }.
    Uploads with empty target_accounts are omitted (not attributed per connection).
    """
    uid = user["id"]
    try:
        async with core.state.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT lower(t.platform) AS platform, t.id::text AS token_id, COUNT(*)::int AS cnt
                FROM uploads u
                CROSS JOIN LATERAL unnest(COALESCE(u.target_accounts, ARRAY[]::text[])) AS ta(token_id)
                INNER JOIN platform_tokens t
                  ON t.user_id = u.user_id AND t.id = ta.token_id::uuid
                WHERE u.user_id = $1
                  AND u.status IN ('completed', 'succeeded', 'partial')
                  AND COALESCE(cardinality(u.target_accounts), 0) > 0
                GROUP BY t.platform, t.id
                """,
                uid,
            )
    except Exception as e:
        logger.warning("upload-counts-by-token: %s", e)
        return {"by_platform": {}}

    by_platform: dict = {}
    for r in rows:
        p = (r["platform"] or "").strip().lower()
        if not p:
            continue
        by_platform.setdefault(p, {})[r["token_id"]] = int(r["cnt"] or 0)
    return {"by_platform": by_platform}


async def _compute_live_platform_metrics(user: dict) -> dict:
    """Live platform API fetch; updates in-memory + DB cache."""
    user_id = str(user["id"])
    now = _time.time()
    async with core.state.db_pool.acquire() as conn:
        token_rows = await conn.fetch(
            "SELECT id, platform, token_blob, account_id FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
            user["id"],
        )
        upload_counts = await conn.fetch(
            """SELECT unnest(platforms) AS platform, COUNT(*)::int AS cnt
               FROM uploads
               WHERE user_id = $1 AND status IN ('succeeded', 'completed', 'partial')
               GROUP BY platform""",
            user["id"],
        )

    upload_map = {r["platform"]: r["cnt"] for r in upload_counts}

    from services.platform_oauth_refresh import refresh_decrypted_token_for_row

    token_map: dict = {}
    for row in token_rows:
        plat = row["platform"]
        blob = row["token_blob"]
        if not blob:
            continue
        try:
            decrypted = decrypt_blob(blob)
        except Exception:
            continue
        if decrypted:
            if plat == "instagram" and not decrypted.get("ig_user_id") and row["account_id"]:
                decrypted["ig_user_id"] = str(row["account_id"])
            if plat == "facebook" and not decrypted.get("page_id") and row["account_id"]:
                decrypted["page_id"] = str(row["account_id"])
            decrypted = await refresh_decrypted_token_for_row(
                plat,
                decrypted,
                db_pool=core.state.db_pool,
                user_id=user_id,
                token_row_id=str(row["id"]),
            )
            token_map[plat] = decrypted

    async def run_tiktok():
        t = token_map.get("tiktok", {})
        return await _fetch_tiktok_metrics(t.get("access_token", ""))

    async def run_youtube():
        t = token_map.get("youtube", {})
        return await _fetch_youtube_metrics(t.get("access_token", ""))

    async def run_instagram():
        t = token_map.get("instagram", {})
        ig_id = (t.get("ig_user_id") or t.get("instagram_user_id") or t.get("instagram_page_id") or "")
        return await _fetch_instagram_metrics(t.get("access_token", ""), ig_id)

    async def run_facebook():
        t = token_map.get("facebook", {})
        page_id = (t.get("page_id") or t.get("facebook_page_id") or t.get("fb_page_id") or "")
        return await _fetch_facebook_metrics(t.get("access_token", ""), page_id)

    tasks = {}
    if "tiktok" in token_map:
        tasks["tiktok"] = run_tiktok()
    if "youtube" in token_map:
        tasks["youtube"] = run_youtube()
    if "instagram" in token_map:
        tasks["instagram"] = run_instagram()
    if "facebook" in token_map:
        tasks["facebook"] = run_facebook()

    platforms_result: dict = {}
    if tasks:
        results = await _asyncio.gather(*tasks.values(), return_exceptions=True)
        for plat, res in zip(tasks.keys(), results):
            platforms_result[plat] = {"status": "error", "error": str(res)} if isinstance(res, Exception) else res

    for plat in ["tiktok", "youtube", "instagram", "facebook"]:
        if plat not in platforms_result:
            platforms_result[plat] = {"status": "not_connected"}
        platforms_result[plat]["uploads"] = upload_map.get(plat, 0)

    output = {
        "platforms": platforms_result,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "cached": False,
        "cache_age_minutes": 0,
        "next_refresh_minutes": int(_PLATFORM_CACHE_TTL / 60),
    }

    _platform_metrics_cache[user_id] = {"fetched_at": now, "data": output}
    async with core.state.db_pool.acquire() as conn:
        await _platform_metrics_db_cache_set(conn, user_id, output)

    return output


@router.get("/api/analytics/platform-metrics")
async def get_platform_metrics(force: bool = False, user: dict = Depends(get_current_user)):
    """
    Fetch live engagement metrics from all connected platform APIs.
    Cached 3 hours per user (memory + DB). Pass ?force=true to bypass cache.
    """
    user_id = str(user["id"])
    now = _time.time()

    # 1) in-memory cache
    cached = _platform_metrics_cache.get(user_id)
    if cached and not force:
        age = now - cached["fetched_at"]
        if age < _PLATFORM_CACHE_TTL:
            result = dict(cached["data"])
            result["cached"] = True
            result["cache_source"] = "memory"
            result["cache_age_minutes"] = int(age / 60)
            result["next_refresh_minutes"] = int((_PLATFORM_CACHE_TTL - age) / 60)
            return result

    # 2) DB cache (survives restarts)
    async with core.state.db_pool.acquire() as conn:
        db_cached = await _platform_metrics_db_cache_get(conn, user_id)
        if db_cached and not force:
            _platform_metrics_cache[user_id] = {"fetched_at": now, "data": db_cached}
            return db_cached

    return await _compute_live_platform_metrics(user)


@router.post("/api/analytics/refresh-all")
async def analytics_refresh_all(
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(True),
    user: dict = Depends(get_current_user),
):
    """Invalidate platform-metrics caches and optionally re-fetch in the background."""
    user_id = str(user["id"])
    _platform_metrics_cache.pop(user_id, None)
    async with core.state.db_pool.acquire() as conn:
        await conn.execute("DELETE FROM platform_metrics_cache WHERE user_id = $1", user["id"])

    if async_mode:
        ucopy = {**user}
        background_tasks.add_task(_compute_live_platform_metrics, ucopy)
        return {"ok": True, "async_mode": True}

    await _compute_live_platform_metrics(user)
    return {"ok": True, "async_mode": False}


@router.get("/api/analytics/platform-metrics/cached")
async def get_platform_metrics_cached(user: dict = Depends(get_current_user)):
    """Return DB-cached platform metrics only (no live API calls)."""
    user_id = str(user["id"])
    async with core.state.db_pool.acquire() as conn:
        cached = await _platform_metrics_db_cache_get(conn, user_id)
    if cached:
        return cached
    return {"platforms": {}, "cached": True, "cache_source": "db", "fetched_at": None}

@router.get("/api/exports/excel")
async def export_excel(type: str = "uploads", range: str = "30d", user: dict = Depends(get_current_user)):
    plan = get_plan(user.get("subscription_tier", "free"))
    if not plan.get("excel"): raise HTTPException(403, "Excel export requires Studio+ plan")

    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)

    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT filename, platforms, title, status, views, likes, put_spent, aic_spent, created_at FROM uploads WHERE user_id = $1 AND created_at >= $2 ORDER BY created_at DESC", user["id"], since)

    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Uploads"
        headers = ["Filename", "Platforms", "Title", "Status", "Views", "Likes", "PUT", "AIC", "Created"]
        ws.append(headers)
        for r in rows:
            ws.append([r["filename"], ",".join(r["platforms"] or []), r["title"], r["status"], r["views"], r["likes"], r["put_spent"], r["aic_spent"], str(r["created_at"])])
        output = BytesIO()
        wb.save(output)
        output.seek(0)
        return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename=uploadm8_exports.xlsx"})
    except ImportError:
        raise HTTPException(500, "Excel export not available")

@router.get("/api/analytics/overview")
async def analytics_overview(days: int = Query(30, ge=1, le=3650), user: dict = Depends(get_current_user)):
    """High-level KPI summary for analytics dashboard."""
    since = _now_utc() - timedelta(days=days)

    async with core.state.db_pool.acquire() as conn:
        # Upload KPIs (defensive against older schemas)
        try:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*)::int AS uploads_total,
                    SUM(CASE WHEN status IN ('completed','succeeded') THEN 1 ELSE 0 END)::int AS uploads_completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS uploads_failed,
                    COALESCE(AVG(EXTRACT(EPOCH FROM (processing_finished_at - processing_started_at))), 0)::double precision AS avg_processing_seconds,
                    COALESCE(SUM(views), 0)::bigint AS views_total,
                    COALESCE(SUM(likes), 0)::bigint AS likes_total,
                    COALESCE(SUM(cost_attributed), 0)::double precision AS cost_total
                FROM uploads
                WHERE user_id = $1 AND created_at >= $2
                """,
                user["id"], since
            )
        except Exception as e:
            if e.__class__.__name__ != "UndefinedColumnError":
                raise
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*)::int AS uploads_total,
                    SUM(CASE WHEN status IN ('completed','succeeded') THEN 1 ELSE 0 END)::int AS uploads_completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS uploads_failed,
                    0::double precision AS avg_processing_seconds,
                    0::bigint AS views_total,
                    0::bigint AS likes_total,
                    0::double precision AS cost_total
                FROM uploads
                WHERE user_id = $1 AND created_at >= $2
                """,
                user["id"], since
            )

        # Revenue (optional)
        revenue_total = 0.0
        try:
            rev = await conn.fetchval(
                "SELECT COALESCE(SUM(amount), 0)::decimal FROM revenue_tracking WHERE user_id = $1 AND created_at >= $2",
                user["id"], since
            )
            revenue_total = float(rev or 0)
        except Exception as e:
            if e.__class__.__name__ != "UndefinedTableError":
                raise

    return {
        "range_days": days,
        "since": since.isoformat(),
        "uploads": {
            "total": int(row["uploads_total"] or 0),
            "completed": int(row["uploads_completed"] or 0),
            "failed": int(row["uploads_failed"] or 0),
            "avg_processing_seconds": float(row["avg_processing_seconds"] or 0),
        },
        "engagement": {
            "views": int(row["views_total"] or 0),
            "likes": int(row["likes_total"] or 0),
        },
        "costs": {
            "cost_total": float(row["cost_total"] or 0),
        },
        "revenue": {
            "revenue_total": revenue_total,
        },
    }


def _normalize_quality_scores_platform(platform: str) -> Optional[str]:
    """Match kpi.html platform filter; None means all platforms."""
    s = (platform or "all").strip().lower()
    if not s or s == "all":
        return None
    if s == "instagram_reels":
        s = "instagram"
    if s == "facebook_reels":
        s = "facebook"
    if s in ("tiktok", "youtube", "instagram", "facebook"):
        return s
    return None


@router.get("/api/analytics/quality-scores")
async def analytics_quality_scores(
    days: int = Query(30, ge=1, le=3650),
    platform: str = Query("all"),
    user: dict = Depends(get_current_user),
):
    """
    Per-strategy engagement rollups from ``upload_quality_scores_daily`` (ML scoring job).
    Used by kpi.html for the “ML strategy” strip; shape: ``{ rows: [...], days, platform }``.
    """
    uid = user["id"]
    lookback = max(1, min(int(days), 3650))
    pf = _normalize_quality_scores_platform(platform)

    sql_all = """
        SELECT strategy_key,
               SUM(samples)::bigint AS samples,
               CASE WHEN SUM(samples) > 0 THEN
                 SUM(COALESCE(mean_engagement, 0) * samples::double precision)
                 / SUM(samples::double precision)
               ELSE 0.0 END AS mean_engagement,
               MAX(ci95_high)::double precision AS ci95_high
          FROM upload_quality_scores_daily
         WHERE user_id = $1::uuid
           AND day >= (CURRENT_DATE - $2::int)
         GROUP BY strategy_key
        HAVING SUM(samples) > 0
         ORDER BY MAX(ci95_high) DESC NULLS LAST
         LIMIT 80
    """
    sql_pf = """
        SELECT strategy_key,
               SUM(samples)::bigint AS samples,
               CASE WHEN SUM(samples) > 0 THEN
                 SUM(COALESCE(mean_engagement, 0) * samples::double precision)
                 / SUM(samples::double precision)
               ELSE 0.0 END AS mean_engagement,
               MAX(ci95_high)::double precision AS ci95_high
          FROM upload_quality_scores_daily
         WHERE user_id = $1::uuid
           AND day >= (CURRENT_DATE - $2::int)
           AND platform = $3
         GROUP BY strategy_key
        HAVING SUM(samples) > 0
         ORDER BY MAX(ci95_high) DESC NULLS LAST
         LIMIT 80
    """

    async with core.state.db_pool.acquire() as conn:
        try:
            if pf is None:
                rows = await conn.fetch(sql_all, uid, lookback)
            else:
                rows = await conn.fetch(sql_pf, uid, lookback, pf)
        except Exception as e:
            if e.__class__.__name__ == "UndefinedTableError":
                return {"rows": [], "days": lookback, "platform": platform}
            raise

    out = []
    for r in rows or []:
        out.append(
            {
                "strategy_key": str(r["strategy_key"] or ""),
                "samples": int(r["samples"] or 0),
                "mean_engagement": float(r["mean_engagement"] or 0),
                "ci95_high": float(r["ci95_high"] or 0),
            }
        )
    return {"rows": out, "days": lookback, "platform": platform}


@router.get("/api/analytics/my-avg-processing")
async def my_avg_processing(user: dict = Depends(get_current_user)):
    """Return this user's personal average processing time in seconds.
    Used by upload.html to calibrate the progress estimate instead of
    using a hardcoded 7-minute fallback.
    Falls back to 420s (7 min) when fewer than 3 completed uploads exist.
    """
    try:
        async with core.state.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) AS sample_count,
                    COALESCE(
                        AVG(EXTRACT(EPOCH FROM (processing_finished_at - processing_started_at))),
                        420
                    )::double precision AS avg_seconds
                FROM uploads
                WHERE user_id      = $1
                  AND status       IN ('succeeded', 'completed', 'partial')
                  AND processing_started_at  IS NOT NULL
                  AND processing_finished_at IS NOT NULL
                  AND processing_finished_at > processing_started_at
                """,
                user["id"],
            )
        sample_count = int(row["sample_count"] or 0)
        avg_seconds  = float(row["avg_seconds"] or 420)
        # Only trust personal average if we have at least 3 data points
        if sample_count < 3:
            avg_seconds = 420.0
        return {
            "avg_processing_seconds": round(avg_seconds, 1),
            "sample_count": sample_count,
            "reliable": sample_count >= 3,
        }
    except Exception as e:
        logger.warning(f"my_avg_processing failed: {e}")
        return {"avg_processing_seconds": 420.0, "sample_count": 0, "reliable": False}


@router.get("/api/analytics/export")
async def analytics_export(days: int = Query(30, ge=1, le=3650), format: str = Query("csv"), user: dict = Depends(get_current_user)):
    """Export analytics for the last N days as CSV (default) or JSON."""
    since = _now_utc() - timedelta(days=days)

    async with core.state.db_pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT
                    id, filename, title, caption, platforms, privacy, status,
                    created_at, completed_at,
                    COALESCE(views, 0)::bigint AS views,
                    COALESCE(likes, 0)::bigint AS likes,
                    COALESCE(comments, 0)::bigint AS comments,
                    COALESCE(shares, 0)::bigint AS shares,
                    COALESCE(cost_attributed, 0)::double precision AS cost_attributed,
                    video_url
                FROM uploads
                WHERE user_id = $1 AND created_at >= $2
                ORDER BY created_at DESC
                """,
                user["id"], since
            )
        except Exception as e:
            if e.__class__.__name__ != "UndefinedColumnError":
                raise
            # Older schema fallback
            rows = await conn.fetch(
                """
                SELECT
                    id, filename, title, caption, platforms, privacy, status,
                    created_at, completed_at,
                    0::bigint AS views,
                    0::bigint AS likes,
                    0::bigint AS comments,
                    0::bigint AS shares,
                    0::double precision AS cost_attributed,
                    video_url
                FROM uploads
                WHERE user_id = $1 AND created_at >= $2
                ORDER BY created_at DESC
                """,
                user["id"], since
            )

    data = [
        {
            "id": str(r["id"]),
            "filename": r["filename"],
            "title": r["title"],
            "caption": r["caption"],
            "platforms": list(r["platforms"]) if r["platforms"] else [],
            "privacy": r["privacy"],
            "status": r["status"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "completed_at": r["completed_at"].isoformat() if r["completed_at"] else None,
            "views": int(r["views"] or 0),
            "likes": int(r["likes"] or 0),
            "comments": int(r["comments"] or 0),
            "shares": int(r["shares"] or 0),
            "cost_attributed": float(r["cost_attributed"] or 0),
            "video_url": r.get("video_url"),
        }
        for r in rows
    ]

    if format.lower() == "json":
        return {"range_days": days, "since": since.isoformat(), "rows": data}

    # CSV default
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "id","filename","title","caption","platforms","privacy","status",
            "created_at","completed_at",
            "views","likes","comments","shares",
            "cost_attributed","video_url",
        ],
    )
    writer.writeheader()
    for item in data:
        item = dict(item)
        item["platforms"] = ",".join(item.get("platforms") or [])
        writer.writerow(item)

    csv_bytes = output.getvalue().encode("utf-8")
    headers = {"Content-Disposition": f'attachment; filename="uploadm8-analytics-{days}d.csv"'}
    return Response(content=csv_bytes, media_type="text/csv", headers=headers)
