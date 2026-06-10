"""Upload analytics and thumbnail sync routes."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple

import asyncpg
import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

import core.state
from core.auth import decrypt_blob
from core.deps import get_current_user
from core.helpers import _safe_json
from services.platform_oauth_refresh import refresh_decrypted_token_for_row
from services.platform_posted_thumbnails import (
    background_sync_posted_thumbnails,
    upload_ids_needing_posted_thumbnail_sync,
)
from services.sync_analytics_helpers import resolve_token_candidates_for_platform_result
from services.uploads_handlers import poll_upload_thumbnails_payload

logger = logging.getLogger("uploadm8-api")

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

_sync_analytics_running: set[str] = set()


async def _fetch_platform_video_engagement(
    client: httpx.AsyncClient,
    plat: str,
    video_id: str,
    pr: dict,
    access_token: str,
) -> Optional[Dict[str, int]]:
    """Call the platform metrics API for one video/reel/post."""
    if not access_token:
        return None
    try:
        if plat == "tiktok" and video_id:
            resp = await client.post(
                "https://open.tiktokapis.com/v2/video/query/",
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                params={"fields": "id,view_count,like_count,comment_count,share_count"},
                json={"filters": {"video_ids": [str(video_id)]}},
            )
            if resp.status_code != 200:
                return None
            vids = resp.json().get("data", {}).get("videos", []) or []
            if not vids:
                return None
            v = vids[0]
            return {
                "views": int(v.get("view_count") or 0),
                "likes": int(v.get("like_count") or 0),
                "comments": int(v.get("comment_count") or 0),
                "shares": int(v.get("share_count") or 0),
            }

        if plat == "youtube" and video_id:
            resp = await client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={"part": "statistics", "id": str(video_id)},
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code != 200:
                return None
            items = resp.json().get("items", []) or []
            if not items:
                return None
            st = items[0].get("statistics", {})
            return {
                "views": int(st.get("viewCount") or 0),
                "likes": int(st.get("likeCount") or 0),
                "comments": int(st.get("commentCount") or 0),
                "shares": 0,
            }

        if plat == "instagram" and video_id:
            media_id = pr.get("platform_video_id") or pr.get("media_id") or video_id
            resp = await client.get(
                f"https://graph.facebook.com/v21.0/{media_id}/insights",
                params={
                    "access_token": access_token,
                    "metric": "views,plays,likes,comments,saved,shares,reach",
                },
            )
            if resp.status_code != 200:
                return None
            s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
            ig_views = ig_plays = 0
            for m in resp.json().get("data", []) or []:
                name = m.get("name", "")
                vals = m.get("values", [])
                val = int(vals[-1].get("value", 0) if vals else m.get("value", 0) or 0)
                if name == "views":
                    ig_views = val
                elif name == "plays":
                    ig_plays = val
                elif name == "likes":
                    s["likes"] += val
                elif name == "comments":
                    s["comments"] += val
                elif name == "shares":
                    s["shares"] += val
            s["views"] = ig_views or ig_plays
            return s

        if plat == "facebook" and video_id:
            resp = await client.get(
                f"https://graph.facebook.com/v21.0/{video_id}",
                params={
                    "access_token": access_token,
                    "fields": "insights.metric(total_video_views,total_video_reactions_by_type_total,total_video_comments,total_video_shares)",
                },
            )
            if resp.status_code != 200:
                return None
            s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
            for m in resp.json().get("insights", {}).get("data", []) or []:
                name = m.get("name", "")
                vals = m.get("values", [{}])
                val = vals[-1].get("value", 0) if vals else 0
                if isinstance(val, dict):
                    val = sum(val.values())
                val = int(val or 0)
                if name == "total_video_views":
                    s["views"] += val
                elif name == "total_video_reactions_by_type_total":
                    s["likes"] += val
                elif name == "total_video_comments":
                    s["comments"] += val
                elif name == "total_video_shares":
                    s["shares"] += val
            return s
    except Exception as e:
        logger.warning("sync-analytics fetch %s/%s: %s", plat, video_id, e)
        return None
    return None


def _merge_stats_into_platform_result(pr: dict, s: Dict[str, int]) -> None:
    pr["views"] = s["views"]
    pr["view_count"] = s["views"]
    pr["likes"] = s["likes"]
    pr["like_count"] = s["likes"]
    pr["comments"] = s["comments"]
    pr["comment_count"] = s["comments"]
    pr["shares"] = s["shares"]
    pr["share_count"] = s["shares"]


def _plat_token_resolution_maps(
    token_rows: list,
    token_map_by_id: Dict[str, dict],
    token_map_by_platform: Dict[str, dict],
) -> Tuple[Dict[Tuple[str, str], dict], Dict[Tuple[str, str], Tuple[str, dict]], Dict[str, List[Tuple[str, dict]]]]:
    token_map_by_plat_account: Dict[Tuple[str, str], dict] = {}
    plat_account_row_map: Dict[Tuple[str, str], Tuple[str, dict]] = {}
    platform_token_rows: Dict[str, List[Tuple[str, dict]]] = {}
    for tr in token_rows:
        tid = str(tr["id"])
        dec = token_map_by_id.get(tid)
        if not dec:
            continue
        plat = str(tr.get("platform") or "").lower()
        aid = tr.get("account_id")
        if aid is not None and str(aid).strip() != "":
            a = str(aid).strip()
            token_map_by_plat_account[(plat, a)] = dec
            plat_account_row_map[(plat, a)] = (tid, dec)
        platform_token_rows.setdefault(plat, []).append((tid, dec))
    return token_map_by_plat_account, plat_account_row_map, platform_token_rows


async def _warm_user_platform_oauth_tokens(user_id: str) -> None:
    """Refresh each connected platform token once per batch (cached in platform_oauth_refresh)."""
    uid = str(user_id)
    async with core.state.db_pool.acquire() as conn:
        token_rows = await conn.fetch(
            "SELECT id, platform, token_blob, account_id FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
            uid,
        )
    for tr in token_rows:
        try:
            dec = decrypt_blob(tr["token_blob"])
            if not dec:
                continue
            if tr["platform"] == "instagram" and not dec.get("ig_user_id") and tr["account_id"]:
                dec["ig_user_id"] = str(tr["account_id"])
            if tr["platform"] == "facebook" and not dec.get("page_id") and tr["account_id"]:
                dec["page_id"] = str(tr["account_id"])
            await refresh_decrypted_token_for_row(
                tr["platform"],
                dec,
                db_pool=core.state.db_pool,
                user_id=uid,
                token_row_id=str(tr["id"]),
            )
        except Exception:
            pass


async def _sync_upload_analytics_core(
    user: dict,
    upload_id: str,
    *,
    skip_token_refresh: bool = False,
) -> dict:
    """Shared implementation for per-upload analytics sync."""
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, platforms, platform_results, status FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id,
            user["id"],
        )
    if not row:
        raise HTTPException(404, "Upload not found")

    if row["status"] not in ("completed", "succeeded", "partial"):
        return {"synced": False, "reason": "not_completed", "views": 0, "likes": 0, "comments": 0, "shares": 0}

    raw_pr = _safe_json(row["platform_results"], [])
    pr_list = []
    if isinstance(raw_pr, list):
        pr_list = [x for x in raw_pr if isinstance(x, dict)]
    elif isinstance(raw_pr, dict):
        pr_list = [{"platform": k, **v} if isinstance(v, dict) else {"platform": k} for k, v in raw_pr.items()]
    for pr in pr_list:
        if pr.get("platform_video_id") and not pr.get("video_id"):
            pr["video_id"] = pr["platform_video_id"]
        if pr.get("platform_url") and not pr.get("url"):
            pr["url"] = pr["platform_url"]

    async with core.state.db_pool.acquire() as conn:
        token_rows = await conn.fetch(
            "SELECT id, platform, token_blob, account_id FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
            user["id"],
        )

    token_map_by_id = {}
    token_map_by_platform = {}
    uid = str(user["id"])
    for tr in token_rows:
        try:
            dec = decrypt_blob(tr["token_blob"])
            if dec:
                if tr["platform"] == "instagram" and not dec.get("ig_user_id") and tr["account_id"]:
                    dec["ig_user_id"] = str(tr["account_id"])
                if tr["platform"] == "facebook" and not dec.get("page_id") and tr["account_id"]:
                    dec["page_id"] = str(tr["account_id"])
                token_id = str(tr["id"])
                if not skip_token_refresh:
                    dec = await refresh_decrypted_token_for_row(
                        tr["platform"],
                        dec,
                        db_pool=core.state.db_pool,
                        user_id=uid,
                        token_row_id=token_id,
                    )
                token_map_by_id[token_id] = dec
                plat_norm = str(tr.get("platform") or "").lower()
                if plat_norm:
                    token_map_by_platform[plat_norm] = dec
        except Exception:
            pass

    token_map_by_plat_account, plat_account_row_map, platform_token_rows = _plat_token_resolution_maps(
        list(token_rows), token_map_by_id, token_map_by_platform
    )

    total_views = total_likes = total_comments = total_shares = 0
    platform_stats: Dict[str, Dict[str, int]] = {}
    rows_with_video_id = 0
    fetched_any = False

    async with httpx.AsyncClient(timeout=20) as client:
        for pr in pr_list:
            plat = str(pr.get("platform") or "").lower()
            video_id = (
                pr.get("platform_video_id")
                or pr.get("video_id")
                or pr.get("videoId")
                or pr.get("id")
                or pr.get("media_id")
                or pr.get("post_id")
                or pr.get("share_id")
            )
            if not video_id:
                continue
            rows_with_video_id += 1

            candidates = resolve_token_candidates_for_platform_result(
                pr,
                token_map_by_id,
                token_map_by_plat_account,
                token_map_by_platform,
                plat_account_row_map=plat_account_row_map,
                platform_token_rows=platform_token_rows,
            )
            if not candidates:
                continue

            s: Optional[Dict[str, int]] = None
            for tok in candidates:
                at = (tok or {}).get("access_token", "")
                s = await _fetch_platform_video_engagement(client, plat, str(video_id), pr, at)
                if s is not None:
                    break

            if not s:
                continue

            _merge_stats_into_platform_result(pr, s)
            fetched_any = True
            total_views += s["views"]
            total_likes += s["likes"]
            total_comments += s["comments"]
            total_shares += s["shares"]
            prev = platform_stats.get(plat)
            if prev:
                platform_stats[plat] = {
                    "views": prev["views"] + s["views"],
                    "likes": prev["likes"] + s["likes"],
                    "comments": prev["comments"] + s["comments"],
                    "shares": prev["shares"] + s["shares"],
                }
            else:
                platform_stats[plat] = dict(s)

    async with core.state.db_pool.acquire() as conn:
        if pr_list:
            pr_json = json.dumps(pr_list)
            await conn.execute(
                """UPDATE uploads SET views=$1, likes=$2, comments=$3, shares=$4,
                       platform_results = $7::jsonb,
                       analytics_synced_at=NOW(), updated_at=NOW()
                   WHERE id=$5 AND user_id=$6""",
                total_views,
                total_likes,
                total_comments,
                total_shares,
                upload_id,
                user["id"],
                pr_json,
            )
        else:
            await conn.execute(
                """UPDATE uploads SET views=$1, likes=$2, comments=$3, shares=$4,
                       analytics_synced_at=NOW(), updated_at=NOW()
                   WHERE id=$5 AND user_id=$6""",
                total_views,
                total_likes,
                total_comments,
                total_shares,
                upload_id,
                user["id"],
            )

    if not rows_with_video_id:
        return {
            "synced": False,
            "reason": "no_platform_video_ids",
            "views": total_views,
            "likes": total_likes,
            "comments": total_comments,
            "shares": total_shares,
            "platform_stats": platform_stats,
        }
    if not fetched_any:
        return {
            "synced": False,
            "reason": "no_tokens_or_metrics",
            "message": "No working OAuth token matched this upload, or platforms returned no data.",
            "views": total_views,
            "likes": total_likes,
            "comments": total_comments,
            "shares": total_shares,
            "platform_stats": platform_stats,
        }

    return {
        "synced": True,
        "views": total_views,
        "likes": total_likes,
        "comments": total_comments,
        "shares": total_shares,
        "platform_stats": platform_stats,
    }


async def _background_sync_uploads_analytics(user_id: str, upload_ids: list[str]) -> None:
    uid = str(user_id)
    if uid in _sync_analytics_running:
        logger.info("sync-analytics/all: skip duplicate batch for user %s", uid[:8])
        return
    _sync_analytics_running.add(uid)
    try:
        try:
            await _warm_user_platform_oauth_tokens(uid)
        except Exception as e:
            logger.warning("sync-analytics/all token warm user=%s: %s", uid[:8], e)
        user_stub = {"id": uid}
        for up_id in upload_ids:
            try:
                await _sync_upload_analytics_core(user_stub, up_id, skip_token_refresh=True)
            except HTTPException:
                pass
            except Exception as e:
                logger.warning("sync-analytics/all upload=%s: %s", up_id, e)
            await asyncio.sleep(0.35)
    finally:
        _sync_analytics_running.discard(uid)


async def _background_sync_uploads_thumbnails(user_id: str, upload_ids: list[str]) -> None:
    await background_sync_posted_thumbnails(core.state.db_pool, user_id, upload_ids)


@router.post("/sync-analytics/all")
async def sync_all_upload_analytics(
    background_tasks: BackgroundTasks,
    max_uploads: int = Query(800, ge=1, le=2000),
    async_mode: bool = Query(True),
    user: dict = Depends(get_current_user),
):
    """Batch engagement sync for many completed uploads."""
    uid = str(user["id"])
    try:
        async with core.state.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id FROM uploads
                WHERE user_id = $1::uuid
                  AND status = ANY($2::varchar[])
                  AND platform_results IS NOT NULL
                  AND platform_results::text NOT IN ('null', '[]', '{}')
                ORDER BY analytics_synced_at ASC NULLS FIRST, created_at DESC
                LIMIT $3
                """,
                uid,
                ["completed", "succeeded", "partial"],
                max_uploads,
            )
    except (TimeoutError, OSError, asyncio.TimeoutError, asyncpg.PostgresConnectionError) as e:
        logger.warning("sync-analytics/all: database unavailable user=%s: %s", uid[:8], e)
        raise HTTPException(503, "Analytics sync temporarily unavailable — try again shortly") from e
    ids = [str(r["id"]) for r in rows]

    if async_mode:
        background_tasks.add_task(_background_sync_uploads_analytics, uid, ids)
        return {"ok": True, "queued": len(ids), "async_mode": True}

    synced = 0
    for up_id in ids:
        try:
            await _sync_upload_analytics_core(user, up_id)
            synced += 1
        except HTTPException:
            pass
        await asyncio.sleep(0.25)
    return {"ok": True, "candidates": len(ids), "synced": synced, "async_mode": False}


@router.post("/sync-thumbnails/all")
async def sync_all_upload_thumbnails(
    background_tasks: BackgroundTasks,
    max_uploads: int = Query(120, ge=1, le=400),
    async_mode: bool = Query(True),
    user: dict = Depends(get_current_user),
):
    """Queue live platform cover fetch for completed uploads."""
    uid = str(user["id"])
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, status, platform_results, thumbnail_r2_key, output_artifacts, platforms
            FROM uploads
            WHERE user_id = $1::uuid
              AND status = ANY($2::varchar[])
              AND platform_results IS NOT NULL
              AND platform_results::text NOT IN ('null', '[]', '{}')
            ORDER BY updated_at DESC
            LIMIT $3
            """,
            uid,
            ["completed", "succeeded", "partial"],
            max_uploads,
        )
    ids = [str(r["id"]) for r in rows if upload_ids_needing_posted_thumbnail_sync(dict(r))]

    if async_mode:
        if ids:
            background_tasks.add_task(_background_sync_uploads_thumbnails, uid, ids)
        return {"ok": True, "queued": len(ids), "async_mode": True}

    await background_sync_posted_thumbnails(core.state.db_pool, uid, ids)
    return {"ok": True, "synced": len(ids), "async_mode": False}


@router.get("/thumbnails/poll")
async def poll_upload_thumbnails(
    ids: str = Query(..., description="Comma-separated upload UUIDs (max 40)"),
    user: dict = Depends(get_current_user),
):
    """Fast DB-only read of thumbnail URLs for visible rows."""
    raw = [x.strip() for x in (ids or "").split(",") if x.strip()]
    upload_ids = raw[:40]
    if not upload_ids:
        return {"thumbnails": {}}
    payload = await poll_upload_thumbnails_payload(core.state.db_pool, str(user["id"]), upload_ids)
    return {"thumbnails": payload}


@router.post("/{upload_id}/sync-analytics")
async def sync_upload_analytics(upload_id: str, user: dict = Depends(get_current_user)):
    """Fetch latest engagement stats for a single completed upload from platform APIs."""
    return await _sync_upload_analytics_core(user, upload_id)
