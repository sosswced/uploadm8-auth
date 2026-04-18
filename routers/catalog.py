"""Unified content catalog API (/api/catalog/*)."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Request

import core.state
from core.config import R2_BUCKET_NAME
from core.deps import get_current_user, get_current_user_readonly
from core.r2 import _normalize_r2_key, get_s3_client
from services.catalog_sync import get_catalog_aggregate, sync_catalog_for_user
from services.platform_metrics_ui import parse_iso_ts
from services.upload_engagement import title_and_metrics_from_upload_platform_results

router = APIRouter(tags=["catalog"])
logger = logging.getLogger("uploadm8-api")


async def _session_user(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    return await get_current_user(request, authorization)


async def _session_user_readonly(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    return await get_current_user_readonly(request, authorization)


@router.post("/api/catalog/sync")
async def trigger_catalog_sync(
    background_tasks: BackgroundTasks,
    force_full: bool = Query(False, description="Clear cursors and re-scan from the beginning"),
    async_mode: bool = Query(True),
    user: dict = Depends(_session_user),
):
    """
    Trigger a catalog sync for the current user.
    async_mode=true (default): queues in background, returns immediately.
    async_mode=false:          runs synchronously, returns totals (max 30 s).
    """
    pool = core.state.db_pool
    if pool is None:
        raise HTTPException(503, "Database not ready")

    if async_mode:
        background_tasks.add_task(sync_catalog_for_user, pool, str(user["id"]), force_full)
        return {"ok": True, "status": "queued", "async_mode": True}

    try:
        totals = await asyncio.wait_for(
            sync_catalog_for_user(pool, str(user["id"]), force_full),
            timeout=30.0,
        )
        return {"ok": True, "status": "done", "async_mode": False, **totals}
    except asyncio.TimeoutError:
        return {"ok": True, "status": "running", "async_mode": False, "note": "still running in background"}


@router.get("/api/catalog/sync-status")
async def get_catalog_sync_status(user: dict = Depends(_session_user)):
    """Return per-token sync state (platform, status, last_synced_at, cursor, counts)."""
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT s.platform, s.account_id, s.status, s.last_synced_at,
                   s.total_discovered, s.total_linked, s.error_detail,
                   s.next_cursor IS NOT NULL AS has_more_pages
            FROM platform_content_sync_state s
            WHERE s.user_id = $1
            ORDER BY s.last_synced_at DESC NULLS LAST
            """,
            user["id"],
        )
    return [
        {
            "platform": r["platform"],
            "account_id": r["account_id"],
            "status": r["status"],
            "last_synced_at": r["last_synced_at"].isoformat() if r["last_synced_at"] else None,
            "total_discovered": r["total_discovered"],
            "total_linked": r["total_linked"],
            "has_more_pages": bool(r["has_more_pages"]),
            "error_detail": r["error_detail"],
        }
        for r in rows
    ]


@router.get("/api/catalog/content")
async def get_catalog_content(
    platform: Optional[str] = Query(None),
    source: Optional[str] = Query(None, description="external|uploadm8|linked|all"),
    account_id: Optional[str] = Query(None, description="Filter to one connected account (catalog account_id)"),
    sort: str = Query("views", description="views|likes|published_at|engagement"),
    order: str = Query("desc"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    days: Optional[int] = Query(None, description="Filter to last N days (published_at)"),
    start: Optional[str] = Query(None, description="ISO-8601 UTC inclusive start (use with end; excludes rolling days)"),
    end: Optional[str] = Query(None, description="ISO-8601 UTC exclusive end"),
    user: dict = Depends(_session_user_readonly),
):
    """
    Paginated list of all known content items — UploadM8 + external.
    Each row includes source badge, upload_id if linked, and per-platform URL.
    """
    uid = str(user["id"])
    conditions = ["pci.user_id = $1"]
    params: List[Any] = [uid]

    if platform:
        conditions.append(f"pci.platform = ${len(params)+1}")
        params.append(platform.lower())
    if source and source != "all":
        conditions.append(f"pci.source = ${len(params)+1}")
        params.append(source.lower())
    if account_id and str(account_id).strip():
        conditions.append(f"pci.account_id = ${len(params)+1}")
        params.append(str(account_id).strip())
    win_start = parse_iso_ts(start)
    win_end = parse_iso_ts(end)
    # Use upload completion / created time when catalog publish date is missing or placeholder.
    _eff_ts = "COALESCE(pci.published_at, u.completed_at, u.created_at)"
    if win_start is not None and win_end is not None:
        conditions.append(f"{_eff_ts} >= ${len(params)+1}")
        params.append(win_start)
        conditions.append(f"{_eff_ts} < ${len(params)+1}")
        params.append(win_end)
        conditions.append(f"{_eff_ts} IS NOT NULL")
    elif days and days > 0:
        conditions.append(f"{_eff_ts} >= NOW() - INTERVAL '{int(days)} days'")
        conditions.append(f"{_eff_ts} IS NOT NULL")

    where = " AND ".join(conditions)

    sort_col_map = {
        "views": "pci.views",
        "likes": "pci.likes",
        "published_at": _eff_ts,
        "engagement": "(CASE WHEN pci.views > 0 THEN (pci.likes + pci.comments + pci.shares)::float / pci.views ELSE 0 END)",
    }
    sort_col = sort_col_map.get(sort, "pci.views")
    sort_dir = "DESC" if order.lower() != "asc" else "ASC"
    null_pos = "NULLS LAST" if sort_dir == "DESC" else "NULLS FIRST"

    async with core.state.db_pool.acquire() as conn:
        total_row = await conn.fetchrow(
            f"""
            SELECT COUNT(*) FROM platform_content_items pci
            LEFT JOIN uploads u ON u.id = pci.upload_id AND u.user_id = pci.user_id
            WHERE {where}
            """,
            *params,
        )
        rows = await conn.fetch(
            f"""
            SELECT pci.id, pci.platform, pci.account_id, pci.platform_video_id, pci.upload_id, pci.source,
                   pci.content_kind, pci.title, pci.published_at, pci.thumbnail_url, pci.platform_url,
                   pci.duration_seconds, pci.views, pci.likes, pci.comments, pci.shares, pci.metrics_synced_at,
                   pci.created_at,
                   u.title AS upload_title, u.thumbnail_r2_key, u.platform_results AS upload_pr,
                   u.views AS upload_views, u.likes AS upload_likes, u.comments AS upload_comments, u.shares AS upload_shares,
                   u.platforms AS upload_platforms,
                   u.filename AS upload_filename, u.caption AS upload_caption,
                   u.ai_title AS upload_ai_title, u.ai_generated_title AS upload_ai_generated_title,
                   u.completed_at AS upload_completed_at, u.created_at AS upload_created_at,
                   pt.account_name AS token_account_name, pt.account_username AS token_account_username
            FROM platform_content_items pci
            LEFT JOIN uploads u ON u.id = pci.upload_id AND u.user_id = pci.user_id
            LEFT JOIN LATERAL (
                SELECT account_name, account_username
                FROM platform_tokens
                WHERE user_id = pci.user_id
                  AND platform = pci.platform
                  AND account_id IS NOT DISTINCT FROM pci.account_id
                  AND revoked_at IS NULL
                ORDER BY created_at DESC NULLS LAST
                LIMIT 1
            ) pt ON TRUE
            WHERE {where}
            ORDER BY {sort_col} {sort_dir} {null_pos}
            LIMIT ${len(params)+1} OFFSET ${len(params)+2}
            """,
            *params, limit, offset,
        )

    row_dicts = [dict(r) for r in rows]
    thumb_urls: dict[int, str | None] = {}
    keys_to_sign: list[tuple[int, str]] = []
    for i, d in enumerate(row_dicts):
        if d.get("thumbnail_r2_key"):
            nk = _normalize_r2_key(d["thumbnail_r2_key"])
            if nk:
                keys_to_sign.append((i, nk))
    if keys_to_sign:
        try:
            s3 = get_s3_client()
            for i, nk in keys_to_sign:
                try:
                    thumb_urls[i] = s3.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": R2_BUCKET_NAME, "Key": nk},
                        ExpiresIn=3600,
                    )
                except Exception as e:
                    logger.debug("catalog content: thumbnail presign failed idx=%s: %s", i, e)
                    thumb_urls[i] = None
        except Exception as e:
            logger.debug("catalog content: batch thumbnail presign unavailable: %s", e)

    items = []
    for idx, r in enumerate(row_dicts):
        pr_title, pr_m = title_and_metrics_from_upload_platform_results(
            r.get("upload_pr"), r.get("platform"), r.get("platform_video_id")
        )
        v = max(int(r["views"] or 0), int(pr_m["views"] or 0))
        l = max(int(r["likes"] or 0), int(pr_m["likes"] or 0))
        c = max(int(r["comments"] or 0), int(pr_m["comments"] or 0))
        s = max(int(r["shares"] or 0), int(pr_m["shares"] or 0))
        plat_row = str(r.get("platform") or "").lower()
        upl = r.get("upload_platforms") or []
        if isinstance(upl, str):
            try:
                upl = json.loads(upl)
            except Exception:
                upl = []
        targets = [str(x).lower() for x in upl if x]
        merge_u = bool(r.get("upload_id")) and (
            not targets or (len(targets) == 1 and targets[0] == plat_row)
        )
        if merge_u:
            v = max(v, int(r.get("upload_views") or 0))
            l = max(l, int(r.get("upload_likes") or 0))
            c = max(c, int(r.get("upload_comments") or 0))
            s = max(s, int(r.get("upload_shares") or 0))
        eng = round((l + c + s) / v * 100, 2) if v > 0 else 0.0
        raw_title = (r.get("title") or "").strip()
        up_t = (r.get("upload_title") or "").strip()
        pr_t = (pr_title or "").strip() if pr_title else ""
        ai_t = (r.get("upload_ai_title") or r.get("upload_ai_generated_title") or "").strip()
        fn_t = (r.get("upload_filename") or "").strip()
        cap_raw = r.get("upload_caption")
        cap_t = ""
        if isinstance(cap_raw, str) and cap_raw.strip():
            cap_t = cap_raw.strip().split("\n")[0][:500]
        title_out = raw_title or up_t or pr_t or ai_t or fn_t or cap_t or None
        thumb_out = (r.get("thumbnail_url") or "").strip() or thumb_urls.get(idx)
        acct_label = (
            (r.get("token_account_name") or "").strip()
            or (r.get("token_account_username") or "").strip()
            or ""
        )
        eff_pub = r.get("published_at") or r.get("upload_completed_at") or r.get("upload_created_at")
        items.append({
            "id": str(r["id"]),
            "platform": r["platform"],
            "account_id": r["account_id"],
            "account_label": acct_label or None,
            "platform_video_id": r["platform_video_id"],
            "upload_id": str(r["upload_id"]) if r["upload_id"] else None,
            "source": r["source"],
            "content_kind": r["content_kind"],
            "title": title_out,
            "published_at": eff_pub.isoformat() if eff_pub else None,
            "thumbnail_url": thumb_out or None,
            "platform_url": r["platform_url"],
            "duration_seconds": r["duration_seconds"],
            "views": v, "likes": l, "comments": c, "shares": s,
            "engagement_rate": eng,
            "metrics_synced_at": r["metrics_synced_at"].isoformat() if r["metrics_synced_at"] else None,
        })

    return {
        "items": items,
        "total": int(total_row[0] or 0),
        "limit": limit,
        "offset": offset,
    }


@router.get("/api/catalog/aggregate")
async def get_catalog_aggregate_endpoint(
    period: Optional[str] = Query(None, description="Time window: '7d','30d','7h','24h','90m','all'. Overrides days."),
    days: Optional[int] = Query(None, description="Legacy: last N days (use period instead)"),
    platform: Optional[str] = Query(None),
    source: Optional[str] = Query(None, description="external|uploadm8|linked"),
    account_id: Optional[str] = Query(None, description="Restrict to one catalog account_id"),
    start: Optional[str] = Query(None, description="ISO-8601 UTC inclusive start (use with end)"),
    end: Optional[str] = Query(None, description="ISO-8601 UTC exclusive end"),
    user: dict = Depends(_session_user),
):
    """
    Aggregated views / likes / comments / shares + per-platform + per-source
    breakdown for the current user's entire content catalog.

    Time window via `period` (preferred) or legacy `days` integer:
      period=7d   → last 7 days
      period=7h   → last 7 hours
      period=30m  → last 30 minutes
      period=all  → all time (default when omitted)

    Custom absolute window: pass `start` and `end` (half-open [start, end)) — overrides period/days.

    Used by analytics chips / KPI cards to show ALL activity on connected
    accounts — not just UploadM8-originated videos.
    """
    ws = parse_iso_ts(start)
    we = parse_iso_ts(end)
    pool = core.state.db_pool
    if pool is None:
        raise HTTPException(503, "Database not ready")

    if ws is not None and we is not None:
        result = await get_catalog_aggregate(
            pool,
            str(user["id"]),
            period=None,
            days=None,
            platform=platform,
            source=source,
            account_id=account_id,
            window_start=ws,
            window_end_exclusive=we,
        )
    else:
        result = await get_catalog_aggregate(
            pool,
            str(user["id"]),
            period=period,
            days=days,
            platform=platform,
            source=source,
            account_id=account_id,
        )
    return result

