"""
Account-level platform metrics → platform_metrics_cache.

Used by:
  - Worker (scheduled refresh, no HTTP / no user session)
  - app.py (optional background refresh after per-upload sync)

Token decryption uses stages.publish_stage.decrypt_token so this module does not
require FastAPI lifespan (init_enc_keys).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import asyncpg

from stages.publish_stage import decrypt_token

logger = logging.getLogger("uploadm8.platform_metrics_job")

_PLATFORM_CACHE_TTL = 3 * 60 * 60


def _aggregate_platform_metrics_live(platforms_result: dict) -> dict:
    agg = {"views": 0, "likes": 0, "comments": 0, "shares": 0, "platforms_included": []}

    def _n(x) -> int:
        try:
            return int(x or 0)
        except Exception:
            return 0

    if not isinstance(platforms_result, dict):
        return agg

    for plat, d in platforms_result.items():
        if not isinstance(d, dict) or d.get("status") != "live":
            continue
        views = _n(d.get("views"))
        likes = _n(d.get("reactions")) if plat == "facebook" else _n(d.get("likes"))
        comments = _n(d.get("comments"))
        shares = _n(d.get("shares"))
        if views or likes or comments or shares:
            agg["views"] += views
            agg["likes"] += likes
            agg["comments"] += comments
            agg["shares"] += shares
            agg["platforms_included"].append(plat)
    return agg


async def _store_platform_metrics_cache(conn: asyncpg.Connection, user_id: str, output: dict) -> None:
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
    except Exception as e:
        logger.warning(f"[platform-metrics-job] cache write failed user={user_id}: {e}")


def _import_fetchers():
    """Lazy import from app — keeps worker startup light."""
    import app as app_module

    return (
        app_module._fetch_tiktok_metrics,
        app_module._fetch_youtube_metrics,
        app_module._fetch_instagram_metrics,
        app_module._fetch_facebook_metrics,
    )


async def refresh_platform_metrics_for_user(pool: asyncpg.Pool, user_id: Any) -> bool:
    """
    Recompute live TikTok/YouTube/Instagram/Facebook metrics for one user and
    upsert platform_metrics_cache. Returns True if a row was written.

    Does not update app.py in-memory _platform_metrics_cache (API process only).
    """
    uid = str(user_id)
    try:
        ftik, fyt, fig, ffb = _import_fetchers()
    except Exception as e:
        logger.error(f"[platform-metrics-job] cannot import fetchers: {e}")
        return False

    async with pool.acquire() as conn:
        token_rows = await conn.fetch(
            "SELECT platform, token_blob, account_id FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
            uid,
        )
        upload_counts = await conn.fetch(
            """SELECT unnest(platforms) AS platform, COUNT(*)::int AS cnt
               FROM uploads
               WHERE user_id = $1 AND status IN ('succeeded', 'completed', 'partial')
               GROUP BY platform""",
            uid,
        )

    upload_map = {r["platform"]: r["cnt"] for r in upload_counts}

    token_map: Dict[str, dict] = {}
    for row in token_rows:
        plat = row["platform"]
        blob = row["token_blob"]
        if not blob:
            continue
        try:
            decrypted = decrypt_token(blob)
        except Exception:
            continue
        if not decrypted:
            continue
        if plat == "instagram" and not decrypted.get("ig_user_id") and row["account_id"]:
            decrypted["ig_user_id"] = str(row["account_id"])
        if plat == "facebook" and not decrypted.get("page_id") and row["account_id"]:
            decrypted["page_id"] = str(row["account_id"])
        token_map[plat] = decrypted

    uid_str = str(user_id)
    try:
        from stages.publish_stage import _refresh_tiktok_token, _refresh_youtube_token

        if token_map.get("tiktok"):
            token_map["tiktok"] = await _refresh_tiktok_token(
                dict(token_map["tiktok"]), db_pool=pool, user_id=uid_str
            )
        if token_map.get("youtube"):
            token_map["youtube"] = await _refresh_youtube_token(
                dict(token_map["youtube"]), db_pool=pool, user_id=uid_str
            )
        async with pool.acquire() as conn:
            trs = await conn.fetch(
                "SELECT platform, token_blob, account_id FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
                uid_str,
            )
        token_map = {}
        for row in trs:
            plat = row["platform"]
            blob = row["token_blob"]
            if not blob:
                continue
            try:
                decrypted = decrypt_token(blob)
            except Exception:
                continue
            if decrypted:
                if plat == "instagram" and not decrypted.get("ig_user_id") and row["account_id"]:
                    decrypted["ig_user_id"] = str(row["account_id"])
                if plat == "facebook" and not decrypted.get("page_id") and row["account_id"]:
                    decrypted["page_id"] = str(row["account_id"])
                token_map[plat] = decrypted
    except Exception as e:
        logger.warning(f"[platform-metrics-job] oauth refresh: {e}")

    async def run_tiktok():
        t = token_map.get("tiktok", {})
        return await ftik(t.get("access_token", ""))

    async def run_youtube():
        t = token_map.get("youtube", {})
        return await fyt(t.get("access_token", ""))

    async def run_instagram():
        t = token_map.get("instagram", {})
        ig_id = t.get("ig_user_id") or t.get("instagram_user_id") or t.get("instagram_page_id") or ""
        return await fig(t.get("access_token", ""), ig_id)

    async def run_facebook():
        t = token_map.get("facebook", {})
        page_id = t.get("page_id") or t.get("facebook_page_id") or t.get("fb_page_id") or ""
        return await ffb(t.get("access_token", ""), page_id)

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
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
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
        "aggregate": _aggregate_platform_metrics_live(platforms_result),
        "cache_source": "worker",
    }

    async with pool.acquire() as conn:
        await _store_platform_metrics_cache(conn, uid, output)

    return True


async def refresh_all_users_platform_metrics_cache(pool: asyncpg.Pool) -> int:
    """Refresh cache for every user that has at least one active platform connection."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT user_id FROM platform_tokens WHERE revoked_at IS NULL"
        )
    user_ids = [str(r["user_id"]) for r in rows]
    refreshed = 0
    for uid in user_ids:
        try:
            if await refresh_platform_metrics_for_user(pool, uid):
                refreshed += 1
        except Exception as e:
            logger.warning(f"[platform-metrics-job] user {uid}: {e}")
        await asyncio.sleep(0.35)
    return refreshed
