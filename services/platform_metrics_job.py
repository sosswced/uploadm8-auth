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
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

from stages.publish_stage import decrypt_token

logger = logging.getLogger("uploadm8.platform_metrics_job")

_PLATFORM_CACHE_TTL = 3 * 60 * 60
_POLL_CONCURRENCY = max(2, int(os.environ.get("PLATFORM_POLL_CONCURRENCY", "8") or 8))
_POLL_RETRIES = max(1, int(os.environ.get("PLATFORM_POLL_RETRIES", "3") or 3))
_POLL_BACKOFF_BASE_SEC = float(os.environ.get("PLATFORM_POLL_BACKOFF_BASE_SEC", "0.6") or 0.6)
_INTER_USER_DELAY_SEC = float(os.environ.get("PLATFORM_METRICS_INTER_USER_DELAY_SEC", "0.2") or 0.2)


def _aggregate_platform_metrics_live(platforms_result: dict) -> dict:
    agg = {"views": 0, "likes": 0, "comments": 0, "shares": 0, "platforms_included": []}

    def _n(x) -> int:
        try:
            return max(0, int(x or 0))
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


def _metric_int(v: Any) -> int:
    try:
        return max(0, int(v or 0))
    except Exception:
        return 0


def _normalize_account_id(platform: str, row: asyncpg.Record, token: Dict[str, Any]) -> str:
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


def _merge_metric_totals(rows: List[Dict[str, Any]], platform: str) -> Dict[str, int]:
    totals = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
    for row in rows:
        if not isinstance(row, dict):
            continue
        totals["views"] += _metric_int(row.get("views"))
        if platform == "facebook":
            totals["likes"] += _metric_int(row.get("reactions"))
        else:
            totals["likes"] += _metric_int(row.get("likes"))
        totals["comments"] += _metric_int(row.get("comments"))
        totals["shares"] += _metric_int(row.get("shares"))
    return totals


def _platform_summary_from_account_rows(plat: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build one platform block (live/error/not_connected) from cached account row payloads."""
    live_rows = [r.get("metrics", {}) for r in rows if (r.get("metrics") or {}).get("status") == "live"]
    if live_rows:
        totals = _merge_metric_totals(live_rows, plat)
        exemplar = dict(live_rows[-1]) if live_rows else {}
        exemplar.update(
            {
                "status": "live",
                "views": totals["views"],
                "likes": totals["likes"],
                "comments": totals["comments"],
                "shares": totals["shares"],
                "accounts_polled": len(rows),
                "accounts_live": len(live_rows),
                "accounts": rows,
            }
        )
        return exemplar
    if rows:
        err = next((r.get("metrics", {}) for r in rows if (r.get("metrics", {}) or {}).get("status") == "error"), {})
        return {
            "status": "error",
            "error": err.get("error", "all_accounts_failed"),
            "accounts_polled": len(rows),
            "accounts_live": 0,
            "accounts": rows,
        }
    return {"status": "not_connected", "accounts_polled": 0, "accounts_live": 0, "accounts": []}


async def prune_platform_metrics_cache_for_disconnected_token(
    pool: asyncpg.Pool, user_id: Any, disconnected_token_row_id: Any
) -> bool:
    """
    Remove one disconnected platform_tokens row from platform_metrics_cache and
    recompute per-platform and aggregate live metrics from remaining cached accounts.

    Upload counts are refreshed from the uploads table (still reflects all user uploads).
    """
    uid = str(user_id)
    tid = str(disconnected_token_row_id)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT data FROM platform_metrics_cache WHERE user_id = $1",
            uid,
        )
        if not row or row["data"] is None:
            return False
        try:
            existing = dict(row["data"]) if isinstance(row["data"], dict) else json.loads(row["data"])
        except Exception:
            return False
        platforms_in = existing.get("platforms") or {}
        if not isinstance(platforms_in, dict):
            return False

        upload_counts = await conn.fetch(
            """SELECT unnest(platforms) AS platform, COUNT(*)::int AS cnt
               FROM uploads
               WHERE user_id = $1 AND status IN ('succeeded', 'completed', 'partial')
               GROUP BY platform""",
            uid,
        )
        upload_map = {r["platform"]: r["cnt"] for r in upload_counts}

        platforms_result: Dict[str, Dict[str, Any]] = {}
        for plat in ("tiktok", "youtube", "instagram", "facebook"):
            old = platforms_in.get(plat) or {}
            if not isinstance(old, dict):
                old = {}
            accs = old.get("accounts")
            if not isinstance(accs, list):
                accs = []
            filtered = [r for r in accs if isinstance(r, dict) and str(r.get("token_row_id", "")) != tid]
            summary = _platform_summary_from_account_rows(plat, filtered)
            summary["uploads"] = upload_map.get(plat, 0)
            platforms_result[plat] = summary

        ttl_min = int(_PLATFORM_CACHE_TTL / 60)
        output = {
            "platforms": platforms_result,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "cached": bool(existing.get("cached", True)),
            "cache_age_minutes": 0,
            "next_refresh_minutes": int(existing.get("next_refresh_minutes") or ttl_min),
            "aggregate": _aggregate_platform_metrics_live(platforms_result),
            "cache_source": str(existing.get("cache_source") or "worker"),
        }
        await _store_platform_metrics_cache(conn, uid, output)
        await _upsert_user_rollup_daily(
            conn,
            user_id=uid,
            aggregate=output.get("aggregate") or {},
            platforms_result=platforms_result,
        )
    return True


async def _with_retry(coro_factory, *, retries: int = _POLL_RETRIES):
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            sleep_s = (_POLL_BACKOFF_BASE_SEC * (2 ** (attempt - 1))) + random.uniform(0.05, 0.25)
            await asyncio.sleep(sleep_s)
    raise last_err or RuntimeError("retry_failed")


async def _store_account_metric_event(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    token_row_id: Optional[str],
    platform: str,
    account_id: str,
    metrics: Dict[str, Any],
) -> None:
    try:
        await conn.execute(
            """
            INSERT INTO platform_account_metrics_events
                (user_id, token_row_id, platform, account_id, metrics, fetched_at)
            VALUES
                ($1, $2, $3, $4, $5::jsonb, NOW())
            """,
            user_id,
            token_row_id,
            platform,
            account_id or None,
            json.dumps(metrics or {}),
        )
    except Exception as e:
        logger.debug(f"[platform-metrics-job] account-event write skipped: {e}")


async def _upsert_user_rollup_daily(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    aggregate: Dict[str, Any],
    platforms_result: Dict[str, Any],
) -> None:
    try:
        await conn.execute(
            """
            INSERT INTO platform_user_metrics_rollups_daily
                (user_id, day, views, likes, comments, shares, platforms_json, updated_at)
            VALUES
                ($1, CURRENT_DATE, $2, $3, $4, $5, $6::jsonb, NOW())
            ON CONFLICT (user_id, day) DO UPDATE
            SET views = EXCLUDED.views,
                likes = EXCLUDED.likes,
                comments = EXCLUDED.comments,
                shares = EXCLUDED.shares,
                platforms_json = EXCLUDED.platforms_json,
                updated_at = NOW()
            """,
            user_id,
            _metric_int(aggregate.get("views")),
            _metric_int(aggregate.get("likes")),
            _metric_int(aggregate.get("comments")),
            _metric_int(aggregate.get("shares")),
            json.dumps(platforms_result or {}),
        )
    except Exception as e:
        logger.debug(f"[platform-metrics-job] rollup write skipped: {e}")


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
            """
            SELECT id, platform, token_blob, account_id
              FROM platform_tokens
             WHERE user_id = $1
               AND revoked_at IS NULL
            """,
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
    account_jobs: List[Tuple[str, str, str, Dict[str, Any]]] = []
    for row in token_rows:
        plat = str(row["platform"] or "").lower()
        blob = row["token_blob"]
        if not blob or plat not in ("tiktok", "youtube", "instagram", "facebook"):
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
        account_id = _normalize_account_id(plat, row, decrypted)
        account_jobs.append((plat, str(row["id"]), account_id, decrypted))

    uid_str = str(user_id)
    try:
        from stages.publish_stage import _refresh_tiktok_token, _refresh_youtube_token, _refresh_meta_token

        refreshed_jobs: List[Tuple[str, str, str, Dict[str, Any]]] = []
        for plat, row_id, account_id, token in account_jobs:
            cur = dict(token)
            if plat == "tiktok":
                cur = await _refresh_tiktok_token(
                    cur,
                    db_pool=pool,
                    user_id=uid_str,
                    token_row_id=row_id,
                )
            elif plat == "youtube":
                cur = await _refresh_youtube_token(
                    cur,
                    db_pool=pool,
                    user_id=uid_str,
                    token_row_id=row_id,
                )
            elif plat in ("instagram", "facebook"):
                cur = await _refresh_meta_token(
                    cur,
                    platform=plat,
                    db_pool=pool,
                    user_id=uid_str,
                    token_row_id=row_id,
                )
            refreshed_jobs.append((plat, row_id, account_id, cur))
        account_jobs = refreshed_jobs
    except Exception as e:
        logger.warning(f"[platform-metrics-job] oauth refresh: {e}")

    semaphore = asyncio.Semaphore(_POLL_CONCURRENCY)

    async def run_one_account(
        platform: str, token_row_id: str, account_id: str, token: Dict[str, Any]
    ) -> Tuple[str, str, str, Dict[str, Any]]:
        async with semaphore:
            async def _run_once():
                if platform == "tiktok":
                    return await ftik((token or {}).get("access_token", ""))
                if platform == "youtube":
                    return await fyt((token or {}).get("access_token", ""))
                if platform == "instagram":
                    ig_id = (token or {}).get("ig_user_id") or (token or {}).get("instagram_user_id") or (token or {}).get("instagram_page_id") or ""
                    return await fig((token or {}).get("access_token", ""), ig_id)
                if platform == "facebook":
                    page_id = (token or {}).get("page_id") or (token or {}).get("facebook_page_id") or (token or {}).get("fb_page_id") or ""
                    return await ffb((token or {}).get("access_token", ""), page_id)
                return {"status": "not_connected"}

            try:
                res = await _with_retry(_run_once)
            except Exception as e:
                res = {"status": "error", "error": str(e)}
            return platform, token_row_id, account_id, res

    task_results = await asyncio.gather(
        *[run_one_account(p, rid, aid, tok) for (p, rid, aid, tok) in account_jobs],
        return_exceptions=False,
    )

    per_platform_rows: Dict[str, List[Dict[str, Any]]] = {
        "tiktok": [],
        "youtube": [],
        "instagram": [],
        "facebook": [],
    }
    for platform, token_row_id, account_id, res in task_results:
        row_payload = {
            "token_row_id": token_row_id,
            "account_id": account_id,
            "status": (res or {}).get("status", "error"),
            "metrics": res or {},
        }
        per_platform_rows.setdefault(platform, []).append(row_payload)

    platforms_result: Dict[str, Dict[str, Any]] = {}
    for plat in ("tiktok", "youtube", "instagram", "facebook"):
        rows = per_platform_rows.get(plat) or []
        platforms_result[plat] = _platform_summary_from_account_rows(plat, rows)

    for plat in ["tiktok", "youtube", "instagram", "facebook"]:
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
        for plat, token_row_id, account_id, res in task_results:
            await _store_account_metric_event(
                conn,
                user_id=uid,
                token_row_id=token_row_id,
                platform=plat,
                account_id=account_id,
                metrics=res or {},
            )
        await _upsert_user_rollup_daily(
            conn,
            user_id=uid,
            aggregate=output.get("aggregate") or {},
            platforms_result=platforms_result,
        )

    return True


async def refresh_all_users_platform_metrics_cache(pool: asyncpg.Pool) -> int:
    """Refresh cache for every user that has at least one active platform connection."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT user_id FROM platform_tokens WHERE revoked_at IS NULL"
        )
    user_ids = [str(r["user_id"]) for r in rows]
    refreshed = 0
    sem = asyncio.Semaphore(max(1, min(_POLL_CONCURRENCY, 12)))

    async def _run(uid: str) -> bool:
        async with sem:
            try:
                ok = await refresh_platform_metrics_for_user(pool, uid)
                await asyncio.sleep(_INTER_USER_DELAY_SEC)
                return ok
            except Exception as e:
                logger.warning(f"[platform-metrics-job] user {uid}: {e}")
                return False

    results = await asyncio.gather(*[_run(uid) for uid in user_ids], return_exceptions=False)
    refreshed = sum(1 for x in results if x)
    return refreshed
