"""GET /api/dashboard/stats body — DB + canonical engagement rollup (rolling 30d UTC)."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Mapping

from core.helpers import _now_utc
from services.upload_engagement import compute_upload_engagement_totals as _compute_upload_engagement_totals
from services.canonical_engagement import (
    ROLLUP_VERSION,
    compute_canonical_engagement_rollup,
    engagement_time_window_for_analytics_range,
    engagement_window_api_dict,
)
from services.platform_metrics_ui import aggregate_platform_metrics_live
from services.upload_metrics import SUCCESSFUL_STATUS_SQL_IN

logger = logging.getLogger(__name__)


async def _with_conn(pool: Any, fn: Any) -> Any:
    async with pool.acquire() as c:
        return await fn(c)


def _assemble_dashboard_stats(
    user: Mapping[str, Any],
    plan: Mapping[str, Any],
    wallet: Mapping[str, Any],
    stats: Any,
    scheduled: Any,
    put_used_month: Any,
    uploads_used_month: Any,
    accounts: Any,
    recent: Any,
    dash_live: dict[str, Any],
    upload_engagement: dict[str, Any],
    cr: dict[str, Any],
    dash_win: dict[str, Any],
) -> dict[str, Any]:
    canon = dict(cr)
    canon["engagement_window_utc"] = dash_win

    total = stats["total"] if stats else 0
    completed = stats["completed"] if stats else 0
    succ_cm = int(stats["successful_this_month"] or 0) if stats else 0
    succ_lm = int(stats["successful_last_month"] or 0) if stats else 0
    put_avail = wallet.get("put_balance", 0) - wallet.get("put_reserved", 0)
    aic_avail = wallet.get("aic_balance", 0) - wallet.get("aic_reserved", 0)
    put_monthly = int(plan.get("put_monthly", 60) or 0)
    uploads_limit = int(
        plan.get("monthly_uploads")
        or plan.get("max_uploads_monthly")
        or plan.get("put_monthly")
        or 0
    )
    role = str(user.get("role") or "").lower()
    tier = str(user.get("subscription_tier") or "").lower()
    unlimited_uploads = bool(
        role == "master_admin"
        or tier in ("master_admin", "friends_family", "lifetime")
        or int(uploads_limit or 0) >= 999999
        or int(put_monthly or 0) >= 999999
    )
    success_rate = (completed / max(total, 1)) * 100 if total else 0

    put_reserved = float(wallet.get("put_reserved", 0) or 0)
    aic_reserved = float(wallet.get("aic_reserved", 0) or 0)
    put_total = float(wallet.get("put_balance", 0) or 0)
    aic_total = float(wallet.get("aic_balance", 0) or 0)

    db_views = int(upload_engagement.get("views") or 0)
    db_likes = int(upload_engagement.get("likes") or 0)
    live_v = int(dash_live.get("views") or 0)
    live_l = int(dash_live.get("likes") or 0)

    return {
        "uploads": {
            "total": total,
            "completed": completed,
            "in_queue": stats["in_queue"] if stats else 0,
            "successful_this_month": succ_cm,
            "successful_last_month": succ_lm,
        },
        "engagement": {
            "views": int(canon["views"] or 0),
            "likes": int(canon["likes"] or 0),
            "comments": int(canon["comments"] or 0),
            "shares": int(canon["shares"] or 0),
            "breakdown": canon["breakdown"],
            "rollup_rule": canon["rollup_rule"],
            "rollup_version": canon.get("rollup_version"),
            "catalog_tracked_videos": int(canon["catalog_tracked_videos"] or 0),
            "engagement_window_utc": canon.get("engagement_window_utc"),
            "views_db": db_views,
            "likes_db": db_likes,
            "live_views": live_v,
            "live_likes": live_l,
            "live_platforms": dash_live.get("platforms_included") or [],
            "kpi_sources": canon.get("kpi_sources"),
        },
        "success_rate": round(success_rate, 1),
        "scheduled": scheduled or 0,
        "quota": {
            "put_used": put_used_month or 0,
            "put_limit": put_monthly,
            "uploads_used": int(uploads_used_month or 0),
            "uploads_limit": (-1 if unlimited_uploads else int(uploads_limit or 0)),
            "uploads_unlimited": unlimited_uploads,
        },
        "wallet": {"put_available": put_avail, "put_total": put_total, "aic_available": aic_avail, "aic_total": aic_total},
        "credits": {
            "put": {"available": put_avail, "reserved": put_reserved, "total": put_total, "monthly_allowance": put_monthly},
            "aic": {
                "available": aic_avail,
                "reserved": aic_reserved,
                "total": aic_total,
                "monthly_allowance": plan.get("aic_monthly", 0),
            },
        },
        "accounts": {"connected": accounts or 0, "limit": int(plan.get("max_accounts", 1) or 1)},
        "recent": [
            {"id": str(r["id"]), "filename": r["filename"], "platforms": r["platforms"], "status": r["status"]}
            for r in recent
        ],
        "plan": plan,
    }


async def build_dashboard_stats_payload(
    conn: Any,
    user: Mapping[str, Any],
    plan: Mapping[str, Any],
    wallet: Mapping[str, Any],
) -> dict[str, Any]:
    """Single-connection path (legacy callers). Prefer ``dashboard_stats_for_user`` for production."""
    uid = user["id"]
    stats = await conn.fetchrow(
        f"""
            SELECT COUNT(*)::int AS total,
                   SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN} THEN 1 ELSE 0 END)::int AS completed,
                   SUM(CASE WHEN status IN ('pending','queued','processing','staged','scheduled','ready_to_publish') THEN 1 ELSE 0 END)::int AS in_queue,
                   COALESCE(SUM(views), 0)::bigint AS views,
                   COALESCE(SUM(likes), 0)::bigint AS likes,
                   COALESCE(SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN}
                        AND created_at >= date_trunc('month', CURRENT_DATE) THEN 1 ELSE 0 END), 0)::int AS successful_this_month,
                   COALESCE(SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN}
                        AND created_at >= date_trunc('month', CURRENT_DATE) - interval '1 month'
                        AND created_at < date_trunc('month', CURRENT_DATE) THEN 1 ELSE 0 END), 0)::int AS successful_last_month
            FROM uploads WHERE user_id = $1
            """,
        uid,
    )
    scheduled = await conn.fetchval(
        """
            SELECT COUNT(*)::int FROM uploads
            WHERE user_id = $1
              AND status IN ('pending','staged','queued','scheduled','ready_to_publish')
              AND schedule_mode IN ('scheduled','smart')
              AND scheduled_time IS NOT NULL
        """,
        uid,
    )
    try:
        put_used_month = await conn.fetchval(
            """
                SELECT COALESCE(SUM(put_spent), 0)::int FROM uploads
                WHERE user_id = $1 AND created_at >= date_trunc('month', CURRENT_DATE)
            """,
            uid,
        )
    except Exception:
        logger.warning("dashboard: put_used_month query failed user=%s", uid, exc_info=True)
        put_used_month = 0
    try:
        uploads_used_month = await conn.fetchval(
            """
                SELECT COUNT(*)::int
                  FROM uploads
                 WHERE user_id = $1
                   AND created_at >= date_trunc('month', CURRENT_DATE)
                """,
            uid,
        )
    except Exception:
        logger.warning("dashboard: uploads_used_month query failed user=%s", uid, exc_info=True)
        uploads_used_month = 0
    try:
        accounts = await conn.fetchval(
            "SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1 AND (revoked_at IS NULL OR revoked_at > NOW())",
            uid,
        )
    except Exception as e:
        logger.warning("dashboard: platform_tokens count with revoked clause failed, fallback: %s", e)
        accounts = await conn.fetchval("SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1", uid)
    recent = await conn.fetch(
        "SELECT id, filename, platforms, status, created_at FROM uploads WHERE user_id = $1 ORDER BY created_at DESC LIMIT 5",
        uid,
    )

    dash_live: dict[str, Any] = {"views": 0, "likes": 0, "comments": 0, "shares": 0, "platforms_included": []}
    if int(accounts or 0) > 0:
        try:
            drow = await conn.fetchrow(
                "SELECT data FROM platform_metrics_cache WHERE user_id = $1",
                uid,
            )
            if drow and drow["data"] is not None:
                pdata = drow["data"]
                if isinstance(pdata, str):
                    pdata = json.loads(pdata)
                if isinstance(pdata, dict):
                    dash_live = aggregate_platform_metrics_live(pdata.get("platforms") or {})
        except Exception as e:
            logger.warning("dashboard: platform_metrics_cache read failed: %s", e)
    win_start, win_end = engagement_time_window_for_analytics_range("30d", now=_now_utc())

    upload_engagement = await _compute_upload_engagement_totals(
        conn, str(uid), since=win_start, until=win_end
    )

    try:
        cr = await compute_canonical_engagement_rollup(
            conn,
            str(uid),
            window_start=win_start,
            window_end_exclusive=win_end,
            platform=None,
        )
    except Exception as e:
        logger.warning("dashboard: canonical engagement rollup failed: %s", e)
        cr = {
            "views": int(upload_engagement.get("views") or 0),
            "likes": int(upload_engagement.get("likes") or 0),
            "comments": int(upload_engagement.get("comments") or 0),
            "shares": int(upload_engagement.get("shares") or 0),
            "breakdown": {
                "compute": {
                    "rollup_version": ROLLUP_VERSION,
                    "complete": False,
                    "warnings": ["rollup_exception"],
                    "error_detail": str(e)[:500],
                },
            },
            "catalog_tracked_videos": 0,
            "rollup_version": ROLLUP_VERSION,
            "rollup_rule": "fallback_upload_table_only",
            "kpi_sources": {"error": str(e), "rollup_version": ROLLUP_VERSION},
        }
    dash_win = engagement_window_api_dict(start=win_start, end_exclusive=win_end)
    return _assemble_dashboard_stats(
        user,
        plan,
        wallet,
        stats,
        scheduled,
        put_used_month,
        uploads_used_month,
        accounts,
        recent,
        dash_live,
        upload_engagement,
        cr,
        dash_win,
    )


async def dashboard_stats_for_user(pool: Any, user: dict[str, Any], plan: dict[str, Any], wallet: dict[str, Any]) -> dict[str, Any]:
    """Parallel DB work across pool connections (asyncpg: one conn cannot multiplex queries)."""
    uid = user["id"]
    win_start, win_end = engagement_time_window_for_analytics_range("30d", now=_now_utc())

    async def q_stats(c: Any) -> Any:
        return await c.fetchrow(
            f"""
            SELECT COUNT(*)::int AS total,
                   SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN} THEN 1 ELSE 0 END)::int AS completed,
                   SUM(CASE WHEN status IN ('pending','queued','processing','staged','scheduled','ready_to_publish') THEN 1 ELSE 0 END)::int AS in_queue,
                   COALESCE(SUM(views), 0)::bigint AS views,
                   COALESCE(SUM(likes), 0)::bigint AS likes,
                   COALESCE(SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN}
                        AND created_at >= date_trunc('month', CURRENT_DATE) THEN 1 ELSE 0 END), 0)::int AS successful_this_month,
                   COALESCE(SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN}
                        AND created_at >= date_trunc('month', CURRENT_DATE) - interval '1 month'
                        AND created_at < date_trunc('month', CURRENT_DATE) THEN 1 ELSE 0 END), 0)::int AS successful_last_month
            FROM uploads WHERE user_id = $1
            """,
            uid,
        )

    async def q_scheduled(c: Any) -> Any:
        return await c.fetchval(
            """
            SELECT COUNT(*)::int FROM uploads
            WHERE user_id = $1
              AND status IN ('pending','staged','queued','scheduled','ready_to_publish')
              AND schedule_mode IN ('scheduled','smart')
              AND scheduled_time IS NOT NULL
            """,
            uid,
        )

    async def q_put_used_month(c: Any) -> Any:
        try:
            return await c.fetchval(
                """
                SELECT COALESCE(SUM(put_spent), 0)::int FROM uploads
                WHERE user_id = $1 AND created_at >= date_trunc('month', CURRENT_DATE)
                """,
                uid,
            )
        except Exception:
            logger.warning("dashboard: put_used_month query failed user=%s", uid, exc_info=True)
            return 0

    async def q_uploads_used_month(c: Any) -> Any:
        try:
            return await c.fetchval(
                """
                SELECT COUNT(*)::int
                  FROM uploads
                 WHERE user_id = $1
                   AND created_at >= date_trunc('month', CURRENT_DATE)
                """,
                uid,
            )
        except Exception:
            logger.warning("dashboard: uploads_used_month query failed user=%s", uid, exc_info=True)
            return 0

    async def q_accounts(c: Any) -> Any:
        try:
            return await c.fetchval(
                "SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1 AND (revoked_at IS NULL OR revoked_at > NOW())",
                uid,
            )
        except Exception as e:
            logger.warning("dashboard: platform_tokens count with revoked clause failed, fallback: %s", e)
            return await c.fetchval("SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1", uid)

    async def q_recent(c: Any) -> Any:
        return await c.fetch(
            "SELECT id, filename, platforms, status, created_at FROM uploads WHERE user_id = $1 ORDER BY created_at DESC LIMIT 5",
            uid,
        )

    stats, scheduled, put_used_month, uploads_used_month, accounts, recent = await asyncio.gather(
        _with_conn(pool, q_stats),
        _with_conn(pool, q_scheduled),
        _with_conn(pool, q_put_used_month),
        _with_conn(pool, q_uploads_used_month),
        _with_conn(pool, q_accounts),
        _with_conn(pool, q_recent),
    )

    dash_live: dict[str, Any] = {"views": 0, "likes": 0, "comments": 0, "shares": 0, "platforms_included": []}
    if int(accounts or 0) > 0:

        async def q_metrics_cache(c: Any) -> dict[str, Any]:
            out: dict[str, Any] = {"views": 0, "likes": 0, "comments": 0, "shares": 0, "platforms_included": []}
            try:
                drow = await c.fetchrow("SELECT data FROM platform_metrics_cache WHERE user_id = $1", uid)
                if drow and drow["data"] is not None:
                    pdata = drow["data"]
                    if isinstance(pdata, str):
                        pdata = json.loads(pdata)
                    if isinstance(pdata, dict):
                        return aggregate_platform_metrics_live(pdata.get("platforms") or {})
            except Exception as e:
                logger.warning("dashboard: platform_metrics_cache read failed: %s", e)
            return out

        dash_live = await _with_conn(pool, q_metrics_cache)

    async def q_upload_engagement(c: Any) -> dict[str, Any]:
        return await _compute_upload_engagement_totals(c, str(uid), since=win_start, until=win_end)

    async def q_canonical(c: Any) -> dict[str, Any]:
        return await compute_canonical_engagement_rollup(
            c,
            str(uid),
            window_start=win_start,
            window_end_exclusive=win_end,
            platform=None,
        )

    ue_res, cr_res = await asyncio.gather(
        _with_conn(pool, q_upload_engagement),
        _with_conn(pool, q_canonical),
        return_exceptions=True,
    )
    if isinstance(ue_res, asyncio.CancelledError):
        raise ue_res
    if isinstance(cr_res, asyncio.CancelledError):
        raise cr_res
    if isinstance(ue_res, Exception):
        logger.warning("dashboard: upload engagement totals failed user=%s: %r", uid, ue_res)
        upload_engagement = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
    else:
        upload_engagement = ue_res

    if isinstance(cr_res, Exception):
        e2 = cr_res
        logger.warning("dashboard: canonical engagement rollup failed: %s", e2)
        cr = {
            "views": int(upload_engagement.get("views") or 0),
            "likes": int(upload_engagement.get("likes") or 0),
            "comments": int(upload_engagement.get("comments") or 0),
            "shares": int(upload_engagement.get("shares") or 0),
            "breakdown": {
                "compute": {
                    "rollup_version": ROLLUP_VERSION,
                    "complete": False,
                    "warnings": ["rollup_exception"],
                    "error_detail": str(e2)[:500],
                },
            },
            "catalog_tracked_videos": 0,
            "rollup_version": ROLLUP_VERSION,
            "rollup_rule": "fallback_upload_table_only",
            "kpi_sources": {"error": str(e2), "rollup_version": ROLLUP_VERSION},
        }
    else:
        cr = cr_res

    if not isinstance(cr, dict):
        cr = {
            "views": int(upload_engagement.get("views") or 0),
            "likes": int(upload_engagement.get("likes") or 0),
            "comments": int(upload_engagement.get("comments") or 0),
            "shares": int(upload_engagement.get("shares") or 0),
            "breakdown": {},
            "catalog_tracked_videos": 0,
            "rollup_version": ROLLUP_VERSION,
            "rollup_rule": "fallback_upload_table_only",
            "kpi_sources": {"rollup_version": ROLLUP_VERSION},
        }

    dash_win = engagement_window_api_dict(start=win_start, end_exclusive=win_end)
    return _assemble_dashboard_stats(
        user,
        plan,
        wallet,
        stats,
        scheduled,
        put_used_month,
        uploads_used_month,
        accounts,
        recent,
        dash_live,
        upload_engagement,
        cr,
        dash_win,
    )
