"""GET /api/dashboard/stats body — DB + canonical engagement rollup (all-time).

Upload counts:
  - uploads.total — lifetime COUNT(*) on uploads (all statuses)
  - quota.uploads_used — calendar-month COUNT(*) (date_trunc month)
  - engagement — all-time canonical merge (same rollup as Analytics range=all)
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Mapping, Optional

from core.db_pool import is_dead_connection_error
from core.deps import require_verified_user_on_conn
from core.helpers import _now_utc, get_plan
from core.wallet import get_wallet
from services.upload_engagement import compute_upload_engagement_totals as _compute_upload_engagement_totals
from services.canonical_engagement import (
    ROLLUP_VERSION,
    compute_canonical_engagement_rollup,
    engagement_time_window_for_analytics_range,
    engagement_window_api_dict,
)
from services.platform_metrics_ui import aggregate_platform_metrics_live
from services.upload_metrics import SUCCESSFUL_STATUS_SQL_IN
from services.uploads_handlers import (
    QUEUE_VIEW_STATUSES,
    SCHEDULED_PIPELINE_STATUSES,
    scheduled_in_clause,
)

# Keep /api/dashboard/stats responsive when all-time rollup is slow — quota must still return.
_ENGAGEMENT_ROLLUP_TIMEOUT_SEC = 8.0

# Inline `status IN (...)` literals for queries where parametrization would
# require restructuring (kept in sync with the canonical tuples).
_SCHEDULED_SQL_LITERAL = "(" + ", ".join(f"'{s}'" for s in SCHEDULED_PIPELINE_STATUSES) + ")"
_QUEUE_VIEW_SQL_LITERAL = "(" + ", ".join(f"'{s}'" for s in QUEUE_VIEW_STATUSES) + ")"

logger = logging.getLogger(__name__)


def _engagement_scope_from_rollup(cr: Mapping[str, Any]) -> str:
    """Honest scope label — never claim all_time for provisional/fallback KPIs."""
    rule = str((cr or {}).get("rollup_rule") or "")
    if rule == "light_metrics_cache":
        return "provisional_cache"
    if rule == "fallback_upload_table_only":
        return "upload_table_fallback"
    return "all_time"


def _fallback_engagement_cr(
    upload_engagement: Mapping[str, Any],
    *,
    warning: str,
    extra_kpi: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    kpi = {"rollup_version": ROLLUP_VERSION}
    if extra_kpi:
        kpi.update(extra_kpi)
    return {
        "views": int(upload_engagement.get("views") or 0),
        "likes": int(upload_engagement.get("likes") or 0),
        "comments": int(upload_engagement.get("comments") or 0),
        "shares": int(upload_engagement.get("shares") or 0),
        "breakdown": {
            "compute": {
                "rollup_version": ROLLUP_VERSION,
                "complete": False,
                "warnings": [warning],
            },
        },
        "catalog_tracked_videos": 0,
        "rollup_version": ROLLUP_VERSION,
        "rollup_rule": "fallback_upload_table_only",
        "kpi_sources": kpi,
    }


async def _canonical_rollup_safe(
    *,
    pool: Any,
    conn: Any,
    uid: str,
    win_start: Any,
    win_end: Any,
    upload_engagement: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Run canonical rollup without poisoning the primary stats connection.

    When ``pool`` is available, rollup uses a separate checkout + timeout; on
    timeout the rollup connection is closed (not released) so a cancelled query
    cannot corrupt the pool or the primary conn.
    """
    if pool is None:
        try:
            cr = await compute_canonical_engagement_rollup(
                conn,
                str(uid),
                window_start=win_start,
                window_end_exclusive=win_end,
                platform=None,
            )
            return cr if isinstance(cr, dict) else _fallback_engagement_cr(
                upload_engagement, warning="rollup_invalid"
            )
        except Exception as e2:
            logger.warning("dashboard: canonical engagement rollup failed: %s", e2)
            return _fallback_engagement_cr(
                upload_engagement, warning="rollup_exception", extra_kpi={"error": str(e2)[:500]}
            )

    rconn = await pool.acquire()
    timed_out = False
    try:
        cr = await asyncio.wait_for(
            compute_canonical_engagement_rollup(
                rconn,
                str(uid),
                window_start=win_start,
                window_end_exclusive=win_end,
                platform=None,
            ),
            timeout=_ENGAGEMENT_ROLLUP_TIMEOUT_SEC,
        )
        if not isinstance(cr, dict):
            return _fallback_engagement_cr(upload_engagement, warning="rollup_invalid")
        return cr
    except asyncio.TimeoutError:
        timed_out = True
        logger.warning(
            "dashboard: canonical engagement rollup timed out after %.0fs user=%s",
            _ENGAGEMENT_ROLLUP_TIMEOUT_SEC,
            uid,
        )
        return _fallback_engagement_cr(
            upload_engagement, warning="rollup_timeout", extra_kpi={"timeout": True}
        )
    except Exception as e2:
        logger.warning("dashboard: canonical engagement rollup failed: %s", e2)
        return _fallback_engagement_cr(
            upload_engagement, warning="rollup_exception", extra_kpi={"error": str(e2)[:500]}
        )
    finally:
        if timed_out:
            # Cancelled query may leave the connection unusable — do not pool-release.
            try:
                await rconn.close()
            except Exception:
                pass
        else:
            try:
                await pool.release(rconn)
            except Exception:
                try:
                    await rconn.close()
                except Exception:
                    pass


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

    # uploads.total = lifetime row count (all statuses). Not rolling 30d.
    # quota.uploads_used = calendar month (date_trunc month), all statuses.
    # successful_*_month = calendar month, completed/succeeded/partial only.
    return {
        "uploads": {
            "total": total,
            "total_scope": "all_time",
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
            "scope": _engagement_scope_from_rollup(canon),
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
                   SUM(CASE WHEN status IN {_QUEUE_VIEW_SQL_LITERAL} THEN 1 ELSE 0 END)::int AS in_queue,
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
    # Canonical SCHEDULED_PIPELINE_STATUSES — same definition used by
    # /api/scheduled/list, /api/scheduled/stats, and queue.html. Single source
    # of truth lives in services/uploads_handlers.py.
    ph, statuses = scheduled_in_clause(2)
    scheduled = await conn.fetchval(
        f"""
            SELECT COUNT(*)::int FROM uploads
            WHERE user_id = $1
              AND status IN ({ph})
        """,
        uid,
        *statuses,
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
        # Schema/SQL fallback only — never reuse a dead pool connection (UPLOADM8-80).
        if is_dead_connection_error(e):
            raise
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
    win_start, win_end = engagement_time_window_for_analytics_range("all", now=_now_utc())

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


async def _dashboard_stats_on_conn(
    conn: Any,
    user: dict[str, Any],
    plan: dict[str, Any],
    wallet: dict[str, Any],
    *,
    light: bool = False,
    pool: Any = None,
) -> dict[str, Any]:
    """Run dashboard stats SQL on an existing connection.

    ``light=True`` skips canonical engagement rollup (shell bootstrap first paint).
    When ``pool`` is provided, the heavy rollup uses a separate checkout + timeout
    so cancellation cannot poison ``conn``.
    """
    uid = user["id"]
    win_start, win_end = engagement_time_window_for_analytics_range("all", now=_now_utc())

    async def q_stats(c: Any) -> Any:
        return await c.fetchrow(
            f"""
            SELECT COUNT(*)::int AS total,
                   SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN} THEN 1 ELSE 0 END)::int AS completed,
                   SUM(CASE WHEN status IN {_QUEUE_VIEW_SQL_LITERAL} THEN 1 ELSE 0 END)::int AS in_queue,
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
        # Canonical SCHEDULED_PIPELINE_STATUSES — matches /api/scheduled/list,
        # /api/scheduled/stats, queue.html. See services/uploads_handlers.py.
        return await c.fetchval(
            f"""
            SELECT COUNT(*)::int FROM uploads
            WHERE user_id = $1
              AND status IN {_SCHEDULED_SQL_LITERAL}
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
            # Schema/SQL fallback only — never reuse a dead pool connection (UPLOADM8-80).
            if is_dead_connection_error(e):
                raise
            logger.warning("dashboard: platform_tokens count with revoked clause failed, fallback: %s", e)
            return await c.fetchval("SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1", uid)

    async def q_recent(c: Any) -> Any:
        return await c.fetch(
            "SELECT id, filename, platforms, status, created_at FROM uploads WHERE user_id = $1 ORDER BY created_at DESC LIMIT 5",
            uid,
        )

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

    if light:
        # One uploads scan for shell bootstrap (UPLOADM8-89 companion) instead of
        # four sequential COUNT/SUM queries on the same table.
        # Avoid put_spent in the primary scan so a missing/legacy column cannot
        # zero out monthly upload quota (uploads_used_month).
        try:
            light_row = await conn.fetchrow(
                f"""
                SELECT
                    COUNT(*)::int AS total,
                    SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN} THEN 1 ELSE 0 END)::int AS completed,
                    SUM(CASE WHEN status IN {_QUEUE_VIEW_SQL_LITERAL} THEN 1 ELSE 0 END)::int AS in_queue,
                    COALESCE(SUM(views), 0)::bigint AS views,
                    COALESCE(SUM(likes), 0)::bigint AS likes,
                    COALESCE(SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN}
                        AND created_at >= date_trunc('month', CURRENT_DATE) THEN 1 ELSE 0 END), 0)::int AS successful_this_month,
                    COALESCE(SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN}
                        AND created_at >= date_trunc('month', CURRENT_DATE) - interval '1 month'
                        AND created_at < date_trunc('month', CURRENT_DATE) THEN 1 ELSE 0 END), 0)::int AS successful_last_month,
                    SUM(CASE WHEN status IN {_SCHEDULED_SQL_LITERAL} THEN 1 ELSE 0 END)::int AS scheduled_count,
                    SUM(CASE WHEN created_at >= date_trunc('month', CURRENT_DATE) THEN 1 ELSE 0 END)::int AS uploads_used_month
                FROM uploads WHERE user_id = $1
                """,
                uid,
            )
        except Exception:
            logger.warning("dashboard light aggregate failed user=%s", uid, exc_info=True)
            light_row = None
        stats = light_row
        scheduled = int(light_row["scheduled_count"] or 0) if light_row else 0
        uploads_used_month = int(light_row["uploads_used_month"] or 0) if light_row else 0
        put_used_month = 0
        if light_row is not None:
            try:
                put_used_month = int(
                    await conn.fetchval(
                        """
                        SELECT COALESCE(SUM(put_spent), 0)::int FROM uploads
                        WHERE user_id = $1 AND created_at >= date_trunc('month', CURRENT_DATE)
                        """,
                        uid,
                    )
                    or 0
                )
            except Exception:
                logger.warning("dashboard light put_used_month failed user=%s", uid, exc_info=True)
                put_used_month = 0
        if light_row is None:
            # Atomic fallback: only accept quota used when totals also load.
            try:
                stats = await q_stats(conn)
                scheduled = await q_scheduled(conn)
                uploads_used_month = int(await q_uploads_used_month(conn) or 0)
                put_used_month = int(await q_put_used_month(conn) or 0)
            except Exception:
                logger.warning("dashboard light quota fallback failed user=%s", uid, exc_info=True)
                stats = None
                scheduled = 0
                uploads_used_month = 0
                put_used_month = 0
    else:
        stats = await q_stats(conn)
        scheduled = await q_scheduled(conn)
        put_used_month = await q_put_used_month(conn)
        uploads_used_month = await q_uploads_used_month(conn)
    accounts = await q_accounts(conn)
    recent = await q_recent(conn)

    dash_live: dict[str, Any] = {"views": 0, "likes": 0, "comments": 0, "shares": 0, "platforms_included": []}
    if int(accounts or 0) > 0:
        dash_live = await q_metrics_cache(conn)

    dash_win = engagement_window_api_dict(start=win_start, end_exclusive=win_end)

    if light:
        # Metrics-cache only — full rollup via GET /api/dashboard/stats after paint.
        upload_engagement = {
            "views": int(dash_live.get("views") or 0),
            "likes": int(dash_live.get("likes") or 0),
            "comments": int(dash_live.get("comments") or 0),
            "shares": int(dash_live.get("shares") or 0),
        }
        cr = {
            "views": upload_engagement["views"],
            "likes": upload_engagement["likes"],
            "comments": upload_engagement["comments"],
            "shares": upload_engagement["shares"],
            "breakdown": {
                "compute": {
                    "rollup_version": ROLLUP_VERSION,
                    "complete": False,
                    "warnings": ["light_bootstrap"],
                },
            },
            "catalog_tracked_videos": 0,
            "rollup_version": ROLLUP_VERSION,
            "rollup_rule": "light_metrics_cache",
            "kpi_sources": {"light": True, "rollup_version": ROLLUP_VERSION},
        }
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

    try:
        upload_engagement = await _compute_upload_engagement_totals(
            conn, str(uid), since=win_start, until=win_end
        )
    except Exception as ue_res:
        logger.warning("dashboard: upload engagement totals failed user=%s: %r", uid, ue_res)
        upload_engagement = {"views": 0, "likes": 0, "comments": 0, "shares": 0}

    cr = await _canonical_rollup_safe(
        pool=pool,
        conn=conn,
        uid=str(uid),
        win_start=win_start,
        win_end=win_end,
        upload_engagement=upload_engagement,
    )

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


async def dashboard_stats_for_user(
    pool: Any,
    user: dict[str, Any],
    plan: dict[str, Any],
    wallet: dict[str, Any],
    *,
    light: bool = False,
) -> dict[str, Any]:
    """
    Dashboard stats when the caller already has ``user`` / ``plan`` / ``wallet`` (e.g. shell bootstrap).

    One pooled connection for the stats queries; user/wallet often came from a separate dep.
    ``light=True`` skips the heavy engagement rollup for first paint.
    """
    async with pool.acquire() as conn:
        return await _dashboard_stats_on_conn(conn, user, plan, wallet, light=light, pool=pool)


async def fetch_dashboard_stats_for_user_id(
    pool: Any, user_id: str, *, light: bool = False
) -> dict[str, Any]:
    """
    ``GET /api/dashboard/stats`` path: JWT-verified ``user_id`` then **one** checkout for
    users row (auth gates) + wallet + all stats SQL — avoids an extra pool release from
    ``get_current_user_readonly`` before this block.

    ``light=True`` skips canonical engagement (quota/counts only) for fast first paint.
    Full path runs rollup on a separate pool checkout with timeout.
    """
    async with pool.acquire() as conn:
        user = await require_verified_user_on_conn(conn, user_id)
        wallet = await get_wallet(conn, user_id)
        plan = get_plan(user.get("subscription_tier", "free"))
        return await _dashboard_stats_on_conn(
            conn, user, plan, wallet, light=light, pool=pool
        )
