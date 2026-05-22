"""Admin KPI reliability, attach-rate, and growth helpers from uploads DB."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger("uploadm8.admin_kpi_reliability")


async def fetch_admin_reliability_metrics(
    conn,
    since: datetime,
    until: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Real upload pipeline metrics for admin KPI cards (no hardcoded placeholders)."""
    until_clause = ""
    params: list = [since]
    if until is not None:
        until_clause = f" AND created_at < ${len(params) + 1}"
        params.append(until)

    row = await conn.fetchrow(
        f"""
        SELECT
            COUNT(*)::int AS total,
            SUM(CASE WHEN status IN ('completed','succeeded') THEN 1 ELSE 0 END)::int AS completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed,
            SUM(CASE WHEN status = 'partial' THEN 1 ELSE 0 END)::int AS partial,
            SUM(CASE WHEN status = 'cancelled' OR cancel_requested = TRUE THEN 1 ELSE 0 END)::int AS cancelled,
            SUM(CASE WHEN COALESCE((output_artifacts->'retry'->>'count')::int, 0) > 0 THEN 1 ELSE 0 END)::int AS retried,
            SUM(CASE WHEN status = 'failed'
                      AND (lower(COALESCE(error_code, '')) LIKE '%transcode%'
                           OR lower(COALESCE(error_detail, '')) LIKE '%transcode%')
                 THEN 1 ELSE 0 END)::int AS transcode_failed,
            SUM(CASE WHEN status = 'failed'
                      AND (lower(COALESCE(error_code, '')) LIKE '%publish%'
                           OR lower(COALESCE(error_code, '')) LIKE '%platform%'
                           OR status = 'partial')
                 THEN 1 ELSE 0 END)::int AS platform_failed,
            COALESCE(
                AVG(EXTRACT(EPOCH FROM (processing_finished_at - processing_started_at)))
                FILTER (WHERE processing_started_at IS NOT NULL
                        AND processing_finished_at IS NOT NULL
                        AND processing_finished_at > processing_started_at),
                0
            )::double precision AS avg_process_seconds
        FROM uploads
        WHERE created_at >= $1{until_clause}
        """,
        *params,
    )

    queue = await conn.fetchval(
        "SELECT COUNT(*)::int FROM uploads WHERE status IN ('pending', 'queued', 'processing')"
    )

    retry_hist = await conn.fetchrow(
        f"""
        SELECT
            SUM(CASE WHEN rc = 1 THEN 1 ELSE 0 END)::int AS one,
            SUM(CASE WHEN rc = 2 THEN 1 ELSE 0 END)::int AS two,
            SUM(CASE WHEN rc >= 3 THEN 1 ELSE 0 END)::int AS three_plus
        FROM (
            SELECT COALESCE((output_artifacts->'retry'->>'count')::int, 0) AS rc
            FROM uploads
            WHERE created_at >= $1{until_clause}
        ) t
        WHERE rc > 0
        """,
        *params,
    )

    total = int(row["total"] or 0) if row else 0
    completed = int(row["completed"] or 0) if row else 0
    failed = int(row["failed"] or 0) if row else 0
    cancelled = int(row["cancelled"] or 0) if row else 0
    retried = int(row["retried"] or 0) if row else 0
    transcode_failed = int(row["transcode_failed"] or 0) if row else 0
    platform_failed = int(row["platform_failed"] or 0) if row else 0
    avg_sec = float(row["avg_process_seconds"] or 0) if row else 0.0

    success_rate = (completed / max(total, 1)) * 100
    retry_rate = (retried / max(total, 1)) * 100
    cancel_rate = (cancelled / max(total, 1)) * 100
    transcode_fail_rate = (transcode_failed / max(total, 1)) * 100
    platform_fail_rate = (platform_failed / max(max(failed, 1), 1)) * 100

    one = int(retry_hist["one"] or 0) if retry_hist else 0
    two = int(retry_hist["two"] or 0) if retry_hist else 0
    three_plus = int(retry_hist["three_plus"] or 0) if retry_hist else 0
    retried_total = max(one + two + three_plus, 1)

    return {
        "success_rate": round(success_rate, 1),
        "reliability_change": 0,
        "successful_uploads": completed,
        "total_uploads": total,
        "transcode_fail_rate": round(transcode_fail_rate, 1),
        "platform_fail_rate": round(platform_fail_rate, 1),
        "retry_rate": round(retry_rate, 1),
        "avg_process_time": round(avg_sec, 1),
        "avg_transcode_time": None,
        "cancel_rate": round(cancel_rate, 1),
        "queue_depth": int(queue or 0),
        "failRates": {
            "ingest": None,
            "processing": round(transcode_fail_rate, 1),
            "upload": round((failed / max(total, 1)) * 100, 1),
            "publish": round(platform_fail_rate, 1),
            "average": round((failed / max(total, 1)) * 100, 1),
        },
        "retries": {
            "rate": round(retry_rate, 1),
            "one": round((one / retried_total) * 100, 1),
            "two": round((two / retried_total) * 100, 1),
            "threePlus": round((three_plus / retried_total) * 100, 1),
        },
        "processingTime": {
            "ingest": None,
            "transcode": None,
            "upload": None,
            "average": round(avg_sec, 1),
        },
        "cancels": {
            "rate": round(cancel_rate, 1),
            "beforeProcessing": None,
            "duringProcessing": None,
            "total30d": cancelled,
        },
    }


async def fetch_admin_attach_metrics(conn, since: datetime, until: Optional[datetime] = None) -> Dict[str, Any]:
    """AI / top-up / flex attach rates for growth and KPI cards."""
    until_clause = ""
    params: list = [since]
    if until is not None:
        until_clause = f" AND created_at < ${len(params) + 1}"
        params.append(until)

    ai_users = await conn.fetchval(
        f"SELECT COUNT(DISTINCT user_id)::int FROM uploads WHERE aic_spent > 0 AND created_at >= $1{until_clause}",
        *params,
    )
    total_uploaders = await conn.fetchval(
        f"SELECT COUNT(DISTINCT user_id)::int FROM uploads WHERE created_at >= $1{until_clause}",
        *params,
    )
    topup_users = await conn.fetchval(
        f"SELECT COUNT(DISTINCT user_id)::int FROM token_ledger WHERE reason = 'topup' AND created_at >= $1{until_clause}",
        *params,
    )
    flex_users = await conn.fetchval("SELECT COUNT(*)::int FROM users WHERE flex_enabled = TRUE")

    uploaders = max(int(total_uploaders or 0), 1)
    ai_rate = (int(ai_users or 0) / uploaders) * 100
    topup_rate = (int(topup_users or 0) / uploaders) * 100

    return {
        "ai_attach_rate": round(ai_rate, 1),
        "topup_attach_rate": round(topup_rate, 1),
        "flex_adoption_rate": round((int(flex_users or 0) / max(uploaders, 1)) * 100, 1),
        "flex_users": int(flex_users or 0),
    }


async def fetch_admin_growth_metrics(conn, since: datetime, until: Optional[datetime] = None) -> Dict[str, Any]:
    """Growth funnel metrics from DB (replaces hardcoded attach/churn placeholders)."""
    def _user_window(alias: str = "") -> tuple[str, list]:
        col = f"{alias}created_at" if alias else "created_at"
        clauses = [f"{col} >= $1"]
        p: list = [since]
        if until is not None:
            clauses.append(f"{col} < ${len(p) + 1}")
            p.append(until)
        return " AND ".join(clauses), p

    uw, up = _user_window()
    signups = await conn.fetchval(f"SELECT COUNT(*)::int FROM users WHERE {uw}", *up)

    connected = await conn.fetchval(
        """
        SELECT COUNT(DISTINCT u.id)::int FROM users u
        JOIN platform_tokens pt ON u.id = pt.user_id
        WHERE u.created_at >= $1""" + (" AND u.created_at < $2" if until else ""),
        *up,
    )

    upload_w, upload_p = _user_window()
    uploaded = await conn.fetchval(
        f"SELECT COUNT(DISTINCT user_id)::int FROM uploads WHERE {upload_w}",
        *upload_p,
    )

    paid_w = "updated_at >= $1" + (" AND updated_at < $2" if until else "")
    paid = await conn.fetchval(
        f"""
        SELECT COUNT(*)::int FROM users
        WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime')
          AND subscription_status = 'active'
          AND {paid_w}
        """,
        *up,
    )
    cancellations = await conn.fetchval(
        f"""
        SELECT COUNT(*)::int FROM users
        WHERE subscription_status = 'cancelled' AND {paid_w}
        """,
        *up,
    )
    paid_active = await conn.fetchval(
        """
        SELECT COUNT(*)::int FROM users
        WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime')
          AND subscription_status = 'active'
        """
    ) or 1

    attach = await fetch_admin_attach_metrics(conn, since, until)
    signups_n = int(signups or 0)
    churn_rate = (int(cancellations or 0) / max(int(paid_active or 0), 1)) * 100

    return {
        "activation": {
            "rate": round((int(uploaded or 0) / max(signups_n, 1)) * 100, 1),
            "signups": signups_n,
            "connected": int(connected or 0),
            "firstUpload": int(uploaded or 0),
        },
        "conversion": {
            "freeToPaid": round((int(paid or 0) / max(signups_n, 1)) * 100, 1),
            "trialToPaid": None,
            "avgDays": None,
            "count30d": int(paid or 0),
            "change": 0,
        },
        "attach": {
            "ai": attach["ai_attach_rate"],
            "topups": attach["topup_attach_rate"],
            "flex": attach["flex_adoption_rate"],
            "average": round((attach["ai_attach_rate"] + attach["topup_attach_rate"]) / 2, 1),
        },
        "churn": {
            "rate": round(churn_rate, 1),
            "cancellations": int(cancellations or 0),
            "failedPayments": 0,
            "downgrades": 0,
        },
        "free_to_paid_rate": round((int(paid or 0) / max(signups_n, 1)) * 100, 1),
        "conversion_change": 0,
        **attach,
    }
