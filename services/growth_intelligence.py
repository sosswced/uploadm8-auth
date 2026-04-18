"""
Aggregates upload, wallet, marketing_events, and Thumbnail Studio usage
for admin dashboards, marketing ops, and per-user coaching (data-driven;
upgrade path to trained models without changing API shapes).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from stages.entitlements import get_entitlements_for_tier, normalize_tier
from stages.m8_engine_brand import (
    M8_ENGINE_AI_DISPLAY,
    M8_ENGINE_AI_SLUG,
    M8_ENGINE_FAMILY_TAGLINE,
    M8_ENGINE_SLUG,
)

logger = logging.getLogger("uploadm8.growth_intelligence")


def sanitize_coach_payload_for_json(obj: Any) -> Any:
    """
    Starlette JSONResponse uses json.dumps(..., allow_nan=False).
    Postgres/Python can still produce NaN/Inf in rare float paths; that raises ValueError → HTTP 500.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0.0
    if isinstance(obj, dict):
        return {k: sanitize_coach_payload_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_coach_payload_for_json(v) for v in obj]
    return obj


_EMPTY_GLOBAL_BASELINES: Dict[str, Any] = {
    "sample_uploads_30d": 0,
    "global_avg_views": 0.0,
    "global_avg_likes": 0.0,
    "global_avg_comments": 0.0,
    "global_avg_shares": 0.0,
    "global_avg_engagement_rate_pct": 0.0,
}


def coach_endpoint_fallback(tier: Optional[Any] = None) -> Dict[str, Any]:
    """Safe empty coach JSON (200) when the full pipeline cannot run."""
    try:
        t = str(tier if tier is not None else "free").strip() or "free"
    except Exception:
        t = "free"
    return sanitize_coach_payload_for_json(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "baselines": dict(_EMPTY_GLOBAL_BASELINES),
            "engagement_snapshot": {"samples_30d": 0},
            "smart_offer": None,
            "suggestions": [],
            "tier": t,
            "m8_engine": m8_engine_identity_payload(),
            "content_attribution_insights": None,
        }
    )


def m8_engine_identity_payload() -> Dict[str, str]:
    """Stable API shape: M8_ENGINE family + unified M8_ENGINE AI (ML + AI) layer."""
    return {
        "family_slug": M8_ENGINE_SLUG,
        "ai_slug": M8_ENGINE_AI_SLUG,
        "ai_display": M8_ENGINE_AI_DISPLAY,
        # Legacy keys — same values as ai_slug / ai_display
        "mlai_slug": M8_ENGINE_AI_SLUG,
        "mlai_display": M8_ENGINE_AI_DISPLAY,
        "tagline": M8_ENGINE_FAMILY_TAGLINE,
    }

RANGE_MINUTES = {
    "24h": 1440,
    "7d": 10080,
    "30d": 43200,
    "90d": 129600,
    "6m": 259200,
    "1y": 525600,
    "365d": 525600,
}


def parse_range_since_until(range_key: str) -> Tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    mins = RANGE_MINUTES.get(range_key, 43200)
    return now - timedelta(minutes=mins), now


def _is_internal_tier(tier: Optional[str]) -> bool:
    t = (tier or "").lower()
    return t in ("master_admin", "friends_family", "lifetime")


async def record_studio_usage_event(
    conn,
    user_id,
    operation: str,
    http_status: int,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        uid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
    except (ValueError, TypeError):
        return
    await conn.execute(
        """
        INSERT INTO studio_usage_events (user_id, operation, http_status, meta)
        VALUES ($1, $2, $3, $4::jsonb)
        """,
        uid,
        (operation or "?")[:80],
        int(http_status),
        json.dumps(meta or {}),
    )


async def fetch_pikzels_studio_usage(conn, since: datetime, until: datetime) -> Dict[str, Any]:
    rows = await conn.fetch(
        """
        SELECT s.operation AS op, COUNT(*)::bigint AS cnt
        FROM studio_usage_events s
        JOIN users u ON u.id = s.user_id
        WHERE s.created_at >= $1 AND s.created_at < $2
          AND COALESCE(u.subscription_tier, '') <> 'master_admin'
          AND COALESCE(u.role, '') <> 'master_admin'
          AND (s.http_status IS NULL OR s.http_status < 400)
        GROUP BY s.operation
        ORDER BY cnt DESC
        """,
        since,
        until,
    )
    by_op = [{"op": r["op"], "count": int(r["cnt"])} for r in rows]
    total = sum(x["count"] for x in by_op)
    return {"total_calls": total, "by_operation": by_op}


async def fetch_user_pikzels_studio_usage(conn, user_id: str, since: datetime, until: datetime) -> Dict[str, Any]:
    """Per-user Thumbnail Studio (Pikzels v2) counts from studio_usage_events (successful calls only)."""
    rows = await conn.fetch(
        """
        SELECT operation AS op, COUNT(*)::bigint AS cnt
        FROM studio_usage_events
        WHERE user_id = $1::uuid
          AND created_at >= $2 AND created_at < $3
          AND (http_status IS NULL OR http_status < 400)
        GROUP BY operation
        ORDER BY cnt DESC
        """,
        user_id,
        since,
        until,
    )
    by_op = [{"op": r["op"], "count": int(r["cnt"])} for r in rows]
    total = sum(x["count"] for x in by_op)
    return {"total_calls": total, "by_operation": by_op}


async def fetch_marketing_funnel(conn, since: datetime, until: datetime) -> Dict[str, Any]:
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE event_type = 'shown')::bigint AS shown,
            COUNT(*) FILTER (WHERE event_type = 'clicked')::bigint AS clicked,
            COUNT(*) FILTER (WHERE event_type = 'dismissed')::bigint AS dismissed
        FROM marketing_events
        WHERE created_at >= $1 AND created_at < $2
        """,
        since,
        until,
    )
    shown = int(row["shown"] or 0) if row else 0
    clicked = int(row["clicked"] or 0) if row else 0
    dismissed = int(row["dismissed"] or 0) if row else 0
    ctr = (100.0 * clicked / max(shown, 1)) if shown else 0.0
    dr = (100.0 * dismissed / max(shown, 1)) if shown else 0.0

    sess_rev = await conn.fetchval(
        """
        SELECT COALESCE(SUM(r.amount), 0)::float
        FROM revenue_tracking r
        WHERE r.user_id IS NOT NULL
          AND r.created_at >= $1 AND r.created_at < $2
          AND EXISTS (
            SELECT 1 FROM marketing_events e
            WHERE e.user_id = r.user_id
              AND e.event_type = 'clicked'
              AND e.created_at >= $1 AND e.created_at < $2
              AND r.created_at >= e.created_at
              AND r.created_at <= e.created_at + interval '1 hour'
          )
        """,
        since,
        until,
    )

    v7 = await conn.fetchval(
        """
        SELECT COALESCE(SUM(r.amount), 0)::float
        FROM revenue_tracking r
        WHERE r.user_id IS NOT NULL
          AND r.created_at >= $1 AND r.created_at < $2
          AND EXISTS (
            SELECT 1 FROM marketing_events e
            WHERE e.user_id = r.user_id
              AND e.event_type = 'clicked'
              AND e.created_at >= $1 AND e.created_at < $2
              AND r.created_at >= e.created_at
              AND r.created_at <= e.created_at + interval '7 days'
          )
        """,
        since,
        until,
    )

    return {
        "shown": shown,
        "clicked": clicked,
        "dismissed": dismissed,
        "ctr_pct": round(ctr, 2),
        "dismiss_rate_pct": round(dr, 2),
        "same_session_attributed_revenue": float(sess_rev or 0),
        "view_through_7d_attributed_revenue": float(v7 or 0),
    }


async def fetch_sales_opportunity_levers(conn) -> Dict[str, Any]:
    free_up = await conn.fetchval(
        """
        SELECT COUNT(DISTINCT u.id)::bigint
        FROM users u
        JOIN uploads up ON up.user_id = u.id
        WHERE COALESCE(u.subscription_tier, 'free') = 'free'
          AND up.created_at >= NOW() - interval '7 days'
        """
    )
    low_put = await conn.fetchval(
        """
        SELECT COUNT(*)::bigint
        FROM wallets w
        JOIN users u ON u.id = w.user_id
        WHERE COALESCE(u.subscription_tier, '') NOT IN ('master_admin', 'friends_family', 'lifetime')
          AND (COALESCE(w.put_balance, 0) - COALESCE(w.put_reserved, 0)) BETWEEN 0 AND 29
        """
    )
    low_aic = await conn.fetchval(
        """
        SELECT COUNT(*)::bigint
        FROM wallets w
        JOIN users u ON u.id = w.user_id
        WHERE COALESCE(u.subscription_tier, '') NOT IN ('master_admin', 'friends_family', 'lifetime')
          AND (COALESCE(w.aic_balance, 0) - COALESCE(w.aic_reserved, 0)) BETWEEN 0 AND 9
        """
    )
    multi = await conn.fetchval(
        """
        SELECT COUNT(*)::bigint FROM (
            SELECT pt.user_id
            FROM platform_tokens pt
            WHERE pt.revoked_at IS NULL
            GROUP BY pt.user_id
            HAVING COUNT(*) >= 3
        ) x
        """
    )
    return {
        "free_users_uploading_last_7d": int(free_up or 0),
        "users_low_put_available_0_29": int(low_put or 0),
        "users_low_aic_available_0_9": int(low_aic or 0),
        "users_3plus_platform_connections": int(multi or 0),
    }


async def fetch_promo_schedule_hints(conn, since: datetime, until: datetime) -> List[Dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT
            TRIM(TO_CHAR(date_trunc('hour', created_at AT TIME ZONE 'UTC'), 'Day')) AS day,
            EXTRACT(HOUR FROM (created_at AT TIME ZONE 'UTC'))::int AS hour_utc,
            COUNT(*) FILTER (WHERE event_type = 'clicked')::bigint AS clicks,
            COUNT(*) FILTER (WHERE event_type = 'converted')::bigint AS conversions_7d
        FROM marketing_events
        WHERE created_at >= $1 AND created_at < $2
        GROUP BY 1, 2
        HAVING COUNT(*) FILTER (WHERE event_type = 'clicked') > 0
        ORDER BY clicks DESC
        LIMIT 8
        """,
        since,
        until,
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        clicks = int(r["clicks"] or 0)
        conv = int(r["conversions_7d"] or 0)
        rate = (100.0 * conv / max(clicks, 1)) if clicks else 0.0
        out.append(
            {
                "day": (r["day"] or "").strip() or "—",
                "hour_utc": int(r["hour_utc"] or 0),
                "clicks": clicks,
                "conversions_7d": conv,
                "conv_rate_7d": round(rate, 2),
            }
        )
    return out


async def build_recommended_comms(levers: Dict[str, Any]) -> List[Dict[str, str]]:
    plans: List[Dict[str, str]] = []
    if levers.get("users_low_put_available_0_29", 0) > 5:
        plans.append(
            {
                "channel": "in_app",
                "cadence": "When PUT drops below 30",
                "trigger": f"~{levers['users_low_put_available_0_29']} accounts need top-up nudges",
            }
        )
    if levers.get("users_low_aic_available_0_9", 0) > 5:
        plans.append(
            {
                "channel": "email",
                "cadence": "Weekly",
                "trigger": f"~{levers['users_low_aic_available_0_9']} accounts low on AI credits — Thumbnail Studio + captions",
            }
        )
    if levers.get("free_users_uploading_last_7d", 0) > 3:
        plans.append(
            {
                "channel": "mixed",
                "cadence": "Bi-weekly",
                "trigger": f"{levers['free_users_uploading_last_7d']} active free uploaders — upgrade path",
            }
        )
    if levers.get("users_3plus_platform_connections", 0) > 2:
        plans.append(
            {
                "channel": "discord",
                "cadence": "Monthly spotlight",
                "trigger": f"{levers['users_3plus_platform_connections']} multi-platform power users — Studio/Agency",
            }
        )
    return plans


async def build_marketing_intel_bundle(conn, range_key: str) -> Dict[str, Any]:
    since, until = parse_range_since_until(range_key)
    funnel = await fetch_marketing_funnel(conn, since, until)
    levers = await fetch_sales_opportunity_levers(conn)
    promos = await fetch_promo_schedule_hints(conn, since, until)
    comms = await build_recommended_comms(levers)
    return {
        "range": range_key,
        "marketing_funnel": funnel,
        "sales_opportunity_levers": levers,
        "promo_schedule_recommendations": promos,
        "recommended_comms_plan": comms,
    }


async def fetch_ml_priors_debug(conn, since: datetime, limit: int) -> Dict[str, Any]:
    studio = await conn.fetch(
        """
        SELECT operation, COUNT(*)::bigint AS c, MAX(created_at) AS last_at
        FROM studio_usage_events
        WHERE created_at >= $1
        GROUP BY operation
        ORDER BY c DESC
        LIMIT 40
        """,
        since,
    )
    items: List[Dict[str, Any]] = []
    for r in studio:
        items.append(
            {
                "source": "studio_usage",
                "signal": str(r["operation"]),
                "weight": int(r["c"] or 0),
                "detail": {"last_at": r["last_at"].isoformat() if r.get("last_at") else None},
            }
        )
    me = await conn.fetch(
        """
        SELECT event_type, COUNT(*)::bigint AS c
        FROM marketing_events
        WHERE created_at >= $1
        GROUP BY event_type
        ORDER BY c DESC
        LIMIT 30
        """,
        since,
    )
    for r in me:
        items.append(
            {
                "source": "marketing_events",
                "signal": str(r["event_type"]),
                "weight": int(r["c"] or 0),
                "detail": {},
            }
        )
    items.sort(key=lambda x: -x["weight"])
    items = items[:limit]
    thumb_bias = sum(1 for i in items if i["source"] == "studio_usage" and i["weight"] > 0)
    return {
        "summary": {
            "total": len(items),
            "thumbnail_bias_present": thumb_bias,
            "m8_strategy_priors_present": sum(1 for i in items if i["source"] == "marketing_events"),
            "since_hours": max(1, int((datetime.now(timezone.utc) - since).total_seconds() // 3600)),
        },
        "items": items,
    }


def _enterprise_fit(uploads_30d: int, rev: float, plat: int, tier: str) -> float:
    score = min(
        100.0,
        uploads_30d * 3.5
        + plat * 11.0
        + (25.0 if tier in ("studio", "agency") else 0.0)
        + (18.0 if rev >= 80 else (10.0 if rev >= 20 else 0.0)),
    )
    return round(score, 2)


async def fetch_account_intelligence(
    conn, range_key: str, q: str, limit: int
) -> List[Dict[str, Any]]:
    since, until = parse_range_since_until(range_key)
    qpat = f"%{q.strip()}%" if q.strip() else "%"
    rows = await conn.fetch(
        """
        WITH wu AS (
            SELECT user_id, COUNT(*)::bigint AS uc, COALESCE(SUM(views), 0)::bigint AS tv
            FROM uploads
            WHERE created_at >= $2 AND created_at < $3
            GROUP BY user_id
        ),
        rev AS (
            SELECT user_id, COALESCE(SUM(amount), 0)::float AS r
            FROM revenue_tracking
            WHERE created_at >= $2 AND created_at < $3 AND user_id IS NOT NULL
            GROUP BY user_id
        ),
        nudge AS (
            SELECT user_id,
                COUNT(*) FILTER (WHERE event_type = 'shown')::bigint AS n_shown,
                COUNT(*) FILTER (WHERE event_type = 'clicked')::bigint AS n_click
            FROM marketing_events
            WHERE created_at >= $2 AND created_at < $3 AND user_id IS NOT NULL
            GROUP BY user_id
        ),
        plat AS (
            SELECT user_id, COUNT(*)::bigint AS pc
            FROM platform_tokens
            WHERE revoked_at IS NULL
            GROUP BY user_id
        )
        SELECT
            u.id,
            u.email,
            u.name,
            COALESCE(u.subscription_tier, 'free') AS subscription_tier,
            COALESCE(wu.uc, 0) AS uploads_30d,
            COALESCE(rev.r, 0.0) AS revenue_30d,
            COALESCE(nu.n_shown, 0) AS n_shown,
            COALESCE(nu.n_click, 0) AS n_click,
            COALESCE(p.pc, 0) AS connected_accounts
        FROM users u
        LEFT JOIN wu ON wu.user_id = u.id
        LEFT JOIN rev ON rev.user_id = u.id
        LEFT JOIN nudge nu ON nu.user_id = u.id
        LEFT JOIN plat p ON p.user_id = u.id
        WHERE (u.email ILIKE $1 OR u.name ILIKE $1 OR $1 = '%')
          AND COALESCE(u.subscription_tier, '') <> 'master_admin'
        ORDER BY COALESCE(rev.r, 0) DESC, COALESCE(wu.uc, 0) DESC
        LIMIT $4
        """,
        qpat,
        since,
        until,
        limit,
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        tier = str(r["subscription_tier"] or "free")
        if _is_internal_tier(tier):
            continue
        up = int(r["uploads_30d"] or 0)
        rev = float(r["revenue_30d"] or 0)
        plat = int(r["connected_accounts"] or 0)
        ns = int(r["n_shown"] or 0)
        nc = int(r["n_click"] or 0)
        ctr = round(100.0 * nc / max(ns, 1), 2) if ns else 0.0
        out.append(
            {
                "id": str(r["id"]),
                "email": r["email"],
                "name": r["name"],
                "subscription_tier": tier,
                "uploads_30d": up,
                "revenue_30d": rev,
                "nudge_ctr_pct": ctr,
                "connected_accounts": plat,
                "enterprise_fit_score": _enterprise_fit(up, rev, plat, tier),
            }
        )
    return out


async def _global_upload_baselines(conn) -> Dict[str, Any]:
    try:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*)::bigint AS n,
                COALESCE(AVG(views), 0)::float AS avg_views,
                COALESCE(AVG(likes), 0)::float AS avg_likes,
                COALESCE(AVG(comments), 0)::float AS avg_comments,
                COALESCE(AVG(shares), 0)::float AS avg_shares,
                COALESCE(AVG(
                    CASE WHEN COALESCE(views, 0) > 0 THEN
                        (COALESCE(likes, 0) + COALESCE(comments, 0) + COALESCE(shares, 0))::float
                        / NULLIF(views::float, 0) * 100.0
                    ELSE 0::float END
                ), 0)::float AS avg_engagement_rate_pct
            FROM uploads
            WHERE status IN ('completed', 'succeeded')
              AND created_at >= NOW() - interval '30 days'
            """
        )
    except Exception as e:
        logger.warning("global upload baselines query failed (schema drift or DB error): %s", e)
        return dict(_EMPTY_GLOBAL_BASELINES)
    return {
        "sample_uploads_30d": int(row["n"] or 0) if row else 0,
        "global_avg_views": float(row["avg_views"] or 0) if row else 0.0,
        "global_avg_likes": float(row["avg_likes"] or 0) if row else 0.0,
        "global_avg_comments": float(row["avg_comments"] or 0) if row else 0.0,
        "global_avg_shares": float(row["avg_shares"] or 0) if row else 0.0,
        "global_avg_engagement_rate_pct": float(row["avg_engagement_rate_pct"] or 0) if row else 0.0,
    }


async def fetch_user_engagement_snapshot(conn, user_id) -> Dict[str, Any]:
    """Per-user 30d engagement from uploads (views/likes/comments/shares = signal of what's hot)."""
    try:
        uid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
    except (ValueError, TypeError):
        return {"samples_30d": 0}
    try:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*)::bigint AS n,
                COALESCE(AVG(views), 0)::float AS avg_views,
                COALESCE(AVG(likes), 0)::float AS avg_likes,
                COALESCE(AVG(comments), 0)::float AS avg_comments,
                COALESCE(AVG(shares), 0)::float AS avg_shares,
                COALESCE(AVG(
                    CASE WHEN COALESCE(views, 0) > 0 THEN
                        (COALESCE(likes, 0) + COALESCE(comments, 0) + COALESCE(shares, 0))::float
                        / NULLIF(views::float, 0) * 100.0
                    ELSE 0::float END
                ), 0)::float AS engagement_rate_pct
            FROM uploads
            WHERE user_id = $1 AND status IN ('completed', 'succeeded')
              AND created_at >= NOW() - INTERVAL '30 days'
            """,
            uid,
        )
    except Exception as e:
        logger.warning("user engagement snapshot failed user_id=%s: %s", user_id, e)
        return {"samples_30d": 0}
    if not row or int(row["n"] or 0) < 1:
        return {"samples_30d": 0}
    return {
        "samples_30d": int(row["n"] or 0),
        "avg_views": round(float(row["avg_views"] or 0), 2),
        "avg_likes": round(float(row["avg_likes"] or 0), 2),
        "avg_comments": round(float(row["avg_comments"] or 0), 2),
        "avg_shares": round(float(row["avg_shares"] or 0), 2),
        "engagement_rate_pct": round(float(row["engagement_rate_pct"] or 0), 3),
    }


async def fetch_window_upload_engagement_aggregate(conn, since: datetime, until: datetime) -> Dict[str, Any]:
    """Creator-wide upload engagement for the selected reporting window."""
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*)::bigint AS n,
            COALESCE(AVG(views), 0)::float AS avg_views,
            COALESCE(AVG(likes), 0)::float AS avg_likes,
            COALESCE(AVG(comments), 0)::float AS avg_comments,
            COALESCE(AVG(shares), 0)::float AS avg_shares,
            COALESCE(AVG(
                CASE WHEN COALESCE(views, 0) > 0 THEN
                    (COALESCE(likes, 0) + COALESCE(comments, 0) + COALESCE(shares, 0))::float
                    / NULLIF(views::float, 0) * 100.0
                ELSE 0::float END
            ), 0)::float AS avg_engagement_rate_pct
        FROM uploads
        WHERE status IN ('completed', 'succeeded')
          AND created_at >= $1 AND created_at < $2
        """,
        since,
        until,
    )
    if not row:
        return {"sample_uploads": 0}
    return {
        "sample_uploads": int(row["n"] or 0),
        "avg_views": round(float(row["avg_views"] or 0), 2),
        "avg_likes": round(float(row["avg_likes"] or 0), 2),
        "avg_comments": round(float(row["avg_comments"] or 0), 2),
        "avg_shares": round(float(row["avg_shares"] or 0), 2),
        "avg_engagement_rate_pct": round(float(row["avg_engagement_rate_pct"] or 0), 3),
    }


async def fetch_platform_engagement_breakdown(conn, since: datetime, until: datetime, limit: int = 10) -> List[Dict[str, Any]]:
    """Per-platform upload counts + average engagement (likes+comments+shares vs views)."""
    rows = await conn.fetch(
        """
        SELECT
            TRIM(LOWER(unnest(platforms))) AS platform,
            COUNT(*)::bigint AS uploads,
            COALESCE(AVG(u.views), 0)::float AS avg_views,
            COALESCE(AVG(u.likes), 0)::float AS avg_likes,
            COALESCE(AVG(u.comments), 0)::float AS avg_comments,
            COALESCE(AVG(u.shares), 0)::float AS avg_shares,
            COALESCE(AVG(
                CASE WHEN COALESCE(u.views, 0) > 0 THEN
                    (COALESCE(u.likes, 0) + COALESCE(u.comments, 0) + COALESCE(u.shares, 0))::float
                    / NULLIF(u.views::float, 0) * 100.0
                ELSE 0::float END
            ), 0)::float AS avg_engagement_rate_pct
        FROM uploads u
        WHERE u.created_at >= $1 AND u.created_at < $2
          AND u.status IN ('completed', 'succeeded')
          AND u.platforms IS NOT NULL AND array_length(u.platforms, 1) > 0
        GROUP BY 1
        HAVING COUNT(*) >= 1
        ORDER BY uploads DESC, avg_engagement_rate_pct DESC NULLS LAST
        LIMIT $3
        """,
        since,
        until,
        limit,
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "platform": str(r["platform"] or ""),
                "uploads": int(r["uploads"] or 0),
                "avg_views": round(float(r["avg_views"] or 0), 2),
                "avg_likes": round(float(r["avg_likes"] or 0), 2),
                "avg_comments": round(float(r["avg_comments"] or 0), 2),
                "avg_shares": round(float(r["avg_shares"] or 0), 2),
                "avg_engagement_rate_pct": round(float(r["avg_engagement_rate_pct"] or 0), 3),
            }
        )
    return out


async def fetch_pci_engagement_by_platform(conn, since: datetime, until: datetime, limit: int = 8) -> List[Dict[str, Any]]:
    """Optional: synced platform_content_items totals (when table exists)."""
    try:
        rows = await conn.fetch(
            """
            SELECT
                platform,
                COUNT(*)::bigint AS items,
                COALESCE(SUM(views), 0)::bigint AS sum_views,
                COALESCE(SUM(likes), 0)::bigint AS sum_likes,
                COALESCE(SUM(comments), 0)::bigint AS sum_comments,
                COALESCE(SUM(shares), 0)::bigint AS sum_shares
            FROM platform_content_items
            WHERE COALESCE(metrics_synced_at, updated_at) >= $1
              AND COALESCE(metrics_synced_at, updated_at) < $2
            GROUP BY platform
            ORDER BY sum_views DESC
            LIMIT $3
            """,
            since,
            until,
            limit,
        )
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for r in rows:
        sv = int(r["sum_views"] or 0)
        eng = int(r["sum_likes"] or 0) + int(r["sum_comments"] or 0) + int(r["sum_shares"] or 0)
        out.append(
            {
                "platform": str(r["platform"] or ""),
                "items": int(r["items"] or 0),
                "sum_views": sv,
                "sum_likes": int(r["sum_likes"] or 0),
                "sum_comments": int(r["sum_comments"] or 0),
                "sum_shares": int(r["sum_shares"] or 0),
                "engagement_rate_pct": round(100.0 * eng / max(sv, 1), 3),
            }
        )
    return out


def _engagement_snapshot_from_coach_upload_row(row: Any) -> Dict[str, Any]:
    """Shape matches fetch_user_engagement_snapshot (30d completed uploads)."""
    if not row:
        return {"samples_30d": 0}
    n = int(row["samples_30d"] or 0)
    if n < 1:
        return {"samples_30d": 0}
    return {
        "samples_30d": n,
        "avg_views": round(float(row["avg_views"] or 0), 2),
        "avg_likes": round(float(row["avg_likes"] or 0), 2),
        "avg_comments": round(float(row["avg_comments"] or 0), 2),
        "avg_shares": round(float(row["avg_shares"] or 0), 2),
        "engagement_rate_pct": round(float(row["engagement_rate_pct"] or 0), 3),
    }


_COACH_UPLOAD_BUNDLE_SQL = """
SELECT
    COALESCE(u.subscription_tier, 'free') AS tier,
    COUNT(*) FILTER (WHERE u2.id IS NOT NULL AND u2.status IN ('completed', 'succeeded'))::bigint AS ok_uploads,
    COALESCE(AVG(u2.views) FILTER (WHERE u2.id IS NOT NULL AND u2.status IN ('completed', 'succeeded')), 0)::double precision AS my_avg_views,
    COALESCE(
        MAX(array_length(u2.platforms, 1)) FILTER (
            WHERE u2.platforms IS NOT NULL AND array_length(u2.platforms, 1) IS NOT NULL
        ),
        0
    )::int AS max_plat_spread,
    COUNT(*) FILTER (WHERE u2.id IS NOT NULL AND u2.status IN ('completed', 'succeeded'))::bigint AS samples_30d,
    COALESCE(AVG(u2.views) FILTER (WHERE u2.id IS NOT NULL AND u2.status IN ('completed', 'succeeded')), 0)::double precision AS avg_views,
    COALESCE(AVG(u2.likes) FILTER (WHERE u2.id IS NOT NULL AND u2.status IN ('completed', 'succeeded')), 0)::double precision AS avg_likes,
    COALESCE(AVG(u2.comments) FILTER (WHERE u2.id IS NOT NULL AND u2.status IN ('completed', 'succeeded')), 0)::double precision AS avg_comments,
    COALESCE(AVG(u2.shares) FILTER (WHERE u2.id IS NOT NULL AND u2.status IN ('completed', 'succeeded')), 0)::double precision AS avg_shares,
    COALESCE(AVG(
        CASE WHEN COALESCE(u2.views, 0) > 0 THEN
            (COALESCE(u2.likes, 0) + COALESCE(u2.comments, 0) + COALESCE(u2.shares, 0))::double precision
            / NULLIF(u2.views::double precision, 0) * 100.0
        ELSE 0::double precision END
    ) FILTER (WHERE u2.id IS NOT NULL AND u2.status IN ('completed', 'succeeded')), 0)::double precision AS engagement_rate_pct
FROM users u
LEFT JOIN uploads u2 ON u2.user_id = u.id
    AND u2.created_at >= NOW() - INTERVAL '30 days'
WHERE u.id = $1::uuid
GROUP BY u.id, u.subscription_tier
"""


async def _coach_fetch_upload_bundle(conn: Any, uid: uuid.UUID):
    try:
        return await conn.fetchrow(_COACH_UPLOAD_BUNDLE_SQL, uid)
    except Exception as e:
        logger.warning("coach upload bundle query failed user_id=%s: %s", uid, e)
        return None


async def _coach_parallel_prefs(conn: Any, uid: uuid.UUID):
    try:
        return await conn.fetchrow(
            """
            SELECT auto_captions, ai_hashtags_enabled
            FROM user_preferences WHERE user_id = $1
            """,
            uid,
        )
    except Exception as e:
        logger.warning("coach user_preferences unavailable user_id=%s: %s", uid, e)
        return None


async def _coach_parallel_wallet(conn: Any, uid: uuid.UUID):
    try:
        return await conn.fetchrow(
            "SELECT put_balance, put_reserved, aic_balance, aic_reserved FROM wallets WHERE user_id = $1", uid
        )
    except Exception as e:
        logger.warning("coach wallet row unavailable user_id=%s: %s", uid, e)
        return None


async def _coach_parallel_studio_n(conn: Any, uid: uuid.UUID) -> int:
    try:
        return int(
            await conn.fetchval(
                """
                SELECT COUNT(*)::bigint FROM studio_usage_events
                WHERE user_id = $1 AND created_at >= NOW() - INTERVAL '30 days'
                """,
                uid,
            )
            or 0
        )
    except Exception as e:
        logger.warning("coach studio_usage_events unavailable user_id=%s: %s", uid, e)
        return 0


async def _coach_parallel_content_insights(conn: Any, uid: uuid.UUID):
    try:
        from services.content_insights import build_user_content_insights

        return await build_user_content_insights(conn, uid)
    except Exception as e:
        logger.debug("coach content_attribution_insights: %s", e)
        return None


async def build_user_coach_payload(pool: Any, user_id) -> Dict[str, Any]:
    try:
        uid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
    except (ValueError, TypeError):
        return coach_endpoint_fallback("free")

    async def _acquire_run(coro):
        async with pool.acquire() as c:
            return await coro(c)

    async def _bl(c):
        return await _global_upload_baselines(c)

    async def _ub(c):
        return await _coach_fetch_upload_bundle(c, uid)

    async def _pr(c):
        return await _coach_parallel_prefs(c, uid)

    async def _wa(c):
        return await _coach_parallel_wallet(c, uid)

    async def _st(c):
        return await _coach_parallel_studio_n(c, uid)

    async def _ins(c):
        return await _coach_parallel_content_insights(c, uid)

    try:
        baselines, upload_row, prefs, wallet, studio_n, content_attribution_insights = await asyncio.gather(
            _acquire_run(_bl),
            _acquire_run(_ub),
            _acquire_run(_pr),
            _acquire_run(_wa),
            _acquire_run(_st),
            _acquire_run(_ins),
        )
    except Exception:
        logger.exception("coach parallel gather failed user_id=%s", user_id)
        return coach_endpoint_fallback(None)

    urow = upload_row
    if not urow:
        return sanitize_coach_payload_for_json(
            {
                "suggestions": [],
                "smart_offer": None,
                "baselines": baselines,
                "engagement_snapshot": {"samples_30d": 0},
                "m8_engine": m8_engine_identity_payload(),
                "content_attribution_insights": content_attribution_insights,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "tier": "free",
            }
        )

    eng_snap = _engagement_snapshot_from_coach_upload_row(urow)

    suggestions: List[Dict[str, Any]] = []
    tier = str(urow["tier"] or "free") if urow else "free"
    ok_u = int(urow["ok_uploads"] or 0) if urow else 0
    my_v = float(urow["my_avg_views"] or 0) if urow else 0.0
    g_v = float(baselines.get("global_avg_views") or 0)
    max_plat = int(urow["max_plat_spread"] or 0) if urow else 0
    g_er = float(baselines.get("global_avg_engagement_rate_pct") or 0)
    my_er = float(eng_snap.get("engagement_rate_pct") or 0) if eng_snap.get("samples_30d") else 0.0

    if ok_u >= 3 and g_v > 50 and my_v < g_v * 0.65:
        suggestions.append(
            {
                "id": "views_vs_cohort",
                "severity": "info",
                "title": "Thumbnails & packaging may be holding back views",
                "body": (
                    f"Your last-{ok_u} successful uploads average {my_v:.0f} views vs "
                    f"~{g_v:.0f} across all creators (30d). Likes, comments, and shares are the clearest "
                    f"signal of resonance — tighten titles, first-frame thumbnails, and hooks so viewers engage."
                ),
                "cta_label": "Open Thumbnail Studio",
                "cta_href": "/thumbnail-studio.html",
                "confidence": 0.72,
                "source": "aggregate_baseline",
            }
        )

    if ok_u >= 3 and eng_snap.get("samples_30d") and g_er > 0.03 and my_er >= g_er * 1.28 and my_v >= g_v * 0.85:
        suggestions.append(
            {
                "id": "engagement_hot_streak",
                "severity": "promo",
                "title": "Your engagement rate is beating the field",
                "body": (
                    f"Audience interaction (likes + comments + shares vs views) is ~{my_er:.2f}% vs "
                    f"~{g_er:.2f}% creator average. Double down on this packaging cadence and cross-post winners."
                ),
                "cta_label": "Review analytics",
                "cta_href": "/analytics.html",
                "confidence": 0.78,
                "source": "engagement_truth",
            }
        )

    if ok_u >= 3 and eng_snap.get("samples_30d") and g_er > 0.04 and my_v >= g_v * 0.7 and my_er < g_er * 0.55:
        suggestions.append(
            {
                "id": "engagement_weak_packaging",
                "severity": "warning",
                "title": "Views without reactions — tighten the hook",
                "body": (
                    f"You are near cohort on views (~{my_v:.0f} vs ~{g_v:.0f}) but engagement per view is low "
                    f"({my_er:.2f}% vs ~{g_er:.2f}% avg). Thumbnails, first 3s, and captions drive likes and comments."
                ),
                "cta_label": "Thumbnail Studio",
                "cta_href": "/thumbnail-studio.html",
                "confidence": 0.74,
                "source": "engagement_truth",
            }
        )

    if max_plat <= 1 and ok_u >= 2:
        suggestions.append(
            {
                "id": "multi_platform",
                "severity": "info",
                "title": "Cross-post to unlock reach",
                "body": "Uploads in the last month mostly hit a single destination. Multi-platform posts often compound discovery.",
                "cta_label": "Manage platforms",
                "cta_href": "/platforms.html",
                "confidence": 0.68,
                "source": "heuristic",
            }
        )

    tier_n = normalize_tier(tier)
    if tier_n in ("creator_pro", "studio", "agency", "friends_family", "lifetime"):
        try:
            n_plat = int(
                await conn.fetchval(
                    """
                    SELECT COUNT(*)::int FROM platform_tokens
                    WHERE user_id = $1 AND revoked_at IS NULL
                    """,
                    uid,
                )
                or 0
            )
            ent = get_entitlements_for_tier(tier_n)
            max_ac = int(ent.max_accounts or 0)
            if max_ac > 0 and n_plat < max_ac:
                headroom = max_ac - n_plat
                hi = tier_n in ("studio", "agency", "friends_family", "lifetime")
                if headroom >= 2 or (hi and headroom >= 1 and n_plat >= 2):
                    sev = "promo" if tier_n in ("studio", "agency") and headroom >= 3 else "info"
                    suggestions.append(
                        {
                            "id": "plan_account_headroom",
                            "severity": sev,
                            "title": "Your plan has unused connection slots",
                            "body": (
                                f"You can link up to {max_ac} channels ({n_plat} connected). "
                                "Add another platform account to widen distribution without upgrading."
                            ),
                            "cta_label": "Open Platforms",
                            "cta_href": "/platforms.html",
                            "confidence": 0.79,
                            "source": "entitlements_headroom",
                        }
                    )
        except Exception as e:
            logger.warning("coach platform_tokens headroom skipped user_id=%s: %s", user_id, e)

    _pref = dict(prefs) if prefs is not None else {}
    ac = bool(_pref.get("auto_captions"))
    ah = bool(_pref.get("ai_hashtags_enabled"))
    if not ac or not ah:
        suggestions.append(
            {
                "id": "ai_caption_stack",
                "severity": "warning",
                "title": "Turn on AI captions & hashtag assist",
                "body": "When both are off, you leave searchable text on the table. Enable in Settings → Preferences.",
                "cta_label": "Open Settings",
                "cta_href": "/settings.html#preferences-panel",
                "confidence": 0.74,
                "source": "settings_gap",
            }
        )

    put_avail = 0
    aic_avail = 0
    if wallet:
        put_avail = int(wallet["put_balance"] or 0) - int(wallet["put_reserved"] or 0)
        aic_avail = int(wallet["aic_balance"] or 0) - int(wallet["aic_reserved"] or 0)

    smart_offer = None
    if put_avail <= 35 and tier == "free":
        smart_offer = {
            "headline": "You are shipping on Free — unlock more PUT + AI",
            "body": "Creator Lite removes the weekly ceiling and funds captions/thumbnails more comfortably.",
            "cta_label": "Compare plans",
            "cta_href": "/index.html#pricing",
            "variant": "free_low_put",
            "confidence": 0.81,
        }
    elif aic_avail <= 12 and tier not in ("agency", "studio", "master_admin"):
        smart_offer = {
            "headline": "AI credits are running thin",
            "body": "Thumbnail Studio and vision captions spend AIC. Top up or upgrade before the next batch.",
            "cta_label": "Billing & tokens",
            "cta_href": "/settings.html#billing-panel",
            "variant": "low_aic",
            "confidence": 0.77,
        }

    if int(studio_n or 0) == 0 and ok_u >= 4 and tier not in ("free",):
        suggestions.append(
            {
                "id": "studio_adoption",
                "severity": "info",
                "title": "Try Thumbnail Studio on your next upload",
                "body": "You publish regularly but have not used Studio tools this month — data shows iterative thumbnails correlate with higher engagement variance.",
                "cta_label": "Launch Studio",
                "cta_href": "/thumbnail-studio.html",
                "confidence": 0.66,
                "source": "adoption_gap",
            }
        )

    def _suggestion_sort_key(x: Dict[str, Any]) -> float:
        c = x.get("confidence")
        try:
            v = float(c) if c is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
        return -v if math.isfinite(v) else 0.0

    suggestions.sort(key=_suggestion_sort_key)

    return sanitize_coach_payload_for_json(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "baselines": baselines,
            "engagement_snapshot": eng_snap,
            "smart_offer": smart_offer,
            "suggestions": suggestions[:8],
            "tier": tier,
            "m8_engine": m8_engine_identity_payload(),
            "content_attribution_insights": content_attribution_insights,
        }
    )


async def build_ai_truth_metrics(conn, range_key: str) -> Dict[str, Any]:
    """Snapshot fed into marketing strategist (OpenAI or deterministic)."""
    since, until = parse_range_since_until(range_key)
    funnel = await fetch_marketing_funnel(conn, since, until)
    levers = await fetch_sales_opportunity_levers(conn)
    base = await _global_upload_baselines(conn)

    paid_users = await conn.fetchval(
        """
        SELECT COUNT(*)::bigint FROM users
        WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime')
          AND subscription_status = 'active'
        """
    )
    active_users = await conn.fetchval(
        "SELECT COUNT(DISTINCT user_id)::bigint FROM uploads WHERE created_at >= $1 AND created_at < $2",
        since,
        until,
    )
    total_uploads = await conn.fetchval(
        "SELECT COUNT(*)::bigint FROM uploads WHERE created_at >= $1 AND created_at < $2",
        since,
        until,
    )
    revenue = await conn.fetchval(
        "SELECT COALESCE(SUM(amount), 0)::float FROM revenue_tracking WHERE created_at >= $1 AND created_at < $2",
        since,
        until,
    )
    mrr_est = await conn.fetchval(
        """
        SELECT COALESCE(SUM(
            CASE subscription_tier
                WHEN 'creator_lite' THEN 9.99 WHEN 'launch' THEN 9.99
                WHEN 'creator_pro' THEN 19.99
                WHEN 'studio' THEN 49.99
                WHEN 'agency' THEN 99.99
                ELSE 0 END
        ), 0)::float
        FROM users
        WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime')
          AND subscription_status = 'active'
        """
    )

    plat_rows = await conn.fetch(
        """
        SELECT unnest(platforms) AS platform, COUNT(*)::bigint AS c
        FROM uploads
        WHERE created_at >= $1 AND created_at < $2 AND platforms IS NOT NULL
        GROUP BY 1 ORDER BY c DESC LIMIT 12
        """,
        since,
        until,
    )
    platform_kpis = [{"platform": str(r["platform"]), "uploads": int(r["c"] or 0)} for r in plat_rows]

    promo = await fetch_promo_schedule_hints(conn, since, until)
    best_windows = [
        {"day": p["day"], "hour_utc": p["hour_utc"], "clicks": p["clicks"]} for p in promo[:5]
    ]

    eng_window = await fetch_window_upload_engagement_aggregate(conn, since, until)
    plat_eng = await fetch_platform_engagement_breakdown(conn, since, until, limit=10)
    pci_eng = await fetch_pci_engagement_by_platform(conn, since, until, limit=8)

    top_strategies: List[str] = []
    if funnel["ctr_pct"] >= 12:
        top_strategies.append("nudge_ctr_healthy_double_down_in_app")
    if levers["users_low_put_available_0_29"] > 10:
        top_strategies.append("token_pressure_micro_tx_bundle")
    if levers["free_users_uploading_last_7d"] > 5:
        top_strategies.append("free_active_to_creator_lite_sequence")
    if base["global_avg_views"] and base["global_avg_views"] < 200:
        top_strategies.append("thumbnail_quality_education_push")
    er_g = float(base.get("global_avg_engagement_rate_pct") or 0)
    er_w = float(eng_window.get("avg_engagement_rate_pct") or 0)
    if eng_window.get("sample_uploads", 0) > 30 and er_w > er_g * 1.15 and er_g > 0.02:
        top_strategies.append("engagement_up_cycle_double_down_content_style")
    if eng_window.get("sample_uploads", 0) > 30 and er_g > 0.03 and er_w < er_g * 0.75:
        top_strategies.append("engagement_down_education_thumbnails_hooks")

    margin_guess = 0.0
    if mrr_est and float(mrr_est) > 0:
        margin_guess = min(95.0, max(5.0, 100.0 - 100.0 * float(revenue or 0) / max(float(mrr_est), 1) * 0.15))

    return {
        "m8_engine": m8_engine_identity_payload(),
        "kpis": {
            "active_users": int(active_users or 0),
            "paid_users": int(paid_users or 0),
            "total_uploads": int(total_uploads or 0),
            "total_revenue": float(revenue or 0),
            "mrr_estimate": float(mrr_est or 0),
            "nudge_ctr_pct": float(funnel["ctr_pct"]),
            "gross_margin_pct": round(margin_guess, 2),
        },
        "segment_signals": {
            "free_high_intent_uploaders": int(levers["free_users_uploading_last_7d"]),
            "token_pressure_accounts": int(levers["users_low_put_available_0_29"]),
            "expansion_ready_accounts": int(levers["users_3plus_platform_connections"]),
            "engaged_no_purchase_accounts": int(levers["free_users_uploading_last_7d"]),
        },
        "platform_kpis": platform_kpis,
        "best_promo_windows_utc": best_windows,
        "ml_truth": {"top_strategies": top_strategies},
        "upload_baselines": base,
        "engagement_truth": {
            "window": eng_window,
            "platform_upload_engagement": plat_eng,
            "platform_synced_metrics": pci_eng,
            "note": "Engagement = likes + comments + shares relative to views on completed uploads; synced rows when PCI present.",
        },
    }
