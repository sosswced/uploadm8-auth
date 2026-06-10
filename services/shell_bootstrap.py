"""Combined payload for dashboard / queue first paint (one round-trip)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping, Optional

from core.helpers import get_plan
from core.r2 import resolve_stored_account_avatar_url
from services.dashboard_user_stats import dashboard_stats_for_user
from services.upload.schedule_guard import bootstrap_repair_user_schedules
from services.uploads_handlers import (
    UPLOAD_VIEW_STATUS,
    _dt_iso,
    fetch_upload_queue_stats,
    fetch_user_uploads_list,
)

BOOTSTRAP_SCHEDULE_REPAIR_TIMEOUT_SEC = 8.0


async def _bootstrap_schedule_repair(pool: Any, user_id: str) -> dict[str, int] | None:
    try:
        return await asyncio.wait_for(
            bootstrap_repair_user_schedules(pool, user_id, limit=30),
            timeout=BOOTSTRAP_SCHEDULE_REPAIR_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.error(
            "shell_bootstrap schedule repair timed out after %.0fs user=%s",
            BOOTSTRAP_SCHEDULE_REPAIR_TIMEOUT_SEC,
            user_id,
        )
    except Exception:
        logger.exception("shell_bootstrap schedule repair failed user=%s", user_id)
    return None

logger = logging.getLogger(__name__)


def _allowed_upload_view(raw: Optional[str]) -> Optional[str]:
    if raw is None or raw == "":
        return None
    if raw == "all":
        return "all"
    if raw in UPLOAD_VIEW_STATUS:
        return raw
    return None


async def _fetch_platforms_bundle(pool: Any, user_id: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Same JSON shape as GET /api/platforms."""
    async with pool.acquire() as conn:
        accounts = await conn.fetch(
            """
            SELECT DISTINCT ON (platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''))
                id, platform, account_id, account_name, account_username, account_avatar, is_primary, created_at
            FROM platform_tokens
            WHERE user_id = $1
              AND revoked_at IS NULL
              AND account_id IS NOT NULL AND account_id <> ''
              AND (COALESCE(account_username,'') <> '' OR COALESCE(account_name,'') <> '')
            ORDER BY platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''), created_at DESC
            """,
            user_id,
        )

    platforms: dict[str, list] = {}
    for acc in accounts:
        p = acc["platform"]
        if p not in platforms:
            platforms[p] = []
        platforms[p].append(
            {
                "id": str(acc["id"]),
                "account_id": acc["account_id"],
                "name": acc["account_name"],
                "username": acc["account_username"],
                "avatar": resolve_stored_account_avatar_url(acc["account_avatar"], presign=False),
                "is_primary": acc["is_primary"],
                "status": "active",
                "connected_at": _dt_iso(acc["created_at"]),
            }
        )

    total = sum(len(v) for v in platforms.values())
    max_accounts = int(plan.get("max_accounts", 1) or 1)
    return {
        "platforms": platforms,
        "total_accounts": total,
        "max_accounts": max_accounts,
        "can_add_more": total < max_accounts,
    }


async def shell_bootstrap_payload(
    pool: Any,
    user: dict[str, Any],
    *,
    context: str,
    upload_limit: int,
    upload_view: Optional[str],
    meta: bool,
    range: str = "30d",
    platform: str = "all",
) -> dict[str, Any]:
    """
    context=dashboard: dashboard_stats + uploads + platforms (queue_stats omitted).
    context=queue: queue_stats + uploads + platforms (dashboard_stats omitted).

    Uses ``asyncio.gather`` so each sub-call may ``pool.acquire()`` on its own connection
    (Sentry: multiple ``pg_advisory_unlock_all`` spans are pool returns, not app advisory locks).
    """
    uid = str(user.get("billing_user_id") or user["id"])
    plan = get_plan(user.get("subscription_tier", "free"))
    wallet = user.get("wallet") or {}
    view = _allowed_upload_view(upload_view)

    if context == "upload":
        return await _upload_bootstrap_payload(user)

    if context == "kpi":
        return await _kpi_bootstrap_payload(
            pool, user, range=range, upload_limit=upload_limit, platform=platform
        )

    uploads_coro = fetch_user_uploads_list(
        pool,
        uid,
        status=None,
        view=view,
        limit=upload_limit,
        offset=0,
        trill_only=False,
        meta=meta,
        presign_r2_thumbnails=False,
        presign_platform_avatars=False,
    )
    platforms_coro = _fetch_platforms_bundle(pool, uid, plan)

    schedule_repair = await _bootstrap_schedule_repair(pool, uid)

    if context == "dashboard":
        stats_coro = dashboard_stats_for_user(pool, user, plan, wallet)
        try:
            dashboard_stats, uploads_payload, platforms_payload = await asyncio.gather(
                stats_coro, uploads_coro, platforms_coro
            )
        except Exception:
            logger.exception("shell_bootstrap dashboard gather failed user=%s", uid)
            raise
        return {
            "context": "dashboard",
            "dashboard_stats": dashboard_stats,
            "queue_stats": None,
            "uploads": uploads_payload,
            "platforms": platforms_payload,
            "schedule_repair": schedule_repair,
        }

    if context == "queue":

        qs_coro = fetch_upload_queue_stats(pool, uid)
        results = await asyncio.gather(
            uploads_coro, platforms_coro, qs_coro, return_exceptions=True
        )
        uploads_payload = _ok(results[0])
        platforms_payload = _ok(results[1])
        queue_stats = _ok(results[2])
        if uploads_payload is None and queue_stats is None:
            raise RuntimeError("queue bootstrap: uploads and queue_stats both failed")
        return {
            "context": "queue",
            "dashboard_stats": None,
            "queue_stats": queue_stats or {
                "processing": 0,
                "completed": 0,
                "partial": 0,
                "failed": 0,
            },
            "uploads": uploads_payload if uploads_payload is not None else [],
            "platforms": platforms_payload,
            "schedule_repair": schedule_repair,
        }

    raise ValueError("context must be dashboard, queue, upload or kpi")


def _ok(value: Any) -> Any:
    """Coerce a gather(return_exceptions=True) slot to None on failure so the
    frontend can fall back per-field instead of losing the whole bootstrap."""
    if isinstance(value, BaseException):
        logger.warning("shell_bootstrap sub-call failed: %s", value)
        return None
    return value


async def _upload_bootstrap_payload(user: dict[str, Any]) -> dict[str, Any]:
    """context=upload: preferences + platform accounts + groups (matches upload page loaders).

    Calls the existing read handlers directly (they each acquire their own
    connection, so ``asyncio.gather`` runs them concurrently). Per-field None on
    failure lets the frontend fall back to the individual endpoints.
    """
    from routers.groups import get_groups
    from routers.platforms import get_platform_accounts
    from routers.preferences import get_user_preferences

    results = await asyncio.gather(
        get_user_preferences(include_personas=False, user=user),
        get_platform_accounts(user=user),
        get_groups(user=user),
        return_exceptions=True,
    )
    preferences, platform_accounts, groups = (_ok(r) for r in results)
    return {
        "context": "upload",
        "preferences": preferences,
        "platform_accounts": platform_accounts,
        "groups": groups,
    }


_KPI_RANGE_DAYS = {"7d": 7, "30d": 30, "90d": 90, "365d": 365, "1y": 365, "all": 3650}


async def _kpi_bootstrap_payload(
    pool: Any,
    user: dict[str, Any],
    *,
    range: str,
    upload_limit: int,
    platform: str = "all",
) -> dict[str, Any]:
    """context=kpi: analytics overview + range analytics + uploads(meta) + content insights.

    Each sub-call acquires its own connection so ``asyncio.gather`` collapses the
    serial phase barrier in kpi.html's loadData into a single round-trip.
    """
    from routers.analytics import analytics_overview, get_analytics
    from services.content_insights import build_user_content_insights

    uid_billing = str(user.get("billing_user_id") or user["id"])
    uid_plain = str(user["id"])
    days = _KPI_RANGE_DAYS.get((range or "30d").strip().lower(), 30)
    workspace_id = (user.get("workspace") or {}).get("id")

    async def _insights() -> Any:
        async with pool.acquire() as conn:
            return await build_user_content_insights(conn, uid_plain)

    results = await asyncio.gather(
        analytics_overview(days=days, platform=platform or "all", user=user),
        get_analytics(
            range=range,
            trill_vehicle_make=None,
            trill_vehicle_model=None,
            trill_vehicle_make_id=None,
            trill_vehicle_model_id=None,
            user=user,
        ),
        fetch_user_uploads_list(
            pool,
            uid_billing,
            status=None,
            view=None,
            limit=upload_limit,
            offset=0,
            trill_only=False,
            meta=True,
            workspace_id=workspace_id,
        ),
        _insights(),
        return_exceptions=True,
    )
    overview, analytics, uploads, insights = (_ok(r) for r in results)
    return {
        "context": "kpi",
        "analytics_overview": overview,
        "analytics": analytics,
        "uploads": uploads,
        "content_insights": insights,
    }
