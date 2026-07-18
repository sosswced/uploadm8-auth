"""Combined payload for dashboard / queue first paint (one round-trip)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping, Optional

from core.db_pool import acquire_db
from stages.entitlements import entitlements_to_dict, get_entitlements_from_user
from services.dashboard_user_stats import dashboard_stats_for_user
from services.platform_accounts import (
    _PLATFORM_TOKEN_SELECT,
    fetch_auth_errors_by_token,
    serialize_platform_account,
)
from services.upload.schedule_guard import bootstrap_repair_user_schedules
from services.uploads_handlers import (
    UPLOAD_VIEW_STATUS,
    fetch_upload_queue_stats,
    fetch_user_uploads_list,
)

BOOTSTRAP_SCHEDULE_REPAIR_TIMEOUT_SEC = 20.0
BOOTSTRAP_SCHEDULE_REPAIR_LIMIT = 12


async def _bootstrap_schedule_repair(pool: Any, user_id: str) -> dict[str, int] | None:
    try:
        return await asyncio.wait_for(
            bootstrap_repair_user_schedules(pool, user_id, limit=BOOTSTRAP_SCHEDULE_REPAIR_LIMIT),
            timeout=BOOTSTRAP_SCHEDULE_REPAIR_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "shell_bootstrap schedule repair timed out after %.0fs user=%s",
            BOOTSTRAP_SCHEDULE_REPAIR_TIMEOUT_SEC,
            user_id,
        )
    except Exception:
        logger.exception("shell_bootstrap schedule repair failed user=%s", user_id)
    return None


async def run_schedule_repair_background(pool: Any, user_id: str) -> None:
    """Fire-and-forget schedule repair — must not block first paint."""
    await _bootstrap_schedule_repair(pool, user_id)

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
    """Same JSON shape as GET /api/platforms (status + reconnect timestamps included).

    Avatars use redirect paths (``presign=False``) for cheap bootstrap first paint.
    """
    async with acquire_db(pool) as conn:
        auth_errors = await fetch_auth_errors_by_token(conn, user_id)
        accounts = await conn.fetch(_PLATFORM_TOKEN_SELECT, user_id)

    platforms: dict[str, list] = {}
    for acc in accounts:
        p = acc["platform"]
        if p not in platforms:
            platforms[p] = []
        platforms[p].append(
            serialize_platform_account(
                acc,
                auth_error_by_token=auth_errors,
                presign=False,
            )
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
    plan = entitlements_to_dict(get_entitlements_from_user(user))
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
            "schedule_repair": None,
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
            logger.warning(
                "queue bootstrap: uploads and queue_stats both failed user=%s — null fields for frontend fallback",
                uid,
            )
            return {
                "context": "queue",
                "dashboard_stats": None,
                "queue_stats": None,
                "uploads": None,
                "platforms": platforms_payload,
                "schedule_repair": None,
            }
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
            "schedule_repair": None,
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
        get_user_preferences(include_personas=True, user=user),
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


# Map KPI UI range presets → analytics_overview ``days``.
# ``all`` is handled via range=all on analytics_overview (ALL_TIME_FLOOR_UTC).
_KPI_RANGE_DAYS = {"7d": 7, "30d": 30, "90d": 90, "365d": 365, "1y": 365}


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
    rk = (range or "30d").strip().lower()
    days = int(_KPI_RANGE_DAYS.get(rk, 30))
    workspace_id = (user.get("workspace") or {}).get("id")

    async def _insights() -> Any:
        async with acquire_db(pool) as conn:
            return await build_user_content_insights(conn, uid_plain)

    overview_kwargs: dict[str, Any] = {
        "platform": platform or "all",
        "user": user,
    }
    if rk == "all":
        # Align with get_analytics(range=all) / ALL_TIME_FLOOR_UTC — not a 3650d clamp.
        overview_kwargs["range"] = "all"
    else:
        overview_kwargs["days"] = days

    results = await asyncio.gather(
        analytics_overview(**overview_kwargs),
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
