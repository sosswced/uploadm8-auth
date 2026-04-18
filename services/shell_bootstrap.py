"""Combined payload for dashboard / queue first paint (one round-trip)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping, Optional

from core.helpers import get_plan
from core.r2 import resolve_stored_account_avatar_url
from services.dashboard_user_stats import dashboard_stats_for_user
from services.uploads_handlers import UPLOAD_VIEW_STATUS, fetch_upload_queue_stats, fetch_user_uploads_list

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
                "avatar": resolve_stored_account_avatar_url(acc["account_avatar"]),
                "is_primary": acc["is_primary"],
                "status": "active",
                "connected_at": acc["created_at"].isoformat() if acc["created_at"] else None,
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
) -> dict[str, Any]:
    """
    context=dashboard: dashboard_stats + uploads + platforms (queue_stats omitted).
    context=queue: queue_stats + uploads + platforms (dashboard_stats omitted).
    """
    uid = str(user["id"])
    plan = get_plan(user.get("subscription_tier", "free"))
    wallet = user.get("wallet") or {}
    view = _allowed_upload_view(upload_view)

    uploads_coro = fetch_user_uploads_list(
        pool,
        uid,
        status=None,
        view=view,
        limit=upload_limit,
        offset=0,
        trill_only=False,
        meta=meta,
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
        }

    if context == "queue":
        qs_coro = fetch_upload_queue_stats(pool, uid)
        try:
            uploads_payload, platforms_payload, queue_stats = await asyncio.gather(
                uploads_coro, platforms_coro, qs_coro
            )
        except Exception:
            logger.exception("shell_bootstrap queue gather failed user=%s", uid)
            raise
        return {
            "context": "queue",
            "dashboard_stats": None,
            "queue_stats": queue_stats,
            "uploads": uploads_payload,
            "platforms": platforms_payload,
        }

    raise ValueError("context must be dashboard or queue")
