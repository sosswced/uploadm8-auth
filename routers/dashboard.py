"""
UploadM8 Dashboard routes — extracted from app.py.
"""

from fastapi import APIRouter, Depends

import core.state
from core.deps import get_current_user
from core.helpers import get_plan

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats")
async def get_dashboard_stats(user: dict = Depends(get_current_user)):
    """Dashboard stats for user: uploads, quota, success rate, accounts, scheduled."""
    plan = get_plan(user.get("subscription_tier", "free"))
    wallet = user.get("wallet", {})

    async with core.state.db_pool.acquire() as conn:
        stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total,
                   SUM(CASE WHEN status IN ('completed','succeeded','partial') THEN 1 ELSE 0 END)::int AS completed,
                   SUM(CASE WHEN status IN ('pending', 'queued', 'processing') THEN 1 ELSE 0 END)::int AS in_queue,
                   COALESCE(SUM(views), 0)::bigint AS views,
                   COALESCE(SUM(likes), 0)::bigint AS likes
            FROM uploads WHERE user_id = $1
        """, user["id"])
        # Scheduled: pending/staged/ready_to_publish with schedule_mode in scheduled/smart
        scheduled = await conn.fetchval("""
            SELECT COUNT(*)::int FROM uploads
            WHERE user_id = $1
              AND status IN ('pending','staged','queued','scheduled','ready_to_publish')
              AND schedule_mode IN ('scheduled','smart')
              AND scheduled_time IS NOT NULL
        """, user["id"])
        # Monthly PUT used (current month)
        try:
            put_used_month = await conn.fetchval("""
                SELECT COALESCE(SUM(put_spent), 0)::int FROM uploads
                WHERE user_id = $1 AND created_at >= date_trunc('month', CURRENT_DATE)
            """, user["id"])
        except Exception:
            put_used_month = 0
        # Connected accounts (exclude revoked)
        try:
            accounts = await conn.fetchval(
                "SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1 AND (revoked_at IS NULL OR revoked_at > NOW())",
                user["id"],
            )
        except Exception:
            accounts = await conn.fetchval("SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1", user["id"])
        recent = await conn.fetch(
            "SELECT id, filename, platforms, status, created_at FROM uploads WHERE user_id = $1 ORDER BY created_at DESC LIMIT 5",
            user["id"],
        )

    total = stats["total"] if stats else 0
    completed = stats["completed"] if stats else 0
    put_avail = wallet.get("put_balance", 0) - wallet.get("put_reserved", 0)
    aic_avail = wallet.get("aic_balance", 0) - wallet.get("aic_reserved", 0)
    put_monthly = plan.get("put_monthly", 60)
    success_rate = (completed / max(total, 1)) * 100 if total else 0

    # Credits display: PUT/AIC wallet balances (not monthly quota)
    put_reserved = float(wallet.get("put_reserved", 0) or 0)
    aic_reserved = float(wallet.get("aic_reserved", 0) or 0)
    put_total = float(wallet.get("put_balance", 0) or 0)
    aic_total = float(wallet.get("aic_balance", 0) or 0)

    return {
        "uploads": {"total": total, "completed": completed, "in_queue": stats["in_queue"] if stats else 0},
        "engagement": {"views": stats["views"] if stats else 0, "likes": stats["likes"] if stats else 0},
        "success_rate": round(success_rate, 1),
        "scheduled": scheduled or 0,
        "quota": {"put_used": put_used_month or 0, "put_limit": put_monthly},
        "wallet": {"put_available": put_avail, "put_total": put_total, "aic_available": aic_avail, "aic_total": aic_total},
        "credits": {
            "put": {"available": put_avail, "reserved": put_reserved, "total": put_total, "monthly_allowance": put_monthly},
            "aic": {"available": aic_avail, "reserved": aic_reserved, "total": aic_total, "monthly_allowance": plan.get("aic_monthly", 0)},
        },
        "accounts": {"connected": accounts or 0, "limit": plan.get("max_accounts", 1)},
        "recent": [{"id": str(r["id"]), "filename": r["filename"], "platforms": r["platforms"], "status": r["status"]} for r in recent],
        "plan": plan,
    }


@router.get("")
async def get_dashboard_alias(user: dict = Depends(get_current_user)):
    """Alias for GET /api/dashboard/stats — some frontends call /api/dashboard."""
    return await get_dashboard_stats(user)
