"""
UploadM8 Dashboard routes — extracted from app.py.
"""

from fastapi import APIRouter, Depends

import core.state
from core.deps import get_current_user_readonly
from core.helpers import get_plan
from services.dashboard_user_stats import dashboard_stats_for_user

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats")
async def get_dashboard_stats(user: dict = Depends(get_current_user_readonly)):
    """Dashboard stats: uploads, canonical engagement (30d), quota, wallet, accounts, recent."""
    plan = get_plan(user.get("subscription_tier", "free"))
    wallet = user.get("wallet", {})
    return await dashboard_stats_for_user(core.state.db_pool, user, plan, wallet)


@router.get("")
async def get_dashboard_alias(user: dict = Depends(get_current_user_readonly)):
    """Alias for GET /api/dashboard/stats — some frontends call /api/dashboard."""
    return await get_dashboard_stats(user)
