"""
UploadM8 Dashboard routes — extracted from app.py.
"""

from fastapi import APIRouter, Depends, HTTPException

import core.state
from core.deps import get_verified_user_id
from services.dashboard_user_stats import fetch_dashboard_stats_for_user_id

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats")
async def get_dashboard_stats(user_id: str = Depends(get_verified_user_id)):
    """
    Dashboard stats: uploads, canonical engagement (30d), quota, wallet, accounts, recent.

    Uses ``get_verified_user_id`` (JWT only) plus ``fetch_dashboard_stats_for_user_id`` so user,
    wallet, and stats share **one** pooled connection (Sentry UPLOADM8-2B).
    """
    pool = core.state.db_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return await fetch_dashboard_stats_for_user_id(pool, user_id)


@router.get("")
async def get_dashboard_alias(user_id: str = Depends(get_verified_user_id)):
    """Alias for GET /api/dashboard/stats — some frontends call /api/dashboard."""
    pool = core.state.db_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return await fetch_dashboard_stats_for_user_id(pool, user_id)
