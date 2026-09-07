"""Connection readiness endpoints — can scheduled work actually publish?

Thin handlers over ``services.oauth_readiness``; the risk math lives there.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

import core.state
from core.deps import get_current_user_readonly, require_master_admin
from services.workspace import resolve_billing_user_id

router = APIRouter(tags=["platforms", "oauth"])


@router.get("/api/me/connection-readiness")
async def my_connection_readiness(
    detail_limit: int = Query(50, ge=0, le=500),
    user: dict = Depends(get_current_user_readonly),
):
    """
    Health of my connections plus any scheduled posts they cannot cover.

    Smart Schedule windows run far ahead, so this reports the refresh-grant
    horizon per account and flags posts scheduled beyond it.
    """
    from services.oauth_readiness import connection_readiness_report

    if core.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    bill_id = await resolve_billing_user_id(user)
    async with core.state.db_pool.acquire() as conn:
        return await connection_readiness_report(
            conn, user_id=str(bill_id), detail_limit=int(detail_limit)
        )


@router.get("/api/admin/oauth-readiness")
async def admin_oauth_readiness(
    user_id: Optional[str] = Query(None, description="Scope to one user"),
    detail_limit: int = Query(200, ge=0, le=2000),
    user: dict = Depends(require_master_admin),
):
    """Fleet-wide connection readiness across every platform, user, and video."""
    from services.oauth_readiness import connection_readiness_report

    if core.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    async with core.state.db_pool.acquire() as conn:
        return await connection_readiness_report(
            conn, user_id=user_id, detail_limit=int(detail_limit)
        )
