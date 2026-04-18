"""Single round-trip bootstrap for dashboard + queue first paint."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

import core.state
from core.deps import get_current_user_readonly
from services.shell_bootstrap import _allowed_upload_view, shell_bootstrap_payload

router = APIRouter(prefix="/api/shell", tags=["shell"])


@router.get("/bootstrap")
async def get_shell_bootstrap(
    context: str = Query("dashboard", pattern="^(dashboard|queue)$"),
    upload_limit: int = Query(200, ge=1, le=500),
    upload_view: Optional[str] = Query(None, description="Same as GET /api/uploads view= (pending, processing, …)"),
    meta: bool = Query(False),
    user: dict = Depends(get_current_user_readonly),
):
    """
    One response for first paint: stats + uploads list + platforms bundle.

    - ``context=dashboard`` → ``dashboard_stats`` (same as GET /api/dashboard/stats) + uploads + platforms.
    - ``context=queue`` → ``queue_stats`` (same as GET /api/uploads/queue-stats) + uploads + platforms.
    """
    if upload_view is not None and upload_view != "" and _allowed_upload_view(upload_view) is None:
        raise HTTPException(400, detail="Invalid upload_view")

    pool = core.state.db_pool
    if pool is None:
        raise HTTPException(503, detail="Database unavailable")

    return await shell_bootstrap_payload(
        pool,
        user,
        context=context,
        upload_limit=upload_limit,
        upload_view=upload_view,
        meta=meta,
    )
