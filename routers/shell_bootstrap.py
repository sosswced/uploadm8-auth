"""Single round-trip bootstrap for dashboard + queue first paint."""

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

import core.state
from core.deps import get_current_user_readonly
from routers.uploads import _schedule_thumbnail_repair
from services.shell_bootstrap import _allowed_upload_view, shell_bootstrap_payload

router = APIRouter(prefix="/api/shell", tags=["shell"])


@router.get("/bootstrap")
async def get_shell_bootstrap(
    background_tasks: BackgroundTasks,
    context: str = Query("dashboard", pattern="^(dashboard|queue|upload|kpi)$"),
    upload_limit: int = Query(200, ge=1, le=800),
    upload_view: Optional[str] = Query(None, description="Same as GET /api/uploads view= (pending, processing, …)"),
    meta: bool = Query(False),
    range: str = Query("30d", description="Analytics range for context=kpi (e.g. 7d, 30d, 90d, 1y, all)"),
    platform: str = Query("all", description="Platform filter for context=kpi analytics overview"),
    user: dict = Depends(get_current_user_readonly),
):
    """
    One response for first paint: stats + uploads list + platforms bundle.

    - ``context=dashboard`` → ``dashboard_stats`` (same as GET /api/dashboard/stats) + uploads + platforms.
    - ``context=queue`` → ``queue_stats`` (same as GET /api/uploads/queue-stats) + uploads + platforms.
    - ``context=upload`` → ``preferences`` + ``platform_accounts`` + ``groups`` (matches upload page loaders).
    - ``context=kpi`` → ``analytics_overview`` + ``analytics`` (range) + ``uploads`` (meta) + ``content_insights``.
    """
    if upload_view is not None and upload_view != "" and _allowed_upload_view(upload_view) is None:
        raise HTTPException(400, detail="Invalid upload_view")

    pool = core.state.db_pool
    if pool is None:
        raise HTTPException(503, detail="Database unavailable")

    uid = str(user.get("billing_user_id") or user["id"])
    payload = await shell_bootstrap_payload(
        pool,
        user,
        context=context,
        upload_limit=upload_limit,
        upload_view=upload_view,
        meta=meta,
        range=range,
        platform=platform,
    )
    uploads = payload.get("uploads") if isinstance(payload, dict) else None
    _schedule_thumbnail_repair(background_tasks, uid, uploads if isinstance(uploads, list) else [])
    return payload
