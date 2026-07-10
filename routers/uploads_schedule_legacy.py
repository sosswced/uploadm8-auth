"""Deprecated smart-schedule preview path under /api/uploads (compat shim)."""

from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

import core.state
from core.deps import get_current_user_readonly
from services.scheduling_preview import preview_response_payload
from services.upload.schedule_guard import (
    _user_timezone,
    build_smart_schedule_for_upload,
    schedule_slot_iso,
)

legacy_uploads_router = APIRouter(prefix="/api/uploads", tags=["uploads"])


@legacy_uploads_router.post("/smart-schedule/preview")
async def preview_smart_schedule_legacy(
    response: Response,
    platforms: List[str] = Query(...),
    days: int = Query(14, ge=1, le=730),
    seed: Optional[str] = Query(None),
    user: dict = Depends(get_current_user_readonly),
):
    """Deprecated — use POST /api/scheduling/preview."""
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = '</api/scheduling/preview>; rel="successor-version"'
    response.headers["Sunset"] = "2026-09-01"

    if not platforms:
        raise HTTPException(400, "At least one platform required")

    bill_id = str(user.get("billing_user_id") or user["id"])
    plats = [p.strip().lower() for p in platforms if p and str(p).strip()]
    schedule_seed = (seed or "").strip() or str(uuid.uuid4())

    if core.state.db_pool is None:
        raise HTTPException(503, "Database unavailable")

    async with core.state.db_pool.acquire() as conn:
        tz = await _user_timezone(conn, bill_id)
        schedule = await build_smart_schedule_for_upload(
            conn,
            bill_id,
            plats,
            num_days=days,
            random_seed=schedule_seed,
        )

    if not schedule:
        raise HTTPException(500, "Could not generate smart schedule preview")

    sm = {p: schedule_slot_iso(dt) for p, dt in schedule.items()}
    return preview_response_payload(
        schedule,
        sm,
        seed=schedule_seed,
        smart_schedule_days=days,
        user_timezone=tz,
    )
