"""Smart schedule preview — no upload row required."""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import core.state
from core.deps import get_current_user_readonly
from services.scheduling_preview import preview_response_payload
from services.upload.schedule_guard import (
    _user_timezone,
    build_smart_schedule_for_upload,
    schedule_slot_iso,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scheduling", tags=["scheduling"])


class SchedulePreviewRequest(BaseModel):
    platforms: List[str] = Field(..., min_length=1)
    smart_schedule_days: int = Field(14, ge=1, le=730)
    seed: Optional[str] = Field(
        None,
        description="Optional seed for reproducible preview; omit for a fresh draw",
    )


@router.post("/preview")
async def preview_smart_schedule(
    body: SchedulePreviewRequest,
    user: dict = Depends(get_current_user_readonly),
):
    """
    Preview per-platform smart times without creating an upload row.

    Pass ``seed`` (same value as presign ``smart_schedule_seed``) so preview matches final slots.
    """
    bill_id = str(user.get("billing_user_id") or user["id"])
    platforms = [p.strip().lower() for p in body.platforms if p and str(p).strip()]
    if not platforms:
        raise HTTPException(400, "Select at least one platform")

    pool = core.state.db_pool
    if pool is None:
        raise HTTPException(503, "Database unavailable")

    seed = (body.seed or "").strip() or str(uuid.uuid4())

    async with pool.acquire() as conn:
        tz = await _user_timezone(conn, bill_id)
        smart = await build_smart_schedule_for_upload(
            conn,
            bill_id,
            platforms,
            num_days=body.smart_schedule_days,
            random_seed=seed,
        )

    if not smart:
        raise HTTPException(
            500,
            detail={
                "code": "schedule_generation_failed",
                "message": "Could not generate smart schedule preview.",
            },
        )

    sm = {p: schedule_slot_iso(dt) for p, dt in smart.items()}
    return preview_response_payload(
        smart,
        sm,
        seed=seed,
        smart_schedule_days=body.smart_schedule_days,
        user_timezone=tz,
    )
