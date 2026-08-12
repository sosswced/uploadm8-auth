"""Smart schedule preview — no upload row required."""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import core.state
from core.deps import get_current_user_readonly
from core.helpers import _now_utc
from services.scheduling_preview import occupancy_from_schedule, preview_response_payload
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
    batch_count: int = Field(
        1,
        ge=1,
        le=100,
        description="Number of videos in the batch (staggered seeds + packed occupancy)",
    )
    video_labels: Optional[List[str]] = Field(
        None,
        description="Optional display names per video (same order as batch slots)",
    )


@router.post("/preview")
async def preview_smart_schedule(
    body: SchedulePreviewRequest,
    user: dict = Depends(get_current_user_readonly),
):
    """
    Preview per-platform smart times without creating an upload row.

    Pass ``seed`` (same value as presign ``smart_schedule_seed``) so preview matches final slots.
    When ``batch_count`` > 1, returns ``batch[]`` with one schedule per video using
    ``{seed}:slot-{i}`` (same stagger as upload.html / orchestrator).
    """
    bill_id = str(user.get("billing_user_id") or user["id"])
    platforms = [p.strip().lower() for p in body.platforms if p and str(p).strip()]
    if not platforms:
        raise HTTPException(400, "Select at least one platform")

    pool = core.state.db_pool
    if pool is None:
        raise HTTPException(503, "Database unavailable")

    seed = (body.seed or "").strip() or str(uuid.uuid4())
    batch_count = int(body.batch_count or 1)
    labels = list(body.video_labels or [])

    async with pool.acquire() as conn:
        tz = await _user_timezone(conn, bill_id)
        extra_occ: dict = {}
        batch_items = []
        first_smart = None
        first_sm = None

        for i in range(batch_count):
            slot_seed = seed if batch_count == 1 else f"{seed}:slot-{i}"
            smart = await build_smart_schedule_for_upload(
                conn,
                bill_id,
                platforms,
                num_days=body.smart_schedule_days,
                random_seed=slot_seed,
                user_timezone=tz,
                extra_day_occupancy=extra_occ or None,
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
            label = ""
            if i < len(labels) and labels[i]:
                label = str(labels[i]).strip()
            if not label:
                label = f"Video {i + 1}"
            batch_items.append(
                {
                    "index": i,
                    "label": label,
                    "seed": slot_seed,
                    "smart_schedule": sm,
                    "schedule": sm,
                }
            )
            for offset, count in occupancy_from_schedule(smart, now=_now_utc()).items():
                extra_occ[offset] = extra_occ.get(offset, 0) + count
            if first_smart is None:
                first_smart = smart
                first_sm = sm

    assert first_smart is not None and first_sm is not None
    return preview_response_payload(
        first_smart,
        first_sm,
        seed=seed,
        smart_schedule_days=body.smart_schedule_days,
        user_timezone=tz,
        batch=batch_items,
    )
