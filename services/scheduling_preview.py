"""Shared smart-schedule preview response builder."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def smart_schedule_explanation(
    smart: Dict[str, datetime],
    *,
    user_timezone: str = "UTC",
) -> Dict[str, Dict[str, str]]:
    """Human-readable per-platform slot summary."""
    out: Dict[str, Dict[str, str]] = {}
    for plat, dt in smart.items():
        out[plat] = {
            "date": dt.strftime("%A, %B %d"),
            "time": dt.strftime("%I:%M %p UTC"),
            "iso": dt.isoformat(),
            "reason": (
                f"Data-informed local slot for {plat.title()} "
                f"(research priors + engagement signals, {user_timezone})"
            ),
        }
    return out


def occupancy_from_schedule(smart: Dict[str, datetime], *, now: Optional[datetime] = None) -> Dict[int, int]:
    """Day-offset occupancy contributed by one per-platform schedule."""
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    else:
        ref = ref.astimezone(timezone.utc)
    today = ref.date()
    occ: Dict[int, int] = {}
    for dt in smart.values():
        aware = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        offset = (aware.date() - today).days
        if offset >= 1:
            occ[offset] = occ.get(offset, 0) + 1
    return occ


def preview_response_payload(
    smart: Dict[str, datetime],
    sm: Dict[str, str],
    *,
    seed: str,
    smart_schedule_days: int,
    user_timezone: str = "UTC",
    batch: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Canonical preview JSON for /api/scheduling/preview and legacy shim."""
    explanation = smart_schedule_explanation(smart, user_timezone=user_timezone)
    scheduled_min = min(sm.values()) if sm else None
    payload: Dict[str, Any] = {
        "smart_schedule": sm,
        "schedule": sm,
        "scheduled_time": scheduled_min,
        "seed": seed,
        "smart_schedule_days": smart_schedule_days,
        "explanation": explanation,
    }
    if batch is not None:
        payload["batch"] = batch
        payload["batch_count"] = len(batch)
    return payload
