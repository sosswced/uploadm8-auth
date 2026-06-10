"""Shared smart-schedule preview response builder."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


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
                f"Data-informed slot for {plat.title()} "
                f"(engagement signals + momentum, shown in {user_timezone})"
            ),
        }
    return out


def preview_response_payload(
    smart: Dict[str, datetime],
    sm: Dict[str, str],
    *,
    seed: str,
    smart_schedule_days: int,
    user_timezone: str = "UTC",
) -> Dict[str, Any]:
    """Canonical preview JSON for /api/scheduling/preview and legacy shim."""
    explanation = smart_schedule_explanation(smart, user_timezone=user_timezone)
    scheduled_min = min(sm.values()) if sm else None
    return {
        "smart_schedule": sm,
        "schedule": sm,
        "scheduled_time": scheduled_min,
        "seed": seed,
        "smart_schedule_days": smart_schedule_days,
        "explanation": explanation,
    }
