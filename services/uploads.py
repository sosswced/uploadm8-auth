"""
Legacy shim: smart-schedule helpers live in ``core.scheduling`` (static priors) and
``services.smart_schedule_insights`` (DB-backed signals). Tools may import from here.
"""

from __future__ import annotations

from core.scheduling import calculate_smart_schedule, get_existing_scheduled_days

__all__ = ("calculate_smart_schedule", "get_existing_scheduled_days")
