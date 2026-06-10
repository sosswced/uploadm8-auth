"""
Shared schedule slot helpers — single source for calendar, stats, and blocked days.

All datetimes are UTC-aware. Smart uploads expose one slot per platform via
``schedule_metadata``; regular scheduled uploads use ``scheduled_time`` only.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from services.deferred_publish_schedule import parse_iso_datetime, parse_schedule_metadata


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def all_slot_datetimes(row: dict[str, Any]) -> list[datetime]:
    """Every publish slot for an upload row."""
    mode = str(row.get("schedule_mode") or "scheduled").strip().lower()
    out: list[datetime] = []

    if mode == "smart":
        for dt in parse_schedule_metadata(row.get("schedule_metadata")).values():
            out.append(_aware(dt))

    st = row.get("scheduled_time")
    if st is not None:
        parsed = parse_iso_datetime(st)
        if parsed is not None:
            aware_st = _aware(parsed)
            if not out or aware_st not in out:
                out.append(aware_st)

    return sorted(set(out))


def slot_dates(row: dict[str, Any]) -> set[date]:
    return {dt.date() for dt in all_slot_datetimes(row)}


def day_offsets_from_today(
    row: dict[str, Any],
    *,
    today: date,
    num_days: int,
) -> set[int]:
    """1-based day offsets from ``today`` occupied by this upload."""
    used: set[int] = set()
    for d in slot_dates(row):
        diff = (d - today).days
        if 0 < diff <= num_days:
            used.add(diff)
    return used


def slots_by_date(row: dict[str, Any]) -> dict[date, list[tuple[str, datetime]]]:
    """date -> [(platform, datetime), ...] for smart; single anonymous slot otherwise."""
    mode = str(row.get("schedule_mode") or "scheduled").strip().lower()
    by_date: dict[date, list[tuple[str, datetime]]] = {}

    if mode == "smart":
        for plat, dt in parse_schedule_metadata(row.get("schedule_metadata")).items():
            d = _aware(dt).date()
            by_date.setdefault(d, []).append((plat, _aware(dt)))
    else:
        for dt in all_slot_datetimes(row):
            by_date.setdefault(dt.date(), []).append(("", _aware(dt)))

    return by_date


def compute_scheduled_stats(
    rows: Iterable[dict[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> dict[str, int]:
    """
    pending = number of uploads; today/week = uploads with any slot in window.
    """
    now = _aware(now or datetime.now(timezone.utc))
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    week_end = now + timedelta(days=7)

    pending = 0
    today = 0
    week = 0

    for row in rows:
        pending += 1
        slots = all_slot_datetimes(row)
        if not slots:
            st = parse_iso_datetime(row.get("scheduled_time"))
            if st is not None:
                slots = [_aware(st)]

        in_today = any(today_start <= _aware(s) < today_end for s in slots)
        in_week = any(now <= _aware(s) < week_end for s in slots)

        if in_today:
            today += 1
        if in_week:
            week += 1

    return {"pending": pending, "today": today, "week": week}


def rows_for_blocked_days(
    rows: Iterable[dict[str, Any]],
    *,
    today: date,
    num_days: int,
) -> set[int]:
    """Union of day offsets occupied by rows."""
    used: set[int] = set()
    for row in rows:
        used |= day_offsets_from_today(row, today=today, num_days=num_days)
    return used
