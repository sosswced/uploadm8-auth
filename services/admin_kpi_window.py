"""
Shared admin KPI time-window resolver.

Half-open UTC ``[since, until)`` — same contract as catalog aggregate and
analytics CRM custom ranges.

Presets / ``Nd`` → trailing window ending at *now* (or *now* override in tests).
Absolute ``start`` + ``end`` ISO → explicit calendar window.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from services.platform_metrics_ui import parse_iso_ts

# Canonical admin family map (180d for 6m — not calendar months).
RANGE_MINUTES: Dict[str, int] = {
    "24h": 24 * 60,
    "7d": 7 * 24 * 60,
    "30d": 30 * 24 * 60,
    "90d": 90 * 24 * 60,
    "6m": 180 * 24 * 60,
    "1y": 365 * 24 * 60,
    "365d": 365 * 24 * 60,
}

_MAX_SPAN_DAYS = 3650
_DEFAULT_RANGE = "30d"


class AdminKpiWindowError(ValueError):
    """Invalid range / start / end for admin KPI windows."""


def range_key_to_minutes(range_key: str | None, *, strict: bool = False) -> int:
    """
    Convert a trailing range key to minutes.

    When ``strict`` is True, unrecognized non-empty keys raise
    ``AdminKpiWindowError``. Empty/None → 30d.
    """
    r = (range_key or "").strip()
    if not r:
        return RANGE_MINUTES[_DEFAULT_RANGE]
    if r.lower() == "all":
        # Callers that need unbounded should use resolve_admin_kpi_window.
        from services.canonical_engagement import ALL_TIME_FLOOR_UTC

        now = datetime.now(timezone.utc)
        delta = now - ALL_TIME_FLOOR_UTC
        return max(1, int(delta.total_seconds() // 60))
    if r in RANGE_MINUTES:
        return RANGE_MINUTES[r]
    m = re.fullmatch(r"(\d{1,4})d", r, flags=re.IGNORECASE)
    if m:
        days = max(1, min(int(m.group(1)), _MAX_SPAN_DAYS))
        return days * 24 * 60
    if strict:
        raise AdminKpiWindowError(
            f"Invalid range '{range_key}'. Use 24h|7d|30d|90d|6m|1y|Nd or start+end."
        )
    return RANGE_MINUTES[_DEFAULT_RANGE]


def trailing_window(
    range_key: str | None,
    *,
    now: Optional[datetime] = None,
    strict: bool = False,
) -> Tuple[datetime, datetime]:
    until = now or datetime.now(timezone.utc)
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    else:
        until = until.astimezone(timezone.utc)
    rk = (range_key or "").strip()
    if rk.lower() == "all":
        from services.canonical_engagement import sql_since_for_analytics_range

        return sql_since_for_analytics_range("all", now=until), until
    mins = range_key_to_minutes(range_key, strict=strict)
    return until - timedelta(minutes=mins), until


def resolve_admin_kpi_window(
    range_key: str | None = None,
    start: str | None = None,
    end: str | None = None,
    *,
    now: Optional[datetime] = None,
    strict_range: bool = True,
) -> Tuple[datetime, datetime, Dict[str, Any]]:
    """
    Resolve admin KPI query params to ``[since, until)`` plus metadata.

    Absolute wins when both ``start`` and ``end`` parse.
    Otherwise trailing preset / ``Nd``.
    """
    ws = parse_iso_ts(start) if start else None
    we = parse_iso_ts(end) if end else None

    if start or end:
        if ws is None or we is None:
            raise AdminKpiWindowError(
                "Both start and end ISO-8601 timestamps are required for a custom window."
            )
        if we <= ws:
            raise AdminKpiWindowError("end must be after start (half-open [start, end)).")
        span_days = (we - ws).total_seconds() / 86400.0
        if span_days > _MAX_SPAN_DAYS:
            raise AdminKpiWindowError(f"Window exceeds {_MAX_SPAN_DAYS} days.")
        meta = {
            "mode": "explicit_utc",
            "range_key": None,
            "window_start_utc": ws.isoformat(),
            "window_end_exclusive_utc": we.isoformat(),
        }
        return ws, we, meta

    rk = (range_key or "").strip() or _DEFAULT_RANGE
    since, until = trailing_window(rk, now=now, strict=strict_range)
    meta = {
        "mode": "trailing",
        "range_key": rk if rk.lower() != "all" else "all",
        "window_start_utc": since.isoformat(),
        "window_end_exclusive_utc": until.isoformat(),
    }
    return since, until, meta


def previous_equal_window(since: datetime, until: datetime) -> Tuple[datetime, datetime]:
    """Equal-duration window immediately before ``since``."""
    delta = until - since
    return since - delta, since


def window_cache_key(since: datetime, until: datetime, prefix: str = "") -> str:
    return f"{prefix}:{since.isoformat()}:{until.isoformat()}"
