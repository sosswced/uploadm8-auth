"""
Funnel conversion metrics from durable upload_funnel_events.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional


async def funnel_conversion_summary(
    pool,
    *,
    lookback_days: Optional[int] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> Dict[str, Any]:
    """presign_ok → r2_complete → worker_started → terminal success rate.

    Prefer absolute ``since``/``until`` (half-open). ``lookback_days`` remains for
    callers that only pass a day count (trailing from now).
    """
    if since is not None and until is not None:
        win_since = since
        win_until = until
    else:
        days = max(1, int(lookback_days or 30))
        win_until = datetime.now(timezone.utc)
        win_since = win_until - timedelta(days=days)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT event, COUNT(DISTINCT upload_id)::bigint AS uploads
            FROM upload_funnel_events
            WHERE ts >= $1 AND ts < $2
            GROUP BY event
            """,
            win_since,
            win_until,
        )
        terminals = await conn.fetchrow(
            """
            SELECT
              COUNT(DISTINCT upload_id) FILTER (
                WHERE event LIKE 'terminal_%'
                  AND event NOT IN ('terminal_failed', 'terminal_cancelled')
              )::bigint AS terminal_ok,
              COUNT(DISTINCT upload_id) FILTER (
                WHERE event LIKE 'terminal_%'
              )::bigint AS terminal_any
            FROM upload_funnel_events
            WHERE ts >= $1 AND ts < $2
            """,
            win_since,
            win_until,
        )
    counts = {str(r["event"]): int(r["uploads"] or 0) for r in rows}
    presign = counts.get("presign_ok", 0)
    complete = counts.get("r2_complete", 0)
    started = counts.get("worker_started", 0)
    term_ok = int((terminals or {}).get("terminal_ok") or 0)
    term_any = int((terminals or {}).get("terminal_any") or 0)
    span_days = max(1, int(round((win_until - win_since).total_seconds() / 86400.0)))
    return {
        "lookback_days": span_days,
        "window_start_utc": win_since.isoformat(),
        "window_end_exclusive_utc": win_until.isoformat(),
        "presign_ok": presign,
        "r2_complete": complete,
        "worker_started": started,
        "terminal_success": term_ok,
        "terminal_any": term_any,
        "complete_rate_pct": round(100.0 * complete / max(presign, 1), 2),
        "worker_start_rate_pct": round(100.0 * started / max(complete, 1), 2),
        "success_rate_pct": round(100.0 * term_ok / max(started, 1), 2),
        "events": counts,
    }
