"""
Funnel conversion metrics from durable upload_funnel_events.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional


async def funnel_conversion_summary(
    pool,
    *,
    lookback_days: int = 30,
) -> Dict[str, Any]:
    """presign_ok → r2_complete → worker_started → terminal success rate."""
    since = datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT event, COUNT(DISTINCT upload_id)::bigint AS uploads
            FROM upload_funnel_events
            WHERE ts >= $1
            GROUP BY event
            """,
            since,
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
            WHERE ts >= $1
            """,
            since,
        )
    counts = {str(r["event"]): int(r["uploads"] or 0) for r in rows}
    presign = counts.get("presign_ok", 0)
    complete = counts.get("r2_complete", 0)
    started = counts.get("worker_started", 0)
    term_ok = int((terminals or {}).get("terminal_ok") or 0)
    term_any = int((terminals or {}).get("terminal_any") or 0)
    return {
        "lookback_days": lookback_days,
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
