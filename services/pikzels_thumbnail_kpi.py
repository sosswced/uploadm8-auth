"""
Admin KPI: Pikzels API key configured vs thumbnail_render_method=template on uploads.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

import asyncpg

from services.pikzels_v2 import resolve_public_api_key


async def fetch_pikzels_template_render_kpi(
    conn: asyncpg.Connection,
    since: datetime,
    until: datetime,
) -> Dict[str, Any]:
    """
    Returns counts of completed uploads where the server has a Pikzels key but
    the worker recorded ``thumbnail_render_method=template`` (PIL fallback).
    """
    key_configured = bool((resolve_public_api_key() or "").strip())
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*)::int AS total_completed,
            COUNT(*) FILTER (
                WHERE COALESCE(NULLIF(u.output_artifacts->>'thumbnail_render_method', ''), '') = 'template'
            )::int AS template_render_count,
            COUNT(*) FILTER (
                WHERE COALESCE(NULLIF(u.output_artifacts->>'thumbnail_render_method', ''), '') = 'studio_renderer'
            )::int AS studio_render_count,
            COUNT(*) FILTER (
                WHERE COALESCE(NULLIF(u.output_artifacts->>'thumbnail_render_method', ''), '') IN ('', 'none')
            )::int AS raw_frame_only_count
        FROM uploads u
        WHERE u.created_at >= $1 AND u.created_at < $2
          AND u.status IN ('completed', 'succeeded', 'partial')
        """,
        since,
        until,
    )
    total = int((row["total_completed"] if row else 0) or 0)
    template_n = int((row["template_render_count"] if row else 0) or 0)
    studio_n = int((row["studio_render_count"] if row else 0) or 0)
    raw_n = int((row["raw_frame_only_count"] if row else 0) or 0)
    pct = round((template_n / total) * 100.0, 2) if total else 0.0
    return {
        "pikzels_api_key_configured": key_configured,
        "total_completed_uploads": total,
        "template_render_count": template_n,
        "studio_render_count": studio_n,
        "raw_frame_only_count": raw_n,
        "template_render_pct": pct,
        "alert_threshold_pct": float(os.environ.get("PIKZELS_TEMPLATE_KPI_ALERT_PCT", "15") or 15),
        "should_alert": key_configured
        and total >= 5
        and pct >= float(os.environ.get("PIKZELS_TEMPLATE_KPI_ALERT_PCT", "15") or 15),
    }
