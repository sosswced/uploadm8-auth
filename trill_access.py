"""Trill access helpers (map unlock predicate shared by /api/me and Trill routes)."""

from __future__ import annotations

from typing import Any

TRILL_ROUTE_PREDICATE = """
    u.trill_score IS NOT NULL
    AND u.status = ANY(ARRAY['completed','succeeded','partial']::varchar[])
    AND (
        COALESCE(jsonb_array_length(COALESCE(u.trill_metadata->'telemetry'->'points', '[]'::jsonb)), 0) >= 3
        OR (
            NULLIF(btrim(u.trill_metadata#>>'{telemetry,mid_lat}'), '') IS NOT NULL
            AND NULLIF(btrim(u.trill_metadata#>>'{telemetry,mid_lon}'), '') IS NOT NULL
        )
    )
"""


async def user_trill_map_unlocked(conn: Any, user_id: str) -> bool:
    """True when the user has at least one completed upload with Trill route evidence."""
    return bool(
        await conn.fetchval(
            f"""
            SELECT EXISTS (
                SELECT 1 FROM uploads u
                WHERE u.user_id = $1
                AND {TRILL_ROUTE_PREDICATE.strip()}
            )
            """,
            user_id,
        )
    )
