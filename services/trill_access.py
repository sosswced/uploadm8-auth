"""Trill access helpers (map unlock predicate shared by /api/me and Trill routes)."""

from __future__ import annotations

from typing import Any

TRILL_COMPLETED_STATUSES = "ARRAY['completed','succeeded','partial']::varchar[]"

# Any finished upload with a persisted Trill score (matches queue/analytics "Trill" chip).
TRILL_SCORED_PREDICATE = f"""
    u.trill_score IS NOT NULL
    AND u.status = ANY({TRILL_COMPLETED_STATUSES})
"""


def trill_route_evidence_sql(alias: str = "u") -> str:
    """
    True when ``trill_metadata`` shows map/GPS evidence (worker nested telemetry or legacy flat keys).
    """
    t = f"{alias}.trill_metadata"
    return f"""(
        COALESCE(jsonb_array_length(COALESCE({t}->'telemetry'->'points', '[]'::jsonb)), 0) >= 3
        OR (
            NULLIF(btrim({t}#>>'{{telemetry,mid_lat}}'), '') IS NOT NULL
            AND NULLIF(btrim({t}#>>'{{telemetry,mid_lon}}'), '') IS NOT NULL
        )
        OR (
            NULLIF(btrim({t}#>>'{{telemetry,start_lat}}'), '') IS NOT NULL
            AND NULLIF(btrim({t}#>>'{{telemetry,start_lon}}'), '') IS NOT NULL
        )
        OR (
            NULLIF(btrim({t}#>>'{{place_lat}}'), '') IS NOT NULL
            AND NULLIF(btrim({t}#>>'{{place_lon}}'), '') IS NOT NULL
        )
        OR (
            NULLIF(btrim({t}#>>'{{start_lat}}'), '') IS NOT NULL
            AND NULLIF(btrim({t}#>>'{{start_lon}}'), '') IS NOT NULL
        )
    )"""


def trill_analysis_evidence_sql(alias: str = "u") -> str:
    """
    Broader than GPS route evidence: nested worker payload, legacy flat keys, or speed telemetry.
    Used with TRILL_SCORED_PREDICATE when geo is optional (leaderboard rollups, map unlock).
    """
    t = f"{alias}.trill_metadata"
    route = trill_route_evidence_sql(alias)
    return f"""(
        {route}
        OR ({t} ? 'trill')
        OR NULLIF(btrim({t}#>>'{{trill,score}}'), '') IS NOT NULL
        OR NULLIF(btrim({t}#>>'{{trill_score}}'), '') IS NOT NULL
        OR COALESCE(jsonb_array_length(COALESCE({t}->'telemetry'->'points', '[]'::jsonb)), 0) >= 1
        OR NULLIF(btrim({t}#>>'{{telemetry,max_speed_mph}}'), '')::double precision > 0
        OR NULLIF(btrim({t}#>>'{{max_speed_mph}}'), '')::double precision > 0
    )"""


TRILL_ROUTE_PREDICATE = f"""
    {TRILL_SCORED_PREDICATE.strip()}
    AND {trill_route_evidence_sql("u")}
"""


def trill_map_lat_sql(alias: str = "u") -> str:
    t = f"{alias}.trill_metadata"
    return f"""COALESCE(
        NULLIF(btrim({t}#>>'{{telemetry,mid_lat}}'), '')::double precision,
        NULLIF(btrim({t}#>>'{{telemetry,start_lat}}'), '')::double precision,
        NULLIF(btrim({t}#>>'{{place_lat}}'), '')::double precision,
        NULLIF(btrim({t}#>>'{{start_lat}}'), '')::double precision
    )"""


def trill_map_lon_sql(alias: str = "u") -> str:
    t = f"{alias}.trill_metadata"
    return f"""COALESCE(
        NULLIF(btrim({t}#>>'{{telemetry,mid_lon}}'), '')::double precision,
        NULLIF(btrim({t}#>>'{{telemetry,start_lon}}'), '')::double precision,
        NULLIF(btrim({t}#>>'{{place_lon}}'), '')::double precision,
        NULLIF(btrim({t}#>>'{{start_lon}}'), '')::double precision
    )"""


async def user_trill_map_unlocked(conn: Any, user_id: str) -> bool:
    """True when the user has at least one completed upload with a Trill score."""
    return bool(
        await conn.fetchval(
            f"""
            SELECT EXISTS (
                SELECT 1 FROM uploads u
                WHERE u.user_id = $1
                AND {TRILL_SCORED_PREDICATE.strip()}
            )
            """,
            user_id,
        )
    )


async def backfill_trill_metadata_evidence(conn: Any) -> str:
    """
    Copy legacy flat ``trill_metadata`` keys into nested ``telemetry.*`` so geo queries
    and route evidence match uploads that already have ``trill_score``.
    Safe to run repeatedly (idempotent merge).
    """
    return await conn.execute(
        """
        UPDATE uploads u
        SET
            trill_metadata = COALESCE(u.trill_metadata, '{}'::jsonb) || jsonb_build_object(
                'telemetry',
                COALESCE(u.trill_metadata->'telemetry', '{}'::jsonb) || jsonb_strip_nulls(jsonb_build_object(
                    'mid_lat', COALESCE(
                        NULLIF(btrim(u.trill_metadata#>>'{telemetry,mid_lat}'), ''),
                        NULLIF(u.trill_metadata->>'place_lat', ''),
                        NULLIF(u.trill_metadata->>'start_lat', '')
                    ),
                    'mid_lon', COALESCE(
                        NULLIF(btrim(u.trill_metadata#>>'{telemetry,mid_lon}'), ''),
                        NULLIF(u.trill_metadata->>'place_lon', ''),
                        NULLIF(u.trill_metadata->>'start_lon', '')
                    ),
                    'start_lat', COALESCE(
                        NULLIF(btrim(u.trill_metadata#>>'{telemetry,start_lat}'), ''),
                        NULLIF(u.trill_metadata->>'start_lat', ''),
                        NULLIF(u.trill_metadata->>'place_lat', '')
                    ),
                    'start_lon', COALESCE(
                        NULLIF(btrim(u.trill_metadata#>>'{telemetry,start_lon}'), ''),
                        NULLIF(u.trill_metadata->>'start_lon', ''),
                        NULLIF(u.trill_metadata->>'place_lon', '')
                    ),
                    'max_speed_mph', COALESCE(
                        NULLIF(btrim(u.trill_metadata#>>'{telemetry,max_speed_mph}'), ''),
                        NULLIF(u.trill_metadata->>'max_speed_mph', '')
                    ),
                    'total_distance_miles', COALESCE(
                        NULLIF(btrim(u.trill_metadata#>>'{telemetry,total_distance_miles}'), ''),
                        NULLIF(u.trill_metadata->>'distance_miles', '')
                    )
                ))
            ),
            updated_at = NOW()
        WHERE u.trill_score IS NOT NULL
          AND u.status = ANY(ARRAY['completed','succeeded','partial']::varchar[])
          AND u.trill_metadata IS NOT NULL
          AND (
              u.trill_metadata ? 'place_lat'
              OR u.trill_metadata ? 'place_lon'
              OR u.trill_metadata ? 'start_lat'
              OR u.trill_metadata ? 'start_lon'
              OR u.trill_metadata ? 'max_speed_mph'
              OR u.trill_metadata ? 'distance_miles'
              OR u.trill_metadata ? 'trill_score'
              OR (u.trill_metadata ? 'trill' AND NOT (u.trill_metadata ? 'telemetry'))
          )
        """
    )
