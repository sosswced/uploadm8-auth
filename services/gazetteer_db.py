"""Nearest US place name lookup in Postgres / PostGIS (optional file-less gazetteer).

Mirrors ``services.padus_db`` style: configurable table/columns via env, safe
identifiers only, graceful no-op when disabled or schema missing.

Env:
  GAZETTEER_DB_ENABLED     — default on; set 0/false/no/off to skip
  GAZETTEER_DB_TABLE       — default ``gazetteer_places``
  GAZETTEER_DB_GEOM_COL    — default ``geom`` (Point, SRID 4326)
  GAZETTEER_DB_NAME_COL    — default ``name``
  GAZETTEER_DB_STATE_COL   — default ``state_usps`` (falls back to ``usps``)

The live Census load has no ``geom`` column. It stores ``intptlat`` / ``intptlong``
as text and the state code in ``usps``. The lookup uses those centroids when
``geom`` is absent so place hashtags still resolve.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict

import asyncpg

logger = logging.getLogger("uploadm8-api.gazetteer_db")

_IDENT = re.compile(r"^[A-Za-z0-9_]+$")


def _safe_ident(raw: str, default: str) -> str:
    s = (raw or "").strip()
    return s if _IDENT.fullmatch(s) else default


def gazetteer_db_enabled() -> bool:
    v = str(os.environ.get("GAZETTEER_DB_ENABLED", "1")).strip().lower()
    return v not in ("0", "false", "no", "off", "disabled")


def gazetteer_table() -> str:
    return _safe_ident(os.environ.get("GAZETTEER_DB_TABLE", ""), "gazetteer_places")


def gazetteer_geom_column() -> str:
    return _safe_ident(os.environ.get("GAZETTEER_DB_GEOM_COL", ""), "geom")


def gazetteer_name_column() -> str:
    return _safe_ident(os.environ.get("GAZETTEER_DB_NAME_COL", ""), "name")


def gazetteer_state_column() -> str:
    return _safe_ident(os.environ.get("GAZETTEER_DB_STATE_COL", ""), "state_usps")


def gazetteer_lookup_plan(columns: set[str] | frozenset[str]) -> str:
    """``geom`` when PostGIS points exist; ``centroid`` for Census INTPTLAT/LONG text."""
    cols = set(columns or ())
    if gazetteer_geom_column() in cols:
        return "geom"
    lat = "intptlat" if "intptlat" in cols else ("INTPTLAT" if "INTPTLAT" in cols else "")
    lon = "intptlong" if "intptlong" in cols else ("INTPTLONG" if "INTPTLONG" in cols else "")
    if lat and lon and ("name" in cols or "NAME" in cols):
        return "centroid"
    return "none"


def _state_column(columns: set[str]) -> str:
    preferred = gazetteer_state_column()
    for cand in (preferred, "usps", "USPS", "state_usps"):
        if cand in columns:
            return cand
    return ""


def _name_column(columns: set[str]) -> str:
    preferred = gazetteer_name_column()
    for cand in (preferred, "name", "NAME"):
        if cand in columns:
            return cand
    return "name"


def _centroid_lat_lon(columns: set[str]) -> tuple[str, str]:
    lat = "intptlat" if "intptlat" in columns else "INTPTLAT"
    lon = "intptlong" if "intptlong" in columns else "INTPTLONG"
    return lat, lon


def gazetteer_nearest_sql(columns: set[str] | frozenset[str]) -> str:
    """SQL for the loaded schema. Census loads use text centroids, not ``geom``."""
    cols = set(columns or ())
    table = gazetteer_table()
    name_col = _name_column(cols)
    state_col = _state_column(cols)
    state_expr = f"btrim({state_col})" if state_col else "NULL"
    plan = gazetteer_lookup_plan(cols)
    if plan == "geom":
        geom = gazetteer_geom_column()
        return f"""
            SELECT
              btrim({name_col}::text) AS pname,
              {state_expr} AS st,
              ST_Distance(
                  {geom}::geography,
                  ST_SetSRID(ST_MakePoint($1::float8, $2::float8), 4326)::geography
              ) AS dist_m
            FROM {table}
            WHERE {geom} IS NOT NULL
            ORDER BY {geom}::geography <-> ST_SetSRID(ST_MakePoint($1::float8, $2::float8), 4326)::geography
            LIMIT 1
        """
    lat_col, lon_col = _centroid_lat_lon(cols)
    point = (
        f"ST_SetSRID(ST_MakePoint(btrim({lon_col})::float8, btrim({lat_col})::float8), 4326)::geography"
    )
    target = "ST_SetSRID(ST_MakePoint($1::float8, $2::float8), 4326)::geography"
    return f"""
        SELECT
          btrim({name_col}::text) AS pname,
          {state_expr} AS st,
          ST_Distance({point}, {target}) AS dist_m
        FROM {table}
        WHERE btrim({lat_col}) ~ '^-?[0-9]'
          AND btrim({lon_col}) ~ '^-?[0-9]'
        ORDER BY {point} <-> {target}
        LIMIT 1
    """


async def _table_columns(conn: asyncpg.Connection, table: str) -> set[str]:
    rows = await conn.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = $1
        """,
        table,
    )
    return {str(r["column_name"]) for r in rows}


async def nearest_gazetteer_place_from_db(
    conn: asyncpg.Connection, lat: float, lon: float
) -> Dict[str, Any]:
    """Return gazetteer keys compatible with telemetry enrichment."""
    if not gazetteer_db_enabled():
        return {}
    table = gazetteer_table()
    try:
        columns = await _table_columns(conn, table)
    except Exception as e:
        logger.warning("Gazetteer column probe failed: %s", e)
        return {}
    if gazetteer_lookup_plan(columns) == "none":
        logger.warning(
            "Gazetteer table %s has no geom or intptlat/intptlong columns", table
        )
        return {}
    sql = gazetteer_nearest_sql(columns)
    try:
        row = await conn.fetchrow(sql, float(lon), float(lat))
    except Exception as e:
        logger.warning("Gazetteer DB lookup failed: %s", e)
        return {}
    if not row:
        return {}
    pname = str(row["pname"] or "").strip() if row["pname"] is not None else ""
    st = str(row["st"] or "").strip() if row["st"] is not None else ""
    dist_km = None
    try:
        if row["dist_m"] is not None:
            dist_km = float(row["dist_m"]) / 1000.0
    except (TypeError, ValueError):
        pass
    if not pname:
        return {}
    out: Dict[str, Any] = {
        "gazetteer_place_name": pname,
        "gazetteer_state_usps": st or None,
    }
    if dist_km is not None:
        out["gazetteer_distance_km"] = round(dist_km, 3)
    return out
