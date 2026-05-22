"""Nearest US place name lookup in Postgres / PostGIS (optional file-less gazetteer).

Mirrors ``services.padus_db`` style: configurable table/columns via env, safe
identifiers only, graceful no-op when disabled or schema missing.

Env:
  GAZETTEER_DB_ENABLED     — default on; set 0/false/no/off to skip
  GAZETTEER_DB_TABLE       — default ``gazetteer_places``
  GAZETTEER_DB_GEOM_COL    — default ``geom`` (Point, SRID 4326)
  GAZETTEER_DB_NAME_COL    — default ``name``
  GAZETTEER_DB_STATE_COL   — default ``state_usps``
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


async def nearest_gazetteer_place_from_db(
    conn: asyncpg.Connection, lat: float, lon: float
) -> Dict[str, Any]:
    """Return gazetteer keys compatible with telemetry enrichment."""
    if not gazetteer_db_enabled():
        return {}
    table = gazetteer_table()
    geom = gazetteer_geom_column()
    name_col = gazetteer_name_column()
    state_col = gazetteer_state_column()
    sql = f"""
        SELECT
          {name_col} AS pname,
          {state_col} AS st,
          ST_Distance(
              {geom}::geography,
              ST_SetSRID(ST_MakePoint($1::float8, $2::float8), 4326)::geography
          ) AS dist_m
        FROM {table}
        WHERE {geom} IS NOT NULL
        ORDER BY {geom}::geography <-> ST_SetSRID(ST_MakePoint($1::float8, $2::float8), 4326)::geography
        LIMIT 1
    """
    try:
        row = await conn.fetchrow(sql, float(lon), float(lat))
    except Exception as e:
        logger.debug("Gazetteer DB lookup failed: %s", e)
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
