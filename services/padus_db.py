"""PAD-US lookups against PostGIS in Neon (no local GDB / /tmp extraction).

Expect a table created from a one-time PADUS load, e.g. ``padus_protected_areas``
with a ``geometry`` column (SRID 4326) and the attribute columns used below.
Override with ``PADUS_DB_TABLE`` / ``PADUS_DB_GEOM_COL`` (alphanumeric + underscore only).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, MutableMapping, Optional

import asyncpg

logger = logging.getLogger("uploadm8-api.padus_db")

_IDENT = re.compile(r"^[A-Za-z0-9_]+$")


def _safe_ident(raw: str, default: str) -> str:
    s = (raw or "").strip()
    return s if _IDENT.fullmatch(s) else default


def padus_table() -> str:
    return _safe_ident(os.environ.get("PADUS_DB_TABLE", ""), "padus_protected_areas")


def padus_geom_column() -> str:
    return _safe_ident(os.environ.get("PADUS_DB_GEOM_COL", ""), "geometry")


async def is_protected_land(
    conn: asyncpg.Connection, lat: float, lon: float
) -> Optional[Dict[str, Any]]:
    """Return one PAD-US row if ``(lon, lat)`` lies inside stored geometry, else ``None``."""
    table = padus_table()
    geom = padus_geom_column()
    sql = f"""
        SELECT "Unit_Nm", "Mang_Type", "Mang_Nam", "GAP_Sts", "State_Nm"
        FROM {table}
        WHERE ST_Contains(
            "{geom}"::geometry,
            ST_SetSRID(ST_MakePoint($1, $2), 4326)
        )
        LIMIT 1
    """
    try:
        row = await conn.fetchrow(sql, lon, lat)
    except Exception as e:
        logger.debug("PADUS DB lookup failed: %s", e)
        return None
    return dict(row) if row else None


async def padus_hit_dict_from_db(conn: asyncpg.Connection, lat: float, lon: float) -> Dict[str, Any]:
    """Shape compatible with ``telemetry_trill.enrich_route_padus_gazetteer`` PAD-US keys."""
    row = await is_protected_land(conn, lat, lon)
    if not row:
        return {"near_padus": False, "padus_unit_name": None, "padus_layer": None}
    raw = row.get("Unit_Nm")
    name = str(raw).strip() if raw is not None else ""
    return {
        "near_padus": True,
        "padus_unit_name": name or None,
        "padus_layer": padus_table(),
    }


def merge_padus_enrichment_into_mapping(target: MutableMapping[str, Any], extra: Dict[str, Any]) -> None:
    """Merge PAD-US dict into Trill API / JSON metadata (``near_protected``, ``protected_name``)."""
    if not extra:
        return
    target.update(extra)
    target["near_protected"] = bool(extra.get("near_padus"))
    pn = extra.get("padus_unit_name")
    target["protected_name"] = str(pn).strip() if pn else None
