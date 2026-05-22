"""NHTSA vPIC-backed vehicle make/model catalog (cached in Postgres)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

NHTSA_BASE = "https://vpic.nhtsa.dot.gov/api"

# Passenger-relevant NHTSA vehicle types (excludes trailers, buses, motorcycles, etc.).
CONSUMER_VEHICLE_TYPES: Tuple[str, ...] = ("car", "mpv", "truck")

# Shown first when the make dropdown loads without a search query.
POPULAR_CONSUMER_MAKES: Tuple[str, ...] = (
    "TOYOTA",
    "FORD",
    "CHEVROLET",
    "HONDA",
    "NISSAN",
    "JEEP",
    "RAM",
    "GMC",
    "DODGE",
    "HYUNDAI",
    "KIA",
    "SUBARU",
    "MAZDA",
    "BMW",
    "MERCEDES-BENZ",
    "AUDI",
    "VOLKSWAGEN",
    "LEXUS",
    "ACURA",
    "INFINITI",
    "CADILLAC",
    "BUICK",
    "LINCOLN",
    "VOLVO",
    "TESLA",
    "GENESIS",
    "PORSCHE",
    "CHRYSLER",
    "MITSUBISHI",
    "MINI",
    "RIVIAN",
    "LUCID",
    "POLESTAR",
)


def _parse_nhtsa_make_row(row: dict) -> Optional[Tuple[int, str]]:
    mid = row.get("Make_ID") or row.get("MakeId")
    name = (row.get("Make_Name") or row.get("MakeName") or "").strip()
    if mid is None or not name:
        return None
    try:
        return int(mid), name[:200]
    except (TypeError, ValueError):
        return None


async def _fetch_makes_for_vehicle_type(client: httpx.AsyncClient, vehicle_type: str) -> List[dict]:
    url = f"{NHTSA_BASE}/vehicles/GetMakesForVehicleType/{vehicle_type}?format=json"
    r = await client.get(url)
    r.raise_for_status()
    data = r.json()
    return list(data.get("Results") or [])


async def sync_consumer_makes(conn: Any) -> int:
    """Fetch car/MPV/truck makes from NHTSA and mark them as consumer_vehicle."""
    merged: Dict[int, str] = {}
    async with httpx.AsyncClient(timeout=90.0) as client:
        for vtype in CONSUMER_VEHICLE_TYPES:
            try:
                rows = await _fetch_makes_for_vehicle_type(client, vtype)
            except Exception as e:
                logger.warning("NHTSA GetMakesForVehicleType(%s) failed: %s", vtype, e)
                continue
            for row in rows:
                parsed = _parse_nhtsa_make_row(row)
                if parsed:
                    merged[parsed[0]] = parsed[1]

    n = 0
    for nhtsa_id, name in merged.items():
        await conn.execute(
            """
            INSERT INTO vehicle_makes (nhtsa_make_id, name, consumer_vehicle)
            VALUES ($1, $2, TRUE)
            ON CONFLICT (nhtsa_make_id) DO UPDATE
            SET name = EXCLUDED.name, consumer_vehicle = TRUE
            """,
            nhtsa_id,
            name,
        )
        n += 1
    return n


async def sync_all_makes(conn: Any) -> int:
    """Backward-compatible alias: sync consumer passenger/light-truck makes only."""
    return await sync_consumer_makes(conn)


async def sync_models_for_make_nhtsa_id(conn: Any, nhtsa_make_id: int) -> int:
    """Fetch models for one NHTSA make id; requires vehicle_makes row with that nhtsa_make_id."""
    row = await conn.fetchrow(
        "SELECT id FROM vehicle_makes WHERE nhtsa_make_id = $1",
        nhtsa_make_id,
    )
    if not row:
        return 0
    make_db_id = int(row["id"])
    url = f"{NHTSA_BASE}/vehicles/GetModelsForMakeId/{nhtsa_make_id}?format=json"
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    results = data.get("Results") or []
    n = 0
    for m in results:
        mname = (m.get("Model_Name") or m.get("ModelName") or "").strip()
        if not mname:
            continue
        raw_mid = m.get("Model_ID") or m.get("ModelId")
        nhtsa_model_id: Optional[int] = None
        if raw_mid is not None:
            try:
                nhtsa_model_id = int(raw_mid)
            except (TypeError, ValueError):
                nhtsa_model_id = None
        if nhtsa_model_id is not None:
            await conn.execute(
                """
                INSERT INTO vehicle_models (make_id, nhtsa_model_id, name)
                VALUES ($1, $2, $3)
                ON CONFLICT (make_id, nhtsa_model_id) WHERE nhtsa_model_id IS NOT NULL
                DO UPDATE SET name = EXCLUDED.name
                """,
                make_db_id,
                nhtsa_model_id,
                mname[:200],
            )
        else:
            await conn.execute(
                """
                INSERT INTO vehicle_models (make_id, nhtsa_model_id, name)
                SELECT $1, NULL, $2
                WHERE NOT EXISTS (
                    SELECT 1 FROM vehicle_models
                    WHERE make_id = $1 AND nhtsa_model_id IS NULL AND name = $2
                )
                """,
                make_db_id,
                mname[:200],
            )
        n += 1
    return n


async def seed_popular_consumer_makes(conn: Any) -> int:
    """No-op placeholder — real makes come from sync_consumer_makes (positive NHTSA ids)."""
    return 0


async def ensure_makes_populated(conn: Any) -> None:
    """Ensure consumer car/truck makes exist without blocking on NHTSA during HTTP requests."""
    try:
        consumer_cnt = await conn.fetchval(
            "SELECT COUNT(*) FROM vehicle_makes WHERE consumer_vehicle = TRUE"
        )
    except Exception:
        # Column may not exist before migration; fall back to total count heuristic.
        consumer_cnt = await conn.fetchval("SELECT COUNT(*) FROM vehicle_makes")
    if (consumer_cnt or 0) < 50:
        try:
            inserted = await sync_consumer_makes(conn)
            logger.info("vehicle_makes consumer NHTSA sync: %s makes", inserted)
        except Exception as e:
            logger.warning("vehicle_makes consumer NHTSA sync failed: %s", e)


async def ensure_models_for_make(conn: Any, make_db_id: int, *, allow_nhtsa_sync: bool = True) -> None:
    """Load models from DB; optionally pull from NHTSA when this make has none cached yet."""
    row = await conn.fetchrow(
        "SELECT nhtsa_make_id FROM vehicle_makes WHERE id = $1",
        make_db_id,
    )
    if not row or row["nhtsa_make_id"] is None or int(row["nhtsa_make_id"]) < 1:
        return
    if not allow_nhtsa_sync:
        return
    n_models = await conn.fetchval(
        "SELECT COUNT(*) FROM vehicle_models WHERE make_id = $1",
        make_db_id,
    )
    if (n_models or 0) > 0:
        return
    try:
        await sync_models_for_make_nhtsa_id(conn, int(row["nhtsa_make_id"]))
    except Exception as e:
        logger.warning("vehicle_models sync failed make_id=%s: %s", make_db_id, e)


def popular_make_names_upper() -> List[str]:
    return [m.upper() for m in POPULAR_CONSUMER_MAKES]


async def fetch_makes_list(
    conn: Any,
    *,
    q: str = "",
    limit: int = 80,
    consumer_only: bool = True,
) -> List[Dict[str, Any]]:
    """Query makes for API / UI — consumer brands first when not searching."""
    await ensure_makes_populated(conn)
    qn = (q or "").strip().lower()
    popular = popular_make_names_upper()

    dedupe_sql = """
        SELECT id, name FROM (
            SELECT id, name,
                ROW_NUMBER() OVER (
                    PARTITION BY UPPER(TRIM(name))
                    ORDER BY
                        CASE WHEN COALESCE(nhtsa_make_id, 0) > 0 THEN 0 ELSE 1 END,
                        id DESC
                ) AS rn
            FROM vehicle_makes
            WHERE consumer_vehicle = TRUE
    """
    if consumer_only:
        if qn:
            rows = await conn.fetch(
                dedupe_sql
                + """
                AND LOWER(name) LIKE '%' || $1 || '%'
            ) ranked
            WHERE rn = 1
            ORDER BY
                CASE WHEN UPPER(name) = ANY($3::text[]) THEN 0 ELSE 1 END,
                name ASC
            LIMIT $2
            """,
                qn,
                limit,
                popular,
            )
        else:
            rows = await conn.fetch(
                dedupe_sql
                + """
            ) ranked
            WHERE rn = 1
            ORDER BY
                CASE WHEN UPPER(name) = ANY($2::text[]) THEN 0 ELSE 1 END,
                name ASC
            LIMIT $1
            """,
                limit,
                popular,
            )
    else:
        if qn:
            rows = await conn.fetch(
                """
                SELECT id, name FROM vehicle_makes
                WHERE LOWER(name) LIKE '%' || $1 || '%'
                ORDER BY name ASC
                LIMIT $2
                """,
                qn,
                limit,
            )
        else:
            rows = await conn.fetch(
                "SELECT id, name FROM vehicle_makes ORDER BY name ASC LIMIT $1",
                limit,
            )

    return [{"id": r["id"], "name": r["name"]} for r in rows]


async def fetch_vehicle_labels(
    conn: Any, vehicle_make_id: Optional[int], vehicle_model_id: Optional[int]
) -> Dict[str, Optional[str]]:
    """Resolve display names for upload/worker context."""
    out: Dict[str, Optional[str]] = {"make_name": None, "model_name": None}
    if vehicle_make_id:
        r = await conn.fetchrow("SELECT name FROM vehicle_makes WHERE id = $1", vehicle_make_id)
        if r:
            out["make_name"] = str(r["name"] or "").strip() or None
    if vehicle_model_id:
        r2 = await conn.fetchrow("SELECT name FROM vehicle_models WHERE id = $1", vehicle_model_id)
        if r2:
            out["model_name"] = str(r2["name"] or "").strip() or None
    return out
