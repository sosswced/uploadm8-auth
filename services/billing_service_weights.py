"""
DB-backed per-service AIC weights (billing_service_weights table).

Seeded from stages.ai_service_costs.SERVICE_WEIGHTS; admin upserts tune live pricing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from stages.ai_service_costs import SERVICE_WEIGHTS

logger = logging.getLogger(__name__)

_MAX_WEIGHT = 5000


async def fetch_service_weights_map(conn: Any) -> Dict[str, int]:
    """Return all rows as service_id -> aic_weight (may be empty before first seed)."""
    try:
        rows = await conn.fetch("SELECT service_id, aic_weight FROM billing_service_weights")
    except Exception as e:
        logger.warning("billing_service_weights fetch failed (table missing?): %s", e)
        return {}
    out: Dict[str, int] = {}
    for r in rows or []:
        sid = str(r["service_id"] or "").strip()
        if not sid:
            continue
        try:
            out[sid] = max(0, min(_MAX_WEIGHT, int(r["aic_weight"] or 0)))
        except (TypeError, ValueError):
            continue
    return out


async def ensure_billing_weights_seeded(conn: Any) -> None:
    """Insert missing service_ids with code defaults (never overwrites ops tuning)."""
    for sid, w in SERVICE_WEIGHTS.items():
        try:
            await conn.execute(
                """
                INSERT INTO billing_service_weights (service_id, aic_weight)
                VALUES ($1, $2)
                ON CONFLICT (service_id) DO NOTHING
                """,
                sid,
                max(0, min(_MAX_WEIGHT, int(w))),
            )
        except Exception as e:
            logger.warning("ensure_billing_weights_seeded skip %s: %s", sid, e)


async def upsert_service_weights(
    conn: Any,
    weights: Dict[str, int],
    *,
    updated_by: Optional[str] = None,
) -> int:
    """
    Upsert validated weights. Returns number of rows written.
    Only keys present in SERVICE_WEIGHTS are accepted.
    """
    n = 0
    for sid, raw in weights.items():
        sid = str(sid or "").strip()
        if sid not in SERVICE_WEIGHTS:
            continue
        try:
            w = max(0, min(_MAX_WEIGHT, int(raw)))
        except (TypeError, ValueError):
            continue
        await conn.execute(
            """
            INSERT INTO billing_service_weights (service_id, aic_weight, updated_at, updated_by)
            VALUES ($1, $2, NOW(), $3)
            ON CONFLICT (service_id) DO UPDATE SET
                aic_weight = EXCLUDED.aic_weight,
                updated_at = NOW(),
                updated_by = COALESCE(EXCLUDED.updated_by, billing_service_weights.updated_by)
            """,
            sid,
            w,
            updated_by,
        )
        n += 1
    return n
