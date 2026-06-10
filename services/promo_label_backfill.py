"""
Backfill ml_outcome_labels from marketing_touchpoint_deliveries + revenue_tracking.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger("uploadm8.promo_label_backfill")


async def backfill_promo_outcome_labels(conn, *, lookback_days: int = 730) -> Dict[str, Any]:
    """Insert missing conversion labels for promo training (idempotent)."""
    rows = await conn.fetch(
        """
        WITH tp AS (
            SELECT
                mtd.id AS touchpoint_id,
                mtd.user_id,
                mtd.channel,
                mtd.created_at AS touchpoint_at,
                COALESCE(mtd.sent_at, mtd.created_at) AS effective_sent_at
            FROM marketing_touchpoint_deliveries mtd
            WHERE mtd.created_at >= NOW() - ($1::int * INTERVAL '1 day')
              AND mtd.status = 'sent'
        )
        SELECT
            tp.user_id,
            tp.touchpoint_id,
            tp.channel,
            tp.effective_sent_at,
            CASE WHEN EXISTS (
                SELECT 1 FROM revenue_tracking rt
                WHERE rt.user_id = tp.user_id
                  AND rt.created_at >= tp.effective_sent_at
                  AND rt.created_at < tp.effective_sent_at + INTERVAL '7 days'
                  AND COALESCE(rt.amount, 0) > 0
            ) OR EXISTS (
                SELECT 1 FROM marketing_events me
                WHERE me.user_id = tp.user_id
                  AND me.created_at >= tp.effective_sent_at
                  AND me.created_at < tp.effective_sent_at + INTERVAL '7 days'
                  AND me.event_type IN ('converted', 'clicked')
            ) THEN 1 ELSE 0 END AS converted_7d
        FROM tp
        WHERE NOT EXISTS (
            SELECT 1 FROM ml_outcome_labels mol
            WHERE mol.user_id = tp.user_id
              AND mol.label_json->>'touchpoint_id' = tp.touchpoint_id::text
        )
        LIMIT 5000
        """,
        str(int(lookback_days)),
    )
    inserted = 0
    for r in rows:
        try:
            from services.ml_marketing import record_outcome_label

            await record_outcome_label(
                conn,
                user_id=str(r["user_id"]),
                upload_id=None,
                variant_id=None,
                feature_snapshot={
                    "touchpoint_id": str(r["touchpoint_id"]),
                    "channel": str(r["channel"] or ""),
                },
                label_json={
                    "touchpoint_id": str(r["touchpoint_id"]),
                    "converted_7d": int(r["converted_7d"] or 0),
                    "source": "promo_label_backfill",
                },
            )
            inserted += 1
        except Exception as e:
            logger.debug("outcome label insert skipped: %s", e)
    return {"inserted": inserted, "candidates": len(rows)}
