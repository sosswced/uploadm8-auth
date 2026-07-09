"""
Master-admin gate for rule-based (non-ML) touchpoint automation.

Path A (``marketing_touchpoint_runner``) is segment/rule based — not propensity ML.
Outbound email/Discord from that path requires an explicit master-admin enable
row in ``marketing_automation_gate`` in addition to ``MARKETING_AUTOMATION_ENABLED``.
In-app wallet nudges may still run when the env flag is on.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import asyncpg

logger = logging.getLogger("uploadm8.marketing_automation_gate")

GATE_KEY = "touchpoints_outbound_v1"


async def outbound_touchpoints_master_enabled(conn: asyncpg.Connection) -> bool:
    row = await conn.fetchrow(
        """
        SELECT enabled, enabled_by, enabled_at, notes
        FROM marketing_automation_gate
        WHERE gate_key = $1
        """,
        GATE_KEY,
    )
    return bool(row and row["enabled"])


async def get_outbound_touchpoints_gate(conn: asyncpg.Connection) -> Dict[str, Any]:
    row = await conn.fetchrow(
        """
        SELECT gate_key, enabled, enabled_by, enabled_at, notes, updated_at
        FROM marketing_automation_gate
        WHERE gate_key = $1
        """,
        GATE_KEY,
    )
    if not row:
        return {
            "gate_key": GATE_KEY,
            "enabled": False,
            "enabled_by": None,
            "enabled_at": None,
            "notes": None,
            "updated_at": None,
            "kind": "rule_based_lifecycle",
            "ml_propensity": False,
        }
    return {
        "gate_key": row["gate_key"],
        "enabled": bool(row["enabled"]),
        "enabled_by": str(row["enabled_by"]) if row.get("enabled_by") else None,
        "enabled_at": row["enabled_at"].isoformat() if row.get("enabled_at") else None,
        "notes": row.get("notes"),
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        "kind": "rule_based_lifecycle",
        "ml_propensity": False,
    }


async def set_outbound_touchpoints_gate(
    conn: asyncpg.Connection,
    *,
    enabled: bool,
    master_user_id: Optional[str],
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    await conn.execute(
        """
        INSERT INTO marketing_automation_gate (gate_key, enabled, enabled_by, enabled_at, notes, updated_at)
        VALUES ($1, $2, $3::uuid, CASE WHEN $2 THEN NOW() ELSE NULL END, $4, NOW())
        ON CONFLICT (gate_key) DO UPDATE SET
            enabled = EXCLUDED.enabled,
            enabled_by = EXCLUDED.enabled_by,
            enabled_at = CASE WHEN EXCLUDED.enabled THEN NOW() ELSE marketing_automation_gate.enabled_at END,
            notes = EXCLUDED.notes,
            updated_at = NOW()
        """,
        GATE_KEY,
        bool(enabled),
        master_user_id,
        (notes or None),
    )
    return await get_outbound_touchpoints_gate(conn)
