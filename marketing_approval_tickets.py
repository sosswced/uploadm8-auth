"""
Master-admin marketing approval tickets for outbound (email / Discord / mixed).

Outbound sends require an ``approved`` row in ``marketing_approval_tickets`` with
``resolved_by`` set (master), in addition to ``marketing_campaigns.approved_at``.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

import asyncpg

logger = logging.getLogger("uploadm8.marketing_approval_tickets")

OUTBOUND_CHANNELS = frozenset({"email", "discord", "mixed"})


def channel_requires_ticket(channel: Optional[str]) -> bool:
    return (channel or "").strip().lower() in OUTBOUND_CHANNELS


async def ensure_open_ticket_for_pending_campaign(
    conn: asyncpg.Connection,
    *,
    campaign_id: uuid.UUID,
    submitted_by: Optional[uuid.UUID],
    channel: str,
    status: str,
) -> None:
    """If campaign is outbound and pending approval, create an ``open`` ticket when none open."""
    if not channel_requires_ticket(channel):
        return
    if (status or "").strip().lower() != "pending_approval":
        return
    existing = await conn.fetchval(
        """
        SELECT 1 FROM marketing_approval_tickets
        WHERE campaign_id = $1
          AND status IN ('open', 'in_review')
        LIMIT 1
        """,
        campaign_id,
    )
    if existing:
        return
    await conn.execute(
        """
        INSERT INTO marketing_approval_tickets (campaign_id, status, submitted_by, resolution_notes)
        VALUES ($1, 'open', $2, NULL)
        """,
        campaign_id,
        submitted_by,
    )


async def outbound_ticket_master_approved(
    conn: asyncpg.Connection, campaign_id: uuid.UUID
) -> bool:
    row = await conn.fetchval(
        """
        SELECT 1 FROM marketing_approval_tickets
        WHERE campaign_id = $1
          AND status = 'approved'
          AND resolved_by IS NOT NULL
        LIMIT 1
        """,
        campaign_id,
    )
    return bool(row)


async def list_tickets(
    conn: asyncpg.Connection,
    *,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    q = """
        SELECT t.id, t.campaign_id, t.status, t.submitted_by, t.resolved_by, t.resolution_notes,
               t.created_at, t.resolved_at, t.copy_snapshot_hash,
               c.name AS campaign_name, c.channel AS campaign_channel, c.status AS campaign_status,
               c.objective AS campaign_objective
        FROM marketing_approval_tickets t
        JOIN marketing_campaigns c ON c.id = t.campaign_id
        WHERE ($1::text IS NULL OR t.status = $1)
        ORDER BY t.created_at DESC
        LIMIT $2 OFFSET $3
    """
    rows = await conn.fetch(q, status, max(1, min(limit, 200)), max(0, offset))
    return [dict(r) for r in rows]


async def _copy_snapshot_hash(conn: asyncpg.Connection, campaign_id: uuid.UUID) -> str:
    row = await conn.fetchrow(
        """
        SELECT COALESCE(template_subject,''), COALESCE(template_body_html,''),
               COALESCE(template_body_text,''), COALESCE(discord_message_text,'')
        FROM marketing_campaigns WHERE id = $1
        """,
        campaign_id,
    )
    if not row:
        return ""
    blob = "\n".join(str(x or "") for x in row.values())
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:64]


async def resolve_ticket_master(
    conn: asyncpg.Connection,
    *,
    ticket_id: uuid.UUID,
    master_user_id: uuid.UUID,
    decision: str,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Master admin only (caller must use require_master_admin).
    decision: ``approved`` | ``rejected``
    """
    dec = (decision or "").strip().lower()
    if dec not in ("approved", "rejected"):
        raise ValueError("decision must be approved or rejected")

    async with conn.transaction():
        t = await conn.fetchrow(
            """
            SELECT id, campaign_id, status
            FROM marketing_approval_tickets
            WHERE id = $1
            FOR UPDATE
            """,
            ticket_id,
        )
        if not t:
            raise LookupError("ticket_not_found")
        if t["status"] not in ("open", "in_review"):
            raise ValueError("ticket_not_open")

        cid: uuid.UUID = t["campaign_id"]
        camp = await conn.fetchrow(
            "SELECT id, channel, status FROM marketing_campaigns WHERE id = $1 FOR UPDATE",
            cid,
        )
        if not camp:
            raise LookupError("campaign_not_found")
        if not channel_requires_ticket(str(camp["channel"] or "")):
            raise ValueError("campaign_not_outbound")

        snap = await _copy_snapshot_hash(conn, cid)

        if dec == "approved":
            await conn.execute(
                """
                UPDATE marketing_approval_tickets
                SET status = 'approved',
                    resolved_by = $2,
                    resolved_at = NOW(),
                    resolution_notes = $3,
                    copy_snapshot_hash = $4
                WHERE id = $1
                """,
                ticket_id,
                master_user_id,
                (notes or "")[:8000] or None,
                snap or None,
            )
            await conn.execute(
                """
                UPDATE marketing_campaigns
                SET approved_at = NOW(),
                    approved_by = $2,
                    status = 'active',
                    updated_at = NOW()
                WHERE id = $1
                """,
                cid,
                master_user_id,
            )
        else:
            await conn.execute(
                """
                UPDATE marketing_approval_tickets
                SET status = 'rejected',
                    resolved_by = $2,
                    resolved_at = NOW(),
                    resolution_notes = $3,
                    copy_snapshot_hash = COALESCE(copy_snapshot_hash, $4)
                WHERE id = $1
                """,
                ticket_id,
                master_user_id,
                (notes or "")[:8000] or None,
                snap or None,
            )
            await conn.execute(
                """
                UPDATE marketing_campaigns
                SET approved_at = NULL,
                    approved_by = NULL,
                    status = CASE
                        WHEN status IN ('pending_approval', 'active') THEN 'draft'
                        ELSE status
                    END,
                    updated_at = NOW()
                WHERE id = $1
                """,
                cid,
            )

    return {"ok": True, "ticket_id": str(ticket_id), "campaign_id": str(cid), "decision": dec}


async def submit_ticket_for_campaign(
    conn: asyncpg.Connection,
    *,
    campaign_id: uuid.UUID,
    submitted_by: uuid.UUID,
) -> Dict[str, Any]:
    """Create an open ticket for outbound campaign (e.g. resubmit after rejection)."""
    camp = await conn.fetchrow(
        "SELECT id, channel, status FROM marketing_campaigns WHERE id = $1",
        campaign_id,
    )
    if not camp:
        raise LookupError("campaign_not_found")
    if not channel_requires_ticket(str(camp["channel"] or "")):
        raise ValueError("campaign_not_outbound")
    open_exists = await conn.fetchval(
        """
        SELECT 1 FROM marketing_approval_tickets
        WHERE campaign_id = $1 AND status IN ('open', 'in_review')
        LIMIT 1
        """,
        campaign_id,
    )
    if open_exists:
        raise ValueError("open_ticket_already_exists")
    tid = await conn.fetchval(
        """
        INSERT INTO marketing_approval_tickets (campaign_id, status, submitted_by)
        VALUES ($1, 'open', $2)
        RETURNING id
        """,
        campaign_id,
        submitted_by,
    )
    return {"ok": True, "ticket_id": str(tid), "campaign_id": str(campaign_id)}
