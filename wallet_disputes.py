"""Wallet ledger dispute tickets: create, list, admin resolve, user notifications."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

from core.config import ADMIN_DISCORD_WEBHOOK_URL
from core.notifications import discord_notify
from services.ops_incidents import record_operational_incident
from stages.emails.base import MAIL_FROM_SUPPORT, SUPPORT_EMAIL, send_email

logger = logging.getLogger("uploadm8.wallet_disputes")


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _user_discord_webhook(conn: Any, user_id: str) -> str:
    row = await conn.fetchrow(
        """
        SELECT COALESCE(
            NULLIF(TRIM(us.discord_webhook), ''),
            NULLIF(TRIM(up.discord_webhook), ''),
            NULLIF(TRIM(COALESCE(u.preferences->>'discordWebhook', u.preferences->>'discord_webhook')), '')
        ) AS wh
        FROM users u
        LEFT JOIN user_settings us ON us.user_id = u.id
        LEFT JOIN user_preferences up ON up.user_id = u.id
        WHERE u.id = $1::uuid
        """,
        user_id,
    )
    if not row:
        return ""
    return str(row["wh"] or "").strip()


async def create_wallet_dispute(
    pool: Any,
    *,
    user_id: str,
    ledger_id: str,
    note: str,
) -> Dict[str, Any]:
    """Validate ledger row belongs to user; insert dispute; ops incident + alerts."""
    try:
        lid = uuid.UUID(str(ledger_id).strip())
    except (ValueError, TypeError) as e:
        raise ValueError("Invalid ledger_id") from e

    note_clean = (note or "").strip()
    if len(note_clean) < 8:
        raise ValueError("Note must be at least 8 characters")

    async with pool.acquire() as conn:
        led = await conn.fetchrow(
            """
            SELECT id, user_id, token_type, delta, reason, upload_id, created_at
            FROM token_ledger WHERE id = $1::uuid AND user_id = $2::uuid
            """,
            lid,
            user_id,
        )
        if not led:
            raise ValueError("Ledger line not found for your account")

        dup = await conn.fetchval(
            """
            SELECT id FROM wallet_disputes
            WHERE ledger_id = $1::uuid AND status IN ('open', 'in_review')
            LIMIT 1
            """,
            lid,
        )
        if dup:
            raise ValueError("A ticket is already open for this ledger line")

        try:
            row = await conn.fetchrow(
                """
                INSERT INTO wallet_disputes (ledger_id, user_id, status, note)
                VALUES ($1::uuid, $2::uuid, 'open', $3)
                RETURNING id, ledger_id, user_id, status, note, created_at
                """,
                lid,
                user_id,
                note_clean,
            )
        except asyncpg.exceptions.UniqueViolationError as e:
            raise ValueError("A ticket is already open for this ledger line") from e
        if not row:
            raise RuntimeError("Could not create dispute")

        dispute_id = row["id"]
        urow = await conn.fetchrow("SELECT email, name FROM users WHERE id = $1::uuid", user_id)
        email = (urow["email"] if urow else "") or ""

    details = {
        "wallet_dispute_id": str(dispute_id),
        "ledger_id": str(lid),
        "token_type": led["token_type"],
        "delta": int(led["delta"] or 0),
        "reason": led["reason"],
        "upload_id": str(led["upload_id"]) if led["upload_id"] else None,
        "user_email": email,
        "note_preview": note_clean[:500],
    }
    inc_id = await record_operational_incident(
        pool,
        source="api",
        incident_type="wallet_dispute",
        subject=f"Wallet dispute from {email} (ledger {str(lid)[:8]}…)",
        body=note_clean[:4000],
        details=details,
        user_id=user_id,
        upload_id=str(led["upload_id"]) if led.get("upload_id") else None,
        send_alerts=True,
        alert_email=True,
        alert_discord=True,
        dedupe_key=f"wallet_dispute:{dispute_id}",
        dedupe_seconds=60,
    )

    if inc_id:
        try:
            inc_uuid = uuid.UUID(str(inc_id))
        except (ValueError, TypeError):
            inc_uuid = None
        if inc_uuid:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE wallet_disputes
                    SET operational_incident_id = $2::uuid, updated_at = NOW()
                    WHERE id = $1::uuid
                    """,
                    dispute_id,
                    inc_uuid,
                )

    out = dict(row)
    out["id"] = str(out["id"])
    out["ledger_id"] = str(out["ledger_id"])
    out["user_id"] = str(out["user_id"])
    out["created_at"] = out["created_at"].isoformat() if out.get("created_at") else None
    return out


async def list_user_wallet_disputes(pool: Any, user_id: str, *, limit: int = 40) -> List[Dict[str, Any]]:
    lim = max(1, min(int(limit), 100))
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.id, d.ledger_id, d.status, d.note, d.resolution_message,
                   d.user_email_sent_at, d.user_discord_sent_at,
                   d.created_at, d.updated_at, d.resolved_at,
                   l.delta AS ledger_delta, l.reason AS ledger_reason, l.token_type AS ledger_token_type
            FROM wallet_disputes d
            JOIN token_ledger l ON l.id = d.ledger_id
            WHERE d.user_id = $1::uuid
            ORDER BY d.created_at DESC
            LIMIT $2
            """,
            user_id,
            lim,
        )
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        for k in ("id", "ledger_id"):
            if d.get(k):
                d[k] = str(d[k])
        for k in ("created_at", "updated_at", "resolved_at", "user_email_sent_at", "user_discord_sent_at"):
            v = d.get(k)
            if hasattr(v, "isoformat"):
                d[k] = v.isoformat()
        out.append(d)
    return out


async def list_admin_wallet_disputes(
    pool: Any,
    *,
    limit: int,
    offset: int,
    status: Optional[str],
) -> Dict[str, Any]:
    lim = max(1, min(int(limit), 200))
    off = max(0, int(offset))
    params: List[Any] = []
    where_sql = ""
    if status and status in ("open", "in_review", "resolved", "rejected"):
        params.append(status)
        where_sql = f"WHERE d.status = ${len(params)}"

    params.append(lim)
    lim_i = len(params)
    params.append(off)
    off_i = len(params)

    async with pool.acquire() as conn:
        total = await conn.fetchval(f"SELECT COUNT(*)::int FROM wallet_disputes d {where_sql}", *params[:-2])
        rows = await conn.fetch(
            f"""
            SELECT d.id, d.ledger_id, d.user_id, d.status, d.note, d.admin_internal_note,
                   d.resolution_message, d.created_at, d.updated_at, d.resolved_at,
                   d.user_email_sent_at, d.user_discord_sent_at,
                   l.delta AS ledger_delta, l.reason AS ledger_reason, l.token_type AS ledger_token_type,
                   l.upload_id AS ledger_upload_id,
                   u.email AS user_email, u.name AS user_name
            FROM wallet_disputes d
            JOIN token_ledger l ON l.id = d.ledger_id
            JOIN users u ON u.id = d.user_id
            {where_sql}
            ORDER BY d.created_at DESC
            LIMIT ${lim_i} OFFSET ${off_i}
            """,
            *params,
        )

    items: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        for k in ("id", "ledger_id", "user_id", "ledger_upload_id"):
            if d.get(k):
                d[k] = str(d[k])
        for k in ("created_at", "updated_at", "resolved_at", "user_email_sent_at", "user_discord_sent_at"):
            v = d.get(k)
            if hasattr(v, "isoformat"):
                d[k] = v.isoformat()
        items.append(d)

    return {"items": items, "total": int(total or 0), "limit": lim, "offset": off}


async def _notify_user_dispute_resolved(
    pool: Any,
    *,
    dispute_id: uuid.UUID,
    user_id: str,
    user_email: str,
    user_name: Optional[str],
    status: str,
    resolution_message: str,
) -> None:
    """Email + optional user Discord webhook; best-effort."""
    subj = (
        "Your UploadM8 wallet ticket was resolved"
        if status == "resolved"
        else "Update on your UploadM8 wallet ticket"
    )
    name = (user_name or "there").strip() or "there"
    safe_msg = resolution_message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    body_html = f"""
    <p>Hi {name},</p>
    <p>We reviewed your wallet / token ledger dispute. <strong>Status: {status}</strong></p>
    <p style="white-space:pre-wrap;border-left:3px solid #f97316;padding-left:12px;margin:16px 0;">{safe_msg}</p>
    <p>If anything still looks wrong, reply to this email or write to <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a>.</p>
    <p style="color:#888;font-size:12px;">Ticket ID: {dispute_id}</p>
    """
    try:
        await send_email(
            user_email,
            subj,
            body_html,
            from_addr=MAIL_FROM_SUPPORT,
            reply_to=SUPPORT_EMAIL,
        )
    except Exception as e:
        logger.warning("wallet_dispute user email failed: %s", e)

    wh = ""
    async with pool.acquire() as conn:
        wh = await _user_discord_webhook(conn, user_id)

    discord_ok = False
    if wh:
        color = 0x22C55E if status == "resolved" else 0xF97316
        discord_ok = await discord_notify(
            wh,
            embeds=[
                {
                    "title": f"Wallet ticket {status}",
                    "color": color,
                    "description": (resolution_message or "")[:1800],
                    "footer": {"text": f"Ticket {dispute_id}"},
                }
            ],
        )
        if not discord_ok:
            logger.warning("wallet_dispute user discord failed dispute=%s", dispute_id)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE wallet_disputes SET
                user_email_sent_at = NOW(),
                user_discord_sent_at = CASE WHEN $3::boolean THEN NOW() ELSE user_discord_sent_at END,
                updated_at = NOW()
            WHERE id = $1::uuid AND user_id = $2::uuid
            """,
            dispute_id,
            user_id,
            bool(discord_ok and wh),
        )


async def _notify_admin_dispute_closed(
    *,
    dispute_id: str,
    user_email: str,
    status: str,
    resolution_message: str,
) -> None:
    url = (ADMIN_DISCORD_WEBHOOK_URL or "").strip()
    if not url:
        return
    try:
        await discord_notify(
            url,
            embeds=[
                {
                    "title": f"Wallet ticket closed ({status})",
                    "color": 0x6366F1,
                    "description": f"**User:** {user_email}\n**Ticket:** `{dispute_id}`\n{(resolution_message or '')[:1200]}",
                }
            ],
        )
    except Exception as e:
        logger.warning("wallet_dispute admin discord: %s", e)


async def admin_patch_wallet_dispute(
    pool: Any,
    *,
    dispute_id: str,
    status: Optional[str],
    admin_internal_note: Optional[str],
    resolution_message: Optional[str],
) -> Dict[str, Any]:
    try:
        did = uuid.UUID(str(dispute_id).strip())
    except (ValueError, TypeError) as e:
        raise ValueError("Invalid dispute id") from e

    async with pool.acquire() as conn:
        cur = await conn.fetchrow(
            """
            SELECT d.*, u.email AS user_email, u.name AS user_name
            FROM wallet_disputes d
            JOIN users u ON u.id = d.user_id
            WHERE d.id = $1::uuid
            """,
            did,
        )
        if not cur:
            raise LookupError("Dispute not found")

        old_status = str(cur["status"] or "")
        new_status = status if status is not None else old_status
        if status is not None and status not in ("open", "in_review", "resolved", "rejected"):
            raise ValueError("Invalid status")

        if resolution_message is not None:
            new_res = resolution_message.strip()
        else:
            crm = cur.get("resolution_message")
            new_res = (crm or "").strip() if crm is not None else ""

        if admin_internal_note is not None:
            new_int = admin_internal_note
        else:
            new_int = cur.get("admin_internal_note")

        terminal = new_status in ("resolved", "rejected")
        becoming_terminal = terminal and old_status not in ("resolved", "rejected")
        if becoming_terminal and not new_res:
            raise ValueError("resolution_message is required when setting status to resolved or rejected")

        resolved_at = cur["resolved_at"]
        if terminal and old_status not in ("resolved", "rejected"):
            resolved_at = _now()
        elif not terminal:
            resolved_at = None

        clear_user_notify = not terminal and old_status in ("resolved", "rejected")

        await conn.execute(
            """
            UPDATE wallet_disputes SET
                status = $2::varchar,
                admin_internal_note = $3,
                resolution_message = $4,
                resolved_at = $5,
                user_email_sent_at = CASE WHEN $6::boolean THEN NULL ELSE user_email_sent_at END,
                user_discord_sent_at = CASE WHEN $6::boolean THEN NULL ELSE user_discord_sent_at END,
                updated_at = NOW()
            WHERE id = $1::uuid
            """,
            did,
            new_status,
            new_int,
            new_res or None,
            resolved_at,
            clear_user_notify,
        )

    row = None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT d.*, u.email AS user_email, u.name AS user_name
            FROM wallet_disputes d
            JOIN users u ON u.id = d.user_id
            WHERE d.id = $1::uuid
            """,
            did,
        )

    if not row:
        raise LookupError("Dispute not found after update")

    msg_out = (row.get("resolution_message") or "").strip()
    if becoming_terminal and msg_out and cur.get("user_email_sent_at") is None:
        await _notify_user_dispute_resolved(
            pool,
            dispute_id=did,
            user_id=str(row["user_id"]),
            user_email=str(row["user_email"] or ""),
            user_name=row.get("user_name"),
            status=new_status,
            resolution_message=msg_out,
        )
        await _notify_admin_dispute_closed(
            dispute_id=str(did),
            user_email=str(row.get("user_email") or ""),
            status=new_status,
            resolution_message=msg_out,
        )
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT d.*, u.email AS user_email, u.name AS user_name
                FROM wallet_disputes d
                JOIN users u ON u.id = d.user_id
                WHERE d.id = $1::uuid
                """,
                did,
            )

    d = dict(row)
    for k in ("id", "ledger_id", "user_id", "operational_incident_id"):
        if d.get(k):
            d[k] = str(d[k])
    for k in ("created_at", "updated_at", "resolved_at", "user_email_sent_at", "user_discord_sent_at"):
        v = d.get(k)
        if hasattr(v, "isoformat"):
            d[k] = v.isoformat()
    return d
