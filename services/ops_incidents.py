"""
Operational incidents: durable log + admin email + Discord for failures and bug reports.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from core.config import ADMIN_OPS_EMAIL

logger = logging.getLogger("uploadm8.ops_incidents")

ERROR_DISCORD_WEBHOOK_URL = os.environ.get("ERROR_DISCORD_WEBHOOK_URL", "")
ADMIN_DISCORD_WEBHOOK_URL = os.environ.get("ADMIN_DISCORD_WEBHOOK_URL", "")


def _discord_url() -> str:
    return (ERROR_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL or "").strip()


async def _send_discord_embed(title: str, description: str, color: int = 0xEF4444) -> None:
    wh = _discord_url()
    if not wh:
        return
    embed = {
        "title": title[:256],
        "color": color,
        "description": description[:1800],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(wh, json={"embeds": [embed]})
    except Exception as e:
        logger.warning("ops_incidents discord: %s", e)


async def _send_ops_email(subject: str, html_body: str) -> None:
    if not ADMIN_OPS_EMAIL:
        return
    try:
        from stages.emails.base import send_email, MAIL_FROM

        await send_email(ADMIN_OPS_EMAIL, subject[:200], html_body, from_addr=MAIL_FROM)
    except Exception as e:
        logger.warning("ops_incidents email: %s", e)


def _parse_uuid(val: Any) -> Optional[uuid.UUID]:
    if val is None or val == "":
        return None
    try:
        return val if isinstance(val, uuid.UUID) else uuid.UUID(str(val))
    except (ValueError, TypeError):
        return None


async def record_operational_incident(
    pool,
    *,
    source: str,
    incident_type: str,
    subject: str,
    body: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    user_id: Any = None,
    upload_id: Any = None,
    screenshot_r2_key: Optional[str] = None,
    send_alerts: bool = True,
    alert_email: bool = True,
    alert_discord: bool = True,
) -> Optional[str]:
    """
    Insert operational_incidents row; optionally email ops + Discord.
    Returns incident id string or None if DB write failed.
    """
    uid = _parse_uuid(user_id)
    upid = _parse_uuid(upload_id)
    det = details if isinstance(details, dict) else {}
    src = (source or "unknown")[:50]
    itype = (incident_type or "unknown")[:120]
    subj = (subject or itype)[:2000]
    bod = (body or "")[:8000] if body else None
    shot = (screenshot_r2_key or "")[:512] or None

    new_id: Optional[uuid.UUID] = None
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO operational_incidents
                    (source, incident_type, user_id, upload_id, subject, body, details, screenshot_r2_key)
                VALUES ($1, $2, $3::uuid, $4::uuid, $5, $6, $7::jsonb, $8)
                RETURNING id
                """,
                src,
                itype,
                uid,
                upid,
                subj,
                bod,
                json.dumps(det, default=str),
                shot,
            )
            if row:
                new_id = row["id"]
    except Exception as e:
        logger.error("record_operational_incident DB failed: %s", e)
        return None

    if not send_alerts or not new_id:
        return str(new_id) if new_id else None

    # Mark alert channels (best-effort)
    email_ok = False
    discord_ok = False
    if alert_email and ADMIN_OPS_EMAIL:
        try:
            safe = json.dumps(det, indent=2, default=str)[:4000]
            html = (
                f"<h2>{subj}</h2><pre style='white-space:pre-wrap;font-size:12px'>{safe}</pre>"
                f"<p>Incident ID: {new_id}</p>"
            )
            await _send_ops_email(f"[UploadM8] {itype}", html)
            email_ok = True
        except Exception:
            pass
    if alert_discord and _discord_url():
        try:
            await _send_discord_embed(
                f"Ops: {itype}",
                f"**{subj}**\n```json\n{json.dumps(det, indent=2, default=str)[:1200]}\n```\n`{new_id}`",
            )
            discord_ok = True
        except Exception:
            pass

    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE operational_incidents
                   SET email_sent_at = CASE WHEN $2 THEN NOW() ELSE email_sent_at END,
                       discord_sent_at = CASE WHEN $3 THEN NOW() ELSE discord_sent_at END
                 WHERE id = $1::uuid
                """,
                new_id,
                email_ok,
                discord_ok,
            )
    except Exception:
        pass

    return str(new_id)
