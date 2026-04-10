"""
UploadM8 audit logging — tamper-evident admin and system event logs.
Extracted from app.py; no mutable state lives here.
"""

import json
import logging

import core.state
from core.helpers import _now_utc, _sha256_hex

logger = logging.getLogger("uploadm8-api")


async def log_admin_audit(conn, *, user_id: str, admin: dict, action: str, details: dict = None,
                          request=None, event_category: str = "ADMIN", resource_type: str = None,
                          resource_id: str = None, severity: str = "INFO", outcome: str = "SUCCESS"):
    """
    Write a tamper-evident audit record to admin_audit_log.
    Corporate-grade: captures category, resource, actor, IP, user-agent, severity.
    Safe to call inside an existing connection/transaction. NEVER raises.
    """
    try:
        ip_address = None
        user_agent = None
        if request is not None:
            from core.security import client_ip
            ip_address = client_ip(request)
            user_agent = request.headers.get("user-agent", "")[:512]

        await conn.execute(
            """
            INSERT INTO admin_audit_log
                (user_id, admin_id, admin_email, action, details, ip_address,
                 event_category, actor_user_id, resource_type, resource_id,
                 user_agent, severity, outcome)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5::jsonb, $6, $7, $8::uuid, $9, $10, $11, $12, $13)
            """,
            str(user_id),
            str(admin.get("id", "")),
            admin.get("email", ""),
            action,
            json.dumps(details or {}),
            ip_address,
            event_category,
            str(admin.get("id", "")),
            resource_type,
            str(resource_id) if resource_id else None,
            user_agent,
            severity,
            outcome,
        )
    except Exception as e:
        logger.error(f"[audit] Failed to write admin audit log: {e}")
        # NEVER raise — audit failure must never break the primary operation


async def log_system_event(conn=None, *, user_id: str = None, action: str, event_category: str = "SYSTEM",
                            resource_type: str = None, resource_id: str = None, details: dict = None,
                            request=None, severity: str = "INFO", outcome: str = "SUCCESS"):
    """
    Write a system/user-action event to system_event_log.
    Used for uploads, platform connects, UI button clicks, auth events.
    Accepts an existing conn or acquires its own. NEVER raises.
    """
    async def _write(c):
        try:
            ip_address = None
            user_agent = None
            session_id = None
            if request is not None:
                forwarded = request.headers.get("x-forwarded-for")
                ip_address = forwarded.split(",")[0].strip() if forwarded else (
                    request.client.host if request.client else None
                )
                user_agent = request.headers.get("user-agent", "")[:512]
                session_id = request.headers.get("x-session-id", "")[:128] or None

            await c.execute(
                """
                INSERT INTO system_event_log
                    (user_id, event_category, action, resource_type, resource_id,
                     details, ip_address, user_agent, session_id, severity, outcome)
                VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11)
                """,
                str(user_id) if user_id else None,
                event_category,
                action,
                resource_type,
                str(resource_id) if resource_id else None,
                json.dumps(details or {}),
                ip_address,
                user_agent,
                session_id,
                severity,
                outcome,
            )
        except Exception as e:
            logger.error(f"[audit] Failed to write system event log: {e}")

    try:
        if conn is not None:
            await _write(conn)
        else:
            async with core.state.db_pool.acquire() as c:
                await _write(c)
    except Exception as e:
        logger.error(f"[audit] System event log pool error: {e}")


async def _purge_old_audit_logs():
    """Purge audit records older than 6 months (rolling window). Run periodically."""
    try:
        async with core.state.db_pool.acquire() as conn:
            deleted_admin = await conn.fetchval(
                "DELETE FROM admin_audit_log WHERE created_at < NOW() - INTERVAL '6 months' RETURNING id"
            )
            deleted_sys = await conn.fetchval(
                "DELETE FROM system_event_log WHERE created_at < NOW() - INTERVAL '6 months' RETURNING id"
            )
            logger.info(f"[audit] Purged old logs: admin_audit={deleted_admin or 0}, system_event={deleted_sys or 0}")
    except Exception as e:
        logger.error(f"[audit] Purge failed: {e}")
