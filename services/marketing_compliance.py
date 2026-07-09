"""
Marketing compliance: consent, suppressions, send rate limits, and LLM payload hygiene.
"""

from __future__ import annotations

import copy
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, Union

import asyncpg

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip() or default)
    except (TypeError, ValueError):
        return default


MARKETING_MAX_EMAILS_PER_HOUR = _env_int("MARKETING_MAX_EMAILS_PER_HOUR", 200)
MARKETING_MAX_DISCORD_PER_HOUR = _env_int("MARKETING_MAX_DISCORD_PER_HOUR", 60)


def sanitize_truth_bundle_for_llm(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip obvious PII and shrink large structures before sending to an external LLM.
    """
    raw = copy.deepcopy(metrics)

    def _scrub_obj(o: Any) -> Any:
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                lk = str(k).lower()
                if lk in ("email", "user_email", "accounts"):
                    if lk == "accounts" and isinstance(v, list):
                        out[k] = [
                            {kk: _scrub_obj(vv) for kk, vv in (item or {}).items() if kk != "email"}
                            for item in v[:80]
                        ]
                        continue
                    if lk != "accounts":
                        continue
                out[k] = _scrub_obj(v)
            return out
        if isinstance(o, list):
            return [_scrub_obj(x) for x in o[:200]]
        if isinstance(o, str):
            if EMAIL_RE.search(o):
                return "[redacted]"
            return o[:2000] if len(o) > 2000 else o
        return o

    scrubbed = _scrub_obj(raw)
    scrubbed["_pii_policy"] = "sanitized_for_llm"
    return scrubbed


async def user_email_marketing_allowed(conn: asyncpg.Connection, user_id: str) -> bool:
    """Opt-in: missing consent row means marketing email is NOT allowed."""
    row = await conn.fetchrow(
        "SELECT email_marketing FROM user_marketing_consent WHERE user_id = $1::uuid",
        user_id,
    )
    if row is None:
        return False
    return bool(row["email_marketing"])


async def user_discord_marketing_allowed(conn: asyncpg.Connection, user_id: str) -> bool:
    """Opt-in: missing consent row means Discord marketing is NOT allowed."""
    row = await conn.fetchrow(
        "SELECT discord_marketing FROM user_marketing_consent WHERE user_id = $1::uuid",
        user_id,
    )
    if row is None:
        return False
    return bool(row["discord_marketing"])


async def get_user_marketing_consent(conn: asyncpg.Connection, user_id: str) -> Dict[str, Any]:
    row = await conn.fetchrow(
        """
        SELECT email_marketing, discord_marketing, allow_pii_in_ml, updated_at
        FROM user_marketing_consent
        WHERE user_id = $1::uuid
        """,
        user_id,
    )
    if not row:
        return {
            "email_marketing": False,
            "discord_marketing": False,
            "allow_pii_in_ml": False,
            "updated_at": None,
            "has_row": False,
        }
    return {
        "email_marketing": bool(row["email_marketing"]),
        "discord_marketing": bool(row["discord_marketing"]),
        "allow_pii_in_ml": bool(row["allow_pii_in_ml"]),
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        "has_row": True,
    }


async def upsert_user_marketing_consent(
    conn: asyncpg.Connection,
    user_id: str,
    *,
    email_marketing: Optional[bool] = None,
    discord_marketing: Optional[bool] = None,
    allow_pii_in_ml: Optional[bool] = None,
) -> Dict[str, Any]:
    """Create/update consent. Omitted fields keep existing values (or False on insert)."""
    cur = await get_user_marketing_consent(conn, user_id)
    em = bool(cur["email_marketing"] if email_marketing is None else email_marketing)
    dm = bool(cur["discord_marketing"] if discord_marketing is None else discord_marketing)
    pii = bool(cur["allow_pii_in_ml"] if allow_pii_in_ml is None else allow_pii_in_ml)
    await conn.execute(
        """
        INSERT INTO user_marketing_consent (user_id, email_marketing, discord_marketing, allow_pii_in_ml, updated_at)
        VALUES ($1::uuid, $2, $3, $4, NOW())
        ON CONFLICT (user_id) DO UPDATE SET
            email_marketing = EXCLUDED.email_marketing,
            discord_marketing = EXCLUDED.discord_marketing,
            allow_pii_in_ml = EXCLUDED.allow_pii_in_ml,
            updated_at = NOW()
        """,
        user_id,
        em,
        dm,
        pii,
    )
    if not em:
        await conn.execute(
            """
            INSERT INTO marketing_suppressions (user_id, channel, reason)
            VALUES ($1::uuid, 'email', 'user_opt_out')
            ON CONFLICT (user_id, channel) DO UPDATE SET reason = EXCLUDED.reason
            """,
            user_id,
        )
    else:
        await conn.execute(
            """
            DELETE FROM marketing_suppressions
            WHERE user_id = $1::uuid AND channel = 'email' AND reason = 'user_opt_out'
            """,
            user_id,
        )
    return await get_user_marketing_consent(conn, user_id)


async def ensure_signup_marketing_consent(conn: asyncpg.Connection, user_id: str) -> Dict[str, Any]:
    """
    Signup auto-consent: insert email_marketing=True only when no consent row exists.

    Does not overwrite an existing opt-out (e.g. user unsubscribed before verify).
    """
    cur = await get_user_marketing_consent(conn, user_id)
    if cur.get("has_row"):
        return cur
    return await upsert_user_marketing_consent(
        conn,
        user_id,
        email_marketing=True,
        discord_marketing=False,
        allow_pii_in_ml=False,
    )


async def is_suppressed(conn: asyncpg.Connection, user_id: str, channel: str) -> bool:
    ch = (channel or "").strip().lower()[:32]
    n = await conn.fetchval(
        """
        SELECT COUNT(*)::int FROM marketing_suppressions
        WHERE user_id = $1::uuid AND (channel = $2 OR channel = 'all')
        """,
        user_id,
        ch,
    )
    return int(n or 0) > 0


async def channel_send_count_last_hour(conn: asyncpg.Connection, channel: str) -> int:
    return int(
        await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM marketing_touchpoint_deliveries
            WHERE channel = $1 AND status = 'sent' AND sent_at >= NOW() - INTERVAL '1 hour'
            """,
            (channel or "")[:32],
        )
        or 0
    )


async def assert_channel_rate_ok(conn: asyncpg.Connection, channel: str) -> Tuple[bool, str]:
    ch = (channel or "").strip().lower()
    if ch == "email":
        n = await channel_send_count_last_hour(conn, "email")
        if n >= MARKETING_MAX_EMAILS_PER_HOUR:
            return False, f"email hourly cap {MARKETING_MAX_EMAILS_PER_HOUR} reached ({n})"
    if ch == "discord":
        n = await channel_send_count_last_hour(conn, "discord")
        if n >= MARKETING_MAX_DISCORD_PER_HOUR:
            return False, f"discord hourly cap {MARKETING_MAX_DISCORD_PER_HOUR} reached ({n})"
    return True, ""


def render_template(template: str, ctx: Dict[str, Any]) -> str:
    if not template:
        return ""
    out = template
    for k, v in ctx.items():
        key = "{{" + str(k) + "}}"
        out = out.replace(key, str(v) if v is not None else "")
    return out


async def recent_delivery_exists(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    campaign_id: str,
    within_days: int = 7,
) -> bool:
    n = await conn.fetchval(
        """
        SELECT COUNT(*)::int FROM marketing_touchpoint_deliveries
        WHERE user_id = $1::uuid AND campaign_id = $2::uuid
          AND created_at >= NOW() - ($3::text || ' days')::interval
        """,
        user_id,
        campaign_id,
        str(max(1, within_days)),
    )
    return int(n or 0) > 0


async def claim_campaign_delivery_slot(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    campaign_id: str,
    channel: str,
    within_days: int = 7,
    delivery_id: Optional[Any] = None,
    subject: str = "",
    body_text: Optional[str] = None,
    body_html: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Atomically claim a campaign send slot for (user, campaign) within the dedupe window.

    Uses advisory lock + insert so concurrent execution ticks cannot double-send.
    Returns delivery id text on success, or None if already claimed recently.
    """
    import uuid as _uuid

    did = delivery_id or _uuid.uuid4()
    days = max(1, int(within_days))
    ch = (channel or "email").strip().lower()[:32]
    # Stable lock key from campaign+user (two int4 halves of UUIDs).
    try:
        c_u = _uuid.UUID(str(campaign_id))
        u_u = _uuid.UUID(str(user_id))
        k1 = c_u.int & 0x7FFFFFFF
        k2 = u_u.int & 0x7FFFFFFF
    except (ValueError, TypeError, AttributeError):
        k1, k2 = 0, 0

    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock($1, $2)", k1, k2)
        n = await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM marketing_touchpoint_deliveries
            WHERE user_id = $1::uuid AND campaign_id = $2::uuid
              AND channel = $3
              AND created_at >= NOW() - ($4::text || ' days')::interval
            """,
            user_id,
            campaign_id,
            ch,
            str(days),
        )
        if int(n or 0) > 0:
            return None
        await conn.execute(
            """
            INSERT INTO marketing_touchpoint_deliveries (
                id, user_id, channel, subject, body_text, body_html, status, scheduled_at, meta, campaign_id
            )
            VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, 'pending', NOW(), $7::jsonb, $8::uuid)
            """,
            did,
            user_id,
            ch,
            (subject or "")[:500],
            body_text,
            body_html,
            json.dumps(meta or {}, default=str),
            campaign_id,
        )
    return str(did)
