"""
Marketing compliance: consent, suppressions, send rate limits, and LLM payload hygiene.
"""

from __future__ import annotations

import copy
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

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
    row = await conn.fetchrow(
        "SELECT email_marketing FROM user_marketing_consent WHERE user_id = $1::uuid",
        user_id,
    )
    if row is None:
        return True
    return bool(row["email_marketing"])


async def user_discord_marketing_allowed(conn: asyncpg.Connection, user_id: str) -> bool:
    row = await conn.fetchrow(
        "SELECT discord_marketing FROM user_marketing_consent WHERE user_id = $1::uuid",
        user_id,
    )
    if row is None:
        return False
    return bool(row["discord_marketing"])


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
