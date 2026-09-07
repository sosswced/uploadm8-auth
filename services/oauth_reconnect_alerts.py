"""
Reconnect alerts — tell users before a connection stops publishing.

The keepalive renews access tokens indefinitely, but the refresh grant behind
them has a platform-imposed ceiling (see ``PLATFORM_REFRESH_TOKEN_LIFETIME``).
Once it lapses, only the user can restore publishing. With Smart Schedule
windows reaching years ahead, a silent lapse would fail every remaining post,
so this job warns ahead of expiry and again on confirmed death.

Idempotent in the same way as ``services/admin_email_jobs``: claim the row by
stamping ``oauth_reconnect_alert_at`` before sending, release it if Mailgun
rejects, and re-alert only after ``RECONNECT_ALERT_REPEAT_DAYS``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import asyncpg

from services.oauth_readiness import (
    STATE_EXPIRING,
    STATE_NEEDS_RECONNECT,
    assess_scheduled_risk,
    fetch_connections,
    fetch_pending_scheduled,
)
from stages.emails import send_platform_reconnect_email

logger = logging.getLogger("uploadm8-admin-jobs")

# Keep nudging a broken connection, without spamming, across a long unattended run.
RECONNECT_ALERT_REPEAT_DAYS = max(
    1, int(os.environ.get("OAUTH_RECONNECT_ALERT_REPEAT_DAYS") or 7)
)

PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube": "YouTube",
    "instagram": "Instagram",
    "facebook": "Facebook",
}


def _platform_label(platform: str) -> str:
    p = str(platform or "").lower()
    return PLATFORM_LABELS.get(p, p.title() or "your platform")


async def _claim_reconnect_alert(conn: asyncpg.Connection, token_row_id: str) -> bool:
    """Stamp the alert marker if it is unset or older than the repeat window."""
    try:
        claimed = await conn.fetchval(
            f"""
            UPDATE platform_tokens
               SET oauth_reconnect_alert_at = NOW()
             WHERE id = $1::uuid
               AND (
                     oauth_reconnect_alert_at IS NULL
                  OR oauth_reconnect_alert_at
                       < NOW() - INTERVAL '{RECONNECT_ALERT_REPEAT_DAYS} days'
                   )
            RETURNING id
            """,
            str(token_row_id),
        )
        return bool(claimed)
    except Exception as e:
        logger.warning("[oauth-reconnect] claim failed row=%s: %s", str(token_row_id)[:8], e)
        return False


async def _release_reconnect_alert(conn: asyncpg.Connection, token_row_id: str) -> None:
    try:
        await conn.execute(
            "UPDATE platform_tokens SET oauth_reconnect_alert_at = NULL WHERE id = $1::uuid",
            str(token_row_id),
        )
    except Exception:
        pass


async def _notifiable_recipients(
    conn: asyncpg.Connection, user_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Active users who accept email, keyed by user id."""
    if not user_ids:
        return {}
    rows = await conn.fetch(
        """
        SELECT u.id, u.email, u.name
          FROM users u
          LEFT JOIN user_preferences up ON up.user_id = u.id
         WHERE u.id = ANY($1::uuid[])
           AND u.status = 'active'
           AND u.email IS NOT NULL AND u.email <> ''
           AND COALESCE(up.email_notifications, TRUE) = TRUE
        """,
        user_ids,
    )
    return {str(r["id"]): {"email": r["email"], "name": r["name"] or "there"} for r in rows}


async def run_oauth_reconnect_alerts(
    pool: asyncpg.Pool, *, triggered_by: str = "manual"
) -> Dict[str, Any]:
    """Warn users about connections that are expiring or already dead."""
    from services.admin_email_jobs import _finish_run, _start_run

    job = "oauth_reconnect_alerts"
    run_id = await _start_run(pool, job, triggered_by)
    sent = skipped = errors = 0
    expiring_sent = dead_sent = 0

    try:
        async with pool.acquire() as conn:
            connections = await fetch_connections(conn)
            actionable = [
                c
                for c in connections
                if c.get("state") in (STATE_NEEDS_RECONNECT, STATE_EXPIRING)
            ]
            if not actionable:
                await _finish_run(
                    pool, run_id, sent, skipped, errors,
                    {"connections_checked": len(connections), "actionable": 0},
                )
                return {
                    "job": job,
                    "sent": 0,
                    "skipped": 0,
                    "errors": 0,
                    "details": {"connections_checked": len(connections), "actionable": 0},
                }

            # Only the affected users' scheduled work is needed for the counts.
            uploads = await fetch_pending_scheduled(conn)
            risk = assess_scheduled_risk(uploads, connections, detail_limit=0)
            per_account = risk.get("by_user_platform") or {}

            recipients = await _notifiable_recipients(
                conn, sorted({str(c.get("user_id")) for c in actionable if c.get("user_id")})
            )

            for c in actionable:
                uid = str(c.get("user_id") or "")
                rid = str(c.get("token_row_id") or "")
                who = recipients.get(uid)
                if not who or not rid:
                    skipped += 1
                    continue

                if not await _claim_reconnect_alert(conn, rid):
                    skipped += 1
                    continue

                is_dead = c.get("state") == STATE_NEEDS_RECONNECT
                try:
                    ok = await send_platform_reconnect_email(
                        email=who["email"],
                        name=who["name"],
                        platform_label=_platform_label(c.get("platform")),
                        account_label=str(c.get("account_name") or ""),
                        urgency="dead" if is_dead else "expiring",
                        expires_label=_expiry_label(c.get("refresh_expires_at")),
                        days_left=int(c.get("publishable_days") or 0),
                        scheduled_at_risk=int(
                            per_account.get(f"{uid}|{c.get('platform')}", 0)
                        ),
                    )
                except Exception as e:
                    logger.warning("[oauth-reconnect] send failed row=%s: %s", rid[:8], e)
                    await _release_reconnect_alert(conn, rid)
                    errors += 1
                    continue

                if not ok:
                    await _release_reconnect_alert(conn, rid)
                    skipped += 1
                    continue

                sent += 1
                if is_dead:
                    dead_sent += 1
                else:
                    expiring_sent += 1

        details = {
            "connections_checked": len(connections),
            "actionable": len(actionable),
            "dead_sent": dead_sent,
            "expiring_sent": expiring_sent,
            "slots_at_risk": risk.get("slots_at_risk", 0),
        }
        await _finish_run(pool, run_id, sent, skipped, errors, details)
        return {"job": job, "sent": sent, "skipped": skipped, "errors": errors, "details": details}

    except Exception as e:
        logger.exception("[oauth-reconnect] job failed: %s", e)
        try:
            await _finish_run(
                pool, run_id, sent, skipped, errors + 1, {}, error_message=str(e)
            )
        except Exception:
            pass
        return {"job": job, "sent": sent, "skipped": skipped, "errors": errors + 1, "error": str(e)}


def _expiry_label(raw: Optional[str]) -> str:
    """Human date for the email body, or empty when there is no known ceiling."""
    if not raw:
        return ""
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(str(raw))
    except ValueError:
        return ""
    return dt.strftime("%B %d, %Y")


__all__ = ["RECONNECT_ALERT_REPEAT_DAYS", "run_oauth_reconnect_alerts"]
