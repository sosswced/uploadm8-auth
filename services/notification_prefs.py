"""Per-channel email notification preference helpers.

Settings → Notifications exposes independent toggles. Upload status uses
``email_notifications``; digests / scheduled alerts / security alerts each have
their own columns on ``user_preferences``.
"""

from __future__ import annotations

from typing import Any, Optional


async def user_pref_bool(
    conn: Any,
    user_id: Any,
    column: str,
    *,
    default: bool = True,
) -> bool:
    """Read a boolean column from ``user_preferences`` (default when missing)."""
    # Allowlist — never interpolate arbitrary column names into SQL.
    allowed = {
        "email_notifications",
        "auth_security_alerts",
        "digest_emails",
        "scheduled_alert_emails",
    }
    if column not in allowed:
        raise ValueError(f"unsupported notification pref column: {column}")
    try:
        row = await conn.fetchval(
            f"SELECT {column} FROM user_preferences WHERE user_id = $1",
            user_id,
        )
    except Exception:
        return default
    if row is None:
        return default
    return bool(row)


async def user_wants_security_alert_emails(conn: Any, user_id: Any) -> bool:
    return await user_pref_bool(conn, user_id, "auth_security_alerts", default=True)


async def user_wants_digest_emails(conn: Any, user_id: Any) -> bool:
    return await user_pref_bool(conn, user_id, "digest_emails", default=True)


async def user_wants_scheduled_alert_emails(conn: Any, user_id: Any) -> bool:
    return await user_pref_bool(conn, user_id, "scheduled_alert_emails", default=True)


async def maybe_queue_password_changed_email(
    background: Any,
    *,
    conn: Any,
    user_id: Any,
    email: Optional[str],
    name: Optional[str],
) -> bool:
    """
    Queue password-changed security email when the user opted in.

    Returns True when queued, False when skipped (opt-out / missing email).
    """
    if not email:
        return False
    if not await user_wants_security_alert_emails(conn, user_id):
        return False
    from stages.emails import send_password_changed_email

    background.add_task(send_password_changed_email, email, name or "there")
    return True
