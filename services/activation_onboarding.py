"""Activation checklist + CRM lifecycle helpers (no third-party APIs)."""

from __future__ import annotations

from typing import Any, Dict, Optional


_LIFECYCLE_ALLOWED = frozenset({"lead", "signed_up", "trial", "paid", "churned"})


def derive_signup_source(
    utm_source: Optional[str],
    utm_medium: Optional[str] = None,
    utm_campaign: Optional[str] = None,
) -> str:
    """Normalize CRM signup_source without overwriting raw utm_source."""
    if utm_source:
        return str(utm_source)[:64]
    if utm_medium or utm_campaign:
        return "campaign_unknown"
    return "direct"


def effective_lifecycle_stage(
    *,
    stored: Optional[str] = None,
    subscription_status: Optional[str] = None,
    subscription_tier: Optional[str] = None,
) -> str:
    """
    Prefer live billing signals; fall back to stored CRM stage.
    """
    status = (subscription_status or "").strip().lower()
    tier = (subscription_tier or "free").strip().lower()
    if status in ("canceled", "cancelled", "unpaid", "incomplete_expired"):
        return "churned"
    if status == "trialing":
        return "trial"
    if status in ("active", "past_due") and tier not in ("", "free"):
        return "paid"
    s = (stored or "signed_up").strip().lower()
    return s if s in _LIFECYCLE_ALLOWED else "signed_up"


async def fetch_activation_checklist(conn, user_id: str) -> Dict[str, Any]:
    """
    Connect → upload → schedule progress from existing tables.
    Dismissals live in users.preferences JSONB.
    """
    prefs_raw = await conn.fetchval("SELECT preferences FROM users WHERE id = $1::uuid", user_id)
    prefs: Dict[str, Any] = {}
    if isinstance(prefs_raw, dict):
        prefs = prefs_raw
    elif isinstance(prefs_raw, str):
        try:
            import json

            prefs = json.loads(prefs_raw) or {}
        except Exception:
            prefs = {}

    connected = bool(
        await conn.fetchval(
            """
            SELECT 1 FROM platform_tokens
            WHERE user_id = $1::uuid AND revoked_at IS NULL
            LIMIT 1
            """,
            user_id,
        )
    )
    uploaded = bool(
        await conn.fetchval(
            "SELECT 1 FROM uploads WHERE user_id = $1::uuid LIMIT 1",
            user_id,
        )
    )
    scheduled = bool(
        await conn.fetchval(
            """
            SELECT 1 FROM uploads
            WHERE user_id = $1::uuid
              AND (
                LOWER(COALESCE(schedule_mode, '')) IN ('scheduled', 'smart')
                OR scheduled_time IS NOT NULL
              )
            LIMIT 1
            """,
            user_id,
        )
    )

    dismissed = bool(prefs.get("activationChecklistDismissed") or prefs.get("activation_checklist_dismissed"))
    playbook_dismissed = bool(prefs.get("playbookModalDismissed") or prefs.get("playbook_modal_dismissed"))

    steps = [
        {
            "id": "connect",
            "label": "Connect your first platform",
            "href": "platforms.html",
            "done": connected,
        },
        {
            "id": "upload",
            "label": "Upload your first video",
            "href": "upload.html",
            "done": uploaded,
        },
        {
            "id": "schedule",
            "label": "Set a posting schedule (Scheduled or Smart)",
            "href": "upload.html",
            "done": scheduled,
        },
    ]
    done_count = sum(1 for s in steps if s["done"])
    complete = done_count >= len(steps)
    show_card = (not dismissed) and (not complete)

    return {
        "steps": steps,
        "done_count": done_count,
        "total": len(steps),
        "complete": complete,
        "dismissed": dismissed,
        "show_card": show_card,
        "show_playbook_modal": (not playbook_dismissed) and (not connected),
        "playbook_dismissed": playbook_dismissed,
    }
