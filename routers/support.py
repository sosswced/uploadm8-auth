"""
UploadM8 Support & Activity Log routes — extracted from app.py.
"""

from fastapi import APIRouter, Depends, Request

import core.state
from core.deps import get_current_user
from core.audit import log_system_event
from core.config import ADMIN_DISCORD_WEBHOOK_URL
from core.notifications import discord_notify
from core.models import SupportContactRequest, ActivityLogIn

router = APIRouter(tags=["support"])


@router.post("/api/support/contact")
async def support_contact(payload: SupportContactRequest, user: dict = Depends(get_current_user)):
    """Create a support ticket/message from the app."""
    async with core.state.db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO support_messages (user_id, name, email, subject, message)
            VALUES ($1, $2, $3, $4, $5)
            """,
            user["id"],
            (payload.name or user.get("name") or "").strip() or None,
            (payload.email or user.get("email") or "").strip() or None,
            payload.subject.strip(),
            payload.message.strip(),
        )

    # Optional admin notification
    if ADMIN_DISCORD_WEBHOOK_URL:
        await discord_notify(
            ADMIN_DISCORD_WEBHOOK_URL,
            embeds=[{
                "title": "\U0001f198 Support Message",
                "color": 0xf97316,
                "fields": [
                    {"name": "User", "value": f"{user.get('email','')} ({user.get('id','')})"},
                    {"name": "Subject", "value": payload.subject[:256]},
                    {"name": "Message", "value": (payload.message[:900] + "\u2026") if len(payload.message) > 900 else payload.message},
                ],
            }],
        )

    return {"status": "received"}


@router.post("/api/activity/log")
async def log_activity(data: ActivityLogIn, request: Request, user: dict = Depends(get_current_user)):
    """
    Frontend button-click and UI action audit trail.
    Called by JavaScript whenever a significant user action occurs.
    Stored in system_event_log with full context.
    """
    allowed_categories = {"UI_ACTION", "UPLOAD", "PLATFORM", "AUTH", "NAVIGATION"}
    category = data.event_category if data.event_category in allowed_categories else "UI_ACTION"

    # Sanitize — prevent log injection
    action_safe = str(data.action or "")[:100].strip()
    if not action_safe:
        return {"ok": False, "error": "action required"}

    await log_system_event(
        user_id=str(user["id"]),
        action=action_safe,
        event_category=category,
        resource_type=data.resource_type,
        resource_id=data.resource_id,
        details={**(data.details or {}), "session_id": data.session_id},
        request=request,
    )
    return {"ok": True}
